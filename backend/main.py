"""Aplicación FastAPI de EduFeel IA."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import PredictionRequest, TrainRequest
from database.repository import get_history, initialize_database, save_prediction
from training.data import DatasetError, load_dataset
from training.model import ModelNotTrainedError, TrainConfig, load_artifact, predict_text, train_model

ROOT = Path(__file__).resolve().parents[1]


def model_path() -> Path:
    return Path(os.getenv("EDUFEEL_MODEL_PATH", ROOT / "saved_models" / "sentiment_model.joblib"))


def database_path() -> Path:
    return Path(os.getenv("EDUFEEL_DB_PATH", ROOT / "database" / "edufeel.db"))


def default_dataset_path() -> Path:
    return Path(os.getenv("EDUFEEL_DATASET_PATH", ROOT / "data" / "comentarios.csv"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database(database_path())
    yield


app = FastAPI(title="EduFeel IA API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "EduFeel IA", "model_trained": model_path().exists()}


@app.post("/api/train")
def train(request: TrainRequest) -> dict:
    try:
        dataset = load_dataset(request.dataset_path or default_dataset_path())
        config = TrainConfig(**request.model_dump(exclude={"dataset_path"}))
        metrics = train_model(dataset, config, model_path())
        return {"message": "Modelo entrenado correctamente", "metrics": metrics}
    except (DatasetError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/predict")
def predict(request: PredictionRequest) -> dict:
    if not request.comentario.strip():
        raise HTTPException(status_code=422, detail="El comentario no puede estar vacío")
    try:
        result = predict_text(request.comentario, load_artifact(model_path()))
        prediction_id = save_prediction(database_path(), request.comentario.strip(), result)
        return {**result, "id": prediction_id}
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/metrics")
def metrics() -> dict:
    try:
        artifact = load_artifact(model_path())
        return artifact["metrics"]
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/history")
def history(limit: int = Query(default=100, ge=1, le=500)) -> list[dict]:
    return get_history(database_path(), limit)


@app.get("/api/model-info")
def model_info() -> dict:
    try:
        artifact = load_artifact(model_path())
        return {key: artifact[key] for key in ("config", "trained_at", "dataset_samples", "classes")}
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

