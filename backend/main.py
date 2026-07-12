"""Aplicación FastAPI de EduFeel IA."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import PredictionRequest, TrainRequest
from database.repository import get_history, initialize_database, save_prediction
from training.data import DatasetError, load_dataset, prepare_dataframe
from training.model import ModelNotTrainedError, TrainConfig, load_artifact, predict_text, train_model

ROOT = Path(__file__).resolve().parents[1]


def model_path() -> Path:
    return Path(os.getenv("EDUFEEL_MODEL_PATH", ROOT / "saved_models" / "sentiment_model.joblib"))


def database_path() -> Path:
    return Path(os.getenv("EDUFEEL_DB_PATH", ROOT / "database" / "edufeel.db"))


def default_dataset_path() -> Path:
    return Path(os.getenv("EDUFEEL_DATASET_PATH", ROOT / "data" / "comentarios.csv"))


def accumulated_dataset_path() -> Path:
    return Path(os.getenv("EDUFEEL_ACCUMULATED_DATASET_PATH", ROOT / "data" / "dataset_acumulado.csv"))


def active_dataset_path() -> Path:
    """Usa datos acumulados cuando existen; de lo contrario, el dataset base."""
    accumulated = accumulated_dataset_path()
    return accumulated if accumulated.exists() else default_dataset_path()


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database(database_path())
    yield


app = FastAPI(title="EduFeel IA API", version="1.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "EduFeel IA", "model_trained": model_path().exists()}


@app.post("/api/train")
def train(request: TrainRequest) -> dict:
    try:
        dataset = load_dataset(request.dataset_path or active_dataset_path())
        config = TrainConfig(**request.model_dump(exclude={"dataset_path"}))
        metrics = train_model(dataset, config, model_path())
        return {"message": "Modelo entrenado correctamente", "metrics": metrics}
    except (DatasetError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/train/upload")
async def train_upload(
    file: UploadFile = File(...),
    algorithm: str = Form("logistic_regression"),
    test_size: float = Form(0.2),
    max_features: int = Form(5000),
    ngram_min: int = Form(1),
    ngram_max: int = Form(2),
    max_iter: int = Form(1000),
    alpha: float = Form(1.0),
) -> dict:
    """Incorpora ejemplos etiquetados al dataset acumulado y reentrena."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="El archivo debe tener extensión .csv")
    try:
        contents = await file.read()
        if not contents:
            raise DatasetError("El archivo CSV está vacío")
        try:
            uploaded_frame = pd.read_csv(BytesIO(contents), encoding="utf-8")
        except (UnicodeError, pd.errors.ParserError) as exc:
            raise DatasetError(f"No se pudo leer el CSV: {exc}") from exc

        uploaded = prepare_dataframe(uploaded_frame, require_minimum=False)
        current = load_dataset(active_dataset_path())
        previous_comments = set(current["comentario"])
        combined = prepare_dataframe(pd.concat([current, uploaded], ignore_index=True))
        added_samples = len(set(combined["comentario"]) - previous_comments)

        destination = accumulated_dataset_path()
        destination.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(destination, index=False, encoding="utf-8")

        config = TrainConfig(
            algorithm=algorithm, test_size=test_size, max_features=max_features,
            ngram_min=ngram_min, ngram_max=ngram_max, max_iter=max_iter, alpha=alpha,
        )
        metrics = train_model(combined, config, model_path())
        return {
            "message": "Datos incorporados y modelo reentrenado correctamente",
            "metrics": metrics,
            "dataset": {
                "total_samples": int(len(combined)),
                "uploaded_samples": int(len(uploaded)),
                "added_samples": int(added_samples),
                "duplicates_ignored": int(len(uploaded) - added_samples),
                "class_distribution": {key: int(value) for key, value in combined["clasificacion"].value_counts().items()},
            },
        }
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
        return load_artifact(model_path())["metrics"]
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

