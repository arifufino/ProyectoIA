"""Contratos HTTP de la API."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    comentario: str = Field(min_length=1, max_length=5000)


class TrainRequest(BaseModel):
    algorithm: str = "logistic_regression"
    test_size: float = 0.2
    max_features: int = 5000
    ngram_min: int = 1
    ngram_max: int = 2
    max_iter: int = 1000
    alpha: float = 1.0
    dataset_path: str | None = None

