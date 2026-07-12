"""Entrenamiento, evaluación y predicción de sentimientos educativos."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from training.data import VALID_CATEGORIES, clean_text


class ModelNotTrainedError(RuntimeError):
    """Se intenta predecir sin un artefacto entrenado."""


@dataclass(slots=True)
class TrainConfig:
    algorithm: str = "logistic_regression"
    test_size: float = 0.2
    max_features: int = 5000
    ngram_min: int = 1
    ngram_max: int = 2
    max_iter: int = 1000
    alpha: float = 1.0
    random_state: int = 42

    def validate(self) -> None:
        if self.algorithm not in {"logistic_regression", "naive_bayes"}:
            raise ValueError("Algoritmo inválido")
        if not 0.1 <= self.test_size <= 0.5:
            raise ValueError("test_size debe estar entre 0.1 y 0.5")
        if self.max_features < 100:
            raise ValueError("max_features debe ser al menos 100")
        if not (1 <= self.ngram_min <= self.ngram_max <= 3):
            raise ValueError("ngram_range debe cumplir 1 <= mínimo <= máximo <= 3")
        if self.max_iter < 100 or self.alpha <= 0:
            raise ValueError("max_iter debe ser >= 100 y alpha > 0")


def train_model(data: pd.DataFrame, config: TrainConfig, output_path: str | Path) -> dict[str, Any]:
    """Entrena un pipeline reproducible, guarda el artefacto y devuelve métricas."""
    config.validate()
    x_train, x_test, y_train, y_test = train_test_split(
        data["comentario"], data["clasificacion"], test_size=config.test_size,
        random_state=config.random_state, stratify=data["clasificacion"],
    )
    classifier = (
        LogisticRegression(max_iter=config.max_iter, class_weight="balanced", random_state=config.random_state)
        if config.algorithm == "logistic_regression"
        else MultinomialNB(alpha=config.alpha)
    )
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=config.max_features, ngram_range=(config.ngram_min, config.ngram_max), sublinear_tf=True)),
        ("classifier", classifier),
    ])
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    labels = sorted(VALID_CATEGORIES)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average="weighted", zero_division=0)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "classification_report": classification_report(y_test, predictions, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "labels": labels,
        "training_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
    }
    trained_at = datetime.now(timezone.utc).isoformat()
    artifact = {
        "pipeline": pipeline,
        "metrics": metrics,
        "config": asdict(config),
        "trained_at": trained_at,
        "dataset_samples": int(len(data)),
        "classes": list(pipeline.classes_),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return {**metrics, "trained_at": trained_at, "algorithm": config.algorithm}


def load_artifact(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ModelNotTrainedError("El modelo aún no ha sido entrenado")
    try:
        return joblib.load(path)
    except Exception as exc:
        raise ModelNotTrainedError(f"No se pudo cargar el modelo: {exc}") from exc


def predict_text(comment: str, artifact: dict[str, Any]) -> dict[str, Any]:
    cleaned = clean_text(comment)
    if not cleaned:
        raise ValueError("El comentario no puede estar vacío")
    pipeline: Pipeline = artifact["pipeline"]
    probabilities = pipeline.predict_proba([cleaned])[0]
    classes = pipeline.classes_
    best = int(np.argmax(probabilities))
    return {
        "clasificacion": str(classes[best]),
        "confianza": float(probabilities[best]),
        "probabilidades": {str(label): float(probability) for label, probability in zip(classes, probabilities)},
        "fecha": datetime.now(timezone.utc).isoformat(),
    }

