from pathlib import Path

import pytest

from training.data import load_dataset
from training.model import ModelNotTrainedError, TrainConfig, load_artifact, predict_text, train_model


@pytest.mark.parametrize("algorithm", ["logistic_regression", "naive_bayes"])
def test_training_and_prediction(tmp_path: Path, algorithm: str):
    output = tmp_path / f"{algorithm}.joblib"
    metrics = train_model(load_dataset("data/comentarios.csv"), TrainConfig(algorithm=algorithm), output)
    assert output.exists()
    assert 0 <= metrics["accuracy"] <= 1
    assert metrics["training_samples"] == 120
    result = predict_text("La explicación fue clara y aprendí mucho", load_artifact(output))
    assert result["clasificacion"] in {"positivo", "neutral", "negativo"}
    assert abs(sum(result["probabilidades"].values()) - 1) < 1e-6


def test_empty_prediction(tmp_path: Path):
    output = tmp_path / "model.joblib"
    train_model(load_dataset("data/comentarios.csv"), TrainConfig(), output)
    with pytest.raises(ValueError, match="vacío"):
        predict_text("   ", load_artifact(output))


def test_missing_model(tmp_path: Path):
    with pytest.raises(ModelNotTrainedError):
        load_artifact(tmp_path / "missing.joblib")

