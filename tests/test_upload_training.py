from pathlib import Path


def test_uploaded_data_is_accumulated_and_retrained(client, isolated_paths: Path, monkeypatch):
    accumulated = isolated_paths / "dataset_acumulado.csv"
    monkeypatch.setenv("EDUFEEL_ACCUMULATED_DATASET_PATH", str(accumulated))
    csv_data = (
        "comentario,clasificacion\n"
        "La tutoría personalizada fue fantástica,positivo\n"
        "Se registró la asistencia del grupo,neutral\n"
        "La plataforma falló durante toda la evaluación,negativo\n"
    ).encode("utf-8")

    first = client.post(
        "/api/train/upload",
        files={"file": ("nuevos.csv", csv_data, "text/csv")},
        data={"algorithm": "logistic_regression"},
    )
    assert first.status_code == 200
    assert first.json()["dataset"]["total_samples"] == 153
    assert first.json()["dataset"]["added_samples"] == 3
    assert accumulated.exists()

    second = client.post(
        "/api/train/upload",
        files={"file": ("nuevos.csv", csv_data, "text/csv")},
    )
    assert second.status_code == 200
    assert second.json()["dataset"]["total_samples"] == 153
    assert second.json()["dataset"]["added_samples"] == 0
    assert second.json()["dataset"]["duplicates_ignored"] == 3


def test_upload_rejects_invalid_categories(client, isolated_paths: Path, monkeypatch):
    monkeypatch.setenv("EDUFEEL_ACCUMULATED_DATASET_PATH", str(isolated_paths / "accumulated.csv"))
    response = client.post(
        "/api/train/upload",
        files={"file": ("invalid.csv", b"comentario,clasificacion\nEjemplo,excelente\n", "text/csv")},
    )
    assert response.status_code == 422
    assert "Categorías incorrectas" in response.json()["detail"]

