def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_missing(client):
    assert client.get("/api/metrics").status_code == 503
    assert client.post("/api/predict", json={"comentario": "Una clase"}).status_code == 503


def test_empty_comment(client):
    assert client.post("/api/predict", json={"comentario": "   "}).status_code == 422


def test_train_predict_metrics_and_history(client):
    trained = client.post("/api/train", json={"algorithm": "logistic_regression"})
    assert trained.status_code == 200
    assert "accuracy" in trained.json()["metrics"]
    prediction = client.post("/api/predict", json={"comentario": "La docente explicó claramente y aprendí bastante"})
    assert prediction.status_code == 200
    assert prediction.json()["clasificacion"] in {"positivo", "neutral", "negativo"}
    assert client.get("/api/metrics").status_code == 200
    assert client.get("/api/model-info").status_code == 200
    history = client.get("/api/history").json()
    assert len(history) == 1


def test_invalid_csv(client, tmp_path):
    invalid = tmp_path / "invalid.csv"
    invalid.write_text("texto,etiqueta\nhola,bien\n", encoding="utf-8")
    response = client.post("/api/train", json={"dataset_path": str(invalid)})
    assert response.status_code == 422
    assert "Faltan columnas" in response.json()["detail"]

