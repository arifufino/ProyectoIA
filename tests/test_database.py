from database.repository import get_history, save_prediction


def test_sqlite_history(tmp_path):
    path = tmp_path / "test.db"
    result = {"clasificacion": "positivo", "confianza": 0.9, "probabilidades": {"positivo": 0.9, "neutral": 0.1}, "fecha": "2026-01-01T00:00:00+00:00"}
    prediction_id = save_prediction(path, "Buena clase", result)
    history = get_history(path)
    assert prediction_id == 1
    assert history[0]["comentario"] == "Buena clase"
    assert history[0]["probabilidades"]["positivo"] == 0.9

