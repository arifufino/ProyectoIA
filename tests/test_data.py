import pandas as pd
import pytest

from training.data import DatasetError, clean_text, load_dataset, prepare_dataframe


def test_clean_text_preserves_spanish_and_no():
    assert clean_text("  ¡NO entendí la LECCIÓN!  ") == "¡no entendí la lección!"


def test_cleanup_removes_empty_and_duplicates():
    rows = []
    for label in ("positivo", "neutral", "negativo"):
        rows.extend({"comentario": f"Ejemplo {label} {i}", "clasificacion": label} for i in range(10))
    rows += [{"comentario": "", "clasificacion": "positivo"}, rows[0].copy()]
    assert len(prepare_dataframe(pd.DataFrame(rows))) == 30


def test_missing_columns():
    with pytest.raises(DatasetError, match="Faltan columnas"):
        prepare_dataframe(pd.DataFrame({"texto": ["hola"]}))


def test_invalid_category():
    frame = pd.DataFrame({"comentario": [f"texto {i}" for i in range(30)], "clasificacion": ["positivo"] * 10 + ["neutral"] * 10 + ["otro"] * 10})
    with pytest.raises(DatasetError, match="Categorías incorrectas"):
        prepare_dataframe(frame)


def test_official_dataset_is_balanced():
    data = load_dataset("data/comentarios.csv")
    assert data["clasificacion"].value_counts().to_dict() == {"positivo": 50, "neutral": 50, "negativo": 50}

