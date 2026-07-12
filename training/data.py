"""Carga, limpieza y validación de comentarios estudiantiles."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"comentario", "clasificacion"}
VALID_CATEGORIES = {"positivo", "neutral", "negativo"}
MIN_SAMPLES_PER_CLASS = 10


class DatasetError(ValueError):
    """Error legible producido por un dataset inválido."""


def clean_text(value: object) -> str:
    """Normaliza texto conservando letras españolas, números y negaciones."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).lower().strip()
    text = re.sub(r"[^a-záéíóúüñ0-9¿?¡!.,;:()\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prepare_dataframe(frame: pd.DataFrame, require_minimum: bool = True) -> pd.DataFrame:
    """Valida el esquema y devuelve datos limpios y sin duplicados."""
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise DatasetError(f"Faltan columnas obligatorias: {', '.join(sorted(missing))}")

    data = frame.loc[:, ["comentario", "clasificacion"]].copy()
    data["comentario"] = data["comentario"].map(clean_text)
    data["clasificacion"] = data["clasificacion"].map(clean_text)
    data = data[(data["comentario"] != "") & (data["clasificacion"] != "")]
    data = data.drop_duplicates(subset=["comentario"], keep="first").reset_index(drop=True)

    invalid = set(data["clasificacion"]) - VALID_CATEGORIES
    if invalid:
        raise DatasetError(f"Categorías incorrectas: {', '.join(sorted(invalid))}")
    if data.empty:
        raise DatasetError("El dataset no contiene comentarios válidos")

    counts = data["clasificacion"].value_counts()
    if require_minimum:
        absent = VALID_CATEGORIES - set(counts.index)
        if absent:
            raise DatasetError(f"Faltan categorías: {', '.join(sorted(absent))}")
        too_small = {label: int(counts[label]) for label in VALID_CATEGORIES if counts[label] < MIN_SAMPLES_PER_CLASS}
        if too_small:
            details = ", ".join(f"{key}={value}" for key, value in sorted(too_small.items()))
            raise DatasetError(
                f"Dataset insuficiente: se requieren al menos {MIN_SAMPLES_PER_CLASS} registros por clase ({details})"
            )
    return data


def load_dataset(path: str | Path, require_minimum: bool = True) -> pd.DataFrame:
    """Lee un CSV UTF-8 y aplica todas las validaciones."""
    try:
        frame = pd.read_csv(path, encoding="utf-8")
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise DatasetError(f"No se pudo leer el CSV: {exc}") from exc
    return prepare_dataframe(frame, require_minimum=require_minimum)

