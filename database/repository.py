"""Repositorio SQLite para el historial de análisis."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def initialize_database(path: str | Path) -> None:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                comentario TEXT NOT NULL,
                clasificacion TEXT NOT NULL,
                confianza REAL NOT NULL,
                probabilidades TEXT NOT NULL,
                fecha TEXT NOT NULL
            )
        """)


def save_prediction(path: str | Path, comment: str, result: dict[str, Any]) -> int:
    initialize_database(path)
    with sqlite3.connect(path) as connection:
        cursor = connection.execute(
            "INSERT INTO predictions (comentario, clasificacion, confianza, probabilidades, fecha) VALUES (?, ?, ?, ?, ?)",
            (comment, result["clasificacion"], result["confianza"], json.dumps(result["probabilidades"], ensure_ascii=False), result["fecha"]),
        )
        return int(cursor.lastrowid)


def get_history(path: str | Path, limit: int = 100) -> list[dict[str, Any]]:
    initialize_database(path)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT id, comentario, clasificacion, confianza, probabilidades, fecha FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [{**dict(row), "probabilidades": json.loads(row["probabilidades"])} for row in rows]

