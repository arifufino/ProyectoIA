from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def isolated_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EDUFEEL_MODEL_PATH", str(tmp_path / "model.joblib"))
    monkeypatch.setenv("EDUFEEL_DB_PATH", str(tmp_path / "history.db"))
    return tmp_path


@pytest.fixture()
def client(isolated_paths):
    from backend.main import app
    with TestClient(app) as test_client:
        yield test_client

