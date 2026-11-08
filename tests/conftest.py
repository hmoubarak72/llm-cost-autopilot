import os
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Every test gets its own config copy, db, logs and classifier, in mock mode."""
    cfg = tmp_path / "config"
    shutil.copytree(ROOT / "config", cfg)
    from cost_autopilot import settings
    monkeypatch.setattr(settings, "MOCK", True)
    monkeypatch.setattr(settings, "CONFIG_DIR", cfg)
    monkeypatch.setattr(settings, "MODELS_YAML", cfg / "models.yaml")
    monkeypatch.setattr(settings, "ROUTING_YAML", cfg / "routing.yaml")
    monkeypatch.setattr(settings, "VAR_DIR", tmp_path / "var")
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "var" / "test.db")
    monkeypatch.setattr(settings, "LOG_PATH", tmp_path / "var" / "requests.jsonl")
    monkeypatch.setattr(settings, "CLASSIFIER_PATH", tmp_path / "var" / "classifier.joblib")
    monkeypatch.setattr(settings, "CLASSIFIER_REPORT", tmp_path / "var" / "classifier_report.json")
    monkeypatch.setattr(settings, "RESULTS_DIR", tmp_path / "results")
    settings.ensure_dirs()
    import cost_autopilot.api as api
    monkeypatch.setattr(api, "_ap", None)
    yield tmp_path


@pytest.fixture
def trained(isolated):
    from cost_autopilot.classifier import train
    return train()
