"""Paths and environment switches. Everything is overridable by env var so the
same code runs locally, in tests and inside docker-compose."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("AUTOPILOT_ROOT", Path(__file__).resolve().parents[2]))


def _p(env: str, default: Path) -> Path:
    return Path(os.environ.get(env, default))


CONFIG_DIR = _p("AUTOPILOT_CONFIG_DIR", ROOT / "config")
MODELS_YAML = CONFIG_DIR / "models.yaml"
ROUTING_YAML = CONFIG_DIR / "routing.yaml"
DATA_DIR = _p("AUTOPILOT_DATA_DIR", ROOT / "data")
LABELED_PROMPTS = DATA_DIR / "labeled_prompts.jsonl"
VAR_DIR = _p("AUTOPILOT_VAR_DIR", ROOT / "var")          # runtime state (db, logs, model)
DB_PATH = _p("AUTOPILOT_DB", VAR_DIR / "autopilot.db")
LOG_PATH = _p("AUTOPILOT_LOG", VAR_DIR / "requests.jsonl")
CLASSIFIER_PATH = _p("AUTOPILOT_CLASSIFIER", VAR_DIR / "classifier.joblib")
CLASSIFIER_REPORT = VAR_DIR / "classifier_report.json"
RESULTS_DIR = _p("AUTOPILOT_RESULTS_DIR", ROOT / "results")

# "1" -> every provider is replaced by the deterministic offline simulator.
MOCK = os.environ.get("AUTOPILOT_MOCK", "0") == "1"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
REQUEST_TIMEOUT_S = float(os.environ.get("AUTOPILOT_TIMEOUT_S", "60"))


def ensure_dirs() -> None:
    for d in (VAR_DIR, RESULTS_DIR, DB_PATH.parent, LOG_PATH.parent, CLASSIFIER_PATH.parent):
        d.mkdir(parents=True, exist_ok=True)
