"""Loads the trained pipeline, threshold, and SHAP explainer (mtime-cached)."""

import json
from pathlib import Path

import joblib

from src.explain import get_explainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.pkl"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "threshold.json"

_cache = {"mtime": None, "pipeline": None, "explainer": None, "threshold": 0.5}


def load():
    """Return (pipeline, explainer, threshold); reloads if the artifact changed."""
    if not MODEL_PATH.exists():
        return None, None, 0.5
    mtime = MODEL_PATH.stat().st_mtime
    if _cache["pipeline"] is None or mtime != _cache["mtime"]:
        _cache["pipeline"] = joblib.load(MODEL_PATH)
        _cache["explainer"] = get_explainer(_cache["pipeline"])
        _cache["mtime"] = mtime
        if THRESHOLD_PATH.exists():
            _cache["threshold"] = float(
                json.loads(THRESHOLD_PATH.read_text())["threshold"])
    return _cache["pipeline"], _cache["explainer"], _cache["threshold"]
