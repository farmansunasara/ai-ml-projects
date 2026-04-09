from functools import lru_cache
from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


@lru_cache(maxsize=None)
def load_model(filename: str):
    return joblib.load(MODELS_DIR / filename)

