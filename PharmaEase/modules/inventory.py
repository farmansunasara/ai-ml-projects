from functools import lru_cache

from modules.common import MODELS_DIR, load_model


def get_available_drugs():
    suffix = "_arima.pkl"
    return sorted(path.name[:-len(suffix)] for path in MODELS_DIR.glob(f"*{suffix}"))


@lru_cache(maxsize=None)
def _get_inventory_model(drug_name):
    return load_model(f"{drug_name}_arima.pkl")


def predict_inventory(drug_name, days):
    if drug_name not in get_available_drugs():
        raise ValueError(f"Unsupported drug: {drug_name}")
    model = _get_inventory_model(drug_name)
    forecast = model.forecast(steps=days)
    return forecast
