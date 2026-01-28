import json
import os
from pathlib import Path
import threading
import joblib
import numpy as np

_model_lock = threading.Lock()
_model = None
_scaler = None
_feature_order = None


def _project_root():
    return Path(__file__).resolve().parent.parent


def load_model():
    global _model, _scaler, _feature_order
    with _model_lock:
        if _model is not None and _scaler is not None:
            return _model, _scaler, _feature_order
        
        project_root = _project_root()
        
        # Try multiple paths in order of preference:
        # 1. ml_models/ (project root) - standard location from train_model.py
        # 2. prediction/ml_models/ - app-specific location
        # 3. prediction/ - legacy app location
        
        model_paths = [
            project_root / 'ml_models' / 'model.joblib',
            project_root / 'ml_models' / 'fault_detection_model.pkl',
            project_root / 'prediction' / 'ml_models' / 'fault_detection_model.pkl',
        ]
        
        scaler_paths = [
            project_root / 'ml_models' / 'scaler.joblib',
            project_root / 'ml_models' / 'scaler.pkl',
            project_root / 'prediction' / 'ml_models' / 'scaler.pkl',
        ]
        
        feature_order_paths = [
            project_root / 'ml_models' / 'feature_order.json',
            project_root / 'prediction' / 'feature_order.json',
        ]
        
        # Load model
        for path in model_paths:
            if path.exists():
                try:
                    _model = joblib.load(path)
                    break
                except Exception as e:
                    print(f"Warning: Could not load model from {path}: {e}")
        
        # Load scaler
        for path in scaler_paths:
            if path.exists():
                try:
                    _scaler = joblib.load(path)
                    break
                except Exception as e:
                    print(f"Warning: Could not load scaler from {path}: {e}")
        
        # Load feature order
        for path in feature_order_paths:
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        _feature_order = json.load(f)
                        # Handle both dict and list formats
                        if isinstance(_feature_order, dict):
                            _feature_order = _feature_order.get('features', _feature_order)
                        break
                except Exception as e:
                    print(f"Warning: Could not load feature_order from {path}: {e}")

        return _model, _scaler, _feature_order


def predict_from_input(payload):
    """
    payload: either dict of feature_name->value OR list of ordered values under key 'values'
    returns: (label, confidence)
    """
    model, scaler, feature_order = load_model()
    if model is None:
        raise RuntimeError('Model not found. Run training to generate ml_models/fault_detection_model.pkl')

    # Accept dict->map to feature order, or list
    if isinstance(payload, dict):
        if feature_order is None:
            # we cannot map dict without feature order
            raise ValueError('feature_order.json missing; provide ordered list of values or create feature_order.json')
        x = [payload.get(f, 0.0) for f in feature_order]
    elif isinstance(payload, list) or isinstance(payload, tuple):
        x = list(payload)
    else:
        raise ValueError('Unsupported payload format')

    X = np.array(x, dtype=float).reshape(1, -1)
    if scaler is not None:
        X = scaler.transform(X)

    proba = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)
        # choose max probability
        confidence = float(proba.max())
    else:
        confidence = 1.0

    label = model.predict(X)[0]
    return label, confidence
