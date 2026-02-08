import os
import logging
import joblib
import numpy as np
import pandas as pd

from config import settings
from ml.features import build_features

logger = logging.getLogger(__name__)
_MODEL_CACHE = None
_MODEL_PATH = None


def load_model(path: str):
    return joblib.load(path)


def predict_proba(model, features: np.ndarray):
    if features.ndim == 1:
        features = features.reshape(1, -1)
    # Use last row if multiple rows passed
    if features.shape[0] > 1:
        features = features[-1:, :]
    probs = model.predict_proba(features)[0]
    # classes: 0=DOWN, 1=FLAT, 2=UP
    p_down = float(probs[0])
    p_flat = float(probs[1])
    p_up = float(probs[2])
    return p_up, p_flat, p_down


def _get_cached_model(model_path: str | None = None):
    global _MODEL_CACHE, _MODEL_PATH
    path = model_path or getattr(settings, 'ML_MODEL_PATH', os.path.join(settings.BASE_DIR, 'models', 'ethusdt_5m_lgbm.pkl'))
    if _MODEL_CACHE is not None and _MODEL_PATH == path:
        return _MODEL_CACHE
    if not os.path.exists(path):
        logger.warning(f"ML model not found at {path}")
        return None
    try:
        _MODEL_CACHE = load_model(path)
        _MODEL_PATH = path
        return _MODEL_CACHE
    except Exception as e:
        logger.error(f"Failed to load ML model: {e}")
        return None


def add_ml_probabilities(df: pd.DataFrame, model_path: str | None = None) -> pd.DataFrame:
    """
    Add ML probability columns to dataframe: p_up, p_flat, p_down.
    Rows with insufficient features get NaN probabilities.
    """
    df = df.copy()
    model = _get_cached_model(model_path)
    if model is None:
        df['p_up'] = np.nan
        df['p_flat'] = np.nan
        df['p_down'] = np.nan
        return df

    features = build_features(df)
    if features.size == 0:
        df['p_up'] = np.nan
        df['p_flat'] = np.nan
        df['p_down'] = np.nan
        return df

    feature_df = pd.DataFrame(features, index=df.index)
    valid = ~feature_df.isna().any(axis=1)

    probs_up = pd.Series(index=df.index, dtype=float)
    probs_flat = pd.Series(index=df.index, dtype=float)
    probs_down = pd.Series(index=df.index, dtype=float)

    if valid.any():
        X = feature_df[valid].values
        probs = model.predict_proba(X)
        probs_down.loc[valid] = probs[:, 0]
        probs_flat.loc[valid] = probs[:, 1]
        probs_up.loc[valid] = probs[:, 2]

    df['p_up'] = probs_up
    df['p_flat'] = probs_flat
    df['p_down'] = probs_down
    return df
