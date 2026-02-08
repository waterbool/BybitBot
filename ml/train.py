import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from lightgbm import LGBMClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from data_fetch.bybit_client import fetch_historical_klines
import importlib.util
from ml.features import build_features

_INDICATORS_PATH = BASE_DIR / "indicators.py"
spec = importlib.util.spec_from_file_location("indicators_file", _INDICATORS_PATH)
indicators_file = importlib.util.module_from_spec(spec)
spec.loader.exec_module(indicators_file)
add_indicators = indicators_file.add_indicators


def load_data(symbol: str, interval: str, days: int) -> pd.DataFrame:
    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)
    return fetch_historical_klines(symbol, interval, start_time, end_time)


def make_labels(df: pd.DataFrame, horizon: int = 12) -> pd.Series:
    future_return = df['close'].shift(-horizon) / df['close'] - 1
    labels = pd.Series(index=df.index, dtype=int)
    labels[future_return > 0.003] = 2  # UP
    labels[future_return < -0.003] = 0  # DOWN
    labels[(future_return <= 0.003) & (future_return >= -0.003)] = 1  # FLAT
    return labels


def main():
    symbol = "ETHUSDT"
    interval = "5"
    days = 120
    horizon = 12

    df = load_data(symbol, interval, days)
    if df.empty:
        raise RuntimeError("No data fetched")

    df = add_indicators(df)
    X = build_features(df)
    y = make_labels(df, horizon=horizon)

    # Align and drop NaNs
    feature_df = pd.DataFrame(X, index=df.index)
    valid = ~feature_df.isna().any(axis=1)
    valid &= y.notna()

    feature_df = feature_df[valid]
    y = y[valid]

    # Time-based split (80/20)
    split_idx = int(len(feature_df) * 0.8)
    X_train = feature_df.iloc[:split_idx].values
    y_train = y.iloc[:split_idx].values
    X_test = feature_df.iloc[split_idx:].values
    y_test = y.iloc[split_idx:].values

    model = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "ethusdt_5m_lgbm.pkl")
    joblib.dump(model, model_path)

    # Simple evaluation
    if len(X_test) > 0:
        preds = model.predict(X_test)
        acc = (preds == y_test).mean()
        print(f"Saved model to {model_path}. Test accuracy: {acc:.4f}")
    else:
        print(f"Saved model to {model_path}. No test data.")


if __name__ == "__main__":
    main()
