import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = "/Users/olgamorozova/Documents/Bybit Bot"
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")
OUT_PATH = os.path.join(OUT_DIR, "eth_feature_dataset.csv")


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Missing timestamp column in {path}")
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    df = df.sort_values("timestamp")
    return df


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    return (series - mean) / std


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load
    df_15m = _load_csv(os.path.join(RAW_DIR, "eth_15m.csv"))
    df_1h = _load_csv(os.path.join(RAW_DIR, "eth_1h.csv"))
    df_funding = _load_csv(os.path.join(RAW_DIR, "eth_funding.csv"))
    df_oi = _load_csv(os.path.join(RAW_DIR, "eth_oi.csv"))

    # Ensure unique timestamps
    df_15m = df_15m.drop_duplicates(subset=["timestamp"]).copy()
    df_1h = df_1h.drop_duplicates(subset=["timestamp"]).copy()
    df_funding = df_funding.drop_duplicates(subset=["timestamp"]).copy()
    df_oi = df_oi.drop_duplicates(subset=["timestamp"]).copy()

    # Merge 1H -> 15m (asof)
    df = pd.merge_asof(
        df_15m.sort_values("timestamp"),
        df_1h.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        suffixes=("", "_1h"),
    )

    # Funding -> 15m (asof)
    df = pd.merge_asof(
        df,
        df_funding.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )

    # OI -> direct merge by timestamp (15m aligned)
    df = df.merge(df_oi, on="timestamp", how="left", suffixes=("", "_oi"))

    # 15m indicators
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_6"] = df["close"].pct_change(6)
    df["atr_14"] = _atr(df["high"], df["low"], df["close"], window=14)
    df["rsi_14"] = _rsi(df["close"], window=14)
    bb_mid = df["close"].rolling(window=20).mean()
    bb_std = df["close"].rolling(window=20).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    df["bb_width"] = (bb_upper - bb_lower) / df["close"].replace(0, np.nan)
    ema50 = _ema(df["close"], span=50)
    df["ema50_slope"] = ema50.diff()

    # 1H indicators (use merged columns)
    if "close_1h" not in df.columns:
        raise ValueError("Missing 1H merged close_1h column")
    ema200_1h = _ema(df["close_1h"], span=200)
    df["ema200_1h"] = ema200_1h
    df["htf_trend"] = (df["close_1h"] > df["ema200_1h"]).astype(int)
    df["distance_to_ema200"] = (df["close_1h"] - df["ema200_1h"]) / df["ema200_1h"].replace(0, np.nan)

    # Funding features
    if "funding_rate" not in df.columns:
        raise ValueError("Missing funding_rate after merge")
    df["funding"] = df["funding_rate"]
    df["funding_zscore_30"] = _zscore(df["funding"], window=30)

    # OI features
    if "open_interest" not in df.columns:
        raise ValueError("Missing open_interest after merge")
    df["delta_oi"] = df["open_interest"].diff()
    df["delta_oi_pct"] = df["open_interest"].pct_change()
    df["oi_zscore_50"] = _zscore(df["open_interest"], window=50)
    df["delta_oi_over_return_1"] = df["delta_oi"] / df["return_1"].replace(0, np.nan)

    # Target
    future_max_high = df["high"].rolling(window=6).max().shift(-5)
    df["target_up"] = ((future_max_high - df["close"]) >= df["atr_14"]).astype(int)

    # Drop NaNs
    df_clean = df.dropna().copy()

    # Save
    df_clean.to_csv(OUT_PATH, index=False)
    logger.info(f"Saved dataset to {OUT_PATH}")

    # Summary
    feature_cols = [c for c in df_clean.columns if c not in ["timestamp"]]
    target_rate = df_clean["target_up"].mean() if not df_clean.empty else 0.0
    print("Summary:")
    print(f"rows: {len(df_clean)}")
    print(f"features: {len(feature_cols)}")
    print(f"target_up_rate: {target_rate:.4f}")


if __name__ == "__main__":
    main()
