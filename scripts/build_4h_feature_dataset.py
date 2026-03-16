import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = "/Users/olgamorozova/Documents/Bybit Bot"
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")
OUT_PATH = os.path.join(OUT_DIR, "eth_4h_feature_dataset.csv")


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Missing timestamp column in {path}")
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
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

    df_4h = _load_csv(os.path.join(RAW_DIR, "eth_4h.csv"))
    df_1d = _load_csv(os.path.join(RAW_DIR, "eth_1d.csv"))
    df_funding = _load_csv(os.path.join(RAW_DIR, "eth_funding.csv"))
    df_oi = _load_csv(os.path.join(RAW_DIR, "eth_oi.csv"))

    # Merge 1D -> 4H (ffill)
    df = pd.merge_asof(
        df_4h.sort_values("timestamp"),
        df_1d.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        suffixes=("", "_1d"),
    )

    # Funding -> 4H (ffill)
    df = pd.merge_asof(
        df,
        df_funding.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )

    # OI -> direct merge by timestamp (4H aligned)
    df = df.merge(df_oi, on="timestamp", how="left", suffixes=("", "_oi"))

    # 4H features
    df["return_1"] = df["close"].pct_change(1)
    df["return_2"] = df["close"].pct_change(2)
    df["return_3"] = df["close"].pct_change(3)
    df["atr_14"] = _atr(df["high"], df["low"], df["close"], window=14)
    df["atr_pct"] = df["atr_14"] / df["close"].replace(0, np.nan)
    df["rsi_14"] = _rsi(df["close"], window=14)
    bb_mid = df["close"].rolling(window=20).mean()
    bb_std = df["close"].rolling(window=20).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    df["bb_width"] = (bb_upper - bb_lower) / df["close"].replace(0, np.nan)
    ema50 = _ema(df["close"], span=50)
    ema200 = _ema(df["close"], span=200)
    df["ema50_slope"] = ema50.diff()
    df["ema200_distance"] = (df["close"] - ema200) / ema200.replace(0, np.nan)

    # 1D features (from merged columns)
    if "close_1d" not in df.columns:
        raise ValueError("Missing close_1d after merge")
    ema200_1d = _ema(df["close_1d"], span=200)
    df["daily_trend"] = (df["close_1d"] > ema200_1d).astype(int)
    df["daily_return_1"] = df["close_1d"].pct_change(1)
    df["daily_rsi_14"] = _rsi(df["close_1d"], window=14)

    # Funding features
    if "funding_rate" not in df.columns:
        raise ValueError("Missing funding_rate after merge")
    df["funding_zscore_20"] = _zscore(df["funding_rate"], window=20)

    # OI features
    if "open_interest" not in df.columns:
        raise ValueError("Missing open_interest after merge")
    df["delta_oi"] = df["open_interest"].diff()
    df["delta_oi_pct"] = df["open_interest"].pct_change()
    df["oi_zscore_20"] = _zscore(df["open_interest"], window=20)

    # Target: next 3 bars max high
    future_max_3 = df["high"].shift(-1).rolling(window=3).max().shift(-2)
    df["target_up_4h"] = ((future_max_3 - df["close"]) >= (1.5 * df["atr_14"])).astype(int)

    # Drop NaNs
    df_clean = df.dropna().copy()

    # Save
    df_clean.to_csv(OUT_PATH, index=False)
    logger.info(f"Saved dataset to {OUT_PATH}")

    # Summary
    feature_cols = [c for c in df_clean.columns if c not in ["timestamp", "target_up_4h"]]
    target_rate = df_clean["target_up_4h"].mean() if not df_clean.empty else 0.0
    print("Summary:")
    print(f"rows: {len(df_clean)}")
    print(f"features: {len(feature_cols)}")
    print(f"target_up_4h rate: {target_rate:.4f}")


if __name__ == "__main__":
    main()
