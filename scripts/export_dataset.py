import os
import time
import logging
import pandas as pd

from data_fetch.bybit_client import (
    fetch_historical_klines,
    fetch_open_interest_history,
    fetch_funding_rate_history,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SYMBOL = "ETHUSDT"
CATEGORY = "linear"

OUTPUT_DIR = "/Users/olgamorozova/Documents/Bybit Bot/data/raw"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _dedupe_sort(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return df


def _gap_report(df: pd.DataFrame, interval_ms: int, name: str) -> tuple[bool, int]:
    if df.empty:
        logger.warning(f"[{name}] empty dataset")
        return False, 0
    ts = df["timestamp"].astype("int64")
    diffs = ts.diff().dropna()
    gaps = diffs[diffs > interval_ms]
    if gaps.empty:
        logger.info(f"[{name}] no gaps detected")
        return True, 0
    logger.warning(f"[{name}] gaps detected: {len(gaps)}")
    for idx in gaps.index[:20]:
        prev_ts = int(ts.loc[idx - 1])
        curr_ts = int(ts.loc[idx])
        diff_min = (curr_ts - prev_ts) / 60000.0
        logger.warning(f"[{name}] gap {prev_ts} -> {curr_ts} ({diff_min:.2f} min)")
    return False, len(gaps)


def _save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    logger.info(f"Saved {len(df)} rows to {path}")


def main():
    _ensure_dir(OUTPUT_DIR)

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (365 * 24 * 60 * 60 * 1000)  # 12 months

    # 15m OHLC
    df_15m = fetch_historical_klines(SYMBOL, "15", start_ms, now_ms, category=CATEGORY)
    df_15m = df_15m[["timestamp", "open", "high", "low", "close", "volume"]]
    df_15m = _dedupe_sort(df_15m)
    _gap_report(df_15m, 15 * 60 * 1000, "15m")
    _save_csv(df_15m, os.path.join(OUTPUT_DIR, "eth_15m.csv"))

    # 1h OHLC
    df_1h = fetch_historical_klines(SYMBOL, "60", start_ms, now_ms, category=CATEGORY)
    df_1h = df_1h[["timestamp", "open", "high", "low", "close", "volume"]]
    df_1h = _dedupe_sort(df_1h)
    _gap_report(df_1h, 60 * 60 * 1000, "1h")
    _save_csv(df_1h, os.path.join(OUTPUT_DIR, "eth_1h.csv"))

    # Funding rate history (typically 8h interval)
    df_funding = fetch_funding_rate_history(SYMBOL, start_ms, now_ms, category=CATEGORY)
    df_funding = df_funding[["timestamp", "funding_rate"]]
    df_funding = _dedupe_sort(df_funding)
    _gap_report(df_funding, 8 * 60 * 60 * 1000, "funding")
    _save_csv(df_funding, os.path.join(OUTPUT_DIR, "eth_funding.csv"))

    # Open interest history (15m interval)
    df_oi = fetch_open_interest_history(SYMBOL, "15min", start_ms, now_ms, category=CATEGORY)
    df_oi = df_oi[["timestamp", "open_interest"]]
    df_oi = _dedupe_sort(df_oi)
    _gap_report(df_oi, 15 * 60 * 1000, "oi")
    _save_csv(df_oi, os.path.join(OUTPUT_DIR, "eth_oi.csv"))

    # Summary
    def summary(df: pd.DataFrame, name: str):
        if df.empty:
            print(f"{name}: rows=0")
            return
        print(f"{name}: rows={len(df)} min_ts={int(df['timestamp'].min())} max_ts={int(df['timestamp'].max())}")

    print("\nSummary:")
    summary(df_15m, "eth_15m.csv")
    summary(df_1h, "eth_1h.csv")
    summary(df_funding, "eth_funding.csv")
    summary(df_oi, "eth_oi.csv")


if __name__ == "__main__":
    main()
