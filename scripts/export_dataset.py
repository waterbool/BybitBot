from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backtest.market_data import symbol_slug
from config import settings
from data_fetch.bybit_client import (
    fetch_funding_rate_history,
    fetch_historical_klines,
    fetch_open_interest_history,
)


RAW_DIR = BASE_DIR / "data" / "raw"
DEFAULT_LOOKBACK_DAYS = 365

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _parse_symbols(raw: str | None) -> list[str]:
    if not raw:
        return [settings.BYBIT_SYMBOL]
    return [part.strip().upper() for part in raw.split(",") if part.strip()]


def _dedupe_sort(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")


def _gap_report(df: pd.DataFrame, interval_ms: int, label: str) -> tuple[bool, int]:
    if df.empty:
        logger.warning("[%s] empty dataset", label)
        return False, 0
    ts = df["timestamp"].astype("int64")
    diffs = ts.diff().dropna()
    gaps = diffs[diffs > interval_ms]
    if gaps.empty:
        logger.info("[%s] no gaps detected", label)
        return True, 0
    logger.warning("[%s] gaps detected: %s", label, len(gaps))
    for idx in gaps.index[:20]:
        prev_ts = int(ts.loc[idx - 1])
        curr_ts = int(ts.loc[idx])
        diff_min = (curr_ts - prev_ts) / 60000.0
        logger.warning("[%s] gap %s -> %s (%.2f min)", label, prev_ts, curr_ts, diff_min)
    return False, len(gaps)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    logger.info("Saved %s rows to %s", len(df), path)


def _summary_line(df: pd.DataFrame, label: str) -> str:
    if df.empty:
        return f"{label}: rows=0"
    return (
        f"{label}: rows={len(df)} "
        f"min_ts={int(df['timestamp'].min())} max_ts={int(df['timestamp'].max())}"
    )


def export_symbol_market_data(
    symbol: str,
    lookback_days: int,
    category: str,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = symbol_slug(symbol)
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (lookback_days * 24 * 60 * 60 * 1000)

    logger.info("Exporting market data for %s (%s days)", symbol, lookback_days)

    df_15m = fetch_historical_klines(symbol, "15", start_ms, now_ms, category=category)
    df_15m = _dedupe_sort(df_15m[["timestamp", "open", "high", "low", "close", "volume"]])
    _gap_report(df_15m, 15 * 60 * 1000, f"{symbol} 15m")
    _save_csv(df_15m, output_dir / f"{prefix}_15m.csv")

    df_1h = fetch_historical_klines(symbol, "60", start_ms, now_ms, category=category)
    df_1h = _dedupe_sort(df_1h[["timestamp", "open", "high", "low", "close", "volume"]])
    _gap_report(df_1h, 60 * 60 * 1000, f"{symbol} 1h")
    _save_csv(df_1h, output_dir / f"{prefix}_1h.csv")

    df_funding = fetch_funding_rate_history(symbol, start_ms, now_ms, category=category)
    df_funding = _dedupe_sort(df_funding[["timestamp", "funding_rate"]])
    _gap_report(df_funding, 8 * 60 * 60 * 1000, f"{symbol} funding")
    _save_csv(df_funding, output_dir / f"{prefix}_funding.csv")

    df_oi = fetch_open_interest_history(symbol, "15min", start_ms, now_ms, category=category)
    df_oi = _dedupe_sort(df_oi[["timestamp", "open_interest"]])
    _gap_report(df_oi, 15 * 60 * 1000, f"{symbol} oi")
    _save_csv(df_oi, output_dir / f"{prefix}_oi.csv")

    return {
        "symbol": symbol,
        "files": {
            "15m": f"{prefix}_15m.csv",
            "1h": f"{prefix}_1h.csv",
            "funding": f"{prefix}_funding.csv",
            "oi": f"{prefix}_oi.csv",
        },
        "summary": [
            _summary_line(df_15m, f"{prefix}_15m.csv"),
            _summary_line(df_1h, f"{prefix}_1h.csv"),
            _summary_line(df_funding, f"{prefix}_funding.csv"),
            _summary_line(df_oi, f"{prefix}_oi.csv"),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export raw OHLC/funding/OI datasets for one or more symbols.")
    parser.add_argument("--symbols", default=settings.BYBIT_SYMBOL, help="Comma-separated symbols, e.g. ETHUSDT,BTCUSDT,SOLUSDT")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS, help="How many days of history to export.")
    parser.add_argument("--category", default=settings.BYBIT_CATEGORY, help="Bybit market category, e.g. linear.")
    parser.add_argument("--output-dir", default=str(RAW_DIR), help="Destination folder for CSV files.")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    symbols = _parse_symbols(args.symbols)
    output_dir = Path(args.output_dir)

    all_summaries = []
    for symbol in symbols:
        all_summaries.append(
            export_symbol_market_data(
                symbol=symbol,
                lookback_days=args.lookback_days,
                category=args.category,
                output_dir=output_dir,
            )
        )

    print("\nSummary:")
    for item in all_summaries:
        print(f"\n{item['symbol']}:")
        for line in item["summary"]:
            print(line)


if __name__ == "__main__":
    main()
