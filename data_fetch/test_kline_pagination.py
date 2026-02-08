import argparse
import time
from datetime import datetime, timezone

from bybit_client import fetch_klines_paginated, _interval_to_ms


def _ts_to_iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()


def run_case(symbol: str, interval: str, days: int):
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - days * 24 * 60 * 60 * 1000
    interval_ms = _interval_to_ms(interval)

    print(f"\nCase: {symbol} {interval} {days}d")
    df = fetch_klines_paginated(
        symbol=symbol,
        interval=interval,
        start_ts_ms=start_ms,
        end_ts_ms=now_ms,
        limit=1000,
        max_retries=5,
    )

    expected = None
    if interval_ms:
        expected = int(days * 24 * 60 * 60 * 1000 / interval_ms)

    print(f"Received candles: {len(df)}")
    if expected is not None:
        print(f"Expected ~{expected} candles")

    if df.empty:
        print("No data received")
        return

    first_ts = int(df.iloc[0]["timestamp"])
    last_ts = int(df.iloc[-1]["timestamp"])
    print(f"First: {_ts_to_iso(first_ts)} | Last: {_ts_to_iso(last_ts)}")

    if interval_ms:
        start_ok = abs(first_ts - start_ms) <= interval_ms
        end_ok = abs(last_ts - now_ms) <= interval_ms
        print(f"Start within 1 candle: {start_ok}")
        print(f"End within 1 candle: {end_ok}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Bybit kline pagination.")
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--interval", default="5")
    args = parser.parse_args()

    run_case(args.symbol, args.interval, 7)
    run_case(args.symbol, args.interval, 30)
