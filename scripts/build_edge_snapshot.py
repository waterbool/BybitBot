from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backtest.multi_asset import DEFAULT_STRATEGIES, normalize_symbols
from config import settings
from live.edge_snapshot import build_edge_snapshot, save_edge_snapshot


def _parse_strategy_names(raw: str | None) -> list[str]:
    if not raw:
        return DEFAULT_STRATEGIES
    return [part.strip() for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a frozen historical edge snapshot for live selection.")
    parser.add_argument("--symbols", default=",".join(settings.LIVE_SELECTOR_SYMBOLS), help="Comma-separated symbols.")
    parser.add_argument("--strategies", default=",".join(settings.LIVE_SELECTOR_STRATEGIES), help="Comma-separated strategy ids.")
    parser.add_argument("--lookback-days", type=int, default=settings.LIVE_SELECTOR_EDGE_LOOKBACK_DAYS, help="Historical lookback window.")
    parser.add_argument("--with-ml", action="store_true", help="Enable ML during snapshot backtest build.")
    parser.add_argument("--output", default=settings.LIVE_SELECTOR_EDGE_SNAPSHOT_PATH, help="Output JSON path.")
    parser.add_argument("--verbose", action="store_true", help="Keep INFO logs during the snapshot build.")
    args = parser.parse_args()

    if not args.verbose:
        logging.getLogger().setLevel(logging.WARNING)

    snapshot = build_edge_snapshot(
        symbols=normalize_symbols([part.strip() for part in args.symbols.split(",")]),
        strategy_names=_parse_strategy_names(args.strategies),
        lookback_days=args.lookback_days,
        enable_ml=args.with_ml,
    )
    target = save_edge_snapshot(snapshot, args.output)
    print(f"Saved edge snapshot to {target}")


if __name__ == "__main__":
    main()
