from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backtest.multi_asset import DEFAULT_STRATEGIES, normalize_symbols, run_multi_symbol_backtests
from config import settings


REPORTS_DIR = BASE_DIR / "reports"
SUMMARY_PATH = REPORTS_DIR / "multi_symbol_strategy_backtest_summary.json"
POSITIONS_PATH = REPORTS_DIR / "multi_symbol_position_candidates.csv"


def _parse_strategy_names(raw: str | None) -> list[str]:
    if not raw:
        return DEFAULT_STRATEGIES
    return [part.strip() for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strategy backtests across multiple symbols.")
    parser.add_argument("--symbols", default=settings.BYBIT_SYMBOL, help="Comma-separated symbols, e.g. ETHUSDT,BTCUSDT,SOLUSDT")
    parser.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES), help="Comma-separated strategy ids.")
    parser.add_argument("--lookback-days", type=int, default=90, help="Lookback window in days.")
    parser.add_argument("--initial-balance", type=float, default=settings.INITIAL_BALANCE, help="Initial balance per run.")
    parser.add_argument("--with-ml", action="store_true", help="Enable ML filter during backtests.")
    parser.add_argument("--summary-path", default=str(SUMMARY_PATH), help="Where to write JSON summary.")
    parser.add_argument("--positions-path", default=str(POSITIONS_PATH), help="Where to write position candidates CSV.")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.WARNING)

    symbols = normalize_symbols([part.strip() for part in args.symbols.split(",")])
    strategies = _parse_strategy_names(args.strategies)
    summary_path = Path(args.summary_path)
    positions_path = Path(args.positions_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    positions_path.parent.mkdir(parents=True, exist_ok=True)

    results, positions_df = run_multi_symbol_backtests(
        symbols=symbols,
        strategy_names=strategies,
        lookback_days=args.lookback_days,
        initial_balance=args.initial_balance,
        enable_ml=args.with_ml,
    )

    report = {
        "symbols": symbols,
        "strategies": strategies,
        "lookback_days": int(args.lookback_days),
        "with_ml": bool(args.with_ml),
        "results": results,
    }
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    positions_to_save = positions_df.copy()
    if not positions_to_save.empty and "score_components" in positions_to_save.columns:
        positions_to_save["score_components"] = positions_to_save["score_components"].apply(json.dumps)
    positions_to_save.to_csv(positions_path, index=False)

    print(json.dumps(report, indent=2))
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved position candidates to {positions_path}")


if __name__ == "__main__":
    main()
