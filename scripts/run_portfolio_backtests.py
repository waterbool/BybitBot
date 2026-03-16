from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backtest.multi_asset import (
    DEFAULT_STRATEGIES,
    normalize_symbols,
    run_multi_symbol_backtests,
    select_top_candidates_by_day,
    summarize_portfolio_selection,
)
from config import settings


REPORTS_DIR = BASE_DIR / "reports"
SUMMARY_PATH = REPORTS_DIR / "portfolio_selection_summary.json"
SELECTED_PATH = REPORTS_DIR / "portfolio_selected_positions.csv"


def _parse_strategy_names(raw: str | None) -> list[str]:
    if not raw:
        return DEFAULT_STRATEGIES
    return [part.strip() for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the top-N scored backtest positions per day across symbols.")
    parser.add_argument("--symbols", default=settings.BYBIT_SYMBOL, help="Comma-separated symbols, e.g. ETHUSDT,BTCUSDT,SOLUSDT")
    parser.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES), help="Comma-separated strategy ids.")
    parser.add_argument("--lookback-days", type=int, default=90, help="Lookback window in days.")
    parser.add_argument("--initial-balance", type=float, default=settings.INITIAL_BALANCE, help="Starting balance for the portfolio summary.")
    parser.add_argument("--top-per-day", type=int, default=max(1, min(int(settings.MAX_TRADES_PER_DAY), 2)), help="Maximum selected entries per day.")
    parser.add_argument("--min-score", type=float, default=0.6, help="Minimum signal score required for selection.")
    parser.add_argument("--min-edge-score", type=float, default=0.55, help="Minimum historical edge score required for a symbol+strategy pair.")
    parser.add_argument("--signal-weight", type=float, default=0.4, help="Weight of raw signal score inside composite selection score.")
    parser.add_argument("--edge-weight", type=float, default=0.6, help="Weight of historical edge score inside composite selection score.")
    parser.add_argument("--max-per-symbol-per-day", type=int, default=1, help="Maximum selected positions per symbol per day.")
    parser.add_argument("--max-per-strategy-per-day", type=int, default=1, help="Maximum selected positions per strategy per day.")
    parser.add_argument("--with-ml", action="store_true", help="Enable ML filter during underlying strategy backtests.")
    parser.add_argument("--summary-path", default=str(SUMMARY_PATH), help="Where to write JSON summary.")
    parser.add_argument("--selected-path", default=str(SELECTED_PATH), help="Where to write selected positions CSV.")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.WARNING)

    symbols = normalize_symbols([part.strip() for part in args.symbols.split(",")])
    strategies = _parse_strategy_names(args.strategies)
    summary_path = Path(args.summary_path)
    selected_path = Path(args.selected_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.parent.mkdir(parents=True, exist_ok=True)

    results, positions_df = run_multi_symbol_backtests(
        symbols=symbols,
        strategy_names=strategies,
        lookback_days=args.lookback_days,
        initial_balance=args.initial_balance,
        enable_ml=args.with_ml,
    )
    selected_df = select_top_candidates_by_day(
        positions_df=positions_df,
        top_n_per_day=args.top_per_day,
        min_score=args.min_score,
        min_edge_score=args.min_edge_score,
        signal_weight=args.signal_weight,
        edge_weight=args.edge_weight,
        max_per_symbol_per_day=args.max_per_symbol_per_day,
        max_per_strategy_per_day=args.max_per_strategy_per_day,
    )
    portfolio_summary = summarize_portfolio_selection(
        positions_df=selected_df,
        initial_balance=args.initial_balance,
        top_n_per_day=args.top_per_day,
        min_score=args.min_score,
        min_edge_score=args.min_edge_score,
        source_candidate_count=len(positions_df),
    )
    report = {
        "symbols": symbols,
        "strategies": strategies,
        "lookback_days": int(args.lookback_days),
        "with_ml": bool(args.with_ml),
        "selection_config": {
            "min_score": float(args.min_score),
            "min_edge_score": float(args.min_edge_score),
            "signal_weight": float(args.signal_weight),
            "edge_weight": float(args.edge_weight),
            "max_per_symbol_per_day": int(args.max_per_symbol_per_day),
            "max_per_strategy_per_day": int(args.max_per_strategy_per_day),
            "top_per_day": int(args.top_per_day),
        },
        "portfolio_summary": portfolio_summary,
        "underlying_results": results,
    }
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    selected_to_save = selected_df.copy()
    if not selected_to_save.empty and "score_components" in selected_to_save.columns:
        selected_to_save["score_components"] = selected_to_save["score_components"].apply(json.dumps)
    selected_to_save.to_csv(selected_path, index=False)

    print(json.dumps(report, indent=2))
    print(f"\nSaved portfolio summary to {summary_path}")
    print(f"Saved selected positions to {selected_path}")


if __name__ == "__main__":
    main()
