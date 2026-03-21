from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backtest.backtester import run_backtest
from backtest.market_data import build_funding_frame, build_mtf_frame, load_base_15m
from backtest.multi_asset import aggregate_positions, select_top_candidates_by_day, summarize_portfolio_selection
from backtest.scoring import compute_signal_score
from config import settings
from strategy.rules import (
    apply_funding_extreme_reversal_strategy,
    apply_mean_reversion_strategy,
    apply_mtf_trend_pullback_strategy,
    apply_strategy,
    apply_volatility_compression_breakout_strategy,
)


LOGGER = logging.getLogger("robust_search")


@dataclass(frozen=True)
class SearchStrategy:
    name: str
    strategy_fn: Any
    frame_builder: Any


STRATEGIES: dict[str, SearchStrategy] = {
    "ema_crossover_baseline": SearchStrategy(
        name="ema_crossover_baseline",
        strategy_fn=apply_strategy,
        frame_builder=load_base_15m,
    ),
    "mean_reversion": SearchStrategy(
        name="mean_reversion",
        strategy_fn=apply_mean_reversion_strategy,
        frame_builder=load_base_15m,
    ),
    "volatility_compression": SearchStrategy(
        name="volatility_compression",
        strategy_fn=apply_volatility_compression_breakout_strategy,
        frame_builder=load_base_15m,
    ),
    "mtf_trend_pullback": SearchStrategy(
        name="mtf_trend_pullback",
        strategy_fn=apply_mtf_trend_pullback_strategy,
        frame_builder=build_mtf_frame,
    ),
    "funding_extreme_reversal": SearchStrategy(
        name="funding_extreme_reversal",
        strategy_fn=apply_funding_extreme_reversal_strategy,
        frame_builder=build_funding_frame,
    ),
}


COMMON_SEARCH_SPACE: dict[str, list[Any]] = {
    "RISK_REWARD_RATIO": [0.5, 0.75, 1.0, 1.25, 1.5],
    "SL_ATR_MULT": [0.8, 1.0, 1.2, 1.5, 2.0],
    "PARTIAL_TP_ATR_MULT": [0.25, 0.5, 0.75, 1.0],
    "PARTIAL_TP_FRACTION": [0.25, 0.4, 0.5, 0.6, 0.75],
    "TRAIL_ACTIVATE_ATR": [0.3, 0.5, 0.8, 1.0, 1.3],
    "TRAIL_ATR_MULT": [0.8, 1.0, 1.2, 1.5, 1.8],
    "TIME_STOP_CANDLES": [4, 8, 12, 16, 24, 32],
    "MIN_SCORE": [0.35, 0.45, 0.55, 0.65, 0.75],
    "TOP_PER_DAY": [1, 2],
}

STRATEGY_SPECIFIC_SPACE: dict[str, dict[str, list[Any]]] = {
    "ema_crossover_baseline": {
        "EMA_FAST": [3, 5, 7, 9, 12, 15],
        "EMA_SLOW": [18, 21, 26, 34, 55, 89],
        "MIN_ATR_THRESHOLD": [0.0005, 0.0008, 0.001, 0.0015, 0.002],
        "IMPULSE_THRESHOLD": [0.0003, 0.0005, 0.0008, 0.001, 0.0015, 0.002],
        "COOLDOWN_CANDLES": [0, 1, 2, 4, 6, 8],
    },
    "mean_reversion": {
        "MEAN_REV_EMA_PERIOD": [20, 34, 50, 89],
        "MEAN_REV_RSI_PERIOD": [7, 14, 21],
        "MEAN_REV_RSI_LONG": [20, 25, 30, 35, 40],
        "MEAN_REV_RSI_SHORT": [60, 65, 70, 75, 80],
        "MEAN_REV_BB_PERIOD": [14, 20, 30],
        "MEAN_REV_BB_STD": [1.5, 1.8, 2.0, 2.2, 2.5],
        "MEAN_REV_ATR_PERIOD": [10, 14, 20],
        "MEAN_REV_MIN_ATR_PCT": [0.0005, 0.001, 0.0015, 0.002, 0.003],
    },
    "volatility_compression": {
        "VOL_COMP_EMA_PERIOD": [20, 34, 50, 89],
        "VOL_COMP_ATR_PERIOD": [10, 14, 20],
        "VOL_COMP_ATR_MA_PERIOD": [20, 30, 50, 80],
        "VOL_COMP_BB_PERIOD": [14, 20, 30],
        "VOL_COMP_BB_STD": [1.5, 2.0, 2.5],
        "VOL_COMP_BB_WIDTH_MULT": [1.02, 1.05, 1.10, 1.15, 1.20],
    },
    "mtf_trend_pullback": {
        "MTF_PULLBACK_EMA_PERIOD": [20, 34, 50],
        "MTF_PULLBACK_RSI_PERIOD": [7, 14, 21],
        "MTF_PULLBACK_RSI_LONG": [35, 40, 45, 50],
        "MTF_PULLBACK_RSI_SHORT": [50, 55, 60, 65],
    },
    "funding_extreme_reversal": {
        "FUNDING_EXTREME_LONG_THRESHOLD": [-0.00003, -0.00005, -0.00007, -0.00010, -0.00015],
        "FUNDING_EXTREME_SHORT_THRESHOLD": [0.00003, 0.00005, 0.00007, 0.00010, 0.00015],
        "FUNDING_EXTREME_MIN_OI_CHANGE": [0.0, 0.001, 0.003, 0.005],
        "FUNDING_EXTREME_MIN_PRICE_MOVE": [0.0, 0.0003, 0.0005, 0.001],
    },
}


def _window_schedule(index: pd.Index, count: int, window_days: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(index.min())
    end = pd.Timestamp(index.max())
    window = pd.Timedelta(days=window_days)
    first_end = start + window
    if first_end >= end:
        return [(start, end)]
    if count <= 1:
        return [(end - window, end)]
    spacing = (end - first_end) / (count - 1)
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for idx in range(count):
        window_end = first_end + (spacing * idx)
        window_start = window_end - window
        windows.append((window_start, window_end))
    return windows


def _profit_factor_numeric(value: Any) -> float:
    if value == "inf":
        return 99.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _default_trial(strategy_name: str) -> dict[str, Any]:
    base = {
        "RISK_REWARD_RATIO": float(settings.RISK_REWARD_RATIO),
        "SL_ATR_MULT": float(settings.SL_ATR_MULT),
        "PARTIAL_TP_ATR_MULT": float(settings.PARTIAL_TP_ATR_MULT),
        "PARTIAL_TP_FRACTION": float(settings.PARTIAL_TP_FRACTION),
        "TRAIL_ACTIVATE_ATR": float(settings.TRAIL_ACTIVATE_ATR),
        "TRAIL_ATR_MULT": float(settings.TRAIL_ATR_MULT),
        "TIME_STOP_CANDLES": int(settings.TIME_STOP_CANDLES),
        "MIN_SCORE": 0.55,
        "TOP_PER_DAY": 2,
    }
    if strategy_name == "ema_crossover_baseline":
        base.update(
            {
                "EMA_FAST": int(settings.EMA_FAST),
                "EMA_SLOW": int(settings.EMA_SLOW),
                "MIN_ATR_THRESHOLD": float(settings.MIN_ATR_THRESHOLD),
                "IMPULSE_THRESHOLD": float(settings.IMPULSE_THRESHOLD),
                "COOLDOWN_CANDLES": int(settings.COOLDOWN_CANDLES),
            }
        )
    elif strategy_name == "mean_reversion":
        base.update(
            {
                "MEAN_REV_EMA_PERIOD": 50,
                "MEAN_REV_RSI_PERIOD": 14,
                "MEAN_REV_RSI_LONG": 30.0,
                "MEAN_REV_RSI_SHORT": 70.0,
                "MEAN_REV_BB_PERIOD": 20,
                "MEAN_REV_BB_STD": 2.0,
                "MEAN_REV_ATR_PERIOD": 14,
                "MEAN_REV_MIN_ATR_PCT": 0.0015,
            }
        )
    elif strategy_name == "volatility_compression":
        base.update(
            {
                "VOL_COMP_EMA_PERIOD": 50,
                "VOL_COMP_ATR_PERIOD": 14,
                "VOL_COMP_ATR_MA_PERIOD": 50,
                "VOL_COMP_BB_PERIOD": 20,
                "VOL_COMP_BB_STD": 2.0,
                "VOL_COMP_BB_WIDTH_MULT": 1.10,
            }
        )
    elif strategy_name == "mtf_trend_pullback":
        base.update(
            {
                "MTF_PULLBACK_EMA_PERIOD": 50,
                "MTF_PULLBACK_RSI_PERIOD": 14,
                "MTF_PULLBACK_RSI_LONG": 45.0,
                "MTF_PULLBACK_RSI_SHORT": 55.0,
            }
        )
    elif strategy_name == "funding_extreme_reversal":
        base.update(
            {
                "FUNDING_EXTREME_LONG_THRESHOLD": -0.00007,
                "FUNDING_EXTREME_SHORT_THRESHOLD": 0.00007,
                "FUNDING_EXTREME_MIN_OI_CHANGE": 0.0,
                "FUNDING_EXTREME_MIN_PRICE_MOVE": 0.0,
            }
        )
    return base


def _random_trial(strategy_name: str, rng: random.Random) -> dict[str, Any]:
    trial = {key: rng.choice(values) for key, values in COMMON_SEARCH_SPACE.items()}
    specific = STRATEGY_SPECIFIC_SPACE[strategy_name]
    for key, values in specific.items():
        trial[key] = rng.choice(values)
    if strategy_name == "ema_crossover_baseline":
        fast = int(trial["EMA_FAST"])
        slow_choices = [value for value in STRATEGY_SPECIFIC_SPACE[strategy_name]["EMA_SLOW"] if int(value) > fast + 4]
        if slow_choices:
            trial["EMA_SLOW"] = rng.choice(slow_choices)
    if strategy_name == "mean_reversion":
        if float(trial["MEAN_REV_RSI_SHORT"]) <= float(trial["MEAN_REV_RSI_LONG"]):
            trial["MEAN_REV_RSI_SHORT"] = float(trial["MEAN_REV_RSI_LONG"]) + 30.0
    if strategy_name == "mtf_trend_pullback":
        if float(trial["MTF_PULLBACK_RSI_SHORT"]) <= float(trial["MTF_PULLBACK_RSI_LONG"]):
            trial["MTF_PULLBACK_RSI_SHORT"] = float(trial["MTF_PULLBACK_RSI_LONG"]) + 15.0
    return trial


@contextmanager
def _temporary_settings(overrides: dict[str, Any]):
    sentinel = object()
    previous: dict[str, Any] = {}
    for key, value in overrides.items():
        previous[key] = getattr(settings, key, sentinel)
        setattr(settings, key, value)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is sentinel:
                delattr(settings, key)
            else:
                setattr(settings, key, value)


def _indicator_overrides(strategy_name: str, params: dict[str, Any]) -> dict[str, Any]:
    if strategy_name == "ema_crossover_baseline":
        return {
            "ema_fast": int(params["EMA_FAST"]),
            "ema_slow": int(params["EMA_SLOW"]),
            "atr_period": int(getattr(settings, "ATR_PERIOD", 20)),
        }
    if strategy_name == "mean_reversion":
        return {
            "rsi_period": int(params["MEAN_REV_RSI_PERIOD"]),
            "atr_period": int(params["MEAN_REV_ATR_PERIOD"]),
            "bb_period": int(params["MEAN_REV_BB_PERIOD"]),
            "bb_std": float(params["MEAN_REV_BB_STD"]),
        }
    if strategy_name == "volatility_compression":
        return {
            "atr_period": int(params["VOL_COMP_ATR_PERIOD"]),
            "bb_period": int(params["VOL_COMP_BB_PERIOD"]),
            "bb_std": float(params["VOL_COMP_BB_STD"]),
        }
    if strategy_name == "mtf_trend_pullback":
        return {
            "rsi_period": int(params["MTF_PULLBACK_RSI_PERIOD"]),
        }
    return {}


def _score_setting_overrides(strategy_name: str, params: dict[str, Any]) -> dict[str, Any]:
    if strategy_name == "mean_reversion":
        return {
            "SCORING_RSI_PERIOD": int(params["MEAN_REV_RSI_PERIOD"]),
            "SCORING_ATR_PERIOD": int(params["MEAN_REV_ATR_PERIOD"]),
            "SCORING_BB_PERIOD": int(params["MEAN_REV_BB_PERIOD"]),
        }
    if strategy_name == "volatility_compression":
        return {
            "SCORING_ATR_PERIOD": int(params["VOL_COMP_ATR_PERIOD"]),
            "SCORING_BB_PERIOD": int(params["VOL_COMP_BB_PERIOD"]),
        }
    if strategy_name == "mtf_trend_pullback":
        return {
            "SCORING_RSI_PERIOD": int(params["MTF_PULLBACK_RSI_PERIOD"]),
        }
    return {
        "SCORING_RSI_PERIOD": 14,
        "SCORING_ATR_PERIOD": int(getattr(settings, "ATR_PERIOD", 20)),
        "SCORING_BB_PERIOD": 20,
    }


def _build_frame_cache(symbols: list[str], lookback_days: int) -> dict[tuple[str, str], pd.DataFrame]:
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    for strategy_name, spec in STRATEGIES.items():
        for symbol in symbols:
            cache[(strategy_name, symbol)] = spec.frame_builder(symbol, lookback_days)
    return cache


def _evaluate_strategy_trial(
    strategy_name: str,
    params: dict[str, Any],
    symbols: list[str],
    frame_cache: dict[tuple[str, str], pd.DataFrame],
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    initial_balance: float,
) -> dict[str, Any]:
    spec = STRATEGIES[strategy_name]
    setting_overrides = {
        "ML_ENABLED": False,
        "POSITION_SIZING_ENABLED": False,
        "PARTIAL_TP_ENABLED": True,
        "BE_ENABLED": True,
        "TRAILING_ENABLED": True,
        "TIME_STOP_ENABLED": True,
        "PREFER_WORST_CASE": True,
        "BACKTEST_EXECUTION_DELAY_CANDLES": 1,
        **params,
        **_score_setting_overrides(strategy_name, params),
    }
    positions_frames: list[pd.DataFrame] = []
    raw_results: list[dict[str, Any]] = []

    with _temporary_settings(setting_overrides):
        for symbol in symbols:
            df = frame_cache[(strategy_name, symbol)].copy()
            metrics, trades_df, _, _ = run_backtest(
                df=df,
                initial_balance=initial_balance,
                strategy_fn=spec.strategy_fn,
                indicator_overrides=_indicator_overrides(strategy_name, params),
                score_fn=compute_signal_score,
                symbol=symbol,
                strategy_name=strategy_name,
            )
            positions_df = aggregate_positions(trades_df)
            if not positions_df.empty:
                positions_frames.append(positions_df)
            raw_results.append(
                {
                    "symbol": symbol,
                    "net_pnl": round(float(metrics.get("net_pnl", 0.0) or 0.0), 6),
                    "profit_factor": metrics.get("profit_factor", 0.0),
                    "total_trades": int(metrics.get("total_trades", 0) or 0),
                    "positions_closed": int(len(positions_df)),
                }
            )

        all_positions = pd.concat(positions_frames, ignore_index=True) if positions_frames else pd.DataFrame()
        if all_positions.empty:
            return {
                "strategy": strategy_name,
                "params": params,
                "overall_summary": summarize_portfolio_selection(
                    positions_df=all_positions,
                    initial_balance=initial_balance,
                    top_n_per_day=int(params["TOP_PER_DAY"]),
                    min_score=float(params["MIN_SCORE"]),
                    min_edge_score=-1.0,
                    source_candidate_count=0,
                ),
                "window_summaries": [],
                "target_windows": 0,
                "profitable_windows": 0,
                "avg_window_win_rate": 0.0,
                "min_window_win_rate": 0.0,
                "raw_results": raw_results,
            }

        all_positions["edge_score"] = 1.0
        selected = select_top_candidates_by_day(
            positions_df=all_positions,
            top_n_per_day=int(params["TOP_PER_DAY"]),
            min_score=float(params["MIN_SCORE"]),
            min_edge_score=-1.0,
            signal_weight=1.0,
            edge_weight=0.0,
            max_per_symbol_per_day=1,
            max_per_strategy_per_day=1,
        )
        overall_summary = summarize_portfolio_selection(
            positions_df=selected,
            initial_balance=initial_balance,
            top_n_per_day=int(params["TOP_PER_DAY"]),
            min_score=float(params["MIN_SCORE"]),
            min_edge_score=-1.0,
            source_candidate_count=len(all_positions),
        )

        window_summaries: list[dict[str, Any]] = []
        target_windows = 0
        profitable_windows = 0
        for window_start, window_end in windows:
            raw_window_positions = all_positions[
                (pd.to_datetime(all_positions["entry_time"], utc=True) >= window_start) &
                (pd.to_datetime(all_positions["entry_time"], utc=True) <= window_end)
            ].copy()
            window_selected = selected[
                (pd.to_datetime(selected["entry_time"], utc=True) >= window_start) &
                (pd.to_datetime(selected["entry_time"], utc=True) <= window_end)
            ].copy()
            summary = summarize_portfolio_selection(
                positions_df=window_selected,
                initial_balance=initial_balance,
                top_n_per_day=int(params["TOP_PER_DAY"]),
                min_score=float(params["MIN_SCORE"]),
                min_edge_score=-1.0,
                source_candidate_count=len(raw_window_positions),
            )
            summary["window_start"] = window_start.isoformat()
            summary["window_end"] = window_end.isoformat()
            window_summaries.append(summary)
            if float(summary["net_pnl"]) > 0:
                profitable_windows += 1
            if (
                int(summary["selected_positions"]) >= 2 and
                float(summary["win_rate_pct"]) >= 70.0 and
                float(summary["net_pnl"]) > 0 and
                _profit_factor_numeric(summary["profit_factor"]) >= 1.0
            ):
                target_windows += 1

        avg_window_win_rate = (
            sum(float(item["win_rate_pct"]) for item in window_summaries) / len(window_summaries)
            if window_summaries else 0.0
        )
        min_window_win_rate = min((float(item["win_rate_pct"]) for item in window_summaries), default=0.0)
        return {
            "strategy": strategy_name,
            "params": params,
            "overall_summary": overall_summary,
            "window_summaries": window_summaries,
            "target_windows": target_windows,
            "profitable_windows": profitable_windows,
            "avg_window_win_rate": round(avg_window_win_rate, 4),
            "min_window_win_rate": round(min_window_win_rate, 4),
            "raw_results": raw_results,
        }


def _trial_sort_key(result: dict[str, Any]) -> tuple[Any, ...]:
    overall = result.get("overall_summary", {})
    return (
        int(result.get("target_windows", 0)),
        int(result.get("profitable_windows", 0)),
        _profit_factor_numeric(overall.get("profit_factor")),
        float(overall.get("net_pnl", 0.0) or 0.0),
        float(overall.get("win_rate_pct", 0.0) or 0.0),
        int(overall.get("selected_positions", 0) or 0),
    )


def _trial_csv_row(trial_id: int, result: dict[str, Any]) -> dict[str, Any]:
    overall = result["overall_summary"]
    return {
        "trial_id": trial_id,
        "strategy": result["strategy"],
        "target_windows": result["target_windows"],
        "profitable_windows": result["profitable_windows"],
        "avg_window_win_rate": result["avg_window_win_rate"],
        "min_window_win_rate": result["min_window_win_rate"],
        "overall_selected_positions": overall.get("selected_positions"),
        "overall_net_pnl": overall.get("net_pnl"),
        "overall_profit_factor": overall.get("profit_factor"),
        "overall_win_rate_pct": overall.get("win_rate_pct"),
        "overall_max_drawdown": overall.get("max_drawdown"),
        "params_json": json.dumps(result["params"], sort_keys=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Search 100 robust multi-symbol backtest configurations.")
    parser.add_argument("--symbols", default="ETHUSDT,BTCUSDT,SOLUSDT", help="Comma-separated symbols.")
    parser.add_argument("--lookback-days", type=int, default=90, help="Backtest lookback range.")
    parser.add_argument("--window-days", type=int, default=21, help="Rolling window size for robustness checks.")
    parser.add_argument("--window-count", type=int, default=20, help="How many rolling windows to score.")
    parser.add_argument("--trials-per-strategy", type=int, default=20, help="Random configurations per strategy family.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--summary-path", default="reports/robust_search_summary.json", help="Output JSON summary path.")
    parser.add_argument("--trials-path", default="reports/robust_search_trials.csv", help="Output CSV path.")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.WARNING)
    rng = random.Random(int(args.seed))
    symbols = [part.strip().upper() for part in args.symbols.split(",") if part.strip()]
    frame_cache = _build_frame_cache(symbols=symbols, lookback_days=int(args.lookback_days))
    reference_index = frame_cache[("ema_crossover_baseline", symbols[0])].index
    windows = _window_schedule(reference_index, count=int(args.window_count), window_days=int(args.window_days))
    initial_balance = float(settings.INITIAL_BALANCE)

    summary_path = Path(args.summary_path)
    trials_path = Path(args.trials_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    trials_path.parent.mkdir(parents=True, exist_ok=True)

    csv_rows: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    trial_id = 0

    for strategy_name in STRATEGIES:
        strategy_trials = [_default_trial(strategy_name)]
        for _ in range(max(0, int(args.trials_per_strategy) - 1)):
            strategy_trials.append(_random_trial(strategy_name, rng))

        for params in strategy_trials:
            trial_id += 1
            result = _evaluate_strategy_trial(
                strategy_name=strategy_name,
                params=params,
                symbols=symbols,
                frame_cache=frame_cache,
                windows=windows,
                initial_balance=initial_balance,
            )
            result["trial_id"] = trial_id
            all_results.append(result)
            csv_rows.append(_trial_csv_row(trial_id, result))
            LOGGER.warning(
                "trial=%s strategy=%s target_windows=%s overall_win_rate=%.2f overall_net_pnl=%.6f",
                trial_id,
                strategy_name,
                result["target_windows"],
                float(result["overall_summary"].get("win_rate_pct", 0.0) or 0.0),
                float(result["overall_summary"].get("net_pnl", 0.0) or 0.0),
            )

    ranked = sorted(all_results, key=_trial_sort_key, reverse=True)
    best = ranked[0] if ranked else None
    report = {
        "symbols": symbols,
        "lookback_days": int(args.lookback_days),
        "window_days": int(args.window_days),
        "window_count": int(args.window_count),
        "trials_per_strategy": int(args.trials_per_strategy),
        "total_trials": int(trial_id),
        "target_definition": {
            "min_selected_positions_per_window": 2,
            "min_win_rate_pct": 70.0,
            "min_window_net_pnl": 0.0,
            "min_window_profit_factor": 1.0,
        },
        "best_result": best,
        "top_10": ranked[:10],
    }
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with trials_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()) if csv_rows else [])
        if csv_rows:
            writer.writeheader()
            writer.writerows(csv_rows)

    print(json.dumps(report, indent=2))
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved trials to {trials_path}")


if __name__ == "__main__":
    main()
