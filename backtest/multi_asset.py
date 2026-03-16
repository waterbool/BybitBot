from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from backtest.backtester import run_backtest
from backtest.market_data import build_funding_frame, build_mtf_frame, load_base_15m
from backtest.scoring import compute_signal_score
from config import settings
from strategy.rules import (
    apply_funding_extreme_reversal_strategy,
    apply_mean_reversion_strategy,
    apply_mtf_trend_pullback_strategy,
    apply_strategy,
    apply_volatility_compression_breakout_strategy,
)


FrameBuilder = Callable[[str, int], pd.DataFrame]
StrategyFn = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class StrategySpec:
    name: str
    strategy_fn: StrategyFn
    frame_builder: FrameBuilder
    indicator_overrides: dict[str, float | int] | None = None


STRATEGY_SPECS: dict[str, StrategySpec] = {
    "ema_crossover_baseline": StrategySpec(
        name="ema_crossover_baseline",
        strategy_fn=apply_strategy,
        frame_builder=load_base_15m,
    ),
    "mean_reversion": StrategySpec(
        name="mean_reversion",
        strategy_fn=apply_mean_reversion_strategy,
        frame_builder=load_base_15m,
        indicator_overrides={"atr_period": 14},
    ),
    "volatility_compression": StrategySpec(
        name="volatility_compression",
        strategy_fn=apply_volatility_compression_breakout_strategy,
        frame_builder=load_base_15m,
        indicator_overrides={"atr_period": 14},
    ),
    "mtf_trend_pullback": StrategySpec(
        name="mtf_trend_pullback",
        strategy_fn=apply_mtf_trend_pullback_strategy,
        frame_builder=build_mtf_frame,
        indicator_overrides={"atr_period": 14},
    ),
    "funding_extreme_reversal": StrategySpec(
        name="funding_extreme_reversal",
        strategy_fn=apply_funding_extreme_reversal_strategy,
        frame_builder=build_funding_frame,
        indicator_overrides={"atr_period": 14},
    ),
}

DEFAULT_STRATEGIES = list(STRATEGY_SPECS.keys())


def normalize_symbols(symbols: list[str]) -> list[str]:
    return [symbol.strip().upper() for symbol in symbols if symbol and symbol.strip()]


def resolve_strategy_specs(strategy_names: list[str] | None = None) -> list[StrategySpec]:
    if not strategy_names:
        return [STRATEGY_SPECS[name] for name in DEFAULT_STRATEGIES]
    specs = []
    for name in strategy_names:
        key = name.strip()
        if key not in STRATEGY_SPECS:
            raise KeyError(f"Unknown strategy: {key}")
        specs.append(STRATEGY_SPECS[key])
    return specs


def aggregate_positions(trades_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "symbol",
        "strategy_name",
        "position_id",
        "entry_time",
        "signal_time",
        "exit_time",
        "entry_day",
        "side",
        "pnl",
        "gross_pnl",
        "entry_fee",
        "exit_fee",
        "slippage_cost",
        "position_size_usdt",
        "signal_score",
        "score_components",
        "final_reason",
        "legs",
        "partial_legs",
        "result",
    ]
    if trades_df is None or trades_df.empty:
        return pd.DataFrame(columns=columns)

    df = trades_df.copy()
    for col in ("entry_time", "signal_time", "exit_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    position_rows = []
    grouped = df.sort_values(["entry_time", "exit_time"]).groupby("position_id", dropna=False)
    for position_id, group in grouped:
        final_rows = group[group.get("is_final", False)] if "is_final" in group.columns else group.iloc[[-1]]
        final_row = final_rows.iloc[-1] if not final_rows.empty else group.iloc[-1]
        signal_scores = pd.to_numeric(group.get("signal_score"), errors="coerce")
        signal_score = float(signal_scores.fillna(0.0).iloc[0]) if signal_scores is not None and not signal_scores.empty else 0.0
        score_components = group["score_components"].dropna().iloc[0] if "score_components" in group.columns and not group["score_components"].dropna().empty else {}
        entry_time = group["entry_time"].iloc[0] if "entry_time" in group.columns else pd.NaT
        exit_time = group["exit_time"].max() if "exit_time" in group.columns else pd.NaT
        pnl = float(group["pnl"].sum()) if "pnl" in group.columns else 0.0
        row = {
            "symbol": final_row.get("symbol"),
            "strategy_name": final_row.get("strategy_name"),
            "position_id": position_id,
            "entry_time": entry_time,
            "signal_time": group["signal_time"].dropna().iloc[0] if "signal_time" in group.columns and not group["signal_time"].dropna().empty else pd.NaT,
            "exit_time": exit_time,
            "entry_day": entry_time.floor("D") if not pd.isna(entry_time) else pd.NaT,
            "side": final_row.get("side"),
            "pnl": pnl,
            "gross_pnl": float(group["gross_pnl"].sum()) if "gross_pnl" in group.columns else 0.0,
            "entry_fee": float(group["entry_fee"].sum()) if "entry_fee" in group.columns else 0.0,
            "exit_fee": float(group["exit_fee"].sum()) if "exit_fee" in group.columns else 0.0,
            "slippage_cost": float(group["slippage_cost"].sum()) if "slippage_cost" in group.columns else 0.0,
            "position_size_usdt": float(group["position_size_usdt"].dropna().iloc[0]) if "position_size_usdt" in group.columns and not group["position_size_usdt"].dropna().empty else 0.0,
            "signal_score": signal_score,
            "score_components": score_components,
            "final_reason": final_row.get("final_reason") or final_row.get("reason"),
            "legs": int(len(group)),
            "partial_legs": int((group.get("reason") == "TP1").sum()) if "reason" in group.columns else 0,
            "result": "WIN" if pnl > 0 else "LOSS",
        }
        position_rows.append(row)

    positions_df = pd.DataFrame(position_rows)
    if not positions_df.empty:
        positions_df = positions_df.sort_values(["entry_time", "exit_time", "signal_score"], ascending=[True, True, False])
    return positions_df


def _profit_factor(series: pd.Series) -> float | str:
    gross_profit = float(series[series > 0].sum())
    gross_loss = float(-series[series < 0].sum())
    if gross_loss == 0:
        return "inf" if gross_profit > 0 else 0.0
    return round(gross_profit / gross_loss, 6)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _as_numeric_profit_factor(value: float | str | None) -> float:
    if value == "inf":
        return 1.25
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def compute_run_edge_score(run_summary: dict) -> dict:
    if not run_summary or run_summary.get("status") != "ok":
        return {"score": 0.0, "components": {}}

    profit_factor = _as_numeric_profit_factor(run_summary.get("profit_factor"))
    net_pnl = float(run_summary.get("net_pnl", 0.0) or 0.0)
    max_drawdown = abs(float(run_summary.get("max_drawdown", 0.0) or 0.0))
    win_rate_pct = float(run_summary.get("win_rate_pct", 0.0) or 0.0)
    avg_signal_score = float(run_summary.get("avg_signal_score", 0.0) or 0.0)
    positions_closed = int(run_summary.get("positions_closed", 0) or 0)

    components = {
        "profit_factor": _clip01(profit_factor / 1.0),
        "net_pnl": _clip01(1.0 if net_pnl >= 0 else 1.0 - (abs(net_pnl) / 0.5)),
        "drawdown": _clip01(1.0 - (max_drawdown / 0.75)),
        "win_rate": _clip01(win_rate_pct / 60.0),
        "signal_consistency": _clip01(avg_signal_score),
        "sample_size": _clip01(positions_closed / 25.0),
    }
    weights = {
        "profit_factor": 0.32,
        "net_pnl": 0.23,
        "drawdown": 0.16,
        "win_rate": 0.12,
        "signal_consistency": 0.07,
        "sample_size": 0.10,
    }
    score = sum(components[name] * weights[name] for name in components)
    return {
        "score": round(_clip01(score), 6),
        "components": {name: round(value, 6) for name, value in components.items()},
    }


def attach_run_context(positions_df: pd.DataFrame, results: list[dict]) -> pd.DataFrame:
    if positions_df is None or positions_df.empty:
        return aggregate_positions(pd.DataFrame())

    run_rows = []
    for result in results:
        if result.get("status") != "ok":
            continue
        edge_payload = compute_run_edge_score(result)
        run_rows.append(
            {
                "symbol": result.get("symbol"),
                "strategy_name": result.get("strategy"),
                "run_net_pnl": float(result.get("net_pnl", 0.0) or 0.0),
                "run_profit_factor": result.get("profit_factor"),
                "run_win_rate_pct": float(result.get("win_rate_pct", 0.0) or 0.0),
                "run_max_drawdown": float(result.get("max_drawdown", 0.0) or 0.0),
                "run_positions_closed": int(result.get("positions_closed", 0) or 0),
                "run_avg_signal_score": float(result.get("avg_signal_score", 0.0) or 0.0),
                "edge_score": edge_payload["score"],
                "edge_components": edge_payload["components"],
            }
        )

    if not run_rows:
        enriched = positions_df.copy()
        enriched["edge_score"] = 0.0
        enriched["edge_components"] = [{} for _ in range(len(enriched))]
        return enriched

    run_df = pd.DataFrame(run_rows)
    enriched = positions_df.merge(run_df, on=["symbol", "strategy_name"], how="left")
    enriched["edge_score"] = pd.to_numeric(enriched["edge_score"], errors="coerce").fillna(0.0)
    if "edge_components" in enriched.columns:
        enriched["edge_components"] = enriched["edge_components"].apply(
            lambda value: value if isinstance(value, dict) else {}
        )
    return enriched


def summarize_run(
    symbol: str,
    strategy_name: str,
    df: pd.DataFrame | None,
    metrics: dict | None,
    positions_df: pd.DataFrame | None,
    trades_df: pd.DataFrame | None,
    equity_df: pd.DataFrame | None,
    monthly_stats: pd.DataFrame | None,
    error: str | None = None,
) -> dict:
    if error is not None:
        return {
            "symbol": symbol,
            "strategy": strategy_name,
            "status": "error",
            "error": error,
        }

    positions_df = positions_df if positions_df is not None else pd.DataFrame()
    avg_signal_score = float(positions_df["signal_score"].mean()) if not positions_df.empty else 0.0
    top_signal_score = float(positions_df["signal_score"].max()) if not positions_df.empty else 0.0
    win_rate = float((positions_df["pnl"] > 0).mean() * 100.0) if not positions_df.empty else 0.0
    return {
        "symbol": symbol,
        "strategy": strategy_name,
        "status": "ok",
        "bars": int(len(df)) if df is not None else 0,
        "from": df.index.min().isoformat() if df is not None and not df.empty else None,
        "to": df.index.max().isoformat() if df is not None and not df.empty else None,
        "net_pnl": round(float(metrics["net_pnl"]), 6) if metrics else 0.0,
        "final_balance": round(float(metrics["final_balance"]), 6) if metrics else 0.0,
        "total_trades": int(metrics["total_trades"]) if metrics else 0,
        "positions_closed": int(len(positions_df)),
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(float(metrics["profit_factor"]), 6) if metrics and metrics["profit_factor"] != float("inf") else ("inf" if metrics and metrics["profit_factor"] == float("inf") else 0.0),
        "max_drawdown": round(float(metrics["max_drawdown"]), 6) if metrics else 0.0,
        "fees": round(float(metrics["total_fees"]), 6) if metrics else 0.0,
        "avg_signal_score": round(avg_signal_score, 6),
        "top_signal_score": round(top_signal_score, 6),
        "edge_score": compute_run_edge_score(
            {
                "status": "ok",
                "profit_factor": metrics["profit_factor"] if metrics else 0.0,
                "net_pnl": metrics["net_pnl"] if metrics else 0.0,
                "max_drawdown": metrics["max_drawdown"] if metrics else 0.0,
                "win_rate_pct": win_rate,
                "avg_signal_score": avg_signal_score,
                "positions_closed": len(positions_df),
            }
        )["score"],
        "monthly_rows": int(len(monthly_stats)) if monthly_stats is not None else 0,
        "equity_rows": int(len(equity_df)) if equity_df is not None else 0,
    }


def run_multi_symbol_backtests(
    symbols: list[str],
    strategy_names: list[str] | None = None,
    lookback_days: int = 90,
    initial_balance: float | None = None,
    enable_ml: bool = False,
) -> tuple[list[dict], pd.DataFrame]:
    symbols = normalize_symbols(symbols)
    specs = resolve_strategy_specs(strategy_names)
    initial_balance = float(initial_balance if initial_balance is not None else settings.INITIAL_BALANCE)

    results: list[dict] = []
    position_frames: list[pd.DataFrame] = []
    original_ml_enabled = settings.ML_ENABLED
    settings.ML_ENABLED = bool(enable_ml)
    try:
        for symbol in symbols:
            for spec in specs:
                try:
                    df = spec.frame_builder(symbol, lookback_days)
                    metrics, trades_df, equity_df, monthly_stats = run_backtest(
                        df=df,
                        initial_balance=initial_balance,
                        strategy_fn=spec.strategy_fn,
                        indicator_overrides=spec.indicator_overrides,
                        score_fn=compute_signal_score,
                        symbol=symbol,
                        strategy_name=spec.name,
                    )
                    positions_df = aggregate_positions(trades_df)
                    results.append(
                        summarize_run(symbol, spec.name, df, metrics, positions_df, trades_df, equity_df, monthly_stats)
                    )
                    if not positions_df.empty:
                        position_frames.append(positions_df)
                except Exception as exc:
                    results.append(
                        summarize_run(symbol, spec.name, None, None, None, None, None, None, error=str(exc))
                    )
    finally:
        settings.ML_ENABLED = original_ml_enabled

    positions_df = pd.concat(position_frames, ignore_index=True) if position_frames else aggregate_positions(pd.DataFrame())
    positions_df = attach_run_context(positions_df, results)
    if not positions_df.empty:
        positions_df = positions_df.sort_values(["entry_time", "signal_score"], ascending=[True, False]).reset_index(drop=True)
    return results, positions_df


def select_top_candidates_by_day(
    positions_df: pd.DataFrame,
    top_n_per_day: int,
    min_score: float = 0.0,
    min_edge_score: float = 0.0,
    signal_weight: float = 0.5,
    edge_weight: float = 0.5,
    max_per_symbol_per_day: int | None = None,
    max_per_strategy_per_day: int | None = None,
) -> pd.DataFrame:
    if positions_df is None or positions_df.empty:
        return aggregate_positions(pd.DataFrame())

    filtered = positions_df.copy()
    filtered["signal_score"] = pd.to_numeric(filtered["signal_score"], errors="coerce").fillna(0.0)
    filtered["edge_score"] = pd.to_numeric(filtered.get("edge_score"), errors="coerce").fillna(0.0)
    filtered = filtered[
        (filtered["signal_score"] >= float(min_score)) &
        (filtered["edge_score"] >= float(min_edge_score))
    ].copy()
    if filtered.empty:
        return filtered

    filtered["entry_day"] = pd.to_datetime(filtered["entry_time"], utc=True, errors="coerce").dt.floor("D")
    total_weight = float(signal_weight) + float(edge_weight)
    if total_weight <= 0:
        total_weight = 1.0
        signal_weight = 1.0
        edge_weight = 0.0
    filtered["selection_score"] = (
        (filtered["signal_score"] * float(signal_weight)) +
        (filtered["edge_score"] * float(edge_weight))
    ) / total_weight
    filtered = filtered.sort_values(
        ["entry_day", "selection_score", "signal_score", "edge_score", "entry_time", "symbol", "strategy_name"],
        ascending=[True, False, False, False, True, True, True],
    )

    selected_groups: list[pd.DataFrame] = []
    for entry_day, day_group in filtered.groupby("entry_day", dropna=True):
        del entry_day
        picked_indices: list[int] = []
        symbol_counts: dict[str, int] = {}
        strategy_counts: dict[str, int] = {}
        for row in day_group.itertuples():
            symbol = getattr(row, "symbol", None)
            strategy_name = getattr(row, "strategy_name", None)
            if max_per_symbol_per_day is not None and symbol_counts.get(symbol, 0) >= int(max_per_symbol_per_day):
                continue
            if max_per_strategy_per_day is not None and strategy_counts.get(strategy_name, 0) >= int(max_per_strategy_per_day):
                continue
            picked_indices.append(row.Index)
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            strategy_counts[strategy_name] = strategy_counts.get(strategy_name, 0) + 1
            if len(picked_indices) >= int(top_n_per_day):
                break
        if picked_indices:
            day_selected = day_group.loc[picked_indices].copy()
            day_selected["daily_rank"] = range(1, len(day_selected) + 1)
            selected_groups.append(day_selected)

    if not selected_groups:
        return filtered.iloc[0:0].copy()
    return pd.concat(selected_groups, ignore_index=True)


def summarize_portfolio_selection(
    positions_df: pd.DataFrame,
    initial_balance: float,
    top_n_per_day: int,
    min_score: float,
    min_edge_score: float,
    source_candidate_count: int,
) -> dict:
    if positions_df is None or positions_df.empty:
        return {
            "status": "ok",
            "selected_positions": 0,
            "source_candidate_count": int(source_candidate_count),
            "top_n_per_day": int(top_n_per_day),
            "min_score": float(min_score),
            "min_edge_score": float(min_edge_score),
            "net_pnl": 0.0,
            "final_balance": float(initial_balance),
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "win_rate_pct": 0.0,
            "avg_signal_score": 0.0,
            "avg_edge_score": 0.0,
            "avg_selection_score": 0.0,
            "symbols_traded": 0,
            "strategies_used": 0,
            "counts_by_symbol": {},
            "counts_by_strategy": {},
        }

    ordered = positions_df.sort_values(["exit_time", "entry_time"]).copy()
    ordered["equity"] = float(initial_balance) + ordered["pnl"].cumsum()
    ordered["peak"] = ordered["equity"].cummax()
    ordered["drawdown"] = ordered["equity"] - ordered["peak"]

    win_rate = float((ordered["pnl"] > 0).mean() * 100.0)
    counts_by_symbol = {key: int(value) for key, value in ordered["symbol"].value_counts().to_dict().items()}
    counts_by_strategy = {key: int(value) for key, value in ordered["strategy_name"].value_counts().to_dict().items()}
    return {
        "status": "ok",
        "selected_positions": int(len(ordered)),
        "source_candidate_count": int(source_candidate_count),
        "top_n_per_day": int(top_n_per_day),
        "min_score": float(min_score),
        "min_edge_score": float(min_edge_score),
        "net_pnl": round(float(ordered["pnl"].sum()), 6),
        "final_balance": round(float(ordered["equity"].iloc[-1]), 6),
        "profit_factor": _profit_factor(ordered["pnl"]),
        "max_drawdown": round(float(ordered["drawdown"].min()), 6),
        "win_rate_pct": round(win_rate, 2),
        "avg_signal_score": round(float(ordered["signal_score"].mean()), 6),
        "avg_edge_score": round(float(ordered["edge_score"].mean()), 6) if "edge_score" in ordered.columns else 0.0,
        "avg_selection_score": round(float(ordered["selection_score"].mean()), 6) if "selection_score" in ordered.columns else 0.0,
        "symbols_traded": int(ordered["symbol"].nunique()),
        "strategies_used": int(ordered["strategy_name"].nunique()),
        "counts_by_symbol": counts_by_symbol,
        "counts_by_strategy": counts_by_strategy,
    }
