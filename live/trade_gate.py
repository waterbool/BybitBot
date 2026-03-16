from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from config import settings
from data_fetch.bybit_client import fetch_ticker
from live.edge_snapshot import snapshot_age_minutes
from live.scanner import LiveCandidate


@dataclass
class GateDecision:
    allowed: bool
    reasons: list[str]
    metrics: dict[str, Any]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _is_within_trading_hours(now: datetime) -> bool:
    start_hour = int(getattr(settings, "TRADING_START_HOUR", 0))
    end_hour = int(getattr(settings, "TRADING_END_HOUR", 24))
    hour = now.hour
    if start_hour == end_hour:
        return True
    if start_hour < end_hour:
        return start_hour <= hour < end_hour
    return hour >= start_hour or hour < end_hour


def _today_trade_counts(trades_history: list[dict], now: datetime) -> tuple[int, dict[str, int], dict[str, int]]:
    total_entries = 0
    symbol_counts: dict[str, int] = {}
    strategy_counts: dict[str, int] = {}
    today = now.date()
    for trade in trades_history:
        if trade.get("event_type", "entry") != "entry":
            continue
        ts_raw = trade.get("timestamp")
        if not ts_raw:
            continue
        try:
            ts = datetime.fromisoformat(ts_raw)
        except ValueError:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts.astimezone(timezone.utc).date() != today:
            continue
        total_entries += 1
        symbol = trade.get("symbol")
        strategy_name = trade.get("strategy_name")
        if symbol:
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        if strategy_name:
            strategy_counts[strategy_name] = strategy_counts.get(strategy_name, 0) + 1
    return total_entries, symbol_counts, strategy_counts


def evaluate_trade_gate(
    candidate: LiveCandidate,
    current_position: dict | None,
    trades_history: list[dict],
    edge_snapshot: dict | None,
    now: datetime | None = None,
) -> GateDecision:
    now = now or _utc_now()
    reasons: list[str] = []
    metrics: dict[str, Any] = {
        "signal_score": candidate.signal_score,
        "edge_score": candidate.edge_score,
        "selection_score": candidate.selection_score,
    }

    if not _is_within_trading_hours(now):
        reasons.append("outside_trading_hours")

    if getattr(settings, "LIVE_SELECTOR_REQUIRE_EDGE_SNAPSHOT", True):
        if not edge_snapshot:
            reasons.append("missing_edge_snapshot")
        else:
            age = snapshot_age_minutes(edge_snapshot, now=now)
            metrics["edge_snapshot_age_minutes"] = age
            if age is None or age > float(settings.LIVE_SELECTOR_EDGE_MAX_AGE_MINUTES):
                reasons.append("stale_edge_snapshot")

    signal_time = pd.to_datetime(candidate.signal_time, utc=True, errors="coerce")
    if pd.isna(signal_time):
        reasons.append("invalid_signal_time")
    else:
        stale_minutes = (now - signal_time.to_pydatetime()).total_seconds() / 60.0
        metrics["signal_age_minutes"] = stale_minutes
        if stale_minutes > float(settings.LIVE_SELECTOR_STALE_DATA_MINUTES):
            reasons.append("stale_signal")

    if candidate.signal_score < float(settings.LIVE_SELECTOR_MIN_SIGNAL_SCORE):
        reasons.append("signal_score_below_threshold")
    if candidate.edge_score < float(settings.LIVE_SELECTOR_MIN_EDGE_SCORE):
        reasons.append("edge_score_below_threshold")

    if current_position is not None:
        reasons.append("position_already_open")
        if current_position.get("symbol") == candidate.symbol:
            reasons.append("symbol_already_open")
        if current_position.get("strategy_name") == candidate.strategy_name:
            reasons.append("strategy_already_open")

    total_entries, symbol_counts, strategy_counts = _today_trade_counts(trades_history, now)
    metrics["today_total_entries"] = total_entries
    metrics["today_symbol_trades"] = symbol_counts.get(candidate.symbol, 0)
    metrics["today_strategy_trades"] = strategy_counts.get(candidate.strategy_name, 0)
    if total_entries >= int(settings.LIVE_SELECTOR_MAX_NEW_TRADES_PER_DAY):
        reasons.append("daily_new_trade_limit")
    if symbol_counts.get(candidate.symbol, 0) >= int(settings.LIVE_SELECTOR_MAX_PER_SYMBOL_PER_DAY):
        reasons.append("symbol_daily_limit")
    if strategy_counts.get(candidate.strategy_name, 0) >= int(settings.LIVE_SELECTOR_MAX_PER_STRATEGY_PER_DAY):
        reasons.append("strategy_daily_limit")

    try:
        ticker = fetch_ticker(candidate.symbol, category=settings.BYBIT_CATEGORY)
    except Exception as exc:
        metrics["ticker_error"] = str(exc)
        reasons.append("ticker_fetch_failed")
        ticker = {}
    metrics["ticker"] = ticker
    bid_price = ticker.get("bid_price")
    ask_price = ticker.get("ask_price")
    if bid_price is None or ask_price is None or bid_price <= 0 or ask_price <= 0:
        reasons.append("invalid_ticker")
    else:
        mid = (bid_price + ask_price) / 2.0
        spread_bps = ((ask_price - bid_price) / mid) * 10000.0 if mid > 0 else None
        metrics["spread_bps"] = spread_bps
        if spread_bps is None or spread_bps > float(settings.LIVE_SELECTOR_MAX_SPREAD_BPS):
            reasons.append("spread_too_wide")

    return GateDecision(allowed=not reasons, reasons=reasons, metrics=metrics)
