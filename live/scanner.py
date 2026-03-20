from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd

from backtest.multi_asset import resolve_strategy_specs
from backtest.scoring import compute_signal_score
from config import settings
from data_fetch.bybit_client import (
    fetch_funding_rate_history,
    fetch_historical_klines,
    fetch_open_interest_history,
)
from indicators.ta_module import add_indicators
from live.edge_snapshot import build_edge_lookup
from ml.model import add_ml_probabilities
from strategy.rules import apply_strategy as apply_baseline_strategy


def _interval_to_ms(interval: str) -> int:
    raw = str(interval).strip().upper()
    if raw.isdigit():
        return int(raw) * 60 * 1000
    if raw.endswith("D"):
        days = raw[:-1] or "1"
        return int(days) * 24 * 60 * 60 * 1000
    raise ValueError(f"Unsupported interval format: {interval}")


def _closed_cutoff_ts(now_ms: int, interval: str) -> int:
    interval_ms = _interval_to_ms(interval)
    return (now_ms // interval_ms) * interval_ms


def _prepare_closed_price_frame(symbol: str, interval: str, lookback_bars: int, now_ms: int) -> pd.DataFrame:
    interval_ms = _interval_to_ms(interval)
    start_ts = now_ms - ((int(lookback_bars) + 5) * interval_ms)
    df = fetch_historical_klines(symbol, interval, start_ts, now_ms, category=settings.BYBIT_CATEGORY)
    cutoff = _closed_cutoff_ts(now_ms, interval)
    df = df[df["timestamp"] < cutoff].copy()
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    return df


def _prepare_indicator_frame(df: pd.DataFrame, indicator_overrides: dict[str, float | int] | None = None) -> pd.DataFrame:
    indicator_overrides = indicator_overrides or {}
    prepared = add_indicators(
        df.set_index("timestamp"),
        ema_fast=int(indicator_overrides.get("ema_fast", settings.EMA_FAST)),
        ema_slow=int(indicator_overrides.get("ema_slow", settings.EMA_SLOW)),
        rsi_period=int(indicator_overrides.get("rsi_period", 14)),
        atr_period=int(indicator_overrides.get("atr_period", settings.ATR_PERIOD)),
        bb_period=int(indicator_overrides.get("bb_period", 20)),
        bb_std=float(indicator_overrides.get("bb_std", 2.0)),
    )
    return prepared


def _build_base_frame(symbol: str, indicator_overrides: dict[str, float | int] | None, now_ms: int) -> pd.DataFrame:
    df = _prepare_closed_price_frame(
        symbol=symbol,
        interval=settings.LIVE_SELECTOR_BASE_INTERVAL,
        lookback_bars=settings.LIVE_SELECTOR_BASE_LOOKBACK_BARS,
        now_ms=now_ms,
    )
    return _prepare_indicator_frame(df, indicator_overrides)


def _build_mtf_frame(symbol: str, indicator_overrides: dict[str, float | int] | None, now_ms: int) -> pd.DataFrame:
    base_df = _prepare_closed_price_frame(
        symbol=symbol,
        interval=settings.LIVE_SELECTOR_BASE_INTERVAL,
        lookback_bars=settings.LIVE_SELECTOR_BASE_LOOKBACK_BARS,
        now_ms=now_ms,
    )
    htf_df = _prepare_closed_price_frame(
        symbol=symbol,
        interval=settings.LIVE_SELECTOR_HTF_INTERVAL,
        lookback_bars=settings.LIVE_SELECTOR_HTF_LOOKBACK_BARS,
        now_ms=now_ms,
    )
    htf_df["ema200_1h"] = htf_df["close"].ewm(span=200, adjust=False).mean()
    htf_df["bias"] = 0
    htf_df.loc[htf_df["close"] > htf_df["ema200_1h"], "bias"] = 1
    htf_df.loc[htf_df["close"] < htf_df["ema200_1h"], "bias"] = -1
    merged = pd.merge_asof(
        base_df.sort_values("timestamp"),
        htf_df[["timestamp", "bias", "ema200_1h"]].sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    return _prepare_indicator_frame(merged, indicator_overrides)


def _build_funding_frame(symbol: str, indicator_overrides: dict[str, float | int] | None, now_ms: int) -> pd.DataFrame:
    base_df = _prepare_closed_price_frame(
        symbol=symbol,
        interval=settings.LIVE_SELECTOR_BASE_INTERVAL,
        lookback_bars=settings.LIVE_SELECTOR_BASE_LOOKBACK_BARS,
        now_ms=now_ms,
    )
    start_ts = int(base_df["timestamp"].min().timestamp() * 1000) if not base_df.empty else now_ms
    end_ts = _closed_cutoff_ts(now_ms, settings.LIVE_SELECTOR_BASE_INTERVAL) - 1
    funding_df = fetch_funding_rate_history(symbol, start_ts, end_ts, category=settings.BYBIT_CATEGORY)
    oi_df = fetch_open_interest_history(symbol, "15min", start_ts, end_ts, category=settings.BYBIT_CATEGORY)
    if not funding_df.empty:
        funding_df["timestamp"] = pd.to_datetime(funding_df["timestamp"].astype("int64"), unit="ms", utc=True)
        funding_df = funding_df.sort_values("timestamp")
    if not oi_df.empty:
        oi_df["timestamp"] = pd.to_datetime(oi_df["timestamp"].astype("int64"), unit="ms", utc=True)
        oi_df = oi_df.sort_values("timestamp")
    merged = pd.merge_asof(
        base_df.sort_values("timestamp"),
        funding_df.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    merged = pd.merge_asof(
        merged.sort_values("timestamp"),
        oi_df.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    return _prepare_indicator_frame(merged, indicator_overrides)


def _build_strategy_frame(symbol: str, strategy_name: str, indicator_overrides: dict[str, float | int] | None, now_ms: int) -> pd.DataFrame:
    if strategy_name == "mtf_trend_pullback":
        return _build_mtf_frame(symbol, indicator_overrides, now_ms)
    if strategy_name == "funding_extreme_reversal":
        return _build_funding_frame(symbol, indicator_overrides, now_ms)
    return _build_base_frame(symbol, indicator_overrides, now_ms)


@dataclass
class LiveCandidate:
    symbol: str
    strategy_name: str
    side: int
    signal_time: str
    close_price: float
    signal_score: float
    edge_score: float
    selection_score: float
    signal_components: dict[str, Any]
    edge_components: dict[str, Any]
    signal_row: dict[str, Any]
    base_interval: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["side_label"] = "BUY" if self.side == 1 else "SELL"
        return payload


def scan_live_candidates(
    symbols: list[str],
    edge_snapshot: dict | None,
    strategy_names: list[str] | None = None,
    now_ms: int | None = None,
) -> list[LiveCandidate]:
    now_ms = int(now_ms if now_ms is not None else pd.Timestamp.utcnow().timestamp() * 1000)
    edge_lookup = build_edge_lookup(edge_snapshot)
    specs = resolve_strategy_specs(strategy_names or settings.LIVE_SELECTOR_STRATEGIES)
    candidates: list[LiveCandidate] = []
    use_ml_filter = bool(settings.LIVE_SELECTOR_USE_ML)

    for symbol in symbols:
        for spec in specs:
            frame = _build_strategy_frame(symbol, spec.name, spec.indicator_overrides, now_ms)
            if frame.empty:
                continue
            working = frame.copy()
            if use_ml_filter:
                working = add_ml_probabilities(working)
            if spec.strategy_fn is apply_baseline_strategy:
                signal_df = spec.strategy_fn(working.copy(), ml_enabled=use_ml_filter)
            else:
                signal_df = spec.strategy_fn(working.copy())
            signal = int(signal_df.iloc[-1]["signal"])
            if signal == 0:
                continue

            score_payload = compute_signal_score(signal_df, signal)
            signal_score = float(score_payload.get("score", 0.0) or 0.0)

            edge_row = edge_lookup.get((symbol, spec.name), {})
            edge_score = float(edge_row.get("edge_score", 0.0) or 0.0)

            total_weight = float(settings.LIVE_SELECTOR_SIGNAL_WEIGHT) + float(settings.LIVE_SELECTOR_EDGE_WEIGHT)
            if total_weight <= 0:
                total_weight = 1.0
            selection_score = (
                (signal_score * float(settings.LIVE_SELECTOR_SIGNAL_WEIGHT)) +
                (edge_score * float(settings.LIVE_SELECTOR_EDGE_WEIGHT))
            ) / total_weight

            last_row = signal_df.iloc[-1]
            row_payload = last_row.to_dict()
            if hasattr(last_row.name, "timestamp"):
                row_payload["timestamp_ms"] = int(last_row.name.timestamp() * 1000)
                row_payload["timestamp"] = last_row.name.isoformat()
            else:
                row_payload["timestamp"] = str(last_row.name)

            candidates.append(
                LiveCandidate(
                    symbol=symbol,
                    strategy_name=spec.name,
                    side=signal,
                    signal_time=last_row.name.isoformat(),
                    close_price=float(last_row["close"]),
                    signal_score=round(signal_score, 6),
                    edge_score=round(edge_score, 6),
                    selection_score=round(selection_score, 6),
                    signal_components=score_payload.get("components", {}) or {},
                    edge_components=edge_row,
                    signal_row=row_payload,
                    base_interval=settings.LIVE_SELECTOR_BASE_INTERVAL,
                )
            )

    candidates.sort(
        key=lambda item: (
            item.selection_score,
            item.signal_score,
            item.edge_score,
        ),
        reverse=True,
    )
    return candidates
