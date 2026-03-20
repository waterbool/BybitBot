from __future__ import annotations

import time

import pandas as pd

from config import settings
from data_fetch.bybit_client import fetch_historical_klines
from indicators.ta_module import add_indicators
from ml.model import add_ml_probabilities
from strategy.rules import apply_strategy


def interval_to_minutes(interval_value: str) -> int:
    """Convert Bybit interval string to minutes (e.g. '1', '60', 'D', '1D')."""
    s = str(interval_value).strip().upper()
    if s.isdigit():
        return int(s)
    if s.endswith("D"):
        days = s[:-1]
        if days == "":
            return 1440
        if days.isdigit():
            return int(days) * 1440
    raise ValueError(f"Unsupported interval format: {interval_value}")


def required_history_bars() -> int:
    ml_min_bars = int(getattr(settings, "ML_MIN_TRAIN_SAMPLES", 300)) + int(getattr(settings, "ML_HORIZON", 12)) + 50
    return max(200, ml_min_bars)


def trim_to_closed_candles(df: pd.DataFrame, interval: str, now_ms: int | None = None) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    now_ms = int(now_ms if now_ms is not None else time.time() * 1000)
    interval_minutes = interval_to_minutes(interval)
    current_open_ts = (now_ms // (interval_minutes * 60 * 1000)) * (interval_minutes * 60 * 1000)
    return df[df["timestamp"] < current_open_ts].copy()


def prepare_indicator_frame(df: pd.DataFrame) -> pd.DataFrame:
    return add_indicators(
        df,
        ema_fast=settings.EMA_FAST,
        ema_slow=settings.EMA_SLOW,
        atr_period=settings.ATR_PERIOD,
    )


def prepare_signal_frame(df: pd.DataFrame, ml_enabled: bool | None = None) -> pd.DataFrame:
    prepared = prepare_indicator_frame(df)
    use_ml = getattr(settings, "ML_ENABLED", False) if ml_enabled is None else bool(ml_enabled)
    if use_ml:
        prepared = add_ml_probabilities(prepared)
    return apply_strategy(prepared, ml_enabled=use_ml)


def fetch_price_frame_in_window(
    symbol: str,
    interval: str,
    start_ts: int,
    end_ts: int,
    category: str | None = None,
    now_ms: int | None = None,
) -> tuple[pd.DataFrame, int]:
    target_category = category or settings.BYBIT_CATEGORY
    raw_df = fetch_historical_klines(symbol, interval, start_ts, end_ts, category=target_category)
    interval_minutes = interval_to_minutes(interval)
    closed_df = trim_to_closed_candles(raw_df, interval, now_ms=now_ms)
    return closed_df, interval_minutes


def fetch_indicator_frame_for_lookback(
    symbol: str,
    interval: str,
    lookback_bars: int,
    category: str | None = None,
    now_ms: int | None = None,
) -> tuple[pd.DataFrame, int]:
    now_ms = int(now_ms if now_ms is not None else time.time() * 1000)
    interval_minutes = interval_to_minutes(interval)
    start_ts = now_ms - ((int(lookback_bars) + 3) * interval_minutes * 60 * 1000)
    closed_df, _ = fetch_price_frame_in_window(
        symbol=symbol,
        interval=interval,
        start_ts=start_ts,
        end_ts=now_ms,
        category=category,
        now_ms=now_ms,
    )
    return prepare_indicator_frame(closed_df), interval_minutes


def fetch_signal_frame_for_lookback(
    symbol: str,
    interval: str,
    lookback_bars: int | None = None,
    category: str | None = None,
    now_ms: int | None = None,
    ml_enabled: bool | None = None,
) -> tuple[pd.DataFrame, int]:
    bars = int(lookback_bars if lookback_bars is not None else required_history_bars())
    now_ms = int(now_ms if now_ms is not None else time.time() * 1000)
    interval_minutes = interval_to_minutes(interval)
    start_ts = now_ms - ((bars + 3) * interval_minutes * 60 * 1000)
    closed_df, _ = fetch_price_frame_in_window(
        symbol=symbol,
        interval=interval,
        start_ts=start_ts,
        end_ts=now_ms,
        category=category,
        now_ms=now_ms,
    )
    return prepare_signal_frame(closed_df, ml_enabled=ml_enabled), interval_minutes
