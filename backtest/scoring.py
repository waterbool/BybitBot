from __future__ import annotations

from typing import Any

import pandas as pd

from config import settings


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    try:
        numerator = float(numerator)
        denominator = float(denominator)
    except (TypeError, ValueError):
        return None
    if denominator == 0:
        return None
    return numerator / denominator


def _first_available(row: pd.Series, candidates: list[str]) -> Any:
    for column in candidates:
        value = row.get(column)
        if value is not None and not pd.isna(value):
            return value
    return None


def compute_signal_score(df: pd.DataFrame, signal: int) -> dict:
    if df.empty or signal == 0:
        return {"score": 0.0, "components": {}}

    row = df.iloc[-1]
    prev_row = df.iloc[-2] if len(df) > 1 else None
    components: dict[str, float] = {}

    close = row.get("close")
    scoring_atr_period = int(getattr(settings, "SCORING_ATR_PERIOD", getattr(settings, "ATR_PERIOD", 14)))
    atr = _first_available(row, [f"ATR_{scoring_atr_period}", "ATR_14", f"ATR_{settings.ATR_PERIOD}"])
    atr_pct = _safe_ratio(atr, close)
    if atr_pct is not None:
        baseline = max(float(getattr(settings, "MIN_ATR_THRESHOLD", 0.0015)), 1e-6)
        components["volatility"] = _clip01(atr_pct / (baseline * 2.0))

    ema200 = row.get("EMA_200")
    if pd.isna(ema200) if ema200 is not None else True:
        ema200 = row.get("SMA_200")
    trend_gap = _safe_ratio(abs(float(close) - float(ema200)), ema200) if close is not None and ema200 not in (None, 0) and not pd.isna(ema200) else None
    if trend_gap is not None:
        aligned = 0.0
        if signal == 1 and float(close) > float(ema200):
            aligned = 1.0
        elif signal == -1 and float(close) < float(ema200):
            aligned = 1.0
        components["trend"] = _clip01((trend_gap / 0.01) * 0.7 + aligned * 0.3)

    volume = row.get("volume")
    volume_sma = row.get("Volume_SMA_20")
    volume_ratio = _safe_ratio(volume, volume_sma)
    if volume_ratio is not None:
        components["volume"] = _clip01(volume_ratio / 2.0)

    if prev_row is not None:
        prev_close = prev_row.get("close")
        ret1 = None
        if prev_close not in (None, 0) and not pd.isna(prev_close) and close is not None and not pd.isna(close):
            ret1 = abs((float(close) / float(prev_close)) - 1.0)
        if ret1 is not None:
            impulse_baseline = max(float(getattr(settings, "IMPULSE_THRESHOLD", 0.002)), 1e-6)
            components["impulse"] = _clip01(ret1 / (impulse_baseline * 2.0))

    scoring_rsi_period = int(getattr(settings, "SCORING_RSI_PERIOD", 14))
    rsi_value = _first_available(row, [f"RSI_{scoring_rsi_period}", "RSI_14"])
    if rsi_value is not None and not pd.isna(rsi_value):
        if signal == 1:
            components["rsi"] = _clip01((50.0 - float(rsi_value)) / 25.0)
        else:
            components["rsi"] = _clip01((float(rsi_value) - 50.0) / 25.0)

    scoring_bb_period = int(getattr(settings, "SCORING_BB_PERIOD", 20))
    bb_lower = _first_available(row, [f"BB_LOWER_{scoring_bb_period}", "BB_LOWER_20"])
    bb_upper = _first_available(row, [f"BB_UPPER_{scoring_bb_period}", "BB_UPPER_20"])
    if close is not None and not pd.isna(close):
        close_f = float(close)
        if signal == 1 and bb_lower is not None and not pd.isna(bb_lower):
            band_distance = _safe_ratio(float(bb_lower) - close_f, close_f)
            if band_distance is not None:
                components["band_extreme"] = _clip01(max(0.0, band_distance) / 0.01)
        elif signal == -1 and bb_upper is not None and not pd.isna(bb_upper):
            band_distance = _safe_ratio(close_f - float(bb_upper), close_f)
            if band_distance is not None:
                components["band_extreme"] = _clip01(max(0.0, band_distance) / 0.01)

    funding_rate = row.get("funding_rate")
    if funding_rate is not None and not pd.isna(funding_rate):
        components["funding_extreme"] = _clip01(abs(float(funding_rate)) / 0.0002)

    open_interest = row.get("open_interest")
    if prev_row is not None:
        prev_oi = prev_row.get("open_interest")
        if open_interest is not None and prev_oi not in (None, 0) and not pd.isna(open_interest) and not pd.isna(prev_oi):
            oi_change = _safe_ratio(float(open_interest) - float(prev_oi), prev_oi)
            if oi_change is not None:
                components["oi_support"] = _clip01(max(0.0, oi_change) / 0.01)

    weights = {
        "volatility": 1.4,
        "trend": 1.2,
        "volume": 0.9,
        "impulse": 1.0,
        "rsi": 1.0,
        "band_extreme": 0.9,
        "funding_extreme": 0.8,
        "oi_support": 0.8,
    }
    weighted_sum = 0.0
    weight_total = 0.0
    for name, value in components.items():
        weight = weights.get(name, 1.0)
        weighted_sum += value * weight
        weight_total += weight

    score = weighted_sum / weight_total if weight_total else 0.0
    rounded_components = {name: round(value, 6) for name, value in components.items()}
    return {"score": round(_clip01(score), 6), "components": rounded_components}
