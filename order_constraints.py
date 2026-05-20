from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
from typing import Any


def _to_decimal(value: Any, default: str = "0") -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal(default)


def _round_to_step(value: Decimal, step: Decimal, rounding: str) -> Decimal:
    if step <= 0:
        return value
    units = (value / step).to_integral_value(rounding=rounding)
    return units * step


def _format_decimal(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def get_order_constraints(instrument_info: dict[str, Any] | None) -> dict[str, Decimal]:
    lot_size_filter = {}
    if isinstance(instrument_info, dict):
        lot_size_filter = instrument_info.get("lotSizeFilter", {}) or {}

    min_qty = max(_to_decimal(lot_size_filter.get("minOrderQty")), Decimal("0"))
    qty_step = max(_to_decimal(lot_size_filter.get("qtyStep")), Decimal("0"))
    if qty_step <= 0 and min_qty > 0:
        qty_step = min_qty

    return {
        "min_qty": min_qty,
        "qty_step": qty_step,
        "max_qty": max(
            _to_decimal(lot_size_filter.get("maxMktOrderQty") or lot_size_filter.get("maxOrderQty")),
            Decimal("0"),
        ),
        "min_notional": max(_to_decimal(lot_size_filter.get("minNotionalValue")), Decimal("0")),
    }


def normalize_entry_qty(
    desired_qty: float,
    price: float,
    instrument_info: dict[str, Any] | None,
    *,
    allow_increase: bool,
) -> dict[str, Any]:
    desired_qty_dec = max(_to_decimal(desired_qty), Decimal("0"))
    price_dec = max(_to_decimal(price), Decimal("0"))
    constraints = get_order_constraints(instrument_info)
    qty_step = constraints["qty_step"]
    min_qty = constraints["min_qty"]
    min_notional = constraints["min_notional"]
    max_qty = constraints["max_qty"]

    rounded_desired = _round_to_step(desired_qty_dec, qty_step, ROUND_FLOOR)
    min_required_qty = min_qty
    if price_dec > 0 and min_notional > 0:
        min_notional_qty = min_notional / price_dec
        min_required_qty = max(
            min_required_qty,
            _round_to_step(min_notional_qty, qty_step, ROUND_CEILING),
        )

    candidate_qty = rounded_desired
    if allow_increase:
        candidate_qty = max(candidate_qty, min_required_qty)

    if candidate_qty <= 0:
        reason = "below_exchange_minimum" if min_required_qty > 0 else "non_positive_qty"
        return {
            "ok": False,
            "reason": reason,
            "qty": None,
            "qty_str": None,
            "adjusted": False,
            "min_qty": float(min_qty),
            "qty_step": float(qty_step),
            "min_required_qty": float(min_required_qty),
            "min_required_notional": float(min_required_qty * price_dec) if price_dec > 0 else None,
            "effective_notional": 0.0,
        }

    if candidate_qty < min_required_qty:
        return {
            "ok": False,
            "reason": "below_exchange_minimum",
            "qty": None,
            "qty_str": None,
            "adjusted": False,
            "min_qty": float(min_qty),
            "qty_step": float(qty_step),
            "min_required_qty": float(min_required_qty),
            "min_required_notional": float(min_required_qty * price_dec) if price_dec > 0 else None,
            "effective_notional": float(candidate_qty * price_dec) if price_dec > 0 else None,
        }

    if max_qty > 0 and candidate_qty > max_qty:
        return {
            "ok": False,
            "reason": "above_exchange_maximum",
            "qty": None,
            "qty_str": None,
            "adjusted": False,
            "min_qty": float(min_qty),
            "qty_step": float(qty_step),
            "min_required_qty": float(min_required_qty),
            "min_required_notional": float(min_required_qty * price_dec) if price_dec > 0 else None,
            "effective_notional": float(candidate_qty * price_dec) if price_dec > 0 else None,
        }

    return {
        "ok": True,
        "reason": None,
        "qty": float(candidate_qty),
        "qty_str": _format_decimal(candidate_qty),
        "adjusted": candidate_qty != desired_qty_dec,
        "min_qty": float(min_qty),
        "qty_step": float(qty_step),
        "min_required_qty": float(min_required_qty),
        "min_required_notional": float(min_required_qty * price_dec) if price_dec > 0 else None,
        "effective_notional": float(candidate_qty * price_dec) if price_dec > 0 else None,
    }


def normalize_reduce_only_qty(
    desired_qty: float,
    remaining_qty: float,
    instrument_info: dict[str, Any] | None,
) -> dict[str, Any]:
    desired_qty_dec = max(_to_decimal(desired_qty), Decimal("0"))
    remaining_qty_dec = max(_to_decimal(remaining_qty), Decimal("0"))
    constraints = get_order_constraints(instrument_info)
    qty_step = constraints["qty_step"]
    min_qty = constraints["min_qty"]

    rounded_desired = _round_to_step(desired_qty_dec, qty_step, ROUND_FLOOR)
    rounded_remaining = _round_to_step(remaining_qty_dec, qty_step, ROUND_FLOOR)
    candidate_qty = min(rounded_desired, rounded_remaining)
    promoted_to_full_close = False

    if candidate_qty <= 0 or (min_qty > 0 and candidate_qty < min_qty):
        if rounded_remaining > 0 and (min_qty <= 0 or rounded_remaining >= min_qty):
            candidate_qty = rounded_remaining
            promoted_to_full_close = True
        else:
            return {
                "ok": False,
                "reason": "reduce_only_qty_below_exchange_minimum",
                "qty": None,
                "qty_str": None,
                "adjusted": False,
                "promoted_to_full_close": False,
                "min_qty": float(min_qty),
                "qty_step": float(qty_step),
            }

    return {
        "ok": True,
        "reason": None,
        "qty": float(candidate_qty),
        "qty_str": _format_decimal(candidate_qty),
        "adjusted": candidate_qty != desired_qty_dec,
        "promoted_to_full_close": promoted_to_full_close,
        "min_qty": float(min_qty),
        "qty_step": float(qty_step),
    }


def format_price_for_order(price: float, instrument_info: dict[str, Any] | None) -> str:
    price_filter = {}
    if isinstance(instrument_info, dict):
        price_filter = instrument_info.get("priceFilter", {}) or {}

    tick_size = _to_decimal(price_filter.get("tickSize") or price_filter.get("minPrice"))
    price_dec = _to_decimal(price)
    if tick_size <= 0:
        return f"{float(price):.2f}"
    return _format_decimal(_round_to_step(price_dec, tick_size, ROUND_FLOOR))
