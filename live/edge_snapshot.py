from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from backtest.multi_asset import DEFAULT_STRATEGIES, normalize_symbols, run_multi_symbol_backtests
from config import settings


def build_edge_snapshot(
    symbols: list[str],
    strategy_names: list[str] | None = None,
    lookback_days: int | None = None,
    initial_balance: float | None = None,
    enable_ml: bool = False,
) -> dict:
    symbols = normalize_symbols(symbols)
    strategies = strategy_names or list(DEFAULT_STRATEGIES)
    lookback_days = int(lookback_days if lookback_days is not None else settings.LIVE_SELECTOR_EDGE_LOOKBACK_DAYS)
    initial_balance = float(initial_balance if initial_balance is not None else settings.INITIAL_BALANCE)

    results, _ = run_multi_symbol_backtests(
        symbols=symbols,
        strategy_names=strategies,
        lookback_days=lookback_days,
        initial_balance=initial_balance,
        enable_ml=enable_ml,
    )

    rows = []
    for result in results:
        if result.get("status") != "ok":
            continue
        rows.append(
            {
                "symbol": result["symbol"],
                "strategy": result["strategy"],
                "edge_score": float(result.get("edge_score", 0.0) or 0.0),
                "avg_signal_score": float(result.get("avg_signal_score", 0.0) or 0.0),
                "net_pnl": float(result.get("net_pnl", 0.0) or 0.0),
                "profit_factor": result.get("profit_factor", 0.0),
                "max_drawdown": float(result.get("max_drawdown", 0.0) or 0.0),
                "win_rate_pct": float(result.get("win_rate_pct", 0.0) or 0.0),
                "positions_closed": int(result.get("positions_closed", 0) or 0),
                "bars": int(result.get("bars", 0) or 0),
                "from": result.get("from"),
                "to": result.get("to"),
            }
        )

    return {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "lookback_days": lookback_days,
        "symbols": symbols,
        "strategies": strategies,
        "with_ml": bool(enable_ml),
        "rows": rows,
    }


def save_edge_snapshot(snapshot: dict, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return target


def load_edge_snapshot(path: str | Path) -> dict | None:
    target = Path(path)
    if not target.exists():
        return None
    return json.loads(target.read_text(encoding="utf-8"))


def build_edge_lookup(snapshot: dict | None) -> dict[tuple[str, str], dict]:
    if not snapshot:
        return {}
    lookup: dict[tuple[str, str], dict] = {}
    for row in snapshot.get("rows", []):
        symbol = str(row.get("symbol", "")).upper()
        strategy = str(row.get("strategy", ""))
        if symbol and strategy:
            lookup[(symbol, strategy)] = row
    return lookup


def snapshot_age_minutes(snapshot: dict | None, now: datetime | None = None) -> float | None:
    if not snapshot or not snapshot.get("built_at"):
        return None
    built_at = datetime.fromisoformat(snapshot["built_at"])
    if built_at.tzinfo is None:
        built_at = built_at.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return (now - built_at).total_seconds() / 60.0


def snapshot_is_fresh(snapshot: dict | None, max_age_minutes: int, now: datetime | None = None) -> bool:
    age = snapshot_age_minutes(snapshot, now=now)
    if age is None:
        return False
    return age <= float(max_age_minutes)
