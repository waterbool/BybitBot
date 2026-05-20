from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from typing import Any

from flask import Flask, jsonify, send_from_directory, request
from pybit.unified_trading import HTTP

from config import settings
from live.edge_snapshot import load_edge_snapshot, snapshot_age_minutes


app = Flask(__name__, static_folder="static", template_folder="static")
logger = logging.getLogger(__name__)

BOT_SERVICE = os.getenv("BOT_MONITOR_SERVICE", "bybit-bot-demo.service")
REFRESH_TIMER = os.getenv("BOT_MONITOR_REFRESH_TIMER", "bybit-bot-refresh.timer")
REFRESH_SERVICE = os.getenv("BOT_MONITOR_REFRESH_SERVICE", "bybit-bot-refresh.service")


def _run_command(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or f"exit code {result.returncode}"
        raise RuntimeError(f"{' '.join(command)} failed: {detail}")
    return result.stdout


def _systemd_props(unit: str, props: list[str]) -> dict[str, str]:
    command = ["systemctl", "show", unit, "--no-pager"]
    for prop in props:
        command.extend(["-p", prop])
    raw = _run_command(command)
    parsed: dict[str, str] = {}
    for line in raw.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key] = value
    return parsed


def _journal_tail(unit: str, count: int = 80) -> list[str]:
    raw = _run_command(
        ["journalctl", "-u", unit, "-n", str(count), "--no-pager", "-o", "cat"]
    )
    return [line for line in raw.splitlines() if line.strip()]


def _parse_runtime_line(lines: list[str]) -> dict[str, Any] | None:
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if "status" in payload and "selector_state" in payload:
            return payload
    return None


def _to_iso_from_systemd(value: str) -> str | None:
    if not value or value in {"n/a", "0"}:
        return None
    try:
        parsed = datetime.strptime(value, "%a %Y-%m-%d %H:%M:%S %Z")
        return parsed.replace(tzinfo=timezone.utc).isoformat()
    except ValueError:
        return value


def _snapshot_summary() -> dict[str, Any]:
    snapshot = load_edge_snapshot(settings.LIVE_SELECTOR_EDGE_SNAPSHOT_PATH)
    if not snapshot:
        return {"exists": False}
    return {
        "exists": True,
        "path": settings.LIVE_SELECTOR_EDGE_SNAPSHOT_PATH,
        "built_at": snapshot.get("built_at"),
        "age_minutes": snapshot_age_minutes(snapshot),
        "symbols": snapshot.get("symbols", []),
        "strategies": snapshot.get("strategies", []),
        "rows": snapshot.get("rows", []),
    }


def _account_summary() -> dict[str, Any]:
    session = HTTP(
        testnet=settings.BYBIT_TESTNET,
        demo=getattr(settings, "BYBIT_DEMO", False),
        api_key=settings.BYBIT_API_KEY,
        api_secret=settings.BYBIT_API_SECRET,
    )

    open_orders_resp = session.get_open_orders(category="linear", settleCoin="USDT")
    positions_resp = session.get_positions(category="linear", settleCoin="USDT")

    open_orders = open_orders_resp.get("result", {}).get("list", []) or []
    positions = positions_resp.get("result", {}).get("list", []) or []
    active_positions = [
        pos for pos in positions if float(pos.get("size") or 0.0) != 0.0
    ]

    summary = {
        "environment": {
            "demo": getattr(settings, "BYBIT_DEMO", False),
            "testnet": settings.BYBIT_TESTNET,
        },
        "open_orders_count": len(open_orders),
        "active_positions_count": len(active_positions),
        "open_orders": [
            {
                "symbol": item.get("symbol"),
                "side": item.get("side"),
                "qty": item.get("qty"),
                "order_status": item.get("orderStatus"),
                "order_type": item.get("orderType"),
                "price": item.get("price"),
                "created_time": item.get("createdTime"),
            }
            for item in open_orders[:20]
        ],
        "active_positions": [
            {
                "symbol": item.get("symbol"),
                "side": item.get("side"),
                "size": item.get("size"),
                "avg_price": item.get("avgPrice"),
                "mark_price": item.get("markPrice"),
                "unrealised_pnl": item.get("unrealisedPnl"),
            }
            for item in active_positions[:20]
        ],
    }

    try:
        balance_resp = session.get_wallet_balance(accountType="UNIFIED")
        balance_items = balance_resp.get("result", {}).get("list", []) or []
        summary["wallet"] = balance_items[:1]
    except Exception as exc:
        summary["wallet_error"] = str(exc)

    return summary


def _service_summary() -> dict[str, Any]:
    props = _systemd_props(
        BOT_SERVICE,
        [
            "ActiveState",
            "SubState",
            "MainPID",
            "ExecMainStartTimestamp",
            "MemoryCurrent",
            "TasksCurrent",
        ],
    )
    timer_props = _systemd_props(
        REFRESH_TIMER,
        ["ActiveState", "SubState", "NextElapseUSecRealtime", "LastTriggerUSecRealtime"],
    )
    refresh_props = _systemd_props(
        REFRESH_SERVICE,
        ["ActiveState", "SubState", "ExecMainStartTimestamp", "ExecMainExitTimestamp", "Result"],
    )

    bot_logs = _journal_tail(BOT_SERVICE, count=120)
    refresh_logs = _journal_tail(REFRESH_SERVICE, count=40)

    return {
        "bot_service": {
            "unit": BOT_SERVICE,
            "active_state": props.get("ActiveState"),
            "sub_state": props.get("SubState"),
            "main_pid": props.get("MainPID"),
            "started_at": _to_iso_from_systemd(props.get("ExecMainStartTimestamp", "")),
            "memory_current": props.get("MemoryCurrent"),
            "tasks_current": props.get("TasksCurrent"),
        },
        "refresh_timer": {
            "unit": REFRESH_TIMER,
            "active_state": timer_props.get("ActiveState"),
            "sub_state": timer_props.get("SubState"),
            "next_run": _to_iso_from_systemd(timer_props.get("NextElapseUSecRealtime", "")),
            "last_trigger": _to_iso_from_systemd(timer_props.get("LastTriggerUSecRealtime", "")),
        },
        "refresh_service": {
            "unit": REFRESH_SERVICE,
            "active_state": refresh_props.get("ActiveState"),
            "sub_state": refresh_props.get("SubState"),
            "started_at": _to_iso_from_systemd(refresh_props.get("ExecMainStartTimestamp", "")),
            "finished_at": _to_iso_from_systemd(refresh_props.get("ExecMainExitTimestamp", "")),
            "result": refresh_props.get("Result"),
        },
        "runtime_status": _parse_runtime_line(bot_logs),
        "recent_bot_logs": bot_logs[-80:],
        "recent_refresh_logs": refresh_logs[-40:],
    }


@app.route("/")
def index():
    return send_from_directory("static", "monitor.html")


@app.route("/api/monitor/summary")
def monitor_summary():
    try:
        payload = {
            "server_time": datetime.now(timezone.utc).isoformat(),
            "service": _service_summary(),
            "snapshot": _snapshot_summary(),
            "account": _account_summary(),
        }
        return jsonify({"success": True, "data": payload})
    except Exception as exc:
        logger.exception("Failed to build monitor summary")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/monitor/logs")
def monitor_logs():
    count = request.args.get("count", default=120, type=int)
    count = max(20, min(count, 500))
    try:
        return jsonify(
            {
                "success": True,
                "bot_logs": _journal_tail(BOT_SERVICE, count=count),
                "refresh_logs": _journal_tail(REFRESH_SERVICE, count=max(20, count // 2)),
            }
        )
    except Exception as exc:
        logger.exception("Failed to fetch logs")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/health")
def health():
    try:
        props = _systemd_props(BOT_SERVICE, ["ActiveState", "SubState"])
        return jsonify(
            {
                "ok": props.get("ActiveState") == "active",
                "active_state": props.get("ActiveState"),
                "sub_state": props.get("SubState"),
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    host = os.getenv("MONITOR_HOST", "127.0.0.1")
    port = int(os.getenv("MONITOR_PORT", "5002"))
    app.run(host=host, port=port, debug=False)
