from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def _is_placeholder(value: str | None) -> bool:
    if not value:
        return True
    normalized = str(value).strip()
    if not normalized:
        return True
    return normalized.upper() in {"YOUR_API_KEY_HERE", "YOUR_API_SECRET_HERE"}


def _set_demo_env(args: argparse.Namespace) -> None:
    os.environ["BYBIT_TESTNET"] = "false"
    os.environ["BYBIT_DEMO"] = "true"
    os.environ["DRY_RUN"] = "false"
    os.environ["LIVE_SELECTOR_ENABLED"] = "true"
    os.environ["LIVE_SELECTOR_EXECUTION_MODE"] = "live"

    if args.api_key:
        os.environ["BYBIT_API_KEY"] = args.api_key
    if args.api_secret:
        os.environ["BYBIT_API_SECRET"] = args.api_secret
    if args.symbols:
        os.environ["LIVE_SELECTOR_SYMBOLS"] = args.symbols
    if args.strategies:
        os.environ["LIVE_SELECTOR_STRATEGIES"] = args.strategies
    if args.fixed_size is not None:
        os.environ["FIXED_USDT_SIZE"] = str(args.fixed_size)
    if args.snapshot_path:
        os.environ["LIVE_SELECTOR_EDGE_SNAPSHOT_PATH"] = args.snapshot_path
    if args.edge_lookback_days is not None:
        os.environ["LIVE_SELECTOR_EDGE_LOOKBACK_DAYS"] = str(args.edge_lookback_days)
    if args.scan_interval_seconds is not None:
        os.environ["LIVE_SELECTOR_SCAN_INTERVAL_SECONDS"] = str(args.scan_interval_seconds)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the multi-symbol live selector against Bybit demo/testnet."
    )
    parser.add_argument("--api-key", default="", help="Optional Bybit testnet API key. Falls back to env/config.")
    parser.add_argument("--api-secret", default="", help="Optional Bybit testnet API secret. Falls back to env/config.")
    parser.add_argument("--symbols", default="ETHUSDT,BTCUSDT,SOLUSDT", help="Comma-separated selector symbols.")
    parser.add_argument(
        "--strategies",
        default="mtf_trend_pullback,funding_extreme_reversal",
        help="Comma-separated selector strategies.",
    )
    parser.add_argument("--fixed-size", type=float, default=2.0, help="Fixed USDT order size per entry.")
    parser.add_argument(
        "--snapshot-path",
        default=str(BASE_DIR / "reports" / "live_edge_snapshot.json"),
        help="Path for the frozen edge snapshot JSON.",
    )
    parser.add_argument("--edge-lookback-days", type=int, default=90, help="Historical lookback for the edge snapshot.")
    parser.add_argument(
        "--scan-interval-seconds",
        type=int,
        default=30,
        help="Delay between live selector scans.",
    )
    parser.add_argument(
        "--skip-snapshot-build",
        action="store_true",
        help="Use the existing snapshot file instead of rebuilding it before launch.",
    )
    parser.add_argument(
        "--status-interval-seconds",
        type=int,
        default=30,
        help="How often to print compact runtime status to stdout.",
    )
    return parser.parse_args()


def _validate_demo_runtime(settings) -> None:
    if not getattr(settings, "BYBIT_DEMO", False):
        raise SystemExit("Demo launcher expected BYBIT_DEMO=true.")
    if settings.DRY_RUN:
        raise SystemExit("Demo launcher expected DRY_RUN=false, but paper mode is still active.")
    if not settings.LIVE_SELECTOR_ENABLED:
        raise SystemExit("Demo launcher expected LIVE_SELECTOR_ENABLED=true.")
    if settings.LIVE_SELECTOR_EXECUTION_MODE != "live":
        raise SystemExit("Demo launcher expected LIVE_SELECTOR_EXECUTION_MODE=live.")
    if _is_placeholder(settings.BYBIT_API_KEY) or _is_placeholder(settings.BYBIT_API_SECRET):
        raise SystemExit(
            "Bybit testnet API keys are missing. Pass --api-key/--api-secret or set BYBIT_API_KEY/BYBIT_API_SECRET."
        )


def _runtime_summary(settings) -> dict:
    return {
        "testnet": settings.BYBIT_TESTNET,
        "demo": getattr(settings, "BYBIT_DEMO", False),
        "dry_run": settings.DRY_RUN,
        "selector_enabled": settings.LIVE_SELECTOR_ENABLED,
        "selector_mode": settings.LIVE_SELECTOR_EXECUTION_MODE,
        "symbols": settings.LIVE_SELECTOR_SYMBOLS,
        "strategies": settings.LIVE_SELECTOR_STRATEGIES,
        "fixed_usdt_size": settings.FIXED_USDT_SIZE,
        "edge_snapshot_path": settings.LIVE_SELECTOR_EDGE_SNAPSHOT_PATH,
        "edge_lookback_days": settings.LIVE_SELECTOR_EDGE_LOOKBACK_DAYS,
        "scan_interval_seconds": settings.LIVE_SELECTOR_SCAN_INTERVAL_SECONDS,
    }


def main() -> int:
    args = _parse_args()
    _set_demo_env(args)

    from bot_controller import BotController
    from config import settings
    from live.edge_snapshot import build_edge_snapshot, save_edge_snapshot

    _validate_demo_runtime(settings)

    if not args.skip_snapshot_build:
        snapshot = build_edge_snapshot(
            symbols=settings.LIVE_SELECTOR_SYMBOLS,
            strategy_names=settings.LIVE_SELECTOR_STRATEGIES,
            lookback_days=settings.LIVE_SELECTOR_EDGE_LOOKBACK_DAYS,
            enable_ml=settings.LIVE_SELECTOR_USE_ML,
        )
        snapshot_path = save_edge_snapshot(snapshot, settings.LIVE_SELECTOR_EDGE_SNAPSHOT_PATH)
        print(f"Saved edge snapshot to {snapshot_path}")

    print(json.dumps({"runtime": _runtime_summary(settings)}, indent=2, ensure_ascii=False, default=str))

    controller = BotController()
    start_result = controller.start_trading()
    print(json.dumps({"start_result": start_result}, indent=2, ensure_ascii=False, default=str))
    if not start_result.get("success"):
        return 1

    should_stop = False

    def _handle_signal(_signum, _frame):
        nonlocal should_stop
        should_stop = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while not should_stop:
            status = controller.get_status()
            compact = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": status.get("status"),
                "total_trades": status.get("stats", {}).get("total_trades"),
                "daily_trades": status.get("stats", {}).get("daily_trades"),
                "total_pnl": status.get("stats", {}).get("total_pnl"),
                "current_position": status.get("current_position"),
                "selector_state": status.get("selector_state"),
            }
            print(json.dumps(compact, ensure_ascii=False, default=str))
            time.sleep(max(5, int(args.status_interval_seconds)))
    finally:
        stop_result = controller.stop_trading()
        print(json.dumps({"stop_result": stop_result}, indent=2, ensure_ascii=False, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
