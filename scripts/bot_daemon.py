"""
bot_daemon.py — autonomous self-healing daemon for the Bybit trading bot.

Responsibilities:
  1. Keep the bot process alive (auto-restart on crash)
  2. Log a compact status line every N minutes

The edge snapshot is rebuilt once on each bot start (run_demo_live.py handles it).
Market data is always fetched live from the Bybit API — no CSV files or periodic
refresh timers needed.

Usage:
    python scripts/bot_daemon.py [--symbols ...] [--strategies ...] [--fixed-size N]
    python scripts/bot_daemon.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import tls_compat  # noqa: E402  — apply TLS-1.2 patch before any pybit import

from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [daemon] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Tunables ──────────────────────────────────────────────────────────────────
HEALTH_CHECK_SECS   = int(os.getenv("HEALTH_CHECK_SECS",   "60"))   # how often daemon polls the process
STATUS_LOG_INTERVAL = int(os.getenv("STATUS_LOG_INTERVAL", "300"))  # compact status every N secs

LOG_FILE = BASE_DIR / "logs" / "bot_live.log"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Autonomous Bybit bot daemon")
    p.add_argument("--symbols",      default="ETHUSDT,BTCUSDT,SOLUSDT")
    p.add_argument("--strategies",   default="mtf_trend_pullback,funding_extreme_reversal")
    p.add_argument("--fixed-size",   type=float, default=None)
    p.add_argument("--scan-interval", type=int, default=15)
    p.add_argument("--skip-snapshot", action="store_true",
                   help="Pass --skip-snapshot-build to the bot on every restart "
                        "(useful when you want fast restarts and the snapshot is already fresh).")
    return p.parse_args()


# ── Bot process management ────────────────────────────────────────────────────

def _build_bot_cmd(args: argparse.Namespace, first_start: bool) -> list[str]:
    cmd = [
        sys.executable,
        str(BASE_DIR / "scripts" / "run_demo_live.py"),
        "--symbols",  args.symbols,
        "--strategies", args.strategies,
        "--scan-interval-seconds", str(args.scan_interval),
        "--status-interval-seconds", "60",
    ]
    if args.fixed_size is not None:
        cmd += ["--fixed-size", str(args.fixed_size)]
    # Rebuild snapshot on first start; skip on crash-restarts (snapshot is still fresh)
    if not first_start or args.skip_snapshot:
        cmd.append("--skip-snapshot-build")
    return cmd


def start_bot(args: argparse.Namespace, first_start: bool = False) -> subprocess.Popen:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(LOG_FILE, "a")
    cmd = _build_bot_cmd(args, first_start)
    logger.info("Starting bot%s: %s",
                " (will rebuild edge snapshot)" if first_start and not args.skip_snapshot else " (skip snapshot)",
                " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"   # flush stdout immediately — prevents status JSON from stalling
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=log_fh,
        cwd=str(BASE_DIR),
        env=env,
    )
    logger.info("Bot started with PID %s", proc.pid)
    return proc


def bot_is_alive(proc: subprocess.Popen | None) -> bool:
    return proc is not None and proc.poll() is None


# ── Status summary ────────────────────────────────────────────────────────────

def _tail_status(n: int = 5) -> list[dict]:
    """Return last N valid JSON status lines from the bot log."""
    if not LOG_FILE.exists():
        return []
    results = []
    try:
        lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in reversed(lines):
            line = line.strip()
            if line.startswith('{"timestamp"'):
                try:
                    results.append(json.loads(line))
                    if len(results) >= n:
                        break
                except Exception:
                    pass
    except Exception:
        pass
    return list(reversed(results))


def log_status(proc: subprocess.Popen | None) -> None:
    statuses = _tail_status(1)
    if statuses:
        s = statuses[0]
        ss = s.get("selector_state", {})
        logger.info(
            "STATUS  alive=%-5s trades=%s pnl=%.2f position=%s skip=%s",
            bot_is_alive(proc),
            s.get("total_trades", "?"),
            float(s.get("total_pnl", 0)),
            s.get("current_position") or "none",
            ss.get("last_skip_reason") or "-",
        )
    else:
        logger.info("STATUS  alive=%s  (no status lines yet)", bot_is_alive(proc))


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # Graceful shutdown on SIGTERM / SIGINT
    _shutdown = {"flag": False}
    def _handle_signal(sig, frame):
        logger.info("Received signal %s — shutting down…", sig)
        _shutdown["flag"] = True
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT,  _handle_signal)

    bot_proc: subprocess.Popen | None = None
    last_status_log = 0.0
    first_start = True   # first launch → rebuild snapshot; crash restarts → skip

    logger.info("Daemon started. symbols=%s strategies=%s", args.symbols, args.strategies)

    while not _shutdown["flag"]:
        now = time.time()

        # ── 1. Restart bot if not running ────────────────────────────────────
        if not bot_is_alive(bot_proc):
            bot_proc = start_bot(args, first_start=first_start)
            first_start = False
            time.sleep(5)  # give it a moment to start

        # ── 2. Periodic status log ────────────────────────────────────────────
        if now - last_status_log >= STATUS_LOG_INTERVAL:
            log_status(bot_proc)
            last_status_log = now

        time.sleep(HEALTH_CHECK_SECS)

    # ── Shutdown ──────────────────────────────────────────────────────────────
    if bot_is_alive(bot_proc):
        logger.info("Terminating bot (PID %s)…", bot_proc.pid)
        bot_proc.terminate()
        try:
            bot_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            bot_proc.kill()
    logger.info("Daemon exited cleanly.")


if __name__ == "__main__":
    main()
