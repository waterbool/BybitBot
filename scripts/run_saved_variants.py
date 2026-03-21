from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.search_robust_configs import _build_frame_cache, _evaluate_strategy_trial, _window_schedule


def _load_presets(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run saved strategy variants.")
    parser.add_argument(
        "--presets",
        default=str(BASE_DIR / "config" / "saved_strategy_variants.json"),
        help="Path to the saved variants JSON.",
    )
    parser.add_argument(
        "--summary-path",
        default=str(BASE_DIR / "reports" / "saved_variants_rerun_summary.json"),
        help="Where to save the rerun summary JSON.",
    )
    parser.add_argument(
        "--names",
        default="",
        help="Optional comma-separated subset of preset names.",
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.WARNING)
    presets = _load_presets(args.presets)
    selected_names = {part.strip() for part in args.names.split(",") if part.strip()}
    variants = presets.get("variants", [])
    if selected_names:
        variants = [variant for variant in variants if variant.get("name") in selected_names]

    symbols = presets.get("symbols", ["ETHUSDT", "BTCUSDT", "SOLUSDT"])
    lookback_days = int(presets.get("lookback_days", 90))
    window_days = int(presets.get("window_days", 21))
    window_count = int(presets.get("window_count", 20))

    frame_cache = _build_frame_cache(symbols=symbols, lookback_days=lookback_days)
    reference_index = frame_cache[("ema_crossover_baseline", symbols[0])].index
    windows = _window_schedule(reference_index, count=window_count, window_days=window_days)

    rerun_results = []
    for variant in variants:
        rerun_results.append(
            {
                "name": variant["name"],
                "trial_id": variant["trial_id"],
                "strategy": variant["strategy"],
                "notes": variant.get("notes"),
                "saved_search_result": variant.get("search_result", {}),
                "rerun_result": _evaluate_strategy_trial(
                    strategy_name=variant["strategy"],
                    params=variant["params"],
                    symbols=symbols,
                    frame_cache=frame_cache,
                    windows=windows,
                    initial_balance=1000.0,
                ),
            }
        )

    report = {
        "presets_path": str(args.presets),
        "symbols": symbols,
        "lookback_days": lookback_days,
        "window_days": window_days,
        "window_count": window_count,
        "results": rerun_results,
    }
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nSaved rerun summary to {summary_path}")


if __name__ == "__main__":
    main()
