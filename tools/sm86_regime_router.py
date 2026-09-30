#!/usr/bin/env python3
"""Route endpoint projections by external clock regime and cache phase."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

TARGETS = ("ncu_gpc_cycles", "ncu_duration_ns")


def load_rows(path: Path, mode: str = "reference") -> dict[str, dict]:
    report = json.loads(path.read_text())
    entries = report.get("predictions")
    if not isinstance(entries, list):
        entries = report.get("rows") or []
    output = {}
    for item in entries:
        workload = item.get("workload")
        if not workload:
            continue
        if "endpoints" in item:
            values = item["endpoints"]
            if any(target not in values or mode not in values[target] for target in TARGETS):
                raise RuntimeError(f"endpoint mode {mode!r} missing for {workload}")
            output[workload] = {
                target: float(values[target][mode]) for target in TARGETS
            }
        else:
            output[workload] = {
                target: float(item[f"{target}_predicted"]) for target in TARGETS
            }
        output[workload]["status"] = item.get("status", "candidate")
        output[workload]["coverage_status"] = item.get("coverage_status")
    if not output:
        raise RuntimeError(f"no prediction rows in {path}")
    return output


def rows(campaign: Path) -> dict[str, dict[str, str]]:
    import csv
    with (campaign / "performance.csv").open(newline="") as stream:
        return {r["workload"]: r for r in csv.DictReader(stream) if r.get("status") == "completed"}


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    observation = json.loads(args.clock_observation.read_text())
    clock = observation.get("sm_clock_mhz", {}).get("active_median")
    if clock is None:
        clock = observation.get("active_clock_mhz")
    clock = float(clock)
    if not math.isfinite(clock) or clock <= 0:
        raise RuntimeError("clock observation has no active median")
    ratio = clock / args.anchor_clock_mhz
    high = ratio >= args.high_regime_threshold
    low = load_rows(args.low_report, "reference")
    high_base = load_rows(args.high_base_report, args.high_mode)
    high_large = load_rows(args.high_large_report, args.high_mode)
    score = rows(args.score_campaign)
    chosen = high_base | high_large if high else low
    source_name = (
        f"high_regime:{args.high_mode}:base/large" if high else "low_regime:frozen_reference"
    )
    output_rows = []
    for workload, score_row in sorted(score.items()):
        if workload not in chosen:
            raise RuntimeError(f"router source lacks {workload}")
        source = chosen[workload]
        item = {
            "workload": workload,
            "status": source.get("status", "candidate"),
            "coverage_status": source.get("coverage_status"),
            "route": source_name,
            "observed_active_clock_mhz": clock,
            "clock_ratio_to_anchor": ratio,
            "ncu_gpc_cycles_predicted": source["ncu_gpc_cycles"],
            "ncu_duration_ns_predicted": source["ncu_duration_ns"],
        }
        for target in TARGETS:
            actual = float(score_row[target])
            prediction = source[target]
            item[f"{target}_actual"] = actual
            item[f"{target}_ape"] = abs(prediction - actual) / actual
            item[f"{target}_abs_error"] = abs(prediction - actual)
        output_rows.append(item)
    metrics = {}
    for target in TARGETS:
        actual = np.asarray([r[f"{target}_actual"] for r in output_rows])
        prediction = np.asarray([r[f"{target}_predicted"] for r in output_rows])
        metrics[target] = {
            "mae": float(np.mean(abs(prediction - actual))),
            "mape": float(np.mean(abs(prediction - actual) / actual)),
            "max_abs_error": float(np.max(abs(prediction - actual))),
        }
    report = {
        "schema": "sm86_regime_router_v1",
        "clock_observation": str(args.clock_observation),
        "anchor_clock_mhz": args.anchor_clock_mhz,
        "high_regime_threshold": args.high_regime_threshold,
        "phase": args.phase,
        "high_mode": args.high_mode,
        "selected_high_regime": high,
        "source_route": source_name,
        "score_campaign": str(args.score_campaign),
        "targets_score_only": True,
        "metrics": metrics,
        "predictions": output_rows,
        "model_seconds": time.perf_counter() - started,
        "limitations": [
            "router selects a committee source, not a physical mechanism",
            "clock regime is campaign-level; launch-local variation remains unknown",
            "high-regime sources are related-family holdouts",
            "cache ownership/read-from and unified timing remain unresolved",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock-observation", type=Path, required=True)
    parser.add_argument("--score-campaign", type=Path, required=True)
    parser.add_argument("--low-report", type=Path, required=True)
    parser.add_argument("--high-base-report", type=Path, required=True)
    parser.add_argument("--high-large-report", type=Path, required=True)
    parser.add_argument("--phase", choices=("cold", "warm"), required=True)
    parser.add_argument("--high-mode", choices=("elastic", "reference"), required=True)
    parser.add_argument("--anchor-clock-mhz", type=float, default=1350.0)
    parser.add_argument("--high-regime-threshold", type=float, default=1.15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"selected_high_regime": report["selected_high_regime"], "route": report["source_route"], "metrics": report["metrics"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
