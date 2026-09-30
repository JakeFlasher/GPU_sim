#!/usr/bin/env python3
"""Regime-aware endpoint router with an explicit unknown-regime fallback.

An external active-clock observation may select the high-clock committee. If no
independent clock observation is supplied, the router chooses the neutral
native-event/frontier crosswalk and records ``regime_unknown``. NCU frequency
fields from the score campaign are never used for routing.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np

TARGETS = ("ncu_gpc_cycles", "ncu_duration_ns")


def _load(path):
    if path.is_dir():
        with (path / "performance.csv").open(newline="") as stream:
            return {row["workload"]: row for row in csv.DictReader(stream) if row.get("status") == "completed"}
    value = json.loads(path.read_text())
    rows = value.get("predictions") or value.get("rows") or []
    result = {row["workload"]: row for row in rows if row.get("workload")}
    if not result:
        raise RuntimeError(f"empty report: {path}")
    return result


def _prediction(row, target):
    value = row.get(f"{target}_predicted")
    if value is None and isinstance(row.get("endpoints"), dict):
        ep = row["endpoints"].get(target, {})
        if isinstance(ep, dict):
            value = ep.get("elastic", ep.get("reference"))
    if value is None and isinstance(row.get("predictions"), dict):
        value = row["predictions"].get(target)
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"invalid {target} source prediction")
    return value


def _clock(path, anchor):
    if path is None:
        return "unknown", None, None
    value = json.loads(path.read_text())
    declared = value.get("regime")
    clock = value.get("sm_clock_mhz", {}).get("active_median", value.get("active_clock_mhz"))
    if clock is None:
        clock = value.get("active_window_clock_mhz", {}).get("median")
    if declared in ("high", "low"):
        # The external sampler's v2 classification uses the high-clock
        # fraction plus maximum burst when the median is diluted by idle time.
        return declared, float(clock) if clock is not None else None, path
    if clock is None or not math.isfinite(float(clock)) or float(clock) <= 0:
        return "unknown", None, path
    ratio = float(clock) / anchor
    return ("high" if ratio >= 1.15 else "low"), float(clock), path


def run(args):
    started = time.perf_counter()
    regime, active_clock, clock_path = _clock(args.clock_observation, args.anchor_clock_mhz)
    score = _load(args.score_campaign)
    if regime == "high":
        if args.phase == "cold":
            gpc_path, duration_path = args.high_cold_gpc, args.high_cold_duration
        else:
            gpc_path, duration_path = args.high_warm_gpc, args.high_warm_duration
        route = "high_clock_committee"
    else:
        gpc_path = duration_path = args.neutral_source
        route = "neutral_event_frontier_unknown_or_low_regime"
    gpc, duration = _load(gpc_path), _load(duration_path)
    names = sorted(set(score) & set(gpc) & set(duration))
    if not names:
        raise RuntimeError("no source/score intersection")
    rows = []
    for name in names:
        item = {"workload": name, "status": "candidate", "coverage_status": "regime_router_covered", "regime": regime, "route": route, "active_clock_mhz": active_clock, "clock_observation": str(clock_path) if clock_path else None}
        for target, source in (("ncu_gpc_cycles", gpc[name]), ("ncu_duration_ns", duration[name])):
            pred = _prediction(source, target)
            actual = float(score[name].get(f"{target}_actual", score[name].get(target)))
            item[f"{target}_predicted"] = pred
            item[f"{target}_actual"] = actual
            item[f"{target}_ape"] = abs(pred - actual) / actual
            item[f"{target}_interval_low"] = float(source.get(f"{target}_interval_low", pred))
            item[f"{target}_interval_high"] = float(source.get(f"{target}_interval_high", pred))
        rows.append(item)
    metrics = {}
    for target in TARGETS:
        apes = [r[f"{target}_ape"] for r in rows]
        errors = [abs(r[f"{target}_predicted"] - r[f"{target}_actual"]) for r in rows]
        metrics[target] = {"count": len(apes), "mae": float(np.mean(errors)), "mape": float(np.mean(apes)), "max_ape": float(np.max(apes))}
    report = {"schema": "sm86_endpoint_regime_router_v2", "phase": args.phase, "regime": regime, "active_clock_mhz": active_clock, "clock_observation": str(clock_path) if clock_path else None, "route": route, "targets_score_only": True, "metrics": metrics, "model_seconds": time.perf_counter() - started, "predictions": rows, "limitations": ["unknown/low regime uses neutral crosswalk rather than guessing clock response", "clock observation is campaign-level and not a unified timer", "physical pipe and cache ownership remain unresolved"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("cold", "warm"), required=True)
    parser.add_argument("--score-campaign", type=Path, required=True)
    parser.add_argument("--clock-observation", type=Path)
    parser.add_argument("--anchor-clock-mhz", type=float, default=1350.0)
    parser.add_argument("--neutral-source", type=Path, required=True)
    parser.add_argument("--high-cold-gpc", type=Path, required=True)
    parser.add_argument("--high-cold-duration", type=Path, required=True)
    parser.add_argument("--high-warm-gpc", type=Path, required=True)
    parser.add_argument("--high-warm-duration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"regime": report["regime"], "route": report["route"], "metrics": report["metrics"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
