#!/usr/bin/env python3
"""Clock-regime adapter for endpoint projections.

The native replay and the promoted NCU projector use a 1,350 MHz anchor.  A
live campaign can run at a different boost clock even when NCU's device
attribute is unchanged.  This opt-in adapter consumes an *independent external
clock observation* and scales only the duration endpoint:

    duration_adjusted = duration_anchor * anchor_clock / active_clock

GPC cycles are left untouched because the relationship between boost clock,
issue progress, and NCU's GPC counter is not public.  Raw and adjusted
endpoints remain separate.  NCU score labels are loaded only after both
predictions are formed, for final comparison.

This is a timing-regime crosswalk, not a unified-clock proof or a mechanism
closure claim.  Every input prediction row is retained.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np


TARGETS = ("ncu_gpc_cycles", "ncu_duration_ns")


def _clock(observation: dict) -> tuple[float, float, float, float, float]:
    values = observation.get("sm_clock_mhz", {})
    active = float(values.get("active_median", 0.0) or 0.0)
    all_min = float(values.get("all_min", active) or active)
    all_max = float(values.get("all_max", active) or active)
    active_min = float(values.get("active_min", active) or active)
    active_max = float(values.get("active_max", active) or active)
    if not all(math.isfinite(value) and value > 0.0 for value in (active, all_min, all_max, active_min, active_max)):
        raise RuntimeError("clock observation has no finite positive SM-clock values")
    return active, all_min, all_max, active_min, active_max


def _metric(rows: list[dict], key: str, adjusted_key: str | None = None) -> dict[str, float] | None:
    if not rows or not all(f"{key}_actual" in row for row in rows):
        return None
    actual = np.asarray([float(row[f"{key}_actual"]) for row in rows], dtype=float)
    field = adjusted_key or f"{key}_predicted"
    prediction = np.asarray([float(row[field]) for row in rows], dtype=float)
    return {
        "mae": float(np.mean(np.abs(prediction - actual))),
        "mape": float(np.mean(np.abs(prediction - actual) / np.maximum(actual, 1.0))),
        "max_abs_error": float(np.max(np.abs(prediction - actual))),
    }


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    projection = json.loads(args.projection_report.read_text())
    observation = json.loads(args.clock_observation.read_text())
    active_clock, all_min, all_max, active_min, active_max = _clock(observation)
    if not math.isfinite(args.anchor_clock_mhz) or args.anchor_clock_mhz <= 0.0:
        raise RuntimeError("anchor clock must be finite and positive")
    scale = args.anchor_clock_mhz / active_clock
    scale_low = args.anchor_clock_mhz / active_max
    scale_high = args.anchor_clock_mhz / active_min
    rows = []
    for source in projection.get("predictions", []):
        # Copy only after reading the source prediction.  Actual labels are
        # carried for post-prediction APE reporting and never enter `scale`.
        item = {
            "workload": source.get("workload"),
            "status": source.get("status", "candidate"),
            "coverage_status": source.get("coverage_status"),
            "clock_active_mhz": active_clock,
            "clock_scale_anchor_over_active": scale,
            "clock_scale_interval_low": scale_low,
            "clock_scale_interval_high": scale_high,
            "ncu_gpc_cycles_predicted": float(source["ncu_gpc_cycles_predicted"]),
            "ncu_duration_ns_predicted": float(source["ncu_duration_ns_predicted"]),
            "ncu_duration_ns_clock_adjusted_predicted": float(source["ncu_duration_ns_predicted"]) * scale,
            "ncu_duration_ns_clock_regime_interval_low": float(source["ncu_duration_ns_predicted"]) * scale_low,
            "ncu_duration_ns_clock_regime_interval_high": float(source["ncu_duration_ns_predicted"]) * scale_high,
        }
        for target in TARGETS:
            if f"{target}_actual" in source:
                item[f"{target}_actual"] = source[f"{target}_actual"]
            if f"{target}_interval_low" in source:
                item[f"{target}_interval_low"] = source[f"{target}_interval_low"]
                item[f"{target}_interval_high"] = source[f"{target}_interval_high"]
        if "ncu_duration_ns_interval_low" in source:
            item["ncu_duration_ns_clock_adjusted_prediction_interval_low"] = float(source["ncu_duration_ns_interval_low"]) * scale
            item["ncu_duration_ns_clock_adjusted_prediction_interval_high"] = float(source["ncu_duration_ns_interval_high"]) * scale
        rows.append(item)

    report = {
        "schema": "sm86_clock_regime_projection_v1",
        "projection_report": str(args.projection_report),
        "clock_observation": str(args.clock_observation),
        "anchor_clock_mhz": args.anchor_clock_mhz,
        "active_clock_mhz": active_clock,
        "observed_clock_range_mhz": {"min": all_min, "max": all_max},
        "observed_active_clock_range_mhz": {"min": active_min, "max": active_max},
        "clock_scale_anchor_over_active": scale,
        "clock_scale_interval": {"low": scale_low, "high": scale_high},
        "model": {
            "duration": "anchor-duration × anchor-clock / independently sampled active clock",
            "gpc": "unchanged; no public clock-to-GPC equivalence assumed",
            "interval": "duration interval scaled by the same external regime factor",
            "coverage": "all source prediction rows retained",
        },
        "targets_are_score_only": True,
        "predictions": rows,
        "metrics": {
            "raw_duration": _metric(rows, "ncu_duration_ns"),
            "clock_adjusted_duration": _metric(rows, "ncu_duration_ns", "ncu_duration_ns_clock_adjusted_predicted"),
            "gpc_cycles_unchanged": _metric(rows, "ncu_gpc_cycles"),
        },
        "model_seconds": time.perf_counter() - started,
        "limitations": [
            "external clock sample is a regime observation, not a synchronized NCU/device timer",
            "active clock is summarized per campaign rather than per kernel launch",
            "GPC cycles are not rescaled because their boost-clock semantics are unobserved",
            "NCU labels are loaded only after the adjustment for comparison",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection-report", type=Path, required=True)
    parser.add_argument("--clock-observation", type=Path, required=True)
    parser.add_argument("--anchor-clock-mhz", type=float, default=1350.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "active_clock_mhz": report["active_clock_mhz"],
        "clock_scale_anchor_over_active": report["clock_scale_anchor_over_active"],
        "metrics": report["metrics"],
        "model_seconds": report["model_seconds"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
