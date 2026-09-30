#!/usr/bin/env python3
"""Observation-only paired cold/warm NCU clock/phase analysis."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import time


TARGETS = ("ncu_duration_ns", "ncu_gpc_cycles")


def number(row, key):
    value = float(row[key])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid {key} for {row.get('workload')}")
    return value


def load(path):
    with (path / "performance.csv").open(newline="") as stream:
        return {row["workload"]: row for row in csv.DictReader(stream) if row.get("status") == "completed"}


def run(args):
    started = time.perf_counter()
    cold = load(args.cold_campaign)
    warm = load(args.warm_campaign)
    workloads = sorted(set(cold) & set(warm))
    if not workloads:
        raise RuntimeError("no paired cold/warm workloads")
    observations = []
    for workload in workloads:
        c, w = cold[workload], warm[workload]
        cold_freq = number(c, "gpc__cycles_elapsed.avg.per_second") * 1000.0
        warm_freq = number(w, "gpc__cycles_elapsed.avg.per_second") * 1000.0
        item = {
            "workload": workload,
            "cold_gpc_frequency_mhz_score_only": cold_freq,
            "warm_gpc_frequency_mhz_score_only": warm_freq,
            "frequency_ratio_warm_over_cold": warm_freq / cold_freq,
        }
        for target in TARGETS:
            cold_value = number(c, target)
            warm_value = number(w, target)
            item[f"{target}_warm_cold_ratio"] = warm_value / cold_value
        duration_ratio = item["ncu_duration_ns_warm_cold_ratio"]
        gpc_ratio = item["ncu_gpc_cycles_warm_cold_ratio"]
        # If NCU GPC cycles and the local NCU frequency were a complete timer,
        # this would be the implied duration response. Keep the residual as an
        # NCU-local diagnostic, never as a simulator fitting feature.
        item["duration_ratio_from_gpc_and_frequency"] = gpc_ratio / item["frequency_ratio_warm_over_cold"]
        item["duration_phase_residual"] = duration_ratio - item["duration_ratio_from_gpc_and_frequency"]
        observations.append(item)
    freq = [x["frequency_ratio_warm_over_cold"] for x in observations]
    duration = [x["ncu_duration_ns_warm_cold_ratio"] for x in observations]
    gpc = [x["ncu_gpc_cycles_warm_cold_ratio"] for x in observations]
    residual = [abs(x["duration_phase_residual"]) for x in observations]
    report = {
        "schema": "sm86_ncu_phase_clock_analysis_v1",
        "cold_campaign": str(args.cold_campaign),
        "warm_campaign": str(args.warm_campaign),
        "workload_count": len(observations),
        "ncu_labels_score_only": True,
        "metrics": {
            "frequency_ratio_warm_over_cold_median": statistics.median(freq),
            "frequency_ratio_warm_over_cold_min": min(freq),
            "frequency_ratio_warm_over_cold_max": max(freq),
            "duration_ratio_median": statistics.median(duration),
            "duration_ratio_min": min(duration),
            "duration_ratio_max": max(duration),
            "gpc_ratio_median": statistics.median(gpc),
            "gpc_ratio_min": min(gpc),
            "gpc_ratio_max": max(gpc),
            "duration_phase_residual_median_abs": statistics.median(residual),
            "duration_phase_residual_max_abs": max(residual),
        },
        "observations": observations,
        "runtime_seconds": time.perf_counter() - started,
        "interpretation": [
            "frequency/cycle/duration relation is audited inside NCU only",
            "warm/cold ratios remain workload-specific cache-phase responses",
            "NCU labels are not model features, tuning inputs, or mechanism witnesses",
            "no cache owner/read-from/DRAM-completion conclusion follows",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cold-campaign", type=Path, required=True)
    parser.add_argument("--warm-campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"workload_count": report["workload_count"], "metrics": report["metrics"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
