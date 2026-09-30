#!/usr/bin/env python3
"""Scenario interval for same-regime warm cache response.

The model predicts a warm endpoint from a cold endpoint projection and an
independent warm/cold response distribution.  It intentionally emits a
scenario interval, not a cache-owner/read-from claim.  Calibration rows are
from a disjoint family in the same clock/cache regime; score NCU labels are
loaded only after the point and interval are formed.
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


def rows(path: Path) -> dict[str, dict[str, str]]:
    with (path / "performance.csv").open(newline="") as stream:
        return {r["workload"]: r for r in csv.DictReader(stream) if r.get("status") == "completed"}


def positive(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid {key} for {row.get('workload')}")
    return value


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    cold = rows(args.cold_calibration)
    warm = rows(args.warm_calibration)
    score = rows(args.score_campaign)
    calibration = sorted(
        workload for workload in cold
        if workload in warm and workload.split("__", 1)[0] == args.calibration_family
    )
    if not calibration:
        raise RuntimeError("no paired calibration rows for requested family")
    reference = json.loads(args.cold_reference.read_text())
    reference_rows = {item["workload"]: item for item in reference.get("predictions", [])}
    score_workloads = sorted(
        workload for workload in score if workload.split("__", 1)[0] == args.holdout_family
    )
    if not score_workloads:
        raise RuntimeError("empty score family")
    if any(workload in calibration for workload in score_workloads):
        raise RuntimeError("calibration and score workloads overlap")

    ratios: dict[str, np.ndarray] = {}
    ratio_center: dict[str, float] = {}
    ratio_residual_q90: dict[str, float] = {}
    for target in TARGETS:
        values = np.asarray([
            positive(warm[w], target) / positive(cold[w], target)
            for w in calibration
        ], dtype=float)
        center = float(np.median(values))
        ratios[target] = values
        ratio_center[target] = center
        ratio_residual_q90[target] = float(np.quantile(np.abs(values - center), 0.90))

    output_rows = []
    for workload in score_workloads:
        if workload not in reference_rows:
            raise RuntimeError(f"cold reference lacks score workload: {workload}")
        source = reference_rows[workload]
        item = {
            "workload": workload,
            "status": "candidate",
            "coverage_status": "same_regime_warm_scenario_covered",
            "calibration_family": args.calibration_family,
            "scenario_interval_semantics": "warm/cold response cone plus calibration residual; no owner/read-from claim",
        }
        for target in TARGETS:
            cold_pred = float(source[f"{target}_predicted"])
            center = ratio_center[target]
            q90 = ratio_residual_q90[target]
            # The extrema form the physical response cone observed in the
            # independent calibration family; q90 residual widens it without
            # treating a finite sample as a hard cache-state boundary.
            low_ratio = max(0.0, float(np.min(ratios[target])) - q90)
            high_ratio = float(np.max(ratios[target])) + q90
            item[f"{target}_cold_reference_predicted"] = cold_pred
            item[f"{target}_warm_predicted"] = cold_pred * center
            item[f"{target}_warm_interval_low"] = cold_pred * low_ratio
            item[f"{target}_warm_interval_high"] = cold_pred * high_ratio
            actual = positive(score[workload], target)
            item[f"{target}_actual"] = actual
            item[f"{target}_ape"] = abs(item[f"{target}_warm_predicted"] - actual) / actual
            item[f"{target}_interval_contains_actual"] = (
                item[f"{target}_warm_interval_low"] <= actual <= item[f"{target}_warm_interval_high"]
            )
        output_rows.append(item)

    metrics = {}
    for target in TARGETS:
        actual = np.asarray([r[f"{target}_actual"] for r in output_rows])
        prediction = np.asarray([r[f"{target}_warm_predicted"] for r in output_rows])
        metrics[target] = {
            "mape": float(np.mean(np.abs(prediction - actual) / actual)),
            "interval_coverage": float(np.mean([
                r[f"{target}_interval_contains_actual"] for r in output_rows
            ])),
            "interval_width_median": float(np.median([
                r[f"{target}_warm_interval_high"] - r[f"{target}_warm_interval_low"]
                for r in output_rows
            ])),
        }
    report = {
        "schema": "sm86_warm_cache_scenario_interval_v1",
        "cold_calibration": str(args.cold_calibration),
        "warm_calibration": str(args.warm_calibration),
        "cold_reference": str(args.cold_reference),
        "score_campaign": str(args.score_campaign),
        "calibration_family": args.calibration_family,
        "holdout_family": args.holdout_family,
        "calibration_workloads": calibration,
        "target_labels_score_only": True,
        "ratio_centers": ratio_center,
        "ratio_residual_q90": ratio_residual_q90,
        "metrics": metrics,
        "predictions": output_rows,
        "model_seconds": time.perf_counter() - started,
        "limitations": [
            "warm/cold ratios are aggregate endpoint responses, not cache ownership/read-from semantics",
            "calibration and score families are related and small",
            "clock regime and cache-control policy must match calibration",
            "interval is a conservative scenario envelope, not confidence probability",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cold-calibration", type=Path, required=True)
    parser.add_argument("--warm-calibration", type=Path, required=True)
    parser.add_argument("--cold-reference", type=Path, required=True)
    parser.add_argument("--score-campaign", type=Path, required=True)
    parser.add_argument("--calibration-family", choices=("base", "large"), required=True)
    parser.add_argument("--holdout-family", choices=("base", "large"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"metrics": report["metrics"], "model_seconds": report["model_seconds"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
