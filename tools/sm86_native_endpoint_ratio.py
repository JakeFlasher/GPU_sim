#!/usr/bin/env python3
"""Project NCU endpoints from an independently observed native CUDA-event time.

This is an endpoint crosswalk, not a timing-mechanism simulator.  It learns
NCU/native ratios from one calibration family and applies them to a disjoint
family in the same campaign/clock/cache regime.  Replay frontiers are retained
as metadata but are not required to claim that CUDA events and NCU share a
clock.  NCU labels are loaded only after prediction for score reporting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np


TARGETS = ("ncu_duration_ns", "ncu_gpc_cycles")


def value(row: dict[str, str], key: str) -> float:
    x = float(row.get(key, "nan"))
    if not math.isfinite(x) or x <= 0:
        raise ValueError(f"non-positive/non-finite {key} in {row.get('workload')}")
    return x


def load(campaign: Path, family: str) -> list[dict[str, str]]:
    with (campaign / "performance.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [r for r in rows if r.get("status") == "completed"
            and r.get("workload", "").split("__", 1)[0] == family]


def features(row: dict[str, str]) -> np.ndarray:
    event_ns = value(row, "native_mean_ms") * 1_000_000.0
    return np.asarray([math.log(event_ns)], dtype=float)


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    scale = np.sqrt(np.mean(x * x, axis=0))
    scale[0] = 1.0
    scale[scale == 0.0] = 1.0
    xs = x / scale
    beta = np.linalg.solve(xs.T @ xs + ridge * np.eye(xs.shape[1]), xs.T @ y)
    return beta, scale


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    train = load(args.train_campaign, args.calibration_family)
    score = load(args.score_campaign, args.holdout_family)
    if not train or not score:
        raise RuntimeError("empty endpoint-ratio calibration or score family")
    train_names = {r["workload"] for r in train}
    if train_names & {r["workload"] for r in score}:
        raise RuntimeError("calibration and score workloads overlap")
    x_train = np.asarray([[1.0, *features(r)] for r in train], dtype=float)
    x_score = np.asarray([[1.0, *features(r)] for r in score], dtype=float)
    native_score = np.asarray([value(r, "native_mean_ms") * 1_000_000.0 for r in score])
    predictions = {}
    intervals = {}
    coefficients = {}
    residual_q90 = {}
    for target in TARGETS:
        y = np.asarray([value(r, target) for r in train], dtype=float)
        native_train = np.asarray([value(r, "native_mean_ms") * 1_000_000.0 for r in train])
        ratios = np.log(y / native_train)
        beta, scale = fit_ridge(x_train, ratios, args.ridge)
        train_ratio = np.exp((x_train / scale) @ beta)
        score_ratio = np.exp((x_score / scale) @ beta)
        train_pred = native_train * train_ratio
        score_pred = native_score * score_ratio
        residual_q90[target] = float(np.quantile(np.abs(train_pred - y), 0.90))
        predictions[target] = score_pred
        intervals[target] = (
            np.maximum(score_pred - residual_q90[target], 0.0),
            score_pred + residual_q90[target],
        )
        coefficients[target] = beta.tolist()

    rows = []
    for i, source in enumerate(score):
        item = {
            "workload": source["workload"],
            "status": "candidate",
            "coverage_status": "native_endpoint_covered",
            "native_event_ns": float(native_score[i]),
            "native_endpoint_feature_distance": 0.0,
        }
        for target in TARGETS:
            actual = value(source, target)
            pred = float(predictions[target][i])
            low, high = intervals[target]
            item[f"{target}_actual"] = actual
            item[f"{target}_predicted"] = pred
            item[f"{target}_abs_error"] = abs(pred - actual)
            item[f"{target}_ape"] = abs(pred - actual) / actual
            item[f"{target}_interval_low"] = float(low[i])
            item[f"{target}_interval_high"] = float(high[i])
        rows.append(item)
    metrics = {}
    for target in TARGETS:
        actual = np.asarray([r[f"{target}_actual"] for r in rows])
        pred = np.asarray([r[f"{target}_predicted"] for r in rows])
        metrics[target] = {
            "mae": float(np.mean(np.abs(pred - actual))),
            "mape": float(np.mean(np.abs(pred - actual) / actual)),
            "max_abs_error": float(np.max(np.abs(pred - actual))),
        }
    elapsed = time.perf_counter() - started
    replay_wall = sum(value(r, "replay_wall_seconds") for r in score)
    report = {
        "schema": "sm86_native_endpoint_ratio_v1",
        "train_campaign": str(args.train_campaign),
        "score_campaign": str(args.score_campaign),
        "calibration_family": args.calibration_family,
        "holdout_family": args.holdout_family,
        "feature_contract": "native CUDA-event mean only; no NCU/pipe/cache counters",
        "target_labels_excluded_from_fit": True,
        "ridge": args.ridge,
        "metrics": metrics,
        "model_seconds": elapsed,
        "projection_only_speedup_vs_replay_wall": replay_wall / elapsed if elapsed else None,
        "calibration_residual_q90": residual_q90,
        "predictions": rows,
        "coefficients": coefficients,
        "limitations": [
            "native CUDA-event and NCU semaphore endpoints remain distinct",
            "same campaign/cache/clock regime is required",
            "related-family holdout is not independent workload generalization",
            "native event includes its own launch/serialization semantics",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-campaign", type=Path, required=True)
    parser.add_argument("--score-campaign", type=Path, required=True)
    parser.add_argument("--calibration-family", choices=("base", "large"), required=True)
    parser.add_argument("--holdout-family", choices=("base", "large"), required=True)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.ridge <= 0 or not math.isfinite(args.ridge):
        parser.error("--ridge must be positive and finite")
    report = run(args)
    print(json.dumps({"metrics": report["metrics"], "model_seconds": report["model_seconds"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
