#!/usr/bin/env python3
"""Frontier-ratio bundle for fast SM86 metric projection.

This model is a deliberately small synthesis of interval analysis, sampled
simulation, and analytical pipeline-efficiency modeling:

* GPUMech-style interval reasoning supplies a physical lower-bound frontier;
* Photon/GPUVision motivate retaining compact frontiers instead of replaying
  every warp or treating a PM sample as an exact route witness;
* PipeWeave motivates learning efficiency ratios over analytical demand
  features rather than fitting an opaque absolute-duration scalar.

For each target metric the model fits ``log(NCU / frontier_base)`` on an
independent calibration cohort.  A separate duration/GPC conversion model is
used as a consistency crosswalk and blended geometrically with the direct
duration prediction.  NCU values from the score campaign are loaded only for
post-prediction APE reporting.  Every score row receives a candidate estimate;
distance-scaled residual intervals expose weak coverage without dropping rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np

from sm86_ncu_projection_surrogate import frontier_features, fit_ridge, load_rows, _float


TARGETS = ("ncu_gpc_cycles", "ncu_duration_ns")
SM_CLOCK_MHZ = 1350.0
NS_PER_SM_CYCLE = 1_000.0 / SM_CLOCK_MHZ


def _feature_map(row: dict[str, str], campaign: Path) -> dict[str, float]:
    f = frontier_features(row, campaign)
    event = max(f.get("sim_event_cycles", 0.0), 1.0)
    envelope = max(event + f.get("sim_envelope_delta", 0.0), 1.0)
    issue = max(event - f.get("sim_tail_delta", 0.0), 1.0)
    work = max(_float(row, "sim_warp_instructions"), 1.0)
    return {
        "log_event": math.log(event),
        "log_envelope": math.log(envelope),
        "log_issue": math.log(issue),
        "log_work": math.log(work),
        "envelope_over_event": envelope / event,
        "tail_over_event": max(event - issue, 0.0) / event,
        "work_over_event": work / event,
        "log_trace_blocks": f.get("log_trace_blocks", 0.0),
        "trace_threads_per_block": f.get("trace_threads_per_block", 0.0),
        "log_trace_insts_per_block": f.get("log_trace_insts_per_block", 0.0),
    }


def _base(row: dict[str, str], target: str) -> float:
    event = max(_float(row, "sim_event_cycles"), 1.0)
    envelope = max(_float(row, "sim_gpc_envelope_cycles"), event)
    if target == "ncu_gpc_cycles":
        return envelope
    return event * NS_PER_SM_CYCLE


def _fit_matrix(train_rows: list[dict[str, str]], train_campaign: Path):
    maps = [_feature_map(row, train_campaign) for row in train_rows]
    names = sorted(maps[0])
    x = np.asarray([[1.0] + [m[name] for name in names] for m in maps], dtype=float)
    return x, names


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    train_rows = load_rows(args.train_campaign)
    score_rows = load_rows(args.score_campaign)
    if args.holdout_family:
        train_rows = [r for r in train_rows if r["workload"].split("__", 1)[0] != args.holdout_family]
        score_rows = [r for r in score_rows if r["workload"].split("__", 1)[0] == args.holdout_family]
    if not train_rows or not score_rows:
        raise RuntimeError("empty train or score family partition")
    x_train, names = _fit_matrix(train_rows, args.train_campaign)
    score_maps = [_feature_map(row, args.score_campaign) for row in score_rows]
    x_score = np.asarray([[1.0] + [m[name] for name in names] for m in score_maps], dtype=float)

    predictions: dict[str, np.ndarray] = {}
    intervals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    residuals: dict[str, float] = {}
    coefficients: dict[str, list[float]] = {}
    train_predictions: dict[str, np.ndarray] = {}
    for target in TARGETS:
        train_y = np.asarray([_float(row, target) for row in train_rows], dtype=float)
        train_base = np.asarray([_base(row, target) for row in train_rows], dtype=float)
        score_base = np.asarray([_base(row, target) for row in score_rows], dtype=float)
        log_ratio = np.log(np.maximum(train_y, 1.0) / np.maximum(train_base, 1.0))
        beta, scale = fit_ridge(x_train, log_ratio, args.ridge)
        train_pred = train_base * np.exp((x_train / scale) @ beta)
        score_pred = score_base * np.exp((x_score / scale) @ beta)
        predictions[target] = np.maximum(score_pred, 0.0)
        train_predictions[target] = train_pred
        residuals[target] = float(np.quantile(np.abs(train_pred - train_y), 0.90))
        coefficients[target] = beta.tolist()

    # Independent endpoint consistency crosswalk: duration/GPC is learned only
    # from calibration rows and then blended with direct duration.  Geometric
    # blending prevents one endpoint's unit scale from dominating the other.
    train_gpc = np.asarray([_float(row, "ncu_gpc_cycles") for row in train_rows])
    train_dur = np.asarray([_float(row, "ncu_duration_ns") for row in train_rows])
    conversion_y = np.log(np.maximum(train_dur, 1.0) / np.maximum(train_gpc, 1.0))
    conversion_beta, conversion_scale = fit_ridge(x_train, conversion_y, args.ridge)
    converted_duration = predictions["ncu_gpc_cycles"] * np.exp((x_score / conversion_scale) @ conversion_beta)
    predictions["ncu_duration_ns"] = np.sqrt(
        np.maximum(predictions["ncu_duration_ns"], 1.0) * np.maximum(converted_duration, 1.0)
    )
    # Re-estimate duration residual after the consistency blend on calibration.
    converted_train = train_predictions["ncu_gpc_cycles"] * np.exp((x_train / conversion_scale) @ conversion_beta)
    blended_train_duration = np.sqrt(np.maximum(train_predictions["ncu_duration_ns"], 1.0) * np.maximum(converted_train, 1.0))
    residuals["ncu_duration_ns"] = float(np.quantile(np.abs(blended_train_duration - train_dur), 0.90))

    # Feature distance is uncertainty metadata only.  It never suppresses a
    # prediction; its factor widens the interval for an unfamiliar row.
    scale = np.sqrt(np.mean(x_train[:, 1:] * x_train[:, 1:], axis=0))
    scale[scale == 0.0] = 1.0
    train_norm = x_train[:, 1:] / scale
    score_norm = x_score[:, 1:] / scale
    distances = np.asarray([float(np.min(np.linalg.norm(train_norm - row, axis=1))) for row in score_norm])
    distance_factor = 1.0 + np.minimum(distances / 3.0, 8.0)
    for target in TARGETS:
        width = residuals[target] * distance_factor
        intervals[target] = (
            np.maximum(predictions[target] - width, 0.0),
            predictions[target] + width,
        )

    rows = []
    for index, row in enumerate(score_rows):
        item = {
            "workload": row["workload"],
            "status": "candidate",
            "coverage_status": "frontier_ratio_covered",
            "feature_distance": float(distances[index]),
            "distance_factor": float(distance_factor[index]),
        }
        for target in TARGETS:
            actual = _float(row, target, float("nan"))
            pred = float(predictions[target][index])
            low, high = intervals[target]
            item[f"{target}_actual"] = actual if math.isfinite(actual) else None
            item[f"{target}_predicted"] = pred
            item[f"{target}_interval_low"] = float(low[index])
            item[f"{target}_interval_high"] = float(high[index])
            item[f"{target}_ape"] = abs(pred - actual) / actual if math.isfinite(actual) and actual > 0 else None
        rows.append(item)
    metrics = {}
    for target in TARGETS:
        actual = np.asarray([row[f"{target}_actual"] for row in rows], dtype=float)
        pred = np.asarray([row[f"{target}_predicted"] for row in rows], dtype=float)
        metrics[target] = {
            "mae": float(np.mean(np.abs(pred - actual))),
            "mape": float(np.mean(np.abs(pred - actual) / actual)),
            "max_abs_error": float(np.max(np.abs(pred - actual))),
        }
    report = {
        "schema": "sm86_frontier_ratio_bundle_v1",
        "train_campaign": str(args.train_campaign),
        "score_campaign": str(args.score_campaign),
        "holdout_family": args.holdout_family,
        "clock_assumption_mhz": SM_CLOCK_MHZ,
        "targets_are_score_only": True,
        "feature_names": names,
        "model": {
            "frontier_bases": "GPC=max(sim_envelope,event); duration=event/SM_clock",
            "fit": "log endpoint/frontier efficiency ratio",
            "endpoint_bundle": "geometric direct-duration and duration/GPC crosswalk",
            "interval": "q90 calibration residual widened by frontier feature distance",
            "mechanism_claim": "none; latent pipe/ABI/cache/endpoint effects remain residuals",
        },
        "metrics": metrics,
        "model_seconds": time.perf_counter() - started,
        "projection_only_speedup_vs_replay_wall": sum(_float(r, "replay_wall_seconds") for r in score_rows) / max(time.perf_counter() - started, 1e-12),
        "predictions": rows,
        "coefficients": coefficients,
        "conversion_coefficients": conversion_beta.tolist(),
        "limitations": [
            "cohort holdout is related-family evidence, not independent silicon generalization",
            "frontier bases are analytical lower-bound proxies, not physical route tags",
            "PM sampling, cache ownership, ABI metadata, and endpoint identity remain unobserved",
            "NCU score labels are loaded only after prediction",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-campaign", type=Path, required=True)
    parser.add_argument("--score-campaign", type=Path, required=True)
    parser.add_argument("--holdout-family", choices=("base", "large"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=1e-3)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"holdout_family": report["holdout_family"], "metrics": report["metrics"], "model_seconds": report["model_seconds"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
