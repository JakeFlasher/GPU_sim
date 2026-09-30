#!/usr/bin/env python3
"""Predict an SM86 warm endpoint from an independent cold/warm cache signature.

This is a narrow cache-phase alternative to a global warm/cold median.  It
interpolates two independent quantities inside an exact shape hull:

1. the cold endpoint as a shape response, and
2. the warm/cold endpoint ratio as a cache-phase response.

The warm prediction is their product.  The cache-sector signature (DRAM reads,
L2 misses, and the saved fraction) is emitted alongside the endpoint but is
not treated as a proof of ownership, read-from source, or DRAM completion.
Rows outside the exact shape hull receive an independent frontier/trace
fallback prediction with explicit provenance and a wide interval; the output
never emits an OOD refusal.  Score-row NCU labels are loaded only after the
prediction for APE reporting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np

from sm86_frontier_fallback import apply_independent_phase_transfer, fit_frontier_fallback


TARGETS = ("ncu_gpc_cycles", "ncu_duration_ns")
CACHE_FIELDS = ("dram_read_sectors", "dram_write_sectors", "l2_hit_sectors", "l2_miss_sectors")


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _rows(path: Path) -> list[dict[str, str]]:
    if path.is_dir():
        path = path / "performance.csv"
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _matrix(path: Path) -> list[dict[str, str]]:
    rows = _rows(path)
    if not rows:
        raise RuntimeError(f"empty cache matrix: {path}")
    for row in rows:
        role = str(row.get("role", "")).lower()
        if "score_target" in role or role == "target":
            raise RuntimeError("cache matrix contains a score target row")
        for key in ("BH", "Sq", "Sk", "D", "cache_phase", *TARGETS, *CACHE_FIELDS):
            if key == "cache_phase":
                if str(row.get(key, "")).lower() not in {"cold", "warm"}:
                    raise RuntimeError(f"invalid cache phase in matrix: {row.get(key)}")
                continue
            if not math.isfinite(_float(row, key, float("nan"))):
                raise RuntimeError(f"non-finite cache matrix field {key}")
    return rows


def _specs(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"shape spec must be an object: {path}")
    return value


def _spec_for(specs: dict, workload: str) -> dict | None:
    target = specs.get("target")
    if isinstance(target, dict) and target.get("workload") == workload:
        return target
    for item in specs.get("workloads", []):
        if isinstance(item, dict) and item.get("workload") == workload:
            return item
    return None


def _quadratic(values: np.ndarray, sk: np.ndarray, target_sk: float) -> float:
    center = float(np.mean(sk))
    scale = max(float(np.ptp(sk)), 1.0)
    x = (sk - center) / scale
    degree = 2 if len(values) >= 3 else 1
    return float(np.polyval(np.polyfit(x, values, degree), (target_sk - center) / scale))


def _matched(spec: dict | None, matrix: list[dict[str, str]]) -> tuple[list[dict[str, str]], float] | None:
    if spec is None or any(key not in spec for key in ("BH", "Sq", "Sk", "D")):
        return None
    target = {key: _float({key: str(spec[key])}, key, float("nan")) for key in ("BH", "Sq", "Sk", "D")}
    if not all(math.isfinite(value) for value in target.values()):
        return None
    archetype = str(spec.get("archetype", ""))
    rows = [
        row for row in matrix
        if str(row.get("cache_phase", "")).lower() in {"cold", "warm"}
        and (not archetype or str(row.get("archetype", archetype)) == archetype)
        and all(_float(row, key) == target[key] for key in ("BH", "Sq", "D"))
    ]
    return rows, target["Sk"]


def _predict(spec: dict | None, matrix: list[dict[str, str]]) -> dict:
    matched = _matched(spec, matrix)
    if matched is None:
        return {"status": "unmatched_shape", "reason": "missing_shape_spec"}
    rows, target_sk = matched
    cold = sorted([row for row in rows if row["cache_phase"].lower() == "cold"], key=lambda row: _float(row, "Sk"))
    warm = sorted([row for row in rows if row["cache_phase"].lower() == "warm"], key=lambda row: _float(row, "Sk"))
    if len(cold) < 2 or len(warm) < 2:
        return {"status": "unmatched_shape", "reason": "fewer_than_two_cold_warm_pairs"}
    cold_sk = np.asarray([_float(row, "Sk") for row in cold], dtype=float)
    warm_by_sk = {_float(row, "Sk"): row for row in warm}
    pairs = [(row, warm_by_sk[_float(row, "Sk")]) for row in cold if _float(row, "Sk") in warm_by_sk]
    if len(pairs) < 2:
        return {"status": "unmatched_shape", "reason": "cold_warm_shape_pairs_missing"}
    if not (float(cold_sk[0]) <= target_sk <= float(cold_sk[-1])):
        return {"status": "unmatched_shape", "reason": "target_outside_shape_hull"}

    predictions: dict[str, float] = {}
    intervals: dict[str, tuple[float, float]] = {}
    ratios: dict[str, float] = {}
    for target in TARGETS:
        cold_values = np.asarray([_float(row, target) for row in cold], dtype=float)
        ratio_values = np.asarray([
            _float(warm_row, target) / max(_float(cold_row, target), 1.0)
            for cold_row, warm_row in pairs
        ], dtype=float)
        cold_pred = _quadratic(cold_values, cold_sk, target_sk)
        ratio_pred = _quadratic(ratio_values, np.asarray([_float(a, "Sk") for a, _ in pairs]), target_sk)
        warm_pred = cold_pred * ratio_pred
        predictions[f"{target}_cold"] = cold_pred
        predictions[f"{target}_warm"] = warm_pred
        ratios[target] = ratio_pred
        # Anchor extrema are intentionally used as a bounded phase envelope;
        # they are not statistical confidence limits.
        low_ratio = float(np.min(ratio_values))
        high_ratio = float(np.max(ratio_values))
        intervals[f"{target}_warm"] = (
            cold_pred * min(low_ratio, high_ratio),
            cold_pred * max(low_ratio, high_ratio),
        )
        intervals[f"{target}_cold"] = (float(np.min(cold_values)), float(np.max(cold_values)))

    cache_signature: dict[str, float] = {}
    for field in CACHE_FIELDS:
        cold_values = np.asarray([_float(row, field) for row in cold], dtype=float)
        warm_values = np.asarray([_float(warm_by_sk[_float(row, "Sk")], field) for row in cold], dtype=float)
        cache_signature[f"{field}_cold"] = _quadratic(cold_values, cold_sk, target_sk)
        cache_signature[f"{field}_warm"] = _quadratic(warm_values, cold_sk, target_sk)
    cold_dram = cache_signature["dram_read_sectors_cold"]
    warm_dram = cache_signature["dram_read_sectors_warm"]
    cold_l2 = cache_signature["l2_miss_sectors_cold"]
    warm_l2 = cache_signature["l2_miss_sectors_warm"]
    cache_signature["dram_read_warm_cold_ratio"] = warm_dram / max(cold_dram, 1.0)
    cache_signature["l2_miss_warm_cold_ratio"] = warm_l2 / max(cold_l2, 1.0)
    cache_signature["dram_read_saved_fraction"] = 1.0 - cache_signature["dram_read_warm_cold_ratio"]
    cache_signature["l2_miss_saved_fraction"] = 1.0 - cache_signature["l2_miss_warm_cold_ratio"]
    return {
        "status": "candidate",
        "path": "independent_cache_signature_ratio",
        "anchor_count": len(pairs),
        "anchor_low_Sk": float(cold_sk[0]),
        "anchor_high_Sk": float(cold_sk[-1]),
        "target_Sk": target_sk,
        "predictions": predictions,
        "intervals": intervals,
        "warm_cold_endpoint_ratios": ratios,
        "cache_signature": cache_signature,
    }


def _global_cache_signature(matrix: list[dict[str, str]]) -> dict[str, float]:
    """Return aggregate independent phase medians for fallback diagnostics."""
    signature: dict[str, float] = {}
    for field in CACHE_FIELDS:
        for phase in ("cold", "warm"):
            values = [
                _float(row, field)
                for row in matrix
                if str(row.get("cache_phase", "")).lower() == phase
            ]
            signature[f"{field}_{phase}"] = float(np.median(values)) if values else 0.0
    cold_dram = signature.get("dram_read_sectors_cold", 0.0)
    warm_dram = signature.get("dram_read_sectors_warm", 0.0)
    cold_l2 = signature.get("l2_miss_sectors_cold", 0.0)
    warm_l2 = signature.get("l2_miss_sectors_warm", 0.0)
    signature["dram_read_warm_cold_ratio"] = warm_dram / max(cold_dram, 1.0)
    signature["l2_miss_warm_cold_ratio"] = warm_l2 / max(cold_l2, 1.0)
    signature["dram_read_saved_fraction"] = 1.0 - signature["dram_read_warm_cold_ratio"]
    signature["l2_miss_saved_fraction"] = 1.0 - signature["l2_miss_warm_cold_ratio"]
    return signature


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    matrix = _matrix(args.cache_matrix)
    specs = _specs(args.shape_specs)
    score_rows = _rows(args.score_campaign)
    if args.holdout_family:
        score_rows = [row for row in score_rows if row.get("workload", "").split("__", 1)[0] == args.holdout_family]
    if not score_rows:
        raise RuntimeError("empty score family partition")
    fallback_train_rows = _rows(args.fallback_train_campaign)
    if args.holdout_family:
        fallback_train_rows = [
            row for row in fallback_train_rows
            if row.get("workload", "").split("__", 1)[0] != args.holdout_family
        ]
    fallback = fit_frontier_fallback(
        args.fallback_train_campaign,
        args.score_campaign,
        fallback_train_rows,
        score_rows,
    )
    if args.fallback_transfer_campaign is not None:
        fallback = apply_independent_phase_transfer(
            fallback,
            fallback_train_rows,
            args.fallback_transfer_campaign,
            args.holdout_family,
        )
    global_signature = _global_cache_signature(matrix)
    results = []
    for index, row in enumerate(score_rows):
        prediction = _predict(_spec_for(specs, row["workload"]), matrix)
        if prediction.get("status") != "candidate":
            phase_target = {
                target: float(fallback["predictions"][target][index])
                for target in TARGETS
            }
            fallback_intervals = {
                target: (
                    float(fallback["intervals"][target][0][index]),
                    float(fallback["intervals"][target][1][index]),
                )
                for target in TARGETS
            }
            prediction = {
                "status": "candidate_fallback",
                "path": "independent_frontier_trace_fallback",
                "reason": prediction.get("reason", "shape_not_matched"),
                "coverage_status": "fallback_covered",
                "feature_distance": float(fallback["feature_distances"][index]),
                "fallback_distance_factor": float(fallback["distance_factor"][index]),
                "predictions": {
                    f"{target}_{args.phase}": phase_target[target]
                    for target in TARGETS
                },
                "intervals": {
                    f"{target}_{args.phase}": fallback_intervals[target]
                    for target in TARGETS
                },
                "cache_signature": {
                    **global_signature,
                    "source": "independent_phase_median_fallback",
                },
            }
        item = {"workload": row["workload"], **prediction}
        item["coverage_status"] = prediction.get("coverage_status", "exact_shape")
        item["feature_distance"] = prediction.get("feature_distance")
        item["fallback_distance_factor"] = prediction.get("fallback_distance_factor")
        for target in TARGETS:
            key = f"{target}_{args.phase}"
            actual = _float(row, target, float("nan"))
            predicted = prediction.get("predictions", {}).get(key)
            item[f"{target}_actual"] = actual if math.isfinite(actual) else None
            item[f"{target}_predicted"] = predicted
            item[f"{target}_ape"] = abs(predicted - actual) / actual if predicted is not None and math.isfinite(actual) and actual > 0 else None
        results.append(item)
    scored = [item for item in results if item.get("status") in ("candidate", "candidate_fallback")]
    metrics = {}
    for target in TARGETS:
        apes = [item[f"{target}_ape"] for item in scored if item[f"{target}_ape"] is not None]
        metrics[target] = {"count": len(apes), "mape": float(np.mean(apes)) if apes else None, "max_ape": float(np.max(apes)) if apes else None}
    report = {
        "schema": "sm86_cache_reuse_signature_v2",
        "score_campaign": str(args.score_campaign),
        "phase": args.phase,
        "holdout_family": args.holdout_family,
        "cache_matrix": str(args.cache_matrix),
        "shape_specs": str(args.shape_specs),
        "fallback_train_campaign": str(args.fallback_train_campaign),
        "fallback_transfer_campaign": (
            str(args.fallback_transfer_campaign)
            if args.fallback_transfer_campaign is not None
            else None
        ),
        "target_labels_score_only": True,
        "model": {
            "absolute_phase": "quadratic cold endpoint in Sk",
            "transfer_phase": "quadratic warm/cold endpoint ratio in Sk",
            "cache_signature": "quadratic sector/L2-miss phase response in Sk",
            "interval": "observed independent anchor ratio range; not confidence",
            "coverage_rule": "exact cache signature first; independent frontier/trace fallback otherwise",
        },
        "metrics": metrics,
        "model_seconds": time.perf_counter() - started,
        "predictions": results,
        "limitations": [
            "all score rows receive a candidate or candidate_fallback prediction; fallback intervals widen with feature distance",
            "cache signature is aggregate traffic, not ownership/read-from/DRAM completion",
            "NCU score labels are loaded only after prediction",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-campaign", type=Path, required=True)
    parser.add_argument("--holdout-family", default="")
    parser.add_argument("--phase", choices=("cold", "warm"), required=True)
    parser.add_argument("--cache-matrix", type=Path, required=True)
    parser.add_argument("--shape-specs", type=Path, required=True)
    parser.add_argument("--fallback-train-campaign", type=Path, required=True)
    parser.add_argument("--fallback-transfer-campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"phase": report["phase"], "metrics": report["metrics"], "model_seconds": report["model_seconds"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
