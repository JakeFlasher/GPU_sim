#!/usr/bin/env python3
"""Independent-anchor tile/resource envelope for SM86 endpoint projection.

This is a deliberately small, white-box alternative to the generic frontier
regression.  It borrows the useful part of tile-centric models (NeuSight,
TileSight, PPT-GPU, and the analytical component of PipeWeave/SynPerf): split
work into a few resource demands, calibrate rates on independent anchors, and
combine the demands with a bottleneck envelope.  It does *not* claim that its
``math``/``memory``/``instruction`` terms are NVIDIA physical pipes.

The exact implementation uses a shape-archetype sidecar.  It fits shape laws
for resource populations using only rows in an independent anchor matrix, then
predicts a score-only target inside the exact BH/Sq/D and Sk hull.  Every row
outside that hull is covered by an independent frontier/trace fallback with a
wide distance-scaled interval and explicit ``candidate_fallback`` provenance;
there is no silent shape extrapolation and no OOD refusal in the output.

The model is an endpoint projection/committee member, not a cycle-accurate
replacement for native replay.  Its latent terms are useful because they keep
issue, memory, and launch/tail effects as a partial order instead of choosing
an invented physical pipe or cache owner.
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
RESOURCE_FIELDS = ("ncu_alu", "ncu_fma", "ncu_lsu", "ncu_tensor", "ncu_instructions")


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _load_rows(path: Path) -> list[dict[str, str]]:
    if path.is_dir():
        path = path / "performance.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [row for row in rows if row.get("status", "completed") == "completed"]


def _load_matrix(path: Path) -> list[dict[str, str]]:
    rows = _load_rows(path)
    if not rows:
        raise RuntimeError(f"empty anchor matrix: {path}")
    for row in rows:
        role = str(row.get("role", "")).lower()
        if "score_target" in role or role == "target":
            raise RuntimeError("anchor matrix contains a score target row")
        for key in ("BH", "Sq", "Sk", "D", *RESOURCE_FIELDS, *TARGETS):
            if not math.isfinite(_float(row, key, float("nan"))):
                raise RuntimeError(f"non-finite anchor field {key}")
    return rows


def _load_specs(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"shape spec must be an object: {path}")
    return value


def _spec_for(specs: dict, workload: str) -> dict | None:
    target = specs.get("target")
    if isinstance(target, dict) and target.get("workload") == workload:
        return target
    for candidate in specs.get("workloads", []):
        if isinstance(candidate, dict) and candidate.get("workload") == workload:
            return candidate
    return None


def _resource_vector(row: dict[str, str]) -> np.ndarray:
    # Keep the decomposition semantic rather than pretending each field is a
    # physical SM86 pipe.  ``math`` is an aggregate demand class; ``memory``
    # is the observed LSU population; ``instruction`` is a scheduler-work
    # proxy.  The three terms deliberately overlap, as a max-plus envelope
    # needs competing legal bottlenecks rather than a scalar sum.
    return np.asarray(
        [
            _float(row, "ncu_alu") + _float(row, "ncu_fma") + _float(row, "ncu_tensor"),
            _float(row, "ncu_lsu"),
            _float(row, "ncu_instructions"),
        ],
        dtype=float,
    )


def _quadratic(values: np.ndarray, sk: np.ndarray, target_sk: float) -> float:
    center = float(np.mean(sk))
    scale = max(float(np.ptp(sk)), 1.0)
    x = (sk - center) / scale
    target = (target_sk - center) / scale
    degree = 2 if len(values) >= 3 else 1
    coeff = np.polyfit(x, values, degree)
    return float(np.polyval(coeff, target))


def _predict_one(spec: dict | None, matrix: list[dict[str, str]]) -> dict:
    if spec is None:
        return {"status": "unmatched_shape", "reason": "missing_shape_spec"}
    required = ("BH", "Sq", "Sk", "D")
    if any(key not in spec for key in required):
        return {"status": "unmatched_shape", "reason": "shape_spec_missing_dimension"}
    target = {key: _float({key: str(spec[key])}, key, float("nan")) for key in required}
    archetype = str(spec.get("archetype", ""))
    anchors = []
    for row in matrix:
        if archetype and str(row.get("archetype", archetype)) != archetype:
            continue
        if all(_float(row, key) == target[key] for key in ("BH", "Sq", "D")):
            anchors.append(row)
    anchors.sort(key=lambda row: _float(row, "Sk"))
    if len(anchors) < 2:
        return {
            "status": "unmatched_shape",
            "reason": "fewer_than_two_matching_anchors",
            "archetype": archetype,
        }
    sk = np.asarray([_float(row, "Sk") for row in anchors], dtype=float)
    if not (float(sk[0]) <= target["Sk"] <= float(sk[-1])):
        return {
            "status": "unmatched_shape",
            "reason": "target_outside_matching_Sk_hull",
            "archetype": archetype,
            "anchor_low_Sk": float(sk[0]),
            "anchor_high_Sk": float(sk[-1]),
        }

    # First infer the target's resource demand from independent shape laws.
    anchor_resources = np.asarray([_resource_vector(row) for row in anchors])
    demand = np.asarray(
        [_quadratic(anchor_resources[:, index], sk, target["Sk"]) for index in range(3)],
        dtype=float,
    )
    demand = np.maximum(demand, 0.0)

    predictions: dict[str, float] = {}
    intervals: dict[str, tuple[float, float]] = {}
    latent: dict[str, dict[str, float]] = {}
    for target_name in TARGETS:
        # Each anchor gives a throughput rate for the three semantic demand
        # classes.  Median rates are robust to one campaign-state outlier;
        # q10/q90 rates form a deliberately conservative endpoint interval.
        rates = np.asarray(
            [
                _resource_vector(row) / max(_float(row, target_name), 1.0)
                for row in anchors
            ],
            dtype=float,
        )
        median_rate = np.median(rates, axis=0)
        q10_rate = np.quantile(rates, 0.10, axis=0)
        q90_rate = np.quantile(rates, 0.90, axis=0)
        terms = demand / np.maximum(median_rate, 1e-12)
        lower_terms = demand / np.maximum(q90_rate, 1e-12)
        upper_terms = demand / np.maximum(q10_rate, 1e-12)
        predictions[target_name] = float(np.max(terms))
        intervals[target_name] = (float(np.max(lower_terms)), float(np.max(upper_terms)))
        latent[target_name] = {
            "math": float(terms[0]),
            "memory": float(terms[1]),
            "instruction": float(terms[2]),
            "bottleneck_index": int(np.argmax(terms)),
            "rate_math_median": float(median_rate[0]),
            "rate_memory_median": float(median_rate[1]),
            "rate_instruction_median": float(median_rate[2]),
        }
    return {
        "status": "candidate",
        "path": "independent_tile_resource_max_envelope",
        "archetype": archetype,
        "anchor_count": len(anchors),
        "anchor_low": anchors[0].get("name", "lower"),
        "anchor_high": anchors[-1].get("name", "upper"),
        "anchor_low_Sk": float(sk[0]),
        "anchor_high_Sk": float(sk[-1]),
        "target_Sk": target["Sk"],
        "target_resource_demand": {
            "math": float(demand[0]),
            "memory": float(demand[1]),
            "instruction": float(demand[2]),
        },
        "predictions": predictions,
        "intervals": intervals,
        "latent_terms": latent,
    }


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    matrix = _load_matrix(args.archetype_matrix)
    specs = _load_specs(args.shape_specs)
    score_rows = _load_rows(args.score_campaign)
    if args.holdout_family:
        score_rows = [
            row for row in score_rows
            if row.get("workload", "").split("__", 1)[0] == args.holdout_family
        ]
    if not score_rows:
        raise RuntimeError("empty score family partition")

    fallback_train_rows = _load_rows(args.fallback_train_campaign)
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

    results: list[dict] = []
    for index, row in enumerate(score_rows):
        prediction = _predict_one(_spec_for(specs, row["workload"]), matrix)
        if prediction.get("status") != "candidate":
            prediction = {
                "status": "candidate_fallback",
                "path": "independent_frontier_trace_fallback",
                "reason": prediction.get("reason", "shape_not_matched"),
                "coverage_status": "fallback_covered",
                "feature_distance": float(fallback["feature_distances"][index]),
                "fallback_distance_factor": float(fallback["distance_factor"][index]),
                "predictions": {
                    target: float(fallback["predictions"][target][index])
                    for target in TARGETS
                },
                "intervals": {
                    target: (
                        float(fallback["intervals"][target][0][index]),
                        float(fallback["intervals"][target][1][index]),
                    )
                    for target in TARGETS
                },
            }
        item = {
            "workload": row["workload"],
            "status": prediction.get("status"),
            "path": prediction.get("path"),
            "reason": prediction.get("reason"),
            "archetype": prediction.get("archetype"),
            "anchor_low": prediction.get("anchor_low"),
            "anchor_high": prediction.get("anchor_high"),
            "latent_terms": prediction.get("latent_terms"),
            "target_resource_demand": prediction.get("target_resource_demand"),
            "coverage_status": prediction.get("coverage_status", "exact_shape"),
            "feature_distance": prediction.get("feature_distance"),
            "fallback_distance_factor": prediction.get("fallback_distance_factor"),
        }
        for target in TARGETS:
            actual = _float(row, target, float("nan"))
            predicted = prediction.get("predictions", {}).get(target)
            interval = prediction.get("intervals", {}).get(target)
            item[f"{target}_actual"] = actual if math.isfinite(actual) else None
            item[f"{target}_predicted"] = predicted
            item[f"{target}_interval_low"] = interval[0] if interval else None
            item[f"{target}_interval_high"] = interval[1] if interval else None
            item[f"{target}_ape"] = (
                abs(float(predicted) - actual) / actual
                if predicted is not None and math.isfinite(actual) and actual > 0
                else None
            )
        results.append(item)

    metrics = {}
    for target in TARGETS:
        scored = [
            item for item in results
            if item["status"] in ("candidate", "candidate_fallback")
            and item[f"{target}_ape"] is not None
        ]
        if scored:
            metrics[target] = {
                "count": len(scored),
                "mape": float(np.mean([item[f"{target}_ape"] for item in scored])),
                "max_ape": float(np.max([item[f"{target}_ape"] for item in scored])),
            }
        else:
            metrics[target] = {"count": 0, "mape": None, "max_ape": None}

    report = {
        "schema": "sm86_tile_pipeline_envelope_v2",
        "score_campaign": str(args.score_campaign),
        "holdout_family": args.holdout_family,
        "archetype_matrix": str(args.archetype_matrix),
        "shape_specs": str(args.shape_specs),
        "fallback_train_campaign": str(args.fallback_train_campaign),
        "fallback_transfer_campaign": (
            str(args.fallback_transfer_campaign)
            if args.fallback_transfer_campaign is not None
            else None
        ),
        "targets_are_score_only": True,
        "anchor_target_labels_excluded_from_fit": True,
        "model": {
            "work_partition": "math=ALU+FMA+Tensor, memory=LSU, instruction=all instructions",
            "shape_law": "degree-2 Sk interpolation inside exact BH/Sq/D hull",
            "endpoint_rule": "max-plus resource-rate envelope",
            "rate_estimator": "median independent-anchor demand/endpoint rate",
            "interval_rule": "q10/q90 anchor-rate envelope; not a confidence interval",
            "physical_claim": "none; terms are semantic demand classes, not physical pipes",
            "coverage_rule": "exact shape envelope first; independent frontier/trace fallback otherwise",
        },
        "metrics": metrics,
        "model_seconds": time.perf_counter() - started,
        "predictions": results,
        "limitations": [
            "all score rows receive a candidate or candidate_fallback prediction; fallback intervals widen with feature distance",
            "resource populations are independent-anchor calibration facts, not target NCU features",
            "cache ownership, physical issue route, ABI metadata, and endpoint identity remain latent",
            "target NCU values are loaded only for post-prediction APE reporting",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-campaign", type=Path, required=True)
    parser.add_argument("--holdout-family", default="")
    parser.add_argument("--archetype-matrix", type=Path, required=True)
    parser.add_argument("--shape-specs", type=Path, required=True)
    parser.add_argument("--fallback-train-campaign", type=Path, required=True)
    parser.add_argument("--fallback-transfer-campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "holdout_family": report["holdout_family"],
        "metrics": report["metrics"],
        "model_seconds": report["model_seconds"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
