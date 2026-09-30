#!/usr/bin/env python3
"""Fast endpoint-lattice projection for SM86 NCU metrics.

This is a metric model, not a hidden-hardware reconstruction.  It treats the
simulator's issue/event/launch frontiers as separate axes and fits an endpoint
response surface from a disjoint training cohort.  A named archetype may add a
strict independent shape bracket (for example flash backward); the target NCU
row is never used to fit that bracket.

The model is intentionally small: one matrix solve, a bounded feature walk,
and optional local shape interpolation.  It is meant to replace detailed
replay when only an NCU-oriented estimate is needed, while leaving the native
replay as the mechanism reference.
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


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def load_rows(campaign: Path) -> list[dict[str, str]]:
    with (campaign / "performance.csv").open(newline="") as stream:
        return [row for row in csv.DictReader(stream) if row.get("status") == "completed"]


def trace_launch_features(campaign: Path | None, workload: str) -> tuple[float, float, float]:
    """Return trace-only launch statistics; absent traces are conservative zeroes."""
    if campaign is None:
        return 0.0, 0.0, 0.0
    path = campaign / workload / "trace" / "stats.csv"
    if not path.is_file():
        return 0.0, 0.0, 0.0
    try:
        with path.open(newline="") as stream:
            row = next(csv.DictReader(stream), None)
        if row is None:
            return 0.0, 0.0, 0.0
        return max(f(row, "#blocks"), 0.0), max(f(row, "#threads"), 0.0), max(
            f(row, "total_insts"), 0.0
        )
    except OSError:
        return 0.0, 0.0, 0.0


def lattice_features(row: dict[str, str], campaign: Path | None) -> dict[str, float]:
    """Create causal axes without NCU or aggregate pipe fields."""
    issue = max(f(row, "sim_last_issue_cycles"), 0.0)
    event = max(f(row, "sim_event_cycles"), 0.0)
    envelope = max(f(row, "sim_gpc_envelope_cycles"), 0.0)
    work = max(f(row, "sim_warp_instructions"), 1.0)
    tail = max(event - issue, 0.0)
    launch = max(envelope - event, 0.0)
    blocks, threads, trace_insts = trace_launch_features(campaign, row["workload"])
    log_work = math.log1p(work)
    log_grid = math.log1p(blocks)
    # Endpoint-lattice basis: separate issue/event/launch frontiers, then
    # population and launch-density terms.  Products are not NCU labels; they
    # encode the hypothesis that the same frontier has different endpoint cost
    # under different population densities.
    values = {
        "sim_issue": issue,
        "sim_event": event,
        "sim_envelope": envelope,
        "sim_tail": tail,
        "sim_launch_delta": launch,
        "log_sim_work": log_work,
        "log_trace_blocks": log_grid,
        "log_trace_threads": math.log1p(threads),
        "trace_inst_per_block": trace_insts / max(blocks, 1.0),
        "event_x_log_work": event * log_work,
        "envelope_x_log_grid": envelope * log_grid,
    }
    return values


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    scale = np.sqrt(np.mean(x * x, axis=0))
    scale[0] = 1.0
    scale[scale == 0.0] = 1.0
    xs = x / scale
    beta = np.linalg.solve(xs.T @ xs + ridge * np.eye(xs.shape[1]), xs.T @ y)
    return beta, scale


def load_shape_specs(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"shape spec must be an object: {path}")
    return value


def load_shape_matrix(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"empty shape matrix: {path}")
    for row in rows:
        role = str(row.get("role", "")).lower()
        if "score_target" in role or role == "target":
            raise RuntimeError("shape matrix contains score target labels")
    return rows


def shape_for_workload(specs: dict, workload: str) -> dict | None:
    target = specs.get("target")
    if isinstance(target, dict) and target.get("workload") == workload:
        return target
    for item in specs.get("workloads", []):
        if isinstance(item, dict) and item.get("workload") == workload:
            return item
    return None


def archetype_projection(
    spec: dict | None, matrix: list[dict[str, str]], default_archetype: str = ""
) -> dict | None:
    if spec is None:
        return None
    required = ("BH", "Sq", "Sk", "D")
    if any(key not in spec for key in required):
        return {"status": "unmatched_shape", "reason": "shape_spec_missing_dimension"}
    archetype = str(spec.get("archetype", default_archetype))
    target = {key: float(spec[key]) for key in required}
    rows = [
        row
        for row in matrix
        if str(row.get("archetype", archetype)) == archetype
        and all(f(row, key) == target[key] for key in ("BH", "Sq", "D"))
    ]
    rows.sort(key=lambda row: f(row, "Sk"))
    if not rows:
        return {"status": "unmatched_shape", "reason": "no_matching_BH_Sq_D_anchor"}
    exact = [row for row in rows if f(row, "Sk") == target["Sk"]]
    if exact:
        row = exact[0]
        return {
            "status": "candidate", "path": "independent_exact_anchor",
            "anchor_low": row.get("name", "exact"), "anchor_high": row.get("name", "exact"),
            "feature_distance": 0.0,
            "predictions": {key: f(row, key) for key in TARGETS},
            "intervals": {key: (f(row, key), f(row, key)) for key in TARGETS},
        }
    if len(rows) >= 3:
        min_sk, max_sk = f(rows[0], "Sk"), f(rows[-1], "Sk")
        if not (min_sk <= target["Sk"] <= max_sk):
            return {"status": "unmatched_shape", "reason": "target_outside_matching_Sk_hull"}
        anchor_sk = np.asarray([f(row, "Sk") for row in rows], dtype=float)
        center = float(np.mean(anchor_sk))
        scale = max(float(np.ptp(anchor_sk)), 1.0)
        normalized_sk = (anchor_sk - center) / scale
        target_sk = (target["Sk"] - center) / scale
        predictions, intervals = {}, {}
        for key in TARGETS:
            values = np.asarray([f(row, key) for row in rows], dtype=float)
            coefficients = np.polyfit(normalized_sk, values, deg=2)
            predictions[key] = float(np.polyval(coefficients, target_sk))
            intervals[key] = (float(np.min(values)), float(np.max(values)))
        return {
            "status": "candidate", "path": "independent_same_shape_quadratic_lattice",
            "anchor_low": rows[0].get("name", "lower"),
            "anchor_high": rows[-1].get("name", "upper"),
            "anchor_low_Sk": min_sk, "anchor_high_Sk": max_sk,
            "anchor_count": len(rows), "polynomial_degree": 2,
            "feature_distance": min(abs(target["Sk"] - min_sk), abs(max_sk - target["Sk"])) / max(max_sk - min_sk, 1.0),
            "predictions": predictions, "intervals": intervals,
        }
    lower = [row for row in rows if f(row, "Sk") < target["Sk"]]
    upper = [row for row in rows if f(row, "Sk") > target["Sk"]]
    if not lower or not upper:
        return {"status": "unmatched_shape", "reason": "target_outside_matching_Sk_bracket"}
    low, high = lower[-1], upper[0]
    low_sk, high_sk = f(low, "Sk"), f(high, "Sk")
    alpha = (target["Sk"] - low_sk) / max(high_sk - low_sk, 1.0)
    predictions, intervals = {}, {}
    for key in TARGETS:
        y0, y1 = f(low, key), f(high, key)
        predictions[key] = y0 + alpha * (y1 - y0)
        intervals[key] = (min(y0, y1), max(y0, y1))
    return {
        "status": "candidate", "path": "independent_same_shape_linear_bracket",
        "anchor_low": low.get("name", "lower"), "anchor_high": high.get("name", "upper"),
        "anchor_low_Sk": low_sk, "anchor_high_Sk": high_sk,
        "interpolation_fraction": alpha, "feature_distance": min(alpha, 1.0 - alpha),
        "predictions": predictions, "intervals": intervals,
    }


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    train_all, score_all = load_rows(args.train_campaign), load_rows(args.score_campaign)
    family = lambda row: row["workload"].split("__", 1)[0]
    train = [row for row in train_all if family(row) != args.holdout_family]
    score = [row for row in score_all if family(row) == args.holdout_family]
    if not train or not score:
        raise RuntimeError("empty train or score family partition")

    maps = [lattice_features(row, args.train_campaign) for row in train]
    score_maps = [lattice_features(row, args.score_campaign) for row in score]
    names = sorted(maps[0])
    if any(sorted(item) != names for item in maps + score_maps):
        raise RuntimeError("lattice feature schema differs")
    x_train = np.asarray([[1.0] + [item[name] for name in names] for item in maps], dtype=float)
    x_score = np.asarray([[1.0] + [item[name] for name in names] for item in score_maps], dtype=float)

    transfer: dict[str, float] = {}
    if args.cache_transfer_campaign is not None:
        transfer_rows = load_rows(args.cache_transfer_campaign)
        by_workload = {row["workload"]: row for row in transfer_rows}
        for target in TARGETS:
            ratios = [
                f(by_workload[row["workload"]], target) / f(row, target)
                for row in train
                if row["workload"] in by_workload and f(row, target) > 0
            ]
            if not ratios:
                raise RuntimeError("cache-transfer campaign has no training overlap")
            transfer[target] = float(np.median(ratios))

    predictions, intervals, coefficients, residuals = {}, {}, {}, {}
    for target in TARGETS:
        y = np.asarray([f(row, target) for row in train], dtype=float)
        beta, scale = fit_ridge(x_train, y, args.ridge)
        pred_train = (x_train / scale) @ beta
        pred_score = (x_score / scale) @ beta
        if target in transfer:
            pred_train *= transfer[target]
            pred_score *= transfer[target]
        pred_train = np.maximum(pred_train, 0.0)
        pred_score = np.maximum(pred_score, 0.0)
        residual = float(np.quantile(np.abs(pred_train - y), 0.90))
        predictions[target] = pred_score
        intervals[target] = [np.maximum(pred_score - residual, 0.0), pred_score + residual]
        coefficients[target] = beta.tolist()
        residuals[target] = residual

    # Nearest-training distance selects interval width, not workload coverage.
    # The archetype can explicitly resolve a row only from its independent
    # shape bracket; all other rows retain a generic candidate fallback.
    scale = np.sqrt(np.mean(x_train[:, 1:] * x_train[:, 1:], axis=0))
    scale[scale == 0.0] = 1.0
    dist = np.asarray([
        float(np.min(np.linalg.norm(x_train[:, 1:] / scale - item[1:] / scale, axis=1)))
        for item in x_score
    ])
    status = ["candidate" if value <= args.ood_threshold else "candidate_fallback" for value in dist]
    distance_factor = 1.0 + np.maximum(dist - args.ood_threshold, 0.0) / max(args.ood_threshold, 1.0)
    for target in TARGETS:
        width = residuals[target] * distance_factor
        intervals[target] = (
            np.maximum(predictions[target] - width, 0.0),
            predictions[target] + width,
        )
    archetypes: list[dict | None] = [None] * len(score)
    if args.shape_specs is not None or args.shape_matrix is not None:
        if args.shape_specs is None or args.shape_matrix is None:
            raise RuntimeError("--shape-specs and --shape-matrix must be supplied together")
        specs = load_shape_specs(args.shape_specs)
        matrix = load_shape_matrix(args.shape_matrix)
        for index, row in enumerate(score):
            archetypes[index] = archetype_projection(
                shape_for_workload(specs, row["workload"]),
                matrix,
                str(specs.get("archetype", "")),
            )
            local = archetypes[index]
            if not local or local.get("status") != "candidate":
                continue
            for target in TARGETS:
                factor = transfer.get(target, 1.0) if args.archetype_cache_transfer == "apply" else 1.0
                predictions[target][index] = local["predictions"][target] * factor
                intervals[target][0][index] = local["intervals"][target][0] * factor
                intervals[target][1][index] = local["intervals"][target][1] * factor
            status[index] = "candidate"

    output_rows = []
    for index, row in enumerate(score):
        item = {
            "workload": row["workload"],
            "feature_distance": float(dist[index]),
            "projection_status": status[index],
            "coverage_status": "fallback_covered" if status[index] == "candidate_fallback" else "frontier_covered",
            "fallback_distance_factor": float(distance_factor[index]),
        }
        local = archetypes[index]
        if local is not None:
            item.update({
                "generic_projection_status": status[index],
                "archetype_projection_status": local.get("status"),
                "archetype_projection_path": local.get("path"),
                "archetype_reason": local.get("reason"),
                "archetype_anchor_low": local.get("anchor_low"),
                "archetype_anchor_high": local.get("anchor_high"),
            })
        for target in TARGETS:
            actual = f(row, target)
            predicted = float(predictions[target][index])
            item[f"{target}_actual"] = actual
            item[f"{target}_predicted"] = predicted
            item[f"{target}_ape"] = abs(predicted - actual) / actual if actual else None
            item[f"{target}_interval_low"] = float(intervals[target][0][index])
            item[f"{target}_interval_high"] = float(intervals[target][1][index])
        output_rows.append(item)

    metrics = {}
    for target in TARGETS:
        actual = np.asarray([item[f"{target}_actual"] for item in output_rows])
        predicted = np.asarray([item[f"{target}_predicted"] for item in output_rows])
        metrics[target] = {
            "mae": float(np.mean(np.abs(predicted - actual))),
            "mape": float(np.mean(np.abs(predicted - actual) / actual)),
            "max_abs_error": float(np.max(np.abs(predicted - actual))),
        }
    elapsed = time.perf_counter() - started
    replay = sum(f(row, "replay_wall_seconds") for row in score)
    report = {
        "schema": "sm86_endpoint_lattice_projection_v2",
        "coverage_policy": "every score row receives candidate or candidate_fallback; no OOD refusal",
        "model": "separate issue/event/launch frontier lattice with trace-population products",
        "train_campaign": str(args.train_campaign),
        "score_campaign": str(args.score_campaign),
        "holdout_family": args.holdout_family,
        "feature_names": names,
        "feature_contract": "sim frontiers and trace launch metadata only; no NCU or pipe counters",
        "targets": list(TARGETS),
        "ridge": args.ridge,
        "ood_threshold": args.ood_threshold,
        "cache_transfer_campaign": str(args.cache_transfer_campaign) if args.cache_transfer_campaign else None,
        "cache_transfer_ratios": transfer,
        "archetype_cache_transfer": args.archetype_cache_transfer,
        "calibration_residual_q90": residuals,
        "metrics": metrics,
        "model_seconds": elapsed,
        "score_replay_wall_seconds_sum": replay,
        "projection_only_speedup_vs_replay_wall": replay / elapsed if elapsed else None,
        "predictions": output_rows,
        "coefficients": coefficients,
        "limitations": [
            "NCU labels are used only for the disjoint training cohort and final score",
            "a same-shape archetype bracket is a local endpoint response, not a mechanism witness",
            "warm cache state is valid only for the selected paired or transfer domain",
            "physical pipe tags, ABI generation, cache ownership, and unified timing remain latent",
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
    parser.add_argument("--cache-transfer-campaign", type=Path)
    parser.add_argument("--shape-matrix", type=Path)
    parser.add_argument("--shape-specs", type=Path)
    parser.add_argument("--archetype-cache-transfer", choices=("apply", "skip"), default="apply")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--ood-threshold", type=float, default=3.0)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "model": report["model"], "holdout_family": report["holdout_family"],
        "metrics": report["metrics"], "model_seconds": report["model_seconds"],
        "projection_only_speedup_vs_replay_wall": report["projection_only_speedup_vs_replay_wall"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
