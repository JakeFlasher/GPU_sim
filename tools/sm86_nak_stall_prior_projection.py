#!/usr/bin/env python3
"""Project SM86 endpoints with an independent CUPTI/NAK stall-prior surface.

This is a metric-targeted alternative, not a physical scheduler simulator.
The independent NAK/CUPTI family matrix supplies aggregate stall-reason
priors for semantic operation groups.  Those priors are combined with sparse
replay frontiers and trace populations in a small log-ratio model.  Physical
pipe, sub-unit, cache-owner, and issue-slot identities remain unknown.

The score cohort is held out by workload family.  NCU values in the score
campaign are read only after prediction to report error; they never enter the
feature vector, fit, route, or model selection.  Rows outside the feature
hull receive a widened candidate interval rather than an OOD refusal.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np

from sm86_ncu_projection_surrogate import trace_features


TARGETS = ("ncu_gpc_cycles", "ncu_duration_ns")
REASONS = (
    "long_scoreboard",
    "imc_miss",
    "short_scoreboard",
    "wait",
    "branch_resolving",
    "drain",
    "misc",
)
GROUPS = ("math", "tensor", "memory", "sfu", "control", "special")


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _load_rows(path: Path) -> list[dict[str, str]]:
    with (path / "performance.csv").open(newline="") as stream:
        return [row for row in csv.DictReader(stream) if row.get("status") == "completed"]


def _family(workload: str) -> str:
    return workload.split("__", 1)[0]


def _load_priors(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Load independent NAK/CUPTI family reason fractions.

    The matrix was collected from repeated NAK CUDA-ELF bridge bodies, not
    from the score rows.  We deliberately use medians and a broad global
    fallback: the values are semantic stall-state priors, not route tags.
    """
    matrix = json.loads(path.read_text())
    families = matrix.get("families")
    if not isinstance(families, list) or not families:
        raise RuntimeError(f"empty NAK/CUPTI family matrix: {path}")
    vectors: dict[str, np.ndarray] = {}
    for item in families:
        fractions = item.get("base_reason_fraction", {})
        vector = np.asarray([max(float(fractions.get(reason, 0.0)), 0.0) for reason in REASONS], dtype=float)
        total = float(np.sum(vector))
        if not math.isfinite(total) or total <= 0.0:
            raise RuntimeError(f"invalid CUPTI prior for {item.get('family')}")
        vectors[str(item.get("family"))] = vector / total
    global_prior = np.median(np.asarray(list(vectors.values())), axis=0)
    global_prior /= max(float(np.sum(global_prior)), 1e-12)

    # NAK bodies provide operation-local semantic probes.  Scalar arithmetic
    # groups use the arithmetic family median; SFU uses reciprocal; tensor and
    # memory deliberately use the broad median because no NAK body establishes
    # a tensor pipe or a GA102 cache-owner mapping.
    arithmetic_keys = ("03_mul", "04_fma_like", "06_shift", "07_xor", "08_maximum")
    arithmetic = np.median(np.asarray([vectors[key] for key in arithmetic_keys if key in vectors]), axis=0)
    arithmetic /= max(float(np.sum(arithmetic)), 1e-12)
    sfu = vectors.get("05_reciprocal", global_prior)
    priors = {
        "math": arithmetic,
        "tensor": arithmetic,
        "memory": global_prior,
        "sfu": sfu,
        "control": global_prior,
        "special": global_prior,
    }
    metadata = {
        "source": str(path),
        "independent_family_count": len(vectors),
        "reason_order": list(REASONS),
        "operation_group_rule": {
            "math": "median(03_mul,04_fma_like,06_shift,07_xor,08_maximum)",
            "tensor": "arithmetic median; no tensor-specific NAK witness",
            "memory": "global NAK-family median; no cache ownership inference",
            "sfu": "05_reciprocal",
            "control": "global NAK-family median",
            "special": "global NAK-family median",
        },
        "priors": {key: value.tolist() for key, value in priors.items()},
        "global_prior": global_prior.tolist(),
        "score_labels_not_used": True,
    }
    return priors, metadata


def _frontier_base(row: dict[str, str], target: str) -> float:
    event = max(_float(row, "sim_event_cycles"), 1.0)
    envelope = max(_float(row, "sim_gpc_envelope_cycles"), event)
    if target == "ncu_gpc_cycles":
        return envelope
    # SM86 anchor is 1.35 GHz. This is a domain conversion, not a claim that
    # CUDA events, NCU, and simulator clocks share a synchronized endpoint.
    return event * (1000.0 / 1350.0)


def _feature_map(row: dict[str, str], campaign: Path, priors: dict[str, np.ndarray]) -> dict[str, float]:
    static, _ = trace_features(campaign, row["workload"])
    mix = np.zeros(len(REASONS), dtype=float)
    total = 0.0
    for group in GROUPS:
        weight = max(float(static.get(f"fraction_{group}", 0.0)), 0.0)
        mix += weight * priors[group]
        total += weight
    if total <= 0.0:
        mix = np.mean(np.asarray(list(priors.values())), axis=0)
        total = 1.0
    mix /= max(total, 1e-12)
    event = max(_float(row, "sim_event_cycles"), 1.0)
    envelope = max(_float(row, "sim_gpc_envelope_cycles"), event)
    issue = max(_float(row, "sim_last_issue_cycles"), 0.0)
    instructions = max(_float(row, "sim_warp_instructions"), 1.0)
    values = {
        "log_frontier": math.log1p(envelope),
        "log_event": math.log1p(event),
        "log_instructions": math.log1p(instructions),
        "envelope_delta_per_instruction": max(envelope - event, 0.0) / instructions,
        "tail_delta_per_instruction": max(event - issue, 0.0) / instructions,
        "memory_fraction": float(static.get("fraction_memory", 0.0)),
        "tensor_fraction": float(static.get("fraction_tensor", 0.0)),
        "sfu_fraction": float(static.get("fraction_sfu", 0.0)),
        "control_fraction": float(static.get("fraction_control", 0.0)),
        "trace_memory_sector_log": math.log1p(max(static.get("log_trace_memory_sectors", 0.0), 0.0)),
    }
    for index, reason in enumerate(REASONS):
        values[f"prior_{reason}"] = float(mix[index])
    # Keep a few explicit pressure combinations. They are semantic features,
    # not decoded hardware meanings.
    values["prior_memory_wait_pressure"] = float(mix[0] + mix[1] + mix[2] * 0.5)
    values["prior_control_pressure"] = float(mix[3] + mix[4] + mix[5] * 0.5)
    values["prior_nonissue_pressure"] = float(np.sum(mix))
    return values


def _fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    scale = np.sqrt(np.mean(x[:, 1:] * x[:, 1:], axis=0))
    scale[scale == 0.0] = 1.0
    xs = x.copy()
    xs[:, 1:] /= scale
    beta = np.linalg.solve(xs.T @ xs + ridge * np.eye(xs.shape[1]), xs.T @ y)
    return beta, scale


def _predict(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    train_all = _load_rows(args.train_campaign)
    score_all = _load_rows(args.score_campaign)
    train = [row for row in train_all if _family(row["workload"]) != args.holdout_family]
    score = [row for row in score_all if _family(row["workload"]) == args.holdout_family]
    if not train or not score:
        raise RuntimeError("empty disjoint train/score cohort")
    priors, prior_metadata = _load_priors(args.cupti_matrix)
    train_maps = [_feature_map(row, args.train_campaign, priors) for row in train]
    score_maps = [_feature_map(row, args.score_campaign, priors) for row in score]
    names = sorted(train_maps[0])
    if any(sorted(item) != names for item in train_maps + score_maps):
        raise RuntimeError("train/score feature schema differs")
    x_train = np.asarray([[1.0] + [item[name] for name in names] for item in train_maps], dtype=float)
    x_score = np.asarray([[1.0] + [item[name] for name in names] for item in score_maps], dtype=float)
    predictions: dict[str, np.ndarray] = {}
    intervals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    coefficients: dict[str, list[float]] = {}
    residual_q90: dict[str, float] = {}
    base_train_by_target: dict[str, np.ndarray] = {}
    base_score_by_target: dict[str, np.ndarray] = {}
    for target in TARGETS:
        base_train = np.asarray([_frontier_base(row, target) for row in train], dtype=float)
        base_score = np.asarray([_frontier_base(row, target) for row in score], dtype=float)
        y = np.asarray([_float(row, target) for row in train], dtype=float)
        if np.any(y <= 0.0):
            raise RuntimeError(f"non-positive train target {target}")
        beta, scale = _fit_ridge(x_train, np.log(y / base_train), args.ridge)
        train_ratio = np.exp((x_train / np.r_[1.0, scale]) @ beta)
        score_ratio = np.exp((x_score / np.r_[1.0, scale]) @ beta)
        train_pred = base_train * train_ratio
        score_pred = base_score * score_ratio
        residual = np.abs(train_pred - y)
        q90 = float(np.quantile(residual, 0.90))
        predictions[target] = np.maximum(score_pred, 0.0)
        residual_q90[target] = q90
        base_train_by_target[target] = base_train
        base_score_by_target[target] = base_score
        coefficients[target] = beta.tolist()
        intervals[target] = (np.maximum(predictions[target] - q90, 0.0), predictions[target] + q90)

    feature_scale = np.sqrt(np.mean(x_train[:, 1:] * x_train[:, 1:], axis=0))
    feature_scale[feature_scale == 0.0] = 1.0
    distances = np.asarray([
        float(np.min(np.linalg.norm((x_train[:, 1:] - item[1:]) / feature_scale, axis=1)))
        for item in x_score
    ])
    distance_factor = 1.0 + np.maximum(distances - args.ood_threshold, 0.0) / max(args.ood_threshold, 1.0)
    for target in TARGETS:
        width = residual_q90[target] * distance_factor
        intervals[target] = (
            np.maximum(predictions[target] - width, 0.0),
            predictions[target] + width,
        )
    rows = []
    for index, row in enumerate(score):
        item = {
            "workload": row["workload"],
            "status": "candidate" if distances[index] <= args.ood_threshold else "candidate_fallback",
            "coverage_status": "stall_prior_covered" if distances[index] <= args.ood_threshold else "stall_prior_fallback_covered",
            "feature_distance": float(distances[index]),
            "fallback_distance_factor": float(distance_factor[index]),
            "prior_mix": score_maps[index],
        }
        for target in TARGETS:
            actual = _float(row, target, float("nan"))
            predicted = float(predictions[target][index])
            low, high = intervals[target]
            item[f"{target}_predicted"] = predicted
            item[f"{target}_actual"] = actual if math.isfinite(actual) else None
            item[f"{target}_ape"] = abs(predicted - actual) / actual if math.isfinite(actual) and actual > 0 else None
            item[f"{target}_interval_low"] = float(low[index])
            item[f"{target}_interval_high"] = float(high[index])
        rows.append(item)
    metrics = {}
    for target in TARGETS:
        apes = [row[f"{target}_ape"] for row in rows if row[f"{target}_ape"] is not None]
        metrics[target] = {
            "count": len(apes),
            "mape": float(np.mean(apes)) if apes else None,
            "max_ape": float(np.max(apes)) if apes else None,
        }
    report = {
        "schema": "sm86_nak_stall_prior_projection_v1",
        "train_campaign": str(args.train_campaign),
        "score_campaign": str(args.score_campaign),
        "holdout_family": args.holdout_family,
        "cupti_matrix": str(args.cupti_matrix),
        "train_workloads": [row["workload"] for row in train],
        "score_workloads": [row["workload"] for row in score],
        "target_labels_score_only": True,
        "feature_names": names,
        "prior_metadata": prior_metadata,
        "model": {
            "base": "replay GPC envelope or 1350-MHz event-domain conversion",
            "fit": "ridge on log(NCU/base) using opposite workload cohort only",
            "prior_projection": "semantic opcode-group mixture of independent NAK/CUPTI reason vectors",
            "interval": "training residual q90 widened by feature distance; not confidence",
            "ood_policy": "candidate_fallback, never refusal",
        },
        "ridge": args.ridge,
        "ood_threshold": args.ood_threshold,
        "residual_q90": residual_q90,
        "metrics": metrics,
        "model_seconds": time.perf_counter() - started,
        "projection_only_speedup_vs_replay_wall": sum(_float(row, "replay_wall_seconds") for row in score) / max(time.perf_counter() - started, 1e-12),
        "predictions": rows,
        "coefficients": coefficients,
        "limitations": [
            "CUPTI PC samples are aggregate sampled stall states, not physical pipe or issue-slot tags",
            "NAK bridge bodies are independent semantic probes and do not establish target-kernel opcode causality",
            "cache ownership/read-from/complete DRAM and unified timing remain latent",
            "related-family holdout is not architecture-wide generalization",
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
    parser.add_argument("--cupti-matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=0.25)
    parser.add_argument("--ood-threshold", type=float, default=3.0)
    args = parser.parse_args()
    if not math.isfinite(args.ridge) or args.ridge <= 0.0:
        parser.error("--ridge must be positive and finite")
    if not math.isfinite(args.ood_threshold) or args.ood_threshold <= 0.0:
        parser.error("--ood-threshold must be positive and finite")
    report = _predict(args)
    print(json.dumps({
        "holdout_family": report["holdout_family"],
        "metrics": report["metrics"],
        "score_rows": len(report["score_workloads"]),
        "model_seconds": report["model_seconds"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
