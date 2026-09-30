#!/usr/bin/env python3
"""Reuse-distance phase transfer for fast SM86 endpoint projection.

The model borrows the cache-model idea of stack/reuse distance and the
interval-model idea of keeping semantic bottlenecks separate.  It does not
reconstruct a GA102 cache directory: addresses are reduced to per-threadblock
32-byte sector reuse signatures, and only a warm/cold endpoint ratio is fit.

Cold endpoint predictions come from a score-label-free baseline report.  The
warm/cold ratio is fit on the *opposite workload cohort* using paired cold and
warm NCU campaigns, then applied to the held-out cohort's trace signature.
Target NCU values in the score campaign are loaded after prediction only.
Every row receives a candidate or widened candidate_fallback interval.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


PB_ROOT = Path(__file__).resolve().parents[1] / "workloads" / "simbench_bw" / "_pb_generated"
sys.path.insert(0, str(PB_ROOT))
import threadblock_pb2  # type: ignore[import-not-found]


TARGETS = ("ncu_gpc_cycles", "ncu_duration_ns")


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _rows(campaign: Path) -> dict[str, dict[str, str]]:
    with (campaign / "performance.csv").open(newline="") as stream:
        return {row["workload"]: row for row in csv.DictReader(stream) if row.get("status") == "completed"}


class _Fenwick:
    def __init__(self, size: int):
        self.tree = [0] * (size + 2)

    def add(self, index: int, value: int) -> None:
        index += 1
        while index < len(self.tree):
            self.tree[index] += value
            index += index & -index

    def prefix(self, index: int) -> int:
        index += 1
        result = 0
        while index > 0:
            result += self.tree[index]
            index -= index & -index
        return result


def _sector_addresses(address, mask: int) -> list[int]:
    """Decode a bounded address record into sectors; unresolved records skip."""
    lanes = [lane for lane in range(32) if (mask >> lane) & 1]
    fmt = int(address.compression_format)
    if fmt == 0:
        values = [int(value) for value in address.addrs]
        return [value >> 5 for lane, value in enumerate(values[:32]) if lane in lanes and value >= 0]
    if fmt == 1:
        base = int(address.base_address)
        stride = int(address.stride)
        return [(base + lane * stride) >> 5 for lane in lanes]
    if fmt == 2:
        base = int(address.base_address)
        deltas = [int(value) for value in address.addrs]
        values = []
        for lane in lanes:
            if lane == 0:
                values.append(base)
            elif lane - 1 < len(deltas):
                values.append(base + deltas[lane - 1])
        return [value >> 5 for value in values]
    return []


def reuse_signature(campaign: Path, workload: str) -> dict[str, float]:
    """Compute per-threadblock sector reuse features in O(N log N)."""
    root = campaign / workload / "trace" / "threadblocks"
    accesses = 0
    unique_total = 0
    cold_total = 0
    distances: list[int] = []
    max_span = 0
    file_count = 0
    for path in sorted(root.rglob("*.pb")):
        block = threadblock_pb2.threadblock()
        block.ParseFromString(path.read_bytes())
        sequence: list[int] = []
        for warp in block.warps.values():
            for instruction in warp.instructions:
                mask = int(instruction.active_mask) & int(instruction.predicate_mask)
                if not mask:
                    continue
                for address in instruction.addresses:
                    # MEMORY_SPACE_GLOBAL=1; generic/legacy traces are kept
                    # out unless explicitly marked global to avoid inventing
                    # cache traffic for shared/local operands.
                    if int(address.address_space) != 1:
                        continue
                    sequence.extend(_sector_addresses(address, mask))
        if not sequence:
            continue
        file_count += 1
        accesses += len(sequence)
        last: dict[int, int] = {}
        bit = _Fenwick(len(sequence) + 1)
        block_unique = set(sequence)
        unique_total += len(block_unique)
        for index, sector in enumerate(sequence):
            previous = last.get(sector)
            if previous is None:
                cold_total += 1
            else:
                # The Fenwick tree stores one marker per sector's latest use;
                # the difference is the number of distinct sectors touched
                # since the previous use (stack/reuse distance).
                distance = bit.prefix(index - 1) - bit.prefix(previous)
                distances.append(max(distance, 0))
                max_span = max(max_span, distance)
                bit.add(previous, -1)
            bit.add(index, 1)
            last[sector] = index
    if accesses <= 0:
        return {
            "global_accesses": 0.0, "global_unique_per_block": 0.0,
            "global_cold_fraction": 1.0, "reuse_fraction": 0.0,
            "reuse_q50_log": 0.0, "reuse_q90_log": 0.0,
            "reuse_l1_fraction": 0.0, "reuse_l2_fraction": 0.0,
            "max_reuse_log": 0.0, "reuse_blocks": 0.0,
        }
    return {
        "global_accesses": float(accesses),
        "global_unique_per_block": unique_total / max(file_count, 1),
        "global_cold_fraction": cold_total / accesses,
        "reuse_fraction": len(distances) / accesses,
        "reuse_q50_log": math.log1p(float(np.quantile(distances, 0.50))) if distances else 0.0,
        "reuse_q90_log": math.log1p(float(np.quantile(distances, 0.90))) if distances else 0.0,
        "reuse_l1_fraction": sum(distance <= 4096 for distance in distances) / max(len(distances), 1),
        "reuse_l2_fraction": sum(distance <= 196608 for distance in distances) / max(len(distances), 1),
        "max_reuse_log": math.log1p(max_span),
        "reuse_blocks": float(file_count),
    }


def _features(row: dict[str, str], signature: dict[str, float]) -> dict[str, float]:
    event = max(_float(row, "sim_event_cycles"), 1.0)
    envelope = max(_float(row, "sim_gpc_envelope_cycles"), event)
    instructions = max(_float(row, "sim_warp_instructions"), 1.0)
    return {
        "log_event": math.log1p(event),
        "log_envelope": math.log1p(envelope),
        "log_instructions": math.log1p(instructions),
        "tail_per_instruction": max(event - _float(row, "sim_last_issue_cycles"), 0.0) / instructions,
        **signature,
    }


def _fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    scale = np.sqrt(np.mean(x[:, 1:] * x[:, 1:], axis=0))
    scale[scale == 0.0] = 1.0
    scaled = x.copy()
    scaled[:, 1:] /= scale
    beta = np.linalg.solve(scaled.T @ scaled + ridge * np.eye(scaled.shape[1]), scaled.T @ y)
    return beta, scale


def _load_baseline(path: Path) -> dict[str, dict]:
    report = json.loads(path.read_text())
    rows = report.get("predictions", [])
    result = {}
    for item in rows:
        workload = item.get("workload")
        if not workload:
            continue
        result[workload] = item
    if not result:
        raise RuntimeError(f"empty baseline prediction report: {path}")
    return result


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    cold_train = _rows(args.train_cold_campaign)
    warm_train = _rows(args.train_warm_campaign)
    cold_score = _rows(args.score_cold_campaign)
    warm_score = _rows(args.score_warm_campaign)
    train_names = sorted(set(cold_train) & set(warm_train))
    train_names = [name for name in train_names if name.split("__", 1)[0] != args.holdout_family]
    score_names = sorted(set(cold_score) & set(warm_score))
    score_names = [name for name in score_names if name.split("__", 1)[0] == args.holdout_family]
    if not train_names or not score_names:
        raise RuntimeError("empty disjoint paired train/score cohort")
    train_maps = []
    score_maps = []
    signatures = {}
    for name in train_names:
        signature = reuse_signature(args.train_cold_campaign, name)
        signatures[f"train:{name}"] = signature
        train_maps.append(_features(cold_train[name], signature))
    for name in score_names:
        signature = reuse_signature(args.score_cold_campaign, name)
        signatures[f"score:{name}"] = signature
        score_maps.append(_features(cold_score[name], signature))
    names = sorted(train_maps[0])
    if any(sorted(item) != names for item in train_maps + score_maps):
        raise RuntimeError("reuse feature schema differs")
    x_train = np.asarray([[1.0] + [item[name] for name in names] for item in train_maps], dtype=float)
    x_score = np.asarray([[1.0] + [item[name] for name in names] for item in score_maps], dtype=float)
    ratios: dict[str, np.ndarray] = {}
    ratio_intervals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    coefficients = {}
    residual_q90 = {}
    for target in TARGETS:
        cold_values = np.asarray([_float(cold_train[name], target) for name in train_names], dtype=float)
        warm_values = np.asarray([_float(warm_train[name], target) for name in train_names], dtype=float)
        if np.any(cold_values <= 0.0) or np.any(warm_values <= 0.0):
            raise RuntimeError(f"non-positive training endpoint for {target}")
        beta, scale = _fit_ridge(x_train, np.log(warm_values / cold_values), args.ridge)
        train_pred = np.exp((x_train / np.r_[1.0, scale]) @ beta)
        score_pred = np.exp((x_score / np.r_[1.0, scale]) @ beta)
        ratios[target] = score_pred
        residual = np.abs(np.log(train_pred) - np.log(warm_values / cold_values))
        q90 = float(np.quantile(residual, 0.90))
        residual_q90[target] = q90
        ratio_intervals[target] = (score_pred * math.exp(-q90), score_pred * math.exp(q90))
        coefficients[target] = beta.tolist()

    baseline = _load_baseline(args.cold_baseline)
    results = []
    for index, workload in enumerate(score_names):
        if workload not in baseline:
            raise RuntimeError(f"baseline report lacks {workload}")
        baseline_item = baseline[workload]
        item = {
            "workload": workload,
            "status": "candidate",
            "coverage_status": "reuse_distance_phase_covered",
            "feature_distance": 0.0,
            "reuse_signature": signatures[f"score:{workload}"],
            "cold_baseline_source": str(args.cold_baseline),
        }
        for target in TARGETS:
            cold_pred = float(baseline_item[f"{target}_predicted"])
            ratio = float(ratios[target][index])
            predicted = max(cold_pred * ratio, 0.0)
            low_ratio, high_ratio = ratio_intervals[target]
            actual = _float(warm_score[workload], target, float("nan"))
            item[f"{target}_cold_predicted"] = cold_pred
            item[f"{target}_warm_cold_ratio_predicted"] = ratio
            item[f"{target}_predicted"] = predicted
            item[f"{target}_actual"] = actual if math.isfinite(actual) else None
            item[f"{target}_interval_low"] = cold_pred * float(low_ratio[index]) / max(ratio, 1e-12)
            item[f"{target}_interval_high"] = cold_pred * float(high_ratio[index]) / max(ratio, 1e-12)
            item[f"{target}_ape"] = abs(predicted - actual) / actual if math.isfinite(actual) and actual > 0 else None
        results.append(item)
    metrics = {}
    for target in TARGETS:
        apes = [item[f"{target}_ape"] for item in results if item[f"{target}_ape"] is not None]
        metrics[target] = {"count": len(apes), "mape": float(np.mean(apes)) if apes else None, "max_ape": float(np.max(apes)) if apes else None}
    report = {
        "schema": "sm86_reuse_distance_phase_projection_v1",
        "train_cold_campaign": str(args.train_cold_campaign),
        "train_warm_campaign": str(args.train_warm_campaign),
        "score_cold_campaign": str(args.score_cold_campaign),
        "score_warm_campaign": str(args.score_warm_campaign),
        "cold_baseline": str(args.cold_baseline),
        "holdout_family": args.holdout_family,
        "train_workloads": train_names,
        "score_workloads": score_names,
        "target_labels_score_only": True,
        "model": {
            "trace_signature": "per-threadblock global 32-byte sector reuse distance via Fenwick stack distance",
            "phase_fit": "ridge on log(warm/cold NCU endpoint ratio), opposite workload cohort only",
            "cold_endpoint": "precomputed score-label-free baseline prediction",
            "interval": "training log-ratio q90 envelope; not confidence",
            "coverage": "all paired score rows receive candidate; no OOD refusal",
        },
        "feature_names": names,
        "ridge": args.ridge,
        "residual_log_ratio_q90": residual_q90,
        "metrics": metrics,
        "model_seconds": time.perf_counter() - started,
        "projection_only_speedup_vs_replay_wall": sum(_float(warm_score[name], "replay_wall_seconds") for name in score_names) / max(time.perf_counter() - started, 1e-12),
        "coefficients": coefficients,
        "signatures": signatures,
        "predictions": results,
        "limitations": [
            "per-threadblock order is not a global SM/L2 ownership trace",
            "reuse distance is a cache-pressure signature, not line owner/read-from/DRAM completion",
            "paired training cohorts are related-family holdouts, not architecture-wide generalization",
            "baseline timing domains remain distinct and NCU score values are loaded only for final APE",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cold-campaign", type=Path, required=True)
    parser.add_argument("--train-warm-campaign", type=Path, required=True)
    parser.add_argument("--score-cold-campaign", type=Path, required=True)
    parser.add_argument("--score-warm-campaign", type=Path, required=True)
    parser.add_argument("--cold-baseline", type=Path, required=True)
    parser.add_argument("--holdout-family", choices=("base", "large"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=1.0)
    args = parser.parse_args()
    if not math.isfinite(args.ridge) or args.ridge <= 0.0:
        parser.error("--ridge must be positive and finite")
    report = run(args)
    print(json.dumps({"holdout_family": report["holdout_family"], "metrics": report["metrics"], "score_rows": len(report["score_workloads"]), "model_seconds": report["model_seconds"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
