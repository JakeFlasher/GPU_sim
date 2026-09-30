#!/usr/bin/env python3
"""PM-state-conditioned endpoint envelope for SM86.

Independent PM-sampling anchors are converted into a compact state vector:
long-scoreboard, wait, short-scoreboard, branch, barrier, and math-throttle
shares plus total sampled stall population.  A ridge surface predicts endpoint
cost from that state and shape work.  The model treats samples as aggregate
phase emissions, never as physical per-PC route tags.  Unmatched score rows
use the independent frontier fallback so coverage remains complete.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np

from sm86_frontier_fallback import fit_frontier_fallback
from sm86_ncu_projection_surrogate import fit_ridge, load_rows, _float


STALLS = (
    "pmsampling:smsp__warps_issue_stalled_long_scoreboard.avg",
    "pmsampling:smsp__warps_issue_stalled_wait.avg",
    "pmsampling:smsp__warps_issue_stalled_short_scoreboard.avg",
    "pmsampling:smsp__warps_issue_stalled_branch_resolving.avg",
    "pmsampling:smsp__warps_issue_stalled_barrier.avg",
    "pmsampling:smsp__warps_issue_stalled_math_pipe_throttle.avg",
)
TARGETS = ("ncu_gpc_cycles", "ncu_duration_ns")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _matrix(path: Path) -> list[dict[str, str]]:
    rows = _rows(path)
    if not rows:
        raise RuntimeError(f"empty PM matrix: {path}")
    for row in rows:
        role = str(row.get("role", "")).lower()
        if "score_target" in role or role == "target":
            raise RuntimeError("PM matrix contains a score target")
        for key in ("BH", "Sq", "Sk", "D", *STALLS, "gpc__cycles_elapsed.max", "gpu__time_duration.sum", "smsp__inst_executed.sum"):
            if not math.isfinite(_float(row, key, float("nan"))):
                raise RuntimeError(f"non-finite PM field {key}")
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


def _state_rows(rows: list[dict[str, str]], target_sk: float) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    sk = np.asarray([_float(row, "Sk") for row in rows], dtype=float)
    if len(rows) < 3 or not (float(np.min(sk)) <= target_sk <= float(np.max(sk))):
        return None
    raw = np.asarray([[_float(row, key) for key in STALLS] for row in rows], dtype=float)
    total = np.maximum(raw.sum(axis=1), 1e-9)
    shares = raw / total[:, None]
    instructions = np.asarray([_float(row, "smsp__inst_executed.sum") for row in rows], dtype=float)
    x = (sk - np.mean(sk)) / max(float(np.ptp(sk)), 1.0)
    tx = (target_sk - np.mean(sk)) / max(float(np.ptp(sk)), 1.0)
    state = np.asarray([np.polyval(np.polyfit(x, shares[:, j], 2), tx) for j in range(len(STALLS))])
    stall_total = float(np.polyval(np.polyfit(x, total, 2), tx))
    inst = float(np.polyval(np.polyfit(x, instructions, 2), tx))
    state = np.maximum(state, 0.0)
    state /= max(float(np.sum(state)), 1e-9)
    return state, np.asarray([stall_total, math.log(max(inst, 1.0))], dtype=float), sk


def _predict_exact(spec: dict | None, matrix: list[dict[str, str]]) -> dict | None:
    if spec is None or any(key not in spec for key in ("BH", "Sq", "Sk", "D")):
        return None
    target = {key: _float({key: str(spec[key])}, key, float("nan")) for key in ("BH", "Sq", "Sk", "D")}
    rows = [
        row for row in matrix
        if all(_float(row, key) == target[key] for key in ("BH", "Sq", "D"))
    ]
    rows.sort(key=lambda row: _float(row, "Sk"))
    state_info = _state_rows(rows, target["Sk"])
    if state_info is None:
        return None
    state, compact, sk = state_info
    x_values = []
    for row in rows:
        raw = np.asarray([_float(row, key) for key in STALLS], dtype=float)
        shares = raw / max(float(raw.sum()), 1e-9)
        x_values.append([1.0, *shares, raw.sum(), math.log(max(_float(row, "smsp__inst_executed.sum"), 1.0))])
    x_values = np.asarray(x_values, dtype=float)
    target_x = np.asarray([[1.0, *state, *compact]], dtype=float)
    predictions: dict[str, float] = {}
    intervals: dict[str, tuple[float, float]] = {}
    for target_name, source_name, scale in (
        ("ncu_gpc_cycles", "gpc__cycles_elapsed.max", 1.0),
        ("ncu_duration_ns", "gpu__time_duration.sum", 1000.0),
    ):
        y = np.asarray([_float(row, source_name) * scale for row in rows], dtype=float)
        beta, fit_scale = fit_ridge(x_values, y, 1e-3)
        train_pred = (x_values / fit_scale) @ beta
        pred = float(((target_x / fit_scale) @ beta).reshape(-1)[0])
        residual = float(np.quantile(np.abs(train_pred - y), 0.90))
        predictions[target_name] = max(pred, 0.0)
        intervals[target_name] = (max(pred - residual, 0.0), pred + residual)
    return {
        "status": "candidate",
        "path": "independent_pm_state_envelope",
        "coverage_status": "pm_exact_shape",
        "anchor_count": len(rows),
        "anchor_low_Sk": float(sk[0]),
        "anchor_high_Sk": float(sk[-1]),
        "target_Sk": target["Sk"],
        "state_shares": {name: float(value) for name, value in zip(STALLS, state)},
        "predictions": predictions,
        "intervals": intervals,
    }


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    matrix = _matrix(args.pm_matrix)
    specs = _specs(args.shape_specs)
    score_rows = load_rows(args.score_campaign)
    train_rows = load_rows(args.fallback_train_campaign)
    if args.holdout_family:
        score_rows = [r for r in score_rows if r["workload"].split("__", 1)[0] == args.holdout_family]
        train_rows = [r for r in train_rows if r["workload"].split("__", 1)[0] != args.holdout_family]
    fallback = fit_frontier_fallback(args.fallback_train_campaign, args.score_campaign, train_rows, score_rows)
    output_rows = []
    for index, row in enumerate(score_rows):
        exact = _predict_exact(_spec_for(specs, row["workload"]), matrix)
        if exact is None:
            prediction = {
                "status": "candidate_fallback",
                "path": "independent_frontier_trace_fallback",
                "coverage_status": "fallback_covered",
                "predictions": {target: float(fallback["predictions"][target][index]) for target in TARGETS},
                "intervals": {
                    target: (float(fallback["intervals"][target][0][index]), float(fallback["intervals"][target][1][index]))
                    for target in TARGETS
                },
            }
        else:
            prediction = exact
        item = {"workload": row["workload"], **prediction}
        for target in TARGETS:
            actual = _float(row, target, float("nan")); pred = prediction["predictions"][target]
            low, high = prediction["intervals"][target]
            item[f"{target}_actual"] = actual if math.isfinite(actual) else None
            item[f"{target}_predicted"] = pred
            item[f"{target}_interval_low"] = low; item[f"{target}_interval_high"] = high
            item[f"{target}_ape"] = abs(pred - actual) / actual if math.isfinite(actual) and actual > 0 else None
        output_rows.append(item)
    metrics = {}
    for target in TARGETS:
        actual = np.asarray([r[f"{target}_actual"] for r in output_rows]); pred = np.asarray([r[f"{target}_predicted"] for r in output_rows])
        metrics[target] = {"mae": float(np.mean(abs(pred - actual))), "mape": float(np.mean(abs(pred - actual) / actual)), "max_abs_error": float(np.max(abs(pred - actual)))}
    report = {
        "schema": "sm86_pm_state_envelope_v1",
        "pm_matrix": str(args.pm_matrix), "shape_specs": str(args.shape_specs),
        "score_campaign": str(args.score_campaign), "holdout_family": args.holdout_family,
        "target_labels_score_only": True,
        "model": {"state_vector": "normalized long/wait/short/branch/barrier/math stall shares plus total stalls and log instructions", "endpoint": "ridge state envelope on independent anchors", "coverage": "exact PM-shape envelope or frontier fallback", "physical_claim": "aggregate phase state only; no physical route identity"},
        "metrics": metrics, "model_seconds": time.perf_counter() - started, "predictions": output_rows,
        "limitations": ["PM anchor phase is warm-cache and shape-local", "sample states are aggregate emissions, not per-PC route tags", "cache ownership, ABI metadata, and unified endpoint remain latent"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2) + "\n"); return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pm-matrix", type=Path, required=True); parser.add_argument("--shape-specs", type=Path, required=True)
    parser.add_argument("--score-campaign", type=Path, required=True); parser.add_argument("--fallback-train-campaign", type=Path, required=True)
    parser.add_argument("--holdout-family", choices=("base", "large"), required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); report = run(args); print(json.dumps({"metrics": report["metrics"], "model_seconds": report["model_seconds"], "output": str(args.output)}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
