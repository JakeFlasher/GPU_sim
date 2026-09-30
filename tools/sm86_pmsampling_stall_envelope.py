#!/usr/bin/env python3
"""Interpolate statistical PM-sampling stall populations conservatively.

The input matrix must contain independent shape rows. A target observation may be
provided only for final comparison; it is never used in the interpolation.
Outputs are sampled stall envelopes, not per-PC pipe/issue traces.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

STALLS = (
    "pmsampling:smsp__warps_issue_stalled_long_scoreboard.avg",
    "pmsampling:smsp__warps_issue_stalled_wait.avg",
    "pmsampling:smsp__warps_issue_stalled_short_scoreboard.avg",
    "pmsampling:smsp__warps_issue_stalled_branch_resolving.avg",
    "pmsampling:smsp__warps_issue_stalled_barrier.avg",
    "pmsampling:smsp__warps_issue_stalled_math_pipe_throttle.avg",
)


def num(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--target-sk", type=float, required=True)
    parser.add_argument("--target-observation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.matrix.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 3:
        raise RuntimeError("at least three independent PM-sampling rows are required")
    if any("target" in str(row.get("role", "")).lower() for row in rows):
        raise RuntimeError("target-labelled PM-sampling row is not allowed in the matrix")
    rows.sort(key=lambda row: num(row["Sk"]))
    x = np.asarray([num(row["Sk"]) for row in rows], dtype=float)
    center = float(np.mean(x))
    scale = max(float(np.ptp(x)), 1.0)
    xn = (x - center) / scale
    target_x = (args.target_sk - center) / scale
    if not (x.min() <= args.target_sk <= x.max()):
        raise RuntimeError("target Sk is outside the independent interpolation hull")

    result = {
        "schema": "sm86_pmsampling_stall_envelope_v1",
        "matrix": str(args.matrix),
        "target_sk": args.target_sk,
        "degree": min(2, len(rows) - 1),
        "independent_anchor_count": len(rows),
        "target_labels_excluded_from_fit": True,
        "predictions": {},
        "anchor_intervals": {},
    }
    for metric in STALLS:
        y = np.asarray([num(row[metric]) for row in rows], dtype=float)
        degree = min(2, len(rows) - 1)
        coeff = np.polyfit(xn, y, degree)
        pred = float(np.polyval(coeff, target_x))
        fit = np.polyval(coeff, xn)
        result["predictions"][metric] = {
            "value": pred,
            "fit_mae": float(np.mean(np.abs(fit - y))),
            "fit_max_abs_error": float(np.max(np.abs(fit - y))),
            "anchor_min": float(np.min(y)),
            "anchor_max": float(np.max(y)),
            "coefficients": coeff.tolist(),
        }
    if args.target_observation is not None:
        observation = json.loads(args.target_observation.read_text())
        result["target_observation"] = observation
        result["target_ape"] = {}
        for metric in STALLS:
            actual = num(str(observation.get(metric, 0.0)))
            pred = result["predictions"][metric]["value"]
            result["target_ape"][metric] = abs(pred - actual) / actual if actual else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
