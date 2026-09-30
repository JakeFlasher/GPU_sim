#!/usr/bin/env python3
"""Fit a bounded CUDA-event -> NCU-duration endpoint transform.

This model addresses measurement-endpoint translation only. It does not equate
CUDA events with NCU semaphores, and it never uses a scored target NCU value in
its fit. Calibration rows must be independent anchors; a target observation is
accepted only as a final comparison record.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

TARGETS = ("ncu_duration_ns", "ncu_gpc_cycles")


def num(value: str) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else 0.0
    except (TypeError, ValueError):
        return 0.0


def load_matrix(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 3:
        raise RuntimeError("endpoint transform needs at least three independent rows")
    for row in rows:
        if "target" in str(row.get("role", "")).lower():
            raise RuntimeError("target-labelled row is not allowed in endpoint calibration")
    return rows


def run(args: argparse.Namespace) -> dict:
    rows = load_matrix(args.matrix)
    x = np.asarray([num(row["cuda_event_ns"]) for row in rows], dtype=float)
    x_center = float(np.mean(x))
    x_scale = max(float(np.ptp(x)), 1.0)
    xn = (x - x_center) / x_scale
    target_xn = (args.target_event_ns - x_center) / x_scale
    predictions = {}
    residuals = {}
    coefficients = {}
    for target in TARGETS:
        y = np.asarray([num(row[target]) for row in rows], dtype=float)
        degree = min(2, len(rows) - 1)
        coeff = np.polyfit(xn, y, degree)
        prediction = float(np.polyval(coeff, target_xn))
        fit = np.polyval(coeff, xn)
        predictions[target] = prediction
        residuals[target] = {
            "max_abs_calibration_residual": float(np.max(np.abs(fit - y))),
            "mae_calibration_residual": float(np.mean(np.abs(fit - y))),
        }
        coefficients[target] = coeff.tolist()
    report = {
        "schema": "sm86_cuda_event_to_ncu_endpoint_transform_v1",
        "matrix": str(args.matrix),
        "target_event_ns": args.target_event_ns,
        "degree": min(2, len(rows) - 1),
        "normalization": {"center": x_center, "scale": x_scale},
        "predictions": predictions,
        "calibration_residuals": residuals,
        "coefficients": coefficients,
        "target_ncu_excluded_from_fit": True,
        "limitations": [
            "CUDA-event and NCU endpoint semantics remain distinct",
            "cache phase must match the calibration matrix",
            "this is a local endpoint transform, not a unified clock model",
        ],
    }
    if args.target_observation is not None:
        observation = json.loads(args.target_observation.read_text())
        report["target_observation"] = observation
        for target in TARGETS:
            actual = num(str(observation[target]))
            report.setdefault("target_ape", {})[target] = (
                abs(predictions[target] - actual) / actual if actual else None
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--target-event-ns", type=float, required=True)
    parser.add_argument("--target-observation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
