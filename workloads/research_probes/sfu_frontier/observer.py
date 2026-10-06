#!/usr/bin/env python3
"""Empirical SFU grid-batch observer, separate from native replay timing.

Fit a line on workload medians with x = ceil(blocks / SM count). This is an
empirical service-batch feature, NOT CUDA occupancy or physical CTA waves.
Interval calibration uses separate workload shapes, not repeated executions
as independent shapes. The split-calibration rank has a coverage interpretation
only under exchangeability, which an extrapolation experiment does not prove.
No scored target measurements or simulator timing/card parameters are inputs.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import statistics


def positive_int(value: object, label: str) -> int:
    if type(value) is not int or not 1 <= value <= 65536:
        raise ValueError(f"{label} must be an integer in 1..65536")
    return value


def workload_medians(rows: list[dict]) -> dict[int, float]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("workload rows must be a nonempty list")
    result = {}
    for row in rows:
        blocks = positive_int(row["blocks"], "blocks")
        if blocks in result:
            raise ValueError(f"duplicate workload shape: {blocks} blocks")
        samples = row["cycles"]
        if not isinstance(samples, list) or not samples:
            raise ValueError("cycles must be a nonempty list")
        if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0
               for x in samples):
            raise ValueError("cycles must be positive finite numbers")
        result[blocks] = float(statistics.median(samples))
    return result


def predict(fit_rows: list[dict], interval_rows: list[dict], blocks: int,
            sm_count: int, alpha: str = "0.1") -> dict:
    blocks = positive_int(blocks, "target blocks")
    sm_count = positive_int(sm_count, "SM count")
    miscoverage = Fraction(alpha)
    if not 0 < miscoverage < 1:
        raise ValueError("alpha must be between zero and one")
    fit = workload_medians(fit_rows)
    interval = workload_medians(interval_rows)
    if fit.keys() & interval.keys() or blocks in fit or blocks in interval:
        raise ValueError("fit, interval calibration, and target shapes must be disjoint")
    feature = lambda b: (b + sm_count - 1) // sm_count
    xs, ys = [feature(b) for b in fit], list(fit.values())
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        raise ValueError("fit needs at least two distinct grid-batch feature values")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    intercept = mean_y - slope * mean_x
    target_feature = feature(blocks)
    point = intercept + slope * target_feature
    residuals = {b: abs(y - intercept - slope * feature(b)) for b, y in interval.items()}
    if not all(math.isfinite(x) for x in [point, slope, intercept, *residuals.values()]):
        raise ValueError("observer arithmetic produced non-finite values")
    if point <= 0:
        raise ValueError("observer predicted nonpositive cycles")
    # Do not clamp a rank of n+1 to n: too few workload shapes give no finite
    # split-calibration radius. Repeating a workload does not increase n.
    rank = math.ceil((len(interval) + 1) * (1 - miscoverage))
    radius = sorted(residuals.values())[rank - 1] if rank <= len(interval) else None
    return {
        "feature": "ceil_grid_blocks_over_sm_count_empirical_not_hardware_waves",
        "sm_count": sm_count,
        "fit_blocks": list(fit),
        "interval_calibration_blocks": list(interval),
        "target_blocks": blocks,
        "target_feature": target_feature,
        "extrapolates_fit_feature_range": not min(xs) <= target_feature <= max(xs),
        "intercept_cycles": intercept,
        "slope_cycles_per_grid_batch": slope,
        "point_cycles": point,
        "alpha": str(miscoverage),
        "interval_calibration_shape_count": len(interval),
        "interval_calibration_rank": rank,
        "interval_calibration_residuals_cycles": residuals,
        "radius_cycles": radius,
        "interval_lower_cycles": max(0.0, point - radius) if radius is not None else 0.0,
        "interval_upper_cycles": point + radius if radius is not None else None,
        "interval_target": "median_gpc_cycles_for_workload_shape",
        "coverage_guarantee_for_designed_extrapolation": False,
        "semantic_completion_upper": None,
    }


def _solve_normal_equations(design: list[list[float]], values: list[float]) -> list[float]:
    """Solve a tiny checked least-squares system without a numerical package."""
    width = len(design[0])
    normal = [[0.0] * (width + 1) for _ in range(width)]
    for row, value in zip(design, values):
        for i in range(width):
            for j in range(width):
                normal[i][j] += row[i] * row[j]
            normal[i][width] += row[i] * value
    for pivot in range(width):
        best = max(range(pivot, width), key=lambda i: abs(normal[i][pivot]))
        if abs(normal[best][pivot]) <= 1e-12:
            raise ValueError("observer feature design is rank deficient")
        normal[pivot], normal[best] = normal[best], normal[pivot]
        scale = normal[pivot][pivot]
        for column in range(pivot, width + 1):
            normal[pivot][column] /= scale
        for row in range(width):
            if row == pivot:
                continue
            factor = normal[row][pivot]
            for column in range(pivot, width + 1):
                normal[row][column] -= factor * normal[pivot][column]
    result = [normal[i][width] for i in range(width)]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("observer fit produced non-finite coefficients")
    return result


def predict_wave(fit_rows: list[dict], interval_rows: list[dict], blocks: int,
                 sm_count: int, alpha: str = "0.1") -> dict:
    """Fit a block-count + empirical batch feature model.

    The second feature is only ``ceil(blocks / sm_count)``. It is deliberately
    named *empirical batch* rather than a hardware wave or occupancy fact.
    """
    blocks = positive_int(blocks, "target blocks")
    sm_count = positive_int(sm_count, "SM count")
    miscoverage = Fraction(alpha)
    if not 0 < miscoverage < 1:
        raise ValueError("alpha must be between zero and one")
    fit = workload_medians(fit_rows)
    interval = workload_medians(interval_rows)
    if fit.keys() & interval.keys() or blocks in fit or blocks in interval:
        raise ValueError("fit, interval calibration, and target shapes must be disjoint")
    feature = lambda value: (value + sm_count - 1) // sm_count
    design = [[1.0, float(value), float(feature(value))] for value in fit]
    coefficients = _solve_normal_equations(design, list(fit.values()))
    point_row = [1.0, float(blocks), float(feature(blocks))]
    point = sum(a * b for a, b in zip(coefficients, point_row))
    residuals = {
        value: abs(observed - sum(a * b for a, b in zip(
            coefficients, [1.0, float(value), float(feature(value))])))
        for value, observed in interval.items()
    }
    if not all(math.isfinite(value) for value in [point, *coefficients,
                                                  *residuals.values()]):
        raise ValueError("observer arithmetic produced non-finite values")
    if point <= 0:
        raise ValueError("observer predicted nonpositive cycles")
    rank = math.ceil((len(interval) + 1) * (1 - miscoverage))
    radius = sorted(residuals.values())[rank - 1] if rank <= len(interval) else None
    return {
        "feature": "grid_blocks_plus_ceil_grid_batches_empirical_not_hardware_waves",
        "sm_count": sm_count,
        "fit_blocks": list(fit),
        "interval_calibration_blocks": list(interval),
        "target_blocks": blocks,
        "target_feature": feature(blocks),
        "coefficients": {"intercept_cycles": coefficients[0],
                         "cycles_per_block": coefficients[1],
                         "cycles_per_empirical_batch": coefficients[2]},
        "point_cycles": point,
        "alpha": str(miscoverage),
        "interval_calibration_shape_count": len(interval),
        "interval_calibration_rank": rank,
        "interval_calibration_residuals_cycles": residuals,
        "radius_cycles": radius,
        "interval_lower_cycles": max(0.0, point - radius) if radius is not None else 0.0,
        "interval_upper_cycles": point + radius if radius is not None else None,
        "coverage_guarantee_for_designed_extrapolation": False,
        "semantic_completion_upper": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--blocks", required=True, type=int)
    parser.add_argument("--sm-count", required=True, type=int)
    parser.add_argument("--alpha", default="0.1")
    args = parser.parse_args()
    try:
        data = json.loads(args.calibration.read_text())
        result = predict(data["fit"], data["interval_calibration"], args.blocks,
                         args.sm_count, args.alpha)
    except (ValueError, KeyError, TypeError, OSError, OverflowError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
