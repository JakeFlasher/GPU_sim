#!/usr/bin/env python3
"""Predict NVBit trace size for a SimBench-BW shape point.

Two estimator sources are supported and persisted on the per-shape-point
record under `trace_shapes.<point>.predicted_bytes`:

1. static (default): a launch-config and declared-op-count model
   predicted_bytes = dynamic_warp_instructions_estimate * bytes_per_warp_record
                   + memory_reference_count       * bytes_per_memref_record

2. empirical_calibration: an `empirical_calibration_manifest.json` committed
   per-kernel records an empirical trace-byte measurement on a reference
   shape; the loader interpolates from that anchor for nearby shape points.

Exit codes:
    0  all validated shape points fit their declared trace_budget_mib.
    1  at least one shape point's predicted_bytes exceeds budget, or is stale
       relative to the static model when the shape file is the source of truth.
    2  usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any


SUITE_ROOT = Path(__file__).resolve().parent

# Canonical per-record sizes for the NVBit tracer at tools/nvbit_tracer/.
BYTES_PER_WARP_RECORD = 32
BYTES_PER_MEMREF_RECORD = 16

# Sensitivity factor for the static predictor: how many instructions we
# estimate a warp retires per declared op-count unit, per shape-axis product.
# The constant is deliberately conservative (high); the goal is to refuse
# admission to any shape point that would come near the budget envelope.
STATIC_WARP_INSTR_PER_OP = 4.0
STATIC_MEMREF_RATIO = 0.5


def _load_roster() -> dict[str, dict[str, Any]]:
    with open(SUITE_ROOT / "roster.toml", "rb") as f:
        return {r["slug"]: r for r in tomllib.load(f)["rows"]}


def _load_shape_file(slug: str) -> dict[str, Any] | None:
    path = SUITE_ROOT / "shapes" / f"{slug}.toml"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return tomllib.load(f)


def _shape_axis_product(axes: dict[str, Any]) -> int:
    """Product of every positive-integer leaf in the axis map, excluding
    axes that describe buffer sizes or per-launch configuration knobs
    rather than dynamic iteration counts.

    The estimator's predicted-trace-bytes formula is
    `dynamic_warp_instructions ~= STATIC_WARP_INSTR_PER_OP * axis_product`,
    so axes must reflect the per-warp dynamic instruction-count scale.
    Byte-valued axes (e.g. `working_set_bytes`, `allocation_bytes`,
    `persisting_window_bytes`, `eviction_bytes`) sweep the kernel's
    memory-allocation footprint without changing the per-warp
    instruction count; including them as multiplicative factors
    over-predicts trace volume by buffer-size orders of magnitude.
    Per-launch configuration knobs (e.g. `carveout_percent`,
    `prefetch_distance`) likewise tune kernel behavior without
    multiplying the dynamic instruction count. R49 excludes these
    name patterns so the static estimator matches what NVBit
    actually records on the wire.
    """
    total = 1
    for k, v in axes.items():
        if isinstance(k, str) and (
            k.endswith("_bytes")
            or k == "carveout_percent"
            or k == "prefetch_distance"
        ):
            continue
        if isinstance(v, int):
            total *= max(1, v)
        elif isinstance(v, dict):
            total *= _shape_axis_product(v)
    return total


def _launch_warp_count(launch_config: dict[str, Any] | None) -> int:
    if not launch_config:
        return 1
    block = launch_config.get("block", [1])
    grid = launch_config.get("grid", [1])
    threads_per_block = 1
    for d in block:
        threads_per_block *= d
    blocks_total = 1
    for d in grid:
        blocks_total *= d
    warps = (threads_per_block + 31) // 32
    return warps * max(1, blocks_total)


def static_predict_bytes(shape_point: dict[str, Any], declared_ops: int = 1) -> int:
    """Return the static predicted_bytes for a shape point."""
    axes = shape_point.get("axes", {})
    axis_product = _shape_axis_product(axes) if axes else 1
    warp_count = _launch_warp_count(shape_point.get("launch_config"))
    dynamic_warp_instructions = int(STATIC_WARP_INSTR_PER_OP * declared_ops * max(1, axis_product // max(1, warp_count)) * warp_count)
    memref_count = int(STATIC_MEMREF_RATIO * dynamic_warp_instructions)
    return dynamic_warp_instructions * BYTES_PER_WARP_RECORD + memref_count * BYTES_PER_MEMREF_RECORD


def empirical_predict_bytes(manifest_path: Path, shape_point: dict[str, Any]) -> int:
    """Load an empirical-calibration manifest and interpolate for shape_point.

    The manifest format is simple:
        {"schema_version": 1, "anchor_axes": {...}, "anchor_bytes": 12345,
         "scaling_exponent": 1.0}

    The predicted bytes for a new shape point is:
        anchor_bytes * (axis_product_new / axis_product_anchor) ** scaling_exponent
    """
    with open(manifest_path) as f:
        m = json.load(f)
    anchor_product = _shape_axis_product(m["anchor_axes"])
    new_product = _shape_axis_product(shape_point.get("axes", {}))
    ratio = new_product / max(1, anchor_product)
    exponent = m.get("scaling_exponent", 1.0)
    return int(m["anchor_bytes"] * (ratio ** exponent))


def validate_slug(slug: str) -> list[str]:
    """Validate every trace_shapes.<point>.predicted_bytes against the row's
    trace_budget_mib. Returns a list of error strings (empty on success)."""
    roster = _load_roster()
    if slug not in roster:
        return [f"unknown_slug: {slug}"]
    row = roster[slug]
    shape = _load_shape_file(slug)
    if shape is None:
        # No shape file yet: cannot validate, but not an error (kernel is still scaffold).
        return []
    budget_bytes = row["trace_budget_mib"] * 1024 * 1024
    errors: list[str] = []
    for point in shape.get("trace_shapes", []):
        name = point.get("name", "<?>")
        pred = point.get("predicted_bytes")
        if pred is None:
            errors.append(f"trace_prediction_stale: {slug}/{name} missing predicted_bytes")
            continue
        source = point.get("estimator_source", "static")
        if source == "static":
            recomputed = static_predict_bytes(point)
        else:
            cal_path = shape.get("shapes_dir", SUITE_ROOT / "shapes") / point.get("empirical_calibration_manifest", "")
            recomputed = empirical_predict_bytes(Path(cal_path), point)
        # Allow ±10% drift before flagging as stale.
        if abs(recomputed - pred) > 0.1 * max(1, pred):
            errors.append(f"trace_prediction_stale: {slug}/{name} predicted_bytes={pred} but estimator produces {recomputed}")
        if pred > budget_bytes:
            errors.append(f"trace_shape_overbudget: {slug}/{name} predicted_bytes={pred} > budget_bytes={budget_bytes}")
    return errors


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SimBench-BW trace-size predictor")
    p.add_argument("--validate", metavar="SLUG",
                   help="Re-evaluate every trace_shape point for SLUG and fail on stale / overbudget predictions")
    p.add_argument("--validate-all", action="store_true",
                   help="Run --validate across every roster row with a shape file")
    args = p.parse_args(argv)

    if not (args.validate or args.validate_all):
        p.error("specify --validate SLUG or --validate-all")

    errors: list[str] = []
    if args.validate:
        errors.extend(validate_slug(args.validate))
    if args.validate_all:
        roster = _load_roster()
        for slug in roster:
            errors.extend(validate_slug(slug))

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        print(f"{len(errors)} trace-size error(s)", file=sys.stderr)
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
