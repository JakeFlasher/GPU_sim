"""Compute shadow headline diagnostics from a ``comparison_bundle.json``.

The helper exposes per-round visibility into the four numbers that the
final headline-validity guard will eventually fail-closed on:

  - per-cell prediction stddev across admitted cells;
  - the microbench-admitted cell identities (slugs whose
    ``(slug, shape_point)`` falls in the 8-cell microbench subset of
    ``_PREDICTOR_DESIGN_COVERAGE_BY_CELL``);
  - excluded-reason counts (keyed by typed diagnostic);
  - geomean CI width.

This module is the SCAFFOLDING for the eventual guard. The numbers it
emits are visible per round summary; no fail-closed teeth land here.
The guard with teeth lands in a later round once the upstream
predictor + microbench rehabilitation work has completed.

CLI:

  python3 tools/local_labels/shadow_headline_diagnostics.py \
      <bundle_path> [--format markdown|json]

Library API:

  from shadow_headline_diagnostics import compute_from_bundle, format_block_markdown
  block = compute_from_bundle(Path(".../comparison_bundle.json"))
  print(format_block_markdown(block))
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


# The eight microbench slugs named in the comparison-runner's
# ``_PREDICTOR_DESIGN_COVERAGE_BY_CELL`` mapping. Three friendly,
# five hostile. Kept here as a frozenset of SLUGS — not
# ``(slug, shape_point)`` tuples — so the diagnostic counts
# microbench coverage over the durable 8-slug population rather
# than a frozen shape-point set. The DEC-3-resolved microbench
# rehabilitation introduces new shape variants (e.g.
# ``trace_l2_accuracy_xl``) on the same slugs; a tuple identity
# would silently report ``0 of 8`` on the rehabilitated bundles
# and weaken the eventual headline-validity coverage threshold.
MICROBENCH_SLUGS_8: frozenset[str] = frozenset({
    "working_set_capacity_sweep",
    "l2_thrash_pointer_chase",
    "dram_stream_stride_mix",
    "carveout_sensitivity_probe",
    "l2_persisting_window_sweep",
    "l2_prefetch_hint_sweep",
    "warm_cold_pair_relaunch",
    "register_spill_local_mem",
})


def _safe_pop_diagnostic(cell: dict[str, Any]) -> str:
    """Return the typed diagnostic/reason key for an excluded-cell record.

    The comparison-runner records excluded cells with either a top-level
    ``diagnostic`` field (typed) or a fallback ``reason`` field. Both
    forms exist in committed bundles. If neither is present, fall back
    to ``unknown`` rather than dropping the record silently.
    """
    raw = cell.get("diagnostic")
    if raw is None:
        raw = cell.get("reason")
    if raw is None:
        return "unknown"
    return str(raw)


def _stddev_population(values: list[float]) -> float:
    """Population stddev. Returns 0.0 for fewer than two samples."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def _variance_population(values: list[float]) -> float:
    """Population variance. Returns 0.0 for fewer than two samples.

    Plan AC-T10 requires the shadow block to report per-cell
    prediction VARIANCE (not just stddev). The two are
    complementary surfaces: variance is the plan-named
    invariant; stddev is kept alongside for human-friendly
    visibility at audit time.
    """
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def compute_from_bundle(
    bundle_path: Path,
    calibration_evaluation_sidecar: dict[str, Any] | None = None,
    calibration_evaluation_sidecar_path: Path | str | None = None,
) -> dict[str, Any]:
    """Compute the shadow-diagnostic fields from a comparison bundle.

    The returned dict has a stable shape and is byte-deterministic for
    the same input bundle file (sorted lists; numeric values rounded
    to 6 decimal places). The scanner test uses this stability to
    detect summary-vs-bundle drift.

    When ``calibration_evaluation_sidecar`` is provided, the helper
    additionally populates ``calibration_loocv_p90_per_head`` from
    the sidecar's ``loocv_table`` so the round summary can surface
    every fitted head's LOOCV p90 residual at audit time. The
    ``arch_set_includes`` field carries the sorted list of arches
    present under ``per_arch_diagnostics.per_arch`` (empty when the
    bundle predates the per-arch shape).
    """
    with bundle_path.open("r", encoding="utf-8") as fh:
        bundle = json.load(fh)

    included = bundle.get("included_cells") or []
    excluded = bundle.get("excluded_cells") or []

    predicted: list[float] = []
    # AC-T10 requires microbench identities as (slug,
    # shape_point) per cell, not just slug. Two admitted cells
    # on the same slug at different shape points must surface
    # as TWO distinct identities. The rendered shape is
    # ``slug@shape_point``; sorting on the tuple keeps the
    # output byte-deterministic across re-runs.
    microbench_admitted: set[tuple[str, str]] = set()
    for cell in included:
        join_key = cell.get("join_key") or {}
        slug = join_key.get("slug")
        shape_point = join_key.get("shape_point")
        pred = cell.get("predicted_hit_rate")
        if pred is not None:
            predicted.append(float(pred))
        if (
            slug is not None
            and shape_point is not None
            and slug in MICROBENCH_SLUGS_8
        ):
            microbench_admitted.add((str(slug), str(shape_point)))

    excluded_counts: dict[str, int] = {}
    for cell in excluded:
        key = _safe_pop_diagnostic(cell)
        excluded_counts[key] = excluded_counts.get(key, 0) + 1

    per_arch = (
        bundle.get("per_arch_diagnostics") or {}
    ).get("per_arch") or {}
    arch_set_includes = sorted(per_arch.keys())

    ci_low = bundle.get("geomean_ci_low")
    ci_high = bundle.get("geomean_ci_high")
    if ci_low is None or ci_high is None:
        # Fall back to the per-arch sm_86 CI when top-level CI fields
        # are absent. The active comparison bundle stores per-arch
        # geomean CI under
        # ``per_arch_diagnostics.per_arch.sm_86.relative_error_geomean_ci``;
        # the top-level keys are only emitted when the comparison
        # runner produces an aggregate sm_86+sm_120 headline. Reading
        # the per-arch path keeps the shadow diagnostic guard-
        # consistent with the eventual headline-validity guard, which
        # also gates on the per-arch CI for sm_86-only loops.
        sm_86 = per_arch.get("sm_86") or {}
        per_arch_ci = sm_86.get("relative_error_geomean_ci") or {}
        ci_low = per_arch_ci.get("ci_low")
        ci_high = per_arch_ci.get("ci_high")
    if ci_low is None or ci_high is None:
        ci_width = 0.0
    else:
        ci_width = float(ci_high) - float(ci_low)

    stddev = _stddev_population(predicted)
    variance = _variance_population(predicted)
    microbench_identities = sorted(
        f"{slug}@{shape_point}"
        for slug, shape_point in microbench_admitted
    )

    calibration_loocv: dict[str, float] = {}
    if calibration_evaluation_sidecar is not None:
        loocv_table = (
            calibration_evaluation_sidecar.get("loocv_table") or {}
        )
        for head, row in loocv_table.items():
            if not isinstance(row, dict):
                continue
            p90 = row.get("p90")
            if isinstance(p90, bool) or not isinstance(p90, (int, float)):
                continue
            calibration_loocv[str(head)] = round(float(p90), 6)

    block: dict[str, Any] = {
        "bundle_path": str(bundle_path),
        "admitted_cell_count": len(predicted),
        "per_cell_prediction_stddev": round(stddev, 6),
        "per_cell_prediction_variance": round(variance, 6),
        "microbench_admitted_cells": microbench_identities,
        "microbench_admitted_count": len(microbench_identities),
        "microbench_total": len(MICROBENCH_SLUGS_8),
        "excluded_reason_counts": dict(sorted(excluded_counts.items())),
        "ci_width": round(ci_width, 6),
        "arch_set_includes": arch_set_includes,
        "calibration_loocv_p90_per_head": dict(sorted(calibration_loocv.items())),
    }
    if calibration_evaluation_sidecar_path is not None:
        block["calibration_sidecar_path"] = str(
            calibration_evaluation_sidecar_path,
        )
    return block


def format_block_markdown(block: dict[str, Any]) -> str:
    """Render the diagnostic dict as a markdown block.

    The scanner parses this exact shape (see the regexes in
    ``test_round_summary_shadow_diagnostics_scanner.py``). Adding or
    re-ordering fields requires updating the scanner in lockstep.

    The ``arch_set_includes`` line + ``Calibration LOOCV p90 per
    head:`` block were added for the AC-T10 Tier-2 bundle-backed
    extension. Both lines always appear; they emit ``(none)`` when
    the corresponding field is empty so the rendered shape is
    deterministic across bundle vintages.
    """
    microbench_admitted = block["microbench_admitted_cells"]
    if microbench_admitted:
        microbench_list = ", ".join(microbench_admitted)
    else:
        microbench_list = "(none)"
    arch_set_includes = block.get("arch_set_includes") or []
    if arch_set_includes:
        arch_list = ", ".join(arch_set_includes)
    else:
        arch_list = "(none)"
    excluded_counts = block["excluded_reason_counts"]
    if excluded_counts:
        excluded_lines = "\n".join(
            f"  - `{reason}`: {count}" for reason, count in excluded_counts.items()
        )
    else:
        excluded_lines = "  - (none)"
    calibration_p90 = block.get("calibration_loocv_p90_per_head") or {}
    if calibration_p90:
        calibration_lines = "\n".join(
            f"  - {head}: {value}"
            for head, value in calibration_p90.items()
        )
    else:
        calibration_lines = "  - (none)"
    sidecar_path = block.get("calibration_sidecar_path")
    if sidecar_path:
        sidecar_line = f"- Calibration sidecar: `{sidecar_path}`\n"
    else:
        sidecar_line = ""
    return (
        "## Shadow Headline Diagnostics\n"
        "\n"
        f"- Source bundle: `{block['bundle_path']}`\n"
        f"{sidecar_line}"
        f"- Admitted cell count: {block['admitted_cell_count']}\n"
        f"- Per-cell prediction stddev: {block['per_cell_prediction_stddev']}\n"
        f"- Per-cell prediction variance: {block['per_cell_prediction_variance']}\n"
        f"- Microbench admitted count: {block['microbench_admitted_count']} of {block['microbench_total']}\n"
        f"- Microbench admitted cells: {microbench_list}\n"
        f"- Arch set includes: {arch_list}\n"
        f"- Excluded reason counts:\n"
        f"{excluded_lines}\n"
        f"- CI width: {block['ci_width']}\n"
        f"- Calibration LOOCV p90 per head:\n"
        f"{calibration_lines}\n"
    )


_VALUE_KEYS = (
    "bundle_path",
    "admitted_cell_count",
    "per_cell_prediction_stddev",
    "per_cell_prediction_variance",
    "microbench_admitted_count",
    "microbench_admitted_cells",
    "excluded_reason_counts",
    "ci_width",
)


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="shadow_headline_diagnostics.py",
        description=(
            "Compute per-round shadow headline diagnostics from a "
            "comparison_bundle.json. Use --format markdown for the "
            "block emitted into round summaries; --format json for "
            "machine-readable output (e.g., a downstream guard test)."
        ),
    )
    parser.add_argument(
        "bundle_path",
        type=Path,
        help="Path to a comparison_bundle.json artifact.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format. Default: markdown.",
    )
    parser.add_argument(
        "--calibration-evaluation-sidecar",
        type=Path,
        default=None,
        help=(
            "Optional path to a calibration_evaluation_v1.json "
            "sidecar produced by "
            "tools/local_labels/fit_calibration_head.py. When "
            "provided, the parsed sidecar's loocv_table is read "
            "for the per-head LOOCV p90 residual block AND the "
            "sidecar path itself is rendered as the "
            "'Calibration sidecar:' line so a downstream scanner "
            "can re-load the sidecar for drift detection."
        ),
    )
    args = parser.parse_args(argv)
    if not args.bundle_path.is_file():
        print(
            f"error: bundle_path does not exist: {args.bundle_path}",
            file=sys.stderr,
        )
        return 2
    sidecar_payload: dict[str, Any] | None = None
    sidecar_path: Path | None = args.calibration_evaluation_sidecar
    if sidecar_path is not None:
        if not sidecar_path.is_file():
            print(
                f"error: --calibration-evaluation-sidecar path does "
                f"not exist: {sidecar_path}",
                file=sys.stderr,
            )
            return 2
        try:
            with sidecar_path.open("r", encoding="utf-8") as fh:
                sidecar_payload = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"error: --calibration-evaluation-sidecar payload "
                f"unreadable ({sidecar_path}): {exc!s}",
                file=sys.stderr,
            )
            return 2
    block = compute_from_bundle(
        args.bundle_path,
        calibration_evaluation_sidecar=sidecar_payload,
        calibration_evaluation_sidecar_path=sidecar_path,
    )
    if args.format == "json":
        print(json.dumps(block, indent=2, sort_keys=True))
    else:
        print(format_block_markdown(block), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
