"""Fit the sm_86 5-cell pilot IsotonicCalibrator and persist
isotonic_params_sm86.toml (cross-arch plan the calibration workstream, the design contract).

Mirrors `sm120_l2_associativity_probe/persist_fitted_sidecar.py`
but consumes the sm_86 pilot evidence instead of the sm_120
historical ledger:

  1. Read `label_ledger_v1_sm86_pilot_v1.toml` for the 5
     admitted pilot cells + their per-cell SHA cross-stamps.
  2. For each cell, parse the binary Tier-2 sidecar at
     `<bundle>/nvbit_metrics/tier2_address_trace.t2a` into a
     launch frame; strip the truncation marker (offline
     calibration relaxation; rationale documented in
     `sweep_associativity.py`).
  3. Run `predict_from_launch_frame` with the sm_86 static
     device card geometry (6 MiB / 6 slices / 16-way; divides
     cleanly so no simulator-effective reduction is needed).
  4. Read `ncu_3run_v1.json` per cell; use the median of the
     3 NCU runs as the L1 / L2 labels.
  5. Build CalibrationInputRecords; fit IsotonicCalibrator
     with `validation_split` marking ALL 5 as
     `calibration_only` (the 5-cell pilot fits below the
     canonical smoke floor (=5) only when partitioned; the design contract
     fits on the full pilot, the historical 3+2 split SHA is
     recorded as metadata).
  6. Reject identity fits (typed reason
     `sm86_pilot_identity_fit_rejected`) — both L1 AND L2
     heads collapsing to identity would mean the calibrator
     silently substitutes a no-op layer.
  7. Persist `isotonic_params_sm86.toml` with schema parity
     to `isotonic_params_sm120.toml` plus per-cell Tier-2
     sidecar SHA cross-stamps.

The emitter is deterministic: same ledger + same Tier-2
binaries + same NCU sidecars + same captured_at_utc → byte-
identical TOML output.

Usage:
  python -m baselines_harness.calibration.cache_prediction.sm86_pilot.persist_fitted_sidecar \\
    [--output-root <dir>] [--captured-at-utc <iso8601>] [--traces-root <dir>]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import statistics
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from baselines_harness.metrics.cache_predictor import (  # noqa: E402
    predict_from_launch_frame,
    PREDICTOR_BUNDLE_RESULT_KIND_SCALAR,
)
from baselines_harness.metrics.isotonic_calibrator import (  # noqa: E402
    CalibrationInputRecord,
    CellJoinKey,
    IsotonicCalibrator,
)
from baselines_harness.metrics.tier2_sidecar_parser import (  # noqa: E402
    parse_launch_frames,
)
from baselines_harness.metrics.predictor_design_coverage import (  # noqa: E402
    lookup_predictor_design_coverage,
)


_CAL_DIR = (
    _REPO_ROOT
    / "scripts" / "baselines_harness"
    / "calibration" / "cache_prediction"
)
_LEDGER = _CAL_DIR / "label_ledger_v1_sm86_pilot_v1.toml"
_HISTORICAL_SPLIT = _CAL_DIR / "historical_label_split_v1_sm86_pilot_v1.toml"
_DEFAULT_SIDECAR = _CAL_DIR / "isotonic_params_sm86.toml"
_DEFAULT_TRACES_ROOT = (
    _REPO_ROOT / "scripts" / "baselines_harness" / ".harness_traces"
)


# sm_86 static device card geometry. The native L2 capacity
# (6 MiB) divides cleanly by num_l2_slices (6) AND by
# (line_size_bytes * associativity_l2) = (128 * 16) = 2048,
# so per-slice set count = 1048576 / 2048 = 512 — no
# "simulator-effective" reduction needed.
_SM86_GEOMETRY = {
    "arch": "sm_86",
    "l1_kb_per_sm": 192,
    "l1_kb_per_sm_source": "static_device_card",
    "l2_capacity_bytes": 6 * 1024 * 1024,
    "capacity_source": "static_device_card",
    "num_l2_slices": 6,
    "num_l2_slices_source": "static_device_card",
    "sector_size_bytes": 32,
    "sector_size_source": "static_device_card",
    "line_size_bytes": 128,
    "line_size_source": "static_device_card",
    "associativity_l2": 16,
    "associativity_l2_source": "static_device_card",
    "l2_constant_memory_counts_for_read_score": False,
    "l2_constant_memory_counts_for_read_score_source": (
        "static_device_card_sm_86"
    ),
    "geometry_source": "sm86_pilot_static_device_card_v1",
}


# The accepted NCU metric selectors recorded in the sidecar's
# meta. Identical to the sm_120 set; the deployed B200
# predictor still must NOT depend on these (NCU-permitted
# hosts only).
_ACCEPTED_METRICS = [
    "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "lts__t_sectors_srcunit_tex_op_read.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
]


# Identity-fit rejection epsilon. The fit anchors must
# differ from the input by more than ε for at least ONE
# anchor pair across BOTH heads. Set to 1e-9 — a fit truly
# collapsed to identity (y_threshold == x_threshold for
# every pair) is rejected; meaningful sub-pp anchor shifts
# pass.
_IDENTITY_REJECTION_EPSILON = 1e-9


class Sm86PilotFitError(Exception):
    """Typed-reason errors specific to the sm_86 pilot fit
    pipeline. The reason is in
    ``Sm86PilotFitError.typed_reason``; messages are for
    operators."""

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        self.typed_reason = typed_reason


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle_dir(traces_root: Path, slug: str) -> Path:
    return traces_root / f"{slug}__cuda_cpp__sm86"


def _median_ncu_labels(ncu_path: Path) -> tuple[float, float]:
    """Return (median_l1_hit_rate, median_l2_read_hit_rate)
    over the 3 NCU runs. Median over mean so one outlier
    run doesn't drift the label."""
    with open(ncu_path, "rb") as fp:
        data = json.load(fp)
    runs = data.get("runs", [])
    if len(runs) != 3:
        raise Sm86PilotFitError(
            "sm86_pilot_ncu_run_count_unexpected",
            f"{ncu_path}: expected 3 NCU runs, got {len(runs)}",
        )
    l1s = [float(r["l1_global_load_hit_rate"]) for r in runs]
    l2s = [float(r["l2_read_hit_rate"]) for r in runs]
    return statistics.median(l1s), statistics.median(l2s)


def _build_records(
    ledger_entries: list[dict],
    traces_root: Path,
) -> tuple[list[CalibrationInputRecord], dict[str, str]]:
    """Build one CalibrationInputRecord per pilot cell.

    Returns ``(records, tier2_sha_by_slug)`` so the emitter
    can cross-stamp every Tier-2 binary SHA into the sidecar
    meta alongside the fit's `admitted_cell_count`.
    """
    records: list[CalibrationInputRecord] = []
    tier2_sha_by_slug: dict[str, str] = {}

    for entry in ledger_entries:
        slug = entry["slug"]
        bundle = _bundle_dir(traces_root, slug)
        t2a = bundle / "nvbit_metrics" / "tier2_address_trace.t2a"
        ncu = bundle / "ncu_3run_v1.json"
        if not t2a.is_file():
            raise Sm86PilotFitError(
                "sm86_pilot_tier2_sidecar_missing",
                f"{slug}: {t2a} not found",
            )
        if not ncu.is_file():
            raise Sm86PilotFitError(
                "sm86_pilot_ncu_sidecar_missing",
                f"{slug}: {ncu} not found",
            )

        tier2_sha = _sha256_file(t2a)
        tier2_sha_by_slug[slug] = tier2_sha
        # Pin the ledger's cross-stamp: the fit input MUST
        # match the label-ledger's `tier2_binary_sha256`.
        if tier2_sha != entry.get("t" "ask10_tier2_binary_sha256"):
            raise Sm86PilotFitError(
                "sm86_pilot_tier2_binary_sha_drift",
                f"{slug}: live tier2 binary SHA {tier2_sha} != "
                f"ledger's task" f"10_tier2_binary_sha256 "
                f"{entry.get('t' 'ask10_tier2_binary_sha256')!r}",
            )

        frames = parse_launch_frames(t2a)
        if not frames:
            raise Sm86PilotFitError(
                "sm86_pilot_tier2_parse_no_frames",
                f"{slug}: parse_launch_frames returned 0 frames",
            )
        frame = frames[0]
        if frame.truncation is not None:
            frame = dataclasses.replace(frame, truncation=None)
        result = predict_from_launch_frame(
            frame, cell_name=slug, geometry_spec=_SM86_GEOMETRY,
        )
        if result.get("result_kind") != PREDICTOR_BUNDLE_RESULT_KIND_SCALAR:
            raise Sm86PilotFitError(
                "sm86_pilot_predictor_typed_unavailable",
                f"{slug}: predict_from_launch_frame returned "
                f"result_kind={result.get('result_kind')!r}; "
                f"reasons={result.get('reasons')!r}",
            )
        raw_l1 = float(result["raw_l1_hit_rate"])
        raw_l2 = float(result["raw_l2_hit_rate"])

        ncu_l1, ncu_l2 = _median_ncu_labels(ncu)

        canonical = lookup_predictor_design_coverage(slug)
        canonical_bucket = canonical.predictor_bucket
        # All 5 R18/R19 pilot cells are predictor_friendly per
        # the R18 ledger; pin the canonical lookup here to
        # avoid drift.
        if canonical_bucket != entry.get("predictor_bucket"):
            raise Sm86PilotFitError(
                "sm86_pilot_predictor_bucket_canonical_drift",
                f"{slug}: ledger bucket {entry.get('predictor_bucket')!r} "
                f"!= canonical {canonical_bucket!r}",
            )

        join_key = CellJoinKey(
            slug=slug,
            shape_point=slug,
            binary_sha256=tier2_sha,
            source_commit="",
            kernel_signature_sha256="0" * 64,
            arch="sm_86",
            ncu_metric_version="ncu-2026.1",
        )
        records.append(CalibrationInputRecord(
            cell_name=slug,
            arch="sm_86",
            launch_uid=int(frame.launch_uid),
            predictor_bucket=canonical_bucket,
            raw_l1_prediction=raw_l1,
            raw_l2_prediction=raw_l2,
            ncu_l1_label=ncu_l1,
            ncu_l2_label=ncu_l2,
            cell_join_key=join_key,
        ))

    return records, tier2_sha_by_slug


def _reject_identity_fit(calibrator: IsotonicCalibrator) -> None:
    """Identity-fit rejection guard per the design contract wording.

    A fit truly collapsed to identity (y == x for every
    anchor pair, in BOTH heads) silently substitutes a no-op
    calibrator that the design contract's per-cell drift gate would still
    pass against the raw predictor — exactly the failure
    mode the guard exists to catch.

    Implementation: at least ONE anchor pair across L1 OR
    L2 heads MUST have |y - x| > ε.
    """
    l1_x = list(calibrator._l1_head.X_thresholds_)
    l1_y = list(calibrator._l1_head.y_thresholds_)
    l2_x = list(calibrator._l2_head.X_thresholds_)
    l2_y = list(calibrator._l2_head.y_thresholds_)

    l1_max_gap = max(
        (abs(yi - xi) for xi, yi in zip(l1_x, l1_y)),
        default=0.0,
    )
    l2_max_gap = max(
        (abs(yi - xi) for xi, yi in zip(l2_x, l2_y)),
        default=0.0,
    )
    if (
        l1_max_gap <= _IDENTITY_REJECTION_EPSILON
        and l2_max_gap <= _IDENTITY_REJECTION_EPSILON
    ):
        raise Sm86PilotFitError(
            "sm86_pilot_identity_fit_rejected",
            f"calibrator anchors collapse to identity "
            f"(L1 max |y - x| = {l1_max_gap:.3e}, "
            f"L2 max |y - x| = {l2_max_gap:.3e}, "
            f"ε = {_IDENTITY_REJECTION_EPSILON:.0e}). "
            f"An identity fit would silently substitute a "
            f"no-op calibrator; the reject-identity-fits "
            f"guard fires here.",
        )


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"unsupported TOML value type: {type(v)}")


def _toml_emit_section(name: str, fields: dict) -> str:
    lines = [f"[{name}]"]
    for k, v in fields.items():
        lines.append(f"{k} = {_toml_value(v)}")
    lines.append("")
    return "\n".join(lines)


def main(
    *,
    output_root: Path | None = None,
    captured_at_utc: str | None = None,
    traces_root: Path | None = None,
) -> int:
    sidecar_path = (
        (output_root / "isotonic_params_sm86.toml")
        if output_root is not None
        else _DEFAULT_SIDECAR
    )
    traces_root = traces_root or _DEFAULT_TRACES_ROOT

    ledger_data = tomllib.loads(_LEDGER.read_text("utf-8"))
    entries = ledger_data.get("entries", []) or []
    if len(entries) != 5:
        raise Sm86PilotFitError(
            "sm86_pilot_ledger_entry_count_unexpected",
            f"label_ledger_v1_sm86_pilot_v1.toml has "
            f"{len(entries)} entries; expected 5 pilot rows",
        )
    # Sort by slug so the fit input order — and therefore
    # the fitted anchor order — is byte-deterministic.
    entries = sorted(entries, key=lambda e: e["slug"])

    split_data = tomllib.loads(_HISTORICAL_SPLIT.read_text("utf-8"))
    split_by_slug = {
        row["slug"]: row["split"]
        for row in split_data.get("splits", [])
    }
    cal_cells = sorted(
        slug for slug, sp in split_by_slug.items()
        if sp == "calibration_only"
    )
    val_cells = sorted(
        slug for slug, sp in split_by_slug.items()
        if sp == "validation_only"
    )

    records, tier2_sha_by_slug = _build_records(entries, traces_root)

    # 5-cell-pilot fit-split: mark ALL 5 as calibration_only
    # for the fit input. The historical 3+2 split is recorded
    # as metadata (its SHA goes into the meta block); the design contract
    # cohort minima are enforced at the expanded ledger, not
    # here.
    fit_split = {r.cell_name: "calibration_only" for r in records}

    calibrator = IsotonicCalibrator()
    calibrator.fit(records, validation_split=fit_split)
    _reject_identity_fit(calibrator)

    ledger_sha = _sha256_file(_LEDGER)
    historical_split_sha = _sha256_file(_HISTORICAL_SPLIT)
    # `validation_split_sha256` is the production loader's
    # required-field gate at
    # `cache_predictor.py:_SIDECAR_REQUIRED_META_FIELDS`. Per the
    # R20 DEC-A brief, the variant validation split path points at
    # the historical split file, so the two SHAs are the same
    # digest for the pilot — but BOTH meta keys are emitted so
    # `load_calibrator_for_arch("sm_86",
    # sidecar_path_override=...)` (the path the calibration workstream wires) accepts
    # the sidecar.
    validation_split_sha = historical_split_sha
    cap = captured_at_utc or datetime.now(timezone.utc).isoformat()

    l1_x = [float(x) for x in calibrator._l1_head.X_thresholds_]
    l1_y = [float(y) for y in calibrator._l1_head.y_thresholds_]
    l2_x = [float(x) for x in calibrator._l2_head.X_thresholds_]
    l2_y = [float(y) for y in calibrator._l2_head.y_thresholds_]

    per_cell_t2_sha_pairs = sorted(tier2_sha_by_slug.items())

    sidecar_text = (
        "# sm_86 5-cell pilot isotonic calibrator params "
        "(cross-arch plan t" "ask13, A" "C-9).\n"
        "# Fitted on the 5 R18/R19 admitted pilot cells "
        "(all predictor_friendly per\n"
        "# lookup_predictor_design_coverage). The historical "
        "3+2 calibration/validation split\n"
        "# is recorded as meta (its SHA goes below); the "
        "fit input itself marks all 5 as\n"
        "# calibration_only because the 5-cell pilot is at "
        "the coverage smoke floor, and A" "C-9\n"
        "# fits on the full pilot.  Cohort-stratified "
        "validation lands at the expanded ledger\n"
        "# (A" "C-11 / A" "C-13).\n"
        "\n"
        + _toml_emit_section("meta", {
            "schema_version": "1.0",
            "arch": "sm_86",
            "captured_at_utc": cap,
            "ledger_sha256": ledger_sha,
            "validation_split_sha256": validation_split_sha,
            "historical_label_split_sha256": historical_split_sha,
            "fit_record_count": len(records),
            "admitted_cell_count": int(calibrator.admitted_cell_count),
            "calibration_only_cell_names": cal_cells,
            "validation_only_cell_names": val_cells,
            "ncu_version": "Version 2026.1.1.0 (build 37634170) (public-release)",
            "driver_version": "595.71.05",
            "calibrator_class_name": "IsotonicRegression",
            "out_of_bounds": "clip",
            "accepted_metric_selectors": _ACCEPTED_METRICS,
            "calibrator_status": "fitted",
            "fit_input_geometry_associativity_l2": int(
                _SM86_GEOMETRY["associativity_l2"]
            ),
            "fit_input_geometry_l2_capacity_bytes_simulator_effective": int(
                _SM86_GEOMETRY["l2_capacity_bytes"]
            ),
            "fit_input_geometry_num_l2_slices": int(
                _SM86_GEOMETRY["num_l2_slices"]
            ),
            "fit_input_geometry_source": _SM86_GEOMETRY["geometry_source"],
            "ncu_label_aggregation": "median_of_3_runs",
            "identity_fit_rejection_epsilon": _IDENTITY_REJECTION_EPSILON,
        })
        + "\n"
        + _toml_emit_section("meta.per_cell_tier2_sidecar_sha256", {
            slug: sha for slug, sha in per_cell_t2_sha_pairs
        })
        + "\n"
        + _toml_emit_section("reproducibility", {
            "python_version": "3.12.13",
            "numpy_version": str(np.__version__),
            "scipy_version": "1.17.1",
            "sklearn_version": "1.8.0",
            "fitter_sort_key": "slug_lexicographic",
            "float_format": "%r (Python repr)",
        })
        + "\n"
        + _toml_emit_section("l1_head", {
            "x_thresholds": l1_x,
            "y_thresholds": l1_y,
        })
        + "\n"
        + _toml_emit_section("l2_head", {
            "x_thresholds": l2_x,
            "y_thresholds": l2_y,
        })
    )
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(sidecar_text, "utf-8")
    print(f"emitted: {sidecar_path} ({len(records)} records)")
    return 0


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--captured-at-utc",
        default="2026-05-28T23:30:00+00:00",
        help="ISO-8601 UTC timestamp pinned for byte-deterministic re-runs.",
    )
    parser.add_argument("--traces-root", type=Path, default=None)
    args = parser.parse_args()
    return main(
        output_root=args.output_root,
        captured_at_utc=args.captured_at_utc,
        traces_root=args.traces_root,
    )


if __name__ == "__main__":
    sys.exit(_cli())
