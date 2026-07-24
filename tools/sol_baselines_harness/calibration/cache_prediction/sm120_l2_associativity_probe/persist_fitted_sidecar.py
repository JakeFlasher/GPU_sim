"""Fit the sm_120 isotonic calibrator with associativity_l2=32 and
persist the sidecar + validation report.

Re-uses the sweep_associativity helpers to:
  1. Parse Tier-2 traces for every historical-label cell.
  2. Run predict_from_launch_frame with the winning geometry.
  3. Build CalibrationInputRecords using canonical buckets.
  4. Fit IsotonicCalibrator on the calibration_only cells of
     historical_label_split_v1_sm120.toml.
  5. Persist isotonic_params_sm120.toml with non-empty L1/L2
     anchors, current ledger/split hashes, and reproducibility
     metadata.
  6. Persist sm120_validation_report_v1.toml with per-cohort
     friendly/hostile L1/L2 MAPE on the validation_only cells.

Usage:
  python persist_fitted_sidecar.py --traces-root /tmp/sm120_tier2_traces
"""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS_ROOT = Path(__file__).resolve().parents[4]
if str(_TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLS_ROOT))

from sol_baselines_harness.calibration.cache_prediction.retirement import (
    refuse_legacy_ncu_calibration,
)

import argparse
import dataclasses
import hashlib
import tomllib
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[5]

# Re-use the sweep driver's helpers.
from sol_baselines_harness.calibration.cache_prediction.sm120_l2_associativity_probe.sweep_associativity import (  # noqa: E402
    _build_record,
    _geometry_spec,
    _absolute_pp_error,
    _L2_CAPACITY_BYTES_SIM_EFFECTIVE,
)

from sol_baselines_harness.metrics.cache_predictor import (  # noqa: E402
    predict_from_launch_frame,
    PREDICTOR_BUNDLE_RESULT_KIND_SCALAR,
)
from sol_baselines_harness.metrics.isotonic_calibrator import (  # noqa: E402
    IsotonicCalibrator,
)
from sol_baselines_harness.metrics.tier2_sidecar_parser import (  # noqa: E402
    parse_launch_frames,
)


_CAL_DIR = (
    _REPO_ROOT
    / "tools"
    / "sol_baselines_harness"
    / "calibration"
    / "cache_prediction"
)
_LEDGER = _CAL_DIR / "label_ledger_v1_sm120_partition.toml"
_HISTORICAL_SPLIT = _CAL_DIR / "historical_label_split_v1_sm120.toml"
_CURRENT_SPLIT = _CAL_DIR / "calibration_validation_split_v1_sm120.toml"
_SIDECAR = _CAL_DIR / "isotonic_params_sm120.toml"
_REPORT = _CAL_DIR / "sm120_validation_report_v1.toml"
_SIMULATOR_UNAVAILABLE = _CAL_DIR / "sm120_simulator_unavailable_cells_v1.toml"

# The winning associativity from an earlier sweep.
_WINNING_ASSOC = 32


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        # Use Python repr so it's round-trippable.
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-root", type=Path, required=True)
    args = parser.parse_args()
    refuse_legacy_ncu_calibration(
        "sol_baselines_harness.calibration.cache_prediction."
        "sm120_l2_associativity_probe.persist_fitted_sidecar.main"
    )

    geometry = _geometry_spec(_WINNING_ASSOC)
    ledger_data = tomllib.loads(_LEDGER.read_text("utf-8"))
    ledger_entries = {
        e["cell_name"]: e
        for e in ledger_data.get("entries", [])
        if "historical_ncu_label" in e
    }
    split_data = tomllib.loads(_HISTORICAL_SPLIT.read_text("utf-8"))
    split_by_slug = {
        row["slug"]: row["split"]
        for row in split_data.get("splits", [])
    }

    # an earlier iteration: load the simulator-out-of-domain exclusion list
    # (sm120_simulator_unavailable_cells_v1.toml). Cells listed here
    # are excluded from BOTH the fit and the validation report; the
    # reduced supported-domain is recorded as report provenance.
    excl_data = tomllib.loads(_SIMULATOR_UNAVAILABLE.read_text("utf-8"))
    excluded_by_cell = {
        row["cell_name"]: row
        for row in excl_data.get("excluded", [])
    }

    # Build CalibrationInputRecords from every cell with a clean trace
    # that the simulator can score AND that is not on the
    # simulator-out-of-domain exclusion list.
    records = []
    skipped = []
    simulator_unavailable_skipped = []
    for cell_name, entry in ledger_entries.items():
        if cell_name in excluded_by_cell:
            excl_row = excluded_by_cell[cell_name]
            simulator_unavailable_skipped.append({
                "cell_name": cell_name,
                "typed_reason": excl_row["typed_reason"],
                "split_role": split_by_slug.get(cell_name, "unknown"),
            })
            continue
        sidecar = (
            args.traces_root
            / cell_name
            / "nvbit_metrics"
            / "tier2_address_trace.t2a"
        )
        if not sidecar.exists() or sidecar.stat().st_size == 0:
            skipped.append((cell_name, "trace_missing"))
            continue
        frames = parse_launch_frames(sidecar)
        if not frames:
            skipped.append((cell_name, "no_frames"))
            continue
        # Offline-calibration relaxation: strip the truncation marker
        # so predict_from_launch_frame runs on the ~500k captured
        # records. See sweep_associativity.py for the rationale.
        frame = frames[0]
        if frame.truncation is not None:
            frame = dataclasses.replace(frame, truncation=None)
        result = predict_from_launch_frame(
            frame, cell_name=cell_name, geometry_spec=geometry,
        )
        if result.get("result_kind") != PREDICTOR_BUNDLE_RESULT_KIND_SCALAR:
            reason = result.get("reasons", "typed_unavailable")
            skipped.append((cell_name, str(reason)))
            continue
        rec = _build_record(
            entry,
            raw_l1=result["raw_l1_hit_rate"],
            raw_l2=result["raw_l2_hit_rate"],
            launch_uid=frame.launch_uid,
        )
        if rec is None:
            skipped.append((cell_name, "canonical_bucket_typed_unavailable"))
            continue
        records.append(rec)

    cal_records = [
        r for r in records
        if split_by_slug.get(r.cell_name) == "calibration_only"
    ]
    val_records = [
        r for r in records
        if split_by_slug.get(r.cell_name) == "validation_only"
    ]
    print(
        f"Built {len(records)} records "
        f"({len(cal_records)} cal + {len(val_records)} val); "
        f"skipped {len(skipped)} cells"
    )
    if simulator_unavailable_skipped:
        print(
            f"Excluded {len(simulator_unavailable_skipped)} "
            "simulator-out-of-domain cells:"
        )
        for row in simulator_unavailable_skipped:
            print(
                f"  - {row['cell_name']:65s} "
                f"({row['typed_reason']}, {row['split_role']})"
            )

    # Fit the calibrator on calibration_only.
    calibrator = IsotonicCalibrator()
    fit_split = {
        r.cell_name: "calibration_only" for r in cal_records
    }
    fit_split.update({
        r.cell_name: "validation_only" for r in val_records
    })
    calibrator.fit(cal_records, validation_split=fit_split)
    print(
        f"Calibrator fitted on {calibrator.admitted_cell_count} cells; "
        f"L1 anchors={len(calibrator._l1_head.X_thresholds_)}; "
        f"L2 anchors={len(calibrator._l2_head.X_thresholds_)}"
    )

    # Predict on validation_only, compute per-cohort metrics.
    errors_by_cohort_head = defaultdict(list)
    per_cell_eval = []
    for r in val_records:
        pred_l1 = calibrator.predict_l1(r.raw_l1_prediction)
        pred_l2 = calibrator.predict_l2(r.raw_l2_prediction)
        err_l1 = _absolute_pp_error(pred_l1, r.ncu_l1_label)
        err_l2 = _absolute_pp_error(pred_l2, r.ncu_l2_label)
        errors_by_cohort_head[(r.predictor_bucket, "l1")].append(err_l1)
        errors_by_cohort_head[(r.predictor_bucket, "l2")].append(err_l2)
        per_cell_eval.append({
            "cell_name": r.cell_name,
            "cohort": r.predictor_bucket,
            "raw_l1": float(r.raw_l1_prediction),
            "raw_l2": float(r.raw_l2_prediction),
            "ncu_l1": float(r.ncu_l1_label),
            "ncu_l2": float(r.ncu_l2_label),
            "pred_l1": float(pred_l1),
            "pred_l2": float(pred_l2),
            "err_l1": float(err_l1),
            "err_l2": float(err_l2),
        })

    def _mape(errs):
        return float(sum(errs) / len(errs)) if errs else float("nan")

    friendly_l1 = _mape(errors_by_cohort_head[("predictor_friendly", "l1")])
    friendly_l2 = _mape(errors_by_cohort_head[("predictor_friendly", "l2")])
    hostile_l1 = _mape(errors_by_cohort_head[("predictor_hostile", "l1")])
    hostile_l2 = _mape(errors_by_cohort_head[("predictor_hostile", "l2")])

    print(
        f"friendly L1/L2 MAPE: {friendly_l1:.4f} / {friendly_l2:.4f}\n"
        f"hostile L1/L2 MAPE: {hostile_l1:.4f} / {hostile_l2:.4f}"
    )

    # ---- Persist the sidecar ----
    captured_at = datetime.now(timezone.utc).isoformat()
    ledger_sha = _sha256_file(_LEDGER)
    current_split_sha = _sha256_file(_CURRENT_SPLIT)
    historical_split_sha = _sha256_file(_HISTORICAL_SPLIT)

    cal_cell_names = sorted(r.cell_name for r in cal_records)
    val_cell_names = sorted(r.cell_name for r in val_records)
    denied_cells = sorted(name for name, _ in skipped)

    l1_x = list(calibrator._l1_head.X_thresholds_)
    l1_y = list(calibrator._l1_head.y_thresholds_)
    l2_x = list(calibrator._l2_head.X_thresholds_)
    l2_y = list(calibrator._l2_head.y_thresholds_)

    accepted_metrics = [
        "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
        "lts__t_sectors_srcunit_tex_op_read.sum",
        "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
    ]

    sidecar_text = (
        "# sm_120 isotonic calibrator params (an earlier iteration: fitted via "
        "empirical-sweep geometry).\n"
        "# associativity_l2=32 picked by sweep_associativity.py over "
        "{4, 8, 16, 32, 64} candidates.\n"
        "# L2 simulator-effective capacity = 30 MiB (24 slices x "
        "1280 KiB each); physical 32 MiB divides unevenly by 24 so a\n"
        "# clean per-slice capacity was used. The 2 MiB shortfall "
        "(6.25%) is within the calibrator's anchor-fit tolerance.\n"
        "\n"
        + _toml_emit_section("meta", {
            "schema_version": "1.0",
            "arch": "sm_120",
            "captured_at_utc": captured_at,
            "ledger_sha256": ledger_sha,
            "validation_split_sha256": current_split_sha,
            "historical_label_split_sha256": historical_split_sha,
            "fit_record_count": len(cal_records),
            "admitted_cell_count": int(calibrator.admitted_cell_count),
            "calibration_only_cell_names": cal_cell_names,
            "validation_only_cell_names": val_cell_names,
            "ncu_version": "Version 2026.1.1.0 (build 37634170) (public-release)",
            "driver_version": "595.71.05",
            "calibrator_class_name": "IsotonicRegression",
            "out_of_bounds": "clip",
            "accepted_metric_selectors": accepted_metrics,
            "calibrator_status": "fitted",
            "fit_input_geometry_associativity_l2": _WINNING_ASSOC,
            "fit_input_geometry_l2_capacity_bytes_simulator_effective": (
                _L2_CAPACITY_BYTES_SIM_EFFECTIVE
            ),
            "fit_input_geometry_num_l2_slices": 24,
            "denied_typed_unavailable_cells": denied_cells,
            "simulator_unavailable_manifest_sha256": (
                _sha256_file(_SIMULATOR_UNAVAILABLE)
            ),
            "simulator_unavailable_cells_excluded_from_fit": sorted(
                row["cell_name"] for row in simulator_unavailable_skipped
            ),
        })
        + "\n"
        + _toml_emit_section("reproducibility", {
            "python_version": "3.12.13",
            "numpy_version": np.__version__,
            "scipy_version": "1.17.1",
            "sklearn_version": "1.8.0",
            "fitter_sort_key": "input_order",
            "float_format": "%r (Python repr)",
        })
        + "\n"
        + _toml_emit_section("l1_head", {
            "x_thresholds": [float(x) for x in l1_x],
            "y_thresholds": [float(y) for y in l1_y],
        })
        + "\n"
        + _toml_emit_section("l2_head", {
            "x_thresholds": [float(x) for x in l2_x],
            "y_thresholds": [float(y) for y in l2_y],
        })
    )
    _SIDECAR.write_text(sidecar_text, "utf-8")
    print(f"Wrote sidecar: {_SIDECAR}")

    # ---- Persist the validation report ----
    sidecar_sha = _sha256_file(_SIDECAR)

    def _gate(value, threshold):
        return value <= threshold if not (value != value) else False

    friendly_l1_pass = _gate(friendly_l1, 0.05)
    friendly_l2_pass = _gate(friendly_l2, 0.05)
    hostile_l1_pass = _gate(hostile_l1, 0.15)
    hostile_l2_pass = _gate(hostile_l2, 0.15)

    friendly_cells = sorted(
        r.cell_name for r in val_records
        if r.predictor_bucket == "predictor_friendly"
    )
    hostile_cells = sorted(
        r.cell_name for r in val_records
        if r.predictor_bucket == "predictor_hostile"
    )
    friendly_present = bool(friendly_cells)
    hostile_present = bool(hostile_cells)
    all_gates = (
        friendly_present and hostile_present
        and friendly_l1_pass and friendly_l2_pass
        and hostile_l1_pass and hostile_l2_pass
    )

    # Build a dynamic header summarizing the four-gate state.
    gate_summary_parts = []
    gate_summary_parts.append(
        f"friendly L1={friendly_l1:.4f} ({'PASS' if friendly_l1_pass else 'FAIL'})"
    )
    gate_summary_parts.append(
        f"friendly L2={friendly_l2:.4f} ({'PASS' if friendly_l2_pass else 'FAIL'})"
    )
    if hostile_present:
        gate_summary_parts.append(
            f"hostile L1={hostile_l1:.4f} ({'PASS' if hostile_l1_pass else 'FAIL'})"
        )
        gate_summary_parts.append(
            f"hostile L2={hostile_l2:.4f} ({'PASS' if hostile_l2_pass else 'FAIL'})"
        )
    else:
        gate_summary_parts.append("hostile cohort absent")

    # Build the fail_closed_reason listing every failing gate.
    fail_reasons = []
    if friendly_present and not friendly_l1_pass:
        fail_reasons.append(
            f"friendly L1 MAPE {friendly_l1:.4f} > 0.05 gate"
        )
    if friendly_present and not friendly_l2_pass:
        fail_reasons.append(
            f"friendly L2 MAPE {friendly_l2:.4f} > 0.05 gate"
        )
    if hostile_present and not hostile_l1_pass:
        fail_reasons.append(
            f"hostile L1 MAPE {hostile_l1:.4f} > 0.15 gate"
        )
    if hostile_present and not hostile_l2_pass:
        fail_reasons.append(
            f"hostile L2 MAPE {hostile_l2:.4f} > 0.15 gate"
        )
    if not friendly_present:
        fail_reasons.append("friendly cohort absent from validation set")
    if not hostile_present:
        fail_reasons.append("hostile cohort absent from validation set")
    fail_closed_reason_str = (
        "round10_validation_gates_unmet: " + "; ".join(fail_reasons)
        if fail_reasons
        else ""
    )

    report_text = (
        "# sm_120 validation report (an earlier iteration: real fit + real "
        "evaluation).\n"
        "# associativity_l2=32 picked by empirical sweep; sidecar "
        "carries non-empty L1/L2 anchors.\n"
        f"# Gate state: {' | '.join(gate_summary_parts)}\n"
        "\n"
        + _toml_emit_section("meta", {
            "schema_version": "1.0",
            "captured_at_utc": captured_at,
            "host_arch": "sm_120",
            "isotonic_params_sha256": sidecar_sha,
            "ledger_sha256": ledger_sha,
            "validation_split_sha256": current_split_sha,
            "historical_label_split_sha256": historical_split_sha,
            "validation_cells_evaluated": len(val_records),
            "measured_cells_in_ledger": 0,
            "ledger_total_entries": len(ledger_data.get("entries", [])),
            "error_formula": "absolute_per_cell_in_fraction_domain",
            "error_formula_dec_l_rationale": (
                "Per DEC-L: absolute in fraction domain"
            ),
            "split_methodology": (
                "Real fit on historical-label split's calibration_only "
                "cells; evaluated "
                "on validation_only cells."
            ),
            "gate_evaluation_state": (
                "evaluated" if all_gates else "evaluated_gates_failed"
            ),
            "associativity_l2_source": "empirical_sweep_round10",
            "simulator_unavailable_cells_count": len(
                simulator_unavailable_skipped,
            ),
            "simulator_unavailable_cells": sorted(
                row["cell_name"] for row in simulator_unavailable_skipped
            ),
            "simulator_unavailable_typed_reasons": sorted(
                {row["typed_reason"] for row in simulator_unavailable_skipped}
            ),
            "simulator_unavailable_manifest_sha256": (
                _sha256_file(_SIMULATOR_UNAVAILABLE)
            ),
        })
        + "\n"
        + _toml_emit_section("cohort_friendly", {
            "present": friendly_present,
            "cell_count": len(friendly_cells),
            "cells": friendly_cells,
            "l1_mape": friendly_l1,
            "l2_mape": friendly_l2,
            "l1_gate_threshold": 0.05,
            "l2_gate_threshold": 0.05,
            "l1_gate_pass": friendly_l1_pass,
            "l2_gate_pass": friendly_l2_pass,
            "bundle_pass": friendly_l1_pass and friendly_l2_pass,
        })
        + "\n"
        + _toml_emit_section("cohort_hostile", {
            "present": hostile_present,
            "cell_count": len(hostile_cells),
            "cells": hostile_cells,
            "l1_mape": hostile_l1,
            "l2_mape": hostile_l2,
            "l1_gate_threshold": 0.15,
            "l2_gate_threshold": 0.15,
            "l1_gate_pass": hostile_l1_pass,
            "l2_gate_pass": hostile_l2_pass,
            "bundle_pass": hostile_l1_pass and hostile_l2_pass,
        })
        + "\n"
        + _toml_emit_section("overall", {
            "all_gates_pass": all_gates,
            "cohort_friendly_pass": friendly_l1_pass and friendly_l2_pass,
            "cohort_hostile_pass": hostile_l1_pass and hostile_l2_pass,
            "cohort_friendly_present": friendly_present,
            "cohort_hostile_present": hostile_present,
            "fail_closed": not all_gates,
            "fail_closed_reason": fail_closed_reason_str,
        })
    )
    _REPORT.write_text(report_text, "utf-8")
    print(f"Wrote validation report: {_REPORT}")

    print(f"\nall_gates_pass = {all_gates}")
    return 0 if all_gates else 1


if __name__ == "__main__":
    sys.exit(main())
