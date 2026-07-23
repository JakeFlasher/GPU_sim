"""Historical sm_120 calibration-head v4 fitter.

The v4 implementation is retained for evidence/source archaeology, but the
production training path is intentionally disabled by the sm120-template
safety gate. Calling ``fit_and_emit_v4`` returns a typed-unavailable result
without reading bundles or overwriting either sidecar. The former 15/8/7
fit logic remains below so its contracts and historical artifacts stay
recoverable.

`split == "calibration"` cells (per AC-2.3 + AC-6.1) and emits two
sidecars under ``data/local_labels/``:

* ``calibration_head_v4.json`` — the trained head's serialized
  parameters + provenance (target_arch, training_cohort_size = 15,
  training_pair_count = 15, training_timestamp, source_commit,
  comparison_contract_sha256, the exact 15-cell training_cells
  list).
* ``calibration_evaluation_v4.json`` — the residual-statistics
  block over all 30 admitted sm_120 cells, partitioned by
  effective split (train: 15 / validation: 8 / holdout: 7 per
  AC-6.3). Each block carries ``geomean_residual``,
  ``max_residual``, and ``per_cell_residuals``.

Effective-split resolution
--------------------------

The committed 30 admitted sm_120 cells include 2 typed-excluded
primaries that were replaced by same-family-bucket replacement-pool
cells (per the AC-2.3 replacement semantics + Plan Evolution Log
Round 9 review). The roster encodes replacement rows with
``split = "replacement"``, but their EFFECTIVE split when admitted
is the typed-excluded primary's split. The module-level
``EFFECTIVE_SPLIT_FOR_ADMITTED_REPLACEMENT`` map records this
substitution explicitly so the v4 fit is reproducible from
committed loop state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_ROOT = _REPO_ROOT / "data" / "local_labels"
_DEFAULT_ROSTER_PATH = _DATA_ROOT / "sm120_cohort_expansion_roster.toml"
_DEFAULT_ADMITTED_PATH = _DATA_ROOT / "label_admitted.json"
_DEFAULT_BUNDLES_ROOT = _DATA_ROOT / "sweeps"
_DEFAULT_COMPARISON_CONTRACT_PATH = (
    _REPO_ROOT / "workloads" / "simbench_bw" / "comparison_contract.toml"
)
_DEFAULT_OUTPUT_HEAD_PATH = _DATA_ROOT / "calibration_head_v4.json"
_DEFAULT_OUTPUT_EVAL_PATH = _DATA_ROOT / "calibration_evaluation_v4.json"


# Effective-split substitution map. Two primary cells typed-excluded
# during the sweep (dram_stream_stride_mix in Round 4;
# flash_attn_bwd_bf16_sm80 in Round 8) and were replaced by
# same-family-bucket replacement-pool cells under the AC-2.3
# replacement rule. The replacements INHERIT the typed-excluded
# primary's split role so the loop-level 15/8/7 split contract is
# preserved.
EFFECTIVE_SPLIT_FOR_ADMITTED_REPLACEMENT: dict[tuple[str, str], str] = {
    # Replaces dram_stream_stride_mix/trace_l2_accuracy_xl
    # (calibration / memory_stress, typed-excluded Round 4).
    ("cross_entropy_fwd_fp32_sm80", "correctness_shape"): "calibration",
    # Replaces flash_attn_bwd_bf16_sm80/timing_shape
    # (validation / tensor_compute, typed-excluded Round 8).
    ("gemm_hmma_bf16_sm80", "trace_l2_accuracy_lg"): "validation",
}


# Pinned effective split counts per AC-2.3.
_EXPECTED_EFFECTIVE_SPLIT_COUNTS = {
    "calibration": 15,
    "validation": 8,
    "holdout": 7,
}


# ----- helpers reused via in-process import from fit_calibration_head -----


sys.path.insert(0, str(_REPO_ROOT / "tools" / "local_labels"))
import fit_calibration_head as fch  # noqa: E402
try:
    from tools.local_labels.sm120_template_safety import (  # noqa: E402
        unavailable_result,
    )
except ModuleNotFoundError:  # direct script execution from tools/local_labels
    from sm120_template_safety import (  # noqa: E402
        unavailable_result,
    )


# ----- effective-split resolution ----------------------------------


def _load_roster_rows(roster_path: Path) -> list[dict]:
    with roster_path.open("rb") as fh:
        return tomllib.load(fh).get("rows", []) or []


def resolve_effective_splits(
    *,
    roster_path: Path = _DEFAULT_ROSTER_PATH,
    admitted_path: Path = _DEFAULT_ADMITTED_PATH,
    replacement_map: dict[tuple[str, str], str] | None = None,
) -> dict[tuple[str, str], str]:
    """Return ``{(slug, shape_point): effective_split}`` for every
    admitted sm_120 cell. Effective split is the roster's ``split``
    field for primary cells (calibration / validation / holdout);
    for ``replacement`` rows it is the ``replacement_map`` entry
    (typed-excluded primary's original split) -- absence of a
    mapping for an admitted replacement is a typed defect.

    Raises ``RuntimeError`` when (a) an admitted sm_120 cell is
    not in the roster (a defect mode that violates AC-2.5's
    buildability witness) OR (b) an admitted replacement row has
    no entry in ``replacement_map``."""
    if replacement_map is None:
        replacement_map = EFFECTIVE_SPLIT_FOR_ADMITTED_REPLACEMENT
    roster_index = {
        (str(r.get("slug", "")), str(r.get("shape_point", ""))): r
        for r in _load_roster_rows(roster_path)
    }
    with admitted_path.open("r", encoding="utf-8") as fh:
        label_data = json.load(fh)
    admitted_keys = sorted({
        (str(c.get("slug", "")), str(c.get("shape_point", "")))
        for c in (label_data.get("cells") or [])
        if c.get("arch") == "sm_120"
    })
    result: dict[tuple[str, str], str] = {}
    unmapped: list[tuple[str, str]] = []
    off_roster: list[tuple[str, str]] = []
    for key in admitted_keys:
        row = roster_index.get(key)
        if row is None:
            off_roster.append(key)
            continue
        split = str(row.get("split", ""))
        if split in ("calibration", "validation", "holdout"):
            result[key] = split
            continue
        if split == "replacement":
            effective = replacement_map.get(key)
            if effective is None:
                unmapped.append(key)
                continue
            result[key] = effective
            continue
        unmapped.append(key)
    if off_roster:
        raise RuntimeError(
            f"admitted sm_120 cell(s) off-roster: {off_roster!r}"
        )
    if unmapped:
        raise RuntimeError(
            f"admitted sm_120 replacement cell(s) lack an "
            f"EFFECTIVE_SPLIT_FOR_ADMITTED_REPLACEMENT entry: "
            f"{unmapped!r}"
        )
    return result


def assert_effective_split_counts(
    effective_splits: dict[tuple[str, str], str],
    *,
    expected: dict[str, int] = _EXPECTED_EFFECTIVE_SPLIT_COUNTS,
) -> None:
    """Verify the effective-split partition matches the AC-2.3
    pinned 15/8/7 counts. Raises ``RuntimeError`` on mismatch."""
    actual: dict[str, int] = {}
    for split in effective_splits.values():
        actual[split] = actual.get(split, 0) + 1
    if actual != expected:
        raise RuntimeError(
            f"effective-split counts mismatch AC-2.3 contract: "
            f"got {actual!r}, expected {expected!r}"
        )


# ----- residual computation ----------------------------------------


def _compute_calibrated_residual(
    *,
    head: Any,
    predicted: float,
    label: float,
) -> float:
    """Apply the trained calibrator to a single predicted hit-rate
    and return the absolute residual ``|calibrated - label|``. The
    calibrator's ``transform`` returns ``(calibrated_array,
    envelope_mask_array)``; we use only the calibrated value."""
    import numpy as _np
    arr = _np.array([predicted])
    calibrated_arr, _envelope_mask = head.transform(arr)
    return abs(float(calibrated_arr[0]) - label)


def _per_cell_residual_record(
    *,
    cell: dict,
    predicted: float,
    label: float,
    calibrated_residual: float,
) -> dict[str, Any]:
    return {
        "slug": str(cell.get("slug", "")),
        "shape_point": str(cell.get("shape_point", "")),
        "arch": str(cell.get("arch", "")),
        "predicted": float(predicted),
        "calibrated_label": float(label),
        "calibrated_residual": float(calibrated_residual),
    }


def _residual_block(
    *,
    scalar_records: list[dict],
    typed_unavailable_records: list[dict],
) -> dict[str, Any]:
    """Build a per-split residual block. Aggregates
    ``geomean_residual`` + ``max_residual`` over the scalar
    records only; typed-unavailable cells are recorded in
    ``per_cell_residuals`` with ``calibrated_residual = None`` so
    the cohort cell count + roster identity is fully visible to
    the split-leakage scanner. ``n_cells`` is the total cohort
    cell count for the split (scalar + typed_unavailable);
    ``n_scalar`` and ``n_typed_unavailable`` are the per-class
    sub-counts."""
    residuals = [r["calibrated_residual"] for r in scalar_records]
    if not residuals:
        geomean = 0.0
        maxr = 0.0
    else:
        floor = 1e-12
        log_sum = sum(math.log(max(r, floor)) for r in residuals)
        geomean = math.exp(log_sum / len(residuals))
        maxr = max(residuals)
    per_cell: list[dict] = list(scalar_records)
    for u in typed_unavailable_records:
        per_cell.append({
            "slug": u["slug"],
            "shape_point": u["shape_point"],
            "arch": u["arch"],
            "predicted": None,
            "calibrated_label": float(u["label"]),
            "calibrated_residual": None,
            "typed_reason": u["typed_reason"],
        })
    return {
        "n_cells": len(scalar_records) + len(typed_unavailable_records),
        "n_scalar": len(scalar_records),
        "n_typed_unavailable": len(typed_unavailable_records),
        "geomean_residual": float(geomean),
        "max_residual": float(maxr),
        "per_cell_residuals": per_cell,
    }


# ----- main fit + emit ---------------------------------------------


def fit_and_emit_v4(
    *,
    roster_path: Path = _DEFAULT_ROSTER_PATH,
    admitted_path: Path = _DEFAULT_ADMITTED_PATH,
    bundles_root: Path = _DEFAULT_BUNDLES_ROOT,
    comparison_contract_path: Path = _DEFAULT_COMPARISON_CONTRACT_PATH,
    head_output_path: Path = _DEFAULT_OUTPUT_HEAD_PATH,
    eval_output_path: Path = _DEFAULT_OUTPUT_EVAL_PATH,
    replacement_map: dict[tuple[str, str], str] | None = None,
) -> tuple[Path | None, Path | None, dict[str, Any]]:
    """Fit the v4 calibration head on the 15 effective-calibration
    cells and emit the head + evaluation sidecars.

    Returns ``(head_path, eval_path, summary)``. For the disabled sm_120
    template lane both paths are ``None`` and ``summary`` is a typed-
    unavailable report; no input is loaded and no sidecar is written. For
    the historical implementation below, the summary is a JSON-serializable
    dict with the chosen head's LOOCV p90, per-split residual geomeans, and
    per-sidecar paths.
    """
    # The v4 fitter is retained so historical evidence can be inspected, but
    # it must never train or overwrite a production sm_120 head. Keep this
    # before NumPy import, roster/label reads, and output-directory creation.
    unavailable = unavailable_result(
        operation="fit_calibration_head_v4_sm120",
        arch="sm_120",
        head_output_path=None,
        eval_output_path=None,
        training_cohort_size=0,
        training_pair_count=0,
    )
    return None, None, unavailable

    np = fch._require_numpy()
    effective_splits = resolve_effective_splits(
        roster_path=roster_path,
        admitted_path=admitted_path,
        replacement_map=replacement_map,
    )
    assert_effective_split_counts(effective_splits)
    with admitted_path.open("r", encoding="utf-8") as fh:
        label_data = json.load(fh)
    all_cells_by_key: dict[tuple[str, str], dict] = {}
    for cell in label_data.get("cells") or []:
        if cell.get("arch") != "sm_120":
            continue
        all_cells_by_key[
            (str(cell["slug"]), str(cell["shape_point"]))
        ] = cell
    # Build a (slug, shape_point) -> bundle index for the sm_120
    # cohort. The legacy ``_scan_predictor_bundles`` indexes by
    # the 5-tuple (slug + shape + binary_sha + source_commit +
    # kernel_sig_sha); but the committed sm_120 predictor bundles
    # were produced in a later round than the label admissions,
    # so the SHA chain drifts. For AC-6 the bundle's identity IS
    # the (slug, shape_point) of the admitted sm_120 cohort cell
    # (per the Round-18 derived-cohort contract); use that
    # narrower key and require arch == "sm_120" + cache_layer ==
    # "L2" on every bundle. Both scalar and typed_unavailable
    # bundles are indexed; typed_unavailable cells appear in the
    # cohort with a None ``predicted`` field and are recorded in
    # the per-split residual blocks as typed-unavailable entries.
    bundle_by_slug_shape: dict[tuple[str, str], dict] = {}
    for path in bundles_root.glob("*/predictor_bundles/*.json"):
        with path.open("r", encoding="utf-8") as fh_b:
            bundle = json.load(fh_b)
        mpi = bundle.get("manifest_pairing_index") or {}
        if mpi.get("arch") != "sm_120":
            continue
        if bundle.get("cache_layer", "L2") != "L2":
            continue
        key_b = (str(mpi.get("slug", "")), str(mpi.get("shape_point", "")))
        if not key_b[0] or not key_b[1]:
            continue
        bundle_by_slug_shape[key_b] = bundle
    rows_by_split: dict[str, list[dict]] = {
        "calibration": [],
        "validation": [],
        "holdout": [],
    }
    typed_unavailable_by_split: dict[str, list[dict]] = {
        "calibration": [],
        "validation": [],
        "holdout": [],
    }
    for key, eff_split in effective_splits.items():
        cell = all_cells_by_key.get(key)
        if cell is None:
            raise RuntimeError(
                f"effective-split cell {key!r} not present in "
                f"label_admitted (corpus drift)"
            )
        bundle = bundle_by_slug_shape.get(key)
        if bundle is None:
            raise RuntimeError(
                f"admitted sm_120 cell {key!r} has no matching "
                f"sm_120 / L2 predictor bundle in {bundles_root!s}"
            )
        cell_pct = cell.get("cache_hit_rate_l2_sector_pct")
        if not isinstance(cell_pct, (int, float)) or isinstance(cell_pct, bool):
            raise RuntimeError(
                f"admitted sm_120 cell {key!r} has no usable "
                f"cache_hit_rate_l2_sector_pct"
            )
        label = float(cell_pct) / 100.0
        if bundle.get("result_kind") == "scalar" and isinstance(
            bundle.get("predicted_hit_rate"), (int, float),
        ) and not isinstance(bundle["predicted_hit_rate"], bool):
            rows_by_split[eff_split].append({
                "arch": "sm_120",
                "slug": key[0],
                "shape_point": key[1],
                "binary_sha256": str(cell.get("binary_sha256", "")),
                "kernel_signature_sha256": str(
                    cell.get("kernel_signature_sha256", "")
                ),
                "source_commit": str(cell.get("source_commit", "")),
                "ncu_metric_version": str(
                    cell.get("ncu_metric_version", "")
                ),
                "predicted": float(bundle["predicted_hit_rate"]),
                "label": label,
            })
        else:
            typed_unavailable_by_split[eff_split].append({
                "slug": key[0],
                "shape_point": key[1],
                "arch": "sm_120",
                "label": label,
                "typed_reason": str(
                    bundle.get("typed_reason", "unspecified")
                ),
            })
    cal_rows = rows_by_split["calibration"]
    val_rows = rows_by_split["validation"]
    hold_rows = rows_by_split["holdout"]
    cal_unavail = typed_unavailable_by_split["calibration"]
    val_unavail = typed_unavailable_by_split["validation"]
    hold_unavail = typed_unavailable_by_split["holdout"]
    # AC-2.3 contract: 15 calibration + 8 validation + 7 holdout
    # CELLS in the cohort. Some cells may have typed_unavailable
    # predictor bundles (no scalar prediction). The cohort cell
    # count includes both scalar + typed_unavailable; the
    # training_pair_count records the rows that actually fit the
    # calibrator (scalar bundles only). The per-split residual
    # block records all cells, with typed_unavailable cells
    # carrying ``calibrated_residual = None``.
    if len(cal_rows) + len(cal_unavail) != 15:
        raise RuntimeError(
            f"AC-6.1: effective-calibration cohort cell count must "
            f"be 15; got "
            f"{len(cal_rows) + len(cal_unavail)} "
            f"(scalar={len(cal_rows)}, "
            f"typed_unavailable={len(cal_unavail)})"
        )
    if len(val_rows) + len(val_unavail) != 8:
        raise RuntimeError(
            f"AC-6.3: effective-validation cohort cell count must "
            f"be 8; got "
            f"{len(val_rows) + len(val_unavail)} "
            f"(scalar={len(val_rows)}, "
            f"typed_unavailable={len(val_unavail)})"
        )
    if len(hold_rows) + len(hold_unavail) != 7:
        raise RuntimeError(
            f"AC-6.3: effective-holdout cohort cell count must "
            f"be 7; got "
            f"{len(hold_rows) + len(hold_unavail)} "
            f"(scalar={len(hold_rows)}, "
            f"typed_unavailable={len(hold_unavail)})"
        )
    # Fit calibrators on calibration rows only.
    train_p = np.array([r["predicted"] for r in cal_rows])
    train_y = np.array([r["label"] for r in cal_rows])
    factories = fch._calibrator_factories()
    loocv_table: dict[str, Any] = {}
    fitted_heads: dict[str, Any] = {}
    for head_name, factory in factories.items():
        residuals = fch._loocv_residuals(factory, cal_rows)
        loocv_table[head_name] = fch._per_head_residual_block(residuals)
        full = factory()
        full.fit(train_p, train_y)
        fitted_heads[head_name] = full
    chosen, rationale = fch._choose_head_simplicity_one_se(loocv_table)
    chosen_head = fitted_heads[chosen]
    chosen_params = fch._fitted_head_serialized(chosen, chosen_head)
    contract_sha256 = fch._sha256_of_file(comparison_contract_path)
    source_commit = fch._git_head_sha()
    # training_cells records the 15 EFFECTIVE-calibration COHORT
    # cells (15 per AC-2.3 / AC-6.1), including the 2 cells whose
    # bundles are typed_unavailable (those cells did not enter
    # the fit but are still in the cohort and must be visible to
    # the split-leakage scanner). training_pair_count records the
    # row count that actually fit the calibrator (scalar bundles
    # only).
    training_cells_combined = sorted(
        list({(r["slug"], r["shape_point"], "sm_120") for r in cal_rows})
        + [(c["slug"], c["shape_point"], "sm_120") for c in cal_unavail]
    )
    head_payload = {
        "schema_version": 1,
        "calibration_head_version": 4,
        "chosen_head": chosen,
        # The fitted-head parameters (e.g., PlattCalibrator A + B
        # sigmoid coefficients) are persisted under
        # chosen_head_parameters so downstream consumers
        # (notably tools/baseline_runners/nugteren2014_rdh_baseline.py's
        # v4_platt_prediction helper) can apply the calibrator to
        # uncalibrated walker scores without re-fitting. Previously
        # the params were captured only in trained_head_sha256
        # (hash-only), so the baseline runner's lookup at
        # head.get("chosen_head_parameters", {}) returned an empty
        # dict and v4_platt_prediction returned None for every cell.
        "chosen_head_parameters": chosen_params,
        "calibrator_libraries": fch._calibrator_library_provenance(),
        "training_cells": [
            {"slug": s, "shape_point": sp, "arch": a}
            for (s, sp, a) in training_cells_combined
        ],
        "training_cohort_size": 15,
        "training_pair_count": len(cal_rows),
        "training_typed_unavailable_count": len(cal_unavail),
        "training_timestamp": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "trained_head_sha256": hashlib.sha256(
            json.dumps(chosen_params, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "comparison_contract_sha256": contract_sha256,
        "training_rows": cal_rows,
        "source_commit": source_commit,
        "target_arch": "sm_120",
    }
    head_output_path.parent.mkdir(parents=True, exist_ok=True)
    with head_output_path.open("w", encoding="utf-8") as fh_out:
        json.dump(head_payload, fh_out, indent=2, sort_keys=True)
        fh_out.write("\n")
    # Apply the trained calibrator to every split's rows and
    # compute per-cell residuals.
    def _records_for(rows: list[dict]) -> list[dict]:
        recs: list[dict] = []
        for r in rows:
            cell = all_cells_by_key[(r["slug"], r["shape_point"])]
            residual = _compute_calibrated_residual(
                head=chosen_head,
                predicted=r["predicted"],
                label=r["label"],
            )
            recs.append(_per_cell_residual_record(
                cell=cell,
                predicted=r["predicted"],
                label=r["label"],
                calibrated_residual=residual,
            ))
        return recs
    train_records = _records_for(cal_rows)
    val_records = _records_for(val_rows)
    hold_records = _records_for(hold_rows)
    eval_payload = {
        "schema_version": 1,
        "calibration_evaluation_version": 4,
        "chosen_head": chosen,
        "selection_rule": fch._SELECTION_RULE_PIN,
        "target_arch": "sm_120",
        "comparison_contract_sha256": contract_sha256,
        "source_commit": source_commit,
        "training_cells": [
            {"slug": s, "shape_point": sp, "arch": a}
            for (s, sp, a) in training_cells_combined
        ],
        "training_cohort_size": 15,
        "training_pair_count": len(cal_rows),
        "loocv_table": loocv_table,
        "rationale": rationale,
        "skipped_heads": fch._skipped_heads_for_provenance(),
        "train": _residual_block(
            scalar_records=train_records,
            typed_unavailable_records=cal_unavail,
        ),
        "validation": _residual_block(
            scalar_records=val_records,
            typed_unavailable_records=val_unavail,
        ),
        "holdout": _residual_block(
            scalar_records=hold_records,
            typed_unavailable_records=hold_unavail,
        ),
    }
    eval_output_path.parent.mkdir(parents=True, exist_ok=True)
    with eval_output_path.open("w", encoding="utf-8") as fh_out:
        json.dump(eval_payload, fh_out, indent=2, sort_keys=True)
        fh_out.write("\n")
    chosen_p90 = float(loocv_table[chosen]["p90"])
    summary = {
        "chosen_head": chosen,
        "chosen_head_p90": chosen_p90,
        "training_cohort_size": 15,
        "training_pair_count": len(cal_rows),
        "train_geomean_residual": eval_payload["train"]["geomean_residual"],
        "validation_geomean_residual": eval_payload["validation"]["geomean_residual"],
        "holdout_geomean_residual": eval_payload["holdout"]["geomean_residual"],
        "head_output_path": str(head_output_path),
        "eval_output_path": str(eval_output_path),
    }
    return head_output_path, eval_output_path, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roster-path", type=Path, default=_DEFAULT_ROSTER_PATH,
    )
    parser.add_argument(
        "--admitted-path", type=Path, default=_DEFAULT_ADMITTED_PATH,
    )
    parser.add_argument(
        "--bundles-root", type=Path, default=_DEFAULT_BUNDLES_ROOT,
    )
    parser.add_argument(
        "--comparison-contract-path",
        type=Path,
        default=_DEFAULT_COMPARISON_CONTRACT_PATH,
    )
    parser.add_argument(
        "--head-output-path",
        type=Path,
        default=_DEFAULT_OUTPUT_HEAD_PATH,
    )
    parser.add_argument(
        "--eval-output-path",
        type=Path,
        default=_DEFAULT_OUTPUT_EVAL_PATH,
    )
    args = parser.parse_args(argv)
    head_path, eval_path, summary = fit_and_emit_v4(
        roster_path=args.roster_path,
        admitted_path=args.admitted_path,
        bundles_root=args.bundles_root,
        comparison_contract_path=args.comparison_contract_path,
        head_output_path=args.head_output_path,
        eval_output_path=args.eval_output_path,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary.get("result_kind") == "typed_unavailable":
        print(
            f"{summary['typed_diagnostic']}: {summary['typed_reason']}",
            file=sys.stderr,
        )
        return 2
    print(f"head sidecar: {head_path}", file=sys.stderr)
    print(f"eval sidecar: {eval_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
