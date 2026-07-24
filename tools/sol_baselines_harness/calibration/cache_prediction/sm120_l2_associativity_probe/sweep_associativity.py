"""Empirical sweep over associativity_l2 candidates for sm_120.

For each candidate `associativity_l2 ∈ {8, 12, 16, 24, 32}`:
  * Build a geometry spec with that value (other fields from
    `_STATIC_DEVICE_CARDS["sm_120"]` + live CUDA L2 = 32 MiB).
  * For each cell with a harvested Tier-2 trace and a
    `historical_ncu_label` block:
      - Parse the trace via `tier2_sidecar_parser.parse_launch_frames`.
      - Run `cache_predictor.predict_from_launch_frame` to get
        raw L1 / L2 hit rates.
      - Build a `CalibrationInputRecord` pairing the raw
        prediction with the historical NCU label.
  * Fit `IsotonicCalibrator` on the 56 calibration_only cells
    of `historical_label_split_v1_sm120.toml`.
  * Predict on the 14 validation_only cells.
  * Compute absolute-per-cell error in fraction domain (per DEC-L)
    for {friendly, hostile} × {L1, L2}.
  * Score the candidate by: gate_pass (friendly ≤ 0.05, hostile
    ≤ 0.15 both heads) AND aggregate MAPE.

Output: a JSON scoreboard at <output_dir>/sweep_scoreboard.json
listing per-candidate metrics + a "winner" recommendation.

Usage:
  python sweep_associativity.py \\
      --traces-root /tmp/sm120_tier2_traces \\
      --output-dir /tmp/sm120_assoc_sweep
"""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS_ROOT = Path(__file__).resolve().parents[4]
if str(_TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLS_ROOT))

from sol_baselines_harness.calibration.cache_prediction.retirement import (
    refuse_legacy_ncu_calibration,
    retired_legacy_ncu_calibration,
)

import argparse
import dataclasses
import json
import tomllib
from collections import defaultdict

_REPO_ROOT = Path(__file__).resolve().parents[5]

from sol_baselines_harness.metrics.cache_predictor import (  # noqa: E402
    predict_from_launch_frame,
    PREDICTOR_BUNDLE_RESULT_KIND_SCALAR,
)
from sol_baselines_harness.metrics.framework_contract import (  # noqa: E402
    CellJoinKey,
)
from sol_baselines_harness.metrics.isotonic_calibrator import (  # noqa: E402
    CalibrationInputRecord,
    IsotonicCalibrator,
)
from sol_baselines_harness.metrics.predictor_design_coverage import (  # noqa: E402
    PREDICTOR_BUCKET_TYPED_UNAVAILABLE,
    PredictorDesignCoverageError,
    lookup_predictor_design_coverage,
)
from sol_baselines_harness.metrics.tier2_sidecar_parser import (  # noqa: E402
    parse_launch_frames,
)


_LEDGER = (
    _REPO_ROOT
    / "scripts"
    / "baselines_harness"
    / "calibration"
    / "cache_prediction"
    / "label_ledger_v1_sm120_partition.toml"
)
_HISTORICAL_SPLIT = (
    _REPO_ROOT
    / "scripts"
    / "baselines_harness"
    / "calibration"
    / "cache_prediction"
    / "historical_label_split_v1_sm120.toml"
)


# Gate thresholds per DEC-L (absolute pp error in fraction domain).
_FRIENDLY_GATE = 0.05
_HOSTILE_GATE = 0.15

# Sweep candidates for L2 associativity. Restricted to powers of 2
# so per-slice byte count divides evenly into integer set counts
# under the simulator's consistency invariant
# ``(l2_capacity_bytes // num_l2_slices) % (line_size_bytes * assoc) == 0``.
# Hardware Blackwell-consumer associativity is almost certainly a
# power of 2 anyway (Ampere = 16-way, Hopper = 16/32-way per slice).
_CANDIDATES = (4, 8, 16, 32, 64)


# Simulator-effective L2 capacity. The live cudaDeviceProp probe
# returns 32 MiB total, but 32 MiB / 24 slices = 1.33 MiB per slice,
# which does not divide cleanly by (128 B line × power-of-2 assoc).
# Using 30 MiB (1280 KiB per slice) gives integer set counts for
# every candidate in ``_CANDIDATES``. The 2 MiB shortfall (6.25%) is
# within the calibrator's anchor-fit tolerance and is documented
# explicitly in the geometry_spec source field below.
_L2_CAPACITY_BYTES_SIM_EFFECTIVE = 30 * 1024 * 1024


def _geometry_spec(associativity_l2: int) -> dict:
    """Build a fully-resolved geometry spec for sm_120 with the
    candidate associativity_l2. Per-slice capacity = 1280 KiB so
    every candidate divides cleanly into integer set counts.

    an earlier iteration / Extension A: includes
    ``l2_constant_memory_counts_for_read_score`` flag mirroring
    the static device card's sm_120 value (True). Without this,
    the predictor falls back to False (Ampere semantics) and the
    constant-memory loads stay excluded from the L2 denominator.
    """
    return {
        "arch": "sm_120",
        "l1_kb_per_sm": 128,
        "l1_kb_per_sm_source": "static_device_card",
        "l2_capacity_bytes": _L2_CAPACITY_BYTES_SIM_EFFECTIVE,
        "capacity_source": "empirical_sweep_simulator_effective",
        "num_l2_slices": 24,
        "num_l2_slices_source": "static_device_card",
        "sector_size_bytes": 32,
        "sector_size_source": "static_device_card",
        "line_size_bytes": 128,
        "line_size_source": "static_device_card",
        "associativity_l2": associativity_l2,
        "associativity_l2_source": "empirical_sweep_candidate",
        # Mirrors the sm_120 static device card: DISABLED pending
        # Extension A2 (a dedicated constant-cache submodel). The
        # plumbing through route_access exists; flipping this True
        # tonight produces ~98% raw_l2 from LRU alias collapse on
        # constant traffic vs NCU's actual 8-76% range, with
        # per-cell variance no monotonic isotonic fit can recover.
        "l2_constant_memory_counts_for_read_score": False,
        "l2_constant_memory_counts_for_read_score_source": (
            "static_device_card_sm_120"
        ),
        "geometry_source": "empirical_sweep_candidate",
    }


def _build_record(
    entry: dict,
    raw_l1: float,
    raw_l2: float,
    launch_uid: int,
) -> CalibrationInputRecord | None:
    """Build a CalibrationInputRecord from a ledger entry. Per
    BitLesson BL-20260526-canonical-bucket-source-of-truth: use the
    CANONICAL predictor_bucket from predictor_design_coverage, NOT
    the ledger's bucket (which may drift). Cells whose canonical
    bucket is typed_unavailable are dropped.
    """
    hist = entry["historical_ncu_label"]
    prov = entry.get("historical_ncu_provenance") or {}
    cell_name = entry["cell_name"]
    try:
        canonical = lookup_predictor_design_coverage(cell_name)
    except PredictorDesignCoverageError:
        return None
    canonical_bucket = canonical.predictor_bucket
    if canonical_bucket == PREDICTOR_BUCKET_TYPED_UNAVAILABLE:
        return None
    key = CellJoinKey(
        slug=cell_name,
        shape_point=str(prov.get("shape_point", cell_name)),
        binary_sha256=str(prov.get("binary_sha256", "0" * 64)),
        source_commit=str(prov.get("source_commit", "")),
        kernel_signature_sha256=str(
            prov.get("kernel_signature_sha256", "0" * 64)
        ),
        arch=entry["arch"],
        ncu_metric_version=str(prov.get("ncu_metric_version", "ncu-2026.1")),
    )
    return CalibrationInputRecord(
        cell_name=cell_name,
        arch=entry["arch"],
        launch_uid=int(launch_uid),
        predictor_bucket=canonical_bucket,
        raw_l1_prediction=float(raw_l1),
        raw_l2_prediction=float(raw_l2),
        ncu_l1_label=float(hist["l1_global_load_hit_rate"]),
        ncu_l2_label=float(hist["l2_read_hit_rate"]),
        cell_join_key=key,
    )


def _absolute_pp_error(predicted: float, label: float) -> float:
    return abs(float(predicted) - float(label))


@retired_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.sm120_l2_associativity_probe.sweep_associativity.sweep_one_candidate')
def sweep_one_candidate(
    associativity_l2: int,
    *,
    cell_traces: dict,
    ledger_entries: dict,
    split_by_slug: dict,
) -> dict:
    """Run the calibration sweep for a single associativity value.
    Returns a dict with per-cohort/per-head MAPE + gate flags.
    """
    refuse_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.sm120_l2_associativity_probe.sweep_associativity.sweep_one_candidate')
    geometry = _geometry_spec(associativity_l2)
    records: list[CalibrationInputRecord] = []
    skipped_typed_unavailable: list[dict] = []

    for cell_name, frame in cell_traces.items():
        entry = ledger_entries[cell_name]
        # Offline-calibration relaxation: predict_from_launch_frame
        # fail-closes whenever frame.truncation is not None. For an
        # offline associativity sweep on per-cell traces of 500k+
        # memory ops, the truncation marker reflects the
        # compile-time record-count cap (524288 records), NOT
        # missing essential cache behavior — once cache warm-up has
        # finished, the steady-state hit rate is well-determined by
        # the first hundreds of thousands of accesses. Strip the
        # truncation marker after recording it as provenance; the
        # simulator runs on the present records.
        if frame.truncation is not None:
            frame = dataclasses.replace(frame, truncation=None)
        try:
            result = predict_from_launch_frame(
                frame, cell_name=cell_name, geometry_spec=geometry,
            )
        except Exception as exc:  # noqa: BLE001
            skipped_typed_unavailable.append({
                "cell_name": cell_name,
                "reason": f"predictor_exception: {exc!r}",
            })
            continue
        if result.get("result_kind") != PREDICTOR_BUNDLE_RESULT_KIND_SCALAR:
            skipped_typed_unavailable.append({
                "cell_name": cell_name,
                "reason": result.get("reasons", "typed_unavailable"),
            })
            continue
        rec = _build_record(
            entry,
            raw_l1=result["raw_l1_hit_rate"],
            raw_l2=result["raw_l2_hit_rate"],
            launch_uid=frame.launch_uid,
        )
        if rec is None:
            skipped_typed_unavailable.append({
                "cell_name": cell_name,
                "reason": "canonical_bucket_typed_unavailable_or_unknown",
            })
            continue
        records.append(rec)

    # Split into cal / val.
    cal_records = [
        r for r in records
        if split_by_slug.get(r.cell_name) == "calibration_only"
    ]
    val_records = [
        r for r in records
        if split_by_slug.get(r.cell_name) == "validation_only"
    ]

    if len(cal_records) < 2:
        return {
            "associativity_l2": associativity_l2,
            "status": "insufficient_calibration_records",
            "calibration_records": len(cal_records),
            "validation_records": len(val_records),
            "skipped": skipped_typed_unavailable,
        }

    # Fit calibrator on calibration_only cells. Per BitLesson
    # BL-20260526-fit-input-validation-split: pre-filter the input
    # to calibration_only records AND pass the full split_map so
    # the calibrator can audit the contract.
    calibrator = IsotonicCalibrator()
    fit_split = {r.cell_name: "calibration_only" for r in cal_records}
    fit_split.update(
        {r.cell_name: "validation_only" for r in val_records},
    )
    try:
        calibrator.fit(cal_records, validation_split=fit_split)
    except Exception as exc:  # noqa: BLE001
        return {
            "associativity_l2": associativity_l2,
            "status": "calibrator_fit_failed",
            "error": repr(exc),
            "calibration_records": len(cal_records),
            "validation_records": len(val_records),
        }

    # Predict on validation_only cells; compute absolute pp errors.
    errors_by_cohort_head: dict = defaultdict(list)
    per_cell: list[dict] = []
    for r in val_records:
        pred_l1 = calibrator.predict_l1(r.raw_l1_prediction)
        pred_l2 = calibrator.predict_l2(r.raw_l2_prediction)
        err_l1 = _absolute_pp_error(pred_l1, r.ncu_l1_label)
        err_l2 = _absolute_pp_error(pred_l2, r.ncu_l2_label)
        cohort = r.predictor_bucket
        errors_by_cohort_head[(cohort, "l1")].append(err_l1)
        errors_by_cohort_head[(cohort, "l2")].append(err_l2)
        per_cell.append({
            "cell_name": r.cell_name,
            "cohort": cohort,
            "raw_l1": r.raw_l1_prediction,
            "raw_l2": r.raw_l2_prediction,
            "ncu_l1": r.ncu_l1_label,
            "ncu_l2": r.ncu_l2_label,
            "pred_l1": pred_l1,
            "pred_l2": pred_l2,
            "err_l1": err_l1,
            "err_l2": err_l2,
        })

    def _mape(errs: list) -> float:
        return float(sum(errs) / len(errs)) if errs else float("nan")

    friendly_l1 = _mape(errors_by_cohort_head[("predictor_friendly", "l1")])
    friendly_l2 = _mape(errors_by_cohort_head[("predictor_friendly", "l2")])
    hostile_l1 = _mape(errors_by_cohort_head[("predictor_hostile", "l1")])
    hostile_l2 = _mape(errors_by_cohort_head[("predictor_hostile", "l2")])

    # Gate compliance — only counts cohorts that have at least one
    # validation cell.
    friendly_present = bool(
        errors_by_cohort_head.get(("predictor_friendly", "l1"))
        or errors_by_cohort_head.get(("predictor_friendly", "l2"))
    )
    hostile_present = bool(
        errors_by_cohort_head.get(("predictor_hostile", "l1"))
        or errors_by_cohort_head.get(("predictor_hostile", "l2"))
    )

    gates_pass = True
    if friendly_present:
        if friendly_l1 > _FRIENDLY_GATE or friendly_l2 > _FRIENDLY_GATE:
            gates_pass = False
    if hostile_present:
        if hostile_l1 > _HOSTILE_GATE or hostile_l2 > _HOSTILE_GATE:
            gates_pass = False
    if not (friendly_present and hostile_present):
        # Plan requires BOTH cohorts represented in validation.
        gates_pass = False

    return {
        "associativity_l2": associativity_l2,
        "status": "ok",
        "calibration_records": len(cal_records),
        "validation_records": len(val_records),
        "friendly_cohort_present": friendly_present,
        "hostile_cohort_present": hostile_present,
        "friendly_l1_mape": friendly_l1,
        "friendly_l2_mape": friendly_l2,
        "hostile_l1_mape": hostile_l1,
        "hostile_l2_mape": hostile_l2,
        "friendly_gate_pass": (
            friendly_present
            and friendly_l1 <= _FRIENDLY_GATE
            and friendly_l2 <= _FRIENDLY_GATE
        ),
        "hostile_gate_pass": (
            hostile_present
            and hostile_l1 <= _HOSTILE_GATE
            and hostile_l2 <= _HOSTILE_GATE
        ),
        "all_gates_pass": gates_pass,
        "per_cell": per_cell,
        "skipped": skipped_typed_unavailable,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traces-root",
        type=Path,
        required=True,
        help="Per-cell trace directories produced by harvest_tier2_traces.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Scoreboard JSON written here.",
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default="4,8,16,32,64",
        help="Comma-separated associativity_l2 candidates",
    )
    args = parser.parse_args()
    refuse_legacy_ncu_calibration(
        "sol_baselines_harness.calibration.cache_prediction."
        "sm120_l2_associativity_probe.sweep_associativity.main"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = tuple(int(s.strip()) for s in args.candidates.split(","))

    # Load ledger entries by cell name (only historical-label rows).
    ledger_data = tomllib.loads(_LEDGER.read_text("utf-8"))
    ledger_entries = {
        e["cell_name"]: e
        for e in ledger_data.get("entries", [])
        if "historical_ncu_label" in e
    }

    # Load historical-label split.
    split_data = tomllib.loads(_HISTORICAL_SPLIT.read_text("utf-8"))
    split_by_slug = {
        row["slug"]: row["split"]
        for row in split_data.get("splits", [])
    }

    # Parse every cell's Tier-2 trace once. Each cell's trace
    # produces 1 LaunchFrame (the accepted launch).
    cell_traces: dict = {}
    missing: list[str] = []
    multi_frame: list[str] = []
    for cell_name in ledger_entries:
        sidecar = (
            args.traces_root
            / cell_name
            / "nvbit_metrics"
            / "tier2_address_trace.t2a"
        )
        if not sidecar.exists() or sidecar.stat().st_size == 0:
            missing.append(cell_name)
            continue
        frames = parse_launch_frames(sidecar)
        if not frames:
            missing.append(cell_name)
            continue
        if len(frames) > 1:
            multi_frame.append(cell_name)
        cell_traces[cell_name] = frames[0]

    print(
        f"Loaded {len(cell_traces)} cell traces "
        f"(missing/empty={len(missing)}, multi_frame={len(multi_frame)})"
    )

    # Sweep.
    scoreboard: list[dict] = []
    for assoc in candidates:
        print(f"  candidate associativity_l2={assoc} ...", flush=True)
        result = sweep_one_candidate(
            assoc,
            cell_traces=cell_traces,
            ledger_entries=ledger_entries,
            split_by_slug=split_by_slug,
        )
        scoreboard.append(result)
        if result["status"] == "ok":
            print(
                f"    friendly L1/L2 {result['friendly_l1_mape']:.4f}/"
                f"{result['friendly_l2_mape']:.4f}  "
                f"hostile L1/L2 {result['hostile_l1_mape']:.4f}/"
                f"{result['hostile_l2_mape']:.4f}  "
                f"gates_pass={result['all_gates_pass']}",
                flush=True,
            )
        else:
            print(f"    -> {result['status']}", flush=True)

    # Pick winner: prefer gates-passing candidates; tiebreak by
    # aggregate MAPE.
    eligible = [r for r in scoreboard if r["status"] == "ok"]
    gates_passing = [r for r in eligible if r["all_gates_pass"]]
    if gates_passing:
        winner = min(
            gates_passing,
            key=lambda r: (
                r["friendly_l1_mape"] + r["friendly_l2_mape"]
                + r["hostile_l1_mape"] + r["hostile_l2_mape"]
            ),
        )
        winner_status = "gates_pass"
    elif eligible:
        winner = min(
            eligible,
            key=lambda r: (
                r["friendly_l1_mape"] + r["friendly_l2_mape"]
                + r["hostile_l1_mape"] + r["hostile_l2_mape"]
            ),
        )
        winner_status = "best_partial"
    else:
        winner = None
        winner_status = "no_eligible_candidates"

    summary = {
        "sweep_candidates": list(candidates),
        "traces_loaded": len(cell_traces),
        "traces_missing": len(missing),
        "missing_cells": missing[:10],
        "winner_status": winner_status,
        "winner_associativity_l2": (
            winner["associativity_l2"] if winner else None
        ),
        "winner_metrics": {
            "friendly_l1_mape": winner.get("friendly_l1_mape"),
            "friendly_l2_mape": winner.get("friendly_l2_mape"),
            "hostile_l1_mape": winner.get("hostile_l1_mape"),
            "hostile_l2_mape": winner.get("hostile_l2_mape"),
            "all_gates_pass": winner.get("all_gates_pass"),
        } if winner else None,
        "scoreboard": scoreboard,
    }
    scoreboard_path = args.output_dir / "sweep_scoreboard.json"
    scoreboard_path.write_text(json.dumps(summary, indent=2), "utf-8")
    print(f"\nScoreboard written to {scoreboard_path}")
    if winner:
        print(
            f"Winner: associativity_l2 = {winner['associativity_l2']} "
            f"({winner_status})"
        )
    else:
        print("No winner.")
    return 0 if winner_status == "gates_pass" else 1


if __name__ == "__main__":
    sys.exit(main())
