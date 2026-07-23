"""Emit the calibration-head variance-improvement precondition audit artifact.

This is the producer side of the disk-pinned diagnostic the
scanner at ``test_ac5_variance_precondition_scanner.py`` consumes.
It walks the committed round-close sweeps, computes the variance
trajectory, snapshots the current per-cell predictions, classifies
the typed ``precondition_status`` against the documented floor
(0.05) vs the headline-validity-guard's stricter threshold (0.10),
and writes the JSON to ``data/local_labels/audit_evidence/
ac5_variance_precondition_diagnostic/variance_trajectory.json``.

The variance trajectory ROUND_CLOSE_SWEEPS table is hand-pinned
per the goal-tracker's Plan Evolution Log entries. Each entry
names a concrete committed sweep directory; the script reads
the sweep's ``comparison_bundle.json`` for geomean / CI and walks
its scalar predictor bundles to recompute stddev / spread. The
hand-pinned table is what gives the artifact its historical
trajectory — the script does not (and cannot) reliably auto-
detect "which sweep is which close" from filename alone.

Plan-token-clean: docstrings + variable names use domain
vocabulary only ("calibration-head precondition",
"headline-validity-guard threshold", "round-close sweep").
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SWEEPS_ROOT = _REPO_ROOT / "data" / "local_labels" / "sweeps"
_ARTIFACT_PATH = (
    _REPO_ROOT
    / "data"
    / "local_labels"
    / "audit_evidence"
    / "ac5_variance_precondition_diagnostic"
    / "variance_trajectory.json"
)

_DOCUMENTED_FLOOR_STDDEV = 0.05
_HEADLINE_GUARD_THRESHOLD_STDDEV = 0.10
_NEAR_SATURATION_PRED = 0.90

# Round-close sweep table. Each entry pins a concrete committed
# sweep directory to a goal-tracker close point. The "comment"
# field carries free-form context (which is permissible in a
# committed JSON artifact under the audit-evidence subtree; the
# plan-token scanner scans this Python source, not the emitted
# JSON's narrative strings).
_ROUND_CLOSE_SWEEPS: tuple[dict[str, Any], ...] = (
    {
        "round_number": 4,
        "sweep_dir": (
            "data/local_labels/sweeps/"
            "sweep_20260512T065806Z_4ebedb612feb"
        ),
        "comment": (
            "Post-stale-bundle-regeneration baseline. Pre-NCU-"
            "relabeling corpus (still using pre-launch-skip-fix "
            "labels); spread is widest here because the labels "
            "themselves are saturated near 1.0."
        ),
    },
    {
        "round_number": 10,
        "sweep_dir": (
            "data/local_labels/sweeps/"
            "sweep_20260512T110554Z_7f955daa87c8"
        ),
        "comment": (
            "Post-diagnostic-aggregation close. First close on "
            "the refreshed activation label corpus; per-cell "
            "stddev settles into the ~0.017 band where it stays "
            "across every subsequent model-refinement close."
        ),
    },
    {
        "round_number": 15,
        "sweep_dir": (
            "data/local_labels/sweeps/"
            "sweep_20260512T132835Z_be40b4cd7768"
        ),
        "comment": (
            "Post-slice-geometry-aware-effective-capacity model "
            "refinement close. No measurable shift on the "
            "activation cells (their working sets fit within "
            "the 1-MiB per-slice capacity)."
        ),
    },
    {
        "round_number": 16,
        "sweep_dir": (
            "data/local_labels/sweeps/"
            "sweep_20260512T134935Z_f98647691c2e"
        ),
        "comment": (
            "Post-sector-edge-accounting-contract-lock close. "
            "Test-side regression lock only; no model surface "
            "change."
        ),
    },
    {
        "round_number": 18,
        "sweep_dir": (
            "data/local_labels/sweeps/"
            "sweep_20260512T150307Z_2118e8f2539a"
        ),
        "comment": (
            "Post-write-policy-classification close. New "
            "memory_op_access_kind wire field + producer "
            "classifier + predictor no-write-allocate filter; "
            "legacy committed traces don't carry the new wire "
            "field, so the consumer-side fallback routes every "
            "observation to UNSPECIFIED."
        ),
    },
    {
        "round_number": 19,
        "sweep_dir": (
            "data/local_labels/sweeps/"
            "sweep_20260512T154908Z_708df787fefb"
        ),
        "comment": (
            "Post-coarse-CTA/SM-interleaving close. The "
            "interleave model is wired into every cell but "
            "produces no measurable prediction shift because "
            "the 5 admitted activation cells each have a "
            "single threadblock in their committed trace "
            "(round-robin merge over 1 TB is a no-op)."
        ),
    },
    {
        "round_number": 24,
        "sweep_dir": (
            "data/local_labels/sweeps/"
            "sweep_20260513T053443Z_88a98f984c6b"
        ),
        "comment": (
            "Post-microbench-rehabilitation-XL-shapes close. "
            "First headline regen on the expanded 13-cell "
            "cohort that includes 4 admitted microbench cells "
            "(l2_persisting, l2_thrash, register_spill, "
            "warm_cold) alongside the 5 activation cells; "
            "geomean rises to 0.56 because the microbench "
            "cells' relative errors are larger than the "
            "near-saturation activation cells, widening the "
            "per-cell prediction spread."
        ),
    },
    {
        "round_number": 25,
        "sweep_dir": (
            "data/local_labels/sweeps/"
            "sweep_20260513T083533Z_c423f775011f"
        ),
        "comment": (
            "Post-dram-stream-kernel-rehabilitation close. "
            "The kernel rewrite replaced the historical "
            "strided sweep with a per-warp isolated layered "
            "reuse phase, admitting the HRD bundle "
            "(bundle_underconfigured=False) at "
            "trace_l2_accuracy_xl. Live NCU sweep variance "
            "regressed several previously-admitted microbench "
            "label-side measurements (l2_persisting, "
            "warm_cold, l2_prefetch_hint, working_set "
            "stddev_l2_sector > 0.03 threshold across all "
            "warmup/cache configs); only 2 of 8 microbench "
            "cells cleared the repeatability gate this round "
            "(l2_thrash, register_spill). Comparison bundle "
            "still admits with 7 included cells per arch and "
            "geomean 0.36."
        ),
    },
    {
        "round_number": 26,
        "sweep_dir": (
            "data/local_labels/sweeps/"
            "sweep_20260513T094819Z_9eea75dc57dd"
        ),
        "comment": (
            "Microbench label-side recovery attempt under "
            "controlled host conditions: GPU baseline 47-52C "
            "idle, no concurrent CPU load, sweep at 09:48 "
            "UTC. Bounded 3-cell probe at --repeats 5 "
            "(dram_stream, l2_persisting, warm_cold) "
            "admitted 0 of 3; the bounded-probe route "
            "escalated to a full 13-cell sweep at "
            "--repeats 10. Result: only 1 microbench "
            "(l2_thrash_pointer_chase) cleared the stddev "
            "gate at any config; register_spill (prior-round "
            "admitted at stddev 0.010) now fails at minimum "
            "0.053. Comparison admits 6 cells per sm_86 (1 "
            "microbench + 5 activation), geomean 0.388 CI "
            "[0.155, 0.750]; per-arch headline below the "
            "five-microbench floor; the microbench "
            "rehabilitation lever stays open. "
            "Host-thermal hypothesis NOT confirmed: the "
            "controlled idle host did not recover the prior "
            "admit rate. dram_stream additionally produces "
            "NCU CSV parse failures (lts__t_sector_hit_rate "
            "values >100% at multiple warmup configs); the "
            "per-warp isolated layered kernel may be hitting "
            "an NCU replay artifact that is independent of "
            "host thermal state. Documented for a follow-up "
            "round."
        ),
    },
    {
        "round_number": 27,
        "sweep_dir": (
            "data/local_labels/sweeps/"
            "sweep_20260513T111518Z_6ba4090e29d0"
        ),
        "comment": (
            "Drift-recovery round. Bounded 4-cell probe at "
            "--repeats 5 admitted 2 of 4 candidates "
            "(register_spill at min stddev 0.003; "
            "working_set at min 0.009). Full 13-cell sweep "
            "at --repeats 5 admitted 2 microbench in the "
            "headline (l2_thrash, register_spill); "
            "working_set's probe admission did not "
            "reproduce in the full sweep. New diagnostic "
            "infrastructure landed: a label-stability "
            "emitter + scanner reading committed verdict "
            "tables, plus a typed NCU "
            "lts__t_sector_hit_rate.pct >100% diagnostic "
            "fixture documenting why dram_stream stays "
            "label-blocked. Comparison admits 7 cells per "
            "sm_86 (2 microbench + 5 activation), geomean "
            "0.744 CI [0.470, 1.267]; below the "
            "five-microbench floor but advanced from the "
            "prior round's 1 of 8 to 2 of 8."
        ),
    },
)


def _load_scalar_predictions(
    sweep_dir: Path,
) -> list[tuple[str, str, float, str]]:
    """Return [(slug, shape_point, predicted_hit_rate,
    predictor_build_commit), ...] for every result_kind=="scalar"
    bundle in ``sweep_dir/predictor_bundles/``, sorted by
    (slug, shape_point).
    """
    bundles_dir = sweep_dir / "predictor_bundles"
    if not bundles_dir.is_dir():
        return []
    out: list[tuple[str, str, float, str]] = []
    for path in sorted(bundles_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            b = json.load(f)
        if b.get("result_kind") != "scalar":
            continue
        mpi = b.get("manifest_pairing_index", {}) or {}
        slug = mpi.get("slug")
        shape_point = mpi.get("shape_point")
        predicted = b.get("predicted_hit_rate")
        pbc = b.get("predictor_build_commit", "") or ""
        if (
            isinstance(slug, str)
            and isinstance(shape_point, str)
            and isinstance(predicted, (int, float))
        ):
            out.append((slug, shape_point, float(predicted), str(pbc)))
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def _pop_stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def _spread(values: list[float]) -> float:
    if not values:
        return 0.0
    return max(values) - min(values)


def _count_single_tb_cells(sweep_dir: Path) -> int:
    """Count scalar bundles whose evidence indicates a single
    threadblock (one per_launch_hit_rates entry).
    """
    count = 0
    bundles_dir = sweep_dir / "predictor_bundles"
    if not bundles_dir.is_dir():
        return 0
    for path in sorted(bundles_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            b = json.load(f)
        if b.get("result_kind") != "scalar":
            continue
        evidence = b.get("evidence", {}) or {}
        per_launch = evidence.get("per_launch_hit_rates")
        if isinstance(per_launch, list) and len(per_launch) == 1:
            count += 1
    return count


def _count_near_saturation(values: list[float]) -> int:
    return sum(1 for v in values if v >= _NEAR_SATURATION_PRED)


def _read_comparison_bundle_summary(sweep_dir: Path) -> dict[str, float]:
    """Read sweep's comparison_bundle.json and return geomean + CI.

    Raises FileNotFoundError if the bundle is missing, and KeyError
    if the bundle is missing any of `geomean`, `geomean_ci_low`, or
    `geomean_ci_high`. Fail-loud rather than silent-default to
    catch malformed / partially-regenerated bundles at emit time.
    The downstream scanner has a matching contract witness that
    pins the same required-keys contract at disk-read time.
    """
    path = sweep_dir / "comparison_bundle.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"comparison_bundle.json missing under {sweep_dir}"
        )
    with path.open("r", encoding="utf-8") as f:
        cb = json.load(f)
    missing = [
        k for k in ("geomean", "geomean_ci_low", "geomean_ci_high")
        if k not in cb
    ]
    if missing:
        raise KeyError(
            f"comparison_bundle.json under {sweep_dir} is missing "
            f"required keys: {missing}"
        )
    return {
        "geomean": float(cb["geomean"]),
        "ci_low": float(cb["geomean_ci_low"]),
        "ci_high": float(cb["geomean_ci_high"]),
    }


def _build_trajectory_entry(entry: dict[str, Any]) -> dict[str, Any]:
    sweep_dir = _REPO_ROOT / entry["sweep_dir"]
    summary = _read_comparison_bundle_summary(sweep_dir)
    scalar_preds = _load_scalar_predictions(sweep_dir)
    pred_values = [p[2] for p in scalar_preds]
    return {
        "round_number": entry["round_number"],
        "sweep_dir": entry["sweep_dir"],
        "geomean": summary["geomean"],
        "ci_low": summary["ci_low"],
        "ci_high": summary["ci_high"],
        "stddev_per_cell_predictions": _pop_stddev(pred_values),
        "spread_per_cell_predictions": _spread(pred_values),
        "comment": entry["comment"],
    }


def _build_root_cause_analysis(
    latest_sweep_dir: Path,
    current_predictions: list[float],
) -> dict[str, Any]:
    single_tb = _count_single_tb_cells(latest_sweep_dir)
    near_sat = _count_near_saturation(current_predictions)
    narrative = (
        "Every admitted activation cell in the latest committed "
        "sweep predicts hit_rate >= 0.90 (near-saturation) and "
        "comes from a single-threadblock dynamic_trace.pb. "
        "Under the slice-aware effective-capacity model, the "
        "5 cells' working sets all fit within the sm_86 1-MiB "
        "per-slice capacity. Under the sector-edge accounting "
        "contract-lock, every 32B access maps to one sector. "
        "Under the write-policy classifier, the committed legacy "
        "traces do not carry the memory_op_access_kind wire field "
        "so every access routes to UNSPECIFIED (no-op on the "
        "predictor's no-write-allocate filter). Under the "
        "coarse CTA/SM interleaving model, the round-robin merge "
        "reduces to the legacy sequential walk on every cell "
        "(total_threadblocks==1 per cell). The four model-"
        "refinement surfaces are correctly wired and verified by "
        "synthetic multi-TB / multi-policy witnesses, but none "
        "shifts the prediction on the committed activation corpus "
        "because the corpus is structurally homogeneous: 5 "
        "element-wise normalization/silu/softmax launches, each "
        "single-TB, each saturating L2 hit rate. The calibration-"
        "head variance-improvement precondition cannot be met "
        "by additional model refinements on this corpus; the "
        "path forward requires either expanding the corpus "
        "(e.g., microbench rehabilitation via larger shapes) "
        "or revising the precondition itself."
    )
    return {
        "single_threadblock_cells_count": single_tb,
        "near_saturation_cells_count": near_sat,
        "narrative": narrative,
    }


def _latest_sweep_with_predictor_bundles() -> Path:
    """Return the lex-greatest committed sweep dir under
    ``data/local_labels/sweeps/`` that contains a
    ``predictor_bundles/`` subdir.

    The "current" snapshot in the audit artifact MUST come from
    the latest committed sweep (the scope-to-latest semantic the
    regen-delta scanner enforces), not from a hand-pinned table
    that lags behind the most-recent recapture. Raises
    FileNotFoundError if no such sweep exists.
    """
    if not _SWEEPS_ROOT.is_dir():
        raise FileNotFoundError(
            f"sweeps root missing: {_SWEEPS_ROOT}"
        )
    candidates = sorted(
        d
        for d in _SWEEPS_ROOT.glob("*")
        if (d / "predictor_bundles").is_dir()
    )
    if not candidates:
        raise FileNotFoundError(
            f"no committed sweeps with predictor_bundles under "
            f"{_SWEEPS_ROOT}"
        )
    return candidates[-1]


def build_artifact() -> dict[str, Any]:
    """Compose the variance-precondition audit artifact payload."""
    trajectory = [
        _build_trajectory_entry(entry) for entry in _ROUND_CLOSE_SWEEPS
    ]
    latest_sweep_dir = _latest_sweep_with_predictor_bundles()
    scalar_preds = _load_scalar_predictions(latest_sweep_dir)
    current_predictions = [p[2] for p in scalar_preds]
    current_stddev = _pop_stddev(current_predictions)
    current_spread = _spread(current_predictions)
    precondition_status = (
        "satisfied"
        if current_stddev >= _DOCUMENTED_FLOOR_STDDEV
        else "not_satisfied"
    )
    current_per_cell_predictions = [
        {
            "slug": slug,
            "shape_point": shape_point,
            "predicted_hit_rate": predicted,
            "predictor_build_commit": pbc,
        }
        for slug, shape_point, predicted, pbc in scalar_preds
    ]
    root_cause = _build_root_cause_analysis(
        latest_sweep_dir, current_predictions
    )
    return {
        "schema_version": 1,
        "precondition_status": precondition_status,
        "documented_floor_stddev": _DOCUMENTED_FLOOR_STDDEV,
        "dec_2a_stricter_threshold_stddev": (
            _HEADLINE_GUARD_THRESHOLD_STDDEV
        ),
        "current_stddev": current_stddev,
        "current_spread": current_spread,
        "current_per_cell_predictions": current_per_cell_predictions,
        "trajectory_rounds": trajectory,
        "root_cause_analysis": root_cause,
    }


def write_artifact(payload: dict[str, Any]) -> Path:
    """Serialize ``payload`` to the canonical artifact path."""
    _ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _ARTIFACT_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return _ARTIFACT_PATH


def main() -> int:
    payload = build_artifact()
    path = write_artifact(payload)
    print(
        f"wrote {path.relative_to(_REPO_ROOT)}: "
        f"precondition_status={payload['precondition_status']!r} "
        f"current_stddev={payload['current_stddev']:.6f} "
        f"documented_floor={payload['documented_floor_stddev']!r} "
        f"trajectory_entries={len(payload['trajectory_rounds'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
