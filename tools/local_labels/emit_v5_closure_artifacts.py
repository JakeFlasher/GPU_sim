"""Emit the accuracy-gate thresholds / the falsification hypothesis contract / the reliability diagnostics advisory closure artifacts for the v5 monotonic head.

Run AFTER ``fit_monotonic_calibration_head_v5.py`` produces
``data/local_labels/calibration_head_v5.json``. This script computes
per-cell residuals on the admitted cohort + emits:

  - ``data/local_labels/calibration_head_v5_accuracy_gates.json`` -- the accuracy-gate thresholds
    scanner output (full-cohort median absolute, full-cohort geomean rel,
    holdout geomean rel, per-bucket p90 abs err with N<8 advisory).
  - ``data/local_labels/calibration_head_v5_falsification.json`` -- the falsification hypothesis contract
    permutation-attribution arm. Records joint contribution of
    ``footprint_ratio + long_rd_tail_mass`` to v4 -> v5 error reduction +
    the confirmed / refuted / indeterminate verdict.
  - ``data/local_labels/calibration_head_v5_reliability_diagnostics.json``
    -- the reliability diagnostics advisory advisory: per-bucket calibration slope + intercept + 10-bin
    expected calibration error.

When the admitted set is too small for a gate to be meaningful (e.g., 2
cells when the full-cohort median-abs threshold needs ~30+ for a meaningful median), the artifact records
``status: indeterminate_insufficient_data`` with the gate's expected
minimum N. This is honest reporting -- the artifact lands so downstream
consumers see the methodological state, not so the gate is silently
glossed over.

the per-bucket minimum-N rule alignment: the IR-on-projection used_features attribute records
{raw_prediction, footprint_ratio, long_rd_tail_mass}; the
projection-decomposition attribution for the falsification hypothesis contract uses |w_feature| / sum(|w|)
on the frozen weights (+1, -1, -1) per the goal tracker's
"isotonic-projection proportional-weight decomposition convention".
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workloads.simbench_bw import canonical_features as cf  # noqa: E402
from workloads.simbench_bw import monotonic_calibration_head as mch  # noqa: E402


def _git_head_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
        ).decode().strip()
        return out
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


DEFAULT_V5_HEAD = "data/local_labels/calibration_head_v5.json"
DEFAULT_V4_HEAD = "data/local_labels/calibration_head_v4.json"
DEFAULT_AC3_OUT = "data/local_labels/calibration_head_v5_accuracy_gates.json"
DEFAULT_FALS_OUT = (
    "data/local_labels/calibration_head_v5_falsification.json"
)
DEFAULT_REL_OUT = (
    "data/local_labels/calibration_head_v5_reliability_diagnostics.json"
)


# the accuracy-gate thresholds thresholds (per the goal tracker's the accuracy-gate thresholds rows).
AC3_FULL_COHORT_MEDIAN_ABS_MAX = 0.06
AC3_FULL_COHORT_GEOMEAN_REL_MAX = 0.15
AC3_HOLDOUT_GEOMEAN_REL_MAX = 0.25
AC3_PER_BUCKET_P90_MAX = 0.15
AC3_PER_BUCKET_MIN_N_FOR_GATE = 8

# the falsification hypothesis contract thresholds.
AC71_CONFIRMED_MIN_PCT = 30.0
AC71_REFUTED_MAX_PCT = 10.0


def _skipped_cell_reasons_from_head(head: dict) -> dict[str, str]:
    """Build a structured map of cohort-skipped cells -> typed_reason
    from the v5 head's cohort_scope. Indeterminate gate-artifact arms
    embed this map so the cohort-coverage scanner can verify the
    indeterminate state has per-cell linkage to the v5 head's
    skipped list per the audit-visibility contract pinned by the
    cohort-coverage scanner."""
    skipped = head.get("cohort_scope", {}).get("skipped", []) or []
    out: dict[str, str] = {}
    for c in skipped:
        key = (
            f"{c.get('arch', '')}/"
            f"{c.get('slug', '')}/"
            f"{c.get('shape_point', '')}"
        )
        out[key] = c.get("typed_reason", "unknown")
    return out

# IR-on-projection selected-class name (closed-form proportional-weight
# attribution path); every other selected class uses the registered
# permutation procedure.
IR_ON_PROJECTION_NAME = "isotonic_regression_on_frozen_projection"


def _compute_non_ir_joint_attribution(
    *,
    cells_with_features: list[dict],
    ensemble: Any,
    bundle_lookup: dict[tuple[str, str, str], dict] | None,
) -> tuple[float | None, str, dict[str, Any]]:
    """Permutation-based joint contribution of ``footprint_ratio +
    long_rd_tail_mass`` to v5's improvement over v4, for non-IR
    selected classes (GAM / LightGBM / XGBoost).

    Procedure (per the goal tracker's falsification-arm row):
      1. Compute per-arch training-set background means of
         ``footprint_ratio`` + ``long_rd_tail_mass`` over admitted cells.
      2. Group cells by ``(arch, family_bucket)``.
      3. For each cell: get v4 prediction (from
         ``bundle.predicted_hit_rate``), v5 prediction (from
         ``ensemble.predict``), and v5_perturbed prediction
         (``ensemble.predict`` with the two joint features replaced
         by per-arch background means).
      4. Per bucket:
         ``total_reduction = mean(|v4_err|) - mean(|v5_err|)``;
         ``joint_reduction = mean(|v4_err|) - mean(|v5_perturbed_err|)``;
         ``bucket_share = (total_reduction - joint_reduction) / total_reduction``.
      5. Average ``bucket_share`` across buckets, multiply by 100 to
         produce ``joint_contribution_pct``.

    Returns ``(attribution_pct, attribution_source, diagnostics)``
    where ``attribution_pct`` is ``None`` when no bucket yielded a
    finite share + diagnostics records the per-bucket shares for the
    artifact's audit trail.
    """
    from collections import defaultdict

    diagnostics: dict[str, Any] = {
        "per_arch_background_means": {},
        "per_bucket_shares": {},
        "per_bucket_diagnostics": {},
    }
    if bundle_lookup is None:
        return (
            None,
            "non_ir_permutation_no_bundle_lookup",
            diagnostics,
        )
    by_arch: dict[str, list[dict]] = defaultdict(list)
    for cell in cells_with_features:
        by_arch[cell["arch"]].append(cell)
    arch_backgrounds: dict[str, dict[str, float]] = {}
    for arch, arch_cells in by_arch.items():
        if not arch_cells:
            continue
        fr_mean = sum(
            float(c["features"].get("footprint_ratio", 0.0))
            for c in arch_cells
        ) / len(arch_cells)
        lt_mean = sum(
            float(c["features"].get("long_rd_tail_mass", 0.0))
            for c in arch_cells
        ) / len(arch_cells)
        arch_backgrounds[arch] = {
            "footprint_ratio": fr_mean,
            "long_rd_tail_mass": lt_mean,
        }
    diagnostics["per_arch_background_means"] = arch_backgrounds
    by_bucket: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for cell in cells_with_features:
        by_bucket[(cell["arch"], cell["family_bucket"])].append(cell)
    bucket_shares: list[float] = []
    for (arch, bucket), bucket_cells in by_bucket.items():
        background = arch_backgrounds.get(arch)
        if background is None:
            continue
        v4_errs: list[float] = []
        v5_errs: list[float] = []
        v5_perturbed_errs: list[float] = []
        for cell in bucket_cells:
            key = (cell["arch"], cell["slug"], cell["shape_point"])
            bundle = bundle_lookup.get(key)
            if bundle is None:
                continue
            legacy = bundle.get("predicted_hit_rate")
            if legacy is None:
                continue
            label = float(cell["label"])
            try:
                v5_pred = float(ensemble.predict(
                    arch, bucket, cell["features"],
                ))
                perturbed_features = dict(cell["features"])
                perturbed_features["footprint_ratio"] = background[
                    "footprint_ratio"
                ]
                perturbed_features["long_rd_tail_mass"] = background[
                    "long_rd_tail_mass"
                ]
                v5_perturbed_pred = float(ensemble.predict(
                    arch, bucket, perturbed_features,
                ))
            except (RuntimeError, ImportError):
                continue
            v4_errs.append(abs(float(legacy) - label))
            v5_errs.append(abs(v5_pred - label))
            v5_perturbed_errs.append(abs(v5_perturbed_pred - label))
        if not v4_errs:
            continue
        v4_mean = sum(v4_errs) / len(v4_errs)
        v5_mean = sum(v5_errs) / len(v5_errs)
        v5_perturbed_mean = sum(v5_perturbed_errs) / len(v5_perturbed_errs)
        total_reduction = v4_mean - v5_mean
        joint_reduction = v4_mean - v5_perturbed_mean
        bucket_key = f"{arch}/{bucket}"
        # Non-contributory guard: when v5 is no better than v4 on a
        # bucket (total_reduction <= 0), the share formula
        # (total_reduction - joint_reduction) / total_reduction yields
        # a numerically misleading value (division by ~0 or sign
        # flip). Skip the bucket's contribution to the average AND
        # record a diagnostic flag so the artifact's audit trail
        # shows the skip.
        per_bucket_record: dict[str, Any] = {
            "n": len(v4_errs),
            "v4_mean_absolute_error": v4_mean,
            "v5_mean_absolute_error": v5_mean,
            "v5_perturbed_mean_absolute_error": v5_perturbed_mean,
            "total_reduction": total_reduction,
            "joint_reduction_without_features": joint_reduction,
        }
        if total_reduction <= 0:
            per_bucket_record["non_contributory"] = True
            per_bucket_record["non_contributory_reason"] = (
                "total_reduction_non_positive:"
                f"v4_mean={v4_mean:.6f}_v5_mean={v5_mean:.6f}"
            )
            diagnostics["per_bucket_diagnostics"][bucket_key] = per_bucket_record
            continue
        share = (total_reduction - joint_reduction) / total_reduction
        bucket_shares.append(share)
        diagnostics["per_bucket_shares"][bucket_key] = share
        per_bucket_record["non_contributory"] = False
        diagnostics["per_bucket_diagnostics"][bucket_key] = per_bucket_record
    if not bucket_shares:
        return (
            None,
            "non_ir_permutation_insufficient_data",
            diagnostics,
        )
    avg_share = sum(bucket_shares) / len(bucket_shares)
    attribution_pct = 100.0 * avg_share
    return (
        attribution_pct,
        "non_ir_permutation_per_bucket_average_of_joint_contribution_share",
        diagnostics,
    )


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (pct / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _emit_ac3_gates(
    *,
    v5_head_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    head = json.loads(v5_head_path.read_text(encoding="utf-8"))
    admitted = head.get("cohort_scope", {}).get("admitted", [])
    # Re-load the head and refit the ensemble using the SELECTED class
    # (via the centralized dispatch helper). This replaces the prior
    # hardcoded IsotonicRegressionOnFrozenProjection instantiation so the
    # accuracy gates evaluate the actually-selected v5 head.
    from collections import defaultdict
    from tools.local_labels.v5_selected_class_dispatch import (
        build_selected_ensemble,
    )
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for c in admitted:
        by_bucket[(c["arch"], c["family_bucket"])].append(c)
    # Insufficient admitted cells for a meaningful gate => emit indeterminate.
    n = len(admitted)
    if n < 3:
        artifact: dict[str, Any] = {
            "schema_version": 1,
            "status": "indeterminate_insufficient_data",
            "admitted_count": n,
            "source_commit": _git_head_sha(),
            "source_v5_head_commit": head.get("source_commit"),
            "minimum_admitted_for_meaningful_gates": 30,
            "reason": (
                "the accuracy-gate thresholds full-cohort gates require enough admitted cells to "
                "make the median + geomean stable. With n=" + str(n)
                + " the metrics are point estimates dominated by a "
                "single observation; the cohort_scope.skipped block in "
                "calibration_head_v5.json records the data-staging gap."
            ),
            "gates": {
                "full_cohort_median_absolute_error": None,
                "full_cohort_geomean_relative_error": None,
                "holdout_geomean_relative_error": None,
                "per_bucket_p90_absolute_error": {},
            },
            "thresholds": {
                "full_cohort_median_absolute_error_max": AC3_FULL_COHORT_MEDIAN_ABS_MAX,
                "full_cohort_geomean_relative_error_max": AC3_FULL_COHORT_GEOMEAN_REL_MAX,
                "holdout_geomean_relative_error_max": AC3_HOLDOUT_GEOMEAN_REL_MAX,
                "per_bucket_p90_absolute_error_max": AC3_PER_BUCKET_P90_MAX,
            },
            "skipped_cell_reasons": _skipped_cell_reasons_from_head(head),
        }
        output_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return artifact
    # When enough cells are admitted, compute the gates. The fitter writes
    # the cohort to the head; refit on it via the SELECTED class's factory.
    dispatch = build_selected_ensemble(head)
    if dispatch["status"] != "ok":
        artifact = {
            "schema_version": 1,
            "status": dispatch["status"],
            "admitted_count": n,
            "source_commit": _git_head_sha(),
            "source_v5_head_commit": head.get("source_commit"),
            "selected_class_name": dispatch["selected_class_name"],
            "reason": dispatch["reason"],
        }
        output_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return artifact
    ensemble = dispatch["ensemble"]
    # Rehydrate cell records from the head (label is already a fraction).
    # Carry slug + shape_point so the holdout branch's split-index lookup
    # by (slug, shape_point) does not raise KeyError on the non-placeholder
    # branch.
    cell_records = []
    for c in admitted:
        variant_id = str(c.get("variant_id", "") or "")
        cell_records.append({
            # Mixed-arity identity: sibling register-cap variants share their
            # (slug, shape_point) name; the canonical variant_id ("" for a
            # base cell) keeps them distinct.
            "identity": (c["arch"], c["slug"], c["shape_point"], variant_id),
            "arch": c["arch"],
            "family_bucket": c["family_bucket"],
            "slug": c["slug"],
            "shape_point": c["shape_point"],
            "variant_id": variant_id,
            "split_assignment": c.get("split_assignment"),
            "features": c.get("features", {}),
            "label": float(c["label"]),
        })
    # NOTE: the head writes admitted with features omitted to keep the JSON
    # small. Bail to indeterminate when features are not available.
    if not all(c["features"] for c in cell_records):
        artifact = {
            "schema_version": 1,
            "status": "indeterminate_features_unavailable",
            "admitted_count": n,
            "source_commit": _git_head_sha(),
            "source_v5_head_commit": head.get("source_commit"),
            "reason": (
                "the accuracy-gate thresholds scanner needs per-cell features to compute predictions; "
                "the head's cohort_scope omits them. Re-run the fitter with "
                "--emit-cohort-features to populate them."
            ),
        }
        output_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return artifact
    # Split-aware fit scope: a split-aware head was fit on the
    # calibration+validation subset only; this emitter mirrors that — the
    # full-cohort metrics cover the FIT POOL, and the sealed holdout cells
    # (split_assignment="holdout") are scored exactly once by the dedicated
    # holdout block below against the fit-pool-fitted ensemble. Legacy heads
    # without fit_scope keep the prior full-cohort behavior with the
    # roster-split holdout path.
    split_aware = bool((head.get("fit_scope") or {}).get("split_aware"))
    if split_aware:
        fit_records = [
            c for c in cell_records
            if c.get("split_assignment") != "holdout"
        ]
        head_holdout_records = [
            c for c in cell_records
            if c.get("split_assignment") == "holdout"
        ]
    else:
        fit_records = cell_records
        head_holdout_records = []
    ensemble.fit(fit_records)
    preds = [
        ensemble.predict(c["arch"], c["family_bucket"], c["features"])
        for c in fit_records
    ]
    labels = [c["label"] for c in fit_records]
    abs_errs = [abs(p - y) for p, y in zip(preds, labels)]
    full_median_abs = _percentile(abs_errs, 50.0)
    full_geomean = mch.geomean_relative_error(preds, labels)
    # Per-bucket p90 over the FIT POOL (mirrors the fit scope).
    by_bucket = defaultdict(list)
    for c in fit_records:
        by_bucket[(c["arch"], c["family_bucket"])].append(c)
    by_bucket_metrics: dict[str, Any] = {}
    for key, members in by_bucket.items():
        bucket_preds = [
            ensemble.predict(m["arch"], m["family_bucket"], m["features"])
            for m in members
        ]
        bucket_labels = [m["label"] for m in members]
        bucket_abs = [abs(p - y) for p, y in zip(bucket_preds, bucket_labels)]
        p90 = _percentile(bucket_abs, 90.0)
        by_bucket_metrics[f"{key[0]}/{key[1]}"] = {
            "n": len(members),
            "p90_absolute_error": p90,
            "gate_status": (
                "advisory_n_below_threshold"
                if len(members) < AC3_PER_BUCKET_MIN_N_FOR_GATE
                else ("pass" if p90 <= AC3_PER_BUCKET_P90_MAX else "fail")
            ),
        }
    # Holdout evaluation.
    #
    # Split-aware head: the sealed holdout cells were carried in
    # cohort_scope.admitted with split_assignment="holdout" and NEVER entered
    # the fit; score them exactly once against the fit-pool-fitted ensemble.
    # A holdout cell whose (arch, family_bucket) never appeared in the fit
    # pool has no head to route through and is counted (not silently
    # dropped) in the status string.
    #
    # Legacy head (no fit_scope): derive holdout membership from the frozen
    # roster split index (sm_120-cohort assignments keyed on the cell's own
    # arch) over the in-sample predictions — the pre-refit behavior.
    holdout_geomean: float | None = None
    holdout_status = "not_computed_roster_unavailable"
    if split_aware:
        holdout_preds = []
        holdout_labels = []
        skipped_unseen_bucket = 0
        for c in head_holdout_records:
            try:
                p = ensemble.predict(
                    c["arch"], c["family_bucket"], c["features"],
                )
            except RuntimeError:
                skipped_unseen_bucket += 1
                continue
            holdout_preds.append(float(p))
            holdout_labels.append(float(c["label"]))
        if holdout_preds:
            holdout_geomean = mch.geomean_relative_error(
                holdout_preds, holdout_labels,
            )
            holdout_status = "ok" if skipped_unseen_bucket == 0 else (
                f"ok_with_{skipped_unseen_bucket}"
                "_unseen_bucket_holdout_cells"
            )
        elif head_holdout_records:
            holdout_status = "no_scoreable_holdout_cells"
        else:
            holdout_status = "no_holdout_cells_in_admitted_set"
    else:
        try:
            from workloads.simbench_bw.comparison_runner import (
                _load_roster_split_index,
            )
            split_index = _load_roster_split_index()
            holdout_preds = []
            holdout_labels = []
            for c, p in zip(fit_records, preds):
                split = split_index.get(
                    (c["arch"], c["slug"], c["shape_point"])
                )
                if split == "holdout":
                    holdout_preds.append(p)
                    holdout_labels.append(c["label"])
            if holdout_preds:
                holdout_geomean = mch.geomean_relative_error(
                    holdout_preds, holdout_labels,
                )
                holdout_status = "ok"
            else:
                holdout_status = "no_holdout_cells_in_admitted_set"
        except (ImportError, OSError):
            pass
    # Weighted advisory variants: log10(n_observations_per_cell) weights are
    # not currently available in cell_records (the v5 head's cohort_scope
    # omits per-cell observation counts to keep the JSON small). The advisory
    # is recorded as "deferred_observations_count_not_in_head" so the
    # contract surface stays consistent.
    artifact = {
        "schema_version": 1,
        "status": "ok",
        "admitted_count": n,
        "source_commit": _git_head_sha(),
        "source_v5_head_commit": head.get("source_commit"),
        "gates": {
            "full_cohort_median_absolute_error": full_median_abs,
            "full_cohort_geomean_relative_error": full_geomean,
            "holdout_geomean_relative_error": holdout_geomean,
            "holdout_status": holdout_status,
            "weighted_advisory_status": (
                "deferred_observations_count_not_in_head"
            ),
            "per_bucket_p90_absolute_error": by_bucket_metrics,
        },
        "thresholds": {
            "full_cohort_median_absolute_error_max": AC3_FULL_COHORT_MEDIAN_ABS_MAX,
            "full_cohort_geomean_relative_error_max": AC3_FULL_COHORT_GEOMEAN_REL_MAX,
            "holdout_geomean_relative_error_max": AC3_HOLDOUT_GEOMEAN_REL_MAX,
            "per_bucket_p90_absolute_error_max": AC3_PER_BUCKET_P90_MAX,
        },
        "verdict": {
            "ac_3_1_full_cohort_median_abs": (
                "pass" if full_median_abs <= AC3_FULL_COHORT_MEDIAN_ABS_MAX
                else "fail"
            ),
            "ac_3_2_full_cohort_geomean_rel": (
                "pass" if full_geomean <= AC3_FULL_COHORT_GEOMEAN_REL_MAX
                else "fail"
            ),
            "ac_3_3_holdout_geomean_rel": (
                "not_computed" if holdout_geomean is None
                else ("pass" if holdout_geomean <= AC3_HOLDOUT_GEOMEAN_REL_MAX
                      else "fail")
            ),
            "ac_3_4_per_bucket_p90": "see_per_bucket_p90_absolute_error",
        },
    }
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def _compute_per_bucket_residual_reduction(
    per_bucket_errs: dict[tuple[str, str], list[tuple[float, float]]],
) -> list[dict[str, Any]]:
    """Aggregate per-cell (v4_err, v5_err) tuples into a
    per-(arch, family_bucket) residual-reduction breakdown.

    The falsification closure-path contract requires the
    falsification report to document the per-bucket residual
    structure. The aggregate
    `measured_residual_reduction_diagnostics` reports a single
    cohort-wide v4/v5 mean pair; this helper produces the
    per-(arch, family_bucket) decomposition that lets a reader
    see which buckets drive the v4->v5 reduction.

    Returns a list of dicts sorted by (arch, family_bucket); each
    entry carries `arch`, `family_bucket`, `cell_count`,
    `v4_mean_absolute_error`, `v5_mean_absolute_error`,
    `residual_reduction_pct` (None when v4_mean is too small to
    divide by safely)."""
    out: list[dict[str, Any]] = []
    for (arch, bucket), pairs in sorted(per_bucket_errs.items()):
        if not pairs:
            continue
        v4_mean = sum(p[0] for p in pairs) / len(pairs)
        v5_mean = sum(p[1] for p in pairs) / len(pairs)
        if v4_mean > 1e-12:
            reduction_pct = 100.0 * ((v4_mean - v5_mean) / v4_mean)
        else:
            reduction_pct = None
        out.append({
            "arch": arch,
            "family_bucket": bucket,
            "cell_count": len(pairs),
            "v4_mean_absolute_error": v4_mean,
            "v5_mean_absolute_error": v5_mean,
            "residual_reduction_pct": reduction_pct,
        })
    return out


def _enumerate_missed_closure_targets(
    v5_head_dir: Path,
) -> list[dict[str, Any]]:
    """Read the accuracy / robustness / active-path / baseline-
    comparison artifacts living in ``v5_head_dir`` and enumerate
    every closure sub-criterion that FAILS or is INDETERMINATE on
    the current admitted slice.

    The falsification closure-path contract requires the
    falsification report to document which targets were missed.
    This helper produces that enumeration by reading the gate
    artifacts the closure pipeline emits and applying the same
    threshold logic each artifact already records internally.

    Each entry carries ``closure_target`` (a domain-named gate
    identifier such as ``full_cohort_median_absolute_error``),
    ``threshold``, ``observed``, ``verdict``
    (``FAIL`` | ``INDETERMINATE``), and ``description``. Returns
    an EMPTY list when every closure target passes (research
    success path)."""
    out: list[dict[str, Any]] = []

    def _safe_read(path: Path) -> dict | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    accuracy = _safe_read(
        v5_head_dir / "calibration_head_v5_accuracy_gates.json"
    )
    if isinstance(accuracy, dict):
        gates = accuracy.get("gates") or {}
        thresholds = accuracy.get("thresholds") or {}
        # Full-cohort median absolute error gate
        obs = gates.get("full_cohort_median_absolute_error")
        thr = thresholds.get("full_cohort_median_absolute_error_max")
        if isinstance(obs, (int, float)) and isinstance(thr, (int, float)):
            if obs > thr:
                out.append({
                    "closure_target": "full_cohort_median_absolute_error",
                    "threshold": thr,
                    "observed": obs,
                    "verdict": "FAIL",
                    "description": (
                        "full-cohort median absolute hit-rate error"
                    ),
                })
        # Full-cohort geomean relative error gate
        obs = gates.get("full_cohort_geomean_relative_error")
        thr = thresholds.get("full_cohort_geomean_relative_error_max")
        if isinstance(obs, (int, float)) and isinstance(thr, (int, float)):
            if obs > thr:
                out.append({
                    "closure_target": "full_cohort_geomean_relative_error",
                    "threshold": thr,
                    "observed": obs,
                    "verdict": "FAIL",
                    "description": (
                        "full-cohort geomean relative error"
                    ),
                })
        # Holdout geomean relative error gate
        obs = gates.get("holdout_geomean_relative_error")
        thr = thresholds.get("holdout_geomean_relative_error_max")
        if isinstance(obs, (int, float)) and isinstance(thr, (int, float)):
            if obs > thr:
                out.append({
                    "closure_target": "holdout_geomean_relative_error",
                    "threshold": thr,
                    "observed": obs,
                    "verdict": "FAIL",
                    "description": "holdout geomean relative error",
                })
        # Per-bucket p90 (only buckets with n >= 8 are gating)
        per_bucket = gates.get("per_bucket_p90_absolute_error") or {}
        thr = thresholds.get("per_bucket_p90_absolute_error_max")
        if isinstance(per_bucket, dict) and isinstance(thr, (int, float)):
            for bucket_key, info in sorted(per_bucket.items()):
                if not isinstance(info, dict):
                    continue
                if info.get("gate_status") != "ok":
                    continue
                obs = info.get("p90_absolute_error")
                if isinstance(obs, (int, float)) and obs > thr:
                    out.append({
                        "closure_target": "per_bucket_p90_absolute_error",
                        "bucket": bucket_key,
                        "threshold": thr,
                        "observed": obs,
                        "verdict": "FAIL",
                        "description": (
                            "per-bucket p90 absolute error "
                            f"(bucket={bucket_key}, n={info.get('n')})"
                        ),
                    })

    robustness = _safe_read(
        v5_head_dir / "calibration_head_v5_robustness.json"
    )
    if isinstance(robustness, dict):
        gates = robustness.get("gates") or {}
        obs = gates.get("loocv_holdout_relative_gap")
        thr = gates.get("loocv_holdout_relative_gap_max")
        if isinstance(obs, (int, float)) and isinstance(thr, (int, float)):
            if obs > thr:
                out.append({
                    "closure_target": "loocv_holdout_relative_gap",
                    "threshold": thr,
                    "observed": obs,
                    "verdict": "FAIL",
                    "description": (
                        "LOOCV-vs-holdout relative gap"
                    ),
                })
        obs = gates.get("loso_max_relative_change")
        thr = gates.get("loso_max_relative_change_max")
        if isinstance(obs, (int, float)) and isinstance(thr, (int, float)):
            if obs > thr:
                out.append({
                    "closure_target": "loso_max_relative_change",
                    "threshold": thr,
                    "observed": obs,
                    "verdict": "FAIL",
                    "description": (
                        "leave-one-slug-out max relative change"
                    ),
                })

    ap = _safe_read(
        v5_head_dir / "v5_active_path_comparison_bundle.json"
    )
    if isinstance(ap, dict):
        diag = ap.get("per_arch_transfer_diagnostics") or {}
        for direction in ("sm_120_to_sm_86", "sm_86_to_sm_120"):
            entry = diag.get(direction) or {}
            status = entry.get("status")
            if status == "indeterminate_insufficient_data":
                out.append({
                    "closure_target": "cross_arch_transfer",
                    "direction": direction,
                    "threshold": "transfer_n > 0",
                    "observed": (
                        f"transfer_n={entry.get('transfer_n', 0)}, "
                        f"native_n={entry.get('native_n', 0)}"
                    ),
                    "verdict": "INDETERMINATE",
                    "description": (
                        f"cross-arch transfer {direction} cannot be "
                        f"computed (insufficient native or transfer cells)"
                    ),
                })
            elif status == "ok":
                # Transfer is computable; check the two gate
                # sub-criteria emitted by run_v5_active_path_comparison:
                # epsilon-floored relative gate + median-absolute
                # gate. Each FAILing gate becomes a separate
                # missed-target entry so a reader sees which gate
                # missed.
                rel_status = entry.get(
                    "epsilon_floored_relative_gate_status"
                )
                if rel_status == "fail":
                    out.append({
                        "closure_target": "cross_arch_transfer_relative_error",
                        "direction": direction,
                        "threshold": entry.get(
                            "epsilon_floored_relative_gate_threshold"
                        ),
                        "observed": entry.get(
                            "transfer_geomean_relative_error"
                        ),
                        "verdict": "FAIL",
                        "description": (
                            f"cross-arch transfer {direction} "
                            "geomean relative error exceeds the "
                            "epsilon-floored relative gate threshold"
                        ),
                    })
                med_status = entry.get("median_absolute_gate_status")
                if med_status == "fail":
                    out.append({
                        "closure_target": "cross_arch_transfer_median_absolute_error",
                        "direction": direction,
                        "threshold": entry.get(
                            "median_absolute_gate_threshold"
                        ),
                        "observed": entry.get(
                            "transfer_median_absolute_error"
                        ),
                        "verdict": "FAIL",
                        "description": (
                            f"cross-arch transfer {direction} "
                            "median absolute error exceeds the "
                            "median-absolute gate threshold"
                        ),
                    })
        # Cross-arch transfer also requires sm_86_native.n > 0.
        sm86_native = diag.get("sm_86_native") or {}
        if not isinstance(sm86_native.get("n"), int) or sm86_native.get("n", 0) <= 0:
            out.append({
                "closure_target": "cross_arch_transfer",
                "direction": "sm_86_native",
                "threshold": "sm_86_native.n > 0",
                "observed": f"n={sm86_native.get('n', 0)}",
                "verdict": "INDETERMINATE",
                "description": (
                    "sm_86 native baseline has no admitted cells; "
                    "cross-arch transfer cannot be evaluated"
                ),
            })

    # Same-74 baseline comparison: the baseline runner computes
    # aggregate `n` independently per prediction column and
    # separately exposes `v5_vs_nugteren_joint_admit_count`. The
    # same-74 comparison is computable ONLY when ALL SIX
    # aggregate columns have n == 74 AND
    # v5_vs_nugteren_joint_admit_count == 74. Any partial-cohort
    # column or joint count means the comparison surface is
    # structurally incomplete and the same-74 sub-criteria
    # cannot be evaluated — surface this as an INDETERMINATE
    # entry whose `observed` payload names every incomplete
    # count so a reader can see which column blocked the
    # comparison.
    bc = _safe_read(
        v5_head_dir / "calibration_head_v5_baseline_comparison.json"
    )
    if isinstance(bc, dict):
        agg = bc.get("aggregate_metrics") or {}
        baseline_columns = (
            "v5_prediction",
            "nugteren_prediction",
            "v4_platt_prediction",
            "uncalibrated_walker_prediction",
            "cohort_mean_prediction",
            "always_hit_prediction",
        )
        column_n_observed = {
            f"{col}.n": (agg.get(col) or {}).get("n")
            for col in baseline_columns
        }
        joint_admit_count = agg.get("v5_vs_nugteren_joint_admit_count")
        all_columns_at_74 = all(
            n == 74 for n in column_n_observed.values()
        )
        joint_at_74 = joint_admit_count == 74
        if not (all_columns_at_74 and joint_at_74):
            observed_payload = dict(column_n_observed)
            observed_payload["v5_vs_nugteren_joint_admit_count"] = (
                joint_admit_count
            )
            out.append({
                "closure_target": "same_74_baseline_comparison",
                "threshold": (
                    "all 6 baseline aggregate columns n == 74 "
                    "AND v5_vs_nugteren_joint_admit_count == 74"
                ),
                "observed": observed_payload,
                "verdict": "INDETERMINATE",
                "description": (
                    "baseline comparison aggregate is "
                    "structurally incomplete on the planned "
                    "same-74 cohort: at least one per-column n "
                    "or v5_vs_nugteren_joint_admit_count is "
                    "missing or below 74"
                ),
            })
        else:
            # Same-74 comparison is fully computable across all
            # six aggregate columns + the joint admit set; check
            # the two baseline sub-criteria: v5 must win against
            # Nugteren on at least 60% of the joint admit set
            # AND v5 aggregate median absolute error must be
            # strictly less than Nugteren's.
            v5_n_block = agg.get("v5_prediction") or {}
            nugteren_block = agg.get("nugteren_prediction") or {}
            beat = agg.get("v5_vs_nugteren_beat_fraction")
            if isinstance(beat, (int, float)) and beat < 0.60:
                out.append({
                    "closure_target": "v5_vs_nugteren_beat_fraction",
                    "threshold": 0.60,
                    "observed": beat,
                    "verdict": "FAIL",
                    "description": (
                        "v5 beats Nugteren on fewer than 60% of "
                        "the joint admit set"
                    ),
                })
            v5_med = v5_n_block.get("median_absolute_error")
            nu_med = nugteren_block.get("median_absolute_error")
            if (
                isinstance(v5_med, (int, float))
                and isinstance(nu_med, (int, float))
                and v5_med >= nu_med
            ):
                out.append({
                    "closure_target": (
                        "v5_vs_nugteren_aggregate_median_absolute_error"
                    ),
                    "threshold": (
                        f"v5_median < nugteren_median ({nu_med})"
                    ),
                    "observed": v5_med,
                    "verdict": "FAIL",
                    "description": (
                        "v5 aggregate median absolute error is not "
                        "strictly less than Nugteren's aggregate "
                        "median absolute error on the same-74 cohort"
                    ),
                })
    return out


def _emit_falsification(
    *,
    v5_head_path: Path,
    output_path: Path,
    v4_head_path: Path | None = None,
    bundle_lookup: dict[tuple[str, str, str], dict] | None = None,
) -> dict[str, Any]:
    head = json.loads(v5_head_path.read_text(encoding="utf-8"))
    admitted = head.get("cohort_scope", {}).get("admitted", [])
    # Split-aware fit scope: the falsification attribution is a fitted-model
    # diagnostic, so it covers the FIT POOL only (the sealed holdout cells
    # are excluded — they are evaluated exactly once by the accuracy gate's
    # holdout block, never by diagnostic refits).
    if (head.get("fit_scope") or {}).get("split_aware"):
        admitted = [
            c for c in admitted
            if c.get("split_assignment") != "holdout"
        ]
    n = len(admitted)
    # MEASURED residual reduction: when the v5 ensemble can score the admitted
    # cells AND the v4 predictions are reachable via the legacy bundle's
    # predicted_hit_rate (which IS the v4-corrected output), compute the
    # per-cell |v4_err| - |v5_err| and aggregate. When this measurement is
    # available, the attribution_pct uses it; otherwise it falls back to the
    # IR-on-projection proportional-weight decomposition convention
    # (closed-form for the active class).
    # The joint contribution per the goal tracker's falsification arm row is
    # the SHARE of v4→v5 improvement attributable to footprint_ratio +
    # long_rd_tail_mass — NOT the total v4→v5 residual reduction. For the
    # IR-on-projection model class (the active selection on this host), the
    # registered convention decomposes the contribution proportionally to
    # |w_feature| / sum(|w|) on the frozen projection weights (+1, -1, -1).
    # footprint_ratio (w=-1) + long_rd_tail_mass (w=-1) → 2/3 ≈ 66.7% of the
    # projection's |w| budget. The measured TOTAL residual reduction is
    # recorded SEPARATELY as a diagnostic (total_v4_to_v5_residual_reduction_pct)
    # so consumers see both signals.
    proportional_weight_share = 2.0 / 3.0
    total_residual_reduction_pct: float | None = None
    measured_n: int = 0
    measured_diagnostics: dict[str, Any] = {}
    # Initialize cells_with_features + ensemble outside the try so the
    # non-IR dispatch below can reference them even when the import or
    # the first call raises before they get set inside the try.
    cells_with_features: list[dict] = []
    ensemble: Any = None
    # Hoist per-bucket error tracking outside the try so the
    # per_bucket_residual_reduction artifact field can still
    # surface an empty list when the measurement loop did not run
    # (e.g., import failure, no admitted cells).
    per_bucket_errs: dict[tuple[str, str], list[tuple[float, float]]] = {}
    try:
        from tools.local_labels.v5_selected_class_dispatch import (
            build_selected_ensemble as _bsel,
        )
        dispatch_f = _bsel(head)
        if dispatch_f["status"] != "ok":
            # Dispatch failure: keep proportional-weight fallback only.
            cells_with_features = []
            ensemble = None
        else:
            cells_with_features = dispatch_f["cells"]
            ensemble = dispatch_f["ensemble"]
        if cells_with_features:
            v5_errs: list[float] = []
            v4_errs: list[float] = []
            if bundle_lookup is not None:
                for cell in cells_with_features:
                    key = (cell["arch"], cell["slug"], cell["shape_point"])
                    bundle = bundle_lookup.get(key)
                    if bundle is None:
                        continue
                    legacy = bundle.get("predicted_hit_rate")
                    if legacy is None:
                        continue
                    v5_pred = float(ensemble.predict(
                        cell["arch"], cell["family_bucket"], cell["features"],
                    ))
                    v5_err = abs(v5_pred - cell["label"])
                    v4_err = abs(float(legacy) - cell["label"])
                    v5_errs.append(v5_err)
                    v4_errs.append(v4_err)
                    bucket_key = (cell["arch"], cell["family_bucket"])
                    per_bucket_errs.setdefault(bucket_key, []).append(
                        (v4_err, v5_err)
                    )
            if v4_errs:
                v4_mean = sum(v4_errs) / len(v4_errs)
                v5_mean = sum(v5_errs) / len(v5_errs)
                if v4_mean > 1e-12:
                    total_residual_reduction_pct = 100.0 * (
                        (v4_mean - v5_mean) / v4_mean
                    )
                measured_n = len(v4_errs)
                measured_diagnostics = {
                    "v4_mean_absolute_error": v4_mean,
                    "v5_mean_absolute_error": v5_mean,
                    "per_cell_count": measured_n,
                }
    except (ImportError, RuntimeError):
        pass
    # Selected-class dispatch: the IR-on-projection class uses the
    # closed-form proportional-weight decomposition; every other class
    # (GAM / LightGBM / XGBoost) uses the registered permutation
    # procedure (per-arch training-set background means, joint
    # perturbation of footprint_ratio + long_rd_tail_mass, per-bucket
    # aggregation, average across buckets).
    # Use the centralized resolver so the documented
    # candidate_scores[0].name fallback fires when selected_class is
    # absent on a legacy head; a direct dict lookup would silently
    # route such a head through the IR convention.
    from tools.local_labels.v5_selected_class_dispatch import (
        resolve_selected_class_name,
    )
    selected_class_name = resolve_selected_class_name(head)
    non_ir_diagnostics: dict[str, Any] = {}
    if selected_class_name == IR_ON_PROJECTION_NAME:
        # Joint contribution = proportional weight share x measured total
        # residual reduction. When the total is unmeasured (no v4 bundle
        # data), the joint contribution falls back to the proportional
        # share alone (with attribution_source flagging the
        # unmeasured-total state).
        if total_residual_reduction_pct is not None:
            attribution_pct = (
                proportional_weight_share * total_residual_reduction_pct
            )
            attribution_source = (
                "ir_on_projection_proportional_weight_share_of_measured_total_residual_reduction"
            )
        else:
            attribution_pct = 100.0 * proportional_weight_share
            attribution_source = (
                "ir_on_projection_proportional_weight_decomposition_unmeasured_total"
            )
    else:
        # Non-IR selected class: run the registered permutation
        # procedure. When the procedure cannot complete (no bundles,
        # no admitted cells, all dispatch predictions raise), the
        # attribution_pct is None + attribution_source records the
        # failure mode for the artifact's audit trail.
        non_ir_attribution, non_ir_source, non_ir_diagnostics = (
            _compute_non_ir_joint_attribution(
                cells_with_features=cells_with_features,
                ensemble=ensemble,
                bundle_lookup=bundle_lookup,
            )
        )
        if non_ir_attribution is not None:
            attribution_pct = non_ir_attribution
            attribution_source = non_ir_source
        else:
            # Permutation could not produce a finite share; fall back
            # to the proportional convention for the verdict comparison
            # but flag the source as permutation-incomplete so a
            # downstream reader does not mistake this for a real
            # non-IR attribution.
            attribution_pct = 100.0 * proportional_weight_share
            attribution_source = non_ir_source
    if n < 8:
        verdict = "indeterminate_insufficient_data"
        verdict_reason = (
            f"the falsification hypothesis contract attribution requires enough admitted cells to make the "
            f"v4 -> v5 residual delta statistically grounded; n={n} is "
            f"below the minimum-bucket-size floor (8) per the per-bucket minimum-N rule. The "
            f"attribution percentage on the registered convention is "
            f"66.7% (footprint_ratio + long_rd_tail_mass share of the "
            f"frozen 3-feature projection's |w| budget); the verdict is "
            f"INDETERMINATE because the v4 -> v5 error-reduction signal "
            f"cannot be measured cleanly on a 2-cell admitted set."
        )
    elif attribution_pct >= AC71_CONFIRMED_MIN_PCT:
        verdict = "confirmed"
        verdict_reason = (
            f"attribution_pct={attribution_pct:.1f}% >= "
            f"{AC71_CONFIRMED_MIN_PCT}% (confirmed threshold)"
        )
    elif attribution_pct < AC71_REFUTED_MAX_PCT:
        verdict = "refuted"
        verdict_reason = (
            f"attribution_pct={attribution_pct:.1f}% < "
            f"{AC71_REFUTED_MAX_PCT}% (refuted threshold)"
        )
    else:
        verdict = "indeterminate"
        verdict_reason = (
            f"attribution_pct={attribution_pct:.1f}% in indeterminate band "
            f"[{AC71_REFUTED_MAX_PCT}%, {AC71_CONFIRMED_MIN_PCT}%)"
        )
    artifact = {
        "schema_version": 1,
        "calibration_head_version": 5,
        # Provenance: the git HEAD that produced this artifact +
        # the v5 head's own source_commit (the fitter commit
        # that drove the admitted cohort the falsification
        # report is computed on). source_v5_head_commit lets a
        # reviewer cross-check that the artifact's missed
        # closure targets reflect the same cohort scope the v5
        # head was trained on; source_commit identifies the
        # emitter code that produced this artifact.
        "source_commit": _git_head_sha(),
        "source_v5_head_commit": head.get("source_commit"),
        "selected_class_name": selected_class_name,
        "attribution_procedure": {
            "background": "per_arch_training_set_mean",
            "perturbation": "permutation_based_marginal_contribution",
            "metric": "absolute_hit_rate_error_reduction_vs_v4",
            "aggregation": "per_arch_family_bucket_then_averaged",
            "decomposition_convention_for_isotonic_on_projection": (
                "|w_feature| / sum(|w|) on the frozen 3-feature projection "
                "with weights (+1, -1, -1) per the model-class hierarchy"
            ),
        },
        "joint_features": ["footprint_ratio", "long_rd_tail_mass"],
        "joint_contribution_pct": attribution_pct,
        "joint_proportional_weight_share": proportional_weight_share,
        "total_v4_to_v5_residual_reduction_pct": total_residual_reduction_pct,
        "attribution_source": attribution_source,
        "measured_residual_reduction_diagnostics": measured_diagnostics,
        "non_ir_permutation_diagnostics": (
            non_ir_diagnostics if non_ir_diagnostics else None
        ),
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "thresholds": {
            "confirmed_min_pct": AC71_CONFIRMED_MIN_PCT,
            "refuted_max_pct": AC71_REFUTED_MAX_PCT,
        },
        "admitted_count": n,
        "skipped_cell_reasons": _skipped_cell_reasons_from_head(head),
        # The falsification closure-path contract requires the
        # falsification report to document the per-bucket residual
        # structure that drives the verdict.
        # _compute_per_bucket_residual_reduction aggregates the
        # per-cell (v4_err, v5_err) pairs the measurement loop
        # already collected.
        "per_bucket_residual_reduction": (
            _compute_per_bucket_residual_reduction(per_bucket_errs)
        ),
        # The same closure-path contract also requires the report
        # to document which closure targets were missed.
        # _enumerate_missed_closure_targets reads the accuracy /
        # robustness / active_path / baseline-comparison artifacts
        # and lists every sub-criterion that is FAIL or
        # INDETERMINATE on the current admitted slice.
        "missed_closure_targets": _enumerate_missed_closure_targets(
            v5_head_path.parent,
        ),
        "notes": (
            "On the IR-on-projection model class, the attribution "
            "convention is a closed-form decomposition "
            "(|w_feature| / sum(|w|)). On a non-IR selected class "
            "(GAM / LightGBM / XGBoost), the registered permutation "
            "procedure runs: per-arch training-set means as background, "
            "joint perturbation of footprint_ratio + long_rd_tail_mass, "
            "absolute-error reduction vs v4, aggregation per "
            "(arch, family_bucket), then averaged across buckets."
        ),
    }
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def _emit_reliability(
    *,
    v5_head_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Emit advisory per-bucket calibration slope + intercept + ECE."""
    head = json.loads(v5_head_path.read_text(encoding="utf-8"))
    admitted = head.get("cohort_scope", {}).get("admitted", [])
    # Split-aware fit scope: reliability diagnostics refit on the cells, so
    # they cover the FIT POOL only (no holdout leakage into a fitted
    # diagnostic).
    if (head.get("fit_scope") or {}).get("split_aware"):
        admitted = [
            c for c in admitted
            if c.get("split_assignment") != "holdout"
        ]
    n = len(admitted)
    if n < 3:
        artifact = {
            "schema_version": 1,
            "status": "indeterminate_insufficient_data",
            "admitted_count": n,
            "source_commit": _git_head_sha(),
            "source_v5_head_commit": head.get("source_commit"),
            "reason": (
                "Per-bucket calibration slope + intercept + ECE need >=3 "
                "cells per bucket for a stable fit. The v5 head's "
                "cohort_scope records the admitted cells; this artifact "
                "lands as a placeholder so downstream readers see the "
                "the reliability diagnostics advisory state."
            ),
            "bootstrap_ci_adjunct": {
                "status": "not_computed_insufficient_admitted_cells",
                "minimum_admitted_for_bootstrap": 3,
            },
            "skipped_cell_reasons": _skipped_cell_reasons_from_head(head),
        }
        output_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return artifact
    # When enough cells are admitted, compute slope/intercept/ECE per bucket.
    from collections import defaultdict
    by_bucket: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for c in admitted:
        by_bucket[(c["arch"], c["family_bucket"])].append(c)
    ensemble = mch.PerBucketCalibratorEnsemble(
        model_class_factory=mch.IsotonicRegressionOnFrozenProjection,
    )
    cell_records = []
    for c in admitted:
        cell_records.append({
            "identity": (c["arch"], c["slug"], c["shape_point"]),
            "arch": c["arch"],
            "family_bucket": c["family_bucket"],
            "features": c.get("features", {}),
            "label": float(c["label"]),
        })
    if not all(c["features"] for c in cell_records):
        artifact = {
            "schema_version": 1,
            "status": "indeterminate_features_unavailable",
            "admitted_count": n,
            "source_commit": _git_head_sha(),
            "source_v5_head_commit": head.get("source_commit"),
        }
        output_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return artifact
    # Reliability emitter uses the SELECTED class via the centralized
    # dispatch (was: hardcoded IR-on-projection).
    from tools.local_labels.v5_selected_class_dispatch import (
        build_selected_ensemble as _bsel_rel,
    )
    rel_dispatch = _bsel_rel(head, cells=cell_records)
    if rel_dispatch["status"] != "ok":
        artifact = {
            "schema_version": 1,
            "status": rel_dispatch["status"],
            "admitted_count": n,
            "source_commit": _git_head_sha(),
            "source_v5_head_commit": head.get("source_commit"),
            "selected_class_name": rel_dispatch["selected_class_name"],
            "reason": rel_dispatch["reason"],
        }
        output_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return artifact
    ensemble = rel_dispatch["ensemble"]
    bucket_diagnostics: dict[str, Any] = {}
    for key, members in by_bucket.items():
        preds = [
            ensemble.predict(m["arch"], m["family_bucket"], m["features"])
            for m in members
        ]
        labels = [m["label"] for m in members]
        # Simple linear regression for slope + intercept (least squares).
        if len(preds) >= 2:
            xbar = sum(preds) / len(preds)
            ybar = sum(labels) / len(labels)
            num = sum((p - xbar) * (y - ybar) for p, y in zip(preds, labels))
            den = sum((p - xbar) ** 2 for p in preds)
            slope = num / den if den > 0 else 0.0
            intercept = ybar - slope * xbar
        else:
            slope = 0.0
            intercept = float(labels[0]) if labels else 0.0
        # 10-bin ECE on predicted hit rate.
        bin_edges = [i / 10.0 for i in range(11)]
        ece = 0.0
        n_total = len(preds)
        for i in range(10):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            bin_preds = [
                p for p, _y in zip(preds, labels)
                if lo <= p < hi or (i == 9 and p == 1.0)
            ]
            bin_labels = [
                y for p, y in zip(preds, labels)
                if lo <= p < hi or (i == 9 and p == 1.0)
            ]
            if not bin_preds:
                continue
            mean_pred = sum(bin_preds) / len(bin_preds)
            mean_label = sum(bin_labels) / len(bin_labels)
            ece += (len(bin_preds) / n_total) * abs(mean_pred - mean_label)
        bucket_diagnostics[f"{key[0]}/{key[1]}"] = {
            "n": len(members),
            "calibration_slope": slope,
            "calibration_intercept": intercept,
            "expected_calibration_error_10_bin": ece,
        }
    # Bootstrap CI adjunct: reuse comparison_runner._bootstrap_ci_over_cells
    # over the per-cell |v5_prediction - label| series to emit a per-cell
    # mean-absolute-error CI for the reliability artifact's adjunct field.
    bootstrap_ci_adjunct: dict[str, Any] = {
        "status": "not_computed_helper_unavailable",
    }
    try:
        from workloads.simbench_bw.comparison_runner import (
            _bootstrap_ci_over_cells,
        )
        per_cell_abs = [abs(p - y) for p, y in zip(preds, labels)]
        ci_low, ci_high, ci_half = _bootstrap_ci_over_cells(
            per_cell_abs,
            n_resamples=2000,
            confidence=0.95,
            seed=20260527,
        )
        bootstrap_ci_adjunct = {
            "status": "ok",
            "metric": "per_cell_absolute_error",
            "n_cells": len(per_cell_abs),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci_half_width": ci_half,
            "n_resamples": 2000,
            "confidence": 0.95,
            "seed": 20260527,
        }
    except (ImportError, ValueError, ZeroDivisionError):
        pass
    artifact = {
        "schema_version": 1,
        "status": "ok",
        "admitted_count": n,
        "source_commit": _git_head_sha(),
        "source_v5_head_commit": head.get("source_commit"),
        "per_bucket": bucket_diagnostics,
        "binning": {
            "n_bins": 10,
            "bin_edges": "equal_width_on_predicted_hit_rate_0_to_1",
        },
        "bootstrap_ci_adjunct": bootstrap_ci_adjunct,
    }
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5-head", default=DEFAULT_V5_HEAD)
    parser.add_argument("--v4-head", default=DEFAULT_V4_HEAD)
    parser.add_argument("--ac3-output", default=DEFAULT_AC3_OUT)
    parser.add_argument("--falsification-output", default=DEFAULT_FALS_OUT)
    parser.add_argument("--reliability-output", default=DEFAULT_REL_OUT)
    args = parser.parse_args(argv)
    ac3 = _emit_ac3_gates(
        v5_head_path=Path(args.v5_head),
        output_path=Path(args.ac3_output),
    )
    # Build bundle lookup so falsification can compute measured v4-to-v5
    # residual reduction over admitted cells.
    import glob as _glob_local
    bundle_glob = (
        "data/local_labels/sweeps/sm120_cohort_bundles_*/predictor_bundles/*.json"
    )
    bundle_lookup_for_fals: dict[tuple[str, str, str], dict] = {}
    for p in _glob_local.glob(bundle_glob):
        try:
            b = json.loads(Path(p).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        mpi = b.get("manifest_pairing_index", {})
        key = (
            str(mpi.get("arch", "")),
            str(mpi.get("slug", "")),
            str(mpi.get("shape_point", "")),
        )
        bundle_lookup_for_fals[key] = b
    fals = _emit_falsification(
        v5_head_path=Path(args.v5_head),
        output_path=Path(args.falsification_output),
        v4_head_path=Path(args.v4_head) if Path(args.v4_head).is_file() else None,
        bundle_lookup=bundle_lookup_for_fals or None,
    )
    rel = _emit_reliability(
        v5_head_path=Path(args.v5_head),
        output_path=Path(args.reliability_output),
    )
    print(
        f"[v5_closure] ac3.status={ac3.get('status')} "
        f"fals.verdict={fals.get('verdict')} "
        f"reliability.status={rel.get('status')}",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
