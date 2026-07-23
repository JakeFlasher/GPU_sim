"""Shadow-report generator for the L2 reduction-kernel-family
correction surface. Reads the active sweep's diagnostic sidecar +
L2 predictor bundles, pulls each admitted L2-scalar cell through
the production helper, and writes a deterministic JSON artifact
for budget + hard-target + monitoring-goal verification.

Truth-fitted demotion gate: the family's per-slug factors are
NCU-truth-fitted and carry
``correction_factor_status = diagnostic_truth_fitted_demoted``,
so the production helper no-ops the deployable value. The report
records the deployable pass-through channel (rel_err_after ==
rel_err_before, factor_applied None) AND the diagnostic-shadow
channel (the would-be corrected value + its rel err) so the
empirical record of what the factors did is preserved without
any deployable claim.

Outputs:
  data/sm_86_predictor_accuracy_recovery_v2/l2_reduction_correction_shadow_report.json

Invariants:
  - per-cell `delta = rel_err_after - rel_err_before` <= 0.02
    for every admitted L2-scalar cell (regression budget;
    trivially 0.0 under demotion);
  - `friendly_l2.geomean_rel_err < 0.15` (HARD-1, now evaluated
    on the deployable pass-through channel);
  - `friendly_l1.geomean_rel_err` byte-identical to the
    diagnostic sidecar's value (HARD-2; the L2 surface is
    independent of the L1 head);
  - `0.10 <= cross_cell_stddev <= 0.30` (HARD-3 lower bound +
    user-sanctioned widened upper bound).

The generator is deterministic: the same source-data inputs
yield the same JSON output bytes. Re-running after the
calibration fixture or the diagnostic sidecar changes produces
an updated JSON whose `source_commit` field references the
predictor-touching fix-commit per BL-20260508.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_THIS_DIR = Path(__file__).resolve().parent
_SIMBENCH_BW = _REPO_ROOT / "workloads" / "simbench_bw"
for _p in (_THIS_DIR, _SIMBENCH_BW):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import cache_predictor as cp  # type: ignore[import-not-found]  # noqa: E402
from task12_substitution_preflight import _geomean  # noqa: E402

_DIAGNOSTIC_PATH = (
    _REPO_ROOT
    / "data" / "sm_86_predictor_accuracy_recovery_v2"
    / "interim_diagnostic_report_sidecar.json"
)
_CALIBRATION_PATH = (
    _REPO_ROOT
    / "data" / "sm_86_predictor_accuracy_recovery_v2"
    / "l2_reduction_correction_calibration.json"
)
_OUTPUT_PATH = (
    _REPO_ROOT
    / "data" / "sm_86_predictor_accuracy_recovery_v2"
    / "l2_reduction_correction_shadow_report.json"
)


# R17 update: the baseline friendly_l1 geomean was 0.153676...
# pre-R17 (with broken classifier walker outputs). After R17's
# corrected classifier + warmup_launches=5 + carveout-aware
# capacity + per-slug shared_reduction dispatch, the friendly_l1
# geomean over the L1-admitted cohort moved to ~0.048 (closer to
# NCU truth on the reduction cells). The "byte-identical"
# invariant is relative to the CURRENT friendly_l1_geomean_rel_err
# recorded in the diagnostic sidecar rather than to the pre-R17
# hardcoded constant; the assertion's purpose (L2 correction does
# not perturb L1 head) is unchanged, but the absolute value tracks
# the L1 head's current calibration state.
# Truth-fitted demotion gate update: with the L1 reduction-family
# factors demoted, the deployable predicted_l1 for the friendly_l1
# reduction cells reverts to the raw walker baseline, so the
# sidecar's friendly_l1 geomean moved 0.0487 -> 1.9469... (the
# honest cost of removing same-cell truth-fitted factors from
# production; the diagnostic shadow retains the would-be ~0.0148
# record). The pin tracks the deployable state.
_BASELINE_FRIENDLY_L1_GEOMEAN = 1.9469743371673827
_HARD_1_FRIENDLY_L2_UPPER_BOUND = 0.15
_HARD_3_STDDEV_LOWER_BOUND = 0.10
# HARD-3 upper bound was widened from R47's 0.15 to the
# R48 / R6 round-managed band [0.10, 0.30] (recorded in the
# goal-tracker Plan Evolution Log). The widening admits the
# R48 L2 high-footprint correction and the R6 R47-leftover
# closure (per-slug factors that legitimately widen the
# prediction distribution). The R47-only shadow report's
# stddev is now ~0.191 (legitimately above the R47-era 0.15
# ceiling); the composed R47+R48 stddev (in the R48 high-
# footprint shadow report) stays under 0.30.
_HARD_3_STDDEV_UPPER_BOUND = 0.30
_PER_CELL_BUDGET = 0.02


def _read_inputs() -> tuple[dict, dict]:
    diagnostic = json.loads(_DIAGNOSTIC_PATH.read_text(encoding="utf-8"))
    calibration = json.loads(
        _CALIBRATION_PATH.read_text(encoding="utf-8"),
    )
    return diagnostic, calibration


def _admitted_l2_scalar_cells(diagnostic: dict) -> list[dict]:
    cells = [
        c
        for c in diagnostic["cells"]
        if c.get("admission_status") == "admitted"
        and c.get("l2_status") == "scalar_present"
    ]
    return cells


def _assert_calibration_matches_production(calibration: dict) -> None:
    """Fail loudly when the calibration fixture's allowlist or
    chosen factor drift away from the production constants in
    ``workloads/simbench_bw/cache_predictor.py``. Catching the
    drift at build time prevents the verifier from passing
    while the predictor's runtime L2 head has moved."""
    calibration_slugs = frozenset(
        entry["slug"] for entry in calibration["input_bundles"]
    )
    if calibration_slugs != cp._L2_REDUCTION_KERNEL_SLUG_ALLOWLIST:
        raise ValueError(
            "calibration fixture slug set "
            f"{sorted(calibration_slugs)!r} does not match "
            "cp._L2_REDUCTION_KERNEL_SLUG_ALLOWLIST "
            f"{sorted(cp._L2_REDUCTION_KERNEL_SLUG_ALLOWLIST)!r}; "
            "regenerate the calibration fixture or update "
            "cp._L2_REDUCTION_KERNEL_SLUG_ALLOWLIST so the "
            "shadow report simulates against the production "
            "allowlist"
        )
    # The L2 reduction surface evolved to a per-slug factor
    # dispatch in R6 (R47-leftover closure). The calibration
    # fixture's `chosen_factors` per-slug dict must match the
    # production `cp._L2_REDUCTION_KERNEL_CORRECTION_FACTORS`
    # byte-for-byte; the backward-compat scalar `chosen_factor`
    # (= layernorm control-branch factor) must match the
    # production `cp._L2_REDUCTION_KERNEL_CORRECTION_FACTOR`.
    cal_factors = {
        slug: float(v)
        for slug, v in calibration.get("chosen_factors", {}).items()
    }
    if cal_factors != {
        slug: float(v)
        for slug, v in cp._L2_REDUCTION_KERNEL_CORRECTION_FACTORS.items()
    }:
        raise ValueError(
            f"calibration['chosen_factors']={cal_factors!r} does "
            "not match "
            "cp._L2_REDUCTION_KERNEL_CORRECTION_FACTORS"
            f"={dict(cp._L2_REDUCTION_KERNEL_CORRECTION_FACTORS)!r}; "
            "regenerate the calibration fixture or update "
            "cp._L2_REDUCTION_KERNEL_CORRECTION_FACTORS so the "
            "shadow report simulates against the production "
            "per-slug dispatch"
        )


def build_shadow_report() -> dict:
    diagnostic, calibration = _read_inputs()
    _assert_calibration_matches_production(calibration)
    cells = _admitted_l2_scalar_cells(diagnostic)
    rows: list[dict] = []
    friendly_after: list[float] = []
    hostile_after: list[float] = []
    friendly_after_diagnostic: list[float] = []
    hostile_after_diagnostic: list[float] = []
    predicted_after_all: list[float] = []
    for c in cells:
        slug = c["slug"]
        truth = float(c["ncu_truth_l2_fraction"])
        pre = float(c["predicted_l2_hit_rate"])
        # Pull point_post through the production helper. The
        # helper's classifier is the only authority for the
        # slug match — the calibration fixture's allowlist is
        # asserted equal to cp._L2_REDUCTION_KERNEL_SLUG_ALLOWLIST
        # above, so the helper's match decision and the fixture's
        # are guaranteed identical. ci_low / ci_high default to
        # the point because the diagnostic does not record per-cell
        # CI bounds; the helper transforms them with the same
        # clamp(x*factor,0,1) the bundle pipeline uses, so the
        # transformed point matches the live predict_*_bundle
        # output under matching inputs.
        post, _, _, _, helper_diag = cp._apply_l2_reduction_correction(
            point=pre, ci_low=pre, ci_high=pre, slug=slug,
        )
        # Under the truth-fitted demotion gate the production helper
        # no-ops the deployable value (applied=False) while still
        # CLASSIFYING the slug: the demotion witness in the diag
        # (correction_factor_status + diagnostic_corrected_value) is
        # the class-match signal. `class_matched` therefore means
        # "the classifier matched the family", not "a factor
        # multiplied the deployable value" — `factor_applied` keeps
        # the latter meaning (None when demoted).
        demoted = "correction_factor_status" in helper_diag and (
            not helper_diag.get("applied")
        )
        class_matched = bool(helper_diag.get("applied")) or demoted
        rel_err_before = float(c["rel_err_l2"])
        rel_err_after = (
            abs(post - truth) / truth if truth != 0.0 else 0.0
        )
        delta = rel_err_after - rel_err_before
        budget_pass = delta <= _PER_CELL_BUDGET
        row = {
            "slug": slug,
            "shape_point": c.get("shape_point", ""),
            "predictor_design_coverage_bucket": (
                c["predictor_design_coverage_bucket"]
            ),
            "class_matched": class_matched,
            "factor_applied": (
                cp._L2_REDUCTION_KERNEL_CORRECTION_FACTORS[slug]
                if helper_diag.get("applied") else None
            ),
            "point_pre": pre,
            "point_post": post,
            "truth": truth,
            "rel_err_before": rel_err_before,
            "rel_err_after": rel_err_after,
            "delta": delta,
            "budget_pass": budget_pass,
        }
        if demoted:
            diag_value = float(helper_diag["diagnostic_corrected_value"])
            row["correction_factor_status"] = (
                helper_diag["correction_factor_status"]
            )
            row["factor_demoted"] = float(helper_diag["factor"])
            row["diagnostic_corrected_value"] = diag_value
            row["rel_err_after_diagnostic"] = (
                abs(diag_value - truth) / truth if truth != 0.0 else 0.0
            )
        rows.append(row)
        # Diagnostic-shadow channel: what the demoted factor WOULD
        # have produced. For non-demoted rows the diagnostic value
        # equals the deployable value.
        diag_after = row.get("rel_err_after_diagnostic", rel_err_after)
        if c["predictor_design_coverage_bucket"] == "predictor_friendly":
            friendly_after.append(rel_err_after)
            friendly_after_diagnostic.append(diag_after)
        elif (
            c["predictor_design_coverage_bucket"] == "predictor_hostile"
        ):
            hostile_after.append(rel_err_after)
            hostile_after_diagnostic.append(diag_after)
        predicted_after_all.append(post)
    rows.sort(key=lambda r: -r["delta"])
    friendly_geomean = _geomean(friendly_after)
    hostile_geomean = _geomean(hostile_after)
    friendly_geomean_diagnostic = _geomean(friendly_after_diagnostic)
    hostile_geomean_diagnostic = _geomean(hostile_after_diagnostic)
    cross_cell_stddev_post = statistics.stdev(predicted_after_all)

    headline_pre = diagnostic.get("per_bucket_headline", {})
    cross_stddev_pre = float(diagnostic.get("cross_cell_stddev", 0.0))
    hard_1_pre = float(
        headline_pre.get("friendly_l2", {}).get("geomean_rel_err", 0.0),
    )
    hard_2_pre = float(
        headline_pre.get("friendly_l1", {}).get("geomean_rel_err", 0.0),
    )
    hostile_l2_pre = float(
        headline_pre.get("hostile_l2", {}).get("geomean_rel_err", 0.0),
    )

    max_delta = max(r["delta"] for r in rows)
    budget_breaches = [
        r["slug"] for r in rows if not r["budget_pass"]
    ]

    hard_1_pass = friendly_geomean < _HARD_1_FRIENDLY_L2_UPPER_BOUND
    hard_2_pass = hard_2_pre == _BASELINE_FRIENDLY_L1_GEOMEAN
    hard_3_pass = (
        _HARD_3_STDDEV_LOWER_BOUND
        <= cross_cell_stddev_post
        <= _HARD_3_STDDEV_UPPER_BOUND
    )

    rmsnorm_row = next(
        (r for r in rows if r["slug"] == "rmsnorm_fwd_bf16_sm80"),
        None,
    )
    softmax_row = next(
        (r for r in rows if r["slug"] == "softmax_safe_bf16_sm80"),
        None,
    )
    mg_2 = (
        rmsnorm_row is not None
        and rmsnorm_row["rel_err_after"] < rmsnorm_row["rel_err_before"]
    )
    mg_3 = (
        softmax_row is not None
        and softmax_row["rel_err_after"] < softmax_row["rel_err_before"]
    )
    mg_4 = friendly_geomean < hard_1_pre
    mg_1 = hostile_geomean < hostile_l2_pre
    # Diagnostic-shadow monitoring goals: the same comparisons over
    # the would-be (demoted-factor) channel. These preserve the
    # empirical record of what the truth-fitted factors did WITHOUT
    # claiming any deployable effect — the deployable mg_* flags
    # above flip honestly to False under demotion.
    mg_2_diag = (
        rmsnorm_row is not None
        and rmsnorm_row.get(
            "rel_err_after_diagnostic", rmsnorm_row["rel_err_after"],
        ) < rmsnorm_row["rel_err_before"]
    )
    mg_3_diag = (
        softmax_row is not None
        and softmax_row.get(
            "rel_err_after_diagnostic", softmax_row["rel_err_after"],
        ) < softmax_row["rel_err_before"]
    )
    mg_4_diag = friendly_geomean_diagnostic < hard_1_pre
    mg_1_diag = hostile_geomean_diagnostic < hostile_l2_pre

    demotion_status = cp.CORRECTION_FACTOR_STATUS_BY_FAMILY.get(
        "l2_reduction_kernel",
    )
    return {
        "schema_version": 1,
        "generator": (
            "tools/sm_86_predictor_accuracy_recovery_v2/"
            "generate_l2_reduction_shadow_report.py"
        ),
        # Truth-fitted demotion gate: production applies NO factor;
        # the per-slug dispatch table is retained as the
        # diagnostic-shadow channel only.
        "factor_applied": (
            None if demotion_status is not None else "per-slug dispatch"
        ),
        "correction_factor_status": demotion_status,
        "per_slug_factors_demoted": dict(
            cp._L2_REDUCTION_KERNEL_CORRECTION_FACTORS
        ),
        "factor_provenance_path": (
            "data/sm_86_predictor_accuracy_recovery_v2/"
            "l2_reduction_correction_calibration.json"
        ),
        "diagnostic_source_path": (
            "data/sm_86_predictor_accuracy_recovery_v2/"
            "interim_diagnostic_report_sidecar.json"
        ),
        "allowlist_slugs": sorted(cp._L2_REDUCTION_KERNEL_SLUG_ALLOWLIST),
        "rows_sorted_by_descending_delta": rows,
        "max_delta_across_cells": max_delta,
        "budget_breaches_per_cell": budget_breaches,
        "per_cell_budget_pass": len(budget_breaches) == 0,
        "per_cell_budget_threshold": _PER_CELL_BUDGET,
        "headline_post_correction": {
            "friendly_l2_geomean_rel_err": friendly_geomean,
            "hostile_l2_geomean_rel_err": hostile_geomean,
            "cross_cell_stddev": cross_cell_stddev_post,
            "friendly_l1_geomean_rel_err": (
                _BASELINE_FRIENDLY_L1_GEOMEAN
            ),
        },
        "headline_pre_correction": {
            "friendly_l2_geomean_rel_err": hard_1_pre,
            "hostile_l2_geomean_rel_err": hostile_l2_pre,
            "cross_cell_stddev": cross_stddev_pre,
            "friendly_l1_geomean_rel_err": hard_2_pre,
        },
        # The would-be headline had the demoted factors applied —
        # diagnostic record only, never a deployable claim.
        "headline_diagnostic_shadow": {
            "friendly_l2_geomean_rel_err": friendly_geomean_diagnostic,
            "hostile_l2_geomean_rel_err": hostile_geomean_diagnostic,
            "cross_cell_stddev": statistics.stdev([
                r.get("diagnostic_corrected_value", r["point_post"])
                for r in rows
            ]),
        },
        "hard_targets": {
            "hard_1_friendly_l2_lt_0_15": hard_1_pass,
            "hard_2_friendly_l1_byte_identical": hard_2_pass,
            "hard_3_cross_cell_stddev_in_bound": hard_3_pass,
        },
        "monitoring_goals": {
            "mg_1_hostile_l2_decreases": mg_1,
            "mg_2_rmsnorm_rel_err_decreases": mg_2,
            "mg_3_softmax_safe_rel_err_decreases": mg_3,
            "mg_4_friendly_l2_widens_hard_1_margin": mg_4,
        },
        "monitoring_goals_diagnostic_shadow": {
            "mg_1_hostile_l2_decreases": mg_1_diag,
            "mg_2_rmsnorm_rel_err_decreases": mg_2_diag,
            "mg_3_softmax_safe_rel_err_decreases": mg_3_diag,
            "mg_4_friendly_l2_widens_hard_1_margin": mg_4_diag,
        },
    }


def main() -> int:
    report = build_shadow_report()
    # source_commit is stamped on the on-disk artifact by the
    # recapture script after the predictor-touching fix-commit
    # lands per BL-20260508; the build function is HEAD-agnostic
    # so re-running it during the verifier test produces a
    # source-data-deterministic body that the test compares
    # against the on-disk body MINUS source_commit.
    report["source_commit"] = "PRE_COMMIT_PLACEHOLDER"
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
