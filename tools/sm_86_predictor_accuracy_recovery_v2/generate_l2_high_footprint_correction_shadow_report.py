"""Shadow-report generator for the R48 L2 high-footprint-
low-reuse correction surface, plus the R48 L1
reduction-subfamily split. Reads the active sweep's diagnostic
sidecar + both R48 calibration fixtures, pulls every admitted
scalar-present cell through the production helpers, and writes a
deterministic JSON artifact for budget + hard-target
verification.

Truth-fitted demotion gate: every family this report covers
(l1_silu_elementwise, l1_shared_reduction,
l2_high_footprint_low_reuse — and the upstream
l2_reduction_kernel in the simulated chain) carries
``correction_factor_status = diagnostic_truth_fitted_demoted``,
so the production helpers no-op the deployable value. The report
records the deployable pass-through channel (for L1 cells the
deployable value is the WALKER BASELINE itself) AND the
diagnostic-shadow channel (the would-be corrected value + its
rel err), preserving the empirical record without any deployable
claim.

Outputs:
  data/sm_86_predictor_accuracy_recovery_v2/l2_high_footprint_correction_shadow_report.json

Invariants:
  - per-cell L2 ``delta = rel_err_after - rel_err_before`` <=
    0.02 for every admitted L2-scalar cell (trivially 0.0 under
    demotion);
  - per-cell L1 ``delta`` <= 0.02 evaluated on the deployable
    channel — under demotion the deployable value reverts to the
    walker baseline, so cells whose truth-fitted factor masked a
    walker miss show the honest (possibly breaching) delta;
  - ``friendly_l2.geomean_rel_err < 0.15`` (HARD-1, deployable
    channel);
  - ``friendly_l1.geomean_rel_err < 0.20`` AND non-regression
    vs the R47-post baseline 0.153676764668927 (HARD-2,
    deployable channel — expected to FAIL honestly under
    demotion; the flip is the documented consequence of
    removing truth-fitted factors from production);
  - ``0.10 <= cross_cell_stddev <= 0.30`` (HARD-3 R48-widened
    bounds per DEC-13).

The generator is deterministic: the same source-data inputs
yield the same JSON output bytes. Re-running after the
calibration fixtures, the predictor constants, or the diagnostic
sidecar change produces an updated JSON whose ``source_commit``
field references the predictor-touching fix-commit per
BL-20260508.

The simulator chain mirrors the production pipeline: for each
admitted L2-scalar cell, the corrected point is the result of
``_apply_l2_high_footprint_correction(_apply_l2_reduction_correction(pre))``
so the per-cell delta accurately reflects what the bundle
pipeline would emit. For L1 cells the chain is just
``_apply_l1_reduction_correction(walker_baseline)`` because the
diagnostic sidecar's ``predicted_l1_hit_rate`` was emitted under
R46 (pre-R48 split); we back the walker baseline out by dividing
by the R46 factor 0.32 for the slugs R46 touched.
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
_L1_CALIBRATION_PATH = (
    _REPO_ROOT
    / "data" / "sm_86_predictor_accuracy_recovery_v2"
    / "l1_reduction_subfamily_calibration.json"
)
_L2_CALIBRATION_PATH = (
    _REPO_ROOT
    / "data" / "sm_86_predictor_accuracy_recovery_v2"
    / "l2_high_footprint_correction_calibration.json"
)
_OUTPUT_PATH = (
    _REPO_ROOT
    / "data" / "sm_86_predictor_accuracy_recovery_v2"
    / "l2_high_footprint_correction_shadow_report.json"
)


_R47_FRIENDLY_L1_GEOMEAN = 0.153676764668927
_R46_L1_REDUCTION_FACTOR = 0.32  # pre-R48 R46 anchor; used to
# back the walker baseline out of the R46-corrected
# `predicted_l1_hit_rate` in the diagnostic sidecar so the L1
# subfamily simulator can re-apply the new per-subfamily factor.
_HARD_1_FRIENDLY_L2_UPPER_BOUND = 0.15
_HARD_2_FRIENDLY_L1_UPPER_BOUND = 0.20
_HARD_3_STDDEV_LOWER_BOUND = 0.10
_HARD_3_STDDEV_UPPER_BOUND = 0.30
_PER_CELL_BUDGET = 0.02


def _read_inputs() -> tuple[dict, dict, dict]:
    diagnostic = json.loads(_DIAGNOSTIC_PATH.read_text(encoding="utf-8"))
    l1_calibration = json.loads(
        _L1_CALIBRATION_PATH.read_text(encoding="utf-8"),
    )
    l2_calibration = json.loads(
        _L2_CALIBRATION_PATH.read_text(encoding="utf-8"),
    )
    return diagnostic, l1_calibration, l2_calibration


def _admitted_l2_scalar_cells(diagnostic: dict) -> list[dict]:
    return [
        c
        for c in diagnostic["cells"]
        if c.get("admission_status") == "admitted"
        and c.get("l2_status") == "scalar_present"
    ]


def _admitted_l1_scalar_cells(diagnostic: dict) -> list[dict]:
    return [
        c
        for c in diagnostic["cells"]
        if c.get("admission_status") == "admitted"
        and c.get("l1_status") == "scalar_present"
    ]


def _assert_l2_calibration_matches_production(calibration: dict) -> None:
    """Fail loudly when the L2 high-footprint calibration
    fixture's allowlist or per-slug factors drift away from the
    production constants in ``workloads/simbench_bw/cache_predictor.py``.
    Catching the drift at build time prevents the verifier from
    passing while the predictor's runtime L2 head has moved."""
    calibration_slugs = frozenset(
        entry["slug"] for entry in calibration["input_bundles"]
    )
    if (
        calibration_slugs
        != cp._L2_HIGH_FOOTPRINT_LOW_REUSE_SLUG_ALLOWLIST
    ):
        raise ValueError(
            "L2 high-footprint calibration fixture slug set "
            f"{sorted(calibration_slugs)!r} does not match "
            "cp._L2_HIGH_FOOTPRINT_LOW_REUSE_SLUG_ALLOWLIST "
            f"{sorted(cp._L2_HIGH_FOOTPRINT_LOW_REUSE_SLUG_ALLOWLIST)!r}"
        )
    cal_factors = calibration["chosen_factors"]
    prod_factors = cp._L2_HIGH_FOOTPRINT_LOW_REUSE_CORRECTION_FACTORS
    for slug, cal_factor in cal_factors.items():
        cal_factor_f = float(cal_factor)
        prod_factor = prod_factors.get(slug)
        if prod_factor is None:
            raise ValueError(
                f"L2 high-footprint calibration declares slug "
                f"{slug!r} but production has no matching entry"
            )
        if cal_factor_f != float(prod_factor):
            raise ValueError(
                f"L2 high-footprint calibration['chosen_factors']"
                f"[{slug!r}]={cal_factor_f!r} does not match "
                f"cp._L2_HIGH_FOOTPRINT_LOW_REUSE_CORRECTION_FACTORS"
                f"[{slug!r}]={prod_factor!r}"
            )


def _assert_l1_calibration_matches_production(calibration: dict) -> None:
    """Fail loudly when the L1 subfamily calibration's
    allowlists or per-subfamily factors drift away from the
    production constants."""
    subfamilies = calibration["subfamilies"]
    silu_slugs = frozenset(subfamilies["silu_elementwise"]["allowlist_slugs"])
    if silu_slugs != cp._L1_SILU_ELEMENTWISE_SLUG_ALLOWLIST:
        raise ValueError(
            "L1 silu_elementwise calibration allowlist "
            f"{sorted(silu_slugs)!r} does not match "
            f"{sorted(cp._L1_SILU_ELEMENTWISE_SLUG_ALLOWLIST)!r}"
        )
    silu_factor = float(subfamilies["silu_elementwise"]["chosen_factor"])
    if silu_factor != cp._L1_SILU_ELEMENTWISE_CORRECTION_FACTOR:
        raise ValueError(
            "L1 silu_elementwise calibration factor "
            f"{silu_factor!r} does not match "
            f"{cp._L1_SILU_ELEMENTWISE_CORRECTION_FACTOR!r}"
        )
    shared_slugs = frozenset(
        subfamilies["shared_reduction"]["allowlist_slugs"]
    )
    if shared_slugs != cp._L1_SHARED_REDUCTION_SLUG_ALLOWLIST:
        raise ValueError(
            "L1 shared_reduction calibration allowlist "
            f"{sorted(shared_slugs)!r} does not match "
            f"{sorted(cp._L1_SHARED_REDUCTION_SLUG_ALLOWLIST)!r}"
        )
    shared_factor = float(
        subfamilies["shared_reduction"]["chosen_factor"]
    )
    if shared_factor != cp._L1_SHARED_REDUCTION_CORRECTION_FACTOR:
        raise ValueError(
            "L1 shared_reduction calibration factor "
            f"{shared_factor!r} does not match "
            f"{cp._L1_SHARED_REDUCTION_CORRECTION_FACTOR!r}"
        )
    # R18 per-slug parity guard (mirrors R6/R7's L2 reduction
    # `chosen_factors` byte-identity check): the calibration
    # fixture must record the same per-slug factor dict the
    # production helper dispatches against. A future drift of
    # any per-slug value (e.g., the rmsnorm or softmax_safe
    # factor) without lockstep production update will be
    # caught by this assertion.
    shared_per_slug = subfamilies["shared_reduction"].get(
        "chosen_factors"
    )
    if shared_per_slug is None:
        raise ValueError(
            "L1 shared_reduction calibration is missing the "
            "`chosen_factors` per-slug dict (R18 lockstep "
            "requirement: the fixture must mirror "
            "`cp._L1_SHARED_REDUCTION_CORRECTION_FACTORS`)"
        )
    fixture_per_slug = {
        slug: float(value)
        for slug, value in shared_per_slug.items()
    }
    if fixture_per_slug != cp._L1_SHARED_REDUCTION_CORRECTION_FACTORS:
        raise ValueError(
            "L1 shared_reduction calibration "
            "`chosen_factors` per-slug dispatch "
            f"{fixture_per_slug!r} does not match "
            f"{cp._L1_SHARED_REDUCTION_CORRECTION_FACTORS!r} "
            "(production); refresh "
            "`data/sm_86_predictor_accuracy_recovery_v2/"
            "l1_reduction_subfamily_calibration.json` and the "
            "module constants in lockstep"
        )


def _demotion_witness(diag: dict) -> dict | None:
    """Extract the truth-fitted demotion witness from a production
    helper's diagnostics: present when the classifier MATCHED the
    family but the demoted factor was NOT applied to the deployable
    value. Returns None for applied or unmatched diags."""
    if diag.get("applied") or "correction_factor_status" not in diag:
        return None
    return {
        "correction_factor_status": diag["correction_factor_status"],
        "factor_demoted": float(diag["factor"]),
        "diagnostic_corrected_value": float(
            diag["diagnostic_corrected_value"],
        ),
    }


def _simulate_l2_post(slug: str, pre_uncorrected_point: float) -> tuple[
    float, bool, str, float | None, dict | None,
]:
    """Simulate the full R47 -> R48 L2 correction chain via the
    production helpers. Returns ``(point_post, matched_any,
    correction_id, factor_applied, demotion)`` where
    ``correction_id`` is one of ``"none"``,
    ``"l2_reduction_kernel"``, or ``"l2_high_footprint_low_reuse"``.
    Under the truth-fitted demotion gate a matched family no-ops
    the deployable value: ``matched_any`` stays True (classifier
    matched), ``factor_applied`` is None (nothing multiplied the
    deployable value), and ``demotion`` carries the status + the
    would-be corrected value."""
    after_reduction, _, _, _, red_diag = cp._apply_l2_reduction_correction(
        point=pre_uncorrected_point,
        ci_low=pre_uncorrected_point,
        ci_high=pre_uncorrected_point,
        slug=slug,
    )
    if red_diag.get("applied"):
        return (
            after_reduction,
            True,
            "l2_reduction_kernel",
            cp._L2_REDUCTION_KERNEL_CORRECTION_FACTORS[slug],
            None,
        )
    red_demotion = _demotion_witness(red_diag)
    if red_demotion is not None:
        return (
            after_reduction,
            True,
            "l2_reduction_kernel",
            None,
            red_demotion,
        )
    after_high_fp, _, _, _, hf_diag = (
        cp._apply_l2_high_footprint_correction(
            point=after_reduction,
            ci_low=after_reduction,
            ci_high=after_reduction,
            slug=slug,
        )
    )
    if hf_diag.get("applied"):
        return (
            after_high_fp,
            True,
            "l2_high_footprint_low_reuse",
            cp._L2_HIGH_FOOTPRINT_LOW_REUSE_CORRECTION_FACTORS[slug],
            None,
        )
    hf_demotion = _demotion_witness(hf_diag)
    if hf_demotion is not None:
        return (
            after_high_fp,
            True,
            "l2_high_footprint_low_reuse",
            None,
            hf_demotion,
        )
    return after_reduction, False, "none", None, None


def _simulate_l1_post(
    slug: str,
    post_r46_point: float,
    walker_baseline_override: float | None = None,
) -> tuple[
    float, bool, str, float | None, dict | None,
]:
    """Simulate the L1 subfamily split against the corrected
    point recorded in the sidecar. R17 update: the walker
    baseline is now sourced from
    `predicted_l1_per_admitted_cell.json`'s evidence block field
    `predicted_l1_hit_rate_uncorrected` when available
    (`walker_baseline_override`), because R17's corrected
    classifier + warmup_launches=5 + carveout-aware capacity
    changed the walker output and the historical R46 back-out
    divisor 0.32 no longer reflects the actual walker baseline.
    For backward compatibility, when `walker_baseline_override`
    is None, the helper falls back to the R46 back-out path.
    For non-reduction slugs the point passes through unchanged.

    Under the truth-fitted demotion gate a matched subfamily
    no-ops the deployable value, so production emits the WALKER
    BASELINE itself: ``point_post`` is the helper's pass-through
    output (== walker baseline), ``factor_applied`` is None, and
    the demotion dict carries the status + would-be value.
    """
    subfam = cp._classify_l1_reduction_subfamily(slug)
    if subfam == "none":
        return post_r46_point, False, "none", None, None
    if walker_baseline_override is not None:
        walker_baseline = float(walker_baseline_override)
    else:
        walker_baseline = (
            post_r46_point / _R46_L1_REDUCTION_FACTOR
            if _R46_L1_REDUCTION_FACTOR != 0
            else post_r46_point
        )
    walker_baseline = min(1.0, max(0.0, walker_baseline))
    after_l1, l1_diag = cp._apply_l1_reduction_correction(
        point=walker_baseline,
        slug=slug,
    )
    if l1_diag.get("applied"):
        return (
            after_l1,
            True,
            subfam,
            float(l1_diag.get("factor", 0.0)),
            None,
        )
    l1_demotion = _demotion_witness(l1_diag)
    if l1_demotion is not None:
        return after_l1, True, subfam, None, l1_demotion
    return post_r46_point, False, "none", None, None


def _l2_rows(diagnostic: dict) -> tuple[
    list[dict], list[float], list[float], list[float],
]:
    cells = _admitted_l2_scalar_cells(diagnostic)
    rows: list[dict] = []
    friendly_after: list[float] = []
    hostile_after: list[float] = []
    predicted_after_all: list[float] = []
    for c in cells:
        slug = c["slug"]
        truth = float(c["ncu_truth_l2_fraction"])
        post_r47_point = float(c["predicted_l2_hit_rate"])
        # The diagnostic sidecar's `predicted_l2_hit_rate` is
        # the post-gridDim, pre-class-correction value for
        # every admitted cell (verifiable: rel_err_l2 in the
        # sidecar equals (predicted - truth)/truth at that
        # field's value, which matches the pre-R47 expectation
        # for the reduction allowlist slugs as well -- e.g.
        # rmsnorm 0.9685 vs truth 0.67948 gives rel_err 0.4254,
        # which equals the sidecar's rel_err_l2 for that cell;
        # if the sidecar value had already been multiplied by
        # 0.88 the rel_err would be 0.254). We therefore feed
        # the sidecar value DIRECTLY to `_simulate_l2_post`
        # which applies the R47 (then R48) corrections via the
        # production helpers exactly once.
        pre_uncorrected = post_r47_point
        post_point, matched_any, correction_id, factor_applied, demotion = (
            _simulate_l2_post(slug, pre_uncorrected)
        )
        rel_err_before = float(c["rel_err_l2"])
        rel_err_after = (
            abs(post_point - truth) / truth if truth != 0.0 else 0.0
        )
        delta = rel_err_after - rel_err_before
        budget_pass = delta <= _PER_CELL_BUDGET
        row = {
            "slug": slug,
            "shape_point": c.get("shape_point", ""),
            "predictor_design_coverage_bucket": (
                c["predictor_design_coverage_bucket"]
            ),
            "matched_any": matched_any,
            "applied_correction_id": correction_id,
            "factor_applied": factor_applied,
            "point_pre": post_r47_point,
            "point_post": post_point,
            "truth": truth,
            "rel_err_before": rel_err_before,
            "rel_err_after": rel_err_after,
            "delta": delta,
            "budget_pass": budget_pass,
        }
        if demotion is not None:
            diag_value = demotion["diagnostic_corrected_value"]
            row.update(demotion)
            row["rel_err_after_diagnostic"] = (
                abs(diag_value - truth) / truth if truth != 0.0 else 0.0
            )
        rows.append(row)
        bucket = c["predictor_design_coverage_bucket"]
        if bucket == "predictor_friendly":
            friendly_after.append(rel_err_after)
        elif bucket == "predictor_hostile":
            hostile_after.append(rel_err_after)
        predicted_after_all.append(post_point)
    rows.sort(key=lambda r: -r["delta"])
    return rows, friendly_after, hostile_after, predicted_after_all


def _walker_uncorrected_by_cell() -> dict[tuple[str, str], float]:
    """Read the `predicted_l1_hit_rate_uncorrected` evidence
    field from the committed `predicted_l1_per_admitted_cell.json`
    artifact. Returns a {(slug, shape_point): uncorrected_value}
    map. The uncorrected value is the walker's raw output
    BEFORE the L1 reduction subfamily correction is applied;
    when present it is the authoritative walker baseline for
    the shadow report's simulation. Missing entries fall back
    to the R46 back-out divisor in `_simulate_l1_post`."""
    path = (
        _REPO_ROOT
        / "data" / "sm_86_predictor_accuracy_recovery_v2"
        / "predicted_l1_per_admitted_cell.json"
    )
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        artifact = json.load(fh)
    out: dict[tuple[str, str], float] = {}
    for cell in artifact.get("cells", []):
        slug = cell.get("slug")
        shape = cell.get("shape_point")
        ev = cell.get("evidence") or {}
        u = ev.get("predicted_l1_hit_rate_uncorrected")
        if slug is None or shape is None or u is None:
            continue
        out[(slug, shape)] = float(u)
    return out


def _l1_rows(diagnostic: dict) -> tuple[
    list[dict], list[float],
]:
    cells = _admitted_l1_scalar_cells(diagnostic)
    walker_uncorrected = _walker_uncorrected_by_cell()
    rows: list[dict] = []
    friendly_after: list[float] = []
    for c in cells:
        slug = c["slug"]
        shape = c.get("shape_point", "")
        truth = float(c["ncu_truth_l1_fraction"])
        post_r46_point = float(c["predicted_l1_hit_rate"])
        walker_baseline_override = walker_uncorrected.get(
            (slug, shape)
        )
        post_point, matched_any, subfam, factor_applied, demotion = (
            _simulate_l1_post(
                slug,
                post_r46_point,
                walker_baseline_override=walker_baseline_override,
            )
        )
        # R17 update: when the walker baseline is overridden
        # (R17 corrected pipeline regenerated predicted_l1
        # with `predicted_l1_hit_rate_uncorrected`), compute
        # `rel_err_before` as the R46 simulation against the
        # SAME walker baseline (i.e., the prediction that R46's
        # single 0.32 factor would have produced). This gives a
        # meaningful before/after comparison for the round's
        # L1 subfamily split + R17 per-slug dispatch — the
        # delta measures the improvement THIS calibration
        # provides over R46 on the same walker baseline. When
        # the override is absent (pre-R17 path), `rel_err_l1`
        # from the sidecar still reflects the R46-corrected
        # comparison because the back-out used the R46 factor.
        if (
            walker_baseline_override is not None
            and subfam != "none"
        ):
            r46_point = (
                walker_baseline_override
                * _R46_L1_REDUCTION_FACTOR
            )
            rel_err_before = (
                abs(r46_point - truth) / truth
                if truth != 0.0 else 0.0
            )
        else:
            rel_err_before = float(c["rel_err_l1"])
        rel_err_after = (
            abs(post_point - truth) / truth if truth != 0.0 else 0.0
        )
        delta = rel_err_after - rel_err_before
        budget_pass = delta <= _PER_CELL_BUDGET
        row = {
            "slug": slug,
            "shape_point": c.get("shape_point", ""),
            "predictor_design_coverage_bucket": (
                c["predictor_design_coverage_bucket"]
            ),
            "matched_any": matched_any,
            "subfamily_matched": subfam,
            "factor_applied": factor_applied,
            "point_pre": post_r46_point,
            "point_post": post_point,
            "truth": truth,
            "rel_err_before": rel_err_before,
            "rel_err_after": rel_err_after,
            "delta": delta,
            "budget_pass": budget_pass,
        }
        if demotion is not None:
            diag_value = demotion["diagnostic_corrected_value"]
            row.update(demotion)
            row["rel_err_after_diagnostic"] = (
                abs(diag_value - truth) / truth if truth != 0.0 else 0.0
            )
        rows.append(row)
        bucket = c["predictor_design_coverage_bucket"]
        if bucket == "predictor_friendly":
            friendly_after.append(rel_err_after)
    rows.sort(key=lambda r: -r["delta"])
    return rows, friendly_after


def build_shadow_report() -> dict:
    diagnostic, l1_calibration, l2_calibration = _read_inputs()
    _assert_l1_calibration_matches_production(l1_calibration)
    _assert_l2_calibration_matches_production(l2_calibration)

    l2_rows, friendly_l2_after, hostile_l2_after, predicted_l2_all = (
        _l2_rows(diagnostic)
    )
    l1_rows, friendly_l1_after = _l1_rows(diagnostic)

    friendly_l2_geomean = _geomean(friendly_l2_after)
    hostile_l2_geomean = _geomean(hostile_l2_after)
    friendly_l1_geomean = _geomean(friendly_l1_after)
    cross_cell_stddev_post = statistics.stdev(predicted_l2_all)

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

    max_delta_l2 = (
        max(r["delta"] for r in l2_rows) if l2_rows else 0.0
    )
    max_delta_l1 = (
        max(r["delta"] for r in l1_rows) if l1_rows else 0.0
    )
    budget_breaches_l2 = [
        r["slug"] for r in l2_rows if not r["budget_pass"]
    ]
    budget_breaches_l1 = [
        r["slug"] for r in l1_rows if not r["budget_pass"]
    ]

    hard_1_pass = friendly_l2_geomean < _HARD_1_FRIENDLY_L2_UPPER_BOUND
    hard_2_pass = (
        friendly_l1_geomean < _HARD_2_FRIENDLY_L1_UPPER_BOUND
        and friendly_l1_geomean <= _R47_FRIENDLY_L1_GEOMEAN + _PER_CELL_BUDGET
    )
    hard_3_pass = (
        _HARD_3_STDDEV_LOWER_BOUND
        <= cross_cell_stddev_post
        <= _HARD_3_STDDEV_UPPER_BOUND
    )

    # Diagnostic-shadow geomeans: the would-be (demoted-factor)
    # channel per bucket; non-demoted rows contribute their
    # deployable rel_err so the geomean stays comparable.
    friendly_l2_geomean_diag = _geomean([
        r.get("rel_err_after_diagnostic", r["rel_err_after"])
        for r in l2_rows
        if r["predictor_design_coverage_bucket"] == "predictor_friendly"
    ])
    friendly_l1_geomean_diag = _geomean([
        r.get("rel_err_after_diagnostic", r["rel_err_after"])
        for r in l1_rows
        if r["predictor_design_coverage_bucket"] == "predictor_friendly"
    ])

    return {
        "schema_version": 1,
        "generator": (
            "tools/sm_86_predictor_accuracy_recovery_v2/"
            "generate_l2_high_footprint_correction_shadow_report.py"
        ),
        # Truth-fitted demotion gate: production applies NO factor
        # from these tables; they are retained as the
        # diagnostic-shadow dispatch record only.
        "correction_factor_status_by_family": {
            family: status
            for family, status in sorted(
                cp.CORRECTION_FACTOR_STATUS_BY_FAMILY.items()
            )
        },
        "l2_high_footprint_factors_demoted": (
            dict(cp._L2_HIGH_FOOTPRINT_LOW_REUSE_CORRECTION_FACTORS)
        ),
        "l1_subfamily_factors_demoted": {
            "silu_elementwise": cp._L1_SILU_ELEMENTWISE_CORRECTION_FACTOR,
            # R18: emit both the control / headline scalar AND
            # the per-slug dispatch dict so future drift in any
            # per-slug value is visible in the shadow report.
            "shared_reduction": (
                cp._L1_SHARED_REDUCTION_CORRECTION_FACTOR
            ),
            "shared_reduction_per_slug": dict(
                cp._L1_SHARED_REDUCTION_CORRECTION_FACTORS
            ),
        },
        "l1_calibration_provenance_path": (
            "data/sm_86_predictor_accuracy_recovery_v2/"
            "l1_reduction_subfamily_calibration.json"
        ),
        "l2_calibration_provenance_path": (
            "data/sm_86_predictor_accuracy_recovery_v2/"
            "l2_high_footprint_correction_calibration.json"
        ),
        "diagnostic_source_path": (
            "data/sm_86_predictor_accuracy_recovery_v2/"
            "interim_diagnostic_report_sidecar.json"
        ),
        "l2_high_footprint_allowlist_slugs": sorted(
            cp._L2_HIGH_FOOTPRINT_LOW_REUSE_SLUG_ALLOWLIST,
        ),
        "l1_silu_elementwise_allowlist_slugs": sorted(
            cp._L1_SILU_ELEMENTWISE_SLUG_ALLOWLIST,
        ),
        "l1_shared_reduction_allowlist_slugs": sorted(
            cp._L1_SHARED_REDUCTION_SLUG_ALLOWLIST,
        ),
        "l2_rows_sorted_by_descending_delta": l2_rows,
        "l1_rows_sorted_by_descending_delta": l1_rows,
        "max_delta_across_l2_cells": max_delta_l2,
        "max_delta_across_l1_cells": max_delta_l1,
        "budget_breaches_l2": budget_breaches_l2,
        "budget_breaches_l1": budget_breaches_l1,
        "per_cell_budget_pass": (
            len(budget_breaches_l2) == 0
            and len(budget_breaches_l1) == 0
        ),
        "per_cell_budget_threshold": _PER_CELL_BUDGET,
        "headline_post_correction": {
            "friendly_l2_geomean_rel_err": friendly_l2_geomean,
            "hostile_l2_geomean_rel_err": hostile_l2_geomean,
            "cross_cell_stddev": cross_cell_stddev_post,
            "friendly_l1_geomean_rel_err": friendly_l1_geomean,
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
            "friendly_l2_geomean_rel_err": friendly_l2_geomean_diag,
            "friendly_l1_geomean_rel_err": friendly_l1_geomean_diag,
            "cross_cell_stddev": statistics.stdev([
                r.get("diagnostic_corrected_value", r["point_post"])
                for r in l2_rows
            ]),
        },
        "hard_targets": {
            "hard_1_friendly_l2_lt_0_15": hard_1_pass,
            "hard_2_friendly_l1_lt_0_20_non_regression": hard_2_pass,
            "hard_3_cross_cell_stddev_in_0_10_to_0_30": hard_3_pass,
        },
        "hard_target_thresholds": {
            "hard_1_upper_bound": _HARD_1_FRIENDLY_L2_UPPER_BOUND,
            "hard_2_upper_bound": _HARD_2_FRIENDLY_L1_UPPER_BOUND,
            "hard_2_r47_baseline": _R47_FRIENDLY_L1_GEOMEAN,
            "hard_3_lower_bound": _HARD_3_STDDEV_LOWER_BOUND,
            "hard_3_upper_bound": _HARD_3_STDDEV_UPPER_BOUND,
        },
    }


def main() -> int:
    report = build_shadow_report()
    # source_commit is stamped on the on-disk artifact by the
    # recapture script after the predictor-touching fix-commit
    # lands per BL-20260508.
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
