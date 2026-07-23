"""Tiered accuracy-gate evaluator (deployed scope).

Consumes a `LOOCVResult` from the lower-bound isotonic
calibrator and emits a `TieredGateReport` carrying
per-cohort x per-cache-level gate verdicts:

  * Friendly cohort: per-launch absolute pp error
    `<= 5.0` on L1 AND L2 independently.
  * Hostile cohort (excluding predeclared
    `predictor_typed_unavailable` cells): per-launch
    absolute pp error `<= 15.0`.
  * Sibling-style log-floor relative error
    `abs(pred - label) / max(label, epsilon)` reported
    alongside as the secondary metric (not gate-
    blocking), aggregated as the log-floor geomean
    `exp(mean(log(max(rel_err, 1e-12))))` to match the
    cross-study `comparison_runner.py` convention.
  * 2000-resample bootstrap CI at 0.95 confidence on
    both the mean absolute pp error and the secondary
    log-floor geomean per (cohort, cache_level)
    bucket; resampling uses
    `numpy.random.default_rng(random_seed)` for
    determinism.

The evaluator routes every residual's cell name
through the canonical deployed partition lookup so a
caller cannot smuggle a typed-unavailable cell into a
friendly/hostile gate. Cells whose canonical bucket
is `predictor_typed_unavailable` are dropped before
the gate computation (per the contract).

Imports only stdlib + `numpy` + the deployed
`predictor_design_coverage` and
`isotonic_calibrator` modules. NO calibration-subtree
imports; the NCU quarantine boundary is preserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from baselines_harness.metrics.framework_contract import (
    BOOTSTRAP_RESAMPLES,
    CI_CONFIDENCE,
)
from baselines_harness.metrics.isotonic_calibrator import (
    LOOCVResidual,
    LOOCVResult,
)
from baselines_harness.metrics.predictor_design_coverage import (
    PREDICTOR_BUCKET_FRIENDLY,
    PREDICTOR_BUCKET_HOSTILE,
    PREDICTOR_BUCKET_TYPED_UNAVAILABLE,
    PredictorDesignCoverageEntry,
    PredictorDesignCoverageError,
    lookup_predictor_design_coverage,
)


FRIENDLY_PP_THRESHOLD: float = 5.0
HOSTILE_PP_THRESHOLD: float = 15.0
# Back-compat alias for the canonical CI_CONFIDENCE constant
# from the framework-contract module. Downstream callers may
# still pin against the BOOTSTRAP_CONFIDENCE name; new code
# should import CI_CONFIDENCE from `framework_contract`
# directly.
BOOTSTRAP_CONFIDENCE: float = CI_CONFIDENCE

# Plan-defined epsilon floor on the label denominator
# for the secondary relative error
# (abs(pred - label) / max(label, SECONDARY_REL_ERR_EPSILON)).
# A near-zero label such as 0.001 would otherwise blow up
# the rel-err to large multiples of pp_error/100; the
# floor caps the worst-case denominator at 1% so the
# secondary metric stays bounded and meaningful.
SECONDARY_REL_ERR_EPSILON: float = 0.01

# Log floor used by the sibling-style log-space geomean
# aggregator. Cross-study `comparison_runner` uses the
# same convention: clamp the per-sample rel-err at the
# floor BEFORE taking the log, so all-zero samples
# converge to `~1e-12` rather than collapsing to `0.0`
# or blowing up to `-inf`. This is the plan-defined
# cross-arch headline aggregator; the empty-input
# return value is the only carve-out (still `0.0`).
SECONDARY_REL_ERR_GEOMEAN_LOG_FLOOR: float = 1e-12

# Tolerance for cross-validating that the stored
# `l*_residual_pp` matches the pp error freshly
# computed from `predicted_l*_calibrated` and
# `ncu_l*_label`. The canonical `run_loocv()` path
# stores `(|pred - label| * 100)` in IEEE-754 double,
# so any mismatch beyond floating-point round-trip
# noise indicates a corrupted input — the evaluator
# raises `tiered_gate_input_invalid` rather than
# silently honoring the stored value. The 1e-9 floor
# is conservative against round-trip noise on
# fractions in `[0, 1]` multiplied by `100`.
_PP_RESIDUAL_CONSISTENCY_TOL: float = 1e-9


CACHE_LEVEL_L1: str = "l1"
CACHE_LEVEL_L2: str = "l2"


TIERED_GATE_TYPED_REASONS: frozenset[str] = frozenset({
    "tiered_gate_input_invalid",
    "tiered_gate_residuals_empty",
    "tiered_gate_below_smoke_floor",
})


class TieredGateEvaluatorError(Exception):
    """Raised on tiered-gate contract violations.

    Carries the typed-reason string so downstream
    callers (the bundle factory, the report tables) can
    compose typed-unavailable bundles or report-table
    errors without parsing message strings.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        self.typed_reason = typed_reason
        self.message = message


@dataclass(frozen=True)
class CohortGateResult:
    """Per-cohort, per-cache-level gate verdict.

    Primary surface (gate-blocking): per-launch
    absolute pp error. `passed` is `True` iff the
    bucket has at least one residual AND
    `max_abs_pp_error <= threshold_pp`. An empty
    bucket (`sample_count == 0`) records
    `passed = False` — a vacuous pass would let a
    cohort with zero admitted cells claim gate
    success, which is exactly the failure mode the
    coverage-floor evaluator and the smoke-floor gate
    are designed to prevent.

    Secondary surface (informational): the plan-
    defined relative error
    `abs(pred - label) / max(label, epsilon)` with
    epsilon floor `SECONDARY_REL_ERR_EPSILON`,
    aggregated as the sibling-style log-floor geomean
    `exp(mean(log(max(rel_err, 1e-12))))` and reported
    with a percentile-method bootstrap CI plus a
    derived half-width
    `(ci_upper - ci_lower) / 2.0` for easy comparison
    against the sibling cross-study tables.
    """

    cohort: str
    cache_level: str
    sample_count: int
    # Primary: per-launch max abs pp error vs threshold
    mean_abs_pp_error: float
    max_abs_pp_error: float
    threshold_pp: float
    passed: bool
    # Diagnostic: bootstrap CI on the mean abs pp error
    # (clearly named so callers don't confuse it with
    # the secondary rel-err CI).
    mean_abs_pp_error_ci_lower: float
    mean_abs_pp_error_ci_upper: float
    # Secondary: sibling-style log-floor geomean of
    # abs(pred - label) / max(label, epsilon), plus
    # bootstrap CI on that geomean and a derived
    # half-width for at-a-glance comparison against
    # cross-study tables.
    secondary_rel_err_geomean: float
    secondary_rel_err_geomean_ci_lower: float
    secondary_rel_err_geomean_ci_upper: float
    secondary_rel_err_geomean_ci_half_width: float


@dataclass(frozen=True)
class TieredGateReport:
    """Aggregated report for one LOOCV evaluation.

    `overall_passed` is the AND of the four cohort x
    cache-level results.
    """

    friendly_l1: CohortGateResult
    friendly_l2: CohortGateResult
    hostile_l1: CohortGateResult
    hostile_l2: CohortGateResult
    bootstrap_resamples: int
    bootstrap_confidence: float
    random_seed: int
    overall_passed: bool


def _validate_residual(residual: LOOCVResidual) -> None:
    if not isinstance(residual, LOOCVResidual):
        raise TieredGateEvaluatorError(
            "tiered_gate_input_invalid",
            f"residual must be LOOCVResidual; got "
            f"{type(residual).__name__}",
        )
    if not residual.cell_name or not residual.cell_name.strip():
        raise TieredGateEvaluatorError(
            "tiered_gate_input_invalid",
            "residual.cell_name is empty",
        )
    for name, value in (
        ("l1_residual_pp", residual.l1_residual_pp),
        ("l2_residual_pp", residual.l2_residual_pp),
    ):
        if not isinstance(value, (int, float)):
            raise TieredGateEvaluatorError(
                "tiered_gate_input_invalid",
                f"residual.{name} must be a number; got "
                f"{type(value).__name__}",
            )
        fvalue = float(value)
        if math.isnan(fvalue) or math.isinf(fvalue):
            raise TieredGateEvaluatorError(
                "tiered_gate_input_invalid",
                f"residual.{name} is non-finite "
                f"({fvalue!r}) for cell "
                f"{residual.cell_name!r}",
            )
        if fvalue < 0.0:
            raise TieredGateEvaluatorError(
                "tiered_gate_input_invalid",
                f"residual.{name} is negative "
                f"({fvalue!r}) for cell "
                f"{residual.cell_name!r}; absolute "
                f"errors are non-negative",
            )
    for name, value in (
        ("predicted_l1_calibrated", residual.predicted_l1_calibrated),
        ("predicted_l2_calibrated", residual.predicted_l2_calibrated),
        ("ncu_l1_label", residual.ncu_l1_label),
        ("ncu_l2_label", residual.ncu_l2_label),
    ):
        if not isinstance(value, (int, float)):
            raise TieredGateEvaluatorError(
                "tiered_gate_input_invalid",
                f"residual.{name} must be a number; got "
                f"{type(value).__name__}",
            )
        fvalue = float(value)
        if math.isnan(fvalue) or math.isinf(fvalue):
            raise TieredGateEvaluatorError(
                "tiered_gate_input_invalid",
                f"residual.{name} is non-finite "
                f"({fvalue!r}) for cell "
                f"{residual.cell_name!r}",
            )
        if fvalue < 0.0 or fvalue > 1.0:
            raise TieredGateEvaluatorError(
                "tiered_gate_input_invalid",
                f"residual.{name} is out of [0, 1] "
                f"({fvalue!r}) for cell "
                f"{residual.cell_name!r}; hit-rate "
                f"fractions and predictions live in "
                f"[0, 1]",
            )


def _bootstrap_mean_ci(
    samples: np.ndarray,
    *,
    rng: np.random.Generator,
    resamples: int,
    confidence: float,
) -> tuple[float, float]:
    """Percentile-method bootstrap CI on the mean.

    Returns `(lower, upper)` at the `confidence`
    centered confidence level (e.g.,
    `confidence=0.95` → `[2.5%, 97.5%]`). Resamples
    with replacement using `rng`.

    Empty input collapses to `(0.0, 0.0)`; a
    single-sample input returns `(sample, sample)`.
    """
    n = samples.shape[0]
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        v = float(samples[0])
        return (v, v)
    boot_means = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        draw = rng.choice(samples, size=n, replace=True)
        boot_means[i] = float(draw.mean())
    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(boot_means, alpha))
    upper = float(np.quantile(boot_means, 1.0 - alpha))
    return (lower, upper)


def _primary_abs_pp_error(pred: float, label: float) -> float:
    """Compute the per-launch absolute pp error from
    the prediction and the held-out label (both
    fractions in `[0, 1]`).

    This is the plan-defined primary gate quantity:
    `abs(predicted - label) * 100.0`. The evaluator
    computes this from the residual's prediction and
    label fields rather than trusting the stored
    `l*_residual_pp`, so a corrupted input cannot
    silently pass the gate.
    """
    return abs(pred - label) * 100.0


def _secondary_rel_err(pred: float, label: float) -> float:
    """Plan-defined secondary relative error:
    `abs(pred - label) / max(label, SECONDARY_REL_ERR_EPSILON)`.

    The epsilon floor keeps the denominator bounded at
    1% so a near-zero label cannot blow up the rel-err
    to large multiples. For `label >= epsilon`, the
    formula reduces to standard MAPE; below the floor
    it caps at the floor.
    """
    return abs(pred - label) / max(label, SECONDARY_REL_ERR_EPSILON)


def _geomean_log_floor(samples: np.ndarray) -> float:
    """Sibling-style log-space geomean.

    Per-sample value is clamped at
    `SECONDARY_REL_ERR_GEOMEAN_LOG_FLOOR` BEFORE the
    log is taken, then the standard geomean
    `exp(mean(log(...)))` is applied. This is the
    cross-study `comparison_runner` convention and
    the plan's headline aggregator: it makes the
    transform stable at zero (no `log(0) = -inf`
    blowup) while staying a true geometric mean
    above the floor.

    Returns `0.0` ONLY for empty input. A non-empty
    samples array with all-zero entries collapses to
    the floor (~1e-12), NOT to `0.0`.
    """
    if samples.shape[0] == 0:
        return 0.0
    floored = np.maximum(samples, SECONDARY_REL_ERR_GEOMEAN_LOG_FLOOR)
    return float(np.exp(np.mean(np.log(floored))))


def _bootstrap_geomean_log_floor_ci(
    samples: np.ndarray,
    *,
    rng: np.random.Generator,
    resamples: int,
    confidence: float,
) -> tuple[float, float]:
    """Percentile-method bootstrap CI on the
    sibling-style log-floor geomean.

    Each bootstrap iteration applies the same
    log-floor geomean transform `_geomean_log_floor`
    uses. Empty input collapses to `(0.0, 0.0)`;
    single-sample input returns
    `(floored, floored)` where `floored` is the
    sample clamped at the log floor.
    """
    n = samples.shape[0]
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        floored = max(
            float(samples[0]), SECONDARY_REL_ERR_GEOMEAN_LOG_FLOOR,
        )
        return (floored, floored)
    boot_geomeans = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        draw = rng.choice(samples, size=n, replace=True)
        floored = np.maximum(
            draw, SECONDARY_REL_ERR_GEOMEAN_LOG_FLOOR,
        )
        boot_geomeans[i] = float(np.exp(np.mean(np.log(floored))))
    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(boot_geomeans, alpha))
    upper = float(np.quantile(boot_geomeans, 1.0 - alpha))
    return (lower, upper)


def _evaluate_bucket(
    cohort: str,
    cache_level: str,
    pp_samples: np.ndarray,
    rel_err_samples: np.ndarray,
    threshold: float,
    *,
    rng: np.random.Generator,
    resamples: int,
    confidence: float,
) -> CohortGateResult:
    """Build a `CohortGateResult` from per-cell pp
    samples (primary) + per-cell rel-err samples
    (secondary). Both arrays must have the same length.
    """
    sample_count = int(pp_samples.shape[0])
    if sample_count == 0:
        return CohortGateResult(
            cohort=cohort,
            cache_level=cache_level,
            sample_count=0,
            mean_abs_pp_error=0.0,
            max_abs_pp_error=0.0,
            threshold_pp=threshold,
            passed=False,
            mean_abs_pp_error_ci_lower=0.0,
            mean_abs_pp_error_ci_upper=0.0,
            secondary_rel_err_geomean=0.0,
            secondary_rel_err_geomean_ci_lower=0.0,
            secondary_rel_err_geomean_ci_upper=0.0,
            secondary_rel_err_geomean_ci_half_width=0.0,
        )
    mean_abs = float(np.mean(pp_samples))
    max_abs = float(np.max(pp_samples))
    pp_ci_lower, pp_ci_upper = _bootstrap_mean_ci(
        pp_samples, rng=rng, resamples=resamples,
        confidence=confidence,
    )
    rel_err_geomean = _geomean_log_floor(rel_err_samples)
    rel_err_ci_lower, rel_err_ci_upper = _bootstrap_geomean_log_floor_ci(
        rel_err_samples, rng=rng, resamples=resamples,
        confidence=confidence,
    )
    rel_err_ci_half_width = (rel_err_ci_upper - rel_err_ci_lower) / 2.0
    return CohortGateResult(
        cohort=cohort,
        cache_level=cache_level,
        sample_count=sample_count,
        mean_abs_pp_error=mean_abs,
        max_abs_pp_error=max_abs,
        threshold_pp=threshold,
        passed=max_abs <= threshold,
        mean_abs_pp_error_ci_lower=pp_ci_lower,
        mean_abs_pp_error_ci_upper=pp_ci_upper,
        secondary_rel_err_geomean=rel_err_geomean,
        secondary_rel_err_geomean_ci_lower=rel_err_ci_lower,
        secondary_rel_err_geomean_ci_upper=rel_err_ci_upper,
        secondary_rel_err_geomean_ci_half_width=rel_err_ci_half_width,
    )


def evaluate_tiered_gate(
    loocv_result: LOOCVResult,
    *,
    partition_lookup: Callable[[str], PredictorDesignCoverageEntry] | None = None,
    random_seed: int = 0,
) -> TieredGateReport:
    """Evaluate the tiered accuracy gate for a LOOCV
    result.

    Iterates over `loocv_result.per_cell_residuals`;
    resolves each cell name through the deployed
    partition (the optional `partition_lookup`
    injection seam takes precedence — tests use this
    to bind synthetic cell names to canonical
    buckets); drops cells whose canonical bucket is
    `predictor_typed_unavailable`; partitions the
    remaining residuals into friendly + hostile;
    computes per (cohort, cache_level) bucket
    statistics (mean abs pp error, max abs pp error,
    bootstrap CI on the mean, log-space rel-err);
    and returns the aggregated `TieredGateReport`.

    Typed errors raised:
      `tiered_gate_input_invalid`: non-finite residual,
        negative residual, wrong type, empty cell_name,
        or unmapped cell name (when the canonical
        partition lookup fails).
    """
    if not isinstance(loocv_result, LOOCVResult):
        raise TieredGateEvaluatorError(
            "tiered_gate_input_invalid",
            f"loocv_result must be LOOCVResult; got "
            f"{type(loocv_result).__name__}",
        )
    lookup = (
        partition_lookup
        if partition_lookup is not None
        else lookup_predictor_design_coverage
    )
    # Collect both pp residuals (primary) AND secondary
    # rel-err samples per (cohort, cache_level) bucket.
    # The plan's secondary metric needs the label
    # denominator with an epsilon floor — see
    # `_secondary_rel_err`.
    friendly_l1_pp: list[float] = []
    friendly_l2_pp: list[float] = []
    hostile_l1_pp: list[float] = []
    hostile_l2_pp: list[float] = []
    friendly_l1_rel: list[float] = []
    friendly_l2_rel: list[float] = []
    hostile_l1_rel: list[float] = []
    hostile_l2_rel: list[float] = []
    for residual in loocv_result.per_cell_residuals:
        _validate_residual(residual)
        try:
            entry = lookup(residual.cell_name)
        except PredictorDesignCoverageError as exc:
            raise TieredGateEvaluatorError(
                "tiered_gate_input_invalid",
                f"residual for cell {residual.cell_name!r} "
                f"is not in the deployed predictor "
                f"design-coverage partition (reason: "
                f"{exc.typed_reason})",
            ) from exc
        if entry.predictor_bucket == PREDICTOR_BUCKET_TYPED_UNAVAILABLE:
            # Hostile cohort excludes
            # predeclared typed_unavailable subclasses.
            # The friendly cohort never contains
            # typed-unavailable cells by partition
            # construction, but the exclusion is symmetric
            # here so a future partition change cannot
            # smuggle typed-unavailable cells into either
            # bucket.
            continue
        # Compute the primary pp gate quantity from
        # `predicted_l*_calibrated` and `ncu_l*_label`
        # rather than trusting the stored
        # `l*_residual_pp`. The stored value is
        # cross-validated for consistency at
        # `_PP_RESIDUAL_CONSISTENCY_TOL` (1e-9) so a
        # corrupted input fails loudly; the gate
        # computation itself uses the freshly computed
        # value. Without this, a residual with
        # `predicted=0.9, label=0.0, l1_residual_pp=0.0`
        # would silently pass a 5pp gate despite a real
        # 90pp error.
        computed_pp_l1 = _primary_abs_pp_error(
            residual.predicted_l1_calibrated, residual.ncu_l1_label,
        )
        computed_pp_l2 = _primary_abs_pp_error(
            residual.predicted_l2_calibrated, residual.ncu_l2_label,
        )
        for name, stored, computed in (
            ("l1_residual_pp", residual.l1_residual_pp, computed_pp_l1),
            ("l2_residual_pp", residual.l2_residual_pp, computed_pp_l2),
        ):
            if abs(float(stored) - computed) > _PP_RESIDUAL_CONSISTENCY_TOL:
                raise TieredGateEvaluatorError(
                    "tiered_gate_input_invalid",
                    f"residual.{name} for cell "
                    f"{residual.cell_name!r} is "
                    f"{float(stored)!r} but the computed "
                    f"pp error from pred + label is "
                    f"{computed!r}; mismatch "
                    f"{abs(float(stored) - computed)!r} "
                    f"exceeds tolerance "
                    f"{_PP_RESIDUAL_CONSISTENCY_TOL!r}",
                )
        rel_l1 = _secondary_rel_err(
            residual.predicted_l1_calibrated, residual.ncu_l1_label,
        )
        rel_l2 = _secondary_rel_err(
            residual.predicted_l2_calibrated, residual.ncu_l2_label,
        )
        if entry.predictor_bucket == PREDICTOR_BUCKET_FRIENDLY:
            friendly_l1_pp.append(computed_pp_l1)
            friendly_l2_pp.append(computed_pp_l2)
            friendly_l1_rel.append(rel_l1)
            friendly_l2_rel.append(rel_l2)
        elif entry.predictor_bucket == PREDICTOR_BUCKET_HOSTILE:
            hostile_l1_pp.append(computed_pp_l1)
            hostile_l2_pp.append(computed_pp_l2)
            hostile_l1_rel.append(rel_l1)
            hostile_l2_rel.append(rel_l2)
        else:
            raise TieredGateEvaluatorError(
                "tiered_gate_input_invalid",
                f"residual for cell {residual.cell_name!r} "
                f"has canonical bucket "
                f"{entry.predictor_bucket!r} which is not "
                f"a recognized friendly/hostile/typed_unavailable "
                f"value",
            )
    rng = np.random.default_rng(int(random_seed))
    friendly_l1_result = _evaluate_bucket(
        PREDICTOR_BUCKET_FRIENDLY, CACHE_LEVEL_L1,
        np.asarray(friendly_l1_pp, dtype=np.float64),
        np.asarray(friendly_l1_rel, dtype=np.float64),
        FRIENDLY_PP_THRESHOLD,
        rng=rng, resamples=BOOTSTRAP_RESAMPLES,
        confidence=BOOTSTRAP_CONFIDENCE,
    )
    friendly_l2_result = _evaluate_bucket(
        PREDICTOR_BUCKET_FRIENDLY, CACHE_LEVEL_L2,
        np.asarray(friendly_l2_pp, dtype=np.float64),
        np.asarray(friendly_l2_rel, dtype=np.float64),
        FRIENDLY_PP_THRESHOLD,
        rng=rng, resamples=BOOTSTRAP_RESAMPLES,
        confidence=BOOTSTRAP_CONFIDENCE,
    )
    hostile_l1_result = _evaluate_bucket(
        PREDICTOR_BUCKET_HOSTILE, CACHE_LEVEL_L1,
        np.asarray(hostile_l1_pp, dtype=np.float64),
        np.asarray(hostile_l1_rel, dtype=np.float64),
        HOSTILE_PP_THRESHOLD,
        rng=rng, resamples=BOOTSTRAP_RESAMPLES,
        confidence=BOOTSTRAP_CONFIDENCE,
    )
    hostile_l2_result = _evaluate_bucket(
        PREDICTOR_BUCKET_HOSTILE, CACHE_LEVEL_L2,
        np.asarray(hostile_l2_pp, dtype=np.float64),
        np.asarray(hostile_l2_rel, dtype=np.float64),
        HOSTILE_PP_THRESHOLD,
        rng=rng, resamples=BOOTSTRAP_RESAMPLES,
        confidence=BOOTSTRAP_CONFIDENCE,
    )
    overall_passed = (
        friendly_l1_result.passed
        and friendly_l2_result.passed
        and hostile_l1_result.passed
        and hostile_l2_result.passed
    )
    return TieredGateReport(
        friendly_l1=friendly_l1_result,
        friendly_l2=friendly_l2_result,
        hostile_l1=hostile_l1_result,
        hostile_l2=hostile_l2_result,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        bootstrap_confidence=BOOTSTRAP_CONFIDENCE,
        random_seed=int(random_seed),
        overall_passed=overall_passed,
    )


__all__ = [
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_RESAMPLES",
    "CACHE_LEVEL_L1",
    "CACHE_LEVEL_L2",
    "CohortGateResult",
    "FRIENDLY_PP_THRESHOLD",
    "HOSTILE_PP_THRESHOLD",
    "SECONDARY_REL_ERR_EPSILON",
    "SECONDARY_REL_ERR_GEOMEAN_LOG_FLOOR",
    "TIERED_GATE_TYPED_REASONS",
    "TieredGateEvaluatorError",
    "TieredGateReport",
    "evaluate_tiered_gate",
]
