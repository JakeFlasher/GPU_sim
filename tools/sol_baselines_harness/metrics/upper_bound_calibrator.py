"""Upper-bound calibration head ensemble (deployed
scope).

The deployed predictor publishes a calibrated
lower bound via the isotonic calibrator. This
module supplies the companion upper-bound head
ensemble: Platt, Beta, and an arithmetic-mean
ensemble that composes Platt + Isotonic + Beta.
A leave-one-cell-out cross-validation (LOOCV)
picker selects which head is published using a
`simplicity_one_se_rule` tiebreaker — the
simplest candidate whose mean error is within
one standard error of the best wins.

Public surface:

  * `UPPER_BOUND_CALIBRATOR_TYPED_REASONS` —
    closed frozenset of typed-reason strings the
    module raises.
  * `UpperBoundCalibratorError(typed_reason,
    message)` — exception carrying the typed
    reason so callers can compose typed-
    unavailable bundles or report-table errors
    without parsing message strings.
  * `CANDIDATE_HEAD_NAMES` — pinned tuple
    `("platt", "isotonic", "beta", "ensemble")`
    declaring the simplicity order (lower index =
    simpler) explicitly.
  * `SIMPLICITY_ORDER_BY_NAME` — index-by-name
    lookup derived from `CANDIDATE_HEAD_NAMES`
    for deterministic tie-breaking.
  * `PlattCalibrator` — wraps two independent
    ridge-regularized 2-parameter sigmoid fits
    (L1, L2) each modeling
    `sigmoid(a * logit(raw) + b)` against the
    NCU label. The `a` parameter is constrained
    strictly positive so the fitted head is
    monotone non-decreasing in raw by
    construction.
  * `BetaCalibrator` — wraps two independent
    ridge-regularized 3-parameter beta-function
    fits (L1, L2) each modeling
    `sigmoid(a * log(raw) + b * log(1 - raw) + c)`
    against the NCU label. The `a` parameter is
    constrained strictly positive and `b` is
    constrained non-positive so the fitted head
    is monotone non-decreasing in raw by
    construction.
  * `EnsembleCalibrator` — composes one of each
    `PlattCalibrator`, `IsotonicCalibrator`
    (from `isotonic_calibrator`), and
    `BetaCalibrator`, and returns the arithmetic
    mean of the three sub-head predictions per
    cache level.
  * `UpperBoundLOOCVResult` — frozen dataclass
    carrying per-candidate `LOOCVResult` entries,
    the selected head name, the published L1 + L2
    mean absolute percentage-point errors, and a
    selection-provenance mapping (per-candidate
    combined L1+L2 means, standard errors, and
    the one-SE threshold used by the picker).
  * `run_upper_bound_loocv(records, *,
    cohort="", random_seed=0)` — runs LOOCV over
    each candidate head and returns the
    `UpperBoundLOOCVResult` after applying the
    `simplicity_one_se_rule` picker.

Imports only stdlib + `numpy` + `scipy.optimize`
+ the deployed `isotonic_calibrator`,
`predictor_design_coverage`, and
`typed_unavailable_bundle` modules. NCU
quarantine is preserved: no calibration-subtree
imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from baselines_harness.metrics.isotonic_calibrator import (
    CalibrationInputRecord,
    IsotonicCalibrator,
    IsotonicCalibratorError,
    LOOCVResidual,
    LOOCVResult,
    _abs_percentage_point_error,
    _select_admitted,
    _validate_fraction,
    _validate_record,
)
from baselines_harness.metrics.typed_unavailable_bundle import (
    evaluate_coverage_tier,
)


UPPER_BOUND_CALIBRATOR_TYPED_REASONS: frozenset[str] = frozenset({
    "calibration_below_smoke_floor",
    "calibration_input_invalid",
    "calibration_typed_unavailable_cell_included",
    "calibrator_not_fitted",
    "calibration_fit_failed",
    "calibration_non_finite_prediction",
    "calibration_selection_invalid",
})


CANDIDATE_HEAD_NAMES: tuple[str, ...] = (
    "platt",
    "isotonic",
    "beta",
    "ensemble",
)


SIMPLICITY_ORDER_BY_NAME: Mapping[str, int] = {
    name: index for index, name in enumerate(CANDIDATE_HEAD_NAMES)
}


_RAW_CLIP_EPS: float = 1e-6
_PRED_CLIP_EPS: float = 1e-12
_PARAM_POSITIVE_EPS: float = 1e-6
_RIDGE_LAMBDA: float = 1e-3


class UpperBoundCalibratorError(Exception):
    """Raised by the upper-bound calibrators on
    input / state contract violations or fit
    failures. Carries the typed-reason string so
    callers can compose typed-unavailable bundles
    or report-table errors without parsing
    message strings.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        self.typed_reason = typed_reason
        self.message = message


def _clip_raw_array(raw: Sequence[float]) -> np.ndarray:
    return np.clip(np.asarray(raw, dtype=np.float64), _RAW_CLIP_EPS, 1.0 - _RAW_CLIP_EPS)


def _logit(values: np.ndarray) -> np.ndarray:
    return np.log(values) - np.log(1.0 - values)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _bce_with_ridge(
    probs: np.ndarray, labels: np.ndarray, theta: np.ndarray,
) -> float:
    safe = np.clip(probs, _PRED_CLIP_EPS, 1.0 - _PRED_CLIP_EPS)
    bce = -np.mean(
        labels * np.log(safe) + (1.0 - labels) * np.log(1.0 - safe),
    )
    ridge = _RIDGE_LAMBDA * float(np.sum(theta * theta))
    return float(bce + ridge)


def _platt_objective(
    theta: np.ndarray, logit_raw: np.ndarray, labels: np.ndarray,
) -> float:
    a, b = theta
    probs = _sigmoid(a * logit_raw + b)
    return _bce_with_ridge(probs, labels, theta)


def _beta_objective(
    theta: np.ndarray,
    log_raw: np.ndarray,
    log_one_minus_raw: np.ndarray,
    labels: np.ndarray,
) -> float:
    a, b, c = theta
    probs = _sigmoid(a * log_raw + b * log_one_minus_raw + c)
    return _bce_with_ridge(probs, labels, theta)


def _fit_platt_head(
    raw: Sequence[float], labels: Sequence[float],
) -> tuple[float, float]:
    raw_clipped = _clip_raw_array(raw)
    labels_arr = np.asarray(labels, dtype=np.float64)
    logit_raw = _logit(raw_clipped)
    seed = np.asarray([1.0, 0.0], dtype=np.float64)
    bounds = [(_PARAM_POSITIVE_EPS, None), (None, None)]
    result = minimize(
        _platt_objective,
        seed,
        args=(logit_raw, labels_arr),
        method="L-BFGS-B",
        bounds=bounds,
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise UpperBoundCalibratorError(
            "calibration_fit_failed",
            f"Platt fit did not converge: "
            f"message={result.message!r} x={result.x!r}",
        )
    return float(result.x[0]), float(result.x[1])


def _fit_beta_head(
    raw: Sequence[float], labels: Sequence[float],
) -> tuple[float, float, float]:
    raw_clipped = _clip_raw_array(raw)
    labels_arr = np.asarray(labels, dtype=np.float64)
    log_raw = np.log(raw_clipped)
    log_one_minus_raw = np.log(1.0 - raw_clipped)
    seed = np.asarray([1.0, -1.0, 0.0], dtype=np.float64)
    bounds = [
        (_PARAM_POSITIVE_EPS, None),
        (None, 0.0),
        (None, None),
    ]
    result = minimize(
        _beta_objective,
        seed,
        args=(log_raw, log_one_minus_raw, labels_arr),
        method="L-BFGS-B",
        bounds=bounds,
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise UpperBoundCalibratorError(
            "calibration_fit_failed",
            f"Beta fit did not converge: "
            f"message={result.message!r} x={result.x!r}",
        )
    return float(result.x[0]), float(result.x[1]), float(result.x[2])


def _check_finite_prediction(value: float, head_name: str) -> float:
    if not math.isfinite(value):
        raise UpperBoundCalibratorError(
            "calibration_non_finite_prediction",
            f"{head_name} produced non-finite "
            f"prediction {value!r}",
        )
    return value


class PlattCalibrator:
    """Wrap two independent ridge-regularized
    2-parameter sigmoid heads (L1, L2).

    Each head models `sigmoid(a * logit(raw) + b)`
    fit against the NCU label by minimizing
    binary cross-entropy plus ridge regularization
    over `(a, b)`. The `a` parameter is bounded
    strictly positive so the fitted head is
    monotone non-decreasing in raw by
    construction. Outputs lie in `(0, 1)` and are
    rejected if non-finite.

    `fit` filters out typed-unavailable cells via
    the canonical partition (matching the
    isotonic path) and enforces the smoke floor
    via `evaluate_coverage_tier`; fewer than
    `COVERAGE_FLOOR_SMOKE` admitted records
    raises `calibration_below_smoke_floor`.
    """

    def __init__(self) -> None:
        self._l1_theta: tuple[float, float] | None = None
        self._l2_theta: tuple[float, float] | None = None
        self._admitted_cell_count: int = 0

    @property
    def is_fitted(self) -> bool:
        return self._l1_theta is not None and self._l2_theta is not None

    @property
    def admitted_cell_count(self) -> int:
        return self._admitted_cell_count

    def _fit_heads_unchecked(
        self,
        admitted: Sequence[CalibrationInputRecord],
    ) -> None:
        """Fit the two Platt heads from a sequence
        that has already passed `_select_admitted`
        AND the public smoke-floor gate. Bypasses
        the coverage-floor check so per-fold LOOCV
        training (whose fold count is admitted - 1)
        does not retrigger the public gate. Callers
        MUST run `evaluate_coverage_tier` themselves
        on the full admitted set before invoking
        this helper.
        """
        raw_l1 = [record.raw_l1_prediction for record in admitted]
        raw_l2 = [record.raw_l2_prediction for record in admitted]
        labels_l1 = [record.ncu_l1_label for record in admitted]
        labels_l2 = [record.ncu_l2_label for record in admitted]
        self._l1_theta = _fit_platt_head(raw_l1, labels_l1)
        self._l2_theta = _fit_platt_head(raw_l2, labels_l2)
        self._admitted_cell_count = len(admitted)

    def fit(
        self,
        records: Iterable[CalibrationInputRecord],
        *,
        cohort: str = "",
    ) -> None:
        records_seq = tuple(records)
        admitted = _select_admitted(records_seq, cohort=cohort)
        tier = evaluate_coverage_tier(len(admitted))
        if not tier.scalar_headline_allowed:
            raise UpperBoundCalibratorError(
                "calibration_below_smoke_floor",
                f"admitted cell count {len(admitted)} below "
                f"coverage smoke floor; refusing to fit a "
                f"Platt calibrator that cannot honestly "
                f"publish a scalar headline",
            )
        self._fit_heads_unchecked(admitted)

    def predict_l1(self, raw_l1: float) -> float:
        if self._l1_theta is None:
            raise UpperBoundCalibratorError(
                "calibrator_not_fitted",
                "predict_l1 called before fit",
            )
        fvalue = _validate_fraction(raw_l1, "raw_l1", "<predict_l1>")
        a, b = self._l1_theta
        clipped = float(np.clip(fvalue, _RAW_CLIP_EPS, 1.0 - _RAW_CLIP_EPS))
        prediction = float(_sigmoid(np.asarray(a * _logit(np.asarray([clipped]))[0] + b))[()])
        return _check_finite_prediction(prediction, "PlattCalibrator.predict_l1")

    def predict_l2(self, raw_l2: float) -> float:
        if self._l2_theta is None:
            raise UpperBoundCalibratorError(
                "calibrator_not_fitted",
                "predict_l2 called before fit",
            )
        fvalue = _validate_fraction(raw_l2, "raw_l2", "<predict_l2>")
        a, b = self._l2_theta
        clipped = float(np.clip(fvalue, _RAW_CLIP_EPS, 1.0 - _RAW_CLIP_EPS))
        prediction = float(_sigmoid(np.asarray(a * _logit(np.asarray([clipped]))[0] + b))[()])
        return _check_finite_prediction(prediction, "PlattCalibrator.predict_l2")


class BetaCalibrator:
    """Wrap two independent ridge-regularized
    3-parameter Kull/Filho/Flach beta-function
    heads (L1, L2).

    Each head models
    `sigmoid(a * log(raw) + b * log(1 - raw) + c)`
    fit against the NCU label by minimizing
    binary cross-entropy plus ridge regularization
    over `(a, b, c)`. The `a` parameter is bounded
    strictly positive and `b` is bounded
    non-positive so the fitted head is monotone
    non-decreasing in raw by construction.
    Outputs lie in `(0, 1)` and are rejected if
    non-finite.

    `fit` follows the same canonical-partition
    admission and smoke-floor gate as
    `PlattCalibrator` and the isotonic path.
    """

    def __init__(self) -> None:
        self._l1_theta: tuple[float, float, float] | None = None
        self._l2_theta: tuple[float, float, float] | None = None
        self._admitted_cell_count: int = 0

    @property
    def is_fitted(self) -> bool:
        return self._l1_theta is not None and self._l2_theta is not None

    @property
    def admitted_cell_count(self) -> int:
        return self._admitted_cell_count

    def _fit_heads_unchecked(
        self,
        admitted: Sequence[CalibrationInputRecord],
    ) -> None:
        raw_l1 = [record.raw_l1_prediction for record in admitted]
        raw_l2 = [record.raw_l2_prediction for record in admitted]
        labels_l1 = [record.ncu_l1_label for record in admitted]
        labels_l2 = [record.ncu_l2_label for record in admitted]
        self._l1_theta = _fit_beta_head(raw_l1, labels_l1)
        self._l2_theta = _fit_beta_head(raw_l2, labels_l2)
        self._admitted_cell_count = len(admitted)

    def fit(
        self,
        records: Iterable[CalibrationInputRecord],
        *,
        cohort: str = "",
    ) -> None:
        records_seq = tuple(records)
        admitted = _select_admitted(records_seq, cohort=cohort)
        tier = evaluate_coverage_tier(len(admitted))
        if not tier.scalar_headline_allowed:
            raise UpperBoundCalibratorError(
                "calibration_below_smoke_floor",
                f"admitted cell count {len(admitted)} below "
                f"coverage smoke floor; refusing to fit a "
                f"Beta calibrator that cannot honestly "
                f"publish a scalar headline",
            )
        self._fit_heads_unchecked(admitted)

    def _predict_one(
        self, raw: float, theta: tuple[float, float, float],
    ) -> float:
        a, b, c = theta
        clipped = float(np.clip(raw, _RAW_CLIP_EPS, 1.0 - _RAW_CLIP_EPS))
        score = a * math.log(clipped) + b * math.log(1.0 - clipped) + c
        return float(1.0 / (1.0 + math.exp(-score)))

    def predict_l1(self, raw_l1: float) -> float:
        if self._l1_theta is None:
            raise UpperBoundCalibratorError(
                "calibrator_not_fitted",
                "predict_l1 called before fit",
            )
        fvalue = _validate_fraction(raw_l1, "raw_l1", "<predict_l1>")
        prediction = self._predict_one(fvalue, self._l1_theta)
        return _check_finite_prediction(prediction, "BetaCalibrator.predict_l1")

    def predict_l2(self, raw_l2: float) -> float:
        if self._l2_theta is None:
            raise UpperBoundCalibratorError(
                "calibrator_not_fitted",
                "predict_l2 called before fit",
            )
        fvalue = _validate_fraction(raw_l2, "raw_l2", "<predict_l2>")
        prediction = self._predict_one(fvalue, self._l2_theta)
        return _check_finite_prediction(prediction, "BetaCalibrator.predict_l2")


class EnsembleCalibrator:
    """Compose one of each `PlattCalibrator`,
    `IsotonicCalibrator`, and `BetaCalibrator`,
    and return the arithmetic mean of the three
    sub-head predictions per cache level.

    The ensemble is deterministic: each sub-head's
    fit is deterministic given a fixed admitted-
    record order, and the arithmetic mean of the
    three predictions is order-invariant. The
    smoke-floor gate runs exactly once on the full
    admitted set; sub-head `_fit_heads_unchecked`
    paths bypass the public gate.
    """

    def __init__(self) -> None:
        self._platt = PlattCalibrator()
        self._isotonic = IsotonicCalibrator()
        self._beta = BetaCalibrator()
        self._admitted_cell_count: int = 0

    @property
    def is_fitted(self) -> bool:
        return (
            self._platt.is_fitted
            and self._isotonic.is_fitted
            and self._beta.is_fitted
        )

    @property
    def admitted_cell_count(self) -> int:
        return self._admitted_cell_count

    def _fit_heads_unchecked(
        self,
        admitted: Sequence[CalibrationInputRecord],
    ) -> None:
        self._platt._fit_heads_unchecked(admitted)
        self._isotonic._fit_heads_unchecked(admitted)
        self._beta._fit_heads_unchecked(admitted)
        self._admitted_cell_count = len(admitted)

    def fit(
        self,
        records: Iterable[CalibrationInputRecord],
        *,
        cohort: str = "",
    ) -> None:
        records_seq = tuple(records)
        admitted = _select_admitted(records_seq, cohort=cohort)
        tier = evaluate_coverage_tier(len(admitted))
        if not tier.scalar_headline_allowed:
            raise UpperBoundCalibratorError(
                "calibration_below_smoke_floor",
                f"admitted cell count {len(admitted)} below "
                f"coverage smoke floor; refusing to fit an "
                f"Ensemble calibrator that cannot honestly "
                f"publish a scalar headline",
            )
        self._fit_heads_unchecked(admitted)

    def predict_l1(self, raw_l1: float) -> float:
        if not self.is_fitted:
            raise UpperBoundCalibratorError(
                "calibrator_not_fitted",
                "predict_l1 called before fit",
            )
        platt = self._platt.predict_l1(raw_l1)
        isotonic = float(self._isotonic.predict_l1(raw_l1))
        beta = self._beta.predict_l1(raw_l1)
        prediction = (platt + isotonic + beta) / 3.0
        return _check_finite_prediction(prediction, "EnsembleCalibrator.predict_l1")

    def predict_l2(self, raw_l2: float) -> float:
        if not self.is_fitted:
            raise UpperBoundCalibratorError(
                "calibrator_not_fitted",
                "predict_l2 called before fit",
            )
        platt = self._platt.predict_l2(raw_l2)
        isotonic = float(self._isotonic.predict_l2(raw_l2))
        beta = self._beta.predict_l2(raw_l2)
        prediction = (platt + isotonic + beta) / 3.0
        return _check_finite_prediction(prediction, "EnsembleCalibrator.predict_l2")


@dataclass(frozen=True)
class UpperBoundLOOCVResult:
    """Aggregated LOOCV result over the upper-bound
    candidate set.

    Carries one `LOOCVResult` per candidate plus
    the head chosen by `simplicity_one_se_rule`
    and its published L1 + L2 mean absolute
    percentage-point errors. The
    `selection_provenance` mapping carries the
    per-candidate combined L1+L2 means + standard
    errors + the one-SE threshold used so a
    downstream consumer can audit the head pick
    without re-running LOOCV.
    """

    per_candidate_results: Mapping[str, LOOCVResult]
    selected_head_name: str
    selected_head_l1_mean_abs_pp_error: float
    selected_head_l2_mean_abs_pp_error: float
    admitted_cell_count: int
    random_seed: int
    selection_provenance: Mapping[str, Any]


def _candidate_factory(name: str) -> Any:
    if name == "platt":
        return PlattCalibrator()
    if name == "isotonic":
        return IsotonicCalibrator()
    if name == "beta":
        return BetaCalibrator()
    if name == "ensemble":
        return EnsembleCalibrator()
    raise UpperBoundCalibratorError(
        "calibration_selection_invalid",
        f"unknown candidate head {name!r}; valid names are "
        f"{sorted(CANDIDATE_HEAD_NAMES)}",
    )


def _run_candidate_loocv(
    name: str,
    admitted: Sequence[CalibrationInputRecord],
    random_seed: int,
) -> LOOCVResult:
    residuals: list[LOOCVResidual] = []
    indexed = list(enumerate(admitted))
    for hold_out_index, held in indexed:
        train = tuple(
            record for index, record in indexed
            if index != hold_out_index
        )
        calibrator = _candidate_factory(name)
        calibrator._fit_heads_unchecked(train)
        predicted_l1 = float(calibrator.predict_l1(held.raw_l1_prediction))
        predicted_l2 = float(calibrator.predict_l2(held.raw_l2_prediction))
        residual = LOOCVResidual(
            cell_name=held.cell_name,
            predicted_l1_calibrated=predicted_l1,
            predicted_l2_calibrated=predicted_l2,
            l1_residual_pp=_abs_percentage_point_error(
                predicted_l1, held.ncu_l1_label,
            ),
            l2_residual_pp=_abs_percentage_point_error(
                predicted_l2, held.ncu_l2_label,
            ),
            ncu_l1_label=held.ncu_l1_label,
            ncu_l2_label=held.ncu_l2_label,
            cell_join_key=held.cell_join_key,
        )
        residuals.append(residual)
    mean_abs_l1_pp = (
        sum(r.l1_residual_pp for r in residuals) / len(residuals)
    )
    mean_abs_l2_pp = (
        sum(r.l2_residual_pp for r in residuals) / len(residuals)
    )
    return LOOCVResult(
        per_cell_residuals=tuple(residuals),
        mean_abs_l1_pp_error=mean_abs_l1_pp,
        mean_abs_l2_pp_error=mean_abs_l2_pp,
        admitted_cell_count=len(admitted),
        random_seed=int(random_seed),
    )


def _combined_per_cell_pp_errors(
    result: LOOCVResult,
) -> np.ndarray:
    """Combine L1 and L2 absolute percentage-point
    errors into a single per-cell score via the
    arithmetic mean of the two errors.

    The combined score preserves the per-cell
    LOOCV granularity so the standard error in
    the picker reflects the actual cell-to-cell
    variance, not the variance of an aggregated
    headline.
    """
    return np.asarray(
        [
            (residual.l1_residual_pp + residual.l2_residual_pp) / 2.0
            for residual in result.per_cell_residuals
        ],
        dtype=np.float64,
    )


def _standard_error_of_mean(values: np.ndarray) -> float:
    n = values.shape[0]
    if n < 2:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(n))


def _simplicity_one_se_pick(
    per_candidate_mean: Mapping[str, float],
    per_candidate_se: Mapping[str, float],
    simplicity_order: Mapping[str, int],
) -> tuple[str, dict[str, Any]]:
    """Deterministic `simplicity_one_se_rule`
    picker.

    Finds the candidate with the smallest mean
    error (the best), computes its standard
    error, and selects the simplest candidate
    whose mean error is at most
    `best_mean + best_se`. Ties broken by
    `simplicity_order` (lower index wins). Raises
    `calibration_selection_invalid` if the
    candidate set is empty.
    """
    if not per_candidate_mean:
        raise UpperBoundCalibratorError(
            "calibration_selection_invalid",
            "per_candidate_mean is empty; cannot run picker",
        )
    best_name = min(
        per_candidate_mean.keys(),
        key=lambda n: (per_candidate_mean[n], simplicity_order[n]),
    )
    best_mean = per_candidate_mean[best_name]
    best_se = per_candidate_se[best_name]
    one_se_threshold = best_mean + best_se
    eligible = [
        name for name in per_candidate_mean.keys()
        if per_candidate_mean[name] <= one_se_threshold
    ]
    if not eligible:
        eligible = [best_name]
    selected = min(eligible, key=lambda n: simplicity_order[n])
    provenance: dict[str, Any] = {
        "per_candidate_combined_mean": dict(per_candidate_mean),
        "per_candidate_combined_standard_error": dict(per_candidate_se),
        "best_candidate_name": best_name,
        "best_candidate_combined_mean": best_mean,
        "best_candidate_combined_standard_error": best_se,
        "one_se_threshold": one_se_threshold,
        "eligible_candidates": sorted(
            eligible, key=lambda n: simplicity_order[n],
        ),
        "simplicity_order_by_name": dict(simplicity_order),
    }
    return selected, provenance


def run_upper_bound_loocv(
    records: Iterable[CalibrationInputRecord],
    *,
    cohort: str = "",
    random_seed: int = 0,
) -> UpperBoundLOOCVResult:
    """Deterministic leave-one-cell-out evaluation
    over the upper-bound candidate set
    (Platt / Isotonic / Beta / Ensemble).

    Filters records to the admitted set
    (typed-unavailable excluded; cohort-matched
    when `cohort` is non-empty); enforces the
    smoke floor once on the full admitted set;
    then for each candidate head runs LOOCV by
    holding out each admitted record in turn,
    fitting the candidate on the remaining
    records, predicting the held-out L1 + L2
    raw values, and recording the absolute
    percentage-point errors against the NCU
    labels. The `simplicity_one_se_rule` picker
    then chooses the published head and returns
    the L1 + L2 mean absolute pp errors from the
    selected head plus the selection provenance.
    """
    records_seq = tuple(records)
    for record in records_seq:
        _validate_record(record)
    admitted = _select_admitted(records_seq, cohort=cohort)
    tier = evaluate_coverage_tier(len(admitted))
    if not tier.scalar_headline_allowed:
        raise UpperBoundCalibratorError(
            "calibration_below_smoke_floor",
            f"admitted cell count {len(admitted)} below "
            f"coverage smoke floor; refusing to evaluate "
            f"upper-bound LOOCV when no fold can publish "
            f"a scalar residual",
        )
    per_candidate_results: dict[str, LOOCVResult] = {}
    per_candidate_combined_mean: dict[str, float] = {}
    per_candidate_combined_se: dict[str, float] = {}
    for name in CANDIDATE_HEAD_NAMES:
        result = _run_candidate_loocv(name, admitted, random_seed)
        per_candidate_results[name] = result
        combined = _combined_per_cell_pp_errors(result)
        per_candidate_combined_mean[name] = float(np.mean(combined))
        per_candidate_combined_se[name] = _standard_error_of_mean(combined)
    selected_name, provenance = _simplicity_one_se_pick(
        per_candidate_combined_mean,
        per_candidate_combined_se,
        SIMPLICITY_ORDER_BY_NAME,
    )
    selected_result = per_candidate_results[selected_name]
    return UpperBoundLOOCVResult(
        per_candidate_results=per_candidate_results,
        selected_head_name=selected_name,
        selected_head_l1_mean_abs_pp_error=selected_result.mean_abs_l1_pp_error,
        selected_head_l2_mean_abs_pp_error=selected_result.mean_abs_l2_pp_error,
        admitted_cell_count=len(admitted),
        random_seed=int(random_seed),
        selection_provenance=provenance,
    )


__all__ = [
    "BetaCalibrator",
    "CANDIDATE_HEAD_NAMES",
    "EnsembleCalibrator",
    "PlattCalibrator",
    "SIMPLICITY_ORDER_BY_NAME",
    "UPPER_BOUND_CALIBRATOR_TYPED_REASONS",
    "UpperBoundCalibratorError",
    "UpperBoundLOOCVResult",
    "run_upper_bound_loocv",
]
