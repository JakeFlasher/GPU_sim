"""Lower-bound isotonic calibrator + leave-one-cell-out
(LOOCV) picker (deployed scope).

The deployed predictor is a calibrated lower bound, not
an arbitrary regression head. Calibration is performed
with `sklearn.isotonic.IsotonicRegression`: it maps raw
simulator hit-rate predictions to monotone, bounded
calibrated predictions whose error is evaluated per cell
via LOOCV.

Public surface:

  * `CALIBRATOR_CLASS_NAME` — pinned string
    `IsotonicRegression`. Downstream contract tests assert
    this value AND grep the module source for the literal
    string so a future regression to a wrapper class fails.
  * `CalibrationInputRecord` — frozen dataclass carrying
    cell_name, arch, launch_uid, predictor_bucket, raw L1
    + L2 predictions, NCU L1 + L2 labels (fractions in
    [0,1]), and a provenance mapping.
  * `IsotonicCalibrator` — wraps two independent
    `IsotonicRegression(out_of_bounds="clip")` heads (L1,
    L2) fit on admitted records only. Typed-unavailable
    records are excluded from fitting; the predict_*
    methods return calibrated lower-bound predictions for
    raw simulator outputs.
  * `LOOCVResidual` — per-cell residual record carrying
    the cell name, calibrated L1 + L2 predictions, and
    absolute percentage-point errors against the NCU
    labels.
  * `LOOCVResult` — aggregated LOOCV result with per-cell
    residuals, mean absolute pp errors for L1 and L2,
    admitted-cell count, and the random seed for
    provenance.
  * `IsotonicCalibratorError(typed_reason, message)` —
    exception with typed reasons:
    `calibration_below_smoke_floor`,
    `calibration_input_invalid`,
    `calibration_typed_unavailable_cell_included`,
    `calibrator_not_fitted`.
  * `run_loocv(records, *, cohort="", random_seed=0)` —
    deterministic LOOCV evaluator that returns a
    `LOOCVResult` aggregating per-cell residuals.

Imports only stdlib + `sklearn.isotonic` + the deployed
`predictor_design_coverage` and `typed_unavailable_bundle`
modules. NCU quarantine is preserved: no calibration-
subtree imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from sklearn.isotonic import IsotonicRegression

from baselines_harness.metrics.framework_contract import (
    CellJoinKey,
)
from baselines_harness.metrics.predictor_design_coverage import (
    PREDICTOR_BUCKET_TYPED_UNAVAILABLE,
    PREDICTOR_BUCKETS,
    PredictorDesignCoverageError,
    lookup_predictor_design_coverage,
)
from baselines_harness.metrics.typed_unavailable_bundle import (
    evaluate_coverage_tier,
)


CALIBRATOR_CLASS_NAME: str = "IsotonicRegression"


CALIBRATOR_TYPED_REASONS: frozenset[str] = frozenset({
    "calibration_below_smoke_floor",
    "calibration_input_invalid",
    "calibration_typed_unavailable_cell_included",
    "calibration_input_contains_validation_only_slug",
    "calibration_input_missing_split_assignment",
    "calibrator_not_fitted",
})


class IsotonicCalibratorError(Exception):
    """Raised by the calibrator on input / state contract
    violations. Carries the typed-reason string so callers
    (the predictor body, the scoreboard, downstream report
    layers) can compose typed-unavailable bundles or
    report-table errors without parsing message strings.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        self.typed_reason = typed_reason
        self.message = message


@dataclass(frozen=True)
class CalibrationInputRecord:
    cell_name: str
    arch: str
    launch_uid: int
    predictor_bucket: str
    raw_l1_prediction: float
    raw_l2_prediction: float
    ncu_l1_label: float
    ncu_l2_label: float
    cell_join_key: CellJoinKey
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LOOCVResidual:
    """Per-cell residual record from leave-one-out
    cross-validation.

    Carries both the calibrated prediction AND the
    held-out NCU label per cache level so downstream
    consumers (e.g., the tiered-gate evaluator's
    secondary relative-error report) can compute
    metrics that need the label denominator without
    going back to the calibrator. The `cell_join_key`
    field carries the held-out record's 7-tuple so a
    downstream consumer can validate join consistency
    against manifest / ledger / prediction records via
    `framework_contract.validate_join_key_consistency_across_sources`.
    """

    cell_name: str
    predicted_l1_calibrated: float
    predicted_l2_calibrated: float
    l1_residual_pp: float
    l2_residual_pp: float
    ncu_l1_label: float
    ncu_l2_label: float
    cell_join_key: CellJoinKey


@dataclass(frozen=True)
class LOOCVResult:
    per_cell_residuals: tuple[LOOCVResidual, ...]
    mean_abs_l1_pp_error: float
    mean_abs_l2_pp_error: float
    admitted_cell_count: int
    random_seed: int


def _validate_fraction(value: float, name: str, cell_name: str) -> float:
    # Python's `isinstance(True, int)` is True, so a bare
    # `(int, float)` check would silently admit `True` /
    # `False` as 1.0 / 0.0. Reject bool explicitly before
    # the numeric check, mirroring the same guard in
    # `predictor_bundle_schema._as_float` and
    # `_as_nonnegative_int`. Without this guard a fitted
    # calibrator's `predict_l1(True)` returns the model's
    # response at raw=1.0 instead of failing closed.
    if isinstance(value, bool):
        raise IsotonicCalibratorError(
            "calibration_input_invalid",
            f"cell {cell_name!r}: {name} must be a number; "
            f"got bool {value!r}",
        )
    if not isinstance(value, (int, float)):
        raise IsotonicCalibratorError(
            "calibration_input_invalid",
            f"cell {cell_name!r}: {name} must be a number; "
            f"got {type(value).__name__}",
        )
    fvalue = float(value)
    if math.isnan(fvalue) or math.isinf(fvalue):
        raise IsotonicCalibratorError(
            "calibration_input_invalid",
            f"cell {cell_name!r}: {name} is non-finite "
            f"({fvalue!r})",
        )
    if fvalue < 0.0 or fvalue > 1.0:
        raise IsotonicCalibratorError(
            "calibration_input_invalid",
            f"cell {cell_name!r}: {name} is out of "
            f"[0, 1] ({fvalue!r})",
        )
    return fvalue


def _validate_record(record: CalibrationInputRecord) -> str:
    """Validate `record` and return the canonical
    predictor_bucket from the deployed partition.

    Cells outside the partition propagate
    `PredictorDesignCoverageError` (typed reason
    `cell_missing_cohort_partition`) unchanged from
    `lookup_predictor_design_coverage`. This matches the
    fail-closed pattern used by
    `cache_predictor.predict_from_launch_frame` and the
    typed-unavailable bundle factory.

    Bucket drift between the record's `predictor_bucket`
    and the canonical partition raises
    `IsotonicCalibratorError("calibration_input_invalid",
    ...)`. The message names the canonical bucket so a
    debugger can read the contract violation at a glance.
    """
    if not isinstance(record, CalibrationInputRecord):
        raise IsotonicCalibratorError(
            "calibration_input_invalid",
            f"record must be CalibrationInputRecord; got "
            f"{type(record).__name__}",
        )
    if not record.cell_name or not record.cell_name.strip():
        raise IsotonicCalibratorError(
            "calibration_input_invalid",
            "cell_name is empty",
        )
    if record.predictor_bucket not in PREDICTOR_BUCKETS:
        raise IsotonicCalibratorError(
            "calibration_input_invalid",
            f"cell {record.cell_name!r}: predictor_bucket "
            f"{record.predictor_bucket!r} is not in "
            f"{sorted(PREDICTOR_BUCKETS)}",
        )
    _validate_fraction(
        record.raw_l1_prediction, "raw_l1_prediction", record.cell_name,
    )
    _validate_fraction(
        record.raw_l2_prediction, "raw_l2_prediction", record.cell_name,
    )
    _validate_fraction(
        record.ncu_l1_label, "ncu_l1_label", record.cell_name,
    )
    _validate_fraction(
        record.ncu_l2_label, "ncu_l2_label", record.cell_name,
    )
    canonical = lookup_predictor_design_coverage(record.cell_name)
    if record.predictor_bucket != canonical.predictor_bucket:
        raise IsotonicCalibratorError(
            "calibration_input_invalid",
            f"cell {record.cell_name!r}: record "
            f"predictor_bucket {record.predictor_bucket!r} "
            f"drifts from canonical partition bucket "
            f"{canonical.predictor_bucket!r}; the deployed "
            f"partition is the single source of truth and "
            f"this calibrator refuses to admit records "
            f"whose record-side bucket disagrees",
        )
    return canonical.predictor_bucket


def _select_admitted(
    records: Sequence[CalibrationInputRecord],
    *,
    cohort: str,
) -> tuple[CalibrationInputRecord, ...]:
    """Filter to admitted records via the canonical
    deployed partition.

    Each record's `cell_name` is resolved against
    `lookup_predictor_design_coverage`; the partition's
    bucket — not the record's — drives typed-unavailable
    exclusion and cohort filtering. This means a record
    that carries the wrong bucket cannot smuggle a
    canonical typed-unavailable cell into the admitted
    set.
    """
    admitted: list[CalibrationInputRecord] = []
    for record in records:
        canonical_bucket = _validate_record(record)
        if canonical_bucket == PREDICTOR_BUCKET_TYPED_UNAVAILABLE:
            continue
        if cohort and canonical_bucket != cohort:
            continue
        admitted.append(record)
    return tuple(admitted)


def _fit_head(
    raw: Sequence[float], labels: Sequence[float],
) -> IsotonicRegression:
    head = IsotonicRegression(out_of_bounds="clip")
    head.fit(list(raw), list(labels))
    return head


class IsotonicCalibrator:
    """Wrap two independent
    `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")`
    heads (L1, L2).

    `fit` filters out typed-unavailable cells (and, if a
    cohort filter is supplied, restricts to records whose
    predictor_bucket matches). It enforces the smoke floor
    via `evaluate_coverage_tier`: fewer than
    `COVERAGE_FLOOR_SMOKE` admitted records raises
    `calibration_below_smoke_floor`, which the caller
    composes into a typed-unavailable bundle (the typed
    reason maps to the canonical `coverage_below_smoke_floor`
    in `PREDICTOR_ABSTENTION_REASONS`).
    """

    def __init__(self) -> None:
        self._l1_head: IsotonicRegression | None = None
        self._l2_head: IsotonicRegression | None = None
        self._admitted_cell_count: int = 0

    @property
    def is_fitted(self) -> bool:
        return self._l1_head is not None and self._l2_head is not None

    @property
    def admitted_cell_count(self) -> int:
        return self._admitted_cell_count

    def _fit_heads_unchecked(
        self,
        admitted: Sequence[CalibrationInputRecord],
    ) -> None:
        """Fit the two isotonic heads from a sequence
        that has already passed `_select_admitted` AND
        the public smoke-floor gate. Bypasses the
        coverage-floor check so per-fold LOOCV training
        (whose fold count is admitted - 1) does not
        retrigger the public gate. Callers MUST run
        `evaluate_coverage_tier` themselves on the full
        admitted set before invoking this helper.
        """
        raw_l1 = [record.raw_l1_prediction for record in admitted]
        raw_l2 = [record.raw_l2_prediction for record in admitted]
        labels_l1 = [record.ncu_l1_label for record in admitted]
        labels_l2 = [record.ncu_l2_label for record in admitted]
        self._l1_head = _fit_head(raw_l1, labels_l1)
        self._l2_head = _fit_head(raw_l2, labels_l2)
        self._admitted_cell_count = len(admitted)

    def fit(
        self,
        records: Iterable[CalibrationInputRecord],
        *,
        cohort: str = "",
        validation_split: Mapping[str, str] | None = None,
    ) -> None:
        """Fit the L1 / L2 isotonic heads.

        `validation_split` is the by-slug split map from
        `CalibrationValidationSplit.by_slug`; when provided,
        the fit guard enforces the calibration / validation
        split contract:

          * Records whose `cell_name` is mapped to
            `validation_only` OR `quarantine` are rejected
            with `calibration_input_contains_validation_only_slug`.
          * Records whose `cell_name` is ABSENT from the
            split map are rejected with
            `calibration_input_missing_split_assignment` —
            new corpus additions default to validation-only
            until explicitly promoted, so a missing
            assignment in the canonical split map means the
            slug has not been promoted to calibration-only.
            The operator must either add the slug to the
            split map (with explicit promotion-log evidence)
            or filter the record out before calling fit.

        Backward-compatible default: when `validation_split
        is None`, the existing fit behavior is unchanged.
        """
        records_seq = tuple(records)
        if validation_split is not None:
            forbidden: list[str] = []
            missing: list[str] = []
            for record in records_seq:
                assigned_split = validation_split.get(record.cell_name)
                if assigned_split is None:
                    missing.append(record.cell_name)
                    continue
                if assigned_split in (
                    "validation_only",
                    "quarantine",
                ):
                    forbidden.append(record.cell_name)
            if missing:
                raise IsotonicCalibratorError(
                    "calibration_input_missing_split_assignment",
                    f"calibrator fit input contains "
                    f"{len(missing)} record(s) whose slug is "
                    f"absent from the split map: "
                    f"{sorted(set(missing))!r}. New corpus "
                    f"additions default to validation_only "
                    f"until explicitly promoted; add the slug "
                    f"to the split manifest (with a per-slug "
                    f"[[promotion_log]] entry) or filter the "
                    f"record out of the fit input.",
                )
            if forbidden:
                raise IsotonicCalibratorError(
                    "calibration_input_contains_validation_only_slug",
                    f"calibrator fit input contains "
                    f"{len(forbidden)} record(s) whose slug "
                    f"is marked validation_only or quarantine "
                    f"in the split manifest: "
                    f"{sorted(set(forbidden))!r}. Refit with a "
                    f"filtered input that excludes these slugs, "
                    f"or promote them to calibration_only via "
                    f"a [[promotion_log]] entry.",
                )
        admitted = _select_admitted(records_seq, cohort=cohort)
        tier = evaluate_coverage_tier(len(admitted))
        if not tier.scalar_headline_allowed:
            raise IsotonicCalibratorError(
                "calibration_below_smoke_floor",
                f"admitted cell count {len(admitted)} below "
                f"coverage smoke floor; refusing to fit a "
                f"calibrator that cannot honestly publish a "
                f"scalar headline",
            )
        self._fit_heads_unchecked(admitted)

    def predict_l1(self, raw_l1: float) -> float:
        if self._l1_head is None:
            raise IsotonicCalibratorError(
                "calibrator_not_fitted",
                "predict_l1 called before fit",
            )
        fvalue = _validate_fraction(raw_l1, "raw_l1", "<predict_l1>")
        return float(self._l1_head.predict([fvalue])[0])

    def predict_l2(self, raw_l2: float) -> float:
        if self._l2_head is None:
            raise IsotonicCalibratorError(
                "calibrator_not_fitted",
                "predict_l2 called before fit",
            )
        fvalue = _validate_fraction(raw_l2, "raw_l2", "<predict_l2>")
        return float(self._l2_head.predict([fvalue])[0])


def _abs_percentage_point_error(
    prediction: float, label: float,
) -> float:
    return abs(prediction - label) * 100.0


def run_loocv(
    records: Iterable[CalibrationInputRecord],
    *,
    cohort: str = "",
    random_seed: int = 0,
) -> LOOCVResult:
    """Deterministic leave-one-cell-out evaluation.

    Filters records to the admitted set (typed-unavailable
    excluded; cohort-matched when `cohort` is non-empty);
    then for each admitted record holds it out, fits the
    calibrator on the remaining records, predicts the
    calibrated L1 + L2 for the held-out record, and records
    the absolute percentage-point error against the NCU
    label. Returns a `LOOCVResult` aggregating the per-cell
    residuals plus mean absolute errors.

    `random_seed` is recorded in provenance. The isotonic
    fit itself is deterministic given a fixed input order,
    so the seed is reserved for future stochastic-bootstrap
    extensions; passing the same seed twice with the same
    records guarantees identical residuals today.

    Raises `IsotonicCalibratorError("calibration_below_smoke_floor",
    ...)` when the FULL admitted-cell count falls below
    the smoke floor. The public coverage-floor gate is
    checked exactly once, on the full admitted set; each
    fold's training set has admitted-count - 1 records
    and bypasses the public gate via
    `_fit_heads_unchecked`, so exactly 5 admitted records
    produce 5 residuals (the per-fold count of 4 does NOT
    re-trigger the smoke-floor failure).
    """
    records_seq = tuple(records)
    admitted = _select_admitted(records_seq, cohort=cohort)
    tier = evaluate_coverage_tier(len(admitted))
    if not tier.scalar_headline_allowed:
        raise IsotonicCalibratorError(
            "calibration_below_smoke_floor",
            f"admitted cell count {len(admitted)} below "
            f"coverage smoke floor; refusing to evaluate "
            f"LOOCV when no fold can publish a scalar "
            f"residual",
        )
    residuals: list[LOOCVResidual] = []
    indexed = list(enumerate(admitted))
    for hold_out_index, held in indexed:
        train = tuple(
            record for index, record in indexed
            if index != hold_out_index
        )
        calibrator = IsotonicCalibrator()
        calibrator._fit_heads_unchecked(train)
        predicted_l1 = calibrator.predict_l1(held.raw_l1_prediction)
        predicted_l2 = calibrator.predict_l2(held.raw_l2_prediction)
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


__all__ = [
    "CALIBRATOR_CLASS_NAME",
    "CALIBRATOR_TYPED_REASONS",
    "CalibrationInputRecord",
    "IsotonicCalibrator",
    "IsotonicCalibratorError",
    "LOOCVResidual",
    "LOOCVResult",
    "run_loocv",
]
