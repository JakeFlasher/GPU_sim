"""Typed metric comparison runner for the cache/memory metric registry.

Joins a typed NCU ground-truth label (``typed_ncu_label_path.TypedMetricLabel``)
to a predictor's value on the FULL cell identity (``arch``, ``slug``,
``shape_point``) plus the metric key, and compares them with the per-kind error
geometry the registry declares:

  - ``fraction`` -> epsilon-floored RELATIVE error
    ``abs(pred - gt) / max(abs(gt), epsilon)`` (``epsilon = 0.01``, matching
    ``comparison_contract.toml``'s ``[error_formula]``) compared against a
    relative tolerance -- the floor keeps a near-zero ground-truth fraction from
    blowing the error up;
  - ``count`` -> absolute / count-floor relative error (NEVER an unbounded
    relative error: a count-floor caps the denominator);
  - ``ratio`` -> denominator-aware relative error with an epsilon floor.

Every non-comparable case is recorded with a TYPED REASON rather than crashing:
an unavailable target, a metric-version drift, a zero-denominator ratio, an
invalid (negative / non-integer) predicted count, a value out of bounds, or an
identity mismatch.

For an L1/L2 HEADLINE comparison (the scope spans both L1 and L2) the runner
refuses a full-stream L2 prediction BEFORE emitting any result, reusing the
L1->miss-stream->L2 filter's headline guard: NCU's L2 observes only the L1-miss
stream, so a full-stream L2 prediction would be compared against the wrong thing.

This is metric comparison / cache accounting, not scheduler or timing modeling.
It invokes no NCU and writes no ``.ncu-rep`` -- it consumes already-typed label
values and predictor values (synthetic fixtures in tests; committed label/
prediction values in production).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CACHE_REPLAY = _HERE.parent / "cache_replay"
for _p in (_HERE, _CACHE_REPLAY):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ncu_metric_registry as reg  # noqa: E402
import typed_ncu_label_path as tlp  # noqa: E402
from l1_to_l2_miss_filter import (  # noqa: E402
    L2_INPUT_FULL_STREAM,
    enforce_l1_miss_filtered_for_l1l2_headline,
)


# Comparison outcome vocabulary (closed). STATUS_COMPARED means the pair was
# compared; every other value is a TYPED REASON the pair was not comparable.
STATUS_COMPARED = "compared"
REASON_TYPED_UNAVAILABLE = "typed_unavailable"
REASON_METRIC_VERSION_DRIFT = "metric_version_drift"
REASON_ZERO_DENOMINATOR_RATIO = "zero_denominator_ratio"
REASON_INVALID_COUNT = "invalid_count"
REASON_VALUE_OUT_OF_BOUNDS = "value_out_of_bounds"
REASON_IDENTITY_MISMATCH = "identity_mismatch"
REASON_PREDICTION_NOT_FINITE = "prediction_not_finite"

TYPED_REASONS = frozenset(
    {
        REASON_TYPED_UNAVAILABLE,
        REASON_METRIC_VERSION_DRIFT,
        REASON_ZERO_DENOMINATOR_RATIO,
        REASON_INVALID_COUNT,
        REASON_VALUE_OUT_OF_BOUNDS,
        REASON_IDENTITY_MISMATCH,
        REASON_PREDICTION_NOT_FINITE,
    }
)


@dataclass(frozen=True)
class ComparisonTolerances:
    """Per-kind comparison-geometry knobs (the only tunables).

    ``fraction_epsilon`` (0.01) matches ``comparison_contract.toml``'s
    ``[error_formula]`` epsilon: the fraction error is the epsilon-floored
    RELATIVE error ``abs(pred - gt) / max(abs(gt), epsilon)`` (NOT a plain
    absolute error), compared against ``fraction_rel_tol``."""

    fraction_rel_tol: float = 0.05
    fraction_epsilon: float = 0.01
    count_rel_tol: float = 0.05
    count_floor: float = 16.0
    ratio_rel_tol: float = 0.05
    ratio_epsilon: float = 0.125


@dataclass(frozen=True)
class MetricIdentity:
    """The full join key: a cell plus the metric within it."""

    arch: str
    slug: str
    shape_point: str
    metric_key: str


@dataclass(frozen=True)
class LabeledMetric:
    """An NCU ground-truth typed label at a cell identity."""

    identity: MetricIdentity
    label: tlp.TypedMetricLabel


@dataclass(frozen=True)
class MetricPrediction:
    """A predictor's value at a cell identity.

    ``registry_version`` is the contract the predictor was built against (drift
    from the committed version is a typed reason). ``l2_input_provenance`` is the
    L1->L2 filter provenance for an L2-layer prediction (``l1_miss_filtered`` is
    required in an L1/L2 headline comparison). ``ratio_denominator`` is the
    request count behind a ratio prediction; ``0`` makes the ratio undefined.
    """

    identity: MetricIdentity
    value: float
    registry_version: str
    l2_input_provenance: str | None = None
    ratio_denominator: float | None = None


@dataclass(frozen=True)
class ComparisonResult:
    identity: MetricIdentity
    kind: str | None
    comparison_formula: str | None
    status: str
    typed_reason: str | None
    label_value: float | None
    predicted_value: float | None
    error: float | None
    error_geometry: str | None
    within_tolerance: bool | None
    tolerance: float | None

    @property
    def comparable(self) -> bool:
        return self.status == STATUS_COMPARED

    def to_dict(self) -> dict:
        return {
            "arch": self.identity.arch,
            "slug": self.identity.slug,
            "shape_point": self.identity.shape_point,
            "metric_key": self.identity.metric_key,
            "kind": self.kind,
            "comparison_formula": self.comparison_formula,
            "status": self.status,
            "typed_reason": self.typed_reason,
            "label_value": self.label_value,
            "predicted_value": self.predicted_value,
            "error": self.error,
            "error_geometry": self.error_geometry,
            "within_tolerance": self.within_tolerance,
            "tolerance": self.tolerance,
        }


def _reason(
    identity: MetricIdentity,
    label: tlp.TypedMetricLabel | None,
    reason: str,
    predicted_value: float | None,
) -> ComparisonResult:
    return ComparisonResult(
        identity=identity,
        kind=label.kind if label else None,
        comparison_formula=label.comparison_formula if label else None,
        status=reason,
        typed_reason=reason,
        label_value=label.value if label else None,
        predicted_value=predicted_value,
        error=None,
        error_geometry=None,
        within_tolerance=None,
        tolerance=None,
    )


def _is_real_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def compare_one(
    labeled: LabeledMetric,
    predicted: MetricPrediction,
    *,
    registry_version: str = reg.REGISTRY_VERSION,
    tol: ComparisonTolerances = ComparisonTolerances(),
) -> ComparisonResult:
    """Compare one typed label to one prediction with the registry's per-kind
    geometry, returning a typed reason (never raising) for any non-comparable
    case. The join is asserted: a mismatched identity is ``identity_mismatch``."""
    identity = labeled.identity
    label = labeled.label

    if predicted.identity != identity:
        return _reason(identity, label, REASON_IDENTITY_MISMATCH, predicted.value)

    # Metric-version drift: both the label and the prediction must match the
    # committed registry contract.
    if (
        label.registry_version != registry_version
        or predicted.registry_version != registry_version
    ):
        return _reason(identity, label, REASON_METRIC_VERSION_DRIFT, predicted.value)

    if not _is_real_number(predicted.value):
        return _reason(identity, label, REASON_PREDICTION_NOT_FINITE, None)
    pv = float(predicted.value)

    kind = label.kind
    formula = label.comparison_formula

    if kind == reg.KIND_FRACTION:
        upper = label.bounds_upper
        if pv < label.bounds_lower or (upper is not None and pv > upper):
            return _reason(identity, label, REASON_VALUE_OUT_OF_BOUNDS, pv)
        # Epsilon-floored RELATIVE error per comparison_contract.toml's
        # [error_formula]: abs(pred - gt) / max(abs(gt), epsilon). The floor
        # keeps a near-zero ground-truth fraction from blowing the error up.
        error = abs(pv - label.value) / max(abs(label.value), tol.fraction_epsilon)
        tolerance = tol.fraction_rel_tol
    elif kind == reg.KIND_COUNT:
        if pv < 0 or pv != int(pv):
            return _reason(identity, label, REASON_INVALID_COUNT, pv)
        error = abs(pv - label.value) / max(abs(label.value), tol.count_floor)
        tolerance = tol.count_rel_tol
    elif kind == reg.KIND_RATIO:
        if predicted.ratio_denominator is not None and predicted.ratio_denominator == 0:
            return _reason(identity, label, REASON_ZERO_DENOMINATOR_RATIO, pv)
        if pv < label.bounds_lower:
            return _reason(identity, label, REASON_VALUE_OUT_OF_BOUNDS, pv)
        error = abs(pv - label.value) / max(abs(label.value), tol.ratio_epsilon)
        tolerance = tol.ratio_rel_tol
    else:  # pragma: no cover - registry kinds are closed
        return _reason(identity, label, REASON_VALUE_OUT_OF_BOUNDS, pv)

    return ComparisonResult(
        identity=identity,
        kind=kind,
        comparison_formula=formula,
        status=STATUS_COMPARED,
        typed_reason=None,
        label_value=label.value,
        predicted_value=pv,
        error=error,
        error_geometry=formula,
        within_tolerance=error <= tolerance,
        tolerance=tolerance,
    )


def _is_l1l2_headline(cache_layers) -> bool:
    layers = {str(layer).strip().upper() for layer in cache_layers}
    return {"L1", "L2"} <= layers


def run_comparisons(
    labels: list[LabeledMetric],
    predictions: list[MetricPrediction],
    *,
    cache_layers,
    registry_version: str = reg.REGISTRY_VERSION,
    tol: ComparisonTolerances = ComparisonTolerances(),
) -> list[ComparisonResult]:
    """Join ``labels`` to ``predictions`` on the full identity key and compare
    each pair; a label with no matching prediction is ``typed_unavailable``.

    When the comparison scope is an L1/L2 HEADLINE (both layers present), every
    L2-layer prediction must be the L1-miss-filtered stream -- this is enforced
    (``enforce_l1_miss_filtered_for_l1l2_headline``) BEFORE any result is emitted,
    so a full-stream L2 prediction fails the whole run closed."""
    predictions_by_identity: dict[MetricIdentity, MetricPrediction] = {
        prediction.identity: prediction for prediction in predictions
    }

    if _is_l1l2_headline(cache_layers):
        # Fail closed before emitting anything: EVERY L2-layer prediction in an
        # L1/L2 headline run must be L1-miss-filtered (None counts as
        # not-filtered) -- scan all predictions, not only label-matched ones, so
        # an unmatched full-stream L2 prediction cannot escape the guard. The
        # prediction's metric_key is resolved through the registry to decide
        # whether it describes an L2 layer.
        for prediction in predictions:
            spec = reg.REGISTRY.get(prediction.identity.metric_key)
            if spec is None or "L2" not in spec.admissible_layers:
                continue
            enforce_l1_miss_filtered_for_l1l2_headline(
                cache_layers,
                prediction.l2_input_provenance or L2_INPUT_FULL_STREAM,
            )

    results: list[ComparisonResult] = []
    for labeled in labels:
        prediction = predictions_by_identity.get(labeled.identity)
        if prediction is None:
            results.append(
                _reason(labeled.identity, labeled.label, REASON_TYPED_UNAVAILABLE, None)
            )
            continue
        results.append(
            compare_one(labeled, prediction, registry_version=registry_version, tol=tol)
        )
    return results


__all__ = [
    "ComparisonResult",
    "ComparisonTolerances",
    "LabeledMetric",
    "MetricIdentity",
    "MetricPrediction",
    "REASON_IDENTITY_MISMATCH",
    "REASON_INVALID_COUNT",
    "REASON_METRIC_VERSION_DRIFT",
    "REASON_PREDICTION_NOT_FINITE",
    "REASON_TYPED_UNAVAILABLE",
    "REASON_VALUE_OUT_OF_BOUNDS",
    "REASON_ZERO_DENOMINATOR_RATIO",
    "STATUS_COMPARED",
    "TYPED_REASONS",
    "compare_one",
    "run_comparisons",
]
