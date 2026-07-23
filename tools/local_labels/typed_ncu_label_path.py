"""Separate typed label path for the cache/memory NCU metric registry.

This is the typed extraction/admission path that consumes the typed metric
registry (``ncu_metric_registry``) and admits an already-collected NCU metric
VALUE (a ``fraction``, a ``count``, or a ``ratio``) into a typed label that
carries the registry metadata. Admission is KIND-ROUTED: a value pushed through
the wrong-kind entry point (e.g. a ``count`` through the fraction path) is
refused, mirroring the registry's own kind/comparison-formula coupling.

It is DELIBERATELY separate from ``parse_ncu_metrics.py``, whose narrow
percentage-only ``[0,1]``-fraction contract is left untouched: that parser admits
only ``.pct`` fraction metrics, while this path admits the full
``fraction | count | ratio`` typed registry and emits the metadata the metric
comparison runner will read (the lockstep: both read the SAME registry, and the
label stamps ``registry_version`` so a version-drift check is possible).

This path NEVER invokes NCU and writes no ``.ncu-rep`` artifact -- it consumes
VALUES that were collected elsewhere (the live local-label NCU collection lives
in its own driver). In tests the values are synthetic fixtures.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import ncu_metric_registry as reg  # noqa: E402


class TypedLabelAdmissionError(ValueError):
    """Raised when a metric value cannot be admitted as its registry kind:
    the key is unregistered or forbidden, the value violates the kind's bounds /
    type, or the value was pushed through the wrong-kind admission path."""

    TOKEN = "typed_ncu_label_admission_rejected"


@dataclass(frozen=True)
class TypedMetricLabel:
    """An admitted typed metric value + the registry metadata the comparison
    runner reads. ``registry_version`` pins the contract the label was admitted
    against (so a later metric-version-drift check has something to compare)."""

    key: str
    kind: str
    value: float
    unit: str
    bounds_lower: float
    bounds_upper: float | None
    admissible_layers: tuple[str, ...]
    repeatability_stat: str
    comparison_formula: str
    prediction_source: str
    ncu_token: str | None
    numerator_token: str | None
    denominator_token: str | None
    registry_version: str

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "kind": self.kind,
            "value": self.value,
            "unit": self.unit,
            "bounds_lower": self.bounds_lower,
            "bounds_upper": self.bounds_upper,
            "admissible_layers": list(self.admissible_layers),
            "repeatability_stat": self.repeatability_stat,
            "comparison_formula": self.comparison_formula,
            "prediction_source": self.prediction_source,
            "ncu_token": self.ncu_token,
            "numerator_token": self.numerator_token,
            "denominator_token": self.denominator_token,
            "registry_version": self.registry_version,
        }


def _spec(key: str) -> reg.MetricSpec:
    """Resolve a registry key to its MetricSpec, failing closed for an
    unregistered or forbidden key/token (surfacing the registry's own reason)."""
    spec = reg.REGISTRY.get(key)
    if spec is not None:
        return spec
    # Not a registry key. If it is an NCU token, surface the registry's typed
    # allowlist/forbidden reason; otherwise it is simply unregistered.
    try:
        reg.assert_token_admissible(key)
    except reg.RegistryValidationError as exc:
        raise TypedLabelAdmissionError(f"{key!r}: {exc}") from exc
    raise TypedLabelAdmissionError(
        f"{key!r} is an admissible base token but not a registry KEY; admit by "
        f"registry key (one of REGISTRY), not by raw token"
    )


def _coerce_number(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypedLabelAdmissionError(
            f"{key}: value {value!r} is not a real number"
        )
    v = float(value)
    if not math.isfinite(v):
        raise TypedLabelAdmissionError(f"{key}: value {value!r} is not finite")
    return v


def _within_bounds(v: float, spec: reg.MetricSpec) -> bool:
    if v < spec.bounds_lower:
        return False
    if spec.bounds_upper is not None and v > spec.bounds_upper:
        return False
    return True


def _label(spec: reg.MetricSpec, value: float) -> TypedMetricLabel:
    return TypedMetricLabel(
        key=spec.key,
        kind=spec.kind,
        value=value,
        unit=spec.unit,
        bounds_lower=spec.bounds_lower,
        bounds_upper=spec.bounds_upper,
        admissible_layers=spec.admissible_layers,
        repeatability_stat=spec.repeatability_stat,
        comparison_formula=spec.comparison_formula,
        prediction_source=spec.prediction_source,
        ncu_token=spec.ncu_token,
        numerator_token=spec.numerator_token,
        denominator_token=spec.denominator_token,
        registry_version=reg.REGISTRY_VERSION,
    )


def _require_kind(spec: reg.MetricSpec, expected: str) -> None:
    if spec.kind != expected:
        raise TypedLabelAdmissionError(
            f"{spec.key}: declared kind {spec.kind!r} is not {expected!r}; "
            f"refusing to admit it through the {expected} path"
        )


def admit_fraction(key: str, value: object) -> TypedMetricLabel:
    """Admit a fraction value (must be a registry ``fraction`` in ``[0,1]``)."""
    spec = _spec(key)
    _require_kind(spec, reg.KIND_FRACTION)
    v = _coerce_number(value, key)
    if not _within_bounds(v, spec):
        raise TypedLabelAdmissionError(
            f"{key}: fraction {v} outside "
            f"[{spec.bounds_lower}, {spec.bounds_upper}]"
        )
    return _label(spec, v)


def admit_count(key: str, value: object) -> TypedMetricLabel:
    """Admit a count value (registry ``count``; non-negative + integer-valued)."""
    spec = _spec(key)
    _require_kind(spec, reg.KIND_COUNT)
    v = _coerce_number(value, key)
    if not _within_bounds(v, spec):
        raise TypedLabelAdmissionError(f"{key}: count {v} < {spec.bounds_lower}")
    if v != int(v):
        raise TypedLabelAdmissionError(f"{key}: count {v} is not integer-valued")
    return _label(spec, float(int(v)))


def admit_ratio(key: str, value: object) -> TypedMetricLabel:
    """Admit a ratio value directly (registry ``ratio``; non-negative). For a
    ratio derived from its component counts use ``admit_ratio_from_components``."""
    spec = _spec(key)
    _require_kind(spec, reg.KIND_RATIO)
    v = _coerce_number(value, key)
    if not _within_bounds(v, spec):
        raise TypedLabelAdmissionError(f"{key}: ratio {v} < {spec.bounds_lower}")
    return _label(spec, v)


def admit_ratio_from_components(
    key: str, numerator: object, denominator: object
) -> TypedMetricLabel:
    """Admit a ratio computed from its component counts (the registry
    ``numerator_token`` / ``denominator_token`` semantics): both must be
    non-negative integer-valued counts and the denominator must be positive
    (a zero denominator is refused here -- a ratio is undefined)."""
    spec = _spec(key)
    _require_kind(spec, reg.KIND_RATIO)
    num = _coerce_number(numerator, key)
    den = _coerce_number(denominator, key)
    for label, raw in (("numerator", num), ("denominator", den)):
        if raw < 0 or raw != int(raw):
            raise TypedLabelAdmissionError(
                f"{key}: ratio {label} {raw} must be a non-negative integer count"
            )
    if den == 0:
        raise TypedLabelAdmissionError(
            f"{key}: ratio denominator is zero; the ratio is undefined"
        )
    return admit_ratio(key, num / den)


_KIND_TO_ADMIT = {
    reg.KIND_FRACTION: admit_fraction,
    reg.KIND_COUNT: admit_count,
    reg.KIND_RATIO: admit_ratio,
}


def admit_typed_label(key: str, value: object) -> TypedMetricLabel:
    """Admit ``value`` by the registry's DECLARED kind for ``key`` (dispatches to
    the kind-specific admit). To assert a CALLER's expected kind, call the
    kind-specific entry point directly so a wrong-kind value fails closed."""
    spec = _spec(key)
    return _KIND_TO_ADMIT[spec.kind](key, value)


__all__ = [
    "TypedLabelAdmissionError",
    "TypedMetricLabel",
    "admit_count",
    "admit_fraction",
    "admit_ratio",
    "admit_ratio_from_components",
    "admit_typed_label",
]
