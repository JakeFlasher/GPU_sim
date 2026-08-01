"""Immutable consumption of the frozen exact 7+1 metric projection."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from eval.contracts.validation import (
    METRIC_KEYS,
    ContractValidationError,
    validate_registry_projection,
)

CYCLE_METRIC_KEY = "exec_cycles_gpc_max"
CYCLE_BLOCKER = "aggregation_semantics_study"
SCORABLE_METRIC_KEYS = METRIC_KEYS[:-1]


@dataclass(frozen=True)
class MetricDescriptor:
    """One exact projected metric plus its evaluator scoring disposition."""

    key: str
    ncu_metric: str
    unit: str
    family: str
    metric_class: str
    repeatability_rule: str
    scorable: bool
    unavailable_status: str | None = None
    unavailable_blocking: str | None = None


@dataclass(frozen=True)
class MetricRegistry:
    """An immutable, trust-pinned view of the exact metric projection."""

    projection_sha256: str
    source_registry_version: int
    source_registry_sha256: str
    source_label_schema_version: int
    repeatability_thresholds: Mapping[str, float]
    metrics: tuple[MetricDescriptor, ...]

    def by_key(self, key: str) -> MetricDescriptor:
        for metric in self.metrics:
            if metric.key == key:
                return metric
        raise KeyError(key)


def load_metric_registry() -> MetricRegistry:
    """Validate and snapshot the frozen projection into immutable values.

    The P1 validator authenticates both the projection seal and the byte-exact
    source registry projection. This reader deliberately adds no fallback path.
    """

    projection = validate_registry_projection()
    rows = projection["metrics"]
    observed_keys = tuple(row["key"] for row in rows)
    if observed_keys != METRIC_KEYS:
        raise ContractValidationError(
            "REGISTRY_READER_INVARIANT_VIOLATED",
            "validated metric projection order invariant violated",
        )

    metrics = tuple(
        MetricDescriptor(
            key=row["key"],
            ncu_metric=row["ncu_metric"],
            unit=row["unit"],
            family=row["family"],
            metric_class=row["metric_class"],
            repeatability_rule=row["repeatability_rule"],
            scorable=row["key"] != CYCLE_METRIC_KEY,
            unavailable_status=(
                "required_unimplemented"
                if row["key"] == CYCLE_METRIC_KEY
                else None
            ),
            unavailable_blocking=(
                CYCLE_BLOCKER if row["key"] == CYCLE_METRIC_KEY else None
            ),
        )
        for row in rows
    )

    if tuple(metric.key for metric in metrics if metric.scorable) != SCORABLE_METRIC_KEYS:
        raise ContractValidationError(
            "REGISTRY_READER_INVARIANT_VIOLATED",
            "validated metric scoring disposition invariant violated",
        )
    cycles = metrics[-1]
    if (
        cycles.key != CYCLE_METRIC_KEY
        or cycles.scorable
        or cycles.unavailable_status != "required_unimplemented"
        or cycles.unavailable_blocking != CYCLE_BLOCKER
    ):
        raise ContractValidationError(
            "REGISTRY_READER_INVARIANT_VIOLATED",
            "validated cycle blocker invariant violated",
        )

    thresholds = MappingProxyType(dict(projection["repeatability_thresholds"]))
    return MetricRegistry(
        projection_sha256=projection["projection_sha256"],
        source_registry_version=projection["source_registry_version"],
        source_registry_sha256=projection["source_registry_sha256"],
        source_label_schema_version=projection["source_label_schema_version"],
        repeatability_thresholds=thresholds,
        metrics=metrics,
    )
