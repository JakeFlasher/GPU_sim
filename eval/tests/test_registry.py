from __future__ import annotations

import pytest

from eval.contracts.validation import METRIC_KEYS
from eval.registry import (
    CYCLE_BLOCKER,
    CYCLE_METRIC_KEY,
    SCORABLE_METRIC_KEYS,
    load_metric_registry,
)


def test_loads_exact_immutable_7plus1_projection() -> None:
    registry = load_metric_registry()

    assert tuple(metric.key for metric in registry.metrics) == METRIC_KEYS
    assert tuple(metric.key for metric in registry.metrics if metric.scorable) == (
        SCORABLE_METRIC_KEYS
    )
    assert registry.by_key(CYCLE_METRIC_KEY).scorable is False
    assert registry.by_key(CYCLE_METRIC_KEY).unavailable_status == (
        "required_unimplemented"
    )
    assert registry.by_key(CYCLE_METRIC_KEY).unavailable_blocking == CYCLE_BLOCKER
    assert all(
        metric.unavailable_status is None
        and metric.unavailable_blocking is None
        for metric in registry.metrics[:-1]
    )

    with pytest.raises(TypeError):
        registry.repeatability_thresholds["stddev_abs"] = 1.0  # type: ignore[index]


def test_unknown_metric_has_no_fallback() -> None:
    registry = load_metric_registry()

    with pytest.raises(KeyError, match="not_a_metric"):
        registry.by_key("not_a_metric")
