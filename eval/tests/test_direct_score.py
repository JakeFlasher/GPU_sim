from __future__ import annotations

import math

import pytest

from eval.direct_score import ScoringError, summarize_metric


def test_summarize_metric_reports_direct_errors() -> None:
    summary = summarize_metric([100.0, 200.0], [90.0, 220.0])
    assert summary.count == 2
    assert summary.mean_absolute_error == 15.0
    assert summary.mean_absolute_percentage_error == pytest.approx(0.1)
    assert summary.root_mean_squared_error == pytest.approx(math.sqrt(250.0))


def test_zero_observation_disables_only_mape() -> None:
    summary = summarize_metric([0.0, 2.0], [1.0, 2.0])
    assert summary.mean_absolute_error == 0.5
    assert summary.mean_absolute_percentage_error is None


@pytest.mark.parametrize(
    ("observed", "predicted"),
    [([], []), ([1.0], []), ([float("nan")], [1.0])],
)
def test_invalid_inputs_are_rejected(
    observed: list[float], predicted: list[float]
) -> None:
    with pytest.raises(ScoringError):
        summarize_metric(observed, predicted)
