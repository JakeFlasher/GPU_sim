from __future__ import annotations

import json
import math

import pytest

from eval.direct_score import (
    ScoringError,
    main,
    summarize_keyed_metrics,
    summarize_metric,
)


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


@pytest.mark.parametrize("value", [None, True, "1", 10**1000])
def test_array_metrics_reject_non_numeric_and_unrepresentable_values(value: object) -> None:
    with pytest.raises(ScoringError):
        summarize_metric([value], [1.0])


def test_keyed_metrics_join_numeric_keys_and_report_coverage() -> None:
    summary = summarize_keyed_metrics(
        {"a": 100.0, "b": 200.0, "source": "ncu"},
        {"b": 220.0, "a": 90.0, "extra": 7.0, "enabled": True},
    )

    assert summary.shared_keys == ("a", "b")
    assert summary.coverage.observed_numeric_count == 2
    assert summary.coverage.predicted_numeric_count == 3
    assert summary.coverage.shared_numeric_count == 2
    assert summary.coverage.observed_fraction == 1.0
    assert summary.coverage.predicted_fraction == pytest.approx(2 / 3)
    assert summary.errors.mean_absolute_error == 15.0


def test_keyed_metrics_require_a_shared_numeric_key() -> None:
    with pytest.raises(ScoringError, match="share no numeric keys"):
        summarize_keyed_metrics({"observed": 1.0}, {"predicted": 1.0})


def test_cli_accepts_plain_json_objects(tmp_path, capsys) -> None:
    observed = tmp_path / "observed.json"
    predicted = tmp_path / "predicted.json"
    observed.write_text(json.dumps({"a": 10.0, "b": 20.0}), encoding="utf-8")
    predicted.write_text(json.dumps({"a": 8.0, "b": 23.0}), encoding="utf-8")

    assert main(["--observed", str(observed), "--predicted", str(predicted)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["coverage"]["shared_numeric_count"] == 2
    assert output["errors"]["mean_absolute_error"] == 2.5


@pytest.mark.parametrize("value", [None, True, "1", 10**1000])
def test_cli_reports_invalid_array_values_without_a_traceback(
    tmp_path, capsys, value: object
) -> None:
    observed = tmp_path / "observed.json"
    predicted = tmp_path / "predicted.json"
    observed.write_text(json.dumps([value]), encoding="utf-8")
    predicted.write_text("[1]", encoding="utf-8")

    assert main(["--observed", str(observed), "--predicted", str(predicted)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("eval.direct_score: ")
    assert "Traceback" not in captured.err


def test_keyed_metric_rejects_integer_outside_float_range() -> None:
    with pytest.raises(ScoringError, match="finite float range"):
        summarize_keyed_metrics({"metric": 10**1000}, {"metric": 1})
