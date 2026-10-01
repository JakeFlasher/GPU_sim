from __future__ import annotations

import pytest

from evaluate_prediction_splits import evaluate_prediction_splits


def test_ncu_join_happens_after_split_prediction_and_reports_coverage_refusal_and_cost():
    predictions = {
        "schema": "reification_prediction_v1",
        "rows": [
            {"workload": "cal-a", "split": "calibration", "prediction": {"point": 10, "lower": 9, "upper": 11}, "runtime_ms": 2, "memory_bytes": 100, "compression_ratio": 3},
            {"workload": "dev-a", "split": "development", "prediction": {"point": 20, "lower": 18, "upper": 22}, "runtime_ms": 3, "memory_bytes": 120, "compression_ratio": 2},
            {"workload": "ood-a", "split": "held_out", "status": "refused"},
        ],
    }
    observations = {
        "schema": "ncu_observation_v1",
        "rows": [{"workload": "cal-a", "value": 10}, {"workload": "dev-a", "value": 21}],
    }
    report = evaluate_prediction_splits(predictions, observations)
    assert report["target_join"] == "independent_ncu_after_prediction"
    assert report["selection_used_observation_error"] is False
    assert report["split_reports"]["calibration"]["interval_coverage"] == 1.0
    assert report["split_reports"]["development"]["mean_ape_pct"] == pytest.approx(100 / 21)
    assert report["split_reports"]["held_out"]["refusal_rate"] == 1.0


def test_prediction_cannot_smuggle_an_observed_target_into_the_feature_document():
    with pytest.raises(ValueError, match="must not contain observed"):
        evaluate_prediction_splits(
            {
                "schema": "reification_prediction_v1",
                "rows": [{"workload": "x", "split": "held_out", "observed": 7}],
            },
            {"schema": "ncu_observation_v1", "rows": []},
        )


def test_open_upper_interval_is_coverage_valid_but_has_no_finite_width():
    report = evaluate_prediction_splits(
        {
            "schema": "reification_prediction_v1",
            "rows": [{"workload": "x", "split": "held_out", "prediction": {"point": 10, "lower": 9, "upper": None}}],
        },
        {"schema": "ncu_observation_v1", "rows": [{"workload": "x", "value": 100}]},
    )
    held = report["split_reports"]["held_out"]
    assert held["interval_coverage"] == 1.0
    assert held["mean_interval_width"] is None


def test_variant_ranking_is_reported_without_using_observation_error():
    predictions = {
        "schema": "reification_prediction_v1",
        "rows": [
            {"workload": "x", "split": "held_out", "intervention": "fast", "prediction": {"point": 10, "lower": 10, "upper": None}},
            {"workload": "x", "split": "held_out", "intervention": "slow", "prediction": {"point": 20, "lower": 20, "upper": None}},
        ],
    }
    report = evaluate_prediction_splits(
        predictions,
        {"schema": "ncu_observation_v1", "rows": [{"workload": "x", "value": 100}]},
    )
    assert report["split_reports"]["held_out"]["variant_rankings"] == {"x": ["fast", "slow"]}
