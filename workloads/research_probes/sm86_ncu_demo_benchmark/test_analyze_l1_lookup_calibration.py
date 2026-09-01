from __future__ import annotations

import copy

from workloads.research_probes.sm86_ncu_demo_benchmark import (
    analyze_l1_lookup_calibration as subject,
)


def _document(unified: float, read: float) -> dict:
    return {
        "launches": [
            {
                "metric_predictions": [
                    {
                        "ncu_metric": subject.UNIFIED_METRIC,
                        "outcome": {
                            "status": "predicted",
                            "value": {"kind": "fraction", "value": unified},
                        },
                    },
                    {
                        "ncu_metric": subject.READ_METRIC,
                        "outcome": {
                            "status": "predicted",
                            "value": {"kind": "fraction", "value": read},
                        },
                    },
                    {
                        "ncu_metric": "other",
                        "outcome": {"status": "predicted", "value": 7},
                    },
                ]
            }
        ],
        "unchanged": {"counter": 9},
    }


def test_prediction_fraction_converts_to_percent() -> None:
    document = _document(0.75, 0.25)
    assert subject._prediction_percent(document, subject.UNIFIED_METRIC) == 75.0
    assert subject._prediction_percent(document, subject.READ_METRIC) == 25.0


def test_normalization_removes_only_the_two_l1_outcomes() -> None:
    first = _document(0.1, 0.2)
    second = _document(0.9, 0.8)
    assert subject._without_l1_outcomes(first) == subject._without_l1_outcomes(second)
    changed = copy.deepcopy(second)
    changed["unchanged"]["counter"] = 10
    assert subject._without_l1_outcomes(first) != subject._without_l1_outcomes(changed)


def test_zero_target_has_null_ape_but_retains_percentage_point_error() -> None:
    assert subject._score(0.0, 0.0) == {
        "absolute_percentage_point_error": 0.0,
        "ape_pct": None,
    }
