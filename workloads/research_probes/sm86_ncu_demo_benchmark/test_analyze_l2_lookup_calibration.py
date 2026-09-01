from __future__ import annotations

import copy

from workloads.research_probes.sm86_ncu_demo_benchmark import (
    analyze_l2_lookup_calibration as subject,
)


def _document(fraction: float) -> dict:
    return {
        "launches": [
            {
                "metric_predictions": [
                    {
                        "ncu_metric": subject.L2_METRIC,
                        "outcome": {
                            "status": "predicted",
                            "value": {"kind": "fraction", "value": fraction},
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
    assert subject._prediction_percent(_document(0.75)) == 75.0


def test_normalization_removes_only_l2_outcome() -> None:
    first = _document(0.1)
    second = copy.deepcopy(first)
    second["launches"][0]["metric_predictions"][0]["outcome"] = {
        "status": "predicted",
        "value": {"kind": "fraction", "value": 0.9},
    }
    assert subject._without_l2_outcome(first) == subject._without_l2_outcome(second)
    second["unchanged"]["counter"] = 10
    assert subject._without_l2_outcome(first) != subject._without_l2_outcome(second)
