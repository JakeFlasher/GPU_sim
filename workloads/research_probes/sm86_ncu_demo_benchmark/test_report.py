from __future__ import annotations

import math

from workloads.research_probes.sm86_ncu_demo_benchmark import report


def test_ape_is_unavailable_for_zero_target() -> None:
    assert report._ape(0.0, 0.0) is None
    assert report._ape(12.0, 10.0) == 20.0
    assert report._absolute_error(12.0, 10.0) == 2.0


def test_simulator_fraction_is_converted_to_percent() -> None:
    projected = report._simulator_projection(
        {
            "outcome": {
                "status": "predicted",
                "value": {"kind": "fraction", "value": 0.25},
            }
        },
        "l2",
    )
    assert projected == {"status": "predicted", "value": 25.0, "kind": "fraction"}


def test_ncu_repeat_mean_preserves_raw_values() -> None:
    summary = report._ncu_metric(
        [
            {"metrics": {"m": {"status": "measured", "value": 10.0}}},
            {"metrics": {"m": {"status": "measured", "value": 14.0}}},
        ],
        "m",
    )
    assert summary["value"] == 12.0
    assert summary["repeat_values"] == [10.0, 14.0]
    assert summary["coverage"] == 2


def test_total_time_ratios_use_summed_numerators_and_denominators() -> None:
    def row(scale: float) -> dict:
        metrics = {
            metric: {"ape_pct": scale}
            for metric in report.COUNTER_IDS
        }
        metrics.update(
            {
                metric: {
                    "ape_pct": scale,
                    "absolute_percentage_point_error": scale / 2,
                }
                for metric in report.HIT_RATE_IDS
            }
        )
        return {
            "launches": 1,
            "timing": {
                "gpu_kernel_seconds": scale,
                "ncu_profile_wall_seconds_mean": scale * 2,
                "ncu_profile_wall_seconds_total": scale * 6,
                "simulator_wall_seconds": scale * 4,
            },
            "accuracy": {
                "terminal_lower_bound_mape_pct": scale,
                "issue_frontier_mape_pct": scale,
            },
            "metrics": metrics,
        }

    total = report._totals([row(1.0), row(3.0)])
    assert math.isclose(total["timing"]["gpu_execution_over_simulator_wall"], 0.25)
    assert math.isclose(total["timing"]["ncu_profile_wall_over_simulator_wall"], 0.5)
    assert total["accuracy"]["counter_coverage"] == 8


def test_missing_percentage_formats_without_a_percent_suffix() -> None:
    assert report._pct(None) == "n/a"
    assert report._pct(12.5) == "12.50%"
