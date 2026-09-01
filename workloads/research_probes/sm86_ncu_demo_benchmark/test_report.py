from __future__ import annotations

import math
import json

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
                "value": {
                    "kind": "fraction",
                    "value": 0.25,
                    "numerator": 25,
                    "denominator": 100,
                },
            }
        },
        "l2",
    )
    assert projected == {
        "status": "predicted",
        "value": 25.0,
        "kind": "fraction",
        "hit_sectors": 25.0,
        "lookup_sectors": 100.0,
        "miss_sectors": 75.0,
    }


def test_exact_zero_hit_rate_is_valid_not_unavailable() -> None:
    comparison = report._metric_comparison(
        {"hit_rate_member": True},
        {"value": 0.0},
        {"value": 0.0},
    )
    assert comparison["ape_pct"] is None
    assert comparison["absolute_percentage_point_error"] == 0.0
    assert comparison["zero_safe_full_scale_error_pct"] == 0.0
    assert comparison["zero_safe_status"] == "exact_zero"


def test_ncu_hit_miss_sources_retain_zero_hit_denominator() -> None:
    summary = report._ncu_metric(
        [
            {
                "metrics": {
                    "m": {
                        "status": "measured",
                        "value": 0.0,
                        "hit_sectors": 0.0,
                        "miss_sectors": 64.0,
                    }
                }
            }
            for _ in range(3)
        ],
        "m",
    )
    assert summary["source_sector_counts"]["hit_sectors"]["median"] == 0.0
    assert summary["source_sector_counts"]["miss_sectors"]["median"] == 64.0


def test_ncu_repeat_mean_preserves_raw_values() -> None:
    summary = report._ncu_metric(
        [
            {"metrics": {"m": {"status": "measured", "value": 10.0}}},
            {"metrics": {"m": {"status": "measured", "value": 14.0}}},
            {"metrics": {"m": {"status": "measured", "value": 100.0}}},
        ],
        "m",
    )
    assert summary["value"] == 14.0
    assert summary["mean"] == 124.0 / 3.0
    assert summary["repeat_values"] == [10.0, 14.0, 100.0]
    assert summary["coverage"] == 3


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
            "target_launches_per_process": 1,
            "ncu_launch_samples": 7,
            "simulator_process_samples": 7,
            "timing": {
                "gpu_kernel_seconds": scale,
                "ncu_profile_wall_seconds_mean": scale * 2,
                "ncu_profile_wall_seconds_total": scale * 6,
                "simulator_wall_seconds": scale * 4,
            },
            "accuracy": {
                "terminal_lower_bound_mape_pct": scale,
                "issue_frontier_mape_pct": scale,
                "runtime_equivalent_mape_pct": scale,
                "instruction_throughput_mape_pct": scale,
            },
            "workload_id": f"row-{scale}",
            "ncu_repeatability": {
                "max_relative_range_pct": scale,
                "max_relative_range_metric": "m",
            },
            "simulator_repeatability": {
                "wall_relative_range_pct": scale,
                "deterministic_predictions": True,
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


def test_timing_crosswalk_preserves_two_distinct_envelopes() -> None:
    crosswalk = json.loads((report.HERE / "timing_crosswalk.json").read_text())
    assert report._timing_envelopes(crosswalk) == (964.0, 1642.0)
    assert report._timing_envelopes(None) == (0.0, 0.0)


def test_caveats_distinguish_smoke_from_final_repeats() -> None:
    assert any("3 independent" in note for note in report._caveats("smoke", 3, 3))
    assert any("7 independent" in note for note in report._caveats("final", 7, 7))
    single_process = report._caveats("final", 7, 1)
    assert any("One simulator process" in note for note in single_process)
    assert any("not reassessed" in note for note in single_process)
    assert all(
        "Every smoke" not in note for note in report._caveats("final", 7, 7)
    )


def test_report_campaign_path_is_repository_relative() -> None:
    path = report.REPO_ROOT / "results/captures/example/campaign_manifest.json"
    assert report._portable_path(path) == "results/captures/example/campaign_manifest.json"


def test_absent_optional_archive_is_explicit(tmp_path) -> None:
    campaign_path = tmp_path / "campaign_manifest.json"
    campaign_path.write_text("{}", encoding="utf-8")
    assert report._evidence_archive(campaign_path) is None


def test_accuracy_gate_requires_all_metric_and_repeat_checks() -> None:
    gate = json.loads((report.HERE / "accuracy_gate.json").read_text())

    def row(index: int, error: float) -> dict:
        metrics = {
            metric: {"ape_pct": error} for metric in report.COUNTER_IDS
        }
        metrics.update(
            {
                metric: {
                    "ape_pct": error,
                    "absolute_percentage_point_error": error / 2,
                }
                for metric in report.HIT_RATE_IDS
            }
        )
        return {
            "workload_id": f"w{index}",
            "target_launches_per_process": 1,
            "ncu_launch_samples": 7,
            "simulator_process_samples": 7,
            "timing": {
                "gpu_kernel_seconds": 1.0,
                "ncu_profile_wall_seconds_mean": 1.0,
                "ncu_profile_wall_seconds_total": 7.0,
                "simulator_wall_seconds": 1.0,
            },
            "accuracy": {
                "terminal_lower_bound_mape_pct": error,
                "issue_frontier_mape_pct": error,
                "runtime_equivalent_mape_pct": error,
                "instruction_throughput_mape_pct": error,
            },
            "ncu_repeatability": {
                "max_relative_range_pct": 1.0,
                "max_relative_range_metric": "m",
            },
            "simulator_repeatability": {
                "wall_relative_range_pct": 1.0,
                "deterministic_predictions": True,
            },
            "metrics": metrics,
        }

    passing_rows = [row(index, 10.0) for index in range(8)]
    passing = report._evaluate_accuracy_gate(
        gate, passing_rows, report._totals(passing_rows)
    )
    assert passing["technical_status"] == "pass"
    assert passing["manual_approval_status"] == "pending"
    failing_rows = [row(index, 25.0) for index in range(8)]
    failing = report._evaluate_accuracy_gate(
        gate, failing_rows, report._totals(failing_rows)
    )
    assert failing["technical_status"] == "fail"
    assert not failing["candidate_promotable_for_user_review"]


def test_count_target_gate_uses_all_hit_and_miss_components() -> None:
    gate = json.loads((report.HERE / "accuracy_gate_counts.json").read_text())

    def row(index: int) -> dict:
        metrics = {
            metric: {"ape_pct": 1.0} for metric in report.COUNTER_IDS
        }
        metrics.update(
            {
                metric: {
                    "ape_pct": 1.0,
                    "absolute_percentage_point_error": 0.5,
                }
                for metric in report.HIT_RATE_IDS
            }
        )
        metrics.update(
            {
                metric: {
                    "ape_pct": 1.0,
                    "pair_normalized_absolute_error_pct": 2.0,
                }
                for metric in report.CACHE_LOOKUP_COUNT_IDS
            }
        )
        return {
            "workload_id": f"w{index}",
            "target_launches_per_process": 1,
            "ncu_launch_samples": 7,
            "simulator_process_samples": 7,
            "timing": {
                "gpu_kernel_seconds": 1.0,
                "ncu_profile_wall_seconds_mean": 1.0,
                "ncu_profile_wall_seconds_total": 7.0,
                "simulator_wall_seconds": 1.0,
            },
            "accuracy": {
                "terminal_lower_bound_mape_pct": 5.0,
                "issue_frontier_mape_pct": 5.0,
                "runtime_equivalent_mape_pct": 5.0,
                "instruction_throughput_mape_pct": 5.0,
            },
            "ncu_repeatability": {
                "max_relative_range_pct": 1.0,
                "max_relative_range_metric": "m",
            },
            "simulator_repeatability": {
                "wall_relative_range_pct": 1.0,
                "deterministic_predictions": True,
            },
            "metrics": metrics,
        }

    rows = [row(index) for index in range(8)]
    total = report._totals(rows)
    assert total["accuracy"]["cache_lookup_count_coverage"] == 48
    assert total["accuracy"]["cache_lookup_count_normalized_error_pct"] == 2.0
    evaluated = report._evaluate_accuracy_gate(gate, rows, total)
    assert evaluated["technical_status"] == "pass"


def test_single_simulator_gate_does_not_claim_unmeasured_determinism() -> None:
    gate = json.loads(
        (report.HERE / "accuracy_gate_counts_single_sim.json").read_text()
    )

    def row(index: int) -> dict:
        metrics = {metric: {"ape_pct": 1.0} for metric in report.COUNTER_IDS}
        metrics.update(
            {
                metric: {
                    "ape_pct": 1.0,
                    "absolute_percentage_point_error": 0.5,
                }
                for metric in report.HIT_RATE_IDS
            }
        )
        metrics.update(
            {
                metric: {
                    "ape_pct": 1.0,
                    "pair_normalized_absolute_error_pct": 2.0,
                }
                for metric in report.CACHE_LOOKUP_COUNT_IDS
            }
        )
        return {
            "workload_id": f"w{index}",
            "target_launches_per_process": 1,
            "ncu_launch_samples": 7,
            "simulator_process_samples": 1,
            "timing": {
                "gpu_kernel_seconds": 1.0,
                "ncu_profile_wall_seconds_mean": 1.0,
                "ncu_profile_wall_seconds_total": 7.0,
                "simulator_wall_seconds": 1.0,
            },
            "accuracy": {
                "terminal_lower_bound_mape_pct": 5.0,
                "issue_frontier_mape_pct": 5.0,
                "runtime_equivalent_mape_pct": 5.0,
                "instruction_throughput_mape_pct": 5.0,
            },
            "ncu_repeatability": {
                "max_relative_range_pct": 1.0,
                "max_relative_range_metric": "m",
            },
            "simulator_repeatability": {
                "wall_relative_range_pct": None,
                "deterministic_predictions": None,
                "prediction_determinism_status": "not_assessed_single_process",
            },
            "metrics": metrics,
        }

    rows = [row(index) for index in range(8)]
    total = report._totals(rows)
    assert total["simulator_repeatability"]["all_predictions_deterministic"] is None
    assert total["simulator_repeatability"]["maximum_wall_relative_range_pct"] is None
    assert (
        total["simulator_repeatability"]["prediction_determinism_status"]
        == "not_assessed_single_process"
    )
    evaluated = report._evaluate_accuracy_gate(gate, rows, total)
    assert evaluated["technical_status"] == "pass"
    assert evaluated["repeat_checks"]["simulator_repeats_per_workload"] == {
        "value": 1,
        "required": 1,
        "passed": True,
    }
    assert evaluated["repeat_checks"]["simulator_prediction_determinism"] == {
        "value": "not_assessed_single_process",
        "required": "not_required_for_single_process_policy",
        "passed": True,
    }


def test_large_workload_gate_rejects_aggregate_pass_with_row_failure() -> None:
    gate = json.loads((report.HERE / "accuracy_gate_large_workloads.json").read_text())

    def row(index: int, timing_error: float) -> dict:
        metrics = {metric: {"ape_pct": 10.0} for metric in report.COUNTER_IDS}
        metrics["gpc_elapsed_cycles"] = {"ncu": {"value": 10000.0}}
        metrics.update(
            {
                metric: {
                    "ape_pct": 10.0,
                    "absolute_percentage_point_error": 1.0,
                }
                for metric in report.HIT_RATE_IDS
            }
        )
        metrics.update(
            {
                metric: {
                    "ape_pct": 10.0,
                    "pair_normalized_absolute_error_pct": 10.0,
                }
                for metric in report.CACHE_LOOKUP_COUNT_IDS
            }
        )
        return {
            "workload_id": f"large-{index}",
            "target_launches_per_process": 1,
            "ncu_launch_samples": 7,
            "simulator_process_samples": 1,
            "timing": {
                "gpu_kernel_seconds": 1.0,
                "ncu_profile_wall_seconds_mean": 1.0,
                "ncu_profile_wall_seconds_total": 7.0,
                "simulator_wall_seconds": 1.0,
            },
            "accuracy": {
                "terminal_lower_bound_mape_pct": timing_error,
                "issue_frontier_mape_pct": timing_error,
                "runtime_equivalent_mape_pct": timing_error,
                "instruction_throughput_mape_pct": timing_error,
            },
            "ncu_repeatability": {
                "max_relative_range_pct": 1.0,
                "max_relative_range_metric": "m",
            },
            "simulator_repeatability": {
                "wall_relative_range_pct": None,
                "deterministic_predictions": None,
                "prediction_determinism_status": "not_assessed_single_process",
            },
            "metrics": metrics,
        }

    rows = [row(0, 25.0), *[row(index, 10.0) for index in range(1, 7)]]
    total = report._totals(rows)
    evaluated = report._evaluate_accuracy_gate(gate, rows, total)
    terminal = evaluated["metric_checks"]["terminal_cycle_mape_pct"]
    assert terminal["value"] < 20.0
    assert terminal["maximum_observed_per_workload_pct"] == 25.0
    assert terminal["maximum_observed_identity"] == "large-0"
    assert not terminal["passed"]
    assert evaluated["technical_status"] == "fail"
