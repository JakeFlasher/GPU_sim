#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import validate_spec as subject  # noqa: E402


class BenchmarkSpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = json.loads((HERE / "benchmark_spec.json").read_text())
        cls.metrics = json.loads((HERE / "metric_schema.json").read_text())
        cls.profile = json.loads((HERE / "host_profile.json").read_text())
        cls.gate = json.loads((HERE / "accuracy_gate.json").read_text())
        cls.crosswalk = json.loads((HERE / "timing_crosswalk.json").read_text())
        cls.count_metrics = json.loads(
            (HERE / "metric_schema_counts.json").read_text()
        )
        cls.count_gate = json.loads(
            (HERE / "accuracy_gate_counts.json").read_text()
        )
        cls.single_sim_gate = json.loads(
            (HERE / "accuracy_gate_counts_single_sim.json").read_text()
        )
        cls.large_spec = json.loads((HERE / "large_workload_spec.json").read_text())
        cls.large_gate = json.loads(
            (HERE / "accuracy_gate_large_workloads.json").read_text()
        )

    def test_frozen_spec_is_valid(self) -> None:
        subject.validate_spec(self.spec, self.count_metrics, self.profile)
        subject.validate_accuracy_gate(self.single_sim_gate)
        subject.validate_timing_crosswalk(self.crosswalk)

    def test_timing_crosswalk_recomputes_control_residuals(self) -> None:
        mutated = copy.deepcopy(self.crosswalk)
        mutated["calibration"]["cells"][0]["simulator_issue_cycles"] += 1
        with self.assertRaisesRegex(subject.ValidationError, "SM-active residual drift"):
            subject.validate_timing_crosswalk(mutated)

    def test_count_target_contract_is_valid(self) -> None:
        subject.validate_spec(self.spec, self.count_metrics, self.profile)
        subject.validate_accuracy_gate(self.count_gate)
        subject.validate_accuracy_gate(self.single_sim_gate)

    def test_single_simulator_gate_requires_one_process(self) -> None:
        mutated = copy.deepcopy(self.single_sim_gate)
        mutated["required_simulator_repeats"] = 7
        with self.assertRaisesRegex(subject.ValidationError, "require 1 simulator"):
            subject.validate_accuracy_gate(mutated)

    def test_large_workload_contract_is_valid_and_per_row_gated(self) -> None:
        subject.validate_spec(self.large_spec, self.count_metrics, self.profile)
        subject.validate_accuracy_gate(self.large_gate)
        mutated = copy.deepcopy(self.large_gate)
        mutated["checks"]["terminal_cycle_mape_pct"]["maximum_per_workload"] = 20.1
        with self.assertRaisesRegex(subject.ValidationError, "per-workload maximum drift"):
            subject.validate_accuracy_gate(mutated)

    def test_large_workload_feedback_is_exact_and_independently_bound(self) -> None:
        subject.validate_simulator_feedback(self.large_spec["simulator"])
        mutated = copy.deepcopy(self.large_spec["simulator"])
        mutated["memory_admission_feedback"] = "candidate.json"
        with self.assertRaisesRegex(subject.ValidationError, "feedback path drift"):
            subject.validate_simulator_feedback(mutated)

    def test_count_target_gate_requires_all_six_components(self) -> None:
        mutated = copy.deepcopy(self.count_gate)
        mutated["checks"]["cache_lookup_count_normalized_error_pct"][
            "required_coverage"
        ] = 47
        with self.assertRaisesRegex(subject.ValidationError, "coverage drift"):
            subject.validate_accuracy_gate(mutated)

    def test_duplicate_workload_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.spec)
        mutated["workloads"][1]["id"] = mutated["workloads"][0]["id"]
        with self.assertRaisesRegex(subject.ValidationError, "workload ids must be unique"):
            subject.validate_spec(mutated, self.metrics, self.profile)

    def test_ncu_metric_order_drift_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.spec)
        mutated["ncu"]["metrics"].reverse()
        with self.assertRaisesRegex(subject.ValidationError, "collection order differs"):
            subject.validate_spec(mutated, self.metrics, self.profile)

    def test_accuracy_gate_cannot_relax_twenty_percent(self) -> None:
        mutated = copy.deepcopy(self.gate)
        mutated["checks"]["l2_unified_hit_rate_mape_pct"]["maximum"] = 20.1
        with self.assertRaisesRegex(subject.ValidationError, "maximum drift"):
            subject.validate_accuracy_gate(mutated)

    def test_independent_cache_calibration_spec_is_valid(self) -> None:
        calibration_spec = json.loads(
            (HERE / "cache_calibration_spec.json").read_text()
        )
        calibration_metrics = json.loads(
            (HERE / "cache_calibration_metrics.json").read_text()
        )
        subject.validate_spec(calibration_spec, calibration_metrics, self.profile)

    def test_post_candidate_l2_holdout_spec_is_valid(self) -> None:
        holdout = json.loads((HERE / "cache_lookup_holdout_spec.json").read_text())
        calibration_metrics = json.loads(
            (HERE / "cache_calibration_metrics.json").read_text()
        )
        subject.validate_spec(holdout, calibration_metrics, self.profile)

    def test_post_candidate_l1_holdout_spec_is_valid(self) -> None:
        holdout = json.loads((HERE / "l1_lookup_holdout_spec.json").read_text())
        calibration_metrics = json.loads(
            (HERE / "cache_calibration_metrics.json").read_text()
        )
        subject.validate_spec(holdout, calibration_metrics, self.profile)


if __name__ == "__main__":
    unittest.main()
