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

    def test_frozen_spec_is_valid(self) -> None:
        subject.validate_spec(self.spec, self.metrics, self.profile)
        subject.validate_accuracy_gate(self.gate)

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


if __name__ == "__main__":
    unittest.main()
