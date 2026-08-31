from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_line_ownership as analysis


class LineOwnershipAnalyzerTest(unittest.TestCase):
    def test_frozen_signal_gate(self) -> None:
        passed = analysis.signal_gate(
            {
                "median_issue_delta_cycles_per_group": 0.0,
                "median_wait_delta_cycles_per_group": -1.0,
                "negative_wait_repeats": 8,
            }
        )
        self.assertTrue(all(passed.values()))
        failed = analysis.signal_gate(
            {
                "median_issue_delta_cycles_per_group": 0.0,
                "median_wait_delta_cycles_per_group": -0.999,
                "negative_wait_repeats": 10,
            }
        )
        self.assertFalse(failed["wait_median_at_most_negative_one"])

    def test_structural_no_go_precedes_direct_signal(self) -> None:
        status, return_code = analysis.choose_status(
            errors=[],
            missing=[],
            structural=True,
            smoke_complete=True,
            smoke_passed=False,
            smoke_only=True,
            identification=None,
            holdout=None,
        )
        self.assertEqual(status, "complete_line_ownership_structural_no_go")
        self.assertEqual(return_code, 0)

    def test_identity_and_naming_contract(self) -> None:
        self.assertEqual(
            analysis.endpoint_name("early_fanout", "first_line_early"),
            "async_source_line_ownership_early_fanout_first_line_early_g64",
        )
        self.assertEqual(analysis.EXPECTED_VALUES["shared_wavefronts"], 2048)
        with self.assertRaises(analysis.AnalysisError):
            analysis.require_cross_endpoint_tracer_identity(
                [{"tracer_sha256": "a"}, {"tracer_sha256": "b"}]
            )


if __name__ == "__main__":
    unittest.main()
