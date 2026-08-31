from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_line_ownership as analysis
import capture_line_ownership_execution as execution


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

    def test_full_campaign_requires_identity_matched_smoke_pass(self) -> None:
        binary = {"path": "/probe", "sha256": "abc", "size_bytes": 1}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            report_path = root / "smoke.json"
            report = {
                "schema_version": 1,
                "assay": "sm86_async_source_line_ownership",
                "mode": "smoke_only",
                "status": "complete_line_ownership_structural_no_go",
                "root": str(root),
                "binary": binary,
                "timing_promotion_authorized": False,
                "typed_visibility_tail_required": True,
                "missing_artifact_count": 0,
                "validation_error_count": 0,
                "smoke_gates": {"gate": False},
                "criteria": {"smoke_passed": False},
            }
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(execution.CaptureError):
                execution.require_full_campaign_authorization(
                    root, binary, report_path
                )

            report.update(
                {
                    "status": "source_line_ownership_smoke_passed",
                    "smoke_gates": {"gate": True},
                    "criteria": {"smoke_passed": True},
                }
            )
            report_path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(
                execution.require_full_campaign_authorization(
                    root, binary, report_path
                )["status"],
                "source_line_ownership_smoke_passed",
            )

    def test_direct_artifact_is_required_before_structural_no_go(self) -> None:
        binary = {"path": "/probe", "sha256": "abc", "size_bytes": 1}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                analysis.validate_direct_artifact(
                    root, "first_line_early", 1, binary
                )

            directory = (
                root
                / "direct"
                / analysis.paired_cell_name(
                    "first_line_early", analysis.DIRECT_GROUPS
                )
                / "repeat-1"
            )
            directory.mkdir(parents=True)
            for name in (
                "result.json",
                "command.json",
                "direct.run.json",
                "stderr.txt",
            ):
                (directory / name).touch()
            with mock.patch.object(
                analysis.execution_producer,
                "validate_direct",
                side_effect=analysis.execution_producer.CaptureError("corrupt"),
            ):
                with self.assertRaises(analysis.AnalysisError):
                    analysis.validate_direct_artifact(
                        root, "first_line_early", 1, binary
                    )


if __name__ == "__main__":
    unittest.main()
