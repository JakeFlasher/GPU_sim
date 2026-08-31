from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_line_ownership as analysis
import capture_line_ownership_execution as execution
import capture_multiline_execution as shared_execution


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

    def test_structural_decision_reports_only_observed_reason(self) -> None:
        static_decision = analysis.decision_for_status(
            "complete_line_ownership_structural_no_go",
            static={"assay_valid": False},
        )
        self.assertIn("static", static_decision.lower())
        self.assertNotIn("1,280", static_decision)

        issue_decision = analysis.decision_for_status(
            "complete_line_ownership_structural_no_go",
            static={"assay_valid": True},
            smoke={"direct_issue_exact_zero": False},
        )
        self.assertIn("issue", issue_decision.lower())
        self.assertNotIn("1,280", issue_decision)

        ncu_decision = analysis.decision_for_status(
            "complete_line_ownership_structural_no_go",
            static={"assay_valid": True},
            smoke={"ncu_contract_exact": False},
            bundles=(
                {
                    "values": {
                        "shared_wavefronts": {
                            "early_fanout": 1280,
                            "late_fanout": 1280,
                        }
                    }
                },
            ),
        )
        self.assertIn("1,280", ncu_decision)

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
                "validate_direct_provenance",
                return_value={"result": {"sha256": "x"}},
            ), mock.patch.object(
                analysis, "direct_non_timing_payload", return_value={}
            ), mock.patch.object(
                analysis,
                "direct_issue_cell",
                side_effect=AssertionError("issue fields must stay unopened"),
            ), mock.patch.object(
                analysis,
                "direct_wait_cell",
                side_effect=AssertionError("wait fields must stay unopened"),
            ):
                validated = analysis.validate_direct_artifact(
                    root, "first_line_early", 1, binary
                )
                self.assertTrue(validated["process_completed"])

            with mock.patch.object(
                analysis.execution_producer,
                "validate_direct_provenance",
                side_effect=analysis.execution_producer.CaptureError("corrupt"),
            ):
                with self.assertRaises(analysis.AnalysisError):
                    analysis.validate_direct_artifact(
                        root, "first_line_early", 1, binary
                    )

    def test_trace_payload_manifest_detects_payload_and_sidecar_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "threadblocks/device_0/stream_0/kernel_0").mkdir(
                parents=True
            )
            (root / "extra_info").mkdir()
            (root / "dynamic_trace.pb").write_bytes(b"root")
            (root / "stats.csv").write_text("header\n", encoding="utf-8")
            (root / "stdout.txt").write_text("{}\n", encoding="utf-8")
            payload = root / "threadblocks/device_0/stream_0/kernel_0/block_0.pb"
            sidecar = root / "extra_info/allocations.json"
            payload.write_bytes(b"payload")
            sidecar.write_text("[]\n", encoding="utf-8")
            execution.write_trace_payload_manifest(root)
            execution.validate_trace_payload_manifest(root)

            payload.write_bytes(b"payload\x80\x01\x00")
            with self.assertRaises(execution.CaptureError):
                execution.validate_trace_payload_manifest(root)
            execution.write_trace_payload_manifest(root)
            sidecar.write_text("[{}]\n", encoding="utf-8")
            with self.assertRaises(execution.CaptureError):
                execution.validate_trace_payload_manifest(root)

    def test_paired_deltas_are_derived_from_arm_counters(self) -> None:
        binary = {"path": "/probe", "sha256": "abc", "size_bytes": 1}
        issue_payload = {
            "early_fanout": {"issue_cycles": 100},
            "late_fanout": {"issue_cycles": 100},
            "paired_issue_delta_cycles_per_group": -1.0,
        }
        with mock.patch.object(
            analysis, "validate_direct_artifact", return_value={}
        ), mock.patch.object(
            analysis, "direct_non_timing_payload", return_value=issue_payload
        ):
            with self.assertRaises(analysis.AnalysisError):
                analysis.direct_issue_cell(
                    Path("/unused"), "first_line_early", (1,), binary
                )

        wait_payload = {
            "early_fanout": {"wait_cycles": 200},
            "late_fanout": {"wait_cycles": 100},
            "paired_wait_delta_cycles_per_group": -1.0,
        }
        with mock.patch.object(
            analysis, "validate_direct_artifact", return_value={}
        ), mock.patch.object(
            analysis, "direct_non_timing_payload", return_value=wait_payload
        ):
            with self.assertRaises(analysis.AnalysisError):
                analysis.direct_wait_cell(
                    Path("/unused"), "first_line_early", (1,), binary
                )

    def test_v1_target_validation_retains_full_timing_behavior(self) -> None:
        ncu = analysis.ncu_producer
        context = SimpleNamespace(device=dict(analysis.EXPECTED_DEVICE))
        cell = ncu.Cell("first_line_early")
        payload = ncu.expected_target_output(cell, 1, context)

        def arm() -> dict[str, object]:
            return {
                "issue_cycles": 64,
                "wait_cycles": 128,
                "total_cycles": 192,
                "issue_cycles_per_group": 1.0,
                "wait_cycles_per_group": 2.0,
                "total_cycles_per_group": 3.0,
                "token": 1,
                "mismatches": 0,
                "sum_consistent": True,
                "correct": True,
            }

        payload["early_fanout"] = arm()
        payload["late_fanout"] = arm()
        payload["paired_issue_delta_cycles_per_group"] = 0.0
        payload["paired_wait_delta_cycles_per_group"] = 0.0
        observed = ncu.validate_target_output(json.dumps(payload), cell, 1, context)
        self.assertIn("issue_cycles", observed["early_fanout"])
        analysis.validate_target_payload(
            observed,
            orientation="first_line_early",
            groups=analysis.TRACE_GROUPS,
            mode="pair",
            order="ab",
        )
        observed["early_fanout"]["issue_cycles"] = "poison"
        with self.assertRaises(analysis.AnalysisError):
            analysis.validate_target_payload(
                observed,
                orientation="first_line_early",
                groups=analysis.TRACE_GROUPS,
                mode="pair",
                order="ab",
            )

    def test_shared_static_makefile_selection_is_synthetic(self) -> None:
        with mock.patch.multiple(
            shared_execution,
            HERE=Path("/probe"),
            MAKEFILE=Path("/probe/Makefile"),
            BINARY=Path("/probe/base_probe"),
            NVCC=Path("/cuda/nvcc"),
        ):
            command = shared_execution.static_build_command()
            self.assertNotIn("-f", command)
            self.assertEqual(command[-1], "base_probe")
        with mock.patch.multiple(
            shared_execution,
            HERE=Path("/probe"),
            MAKEFILE=Path("/probe/Makefile.special"),
            BINARY=Path("/probe/special_probe"),
            NVCC=Path("/cuda/nvcc"),
        ):
            command = shared_execution.static_build_command()
            self.assertIn("-f", command)
            self.assertIn("/probe/Makefile.special", command)
            self.assertEqual(command[-1], "special_probe")


if __name__ == "__main__":
    unittest.main()
