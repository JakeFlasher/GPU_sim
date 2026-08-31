from __future__ import annotations

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_multiline_response as analysis


class MultilineResponseReducerTest(unittest.TestCase):
    def test_frozen_destination_wavefront_geometry_rejects_current_32_per_copy(self) -> None:
        expected = analysis.replay_geometry_expectations(64)
        self.assertEqual(expected["destination_wavefronts_observed"], 1280)
        self.assertNotEqual(expected["destination_wavefronts_observed"], 64 * 32)

    def test_smoke_state_machine_has_typed_success_no_go_and_error_outcomes(self) -> None:
        passed = analysis.select_outcome(
            errors=[],
            missing=[],
            structural_no_go=False,
            smoke_complete=True,
            smoke_passed=True,
            smoke_only=True,
            full_admitted=False,
            score=None,
        )
        self.assertEqual(passed[::2], ("response_envelope_smoke_passed", 0))

        no_go = analysis.select_outcome(
            errors=[],
            missing=[],
            structural_no_go=True,
            smoke_complete=True,
            smoke_passed=False,
            smoke_only=True,
            full_admitted=False,
            score=None,
        )
        self.assertEqual(
            no_go[::2], ("complete_response_envelope_structural_no_go", 0)
        )

        incomplete = analysis.select_outcome(
            errors=[],
            missing=["missing"],
            structural_no_go=False,
            smoke_complete=False,
            smoke_passed=False,
            smoke_only=True,
            full_admitted=False,
            score=None,
        )
        self.assertEqual(incomplete[2], 2)

        invalid = analysis.select_outcome(
            errors=["malformed"],
            missing=[],
            structural_no_go=False,
            smoke_complete=False,
            smoke_passed=False,
            smoke_only=True,
            full_admitted=False,
            score=None,
        )
        self.assertEqual(invalid[2], 1)

    def test_cross_endpoint_tracer_mismatch_is_an_artifact_error(self) -> None:
        self.assertEqual(
            analysis.retained_tracer_identity(
                {"tracer_sha256": "a"}, {"tracer": {"sha256": "a"}}, "cell"
            ),
            "a",
        )
        with self.assertRaises(analysis.AnalysisError):
            analysis.retained_tracer_identity(
                {"tracer_sha256": "a"}, {"tracer": {"sha256": "b"}}, "cell"
            )
        with self.assertRaises(analysis.AnalysisError):
            analysis.require_cross_endpoint_tracer_identity(
                [{"tracer_sha256": "a"}, {"tracer_sha256": "b"}]
            )
        self.assertEqual(
            analysis.require_cross_endpoint_tracer_identity(
                [{"tracer_sha256": "a"}, {"tracer_sha256": "a"}]
            ),
            "a",
        )

    def test_tracked_aggregate_is_order_stable_and_cli_exposes_smoke_mode(self) -> None:
        base = {
            "assay": "assay",
            "mode": "smoke_only",
            "status": "response_envelope_smoke_passed",
            "decision": "done",
            "counts": {"b": 2, "a": 1},
            "static": None,
            "smoke_scientific_gates": {"z": True, "a": True},
            "full_replay_gates": None,
            "score": None,
            "criteria": {"z": None, "a": True},
            "missing_artifacts": [],
            "validation_errors": [],
            "simulator_cells": [
                {
                    "variant": "span_expensive",
                    "depth": 1,
                    "lead_samples": 16,
                    "geometry_expected": {"x": 1},
                    "geometry_observed": {"x": 1},
                    "geometry_gates": {"x": True},
                    "work_gates": {"y": True},
                    "cohort_response_envelope_cycles": 1,
                    "cohort_topology_prearrival_hidden_cycles": 0,
                    "cohort_topology_stream_hidden_cycles": 1,
                    "cohort_topology_residual_exposure_cycles": 0,
                },
                {
                    "variant": "span_cheap",
                    "depth": 1,
                    "lead_samples": 16,
                    "geometry_expected": {"x": 1},
                    "geometry_observed": {"x": 1},
                    "geometry_gates": {"x": True},
                    "work_gates": {"y": True},
                    "cohort_response_envelope_cycles": 1,
                    "cohort_topology_prearrival_hidden_cycles": 0,
                    "cohort_topology_stream_hidden_cycles": 0,
                    "cohort_topology_residual_exposure_cycles": 1,
                },
            ],
        }
        reversed_cells = dict(base)
        reversed_cells["simulator_cells"] = list(reversed(base["simulator_cells"]))
        left = json.dumps(analysis.deterministic_aggregate(base), sort_keys=True)
        right = json.dumps(
            analysis.deterministic_aggregate(reversed_cells), sort_keys=True
        )
        self.assertEqual(left, right)

        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as stopped:
            analysis.main(["--help"])
        self.assertEqual(stopped.exception.code, 0)
        self.assertIn("--smoke-only", output.getvalue())
        self.assertIn("--aggregate-output", output.getvalue())


if __name__ == "__main__":
    unittest.main()
