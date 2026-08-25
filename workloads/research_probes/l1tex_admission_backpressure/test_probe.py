#!/usr/bin/env python3
"""Focused regression tests for the admission/backpressure campaign pipeline."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import analyze
import analyze_feedback_service_sweep as service_sweep
import campaign


class CampaignBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cell = campaign.Cell(
            "burst_depth", "store", "line_unique", 84, 8, 4, 0, "all"
        )

    def measurement(self) -> dict[str, object]:
        return {
            "tail": self.cell.tail,
            "layout": self.cell.layout,
            "blocks": self.cell.blocks,
            "warps": self.cell.warps,
            "depth": self.cell.depth,
            "active_lanes": self.cell.active_lanes,
            "gap_steps": self.cell.gap_steps,
            "pattern": self.cell.pattern,
            "selected_warps_per_block": self.cell.selected_warps_per_block,
            "memory_instructions": self.cell.memory_instructions,
            "l2_eviction_bytes": 32 * 1024 * 1024,
            "trials": 1,
            "correct": True,
        }

    def test_measurement_binding_rejects_wrong_cell(self) -> None:
        measurement = self.measurement()
        self.assertTrue(
            campaign.measurement_matches_cell(measurement, self.cell, trials=1)
        )
        measurement["depth"] = 8
        self.assertFalse(
            campaign.measurement_matches_cell(measurement, self.cell, trials=1)
        )

    def test_feedback_singletons_exactly_match_schema9_candidates(self) -> None:
        _, candidates = campaign.validated_admission_documents()
        self.assertEqual(set(candidates), set(range(1, 17)))
        for candidate_id, path in campaign.FEEDBACK_CANDIDATE_FILES.items():
            feedback = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(feedback["candidates"], [candidates[candidate_id]])

    def test_feedback_service_sweep_varies_only_rate_and_width(self) -> None:
        sweep = campaign.validated_feedback_service_sweep()
        self.assertEqual(
            set(sweep),
            set(range(101, 117)) | set(range(121, 141)),
        )
        self.assertEqual(
            {
                (
                    candidate["l2_miss_sector_cycles"],
                    candidate["l2_miss_service_width_sectors"],
                )
                for _, candidate in (sweep[candidate_id] for candidate_id in range(101, 117))
            },
            {(rate, width) for rate in (8, 10, 12, 14) for width in (1, 2, 4, 8)},
        )
        self.assertEqual(
            {
                (
                    candidate["l2_miss_sector_cycles"],
                    candidate["l2_miss_service_width_sectors"],
                )
                for _, candidate in (sweep[candidate_id] for candidate_id in range(121, 137))
            },
            {(rate, width) for rate in (12, 16, 20, 24) for width in (1, 2, 4, 8)},
        )
        self.assertEqual(
            {
                (
                    candidate["l2_miss_sector_cycles"],
                    candidate["l2_miss_service_width_sectors"],
                )
                for _, candidate in (sweep[candidate_id] for candidate_id in range(137, 141))
            },
            {(8, width) for width in (1, 2, 4, 8)},
        )
        self.assertEqual(campaign.selected_feedback_service_candidates("101,136"), [101, 136])

    def test_write_completion_candidates_change_only_membar_wait(self) -> None:
        candidates = campaign.validated_write_completion_feedback()
        self.assertEqual(
            set(candidates),
            {201, 202, 203, 204, 205, 206, 207, 208, 209, 210},
        )
        self.assertEqual(candidates[201][1]["operation"], "store")
        self.assertEqual(candidates[202][1]["operation"], "reduction")
        self.assertTrue(
            all(
                candidate["membar_waits_for_service_completion"] is True
                for _, candidate in candidates.values()
            )
        )
        self.assertEqual(
            [candidates[candidate_id][1]["write_completion_latency_cycles"] for candidate_id in range(203, 207)],
            [512, 1024, 1536, 2048],
        )
        self.assertEqual(campaign.selected_write_completion_candidates("202"), [202])

    def test_red_rmw_completion_grid_varies_basis_rate_and_lanes(self) -> None:
        candidates = campaign.validated_red_rmw_completion_sweep()
        self.assertEqual(set(candidates), set(range(211, 229)))
        request = [candidates[candidate_id][1] for candidate_id in range(211, 220)]
        region = [candidates[candidate_id][1] for candidate_id in range(220, 229)]
        self.assertEqual(
            {
                (item["write_completion_request_cycles"], item["write_completion_parallel_lanes"])
                for item in request
            },
            {(rate, lanes) for rate in (2, 4, 8) for lanes in (1, 2, 4)},
        )
        self.assertEqual(
            {
                (item["write_completion_region_cycles"], item["write_completion_parallel_lanes"])
                for item in region
            },
            {(rate, lanes) for rate in (4, 8, 16) for lanes in (1, 2, 4)},
        )

    def test_resume_requires_exact_command_and_fresh_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dependency = root / "binary"
            output = root / "result.json"
            run = root / "run.json"
            dependency.write_bytes(b"input")
            output.write_text("{}\n", encoding="utf-8")
            command = ["probe", "store"]
            run.write_text(
                json.dumps({"return_code": 0, "command": command}), encoding="utf-8"
            )
            os.utime(output, ns=(1, 1))
            os.utime(dependency, ns=(2, 2))
            self.assertTrue(campaign.captured_run_matches(root, command))
            self.assertFalse(campaign.outputs_are_fresh((output,), (dependency,)))
            os.utime(output, ns=(2, 2))
            self.assertFalse(campaign.outputs_are_fresh((output,), (dependency,)))
            os.utime(output, ns=(3, 3))
            self.assertTrue(campaign.outputs_are_fresh((output,), (dependency,)))
            self.assertFalse(campaign.captured_run_matches(root, ["probe", "red"]))


class AnalysisTests(unittest.TestCase):
    def test_service_sweep_pareto_uses_issue_and_completion(self) -> None:
        scores = {
            1: {"issue": {"mae": 10.0}, "service_completion": {"mae": 20.0}},
            2: {"issue": {"mae": 9.0}, "service_completion": {"mae": 21.0}},
            3: {"issue": {"mae": 11.0}, "service_completion": {"mae": 22.0}},
        }
        self.assertEqual(service_sweep.pareto_ids(scores), [1, 2])

    def test_feedback_partition_is_strict_five_twenty(self) -> None:
        rows = [
            {"stage": "burst_depth", "depth": depth, "cell": f"cal-{depth}"}
            for depth in (1, 2, 4, 8, 16)
        ]
        rows.extend(
            {"stage": "held_out", "depth": index, "cell": f"hold-{index}"}
            for index in range(20)
        )
        calibration, held_out = analyze.feedback_partitions(rows)
        self.assertEqual(len(calibration), 5)
        self.assertEqual(len(held_out), 20)
        self.assertTrue(all(row["cell"].startswith("cal-") for row in calibration))

    def test_operation_observer_uses_same_candidate_baseline(self) -> None:
        rows = [
            {
                "feedback_hit": 1,
                "feedback_miss": 9,
                "schema9_hit": 1,
                "schema9_miss": 9,
                "aggregate_hit": 0,
                "aggregate_miss": 10,
            }
        ]
        self.assertEqual(
            analyze.count_changed_outcomes(
                rows,
                "feedback_hit",
                "feedback_miss",
                "schema9_hit",
                "schema9_miss",
            ),
            0,
        )
        self.assertEqual(
            analyze.count_changed_outcomes(
                rows,
                "feedback_hit",
                "feedback_miss",
                "aggregate_hit",
                "aggregate_miss",
            ),
            1,
        )

    def test_observation_envelope_bounds_model_error(self) -> None:
        rows = [
            {"cell": "inside", "prediction": 5, "low": 4, "high": 6},
            {"cell": "outside", "prediction": 10, "low": 3, "high": 7},
        ]
        result = analyze.score_observation_envelope(
            rows, "prediction", "low", "high"
        )
        self.assertEqual(result["mae_lower_bound"], 1.5)
        self.assertEqual(result["mae_upper_bound"], 4.0)


if __name__ == "__main__":
    unittest.main()
