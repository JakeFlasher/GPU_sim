#!/usr/bin/env python3
"""Focused tests for the retained LDGSTS front-end evaluator."""

from __future__ import annotations

import copy
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate


def fixture_inputs() -> dict[str, dict[str, float]]:
    live_v4: dict[str, float] = {}
    for width in (2, 4, 8):
        gaps = (0, 64) if width == 4 else (0,)
        localities = ("l1", "l2", "dram") if width == 4 else ("l1", "dram")
        for gap in gaps:
            live_v4[f"w{width}_c00_na_g{gap}"] = 1.0 + gap
            live_v4[f"w{width}_c01_na_g{gap}"] = 2.0 + gap
            for locality_index, locality in enumerate(localities, 1):
                wait = 10.0 * locality_index + gap
                live_v4[f"w{width}_c10_{locality}_g{gap}"] = wait
                live_v4[f"w{width}_c11_{locality}_g{gap}"] = wait + 3.0
                if width in (2, 8):
                    live_v4[f"w{width}_c12_{locality}_g{gap}"] = wait + 2.0

    live_coupled: dict[str, float] = {}
    for width in (2, 4, 8):
        width_offset = float(width)
        for protocol, protocol_offset in (("c10", 0.0), ("c11", 2.0)):
            for locality, route in (("l1", 10.0), ("l2", 20.0), ("dram", 30.0)):
                live_coupled[f"w{width}_{protocol}_{locality}_g0"] = (
                    width_offset + protocol_offset + route
                )

    live_guard = {f"w{width}": 100.0 + width for width in range(1, 9)}
    return {
        "live_v4": live_v4,
        "baseline_v4": {key: value + 10.0 for key, value in live_v4.items()},
        "candidate_v4": {key: value + 1.0 for key, value in live_v4.items()},
        "live_coupled": live_coupled,
        "baseline_coupled": {
            key: value + 10.0 for key, value in live_coupled.items()
        },
        "candidate_coupled": {
            key: value + 1.0 for key, value in live_coupled.items()
        },
        "live_guard": live_guard,
        "baseline_guard": {key: value + 2.0 for key, value in live_guard.items()},
        "candidate_guard": {key: value + 1.0 for key, value in live_guard.items()},
    }


def replay_documents() -> tuple[dict[str, object], dict[str, object]]:
    launch = {
        "device_id": 0,
        "stream_id": 0,
        "kernel_id": 1,
        "function_unique_id": 7,
        "kernel_name": "fixture_kernel",
        "grid": [1, 1, 1],
        "block": [32, 1, 1],
        "cluster": [1, 1, 1],
        "coverage": {"kind": "full", "blocks_observed": 1, "population_blocks": 1},
        "counters": {
            "instructions_issued": 4,
            "records_seen": 4,
            "l1": {"hits": 1},
            "l2": {"hits": 1, "misses": 0},
            "dram": {"read_sectors": 0},
            "unmodeled_tails": {"coherence": True},
        },
        "timing": {"diagnostic_last_issue_cycles": 10},
    }
    baseline = {"architecture": "sm_86", "memory_mode": "l1-l2", "launches": [launch]}
    candidate = {
        "schema_version": 1,
        "architecture": "sm_86",
        "memory_mode": "l1-l2",
        "cache_scope": "launch",
        "ldgsts_front_end": "ampere_queue_v1",
        "launches": [copy.deepcopy(launch)],
    }
    candidate["launches"][0]["counters"]["l2_by_source"] = {
        "l1_fill": {"sectors": 1, "hits": 1},
        "bypass": {"sectors": 0, "hits": 0},
        "untagged": {"sectors": 0, "hits": 0},
        "store": {"sectors": 0, "hits": 0},
    }
    return baseline, candidate


class EvaluatorTests(unittest.TestCase):
    def test_frozen_reductions_can_retain_candidate(self) -> None:
        result = evaluate.reduce_core(**fixture_inputs())

        self.assertEqual(result["verdict"], "retained_opt_in")
        self.assertEqual(result["failed_gates"], [])
        self.assertEqual(result["transfer"]["cell_count"], 30)
        self.assertEqual(result["warmed_source_guard"]["cell_count"], 8)
        self.assertEqual(result["route_order"]["group_count"], 6)
        self.assertEqual(result["v4_corrected_interaction_signs"]["cell_count"], 10)
        self.assertEqual(result["coupled_contrast_signs"]["cell_count"], 9)

    def test_missing_and_extra_cell_keys_fail_closed(self) -> None:
        missing = fixture_inputs()
        missing["candidate_coupled"].pop("w2_c10_l1_g0")
        with self.assertRaisesRegex(evaluate.EvaluationError, "key mismatch"):
            evaluate.reduce_core(**missing)

        extra = fixture_inputs()
        extra["candidate_v4"]["unexpected"] = 1.0
        with self.assertRaisesRegex(evaluate.EvaluationError, r"extra=\['unexpected'\]"):
            evaluate.reduce_core(**extra)

    def test_failed_gate_is_named_without_hiding_other_results(self) -> None:
        inputs = fixture_inputs()
        inputs["candidate_v4"]["w4_c11_l1_g0"] = (
            inputs["candidate_v4"]["w4_c10_l1_g0"] - 2.0
        )

        result = evaluate.reduce_core(**inputs)

        self.assertEqual(result["verdict"], "not_retained")
        self.assertEqual(
            result["failed_gates"], ["v4_corrected_interaction_signs_match"]
        )
        self.assertEqual(result["transfer"]["cell_count"], 30)

    def test_candidate_accounting_mismatch_becomes_a_named_gate(self) -> None:
        spec = evaluate.ReplaySpec(
            campaign="fixture",
            key="cell",
            trace=Path("trace"),
            control=Path("control"),
            baseline=Path("baseline"),
        )
        baseline, candidate = replay_documents()
        _, failures = evaluate.validate_candidate(candidate, baseline, spec)
        self.assertEqual(failures, [])

        candidate["launches"][0]["counters"]["l1"]["hits"] = 2

        _, failures = evaluate.validate_candidate(candidate, baseline, spec)
        self.assertEqual(failures, ["launch_0:traffic_or_counter_drift"])

        inputs = fixture_inputs()
        inputs["accounting_failures"] = {"guard:w1": failures}
        result = evaluate.reduce_core(**inputs)
        self.assertEqual(result["verdict"], "not_retained")
        self.assertEqual(
            result["failed_gates"],
            ["traffic_and_instruction_accounting_exact"],
        )
        self.assertEqual(
            result["accounting"]["failed_cells"],
            [
                {
                    "cell": "guard:w1",
                    "failures": ["launch_0:traffic_or_counter_drift"],
                }
            ],
        )

        cardinality = copy.deepcopy(candidate)
        cardinality["launches"][0]["counters"]["l1"]["hits"] = 1
        cardinality["launches"][0]["counters"]["records_seen"] = 3
        _, failures = evaluate.validate_candidate(cardinality, baseline, spec)
        self.assertEqual(
            failures,
            ["launch_0:instruction_cardinality", "launch_0:traffic_or_counter_drift"],
        )

    def test_launch_geometry_kernel_and_cache_scope_fail_closed(self) -> None:
        baseline, candidate = replay_documents()
        spec = evaluate.ReplaySpec("fixture", "cell", Path("t"), Path("c"), Path("b"))
        changed = copy.deepcopy(candidate)
        changed["launches"][0]["grid"] = [2, 1, 1]
        with self.assertRaisesRegex(evaluate.EvaluationError, "identity mismatch"):
            evaluate.validate_candidate(changed, baseline, spec)

        selected = copy.deepcopy(baseline)
        selected["launches"][0]["kernel_id"] = 2
        with self.assertRaisesRegex(evaluate.EvaluationError, "baseline target kernel"):
            evaluate.validate_baseline(
                selected,
                evaluate.ReplaySpec("fixture", "cell", Path("t"), Path("c"), Path("b"), kernel_id=1),
            )

        retained_l2 = copy.deepcopy(baseline)
        retained_l2["cache_session"] = {"scope": "launch", "completed_launches": 1}
        with self.assertRaisesRegex(evaluate.EvaluationError, "cache-session scope"):
            evaluate.validate_baseline(
                retained_l2,
                evaluate.ReplaySpec(
                    "fixture", "cell", Path("t"), Path("c"), Path("b"), cache_scope="stream-l2"
                ),
            )

    def test_malformed_numeric_accounting_and_timing_fail_closed(self) -> None:
        spec = evaluate.ReplaySpec("fixture", "cell", Path("t"), Path("c"), Path("b"))
        for value in (True, -1):
            baseline, _ = replay_documents()
            baseline["launches"][0]["counters"]["instructions_issued"] = value
            baseline["launches"][0]["counters"]["records_seen"] = value
            with self.assertRaises(evaluate.EvaluationError):
                evaluate.validate_baseline(baseline, spec)

        baseline, candidate = replay_documents()
        candidate["launches"][0]["timing"] = None
        with self.assertRaisesRegex(evaluate.EvaluationError, "malformed timing"):
            evaluate.validate_candidate(candidate, baseline, spec)

    def test_reducer_boundaries_and_composite_provenance(self) -> None:
        inputs = fixture_inputs()
        equal_guard = copy.deepcopy(inputs)
        equal_guard["candidate_guard"] = copy.deepcopy(equal_guard["baseline_guard"])
        self.assertEqual(evaluate.reduce_core(**equal_guard)["verdict"], "retained_opt_in")

        equal_transfer = copy.deepcopy(inputs)
        for key in evaluate.V4_TRANSFER_KEYS:
            equal_transfer["candidate_v4"][key] = equal_transfer["baseline_v4"][key]
        equal_transfer["candidate_coupled"] = copy.deepcopy(equal_transfer["baseline_coupled"])
        result = evaluate.reduce_core(**equal_transfer)
        self.assertIn("transfer_mae_strictly_below_baseline", result["failed_gates"])

        overlap = "w2_c10_l1_g0"
        rows = [row for row in evaluate.reduce_core(**inputs)["transfer"]["rows"] if row["key"] == overlap]
        self.assertEqual({row["campaign"] for row in rows}, {"v4", "coupled"})
        self.assertEqual(len(rows), 2)

        with self.assertRaisesRegex(evaluate.EvaluationError, "non-probe cell"):
            evaluate.reduce_core(
                **inputs, accounting_failures={"production:attention": ["drift"]}
            )

    def test_duplicate_replay_spec_fails_before_execution(self) -> None:
        spec = evaluate.ReplaySpec("fixture", "cell", Path("t"), Path("c"), Path("b"))
        with self.assertRaisesRegex(evaluate.EvaluationError, "duplicate campaign cells"):
            evaluate.run_campaign(Path("sim-run"), [spec, spec], 1)

    def test_extreme_production_metrics_fail_cleanly(self) -> None:
        observed = {key: 1.0 for key in evaluate.PRODUCTION_KEYS}
        predicted = {key: math.nextafter(float("inf"), 0.0) for key in evaluate.PRODUCTION_KEYS}
        with self.assertRaises(evaluate.EvaluationError):
            evaluate.score_production(observed, predicted)

    def test_production_labels_cannot_change_verdict(self) -> None:
        core = evaluate.reduce_core(**fixture_inputs())
        live = {key: 20_000.0 + index for index, key in enumerate(sorted(evaluate.PRODUCTION_KEYS))}
        baseline = {key: value - 2_000.0 for key, value in live.items()}
        candidate = {key: value - 1_000.0 for key, value in live.items()}

        first = evaluate.attach_production(
            core, live=live, baseline=baseline, candidate=candidate
        )
        shifted_live = {key: value * 3.0 for key, value in live.items()}
        second = evaluate.attach_production(
            core, live=shifted_live, baseline=baseline, candidate=candidate
        )

        self.assertTrue(first["production"]["verdict_inert"])
        self.assertEqual(first["verdict"], second["verdict"])
        self.assertEqual(first["failed_gates"], second["failed_gates"])
        self.assertEqual(first["gates"], second["gates"])
        self.assertNotEqual(
            first["production"]["candidate"]["mape"],
            second["production"]["candidate"]["mape"],
        )

    def test_all_replay_commands_use_only_the_explicit_candidate(self) -> None:
        binary = Path("/tmp/sim-run")
        specs = evaluate.replay_specs()
        self.assertEqual(len(specs["v4"]), 24)
        self.assertEqual(len(specs["coupled"]), 18)
        self.assertEqual(len(specs["guard"]), 8)
        self.assertEqual(len(specs["production"]), 6)
        self.assertTrue(all(spec.kernel_id == 1 for spec in specs["guard"]))
        for campaign_specs in specs.values():
            for spec in campaign_specs:
                command = evaluate.replay_command(binary, spec)
                self.assertEqual(Path(command[0]).name, "sim-run")
                self.assertEqual(command.count("--ldgsts-front-end"), 1)
                self.assertEqual(
                    command[command.index("--ldgsts-front-end") + 1],
                    "ampere-queue-v1",
                )
                self.assertEqual(command[command.index("--mode") + 1], "l1-l2")
                self.assertNotIn("ncu", command)
                self.assertNotIn("nvbit", command)


if __name__ == "__main__":
    unittest.main()
