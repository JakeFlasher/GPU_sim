"""Coverage for lossless boundary decoding and comparison failure handling."""

import copy
import json
import os
import pathlib
import shlex
import subprocess
import sys
import tempfile
import textwrap
import unittest

from tools.semantic_events import sm86_reference_compare
from tools.semantic_events.sm86_reference_compare import (
    compare_boundary_traces,
    summarize_native_boundary_refinement,
)


class BoundaryTraceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name)

    def write(self, mode, events, *, encoding="full", footer=True):
        path = self.root / f"{mode}.jsonl"
        rows = [{"kind": "header", "mode": mode, "state_projection_complete": False,
                 "event_encoding": encoding}]
        rows.extend(events)
        if footer:
            rows.append({"kind": "footer", "records_before_footer": len(rows),
                         "queue_events_popped": len(events)})
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def test_delta_matches_full_stream_and_detects_an_intermediate_difference(self):
        first = {"kind": "event", "time": 0, "state": [{"pending": [1, 2]}, None]}
        second = {"kind": "event", "time": 3, "state": [{"pending": [2]}, {"done": True}]}
        delta = {"kind": "event_delta", "base_record": 1, "replacements": [
            {"path": ["time"], "value": 3},
            {"path": ["state", 0, "pending"], "value": [2]},
            {"path": ["state", 1], "value": {"done": True}},
        ]}
        event = self.write("event", [first, delta], encoding="full_then_delta_v1")
        dense = self.write("dense", [first, second])
        self.assertEqual(compare_boundary_traces(event, dense), {
            "boundary_trace_equal": True, "boundary_event_records": 2,
            "first_different_boundary_record": None,
        })
        wrong_first = copy.deepcopy(first)
        wrong_first["state"][0]["pending"] = [9]
        dense = self.write("dense", [wrong_first, second])
        report = compare_boundary_traces(event, dense)
        self.assertFalse(report["boundary_trace_equal"])
        self.assertEqual(report["first_different_boundary_record"], 1)

    def test_truncated_or_wrong_base_trace_is_rejected(self):
        first = {"kind": "event", "time": 0}
        dense = self.write("dense", [first])
        event = self.write("event", [first], footer=False)
        with self.assertRaisesRegex(RuntimeError, "missing its footer"):
            compare_boundary_traces(event, dense)
        event = self.write("event", [first, {"kind": "event_delta", "base_record": 0,
                                           "replacements": []}], encoding="full_then_delta_v1")
        with self.assertRaisesRegex(RuntimeError, "no matching base"):
            compare_boundary_traces(event, dense)

    def test_unknown_record_and_wrong_footer_are_rejected(self):
        first = {"kind": "event", "time": 0}
        dense = self.write("dense", [first])
        event = self.write("event", [{"kind": "silently_dropped_before"}])
        with self.assertRaisesRegex(RuntimeError, "unexpected reference record"):
            compare_boundary_traces(event, dense)
        event = self.write("event", [first])
        event.write_text(event.read_text().replace('"queue_events_popped": 1', '"queue_events_popped": 2'))
        with self.assertRaisesRegex(RuntimeError, "footer count mismatch"):
            compare_boundary_traces(event, dense)

    def test_v2_array_splice_reconstructs_shifted_state(self):
        first = {"kind": "event", "state": {"unknown": [1, 2, 3, 4]}}
        second = {"kind": "event", "state": {"unknown": [1, 8, 9, 4, 5]}}
        delta = {
            "kind": "event_delta",
            "base_record": 1,
            "operations": [{
                "op": "splice",
                "path": ["state", "unknown"],
                "start": 1,
                "delete_count": 3,
                "values": [8, 9, 4, 5],
            }],
        }
        event = self.write("event", [first, delta], encoding="full_then_delta_v2")
        dense = self.write("dense", [first, second])
        self.assertEqual(compare_boundary_traces(event, dense)["boundary_trace_equal"], True)

    def test_v2_splice_rejects_invalid_bounds(self):
        first = {"kind": "event", "state": {"items": [1, 2]}}
        delta = {
            "kind": "event_delta",
            "base_record": 1,
            "operations": [{
                "op": "splice", "path": ["state", "items"],
                "start": 2, "delete_count": 1, "values": [],
            }],
        }
        event = self.write("event", [first, delta], encoding="full_then_delta_v2")
        dense = self.write("dense", [first, first])
        with self.assertRaisesRegex(RuntimeError, "splice bounds"):
            compare_boundary_traces(event, dense)

    def test_native_boundary_summary_forces_fallback_on_unknown_state(self):
        event = self.write(
            "event",
            [
                {
                    "kind": "event",
                    "typed_boundary": {
                        "completeness": {
                            "incomplete": {
                                "unknown": ["shared_l2_transient_state", "unmodeled_completion_tails"]
                            }
                        }
                    },
                },
                {
                    "kind": "event",
                    "typed_boundary": {"completeness": "complete"},
                },
            ],
        )
        self.assertEqual(
            summarize_native_boundary_refinement(event, "event"),
            {
                "native_boundary_records": 2,
                "complete_boundary_records": 1,
                "exact_fallback_boundary_records": 1,
                "exact_fallback_required": True,
                "unknown_or_refused_fields": {
                    "shared_l2_transient_state": 1,
                    "unmodeled_completion_tails": 1,
                },
            },
        )

    def test_explicit_reify_decision_overrides_complete_typed_shape(self):
        event = self.write(
            "event",
            [{
                "kind": "event",
                "typed_boundary": {"completeness": "complete"},
                "native_boundary_refinement": {
                    "decision": "reify",
                    "reason": {"projection_mismatch": {}},
                },
            }],
        )
        report = summarize_native_boundary_refinement(event, "event")
        self.assertEqual(report["complete_boundary_records"], 0)
        self.assertEqual(report["exact_fallback_boundary_records"], 1)
        self.assertTrue(report["exact_fallback_required"])

    def test_failed_run_removes_partial_trace(self):
        trace = self.root / "partial.jsonl"
        command = self.root / "failed.py"
        command.write_text(
            "import os, pathlib, sys; "
            "pathlib.Path(os.environ['SIM_SM_REFERENCE_TRACE']).write_text('partial'); "
            "print('record limit 8 reached', file=sys.stderr); sys.exit(1)"
        )
        with self.assertRaisesRegex(RuntimeError, "run failed"):
            sm86_reference_compare.run_mode(
                [sys.executable, str(command)],
                self.root,
                "event",
                10.0,
                trace,
                8,
            )
        self.assertFalse(trace.exists())

    def test_timed_out_run_removes_partial_trace(self):
        trace = self.root / "timed-out.jsonl"
        command = self.root / "sleeping.py"
        command.write_text(
            "import os, pathlib, time; "
            "pathlib.Path(os.environ['SIM_SM_REFERENCE_TRACE']).write_text('partial'); "
            "time.sleep(10)"
        )
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            sm86_reference_compare.run_mode(
                [sys.executable, str(command)],
                self.root,
                "event",
                0.01,
                trace,
                8,
            )
        self.assertFalse(trace.exists())

    def test_relative_trace_path_is_resolved_outside_child_cwd(self):
        trace_dir = self.root / "relative-traces"
        trace_dir.mkdir()
        command = self.root / "failed-relative.py"
        command.write_text(
            "import os, pathlib, sys; "
            "pathlib.Path(os.environ['SIM_SM_REFERENCE_TRACE']).write_text('partial'); "
            "sys.exit(1)"
        )
        # Express a path relative to the comparator's parent while executing
        # the command from a different temporary child directory.
        relative_trace = pathlib.Path(
            os.path.relpath(trace_dir / "partial.jsonl", pathlib.Path.cwd())
        )
        with self.assertRaisesRegex(RuntimeError, "run failed"):
            sm86_reference_compare.run_mode(
                [sys.executable, str(command)],
                self.root,
                "event",
                10.0,
                relative_trace,
                8,
            )
        self.assertFalse(trace_dir.joinpath("partial.jsonl").exists())


class CommandSelectionTests(unittest.TestCase):
    def test_path_rewrite_preserves_argument_and_directory_boundaries(self):
        rewrites = sm86_reference_compare.parse_path_rewrites(
            ["/tmp/gpu_full_small/=/data1/capture's results"]
        )
        self.assertEqual(
            sm86_reference_compare.rewrite_command(
                ["sim-run", "--trace", "/tmp/gpu_full_small/evidence/raw/nvbit/x",
                 "--control=/tmp/gpu_full_small/card.json", "/tmp/gpu_full_small",
                 "/tmp/gpu_full_small_extra/card.json", "prefix/tmp/gpu_full_small"],
                rewrites,
            ),
            ["sim-run", "--trace", "/data1/capture's results/evidence/raw/nvbit/x",
             "--control=/data1/capture's results/card.json", "/data1/capture's results",
             "/tmp/gpu_full_small_extra/card.json", "prefix/tmp/gpu_full_small"],
        )
        with self.assertRaises(SystemExit):
            sm86_reference_compare.parse_path_rewrites(["missing_separator"])

    def test_cli_selects_every_directory_and_preserves_same_named_workload_traces(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            original_root = root / "old capture"
            relocated_root = root / "capture's new location"
            relocated_root.mkdir()
            simulator = relocated_root / "simulator.py"
            simulator.write_text(textwrap.dedent("""
                import json, os, pathlib, sys
                mode = os.environ["SIM_SM_REFERENCE_MODE"]
                cycle = int(sys.argv[1])
                print(json.dumps({"launches": [{"timing": {
                    "terminal_modeled_event_time": cycle}}]}))
                print(f"SIM_REFERENCE_PERF mode={mode} calls=1 "
                      "local_cycle_steps=1 idle_jumps=0 skipped_cycles=0",
                      file=sys.stderr)
                if path := os.environ.get("SIM_SM_REFERENCE_TRACE"):
                    rows = [
                        {"kind": "header", "mode": mode,
                         "state_projection_complete": False},
                        {"kind": "event", "time": cycle},
                        {"kind": "footer", "records_before_footer": 2,
                         "queue_events_popped": 1},
                    ]
                    pathlib.Path(path).write_text(
                        "".join(json.dumps(row) + "\\n" for row in rows))
            """))
            directories = [root / "small", root / "large"]
            commands = []
            for directory, cycle in zip(directories, [3, 5]):
                command = directory / "same_workload" / "command.txt"
                command.parent.mkdir(parents=True)
                command.write_text(shlex.join(
                    [sys.executable, str(original_root / "simulator.py"), str(cycle)]
                ))
                commands.append(command)
            traces = root / "traces"
            invocation = [
                sys.executable, sm86_reference_compare.__file__,
                "--command-dir", str(directories[0]),
                "--command-dir", str(directories[1]),
                "--trace-dir", str(traces),
                "--path-rewrite", f"{original_root}={relocated_root}",
            ]
            result = subprocess.run(
                invocation, capture_output=True, text=True, timeout=10, check=True,
            )
            report = json.loads(result.stdout)
            self.assertEqual([row["command_file"] for row in report["runs"]],
                             [str(command) for command in commands])
            self.assertEqual([row["terminal_event_time"] for row in report["runs"]], [3, 5])
            self.assertEqual([row["command"] for row in report["runs"]], [
                [sys.executable, str(simulator), str(cycle)] for cycle in [3, 5]
            ])
            self.assertEqual(report["cwd"], str(pathlib.Path.cwd()))
            self.assertTrue(report["all_outputs_equal"])
            self.assertTrue(report["all_boundary_traces_equal"])
            self.assertEqual(len(list(traces.glob("*.jsonl"))), 4)
            event_cycles = [
                json.loads(path.read_text().splitlines()[1])["time"]
                for path in sorted(traces.glob("*.event.jsonl"))
            ]
            self.assertEqual(event_cycles, [3, 5])

            # A missing second cohort must not produce a partial success.
            missing = root / "missing"
            result = subprocess.run(
                [*invocation, "--command-dir", str(missing)],
                capture_output=True, text=True, timeout=10,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(str(missing), result.stderr)


if __name__ == "__main__":
    unittest.main()
