from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
benchmark = importlib.import_module("benchmark")
events = importlib.import_module("collect_event_counts")
phases = importlib.import_module("collect_phase_times")


def test_case_matrix_is_unique_and_covers_feedback_layers() -> None:
    keys = [case.key for case in benchmark.CASES]
    assert len(keys) == len(set(keys)) == 12
    assert {
        "admission_red_d128_schema10",
        "admission_red_d128_schema11",
        "admission_red_d128_schema13",
    } <= set(keys)


def test_unknown_case_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="unknown case"):
        benchmark.selected_cases("not-a-case")


def test_simulator_environment_clears_ambient_measurement_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIM_RUN_DISABLE_L2_ACCESS_BATCHING", "ambient")
    monkeypatch.setenv("SIM_RUN_PERFORMANCE_COUNTERS", "ambient")
    monkeypatch.setenv("SIM_RUN_RETAIN_RAW_TRACE_PAYLOAD", "ambient")
    monkeypatch.setenv("SIM_RUN_STREAM_TRACE_PAYLOAD", "ambient")
    clean = benchmark.simulator_environment()
    assert "SIM_RUN_DISABLE_L2_ACCESS_BATCHING" not in clean
    assert "SIM_RUN_PERFORMANCE_COUNTERS" not in clean
    assert "SIM_RUN_RETAIN_RAW_TRACE_PAYLOAD" not in clean
    assert "SIM_RUN_STREAM_TRACE_PAYLOAD" not in clean
    controlled = benchmark.simulator_environment(
        disable_batching=True,
        performance_counters=True,
        retain_raw_trace_payload=True,
        stream_trace_payload=True,
    )
    assert controlled["SIM_RUN_DISABLE_L2_ACCESS_BATCHING"] == "1"
    assert controlled["SIM_RUN_PERFORMANCE_COUNTERS"] == "1"
    assert controlled["SIM_RUN_RETAIN_RAW_TRACE_PAYLOAD"] == "1"
    assert controlled["SIM_RUN_STREAM_TRACE_PAYLOAD"] == "1"


def test_performance_feature_probe_fails_closed_and_binds_binary_stat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shim = tmp_path / "sim-run-shim"
    shim.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "if os.environ.get('EMIT_PERF'):\n"
        "    print('SIM_EVENT_PERF events_processed=1', file=sys.stderr)\n"
        "    print('SIM_PHASE_PERF setup_ns=1', file=sys.stderr)\n"
        "print(json.dumps({'simulator_version': 'shim', 'launches': [{}]}))\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)
    monkeypatch.delenv("EMIT_PERF", raising=False)
    cpu = min(benchmark.os.sched_getaffinity(0))
    with pytest.raises(RuntimeError, match="feature is absent or inactive"):
        benchmark.require_performance_feature(shim, cpu, 5.0)
    monkeypatch.setenv("EMIT_PERF", "1")
    identity = benchmark.require_performance_feature(shim, cpu, 5.0)
    assert identity["simulator_version"] == "shim"
    benchmark.verify_binary_identity(identity)
    shim.write_text(shim.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="binary changed during measurement"):
        benchmark.verify_binary_identity(identity)


def test_feedback_case_binding_rejects_mislabeled_schema_and_candidate() -> None:
    case = next(
        case for case in benchmark.CASES if case.key == "admission_red_d128_schema13"
    )
    document = {
        "launches": [
            {
                "timing": {
                    "readiness": {
                        "schema_version": 11,
                        "memory_admission_feedback": {
                            "candidate": {"candidate_id": 202}
                        },
                    }
                }
            }
        ]
    }
    with pytest.raises(RuntimeError, match="readiness schema 11"):
        benchmark.validate_run_document(case, document)

    document["launches"][0]["timing"]["readiness"]["schema_version"] = 13
    with pytest.raises(RuntimeError, match="candidate 202"):
        benchmark.validate_run_document(case, document)


def test_run_once_success_path_returns_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shim = tmp_path / "sim-run-shim"
    shim.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "print(json.dumps({'simulator_version': 'shim', 'launches': ["
        "{'counters': {'records_seen': 7, 'instructions_issued': 6}}]}))\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)
    monkeypatch.setattr(
        benchmark,
        "command",
        lambda case, cpu, binary=benchmark.BINARY: [str(binary)],
    )
    case = benchmark.Case("shim", tmp_path, tmp_path)
    run, output = benchmark.run_once(case, 0, 5.0, binary=shim)
    assert run["records_seen"] == 7
    assert run["instructions_issued"] == 6
    assert run["result_bytes"] == len(output)


def test_paired_changes_are_aligned_by_repeat() -> None:
    baseline = [
        {"repeat": 2, "elapsed_seconds": 20.0},
        {"repeat": 1, "elapsed_seconds": 10.0},
    ]
    changed = [
        {"repeat": 1, "elapsed_seconds": 8.0},
        {"repeat": 2, "elapsed_seconds": 22.0},
    ]
    assert benchmark.paired_percent_changes(
        baseline, changed, "elapsed_seconds"
    ) == pytest.approx([-20.0, 10.0])


def test_comparison_evidence_requires_retained_canonical_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = tmp_path / "trace"
    trace.mkdir()
    (trace / "payload.pb").write_bytes(b"trace")
    control = tmp_path / "control.json"
    control.write_text("{}", encoding="utf-8")
    case = benchmark.Case("one", trace, control)
    monkeypatch.setattr(benchmark, "CASES", (case,))
    monkeypatch.setattr(benchmark, "REPO_ROOT", tmp_path)
    comparison_root = tmp_path / "comparison"
    canonical = comparison_root / "canonical/one.json"
    canonical.parent.mkdir(parents=True)
    payload = b'{"launches":[{}]}'
    canonical.write_bytes(payload)
    run = {"repeat": 1, "result_bytes": len(payload)}
    document = {
        "schema_version": 1,
        "method": {
            "feature_binary": "sim-run/performance-counters",
            "cross_condition_byte_equality_required": True,
            "measured_repeats": 1,
            "binary": {"performance_feature_probe": True},
        },
        "summaries": {"one": {}},
        "cases": {
            "one": {
                "trace": "trace",
                "control": "control.json",
                "extra_arguments": [],
                "trace_bytes": 5,
                "expected_readiness_schema": None,
                "expected_feedback_candidate_id": None,
                "canonical_output": "canonical/one.json",
                "canonical_result_bytes": len(payload),
            }
        },
        "runs": {"one": {"old": [run], "new": [dict(run)]}},
    }
    assert benchmark.validate_comparison_evidence(
        comparison_root, document, ("old", "new")
    )
    canonical.write_bytes(payload + b"\n")
    with pytest.raises(RuntimeError, match="canonical byte length differs"):
        benchmark.validate_comparison_evidence(
            comparison_root, document, ("old", "new")
        )


def test_event_counter_line_requires_complete_known_fields() -> None:
    parsed = events.parse_line(
        "SIM_EVENT_PERF events_processed=8 queue_events_processed=7 sm_wake=1 "
        "l2_access=2 l2_access_batches=1 l2_access_batched_requests=2 l2_fill=1 "
        "atomic_completion=1 mem_response=3 same_cycle_after_first=4 "
        "queue_same_cycle_after_first=3 maximum_queue_depth=5"
    )
    assert parsed is not None
    assert parsed["events_processed"] == 8
    assert parsed["maximum_queue_depth"] == 5
    assert events.parse_line("ordinary stderr") is None
    with pytest.raises(RuntimeError, match="missing event-performance counters"):
        events.parse_line("SIM_EVENT_PERF events_processed=1")
    with pytest.raises(RuntimeError, match="duplicate"):
        events.parse_line(
            "SIM_EVENT_PERF "
            + " ".join(f"{key}=0" for key in events.COUNTERS)
            + " events_processed=0"
        )
    with pytest.raises(RuntimeError, match="negative"):
        events.parse_line(
            "SIM_EVENT_PERF "
            + " ".join(
                f"{key}={-1 if key == 'events_processed' else 0}"
                for key in events.COUNTERS
            )
        )


def test_phase_counter_line_is_strictly_prefixed() -> None:
    assert phases.parse_line("ordinary stderr") is None
    assert phases.parse_line("SIM_PHASE_PERF replay_ns=17") == {"replay_ns": 17}
    with pytest.raises(RuntimeError, match="duplicate"):
        phases.parse_line("SIM_PHASE_PERF replay_ns=1 replay_ns=2")
    with pytest.raises(RuntimeError, match="negative"):
        phases.parse_line("SIM_PHASE_PERF replay_ns=-1")


def test_phase_fractions_use_matched_per_run_ratios() -> None:
    first = {field: 0 for field in phases.RUN_FIELDS}
    first.update({"setup_ns": 50, "run_total_ns": 100, "serialization_ns": 1})
    second = {field: 0 for field in phases.RUN_FIELDS}
    second.update({"setup_ns": 1, "run_total_ns": 10, "serialization_ns": 1})
    summary = phases.summarize_runs([first, second])
    assert summary["accounted_run_fraction"] == pytest.approx(0.3)
