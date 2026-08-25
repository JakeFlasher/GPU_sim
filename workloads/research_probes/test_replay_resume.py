from __future__ import annotations

import importlib.util
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent


def _load_module():
    path = HERE / "replay_resume.py"
    spec = importlib.util.spec_from_file_location("tested_replay_resume", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REPLAY = _load_module()


def _launch(kernel_id: int = 1) -> dict[str, object]:
    return {
        "device_id": 0,
        "stream_id": 0,
        "kernel_id": kernel_id,
        "function_unique_id": 7,
        "kernel_name": "_Z4testv",
        "grid": [2, 1, 1],
        "block": [32, 1, 1],
        "cluster": [1, 1, 1],
    }


def _counters() -> dict[str, object]:
    return {
        "instructions_issued": 2,
        "records_seen": 2,
        "l1": {
            "hits": 0,
            "misses": 0,
            "merges": 0,
            "stall_events": 0,
            "line_fills": 0,
            "bypass_sectors": 0,
            "store_sectors": 0,
            "reduction_sectors": 0,
        },
        "l2": {
            "hits": 0,
            "misses": 0,
            "line_misses": 0,
            "sector_fill_misses": 0,
            "write_tag_hits": 0,
        },
        "l2_by_source": {
            source: {"sectors": 0, "hits": 0}
            for source in ("l1_fill", "bypass", "untagged", "store")
        },
        "dram": {"read_sectors": 0, "bandwidth_delayed_responses": 0},
        "unmodeled_tails": {
            "reduction_read_modify_write": False,
            "dirty_writeback": False,
            "coherence": False,
            "async_copy_shared_visibility": False,
        },
    }


def _readiness(schema: int) -> dict[str, object]:
    cause = {
        "control_stall": 0,
        "dependence_counter_read": 0,
        "dependence_counter_non_memory_write": 0,
        "dependence_counter_shared_write": 0,
        "dependence_counter_l1tex": 0,
        "dependence_counter_unknown_origin": 0,
        "async_source": 0,
        "async_destination_local": 0,
        "async_destination_sm_service": 0,
        "cta_barrier_rendezvous": 0,
        "pipeline_tensor": 0,
        "eligible_not_selected": 0,
    }
    return {
        "schema_version": schema,
        "limiting_cycles": dict(cause),
        "constraint_exposure_cycles": dict(cause),
        "delayed_ready_issues": 0,
        "tied_limiting_causes": 0,
        "async_groups": {
            "groups_committed": 0,
            "groups_waited": 0,
            "source_limited_groups": 0,
            "destination_local_limited_groups": 0,
            "destination_sm_service_limited_groups": 0,
            "source_cycles": 0,
            "destination_local_cycles": 0,
            "destination_sm_service_cycles": 0,
            "completion_hidden_before_wait_cycles": 0,
        },
        "cta_barriers": {
            "generations": 0,
            "total_arrival_spread_cycles": 0,
            "max_arrival_spread_cycles": 0,
            "total_rendezvous_warp_cycles": 0,
            "releases_after_participant_retirement": 0,
        },
    }


def _report(
    expectation,
    *,
    readiness_schema: int = 1,
) -> dict[str, object]:
    launches = []
    prior_ids: list[int] = []
    for index, identity in enumerate(expectation.launches):
        readiness = _readiness(readiness_schema)
        if expectation.service_candidates:
            readiness["memory_service_candidates"] = [
                {"candidate": dict(candidate)}
                for candidate in expectation.service_candidates
            ]
        if expectation.admission_candidates:
            readiness["memory_admission_candidates"] = [
                {"candidate": dict(candidate)}
                for candidate in expectation.admission_candidates
            ]
        if expectation.feedback_candidate is not None:
            readiness["memory_admission_feedback"] = {
                "candidate": dict(expectation.feedback_candidate)
            }
        launch = {
            **identity,
            "coverage": {
                "kind": "full",
                "blocks_observed": 2,
                "population_blocks": 2,
            },
            "counters": _counters(),
            "timing": {
                "diagnostic_last_issue_cycles": 1,
                "last_processed_modeled_system_event_time": None,
                "terminal_modeled_event_time": 1,
                "completion_frontier": {"status": "complete", "cycles": 1},
                "readiness": readiness,
            },
        }
        if expectation.cache_scope == "stream-l2":
            launch["cache_handoff"] = {
                "scope": "stream-l2",
                "sequence_index": index,
                "seeded_by_kernel_ids": list(prior_ids),
            }
        prior_ids.append(identity["kernel_id"])
        launches.append(launch)
    document: dict[str, object] = {
        "schema_version": 1,
        "simulator_version": REPLAY.simulator_package_version(),
        "architecture": expectation.architecture,
        "memory_mode": expectation.memory_mode,
        "cache_scope": expectation.cache_scope,
        "carded_dram_bandwidth": expectation.carded_dram,
        "memory_service_candidate_count": len(expectation.service_candidates),
        "memory_admission_candidate_count": len(expectation.admission_candidates),
        "launches": launches,
    }
    if expectation.feedback_candidate is not None:
        document["memory_admission_feedback_candidate_id"] = (
            expectation.feedback_candidate["candidate_id"]
        )
    if expectation.cache_scope == "stream-l2":
        document["cache_session"] = {
            "scope": "stream-l2",
            "device_id": 0,
            "stream_id": 0,
            "completed_launches": len(expectation.launches),
            "cumulative_l2": dict(_counters()["l2"]),
        }
    return document


def _write_with_mtime(path: Path, value: object, mtime_ns: int) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    os.utime(path, ns=(mtime_ns, mtime_ns))


def test_expected_launches_are_derived_from_trace_metadata(tmp_path: Path) -> None:
    trace = tmp_path / "trace"
    (trace / "extra_info").mkdir(parents=True)
    (trace / "extra_info/enhanced_execution_info.json").write_text(
        json.dumps(
            {
                "kernels": [
                    {"unique_function_id": 7, "kernel_name": "_Z4testv"}
                ]
            }
        ),
        encoding="utf-8",
    )
    (trace / "threadblocks/device_0/stream_0/kernel_2").mkdir(parents=True)
    (trace / "threadblocks/device_0/stream_0/kernel_1").mkdir(parents=True)

    launches = REPLAY.expected_launches_from_trace(
        trace, grid=(2, 1, 1), block=(32, 1, 1)
    )

    assert [launch["kernel_id"] for launch in launches] == [1, 2]
    assert launches[0] == _launch(1)


@pytest.mark.parametrize(
    "payload",
    [
        '{"schema_version":1,"schema_version":1,"candidates":[{}]}',
        '{"schema_version":1,"candidates":[{"rate":NaN}]}',
    ],
)
def test_candidate_configuration_loading_is_strict(
    tmp_path: Path, payload: str
) -> None:
    candidate_file = tmp_path / "candidates.json"
    candidate_file.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        REPLAY.candidate_configurations(candidate_file)


def test_feedback_candidate_loading_requires_exactly_one(tmp_path: Path) -> None:
    candidate_file = tmp_path / "feedback.json"
    candidate_file.write_text(
        '{"schema_version":1,"candidates":[{},{}]}', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="exactly one candidate, got 2"):
        REPLAY.candidate_configurations(candidate_file, require_exactly_one=True)


def test_resume_requires_strict_dependency_freshness_and_exact_root(tmp_path: Path) -> None:
    dependency = tmp_path / "input"
    dependency.write_text("input", encoding="utf-8")
    expectation = REPLAY.ReplayExpectation(
        launches=(_launch(),), carded_dram=False
    )
    result = tmp_path / "result.json"
    _write_with_mtime(dependency, "input", 1_000_000_000)
    _write_with_mtime(result, _report(expectation), 2_000_000_000)
    assert REPLAY.reusable_simulator_output(result, [dependency], expectation)

    os.utime(dependency, ns=(2_000_000_000, 2_000_000_000))
    assert "strictly postdate" in REPLAY.simulator_output_reuse_error(
        result, [dependency], expectation
    )

    os.utime(dependency, ns=(1_000_000_000, 1_000_000_000))
    with_extra = _report(expectation)
    with_extra["unexpected"] = True
    _write_with_mtime(result, with_extra, 2_000_000_000)
    assert "root fields" in REPLAY.simulator_output_reuse_error(
        result, [dependency], expectation
    )


def test_resume_rejects_gutted_counters_and_malformed_nested_timing(
    tmp_path: Path,
) -> None:
    dependency = tmp_path / "input"
    expectation = REPLAY.ReplayExpectation(
        launches=(_launch(),), carded_dram=False
    )
    result = tmp_path / "result.json"
    _write_with_mtime(dependency, "input", 1_000_000_000)

    gutted = _report(expectation)
    gutted["launches"][0]["counters"] = {}
    _write_with_mtime(result, gutted, 2_000_000_000)
    assert "counters fields differ" in REPLAY.simulator_output_reuse_error(
        result, [dependency], expectation
    )

    malformed = _report(expectation)
    malformed["launches"][0]["timing"]["diagnostic_last_issue_cycles"] = "one"
    _write_with_mtime(result, malformed, 2_000_000_000)
    assert "diagnostic_last_issue_cycles is malformed" in (
        REPLAY.simulator_output_reuse_error(result, [dependency], expectation)
    )

    malformed = _report(expectation)
    malformed["launches"][0]["timing"]["readiness"]["limiting_cycles"][
        "control_stall"
    ] = -1
    _write_with_mtime(result, malformed, 2_000_000_000)
    assert "malformed counter" in REPLAY.simulator_output_reuse_error(
        result, [dependency], expectation
    )


@pytest.mark.parametrize(
    ("mode", "schema"),
    [("service", 7), ("admission", 9), ("feedback", 10)],
)
def test_same_count_candidate_configuration_drift_is_not_reused(
    tmp_path: Path, mode: str, schema: int
) -> None:
    dependency = tmp_path / "candidate.json"
    dependency.write_text("{}", encoding="utf-8")
    old_candidate = {"candidate_id": 4, "rate": 8}
    new_candidate = {"candidate_id": 4, "rate": 9}
    kwargs: dict[str, object] = {}
    if mode == "service":
        kwargs["service_candidates"] = (old_candidate,)
    elif mode == "admission":
        kwargs["admission_candidates"] = (old_candidate,)
    else:
        kwargs["feedback_candidate"] = old_candidate
    old_expectation = REPLAY.ReplayExpectation(
        launches=(_launch(),), carded_dram=False, **kwargs
    )
    result = tmp_path / "result.json"
    _write_with_mtime(dependency, {}, 1_000_000_000)
    _write_with_mtime(result, _report(old_expectation, readiness_schema=schema), 2_000_000_000)

    kwargs[{
        "service": "service_candidates",
        "admission": "admission_candidates",
        "feedback": "feedback_candidate",
    }[mode]] = (new_candidate,) if mode != "feedback" else new_candidate
    current_expectation = REPLAY.ReplayExpectation(
        launches=(_launch(),), carded_dram=False, **kwargs
    )
    error = REPLAY.simulator_output_reuse_error(
        result, [dependency], current_expectation
    )
    assert error is not None and "configuration differs" in error


def test_completion_feedback_requires_readiness_schema_eleven(tmp_path: Path) -> None:
    dependency = tmp_path / "candidate.json"
    dependency.write_text("{}", encoding="utf-8")
    candidate = {
        "candidate_id": 201,
        "membar_waits_for_service_completion": True,
    }
    expectation = REPLAY.ReplayExpectation(
        launches=(_launch(),),
        carded_dram=False,
        feedback_candidate=candidate,
    )
    result = tmp_path / "result.json"
    _write_with_mtime(dependency, {}, 1_000_000_000)
    _write_with_mtime(result, _report(expectation, readiness_schema=11), 2_000_000_000)
    assert REPLAY.reusable_simulator_output(result, [dependency], expectation)

    _write_with_mtime(result, _report(expectation, readiness_schema=10), 2_000_000_000)
    assert "readiness schema" in REPLAY.simulator_output_reuse_error(
        result, [dependency], expectation
    )

    latency_candidate = {
        **candidate,
        "candidate_id": 203,
        "write_completion_latency_cycles": 512,
    }
    latency_expectation = REPLAY.ReplayExpectation(
        launches=(_launch(),),
        carded_dram=False,
        feedback_candidate=latency_candidate,
    )
    _write_with_mtime(
        result,
        _report(latency_expectation, readiness_schema=12),
        2_000_000_000,
    )
    assert REPLAY.reusable_simulator_output(
        result, [dependency], latency_expectation
    )

    work_candidate = {
        **candidate,
        "candidate_id": 211,
        "write_completion_request_cycles": 4,
        "write_completion_parallel_lanes": 2,
    }
    work_expectation = REPLAY.ReplayExpectation(
        launches=(_launch(),),
        carded_dram=False,
        feedback_candidate=work_candidate,
    )
    _write_with_mtime(
        result,
        _report(work_expectation, readiness_schema=13),
        2_000_000_000,
    )
    assert REPLAY.reusable_simulator_output(result, [dependency], work_expectation)


def test_launch_identity_and_stream_handoff_are_exact(tmp_path: Path) -> None:
    dependency = tmp_path / "input"
    dependency.write_text("input", encoding="utf-8")
    expectation = REPLAY.ReplayExpectation(
        launches=(_launch(1), _launch(2)),
        carded_dram=True,
        cache_scope="stream-l2",
    )
    report = _report(expectation)
    result = tmp_path / "result.json"
    _write_with_mtime(dependency, "input", 1_000_000_000)
    _write_with_mtime(result, report, 2_000_000_000)
    assert REPLAY.reusable_simulator_output(result, [dependency], expectation)

    changed = deepcopy(report)
    changed["launches"][1]["kernel_name"] = "_Z5otherv"
    _write_with_mtime(result, changed, 2_000_000_000)
    assert "identity differs" in REPLAY.simulator_output_reuse_error(
        result, [dependency], expectation
    )

    changed = deepcopy(report)
    changed["launches"][1]["cache_handoff"]["seeded_by_kernel_ids"] = []
    _write_with_mtime(result, changed, 2_000_000_000)
    assert "cache handoff" in REPLAY.simulator_output_reuse_error(
        result, [dependency], expectation
    )


def test_pair_validation_requires_identical_bytes(tmp_path: Path) -> None:
    dependency = tmp_path / "input"
    dependency.write_text("input", encoding="utf-8")
    expectation = REPLAY.ReplayExpectation(
        launches=(_launch(),), carded_dram=False
    )
    document = _report(expectation)
    first = tmp_path / "replay-1.json"
    second = tmp_path / "replay-2.json"
    _write_with_mtime(dependency, "input", 1_000_000_000)
    first.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    second.write_text(json.dumps(document, sort_keys=True, indent=2), encoding="utf-8")
    os.utime(first, ns=(2_000_000_000, 2_000_000_000))
    os.utime(second, ns=(2_000_000_000, 2_000_000_000))

    assert REPLAY.replay_pair_error(
        first, second, [dependency], expectation
    ) == "replay pair is not byte-identical"
