"""Focused checks for the active-replay validator's supplemental allowlist."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from workloads.research_probes.semantic_replay_validation import (
    _async_source_error,
    _global_memory_error,
)
from workloads.research_probes import semantic_replay_validation as SEMANTIC
from workloads.research_probes.trace_invariants import (
    LaunchTraceInvariants,
    TraceInvariants,
)


def _load_validator():
    path = Path(__file__).with_name("validate_active_replay_pairs.py")
    spec = importlib.util.spec_from_file_location("tested_active_validator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_validator()


def _load_replay_test_helpers():
    path = Path(__file__).with_name("test_replay_resume.py")
    spec = importlib.util.spec_from_file_location("replay_test_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REPLAY_HELPERS = _load_replay_test_helpers()


def test_full_campaign_supplemental_directory_is_recognized(tmp_path: Path) -> None:
    root = tmp_path / "admission"
    known = root / "simulator-memory-admission-feedback-service-101" / (
        "b84_w8_store_line_unique_d32_g0_all"
    )
    assert VALIDATOR._known_supplemental_replay_directory(known, root)


def test_unknown_candidate_or_cell_is_not_recognized(tmp_path: Path) -> None:
    root = tmp_path / "admission"
    unknown_id = root / "simulator-memory-admission-feedback-service-999" / (
        "b84_w8_store_line_unique_d32_g0_all"
    )
    unknown_cell = root / "simulator-memory-admission-feedback-service-101" / (
        "not-a-campaign-cell"
    )
    assert not VALIDATOR._known_supplemental_replay_directory(unknown_id, root)
    assert not VALIDATOR._known_supplemental_replay_directory(unknown_cell, root)


def test_recognized_supplemental_pair_is_not_exempt_when_missing(
    monkeypatch, tmp_path: Path
) -> None:
    root = tmp_path / "admission"
    destination = root / "simulator-memory-admission-feedback-service-101" / (
        "b84_w8_store_line_unique_d32_g0_all"
    )
    destination.mkdir(parents=True)
    sentinel = object()
    monkeypatch.setattr(
        VALIDATOR,
        "_supplemental_pair_spec",
        lambda _path, _root, _binary: (
            destination / "replay-1.json",
            destination / "replay-2.json",
            [],
            sentinel,
            root / "nvbit" / destination.name,
        ),
    )
    assert "missing or empty" in VALIDATOR._validate_supplemental_directory(
        destination, root, tmp_path / "sim-run", []
    )


def test_supplemental_pair_rejects_mismatch_stale_and_schema(
    monkeypatch, tmp_path: Path
) -> None:
    root = tmp_path / "admission"
    destination = root / "simulator-memory-admission-feedback-service-101" / (
        "b84_w8_store_line_unique_d32_g0_all"
    )
    destination.mkdir(parents=True)
    dependency = tmp_path / "dependency"
    dependency.write_text("input", encoding="utf-8")
    expectation = VALIDATOR.REPLAY.ReplayExpectation(
        launches=(REPLAY_HELPERS._launch(),), carded_dram=False
    )
    trace = tmp_path / "trace"
    monkeypatch.setattr(
        VALIDATOR,
        "_supplemental_pair_spec",
        lambda _path, _root, _binary: (
            destination / "replay-1.json",
            destination / "replay-2.json",
            [dependency],
            expectation,
            trace,
        ),
    )
    monkeypatch.setattr(VALIDATOR, "validate_replay_document", lambda *args, **kwargs: ([{}], []))

    valid = REPLAY_HELPERS._report(expectation)

    def write_pair(left: object, right: object, result_time: int, dependency_time: int) -> None:
        dependency_time = int(dependency_time)
        result_time = int(result_time)
        dependency.touch()
        os.utime(dependency, ns=(dependency_time, dependency_time))
        for name, value in (("replay-1.json", left), ("replay-2.json", right)):
            path = destination / name
            path.write_text(json.dumps(value), encoding="utf-8")
            os.utime(path, ns=(result_time, result_time))

    different = json.loads(json.dumps(valid))
    different["launches"][0]["timing"]["diagnostic_last_issue_cycles"] = 2
    write_pair(valid, different, 2_000_000_000, 1_000_000_000)
    assert "not byte-identical" in VALIDATOR._validate_supplemental_directory(
        destination, root, tmp_path / "sim-run", []
    )

    write_pair(valid, valid, 2_000_000_000, 3_000_000_000)
    assert "strictly postdate" in VALIDATOR._validate_supplemental_directory(
        destination, root, tmp_path / "sim-run", []
    )

    write_pair({}, {}, 2_000_000_000, 1_000_000_000)
    assert "root fields differ" in VALIDATOR._validate_supplemental_directory(
        destination, root, tmp_path / "sim-run", []
    )


def test_trace_derived_traffic_mutation_is_rejected() -> None:
    expected = SimpleNamespace(
        global_load_instructions=2,
        global_store_instructions=3,
        global_atomic_instructions=4,
        global_load_sectors=8,
        global_store_sectors=12,
        global_atomic_sectors=16,
        async_source_sectors=0,
    )
    memory = {
        "load_instructions": 2,
        "store_instructions": 3,
        "atomic_instructions": 0,
        "reduction_instructions": 4,
        "load_sectors": 8,
        "store_sectors": 12,
        "atomic_sectors": 0,
        "reduction_sectors": 16,
    }
    error, _snapshot = _global_memory_error(memory, expected, "fixture")
    assert error is None
    memory["store_sectors"] = 13
    error, _snapshot = _global_memory_error(memory, expected, "fixture")
    assert "global_store_sectors mismatch" in error


def test_async_source_sector_mutation_is_rejected() -> None:
    expected = SimpleNamespace(
        global_load_sectors=0,
        global_atomic_sectors=0,
        async_source_sectors=32,
    )
    counters = {"l1": {"hits": 24, "misses": 8, "bypass_sectors": 0}}
    error, snapshot = _async_source_error(counters, expected, "async-fixture")
    assert error is None and snapshot["passed"] is True
    counters["l1"]["misses"] = 7
    error, _snapshot = _async_source_error(counters, expected, "async-fixture")
    assert "async_source_sectors mismatch" in error


def test_global_memory_diagnostic_is_required_when_trace_has_global_traffic(
    monkeypatch, tmp_path: Path
) -> None:
    expected = LaunchTraceInvariants(
        device_id=0,
        stream_id=0,
        kernel_id=1,
        blocks=2,
        warps=2,
        instructions=2,
        memory_instructions=0,
        load_instructions=0,
        store_instructions=0,
        atomic_instructions=0,
        async_copy_instructions=0,
        global_load_instructions=0,
        global_store_instructions=0,
        global_atomic_instructions=0,
        global_load_sectors=0,
        global_store_sectors=32,
        global_atomic_sectors=0,
        async_source_sectors=0,
        local_load_sectors=0,
        local_store_sectors=0,
        shared_sectors=0,
    )
    trace = TraceInvariants(trace_dir=str(tmp_path), launches=(expected,))
    monkeypatch.setattr(SEMANTIC, "derive_trace_invariants", lambda _path: trace)
    expectation = VALIDATOR.REPLAY.ReplayExpectation(
        launches=(REPLAY_HELPERS._launch(),), carded_dram=False
    )
    document = REPLAY_HELPERS._report(expectation)
    document["launches"][0]["timing"]["readiness"]["global_memory_issue"] = {
        "load_instructions": 0,
        "store_instructions": 0,
        "atomic_instructions": 0,
        "reduction_instructions": 0,
        "load_sectors": 0,
        "store_sectors": 32,
        "atomic_sectors": 0,
        "reduction_sectors": 0,
    }
    rows, errors = SEMANTIC.validate_replay_document(
        document, tmp_path, label="global-fixture"
    )
    assert not errors and rows[0]["checks"]["trace_global_memory_conservation"]
    del document["launches"][0]["timing"]["readiness"]["global_memory_issue"]
    _rows, errors = SEMANTIC.validate_replay_document(
        document, tmp_path, label="global-fixture"
    )
    assert any("global_memory_issue diagnostic is required" in error for error in errors)


def test_global_instruction_without_sectors_still_requires_diagnostic(
    monkeypatch, tmp_path: Path
) -> None:
    expected = LaunchTraceInvariants(
        device_id=0,
        stream_id=0,
        kernel_id=1,
        blocks=2,
        warps=2,
        instructions=2,
        memory_instructions=0,
        load_instructions=0,
        store_instructions=0,
        atomic_instructions=0,
        async_copy_instructions=0,
        global_load_instructions=0,
        global_store_instructions=1,
        global_atomic_instructions=0,
        global_load_sectors=0,
        global_store_sectors=0,
        global_atomic_sectors=0,
        async_source_sectors=0,
        local_load_sectors=0,
        local_store_sectors=0,
        shared_sectors=0,
    )
    monkeypatch.setattr(
        SEMANTIC,
        "derive_trace_invariants",
        lambda _path: TraceInvariants(trace_dir=str(tmp_path), launches=(expected,)),
    )
    expectation = VALIDATOR.REPLAY.ReplayExpectation(
        launches=(REPLAY_HELPERS._launch(),), carded_dram=False
    )
    document = REPLAY_HELPERS._report(expectation)
    document["launches"][0]["timing"]["readiness"]["global_memory_issue"] = {
        "load_instructions": 0,
        "store_instructions": 1,
        "atomic_instructions": 0,
        "reduction_instructions": 0,
        "load_sectors": 0,
        "store_sectors": 0,
        "atomic_sectors": 0,
        "reduction_sectors": 0,
    }
    del document["launches"][0]["timing"]["readiness"]["global_memory_issue"]
    _rows, errors = SEMANTIC.validate_replay_document(
        document, tmp_path, label="global-instruction-fixture"
    )
    assert any("global_memory_issue diagnostic is required" in error for error in errors)


def test_global_memory_diagnostic_rejects_partial_or_non_integer_fields(
    monkeypatch, tmp_path: Path
) -> None:
    expected = LaunchTraceInvariants(
        device_id=0,
        stream_id=0,
        kernel_id=1,
        blocks=2,
        warps=2,
        instructions=2,
        memory_instructions=0,
        load_instructions=0,
        store_instructions=1,
        atomic_instructions=0,
        async_copy_instructions=0,
        global_load_instructions=0,
        global_store_instructions=1,
        global_atomic_instructions=0,
        global_load_sectors=0,
        global_store_sectors=0,
        global_atomic_sectors=0,
        async_source_sectors=0,
        local_load_sectors=0,
        local_store_sectors=0,
        shared_sectors=0,
    )
    monkeypatch.setattr(
        SEMANTIC,
        "derive_trace_invariants",
        lambda _path: TraceInvariants(trace_dir=str(tmp_path), launches=(expected,)),
    )
    expectation = VALIDATOR.REPLAY.ReplayExpectation(
        launches=(REPLAY_HELPERS._launch(),), carded_dram=False
    )
    document = REPLAY_HELPERS._report(expectation)
    diagnostic = {
        "load_instructions": 0,
        "store_instructions": 1,
        "atomic_instructions": 0,
        "reduction_instructions": 0,
        "load_sectors": 0,
        "store_sectors": 0,
        "atomic_sectors": 0,
        "reduction_sectors": 0,
    }
    document["launches"][0]["timing"]["readiness"]["global_memory_issue"] = diagnostic
    del diagnostic["store_sectors"]
    _rows, errors = SEMANTIC.validate_replay_document(
        document, tmp_path, label="partial-global-fixture"
    )
    assert any("malformed" in error for error in errors)
    diagnostic["store_sectors"] = "0"
    _rows, errors = SEMANTIC.validate_replay_document(
        document, tmp_path, label="typed-global-fixture"
    )
    assert any("malformed" in error for error in errors)
