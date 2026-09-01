"""Focused checks for the active-replay validator's supplemental allowlist."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def _load_live_cohort_validator():
    path = Path(__file__).with_name("validate_sm86_live_cohort.py")
    spec = importlib.util.spec_from_file_location("tested_live_cohort_validator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


LIVE_VALIDATOR = _load_live_cohort_validator()


@pytest.mark.parametrize("value", [False, 0.0, "0", 1, None])
def test_stage_success_requires_exact_integer_zero(value: object) -> None:
    assert LIVE_VALIDATOR._is_exact_zero(value) is False
    assert LIVE_VALIDATOR._is_exact_zero(0) is True


def test_nvbit_expected_argv_is_cell_bound() -> None:
    cell = {
        "id": "gemm_hmma_bf16_sm80",
        "binary": {
            "path": "workloads/portable_lcd/gemm_hmma_bf16_sm80/gemm_hmma_bf16_sm80_run",
            "trace_path": "workloads/portable_lcd/gemm_hmma_bf16_sm80/gemm_hmma_bf16_sm80_run",
        },
    }
    expected = LIVE_VALIDATOR._expected_nvbit_argv(cell)
    assert expected[-5:] == ["--timing_shape", "--warmup", "0", "--reps", "1"]
    altered = list(expected)
    altered[-1] = "2"
    assert altered != expected


def _live_manifest_cell(cell_id: str) -> tuple[Path, dict[str, object], dict[str, object]]:
    root = Path("results/captures/sm86_simulator_live_baseline_20260829_v2")
    manifest = json.loads((root / "campaign_manifest.json").read_text(encoding="utf-8"))
    cell = next(cell for cell in manifest["cells"] if cell["id"] == cell_id)
    return root, cell, manifest["repository"]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("LD_PRELOAD", "/tmp/untrusted.so"),
        ("CUDA_VISIBLE_DEVICES", "1"),
        ("UNAPPROVED_FLAG", "1"),
        ("LD_LIBRARY_PATH", "/tmp/untrusted-libs"),
    ],
)
def test_nvbit_environment_allowlist_rejects_unbound_values(key: str, value: str) -> None:
    root, cell, repository = _live_manifest_cell("gemm_hmma_bf16_sm80")
    stage = deepcopy(cell["stages"]["nvbit"])
    stage["env"][key] = value
    with pytest.raises(ValueError, match="environment"):
        LIVE_VALIDATOR._validate_nvbit_stage(
            root,
            cell,
            cell["binary"],
            stage,
            repository["tracer_dso_sha256"],
        )


def test_current_nvbit_environment_matches_exact_allowlist() -> None:
    root, cell, repository = _live_manifest_cell("gemm_hmma_bf16_sm80")
    LIVE_VALIDATOR._validate_nvbit_stage(
        root,
        cell,
        cell["binary"],
        cell["stages"]["nvbit"],
        repository["tracer_dso_sha256"],
    )


def test_nvbit_effective_environment_record_rejects_inherited_controls(tmp_path: Path) -> None:
    cell = {
        "id": "gemm_hmma_bf16_sm80",
        "binary": {
            "path": "workloads/portable_lcd/gemm_hmma_bf16_sm80/gemm_hmma_bf16_sm80_run"
        },
    }
    output = "evidence/raw/nvbit/gemm_hmma_bf16_sm80"
    argv = LIVE_VALIDATOR._expected_nvbit_argv(cell)
    environment = LIVE_VALIDATOR._expected_nvbit_environment(cell, output, root=tmp_path)
    record_path = tmp_path / output / "environment.json"
    record_path.parent.mkdir(parents=True)
    record = {
        "schema_version": 1,
        "kind": "sanitized_nvbit_capture_environment",
        "cell": cell["id"],
        "argv": argv,
        "output": output,
        "launch_mode": "env_i_exact_allowlist",
        "environment": environment,
        "effective_environment": {**environment, "LD_PRELOAD": "/tmp/hostile.so"},
        "forbidden_variables_absent": sorted(
            LIVE_VALIDATOR._NVBIT_FORBIDDEN_ENVIRONMENT_CONTROLS
        ),
        "return_code": 0,
        "timed_out": False,
        "cleanup_complete": True,
        "timeout_seconds": 900,
        "timeout_policy": "process_group_terminate_reap_v1",
        "timeout_policy_version": 1,
        "runtime_manifest_path": LIVE_VALIDATOR._EXPECTED_RUNTIME_MANIFEST_PATH,
        "runtime_manifest_sha256": hashlib.sha256(
            Path(LIVE_VALIDATOR._EXPECTED_RUNTIME_MANIFEST_PATH).read_bytes()
        ).hexdigest(),
        "runtime_dso_closure": {
            str(row["soname"]): str(row["sha256"])
            for row in json.loads(
                Path(LIVE_VALIDATOR._EXPECTED_RUNTIME_MANIFEST_PATH).read_text()
            )["dsos"]
        },
        "runtime_dso_closure_sha256": "0" * 64,
        "runtime_error": None,
        "output_contract_version": 1,
        "required_outputs": list(LIVE_VALIDATOR._NVBIT_REQUIRED_OUTPUTS),
        "payload_file_count": 1,
        "output_contract_passed": True,
        "output_contract_error": None,
        "started_at": "2026-08-30T00:00:00+00:00",
        "finished_at": "2026-08-30T00:00:01+00:00",
    }
    record_path.write_text(json.dumps(record), encoding="utf-8")
    stage = {
        "environment_record": output + "/environment.json",
        "environment_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
    }
    record["runtime_dso_closure_sha256"] = LIVE_VALIDATOR._runtime_closure_sha256(
        record["runtime_dso_closure"]
    )
    record_path.write_text(json.dumps(record), encoding="utf-8")
    stage["environment_sha256"] = hashlib.sha256(record_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="effective environment"):
        LIVE_VALIDATOR._validate_nvbit_environment_record(
            tmp_path, cell, stage, argv, output, environment
        )


def test_cp_async_ncu_target_is_the_matched_trace_binary() -> None:
    _root, cell, _repository = _live_manifest_cell("cp_async_canary")
    assert LIVE_VALIDATOR.closure_analyzer._expected_ncu_target(cell) == [
        "workloads/research_probes/async_barrier_overlap/async_barrier_overlap_trace",
        "c11",
        "l1",
        "0",
        "4",
        "1",
    ]


def test_cp_async_matched_execution_requires_exact_correctness_stage() -> None:
    root, cell, repository = _live_manifest_cell("cp_async_canary")
    manifest = json.loads((root / "campaign_manifest.json").read_text(encoding="utf-8"))
    defaults = manifest["stage_timeouts_seconds"]
    LIVE_VALIDATOR._validate_cell(
        root,
        cell,
        defaults,
        repository["tracer_dso_sha256"],
        manifest["ncu_identity"],
    )

    missing = deepcopy(cell)
    del missing["stages"]["correctness"]
    with pytest.raises(ValueError, match="stage set is incomplete"):
        LIVE_VALIDATOR._validate_cell(
            root,
            missing,
            defaults,
            repository["tracer_dso_sha256"],
            manifest["ncu_identity"],
        )

    wrong_binary = deepcopy(cell)
    wrong_binary["stages"]["correctness"]["binary_path"] = cell["binary"]["path"]
    with pytest.raises(ValueError, match="correctness binary identity"):
        LIVE_VALIDATOR._validate_cell(
            root,
            wrong_binary,
            defaults,
            repository["tracer_dso_sha256"],
            manifest["ncu_identity"],
        )


def test_cp_async_limitation_must_match_structured_group_provenance() -> None:
    _root, cell, _repository = _live_manifest_cell("cp_async_canary")
    altered = deepcopy(cell)
    altered["limitations"][0] = (
        "Trace binary uses 64 groups while the NCU target uses 16384 groups; "
        "raw cycle totals are not directly comparable."
    )
    with pytest.raises(ValueError, match="limitation"):
        LIVE_VALIDATOR._validate_cp_async_matched_execution(altered)

def test_nvbit_timeout_override_is_rejected_by_the_campaign_contract(
    tmp_path: Path,
) -> None:
    root, cell, _repository = _live_manifest_cell("gemm_hmma_bf16_sm80")
    stage = deepcopy(cell["stages"]["nvbit"])
    output = stage["output"]
    source = root / stage["environment_record"]
    destination = tmp_path / stage["environment_record"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = json.loads(source.read_text(encoding="utf-8"))
    expected = LIVE_VALIDATOR._expected_nvbit_environment(cell, output, root=tmp_path)
    record["environment"] = expected
    effective = dict(expected)
    effective["LD_LIBRARY_PATH"] = "/proc/self/fd/1"
    effective["CUDA_INJECTION64_PATH"] = "/proc/self/fd/2"
    record["effective_environment"] = effective
    record["timeout_seconds"] = 901
    destination.write_text(json.dumps(record), encoding="utf-8")
    stage["environment_record"] = str(Path(stage["environment_record"]))
    stage["environment_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="sanitized contract|timeout policy"):
        LIVE_VALIDATOR._validate_nvbit_environment_record(
            tmp_path,
            cell,
            stage,
            stage["argv"],
            output,
            expected,
        )


def _scaling_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    source_root, source_cell, _repository = _live_manifest_cell("cp_async_canary")
    for relative in (
        "evidence/raw/correctness/cp_async_canary",
        "evidence/raw/ncu_scaling/cp_async_canary_16384",
    ):
        shutil.copytree(source_root / relative, tmp_path / relative)
    return tmp_path, deepcopy(source_cell)


@pytest.mark.parametrize("mutation", ["export_status", "empty_csv", "timeout"])
def test_cp_async_scaling_observation_validates_export_and_csv_contract(
    tmp_path: Path, mutation: str
) -> None:
    root, cell = _scaling_fixture(tmp_path)
    scaling_ncu = cell["scaling_observation"]["ncu"]
    if mutation == "export_status":
        metadata_path = root / scaling_ncu["metadata"]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["export_return_code"] = 1
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        scaling_ncu["metadata_sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    elif mutation == "timeout":
        metadata_path = root / scaling_ncu["metadata"]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["timeout_seconds"] = 901
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        scaling_ncu["metadata_sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    else:
        csv_path = root / scaling_ncu["csv"]
        csv_path.write_text("", encoding="utf-8")
        scaling_ncu["csv_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        LIVE_VALIDATOR._validate_cp_async_scaling_observation(root, cell)


def test_ncu_capture_metadata_is_bound_to_trusted_executable(tmp_path: Path) -> None:
    root, cell, _repository = _live_manifest_cell("gemm_hmma_bf16_sm80")
    stage = deepcopy(cell["stages"]["ncu"])
    source = root / stage["metadata"]
    # The fixture is materialized under the expected campaign-relative paths so
    # the helper exercises hashes and command normalization, not just field types.
    for relative in (
        stage["report"],
        stage["csv"],
        stage["metadata"],
        stage["command_file"],
        stage["export_command_file"],
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, destination)
    for key in ("report", "csv", "metadata", "command_file", "export_command_file"):
        path = tmp_path / stage[key]
        stage[f"{key}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata_path = tmp_path / stage["metadata"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["command"][0] = "ncu"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    stage["metadata_sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="trusted executable"):
        LIVE_VALIDATOR._validate_ncu_capture_provenance(tmp_path, cell, stage)


def test_static_control_producer_and_schema_are_pinned(tmp_path: Path) -> None:
    root, cell, _repository = _live_manifest_cell("gemm_hmma_bf16_sm80")
    static = deepcopy(cell["stages"]["static_control"])
    trace_binary = cell["binary"].get("trace_path", cell["binary"]["path"])
    trace_hash = cell["binary"].get("trace_sha256", cell["binary"]["sha256"])
    LIVE_VALIDATOR._validate_static_control_stage(
        root, cell, static, trace_binary, trace_hash
    )

    for field in ("argv", "cuobjdump_argv"):
        altered = deepcopy(static)
        altered[field] = list(altered[field])
        altered[field][-1] = "sm_80" if field == "argv" else "wrong-binary"
        with pytest.raises(ValueError, match="canonical"):
            LIVE_VALIDATOR._validate_static_control_stage(
                root, cell, altered, trace_binary, trace_hash
            )

    source = root / static["control"]
    original = json.loads(source.read_text(encoding="utf-8"))
    for mutation in ("arch", "toolkit_version", "parser_contract"):
        altered_doc = deepcopy(original)
        if mutation == "arch":
            altered_doc["arch"] = "sm_80"
        elif mutation == "toolkit_version":
            altered_doc["toolkit_version"] = "V13.3"
        else:
            altered_doc["parser_contract"]["control_layout"]["stall"] = "bits 1:4"
        control = tmp_path / f"{mutation}.json"
        control.write_text(json.dumps(altered_doc), encoding="utf-8")
        altered = deepcopy(static)
        altered["control"] = control.name
        altered["control_sha256"] = hashlib.sha256(control.read_bytes()).hexdigest()
        altered["argv"] = LIVE_VALIDATOR._expected_static_argv(cell, trace_binary, control.name)
        with pytest.raises(ValueError, match="arch/toolkit/parser|schema"):
            LIVE_VALIDATOR._validate_static_control_stage(
                tmp_path, cell, altered, trace_binary, trace_hash
            )


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


def test_content_bound_pair_does_not_require_replay_mtime_after_inputs(
    monkeypatch, tmp_path: Path
) -> None:
    expectation = VALIDATOR.REPLAY.ReplayExpectation(
        launches=(REPLAY_HELPERS._launch(),), carded_dram=False
    )
    first = tmp_path / "replay-1.json"
    second = tmp_path / "replay-2.json"
    dependency = tmp_path / "control.json"
    trace = tmp_path / "trace"
    replay = REPLAY_HELPERS._report(expectation)
    first.write_text(json.dumps(replay), encoding="utf-8")
    second.write_text(json.dumps(replay), encoding="utf-8")
    dependency.write_text("input", encoding="utf-8")
    os.utime(dependency, ns=(2_000_000_000, 2_000_000_000))
    os.utime(first, ns=(1_000_000_000, 1_000_000_000))
    os.utime(second, ns=(1_000_000_000, 1_000_000_000))
    monkeypatch.setattr(
        VALIDATOR,
        "validate_replay_document",
        lambda *args, **kwargs: ([{}], []),
    )

    class Binding:
        def check_pair(self, *args, **kwargs):
            return None

    rows: list[dict[str, object]] = []
    assert (
        VALIDATOR._validate_pair(
            "stale-bound",
            "retirement",
            first,
            second,
            [dependency],
            expectation,
            trace,
            rows,
            Binding(),
        )
        is None
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


def test_mixed_atomic_async_source_is_partitioned_exactly() -> None:
    expected = SimpleNamespace(
        global_load_sectors=0,
        global_atomic_sectors=8,
        async_source_sectors=32,
    )
    counters = {"l1": {"hits": 5, "misses": 5, "bypass_sectors": 30}}
    error, snapshot = _async_source_error(counters, expected, "mixed-fixture")
    assert error is None
    assert snapshot["status"] == "checked"
    assert snapshot["excluded_atomic_bypass_sectors"] == 8
    assert snapshot["passed"] is True
    assert snapshot["promotion_eligible"] is True


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
