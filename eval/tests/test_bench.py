from __future__ import annotations

import hashlib
import inspect
import json
import os
import shlex
import signal
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

from eval import bench
from eval.contracts.validation import (
    canonical_sha256,
    load_json_file_strict,
    validate_result,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_SOURCE = (
    ROOT
    / "sim/run/tests/fixtures/addr_carry_canary_bundle/run_manifest.v2.json"
)
RESULT_TEMPLATE = (
    Path(__file__).parent
    / "fixtures/runtime_record/sim_result_template.v2.json"
)


@dataclass(frozen=True, slots=True)
class _RuntimeInputs:
    repo_root: Path
    binary_path: Path
    manifest_path: Path
    result_document: dict[str, Any]
    result_bytes: bytes


class _SyntheticSpawn:
    def __init__(
        self,
        payloads: list[bytes],
        walls: list[int],
        *,
        return_codes: list[int] | None = None,
        stderrs: list[bytes] | None = None,
        timed_out: list[bool] | None = None,
        mutate: Callable[[int, bench._RetainedFile, bench._RetainedFile], None]
        | None = None,
    ) -> None:
        self.payloads = payloads
        self.walls = walls
        self.return_codes = return_codes or [0]
        self.stderrs = stderrs or [b""]
        self.timed_out = timed_out or [False]
        self.mutate = mutate
        self.calls = 0
        self.environments: list[dict[str, str]] = []
        self.timeouts: list[float] = []

    @staticmethod
    def _pick(values: list[Any], index: int) -> Any:
        return values[min(index, len(values) - 1)]

    def __call__(
        self,
        binary: bench._RetainedFile,
        manifest: bench._RetainedFile,
        *,
        working_directory: Path,
        environment: dict[str, str],
        timeout_seconds: float,
    ) -> bench._SpawnObservation:
        index = self.calls
        self.calls += 1
        self.environments.append(dict(environment))
        self.timeouts.append(timeout_seconds)
        assert working_directory.is_absolute()
        if self.mutate is not None:
            self.mutate(index, binary, manifest)
        start = 1_000_000_000 + index * 1_000_000
        wall = self._pick(self.walls, index)
        return bench._SpawnObservation(
            background_load=(0.25, 0.5, 0.75),
            monotonic_start_ns=start,
            monotonic_stop_ns=start + wall,
            user_cpu_ns=10_000 + index,
            system_cpu_ns=20_000 + index,
            peak_rss_kib=30_000 + index,
            return_code=self._pick(self.return_codes, index),
            timed_out=self._pick(self.timed_out, index),
            stdout=self._pick(self.payloads, index),
            stdout_complete=True,
            stderr=self._pick(self.stderrs, index),
            stderr_complete=True,
            capture_error=None,
        )


def _canonical_line(document: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _runtime_inputs(
    tmp_path: Path, *, clean_build: bool = False, fixture_scope: bool = False
) -> _RuntimeInputs:
    repo_root = tmp_path / "repo"
    binary_path = repo_root / bench.RELEASE_BINARY_RELATIVE
    binary_path.parent.mkdir(parents=True)
    binary_path.write_bytes(b"independent synthetic sim-run executable image\n")
    binary_path.chmod(0o700)

    manifest = load_json_file_strict(MANIFEST_SOURCE)
    assert isinstance(manifest, dict)
    if not fixture_scope:
        manifest["scope"] = "runtime-recorder-unit/non-authoritative:v1"
        launch_key = manifest["launches"][0]["launch_key"]
        launch_key["workload_id"] = "runtime-recorder-unit.workload"
        launch_key["matched_execution_id"] = "runtime-recorder-unit.execution"
    manifest_path = tmp_path / "run_manifest.v2.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_sha256 = bench._sha256_canonical(manifest)

    result = load_json_file_strict(RESULT_TEMPLATE)
    assert isinstance(result, dict)
    result = deepcopy(result)
    result["build_identity"]["binary_sha256"] = hashlib.sha256(
        binary_path.read_bytes()
    ).hexdigest()
    result["build_identity"]["sim_git_dirty"] = not clean_build
    result["build_identity"]["claimability"] = (
        {"status": "claimable", "basis": "clean_build"}
        if clean_build
        else {"status": "unclaimable", "reason": "dirty_build"}
    )
    result["configuration_identity"]["run_manifest_sha256"] = manifest_sha256
    result["configuration_identity"]["arch_card_sha256"] = manifest[
        "configuration_evidence"
    ]["arch_card_sha256"]
    result["configuration_identity"]["pchase_card_sha256"] = manifest[
        "configuration_evidence"
    ]["pchase_card_sha256"]
    result["arch_card_identity"]["card_sha256"] = manifest[
        "configuration_evidence"
    ]["arch_card_sha256"]
    result["arch_card_identity"]["sm_arch"] = manifest["architecture"]["arch_id"]
    result["arch_card_identity"]["sku"] = manifest["architecture"]["sku"]
    launch_key = deepcopy(manifest["launches"][0]["launch_key"])
    result["launches"][0]["launch_key"] = deepcopy(launch_key)
    result["diagnostics"]["per_launch"][0]["launch_key"] = deepcopy(launch_key)
    result["result_sha256"] = canonical_sha256(result, "result_sha256")
    validate_result(result)
    return _RuntimeInputs(
        repo_root=repo_root,
        binary_path=binary_path,
        manifest_path=manifest_path,
        result_document=result,
        result_bytes=_canonical_line(result),
    )


def _record(
    inputs: _RuntimeInputs,
    output_path: Path,
    spawn: _SyntheticSpawn,
    *,
    timeout_seconds: float = 12.5,
) -> dict[str, Any]:
    return bench._record_runtime(
        inputs.manifest_path,
        output_path,
        timeout_seconds=timeout_seconds,
        repo_root=inputs.repo_root,
        spawn=spawn,
    )


def _load(inputs: _RuntimeInputs, path: Path) -> dict[str, Any]:
    return bench._load_runtime_record(path, repo_root=inputs.repo_root)


def _reseal_runtime_record(document: dict[str, Any]) -> None:
    document["record_sha256"] = canonical_sha256(document, "record_sha256")


def _alternate_result(inputs: _RuntimeInputs) -> bytes:
    alternate = deepcopy(inputs.result_document)
    timing = alternate["diagnostics"]["per_launch"][0]
    timing["diagnostic_last_issue_cycles"] += 1
    alternate["result_sha256"] = canonical_sha256(alternate, "result_sha256")
    validate_result(alternate)
    return _canonical_line(alternate)


def test_public_record_boundary_has_no_binary_or_spawn_override() -> None:
    signature = inspect.signature(bench.record_runtime)

    assert tuple(signature.parameters) == (
        "manifest_path",
        "output_path",
        "timeout_seconds",
    )
    assert signature.parameters["timeout_seconds"].kind is inspect.Parameter.KEYWORD_ONLY


def test_invalid_manifest_argument_does_not_reserve_output_generation(
    tmp_path: Path,
) -> None:
    inputs = _runtime_inputs(tmp_path)
    invalid_manifest_path = tmp_path / os.fsdecode(b"run-manifest-\xff.json")
    inputs.manifest_path.rename(invalid_manifest_path)
    spawn = _SyntheticSpawn([inputs.result_bytes], [100] * 6)
    output = tmp_path / "runtime-record.v1.json"

    with pytest.raises(
        bench.RuntimeRecorderError,
        match="RUNTIME_INPUT_PATH_INVALID",
    ):
        bench._record_runtime(
            invalid_manifest_path,
            output,
            timeout_seconds=12.5,
            repo_root=inputs.repo_root,
            spawn=spawn,
        )

    assert spawn.calls == 0
    assert not output.exists()
    assert not output.with_name(f"{output.name}.samples").exists()


def test_records_one_warmup_five_timed_repeats_and_reloads(tmp_path: Path) -> None:
    inputs = _runtime_inputs(tmp_path)
    spawn = _SyntheticSpawn(
        [inputs.result_bytes],
        [1_000, 95, 95, 100, 105, 105],
    )
    output = tmp_path / "runtime-record.v1.json"

    record = _record(inputs, output, spawn)

    assert record["outcome"]["status"] == "MEASURED"
    assert spawn.calls == 6
    assert len(record["samples"]) == 6
    assert record["samples"][0]["kind"] == "warmup"
    assert record["samples"][0]["timed_repeat_index"] is None
    assert [sample["timed_repeat_index"] for sample in record["samples"][1:]] == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert record["outcome"]["summary"]["mean_wall_time_ns"] == 100
    assert record["outcome"]["summary"]["coefficient_of_variation"] == 0.05
    assert record["samples"][2]["user_cpu_ns"] == 10_002
    assert record["samples"][2]["system_cpu_ns"] == 20_002
    assert record["samples"][2]["peak_rss_kib"] == 30_002
    assert record["timing_protocol"]["timeout_ns"] == 12_500_000_000
    invocation = record["invocation"]
    assert invocation["executable_transport"] == (
        "proc_self_fd_retained_release_inode.v1"
    )
    assert invocation["manifest_transport"] == (
        "inherited_retained_manifest_and_evidence_base_fds.v1"
    )
    assert len(invocation["argv"]) == 2
    descriptor_fields = invocation["argv"][1].split(":")
    assert descriptor_fields[0] == "--retained-manifest-fds-v1"
    assert descriptor_fields[1].isdigit()
    assert descriptor_fields[2].isdigit()
    assert descriptor_fields[1] != descriptor_fields[2]
    assert bytes.fromhex(descriptor_fields[3]).decode("utf-8") == (
        inputs.manifest_path.name
    )
    assert record["accuracy_binding"] == {
        "status": "unavailable",
        "reason": "production_matched_label_crosswalk_authority_unavailable",
    }
    assert record["claimability"] == {
        "status": "unclaimable",
        "reason": "DIRTY_BUILD_UNCLAIMABLE",
    }
    assert all(
        environment
        == {
            "LANG": "C",
            "LC_ALL": "C",
            "NO_COLOR": "1",
            "RUST_BACKTRACE": "0",
            "TZ": "UTC",
        }
        for environment in spawn.environments
    )
    assert _load(inputs,output) == record


def test_throughput_uses_selected_replay_count_not_whole_bundle_count(
    tmp_path: Path,
) -> None:
    original = _runtime_inputs(tmp_path)
    result = deepcopy(original.result_document)
    result["trace_identity"]["record_count"] = 108
    result["trace_identity"]["instruction_count"] = 108
    result["result_sha256"] = canonical_sha256(result, "result_sha256")
    validate_result(result)
    inputs = _RuntimeInputs(
        repo_root=original.repo_root,
        binary_path=original.binary_path,
        manifest_path=original.manifest_path,
        result_document=result,
        result_bytes=_canonical_line(result),
    )

    output = tmp_path / "subset-runtime-record.json"
    record = _record(
        inputs,
        output,
        _SyntheticSpawn([inputs.result_bytes], [100] * 6),
    )

    binding = record["sim_result_binding"]
    assert binding["trace_identity"]["record_count"] == 108
    assert binding["selected_replay_counts"] == {
        "basis": "aggregate_exact_instructions_with_checked_issue_record_equality.v1",
        "record_count": 54,
        "instruction_count": 54,
    }
    summary = record["outcome"]["summary"]
    assert summary["throughput_basis"] == (
        "selected_replay_count_divided_by_median_external_wall_time"
    )
    assert summary["records_per_second"] == 540_000_000.0
    assert summary["instructions_per_second"] == 540_000_000.0
    assert _load(inputs, output) == record


def test_clean_build_still_lacks_workload_tier_authority(tmp_path: Path) -> None:
    inputs = _runtime_inputs(tmp_path, clean_build=True)
    spawn = _SyntheticSpawn([inputs.result_bytes], [100] * 6)

    record = _record(inputs, tmp_path / "clean-record.json", spawn)

    assert record["outcome"]["status"] == "MEASURED"
    assert record["claimability"] == {
        "status": "unclaimable",
        "reason": "WORKLOAD_TIER_AUTHORITY_UNAVAILABLE",
    }


def test_fixture_manifest_is_always_typed_synthetic(tmp_path: Path) -> None:
    inputs = _runtime_inputs(tmp_path, clean_build=True, fixture_scope=True)
    spawn = _SyntheticSpawn([inputs.result_bytes], [100] * 6)

    record = _record(inputs, tmp_path / "fixture-record.json", spawn)

    assert record["claimability"] == {
        "status": "unclaimable",
        "reason": "SYNTHETIC_FIXTURE",
    }


def test_noise_escalates_to_nine_timed_repeats_and_fails_closed(
    tmp_path: Path,
) -> None:
    inputs = _runtime_inputs(tmp_path)
    spawn = _SyntheticSpawn(
        [inputs.result_bytes],
        [100, 100, 200, 100, 200, 100, 200, 100, 200, 100],
    )

    record = _record(inputs, tmp_path / "noisy-record.json", spawn)

    assert spawn.calls == 10
    assert len(record["samples"]) == 10
    assert record["outcome"]["status"] == "RUNTIME_NOISY"
    assert record["outcome"]["summary"]["timed_repeat_count"] == 9
    assert record["outcome"]["summary"]["coefficient_of_variation"] > 0.05
    assert _load(inputs,tmp_path / "noisy-record.json") == record


def test_semantic_validator_rejects_unjustified_additional_repeats(
    tmp_path: Path,
) -> None:
    inputs = _runtime_inputs(tmp_path)
    record = _record(
        inputs,
        tmp_path / "noisy-record.json",
        _SyntheticSpawn(
            [inputs.result_bytes],
            [100, 100, 200, 100, 200, 100, 200, 100, 200, 100],
        ),
    )
    for sample in record["samples"][1:]:
        sample["monotonic_stop_ns"] = sample["monotonic_start_ns"] + 100
        sample["wall_time_ns"] = 100
    binding = record["sim_result_binding"]
    record["outcome"] = {
        "status": "MEASURED",
        "summary": bench._summary(
            [100] * 9,
            record_count=binding["selected_replay_counts"]["record_count"],
            instruction_count=binding["selected_replay_counts"]["instruction_count"],
        ),
    }
    _reseal_runtime_record(record)

    with pytest.raises(
        bench.RuntimeRecorderError,
        match="additional runtime repeats lack a noisy initial sample set",
    ):
        bench.validate_runtime_record(record)


def test_semantic_validator_rejects_overlapping_monotonic_samples(
    tmp_path: Path,
) -> None:
    inputs = _runtime_inputs(tmp_path)
    record = _record(
        inputs,
        tmp_path / "runtime-record.json",
        _SyntheticSpawn([inputs.result_bytes], [100] * 6),
    )
    prior = record["samples"][1]
    sample = record["samples"][2]
    sample["monotonic_start_ns"] = prior["monotonic_stop_ns"] - 1
    sample["monotonic_stop_ns"] = sample["monotonic_start_ns"] + sample["wall_time_ns"]
    _reseal_runtime_record(record)

    with pytest.raises(bench.RuntimeRecorderError, match="overlap"):
        bench.validate_runtime_record(record)


def test_nondeterministic_result_is_typed_and_unbound(tmp_path: Path) -> None:
    inputs = _runtime_inputs(tmp_path)
    spawn = _SyntheticSpawn(
        [inputs.result_bytes, _alternate_result(inputs)],
        [100] * 6,
    )
    output = tmp_path / "nondeterministic-record.json"

    record = _record(inputs, output, spawn)

    assert spawn.calls == 2
    assert record["outcome"]["status"] == "NONDETERMINISTIC_RESULT"
    assert record["sim_result_binding"] == {
        "status": "unavailable",
        "reason": "nondeterministic_result",
    }
    assert record["outcome"]["summary"] is None
    assert _load(inputs,output) == record


def test_semantic_validator_rejects_execution_failure_after_complete_protocol(
    tmp_path: Path,
) -> None:
    inputs = _runtime_inputs(tmp_path)
    record = _record(
        inputs,
        tmp_path / "runtime-record.json",
        _SyntheticSpawn([inputs.result_bytes], [100] * 6),
    )
    record["outcome"] = {
        "status": "RUNTIME_EXECUTION_FAILED",
        "detail": "forged terminal status",
        "summary": None,
    }
    _reseal_runtime_record(record)

    with pytest.raises(
        bench.RuntimeRecorderError,
        match="complete successful runtime protocol",
    ):
        bench.validate_runtime_record(record)


def test_semantic_validator_requires_first_nondeterministic_divergence(
    tmp_path: Path,
) -> None:
    inputs = _runtime_inputs(tmp_path)
    record = _record(
        inputs,
        tmp_path / "nondeterministic-record.json",
        _SyntheticSpawn(
            [inputs.result_bytes, _alternate_result(inputs)],
            [100] * 6,
        ),
    )
    extra = deepcopy(record["samples"][-1])
    extra["sample_index"] = 2
    extra["timed_repeat_index"] = 1
    extra["monotonic_start_ns"] = record["samples"][-1]["monotonic_stop_ns"]
    extra["monotonic_stop_ns"] = extra["monotonic_start_ns"] + extra["wall_time_ns"]
    record["samples"].append(extra)
    _reseal_runtime_record(record)

    with pytest.raises(
        bench.RuntimeRecorderError,
        match="first differing result",
    ):
        bench.validate_runtime_record(record)


def test_loader_revalidates_every_nondeterministic_result_variant(
    tmp_path: Path,
) -> None:
    inputs = _runtime_inputs(tmp_path)
    output = tmp_path / "nondeterministic-record.json"
    record = _record(
        inputs,
        output,
        _SyntheticSpawn(
            [inputs.result_bytes, _alternate_result(inputs)],
            [100] * 6,
        ),
    )
    artifact = record["samples"][1]["stdout"]
    invalid = b"not a sim_result.v2\n"
    (tmp_path / artifact["path"]).write_bytes(invalid)
    artifact["size_bytes"] = len(invalid)
    artifact["sha256"] = hashlib.sha256(invalid).hexdigest()
    _reseal_runtime_record(record)
    output.write_bytes(_canonical_line(record))

    with pytest.raises(bench.RuntimeRecorderError, match="RUNTIME_RESULT_INVALID"):
        _load(inputs, output)


def test_child_failure_retains_streams_and_typed_record(tmp_path: Path) -> None:
    inputs = _runtime_inputs(tmp_path)
    spawn = _SyntheticSpawn(
        [b"not a result\n"],
        [100],
        return_codes=[7],
        stderrs=[b"synthetic child failure\n"],
    )
    output = tmp_path / "failed-record.json"

    record = _record(inputs, output, spawn)

    assert spawn.calls == 1
    assert record["outcome"]["status"] == "RUNTIME_EXECUTION_FAILED"
    assert record["samples"][0]["termination"] == {"kind": "exit", "exit_code": 7}
    assert record["samples"][0]["stderr"]["size_bytes"] == len(
        b"synthetic child failure\n"
    )
    assert record["sim_result_binding"] == {
        "status": "unavailable",
        "reason": "no_successful_result",
    }
    assert _load(inputs,output) == record


def test_timeout_is_typed_in_retained_record(tmp_path: Path) -> None:
    inputs = _runtime_inputs(tmp_path)
    spawn = _SyntheticSpawn(
        [b""],
        [100],
        return_codes=[-9],
        timed_out=[True],
    )

    record = _record(inputs, tmp_path / "timeout-record.json", spawn)

    assert record["outcome"]["status"] == "RUNTIME_EXECUTION_FAILED"
    assert record["samples"][0]["timed_out"] is True
    assert record["samples"][0]["termination"] == {"kind": "signal", "signal": 9}


@pytest.mark.parametrize("drift_role", ["binary", "manifest"])
def test_input_drift_during_repeat_fails_closed(
    tmp_path: Path, drift_role: str
) -> None:
    inputs = _runtime_inputs(tmp_path)

    def mutate(
        index: int, binary: bench._RetainedFile, manifest: bench._RetainedFile
    ) -> None:
        if index != 0:
            return
        target = binary.path if drift_role == "binary" else manifest.path
        with target.open("r+b") as stream:
            first = stream.read(1)
            stream.seek(0)
            stream.write(b"X" if first != b"X" else b"Y")
            stream.flush()

    spawn = _SyntheticSpawn([inputs.result_bytes], [100], mutate=mutate)

    record = _record(inputs, tmp_path / f"{drift_role}-drift.json", spawn)

    assert record["outcome"]["status"] == "RUNTIME_EXECUTION_FAILED"
    assert "RUNTIME_INPUT_DRIFT" in record["outcome"]["detail"]


@pytest.mark.parametrize(
    ("mutation_index", "walls"),
    (
        (5, [100] * 6),
        (9, [100, 100, 200, 100, 200, 100, 200, 100, 200, 100]),
    ),
    ids=("terminal-initial", "terminal-noise-extension"),
)
def test_terminal_input_drift_publishes_typed_record(
    tmp_path: Path,
    mutation_index: int,
    walls: list[int],
) -> None:
    inputs = _runtime_inputs(tmp_path)
    output = tmp_path / f"terminal-drift-{mutation_index}.json"

    def mutate_manifest(
        index: int,
        _binary: bench._RetainedFile,
        manifest: bench._RetainedFile,
    ) -> None:
        if index == mutation_index:
            with manifest.path.open("r+b") as stream:
                stream.write(b"[")
                stream.flush()
                os.fsync(stream.fileno())

    spawn = _SyntheticSpawn(
        [inputs.result_bytes],
        walls,
        mutate=mutate_manifest,
    )

    record = _record(inputs, output, spawn)

    assert spawn.calls == mutation_index + 1
    assert len(record["samples"]) == mutation_index + 1
    assert record["outcome"]["status"] == "RUNTIME_EXECUTION_FAILED"
    assert record["outcome"]["failure_code"] == "RUNTIME_INPUT_DRIFT"
    assert record["outcome"]["detail"].startswith("RUNTIME_INPUT_DRIFT:")
    assert output.is_file()
    assert (tmp_path / f"{output.name}.samples").is_dir()
    with pytest.raises(
        bench.RuntimeRecorderError,
        match="complete successful runtime protocol",
    ):
        bench.validate_runtime_record(record)
    assert bench._load_runtime_record(
        output,
        repo_root=inputs.repo_root,
        permit_recorded_input_drift=True,
    ) == record


@pytest.mark.parametrize(
    "walls",
    (
        [100] * 6,
        [100, 100, 200, 100, 200, 100, 200, 100, 200, 100],
    ),
    ids=("complete-initial", "complete-noise-extension"),
)
def test_terminal_input_drift_detail_cannot_self_authorize(
    tmp_path: Path,
    walls: list[int],
) -> None:
    inputs = _runtime_inputs(tmp_path)
    record = _record(
        inputs,
        tmp_path / "completed-runtime.json",
        _SyntheticSpawn([inputs.result_bytes], walls),
    )
    record["outcome"] = {
        "status": "RUNTIME_EXECUTION_FAILED",
        "detail": "RUNTIME_INPUT_DRIFT: forged free-form exemption",
        "summary": None,
    }
    _reseal_runtime_record(record)

    with pytest.raises(
        bench.RuntimeRecorderError,
        match="matching typed failure code",
    ):
        bench.validate_runtime_record(
            record,
            permit_terminal_input_drift=True,
        )


def test_manifest_read_is_revalidated_after_initial_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _runtime_inputs(tmp_path)
    output = tmp_path / "manifest-read-drift.json"
    real_read_fd = bench._read_fd
    mutated = False

    def mutate_before_read(fd: int, size: int, *, role: str) -> bytes:
        nonlocal mutated
        if role == "run manifest" and not mutated:
            mutated = True
            with inputs.manifest_path.open("r+b") as stream:
                stream.write(b"[")
                stream.flush()
                os.fsync(stream.fileno())
        return real_read_fd(fd, size, role=role)

    monkeypatch.setattr(bench, "_read_fd", mutate_before_read)

    with pytest.raises(bench.RuntimeRecorderError, match="RUNTIME_INPUT_DRIFT"):
        _record(inputs, output, _SyntheticSpawn([inputs.result_bytes], [100]))

    assert mutated is True
    assert not output.exists()


def test_input_drift_record_replays_sample_artifacts_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _runtime_inputs(tmp_path)
    output = tmp_path / "drift-with-mutated-sample.json"

    def mutate_manifest(
        index: int, _binary: bench._RetainedFile, manifest: bench._RetainedFile
    ) -> None:
        if index == 0:
            with manifest.path.open("r+b") as stream:
                stream.write(b"[")
                stream.flush()
                os.fsync(stream.fileno())

    real_write_stream = bench._Publisher.write_stream
    mutated = False

    def write_then_mutate(
        publisher: bench._Publisher, **kwargs: Any
    ) -> dict[str, Any]:
        nonlocal mutated
        artifact = real_write_stream(publisher, **kwargs)
        if kwargs["stream_name"] == "stdout" and not mutated:
            mutated = True
            artifact_path = publisher.output_path.parent / artifact["path"]
            with artifact_path.open("r+b") as stream:
                first = stream.read(1)
                stream.seek(0)
                stream.write(b"X" if first != b"X" else b"Y")
                stream.flush()
                os.fsync(stream.fileno())
        return artifact

    monkeypatch.setattr(bench._Publisher, "write_stream", write_then_mutate)

    with pytest.raises(
        bench.RuntimeRecorderError, match="RUNTIME_SAMPLE_IDENTITY_MISMATCH"
    ):
        _record(
            inputs,
            output,
            _SyntheticSpawn(
                [inputs.result_bytes],
                [100],
                mutate=mutate_manifest,
            ),
        )

    assert mutated is True


def test_retained_input_rechecks_identity_after_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "retained-input"
    path.write_bytes(b"original-input")
    retained = bench._open_retained_file(
        path,
        role="synthetic retained input",
        maximum_bytes=bench.MAX_MANIFEST_BYTES,
    )
    original_hash_fd = bench._hash_fd

    def hash_then_mutate(fd: int, size: int, *, role: str) -> str:
        digest = original_hash_fd(fd, size, role=role)
        with path.open("r+b") as stream:
            stream.write(b"X")
            stream.flush()
            os.fsync(stream.fileno())
        return digest

    monkeypatch.setattr(bench, "_hash_fd", hash_then_mutate)
    try:
        with pytest.raises(bench.RuntimeRecorderError, match="RUNTIME_INPUT_DRIFT"):
            retained.revalidate(role="synthetic retained input")
    finally:
        retained.close()


def test_loader_rejects_tampered_sample_bytes(tmp_path: Path) -> None:
    inputs = _runtime_inputs(tmp_path)
    output = tmp_path / "runtime-record.json"
    record = _record(
        inputs,
        output,
        _SyntheticSpawn([inputs.result_bytes], [100] * 6),
    )
    artifact_path = tmp_path / record["samples"][0]["stdout"]["path"]
    payload = artifact_path.read_bytes()
    artifact_path.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])

    with pytest.raises(
        bench.RuntimeRecorderError, match="RUNTIME_SAMPLE_IDENTITY_MISMATCH"
    ):
        _load(inputs,output)


def test_loader_rejects_oversized_sample_before_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _runtime_inputs(tmp_path)
    output = tmp_path / "runtime-record.json"
    record = _record(
        inputs,
        output,
        _SyntheticSpawn([inputs.result_bytes], [100] * 6),
    )
    artifact_path = tmp_path / record["samples"][0]["stdout"]["path"]
    admitted_size = artifact_path.stat().st_size
    with artifact_path.open("ab") as stream:
        stream.write(b"X")
        stream.flush()
        os.fsync(stream.fileno())

    original_hash_fd = bench._hash_fd
    artifact_hash_reached = False

    def reject_oversized_hash(fd: int, size: int, *, role: str) -> str:
        nonlocal artifact_hash_reached
        if role == "runtime stdout artifact":
            artifact_hash_reached = True
            assert size <= admitted_size, "oversized artifact reached content hashing"
        return original_hash_fd(fd, size, role=role)

    monkeypatch.setattr(bench, "MAX_RESULT_STREAM_BYTES", admitted_size)
    monkeypatch.setattr(bench, "_hash_fd", reject_oversized_hash)

    with pytest.raises(
        bench.RuntimeRecorderError,
        match="RUNTIME_SAMPLE_IDENTITY_MISMATCH",
    ):
        _load(inputs, output)

    assert artifact_hash_reached is False


def test_retained_artifact_rechecks_identity_after_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "sample.stdout.json"
    path.write_bytes(b"original-artifact")
    directory_fd = os.open(
        tmp_path,
        os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0),
    )
    artifact = bench._open_relative_artifact(
        directory_fd,
        path.name,
        role="synthetic retained artifact",
        maximum_bytes=bench.MAX_RESULT_STREAM_BYTES,
    )
    original_hash_fd = bench._hash_fd

    def hash_then_mutate(fd: int, size: int, *, role: str) -> str:
        digest = original_hash_fd(fd, size, role=role)
        with path.open("r+b") as stream:
            stream.write(b"X")
            stream.flush()
            os.fsync(stream.fileno())
        return digest

    monkeypatch.setattr(bench, "_hash_fd", hash_then_mutate)
    try:
        with pytest.raises(
            bench.RuntimeRecorderError,
            match="RUNTIME_SAMPLE_IDENTITY_MISMATCH",
        ):
            artifact.revalidate(directory_fd, role="synthetic retained artifact")
    finally:
        artifact.close()
        os.close(directory_fd)


@pytest.mark.parametrize("mutation", ["hardlink", "rename"])
def test_loader_revalidates_retained_artifacts_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    inputs = _runtime_inputs(tmp_path)
    output = tmp_path / "runtime-record.json"
    record = _record(
        inputs,
        output,
        _SyntheticSpawn([inputs.result_bytes], [100] * 6),
    )
    artifact_path = tmp_path / record["samples"][0]["stdout"]["path"]
    original = bench._validate_result_binding
    changed = False

    def mutate_after_validation(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], str]:
        nonlocal changed
        result = original(*args, **kwargs)
        if not changed:
            changed = True
            if mutation == "hardlink":
                (tmp_path / "late-artifact-alias").hardlink_to(artifact_path)
            else:
                artifact_path.rename(tmp_path / "late-moved-artifact")
        return result

    monkeypatch.setattr(bench, "_validate_result_binding", mutate_after_validation)

    with pytest.raises(
        bench.RuntimeRecorderError,
        match="RUNTIME_SAMPLE_IDENTITY_MISMATCH",
    ):
        _load(inputs, output)


def test_public_loader_rejects_a_self_declared_repository_root(tmp_path: Path) -> None:
    inputs = _runtime_inputs(tmp_path)
    output = tmp_path / "runtime-record.json"
    record = _record(
        inputs,
        output,
        _SyntheticSpawn([inputs.result_bytes], [100] * 6),
    )

    assert _load(inputs, output) == record
    with pytest.raises(
        bench.RuntimeRecorderError,
        match="not bound to this repository root",
    ):
        bench.load_runtime_record(output)


def test_semantic_validator_rejects_resealed_summary_and_authority_forgery(
    tmp_path: Path,
) -> None:
    inputs = _runtime_inputs(tmp_path, clean_build=True)
    record = _record(
        inputs,
        tmp_path / "runtime-record.json",
        _SyntheticSpawn([inputs.result_bytes], [100] * 6),
    )

    wrong_summary = deepcopy(record)
    wrong_summary["outcome"]["summary"]["median_wall_time_ns"] = 101
    _reseal_runtime_record(wrong_summary)
    with pytest.raises(bench.RuntimeRecorderError, match="not derived from raw samples"):
        bench.validate_runtime_record(wrong_summary)

    forged_claim = deepcopy(record)
    forged_claim["claimability"] = {
        "status": "claimable",
        "basis": "clean_committed_release_and_admitted_workload_tier",
    }
    _reseal_runtime_record(forged_claim)
    with pytest.raises(
        bench.RuntimeRecorderError, match="claimable runtime authority is not installed"
    ):
        bench.validate_runtime_record(forged_claim)

    forged_descriptor = deepcopy(record)
    descriptor_fields = forged_descriptor["invocation"]["argv"][1].split(":")
    descriptor_fields[1] = "01"
    forged_descriptor["invocation"]["argv"][1] = ":".join(descriptor_fields)
    _reseal_runtime_record(forged_descriptor)
    with pytest.raises(
        bench.RuntimeRecorderError,
        match="runtime descriptor argv is not canonical",
    ):
        bench.validate_runtime_record(forged_descriptor)

    forged_name_encoding = deepcopy(record)
    descriptor_fields = forged_name_encoding["invocation"]["argv"][1].split(":")
    descriptor_fields[3] = descriptor_fields[3].upper()
    forged_name_encoding["invocation"]["argv"][1] = ":".join(descriptor_fields)
    _reseal_runtime_record(forged_name_encoding)
    with pytest.raises(
        bench.RuntimeRecorderError,
        match="runtime descriptor argv is not canonical",
    ):
        bench.validate_runtime_record(forged_name_encoding)

    forged_family = deepcopy(record)
    forged_family["workload_identity"]["workload_family"] = {
        "status": "available",
        "value": "forged-family",
    }
    _reseal_runtime_record(forged_family)
    with pytest.raises(
        bench.RuntimeRecorderError,
        match="workload-family authority is not installed",
    ):
        bench.validate_runtime_record(forged_family)

    forged_accuracy = deepcopy(record)
    forged_accuracy["accuracy_binding"] = {
        "status": "bound",
        "report_id": "forged",
        "report_sha256": "1" * 64,
        "file_sha256": "2" * 64,
    }
    _reseal_runtime_record(forged_accuracy)
    with pytest.raises(
        bench.RuntimeRecorderError, match="accuracy-link authority is not installed"
    ):
        bench.validate_runtime_record(forged_accuracy)


def test_output_is_exclusive_and_existing_record_is_not_replaced(tmp_path: Path) -> None:
    inputs = _runtime_inputs(tmp_path)
    output = tmp_path / "runtime-record.json"
    first = _record(
        inputs,
        output,
        _SyntheticSpawn([inputs.result_bytes], [100] * 6),
    )
    original = output.read_bytes()

    with pytest.raises(bench.RuntimeRecorderError, match="RUNTIME_OUTPUT_EXISTS"):
        _record(
            inputs,
            output,
            _SyntheticSpawn([inputs.result_bytes], [100] * 6),
        )

    assert output.read_bytes() == original
    assert _load(inputs,output) == first


def test_spawn_failure_publishes_typed_zero_sample_record(tmp_path: Path) -> None:
    inputs = _runtime_inputs(tmp_path)

    def fail_spawn(*_args: Any, **_kwargs: Any) -> bench._SpawnObservation:
        raise bench.RuntimeRecorderError(
            "RUNTIME_SPAWN_FAILED", "synthetic spawn refusal"
        )

    output = tmp_path / "spawn-failure.json"
    record = bench._record_runtime(
        inputs.manifest_path,
        output,
        timeout_seconds=1.0,
        repo_root=inputs.repo_root,
        spawn=fail_spawn,
    )

    assert record["samples"] == []
    assert record["outcome"] == {
        "status": "RUNTIME_EXECUTION_FAILED",
        "failure_code": "RUNTIME_SPAWN_FAILED",
        "detail": "RUNTIME_SPAWN_FAILED: synthetic spawn refusal",
        "summary": None,
    }
    assert _load(inputs,output) == record


@pytest.mark.parametrize("drift_role", ["binary", "manifest"])
def test_loader_rejects_postpublication_input_drift(
    tmp_path: Path, drift_role: str
) -> None:
    inputs = _runtime_inputs(tmp_path)
    output = tmp_path / "runtime-record.json"
    _record(
        inputs,
        output,
        _SyntheticSpawn([inputs.result_bytes], [100] * 6),
    )
    target = inputs.binary_path if drift_role == "binary" else inputs.manifest_path
    with target.open("r+b") as stream:
        first = stream.read(1)
        stream.seek(0)
        stream.write(b"X" if first != b"X" else b"Y")
        stream.flush()

    with pytest.raises(
        bench.RuntimeRecorderError, match="RUNTIME_INPUT_IDENTITY_MISMATCH"
    ):
        _load(inputs,output)


def test_final_symlinks_are_rejected_for_all_fixed_inputs(tmp_path: Path) -> None:
    inputs = _runtime_inputs(tmp_path)
    manifest_link = tmp_path / "manifest-link.json"
    manifest_link.symlink_to(inputs.manifest_path)

    with pytest.raises(bench.RuntimeRecorderError, match="RUNTIME_INPUT_OPEN_FAILED"):
        bench._record_runtime(
            manifest_link,
            tmp_path / "manifest-symlink-record.json",
            timeout_seconds=1.0,
            repo_root=inputs.repo_root,
            spawn=_SyntheticSpawn([inputs.result_bytes], [100] * 6),
        )

    output_parent = tmp_path / "real-output"
    output_parent.mkdir()
    output_link = tmp_path / "output-link"
    output_link.symlink_to(output_parent, target_is_directory=True)
    with pytest.raises(
        bench.RuntimeRecorderError, match="RUNTIME_OUTPUT_PARENT_INVALID"
    ):
        _record(
            inputs,
            output_link / "runtime-record.json",
            _SyntheticSpawn([inputs.result_bytes], [100] * 6),
        )


def test_hardlink_aliases_are_rejected_for_inputs_and_artifacts(tmp_path: Path) -> None:
    inputs = _runtime_inputs(tmp_path)
    manifest_alias = tmp_path / "manifest-alias.json"
    manifest_alias.hardlink_to(inputs.manifest_path)
    with pytest.raises(
        bench.RuntimeRecorderError, match="must not have hardlink aliases"
    ):
        bench._record_runtime(
            inputs.manifest_path,
            tmp_path / "hardlinked-input-record.json",
            timeout_seconds=1.0,
            repo_root=inputs.repo_root,
            spawn=_SyntheticSpawn([inputs.result_bytes], [100] * 6),
        )
    manifest_alias.unlink()

    output = tmp_path / "runtime-record.json"
    record = _record(
        inputs,
        output,
        _SyntheticSpawn([inputs.result_bytes], [100] * 6),
    )
    artifact = tmp_path / record["samples"][0]["stdout"]["path"]
    artifact_alias = tmp_path / "artifact-alias"
    artifact_alias.hardlink_to(artifact)
    with pytest.raises(bench.RuntimeRecorderError, match="hardlink aliases"):
        _load(inputs,output)


def test_cargo_like_hardlinked_release_binary_is_admitted(tmp_path: Path) -> None:
    inputs = _runtime_inputs(tmp_path)
    binary_alias = inputs.binary_path.with_name("sim-run.cargo-hardlink")
    binary_alias.hardlink_to(inputs.binary_path)
    output = tmp_path / "runtime-record.json"

    record = _record(
        inputs,
        output,
        _SyntheticSpawn([inputs.result_bytes], [100] * 6),
    )

    assert _load(inputs, output) == record


def test_symlinked_ancestor_is_rejected_for_retained_inputs(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    retained_path = real_parent / "manifest.json"
    retained_path.write_text("{}\n", encoding="utf-8")
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(bench.RuntimeRecorderError, match="RUNTIME_INPUT_OPEN_FAILED"):
        bench._open_retained_file(
            linked_parent / retained_path.name,
            role="synthetic manifest",
            maximum_bytes=bench.MAX_MANIFEST_BYTES,
        )


def _retained_shell(tmp_path: Path) -> tuple[bench._RetainedFile, Path]:
    binary_path = tmp_path / "sim-run"
    binary_path.write_text(
        "#!/bin/sh\n"
        "test \"$#\" -eq 1 || exit 97\n"
        "case \"$1\" in --retained-manifest-fds-v1:*) ;; *) exit 96 ;; esac\n"
        "descriptor_fields=${1#--retained-manifest-fds-v1:}\n"
        "manifest_fd=${descriptor_fields%%:*}\n"
        "exec /bin/sh \"/proc/self/fd/$manifest_fd\"\n",
        encoding="utf-8",
    )
    binary_path.chmod(0o700)
    return (
        bench._open_retained_file(
            binary_path,
            role="synthetic shell",
            maximum_bytes=bench.MAX_BINARY_BYTES,
            require_executable=True,
        ),
        binary_path,
    )


def test_spawn_once_validates_manifest_argument_before_pipe_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary, _ = _retained_shell(tmp_path)
    script = tmp_path / os.fsdecode(b"manifest-\xff")
    script.write_text("exit 0\n", encoding="utf-8")
    manifest = bench._open_retained_file(
        script,
        role="non-UTF-8 manifest name",
        maximum_bytes=bench.MAX_MANIFEST_BYTES,
    )
    _, environment, _ = bench._environment()
    pipe_calls = 0

    def reject_pipe_allocation(_flags: int) -> tuple[int, int]:
        nonlocal pipe_calls
        pipe_calls += 1
        raise AssertionError("invalid argument reached pipe allocation")

    monkeypatch.setattr(bench.os, "pipe2", reject_pipe_allocation)
    try:
        with pytest.raises(
            bench.RuntimeRecorderError,
            match="RUNTIME_INPUT_PATH_INVALID",
        ):
            bench._spawn_once(
                binary,
                manifest,
                working_directory=tmp_path,
                environment=environment,
                timeout_seconds=2.0,
            )
    finally:
        manifest.close()
        binary.close()

    assert pipe_calls == 0


def test_spawn_once_uses_exact_child_wait4_resources(tmp_path: Path) -> None:
    binary, _ = _retained_shell(tmp_path)
    script = tmp_path / "child-script"
    script.write_text(
        "printf child-stdout\nprintf child-stderr >&2\nexit 7\n",
        encoding="utf-8",
    )
    manifest = bench._open_retained_file(
        script,
        role="synthetic child script",
        maximum_bytes=bench.MAX_MANIFEST_BYTES,
    )
    _, environment, _ = bench._environment()
    try:
        observation = bench._spawn_once(
            binary,
            manifest,
            working_directory=tmp_path,
            environment=environment,
            timeout_seconds=2.0,
        )
    finally:
        manifest.close()
        binary.close()

    assert observation.return_code == 7
    assert observation.stdout == b"child-stdout"
    assert observation.stderr == b"child-stderr"
    assert observation.timed_out is False
    assert observation.monotonic_stop_ns > observation.monotonic_start_ns
    assert observation.user_cpu_ns >= 0
    assert observation.system_cpu_ns >= 0
    assert observation.peak_rss_kib >= 0


def test_spawn_once_consumes_retained_manifest_after_ancestor_swap(
    tmp_path: Path,
) -> None:
    binary, _ = _retained_shell(tmp_path)
    manifest_parent = tmp_path / "manifest-parent"
    manifest_parent.mkdir()
    script = manifest_parent / "child-script"
    script.write_text("printf retained-manifest\n", encoding="utf-8")
    manifest = bench._open_retained_file(
        script,
        role="synthetic retained script",
        maximum_bytes=bench.MAX_MANIFEST_BYTES,
    )
    displaced_parent = tmp_path / "displaced-parent"
    manifest_parent.rename(displaced_parent)
    manifest_parent.mkdir()
    (manifest_parent / script.name).write_text(
        "printf substituted-pathname\n",
        encoding="utf-8",
    )
    _, environment, _ = bench._environment()
    try:
        observation = bench._spawn_once(
            binary,
            manifest,
            working_directory=tmp_path,
            environment=environment,
            timeout_seconds=2.0,
        )
    finally:
        manifest.close()
        binary.close()

    assert observation.return_code == 0
    assert observation.stdout == b"retained-manifest"
    assert observation.stderr == b""


def test_spawn_once_kills_exact_process_group_at_deadline(tmp_path: Path) -> None:
    binary, _ = _retained_shell(tmp_path)
    script = tmp_path / "timeout-script"
    script.write_text("exec /usr/bin/sleep 2\n", encoding="utf-8")
    manifest = bench._open_retained_file(
        script,
        role="synthetic timeout script",
        maximum_bytes=bench.MAX_MANIFEST_BYTES,
    )
    _, environment, _ = bench._environment()
    try:
        observation = bench._spawn_once(
            binary,
            manifest,
            working_directory=tmp_path,
            environment=environment,
            timeout_seconds=0.05,
        )
    finally:
        manifest.close()
        binary.close()

    assert observation.timed_out is True
    assert observation.return_code < 0
    assert observation.monotonic_stop_ns - observation.monotonic_start_ns < 1_000_000_000


def test_spawn_once_retries_interrupted_group_termination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary, _ = _retained_shell(tmp_path)
    script = tmp_path / "interrupted-termination-script"
    script.write_text("exec /usr/bin/sleep 30\n", encoding="utf-8")
    manifest = bench._open_retained_file(
        script,
        role="synthetic interrupted termination script",
        maximum_bytes=bench.MAX_MANIFEST_BYTES,
    )
    _, environment, _ = bench._environment()
    real_kill_process_group = bench._kill_process_group
    kill_calls = 0

    def interrupt_first_termination(pid: int) -> None:
        nonlocal kill_calls
        kill_calls += 1
        if kill_calls == 1:
            raise KeyboardInterrupt("synthetic pre-signal interruption")
        real_kill_process_group(pid)

    monkeypatch.setattr(
        bench,
        "_kill_process_group",
        interrupt_first_termination,
    )
    started = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt, match="pre-signal interruption"):
            bench._spawn_once(
                binary,
                manifest,
                working_directory=tmp_path,
                environment=environment,
                timeout_seconds=0.02,
            )
    finally:
        manifest.close()
        binary.close()

    assert kill_calls == 2
    assert time.monotonic() - started < 1.0


def test_spawn_once_kills_successful_child_descendants(tmp_path: Path) -> None:
    binary, _ = _retained_shell(tmp_path)
    pid_path = tmp_path / "descendant.pid"
    script = tmp_path / "descendant-script"
    script.write_text(
        f"/usr/bin/sleep 30 &\nprintf '%s' \"$!\" > {pid_path}\nexit 0\n",
        encoding="utf-8",
    )
    manifest = bench._open_retained_file(
        script,
        role="synthetic descendant script",
        maximum_bytes=bench.MAX_MANIFEST_BYTES,
    )
    _, environment, _ = bench._environment()
    try:
        observation = bench._spawn_once(
            binary,
            manifest,
            working_directory=tmp_path,
            environment=environment,
            timeout_seconds=2.0,
        )
    finally:
        manifest.close()
        binary.close()

    descendant_pid = int(pid_path.read_text(encoding="utf-8"))

    def descendant_is_running() -> bool:
        try:
            fields = Path(f"/proc/{descendant_pid}/stat").read_text(
                encoding="utf-8"
            ).split()
        except OSError:
            return False
        return len(fields) > 2 and fields[2] not in {"X", "Z"}

    deadline = time.monotonic() + 1.0
    while descendant_is_running() and time.monotonic() < deadline:
        time.sleep(0.01)

    assert observation.return_code == 0
    assert observation.timed_out is False
    assert not descendant_is_running()


def test_spawn_once_never_signals_reaped_numeric_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not Path("/usr/bin/setsid").exists():
        pytest.skip("/usr/bin/setsid is unavailable")
    binary, _ = _retained_shell(tmp_path)
    pid_path = tmp_path / "escaped-descendant.pid"
    nested = (
        f"printf '%s' \"$$\" > {shlex.quote(str(pid_path))}; "
        "exec /usr/bin/sleep 30"
    )
    script = tmp_path / "escaped-descendant-script"
    script.write_text(
        f"/usr/bin/setsid -f /bin/sh -c {shlex.quote(nested)}\n"
        f"while test ! -s {shlex.quote(str(pid_path))}; do :; done\n"
        "exit 0\n",
        encoding="utf-8",
    )
    manifest = bench._open_retained_file(
        script,
        role="synthetic escaped-descendant script",
        maximum_bytes=bench.MAX_MANIFEST_BYTES,
    )
    _, environment, _ = bench._environment()
    reaped = False
    real_wait4 = bench.os.wait4
    real_kill_process_group = bench._kill_process_group

    def tracked_wait4(pid: int, options: int):
        nonlocal reaped
        result = real_wait4(pid, options)
        reaped = True
        return result

    def tracked_kill_process_group(pid: int) -> None:
        assert not reaped, "numeric process-group signal occurred after exact-child reap"
        real_kill_process_group(pid)

    monkeypatch.setattr(bench.os, "wait4", tracked_wait4)
    monkeypatch.setattr(bench, "_kill_process_group", tracked_kill_process_group)
    monkeypatch.setattr(bench, "_PIPE_DRAIN_TIMEOUT_NS", 20_000_000)
    descendant_pid: int | None = None
    try:
        observation = bench._spawn_once(
            binary,
            manifest,
            working_directory=tmp_path,
            environment=environment,
            timeout_seconds=2.0,
        )
        descendant_pid = int(pid_path.read_text(encoding="utf-8"))
    finally:
        if descendant_pid is None and pid_path.exists():
            descendant_pid = int(pid_path.read_text(encoding="utf-8"))
        if descendant_pid is not None:
            try:
                os.kill(descendant_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        manifest.close()
        binary.close()

    assert observation.return_code == 0
    assert observation.timed_out is False
    assert observation.stdout_complete is False
    assert observation.stderr_complete is False
    assert observation.capture_error == "LingeringPipe,LingeringPipe"


def test_spawn_once_clock_failure_after_wait4_never_signals_reaped_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary, _ = _retained_shell(tmp_path)
    script = tmp_path / "clock-failure-script"
    script.write_text("exit 0\n", encoding="utf-8")
    manifest = bench._open_retained_file(
        script,
        role="synthetic clock-failure script",
        maximum_bytes=bench.MAX_MANIFEST_BYTES,
    )
    _, environment, _ = bench._environment()
    reaped = False
    real_wait4 = bench.os.wait4
    real_monotonic_ns = bench.time.monotonic_ns
    real_kill_process_group = bench._kill_process_group

    def tracked_wait4(pid: int, options: int):
        nonlocal reaped
        result = real_wait4(pid, options)
        reaped = True
        return result

    def fail_clock_after_reap() -> int:
        if reaped:
            raise RuntimeError("synthetic post-wait4 clock failure")
        return real_monotonic_ns()

    def tracked_kill_process_group(pid: int) -> None:
        assert not reaped, "numeric process-group signal occurred after exact-child reap"
        real_kill_process_group(pid)

    monkeypatch.setattr(bench.os, "wait4", tracked_wait4)
    monkeypatch.setattr(bench.time, "monotonic_ns", fail_clock_after_reap)
    monkeypatch.setattr(bench, "_kill_process_group", tracked_kill_process_group)
    try:
        with pytest.raises(RuntimeError, match="post-wait4 clock failure"):
            bench._spawn_once(
                binary,
                manifest,
                working_directory=tmp_path,
                environment=environment,
                timeout_seconds=2.0,
            )
    finally:
        manifest.close()
        binary.close()

    assert reaped is True


def test_spawn_once_terminates_immediately_on_output_overflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not Path("/usr/bin/yes").exists():
        pytest.skip("/usr/bin/yes is unavailable")
    binary, _ = _retained_shell(tmp_path)
    script = tmp_path / "overflow-script"
    script.write_text("exec /usr/bin/yes x\n", encoding="utf-8")
    manifest = bench._open_retained_file(
        script,
        role="synthetic overflow script",
        maximum_bytes=bench.MAX_MANIFEST_BYTES,
    )
    _, environment, _ = bench._environment()
    monkeypatch.setattr(bench, "MAX_RESULT_STREAM_BYTES", 1_024)
    try:
        observation = bench._spawn_once(
            binary,
            manifest,
            working_directory=tmp_path,
            environment=environment,
            timeout_seconds=2.0,
        )
    finally:
        manifest.close()
        binary.close()

    assert observation.stdout_complete is False
    assert len(observation.stdout) == 1_024
    assert observation.return_code < 0
    assert observation.timed_out is False
    assert observation.monotonic_stop_ns - observation.monotonic_start_ns < 1_000_000_000
