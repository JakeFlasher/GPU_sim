from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.cache_replay import sector_cache_replay
from tools.local_labels import immutable_artifact as immutable
from tools.local_labels.canary import emit_canary_manifest as canary
from tools.nvbit_tracer.trace_admission import (
    EXPECTED_MEMORY_PAYLOAD_CLASSES,
    EXPECTED_TRACE_CONTRACT_ID,
    LEGACY_TRACE_CONTRACT_ID,
)


def _event(
    *,
    contract_id: str = EXPECTED_TRACE_CONTRACT_ID,
    capabilities: list[int] | None = None,
) -> dict:
    return {
        "validate_trace_dir": {"ok": True, "typed_reason": "trace_admission_passed"},
        "trace_contract_id": contract_id,
        "memory_payload_classes": list(
            EXPECTED_MEMORY_PAYLOAD_CLASSES
            if capabilities is None
            else capabilities
        ),
        "admitted_records_by_class": {
            canary.ATOMIC: 3,
            canary.ASYNC_COPY: 5,
        },
        "admitted_atomic": 3,
        "admitted_async_copy": 5,
        "admitted_global": 2,
        "address_space_histogram": {canary.GLOBAL: 10},
        "ldgsts_source_records_with_allocation_id": 4,
        "unsupported_memory_path_total": 0,
        "unsupported_memory_path_flipped_classes": 0,
        "allocations_sidecar_entries": 1,
    }


class _VerifiedCapability:
    def __init__(self) -> None:
        self.verify_count = 0

    def verify_unchanged(self) -> None:
        self.verify_count += 1


def _analyze_witness(
    monkeypatch: pytest.MonkeyPatch,
    event: dict,
) -> dict:
    sources = _VerifiedCapability()
    unsupported = _VerifiedCapability()
    monkeypatch.setattr(canary, "analyze", lambda *_args: event)
    monkeypatch.setattr(
        canary,
        "_decode_admitted_access_stream",
        lambda _sources: SimpleNamespace(
            n_global=7,
            included_op_class_counts={canary.ASYNC_COPY: 4},
            excluded_op_class_counts={canary.ASYNC_COPY: 2},
        ),
    )
    witness = canary.analyze_witness(
        sources,
        unsupported,
        "op_ld",
        canary.ASYNC_COPY,
    )
    assert sources.verify_count == 1
    assert unsupported.verify_count == 1
    return witness


def test_split_v2_passes_canary_and_both_witness_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = _event()

    assert canary.canary_verdict(event) == (True, [])
    witness = _analyze_witness(monkeypatch, event)
    assert witness["scorable"] is True
    assert canary.witness_verdict("gemm_hmma_bf16", witness) == []


def test_reversed_capability_order_fails_live_canary_and_witness_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = _event(capabilities=list(reversed(EXPECTED_MEMORY_PAYLOAD_CLASSES)))

    ok, failures = canary.canary_verdict(event)
    assert ok is False
    assert any("memory_payload_classes" in failure for failure in failures)
    witness = _analyze_witness(monkeypatch, event)
    assert witness["scorable"] is False
    assert any(
        "memory_payload_classes" in failure
        for failure in canary.witness_verdict("gemm_hmma_bf16", witness)
    )


def test_retired_v1_fails_canary_and_both_witness_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = _event(contract_id=LEGACY_TRACE_CONTRACT_ID)

    ok, canary_failures = canary.canary_verdict(event)
    assert ok is False
    assert any("trace_contract_id" in failure for failure in canary_failures)

    witness = _analyze_witness(monkeypatch, event)
    assert witness["scorable"] is False
    witness_failures = canary.witness_verdict("gemm_hmma_bf16", witness)
    assert any("wrong trace_contract_id" in failure for failure in witness_failures)
    assert any("not scorable" in failure for failure in witness_failures)


def test_witness_requires_selected_class_in_the_l2_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = _event()
    sources = _VerifiedCapability()
    unsupported = _VerifiedCapability()
    monkeypatch.setattr(canary, "analyze", lambda *_args: event)
    monkeypatch.setattr(
        canary,
        "_decode_admitted_access_stream",
        lambda _sources: SimpleNamespace(
            n_global=7,
            included_op_class_counts={canary.GLOBAL: 7},
            excluded_op_class_counts={canary.ASYNC_COPY: 2},
        ),
    )

    witness = canary.analyze_witness(
        sources,
        unsupported,
        "op_ld",
        canary.ASYNC_COPY,
    )

    assert witness["scorable"] is False
    assert any(
        "no admitted L2 sectors" in failure
        for failure in canary.witness_verdict("gemm_hmma_bf16", witness)
    )


@pytest.mark.parametrize(
    "capabilities",
    [
        [1, 2, 3, 6],
        [1, 2, 3, 6, 8, 8],
        [1, 2, 3, 6, 8, 9],
    ],
    ids=("missing", "duplicate", "extra"),
)
def test_canary_requires_the_exact_canonical_capability_vector(
    monkeypatch: pytest.MonkeyPatch,
    capabilities: list[int],
) -> None:
    event = _event(capabilities=capabilities)

    ok, failures = canary.canary_verdict(event)
    assert ok is False
    assert any("memory_payload_classes" in failure for failure in failures)

    witness = _analyze_witness(monkeypatch, event)
    assert witness["scorable"] is False
    assert any(
        "memory_payload_classes" in failure
        for failure in canary.witness_verdict("gemm_hmma_bf16", witness)
    )


def test_canary_git_status_failure_cannot_be_inferred_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        canary,
        "run_retained_git",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=7,
            stdout="",
            stderr="injected status failure",
        ),
    )

    with pytest.raises(RuntimeError, match=r"Git status .* exited 7"):
        canary._git(
            object(),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )


def test_canary_git_head_failure_rejects_valid_looking_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        canary,
        "run_retained_git",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=9,
            stdout=("a" * 40) + "\n",
            stderr="injected rev-parse failure",
        ),
    )

    with pytest.raises(RuntimeError, match="cannot resolve exact canary source_commit"):
        canary._canary_git_head(object())


class _RetainedAuthority:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.verify_count = 0

    def read_bytes(self) -> bytes:
        return self.payload

    def verify_unchanged(self) -> None:
        self.verify_count += 1


def test_canary_source_commit_authority_is_exact_and_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = "a" * 40
    retained = {
        relative_path: _RetainedAuthority(
            f"authority:{relative_path}\n".encode()
        )
        for relative_path in canary._SOURCE_COMMIT_AUTHORITY_PATHS
    }
    monkeypatch.setattr(canary, "REPO", tmp_path)
    monkeypatch.setattr(canary, "_canary_git_head", lambda: head)

    def git_run(command, **_kwargs):
        if "status" in command:
            return SimpleNamespace(returncode=0, stdout="")
        if "show" in command:
            show_index = command.index("show")
            relative_path = command[show_index + 1].split(":", 1)[1]
            return SimpleNamespace(
                returncode=0,
                stdout=retained[relative_path].payload,
            )
        raise AssertionError(command)

    monkeypatch.setattr(canary.subprocess, "run", git_run)

    assert canary._resolve_canary_source_commit(retained) == head
    assert all(authority.verify_count == 1 for authority in retained.values())
    assert set(canary._SOURCE_COMMIT_AUTHORITY_PATHS) == {
        "tools/local_labels/canary/emit_canary_manifest.py",
        "tools/local_labels/canary/sm86_addr_carry_canary.cu",
        "tools/local_labels/canary/ac4_gemm_hmma_bf16.cu",
        "tools/local_labels/canary/ac4_global_atomic_histogram.cu",
        "tools/local_labels/delegated_cgroup.py",
        "tools/local_labels/immutable_artifact.py",
        "tools/local_labels/protected_paths.py",
        "tools/nvbit_tracer/capture_process.py",
        "tools/nvbit_tracer/external_tracer.py",
        "tools/nvbit_tracer/import_trace_contract.py",
        "tools/nvbit_tracer/proto_binding_loader.py",
        "tools/nvbit_tracer/protobuf_wire_preflight.py",
        "tools/nvbit_tracer/runtime_env.py",
        "tools/nvbit_tracer/trace_admission.py",
        "tools/cache_replay/sector_cache_replay.py",
        *{
            f"workloads/simbench_bw/_pb_generated/{name}_pb2.py"
            for name in (
                "address", "cuda_stream", "dim3d", "gpu_device",
                "instruction", "kernel", "threadblock", "trace", "warp",
            )
        },
    }


def test_canary_source_commit_authority_rejects_dirty_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained = {
        relative_path: _RetainedAuthority(b"authority")
        for relative_path in canary._SOURCE_COMMIT_AUTHORITY_PATHS
    }
    monkeypatch.setattr(canary, "REPO", tmp_path)
    monkeypatch.setattr(canary, "_canary_git_head", lambda: "a" * 40)
    monkeypatch.setattr(
        canary.subprocess,
        "run",
        lambda command, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=" M tools/nvbit_tracer/runtime_env.py\n",
        ),
    )

    with pytest.raises(RuntimeError, match="dirty or untracked"):
        canary._resolve_canary_source_commit(retained)
    assert all(authority.verify_count == 0 for authority in retained.values())


def test_canary_retained_source_build_forces_cuda_language(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "fixture.cu"
    source.write_bytes(b"__global__ void fixture() {}\n")
    binary = tmp_path / "fixture"
    descriptor = os.open(source, os.O_RDONLY)
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = list(command)
        observed["pass_fds"] = kwargs.get("pass_fds")
        binary.write_bytes(b"binary")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(
        canary, "_assert_staging_output", lambda path, **_kwargs: Path(path)
    )
    monkeypatch.setattr(canary, "_run", fake_run)
    try:
        assert canary.build_cu(
            source,
            tmp_path,
            binary=binary,
            retained_descriptor=descriptor,
        ) == binary
    finally:
        os.close(descriptor)

    command = observed["command"]
    source_argument = f"/proc/self/fd/{descriptor}"
    assert command[command.index("-x") + 1] == "cu"
    assert command.index("-x") < command.index(source_argument)
    pass_fds = observed["pass_fds"]
    assert type(pass_fds) is tuple
    assert len(pass_fds) == 3
    assert pass_fds[-1] == descriptor
    assert command[0] == f"/proc/self/fd/{pass_fds[0]}"
    assert command[1] == f"-ccbin=/proc/self/fd/{pass_fds[1]}"


def test_canary_compiler_identity_drift_after_compile_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "fixture.cu"
    source.write_bytes(b"__global__ void fixture() {}\n")
    binary = tmp_path / "fixture"
    source_descriptor = os.open(source, os.O_RDONLY)

    class FakeCompiler:
        def __init__(self, descriptor: int, *, fail_after_first: bool) -> None:
            self.descriptor = descriptor
            self.proc_path = f"/proc/self/fd/{descriptor}"
            self.verify_count = 0
            self.fail_after_first = fail_after_first

        def verify_unchanged(self) -> None:
            self.verify_count += 1
            if self.fail_after_first and self.verify_count > 1:
                raise RuntimeError("planted compiler replacement")

    nvcc = FakeCompiler(701, fail_after_first=False)
    host = FakeCompiler(702, fail_after_first=True)

    def fake_run(_command, **_kwargs):
        binary.write_bytes(b"binary")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(
        canary, "_assert_staging_output", lambda path, **_kwargs: Path(path)
    )
    monkeypatch.setattr(canary, "_run", fake_run)
    try:
        with pytest.raises(RuntimeError, match="compiler replacement"):
            canary.build_cu(
                source,
                tmp_path,
                binary=binary,
                retained_descriptor=source_descriptor,
                retained_nvcc=nvcc,
                retained_host_compiler=host,
            )
    finally:
        os.close(source_descriptor)
    assert nvcc.verify_count == 2
    assert host.verify_count == 2


def test_manifest_reruns_remain_reachable_and_advance_current_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "canary_manifest.json"
    output.write_bytes(b"legacy manifest\n")
    monkeypatch.setattr(
        canary, "_assert_staging_output", lambda path, **_kwargs: Path(path)
    )

    first = canary._publish_manifest(
        output,
        b"first manifest\n",
        attempt_id="attempt-one",
    )
    second = canary._publish_manifest(
        output,
        b"second manifest\n",
        attempt_id="attempt-two",
    )
    locator_before_failure = output.read_bytes()

    def fail_locator_update(*_args, **_kwargs):
        raise RuntimeError("injected locator failure")

    monkeypatch.setattr(canary, "replace_regular_bytes", fail_locator_update)
    with pytest.raises(RuntimeError, match="injected locator failure"):
        canary._publish_manifest(
            output,
            b"third manifest\n",
            attempt_id="attempt-three",
        )

    generation_root = tmp_path / "canary_manifest.json.generations"
    assert first.read_bytes() == b"first manifest\n"
    assert second.read_bytes() == b"second manifest\n"
    assert (generation_root / "attempt-three.json").read_bytes() == b"third manifest\n"
    legacy = list(generation_root.glob("legacy-*.json"))
    assert len(legacy) == 1
    assert legacy[0].read_bytes() == b"legacy manifest\n"
    assert output.read_bytes() == locator_before_failure
    locator = json.loads(locator_before_failure)
    assert locator["manifest_generation"]["attempt_id"] == "attempt-two"
    assert locator["manifest_generation"]["path"].endswith("attempt-two.json")
    assert locator["manifest_generation"]["sha256"] == hashlib.sha256(
        b"second manifest\n"
    ).hexdigest()
    assert locator["manifest_generation"]["size_bytes"] == len(
        b"second manifest\n"
    )


def test_source_drift_during_canary_locator_update_retires_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "canary_manifest.json"
    monkeypatch.setattr(
        canary, "_assert_staging_output", lambda path, **_kwargs: Path(path)
    )
    checks = 0

    def publication_check() -> None:
        nonlocal checks
        checks += 1
        if checks == 3:
            raise RuntimeError("planted canary source_commit drift")

    with pytest.raises(RuntimeError, match="source_commit drift"):
        canary._publish_manifest(
            output,
            b"manifest bytes\n",
            attempt_id="attempt-drift",
            publication_check=publication_check,
        )

    assert checks == 3
    assert not output.exists()
    assert (
        tmp_path / "canary_manifest.json.generations" / "attempt-drift.json"
    ).read_bytes() == b"manifest bytes\n"


@pytest.mark.parametrize(
    "path_spelling",
    ("absolute", "dot-relative"),
)
def test_noncanonical_current_locator_is_rejected_and_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path_spelling: str,
) -> None:
    output = tmp_path / "canary_manifest.json"
    generation_root = tmp_path / "canary_manifest.json.generations"
    generation_root.mkdir()
    current_payload = b"current manifest\n"
    current_generation = generation_root / "attempt-current.json"
    current_generation.write_bytes(current_payload)
    canonical_relative = (
        "canary_manifest.json.generations/attempt-current.json"
    )
    generation_path = (
        str(current_generation)
        if path_spelling == "absolute"
        else f"./{canonical_relative}"
    )
    locator_payload = (
        json.dumps(
            {
                "schema_version": 1,
                "artifact_kind": (
                    "sm86_addr_carry_canary_manifest_current"
                ),
                "manifest_generation": {
                    "attempt_id": "attempt-current",
                    "path": generation_path,
                    "sha256": hashlib.sha256(current_payload).hexdigest(),
                    "size_bytes": len(current_payload),
                },
            },
            sort_keys=True,
        )
        + "\n"
    ).encode()
    output.write_bytes(locator_payload)
    monkeypatch.setattr(
        canary, "_assert_staging_output", lambda path, **_kwargs: Path(path)
    )

    canary._publish_manifest(
        output,
        b"new manifest\n",
        attempt_id="attempt-new",
    )

    preserved = list(generation_root.glob("legacy-*.json"))
    assert len(preserved) == 1
    assert preserved[0].read_bytes() == locator_payload
    locator = json.loads(output.read_bytes())
    assert locator["manifest_generation"]["path"] == (
        "canary_manifest.json.generations/attempt-new.json"
    )


def test_identical_manifest_retry_after_locator_failure_advances_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "canary_manifest.json"
    monkeypatch.setattr(
        canary, "_assert_staging_output", lambda path, **_kwargs: Path(path)
    )
    canary._publish_manifest(
        output,
        b"prior manifest\n",
        attempt_id="attempt-prior",
    )
    prior_locator = output.read_bytes()
    real_replace = canary.replace_regular_bytes
    fail_locator_update = True

    def fail_locator_update_once(*args, **kwargs):
        nonlocal fail_locator_update
        if fail_locator_update:
            fail_locator_update = False
            raise RuntimeError("injected locator failure")
        return real_replace(*args, **kwargs)

    monkeypatch.setattr(
        canary, "replace_regular_bytes", fail_locator_update_once
    )
    payload = b"retry manifest\n"
    with pytest.raises(RuntimeError, match="injected locator failure"):
        canary._publish_manifest(
            output,
            payload,
            attempt_id="attempt-retry",
        )
    assert output.read_bytes() == prior_locator

    generation = canary._publish_manifest(
        output,
        payload,
        attempt_id="attempt-retry",
    )

    assert generation.read_bytes() == payload
    locator = json.loads(output.read_bytes())
    assert locator["manifest_generation"] == {
        "attempt_id": "attempt-retry",
        "path": "canary_manifest.json.generations/attempt-retry.json",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def test_conflicting_manifest_retry_after_locator_failure_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "canary_manifest.json"
    monkeypatch.setattr(
        canary, "_assert_staging_output", lambda path, **_kwargs: Path(path)
    )
    canary._publish_manifest(
        output,
        b"prior manifest\n",
        attempt_id="attempt-prior",
    )
    prior_locator = output.read_bytes()
    real_replace = canary.replace_regular_bytes
    fail_locator_update = True

    def fail_locator_update_once(*args, **kwargs):
        nonlocal fail_locator_update
        if fail_locator_update:
            fail_locator_update = False
            raise RuntimeError("injected locator failure")
        return real_replace(*args, **kwargs)

    monkeypatch.setattr(
        canary, "replace_regular_bytes", fail_locator_update_once
    )
    generation_root = tmp_path / "canary_manifest.json.generations"
    original_payload = b"retry manifest\n"
    with pytest.raises(RuntimeError, match="injected locator failure"):
        canary._publish_manifest(
            output,
            original_payload,
            attempt_id="attempt-retry",
        )

    with pytest.raises(canary.ImmutableArtifactError) as failure:
        canary._publish_manifest(
            output,
            b"other manifest\n",
            attempt_id="attempt-retry",
        )

    assert isinstance(failure.value.__cause__, RuntimeError)
    assert "different size or SHA-256" in str(failure.value.__cause__)
    assert output.read_bytes() == prior_locator
    assert (generation_root / "attempt-retry.json").read_bytes() == (
        original_payload
    )


def test_legacy_manifest_preservation_tolerates_concurrent_identical_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "canary_manifest.json"
    legacy_payload = b"legacy manifest\n"
    output.write_bytes(legacy_payload)
    monkeypatch.setattr(
        canary, "_assert_staging_output", lambda path, **_kwargs: Path(path)
    )
    real_publish = canary.publish_regular_file
    raced = False

    def publish_with_concurrent_winner(parent, name, payload, **kwargs):
        nonlocal raced
        if name.startswith("legacy-") and not raced:
            raced = True
            real_publish(parent, name, payload, **kwargs)
            raise canary.ImmutableArtifactError(
                "publication already exists after concurrent winner"
            )
        return real_publish(parent, name, payload, **kwargs)

    monkeypatch.setattr(
        canary, "publish_regular_file", publish_with_concurrent_winner
    )

    generation = canary._publish_manifest(
        output,
        b"new manifest\n",
        attempt_id="attempt-race",
    )

    generation_root = tmp_path / "canary_manifest.json.generations"
    legacy = list(generation_root.glob("legacy-*.json"))
    assert raced is True
    assert generation.read_bytes() == b"new manifest\n"
    assert len(legacy) == 1
    assert legacy[0].read_bytes() == legacy_payload


def test_malformed_current_locator_is_preserved_before_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "canary_manifest.json"
    malformed = (
        b'{"schema_version":1,"artifact_kind":'
        b'"sm86_addr_carry_canary_manifest_current",'
        b'"manifest_generation":{}}\n'
    )
    output.write_bytes(malformed)
    monkeypatch.setattr(
        canary, "_assert_staging_output", lambda path, **_kwargs: Path(path)
    )

    canary._publish_manifest(
        output,
        b"new manifest\n",
        attempt_id="attempt-after-malformed",
    )

    generation_root = tmp_path / "canary_manifest.json.generations"
    legacy = list(generation_root.glob("legacy-*.json"))
    assert len(legacy) == 1
    assert legacy[0].read_bytes() == malformed
    locator = json.loads(output.read_bytes())
    assert (
        locator["manifest_generation"]["attempt_id"]
        == "attempt-after-malformed"
    )


@contextmanager
def _runtime_fixture(*_args, **_kwargs):
    yield SimpleNamespace(environment={}, pass_fds=())


def test_capture_reruns_preserve_prior_and_failed_trace_generations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = tmp_path / "canary"
    tracer = tmp_path / "tracer_tool.so"
    generation_root = tmp_path / "trace-generations"
    binary.write_bytes(b"executable")
    binary.chmod(0o755)
    tracer.write_bytes(b"tracer")
    with immutable.open_retained_regular(tracer.resolve()) as retained_tracer:
        path_identity = retained_tracer.path_identity
    tracer_authority = canary.ExternalTracerAuthority(
        path=tracer.resolve(),
        sha256=hashlib.sha256(b"tracer").hexdigest(),
        path_identity=path_identity,
    )

    attempts = iter(("attempt-one", "attempt-two", "attempt-three"))
    returncodes = iter((0, 0, 7))
    invocations = 0

    def fake_run(_command: list[str], **kwargs):
        nonlocal invocations
        invocations += 1
        trace_root = Path(kwargs["env"]["TRACES_FOLDER"])
        (trace_root / "dynamic_trace.pb").write_bytes(
            f"trace-{invocations}".encode()
        )
        return SimpleNamespace(
            returncode=next(returncodes),
            stdout=b"",
            stderr=b"fixture failure",
        )

    admission = SimpleNamespace(
        trace_admission_sha256="a" * 64,
        trace_filesystem_generation_sha256="b" * 64,
    )
    monkeypatch.setattr(canary, "PB_LIB", tmp_path / "pb")
    monkeypatch.setattr(
        canary, "_assert_staging_output", lambda path, **_kwargs: Path(path)
    )
    monkeypatch.setattr(canary, "_new_capture_attempt_id", lambda: next(attempts))
    monkeypatch.setattr(canary, "retained_tracer_runtime_env", _runtime_fixture)
    monkeypatch.setattr(canary, "require_trace_admission", lambda _path: admission)
    monkeypatch.setattr(canary, "_run", fake_run)

    first = canary.capture(
        binary,
        generation_root,
        tracer_authority=tracer_authority,
    )
    second = canary.capture(
        binary,
        generation_root,
        tracer_authority=tracer_authority,
    )
    with pytest.raises(
        canary.CanaryCaptureNonzero, match="target exited 7"
    ) as caught:
        canary.capture(
            binary,
            generation_root,
            tracer_authority=tracer_authority,
        )
    assert caught.value.returncode == 7
    assert caught.value.stderr == b"fixture failure"

    assert first.path == generation_root / "attempt-one"
    assert second.path == generation_root / "attempt-two"
    assert (first.path / "dynamic_trace.pb").read_bytes() == b"trace-1"
    assert (second.path / "dynamic_trace.pb").read_bytes() == b"trace-2"
    assert (
        generation_root / ".attempt-three.staging" / "dynamic_trace.pb"
    ).read_bytes() == b"trace-3"


def test_admitted_cache_decode_ignores_a_replaced_lexical_tree(
    tmp_path: Path,
) -> None:
    threadblock_pb2, _, _ = sector_cache_replay.import_trace_contract._import_pb_modules()
    payload = threadblock_pb2.threadblock().SerializeToString()
    relative_path = (
        "threadblocks/device_0/stream_0/kernel_1/kernel_1_0,0,0.pb"
    )

    lexical = tmp_path / relative_path
    lexical.parent.mkdir(parents=True)
    lexical.write_bytes(b"poisoned replacement")

    class Sources:
        threadblock_paths = (relative_path,)
        verify_count = 0

        def read_threadblock_snapshot(self, observed_path: str):
            assert observed_path == relative_path
            return SimpleNamespace(payload=payload)

        def verify_unchanged(self) -> None:
            self.verify_count += 1

    sources = Sources()
    stream = sector_cache_replay.decode_admitted_access_stream(sources)

    assert stream.n_launches == 1
    assert stream.n_threadblocks == 1
    assert sources.verify_count == 1
