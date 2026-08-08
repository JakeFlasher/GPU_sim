"""Default-gate regressions for the retained golden NVBit launcher."""
from __future__ import annotations

from collections.abc import Iterator
import errno
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
import tempfile
from types import SimpleNamespace

import pytest

from tools.local_labels import immutable_artifact as immutable
from tools.nvbit_tracer import capture_process
from tools.trace_sampling.golden_cells import retained_nvbit_launch


_BINARY_PAYLOAD = b"#!/bin/sh\nexit 0\n"
_TRACER_PAYLOAD = b"external split tracer"
_STDOUT = b"stdout\x00bytes"
_STDERR = b"stderr\xffbytes"
_PROCESS_ID = 123
_STARTED_AT_UTC = "2026-08-03T00:00:00Z"


@pytest.fixture
def retained_nvbit_tmp_path() -> Iterator[Path]:
    real_tmp = Path("/tmp").resolve(strict=True)
    with tempfile.TemporaryDirectory(
        prefix="retained-nvbit-launch-",
        dir=real_tmp,
    ) as directory:
        root = Path(directory).resolve(strict=True)
        root.relative_to(real_tmp)
        yield root


def _load_attempt(trace_dir: Path) -> dict[str, object]:
    receipt = json.loads(
        (trace_dir / retained_nvbit_launch._ATTEMPT_NAME).read_text()
    )
    unsealed = dict(receipt)
    seal = unsealed.pop("attempt_seal_sha256")
    assert seal == immutable.sha256_bytes(
        immutable.canonical_json_bytes(unsealed)
    )
    return receipt


def _run_retained_nvbit_fixture(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    returncode: int,
    runtime_failure: bool = False,
    process_failure: bool = False,
    identity_failure: bool = False,
    replace_trace_directory: bool = False,
    verify_oserror_at: str | None = None,
    attempt_close_failure: bool = False,
    final_close_failure: bool = False,
    replace_on_final_close: bool = False,
) -> tuple[int, Path]:
    staging_root = root / "results" / "nvbit_capture_staging"
    trace_dir = staging_root / "golden_cells" / "fixture" / "trace.staged.1"
    trace_dir.parent.mkdir(parents=True)
    binary = root / "fixture.bin"
    binary.write_bytes(_BINARY_PAYLOAD)
    binary.chmod(0o555)
    tracer = root / "external" / "tracer_tool.so"
    tracer.parent.mkdir()
    tracer.write_bytes(_TRACER_PAYLOAD)
    pb_lib = root / "pb-lib"
    pb_lib.mkdir()
    monkeypatch.setattr(
        retained_nvbit_launch,
        "_GOLDEN_STAGING_ROOT",
        staging_root.resolve(),
    )

    if attempt_close_failure or final_close_failure or replace_on_final_close:
        if attempt_close_failure and (final_close_failure or replace_on_final_close):
            raise AssertionError("close-failure fixtures are mutually exclusive")
        if final_close_failure and replace_on_final_close:
            raise AssertionError("final-close fixtures are mutually exclusive")
        real_dup = os.dup
        real_close = os.close
        retained_trace_descriptors: list[int] = []
        close_failed = False
        failure_index = (
            1 if final_close_failure or replace_on_final_close else 0
        )

        def track_dup(descriptor: int) -> int:
            duplicate = real_dup(descriptor)
            try:
                source_path: Path | None = Path(
                    os.readlink(f"/proc/self/fd/{descriptor}")
                )
            except OSError:
                source_path = None
            if source_path == trace_dir:
                retained_trace_descriptors.append(duplicate)
            return duplicate

        def fail_attempt_close(descriptor: int) -> None:
            nonlocal close_failed
            target_descriptor = (
                retained_trace_descriptors[failure_index]
                if len(retained_trace_descriptors) > failure_index
                else None
            )
            if descriptor == target_descriptor and not close_failed:
                close_failed = True
                if replace_on_final_close:
                    retained_trace_dir = trace_dir.with_name("trace.retained.1")
                    trace_dir.rename(retained_trace_dir)
                    trace_dir.mkdir()
                    real_close(descriptor)
                    return
                real_close(descriptor)
                raise OSError(
                    errno.EIO,
                    (
                        "synthetic final retained trace descriptor close failure"
                        if final_close_failure
                        else "synthetic attempt descriptor close failure"
                    ),
                )
            real_close(descriptor)

        monkeypatch.setattr(os, "dup", track_dup)
        monkeypatch.setattr(os, "close", fail_attempt_close)

    execute_returned = False
    verify_after_execute_calls = 0
    if verify_oserror_at is not None:
        original_verify = immutable.RetainedExecutable.verify_unchanged

        def verify_unchanged(retained) -> None:
            nonlocal verify_after_execute_calls
            if execute_returned:
                verify_after_execute_calls += 1
                should_fail = (
                    verify_oserror_at == "post_launch"
                    and verify_after_execute_calls == 1
                ) or (
                    verify_oserror_at == "context_exit"
                    and verify_after_execute_calls == (
                        1 if process_failure else 2
                    )
                )
                if should_fail:
                    raise OSError(
                        f"synthetic {verify_oserror_at} identity failure"
                    )
            original_verify(retained)

        monkeypatch.setattr(
            immutable.RetainedExecutable,
            "verify_unchanged",
            verify_unchanged,
        )

    def runtime_environment(path, *, base_env, pb_lib, runtime_manifest, retained_descriptor):
        assert path == tracer.resolve()
        assert os.pread(retained_descriptor, 4096, 0) == _TRACER_PAYLOAD
        assert "CUDA_INJECTION64_PATH" not in base_env
        if runtime_failure:
            raise retained_nvbit_launch.runtime_env.TracerRuntimeError(
                "synthetic dependency failure"
            )

        @contextmanager
        def retained_runtime():
            runtime_descriptor = os.dup(retained_descriptor)
            try:
                yield SimpleNamespace(
                    environment={
                        **base_env,
                        "LD_LIBRARY_PATH": (
                            f"/proc/self/fd/{runtime_descriptor}"
                        ),
                    },
                    pass_fds=(runtime_descriptor,),
                )
            finally:
                os.close(runtime_descriptor)

        return retained_runtime()

    def execute(command, *, environment, inherited_descriptors, **_kwargs):
        nonlocal execute_returned
        binary_descriptor = int(command[0].rsplit("/", 1)[1])
        tracer_descriptor = int(
            environment["CUDA_INJECTION64_PATH"].rsplit("/", 1)[1]
        )
        trace_descriptor = int(environment["TRACES_FOLDER"].rsplit("/", 1)[1])
        runtime_descriptor = int(
            environment["LD_LIBRARY_PATH"].rsplit("/", 1)[1]
        )
        assert inherited_descriptors == (
            binary_descriptor,
            tracer_descriptor,
            trace_descriptor,
            runtime_descriptor,
        )
        assert os.pread(binary_descriptor, 4096, 0) == _BINARY_PAYLOAD
        assert os.pread(tracer_descriptor, 4096, 0) == _TRACER_PAYLOAD
        if identity_failure:
            replacement = binary.with_name("replacement.bin")
            replacement.write_bytes(b"replaced target")
            replacement.chmod(0o555)
            replacement.replace(binary)
        if replace_trace_directory:
            retained_trace_dir = trace_dir.with_name("trace.retained.1")
            trace_dir.rename(retained_trace_dir)
            trace_dir.mkdir()
        execute_returned = True
        if process_failure:
            error = capture_process.NvbitCaptureProcessError(
                "synthetic_process_failure",
                "synthetic process failure",
            )
            error.stdout = _STDOUT
            error.stderr = _STDERR
            error.process_id = _PROCESS_ID
            error.started_at_utc = _STARTED_AT_UTC
            raise error
        return capture_process.NvbitCaptureProcessResult(
            returncode=returncode,
            stdout=_STDOUT,
            stderr=_STDERR,
            process_id=_PROCESS_ID,
            started_at_utc=_STARTED_AT_UTC,
        )

    monkeypatch.setattr(
        retained_nvbit_launch.runtime_env,
        "retained_tracer_runtime_env",
        runtime_environment,
    )
    monkeypatch.setattr(
        retained_nvbit_launch.capture_process,
        "execute_capture_process",
        execute,
    )
    result = retained_nvbit_launch.main([
        "--binary", str(binary),
        "--binary-sha256", hashlib.sha256(_BINARY_PAYLOAD).hexdigest(),
        "--tracer", str(tracer.resolve()),
        "--tracer-sha256", hashlib.sha256(_TRACER_PAYLOAD).hexdigest(),
        "--tracer-source-kind", "external_non_git_retained_descriptor_v1",
        "--preflight-sha256", "a" * 64,
        "--pb-lib", str(pb_lib),
        "--trace-dir", str(trace_dir),
    ])
    if attempt_close_failure or final_close_failure or replace_on_final_close:
        assert len(retained_trace_descriptors) > failure_index
        assert close_failed
    published_trace_dir = (
        trace_dir.with_name("trace.retained.1")
        if replace_trace_directory or replace_on_final_close
        else trace_dir
    )
    return result, published_trace_dir


@pytest.mark.parametrize(
    ("returncode", "expected_status"),
    [(0, "succeeded"), (7, "failed")],
)
def test_retained_nvbit_launch_binds_external_tracer_and_attempt_streams(
    retained_nvbit_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    expected_status: str,
) -> None:
    result, trace_dir = _run_retained_nvbit_fixture(
        retained_nvbit_tmp_path,
        monkeypatch,
        returncode=returncode,
    )

    assert result == returncode
    assert (
        trace_dir / retained_nvbit_launch._RETAINED_TRACER_NAME
    ).read_bytes() == _TRACER_PAYLOAD
    assert (trace_dir / retained_nvbit_launch._STDOUT_NAME).read_bytes() == _STDOUT
    assert (trace_dir / retained_nvbit_launch._STDERR_NAME).read_bytes() == _STDERR
    receipt = _load_attempt(trace_dir)
    assert receipt["status"] == expected_status
    assert receipt["preflight_sha256"] == "a" * 64
    assert receipt["tracer"]["source_kind"] == (
        "external_non_git_retained_descriptor_v1"
    )
    assert receipt["tracer"]["sha256"] == hashlib.sha256(
        _TRACER_PAYLOAD
    ).hexdigest()
    assert receipt["process"]["returncode"] == returncode
    trace_metadata = trace_dir.stat()
    assert receipt["trace_generation"] == {
        "device": trace_metadata.st_dev,
        "inode": trace_metadata.st_ino,
    }


def test_retained_nvbit_launch_seals_prelaunch_runtime_failure_with_null_process(
    retained_nvbit_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, trace_dir = _run_retained_nvbit_fixture(
        retained_nvbit_tmp_path,
        monkeypatch,
        returncode=0,
        runtime_failure=True,
    )

    assert result == 2
    assert (
        trace_dir / retained_nvbit_launch._RETAINED_TRACER_NAME
    ).read_bytes() == _TRACER_PAYLOAD
    receipt = _load_attempt(trace_dir)
    assert receipt["status"] == "failed"
    assert receipt["process"]["error_code"] == "TracerRuntimeError"
    assert "synthetic dependency failure" in receipt["process"]["error_detail"]
    assert receipt["process"]["returncode"] is None
    assert receipt["process"]["process_id"] is None
    assert receipt["process"]["started_at_utc"] is None


def test_retained_nvbit_launch_seals_postlaunch_identity_failure(
    retained_nvbit_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, trace_dir = _run_retained_nvbit_fixture(
        retained_nvbit_tmp_path,
        monkeypatch,
        returncode=0,
        identity_failure=True,
    )

    assert result == 2
    assert (trace_dir / retained_nvbit_launch._STDOUT_NAME).read_bytes() == _STDOUT
    assert (trace_dir / retained_nvbit_launch._STDERR_NAME).read_bytes() == _STDERR
    receipt = _load_attempt(trace_dir)
    assert receipt["status"] == "failed"
    assert receipt["process"]["error_code"] == "ImmutableArtifactError"
    assert "changed" in receipt["process"]["error_detail"]
    assert receipt["process"]["process_id"] == _PROCESS_ID
    assert receipt["process"]["started_at_utc"] == _STARTED_AT_UTC


def test_retained_nvbit_launch_keeps_receipt_with_renamed_trace_generation(
    retained_nvbit_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, retained_trace_dir = _run_retained_nvbit_fixture(
        retained_nvbit_tmp_path,
        monkeypatch,
        returncode=0,
        replace_trace_directory=True,
    )

    replacement_trace_dir = retained_trace_dir.with_name("trace.staged.1")
    assert result == 2
    assert (
        retained_trace_dir / retained_nvbit_launch._RETAINED_TRACER_NAME
    ).read_bytes() == _TRACER_PAYLOAD
    assert (
        retained_trace_dir / retained_nvbit_launch._STDOUT_NAME
    ).read_bytes() == _STDOUT
    assert (
        retained_trace_dir / retained_nvbit_launch._STDERR_NAME
    ).read_bytes() == _STDERR
    receipt = _load_attempt(retained_trace_dir)
    assert receipt["status"] == "failed"
    assert receipt["process"]["error_code"] == "ImmutableArtifactError"
    assert not any(replacement_trace_dir.iterdir())


def test_retained_nvbit_launch_seals_postlaunch_identity_oserror(
    retained_nvbit_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, trace_dir = _run_retained_nvbit_fixture(
        retained_nvbit_tmp_path,
        monkeypatch,
        returncode=0,
        verify_oserror_at="post_launch",
    )

    assert result == 2
    receipt = _load_attempt(trace_dir)
    assert receipt["status"] == "failed"
    assert receipt["process"]["error_code"] == "OSError"
    assert "synthetic post_launch identity failure" in (
        receipt["process"]["error_detail"]
    )
    assert receipt["process"]["process_id"] == _PROCESS_ID
    assert receipt["process"]["started_at_utc"] == _STARTED_AT_UTC


def test_retained_nvbit_launch_seals_failure_after_context_exit_failure(
    retained_nvbit_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, trace_dir = _run_retained_nvbit_fixture(
        retained_nvbit_tmp_path,
        monkeypatch,
        returncode=0,
        verify_oserror_at="context_exit",
    )

    assert result == 2
    receipt = _load_attempt(trace_dir)
    assert receipt["status"] == "failed"
    assert receipt["process"]["error_code"] == "OSError"
    assert "synthetic context_exit identity failure" in (
        receipt["process"]["error_detail"]
    )
    assert receipt["process"]["process_id"] == _PROCESS_ID
    assert receipt["process"]["started_at_utc"] == _STARTED_AT_UTC


def test_retained_nvbit_launch_preserves_process_failure_across_context_exit_failure(
    retained_nvbit_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, trace_dir = _run_retained_nvbit_fixture(
        retained_nvbit_tmp_path,
        monkeypatch,
        returncode=0,
        process_failure=True,
        verify_oserror_at="context_exit",
    )

    assert result == 2
    assert (trace_dir / retained_nvbit_launch._STDOUT_NAME).read_bytes() == _STDOUT
    assert (trace_dir / retained_nvbit_launch._STDERR_NAME).read_bytes() == _STDERR
    receipt = _load_attempt(trace_dir)
    assert receipt["status"] == "failed"
    assert receipt["process"]["error_code"] == "synthetic_process_failure"
    assert "synthetic process failure" in receipt["process"]["error_detail"]
    assert receipt["process"]["returncode"] is None
    assert receipt["process"]["process_id"] == _PROCESS_ID
    assert receipt["process"]["started_at_utc"] == _STARTED_AT_UTC
    assert receipt["teardown_failures"] == [{
        "error_code": "OSError",
        "error_detail": "synthetic context_exit identity failure",
    }]


def test_retained_nvbit_launch_seals_descriptor_close_failure_instead_of_success(
    retained_nvbit_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, trace_dir = _run_retained_nvbit_fixture(
        retained_nvbit_tmp_path,
        monkeypatch,
        returncode=0,
        attempt_close_failure=True,
    )

    assert result == 2
    receipt = _load_attempt(trace_dir)
    assert receipt["status"] == "failed"
    assert receipt["process"]["returncode"] == 0
    assert receipt["process"]["error_code"] is None
    assert receipt["process"]["error_detail"] is None
    assert receipt["process"]["process_id"] == _PROCESS_ID
    assert receipt["process"]["started_at_utc"] == _STARTED_AT_UTC
    assert receipt["teardown_failures"][0]["error_code"] == "OSError"
    assert "synthetic attempt descriptor close failure" in (
        receipt["teardown_failures"][0]["error_detail"]
    )


def test_retained_nvbit_launch_refuses_success_on_final_close_failure(
    retained_nvbit_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    result, trace_dir = _run_retained_nvbit_fixture(
        retained_nvbit_tmp_path,
        monkeypatch,
        returncode=0,
        final_close_failure=True,
    )

    assert result == 2
    receipt = _load_attempt(trace_dir)
    assert receipt["status"] == "succeeded"
    assert receipt["teardown_failures"] == []
    stderr = capsysbinary.readouterr().err
    assert (
        b"retained_nvbit_launch_refused: DURABILITY_UNPROVEN: final retained "
        b"trace descriptor close:"
    ) in stderr
    assert b"synthetic final retained trace descriptor close failure" in stderr


def test_retained_nvbit_launch_refuses_final_trace_generation_replacement(
    retained_nvbit_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, retained_trace_dir = _run_retained_nvbit_fixture(
        retained_nvbit_tmp_path,
        monkeypatch,
        returncode=0,
        replace_on_final_close=True,
    )

    replacement_trace_dir = retained_trace_dir.with_name("trace.staged.1")
    assert result == 2
    receipt = _load_attempt(retained_trace_dir)
    assert receipt["status"] == "succeeded"
    assert not any(replacement_trace_dir.iterdir())


def test_retained_nvbit_launch_types_external_tracer_valueerror(
    retained_nvbit_tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_dir = (
        retained_nvbit_tmp_path
        / "results"
        / "nvbit_capture_staging"
        / "trace.staged.1"
    )
    result = retained_nvbit_launch.main([
        "--binary", str(retained_nvbit_tmp_path / "unused.bin"),
        "--binary-sha256", "b" * 64,
        "--tracer", "relative-tracer.so",
        "--tracer-sha256", "c" * 64,
        "--tracer-source-kind", "external_non_git_retained_descriptor_v1",
        "--preflight-sha256", "a" * 64,
        "--pb-lib", str(retained_nvbit_tmp_path / "pb-lib"),
        "--trace-dir", str(trace_dir),
    ])

    assert result == 2
    assert not trace_dir.exists()
    assert (
        "retained_nvbit_launch_refused: external tracer validation failed: "
        "NVBit tracer must be an absolute path"
    ) in capsys.readouterr().err
