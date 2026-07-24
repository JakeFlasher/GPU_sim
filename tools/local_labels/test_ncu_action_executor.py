"""Offline contracts for descriptor-bound authorized NCU execution."""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import pytest

from tools.local_labels import immutable_artifact as immutable
from tools.local_labels import ncu_action_executor as executor


_VERSION = "Version 2026.1.0.0 (public-release)"


def _fixture_ncu(path: Path, *, version_mode: str = "ok") -> Path:
    source = f"""#!{sys.executable}
import json
import os
import subprocess
import sys
import time

if sys.argv[1:] == [\"--version\"]:
    mode = {version_mode!r}
    if mode == \"missing\":
        print(\"synthetic banner without release\")
    elif mode == \"duplicate\":
        print({_VERSION!r})
        print({_VERSION!r})
    elif mode == \"nonzero\":
        print(\"version probe failed\", file=sys.stderr)
        raise SystemExit(9)
    elif mode == \"sleep\":
        time.sleep(10)
    elif mode == \"flood\":
        os.write(1, b\"x\" * 4096)
    else:
        print({_VERSION!r})
    raise SystemExit(0)

operation = sys.argv[1] if len(sys.argv) > 1 else \"\"
arguments = sys.argv[2:]
if operation == \"--echo\":
    print(json.dumps({{
        \"arguments\": arguments,
        \"environment\": dict(os.environ),
    }}, sort_keys=True))
    print(\"synthetic stderr\", file=sys.stderr)
elif operation == \"--exit\":
    print(\"ordinary nonzero\")
    print(\"ordinary nonzero stderr\", file=sys.stderr)
    raise SystemExit(int(arguments[0]))
elif operation == \"--sleep\":
    time.sleep(float(arguments[0]))
elif operation == \"--flood\":
    os.write(1, b\"x\" * int(arguments[0]))
elif operation == \"--mutate-restore\":
    Path = __import__(\"pathlib\").Path
    target = Path(arguments[0])
    payload = target.read_bytes()
    target.chmod(0o755)
    target.write_bytes(b\"mutated\")
    target.write_bytes(payload)
    target.chmod(0o555)
elif operation == \"--spawn-sleep\":
    child = subprocess.Popen([sys.executable, \"-c\", \"import time; time.sleep(60)\"])
    Path = __import__(\"pathlib\").Path
    Path(arguments[0]).write_text(str(child.pid))
    time.sleep(60)
else:
    print(\"unknown operation\", file=sys.stderr)
    raise SystemExit(64)
"""
    path.write_text(source)
    path.chmod(0o555)
    return path


def _identity(path: Path, *, version: str = _VERSION, sha: str | None = None):
    digest, _size = immutable.sha256_file(path)
    return executor.ExpectedNcuIdentity(str(path), sha or digest, version)


def _assert_code(code: str):
    return pytest.raises(executor.NcuActionExecutorError, match=code)


def test_expected_identity_factory_returns_one_validated_authority() -> None:
    expected = executor.build_expected_ncu_identity(
        resolved_path="/opt/nvidia/nsight-compute/ncu",
        executable_sha256="a" * 64,
        version_line=_VERSION,
    )
    assert expected == executor.ExpectedNcuIdentity(
        "/opt/nvidia/nsight-compute/ncu", "a" * 64, _VERSION,
    )
    assert executor.validate_expected_ncu_identity(expected) is expected


@pytest.mark.parametrize(
    "overrides",
    [
        {"resolved_path": "//opt/ncu"},
        {"resolved_path": "/tmp/../tmp/ncu"},
        {"resolved_path": "/" + "x" * 4096},
        {"executable_sha256": "A" * 64},
        {"version_line": _VERSION + "\nsecond"},
        {"version_line": "Version x\vsecond"},
        {"version_line": "Version " + "x" * 4096},
    ],
)
def test_expected_identity_factory_rejects_malformed_authority(overrides) -> None:
    values = {
        "resolved_path": "/opt/nvidia/nsight-compute/ncu",
        "executable_sha256": "a" * 64,
        "version_line": _VERSION,
        **overrides,
    }
    with _assert_code(executor.NCU_IDENTITY_INPUT_INVALID):
        executor.build_expected_ncu_identity(**values)


def _fixture_shell_wrapper(path: Path) -> Path:
    path.write_text(f"""#!/bin/sh
set -eu
dirname /tmp/report.ncu-rep >/dev/null
readlink -f /tmp >/dev/null
uname -s >/dev/null
if [ "$#" -eq 1 ] && [ "$1" = "--version" ]; then
    printf '%s\\n' {_VERSION!r}
    exit 0
fi
if [ "$#" -eq 1 ] && [ "$1" = "--emit" ]; then
    printf '%s\\n' 'wrapper action'
    exit 0
fi
exit 64
""")
    path.chmod(0o555)
    return path


def test_action_environment_is_closed_literal_and_fresh(
    tmp_path: Path, monkeypatch,
) -> None:
    ncu = _fixture_ncu(tmp_path / "ncu")
    hostile_names = (
        "LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT", "CUDA_INJECTION64_PATH",
        "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "NV_COMPUTE_PROFILER_LOCAL_CONNECTION_OVERRIDE",
        "NCU_CONFIG_FILE", "CUDA_HOME", "CUDA_PATH", "NVCC_CCBIN",
        "PYTHONPATH", "PYTHONHOME", "BASH_ENV", "ENV", "HOME", "TMPDIR",
        "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "LOCPATH", "NLSPATH",
        "LABEL_DRIVER_GPU_INDEX", "FLASHINFER_TRACE_DIR", "SOL_NATIVE_ENV_HASH",
    )
    for name in hostile_names:
        monkeypatch.setenv(name, f"hostile-{name}")
    monkeypatch.setenv("PATH", "/hostile/bin")
    monkeypatch.setattr(executor.os, "defpath", "/hostile/default")
    observed = []
    original_popen = executor.subprocess.Popen

    def popen(*args, **kwargs):
        observed.append(dict(kwargs["env"]))
        return original_popen(*args, **kwargs)

    monkeypatch.setattr(executor.subprocess, "Popen", popen)
    result = executor.execute_ncu_action(
        _identity(ncu), ["--echo"], timeout_seconds=2,
    )
    expected = {
        "LANG": "C",
        "LANGUAGE": "C",
        "LC_ALL": "C",
        "PATH": "/bin:/usr/bin",
    }
    assert result.returncode == 0
    assert observed == [expected, expected]
    assert all(name not in observed[0] for name in hostile_names)
    first = executor.ncu_action_environment()
    first["PATH"] = "/mutated"
    assert executor.ncu_action_environment() == expected


def test_fixed_action_path_supports_wrapper_without_ambient_tools(
    tmp_path: Path, monkeypatch,
) -> None:
    ncu = _fixture_shell_wrapper(tmp_path / "ncu")
    hostile = tmp_path / "hostile"
    hostile.mkdir()
    marker = tmp_path / "hostile-tool-ran"
    for name in ("dirname", "readlink", "uname"):
        tool = hostile / name
        tool.write_text(f"#!/bin/sh\ntouch {marker}\nexit 99\n")
        tool.chmod(0o755)
    monkeypatch.setenv("PATH", str(hostile))
    result = executor.execute_ncu_action(
        _identity(ncu), ["--emit"], timeout_seconds=2,
    )
    assert result == executor.NcuActionResult(0, b"wrapper action\n", b"")
    assert not marker.exists()


def test_authorized_action_preserves_argv_bytes_locale_and_no_shell(
    tmp_path: Path, monkeypatch,
) -> None:
    ncu = _fixture_ncu(tmp_path / "ncu")
    marker = tmp_path / "must-not-exist"
    hostile = f"$(touch {marker})"
    monkeypatch.setenv("LD_PRELOAD", "/untrusted/injection.so")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/untrusted/libraries")
    monkeypatch.setenv("PYTHONPATH", "/untrusted/modules")
    result = executor.execute_ncu_action(
        _identity(ncu),
        ["--echo", "", "value with spaces", hostile],
        timeout_seconds=2,
    )
    assert result.returncode == 0
    observed = json.loads(result.stdout)
    assert observed == {
        "arguments": ["", "value with spaces", hostile],
        "environment": {
            "LANG": "C",
            "LANGUAGE": "C",
            "LC_ALL": "C",
            "PATH": os.defpath,
        },
    }
    assert result.stderr == b"synthetic stderr\n"
    assert not marker.exists()


def test_authorized_action_returns_ordinary_nonzero_result(tmp_path: Path) -> None:
    ncu = _fixture_ncu(tmp_path / "ncu")
    result = executor.execute_ncu_action(
        _identity(ncu), ["--exit", "7"], timeout_seconds=2,
    )
    assert result == executor.NcuActionResult(
        7, b"ordinary nonzero\n", b"ordinary nonzero stderr\n",
    )


@pytest.mark.parametrize(
    "identity",
    [
        None,
        executor.ExpectedNcuIdentity("relative/ncu", "a" * 64, _VERSION),
        executor.ExpectedNcuIdentity("//tmp/ncu", "a" * 64, _VERSION),
        executor.ExpectedNcuIdentity("/tmp/../tmp/ncu", "a" * 64, _VERSION),
        executor.ExpectedNcuIdentity("/tmp/ncu", "A" * 64, _VERSION),
        executor.ExpectedNcuIdentity("/tmp/ncu", "a" * 63, _VERSION),
        executor.ExpectedNcuIdentity("/tmp/ncu", "a" * 64, "not a version"),
        executor.ExpectedNcuIdentity("/tmp/ncu", "a" * 64, _VERSION + "\n"),
        executor.ExpectedNcuIdentity("/tmp/ncu", "a" * 64, "Version x\x00y"),
        executor.ExpectedNcuIdentity("/tmp/ncu", "a" * 64, "Version x\vsecond"),
        executor.ExpectedNcuIdentity("/tmp/ncu", "a" * 64, "Version x\fsecond"),
        executor.ExpectedNcuIdentity("/tmp/ncu", "a" * 64, "Version x\x85second"),
        executor.ExpectedNcuIdentity(
            "/tmp/ncu", "a" * 64, "Version x" + chr(0x2028) + "second",
        ),
        executor.ExpectedNcuIdentity(
            "/tmp/ncu", "a" * 64, "Version x" + chr(0x2029) + "second",
        ),
    ],
)
def test_rejects_malformed_expected_identity_before_open(identity) -> None:
    with _assert_code(executor.NCU_IDENTITY_INPUT_INVALID):
        executor.execute_ncu_action(identity, ["--echo"], timeout_seconds=1)


@pytest.mark.parametrize(
    ("arguments", "timeout"),
    [
        ([], 1),
        ("--echo", 1),
        ([b"--echo"], 1),
        (["bad\x00argument"], 1),
        (["x"] * 4097, 1),
        (["x" * (1024 * 1024)], 1),
        (["--echo"], True),
        (["--echo"], 0),
        (["--echo"], float("nan")),
        (["--echo"], 10**1000),
        (["--echo"], 86401),
    ],
)
def test_rejects_invalid_action_request_before_open(arguments, timeout) -> None:
    expected = executor.ExpectedNcuIdentity("/missing/ncu", "a" * 64, _VERSION)
    with _assert_code(executor.NCU_ACTION_REQUEST_INVALID):
        executor.execute_ncu_action(
            expected, arguments, timeout_seconds=timeout,
        )


def test_rejects_declared_oversized_sequence_without_materializing_items() -> None:
    class OversizedSequence(Sequence):
        def __len__(self):
            return 10**12

        def __getitem__(self, _index):
            pytest.fail("oversized sequence was indexed")

    expected = executor.ExpectedNcuIdentity("/missing/ncu", "a" * 64, _VERSION)
    with _assert_code(executor.NCU_ACTION_REQUEST_INVALID):
        executor.execute_ncu_action(
            expected, OversizedSequence(), timeout_seconds=1,
        )


def test_rejects_single_argument_above_linux_exec_string_bound() -> None:
    expected = executor.ExpectedNcuIdentity("/missing/ncu", "a" * 64, _VERSION)
    with _assert_code(executor.NCU_ACTION_REQUEST_INVALID):
        executor.execute_ncu_action(
            expected,
            ["x" * (executor._MAX_SINGLE_ARGUMENT_BYTES + 1)],
            timeout_seconds=1,
        )


@pytest.mark.parametrize("kind", ["missing", "nonexec", "leaf_symlink", "ancestor_symlink", "hardlink"])
def test_rejects_unretainable_executable(tmp_path: Path, kind: str) -> None:
    ncu = tmp_path / "ncu"
    candidate = ncu
    if kind != "missing":
        _fixture_ncu(ncu)
    if kind == "nonexec":
        ncu.chmod(0o644)
    elif kind == "leaf_symlink":
        target = tmp_path / "target"
        ncu.rename(target)
        ncu.symlink_to(target)
    elif kind == "ancestor_symlink":
        real = tmp_path / "real"
        real.mkdir()
        ncu.rename(real / "ncu")
        linked = tmp_path / "linked"
        linked.symlink_to(real, target_is_directory=True)
        candidate = linked / "ncu"
    elif kind == "hardlink":
        os.link(ncu, tmp_path / "second-link")
    expected = executor.ExpectedNcuIdentity(
        str(candidate), "a" * 64, _VERSION,
    )
    with _assert_code(executor.NCU_EXECUTABLE_IDENTITY_INVALID):
        executor.execute_ncu_action(
            expected, ["--echo"], timeout_seconds=1,
        )


def test_rejects_sha_before_any_subprocess(tmp_path: Path, monkeypatch) -> None:
    ncu = _fixture_ncu(tmp_path / "ncu")
    monkeypatch.setattr(
        executor.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("subprocess reached after SHA mismatch"),
    )
    with _assert_code(executor.NCU_EXECUTABLE_IDENTITY_INVALID):
        executor.execute_ncu_action(
            _identity(ncu, sha="0" * 64), ["--echo"], timeout_seconds=1,
        )


def test_rejects_capture_process_writable_executable_before_subprocess(
    tmp_path: Path, monkeypatch,
) -> None:
    ncu = _fixture_ncu(tmp_path / "ncu")
    ncu.chmod(0o755)
    expected = _identity(ncu)
    monkeypatch.setattr(
        executor.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail(
            "subprocess reached for writable executable"
        ),
    )
    with _assert_code(executor.NCU_EXECUTABLE_IDENTITY_INVALID):
        executor.execute_ncu_action(
            expected, ["--echo"], timeout_seconds=1,
        )


@pytest.mark.parametrize(
    ("mode", "code"),
    [
        ("missing", executor.NCU_VERSION_LINE_MISMATCH),
        ("duplicate", executor.NCU_VERSION_LINE_MISMATCH),
        ("nonzero", executor.NCU_VERSION_PROBE_FAILED),
    ],
)
def test_rejects_invalid_version_probe(
    tmp_path: Path, monkeypatch, mode: str, code: str,
) -> None:
    ncu = _fixture_ncu(tmp_path / "ncu", version_mode=mode)
    with _assert_code(code):
        executor.execute_ncu_action(
            _identity(ncu), ["--echo"], timeout_seconds=1,
        )


@pytest.mark.parametrize("semantic_failure", ["nonzero", "version"])
def test_version_semantic_failure_cannot_mask_late_identity_drift(
    tmp_path: Path, monkeypatch, semantic_failure: str,
) -> None:
    ncu = _fixture_ncu(tmp_path / "ncu")
    probe = (
        executor.NcuActionResult(9, b"", b"failed")
        if semantic_failure == "nonzero"
        else executor.NcuActionResult(0, b"Version 2026.2.0\n", b"")
    )
    monkeypatch.setattr(executor, "_run_bounded", lambda *_args, **_kwargs: probe)
    original_reverify = executor._reverify

    def reverify(retained, *, pending=None):
        if pending is not None:
            ncu.chmod(0o755)
            ncu.write_bytes(b"late mutation")
            ncu.chmod(0o555)
        return original_reverify(retained, pending=pending)

    monkeypatch.setattr(executor, "_reverify", reverify)
    with _assert_code(executor.NCU_EXECUTABLE_IDENTITY_INVALID):
        executor.execute_ncu_action(
            _identity(ncu), ["--echo"], timeout_seconds=1,
        )


def test_requires_exact_expected_version_line(tmp_path: Path) -> None:
    ncu = _fixture_ncu(tmp_path / "ncu")
    with _assert_code(executor.NCU_VERSION_LINE_MISMATCH):
        executor.execute_ncu_action(
            _identity(ncu, version="Version 2026.1.0.1 (different)"),
            ["--echo"],
            timeout_seconds=1,
        )


def test_version_probe_timeout_is_bounded_and_typed(
    tmp_path: Path, monkeypatch,
) -> None:
    ncu = _fixture_ncu(tmp_path / "ncu", version_mode="sleep")
    monkeypatch.setattr(executor, "_VERSION_TIMEOUT_SECONDS", 0.05)
    started = time.monotonic()
    with _assert_code(executor.NCU_VERSION_PROBE_FAILED):
        executor.execute_ncu_action(
            _identity(ncu), ["--echo"], timeout_seconds=1,
        )
    assert time.monotonic() - started < 3


def test_output_monitor_setup_failure_terminates_and_reaps_process(
    tmp_path: Path, monkeypatch,
) -> None:
    ncu = _fixture_ncu(tmp_path / "ncu", version_mode="sleep")
    original_popen = executor.subprocess.Popen
    observed = []

    def popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        observed.append(process)
        return process

    def fail_selector():
        raise OSError("synthetic selector setup failure")

    monkeypatch.setattr(executor.subprocess, "Popen", popen)
    monkeypatch.setattr(executor.selectors, "DefaultSelector", fail_selector)
    with _assert_code(executor.NCU_VERSION_PROBE_FAILED):
        executor.execute_ncu_action(
            _identity(ncu), ["--echo"], timeout_seconds=1,
        )
    assert len(observed) == 1
    assert observed[0].poll() is not None
    assert observed[0].stdout.closed
    assert observed[0].stderr.closed


def test_output_monitor_cleanup_failure_cannot_mask_identity_drift(
    tmp_path: Path, monkeypatch,
) -> None:
    ncu = _fixture_ncu(tmp_path / "ncu")
    original_selector = executor.selectors.DefaultSelector

    class FailingCloseSelector:
        def __init__(self):
            self._selector = original_selector()

        def __getattr__(self, name):
            return getattr(self._selector, name)

        def close(self):
            ncu.chmod(0o755)
            ncu.write_bytes(b"mutated during selector cleanup")
            ncu.chmod(0o555)
            self._selector.close()
            raise OSError("synthetic selector close failure")

    monkeypatch.setattr(
        executor.selectors, "DefaultSelector", FailingCloseSelector,
    )
    with _assert_code(executor.NCU_EXECUTABLE_IDENTITY_INVALID):
        executor.execute_ncu_action(
            _identity(ncu), ["--echo"], timeout_seconds=1,
        )


def test_output_pipe_cleanup_failure_cannot_mask_identity_drift(
    tmp_path: Path, monkeypatch,
) -> None:
    ncu = _fixture_ncu(tmp_path / "ncu")
    original_popen = executor.subprocess.Popen

    class FailingClosePipe:
        def __init__(self, pipe):
            self._pipe = pipe

        def __getattr__(self, name):
            return getattr(self._pipe, name)

        def close(self):
            ncu.chmod(0o755)
            ncu.write_bytes(b"mutated during pipe cleanup")
            ncu.chmod(0o555)
            self._pipe.close()
            raise OSError("synthetic pipe close failure")

    def popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        process.stdout = FailingClosePipe(process.stdout)
        return process

    monkeypatch.setattr(executor.subprocess, "Popen", popen)
    with _assert_code(executor.NCU_EXECUTABLE_IDENTITY_INVALID):
        executor.execute_ncu_action(
            _identity(ncu), ["--echo"], timeout_seconds=1,
        )


def test_version_and_action_output_are_bounded(tmp_path: Path, monkeypatch) -> None:
    flood_version_ncu = _fixture_ncu(
        tmp_path / "ncu-flood-version", version_mode="flood",
    )
    monkeypatch.setattr(executor, "_MAX_VERSION_OUTPUT_BYTES", 1024)
    with _assert_code(executor.NCU_ACTION_OUTPUT_LIMIT):
        executor.execute_ncu_action(
            _identity(flood_version_ncu), ["--echo"], timeout_seconds=1,
        )

    ncu = _fixture_ncu(tmp_path / "ncu")
    monkeypatch.setattr(executor, "_MAX_ACTION_OUTPUT_BYTES", 1024)
    with _assert_code(executor.NCU_ACTION_OUTPUT_LIMIT):
        executor.execute_ncu_action(
            _identity(ncu), ["--flood", "4096"], timeout_seconds=1,
        )


def test_action_timeout_kills_process_group(tmp_path: Path) -> None:
    ncu = _fixture_ncu(tmp_path / "ncu")
    pid_path = tmp_path / "child.pid"
    with _assert_code(executor.NCU_ACTION_TIMEOUT):
        executor.execute_ncu_action(
            _identity(ncu),
            ["--spawn-sleep", str(pid_path)],
            timeout_seconds=0.1,
        )
    child_pid = int(pid_path.read_text())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    else:
        pytest.fail(f"timed-out child process {child_pid} is still alive")


def test_real_in_action_aba_byte_restore_is_detected(tmp_path: Path) -> None:
    ncu = _fixture_ncu(tmp_path / "ncu")
    with _assert_code(executor.NCU_EXECUTABLE_IDENTITY_INVALID):
        executor.execute_ncu_action(
            _identity(ncu),
            ["--mutate-restore", str(ncu)],
            timeout_seconds=1,
        )


@pytest.mark.parametrize("mutation", ["bytes", "replacement", "mode", "hardlink", "ancestor"])
def test_identity_drift_after_probe_prevents_action(
    tmp_path: Path, monkeypatch, mutation: str,
) -> None:
    directory = tmp_path / "bin"
    directory.mkdir()
    ncu = _fixture_ncu(directory / "ncu")
    expected = _identity(ncu)
    original = executor._run_bounded
    calls = 0

    def run(*args, **kwargs):
        nonlocal calls, ncu, directory
        result = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            if mutation == "bytes":
                ncu.chmod(0o755)
                ncu.write_bytes(b"mutated")
                ncu.chmod(0o555)
            elif mutation == "replacement":
                payload = ncu.read_bytes()
                ncu.unlink()
                ncu.write_bytes(payload)
                ncu.chmod(0o555)
            elif mutation == "mode":
                ncu.chmod(0o644)
            elif mutation == "hardlink":
                os.link(ncu, tmp_path / "second-link")
            else:
                moved = tmp_path / "moved"
                directory.rename(moved)
                directory.mkdir()
                ncu = directory / "ncu"
                _fixture_ncu(ncu)
        return result

    monkeypatch.setattr(executor, "_run_bounded", run)
    with _assert_code(executor.NCU_EXECUTABLE_IDENTITY_INVALID):
        executor.execute_ncu_action(
            expected, ["--echo"], timeout_seconds=1,
        )
    assert calls == 1


@pytest.mark.parametrize("pending", ["nonzero", "timeout"])
def test_action_failure_cannot_mask_simultaneous_identity_drift(
    tmp_path: Path, monkeypatch, pending: str,
) -> None:
    ncu = _fixture_ncu(tmp_path / "ncu")
    original = executor._run_bounded
    calls = 0

    def run(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return original(*args, **kwargs)
        ncu.chmod(0o755)
        ncu.write_bytes(b"mutated during failed action")
        ncu.chmod(0o555)
        if pending == "timeout":
            raise executor.NcuActionExecutorError(
                executor.NCU_ACTION_TIMEOUT, "synthetic timeout",
            )
        return executor.NcuActionResult(7, b"", b"action failed")

    monkeypatch.setattr(executor, "_run_bounded", run)
    with _assert_code(executor.NCU_EXECUTABLE_IDENTITY_INVALID):
        executor.execute_ncu_action(
            _identity(ncu), ["--echo"], timeout_seconds=1,
        )


def test_action_launch_failure_is_typed_after_successful_probe(
    tmp_path: Path, monkeypatch,
) -> None:
    ncu = _fixture_ncu(tmp_path / "ncu")
    original = executor._run_bounded
    calls = 0

    def run(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return original(*args, **kwargs)
        raise executor.NcuActionExecutorError(
            executor.NCU_ACTION_LAUNCH_FAILED, "synthetic launch failure",
        )

    monkeypatch.setattr(executor, "_run_bounded", run)
    with _assert_code(executor.NCU_ACTION_LAUNCH_FAILED):
        executor.execute_ncu_action(
            _identity(ncu), ["--echo"], timeout_seconds=1,
        )


@pytest.mark.parametrize("active_error", [False, True])
def test_close_failure_is_typed_or_attached_to_primary_error(
    tmp_path: Path, monkeypatch, active_error: bool,
) -> None:
    ncu = _fixture_ncu(tmp_path / "ncu")
    original_open = executor.immutable.open_retained_executable

    def open_retained(path):
        retained = original_open(path)
        original_close = retained.close

        def close_then_fail():
            original_close()
            raise OSError("synthetic close failure")

        retained.close = close_then_fail
        return retained

    monkeypatch.setattr(
        executor.immutable, "open_retained_executable", open_retained,
    )
    expected = _identity(ncu, sha="0" * 64) if active_error else _identity(ncu)
    with _assert_code(executor.NCU_EXECUTABLE_IDENTITY_INVALID) as caught:
        executor.execute_ncu_action(
            expected, ["--echo"], timeout_seconds=1,
        )
    if active_error:
        assert any(
            "close failed" in note
            for note in getattr(caught.value, "__notes__", [])
        )
    else:
        assert "cannot close retained executable" in caught.value.detail


@pytest.mark.parametrize("failure", [False, True])
def test_retained_descriptor_is_closed_on_every_exit(
    tmp_path: Path, monkeypatch, failure: bool,
) -> None:
    ncu = _fixture_ncu(tmp_path / "ncu")
    observed = {}
    original = executor.immutable.open_retained_executable

    def open_retained(path):
        retained = original(path)
        observed["retained"] = retained
        return retained

    monkeypatch.setattr(
        executor.immutable, "open_retained_executable", open_retained,
    )
    expected = _identity(ncu, sha="0" * 64) if failure else _identity(ncu)
    if failure:
        with _assert_code(executor.NCU_EXECUTABLE_IDENTITY_INVALID):
            executor.execute_ncu_action(
                expected, ["--echo"], timeout_seconds=1,
            )
    else:
        executor.execute_ncu_action(
            expected, ["--echo"], timeout_seconds=1,
        )
    with pytest.raises(
        immutable.ImmutableArtifactError, match="retained executable is closed",
    ):
        _ = observed["retained"].descriptor
