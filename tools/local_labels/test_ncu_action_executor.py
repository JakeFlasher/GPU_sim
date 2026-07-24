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
elif operation == \"--read-fds\":
    Path = __import__(\"pathlib\").Path
    print(json.dumps([Path(value).read_bytes().decode(\"utf-8\") for value in arguments]))
elif operation == \"--mutate-report-restore\":
    Path = __import__(\"pathlib\").Path
    target = Path(arguments[0])
    payload = target.read_bytes()
    target.write_bytes(b\"mutated report\")
    target.write_bytes(payload)
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


def _collection_environment(**overrides) -> executor.NcuCollectionEnvironment:
    return executor.NcuCollectionEnvironment({
        "LANG": "C",
        "LANGUAGE": "C",
        "LC_ALL": "C",
        "PATH": "/usr/local/cuda-13.2/bin:/usr/bin:/bin",
        "PWD": "/",
        "PYTHONNOUSERSITE": "1",
        "LABEL_DRIVER_PRESERVE_FAILED_EVIDENCE": "1",
        **overrides,
    })


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


@pytest.mark.parametrize(
    "name",
    [
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "LD_AUDIT",
        "CUDA_INJECTION64_PATH",
        "PYTHONPATH",
        "BASH_ENV",
        "CUDA_VISIBLE_DEVICES",
        "NCU_CONFIG_FILE",
    ],
)
def test_collection_environment_rejects_non_allowlisted_names(name: str) -> None:
    with _assert_code(executor.NCU_ACTION_REQUEST_INVALID):
        _collection_environment(**{name: "hostile"})


def test_collection_environment_rejects_canonical_hostile_path(
    tmp_path: Path,
) -> None:
    hostile = tmp_path / "hostile"
    hostile.mkdir()
    with _assert_code(executor.NCU_ACTION_REQUEST_INVALID):
        _collection_environment(
            PATH=f"{hostile}:/usr/local/cuda-13.2/bin:/usr/bin:/bin"
        )


@pytest.mark.parametrize("name", [None, 1, ("LD_PRELOAD",)])
def test_collection_environment_rejects_non_string_names(name) -> None:
    values = _collection_environment().as_dict()
    values[name] = "hostile"
    with _assert_code(executor.NCU_ACTION_REQUEST_INVALID):
        executor.NcuCollectionEnvironment(values)


@pytest.mark.parametrize(
    "overrides",
    [
        {"LANG": "en_US.UTF-8"},
        {"PYTHONNOUSERSITE": "0"},
        {"PATH": "relative:/bin"},
        {"PATH": "/bin::/usr/bin"},
        {"PATH": "//bin:/usr/bin"},
    ],
)
def test_collection_environment_rejects_required_or_path_drift(overrides) -> None:
    with _assert_code(executor.NCU_ACTION_REQUEST_INVALID):
        _collection_environment(**overrides)


def test_supervised_child_inherits_one_authorized_ncu_descriptor(
    tmp_path: Path, monkeypatch,
) -> None:
    ncu = _fixture_ncu(tmp_path / "ncu")
    driver = tmp_path / "driver"
    driver.write_text("#!/bin/sh\nexec \"$1\" --echo \"$2\"\n")
    driver.chmod(0o555)
    observed = []
    original_popen = executor.subprocess.Popen

    def popen(*args, **kwargs):
        observed.append((
            tuple(args[0]), tuple(kwargs["pass_fds"]), kwargs["env"],
            kwargs["cwd"],
        ))
        return original_popen(*args, **kwargs)

    monkeypatch.setattr(executor.subprocess, "Popen", popen)
    environment = _collection_environment()
    result = executor.execute_ncu_supervised_child(
        _identity(ncu),
        [str(driver), executor.AUTHORIZED_NCU_PROC_PATH, "inherited"],
        timeout_seconds=2,
        environment=environment,
    )
    assert result.returncode == 0
    observed_target = json.loads(result.stdout)
    assert observed_target["arguments"] == ["inherited"]
    target_environment = observed_target["environment"]
    assert target_environment.pop("SHLVL") == "0"
    assert target_environment == environment.as_dict()
    assert result.stderr == b"synthetic stderr\n"
    assert len(observed) == 2
    descriptor = observed[1][1][0]
    assert observed[0][0] == (f"/proc/self/fd/{descriptor}", "--version")
    assert observed[0][1] == (descriptor,)
    assert observed[0][2] == executor.ncu_action_environment()
    assert observed[0][3] == "/"
    assert observed[1][0] == (
        str(driver), f"/proc/self/fd/{descriptor}", "inherited",
    )
    assert observed[1][1] == (descriptor,)
    assert observed[1][2] == environment.as_dict()
    assert observed[1][3] == "/"
    with pytest.raises(OSError):
        os.fstat(descriptor)


@pytest.mark.parametrize(
    "command",
    [
        ["/bin/true"],
        ["/bin/true", executor.AUTHORIZED_NCU_PROC_PATH,
         executor.AUTHORIZED_NCU_PROC_PATH],
        [executor.AUTHORIZED_NCU_PROC_PATH],
        ["relative", executor.AUTHORIZED_NCU_PROC_PATH],
        ["/bin/true", object()],
    ],
)
def test_supervised_child_rejects_invalid_placeholder_before_open(command) -> None:
    expected = executor.ExpectedNcuIdentity("/missing/ncu", "a" * 64, _VERSION)
    with _assert_code(executor.NCU_ACTION_REQUEST_INVALID):
        executor.execute_ncu_supervised_child(
            expected,
            command,
            timeout_seconds=1,
            environment=_collection_environment(),
            )


def test_supervised_child_rejects_oversized_sequence_without_indexing() -> None:
    class OversizedSequence(Sequence):
        def __len__(self):
            return 10**12

        def __getitem__(self, _index):
            pytest.fail("oversized supervised command was indexed")

    expected = executor.ExpectedNcuIdentity("/missing/ncu", "a" * 64, _VERSION)
    with _assert_code(executor.NCU_ACTION_REQUEST_INVALID):
        executor.execute_ncu_supervised_child(
            expected,
            OversizedSequence(),
            timeout_seconds=1,
            environment=_collection_environment(),
        )


def test_supervised_success_kills_descendant_retaining_ncu_descriptor(
    tmp_path: Path,
) -> None:
    ncu = _fixture_ncu(tmp_path / "ncu")
    pid_path = tmp_path / "descendant.pid"
    driver = tmp_path / "driver"
    child_source = (
        "import os,sys,time; from pathlib import Path; "
        "fd=int(sys.argv[1]); os.fstat(fd); "
        "Path(sys.argv[2]).write_text(str(os.getpid())); time.sleep(60)"
    )
    driver.write_text(f"""#!{sys.executable}
import os
import subprocess
import sys
import time
from pathlib import Path

ncu_path = sys.argv[1]
descriptor = int(ncu_path.rsplit('/', 1)[1])
pid_path = Path(sys.argv[2])
subprocess.Popen(
    [sys.executable, '-c', {child_source!r}, str(descriptor), str(pid_path)],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    pass_fds=(descriptor,),
)
deadline = time.monotonic() + 2
while not pid_path.is_file() and time.monotonic() < deadline:
    time.sleep(0.01)
raise SystemExit(0 if pid_path.is_file() else 9)
""")
    driver.chmod(0o555)
    result = executor.execute_ncu_supervised_child(
        _identity(ncu),
        [str(driver), executor.AUTHORIZED_NCU_PROC_PATH, str(pid_path)],
        timeout_seconds=3,
        environment=_collection_environment(),
    )
    assert result.returncode == 0
    descendant_pid = int(pid_path.read_text())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(descendant_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    else:
        pytest.fail(
            f"successful supervised child leaked descendant {descendant_pid}"
        )


def test_supervised_child_identity_drift_precedes_ordinary_result(
    tmp_path: Path,
) -> None:
    ncu = _fixture_ncu(tmp_path / "ncu")
    driver = tmp_path / "driver"
    driver.write_text(f"""#!{sys.executable}
import os
import sys
from pathlib import Path

os.fstat(int(sys.argv[1].rsplit('/', 1)[1]))
target = Path(sys.argv[2])
payload = target.read_bytes()
target.unlink()
target.write_bytes(payload)
target.chmod(0o555)
raise SystemExit(7)
""")
    driver.chmod(0o555)
    with _assert_code(executor.NCU_EXECUTABLE_IDENTITY_INVALID):
        executor.execute_ncu_supervised_child(
            _identity(ncu),
            [
                str(driver), executor.AUTHORIZED_NCU_PROC_PATH, str(ncu),
            ],
            timeout_seconds=2,
            environment=_collection_environment(),
            )


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


def test_retained_report_and_auxiliary_use_exact_proc_paths_and_pass_fds(
    tmp_path: Path, monkeypatch,
) -> None:
    ncu = _fixture_ncu(tmp_path / "ncu")
    report_path = tmp_path / "report.ncu-rep"
    report_path.write_bytes(b"report bytes")
    report_path.chmod(0o444)
    auxiliary_path = tmp_path / "auxiliary.txt"
    auxiliary_path.write_bytes(b"auxiliary bytes")
    auxiliary = os.open(auxiliary_path, os.O_RDONLY)
    observed = []
    original_popen = executor.subprocess.Popen

    def popen(*args, **kwargs):
        observed.append((tuple(args[0]), tuple(kwargs["pass_fds"])))
        return original_popen(*args, **kwargs)

    monkeypatch.setattr(executor.subprocess, "Popen", popen)
    try:
        with immutable.open_retained_regular(report_path) as report:
            report_descriptor = report.descriptor
            result = executor.execute_ncu_action(
                _identity(ncu),
                [
                    "--read-fds",
                    report.proc_path,
                    executor.proc_fd_path(auxiliary),
                ],
                timeout_seconds=2,
                retained_report=report,
                auxiliary_descriptors=(auxiliary,),
            )
            assert json.loads(result.stdout) == [
                "report bytes", "auxiliary bytes",
            ]
            assert observed[0][0][1:] == ("--version",)
            assert observed[0][1] == (observed[0][1][0],)
            executable_descriptor = observed[0][1][0]
            assert observed[1][0] == (
                f"/proc/self/fd/{executable_descriptor}",
                "--read-fds",
                f"/proc/self/fd/{report_descriptor}",
                f"/proc/self/fd/{auxiliary}",
            )
            assert observed[1][1] == (
                executable_descriptor, report_descriptor, auxiliary,
            )
            os.fstat(report_descriptor)
            os.fstat(auxiliary)
    finally:
        os.close(auxiliary)


@pytest.mark.parametrize(
    "auxiliary_descriptors", [None, "3", (True,), (2,), (10**1000,)],
)
def test_rejects_invalid_auxiliary_descriptor_declaration_before_open(
    auxiliary_descriptors,
) -> None:
    expected = executor.ExpectedNcuIdentity("/missing/ncu", "a" * 64, _VERSION)
    with _assert_code(executor.NCU_ACTION_REQUEST_INVALID):
        executor.execute_ncu_action(
            expected,
            ["--echo"],
            timeout_seconds=1,
            auxiliary_descriptors=auxiliary_descriptors,
        )


def test_proc_fd_path_rejects_oversized_descriptor() -> None:
    with _assert_code(executor.NCU_ACTION_REQUEST_INVALID):
        executor.proc_fd_path(10**1000)


def test_rejects_closed_or_wrong_retained_report_before_open(
    tmp_path: Path,
) -> None:
    expected = executor.ExpectedNcuIdentity("/missing/ncu", "a" * 64, _VERSION)
    with _assert_code(executor.NCU_ACTION_REQUEST_INVALID):
        executor.execute_ncu_action(
            expected,
            ["--echo"],
            timeout_seconds=1,
            retained_report=object(),
        )
    path = tmp_path / "report.ncu-rep"
    path.write_bytes(b"report")
    report = immutable.open_retained_regular(path)
    proc_path = report.proc_path
    report.close()
    with _assert_code(executor.NCU_ACTION_REQUEST_INVALID):
        executor.execute_ncu_action(
            expected,
            ["--echo", proc_path],
            timeout_seconds=1,
            retained_report=report,
        )


def test_rejects_closed_duplicate_and_report_auxiliary_descriptors(
    tmp_path: Path,
) -> None:
    expected = executor.ExpectedNcuIdentity("/missing/ncu", "a" * 64, _VERSION)
    auxiliary_path = tmp_path / "auxiliary.txt"
    auxiliary_path.write_bytes(b"auxiliary")
    descriptor = os.open(auxiliary_path, os.O_RDONLY)
    os.close(descriptor)
    with _assert_code(executor.NCU_ACTION_REQUEST_INVALID):
        executor.execute_ncu_action(
            expected,
            ["--echo", f"/proc/self/fd/{descriptor}"],
            timeout_seconds=1,
            auxiliary_descriptors=(descriptor,),
        )

    descriptor = os.open(auxiliary_path, os.O_RDONLY)
    try:
        with _assert_code(executor.NCU_ACTION_REQUEST_INVALID):
            executor.execute_ncu_action(
                expected,
                ["--echo", executor.proc_fd_path(descriptor)],
                timeout_seconds=1,
                auxiliary_descriptors=(descriptor, descriptor),
            )
        report_path = tmp_path / "report.ncu-rep"
        report_path.write_bytes(b"report")
        with immutable.open_retained_regular(report_path) as report:
            with _assert_code(executor.NCU_ACTION_REQUEST_INVALID):
                executor.execute_ncu_action(
                    expected,
                    ["--echo", report.proc_path],
                    timeout_seconds=1,
                    retained_report=report,
                    auxiliary_descriptors=(report.descriptor,),
                )
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    "argument",
    [
        "/proc/self/fd/{descriptor}",
        "/proc/self/fd/0{descriptor}",
        "/proc/self/fd/{descriptor}/",
        "/proc/{pid}/fd/{descriptor}",
        "prefix=/proc/self/fd/{descriptor}",
    ],
)
def test_rejects_unregistered_or_noncanonical_proc_fd_argument(
    tmp_path: Path, argument: str,
) -> None:
    path = tmp_path / "auxiliary.txt"
    path.write_bytes(b"auxiliary")
    descriptor = os.open(path, os.O_RDONLY)
    expected = executor.ExpectedNcuIdentity("/missing/ncu", "a" * 64, _VERSION)
    try:
        rendered = argument.format(descriptor=descriptor, pid=os.getpid())
        declared = () if argument == "/proc/self/fd/{descriptor}" else (descriptor,)
        with _assert_code(executor.NCU_ACTION_REQUEST_INVALID):
            executor.execute_ncu_action(
                expected,
                ["--echo", rendered],
                timeout_seconds=1,
                auxiliary_descriptors=declared,
            )
    finally:
        os.close(descriptor)


def test_retained_report_drift_after_probe_prevents_action(
    tmp_path: Path, monkeypatch,
) -> None:
    ncu = _fixture_ncu(tmp_path / "ncu")
    report_path = tmp_path / "report.ncu-rep"
    report_path.write_bytes(b"report bytes")
    writer = os.open(report_path, os.O_WRONLY)
    report_path.chmod(0o444)
    report = immutable.open_retained_regular(report_path)
    original = executor._run_bounded
    calls = 0

    def run(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            os.pwrite(writer, b"changed bytes", 0)
        return result

    monkeypatch.setattr(executor, "_run_bounded", run)
    try:
        with _assert_code(executor.NCU_ACTION_INPUT_IDENTITY_INVALID):
            executor.execute_ncu_action(
                _identity(ncu),
                ["--read-fds", report.proc_path],
                timeout_seconds=2,
                retained_report=report,
            )
        assert calls == 1
    finally:
        report.close()
        os.close(writer)


def test_real_in_action_report_aba_byte_restore_is_detected(
    tmp_path: Path,
) -> None:
    ncu = _fixture_ncu(tmp_path / "ncu")
    report_path = tmp_path / "report.ncu-rep"
    report_path.write_bytes(b"report bytes")
    report = immutable.open_retained_regular(report_path)
    try:
        with _assert_code(executor.NCU_ACTION_INPUT_IDENTITY_INVALID):
            executor.execute_ncu_action(
                _identity(ncu),
                ["--mutate-report-restore", report.proc_path],
                timeout_seconds=2,
                retained_report=report,
            )
    finally:
        report.close()


@pytest.mark.parametrize("pending", ["ordinary", "timeout"])
def test_action_failure_cannot_mask_report_identity_drift(
    tmp_path: Path, monkeypatch, pending: str,
) -> None:
    ncu = _fixture_ncu(tmp_path / "ncu")
    report_path = tmp_path / "report.ncu-rep"
    report_path.write_bytes(b"report bytes")
    writer = os.open(report_path, os.O_WRONLY)
    report_path.chmod(0o444)
    report = immutable.open_retained_regular(report_path)
    original = executor._run_bounded
    calls = 0

    def run(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return original(*args, **kwargs)
        os.pwrite(writer, b"changed bytes", 0)
        if pending == "timeout":
            raise executor.NcuActionExecutorError(
                executor.NCU_ACTION_TIMEOUT, "synthetic timeout",
            )
        return executor.NcuActionResult(7, b"", b"action failed")

    monkeypatch.setattr(executor, "_run_bounded", run)
    try:
        with _assert_code(executor.NCU_ACTION_INPUT_IDENTITY_INVALID):
            executor.execute_ncu_action(
                _identity(ncu),
                ["--read-fds", report.proc_path],
                timeout_seconds=2,
                retained_report=report,
            )
        assert calls == 2
    finally:
        report.close()
        os.close(writer)


def test_untyped_boundary_failure_cannot_mask_report_identity_drift(
    tmp_path: Path, monkeypatch,
) -> None:
    ncu = _fixture_ncu(tmp_path / "ncu")
    report_path = tmp_path / "report.ncu-rep"
    report_path.write_bytes(b"report bytes")
    writer = os.open(report_path, os.O_WRONLY)
    report_path.chmod(0o444)
    report = immutable.open_retained_regular(report_path)
    original = executor._run_bounded
    calls = 0

    def run(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return original(*args, **kwargs)
        os.pwrite(writer, b"changed bytes", 0)
        raise OSError("synthetic wait failure")

    monkeypatch.setattr(executor, "_run_bounded", run)
    try:
        with _assert_code(executor.NCU_ACTION_INPUT_IDENTITY_INVALID) as caught:
            executor.execute_ncu_action(
                _identity(ncu),
                ["--read-fds", report.proc_path],
                timeout_seconds=2,
                retained_report=report,
            )
        assert isinstance(caught.value.__cause__, OSError)
        assert calls == 2
    finally:
        report.close()
        os.close(writer)


def test_executable_identity_precedes_report_identity_when_both_drift(
    tmp_path: Path, monkeypatch,
) -> None:
    ncu = _fixture_ncu(tmp_path / "ncu")
    report_path = tmp_path / "report.ncu-rep"
    report_path.write_bytes(b"report bytes")
    writer = os.open(report_path, os.O_WRONLY)
    report_path.chmod(0o444)
    report = immutable.open_retained_regular(report_path)
    original = executor._run_bounded
    calls = 0

    def run(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 2:
            ncu.chmod(0o755)
            ncu.write_bytes(b"mutated executable")
            ncu.chmod(0o555)
            os.pwrite(writer, b"changed bytes", 0)
        return result

    monkeypatch.setattr(executor, "_run_bounded", run)
    try:
        with _assert_code(executor.NCU_EXECUTABLE_IDENTITY_INVALID) as caught:
            executor.execute_ncu_action(
                _identity(ncu),
                ["--read-fds", report.proc_path],
                timeout_seconds=2,
                retained_report=report,
            )
        assert any(
            executor.NCU_ACTION_INPUT_IDENTITY_INVALID in note
            for note in getattr(caught.value, "__notes__", [])
        )
    finally:
        report.close()
        os.close(writer)


def test_auxiliary_descriptor_identity_drift_after_probe_prevents_action(
    tmp_path: Path, monkeypatch,
) -> None:
    ncu = _fixture_ncu(tmp_path / "ncu")
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    auxiliary = os.open(first, os.O_RDONLY)
    original = executor._run_bounded
    calls = 0

    def run(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            replacement = os.open(second, os.O_RDONLY)
            try:
                os.dup2(replacement, auxiliary)
            finally:
                if replacement != auxiliary:
                    os.close(replacement)
        return result

    monkeypatch.setattr(executor, "_run_bounded", run)
    try:
        with _assert_code(executor.NCU_ACTION_INPUT_IDENTITY_INVALID):
            executor.execute_ncu_action(
                _identity(ncu),
                ["--read-fds", executor.proc_fd_path(auxiliary)],
                timeout_seconds=2,
                auxiliary_descriptors=(auxiliary,),
            )
        assert calls == 1
    finally:
        os.close(auxiliary)


def test_borrowed_descriptors_remain_open_after_executor_failure(
    tmp_path: Path,
) -> None:
    ncu = _fixture_ncu(tmp_path / "ncu")
    report_path = tmp_path / "report.ncu-rep"
    report_path.write_bytes(b"report bytes")
    auxiliary_path = tmp_path / "auxiliary.txt"
    auxiliary_path.write_bytes(b"auxiliary bytes")
    auxiliary = os.open(auxiliary_path, os.O_RDONLY)
    try:
        with immutable.open_retained_regular(report_path) as report:
            with _assert_code(executor.NCU_VERSION_LINE_MISMATCH):
                executor.execute_ncu_action(
                    _identity(ncu, version="Version 2026.1.0.1 (different)"),
                    [
                        "--read-fds", report.proc_path,
                        executor.proc_fd_path(auxiliary),
                    ],
                    timeout_seconds=2,
                    retained_report=report,
                    auxiliary_descriptors=(auxiliary,),
                )
            os.fstat(report.descriptor)
            os.fstat(auxiliary)
    finally:
        os.close(auxiliary)


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

    def reverify(retained, borrowed, *, pending=None):
        if pending is not None:
            ncu.chmod(0o755)
            ncu.write_bytes(b"late mutation")
            ncu.chmod(0o555)
        return original_reverify(retained, borrowed, pending=pending)

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
