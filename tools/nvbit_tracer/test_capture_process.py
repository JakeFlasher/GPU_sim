from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tools.local_labels import delegated_cgroup
from tools.nvbit_tracer import capture_process


def _evidence() -> delegated_cgroup.CgroupContainmentEvidence:
    return delegated_cgroup.CgroupContainmentEvidence(
        kind=delegated_cgroup.CGROUP_V2_KILL_V1,
        parent_cgroup_sha256="a" * 64,
        cgroup_inode=123,
        terminal_populated_zero=True,
        leaf_removed=True,
    )


def test_required_containment_failure_never_launches(monkeypatch) -> None:
    monkeypatch.setattr(
        delegated_cgroup.DelegatedCgroupV2,
        "create",
        lambda: (_ for _ in ()).throw(
            delegated_cgroup.CgroupContainmentError("delegation unavailable")
        ),
    )
    monkeypatch.setattr(
        capture_process.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("uncontained launch must not run"),
    )

    with pytest.raises(
        capture_process.NvbitCaptureProcessError,
        match="cannot establish NVBit delegated cgroup containment",
    ):
        capture_process.execute_capture_process(
            (sys.executable, "-c", "pass"),
            environment=dict(os.environ),
            working_directory="/",
            inherited_descriptors=(),
            timeout_seconds=1,
            max_output_bytes=1024,
            timeout_code="fixture_timeout",
            launch_code="fixture_launch_failed",
        )


def test_success_kills_session_escaped_descendant_retaining_pipe(
    tmp_path: Path, monkeypatch,
) -> None:
    pid_path = tmp_path / "descendant.pid"

    class TrackingContainment:
        def __init__(self) -> None:
            self.kill_calls = 0
            self.finished = False

        def popen(self, command, **kwargs):
            return subprocess.Popen(command, **kwargs)

        def kill_all(self) -> None:
            self.kill_calls += 1
            deadline = time.monotonic() + 2
            while not pid_path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            if pid_path.exists():
                try:
                    os.kill(int(pid_path.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass

        def finish(self):
            self.kill_all()
            self.finished = True
            return _evidence()

        def close_after_error(self) -> None:
            self.finished = True

    containment = TrackingContainment()
    monkeypatch.setattr(
        delegated_cgroup.DelegatedCgroupV2,
        "create",
        lambda: containment,
    )
    child_source = (
        "import os,sys,time; from pathlib import Path; "
        "os.setsid(); Path(sys.argv[1]).write_text(str(os.getpid())); "
        "print('descendant-ready', flush=True); time.sleep(60)"
    )
    leader_source = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',{child_source!r},sys.argv[1]]); "
        "time.sleep(0.1)"
    )

    result = capture_process.execute_capture_process(
        (sys.executable, "-c", leader_source, str(pid_path)),
        environment=dict(os.environ),
        working_directory=str(tmp_path),
        inherited_descriptors=(),
        timeout_seconds=3,
        max_output_bytes=4096,
        timeout_code="fixture_timeout",
        launch_code="fixture_launch_failed",
    )

    assert result.returncode == 0
    assert b"descendant-ready" in result.stdout
    assert result.containment == _evidence()
    assert containment.kill_calls >= 1
    assert containment.finished is True
    descendant_pid = int(pid_path.read_text())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(descendant_pid, 0)
        except ProcessLookupError:
            break
        try:
            fields = Path(f"/proc/{descendant_pid}/stat").read_text().split()
        except FileNotFoundError:
            break
        if fields[2] == "Z":
            break
        time.sleep(0.02)
    else:
        pytest.fail(f"session-escaped descendant survived: {descendant_pid}")


def test_cleanup_failure_supersedes_interruption_and_retains_identity(
    monkeypatch,
) -> None:
    cleanup_error = RuntimeError("synthetic cgroup cleanup failure")

    class Process:
        pid = 4321

    class Containment:
        def popen(self, _command, **_kwargs):
            return Process()

        def finish(self):
            raise cleanup_error

    monkeypatch.setattr(
        delegated_cgroup.DelegatedCgroupV2,
        "create",
        lambda: Containment(),
    )
    monkeypatch.setattr(
        capture_process,
        "_collect_bounded",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            KeyboardInterrupt("synthetic operator interruption")
        ),
    )

    with pytest.raises(
        capture_process.NvbitCaptureProcessError,
        match="delegated cgroup cleanup failed",
    ) as failure:
        capture_process.execute_capture_process(
            ("/fixture/target",),
            environment={"PATH": "/bin:/usr/bin"},
            working_directory="/",
            inherited_descriptors=(),
            timeout_seconds=1,
            max_output_bytes=1024,
            timeout_code="fixture_timeout",
            launch_code="fixture_launch_failed",
        )

    assert failure.value.code == "fixture_launch_failed"
    assert failure.value.process_cleanup_incomplete is True
    assert failure.value.process_id == 4321
    assert failure.value.__cause__ is cleanup_error
    assert any(
        "active NVBit capture error: synthetic operator interruption" in note
        for note in failure.value.__notes__
    )


def test_cleanup_failure_retains_output_overflow_lower_bound(monkeypatch) -> None:
    cleanup_error = RuntimeError("synthetic cgroup cleanup failure")
    overflow = capture_process.NvbitCaptureProcessError(
        capture_process.NVBIT_CAPTURE_OUTPUT_LIMIT,
        "synthetic output overflow",
        observed_output_bytes_lower_bound=8193,
    )
    overflow.stdout = b"bounded-stdout-\xff"
    overflow.stderr = b"bounded-stderr-\x80"

    class Process:
        pid = 4321

    class Containment:
        def popen(self, _command, **_kwargs):
            return Process()

        def finish(self):
            raise cleanup_error

    monkeypatch.setattr(
        delegated_cgroup.DelegatedCgroupV2,
        "create",
        lambda: Containment(),
    )
    monkeypatch.setattr(
        capture_process,
        "_collect_bounded",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(overflow),
    )

    with pytest.raises(capture_process.NvbitCaptureProcessError) as failure:
        capture_process.execute_capture_process(
            ("/fixture/target",),
            environment={"PATH": "/bin:/usr/bin"},
            working_directory="/",
            inherited_descriptors=(),
            timeout_seconds=1,
            max_output_bytes=1024,
            timeout_code="fixture_timeout",
            launch_code="fixture_launch_failed",
        )

    assert failure.value.code == "fixture_launch_failed"
    assert failure.value.process_cleanup_incomplete is True
    assert failure.value.observed_output_bytes_lower_bound == 8193
    assert failure.value.stdout == b"bounded-stdout-\xff"
    assert failure.value.stderr == b"bounded-stderr-\x80"


def test_successful_child_with_failed_containment_preserves_streams(
    monkeypatch,
) -> None:
    cleanup_error = RuntimeError("synthetic successful-child cleanup failure")

    class Process:
        pid = 9876

    class Containment:
        def popen(self, _command, **_kwargs):
            return Process()

        def finish(self):
            raise cleanup_error

    monkeypatch.setattr(
        delegated_cgroup.DelegatedCgroupV2,
        "create",
        lambda: Containment(),
    )
    monkeypatch.setattr(
        capture_process,
        "_collect_bounded",
        lambda *_args, **_kwargs: capture_process.NvbitCaptureProcessResult(
            0,
            b"retained stdout",
            b"retained stderr",
            process_id=9876,
            started_at_utc="2026-08-04T00:00:00Z",
        ),
    )

    with pytest.raises(capture_process.NvbitCaptureProcessError) as failure:
        capture_process.execute_capture_process(
            ("/fixture/target",),
            environment={"PATH": "/bin:/usr/bin"},
            working_directory="/",
            inherited_descriptors=(),
            timeout_seconds=1,
            max_output_bytes=1024,
            timeout_code="fixture_timeout",
            launch_code="fixture_launch_failed",
        )

    assert failure.value.process_cleanup_incomplete is True
    assert failure.value.stdout == b"retained stdout"
    assert failure.value.stderr == b"retained stderr"
    assert failure.value.process_id == 9876
    assert failure.value.__cause__ is cleanup_error


def test_output_limit_retains_exact_bounded_prefix_and_finishes_cgroup(
    monkeypatch,
) -> None:
    class Containment:
        def __init__(self) -> None:
            self.finished = False

        def popen(self, command, **kwargs):
            return subprocess.Popen(command, **kwargs)

        def kill_all(self) -> None:
            return None

        def finish(self):
            self.finished = True
            return _evidence()

        def close_after_error(self) -> None:
            self.finished = True

    containment = Containment()
    monkeypatch.setattr(
        delegated_cgroup.DelegatedCgroupV2,
        "create",
        lambda: containment,
    )

    with pytest.raises(capture_process.NvbitCaptureProcessError) as failure:
        capture_process.execute_capture_process(
            (sys.executable, "-c", "import os; os.write(1, b'x' * 4096)"),
            environment=dict(os.environ),
            working_directory="/",
            inherited_descriptors=(),
            timeout_seconds=2,
            max_output_bytes=128,
            timeout_code="fixture_timeout",
            launch_code="fixture_launch_failed",
        )

    assert failure.value.code == capture_process.NVBIT_CAPTURE_OUTPUT_LIMIT
    assert failure.value.observed_output_bytes_lower_bound == 4096
    assert failure.value.stdout == b"x" * 128
    assert failure.value.stderr == b""
    assert failure.value.containment == _evidence()
    assert containment.finished is True


def test_kill_all_failure_still_reaps_exited_leader() -> None:
    cleanup_error = RuntimeError("synthetic cgroup.kill failure")

    class Containment:
        def kill_all(self) -> None:
            raise cleanup_error

    process = subprocess.Popen(
        (sys.executable, "-c", "pass"),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert capture_process._wait_for_exit_unreaped(process, 1)

    with pytest.raises(
        capture_process.NvbitCaptureProcessError,
        match="descendant termination failed",
    ) as failure:
        capture_process._stop_and_reap(process, Containment())

    assert failure.value.process_cleanup_incomplete is True
    assert failure.value.__cause__ is cleanup_error
    assert process.returncode is not None
    assert process.stdout is not None
    assert process.stderr is not None
    process.stdout.close()
    process.stderr.close()


def test_reaped_leader_allows_descendant_cleanup_retry() -> None:
    class Containment:
        def __init__(self) -> None:
            self.calls = 0

        def kill_all(self) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("synthetic transient cgroup.kill failure")

    containment = Containment()
    process = subprocess.Popen(
        (sys.executable, "-c", "pass"),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert capture_process._wait_for_exit_unreaped(process, 1)

    with pytest.raises(capture_process.NvbitCaptureProcessError):
        capture_process._stop_and_reap(process, containment)
    capture_process._stop_and_reap(process, containment)

    assert containment.calls == 2
    assert process.returncode is not None
    assert process.stdout is not None
    assert process.stderr is not None
    process.stdout.close()
    process.stderr.close()


def test_launch_timeout_closes_containment_and_uses_timeout_code(monkeypatch) -> None:
    class Containment:
        def __init__(self) -> None:
            self.closed = False

        def popen(self, _command, **_kwargs):
            time.sleep(60)
            pytest.fail("launch deadline did not interrupt stalled popen")

        def close_after_error(self) -> None:
            self.closed = True

    containment = Containment()
    monkeypatch.setattr(
        delegated_cgroup.DelegatedCgroupV2,
        "create",
        lambda: containment,
    )

    started = time.monotonic()
    with pytest.raises(capture_process.NvbitCaptureProcessError) as failure:
        capture_process.execute_capture_process(
            ("/fixture/target",),
            environment={"PATH": "/bin:/usr/bin"},
            working_directory="/",
            inherited_descriptors=(),
            timeout_seconds=0.05,
            max_output_bytes=1024,
            timeout_code="fixture_timeout",
            launch_code="fixture_launch_failed",
        )

    assert time.monotonic() - started < 1
    assert failure.value.code == "fixture_timeout"
    assert failure.value.process_cleanup_incomplete is False
    assert containment.closed is True


def test_containment_creation_is_bounded_by_capture_timeout(monkeypatch) -> None:
    def stalled_create():
        time.sleep(60)
        pytest.fail("capture deadline did not interrupt containment creation")

    monkeypatch.setattr(
        delegated_cgroup.DelegatedCgroupV2,
        "create",
        stalled_create,
    )

    started = time.monotonic()
    with pytest.raises(capture_process.NvbitCaptureProcessError) as failure:
        capture_process.execute_capture_process(
            ("/fixture/target",),
            environment={"PATH": "/bin:/usr/bin"},
            working_directory="/",
            inherited_descriptors=(),
            timeout_seconds=0.05,
            max_output_bytes=1024,
            timeout_code="fixture_timeout",
            launch_code="fixture_launch_failed",
        )

    assert time.monotonic() - started < 1
    assert failure.value.code == "fixture_timeout"


def test_launch_interruption_closes_containment_before_reraising(monkeypatch) -> None:
    interruption = KeyboardInterrupt("synthetic launch interruption")

    class Containment:
        def __init__(self) -> None:
            self.closed = False

        def popen(self, _command, **_kwargs):
            raise interruption

        def close_after_error(self) -> None:
            self.closed = True

    containment = Containment()
    monkeypatch.setattr(
        delegated_cgroup.DelegatedCgroupV2,
        "create",
        lambda: containment,
    )

    with pytest.raises(KeyboardInterrupt) as failure:
        capture_process.execute_capture_process(
            ("/fixture/target",),
            environment={"PATH": "/bin:/usr/bin"},
            working_directory="/",
            inherited_descriptors=(),
            timeout_seconds=1,
            max_output_bytes=1024,
            timeout_code="fixture_timeout",
            launch_code="fixture_launch_failed",
        )

    assert failure.value is interruption
    assert containment.closed is True
