from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from tools.local_labels import capture_cutlass_successor_receipts as capture
from tools.local_labels import delegated_cgroup as cgroup
from tools.local_labels import ncu_action_executor as executor
from tools.nvbit_tracer import capture_process


def _write_proc_contracts(
    tmp_path: Path, monkeypatch, membership: bytes,
) -> Path:
    mount = tmp_path / "cgroup"
    mount.mkdir()
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        f"1 2 0:1 / {mount} rw - cgroup2 cgroup2 rw\n"
    )
    current = tmp_path / "self-cgroup"
    current.write_bytes(membership)
    monkeypatch.setattr(cgroup, "_CGROUP_MOUNT", mount)
    monkeypatch.setattr(cgroup, "_PROC_SELF_MOUNTINFO", mountinfo)
    monkeypatch.setattr(cgroup, "_PROC_SELF_CGROUP", current)
    return mount


def _scope() -> cgroup.DelegatedCgroupV2:
    scope = object.__new__(cgroup.DelegatedCgroupV2)
    scope._parent_fd = 101
    scope._leaf_fd = 102
    scope._procs_fd = -1
    scope._kill_fd = 103
    scope._events_fd = 104
    scope._leaf_name = "leaf"
    scope._parent_cgroup = "/delegated"
    scope._cgroup_inode = 105
    scope._launched = True
    scope._empty = False
    scope._removed = False
    return scope


def test_bounded_proc_reader_joins_short_reads(tmp_path: Path, monkeypatch):
    payload = b"0::/delegated/example\n"
    path = tmp_path / "proc-file"
    path.write_bytes(payload)
    real_read = os.read

    def short_read(descriptor: int, count: int) -> bytes:
        return real_read(descriptor, min(count, 2))

    monkeypatch.setattr(cgroup.os, "read", short_read)

    assert cgroup._read_bounded(path, "test proc file") == payload


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"0::relative\n",
        b"0::/one\n0::/two\n",
        b"1:name=/legacy\n",
        b"0::/bad/../path\n",
        b"0::/bad/./path\n",
        b"\xff\n",
    ],
)
def test_malformed_unified_membership_fails_closed(
    tmp_path: Path, monkeypatch, payload: bytes,
):
    path = tmp_path / "self-cgroup"
    path.write_bytes(payload)
    monkeypatch.setattr(cgroup, "_PROC_SELF_CGROUP", path)

    with pytest.raises(cgroup.CgroupContainmentError):
        cgroup._current_cgroup_components()


def test_non_cgroup_v2_mount_fails_closed(tmp_path: Path, monkeypatch):
    path = tmp_path / "mountinfo"
    path.write_text("1 2 0:1 / /sys/fs/cgroup rw - tmpfs tmpfs rw\n")
    monkeypatch.setattr(cgroup, "_PROC_SELF_MOUNTINFO", path)

    with pytest.raises(cgroup.CgroupContainmentError, match="not one exact"):
        cgroup._require_cgroup_v2_mount()


def test_multithreaded_parent_fails_before_cgroup_discovery(monkeypatch):
    monkeypatch.setattr(cgroup.threading, "active_count", lambda: 2)
    monkeypatch.setattr(
        cgroup,
        "_require_cgroup_v2_mount",
        lambda: pytest.fail("cgroup discovery must not run"),
    )

    with pytest.raises(cgroup.CgroupContainmentError, match="single-threaded"):
        cgroup.DelegatedCgroupV2.create()


def test_missing_delegated_controls_remove_partial_leaf(
    tmp_path: Path, monkeypatch,
):
    mount = _write_proc_contracts(tmp_path, monkeypatch, b"0::/delegated\n")
    parent = mount / "delegated"
    parent.mkdir()
    monkeypatch.setattr(cgroup.secrets, "token_hex", lambda _length: "a" * 16)

    with pytest.raises(cgroup.CgroupContainmentError, match="cgroup.procs"):
        cgroup.DelegatedCgroupV2.create()

    assert list(parent.iterdir()) == []


def test_create_preserves_operator_interruption_during_leaf_creation(
    tmp_path: Path, monkeypatch,
):
    mount = _write_proc_contracts(tmp_path, monkeypatch, b"0::/delegated\n")
    parent = mount / "delegated"
    parent.mkdir()
    interruption = KeyboardInterrupt("synthetic cgroup creation interruption")
    monkeypatch.setattr(
        cgroup.os,
        "mkdir",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(interruption),
    )

    with pytest.raises(KeyboardInterrupt) as failure:
        cgroup.DelegatedCgroupV2.create()

    assert failure.value is interruption


def test_create_preserves_capture_deadline_during_leaf_creation(
    tmp_path: Path, monkeypatch,
):
    mount = _write_proc_contracts(tmp_path, monkeypatch, b"0::/delegated\n")
    parent = mount / "delegated"
    parent.mkdir()
    deadline = capture_process.CaptureLaunchTimeout("synthetic deadline")
    monkeypatch.setattr(
        cgroup.os,
        "mkdir",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(deadline),
    )

    with pytest.raises(capture_process.CaptureLaunchTimeout) as failure:
        cgroup.DelegatedCgroupV2.create()

    assert failure.value is deadline


def test_required_capture_containment_never_falls_back(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        cgroup.DelegatedCgroupV2,
        "create",
        lambda: (_ for _ in ()).throw(
            cgroup.CgroupContainmentError("delegation unavailable")
        ),
    )
    monkeypatch.setattr(
        capture.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("uncontained launch must not run"),
    )

    with pytest.raises(capture.CaptureError, match="containment is unavailable"):
        capture._run_capture_command(
            [sys.executable, "-c", "pass"],
            dict(os.environ),
            tmp_path,
            require_cgroup_containment=True,
        )


def test_required_ncu_containment_never_falls_back(monkeypatch):
    monkeypatch.setattr(
        cgroup.DelegatedCgroupV2,
        "create",
        lambda: (_ for _ in ()).throw(
            cgroup.CgroupContainmentError("delegation unavailable")
        ),
    )
    monkeypatch.setattr(
        executor.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("uncontained launch must not run"),
    )

    with pytest.raises(
        executor.NcuActionExecutorError, match="cannot establish delegated",
    ):
        executor.execute_bounded_command(
            (sys.executable, "-c", "pass"),
            environment=executor.ncu_action_environment(),
            working_directory="/",
            inherited_descriptors=(),
            timeout_seconds=1,
            max_output_bytes=1024,
            timeout_code=executor.NCU_ACTION_TIMEOUT,
            launch_code=executor.NCU_ACTION_LAUNCH_FAILED,
            require_cgroup_containment=True,
        )


def test_kill_failure_retains_cleanup_capabilities(monkeypatch):
    scope = _scope()
    monkeypatch.setattr(
        scope,
        "kill_all",
        lambda: (_ for _ in ()).throw(
            cgroup.CgroupContainmentError("synthetic cgroup.kill failure")
        ),
    )
    monkeypatch.setattr(
        cgroup.os,
        "close",
        lambda _descriptor: pytest.fail("controls must remain retained"),
    )
    monkeypatch.setattr(
        cgroup.os,
        "rmdir",
        lambda *_args, **_kwargs: pytest.fail("populated leaf must not be removed"),
    )

    with pytest.raises(cgroup.CgroupContainmentError, match="cgroup.kill failure"):
        scope.finish()

    assert scope._empty is False
    assert scope._removed is False
    assert scope._kill_fd == 103


def test_populated_leaf_retains_cleanup_capabilities(monkeypatch):
    scope = _scope()
    monkeypatch.setattr(scope, "kill_all", lambda: None)
    monkeypatch.setattr(
        scope,
        "wait_empty",
        lambda: (_ for _ in ()).throw(
            cgroup.CgroupContainmentError("remained populated")
        ),
    )
    monkeypatch.setattr(
        cgroup.os,
        "close",
        lambda _descriptor: pytest.fail("controls must remain retained"),
    )

    with pytest.raises(cgroup.CgroupContainmentError, match="remained populated"):
        scope.finish()

    assert scope._empty is False
    assert scope._removed is False


def test_leaf_removal_failure_is_retryable(monkeypatch):
    scope = _scope()
    kills = 0
    removals = 0
    closed: list[int] = []

    def kill() -> None:
        nonlocal kills
        kills += 1

    def remove(_name: str, *, dir_fd: int) -> None:
        nonlocal removals
        assert dir_fd == 101
        removals += 1
        if removals == 1:
            raise OSError("synthetic removal race")

    monkeypatch.setattr(scope, "kill_all", kill)
    monkeypatch.setattr(scope, "wait_empty", lambda: None)
    monkeypatch.setattr(cgroup.os, "close", closed.append)
    monkeypatch.setattr(cgroup.os, "rmdir", remove)

    with pytest.raises(cgroup.CgroupContainmentError, match="removal race"):
        scope.finish()

    assert scope._empty is True
    assert scope._removed is False
    assert scope._parent_fd == 101
    evidence = scope.finish()
    assert evidence.terminal_populated_zero is True
    assert evidence.leaf_removed is True
    assert kills == 1
    assert removals == 2
    assert closed == [104, 103, 102, 101]


def test_popen_interruption_cleans_launched_leaf(monkeypatch):
    scope = _scope()
    scope._procs_fd = 106
    scope._launched = False
    closed: list[int] = []
    cleaned: list[bool] = []
    monkeypatch.setattr(cgroup.threading, "active_count", lambda: 1)
    monkeypatch.setattr(cgroup.os, "close", closed.append)
    monkeypatch.setattr(
        cgroup.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    monkeypatch.setattr(scope, "close_after_error", lambda: cleaned.append(True))

    with pytest.raises(KeyboardInterrupt):
        scope.popen([sys.executable, "-c", "pass"])

    assert scope._launched is True
    assert scope._procs_fd == -1
    assert closed == [106]
    assert cleaned == [True]


def test_popen_cleanup_failure_supersedes_interruption(monkeypatch):
    scope = _scope()
    scope._procs_fd = 106
    scope._launched = False
    cleanup_error = RuntimeError("synthetic launch cleanup failure")
    monkeypatch.setattr(cgroup.threading, "active_count", lambda: 1)
    monkeypatch.setattr(cgroup.os, "close", lambda _descriptor: None)
    monkeypatch.setattr(
        cgroup.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            KeyboardInterrupt("synthetic launch interruption")
        ),
    )
    monkeypatch.setattr(
        scope,
        "close_after_error",
        lambda: (_ for _ in ()).throw(cleanup_error),
    )

    with pytest.raises(
        cgroup.CgroupContainmentError,
        match="delegated cgroup launch cleanup failed",
    ) as failure:
        scope.popen([sys.executable, "-c", "pass"])

    assert failure.value.__cause__ is cleanup_error
    assert any(
        "active delegated cgroup launch error: synthetic launch interruption" in note
        for note in failure.value.__notes__
    )


def test_failed_preexec_placement_is_contained(tmp_path: Path, monkeypatch):
    if os.environ.get("GPU_SIM_DELEGATED_CGROUP_TEST") != "1":
        pytest.skip("requires a systemd-delegated cgroup-v2 test service")
    scope = cgroup.DelegatedCgroupV2.create()
    placement_fd = scope._procs_fd
    real_write = os.write

    def refuse_placement(descriptor: int, payload: bytes) -> int:
        if descriptor == placement_fd:
            raise OSError("synthetic placement refusal")
        return real_write(descriptor, payload)

    monkeypatch.setattr(cgroup.os, "write", refuse_placement)
    with pytest.raises(cgroup.CgroupContainmentError, match="cannot launch process"):
        scope.popen(
            [sys.executable, "-c", "pass"],
            cwd=str(tmp_path),
            stdin=capture.subprocess.DEVNULL,
            stdout=capture.subprocess.PIPE,
            stderr=capture.subprocess.PIPE,
            start_new_session=True,
        )
    scope.close_after_error()
    assert scope._removed is True
