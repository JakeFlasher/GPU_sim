"""Hermetic watchdog tests for the NCU and NVBit campaign launchers."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from tools.local_labels import collect_ncu
from tools.local_labels.collect_ncu import CaptureError
from workloads.research_probes import capture_sm86_nvbit


def _orphan_launcher(path: Path) -> Path:
    """Create a launcher whose child inherits both captured output pipes."""

    path.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, subprocess, sys, time\n"
        "pid_file = pathlib.Path(next(value for value in reversed(sys.argv[1:]) if value.endswith('.pid')))\n"
        "child = subprocess.Popen(['/usr/bin/python3', '-c', 'import time; time.sleep(60)'])\n"
        "pid_file.write_text(str(child.pid))\n"
        "time.sleep(0.05)\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _pid_from(path: Path) -> int:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if path.is_file():
            return int(path.read_text(encoding="utf-8"))
        time.sleep(0.01)
    raise AssertionError(f"descendant pid file was not created: {path}")


def _assert_pid_stopped(pid: int) -> None:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            return
        # A descendant reparented to init can briefly remain a zombie while
        # it is reaped; it no longer owns resources in that state.
        stat = Path(f"/proc/{pid}/stat")
        if stat.is_file():
            state = stat.read_text(encoding="utf-8", errors="replace").split()[2:3]
            if state == ["Z"]:
                return
        time.sleep(0.02)
    raise AssertionError(f"descendant process {pid} survived watchdog cleanup")


def test_collect_ncu_timeout_kills_descendant_after_leader_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _orphan_launcher(tmp_path / "launcher.py")
    pid_file = tmp_path / "ncu-child.pid"
    monkeypatch.setattr(collect_ncu, "TERMINATION_GRACE_SECONDS", 0.1)
    with pytest.raises(CaptureError, match="capture timed out"):
        collect_ncu.collect(
            output=tmp_path / "capture",
            ncu=str(launcher),
            metrics="metric",
            sections=[],
            launch_skip=0,
            launch_count=1,
            replay_mode="application",
            clock_control="none",
            target=[str(pid_file)],
            environment={},
            timeout_seconds=1,
        )
    _assert_pid_stopped(_pid_from(pid_file))
    metadata = json.loads(
        (tmp_path / "capture" / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["timed_out"] is True
    assert metadata["process_group_cleanup"] is True


def test_nvbit_timeout_kills_descendant_after_leader_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _orphan_launcher(tmp_path / "launcher.py")
    monkeypatch.setattr(capture_sm86_nvbit, "TERMINATION_GRACE_SECONDS", 0.1)
    monkeypatch.setattr(
        capture_sm86_nvbit,
        "_expected_nvbit_argv",
        lambda _cell: [str(launcher), str(tmp_path / "nvbit-child.pid")],
    )
    output_root = tmp_path / "campaign"
    with pytest.raises(RuntimeError, match="timed out"):
        capture_sm86_nvbit.capture("gemm_hmma_bf16_sm80", output_root, 1)
    destination = output_root / "evidence/raw/nvbit/gemm_hmma_bf16_sm80"
    pid = _pid_from(tmp_path / "nvbit-child.pid")
    _assert_pid_stopped(pid)
    record = json.loads((destination / "environment.json").read_text(encoding="utf-8"))
    assert record["timed_out"] is True
    assert record["timeout_seconds"] == 1
    assert record["timeout_policy"] == "process_group_terminate_reap_v1"
