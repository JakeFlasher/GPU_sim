from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools.local_labels.collect_ncu import CaptureError, build_capture_command, collect


def test_build_capture_command_is_direct() -> None:
    command = build_capture_command(
        ncu="ncu",
        report_base=Path("out/ncu-report"),
        metrics="a,b",
        sections=["MemoryWorkloadAnalysis"],
        launch_skip=2,
        launch_count=3,
        replay_mode="kernel",
        clock_control="none",
        target=["/tmp/workload", "--size", "8"],
    )
    assert command == [
        "ncu",
        "--force-overwrite",
        "--export",
        "out/ncu-report",
        "--launch-skip",
        "2",
        "--launch-count",
        "3",
        "--replay-mode",
        "kernel",
        "--clock-control",
        "none",
        "--metrics",
        "a,b",
        "--section",
        "MemoryWorkloadAnalysis",
        "/tmp/workload",
        "--size",
        "8",
    ]


@pytest.mark.parametrize(
    ("launch_skip", "launch_count", "target"),
    [(-1, 1, ["app"]), (0, 0, ["app"]), (0, 1, [])],
)
def test_invalid_capture_request_is_rejected(
    launch_skip: int, launch_count: int, target: list[str]
) -> None:
    with pytest.raises(CaptureError):
        build_capture_command(
            ncu="ncu",
            report_base=Path("out/ncu-report"),
            metrics=None,
            sections=[],
            launch_skip=launch_skip,
            launch_count=launch_count,
            replay_mode=None,
            clock_control=None,
            target=target,
        )


def _fake_ncu(path: Path, *, export_sleeps: bool = False) -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib,sys,time\n"
        "if '--import' in sys.argv:\n"
        f"    {'time.sleep(3)' if export_sleeps else 'pathlib.Path(sys.argv[sys.argv.index(\"--log-file\") + 1]).write_text(\"csv\")'}\n"
        "else:\n"
        "    report = pathlib.Path(sys.argv[sys.argv.index('--export') + 1] + '.ncu-rep')\n"
        "    report.parent.mkdir(parents=True, exist_ok=True)\n"
        "    report.write_bytes(b'report')\n"
        f"    time.sleep({0 if export_sleeps else 3})\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def test_capture_timeout_is_typed_and_reaps_child(tmp_path: Path) -> None:
    fake = _fake_ncu(tmp_path / "fake-ncu")
    output = tmp_path / "capture"
    with pytest.raises(CaptureError, match="capture timed out"):
        collect(
            output=output,
            ncu=str(fake),
            metrics="metric",
            sections=[],
            launch_skip=0,
            launch_count=1,
            replay_mode="application",
            clock_control="none",
            target=["app"],
            environment={},
            timeout_seconds=1,
        )
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["timed_out"] is True
    assert metadata["timeout_stage"] == "capture"
    assert metadata["timeout_seconds"] == 1


def test_export_timeout_is_typed_and_reaps_child(tmp_path: Path) -> None:
    fake = _fake_ncu(tmp_path / "fake-ncu", export_sleeps=True)
    output = tmp_path / "export"
    with pytest.raises(CaptureError, match="export timed out"):
        collect(
            output=output,
            ncu=str(fake),
            metrics="metric",
            sections=[],
            launch_skip=0,
            launch_count=1,
            replay_mode="application",
            clock_control="none",
            target=["app"],
            environment={},
            timeout_seconds=1,
        )
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["timed_out"] is True
    assert metadata["timeout_stage"] == "export"
    assert metadata["export_timed_out"] is True
    assert (output / "export_command.txt").is_file()
