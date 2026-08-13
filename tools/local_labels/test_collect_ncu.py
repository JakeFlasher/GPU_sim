from __future__ import annotations

from pathlib import Path

import pytest

from tools.local_labels.collect_ncu import CaptureError, build_capture_command


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
