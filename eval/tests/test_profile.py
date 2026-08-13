from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from eval.profile import ProfileError, build_record_command, profile


def test_record_command_is_direct() -> None:
    command = build_record_command(
        perf="perf",
        perf_data=Path("out/perf.data"),
        event="cycles:u",
        call_graph="dwarf",
        command=["sim-run", "--help"],
    )
    assert command == [
        "perf",
        "record",
        "--no-buildid-cache",
        "--event",
        "cycles:u",
        "--output",
        "out/perf.data",
        "--call-graph",
        "dwarf",
        "--",
        "sim-run",
        "--help",
    ]


def test_profile_writes_plain_outputs(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[1] == "record":
            Path(command[command.index("--output") + 1]).write_bytes(b"perf data")
            return subprocess.CompletedProcess(command, 0, "program output\n", "record note\n")
        return subprocess.CompletedProcess(command, 0, "report output\n", "")

    result = profile(
        command=["sim-run", "--help"],
        output=tmp_path / "profile",
        perf="perf",
        runner=runner,
        environment={},
    )

    assert len(calls) == 2
    assert Path(result.report).read_text(encoding="utf-8") == "report output\n"
    assert (tmp_path / "profile/program.stdout.txt").read_text() == "program output\n"
    metadata = json.loads((tmp_path / "profile/metadata.json").read_text())
    assert metadata["command"] == ["sim-run", "--help"]
    assert "receipt" not in metadata
    assert "authority" not in metadata


def test_profile_reports_record_failure(tmp_path: Path) -> None:
    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 3, "", "permission denied\n")

    with pytest.raises(ProfileError, match="status 3"):
        profile(
            command=["sim-run"],
            output=tmp_path / "profile",
            perf="perf",
            runner=runner,
            environment={},
        )
    assert "permission denied" in (tmp_path / "profile/perf-record.stderr.txt").read_text()


def test_empty_command_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ProfileError, match="command is required"):
        profile(command=[], output=tmp_path, environment={})
