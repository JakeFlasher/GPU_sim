#!/usr/bin/env python3
"""Run Nsight Compute directly and export a machine-readable CSV report."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


class CaptureError(RuntimeError):
    """NCU could not produce the requested experiment output."""


@dataclass(slots=True)
class CaptureResult:
    command: list[str]
    return_code: int
    report: str
    csv: str
    started_at: str
    finished_at: str


def build_capture_command(
    *,
    ncu: str,
    report_base: Path,
    metrics: str | None,
    sections: Sequence[str],
    launch_skip: int,
    launch_count: int | None,
    replay_mode: str | None,
    clock_control: str | None,
    target: Sequence[str],
) -> list[str]:
    if not target:
        raise CaptureError("a target command is required after --")
    if launch_skip < 0:
        raise CaptureError("--launch-skip must be non-negative")
    if launch_count is not None and launch_count < 1:
        raise CaptureError("--launch-count must be positive")

    command = [
        ncu,
        "--force-overwrite",
        "--export",
        str(report_base),
        "--launch-skip",
        str(launch_skip),
    ]
    if launch_count is not None:
        command.extend(("--launch-count", str(launch_count)))
    if replay_mode:
        command.extend(("--replay-mode", replay_mode))
    if clock_control:
        command.extend(("--clock-control", clock_control))
    if metrics:
        command.extend(("--metrics", metrics))
    for section in sections:
        command.extend(("--section", section))
    command.extend(target)
    return command


def collect(
    *,
    output: Path,
    ncu: str,
    metrics: str | None,
    sections: Sequence[str],
    launch_skip: int,
    launch_count: int | None,
    replay_mode: str | None,
    clock_control: str | None,
    target: Sequence[str],
    environment: dict[str, str] | None = None,
) -> CaptureResult:
    output.mkdir(parents=True, exist_ok=True)
    report_base = output / "ncu-report"
    report = output / "ncu-report.ncu-rep"
    csv_path = output / "ncu.csv"
    command = build_capture_command(
        ncu=ncu,
        report_base=report_base,
        metrics=metrics,
        sections=sections,
        launch_skip=launch_skip,
        launch_count=launch_count,
        replay_mode=replay_mode,
        clock_control=clock_control,
        target=target,
    )

    started = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run(
        command,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    finished = datetime.now(timezone.utc).isoformat()
    (output / "ncu.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output / "ncu.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    (output / "command.txt").write_text(
        shlex.join(command) + "\n", encoding="utf-8"
    )

    result = CaptureResult(
        command=command,
        return_code=completed.returncode,
        report=str(report),
        csv=str(csv_path),
        started_at=started,
        finished_at=finished,
    )
    (output / "metadata.json").write_text(
        json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8"
    )
    if completed.returncode != 0:
        raise CaptureError(
            f"NCU exited with status {completed.returncode}; see {output / 'ncu.stderr.txt'}"
        )
    if not report.is_file():
        raise CaptureError(f"NCU completed but did not create {report}")

    export_command = [
        ncu,
        "--import",
        str(report),
        "--csv",
        "--page",
        "raw",
        "--log-file",
        str(csv_path),
    ]
    (output / "export_command.txt").write_text(
        shlex.join(export_command) + "\n", encoding="utf-8"
    )
    exported = subprocess.run(
        export_command,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (output / "export.stdout.txt").write_text(exported.stdout, encoding="utf-8")
    (output / "export.stderr.txt").write_text(exported.stderr, encoding="utf-8")
    if exported.returncode != 0 or not csv_path.is_file():
        raise CaptureError(
            f"NCU report export failed with status {exported.returncode}; "
            f"see {output / 'export.stderr.txt'}"
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ncu", default=os.environ.get("NCU", "ncu"))
    parser.add_argument("--metrics")
    parser.add_argument("--section", action="append", default=[])
    parser.add_argument("--launch-skip", type=int, default=0)
    parser.add_argument("--launch-count", type=int, default=1)
    parser.add_argument("--replay-mode", choices=("kernel", "application", "range"))
    parser.add_argument("--clock-control", choices=("base", "boost", "none", "reset"))
    parser.add_argument("target", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    target = args.target[1:] if args.target[:1] == ["--"] else args.target
    try:
        collect(
            output=args.output,
            ncu=args.ncu,
            metrics=args.metrics,
            sections=args.section,
            launch_skip=args.launch_skip,
            launch_count=args.launch_count,
            replay_mode=args.replay_mode,
            clock_control=args.clock_control,
            target=target,
            environment=dict(os.environ),
        )
    except CaptureError as error:
        print(f"collect_ncu: {error}", file=sys.stderr)
        return 2
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
