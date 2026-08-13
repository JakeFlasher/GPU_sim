"""Run Linux ``perf`` directly around a research command.

This module is deliberately an ordinary tool wrapper. It records no source
authority, clean-tree state, receipt, seal, digest, or publication claim.
"""

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
from typing import Callable, Mapping, Sequence


class ProfileError(RuntimeError):
    """A direct profiling command could not complete."""


@dataclass(slots=True)
class ProfileResult:
    command: list[str]
    record_command: list[str]
    report_command: list[str]
    record_return_code: int
    report_return_code: int
    output: str
    perf_data: str
    report: str
    started_at: str
    finished_at: str


Runner = Callable[..., subprocess.CompletedProcess[str]]


_OUTPUT_NAMES = (
    "perf.data",
    "perf.data.old",
    "program.stdout.txt",
    "perf-record.stderr.txt",
    "perf.report.txt",
    "perf-report.stderr.txt",
    "command.txt",
    "metadata.json",
)


def build_record_command(
    *,
    perf: str,
    perf_data: Path,
    event: str,
    call_graph: str | None,
    command: Sequence[str],
) -> list[str]:
    if not command:
        raise ProfileError("a command is required after --")
    if not event:
        raise ProfileError("the perf event must not be empty")
    result = [
        perf,
        "record",
        "--no-buildid-cache",
        "--event",
        event,
        "--output",
        str(perf_data),
    ]
    if call_graph:
        result.extend(("--call-graph", call_graph))
    result.append("--")
    result.extend(command)
    return result


def profile(
    *,
    command: Sequence[str],
    output: Path,
    perf: str = "/usr/bin/perf",
    event: str = "cycles:u",
    call_graph: str | None = "dwarf",
    timeout: float = 1800.0,
    cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
    runner: Runner = subprocess.run,
) -> ProfileResult:
    if timeout <= 0:
        raise ProfileError("timeout must be positive")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in _OUTPUT_NAMES:
        path = output / name
        if path.is_file() or path.is_symlink():
            path.unlink()

    perf_data = output / "perf.data"
    report_path = output / "perf.report.txt"
    record_command = build_record_command(
        perf=perf,
        perf_data=perf_data,
        event=event,
        call_graph=call_graph,
        command=command,
    )
    run_environment = dict(os.environ if environment is None else environment)
    started = datetime.now(timezone.utc).isoformat()
    try:
        recorded = runner(
            record_command,
            cwd=cwd,
            env=run_environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ProfileError(f"perf record could not run: {error}") from error

    (output / "program.stdout.txt").write_text(recorded.stdout, encoding="utf-8")
    (output / "perf-record.stderr.txt").write_text(recorded.stderr, encoding="utf-8")
    (output / "command.txt").write_text(
        shlex.join(record_command) + "\n", encoding="utf-8"
    )
    if recorded.returncode != 0:
        raise ProfileError(
            f"perf record exited with status {recorded.returncode}; "
            f"see {output / 'perf-record.stderr.txt'}"
        )
    if not perf_data.is_file():
        raise ProfileError(f"perf record completed but did not create {perf_data}")

    report_command = [
        perf,
        "report",
        "--stdio",
        "--no-children",
        "--input",
        str(perf_data),
    ]
    try:
        reported = runner(
            report_command,
            cwd=cwd,
            env=run_environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ProfileError(f"perf report could not run: {error}") from error
    report_path.write_text(reported.stdout, encoding="utf-8")
    (output / "perf-report.stderr.txt").write_text(
        reported.stderr, encoding="utf-8"
    )
    finished = datetime.now(timezone.utc).isoformat()
    result = ProfileResult(
        command=list(command),
        record_command=record_command,
        report_command=report_command,
        record_return_code=recorded.returncode,
        report_return_code=reported.returncode,
        output=str(output),
        perf_data=str(perf_data),
        report=str(report_path),
        started_at=started,
        finished_at=finished,
    )
    (output / "metadata.json").write_text(
        json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8"
    )
    if reported.returncode != 0:
        raise ProfileError(
            f"perf report exited with status {reported.returncode}; "
            f"see {output / 'perf-report.stderr.txt'}"
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--perf", default="/usr/bin/perf")
    parser.add_argument("--event", default="cycles:u")
    parser.add_argument(
        "--call-graph",
        choices=("dwarf", "fp", "lbr", "none"),
        default="dwarf",
    )
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        result = profile(
            command=command,
            output=args.output,
            perf=args.perf,
            event=args.event,
            call_graph=None if args.call_graph == "none" else args.call_graph,
            timeout=args.timeout,
            cwd=args.cwd,
        )
    except ProfileError as error:
        print(f"profile: {error}", file=sys.stderr)
        return 2
    print(result.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
