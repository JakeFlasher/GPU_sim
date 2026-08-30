#!/usr/bin/env python3
"""Run Nsight Compute directly and export a machine-readable CSV report."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
from contextlib import suppress
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


DEFAULT_TIMEOUT_SECONDS = 900
MAX_TIMEOUT_SECONDS = 36_000
TERMINATION_GRACE_SECONDS = 5


@dataclass(slots=True)
class BoundedProcessResult:
    return_code: int
    stdout: str
    stderr: str
    timed_out: bool


def _terminate_process(process: subprocess.Popen[str]) -> None:
    """Terminate a process group and reap it after a bounded timeout."""

    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=TERMINATION_GRACE_SECONDS)
    except (OSError, subprocess.TimeoutExpired):
        try:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=TERMINATION_GRACE_SECONDS)


def _run_bounded(
    command: Sequence[str],
    *,
    environment: dict[str, str] | None,
    timeout_seconds: int,
) -> BoundedProcessResult:
    """Run one NCU subprocess with a process-group deadline."""

    try:
        process = subprocess.Popen(
            list(command),
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=os.name != "nt",
        )
    except OSError as exc:
        raise CaptureError(f"unable to launch NCU command: {exc}") from exc
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        return BoundedProcessResult(
            process.returncode or 0,
            stdout,
            stderr,
            False,
        )
    except subprocess.TimeoutExpired as exc:
        _terminate_process(process)
        try:
            stdout, stderr = process.communicate(timeout=TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            stdout = exc.output or ""
            stderr = exc.stderr or ""
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
        return BoundedProcessResult(
            process.returncode if process.returncode is not None else -signal.SIGTERM,
            stdout,
            stderr,
            True,
        )


def _write_metadata(output: Path, result: CaptureResult, **extra: object) -> None:
    payload = asdict(result)
    payload.update(extra)
    (output / "metadata.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


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
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> CaptureResult:
    if (
        type(timeout_seconds) is not int
        or not 1 <= timeout_seconds <= MAX_TIMEOUT_SECONDS
    ):
        raise CaptureError(
            f"--timeout must be an integer between 1 and {MAX_TIMEOUT_SECONDS} seconds"
        )
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
    completed = _run_bounded(
        command,
        environment=environment,
        timeout_seconds=timeout_seconds,
    )
    finished = datetime.now(timezone.utc).isoformat()
    (output / "ncu.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output / "ncu.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    (output / "command.txt").write_text(
        shlex.join(command) + "\n", encoding="utf-8"
    )

    result = CaptureResult(
        command=command,
        return_code=completed.return_code,
        report=str(report),
        csv=str(csv_path),
        started_at=started,
        finished_at=finished,
    )
    _write_metadata(
        output,
        result,
        timeout_seconds=timeout_seconds,
        timed_out=completed.timed_out,
        timeout_stage="capture" if completed.timed_out else None,
        export_command=None,
        export_return_code=None,
        export_timed_out=False,
    )
    if completed.timed_out:
        raise CaptureError(
            f"NCU capture timed out after {timeout_seconds} seconds; "
            f"see {output / 'ncu.stderr.txt'}"
        )
    if completed.return_code != 0:
        raise CaptureError(
            f"NCU exited with status {completed.return_code}; see {output / 'ncu.stderr.txt'}"
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
    exported = _run_bounded(
        export_command,
        environment=environment,
        timeout_seconds=timeout_seconds,
    )
    (output / "export.stdout.txt").write_text(exported.stdout, encoding="utf-8")
    (output / "export.stderr.txt").write_text(exported.stderr, encoding="utf-8")
    _write_metadata(
        output,
        result,
        timeout_seconds=timeout_seconds,
        timed_out=exported.timed_out,
        timeout_stage="export" if exported.timed_out else None,
        export_command=export_command,
        export_return_code=exported.return_code,
        export_timed_out=exported.timed_out,
    )
    if exported.timed_out:
        raise CaptureError(
            f"NCU export timed out after {timeout_seconds} seconds; "
            f"see {output / 'export.stderr.txt'}"
        )
    if exported.return_code != 0 or not csv_path.is_file():
        raise CaptureError(
            f"NCU report export failed with status {exported.return_code}; "
            f"see {output / 'export.stderr.txt'}"
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ncu", default=os.environ.get("NCU", "ncu"))
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
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
            timeout_seconds=args.timeout,
        )
    except CaptureError as error:
        print(f"collect_ncu: {error}", file=sys.stderr)
        return 2
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
