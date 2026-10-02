#!/usr/bin/env python3
"""Run one CUDA workload with bounded resources and explicit outcomes.

This is an ordinary experiment runner, not an approval or evidence workflow.
It keeps the workload's stdout/stderr and reports why a run stopped.  In
particular, CUDA out-of-memory, unsupported/invalid shapes, timeouts, and
ordinary runtime failures are not collapsed into one generic failure and are
never silently replaced by a smaller workload.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "sm86_workload_run_v1"
MAX_TIMEOUT_SECONDS = 86_400
TERMINATION_GRACE_SECONDS = 5
STATUS_EXIT_CODES = {
    "passed": 0,
    "preflight_memory_low": 78,
    "oom": 79,
    "out_of_domain": 80,
    "timeout": 124,
    "runtime_error": 1,
    "host_query_error": 2,
    "launcher_error": 3,
}

_OOM_RE = re.compile(
    r"(?:out\s+of\s+memory|cuda(?:error)?memoryallocation|"
    r"cuda_error_out_of_memory|failed\s+to\s+allocate|\boom\b)",
    re.IGNORECASE,
)
_OOD_RE = re.compile(
    r"(?:out.of.domain|unknown_shape_point|shape_override|tile_alignment|"
    r"unsupported.*(?:shape|dimension|device)|invalid configuration|"
    r"no kernel image|too many resources requested|invalid device function|"
    r"must be a positive integer)",
    re.IGNORECASE,
)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _tail(path: Path, limit: int = 64 * 1024) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            return stream.read(limit).decode(errors="replace")
    except OSError:
        return ""


def _terminate_group(process: subprocess.Popen[Any]) -> bool:
    """Terminate and reap a workload process group within a fixed deadline."""

    cleanup_ok = True
    if os.name == "nt":
        with suppress(OSError, subprocess.TimeoutExpired):
            result = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=TERMINATION_GRACE_SECONDS,
            )
            cleanup_ok = result.returncode in (0, 128, 255)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            cleanup_ok = False
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=TERMINATION_GRACE_SECONDS)
    if process.poll() is None:
        cleanup_ok = False
        if os.name != "nt":
            with suppress(ProcessLookupError, OSError):
                os.killpg(process.pid, signal.SIGKILL)
        else:
            with suppress(OSError):
                process.kill()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=TERMINATION_GRACE_SECONDS)
    return cleanup_ok and process.poll() is not None


def query_gpu(
    *, nvidia_smi: str = "nvidia-smi", device_index: int = 0
) -> tuple[dict[str, Any] | None, str | None]:
    """Read one bounded, metadata-only GPU memory snapshot."""

    command = [
        nvidia_smi,
        f"--id={device_index}",
        "--query-gpu=index,name,uuid,memory.total,memory.free,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"nvidia-smi unavailable: {exc}"
    if result.returncode != 0:
        return None, result.stderr.strip() or "nvidia-smi returned a nonzero status"
    rows = list(csv.reader(result.stdout.splitlines()))
    if len(rows) != 1 or len(rows[0]) != 6:
        return None, "nvidia-smi returned an unexpected memory row"
    index, name, uuid, total, free, used = [part.strip() for part in rows[0]]
    try:
        parsed_index = int(index)
        total_mib = int(total)
        free_mib = int(free)
        used_mib = int(used)
    except ValueError as exc:
        return None, f"nvidia-smi memory values are not integers: {exc}"
    if parsed_index != device_index or min(total_mib, free_mib, used_mib) < 0:
        return None, "nvidia-smi returned an inconsistent device or memory row"
    return {
        "device_index": parsed_index,
        "name": name,
        "uuid": uuid,
        "total_mib": total_mib,
        "free_mib": free_mib,
        "used_mib": used_mib,
    }, None


def _classify_failure(
    *, returncode: int | None, timed_out: bool, stdout_path: Path, stderr_path: Path
) -> tuple[str, str | None]:
    if timed_out:
        return "timeout", "wall_clock_timeout"
    text = _tail(stderr_path) + "\n" + _tail(stdout_path)
    if _OOM_RE.search(text):
        return "oom", "cuda_memory_allocation_failure"
    if _OOD_RE.search(text):
        return "out_of_domain", "unsupported_or_invalid_shape_or_device"
    if returncode == 0:
        return "passed", None
    return "runtime_error", f"child_exit_{returncode}"


def run_workload(
    command: Sequence[str],
    *,
    workload_id: str,
    output_dir: Path,
    timeout_seconds: int = 900,
    expected_device_memory_mib: int | None = None,
    reserve_free_memory_mib: int = 1024,
    device_index: int = 0,
    nvidia_smi: str = "nvidia-smi",
    skip_gpu_preflight: bool = False,
    environment: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> dict[str, Any]:
    """Execute one command and write a bounded run document."""

    if not command:
        raise ValueError("workload command must not be empty")
    if not workload_id or any(ch.isspace() for ch in workload_id):
        raise ValueError("workload_id must be a nonempty token")
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= MAX_TIMEOUT_SECONDS:
        raise ValueError(f"timeout_seconds must be between 1 and {MAX_TIMEOUT_SECONDS}")
    if expected_device_memory_mib is not None and expected_device_memory_mib < 0:
        raise ValueError("expected_device_memory_mib must be nonnegative")
    if reserve_free_memory_mib < 0:
        raise ValueError("reserve_free_memory_mib must be nonnegative")

    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "stdout.txt"
    stderr_path = output_dir / "stderr.txt"
    command_path = output_dir / "command.txt"
    run_path = output_dir / "run.json"
    command_path.write_text(" ".join(_shell_quote(part) for part in command) + "\n", encoding="utf-8")

    started_at = _now()
    gpu_before: dict[str, Any] | None = None
    gpu_error: str | None = None
    if not skip_gpu_preflight:
        gpu_before, gpu_error = query_gpu(nvidia_smi=nvidia_smi, device_index=device_index)
        if gpu_before is None:
            result = _document(
                workload_id=workload_id,
                command=command,
                cwd=cwd,
                started_at=started_at,
                finished_at=_now(),
                status="host_query_error",
                returncode=None,
                timed_out=False,
                cleanup_ok=True,
                elapsed_seconds=0.0,
                gpu_before=None,
                gpu_after=None,
                gpu_error=gpu_error,
                expected_device_memory_mib=expected_device_memory_mib,
                reserve_free_memory_mib=reserve_free_memory_mib,
                failure_reason=gpu_error,
            )
            run_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return result
        required = (expected_device_memory_mib or 0) + reserve_free_memory_mib
        if gpu_before["free_mib"] < required:
            reason = f"free_memory_{gpu_before['free_mib']}MiB_below_required_{required}MiB"
            result = _document(
                workload_id=workload_id,
                command=command,
                cwd=cwd,
                started_at=started_at,
                finished_at=_now(),
                status="preflight_memory_low",
                returncode=None,
                timed_out=False,
                cleanup_ok=True,
                elapsed_seconds=0.0,
                gpu_before=gpu_before,
                gpu_after=gpu_before,
                gpu_error=None,
                expected_device_memory_mib=expected_device_memory_mib,
                reserve_free_memory_mib=reserve_free_memory_mib,
                failure_reason=reason,
            )
            run_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return result

    child_env = os.environ.copy()
    if environment:
        child_env.update({str(key): str(value) for key, value in environment.items()})
    started_clock = time.monotonic()
    timed_out = False
    cleanup_ok = True
    returncode: int | None = None
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            process = subprocess.Popen(
                list(command),
                cwd=str(cwd) if cwd else None,
                env=child_env,
                stdout=stdout,
                stderr=stderr,
                start_new_session=(os.name != "nt"),
                creationflags=(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0),
                text=True,
            )
            try:
                returncode = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                cleanup_ok = _terminate_group(process)
                returncode = process.returncode
    except OSError as exc:
        finished_at = _now()
        result = _document(
            workload_id=workload_id,
            command=command,
            cwd=cwd,
            started_at=started_at,
            finished_at=finished_at,
            status="launcher_error",
            returncode=None,
            timed_out=False,
            cleanup_ok=cleanup_ok,
            elapsed_seconds=time.monotonic() - started_clock,
            gpu_before=gpu_before,
            gpu_after=None,
            gpu_error=gpu_error,
            expected_device_memory_mib=expected_device_memory_mib,
            reserve_free_memory_mib=reserve_free_memory_mib,
            failure_reason=str(exc),
        )
        run_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result

    gpu_after, after_error = (
        query_gpu(nvidia_smi=nvidia_smi, device_index=device_index)
        if not skip_gpu_preflight
        else (None, None)
    )
    status, reason = _classify_failure(
        returncode=returncode,
        timed_out=timed_out,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    result = _document(
        workload_id=workload_id,
        command=command,
        cwd=cwd,
        started_at=started_at,
        finished_at=_now(),
        status=status,
        returncode=returncode,
        timed_out=timed_out,
        cleanup_ok=cleanup_ok,
        elapsed_seconds=time.monotonic() - started_clock,
        gpu_before=gpu_before,
        gpu_after=gpu_after,
        gpu_error=after_error or gpu_error,
        expected_device_memory_mib=expected_device_memory_mib,
        reserve_free_memory_mib=reserve_free_memory_mib,
        failure_reason=reason,
    )
    run_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _shell_quote(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:=+,-]+", value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


def _document(
    *,
    workload_id: str,
    command: Sequence[str],
    cwd: Path | None,
    started_at: str,
    finished_at: str,
    status: str,
    returncode: int | None,
    timed_out: bool,
    cleanup_ok: bool,
    elapsed_seconds: float,
    gpu_before: dict[str, Any] | None,
    gpu_after: dict[str, Any] | None,
    gpu_error: str | None,
    expected_device_memory_mib: int | None,
    reserve_free_memory_mib: int,
    failure_reason: str | None,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "workload_id": workload_id,
        "command": list(command),
        "cwd": str(cwd) if cwd else None,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "status": status,
        "returncode": returncode,
        "timed_out": timed_out,
        "process_group_cleanup": cleanup_ok,
        "elapsed_seconds": elapsed_seconds,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "gpu_query_error": gpu_error,
        "expected_device_memory_mib": expected_device_memory_mib,
        "reserve_free_memory_mib": reserve_free_memory_mib,
        "failure_reason": failure_reason,
        "stdout_path": "stdout.txt",
        "stderr_path": "stderr.txt",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--expected-device-memory-mib", type=int)
    parser.add_argument("--reserve-free-memory-mib", type=int, default=1024)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--skip-gpu-preflight", action="store_true")
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--set-env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a workload command is required after --")
    environment: dict[str, str] = {}
    for assignment in args.set_env:
        if "=" not in assignment or not assignment.split("=", 1)[0]:
            parser.error(f"--set-env requires KEY=VALUE: {assignment!r}")
        key, value = assignment.split("=", 1)
        environment[key] = value
    try:
        result = run_workload(
            command,
            workload_id=args.workload_id,
            output_dir=args.output_dir,
            timeout_seconds=args.timeout_seconds,
            expected_device_memory_mib=args.expected_device_memory_mib,
            reserve_free_memory_mib=args.reserve_free_memory_mib,
            device_index=args.device_index,
            nvidia_smi=args.nvidia_smi,
            skip_gpu_preflight=args.skip_gpu_preflight,
            environment=environment,
            cwd=args.cwd,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({"status": result["status"], "run": str(args.output_dir / "run.json")}, sort_keys=True))
    return STATUS_EXIT_CODES.get(str(result["status"]), 1)


if __name__ == "__main__":
    raise SystemExit(main())
