#!/usr/bin/env python3
"""Capture the SM86 cohort with an exact ``env -i``-style NVBit environment."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workloads.research_probes.validate_sm86_live_cohort import (
    _EXPECTED_CELL_SPECS,
    _NVBIT_FORBIDDEN_ENVIRONMENT_CONTROLS,
    _expected_nvbit_argv,
    _expected_nvbit_environment,
)
from tools.nvbit_tracer.runtime_env import sanitized_capture_environment


DEFAULT_OUTPUT_ROOT = "results/captures/sm86_simulator_live_baseline_20260829_v2"
DEFAULT_TIMEOUT_SECONDS = 900
MAX_TIMEOUT_SECONDS = 3600
TERMINATION_GRACE_SECONDS = 5
TIMEOUT_POLICY = "process_group_terminate_reap_v1"
TIMEOUT_POLICY_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _terminate(process: subprocess.Popen[str]) -> bool:
    """Terminate the complete capture process group and reap its leader.

    A timed-out ``communicate`` may have observed an exited leader while a
    descendant still owns stdout/stderr.  Always signal the group; polling the
    leader first would leak that descendant and can retain CUDA resources.
    """

    cleanup_ok = True
    if os.name == "nt":
        with suppress(OSError, subprocess.TimeoutExpired):
            taskkill = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=TERMINATION_GRACE_SECONDS,
            )
            if taskkill.returncode not in (0, 128, 255) and process.poll() is None:
                cleanup_ok = False
        with suppress(OSError):
            process.terminate()
        if process.poll() is None:
            cleanup_ok = False
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=TERMINATION_GRACE_SECONDS)
        if process.poll() is None:
            cleanup_ok = False
        with suppress(OSError):
            process.kill()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=TERMINATION_GRACE_SECONDS)
        if process.poll() is None:
            cleanup_ok = False
        return cleanup_ok

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
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        cleanup_ok = False
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=TERMINATION_GRACE_SECONDS)
    if process.poll() is None:
        cleanup_ok = False
    return cleanup_ok


def _text_output(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value if isinstance(value, str) else ""


def _cell(cell_id: str) -> dict[str, object]:
    spec = _EXPECTED_CELL_SPECS.get(cell_id)
    if spec is None:
        raise ValueError(f"unknown SM86 cohort cell: {cell_id}")
    return {
        "id": cell_id,
        "binary": {
            "path": spec["binary"],
            **(
                {"trace_path": spec["trace_binary"]}
                if "trace_binary" in spec
                else {}
            ),
        },
    }


def capture(cell_id: str, output_root: Path, timeout_seconds: int) -> int:
    cell = _cell(cell_id)
    relative_output = f"evidence/raw/nvbit/{cell_id}"
    destination = output_root / "evidence" / "raw" / "nvbit" / cell_id
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(
            f"{cell_id}: destination is nonempty; move the prior capture aside before recapturing"
        )
    destination.mkdir(parents=True, exist_ok=True)
    argv = _expected_nvbit_argv(cell)
    # Call the shared sanitizer first so loader/tracer controls from the
    # parent can never flow into the child; the exact contract below then
    # restricts the sanitized result to the small reproducible allowlist.
    sanitized_capture_environment(dict(os.environ))
    environment = _expected_nvbit_environment(cell, relative_output)
    environment["TRACES_FOLDER"] = str(destination.resolve())
    started = _now()
    timed_out = False
    cleanup_complete = True
    try:
        popen_kwargs: dict[str, object] = {
            "cwd": REPO_ROOT,
            "env": environment,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
        else:
            popen_kwargs["start_new_session"] = True
        process = subprocess.Popen(argv, **popen_kwargs)
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            cleanup_complete = _terminate(process)
            try:
                stdout, stderr = process.communicate(timeout=TERMINATION_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                cleanup_complete = _terminate(process) and cleanup_complete
                stdout = _text_output(exc.output)
                stderr = _text_output(exc.stderr)
                if process.stdout is not None:
                    process.stdout.close()
                if process.stderr is not None:
                    process.stderr.close()
        stdout = _text_output(stdout)
        stderr = _text_output(stderr)
        return_code = process.returncode if process.returncode is not None else -signal.SIGTERM
    except OSError as exc:
        stdout = ""
        stderr = str(exc)
        return_code = 127
    (destination / "stdout.txt").write_text(stdout, encoding="utf-8")
    (destination / "stderr.txt").write_text(stderr, encoding="utf-8")
    (destination / "command.txt").write_text(shlex.join(argv) + "\n", encoding="utf-8")
    environment_record = {
        "schema_version": 1,
        "kind": "sanitized_nvbit_capture_environment",
        "cell": cell_id,
        "argv": argv,
        "output": relative_output,
        "launch_mode": "env_i_exact_allowlist",
        "environment": environment,
        "effective_environment": environment,
        "forbidden_variables_absent": sorted(_NVBIT_FORBIDDEN_ENVIRONMENT_CONTROLS),
        "return_code": return_code,
        "timed_out": timed_out,
        "cleanup_complete": cleanup_complete,
        "timeout_seconds": timeout_seconds,
        "timeout_policy": TIMEOUT_POLICY,
        "timeout_policy_version": TIMEOUT_POLICY_VERSION,
        "started_at": started,
        "finished_at": _now(),
    }
    (destination / "environment.json").write_text(
        json.dumps(environment_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    trace = destination / "dynamic_trace.pb"
    if timed_out:
        raise RuntimeError(f"{cell_id}: NVBit capture timed out after {timeout_seconds}s")
    if return_code != 0:
        raise RuntimeError(f"{cell_id}: NVBit capture exited {return_code}; see {destination / 'stderr.txt'}")
    if not trace.is_file() or trace.stat().st_size == 0:
        raise RuntimeError(f"{cell_id}: NVBit capture produced no dynamic trace")
    print(f"captured {cell_id}: {trace.stat().st_size} dynamic-trace bytes")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", action="append", dest="cells")
    parser.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)
    if type(args.timeout) is not int or not 1 <= args.timeout <= MAX_TIMEOUT_SECONDS:
        parser.error(f"--timeout must be between 1 and {MAX_TIMEOUT_SECONDS} seconds")
    cells = args.cells or sorted(_EXPECTED_CELL_SPECS)
    failures: list[str] = []
    for cell_id in cells:
        try:
            capture(cell_id, args.output_root.expanduser().resolve(), args.timeout)
        except (OSError, RuntimeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            failures.append(cell_id)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
