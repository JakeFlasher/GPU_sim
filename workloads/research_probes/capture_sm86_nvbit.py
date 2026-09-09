#!/usr/bin/env python3
"""Capture the SM86 cohort with an exact ``env -i``-style NVBit environment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workloads.research_probes.validate_sm86_live_cohort import (
    _EXPECTED_CELL_SPECS,
    _NVBIT_FORBIDDEN_ENVIRONMENT_CONTROLS,
    _expected_nvbit_argv,
    _expected_nvbit_environment,
)
from tools.local_labels.immutable_artifact import open_retained_regular
from tools.nvbit_tracer.runtime_env import (
    TracerRuntimeError,
    default_runtime_manifest_path,
    retained_tracer_runtime_env,
)


DEFAULT_OUTPUT_ROOT = "results/captures/sm86_simulator_live_baseline_20260829_v2"
DEFAULT_TIMEOUT_SECONDS = 900
MAX_TIMEOUT_SECONDS = 3600
TERMINATION_GRACE_SECONDS = 5
TIMEOUT_POLICY = "process_group_terminate_reap_v1"
TIMEOUT_POLICY_VERSION = 1
OUTPUT_CONTRACT_VERSION = 1
REQUIRED_OUTPUTS = (
    "dynamic_trace.pb",
    "stats.csv",
    "extra_info/traced_kernel_modules.json",
    "extra_info/enhanced_execution_info.json",
    "extra_info/unsupported_memory_path.json",
    "extra_info/unmapped_memory_range.json",
    "extra_info/cross_allocation_lane_span.json",
)
TRACER_PATH = REPO_ROOT / "tools/nvbit_tracer/tracer_tool/tracer_tool.so"
PB_LIB_PATH = REPO_ROOT / ".toolchain/pb/lib"
RUNTIME_MANIFEST_PATH = default_runtime_manifest_path(TRACER_PATH)


def _runtime_closure_sha256(closure: Mapping[str, str]) -> str:
    payload = json.dumps(sorted(closure.items()), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _logical_environment(
    cell: Mapping[str, object], relative_output: str, destination: Path
) -> dict[str, str]:
    environment = _expected_nvbit_environment(cell, relative_output, root=REPO_ROOT)
    environment["TRACES_FOLDER"] = str(destination.resolve())
    return environment


def _effective_environment(
    runtime: object,
    retained_tracer: object,
    logical: Mapping[str, str],
    destination: Path,
) -> dict[str, str]:
    """Compose the child environment from authenticated descriptor paths."""

    environment = dict(getattr(runtime, "environment"))
    environment["CUDA_INJECTION64_PATH"] = str(
        getattr(retained_tracer, "proc_path")
    )
    environment["USER_DEFINED_FOLDERS"] = logical["USER_DEFINED_FOLDERS"]
    environment["TRACES_FOLDER"] = str(destination.resolve())
    environment["ACTIVE_FROM_START"] = logical["ACTIVE_FROM_START"]
    environment["SKIP_ENHANCED_TRACER"] = logical["SKIP_ENHANCED_TRACER"]
    if "DYNAMIC_KERNEL_LIMIT_START" in logical:
        environment["DYNAMIC_KERNEL_LIMIT_START"] = logical[
            "DYNAMIC_KERNEL_LIMIT_START"
        ]
        environment["DYNAMIC_KERNEL_LIMIT_END"] = logical["DYNAMIC_KERNEL_LIMIT_END"]
        environment["ENHANCED_TRACER_MAX_PARSER_INPUT_KB"] = logical[
            "ENHANCED_TRACER_MAX_PARSER_INPUT_KB"
        ]
        environment["ENHANCED_TRACER_PER_CUBIN_TIMEOUT_S"] = logical[
            "ENHANCED_TRACER_PER_CUBIN_TIMEOUT_S"
        ]
    return environment


def _output_contract(
    destination: Path, *, allow_multiple_launches: bool = False
) -> dict[str, object]:
    """Check the complete NVBit artifact set before reporting success."""

    missing = [
        relative
        for relative in REQUIRED_OUTPUTS
        if not (destination / relative).is_file()
        or (destination / relative).stat().st_size == 0
    ]
    payloads = sorted(
        path
        for path in (destination / "threadblocks").rglob("*.pb")
        if path.is_file() and path.stat().st_size > 0
    ) if (destination / "threadblocks").is_dir() else []
    errors: list[str] = []
    if missing:
        errors.append("missing or empty outputs: " + ", ".join(missing))
    if not payloads:
        errors.append("no non-empty threadblocks/*.pb payload files")
    if not errors:
        try:
            rows = []
            with (destination / "stats.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                for raw in csv.DictReader(handle):
                    row = {
                        str(key).strip(): str(value).strip()
                        for key, value in raw.items()
                    }
                    if row.get("kernel mangled name"):
                        rows.append(row)
            if not rows:
                errors.append("stats.csv must contain at least one kernel row")
            elif not allow_multiple_launches and len(rows) != 1:
                errors.append("stats.csv must contain exactly one kernel row")
            elif any(
                row.get("total_insts") != row.get("total_reported_insts")
                for row in rows
            ):
                errors.append("stats.csv total_insts differs from total_reported_insts")
            mapping = json.loads(
                (destination / "extra_info/traced_kernel_modules.json").read_text()
            )
            execution = json.loads(
                (destination / "extra_info/enhanced_execution_info.json").read_text()
            )
            selection = mapping.get("selection") if isinstance(mapping, Mapping) else None
            launches = mapping.get("launches") if isinstance(mapping, Mapping) else None
            if not isinstance(launches, list) or not launches:
                errors.append("traced-kernel mapping must contain launch entries")
            elif not allow_multiple_launches and len(launches) != 1:
                errors.append("traced-kernel mapping must contain one launch")
            elif len(launches) != len(rows):
                errors.append(
                    f"traced-kernel launch rows={len(launches)} differ from stats rows={len(rows)}"
                )
            if isinstance(launches, list) and launches:
                launch_ids = []
                accepted_indices = []
                for launch in launches:
                    if not isinstance(launch, Mapping):
                        continue
                    launch_ids.append(launch.get("trace_launch_id"))
                    accepted_indices.append(launch.get("accepted_launch_index"))
                if (
                    len(launch_ids) != len(launches)
                    or any(type(value) is not int for value in launch_ids)
                    or len(set(launch_ids)) != len(launch_ids)
                ):
                    errors.append("traced-kernel launch IDs must be unique integers")
                if accepted_indices != list(range(len(launches))):
                    errors.append(
                        "traced-kernel accepted launch indices must be contiguous"
                    )
            if not isinstance(selection, Mapping) or (
                selection.get("concurrent_limit_violations") != 0
                or selection.get("exact_contract_satisfied") is not True
            ):
                errors.append("traced-kernel selection contract is not satisfied")
            elif not allow_multiple_launches and (
                selection.get("accepted_active_launches") != 1
                or selection.get("active_launch_attempts") != 1
            ):
                errors.append("traced-kernel selection must admit exactly one active launch")
            if isinstance(selection, Mapping) and isinstance(launches, list):
                if selection.get("module_mapping_count") != len(launches):
                    errors.append(
                        "traced-kernel module mapping count differs from launches"
                    )
                try:
                    reported_total = sum(
                        int(row["total_reported_insts"]) for row in rows
                    )
                except (KeyError, TypeError, ValueError):
                    reported_total = None
                if (
                    reported_total is not None
                    and selection.get("accepted_channel_records") != reported_total
                ):
                    errors.append(
                        "traced-kernel channel count differs from stats totals"
                    )
            if not isinstance(execution, Mapping) or not isinstance(
                execution.get("kernels"), list
            ) or not execution["kernels"]:
                errors.append("enhanced execution metadata must contain kernel entries")
            elif not allow_multiple_launches and len(execution["kernels"]) != 1:
                errors.append("enhanced execution metadata must contain one kernel")
            for sidecar in (
                "unsupported_memory_path.json",
                "unmapped_memory_range.json",
                "cross_allocation_lane_span.json",
            ):
                if json.loads((destination / "extra_info" / sidecar).read_text()) != []:
                    errors.append(f"{sidecar} must contain an empty list")
        except (OSError, ValueError, json.JSONDecodeError, csv.Error) as exc:
            errors.append(f"auxiliary output schema error: {exc}")
    return {
        "version": OUTPUT_CONTRACT_VERSION,
        "required_outputs": list(REQUIRED_OUTPUTS),
        "payload_file_count": len(payloads),
        "passed": not errors,
        "error": "; ".join(errors) if errors else None,
    }


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


def capture_argv(
    cell_id: str,
    argv: list[str],
    logical_environment: Mapping[str, str],
    output_root: Path,
    timeout_seconds: int,
    allow_multiple_launches: bool = False,
) -> int:
    """Capture one caller-bound command with the retained SM86 NVBit runtime."""

    relative_output = f"evidence/raw/nvbit/{cell_id}"
    destination = output_root / "evidence" / "raw" / "nvbit" / cell_id
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(
            f"{cell_id}: destination is nonempty; move the prior capture aside before recapturing"
        )
    destination.mkdir(parents=True, exist_ok=True)
    # Keep the durable logical contract separate from the descriptor-backed
    # effective environment.  The latter contains /proc/self/fd paths whose
    # numbers are intentionally ephemeral and must never be treated as a
    # portable artifact identity.
    logical_environment = dict(logical_environment)
    logical_environment["TRACES_FOLDER"] = str(destination.resolve())
    effective_environment = dict(logical_environment)
    started = _now()
    timed_out = False
    cleanup_complete = True
    runtime_manifest_path: str | None = None
    runtime_manifest_sha256: str | None = None
    runtime_dso_closure: dict[str, str] = {}
    runtime_dso_closure_sha256: str | None = None
    runtime_error: str | None = None
    output_contract: dict[str, object] = {
        "version": OUTPUT_CONTRACT_VERSION,
        "required_outputs": list(REQUIRED_OUTPUTS),
        "payload_file_count": 0,
        "passed": False,
        "error": "capture did not reach output validation",
    }
    try:
        with open_retained_regular(TRACER_PATH) as retained_tracer:
            with open_retained_regular(RUNTIME_MANIFEST_PATH) as retained_manifest:
                with retained_tracer_runtime_env(
                    retained_tracer.path,
                    base_env={
                        "HOME": "/home/jakeshea",
                        "LANG": "C",
                        "LC_ALL": "C",
                        "PATH": "/usr/local/cuda-13.2/bin:/opt/cuda/bin:/usr/bin:/bin",
                    },
                    pb_lib=PB_LIB_PATH,
                    runtime_manifest=RUNTIME_MANIFEST_PATH,
                    retained_descriptor=retained_tracer.descriptor,
                    retained_manifest_descriptor=retained_manifest.descriptor,
                ) as retained_runtime:
                    retained_runtime.verify_unchanged()
                    runtime_manifest_path = str(retained_runtime.runtime_manifest_path)
                    runtime_manifest_sha256 = retained_runtime.runtime_manifest_sha256
                    runtime_dso_closure = dict(retained_runtime.dso_sha256)
                    runtime_dso_closure_sha256 = _runtime_closure_sha256(
                        runtime_dso_closure
                    )
                    effective_environment = _effective_environment(
                        retained_runtime,
                        retained_tracer,
                        logical_environment,
                        destination,
                    )
                    popen_kwargs: dict[str, object] = {
                        "cwd": REPO_ROOT,
                        "env": effective_environment,
                        "stdin": subprocess.DEVNULL,
                        "stdout": subprocess.PIPE,
                        "stderr": subprocess.PIPE,
                        "text": True,
                        "errors": "replace",
                    }
                    if os.name == "nt":
                        popen_kwargs["creationflags"] = getattr(
                            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
                        )
                    else:
                        popen_kwargs["start_new_session"] = True
                        popen_kwargs["pass_fds"] = tuple(
                            dict.fromkeys(
                                (
                                    retained_tracer.descriptor,
                                    retained_manifest.descriptor,
                                    *retained_runtime.pass_fds,
                                )
                            )
                        )
                    process = subprocess.Popen(argv, **popen_kwargs)
                    try:
                        stdout, stderr = process.communicate(timeout=timeout_seconds)
                    except subprocess.TimeoutExpired as exc:
                        timed_out = True
                        cleanup_complete = _terminate(process)
                        try:
                            stdout, stderr = process.communicate(
                                timeout=TERMINATION_GRACE_SECONDS
                            )
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
                    return_code = (
                        process.returncode
                        if process.returncode is not None
                        else -signal.SIGTERM
                    )
                    retained_runtime.verify_unchanged()
    except OSError as exc:
        stdout = ""
        stderr = str(exc)
        return_code = 127
        runtime_error = str(exc)
    except TracerRuntimeError as exc:
        stdout = ""
        stderr = f"runtime provenance: {exc}"
        return_code = 126
        runtime_error = f"{exc.reason}: {exc.detail}"
    (destination / "stdout.txt").write_text(stdout, encoding="utf-8")
    (destination / "stderr.txt").write_text(stderr, encoding="utf-8")
    (destination / "command.txt").write_text(shlex.join(argv) + "\n", encoding="utf-8")
    environment_record = {
        "schema_version": 1,
        "kind": "sanitized_nvbit_capture_environment",
        "cell": cell_id,
        "argv": argv,
        "allow_multiple_launches": allow_multiple_launches,
        "output": relative_output,
        "launch_mode": "env_i_exact_allowlist",
        "environment": logical_environment,
        "effective_environment": effective_environment,
        "forbidden_variables_absent": sorted(_NVBIT_FORBIDDEN_ENVIRONMENT_CONTROLS),
        "return_code": return_code,
        "timed_out": timed_out,
        "cleanup_complete": cleanup_complete,
        "timeout_seconds": timeout_seconds,
        "timeout_policy": TIMEOUT_POLICY,
        "timeout_policy_version": TIMEOUT_POLICY_VERSION,
        "runtime_manifest_path": runtime_manifest_path,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "runtime_dso_closure": runtime_dso_closure,
        "runtime_dso_closure_sha256": runtime_dso_closure_sha256,
        "runtime_error": runtime_error,
        "output_contract_version": OUTPUT_CONTRACT_VERSION,
        "required_outputs": list(REQUIRED_OUTPUTS),
        "payload_file_count": 0,
        "output_contract_passed": False,
        "output_contract_error": "capture did not reach output validation",
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
    if runtime_error is not None:
        raise RuntimeError(f"{cell_id}: NVBit runtime provenance failed: {runtime_error}")
    if return_code != 0:
        raise RuntimeError(f"{cell_id}: NVBit capture exited {return_code}; see {destination / 'stderr.txt'}")
    output_contract = _output_contract(
        destination, allow_multiple_launches=allow_multiple_launches
    )
    environment_record.update(
        {
            "payload_file_count": output_contract["payload_file_count"],
            "output_contract_passed": output_contract["passed"],
            "output_contract_error": output_contract["error"],
        }
    )
    (destination / "environment.json").write_text(
        json.dumps(environment_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if output_contract["passed"] is not True:
        raise RuntimeError(
            f"{cell_id}: NVBit output contract failed: {output_contract['error']}"
        )
    print(f"captured {cell_id}: {trace.stat().st_size} dynamic-trace bytes")
    return 0


def capture(
    cell_id: str,
    output_root: Path,
    timeout_seconds: int,
    *,
    allow_multiple_launches: bool = False,
) -> int:
    cell = _cell(cell_id)
    relative_output = f"evidence/raw/nvbit/{cell_id}"
    destination = output_root / "evidence" / "raw" / "nvbit" / cell_id
    return capture_argv(
        cell_id,
        _expected_nvbit_argv(cell),
        _logical_environment(cell, relative_output, destination),
        output_root,
        timeout_seconds,
        allow_multiple_launches=allow_multiple_launches,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", action="append", dest="cells")
    parser.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--allow-multiple-launches",
        action="store_true",
        help="accept a lossless multi-kernel trace instead of enforcing one launch",
    )
    args = parser.parse_args(argv)
    if type(args.timeout) is not int or not 1 <= args.timeout <= MAX_TIMEOUT_SECONDS:
        parser.error(f"--timeout must be between 1 and {MAX_TIMEOUT_SECONDS} seconds")
    cells = args.cells or sorted(_EXPECTED_CELL_SPECS)
    failures: list[str] = []
    for cell_id in cells:
        try:
            capture(
                cell_id,
                args.output_root.expanduser().resolve(),
                args.timeout,
                allow_multiple_launches=args.allow_multiple_launches,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            failures.append(cell_id)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
