#!/usr/bin/env python3
"""Capture one LDGSTS burst cell with the retained NVBit runtime."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.local_labels.immutable_artifact import (  # noqa: E402
    PinnedDirectoryReader,
    open_retained_executable,
    open_retained_regular,
)
from tools.nvbit_tracer import capture_process  # noqa: E402
from tools.nvbit_tracer.runtime_env import (  # noqa: E402
    default_runtime_manifest_path,
    retained_tracer_runtime_env,
)


TRACER = REPO_ROOT / "tools/nvbit_tracer/tracer_tool/tracer_tool.so"
BINARY = HERE / "ldgsts_burst_release"
RUNTIME_MANIFEST = default_runtime_manifest_path(TRACER)
PB_LIB = REPO_ROOT / ".toolchain/pb/lib"
TIMEOUT_SECONDS = 900
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
REQUIRED_OUTPUTS = (
    "dynamic_trace.pb",
    "stats.csv",
    "extra_info/traced_kernel_modules.json",
    "extra_info/enhanced_execution_info.json",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def closure_hash(closure: Mapping[str, str]) -> str:
    payload = json.dumps(sorted(closure.items()), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def output_contract(destination: Path, *, require_enhanced: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    required_outputs = (
        REQUIRED_OUTPUTS
        if require_enhanced
        else ("dynamic_trace.pb", "stats.csv")
    )
    for relative in required_outputs:
        path = destination / relative
        if not path.is_file() or path.stat().st_size <= 0:
            errors.append(f"missing or empty {relative}")
    payloads = sorted(
        path
        for path in (destination / "threadblocks").rglob("*.pb")
        if path.is_file() and path.stat().st_size > 0
    ) if (destination / "threadblocks").is_dir() else []
    if not payloads:
        errors.append("no non-empty threadblocks/*.pb payload")
    if not errors:
        try:
            with (destination / "stats.csv").open(encoding="utf-8", newline="") as handle:
                rows = [
                    {
                        str(key).strip(): str(value).strip()
                        for key, value in row.items()
                    }
                    for row in csv.DictReader(handle)
                    if row.get(" kernel mangled name", row.get("kernel mangled name"))
                ]
            if len(rows) != 1:
                errors.append(f"stats.csv kernel rows={len(rows)}, expected 1")
            elif rows[0].get("total_insts") != rows[0].get("total_reported_insts"):
                errors.append("stats.csv total_insts differs from total_reported_insts")
            metadata_requirements = (
                (
                    ("extra_info/traced_kernel_modules.json", "launches"),
                    ("extra_info/enhanced_execution_info.json", "kernels"),
                )
                if require_enhanced
                else ()
            )
            for relative, field in metadata_requirements:
                value = json.loads((destination / relative).read_text(encoding="utf-8"))
                if not isinstance(value, Mapping) or not isinstance(value.get(field), list) or len(value[field]) != 1:
                    errors.append(f"{relative}: expected one {field} entry")
        except (OSError, ValueError, json.JSONDecodeError, csv.Error) as error:
            errors.append(f"auxiliary output schema error: {error}")
    return {
        "version": 1,
        "required_outputs": list(required_outputs),
        "payload_file_count": len(payloads),
        "passed": not errors,
        "error": "; ".join(errors) if errors else None,
    }


def capture(
    destination: Path,
    *,
    cell: str,
    binary: Path = BINARY,
    target_args: Sequence[str] | None = None,
    gpu: str,
    bursts: int,
    warmup: int,
    repetition: int,
    order: str,
    timeout: int,
    skip_enhanced: bool,
    async_order_enabled: bool,
    mode: str,
) -> int:
    binary = binary.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"target binary is not executable: {binary}")
    if not TRACER.is_file() or not RUNTIME_MANIFEST.is_file():
        raise RuntimeError("retained tracer or runtime manifest is missing")
    destination = destination.resolve()
    if destination.exists():
        if any(destination.iterdir()):
            raise RuntimeError(f"capture destination is nonempty: {destination}")
    else:
        destination.mkdir(parents=True, exist_ok=False)
    argv = (
        [str(binary), *target_args]
        if target_args is not None
        else [
            str(binary),
            "--mode", mode,
            "--assay", "ldgsts",
            "--cells", cell,
            "--bursts", str(bursts),
            "--warmup", str(warmup),
            "--repetition", str(repetition),
            "--order", order,
        ]
    )
    logical_environment = {
        "CUDA_VISIBLE_DEVICES": gpu,
        "CUDA_INJECTION64_PATH": str(TRACER),
        "LD_LIBRARY_PATH": ".toolchain/pb/lib",
        "USER_DEFINED_FOLDERS": "1",
        "TRACES_FOLDER": str(destination),
        "ACTIVE_FROM_START": "0" if mode == "ncu" else "1",
        "SKIP_ENHANCED_TRACER": "1" if skip_enhanced else "0",
        # The target initializes a large source buffer in kernel 1.  Admit
        # only the assay's run_cells kernel (kernel 2) so initialization does
        # not dominate the retained trace or hide the measured launch.
        "DYNAMIC_KERNEL_LIMIT_START": "1",
        "DYNAMIC_KERNEL_LIMIT_END": "2",
        "ENHANCED_TRACER_MAX_PARSER_INPUT_KB": "256",
        "ENHANCED_TRACER_PER_CUBIN_TIMEOUT_S": "45",
    }
    if async_order_enabled:
        logical_environment["SIMBENCH_NVBIT_ASYNC_ORDER"] = "1"
    started = now()
    result: capture_process.NvbitCaptureProcessResult | None = None
    failure: BaseException | None = None
    runtime_manifest_sha256: str | None = None
    runtime_dso_closure: dict[str, str] = {}
    effective_environment: dict[str, str] = dict(logical_environment)
    binary_sha256: str | None = None
    tracer_sha256: str | None = None
    try:
        with (
            open_retained_executable(binary) as retained_binary,
            open_retained_regular(TRACER) as tracer,
            open_retained_regular(RUNTIME_MANIFEST) as manifest,
            PinnedDirectoryReader(destination) as trace_root,
        ):
            binary_sha256 = retained_binary.sha256
            tracer_sha256 = tracer.sha256
            runtime_manifest_sha256 = sha256_file(RUNTIME_MANIFEST)
            with retained_tracer_runtime_env(
                TRACER,
                base_env={
                    "HOME": "/home/jakeshea",
                    "LANG": "C",
                    "LC_ALL": "C",
                    "PATH": "/usr/local/cuda-13.2/bin:/opt/cuda/bin:/usr/bin:/bin",
                },
                pb_lib=PB_LIB,
                runtime_manifest=RUNTIME_MANIFEST,
                retained_descriptor=tracer.descriptor,
                retained_manifest_descriptor=manifest.descriptor,
            ) as runtime:
                runtime.verify_unchanged()
                runtime_dso_closure = dict(runtime.dso_sha256)
                effective_environment = dict(runtime.environment)
                effective_environment.update(
                    {
                        "CUDA_VISIBLE_DEVICES": gpu,
                        "CUDA_INJECTION64_PATH": tracer.proc_path,
                        "USER_DEFINED_FOLDERS": "1",
                        "TRACES_FOLDER": f"/proc/self/fd/{trace_root.descriptor}",
                        "ACTIVE_FROM_START": "0" if mode == "ncu" else "1",
                        "SKIP_ENHANCED_TRACER": "1" if skip_enhanced else "0",
                        "DYNAMIC_KERNEL_LIMIT_START": "1",
                        "DYNAMIC_KERNEL_LIMIT_END": "2",
                        "ENHANCED_TRACER_MAX_PARSER_INPUT_KB": "256",
                        "ENHANCED_TRACER_PER_CUBIN_TIMEOUT_S": "45",
                    }
                )
                if async_order_enabled:
                    effective_environment["SIMBENCH_NVBIT_ASYNC_ORDER"] = "1"
                try:
                    result = capture_process.execute_capture_process(
                        (retained_binary.proc_path, *argv[1:]),
                        environment=effective_environment,
                        working_directory=str(REPO_ROOT),
                        inherited_descriptors=tuple(
                            dict.fromkeys(
                                (
                                    retained_binary.descriptor,
                                    tracer.descriptor,
                                    manifest.descriptor,
                                    trace_root.descriptor,
                                    *runtime.pass_fds,
                                )
                            )
                        ),
                        timeout_seconds=timeout,
                        max_output_bytes=MAX_OUTPUT_BYTES,
                        timeout_code="ldgsts_nvbit_timeout",
                        launch_code="ldgsts_nvbit_launch_failed",
                    )
                except BaseException as error:
                    failure = error
                runtime.verify_unchanged()
            retained_binary.verify_unchanged()
            tracer.verify_unchanged()
    except BaseException as error:
        if failure is None:
            failure = error

    stdout = result.stdout if result is not None else getattr(failure, "stdout", b"")
    stderr = result.stderr if result is not None else getattr(failure, "stderr", b"")
    (destination / "stdout.txt").write_bytes(stdout)
    (destination / "stderr.txt").write_bytes(stderr)
    (destination / "command.txt").write_text(shlex.join(argv) + "\n", encoding="utf-8")
    return_code = result.returncode if result is not None else None
    timed_out = bool(getattr(failure, "code", "") == "ldgsts_nvbit_timeout")
    cleanup_complete = not bool(getattr(failure, "process_cleanup_incomplete", False))
    environment_record = {
        "schema_version": 1,
        "kind": "ldgsts_nvbit_capture_environment",
        "cell": cell,
        "argv": argv,
        "mode": mode,
        "skip_enhanced_tracer": skip_enhanced,
        "async_order_enabled": async_order_enabled,
        "environment": logical_environment,
        "effective_environment": effective_environment,
        "return_code": return_code,
        "timed_out": timed_out,
        "cleanup_complete": cleanup_complete,
        "timeout_seconds": timeout,
        "runtime_manifest_path": str(RUNTIME_MANIFEST),
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "runtime_dso_closure": runtime_dso_closure,
        "runtime_dso_closure_sha256": closure_hash(runtime_dso_closure),
        "binary_sha256": binary_sha256,
        "tracer_sha256": tracer_sha256,
        "started_at": started,
        "finished_at": now(),
        "error": str(failure) if failure is not None else None,
    }
    write_json(destination / "environment.json", environment_record)
    if failure is not None:
        write_json(destination / "failure.json", {"failed_at": now(), "error": str(failure)})
        raise RuntimeError(f"NVBit capture failed: {failure}")
    if return_code != 0:
        raise RuntimeError(f"NVBit target exited {return_code}")
    contract = output_contract(destination, require_enhanced=not skip_enhanced)
    environment_record["output_contract"] = contract
    write_json(destination / "environment.json", environment_record)
    if not contract["passed"]:
        raise RuntimeError(f"NVBit output contract failed: {contract['error']}")
    binding = {
        "schema_version": 1,
        "cell": cell,
        "argv": argv,
        "gpu": gpu,
        "bursts": bursts,
        "warmup": warmup,
        "repetition": repetition,
        "order": order,
        "mode": mode,
        "skip_enhanced_tracer": skip_enhanced,
        "async_order_enabled": async_order_enabled,
        "binary": {"path": str(binary), "sha256": binary_sha256},
        "tracer": {"path": str(TRACER), "sha256": tracer_sha256},
        "runtime_manifest": {"path": str(RUNTIME_MANIFEST), "sha256": runtime_manifest_sha256},
        "runtime_dso_closure_sha256": closure_hash(runtime_dso_closure),
        "output_contract": contract,
        "paths": {name: name for name in contract["required_outputs"]},
    }
    write_json(destination / "campaign_binding.json", binding)
    required = (
        *tuple(contract["required_outputs"]),
        "environment.json",
        "campaign_binding.json",
        "command.txt",
    )
    write_json(
        destination / "complete.json",
        {
            "complete": True,
            "kind": "ldgsts_nvbit_capture",
            "key": cell,
            "required_files": list(required),
            "completed_at": now(),
            "binary_sha256": binary_sha256,
            "tracer_sha256": tracer_sha256,
        },
    )
    print(f"captured {cell}: {contract['payload_file_count']} threadblock payload(s)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell", default="ldgsts_w1_l1_d4_g0")
    parser.add_argument("--binary", type=Path, default=BINARY)
    parser.add_argument(
        "--target-args",
        nargs=argparse.REMAINDER,
        help="use these complete target arguments instead of the built-in LDGSTS command",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--bursts", type=int, default=16_384)
    parser.add_argument("--warmup", type=int, default=64)
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--mode", choices=("direct", "ncu"), default="direct")
    parser.add_argument("--timeout", type=int, default=TIMEOUT_SECONDS)
    parser.add_argument(
        "--skip-enhanced",
        action="store_true",
        help="retain primary NVBit trace files without the post-execution cubin parser",
    )
    parser.add_argument(
        "--async-order",
        action="store_true",
        help="enable selected LDGSTS/commit/SB0-wait instrumentation ordering",
    )
    args = parser.parse_args(argv)
    try:
        return capture(
            args.output,
            cell=args.cell,
            binary=args.binary,
            target_args=args.target_args,
            gpu=args.gpu,
            bursts=args.bursts,
            warmup=args.warmup,
            repetition=args.repetition,
            order=args.order,
            timeout=args.timeout,
            skip_enhanced=args.skip_enhanced,
            async_order_enabled=args.async_order,
            mode=args.mode,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ldgsts NVBit capture: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
