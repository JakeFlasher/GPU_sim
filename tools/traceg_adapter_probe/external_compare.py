#!/usr/bin/env python3
"""Run an independent pinned Accel-Sim trace-parser capability probe.

This is intentionally separate from ``adapter.py``'s Python parser. It builds
the small C++ reader in a temporary directory against an externally supplied
Accel-Sim/GPGPU-Sim checkout and compares its canonical records with the
adapter result. An optional Accel-Sim executable run is reported as an E10
model-relative compatibility datum, never as an NCU or silicon score; the
published aggregate lower-bound lane is maintained in the separate manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence


REPO = Path(__file__).resolve().parents[2]
READER_SOURCE = Path(__file__).with_name("external_traceg_reader.cc")
MAX_INPUT_BYTES = 64 * 1024 * 1024


class ExternalRefusal(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(detail)


def _read_bounded(path: Path) -> bytes:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ExternalRefusal("input_missing", f"cannot read {path}: {exc}") from exc
    if len(data) > MAX_INPUT_BYTES:
        raise ExternalRefusal("input_limit_exceeded", f"{path} exceeds {MAX_INPUT_BYTES} bytes")
    return data


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_identity(path: Path, *, bound: bool = False) -> dict[str, Any]:
    data = _read_bounded(path)
    record = {"path": str(path), "size_bytes": len(data)}
    if bound:
        record["sha256"] = _sha256(data)
    return record


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(_read_bounded(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExternalRefusal("json_decode_failed", f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExternalRefusal("json_shape_invalid", f"expected object in {path}")
    return value


def _paths(parser_root: Path, parser_library: Path | None) -> tuple[Path, Path, Path]:
    root = parser_root.resolve()
    if (root / "trace_parser.h").is_file():
        # Directly supplied gpu-simulator/trace-parser directory.
        header = root / "trace_parser.h"
        include_root = root
        build_root = root.parent
    else:
        # Supplied gpu-simulator directory.
        header = root / "trace-parser" / "trace_parser.h"
        include_root = root / "trace-parser"
        build_root = root
    if not header.is_file():
        raise ExternalRefusal("external_parser_missing", f"missing external header {header}")
    if parser_library is None:
        candidates = [
            build_root / "build" / "trace-parser" / "libtrace-parser.a",
            build_root / "build-local-cuda128" / "trace-parser" / "libtrace-parser.a",
            build_root / "build" / "libtrace-parser.a",
        ]
        parser_library = next((candidate for candidate in candidates if candidate.is_file()), None)
    if parser_library is None or not parser_library.is_file():
        raise ExternalRefusal("external_parser_library_missing", "pass --parser-library pointing to libtrace-parser.a")
    return include_root, header, parser_library.resolve()


def _build_reader(parser_root: Path, parser_library: Path | None, cxx: str) -> Path:
    include_root, _header, library = _paths(parser_root, parser_library)
    cuda_include = Path(os.environ.get("CUDA_HOME", "/opt/cuda")) / "targets/x86_64-linux/include"
    if not (cuda_include / "cuda_runtime.h").is_file():
        # The local installation can expose headers through /usr/local/cuda.
        fallback = Path("/usr/local/cuda") / "targets/x86_64-linux/include"
        if (fallback / "cuda_runtime.h").is_file():
            cuda_include = fallback
        else:
            raise ExternalRefusal("cuda_headers_missing", f"cuda_runtime.h not found under {cuda_include}")
    with tempfile.TemporaryDirectory(prefix="traceg_external_reader_") as directory:
        binary = Path(directory) / "external_traceg_reader"
        command = [
            cxx,
            "-std=c++17",
            "-O2",
            f"-I{cuda_include}",
            f"-I{include_root}",
            str(READER_SOURCE),
            str(library),
            "-lzstd",
            "-o",
            str(binary),
        ]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ExternalRefusal("external_reader_build_failed", str(exc)) from exc
        if completed.returncode != 0:
            raise ExternalRefusal(
                "external_reader_build_failed",
                f"returncode={completed.returncode}: {completed.stderr[-4000:]}",
            )
        # The temporary directory is removed when this function returns, so
        # copy the executable to a second temporary path owned by the caller.
        fd, retained_name = tempfile.mkstemp(prefix="traceg_external_reader_bin_")
        os.close(fd)
        retained = Path(retained_name)
        retained.unlink()
        retained.write_bytes(binary.read_bytes())
        retained.chmod(0o700)
    return retained


def _run_reader(binary: Path, traceg: Path, timeout: float) -> tuple[dict[str, Any], str]:
    try:
        completed = subprocess.run(
            [str(binary), str(traceg)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise ExternalRefusal("external_reader_timeout", f"reader exceeded {timeout}s") from exc
    except OSError as exc:
        raise ExternalRefusal("external_reader_exec_failed", str(exc)) from exc
    if completed.returncode != 0:
        raise ExternalRefusal(
            "external_parser_refused",
            f"returncode={completed.returncode}: {completed.stderr[-4000:]}",
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ExternalRefusal("external_reader_output_invalid", str(exc)) from exc
    if not isinstance(result, dict) or result.get("status") != "ok":
        raise ExternalRefusal("external_reader_output_invalid", "reader did not emit status=ok")
    return result, completed.stderr


def _run_accel_sim(
    accel_sim: Path,
    kernelslist: Path,
    config: Path,
    timeout: float,
    extra_args: Sequence[str],
    *,
    bound: bool = False,
) -> dict[str, Any]:
    command = [str(accel_sim), "-config", str(config), "-trace", str(kernelslist), *extra_args]
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "refused",
            "reason": "accel_sim_timeout",
            "timeout_seconds": timeout,
            "command": command,
        }
    except OSError as exc:
        return {"status": "refused", "reason": "accel_sim_exec_failed", "detail": str(exc), "command": command}
    child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    elapsed = time.monotonic() - started
    stdout = completed.stdout
    def metric(name: str) -> int | None:
        import re

        match = re.search(rf"{name}\s*=\s*([0-9]+)", stdout)
        return None if match is None else int(match.group(1))

    cycle = metric("gpu_tot_sim_cycle")
    insn = metric("gpu_tot_sim_insn")
    cta = metric("gpu_tot_issued_cta")
    result = {
        "status": "ok",
        "returncode": completed.returncode,
        "command": command,
        "gpu_tot_sim_cycle": cycle,
        "gpu_tot_sim_insn": insn,
        "gpu_tot_issued_cta": cta,
        "stderr_tail": completed.stderr[-4000:],
        "resource": {
            "wall_seconds": elapsed,
            "child_user_seconds": max(0.0, child_after.ru_utime - child_before.ru_utime),
            "child_system_seconds": max(0.0, child_after.ru_stime - child_before.ru_stime),
            "child_max_rss_kb": child_after.ru_maxrss,
        },
    }
    if bound:
        result.update(
            {
                "stdout_sha256": _sha256(stdout.encode("utf-8")),
                "counter_digest_sha256": _sha256(
                    f"gpu_tot_sim_cycle={cycle};gpu_tot_sim_insn={insn};gpu_tot_issued_cta={cta}".encode("utf-8")
                ),
                "stderr_sha256": _sha256(completed.stderr.encode("utf-8")),
            }
        )
    if completed.returncode != 0:
        result.update({"status": "refused", "reason": "accel_sim_nonzero_exit"})
    elif cycle is None or insn is None or cta is None:
        result.update({"status": "refused", "reason": "accel_sim_counters_missing"})
    elif completed.stderr.strip():
        # A successful process with a missing trace or an ignored parser error
        # can still return zero.  Treat unexpected stderr as a refusal rather
        # than publishing a compatibility datum with unknown provenance.
        result.update({"status": "refused", "reason": "accel_sim_stderr_nonempty"})
    return result


def compare(
    traceg: Path,
    adapter_result: Path,
    parser_root: Path,
    parser_library: Path | None,
    *,
    cxx: str = "g++",
    timeout: float = 30.0,
    accel_sim: Path | None = None,
    config: Path | None = None,
    kernelslist: Path | None = None,
    extra_args: Sequence[str] = (),
    bound: bool = False,
) -> dict[str, Any]:
    traceg_bytes = _read_bounded(traceg)
    adapter = _json(adapter_result)
    expected = adapter.get("canonical_records")
    if not isinstance(expected, list):
        raise ExternalRefusal("adapter_result_shape_invalid", "canonical_records must be a list")
    expected_launch = adapter.get("launch")
    if not isinstance(expected_launch, dict):
        raise ExternalRefusal("adapter_launch_missing", "adapter result has no launch metadata")
    include_root, parser_header, resolved_library = _paths(parser_root, parser_library)
    binary = _build_reader(parser_root, resolved_library, cxx)
    try:
        external, stderr = _run_reader(binary, traceg, timeout)
    finally:
        binary.unlink(missing_ok=True)
    observed = external.get("records")
    terminal = external.get("terminal")
    if not isinstance(terminal, dict) or terminal.get("opcode") != "EXIT":
        raise ExternalRefusal(
            "external_terminal_parse_missing",
            "pinned external parser did not return a canonical EXIT record",
        )
    expected_terminal = adapter.get("canonical_terminal")
    if not isinstance(expected_terminal, dict):
        raise ExternalRefusal(
            "adapter_terminal_missing",
            "adapter result has no canonical_terminal tuple",
        )
    if observed != expected:
        raise ExternalRefusal(
            "canonical_mismatch",
            f"external parser records differ: expected={len(expected)} observed={len(observed or [])}",
        )
    if terminal != expected_terminal:
        raise ExternalRefusal(
            "canonical_terminal_mismatch",
            "external parser terminal tuple differs from the source dynamic EXIT",
        )
    observed_launch = external.get("launch")
    if observed_launch != expected_launch:
        raise ExternalRefusal(
            "launch_header_mismatch",
            "external parser launch/header metadata differs from the source adapter result",
        )
    result: dict[str, Any] = {
        "schema": "sm86_external_traceg_comparison_v1",
        "status": "ok",
        "scoring": False,
        "comparison_role": "executable_parser_representation_smoke",
        "ground_truth_lane": "local_ncu",
        "literature_floor_manifest": str(
            REPO / "docs/accelsim_literature_lower_bound_manifest_20260928.json"
        ),
        "parser": {
            "reader_schema": external.get("schema"),
            "stderr": stderr,
            "record_count": len(observed),
            "terminal": terminal,
            "terminal_equality": "canonical_equal",
            "launch_equality": "canonical_equal",
            "termination_parser": "external_canonical_exit",
        },
        "input": {
            "traceg": str(traceg),
            "adapter_result": str(adapter_result),
        },
        "provenance": {
            "external_parser_header": _file_identity(parser_header, bound=bound),
            "external_parser_library": _file_identity(resolved_library, bound=bound),
            "reader_source": _file_identity(READER_SOURCE, bound=bound),
            "compiler": cxx,
            "include_root": str(include_root),
        },
        "endpoint": "legacy_traceg_representation_only",
        "notes": [
            "independent external parser canonical equality; not NCU or silicon timing",
            "Accel-Sim gpu_tot_sim_cycle is reported only when an optional executable run is requested",
            "external card/config and local gpc__cycles_elapsed.max are separate domains",
        ],
    }
    if accel_sim is not None or config is not None or kernelslist is not None:
        if accel_sim is None or config is None or kernelslist is None:
            raise ExternalRefusal("accel_sim_arguments_incomplete", "accel-sim, config, and kernelslist are required together")
        accel_result = _run_accel_sim(
            accel_sim, kernelslist, config, timeout, extra_args, bound=bound
        )
        if accel_result.get("status") != "ok":
            raise ExternalRefusal(
                "accel_sim_refused",
                json.dumps(accel_result, sort_keys=True),
            )
        result["accel_sim"] = accel_result
        result["provenance"].update(
            {
                "accel_sim_binary": _file_identity(accel_sim, bound=bound),
                "accel_config": _file_identity(config, bound=bound),
                "kernelslist": _file_identity(kernelslist, bound=bound),
            }
        )
    return result


def _cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traceg", type=Path, required=True)
    parser.add_argument("--adapter-result", type=Path, required=True)
    parser.add_argument("--parser-root", type=Path, required=True, help="external gpu-simulator/trace-parser directory")
    parser.add_argument("--parser-library", type=Path)
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--accel-sim", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--kernelslist", type=Path)
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--bound", action="store_true", help="opt into hash-bound provenance")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = compare(
            args.traceg,
            args.adapter_result,
            args.parser_root,
            args.parser_library,
            cxx=args.cxx,
            timeout=args.timeout,
            accel_sim=args.accel_sim,
            config=args.config,
            kernelslist=args.kernelslist,
            extra_args=args.extra_arg,
            bound=args.bound,
        )
    except ExternalRefusal as exc:
        print(json.dumps({"schema": "sm86_external_traceg_comparison_v1", "status": "refused", "code": exc.code, "detail": exc.detail}, sort_keys=True))
        return 2
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
