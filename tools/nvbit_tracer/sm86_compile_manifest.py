#!/usr/bin/env python3
"""Compile a finite SM86 PTX target and emit an artifact record.

This is an external NVIDIA-toolchain integration lane, not an arbitrary
compiler and not value-dependent regeneration.  ``ptxas`` produces static
SM86 SASS; runtime values can only select dynamic paths/results.  Every input
and generated artifact is hashed only with ``--bound``.  The default emits a
lightweight development record without hashes or verification metadata for the
fast local branch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

MAX_INPUT_BYTES = 16 * 1024 * 1024
MAX_OUTPUT_BYTES = 128 * 1024 * 1024


class CompileRefusal(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(detail)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(path: Path, limit: int) -> bytes:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise CompileRefusal("input_missing", f"cannot read {path}: {exc}") from exc
    if len(data) > limit:
        raise CompileRefusal("input_limit_exceeded", f"{path} exceeds {limit} bytes")
    return data


def _tool_version(tool: str, timeout: float) -> str:
    try:
        result = subprocess.run(
            [tool, "--version"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CompileRefusal("tool_unavailable", f"cannot query {tool}: {exc}") from exc
    if result.returncode != 0:
        raise CompileRefusal("tool_version_failed", f"{tool} returned {result.returncode}: {result.stderr[-2000:]}")
    return (result.stdout or result.stderr).strip()


def _run(command: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise CompileRefusal("tool_timeout", f"command exceeded {timeout}s: {command[0]}") from exc
    except OSError as exc:
        raise CompileRefusal("tool_exec_failed", f"cannot execute {command[0]}: {exc}") from exc


def compile_sm86(
    ptx_path: Path,
    out_dir: Path,
    *,
    ptxas: str = "ptxas",
    nvdisasm: str = "nvdisasm",
    cuobjdump: str = "cuobjdump",
    timeout: float = 30.0,
    alpha: bool = True,
) -> dict[str, Any]:
    if out_dir.exists() and not alpha:
        raise CompileRefusal("output_exists", f"refusing to overwrite {out_dir}")
    source = _read(ptx_path, MAX_INPUT_BYTES)
    if b".target sm_86" not in source and b".target sm_86a" not in source:
        raise CompileRefusal("target_mismatch", "PTX must explicitly declare .target sm_86 or .target sm_86a")
    out_dir.mkdir(parents=True, exist_ok=True)
    cubin = out_dir / "kernel.sm_86.cubin"
    sass = out_dir / "kernel.sm_86.sass.txt"
    manifest_path = out_dir / "sm86_compile_manifest.json"
    if alpha:
        ptxas_version = nvdisasm_version = cuobjdump_version = "unreported-alpha"
    else:
        ptxas_version = _tool_version(ptxas, timeout)
        nvdisasm_version = _tool_version(nvdisasm, timeout)
        cuobjdump_version = _tool_version(cuobjdump, timeout)
    ptxas_command = [ptxas, "-arch=sm_86", "-o", str(cubin), str(ptx_path)]
    compiled = _run(ptxas_command, timeout)
    if compiled.returncode != 0:
        raise CompileRefusal(
            "ptxas_refused",
            f"ptxas returned {compiled.returncode}: {compiled.stderr[-4000:]}",
        )
    cubin_bytes = _read(cubin, MAX_OUTPUT_BYTES)
    disasm_command = [nvdisasm, "--print-code", str(cubin)]
    disassembled = _run(disasm_command, timeout)
    if disassembled.returncode != 0:
        raise CompileRefusal(
            "nvdisasm_refused",
            f"nvdisasm returned {disassembled.returncode}: {disassembled.stderr[-4000:]}",
        )
    sass_bytes = disassembled.stdout.encode("utf-8")
    if len(sass_bytes) > MAX_OUTPUT_BYTES:
        raise CompileRefusal("output_limit_exceeded", "disassembly exceeds output limit")
    sass.write_bytes(sass_bytes)
    cuobjdump_command = [cuobjdump, "--dump-sass", str(cubin)]
    cuobjdumped = _run(cuobjdump_command, timeout)
    if cuobjdumped.returncode != 0:
        raise CompileRefusal(
            "cuobjdump_refused",
            f"cuobjdump returned {cuobjdumped.returncode}: {cuobjdumped.stderr[-4000:]}",
        )
    cuobjdump_bytes = cuobjdumped.stdout.encode("utf-8")
    if len(cuobjdump_bytes) > MAX_OUTPUT_BYTES:
        raise CompileRefusal("output_limit_exceeded", "cuobjdump output exceeds output limit")
    cuobjdump_path = out_dir / "kernel.sm_86.cuobjdump.sass.txt"
    cuobjdump_path.write_bytes(cuobjdump_bytes)
    manifest = {
        "schema": "sm86_static_sass_alpha_v1" if alpha else "sm86_static_sass_artifact_v1",
        "status": "ok",
        "development_alpha": alpha,
        "target": "sm_86",
        "static_sass": True,
        "value_dependent_dynamic_trace": False,
        "arbitrary_isa_compiler": False,
        "source": {"path": str(ptx_path), "size_bytes": len(source)},
        "toolchain": {
            "ptxas": ptxas_version,
            "nvdisasm": nvdisasm_version,
            "cuobjdump": cuobjdump_version,
            "ptxas_command": ptxas_command,
            "nvdisasm_command": disasm_command,
            "cuobjdump_command": cuobjdump_command,
        },
        "artifacts": {
            "cubin": {"path": str(cubin), "size_bytes": len(cubin_bytes)},
            "sass": {"path": str(sass), "size_bytes": len(sass_bytes)},
            "cuobjdump_sass": {"path": str(cuobjdump_path), "size_bytes": len(cuobjdump_bytes)},
        },
        "claims": {
            "static_sm86_sass_generation": True,
            "runtime_value_regeneration": False,
            "general_ptx_compiler": False,
            "arbitrary_isa_generation": False,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ptx", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ptxas", default="ptxas")
    parser.add_argument("--nvdisasm", default="nvdisasm")
    parser.add_argument("--cuobjdump", default="cuobjdump")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--bound",
        action="store_true",
        help="opt into the historical content-bound record",
    )
    args = parser.parse_args(argv)
    try:
        result = compile_sm86(
            args.ptx,
            args.out_dir,
            ptxas=args.ptxas,
            nvdisasm=args.nvdisasm,
            cuobjdump=args.cuobjdump,
            timeout=args.timeout,
            alpha=not args.bound,
        )
    except CompileRefusal as exc:
        print(json.dumps({"schema": "sm86_static_sass_artifact_v1", "status": "refused", "code": exc.code, "detail": exc.detail}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
