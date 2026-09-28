#!/usr/bin/env python3
"""Run the bounded PTX subset compiler with an optional artifact record.

The default is the fast development path: it emits only paths and sizes, with
no hashes or verification metadata.  ``--bound`` opts into the historical
content-bound record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


class CompileManifestError(RuntimeError):
    pass


def digest(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {"path": str(path), "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def run(
    compiler: Path,
    architecture: str,
    source: Path,
    output: Path,
    manifest: Path,
    max_steps: int,
    initial_memory: Path | None,
    alpha: bool = True,
) -> dict[str, object]:
    if not alpha:
        for path in (output, manifest):
            if path.exists():
                raise CompileManifestError(f"refusing to overwrite {path}")
    command = [str(compiler), architecture, str(source), str(output), str(max_steps)]
    if initial_memory is not None:
        command.append(str(initial_memory))
    completed = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    if completed.returncode != 0:
        raise CompileManifestError(f"compiler returned {completed.returncode}: {completed.stderr[-4000:]}")
    source_record = (
        {"path": str(source), "size_bytes": source.stat().st_size}
        if alpha else digest(source)
    )
    generated_record = (
        {"path": str(output), "size_bytes": output.stat().st_size}
        if alpha else digest(output)
    )
    memory_record = None if initial_memory is None else (
        {"path": str(initial_memory), "size_bytes": initial_memory.stat().st_size}
        if alpha else digest(initial_memory)
    )
    result: dict[str, object] = {
        "schema": "ptx_subset_compiler_alpha_v1" if alpha else "ptx_subset_compiler_artifact_v1",
        "status": "ok",
        "development_alpha": alpha,
        "contract": "ptx_subset_v2",
        "architecture": architecture,
        "compiler": {"path": str(compiler), "command": command},
        "source": source_record,
        "generated_program": generated_record,
        "initial_memory": memory_record,
        "claims": {
            "bounded_functional_compiler": True,
            "arbitrary_ptx": False,
            "arbitrary_isa": False,
            "cuda_abi": False,
            "native_sass_generation": False,
            "timing_backend": False,
        },
        "unsupported": ["calls", "ABI/linking", "vectors", "fences/memory-model", "arbitrary PTX", "SASS"],
    }
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--initial-memory", type=Path)
    parser.add_argument("--bound", action="store_true", help="opt into hashes and verification metadata")
    args = parser.parse_args()
    try:
        result = run(
            args.compiler,
            args.architecture,
            args.source,
            args.output,
            args.manifest,
            args.max_steps,
            args.initial_memory,
            not args.bound,
        )
    except (OSError, subprocess.SubprocessError, CompileManifestError) as exc:
        print(json.dumps({"schema": "ptx_subset_compiler_artifact_v1", "status": "refused", "detail": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
