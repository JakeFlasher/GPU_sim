#!/usr/bin/env python3
"""Build and validate the direct FP32 CUTLASS successor witness.

This artifact is independent of both the frozen cuBLAS row and the BF16
successor witness. It shares only the pinned CUTLASS/toolchain and the
fail-closed source/ELF guard; its source, shape, executable, SASS, control, and
seals are separate.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import tomllib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import jsonschema

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.local_labels import build_cutlass_successor_witness as base  # noqa: E402
from tools.local_labels import verify_cutlass_successor_artifact as guard  # noqa: E402

WORKLOAD_REL = Path("workloads/simbench_bw/l1/vendor_sgemm_cutlass")
WORKLOAD_DIR = ROOT / WORKLOAD_REL
BINARY_REL = WORKLOAD_REL / "vendor_sgemm_cutlass_run"
SHAPE_REL = Path("workloads/simbench_bw/shapes/vendor_sgemm_cutlass.toml")
COMMON_MK_REL = Path("workloads/simbench_bw/common.mk")
BUILDER_REL = Path("tools/local_labels/build_cutlass_sgemm_successor_witness.py")
SCHEMA_REL = Path(
    "workloads/simbench_bw/schemas/cutlass_sgemm_successor_build_witness.schema.json"
)
GUARD_REL = Path("tools/local_labels/verify_cutlass_successor_artifact.py")
BASE_MATRIX_REL = Path("data/local_labels/campaign/sm86_reference_capture_matrix.json")
BASE_AUTHORITY_REL = Path("data/local_labels/campaign/capture_roster_authority_sm86.json")
PROVENANCE_REL = Path("third_party/cutlass.provenance.json")
CUTLASS_REL = Path("third_party/cutlass")
EVIDENCE_REL = Path("data/annex/cutlass_sgemm_successor_sm86_20260723")
OUT_REL = EVIDENCE_REL / "vendor_sgemm_cutlass.build_witness.json"
SASS_NAME = "vendor_sgemm_cutlass.sass"
CONTROL_NAME = "vendor_sgemm_cutlass.control_bits.json"
BUILD_LOG_NAME = "vendor_sgemm_cutlass.build.log"
ARCH = "sm_86"
SKU = "rtx_3090_ti"
SLUG = "vendor_sgemm_cutlass"
BASE_SLUG = "vendor_sgemm_cublas"
WITNESS_ID = "sm86_sgemm_cutlass_successor_build_witness_v1"
ARTIFACT_ID = "cutlass_sgemm_successor_sm86_v1"
NVCC = Path("/usr/local/cuda-13.2/bin/nvcc")
HOST_CXX = Path("/usr/bin/g++-14")
CUOBJDUMP = Path("/usr/local/cuda-13.2/bin/cuobjdump")
PYTHON = ROOT / ".venv/bin/python"
FROZEN_MATRIX_SHA256 = base.FROZEN_MATRIX_SHA256
FROZEN_MATRIX_SEAL = base.FROZEN_MATRIX_SEAL
FROZEN_SOURCE_COMMIT = base.FROZEN_SOURCE_COMMIT
FROZEN_AUTHORITY_SHA256 = base.FROZEN_AUTHORITY_SHA256
FROZEN_BASE_CELL_SHA256 = "328127e1ea186daea64cbc09527c954c5ab6bdb3453e8c38df0fd0122bcd294a"


class WitnessError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    return base.sha256_file(path)


def _canonical_sha(value: Any) -> str:
    return base.canonical_sha256(value)


def _relative(path: Path) -> str:
    return base.relative(path)


def _regular(path: Path, *, label: str, executable: bool = False) -> None:
    base._regular(path, label=label, executable=executable)


def _run(command: list[str], *, cwd: Path = ROOT, timeout: int = 600):
    return base.run_command(command, cwd=cwd, timeout=timeout)


def _git_blob(commit: str, path: str) -> bytes:
    return base.git_blob(commit, path)


def _write_or_match(path: Path, payload: bytes) -> None:
    base._write_or_match(path, payload)


@contextmanager
def _base_artifact_tools() -> Iterator[None]:
    """Temporarily point the shared static extractors at this workload."""
    saved = {
        "CUOBJDUMP": base.CUOBJDUMP,
        "PYTHON": base.PYTHON,
        "NVCC": base.NVCC,
        "SASS_NAME": base.SASS_NAME,
        "CONTROL_NAME": base.CONTROL_NAME,
    }
    base.CUOBJDUMP = CUOBJDUMP
    base.PYTHON = PYTHON
    base.NVCC = NVCC
    base.SASS_NAME = SASS_NAME
    base.CONTROL_NAME = CONTROL_NAME
    try:
        yield
    finally:
        for key, value in saved.items():
            setattr(base, key, value)


def _load_shape() -> tuple[dict[str, Any], list[str], str]:
    try:
        document = tomllib.loads((ROOT / SHAPE_REL).read_text())
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise WitnessError(f"cannot load FP32 shape contract: {exc}") from exc
    if document.get("schema_version") != 1 or document.get("slug") != SLUG:
        raise WitnessError("FP32 shape identity is invalid")
    shapes = [
        value for value in document.get("trace_shapes", [])
        if isinstance(value, dict) and value.get("name") == "trace_00"
    ]
    if len(shapes) != 1 or shapes[0].get("axes") != {"m": 32, "n": 32, "k": 32}:
        raise WitnessError("FP32 trace_00 shape is invalid")
    shape = shapes[0]
    args = [
        "--m", "32", "--n", "32", "--k", "32", "--seed", "42",
        "--shape-point", "trace_00",
    ]
    return shape, args, _canonical_sha(shape)


def _load_base() -> dict[str, Any]:
    matrix = ROOT / BASE_MATRIX_REL
    authority = ROOT / BASE_AUTHORITY_REL
    _regular(matrix, label="frozen matrix")
    _regular(authority, label="frozen authority")
    matrix_bytes = matrix.read_bytes()
    authority_bytes = authority.read_bytes()
    if base.sha256_bytes(matrix_bytes) != FROZEN_MATRIX_SHA256:
        raise WitnessError("frozen matrix file hash changed")
    if base.sha256_bytes(authority_bytes) != FROZEN_AUTHORITY_SHA256:
        raise WitnessError("frozen authority hash changed")
    document = json.loads(matrix_bytes)
    unsealed = {key: value for key, value in document.items() if key != "matrix_seal_sha256"}
    if document.get("matrix_seal_sha256") != FROZEN_MATRIX_SEAL or base.frozen_semantic_sha256(unsealed) != FROZEN_MATRIX_SEAL:
        raise WitnessError("frozen matrix semantic seal changed")
    cells = [
        cell for cell in document.get("cells", [])
        if isinstance(cell, dict) and (cell.get("identity") or {}).get("slug") == BASE_SLUG
    ]
    if len(cells) != 1 or cells[0].get("disposition") != "runnable" or cells[0].get("authority_cell_sha256") != FROZEN_BASE_CELL_SHA256:
        raise WitnessError("frozen FP32 cuBLAS base cell changed")
    return {
        "matrix_path": BASE_MATRIX_REL.as_posix(),
        "matrix_file_sha256": base.sha256_bytes(matrix_bytes),
        "matrix_source_commit": document.get("source_commit"),
        "matrix_seal_sha256": document.get("matrix_seal_sha256"),
        "authority_path": BASE_AUTHORITY_REL.as_posix(),
        "authority_sha256": base.sha256_bytes(authority_bytes),
        "base_slug": BASE_SLUG,
        "base_authority_cell_sha256": cells[0]["authority_cell_sha256"],
        "base_disposition": "runnable",
        "relation": "evaluation_successor_no_relabel",
        "claim_policy": "none_additive_supplement_only",
    }


def _source_contract(commit: str) -> dict[str, Any]:
    guarded = guard.verify_source_inputs(
        [
            ROOT / COMMON_MK_REL, ROOT / SHAPE_REL,
            WORKLOAD_DIR / "Makefile", WORKLOAD_DIR / "vendor_sgemm_cutlass.cu",
            WORKLOAD_DIR / "main.cu",
        ],
        repo_root=ROOT,
    )
    records = dict(guarded["files"])
    for contract_rel in (BUILDER_REL, GUARD_REL, SCHEMA_REL):
        records[contract_rel.as_posix()] = _sha256_file(ROOT / contract_rel)
    records = dict(sorted(records.items()))
    for path, digest in records.items():
        if base.sha256_bytes(_git_blob(commit, path)) != digest:
            raise WitnessError(f"source input differs from producer commit: {path}")
    return {
        "files": records,
        "input_set_sha256": _canonical_sha(records),
        "policy": "source_and_makefile_no_cublas_references",
    }


def _build_command() -> list[str]:
    return [
        "make", "-B", "-C", WORKLOAD_REL.as_posix(), "run-bin",
        f"NVCC={NVCC}", f"NVCC_CCBIN={HOST_CXX}",
        f"CUOBJDUMP={CUOBJDUMP}",
        f"CUTLASS_DIR={(ROOT / CUTLASS_REL).resolve()}",
    ]


def _cubin(binary: Path) -> dict[str, Any]:
    with _base_artifact_tools():
        return base.cubin_inventory(binary)


def _sass(binary: Path, evidence: Path) -> dict[str, Any]:
    with _base_artifact_tools():
        return base.record_sass(binary, evidence)


def _control(binary: Path, evidence: Path) -> dict[str, Any]:
    with _base_artifact_tools():
        return base.record_control_bits(binary, evidence)


def _smoke(binary: Path, shape: dict[str, Any], args: list[str]) -> dict[str, Any]:
    command = [f"./{binary.name}", *args]
    rc, stdout, stderr = _run(command, cwd=WORKLOAD_DIR, timeout=180)
    if rc != 0 or stderr:
        raise WitnessError(f"FP32 smoke failed rc={rc}: {stderr[-300:]!r}")
    try:
        output = json.loads(stdout)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise WitnessError(f"FP32 smoke output is not JSON: {exc}") from exc
    axes = shape["axes"]
    tensor = output.get("output")
    dimensions_ok = (
        isinstance(tensor, list) and len(tensor) == axes["m"]
        and all(isinstance(row, list) and len(row) == axes["n"] for row in tensor)
        and all(math.isfinite(float(value)) for row in tensor for value in row)
    )
    if not (
        output.get("schema_version") == 1
        and output.get("slug") == SLUG
        and output.get("shape_set") == "successor_shapes"
        and output.get("shape_role") == "trace"
        and output.get("shape_point") == "trace_00"
        and output.get("shape_axes") == axes
        and dimensions_ok
    ):
        raise WitnessError("FP32 smoke shape contract failed")
    return {
        "command": command,
        "returncode": rc,
        "output_sha256": base.sha256_bytes(stdout),
        "output_bytes": len(stdout),
        "reported_slug": output["slug"],
        "reported_shape_set": output["shape_set"],
        "reported_shape_role": output["shape_role"],
        "reported_shape_point": output["shape_point"],
        "reported_shape_axes": output["shape_axes"],
        "contract_ok": True,
    }


def build_witness(*, evidence_dir: Path, source_commit: str) -> dict[str, Any]:
    evidence_dir = evidence_dir.resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise WitnessError("invalid FP32 producer commit")
    base_pin = _load_base()
    shape, args, shape_digest = _load_shape()
    source = _source_contract(source_commit)
    provenance, provenance_sha = base.load_cutlass_provenance()
    tree_hash = base.live_cutlass_tree_hash(ROOT / CUTLASS_REL)
    for tool in (NVCC, HOST_CXX, CUOBJDUMP):
        _regular(tool, label="pinned tool")
    command = _build_command()
    try:
        result = subprocess.run(
            command, cwd=str(ROOT), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=600, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WitnessError(f"FP32 build failed to execute: {exc}") from exc
    transcript = base._normalize_build_transcript(result.stdout)
    log = evidence_dir / BUILD_LOG_NAME
    _write_or_match(log, transcript)
    if result.returncode != 0:
        raise WitnessError(f"FP32 build failed rc={result.returncode}")
    binary = ROOT / BINARY_REL
    _regular(binary, label="FP32 executable", executable=True)
    no_cublas = guard.verify_cutlass_artifact(
        binary=binary,
        source_inputs=[
            ROOT / COMMON_MK_REL, ROOT / SHAPE_REL, WORKLOAD_DIR / "Makefile",
            WORKLOAD_DIR / "vendor_sgemm_cutlass.cu", WORKLOAD_DIR / "main.cu",
        ],
        repo_root=ROOT,
    )
    with _base_artifact_tools():
        inventory = _cubin(binary)
        sass = _sass(binary, evidence_dir)
        control = _control(binary, evidence_dir)
    smoke = _smoke(binary, shape, args)
    toolchain = {
        "nvcc": base.command_identity(NVCC),
        "host_cxx": base.command_identity(HOST_CXX),
        "cuobjdump": base.command_identity(CUOBJDUMP),
        "python": base.command_identity(PYTHON),
        "readelf": base.command_identity(Path("/usr/bin/readelf")),
        "make": base.command_identity(Path("/usr/bin/make")),
        "cutlass_dir": str((ROOT / CUTLASS_REL).resolve()),
    }
    document: dict[str, Any] = {
        "schema_version": 1,
        "witness_id": WITNESS_ID,
        "artifact_id": ARTIFACT_ID,
        "arch": ARCH,
        "sku": SKU,
        "source_commit": source_commit,
        "generated_at_utc": "2026-07-23T00:00:00Z",
        "base_frozen_v1": base_pin,
        "source_inputs": source,
        "shape_authority": {
            "path": SHAPE_REL.as_posix(),
            "sha256": _sha256_file(ROOT / SHAPE_REL),
            "slug": SLUG,
            "trace_shape": shape,
            "trace_shape_sha256": shape_digest,
            "shape_role": "trace",
            "actual_args": args,
            "shape_contract": "trace_shape_is_actual_direct_executable_launch",
        },
        "cutlass_provenance": {
            "path": PROVENANCE_REL.as_posix(),
            "sha256": provenance_sha,
            "upstream_tag": provenance["upstream_tag"],
            "upstream_commit": provenance["upstream_commit"],
            "upstream_tree": provenance["upstream_tree"],
            "scoped_digest": provenance["digest"]["sha256"],
            "version": provenance["version"],
        },
        "live_cutlass_inputs": {
            "root": CUTLASS_REL.as_posix(),
            "build_subdirectories": ["include", "tools/util/include"],
            "tree_hash": tree_hash,
        },
        "toolchain": toolchain,
        "build": {
            "executed": True,
            "command": command,
            "returncode": 0,
            "transcript_path": _relative(log),
            "transcript_sha256": _sha256_file(log),
            "transcript_bytes": log.stat().st_size,
        },
        "artifacts": {
            "binary": {
                "path": BINARY_REL.as_posix(),
                "sha256": _sha256_file(binary),
                "bytes": binary.stat().st_size,
            },
            "cubin_inventory": inventory,
            "sass": sass,
            "control_bits": control,
            "no_cublas": no_cublas,
        },
        "smoke": smoke,
        "control_contract": {
            "policy": "direct_executable",
            "execution_arch": ARCH,
            "code_arch": ARCH,
            "function_count": control["function_count"],
            "all_records_decoded": control["unknown_encoding"] == 0,
            "hardware_capture_status": "not_run_build_witness_only",
        },
        "integrity": {
            "row_count": 1,
            "unique_slugs": True,
            "source_commit_inputs_match": True,
            "frozen_base_unchanged": True,
            "all_checks_pass": True,
        },
        "pass": True,
    }
    document["witness_seal_sha256"] = _canonical_sha(document)
    return document


def validate_witness(document: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        schema = json.loads((ROOT / SCHEMA_REL).read_text())
        jsonschema.Draft202012Validator(schema).validate(document)
    except (OSError, json.JSONDecodeError, jsonschema.ValidationError, jsonschema.SchemaError) as exc:
        return [f"schema:{exc}"]
    if document.get("witness_seal_sha256") != _canonical_sha({key: value for key, value in document.items() if key != "witness_seal_sha256"}):
        errors.append("witness_seal")
    for path, digest in document["source_inputs"]["files"].items():
        target = ROOT / path
        if not target.is_file() or _sha256_file(target) != digest:
            errors.append(f"source_input:{path}")
    for section in ("binary", "sass", "control_bits"):
        artifact = document["artifacts"][section]
        target = ROOT / artifact["path"]
        # Workload executables are intentionally ignored build products. Their
        # exact hash remains sealed; absence in a fresh checkout is a typed
        # re-materialization requirement, while committed SASS/control bytes
        # must always be present and exact.
        if section == "binary" and not target.exists():
            continue
        if not target.is_file() or _sha256_file(target) != artifact["sha256"]:
            errors.append(f"artifact:{section}")
    return sorted(set(errors))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--evidence-dir", default=str(EVIDENCE_REL))
    parser.add_argument("--out", default=str(OUT_REL))
    args = parser.parse_args(argv)
    try:
        evidence = Path(args.evidence_dir)
        if not evidence.is_absolute():
            evidence = ROOT / evidence
        document = build_witness(evidence_dir=evidence, source_commit=args.source_commit)
        problems = validate_witness(document)
        if problems:
            raise WitnessError("self-validation failed: " + ", ".join(problems))
        output = Path(args.out)
        if not output.is_absolute():
            output = ROOT / output
        _write_or_match(output, (json.dumps(document, indent=2, sort_keys=True) + "\n").encode())
    except (OSError, ValueError, WitnessError, jsonschema.ValidationError) as exc:
        print(f"FP32 CUTLASS witness failed: {exc}", file=sys.stderr)
        return 1
    print(f"FP32 CUTLASS witness: pass=True -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
