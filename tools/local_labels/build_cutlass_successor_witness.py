#!/usr/bin/env python3
"""Build and validate the direct BF16 CUTLASS successor witness.

This is intentionally separate from the frozen 31-row SimBench witness.  It
seals one namespaced, project-owned executable and its static SASS/control
inputs without changing the canonical cuBLAS rows or their matrix authority.
The executable is a build/smoke witness only; hardware labels and NVBit traces
are separate later evidence surfaces.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jsonschema

from tools.local_labels import verify_cutlass_successor_artifact as artifact_guard


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKLOAD_REL = Path("workloads/simbench_bw/l1/vendor_gemm_cutlass")
WORKLOAD_DIR = REPO_ROOT / WORKLOAD_REL
BINARY_REL = WORKLOAD_REL / "vendor_gemm_cutlass_run"
SHAPE_REL = Path("workloads/simbench_bw/shapes/vendor_gemm_cutlass.toml")
COMMON_MK_REL = Path("workloads/simbench_bw/common.mk")
PROVENANCE_REL = Path("third_party/cutlass.provenance.json")
CUTLASS_REL = Path("third_party/cutlass")
BASE_MATRIX_REL = Path("data/local_labels/campaign/sm86_reference_capture_matrix.json")
BASE_AUTHORITY_REL = Path("data/local_labels/campaign/capture_roster_authority_sm86.json")
SCHEMA_REL = Path(
    "workloads/simbench_bw/schemas/cutlass_successor_build_witness.schema.json"
)
DEFAULT_EVIDENCE_REL = Path("data/annex/cutlass_successor_sm86_20260723")
DEFAULT_OUT_REL = DEFAULT_EVIDENCE_REL / "vendor_gemm_cutlass.build_witness.json"
SASS_NAME = "vendor_gemm_cutlass.sass"
CONTROL_NAME = "vendor_gemm_cutlass.control_bits.json"
BUILD_LOG_NAME = "vendor_gemm_cutlass.build.log"
ARCH = "sm_86"
SKU = "rtx_3090_ti"
SLUG = "vendor_gemm_cutlass"
BASE_SLUG = "vendor_gemm_cublas"
NVCC = Path("/usr/local/cuda-13.2/bin/nvcc")
HOST_CXX = Path("/usr/bin/g++-14")
CUOBJDUMP = Path("/usr/local/cuda-13.2/bin/cuobjdump")
PYTHON = REPO_ROOT / ".venv/bin/python"
FROZEN_MATRIX_SHA256 = "af29d72e955cf42e281036190f2d55d72af0af92a632c78c8d159073a072fb0e"
FROZEN_MATRIX_SEAL = "6376da9721c75bb9299e84a9d11bf607bc21f4cbaa30da25b672c86f6757a5e1"
FROZEN_SOURCE_COMMIT = "124ce358733f97ba6e0af808d10ab69ffe4d5a78"
FROZEN_AUTHORITY_SHA256 = "d02c817091b0a9f06c2106c0987c59a481910afcc3b78ab1de7a7d5bd5a6b4b4"
FROZEN_BASE_CELL_SHA256 = "1e84352627f783b1d28fb3684cab96a61fa10cbdbb9c0e0b8844b027c1dc6142"
CUTLASS_PROVENANCE_SHA256 = ""
CUTLASS_SCOPED_DIGEST = "ad131382834231e4076b50c8fdf8f7783504fe83aac53982dab353be9908e113"
CUTLASS_UPSTREAM_COMMIT = "e05f953a5b3d38adc240df2ff928e0421c2abba3"
CUTLASS_UPSTREAM_TREE = "0d3dd31da2e1ccc8ab15047a2edd9ebf5d98218b"


class WitnessError(RuntimeError):
    """A successor witness cannot be admitted."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise WitnessError(f"cannot read {path}: {exc}") from exc
    return sha256_bytes(payload)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(payload)


def frozen_semantic_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(payload)


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise WitnessError(f"path is outside repository: {path}") from exc


def _regular(path: Path, *, label: str, executable: bool = False) -> None:
    if path.is_symlink() or not path.is_file():
        raise WitnessError(f"{label} is not a regular file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise WitnessError(f"{label} is not executable: {path}")


def run_command(
    command: list[str], *, cwd: Path = REPO_ROOT, timeout: int = 600,
) -> tuple[int, bytes, bytes]:
    try:
        result = subprocess.run(
            command, cwd=str(cwd), capture_output=True, timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WitnessError(f"command failed to start or timed out: {command}: {exc}") from exc
    return result.returncode, result.stdout, result.stderr


def command_identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    _regular(path, label="tool")
    rc, stdout, stderr = run_command([str(path), "--version"], timeout=60)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "version_returncode": rc,
        "version_stdout": stdout.decode("utf-8", errors="replace")[:2000],
        "version_stderr": stderr.decode("utf-8", errors="replace")[:2000],
    }


def git_head() -> str:
    rc, stdout, _ = run_command(["git", "rev-parse", "HEAD"], timeout=60)
    value = stdout.decode("ascii", errors="ignore").strip()
    if rc != 0 or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise WitnessError("cannot resolve a 40-character producer source commit")
    return value


def git_blob(commit: str, path: str) -> bytes:
    rc, stdout, stderr = run_command(
        ["git", "cat-file", "blob", f"{commit}:{path}"], timeout=60,
    )
    if rc != 0:
        raise WitnessError(
            f"source input is absent at producer commit {commit}: {path}: "
            f"{stderr.decode('utf-8', errors='replace')[-200:]}"
        )
    return stdout


def _write_or_match(path: Path, payload: bytes) -> None:
    """Preserve an existing evidence file unless its bytes are identical."""
    if path.exists():
        _regular(path, label="existing evidence")
        if path.read_bytes() != payload:
            raise WitnessError(f"existing evidence bytes differ: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def live_cutlass_tree_hash(root: Path) -> str:
    """Match native_env.py's length-delimited build-input identity."""
    digest = hashlib.sha256()
    count = 0
    for subdir in (Path("include"), Path("tools/util/include")):
        target = root / subdir
        if not target.is_dir() or target.is_symlink():
            raise WitnessError(f"CUTLASS build-input directory is unavailable: {target}")
        for path in sorted(target.rglob("*")):
            if path.is_symlink():
                raise WitnessError(f"CUTLASS build input is a symlink: {path}")
            if not path.is_file():
                continue
            rel = (subdir / path.relative_to(target)).as_posix().encode("utf-8")
            payload = path.read_bytes()
            digest.update(len(rel).to_bytes(8, "big"))
            digest.update(rel)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
            digest.update(b"\x00")
            count += 1
    if count == 0:
        raise WitnessError("CUTLASS build-input tree is empty")
    return digest.hexdigest()


def load_shape() -> tuple[dict[str, Any], list[str], str]:
    try:
        document = tomllib.loads((REPO_ROOT / SHAPE_REL).read_text())
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise WitnessError(f"cannot load successor shape contract: {exc}") from exc
    if document.get("schema_version") != 1 or document.get("slug") != SLUG:
        raise WitnessError("successor shape identity is invalid")
    trace_shapes = document.get("trace_shapes")
    if not isinstance(trace_shapes, list):
        raise WitnessError("successor trace_shapes is not a list")
    matches = [item for item in trace_shapes if isinstance(item, dict) and item.get("name") == "trace_00"]
    if len(matches) != 1:
        raise WitnessError("successor must contain exactly one trace_00 shape")
    shape = matches[0]
    axes = shape.get("axes")
    if axes != {"m": 32, "n": 32, "k": 32}:
        raise WitnessError(f"unexpected successor trace axes: {axes!r}")
    args: list[str] = []
    for key in ("m", "n", "k"):
        args.extend([f"--{key}", str(axes[key])])
    args.extend(["--seed", "42", "--shape-point", "trace_00"])
    return shape, args, canonical_sha256(shape)


def load_frozen_base() -> dict[str, Any]:
    matrix_path = REPO_ROOT / BASE_MATRIX_REL
    authority_path = REPO_ROOT / BASE_AUTHORITY_REL
    _regular(matrix_path, label="frozen matrix")
    _regular(authority_path, label="frozen authority")
    matrix_bytes = matrix_path.read_bytes()
    if sha256_bytes(matrix_bytes) != FROZEN_MATRIX_SHA256:
        raise WitnessError("frozen matrix file hash changed")
    matrix = json.loads(matrix_bytes)
    unsealed = {key: value for key, value in matrix.items() if key != "matrix_seal_sha256"}
    if matrix.get("matrix_seal_sha256") != FROZEN_MATRIX_SEAL or frozen_semantic_sha256(unsealed) != FROZEN_MATRIX_SEAL:
        raise WitnessError("frozen matrix semantic seal changed")
    authority_bytes = authority_path.read_bytes()
    if sha256_bytes(authority_bytes) != FROZEN_AUTHORITY_SHA256:
        raise WitnessError("frozen capture authority hash changed")
    cells = matrix.get("cells")
    if not isinstance(cells, list):
        raise WitnessError("frozen matrix cells are unavailable")
    matches = [
        cell for cell in cells
        if isinstance(cell, dict)
        and (cell.get("identity") or {}).get("slug") == BASE_SLUG
    ]
    if len(matches) != 1:
        raise WitnessError("frozen BF16 cuBLAS base cell is not unique")
    cell = matches[0]
    if (
        cell.get("authority_cell_sha256") != FROZEN_BASE_CELL_SHA256
        or cell.get("disposition") != "runnable"
    ):
        raise WitnessError("frozen BF16 cuBLAS base cell changed")
    return {
        "matrix_path": BASE_MATRIX_REL.as_posix(),
        "matrix_file_sha256": sha256_bytes(matrix_bytes),
        "matrix_source_commit": matrix.get("source_commit"),
        "matrix_seal_sha256": matrix.get("matrix_seal_sha256"),
        "authority_path": BASE_AUTHORITY_REL.as_posix(),
        "authority_sha256": sha256_bytes(authority_bytes),
        "base_slug": BASE_SLUG,
        "base_authority_cell_sha256": cell["authority_cell_sha256"],
        "base_disposition": cell["disposition"],
        "relation": "evaluation_successor_no_relabel",
        "claim_policy": "none_additive_supplement_only",
    }


def load_cutlass_provenance() -> tuple[dict[str, Any], str]:
    path = REPO_ROOT / PROVENANCE_REL
    _regular(path, label="CUTLASS provenance")
    payload = path.read_bytes()
    document = json.loads(payload)
    if (
        document.get("upstream_tag") != "v4.6.1"
        or document.get("upstream_commit") != CUTLASS_UPSTREAM_COMMIT
        or document.get("upstream_tree") != CUTLASS_UPSTREAM_TREE
        or (document.get("digest") or {}).get("sha256") != CUTLASS_SCOPED_DIGEST
        or document.get("version") != {"major": 4, "minor": 6, "patch": 1}
    ):
        raise WitnessError("CUTLASS provenance identity is not v4.6.1")
    return document, sha256_bytes(payload)


def source_contract(source_commit: str) -> dict[str, Any]:
    paths = [
        REPO_ROOT / COMMON_MK_REL,
        REPO_ROOT / SHAPE_REL,
        WORKLOAD_DIR / "Makefile",
        WORKLOAD_DIR / "vendor_gemm_cutlass.cu",
        WORKLOAD_DIR / "main.cu",
    ]
    guarded = artifact_guard.verify_source_inputs(paths, repo_root=REPO_ROOT)
    records = guarded["files"]
    for path, expected in records.items():
        if sha256_bytes(git_blob(source_commit, path)) != expected:
            raise WitnessError(f"source input differs from producer commit: {path}")
    return {
        "files": records,
        "input_set_sha256": guarded["input_set_sha256"],
        "policy": "source_and_makefile_no_cublas_references",
    }


def build_command() -> list[str]:
    return [
        "make", "-B", "-C", WORKLOAD_REL.as_posix(), "run-bin",
        f"NVCC={NVCC}", f"NVCC_CCBIN={HOST_CXX}",
        f"CUOBJDUMP={CUOBJDUMP}", f"CUTLASS_DIR={CUTLASS_REL}",
    ]


def cubin_inventory(binary: Path) -> dict[str, Any]:
    command = [str(CUOBJDUMP), "--list-elf", str(binary)]
    rc, stdout, stderr = run_command(command, timeout=120)
    if rc != 0 or stderr:
        raise WitnessError(f"cuobjdump --list-elf failed: {stderr[-300:]!r}")
    text = stdout.decode("utf-8", errors="strict")
    cubins: list[str] = []
    for line in text.splitlines():
        match = re.fullmatch(r"ELF file\s+([1-9][0-9]*): (.+)", line)
        if match is None:
            raise WitnessError(f"malformed cuobjdump ELF inventory line: {line!r}")
        name = match.group(2)
        if Path(name).name != name or not name.endswith(".cubin"):
            raise WitnessError(f"unsafe cubin inventory name: {name!r}")
        cubins.append(name)
    if not cubins or not any(".sm_86.cubin" in name for name in cubins):
        raise WitnessError("BF16 successor has no sm_86 cubin")
    return {
        "command": command,
        "transcript_sha256": sha256_bytes(stdout),
        "transcript_bytes": len(stdout),
        "cubins": cubins,
        "sm86_cubin_present": True,
    }


def record_sass(binary: Path, evidence_dir: Path) -> dict[str, Any]:
    command = [str(CUOBJDUMP), "--dump-sass", str(binary)]
    rc, stdout, stderr = run_command(command, timeout=600)
    if rc != 0 or stderr:
        raise WitnessError(f"cuobjdump --dump-sass failed: {stderr[-300:]!r}")
    path = evidence_dir / SASS_NAME
    _write_or_match(path, stdout)
    return {
        "path": relative(path),
        "sha256": sha256_bytes(stdout),
        "bytes": len(stdout),
        "command": command,
    }


def record_control_bits(binary: Path, evidence_dir: Path) -> dict[str, Any]:
    path = evidence_dir / CONTROL_NAME
    with tempfile.TemporaryDirectory(prefix="cutlass-control-") as temporary:
        temporary_output = Path(temporary) / CONTROL_NAME
        command = [
            str(PYTHON), "tools/sass_control/emit_control_bits_sidecar.py",
            "--binary", str(binary), "--arch", ARCH, "--out", str(temporary_output),
            "--toolkit-bindir", str(NVCC.parent),
        ]
        rc, stdout, stderr = run_command(command, timeout=600)
        if rc != 0:
            raise WitnessError(
                f"control-bit extraction failed rc={rc}: "
                f"{(stdout + stderr)[-500:]!r}"
            )
        try:
            payload = temporary_output.read_bytes()
            document = json.loads(payload)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise WitnessError(f"control sidecar is unreadable: {exc}") from exc
    if document.get("arch") != ARCH or document.get("toolkit_version") != "V13.2":
        raise WitnessError("control sidecar architecture/toolkit identity mismatch")
    functions = document.get("functions")
    if not isinstance(functions, dict) or not functions:
        raise WitnessError("control sidecar has no functions")
    records = [record for items in functions.values() for record in items]
    if not records or any(record.get("status") != "decoded" for record in records):
        raise WitnessError("control sidecar contains unknown or missing encodings")
    summary = document.get("summary") or {}
    if summary.get("total_unknown_encoding") != 0 or summary.get("total_decoded") != len(records):
        raise WitnessError("control sidecar summary does not match records")
    _write_or_match(path, payload)
    return {
        "path": relative(path),
        "sha256": sha256_bytes(payload),
        "bytes": len(payload),
        "decoded_instructions": len(records),
        "unknown_encoding": summary["total_unknown_encoding"],
        "toolkit_version": document["toolkit_version"],
        "function_count": len(functions),
        "command": command,
        "command_stdout_sha256": sha256_bytes(stdout),
        "command_stderr_sha256": sha256_bytes(stderr),
    }


def smoke(binary: Path, shape: dict[str, Any], args: list[str]) -> dict[str, Any]:
    command = [str(binary), *args]
    rc, stdout, stderr = run_command(command, cwd=WORKLOAD_DIR, timeout=180)
    if rc != 0 or stderr:
        raise WitnessError(f"successor smoke failed: rc={rc}, stderr={stderr[-300:]!r}")
    try:
        document = json.loads(stdout)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise WitnessError(f"successor smoke output is not JSON: {exc}") from exc
    axes = shape["axes"]
    output = document.get("output")
    dimensions_ok = (
        isinstance(output, list) and len(output) == axes["m"]
        and all(isinstance(row, list) and len(row) == axes["n"] for row in output)
        and all(math.isfinite(float(value)) for row in output for value in row)
    )
    contract_ok = (
        document.get("schema_version") == 1
        and document.get("slug") == SLUG
        and document.get("shape_set") == "successor_shapes"
        and document.get("shape_role") == "trace"
        and document.get("shape_point") == shape["name"]
        and document.get("shape_axes") == axes
        and dimensions_ok
    )
    if not contract_ok:
        raise WitnessError("successor smoke output shape contract failed")
    return {
        "command": command,
        "returncode": rc,
        "output_sha256": sha256_bytes(stdout),
        "output_bytes": len(stdout),
        "reported_slug": document["slug"],
        "reported_shape_set": document["shape_set"],
        "reported_shape_role": document["shape_role"],
        "reported_shape_point": document["shape_point"],
        "reported_shape_axes": document["shape_axes"],
        "contract_ok": True,
    }


def build_witness(
    *, evidence_dir: Path, execute_build: bool, execute_smoke: bool,
    source_commit: str | None = None,
) -> dict[str, Any]:
    evidence_dir = evidence_dir.resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    producer_commit = source_commit or git_head()
    if not re.fullmatch(r"[0-9a-f]{40}", producer_commit):
        raise WitnessError("invalid producer source commit")
    base = load_frozen_base()
    shape, args, shape_digest = load_shape()
    provenance, provenance_sha = load_cutlass_provenance()
    source = source_contract(producer_commit)
    tree_hash = live_cutlass_tree_hash(REPO_ROOT / CUTLASS_REL)
    if not NVCC.is_file() or not HOST_CXX.is_file() or not CUOBJDUMP.is_file():
        raise WitnessError("pinned CUDA/compiler tools are unavailable")

    build = build_command()
    if execute_build:
        rc, stdout, stderr = run_command(build, timeout=600)
        transcript = stdout + stderr
        _write_or_match(evidence_dir / BUILD_LOG_NAME, transcript)
        if rc != 0:
            raise WitnessError(f"successor build failed rc={rc}")
    else:
        raise WitnessError("build witness requires --build")

    binary = REPO_ROOT / BINARY_REL
    _regular(binary, label="successor executable", executable=True)
    no_cublas = artifact_guard.verify_cutlass_artifact(
        binary=binary,
        source_inputs=[
            REPO_ROOT / COMMON_MK_REL, REPO_ROOT / SHAPE_REL,
            WORKLOAD_DIR / "Makefile", WORKLOAD_DIR / "vendor_gemm_cutlass.cu",
            WORKLOAD_DIR / "main.cu",
        ],
        repo_root=REPO_ROOT,
    )
    inventory = cubin_inventory(binary)
    sass = record_sass(binary, evidence_dir)
    control = record_control_bits(binary, evidence_dir)
    if execute_smoke:
        smoke_doc = smoke(binary, shape, args)
    else:
        raise WitnessError("build witness requires smoke execution")

    build_log = evidence_dir / BUILD_LOG_NAME
    _regular(build_log, label="build transcript")
    transcript = build_log.read_bytes()
    toolchain = {
        "nvcc": command_identity(NVCC),
        "host_cxx": command_identity(HOST_CXX),
        "cuobjdump": command_identity(CUOBJDUMP),
        "python": command_identity(PYTHON),
        "readelf": command_identity(Path("/usr/bin/readelf")),
        "make": command_identity(Path("/usr/bin/make")),
        "cutlass_dir": str((REPO_ROOT / CUTLASS_REL).resolve()),
    }
    document: dict[str, Any] = {
        "schema_version": 1,
        "witness_id": "sm86_cutlass_successor_build_witness_v1",
        "artifact_id": "cutlass_successor_sm86_v1",
        "arch": ARCH,
        "sku": SKU,
        "source_commit": producer_commit,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_frozen_v1": base,
        "source_inputs": source,
        "shape_authority": {
            "path": SHAPE_REL.as_posix(),
            "sha256": sha256_file(REPO_ROOT / SHAPE_REL),
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
            "command": build,
            "returncode": 0,
            "transcript_path": relative(build_log),
            "transcript_sha256": sha256_bytes(transcript),
            "transcript_bytes": len(transcript),
        },
        "artifacts": {
            "binary": {
                "path": BINARY_REL.as_posix(),
                "sha256": sha256_file(binary),
                "bytes": binary.stat().st_size,
            },
            "cubin_inventory": inventory,
            "sass": sass,
            "control_bits": control,
            "no_cublas": no_cublas,
        },
        "smoke": smoke_doc,
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
    document["witness_seal_sha256"] = canonical_sha256(document)
    return document


def validate_witness(document: dict[str, Any], *, repo_root: Path = REPO_ROOT) -> list[str]:
    """Validate schema and the byte/hash identities of a committed witness."""
    errors: list[str] = []
    schema_path = repo_root / SCHEMA_REL
    try:
        schema = json.loads(schema_path.read_text())
        jsonschema.Draft202012Validator(schema).validate(document)
    except (OSError, json.JSONDecodeError, jsonschema.ValidationError, jsonschema.SchemaError) as exc:
        errors.append(f"schema:{exc}")
        return errors
    expected_seal = canonical_sha256({key: value for key, value in document.items() if key != "witness_seal_sha256"})
    if document.get("witness_seal_sha256") != expected_seal:
        errors.append("witness_seal")
    for path_key, digest in document["source_inputs"]["files"].items():
        path = repo_root / path_key
        if not path.is_file() or sha256_file(path) != digest:
            errors.append(f"source_input:{path_key}")
    base = document["base_frozen_v1"]
    matrix = repo_root / base["matrix_path"]
    authority = repo_root / base["authority_path"]
    if not matrix.is_file() or sha256_file(matrix) != base["matrix_file_sha256"]:
        errors.append("frozen_matrix")
    if not authority.is_file() or sha256_file(authority) != base["authority_sha256"]:
        errors.append("frozen_authority")
    for section in ("binary", "sass", "control_bits"):
        artifact = document["artifacts"][section]
        path = repo_root / artifact["path"]
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            errors.append(f"artifact:{section}")
    log = repo_root / document["build"]["transcript_path"]
    if not log.is_file() or sha256_file(log) != document["build"]["transcript_sha256"]:
        errors.append("build_transcript")
    return sorted(set(errors))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(DEFAULT_OUT_REL))
    parser.add_argument("--evidence-dir", default=str(DEFAULT_EVIDENCE_REL))
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--source-commit")
    args = parser.parse_args(argv)
    if not args.build:
        print("cutlass successor witness: --build is required", file=sys.stderr)
        return 2
    try:
        document = build_witness(
            evidence_dir=(REPO_ROOT / args.evidence_dir).resolve()
            if not Path(args.evidence_dir).is_absolute() else Path(args.evidence_dir),
            execute_build=True,
            execute_smoke=not args.no_smoke,
            source_commit=args.source_commit,
        )
        problems = validate_witness(document)
        if problems:
            raise WitnessError("self-validation failed: " + ", ".join(problems))
        output = (REPO_ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
        _write_or_match(output, (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    except (OSError, ValueError, WitnessError, jsonschema.ValidationError) as exc:
        print(f"cutlass successor witness: failed: {exc}", file=sys.stderr)
        return 1
    print(f"cutlass successor witness: pass=True -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
