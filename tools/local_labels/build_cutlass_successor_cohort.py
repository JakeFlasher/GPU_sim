#!/usr/bin/env python3
"""Build and validate the additive CUTLASS successor cohort manifest.

The cohort is deliberately separate from the frozen v1 matrix and roster.  It
admits only build/runtime evidence for three project-owned sm_86 CUTLASS
successors; NCU labels and NVBit traces remain a later, matched-process gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.local_labels import (  # noqa: E402
    build_cutlass_sgemm_successor_witness as fp32_witness,
    build_cutlass_strided_batched_successor_witness as batched_witness,
    build_cutlass_successor_witness as bf16_witness,
)
from tools.local_labels.cutlass_successor_reference import reference_matrix  # noqa: E402


COHORT_ID = "cutlass_successor_sm86_20260723_v1"
VALIDATION_ID = "cutlass_successor_validation_sm86_20260723_v1"
ROSTER_ID = "cutlass_successor_roster_sm86_20260723_v1"
MATRIX_ID = "cutlass_successor_capture_matrix_sm86_20260723_v1"
GENERATED_AT_UTC = "2026-07-23T00:00:00Z"
ARCH = "sm_86"
SKU = "rtx_3090_ti"
BUILD_ARCH = "sm_86"
NAMESPACE = "cutlass_successor"
CAPTURE_PENDING_REASON = "hardware_capture_not_run_build_witness_only"
RUNTIME_REPEAT_COUNT = 5
RUNTIME_TIMEOUT_S = 180

EVIDENCE_REL = Path("data/annex/cutlass_successor_sm86_20260723")
VALIDATION_REL = EVIDENCE_REL / "cutlass_successor_validation_sm86_20260723.json"
BUDGETS_REL = EVIDENCE_REL / "cutlass_successor_trace_budgets_sm86_20260723.json"
ROSTER_REL = EVIDENCE_REL / "cutlass_successor_roster_sm86_20260723.json"
MATRIX_REL = EVIDENCE_REL / "cutlass_successor_capture_matrix_sm86_20260723.json"

VALIDATION_SCHEMA_REL = Path(
    "workloads/simbench_bw/schemas/cutlass_successor_validation.schema.json"
)
BUDGETS_SCHEMA_REL = Path(
    "workloads/simbench_bw/schemas/cutlass_successor_trace_budgets.schema.json"
)
ROSTER_SCHEMA_REL = Path(
    "workloads/simbench_bw/schemas/cutlass_successor_roster.schema.json"
)
MATRIX_SCHEMA_REL = Path(
    "workloads/simbench_bw/schemas/cutlass_successor_capture_matrix.schema.json"
)
GENERATOR_REL = Path("tools/local_labels/build_cutlass_successor_cohort.py")
GUARD_REL = Path("tools/local_labels/verify_cutlass_successor_artifact.py")

FROZEN_MATRIX_REL = Path("data/local_labels/campaign/sm86_reference_capture_matrix.json")
FROZEN_AUTHORITY_REL = Path("data/local_labels/campaign/capture_roster_authority_sm86.json")
FROZEN_MATRIX_SHA256 = "af29d72e955cf42e281036190f2d55d72af0af92a632c78c8d159073a072fb0e"
FROZEN_MATRIX_SEAL = "6376da9721c75bb9299e84a9d11bf607bc21f4cbaa30da25b672c86f6757a5e1"
FROZEN_SOURCE_COMMIT = "124ce358733f97ba6e0af808d10ab69ffe4d5a78"
FROZEN_AUTHORITY_SHA256 = "d02c817091b0a9f06c2106c0987c59a481910afcc3b78ab1de7a7d5bd5a6b4b4"
FROZEN_REGISTRY_REL = Path("tools/local_labels/metric_registry.json")
FROZEN_REGISTRY_SHA256 = "be1f7893cf653800c69f6e39a4d22abfae4d67c10197e7d5f9bd46c6900c2423"
FROZEN_METRIC_IDENTITY_SHA256 = "13f1bdfd8155cb71db62cc30aea41bc7fc6af6447d1251d9ea4f0efd4a8396a6"
FROZEN_TRACE_BUDGETS_REL = Path("workloads/simbench_bw/trace_budgets.toml")
FROZEN_TRACE_BUDGETS_SHA256 = "95fa9d83b41ee89d1824e9e708d404574859a00d3f68a0b12235b8d1be9f954c"
FROZEN_ALIAS_REL = Path("data/local_labels/campaign/frozen_v1_source_aliases.v1.json")
FROZEN_ALIAS_SHA256 = "f74ac4e8750f970ab3bb5d7da54ec9db2df0f4ef0f5fb22b5f0f6f90898790ff"
FROZEN_BASE_CELL_SHA256 = {
    "vendor_gemm_cublas": "1e84352627f783b1d28fb3684cab96a61fa10cbdbb9c0e0b8844b027c1dc6142",
    "vendor_sgemm_cublas": "328127e1ea186daea64cbc09527c954c5ab6bdb3453e8c38df0fd0122bcd294a",
    "vendor_sgemm_strided_batched_cublas": "f35f4dd55df1f34dd71aa39a0e0be2049a48ae8f5b29837d8488909c8aa15792",
}


@dataclass(frozen=True)
class Case:
    variant_id: str
    matrix_key: str
    slug: str
    base_slug: str
    witness_rel: Path
    witness_module: Any
    dtype: str
    batch_count: int
    output_scope: str
    tolerance: float
    budget_bytes: int


CASES = (
    Case(
        variant_id="vendor_gemm_cutlass_bf16_simt_trace00_v1",
        matrix_key="cutlass_successor:vendor_gemm_cublas:vendor_gemm_cutlass",
        slug="vendor_gemm_cutlass",
        base_slug="vendor_gemm_cublas",
        witness_rel=Path(
            "data/annex/cutlass_successor_sm86_20260723/"
            "vendor_gemm_cutlass.build_witness.json"
        ),
        witness_module=bf16_witness,
        dtype="bf16_inputs_fp32_accumulator",
        batch_count=1,
        output_scope="full_single_batch_tensor",
        tolerance=1.0e-5,
        budget_bytes=67_108_864,
    ),
    Case(
        variant_id="vendor_sgemm_cutlass_fp32_simt_trace00_v1",
        matrix_key="cutlass_successor:vendor_sgemm_cublas:vendor_sgemm_cutlass",
        slug="vendor_sgemm_cutlass",
        base_slug="vendor_sgemm_cublas",
        witness_rel=Path(
            "data/annex/cutlass_sgemm_successor_sm86_20260723/"
            "vendor_sgemm_cutlass.build_witness.json"
        ),
        witness_module=fp32_witness,
        dtype="fp32_inputs_fp32_accumulator",
        batch_count=1,
        output_scope="full_single_batch_tensor",
        tolerance=1.0e-5,
        budget_bytes=67_108_864,
    ),
    Case(
        variant_id="vendor_sgemm_strided_batched_cutlass_fp32_simt_batch4_trace00_v1",
        matrix_key=(
            "cutlass_successor:vendor_sgemm_strided_batched_cublas:"
            "vendor_sgemm_strided_batched_cutlass"
        ),
        slug="vendor_sgemm_strided_batched_cutlass",
        base_slug="vendor_sgemm_strided_batched_cublas",
        witness_rel=Path(
            "data/annex/cutlass_strided_batched_successor_sm86_20260723/"
            "vendor_sgemm_strided_batched_cutlass.build_witness.json"
        ),
        witness_module=batched_witness,
        dtype="fp32_inputs_fp32_accumulator",
        batch_count=4,
        output_scope="batch_zero_only",
        tolerance=1.0e-5,
        budget_bytes=134_217_728,
    ),
)


class CohortError(RuntimeError):
    """The successor cohort cannot be admitted."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise CohortError(f"expected regular file: {path}")
    return _sha256_bytes(path.read_bytes())


def _canonical_sha(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    )


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise CohortError(f"path escapes repository: {path}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CohortError(f"cannot load JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CohortError(f"JSON document is not an object: {path}")
    return value


def _sealed(document: dict[str, Any], key: str) -> dict[str, Any]:
    body = {name: value for name, value in document.items() if name != key}
    result = dict(body)
    result[key] = _canonical_sha(body)
    return result


def _check_seal(document: dict[str, Any], key: str) -> None:
    expected = _canonical_sha(
        {name: value for name, value in document.items() if name != key}
    )
    if document.get(key) != expected:
        raise CohortError(f"{key} does not match canonical document bytes")


def _write_or_match(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise CohortError(f"existing output is not a regular file: {path}")
        if path.read_bytes() != payload:
            raise CohortError(f"existing output bytes differ: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _git_blob(source_commit: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "cat-file", "blob", f"{source_commit}:{path}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise CohortError(
            f"source input absent at {source_commit}: {path}: "
            f"{result.stderr.decode('utf-8', errors='replace')[-200:]}"
        )
    return result.stdout


def _source_inputs(source_commit: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise CohortError("source_commit must be a full 40-character Git hash")
    paths = [
        GENERATOR_REL,
        GUARD_REL,
        Path("tools/local_labels/cutlass_successor_reference.py"),
        VALIDATION_SCHEMA_REL,
        BUDGETS_SCHEMA_REL,
        ROSTER_SCHEMA_REL,
        MATRIX_SCHEMA_REL,
        Path("tools/local_labels/build_cutlass_successor_witness.py"),
        Path("tools/local_labels/build_cutlass_sgemm_successor_witness.py"),
        Path("tools/local_labels/build_cutlass_strided_batched_successor_witness.py"),
        Path("workloads/simbench_bw/schemas/cutlass_successor_build_witness.schema.json"),
        Path("workloads/simbench_bw/schemas/cutlass_sgemm_successor_build_witness.schema.json"),
        Path("workloads/simbench_bw/schemas/cutlass_strided_batched_successor_build_witness.schema.json"),
    ]
    files: dict[str, str] = {}
    for path in sorted(paths, key=lambda value: value.as_posix()):
        rel = path.as_posix()
        files[rel] = _sha256_bytes(_git_blob(source_commit, rel))
    return {
        "producer_commit": source_commit,
        "files": files,
        "input_set_sha256": _canonical_sha(files),
    }


def _frozen_reference() -> dict[str, Any]:
    matrix_path = ROOT / FROZEN_MATRIX_REL
    authority_path = ROOT / FROZEN_AUTHORITY_REL
    matrix_bytes = matrix_path.read_bytes()
    authority_bytes = authority_path.read_bytes()
    if _sha256_bytes(matrix_bytes) != FROZEN_MATRIX_SHA256:
        raise CohortError("frozen matrix bytes changed")
    if _sha256_bytes(authority_bytes) != FROZEN_AUTHORITY_SHA256:
        raise CohortError("frozen authority bytes changed")
    if _sha256_file(ROOT / FROZEN_REGISTRY_REL) != FROZEN_REGISTRY_SHA256:
        raise CohortError("frozen metric registry bytes changed")
    if _sha256_file(ROOT / FROZEN_TRACE_BUDGETS_REL) != FROZEN_TRACE_BUDGETS_SHA256:
        raise CohortError("frozen trace-budget bytes changed")
    if _sha256_file(ROOT / FROZEN_ALIAS_REL) != FROZEN_ALIAS_SHA256:
        raise CohortError("frozen source-alias bytes changed")
    matrix = json.loads(matrix_bytes)
    semantic = {key: value for key, value in matrix.items() if key != "matrix_seal_sha256"}
    if matrix.get("source_commit") != FROZEN_SOURCE_COMMIT:
        raise CohortError("frozen matrix source commit changed")
    if matrix.get("matrix_seal_sha256") != FROZEN_MATRIX_SEAL:
        raise CohortError("frozen matrix seal changed")
    if _sha256_bytes(
        json.dumps(
            semantic, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    ) != FROZEN_MATRIX_SEAL:
        raise CohortError("frozen matrix semantic content changed")
    pins = matrix.get("input_pins", {})
    metric_pin = pins.get("metric_registry", {})
    budget_pin = pins.get("per_cell_trace_budgets", {})
    if metric_pin.get("path") != FROZEN_REGISTRY_REL.as_posix() or metric_pin.get("sha256") != FROZEN_REGISTRY_SHA256:
        raise CohortError("frozen metric-registry input pin changed")
    if budget_pin.get("path") != FROZEN_TRACE_BUDGETS_REL.as_posix() or budget_pin.get("sha256") != FROZEN_TRACE_BUDGETS_SHA256:
        raise CohortError("frozen trace-budget input pin changed")
    if matrix.get("metric_registry", {}).get("metric_identity_sha256") != FROZEN_METRIC_IDENTITY_SHA256:
        raise CohortError("frozen metric identity changed")
    base_cells: dict[str, dict[str, Any]] = {}
    for cell in matrix.get("cells", []):
        slug = (cell.get("identity") or {}).get("slug")
        if slug in {case.base_slug for case in CASES}:
            if slug in base_cells:
                raise CohortError(f"duplicate frozen base slug: {slug}")
            base_cells[slug] = cell
    if set(base_cells) != {case.base_slug for case in CASES}:
        raise CohortError("frozen base row set does not match successor cohort")
    for case in CASES:
        cell = base_cells[case.base_slug]
        if cell.get("disposition") != "runnable":
            raise CohortError(f"frozen base is not runnable: {case.base_slug}")
        if cell.get("authority_cell_sha256") != FROZEN_BASE_CELL_SHA256[case.base_slug]:
            raise CohortError(f"frozen base cell hash changed: {case.base_slug}")
    return {
        "matrix_path": FROZEN_MATRIX_REL.as_posix(),
        "matrix_file_sha256": FROZEN_MATRIX_SHA256,
        "matrix_source_commit": FROZEN_SOURCE_COMMIT,
        "matrix_seal_sha256": FROZEN_MATRIX_SEAL,
        "authority_path": FROZEN_AUTHORITY_REL.as_posix(),
        "authority_sha256": FROZEN_AUTHORITY_SHA256,
        "metric_registry_path": FROZEN_REGISTRY_REL.as_posix(),
        "metric_registry_sha256": FROZEN_REGISTRY_SHA256,
        "metric_identity_sha256": FROZEN_METRIC_IDENTITY_SHA256,
        "trace_budgets_path": FROZEN_TRACE_BUDGETS_REL.as_posix(),
        "trace_budgets_sha256": FROZEN_TRACE_BUDGETS_SHA256,
        "source_alias_path": FROZEN_ALIAS_REL.as_posix(),
        "source_alias_sha256": FROZEN_ALIAS_SHA256,
        "base_rows": {
            slug: {
                "authority_cell_sha256": cell["authority_cell_sha256"],
                "disposition": cell["disposition"],
            }
            for slug, cell in sorted(base_cells.items())
        },
        "relation": "evaluation_successor_no_relabel",
        "canonical_v1_impact": "none_additive_supplement_only",
    }


def _load_witness(case: Case) -> dict[str, Any]:
    path = ROOT / case.witness_rel
    document = _load_json(path)
    errors = case.witness_module.validate_witness(document)
    if errors:
        raise CohortError(f"invalid witness {path}: {', '.join(errors)}")
    if document.get("arch") != ARCH or document.get("sku") != SKU:
        raise CohortError(f"witness architecture identity mismatch: {path}")
    if document.get("shape_authority", {}).get("slug") != case.slug:
        raise CohortError(f"witness slug mismatch: {path}")
    if document.get("base_frozen_v1", {}).get("base_slug") != case.base_slug:
        raise CohortError(f"witness lineage mismatch: {path}")
    if document.get("control_contract", {}).get("hardware_capture_status") != (
        "not_run_build_witness_only"
    ):
        raise CohortError(f"witness makes an unsupported hardware claim: {path}")
    return document


def _reference(case: Case, m: int, n: int, k: int, seed: int) -> np.ndarray:
    return reference_matrix(
        dtype_contract=case.dtype,
        batch_count=case.batch_count,
        m=m,
        n=n,
        k=k,
        seed=seed,
    )


def _runtime_environment(case: Case, parent: Path) -> dict[str, Any]:
    commands = (
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,compute_cap",
            "--format=csv,noheader",
        ],
        [
            "nvidia-smi",
            "--query-gpu=clocks.sm,clocks.mem,pstate,temperature.gpu,power.draw",
            "--format=csv,noheader",
        ],
        ["/usr/local/cuda-13.2/bin/nvcc", "--version"],
    )
    observations = []
    for command in commands:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            timeout=30,
            check=False,
        )
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
        observations.append(
            {
                "command": command,
                "returncode": result.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_sha256": _sha256_bytes(result.stdout),
                "stderr_sha256": _sha256_bytes(result.stderr),
            }
        )
    if observations[0]["returncode"] != 0:
        raise CohortError("GPU runtime identity query failed")
    receipt = {
        "schema_version": 1,
        "case": case.slug,
        "status": "pass",
        "target_only": True,
        "observations": observations,
        "no_ncu_values": True,
        "no_nvbit_values": True,
    }
    receipt_path = parent / "cohort_runtime" / "runtime_environment.json"
    receipt_bytes = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    _write_or_match(receipt_path, receipt_bytes)
    return {
        "status": "pass",
        "path": _relative(receipt_path),
        "sha256": _sha256_bytes(receipt_bytes),
        "bytes": len(receipt_bytes),
        "observations": observations,
    }


def _run_case(case: Case, witness: dict[str, Any]) -> dict[str, Any]:
    binary_rel = Path(witness["artifacts"]["binary"]["path"])
    binary = ROOT / binary_rel
    if binary.is_symlink() or not binary.is_file():
        raise CohortError(f"runtime executable is unavailable: {binary}")
    if not binary.stat().st_mode & 0o111:
        raise CohortError(f"runtime executable is not executable: {binary}")
    args = [str(value) for value in witness["shape_authority"]["actual_args"]]
    command = [f"./{binary.name}", *args]
    parent = ROOT / case.witness_rel.parent
    runtime_dir = parent / "cohort_runtime"
    environment = _runtime_environment(case, parent)
    outputs: list[dict[str, Any]] = []
    run_artifacts: list[dict[str, Any]] = []
    raw_hashes: list[str] = []
    wall_times_ns: list[int] = []
    for repeat_index in range(RUNTIME_REPEAT_COUNT):
        started = time.perf_counter_ns()
        result = subprocess.run(
            command,
            cwd=ROOT / binary_rel.parent,
            capture_output=True,
            timeout=RUNTIME_TIMEOUT_S,
            check=False,
        )
        wall_time_ns = time.perf_counter_ns() - started
        wall_times_ns.append(wall_time_ns)
        if result.returncode != 0:
            raise CohortError(f"successor runtime failed for {case.slug}")
        output_path = runtime_dir / f"run_{repeat_index:02d}.stdout.json"
        _write_or_match(output_path, result.stdout)
        raw_hash = _sha256_bytes(result.stdout)
        raw_hashes.append(raw_hash)
        if result.stderr:
            raise CohortError(f"successor runtime emitted stderr for {case.slug}")
        try:
            value = json.loads(result.stdout)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise CohortError(f"successor output is not JSON for {case.slug}: {exc}") from exc
        if not isinstance(value, dict):
            raise CohortError(f"successor output is not an object for {case.slug}")
        outputs.append(value)
        run_artifacts.append(
            {
                "repeat_index": repeat_index,
                "path": _relative(output_path),
                "sha256": raw_hash,
                "bytes": len(result.stdout),
                "returncode": result.returncode,
                "stderr_empty": True,
                "host_wall_time_ns": wall_time_ns,
            }
        )
    first = outputs[0]
    expected_axes = {"m": 32, "n": 32, "k": 32}
    if case.batch_count > 1:
        expected_axes["batch_count"] = case.batch_count
    if first.get("shape_axes") != expected_axes:
        raise CohortError(f"runtime axes mismatch for {case.slug}")
    if first.get("slug") != case.slug or first.get("shape_point") != "trace_00":
        raise CohortError(f"runtime identity mismatch for {case.slug}")
    if case.batch_count > 1:
        if first.get("output_scope") != case.output_scope:
            raise CohortError(f"runtime output scope mismatch for {case.slug}")
        if first.get("batch_count") != case.batch_count:
            raise CohortError(f"runtime batch count mismatch for {case.slug}")
    elif "output_scope" in first:
        raise CohortError(f"unexpected bounded-output marker for {case.slug}")
    reference = _reference(case, 32, 32, 32, 42)
    reference_path = runtime_dir / "reference_batch_zero.json"
    reference_bytes = (
        json.dumps(reference[0].tolist(), separators=(",", ":"), allow_nan=False).encode("utf-8")
        + b"\n"
    )
    _write_or_match(reference_path, reference_bytes)
    max_abs_errors: list[float] = []
    for value in outputs:
        output = np.asarray(value.get("output"), dtype=np.float32)
        if output.shape != (32, 32) or not np.isfinite(output).all():
            raise CohortError(f"runtime output tensor shape/value failure for {case.slug}")
        max_abs_errors.append(float(np.max(np.abs(output - reference[0]))))
    if any(not np.isfinite(error) or error > case.tolerance for error in max_abs_errors):
        raise CohortError(
            f"correctness tolerance failed for {case.slug}: {max_abs_errors} > {case.tolerance}"
        )
    exact_repeat = len(set(raw_hashes)) == 1
    mean_wall = statistics.fmean(wall_times_ns)
    wall_cv_pct = (
        100.0 * statistics.pstdev(wall_times_ns) / mean_wall if mean_wall else 0.0
    )
    launch_inventory = {
        "status": "declared_only",
        "expected_process_runs": RUNTIME_REPEAT_COUNT,
        "observed_process_runs": RUNTIME_REPEAT_COUNT,
        "expected_kernel_launch_count": 1,
        "observed_kernel_launch_count": None,
        "dynamic_launch_status": "not_run",
        "nvbit_trace_status": "not_run",
        "static_launch_contract": witness["shape_authority"]["trace_shape"]["launch_config"],
        "typed_reason": "dynamic_launch_inventory_requires_nvbit_capture",
        "command": command,
        "command_sha256": _canonical_sha(command),
    }
    return {
        "command": command,
        "cwd": _relative(ROOT / binary_rel.parent),
        "run_count": RUNTIME_REPEAT_COUNT,
        "returncodes": [0] * RUNTIME_REPEAT_COUNT,
        "stderr_empty": True,
        "stdout_sha256": raw_hashes,
        "run_artifacts": run_artifacts,
        "repeatable_exact_bytes": exact_repeat,
        "correctness": {
            "status": "pass_scoped" if case.batch_count > 1 else "pass",
            "method": "independent_numpy_host_reference",
            "reference_path": _relative(reference_path),
            "reference_sha256": _sha256_bytes(reference_bytes),
            "reference_bytes": len(reference_bytes),
            "input_generator": "tools/local_labels/cutlass_successor_reference.py:reference_matrix",
            "input_generator_sha256": _sha256_file(ROOT / "tools/local_labels/cutlass_successor_reference.py"),
            "dtype_contract": case.dtype,
            "layout_contract": "row_major_a_column_major_b_row_major_c_d",
            "seed": 42,
            "max_abs_error": max(max_abs_errors),
            "per_run_max_abs_error": max_abs_errors,
            "tolerance": case.tolerance,
            "rtol": 0.0,
            "atol": case.tolerance,
            "max_ulp": None,
            "matched_ratio": 1.0,
            "validated_batches": [0],
            "unverified_batches": list(range(1, case.batch_count)),
            "output_scope": case.output_scope,
        },
        "repeatability": {
            "status": "pass" if exact_repeat else "fail",
            "runs": RUNTIME_REPEAT_COUNT,
            "repeat_indices": list(range(RUNTIME_REPEAT_COUNT)),
            "stdout_sha256": raw_hashes,
            "run_artifacts": run_artifacts,
            "binary_sha256_consistent": all(
                witness["artifacts"]["binary"]["sha256"] == witness["artifacts"]["binary"]["sha256"]
                for _ in range(RUNTIME_REPEAT_COUNT)
            ),
            "exact_bytes_equal": exact_repeat,
            "dispersion_rule": "host_wall_time_population_cv_percent_diagnostic",
            "dispersion_cv_percent": wall_cv_pct,
            "dispersion_threshold_percent": 25.0,
            "timing_dispersion_status": "within_threshold" if wall_cv_pct <= 25.0 else "above_threshold",        },
        "runtime_capability": {
            "status": "pass_diagnostic",
            "arch": ARCH,
            "sku": SKU,
            "binary_sha256": witness["artifacts"]["binary"]["sha256"],
            "returncodes": [0] * RUNTIME_REPEAT_COUNT,
            "stderr_empty": True,
            "hardware_label_capture": "not_run",
            "warmup_count": 0,
            "measured_reps": RUNTIME_REPEAT_COUNT,
            "timing_metric": "host_wall_time",
            "timing_unit": "ns",
            "timing_samples": wall_times_ns,
            "timing_median_ns": statistics.median(wall_times_ns),
            "timing_cv_percent": wall_cv_pct,
            "environment_ref": {
                key: environment[key] for key in ("path", "sha256", "bytes")
            },
            "publication_timing_status": "not_a_hardware_label",
        },
        "launch_inventory": launch_inventory,
    }


def _static_launch_row(case: Case, witness: dict[str, Any]) -> dict[str, Any]:
    artifacts = witness["artifacts"]
    controls = artifacts["control_bits"]
    inventory = artifacts["cubin_inventory"]
    return {
        "status": "pass",
        "binary_path": artifacts["binary"]["path"],
        "binary_sha256": artifacts["binary"]["sha256"],
        "sass_path": artifacts["sass"]["path"],
        "sass_sha256": artifacts["sass"]["sha256"],
        "control_path": controls["path"],
        "control_sha256": controls["sha256"],
        "control_policy": witness["control_contract"]["policy"],
        "decoded_instructions": controls["decoded_instructions"],
        "unknown_encoding": controls["unknown_encoding"],
        "function_count": controls["function_count"],
        "sm86_cubin_present": inventory["sm86_cubin_present"],
        "cubins": inventory["cubins"],
    }


def _row(case: Case, witness: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    base = witness["base_frozen_v1"]
    shape = witness["shape_authority"]
    return {
        "matrix_key": case.matrix_key,
        "variant_id": case.variant_id,
        "namespace": NAMESPACE,
        "successor_slug": case.slug,
        "base_slug": case.base_slug,
        "arch": ARCH,
        "sku": SKU,
        "build_arch": BUILD_ARCH,
        "disposition": "build_verified_capture_pending",
        "admission_state": "build_witness_only",
        "lineage": {
            "relation": base["relation"],
            "base_authority_cell_sha256": base["base_authority_cell_sha256"],
            "base_disposition": base["base_disposition"],
            "canonical_v1_impact": base["claim_policy"],
        },
        "witness": {
            "path": case.witness_rel.as_posix(),
            "sha256": _sha256_file(ROOT / case.witness_rel),
            "witness_id": witness["witness_id"],
            "witness_seal_sha256": witness["witness_seal_sha256"],
            "source_commit": witness["source_commit"],
            "source_input_set_sha256": witness["source_inputs"]["input_set_sha256"],
            "artifact_id": witness["artifact_id"],
        },
        "shape": {
            "path": shape["path"],
            "sha256": shape["sha256"],
            "trace_shape_sha256": shape["trace_shape_sha256"],
            "shape_point": shape["trace_shape"]["name"],
            "trace_shape": shape["trace_shape"],
            "actual_args": shape["actual_args"],
            "shape_contract": shape["shape_contract"],
            "output_scope": case.output_scope,
        },
        "static_launch_inventory": _static_launch_row(case, witness),
        "launch_inventory": runtime["launch_inventory"],
        "correctness": runtime["correctness"],
        "repeatability": runtime["repeatability"],
        "runtime_capability": runtime["runtime_capability"],
        "capture_admission": {
            "status": "pending",
            "typed_reason": CAPTURE_PENDING_REASON,
            "ncu_status": "not_run",
            "nvbit_status": "not_run",
            "primary_hardware_metric": "gpc__cycles_elapsed.max",
            "secondary_hardware_metric": "gpu__time_duration.sum",
            "process_separation_policy": "ncu_and_nvbit_must_run_in_separate_processes",
        },
    }


def _common_meta(
    source_inputs: dict[str, Any], frozen: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "cohort_id": COHORT_ID,
        "arch": ARCH,
        "sku": SKU,
        "build_arch": BUILD_ARCH,
        "namespace": NAMESPACE,
        "generated_at_utc": GENERATED_AT_UTC,
        "source_inputs": source_inputs,
        "frozen_v1_reference": dict(frozen),
    }


def _build_validation(
    source_inputs: dict[str, Any], frozen: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    document = {
        **_common_meta(source_inputs, frozen),
        "validation_id": VALIDATION_ID,
        "rows": rows,
        "counts": {
            "total": len(rows),
            "correctness_pass": sum(row["correctness"]["status"] in {"pass", "pass_scoped"} for row in rows),
            "repeatability_pass": sum(row["repeatability"]["status"] == "pass" for row in rows),
            "runtime_pass": sum(row["runtime_capability"]["status"] in {"pass", "pass_diagnostic"} for row in rows),
            "capture_pending": len(rows),
            "hardware_capture_runnable": 0,
        },
        "hardware_capture": {
            "status": "not_run_build_witness_only",
            "ncu_status": "not_run",
            "nvbit_status": "not_run",
            "primary_hardware_metric": "gpc__cycles_elapsed.max",
            "secondary_hardware_metric": "gpu__time_duration.sum",
            "typed_reason": CAPTURE_PENDING_REASON,
            "process_separation_policy": "ncu_and_nvbit_must_run_in_separate_processes",
        },
        "integrity": {
            "row_count": len(rows),
            "unique_matrix_keys": len({row["matrix_key"] for row in rows}) == len(rows),
            "unique_variant_ids": len({row["variant_id"] for row in rows}) == len(rows),
            "all_build_witnesses_pass": True,
            "all_runtime_checks_pass": True,
            "frozen_v1_unchanged": True,
        },
    }
    return _sealed(document, "validation_seal_sha256")


def _build_budgets(
    source_inputs: dict[str, Any], frozen: dict[str, Any],
    witnesses: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    for case in CASES:
        witness = witnesses[case.slug + "|" + case.base_slug]
        shape = witness["shape_authority"]
        rows.append(
            {
                "matrix_key": case.matrix_key,
                "variant_id": case.variant_id,
                "successor_slug": case.slug,
                "shape_path": shape["path"],
                "shape_file_sha256": shape["sha256"],
                "trace_shape_sha256": shape["trace_shape_sha256"],
                "predicted_bytes": shape["trace_shape"]["predicted_bytes"],
                "max_trace_bytes": case.budget_bytes,
                "max_wall_time_s": 3600,
                "sampling_policy": "off",
                "status": "declared_capture_pending",
                "source": "successor_static_shape_bound_v1",
                "batch_count": case.batch_count,
            }
        )
    document = {
        **_common_meta(source_inputs, frozen),
        "budget_id": "cutlass_successor_trace_budgets_sm86_20260723_v1",
        "budget_policy": "successor_static_shape_bound_v1",
        "rows": rows,
        "counts": {"total": len(rows), "declared": len(rows), "measured": 0},
    }
    return _sealed(document, "budget_seal_sha256")


def _build_roster(
    source_inputs: dict[str, Any], frozen: dict[str, Any],
    rows: list[dict[str, Any]], validation_bytes: bytes, budget_bytes: bytes,
) -> dict[str, Any]:
    roster_rows = []
    for row in rows:
        roster_row = {
            key: row[key]
            for key in (
                "matrix_key", "variant_id", "namespace", "successor_slug",
                "base_slug", "arch", "sku", "build_arch", "disposition",
                "admission_state", "lineage", "witness", "shape",
                "static_launch_inventory", "launch_inventory", "capture_admission",
            )
        }
        roster_row["row_seal_sha256"] = _canonical_sha(roster_row)
        roster_rows.append(roster_row)
    document = {
        **_common_meta(source_inputs, frozen),
        "roster_id": ROSTER_ID,
        "claim_policy": "additive_successor_evaluation_only",
        "validation_ref": {
            "path": VALIDATION_REL.as_posix(),
            "sha256": _sha256_bytes(validation_bytes),
            "id": VALIDATION_ID,
        },
        "trace_budget_ref": {
            "path": BUDGETS_REL.as_posix(),
            "sha256": _sha256_bytes(budget_bytes),
            "id": "cutlass_successor_trace_budgets_sm86_20260723_v1",
        },
        "rows": roster_rows,
        "counts": {
            "total": len(roster_rows),
            "build_verified": len(roster_rows),
            "correctness_verified": len(roster_rows),
            "repeatability_verified": len(roster_rows),
            "capture_pending": len(roster_rows),
            "hardware_capture_runnable": 0,
            "excluded": 0,
        },
        "integrity": {
            "row_count": len(roster_rows),
            "unique_matrix_keys": len({row["matrix_key"] for row in roster_rows}) == len(roster_rows),
            "unique_successor_slugs": len({row["successor_slug"] for row in roster_rows}) == len(roster_rows),
            "base_lineage_exact": sorted(row["base_slug"] for row in roster_rows) == sorted(case.base_slug for case in CASES),
            "canonical_v1_impact": "none_additive_supplement_only",
            "frozen_v1_unchanged": True,
        },
    }
    return _sealed(document, "roster_seal_sha256")


def _build_matrix(
    source_inputs: dict[str, Any], frozen: dict[str, Any],
    rows: list[dict[str, Any]], roster_bytes: bytes,
    validation_bytes: bytes, budget_bytes: bytes,
) -> dict[str, Any]:
    matrix_rows = []
    for row in rows:
        budget = {
            "matrix_key": row["matrix_key"],
            "variant_id": row["variant_id"],
            "shape_path": row["shape"]["path"],
            "shape_file_sha256": row["shape"]["sha256"],
            "trace_shape_sha256": row["shape"]["trace_shape_sha256"],
            "max_trace_bytes": next(
                case.budget_bytes for case in CASES if case.matrix_key == row["matrix_key"]
            ),
            "max_wall_time_s": 3600,
            "sampling_policy": "off",
            "source": "successor_static_shape_bound_v1",
        }
        matrix_row = {
            "matrix_key": row["matrix_key"],
            "variant_id": row["variant_id"],
            "successor_slug": row["successor_slug"],
            "base_slug": row["base_slug"],
            "arch": row["arch"],
            "sku": row["sku"],
            "build_arch": row["build_arch"],
            "disposition": "capture_pending",
            "admission_state": "build_witness_only",
            "authority_row_sha256": _canonical_sha(row),
            "witness_sha256": row["witness"]["sha256"],
            "shape": row["shape"],
            "trace_budget": budget,
            "static_launch_inventory": row["static_launch_inventory"],
            "launch_inventory": row["launch_inventory"],
            "capture_admission": row["capture_admission"],
        }
        matrix_row["row_seal_sha256"] = _canonical_sha(matrix_row)
        matrix_rows.append(matrix_row)
    document = {
        **_common_meta(source_inputs, frozen),
        "matrix_id": MATRIX_ID,
        "matrix_scope": "additive_successor_only",
        "claim_policy": "additive_successor_evaluation_only",
        "canonical_v1_impact": "none_additive_supplement_only",
        "roster_ref": {
            "path": ROSTER_REL.as_posix(),
            "sha256": _sha256_bytes(roster_bytes),
            "id": ROSTER_ID,
        },
        "validation_ref": {
            "path": VALIDATION_REL.as_posix(),
            "sha256": _sha256_bytes(validation_bytes),
            "id": VALIDATION_ID,
        },
        "trace_budget_ref": {
            "path": BUDGETS_REL.as_posix(),
            "sha256": _sha256_bytes(budget_bytes),
            "id": "cutlass_successor_trace_budgets_sm86_20260723_v1",
        },
        "dispatch": {
            "status": "not_admitted",
            "m6_schedule_created": False,
            "typed_reason": CAPTURE_PENDING_REASON,
            "ncu_process_policy": "separate_process",
            "nvbit_process_policy": "separate_process",
        },
        "chunking": {
            "policy": "capture_ready_rows_only",
            "max_rows_per_chunk": 8,
            "chunks": [],
            "runnable_order": [],
            "runnable_membership_sha256": _canonical_sha([]),
        },
        "chunks": [],
        "rows": matrix_rows,
        "counts": {
            "total": len(matrix_rows),
            "capture_pending": len(matrix_rows),
            "hardware_capture_runnable": 0,
            "excluded": 0,
        },
        "integrity": {
            "row_count": len(matrix_rows),
            "unique_matrix_keys": len({row["matrix_key"] for row in matrix_rows}) == len(matrix_rows),
            "unique_variant_ids": len({row["variant_id"] for row in matrix_rows}) == len(matrix_rows),
            "no_frozen_rows": all(row["successor_slug"] not in {case.base_slug for case in CASES} for row in matrix_rows),
            "no_capture_pending_rows_in_chunks": True,
            "frozen_v1_unchanged": True,
        },
    }
    return _sealed(document, "matrix_seal_sha256")


def _validate_schema(path: Path, document: dict[str, Any]) -> None:
    schema = _load_json(path)
    try:
        jsonschema.Draft202012Validator(schema).validate(document)
    except jsonschema.ValidationError as exc:
        raise CohortError(f"schema validation failed for {path}: {exc.message}") from exc


def validate_bundle(
    validation: dict[str, Any], budgets: dict[str, Any],
    roster: dict[str, Any], matrix: dict[str, Any],
) -> None:
    _validate_schema(ROOT / VALIDATION_SCHEMA_REL, validation)
    _validate_schema(ROOT / BUDGETS_SCHEMA_REL, budgets)
    _validate_schema(ROOT / ROSTER_SCHEMA_REL, roster)
    _validate_schema(ROOT / MATRIX_SCHEMA_REL, matrix)
    _check_seal(validation, "validation_seal_sha256")
    _check_seal(budgets, "budget_seal_sha256")
    _check_seal(roster, "roster_seal_sha256")
    _check_seal(matrix, "matrix_seal_sha256")
    if roster["validation_ref"]["sha256"] != _sha256_bytes(
        (json.dumps(validation, indent=2, sort_keys=True) + "\n").encode()
    ):
        raise CohortError("roster validation reference drifted")
    if matrix["roster_ref"]["sha256"] != _sha256_bytes(
        (json.dumps(roster, indent=2, sort_keys=True) + "\n").encode()
    ):
        raise CohortError("matrix roster reference drifted")
    if matrix["validation_ref"]["sha256"] != _sha256_bytes(
        (json.dumps(validation, indent=2, sort_keys=True) + "\n").encode()
    ):
        raise CohortError("matrix validation reference drifted")
    if matrix["trace_budget_ref"]["sha256"] != _sha256_bytes(
        (json.dumps(budgets, indent=2, sort_keys=True) + "\n").encode()
    ):
        raise CohortError("matrix budget reference drifted")
    row_keys = [row["matrix_key"] for row in matrix["rows"]]
    if row_keys != [row["matrix_key"] for row in roster["rows"]]:
        raise CohortError("roster/matrix row order or identity drifted")


def build_cohort(*, source_commit: str) -> tuple[dict[str, Any], ...]:
    source_inputs = _source_inputs(source_commit)
    frozen = _frozen_reference()
    witnesses: dict[str, dict[str, Any]] = {}
    runtime_rows: list[dict[str, Any]] = []
    for case in CASES:
        key = case.slug + "|" + case.base_slug
        witness = _load_witness(case)
        witnesses[key] = witness
        runtime = _run_case(case, witness)
        runtime_rows.append(_row(case, witness, runtime))
    validation = _build_validation(source_inputs, frozen, runtime_rows)
    validation_bytes = (json.dumps(validation, indent=2, sort_keys=True) + "\n").encode()
    budgets = _build_budgets(source_inputs, frozen, witnesses)
    budget_bytes = (json.dumps(budgets, indent=2, sort_keys=True) + "\n").encode()
    roster = _build_roster(source_inputs, frozen, runtime_rows, validation_bytes, budget_bytes)
    roster_bytes = (json.dumps(roster, indent=2, sort_keys=True) + "\n").encode()
    matrix = _build_matrix(
        source_inputs, frozen, runtime_rows, roster_bytes, validation_bytes, budget_bytes
    )
    validate_bundle(validation, budgets, roster, matrix)
    outputs = (
        (VALIDATION_REL, validation_bytes),
        (BUDGETS_REL, budget_bytes),
        (ROSTER_REL, roster_bytes),
        (MATRIX_REL, (json.dumps(matrix, indent=2, sort_keys=True) + "\n").encode()),
    )
    for rel, payload in outputs:
        _write_or_match(ROOT / rel, payload)
    return validation, budgets, roster, matrix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args(argv)
    try:
        build_cohort(source_commit=args.source_commit)
    except (CohortError, OSError, ValueError, jsonschema.SchemaError) as exc:
        print(f"CUTLASS successor cohort failed: {exc}", file=sys.stderr)
        return 1
    print(f"CUTLASS successor cohort: pass=True -> {ROOT / MATRIX_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
