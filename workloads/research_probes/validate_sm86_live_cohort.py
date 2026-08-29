#!/usr/bin/env python3
"""Validate the durable six-cell SM86 live-campaign handoff.

The validator consumes the repository-relative manifest and materialized raw
evidence bundle.  It keeps NCU observations on the comparison side of the
boundary and independently replays the NVBit trace through the semantic
conservation checks used by the retained-corpus validator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workloads.research_probes.semantic_replay_validation import (
    validate_replay_document,
)
from workloads.research_probes import replay_resume as replay_resume
from workloads.research_probes.trace_invariants import derive_trace_invariants


DEFAULT_MANIFEST = (
    REPO_ROOT
    / "results/captures/sm86_simulator_live_baseline_20260829_v2/campaign_manifest.json"
)
STAGES = ("correctness", "ncu", "nvbit", "static_control", "simulator")
_CURRENT_TIMEOUTS = {
    "correctness": 180,
    "ncu": 900,
    "nvbit": 900,
    "static_control": 300,
    "simulator": 900,
}

_CORRECTNESS_INJECTION_VARIABLES = frozenset(
    {
        "BASH_ENV",
        "CUDA_INJECTION32_PATH",
        "CUDA_INJECTION64_PATH",
        "CUDA_VISIBLE_DEVICES",
        "DYNAMIC_KERNEL_LIMIT_END",
        "DYNAMIC_KERNEL_LIMIT_START",
        "ENHANCED_TRACER_MAX_PARSER_INPUT_KB",
        "ENHANCED_TRACER_PER_CUBIN_TIMEOUT_S",
        "LD_AUDIT",
        "LD_PRELOAD",
        "NVBIT_CONFIG_FILE",
        "NVBIT_PB_PREFIX",
        "NVBIT_TOOL",
        "NOBANNER",
        "SKIP_ENHANCED_TRACER",
        "TERMINATE_UPON_LIMIT",
        "TOOL_VERBOSE",
        "TRACES_FOLDER",
        "USER_DEFINED_FOLDERS",
        "NVBIT_TRACE_FOLDER",
    }
)
_EXPECTED_COHORT_IDS = frozenset(
    {
        "cp_async_canary",
        "gemm_hmma_bf16_sm80",
        "gemm_hmma_fp16_sm80",
        "gemm_imma_int8_sm80",
        "gemm_sparse_2to4_bf16_sm80",
        "flash_attn_fwd_bf16_sm80",
    }
)
_EXPECTED_CELL_SPECS: dict[str, dict[str, Any]] = {
    "cp_async_canary": {
        "workload": "async_barrier_overlap",
        "binary": "workloads/research_probes/async_barrier_overlap/async_barrier_overlap",
        "trace_binary": "workloads/research_probes/async_barrier_overlap/async_barrier_overlap_trace",
        "grid": [1, 1, 1],
        "block": [128, 1, 1],
    },
    "gemm_hmma_bf16_sm80": {
        "workload": "gemm_hmma_bf16_sm80",
        "binary": "workloads/portable_lcd/gemm_hmma_bf16_sm80/gemm_hmma_bf16_sm80_run",
        "grid": [16, 32, 1],
        "block": [32, 1, 1],
    },
    "gemm_hmma_fp16_sm80": {
        "workload": "gemm_hmma_fp16_sm80",
        "binary": "workloads/portable_lcd/gemm_hmma_fp16_sm80/gemm_hmma_fp16_sm80_run",
        "grid": [16, 32, 1],
        "block": [32, 1, 1],
    },
    "gemm_imma_int8_sm80": {
        "workload": "gemm_imma_int8_sm80",
        "binary": "workloads/portable_lcd/gemm_imma_int8_sm80/gemm_imma_int8_sm80_run",
        "grid": [16, 32, 1],
        "block": [32, 1, 1],
    },
    "gemm_sparse_2to4_bf16_sm80": {
        "workload": "gemm_sparse_2to4_bf16_sm80",
        "binary": "workloads/portable_lcd/gemm_sparse_2to4_bf16_sm80/gemm_sparse_2to4_bf16_sm80_run",
        "grid": [8, 16, 1],
        "block": [32, 1, 1],
    },
    "flash_attn_fwd_bf16_sm80": {
        "workload": "flash_attn_fwd_bf16_sm80",
        "binary": "workloads/portable_lcd/flash_attn_fwd_bf16_sm80/flash_attn_fwd_bf16_sm80_run",
        "grid": [4, 4, 1],
        "block": [32, 1, 1],
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _last_json_object(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    decoder = json.JSONDecoder()
    for offset in reversed([index for index, value in enumerate(text) if value == "{"]):
        try:
            value, _end = decoder.raw_decode(text[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(f"{path}: no JSON object in captured stdout")


def _kernel_row(path: Path) -> Mapping[str, str]:
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    for row in rows:
        if row.get("Kernel Name"):
            return row
    raise ValueError(f"{path}: NCU CSV has no kernel row")


def _assert_file(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label}: missing repository-relative path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label}: path escapes manifest root: {relative}") from exc
    if not path.is_file():
        raise ValueError(f"{label}: artifact is missing: {relative}")
    return path


def _campaign_argv_matches(token: object, relative: object) -> bool:
    """Accept campaign-relative argv tokens and their normalized repo form."""

    if not isinstance(token, str) or not isinstance(relative, str):
        return False
    if token == relative:
        return True
    return token == (
        "results/captures/sm86_simulator_live_baseline_20260829_v2/" + relative
    )


def _required_int(record: Mapping[str, Any], key: str, label: str) -> int:
    value = record.get(key)
    try:
        parsed = int(str(value).replace(",", ""))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: integer field {key} is missing or malformed") from exc
    return parsed


def _check_timeout(
    stage: Mapping[str, Any], label: str, defaults: Mapping[str, Any]
) -> int:
    timeout = stage.get("timeout_seconds", defaults.get(label.rsplit("/", 1)[-1]))
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise ValueError(f"{label}: finite timeout_seconds is required")
    return timeout


def _validate_cell_identity(cell: Mapping[str, Any]) -> None:
    cell_id = cell.get("id")
    if not isinstance(cell_id, str) or cell_id not in _EXPECTED_CELL_SPECS:
        raise ValueError(f"unknown SM86 cohort cell id: {cell_id!r}")
    spec = _EXPECTED_CELL_SPECS[cell_id]
    if cell.get("workload") != spec["workload"] or cell.get("architecture") != "sm_86":
        raise ValueError(f"{cell_id}: workload or architecture identity differs")
    binary = cell.get("binary")
    if not isinstance(binary, Mapping):
        raise ValueError(f"{cell_id}: binary identity is missing")
    if binary.get("path") != spec["binary"]:
        raise ValueError(f"{cell_id}: base binary identity differs")
    expected_trace = spec.get("trace_binary", spec["binary"])
    if binary.get("trace_path", binary.get("path")) != expected_trace:
        raise ValueError(f"{cell_id}: traced binary identity differs")
    if cell.get("expected_grid") != spec["grid"] or cell.get("expected_block") != spec["block"]:
        raise ValueError(f"{cell_id}: trusted geometry differs from the cohort contract")


def _validate_correctness_stage(
    root: Path,
    cell: Mapping[str, Any],
    binary: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate a clean base-binary correctness invocation and its output."""

    cell_id = str(cell["id"])
    binary_path = binary.get("path")
    argv = stage.get("argv")
    if not isinstance(binary_path, str) or not isinstance(argv, list) or any(
        not isinstance(item, str) for item in argv
    ):
        raise ValueError(f"{cell_id}: correctness binary/argv identity is malformed")
    output_dir = None
    expected_repo_relative: list[str] | None = None
    if cell_id == "cp_async_canary":
        expected = [binary_path, "c11", "l1", "0", "4", "1"]
    else:
        stdout_path = stage.get("stdout")
        if not isinstance(stdout_path, str):
            raise ValueError(f"{cell_id}: correctness stdout path is missing")
        output_dir = str(Path(stdout_path).parent)
        expected = [
            binary_path,
            "--correctness_shape",
            "--seed",
            "42",
            "--warmup",
            "1",
            "--reps",
            "1",
            "--output_dir",
            output_dir,
        ]
        expected_repo_relative = list(expected)
        expected_repo_relative[-1] = (
            "results/captures/sm86_simulator_live_baseline_20260829_v2/" + output_dir
        )
    if argv != expected and argv != expected_repo_relative:
        raise ValueError(f"{cell_id}: correctness argv does not match its expected contract")
    stdout = _assert_file(root, stage.get("stdout"), f"{cell_id}/correctness.stdout")
    stderr = _assert_file(root, stage.get("stderr"), f"{cell_id}/correctness.stderr")
    for key, path in (("stdout_sha256", stdout), ("stderr_sha256", stderr)):
        expected_hash = stage.get(key)
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"{cell_id}: correctness {key} is missing")
        if _sha256(path) != expected_hash:
            raise ValueError(f"{cell_id}: correctness {key} does not match artifact")

    environment = stage.get("environment")
    if not isinstance(environment, Mapping) or environment.get("mode") != "clean_base_binary":
        raise ValueError(f"{cell_id}: correctness environment is not clean_base_binary")
    environment_record = _assert_file(
        root, environment.get("record"), f"{cell_id}/correctness.environment"
    )
    environment_hash = environment.get("record_sha256")
    if not isinstance(environment_hash, str) or len(environment_hash) != 64:
        raise ValueError(f"{cell_id}: correctness environment hash is missing")
    if _sha256(environment_record) != environment_hash:
        raise ValueError(f"{cell_id}: correctness environment hash does not match artifact")
    environment_doc = json.load(open(environment_record))
    if not isinstance(environment_doc, Mapping) or environment_doc.get("schema_version") != 1:
        raise ValueError(f"{cell_id}: correctness environment record schema is unsupported")
    if environment_doc.get("argv") != argv or environment_doc.get("effective_argv") != argv:
        raise ValueError(f"{cell_id}: correctness environment argv differs from manifest")
    if environment_doc.get("return_code") != 0 or environment_doc.get("output_contains_instrumentation") is not False:
        raise ValueError(f"{cell_id}: correctness environment record is not a clean successful run")
    effective_environment = environment_doc.get("environment")
    if not isinstance(effective_environment, Mapping):
        raise ValueError(f"{cell_id}: effective correctness environment is missing")
    expected_environment_keys = {"HOME", "LANG", "LC_ALL", "PATH", "LD_LIBRARY_PATH"}
    if set(effective_environment) != expected_environment_keys:
        raise ValueError(f"{cell_id}: correctness environment is not the exact minimal environment")
    if effective_environment.get("LANG") != "C" or effective_environment.get("LC_ALL") != "C":
        raise ValueError(f"{cell_id}: correctness locale is not deterministic")
    leaked = sorted(_CORRECTNESS_INJECTION_VARIABLES.intersection(effective_environment))
    if leaked:
        raise ValueError(f"{cell_id}: profiler variables leaked into correctness environment: {leaked}")
    cleared = environment_doc.get("cleared_instrumentation_variables")
    if not isinstance(cleared, list) or not _CORRECTNESS_INJECTION_VARIABLES.issubset(cleared):
        raise ValueError(f"{cell_id}: correctness environment does not record cleared profiler variables")

    result = _last_json_object(stdout)
    if cell_id == "cp_async_canary":
        expected_fields = {
            "protocol": "c11_wait_barrier",
            "locality": "l1_reuse",
            "gap_cycles": 0,
            "warps": 4,
            "groups": 16384,
            "copies_per_group": 8,
            "payload_bytes_per_copy": 512,
            "region_bytes": 512,
            "source_stride": 0,
        }
        for key, expected_value in expected_fields.items():
            if result.get(key) != expected_value:
                raise ValueError(f"{cell_id}: correctness result field {key} differs")
        if result.get("correct") is not True:
            raise ValueError(f"{cell_id}: correctness result is not true")
    else:
        if result.get("slug") != cell.get("workload"):
            raise ValueError(f"{cell_id}: correctness result slug differs from workload")
        for key, expected_value in (
            ("mode", "correctness"),
            ("shape_point", "correctness_00"),
            ("reps", 1),
            ("warmup", 1),
        ):
            if result.get(key) != expected_value:
                raise ValueError(f"{cell_id}: correctness result field {key} differs")
        if result.get("correctness_ok") is not True:
            raise ValueError(f"{cell_id}: correctness result is not true")
        result_manifest = _assert_file(
            root, stage.get("manifest"), f"{cell_id}/correctness.manifest"
        )
        result_manifest_doc = json.load(open(result_manifest))
        if not isinstance(result_manifest_doc, Mapping):
            raise ValueError(f"{cell_id}: correctness manifest is malformed")
        if result_manifest_doc.get("slug") != cell.get("workload"):
            raise ValueError(f"{cell_id}: correctness manifest slug differs")
        if result_manifest_doc.get("binary_sha256") != binary.get("sha256"):
            raise ValueError(f"{cell_id}: correctness manifest binary identity differs")
        if (
            result_manifest_doc.get("compute_capability_major") != 8
            or result_manifest_doc.get("compute_capability_minor") != 6
        ):
            raise ValueError(f"{cell_id}: correctness manifest compute capability differs")
    return result


def _validate_nvbit_stage(
    root: Path,
    cell: Mapping[str, Any],
    binary: Mapping[str, Any],
    stage: Mapping[str, Any],
    tracer_sha256: str | None = None,
) -> tuple[Path, dict[str, str], Mapping[str, Any], Mapping[str, Any]]:
    """Validate the trace root and every auxiliary file as one capture unit."""

    cell_id = str(cell["id"])
    trace_binary = binary.get("trace_path", binary.get("path"))
    trace_binary_hash = binary.get("trace_sha256", binary.get("sha256"))
    argv = stage.get("argv")
    environment = stage.get("env")
    output = stage.get("output")
    if not isinstance(argv, list) or any(not isinstance(item, str) for item in argv):
        raise ValueError(f"{cell_id}: NVBit argv is malformed")
    if not isinstance(environment, Mapping) or not isinstance(output, str) or not output:
        raise ValueError(f"{cell_id}: NVBit environment/output binding is missing")
    if not isinstance(trace_binary, str) or not isinstance(trace_binary_hash, str):
        raise ValueError(f"{cell_id}: traced binary identity is malformed")
    if stage.get("trace_binary_path") != trace_binary or stage.get("trace_binary_sha256") != trace_binary_hash:
        raise ValueError(f"{cell_id}: NVBit traced-binary binding differs from the cell binary")
    expected_active = "0" if cell_id == "cp_async_canary" else "1"
    expected_environment = {
        "CUDA_INJECTION64_PATH": "tools/nvbit_tracer/tracer_tool/tracer_tool.so",
        "TRACES_FOLDER": output,
        "USER_DEFINED_FOLDERS": "1",
        "ACTIVE_FROM_START": expected_active,
    }
    for key, expected in expected_environment.items():
        if environment.get(key) != expected:
            raise ValueError(f"{cell_id}: NVBit environment {key} is not bound to this cell")
    if not isinstance(environment.get("CUDA_INJECTION64_PATH"), str):
        raise ValueError(f"{cell_id}: NVBit tracer path is missing")
    tracer_path = _assert_file(
        REPO_ROOT,
        environment["CUDA_INJECTION64_PATH"],
        f"{cell_id}.nvbit_tracer",
    )
    actual_tracer_hash = _sha256(tracer_path)
    if len(actual_tracer_hash) != 64 or (tracer_sha256 is not None and actual_tracer_hash != tracer_sha256):
        raise ValueError(f"{cell_id}: NVBit tracer hash could not be read")
    if stage.get("tracer_sha256") != actual_tracer_hash:
        raise ValueError(f"{cell_id}: NVBit stage tracer hash is not recorded accurately")
    if trace_binary not in argv:
        raise ValueError(f"{cell_id}: NVBit argv is not bound to the traced binary")

    trace = _assert_file(root, output + "/dynamic_trace.pb", f"{cell_id}/nvbit.dynamic_trace").parent
    expected_stats = output + "/stats.csv"
    expected_mapping = output + "/extra_info/traced_kernel_modules.json"
    expected_binding = output + "/campaign_binding.json"
    for key, expected in (
        ("stats", expected_stats),
        ("mapping", expected_mapping),
        ("binding", expected_binding),
    ):
        if stage.get(key) != expected:
            raise ValueError(f"{cell_id}: NVBit {key} path is not under its trace root")
    stats = _assert_file(root, stage.get("stats"), f"{cell_id}/nvbit.stats")
    mapping_path = _assert_file(root, stage.get("mapping"), f"{cell_id}/nvbit.mapping")
    binding_path = _assert_file(root, stage.get("binding"), f"{cell_id}/nvbit.binding")
    binding_hash = stage.get("binding_sha256")
    if not isinstance(binding_hash, str) or len(binding_hash) != 64 or _sha256(binding_path) != binding_hash:
        raise ValueError(f"{cell_id}: NVBit binding sidecar hash does not match")
    binding = json.load(open(binding_path))
    if not isinstance(binding, Mapping) or binding.get("schema_version") != 1:
        raise ValueError(f"{cell_id}: NVBit binding sidecar schema is unsupported")
    if binding.get("cell") != cell_id or binding.get("trace_root") != output:
        raise ValueError(f"{cell_id}: NVBit binding sidecar root identity differs")
    if binding.get("argv") != argv or binding.get("environment") != dict(environment):
        raise ValueError(f"{cell_id}: NVBit binding sidecar command/environment differs")
    bound_binary = binding.get("trace_binary")
    if not isinstance(bound_binary, Mapping) or bound_binary.get("path") != trace_binary or bound_binary.get("sha256") != trace_binary_hash:
        raise ValueError(f"{cell_id}: NVBit binding sidecar binary identity differs")
    paths = binding.get("paths")
    expected_execution = output + "/extra_info/enhanced_execution_info.json"
    if not isinstance(paths, Mapping) or paths.get("dynamic_trace") != output + "/dynamic_trace.pb" or paths.get("stats") != expected_stats or paths.get("mapping") != expected_mapping or paths.get("execution_info") != expected_execution:
        raise ValueError(f"{cell_id}: NVBit binding sidecar auxiliary paths differ")
    hashes = binding.get("hashes")
    if not isinstance(hashes, Mapping):
        raise ValueError(f"{cell_id}: NVBit binding sidecar hashes are missing")
    expected_hashes = {
        "dynamic_trace_sha256": _sha256(trace / "dynamic_trace.pb"),
        "stats_sha256": _sha256(stats),
        "mapping_sha256": _sha256(mapping_path),
    }
    for key, expected in expected_hashes.items():
        if hashes.get(key) != expected:
            raise ValueError(f"{cell_id}: NVBit binding hash differs for {key}")
    observed = stage.get("observed")
    if not isinstance(observed, Mapping) or observed.get("dynamic_trace_sha256") != expected_hashes["dynamic_trace_sha256"]:
        raise ValueError(f"{cell_id}: NVBit dynamic trace hash is not recorded accurately")

    stat_rows = list(csv.DictReader(stats.open(encoding="utf-8", newline="")))
    if len(stat_rows) != 1:
        raise ValueError(f"{cell_id}: expected exactly one NVBit stats row")
    stat = {str(key).strip(): str(value).strip() for key, value in stat_rows[0].items()}
    if stat.get("total_insts") != stat.get("total_reported_insts"):
        raise ValueError(f"{cell_id}: NVBit total/report instruction mismatch")
    mapping_doc = json.load(open(mapping_path))
    if not isinstance(mapping_doc, Mapping):
        raise ValueError(f"{cell_id}: NVBit mapping is malformed")
    selection = mapping_doc.get("selection")
    launches = mapping_doc.get("launches")
    module_identity = mapping_doc.get("module_identity")
    if not isinstance(module_identity, Mapping) or module_identity.get("stable_key") != "sha256":
        raise ValueError(f"{cell_id}: NVBit mapping module identity contract is missing")
    if not isinstance(selection, Mapping) or selection.get("accepted_active_launches") != 1 or selection.get("exact_contract_satisfied") is not True:
        raise ValueError(f"{cell_id}: NVBit launch selection is not exact")
    if not isinstance(launches, list) or len(launches) != 1:
        raise ValueError(f"{cell_id}: NVBit mapping must contain exactly one launch")
    if selection.get("accepted_channel_records") != _required_int(stat, "total_reported_insts", f"{cell_id}/nvbit.stats"):
        raise ValueError(f"{cell_id}: NVBit mapping/stat record counts differ")
    mapping_launch = launches[0]
    if not isinstance(mapping_launch, Mapping) or mapping_launch.get("accepted_launch_index") != 0 or mapping_launch.get("trace_launch_id") != 1 or mapping_launch.get("device_id") != 0 or mapping_launch.get("stream_id") != 0:
        raise ValueError(f"{cell_id}: NVBit mapping launch identity is malformed")
    module = mapping_launch.get("module")
    if not isinstance(module, Mapping) or module.get("status") not in {"no_stable_static_module", "stable_static_module"}:
        raise ValueError(f"{cell_id}: NVBit mapping module status is malformed")
    if module.get("status") == "stable_static_module" and (
        module.get("sha256") != trace_binary_hash
    ):
        raise ValueError(f"{cell_id}: stable NVBit module hash differs from traced binary")
    if mapping_launch.get("kernel_name") != stat.get("kernel mangled name"):
        raise ValueError(f"{cell_id}: NVBit mapping kernel identity differs from stats")
    execution_path = _assert_file(root, expected_execution, f"{cell_id}/nvbit.execution_info")
    execution = json.load(open(execution_path))
    if hashes.get("execution_info_sha256") != _sha256(execution_path):
        raise ValueError(f"{cell_id}: NVBit execution metadata hash differs")
    kernels = execution.get("kernels") if isinstance(execution, Mapping) else None
    if not isinstance(kernels, list) or len(kernels) != 1 or not isinstance(kernels[0], Mapping):
        raise ValueError(f"{cell_id}: NVBit execution metadata must contain one kernel")
    execution_kernel = kernels[0]
    execution_name = execution_kernel.get("kernel_name")
    stats_name = stat.get("kernel mangled name")
    if not isinstance(execution_name, str) or not isinstance(stats_name, str) or not (execution_name == stats_name or execution_name.startswith(stats_name + "___")):
        raise ValueError(f"{cell_id}: NVBit mapping kernel name is not bound to execution metadata")
    geometry = binding.get("geometry")
    if not isinstance(geometry, Mapping) or geometry.get("grid") != cell.get("expected_grid") or geometry.get("block") != cell.get("expected_block"):
        raise ValueError(f"{cell_id}: NVBit binding geometry differs from the cohort contract")
    if _required_int(stat, "device_id", f"{cell_id}/nvbit.stats") != mapping_launch.get("device_id") or _required_int(stat, "stream_id", f"{cell_id}/nvbit.stats") != mapping_launch.get("stream_id"):
        raise ValueError(f"{cell_id}: NVBit mapping and stats device/stream identities differ")
    return trace, stat, mapping_doc, execution


def _validate_host_preflight(
    root: Path,
    host_record_path: Path,
    repository: Mapping[str, Any],
    manifest_host: Mapping[str, Any],
    manifest_tools: Mapping[str, Any],
) -> None:
    try:
        host = json.load(open(host_record_path))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"host preflight record is not valid JSON: {host_record_path}") from exc
    if not isinstance(host, Mapping) or host.get("schema_version") != 2:
        raise ValueError("host preflight record schema is unsupported")
    binding = host.get("revision_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("host preflight revision binding is missing")
    for key in (
        "capture_revision",
        "evidence_source_revision",
        "candidate_revision",
        "correction_revision",
    ):
        value = binding.get(key)
        if not isinstance(value, str) or len(value) != 40:
            raise ValueError(f"host preflight revision binding {key} is malformed")
    if binding.get("capture_revision") != repository.get("capture_revision"):
        raise ValueError("host preflight capture revision differs from manifest")
    if binding.get("candidate_revision") != repository.get("candidate_revision"):
        raise ValueError("host preflight candidate revision differs from manifest")
    if binding.get("correction_revision") != repository.get("correction_revision"):
        raise ValueError("host preflight correction revision differs from manifest")
    simulator_identity = repository.get("simulator_binary")
    if not isinstance(simulator_identity, Mapping):
        raise ValueError("manifest simulator binary identity is malformed")
    expected_hashes = {
        "simulator_binary_sha256": simulator_identity.get("sha256"),
        "tracer_dso_sha256": repository.get("tracer_dso_sha256"),
        "architecture_card_sha256": repository.get("architecture_card_sha256"),
        "pchase_card_sha256": repository.get("pchase_card_sha256"),
    }
    for key, expected in expected_hashes.items():
        if not isinstance(expected, str) or binding.get(key) != expected:
            raise ValueError(f"host preflight {key} is not bound to the manifest input")
    gpu = host.get("gpu")
    if not isinstance(gpu, Mapping):
        raise ValueError("host preflight GPU record is missing")
    for key, expected in (
        ("name", manifest_host.get("device")),
        ("compute_capability", manifest_host.get("compute_capability")),
        ("driver_version", manifest_host.get("driver")),
    ):
        if gpu.get(key) != expected:
            raise ValueError(f"host preflight GPU field {key} differs from manifest")
    toolchain = host.get("toolchain")
    if not isinstance(toolchain, Mapping):
        raise ValueError("host preflight toolchain record is missing")
    expected_tool_values = {
        "nvcc": str(manifest_tools.get("cuda_nvcc", "")).split("(")[-1].rstrip(")"),
        "ncu": str(manifest_tools.get("ncu", "")).split("(")[-1].rstrip(")"),
        "nvbit": manifest_tools.get("nvbit", {}).get("version")
        if isinstance(manifest_tools.get("nvbit"), Mapping)
        else None,
        "protobuf_runtime": str(manifest_tools.get("protobuf_runtime", "")).split()[0],
    }
    for key, expected in expected_tool_values.items():
        actual = str(toolchain.get(key, "")).split()[0]
        if not isinstance(expected, str) or not expected or actual != expected:
            raise ValueError(f"host preflight toolchain field {key} differs from manifest")
    commands = host.get("commands")
    if not isinstance(commands, list):
        raise ValueError("host preflight command records are missing")
    required_names = {
        "gpu_query",
        "gpu_list",
        "nvcc_version",
        "ncu_version",
        "rust_version",
        "cargo_version",
        "sudo_noninteractive",
        "git_revision",
        "input_hashes",
    }
    expected_argv = {
        "gpu_query": ["nvidia-smi", "--query-gpu=index,name,uuid,compute_cap,driver_version,clocks.current.graphics,clocks.current.memory,temperature.gpu", "--format=csv,noheader,nounits"],
        "gpu_list": ["nvidia-smi", "-L"],
        "nvcc_version": ["/opt/cuda/bin/nvcc", "--version"],
        "ncu_version": ["/usr/local/cuda-13.2/nsight-compute-2026.1.0/ncu", "--version"],
        "rust_version": ["rustc", "--version"],
        "cargo_version": ["cargo", "--version"],
        "sudo_noninteractive": ["sudo", "-n", "true"],
        "git_revision": ["git", "rev-parse", "HEAD"],
        "input_hashes": ["sha256sum", "sim/target/release/sim-run", "tools/nvbit_tracer/tracer_tool/tracer_tool.so", "arch_profiles/cards/sm_86.json", "data/pchase/sm_86_static_card.json"],
    }
    allowed_nonzero = {"sudo_noninteractive"}
    seen: set[str] = set()
    for command in commands:
        if not isinstance(command, Mapping):
            raise ValueError("host preflight command record is malformed")
        name = command.get("name")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("host preflight command names must be unique")
        seen.add(name)
        argv = command.get("argv")
        if not isinstance(argv, list) or not argv or any(not isinstance(item, str) for item in argv):
            raise ValueError(f"host preflight {name}: argv is missing")
        if name in expected_argv and argv != expected_argv[name]:
            raise ValueError(f"host preflight {name}: argv differs from the declared command")
        timeout = command.get("timeout_seconds")
        if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
            raise ValueError(f"host preflight {name}: finite timeout is missing")
        return_code = command.get("return_code")
        if isinstance(return_code, bool) or not isinstance(return_code, int):
            raise ValueError(f"host preflight {name}: return code is missing")
        if return_code != 0:
            limitation = command.get("limitation")
            if name not in allowed_nonzero or not isinstance(limitation, str) or not limitation.strip():
                raise ValueError(f"host preflight {name}: failed required command lacks a typed limitation")
        stdout_path = _assert_file(root, command.get("output_path"), f"host preflight {name}.stdout")
        stderr_path = _assert_file(root, command.get("stderr_path"), f"host preflight {name}.stderr")
        if return_code == 0 and not stdout_path.read_text(encoding="utf-8", errors="replace").strip():
            raise ValueError(f"host preflight {name}: successful command has no stdout record")
        if name == "sudo_noninteractive" and return_code != 1:
            raise ValueError("host preflight sudo probe must record the observed unavailable permission")
    if not required_names.issubset(seen):
        raise ValueError(f"host preflight command set is incomplete: {sorted(required_names - seen)}")
    git_command = next(command for command in commands if command.get("name") == "git_revision")
    revision_output = _assert_file(root, git_command["output_path"], "host preflight git revision")
    if revision_output.read_text(encoding="utf-8").strip() != repository.get("candidate_revision"):
        raise ValueError("host preflight git revision does not identify the candidate")
    output_text = {
        command["name"]: _assert_file(root, command["output_path"], f"host preflight {command['name']}.stdout").read_text(
            encoding="utf-8", errors="replace"
        )
        for command in commands
    }
    expected_markers = {
        "gpu_query": [manifest_host.get("device"), manifest_host.get("compute_capability"), manifest_host.get("driver")],
        "gpu_list": [manifest_host.get("device")],
        "nvcc_version": ["V13.3.73"],
        "ncu_version": ["Version 2026.1.0.0"],
        "rust_version": ["rustc 1.96.1"],
        "cargo_version": ["cargo 1.96.1"],
        "input_hashes": [binding.get("simulator_binary_sha256"), binding.get("tracer_dso_sha256"), binding.get("architecture_card_sha256"), binding.get("pchase_card_sha256")],
    }
    for name, markers in expected_markers.items():
        if any(not isinstance(marker, str) or marker not in output_text.get(name, "") for marker in markers):
            raise ValueError(f"host preflight {name}: recorded output does not contain the bound value")


def _validate_cell(
    root: Path,
    cell: Mapping[str, Any],
    defaults: Mapping[str, Any],
    tracer_sha256: str | None = None,
) -> dict[str, Any]:
    cell_id = cell.get("id")
    if not isinstance(cell_id, str) or not cell_id:
        raise ValueError("cell has no id")
    _validate_cell_identity(cell)
    binary = cell.get("binary")
    if not isinstance(binary, Mapping):
        raise ValueError(f"{cell_id}: binary record is missing")
    binary_path = _assert_file(REPO_ROOT, binary.get("path"), f"{cell_id}.binary")
    if _sha256(binary_path) != binary.get("sha256"):
        raise ValueError(f"{cell_id}: binary SHA-256 does not match manifest")
    trace_binary_path = _assert_file(
        REPO_ROOT,
        binary.get("trace_path", binary.get("path")),
        f"{cell_id}.trace_binary",
    )
    trace_binary_hash = binary.get("trace_sha256", binary.get("sha256"))
    if _sha256(trace_binary_path) != trace_binary_hash:
        raise ValueError(f"{cell_id}: traced-binary SHA-256 does not match manifest")
    stages = cell.get("stages")
    if not isinstance(stages, Mapping) or set(stages) != set(STAGES):
        raise ValueError(f"{cell_id}: stage set is incomplete")

    row: dict[str, Any] = {"cell": cell_id, "checks": {}}
    # The manifest's stage-timeout table is part of the reproducibility
    # contract; an individual stage may override it only with another finite
    # value.
    for stage_name in STAGES:
        stage = stages[stage_name]
        if not isinstance(stage, Mapping):
            raise ValueError(f"{cell_id}/{stage_name}: stage record is malformed")
        _check_timeout(stage, f"{cell_id}/{stage_name}", defaults)
        if stage.get("exit_code") != 0:
            raise ValueError(f"{cell_id}/{stage_name}: exit_code is not zero")
        for key in ("stdout", "stderr", "csv", "stats", "mapping", "control"):
            if key in stage:
                _assert_file(root, stage[key], f"{cell_id}/{stage_name}.{key}")

    correctness = stages["correctness"]
    correctness_result = _validate_correctness_stage(
        root, cell, binary, correctness
    )
    row["checks"]["correctness"] = True

    ncu = stages["ncu"]
    ncu_argv = ncu.get("argv")
    if not isinstance(ncu_argv, list) or binary.get("path") not in ncu_argv:
        raise ValueError(f"{cell_id}: NCU argv is not bound to the cell binary")
    ncu_csv = _assert_file(
        root,
        ncu["csv"],
        f"{cell_id}/ncu.csv",
    )
    ncu_row = _kernel_row(ncu_csv)
    ncu_observed = ncu.get("observed")
    if not isinstance(ncu_observed, Mapping):
        raise ValueError(f"{cell_id}: NCU observed metrics are missing")
    for metric, expected in ncu_observed.items():
        if metric not in ncu_row:
            raise ValueError(f"{cell_id}: NCU metric is absent: {metric}")
        try:
            actual = float(str(ncu_row[metric]).replace(",", ""))
        except ValueError as exc:
            raise ValueError(f"{cell_id}: NCU metric is not numeric: {metric}") from exc
        if abs(actual - float(expected)) > 0.5:
            raise ValueError(f"{cell_id}: NCU metric differs: {metric}")
    row["checks"]["ncu"] = True

    nvbit = stages["nvbit"]
    trace, stat, mapping, execution = _validate_nvbit_stage(
        root, cell, binary, nvbit, tracer_sha256
    )
    trace_binary = binary.get("trace_path", binary.get("path"))
    trace_binary_hash = binary.get("trace_sha256", binary.get("sha256"))
    row["checks"]["nvbit"] = True

    static = stages["static_control"]
    control = _assert_file(root, static["control"], f"{cell_id}/static.control")
    control_hash = static.get("control_sha256")
    if not isinstance(control_hash, str) or len(control_hash) != 64:
        raise ValueError(f"{cell_id}: static control hash is missing")
    if _sha256(control) != control_hash:
        raise ValueError(f"{cell_id}: static control hash does not match artifact")
    control_doc = json.load(open(control))
    decoded = sum(
        1
        for entries in control_doc.get("functions", {}).values()
        for entry in entries
        if entry.get("status") == "decoded"
    )
    if decoded <= 0:
        raise ValueError(f"{cell_id}: static control sidecar has no decoded instructions")
    static_binary = static.get("binary_path")
    static_hash = static.get("binary_sha256")
    if static_binary != trace_binary:
        raise ValueError(f"{cell_id}: static binary is not the traced capture binary")
    if not isinstance(static_hash, str) or len(static_hash) != 64:
        raise ValueError(f"{cell_id}: static binary identity is missing")
    static_binary_path = _assert_file(
        REPO_ROOT, static_binary, f"{cell_id}.static_binary"
    )
    if _sha256(static_binary_path) != static_hash:
        raise ValueError(f"{cell_id}: static binary hash mismatch")
    if static_hash != trace_binary_hash:
        raise ValueError(f"{cell_id}: static and traced binary identities differ")
    if control_doc.get("binary") != static_binary:
        raise ValueError(f"{cell_id}: control sidecar binary identity differs")
    row["checks"]["static_control"] = True

    simulator = stages["simulator"]
    simulator_argv = simulator.get("argv")
    if not isinstance(simulator_argv, list):
        raise ValueError(f"{cell_id}: simulator argv is missing")
    if not any(_campaign_argv_matches(token, nvbit.get("output")) for token in simulator_argv) or not any(
        _campaign_argv_matches(token, static.get("control")) for token in simulator_argv
    ):
        raise ValueError(f"{cell_id}: simulator argv is not bound to trace/control inputs")
    for option, expected in (("--trace", nvbit.get("output")), ("--control", static.get("control"))):
        try:
            option_index = simulator_argv.index(option)
        except ValueError as exc:
            raise ValueError(f"{cell_id}: simulator argv is missing {option}") from exc
        if option_index + 1 >= len(simulator_argv) or not _campaign_argv_matches(
            simulator_argv[option_index + 1], expected
        ):
            raise ValueError(f"{cell_id}: simulator {option} input is misbound")
    outputs = simulator.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 2:
        raise ValueError(f"{cell_id}: simulator twin outputs are missing")
    first = _assert_file(root, outputs[0], f"{cell_id}/simulator.replay-1")
    second = _assert_file(root, outputs[1], f"{cell_id}/simulator.replay-2")
    expected_output_prefix = f"evidence/raw/simulator/{cell_id}/"
    if any(not isinstance(path, str) or not path.startswith(expected_output_prefix) for path in outputs):
        raise ValueError(f"{cell_id}: simulator replay outputs are outside the cell directory")
    if first.read_bytes() != second.read_bytes():
        raise ValueError(f"{cell_id}: simulator twins differ")
    replay_document = json.load(open(first))
    expected_grid = cell.get("expected_grid")
    expected_block = cell.get("expected_block")
    if not (
        isinstance(expected_grid, list)
        and len(expected_grid) == 3
        and all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in expected_grid)
        and isinstance(expected_block, list)
        and len(expected_block) == 3
        and all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in expected_block)
    ):
        raise ValueError(f"{cell_id}: trusted expected grid/block geometry is missing")
    trace_stats = {key.strip(): value.strip() for key, value in stat.items()}
    expected_blocks = expected_grid[0] * expected_grid[1] * expected_grid[2]
    # NVBit's ``#threads`` column is the per-block launch width; NCU's
    # launch__thread_count below is the whole-grid population.
    expected_threads = expected_block[0] * expected_block[1] * expected_block[2]
    for field, expected_value in (
        ("grid_dimX", expected_grid[0]),
        ("grid_dimY", expected_grid[1]),
        ("grid_dimZ", expected_grid[2]),
        ("block_dimX", expected_block[0]),
        ("block_dimY", expected_block[1]),
        ("block_dimZ", expected_block[2]),
        ("#blocks", expected_blocks),
        ("#threads", expected_threads),
    ):
        if _required_int(trace_stats, field, f"{cell_id}/nvbit.stats") != expected_value:
            raise ValueError(f"{cell_id}: NVBit geometry field {field} differs from trusted manifest")
    for field, expected_value in (
        ("launch__grid_dim_x", expected_grid[0]),
        ("launch__grid_dim_y", expected_grid[1]),
        ("launch__grid_dim_z", expected_grid[2]),
        ("launch__block_dim_x", expected_block[0]),
        ("launch__block_dim_y", expected_block[1]),
        ("launch__block_dim_z", expected_block[2]),
        ("launch__thread_count", expected_blocks * expected_threads),
    ):
        if _required_int(ncu_row, field, f"{cell_id}/ncu.csv") != expected_value:
            raise ValueError(f"{cell_id}: NCU geometry differs from trusted manifest")
    expected_launches = replay_resume.expected_launches_from_trace(
        trace, grid=expected_grid, block=expected_block
    )
    trace_invariants = derive_trace_invariants(trace).by_identity()
    if len(expected_launches) != len(trace_invariants):
        raise ValueError(f"{cell_id}: trace launch/payload identity count differs")
    if len(trace_invariants) != 1:
        raise ValueError(f"{cell_id}: trace must contain exactly one launch identity")
    execution_kernel = execution["kernels"][0]
    expected_function_id = expected_launches[0]["function_unique_id"]
    if execution_kernel.get("unique_function_id") != expected_function_id:
        raise ValueError(f"{cell_id}: NVBit execution function identity differs from replay identity")
    mapping_kernel_name = mapping["launches"][0].get("kernel_name")
    expected_kernel_name = execution_kernel.get("kernel_name")
    if not isinstance(mapping_kernel_name, str) or not isinstance(expected_kernel_name, str) or not (
        expected_kernel_name == mapping_kernel_name
        or expected_kernel_name.startswith(mapping_kernel_name + "___")
    ):
        raise ValueError(f"{cell_id}: NVBit mapping kernel identity differs from trace metadata")
    trace_row = next(iter(trace_invariants.values()))
    if trace_row.blocks != expected_blocks:
        raise ValueError(f"{cell_id}: trace payload block population is incomplete")
    expected_warps_per_block = (expected_block[0] * expected_block[1] * expected_block[2] + 31) // 32
    if trace_row.warps != expected_blocks * expected_warps_per_block:
        raise ValueError(f"{cell_id}: trace warp population is incomplete")
    if _required_int(trace_stats, "total_reported_insts", f"{cell_id}/nvbit.stats") <= 0:
        raise ValueError(f"{cell_id}: NVBit trace has no reported instructions")
    simulator_binary = _assert_file(REPO_ROOT, "sim/target/release/sim-run", f"{cell_id}.simulator_binary")
    simulator_path = simulator.get("binary_path")
    simulator_hash = simulator.get("binary_sha256")
    if simulator_path != "sim/target/release/sim-run" or not isinstance(simulator_hash, str):
        raise ValueError(f"{cell_id}: simulator binary path is not current sim-run")
    if simulator_hash != _sha256(simulator_binary):
        raise ValueError(f"{cell_id}: simulator binary hash differs from current sim-run")
    if simulator_argv[0] != simulator_path:
        raise ValueError(f"{cell_id}: simulator argv is not bound to current sim-run")
    dependencies = [
        simulator_binary,
        control,
        _assert_file(REPO_ROOT, "arch_profiles/cards/sm_86.json", f"{cell_id}.arch_card"),
        _assert_file(REPO_ROOT, "data/pchase/sm_86_static_card.json", f"{cell_id}.pchase_card"),
        trace,
    ]
    expectation = replay_resume.ReplayExpectation(
        launches=expected_launches, carded_dram=False
    )
    replay_error = replay_resume.replay_pair_error(
        first, second, dependencies, expectation
    )
    if replay_error is not None:
        raise ValueError(f"{cell_id}: strict direct replay validation failed: {replay_error}")
    semantic_rows, semantic_errors = validate_replay_document(
        replay_document, trace, label=cell_id
    )
    if semantic_errors:
        raise ValueError(f"{cell_id}: semantic replay errors: {'; '.join(semantic_errors[:3])}")
    row["checks"]["simulator_semantics"] = True
    row["semantic_launches"] = semantic_rows
    return row


def validate(manifest_path: Path, report_path: Path | None) -> int:
    manifest_path = manifest_path.expanduser().resolve()
    root = manifest_path.parent
    document = json.load(open(manifest_path))
    if document.get("schema_version") != "sm86.live.campaign.v2":
        raise ValueError("unsupported campaign manifest schema")
    repository = document.get("repository")
    if not isinstance(repository, Mapping):
        raise ValueError("manifest repository identity is missing")
    for revision_key in (
        "capture_revision",
        "handoff_revision",
        "candidate_revision",
        "correction_revision",
    ):
        revision = repository.get(revision_key)
        if not isinstance(revision, str) or len(revision) != 40:
            raise ValueError(f"manifest repository {revision_key} is malformed")
    simulator_record = repository.get("simulator_binary")
    if not isinstance(simulator_record, Mapping):
        raise ValueError("manifest simulator binary identity is missing")
    simulator_binary = _assert_file(
        REPO_ROOT, simulator_record.get("path"), "repository.simulator_binary"
    )
    if _sha256(simulator_binary) != simulator_record.get("sha256"):
        raise ValueError("manifest simulator binary SHA-256 does not match")
    card_paths = {
        "architecture_card_sha256": "arch_profiles/cards/sm_86.json",
        "pchase_card_sha256": "data/pchase/sm_86_static_card.json",
    }
    for hash_key, card_path in card_paths.items():
        expected_hash = repository.get(hash_key)
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"manifest {hash_key} is missing")
        card = _assert_file(REPO_ROOT, card_path, f"repository.{hash_key}")
        if _sha256(card) != expected_hash:
            raise ValueError(f"manifest {hash_key} does not match the current card")
    host = document.get("host")
    if not isinstance(host, Mapping):
        raise ValueError("manifest host preflight record is missing")
    tools = document.get("tools")
    if not isinstance(tools, Mapping):
        raise ValueError("manifest toolchain record is missing")
    host_record_path = _assert_file(root, host.get("preflight_record"), "host.preflight_record")
    _validate_host_preflight(root, host_record_path, repository, host, tools)
    bundle = document.get("evidence_bundle")
    if not isinstance(bundle, Mapping):
        raise ValueError("manifest evidence_bundle is missing")
    archive = _assert_file(root, bundle.get("path"), "evidence_bundle")
    if _sha256(archive) != bundle.get("sha256"):
        raise ValueError("evidence bundle SHA-256 does not match manifest")
    if archive.stat().st_size != bundle.get("size_bytes"):
        raise ValueError("evidence bundle size does not match manifest")
    closure_analysis = document.get("closure_analysis")
    if not isinstance(closure_analysis, Mapping):
        raise ValueError("manifest closure_analysis is missing")
    _check_timeout(closure_analysis, "closure_analysis", {"closure_analysis": 180})
    if closure_analysis.get("exit_code") != 0:
        raise ValueError("closure analysis did not exit zero")
    closure_report_path = _assert_file(
        root, closure_analysis.get("report"), "closure_analysis.report"
    )
    if _sha256(closure_report_path) != closure_analysis.get("report_sha256"):
        raise ValueError("closure analysis report SHA-256 does not match manifest")
    closure_report = json.load(open(closure_report_path))
    if not isinstance(closure_report, Mapping) or closure_report.get("schema_version") != 1:
        raise ValueError("closure analysis report schema is unsupported")
    analysis = closure_report.get("analysis")
    if not isinstance(analysis, Mapping) or analysis.get("decision") != "no-go":
        raise ValueError("closure analysis decision is missing or not no-go")
    analysis_manifest = analysis.get("manifest")
    if not isinstance(analysis_manifest, Mapping) or analysis_manifest.get("candidate_revision") != repository.get("candidate_revision"):
        raise ValueError("closure analysis candidate revision differs from manifest")
    input_validation = analysis.get("input_validation")
    if (
        not isinstance(input_validation, Mapping)
        or input_validation.get("decision_eligible") is not True
        or input_validation.get("repeat_stable_rows") != 6
        or input_validation.get("required_diagnostic_rows") != 6
        or input_validation.get("retirement_tail_rows") != 5
    ):
        raise ValueError("closure analysis lacks complete data-dependent validation")
    closure_rows = closure_report.get("rows")
    if not isinstance(closure_rows, list) or len(closure_rows) != 6:
        raise ValueError("closure analysis must contain one row per cohort cell")
    closure_ids = [
        row.get("cell") if isinstance(row, Mapping) else None
        for row in closure_rows
    ]
    if len(set(closure_ids)) != 6 or set(closure_ids) != _EXPECTED_COHORT_IDS:
        raise ValueError("closure analysis must contain one row for each unique cohort cell")
    if any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("cell"), str)
        or not isinstance(row.get("live"), Mapping)
        or row["live"].get("stable_within_five_percent") is not True
        or not isinstance(row.get("simulator"), Mapping)
        or row["simulator"].get("twin_byte_identical") is not True
        for row in closure_rows
    ):
        raise ValueError("closure analysis rows lack repeatability or simulator-twin evidence")
    if analysis.get("ncu_values_used_as_simulator_inputs") is not False:
        raise ValueError("closure analysis violates the NCU input boundary")
    candidate_ranking = closure_report.get("candidate_ranking")
    if not isinstance(candidate_ranking, list) or not any(
        isinstance(candidate, Mapping)
        and candidate.get("candidate") == "dirty_writeback_coherence_retirement"
        and candidate.get("selection") == "next_discriminator"
        and isinstance(candidate.get("reason"), str)
        for candidate in candidate_ranking
    ):
        raise ValueError("closure analysis diagnostic selection record is missing")
    cells = document.get("cells")
    if not isinstance(cells, list) or len(cells) != 6:
        raise ValueError("manifest must contain exactly six cohort cells")
    cell_ids = [cell.get("id") if isinstance(cell, Mapping) else None for cell in cells]
    if len(set(cell_ids)) != 6 or set(cell_ids) != _EXPECTED_COHORT_IDS:
        raise ValueError("manifest must contain the exact unique six-cell SM86 cohort")
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise ValueError("cohort cell record is malformed")
        _validate_cell_identity(cell)
    timeouts = document.get("stage_timeouts_seconds")
    if not isinstance(timeouts, Mapping):
        raise ValueError("manifest stage_timeouts_seconds is missing")
    if any(
        timeouts.get(stage) != value for stage, value in _CURRENT_TIMEOUTS.items()
    ):
        raise ValueError("manifest stage timeout table differs from the validated contract")
    rows = [
        _validate_cell(
            root,
            cell,
            timeouts,
            repository.get("tracer_dso_sha256"),
        )
        for cell in cells
    ]
    try:
        manifest_label = str(manifest_path.relative_to(REPO_ROOT))
    except ValueError:
        manifest_label = str(manifest_path)
    report = {
        "schema_version": 1,
        "manifest": manifest_label,
        "cell_count": len(rows),
        "semantic_launch_count": sum(len(row["semantic_launches"]) for row in rows),
        "passed": True,
        "rows": rows,
        "ncu_values_used_as_simulator_inputs": False,
    }
    if report_path is not None:
        report_path = report_path.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"validated fresh SM86 cohort: {len(rows)} cells, {report['semantic_launch_count']} semantic launches")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        return validate(args.manifest, args.report)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"live cohort validation: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
