#!/usr/bin/env python3
"""Reduce the SM86 multi-line asynchronous-copy response-envelope assay.

The depth-1/lead-16 vertical smoke is a mandatory structural gate.  A complete
static, attribution, or response-envelope failure is a successful scientific
no-go and does not require the full matrix.  When the smoke passes, the
reducer admits the complete static/NCU/NVBit/replay matrix before opening any
live timing result.  Depth 1 identifies the fixed schema-18 score; depth 2 is
read only after identification passes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import capture_multiline_execution as execution_capture
import capture_multiline_response as ncu_capture
from analyze_cohort_reservation import (
    AnalysisError,
    integer,
    parse_csv_rows,
    read_json,
    schema_observation,
    sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO_ROOT / "results/captures/sm86_async_multiline_response_20260831_v1"
PROBE_BINARY = (
    REPO_ROOT / "workloads/research_probes/async_source_only/async_multiline_response"
)
TRACE_GROUPS = 64
DIRECT_GROUPS = 8192
DEPTHS = (1, 2)
LEADS = (0, 4, 8, 16)
REPETITIONS = tuple(range(1, 11))
VARIANTS = ("span_expensive", "span_cheap")
SMOKE_SPEC = (1, 16)
IDENTIFICATION_SPECS = tuple((1, lead) for lead in LEADS)
HOLDOUT_SPECS = tuple((2, lead) for lead in LEADS)

FAMILY_METRICS = {
    "global_ld": "smsp__inst_executed_op_global_ld.sum",
    "ldgsts": "smsp__inst_executed_op_ldgsts.sum",
    "lts_total": "lts__t_sectors_srcunit_tex_aperture_device_op_read.sum",
    "lts_hit": (
        "lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_hit.sum"
    ),
    "lts_miss": (
        "lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_miss.sum"
    ),
    "dram": "dram__sectors_read.sum",
}
FAMILIES = tuple(FAMILY_METRICS)

EXPECTED_DEVICE = {
    "index": "0",
    "name": "NVIDIA GeForce RTX 3090 Ti",
    "uuid": "GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c",
    "compute_capability": "8.6",
    "driver": "610.57.04",
}
EXPECTED_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "0",
    "LANG": "C",
    "LC_ALL": "C",
}
EXPECTED_NCU_VERSION = "Version 2026.1.0.0 (build 37166530) (public-release)"

SCHEMA18_FIELDS = (
    "cohort_overlap_groups_observed",
    "cohort_response_envelope_cycles",
    "cohort_topology_prearrival_hidden_cycles",
    "cohort_topology_stream_hidden_cycles",
    "cohort_topology_residual_exposure_cycles",
)
MAE_LIMIT = 1.0
MAX_ERROR_LIMIT = 2.0

RESOURCE_RE = re.compile(
    r"^ Function (?P<name>\S+):\n"
    r"  REG:(?P<registers>\d+) STACK:(?P<stack>\d+) "
    r"SHARED:(?P<shared>\d+) LOCAL:(?P<local>\d+)",
    re.MULTILINE,
)
SPILL_RE = re.compile(
    r"(?P<stack>\d+) bytes stack frame, (?P<stores>\d+) bytes spill stores, "
    r"(?P<loads>\d+) bytes spill loads"
)
def read_json_list(path: Path) -> list[Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisError(f"cannot read {path}: {error}") from error
    if not isinstance(value, list):
        raise AnalysisError(f"{path}: expected one JSON array")
    return value


def last_json_object(path: Path) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as error:
        raise AnalysisError(f"cannot read {path}: {error}") from error
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise AnalysisError(f"{path}: no JSON result object")


def expected_order(repetition: int) -> str:
    return "ab" if repetition % 2 else "ba"


def row_variants(order: str) -> tuple[str, str]:
    if order == "ab":
        return VARIANTS
    if order == "ba":
        return tuple(reversed(VARIANTS))
    raise AnalysisError(f"unsupported order {order!r}")


def cell_specs() -> list[tuple[int, int]]:
    return [(depth, lead) for depth in DEPTHS for lead in LEADS]


def replay_geometry_expectations(copies: int) -> dict[str, int]:
    """Frozen trace/replay geometry; scientific mismatches are no-go evidence."""

    return {
        "copies_observed": copies,
        # The probe contract freezes 20 destination wavefronts per copy.
        # Current replay reports 32; that is a structural assay result, not a
        # value to silently relabel as the intended destination topology.
        "destination_wavefronts_observed": copies * 20,
        "bypass_wavefronts_observed": copies * 32,
        "source_cohorts_observed": copies * 8,
        "destination_lines_observed": copies * 20,
        "max_bank_wavefronts_sum": copies * 4,
        "max_bypass_wavefronts_sum": copies * 4,
        "cohort_groups_observed": TRACE_GROUPS,
        "cohort_copies_observed": copies,
        "cohort_response_memberships_observed": copies * 32,
        "cohort_topology_penalty_cycles": copies * 72,
        "cohort_overlap_groups_observed": TRACE_GROUPS,
    }


def endpoint_specs() -> list[tuple[str, int, int]]:
    return [
        (variant, depth, lead)
        for variant in VARIANTS
        for depth, lead in cell_specs()
    ]


def pair_cell_name(depth: int, lead: int, groups: int = DIRECT_GROUPS) -> str:
    return f"async_multiline_response_d{depth}_l{lead}_g{groups}"


def endpoint_name(variant: str, depth: int, lead: int) -> str:
    return f"async_multiline_response_{variant}_d{depth}_l{lead}_g{TRACE_GROUPS}"


def function_name(depth: int, lead: int) -> str:
    return (
        f"_Z25multiline_response_kernelILi{depth}ELi{lead}EEv"
        "PKcjibP15MultilineResult"
    )


def expected_instruction_count(depth: int) -> int:
    return TRACE_GROUPS * depth


def expected_lts_total(depth: int) -> int:
    return 56 * TRACE_GROUPS * depth


def paired_tolerance(depth: int) -> int:
    return max(4, math.ceil(0.01 * expected_lts_total(depth)))


def expected_target_arguments(depth: int, lead: int, repetition: int) -> list[str]:
    return [
        "--groups",
        str(TRACE_GROUPS),
        "--depth",
        str(depth),
        "--lead-samples",
        str(lead),
        "--order",
        expected_order(repetition),
        "--markers",
        "on",
    ]


def binary_report(root: Path) -> dict[str, Any]:
    try:
        return execution_capture.retained_binary(root)
    except execution_capture.CaptureError as error:
        raise AnalysisError(str(error)) from error


def common_target_contract(
    *, depth: int, lead: int, groups: int, mode: str
) -> dict[str, Any]:
    return {
        "workload": "async_multiline_response",
        "route": "l2",
        "mode": mode,
        "depth": depth,
        "groups": groups,
        "warmup_groups": 64,
        "lead_clock_samples": lead,
        "source_cohorts_per_copy": 8,
        "aligned_source_cohorts_per_copy": 4,
        "split_source_cohorts_per_copy": 4,
        "source_sectors_per_cohort": 4,
        "payload_bytes_per_cohort": 64,
        "source_step_stride_bytes": 2048,
        "source_step_max_exclusive_byte": 1968,
        "source_step_guard_bytes": 80,
        "scalar_ldg_instructions_per_copy": 1,
        "scalar_ldg_sectors_per_copy": 24,
        "scalar_preload_lines_per_copy": 8,
        "ldgsts_instructions_per_copy": 1,
        "ldgsts_sectors_per_copy": 32,
        "unique_ldgsts_lines_per_copy": 12,
        "destination_lines_per_copy": 20,
        "local_device_lts_lookups_per_copy": 56,
        "compact_destination_cohorts_per_copy": 4,
        "severe_destination_cohorts_per_copy": 4,
        "source_ring_bytes_per_arm": 64 << 20,
        "source_ring_steps_per_arm": 32768,
        "max_admitted_groups": 16320,
        "source_ring_no_wrap": True,
        "source_pool_bytes": 128 << 20,
        "source_pool_alignment_bytes": 128,
        "source_arm_a_offset_bytes": 0,
        "source_arm_b_offset_bytes": 64 << 20,
        "source_pool_single_allocation": True,
        "independent_source_arms": True,
        "source_vector_preserved": True,
        "preload_vector_preserved": True,
        "destination_multiset_preserved": True,
        "result_store_after_intervals": True,
        "device": EXPECTED_DEVICE["name"],
        "compute_capability": EXPECTED_DEVICE["compute_capability"],
        "l2_cache_bytes": 6291456,
        "correct": True,
    }


def validate_arm(value: Any, groups: int, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AnalysisError(f"{context}: expected result object")
    for field in ("issue_cycles", "wait_cycles", "total_cycles", "token", "mismatches"):
        integer(value.get(field), f"{context}.{field}")
    if value["issue_cycles"] + value["wait_cycles"] != value["total_cycles"]:
        raise AnalysisError(f"{context}: measured intervals do not sum")
    for field, numerator in (
        ("issue_cycles_per_group", value["issue_cycles"]),
        ("wait_cycles_per_group", value["wait_cycles"]),
        ("total_cycles_per_group", value["total_cycles"]),
    ):
        observed = value.get(field)
        expected = numerator / groups
        if not isinstance(observed, (int, float)) or not math.isclose(
            float(observed), expected, rel_tol=0.0, abs_tol=1.0e-8
        ):
            raise AnalysisError(f"{context}.{field}: inconsistent interval")
    if value.get("sum_consistent") is not True or value.get("correct") is not True:
        raise AnalysisError(f"{context}: correctness flag failed")
    if value["mismatches"] != 0:
        raise AnalysisError(f"{context}: mismatches={value['mismatches']}")
    return value


def validate_target_payload(
    payload: Mapping[str, Any],
    *,
    depth: int,
    lead: int,
    groups: int,
    mode: str,
    order: str,
    variant: str | None = None,
) -> None:
    expected = common_target_contract(depth=depth, lead=lead, groups=groups, mode=mode)
    expected["order"] = order
    mismatches = {
        field: {"observed": payload.get(field), "expected": value}
        for field, value in expected.items()
        if payload.get(field) != value
    }
    if mismatches:
        raise AnalysisError(f"target contract mismatch: {mismatches}")
    if mode == "pair":
        for arm in VARIANTS:
            validate_arm(payload.get(arm), groups, f"target.{arm}")
    else:
        if payload.get("variant") != variant:
            raise AnalysisError(
                f"target variant={payload.get('variant')!r}, expected {variant!r}"
            )
        validate_arm(payload.get("result"), groups, f"target.{variant}")


def load_resources(path: Path) -> dict[str, dict[str, int]]:
    text = path.read_text(encoding="utf-8")
    return {
        match.group("name"): {
            "registers": int(match.group("registers")),
            "stack": int(match.group("stack")),
            "shared": int(match.group("shared")),
            "local": int(match.group("local")),
        }
        for match in RESOURCE_RE.finditer(text)
    }


def static_report(root: Path) -> dict[str, Any]:
    base = root / "static"
    paths = {
        "control": base / "control_bits.json",
        "resources": base / "resources.txt",
        "sass": base / "sass.txt",
        "build": base / "build.log",
    }
    absent = [str(path) for path in paths.values() if not path.is_file()]
    if absent:
        raise FileNotFoundError(", ".join(absent))
    control = read_json(paths["control"])
    functions = control.get("functions")
    summary = control.get("summary")
    if not isinstance(functions, dict) or not isinstance(summary, dict):
        raise AnalysisError(f"{paths['control']}: malformed control sidecar")
    resources = load_resources(paths["resources"])
    sass = paths["sass"].read_text(encoding="utf-8")
    build = paths["build"].read_text(encoding="utf-8")
    expected_names = {function_name(depth, lead) for depth, lead in cell_specs()}
    artifact_errors: list[str] = []
    scientific_failures: list[dict[str, Any]] = []
    decoded = 0
    for depth, lead in cell_specs():
        name = function_name(depth, lead)
        rows = functions.get(name)
        resource = resources.get(name)
        if not isinstance(rows, list):
            artifact_errors.append(f"{name}: missing decoded function")
            continue
        decoded += len(rows)
        if any(row.get("status") != "decoded" for row in rows):
            artifact_errors.append(f"{name}: undecoded instruction")
            continue
        failures: list[str] = []
        opcodes = [str(row.get("opcode_text", "")) for row in rows]
        scalar_ldg = [
            index
            for index, opcode in enumerate(opcodes)
            if opcode.startswith("LDG.E.STRONG.GPU")
        ]
        ldgsts = [index for index, opcode in enumerate(opcodes) if opcode.startswith("LDGSTS")]
        commits = [index for index, opcode in enumerate(opcodes) if opcode == "LDGDEPBAR"]
        waits = [
            index
            for index, opcode in enumerate(opcodes)
            if opcode.startswith("DEPBAR.LE SB0, 0x0")
        ]
        primary_clocks = [
            index
            for index, opcode in enumerate(opcodes)
            if opcode.startswith("CS2R ") and "SR_CLOCKLO" in opcode
        ]
        lead_clocks = [
            index
            for index, opcode in enumerate(opcodes)
            if (
                opcode.startswith("CS2R.32 ")
                or opcode.startswith("S2UR ")
            )
            and "SR_CLOCKLO" in opcode
        ]
        stores = [index for index, opcode in enumerate(opcodes) if opcode.startswith("STG")]
        if len(scalar_ldg) != depth:
            failures.append(f"scalar LDG count {len(scalar_ldg)} != {depth}")
        if len(ldgsts) != depth:
            failures.append(f"LDGSTS count {len(ldgsts)} != {depth}")
        if any(not opcodes[index].startswith("LDGSTS.E.BYPASS.128 ") for index in ldgsts):
            failures.append("LDGSTS route/width drift")
        if len(commits) != 1 or len(waits) != 1:
            failures.append("async commit/wait count drift")
        if len(primary_clocks) != 3:
            failures.append(f"primary clock count {len(primary_clocks)} != 3")
        if len(lead_clocks) != lead:
            failures.append(f"retained lead clock count {len(lead_clocks)} != {lead}")
        if primary_clocks and stores and min(stores) <= primary_clocks[-1]:
            failures.append("global result store precedes post-wait clock")
        if (
            scalar_ldg
            and ldgsts
            and len(commits) == 1
            and len(waits) == 1
            and len(primary_clocks) == 3
            and not (
                max(scalar_ldg)
                < primary_clocks[0]
                < min(ldgsts)
                <= max(ldgsts)
                < primary_clocks[1]
                < commits[0]
                < waits[0]
                < primary_clocks[2]
            )
        ):
            failures.append("preload/clock/LDGSTS/commit/wait ordering drift")
        if scalar_ldg and len(primary_clocks) == 3:
            preload_barriers = {
                int(rows[index].get("control", {}).get("write_barrier", 7))
                for index in scalar_ldg
            }
            preload_barriers.discard(7)
            early_start = min(scalar_ldg) + 1
            early_end = primary_clocks[2] + 1
            for index in range(early_start, early_end):
                wait_mask = int(rows[index].get("control", {}).get("wait_mask", 0))
                consumed = [barrier for barrier in preload_barriers if wait_mask & (1 << barrier)]
                if consumed:
                    failures.append(
                        f"preload barrier(s) {consumed} consumed before post-wait clock at pc index {index}"
                    )
                    break
        if any(opcode.startswith("BAR") for opcode in opcodes):
            failures.append("CTA barrier entered the probe kernel")
        if resource is None:
            artifact_errors.append(f"{name}: missing resource record")
        else:
            if resource["stack"] != 0 or resource["local"] != 0:
                failures.append("stack/local allocation is nonzero")
            if resource["shared"] != 8192:
                failures.append(f"shared allocation {resource['shared']} != 8192")
            if resource["registers"] <= 0:
                failures.append("register count is not positive")
        if name not in sass:
            artifact_errors.append(f"{name}: absent from SASS dump")
        if failures:
            scientific_failures.append(
                {"function": name, "depth": depth, "lead_samples": lead, "failures": failures}
            )
    if set(functions) != expected_names:
        artifact_errors.append("control function set is not the frozen eight specializations")
    if set(resources) != expected_names:
        artifact_errors.append("resource function set is not the frozen eight specializations")
    if summary.get("total_decoded") != decoded or summary.get("total_unknown_encoding") != 0:
        artifact_errors.append("control decode summary is inconsistent or has unknown encodings")
    if decoded != 1208:
        scientific_failures.append(
            {"function": "all", "failures": [f"decoded instruction count {decoded} != 1208"]}
        )
    spill_rows = [
        tuple(int(match.group(field)) for field in ("stack", "stores", "loads"))
        for match in SPILL_RE.finditer(build)
    ]
    if len(spill_rows) != 8:
        artifact_errors.append(f"build log has {len(spill_rows)} spill records, expected 8")
    elif any(any(row) for row in spill_rows):
        scientific_failures.append(
            {"function": "all", "failures": ["ptxas reports stack or spills"]}
        )
    return {
        "artifact_valid": not artifact_errors,
        "assay_valid": not artifact_errors and not scientific_failures,
        "artifact_errors": artifact_errors,
        "scientific_failures": scientific_failures,
        "function_count": len(functions),
        "decoded_instructions": decoded,
        "artifacts": {name: sha256(path) for name, path in paths.items()},
    }


def validate_ncu_command(
    command: Mapping[str, Any],
    directory: Path,
    family: str,
    depth: int,
    lead: int,
    repetition: int,
    binary: Mapping[str, Any],
) -> None:
    order = expected_order(repetition)
    expected = {
        "schema_version": 1,
        "cell": pair_cell_name(depth, lead, TRACE_GROUPS),
        "depth": depth,
        "lead_samples": lead,
        "groups": TRACE_GROUPS,
        "repetition": repetition,
        "order": order,
        "row_variants": list(row_variants(order)),
        "family": family,
        "metric": FAMILY_METRICS[family],
        "binary": dict(binary),
        "device": EXPECTED_DEVICE,
        "environment_overrides": EXPECTED_ENVIRONMENT,
        "target_arguments": expected_target_arguments(depth, lead, repetition),
    }
    timeout = command.get("timeout_seconds")
    if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout < 1:
        raise AnalysisError(f"{directory}: invalid configured NCU timeout {timeout!r}")
    mismatches = {
        field: {"observed": command.get(field), "expected": value}
        for field, value in expected.items()
        if command.get(field) != value
    }
    if mismatches:
        raise AnalysisError(f"{directory}: NCU command binding mismatch: {mismatches}")
    ncu = command.get("ncu")
    if not isinstance(ncu, dict):
        raise AnalysisError(f"{directory}: missing NCU identity")
    ncu_path = ncu.get("path")
    version = ncu.get("version")
    if not isinstance(ncu_path, str) or not ncu_path.endswith("/ncu"):
        raise AnalysisError(f"{directory}: invalid NCU path")
    if not isinstance(version, str) or EXPECTED_NCU_VERSION not in version:
        raise AnalysisError(f"{directory}: wrong NCU version")
    expected_profile = [
        ncu_path,
        "--devices",
        "0",
        "--target-processes",
        "application-only",
        "--profile-from-start",
        "off",
        "--launch-count",
        "2",
        "--replay-mode",
        "application",
        "--app-replay-mode",
        "strict",
        "--app-replay-match",
        "grid",
        "--cache-control",
        "none",
        "--clock-control",
        "none",
        "--kernel-name",
        "regex:multiline_response_kernel",
        "--metrics",
        FAMILY_METRICS[family],
        "--export",
        str(directory / "profile"),
        "--force-overwrite",
        str(PROBE_BINARY),
        *expected_target_arguments(depth, lead, repetition),
    ]
    expected_import = [
        ncu_path,
        "--import",
        str(directory / "profile.ncu-rep"),
        "--page",
        "raw",
        "--csv",
        "--print-units",
        "base",
        "--print-fp",
    ]
    if command.get("profile_command") != expected_profile:
        raise AnalysisError(f"{directory}: NCU profile command drift")
    if command.get("import_command") != expected_import:
        raise AnalysisError(f"{directory}: NCU import command drift")


def validate_ncu_family(
    directory: Path,
    family: str,
    depth: int,
    lead: int,
    repetition: int,
    binary: Mapping[str, Any],
) -> dict[str, Any]:
    paths = {
        name: directory / name
        for name in (
            "command.json",
            "complete.json",
            "ncu.csv",
            "profile.ncu-rep",
            "profile.stdout.txt",
            "profile.stderr.txt",
            "import.stderr.txt",
            "profile.run.json",
            "import.run.json",
        )
    }
    absent = [str(path) for path in paths.values() if not path.is_file()]
    if absent:
        raise FileNotFoundError(", ".join(absent))
    if paths["profile.ncu-rep"].stat().st_size == 0:
        raise AnalysisError(f"{paths['profile.ncu-rep']}: empty report")
    command = read_json(paths["command.json"])
    complete = read_json(paths["complete.json"])
    validate_ncu_command(command, directory, family, depth, lead, repetition, binary)
    try:
        ncu_capture.validate_run_manifest(
            paths["profile.run.json"], "ncu_profile", command, directory
        )
        ncu_capture.validate_run_manifest(
            paths["import.run.json"], "ncu_import", command, directory
        )
    except ncu_capture.CampaignError as error:
        raise AnalysisError(f"{directory}: invalid retained run manifest: {error}") from error
    expected_files = {
        "command.json": sha256(paths["command.json"]),
        "profile.ncu-rep": sha256(paths["profile.ncu-rep"]),
        "ncu.csv": sha256(paths["ncu.csv"]),
        "profile.stdout.txt": sha256(paths["profile.stdout.txt"]),
        "profile.stderr.txt": sha256(paths["profile.stderr.txt"]),
        "import.stderr.txt": sha256(paths["import.stderr.txt"]),
        "profile.run.json": sha256(paths["profile.run.json"]),
        "import.run.json": sha256(paths["import.run.json"]),
    }
    if complete.get("schema_version") != 1 or complete.get("valid") is not True:
        raise AnalysisError(f"{paths['complete.json']}: invalid completion record")
    if complete.get("files") != expected_files:
        raise AnalysisError(f"{paths['complete.json']}: file identity mismatch")
    binding_fields = (
        "cell",
        "depth",
        "lead_samples",
        "groups",
        "repetition",
        "order",
        "row_variants",
        "family",
        "metric",
        "timeout_seconds",
        "binary",
        "device",
        "ncu",
        "target_arguments",
    )
    if complete.get("binding") != {field: command[field] for field in binding_fields}:
        raise AnalysisError(f"{paths['complete.json']}: binding mismatch")
    rows = parse_csv_rows(paths["ncu.csv"])
    if len(rows) != 2 or [str(row.get("ID", "")).strip() for row in rows] != ["0", "1"]:
        raise AnalysisError(f"{paths['ncu.csv']}: expected ordered rows 0/1")
    pids = [str(row.get("Process ID", "")).strip() for row in rows]
    if not pids[0] or pids[0] != pids[1]:
        raise AnalysisError(f"{paths['ncu.csv']}: rows are not one process")
    variants = row_variants(expected_order(repetition))
    values: dict[str, int] = {}
    summaries: list[dict[str, Any]] = []
    metric = FAMILY_METRICS[family]
    for index, (row, variant) in enumerate(zip(rows, variants, strict=True)):
        label = f"{paths['ncu.csv']} row {index}"
        kernel = str(row.get("Kernel Name", ""))
        if f"multiline_response_kernel<{depth},{lead}>" not in kernel.replace(" ", ""):
            raise AnalysisError(f"{label}: wrong kernel {kernel!r}")
        strings = {
            "Process Name": "async_multiline_response",
            "Device": "0",
            "CC": "8.6",
            "Grid Size": "(1, 1, 1)",
            "Block Size": "(32, 1, 1)",
            "device__attribute_display_name": EXPECTED_DEVICE["name"],
        }
        for field, expected in strings.items():
            if str(row.get(field, "")).strip() != expected:
                raise AnalysisError(f"{label}: {field} mismatch")
        for field, expected in {
            "device__attribute_device_index": 0,
            "device__attribute_compute_capability_major": 8,
            "device__attribute_compute_capability_minor": 6,
            "launch__grid_size": 1,
            "launch__block_size": 32,
            "profiler__replayer_passes": 1,
        }.items():
            if integer(row.get(field), f"{label}.{field}") != expected:
                raise AnalysisError(f"{label}: {field} mismatch")
        value = integer(row.get(metric), f"{label}.{metric}")
        values[variant] = value
        summaries.append(
            {
                "id": index,
                "variant": variant,
                "process_id": pids[index],
                "kernel": kernel,
                "context": str(row.get("Context", "")),
                "stream": str(row.get("Stream", "")),
                "metric_value": value,
                "profiler_replayer_passes": 1,
            }
        )
    payload = last_json_object(paths["profile.stdout.txt"])
    validate_target_payload(
        payload,
        depth=depth,
        lead=lead,
        groups=TRACE_GROUPS,
        mode="pair",
        order=expected_order(repetition),
    )
    expected_value = None
    if family in {"global_ld", "ldgsts"}:
        expected_value = expected_instruction_count(depth)
    elif family == "lts_total":
        expected_value = expected_lts_total(depth)
    scientific_gate = {
        "expected_value": expected_value,
        "exact_expected_value": (
            all(value == expected_value for value in values.values())
            if expected_value is not None
            else None
        ),
        "positive_population": (
            all(value > 0 for value in values.values())
            if family in {"lts_hit", "lts_miss"}
            else None
        ),
    }
    expected_summary = {
        "order": expected_order(repetition),
        "row_variants": list(variants),
        "metric": metric,
        "values_by_variant": values,
        "rows": summaries,
        "scientific_gate": scientific_gate,
    }
    if complete.get("summary") != expected_summary:
        raise AnalysisError(f"{paths['complete.json']}: summary mismatch")
    if complete.get("target_result") != payload:
        raise AnalysisError(f"{paths['complete.json']}: target result mismatch")
    return {
        "family": family,
        "metric": metric,
        "values_by_variant": values,
        "rows": summaries,
        "files": expected_files,
        "scientific_gate": scientific_gate,
    }


def expected_bundle(
    depth: int,
    lead: int,
    repetition: int,
    families: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    values = {family: dict(families[family]["values_by_variant"]) for family in FAMILIES}
    expected_inst = expected_instruction_count(depth)
    expected_total = expected_lts_total(depth)
    tolerance = paired_tolerance(depth)
    hit_delta = abs(values["lts_hit"][VARIANTS[0]] - values["lts_hit"][VARIANTS[1]])
    miss_delta = abs(values["lts_miss"][VARIANTS[0]] - values["lts_miss"][VARIANTS[1]])
    closure = {
        variant: abs(
            values["lts_hit"][variant]
            + values["lts_miss"][variant]
            - values["lts_total"][variant]
        )
        for variant in VARIANTS
    }
    gates = {
        "global_ld_exact": all(values["global_ld"][v] == expected_inst for v in VARIANTS),
        "ldgsts_exact": all(values["ldgsts"][v] == expected_inst for v in VARIANTS),
        "lts_total_exact": all(values["lts_total"][v] == expected_total for v in VARIANTS),
        "lts_hit_positive": all(values["lts_hit"][v] > 0 for v in VARIANTS),
        "lts_miss_positive": all(values["lts_miss"][v] > 0 for v in VARIANTS),
        "paired_lts_hit_within_tolerance": hit_delta <= tolerance,
        "paired_lts_miss_within_tolerance": miss_delta <= tolerance,
        "separate_profile_closure_within_tolerance": all(
            value <= tolerance for value in closure.values()
        ),
        "all_rows_one_replay_pass": True,
    }
    background = {
        variant: values["dram"][variant] - values["lts_miss"][variant]
        for variant in VARIANTS
    }
    return {
        "schema_version": 1,
        "valid": all(gates.values()),
        "cell": pair_cell_name(depth, lead, TRACE_GROUPS),
        "depth": depth,
        "lead_samples": lead,
        "groups": TRACE_GROUPS,
        "repetition": repetition,
        "order": expected_order(repetition),
        "row_variants": list(row_variants(expected_order(repetition))),
        "families": {
            family: {
                "metric": FAMILY_METRICS[family],
                "values_by_variant": values[family],
                "files": dict(families[family]["files"]),
            }
            for family in FAMILIES
        },
        "attribution": {
            "frontier": "l1tex_local_device_aperture_lts_requested_demand",
            "physical_dram_completion_claimed": False,
            "expected_global_ld_instructions_per_variant": expected_inst,
            "expected_ldgsts_instructions_per_variant": expected_inst,
            "expected_lts_lookup_sectors_per_variant": expected_total,
            "paired_tolerance": tolerance,
            "absolute_lts_hit_delta": hit_delta,
            "absolute_lts_miss_delta": miss_delta,
            "absolute_hit_plus_miss_minus_total_by_variant": closure,
            "all_rows_one_replay_pass": True,
            "gates": gates,
        },
        "non_gating_dram_background": {
            "raw_dram_sectors_by_variant": values["dram"],
            "raw_dram_minus_attributed_lts_misses": background,
        },
    }


def validate_bundle(
    root: Path,
    depth: int,
    lead: int,
    repetition: int,
    binary: Mapping[str, Any],
) -> dict[str, Any]:
    base = (
        root
        / "paired_ncu"
        / pair_cell_name(depth, lead, TRACE_GROUPS)
        / f"repeat-{repetition}"
    )
    families = {
        family: validate_ncu_family(
            base / family, family, depth, lead, repetition, binary
        )
        for family in FAMILIES
    }
    bundle_path = base / "bundle.json"
    if not bundle_path.is_file():
        raise FileNotFoundError(str(bundle_path))
    expected = expected_bundle(depth, lead, repetition, families)
    if read_json(bundle_path) != expected:
        raise AnalysisError(f"{bundle_path}: bundle differs from raw reports")
    return {
        "cell": expected["cell"],
        "depth": depth,
        "lead_samples": lead,
        "repetition": repetition,
        "order": expected_order(repetition),
        "valid": expected["valid"],
        "attribution": expected["attribution"],
        "non_gating_dram_background": expected["non_gating_dram_background"],
        "families": families,
        "bundle_sha256": sha256(bundle_path),
    }


def argv_options(argv: Sequence[Any]) -> dict[str, str]:
    values = [str(value) for value in argv]
    options: dict[str, str] = {}
    index = 1
    while index + 1 < len(values):
        if values[index].startswith("--"):
            options[values[index]] = values[index + 1]
            index += 2
        else:
            index += 1
    return options


def retained_tracer_identity(
    complete: Mapping[str, Any], binding: Mapping[str, Any], context: str
) -> str:
    tracer_sha = complete.get("tracer_sha256")
    bound_tracer = binding.get("tracer")
    if (
        not isinstance(tracer_sha, str)
        or not isinstance(bound_tracer, dict)
        or bound_tracer.get("sha256") != tracer_sha
    ):
        raise AnalysisError(f"{context}: complete/binding tracer identity mismatch")
    return tracer_sha


def nvbit_cell(
    root: Path, variant: str, depth: int, lead: int, expected_binary_sha256: str
) -> dict[str, Any]:
    name = endpoint_name(variant, depth, lead)
    base = root / "nvbit" / name
    complete_path = base / "complete.json"
    binding_path = base / "campaign_binding.json"
    stats_path = base / "stats.csv"
    trace_path = base / "dynamic_trace.pb"
    stdout_path = base / "stdout.txt"
    for path in (complete_path, binding_path, stats_path, trace_path, stdout_path):
        if not path.is_file():
            raise FileNotFoundError(str(path))
    complete = read_json(complete_path)
    binding = read_json(binding_path)
    if complete.get("complete") is not True or complete.get("key") != name:
        raise AnalysisError(f"{name}: invalid NVBit completion record")
    required = complete.get("required_files")
    if not isinstance(required, list) or any(not (base / str(item)).is_file() for item in required):
        raise AnalysisError(f"{name}: incomplete required NVBit files")
    if trace_path.stat().st_size == 0:
        raise AnalysisError(f"{name}: empty dynamic trace")
    if binding.get("cell") != name or binding.get("output_contract", {}).get("passed") is not True:
        raise AnalysisError(f"{name}: invalid NVBit binding")
    options = argv_options(binding.get("argv", []))
    expected_options = {
        "--groups": str(TRACE_GROUPS),
        "--depth": str(depth),
        "--lead-samples": str(lead),
        "--variant": variant,
        "--markers": "on",
    }
    if any(options.get(key) != value for key, value in expected_options.items()):
        raise AnalysisError(f"{name}: NVBit argv drift: {options}")
    binary_sha = complete.get("binary_sha256")
    if binary_sha != expected_binary_sha256 or binary_sha != binding.get("binary", {}).get("sha256"):
        raise AnalysisError(f"{name}: NVBit binary identity mismatch")
    tracer_sha = retained_tracer_identity(complete, binding, name)
    exclusions: dict[str, Any] = {}
    for reason in (
        "unsupported_memory_path",
        "unmapped_memory_range",
        "cross_allocation_lane_span",
    ):
        path = base / f"extra_info/{reason}.json"
        if not path.is_file():
            raise FileNotFoundError(str(path))
        entries = read_json_list(path)
        exclusions[reason] = {"entries": len(entries), "sha256": sha256(path)}
        if entries:
            raise AnalysisError(f"{name}: nonempty {reason} sidecar")
    try:
        with stats_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, skipinitialspace=True))
    except (OSError, csv.Error) as error:
        raise AnalysisError(f"cannot read {stats_path}: {error}") from error
    if len(rows) != 1:
        raise AnalysisError(f"{stats_path}: expected one kernel row")
    try:
        total = int(rows[0]["total_insts"])
        reported = int(rows[0]["total_reported_insts"])
    except (KeyError, ValueError) as error:
        raise AnalysisError(f"{stats_path}: malformed totals") from error
    if total <= 0 or total != reported:
        raise AnalysisError(f"{stats_path}: incomplete instruction trace")
    payload = last_json_object(stdout_path)
    validate_target_payload(
        payload,
        depth=depth,
        lead=lead,
        groups=TRACE_GROUPS,
        mode="endpoint",
        order="ab",
        variant=variant,
    )
    return {
        "cell": name,
        "variant": variant,
        "depth": depth,
        "lead_samples": lead,
        "total_instructions": total,
        "trace_bytes": trace_path.stat().st_size,
        "binary_sha256": binary_sha,
        "tracer_sha256": tracer_sha,
        "exclusions": exclusions,
        "trace_sha256": sha256(trace_path),
        "binding_sha256": sha256(binding_path),
        "complete_sha256": sha256(complete_path),
    }


def require_cross_endpoint_tracer_identity(
    cells: Sequence[Mapping[str, Any]],
) -> str:
    """Return the one tracer identity or reject the evidence as malformed."""

    identities = {cell.get("tracer_sha256") for cell in cells}
    if len(identities) != 1 or not all(isinstance(value, str) for value in identities):
        raise AnalysisError(
            f"NVBit endpoints do not share one tracer identity: {sorted(map(str, identities))}"
        )
    return str(next(iter(identities)))


def production_contract(launch: Mapping[str, Any]) -> dict[str, bool]:
    readiness = launch.get("timing", {}).get("readiness", {})
    tails = launch.get("counters", {}).get("unmodeled_tails", {})
    limiting = readiness.get("limiting_cycles", {})
    exposure = readiness.get("constraint_exposure_cycles", {})
    groups = readiness.get("async_groups", {})
    return {
        "typed_visibility_tail_retained": tails.get("async_copy_shared_visibility") is True,
        "sm_service_not_limiting": limiting.get("async_destination_sm_service") == 0,
        "sm_service_exposure_zero": exposure.get("async_destination_sm_service") == 0,
        "sm_service_groups_zero": groups.get("destination_sm_service_limited_groups") == 0,
        "groups_waited_exact": groups.get("groups_waited") == TRACE_GROUPS,
    }


def production_fingerprint(launch: Mapping[str, Any]) -> dict[str, Any]:
    timing = launch.get("timing", {})
    readiness = timing.get("readiness", {})
    return {
        "counters": launch.get("counters"),
        "diagnostic_last_issue_cycles": timing.get("diagnostic_last_issue_cycles"),
        "last_processed_modeled_system_event_time": timing.get(
            "last_processed_modeled_system_event_time"
        ),
        "terminal_modeled_event_time": timing.get("terminal_modeled_event_time"),
        "completion_frontier": timing.get("completion_frontier"),
        "limiting_cycles": readiness.get("limiting_cycles"),
        "constraint_exposure_cycles": readiness.get("constraint_exposure_cycles"),
        "async_groups": readiness.get("async_groups"),
    }


def replay_cell(root: Path, variant: str, depth: int, lead: int) -> dict[str, Any]:
    name = endpoint_name(variant, depth, lead)
    base = root / "simulator" / name
    first = base / "replay-1.json"
    second = base / "replay-2.json"
    if not first.is_file() or not second.is_file():
        raise FileNotFoundError(f"{first}, {second}")
    if first.read_bytes() != second.read_bytes():
        raise AnalysisError(f"{name}: replay twins differ")
    try:
        execution_capture.validate_replay(
            root,
            base,
            execution_capture.Cell(depth, lead),
            variant,
        )
    except execution_capture.CaptureError as error:
        raise AnalysisError(f"{name}: {error}") from error
    document = read_json(first)
    launch, _, observation = schema_observation(document, 18, name)
    for field in SCHEMA18_FIELDS:
        integer(observation.get(field), f"{name}.{field}")
    copies = TRACE_GROUPS * depth
    exact = replay_geometry_expectations(copies)
    observed_geometry = {field: observation.get(field) for field in exact}
    geometry_gates = {
        field: observation.get(field) == expected for field, expected in exact.items()
    }
    counters = launch.get("counters", {})
    global_issue = launch.get("timing", {}).get("readiness", {}).get(
        "global_memory_issue", {}
    )
    work_gates = {
        "scalar_load_instructions_exact": global_issue.get("load_instructions") == copies,
        "destination_wavefronts_observed_exact": observation.get(
            "destination_wavefronts_observed"
        )
        == copies * 20,
        "bypass_sector_work_exact": counters.get("l1", {}).get(
            "bypass_sectors"
        )
        == copies * 56,
        "l2_bypass_sectors_exact": counters.get("l2_by_source", {})
        .get("bypass", {})
        .get("sectors")
        == copies * 56,
    }
    contract = production_contract(launch)
    if not all(contract.values()):
        raise AnalysisError(f"{name}: production/tail contract failed: {contract}")
    return {
        "cell": name,
        "variant": variant,
        "depth": depth,
        "lead_samples": lead,
        "schema_version": 18,
        "geometry_expected": exact,
        "geometry_observed": observed_geometry,
        "geometry_gates": geometry_gates,
        "work_gates": work_gates,
        "production_contract": contract,
        "production_fingerprint": production_fingerprint(launch),
        "cohort_response_envelope_cycles": observation[
            "cohort_response_envelope_cycles"
        ],
        "cohort_topology_prearrival_hidden_cycles": observation[
            "cohort_topology_prearrival_hidden_cycles"
        ],
        "cohort_topology_stream_hidden_cycles": observation[
            "cohort_topology_stream_hidden_cycles"
        ],
        "cohort_topology_residual_exposure_cycles": observation[
            "cohort_topology_residual_exposure_cycles"
        ],
        "replay_sha256": sha256(first),
    }


def direct_cell(
    root: Path,
    depth: int,
    lead: int,
    repetitions: Sequence[int],
    binary: Mapping[str, Any],
) -> dict[str, Any]:
    name = pair_cell_name(depth, lead)
    payloads: list[dict[str, Any]] = []
    hashes: list[str] = []
    for repetition in repetitions:
        directory = root / "direct" / name / f"repeat-{repetition}"
        path = directory / "result.json"
        if not path.is_file():
            raise FileNotFoundError(str(path))
        try:
            execution_capture.validate_direct(
                directory,
                execution_capture.Cell(depth, lead),
                repetition,
                binary,
            )
        except execution_capture.CaptureError as error:
            raise AnalysisError(f"{directory}: {error}") from error
        payload = read_json(path)
        validate_target_payload(
            payload,
            depth=depth,
            lead=lead,
            groups=DIRECT_GROUPS,
            mode="pair",
            order=expected_order(repetition),
        )
        expensive = payload["span_expensive"]
        cheap = payload["span_cheap"]
        expected_wait = (expensive["wait_cycles"] - cheap["wait_cycles"]) / DIRECT_GROUPS
        expected_issue = (expensive["issue_cycles"] - cheap["issue_cycles"]) / DIRECT_GROUPS
        if not math.isclose(
            float(payload.get("paired_wait_delta_cycles_per_group")),
            expected_wait,
            rel_tol=0.0,
            abs_tol=1.0e-8,
        ):
            raise AnalysisError(f"{path}: paired wait delta mismatch")
        if not math.isclose(
            float(payload.get("paired_issue_delta_cycles_per_group")),
            expected_issue,
            rel_tol=0.0,
            abs_tol=1.0e-8,
        ):
            raise AnalysisError(f"{path}: paired issue delta mismatch")
        payloads.append(payload)
        hashes.append(sha256(path))
    issue = [float(payload["paired_issue_delta_cycles_per_group"]) for payload in payloads]
    wait = [float(payload["paired_wait_delta_cycles_per_group"]) for payload in payloads]
    return {
        "cell": name,
        "depth": depth,
        "lead_samples": lead,
        "repeat_count": len(payloads),
        "orders": [payload["order"] for payload in payloads],
        "issue_delta_cycles_per_group": issue,
        "wait_delta_cycles_per_group": wait,
        "median_issue_delta_cycles_per_group": statistics.median(issue),
        "median_wait_delta_cycles_per_group": statistics.median(wait),
        "issue_median_exact_zero": statistics.median(issue) == 0.0,
        "result_sha256": hashes,
    }


def sign(value: float) -> int:
    return 1 if value > 0 else (-1 if value < 0 else 0)


def comparison(
    spec: tuple[int, int],
    split: str,
    direct: Mapping[tuple[int, int], Mapping[str, Any]],
    simulator: Mapping[tuple[str, int, int], Mapping[str, Any]],
) -> dict[str, Any]:
    depth, lead = spec
    predicted = (
        int(simulator[("span_expensive", depth, lead)][
            "cohort_topology_residual_exposure_cycles"
        ])
        - int(simulator[("span_cheap", depth, lead)][
            "cohort_topology_residual_exposure_cycles"
        ])
    ) / TRACE_GROUPS
    observed = float(direct[spec]["median_wait_delta_cycles_per_group"])
    return {
        "depth": depth,
        "lead_samples": lead,
        "split": split,
        "observed_delta_cycles_per_group": observed,
        "predicted_delta_cycles_per_group": predicted,
        "signed_error": predicted - observed,
        "absolute_error": abs(predicted - observed),
        "prediction_nonzero": predicted != 0.0,
        "sign_correct_when_required": predicted == 0.0 or sign(predicted) == sign(observed),
    }


def score_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors = [float(row["absolute_error"]) for row in rows]
    return {
        "comparisons": len(rows),
        "mae": statistics.mean(errors),
        "maximum_absolute_error": max(errors),
        "all_nonzero_prediction_signs_correct": all(
            bool(row["sign_correct_when_required"]) for row in rows
        ),
    }


def split_passed(summary: Mapping[str, Any]) -> bool:
    return (
        float(summary["mae"]) <= MAE_LIMIT
        and float(summary["maximum_absolute_error"]) <= MAX_ERROR_LIMIT
        and bool(summary["all_nonzero_prediction_signs_correct"])
    )


def smoke_scientific_gates(
    bundle: Mapping[str, Any],
    simulator: Mapping[tuple[str, int, int], Mapping[str, Any]],
    direct: Mapping[str, Any],
) -> dict[str, bool]:
    expensive = simulator[("span_expensive", 1, 16)]
    cheap = simulator[("span_cheap", 1, 16)]
    envelope = int(expensive["cohort_response_envelope_cycles"])
    return {
        "ncu_attribution_valid": bool(bundle["valid"]),
        "trace_and_replay_geometry_exact": all(
            all(cell["geometry_gates"].values()) and all(cell["work_gates"].values())
            for cell in (expensive, cheap)
        ),
        "source_envelope_pair_equal": envelope
        == int(cheap["cohort_response_envelope_cycles"]),
        "envelope_within_18_179_cycles_per_copy": 18 * TRACE_GROUPS
        <= envelope
        <= 179 * TRACE_GROUPS,
        "prearrival_hidden_zero": expensive[
            "cohort_topology_prearrival_hidden_cycles"
        ]
        == 0
        and cheap["cohort_topology_prearrival_hidden_cycles"] == 0,
        "expensive_stream_hiding_saturated": expensive[
            "cohort_topology_stream_hidden_cycles"
        ]
        == 4 * 18 * TRACE_GROUPS,
        "cheap_stream_hiding_zero": cheap["cohort_topology_stream_hidden_cycles"] == 0,
        "expensive_residual_zero": expensive[
            "cohort_topology_residual_exposure_cycles"
        ]
        == 0,
        "cheap_residual_eighteen_per_group": cheap[
            "cohort_topology_residual_exposure_cycles"
        ]
        == 18 * TRACE_GROUPS,
        "paired_score_negative_eighteen": (
            expensive["cohort_topology_residual_exposure_cycles"]
            - cheap["cohort_topology_residual_exposure_cycles"]
        )
        / TRACE_GROUPS
        == -18,
        "direct_correct_and_issue_median_zero": bool(direct["issue_median_exact_zero"]),
    }


def full_replay_gates(
    simulator: Mapping[tuple[str, int, int], Mapping[str, Any]],
) -> dict[str, bool]:
    geometry = all(
        all(cell["geometry_gates"].values()) and all(cell["work_gates"].values())
        for cell in simulator.values()
    )
    paired_source = all(
        simulator[("span_expensive", depth, lead)]["cohort_response_envelope_cycles"]
        == simulator[("span_cheap", depth, lead)]["cohort_response_envelope_cycles"]
        for depth, lead in cell_specs()
    )
    monotone = True
    for depth in DEPTHS:
        for variant in VARIANTS:
            values = [
                simulator[(variant, depth, lead)]["cohort_response_envelope_cycles"]
                for lead in LEADS
            ]
            monotone = monotone and values == sorted(values)
    saturated = True
    for depth in DEPTHS:
        copies = TRACE_GROUPS * depth
        expensive = simulator[("span_expensive", depth, 16)]
        cheap = simulator[("span_cheap", depth, 16)]
        saturated = saturated and (
            18 * copies
            <= expensive["cohort_response_envelope_cycles"]
            <= 179 * copies
            and expensive["cohort_topology_prearrival_hidden_cycles"] == 0
            and cheap["cohort_topology_prearrival_hidden_cycles"] == 0
            and expensive["cohort_topology_stream_hidden_cycles"] == 4 * 18 * copies
            and cheap["cohort_topology_stream_hidden_cycles"] == 0
            and expensive["cohort_topology_residual_exposure_cycles"] == 0
            and cheap["cohort_topology_residual_exposure_cycles"]
            == 18 * TRACE_GROUPS
        )
    variant_production_equal = all(
        simulator[("span_expensive", depth, lead)]["production_fingerprint"]
        == simulator[("span_cheap", depth, lead)]["production_fingerprint"]
        for depth, lead in cell_specs()
    )
    return {
        "all_trace_replay_geometry_exact": geometry,
        "paired_source_envelopes_equal": paired_source,
        "lead_envelopes_monotone_nondecreasing": monotone,
        "lead16_saturated_all_depths": saturated,
        "paired_production_fingerprints_equal": variant_production_equal,
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    counts = report["counts"]
    lines = [
        "# SM86 async-copy multi-line response-envelope report",
        "",
        f"Status: `{report['status']}`",
        "",
        str(report["decision"]),
        "",
        "## Evidence coverage",
        "",
        "| Lane | Complete | Full expected |",
        "|---|---:|---:|",
        f"| Paired NCU bundles | {counts['ncu_bundles']} | 80 |",
        f"| Focused one-pass NCU reports | {counts['ncu_reports']} | 480 |",
        f"| Ordered NCU rows | {counts['ncu_rows']} | 960 |",
        f"| NVBit endpoint traces | {counts['nvbit_cells']} | 16 |",
        f"| Schema-18 replay twins | {counts['simulator_cells']} | 16 |",
        f"| Direct paired cells read | {counts['direct_cells_read']} | 8 |",
        "",
        "## Gates",
        "",
        "| Criterion | Result |",
        "|---|---|",
    ]
    for name, value in report["criteria"].items():
        shown = "pending" if value is None else ("pass" if value else "fail")
        lines.append(f"| `{name}` | {shown} |")
    smoke = report.get("smoke_scientific_gates")
    if isinstance(smoke, dict):
        lines.extend(["", "## Vertical smoke", "", "| Gate | Result |", "|---|---|"])
        for name, value in smoke.items():
            lines.append(f"| `{name}` | {'pass' if value else 'fail'} |")
    score = report.get("score")
    if isinstance(score, dict):
        lines.extend(
            [
                "",
                "## Frozen score",
                "",
                "| Split | Evaluated | MAE | Maximum error | Required signs correct |",
                "|---|---|---:|---:|---|",
            ]
        )
        identification = score["identification"]
        lines.append(
            f"| Identification | yes | {identification['mae']:.3f} | "
            f"{identification['maximum_absolute_error']:.3f} | "
            f"{'yes' if identification['all_nonzero_prediction_signs_correct'] else 'no'} |"
        )
        holdout = score.get("holdout")
        if isinstance(holdout, dict):
            lines.append(
                f"| Holdout | yes | {holdout['mae']:.3f} | "
                f"{holdout['maximum_absolute_error']:.3f} | "
                f"{'yes' if holdout['all_nonzero_prediction_signs_correct'] else 'no'} |"
            )
        else:
            lines.append("| Holdout | no | — | — | — |")
        lines.extend(
            [
                "",
                "| Depth | Lead | Split | Live delta | Predicted delta | Absolute error | Sign gate |",
                "|---:|---:|---|---:|---:|---:|---|",
            ]
        )
        for row in score["comparisons"]:
            lines.append(
                f"| {row['depth']} | {row['lead_samples']} | {row['split']} | "
                f"{row['observed_delta_cycles_per_group']:.3f} | "
                f"{row['predicted_delta_cycles_per_group']:.3f} | "
                f"{row['absolute_error']:.3f} | "
                f"{'pass' if row['sign_correct_when_required'] else 'fail'} |"
            )
    if report["missing_artifacts"]:
        lines.extend(["", "## Missing artifacts", ""])
        lines.extend(f"- `{item}`" for item in report["missing_artifacts"])
    if report["validation_errors"]:
        lines.extend(["", "## Validation errors", ""])
        lines.extend(f"- {item}" for item in report["validation_errors"])
    lines.extend(
        [
            "",
            "Schema 18 remains observational. Production asynchronous-copy timing is "
            "unchanged and the typed `async_copy_shared_visibility` tail remains required.",
            "",
        ]
    )
    return "\n".join(lines)


def select_outcome(
    *,
    errors: Sequence[str],
    missing: Sequence[str],
    structural_no_go: bool,
    smoke_complete: bool,
    smoke_passed: bool,
    smoke_only: bool,
    full_admitted: bool,
    score: Mapping[str, Any] | None,
) -> tuple[str, str, int]:
    if errors:
        return (
            "invalid_response_envelope_campaign",
            "One or more retained artifacts are malformed, stale, reordered, "
            "multi-pass, or violate production/tail integrity.",
            1,
        )
    if structural_no_go and not missing:
        return (
            "complete_response_envelope_structural_no_go",
            "A complete static, attribution, response-envelope, destination-"
            "wavefront, or issue-control gate rejects the assay before timing transfer.",
            0,
        )
    if missing or not smoke_complete:
        return (
            "incomplete_response_envelope_campaign",
            "Required smoke evidence is incomplete.",
            2,
        )
    if smoke_only:
        if smoke_passed:
            return (
                "response_envelope_smoke_passed",
                "The D1/lead-16 static, attribution, NVBit, replay, and direct "
                "vertical smoke passes every frozen scientific gate.",
                0,
            )
        return (
            "complete_response_envelope_structural_no_go",
            "The complete D1/lead-16 vertical smoke rejects a frozen structural gate.",
            0,
        )
    if not full_admitted or score is None:
        return (
            "incomplete_response_envelope_campaign",
            "Required full-matrix evidence is incomplete.",
            2,
        )
    if not score["identification_passed"]:
        return (
            "complete_response_envelope_identification_no_go",
            "The fixed score fails depth-1 identification; depth-2 timing remains unread.",
            0,
        )
    if not score["transfer_passed"]:
        return (
            "complete_response_envelope_holdout_no_go",
            "The fixed score passes identification but fails the untouched depth-2 holdout.",
            0,
        )
    return (
        "complete_response_envelope_transfer",
        "The fixed schema-18 score transfers within the frozen observational bounds; "
        "production timing remains unchanged in this milestone.",
        0,
    )


def deterministic_aggregate(report: Mapping[str, Any]) -> dict[str, Any]:
    smoke_geometry = []
    for cell in report.get("simulator_cells", []):
        if cell.get("depth") == 1 and cell.get("lead_samples") == 16:
            smoke_geometry.append(
                {
                    "variant": cell.get("variant"),
                    "geometry_expected": cell.get("geometry_expected"),
                    "geometry_observed": cell.get("geometry_observed"),
                    "geometry_gates": cell.get("geometry_gates"),
                    "work_gates": cell.get("work_gates"),
                    "cohort_response_envelope_cycles": cell.get(
                        "cohort_response_envelope_cycles"
                    ),
                    "cohort_topology_prearrival_hidden_cycles": cell.get(
                        "cohort_topology_prearrival_hidden_cycles"
                    ),
                    "cohort_topology_stream_hidden_cycles": cell.get(
                        "cohort_topology_stream_hidden_cycles"
                    ),
                    "cohort_topology_residual_exposure_cycles": cell.get(
                        "cohort_topology_residual_exposure_cycles"
                    ),
                }
            )
    smoke_geometry.sort(key=lambda value: str(value["variant"]))
    static = report.get("static")
    ncu_identities = [
        {
            "cell": item.get("cell"),
            "repetition": item.get("repetition"),
            "bundle_sha256": item.get("bundle_sha256"),
            "families": item.get("families"),
        }
        for item in report.get("ncu_bundles", [])
    ]
    nvbit_identities = [
        {
            key: item.get(key)
            for key in (
                "cell",
                "variant",
                "binary_sha256",
                "tracer_sha256",
                "trace_sha256",
                "binding_sha256",
                "complete_sha256",
            )
        }
        for item in report.get("nvbit_cells", [])
    ]
    replay_identities = [
        {
            key: item.get(key)
            for key in ("cell", "variant", "replay_sha256")
        }
        for item in report.get("simulator_cells", [])
    ]
    direct_identities = [
        {
            key: item.get(key)
            for key in ("cell", "depth", "lead_samples", "result_sha256")
        }
        for item in report.get("direct_cells", [])
    ]
    ncu_identities.sort(
        key=lambda item: (str(item["cell"]), int(item["repetition"] or 0))
    )
    nvbit_identities.sort(
        key=lambda item: (str(item["cell"]), str(item["variant"]))
    )
    replay_identities.sort(
        key=lambda item: (str(item["cell"]), str(item["variant"]))
    )
    direct_identities.sort(key=lambda item: str(item["cell"]))
    return {
        "schema_version": 1,
        "assay": report.get("assay"),
        "mode": report.get("mode"),
        "status": report.get("status"),
        "decision": report.get("decision"),
        "timing_promotion_authorized": False,
        "typed_visibility_tail_required": True,
        "binary": report.get("binary"),
        "counts": report.get("counts"),
        "static_gate": (
            {
                "artifact_valid": static.get("artifact_valid"),
                "assay_valid": static.get("assay_valid"),
                "scientific_failures": static.get("scientific_failures"),
                "artifacts": static.get("artifacts"),
            }
            if isinstance(static, dict)
            else None
        ),
        "smoke_scientific_gates": report.get("smoke_scientific_gates"),
        "smoke_replay_geometry": smoke_geometry,
        "evidence_identities": {
            "direct": direct_identities,
            "ncu": ncu_identities,
            "nvbit": nvbit_identities,
            "replay": replay_identities,
        },
        "full_replay_gates": report.get("full_replay_gates"),
        "score": report.get("score"),
        "criteria": report.get("criteria"),
        "missing_artifact_count": len(report.get("missing_artifacts", [])),
        "validation_error_count": len(report.get("validation_errors", [])),
    }


def run(
    root: Path,
    output_json: Path,
    output_md: Path,
    *,
    smoke_only: bool = False,
    aggregate_output: Path | None = None,
) -> int:
    root = root.resolve()
    missing: list[str] = []
    errors: list[str] = []

    binary: dict[str, Any] | None = None
    static: dict[str, Any] | None = None
    try:
        binary = binary_report(root)
    except FileNotFoundError as error:
        missing.append(str(error))
    except (AnalysisError, OSError) as error:
        errors.append(str(error))
    try:
        static = static_report(root)
        errors.extend(static["artifact_errors"])
    except FileNotFoundError as error:
        missing.append(str(error))
    except (AnalysisError, OSError, KeyError, TypeError, ValueError) as error:
        errors.append(str(error))

    ncu: dict[tuple[int, int, int], dict[str, Any]] = {}
    nvbit: dict[tuple[str, int, int], dict[str, Any]] = {}
    simulator: dict[tuple[str, int, int], dict[str, Any]] = {}
    direct: dict[tuple[int, int], dict[str, Any]] = {}
    smoke_gates: dict[str, bool] | None = None
    full_gates: dict[str, bool] | None = None
    score: dict[str, Any] | None = None

    # A fully decoded compiler/SASS failure is itself a complete structural
    # result and intentionally short-circuits all live collection requirements.
    static_structural_no_go = (
        static is not None
        and static["artifact_valid"]
        and not static["assay_valid"]
    )

    if not errors and not missing and not static_structural_no_go and binary is not None:
        depth, lead = SMOKE_SPEC
        try:
            smoke_bundle = validate_bundle(root, depth, lead, 1, binary)
            ncu[(depth, lead, 1)] = smoke_bundle
            for variant in VARIANTS:
                nvbit[(variant, depth, lead)] = nvbit_cell(
                    root, variant, depth, lead, str(binary["sha256"])
                )
                simulator[(variant, depth, lead)] = replay_cell(root, variant, depth, lead)
            require_cross_endpoint_tracer_identity(list(nvbit.values()))
            smoke_direct = direct_cell(root, depth, lead, (1,), binary)
            direct[(depth, lead)] = smoke_direct
            smoke_gates = smoke_scientific_gates(smoke_bundle, simulator, smoke_direct)
        except FileNotFoundError as error:
            missing.append(str(error))
        except (AnalysisError, OSError, KeyError, TypeError, ValueError) as error:
            errors.append(str(error))

    smoke_complete = smoke_gates is not None and not errors and not missing
    smoke_passed = smoke_complete and all(smoke_gates.values())
    structural_no_go = static_structural_no_go or (smoke_complete and not smoke_passed)

    if smoke_passed and not smoke_only and binary is not None:
        # Complete every attribution/trace/replay lane before direct live labels.
        for depth, lead in cell_specs():
            for repetition in REPETITIONS:
                key = (depth, lead, repetition)
                if key in ncu:
                    continue
                try:
                    ncu[key] = validate_bundle(root, depth, lead, repetition, binary)
                except FileNotFoundError as error:
                    missing.append(str(error))
                except (AnalysisError, OSError, KeyError, TypeError, ValueError) as error:
                    errors.append(str(error))
            for variant in VARIANTS:
                key = (variant, depth, lead)
                if key not in nvbit:
                    try:
                        nvbit[key] = nvbit_cell(
                            root, variant, depth, lead, str(binary["sha256"])
                        )
                    except FileNotFoundError as error:
                        missing.append(str(error))
                    except (AnalysisError, OSError, KeyError, TypeError, ValueError) as error:
                        errors.append(str(error))
                if key not in simulator:
                    try:
                        simulator[key] = replay_cell(root, variant, depth, lead)
                    except FileNotFoundError as error:
                        missing.append(str(error))
                    except (AnalysisError, OSError, KeyError, TypeError, ValueError) as error:
                        errors.append(str(error))
        ncu_complete = len(ncu) == 80
        nvbit_complete = len(nvbit) == 16
        simulator_complete = len(simulator) == 16
        if ncu_complete and nvbit_complete and simulator_complete and not errors and not missing:
            try:
                require_cross_endpoint_tracer_identity(list(nvbit.values()))
            except AnalysisError as error:
                errors.append(str(error))
        if ncu_complete and nvbit_complete and simulator_complete and not errors and not missing:
            full_gates = full_replay_gates(simulator)
            full_gates["all_ncu_attribution_bundles_valid"] = all(
                bundle["valid"] for bundle in ncu.values()
            )
            full_gates["all_nvbit_one_binary"] = (
                len({cell["binary_sha256"] for cell in nvbit.values()}) == 1
            )
            structural_no_go = not all(full_gates.values())

    # The full campaign is expected on disk before scoring, but depth-2 JSON is
    # not opened until depth-1 identification passes.
    full_admitted = (
        smoke_passed
        and full_gates is not None
        and all(full_gates.values())
        and len(ncu) == 80
        and len(nvbit) == 16
        and len(simulator) == 16
        and not errors
        and not missing
    )
    if full_admitted:
        holdout_paths_present = all(
            (
                root
                / "direct"
                / pair_cell_name(2, lead)
                / f"repeat-{repetition}/result.json"
            ).is_file()
            for lead in LEADS
            for repetition in REPETITIONS
        )
        if not holdout_paths_present:
            missing.append("depth-2 direct holdout matrix is incomplete")
        for spec in IDENTIFICATION_SPECS:
            try:
                direct[spec] = direct_cell(root, *spec, REPETITIONS, binary)
            except FileNotFoundError as error:
                missing.append(str(error))
            except (AnalysisError, OSError, KeyError, TypeError, ValueError) as error:
                errors.append(str(error))
        if not errors and not missing:
            issue_gate = all(direct[spec]["issue_median_exact_zero"] for spec in IDENTIFICATION_SPECS)
            if not issue_gate:
                structural_no_go = True
            else:
                identification_rows = [
                    comparison(spec, "identification", direct, simulator)
                    for spec in IDENTIFICATION_SPECS
                ]
                identification = score_summary(identification_rows)
                identification_passed = split_passed(identification)
                score = {
                    "model": (
                        "(residual_span_expensive - residual_span_cheap) / groups"
                    ),
                    "fitted_parameters": 0,
                    "identification": identification,
                    "identification_passed": identification_passed,
                    "holdout_evaluated": False,
                    "holdout": None,
                    "transfer_passed": False,
                    "comparisons": identification_rows,
                }
                if identification_passed:
                    for spec in HOLDOUT_SPECS:
                        try:
                            direct[spec] = direct_cell(root, *spec, REPETITIONS, binary)
                        except FileNotFoundError as error:
                            missing.append(str(error))
                        except (
                            AnalysisError,
                            OSError,
                            KeyError,
                            TypeError,
                            ValueError,
                        ) as error:
                            errors.append(str(error))
                    if not errors and not missing:
                        holdout_issue = all(
                            direct[spec]["issue_median_exact_zero"] for spec in HOLDOUT_SPECS
                        )
                        if not holdout_issue:
                            structural_no_go = True
                        else:
                            holdout_rows = [
                                comparison(spec, "holdout", direct, simulator)
                                for spec in HOLDOUT_SPECS
                            ]
                            holdout = score_summary(holdout_rows)
                            score.update(
                                {
                                    "holdout_evaluated": True,
                                    "holdout": holdout,
                                    "transfer_passed": split_passed(holdout),
                                    "comparisons": identification_rows + holdout_rows,
                                }
                            )

    status, decision, exit_code = select_outcome(
        errors=errors,
        missing=missing,
        structural_no_go=structural_no_go,
        smoke_complete=smoke_complete,
        smoke_passed=smoke_passed,
        smoke_only=smoke_only,
        full_admitted=full_admitted,
        score=score,
    )

    criteria: dict[str, bool | None] = {
        "probe_binary_exact": binary is not None,
        "static_artifacts_valid": static.get("artifact_valid") if static else None,
        "static_assay_gate_passed": static.get("assay_valid") if static else None,
        "vertical_smoke_complete": smoke_complete,
        "vertical_smoke_scientific_gate_passed": smoke_passed if smoke_complete else None,
        "full_80_bundle_attribution_admitted": (None if smoke_only else (
            len(ncu) == 80 and all(bundle["valid"] for bundle in ncu.values())
            if len(ncu) == 80
            else None
        )),
        "full_16_nvbit_traces_complete": None if smoke_only else len(nvbit) == 16,
        "full_16_schema18_twins_complete": None if smoke_only else len(simulator) == 16,
        "full_replay_scientific_gates_passed": (
            None
            if smoke_only
            else (all(full_gates.values()) if full_gates is not None else None)
        ),
        "attribution_preceded_live_labels": True,
        "identification_within_limits": (
            score["identification_passed"] if score is not None else None
        ),
        "holdout_within_limits": (
            score["transfer_passed"]
            if score is not None and score["holdout_evaluated"]
            else None
        ),
        "production_timing_promotion_disabled": True,
        "typed_visibility_tail_required": True,
    }
    report = {
        "schema_version": 1,
        "assay": "sm86_async_multiline_response_envelope",
        "mode": "smoke_only" if smoke_only else "full",
        "root": str(root),
        "status": status,
        "decision": decision,
        "timing_promotion_authorized": False,
        "typed_visibility_tail_required": True,
        "binary": binary,
        "static": static,
        "counts": {
            "ncu_bundles": len(ncu),
            "ncu_reports": len(ncu) * 6,
            "ncu_rows": len(ncu) * 12,
            "nvbit_cells": len(nvbit),
            "simulator_cells": len(simulator),
            "simulator_outputs": len(simulator) * 2,
            "direct_cells_read": len(direct),
            "direct_results_read": sum(cell["repeat_count"] for cell in direct.values()),
        },
        "smoke_scientific_gates": smoke_gates,
        "full_replay_gates": full_gates,
        "ncu_bundles": [ncu[key] for key in sorted(ncu)],
        "nvbit_cells": [nvbit[key] for key in sorted(nvbit)],
        "simulator_cells": [simulator[key] for key in sorted(simulator)],
        "direct_cells": [direct[key] for key in sorted(direct)],
        "score": score,
        "criteria": criteria,
        "missing_artifacts": sorted(set(missing)),
        "validation_errors": errors,
        "interface_assumptions": {
            "static_root": "static/{control_bits.json,resources.txt,sass.txt,build.log}",
            "direct_root": "direct/async_multiline_response_dD_lL_g8192/repeat-R/result.json",
            "ncu_root": (
                "paired_ncu/async_multiline_response_dD_lL_g64/repeat-R/"
                "{global_ld,ldgsts,lts_total,lts_hit,lts_miss,dram}"
            ),
            "nvbit_root": (
                "nvbit/async_multiline_response_{span_expensive|span_cheap}_dD_lL_g64"
            ),
            "simulator_root": (
                "simulator/async_multiline_response_{span_expensive|span_cheap}_dD_lL_g64"
            ),
            "schema17_normalization": "not_applicable_new_trace_no_predecessor",
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_md.write_text(markdown_report(report), encoding="utf-8")
    if aggregate_output is not None:
        aggregate_output.parent.mkdir(parents=True, exist_ok=True)
        aggregate_output.write_text(
            json.dumps(deterministic_aggregate(report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_md": str(output_md),
                "aggregate_output": (
                    str(aggregate_output) if aggregate_output is not None else None
                ),
                "status": status,
                "counts": report["counts"],
                "missing": len(report["missing_artifacts"]),
                "errors": len(errors),
                "timing_promotion_authorized": False,
            },
            sort_keys=True,
        )
    )
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="consume only static plus D1/lead-16 repeat-1 smoke artifacts",
    )
    parser.add_argument("--output", "--output-json", dest="output_json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument(
        "--aggregate-output",
        type=Path,
        help="optional deterministic compact JSON suitable for a tracked path",
    )
    args = parser.parse_args(argv)
    output_json = args.output_json or args.root / "analysis/report.json"
    output_md = args.output_md or output_json.with_suffix(".md")
    try:
        return run(
            args.root,
            output_json,
            output_md,
            smoke_only=args.smoke_only,
            aggregate_output=args.aggregate_output,
        )
    except (AnalysisError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"multiline-response analysis: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
