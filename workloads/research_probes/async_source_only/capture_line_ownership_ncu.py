#!/usr/bin/env python3
"""Capture paired NCU evidence for the SM86 source-line ownership assay.

The frozen matrix has two early-line orientations and ten repetitions. Odd
repetitions run early_fanout then late_fanout; even repetitions reverse that
order. Seven metrics are collected in separate one-pass application replays.
Raw DRAM remains a non-gating background sentinel.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import capture_multiline_response as support
import line_ownership_campaign_contract as campaign_contract
import capture_line_ownership_execution as execution_support


REPO_ROOT = support.REPO_ROOT
DEFAULT_ROOT = campaign_contract.V1_CONTRACT.evidence_root
BINARY = campaign_contract.V1_CONTRACT.binary

GROUPS = 64
ORIENTATIONS = campaign_contract.V1_CONTRACT.orientations
REPETITIONS = tuple(range(1, 11))
VARIANTS = campaign_contract.V1_CONTRACT.variants
GPU_INDEX = "0"
EXPECTED_GPU = dict(support.EXPECTED_GPU)
DEFAULT_TIMEOUT = 900

FAMILY_METRICS = {
    "global_ld": "smsp__inst_executed_op_global_ld.sum",
    "ldgsts": "smsp__inst_executed_op_ldgsts.sum",
    "shared_wavefronts": (
        "sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum"
    ),
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


class CampaignError(RuntimeError):
    """Raised for malformed, stale, incomplete, or failed tool evidence."""


def active_contract() -> campaign_contract.CampaignContract:
    return campaign_contract.current_contract()


def configure_contract(
    contract: campaign_contract.CampaignContract,
) -> campaign_contract.CampaignContract:
    """Select one immutable generation before any campaign action."""

    configured = campaign_contract.configure_once(contract)
    execution_support.configure_campaign(configured)
    return configured


def _coordinate_artifact(path: Path) -> bool:
    contract = active_contract()
    if contract == campaign_contract.V1_CONTRACT:
        return False
    candidate = contract.require_allowed_path(path)
    root = contract.require_allowed_path(contract.evidence_root)
    aggregate = contract.require_allowed_path(contract.aggregate_report)
    return candidate == aggregate or candidate == root or root in candidate.parents


def same_json(observed: Any, expected: Any) -> bool:
    if active_contract() == campaign_contract.V1_CONTRACT:
        return observed == expected
    return campaign_contract.strict_json_equal(observed, expected)


def read_json(path: Path) -> dict[str, Any]:
    if _coordinate_artifact(path):
        return campaign_contract.read_artifact_json(active_contract(), path)
    return support.read_json(path)


def read_text(path: Path) -> str:
    if _coordinate_artifact(path):
        return campaign_contract.read_artifact_text(active_contract(), path)
    return path.read_text(encoding="utf-8")


def sha256(path: Path) -> str:
    if _coordinate_artifact(path):
        return campaign_contract.artifact_identity(active_contract(), path)["sha256"]
    return support.sha256(path)


def file_identity(path: Path, family_directory: Path) -> dict[str, Any]:
    if _coordinate_artifact(path):
        payload = campaign_contract.read_artifact_bytes(active_contract(), path)
        return {
            "path": str(path.relative_to(family_directory)),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    return support.file_identity(path, family_directory)


def optional_file_identity(
    path: Path, family_directory: Path
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return file_identity(path, family_directory)


def write_text(path: Path, value: str) -> None:
    if _coordinate_artifact(path):
        campaign_contract.replace_artifact_text(active_contract(), path, value)
    else:
        support.atomic_write_text(path, value)


def write_json(path: Path, value: Any) -> None:
    if _coordinate_artifact(path):
        campaign_contract.replace_artifact_json(active_contract(), path, value)
    else:
        support.atomic_write_json(path, value)


def ensure_output_directory(path: Path) -> None:
    if _coordinate_artifact(path):
        try:
            campaign_contract.ensure_directory(path)
        except campaign_contract.ImmutableArtifactError as error:
            raise CampaignError(f"cannot create coordinate directory {path}: {error}") from error
    else:
        path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Cell:
    orientation: str

    @property
    def key(self) -> str:
        contract = active_contract()
        return f"{contract.binary.name}_{self.orientation}_g{contract.trace_groups}"


def expected_order(repetition: int) -> str:
    return "ab" if repetition % 2 else "ba"


def is_smoke_bundle(cell: Cell, repetition: int) -> bool:
    return cell.orientation == active_contract().orientations[0] and repetition == 1


def row_variants(order: str) -> tuple[str, str]:
    variants = active_contract().variants
    return variants if order == "ab" else tuple(reversed(variants))


def expected_instruction_count() -> int:
    contract = active_contract()
    return contract.scalar_ldg_instructions_per_copy * contract.trace_groups


def expected_payload_wavefronts() -> int:
    contract = active_contract()
    return contract.ncu_shared_wavefronts_per_copy * contract.trace_groups


def expected_lts_total() -> int:
    contract = active_contract()
    return contract.local_device_lts_lookups_per_copy * contract.trace_groups


def paired_tolerance() -> int:
    return max(4, math.ceil(0.01 * expected_lts_total()))


def require_full_campaign_authorization(
    root: Path, binary: Mapping[str, Any]
) -> dict[str, Any]:
    contract = active_contract()
    contract.require_allowed_path(root)
    retained_authorization = root / "analysis/smoke-authorization.json"
    report_path = (
        contract.require_allowed_path(retained_authorization)
        if retained_authorization.is_file()
        else contract.require_allowed_path(contract.aggregate_report)
    )
    try:
        report = campaign_contract.read_artifact_json(contract, report_path)
    except campaign_contract.ContractError as error:
        raise CampaignError(
            f"full campaign requires a smoke-pass aggregate: {report_path}: {error}"
        ) from error
    try:
        return campaign_contract.validate_smoke_authorization(
            report,
            contract=contract,
            root=root,
            binary=binary,
        )
    except campaign_contract.ContractError as error:
        raise CampaignError(
            f"full campaign is not authorized by this root/binary smoke: {error}"
        ) from error


def retained_binary(root: Path) -> dict[str, Any]:
    value = read_json(root / "static/command.json").get("binary")
    if not isinstance(value, dict):
        raise CampaignError("static command record has no binary identity")
    if (
        not isinstance(value.get("path"), str)
        or not isinstance(value.get("sha256"), str)
        or type(value.get("size_bytes")) is not int
    ):
        raise CampaignError("static command record has malformed binary identity")
    return dict(value)


def preflight(root: Path, ncu: str) -> support.Preflight:
    contract = active_contract()
    binary = contract.binary
    root = contract.require_allowed_path(root)
    if contract != campaign_contract.V1_CONTRACT:
        if root != contract.require_allowed_path(contract.evidence_root):
            raise CampaignError(
                f"generation {contract.generation_id} requires root "
                f"{contract.evidence_root}, got {root}"
            )
        contract_record = root / "campaign_contract.json"
        try:
            retained_contract = campaign_contract.read_artifact_json(
                contract, contract_record
            )
        except campaign_contract.ContractError as error:
            raise CampaignError(
                f"{contract_record}: campaign contract identity mismatch"
            ) from error
        if not campaign_contract.strict_json_equal(
            retained_contract, contract.to_json()
        ):
            raise CampaignError(f"{contract_record}: campaign contract identity mismatch")
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise CampaignError(f"probe binary is missing or not executable: {binary}")
    static_command = read_json(root / "static/command.json")
    if contract != campaign_contract.V1_CONTRACT:
        static_binding = static_command.get("binding")
        if (
            not isinstance(static_binding, dict)
            or not campaign_contract.strict_json_equal(
                static_binding.get("campaign_contract"), contract.to_json()
            )
        ):
            raise CampaignError("static evidence is not bound to the active contract")
    try:
        execution_support.validate_campaign_contract_binding(root)
        execution_support.support.validate_static(root)
    except execution_support.CaptureError as error:
        raise CampaignError(f"static evidence validation failed: {error}") from error
    retained_binary = static_command.get("binary")
    if not isinstance(retained_binary, dict):
        raise CampaignError("static command record has no binary identity")
    expected_path = str(binary)
    if retained_binary.get("path") != expected_path:
        raise CampaignError(
            f"static binary path {retained_binary.get('path')!r} != {expected_path!r}"
        )
    expected_hash = retained_binary.get("sha256")
    expected_size = retained_binary.get("size_bytes")
    if not isinstance(expected_hash, str) or type(expected_size) is not int:
        raise CampaignError("static command record has malformed binary identity")
    observed_hash = sha256(binary)
    if observed_hash != expected_hash or binary.stat().st_size != expected_size:
        raise CampaignError("probe binary differs from static evidence")

    ncu_found = shutil.which(ncu)
    if ncu_found is None:
        raise CampaignError(f"NCU executable is unavailable: {ncu}")
    ncu_path = str(Path(ncu_found).resolve())
    version = support.run_command([ncu_path, "--version"], timeout=30)
    if version.returncode != 0:
        raise CampaignError(f"NCU version query failed: {version.stderr.strip()}")

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise CampaignError("nvidia-smi is unavailable")
    queried = support.run_command(
        [
            nvidia_smi,
            "-i",
            GPU_INDEX,
            "--query-gpu=index,name,uuid,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ],
        timeout=30,
    )
    if queried.returncode != 0:
        raise CampaignError(f"GPU identity query failed: {queried.stderr.strip()}")
    try:
        rows = list(csv.reader(io.StringIO(queried.stdout), skipinitialspace=True))
    except csv.Error as error:
        raise CampaignError(f"invalid GPU identity output: {error}") from error
    if len(rows) != 1 or len(rows[0]) != 5:
        raise CampaignError(f"expected one five-field GPU identity row: {rows!r}")
    device = dict(zip(EXPECTED_GPU, (item.strip() for item in rows[0]), strict=True))
    if not same_json(device, EXPECTED_GPU):
        raise CampaignError(f"GPU identity mismatch: {device} != {EXPECTED_GPU}")

    return support.Preflight(
        binary={
            "path": expected_path,
            "sha256": observed_hash,
            "size_bytes": expected_size,
        },
        device=device,
        ncu_path=ncu_path,
        ncu_version=(version.stdout + version.stderr).strip(),
        environment={
            "CUDA_VISIBLE_DEVICES": GPU_INDEX,
            "LANG": "C",
            "LC_ALL": "C",
        },
    )


def target_arguments(cell: Cell, repetition: int) -> list[str]:
    return [
        "--groups",
        str(active_contract().trace_groups),
        "--orientation",
        cell.orientation,
        "--order",
        expected_order(repetition),
        "--markers",
        "on",
    ]


def profile_command(
    context: support.Preflight,
    family_directory: Path,
    cell: Cell,
    repetition: int,
    family: str,
) -> list[str]:
    return [
        context.ncu_path,
        "--devices",
        GPU_INDEX,
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
        "regex:source_line_ownership_kernel",
        "--metrics",
        FAMILY_METRICS[family],
        "--export",
        str(family_directory / "profile"),
        "--force-overwrite",
        str(active_contract().binary),
        *target_arguments(cell, repetition),
    ]


def import_command(
    context: support.Preflight, family_directory: Path
) -> list[str]:
    return [
        context.ncu_path,
        "--import",
        str(family_directory / "profile.ncu-rep"),
        "--page",
        "raw",
        "--csv",
        "--print-units",
        "base",
        "--print-fp",
    ]


def command_document(
    context: support.Preflight,
    family_directory: Path,
    cell: Cell,
    repetition: int,
    family: str,
    timeout: int,
) -> dict[str, Any]:
    order = expected_order(repetition)
    contract = active_contract()
    document = {
        "schema_version": 1,
        "cell": cell.key,
        "orientation": cell.orientation,
        "groups": contract.trace_groups,
        "fixed_depth": contract.fixed_depth,
        "fixed_lead_samples": contract.fixed_lead_samples,
        "repetition": repetition,
        "order": order,
        "row_variants": list(row_variants(order)),
        "family": family,
        "metric": FAMILY_METRICS[family],
        "timeout_seconds": timeout,
        "binary": dict(context.binary),
        "device": dict(context.device),
        "ncu": {"path": context.ncu_path, "version": context.ncu_version},
        "environment_overrides": dict(context.environment),
        "target_arguments": target_arguments(cell, repetition),
        "profile_command": profile_command(
            context, family_directory, cell, repetition, family
        ),
        "import_command": import_command(context, family_directory),
    }
    if contract != campaign_contract.V1_CONTRACT:
        document["campaign_contract"] = contract.to_json()
        if not is_smoke_bundle(cell, repetition):
            authorization = execution_support.smoke_authorization_path(
                family_directory.parents[3]
            )
            if not authorization.is_file():
                raise CampaignError(
                    f"full NCU artifact requires smoke authorization: {authorization}"
                )
            document["smoke_authorization"] = campaign_contract.artifact_identity(
                contract, authorization
            )
    return document


def expected_target_output(
    cell: Cell,
    repetition: int,
    context: support.Preflight,
) -> dict[str, Any]:
    contract = active_contract()
    return {
        "workload": active_contract().workload,
        "route": "l2",
        "mode": "pair",
        "order": expected_order(repetition),
        "orientation": cell.orientation,
        "groups": active_contract().trace_groups,
        "warmup_groups": contract.warmup_groups,
        "depth": contract.fixed_depth,
        "lead_clock_samples": contract.fixed_lead_samples,
        "source_cohorts_per_copy": contract.source_cohorts_per_copy,
        "aligned_source_cohorts_per_copy": contract.aligned_source_cohorts_per_copy,
        "split_source_cohorts_per_copy": contract.split_source_cohorts_per_copy,
        "source_sectors_per_cohort": 4,
        "payload_bytes_per_cohort": 64,
        "source_step_stride_bytes": 2048,
        "source_step_max_exclusive_byte": 1968,
        "source_step_guard_bytes": 80,
        "scalar_ldg_instructions_per_copy": contract.scalar_ldg_instructions_per_copy,
        "scalar_ldg_sectors_per_copy": contract.scalar_ldg_sectors_per_copy,
        "scalar_preload_lines_per_copy": contract.scalar_preload_lines_per_copy,
        "ldgsts_instructions_per_copy": contract.ldgsts_instructions_per_copy,
        "ldgsts_sectors_per_copy": contract.ldgsts_sectors_per_copy,
        "unique_ldgsts_lines_per_copy": contract.unique_ldgsts_lines_per_copy,
        "payload_wavefronts_per_copy": contract.payload_wavefronts_per_copy,
        "projected_bypass_wavefronts_per_copy": contract.bypass_wavefronts_per_copy,
        "destination_lines_per_copy": contract.destination_lines_per_copy,
        "max_cohort_bank_wavefronts": contract.max_cohort_bank_wavefronts,
        "max_cohort_bypass_wavefronts": contract.max_cohort_bypass_wavefronts,
        "cohort_topology_penalty_cycles_per_copy": (
            contract.cohort_topology_penalty_cycles_per_copy
        ),
        "response_memberships_per_copy": contract.response_memberships_per_copy,
        "local_device_lts_lookups_per_copy": (
            contract.local_device_lts_lookups_per_copy
        ),
        "source_ring_bytes_per_arm": 32 << 20,
        "source_ring_steps_per_arm": 16384,
        "max_admitted_groups": 16320,
        "source_ring_no_wrap": True,
        "source_pool_bytes": 64 << 20,
        "source_pool_alignment_bytes": 128,
        "source_arm_a_offset_bytes": 0,
        "source_arm_b_offset_bytes": 32 << 20,
        "source_pool_single_allocation": True,
        "independent_source_arms": True,
        "source_vector_preserved": True,
        "preload_vector_preserved": True,
        "destination_multiset_preserved": True,
        "ownership_permutation_preserved": True,
        "static_ldgsts_positions_preserved": True,
        "result_store_after_intervals": True,
        "correct": True,
        "device": context.device["name"],
        "compute_capability": context.device["compute_capability"],
    }


def validate_target_output(
    text: str,
    cell: Cell,
    repetition: int,
    context: support.Preflight,
) -> dict[str, Any]:
    contract = active_contract()
    expected = expected_target_output(cell, repetition, context)
    payloads: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.lstrip().startswith("{"):
            continue
        if contract != campaign_contract.V1_CONTRACT:
            try:
                value = execution_support.select_structural_target_json(
                    line,
                    expected,
                    mode="pair",
                    context=f"NCU target stdout:{line_number}",
                )
            except execution_support.StructuralTargetError as error:
                raise CampaignError(f"invalid target JSON: {error}") from error
        else:
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise CampaignError(f"invalid target JSON: {error}") from error
        if isinstance(value, dict):
            payloads.append(value)
    if len(payloads) != 1:
        raise CampaignError(
            f"expected one target JSON from one replay pass, got {len(payloads)}"
        )
    payload = payloads[0]
    for field, expected_value in expected.items():
        if not same_json(payload.get(field), expected_value):
            raise CampaignError(
                f"target {field}={payload.get(field)!r}, expected {expected_value!r}"
            )
    for variant in active_contract().variants:
        result = payload.get(variant)
        if not isinstance(result, dict) or result.get("correct") is not True:
            raise CampaignError(f"target result is not correct for {variant}")
        if not same_json(result.get("mismatches"), 0):
            raise CampaignError(f"target reported mismatches for {variant}")
    return payload


def expected_kernel_fragment(cell: Cell) -> str:
    return (
        "source_line_ownership_kernel<1>"
        if cell.orientation == "first_line_early"
        else "source_line_ownership_kernel<0>"
    )


def validate_rows(
    rows: Sequence[Mapping[str, str]],
    cell: Cell,
    repetition: int,
    family: str,
    context: support.Preflight,
) -> dict[str, Any]:
    label = f"{cell.key} repeat-{repetition}/{family}"
    if len(rows) != 2:
        raise CampaignError(f"{label}: expected two target rows, got {len(rows)}")
    ids = [str(row.get("ID", "")).strip() for row in rows]
    if ids != ["0", "1"]:
        raise CampaignError(f"{label}: row order is {ids}, expected 0/1")
    process_ids = [str(row.get("Process ID", "")).strip() for row in rows]
    if not process_ids[0] or process_ids[0] != process_ids[1]:
        raise CampaignError(f"{label}: rows are not one application process")

    variants = row_variants(expected_order(repetition))
    metric = FAMILY_METRICS[family]
    values: dict[str, int] = {}
    summaries: list[dict[str, Any]] = []
    for index, (row, variant) in enumerate(zip(rows, variants, strict=True)):
        row_label = f"{label} row {index}"
        kernel = str(row.get("Kernel Name", ""))
        if expected_kernel_fragment(cell) not in kernel.replace(" ", ""):
            raise CampaignError(f"{row_label}: wrong kernel {kernel}")
        expected_strings = {
            "Process Name": active_contract().binary.name,
            "Device": GPU_INDEX,
            "CC": context.device["compute_capability"],
            "Grid Size": "(1, 1, 1)",
            "Block Size": "(32, 1, 1)",
            "device__attribute_display_name": context.device["name"],
        }
        for field, expected_value in expected_strings.items():
            observed = str(row.get(field, "")).strip()
            if observed != expected_value:
                raise CampaignError(
                    f"{row_label}: {field}={observed!r}, expected {expected_value!r}"
                )
        numeric_identity = {
            "device__attribute_device_index": 0,
            "device__attribute_compute_capability_major": 8,
            "device__attribute_compute_capability_minor": 6,
            "launch__grid_size": 1,
            "launch__block_size": 32,
            "profiler__replayer_passes": 1,
        }
        for field, expected_value in numeric_identity.items():
            observed = support.parse_integer(row, field, row_label)
            if observed != expected_value:
                raise CampaignError(
                    f"{row_label}: {field}={observed}, expected {expected_value}"
                )
        value = support.parse_integer(row, metric, row_label)
        values[variant] = value
        summaries.append(
            {
                "id": index,
                "variant": variant,
                "process_id": process_ids[index],
                "kernel": kernel,
                "context": str(row.get("Context", "")),
                "stream": str(row.get("Stream", "")),
                "metric_value": value,
                "profiler_replayer_passes": 1,
            }
        )

    expected_value = {
        "global_ld": expected_instruction_count(),
        "ldgsts": expected_instruction_count(),
        "shared_wavefronts": expected_payload_wavefronts(),
        "lts_total": expected_lts_total(),
    }.get(family)
    return {
        "order": expected_order(repetition),
        "row_variants": list(variants),
        "metric": metric,
        "values_by_variant": values,
        "rows": summaries,
        "scientific_gate": {
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
        },
    }


def manifest_binding(command: Mapping[str, Any]) -> dict[str, Any]:
    binding = {
        key: command[key]
        for key in (
            "cell",
            "orientation",
            "groups",
            "fixed_depth",
            "fixed_lead_samples",
            "repetition",
            "order",
            "row_variants",
            "family",
            "metric",
            "timeout_seconds",
            "target_arguments",
        )
    }
    if "campaign_contract" in command:
        binding["campaign_contract"] = dict(command["campaign_contract"])
    if "smoke_authorization" in command:
        binding["smoke_authorization"] = dict(command["smoke_authorization"])
    return binding


def manifest_inputs(
    command: Mapping[str, Any],
    family_directory: Path,
    *,
    include_report_input: bool,
) -> dict[str, Any]:
    inputs = {
        "binding": manifest_binding(command),
        "binary": dict(command["binary"]),
        "device": dict(command["device"]),
        "ncu": dict(command["ncu"]),
        "environment_overrides": dict(command["environment_overrides"]),
        "command_record": file_identity(
            family_directory / "command.json", family_directory
        ),
    }
    if include_report_input:
        inputs["profile_report"] = optional_file_identity(
            family_directory / "profile.ncu-rep", family_directory
        )
    return inputs


def run_manifest_document(
    kind: str,
    run: support.CapturedRun,
    command: Mapping[str, Any],
    family_directory: Path,
) -> dict[str, Any]:
    if kind == "ncu_profile":
        expected_command = command["profile_command"]
        stdout_path = family_directory / "profile.stdout.txt"
        stderr_path = family_directory / "profile.stderr.txt"
        include_report_input = False
    elif kind == "ncu_import":
        expected_command = command["import_command"]
        stdout_path = family_directory / "ncu.csv"
        stderr_path = family_directory / "import.stderr.txt"
        include_report_input = True
    else:
        raise CampaignError(f"unsupported manifest kind: {kind}")
    if list(run.command) != list(expected_command):
        raise CampaignError(f"{kind}: executed command differs from command record")
    if run.configured_timeout_seconds != command["timeout_seconds"]:
        raise CampaignError(f"{kind}: timeout differs from command record")
    document = {
        "schema_version": 1,
        "kind": kind,
        "command": list(run.command),
        "configured_timeout_seconds": run.configured_timeout_seconds,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "return_code": run.return_code,
        "timed_out": run.timed_out,
        "stdout": file_identity(stdout_path, family_directory),
        "stderr": file_identity(stderr_path, family_directory),
        "profile_report": optional_file_identity(
            family_directory / "profile.ncu-rep", family_directory
        ),
        "csv": optional_file_identity(
            family_directory / "ncu.csv", family_directory
        ),
        "inputs": manifest_inputs(
            command,
            family_directory,
            include_report_input=include_report_input,
        ),
    }
    return document


def validate_run_manifest(
    path: Path,
    kind: str,
    command: Mapping[str, Any],
    family_directory: Path,
) -> dict[str, Any]:
    observed = read_json(path)
    observed_command = observed.get("command")
    timeout = observed.get("configured_timeout_seconds")
    return_code = observed.get("return_code")
    if not isinstance(observed_command, list) or not all(
        isinstance(item, str) for item in observed_command
    ):
        raise CampaignError(f"{path}: command is not a string list")
    if not isinstance(timeout, int) or isinstance(timeout, bool):
        raise CampaignError(f"{path}: timeout is not an integer")
    if not isinstance(return_code, int) or isinstance(return_code, bool):
        raise CampaignError(f"{path}: return code is not an integer")
    if return_code != 0 or observed.get("timed_out") is not False:
        raise CampaignError(f"{path}: retained run was unsuccessful")
    started = support.parse_manifest_time(
        observed.get("started_at"), f"{path} started_at"
    )
    finished = support.parse_manifest_time(
        observed.get("finished_at"), f"{path} finished_at"
    )
    if finished < started:
        raise CampaignError(f"{path}: finish precedes start")
    run = support.CapturedRun(
        command=tuple(observed_command),
        configured_timeout_seconds=timeout,
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        return_code=return_code,
        timed_out=False,
        stdout="",
        stderr="",
    )
    expected = run_manifest_document(kind, run, command, family_directory)
    if not same_json(observed, expected):
        raise CampaignError(f"{path}: manifest or artifact identity mismatch")
    return observed


def complete_document(
    family_directory: Path,
    command: Mapping[str, Any],
    summary: Mapping[str, Any],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    paths = {
        name: family_directory / name
        for name in (
            "command.json",
            "profile.ncu-rep",
            "ncu.csv",
            "profile.stdout.txt",
            "profile.stderr.txt",
            "import.stderr.txt",
            "profile.run.json",
            "import.run.json",
        )
    }
    return {
        "schema_version": 1,
        "valid": True,
        "binding": {
            **manifest_binding(command),
            "binary": dict(command["binary"]),
            "device": dict(command["device"]),
            "ncu": dict(command["ncu"]),
        },
        "files": {
            name: sha256(path) for name, path in paths.items()
        },
        "summary": dict(summary),
        "target_result": dict(target),
    }


def read_structural_complete(
    path: Path, expected_target: Mapping[str, Any]
) -> dict[str, Any]:
    if active_contract() == campaign_contract.V1_CONTRACT:
        return read_json(path)
    try:
        return execution_support.select_structural_complete_json(
            read_text(path),
            expected_target,
            mode="pair",
            context=str(path),
        )
    except execution_support.StructuralTargetError as error:
        raise CampaignError(f"{path}: invalid structural completion: {error}") from error


def validate_retained_family(
    family_directory: Path,
    context: support.Preflight,
    cell: Cell,
    repetition: int,
    family: str,
    timeout: int,
) -> dict[str, Any]:
    required = {
        name: family_directory / name
        for name in (
            "command.json",
            "profile.ncu-rep",
            "ncu.csv",
            "profile.stdout.txt",
            "profile.stderr.txt",
            "import.stderr.txt",
            "profile.run.json",
            "import.run.json",
            "complete.json",
        )
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise CampaignError(f"{family_directory}: missing {missing}")
    if required["profile.ncu-rep"].stat().st_size == 0:
        raise CampaignError(f"{family_directory}: empty NCU report")
    command = read_json(required["command.json"])
    expected_command = command_document(
        context, family_directory, cell, repetition, family, timeout
    )
    if not same_json(command, expected_command):
        raise CampaignError(f"{family_directory}: retained command mismatch")
    validate_run_manifest(
        required["profile.run.json"], "ncu_profile", command, family_directory
    )
    validate_run_manifest(
        required["import.run.json"], "ncu_import", command, family_directory
    )
    rows = support.parse_csv_rows(
        read_text(required["ncu.csv"]), str(required["ncu.csv"])
    )
    summary = validate_rows(rows, cell, repetition, family, context)
    target = validate_target_output(
        read_text(required["profile.stdout.txt"]),
        cell,
        repetition,
        context,
    )
    expected_complete = complete_document(family_directory, command, summary, target)
    observed_complete = read_structural_complete(required["complete.json"], target)
    if not same_json(observed_complete, expected_complete):
        raise CampaignError(f"{family_directory}: completion record mismatch")
    return expected_complete


def execution_environment(context: support.Preflight) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(context.environment)
    return environment


def execute_family(
    family_directory: Path,
    context: support.Preflight,
    cell: Cell,
    repetition: int,
    family: str,
    timeout: int,
) -> dict[str, Any]:
    try:
        return validate_retained_family(
            family_directory, context, cell, repetition, family, timeout
        )
    except CampaignError:
        pass
    ensure_output_directory(family_directory)
    command = command_document(
        context, family_directory, cell, repetition, family, timeout
    )
    write_json(family_directory / "command.json", command)
    profiled = support.run_captured(
        command["profile_command"],
        timeout=timeout,
        environment=execution_environment(context),
    )
    write_text(
        family_directory / "profile.stdout.txt", profiled.stdout
    )
    write_text(
        family_directory / "profile.stderr.txt", profiled.stderr
    )
    write_json(
        family_directory / "profile.run.json",
        run_manifest_document("ncu_profile", profiled, command, family_directory),
    )
    if profiled.timed_out or profiled.return_code != 0:
        raise CampaignError(
            f"{family_directory}: profile failed: "
            f"return_code={profiled.return_code}, timed_out={profiled.timed_out}"
        )
    report = family_directory / "profile.ncu-rep"
    if not report.is_file() or report.stat().st_size == 0:
        raise CampaignError(f"{family_directory}: report was not created")

    imported = support.run_captured(
        command["import_command"],
        timeout=timeout,
        environment=execution_environment(context),
    )
    write_text(family_directory / "ncu.csv", imported.stdout)
    write_text(
        family_directory / "import.stderr.txt", imported.stderr
    )
    write_json(
        family_directory / "profile.run.json",
        run_manifest_document("ncu_profile", profiled, command, family_directory),
    )
    write_json(
        family_directory / "import.run.json",
        run_manifest_document("ncu_import", imported, command, family_directory),
    )
    if imported.timed_out or imported.return_code != 0:
        raise CampaignError(
            f"{family_directory}: import failed: "
            f"return_code={imported.return_code}, timed_out={imported.timed_out}"
        )
    if sha256(active_contract().binary) != context.binary["sha256"]:
        raise CampaignError(f"{family_directory}: binary changed during capture")

    rows = support.parse_csv_rows(imported.stdout, str(family_directory / "ncu.csv"))
    summary = validate_rows(rows, cell, repetition, family, context)
    target = validate_target_output(profiled.stdout, cell, repetition, context)
    complete = complete_document(family_directory, command, summary, target)
    write_json(family_directory / "complete.json", complete)
    return validate_retained_family(
        family_directory, context, cell, repetition, family, timeout
    )


def bundle_document(
    cell: Cell,
    repetition: int,
    families: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(families) != set(FAMILIES):
        raise CampaignError(f"{cell.key} repeat-{repetition}: incomplete family set")
    values = {
        family: dict(document["summary"]["values_by_variant"])
        for family, document in families.items()
    }
    tolerance = paired_tolerance()
    variants = active_contract().variants
    hit_delta = abs(values["lts_hit"][variants[0]] - values["lts_hit"][variants[1]])
    miss_delta = abs(
        values["lts_miss"][variants[0]] - values["lts_miss"][variants[1]]
    )
    closure = {
        variant: abs(
            values["lts_hit"][variant]
            + values["lts_miss"][variant]
            - values["lts_total"][variant]
        )
        for variant in variants
    }
    gates = {
        "global_ld_exact": all(
            values["global_ld"][variant] == expected_instruction_count()
            for variant in variants
        ),
        "ldgsts_exact": all(
            values["ldgsts"][variant] == expected_instruction_count()
            for variant in variants
        ),
        "shared_wavefronts_exact": all(
            values["shared_wavefronts"][variant] == expected_payload_wavefronts()
            for variant in variants
        ),
        "lts_total_exact": all(
            values["lts_total"][variant] == expected_lts_total()
            for variant in variants
        ),
        "lts_hit_positive": all(values["lts_hit"][variant] > 0 for variant in variants),
        "lts_miss_positive": all(
            values["lts_miss"][variant] > 0 for variant in variants
        ),
        "paired_lts_hit_within_tolerance": hit_delta <= tolerance,
        "paired_lts_miss_within_tolerance": miss_delta <= tolerance,
        "separate_profile_closure_within_tolerance": all(
            value <= tolerance for value in closure.values()
        ),
        "all_rows_one_replay_pass": True,
    }
    background = {
        variant: values["dram"][variant] - values["lts_miss"][variant]
        for variant in variants
    }
    document = {
        "schema_version": 1,
        "valid": all(gates.values()),
        "cell": cell.key,
        "orientation": cell.orientation,
        "groups": active_contract().trace_groups,
        "fixed_depth": active_contract().fixed_depth,
        "fixed_lead_samples": active_contract().fixed_lead_samples,
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
            "expected_global_ld_instructions_per_variant": (
                expected_instruction_count()
            ),
            "expected_ldgsts_instructions_per_variant": (
                expected_instruction_count()
            ),
            "expected_shared_payload_wavefronts_per_variant": (
                expected_payload_wavefronts()
            ),
            "expected_lts_lookup_sectors_per_variant": expected_lts_total(),
            "paired_tolerance": tolerance,
            "absolute_lts_hit_delta": hit_delta,
            "absolute_lts_miss_delta": miss_delta,
            "absolute_hit_plus_miss_minus_total_by_variant": closure,
            "gates": gates,
        },
        "non_gating_dram_background": {
            "raw_dram_sectors_by_variant": values["dram"],
            "raw_dram_minus_attributed_lts_misses": background,
        },
    }
    if active_contract() != campaign_contract.V1_CONTRACT:
        document["campaign_contract"] = active_contract().to_json()
    return document


def parse_orientations(value: str | None) -> list[Cell]:
    if value is None:
        return [Cell(orientation) for orientation in active_contract().orientations]
    cells: list[Cell] = []
    for token in value.split(","):
        orientation = token.strip()
        if orientation not in active_contract().orientations:
            raise CampaignError(
                f"invalid orientation {orientation!r}; choose from "
                f"{','.join(active_contract().orientations)}"
            )
        cell = Cell(orientation)
        if cell not in cells:
            cells.append(cell)
    if not cells:
        raise CampaignError("--orientations selected no cells")
    return cells


def parse_repetitions(value: str | None) -> list[int]:
    if value is None:
        return list(REPETITIONS)
    repetitions: list[int] = []
    for token in value.split(","):
        try:
            repetition = int(token.strip())
        except ValueError as error:
            raise CampaignError(f"invalid repetition {token!r}") from error
        if repetition not in REPETITIONS:
            raise CampaignError("repetitions must be in 1..10")
        if repetition not in repetitions:
            repetitions.append(repetition)
    if not repetitions:
        raise CampaignError("--repeats selected no repetitions")
    return repetitions


def parse_families(value: str | None) -> list[str]:
    if value is None:
        return list(FAMILIES)
    families: list[str] = []
    for token in value.split(","):
        family = token.strip()
        if family not in FAMILY_METRICS:
            raise CampaignError(
                f"invalid family {family!r}; choose from {','.join(FAMILIES)}"
            )
        if family not in families:
            families.append(family)
    if not families:
        raise CampaignError("--families selected no families")
    return families


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, default=active_contract().evidence_root)
    result.add_argument(
        "--orientations",
        help="comma-separated first_line_early,second_line_early (default: both)",
    )
    result.add_argument(
        "--repeats", help="comma-separated subset in 1..10 (default: all 10)"
    )
    result.add_argument(
        "--families",
        help=f"comma-separated subset of {','.join(FAMILIES)} (default: all)",
    )
    result.add_argument(
        "--smoke",
        action="store_true",
        help="select first_line_early repeat 1 with all seven metric families",
    )
    result.add_argument("--dry-run", action="store_true")
    result.add_argument(
        "--validate-only",
        action="store_true",
        help="validate retained selected families without invoking NCU",
    )
    result.add_argument("--ncu", default=os.environ.get("NCU", "ncu"))
    result.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.timeout < 1:
            raise CampaignError("--timeout must be positive")
        if args.dry_run and args.validate_only:
            raise CampaignError("--dry-run and --validate-only are mutually exclusive")
        if args.smoke and (args.orientations or args.repeats or args.families):
            raise CampaignError("--smoke cannot be combined with subset options")
        contract = campaign_contract.begin_campaign()
        cells = (
            [Cell(contract.orientations[0])]
            if args.smoke
            else parse_orientations(args.orientations)
        )
        repetitions = [1] if args.smoke else parse_repetitions(args.repeats)
        families = list(FAMILIES) if args.smoke else parse_families(args.families)
        root = campaign_contract.require_canonical_evidence_root(
            contract, args.root
        )
        campaign_contract.validate_evidence_tree(contract, root)
        if not args.smoke:
            require_full_campaign_authorization(root, retained_binary(root))
        context = preflight(root, args.ncu)

        planned: list[dict[str, Any]] = []
        for cell in cells:
            for repetition in repetitions:
                repeat_directory = (
                    root / "paired_ncu" / cell.key / f"repeat-{repetition}"
                )
                for family in families:
                    planned.append(
                        command_document(
                            context,
                            repeat_directory / family,
                            cell,
                            repetition,
                            family,
                            args.timeout,
                        )
                    )
        if args.dry_run:
            print(
                support.json_text(
                    {
                        "dry_run": True,
                        "root": str(root),
                        "selected_cells": [cell.key for cell in cells],
                        "selected_repetitions": repetitions,
                        "selected_families": families,
                        "profile_count": len(planned),
                        "commands": planned,
                    }
                ),
                end="",
            )
            campaign_contract.validate_evidence_tree(contract, root)
            return 0

        total = len(planned)
        progress = 0
        for cell in cells:
            for repetition in repetitions:
                repeat_directory = (
                    root / "paired_ncu" / cell.key / f"repeat-{repetition}"
                )
                completed: dict[str, Mapping[str, Any]] = {}
                for family in families:
                    progress += 1
                    directory = repeat_directory / family
                    if args.validate_only:
                        document = validate_retained_family(
                            directory,
                            context,
                            cell,
                            repetition,
                            family,
                            args.timeout,
                        )
                        state = "valid"
                    else:
                        try:
                            document = validate_retained_family(
                                directory,
                                context,
                                cell,
                                repetition,
                                family,
                                args.timeout,
                            )
                            state = "existing"
                        except CampaignError:
                            document = execute_family(
                                directory,
                                context,
                                cell,
                                repetition,
                                family,
                                args.timeout,
                            )
                            state = "captured"
                    completed[family] = document
                    print(
                        f"line-ownership-ncu {progress:03d}/{total:03d} "
                        f"{cell.orientation} r{repetition} {family} ({state})",
                        flush=True,
                    )

                if set(completed) != set(FAMILIES):
                    for family in FAMILIES:
                        if family in completed:
                            continue
                        try:
                            completed[family] = validate_retained_family(
                                repeat_directory / family,
                                context,
                                cell,
                                repetition,
                                family,
                                args.timeout,
                            )
                        except CampaignError:
                            pass
                if set(completed) == set(FAMILIES):
                    bundle = bundle_document(cell, repetition, completed)
                    bundle_path = repeat_directory / "bundle.json"
                    if args.validate_only:
                        try:
                            retained_bundle = campaign_contract.read_artifact_json(
                                contract, bundle_path
                            )
                        except campaign_contract.ContractError as error:
                            raise CampaignError(
                                f"missing retained bundle: {bundle_path}: {error}"
                            ) from error
                        if not campaign_contract.strict_json_equal(
                            retained_bundle, bundle
                        ):
                            raise CampaignError(f"{bundle_path}: bundle mismatch")
                    else:
                        write_json(bundle_path, bundle)
                elif args.validate_only and set(families) == set(FAMILIES):
                    raise CampaignError(f"{cell.key} repeat-{repetition}: incomplete bundle")
        campaign_contract.validate_evidence_tree(contract, root)
        return 0
    except (
        CampaignError,
        support.CampaignError,
        campaign_contract.ContractError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
