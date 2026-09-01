#!/usr/bin/env python3
"""Reduce the SM86 asynchronous-copy source-line ownership discriminator."""

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

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from analyze_cohort_reservation import (  # noqa: E402
    AnalysisError,
    integer,
    parse_csv_rows,
    read_json,
    schema_observation,
    sha256,
)
from analyze_multiline_response import (  # noqa: E402
    EXPECTED_DEVICE,
    EXPECTED_ENVIRONMENT,
    EXPECTED_NCU_VERSION,
    SCHEMA18_FIELDS,
    argv_options,
    last_json_object,
    production_contract,
    require_cross_endpoint_tracer_identity,
    retained_tracer_identity,
)
import capture_line_ownership_ncu as ncu_producer  # noqa: E402
from line_ownership_campaign_contract import (  # noqa: E402
    CampaignContract,
    ContractError,
    artifact_identity as contract_artifact_identity,
    begin_campaign,
    current_contract,
    read_artifact_bytes,
    read_artifact_json,
    read_artifact_text,
    strict_json_equal as contract_strict_json_equal,
    strict_json_loads,
)
import capture_line_ownership_execution as execution_producer  # noqa: E402
from tools.nvbit_tracer.import_trace_contract import (  # noqa: E402
    LANE_INACTIVE_SENTINEL,
    _reconstruct_lane_vector,
)
from tools.nvbit_tracer.proto_binding_loader import (  # noqa: E402
    load_dynamic_trace_bindings,
)


ACTIVE_CONTRACT: CampaignContract = current_contract()
DEFAULT_ROOT = ACTIVE_CONTRACT.evidence_root
DEFAULT_AGGREGATE = ACTIVE_CONTRACT.aggregate_report
PROBE_BINARY = ACTIVE_CONTRACT.binary
SIM_RUN = REPO_ROOT / "sim/target/release/sim-run"

TRACE_GROUPS = 64
DIRECT_GROUPS = 8192
REPETITIONS = tuple(range(1, 11))
ORIENTATIONS = ("first_line_early", "second_line_early")
VARIANTS = ("early_fanout", "late_fanout")
SMOKE_ORIENTATION = "first_line_early"

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
EXPECTED_VALUES = {
    "global_ld": 64,
    "ldgsts": 64,
    "shared_wavefronts": ACTIVE_CONTRACT.ncu_shared_wavefronts_per_copy * TRACE_GROUPS,
    "lts_total": 3584,
}
PAIRED_TOLERANCE = 36

_legacy_read_json = read_json
_legacy_sha256 = sha256


def _coordinate_artifact(path: Path) -> bool:
    if ACTIVE_CONTRACT.generation_id == "source_line_ownership_v1":
        return False
    candidate = ACTIVE_CONTRACT.require_allowed_path(path)
    root = ACTIVE_CONTRACT.require_allowed_path(ACTIVE_CONTRACT.evidence_root)
    aggregate = ACTIVE_CONTRACT.require_allowed_path(ACTIVE_CONTRACT.aggregate_report)
    return candidate == aggregate or candidate == root or root in candidate.parents


def same_json(observed: Any, expected: Any) -> bool:
    if ACTIVE_CONTRACT.generation_id == "source_line_ownership_v1":
        return observed == expected
    return contract_strict_json_equal(observed, expected)


def read_json(path: Path) -> dict[str, Any]:
    if _coordinate_artifact(path):
        return read_artifact_json(ACTIVE_CONTRACT, path)
    return _legacy_read_json(path)


def sha256(path: Path) -> str:
    if _coordinate_artifact(path):
        return contract_artifact_identity(ACTIVE_CONTRACT, path)["sha256"]
    return _legacy_sha256(path)


def read_text(path: Path) -> str:
    if _coordinate_artifact(path):
        return read_artifact_text(ACTIVE_CONTRACT, path)
    return path.read_text(encoding="utf-8")


def read_bytes(path: Path) -> bytes:
    if _coordinate_artifact(path):
        return read_artifact_bytes(ACTIVE_CONTRACT, path)
    return path.read_bytes()


def decode_json_value(payload: str | bytes, context: str | Path) -> Any:
    if ACTIVE_CONTRACT.generation_id == "source_line_ownership_v1":
        return json.loads(payload)
    try:
        return strict_json_loads(payload, context=context)
    except ContractError as error:
        raise AnalysisError(f"cannot decode {context}: {error}") from error


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


def expected_order(repetition: int) -> str:
    return "ab" if repetition % 2 else "ba"


def row_variants(order: str) -> tuple[str, str]:
    if order == "ab":
        return VARIANTS
    if order == "ba":
        return tuple(reversed(VARIANTS))
    raise AnalysisError(f"unsupported order {order!r}")


def paired_cell_name(orientation: str, groups: int = TRACE_GROUPS) -> str:
    return f"{ACTIVE_CONTRACT.binary.name}_{orientation}_g{groups}"


def endpoint_name(variant: str, orientation: str) -> str:
    return f"async_source_line_ownership_{variant}_{orientation}_g{TRACE_GROUPS}"


def expected_target_arguments(orientation: str, repetition: int) -> list[str]:
    return [
        "--groups",
        str(TRACE_GROUPS),
        "--orientation",
        orientation,
        "--order",
        expected_order(repetition),
        "--markers",
        "on",
    ]


def binary_report(root: Path) -> dict[str, Any]:
    try:
        return execution_producer.retained_binary(root)
    except execution_producer.CaptureError as error:
        raise AnalysisError(str(error)) from error


def execution_cell(orientation: str) -> execution_producer.Cell:
    return execution_producer.Cell(
        "first" if orientation == "first_line_early" else "second"
    )


def validate_arm(value: Any, groups: int, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AnalysisError(f"{context}: expected result object")
    for field in ("issue_cycles", "wait_cycles", "total_cycles", "token", "mismatches"):
        integer(value.get(field), f"{context}.{field}")
    if value["issue_cycles"] + value["wait_cycles"] != value["total_cycles"]:
        raise AnalysisError(f"{context}: intervals do not sum")
    if value.get("sum_consistent") is not True or value.get("correct") is not True:
        raise AnalysisError(f"{context}: correctness failed")
    if value["mismatches"] != 0:
        raise AnalysisError(f"{context}: mismatches={value['mismatches']}")
    for field, numerator in (
        ("issue_cycles_per_group", value["issue_cycles"]),
        ("wait_cycles_per_group", value["wait_cycles"]),
        ("total_cycles_per_group", value["total_cycles"]),
    ):
        observed = value.get(field)
        if not isinstance(observed, (int, float)) or not math.isclose(
            float(observed), numerator / groups, rel_tol=0.0, abs_tol=1.0e-8
        ):
            raise AnalysisError(f"{context}.{field}: inconsistent")
    return value


def common_target_contract(orientation: str, groups: int, mode: str) -> dict[str, Any]:
    return {
        "workload": "async_source_line_ownership",
        "route": "l2",
        "mode": mode,
        "orientation": orientation,
        "depth": 1,
        "groups": groups,
        "warmup_groups": 64,
        "lead_clock_samples": 16,
        "source_cohorts_per_copy": 8,
        "aligned_source_cohorts_per_copy": 4,
        "split_source_cohorts_per_copy": 4,
        "source_sectors_per_cohort": 4,
        "payload_bytes_per_cohort": 64,
        "scalar_ldg_instructions_per_copy": 1,
        "scalar_ldg_sectors_per_copy": 24,
        "scalar_preload_lines_per_copy": 8,
        "ldgsts_instructions_per_copy": 1,
        "ldgsts_sectors_per_copy": 32,
        "unique_ldgsts_lines_per_copy": 12,
        "payload_wavefronts_per_copy": ACTIVE_CONTRACT.payload_wavefronts_per_copy,
        "projected_bypass_wavefronts_per_copy": ACTIVE_CONTRACT.bypass_wavefronts_per_copy,
        "destination_lines_per_copy": ACTIVE_CONTRACT.destination_lines_per_copy,
        "max_cohort_bank_wavefronts": 3,
        "max_cohort_bypass_wavefronts": 4,
        "cohort_topology_penalty_cycles_per_copy": 120,
        "response_memberships_per_copy": 32,
        "local_device_lts_lookups_per_copy": 56,
        "source_step_stride_bytes": 2048,
        "source_step_max_exclusive_byte": 1968,
        "source_step_guard_bytes": 80,
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
        "device": EXPECTED_DEVICE["name"],
        "compute_capability": EXPECTED_DEVICE["compute_capability"],
        "correct": True,
    }


def expected_target_contract(
    orientation: str, groups: int, mode: str, order: str
) -> dict[str, Any]:
    expected = common_target_contract(orientation, groups, mode)
    expected["order"] = order
    return expected


def select_structural_target_output(
    path: Path,
    *,
    orientation: str,
    groups: int,
    mode: str,
    order: str,
    exactly_one: bool,
) -> dict[str, Any]:
    expected = expected_target_contract(orientation, groups, mode, order)
    lines = read_text(path).splitlines()
    if exactly_one:
        payloads: list[dict[str, Any]] = []
        for line_number, line in enumerate(lines, start=1):
            if not line.lstrip().startswith("{"):
                continue
            try:
                payloads.append(
                    execution_producer.select_structural_target_json(
                        line,
                        expected,
                        mode=mode,
                        context=f"{path}:{line_number}",
                    )
                )
            except execution_producer.StructuralTargetError as error:
                raise AnalysisError(str(error)) from error
        if len(payloads) != 1:
            raise AnalysisError(
                f"{path}: expected one target JSON object, got {len(payloads)}"
            )
        return payloads[0]
    for line_number, line in reversed(tuple(enumerate(lines, start=1))):
        if not line.lstrip().startswith("{"):
            continue
        try:
            return execution_producer.select_structural_target_json(
                line,
                expected,
                mode=mode,
                context=f"{path}:{line_number}",
            )
        except execution_producer.StructuralTargetError as error:
            raise AnalysisError(str(error)) from error
    raise AnalysisError(f"{path}: no JSON result object")


def validate_target_payload(
    payload: Mapping[str, Any],
    *,
    orientation: str,
    groups: int,
    mode: str,
    order: str,
    variant: str | None = None,
) -> None:
    expected = expected_target_contract(orientation, groups, mode, order)
    mismatches = {
        field: {"observed": payload.get(field), "expected": value}
        for field, value in expected.items()
        if not same_json(payload.get(field), value)
    }
    if mismatches:
        raise AnalysisError(f"target contract mismatch: {mismatches}")
    if mode == "pair":
        for arm in VARIANTS:
            if ACTIVE_CONTRACT.generation_id == "source_line_ownership_v1":
                validate_arm(payload.get(arm), groups, f"target.{arm}")
            else:
                try:
                    execution_producer.validate_structural_arm(
                        payload.get(arm), f"target.{arm}"
                    )
                except execution_producer.CaptureError as error:
                    raise AnalysisError(str(error)) from error
    else:
        if payload.get("variant") != variant:
            raise AnalysisError(f"target variant mismatch: {payload.get('variant')!r}")
        if ACTIVE_CONTRACT.generation_id == "source_line_ownership_v1":
            validate_arm(payload.get("result"), groups, f"target.{variant}")
        else:
            try:
                execution_producer.validate_structural_arm(
                    payload.get("result"), f"target.{variant}"
                )
            except execution_producer.CaptureError as error:
                raise AnalysisError(str(error)) from error


def load_resources(path: Path) -> dict[str, dict[str, int]]:
    text = read_text(path)
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
        "command": base / "command.json",
    }
    absent = [str(path) for path in paths.values() if not path.is_file()]
    if absent:
        raise FileNotFoundError(", ".join(absent))
    control = read_json(paths["control"])
    functions = control.get("functions")
    summary = control.get("summary")
    if not isinstance(functions, dict) or not isinstance(summary, dict):
        raise AnalysisError("malformed static control sidecar")
    selected = {
        name: rows
        for name, rows in functions.items()
        if "source_line_ownership_kernel" in name
    }
    resources = load_resources(paths["resources"])
    build = read_text(paths["build"])
    sass = read_text(paths["sass"])
    artifact_errors: list[str] = []
    scientific_failures: list[dict[str, Any]] = []
    if len(selected) != 2:
        artifact_errors.append(f"expected two orientation specializations, got {len(selected)}")
    for name, rows in selected.items():
        if not isinstance(rows, list) or any(row.get("status") != "decoded" for row in rows):
            artifact_errors.append(f"{name}: missing or undecoded rows")
            continue
        opcodes = [str(row.get("opcode_text", "")) for row in rows]
        failures: list[str] = []
        scalar = [i for i, opcode in enumerate(opcodes) if opcode.startswith("LDG.E.STRONG.GPU")]
        ldgsts = [i for i, opcode in enumerate(opcodes) if opcode.startswith("LDGSTS")]
        clocks = [i for i, opcode in enumerate(opcodes) if opcode.startswith("CS2R ") and "SR_CLOCKLO" in opcode]
        lead = [i for i, opcode in enumerate(opcodes) if (opcode.startswith("CS2R.32 ") or opcode.startswith("S2UR ")) and "SR_CLOCKLO" in opcode]
        commits = [i for i, opcode in enumerate(opcodes) if opcode == "LDGDEPBAR"]
        waits = [i for i, opcode in enumerate(opcodes) if opcode.startswith("DEPBAR.LE SB0, 0x0")]
        stores = [i for i, opcode in enumerate(opcodes) if opcode.startswith("STG")]
        if len(scalar) != 1 or len(ldgsts) != 1:
            failures.append(f"scalar/LDGSTS counts {len(scalar)}/{len(ldgsts)} != 1/1")
        if len(clocks) != 3 or len(lead) != 16:
            failures.append(f"primary/lead clock counts {len(clocks)}/{len(lead)} != 3/16")
        if len(commits) != 1 or len(waits) != 1:
            failures.append("commit/wait count drift")
        if clocks and stores and min(stores) <= clocks[-1]:
            failures.append("result store precedes post-wait clock")
        if scalar and len(clocks) == 3:
            barriers = {int(rows[i].get("control", {}).get("write_barrier", 7)) for i in scalar}
            barriers.discard(7)
            for i in range(scalar[0] + 1, clocks[-1] + 1):
                mask = int(rows[i].get("control", {}).get("wait_mask", 0))
                if any(mask & (1 << barrier) for barrier in barriers):
                    failures.append("preload barrier consumed before post-wait clock")
                    break
        resource = resources.get(name)
        if resource is None:
            artifact_errors.append(f"{name}: missing resource record")
        elif resource["stack"] != 0 or resource["local"] != 0:
            failures.append("stack/local allocation nonzero")
        if name not in sass:
            artifact_errors.append(f"{name}: absent from SASS")
        if failures:
            scientific_failures.append({"function": name, "failures": failures})
    if summary.get("total_unknown_encoding") != 0:
        artifact_errors.append("unknown control encoding")
    spills = [
        tuple(int(match.group(field)) for field in ("stack", "stores", "loads"))
        for match in SPILL_RE.finditer(build)
    ]
    if len(spills) != 2:
        artifact_errors.append(f"expected two ptxas spill records, got {len(spills)}")
    elif any(any(row) for row in spills):
        scientific_failures.append({"function": "all", "failures": ["ptxas stack/spill"]})
    return {
        "artifact_valid": not artifact_errors,
        "assay_valid": not artifact_errors and not scientific_failures,
        "artifact_errors": artifact_errors,
        "scientific_failures": scientific_failures,
        "function_count": len(selected),
        "decoded_instructions": sum(len(rows) for rows in selected.values()),
        "artifacts": {name: sha256(path) for name, path in paths.items()},
    }


def expected_ncu_command(
    command: Mapping[str, Any],
    directory: Path,
    family: str,
    orientation: str,
    repetition: int,
    binary: Mapping[str, Any],
) -> None:
    timeout = command.get("timeout_seconds")
    if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout < 1:
        raise AnalysisError(f"{directory}: invalid timeout")
    expected = {
        "schema_version": 1,
        "cell": paired_cell_name(orientation),
        "orientation": orientation,
        "fixed_depth": 1,
        "fixed_lead_samples": 16,
        "groups": TRACE_GROUPS,
        "repetition": repetition,
        "order": expected_order(repetition),
        "row_variants": list(row_variants(expected_order(repetition))),
        "family": family,
        "metric": FAMILY_METRICS[family],
        "binary": dict(binary),
        "device": EXPECTED_DEVICE,
        "environment_overrides": EXPECTED_ENVIRONMENT,
        "target_arguments": expected_target_arguments(orientation, repetition),
    }
    if ACTIVE_CONTRACT.generation_id != "source_line_ownership_v1":
        expected["campaign_contract"] = ACTIVE_CONTRACT.to_json()
        ncu = command.get("ncu")
        if not isinstance(ncu, dict) or not isinstance(ncu.get("path"), str):
            raise AnalysisError(f"{directory}: malformed NCU identity")
        try:
            expected["ncu_executable"] = ncu_producer.ncu_executable_identity(
                Path(ncu["path"])
            )
        except ncu_producer.CampaignError as error:
            raise AnalysisError(f"{directory}: invalid NCU executable: {error}") from error
        if orientation != SMOKE_ORIENTATION or repetition != 1:
            authorization = execution_producer.smoke_authorization_path(
                directory.parents[3]
            )
            expected["smoke_authorization"] = contract_artifact_identity(
                ACTIVE_CONTRACT, authorization
            )
    mismatches = {
        field: (command.get(field), value)
        for field, value in expected.items()
        if not same_json(command.get(field), value)
    }
    if mismatches:
        raise AnalysisError(f"{directory}: NCU binding mismatch: {mismatches}")
    ncu = command.get("ncu")
    if not isinstance(ncu, dict) or EXPECTED_NCU_VERSION not in str(ncu.get("version")):
        raise AnalysisError(f"{directory}: NCU identity mismatch")


def require_coordinate_completion_contract(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    context: Path,
) -> None:
    if not same_json(dict(observed), dict(expected)):
        raise AnalysisError(f"{context}: full producer completion mismatch")


def require_coordinate_bundle_contract(
    observed: Mapping[str, Any],
    *,
    orientation: str,
    repetition: int,
    families: Mapping[str, Mapping[str, Any]],
    context: Path,
) -> None:
    try:
        expected = ncu_producer.bundle_document(
            ncu_producer.Cell(orientation), repetition, families
        )
    except (KeyError, TypeError, ncu_producer.CampaignError) as error:
        raise AnalysisError(
            f"{context}: cannot reconstruct producer bundle: {error}"
        ) from error
    if not same_json(dict(observed), expected):
        raise AnalysisError(f"{context}: full producer bundle mismatch")


def validate_family(
    directory: Path,
    family: str,
    orientation: str,
    repetition: int,
    binary: Mapping[str, Any],
) -> dict[str, Any]:
    names = (
        "command.json", "complete.json", "ncu.csv", "profile.ncu-rep",
        "profile.stdout.txt", "profile.stderr.txt", "import.stderr.txt",
        "profile.run.json", "import.run.json",
    )
    paths = {name: directory / name for name in names}
    absent = [str(path) for path in paths.values() if not path.is_file()]
    if absent:
        raise FileNotFoundError(", ".join(absent))
    command = read_json(paths["command.json"])
    expected_ncu_command(command, directory, family, orientation, repetition, binary)
    if ACTIVE_CONTRACT.generation_id == "source_line_ownership_v1":
        complete = read_json(paths["complete.json"])
        # Preserve the predecessor path while reusing its timestamp-bearing
        # manifest validator.
        try:
            ncu_producer.validate_run_manifest(
                paths["profile.run.json"], "ncu_profile", command, directory
            )
            ncu_producer.validate_run_manifest(
                paths["import.run.json"], "ncu_import", command, directory
            )
        except ncu_producer.CampaignError as error:
            raise AnalysisError(
                f"{directory}: invalid run manifest: {error}"
            ) from error
    else:
        ncu = command.get("ncu")
        if (
            not isinstance(ncu, Mapping)
            or not isinstance(ncu.get("path"), str)
            or not isinstance(ncu.get("version"), str)
        ):
            raise AnalysisError(f"{directory}: malformed NCU identity")
        ncu_executable = command.get("ncu_executable")
        if not isinstance(ncu_executable, Mapping):
            raise AnalysisError(f"{directory}: missing NCU executable identity")
        context = ncu_producer.NcuPreflight(
            binary=dict(binary),
            device=dict(EXPECTED_DEVICE),
            ncu_path=ncu["path"],
            ncu_version=ncu["version"],
            ncu_executable=dict(ncu_executable),
            environment=dict(EXPECTED_ENVIRONMENT),
        )
        try:
            complete = ncu_producer.validate_retained_family(
                directory,
                context,
                ncu_producer.Cell(orientation),
                repetition,
                family,
                int(command["timeout_seconds"]),
            )
        except (KeyError, TypeError, ValueError, ncu_producer.CampaignError) as error:
            raise AnalysisError(
                f"{directory}: full producer completion contract failed: {error}"
            ) from error
        target_contract = expected_target_contract(
            orientation,
            TRACE_GROUPS,
            "pair",
            expected_order(repetition),
        )
        try:
            observed_complete = ncu_producer.read_structural_complete(
                paths["complete.json"], target_contract
            )
        except ncu_producer.CampaignError as error:
            raise AnalysisError(str(error)) from error
        require_coordinate_completion_contract(
            observed_complete, complete, paths["complete.json"]
        )
    expected_files = {name: sha256(path) for name, path in paths.items() if name != "complete.json"}
    if not same_json(complete.get("files"), expected_files) or complete.get(
        "valid"
    ) is not True:
        raise AnalysisError(f"{directory}: completion/file identity mismatch")
    binding_fields = (
        "cell", "orientation", "fixed_depth", "fixed_lead_samples", "groups",
        "repetition", "order", "row_variants", "family", "metric",
        "timeout_seconds", "binary", "device", "ncu", "target_arguments",
    )
    if "campaign_contract" in command:
        binding_fields = (*binding_fields, "campaign_contract")
    if "ncu_executable" in command:
        binding_fields = (*binding_fields, "ncu_executable")
    if "smoke_authorization" in command:
        binding_fields = (*binding_fields, "smoke_authorization")
    if not same_json(
        complete.get("binding"),
        {field: command[field] for field in binding_fields},
    ):
        raise AnalysisError(f"{directory}: completion binding mismatch")
    rows = parse_csv_rows(paths["ncu.csv"])
    if len(rows) != 2 or [str(row.get("ID", "")).strip() for row in rows] != ["0", "1"]:
        raise AnalysisError(f"{directory}: NCU rows are not ordered 0/1")
    variants = row_variants(expected_order(repetition))
    values: dict[str, int] = {}
    summaries: list[dict[str, Any]] = []
    for index, (row, variant) in enumerate(zip(rows, variants, strict=True)):
        if integer(row.get("profiler__replayer_passes"), "replay passes") != 1:
            raise AnalysisError(f"{directory}: multi-pass row")
        value = integer(row.get(FAMILY_METRICS[family]), f"{directory}.{family}")
        values[variant] = value
        summaries.append({"id": index, "variant": variant, "metric_value": value})
    if ACTIVE_CONTRACT.generation_id == "source_line_ownership_v1":
        payload = last_json_object(paths["profile.stdout.txt"])
    else:
        payload = select_structural_target_output(
            paths["profile.stdout.txt"],
            orientation=orientation,
            groups=TRACE_GROUPS,
            mode="pair",
            order=expected_order(repetition),
            exactly_one=True,
        )
    validate_target_payload(payload, orientation=orientation, groups=TRACE_GROUPS, mode="pair", order=expected_order(repetition))
    if (
        ACTIVE_CONTRACT.generation_id != "source_line_ownership_v1"
        and not same_json(complete.get("target_result"), payload)
    ):
        raise AnalysisError(f"{directory}: completion target projection mismatch")
    return {"family": family, "values_by_variant": values, "rows": summaries, "files": expected_files, "target": payload, "producer_complete": complete}


def validate_bundle(root: Path, orientation: str, repetition: int, binary: Mapping[str, Any]) -> dict[str, Any]:
    base = root / "paired_ncu" / paired_cell_name(orientation) / f"repeat-{repetition}"
    families = {family: validate_family(base / family, family, orientation, repetition, binary) for family in FAMILIES}
    bundle_path = base / "bundle.json"
    if not bundle_path.is_file():
        raise FileNotFoundError(str(bundle_path))
    bundle = read_json(bundle_path)
    if ACTIVE_CONTRACT.generation_id != "source_line_ownership_v1":
        require_coordinate_bundle_contract(
            bundle,
            orientation=orientation,
            repetition=repetition,
            families={
                family: families[family]["producer_complete"]
                for family in FAMILIES
            },
            context=bundle_path,
        )
    values = {family: families[family]["values_by_variant"] for family in FAMILIES}
    hit_delta = abs(values["lts_hit"][VARIANTS[0]] - values["lts_hit"][VARIANTS[1]])
    miss_delta = abs(values["lts_miss"][VARIANTS[0]] - values["lts_miss"][VARIANTS[1]])
    closure = {variant: abs(values["lts_hit"][variant] + values["lts_miss"][variant] - values["lts_total"][variant]) for variant in VARIANTS}
    gates = {
        **{f"{family}_exact": all(values[family][variant] == expected for variant in VARIANTS) for family, expected in EXPECTED_VALUES.items()},
        "lts_hit_positive": all(values["lts_hit"][variant] > 0 for variant in VARIANTS),
        "lts_miss_positive": all(values["lts_miss"][variant] > 0 for variant in VARIANTS),
        "paired_hit_within_tolerance": hit_delta <= PAIRED_TOLERANCE,
        "paired_miss_within_tolerance": miss_delta <= PAIRED_TOLERANCE,
        "closure_within_tolerance": all(value <= PAIRED_TOLERANCE for value in closure.values()),
    }
    if bundle.get("valid") != all(gates.values()):
        raise AnalysisError(f"{bundle_path}: bundle validity disagrees with raw metrics")
    return {"orientation": orientation, "repetition": repetition, "valid": all(gates.values()), "gates": gates, "values": values, "bundle_sha256": sha256(bundle_path)}


def first_kernel_dir(trace_dir: Path) -> Path:
    kernels = sorted((trace_dir / "threadblocks").glob("device_*/stream_*/kernel_*"))
    if len(kernels) != 1:
        raise AnalysisError(f"{trace_dir}: expected one traced kernel directory, got {len(kernels)}")
    return kernels[0]


def lane_ownership_oracle(trace_dir: Path, variant: str, orientation: str) -> dict[str, Any]:
    bindings = load_dynamic_trace_bindings()
    records = 0
    preload_records = 0
    sample: dict[str, Any] | None = None
    for pb_path in sorted(first_kernel_dir(trace_dir).rglob("*.pb")):
        tb = bindings.threadblock_pb2.threadblock()
        tb.ParseFromString(read_bytes(pb_path))
        for warp_id in sorted(tb.warps.keys()):
            for inst in tb.warps[warp_id].instructions:
                if not inst.HasField("memory_op_class"):
                    continue
                and_mask = int(inst.active_mask) & int(inst.predicate_mask)
                memory_class = int(inst.memory_op_class)
                if (
                    memory_class == 1
                    and len(inst.addresses) == 1
                    and int(inst.addresses[0].data_width) == 4
                ):
                    address_mask = (
                        int(inst.addresses[0].lane_mask)
                        if inst.addresses[0].HasField("lane_mask")
                        else and_mask
                    )
                    vector, malformed = _reconstruct_lane_vector(
                        inst.addresses[0], address_mask
                    )
                    if vector is None:
                        raise AnalysisError(
                            f"{trace_dir}: malformed preload vector: {malformed}"
                        )
                    expected_mask = (
                        0x3333FFFF
                        if orientation == "first_line_early"
                        else 0xCCCCFFFF
                    )
                    if and_mask != expected_mask:
                        raise AnalysisError(
                            f"{trace_dir}: preload mask {and_mask:#x} != "
                            f"{expected_mask:#x}"
                        )
                    base = int(vector[0])
                    observed = [
                        LANE_INACTIVE_SENTINEL
                        if value == LANE_INACTIVE_SENTINEL
                        else int(value - base)
                        for value in vector
                    ]
                    expected = []
                    for lane in range(32):
                        cohort = lane // 4
                        within = lane % 4
                        active = cohort < 4 or (
                            within < 2
                            if orientation == "first_line_early"
                            else within >= 2
                        )
                        cohort_base = cohort * 256 + (64 if cohort >= 4 else 0)
                        expected.append(
                            cohort_base + 32 * within
                            if active
                            else LANE_INACTIVE_SENTINEL
                        )
                    if observed != expected:
                        raise AnalysisError(
                            f"{trace_dir}: scalar preload vector mismatch"
                        )
                    preload_records += 1
                    continue
                if memory_class != 8:
                    continue
                operands: dict[int, list[int]] = {}
                widths: dict[int, int] = {}
                for address in inst.addresses:
                    address_mask = (
                        int(address.lane_mask)
                        if address.HasField("lane_mask")
                        else and_mask
                    )
                    vector, malformed = _reconstruct_lane_vector(address, address_mask)
                    if vector is None:
                        raise AnalysisError(f"{trace_dir}: malformed lane vector: {malformed}")
                    operands[int(address.address_space)] = vector
                    widths[int(address.address_space)] = int(address.data_width)
                source = operands.get(1) or operands.get(4)
                destination = operands.get(2)
                if source is None or destination is None or widths.get(1, widths.get(4)) != 16 or widths.get(2) != 16:
                    raise AnalysisError(f"{trace_dir}: malformed LDGSTS source/destination operands")
                if any(value == LANE_INACTIVE_SENTINEL for value in source + destination):
                    raise AnalysisError(f"{trace_dir}: inactive LDGSTS lane")
                source_base = source[0]
                destination_base = destination[0]
                observed_source = [int(value - source_base) for value in source]
                observed_destination = [int(value - destination_base) for value in destination]
                expected_source: list[int] = []
                expected_destination: list[int] = []
                early_lanes = (0, 1) if orientation == "first_line_early" else (2, 3)
                for lane in range(32):
                    cohort = lane // 4
                    within = lane % 4
                    cohort_base = cohort * 256 + (64 if cohort >= 4 else 0)
                    expected_source.append(cohort_base + 32 * within)
                    slot = cohort * 512
                    if cohort < 4:
                        offset = (0, 32, 128, 160)[within]
                    else:
                        early = within in early_lanes
                        fanout = early if variant == "early_fanout" else not early
                        pair_index = early_lanes.index(within) if early else tuple(i for i in range(4) if i not in early_lanes).index(within)
                        offset = (0, 128)[pair_index] if fanout else (256, 288)[pair_index]
                    expected_destination.append(slot + offset)
                if observed_source != expected_source or observed_destination != expected_destination:
                    raise AnalysisError(f"{trace_dir}: per-lane ownership permutation mismatch")
                records += 1
                sample = sample or {"source": observed_source, "destination": observed_destination}
    if records != TRACE_GROUPS:
        raise AnalysisError(f"{trace_dir}: LDGSTS record count {records} != {TRACE_GROUPS}")
    if preload_records != TRACE_GROUPS:
        raise AnalysisError(
            f"{trace_dir}: preload record count {preload_records} != {TRACE_GROUPS}"
        )
    return {
        "records": records,
        "preload_records": preload_records,
        "preload_vector_exact": True,
        "per_lane_ownership_exact": True,
        "sample": sample,
    }


def nvbit_cell(root: Path, variant: str, orientation: str, binary_sha: str) -> dict[str, Any]:
    name = endpoint_name(variant, orientation)
    base = root / "nvbit" / name
    complete_path = base / "complete.json"
    binding_path = base / "campaign_binding.json"
    for path in (complete_path, binding_path, base / "dynamic_trace.pb", base / "stats.csv", base / "stdout.txt"):
        if not path.is_file():
            raise FileNotFoundError(str(path))
    complete = read_json(complete_path)
    binding = read_json(binding_path)
    try:
        execution_producer.validate_nvbit(
            root,
            base,
            execution_cell(orientation),
            variant,
            execution_producer.retained_binary(root),
        )
    except execution_producer.CaptureError as error:
        raise AnalysisError(f"{name}: {error}") from error
    if complete.get("binary_sha256") != binary_sha or binding.get("binary", {}).get("sha256") != binary_sha:
        raise AnalysisError(f"{name}: binary mismatch")
    tracer = retained_tracer_identity(complete, binding, name)
    options = argv_options(binding.get("argv", []))
    expected = {"--groups": "64", "--orientation": orientation, "--variant": variant, "--markers": "on"}
    if any(options.get(key) != value for key, value in expected.items()):
        raise AnalysisError(f"{name}: NVBit argv mismatch")
    for reason in ("unsupported_memory_path", "unmapped_memory_range", "cross_allocation_lane_span"):
        path = base / f"extra_info/{reason}.json"
        if not path.is_file():
            raise FileNotFoundError(str(path))
        value = decode_json_value(read_text(path), path)
        if value != []:
            raise AnalysisError(f"{name}: nonempty {reason}")
    if ACTIVE_CONTRACT.generation_id == "source_line_ownership_v1":
        payload = last_json_object(base / "stdout.txt")
    else:
        payload = select_structural_target_output(
            base / "stdout.txt",
            orientation=orientation,
            groups=TRACE_GROUPS,
            mode="endpoint",
            order="ab",
            exactly_one=False,
        )
    validate_target_payload(payload, orientation=orientation, groups=TRACE_GROUPS, mode="endpoint", order="ab", variant=variant)
    ownership = lane_ownership_oracle(base, variant, orientation)
    return {"cell": name, "variant": variant, "orientation": orientation, "binary_sha256": binary_sha, "tracer_sha256": tracer, "ownership": ownership, "trace_sha256": sha256(base / "dynamic_trace.pb"), "payload_manifest_sha256": sha256(base / "payload_manifest.json")}


def production_fingerprint(launch: Mapping[str, Any]) -> dict[str, Any]:
    timing = launch.get("timing", {})
    readiness = timing.get("readiness", {})
    return {"counters": launch.get("counters"), "terminal": timing.get("terminal_modeled_event_time"), "frontier": timing.get("completion_frontier"), "limiting": readiness.get("limiting_cycles"), "exposure": readiness.get("constraint_exposure_cycles"), "async_groups": readiness.get("async_groups")}


def replay_cell(root: Path, variant: str, orientation: str) -> dict[str, Any]:
    name = endpoint_name(variant, orientation)
    base = root / "simulator" / name
    first, second, command_path = base / "replay-1.json", base / "replay-2.json", base / "command.json"
    for path in (first, second, command_path):
        if not path.is_file():
            raise FileNotFoundError(str(path))
    if read_bytes(first) != read_bytes(second):
        raise AnalysisError(f"{name}: replay twins differ")
    try:
        execution_producer.validate_replay(
            root, base, execution_cell(orientation), variant
        )
    except execution_producer.CaptureError as error:
        raise AnalysisError(f"{name}: {error}") from error
    document = read_json(first)
    launch, _, observation = schema_observation(document, 18, name)
    expected = {"copies_observed": 64, "destination_wavefronts_observed": ACTIVE_CONTRACT.simulator_destination_wavefronts_per_copy * 64, "bypass_wavefronts_observed": ACTIVE_CONTRACT.bypass_wavefronts_per_copy * 64, "source_cohorts_observed": 512, "destination_lines_observed": ACTIVE_CONTRACT.destination_lines_per_copy * 64, "max_bank_wavefronts_sum": 192, "max_bypass_wavefronts_sum": 256, "cohort_groups_observed": 64, "cohort_copies_observed": 64, "cohort_response_memberships_observed": 2048, "cohort_topology_penalty_cycles": 7680, "cohort_overlap_groups_observed": 64}
    gates = {field: observation.get(field) == value for field, value in expected.items()}
    contract = production_contract(launch)
    if not all(contract.values()):
        raise AnalysisError(f"{name}: production/tail contract failed")
    schema = {field: observation.get(field) for field in SCHEMA18_FIELDS}
    result = {"cell": name, "variant": variant, "orientation": orientation, "aggregate_gates": gates, "schema18": schema, "production": production_fingerprint(launch), "replay_sha256": sha256(first)}
    if ACTIVE_CONTRACT.generation_id != "source_line_ownership_v1":
        # The coordinate-corrected campaign freezes equality over the complete
        # observation while preserving the accepted v1 report byte shape.
        result["schema18_complete"] = dict(observation)
    return result


def direct_non_timing_payload(
    root: Path, orientation: str, repetition: int
) -> dict[str, Any]:
    path = (
        root
        / "direct"
        / paired_cell_name(orientation, DIRECT_GROUPS)
        / f"repeat-{repetition}/result.json"
    )
    payload = read_json(path)
    expected = common_target_contract(orientation, DIRECT_GROUPS, "pair")
    expected["order"] = expected_order(repetition)
    mismatches = {
        field: {"observed": payload.get(field), "expected": value}
        for field, value in expected.items()
        if not same_json(payload.get(field), value)
    }
    if mismatches:
        raise AnalysisError(f"{path}: direct non-timing contract mismatch: {mismatches}")
    for variant in VARIANTS:
        arm = payload.get(variant)
        if (
            not isinstance(arm, dict)
            or arm.get("correct") is not True
            or arm.get("mismatches") != 0
            or arm.get("sum_consistent") is not True
        ):
            raise AnalysisError(f"{path}: invalid correctness for {variant}")
    return payload


def direct_issue_cell(
    root: Path,
    orientation: str,
    repetitions: Sequence[int],
    binary: Mapping[str, Any],
) -> dict[str, Any]:
    issue: list[float] = []
    for repetition in repetitions:
        validate_direct_artifact(root, orientation, repetition, binary)
        payload = direct_non_timing_payload(root, orientation, repetition)
        early = integer(payload["early_fanout"].get("issue_cycles"), "early issue")
        late = integer(payload["late_fanout"].get("issue_cycles"), "late issue")
        derived = (early - late) / DIRECT_GROUPS
        retained = payload.get("paired_issue_delta_cycles_per_group")
        if not isinstance(retained, (int, float)) or not math.isclose(
            float(retained), derived, rel_tol=0.0, abs_tol=1.0e-8
        ):
            raise AnalysisError("paired issue delta disagrees with arm counters")
        issue.append(derived)
    return {
        "orientation": orientation,
        "repeat_count": len(issue),
        "issue_delta_cycles_per_group": issue,
        "median_issue_delta_cycles_per_group": statistics.median(issue),
    }


def direct_wait_cell(
    root: Path,
    orientation: str,
    repetitions: Sequence[int],
    binary: Mapping[str, Any],
) -> dict[str, Any]:
    wait: list[float] = []
    for repetition in repetitions:
        validate_direct_artifact(root, orientation, repetition, binary)
        payload = direct_non_timing_payload(root, orientation, repetition)
        early = integer(payload["early_fanout"].get("wait_cycles"), "early wait")
        late = integer(payload["late_fanout"].get("wait_cycles"), "late wait")
        derived = (early - late) / DIRECT_GROUPS
        retained = payload.get("paired_wait_delta_cycles_per_group")
        if not isinstance(retained, (int, float)) or not math.isclose(
            float(retained), derived, rel_tol=0.0, abs_tol=1.0e-8
        ):
            raise AnalysisError("paired wait delta disagrees with arm counters")
        wait.append(derived)
    return {
        "orientation": orientation,
        "repeat_count": len(wait),
        "wait_delta_cycles_per_group": wait,
        "median_wait_delta_cycles_per_group": statistics.median(wait),
        "negative_wait_repeats": sum(value < 0 for value in wait),
    }


def validate_direct_artifact(
    root: Path,
    orientation: str,
    repetition: int,
    binary: Mapping[str, Any],
) -> dict[str, Any]:
    directory = root / "direct" / paired_cell_name(orientation, DIRECT_GROUPS) / f"repeat-{repetition}"
    required = {
        "result": directory / "result.json",
        "command": directory / "command.json",
        "run_manifest": directory / "direct.run.json",
        "stderr": directory / "stderr.txt",
    }
    absent = [str(path) for path in required.values() if not path.is_file()]
    if absent:
        raise FileNotFoundError(", ".join(absent))
    try:
        files = execution_producer.validate_direct_provenance(
            directory, execution_cell(orientation), repetition, binary
        )
    except execution_producer.CaptureError as error:
        raise AnalysisError(f"{directory}: {error}") from error
    direct_non_timing_payload(root, orientation, repetition)
    return {
        "orientation": orientation,
        "repetition": repetition,
        "process_completed": True,
        "files": files,
    }


def signal_gate(cell: Mapping[str, Any]) -> dict[str, bool]:
    return {"issue_median_exact_zero": cell["median_issue_delta_cycles_per_group"] == 0.0, "wait_median_at_most_negative_one": cell["median_wait_delta_cycles_per_group"] <= -1.0, "at_least_eight_negative_repeats": cell["negative_wait_repeats"] >= 8}


def smoke_gates(bundle: Mapping[str, Any], nvbit: Mapping[tuple[str, str], Mapping[str, Any]], simulator: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, bool]:
    first = simulator[("early_fanout", SMOKE_ORIENTATION)]
    second = simulator[("late_fanout", SMOKE_ORIENTATION)]
    return {"direct_artifact_valid": True, "ncu_contract_exact": bool(bundle["valid"]), "per_lane_ownership_exact": all(value["ownership"]["per_lane_ownership_exact"] and value["ownership"]["preload_vector_exact"] for value in nvbit.values()), "aggregate_topology_exact": all(all(value["aggregate_gates"].values()) for value in simulator.values()), "schema18_paired_tie": first["schema18"] == second["schema18"], "production_paired_tie": first["production"] == second["production"]}


def choose_status(*, errors: Sequence[str], missing: Sequence[str], structural: bool, smoke_complete: bool, smoke_passed: bool, smoke_only: bool, identification: Mapping[str, bool] | None, holdout: Mapping[str, bool] | None) -> tuple[str, int]:
    if errors: return "invalid_source_line_ownership_campaign", 1
    if structural and not missing: return "complete_line_ownership_structural_no_go", 0
    if missing or not smoke_complete: return "incomplete_source_line_ownership_campaign", 2
    if smoke_only: return ("source_line_ownership_smoke_passed", 0) if smoke_passed else ("complete_line_ownership_structural_no_go", 0)
    if identification is None: return "incomplete_source_line_ownership_campaign", 2
    if not all(identification.values()): return "complete_source_line_ownership_identification_no_go", 0
    if holdout is None: return "incomplete_source_line_ownership_campaign", 2
    if not all(holdout.values()): return "complete_source_line_ownership_holdout_no_go", 0
    return "complete_source_line_ownership_signal", 0


def deterministic_aggregate(report: Mapping[str, Any]) -> dict[str, Any]:
    nvbit = [
        {
            key: cell.get(key)
            for key in (
                "cell",
                "variant",
                "binary_sha256",
                "tracer_sha256",
                "trace_sha256",
                "payload_manifest_sha256",
            )
        }
        for cell in report.get("nvbit_cells", [])
    ]
    replay = [
        {
            "cell": cell.get("cell"),
            "variant": cell.get("variant"),
            "aggregate_gates": cell.get("aggregate_gates"),
            "schema18": cell.get("schema18"),
            "replay_sha256": cell.get("replay_sha256"),
        }
        for cell in report.get("simulator_cells", [])
    ]
    nvbit.sort(key=lambda cell: str(cell["cell"]))
    replay.sort(key=lambda cell: str(cell["cell"]))
    return {
        key: report.get(key)
        for key in (
            "schema_version",
            "assay",
            "mode",
            "status",
            "decision",
            "binary",
            "root",
            "counts",
            "smoke_gates",
            "identification_gate",
            "holdout_gate",
            "schema18_prediction",
            "criteria",
            "timing_promotion_authorized",
            "typed_visibility_tail_required",
        )
    } | {
        "static": report.get("static"),
        "direct_artifacts": report.get("direct_artifacts"),
        "ncu_bundles": report.get("ncu_bundles"),
        "nvbit_identities": nvbit,
        "replay_identities": replay,
        "missing_artifact_count": len(report.get("missing_artifacts", [])),
        "validation_error_count": len(report.get("validation_errors", [])),
    }


def decision_for_status(
    status: str,
    *,
    static: Mapping[str, Any] | None = None,
    smoke: Mapping[str, bool] | None = None,
    bundles: Sequence[Mapping[str, Any]] = (),
) -> str:
    if status == "complete_line_ownership_structural_no_go":
        if isinstance(static, Mapping) and static.get("assay_valid") is False:
            return "The static assay gate fails before live evidence or timing fields."
        failed = [
            name for name, passed in (smoke or {}).items() if passed is False
        ]
        if "ncu_contract_exact" in failed:
            for bundle in bundles:
                values = bundle.get("values")
                if not isinstance(values, Mapping):
                    continue
                observed = values.get("shared_wavefronts")
                if observed == {"early_fanout": 1280, "late_fanout": 1280}:
                    return (
                        "The frozen shared-wavefront contract fails before timing: "
                        "both NCU arms report 1,280, not 2,048; direct issue and "
                        "wait labels remain unread."
                    )
        if "direct_issue_exact_zero" in failed:
            return (
                "The direct issue-control smoke gate fails; wait labels remain unread."
            )
        if failed:
            return (
                "The smoke fails these structural gates before timing admission: "
                + ", ".join(sorted(failed))
                + "."
            )
        return "A structural gate fails before timing admission."
    if status == "source_line_ownership_smoke_passed":
        return "The prospective smoke passes and authorizes the frozen full campaign."
    if status == "complete_source_line_ownership_identification_no_go":
        return "The first-line-early identification does not show the frozen material signal."
    if status == "complete_source_line_ownership_holdout_no_go":
        return "The identification signal fails the untouched second-line-early holdout."
    if status == "complete_source_line_ownership_signal":
        return "The line-ownership signal passes identification and physical-line reversal."
    return status.replace("_", " ")


def run(root: Path, output: Path, output_md: Path, aggregate: Path, *, smoke_only: bool) -> int:
    locked_contract = begin_campaign()
    if locked_contract != ACTIVE_CONTRACT:
        raise AnalysisError("line-ownership analyzer contract changed after import")
    root = root.resolve()
    missing: list[str] = []
    errors: list[str] = []
    binary = static = None
    try: binary = binary_report(root)
    except FileNotFoundError as error: missing.append(str(error))
    except Exception as error: errors.append(str(error))
    try:
        static = static_report(root)
        errors.extend(static["artifact_errors"])
    except FileNotFoundError as error: missing.append(str(error))
    except Exception as error: errors.append(str(error))
    bundles: dict[tuple[str, int], Any] = {}
    traces: dict[tuple[str, str], Any] = {}
    replays: dict[tuple[str, str], Any] = {}
    direct_artifacts: dict[tuple[str, int], Any] = {}
    direct_issues: dict[str, Any] = {}
    direct_waits: dict[str, Any] = {}
    smoke = identification = holdout = None
    structural = bool(static and static["artifact_valid"] and not static["assay_valid"])
    if not missing and not errors and not structural and binary:
        try:
            direct_artifacts[(SMOKE_ORIENTATION, 1)] = validate_direct_artifact(
                root, SMOKE_ORIENTATION, 1, binary
            )
            bundles[(SMOKE_ORIENTATION, 1)] = validate_bundle(root, SMOKE_ORIENTATION, 1, binary)
            for variant in VARIANTS:
                traces[(variant, SMOKE_ORIENTATION)] = nvbit_cell(root, variant, SMOKE_ORIENTATION, binary["sha256"])
                replays[(variant, SMOKE_ORIENTATION)] = replay_cell(root, variant, SMOKE_ORIENTATION)
            require_cross_endpoint_tracer_identity(list(traces.values()))
            smoke = smoke_gates(bundles[(SMOKE_ORIENTATION, 1)], traces, replays)
            # Direct result bytes are identity-bound above, but issue fields
            # stay unread until every static/NCU/NVBit/replay gate passes.
            if all(smoke.values()):
                direct_issues[SMOKE_ORIENTATION] = direct_issue_cell(
                    root, SMOKE_ORIENTATION, (1,), binary
                )
                smoke["direct_issue_exact_zero"] = (
                    direct_issues[SMOKE_ORIENTATION][
                        "median_issue_delta_cycles_per_group"
                    ]
                    == 0.0
                )
            structural = not all(smoke.values())
        except FileNotFoundError as error: missing.append(str(error))
        except Exception as error: errors.append(str(error))
    smoke_complete = smoke is not None and not errors and not missing
    smoke_passed = smoke_complete and all(smoke.values())
    if smoke_passed and not smoke_only and binary:
        for orientation in ORIENTATIONS:
            for repetition in REPETITIONS:
                if (orientation, repetition) not in direct_artifacts:
                    try:
                        direct_artifacts[(orientation, repetition)] = (
                            validate_direct_artifact(
                                root, orientation, repetition, binary
                            )
                        )
                    except FileNotFoundError as error:
                        missing.append(str(error))
                    except Exception as error:
                        errors.append(str(error))
                if (orientation, repetition) not in bundles:
                    try: bundles[(orientation, repetition)] = validate_bundle(root, orientation, repetition, binary)
                    except FileNotFoundError as error: missing.append(str(error))
                    except Exception as error: errors.append(str(error))
            for variant in VARIANTS:
                key = (variant, orientation)
                if key not in traces:
                    try: traces[key] = nvbit_cell(root, variant, orientation, binary["sha256"])
                    except FileNotFoundError as error: missing.append(str(error))
                    except Exception as error: errors.append(str(error))
                if key not in replays:
                    try: replays[key] = replay_cell(root, variant, orientation)
                    except FileNotFoundError as error: missing.append(str(error))
                    except Exception as error: errors.append(str(error))
        if not errors and not missing:
            try: require_cross_endpoint_tracer_identity(list(traces.values()))
            except Exception as error: errors.append(str(error))
        if len(bundles) == 20 and len(traces) == 4 and len(replays) == 4 and not errors and not missing:
            structural = not all(value["valid"] for value in bundles.values()) or not all(all(value["aggregate_gates"].values()) for value in replays.values())
            for orientation in ORIENTATIONS:
                a, b = replays[("early_fanout", orientation)], replays[("late_fanout", orientation)]
                structural = structural or a["schema18"] != b["schema18"] or a["production"] != b["production"]
    if smoke_passed and not smoke_only and not structural and not errors and not missing and len(bundles) == 20:
        try:
            direct_issues[SMOKE_ORIENTATION] = direct_issue_cell(
                root, SMOKE_ORIENTATION, REPETITIONS, binary
            )
            issue_passed = (
                direct_issues[SMOKE_ORIENTATION][
                    "median_issue_delta_cycles_per_group"
                ]
                == 0.0
            )
            if issue_passed:
                direct_waits[SMOKE_ORIENTATION] = direct_wait_cell(
                    root, SMOKE_ORIENTATION, REPETITIONS, binary
                )
                identification = signal_gate(
                    {
                        **direct_issues[SMOKE_ORIENTATION],
                        **direct_waits[SMOKE_ORIENTATION],
                    }
                )
            else:
                identification = {
                    "issue_median_exact_zero": False,
                    "wait_fields_evaluated": False,
                }
        except Exception as error: errors.append(str(error))
        if identification and all(identification.values()) and not errors:
            try:
                direct_issues["second_line_early"] = direct_issue_cell(
                    root, "second_line_early", REPETITIONS, binary
                )
                holdout_issue_passed = (
                    direct_issues["second_line_early"][
                        "median_issue_delta_cycles_per_group"
                    ]
                    == 0.0
                )
                if holdout_issue_passed:
                    direct_waits["second_line_early"] = direct_wait_cell(
                        root, "second_line_early", REPETITIONS, binary
                    )
                    holdout = signal_gate(
                        {
                            **direct_issues["second_line_early"],
                            **direct_waits["second_line_early"],
                        }
                    )
                else:
                    holdout = {
                        "issue_median_exact_zero": False,
                        "wait_fields_evaluated": False,
                    }
            except Exception as error: errors.append(str(error))
    status, rc = choose_status(errors=errors, missing=missing, structural=structural, smoke_complete=smoke_complete, smoke_passed=smoke_passed, smoke_only=smoke_only, identification=identification, holdout=holdout)
    ordered_bundles = [bundles[key] for key in sorted(bundles)]
    decision = decision_for_status(
        status, static=static, smoke=smoke, bundles=ordered_bundles
    )
    report = {"schema_version": 1, "assay": "sm86_async_source_line_ownership", "mode": "smoke_only" if smoke_only else "full", "status": status, "decision": decision, "timing_promotion_authorized": False, "typed_visibility_tail_required": True, "binary": binary, "root": str(root), "counts": {"ncu_bundles": len(bundles), "ncu_reports": 7 * len(bundles), "ncu_rows": 14 * len(bundles), "nvbit_cells": len(traces), "simulator_cells": len(replays), "direct_artifacts_validated": len(direct_artifacts), "direct_issue_cells_read": len(direct_issues), "direct_wait_cells_read": len(direct_waits)}, "smoke_gates": smoke, "identification_gate": identification, "holdout_gate": holdout, "schema18_prediction": "exact paired tie / zero", "static": static, "direct_artifacts": [direct_artifacts[key] for key in sorted(direct_artifacts)], "ncu_bundles": ordered_bundles, "nvbit_cells": [traces[key] for key in sorted(traces)], "simulator_cells": [replays[key] for key in sorted(replays)], "direct_issue_cells": [direct_issues[key] for key in sorted(direct_issues)], "direct_wait_cells": [direct_waits[key] for key in sorted(direct_waits)], "criteria": {"static_passed": static.get("assay_valid") if static else None, "smoke_passed": smoke_passed if smoke_complete else None, "schema18_tie": smoke.get("schema18_paired_tie") if smoke else None, "identification_signal": all(identification.values()) if identification else None, "holdout_signal": all(holdout.values()) if holdout else None, "timing_unchanged": True}, "missing_artifacts": sorted(set(missing)), "validation_errors": errors}
    output.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    aggregate.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output_md.write_text(f"# Source-line ownership report\n\nStatus: `{status}`\n\n{decision}.\n")
    aggregate.write_text(json.dumps(deterministic_aggregate(report), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status, "output": str(output), "aggregate": str(aggregate), "errors": len(errors), "missing": len(missing)}, sort_keys=True))
    return rc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--aggregate-output", type=Path, default=DEFAULT_AGGREGATE)
    args = parser.parse_args(argv)
    output = args.output or args.root / "analysis/report.json"
    output_md = args.output_md or output.with_suffix(".md")
    return run(args.root, output, output_md, args.aggregate_output, smoke_only=args.smoke_only)


if __name__ == "__main__":
    raise SystemExit(main())
