#!/usr/bin/env python3
"""Capture static, direct, NVBit, and replay evidence for line ownership."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence

import capture_multiline_execution as support
from line_ownership_campaign_contract import (
    V1_CONTRACT,
    CampaignContract,
    ContractError,
    begin_campaign,
    configure_once,
    current_contract,
    artifact_identity as contract_artifact_identity,
    read_artifact_json,
    replace_artifact_json,
    require_canonical_evidence_root,
    strict_json_equal,
    validate_evidence_tree,
    validate_smoke_authorization as validate_contract_smoke_authorization,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CAMPAIGN_CONTRACT = current_contract()
DEFAULT_ROOT = CAMPAIGN_CONTRACT.evidence_root
AGGREGATE_REPORT = CAMPAIGN_CONTRACT.aggregate_report
BINARY = CAMPAIGN_CONTRACT.binary
SOURCE = CAMPAIGN_CONTRACT.source
MAKEFILE = CAMPAIGN_CONTRACT.makefile
NVBIT_CAPTURE = (
    REPO_ROOT / "workloads/research_probes/ldgsts_burst_release/capture_nvbit.py"
)
TRACER = REPO_ROOT / "tools/nvbit_tracer/tracer_tool/tracer_tool.so"
SIM_RUN = REPO_ROOT / "sim/target/release/sim-run"
ARCH_CARD = REPO_ROOT / "arch_profiles/cards/sm_86.json"
PCHASE_CARD = REPO_ROOT / "data/pchase/sm_86_static_card.json"

ORIENTATIONS = ("first", "second")
VARIANTS = ("early_fanout", "late_fanout")
REPETITIONS = tuple(range(1, 11))
TRACE_GROUPS = 64
DIRECT_GROUPS = 8192

def _apply_campaign_contract(contract: CampaignContract) -> None:
    global CAMPAIGN_CONTRACT, DEFAULT_ROOT, AGGREGATE_REPORT
    global BINARY, SOURCE, MAKEFILE, TRACE_GROUPS, DIRECT_GROUPS
    CAMPAIGN_CONTRACT = contract
    DEFAULT_ROOT = contract.evidence_root
    AGGREGATE_REPORT = contract.aggregate_report
    BINARY = contract.binary
    SOURCE = contract.source
    MAKEFILE = contract.makefile
    TRACE_GROUPS = contract.trace_groups
    DIRECT_GROUPS = contract.direct_groups
    support.BINARY = BINARY
    support.SOURCE = SOURCE
    support.MAKEFILE = MAKEFILE
    support.STATIC_CAMPAIGN_CONTRACT = (
        None if contract == V1_CONTRACT else contract.to_json()
    )


def configure_campaign(contract: CampaignContract) -> CampaignContract:
    configured = configure_once(contract)
    _apply_campaign_contract(configured)
    return configured


_apply_campaign_contract(CAMPAIGN_CONTRACT)

CaptureError = support.CaptureError


class StructuralTargetError(RuntimeError):
    """Raised when a selectively projected target object is malformed."""


_JSON_PRIMITIVE = re.compile(
    r"(?:true|false|null|-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
)
_MAX_FINITE_BINARY64 = Decimal("1.7976931348623157e308")


class _SelectiveTargetReader:
    """Decode selected object fields and only scan unselected JSON values."""

    def __init__(self, text: str, context: str) -> None:
        self.text = text
        self.context = context
        self.index = 0

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise StructuralTargetError(
                        f"{self.context}: duplicate JSON key {key!r}"
                    )
                result[key] = value
            return result

        def reject_constant(token: str) -> Any:
            raise StructuralTargetError(
                f"{self.context}: non-finite JSON constant {token!r}"
            )

        self.decoder = json.JSONDecoder(
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )

    def require_finite(self, value: Any) -> None:
        if type(value) is float and not math.isfinite(value):
            raise StructuralTargetError(
                f"{self.context}: non-finite decoded JSON number"
            )
        if isinstance(value, dict):
            for nested in value.values():
                self.require_finite(nested)
        elif isinstance(value, list):
            for nested in value:
                self.require_finite(nested)

    def whitespace(self) -> None:
        while self.index < len(self.text) and self.text[self.index].isspace():
            self.index += 1

    def expect(self, token: str) -> None:
        self.whitespace()
        if not self.text.startswith(token, self.index):
            raise StructuralTargetError(
                f"{self.context}: expected {token!r} at byte {self.index}"
            )
        self.index += len(token)

    def decoded_value(self) -> Any:
        self.whitespace()
        try:
            value, end = self.decoder.raw_decode(self.text, self.index)
        except (json.JSONDecodeError, ValueError) as error:
            raise StructuralTargetError(f"{self.context}: {error}") from error
        self.index = end
        self.require_finite(value)
        return value

    def skip_string(self) -> None:
        self.whitespace()
        if self.index >= len(self.text) or self.text[self.index] != '"':
            raise StructuralTargetError(
                f"{self.context}: expected JSON string at byte {self.index}"
            )
        index = self.index + 1
        while index < len(self.text):
            character = self.text[index]
            if character == '"':
                self.index = index + 1
                return
            if ord(character) < 0x20:
                raise StructuralTargetError(
                    f"{self.context}: control character in JSON string"
                )
            if character == "\\":
                index += 1
                if index >= len(self.text):
                    break
                escape = self.text[index]
                if escape == "u":
                    digits = self.text[index + 1 : index + 5]
                    if len(digits) != 4 or not re.fullmatch(r"[0-9a-fA-F]{4}", digits):
                        raise StructuralTargetError(
                            f"{self.context}: invalid Unicode escape in JSON string"
                        )
                    index += 5
                    continue
                if escape not in '"\\/bfnrt':
                    raise StructuralTargetError(
                        f"{self.context}: invalid JSON string escape {escape!r}"
                    )
            index += 1
        raise StructuralTargetError(f"{self.context}: truncated JSON string")

    def skip_object(self) -> None:
        self.expect("{")
        seen: set[str] = set()
        self.whitespace()
        if self.index < len(self.text) and self.text[self.index] == "}":
            self.index += 1
            return
        while True:
            key = self.decoded_value()
            if not isinstance(key, str):
                raise StructuralTargetError(f"{self.context}: non-string JSON key")
            if key in seen:
                raise StructuralTargetError(
                    f"{self.context}: duplicate JSON key {key!r}"
                )
            seen.add(key)
            self.expect(":")
            self.skip_value()
            self.whitespace()
            if self.index < len(self.text) and self.text[self.index] == ",":
                self.index += 1
                continue
            self.expect("}")
            return

    def skip_array(self) -> None:
        self.expect("[")
        self.whitespace()
        if self.index < len(self.text) and self.text[self.index] == "]":
            self.index += 1
            return
        while True:
            self.skip_value()
            self.whitespace()
            if self.index < len(self.text) and self.text[self.index] == ",":
                self.index += 1
                continue
            self.expect("]")
            return

    def skip_value(self) -> None:
        self.whitespace()
        if self.index >= len(self.text):
            raise StructuralTargetError(f"{self.context}: truncated JSON value")
        first = self.text[self.index]
        if first == '"':
            self.skip_string()
            return
        if first == "{":
            self.skip_object()
            return
        if first == "[":
            self.skip_array()
            return
        end = self.index
        while end < len(self.text) and self.text[end] not in ",]} \t\r\n":
            end += 1
        token = self.text[self.index:end]
        if not _JSON_PRIMITIVE.fullmatch(token):
            raise StructuralTargetError(
                f"{self.context}: malformed JSON primitive at byte {self.index}"
            )
        if token not in {"true", "false", "null"}:
            try:
                numeric = Decimal(token)
            except InvalidOperation as error:
                raise StructuralTargetError(
                    f"{self.context}: invalid JSON number at byte {self.index}"
                ) from error
            if not numeric.is_finite() or abs(numeric) > _MAX_FINITE_BINARY64:
                raise StructuralTargetError(
                    f"{self.context}: non-finite JSON number at byte {self.index}"
                )
        self.index = end

    def object(
        self, fields: Mapping[str, Any], *, reject_unselected: bool = False
    ) -> dict[str, Any]:
        self.expect("{")
        result: dict[str, Any] = {}
        seen: set[str] = set()
        self.whitespace()
        if self.index < len(self.text) and self.text[self.index] == "}":
            self.index += 1
        else:
            while True:
                key = self.decoded_value()
                if not isinstance(key, str):
                    raise StructuralTargetError(
                        f"{self.context}: non-string JSON key"
                    )
                if key in seen:
                    raise StructuralTargetError(
                        f"{self.context}: duplicate JSON key {key!r}"
                    )
                seen.add(key)
                self.expect(":")
                if key not in fields:
                    if reject_unselected:
                        raise StructuralTargetError(
                            f"{self.context}: unexpected JSON field {key!r}"
                        )
                    self.skip_value()
                elif isinstance(fields[key], Mapping):
                    result[key] = self.object(fields[key])
                else:
                    result[key] = self.decoded_value()
                self.whitespace()
                if self.index < len(self.text) and self.text[self.index] == ",":
                    self.index += 1
                    continue
                self.expect("}")
                break
        missing = sorted(set(fields) - set(result))
        if missing:
            raise StructuralTargetError(
                f"{self.context}: missing structural JSON fields {missing}"
            )
        return result


def structural_target_fields(
    expected: Mapping[str, Any], *, mode: str, context: str
) -> dict[str, Any]:
    fields: dict[str, Any] = {field: None for field in expected}
    arm_fields = {"correct": None, "mismatches": None}
    if mode == "pair":
        fields["order"] = None
        for variant in VARIANTS:
            fields[variant] = arm_fields
    elif mode == "endpoint":
        fields["variant"] = None
        fields["result"] = arm_fields
    else:
        raise StructuralTargetError(f"{context}: unsupported target mode {mode!r}")
    return fields


def select_structural_target_json(
    text: str,
    expected: Mapping[str, Any],
    *,
    mode: str,
    context: str,
) -> dict[str, Any]:
    """Project identity/topology/correctness without decoding timing labels."""

    fields = structural_target_fields(expected, mode=mode, context=context)
    reader = _SelectiveTargetReader(text, context)
    result = reader.object(fields)
    reader.whitespace()
    if reader.index != len(reader.text):
        raise StructuralTargetError(f"{context}: trailing JSON content")
    return result


def select_structural_complete_json(
    text: str,
    expected_target: Mapping[str, Any],
    *,
    mode: str,
    context: str,
) -> dict[str, Any]:
    """Decode an NCU completion record without decoding nested timing labels."""

    fields = {
        "schema_version": None,
        "valid": None,
        "binding": None,
        "files": None,
        "summary": None,
        "target_result": structural_target_fields(
            expected_target, mode=mode, context=f"{context}.target_result"
        ),
    }
    reader = _SelectiveTargetReader(text, context)
    result = reader.object(fields, reject_unselected=True)
    reader.whitespace()
    if reader.index != len(reader.text):
        raise StructuralTargetError(f"{context}: trailing JSON content")
    return result


def same_json(observed: Any, expected: Any) -> bool:
    if CAMPAIGN_CONTRACT == V1_CONTRACT:
        return observed == expected
    return strict_json_equal(observed, expected)


@dataclass(frozen=True)
class Cell:
    orientation: str

    def __post_init__(self) -> None:
        if self.orientation not in ORIENTATIONS:
            raise ValueError(f"unsupported orientation: {self.orientation}")

    @property
    def orientation_name(self) -> str:
        return f"{self.orientation}_line_early"

    @property
    def short(self) -> str:
        return "first" if self.orientation == "first" else "second"

    @property
    def direct_name(self) -> str:
        return f"async_source_line_ownership_{self.orientation_name}_g{DIRECT_GROUPS}"

    def endpoint_name(self, variant: str) -> str:
        if variant not in VARIANTS:
            raise ValueError(f"unsupported variant: {variant}")
        return (
            f"async_source_line_ownership_{variant}_{self.orientation_name}_"
            f"g{TRACE_GROUPS}"
        )


def is_smoke_direct(cell: Cell, repetition: int) -> bool:
    return cell.orientation == "first" and repetition == 1


def is_smoke_endpoint(cell: Cell) -> bool:
    return cell.orientation == "first"


def retained_binary(root: Path) -> dict[str, Any]:
    return support.retained_binary(root)


def validate_campaign_contract_binding(root: Path) -> Path | None:
    """Require the retained root to carry the active non-default contract."""

    if CAMPAIGN_CONTRACT == V1_CONTRACT:
        return None
    resolved = CAMPAIGN_CONTRACT.require_allowed_path(root)
    if resolved != CAMPAIGN_CONTRACT.require_allowed_path(
        CAMPAIGN_CONTRACT.evidence_root
    ):
        raise CaptureError(
            f"generation {CAMPAIGN_CONTRACT.generation_id} requires root "
            f"{CAMPAIGN_CONTRACT.evidence_root}, got {resolved}"
        )
    record = resolved / "campaign_contract.json"
    try:
        observed = read_artifact_json(CAMPAIGN_CONTRACT, record)
    except ContractError as error:
        raise CaptureError(f"{record}: campaign contract identity mismatch") from error
    if not strict_json_equal(observed, CAMPAIGN_CONTRACT.to_json()):
        raise CaptureError(f"{record}: campaign contract identity mismatch")
    return record


def campaign_manifest_inputs(
    root: Path, *, authorized: bool = False
) -> dict[str, Path]:
    record = validate_campaign_contract_binding(root)
    if record is None:
        return {}
    result = {"campaign_contract": record}
    if authorized:
        authorization = smoke_authorization_path(root)
        if not authorization.is_file():
            raise CaptureError(
                f"full artifact requires retained smoke authorization: {authorization}"
            )
        result["smoke_authorization"] = authorization
    return result


def campaign_document_fields(
    root: Path, *, authorized: bool = False
) -> dict[str, Any]:
    if CAMPAIGN_CONTRACT == V1_CONTRACT:
        return {}
    validate_campaign_contract_binding(root)
    result = {
        "generation_id": CAMPAIGN_CONTRACT.generation_id,
        "assay": CAMPAIGN_CONTRACT.assay_id,
        "root": str(CAMPAIGN_CONTRACT.require_allowed_path(root)),
        "campaign_contract": CAMPAIGN_CONTRACT.to_json(),
    }
    if authorized:
        authorization = smoke_authorization_path(root)
        if not authorization.is_file():
            raise CaptureError(
                f"full artifact requires retained smoke authorization: {authorization}"
            )
        result["smoke_authorization"] = contract_artifact_identity(
            CAMPAIGN_CONTRACT, authorization
        )
    return result


def smoke_authorization_path(root: Path) -> Path:
    return root / "analysis/smoke-authorization.json"


def require_full_campaign_authorization(
    root: Path,
    binary: Mapping[str, Any],
    report_path: Path | None = None,
) -> dict[str, Any]:
    validate_campaign_contract_binding(root)
    if report_path is None:
        retained = smoke_authorization_path(root)
        report_path = retained if retained.is_file() else AGGREGATE_REPORT
    try:
        report = read_artifact_json(CAMPAIGN_CONTRACT, report_path)
    except ContractError as error:
        raise CaptureError(
            f"full campaign requires a smoke-pass aggregate: {report_path}: {error}"
        ) from error
    try:
        return validate_contract_smoke_authorization(
            report,
            contract=CAMPAIGN_CONTRACT,
            root=root,
            binary=binary,
        )
    except ContractError as error:
        raise CaptureError(
            f"full campaign is not authorized by this root/binary smoke: {error}"
        ) from error


def bind_campaign_contract(root: Path) -> Path | None:
    """Bind non-default generation identity without changing v1 artifacts."""

    if CAMPAIGN_CONTRACT == V1_CONTRACT:
        return None
    resolved = CAMPAIGN_CONTRACT.require_allowed_path(root)
    if resolved != CAMPAIGN_CONTRACT.require_allowed_path(
        CAMPAIGN_CONTRACT.evidence_root
    ):
        raise CaptureError(
            f"generation {CAMPAIGN_CONTRACT.generation_id} requires root "
            f"{CAMPAIGN_CONTRACT.evidence_root}, got {resolved}"
        )
    record = resolved / "campaign_contract.json"
    expected = CAMPAIGN_CONTRACT.to_json()
    if record.exists() or record.is_symlink():
        if not strict_json_equal(
            read_artifact_json(CAMPAIGN_CONTRACT, record), expected
        ):
            raise CaptureError(f"{record}: campaign contract identity drift")
    else:
        replace_artifact_json(CAMPAIGN_CONTRACT, record, expected)
    return validate_campaign_contract_binding(resolved)


def trace_payload_manifest_path(destination: Path) -> Path:
    return destination / "payload_manifest.json"


def trace_payload_files(destination: Path) -> list[Path]:
    paths = [
        destination / "dynamic_trace.pb",
        destination / "stats.csv",
    ]
    if CAMPAIGN_CONTRACT != V1_CONTRACT:
        paths.append(destination / "stdout.txt")
    paths.extend(sorted((destination / "threadblocks").rglob("*.pb")))
    if (destination / "extra_info").is_dir():
        paths.extend(
            sorted(
                path
                for path in (destination / "extra_info").rglob("*")
                if path.is_file()
            )
        )
    unique = sorted(set(paths), key=lambda path: path.relative_to(destination).as_posix())
    if not unique:
        raise CaptureError(f"{destination}: no trace payload files")
    for path in unique:
        if not path.is_file() or path.is_symlink():
            raise CaptureError(f"{destination}: invalid trace payload {path}")
    return unique


def current_trace_payload_manifest(destination: Path) -> dict[str, Any]:
    entries = [
        {
            "path": path.relative_to(destination).as_posix(),
            "sha256": support.sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in trace_payload_files(destination)
    ]
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": 1,
        "kind": "line_ownership_trace_payload_manifest",
        "entries": entries,
        "aggregate_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def write_trace_payload_manifest(destination: Path) -> Path:
    path = trace_payload_manifest_path(destination)
    support.atomic_write_json(path, current_trace_payload_manifest(destination))
    return path


def validate_trace_payload_manifest(destination: Path) -> dict[str, Any]:
    path = trace_payload_manifest_path(destination)
    observed = support.read_json(path)
    expected = current_trace_payload_manifest(destination)
    if not same_json(observed, expected):
        raise CaptureError(f"{path}: trace payload identity drift")
    return observed


def add_run_manifest_output(
    manifest_path: Path, name: str, output_path: Path
) -> None:
    document = support.read_json(manifest_path)
    outputs = document.get("outputs")
    if not isinstance(outputs, dict):
        raise CaptureError(f"{manifest_path}: missing outputs")
    outputs[name] = support.file_identity(output_path)
    support.atomic_write_json(manifest_path, document)


def nvbit_output_paths(
    destination: Path, *, include_payload_manifest: bool
) -> dict[str, Path]:
    outputs = {
        "binding": destination / "campaign_binding.json",
        "complete": destination / "complete.json",
        "stats": destination / "stats.csv",
        "trace": destination / "dynamic_trace.pb",
    }
    if CAMPAIGN_CONTRACT != V1_CONTRACT:
        outputs["target_stdout"] = destination / "stdout.txt"
    if include_payload_manifest:
        outputs["payload_manifest"] = trace_payload_manifest_path(destination)
    return outputs


def expected_order(repetition: int) -> str:
    return "ab" if repetition % 2 else "ba"


def cells(smoke: bool, selector: str | None) -> list[Cell]:
    if smoke:
        if selector is not None:
            raise CaptureError("--smoke and --cells are mutually exclusive")
        return [Cell("first")]
    if selector is None:
        return [Cell(value) for value in ORIENTATIONS]
    result: list[Cell] = []
    for raw in selector.split(","):
        value = raw.strip().removesuffix("_line_early")
        if value not in ORIENTATIONS:
            raise CaptureError("--cells accepts first,second")
        cell = Cell(value)
        if cell not in result:
            result.append(cell)
    if not result:
        raise CaptureError("--cells selected no cells")
    return result


def repetitions(smoke: bool, selector: str | None) -> list[int]:
    if smoke:
        if selector is not None:
            raise CaptureError("--smoke and --repeats are mutually exclusive")
        return [1]
    if selector is None:
        return list(REPETITIONS)
    result: list[int] = []
    for raw in selector.split(","):
        try:
            value = int(raw.strip())
        except ValueError as error:
            raise CaptureError(f"invalid repetition: {raw!r}") from error
        if value not in REPETITIONS:
            raise CaptureError("repetitions must be in 1..10")
        if value not in result:
            result.append(value)
    if not result:
        raise CaptureError("--repeats selected no repetitions")
    return result


def target_arguments(
    cell: Cell,
    *,
    groups: int,
    markers: str,
    order: str | None = None,
    variant: str | None = None,
) -> list[str]:
    if (order is None) == (variant is None):
        raise ValueError("select exactly one paired order or endpoint variant")
    result = [
        "--groups",
        str(groups),
        "--orientation",
        cell.orientation_name,
    ]
    if order is not None:
        result.extend(("--order", order))
    else:
        result.extend(("--variant", str(variant)))
    result.extend(("--markers", markers))
    return result


def common_target_contract(cell: Cell, groups: int, mode: str) -> dict[str, Any]:
    contract = CAMPAIGN_CONTRACT
    return {
        "workload": contract.workload,
        "route": "l2",
        "mode": mode,
        "orientation": cell.orientation_name,
        "depth": contract.fixed_depth,
        "groups": groups,
        "warmup_groups": contract.warmup_groups,
        "lead_clock_samples": contract.fixed_lead_samples,
        "source_cohorts_per_copy": contract.source_cohorts_per_copy,
        "aligned_source_cohorts_per_copy": contract.aligned_source_cohorts_per_copy,
        "split_source_cohorts_per_copy": contract.split_source_cohorts_per_copy,
        "source_sectors_per_cohort": contract.source_sectors_per_cohort,
        "payload_bytes_per_cohort": contract.payload_bytes_per_cohort,
        "source_step_stride_bytes": contract.source_step_stride_bytes,
        "source_step_max_exclusive_byte": contract.source_step_max_exclusive_byte,
        "source_step_guard_bytes": contract.source_step_guard_bytes,
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
        "local_device_lts_lookups_per_copy": contract.local_device_lts_lookups_per_copy,
        "source_ring_bytes_per_arm": contract.source_ring_bytes_per_arm,
        "source_ring_steps_per_arm": contract.source_ring_steps_per_arm,
        "max_admitted_groups": contract.max_admitted_groups,
        "source_ring_no_wrap": True,
        "source_pool_bytes": contract.source_pool_bytes,
        "source_pool_alignment_bytes": contract.source_pool_alignment_bytes,
        "source_arm_a_offset_bytes": contract.source_arm_a_offset_bytes,
        "source_arm_b_offset_bytes": contract.source_arm_b_offset_bytes,
        "source_pool_single_allocation": True,
        "independent_source_arms": True,
        "source_vector_preserved": True,
        "preload_vector_preserved": True,
        "destination_multiset_preserved": True,
        "ownership_permutation_preserved": True,
        "static_ldgsts_positions_preserved": True,
        "result_store_after_intervals": True,
        "device": "NVIDIA GeForce RTX 3090 Ti",
        "compute_capability": "8.6",
        "correct": True,
    }


def validate_arm(value: Any, groups: int, context: str) -> None:
    if not isinstance(value, dict) or value.get("correct") is not True:
        raise CaptureError(f"{context}: invalid result arm")
    if not same_json(value.get("mismatches"), 0) or value.get(
        "sum_consistent"
    ) is not True:
        raise CaptureError(f"{context}: correctness/sum invariant failed")
    for field in ("issue_cycles", "wait_cycles", "total_cycles"):
        observed = value.get(field)
        if type(observed) is not int or observed < 0:
            raise CaptureError(f"{context}: invalid {field}")
    if value["issue_cycles"] + value["wait_cycles"] != value["total_cycles"]:
        raise CaptureError(f"{context}: cycle sum mismatch")
    if groups <= 0:
        raise CaptureError(f"{context}: invalid group count")


def validate_structural_arm(value: Any, context: str) -> None:
    if not isinstance(value, dict) or value.get("correct") is not True:
        raise CaptureError(f"{context}: invalid result arm")
    mismatches = value.get("mismatches")
    if type(mismatches) is not int or mismatches != 0:
        raise CaptureError(f"{context}: correctness mismatch count is not zero")


def validate_target(
    payload: Mapping[str, Any],
    cell: Cell,
    *,
    groups: int,
    mode: str,
    order: str = "ab",
    variant: str | None = None,
    structural_only: bool = False,
) -> None:
    expected = common_target_contract(cell, groups, mode)
    mismatches = {
        key: {"observed": payload.get(key), "expected": value}
        for key, value in expected.items()
        if not same_json(payload.get(key), value)
    }
    if mismatches:
        raise CaptureError(f"target contract mismatch: {mismatches}")
    if mode == "pair":
        if payload.get("order") != order:
            raise CaptureError("target pair order mismatch")
        for arm in VARIANTS:
            if structural_only:
                validate_structural_arm(payload.get(arm), f"target.{arm}")
            else:
                validate_arm(payload.get(arm), groups, f"target.{arm}")
    else:
        if payload.get("variant") != variant:
            raise CaptureError("target endpoint variant mismatch")
        if structural_only:
            validate_structural_arm(payload.get("result"), f"target.{variant}")
        else:
            validate_arm(payload.get("result"), groups, f"target.{variant}")


def direct_command(cell: Cell, repetition: int) -> list[str]:
    return [
        str(BINARY),
        *target_arguments(
            cell,
            groups=DIRECT_GROUPS,
            markers="off",
            order=expected_order(repetition),
        ),
    ]


def validate_direct_provenance(
    destination: Path,
    cell: Cell,
    repetition: int,
    binary: Mapping[str, Any],
) -> dict[str, Any]:
    result_path = destination / "result.json"
    command_path = destination / "command.json"
    command_document = support.read_json(command_path)
    timeout = command_document.get("configured_timeout_seconds")
    if not isinstance(timeout, int) or timeout < 1:
        raise CaptureError(f"{command_path}: invalid timeout")
    manifest = destination / "direct.run.json"
    support.validate_run_manifest(
        manifest,
        command=direct_command(cell, repetition),
        timeout=timeout,
        stdout_path=result_path,
        stderr_path=destination / "stderr.txt",
        input_paths={
            **campaign_manifest_inputs(
                destination.parents[2],
                authorized=not is_smoke_direct(cell, repetition),
            ),
            "probe_binary": BINARY,
        },
        output_paths={"result": result_path},
        environment_value=support.environment(),
    )
    expected = {
        "schema_version": 1,
        "cell": cell.orientation_name,
        "repetition": repetition,
        "order": expected_order(repetition),
        "configured_timeout_seconds": timeout,
        "command": direct_command(cell, repetition),
        "binary": dict(binary),
        "run_manifest": support.file_identity(manifest),
        **campaign_document_fields(
            destination.parents[2],
            authorized=not is_smoke_direct(cell, repetition),
        ),
    }
    if not same_json(command_document, expected):
        raise CaptureError(f"{command_path}: direct binding mismatch")
    return {
        "result": support.file_identity(result_path),
        "command": support.file_identity(command_path),
        "run_manifest": support.file_identity(manifest),
        "stderr": support.file_identity(destination / "stderr.txt"),
    }


def validate_direct(
    destination: Path,
    cell: Cell,
    repetition: int,
    binary: Mapping[str, Any],
) -> dict[str, Any]:
    validate_direct_provenance(destination, cell, repetition, binary)
    payload = support.read_json(destination / "result.json")
    validate_target(
        payload,
        cell,
        groups=DIRECT_GROUPS,
        mode="pair",
        order=expected_order(repetition),
    )
    return payload


def capture_direct(
    root: Path,
    selected_cells: Sequence[Cell],
    selected_repetitions: Sequence[int],
    timeout: int,
) -> None:
    binary = retained_binary(root)
    for cell in selected_cells:
        for repetition in selected_repetitions:
            destination = root / "direct" / cell.direct_name / f"repeat-{repetition}"
            result_path = destination / "result.json"
            if result_path.is_file():
                validate_direct_provenance(destination, cell, repetition, binary)
                print(f"direct existing: {cell.short} r{repetition}")
                continue
            support.ensure_output_directory(destination)
            command = direct_command(cell, repetition)
            manifest = destination / "direct.run.json"
            support.recorded_run(
                command,
                timeout=timeout,
                manifest_path=manifest,
                stdout_path=result_path,
                stderr_path=destination / "stderr.txt",
                input_paths={
                    **campaign_manifest_inputs(
                        root, authorized=not is_smoke_direct(cell, repetition)
                    ),
                    "probe_binary": BINARY,
                },
                output_paths={"result": result_path},
                environment_value=support.environment(),
            )
            support.atomic_write_json(
                destination / "command.json",
                {
                    "schema_version": 1,
                    "cell": cell.orientation_name,
                    "repetition": repetition,
                    "order": expected_order(repetition),
                    "configured_timeout_seconds": timeout,
                    "command": command,
                    "binary": binary,
                    "run_manifest": support.file_identity(manifest),
                    **campaign_document_fields(
                        root, authorized=not is_smoke_direct(cell, repetition)
                    ),
                },
            )
            validate_direct_provenance(destination, cell, repetition, binary)
            print(f"direct: {cell.short} r{repetition}", flush=True)


def nvbit_commands(
    destination: Path, cell: Cell, variant: str, target_timeout: int
) -> tuple[list[str], list[str]]:
    capture = [
        sys.executable,
        str(NVBIT_CAPTURE),
        "--output",
        str(destination),
        "--cell",
        cell.endpoint_name(variant),
        "--binary",
        str(BINARY),
        "--mode",
        "ncu",
        "--gpu",
        "0",
        "--timeout",
        str(target_timeout),
        "--target-args",
        *target_arguments(
            cell,
            groups=TRACE_GROUPS,
            markers="on",
            variant=variant,
        ),
    ]
    wrapper = [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "-p",
        "Delegate=yes",
        *capture,
    ]
    return capture, wrapper


def validate_nvbit(
    root: Path,
    destination: Path,
    cell: Cell,
    variant: str,
    binary: Mapping[str, Any],
) -> None:
    name = cell.endpoint_name(variant)
    complete = support.read_json(destination / "complete.json")
    binding = support.read_json(destination / "campaign_binding.json")
    if complete.get("complete") is not True or complete.get("key") != name:
        raise CaptureError(f"{destination}: invalid NVBit completion")
    if complete.get("binary_sha256") != binary["sha256"]:
        raise CaptureError(f"{destination}: NVBit binary mismatch")
    if binding.get("cell") != name or binding.get("output_contract", {}).get("passed") is not True:
        raise CaptureError(f"{destination}: invalid NVBit binding")
    run_root = root / "runs/nvbit" / name
    command_path = run_root / "command.json"
    command_document = support.read_json(command_path)
    target_timeout = command_document.get("configured_target_timeout_seconds")
    outer_timeout = command_document.get("configured_timeout_seconds")
    if not isinstance(target_timeout, int) or target_timeout < 1:
        raise CaptureError(f"{command_path}: invalid target timeout")
    if not isinstance(outer_timeout, int) or outer_timeout < target_timeout:
        raise CaptureError(f"{command_path}: invalid wrapper timeout")
    capture, wrapper = nvbit_commands(destination, cell, variant, target_timeout)
    manifest = run_root / "wrapper.run.json"
    payload_manifest = trace_payload_manifest_path(destination)
    validate_trace_payload_manifest(destination)
    outputs = nvbit_output_paths(destination, include_payload_manifest=True)
    support.validate_run_manifest(
        manifest,
        command=wrapper,
        timeout=outer_timeout,
        stdout_path=run_root / "wrapper.stdout.txt",
        stderr_path=run_root / "wrapper.stderr.txt",
        input_paths={
            **campaign_manifest_inputs(
                root, authorized=not is_smoke_endpoint(cell)
            ),
            "capture_helper": NVBIT_CAPTURE,
            "probe_binary": BINARY,
            "tracer": TRACER,
        },
        output_paths=outputs,
        environment_value=support.environment(),
    )
    expected = {
        "schema_version": 1,
        "cell": name,
        "variant": variant,
        "orientation": cell.orientation,
        "configured_target_timeout_seconds": target_timeout,
        "configured_timeout_seconds": outer_timeout,
        "capture_command": capture,
        "command": wrapper,
        "binary": dict(binary),
        "run_manifest": support.file_identity(manifest),
        "outputs": support.identities(outputs),
        **campaign_document_fields(root, authorized=not is_smoke_endpoint(cell)),
    }
    if not same_json(command_document, expected):
        raise CaptureError(f"{command_path}: NVBit wrapper binding mismatch")
    stdout_payload = None
    structural_only = CAMPAIGN_CONTRACT != V1_CONTRACT
    target_text = support.read_text(destination / "stdout.txt")
    expected_target = common_target_contract(cell, TRACE_GROUPS, "endpoint")
    for line_number, line in reversed(
        tuple(enumerate(target_text.splitlines(), start=1))
    ):
        if structural_only:
            if not line.lstrip().startswith("{"):
                continue
            try:
                value = select_structural_target_json(
                    line,
                    expected_target,
                    mode="endpoint",
                    context=f"{destination}/stdout.txt:{line_number}",
                )
            except StructuralTargetError as error:
                raise CaptureError(f"invalid target JSON: {error}") from error
        else:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
        if isinstance(value, dict):
            stdout_payload = value
            break
    if stdout_payload is None:
        raise CaptureError(f"{destination}: no target JSON")
    validate_target(
        stdout_payload,
        cell,
        groups=TRACE_GROUPS,
        mode="endpoint",
        variant=variant,
        structural_only=structural_only,
    )


def capture_nvbit(root: Path, selected_cells: Sequence[Cell], timeout: int) -> None:
    binary = retained_binary(root)
    for cell in selected_cells:
        for variant in VARIANTS:
            name = cell.endpoint_name(variant)
            destination = root / "nvbit" / name
            if (destination / "complete.json").is_file():
                validate_nvbit(root, destination, cell, variant, binary)
                print(f"nvbit existing: {name}")
                continue
            run_root = root / "runs/nvbit" / name
            support.ensure_output_directory(run_root)
            target_timeout = timeout
            outer_timeout = timeout + 120
            capture, wrapper = nvbit_commands(destination, cell, variant, target_timeout)
            manifest = run_root / "wrapper.run.json"
            outputs = nvbit_output_paths(
                destination, include_payload_manifest=False
            )
            support.recorded_run(
                wrapper,
                timeout=outer_timeout,
                manifest_path=manifest,
                stdout_path=run_root / "wrapper.stdout.txt",
                stderr_path=run_root / "wrapper.stderr.txt",
                input_paths={
                    **campaign_manifest_inputs(
                        root, authorized=not is_smoke_endpoint(cell)
                    ),
                    "capture_helper": NVBIT_CAPTURE,
                    "probe_binary": BINARY,
                    "tracer": TRACER,
                },
                output_paths=outputs,
                environment_value=support.environment(),
            )
            payload_manifest = write_trace_payload_manifest(destination)
            add_run_manifest_output(
                manifest, "payload_manifest", payload_manifest
            )
            outputs["payload_manifest"] = payload_manifest
            support.atomic_write_json(
                run_root / "command.json",
                {
                    "schema_version": 1,
                    "cell": name,
                    "variant": variant,
                    "orientation": cell.orientation,
                    "configured_target_timeout_seconds": target_timeout,
                    "configured_timeout_seconds": outer_timeout,
                    "capture_command": capture,
                    "command": wrapper,
                    "binary": binary,
                    "run_manifest": support.file_identity(manifest),
                    "outputs": support.identities(outputs),
                    **campaign_document_fields(
                        root, authorized=not is_smoke_endpoint(cell)
                    ),
                },
            )
            validate_nvbit(root, destination, cell, variant, binary)
            print(f"nvbit: {name}", flush=True)


def simulator_build_command() -> list[str]:
    return [
        "cargo",
        "build",
        "--release",
        "--manifest-path",
        str(REPO_ROOT / "sim/Cargo.toml"),
        "-p",
        "sim-run",
    ]


def capture_simulator_build(root: Path, timeout: int) -> None:
    destination = root / "simulator-build"
    manifest = destination / "build.run.json"
    stdout = destination / "stdout.txt"
    stderr = destination / "stderr.txt"
    command = simulator_build_command()
    inputs = {
        **campaign_manifest_inputs(root),
        "cargo_manifest": REPO_ROOT / "sim/Cargo.toml",
        "cargo_lock": REPO_ROOT / "sim/Cargo.lock",
    }
    outputs = {"sim_run": SIM_RUN}
    if manifest.is_file():
        validate_simulator_build(root)
        return
    support.ensure_output_directory(destination)
    support.recorded_run(
        command,
        timeout=timeout,
        manifest_path=manifest,
        stdout_path=stdout,
        stderr_path=stderr,
        input_paths=inputs,
        output_paths=outputs,
        environment_value=support.environment(),
    )


def validate_simulator_build(root: Path) -> None:
    destination = root / "simulator-build"
    manifest = destination / "build.run.json"
    configured = support.read_json(manifest).get("configured_timeout_seconds")
    if not isinstance(configured, int) or configured < 1:
        raise CaptureError(f"{manifest}: invalid timeout")
    support.validate_run_manifest(
        manifest,
        command=simulator_build_command(),
        timeout=configured,
        stdout_path=destination / "stdout.txt",
        stderr_path=destination / "stderr.txt",
        input_paths={
            **campaign_manifest_inputs(root),
            "cargo_manifest": REPO_ROOT / "sim/Cargo.toml",
            "cargo_lock": REPO_ROOT / "sim/Cargo.lock",
        },
        output_paths={"sim_run": SIM_RUN},
        environment_value=support.environment(),
    )


def replay_command(root: Path, cell: Cell, variant: str) -> list[str]:
    return [
        str(SIM_RUN),
        "--trace",
        str(root / "nvbit" / cell.endpoint_name(variant)),
        "--control",
        str(root / "static/control_bits.json"),
        "--arch-card",
        str(ARCH_CARD),
        "--pchase-card",
        str(PCHASE_CARD),
        "--mode",
        "l1-l2",
    ]


def validate_replay(root: Path, destination: Path, cell: Cell, variant: str) -> None:
    first = destination / "replay-1.json"
    second = destination / "replay-2.json"
    if (
        not first.is_file()
        or not second.is_file()
        or support.read_bytes(first) != support.read_bytes(second)
    ):
        raise CaptureError(f"{destination}: incomplete or non-deterministic replay twins")
    support.read_json(first)
    command_path = destination / "command.json"
    command_document = support.read_json(command_path)
    timeout = command_document.get("configured_timeout_seconds")
    if not isinstance(timeout, int) or timeout < 1:
        raise CaptureError(f"{command_path}: invalid timeout")
    command = replay_command(root, cell, variant)
    trace_directory = root / "nvbit" / cell.endpoint_name(variant)
    validate_trace_payload_manifest(trace_directory)
    inputs = {
        **campaign_manifest_inputs(root, authorized=not is_smoke_endpoint(cell)),
        "arch_card": ARCH_CARD,
        "control": root / "static/control_bits.json",
        "pchase_card": PCHASE_CARD,
        "sim_run": SIM_RUN,
        "trace": trace_directory / "dynamic_trace.pb",
        "trace_payload_manifest": trace_payload_manifest_path(trace_directory),
    }
    manifests: dict[str, Any] = {}
    for repeat in (1, 2):
        manifest = destination / f"replay-{repeat}.run.json"
        support.validate_run_manifest(
            manifest,
            command=command,
            timeout=timeout,
            stdout_path=destination / f"replay-{repeat}.json",
            stderr_path=destination / f"stderr-{repeat}.txt",
            input_paths=inputs,
            output_paths={"replay": destination / f"replay-{repeat}.json"},
            environment_value=support.environment(),
        )
        manifests[f"replay_{repeat}"] = support.file_identity(manifest)
    expected = {
        "schema_version": 1,
        "cell": cell.endpoint_name(variant),
        "variant": variant,
        "orientation": cell.orientation,
        "configured_timeout_seconds": timeout,
        "command": command,
        "inputs": support.identities(inputs),
        "outputs": support.identities({"replay_1": first, "replay_2": second}),
        "run_manifests": manifests,
        **campaign_document_fields(root, authorized=not is_smoke_endpoint(cell)),
    }
    if not same_json(command_document, expected):
        raise CaptureError(f"{command_path}: replay binding mismatch")


def capture_replay(root: Path, selected_cells: Sequence[Cell], timeout: int) -> None:
    capture_simulator_build(root, timeout)
    for cell in selected_cells:
        for variant in VARIANTS:
            name = cell.endpoint_name(variant)
            trace = root / "nvbit" / name / "dynamic_trace.pb"
            if not trace.is_file():
                raise CaptureError(f"missing trace: {trace}")
            validate_trace_payload_manifest(root / "nvbit" / name)
            destination = root / "simulator" / name
            if (destination / "command.json").is_file():
                validate_replay(root, destination, cell, variant)
                print(f"replay existing: {name}")
                continue
            support.ensure_output_directory(destination)
            command = replay_command(root, cell, variant)
            inputs = {
                **campaign_manifest_inputs(
                    root, authorized=not is_smoke_endpoint(cell)
                ),
                "arch_card": ARCH_CARD,
                "control": root / "static/control_bits.json",
                "pchase_card": PCHASE_CARD,
                "sim_run": SIM_RUN,
                "trace": trace,
                "trace_payload_manifest": trace_payload_manifest_path(
                    root / "nvbit" / name
                ),
            }
            manifests: dict[str, Any] = {}
            for repeat in (1, 2):
                manifest = destination / f"replay-{repeat}.run.json"
                support.recorded_run(
                    command,
                    timeout=timeout,
                    manifest_path=manifest,
                    stdout_path=destination / f"replay-{repeat}.json",
                    stderr_path=destination / f"stderr-{repeat}.txt",
                    input_paths=inputs,
                    output_paths={"replay": destination / f"replay-{repeat}.json"},
                    environment_value=support.environment(),
                )
                manifests[f"replay_{repeat}"] = support.file_identity(manifest)
            support.atomic_write_json(
                destination / "command.json",
                {
                    "schema_version": 1,
                    "cell": name,
                    "variant": variant,
                    "orientation": cell.orientation,
                    "configured_timeout_seconds": timeout,
                    "command": command,
                    "inputs": support.identities(inputs),
                    "outputs": support.identities(
                        {
                            "replay_1": destination / "replay-1.json",
                            "replay_2": destination / "replay-2.json",
                        }
                    ),
                    "run_manifests": manifests,
                    **campaign_document_fields(
                        root, authorized=not is_smoke_endpoint(cell)
                    ),
                },
            )
            validate_replay(root, destination, cell, variant)
            print(f"replay: {name}", flush=True)


def validate_selected(
    root: Path,
    selected_cells: Sequence[Cell],
    selected_repetitions: Sequence[int],
) -> None:
    binary = retained_binary(root)
    validate_simulator_build(root)
    for cell in selected_cells:
        for repetition in selected_repetitions:
            validate_direct_provenance(
                root / "direct" / cell.direct_name / f"repeat-{repetition}",
                cell,
                repetition,
                binary,
            )
        for variant in VARIANTS:
            validate_nvbit(
                root,
                root / "nvbit" / cell.endpoint_name(variant),
                cell,
                variant,
                binary,
            )
            validate_replay(
                root,
                root / "simulator" / cell.endpoint_name(variant),
                cell,
                variant,
            )


def main(argv: Sequence[str] | None = None) -> int:
    _apply_campaign_contract(begin_campaign())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--phase",
        choices=("all", "static", "direct", "nvbit", "replay"),
        default="all",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cells", help="first,second")
    parser.add_argument("--repeats", help="comma-separated subset in 1..10")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)
    try:
        if args.timeout < 1:
            raise CaptureError("--timeout must be positive")
        selected_cells = cells(args.smoke, args.cells)
        selected_repetitions = repetitions(args.smoke, args.repeats)
        root = require_canonical_evidence_root(CAMPAIGN_CONTRACT, args.root)
        if root.exists():
            validate_evidence_tree(CAMPAIGN_CONTRACT, root)
        if args.validate_only:
            validate_campaign_contract_binding(root)
        else:
            bind_campaign_contract(root)
        full_execution = not args.smoke and (
            args.validate_only
            or args.phase in {"all", "direct", "nvbit", "replay"}
        )
        if full_execution:
            require_full_campaign_authorization(root, retained_binary(root))
        if args.validate_only:
            validate_selected(root, selected_cells, selected_repetitions)
            print(
                json.dumps(
                    {
                        "valid": True,
                        "root": str(root),
                        "cells": [cell.orientation_name for cell in selected_cells],
                        "repetitions": selected_repetitions,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.phase in {"all", "static"}:
            support.capture_static(root, args.timeout)
        if args.phase in {"all", "direct"}:
            capture_direct(root, selected_cells, selected_repetitions, args.timeout)
        if args.phase in {"all", "nvbit"}:
            capture_nvbit(root, selected_cells, args.timeout)
        if args.phase in {"all", "replay"}:
            capture_replay(root, selected_cells, args.timeout)
        validate_evidence_tree(CAMPAIGN_CONTRACT, root)
        return 0
    except (CaptureError, ContractError, OSError, ValueError) as error:
        print(f"line ownership execution capture: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
