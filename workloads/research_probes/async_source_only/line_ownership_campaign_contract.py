#!/usr/bin/env python3
"""Immutable generation contracts for SM86 source-line ownership campaigns."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.local_labels.immutable_artifact import (  # noqa: E402
    ImmutableArtifactError,
    PinnedDirectoryReader,
    ensure_directory,
    read_regular,
    replace_regular_bytes,
)


class ContractError(RuntimeError):
    """A campaign attempted to mutate or escape its frozen generation."""


@dataclass(frozen=True)
class CampaignContract:
    generation_id: str
    assay_id: str
    binary: Path
    source: Path
    makefile: Path
    evidence_root: Path
    aggregate_report: Path
    smoke_pass_status: str
    ncu_shared_wavefronts_per_copy: int
    simulator_destination_wavefronts_per_copy: int
    payload_wavefronts_per_copy: int
    bypass_wavefronts_per_copy: int
    destination_lines_per_copy: int
    copies_per_group: int = 1
    source_cohorts_per_copy: int = 8
    aligned_source_cohorts_per_copy: int = 4
    split_source_cohorts_per_copy: int = 4
    source_sectors_per_cohort: int = 4
    payload_bytes_per_cohort: int = 64
    source_sectors_per_copy: int = 32
    response_memberships_per_copy: int = 32
    scalar_ldg_instructions_per_copy: int = 1
    scalar_ldg_sectors_per_copy: int = 24
    scalar_preload_lines_per_copy: int = 8
    ldgsts_instructions_per_copy: int = 1
    ldgsts_sectors_per_copy: int = 32
    unique_ldgsts_lines_per_copy: int = 12
    max_cohort_bank_wavefronts: int = 3
    max_cohort_bypass_wavefronts: int = 4
    cohort_topology_penalty_cycles_per_copy: int = 120
    local_device_lts_lookups_per_copy: int = 56
    trace_groups: int = 64
    direct_groups: int = 8192
    warmup_groups: int = 64
    fixed_depth: int = 1
    fixed_lead_samples: int = 16
    source_step_stride_bytes: int = 2048
    source_step_max_exclusive_byte: int = 1968
    source_step_guard_bytes: int = 80
    source_ring_bytes_per_arm: int = 32 << 20
    source_ring_steps_per_arm: int = 16384
    max_admitted_groups: int = 16320
    source_pool_bytes: int = 64 << 20
    source_pool_alignment_bytes: int = 128
    source_arm_a_offset_bytes: int = 0
    source_arm_b_offset_bytes: int = 32 << 20
    workload: str = "async_source_line_ownership"
    orientations: tuple[str, str] = (
        "first_line_early",
        "second_line_early",
    )
    variants: tuple[str, str] = ("early_fanout", "late_fanout")
    forbidden_paths: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        string_fields = (
            self.generation_id,
            self.assay_id,
            self.smoke_pass_status,
            self.workload,
        )
        if any(type(value) is not str or not value for value in string_fields):
            raise ContractError("campaign string fields must be nonempty strings")
        if (
            type(self.orientations) is not tuple
            or type(self.variants) is not tuple
            or any(type(value) is not str or not value for value in self.orientations)
            or any(type(value) is not str or not value for value in self.variants)
        ):
            raise ContractError("campaign orientations and variants must be string tuples")
        values = asdict(self)
        non_integer_fields = {
            "generation_id",
            "assay_id",
            "binary",
            "source",
            "makefile",
            "evidence_root",
            "aggregate_report",
            "smoke_pass_status",
            "workload",
            "orientations",
            "variants",
            "forbidden_paths",
        }
        integer_fields = set(values) - non_integer_fields
        malformed_integers = sorted(
            name for name in integer_fields if type(values[name]) is not int
        )
        if malformed_integers:
            raise ContractError(
                f"campaign integer fields have wrong types: {malformed_integers}"
            )
        if any(
            values[name] <= 0
            for name in integer_fields - {"source_arm_a_offset_bytes"}
        ) or values["source_arm_a_offset_bytes"] < 0:
            raise ContractError("campaign structural integers are out of range")
        path_fields = (
            self.binary,
            self.source,
            self.makefile,
            self.evidence_root,
            self.aggregate_report,
            *self.forbidden_paths,
        )
        if type(self.forbidden_paths) is not tuple or any(
            not isinstance(path, Path) for path in path_fields
        ):
            raise ContractError("campaign path fields must be Path values")
        if any(not path.is_absolute() for path in path_fields):
            raise ContractError("campaign contract paths must be absolute")
        if self.binary.name == self.source.name:
            raise ContractError("binary and CUDA source must be distinct paths")
        if len(set(self.orientations)) != 2 or len(set(self.variants)) != 2:
            raise ContractError("orientations and variants must each be distinct pairs")
        counts = (
            self.ncu_shared_wavefronts_per_copy,
            self.simulator_destination_wavefronts_per_copy,
            self.payload_wavefronts_per_copy,
            self.bypass_wavefronts_per_copy,
            self.destination_lines_per_copy,
            self.copies_per_group,
            self.source_cohorts_per_copy,
            self.source_sectors_per_copy,
            self.source_step_stride_bytes,
            self.source_step_max_exclusive_byte,
            self.source_step_guard_bytes,
            self.source_ring_bytes_per_arm,
            self.source_ring_steps_per_arm,
            self.max_admitted_groups,
            self.source_pool_bytes,
            self.source_pool_alignment_bytes,
            self.trace_groups,
            self.direct_groups,
            self.warmup_groups,
        )
        if any(value <= 0 for value in counts):
            raise ContractError("campaign structural counts must be positive")
        if (
            self.aligned_source_cohorts_per_copy
            + self.split_source_cohorts_per_copy
            != self.source_cohorts_per_copy
            or self.source_cohorts_per_copy * self.source_sectors_per_cohort
            != self.source_sectors_per_copy
        ):
            raise ContractError("campaign source-cohort counts are incoherent")
        if (
            self.source_step_max_exclusive_byte + self.source_step_guard_bytes
            != self.source_step_stride_bytes
            or self.source_ring_steps_per_arm * self.source_step_stride_bytes
            != self.source_ring_bytes_per_arm
            or self.source_arm_b_offset_bytes != self.source_ring_bytes_per_arm
            or self.source_pool_bytes != 2 * self.source_ring_bytes_per_arm
        ):
            raise ContractError("campaign source-ring geometry is incoherent")

    def to_json(self) -> dict[str, Any]:
        value = asdict(self)
        for field in (
            "binary",
            "source",
            "makefile",
            "evidence_root",
            "aggregate_report",
        ):
            value[field] = str(value[field])
        value["forbidden_paths"] = [str(path) for path in self.forbidden_paths]
        value["orientations"] = list(self.orientations)
        value["variants"] = list(self.variants)
        return value

    def require_allowed_path(self, path: Path) -> Path:
        raw = Path(path)
        if "\x00" in os.fspath(raw) or ".." in raw.parts:
            raise ContractError(f"generation {self.generation_id} rejects path {raw}")
        resolved = Path(os.path.abspath(os.fspath(raw)))
        for forbidden in self.forbidden_paths:
            forbidden = Path(os.path.abspath(os.fspath(forbidden)))
            if resolved == forbidden or forbidden in resolved.parents:
                raise ContractError(
                    f"generation {self.generation_id} forbids predecessor path {resolved}"
                )
        return resolved


def strict_json_equal(observed: Any, expected: Any) -> bool:
    """Compare JSON-shaped values without Python's bool/int coercions."""

    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(observed) == set(expected) and all(
            strict_json_equal(observed[key], expected[key]) for key in expected
        )
    if isinstance(expected, (list, tuple)):
        return len(observed) == len(expected) and all(
            strict_json_equal(left, right)
            for left, right in zip(observed, expected, strict=True)
        )
    return observed == expected


_TREE_LIMITS = {
    "max_files": 4096,
    "max_directories": 1024,
    "max_total_file_bytes": 1 << 30,
    "max_total_relative_path_bytes": 1 << 20,
}


def validate_evidence_tree(contract: CampaignContract, root: Path) -> None:
    """Reject symlinks, hardlinks, path swaps, and oversized evidence trees."""

    if contract == V1_CONTRACT:
        return
    checked = contract.require_allowed_path(root)
    try:
        with PinnedDirectoryReader(checked) as reader:
            reader.snapshot_tree(**_TREE_LIMITS)
    except (ImmutableArtifactError, OSError) as error:
        raise ContractError(f"unsafe coordinate evidence tree {checked}: {error}") from error


def read_artifact_bytes(contract: CampaignContract, path: Path) -> bytes:
    checked = contract.require_allowed_path(path)
    try:
        return read_regular(checked)
    except (ImmutableArtifactError, OSError) as error:
        raise ContractError(f"cannot read coordinate artifact {checked}: {error}") from error


def read_artifact_text(contract: CampaignContract, path: Path) -> str:
    try:
        return read_artifact_bytes(contract, path).decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError(f"coordinate artifact is not UTF-8: {path}") from error


class _StrictJSONError(ValueError):
    """Internal parse failure for JSON constructs rejected by the contract."""


def strict_json_loads(
    payload: str | bytes | bytearray, *, context: str | Path
) -> Any:
    """Decode standards-compliant JSON with unique object keys.

    Python's default decoder silently keeps the last duplicate object key and
    accepts the non-standard ``NaN``/``Infinity`` constants.  Coordinate
    evidence is identity-bearing, so both forms are ambiguous and rejected at
    every nesting level.  V1 callers retain their predecessor decoder by
    selecting it before calling this coordinate-only helper.
    """

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _StrictJSONError(f"duplicate object key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> Any:
        raise _StrictJSONError(f"non-finite numeric constant {token!r}")

    def require_finite_numbers(value: Any) -> None:
        if type(value) is float and not math.isfinite(value):
            raise _StrictJSONError(f"non-finite decoded number {value!r}")
        if isinstance(value, dict):
            for nested in value.values():
                require_finite_numbers(nested)
        elif isinstance(value, list):
            for nested in value:
                require_finite_numbers(nested)

    try:
        value = json.loads(
            payload,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
        require_finite_numbers(value)
        return value
    except (json.JSONDecodeError, UnicodeDecodeError, _StrictJSONError) as error:
        raise ContractError(f"{context}: invalid strict JSON: {error}") from error


def read_artifact_json(contract: CampaignContract, path: Path) -> dict[str, Any]:
    try:
        payload = read_artifact_bytes(contract, path)
        value = (
            json.loads(payload)
            if contract == V1_CONTRACT
            else strict_json_loads(payload, context=path)
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ContractError(f"coordinate artifact is not JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"coordinate artifact is not a JSON object: {path}")
    return value


def artifact_identity(contract: CampaignContract, path: Path) -> dict[str, Any]:
    checked = contract.require_allowed_path(path)
    payload = read_artifact_bytes(contract, checked)
    return {
        "path": str(checked),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def replace_artifact_text(
    contract: CampaignContract, path: Path, text: str
) -> None:
    checked = contract.require_allowed_path(path)
    try:
        ensure_directory(checked.parent)
        replace_regular_bytes(checked, text.encode("utf-8"))
    except (ImmutableArtifactError, OSError) as error:
        raise ContractError(f"cannot publish coordinate artifact {checked}: {error}") from error


def replace_artifact_json(
    contract: CampaignContract, path: Path, value: Any
) -> None:
    replace_artifact_text(
        contract,
        path,
        json.dumps(value, indent=2, sort_keys=True) + "\n",
    )


V1_CONTRACT = CampaignContract(
    generation_id="source_line_ownership_v1",
    assay_id="sm86_async_source_line_ownership",
    binary=HERE / "async_source_line_ownership",
    source=HERE / "async_source_line_ownership.cu",
    makefile=HERE / "Makefile.line_ownership",
    evidence_root=REPO_ROOT
    / "results/captures/sm86_async_line_ownership_20260831_v1",
    aggregate_report=REPO_ROOT
    / "data/local_labels/sm86_async_line_ownership_report.json",
    smoke_pass_status="source_line_ownership_smoke_passed",
    ncu_shared_wavefronts_per_copy=32,
    simulator_destination_wavefronts_per_copy=32,
    payload_wavefronts_per_copy=32,
    bypass_wavefronts_per_copy=32,
    destination_lines_per_copy=20,
)


COORDINATE_CONTRACT = CampaignContract(
    generation_id="source_line_ownership_coordinates_v1",
    assay_id="sm86_async_source_line_ownership_coordinates_v1",
    binary=HERE / "async_source_line_ownership_coordinates",
    source=HERE / "async_source_line_ownership.cu",
    makefile=HERE / "Makefile.line_ownership_coordinates",
    evidence_root=REPO_ROOT
    / "results/captures/sm86_async_line_coordinates_20260831_v1",
    aggregate_report=REPO_ROOT
    / "data/local_labels/sm86_async_line_coordinates_report.json",
    smoke_pass_status="line_coordinates_smoke_passed",
    ncu_shared_wavefronts_per_copy=20,
    simulator_destination_wavefronts_per_copy=24,
    payload_wavefronts_per_copy=32,
    bypass_wavefronts_per_copy=32,
    destination_lines_per_copy=20,
    forbidden_paths=(
        REPO_ROOT
        / "results/captures/sm86_async_line_ownership_20260831_v1",
        REPO_ROOT
        / "results/captures/sm86_async_multiline_response_20260831_v1",
        REPO_ROOT / "data/local_labels/sm86_async_line_ownership_report.json",
        REPO_ROOT
        / "data/local_labels/sm86_async_multiline_response_envelope_report.json",
    ),
)


_LOCK = Lock()
_ACTIVE = V1_CONTRACT
_EXPLICITLY_CONFIGURED = False
_STARTED = False


def configure_once(contract: CampaignContract) -> CampaignContract:
    """Select one generation before work starts; later mutation is rejected."""

    global _ACTIVE, _EXPLICITLY_CONFIGURED
    with _LOCK:
        if _STARTED:
            if contract == _ACTIVE:
                return _ACTIVE
            raise ContractError("campaign contract is locked after work started")
        if _EXPLICITLY_CONFIGURED and contract != _ACTIVE:
            raise ContractError("campaign contract was already configured")
        _ACTIVE = contract
        _EXPLICITLY_CONFIGURED = True
        return _ACTIVE


def current_contract() -> CampaignContract:
    with _LOCK:
        return _ACTIVE


def begin_campaign() -> CampaignContract:
    """Lock and return the active contract at the first campaign action."""

    global _STARTED
    with _LOCK:
        _STARTED = True
        return _ACTIVE


def campaign_started() -> bool:
    with _LOCK:
        return _STARTED


def _coordinate_smoke_authorization(
    contract: CampaignContract,
    root: Path,
    binary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generation_id": contract.generation_id,
        "campaign_contract": contract.to_json(),
        "assay": contract.assay_id,
        "mode": "smoke_only",
        "status": contract.smoke_pass_status,
        "root": str(contract.require_allowed_path(root)),
        "binary": dict(binary),
        "decision": "Fresh coordinate smoke passed.",
        "counts": {
            "ncu_bundles": 1,
            "ncu_reports": 7,
            "ncu_rows": 14,
            "nvbit_cells": 2,
            "simulator_cells": 2,
            "direct_artifacts_validated": 1,
        },
        "smoke_gates": {
            "static_assay_valid": True,
            "ncu_contract_exact": True,
            "per_lane_ownership_exact": True,
            "aggregate_coordinates_exact": True,
            "schema18_paired_tie": True,
            "production_paired_tie": True,
            "direct_provenance_valid": True,
        },
        "criteria": {
            "static_passed": True,
            "smoke_passed": True,
            "identification_signal": None,
            "holdout_signal": None,
            "schema18_tie": True,
            "timing_unchanged": True,
        },
        "label_reads": {
            "identification_issue_read": False,
            "identification_wait_read": False,
            "holdout_issue_read": False,
            "holdout_wait_read": False,
        },
        "identification_gate": None,
        "holdout_gate": None,
        "schema18_prediction": "exact paired tie / zero",
        "timing_promotion_authorized": False,
        "typed_visibility_tail_required": True,
        "missing_artifact_count": 0,
        "validation_error_count": 0,
    }


def validate_smoke_authorization(
    document: Mapping[str, Any],
    *,
    contract: CampaignContract,
    root: Path,
    binary: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one retained smoke pass under the active generation.

    The coordinate generation has a closed deterministic authorization
    document. V1 intentionally retains its predecessor's looser selected-field
    contract so existing reports and callers remain valid.
    """

    if contract != V1_CONTRACT:
        expected = _coordinate_smoke_authorization(contract, root, binary)
        observed = dict(document)
        if not strict_json_equal(observed, expected):
            differing = sorted(
                key
                for key in set(observed) | set(expected)
                if key not in observed
                or key not in expected
                or not strict_json_equal(observed[key], expected[key])
            )
            raise ContractError(
                "coordinate smoke authorization is not the exact frozen "
                f"document; differing fields={differing}"
            )
        return observed

    expected_scalars = {
        "schema_version": 1,
        "assay": contract.assay_id,
        "mode": "smoke_only",
        "status": contract.smoke_pass_status,
        "root": str(root.resolve()),
        "binary": dict(binary),
        "timing_promotion_authorized": False,
        "typed_visibility_tail_required": True,
        "missing_artifact_count": 0,
        "validation_error_count": 0,
    }
    mismatches = {
        field: {"observed": document.get(field), "expected": value}
        for field, value in expected_scalars.items()
        if document.get(field) != value
    }
    smoke_gates = document.get("smoke_gates")
    criteria = document.get("criteria")
    if (
        not isinstance(smoke_gates, dict)
        or not smoke_gates
        or not all(value is True for value in smoke_gates.values())
    ):
        mismatches["smoke_gates"] = {
            "observed": smoke_gates,
            "expected": "all true",
        }
    if not isinstance(criteria, dict) or criteria.get("smoke_passed") is not True:
        mismatches["criteria.smoke_passed"] = {
            "observed": (
                None if not isinstance(criteria, dict) else criteria.get("smoke_passed")
            ),
            "expected": True,
        }
    if mismatches:
        raise ContractError(
            f"v1 smoke authorization does not satisfy predecessor contract: {mismatches}"
        )
    return dict(document)
