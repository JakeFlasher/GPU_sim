#!/usr/bin/env python3
"""Immutable generation contracts for SM86 source-line ownership campaigns."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]


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
        path_fields = (
            self.binary,
            self.source,
            self.makefile,
            self.evidence_root,
            self.aggregate_report,
            *self.forbidden_paths,
        )
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
        resolved = path.resolve()
        for forbidden in self.forbidden_paths:
            forbidden = forbidden.resolve()
            if resolved == forbidden or forbidden in resolved.parents:
                raise ContractError(
                    f"generation {self.generation_id} forbids predecessor path {resolved}"
                )
        return resolved


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
