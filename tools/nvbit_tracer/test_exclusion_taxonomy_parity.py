"""Parity tests for producer/consumer typed-exclusion taxonomy.

The NVBit tracer's exclusion-sidecar mechanism has THREE coupled
typed-reason layers that must stay in lockstep across the C++
producer and the Python consumer:

1. **Producer use sites**: ``tools/nvbit_tracer/tracer_tool/tracer_tool.cu``
   contains the only call sites of ``nvbit_tracer::aggregate(...)``.
   Each call's last argument (a ``std::string("...")``) is a typed
   reason recorded against an instruction whose memory shape the
   tracer cannot represent on the wire (e.g. unsupported memory
   op-class, unresolved address, cross-allocation lane span).
   These are the originating emission sites for typed reasons.

2. **Sidecar writers**: ``tools/nvbit_tracer/tracer_tool/exclusion_aggregator.cc``
   serializes the aggregator's snapshot to one JSON file per typed
   reason via ``write_one_typed_sidecar(...)``. The first argument
   carries the filename (``/<reason>.json``) and the second carries
   the typed_reason field literal.

3. **Consumer admit list**: ``import_nvbit_exclusion._SCHEMA_FILES``
   maps each typed reason to its JSON-Schema file under
   ``workloads/simbench_bw/schemas/``. The importer's ``validate(...)``
   and ``validate_path(...)`` dispatch on the typed reason against
   this dict.

Drift between any two layers breaks the AC-19 parity invariant. The
most insidious drift is a new ``aggregate(..., "new_reason")`` site
added in (1) without the matching writer call in (2), schema file,
or admit-list update in (3) — the producer would record entries
into the in-memory aggregator that never reach a sidecar file.

Validator-only typed reasons (emitted by the consumer when it
detects malformed sidecars or missing metadata, NOT produced by
the C++ tracer) are documented separately and not included in
the parity check. Examples: ``allocations_sidecar_missing``,
``allocation_id_missing``, ``allocation_id_unknown``,
``allocation_id_duplicate``, ``allocations_sidecar_malformed``,
``lane_address_outside_allocation_range``, ``sidecar_empty``,
``sidecar_not_a_list``, ``sidecar_typed_reason_mismatch``.
"""

from __future__ import annotations

import ast
import io
import json
import re
import sys
import tokenize
import unittest
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import import_nvbit_exclusion  # noqa: E402


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TRACER_TOOL_CU = (
    _REPO_ROOT / "tools" / "nvbit_tracer" / "tracer_tool"
    / "tracer_tool.cu"
)
_AGGREGATOR_CC = (
    _REPO_ROOT / "tools" / "nvbit_tracer" / "tracer_tool"
    / "exclusion_aggregator.cc"
)
_REPEATABILITY_AGGREGATOR_PY = (
    _REPO_ROOT / "tools" / "local_labels" / "repeatability_aggregator.py"
)
_TRACE_BUDGETS_PY = (
    _REPO_ROOT / "tools" / "cloud" / "trace_budgets.py"
)
_RUN_SUITE_PY = (
    _REPO_ROOT / "workloads" / "simbench_bw" / "run_suite.py"
)
_VALIDATE_PY = (
    _REPO_ROOT / "workloads" / "simbench_bw" / "validate.py"
)
_SCHEMA_DIR = _REPO_ROOT / "workloads" / "simbench_bw" / "schemas"
_PLAN_PATH = (
    _REPO_ROOT / "plans" / "nonroot_gpu_cache_observability_plan.md"
)
_MEMORY_OP_CLASS_H = (
    _REPO_ROOT
    / "tools"
    / "nvbit_tracer"
    / "tracer_tool"
    / "memory_op_class.h"
)
_PROTO_INSTRUCTION = (
    _REPO_ROOT
    / "tools"
    / "nvbit_tracer"
    / "traces_enhanced"
    / "dynamic_trace"
    / "instruction.proto"
)
_ADAPTIVE_SAMPLER_LEDGER_PY = (
    _REPO_ROOT
    / "tools"
    / "adaptive_sampler"
    / "trace_convergence_ledger.py"
)


PRODUCER_TYPED_REASONS_EXPECTED: frozenset[str] = frozenset({
    "unsupported_memory_path",
    "unmapped_memory_range",
    "cross_allocation_lane_span",
})

# Local-labels typed reasons emitted by
# tools/local_labels/repeatability_aggregator.py inside label_excluded.json
# entries. Plan AC-19 enumerates the first three as the at-minimum label
# taxonomy; this constant pins the production set in lockstep so future
# additions or removals are caught at test time.
LOCAL_LABELS_TYPED_REASONS_EXPECTED: frozenset[str] = frozenset({
    "label_repeatability_failed",
    "label_repeatability_missing",
    # `label_provenance_inconsistent` is the original-plan reason
    # (cross-binary-sha cell rejection) and stays in the
    # AC19_PLAN_ENUMERATED_TYPED_REASONS set below. The 6 entries
    # below are round-added per the C-AC-C lockstep discipline:
    # the aggregator's `_validate_provenance(...)` runs five
    # fail-fast checks (no manifests; manifests disagree on
    # cell-invariant fields; per-launch sidecar lacks matching
    # manifest by `(repeat_index, launch_index)`; per-launch
    # sidecar's `binary_sha256` disagrees with its matching
    # manifest; raw profiler_version banner does not normalize to
    # the pinned protocol form), and the strict admitted-cell
    # schema gate additionally produces `label_schema_admission_failed`
    # when the producer/consumer wire pin cannot be loaded
    # fail-closed.
    "label_provenance_inconsistent",
    "label_provenance_unavailable",
    "label_provenance_duplicate",
    "label_provenance_per_launch_missing",
    "label_provenance_mismatch",
    "label_profiler_version_drift",
    "label_schema_admission_failed",
    # Build-variant identity fail-closed gate (the register-cap variant
    # fold): a run-manifest or per-launch sidecar whose register_cap and
    # variant_id disagree (or whose variant_id is malformed) fails the
    # cell CLOSED before any admitted record is emitted, so the
    # variant_id-keyed grouping and the admitted register_cap can never
    # describe two different build variants for one cell
    # (repeatability_aggregator._variant_consistency_error).
    "label_variant_identity_inconsistent",
})

# Trace-budget typed reasons emitted by tools/cloud/trace_budgets.py.
# The trace-budget surface uses a different shape from the NVBit-tracer
# and local-labels surfaces: the typed reason is a sidecar filename slug
# defined in a `<UPPER_NAME>_SIDECAR: str = "<reason>.json"` constant,
# and the schema file uses the `<reason>.schema.json` naming convention
# (no `nvbit_` prefix).
TRACE_BUDGET_TYPED_REASONS_EXPECTED: frozenset[str] = frozenset({
    "trace_volume_exceeded",
    "nvbit_slowdown_budget_exceeded",
    "nvbit_drop_rate_exceeded",
})

# Sector-stream typed reasons emitted by
# tools/sector_stream/generator.py for the C-AC-F deliverable. The
# family has two surfaces: the producer-side constant set
# (`SECTOR_STREAM_TYPED_REASONS` in generator.py) and the JSON-schema
# `typed_reason.enum` in
# workloads/simbench_bw/schemas/sector_stream.schema.json. Two of
# the four reasons (`no_lanes_active`, `lane_address_length_invalid`)
# also appear in the original-plan AC-19 enumeration; the other two
# (`data_width_invalid`, `sector_stream_class_unsupported`) are
# round-added under ROUND_ADDED_TYPED_REASONS.
SECTOR_STREAM_TYPED_REASONS_EXPECTED: frozenset[str] = frozenset({
    "no_lanes_active",
    "lane_address_length_invalid",
    "data_width_invalid",
    "sector_stream_class_unsupported",
})

# Divergence-proxies typed-unavailable reasons emitted by
# tools/divergence_proxies/proxies.py for the C-AC-G deliverable. The
# family has two surfaces: the producer-side constant set
# (`DIVERGENCE_PROXIES_TYPED_REASONS` in proxies.py) and the JSON-schema
# `typed_reason.enum` on the typed-unavailable branch of
# workloads/simbench_bw/schemas/divergence_proxies.schema.json. One of
# the eight reasons (`outstanding_load_density_unavailable_no_phase_records`)
# also appears in the original-plan AC-19 enumeration; the other seven
# are round-added under ROUND_ADDED_TYPED_REASONS.
DIVERGENCE_PROXIES_TYPED_REASONS_EXPECTED: frozenset[str] = frozenset({
    "outstanding_load_density_unavailable_no_phase_records",
    "stride_entropy_unavailable_no_lane_table",
    "throughput_saturation_slope_unavailable_no_sweep",
    "pc_repeated_miss_likely_streams_unavailable_no_pc_data",
    "md_loads_per_kilo_inst_unavailable_no_instruction_count",
    "unique_sectors_per_warp_inst_unavailable_no_records",
    "unique_lines_per_warp_inst_unavailable_no_records",
    "sectors_per_request_unavailable_no_records",
})

# HRD-profile typed reasons emitted by tools/hrd_profiler/profiler.py for
# the AC-6 deliverable (Loop 2 R17 / task11). The family has two surfaces:
# the producer-side constant set (`HRD_PROFILE_TYPED_REASONS` in
# profiler.py) and the JSON-schema `typed_reason.enum` in
# workloads/simbench_bw/schemas/hrd_profile.schema.json. Three of the 17
# reasons (cache_model_capacity_missing, cache_model_capacity_inconsistent,
# cache_model_underconfigured) are pinned in the original-plan AC-19
# enumeration; the other 14 are round-added under
# ROUND_ADDED_TYPED_REASONS.
HRD_PROFILE_TYPED_REASONS_EXPECTED: frozenset[str] = frozenset({
    "sector_reuse_cold",
    "sector_reuse_infinite",
    "line_reuse_cold",
    "line_reuse_infinite",
    "allocation_reuse_cold",
    "allocation_reuse_infinite",
    "region_reuse_cold",
    "region_reuse_infinite",
    "reuse_distance_unobservable",
    "cache_model_capacity_missing",
    "cache_model_capacity_inconsistent",
    "cache_model_capacity_static_card_disagreement",
    "cache_model_geometry_missing",
    "cache_model_geometry_inconsistent",
    "cache_model_geometry_static_card_disagreement",
    "cache_model_underconfigured",
    # R54: defense-in-depth budget cap on the HRD producer's
    # input size. Fired when sector_stream.jsonl exceeds
    # `HRD_MAX_SECTOR_STREAM_RECORDS`.
    "hrd_compute_budget_exceeded",
})

# Sharing-locality typed reasons emitted by
# tools/sharing_locality/classifier.py for the original-plan
# sharing-locality clause family (Loop 2 R22 / task12). The
# producer-side constant set
# (`SHARING_LOCALITY_TYPED_REASONS` in
# `tools/sharing_locality/classifier.py`) and the schema enum at
# `workloads/simbench_bw/schemas/sharing_locality.schema.json`
# `$defs.typed_reason.enum` must enumerate the same 7 reasons.
# All seven are round-added (none of these are in the original-plan
# AC-19 enumeration).
SHARING_LOCALITY_TYPED_REASONS_EXPECTED: frozenset[str] = frozenset({
    "intra_lane_unobservable",
    "intra_warp_unobservable",
    "inter_warp_same_cta_unobservable",
    "inter_cta_same_sm_projection_required",
    "inter_cta_cross_sm_projection_required",
    "sequential_kernel_projection_required",
    "private_cold_projection_required",
})

# Order-projections typed reasons emitted by
# tools/order_projections/projector.py (Loop 2 R25 / task13 /
# the order-projections clause). The SM-local unavailable arm
# uses the original-plan AC-19 enumerated reason
# (``projection_unavailable_sm_local``) per Codex's R25 review
# directive #1; the other 6 are round-added.
ORDER_PROJECTIONS_TYPED_REASONS_EXPECTED: frozenset[str] = frozenset({
    "lane_projection_unobservable",
    "warp_local_projection_unobservable",
    "cta_local_projection_unobservable",
    "projection_unavailable_sm_local",
    "kernel_l2_projection_unobservable",
    "sequential_allocation_projection_unobservable_no_allocations_sidecar",
    "insufficient_admitted_projections",
})

# Perturbation-run-ledger typed reasons emitted by
# tools/perturbation_harness/ledger.py (Loop 2 R31 / task15 / the
# perturbation-harness clause). The plan-enumerated reason
# ``modifier_optimized_away`` is in AC19_PLAN_ENUMERATED_TYPED_REASONS;
# ``correctness_equivalence_failed`` and ``perturbation_compile_failed``
# are round-added; the six ``arch_feature_unavailable_*`` reasons are
# instances of the original-plan wildcard family ``arch_feature_unavailable_*``
# and are documented in ROUND_ADDED_TYPED_REASONS below as wildcard-family
# instances rather than literal AC19 strings.
PERTURBATION_RUN_LEDGER_TYPED_REASONS_EXPECTED: frozenset[str] = frozenset({
    "modifier_optimized_away",
    "correctness_equivalence_failed",
    "arch_feature_unavailable_l2_prefetch_64b",
    "arch_feature_unavailable_l2_prefetch_128b",
    "arch_feature_unavailable_l2_prefetch_256b",
    "arch_feature_unavailable_l2_persisting",
    "arch_feature_unavailable_eviction_buffer",
    "arch_feature_unavailable_carveout",
    "perturbation_compile_failed",
})

# Trace-convergence typed reasons emitted by the adaptive-sampler
# clause (Loop 2 R37 / task17 / AC-11). Producer-side constant
# `TRACE_CONVERGENCE_TYPED_REASONS` lives in
# `tools/adaptive_sampler/trace_convergence_ledger.py`; each typed
# reason has a dedicated `oneOf` branch in
# `workloads/simbench_bw/schemas/trace_convergence_confidence.schema.json`
# with `properties.typed_reason.const`. The plan-enumerated reason
# `trace_convergence_not_reached_budget_cap` is in
# AC19_PLAN_ENUMERATED_TYPED_REASONS; the other three
# (`mrc_quantile_underconfigured`, `hrd_layer_unavailable`,
# `trace_convergence_disabled`) are round-added under
# ROUND_ADDED_TYPED_REASONS.
TRACE_CONVERGENCE_TYPED_REASONS_EXPECTED: frozenset[str] = frozenset({
    "trace_convergence_not_reached_budget_cap",
    "mrc_quantile_underconfigured",
    "hrd_layer_unavailable",
    "trace_convergence_disabled",
    "trace_convergence_window_count_underconfigured",
})

# AC-19 plan-enumerated typed reasons (see
# plans/nonroot_gpu_cache_observability_plan.md AC-19 positive tests).
# `arch_feature_unavailable_*` is documented as a wildcard family in the
# plan and is not enumerated as a single string; concrete instances would
# match a prefix-style check rather than literal equality.
AC19_PLAN_ENUMERATED_TYPED_REASONS: frozenset[str] = frozenset({
    "unmapped_memory_range",
    "trace_volume_exceeded",
    "modifier_optimized_away",
    "nvml_alignment_gap",
    "label_repeatability_failed",
    "label_repeatability_missing",
    "label_provenance_inconsistent",
    "unsupported_memory_path",
    "dual_run_manifest_missing",
    "nvbit_slowdown_budget_exceeded",
    "nvbit_drop_rate_exceeded",
    "arch_unsupported",
    "cache_model_capacity_missing",
    "cache_model_capacity_inconsistent",
    "cache_model_underconfigured",
    "nvml_cadence_mismatch",
    "forbidden_surface_attempted",
    "lane_address_length_invalid",
    "no_lanes_active",
    "projection_unavailable_sm_local",
    "trace_convergence_not_reached_budget_cap",
    "outstanding_load_density_unavailable_no_phase_records",
})

# Producer typed reasons added beyond the plan's AC-19 enumeration during
# implementation. The plan says "at minimum", so producers may legitimately
# emit additional taxonomy entries; this set documents which ones.
ROUND_ADDED_TYPED_REASONS: frozenset[str] = frozenset({
    "cross_allocation_lane_span",  # Loop 1 R5: multi-non-zero-allocation-id
                                    # warp instruction, distinct from
                                    # unmapped_memory_range
    "data_width_invalid",           # Loop 2 R5 / C-AC-F: sector-stream
                                    # generator rejection when data_width
                                    # is not in {1, 2, 4, 8, 16}.
    "sector_stream_class_unsupported",  # Loop 2 R5 / C-AC-F: sector-stream
                                    # generator rejection for memory_op_class
                                    # outside {GLOBAL, LOCAL}; the named
                                    # 7 classes are deferred to a later
                                    # loop's per-class handling.
    # Loop 2 R8 / C-AC-G: divergence-proxies typed-unavailable reasons
    # the proxy module emits when the trace lacks the evidence axis a
    # given proxy needs. Only `outstanding_load_density_unavailable_no_phase_records`
    # is in the original-plan AC-19 enumeration; the seven below are
    # round-added per the C-AC-C lockstep discipline.
    "stride_entropy_unavailable_no_lane_table",
    "throughput_saturation_slope_unavailable_no_sweep",
    "pc_repeated_miss_likely_streams_unavailable_no_pc_data",
    "md_loads_per_kilo_inst_unavailable_no_instruction_count",
    "unique_sectors_per_warp_inst_unavailable_no_records",
    "unique_lines_per_warp_inst_unavailable_no_records",
    "sectors_per_request_unavailable_no_records",
    # Loop 2 R17 / task11 / AC-6: HRD-profile typed reasons. 14
    # round-added reasons land here: 9 reuse-state reasons (8
    # cold/infinite per layer + reuse_distance_unobservable), 4
    # round-added cache-model corroboration/geometry reasons, and 1
    # compute-budget reason. The 3 plan-enumerated HRD reasons
    # (cache_model_capacity_missing, cache_model_capacity_inconsistent,
    # cache_model_underconfigured) stay in
    # AC19_PLAN_ENUMERATED_TYPED_REASONS.
    "sector_reuse_cold",
    "sector_reuse_infinite",
    "line_reuse_cold",
    "line_reuse_infinite",
    "allocation_reuse_cold",
    "allocation_reuse_infinite",
    "region_reuse_cold",
    "region_reuse_infinite",
    "reuse_distance_unobservable",
    "cache_model_capacity_static_card_disagreement",
    "cache_model_geometry_missing",
    "cache_model_geometry_inconsistent",
    "cache_model_geometry_static_card_disagreement",
    "hrd_compute_budget_exceeded",
    # Loop 2 R22 / task12 / the sharing-locality clause: all 7
    # sharing-locality typed reasons are round-added (none in the
    # original-plan AC-19 enumeration).
    "intra_lane_unobservable",
    "intra_warp_unobservable",
    "inter_warp_same_cta_unobservable",
    "inter_cta_same_sm_projection_required",
    "inter_cta_cross_sm_projection_required",
    "sequential_kernel_projection_required",
    "private_cold_projection_required",
    # Loop 2 R25 / task13 / the order-projections clause: 6
    # round-added reasons; the seventh
    # (``projection_unavailable_sm_local``) is in the
    # original-plan AC-19 enumeration above and intentionally
    # excluded from this round-added set.
    "lane_projection_unobservable",
    "warp_local_projection_unobservable",
    "cta_local_projection_unobservable",
    "kernel_l2_projection_unobservable",
    "sequential_allocation_projection_unobservable_no_allocations_sidecar",
    "insufficient_admitted_projections",
    # Loop 2 R31 / task15 / the perturbation-harness clause: 2
    # round-added reasons. The plan-enumerated reason
    # `modifier_optimized_away` is in
    # AC19_PLAN_ENUMERATED_TYPED_REASONS; the 6
    # `arch_feature_unavailable_*` reasons are listed below
    # as instances of the original-plan wildcard family
    # (the plan documents `arch_feature_unavailable_*` as a
    # prefix-style family; the perturbation-harness clause's
    # 6 concrete instantiations live in ROUND_ADDED_TYPED_REASONS
    # rather than the literal AC-19 set so the strict
    # subset/superset parity check still holds).
    "correctness_equivalence_failed",
    "perturbation_compile_failed",
    "arch_feature_unavailable_l2_prefetch_64b",
    "arch_feature_unavailable_l2_prefetch_128b",
    "arch_feature_unavailable_l2_prefetch_256b",
    "arch_feature_unavailable_l2_persisting",
    "arch_feature_unavailable_eviction_buffer",
    "arch_feature_unavailable_carveout",
    # Loop 2 R37 / task17 / AC-11: 3 round-added trace-convergence
    # typed reasons. The plan-enumerated reason
    # `trace_convergence_not_reached_budget_cap` is in
    # AC19_PLAN_ENUMERATED_TYPED_REASONS; the 3 below are pure
    # round-added per the C-AC-C lockstep discipline.
    "mrc_quantile_underconfigured",
    "hrd_layer_unavailable",
    "trace_convergence_disabled",
    # Loop 2 R38 / task17 rework: 4th round-added trace-convergence
    # typed reason. R37 fabricated convergence by duplicating one
    # HRD bundle into K+1 windows; Codex's R37 review rejected the
    # synthetic admission path and required a typed-unavailable arm
    # that honestly names the missing evidence when the trace is
    # admittedly sized but does not have enough cumulative bytes
    # for K consecutive windows. The 5th typed reason fires when
    # `actual_trace_bytes < (K + 1) * TRACE_WINDOW_SIZE_BYTES`.
    "trace_convergence_window_count_underconfigured",
    # Local-labels per-launch provenance + schema-admission
    # round-added reasons. The original-plan AC-19 enumerates only
    # `label_provenance_inconsistent` for the cross-binary-sha cell
    # rejection; the granular family below was introduced as the
    # local-label exception's collection pipeline matured. The 5
    # `_validate_provenance(...)` failure modes (no manifests;
    # cell-invariant disagreement; per-launch coverage gap; per-
    # launch sidecar mismatch; profiler-version drift) plus the
    # strict admitted-cell schema gate (label_schema_admission_failed)
    # are committed under the C-AC-C lockstep discipline and
    # documented at `tools/local_labels/repeatability_aggregator.py`'s
    # module docstring.
    "label_provenance_unavailable",
    "label_provenance_duplicate",
    "label_provenance_per_launch_missing",
    "label_provenance_mismatch",
    "label_profiler_version_drift",
    "label_schema_admission_failed",
    # Build-variant identity fail-closed gate (the register-cap variant
    # fold): a run-manifest or per-launch sidecar whose register_cap and
    # variant_id disagree (or whose variant_id is malformed) fails the
    # cell CLOSED before any admitted record is emitted, so the
    # variant_id-keyed grouping and the admitted register_cap can never
    # describe two different build variants for one cell
    # (repeatability_aggregator._variant_consistency_error).
    "label_variant_identity_inconsistent",
})


# ----------------------------------------------------------------------
# Per-taxonomy-family surface inventory (C-AC-C).
#
# Every typed-reason taxonomy family has a defined set of surfaces that
# must stay in lockstep when a typed reason is added or removed. This
# top-level constant declares the inventory; individual per-surface test
# classes below enforce the lockstep at each surface. Adding a new typed
# reason to a family means adding it to the family's expected_typed_reasons
# constant AND landing the matching surface code at every surface
# enumerated here in the same round.
#
# Three "kinds" exist:
#   - `op_class_value`: enum values within an op-class chain (e.g.
#     MemoryOpClass values GLOBAL/SHARED/...).
#   - `typed_reason`: typed reason strings emitted into sidecar JSON
#     (e.g. unsupported_memory_path, label_repeatability_failed).
#
# When this loop adds a new typed reason (e.g. nvbit_slowdown_budget_exceeded),
# the developer:
#   1. Identifies the target family (e.g. "trace_budget").
#   2. Adds the reason to the family's expected_typed_reasons constant.
#   3. Lands the reason at every surface declared for the family in the
#      same round; the per-surface test classes below will fail the round
#      until lockstep is achieved.
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TaxonomyFamilySurface:
    """One surface in a typed-reason taxonomy family.

    `surface_id` is the canonical name of the surface; `description`
    explains what code carries the surface (file paths, helper functions,
    constants); `enforced_by_test_classes` names the existing per-surface
    test classes that drive the lockstep check at this surface so a
    future maintainer can navigate the inventory back to the live
    asserter.
    """

    surface_id: str
    description: str
    enforced_by_test_classes: tuple[str, ...]


@dataclass(frozen=True)
class TaxonomyFamilyInventory:
    """Surface inventory for one typed-reason or op-class taxonomy
    family. The `expected_typed_reasons` field is a reference to the
    canonical set constant for the family; adding a reason to that
    constant flows to every per-surface test that resolves the constant
    at runtime."""

    family_id: str
    description: str
    kind: str  # "op_class_value" | "typed_reason"
    surfaces: tuple[TaxonomyFamilySurface, ...]
    expected_typed_reasons: frozenset[str]


# The op-class enum chain enumerates the eleven members of the
# `MemoryOpClass` enum. The eleven-element set lives here (rather than
# being computed from `import_nvbit_exclusion.VALID_OP_CLASSES` at
# import time) so a refactor of the consumer module does not silently
# weaken the lockstep check.
_OP_CLASS_ENUM_EXPECTED: frozenset[str] = frozenset({
    "GLOBAL",
    "SHARED",
    "LOCAL",
    "TEXTURE",
    "SURFACE",
    "ATOMIC",
    "TMA",
    "ASYNC_COPY",
    "CLUSTER_DSMEM",
    # Blackwell (sm_100) tensor-memory storage class via the
    # `tcgen05.{ld,st}` family (mnemonics `LDT`/`STT`). The
    # producer-side classifier routes both to
    # `memory_op_class_classifier::kTmem`; the proto enum carries
    # `MEMORY_OP_CLASS_TMEM = 10`; the consumer admit list and the
    # JSON-Schema enum both admit the `TMEM` label. Producer +
    # consumer widening land in the same source-tree commit per
    # BL-20260507-multi-surface-lockstep-changes.
    "TMEM",
    # `UNSPECIFIED` is the sentinel for STANDARD_MEM opcodes the
    # producer-side classifier could not match against any known
    # prefix. The producer-side typed exclusion routes those through
    # the sidecar so the validator does not reject them as
    # `memory_op_class_missing_on_memory_instruction`.
    "UNSPECIFIED",
})


_TAXONOMY_FAMILIES: dict[str, TaxonomyFamilyInventory] = {
    "memory_op_class_enum": TaxonomyFamilyInventory(
        family_id="memory_op_class_enum",
        description=(
            "Op-class enum chain: each `MemoryOpClass` member must be "
            "declared at every surface or the producer/consumer wire "
            "becomes asymmetric. Members live as the producer C++ "
            "`memory_op_class_classifier::Value` enum, the proto wire "
            "`MemoryOpClass` enum, the consumer admit list "
            "`import_nvbit_exclusion.VALID_OP_CLASSES`, the JSON-Schema "
            "`excluded_op_class.enum` list, and a C++<->proto numeric "
            "static_assert in `tracer_tool.cu`."
        ),
        kind="op_class_value",
        surfaces=(
            TaxonomyFamilySurface(
                surface_id="producer_cpp_enum",
                description=(
                    "tools/nvbit_tracer/tracer_tool/memory_op_class.h "
                    "`memory_op_class_classifier::Value` enum class."
                ),
                enforced_by_test_classes=("MemoryOpClassParityTest",),
            ),
            TaxonomyFamilySurface(
                surface_id="proto_wire_enum",
                description=(
                    "tools/nvbit_tracer/traces_enhanced/dynamic_trace/"
                    "instruction.proto `MemoryOpClass` enum."
                ),
                enforced_by_test_classes=(
                    "ProtoMemoryOpClassEnumParityTest",
                ),
            ),
            TaxonomyFamilySurface(
                surface_id="consumer_python_admit_list",
                description=(
                    "tools/nvbit_tracer/import_nvbit_exclusion.py "
                    "`VALID_OP_CLASSES` tuple."
                ),
                enforced_by_test_classes=(
                    "ProtoMemoryOpClassEnumParityTest",
                ),
            ),
            TaxonomyFamilySurface(
                surface_id="json_schema_enum",
                description=(
                    "workloads/simbench_bw/schemas/nvbit_*.schema.json "
                    "`excluded_op_class.enum` list."
                ),
                enforced_by_test_classes=("SchemaOpClassEnumParityTest",),
            ),
            TaxonomyFamilySurface(
                surface_id="cpp_proto_static_assert",
                description=(
                    "tools/nvbit_tracer/tracer_tool/tracer_tool.cu "
                    "`static_assert(static_cast<int>(<C++>) == "
                    "<proto-value>)` pairs."
                ),
                enforced_by_test_classes=(
                    "StaticAssertNumericMappingParityTest",
                ),
            ),
        ),
        expected_typed_reasons=_OP_CLASS_ENUM_EXPECTED,
    ),
    "nvbit_exclusion_typed_reason": TaxonomyFamilyInventory(
        family_id="nvbit_exclusion_typed_reason",
        description=(
            "NVBit-tracer exclusion sidecars: the producer aggregates "
            "every cannot-represent instruction into one of three (or "
            "more) typed reasons; the consumer admits a matching set "
            "and validates against per-typed-reason JSON Schemas."
        ),
        kind="typed_reason",
        surfaces=(
            TaxonomyFamilySurface(
                surface_id="producer_aggregate_use_sites",
                description=(
                    "tools/nvbit_tracer/tracer_tool/tracer_tool.cu "
                    "`nvbit_tracer::aggregate(...)` call sites; the "
                    "last argument is the typed_reason literal."
                ),
                enforced_by_test_classes=("ExclusionTaxonomyParityTest",),
            ),
            TaxonomyFamilySurface(
                surface_id="sidecar_writers",
                description=(
                    "tools/nvbit_tracer/tracer_tool/exclusion_aggregator.cc "
                    "`write_one_typed_sidecar(...)` call sites; first "
                    "argument is the filename slug, second is the "
                    "typed_reason literal."
                ),
                enforced_by_test_classes=("ExclusionTaxonomyParityTest",),
            ),
            TaxonomyFamilySurface(
                surface_id="consumer_admit_list",
                description=(
                    "tools/nvbit_tracer/import_nvbit_exclusion.py "
                    "`_SCHEMA_FILES` dict; keys are typed reasons "
                    "admitted by the importer."
                ),
                enforced_by_test_classes=(
                    "ExclusionTaxonomyParityTest",
                    "NvbitSchemaConstParityTest",
                ),
            ),
            TaxonomyFamilySurface(
                surface_id="json_schema_file",
                description=(
                    "workloads/simbench_bw/schemas/nvbit_<reason>.schema.json "
                    "with `typed_reason` const matching the reason."
                ),
                enforced_by_test_classes=(
                    "ExclusionTaxonomyParityTest",
                    "NvbitSchemaConstParityTest",
                    "SchemaIdParityTest",
                ),
            ),
        ),
        expected_typed_reasons=PRODUCER_TYPED_REASONS_EXPECTED,
    ),
    "trace_budget": TaxonomyFamilyInventory(
        family_id="trace_budget",
        description=(
            "Trace-budget typed reasons: producer constants in "
            "`tools/cloud/trace_budgets.py` declare `<UPPER_NAME>_SIDECAR` "
            "string slugs; the simbench writer emits sidecars by literal "
            "filename in `workloads/simbench_bw/run_suite.py`; the "
            "validator in `workloads/simbench_bw/validate.py` discovers "
            "the typed-reason set; per-typed-reason JSON Schemas live "
            "under `workloads/simbench_bw/schemas/`. C-AC-H plans to "
            "extend the canonical set with `nvbit_slowdown_budget_exceeded` "
            "and `nvbit_drop_rate_exceeded` (Track D / task9)."
        ),
        kind="typed_reason",
        surfaces=(
            TaxonomyFamilySurface(
                surface_id="producer_constants",
                description=(
                    "tools/cloud/trace_budgets.py `<UPPER_NAME>_SIDECAR: "
                    "str = \"<reason>.json\"` constants."
                ),
                enforced_by_test_classes=("TraceBudgetTypedReasonsParityTest",),
            ),
            TaxonomyFamilySurface(
                surface_id="simbench_writer_literal",
                description=(
                    "workloads/simbench_bw/run_suite.py f-string sidecar "
                    "writer literal mentions of `<reason>.json` carrying "
                    "the typed_reason field."
                ),
                enforced_by_test_classes=("TraceBudgetLiveLayerParityTest",),
            ),
            TaxonomyFamilySurface(
                surface_id="validator_discovery",
                description=(
                    "workloads/simbench_bw/validate.py validator's "
                    "typed-reason discovery list / dict."
                ),
                enforced_by_test_classes=("TraceBudgetLiveLayerParityTest",),
            ),
            TaxonomyFamilySurface(
                surface_id="json_schema_file",
                description=(
                    "workloads/simbench_bw/schemas/<reason>.schema.json "
                    "(no `nvbit_` prefix) with `typed_reason` const "
                    "matching the reason."
                ),
                enforced_by_test_classes=(
                    "TraceBudgetLiveLayerParityTest",
                    "TraceBudgetSchemaScannerCompatTest",
                ),
            ),
        ),
        expected_typed_reasons=TRACE_BUDGET_TYPED_REASONS_EXPECTED,
    ),
    "local_labels": TaxonomyFamilyInventory(
        family_id="local_labels",
        description=(
            "Local-label typed reasons emitted by "
            "`tools/local_labels/repeatability_aggregator.py` into "
            "`label_excluded.json` entries. Producer-only: there is no "
            "consumer admit list, no JSON Schema, and no separate "
            "validator surface yet — the aggregator is both producer "
            "and consumer for this family today."
        ),
        kind="typed_reason",
        surfaces=(
            TaxonomyFamilySurface(
                surface_id="producer_repeatability_aggregator",
                description=(
                    "tools/local_labels/repeatability_aggregator.py "
                    "typed_reason str literals emitted into "
                    "`label_excluded.json` entries."
                ),
                enforced_by_test_classes=("LocalLabelsTypedReasonsParityTest",),
            ),
        ),
        expected_typed_reasons=LOCAL_LABELS_TYPED_REASONS_EXPECTED,
    ),
    "sector_stream": TaxonomyFamilyInventory(
        family_id="sector_stream",
        description=(
            "Sector-stream typed reasons emitted by "
            "`tools/sector_stream/generator.py` (Loop 2 R5 / C-AC-F). "
            "Producer-side constant `SECTOR_STREAM_TYPED_REASONS` "
            "declares the four reasons; consumer-side schema "
            "enum `typed_reason.enum` lives at "
            "`workloads/simbench_bw/schemas/sector_stream.schema.json`. "
            "Two of the four reasons (`no_lanes_active`, "
            "`lane_address_length_invalid`) are also in the original-"
            "plan AC-19 enumeration; the other two "
            "(`data_width_invalid`, `sector_stream_class_unsupported`) "
            "are round-added per ROUND_ADDED_TYPED_REASONS."
        ),
        kind="typed_reason",
        surfaces=(
            TaxonomyFamilySurface(
                surface_id="producer_constants",
                description=(
                    "tools/sector_stream/generator.py "
                    "`SECTOR_STREAM_TYPED_REASONS: frozenset[str]` "
                    "constant. The generator returns one of these "
                    "values in the `typed_reason` field of any "
                    "rejection record."
                ),
                enforced_by_test_classes=("SectorStreamTypedReasonsParityTest",),
            ),
            TaxonomyFamilySurface(
                surface_id="json_schema_enum",
                description=(
                    "workloads/simbench_bw/schemas/sector_stream.schema.json "
                    "typed_reason.enum on the typed-exclusion branch "
                    "of the schema's oneOf."
                ),
                enforced_by_test_classes=("SectorStreamTypedReasonsParityTest",),
            ),
        ),
        expected_typed_reasons=SECTOR_STREAM_TYPED_REASONS_EXPECTED,
    ),
    "divergence_proxies": TaxonomyFamilyInventory(
        family_id="divergence_proxies",
        description=(
            "Divergence + MSHR-pressure proxy typed-unavailable reasons "
            "emitted by `tools/divergence_proxies/proxies.py` (Loop 2 "
            "R8 / C-AC-G). Producer-side constant "
            "`DIVERGENCE_PROXIES_TYPED_REASONS` declares the eight "
            "reasons; consumer-side schema enum `typed_reason.enum` "
            "lives on the typed-unavailable branch of "
            "`workloads/simbench_bw/schemas/divergence_proxies.schema.json` "
            "(navigated via `$defs/proxy_or_unavailable/oneOf`). One "
            "of the eight reasons "
            "(`outstanding_load_density_unavailable_no_phase_records`) "
            "is also in the original-plan AC-19 enumeration; the other "
            "seven are round-added per ROUND_ADDED_TYPED_REASONS."
        ),
        kind="typed_reason",
        surfaces=(
            TaxonomyFamilySurface(
                surface_id="producer_constants",
                description=(
                    "tools/divergence_proxies/proxies.py "
                    "`DIVERGENCE_PROXIES_TYPED_REASONS: frozenset[str]` "
                    "constant. The proxy module returns one of these "
                    "values in the `typed_reason` field of any "
                    "typed-unavailable proxy record."
                ),
                enforced_by_test_classes=(
                    "DivergenceProxiesTypedReasonsParityTest",
                ),
            ),
            TaxonomyFamilySurface(
                surface_id="json_schema_enum",
                description=(
                    "workloads/simbench_bw/schemas/"
                    "divergence_proxies.schema.json "
                    "typed_reason.enum on the typed-unavailable branch "
                    "of `$defs/proxy_or_unavailable/oneOf`."
                ),
                enforced_by_test_classes=(
                    "DivergenceProxiesTypedReasonsParityTest",
                ),
            ),
        ),
        expected_typed_reasons=DIVERGENCE_PROXIES_TYPED_REASONS_EXPECTED,
    ),
    "hrd_profile": TaxonomyFamilyInventory(
        family_id="hrd_profile",
        description=(
            "Conditional HRD profile typed reasons emitted by "
            "`tools/hrd_profiler/profiler.py` (Loop 2 R17 / task11 / "
            "AC-6). Producer-side constant `HRD_PROFILE_TYPED_REASONS` "
            "declares the 17 reasons; consumer-side schema enum "
            "`typed_reason.enum` lives at "
            "`workloads/simbench_bw/schemas/hrd_profile.schema.json` "
            "`$defs/typed_reason`. Three reasons "
            "(`cache_model_capacity_missing`, "
            "`cache_model_capacity_inconsistent`, "
            "`cache_model_underconfigured`) are pinned in the "
            "original-plan AC-19 enumeration; the other 14 are "
            "round-added per ROUND_ADDED_TYPED_REASONS."
        ),
        kind="typed_reason",
        surfaces=(
            TaxonomyFamilySurface(
                surface_id="producer_constants",
                description=(
                    "tools/hrd_profiler/profiler.py "
                    "`HRD_PROFILE_TYPED_REASONS: frozenset[str]` "
                    "constant. The profiler returns one of these "
                    "values in the `typed_reason` field of any "
                    "typed-unavailable bin or evidence record."
                ),
                enforced_by_test_classes=(
                    "HrdProfileTypedReasonsParityTest",
                ),
            ),
            TaxonomyFamilySurface(
                surface_id="json_schema_enum",
                description=(
                    "workloads/simbench_bw/schemas/hrd_profile.schema.json "
                    "$defs.typed_reason.enum (the shared enum reused "
                    "by both per-bin and per-evidence records)."
                ),
                enforced_by_test_classes=(
                    "HrdProfileTypedReasonsParityTest",
                ),
            ),
        ),
        expected_typed_reasons=HRD_PROFILE_TYPED_REASONS_EXPECTED,
    ),
    "sharing_locality": TaxonomyFamilyInventory(
        family_id="sharing_locality",
        description=(
            "GPU-Shalom sharing-locality typed reasons emitted by "
            "`tools/sharing_locality/classifier.py` (Loop 2 R22 / "
            "task12 / the sharing-locality clause). Producer-side "
            "constant `SHARING_LOCALITY_TYPED_REASONS` declares the "
            "7 reasons; consumer-side schema enum "
            "`typed_reason.enum` lives at "
            "`workloads/simbench_bw/schemas/sharing_locality.schema.json` "
            "`$defs/typed_reason`. All 7 reasons are round-added "
            "per ROUND_ADDED_TYPED_REASONS; none are in the "
            "original-plan AC-19 enumeration."
        ),
        kind="typed_reason",
        surfaces=(
            TaxonomyFamilySurface(
                surface_id="producer_constants",
                description=(
                    "tools/sharing_locality/classifier.py "
                    "`SHARING_LOCALITY_TYPED_REASONS: frozenset[str]` "
                    "constant. The classifier returns one of these "
                    "values in the `typed_reason` field of any "
                    "typed-unavailable record."
                ),
                enforced_by_test_classes=(
                    "SharingLocalityTypedReasonsParityTest",
                ),
            ),
            TaxonomyFamilySurface(
                surface_id="json_schema_enum",
                description=(
                    "workloads/simbench_bw/schemas/sharing_locality.schema.json "
                    "$defs.typed_reason.enum (the shared enum reused "
                    "by both grounded and projected typed-unavailable "
                    "records)."
                ),
                enforced_by_test_classes=(
                    "SharingLocalityTypedReasonsParityTest",
                ),
            ),
        ),
        expected_typed_reasons=SHARING_LOCALITY_TYPED_REASONS_EXPECTED,
    ),
    "order_projections": TaxonomyFamilyInventory(
        family_id="order_projections",
        description=(
            "Schedule-marginalized order-projections typed reasons "
            "emitted by `tools/order_projections/projector.py` "
            "(Loop 2 R25 / task13 / the order-projections "
            "clause). Producer-side constant "
            "`ORDER_PROJECTIONS_TYPED_REASONS` declares the 7 "
            "reasons; consumer-side schema enum lives at "
            "`workloads/simbench_bw/schemas/order_projections.schema.json` "
            "`$defs/typed_reason`. All 7 reasons are round-added."
        ),
        kind="typed_reason",
        surfaces=(
            TaxonomyFamilySurface(
                surface_id="producer_constants",
                description=(
                    "tools/order_projections/projector.py "
                    "`ORDER_PROJECTIONS_TYPED_REASONS: frozenset[str]` "
                    "constant. The projector returns one of these "
                    "values in the `typed_reason` field of any "
                    "typed-unavailable record."
                ),
                enforced_by_test_classes=(
                    "OrderProjectionsTypedReasonsParityTest",
                ),
            ),
            TaxonomyFamilySurface(
                surface_id="json_schema_enum",
                description=(
                    "workloads/simbench_bw/schemas/order_projections.schema.json "
                    "$defs.typed_reason.enum (the shared enum reused "
                    "by both per-projection and disagreement-variance "
                    "typed-unavailable records)."
                ),
                enforced_by_test_classes=(
                    "OrderProjectionsTypedReasonsParityTest",
                ),
            ),
        ),
        expected_typed_reasons=ORDER_PROJECTIONS_TYPED_REASONS_EXPECTED,
    ),
    "perturbation_run_ledger": TaxonomyFamilyInventory(
        family_id="perturbation_run_ledger",
        description=(
            "Perturbation-run-ledger typed reasons emitted by "
            "`tools/perturbation_harness/ledger.py` (Loop 2 R31 / "
            "task15 / the perturbation-harness clause). Producer-side "
            "constant `PERTURBATION_RUN_LEDGER_TYPED_REASONS` declares "
            "the 9 reasons; consumer-side schema enum lives at "
            "`workloads/simbench_bw/schemas/perturbation_run_ledger.schema.json` "
            "`$defs/typed_reason`. The plan-enumerated reason "
            "`modifier_optimized_away` is in AC19_PLAN_ENUMERATED_TYPED_REASONS; "
            "the 6 `arch_feature_unavailable_*` reasons are concrete "
            "instances of the original-plan wildcard family and are "
            "documented in ROUND_ADDED_TYPED_REASONS as wildcard-family "
            "instances; the remaining 2 (`correctness_equivalence_failed`, "
            "`perturbation_compile_failed`) are pure round-added."
        ),
        kind="typed_reason",
        surfaces=(
            TaxonomyFamilySurface(
                surface_id="producer_constants",
                description=(
                    "tools/perturbation_harness/ledger.py "
                    "`PERTURBATION_RUN_LEDGER_TYPED_REASONS: frozenset[str]` "
                    "constant. The perturbation harness returns one of "
                    "these values in the `typed_reason` field of any "
                    "typed-unavailable perturbation record."
                ),
                enforced_by_test_classes=(
                    "PerturbationRunLedgerTypedReasonsParityTest",
                ),
            ),
            TaxonomyFamilySurface(
                surface_id="json_schema_enum",
                description=(
                    "workloads/simbench_bw/schemas/perturbation_run_ledger.schema.json "
                    "$defs.typed_reason.enum (the shared enum reused "
                    "by all four typed-unavailable arms: "
                    "perturbation_compile_failed, modifier_optimized_away, "
                    "correctness_equivalence_failed, "
                    "arch_feature_unavailable_*)."
                ),
                enforced_by_test_classes=(
                    "PerturbationRunLedgerTypedReasonsParityTest",
                ),
            ),
        ),
        expected_typed_reasons=PERTURBATION_RUN_LEDGER_TYPED_REASONS_EXPECTED,
    ),
    "trace_convergence": TaxonomyFamilyInventory(
        family_id="trace_convergence",
        description=(
            "Trace-convergence typed reasons emitted by the "
            "adaptive-sampler clause (Loop 2 R37 / task17 / AC-11). "
            "Producer-side constant `TRACE_CONVERGENCE_TYPED_REASONS` "
            "in `tools/adaptive_sampler/trace_convergence_ledger.py` "
            "declares the 4 reasons. Consumer-side: each reason has "
            "a dedicated `oneOf` branch in "
            "`workloads/simbench_bw/schemas/trace_convergence_confidence.schema.json` "
            "with `properties.typed_reason.const`. Dispatch surfaces: "
            "the controller helper "
            "`_emit_trace_convergence_confidence_for_run_summary` in "
            "`workloads/simbench_bw/run_suite.py` carries the writer "
            "path; the validator helper "
            "`validate_trace_convergence_confidence` in "
            "`workloads/simbench_bw/validate.py` carries the discovery "
            "path. The plan-enumerated reason "
            "`trace_convergence_not_reached_budget_cap` is in "
            "AC19_PLAN_ENUMERATED_TYPED_REASONS; the other three "
            "are round-added."
        ),
        kind="typed_reason",
        surfaces=(
            TaxonomyFamilySurface(
                surface_id="producer_constants",
                description=(
                    "tools/adaptive_sampler/trace_convergence_ledger.py "
                    "`TRACE_CONVERGENCE_TYPED_REASONS: frozenset[str]` "
                    "constant. The producer's admission helpers in "
                    "`tools/adaptive_sampler/sampler.py` return one "
                    "of these values in the `typed_reason` field of "
                    "any typed-unavailable admission record."
                ),
                enforced_by_test_classes=(
                    "TraceConvergenceTypedReasonsParityTest",
                ),
            ),
            TaxonomyFamilySurface(
                surface_id="simbench_writer_helper",
                description=(
                    "workloads/simbench_bw/run_suite.py "
                    "`_emit_trace_convergence_confidence_for_run_summary` "
                    "helper writes the bundle to "
                    "`<trace_dir>/convergence_confidence/convergence_confidence.json` "
                    "and persists the `trace_convergence_typed_reason` "
                    "field in the run summary. The dispatch presence "
                    "check ensures the helper exists with the "
                    "expected name and references the bundle path "
                    "and run-summary field."
                ),
                enforced_by_test_classes=(
                    "TraceConvergenceTypedReasonsParityTest",
                ),
            ),
            TaxonomyFamilySurface(
                surface_id="validator_dispatch",
                description=(
                    "workloads/simbench_bw/validate.py "
                    "`validate_trace_convergence_confidence` "
                    "function dispatches the bundle to "
                    "`trace_convergence_confidence.schema.json`; the "
                    "discovery `rglob(\"convergence_confidence.json\")` "
                    "wires the validator into the dispatch loop. "
                    "Presence check that the function exists with "
                    "the expected name and references the schema."
                ),
                enforced_by_test_classes=(
                    "TraceConvergenceTypedReasonsParityTest",
                ),
            ),
            TaxonomyFamilySurface(
                surface_id="json_schema_oneof_consts",
                description=(
                    "workloads/simbench_bw/schemas/trace_convergence_confidence.schema.json "
                    "`admission.oneOf` branches: the four "
                    "typed-unavailable arms each declare "
                    "`properties.typed_reason.const` with one of "
                    "the four reasons. The lockstep scanner extracts "
                    "the const literals across all `oneOf` branches "
                    "and asserts equality with the producer constant."
                ),
                enforced_by_test_classes=(
                    "TraceConvergenceTypedReasonsParityTest",
                ),
            ),
        ),
        expected_typed_reasons=TRACE_CONVERGENCE_TYPED_REASONS_EXPECTED,
    ),
}


# Documented surface counts per family (the C-AC-C plan literals: op-class
# chain has 5 surfaces, trace-budget has 4, local-labels has 1, plus the
# nvbit-exclusion family the existing test classes have always covered
# at 4 surfaces). The numbers are pinned here so a future widening of
# `_TAXONOMY_FAMILIES` that drifts a family's surface count will fail
# the structural test below loudly.
_EXPECTED_SURFACE_COUNT: dict[str, int] = {
    "memory_op_class_enum": 5,
    "nvbit_exclusion_typed_reason": 4,
    "trace_budget": 4,
    "local_labels": 1,
    # Sector-stream family added in Loop 2 R5 / C-AC-F. Two
    # surfaces: producer constant set in
    # `tools/sector_stream/generator.py` and JSON schema enum
    # at
    # `workloads/simbench_bw/schemas/sector_stream.schema.json`.
    "sector_stream": 2,
    # Divergence-proxies family added in Loop 2 R8 / C-AC-G.
    # Two surfaces: producer constant set in
    # `tools/divergence_proxies/proxies.py` and JSON schema enum
    # on the typed-unavailable branch of
    # `workloads/simbench_bw/schemas/divergence_proxies.schema.json`.
    "divergence_proxies": 2,
    # HRD-profile family added in Loop 2 R17 / task11 / AC-6.
    # Two surfaces: producer constant set in
    # `tools/hrd_profiler/profiler.py` and JSON schema enum at
    # `$defs/typed_reason` of
    # `workloads/simbench_bw/schemas/hrd_profile.schema.json`.
    "hrd_profile": 2,
    # Sharing-locality family added in Loop 2 R22 / task12 / the
    # sharing-locality clause. Two surfaces: producer constant
    # set in `tools/sharing_locality/classifier.py` and JSON
    # schema enum at `$defs/typed_reason` of
    # `workloads/simbench_bw/schemas/sharing_locality.schema.json`.
    "sharing_locality": 2,
    # Order-projections family added in Loop 2 R25 / task13 / the
    # order-projections clause. Two surfaces: producer constant
    # set in `tools/order_projections/projector.py` and JSON
    # schema enum at `$defs/typed_reason` of
    # `workloads/simbench_bw/schemas/order_projections.schema.json`.
    "order_projections": 2,
    # Perturbation-run-ledger family added in Loop 2 R31 / task15 /
    # the perturbation-harness clause. Two surfaces: producer constant
    # set in `tools/perturbation_harness/ledger.py` and JSON schema
    # enum at `$defs/typed_reason` of
    # `workloads/simbench_bw/schemas/perturbation_run_ledger.schema.json`.
    "perturbation_run_ledger": 2,
    # Trace-convergence family added in Loop 2 R37 / task17 / AC-11.
    # Four surfaces: producer constant in
    # `tools/adaptive_sampler/trace_convergence_ledger.py`; controller
    # helper writer in `workloads/simbench_bw/run_suite.py`
    # (`_emit_trace_convergence_confidence_for_run_summary`); validator
    # dispatch in `workloads/simbench_bw/validate.py`
    # (`validate_trace_convergence_confidence`); and JSON schema oneOf
    # const literals in
    # `workloads/simbench_bw/schemas/trace_convergence_confidence.schema.json`.
    "trace_convergence": 4,
}


_AGGREGATE_TOKEN = "nvbit_tracer::aggregate"
_TYPED_REASON_DIRECT = re.compile(r'"(?P<reason>[a-z_]+)"')
_TYPED_REASON_WRAPPED = re.compile(
    r'std::string\s*\(\s*"(?P<reason>[a-z_]+)"\s*\)'
)


def _skip_string_literal(text: str, start: int) -> int:
    """Skip text[start:] starting at ``"``; return index after closing quote."""
    i = start + 1
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
        elif c == '"':
            return i + 1
        else:
            i += 1
    return i


def _skip_raw_string_literal(text: str, start: int) -> int | None:
    """If text[start:] opens a C++ raw string literal, skip past it.

    Recognizes the five raw-string opener forms: ``R"``, ``u8R"``,
    ``uR"``, ``UR"``, ``LR"``. The opener must be at a token boundary
    (preceded by a non-identifier character or the start of file);
    otherwise the prefix letters are part of an identifier and the
    function returns None so ordinary scanning continues.

    The delimiter between the opening ``"`` and ``(`` must follow C++
    raw-string rules: 0 to 16 characters, no spaces, ``(``, ``)``,
    ``\\``, or control characters. A malformed delimiter or absent
    ``(`` returns None.

    For a valid opener, scans forward for the closing sequence
    ``)`` + delim + ``"`` and returns the index immediately after it.
    An unterminated raw string returns ``len(text)`` so the rest of
    the source is consumed without aggregate-token matching (the
    safety property: an unterminated raw string cannot leak content
    as code).

    Returns None when ``text[start:]`` is not a raw-string opener.
    """
    n = len(text)
    if start >= n:
        return None

    if start > 0:
        prev = text[start - 1]
        if prev.isalnum() or prev == "_":
            return None

    if text.startswith('u8R"', start):
        prefix_len = 3
    elif text.startswith('uR"', start):
        prefix_len = 2
    elif text.startswith('UR"', start):
        prefix_len = 2
    elif text.startswith('LR"', start):
        prefix_len = 2
    elif text.startswith('R"', start):
        prefix_len = 1
    else:
        return None

    delim_start = start + prefix_len + 1
    paren_open = text.find("(", delim_start)
    if paren_open == -1 or paren_open - delim_start > 16:
        return None

    delim = text[delim_start:paren_open]
    for c in delim:
        if c == " " or c == ")" or c == "\\":
            return None
        if ord(c) < 32 or ord(c) == 127:
            return None

    closer = ")" + delim + '"'
    close_pos = text.find(closer, paren_open + 1)
    if close_pos == -1:
        return n
    return close_pos + len(closer)


def _skip_char_literal(text: str, start: int) -> int:
    """Skip text[start:] starting at ``'``; return index after closing quote."""
    i = start + 1
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
        elif c == "'":
            return i + 1
        else:
            i += 1
    return i


def _find_matching_close_paren(text: str, start: int) -> int | None:
    """Return index of the matching ``)`` for a paren already opened.

    ``start`` is the index immediately AFTER the opening ``(``. Returns
    None if no matching close exists. Skips string literals, char
    literals, and nested parens.
    """
    depth = 1
    i = start
    while i < len(text):
        c = text[i]
        if c == '"':
            i = _skip_string_literal(text, i)
        elif c == "'":
            i = _skip_char_literal(text, i)
        elif c == "(":
            depth += 1
            i += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
            i += 1
        else:
            i += 1
    return None


def _split_top_level_args(body: str) -> list[str]:
    """Split a comma-separated argument list on top-level commas.

    Respects nested parens / braces / brackets and string / char
    literals. Returns the list of arguments (stripped of leading /
    trailing whitespace).
    """
    args: list[str] = []
    depth = 0
    start = 0
    i = 0
    while i < len(body):
        c = body[i]
        if c == '"':
            i = _skip_string_literal(body, i)
        elif c == "'":
            i = _skip_char_literal(body, i)
        elif c in "([{":
            depth += 1
            i += 1
        elif c in ")]}":
            depth -= 1
            i += 1
        elif c == "," and depth == 0:
            args.append(body[start:i].strip())
            start = i + 1
            i += 1
        else:
            i += 1
    args.append(body[start:].strip())
    return args


def _parse_typed_reason_argument(arg: str) -> str | None:
    """Parse ``aggregate(...)``'s 4th argument; return reason or None.

    Accepts both implicit-conversion and explicit-wrapper forms:

    - ``"reason"`` — direct string literal; the producer signature's
      ``const std::string &typed_reason`` accepts this via implicit
      conversion (also the form used throughout the unit-test source).
    - ``std::string("reason")`` — explicit wrapper form (the form
      currently used in ``tracer_tool.cu``).

    Returns the matched ``reason`` string on success, or None if the
    argument is neither form (caller raises a diagnostic).
    """
    stripped = arg.strip()
    m = _TYPED_REASON_WRAPPED.fullmatch(stripped)
    if m is not None:
        return m.group("reason")
    m = _TYPED_REASON_DIRECT.fullmatch(stripped)
    if m is not None:
        return m.group("reason")
    return None


def _iter_aggregate_call_openings(text: str):
    """Yield ``(call_start, body_start)`` for each in-code aggregate call.

    Walks the source once, skipping ``//`` line comments,
    ``/* ... */`` block comments, ``"..."`` string literals, C++ raw
    string literals (``R"(...)"`` and prefixed variants), and
    ``'...'`` char literals before matching the
    ``nvbit_tracer::aggregate(`` token. This is what makes the
    parity guard count call STATEMENTS instead of any text that
    happens to mention the aggregate token (e.g. a documentation
    comment that names the function, or a stringified diagnostic
    message).

    ``call_start`` is the source offset of ``n`` in the
    ``nvbit_tracer::aggregate`` token; ``body_start`` is the
    offset immediately after the matching ``(``.
    """
    n = len(text)
    i = 0
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            newline = text.find("\n", i + 2)
            i = n if newline == -1 else newline + 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            close = text.find("*/", i + 2)
            i = n if close == -1 else close + 2
            continue
        if c in ("R", "u", "U", "L"):
            raw_end = _skip_raw_string_literal(text, i)
            if raw_end is not None:
                i = raw_end
                continue
        if c == '"':
            i = _skip_string_literal(text, i)
            continue
        if c == "'":
            i = _skip_char_literal(text, i)
            continue
        if text.startswith(_AGGREGATE_TOKEN, i):
            j = i + len(_AGGREGATE_TOKEN)
            while j < n and text[j] in " \t\n\r":
                j += 1
            if j < n and text[j] == "(":
                yield i, j + 1
                i = j + 1
                continue
        i += 1


def _scan_aggregate_use_sites_from_text(text: str, source_name: str) -> frozenset[str]:
    """Scan C++ source text for ``nvbit_tracer::aggregate(...)`` calls.

    Walks the source via ``_iter_aggregate_call_openings`` so the
    scanner counts in-code call statements only, ignoring aggregate
    text that appears inside comments or string / char literals.
    For each accepted call, splits the 4-argument list and extracts
    the typed reason from the final argument. Both ``"reason"`` and
    ``std::string("reason")`` forms are accepted because the producer
    signature is ``const std::string &typed_reason`` and either form
    compiles. The scanner fails closed: a wrong argument count or an
    unparseable final argument raises ``ValueError`` with a diagnostic
    naming ``source_name`` and the offending call body.
    """
    reasons: set[str] = set()
    for call_start, body_start in _iter_aggregate_call_openings(text):
        body_end = _find_matching_close_paren(text, body_start)
        if body_end is None:
            raise ValueError(
                f"{source_name}: unterminated nvbit_tracer::aggregate(...) "
                f"starting at character offset {call_start}"
            )
        body = text[body_start:body_end]
        args = _split_top_level_args(body)
        # Codex R56 code-review P1: accept the legacy 4-arg form
        # (function_unique_id, op_class, vpc, typed_reason) AND
        # the new 5-arg form (legacy 4 + optional path_resolved
        # bool). The typed-reason argument stays at args[3]
        # under both shapes.
        if len(args) not in (4, 5):
            raise ValueError(
                f"{source_name}: nvbit_tracer::aggregate(...) at character "
                f"offset {call_start} has {len(args)} arguments; "
                f"expected 4 or 5. "
                f"Body: {body!r}"
            )
        reason = _parse_typed_reason_argument(args[3])
        if reason is None:
            raise ValueError(
                f"{source_name}: nvbit_tracer::aggregate(...) at character "
                f"offset {call_start} has unparseable typed-reason "
                f"argument {args[3]!r}; expected \"<reason>\" or "
                f"std::string(\"<reason>\")"
            )
        reasons.add(reason)
    return frozenset(reasons)


def _scan_aggregate_use_sites() -> frozenset[str]:
    """Scan ``tracer_tool.cu`` for ``nvbit_tracer::aggregate(...)`` calls.

    Thin wrapper around ``_scan_aggregate_use_sites_from_text`` that
    reads the live producer source. Splitting the file-IO from the
    text-level parser keeps the parser testable against synthetic C++
    fixtures that do not need to live in the repository.
    """
    text = _TRACER_TOOL_CU.read_text(encoding="utf-8")
    return _scan_aggregate_use_sites_from_text(text, str(_TRACER_TOOL_CU))


_LABEL_REASON_VALUE_RE = re.compile(r"^[a-z_]+$")

_TOKENIZE_TRIVIA_TYPES = frozenset(
    {
        tokenize.NEWLINE,
        tokenize.NL,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.COMMENT,
        tokenize.ENCODING,
        tokenize.ENDMARKER,
    }
)


def _decode_python_string_token(token_string: str) -> str | None:
    """Return the unquoted value of a Python STRING token, or None
    when the token cannot be safely decoded.

    Uses ``ast.literal_eval`` so all standard Python string forms are
    handled (single/double quotes, triple-quoted, raw, byte; and
    escape sequences). f-strings (which contain expressions and are
    NOT valid literal expressions) raise ``SyntaxError`` /
    ``ValueError``; this function returns None in that case so the
    caller can fail-closed and skip the match.
    """
    try:
        value = ast.literal_eval(token_string)
    except (SyntaxError, ValueError):
        return None
    return value if isinstance(value, str) else None


def _scan_repeatability_aggregator_typed_reasons_from_text(
    text: str, source_name: str
) -> frozenset[str]:
    """Tokenize-based scan of Python source for ``"typed_reason":
    "<reason>"`` dict-literal entries that occur in true code
    position (not inside line comments, docstrings, or string
    literals).

    Uses ``tokenize.tokenize`` to obtain the token stream, filters
    out trivia tokens (NEWLINE, NL, INDENT, DEDENT, COMMENT,
    ENCODING, ENDMARKER), and walks consecutive non-trivia triples
    (a, b, c). Records ``c``'s decoded string value when ``a`` is a
    STRING token decoding to ``"typed_reason"``, ``b`` is the OP
    ``:``, and ``c`` is a STRING token whose decoded value is a
    non-empty lowercase + underscore identifier (the typed-reason
    slug shape).

    A line comment containing the pattern, a docstring containing
    the pattern, and a single-line string literal containing the
    pattern all surface as a single COMMENT or STRING token whose
    decoded value (after stripping the comment marker / dropping
    the wrapping quotes) is the entire pattern, NOT the three
    separate STRING/OP/STRING tokens that the dict-literal form
    produces. Such occurrences are correctly skipped.
    """
    try:
        token_stream = tokenize.tokenize(
            io.BytesIO(text.encode("utf-8")).readline
        )
        non_trivia = [
            tok
            for tok in token_stream
            if tok.type not in _TOKENIZE_TRIVIA_TYPES
        ]
    except tokenize.TokenError as exc:
        raise ValueError(
            f"Failed to tokenize {source_name}: {exc}"
        ) from exc

    reasons: set[str] = set()
    for i in range(len(non_trivia) - 2):
        a, b, c = non_trivia[i], non_trivia[i + 1], non_trivia[i + 2]
        if a.type != tokenize.STRING:
            continue
        if b.type != tokenize.OP or b.string != ":":
            continue
        if c.type != tokenize.STRING:
            continue
        key_value = _decode_python_string_token(a.string)
        if key_value != "typed_reason":
            continue
        reason_value = _decode_python_string_token(c.string)
        if reason_value is None:
            continue
        if not _LABEL_REASON_VALUE_RE.match(reason_value):
            continue
        reasons.add(reason_value)
    return frozenset(reasons)


def _scan_repeatability_aggregator_typed_reasons() -> frozenset[str]:
    """Scan ``tools/local_labels/repeatability_aggregator.py`` for typed
    reasons emitted into ``label_excluded.json`` entries.

    The aggregator is producer-only; there is no separate sidecar
    writer or consumer admit list. Each typed reason appears as a
    Python dict-literal entry of the shape
    ``"typed_reason": "<reason>"``. This helper extracts the right-
    hand-side string for each such occurrence in true code position
    via ``_scan_repeatability_aggregator_typed_reasons_from_text``,
    which uses Python tokenize to skip line comments, docstrings,
    and string-literal contexts.
    """
    text = _REPEATABILITY_AGGREGATOR_PY.read_text(encoding="utf-8")
    return _scan_repeatability_aggregator_typed_reasons_from_text(
        text, str(_REPEATABILITY_AGGREGATOR_PY)
    )


_TRACE_BUDGET_SIDECAR_CONSTANT_NAME_RE = re.compile(r"^[A-Z_]+_SIDECAR$")


def _scan_trace_budget_typed_reasons_from_text(
    text: str, source_name: str
) -> frozenset[str]:
    """Tokenize-aware scan of Python source for ``<UPPER_NAME>_SIDECAR:
    str = "<reason>.json"`` constant declarations at code position.

    Token pattern (rightmost preceding tokens, in order):
    ``NAME(<X>_SIDECAR) OP(:) NAME(str) OP(=) STRING("<reason>.json")``.

    The previous regex was anchored to ``^[A-Z_]+_SIDECAR`` with
    ``re.MULTILINE`` and was naturally robust against `#`-prefixed line
    comments, but a docstring or multi-line string at column 0 with
    the same constant pattern would have false-matched. R27 closes
    that latent gap by using ``_iter_python_string_token_contexts``
    (R25), so any non-code position is correctly skipped.
    """
    reasons: set[str] = set()
    for tok, preceding, _ in _iter_python_string_token_contexts(
        text, source_name
    ):
        decoded = _decode_python_string_token(tok.string)
        if decoded is None:
            continue
        match = _TRACE_BUDGET_JSON_SLUG_RE.match(decoded)
        if match is None:
            continue
        if len(preceding) < 4:
            continue
        a, b, c, d = preceding[-4:]
        if a.type != tokenize.NAME:
            continue
        if not _TRACE_BUDGET_SIDECAR_CONSTANT_NAME_RE.match(a.string):
            continue
        if b.type != tokenize.OP or b.string != ":":
            continue
        if c.type != tokenize.NAME or c.string != "str":
            continue
        if d.type != tokenize.OP or d.string != "=":
            continue
        reasons.add(match.group(1))
    return frozenset(reasons)


def _scan_trace_budget_typed_reasons() -> frozenset[str]:
    """Scan ``tools/cloud/trace_budgets.py`` for sidecar-filename
    constants of the form ``<UPPER_NAME>_SIDECAR: str = "<reason>.json"``.

    Each such constant defines a typed reason emitted by the trace-
    budget surface. The producer use sites (in
    ``workloads/simbench_bw/run_suite.py``) write the sidecar under the
    constant's filename; the verifier
    (``workloads/simbench_bw/validate.py``) validates against
    ``<reason>.schema.json``.

    Uses Python tokenize to skip line comments, docstrings, and
    string-literal contexts before pattern-matching.
    """
    text = _TRACE_BUDGETS_PY.read_text(encoding="utf-8")
    return _scan_trace_budget_typed_reasons_from_text(
        text, str(_TRACE_BUDGETS_PY)
    )


_AC19_PLAN_ENUMERATION_ANCHOR = (
    "The exclusion taxonomy includes (at minimum):"
)
_PLAN_BACKTICK_REASON_RE = re.compile(r"`([a-z_][a-z0-9_]*\*?)`")


def _scan_plan_ac19_enumerated_typed_reasons_from_text(
    text: str,
) -> frozenset[str]:
    """Scan plan markdown text for the AC-19 typed-reason enumeration.

    The plan's AC-19 positive-tests section contains the source-of-
    truth enumeration on a single line beginning with the anchor
    ``The exclusion taxonomy includes (at minimum):``. Each typed
    reason is wrapped in backticks. The wildcard entry
    ``arch_feature_unavailable_*`` is intentionally excluded from
    the test code's ``AC19_PLAN_ENUMERATED_TYPED_REASONS`` set
    because it is a family pattern, not a single literal.

    Returns the frozenset of concrete (non-wildcard) typed reasons
    extracted from the anchor line. Wildcards (entries ending in
    ``*``) are filtered out.

    Raises ``ValueError`` if the anchor is not found, so a future
    plan revision that renames or removes the anchor fails the
    parity test loudly rather than silently returning an empty set.
    """
    for line in text.splitlines():
        if _AC19_PLAN_ENUMERATION_ANCHOR in line:
            matches = _PLAN_BACKTICK_REASON_RE.findall(line)
            return frozenset(m for m in matches if not m.endswith("*"))
    raise ValueError(
        f"AC-19 enumeration anchor "
        f"{_AC19_PLAN_ENUMERATION_ANCHOR!r} not found in plan text"
    )


def _scan_plan_ac19_enumerated_typed_reasons() -> frozenset[str]:
    """Scan ``plans/nonroot_gpu_cache_observability_plan.md`` for the
    AC-19 typed-reason enumeration.

    The plan is the source-of-truth document for the AC-19 typed-
    reason taxonomy. The test code's
    ``AC19_PLAN_ENUMERATED_TYPED_REASONS`` constant should mirror the
    plan exactly (excluding wildcard entries). This scanner extracts
    the plan's enumeration so the parity test can compare them.
    """
    text = _PLAN_PATH.read_text(encoding="utf-8")
    return _scan_plan_ac19_enumerated_typed_reasons_from_text(text)


def _scrub_cpp_non_code_regions(text: str) -> str:
    """Walk a C++ source once and blank every non-code region with
    spaces of equal length, preserving overall text length so later
    regex offsets are unchanged.

    Recognized non-code regions, checked in this order at each
    position:

    1. Raw string literals ``R"(...)``", ``u8R"(...)"``,
       ``uR"(...)"``, ``UR"(...)"``, ``LR"(...)"``. Checked
       FIRST because the prefix characters (``R``, ``u``, ``U``,
       ``L``) overlap with ordinary identifiers. The token-boundary
       guard inside ``_skip_raw_string_literal`` ensures an
       identifier ending in one of those letters is not misread
       as a raw-string opener.
    2. Line comments ``// ... \\n``.
    3. Block comments ``/* ... */`` (unterminated → blanked to
       end-of-text).
    4. Ordinary string literals ``"..."`` (with backslash escapes).
    5. Char literals ``'...'`` (with backslash escapes).

    R35 used a two-pass chain (a comment stripper followed by a
    string-literal stripper), but that order is broken: the comment
    stripper was not raw-string aware, so a raw string containing
    an embedded ``"`` would expose later ``//`` text to the
    comment stripper before the raw-string skipper ever ran (the
    embedded ``"`` closes the comment-stripper's ordinary-string
    fast-skip, leaving the cursor mid-raw-string at the ``//``).
    R36 unifies both passes into a single walker with the correct
    raw-string-first ordering. R37 migrated the R29/R33 enum
    scanners to this unified walker, and R38 removed the legacy
    comment-only stripper since it had no remaining callers.

    Reuses the ``_skip_string_literal``, ``_skip_char_literal``,
    and ``_skip_raw_string_literal`` primitives without
    introducing a second raw-string parser. Matches the walking
    order used by ``_iter_aggregate_call_openings`` (R17/R18)
    for the aggregate-use-site scanner.
    """
    result = list(text)
    n = len(text)
    i = 0
    while i < n:
        c = text[i]
        if c in ("R", "u", "U", "L"):
            raw_end = _skip_raw_string_literal(text, i)
            if raw_end is not None:
                for k in range(i, raw_end):
                    result[k] = " "
                i = raw_end
                continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            if j == -1:
                j = n
            for k in range(i, j):
                result[k] = " "
            i = j
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            end = n if j == -1 else j + 2
            for k in range(i, end):
                result[k] = " "
            i = end
        elif c == '"':
            end = _skip_string_literal(text, i)
            for k in range(i, end):
                result[k] = " "
            i = end
        elif c == "'":
            end = _skip_char_literal(text, i)
            for k in range(i, end):
                result[k] = " "
            i = end
        else:
            i += 1
    return "".join(result)


def _k_camel_case_to_upper_snake(name: str) -> str:
    """Convert a C++ ``kCamelCase`` enum entry name to
    ``UPPER_SNAKE``.

    Examples:
      - ``kGlobal`` → ``GLOBAL``
      - ``kAsyncCopy`` → ``ASYNC_COPY``
      - ``kClusterDsmem`` → ``CLUSTER_DSMEM``
      - ``kUnspecified`` → ``UNSPECIFIED``

    Raises ``ValueError`` if the input does not start with ``k``
    followed by an uppercase letter.
    """
    if len(name) < 2 or name[0] != "k" or not name[1].isupper():
        raise ValueError(
            f"Expected kCamelCase identifier (k + UpperCase), got "
            f"{name!r}"
        )
    body = name[1:]
    parts: list[str] = []
    for i, ch in enumerate(body):
        if i > 0 and ch.isupper():
            parts.append("_")
        parts.append(ch.upper())
    return "".join(parts)


_MEMORY_OP_CLASS_ENUM_BLOCK = re.compile(
    r"enum\s+Value\s*:\s*int\s*\{(?P<body>[^}]*)\}\s*;"
)
_MEMORY_OP_CLASS_ENUM_ENTRY = re.compile(r"\b(k[A-Z][A-Za-z0-9]*)\b")


def _scan_producer_memory_op_class_enum_from_text(
    text: str,
) -> frozenset[str]:
    """Scan a C++ source containing
    ``enum Value : int { kCamelCase = N, ... };`` for the entry names,
    transformed to ``UPPER_SNAKE``.

    Runs the unified C++ lexical scrubber
    ``_scrub_cpp_non_code_regions`` first so comments, ordinary
    strings, char literals, and raw strings are all blanked out
    before the regex is applied (R37 hardening; closes a
    string-literal false positive where a fake ``enum Value : int``
    block inside a ``"..."`` string was previously matched). Raises
    ``ValueError`` if the enum Value block is not found, so a
    renamed/deleted enum surfaces loudly rather than silently
    returning an empty set.
    """
    scrubbed = _scrub_cpp_non_code_regions(text)
    block_match = _MEMORY_OP_CLASS_ENUM_BLOCK.search(scrubbed)
    if block_match is None:
        raise ValueError(
            "Producer-side `enum Value : int { ... };` block not "
            "found in memory_op_class.h text"
        )
    body = block_match.group("body")
    entries = _MEMORY_OP_CLASS_ENUM_ENTRY.findall(body)
    return frozenset(_k_camel_case_to_upper_snake(e) for e in entries)


def _scan_producer_memory_op_class_enum() -> frozenset[str]:
    """Scan ``tools/nvbit_tracer/tracer_tool/memory_op_class.h`` for
    the producer's op-class enum entries.

    Returns the frozenset of UPPER_SNAKE names that should match the
    consumer-side ``import_nvbit_exclusion.VALID_OP_CLASSES`` admit
    list.
    """
    text = _MEMORY_OP_CLASS_H.read_text(encoding="utf-8")
    return _scan_producer_memory_op_class_enum_from_text(text)


_PROTO_MEMORY_OP_CLASS_ENUM_BLOCK = re.compile(
    r"enum\s+MemoryOpClass\s*\{(?P<body>[^}]*)\}"
)
_PROTO_MEMORY_OP_CLASS_ENUM_ENTRY = re.compile(
    r"\bMEMORY_OP_CLASS_(?P<name>[A-Z][A-Z0-9_]*)\s*=\s*\d+\s*;"
)


def _extract_proto_memory_op_class_enum_from_text(
    text: str,
) -> frozenset[str]:
    """Scan a proto3 source containing
    ``enum MemoryOpClass { MEMORY_OP_CLASS_<NAME> = N; ... }`` for
    the entry names with the ``MEMORY_OP_CLASS_`` prefix stripped.

    Runs the unified C++/proto lexical scrubber
    ``_scrub_cpp_non_code_regions`` first so comments, ordinary
    strings, char literals, and raw strings are all blanked out
    before the regex is applied (R37 hardening; closes a
    string-literal false positive where a fake ``enum
    MemoryOpClass`` block inside an ``option foo = "..."`` string
    was previously matched). Proto3 uses the same ``//`` and
    ``/* */`` comment syntax as C++ and string-literal forms
    compatible with the C++ scrubber. Raises ``ValueError``
    when the ``MemoryOpClass`` enum block is not found, so a
    renamed/deleted enum surfaces loudly rather than silently
    returning an empty set.
    """
    scrubbed = _scrub_cpp_non_code_regions(text)
    block_match = _PROTO_MEMORY_OP_CLASS_ENUM_BLOCK.search(scrubbed)
    if block_match is None:
        raise ValueError(
            "Proto-side `enum MemoryOpClass { ... }` block not "
            "found in instruction.proto text"
        )
    body = block_match.group("body")
    entries = _PROTO_MEMORY_OP_CLASS_ENUM_ENTRY.findall(body)
    return frozenset(entries)


def _scan_proto_memory_op_class_enum() -> frozenset[str]:
    """Scan
    ``tools/nvbit_tracer/traces_enhanced/dynamic_trace/instruction.proto``
    for the proto-side ``MemoryOpClass`` enum entries.

    Returns the frozenset of UPPER_SNAKE names (with the
    ``MEMORY_OP_CLASS_`` prefix stripped) that should match the
    consumer-side ``import_nvbit_exclusion.VALID_OP_CLASSES`` admit
    list and the producer-side C++ enum scanned by
    ``_scan_producer_memory_op_class_enum``.
    """
    text = _PROTO_INSTRUCTION.read_text(encoding="utf-8")
    return _extract_proto_memory_op_class_enum_from_text(text)


_STATIC_ASSERT_PAIR_RE = re.compile(
    r"static_assert\s*\(\s*"
    r"static_cast<int>\s*\(\s*memory_op_class_classifier::"
    r"(?P<cpp>k[A-Za-z0-9]+)\s*\)\s*==\s*"
    r"static_cast<int>\s*\(\s*dynamic_trace::"
    r"MEMORY_OP_CLASS_(?P<proto>[A-Z][A-Z0-9_]*)\s*\)\s*,",
    re.DOTALL,
)


def _extract_static_assert_pairs_from_text(
    text: str,
) -> frozenset[tuple[str, str]]:
    """Scan a C++ source containing
    ``static_assert(static_cast<int>(memory_op_class_classifier::kX) == static_cast<int>(dynamic_trace::MEMORY_OP_CLASS_Y), ...);``
    blocks and extract the (C++ name, proto name) pairs.

    The C++ name retains its ``k`` prefix (e.g. ``kAtomic``); the
    proto name has the ``MEMORY_OP_CLASS_`` prefix stripped (e.g.
    ``ATOMIC``). Runs the unified C++ lexical scrubber
    ``_scrub_cpp_non_code_regions`` first so comments, ordinary
    strings, char literals, and raw strings are all blanked out
    in a single pass with raw-string detection ordered BEFORE
    comment detection (R36 fix for Codex's R35 review counterexample
    where a raw string with embedded ``"`` followed by ``//``
    caused the prior two-pass chain to strip a real
    static_assert).

    Returns an empty frozenset if no pairs are found. The live test
    treats an empty set as drift from the current 11-pair state
    (10 named classes plus UNSPECIFIED, after the Blackwell TMEM
    widening) rather than a hard failure.
    """
    scrubbed = _scrub_cpp_non_code_regions(text)
    return frozenset(
        (m.group("cpp"), m.group("proto"))
        for m in _STATIC_ASSERT_PAIR_RE.finditer(scrubbed)
    )


def _scan_static_assert_pairs() -> frozenset[tuple[str, str]]:
    """Scan ``tools/nvbit_tracer/tracer_tool/tracer_tool.cu`` for the
    `static_assert` block that pins
    ``memory_op_class_classifier::Value`` numeric values 1:1 with
    ``dynamic_trace::MemoryOpClass`` numeric values.

    Returns the frozenset of (C++ name, proto name) pairs, where the
    C++ name retains its ``k`` prefix and the proto name has the
    ``MEMORY_OP_CLASS_`` prefix stripped. The number and identity of
    pairs should equal the producer C++ enum (R29) and the proto
    enum (R33) under the ``kCamelCase`` ↔ ``UPPER_SNAKE``
    transformation.
    """
    text = _TRACER_TOOL_CU.read_text(encoding="utf-8")
    return _extract_static_assert_pairs_from_text(text)


_TRACE_BUDGET_JSON_SLUG_RE = re.compile(r"^([a-z_]+)\.json$")
_TRACE_BUDGET_SCHEMA_SLUG_RE = re.compile(r"^([a-z_]+)\.schema\.json$")

_FSTRING_START_TYPE = getattr(tokenize, "FSTRING_START", None)
_FSTRING_MIDDLE_TYPE = getattr(tokenize, "FSTRING_MIDDLE", None)

_TRACE_SIDECAR_WHOLE_FSTRING_RE = re.compile(
    r'^[fF][rR]?[\'"]{1,3}trace_sidecar:'
)


def _following_tokens_match_trace_sidecar_fstring(following) -> bool:
    """Return True iff the next f-string in the post-comma token sequence
    is an f-string whose body starts with ``trace_sidecar:``.

    Handles BOTH Python tokenization shapes:

    - **Python 3.12+**: f-strings are split into ``FSTRING_START`` (e.g.
      ``f"`` or ``f'``), ``FSTRING_MIDDLE`` (the static text up to the
      first expression hole), one or more expression tokens, and
      ``FSTRING_END``. We accept when ``following[0]`` is FSTRING_START
      with a recognized prefix and ``following[1]`` is FSTRING_MIDDLE
      whose text starts with ``trace_sidecar:``.

    - **Python 3.11**: f-strings tokenize as a single STRING token whose
      ``.string`` retains the f-prefix and quotes (e.g.
      ``'f"trace_sidecar:row {repr(slug)}"'``). We accept when
      ``following[0]`` is STRING with text matching
      ``[fF][rR]?['"]{1,3}trace_sidecar:`` (case-insensitive f-prefix,
      optional raw prefix, single or triple quotes).

    The Python 3.12+ path uses ``getattr(tokenize, "FSTRING_START",
    None)`` and ``getattr(tokenize, "FSTRING_MIDDLE", None)`` so the
    helper does not raise ``AttributeError`` on Python 3.11.
    """
    if not following:
        return False
    head = following[0]
    if (
        _FSTRING_START_TYPE is not None
        and _FSTRING_MIDDLE_TYPE is not None
        and head.type == _FSTRING_START_TYPE
    ):
        if head.string not in ('f"', "f'", 'F"', "F'"):
            return False
        if len(following) < 2:
            return False
        body = following[1]
        if body.type != _FSTRING_MIDDLE_TYPE:
            return False
        return body.string.startswith("trace_sidecar:")
    if head.type == tokenize.STRING:
        return bool(_TRACE_SIDECAR_WHOLE_FSTRING_RE.match(head.string))
    return False


def _iter_python_string_token_contexts(
    text: str, source_name: str, look_around: int = 10
):
    """Yield ``(string_token, preceding, following)`` for each STRING
    token at code position in a Python source.

    ``preceding`` and ``following`` are lists of up to ``look_around``
    adjacent non-trivia tokens. Trivia (NEWLINE, NL, INDENT, DEDENT,
    COMMENT, ENCODING, ENDMARKER) is filtered. STRING tokens that
    appear inside line comments, docstrings, or other string literals
    are NOT yielded — Python tokenize collapses those into a single
    surrounding STRING/COMMENT token whose value is the entire span,
    not a separate STRING in the filtered code-position stream.

    Raises ``ValueError`` (wrapping ``tokenize.TokenError``) if the
    source cannot be tokenized.
    """
    try:
        tokens = list(
            tokenize.tokenize(io.BytesIO(text.encode("utf-8")).readline)
        )
    except tokenize.TokenError as exc:
        raise ValueError(
            f"Failed to tokenize {source_name}: {exc}"
        ) from exc
    non_trivia = [
        tok for tok in tokens if tok.type not in _TOKENIZE_TRIVIA_TYPES
    ]
    for i, tok in enumerate(non_trivia):
        if tok.type != tokenize.STRING:
            continue
        preceding = non_trivia[max(0, i - look_around): i]
        following = non_trivia[i + 1: i + 1 + look_around]
        yield tok, preceding, following


def _scan_trace_budget_run_suite_typed_reasons_from_text(
    text: str, source_name: str
) -> frozenset[str]:
    """Tokenize-aware scan of Python source for ``sidecar = <var> /
    "<reason>.json"`` writer literals.

    Token pattern (rightmost preceding tokens, in order):
    ``NAME(sidecar) OP(=) NAME(any) OP(/) STRING("<reason>.json")``.
    """
    reasons: set[str] = set()
    for tok, preceding, _ in _iter_python_string_token_contexts(
        text, source_name
    ):
        decoded = _decode_python_string_token(tok.string)
        if decoded is None:
            continue
        match = _TRACE_BUDGET_JSON_SLUG_RE.match(decoded)
        if match is None:
            continue
        if len(preceding) < 4:
            continue
        a, b, c, d = preceding[-4:]
        if a.type != tokenize.NAME or a.string != "sidecar":
            continue
        if b.type != tokenize.OP or b.string != "=":
            continue
        if c.type != tokenize.NAME:
            continue
        if d.type != tokenize.OP or d.string != "/":
            continue
        reasons.add(match.group(1))
    return frozenset(reasons)


def _scan_trace_budget_run_suite_typed_reasons() -> frozenset[str]:
    """Scan ``workloads/simbench_bw/run_suite.py`` for the live trace-
    budget producer-use-site literals of the form
    ``sidecar = <dir> / "<reason>.json"``.

    Producer use sites in run_suite.py write the trace-budget sidecar
    by hard-coded filename literal; the parity guard catches drift if
    that literal disagrees with the constant in ``trace_budgets.py``.

    Uses Python tokenize to skip line comments, docstrings, and
    string-literal contexts before pattern-matching.
    """
    text = _RUN_SUITE_PY.read_text(encoding="utf-8")
    return _scan_trace_budget_run_suite_typed_reasons_from_text(
        text, str(_RUN_SUITE_PY)
    )


def _scan_trace_budget_validator_discovery_reasons_from_text(
    text: str, source_name: str
) -> frozenset[str]:
    """Tokenize-aware scan of Python source for ``.rglob("<reason>.
    json"):\\n    errors.extend(validate_trace_sidecar(...))`` patterns.

    Token pattern:
    - preceding (rightmost): ``OP(.) NAME(rglob) OP(()``
    - STRING: ``"<reason>.json"``
    - following: ``OP()) OP(:) NAME(errors) OP(.) NAME(extend) OP(()
      NAME(validate_trace_sidecar)``

    The ``validate_trace_sidecar`` function name disambiguates the
    trace-budget discovery from rglob patterns consumed by other
    validators.
    """
    reasons: set[str] = set()
    for tok, preceding, following in _iter_python_string_token_contexts(
        text, source_name
    ):
        decoded = _decode_python_string_token(tok.string)
        if decoded is None:
            continue
        match = _TRACE_BUDGET_JSON_SLUG_RE.match(decoded)
        if match is None:
            continue
        if len(preceding) < 3:
            continue
        a, b, c = preceding[-3:]
        if a.type != tokenize.OP or a.string != ".":
            continue
        if b.type != tokenize.NAME or b.string != "rglob":
            continue
        if c.type != tokenize.OP or c.string != "(":
            continue
        if len(following) < 7:
            continue
        e, f, g, h, i, j, k = following[:7]
        if e.type != tokenize.OP or e.string != ")":
            continue
        if f.type != tokenize.OP or f.string != ":":
            continue
        if g.type != tokenize.NAME or g.string != "errors":
            continue
        if h.type != tokenize.OP or h.string != ".":
            continue
        if i.type != tokenize.NAME or i.string != "extend":
            continue
        if j.type != tokenize.OP or j.string != "(":
            continue
        if k.type != tokenize.NAME or k.string != "validate_trace_sidecar":
            continue
        reasons.add(match.group(1))
    return frozenset(reasons)


def _scan_trace_budget_validator_discovery_reasons() -> frozenset[str]:
    """Scan ``workloads/simbench_bw/validate.py`` for trace-budget
    discovery literals: ``rglob`` patterns whose match is consumed by
    ``validate_trace_sidecar``.

    Uses Python tokenize to skip non-code regions before pattern
    matching.
    """
    text = _VALIDATE_PY.read_text(encoding="utf-8")
    return _scan_trace_budget_validator_discovery_reasons_from_text(
        text, str(_VALIDATE_PY)
    )


def _scan_trace_budget_validator_schema_reasons_from_text(
    text: str, source_name: str
) -> frozenset[str]:
    """Tokenize-aware scan of Python source for ``_validate_file(...,
    SCHEMAS_DIR / "<reason>.schema.json", f"trace_sidecar:..."``
    patterns.

    Token pattern:
    - preceding (rightmost): ``NAME(SCHEMAS_DIR) OP(/)``
    - STRING: ``"<reason>.schema.json"``
    - following: ``OP(,) FSTRING_START(f") FSTRING_MIDDLE(starts with
      'trace_sidecar:')``

    The ``f"trace_sidecar:`` diagnostic prefix is unique to the
    trace-budget validator and disambiguates from many other
    ``SCHEMAS_DIR /`` references in validate.py. In Python 3.12+
    f-strings tokenize as FSTRING_START + FSTRING_MIDDLE + ... +
    FSTRING_END.
    """
    reasons: set[str] = set()
    for tok, preceding, following in _iter_python_string_token_contexts(
        text, source_name
    ):
        decoded = _decode_python_string_token(tok.string)
        if decoded is None:
            continue
        match = _TRACE_BUDGET_SCHEMA_SLUG_RE.match(decoded)
        if match is None:
            continue
        if len(preceding) < 2:
            continue
        a, b = preceding[-2:]
        if a.type != tokenize.NAME or a.string != "SCHEMAS_DIR":
            continue
        if b.type != tokenize.OP or b.string != "/":
            continue
        if len(following) < 1:
            continue
        if following[0].type != tokenize.OP or following[0].string != ",":
            continue
        if not _following_tokens_match_trace_sidecar_fstring(following[1:]):
            continue
        reasons.add(match.group(1))
    return frozenset(reasons)


def _scan_trace_budget_validator_schema_reasons() -> frozenset[str]:
    """Scan ``workloads/simbench_bw/validate.py`` for trace-budget
    schema literals inside ``validate_trace_sidecar``.

    Uses Python tokenize to skip non-code regions before pattern
    matching.
    """
    text = _VALIDATE_PY.read_text(encoding="utf-8")
    return _scan_trace_budget_validator_schema_reasons_from_text(
        text, str(_VALIDATE_PY)
    )


def _four_way_parity_diff(
    constants: frozenset[str],
    writer: frozenset[str],
    discovery: frozenset[str],
    schema: frozenset[str],
) -> dict[str, frozenset[str]]:
    """Compute drift diagnostics for the trace-budget 4-way parity.

    Returns a dict naming reasons exclusive to each layer (present in
    that layer but not in any of the other three) plus the set
    present in all four layers.

    A passing parity check has all four ``*_only`` keys empty; the
    ``all_layers`` key equals the expected typed-reason set.
    """
    return {
        "constants_only": constants - writer - discovery - schema,
        "writer_only": writer - constants - discovery - schema,
        "discovery_only": discovery - constants - writer - schema,
        "schema_only": schema - constants - writer - discovery,
        "all_layers": constants & writer & discovery & schema,
    }


def _extract_schema_typed_reason_const(schema_text: str) -> str | None:
    """Drill into a JSON Schema document and return ``properties.
    typed_reason.const``.

    Returns ``None`` if the schema does not pin ``typed_reason`` with
    a ``const`` value (e.g. trace-budget schemas have no
    ``typed_reason`` field at all). Raises ``ValueError`` (via
    ``json.JSONDecodeError``) on malformed JSON.
    """
    data = json.loads(schema_text)
    if not isinstance(data, dict):
        return None
    properties = data.get("properties")
    if not isinstance(properties, dict):
        return None
    typed_reason_property = properties.get("typed_reason")
    if not isinstance(typed_reason_property, dict):
        return None
    const_value = typed_reason_property.get("const")
    return const_value if isinstance(const_value, str) else None


def _load_schema_typed_reason_const(schema_path: Path) -> str | None:
    """Read a JSON Schema file from disk and return its
    ``properties.typed_reason.const`` value (or None if absent).

    The const value is the JSON-Schema-enforced pin on the
    ``typed_reason`` field embedded in each typed-reason sidecar
    payload. The producer writes the typed_reason literal that
    matches the schema filename slug; the validator (importer)
    rejects sidecars whose typed_reason value does not equal the
    schema's const. If the const drifts from the filename slug,
    the producer/verifier silently disagree.
    """
    return _extract_schema_typed_reason_const(
        schema_path.read_text(encoding="utf-8")
    )


def _extract_schema_id_basename(schema_text: str) -> str | None:
    """Drill into a JSON Schema document and return the last path
    segment of ``$id``.

    Each AC-19 typed-reason schema's ``$id`` is path-prefixed with the
    schema filename, e.g.
    ``"$id": "simbench_bw/schemas/nvbit_unsupported_memory_path.schema.json"``.
    Splitting on ``/`` and taking the last segment yields the same
    string as the file's basename on disk; a drift between the two
    indicates a rename without ``$id`` update or vice versa.

    Returns ``None`` if ``$id`` is absent or not a string. Raises
    ``ValueError`` (via ``json.JSONDecodeError``) on malformed JSON.
    """
    data = json.loads(schema_text)
    if not isinstance(data, dict):
        return None
    id_value = data.get("$id")
    if not isinstance(id_value, str):
        return None
    return id_value.rsplit("/", 1)[-1]


def _load_schema_id_basename(schema_path: Path) -> str | None:
    """Read a JSON Schema file from disk and return the basename of
    its ``$id`` field (or None if absent).

    Used to detect drift between the schema filename on disk and the
    ``$id`` recorded inside the schema, which would silently break
    cross-schema ``$ref`` resolution and any documentation tooling
    that displays the ``$id``.
    """
    return _extract_schema_id_basename(
        schema_path.read_text(encoding="utf-8")
    )


def _extract_schema_excluded_op_class_enum(
    schema_text: str,
) -> frozenset[str] | None:
    """Drill into a JSON Schema document and return the
    ``properties.excluded_op_class.enum`` list as a frozenset.

    Each NVBit typed-reason schema constrains the
    ``excluded_op_class`` field to one of a fixed set of UPPER_SNAKE
    op-class names that mirrors
    ``import_nvbit_exclusion.VALID_OP_CLASSES``. Drift between the
    schema enum and the consumer admit list would silently allow
    sidecars carrying an op-class that one side admits and the
    other rejects.

    Returns ``None`` if any layer is missing or has the wrong shape,
    if the enum contains any non-string member (e.g. an integer or
    null literal), or if the enum list contains duplicate string
    members. The fail-closed shape lets a parity assertion catch
    malformed enum contents in addition to ordinary
    membership/cardinality drift. Raises ``ValueError`` via
    ``json.JSONDecodeError`` on malformed JSON.
    """
    data = json.loads(schema_text)
    if not isinstance(data, dict):
        return None
    properties = data.get("properties")
    if not isinstance(properties, dict):
        return None
    excluded_op_class_property = properties.get("excluded_op_class")
    if not isinstance(excluded_op_class_property, dict):
        return None
    enum_list = excluded_op_class_property.get("enum")
    if not isinstance(enum_list, list):
        return None
    if not all(isinstance(v, str) for v in enum_list):
        return None
    if len(set(enum_list)) != len(enum_list):
        return None
    return frozenset(enum_list)


def _load_schema_excluded_op_class_enum(
    schema_path: Path,
) -> frozenset[str] | None:
    """Read a JSON Schema file from disk and return the frozenset of
    its ``properties.excluded_op_class.enum`` values.

    Returns ``None`` if any layer is missing or wrong-shape (so a
    parity assertion can fail-closed).
    """
    return _extract_schema_excluded_op_class_enum(
        schema_path.read_text(encoding="utf-8")
    )


def _scan_sidecar_writer_typed_reasons() -> tuple[frozenset[str], list[tuple[str, str]]]:
    """Scan exclusion_aggregator.cc for write_one_typed_sidecar(...) calls.

    Returns the set of typed reasons emitted by the sidecar writer,
    plus the list of (file_slug, typed_reason) pairs so callers can
    verify that the sidecar filename slug matches the typed reason
    argument.
    """
    text = _AGGREGATOR_CC.read_text(encoding="utf-8")
    pattern = re.compile(
        r'"/(?P<file>[a-z_]+)\.json",\s*"(?P<reason>[a-z_]+)"'
    )
    pairs = [(m.group("file"), m.group("reason")) for m in pattern.finditer(text)]
    return frozenset(reason for _, reason in pairs), pairs


def _diff_sets(
    left: frozenset[str], right: frozenset[str]
) -> tuple[frozenset[str], frozenset[str]]:
    """Returns (missing_from_right, missing_from_left)."""
    return left - right, right - left


def _three_way_parity_diff(
    use_sites: frozenset[str],
    writers: frozenset[str],
    consumer: frozenset[str],
) -> dict[str, frozenset[str]]:
    """Compute downstream-direction parity diffs.

    Returns a dict naming reasons present in upstream layers but
    missing downstream:

    - ``missing_from_writers``: present at use-site, absent from writer
    - ``missing_from_consumer``: present at use-site, absent from consumer
    - ``missing_from_use_sites``: present at writer or consumer, absent
      from use-site (catches dead writer/consumer entries)
    """
    return {
        "missing_from_writers": use_sites - writers,
        "missing_from_consumer": use_sites - consumer,
        "missing_from_use_sites":
            (writers | consumer) - use_sites,
    }


class ExclusionTaxonomyParityTest(unittest.TestCase):
    """Producer/consumer typed-exclusion parity invariants."""

    def test_aggregate_use_sites_emit_only_expected_typed_reasons(self):
        use_site_reasons = _scan_aggregate_use_sites()
        self.assertEqual(
            use_site_reasons,
            PRODUCER_TYPED_REASONS_EXPECTED,
            f"Producer aggregate use-site set in {_TRACER_TOOL_CU.name} "
            f"drifted from expected. Update PRODUCER_TYPED_REASONS_EXPECTED "
            f"in this test AND sidecar writer (exclusion_aggregator.cc "
            f"write_one_typed_sidecar) AND consumer side "
            f"(import_nvbit_exclusion._SCHEMA_FILES) AND schema file "
            f"in lockstep when adding a new producer-emitted typed reason.",
        )

    def test_sidecar_writer_emits_only_expected_typed_reasons(self):
        writer_reasons, _ = _scan_sidecar_writer_typed_reasons()
        self.assertEqual(
            writer_reasons,
            PRODUCER_TYPED_REASONS_EXPECTED,
            f"Sidecar writer set in {_AGGREGATOR_CC.name} drifted from "
            f"expected. Each producer aggregate use site needs a matching "
            f"write_one_typed_sidecar call.",
        )

    def test_writer_file_slug_matches_typed_reason(self):
        _, pairs = _scan_sidecar_writer_typed_reasons()
        for file_slug, typed_reason in pairs:
            self.assertEqual(
                file_slug,
                typed_reason,
                f"Sidecar file slug {file_slug!r} does not match "
                f"typed_reason {typed_reason!r}; sidecar filename and "
                f"typed_reason field must agree.",
            )

    def test_three_way_parity_use_sites_writers_consumer(self):
        use_sites = _scan_aggregate_use_sites()
        writers, _ = _scan_sidecar_writer_typed_reasons()
        consumer = frozenset(import_nvbit_exclusion._SCHEMA_FILES.keys())
        diffs = _three_way_parity_diff(use_sites, writers, consumer)
        self.assertFalse(
            diffs["missing_from_writers"],
            f"Producer aggregates typed reasons that have no sidecar "
            f"writer: {sorted(diffs['missing_from_writers'])}. "
            f"Add matching write_one_typed_sidecar call(s) in "
            f"{_AGGREGATOR_CC.name}.",
        )
        self.assertFalse(
            diffs["missing_from_consumer"],
            f"Producer aggregates typed reasons not admitted by consumer: "
            f"{sorted(diffs['missing_from_consumer'])}. "
            f"Update import_nvbit_exclusion._SCHEMA_FILES and add a "
            f"matching JSON Schema file.",
        )
        self.assertFalse(
            diffs["missing_from_use_sites"],
            f"Sidecar writer or consumer carries typed reasons that no "
            f"producer aggregate use site emits: "
            f"{sorted(diffs['missing_from_use_sites'])}. "
            f"Either add a producer use site in {_TRACER_TOOL_CU.name} "
            f"or remove the dead writer/consumer entry.",
        )
        self.assertEqual(
            use_sites,
            PRODUCER_TYPED_REASONS_EXPECTED,
            "Three-way parity holds but does not match the expected set.",
        )

    def test_each_typed_reason_has_schema_file(self):
        for typed_reason in PRODUCER_TYPED_REASONS_EXPECTED:
            schema_path = _SCHEMA_DIR / f"nvbit_{typed_reason}.schema.json"
            self.assertTrue(
                schema_path.exists(),
                f"Schema file missing for typed_reason {typed_reason!r}: "
                f"{schema_path}",
            )

    def test_consumer_schema_filename_matches_typed_reason(self):
        for typed_reason, filename in import_nvbit_exclusion._SCHEMA_FILES.items():
            expected = f"nvbit_{typed_reason}.schema.json"
            self.assertEqual(
                filename,
                expected,
                f"Consumer schema filename {filename!r} does not follow "
                f"the nvbit_<typed_reason>.schema.json convention "
                f"for typed_reason {typed_reason!r}.",
            )

    def test_synthetic_drift_caught_by_diff_function(self):
        left = frozenset({"a", "b", "c"})
        right = frozenset({"a", "b", "d"})
        missing_from_right, missing_from_left = _diff_sets(left, right)
        self.assertEqual(missing_from_right, frozenset({"c"}))
        self.assertEqual(missing_from_left, frozenset({"d"}))

    def test_synthetic_parity_at_equality(self):
        same = frozenset({"a", "b", "c"})
        missing_from_right, missing_from_left = _diff_sets(same, same)
        self.assertFalse(missing_from_right)
        self.assertFalse(missing_from_left)

    def test_aggregate_only_new_reason_caught_by_three_way_diff(self):
        synthetic_use_sites = (
            PRODUCER_TYPED_REASONS_EXPECTED | {"synthetic_aggregate_only"}
        )
        synthetic_writers = PRODUCER_TYPED_REASONS_EXPECTED
        synthetic_consumer = PRODUCER_TYPED_REASONS_EXPECTED
        diffs = _three_way_parity_diff(
            synthetic_use_sites, synthetic_writers, synthetic_consumer
        )
        self.assertEqual(
            diffs["missing_from_writers"],
            frozenset({"synthetic_aggregate_only"}),
            "Aggregate-only synthetic reason must surface as missing "
            "from the sidecar-writer side.",
        )
        self.assertEqual(
            diffs["missing_from_consumer"],
            frozenset({"synthetic_aggregate_only"}),
            "Aggregate-only synthetic reason must surface as missing "
            "from the consumer admit list.",
        )
        self.assertFalse(
            diffs["missing_from_use_sites"],
            "Aggregate-only fixture should not surface false positives "
            "in the missing_from_use_sites diagnostic.",
        )

    def test_writer_only_orphan_caught_by_three_way_diff(self):
        synthetic_use_sites = PRODUCER_TYPED_REASONS_EXPECTED
        synthetic_writers = (
            PRODUCER_TYPED_REASONS_EXPECTED | {"synthetic_writer_orphan"}
        )
        synthetic_consumer = PRODUCER_TYPED_REASONS_EXPECTED
        diffs = _three_way_parity_diff(
            synthetic_use_sites, synthetic_writers, synthetic_consumer
        )
        self.assertEqual(
            diffs["missing_from_use_sites"],
            frozenset({"synthetic_writer_orphan"}),
            "Writer-only orphan reason must surface as missing from "
            "the producer use-site set.",
        )
        self.assertFalse(diffs["missing_from_writers"])
        self.assertFalse(diffs["missing_from_consumer"])

    def test_consumer_only_orphan_caught_by_three_way_diff(self):
        synthetic_use_sites = PRODUCER_TYPED_REASONS_EXPECTED
        synthetic_writers = PRODUCER_TYPED_REASONS_EXPECTED
        synthetic_consumer = (
            PRODUCER_TYPED_REASONS_EXPECTED | {"synthetic_consumer_orphan"}
        )
        diffs = _three_way_parity_diff(
            synthetic_use_sites, synthetic_writers, synthetic_consumer
        )
        self.assertEqual(
            diffs["missing_from_use_sites"],
            frozenset({"synthetic_consumer_orphan"}),
            "Consumer-only orphan reason must surface as missing from "
            "the producer use-site set.",
        )
        self.assertFalse(diffs["missing_from_writers"])
        self.assertFalse(diffs["missing_from_consumer"])


class AggregateScannerCallFormTest(unittest.TestCase):
    """Scanner must extract typed reason from BOTH valid C++ call forms.

    The producer's ``aggregate`` signature is
    ``const std::string &typed_reason``, so the final argument may be
    either a direct string literal (implicit-conversion form) or a
    ``std::string("...")`` wrapper. The unit-test source already uses
    the direct-literal form throughout. The Round 15 scanner only
    matched the wrapped form; Round 16 widens the scanner so future
    direct-literal aggregate calls cannot drift past the parity check.
    """

    def test_scanner_extracts_wrapped_form(self):
        synthetic = (
            "void f() {\n"
            "  nvbit_tracer::aggregate(\n"
            "      static_cast<uint32_t>(fid),\n"
            "      std::string(\"GLOBAL\"),\n"
            "      static_cast<uint64_t>(vpc),\n"
            "      std::string(\"unsupported_memory_path\"));\n"
            "}\n"
        )
        reasons = _scan_aggregate_use_sites_from_text(synthetic, "<synthetic>")
        self.assertEqual(reasons, frozenset({"unsupported_memory_path"}))

    def test_scanner_extracts_direct_literal_form(self):
        synthetic = (
            "void f() {\n"
            "  aggregate(1, \"GLOBAL\", 0x1000, \"unsupported_memory_path\");\n"
            "  nvbit_tracer::aggregate(2, \"TMA\", 0x2000, "
            "\"unmapped_memory_range\");\n"
            "}\n"
        )
        reasons = _scan_aggregate_use_sites_from_text(synthetic, "<synthetic>")
        self.assertEqual(
            reasons,
            frozenset({"unmapped_memory_range"}),
            "Scanner must extract direct-literal form; first call has "
            "no nvbit_tracer:: qualifier and is correctly ignored.",
        )

    def test_scanner_extracts_both_forms_in_one_pass(self):
        synthetic = (
            "void f() {\n"
            "  nvbit_tracer::aggregate(\n"
            "      static_cast<uint32_t>(fid),\n"
            "      std::string(\"GLOBAL\"),\n"
            "      static_cast<uint64_t>(vpc),\n"
            "      std::string(\"unsupported_memory_path\"));\n"
            "  nvbit_tracer::aggregate(\n"
            "      fid2, \"GLOBAL\", vpc2, \"synthetic_aggregate_only\");\n"
            "}\n"
        )
        reasons = _scan_aggregate_use_sites_from_text(synthetic, "<synthetic>")
        self.assertEqual(
            reasons,
            frozenset({"unsupported_memory_path", "synthetic_aggregate_only"}),
        )

    def test_direct_literal_synthetic_reason_caught_by_three_way_parity(self):
        """End-to-end source-fixture: scanner + diff cooperate to surface an
        aggregate-only direct-literal reason that has no writer / consumer /
        schema counterpart. Codex R15 review's exact failure scenario."""
        synthetic = (
            "void f() {\n"
            "  nvbit_tracer::aggregate(\n"
            "      static_cast<uint32_t>(fid),\n"
            "      std::string(\"GLOBAL\"),\n"
            "      static_cast<uint64_t>(vpc),\n"
            "      std::string(\"unsupported_memory_path\"));\n"
            "  nvbit_tracer::aggregate(\n"
            "      static_cast<uint32_t>(fid),\n"
            "      std::string(\"GLOBAL\"),\n"
            "      static_cast<uint64_t>(vpc),\n"
            "      std::string(\"unmapped_memory_range\"));\n"
            "  nvbit_tracer::aggregate(\n"
            "      static_cast<uint32_t>(fid),\n"
            "      std::string(\"GLOBAL\"),\n"
            "      static_cast<uint64_t>(vpc),\n"
            "      std::string(\"cross_allocation_lane_span\"));\n"
            "  nvbit_tracer::aggregate(fid2, \"GLOBAL\", vpc2, "
            "\"synthetic_aggregate_only\");\n"
            "}\n"
        )
        use_sites = _scan_aggregate_use_sites_from_text(synthetic, "<synthetic>")
        self.assertIn("synthetic_aggregate_only", use_sites)
        diffs = _three_way_parity_diff(
            use_sites,
            PRODUCER_TYPED_REASONS_EXPECTED,
            PRODUCER_TYPED_REASONS_EXPECTED,
        )
        self.assertEqual(
            diffs["missing_from_writers"],
            frozenset({"synthetic_aggregate_only"}),
        )
        self.assertEqual(
            diffs["missing_from_consumer"],
            frozenset({"synthetic_aggregate_only"}),
        )

    def test_scanner_raises_on_unparseable_typed_reason_argument(self):
        synthetic = (
            "void f() {\n"
            "  nvbit_tracer::aggregate(fid, \"GLOBAL\", vpc, "
            "get_dynamic_reason());\n"
            "}\n"
        )
        with self.assertRaises(ValueError) as cm:
            _scan_aggregate_use_sites_from_text(synthetic, "<synthetic>")
        self.assertIn("<synthetic>", str(cm.exception))
        self.assertIn("unparseable", str(cm.exception))

    def test_scanner_raises_on_wrong_argument_count(self):
        synthetic = (
            "void f() {\n"
            "  nvbit_tracer::aggregate(fid, \"GLOBAL\", vpc);\n"
            "}\n"
        )
        with self.assertRaises(ValueError) as cm:
            _scan_aggregate_use_sites_from_text(synthetic, "<synthetic>")
        self.assertIn("<synthetic>", str(cm.exception))
        self.assertIn("3 arguments", str(cm.exception))
        self.assertIn("expected 4", str(cm.exception))

    def test_scanner_raises_on_unterminated_aggregate_call(self):
        synthetic = (
            "void f() {\n"
            "  nvbit_tracer::aggregate(fid, \"GLOBAL\", vpc,\n"
            "      \"unsupported_memory_path\"\n"
            "// no closing paren\n"
        )
        with self.assertRaises(ValueError) as cm:
            _scan_aggregate_use_sites_from_text(synthetic, "<synthetic>")
        self.assertIn("<synthetic>", str(cm.exception))
        self.assertIn("unterminated", str(cm.exception))

    def test_scanner_returns_empty_when_no_aggregate_calls(self):
        synthetic = "void f() { /* no aggregate calls here */ }\n"
        reasons = _scan_aggregate_use_sites_from_text(synthetic, "<synthetic>")
        self.assertEqual(reasons, frozenset())


class AggregateScannerCommentAwarenessTest(unittest.TestCase):
    """Scanner must skip comments and string literals while walking source.

    A producer-use-site parity guard that counts any text mentioning
    ``nvbit_tracer::aggregate`` would silently mask removals: a future
    patch that comments out a real call but leaves the writer / consumer
    / schema entries in place would still pass the parity test because
    the commented-out text gets counted. Round 17 lexes the source to
    skip non-code regions before matching the aggregate token.
    """

    def test_scanner_ignores_aggregate_in_line_comment(self):
        synthetic = (
            "// nvbit_tracer::aggregate(fid, \"GLOBAL\", vpc, "
            "\"unmapped_memory_range\");\n"
        )
        reasons = _scan_aggregate_use_sites_from_text(
            synthetic, "<line-comment-fixture>"
        )
        self.assertEqual(
            reasons,
            frozenset(),
            "Scanner must ignore aggregate text inside line comments.",
        )

    def test_scanner_ignores_aggregate_in_block_comment(self):
        synthetic = (
            "/* nvbit_tracer::aggregate(fid, \"GLOBAL\", vpc, "
            "\"fake_block_reason\"); */\n"
        )
        reasons = _scan_aggregate_use_sites_from_text(
            synthetic, "<block-comment-fixture>"
        )
        self.assertEqual(reasons, frozenset())

    def test_scanner_ignores_aggregate_in_multiline_block_comment(self):
        synthetic = (
            "/*\n"
            " * Documentation referring to the producer call:\n"
            " *   nvbit_tracer::aggregate(fid, \"GLOBAL\", vpc,\n"
            " *       \"fake_doc_reason\");\n"
            " */\n"
        )
        reasons = _scan_aggregate_use_sites_from_text(
            synthetic, "<multiline-comment-fixture>"
        )
        self.assertEqual(reasons, frozenset())

    def test_scanner_ignores_aggregate_in_string_literal(self):
        synthetic = (
            "const char* doc = "
            "\"nvbit_tracer::aggregate(fid, \\\"GLOBAL\\\", vpc, "
            "\\\"fake_string_reason\\\");\";\n"
        )
        reasons = _scan_aggregate_use_sites_from_text(
            synthetic, "<string-literal-fixture>"
        )
        self.assertEqual(reasons, frozenset())

    def test_scanner_extracts_real_call_adjacent_to_comments(self):
        synthetic = (
            "// nvbit_tracer::aggregate(fid, \"GLOBAL\", vpc, "
            "\"fake_line_reason\");\n"
            "/* nvbit_tracer::aggregate(fid, \"GLOBAL\", vpc, "
            "\"fake_block_reason\"); */\n"
            "void f() {\n"
            "    const char* doc = \"nvbit_tracer::aggregate(...)\";\n"
            "    nvbit_tracer::aggregate(fid, \"GLOBAL\", vpc, "
            "\"real_reason\");\n"
            "}\n"
        )
        reasons = _scan_aggregate_use_sites_from_text(
            synthetic, "<adjacent-fixture>"
        )
        self.assertEqual(
            reasons,
            frozenset({"real_reason"}),
            "Scanner must extract a real call adjacent to non-code "
            "text without extracting any of the non-code reasons.",
        )

    def test_comment_only_use_site_surfaces_as_missing_from_use_sites(self):
        """Codex's R16-named failure scenario: a real call gets commented
        out, but the writer / consumer / schema still claim that reason.
        The parity guard must surface the reason as ``missing_from_use_sites``.
        """
        synthetic = (
            "// nvbit_tracer::aggregate(fid, \"GLOBAL\", vpc, "
            "\"unsupported_memory_path\");  // call removed in this patch\n"
            "void f() {\n"
            "    nvbit_tracer::aggregate(fid, \"GLOBAL\", vpc, "
            "\"unmapped_memory_range\");\n"
            "    nvbit_tracer::aggregate(fid, \"GLOBAL\", vpc, "
            "\"cross_allocation_lane_span\");\n"
            "}\n"
        )
        use_sites = _scan_aggregate_use_sites_from_text(
            synthetic, "<comment-only-removal-fixture>"
        )
        self.assertEqual(
            use_sites,
            frozenset({"unmapped_memory_range", "cross_allocation_lane_span"}),
            "Scanner must NOT extract the commented-out call's reason.",
        )
        diffs = _three_way_parity_diff(
            use_sites,
            PRODUCER_TYPED_REASONS_EXPECTED,
            PRODUCER_TYPED_REASONS_EXPECTED,
        )
        self.assertEqual(
            diffs["missing_from_use_sites"],
            frozenset({"unsupported_memory_path"}),
            "Commented-out call's reason must surface as missing from "
            "use-sites; otherwise a producer-use-site removal goes "
            "undetected by the parity guard.",
        )
        self.assertFalse(diffs["missing_from_writers"])
        self.assertFalse(diffs["missing_from_consumer"])

    def test_scanner_handles_consecutive_line_comments(self):
        synthetic = (
            "// first comment\n"
            "// nvbit_tracer::aggregate(fid, \"GLOBAL\", vpc, "
            "\"fake_a\");\n"
            "// nvbit_tracer::aggregate(fid, \"GLOBAL\", vpc, "
            "\"fake_b\");\n"
            "nvbit_tracer::aggregate(fid, \"GLOBAL\", vpc, \"real_c\");\n"
        )
        reasons = _scan_aggregate_use_sites_from_text(
            synthetic, "<consecutive-comments-fixture>"
        )
        self.assertEqual(reasons, frozenset({"real_c"}))


class AggregateScannerRawStringTest(unittest.TestCase):
    """Scanner must skip C++ raw string literals.

    Raw strings can contain unescaped ``"`` characters and span
    multiple lines, so an ordinary string-literal skipper that
    treats every ``"`` as a token boundary will fall out of the
    string at the first embedded quote and start parsing the rest
    as code. The walker must recognize ``R"(...)"``, ``u8R"(...)"``,
    ``uR"(...)"``, ``UR"(...)"``, and ``LR"(...)"`` openers and
    skip to the matching closing ``)`` + delim + ``"``.
    """

    def test_scanner_ignores_aggregate_in_raw_string_no_embedded_quote(self):
        synthetic = (
            "const char* doc = R\"(\n"
            "  nvbit_tracer::aggregate(fid, \"GLOBAL\", vpc, "
            "\"raw_no_quote_reason\");\n"
            ")\";\n"
        )
        reasons = _scan_aggregate_use_sites_from_text(
            synthetic, "<raw-no-embedded-quote>"
        )
        self.assertEqual(reasons, frozenset())

    def test_scanner_ignores_aggregate_in_raw_string_with_embedded_quote(self):
        """Codex's R17 reproduction: embedded ``"`` before aggregate text."""
        synthetic = (
            "const char* doc = R\"(prefix \" "
            "nvbit_tracer::aggregate(fid, \"GLOBAL\", vpc, "
            "\"raw_fake_reason\");)\";\n"
        )
        reasons = _scan_aggregate_use_sites_from_text(
            synthetic, "<raw-with-embedded-quote>"
        )
        self.assertEqual(reasons, frozenset())

    def test_scanner_ignores_aggregate_in_raw_string_with_delimiter(self):
        synthetic = (
            "const char* doc = R\"foo(\n"
            "  nvbit_tracer::aggregate(fid, \"GLOBAL\", vpc, "
            "\"raw_delim_reason\");\n"
            ")foo\";\n"
        )
        reasons = _scan_aggregate_use_sites_from_text(
            synthetic, "<raw-with-delimiter>"
        )
        self.assertEqual(reasons, frozenset())

    def test_scanner_ignores_aggregate_in_u8_prefixed_raw_string(self):
        synthetic = (
            "const char* utf8_doc = u8R\"(\n"
            "  nvbit_tracer::aggregate(fid, \"GLOBAL\", vpc, "
            "\"u8_raw_reason\");\n"
            ")\";\n"
        )
        reasons = _scan_aggregate_use_sites_from_text(
            synthetic, "<u8-raw-string>"
        )
        self.assertEqual(reasons, frozenset())

    def test_raw_string_only_use_site_surfaces_as_missing_from_use_sites(self):
        """Raw-string analogue of the R17 comment-only-removal scenario."""
        synthetic = (
            "const char* doc = R\"(\n"
            "  nvbit_tracer::aggregate(fid, \"GLOBAL\", vpc, "
            "\"unsupported_memory_path\");\n"
            ")\";\n"
            "void f() {\n"
            "    nvbit_tracer::aggregate(fid, \"GLOBAL\", vpc, "
            "\"unmapped_memory_range\");\n"
            "    nvbit_tracer::aggregate(fid, \"GLOBAL\", vpc, "
            "\"cross_allocation_lane_span\");\n"
            "}\n"
        )
        use_sites = _scan_aggregate_use_sites_from_text(
            synthetic, "<raw-string-removal-fixture>"
        )
        self.assertEqual(
            use_sites,
            frozenset({"unmapped_memory_range", "cross_allocation_lane_span"}),
        )
        diffs = _three_way_parity_diff(
            use_sites,
            PRODUCER_TYPED_REASONS_EXPECTED,
            PRODUCER_TYPED_REASONS_EXPECTED,
        )
        self.assertEqual(
            diffs["missing_from_use_sites"],
            frozenset({"unsupported_memory_path"}),
            "Raw-string-only use site must surface as missing from "
            "use-sites; otherwise a producer-use-site removal hidden "
            "behind a raw-string mention goes undetected by the parity "
            "guard.",
        )
        self.assertFalse(diffs["missing_from_writers"])
        self.assertFalse(diffs["missing_from_consumer"])

    def test_real_call_after_raw_string_extracted(self):
        synthetic = (
            "const char* doc = R\"(\n"
            "  nvbit_tracer::aggregate(fid, \"GLOBAL\", vpc, "
            "\"raw_fake\");\n"
            ")\";\n"
            "nvbit_tracer::aggregate(fid, \"GLOBAL\", vpc, "
            "\"real_after_raw\");\n"
        )
        reasons = _scan_aggregate_use_sites_from_text(
            synthetic, "<real-after-raw>"
        )
        self.assertEqual(reasons, frozenset({"real_after_raw"}))

    def test_unterminated_raw_string_fails_closed(self):
        """An unterminated raw string should consume the rest of the text;
        aggregate text inside cannot leak as a live use site."""
        synthetic = (
            "const char* doc = R\"(\n"
            "  nvbit_tracer::aggregate(fid, \"GLOBAL\", vpc, "
            "\"exposed_reason\");\n"
            "// no closing delim\n"
        )
        reasons = _scan_aggregate_use_sites_from_text(
            synthetic, "<unterminated-raw>"
        )
        self.assertEqual(reasons, frozenset())

    def test_raw_string_after_identifier_not_recognized(self):
        """R"..." mid-identifier (``foo R"..."``) — the prior char is part
        of an identifier, so the R" is NOT a raw-string opener. The walker
        falls through to the regular ``"`` skipper for the parenthesized
        content. After it, a real call is still extracted.
        """
        synthetic = (
            "const char* fooR = \"plain string\";\n"
            "nvbit_tracer::aggregate(fid, \"GLOBAL\", vpc, "
            "\"real_reason\");\n"
        )
        reasons = _scan_aggregate_use_sites_from_text(
            synthetic, "<id-prefix-fixture>"
        )
        self.assertEqual(reasons, frozenset({"real_reason"}))


class LocalLabelsTypedReasonsParityTest(unittest.TestCase):
    """Parity tests for the local-labels typed-reason surface.

    The repeatability aggregator emits three typed reasons into
    ``label_excluded.json`` entries: ``label_repeatability_failed``,
    ``label_repeatability_missing``, ``label_provenance_inconsistent``.
    Unlike the NVBit-tracer surface, there is no separate sidecar
    writer or consumer admit list — the aggregator is producer-only.
    The parity check is therefore a single producer-side scan against
    the expected set, plus a disjoint-set check ensuring the local-
    labels surface does not reuse NVBit-tracer typed reasons, plus a
    documented-coverage check against the AC-19 plan enumeration.
    """

    def test_local_labels_aggregator_emits_only_expected_typed_reasons(self):
        emitted = _scan_repeatability_aggregator_typed_reasons()
        self.assertEqual(
            emitted,
            LOCAL_LABELS_TYPED_REASONS_EXPECTED,
            f"Local-labels typed-reason set in "
            f"{_REPEATABILITY_AGGREGATOR_PY.name} drifted from expected. "
            f"Update LOCAL_LABELS_TYPED_REASONS_EXPECTED in this test "
            f"AND verify the AC-19 enumeration / documentation in "
            f"plans/nonroot_gpu_cache_observability_plan.md and "
            f"tools/local_labels/README.md when adding or removing a "
            f"label-related typed reason.",
        )

    def test_local_labels_typed_reasons_disjoint_from_nvbit_tracer(self):
        overlap = (
            LOCAL_LABELS_TYPED_REASONS_EXPECTED
            & PRODUCER_TYPED_REASONS_EXPECTED
        )
        self.assertFalse(
            overlap,
            f"Local-labels and NVBit-tracer typed-reason sets must be "
            f"disjoint; overlap detected: {sorted(overlap)}. The two "
            f"surfaces emit different exclusion taxonomies and an "
            f"accidental name collision would break per-surface dispatch.",
        )

    def test_implemented_typed_reasons_are_documented(self):
        """Every currently-implemented typed reason must appear in EITHER
        the AC-19 plan enumeration OR the explicit round-added set. A new
        producer-side typed reason that lands without either documentation
        is a contract drift the parity guard catches.
        """
        implemented = (
            PRODUCER_TYPED_REASONS_EXPECTED
            | LOCAL_LABELS_TYPED_REASONS_EXPECTED
            | TRACE_BUDGET_TYPED_REASONS_EXPECTED
            | SECTOR_STREAM_TYPED_REASONS_EXPECTED
        )
        documented = (
            AC19_PLAN_ENUMERATED_TYPED_REASONS | ROUND_ADDED_TYPED_REASONS
        )
        undocumented = implemented - documented
        self.assertFalse(
            undocumented,
            f"Implemented typed reasons not documented in either the "
            f"AC-19 plan enumeration or ROUND_ADDED_TYPED_REASONS: "
            f"{sorted(undocumented)}. Either add the new reason to the "
            f"plan's AC-19 list (and update AC19_PLAN_ENUMERATED_TYPED_REASONS) "
            f"or document it as a round-added taxonomy extension in "
            f"ROUND_ADDED_TYPED_REASONS.",
        )


class PlanTextParityTest(unittest.TestCase):
    """Parity tests between
    ``plans/nonroot_gpu_cache_observability_plan.md`` (the AC-19
    source-of-truth document) and the test code's
    ``AC19_PLAN_ENUMERATED_TYPED_REASONS`` constant.

    The plan's AC-19 positive-tests section enumerates the typed-
    exclusion taxonomy on a single line:
    ``The exclusion taxonomy includes (at minimum): ...``. Each
    typed reason is wrapped in backticks. The wildcard entry
    ``arch_feature_unavailable_*`` denotes a family pattern (e.g.
    ``arch_feature_unavailable_l2_prefetch_256b``) and is excluded
    from the test code's literal-set constant.

    The R19 ``test_implemented_typed_reasons_are_documented`` test
    catches drift between IMPLEMENTED reasons and the
    ``AC19_PLAN_ENUMERATED_TYPED_REASONS | ROUND_ADDED_TYPED_REASONS``
    union, but it does NOT catch drift between the constant and the
    plan's enumeration. If a future plan revision adds, removes, or
    renames a reason in the AC-19 enumeration but the test code's
    constant isn't updated in lockstep (or vice versa), no current
    test fails. R28 closes that drift class.
    """

    def test_plan_text_ac19_enumeration_matches_constant(self):
        emitted = _scan_plan_ac19_enumerated_typed_reasons()
        self.assertEqual(
            emitted,
            AC19_PLAN_ENUMERATED_TYPED_REASONS,
            f"AC-19 typed-reason enumeration drift detected. The "
            f"plan at {_PLAN_PATH.name} has "
            f"{sorted(emitted - AC19_PLAN_ENUMERATED_TYPED_REASONS)} "
            f"reasons not in AC19_PLAN_ENUMERATED_TYPED_REASONS, and "
            f"the constant has "
            f"{sorted(AC19_PLAN_ENUMERATED_TYPED_REASONS - emitted)} "
            f"reasons not in the plan. Update one or both to match.",
        )

    def test_synthetic_basic_extraction(self):
        text = (
            "    - The exclusion taxonomy includes (at minimum): "
            "`reason_a`, `reason_b`, `reason_c`.\n"
        )
        result = _scan_plan_ac19_enumerated_typed_reasons_from_text(text)
        self.assertEqual(result, frozenset({"reason_a", "reason_b", "reason_c"}))

    def test_synthetic_drift_extra_in_plan(self):
        text = (
            "The exclusion taxonomy includes (at minimum): "
            "`reason_a`, `reason_b`, `reason_extra`.\n"
        )
        result = _scan_plan_ac19_enumerated_typed_reasons_from_text(text)
        self.assertIn("reason_extra", result)
        self.assertEqual(
            result,
            frozenset({"reason_a", "reason_b", "reason_extra"}),
        )

    def test_synthetic_drift_missing_from_plan(self):
        text = (
            "The exclusion taxonomy includes (at minimum): "
            "`reason_a`, `reason_b`.\n"
        )
        result = _scan_plan_ac19_enumerated_typed_reasons_from_text(text)
        self.assertNotIn("reason_c", result)
        self.assertEqual(result, frozenset({"reason_a", "reason_b"}))

    def test_synthetic_wildcard_filtered_out(self):
        text = (
            "The exclusion taxonomy includes (at minimum): "
            "`arch_feature_unavailable_*`, `concrete_reason`.\n"
        )
        result = _scan_plan_ac19_enumerated_typed_reasons_from_text(text)
        self.assertEqual(
            result,
            frozenset({"concrete_reason"}),
            "Wildcard entries (ending in '*') must be filtered out so "
            "the parity assertion against the literal-set constant "
            "remains valid.",
        )

    def test_synthetic_anchor_not_found_raises(self):
        text = (
            "Some other plan content with `reason_a` and `reason_b` "
            "but no AC-19 anchor.\n"
        )
        with self.assertRaises(ValueError) as ctx:
            _scan_plan_ac19_enumerated_typed_reasons_from_text(text)
        self.assertIn(
            "AC-19 enumeration anchor",
            str(ctx.exception),
            "Missing anchor must surface as a ValueError with a "
            "diagnostic message; silently returning an empty set "
            "would mask a renamed or deleted plan section.",
        )


class MemoryOpClassParityTest(unittest.TestCase):
    """Producer/consumer parity for the memory op-class enum.

    The producer side defines the C++ enum
    ``memory_op_class_classifier::Value`` in
    ``tools/nvbit_tracer/tracer_tool/memory_op_class.h`` with 11
    entries: ``kUnspecified``, ``kGlobal``, ``kShared``, ``kLocal``,
    ``kTexture``, ``kSurface``, ``kAtomic``, ``kTma``, ``kAsyncCopy``,
    ``kClusterDsmem``, ``kTmem``. The consumer side declares
    ``import_nvbit_exclusion.VALID_OP_CLASSES`` as a tuple of 11
    UPPER_SNAKE strings: ``UNSPECIFIED``, ``GLOBAL``, ``SHARED``,
    ``LOCAL``, ``TEXTURE``, ``SURFACE``, ``ATOMIC``, ``TMA``,
    ``ASYNC_COPY``, ``CLUSTER_DSMEM``, ``TMEM``. The TMEM entry was
    added for Blackwell (sm_100) tensor-memory accesses via the
    `tcgen05.{ld,st}` family (mnemonics `LDT`/`STT`).

    The names map 1:1 via ``kCamelCase`` → ``UPPER_SNAKE``
    transformation. Drift between producer and consumer breaks the
    validator pathway for STANDARD_MEM exclusions: the producer
    would emit ``excluded_op_class: "FOO"`` and the validator would
    silently reject every record carrying that op-class.

    Pre-R29, ``test_every_op_class_admissible`` in
    ``test_import_nvbit_exclusion.py`` verified that
    ``VALID_OP_CLASSES`` values are accepted by the schema validator
    but did NOT scan the producer's C++ enum. R29 closes that gap.
    """

    def test_producer_enum_matches_consumer_admit_list(self):
        producer_enum = _scan_producer_memory_op_class_enum()
        consumer_admit = frozenset(import_nvbit_exclusion.VALID_OP_CLASSES)
        self.assertEqual(
            producer_enum,
            consumer_admit,
            f"Producer/consumer op-class drift detected. "
            f"memory_op_class.h has "
            f"{sorted(producer_enum - consumer_admit)} not in "
            f"VALID_OP_CLASSES, and VALID_OP_CLASSES has "
            f"{sorted(consumer_admit - producer_enum)} not in "
            f"memory_op_class.h. Either rename the producer enum "
            f"or update the consumer admit list to match.",
        )

    def test_synthetic_basic_extraction(self):
        text = """
namespace memory_op_class_classifier {
enum Value : int {
  kUnspecified  = 0,
  kGlobal       = 1,
  kAsyncCopy    = 2,
};
}
"""
        result = _scan_producer_memory_op_class_enum_from_text(text)
        self.assertEqual(
            result, frozenset({"UNSPECIFIED", "GLOBAL", "ASYNC_COPY"})
        )

    def test_synthetic_enum_block_missing_raises(self):
        text = "namespace x { int kFoo = 0; }"
        with self.assertRaises(ValueError) as ctx:
            _scan_producer_memory_op_class_enum_from_text(text)
        self.assertIn(
            "enum Value",
            str(ctx.exception),
            "Missing enum block must surface as a ValueError with "
            "a diagnostic message; silently returning empty would "
            "mask a renamed or deleted enum.",
        )

    def test_synthetic_drift_extra_in_producer(self):
        text = """
enum Value : int {
  kGlobal = 0,
  kFutureClass = 1,
};
"""
        result = _scan_producer_memory_op_class_enum_from_text(text)
        self.assertIn("FUTURE_CLASS", result)
        self.assertEqual(result, frozenset({"GLOBAL", "FUTURE_CLASS"}))

    def test_synthetic_block_comment_entry_ignored(self):
        text = """
enum Value : int {
  kGlobal = 0,
  /* kCommentedOut = 99, */
};
"""
        result = _scan_producer_memory_op_class_enum_from_text(text)
        self.assertEqual(
            result,
            frozenset({"GLOBAL"}),
            "A block-commented enum entry must NOT be counted as a "
            "live entry.",
        )

    def test_synthetic_line_comment_entry_ignored(self):
        text = """
enum Value : int {
  kGlobal = 0,
  // kLineCommented = 99,
};
"""
        result = _scan_producer_memory_op_class_enum_from_text(text)
        self.assertEqual(
            result,
            frozenset({"GLOBAL"}),
            "A `//` line-commented enum entry must NOT be counted "
            "as a live entry.",
        )

    def test_kcamel_case_to_upper_snake_simple(self):
        self.assertEqual(_k_camel_case_to_upper_snake("kGlobal"), "GLOBAL")
        self.assertEqual(_k_camel_case_to_upper_snake("kShared"), "SHARED")
        self.assertEqual(_k_camel_case_to_upper_snake("kLocal"), "LOCAL")
        self.assertEqual(_k_camel_case_to_upper_snake("kTma"), "TMA")
        self.assertEqual(
            _k_camel_case_to_upper_snake("kUnspecified"), "UNSPECIFIED"
        )

    def test_kcamel_case_to_upper_snake_compound(self):
        self.assertEqual(
            _k_camel_case_to_upper_snake("kAsyncCopy"), "ASYNC_COPY"
        )
        self.assertEqual(
            _k_camel_case_to_upper_snake("kClusterDsmem"),
            "CLUSTER_DSMEM",
        )

    def test_kcamel_case_to_upper_snake_invalid_raises(self):
        with self.assertRaises(ValueError):
            _k_camel_case_to_upper_snake("AsyncCopy")
        with self.assertRaises(ValueError):
            _k_camel_case_to_upper_snake("klowercase")
        with self.assertRaises(ValueError):
            _k_camel_case_to_upper_snake("")
        with self.assertRaises(ValueError):
            _k_camel_case_to_upper_snake("k")

    def test_synthetic_digit_bearing_enum_entry_extracted(self):
        text = """
enum Value : int {
  kGlobal = 0,
  kL2Prefetch = 1,
};
"""
        result = _scan_producer_memory_op_class_enum_from_text(text)
        self.assertEqual(
            result,
            frozenset({"GLOBAL", "L2_PREFETCH"}),
            "Digit-bearing kCamelCase identifiers (e.g. kL2Prefetch) "
            "must be extracted by the enum scanner. The R29 regex "
            "`\\b(k[A-Z][A-Za-z]*)\\b` excluded digits in identifier "
            "tails, silently skipping such entries; R30 widens the "
            "tail to `[A-Za-z0-9]*`.",
        )

    def test_kcamel_case_to_upper_snake_with_digits(self):
        self.assertEqual(
            _k_camel_case_to_upper_snake("kL2Prefetch"),
            "L2_PREFETCH",
            "Digits in the identifier tail are passed through; the "
            "uppercase letter following the digit triggers the "
            "underscore separator.",
        )
        self.assertEqual(
            _k_camel_case_to_upper_snake("kFooBar2"),
            "FOO_BAR2",
            "Trailing digit is passed through unchanged.",
        )
        self.assertEqual(
            _k_camel_case_to_upper_snake("kFoo2Bar"),
            "FOO2_BAR",
            "Digit between camel groups is passed through; the "
            "uppercase letter after the digit still triggers the "
            "underscore separator.",
        )

    def test_synthetic_fake_enum_block_in_string_literal_skipped(self):
        text = (
            'const char* doc = "enum Value : int { kFake = 99, };";\n'
            "enum Value : int { kGlobal = 0, };"
        )
        result = _scan_producer_memory_op_class_enum_from_text(text)
        self.assertEqual(
            result,
            frozenset({"GLOBAL"}),
            "An `enum Value : int { ... };` block appearing inside "
            "an ordinary C++ string literal must NOT be extracted; "
            "only the real block following the string should "
            "produce entries. Pre-R37 the scanner used a "
            "comment-only stripper (which preserved string "
            "literals), so the fake block was matched first by the "
            "regex (the live regex is non-greedy and finds the "
            "first match). R37 switches the scanner to "
            "`_scrub_cpp_non_code_regions`, which blanks string "
            "contents before the regex is applied.",
        )

    def test_synthetic_fake_enum_block_in_raw_string_literal_skipped(
        self,
    ):
        text = (
            'const char* doc = R"D(enum Value : int { kFake = 99, };)D";\n'
            "enum Value : int { kGlobal = 0, };"
        )
        result = _scan_producer_memory_op_class_enum_from_text(text)
        self.assertEqual(
            result,
            frozenset({"GLOBAL"}),
            "An `enum Value : int { ... };` block inside a C++ raw "
            "string literal `R\"D(...)D\"` must NOT be extracted. "
            "The unified scrubber's raw-string-first ordering "
            "consumes the raw string as a single region before any "
            "regex matching occurs.",
        )


class ProtoMemoryOpClassEnumParityTest(unittest.TestCase):
    """Parity tests for the proto-side ``MemoryOpClass`` enum vs the
    consumer admit list ``VALID_OP_CLASSES``.

    The op-class enum is replicated across four files:

    1. ``tools/nvbit_tracer/traces_enhanced/dynamic_trace/instruction.proto``
       — ``enum MemoryOpClass`` with ``MEMORY_OP_CLASS_<NAME>`` entries
       (the wire encoding).
    2. ``tools/nvbit_tracer/tracer_tool/memory_op_class.h`` — C++
       ``enum Value`` with ``k<Name>`` entries (the producer-side
       classifier).
    3. ``tools/nvbit_tracer/import_nvbit_exclusion.py`` —
       ``VALID_OP_CLASSES`` tuple with ``<NAME>`` UPPER_SNAKE entries
       (the consumer admit list).
    4. The 3 NVBit typed-reason JSON schemas under
       ``workloads/simbench_bw/schemas/`` — ``excluded_op_class.enum``
       arrays with the same UPPER_SNAKE entries.

    The producer-side ``static_assert`` block in ``tracer_tool.cu``
    pins the C++ ↔ proto numeric value mapping at compile time, but
    that catches ONLY numeric drift, not name drift. A proto-side
    rename like ``MEMORY_OP_CLASS_GLOBAL`` →
    ``MEMORY_OP_CLASS_DEVICE_GLOBAL`` would preserve the numeric
    value and pass the static_assert, but silently change the wire
    encoding name. R33 closes that gap by directly scanning the
    proto enum.
    """

    def test_live_proto_enum_matches_valid_op_classes(self):
        proto_enum = _scan_proto_memory_op_class_enum()
        expected = frozenset(import_nvbit_exclusion.VALID_OP_CLASSES)
        self.assertEqual(
            len(proto_enum),
            len(import_nvbit_exclusion.VALID_OP_CLASSES),
            "Proto MemoryOpClass enum cardinality must match "
            "VALID_OP_CLASSES (no hidden duplicates or extra "
            "entries). Live proto entries: "
            f"{sorted(proto_enum)}; expected: "
            f"{sorted(expected)}.",
        )
        self.assertEqual(
            proto_enum,
            expected,
            "Proto MemoryOpClass enum must match VALID_OP_CLASSES "
            "exactly. Drift here means the proto wire encoding has "
            "diverged from the consumer admit list. Producer-side "
            "static_assert in tracer_tool.cu catches numeric drift "
            "but not name drift, so a proto rename can silently "
            "change the wire encoding without failing the build.",
        )

    def test_synthetic_basic_extraction(self):
        text = """
syntax = "proto3";

enum MemoryOpClass {
  MEMORY_OP_CLASS_UNSPECIFIED = 0;
  MEMORY_OP_CLASS_GLOBAL = 1;
  MEMORY_OP_CLASS_SHARED = 2;
}
"""
        result = _extract_proto_memory_op_class_enum_from_text(text)
        self.assertEqual(
            result,
            frozenset({"UNSPECIFIED", "GLOBAL", "SHARED"}),
            "Basic extraction with the MEMORY_OP_CLASS_ prefix "
            "stripped should yield the bare entry names.",
        )

    def test_synthetic_missing_enum_block_raises(self):
        text = """
syntax = "proto3";

message instruction {
  uint32 pc = 1;
}
"""
        with self.assertRaises(ValueError):
            _extract_proto_memory_op_class_enum_from_text(text)

    def test_synthetic_drift_extra_in_proto(self):
        text = """
enum MemoryOpClass {
  MEMORY_OP_CLASS_GLOBAL = 0;
  MEMORY_OP_CLASS_SHARED = 1;
  MEMORY_OP_CLASS_FUTURE_OP = 2;
}
"""
        result = _extract_proto_memory_op_class_enum_from_text(text)
        self.assertIn(
            "FUTURE_OP",
            result,
            "An extra proto entry must surface in the extracted set "
            "so the live parity assertion can detect the drift.",
        )

    def test_synthetic_drift_missing_in_proto(self):
        text = """
enum MemoryOpClass {
  MEMORY_OP_CLASS_GLOBAL = 0;
}
"""
        result = _extract_proto_memory_op_class_enum_from_text(text)
        self.assertNotIn(
            "SHARED",
            result,
            "An expected entry that is absent from the proto must "
            "be absent from the extracted set so the live parity "
            "assertion can detect the drift.",
        )

    def test_synthetic_block_comment_skipped(self):
        text = """
enum MemoryOpClass {
  MEMORY_OP_CLASS_GLOBAL = 0;
  /* MEMORY_OP_CLASS_GHOST = 1; */
  MEMORY_OP_CLASS_SHARED = 2;
}
"""
        result = _extract_proto_memory_op_class_enum_from_text(text)
        self.assertEqual(
            result,
            frozenset({"GLOBAL", "SHARED"}),
            "A block-commented entry must NOT be extracted. Proto3 "
            "uses C++-style /* ... */ block comments, which the "
            "shared `_scrub_cpp_non_code_regions` helper removes "
            "before the entry regex is applied.",
        )

    def test_synthetic_line_comment_skipped(self):
        text = """
enum MemoryOpClass {
  MEMORY_OP_CLASS_GLOBAL = 0;
  // MEMORY_OP_CLASS_GHOST = 1;
  MEMORY_OP_CLASS_SHARED = 2;
}
"""
        result = _extract_proto_memory_op_class_enum_from_text(text)
        self.assertEqual(
            result,
            frozenset({"GLOBAL", "SHARED"}),
            "A line-commented entry must NOT be extracted. Proto3 "
            "uses C++-style // line comments, which the shared "
            "`_scrub_cpp_non_code_regions` helper removes before "
            "the entry regex is applied.",
        )

    def test_synthetic_fake_enum_block_in_string_literal_skipped(self):
        text = (
            'syntax = "proto3";\n'
            'option foo = "enum MemoryOpClass { '
            'MEMORY_OP_CLASS_FAKE = 99; }";\n'
            "enum MemoryOpClass { MEMORY_OP_CLASS_GLOBAL = 0; }\n"
        )
        result = _extract_proto_memory_op_class_enum_from_text(text)
        self.assertEqual(
            result,
            frozenset({"GLOBAL"}),
            "An `enum MemoryOpClass { ... }` block appearing inside "
            "an ordinary string literal must NOT be extracted; only "
            "the real block should produce entries. Pre-R37 the "
            "scanner used a comment-only stripper (which preserved "
            "string literals), so the fake block was matched first "
            "by the non-greedy regex. R37 switches the scanner to "
            "`_scrub_cpp_non_code_regions`, which blanks string "
            "contents before the regex is applied.",
        )

    def test_synthetic_fake_enum_block_in_raw_string_literal_skipped(
        self,
    ):
        text = (
            'syntax = "proto3";\n'
            'const char* doc = R"D(enum MemoryOpClass { '
            'MEMORY_OP_CLASS_FAKE = 99; })D";\n'
            "enum MemoryOpClass { MEMORY_OP_CLASS_GLOBAL = 0; }\n"
        )
        result = _extract_proto_memory_op_class_enum_from_text(text)
        self.assertEqual(
            result,
            frozenset({"GLOBAL"}),
            "An `enum MemoryOpClass { ... }` block inside a C++ "
            "raw string literal must NOT be extracted. Proto3 does "
            "not natively support C++ raw strings, but the unified "
            "scrubber is shared between C++ and proto scanners; "
            "raw strings are correctly recognized and skipped here "
            "even when used in proto-style source.",
        )


class StaticAssertNumericMappingParityTest(unittest.TestCase):
    """Parity tests for the static_assert C++/proto numeric-mapping
    block in ``tracer_tool.cu``.

    The producer-side static_assert block at lines 108-147 of
    ``tools/nvbit_tracer/tracer_tool/tracer_tool.cu`` pins each
    ``memory_op_class_classifier::Value`` C++ enum entry 1:1 with
    its corresponding ``dynamic_trace::MemoryOpClass`` proto enum
    entry by numeric value:

    .. code-block:: cpp

        static_assert(
            static_cast<int>(memory_op_class_classifier::kAtomic) ==
                static_cast<int>(dynamic_trace::MEMORY_OP_CLASS_ATOMIC),
            "memory_op_class_classifier::kAtomic must mirror MEMORY_OP_CLASS_ATOMIC");

    The static_assert catches NUMERIC drift at compile time (a swap
    of two values would fail the build). What it does NOT catch:

    1. **Missing static_assert pair**: a developer adds a new entry
       to both ``Value`` and ``MemoryOpClass`` but forgets to add
       the corresponding static_assert. R29 and R33 still pass
       (both enums have the new entry), but the compile-time
       numeric pin is incomplete for that entry — a future swap
       would no longer fail.
    2. **Stale static_assert pair**: a developer removes an entry
       from both enums but forgets to remove the static_assert.
       The compile would fail (dangling references), but a static
       check catches it pre-compile.

    R34 closes both gaps by directly scanning the static_assert
    block and verifying the pairs cover every C++ entry (R29) and
    every proto entry (R33).
    """

    def test_live_static_asserts_cover_all_cpp_entries(self):
        pairs = _scan_static_assert_pairs()
        cpp_names_upper_snake = frozenset(
            _k_camel_case_to_upper_snake(cpp) for cpp, _ in pairs
        )
        producer_enum = _scan_producer_memory_op_class_enum()
        self.assertEqual(
            cpp_names_upper_snake,
            producer_enum,
            "Every entry in the producer C++ enum "
            "(memory_op_class_classifier::Value) must have a "
            "matching static_assert pair in tracer_tool.cu. A "
            "missing pair means the compile-time numeric pin is "
            "incomplete for that entry; a stale pair means the "
            "static_assert references a dangling symbol. "
            f"C++ static_assert names (UPPER_SNAKE): "
            f"{sorted(cpp_names_upper_snake)}; producer enum: "
            f"{sorted(producer_enum)}.",
        )

    def test_live_static_asserts_cover_all_proto_entries(self):
        pairs = _scan_static_assert_pairs()
        proto_names = frozenset(proto for _, proto in pairs)
        proto_enum = _scan_proto_memory_op_class_enum()
        self.assertEqual(
            proto_names,
            proto_enum,
            "Every entry in the proto MemoryOpClass enum must have "
            "a matching static_assert pair in tracer_tool.cu. A "
            "proto-side rename without an updated static_assert "
            "would leave the compile-time pin pointing to a "
            "dangling proto symbol. "
            f"Static_assert proto names: {sorted(proto_names)}; "
            f"proto enum: {sorted(proto_enum)}.",
        )

    def test_live_static_assert_pair_names_match(self):
        pairs = _scan_static_assert_pairs()
        self.assertGreater(
            len(pairs),
            0,
            "The static_assert block must contain at least one "
            "pair; an empty block indicates the compile-time "
            "numeric pin has been removed entirely.",
        )
        for cpp, proto in pairs:
            self.assertEqual(
                _k_camel_case_to_upper_snake(cpp),
                proto,
                "Each static_assert pair must satisfy the "
                "kCamelCase ↔ UPPER_SNAKE name mapping. A pair "
                f"like ({cpp!r}, {proto!r}) where the names "
                "diverge indicates a typo or partial rename. The "
                "C++ compiler would catch a numeric mismatch, but "
                "this test catches the inconsistency pre-compile.",
            )

    def test_synthetic_basic_extraction(self):
        text = """
#include "memory_op_class.h"

static_assert(
    static_cast<int>(memory_op_class_classifier::kGlobal) ==
        static_cast<int>(dynamic_trace::MEMORY_OP_CLASS_GLOBAL),
    "memory_op_class_classifier::kGlobal must mirror MEMORY_OP_CLASS_GLOBAL");
static_assert(
    static_cast<int>(memory_op_class_classifier::kAtomic) ==
        static_cast<int>(dynamic_trace::MEMORY_OP_CLASS_ATOMIC),
    "memory_op_class_classifier::kAtomic must mirror MEMORY_OP_CLASS_ATOMIC");
"""
        result = _extract_static_assert_pairs_from_text(text)
        self.assertEqual(
            result,
            frozenset({("kGlobal", "GLOBAL"), ("kAtomic", "ATOMIC")}),
            "Basic extraction yields (C++ name with k prefix, proto "
            "name with MEMORY_OP_CLASS_ prefix stripped) tuples.",
        )

    def test_synthetic_block_comment_skipped(self):
        text = """
static_assert(
    static_cast<int>(memory_op_class_classifier::kGlobal) ==
        static_cast<int>(dynamic_trace::MEMORY_OP_CLASS_GLOBAL),
    "memory_op_class_classifier::kGlobal must mirror MEMORY_OP_CLASS_GLOBAL");
/* static_assert(
    static_cast<int>(memory_op_class_classifier::kGhost) ==
        static_cast<int>(dynamic_trace::MEMORY_OP_CLASS_GHOST),
    "ghost"); */
static_assert(
    static_cast<int>(memory_op_class_classifier::kAtomic) ==
        static_cast<int>(dynamic_trace::MEMORY_OP_CLASS_ATOMIC),
    "memory_op_class_classifier::kAtomic must mirror MEMORY_OP_CLASS_ATOMIC");
"""
        result = _extract_static_assert_pairs_from_text(text)
        self.assertEqual(
            result,
            frozenset({("kGlobal", "GLOBAL"), ("kAtomic", "ATOMIC")}),
            "A block-commented static_assert must NOT be extracted. "
            "The shared `_scrub_cpp_non_code_regions` helper removes "
            "/* ... */ blocks before the pair regex is applied.",
        )

    def test_synthetic_line_comment_skipped(self):
        text = """
static_assert(
    static_cast<int>(memory_op_class_classifier::kGlobal) ==
        static_cast<int>(dynamic_trace::MEMORY_OP_CLASS_GLOBAL),
    "memory_op_class_classifier::kGlobal must mirror MEMORY_OP_CLASS_GLOBAL");
// static_assert(static_cast<int>(memory_op_class_classifier::kGhost) == static_cast<int>(dynamic_trace::MEMORY_OP_CLASS_GHOST), "ghost");
static_assert(
    static_cast<int>(memory_op_class_classifier::kAtomic) ==
        static_cast<int>(dynamic_trace::MEMORY_OP_CLASS_ATOMIC),
    "memory_op_class_classifier::kAtomic must mirror MEMORY_OP_CLASS_ATOMIC");
"""
        result = _extract_static_assert_pairs_from_text(text)
        self.assertEqual(
            result,
            frozenset({("kGlobal", "GLOBAL"), ("kAtomic", "ATOMIC")}),
            "A line-commented static_assert must NOT be extracted. "
            "The shared `_scrub_cpp_non_code_regions` helper removes "
            "// line comments before the pair regex is applied.",
        )

    def test_synthetic_missing_pair_surfaces_as_drift(self):
        text = """
static_assert(
    static_cast<int>(memory_op_class_classifier::kGlobal) ==
        static_cast<int>(dynamic_trace::MEMORY_OP_CLASS_GLOBAL),
    "memory_op_class_classifier::kGlobal must mirror MEMORY_OP_CLASS_GLOBAL");
static_assert(
    static_cast<int>(memory_op_class_classifier::kAtomic) ==
        static_cast<int>(dynamic_trace::MEMORY_OP_CLASS_ATOMIC),
    "memory_op_class_classifier::kAtomic must mirror MEMORY_OP_CLASS_ATOMIC");
"""
        result = _extract_static_assert_pairs_from_text(text)
        cpp_names_upper_snake = frozenset(
            _k_camel_case_to_upper_snake(cpp) for cpp, _ in result
        )
        expected_full_set = frozenset({"GLOBAL", "ATOMIC", "SHARED"})
        missing = expected_full_set - cpp_names_upper_snake
        self.assertEqual(
            missing,
            frozenset({"SHARED"}),
            "An expected static_assert pair that is absent must "
            "surface as a missing entry when the live extracted set "
            "is diffed against the expected set. This is the live "
            "test's mechanism for catching missing static_asserts "
            "after a new entry is added to both enums.",
        )

    def test_synthetic_static_assert_in_ordinary_string_literal_skipped(
        self,
    ):
        text = (
            'const char* s = "static_assert('
            "static_cast<int>(memory_op_class_classifier::kAtomic) == "
            "static_cast<int>(dynamic_trace::MEMORY_OP_CLASS_ATOMIC),"
            '\";'
        )
        result = _extract_static_assert_pairs_from_text(text)
        self.assertEqual(
            result,
            frozenset(),
            "A static_assert text appearing inside an ordinary C++ "
            "string literal (`\"...\"`) must NOT be extracted. This "
            "is Codex's R34 review reproduction case: the "
            "scanner pre-R35 counted the inside-string snippet as a "
            "real pair; R35 added a string-literal blanking step on "
            "top of comment stripping, and R36 unified both into "
            "`_scrub_cpp_non_code_regions` to blank string-literal "
            "contents before the regex is applied.",
        )

    def test_synthetic_static_assert_in_char_literal_skipped(self):
        text = (
            "char c = 'X';\n"
            'const char* s = "static_assert('
            "static_cast<int>(memory_op_class_classifier::kGlobal) == "
            "static_cast<int>(dynamic_trace::MEMORY_OP_CLASS_GLOBAL),"
            '";'
        )
        result = _extract_static_assert_pairs_from_text(text)
        self.assertEqual(
            result,
            frozenset(),
            "A char literal ('X') alongside a string literal "
            "containing static_assert text must be handled "
            "correctly: the char literal is consumed by "
            "`_skip_char_literal`, then the string literal is "
            "consumed by `_skip_string_literal`, and neither leaks "
            "into the regex.",
        )

    def test_synthetic_static_assert_in_raw_string_literal_skipped(
        self,
    ):
        text = (
            'const char* s = R"(static_assert('
            "static_cast<int>(memory_op_class_classifier::kAtomic) == "
            "static_cast<int>(dynamic_trace::MEMORY_OP_CLASS_ATOMIC),"
            '")";'
        )
        result = _extract_static_assert_pairs_from_text(text)
        self.assertEqual(
            result,
            frozenset(),
            "A static_assert text inside a C++ raw string literal "
            "(`R\"(...)\"`) must NOT be extracted. Raw strings have "
            "different skip semantics than ordinary strings (no "
            "escape processing; closing delimiter is "
            "`)<delim>\"`); `_scrub_cpp_non_code_regions` checks "
            "raw-string opener forms BEFORE the regular `\"` branch "
            "so raw strings are correctly identified.",
        )

    def test_synthetic_static_assert_in_raw_string_with_embedded_quote_skipped(
        self,
    ):
        text = (
            'const char* s = R"DELIM(prefix " static_assert('
            "static_cast<int>(memory_op_class_classifier::kAtomic) == "
            "static_cast<int>(dynamic_trace::MEMORY_OP_CLASS_ATOMIC),"
            ' suffix)DELIM";'
        )
        result = _extract_static_assert_pairs_from_text(text)
        self.assertEqual(
            result,
            frozenset(),
            "A raw string with a non-empty delimiter and an embedded "
            "ordinary `\"` character must still consume the entire "
            "raw string (until the matching `)DELIM\"`), so the "
            "static_assert text after the embedded `\"` is NOT "
            "leaked. This mirrors Codex's R17 reproduction for the "
            "aggregate-use-site scanner.",
        )

    def test_synthetic_real_pair_after_string_literal_extracted(self):
        text = """
const char* s = "static_assert(static_cast<int>(memory_op_class_classifier::kFake) == static_cast<int>(dynamic_trace::MEMORY_OP_CLASS_FAKE),";
static_assert(
    static_cast<int>(memory_op_class_classifier::kAtomic) ==
        static_cast<int>(dynamic_trace::MEMORY_OP_CLASS_ATOMIC),
    "memory_op_class_classifier::kAtomic must mirror MEMORY_OP_CLASS_ATOMIC");
"""
        result = _extract_static_assert_pairs_from_text(text)
        self.assertEqual(
            result,
            frozenset({("kAtomic", "ATOMIC")}),
            "A REAL static_assert pair appearing after a string "
            "literal that contains static_assert text must extract "
            "the real pair only. The string-literal contents are "
            "blanked but the surrounding code remains intact, so "
            "the real call's regex match is unaffected.",
        )

    def test_synthetic_raw_string_with_embedded_quote_and_line_comment_then_real_pair_extracts_real(
        self,
    ):
        text = (
            'const char* s = R"D(prefix " // fake)D";\n'
            "static_assert(\n"
            "    static_cast<int>(memory_op_class_classifier::kAtomic) ==\n"
            "        static_cast<int>(dynamic_trace::MEMORY_OP_CLASS_ATOMIC),\n"
            '    "memory_op_class_classifier::kAtomic must mirror MEMORY_OP_CLASS_ATOMIC");\n'
        )
        result = _extract_static_assert_pairs_from_text(text)
        self.assertEqual(
            result,
            frozenset({("kAtomic", "ATOMIC")}),
            "A real static_assert pair appearing after a raw string "
            "literal whose body contains both an embedded `\"` and "
            "`//` comment-shaped text must still be extracted. This "
            "is Codex's R35 review reproduction case for the "
            "raw-string + comment ordering bug: the pre-R36 chain "
            "ran a comment stripper first, which was not "
            "raw-string aware, so the embedded `\"` closed the "
            "comment stripper's ordinary-string fast-skip and "
            "exposed the `//` to the comment stripper. R36 unifies "
            "the lexical scrubbing into a single walker with "
            "raw-string detection BEFORE comment detection, so the "
            "raw string is consumed as one unit before any `//` "
            "lookups happen.",
        )

    def test_synthetic_raw_string_with_embedded_quote_and_block_comment_then_real_pair_extracts_real(
        self,
    ):
        text = (
            'const char* s = R"D(prefix " /* fake comment */ )D";\n'
            "static_assert(\n"
            "    static_cast<int>(memory_op_class_classifier::kAtomic) ==\n"
            "        static_cast<int>(dynamic_trace::MEMORY_OP_CLASS_ATOMIC),\n"
            '    "memory_op_class_classifier::kAtomic must mirror MEMORY_OP_CLASS_ATOMIC");\n'
        )
        result = _extract_static_assert_pairs_from_text(text)
        self.assertEqual(
            result,
            frozenset({("kAtomic", "ATOMIC")}),
            "Same as the line-comment variant but with `/* ... */` "
            "block comment-shaped text inside the raw string. The "
            "pre-R36 chain would have entered the block-comment "
            "fast-skip after the embedded `\"` closed the "
            "ordinary-string mode, potentially consuming text up "
            "through the next `*/` (which inside the raw string "
            "would not exist, so the chain consumed text to "
            "end-of-text). R36 consumes the raw string as one unit.",
        )

    def test_synthetic_raw_string_with_embedded_quote_and_comment_shaped_static_assert_remains_ignored(
        self,
    ):
        text = (
            'const char* s = R"D(prefix " // static_assert('
            "static_cast<int>(memory_op_class_classifier::kAtomic) == "
            "static_cast<int>(dynamic_trace::MEMORY_OP_CLASS_ATOMIC),"
            ' end)D";\n'
        )
        result = _extract_static_assert_pairs_from_text(text)
        self.assertEqual(
            result,
            frozenset(),
            "A raw string containing both an embedded `\"` and a "
            "static_assert-shaped snippet (separated by a `//` "
            "that is itself inside the raw string) must NOT extract "
            "the snippet. The unified scrubber consumes the entire "
            "raw string as one region; nothing inside the raw "
            "string can leak to the regex.",
        )


class LocalLabelsScannerCommentAwarenessTest(unittest.TestCase):
    """Comment / string-literal awareness tests for the local-labels
    typed-reason scanner.

    The pre-R24 scanner used a plain regex that would have falsely
    matched ``"typed_reason": "<reason>"`` patterns appearing inside
    Python line comments, docstrings, or single-line string literals.
    Codex's R23 review queued this as a valid hardening item: "Local-
    labels typed-reason scanner comment/string awareness ... does not
    block the current parity result because the current
    repeatability_aggregator.py content does not contain non-code
    `\"typed_reason\": \"<reason>\"` snippets."

    Round 24 closes the latent gap by switching to a Python tokenize-
    based scanner that walks token streams and only counts triples of
    code-position tokens (STRING typed_reason, OP `:`, STRING reason
    value). Trivia tokens (line comments, indent/dedent, newlines)
    are filtered before pattern matching; multi-line strings appear
    as a single STRING token whose decoded value contains the embedded
    pattern as one big string, NOT as three separate STRING/OP/STRING
    tokens.

    These fixtures are the Python analog of the R17-R18
    AggregateScannerCommentAwarenessTest / AggregateScannerRawStringTest
    fixtures for the C++ aggregate scanner.
    """

    def test_scanner_extracts_real_dict_literal(self):
        text = '''
{"typed_reason": "real_reason"}
'''
        result = _scan_repeatability_aggregator_typed_reasons_from_text(
            text, "<fixture>"
        )
        self.assertEqual(result, frozenset({"real_reason"}))

    def test_scanner_ignores_typed_reason_in_line_comment(self):
        text = '''
# "typed_reason": "fake_commented_reason"
'''
        result = _scan_repeatability_aggregator_typed_reasons_from_text(
            text, "<fixture>"
        )
        self.assertEqual(result, frozenset())

    def test_scanner_ignores_typed_reason_in_docstring(self):
        text = '''
def f():
    """
    Example output: "typed_reason": "fake_docstring_reason"
    """
    pass
'''
        result = _scan_repeatability_aggregator_typed_reasons_from_text(
            text, "<fixture>"
        )
        self.assertEqual(result, frozenset())

    def test_scanner_ignores_typed_reason_in_single_line_string(self):
        text = '''
template = "the format is \\"typed_reason\\": \\"fake_string_reason\\""
'''
        result = _scan_repeatability_aggregator_typed_reasons_from_text(
            text, "<fixture>"
        )
        self.assertEqual(result, frozenset())

    def test_scanner_extracts_real_call_adjacent_to_comments(self):
        text = '''
# old shape: "typed_reason": "fake_old_reason"
record = {"typed_reason": "real_reason"}
# removed: "typed_reason": "fake_removed_reason"
'''
        result = _scan_repeatability_aggregator_typed_reasons_from_text(
            text, "<fixture>"
        )
        self.assertEqual(result, frozenset({"real_reason"}))

    def test_comment_only_use_site_surfaces_as_missing_from_use_sites(self):
        text = '''
# OLD CODE (removed):
# entry = {"typed_reason": "label_repeatability_failed"}
'''
        result = _scan_repeatability_aggregator_typed_reasons_from_text(
            text, "<fixture>"
        )
        self.assertEqual(
            result,
            frozenset(),
            "A comment-only typed_reason mention must NOT be counted as "
            "a live producer entry. Removing the real call and leaving "
            "only the comment must surface as 'missing' from the live "
            "scan.",
        )

    def test_scanner_returns_empty_when_no_typed_reasons(self):
        text = '''
def empty():
    return {}
'''
        result = _scan_repeatability_aggregator_typed_reasons_from_text(
            text, "<fixture>"
        )
        self.assertEqual(result, frozenset())

    def test_scanner_handles_single_quoted_strings(self):
        text = '''
{'typed_reason': 'single_quoted_reason'}
'''
        result = _scan_repeatability_aggregator_typed_reasons_from_text(
            text, "<fixture>"
        )
        self.assertEqual(
            result,
            frozenset({"single_quoted_reason"}),
            "tokenize is quote-style-agnostic; both single- and double-"
            "quoted Python strings decode to the same value. The scanner "
            "should accept both forms.",
        )


class TraceBudgetScannerCommentAwarenessTest(unittest.TestCase):
    """Comment / string-literal awareness tests for the three R21
    trace-budget scanners.

    The pre-R25 scanners used raw regexes (`sidecar\\s*=\\s*\\w+\\s*/
    \\s*"<reason>.json"`, the rglob discovery regex, and the
    `_validate_file` schema regex) that would have falsely matched
    the same patterns inside Python line comments, docstrings, or
    string literals — a latent robustness gap Codex's R24 review
    named explicitly. The current `run_suite.py` and `validate.py`
    files do not exercise the gap, but the tokenize-based scanners
    fail closed against future drift.

    These fixtures mirror the R24
    LocalLabelsScannerCommentAwarenessTest pattern: each of the 3
    scanners has a real-extract / comment-ignored / docstring-ignored
    triple of fixtures.
    """

    def test_run_suite_scanner_extracts_real_writer_literal(self):
        text = '''
def f(out_dir):
    sidecar = out_dir / "real_reason.json"
    return sidecar
'''
        result = _scan_trace_budget_run_suite_typed_reasons_from_text(
            text, "<fixture>"
        )
        self.assertEqual(result, frozenset({"real_reason"}))

    def test_run_suite_scanner_ignores_writer_literal_in_line_comment(self):
        text = '''
def f(out_dir):
    # sidecar = out_dir / "fake_reason.json"
    pass
'''
        result = _scan_trace_budget_run_suite_typed_reasons_from_text(
            text, "<fixture>"
        )
        self.assertEqual(result, frozenset())

    def test_run_suite_scanner_ignores_writer_literal_in_docstring(self):
        text = '''
def f(out_dir):
    """
    Example: sidecar = out_dir / "fake_reason.json"
    """
    pass
'''
        result = _scan_trace_budget_run_suite_typed_reasons_from_text(
            text, "<fixture>"
        )
        self.assertEqual(result, frozenset())

    def test_discovery_scanner_extracts_real_rglob_pattern(self):
        text = '''
def f(results_root, errors):
    for sp in results_root.rglob("real_reason.json"):
        errors.extend(validate_trace_sidecar(sp))
'''
        result = _scan_trace_budget_validator_discovery_reasons_from_text(
            text, "<fixture>"
        )
        self.assertEqual(result, frozenset({"real_reason"}))

    def test_discovery_scanner_ignores_rglob_in_line_comment(self):
        text = '''
def f(results_root, errors):
    # for sp in results_root.rglob("fake_reason.json"):
    #     errors.extend(validate_trace_sidecar(sp))
    pass
'''
        result = _scan_trace_budget_validator_discovery_reasons_from_text(
            text, "<fixture>"
        )
        self.assertEqual(result, frozenset())

    def test_discovery_scanner_ignores_rglob_in_docstring(self):
        text = '''
def f(results_root, errors):
    """
    Example:
        for sp in results_root.rglob("fake_reason.json"):
            errors.extend(validate_trace_sidecar(sp))
    """
    pass
'''
        result = _scan_trace_budget_validator_discovery_reasons_from_text(
            text, "<fixture>"
        )
        self.assertEqual(result, frozenset())

    def test_schema_scanner_extracts_real_schema_literal(self):
        text = '''
def f(data):
    SCHEMAS_DIR = "/x"
    errors = _validate_file(data, SCHEMAS_DIR / "real_reason.schema.json", f"trace_sidecar:row {repr(slug)}")
    return errors
'''
        result = _scan_trace_budget_validator_schema_reasons_from_text(
            text, "<fixture>"
        )
        self.assertEqual(result, frozenset({"real_reason"}))

    def test_schema_scanner_ignores_schema_literal_in_line_comment(self):
        text = '''
def f(data):
    # errors = _validate_file(data, SCHEMAS_DIR / "fake_reason.schema.json", f"trace_sidecar:row {repr(slug)}")
    pass
'''
        result = _scan_trace_budget_validator_schema_reasons_from_text(
            text, "<fixture>"
        )
        self.assertEqual(result, frozenset())

    def test_schema_scanner_ignores_schema_literal_in_docstring(self):
        text = '''
def f(data):
    """
    Example call:
        errors = _validate_file(data, SCHEMAS_DIR / "fake_reason.schema.json", f"trace_sidecar:row {repr(slug)}")
    """
    pass
'''
        result = _scan_trace_budget_validator_schema_reasons_from_text(
            text, "<fixture>"
        )
        self.assertEqual(result, frozenset())


class TraceBudgetSchemaScannerCompatTest(unittest.TestCase):
    """Cross-Python-version compatibility tests for the trace-budget
    schema scanner's f-string disambiguator.

    R25 introduced a schema scanner that disambiguates the trace-budget
    ``_validate_file(...)`` call from other ``SCHEMAS_DIR /`` references
    by recognizing the unique ``f"trace_sidecar:..."`` diagnostic
    prefix that follows the schema-path argument. R25 over-assumed the
    Python 3.12+ tokenization shape (FSTRING_START + FSTRING_MIDDLE)
    and directly referenced ``tokenize.FSTRING_START`` /
    ``tokenize.FSTRING_MIDDLE``, which do not exist on Python 3.11.

    The repo's declared Python floor is 3.11+ (per
    ``workloads/simbench_bw/DEPENDENCIES.md``), so R26 closes the
    cross-version compatibility gap with a dual-shape helper:

    - Python 3.12+ path: FSTRING_START + FSTRING_MIDDLE (current
      development host shape).
    - Python 3.11 path: a single STRING token whose ``.string``
      retains the f-prefix and quotes (e.g.
      ``'f"trace_sidecar:row {repr(slug)}"'``).

    These fixtures construct synthetic ``tokenize.TokenInfo`` instances
    and pass them through the helper directly, so they exercise both
    tokenization shapes regardless of the host Python version. The
    Python 3.12+-shape test is gated on the FSTRING_START constant
    being available.
    """

    def _make_token(self, type_, string):
        return tokenize.TokenInfo(
            type=type_, string=string, start=(1, 0), end=(1, 0), line=""
        )

    @unittest.skipIf(
        _FSTRING_START_TYPE is None or _FSTRING_MIDDLE_TYPE is None,
        "Python 3.12+ FSTRING_START / FSTRING_MIDDLE not available on this host",
    )
    def test_helper_accepts_python_312_fstring_shape(self):
        following = [
            self._make_token(_FSTRING_START_TYPE, 'f"'),
            self._make_token(_FSTRING_MIDDLE_TYPE, "trace_sidecar:row "),
        ]
        self.assertTrue(
            _following_tokens_match_trace_sidecar_fstring(following)
        )

    def test_helper_accepts_python_311_whole_fstring_shape(self):
        following = [
            self._make_token(
                tokenize.STRING,
                'f"trace_sidecar:row {repr(slug)}/{repr(shape)}"',
            )
        ]
        self.assertTrue(
            _following_tokens_match_trace_sidecar_fstring(following)
        )

    def test_helper_accepts_single_quoted_python_311_fstring(self):
        following = [
            self._make_token(
                tokenize.STRING,
                "f'trace_sidecar:row {repr(slug)}'",
            )
        ]
        self.assertTrue(
            _following_tokens_match_trace_sidecar_fstring(following)
        )

    def test_helper_accepts_capital_F_prefix(self):
        following = [
            self._make_token(
                tokenize.STRING,
                'F"trace_sidecar:row {x}"',
            )
        ]
        self.assertTrue(
            _following_tokens_match_trace_sidecar_fstring(following)
        )

    def test_helper_rejects_non_fstring_string_token(self):
        following = [
            self._make_token(
                tokenize.STRING,
                '"trace_sidecar:not an fstring"',
            )
        ]
        self.assertFalse(
            _following_tokens_match_trace_sidecar_fstring(following),
            "A regular STRING (without f-prefix) must NOT match; the "
            "trace-budget validator uses an f-string for the "
            "diagnostic prefix.",
        )

    def test_helper_rejects_fstring_with_wrong_prefix(self):
        following = [
            self._make_token(
                tokenize.STRING,
                'f"other_diagnostic:wrong prefix"',
            )
        ]
        self.assertFalse(
            _following_tokens_match_trace_sidecar_fstring(following),
            "An f-string whose body starts with anything other than "
            "'trace_sidecar:' must NOT match.",
        )

    def test_helper_rejects_empty_following_tokens_list(self):
        self.assertFalse(
            _following_tokens_match_trace_sidecar_fstring([]),
            "Empty following-tokens list must return False without "
            "raising IndexError.",
        )


class TraceBudgetConstantsScannerCommentAwarenessTest(unittest.TestCase):
    """Comment / string-literal awareness tests for the trace-budget
    constants scanner.

    The pre-R27 scanner used a regex
    ``^[A-Z_]+_SIDECAR\\s*:\\s*str\\s*=\\s*"(?P<reason>[a-z_]+)\\.json"``
    with ``re.MULTILINE``. The ``^`` anchor + uppercase identifier
    requirement made the scanner naturally robust against `#`-prefixed
    line comments (which would start with `# ` not `[A-Z_]+_SIDECAR`),
    but a docstring or multi-line string at column 0 with the same
    constant pattern would have false-matched at the line-start anchor
    despite being non-code.

    Round 27 closes that latent gap by switching to a Python tokenize-
    based scanner that walks token streams and only counts 4-token
    preceding patterns at code position. Trivia tokens (line comments,
    indent/dedent, newlines) are filtered before pattern matching;
    multi-line strings appear as a single STRING token whose decoded
    value contains the embedded pattern as one big string, NOT as four
    separate NAME/OP/NAME/OP tokens preceding a STRING.

    These fixtures mirror the R24-R25 scanner-comment-awareness test
    pattern and round out the AC-19 parity scanner robustness across
    every Python source surface in the test file.
    """

    def test_scanner_extracts_real_constant_declaration(self):
        text = '''
TRACE_VOLUME_EXCEEDED_SIDECAR: str = "real_reason.json"
'''
        result = _scan_trace_budget_typed_reasons_from_text(
            text, "<fixture>"
        )
        self.assertEqual(result, frozenset({"real_reason"}))

    def test_scanner_ignores_constant_pattern_in_docstring(self):
        text = '''
"""
Example constant declaration:
TRACE_VOLUME_EXCEEDED_SIDECAR: str = "fake_docstring_reason.json"
"""
'''
        result = _scan_trace_budget_typed_reasons_from_text(
            text, "<fixture>"
        )
        self.assertEqual(
            result,
            frozenset(),
            "A constant pattern inside a docstring (column-0 line "
            "inside a triple-quoted string) must NOT be counted as a "
            "live constant declaration. The pre-R27 regex would have "
            "false-matched this; the tokenize-based scanner must not.",
        )

    def test_scanner_ignores_constant_pattern_in_line_comment(self):
        text = '''
# TRACE_VOLUME_EXCEEDED_SIDECAR: str = "fake_comment_reason.json"
'''
        result = _scan_trace_budget_typed_reasons_from_text(
            text, "<fixture>"
        )
        self.assertEqual(result, frozenset())

    def test_scanner_ignores_constant_pattern_in_string_literal(self):
        text = '''
template = "TRACE_VOLUME_EXCEEDED_SIDECAR: str = \\"fake_string_reason.json\\""
'''
        result = _scan_trace_budget_typed_reasons_from_text(
            text, "<fixture>"
        )
        self.assertEqual(result, frozenset())

    def test_scanner_extracts_real_constant_adjacent_to_docstring(self):
        text = '''
"""
Older form (deprecated):
DEPRECATED_SIDECAR: str = "deprecated_reason.json"
"""

TRACE_VOLUME_EXCEEDED_SIDECAR: str = "real_reason.json"
'''
        result = _scan_trace_budget_typed_reasons_from_text(
            text, "<fixture>"
        )
        self.assertEqual(
            result,
            frozenset({"real_reason"}),
            "The real code-position constant must be extracted; the "
            "fake docstring-form constant must be ignored.",
        )

    def test_scanner_ignores_constant_with_wrong_type_annotation(self):
        text = '''
MY_SIDECAR: int = "x.json"
'''
        result = _scan_trace_budget_typed_reasons_from_text(
            text, "<fixture>"
        )
        self.assertEqual(
            result,
            frozenset(),
            "The pre-R27 regex required exactly `: str =`; the "
            "tokenize-based scanner preserves that requirement and "
            "skips constants annotated with any other type.",
        )

    def test_scanner_ignores_constant_with_wrong_suffix(self):
        text = '''
MY_THING: str = "x.json"
'''
        result = _scan_trace_budget_typed_reasons_from_text(
            text, "<fixture>"
        )
        self.assertEqual(
            result,
            frozenset(),
            "The constant-name regex requires `_SIDECAR` suffix; "
            "constants without that suffix are not trace-budget "
            "filename declarations.",
        )


class TraceBudgetTypedReasonsParityTest(unittest.TestCase):
    """Parity tests for the trace-budget typed-reason surface.

    The trace-budget surface (``tools/cloud/trace_budgets.py``) emits
    typed reasons via filename-constant declarations of the shape
    ``<UPPER_NAME>_SIDECAR: str = "<reason>.json"``. Producer use
    sites in ``workloads/simbench_bw/run_suite.py`` write the sidecar
    to that filename; the verifier in
    ``workloads/simbench_bw/validate.py`` validates against
    ``<reason>.schema.json`` (no ``nvbit_`` prefix). Plan AC-19
    enumerates ``trace_volume_exceeded`` as one of the at-minimum
    typed reasons; this test pins the production set so future
    additions or removals are caught at test time.
    """

    def test_trace_budgets_emits_only_expected_typed_reasons(self):
        emitted = _scan_trace_budget_typed_reasons()
        self.assertEqual(
            emitted,
            TRACE_BUDGET_TYPED_REASONS_EXPECTED,
            f"Trace-budget typed-reason set in {_TRACE_BUDGETS_PY.name} "
            f"drifted from expected. Update "
            f"TRACE_BUDGET_TYPED_REASONS_EXPECTED in this test AND verify "
            f"the producer use sites in workloads/simbench_bw/run_suite.py "
            f"and the verifier in workloads/simbench_bw/validate.py and "
            f"the schema file in workloads/simbench_bw/schemas/ when "
            f"adding or removing a trace-budget typed reason.",
        )

    def test_each_trace_budget_typed_reason_has_schema_file(self):
        for reason in TRACE_BUDGET_TYPED_REASONS_EXPECTED:
            schema_path = _SCHEMA_DIR / f"{reason}.schema.json"
            self.assertTrue(
                schema_path.exists(),
                f"Schema file missing for trace-budget typed_reason "
                f"{reason!r}: {schema_path}. Trace-budget schemas use the "
                f"<reason>.schema.json naming convention (no nvbit_ prefix).",
            )

    def test_trace_budget_reasons_disjoint_from_other_surfaces(self):
        nvbit_overlap = (
            TRACE_BUDGET_TYPED_REASONS_EXPECTED & PRODUCER_TYPED_REASONS_EXPECTED
        )
        self.assertFalse(
            nvbit_overlap,
            f"Trace-budget and NVBit-tracer typed-reason sets must be "
            f"disjoint; overlap detected: {sorted(nvbit_overlap)}.",
        )
        local_labels_overlap = (
            TRACE_BUDGET_TYPED_REASONS_EXPECTED & LOCAL_LABELS_TYPED_REASONS_EXPECTED
        )
        self.assertFalse(
            local_labels_overlap,
            f"Trace-budget and local-labels typed-reason sets must be "
            f"disjoint; overlap detected: {sorted(local_labels_overlap)}.",
        )


class TraceBudgetLiveLayerParityTest(unittest.TestCase):
    """Four-way parity tests for the trace-budget typed-reason surface
    against its actual live producer/verifier code.

    Round 20 added a parity scanner for the constant declarations in
    ``tools/cloud/trace_budgets.py``, but Codex's R20 review pointed
    out that AC-19 demands "parity check between producer code and
    verifier code" verbatim — and constants alone do not catch drift
    between the writer literals (``run_suite.py``) and the verifier
    literals (``validate.py``). Round 21 closes that gap by scanning
    all four live layers and asserting they are identical:

    1. ``tools/cloud/trace_budgets.py`` constants (R20).
    2. ``workloads/simbench_bw/run_suite.py`` writer literals
       (``sidecar = <dir> / "<reason>.json"``).
    3. ``workloads/simbench_bw/validate.py`` discovery literals
       (``rglob("<reason>.json")`` consumed by
       ``validate_trace_sidecar``).
    4. ``workloads/simbench_bw/validate.py`` schema literals
       (``SCHEMAS_DIR / "<reason>.schema.json"`` inside
       ``validate_trace_sidecar``).

    A drift in any single layer (constant rename, writer literal
    typo, schema rename, etc.) is caught by ``_four_way_parity_diff``
    flagging that layer's exclusive set as non-empty.
    """

    def test_run_suite_writer_literals_match_expected(self):
        emitted = _scan_trace_budget_run_suite_typed_reasons()
        self.assertEqual(
            emitted,
            TRACE_BUDGET_TYPED_REASONS_EXPECTED,
            f"Trace-budget run_suite.py writer literals "
            f"(sidecar = <dir> / \"<reason>.json\") drifted from expected. "
            f"Update TRACE_BUDGET_TYPED_REASONS_EXPECTED in this test AND "
            f"verify the constant in tools/cloud/trace_budgets.py and the "
            f"verifier discovery+schema literals in "
            f"workloads/simbench_bw/validate.py when adding or removing a "
            f"trace-budget typed reason.",
        )

    def test_validator_discovery_literals_match_expected(self):
        emitted = _scan_trace_budget_validator_discovery_reasons()
        self.assertEqual(
            emitted,
            TRACE_BUDGET_TYPED_REASONS_EXPECTED,
            f"Trace-budget validate.py discovery literals "
            f"(rglob(\"<reason>.json\") consumed by validate_trace_sidecar) "
            f"drifted from expected. Update "
            f"TRACE_BUDGET_TYPED_REASONS_EXPECTED in this test AND verify "
            f"the constant in tools/cloud/trace_budgets.py, the writer "
            f"literals in workloads/simbench_bw/run_suite.py, and the "
            f"schema literals in workloads/simbench_bw/validate.py when "
            f"adding or removing a trace-budget typed reason.",
        )

    def test_validator_schema_literals_match_expected(self):
        emitted = _scan_trace_budget_validator_schema_reasons()
        self.assertEqual(
            emitted,
            TRACE_BUDGET_TYPED_REASONS_EXPECTED,
            f"Trace-budget validate.py schema literals "
            f"(SCHEMAS_DIR / \"<reason>.schema.json\" inside "
            f"validate_trace_sidecar) drifted from expected. Update "
            f"TRACE_BUDGET_TYPED_REASONS_EXPECTED in this test AND verify "
            f"the constant in tools/cloud/trace_budgets.py, the writer "
            f"literals in workloads/simbench_bw/run_suite.py, and the "
            f"discovery literals in workloads/simbench_bw/validate.py "
            f"when adding or removing a trace-budget typed reason.",
        )

    def test_all_four_live_layers_identical(self):
        constants = _scan_trace_budget_typed_reasons()
        writer = _scan_trace_budget_run_suite_typed_reasons()
        discovery = _scan_trace_budget_validator_discovery_reasons()
        schema = _scan_trace_budget_validator_schema_reasons()
        diff = _four_way_parity_diff(constants, writer, discovery, schema)
        self.assertEqual(
            diff["constants_only"],
            frozenset(),
            f"Trace-budget typed reasons present only in trace_budgets.py "
            f"constants but missing from writer/discovery/schema layers: "
            f"{sorted(diff['constants_only'])}. Each constant must have a "
            f"matching writer literal in run_suite.py AND a matching "
            f"discovery+schema literal in validate.py.",
        )
        self.assertEqual(
            diff["writer_only"],
            frozenset(),
            f"Trace-budget typed reasons present only in run_suite.py "
            f"writer literals but missing from constants/discovery/schema: "
            f"{sorted(diff['writer_only'])}. Add the constant in "
            f"trace_budgets.py and the matching discovery+schema literals "
            f"in validate.py.",
        )
        self.assertEqual(
            diff["discovery_only"],
            frozenset(),
            f"Trace-budget typed reasons present only in validate.py "
            f"discovery literals but missing from constants/writer/schema: "
            f"{sorted(diff['discovery_only'])}. Add the constant in "
            f"trace_budgets.py, the writer literal in run_suite.py, and "
            f"the matching schema literal in validate.py.",
        )
        self.assertEqual(
            diff["schema_only"],
            frozenset(),
            f"Trace-budget typed reasons present only in validate.py "
            f"schema literals but missing from constants/writer/discovery: "
            f"{sorted(diff['schema_only'])}. Add the constant in "
            f"trace_budgets.py, the writer literal in run_suite.py, and "
            f"the matching discovery literal in validate.py.",
        )
        self.assertEqual(
            diff["all_layers"],
            TRACE_BUDGET_TYPED_REASONS_EXPECTED,
            f"Trace-budget typed reasons present in all four layers "
            f"({sorted(diff['all_layers'])}) does not equal the expected "
            f"set ({sorted(TRACE_BUDGET_TYPED_REASONS_EXPECTED)}).",
        )

    def test_synthetic_constants_only_drift_caught(self):
        base = TRACE_BUDGET_TYPED_REASONS_EXPECTED
        constants = base | frozenset({"synthetic_drift_constants"})
        writer = base
        discovery = base
        schema = base
        diff = _four_way_parity_diff(constants, writer, discovery, schema)
        self.assertEqual(diff["constants_only"], frozenset({"synthetic_drift_constants"}))
        self.assertEqual(diff["writer_only"], frozenset())
        self.assertEqual(diff["discovery_only"], frozenset())
        self.assertEqual(diff["schema_only"], frozenset())
        self.assertEqual(diff["all_layers"], base)

    def test_synthetic_writer_only_drift_caught(self):
        base = TRACE_BUDGET_TYPED_REASONS_EXPECTED
        constants = base
        writer = base | frozenset({"synthetic_drift_writer"})
        discovery = base
        schema = base
        diff = _four_way_parity_diff(constants, writer, discovery, schema)
        self.assertEqual(diff["writer_only"], frozenset({"synthetic_drift_writer"}))
        self.assertEqual(diff["constants_only"], frozenset())
        self.assertEqual(diff["discovery_only"], frozenset())
        self.assertEqual(diff["schema_only"], frozenset())
        self.assertEqual(diff["all_layers"], base)

    def test_synthetic_discovery_only_drift_caught(self):
        base = TRACE_BUDGET_TYPED_REASONS_EXPECTED
        constants = base
        writer = base
        discovery = base | frozenset({"synthetic_drift_discovery"})
        schema = base
        diff = _four_way_parity_diff(constants, writer, discovery, schema)
        self.assertEqual(diff["discovery_only"], frozenset({"synthetic_drift_discovery"}))
        self.assertEqual(diff["constants_only"], frozenset())
        self.assertEqual(diff["writer_only"], frozenset())
        self.assertEqual(diff["schema_only"], frozenset())
        self.assertEqual(diff["all_layers"], base)

    def test_synthetic_schema_only_drift_caught(self):
        base = TRACE_BUDGET_TYPED_REASONS_EXPECTED
        constants = base
        writer = base
        discovery = base
        schema = base | frozenset({"synthetic_drift_schema"})
        diff = _four_way_parity_diff(constants, writer, discovery, schema)
        self.assertEqual(diff["schema_only"], frozenset({"synthetic_drift_schema"}))
        self.assertEqual(diff["constants_only"], frozenset())
        self.assertEqual(diff["writer_only"], frozenset())
        self.assertEqual(diff["discovery_only"], frozenset())
        self.assertEqual(diff["all_layers"], base)

    def test_synthetic_all_match_no_exclusive_entries(self):
        base = TRACE_BUDGET_TYPED_REASONS_EXPECTED
        diff = _four_way_parity_diff(base, base, base, base)
        self.assertEqual(diff["constants_only"], frozenset())
        self.assertEqual(diff["writer_only"], frozenset())
        self.assertEqual(diff["discovery_only"], frozenset())
        self.assertEqual(diff["schema_only"], frozenset())
        self.assertEqual(diff["all_layers"], base)


class NvbitSchemaConstParityTest(unittest.TestCase):
    """Schema-content parity tests for the NVBit-tracer typed-reason
    surface.

    Each NVBit typed-reason JSON schema at
    ``workloads/simbench_bw/schemas/nvbit_<reason>.schema.json``
    pins ``properties.typed_reason.const`` to the literal string
    ``<reason>``. The producer writes the typed_reason field with
    that exact slug into each sidecar, and the JSON-Schema validator
    inside ``import_nvbit_exclusion.validate(...)`` rejects sidecars
    whose typed_reason does not equal the schema's const.

    The R14-R21 parity tests verify the schema FILE EXISTS at the
    expected filename and that producer/writer/consumer-admit-list
    agree on the typed-reason set. They do NOT scan the schema
    file's INTERNAL ``const`` value. A drift between the schema's
    pinned const and its filename slug would silently pass: the
    producer would write ``typed_reason: "<filename slug>"`` and
    the validator (using the drifted const) would reject every
    real sidecar, but no parity test would catch the cause.

    Round 22 closes that drift class. Out of scope:

    - Trace-budget schemas (``trace_volume_exceeded.schema.json``)
      have no ``typed_reason`` field at all; their identity is the
      filename + ``$id`` only.
    - Local-labels typed reasons are written into the combined
      ``label_excluded.json`` (no per-reason schema files).
    """

    def test_each_nvbit_schema_const_matches_filename_slug(self):
        for reason in PRODUCER_TYPED_REASONS_EXPECTED:
            schema_path = _SCHEMA_DIR / f"nvbit_{reason}.schema.json"
            const_value = _load_schema_typed_reason_const(schema_path)
            self.assertEqual(
                const_value,
                reason,
                f"NVBit schema {schema_path.name} pins "
                f"properties.typed_reason.const to {const_value!r} but the "
                f"filename slug is {reason!r}. The producer writes "
                f"typed_reason={reason!r} into sidecars; if the schema's "
                f"const disagrees, validation rejects every real sidecar. "
                f"Either rename the file to match the const or update the "
                f"schema's const to match the filename slug.",
            )

    def test_synthetic_const_drift_caught(self):
        synthetic_schema_text = json.dumps(
            {
                "properties": {
                    "typed_reason": {
                        "type": "string",
                        "const": "drifted_const_value",
                    }
                }
            }
        )
        extracted = _extract_schema_typed_reason_const(synthetic_schema_text)
        self.assertEqual(extracted, "drifted_const_value")
        expected_slug = "unsupported_memory_path"
        self.assertNotEqual(
            extracted,
            expected_slug,
            "Synthetic drift fixture: extracted const must not match the "
            "expected slug, proving a parity assertion using this helper "
            "would fail when const drifts.",
        )

    def test_synthetic_missing_typed_reason_property(self):
        synthetic_no_typed_reason = json.dumps(
            {
                "properties": {
                    "other_field": {"type": "string", "const": "foo"}
                }
            }
        )
        self.assertIsNone(
            _extract_schema_typed_reason_const(synthetic_no_typed_reason),
            "Schema without a properties.typed_reason field must return "
            "None so a parity assertion can fail-closed.",
        )
        synthetic_no_const = json.dumps(
            {
                "properties": {
                    "typed_reason": {"type": "string"}
                }
            }
        )
        self.assertIsNone(
            _extract_schema_typed_reason_const(synthetic_no_const),
            "Schema with typed_reason but no const must return None so a "
            "parity assertion can fail-closed.",
        )


class SchemaIdParityTest(unittest.TestCase):
    """Schema-content parity tests for the ``$id`` field of every
    AC-19 typed-reason JSON schema (3 NVBit-tracer + 1 trace-budget).

    Each schema's ``$id`` field is path-prefixed with the schema
    filename (e.g.
    ``"$id": "simbench_bw/schemas/nvbit_unsupported_memory_path.schema.json"``).
    The basename of that field is the same as the file's basename on
    disk. A drift between the two — typically caused by renaming the
    file without updating the ``$id`` (or vice versa) — would
    silently break cross-schema ``$ref`` resolution and any
    documentation tooling that displays the ``$id``.

    R22 closed the schema-internal ``properties.typed_reason.const``
    drift class for the NVBit-tracer surface (the trace-budget
    schema has no ``typed_reason`` field). R23 closes the
    complementary ``$id`` drift class for both surfaces — every
    typed-reason schema has a ``$id``, regardless of whether it
    pins ``typed_reason`` with a ``const``.

    Local-labels typed reasons are written into the combined
    ``label_excluded.json`` (no per-reason schema files), so the
    local-labels surface has no ``$id`` to scan and is out of scope.
    """

    def test_each_nvbit_schema_id_basename_matches_filename(self):
        for reason in PRODUCER_TYPED_REASONS_EXPECTED:
            filename = f"nvbit_{reason}.schema.json"
            schema_path = _SCHEMA_DIR / filename
            id_basename = _load_schema_id_basename(schema_path)
            self.assertEqual(
                id_basename,
                filename,
                f"NVBit schema {filename} has $id basename "
                f"{id_basename!r} but the file is named {filename!r}. "
                f"Either rename the file to match the $id or update the "
                f"schema's $id to match the filename.",
            )

    def test_trace_budget_schema_id_basename_matches_filename(self):
        for reason in TRACE_BUDGET_TYPED_REASONS_EXPECTED:
            filename = f"{reason}.schema.json"
            schema_path = _SCHEMA_DIR / filename
            id_basename = _load_schema_id_basename(schema_path)
            self.assertEqual(
                id_basename,
                filename,
                f"Trace-budget schema {filename} has $id basename "
                f"{id_basename!r} but the file is named {filename!r}. "
                f"Either rename the file to match the $id or update the "
                f"schema's $id to match the filename.",
            )

    def test_synthetic_id_drift_caught(self):
        synthetic_schema_text = json.dumps(
            {
                "$id": "simbench_bw/schemas/drifted_filename.schema.json",
                "properties": {},
            }
        )
        extracted = _extract_schema_id_basename(synthetic_schema_text)
        self.assertEqual(extracted, "drifted_filename.schema.json")
        expected_filename = "nvbit_unsupported_memory_path.schema.json"
        self.assertNotEqual(
            extracted,
            expected_filename,
            "Synthetic drift fixture: extracted basename must not match "
            "the expected filename, proving a parity assertion using "
            "this helper would fail when $id drifts.",
        )

    def test_synthetic_missing_id_property(self):
        synthetic_no_id = json.dumps({"properties": {}})
        self.assertIsNone(
            _extract_schema_id_basename(synthetic_no_id),
            "Schema without a $id field must return None so a parity "
            "assertion can fail-closed.",
        )
        synthetic_non_string_id = json.dumps({"$id": 42, "properties": {}})
        self.assertIsNone(
            _extract_schema_id_basename(synthetic_non_string_id),
            "Schema with non-string $id must return None so a parity "
            "assertion can fail-closed.",
        )


class SchemaOpClassEnumParityTest(unittest.TestCase):
    """Parity tests for the ``excluded_op_class.enum`` field of every
    NVBit typed-reason JSON schema vs
    ``import_nvbit_exclusion.VALID_OP_CLASSES``.

    Each NVBit typed-reason schema constrains the
    ``excluded_op_class`` field to one of a fixed UPPER_SNAKE
    op-class names list. The list is the same data as
    ``VALID_OP_CLASSES`` (the consumer admit list), copied between
    two files.

    R29 closed producer C++ enum ↔ consumer ``VALID_OP_CLASSES``.
    R31 closes consumer ``VALID_OP_CLASSES`` ↔ schema
    ``excluded_op_class.enum``. Combined with R29, the full op-class
    enum chain (producer C++ → consumer Python → schema JSON) is now
    covered by direct parity tests.

    The pre-R31 ``test_every_op_class_admissible`` in
    ``tools/nvbit_tracer/test_import_nvbit_exclusion.py`` indirectly
    catches schema/admit-list drift by running the validator over
    each ``VALID_OP_CLASSES`` value, but only against the
    ``unsupported_memory_path`` schema (the test uses
    ``_valid_unsupported_sidecar()`` as the base shape). The other
    two NVBit schemas (``unmapped_memory_range``,
    ``cross_allocation_lane_span``) are not directly tested for
    op-class enum parity. R31 closes this gap by scanning all 3
    schemas.
    """

    def test_each_nvbit_schema_op_class_enum_matches_valid_op_classes(
        self,
    ):
        expected = frozenset(import_nvbit_exclusion.VALID_OP_CLASSES)
        expected_cardinality = len(import_nvbit_exclusion.VALID_OP_CLASSES)
        for reason in PRODUCER_TYPED_REASONS_EXPECTED:
            filename = f"nvbit_{reason}.schema.json"
            schema_path = _SCHEMA_DIR / filename
            enum_set = _load_schema_excluded_op_class_enum(schema_path)
            self.assertIsNotNone(
                enum_set,
                f"NVBit schema {filename} excluded_op_class enum is "
                f"None: the helper rejected the schema's enum as "
                f"missing, malformed, or containing non-string or "
                f"duplicate members. Inspect the schema directly to "
                f"diagnose.",
            )
            assert enum_set is not None
            self.assertEqual(
                len(enum_set),
                expected_cardinality,
                f"NVBit schema {filename} excluded_op_class enum has "
                f"{len(enum_set)} entries; VALID_OP_CLASSES has "
                f"{expected_cardinality}. Cardinality drift would "
                f"silently allow one side to admit/reject sidecars "
                f"based on a stale list.",
            )
            self.assertEqual(
                enum_set,
                expected,
                f"NVBit schema {filename} excluded_op_class enum "
                f"{sorted(enum_set)} differs from VALID_OP_CLASSES "
                f"{sorted(expected)}. Either update the schema's "
                f"enum to match the admit list or update "
                f"VALID_OP_CLASSES to match the schema. Drift here "
                f"would silently allow the validator to admit/reject "
                f"sidecars based on stale enum lists.",
            )

    def test_synthetic_basic_extraction(self):
        text = json.dumps(
            {
                "properties": {
                    "excluded_op_class": {
                        "type": "string",
                        "enum": ["GLOBAL", "SHARED", "LOCAL"],
                    }
                }
            }
        )
        result = _extract_schema_excluded_op_class_enum(text)
        self.assertEqual(
            result, frozenset({"GLOBAL", "SHARED", "LOCAL"})
        )

    def test_synthetic_missing_excluded_op_class_property(self):
        text = json.dumps(
            {
                "properties": {
                    "other_field": {"type": "string"}
                }
            }
        )
        self.assertIsNone(
            _extract_schema_excluded_op_class_enum(text),
            "Schema without an excluded_op_class field must return "
            "None so a parity assertion can fail-closed.",
        )

    def test_synthetic_missing_enum_list(self):
        text = json.dumps(
            {
                "properties": {
                    "excluded_op_class": {"type": "string"}
                }
            }
        )
        self.assertIsNone(
            _extract_schema_excluded_op_class_enum(text),
            "Schema with excluded_op_class but no enum key must "
            "return None so a parity assertion can fail-closed.",
        )

    def test_synthetic_drift_extra_in_schema(self):
        text = json.dumps(
            {
                "properties": {
                    "excluded_op_class": {
                        "type": "string",
                        "enum": [
                            "GLOBAL",
                            "SHARED",
                            "LOCAL",
                            "FUTURE_CLASS",
                        ],
                    }
                }
            }
        )
        result = _extract_schema_excluded_op_class_enum(text)
        assert result is not None
        self.assertIn("FUTURE_CLASS", result)
        self.assertEqual(
            result,
            frozenset({"GLOBAL", "SHARED", "LOCAL", "FUTURE_CLASS"}),
        )

    def test_synthetic_drift_missing_in_schema(self):
        text = json.dumps(
            {
                "properties": {
                    "excluded_op_class": {
                        "type": "string",
                        "enum": ["GLOBAL", "SHARED"],
                    }
                }
            }
        )
        result = _extract_schema_excluded_op_class_enum(text)
        assert result is not None
        self.assertNotIn("LOCAL", result)
        self.assertEqual(result, frozenset({"GLOBAL", "SHARED"}))

    def test_synthetic_non_string_enum_member_returns_none(self):
        text = json.dumps(
            {
                "properties": {
                    "excluded_op_class": {
                        "type": "string",
                        "enum": ["GLOBAL", 42, "SHARED"],
                    }
                }
            }
        )
        self.assertIsNone(
            _extract_schema_excluded_op_class_enum(text),
            "An enum list containing a non-string member (e.g. an "
            "integer) must NOT be silently filtered; the helper "
            "must return None so a parity assertion fails closed.",
        )

    def test_synthetic_null_enum_member_returns_none(self):
        text = json.dumps(
            {
                "properties": {
                    "excluded_op_class": {
                        "type": "string",
                        "enum": ["GLOBAL", None, "SHARED"],
                    }
                }
            }
        )
        self.assertIsNone(
            _extract_schema_excluded_op_class_enum(text),
            "An enum list containing a null literal must NOT be "
            "silently filtered; the helper must return None so a "
            "parity assertion fails closed.",
        )

    def test_synthetic_duplicate_enum_member_returns_none(self):
        text = json.dumps(
            {
                "properties": {
                    "excluded_op_class": {
                        "type": "string",
                        "enum": ["GLOBAL", "GLOBAL", "SHARED"],
                    }
                }
            }
        )
        self.assertIsNone(
            _extract_schema_excluded_op_class_enum(text),
            "An enum list containing duplicate string members must "
            "NOT be silently collapsed by frozenset construction; "
            "the helper must return None so a parity assertion "
            "fails closed.",
        )


class SectorStreamTypedReasonsParityTest(unittest.TestCase):
    """Lockstep enforcer for the sector-stream taxonomy family.

    The family has two surfaces:

    1. **Producer-side constant**: the
       ``SECTOR_STREAM_TYPED_REASONS`` frozenset in
       ``tools/sector_stream/generator.py``. The generator returns
       one of these values in the ``typed_reason`` field of every
       rejection record.

    2. **JSON schema enum**: the ``typed_reason.enum`` list on the
       typed-exclusion branch of
       ``workloads/simbench_bw/schemas/sector_stream.schema.json``.
       The schema's ``oneOf`` contains two top-level branches; the
       second branch carries the ``typed_reason`` property whose
       ``enum`` declares the admitted typed-reason vocabulary.

    Both surfaces must enumerate exactly
    ``SECTOR_STREAM_TYPED_REASONS_EXPECTED``. Drift between them
    means a record produced under one set of typed reasons cannot
    be admitted by a validator that uses the other set, which is
    the C-AC-C lockstep failure mode.
    """

    @staticmethod
    def _read_producer_constant() -> frozenset[str]:
        """Parse ``SECTOR_STREAM_TYPED_REASONS`` from the generator
        module.

        We parse the source AST directly rather than importing the
        module so this test does not depend on the
        ``tools/sector_stream`` directory being on the Python
        path. The parity check reads the literal set elements out
        of the assigned ``frozenset({"...", ...})`` expression so a
        future maintainer who renames the constant or wraps it in
        a different container fails the test loudly.
        """
        generator_path = (
            _REPO_ROOT
            / "tools"
            / "sector_stream"
            / "generator.py"
        )
        text = generator_path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            # The constant may be assigned with or without an
            # annotation (`X = frozenset({...})` vs.
            # `X: frozenset[str] = frozenset({...})`); both are
            # accepted.
            if isinstance(node, ast.AnnAssign):
                target = node.target
                value = node.value
                if not isinstance(target, ast.Name):
                    continue
                if target.id != "SECTOR_STREAM_TYPED_REASONS":
                    continue
                if value is None:
                    continue
            elif isinstance(node, ast.Assign):
                if len(node.targets) != 1:
                    continue
                target = node.targets[0]
                if not isinstance(target, ast.Name):
                    continue
                if target.id != "SECTOR_STREAM_TYPED_REASONS":
                    continue
                value = node.value
            else:
                continue
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "frozenset"
                and len(value.args) == 1
                and isinstance(value.args[0], ast.Set)
            ):
                elements: set[str] = set()
                for elt in value.args[0].elts:
                    if isinstance(elt, ast.Constant) and isinstance(
                        elt.value, str
                    ):
                        elements.add(elt.value)
                    else:
                        raise AssertionError(
                            "Non-string constant in "
                            "SECTOR_STREAM_TYPED_REASONS literal"
                        )
                return frozenset(elements)
            raise AssertionError(
                "SECTOR_STREAM_TYPED_REASONS must be assigned a "
                "frozenset({...}) literal so the parity "
                "scanner can read it without executing the "
                "module."
            )
        raise AssertionError(
            "SECTOR_STREAM_TYPED_REASONS not found in "
            "tools/sector_stream/generator.py"
        )

    @staticmethod
    def _read_schema_enum() -> frozenset[str]:
        schema_path = (
            _SCHEMA_DIR / "sector_stream.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        # The schema has oneOf with two branches; the typed-exclusion
        # branch is the one whose result_kind const is
        # "typed_exclusion". Walk the oneOf to find it rather than
        # indexing positionally so a future schema reorder does not
        # silently break the parity scan.
        for branch in schema["oneOf"]:
            properties = branch.get("properties", {})
            kind_spec = properties.get("result_kind", {})
            if kind_spec.get("const") == "typed_exclusion":
                reason_spec = properties.get("typed_reason", {})
                enum = reason_spec.get("enum")
                if enum is None:
                    raise AssertionError(
                        "typed_exclusion branch is missing "
                        "typed_reason.enum"
                    )
                return frozenset(enum)
        raise AssertionError(
            "sector_stream.schema.json oneOf has no "
            "typed_exclusion branch"
        )

    def test_producer_constant_matches_expected(self) -> None:
        producer = self._read_producer_constant()
        self.assertEqual(
            producer,
            SECTOR_STREAM_TYPED_REASONS_EXPECTED,
            msg=(
                "tools/sector_stream/generator.py "
                "SECTOR_STREAM_TYPED_REASONS drifted from "
                "SECTOR_STREAM_TYPED_REASONS_EXPECTED. Update one "
                "or both to match."
            ),
        )

    def test_schema_enum_matches_expected(self) -> None:
        schema_enum = self._read_schema_enum()
        self.assertEqual(
            schema_enum,
            SECTOR_STREAM_TYPED_REASONS_EXPECTED,
            msg=(
                "workloads/simbench_bw/schemas/sector_stream.schema.json "
                "typed_reason.enum drifted from "
                "SECTOR_STREAM_TYPED_REASONS_EXPECTED. Update one "
                "or both to match."
            ),
        )

    def test_producer_and_schema_agree(self) -> None:
        producer = self._read_producer_constant()
        schema_enum = self._read_schema_enum()
        self.assertEqual(
            producer,
            schema_enum,
            msg=(
                "Producer and schema typed-reason sets diverged. "
                "Sector-stream taxonomy lockstep at the family's "
                "two surfaces failed."
            ),
        )

    def test_round_added_documents_new_typed_reasons(self) -> None:
        # The two new typed reasons (data_width_invalid,
        # sector_stream_class_unsupported) must appear in
        # ROUND_ADDED_TYPED_REASONS so test_implemented_typed_reasons
        # _are_documented does not flag them.
        new_reasons = {
            "data_width_invalid",
            "sector_stream_class_unsupported",
        }
        self.assertTrue(
            new_reasons.issubset(ROUND_ADDED_TYPED_REASONS),
            msg=(
                "ROUND_ADDED_TYPED_REASONS must document the new "
                "C-AC-F typed reasons "
                f"{sorted(new_reasons - ROUND_ADDED_TYPED_REASONS)}."
            ),
        )

    def test_pre_existing_typed_reasons_in_ac19(self) -> None:
        # The two pre-existing reasons (no_lanes_active,
        # lane_address_length_invalid) live in the original-plan
        # AC-19 enumeration; ROUND_ADDED would be the wrong bucket
        # for them. This test pins that fact so a future cleanup
        # does not move them into ROUND_ADDED out of laziness.
        pre_existing = {
            "no_lanes_active",
            "lane_address_length_invalid",
        }
        self.assertTrue(
            pre_existing.issubset(AC19_PLAN_ENUMERATED_TYPED_REASONS),
            msg=(
                "Pre-existing sector-stream reasons must remain in "
                "AC19_PLAN_ENUMERATED_TYPED_REASONS: "
                f"{sorted(pre_existing - AC19_PLAN_ENUMERATED_TYPED_REASONS)}"
            ),
        )


class DivergenceProxiesTypedReasonsParityTest(unittest.TestCase):
    """Lockstep enforcer for the divergence-proxies taxonomy family.

    The family has two surfaces:

    1. **Producer-side constant**: the
       ``DIVERGENCE_PROXIES_TYPED_REASONS`` frozenset in
       ``tools/divergence_proxies/proxies.py``. The proxy module
       returns one of these values in the ``typed_reason`` field
       of any typed-unavailable proxy record.

    2. **JSON schema enum**: the ``typed_reason.enum`` list on the
       typed-unavailable branch of
       ``workloads/simbench_bw/schemas/divergence_proxies.schema.json``.
       The schema's ``$defs/proxy_or_unavailable.oneOf`` contains
       two branches; the second branch carries the
       ``typed_reason`` property whose ``enum`` declares the
       admitted typed-reason vocabulary.

    Both surfaces must enumerate exactly
    ``DIVERGENCE_PROXIES_TYPED_REASONS_EXPECTED``. Drift between
    them means a record produced under one set of typed reasons
    cannot be admitted by a validator that uses the other set,
    which is the C-AC-C lockstep failure mode.
    """

    @staticmethod
    def _read_producer_constant() -> frozenset[str]:
        """Parse ``DIVERGENCE_PROXIES_TYPED_REASONS`` from the proxies
        module via AST so the parity check does not depend on the
        ``tools/divergence_proxies`` directory being importable.
        """
        proxies_path = (
            _REPO_ROOT
            / "tools"
            / "divergence_proxies"
            / "proxies.py"
        )
        text = proxies_path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign):
                target = node.target
                value = node.value
                if not isinstance(target, ast.Name):
                    continue
                if target.id != "DIVERGENCE_PROXIES_TYPED_REASONS":
                    continue
                if value is None:
                    continue
            elif isinstance(node, ast.Assign):
                if len(node.targets) != 1:
                    continue
                target = node.targets[0]
                if not isinstance(target, ast.Name):
                    continue
                if target.id != "DIVERGENCE_PROXIES_TYPED_REASONS":
                    continue
                value = node.value
            else:
                continue
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "frozenset"
                and len(value.args) == 1
                and isinstance(value.args[0], ast.Set)
            ):
                elements: set[str] = set()
                for elt in value.args[0].elts:
                    if isinstance(elt, ast.Constant) and isinstance(
                        elt.value, str
                    ):
                        elements.add(elt.value)
                    else:
                        raise AssertionError(
                            "Non-string constant in "
                            "DIVERGENCE_PROXIES_TYPED_REASONS literal"
                        )
                return frozenset(elements)
            raise AssertionError(
                "DIVERGENCE_PROXIES_TYPED_REASONS must be assigned a "
                "frozenset({...}) literal so the parity scanner can "
                "read it without executing the module."
            )
        raise AssertionError(
            "DIVERGENCE_PROXIES_TYPED_REASONS not found in "
            "tools/divergence_proxies/proxies.py"
        )

    @staticmethod
    def _read_schema_enum() -> frozenset[str]:
        schema_path = (
            _SCHEMA_DIR / "divergence_proxies.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        # The schema declares per-proxy slots via $ref to
        # `#/$defs/proxy_or_unavailable`. The typed-unavailable enum
        # is in the second oneOf branch under $defs. Walk the oneOf
        # to find the branch whose `result_kind` const is
        # `typed_unavailable` rather than indexing positionally so a
        # future schema reorder does not silently break the parity
        # scan.
        defs = schema.get("$defs", {})
        proxy_def = defs.get("proxy_or_unavailable")
        if proxy_def is None or "oneOf" not in proxy_def:
            raise AssertionError(
                "divergence_proxies.schema.json missing "
                "$defs.proxy_or_unavailable.oneOf"
            )
        for branch in proxy_def["oneOf"]:
            properties = branch.get("properties", {})
            kind_spec = properties.get("result_kind", {})
            if kind_spec.get("const") == "typed_unavailable":
                reason_spec = properties.get("typed_reason", {})
                enum = reason_spec.get("enum")
                if enum is None:
                    raise AssertionError(
                        "typed_unavailable branch is missing "
                        "typed_reason.enum"
                    )
                return frozenset(enum)
        raise AssertionError(
            "divergence_proxies.schema.json $defs.proxy_or_unavailable."
            "oneOf has no typed_unavailable branch"
        )

    def test_producer_constant_matches_expected(self) -> None:
        producer = self._read_producer_constant()
        self.assertEqual(
            producer,
            DIVERGENCE_PROXIES_TYPED_REASONS_EXPECTED,
            msg=(
                "tools/divergence_proxies/proxies.py "
                "DIVERGENCE_PROXIES_TYPED_REASONS drifted from "
                "DIVERGENCE_PROXIES_TYPED_REASONS_EXPECTED. Update "
                "one or both to match."
            ),
        )

    def test_schema_enum_matches_expected(self) -> None:
        schema_enum = self._read_schema_enum()
        self.assertEqual(
            schema_enum,
            DIVERGENCE_PROXIES_TYPED_REASONS_EXPECTED,
            msg=(
                "workloads/simbench_bw/schemas/"
                "divergence_proxies.schema.json typed_reason.enum "
                "drifted from "
                "DIVERGENCE_PROXIES_TYPED_REASONS_EXPECTED. Update "
                "one or both to match."
            ),
        )

    def test_producer_and_schema_agree(self) -> None:
        producer = self._read_producer_constant()
        schema_enum = self._read_schema_enum()
        self.assertEqual(
            producer,
            schema_enum,
            msg=(
                "Producer and schema typed-reason sets diverged. "
                "Divergence-proxies taxonomy lockstep at the family's "
                "two surfaces failed."
            ),
        )

    def test_round_added_documents_new_typed_reasons(self) -> None:
        # Seven of the eight reasons are round-added in Loop 2 R8
        # under the C-AC-C lockstep discipline. The one exception
        # (`outstanding_load_density_unavailable_no_phase_records`)
        # is in the original-plan AC-19 enumeration and stays in
        # AC19_PLAN_ENUMERATED_TYPED_REASONS.
        new_reasons = {
            "stride_entropy_unavailable_no_lane_table",
            "throughput_saturation_slope_unavailable_no_sweep",
            "pc_repeated_miss_likely_streams_unavailable_no_pc_data",
            "md_loads_per_kilo_inst_unavailable_no_instruction_count",
            "unique_sectors_per_warp_inst_unavailable_no_records",
            "unique_lines_per_warp_inst_unavailable_no_records",
            "sectors_per_request_unavailable_no_records",
        }
        self.assertTrue(
            new_reasons.issubset(ROUND_ADDED_TYPED_REASONS),
            msg=(
                "ROUND_ADDED_TYPED_REASONS must document the new "
                "C-AC-G typed-unavailable reasons "
                f"{sorted(new_reasons - ROUND_ADDED_TYPED_REASONS)}."
            ),
        )

    def test_pre_existing_typed_reasons_in_ac19(self) -> None:
        # The single pre-existing reason
        # (`outstanding_load_density_unavailable_no_phase_records`)
        # lives in the original-plan AC-19 enumeration; ROUND_ADDED
        # would be the wrong bucket for it. This test pins that
        # fact so a future cleanup does not move it into ROUND_ADDED
        # out of laziness.
        pre_existing = {
            "outstanding_load_density_unavailable_no_phase_records",
        }
        self.assertTrue(
            pre_existing.issubset(AC19_PLAN_ENUMERATED_TYPED_REASONS),
            msg=(
                "Pre-existing divergence-proxies reasons must remain "
                "in AC19_PLAN_ENUMERATED_TYPED_REASONS: "
                f"{sorted(pre_existing - AC19_PLAN_ENUMERATED_TYPED_REASONS)}"
            ),
        )


class HrdProfileTypedReasonsParityTest(unittest.TestCase):
    """Lockstep enforcer for the hrd_profile taxonomy family
    (Loop 2 R17 / task11 / AC-6).

    The family has two surfaces:

    1. **Producer-side constant**: the
       ``HRD_PROFILE_TYPED_REASONS`` frozenset in
       ``tools/hrd_profiler/profiler.py``. The profiler returns
       one of these values in the ``typed_reason`` field of any
       typed-unavailable bin or evidence record.

    2. **JSON schema enum**: the ``$defs.typed_reason.enum``
       list in
       ``workloads/simbench_bw/schemas/hrd_profile.schema.json``.

    Both surfaces must enumerate exactly
    ``HRD_PROFILE_TYPED_REASONS_EXPECTED``. Drift between them
    means a record produced under one set of typed reasons
    cannot be admitted by a validator that uses the other set.
    """

    @staticmethod
    def _read_producer_constant() -> frozenset[str]:
        profiler_path = (
            _REPO_ROOT
            / "tools"
            / "hrd_profiler"
            / "profiler.py"
        )
        text = profiler_path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign):
                target = node.target
                value = node.value
                if not isinstance(target, ast.Name):
                    continue
                if target.id != "HRD_PROFILE_TYPED_REASONS":
                    continue
                if value is None:
                    continue
            elif isinstance(node, ast.Assign):
                if len(node.targets) != 1:
                    continue
                target = node.targets[0]
                if not isinstance(target, ast.Name):
                    continue
                if target.id != "HRD_PROFILE_TYPED_REASONS":
                    continue
                value = node.value
            else:
                continue
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "frozenset"
                and len(value.args) == 1
                and isinstance(value.args[0], ast.Set)
            ):
                elements: set[str] = set()
                for elt in value.args[0].elts:
                    if isinstance(elt, ast.Constant) and isinstance(
                        elt.value, str
                    ):
                        elements.add(elt.value)
                    else:
                        raise AssertionError(
                            "Non-string constant in "
                            "HRD_PROFILE_TYPED_REASONS literal"
                        )
                return frozenset(elements)
            raise AssertionError(
                "HRD_PROFILE_TYPED_REASONS must be assigned a "
                "frozenset({...}) literal so the parity scanner can "
                "read it without executing the module."
            )
        raise AssertionError(
            "HRD_PROFILE_TYPED_REASONS not found in "
            "tools/hrd_profiler/profiler.py"
        )

    @staticmethod
    def _read_schema_enum() -> frozenset[str]:
        schema_path = _SCHEMA_DIR / "hrd_profile.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        defs = schema.get("$defs", {})
        typed_reason_def = defs.get("typed_reason")
        if typed_reason_def is None or "enum" not in typed_reason_def:
            raise AssertionError(
                "hrd_profile.schema.json missing $defs.typed_reason.enum"
            )
        return frozenset(typed_reason_def["enum"])

    def test_producer_constant_matches_expected(self) -> None:
        producer = self._read_producer_constant()
        self.assertEqual(
            producer,
            HRD_PROFILE_TYPED_REASONS_EXPECTED,
            msg=(
                "tools/hrd_profiler/profiler.py "
                "HRD_PROFILE_TYPED_REASONS drifted from "
                "HRD_PROFILE_TYPED_REASONS_EXPECTED."
            ),
        )

    def test_schema_enum_matches_expected(self) -> None:
        schema_enum = self._read_schema_enum()
        self.assertEqual(
            schema_enum,
            HRD_PROFILE_TYPED_REASONS_EXPECTED,
            msg=(
                "workloads/simbench_bw/schemas/hrd_profile.schema.json "
                "$defs.typed_reason.enum drifted from "
                "HRD_PROFILE_TYPED_REASONS_EXPECTED."
            ),
        )

    def test_producer_and_schema_agree(self) -> None:
        self.assertEqual(
            self._read_producer_constant(),
            self._read_schema_enum(),
            msg=(
                "Producer and schema typed-reason sets diverged. "
                "HRD-profile taxonomy lockstep failed."
            ),
        )

    def test_round_added_documents_new_typed_reasons(self) -> None:
        # Derive the round-added subset from the authoritative
        # producer-side expected set rather than hardcoding it. The
        # three plan-enumerated HRD reasons live in the original-plan
        # AC-19 enumeration; everything else in
        # HRD_PROFILE_TYPED_REASONS_EXPECTED is round-added and must
        # be documented in ROUND_ADDED_TYPED_REASONS. This derivation
        # automatically admits future producer-side additions (e.g., a
        # new cache-model classification reason) without further test
        # changes.
        plan_enumerated = {
            "cache_model_capacity_missing",
            "cache_model_capacity_inconsistent",
            "cache_model_underconfigured",
        }
        round_added_subset = (
            HRD_PROFILE_TYPED_REASONS_EXPECTED - plan_enumerated
        )
        self.assertTrue(
            round_added_subset.issubset(ROUND_ADDED_TYPED_REASONS),
            msg=(
                "ROUND_ADDED_TYPED_REASONS must document every "
                "non-plan-enumerated HRD typed reason. Missing: "
                f"{sorted(round_added_subset - ROUND_ADDED_TYPED_REASONS)}."
            ),
        )

    def test_pre_existing_typed_reasons_in_ac19(self) -> None:
        # Three pre-existing reasons live in the original-plan
        # AC-19 enumeration.
        pre_existing = {
            "cache_model_capacity_missing",
            "cache_model_capacity_inconsistent",
            "cache_model_underconfigured",
        }
        self.assertTrue(
            pre_existing.issubset(AC19_PLAN_ENUMERATED_TYPED_REASONS),
            msg=(
                "Pre-existing HRD-profile reasons must remain in "
                "AC19_PLAN_ENUMERATED_TYPED_REASONS: "
                f"{sorted(pre_existing - AC19_PLAN_ENUMERATED_TYPED_REASONS)}"
            ),
        )


class TaxonomyFamilySurfaceInventoryTest(unittest.TestCase):
    """C-AC-C: per-taxonomy-family surface-inventory map.

    The top-level `_TAXONOMY_FAMILIES` constant declares which
    surfaces each typed-reason taxonomy family has and which test
    classes drive the lockstep at each surface. New typed reasons
    added to a family (e.g. `nvbit_slowdown_budget_exceeded` +
    `nvbit_drop_rate_exceeded` joining `trace_budget` in C-AC-H)
    must appear at every surface declared for the family; the
    existing per-surface test classes named in each
    `enforced_by_test_classes` tuple drive the lockstep.

    These tests verify the structure of the inventory itself: that
    every documented family is present, that surface counts match
    the C-AC-C plan literals, that surface IDs are unique within a
    family, and that each family's `expected_typed_reasons`
    reference resolves to the same constant the per-surface
    lockstep tests use.
    """

    def test_known_families_present(self) -> None:
        # The four families we currently track: op-class chain (5),
        # nvbit-exclusion typed reasons (4), trace-budget (4), local
        # labels (1). When a new family is introduced, both this set
        # and `_EXPECTED_SURFACE_COUNT` get updated in lockstep.
        self.assertEqual(
            set(_TAXONOMY_FAMILIES.keys()),
            set(_EXPECTED_SURFACE_COUNT.keys()),
            msg=(
                "_TAXONOMY_FAMILIES and _EXPECTED_SURFACE_COUNT must "
                "agree on the family ID set; one drifted from the "
                "other."
            ),
        )

    def test_each_family_has_expected_surface_count(self) -> None:
        for family_id, expected_n in _EXPECTED_SURFACE_COUNT.items():
            with self.subTest(family=family_id):
                family = _TAXONOMY_FAMILIES[family_id]
                self.assertEqual(
                    len(family.surfaces),
                    expected_n,
                    msg=(
                        f"Family {family_id!r} declares "
                        f"{len(family.surfaces)} surfaces, but "
                        f"_EXPECTED_SURFACE_COUNT pins it at "
                        f"{expected_n}. Either update the pinned "
                        f"count (deliberate widening of the family) "
                        f"or restore the expected surface set."
                    ),
                )

    def test_each_family_surface_id_is_unique_within_family(self) -> None:
        for family_id, family in _TAXONOMY_FAMILIES.items():
            with self.subTest(family=family_id):
                ids = [s.surface_id for s in family.surfaces]
                self.assertEqual(
                    len(ids),
                    len(set(ids)),
                    msg=(
                        f"Family {family_id!r} has duplicate surface "
                        f"IDs: {ids}. Each surface_id must name a "
                        f"distinct enforcement surface."
                    ),
                )

    def test_each_family_surface_has_nonempty_description(self) -> None:
        for family_id, family in _TAXONOMY_FAMILIES.items():
            for surface in family.surfaces:
                with self.subTest(
                    family=family_id, surface=surface.surface_id
                ):
                    self.assertTrue(
                        surface.description.strip(),
                        msg=(
                            f"Family {family_id!r} surface "
                            f"{surface.surface_id!r} has empty "
                            f"description; future maintainers need "
                            f"the file-path / function name to "
                            f"navigate from the inventory back to "
                            f"the live code."
                        ),
                    )

    def test_each_family_surface_names_at_least_one_enforcer_class(self) -> None:
        existing_class_names = {
            "ExclusionTaxonomyParityTest",
            "AggregateScannerCallFormTest",
            "AggregateScannerCommentAwarenessTest",
            "AggregateScannerRawStringTest",
            "LocalLabelsTypedReasonsParityTest",
            "PlanTextParityTest",
            "MemoryOpClassParityTest",
            "ProtoMemoryOpClassEnumParityTest",
            "StaticAssertNumericMappingParityTest",
            "LocalLabelsScannerCommentAwarenessTest",
            "TraceBudgetScannerCommentAwarenessTest",
            "TraceBudgetSchemaScannerCompatTest",
            "TraceBudgetConstantsScannerCommentAwarenessTest",
            "TraceBudgetTypedReasonsParityTest",
            "TraceBudgetLiveLayerParityTest",
            "NvbitSchemaConstParityTest",
            "SchemaIdParityTest",
            "SchemaOpClassEnumParityTest",
            "SectorStreamTypedReasonsParityTest",
            "DivergenceProxiesTypedReasonsParityTest",
            "HrdProfileTypedReasonsParityTest",
            "SharingLocalityTypedReasonsParityTest",
            "OrderProjectionsTypedReasonsParityTest",
            "PerturbationRunLedgerTypedReasonsParityTest",
            "TraceConvergenceTypedReasonsParityTest",
        }
        for family_id, family in _TAXONOMY_FAMILIES.items():
            for surface in family.surfaces:
                with self.subTest(
                    family=family_id, surface=surface.surface_id
                ):
                    self.assertGreaterEqual(
                        len(surface.enforced_by_test_classes), 1,
                        msg=(
                            f"Family {family_id!r} surface "
                            f"{surface.surface_id!r} names no "
                            f"enforcer test class; the inventory "
                            f"loses its connection to the live "
                            f"asserter."
                        ),
                    )
                    for enforcer in surface.enforced_by_test_classes:
                        self.assertIn(
                            enforcer, existing_class_names,
                            msg=(
                                f"Family {family_id!r} surface "
                                f"{surface.surface_id!r} names "
                                f"{enforcer!r} as an enforcer, but "
                                f"that class is not present in this "
                                f"file. Either the class was "
                                f"removed (update the inventory) or "
                                f"the name is misspelled."
                            ),
                        )

    def test_known_kinds_only(self) -> None:
        known_kinds = {"op_class_value", "typed_reason"}
        for family_id, family in _TAXONOMY_FAMILIES.items():
            with self.subTest(family=family_id):
                self.assertIn(
                    family.kind,
                    known_kinds,
                    msg=(
                        f"Family {family_id!r} has unknown kind "
                        f"{family.kind!r}. Add it to known_kinds in "
                        f"this test if a new kind is genuinely "
                        f"warranted."
                    ),
                )

    def test_op_class_family_expected_set_resolves_to_op_class_constant(self) -> None:
        family = _TAXONOMY_FAMILIES["memory_op_class_enum"]
        self.assertEqual(
            family.expected_typed_reasons,
            _OP_CLASS_ENUM_EXPECTED,
        )

    def test_nvbit_exclusion_family_expected_set_resolves_to_producer_constant(self) -> None:
        family = _TAXONOMY_FAMILIES["nvbit_exclusion_typed_reason"]
        self.assertEqual(
            family.expected_typed_reasons,
            PRODUCER_TYPED_REASONS_EXPECTED,
        )

    def test_trace_budget_family_expected_set_resolves_to_trace_budget_constant(self) -> None:
        family = _TAXONOMY_FAMILIES["trace_budget"]
        self.assertEqual(
            family.expected_typed_reasons,
            TRACE_BUDGET_TYPED_REASONS_EXPECTED,
        )

    def test_local_labels_family_expected_set_resolves_to_local_labels_constant(self) -> None:
        family = _TAXONOMY_FAMILIES["local_labels"]
        self.assertEqual(
            family.expected_typed_reasons,
            LOCAL_LABELS_TYPED_REASONS_EXPECTED,
        )

    def test_sector_stream_family_expected_set_resolves_to_sector_stream_constant(self) -> None:
        family = _TAXONOMY_FAMILIES["sector_stream"]
        self.assertEqual(
            family.expected_typed_reasons,
            SECTOR_STREAM_TYPED_REASONS_EXPECTED,
        )

    def test_trace_convergence_family_expected_set_resolves_to_trace_convergence_constant(self) -> None:
        family = _TAXONOMY_FAMILIES["trace_convergence"]
        self.assertEqual(
            family.expected_typed_reasons,
            TRACE_CONVERGENCE_TYPED_REASONS_EXPECTED,
        )

    def test_op_class_family_admit_list_matches_consumer(self) -> None:
        # Sanity: the inventory's expected set for the op-class chain
        # equals the consumer-side `VALID_OP_CLASSES` admit list. If a
        # consumer-side widening lands without updating the inventory,
        # this test catches it.
        consumer = frozenset(import_nvbit_exclusion.VALID_OP_CLASSES)
        family = _TAXONOMY_FAMILIES["memory_op_class_enum"]
        self.assertEqual(
            family.expected_typed_reasons,
            consumer,
            msg=(
                "Op-class family inventory drift: "
                f"_OP_CLASS_ENUM_EXPECTED={sorted(family.expected_typed_reasons)}"
                f" but VALID_OP_CLASSES={sorted(consumer)}."
            ),
        )

    def test_nvbit_exclusion_family_admit_list_matches_consumer(self) -> None:
        # Sanity: the inventory's nvbit-exclusion expected set equals
        # the consumer-side `_SCHEMA_FILES` admit list keys.
        consumer = frozenset(import_nvbit_exclusion._SCHEMA_FILES.keys())
        family = _TAXONOMY_FAMILIES["nvbit_exclusion_typed_reason"]
        self.assertEqual(
            family.expected_typed_reasons,
            consumer,
            msg=(
                "NVBit-exclusion inventory drift: "
                f"PRODUCER_TYPED_REASONS_EXPECTED={sorted(family.expected_typed_reasons)}"
                f" but _SCHEMA_FILES={sorted(consumer)}."
            ),
        )


class TaxonomyFamilyInventoryNegativeControlTest(unittest.TestCase):
    """C-AC-C negative-control: synthetic mismatched inventory entries
    must fail the structural validators above. Each test constructs an
    ad-hoc family entry and runs the same shape checks the
    TaxonomyFamilySurfaceInventoryTest applies, asserting the bad
    entry is rejected. These tests would have caught a drift in the
    real `_TAXONOMY_FAMILIES` constant before it landed in CI.
    """

    @staticmethod
    def _check_surface_count(
        family: TaxonomyFamilyInventory, expected: int
    ) -> bool:
        return len(family.surfaces) == expected

    @staticmethod
    def _check_unique_surface_ids(
        family: TaxonomyFamilyInventory,
    ) -> bool:
        ids = [s.surface_id for s in family.surfaces]
        return len(ids) == len(set(ids))

    @staticmethod
    def _check_nonempty_descriptions(
        family: TaxonomyFamilyInventory,
    ) -> bool:
        return all(s.description.strip() for s in family.surfaces)

    def test_short_family_fails_count_check(self) -> None:
        # Synthesize a family with only 3 surfaces; check rejects it
        # against the 4-surface trace-budget pin.
        bad = TaxonomyFamilyInventory(
            family_id="synthetic_short",
            description="too short",
            kind="typed_reason",
            surfaces=tuple(
                TaxonomyFamilySurface(
                    surface_id=f"surface_{i}",
                    description=f"desc {i}",
                    enforced_by_test_classes=("ExclusionTaxonomyParityTest",),
                )
                for i in range(3)
            ),
            expected_typed_reasons=frozenset({"x"}),
        )
        self.assertFalse(self._check_surface_count(bad, expected=4))

    def test_duplicate_surface_id_fails_uniqueness_check(self) -> None:
        bad = TaxonomyFamilyInventory(
            family_id="synthetic_dup",
            description="duplicate ids",
            kind="typed_reason",
            surfaces=(
                TaxonomyFamilySurface(
                    surface_id="dup",
                    description="first",
                    enforced_by_test_classes=("ExclusionTaxonomyParityTest",),
                ),
                TaxonomyFamilySurface(
                    surface_id="dup",
                    description="second",
                    enforced_by_test_classes=("ExclusionTaxonomyParityTest",),
                ),
            ),
            expected_typed_reasons=frozenset({"x"}),
        )
        self.assertFalse(self._check_unique_surface_ids(bad))

    def test_empty_description_fails_nonempty_check(self) -> None:
        bad = TaxonomyFamilyInventory(
            family_id="synthetic_blank",
            description="blank surface description",
            kind="typed_reason",
            surfaces=(
                TaxonomyFamilySurface(
                    surface_id="ok",
                    description="ok",
                    enforced_by_test_classes=("ExclusionTaxonomyParityTest",),
                ),
                TaxonomyFamilySurface(
                    surface_id="blank",
                    description="   \n  ",
                    enforced_by_test_classes=("ExclusionTaxonomyParityTest",),
                ),
            ),
            expected_typed_reasons=frozenset({"x"}),
        )
        self.assertFalse(self._check_nonempty_descriptions(bad))

    def test_partial_lockstep_synthetic_typed_reason_rejected(self) -> None:
        """If a typed reason is present at SOME surfaces but not all
        the surfaces a family declares, the family-level lockstep
        check (modeled here as a set-equality assertion) rejects.

        Concretely: simulate adding a synthetic reason
        (`synthetic_partial_lockstep_test_reason`) to the
        trace-budget family's expected set but only at the
        producer-constants surface (not at writer / discovery /
        schema). The set returned by a hypothetical surface scanner
        is a strict subset of the expected set; assert the equality
        fails. The synthetic reason is a string that is guaranteed
        not to be a real family member so the negative control
        stays meaningful even after future round-added widenings of
        `TRACE_BUDGET_TYPED_REASONS_EXPECTED`."""
        family = _TAXONOMY_FAMILIES["trace_budget"]
        original_expected = family.expected_typed_reasons
        # Synthetic widening: assume the canonical constant included a
        # second reason but only one surface scanner returned it. The
        # placeholder reason MUST be absent from
        # `TRACE_BUDGET_TYPED_REASONS_EXPECTED` so this sanity check
        # stays meaningful as the family grows.
        synthetic_reason = "synthetic_partial_lockstep_test_reason"
        self.assertNotIn(
            synthetic_reason, original_expected,
            msg=(
                "synthetic_partial_lockstep_test_reason must NOT be a "
                "real family member; pick another placeholder if it "
                "ever becomes one."
            ),
        )
        synthetic_expected = original_expected | {synthetic_reason}
        partial_surface_returns = original_expected  # missing the new
        self.assertNotEqual(
            partial_surface_returns,
            synthetic_expected,
            msg=(
                "Sanity: a surface that does NOT carry a newly added "
                "typed reason returns a set that differs from the "
                "expected set. The family-level lockstep equality "
                "check would fail loudly in this case, which is the "
                "intended C-AC-C negative-control behavior."
            ),
        )


class SharingLocalityTypedReasonsParityTest(unittest.TestCase):
    """Lockstep enforcer for the sharing_locality taxonomy family
    (Loop 2 R22 / task12 / the sharing-locality clause).

    Two surfaces:

    1. **Producer-side constant**: the
       ``SHARING_LOCALITY_TYPED_REASONS`` frozenset in
       ``tools/sharing_locality/classifier.py``. The classifier
       returns one of these values in the ``typed_reason`` field
       of any typed-unavailable record.

    2. **JSON schema enum**: the ``$defs.typed_reason.enum`` list
       in
       ``workloads/simbench_bw/schemas/sharing_locality.schema.json``.

    Both surfaces must enumerate exactly
    ``SHARING_LOCALITY_TYPED_REASONS_EXPECTED``.
    """

    @staticmethod
    def _read_producer_constant() -> frozenset[str]:
        classifier_path = (
            _REPO_ROOT
            / "tools"
            / "sharing_locality"
            / "classifier.py"
        )
        text = classifier_path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign):
                target = node.target
                value = node.value
                if not isinstance(target, ast.Name):
                    continue
                if target.id != "SHARING_LOCALITY_TYPED_REASONS":
                    continue
                if value is None:
                    continue
            elif isinstance(node, ast.Assign):
                if len(node.targets) != 1:
                    continue
                target = node.targets[0]
                if not isinstance(target, ast.Name):
                    continue
                if target.id != "SHARING_LOCALITY_TYPED_REASONS":
                    continue
                value = node.value
            else:
                continue
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "frozenset"
                and len(value.args) == 1
                and isinstance(value.args[0], ast.Set)
            ):
                elements: set[str] = set()
                for elt in value.args[0].elts:
                    if isinstance(elt, ast.Constant) and isinstance(
                        elt.value, str
                    ):
                        elements.add(elt.value)
                    else:
                        raise AssertionError(
                            "Non-string constant in "
                            "SHARING_LOCALITY_TYPED_REASONS literal"
                        )
                return frozenset(elements)
            raise AssertionError(
                "SHARING_LOCALITY_TYPED_REASONS must be assigned a "
                "frozenset({...}) literal so the parity scanner can "
                "read it without executing the module."
            )
        raise AssertionError(
            "SHARING_LOCALITY_TYPED_REASONS not found in "
            "tools/sharing_locality/classifier.py"
        )

    @staticmethod
    def _read_schema_enum() -> frozenset[str]:
        schema_path = _SCHEMA_DIR / "sharing_locality.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        defs = schema.get("$defs", {})
        typed_reason_def = defs.get("typed_reason")
        if typed_reason_def is None or "enum" not in typed_reason_def:
            raise AssertionError(
                "sharing_locality.schema.json missing "
                "$defs.typed_reason.enum"
            )
        return frozenset(typed_reason_def["enum"])

    def test_producer_constant_matches_expected(self) -> None:
        producer = self._read_producer_constant()
        self.assertEqual(
            producer,
            SHARING_LOCALITY_TYPED_REASONS_EXPECTED,
            msg=(
                "tools/sharing_locality/classifier.py "
                "SHARING_LOCALITY_TYPED_REASONS drifted from "
                "SHARING_LOCALITY_TYPED_REASONS_EXPECTED."
            ),
        )

    def test_schema_enum_matches_expected(self) -> None:
        schema_enum = self._read_schema_enum()
        self.assertEqual(
            schema_enum,
            SHARING_LOCALITY_TYPED_REASONS_EXPECTED,
            msg=(
                "workloads/simbench_bw/schemas/sharing_locality.schema.json "
                "$defs.typed_reason.enum drifted from "
                "SHARING_LOCALITY_TYPED_REASONS_EXPECTED."
            ),
        )

    def test_producer_and_schema_agree(self) -> None:
        self.assertEqual(
            self._read_producer_constant(),
            self._read_schema_enum(),
            msg=(
                "Producer and schema typed-reason sets diverged. "
                "Sharing-locality taxonomy lockstep failed."
            ),
        )

    def test_round_added_documents_new_typed_reasons(self) -> None:
        # All 7 sharing-locality typed reasons are round-added.
        self.assertTrue(
            SHARING_LOCALITY_TYPED_REASONS_EXPECTED.issubset(
                ROUND_ADDED_TYPED_REASONS,
            ),
            msg=(
                "ROUND_ADDED_TYPED_REASONS must document all 7 "
                "sharing-locality typed reasons. Missing: "
                f"{sorted(SHARING_LOCALITY_TYPED_REASONS_EXPECTED - ROUND_ADDED_TYPED_REASONS)}."
            ),
        )

    def test_no_overlap_with_ac19_plan_enumeration(self) -> None:
        overlap = (
            SHARING_LOCALITY_TYPED_REASONS_EXPECTED
            & AC19_PLAN_ENUMERATED_TYPED_REASONS
        )
        self.assertEqual(
            overlap,
            frozenset(),
            msg=(
                "Sharing-locality typed reasons should be entirely "
                "round-added. The original-plan AC-19 enumeration "
                "does not name them. Overlap: "
                f"{sorted(overlap)}."
            ),
        )


class OrderProjectionsTypedReasonsParityTest(unittest.TestCase):
    """Lockstep enforcer for the order_projections taxonomy
    family (Loop 2 R25 / task13 / the order-projections
    clause).

    Two surfaces: the
    `ORDER_PROJECTIONS_TYPED_REASONS` frozenset in
    `tools/order_projections/projector.py` and the
    `$defs.typed_reason.enum` list in
    `workloads/simbench_bw/schemas/order_projections.schema.json`.
    Both must enumerate
    `ORDER_PROJECTIONS_TYPED_REASONS_EXPECTED`."""

    @staticmethod
    def _read_producer_constant() -> frozenset[str]:
        path = (
            _REPO_ROOT
            / "tools"
            / "order_projections"
            / "projector.py"
        )
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign):
                target = node.target
                value = node.value
                if not isinstance(target, ast.Name):
                    continue
                if target.id != "ORDER_PROJECTIONS_TYPED_REASONS":
                    continue
                if value is None:
                    continue
            elif isinstance(node, ast.Assign):
                if len(node.targets) != 1:
                    continue
                target = node.targets[0]
                if not isinstance(target, ast.Name):
                    continue
                if target.id != "ORDER_PROJECTIONS_TYPED_REASONS":
                    continue
                value = node.value
            else:
                continue
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "frozenset"
                and len(value.args) == 1
                and isinstance(value.args[0], ast.Set)
            ):
                elements: set[str] = set()
                for elt in value.args[0].elts:
                    if isinstance(elt, ast.Constant) and isinstance(
                        elt.value, str
                    ):
                        elements.add(elt.value)
                    else:
                        raise AssertionError(
                            "Non-string constant in "
                            "ORDER_PROJECTIONS_TYPED_REASONS literal"
                        )
                return frozenset(elements)
            raise AssertionError(
                "ORDER_PROJECTIONS_TYPED_REASONS must be assigned a "
                "frozenset({...}) literal so the parity scanner can "
                "read it without executing the module."
            )
        raise AssertionError(
            "ORDER_PROJECTIONS_TYPED_REASONS not found in "
            "tools/order_projections/projector.py"
        )

    @staticmethod
    def _read_schema_enum() -> frozenset[str]:
        schema_path = _SCHEMA_DIR / "order_projections.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        defs = schema.get("$defs", {})
        typed_reason_def = defs.get("typed_reason")
        if typed_reason_def is None or "enum" not in typed_reason_def:
            raise AssertionError(
                "order_projections.schema.json missing "
                "$defs.typed_reason.enum"
            )
        return frozenset(typed_reason_def["enum"])

    def test_producer_constant_matches_expected(self) -> None:
        producer = self._read_producer_constant()
        self.assertEqual(
            producer,
            ORDER_PROJECTIONS_TYPED_REASONS_EXPECTED,
        )

    def test_schema_enum_matches_expected(self) -> None:
        schema_enum = self._read_schema_enum()
        self.assertEqual(
            schema_enum,
            ORDER_PROJECTIONS_TYPED_REASONS_EXPECTED,
        )

    def test_producer_and_schema_agree(self) -> None:
        self.assertEqual(
            self._read_producer_constant(),
            self._read_schema_enum(),
        )

    def test_round_added_documents_remaining_typed_reasons(self) -> None:
        # The SM-local unavailable arm uses the original-plan
        # AC-19 enumerated reason
        # `projection_unavailable_sm_local`; the other 6
        # order-projections reasons are round-added.
        round_added_subset = (
            ORDER_PROJECTIONS_TYPED_REASONS_EXPECTED
            - {"projection_unavailable_sm_local"}
        )
        self.assertTrue(
            round_added_subset.issubset(ROUND_ADDED_TYPED_REASONS),
            msg=(
                "ROUND_ADDED_TYPED_REASONS must document the 6 "
                "round-added order-projections typed reasons. "
                "Missing: "
                f"{sorted(round_added_subset - ROUND_ADDED_TYPED_REASONS)}."
            ),
        )

    def test_overlap_with_ac19_plan_enumeration(self) -> None:
        # The original-plan AC-19 enumeration names
        # `projection_unavailable_sm_local` (Codex's R25 review
        # directive #1: do NOT invent a new reason for the
        # SM-local unavailable arm). The overlap between the
        # order-projections expected set and the AC-19 plan
        # enumeration must be exactly that one reason.
        overlap = (
            ORDER_PROJECTIONS_TYPED_REASONS_EXPECTED
            & AC19_PLAN_ENUMERATED_TYPED_REASONS
        )
        self.assertEqual(
            overlap,
            frozenset({"projection_unavailable_sm_local"}),
            msg=(
                "Order-projections expected set must include "
                "the original-plan AC-19 reason "
                "`projection_unavailable_sm_local` for the "
                "SM-local unavailable arm and NO other AC-19 "
                "reasons."
            ),
        )


class PerturbationRunLedgerTypedReasonsParityTest(unittest.TestCase):
    """Lockstep enforcer for the perturbation_run_ledger taxonomy
    family (Loop 2 R31 / task15 / the perturbation-harness
    clause).

    Two surfaces: the
    `PERTURBATION_RUN_LEDGER_TYPED_REASONS` frozenset in
    `tools/perturbation_harness/ledger.py` and the
    `$defs.typed_reason.enum` list in
    `workloads/simbench_bw/schemas/perturbation_run_ledger.schema.json`.
    Both must enumerate
    `PERTURBATION_RUN_LEDGER_TYPED_REASONS_EXPECTED`."""

    @staticmethod
    def _read_producer_constant() -> frozenset[str]:
        path = (
            _REPO_ROOT
            / "tools"
            / "perturbation_harness"
            / "ledger.py"
        )
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign):
                target = node.target
                value = node.value
                if not isinstance(target, ast.Name):
                    continue
                if target.id != "PERTURBATION_RUN_LEDGER_TYPED_REASONS":
                    continue
                if value is None:
                    continue
            elif isinstance(node, ast.Assign):
                if len(node.targets) != 1:
                    continue
                target = node.targets[0]
                if not isinstance(target, ast.Name):
                    continue
                if target.id != "PERTURBATION_RUN_LEDGER_TYPED_REASONS":
                    continue
                value = node.value
            else:
                continue
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "frozenset"
                and len(value.args) == 1
                and isinstance(value.args[0], ast.Set)
            ):
                elements: set[str] = set()
                for elt in value.args[0].elts:
                    if isinstance(elt, ast.Constant) and isinstance(
                        elt.value, str
                    ):
                        elements.add(elt.value)
                    else:
                        raise AssertionError(
                            "Non-string constant in "
                            "PERTURBATION_RUN_LEDGER_TYPED_REASONS literal"
                        )
                return frozenset(elements)
            raise AssertionError(
                "PERTURBATION_RUN_LEDGER_TYPED_REASONS must be "
                "assigned a frozenset({...}) literal so the parity "
                "scanner can read it without executing the module."
            )
        raise AssertionError(
            "PERTURBATION_RUN_LEDGER_TYPED_REASONS not found in "
            "tools/perturbation_harness/ledger.py"
        )

    @staticmethod
    def _read_schema_enum() -> frozenset[str]:
        schema_path = _SCHEMA_DIR / "perturbation_run_ledger.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        defs = schema.get("$defs", {})
        typed_reason_def = defs.get("typed_reason")
        if typed_reason_def is None or "enum" not in typed_reason_def:
            raise AssertionError(
                "perturbation_run_ledger.schema.json missing "
                "$defs.typed_reason.enum"
            )
        return frozenset(typed_reason_def["enum"])

    def test_producer_constant_matches_expected(self) -> None:
        producer = self._read_producer_constant()
        self.assertEqual(
            producer,
            PERTURBATION_RUN_LEDGER_TYPED_REASONS_EXPECTED,
        )

    def test_schema_enum_matches_expected(self) -> None:
        schema_enum = self._read_schema_enum()
        self.assertEqual(
            schema_enum,
            PERTURBATION_RUN_LEDGER_TYPED_REASONS_EXPECTED,
        )

    def test_producer_and_schema_agree(self) -> None:
        self.assertEqual(
            self._read_producer_constant(),
            self._read_schema_enum(),
        )

    def test_overlap_with_ac19_plan_enumeration(self) -> None:
        # The original-plan AC-19 enumeration names
        # `modifier_optimized_away` as the perturbation-harness
        # clause's plan-enumerated typed reason. The overlap
        # between the perturbation_run_ledger expected set and
        # the AC-19 plan enumeration must be exactly that one
        # reason; the 6 `arch_feature_unavailable_*` reasons are
        # wildcard-family instances documented in
        # ROUND_ADDED_TYPED_REASONS rather than literal AC-19
        # entries.
        overlap = (
            PERTURBATION_RUN_LEDGER_TYPED_REASONS_EXPECTED
            & AC19_PLAN_ENUMERATED_TYPED_REASONS
        )
        self.assertEqual(
            overlap,
            frozenset({"modifier_optimized_away"}),
            msg=(
                "perturbation_run_ledger expected set must include "
                "the original-plan AC-19 reason "
                "`modifier_optimized_away` for the SASS-modifier "
                "audit arm and NO other AC-19 reasons."
            ),
        )

    def test_round_added_documents_remaining_typed_reasons(self) -> None:
        # All reasons except the plan-enumerated
        # `modifier_optimized_away` are round-added. The 6
        # `arch_feature_unavailable_*` reasons are concrete
        # instances of the plan's wildcard family but are
        # documented in ROUND_ADDED_TYPED_REASONS rather than the
        # literal AC-19 set so the strict subset/superset parity
        # check still holds.
        round_added_subset = (
            PERTURBATION_RUN_LEDGER_TYPED_REASONS_EXPECTED
            - {"modifier_optimized_away"}
        )
        self.assertTrue(
            round_added_subset.issubset(ROUND_ADDED_TYPED_REASONS),
            msg=(
                "ROUND_ADDED_TYPED_REASONS must document the 8 "
                "round-added perturbation_run_ledger typed reasons "
                "(2 pure round-added + 6 wildcard-family "
                "instances). Missing: "
                f"{sorted(round_added_subset - ROUND_ADDED_TYPED_REASONS)}."
            ),
        )


class TraceConvergenceTypedReasonsParityTest(unittest.TestCase):
    """Lockstep enforcer for the trace_convergence taxonomy family
    (Loop 2 R37 / task17 / AC-11 — the adaptive-sampler clause).

    Four surfaces:

    1. Producer constant `TRACE_CONVERGENCE_TYPED_REASONS` frozenset
       in `tools/adaptive_sampler/trace_convergence_ledger.py`
       (per-reason enumeration).
    2. Controller helper `_emit_trace_convergence_confidence_for_run_summary`
       in `workloads/simbench_bw/run_suite.py` (dispatch presence:
       the helper writes the bundle to
       `convergence_confidence/convergence_confidence.json` and
       persists the `trace_convergence_typed_reason` run-summary
       field).
    3. Validator helper `validate_trace_convergence_confidence` in
       `workloads/simbench_bw/validate.py` (dispatch presence: the
       helper invokes the strict schema validator on each
       `convergence_confidence.json` discovered via rglob).
    4. JSON Schema `admission.oneOf` `properties.typed_reason.const`
       literals in
       `workloads/simbench_bw/schemas/trace_convergence_confidence.schema.json`
       (per-reason enumeration via 4 oneOf branches).

    Surfaces 1 and 4 enumerate the four typed reasons explicitly and
    must equal `TRACE_CONVERGENCE_TYPED_REASONS_EXPECTED`. Surfaces
    2 and 3 are dispatch presence checks: the helper / validator
    function must exist with the expected name and the expected
    literal references to the bundle path / schema file. The
    plan-enumerated reason
    `trace_convergence_not_reached_budget_cap` is in
    AC19_PLAN_ENUMERATED_TYPED_REASONS; the other three are
    round-added per ROUND_ADDED_TYPED_REASONS.
    """

    @staticmethod
    def _read_producer_constant() -> frozenset[str]:
        text = _ADAPTIVE_SAMPLER_LEDGER_PY.read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign):
                target = node.target
                value = node.value
                if not isinstance(target, ast.Name):
                    continue
                if target.id != "TRACE_CONVERGENCE_TYPED_REASONS":
                    continue
                if value is None:
                    continue
            elif isinstance(node, ast.Assign):
                if len(node.targets) != 1:
                    continue
                target = node.targets[0]
                if not isinstance(target, ast.Name):
                    continue
                if target.id != "TRACE_CONVERGENCE_TYPED_REASONS":
                    continue
                value = node.value
            else:
                continue
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "frozenset"
                and len(value.args) == 1
                and isinstance(value.args[0], ast.Set)
            ):
                elements: set[str] = set()
                for elt in value.args[0].elts:
                    if isinstance(elt, ast.Constant) and isinstance(
                        elt.value, str
                    ):
                        elements.add(elt.value)
                    else:
                        raise AssertionError(
                            "Non-string constant in "
                            "TRACE_CONVERGENCE_TYPED_REASONS literal"
                        )
                return frozenset(elements)
            raise AssertionError(
                "TRACE_CONVERGENCE_TYPED_REASONS must be assigned a "
                "frozenset({...}) literal so the parity scanner can "
                "read it without executing the module."
            )
        raise AssertionError(
            "TRACE_CONVERGENCE_TYPED_REASONS not found in "
            "tools/adaptive_sampler/trace_convergence_ledger.py"
        )

    @staticmethod
    def _read_schema_oneof_const_literals() -> frozenset[str]:
        """Extract the `properties.typed_reason.const` literal from
        every `admission.oneOf` branch that has one. The schema's
        `admission` field is a `$ref` to `#/$defs/admission`, whose
        `oneOf` enumerates the admitted (`result_kind=ok`) branch
        plus the four typed-unavailable branches; only the four
        typed-unavailable branches declare a `typed_reason.const`."""
        schema_path = _SCHEMA_DIR / "trace_convergence_confidence.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        # `admission` lives under `$defs/admission` (the property
        # `properties.admission` is a `$ref` that points to it).
        admission = schema.get("$defs", {}).get("admission")
        if not isinstance(admission, dict):
            raise AssertionError(
                "trace_convergence_confidence.schema.json missing "
                "$defs.admission"
            )
        one_of = admission.get("oneOf")
        if not isinstance(one_of, list) or not one_of:
            raise AssertionError(
                "trace_convergence_confidence.schema.json "
                "$defs.admission.oneOf must be a non-empty list"
            )
        reasons: set[str] = set()
        for branch in one_of:
            if not isinstance(branch, dict):
                continue
            props = branch.get("properties")
            if not isinstance(props, dict):
                continue
            typed_reason = props.get("typed_reason")
            if not isinstance(typed_reason, dict):
                continue
            const_val = typed_reason.get("const")
            if isinstance(const_val, str):
                reasons.add(const_val)
        return frozenset(reasons)

    def test_producer_constant_matches_expected(self) -> None:
        producer = self._read_producer_constant()
        self.assertEqual(
            producer,
            TRACE_CONVERGENCE_TYPED_REASONS_EXPECTED,
            msg=(
                "TRACE_CONVERGENCE_TYPED_REASONS in "
                "tools/adaptive_sampler/trace_convergence_ledger.py "
                "drifted from TRACE_CONVERGENCE_TYPED_REASONS_EXPECTED. "
                "Update the expected set in this test AND verify all "
                "four lockstep surfaces (producer constant + run_suite "
                "writer helper + validate.py discovery + schema oneOf "
                "consts) are aligned."
            ),
        )

    def test_schema_oneof_consts_match_expected(self) -> None:
        schema_consts = self._read_schema_oneof_const_literals()
        self.assertEqual(
            schema_consts,
            TRACE_CONVERGENCE_TYPED_REASONS_EXPECTED,
            msg=(
                "trace_convergence_confidence.schema.json "
                "admission.oneOf typed_reason consts drifted from "
                "TRACE_CONVERGENCE_TYPED_REASONS_EXPECTED. Update "
                "the expected set in this test AND verify the "
                "producer constant, run_suite helper, and validate.py "
                "dispatch all enumerate the same four reasons."
            ),
        )

    def test_producer_and_schema_agree(self) -> None:
        self.assertEqual(
            self._read_producer_constant(),
            self._read_schema_oneof_const_literals(),
            msg=(
                "Producer constant TRACE_CONVERGENCE_TYPED_REASONS and "
                "schema admission.oneOf typed_reason consts must be "
                "identical. C-AC-C lockstep discipline: a typed reason "
                "added at one surface MUST land at the other in the "
                "same round."
            ),
        )

    def test_simbench_writer_helper_dispatch_present(self) -> None:
        """Surface 2 (dispatch presence): the controller helper
        `_emit_trace_convergence_confidence_for_run_summary` must
        exist in run_suite.py and must reference both the bundle
        path (`convergence_confidence.json`) and the run-summary
        field (`trace_convergence_typed_reason`). The dispatch
        carries the typed reasons from the producer admission
        helpers into the run summary; the presence check ensures
        the carrier is wired."""
        text = _RUN_SUITE_PY.read_text(encoding="utf-8")
        self.assertIn(
            "_emit_trace_convergence_confidence_for_run_summary",
            text,
            msg=(
                "run_suite.py is missing the trace-convergence "
                "controller helper "
                "`_emit_trace_convergence_confidence_for_run_summary`. "
                "C-AC-C surface 2 (writer dispatch) is broken; the "
                "producer's typed reasons cannot reach the run "
                "summary."
            ),
        )
        self.assertIn(
            "convergence_confidence.json",
            text,
            msg=(
                "run_suite.py is missing the bundle filename literal "
                "`convergence_confidence.json` referenced by the "
                "trace-convergence controller helper."
            ),
        )
        self.assertIn(
            "trace_convergence_typed_reason",
            text,
            msg=(
                "run_suite.py is missing the "
                "`trace_convergence_typed_reason` run-summary field "
                "literal; the carrier dispatch surface cannot record "
                "the producer's typed reason without it."
            ),
        )

    def test_validator_dispatch_present(self) -> None:
        """Surface 3 (dispatch presence): the validator helper
        `validate_trace_convergence_confidence` must exist in
        validate.py, must invoke the strict schema validator on
        `trace_convergence_confidence.schema.json`, and must be
        wired into the discovery loop via
        `rglob("convergence_confidence.json")`. Without these, the
        validator silently admits malformed bundles and the
        per-reason `oneOf` lockstep at surface 4 is unenforced."""
        text = _VALIDATE_PY.read_text(encoding="utf-8")
        self.assertIn(
            "def validate_trace_convergence_confidence(",
            text,
            msg=(
                "validate.py is missing the trace-convergence "
                "validator helper "
                "`validate_trace_convergence_confidence`. C-AC-C "
                "surface 3 (validator dispatch) is broken; the "
                "schema's `oneOf` typed_reason consts are not "
                "enforced at validation time."
            ),
        )
        self.assertIn(
            "trace_convergence_confidence.schema.json",
            text,
            msg=(
                "validate.py is missing the schema reference "
                "`trace_convergence_confidence.schema.json`; the "
                "validator dispatch cannot route the bundle to the "
                "strict schema validator."
            ),
        )
        self.assertIn(
            'rglob("convergence_confidence.json")',
            text,
            msg=(
                "validate.py is missing the discovery glob "
                "`rglob(\"convergence_confidence.json\")`; the "
                "validator dispatch is not wired into the bundle "
                "discovery loop."
            ),
        )

    def test_overlap_with_ac19_plan_enumeration(self) -> None:
        """The original-plan AC-19 enumeration names
        `trace_convergence_not_reached_budget_cap` for the budget-
        cap exhaustion arm. The overlap between the trace_convergence
        expected set and the AC-19 plan enumeration must be exactly
        that one reason; the other 3 reasons are round-added."""
        overlap = (
            TRACE_CONVERGENCE_TYPED_REASONS_EXPECTED
            & AC19_PLAN_ENUMERATED_TYPED_REASONS
        )
        self.assertEqual(
            overlap,
            frozenset({"trace_convergence_not_reached_budget_cap"}),
            msg=(
                "trace_convergence expected set must include the "
                "original-plan AC-19 reason "
                "`trace_convergence_not_reached_budget_cap` for the "
                "budget-cap exhaustion arm and NO other AC-19 "
                "reasons."
            ),
        )

    def test_round_added_documents_remaining_typed_reasons(self) -> None:
        """All reasons except the plan-enumerated
        `trace_convergence_not_reached_budget_cap` are round-added.
        The 3 remaining reasons must be in
        ROUND_ADDED_TYPED_REASONS."""
        round_added_subset = (
            TRACE_CONVERGENCE_TYPED_REASONS_EXPECTED
            - {"trace_convergence_not_reached_budget_cap"}
        )
        self.assertTrue(
            round_added_subset.issubset(ROUND_ADDED_TYPED_REASONS),
            msg=(
                "ROUND_ADDED_TYPED_REASONS must document the 3 "
                "round-added trace_convergence typed reasons. "
                "Missing: "
                f"{sorted(round_added_subset - ROUND_ADDED_TYPED_REASONS)}."
            ),
        )


if __name__ == "__main__":
    unittest.main()
