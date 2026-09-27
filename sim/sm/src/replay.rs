//! Trace → SM replay driver: flattens one traced kernel launch into per-SM
//! block replays using the REAL hardware placement recorded in the trace
//! (per-instruction `sm_id` / `warpid_sm`), joins each dynamic record to
//! the issue schedule (and, when an L1 setup is supplied, to the cache-op
//! manifest + per-record unique-sector expansion), and runs the issue
//! model on every SM.
//!
//! Placement policy v0: spatial placement is traced — each block runs on the
//! SM the hardware ran it on, and each warp uses its observed physical slot.
//! Traces without `sm_id` are a typed error (a synthetic placement model is a
//! future knob, not a silent fallback). Exact temporal block co-residency is not
//! present on the wire; launch-order overlap under the resident-warp and
//! architecture-card shared-memory ceilings plus disjoint traced physical slots is
//! explicitly typed as [`CoResidency::Modeled`].
//!
//! Memory-record classification (L1 mode): a global-class record takes
//! the [`MemRole`] its pc's SIDECAR OPCODE was classified to at join
//! time (`sim_mem::cache_op` port of the Python token rules) — eligible
//! load → modeled L1 sectors; bypass load → counted bypass; store head →
//! hit-tested without allocating; everything else → unified-denominator
//! count only. Sector expansion follows the `l1_trace_adapter` contract:
//! per-lane `addr>>5 ..= (addr+width-1)>>5`, deduplicated per INSTRUCTION
//! in first-seen order (active lanes are the non-sentinel slots — the
//! canonical positional guarantee).
//!
//! v0 limitation (fail-closed, not silent): every dynamic record of a
//! launch must carry the launch's own `function_unique_id` — traced device
//! function CALLs would need per-segment schedules and are rejected typed
//! until a trace exercises them (none of the 17 M1 traces does).
//!
//! Memory discipline: threadblock payloads are decoded and semantically
//! validated once when `TraceDir` captures the immutable snapshot. The reader
//! retains compact scheduler/memory metadata plus a per-warp sector arena and
//! discards the address-heavy Prost tree. Extraction never performs a second
//! wire walk or lane-address reconstruction.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::Serialize;

use sim_arch::card::{ArchCard, CalibError};
use sim_arch::{ArchId, LdgstsDestinationWavefrontProjection, Sku};
#[cfg(test)]
use sim_mem::SectorExpansionPlan;
use sim_mem::{L1Config, L1CounterError, L1Counters, SectorExpansionError};
#[cfg(test)]
use sim_trace::lanes::LANE_INACTIVE_SENTINEL;
use sim_trace::pb;
use sim_trace::reader::{
    EffectiveDataBytesError, LdgstsDestinationTopology, LdgstsSourceCohort, MemoryAccessMeta,
    MemoryPayloadContract, SharedBankProjectionError, ValidatedMemoryInstruction,
    ValidatedMemoryRoute, ValidatedMemoryRouteError, ValidatedSectorError,
    ValidatedSectorErrorPhase, ValidatedSectorPlan, ValidatedWarp,
};
#[cfg(test)]
use sim_trace::reader::{
    PreflightedMemoryAccess, preflight_memory_access, project_effective_data_bytes,
    project_shared_bank_wavefronts_32x4,
};
use sim_trace::{KernelLoc, TraceDir, TraceError, TraceLineage, kernel_grid};
#[cfg(feature = "campaign")]
use sim_trace::{
    LosslessFullProvenance, TraceSamplingMethod, TraceSamplingSidecar, WaveSegmentSubsetProvenance,
};

use sim_mem::{
    ASYNC_COPY_COMPLETION_VALUE_MASK, ASYNC_COPY_COMPOSITE_PAYLOAD_WORK_MAX, CacheOpToken,
    L2Config, L2Counters, L2Model, LdgstsL1Policy, MemRole, encode_async_copy_destination_work,
    encode_async_copy_payload_and_topology, encode_async_copy_payload_work,
    is_known_untagged_opcode, opcode_head,
};

use crate::causal_frontier::{
    CausalFrontierInstruction, CausalFrontierReport, CausalFrontierResidency,
    summarize_with_residency,
};
#[cfg(feature = "campaign")]
use crate::evidence::AdmittedLaunchEvidence;
use crate::issue::{
    AccessKind, AsyncCopyCohortDescriptor, AsyncCopyCohortReplay, AsyncCopyCohortSpan,
    AsyncInstrumentationOrderDiagnostics, BlockReplay, InstrumentationOrderAnchor, IssueError,
    IssueModelParams, IssueResourceSource, MemoryAdmissionCandidate, MemoryServiceCandidate,
    ReplayInst, SharedMemoryResidency, SmSimResult, TailFlags, WarpReplay,
    simulate_sm_with_residency, validate_memory_admission_candidates,
    validate_memory_service_candidates,
};
use crate::metrics::TraceMetrics;
use crate::program::{FunctionSchedule, OpClass, PipelineClass};
use crate::system::{
    DramBw, DramStats, L2SourceSplit, SystemError,
    simulate_kernel_shared_with_residency_and_candidates,
};

mod aggregation;
pub use aggregation::{
    FullCounterEstimatorProvenance, FullLaunchSetCounters, FullLaunchSetDeclaration,
    FullLaunchSetDeclarationError, FullLaunchTimingDiagnostic, LaunchSetAggregationError,
    LaunchSetCounterComponent, LaunchSetScalarCounter, LaunchSetScope, LaunchSetScopeError,
    ReplayLaunchCoverageBindingError, aggregate_full_launch_set,
};

#[derive(Debug, thiserror::Error, Clone, Copy, PartialEq, Eq)]
pub enum MemoryRouteError {
    #[error("GLOBAL carries {got} address records; active records require at least one operand")]
    GlobalOperandCount { got: usize },
    #[error("GLOBAL operand {operand} carries non-global address_space {space:?}")]
    GlobalSpaceInvalid {
        operand: usize,
        space: pb::MemorySpace,
    },
    #[error(
        "SHARED/LOCAL carries {got} address records; active records require at least one operand"
    )]
    SharedLocalOperandCount { got: usize },
    #[error("SHARED/LOCAL operand {operand} carries incompatible address_space {space:?}")]
    SharedLocalSpaceInvalid {
        operand: usize,
        space: pb::MemorySpace,
    },
    #[error(
        "ATOMIC carries {got} address records; expected exactly one explicit-space operand, or none only when zero-active"
    )]
    AtomicOperandCount { got: usize },
    #[error("ATOMIC operand {operand} is missing address_space")]
    AtomicSpaceMissing { operand: usize },
    #[error("ATOMIC operand {operand} carries UNSPECIFIED address_space")]
    AtomicSpaceUnspecified { operand: usize },
    #[error("legacy ATOMIC operands carry conflicting address-space routes")]
    AtomicSpacesConflict,
    #[error("ASYNC_COPY carries {got} address records; expected exactly one global source")]
    AsyncCopySourceCount { got: usize },
    #[error("ASYNC_COPY source is missing address_space")]
    AsyncCopySourceSpaceMissing,
    #[error("ASYNC_COPY source carries non-global address_space {space:?}")]
    AsyncCopySourceSpaceInvalid { space: pb::MemorySpace },
    #[error("ASYNC_COPY carries {got} operands; destination-bearing contract expects {expected}")]
    AsyncCopyOperandCount { got: usize, expected: usize },
    #[error("ASYNC_COPY destination is missing address_space")]
    AsyncCopyDestinationSpaceMissing,
    #[error("ASYNC_COPY destination carries non-shared address_space {space:?}")]
    AsyncCopyDestinationSpaceInvalid { space: pb::MemorySpace },
    #[error("memory_op_access_kind {got:?} is incompatible; expected {expected}")]
    AccessKindInvalid {
        got: pb::MemoryOpAccessKind,
        expected: &'static str,
    },
    #[error(
        "memory_op_access_kind {got:?} contradicts the joined static opcode direction {expected:?}"
    )]
    StaticAccessKindMismatch {
        got: pb::MemoryOpAccessKind,
        expected: pb::MemoryOpAccessKind,
    },
    #[error("joined static opcode requires {expected:?} address-space routing, got {got:?}")]
    StaticMemoryRouteMismatch {
        expected: pb::MemorySpace,
        got: pb::MemorySpace,
    },
    #[error("joined static opcode requires memory class {expected:?}, got {got:?}")]
    StaticMemoryClassMismatch {
        expected: pb::MemoryOpClass,
        got: pb::MemoryOpClass,
    },
    #[error("static SASS family {family} has no implemented route on architecture {arch}")]
    StaticMemoryFamilyUnsupported { family: &'static str, arch: ArchId },
    #[error("operand {operand} carries address_space value {raw} unknown to the committed schema")]
    UnknownAddressSpace { operand: usize, raw: i32 },
}

#[derive(Debug, thiserror::Error)]
pub enum ReplayError {
    #[error(transparent)]
    Trace(#[from] TraceError),
    #[error(transparent)]
    Issue(#[from] IssueError),
    #[error(transparent)]
    System(#[from] SystemError),
    #[error("L2 config invalid: {0}")]
    L2Config(#[from] sim_mem::L2ConfigError),
    #[error("L2 construction failed: {0}")]
    L2Build(#[from] sim_mem::L2BuildError),
    #[error("L2 snapshot failed: {0}")]
    L2Snapshot(#[from] sim_mem::L2SnapshotError),
    #[error("L2 counter operation failed: {0}")]
    L2Counters(#[from] sim_mem::L2CounterError),
    #[error("retained L2 configuration mismatch: expected {expected:?}, got {got:?}")]
    RetainedL2ConfigurationMismatch { expected: L2Config, got: L2Config },
    #[error(
        "retained L2 launch boundary is not fully drained: {pending_sectors} sector(s) remain pending"
    )]
    RetainedL2NotDrained { pending_sectors: usize },
    #[error(
        "memory-service candidates require L1SharedL2 replay so their issue-time topology is evaluated on the full hierarchy"
    )]
    MemoryServiceCandidatesRequireSharedL2,
    #[error(
        "memory-service candidates require full launch coverage; sampled replay discards per-SM timing diagnostics"
    )]
    MemoryServiceCandidatesRequireFullCoverage,
    #[error(
        "memory-admission candidates require L1SharedL2 replay so exact L2 outcomes are available"
    )]
    MemoryAdmissionCandidatesRequireSharedL2,
    #[error(
        "memory-admission candidates require full launch coverage; sampled replay discards per-SM timing diagnostics"
    )]
    MemoryAdmissionCandidatesRequireFullCoverage,
    #[error("SM {sm_id} L1 counter aggregation failed: {source}")]
    L1Aggregation {
        sm_id: i32,
        #[source]
        source: L1CounterError,
    },
    #[error("SM {sm_id} L1 counter presence mismatch: expected={expected}, found={found}")]
    L1PresenceMismatch {
        sm_id: i32,
        expected: bool,
        found: bool,
    },
    #[error("replay architecture mismatch for {component}: expected {expected}, got {got}")]
    ArchitectureMismatch {
        component: &'static str,
        expected: ArchId,
        got: ArchId,
    },
    #[error(
        "trace binary_version {binary_version} is outside the unsigned architecture-version domain"
    )]
    TraceArchitectureVersionOutOfRange { binary_version: i32 },
    #[error(
        "trace binary_version {binary_version} is not an admitted deployment architecture: {source}"
    )]
    TraceArchitectureUnknown {
        binary_version: u32,
        #[source]
        source: sim_arch::ArchError,
    },
    #[error("architecture-card placement data is unavailable: {source}")]
    ArchitectureCardPlacement {
        #[source]
        source: CalibError,
    },
    #[error("architecture {arch} shared-memory capacity is unavailable from its card: {source}")]
    ArchitectureCardSharedMemoryCapacity {
        arch: ArchId,
        #[source]
        source: CalibError,
    },
    #[error("architecture {arch} card has invalid per-SM shared-memory capacity {bytes} bytes")]
    SharedMemoryCapacityInvalid { arch: ArchId, bytes: u64 },
    #[error(
        "kernel {kernel_id} launch shared-memory total {bytes} is outside the unsigned byte domain"
    )]
    SharedMemoryDemandInvalid { kernel_id: i32, bytes: i32 },
    #[error(
        "kernel {kernel_id} requires {bytes_per_cta} shared-memory bytes per CTA, exceeding architecture {arch} per-SM capacity {max_bytes_per_sm}"
    )]
    CtaSharedMemoryCapacityExceeded {
        kernel_id: i32,
        arch: ArchId,
        bytes_per_cta: u64,
        max_bytes_per_sm: u64,
    },
    #[error(
        "{path}: traced sm_id {sm_id} is outside architecture {arch} SKU {sku:?} range 0..{sm_count}"
    )]
    SmIdOutOfRange {
        path: PathBuf,
        sm_id: i32,
        arch: ArchId,
        sku: Sku,
        sm_count: u32,
    },
    #[cfg(feature = "campaign")]
    #[error(
        "trace_sampling per_sm_slot_map block {launch_index} declares sm_id {sm_id} outside architecture {arch} SKU {sku:?} range 0..{sm_count}"
    )]
    SamplingSmIdOutOfRange {
        launch_index: u64,
        sm_id: u64,
        arch: ArchId,
        sku: Sku,
        sm_count: u32,
    },
    #[cfg(feature = "campaign")]
    #[error(
        "trace_sampling per_sm_slot_map has no placement for observed launch block {launch_index}"
    )]
    SamplingPlacementMissing { launch_index: i64 },
    #[cfg(feature = "campaign")]
    #[error(
        "launch block {launch_index} traced sm_id {traced_sm_id}, but trace_sampling per_sm_slot_map declares {declared_sm_id}"
    )]
    SamplingPlacementMismatch {
        launch_index: i64,
        traced_sm_id: i32,
        declared_sm_id: u64,
    },

    #[error(
        "memory class {class:?} has no implemented replay path for architecture {arch}; the \
         simulator refuses to route it through generic global timing"
    )]
    UnsupportedMemoryPath {
        arch: ArchId,
        class: pb::MemoryOpClass,
    },
    #[error(
        "{path}: warp {warp_id} record {record_ix}: {class:?} operand routing is invalid: {source}"
    )]
    MemoryRoute {
        path: PathBuf,
        warp_id: i32,
        record_ix: usize,
        class: pb::MemoryOpClass,
        #[source]
        source: MemoryRouteError,
    },
    #[error(
        "global memory record at pc {pc:#x} on architecture {arch} has unknown static opcode head {opcode:?}"
    )]
    UnknownGlobalOpcode {
        arch: ArchId,
        pc: u32,
        opcode: String,
    },
    #[error(
        "local memory record at pc {pc:#x} on architecture {arch} has static opcode head {opcode:?}; expected an admitted LDL or STL role"
    )]
    UnknownLocalOpcode {
        arch: ArchId,
        pc: u32,
        opcode: String,
    },
    #[error("no issue schedule for function_unique_id {function_unique_id}")]
    ScheduleMissing { function_unique_id: i32 },
    #[error(
        "admitted launch ({device_id}, {stream_id}, {kernel_id}) is absent from its retained trace snapshot"
    )]
    #[cfg(feature = "campaign")]
    AdmittedLaunchMissing {
        device_id: i64,
        stream_id: i64,
        kernel_id: i32,
    },
    #[error(
        "{path}: warp {warp_id} record {record_ix} carries function_unique_id {found}, launch \
         kernel is {expected} — multi-function replay is not supported in v0"
    )]
    MultiFunctionUnsupported {
        path: PathBuf,
        warp_id: i32,
        record_ix: usize,
        expected: i32,
        found: i32,
    },
    #[error("{path}: no instruction record carries sm_id — v0 replay requires traced placement")]
    SmIdMissing { path: PathBuf },
    #[error(
        "{path}: sm_id changes within the block ({first} then {second}) — blocks never migrate"
    )]
    SmIdInconsistent {
        path: PathBuf,
        first: i32,
        second: i32,
    },
    #[error("{path}: warp {warp_id} warpid_sm changes within the warp ({first} then {second})")]
    WarpSlotInconsistent {
        path: PathBuf,
        warp_id: i32,
        first: i32,
        second: i32,
    },
    #[error(
        "{path}: warp {warp_id} has no warpid_sm on any record — v0 replay requires the traced \
         physical warp slot (sub-core assignment), like sm_id"
    )]
    WarpSlotMissing { path: PathBuf, warp_id: i32 },
    #[error("kernel launch block {block:?} overflows the u64 thread-count domain")]
    BlockSizeOverflow { block: (u32, u32, u32) },
    #[error(
        "{path}: threadblock warp roster has {actual} entries, expected {expected} from blockDim"
    )]
    WarpRosterCardinalityMismatch {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
    #[error(
        "{path}: threadblock warp id {warp_id} is outside the expected block-local range 0..{expected_warps}"
    )]
    WarpIdOutOfRange {
        path: PathBuf,
        warp_id: i32,
        expected_warps: u64,
    },
    #[error("{path}: threadblock warp map key {map_key} disagrees with embedded warp id {warp_id}")]
    WarpIdentityMismatch {
        path: PathBuf,
        map_key: i32,
        warp_id: i32,
    },
    #[error("{path}: warp {warp_id} record {record_ix} has no warpid_tb on the wire")]
    InstructionWarpIdMissing {
        path: PathBuf,
        warp_id: i32,
        record_ix: usize,
    },
    #[error(
        "{path}: warp {warp_id} record {record_ix} carries warpid_tb {record_warp_id}, which disagrees with its enclosing warp"
    )]
    InstructionWarpIdMismatch {
        path: PathBuf,
        warp_id: i32,
        record_ix: usize,
        record_warp_id: i32,
    },
    #[error("architecture {arch} has no admitted physical-warp-slot capacity: {source}")]
    WarpCapacityUnavailable {
        arch: ArchId,
        #[source]
        source: sim_arch::CapabilityError,
    },
    #[error(
        "{path}: warp {warp_id} physical slot {warp_slot} is outside architecture {arch} range 0..{max_resident_warps}"
    )]
    WarpSlotOutOfRange {
        path: PathBuf,
        warp_id: i32,
        warp_slot: i32,
        arch: ArchId,
        max_resident_warps: u32,
    },
    #[error("{path}: physical warp slot {warp_slot} is assigned to multiple warps in one CTA")]
    DuplicateWarpSlot { path: PathBuf, warp_slot: i32 },
    #[error(
        "{path}: warp {warp_id} record {record_ix}: {sectors} unique sectors in one record \
         overflows the u16 arena index"
    )]
    SectorCountOverflow {
        path: PathBuf,
        warp_id: i32,
        record_ix: usize,
        sectors: usize,
    },
    #[error("{path}: warp {warp_id} record {record_ix}: sector expansion failed: {source}")]
    SectorExpansion {
        path: PathBuf,
        warp_id: i32,
        record_ix: usize,
        #[source]
        source: SectorExpansionError,
    },
    #[error(
        "{path}: warp {warp_id} record {record_ix}: effective payload-byte projection failed: {source}"
    )]
    EffectiveDataBytes {
        path: PathBuf,
        warp_id: i32,
        record_ix: usize,
        #[source]
        source: EffectiveDataBytesError,
    },
    #[error(
        "{path}: warp {warp_id} record {record_ix}: async-copy source/payload mismatch: source_active={source_active}, sectors={sectors}, payload_bytes={payload_bytes}"
    )]
    AsyncCopyPayloadMismatch {
        path: PathBuf,
        warp_id: i32,
        record_ix: usize,
        source_active: bool,
        sectors: usize,
        payload_bytes: u64,
    },
    #[error(
        "{path}: warp {warp_id} record {record_ix}: async-copy cohort projection {field} mismatch: expected {expected}, got {got}"
    )]
    AsyncCopyCohortProjectionMismatch {
        path: PathBuf,
        warp_id: i32,
        record_ix: usize,
        field: &'static str,
        expected: usize,
        got: usize,
    },
    #[error(
        "{path}: warp {warp_id} record {record_ix}: async-copy cohort replay arena cannot reserve {additional} additional {buffer} entries"
    )]
    AsyncCopyCohortCapacityUnavailable {
        path: PathBuf,
        warp_id: i32,
        record_ix: usize,
        buffer: &'static str,
        additional: usize,
    },
    #[error(
        "{path}: warp {warp_id} record {record_ix}: async-copy completion work {completion_work} exceeds compact replay-token maximum {max_completion_work}"
    )]
    AsyncCopyCompletionWorkOutOfRange {
        path: PathBuf,
        warp_id: i32,
        record_ix: usize,
        completion_work: u64,
        max_completion_work: u32,
    },
    #[error("{path}: warp {warp_id} record {record_ix}: shared-bank projection failed: {source}")]
    SharedBankProjection {
        path: PathBuf,
        warp_id: i32,
        record_ix: usize,
        #[source]
        source: SharedBankProjectionError,
    },
    #[error(
        "{path}: warp {warp_id} record {record_ix}: record count overflows u64 at {accumulated}"
    )]
    RecordCountOverflow {
        path: PathBuf,
        warp_id: i32,
        record_ix: usize,
        accumulated: u64,
    },
    #[error("collection '{collection}' length {len} does not fit u64")]
    CollectionLengthOutOfRange {
        collection: &'static str,
        len: usize,
    },
    #[error("SM {sm_id} instruction aggregation {accumulated} + {addend} overflows u64")]
    InstructionCountOverflow {
        sm_id: i32,
        accumulated: u64,
        addend: u64,
    },
    #[error("duplicate block launch index {index} across threadblock files")]
    DuplicateLaunchIndex { index: i64 },
    #[error(
        "instrumentation_order token {order} is duplicated within one launch (SM {sm_id}, block {launch_index}, warp slot {warp_slot}, instruction {instruction})"
    )]
    InstrumentationOrderDuplicate {
        order: u64,
        sm_id: i32,
        launch_index: i64,
        warp_slot: i32,
        instruction: usize,
    },
    #[error("instrumentation_order is not dense and 1-based: expected token {expected}, got {got}")]
    InstrumentationOrderNotDense { expected: u64, got: u64 },
    #[error(
        "selected asynchronous-copy record lacks instrumentation_order (SM {sm_id}, block {launch_index}, warp slot {warp_slot}, instruction {instruction}, pc {pc:#x})"
    )]
    InstrumentationOrderMissing {
        sm_id: i32,
        launch_index: i64,
        warp_slot: i32,
        instruction: usize,
        pc: u32,
    },
    #[error(
        "non-selected record carries instrumentation_order={order} (SM {sm_id}, block {launch_index}, warp slot {warp_slot}, instruction {instruction}, pc {pc:#x})"
    )]
    InstrumentationOrderUnexpected {
        order: u64,
        sm_id: i32,
        launch_index: i64,
        warp_slot: i32,
        instruction: usize,
        pc: u32,
    },
    #[error(
        "shared-system unmodeled-tail aggregate mismatch: system={system:?}, per_sm={per_sm:?}"
    )]
    SystemTailMismatch {
        system: TailFlags,
        per_sm: TailFlags,
    },
    #[error(
        "issued {issued} != records {records} — replay invariant broken (every traced record \
         must issue exactly once)"
    )]
    CountMismatch { issued: u64, records: u64 },
    #[error("kernel launch grid {grid:?} overflows the u64 block-count domain")]
    GridSizeOverflow { grid: (u32, u32, u32) },
    #[error("incomplete trace: {seen}/{grid_total} launch blocks present")]
    IncompleteTrace { seen: u64, grid_total: u64 },
    #[error("fast frontier analysis failed: {detail}")]
    FastFrontier { detail: String },
    #[cfg(feature = "campaign")]
    #[error(
        "trace_sampling sidecar original_grid {sidecar:?} != launch gridDim {launch:?} \
         (a subset must NEVER rewrite gridDim)"
    )]
    GridDimRewritten {
        launch: (u32, u32, u32),
        sidecar: (u32, u32, u32),
    },
    #[cfg(feature = "campaign")]
    #[error("records_seen {records_seen} != trace_sampling sidecar records.captured {captured}")]
    RecordsCapturedMismatch { records_seen: u64, captured: u64 },
    #[cfg(feature = "campaign")]
    #[error(
        "trace_sampling sidecar selected block membership does not match replayed threadblocks \
         ({seen} seen, {declared} declared)"
    )]
    SelectedBlockMembershipMismatch { seen: u64, declared: u64 },
    #[cfg(feature = "campaign")]
    #[error("trace_sampling sidecar kernel {sidecar:?} != launch kernel {launch:?}")]
    SamplingKernelMismatch { launch: String, sidecar: String },
    #[cfg(feature = "campaign")]
    #[error("kernel launch block_dim is absent or degenerate")]
    BlockDimMissing,
    #[error("kernel launch cluster_dim is partially populated or has a negative axis: {cluster:?}")]
    ClusterDimInvalid { cluster: (i32, i32, i32) },
    #[error(
        "kernel launch cluster geometry {cluster:?} is not modeled for architecture {arch}; only (1, 1, 1) is admissible"
    )]
    ClusterGeometryUnsupported {
        arch: ArchId,
        cluster: (u32, u32, u32),
    },
    #[cfg(feature = "campaign")]
    #[error("trace_sampling sidecar original_block {sidecar:?} != launch blockDim {launch:?}")]
    BlockDimRewritten {
        launch: (u32, u32, u32),
        sidecar: (u32, u32, u32),
    },
}

/// Launch identity and shape bound into a verified replay-coverage witness.
/// Declaration-side construction creates only an expected identity; the extraction
/// completeness gate remains the only constructor of observed coverage witnesses.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ReplayLaunchCoverageBinding {
    device_id: i64,
    stream_id: i64,
    kernel_id: i32,
    function_unique_id: i32,
    kernel_mangled_name: String,
    grid: (u32, u32, u32),
    block: Option<(u32, u32, u32)>,
    cluster: (u32, u32, u32),
}

impl ReplayLaunchCoverageBinding {
    pub fn device_id(&self) -> i64 {
        self.device_id
    }

    pub fn stream_id(&self) -> i64 {
        self.stream_id
    }

    pub fn kernel_id(&self) -> i32 {
        self.kernel_id
    }

    pub fn function_unique_id(&self) -> i32 {
        self.function_unique_id
    }

    pub fn kernel_mangled_name(&self) -> &str {
        &self.kernel_mangled_name
    }

    pub fn grid(&self) -> (u32, u32, u32) {
        self.grid
    }

    pub fn block(&self) -> Option<(u32, u32, u32)> {
        self.block
    }

    pub fn cluster(&self) -> (u32, u32, u32) {
        self.cluster
    }
}

/// Provenance by which an observed complete launch became full-coverage replay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FullCoverageSource {
    /// No sampling declaration was present; the replay gate observed every CTA.
    UnsampledComplete,
    /// A validated lossless sidecar was present and replay independently observed
    /// every CTA. The sidecar declaration alone never constructs this variant.
    #[cfg(feature = "campaign")]
    LosslessCompressedFull(LosslessFullProvenance),
}

/// Unforgeable-by-consumers witness that replay observed every block in one launch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FullLaunchCoverage {
    launch: ReplayLaunchCoverageBinding,
    blocks_observed: u64,
    source: FullCoverageSource,
}

impl FullLaunchCoverage {
    pub fn launch(&self) -> &ReplayLaunchCoverageBinding {
        &self.launch
    }

    pub fn blocks_observed(&self) -> u64 {
        self.blocks_observed
    }

    pub fn source(&self) -> &FullCoverageSource {
        &self.source
    }
}

/// Unforgeable-by-consumers witness that replay observed exactly the block roster
/// sealed by one validated subset sidecar.
#[cfg(feature = "campaign")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampledLaunchCoverage {
    launch: ReplayLaunchCoverageBinding,
    population_blocks: u64,
    blocks_observed: u64,
    provenance: WaveSegmentSubsetProvenance,
}

#[cfg(feature = "campaign")]
impl SampledLaunchCoverage {
    pub fn launch(&self) -> &ReplayLaunchCoverageBinding {
        &self.launch
    }

    pub fn population_blocks(&self) -> u64 {
        self.population_blocks
    }

    pub fn blocks_observed(&self) -> u64 {
        self.blocks_observed
    }

    pub fn provenance(&self) -> &WaveSegmentSubsetProvenance {
        &self.provenance
    }
}

/// Timing-bearing replay payload kept private so coverage cannot be detached or
/// fabricated at the public API boundary.
#[derive(Debug, Clone)]
struct FoldedKernelReplay {
    arch: ArchId,
    instructions_issued: u64,
    records_seen: u64,
    per_sm: BTreeMap<i32, SmSimResult>,
    diagnostic_last_issue_cycles: u64,
    last_processed_system_event_time: Option<u64>,
    terminal_modeled_event_time: Option<u64>,
    sm_active_terminal_event_time: Option<u64>,
    l1: Option<L1Counters>,
    l2: Option<L2Counters>,
    l2_by_source: Option<L2SourceSplit>,
    dram: Option<DramStats>,
    front_end: FrontEndCounters,
    trace_metrics: TraceMetrics,
    unmodeled_tails: TailFlags,
}

/// Trace-derived compulsory front-end traffic. Every distinct executed 32-byte
/// SASS sector must enter the instruction/constant hierarchy at least once;
/// this observer is independent of the data-cache/L2 timing state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrontEndCounters {
    compulsory_instruction_read_sectors: u64,
}

impl FrontEndCounters {
    pub const fn compulsory_instruction_read_sectors(self) -> u64 {
        self.compulsory_instruction_read_sectors
    }

    fn add_checked(self, other: Self) -> Option<Self> {
        Some(Self {
            compulsory_instruction_read_sectors: self
                .compulsory_instruction_read_sectors
                .checked_add(other.compulsory_instruction_read_sectors)?,
        })
    }
}

/// Completion knowledge for one fully replayed launch.
///
/// `Complete` means every timing-bearing effect known to the replay model is
/// represented in its event stream. `Incomplete` preserves the event frontier
/// only as a lower bound and names the known retirement tails that remain
/// outside that stream. Neither variant includes the profiler launch envelope
/// required for `gpc__cycles_elapsed.max`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModeledCompletionFrontier {
    Complete {
        cycles: Option<u64>,
    },
    Incomplete {
        lower_bound_cycles: Option<u64>,
        unmodeled_tails: TailFlags,
    },
}

impl ModeledCompletionFrontier {
    pub const fn lower_bound_cycles(self) -> Option<u64> {
        match self {
            Self::Complete { cycles } => cycles,
            Self::Incomplete {
                lower_bound_cycles, ..
            } => lower_bound_cycles,
        }
    }

    pub const fn complete_cycles(self) -> Option<u64> {
        match self {
            Self::Complete { cycles } => cycles,
            Self::Incomplete { .. } => None,
        }
    }

    pub const fn unmodeled_tails(self) -> TailFlags {
        match self {
            Self::Complete { .. } => TailFlags {
                red_rmw: false,
                dirty_writeback: false,
                coherence: false,
                ldgsts_dst_shared_timing: false,
            },
            Self::Incomplete {
                unmodeled_tails, ..
            } => unmodeled_tails,
        }
    }

    pub const fn is_complete(self) -> bool {
        matches!(self, Self::Complete { .. })
    }
}

/// Full-coverage replay result. This is the only public replay result carrying
/// per-SM state or temporal diagnostics.
#[derive(Debug, Clone)]
pub struct FullKernelReplayResult {
    coverage: FullLaunchCoverage,
    configuration: ReplayConfiguration,
    replay: FoldedKernelReplay,
}

/// A separate, architecture-scoped GPC observation frontier.  The underlying
/// modeled event frontier remains available as `lower_bound_cycles`; this
/// projection adds only the independently measured SM86 front-end/semaphore
/// envelope and never consumes an NCU target or the timing crosswalk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpcObservationFrontier {
    pub lower_bound_cycles: u64,
    pub envelope_cycles: u64,
    pub grid_blocks: u64,
    pub grid_log2: u32,
    pub cycles: u64,
}

/// SM-domain physical observation frontier. The replay's issue and terminal
/// timestamps remain the unadjusted modeled lower bounds; this explicit
/// projection adds only the independently measured SM-active observation
/// boundary and retains both values for auditability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmActiveObservationFrontier {
    pub issue_lower_bound_cycles: u64,
    pub issue_cycles: u64,
    pub terminal_lower_bound_cycles: Option<u64>,
    pub terminal_cycles: Option<u64>,
    pub envelope_cycles: u64,
}

/// Fast-path evidence produced directly from an admitted SM86 trace. This is
/// deliberately a diagnostic projection: only homogeneous, control-clean,
/// dependency-free scalar regions with no memory role are offered to the
/// causal-frontier certificate; every other warp is retained as detailed or
/// refused. It does not replace native replay timing or prove silicon behavior.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FastFrontierSummary {
    pub contract: &'static str,
    pub architecture: &'static str,
    pub blocks_seen: u64,
    pub warps_seen: u64,
    pub eligible_warps: u64,
    pub candidate_regions: u64,
    pub certified_regions: u64,
    pub reified_regions: u64,
    pub candidate_instructions: u64,
    pub collapsed_instructions: u64,
    pub exact_execution_steps: u64,
    pub summarized_execution_steps: u64,
    pub maximum_exact_final_cycle: u64,
    pub maximum_summarized_final_cycle: u64,
    pub terminal_state_matches: bool,
    pub refusal_reasons: BTreeMap<String, u64>,
}

fn fast_scalar_segments(
    warp: &WarpReplay,
    schedule: &FunctionSchedule,
) -> Vec<Vec<CausalFrontierInstruction>> {
    let mut segments = Vec::new();
    let mut program = Vec::new();
    for (sequence, instruction) in warp.insts.iter().enumerate() {
        let descriptor = match (|| {
            if !matches!(instruction.kind, AccessKind::NonMemory) {
                return None;
            }
            let slot = schedule.issue_slot(u64::from(instruction.pc))?;
            if slot.op_class() != OpClass::Other || !matches!(slot.mem_role(), MemRole::None) {
                return None;
            }
            let PipelineClass::ScalarAlu { latency_cycles, .. } = slot.pipeline_class() else {
                return None;
            };
            let control = slot.control();
            let dependency_free = control.wait_mask().bits() == 0
                && control.write_barrier().is_none()
                && control.read_barrier().is_none();
            let sequence = u64::try_from(sequence).ok()?;
            CausalFrontierInstruction::from_schedule(
                schedule,
                u64::from(instruction.pc),
                sequence,
                warp.warp_slot,
                latency_cycles,
                1,
                dependency_free,
                false,
            )
            .ok()
        })() {
            Some(descriptor) => descriptor,
            None => {
                if program.len() >= 2 {
                    segments.push(std::mem::take(&mut program));
                } else {
                    program.clear();
                }
                continue;
            }
        };
        program.push(descriptor);
    }
    if program.len() >= 2 {
        segments.push(program);
    }
    segments
}

fn increment_fast_reason(
    reasons: &mut BTreeMap<String, u64>,
    reason: &str,
) -> Result<(), ReplayError> {
    let value = reasons.entry(reason.to_owned()).or_default();
    *value = value
        .checked_add(1)
        .ok_or_else(|| ReplayError::FastFrontier {
            detail: format!("refusal reason {reason:?} overflows u64"),
        })?;
    Ok(())
}

/// Build a conservative fast-frontier report from a real compact trace. The
/// producer bridge intentionally refuses memory, barriers, dynamic pipelines,
/// and incomplete scalar regions; callers can use the summary to decide which
/// ranges are safe to summarize while keeping the ordinary replay as fallback.
pub fn fast_frontier_for_trace(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedule: &FunctionSchedule,
    arch_card: &ArchCard,
) -> Result<FastFrontierSummary, ReplayError> {
    let placement = PlacementConfig::from_arch_card(arch_card, schedule.arch)?;
    require_trace_architecture(td, schedule.arch)?;
    let extracted = extract_per_sm_blocks(td, loc, schedule, placement, false)?;
    let mut summary = FastFrontierSummary {
        contract: "sm86_fast_exact_scalar_regions_v1",
        architecture: schedule.arch.token(),
        blocks_seen: 0,
        warps_seen: 0,
        eligible_warps: 0,
        candidate_regions: 0,
        certified_regions: 0,
        reified_regions: 0,
        candidate_instructions: 0,
        collapsed_instructions: 0,
        exact_execution_steps: 0,
        summarized_execution_steps: 0,
        maximum_exact_final_cycle: 0,
        maximum_summarized_final_cycle: 0,
        terminal_state_matches: true,
        refusal_reasons: BTreeMap::new(),
    };
    let add = |counter: &'static str, value: &mut u64, addend: u64| {
        *value = value
            .checked_add(addend)
            .ok_or_else(|| ReplayError::FastFrontier {
                detail: format!("{counter} overflows u64"),
            })?;
        Ok::<(), ReplayError>(())
    };
    for blocks in extracted.per_sm_blocks.values() {
        for block in blocks {
            add("blocks_seen", &mut summary.blocks_seen, 1)?;
            let residents = block
                .warps
                .iter()
                .map(|warp| warp.warp_slot)
                .collect::<Vec<_>>();
            let mut candidate_segments = Vec::new();
            for warp in &block.warps {
                add("warps_seen", &mut summary.warps_seen, 1)?;
                let warp_segments = fast_scalar_segments(warp, schedule);
                if warp_segments.is_empty() {
                    increment_fast_reason(
                        &mut summary.refusal_reasons,
                        "memory_control_or_non_scalar",
                    )?;
                    continue;
                }
                add("eligible_warps", &mut summary.eligible_warps, 1)?;
                for segment in warp_segments {
                    // A single segment is analyzed at a time so disjoint
                    // segments on the same warp never masquerade as one
                    // contiguous certificate. The complete resident roster
                    // still prevents certification when another resident warp
                    // can compete on that SMSP.
                    candidate_segments.push(segment);
                }
            }
            for segment in candidate_segments {
                let segment_len =
                    u64::try_from(segment.len()).map_err(|_| ReplayError::FastFrontier {
                        detail: "scalar segment length exceeds u64".to_owned(),
                    })?;
                let report: CausalFrontierReport = summarize_with_residency(
                    &[segment],
                    CausalFrontierResidency::Complete {
                        physical_warps: residents.clone(),
                    },
                )
                .map_err(|error| ReplayError::FastFrontier {
                    detail: error.to_string(),
                })?;
                // The generic analyzer only increments candidate_regions after
                // it sees a fully summarizable interval. Keep a producer-side
                // candidate/refusal witness for scalar runs rejected by a
                // control/dependency/occupancy guard, so fast-only output does
                // not confuse "no candidate" with "no scalar work".
                if report.candidate_regions == 0 {
                    add("candidate_regions", &mut summary.candidate_regions, 1)?;
                    add("reified_regions", &mut summary.reified_regions, 1)?;
                    add(
                        "candidate_instructions",
                        &mut summary.candidate_instructions,
                        segment_len,
                    )?;
                    increment_fast_reason(&mut summary.refusal_reasons, "causal_guard_refused")?;
                }
                add(
                    "candidate_regions",
                    &mut summary.candidate_regions,
                    report.candidate_regions,
                )?;
                add(
                    "certified_regions",
                    &mut summary.certified_regions,
                    u64::try_from(report.certified_regions.len()).map_err(|_| {
                        ReplayError::FastFrontier {
                            detail: "certified region count exceeds u64".to_owned(),
                        }
                    })?,
                )?;
                add(
                    "reified_regions",
                    &mut summary.reified_regions,
                    report.reified_regions,
                )?;
                add(
                    "candidate_instructions",
                    &mut summary.candidate_instructions,
                    report.candidate_instructions,
                )?;
                add(
                    "collapsed_instructions",
                    &mut summary.collapsed_instructions,
                    report.collapsed_instructions,
                )?;
                add(
                    "exact_execution_steps",
                    &mut summary.exact_execution_steps,
                    report.exact_execution_steps,
                )?;
                add(
                    "summarized_execution_steps",
                    &mut summary.summarized_execution_steps,
                    report.causal_execution_steps,
                )?;
                summary.maximum_exact_final_cycle = summary
                    .maximum_exact_final_cycle
                    .max(report.exact_final_cycle);
                summary.maximum_summarized_final_cycle = summary
                    .maximum_summarized_final_cycle
                    .max(report.causal_final_cycle);
                summary.terminal_state_matches &= report.terminal_state_matches;
                for detail in report.reified_region_details {
                    increment_fast_reason(
                        &mut summary.refusal_reasons,
                        &format!("reified_{:?}", detail.reason),
                    )?;
                }
            }
        }
    }
    Ok(summary)
}

impl FullKernelReplayResult {
    pub fn coverage(&self) -> &FullLaunchCoverage {
        &self.coverage
    }

    pub fn configuration(&self) -> ReplayConfiguration {
        self.configuration
    }

    pub fn co_residency(&self) -> CoResidency {
        self.configuration.co_residency()
    }

    pub fn arch(&self) -> ArchId {
        self.replay.arch
    }

    pub fn instructions_issued(&self) -> u64 {
        self.replay.instructions_issued
    }

    pub fn records_seen(&self) -> u64 {
        self.replay.records_seen
    }

    pub fn per_sm(&self) -> &BTreeMap<i32, SmSimResult> {
        &self.replay.per_sm
    }

    /// Maximum per-SM issue frontier (`last local issue timestamp + 1`). This
    /// is the unadjusted causal scheduler/event lower bound. Consumers that
    /// need the silicon `sm__cycles_active` observation domain must use the
    /// explicit [`Self::sm_active_observation_frontier`] projection.
    pub fn diagnostic_last_issue_cycles(&self) -> u64 {
        self.replay.diagnostic_last_issue_cycles
    }

    /// Unadjusted issue frontier used by causal replay diagnostics. The value
    /// is kept separate from the architecture-scoped observation projection;
    /// it is never a hardware target or a fitted correction.
    pub fn modeled_issue_lower_bound_cycles(&self) -> u64 {
        self.replay.diagnostic_last_issue_cycles
    }

    /// Raw timestamp of the final successfully processed shared-system event,
    /// before it is combined with the issue frontier.
    pub fn last_processed_modeled_system_event_time(&self) -> Option<u64> {
        self.replay.last_processed_system_event_time
    }

    /// Maximum timestamp covered by the modeled event set: the checked issue
    /// frontier (`last local issue timestamp + 1`) and any successfully
    /// processed shared-system event. This remains an unadjusted lower bound;
    /// the explicit SM-active projection adds the observation boundary.
    pub fn terminal_modeled_event_time(&self) -> Option<u64> {
        self.replay.terminal_modeled_event_time
    }

    /// Unadjusted terminal modeled-event lower bound.
    pub fn modeled_terminal_lower_bound_cycles(&self) -> Option<u64> {
        self.replay.terminal_modeled_event_time
    }

    /// SM-active terminal lower bound with post-issue MemoryWriteAck events
    /// excluded from the observation endpoint.  Acknowledgement events remain
    /// in the full modeled frontier and still affect any dependent issue; this
    /// accessor only keeps the two observation domains distinct.
    pub fn sm_active_terminal_event_time(&self) -> Option<u64> {
        self.replay.sm_active_terminal_event_time
    }

    /// Return the architecture-modeled GPC observation frontier for a full
    /// shared-L2 replay.  `None` is reserved for a non-shared compatibility
    /// mode, an empty frontier, an unsupported architecture capability, or a
    /// checked arithmetic overflow; the physical lower-bound field above is
    /// never replaced or hidden.
    pub fn gpc_observation_frontier(&self) -> Option<GpcObservationFrontier> {
        if !matches!(
            self.configuration.memory_mode,
            ReplayMemMode::L1SharedL2 { .. }
        ) {
            return None;
        }
        let lower_bound_cycles = self.modeled_terminal_lower_bound_cycles()?;
        let grid_blocks = self.coverage.launch().grid_blocks().ok()?;
        let envelope = self
            .arch()
            .gpc_observation_envelope()
            .require("gpc_observation_envelope")
            .ok()?;
        let (envelope_cycles, grid_log2) = envelope.cycles_for_grid(grid_blocks)?;
        let cycles = lower_bound_cycles.checked_add(envelope_cycles)?;
        Some(GpcObservationFrontier {
            lower_bound_cycles,
            envelope_cycles,
            grid_blocks,
            grid_log2,
            cycles,
        })
    }

    /// Return the explicit SM-active physical observation frontier for a
    /// shared-L2 replay. The replay clock remains unadjusted; this projection
    /// adds the architecture-scoped observation boundary while retaining the
    /// causal lower bounds in the same structure.
    pub fn sm_active_observation_frontier(&self) -> Option<SmActiveObservationFrontier> {
        if !matches!(
            self.configuration.memory_mode,
            ReplayMemMode::L1SharedL2 { .. }
        ) {
            return None;
        }
        let issue_lower_bound_cycles = self.modeled_issue_lower_bound_cycles();
        let envelope_cycles = self.configuration.sm_active_activation_cycles();
        if issue_lower_bound_cycles == 0 && self.replay.terminal_modeled_event_time.is_none() {
            return None;
        }
        let issue_cycles = issue_lower_bound_cycles.checked_add(envelope_cycles)?;
        let terminal_lower_bound_cycles = self.modeled_terminal_lower_bound_cycles();
        let terminal_cycles =
            terminal_lower_bound_cycles.and_then(|cycles| cycles.checked_add(envelope_cycles));
        Some(SmActiveObservationFrontier {
            issue_lower_bound_cycles,
            issue_cycles,
            terminal_lower_bound_cycles,
            terminal_cycles,
            envelope_cycles,
        })
    }

    /// Typed launch-retirement diagnostic. Consumers must distinguish a
    /// complete modeled frontier from a lower bound with known missing tails.
    pub fn modeled_completion_frontier(&self) -> ModeledCompletionFrontier {
        if self.replay.unmodeled_tails.is_empty() {
            ModeledCompletionFrontier::Complete {
                cycles: self.modeled_terminal_lower_bound_cycles(),
            }
        } else {
            ModeledCompletionFrontier::Incomplete {
                lower_bound_cycles: self.modeled_terminal_lower_bound_cycles(),
                unmodeled_tails: self.replay.unmodeled_tails,
            }
        }
    }

    pub fn l1(&self) -> Option<&L1Counters> {
        self.replay.l1.as_ref()
    }

    pub fn l2(&self) -> Option<&L2Counters> {
        self.replay.l2.as_ref()
    }

    pub fn l2_by_source(&self) -> Option<&L2SourceSplit> {
        self.replay.l2_by_source.as_ref()
    }

    pub fn dram(&self) -> Option<&DramStats> {
        self.replay.dram.as_ref()
    }

    pub fn front_end(&self) -> FrontEndCounters {
        self.replay.front_end
    }

    /// Trace-derived dynamic activity and memory-class counts. These are
    /// independent of NCU and never feed timing configuration.
    pub fn trace_metrics(&self) -> &TraceMetrics {
        &self.replay.trace_metrics
    }

    pub fn unmodeled_tails(&self) -> TailFlags {
        self.replay.unmodeled_tails
    }
}

/// Counter-only observation over a verified sampled block roster. This type is
/// intentionally distinct from the full replay payload and contains no per-SM
/// result, timestamp, cycle count, or timing-result conversion.
#[cfg(feature = "campaign")]
#[derive(Debug, Clone)]
pub struct SampledObservedCounters {
    arch: ArchId,
    instructions_issued: u64,
    records_seen: u64,
    l1: Option<L1Counters>,
    l2: Option<L2Counters>,
    l2_by_source: Option<L2SourceSplit>,
    dram: Option<DramStats>,
    front_end: FrontEndCounters,
    unmodeled_tails: TailFlags,
}

#[cfg(feature = "campaign")]
impl SampledObservedCounters {
    pub fn arch(&self) -> ArchId {
        self.arch
    }

    pub fn instructions_issued(&self) -> u64 {
        self.instructions_issued
    }

    pub fn records_seen(&self) -> u64 {
        self.records_seen
    }

    pub fn l1(&self) -> Option<&L1Counters> {
        self.l1.as_ref()
    }

    pub fn l2(&self) -> Option<&L2Counters> {
        self.l2.as_ref()
    }

    pub fn l2_by_source(&self) -> Option<&L2SourceSplit> {
        self.l2_by_source.as_ref()
    }

    pub fn dram(&self) -> Option<&DramStats> {
        self.dram.as_ref()
    }

    pub fn front_end(&self) -> FrontEndCounters {
        self.front_end
    }

    pub fn unmodeled_tails(&self) -> TailFlags {
        self.unmodeled_tails
    }
}

/// Sampled-observed replay result. No method converts this value into a full or
/// timing-bearing replay result.
#[cfg(feature = "campaign")]
#[derive(Debug, Clone)]
pub struct SampledKernelReplayResult {
    coverage: SampledLaunchCoverage,
    configuration: ReplayConfiguration,
    counters: SampledObservedCounters,
}

#[cfg(feature = "campaign")]
impl SampledKernelReplayResult {
    pub fn coverage(&self) -> &SampledLaunchCoverage {
        &self.coverage
    }

    pub fn configuration(&self) -> ReplayConfiguration {
        self.configuration
    }

    pub fn counters(&self) -> &SampledObservedCounters {
        &self.counters
    }
}

/// Coverage-classified outcome for one kernel launch.
#[derive(Debug, Clone)]
pub enum KernelReplayOutcome {
    Full(FullKernelReplayResult),
    #[cfg(feature = "campaign")]
    SampledObserved(SampledKernelReplayResult),
}

impl KernelReplayOutcome {
    fn l2_counters(&self) -> Option<&L2Counters> {
        match self {
            Self::Full(result) => result.l2(),
            #[cfg(feature = "campaign")]
            Self::SampledObserved(result) => result.counters().l2(),
        }
    }

    /// Consume a coverage-classified result only when it represents the full
    /// launch. Keeping this match inside `sim-sm` makes ordinary callers stable
    /// when Cargo feature unification enables the campaign-only sampled
    /// variant elsewhere in the workspace.
    pub fn into_full(self) -> Option<FullKernelReplayResult> {
        match self {
            Self::Full(result) => Some(result),
            #[cfg(feature = "campaign")]
            Self::SampledObserved(_) => None,
        }
    }
}

/// Coverage classification retained by an explicitly unadmitted diagnostic replay.
#[cfg(feature = "campaign")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnadmittedCoverage {
    Full,
    SampledObserved,
}

/// Read-only diagnostics from raw trace/sidecar inputs. This type deliberately
/// cannot be converted into [`FullKernelReplayResult`] or passed to
/// [`aggregate_full_launch_set`]; only [`replay_kernel`] over
/// [`AdmittedLaunchEvidence`] can mint an aggregatable replay result.
///
/// ```compile_fail
/// use sim_sm::{
///     FullLaunchSetDeclaration, UnadmittedReplayDiagnostics,
///     aggregate_full_launch_set,
/// };
///
/// fn reject(
///     declaration: FullLaunchSetDeclaration,
///     diagnostic: UnadmittedReplayDiagnostics,
/// ) {
///     let _ = aggregate_full_launch_set(declaration, [diagnostic]);
/// }
/// ```
#[cfg(feature = "campaign")]
#[derive(Debug, Clone)]
pub struct UnadmittedReplayDiagnostics {
    coverage: UnadmittedCoverage,
    configuration: ReplayConfiguration,
    arch: ArchId,
    blocks_observed: u64,
    population_blocks: u64,
    instructions_issued: u64,
    records_seen: u64,
    per_sm_count: Option<usize>,
    diagnostic_last_issue_cycles: Option<u64>,
    terminal_modeled_event_time: Option<u64>,
    l1: Option<L1Counters>,
    l2: Option<L2Counters>,
    l2_by_source: Option<L2SourceSplit>,
    dram: Option<DramStats>,
    unmodeled_tails: TailFlags,
}

#[cfg(feature = "campaign")]
impl UnadmittedReplayDiagnostics {
    fn from_outcome(outcome: KernelReplayOutcome) -> Self {
        match outcome {
            KernelReplayOutcome::Full(result) => Self {
                coverage: UnadmittedCoverage::Full,
                configuration: result.configuration(),
                arch: result.arch(),
                blocks_observed: result.coverage().blocks_observed(),
                population_blocks: result.coverage().blocks_observed(),
                instructions_issued: result.instructions_issued(),
                records_seen: result.records_seen(),
                per_sm_count: Some(result.per_sm().len()),
                diagnostic_last_issue_cycles: Some(result.diagnostic_last_issue_cycles()),
                terminal_modeled_event_time: result.terminal_modeled_event_time(),
                l1: result.l1().copied(),
                l2: result.l2().copied(),
                l2_by_source: result.l2_by_source().copied(),
                dram: result.dram().copied(),
                unmodeled_tails: result.unmodeled_tails(),
            },
            KernelReplayOutcome::SampledObserved(result) => {
                let counters = result.counters();
                Self {
                    coverage: UnadmittedCoverage::SampledObserved,
                    configuration: result.configuration(),
                    arch: counters.arch(),
                    blocks_observed: result.coverage().blocks_observed(),
                    population_blocks: result.coverage().population_blocks(),
                    instructions_issued: counters.instructions_issued(),
                    records_seen: counters.records_seen(),
                    per_sm_count: None,
                    diagnostic_last_issue_cycles: None,
                    terminal_modeled_event_time: None,
                    l1: counters.l1().copied(),
                    l2: counters.l2().copied(),
                    l2_by_source: counters.l2_by_source().copied(),
                    dram: counters.dram().copied(),
                    unmodeled_tails: counters.unmodeled_tails(),
                }
            }
        }
    }

    pub fn coverage(&self) -> UnadmittedCoverage {
        self.coverage
    }

    pub fn configuration(&self) -> ReplayConfiguration {
        self.configuration
    }

    pub fn arch(&self) -> ArchId {
        self.arch
    }

    pub fn blocks_observed(&self) -> u64 {
        self.blocks_observed
    }

    pub fn population_blocks(&self) -> u64 {
        self.population_blocks
    }

    pub fn instructions_issued(&self) -> u64 {
        self.instructions_issued
    }

    pub fn records_seen(&self) -> u64 {
        self.records_seen
    }

    pub fn per_sm_count(&self) -> Option<usize> {
        self.per_sm_count
    }

    /// Present only for full-coverage diagnostics. This remains an issue-frontier
    /// diagnostic, never a publication-cycle prediction.
    pub fn diagnostic_last_issue_cycles(&self) -> Option<u64> {
        self.diagnostic_last_issue_cycles
    }

    pub fn terminal_modeled_event_time(&self) -> Option<u64> {
        self.terminal_modeled_event_time
    }

    pub fn l1(&self) -> Option<&L1Counters> {
        self.l1.as_ref()
    }

    pub fn l2(&self) -> Option<&L2Counters> {
        self.l2.as_ref()
    }

    pub fn l2_by_source(&self) -> Option<&L2SourceSplit> {
        self.l2_by_source.as_ref()
    }

    pub fn dram(&self) -> Option<&DramStats> {
        self.dram.as_ref()
    }

    pub fn unmodeled_tails(&self) -> TailFlags {
        self.unmodeled_tails
    }
}

#[derive(Debug)]
pub(crate) enum ExtractedCoverage {
    Full(FullLaunchCoverage),
    #[cfg(feature = "campaign")]
    Sampled(SampledLaunchCoverage),
}

#[derive(Debug)]
struct ExtractedKernel {
    per_sm_blocks: BTreeMap<i32, Vec<BlockReplay>>,
    records_seen: u64,
    coverage: ExtractedCoverage,
    shared_memory_residency: SharedMemoryResidency,
    front_end: FrontEndCounters,
    trace_metrics: TraceMetrics,
}

#[derive(Debug, Clone, Copy)]
enum OrderedAsyncEventKind {
    Issue { effective: bool },
    Commit,
    Wait { max_pending_groups: u32 },
}

#[derive(Debug, Clone, Copy)]
struct OrderedAsyncEvent {
    order: u64,
    launch_index: i64,
    warp_slot: i32,
    kind: OrderedAsyncEventKind,
}

#[derive(Debug, Clone, Copy, Default)]
struct ObservedAsyncWarpState {
    open_effective_copies: u64,
    committed_groups: u64,
}

fn ordered_async_event_kind(
    schedule: &FunctionSchedule,
    instruction: ReplayInst,
) -> Option<OrderedAsyncEventKind> {
    let slot = schedule
        .by_pc
        .get(&u64::from(instruction.pc))
        .unwrap_or_else(|| unreachable!("extraction already validated the complete schedule join"));
    if slot.ldgsts_l1_policy().is_some() {
        return Some(OrderedAsyncEventKind::Issue {
            effective: matches!(instruction.kind, AccessKind::GlobalAsyncCopy { .. }),
        });
    }
    match slot.op_class() {
        OpClass::AsyncCopyCommit => Some(OrderedAsyncEventKind::Commit),
        OpClass::AsyncCopyWait => Some(OrderedAsyncEventKind::Wait {
            max_pending_groups: slot
                .async_copy_wait_max_pending_groups()
                .expect("async-copy wait slot carries its parsed immediate"),
        }),
        OpClass::BlockBarrier | OpClass::Other => None,
    }
}

fn checked_order_observation_add(
    counter: &'static str,
    lhs: u64,
    rhs: u64,
) -> Result<u64, ReplayError> {
    lhs.checked_add(rhs)
        .ok_or(IssueError::CounterOverflow { counter }.into())
}

/// Validate the opt-in selected-record order contract and reconstruct only the
/// asynchronous-copy group populations it directly observes. The returned map
/// is attached to readiness after simulation; neither this scan nor its result
/// is visible to issue selection or any timing/service state.
fn observe_async_instrumentation_order(
    per_sm_blocks: &BTreeMap<i32, Vec<BlockReplay>>,
    schedule: &FunctionSchedule,
) -> Result<BTreeMap<i32, AsyncInstrumentationOrderDiagnostics>, ReplayError> {
    let observation_enabled = per_sm_blocks.values().flatten().any(|block| {
        block.warps.iter().any(|warp| {
            warp.async_copy_cohorts
                .as_deref()
                .is_some_and(|sidecar| !sidecar.instrumentation_order.is_empty())
        })
    });
    if !observation_enabled {
        return Ok(BTreeMap::new());
    }

    let mut seen_orders = BTreeSet::new();
    let mut events_by_sm: BTreeMap<i32, Vec<OrderedAsyncEvent>> = BTreeMap::new();
    for (&sm_id, blocks) in per_sm_blocks {
        for block in blocks {
            for warp in &block.warps {
                let anchors = warp
                    .async_copy_cohorts
                    .as_deref()
                    .map_or(&[][..], |sidecar| sidecar.instrumentation_order.as_slice());
                let mut anchor_index = 0;
                for (instruction_index, &instruction) in warp.insts.iter().enumerate() {
                    let anchor = anchors
                        .get(anchor_index)
                        .filter(|anchor| anchor.instruction == instruction_index)
                        .copied();
                    if anchor.is_some() {
                        anchor_index += 1;
                    }
                    let selected = ordered_async_event_kind(schedule, instruction);
                    match (selected, anchor) {
                        (Some(kind), Some(anchor)) => {
                            if !seen_orders.insert(anchor.order) {
                                return Err(ReplayError::InstrumentationOrderDuplicate {
                                    order: anchor.order,
                                    sm_id,
                                    launch_index: block.launch_index,
                                    warp_slot: warp.warp_slot,
                                    instruction: instruction_index,
                                });
                            }
                            events_by_sm
                                .entry(sm_id)
                                .or_default()
                                .push(OrderedAsyncEvent {
                                    order: anchor.order,
                                    launch_index: block.launch_index,
                                    warp_slot: warp.warp_slot,
                                    kind,
                                });
                        }
                        (Some(_), None) => {
                            return Err(ReplayError::InstrumentationOrderMissing {
                                sm_id,
                                launch_index: block.launch_index,
                                warp_slot: warp.warp_slot,
                                instruction: instruction_index,
                                pc: instruction.pc,
                            });
                        }
                        (None, Some(anchor)) => {
                            return Err(ReplayError::InstrumentationOrderUnexpected {
                                order: anchor.order,
                                sm_id,
                                launch_index: block.launch_index,
                                warp_slot: warp.warp_slot,
                                instruction: instruction_index,
                                pc: instruction.pc,
                            });
                        }
                        (None, None) => {}
                    }
                }
                debug_assert_eq!(
                    anchor_index,
                    anchors.len(),
                    "compact reader anchor index escaped its instruction stream"
                );
            }
        }
    }

    for (offset, &got) in seen_orders.iter().enumerate() {
        let expected = u64::try_from(offset)
            .ok()
            .and_then(|offset| offset.checked_add(1))
            .ok_or(IssueError::CounterOverflow {
                counter: "instrumentation order density",
            })?;
        if got != expected {
            return Err(ReplayError::InstrumentationOrderNotDense { expected, got });
        }
    }

    let mut observations = BTreeMap::new();
    for (sm_id, mut events) in events_by_sm {
        events.sort_unstable_by_key(|event| event.order);
        let mut states: BTreeMap<(i64, i32), ObservedAsyncWarpState> = BTreeMap::new();
        let mut observation = AsyncInstrumentationOrderDiagnostics::default();
        for event in events {
            observation.selected_events = checked_order_observation_add(
                "async instrumentation selected events",
                observation.selected_events,
                1,
            )?;
            let warp_key = (event.launch_index, event.warp_slot);
            match event.kind {
                OrderedAsyncEventKind::Issue { effective } => {
                    observation.issue_events = checked_order_observation_add(
                        "async instrumentation issue events",
                        observation.issue_events,
                        1,
                    )?;
                    if effective {
                        let committed_total = states.values().try_fold(0_u64, |total, state| {
                            checked_order_observation_add(
                                "async instrumentation committed population",
                                total,
                                state.committed_groups,
                            )
                        })?;
                        let own = states
                            .get(&warp_key)
                            .map_or(0, |state| state.committed_groups);
                        let other = committed_total.checked_sub(own).unwrap_or_else(|| {
                            unreachable!("own groups are included in committed total")
                        });
                        if other != 0 {
                            observation.issue_events_with_other_warp_committed_groups =
                                checked_order_observation_add(
                                    "async instrumentation issues with sibling groups",
                                    observation.issue_events_with_other_warp_committed_groups,
                                    1,
                                )?;
                        }
                        observation.other_warp_committed_groups_sum =
                            checked_order_observation_add(
                                "async instrumentation sibling committed groups",
                                observation.other_warp_committed_groups_sum,
                                other,
                            )?;
                        observation.maximum_other_warp_committed_groups =
                            observation.maximum_other_warp_committed_groups.max(other);
                        let state = states.entry(warp_key).or_default();
                        state.open_effective_copies = checked_order_observation_add(
                            "async instrumentation open copies",
                            state.open_effective_copies,
                            1,
                        )?;
                    }
                }
                OrderedAsyncEventKind::Commit => {
                    observation.commit_events = checked_order_observation_add(
                        "async instrumentation commit events",
                        observation.commit_events,
                        1,
                    )?;
                    let state = states.entry(warp_key).or_default();
                    if state.open_effective_copies == 0 {
                        observation.protocol_contradictions = checked_order_observation_add(
                            "async instrumentation protocol contradictions",
                            observation.protocol_contradictions,
                            1,
                        )?;
                    } else {
                        state.committed_groups = checked_order_observation_add(
                            "async instrumentation committed groups",
                            state.committed_groups,
                            1,
                        )?;
                        state.open_effective_copies = 0;
                    }
                }
                OrderedAsyncEventKind::Wait { max_pending_groups } => {
                    observation.wait_events = checked_order_observation_add(
                        "async instrumentation wait events",
                        observation.wait_events,
                        1,
                    )?;
                    let state = states.entry(warp_key).or_default();
                    if state.open_effective_copies != 0 {
                        observation.protocol_contradictions = checked_order_observation_add(
                            "async instrumentation protocol contradictions",
                            observation.protocol_contradictions,
                            1,
                        )?;
                        state.open_effective_copies = 0;
                    }
                    state.committed_groups =
                        state.committed_groups.min(u64::from(max_pending_groups));
                }
            }
        }
        for state in states.values() {
            observation.protocol_contradictions = checked_order_observation_add(
                "async instrumentation protocol contradictions",
                observation.protocol_contradictions,
                u64::from(state.open_effective_copies != 0)
                    + u64::from(state.committed_groups != 0),
            )?;
        }
        observations.insert(sm_id, observation);
    }
    Ok(observations)
}

/// Card-derived spatial-placement parameters. Only the architecture/SKU-local SM
/// range crosses into replay; no label or target value is a simulator input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PlacementConfig {
    arch: ArchId,
    sku: Sku,
    sm_count: u32,
    max_shared_bytes_per_sm: u64,
}

impl PlacementConfig {
    fn from_arch_card(card: &ArchCard, expected_arch: ArchId) -> Result<Self, ReplayError> {
        let card_error = |source| ReplayError::ArchitectureCardPlacement { source };
        let arch = card.arch_id().map_err(card_error)?;
        require_replay_arch(expected_arch, arch, "architecture card")?;
        let sku = card.sku_id().map_err(card_error)?;
        let sm_count = card.sm_count().map_err(card_error)?;
        let max_shared_bytes_per_sm = card
            .require_l1_geometry()
            .map_err(|source| ReplayError::ArchitectureCardSharedMemoryCapacity { arch, source })?
            .max_shared_bytes;
        if max_shared_bytes_per_sm == 0 {
            return Err(ReplayError::SharedMemoryCapacityInvalid {
                arch,
                bytes: max_shared_bytes_per_sm,
            });
        }
        Ok(Self {
            arch,
            sku,
            sm_count,
            max_shared_bytes_per_sm,
        })
    }

    fn shared_memory_residency(
        self,
        kernel: &pb::Kernel,
    ) -> Result<SharedMemoryResidency, ReplayError> {
        let bytes_per_cta = u64::try_from(kernel.size_shared_memory).map_err(|_| {
            ReplayError::SharedMemoryDemandInvalid {
                kernel_id: kernel.id,
                bytes: kernel.size_shared_memory,
            }
        })?;
        if bytes_per_cta > self.max_shared_bytes_per_sm {
            return Err(ReplayError::CtaSharedMemoryCapacityExceeded {
                kernel_id: kernel.id,
                arch: self.arch,
                bytes_per_cta,
                max_bytes_per_sm: self.max_shared_bytes_per_sm,
            });
        }
        Ok(SharedMemoryResidency::new(
            self.max_shared_bytes_per_sm,
            bytes_per_cta,
        ))
    }

    fn require_traced_sm(self, path: &Path, sm_id: i32) -> Result<(), ReplayError> {
        if sm_id < 0 || u32::try_from(sm_id).unwrap_or(u32::MAX) >= self.sm_count {
            return Err(ReplayError::SmIdOutOfRange {
                path: path.to_path_buf(),
                sm_id,
                arch: self.arch,
                sku: self.sku,
                sm_count: self.sm_count,
            });
        }
        Ok(())
    }

    #[cfg(feature = "campaign")]
    fn require_declared_sm(self, launch_index: u64, sm_id: u64) -> Result<(), ReplayError> {
        if sm_id >= u64::from(self.sm_count) {
            return Err(ReplayError::SamplingSmIdOutOfRange {
                launch_index,
                sm_id,
                arch: self.arch,
                sku: self.sku,
                sm_count: self.sm_count,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
const fn sm86_test_placement() -> PlacementConfig {
    PlacementConfig {
        arch: ArchId::Sm86,
        sku: Sku::Rtx3090Ti,
        sm_count: 84,
        max_shared_bytes_per_sm: 102_400,
    }
}

#[cfg(test)]
fn sm86_test_card() -> ArchCard {
    ArchCard::parse(
        Path::new("arch_profiles/cards/sm_86.json"),
        include_bytes!("../../../arch_profiles/cards/sm_86.json"),
    )
    .expect("committed sm_86 architecture card parses")
}

#[cfg(feature = "campaign")]
fn validate_declared_sampling_sm_range(
    declared: impl IntoIterator<Item = (u64, u64)>,
    placement: PlacementConfig,
) -> Result<(), ReplayError> {
    for (launch_index, sm_id) in declared {
        placement.require_declared_sm(launch_index, sm_id)?;
    }
    Ok(())
}

#[cfg(feature = "campaign")]
fn validate_sampling_placement(
    sidecar: &TraceSamplingSidecar,
    traced_sm_by_launch_index: &BTreeMap<i64, i32>,
    placement: PlacementConfig,
) -> Result<(), ReplayError> {
    validate_declared_sampling_sm_range(sidecar.declared_block_placements(), placement)?;
    for (&launch_index, &traced_sm_id) in traced_sm_by_launch_index {
        let declared_sm_id = sidecar
            .declared_sm_for_launch_index(launch_index)
            .ok_or(ReplayError::SamplingPlacementMissing { launch_index })?;
        if u64::try_from(traced_sm_id).ok() != Some(declared_sm_id) {
            return Err(ReplayError::SamplingPlacementMismatch {
                launch_index,
                traced_sm_id,
                declared_sm_id,
            });
        }
    }
    Ok(())
}

fn normalized_cluster_dim(
    kernel: &sim_trace::pb::Kernel,
    arch: ArchId,
) -> Result<(u32, u32, u32), ReplayError> {
    let Some(cluster) = kernel.cluster_dim.as_ref() else {
        return Ok((1, 1, 1));
    };
    let raw = (cluster.x, cluster.y, cluster.z);
    if raw == (0, 0, 0) {
        return Ok((1, 1, 1));
    }
    if raw.0 <= 0 || raw.1 <= 0 || raw.2 <= 0 {
        return Err(ReplayError::ClusterDimInvalid { cluster: raw });
    }
    let cluster = (raw.0 as u32, raw.1 as u32, raw.2 as u32);
    if cluster != (1, 1, 1) {
        return Err(ReplayError::ClusterGeometryUnsupported { arch, cluster });
    }
    Ok(cluster)
}

fn checked_grid_total(grid: (u32, u32, u32)) -> Result<u64, ReplayError> {
    u64::from(grid.0)
        .checked_mul(u64::from(grid.1))
        .and_then(|value| value.checked_mul(u64::from(grid.2)))
        .ok_or(ReplayError::GridSizeOverflow { grid })
}

fn checked_block_warps(block: (u32, u32, u32)) -> Result<u64, ReplayError> {
    const TRACE_WARP_WIDTH: u64 = 32;
    let threads = u64::from(block.0)
        .checked_mul(u64::from(block.1))
        .and_then(|value| value.checked_mul(u64::from(block.2)))
        .ok_or(ReplayError::BlockSizeOverflow { block })?;
    Ok(threads.div_ceil(TRACE_WARP_WIDTH))
}

fn checked_record_count(
    accumulated: u64,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<u64, ReplayError> {
    accumulated
        .checked_add(1)
        .ok_or_else(|| ReplayError::RecordCountOverflow {
            path: path.to_path_buf(),
            warp_id,
            record_ix,
            accumulated,
        })
}

fn checked_collection_len(collection: &'static str, len: usize) -> Result<u64, ReplayError> {
    u64::try_from(len).map_err(|_| ReplayError::CollectionLengthOutOfRange { collection, len })
}

/// Decode one launch into per-SM block replays on the traced placement.
/// `decode_sectors` selects the L1-mode classification (lane decode +
/// unique-sector expansion) over the flat-mode class check.
fn extract_per_sm_blocks(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedule: &FunctionSchedule,
    placement: PlacementConfig,
    decode_sectors: bool,
) -> Result<ExtractedKernel, ReplayError> {
    let expected_fuid = schedule.function_unique_id;
    let shared_memory_residency = placement.shared_memory_residency(loc.kernel())?;
    let block = loc
        .kernel()
        .block_dim
        .as_ref()
        .filter(|dim| dim.x > 0 && dim.y > 0 && dim.z > 0)
        .map(|dim| (dim.x as u32, dim.y as u32, dim.z as u32));
    let expected_block_warps = block.map(checked_block_warps).transpose()?;
    #[cfg(feature = "campaign")]
    let validate_warp_roster = expected_block_warps.is_some()
        && td
            .sampling_sidecar()
            .is_none_or(|sidecar| Some(sidecar.original_block()) == block);
    #[cfg(not(feature = "campaign"))]
    let validate_warp_roster = expected_block_warps.is_some();
    let max_resident_warps = schedule
        .arch
        .issue_hypotheses()
        .require("physical warp-slot capacity")
        .map_err(|source| ReplayError::WarpCapacityUnavailable {
            arch: schedule.arch,
            source,
        })?
        .max_resident_warps_per_sm;
    let cluster = normalized_cluster_dim(loc.kernel(), schedule.arch)?;
    // sm_id → blocks (launch index kept for in-SM ordering).
    let mut per_sm_blocks: BTreeMap<i32, Vec<BlockReplay>> = BTreeMap::new();
    let mut seen_indices: BTreeSet<i64> = BTreeSet::new();
    let mut traced_sm_by_launch_index: BTreeMap<i64, i32> = BTreeMap::new();
    let mut records_seen: u64 = 0;
    let mut instruction_sectors = BTreeSet::new();
    let launch_resources = (loc.kernel().number_of_registers > 0
        || loc.kernel().size_shared_memory > 0
        || loc.kernel().shared_memory_static_bytes.is_some()
        || loc.kernel().shared_memory_dynamic_bytes.is_some()
        || loc.kernel().shared_memory_carveout_preference_pct.is_some()
        || loc.kernel().driver_occupancy_limit_blocks.is_some())
    .then(|| crate::metrics::LaunchResources {
        block_threads: loc
            .kernel()
            .block_dim
            .as_ref()
            .map(|dim| {
                dim.x
                    .max(0)
                    .saturating_mul(dim.y.max(0))
                    .saturating_mul(dim.z.max(0)) as u32
            })
            .unwrap_or(0),
        grid_blocks: loc
            .kernel()
            .grid_dim
            .as_ref()
            .map(|dim| {
                u64::from(dim.x.max(0) as u32)
                    .saturating_mul(u64::from(dim.y.max(0) as u32))
                    .saturating_mul(u64::from(dim.z.max(0) as u32))
            })
            .unwrap_or(0),
        // TraceDir validated nonnegative counts and the split/total sum.
        registers_per_thread: loc.kernel().number_of_registers as u32,
        shared_memory_bytes_per_block: loc.kernel().size_shared_memory as u32,
        static_shared_memory_bytes: loc.kernel().shared_memory_static_bytes,
        dynamic_shared_memory_bytes: loc.kernel().shared_memory_dynamic_bytes,
        shared_memory_carveout_preference_pct: loc.kernel().shared_memory_carveout_preference_pct,
        driver_occupancy_limit_blocks: loc.kernel().driver_occupancy_limit_blocks,
    });
    let mut trace_metrics = TraceMetrics {
        launch_resources,
        sectors_decoded: decode_sectors,
        ..TraceMetrics::default()
    };

    for path in td.threadblock_paths(loc)? {
        let tb = td.validated_threadblock(&path)?;
        let launch_index = tb.launch_index();
        if !seen_indices.insert(launch_index) {
            return Err(ReplayError::DuplicateLaunchIndex {
                index: launch_index,
            });
        }

        let admitted_block_warps = expected_block_warps.filter(|_| validate_warp_roster);
        if let Some(expected_block_warps) = admitted_block_warps {
            let actual_block_warps =
                checked_collection_len("threadblock warp roster", tb.warps().len())?;
            if actual_block_warps != expected_block_warps {
                return Err(ReplayError::WarpRosterCardinalityMismatch {
                    path,
                    expected: expected_block_warps,
                    actual: actual_block_warps,
                });
            }
        }
        let mut block_sm: Option<i32> = None;
        let mut seen_warp_slots = BTreeSet::new();
        let mut warps = Vec::with_capacity(tb.warps().len());
        for warp in tb.warps() {
            let warp_id = warp.map_key();
            if admitted_block_warps.is_some_and(|expected_block_warps| {
                warp_id < 0 || u64::try_from(warp_id).unwrap_or(u64::MAX) >= expected_block_warps
            }) {
                return Err(ReplayError::WarpIdOutOfRange {
                    path,
                    warp_id,
                    expected_warps: admitted_block_warps.expect("checked as present"),
                });
            }
            if warp.warp_id() != warp_id {
                return Err(ReplayError::WarpIdentityMismatch {
                    path,
                    map_key: warp_id,
                    warp_id: warp.warp_id(),
                });
            }
            let mut warp_slot: Option<i32> = None;
            let mut insts = Vec::with_capacity(warp.instructions().len());
            let mut sectors: Vec<i64> = Vec::new();
            let mut async_copy_cohorts = AsyncCopyCohortReplay::default();
            let mut branch_successors = crate::metrics::BranchSuccessors::default();
            let mut previous_control_flow: Option<(i32, u32, u32)> = None;
            for (record_ix, inst) in warp.instructions().enumerate() {
                let record_warp_id = inst.warpid_tb();
                if record_warp_id != warp_id {
                    return Err(ReplayError::InstructionWarpIdMismatch {
                        path,
                        warp_id,
                        record_ix,
                        record_warp_id,
                    });
                }
                records_seen = checked_record_count(records_seen, &path, warp_id, record_ix)?;
                let preflight = preflight_compact_memory(
                    inst.memory(),
                    warp,
                    schedule,
                    inst.pc(),
                    &path,
                    warp_id,
                    record_ix,
                )?;
                if inst.function_unique_id() != expected_fuid {
                    return Err(ReplayError::MultiFunctionUnsupported {
                        path,
                        warp_id,
                        record_ix,
                        expected: expected_fuid,
                        found: inst.function_unique_id(),
                    });
                }
                instruction_sectors.insert((inst.function_unique_id(), inst.pc() / 32));
                if let Some(sm) = inst.sm_id() {
                    match block_sm {
                        None => block_sm = Some(sm),
                        Some(first) if first != sm => {
                            return Err(ReplayError::SmIdInconsistent {
                                path,
                                first,
                                second: sm,
                            });
                        }
                        Some(_) => {}
                    }
                }
                if let Some(slot) = inst.warpid_sm() {
                    match warp_slot {
                        None => warp_slot = Some(slot),
                        Some(first) if first != slot => {
                            return Err(ReplayError::WarpSlotInconsistent {
                                path,
                                warp_id,
                                first,
                                second: slot,
                            });
                        }
                        Some(_) => {}
                    }
                }
                let kind = if decode_sectors {
                    classify_compact_with_l1(
                        preflight,
                        warp,
                        schedule,
                        schedule.arch,
                        &mut sectors,
                        &mut async_copy_cohorts,
                        &path,
                        warp_id,
                        record_ix,
                    )?
                } else {
                    classify_compact_flat(
                        preflight,
                        warp,
                        schedule,
                        schedule.arch,
                        &path,
                        warp_id,
                        record_ix,
                    )?
                };
                let active_mask_present = inst.active_mask_present();
                let predicate_mask_present = inst.predicate_mask_present();
                let (opcode, pipeline) = schedule
                    .issue_slot(u64::from(inst.pc()))
                    .map_or(("UNKNOWN", PipelineClass::DispatchOnly), |slot| {
                        (slot.opcode_text(), slot.pipeline_class())
                    });
                trace_metrics.observe_with_mask_presence(
                    opcode,
                    inst.active_lanes(),
                    inst.predicate_on_lanes(),
                    active_mask_present,
                    predicate_mask_present,
                    kind,
                    pipeline,
                    &sectors,
                )?;
                if active_mask_present && predicate_mask_present {
                    trace_metrics.observe_coherence(
                        inst.sm_id(),
                        inst.predicate_on_lanes(),
                        kind,
                        &sectors,
                    )?;
                }
                if active_mask_present {
                    if let Some((from_function, from_pc, from_active)) = previous_control_flow {
                        trace_metrics.observe_control_flow(
                            from_function,
                            from_pc,
                            from_active,
                            inst.function_unique_id(),
                            inst.pc(),
                            inst.active_mask(),
                        )?;
                    }
                    previous_control_flow =
                        Some((inst.function_unique_id(), inst.pc(), inst.active_mask()));
                    branch_successors.observe(
                        &mut trace_metrics.branch_targets,
                        inst.function_unique_id(),
                        inst.pc(),
                        inst.active_mask(),
                        opcode,
                    )?;
                } else {
                    // Do not connect CFG edges or resolve branch successors
                    // through a record whose lane population is absent.
                    previous_control_flow = None;
                }
                if let Some(order) = inst.instrumentation_order() {
                    async_copy_cohorts
                        .instrumentation_order
                        .push(InstrumentationOrderAnchor {
                            instruction: record_ix,
                            order,
                        });
                }
                insts.push(ReplayInst {
                    pc: inst.pc(),
                    kind,
                });
            }
            branch_successors.finish(&mut trace_metrics.branch_targets)?;
            // Fail-closed traced placement: every roster warp must expose its
            // physical slot. An empty warp has no placement witness and cannot be
            // used to prove either the architecture slot bound or modeled CTA
            // co-residency.
            let Some(warp_slot) = warp_slot else {
                return Err(ReplayError::WarpSlotMissing { path, warp_id });
            };
            if warp_slot < 0 || u32::try_from(warp_slot).unwrap_or(u32::MAX) >= max_resident_warps {
                return Err(ReplayError::WarpSlotOutOfRange {
                    path,
                    warp_id,
                    warp_slot,
                    arch: schedule.arch,
                    max_resident_warps,
                });
            }
            if !seen_warp_slots.insert(warp_slot) {
                return Err(ReplayError::DuplicateWarpSlot { path, warp_slot });
            }
            warps.push(WarpReplay {
                warp_slot,
                insts,
                sectors,
                async_copy_cohorts: (!async_copy_cohorts.spans.is_empty()
                    || !async_copy_cohorts.instrumentation_order.is_empty())
                .then(|| Box::new(async_copy_cohorts)),
            });
        }
        let Some(sm) = block_sm else {
            return Err(ReplayError::SmIdMissing { path });
        };
        placement.require_traced_sm(&path, sm)?;
        traced_sm_by_launch_index.insert(launch_index, sm);
        per_sm_blocks.entry(sm).or_default().push(BlockReplay {
            launch_index,
            warps,
        });
    }
    // Completeness gate (fail-closed) on the path replay ACTUALLY consumes: a
    // subset trace (coverage < grid) is admissible ONLY with a valid
    // trace_sampling.json sidecar whose original_grid matches the launch record
    // (gridDim is never rewritten) and whose captured record count matches
    // records_seen. Absent/non-sampling sidecars never waive full coverage.
    let (gx, gy, gz) = kernel_grid(loc.kernel())?;
    let grid = (gx, gy, gz);
    let grid_total = checked_grid_total(grid)?;
    let seen = checked_collection_len("replayed block membership", seen_indices.len())?;
    let launch = ReplayLaunchCoverageBinding {
        device_id: loc.device_id(),
        stream_id: loc.stream_id(),
        kernel_id: loc.kernel().id,
        function_unique_id: loc.kernel().function_unique_id,
        kernel_mangled_name: loc.kernel().name.clone(),
        grid,
        block,
        cluster,
    };
    #[cfg(not(feature = "campaign"))]
    let coverage = {
        if seen != grid_total {
            return Err(ReplayError::IncompleteTrace { seen, grid_total });
        }
        ExtractedCoverage::Full(FullLaunchCoverage {
            launch,
            blocks_observed: seen,
            source: FullCoverageSource::UnsampledComplete,
        })
    };
    #[cfg(feature = "campaign")]
    let coverage = match td.sampling_sidecar() {
        None => {
            if seen != grid_total {
                return Err(ReplayError::IncompleteTrace { seen, grid_total });
            }
            ExtractedCoverage::Full(FullLaunchCoverage {
                launch,
                blocks_observed: seen,
                source: FullCoverageSource::UnsampledComplete,
            })
        }
        Some(sc) => {
            let launch_block = launch.block.ok_or(ReplayError::BlockDimMissing)?;
            if sc.kernel_mangled_name() != loc.kernel().name.as_str() {
                return Err(ReplayError::SamplingKernelMismatch {
                    launch: loc.kernel().name.clone(),
                    sidecar: sc.kernel_mangled_name().to_owned(),
                });
            }
            if sc.original_grid() != grid {
                return Err(ReplayError::GridDimRewritten {
                    launch: grid,
                    sidecar: sc.original_grid(),
                });
            }
            if sc.original_block() != launch_block {
                return Err(ReplayError::BlockDimRewritten {
                    launch: launch_block,
                    sidecar: sc.original_block(),
                });
            }
            let coverage = match sc.method() {
                TraceSamplingMethod::WaveSegmentStratifiedSubset(provenance) => {
                    if provenance.selected_block_ids() != &seen_indices {
                        let declared = checked_collection_len(
                            "trace_sampling selected block membership",
                            provenance.selected_block_ids().len(),
                        )?;
                        return Err(ReplayError::SelectedBlockMembershipMismatch {
                            seen,
                            declared,
                        });
                    }
                    ExtractedCoverage::Sampled(SampledLaunchCoverage {
                        launch,
                        population_blocks: grid_total,
                        blocks_observed: seen,
                        provenance: provenance.clone(),
                    })
                }
                TraceSamplingMethod::LosslessCompressedFull(provenance) => {
                    if seen != grid_total {
                        return Err(ReplayError::IncompleteTrace { seen, grid_total });
                    }
                    ExtractedCoverage::Full(FullLaunchCoverage {
                        launch,
                        blocks_observed: seen,
                        source: FullCoverageSource::LosslessCompressedFull(provenance.clone()),
                    })
                }
            };
            validate_sampling_placement(sc, &traced_sm_by_launch_index, placement)?;
            if records_seen != sc.captured_records() {
                return Err(ReplayError::RecordsCapturedMismatch {
                    records_seen,
                    captured: sc.captured_records(),
                });
            }
            coverage
        }
    };
    for blocks in per_sm_blocks.values_mut() {
        blocks.sort_by_key(|b| b.launch_index);
    }
    trace_metrics.finish_coherence()?;
    Ok(ExtractedKernel {
        per_sm_blocks,
        records_seen,
        coverage,
        shared_memory_residency,
        front_end: FrontEndCounters {
            compulsory_instruction_read_sectors: u64::try_from(instruction_sectors.len()).map_err(
                |_| IssueError::CounterOverflow {
                    counter: "compulsory instruction-read sector cardinality",
                },
            )?,
        },
        trace_metrics,
    })
}

/// Validate every consumed threadblock and classify launch coverage without
/// running the timing model. The campaign feature uses this for its manifest
/// compatibility path.
#[cfg(feature = "campaign")]
pub(crate) fn admit_launch_coverage(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedule: &FunctionSchedule,
    arch_card: &ArchCard,
) -> Result<ExtractedCoverage, ReplayError> {
    let placement = PlacementConfig::from_arch_card(arch_card, schedule.arch)?;
    require_trace_architecture(td, schedule.arch)?;
    Ok(extract_per_sm_blocks(td, loc, schedule, placement, false)?.coverage)
}

/// Fold per-SM results into the kernel-level view, enforcing the
/// one-issue-per-record invariant.
fn fold_kernel_result(
    arch: ArchId,
    per_sm: BTreeMap<i32, SmSimResult>,
    records_seen: u64,
    expect_l1: bool,
    last_processed_system_event_time: Option<u64>,
    l2: Option<(L2Counters, L2SourceSplit, DramStats)>,
) -> Result<FoldedKernelReplay, ReplayError> {
    fold_kernel_result_with_sm_active_time(
        arch,
        per_sm,
        records_seen,
        expect_l1,
        last_processed_system_event_time,
        last_processed_system_event_time,
        l2,
    )
}

fn fold_kernel_result_with_sm_active_time(
    arch: ArchId,
    per_sm: BTreeMap<i32, SmSimResult>,
    records_seen: u64,
    expect_l1: bool,
    last_processed_system_event_time: Option<u64>,
    sm_active_terminal_event_time: Option<u64>,
    l2: Option<(L2Counters, L2SourceSplit, DramStats)>,
) -> Result<FoldedKernelReplay, ReplayError> {
    let mut instructions_issued: u64 = 0;
    let mut diagnostic_last_issue_cycles: u64 = 0;
    let mut latest_local_completion_cycles: Option<u64> = None;
    let mut unmodeled_tails = TailFlags::default();
    let mut l1_totals = expect_l1.then(|| L1Counters::for_arch(arch));
    for (&sm_id, result) in &per_sm {
        instructions_issued = instructions_issued
            .checked_add(result.instructions_issued)
            .ok_or(ReplayError::InstructionCountOverflow {
                sm_id,
                accumulated: instructions_issued,
                addend: result.instructions_issued,
            })?;
        diagnostic_last_issue_cycles = diagnostic_last_issue_cycles.max(result.cycles);
        if let Some(completion) = result.modeled_local_completion_cycles {
            latest_local_completion_cycles = Some(
                latest_local_completion_cycles
                    .map_or(completion, |current| current.max(completion)),
            );
        }
        unmodeled_tails = unmodeled_tails.union(result.unmodeled_tails);
        let found_l1 = result.l1.is_some();
        if found_l1 != expect_l1 {
            return Err(ReplayError::L1PresenceMismatch {
                sm_id,
                expected: expect_l1,
                found: found_l1,
            });
        }
        if let (Some(c), Some(total)) = (&result.l1, &mut l1_totals) {
            total
                .add_checked(c)
                .map_err(|source| ReplayError::L1Aggregation { sm_id, source })?;
        }
    }
    if instructions_issued != records_seen {
        return Err(ReplayError::CountMismatch {
            issued: instructions_issued,
            records: records_seen,
        });
    }
    let issue_frontier = (instructions_issued != 0).then_some(diagnostic_last_issue_cycles);
    let terminal_modeled_event_time = [
        issue_frontier,
        last_processed_system_event_time,
        latest_local_completion_cycles,
    ]
    .into_iter()
    .flatten()
    .max();
    let (l2, l2_by_source, dram) = match l2 {
        Some((counters, split, dram)) => (Some(counters), Some(split), Some(dram)),
        None => (None, None, None),
    };
    Ok(FoldedKernelReplay {
        arch,
        instructions_issued,
        records_seen,
        per_sm,
        diagnostic_last_issue_cycles,
        last_processed_system_event_time,
        terminal_modeled_event_time,
        sm_active_terminal_event_time: [
            issue_frontier,
            sm_active_terminal_event_time,
            latest_local_completion_cycles,
        ]
        .into_iter()
        .flatten()
        .max(),
        l1: l1_totals,
        l2,
        l2_by_source,
        dram,
        front_end: FrontEndCounters::default(),
        trace_metrics: TraceMetrics::default(),
        unmodeled_tails,
    })
}

fn require_system_tail_consistency(
    system: TailFlags,
    per_sm: TailFlags,
) -> Result<(), ReplayError> {
    if system != per_sm {
        return Err(ReplayError::SystemTailMismatch { system, per_sm });
    }
    Ok(())
}

fn classify_kernel_result(
    coverage: ExtractedCoverage,
    configuration: ReplayConfiguration,
    replay: FoldedKernelReplay,
) -> KernelReplayOutcome {
    match coverage {
        ExtractedCoverage::Full(coverage) => KernelReplayOutcome::Full(FullKernelReplayResult {
            coverage,
            configuration,
            replay,
        }),
        #[cfg(feature = "campaign")]
        ExtractedCoverage::Sampled(coverage) => {
            let FoldedKernelReplay {
                arch,
                instructions_issued,
                records_seen,
                per_sm: _,
                diagnostic_last_issue_cycles: _,
                last_processed_system_event_time: _,
                terminal_modeled_event_time: _,
                sm_active_terminal_event_time: _,
                l1,
                l2,
                l2_by_source,
                dram,
                front_end,
                trace_metrics: _,
                unmodeled_tails,
            } = replay;
            KernelReplayOutcome::SampledObserved(SampledKernelReplayResult {
                coverage,
                configuration,
                counters: SampledObservedCounters {
                    arch,
                    instructions_issued,
                    records_seen,
                    l1,
                    l2,
                    l2_by_source,
                    dram,
                    front_end,
                    unmodeled_tails,
                },
            })
        }
    }
}

fn lookup_schedule<'a>(
    schedules: &'a BTreeMap<i32, FunctionSchedule>,
    loc: &KernelLoc<'_>,
) -> Result<&'a FunctionSchedule, ReplayError> {
    let expected_fuid = loc.kernel().function_unique_id;
    schedules
        .get(&expected_fuid)
        .ok_or(ReplayError::ScheduleMissing {
            function_unique_id: expected_fuid,
        })
}

/// Memory-hierarchy mode for one kernel replay — the ONE typed selector
/// every harness goes through (a pair of optional configs would let a
/// plumbing slip degrade a cell to the wrong mode silently).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayMemMode {
    /// No cache model (flat global latency).
    Flat,
    /// Per-SM L1s with constant fill latency below them (the pre-L2
    /// lockstep/ablation surface).
    L1Const(L1Config),
    /// Per-SM L1s over ONE shared L2 (the full hierarchy): L1 misses
    /// priced L2-hit vs DRAM by the shared state through the event
    /// system.
    L1SharedL2 {
        l1: L1Config,
        l2: L2Config,
        dram: DramBw,
    },
}

/// Temporal CTA co-residency provenance. The current trace wire carries spatial
/// `sm_id`/`warpid_sm` placement but no block start/end timestamps, so exact
/// hardware co-residency cannot be claimed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoResidency {
    /// Blocks are admitted in launch-index order under the modeled resident-warp
    /// ceiling and architecture-card shared-memory capacity; concurrent CTAs must
    /// also use disjoint traced physical warp slots. This is never a claim that the
    /// hardware used the same overlap.
    Modeled,
}

/// Exact structural identity of every issue and memory-hierarchy parameter used
/// for one replay. No digest or caller label substitutes for these values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayConfiguration {
    issue_params: IssueModelParams,
    memory_mode: ReplayMemMode,
    memory_admission_feedback: Option<MemoryAdmissionCandidate>,
    co_residency: CoResidency,
}

impl ReplayConfiguration {
    /// Build a declaration-side identity while enforcing architecture lockstep
    /// across all nested memory configurations.
    pub fn checked(
        issue_params: IssueModelParams,
        memory_mode: ReplayMemMode,
    ) -> Result<Self, ReplayError> {
        validate_replay_mode_architecture(issue_params.arch, memory_mode)?;
        issue_params.validate_resource_source()?;
        Ok(Self {
            issue_params,
            memory_mode,
            memory_admission_feedback: None,
            co_residency: CoResidency::Modeled,
        })
    }

    /// Build the structural identity for the timing-active schema-10 lane.
    /// The candidate's operand-visibility floor must equal the issue model's
    /// existing read-barrier latency: feedback replaces queue readiness only
    /// and must not silently introduce (or ignore) a second latency source.
    pub fn checked_with_memory_admission_feedback(
        issue_params: IssueModelParams,
        memory_mode: ReplayMemMode,
        candidate: MemoryAdmissionCandidate,
    ) -> Result<Self, ReplayError> {
        if !matches!(memory_mode, ReplayMemMode::L1SharedL2 { .. }) {
            return Err(ReplayError::MemoryAdmissionCandidatesRequireSharedL2);
        }
        candidate.validate_feedback(issue_params.read_barrier_latency)?;
        let mut configuration = Self::checked(issue_params, memory_mode)?;
        configuration.memory_admission_feedback = Some(candidate);
        Ok(configuration)
    }

    pub fn arch(&self) -> ArchId {
        self.issue_params.arch
    }

    pub fn issue_params(&self) -> IssueModelParams {
        self.issue_params
    }

    pub fn memory_mode(&self) -> ReplayMemMode {
        self.memory_mode
    }

    /// Timing-active finite admission candidate, when schema 10 is enabled.
    pub fn memory_admission_feedback(&self) -> Option<MemoryAdmissionCandidate> {
        self.memory_admission_feedback
    }

    pub fn co_residency(&self) -> CoResidency {
        self.co_residency
    }

    /// Return the residual architecture-scoped SM-active observation boundary
    /// used by the explicit frontier projection. Once the causal IMC path is
    /// active, the independent 964-cycle legacy residual is not additive: it
    /// represented the missing constant-bank work and would double-charge the
    /// same silicon interval. The current SM86 causal lane therefore exposes a
    /// zero residual while retaining the historical capability for old data.
    pub fn sm_active_activation_cycles(&self) -> u64 {
        if !matches!(self.memory_mode, ReplayMemMode::L1SharedL2 { .. })
            || !matches!(
                self.issue_params.resource_source(),
                IssueResourceSource::ArchitectureDefaults
            )
        {
            return 0;
        }
        if self.issue_params.immediate_constant_cache().is_some() {
            return 0;
        }
        self.issue_params
            .arch
            .sm_active_observation_envelope()
            .require("sm_active_observation_envelope")
            .map_or(0, |envelope| envelope.cycles())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OrderedSessionLaunch {
    kernel_id: i32,
    function_unique_id: i32,
}

/// Typed failures for an ordered same-stream replay session. Session errors
/// are separate from one-launch replay failures so callers can distinguish a
/// malformed launch from a launch-chain ownership/order mismatch.
#[derive(Debug, thiserror::Error)]
pub enum ReplaySessionError {
    #[error(transparent)]
    Replay(#[from] ReplayError),
    #[error("stream-L2 cache scope requires L1SharedL2 replay mode, got {mode:?}")]
    SharedL2ModeRequired { mode: ReplayMemMode },
    #[error("trace stream ({device_id}, {stream_id}) contains no kernel launches")]
    EmptyStream { device_id: i64, stream_id: i64 },
    #[error("replay session is bound to trace root {expected}, got {got}")]
    TraceRootMismatch { expected: PathBuf, got: PathBuf },
    #[error("replay session is bound to a different loaded trace lineage")]
    TraceLineageMismatch,
    #[error("replay session is bound to device {expected}, got {got}")]
    DeviceMismatch { expected: i64, got: i64 },
    #[error("replay session is bound to stream {expected}, got {got}")]
    StreamMismatch { expected: i64, got: i64 },
    #[error("replay session configuration mismatch: expected {expected:?}, got {got:?}")]
    ConfigurationMismatch {
        expected: ReplayConfiguration,
        got: ReplayConfiguration,
    },
    #[error(
        "replay session architecture-card placement mismatch: expected sku={expected_sku:?}, sm_count={expected_sm_count}, max_shared_bytes_per_sm={expected_max_shared_bytes_per_sm}; got sku={got_sku:?}, sm_count={got_sm_count}, max_shared_bytes_per_sm={got_max_shared_bytes_per_sm}"
    )]
    ArchitectureCardMismatch {
        expected_sku: Sku,
        expected_sm_count: u32,
        expected_max_shared_bytes_per_sm: u64,
        got_sku: Sku,
        got_sm_count: u32,
        got_max_shared_bytes_per_sm: u64,
    },
    #[error(
        "replay session launch position {position} expects kernel {expected_kernel_id}/function {expected_function_unique_id}, got kernel {got_kernel_id}/function {got_function_unique_id}"
    )]
    LaunchOrderMismatch {
        position: usize,
        expected_kernel_id: i32,
        expected_function_unique_id: i32,
        got_kernel_id: i32,
        got_function_unique_id: i32,
    },
    #[error("replay session already consumed all {completed_launches} launches")]
    LaunchSequenceComplete { completed_launches: usize },
    #[error("stream-L2 replay outcome omitted its per-launch L2 counter delta")]
    LaunchL2CountersMissing,
    #[error(
        "stream-L2 cumulative counters do not reconcile: reported={reported:?}, model={model:?}"
    )]
    CumulativeL2CounterMismatch {
        reported: L2Counters,
        model: L2Counters,
    },
}

/// Ordered same-device/same-stream replay state. Only the shared L2 model
/// survives a successful launch boundary. Per-SM L1, MSHR, scheduler,
/// dependence, async-copy, pipeline, CTA, event-queue, and DRAM-server state is
/// rebuilt by the ordinary one-launch system path on every call.
pub struct StreamL2ReplaySession {
    trace_root: PathBuf,
    trace_lineage: TraceLineage,
    configuration: ReplayConfiguration,
    placement: PlacementConfig,
    device_id: i64,
    stream_id: i64,
    expected_launches: Vec<OrderedSessionLaunch>,
    completed_launches: usize,
    l2: L2Model,
    cumulative_reported_l2: L2Counters,
}

impl StreamL2ReplaySession {
    /// Bind a session to one trace root and one exact device/stream launch
    /// sequence. Construction is accepted only for the shared-L2 replay mode.
    pub fn new(
        td: &TraceDir,
        arch_card: &ArchCard,
        params: &IssueModelParams,
        mode: ReplayMemMode,
        device_id: i64,
        stream_id: i64,
    ) -> Result<Self, ReplaySessionError> {
        let configuration = ReplayConfiguration::checked(*params, mode)?;
        let l2_config = match mode {
            ReplayMemMode::L1SharedL2 { l2, .. } => l2,
            _ => return Err(ReplaySessionError::SharedL2ModeRequired { mode }),
        };
        require_trace_architecture(td, configuration.arch())?;
        let placement = PlacementConfig::from_arch_card(arch_card, configuration.arch())?;
        let expected_launches: Vec<_> = td
            .kernels()
            .filter(|loc| loc.device_id() == device_id && loc.stream_id() == stream_id)
            .map(|loc| OrderedSessionLaunch {
                kernel_id: loc.kernel().id,
                function_unique_id: loc.kernel().function_unique_id,
            })
            .collect();
        if expected_launches.is_empty() {
            return Err(ReplaySessionError::EmptyStream {
                device_id,
                stream_id,
            });
        }
        let l2 = L2Model::new(l2_config).map_err(ReplayError::from)?;
        Ok(Self {
            trace_root: td.root().to_path_buf(),
            trace_lineage: td.lineage(),
            configuration,
            placement,
            device_id,
            stream_id,
            expected_launches,
            completed_launches: 0,
            l2,
            cumulative_reported_l2: L2Counters::for_arch(configuration.arch()),
        })
    }

    pub fn configuration(&self) -> ReplayConfiguration {
        self.configuration
    }

    pub fn device_id(&self) -> i64 {
        self.device_id
    }

    pub fn stream_id(&self) -> i64 {
        self.stream_id
    }

    pub fn completed_launches(&self) -> usize {
        self.completed_launches
    }

    pub fn remaining_launches(&self) -> usize {
        self.expected_launches.len() - self.completed_launches
    }

    pub fn is_complete(&self) -> bool {
        self.completed_launches == self.expected_launches.len()
    }

    pub fn cumulative_l2_counters(&self) -> L2Counters {
        self.cumulative_reported_l2
    }

    pub fn pending_l2_sectors(&self) -> usize {
        self.l2.pending_sector_count()
    }

    /// Replay the next launch transactionally. The session cache and cursor are
    /// committed together only after the launch result, drained-state check,
    /// per-launch delta, and cumulative-counter reconciliation all succeed.
    pub fn replay_next(
        &mut self,
        td: &TraceDir,
        loc: &KernelLoc<'_>,
        schedules: &BTreeMap<i32, FunctionSchedule>,
        arch_card: &ArchCard,
        params: &IssueModelParams,
        mode: ReplayMemMode,
    ) -> Result<KernelReplayOutcome, ReplaySessionError> {
        if td.root() != self.trace_root {
            return Err(ReplaySessionError::TraceRootMismatch {
                expected: self.trace_root.clone(),
                got: td.root().to_path_buf(),
            });
        }
        if td.lineage() != self.trace_lineage {
            return Err(ReplaySessionError::TraceLineageMismatch);
        }
        if loc.device_id() != self.device_id {
            return Err(ReplaySessionError::DeviceMismatch {
                expected: self.device_id,
                got: loc.device_id(),
            });
        }
        if loc.stream_id() != self.stream_id {
            return Err(ReplaySessionError::StreamMismatch {
                expected: self.stream_id,
                got: loc.stream_id(),
            });
        }
        let Some(expected_launch) = self.expected_launches.get(self.completed_launches) else {
            return Err(ReplaySessionError::LaunchSequenceComplete {
                completed_launches: self.completed_launches,
            });
        };
        if loc.kernel().id != expected_launch.kernel_id
            || loc.kernel().function_unique_id != expected_launch.function_unique_id
        {
            return Err(ReplaySessionError::LaunchOrderMismatch {
                position: self.completed_launches,
                expected_kernel_id: expected_launch.kernel_id,
                expected_function_unique_id: expected_launch.function_unique_id,
                got_kernel_id: loc.kernel().id,
                got_function_unique_id: loc.kernel().function_unique_id,
            });
        }

        let got_configuration = ReplayConfiguration::checked(*params, mode)?;
        if got_configuration != self.configuration {
            return Err(ReplaySessionError::ConfigurationMismatch {
                expected: self.configuration,
                got: got_configuration,
            });
        }
        let got_placement = PlacementConfig::from_arch_card(arch_card, self.configuration.arch())?;
        if got_placement != self.placement {
            return Err(ReplaySessionError::ArchitectureCardMismatch {
                expected_sku: self.placement.sku,
                expected_sm_count: self.placement.sm_count,
                expected_max_shared_bytes_per_sm: self.placement.max_shared_bytes_per_sm,
                got_sku: got_placement.sku,
                got_sm_count: got_placement.sm_count,
                got_max_shared_bytes_per_sm: got_placement.max_shared_bytes_per_sm,
            });
        }
        if self.l2.pending_sector_count() != 0 {
            return Err(ReplayError::RetainedL2NotDrained {
                pending_sectors: self.l2.pending_sector_count(),
            }
            .into());
        }

        let mut staged_l2 = self.l2.try_snapshot().map_err(ReplayError::from)?;
        let schedule = lookup_schedule(schedules, loc)?;
        let outcome = replay_kernel_with_schedule_and_l2(
            td,
            loc,
            schedule,
            arch_card,
            params,
            mode,
            Some(&mut staged_l2),
        )?;
        if staged_l2.pending_sector_count() != 0 {
            return Err(ReplayError::RetainedL2NotDrained {
                pending_sectors: staged_l2.pending_sector_count(),
            }
            .into());
        }
        let launch_delta = outcome
            .l2_counters()
            .copied()
            .ok_or(ReplaySessionError::LaunchL2CountersMissing)?;
        let mut cumulative_reported_l2 = self.cumulative_reported_l2;
        cumulative_reported_l2
            .add_checked(&launch_delta)
            .map_err(ReplayError::from)?;
        let model_counters = *staged_l2.counters().map_err(ReplayError::from)?;
        if cumulative_reported_l2 != model_counters {
            return Err(ReplaySessionError::CumulativeL2CounterMismatch {
                reported: cumulative_reported_l2,
                model: model_counters,
            });
        }

        self.l2 = staged_l2;
        self.cumulative_reported_l2 = cumulative_reported_l2;
        self.completed_launches += 1;
        Ok(outcome)
    }
}

fn require_replay_arch(
    expected: ArchId,
    got: ArchId,
    component: &'static str,
) -> Result<(), ReplayError> {
    if got != expected {
        return Err(ReplayError::ArchitectureMismatch {
            component,
            expected,
            got,
        });
    }
    Ok(())
}

fn require_trace_architecture(td: &TraceDir, expected: ArchId) -> Result<(), ReplayError> {
    let binary_version = td.trace().binary_version;
    let numeric = u32::try_from(binary_version)
        .map_err(|_| ReplayError::TraceArchitectureVersionOutOfRange { binary_version })?;
    let traced =
        ArchId::from_numeric(numeric).map_err(|source| ReplayError::TraceArchitectureUnknown {
            binary_version: numeric,
            source,
        })?;
    require_replay_arch(expected, traced, "trace binary_version")
}

fn validate_replay_mode_architecture(arch: ArchId, mode: ReplayMemMode) -> Result<(), ReplayError> {
    match mode {
        ReplayMemMode::Flat => {}
        ReplayMemMode::L1Const(l1) => {
            require_replay_arch(arch, l1.arch, "L1 configuration")?;
        }
        ReplayMemMode::L1SharedL2 { l1, l2, dram } => {
            require_replay_arch(arch, l1.arch, "L1 configuration")?;
            require_replay_arch(arch, l2.arch, "L2 configuration")?;
            require_replay_arch(arch, dram.arch(), "DRAM configuration")?;
        }
    }
    Ok(())
}

fn validate_replay_architecture(
    schedule: &FunctionSchedule,
    params: &IssueModelParams,
    mode: ReplayMemMode,
    memory_admission_feedback: Option<MemoryAdmissionCandidate>,
) -> Result<ReplayConfiguration, ReplayError> {
    require_replay_arch(schedule.arch, params.arch, "issue parameters")?;
    match memory_admission_feedback {
        Some(candidate) => {
            ReplayConfiguration::checked_with_memory_admission_feedback(*params, mode, candidate)
        }
        None => ReplayConfiguration::checked(*params, mode),
    }
}

fn async_copy_destination_fallback_sms(
    payload_contract: MemoryPayloadContract,
    per_sm_blocks: &BTreeMap<i32, Vec<BlockReplay>>,
) -> BTreeSet<i32> {
    per_sm_blocks
        .iter()
        .filter_map(|(&sm, blocks)| {
            blocks
                .iter()
                .flat_map(|block| &block.warps)
                .flat_map(|warp| &warp.insts)
                .any(|inst| {
                    matches!(inst.kind, AccessKind::GlobalAsyncCopy { .. })
                        && (payload_contract
                            != MemoryPayloadContract::AddressCarryingLdgstsDestinationV3
                            || matches!(
                                inst.kind,
                                AccessKind::GlobalAsyncCopy {
                                    l1_policy: LdgstsL1Policy::BypassL1,
                                    ..
                                }
                            ))
                })
                .then_some(sm)
        })
        .collect()
}

fn replay_kernel_with_schedule_and_l2(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedule: &FunctionSchedule,
    arch_card: &ArchCard,
    params: &IssueModelParams,
    mode: ReplayMemMode,
    retained_l2: Option<&mut L2Model>,
) -> Result<KernelReplayOutcome, ReplayError> {
    replay_kernel_with_schedule_and_l2_and_candidates(
        td,
        loc,
        schedule,
        arch_card,
        params,
        mode,
        retained_l2,
        &[],
        &[],
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn replay_kernel_with_schedule_and_l2_and_candidates(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedule: &FunctionSchedule,
    arch_card: &ArchCard,
    params: &IssueModelParams,
    mode: ReplayMemMode,
    retained_l2: Option<&mut L2Model>,
    memory_service_candidates: &[MemoryServiceCandidate],
    memory_admission_candidates: &[MemoryAdmissionCandidate],
    memory_admission_feedback: Option<MemoryAdmissionCandidate>,
) -> Result<KernelReplayOutcome, ReplayError> {
    if !memory_service_candidates.is_empty() && !matches!(mode, ReplayMemMode::L1SharedL2 { .. }) {
        return Err(ReplayError::MemoryServiceCandidatesRequireSharedL2);
    }
    if !memory_admission_candidates.is_empty() && !matches!(mode, ReplayMemMode::L1SharedL2 { .. })
    {
        return Err(ReplayError::MemoryAdmissionCandidatesRequireSharedL2);
    }
    if memory_admission_feedback.is_some() && !matches!(mode, ReplayMemMode::L1SharedL2 { .. }) {
        return Err(ReplayError::MemoryAdmissionCandidatesRequireSharedL2);
    }
    validate_memory_service_candidates(memory_service_candidates).map_err(IssueError::from)?;
    validate_memory_admission_candidates(memory_admission_candidates).map_err(IssueError::from)?;
    if let Some(candidate) = memory_admission_feedback {
        candidate.validate().map_err(IssueError::from)?;
    }
    let configuration =
        validate_replay_architecture(schedule, params, mode, memory_admission_feedback)?;
    let placement = PlacementConfig::from_arch_card(arch_card, schedule.arch)?;
    require_trace_architecture(td, schedule.arch)?;
    let decode_sectors = !matches!(mode, ReplayMemMode::Flat);
    let ExtractedKernel {
        per_sm_blocks,
        records_seen,
        coverage,
        shared_memory_residency,
        front_end,
        trace_metrics,
    } = extract_per_sm_blocks(td, loc, schedule, placement, decode_sectors)?;
    let instrumentation_order = observe_async_instrumentation_order(&per_sm_blocks, schedule)?;
    #[cfg(feature = "campaign")]
    if !memory_service_candidates.is_empty() && matches!(&coverage, ExtractedCoverage::Sampled(_)) {
        return Err(ReplayError::MemoryServiceCandidatesRequireFullCoverage);
    }
    #[cfg(feature = "campaign")]
    if !memory_admission_candidates.is_empty() && matches!(&coverage, ExtractedCoverage::Sampled(_))
    {
        return Err(ReplayError::MemoryAdmissionCandidatesRequireFullCoverage);
    }
    #[cfg(feature = "campaign")]
    if memory_admission_feedback.is_some() && matches!(&coverage, ExtractedCoverage::Sampled(_)) {
        return Err(ReplayError::MemoryAdmissionCandidatesRequireFullCoverage);
    }
    let async_copy_destination_fallback_sms =
        async_copy_destination_fallback_sms(td.memory_payload_contract(), &per_sm_blocks);
    let mut replay = match mode {
        ReplayMemMode::Flat | ReplayMemMode::L1Const(_) => {
            let l1_config = match mode {
                ReplayMemMode::L1Const(cfg) => Some(cfg),
                _ => None,
            };
            let mut per_sm = BTreeMap::new();
            for (sm, blocks) in &per_sm_blocks {
                let mut result = simulate_sm_with_residency(
                    blocks,
                    schedule,
                    params,
                    shared_memory_residency,
                    l1_config,
                )?;
                if async_copy_destination_fallback_sms.contains(sm) {
                    result.unmodeled_tails.ldgsts_dst_shared_timing = true;
                }
                if let Some(observation) = instrumentation_order.get(sm) {
                    result.readiness.async_instrumentation_order = Some(Box::new(*observation));
                }
                per_sm.insert(*sm, result);
            }
            fold_kernel_result(
                schedule.arch,
                per_sm,
                records_seen,
                l1_config.is_some(),
                None,
                None,
            )?
        }
        ReplayMemMode::L1SharedL2 { l1, l2, dram } => {
            let mut fresh_l2;
            let l2_model = match retained_l2 {
                Some(model) => {
                    if *model.config() != l2 {
                        return Err(ReplayError::RetainedL2ConfigurationMismatch {
                            expected: l2,
                            got: *model.config(),
                        });
                    }
                    model
                }
                None => {
                    fresh_l2 = L2Model::new(l2)?;
                    &mut fresh_l2
                }
            };
            let pending_sectors = l2_model.pending_sector_count();
            if pending_sectors != 0 {
                return Err(ReplayError::RetainedL2NotDrained { pending_sectors });
            }
            let counters_before = *l2_model.counters()?;
            let mut sys = simulate_kernel_shared_with_residency_and_candidates(
                &per_sm_blocks,
                schedule,
                params,
                shared_memory_residency,
                l1,
                l2_model,
                dram,
                memory_service_candidates,
                memory_admission_candidates,
                memory_admission_feedback,
            )?;
            let pending_sectors = l2_model.pending_sector_count();
            if pending_sectors != 0 {
                return Err(ReplayError::RetainedL2NotDrained { pending_sectors });
            }
            let launch_l2 = sys.l2.checked_delta_since(&counters_before)?;
            for sm in &async_copy_destination_fallback_sms {
                if let Some(result) = sys.per_sm.get_mut(sm) {
                    result.unmodeled_tails.ldgsts_dst_shared_timing = true;
                }
            }
            for (sm, observation) in &instrumentation_order {
                let result = sys.per_sm.get_mut(sm).unwrap_or_else(|| {
                    unreachable!("order observation SM escaped the simulated SM roster")
                });
                result.readiness.async_instrumentation_order = Some(Box::new(*observation));
            }
            if !async_copy_destination_fallback_sms.is_empty() {
                sys.unmodeled_tails.ldgsts_dst_shared_timing = true;
            }
            require_replay_arch(schedule.arch, sys.arch, "system result")?;
            let system_unmodeled_tails = sys.unmodeled_tails;
            let replay = fold_kernel_result_with_sm_active_time(
                schedule.arch,
                sys.per_sm,
                records_seen,
                true,
                sys.terminal_event_time,
                sys.sm_active_terminal_event_time,
                Some((launch_l2, sys.l2_by_source, sys.dram)),
            )?;
            require_system_tail_consistency(system_unmodeled_tails, replay.unmodeled_tails)?;
            replay
        }
    };
    replay.front_end = front_end;
    replay.trace_metrics = trace_metrics;
    Ok(classify_kernel_result(coverage, configuration, replay))
}

fn replay_kernel_with_schedule(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedule: &FunctionSchedule,
    arch_card: &ArchCard,
    params: &IssueModelParams,
    mode: ReplayMemMode,
) -> Result<KernelReplayOutcome, ReplayError> {
    replay_kernel_with_schedule_and_l2(td, loc, schedule, arch_card, params, mode, None)
}

/// Replay the one launch represented by [`AdmittedLaunchEvidence`]. This is the
/// compatibility entry point used by the optional historical campaign runner.
///
/// Caches are fresh per kernel (cold, NCU `--cache-control all` parity).
#[cfg(feature = "campaign")]
pub fn replay_kernel(
    evidence: &AdmittedLaunchEvidence,
    params: &IssueModelParams,
    mode: ReplayMemMode,
) -> Result<KernelReplayOutcome, ReplayError> {
    let identity = evidence.identity();
    let td = evidence.trace_dir();
    let loc = td
        .kernels()
        .find(|loc| {
            loc.device_id() == identity.device_id()
                && loc.stream_id() == identity.stream_id()
                && loc.kernel().id == identity.kernel_id()
                && loc.kernel().function_unique_id == identity.function_unique_id()
        })
        .ok_or(ReplayError::AdmittedLaunchMissing {
            device_id: identity.device_id(),
            stream_id: identity.stream_id(),
            kernel_id: identity.kernel_id(),
        })?;
    replay_kernel_with_schedule(
        td,
        &loc,
        evidence.schedule(),
        evidence.arch_card(),
        params,
        mode,
    )
}

/// Replay one launch directly from loaded trace, schedule, architecture-card,
/// and model inputs. This is the ordinary research entry point; it does not
/// require an evidence manifest or an admission step.
pub fn replay_trace_kernel(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedules: &BTreeMap<i32, FunctionSchedule>,
    arch_card: &ArchCard,
    params: &IssueModelParams,
    mode: ReplayMemMode,
) -> Result<KernelReplayOutcome, ReplayError> {
    let schedule = lookup_schedule(schedules, loc)?;
    replay_kernel_with_schedule(td, loc, schedule, arch_card, params, mode)
}

/// Replay one launch while evaluating a bounded set of observational streaming
/// memory-service hypotheses at the exact issue timestamps. Candidates neither
/// delay issue nor add events; their per-SM diagnostics expose counterfactual
/// service busy time, queueing, and terminal completion. This research surface
/// is intentionally restricted to the full shared-L2 hierarchy mode.
pub fn replay_trace_kernel_with_memory_service_candidates(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedules: &BTreeMap<i32, FunctionSchedule>,
    arch_card: &ArchCard,
    params: &IssueModelParams,
    mode: ReplayMemMode,
    memory_service_candidates: &[MemoryServiceCandidate],
) -> Result<KernelReplayOutcome, ReplayError> {
    let schedule = lookup_schedule(schedules, loc)?;
    replay_kernel_with_schedule_and_l2_and_candidates(
        td,
        loc,
        schedule,
        arch_card,
        params,
        mode,
        None,
        memory_service_candidates,
        &[],
        None,
    )
}

/// Replay one launch while evaluating bounded finite operand-admission queues
/// against exact per-instruction L2 hit/miss outcomes. The evaluator preserves
/// the observed issue and cache order and emits schema-9 diagnostics only; it
/// does not alter scheduler timing or hierarchy state.
pub fn replay_trace_kernel_with_memory_admission_candidates(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedules: &BTreeMap<i32, FunctionSchedule>,
    arch_card: &ArchCard,
    params: &IssueModelParams,
    mode: ReplayMemMode,
    memory_admission_candidates: &[MemoryAdmissionCandidate],
) -> Result<KernelReplayOutcome, ReplayError> {
    let schedule = lookup_schedule(schedules, loc)?;
    replay_kernel_with_schedule_and_l2_and_candidates(
        td,
        loc,
        schedule,
        arch_card,
        params,
        mode,
        None,
        &[],
        memory_admission_candidates,
        None,
    )
}

/// Replay one launch with one feedback-coupled finite admission candidate.
/// Unlike schema-9 observation, this lane participates in warp readiness and
/// therefore may change issue and cache order. It remains explicit opt-in.
pub fn replay_trace_kernel_with_memory_admission_feedback(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedules: &BTreeMap<i32, FunctionSchedule>,
    arch_card: &ArchCard,
    params: &IssueModelParams,
    mode: ReplayMemMode,
    memory_admission_feedback: MemoryAdmissionCandidate,
) -> Result<KernelReplayOutcome, ReplayError> {
    let schedule = lookup_schedule(schedules, loc)?;
    replay_kernel_with_schedule_and_l2_and_candidates(
        td,
        loc,
        schedule,
        arch_card,
        params,
        mode,
        None,
        &[],
        &[],
        Some(memory_admission_feedback),
    )
}

/// Replay caller-supplied trace/schedule inputs for debugging only. The returned
/// diagnostics retain counters and directional timestamps but erase the full
/// replay type, so they cannot enter the legacy launch-set aggregation API.
#[cfg(feature = "campaign")]
pub fn replay_unadmitted_diagnostics(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedules: &BTreeMap<i32, FunctionSchedule>,
    arch_card: &ArchCard,
    params: &IssueModelParams,
    mode: ReplayMemMode,
) -> Result<UnadmittedReplayDiagnostics, ReplayError> {
    let schedule = lookup_schedule(schedules, loc)?;
    let outcome = replay_kernel_with_schedule(td, loc, schedule, arch_card, params, mode)?;
    Ok(UnadmittedReplayDiagnostics::from_outcome(outcome))
}

#[cfg(test)]
mod instrumentation_order_tests {
    use super::*;
    use crate::control_bits::ControlBits;
    use crate::program::{IssueSlot, PipelineClass};
    use sim_mem::LdgstsL1Policy;

    fn slot(op_class: OpClass, ldgsts: bool) -> IssueSlot {
        IssueSlot {
            control: ControlBits::new(0, false, 7, 7, 0, 0).unwrap(),
            op_class,
            async_copy_wait_max_pending_groups: 0,
            pipeline_class: PipelineClass::DispatchOnly,
            mem_role: if ldgsts {
                MemRole::BypassLoad(CacheOpToken::Ldgsts)
            } else {
                MemRole::None
            },
            ldgsts_l1_policy: ldgsts.then_some(LdgstsL1Policy::BypassL1),
            fixed_latency_operands: crate::sass_operands::FixedLatencyOperandStatus::NotApplicable,
            clocklo_read: false,
            gpu_membar: false,
            opcode_text: match op_class {
                OpClass::AsyncCopyCommit => "LDGDEPBAR",
                OpClass::AsyncCopyWait => "DEPBAR.LE SB0, 0x0",
                OpClass::BlockBarrier | OpClass::Other if ldgsts => "LDGSTS.E.BYPASS.128",
                OpClass::BlockBarrier | OpClass::Other => "IADD3",
            }
            .into(),
        }
    }

    fn schedule() -> FunctionSchedule {
        FunctionSchedule {
            arch: ArchId::Sm86,
            function_unique_id: 1,
            kernel_name: "ordered".into(),
            function_name: "ordered".into(),
            by_pc: BTreeMap::from([
                (0, slot(OpClass::Other, true)),
                (4, slot(OpClass::AsyncCopyCommit, false)),
                (8, slot(OpClass::AsyncCopyWait, false)),
                (12, slot(OpClass::Other, false)),
            ]),
        }
    }

    fn warp(warp_slot: i32, orders: [u64; 3]) -> WarpReplay {
        WarpReplay {
            warp_slot,
            insts: vec![
                ReplayInst {
                    pc: 0,
                    kind: AccessKind::GlobalAsyncCopy {
                        off: 0,
                        len: 0,
                        completion_work: 0,
                        l1_policy: LdgstsL1Policy::BypassL1,
                    },
                },
                ReplayInst {
                    pc: 4,
                    kind: AccessKind::NonMemory,
                },
                ReplayInst {
                    pc: 8,
                    kind: AccessKind::NonMemory,
                },
            ],
            sectors: Vec::new(),
            async_copy_cohorts: Some(Box::new(AsyncCopyCohortReplay {
                instrumentation_order: vec![
                    InstrumentationOrderAnchor {
                        instruction: 0,
                        order: orders[0],
                    },
                    InstrumentationOrderAnchor {
                        instruction: 1,
                        order: orders[1],
                    },
                    InstrumentationOrderAnchor {
                        instruction: 2,
                        order: orders[2],
                    },
                ],
                ..AsyncCopyCohortReplay::default()
            })),
        }
    }

    fn blocks(first: [u64; 3], second: [u64; 3]) -> BTreeMap<i32, Vec<BlockReplay>> {
        BTreeMap::from([(
            0,
            vec![BlockReplay {
                launch_index: 0,
                warps: vec![warp(0, first), warp(1, second)],
            }],
        )])
    }

    #[test]
    fn selected_order_scan_reconstructs_sibling_committed_population() {
        let observations =
            observe_async_instrumentation_order(&blocks([1, 2, 5], [3, 4, 6]), &schedule())
                .unwrap();
        assert_eq!(
            observations[&0],
            AsyncInstrumentationOrderDiagnostics {
                selected_events: 6,
                issue_events: 2,
                commit_events: 2,
                wait_events: 2,
                issue_events_with_other_warp_committed_groups: 1,
                other_warp_committed_groups_sum: 1,
                maximum_other_warp_committed_groups: 1,
                protocol_contradictions: 0,
            }
        );
    }

    #[test]
    fn selected_order_scan_fails_closed_on_density_partial_or_extra_evidence() {
        assert!(matches!(
            observe_async_instrumentation_order(&blocks([1, 2, 5], [1, 4, 6]), &schedule()),
            Err(ReplayError::InstrumentationOrderDuplicate { order: 1, .. })
        ));

        assert!(matches!(
            observe_async_instrumentation_order(&blocks([1, 2, 5], [3, 4, 7]), &schedule()),
            Err(ReplayError::InstrumentationOrderNotDense {
                expected: 6,
                got: 7,
            })
        ));

        let mut partial = blocks([1, 2, 5], [3, 4, 6]);
        partial.get_mut(&0).unwrap()[0].warps[1]
            .async_copy_cohorts
            .as_mut()
            .unwrap()
            .instrumentation_order
            .remove(1);
        assert!(matches!(
            observe_async_instrumentation_order(&partial, &schedule()),
            Err(ReplayError::InstrumentationOrderMissing { instruction: 1, .. })
        ));

        let mut extra = blocks([1, 2, 5], [3, 4, 6]);
        let warp = &mut extra.get_mut(&0).unwrap()[0].warps[0];
        warp.insts.push(ReplayInst {
            pc: 12,
            kind: AccessKind::NonMemory,
        });
        warp.async_copy_cohorts
            .as_mut()
            .unwrap()
            .instrumentation_order
            .push(InstrumentationOrderAnchor {
                instruction: 3,
                order: 7,
            });
        assert!(matches!(
            observe_async_instrumentation_order(&extra, &schedule()),
            Err(ReplayError::InstrumentationOrderUnexpected { instruction: 3, .. })
        ));
    }

    #[test]
    fn schema_21_aggregation_is_checked_and_observation_only() {
        let mut aggregate = crate::ReadinessDiagnostics {
            async_instrumentation_order: Some(Box::new(AsyncInstrumentationOrderDiagnostics {
                selected_events: 3,
                issue_events: 1,
                commit_events: 1,
                wait_events: 1,
                issue_events_with_other_warp_committed_groups: 1,
                other_warp_committed_groups_sum: 2,
                maximum_other_warp_committed_groups: 2,
                protocol_contradictions: 0,
            })),
            ..crate::ReadinessDiagnostics::default()
        };
        aggregate
            .add_checked(&crate::ReadinessDiagnostics {
                async_instrumentation_order: Some(Box::new(AsyncInstrumentationOrderDiagnostics {
                    selected_events: 3,
                    issue_events: 1,
                    commit_events: 1,
                    wait_events: 1,
                    issue_events_with_other_warp_committed_groups: 0,
                    other_warp_committed_groups_sum: 0,
                    maximum_other_warp_committed_groups: 0,
                    protocol_contradictions: 0,
                })),
                ..crate::ReadinessDiagnostics::default()
            })
            .unwrap();
        assert_eq!(aggregate.schema_version(), 21);
        assert_eq!(
            aggregate.async_instrumentation_order.as_deref(),
            Some(&AsyncInstrumentationOrderDiagnostics {
                selected_events: 6,
                issue_events: 2,
                commit_events: 2,
                wait_events: 2,
                issue_events_with_other_warp_committed_groups: 1,
                other_warp_committed_groups_sum: 2,
                maximum_other_warp_committed_groups: 2,
                protocol_contradictions: 0,
            })
        );

        let mut overflow = crate::ReadinessDiagnostics {
            async_instrumentation_order: Some(Box::new(AsyncInstrumentationOrderDiagnostics {
                selected_events: u64::MAX,
                ..AsyncInstrumentationOrderDiagnostics::default()
            })),
            ..crate::ReadinessDiagnostics::default()
        };
        let before = overflow.clone();
        let error = overflow
            .add_checked(&crate::ReadinessDiagnostics {
                async_instrumentation_order: Some(Box::new(AsyncInstrumentationOrderDiagnostics {
                    selected_events: 1,
                    ..AsyncInstrumentationOrderDiagnostics::default()
                })),
                ..crate::ReadinessDiagnostics::default()
            })
            .unwrap_err();
        assert!(matches!(
            error,
            crate::ReadinessCounterError::Overflow {
                field: "readiness.async_instrumentation_order.selected_events",
                ..
            }
        ));
        assert_eq!(overflow, before);
    }
}

#[cfg(test)]
mod stream_l2_session_tests {
    use super::*;
    use crate::control_bits::ControlBits;
    use crate::issue::IssueResourceSource;
    use crate::program::{IssueSlot, OpClass, PipelineClass};
    use prost::Message;
    use sim_arch::LdgstsCompletionThroughput;
    use sim_mem::{L1Config, L2Replacement, SliceHash};
    use sim_trace::lanes::{FORMAT_LIST_ALL, LANE_INACTIVE_SENTINEL};
    use sim_trace::reader::ADDRESS_CARRYING_TRACE_CONTRACT_ID;

    const BASE: i64 = 0x7000_0000;

    fn params() -> IssueModelParams {
        IssueModelParams {
            arch: ArchId::Sm86,
            resource_source: IssueResourceSource::SyntheticTestFixture,
            instruction_front_end: None,
            memory_issue_pipeline: None,
            scalar_execution_pipeline: None,
            issue_ports: 1,
            dual_issue_supported: false,
            ldgsts_l1_hit_return_service: None,
            constant_memory_return_service: None,
            immediate_constant_cache: None,
            l1_hit_latency: 1,
            l1_miss_latency: 2,
            global_flat_latency: 4,
            shared_write_barrier_latency: 1,
            shared_additional_wavefront_latency: 2,
            non_memory_write_barrier_latency: 1,
            read_barrier_latency: 1,
            ldgsts_l1_mshr_floor: std::num::NonZeroU32::new(1).unwrap(),
            ldgsts_completion_throughput: LdgstsCompletionThroughput {
                cache_all_payload_bytes_per_cycle: 64,
                bypass_payload_bytes_per_cycle: 32,
            },
            ldgsts_cache_all_destination_wavefront_cycles: 2,
            ldgsts_global_to_shared_transport_latency: 0,
            ldgsts_cache_all_destination_service_wavefronts: 2,
            ldgsts_cache_all_destination_service_cycles: 3,
            max_resident_warps_per_sm: 48,
            max_resident_blocks_per_sm: u32::MAX,
        }
    }

    fn shared_mode(associativity: u32, capacity_bytes: u64) -> ReplayMemMode {
        ReplayMemMode::L1SharedL2 {
            l1: L1Config {
                arch: ArchId::Sm86,
                line_capacity: 1,
                mshr_count: Some(1),
                hit_latency: 1,
                miss_latency: 2,
            },
            l2: L2Config {
                arch: ArchId::Sm86,
                capacity_bytes,
                line_bytes: 128,
                sector_bytes: 32,
                num_slices: 1,
                associativity,
                replacement: L2Replacement::Lru,
                slice_hash: SliceHash::LineMod,
            },
            dram: DramBw::Unmodeled { arch: ArchId::Sm86 },
        }
    }

    fn load(function_unique_id: i32, address: i64, width: i32) -> pb::Instruction {
        let mut addrs = vec![LANE_INACTIVE_SENTINEL; 32];
        addrs[0] = address;
        pb::Instruction {
            pc: 16,
            active_mask: Some(1),
            predicate_mask: Some(1),
            function_unique_id,
            vpc: Some(16),
            sm_id: Some(0),
            warpid_sm: Some(0),
            warpid_tb: Some(0),
            data_width: Some(width),
            memory_op_class: Some(pb::MemoryOpClass::Global as i32),
            memory_op_access_kind: Some(pb::MemoryOpAccessKind::Load as i32),
            addresses: vec![pb::Address {
                data_width: width,
                compression_format: FORMAT_LIST_ALL,
                addrs,
                allocation_id: Some(1),
                address_space: Some(pb::MemorySpace::Global as i32),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn schedule(function_unique_id: i32) -> FunctionSchedule {
        FunctionSchedule {
            arch: ArchId::Sm86,
            function_unique_id,
            kernel_name: format!("k{function_unique_id}"),
            function_name: format!("f{function_unique_id}"),
            by_pc: BTreeMap::from([(
                16,
                IssueSlot {
                    control: ControlBits::new(0, false, 7, 7, 0, 0).unwrap(),
                    op_class: OpClass::Other,
                    async_copy_wait_max_pending_groups: 0,
                    pipeline_class: PipelineClass::DispatchOnly,
                    mem_role: MemRole::EligibleLoad(CacheOpToken::LdgECa),
                    ldgsts_l1_policy: None,
                    fixed_latency_operands:
                        crate::sass_operands::FixedLatencyOperandStatus::NotApplicable,
                    clocklo_read: false,
                    gpu_membar: false,
                    opcode_text: "LDG.E.CA R0, [R2.64]".into(),
                },
            )]),
        }
    }

    fn bypass_schedule(function_unique_id: i32) -> FunctionSchedule {
        let mut schedule = schedule(function_unique_id);
        let slot = schedule.by_pc.get_mut(&16).unwrap();
        slot.mem_role = MemRole::BypassLoad(CacheOpToken::LdgECg);
        slot.opcode_text = "LDG.E.CG R0, [R2.64]".into();
        schedule
    }

    fn write_stream(root: &Path, stream_id: i64, launches: &[(i32, i32, Vec<pb::Instruction>)]) {
        let kernels = launches
            .iter()
            .map(|&(kernel_id, function_unique_id, _)| pb::Kernel {
                id: kernel_id,
                name: format!("k{function_unique_id}"),
                function_unique_id,
                size_shared_memory: 0,
                number_of_registers: 8,
                shared_memory_base_address: 0,
                local_memory_base_address: 0,
                grid_dim: Some(pb::Dim3d { x: 1, y: 1, z: 1 }),
                block_dim: Some(pb::Dim3d { x: 32, y: 1, z: 1 }),
                cluster_dim: Some(pb::Dim3d { x: 1, y: 1, z: 1 }),
                shared_memory_static_bytes: None,
                shared_memory_dynamic_bytes: None,
                shared_memory_carveout_preference_pct: None,
                driver_occupancy_limit_blocks: None,
            })
            .collect();
        let stream = pb::CudaStream {
            id: stream_id,
            kernels,
            ordered_cuda_events: Vec::new(),
        };
        let device = pb::GpuDevice {
            id: 0,
            streams: [(stream_id, stream)].into_iter().collect(),
        };
        let trace = pb::Trace {
            name: "stream-l2-session".into(),
            binary_version: 86,
            nvbit_version: "1.8".into(),
            accelsim_version: 0,
            is_gathered_registers_values: false,
            gpu_device: [(0, device)].into_iter().collect(),
            trace_contract_id: ADDRESS_CARRYING_TRACE_CONTRACT_ID.into(),
            memory_payload_classes: vec![
                pb::MemoryOpClass::Global as i32,
                pb::MemoryOpClass::Shared as i32,
                pb::MemoryOpClass::Local as i32,
                pb::MemoryOpClass::Atomic as i32,
                pb::MemoryOpClass::AsyncCopy as i32,
            ],
        };
        std::fs::write(root.join("dynamic_trace.pb"), trace.encode_to_vec()).unwrap();
        let extra = root.join("extra_info");
        std::fs::create_dir_all(&extra).unwrap();
        std::fs::write(
            extra.join("allocations.json"),
            r#"[{"allocation_id":1,"context_id":1,"device_id":0,"base_address":"0x70000000","size_bytes":4096,"alive":true}]"#,
        )
        .unwrap();
        for &(kernel_id, _, ref instructions) in launches {
            let directory = root.join(format!(
                "threadblocks/device_0/stream_{stream_id}/kernel_{kernel_id}"
            ));
            std::fs::create_dir_all(&directory).unwrap();
            let block = pb::Threadblock {
                block_id: Some(pb::Dim3d { x: 0, y: 0, z: 0 }),
                warps: [(
                    0,
                    pb::Warp {
                        id: 0,
                        instructions: instructions.clone(),
                    },
                )]
                .into_iter()
                .collect(),
            };
            std::fs::write(
                directory.join(format!("d_0_s_{stream_id}_k_{kernel_id}_0,0,0.pb")),
                block.encode_to_vec(),
            )
            .unwrap();
        }
    }

    fn write_trace(root: &Path, launches: &[(i32, i32, Vec<pb::Instruction>)]) {
        write_stream(root, 0, launches);
    }

    fn schedules(functions: &[i32]) -> BTreeMap<i32, FunctionSchedule> {
        functions
            .iter()
            .map(|&function| (function, schedule(function)))
            .collect()
    }

    fn full(outcome: KernelReplayOutcome) -> FullKernelReplayResult {
        outcome.into_full().expect("synthetic trace is complete")
    }

    #[test]
    fn stream_l2_converts_a_cold_target_miss_to_a_per_launch_hit_delta() {
        let tmp = tempfile::tempdir().unwrap();
        write_trace(
            tmp.path(),
            &[
                (1, 1, vec![load(1, BASE, 4)]),
                (2, 2, vec![load(2, BASE, 4)]),
            ],
        );
        let trace = TraceDir::open_direct(tmp.path()).unwrap();
        let locs: Vec<_> = trace.kernels().collect();
        let schedules = schedules(&[1, 2]);
        let card = sm86_test_card();
        let params = params();
        let mode = shared_mode(2, 256);

        for loc in &locs {
            let cold =
                full(replay_trace_kernel(&trace, loc, &schedules, &card, &params, mode).unwrap());
            assert_eq!(cold.l2().unwrap().hits, 0);
            assert_eq!(cold.l2().unwrap().misses, 1);
        }

        let mut session = StreamL2ReplaySession::new(&trace, &card, &params, mode, 0, 0).unwrap();
        let seed = full(
            session
                .replay_next(&trace, &locs[0], &schedules, &card, &params, mode)
                .unwrap(),
        );
        assert_eq!((seed.l2().unwrap().hits, seed.l2().unwrap().misses), (0, 1));
        assert_eq!(session.pending_l2_sectors(), 0);

        let target = full(
            session
                .replay_next(&trace, &locs[1], &schedules, &card, &params, mode)
                .unwrap(),
        );
        assert_eq!(
            (target.l2().unwrap().hits, target.l2().unwrap().misses),
            (1, 0)
        );
        assert_eq!(target.l1().unwrap().misses, 1, "L1 is fresh per launch");
        assert_eq!(session.pending_l2_sectors(), 0);
        assert!(session.is_complete());
        assert_eq!(session.completed_launches(), 2);
        assert_eq!(session.remaining_launches(), 0);
        assert_eq!(
            (
                session.cumulative_l2_counters().hits,
                session.cumulative_l2_counters().misses,
            ),
            (1, 1)
        );
    }

    #[test]
    fn partial_sector_handoff_reports_one_hit_and_one_fill_delta() {
        let tmp = tempfile::tempdir().unwrap();
        write_trace(
            tmp.path(),
            &[
                (1, 1, vec![load(1, BASE, 4)]),
                (2, 2, vec![load(2, BASE, 64)]),
            ],
        );
        let trace = TraceDir::open_direct(tmp.path()).unwrap();
        let locs: Vec<_> = trace.kernels().collect();
        let schedules = schedules(&[1, 2]);
        let card = sm86_test_card();
        let params = params();
        let mode = shared_mode(2, 256);
        let mut session = StreamL2ReplaySession::new(&trace, &card, &params, mode, 0, 0).unwrap();
        session
            .replay_next(&trace, &locs[0], &schedules, &card, &params, mode)
            .unwrap();
        let target = full(
            session
                .replay_next(&trace, &locs[1], &schedules, &card, &params, mode)
                .unwrap(),
        );
        let l2 = target.l2().unwrap();
        assert_eq!((l2.hits, l2.misses), (1, 1));
        assert_eq!(l2.sector_fill_misses, 1);
        assert_eq!(
            (
                session.cumulative_l2_counters().hits,
                session.cumulative_l2_counters().misses,
            ),
            (1, 2)
        );
    }

    #[test]
    fn replacement_recency_crosses_the_launch_boundary() {
        let tmp = tempfile::tempdir().unwrap();
        write_trace(
            tmp.path(),
            &[
                (
                    1,
                    1,
                    vec![load(1, BASE, 4), load(1, BASE + 128, 4), load(1, BASE, 4)],
                ),
                (2, 2, vec![load(2, BASE + 256, 4), load(2, BASE + 128, 4)]),
            ],
        );
        let trace = TraceDir::open_direct(tmp.path()).unwrap();
        let locs: Vec<_> = trace.kernels().collect();
        let schedules = BTreeMap::from([(1, bypass_schedule(1)), (2, bypass_schedule(2))]);
        let card = sm86_test_card();
        let params = params();
        let mode = shared_mode(2, 256);
        let mut session = StreamL2ReplaySession::new(&trace, &card, &params, mode, 0, 0).unwrap();
        session
            .replay_next(&trace, &locs[0], &schedules, &card, &params, mode)
            .unwrap();
        let target = full(
            session
                .replay_next(&trace, &locs[1], &schedules, &card, &params, mode)
                .unwrap(),
        );
        assert_eq!(
            (target.l2().unwrap().hits, target.l2().unwrap().misses),
            (0, 2),
            "C must evict LRU B; re-reading B then misses, proving recency survived"
        );
    }

    #[test]
    fn failed_second_launch_preserves_cache_cursor_and_allows_retry() {
        let tmp = tempfile::tempdir().unwrap();
        write_trace(
            tmp.path(),
            &[
                (1, 1, vec![load(1, BASE, 4)]),
                (2, 2, vec![load(2, BASE, 4)]),
            ],
        );
        let trace = TraceDir::open_direct(tmp.path()).unwrap();
        let locs: Vec<_> = trace.kernels().collect();
        let first_only = schedules(&[1]);
        let complete = schedules(&[1, 2]);
        let card = sm86_test_card();
        let params = params();
        let mode = shared_mode(2, 256);
        let mut session = StreamL2ReplaySession::new(&trace, &card, &params, mode, 0, 0).unwrap();
        session
            .replay_next(&trace, &locs[0], &first_only, &card, &params, mode)
            .unwrap();
        let before = session.cumulative_l2_counters();
        assert!(matches!(
            session.replay_next(&trace, &locs[1], &first_only, &card, &params, mode),
            Err(ReplaySessionError::Replay(ReplayError::ScheduleMissing {
                function_unique_id: 2
            }))
        ));
        assert_eq!(session.completed_launches(), 1);
        assert_eq!(session.cumulative_l2_counters(), before);
        assert_eq!(session.pending_l2_sectors(), 0);

        let target = full(
            session
                .replay_next(&trace, &locs[1], &complete, &card, &params, mode)
                .unwrap(),
        );
        assert_eq!(
            (target.l2().unwrap().hits, target.l2().unwrap().misses),
            (1, 0)
        );
        assert!(session.is_complete());
    }

    #[test]
    fn same_path_reopened_trace_cannot_reuse_stream_l2_state() {
        // A path check alone is insufficient: a caller can replace the files
        // in place and reopen a valid trace with the same launch IDs.  The
        // retained L2 state must not be combined with that new payload.
        let tmp = tempfile::tempdir().unwrap();
        write_trace(
            tmp.path(),
            &[
                (1, 1, vec![load(1, BASE, 4)]),
                (2, 2, vec![load(2, BASE, 4)]),
            ],
        );
        let trace = TraceDir::open_direct(tmp.path()).unwrap();
        let locs: Vec<_> = trace.kernels().collect();
        let schedules = schedules(&[1, 2]);
        let card = sm86_test_card();
        let params = params();
        let mode = shared_mode(2, 256);
        let mut session = StreamL2ReplaySession::new(&trace, &card, &params, mode, 0, 0).unwrap();
        session
            .replay_next(&trace, &locs[0], &schedules, &card, &params, mode)
            .unwrap();
        let before = session.cumulative_l2_counters();

        let replacement = tempfile::tempdir().unwrap();
        write_trace(
            replacement.path(),
            &[
                (1, 1, vec![load(1, BASE + 128, 4)]),
                (2, 2, vec![load(2, BASE + 128, 4)]),
            ],
        );
        for relative in [
            "dynamic_trace.pb",
            "extra_info/allocations.json",
            "threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_0,0,0.pb",
            "threadblocks/device_0/stream_0/kernel_2/d_0_s_0_k_2_0,0,0.pb",
        ] {
            std::fs::copy(replacement.path().join(relative), tmp.path().join(relative)).unwrap();
        }
        let reopened = TraceDir::open_direct(tmp.path()).unwrap();
        let reopened_loc = reopened.kernels().nth(1).unwrap();
        assert!(matches!(
            session.replay_next(&reopened, &reopened_loc, &schedules, &card, &params, mode),
            Err(ReplaySessionError::TraceLineageMismatch)
        ));
        assert_eq!(session.completed_launches(), 1);
        assert_eq!(session.cumulative_l2_counters(), before);
    }

    #[test]
    fn order_configuration_root_and_stream_mismatches_are_typed_and_nonmutating() {
        let tmp = tempfile::tempdir().unwrap();
        write_trace(
            tmp.path(),
            &[
                (1, 1, vec![load(1, BASE, 4)]),
                (2, 2, vec![load(2, BASE, 4)]),
            ],
        );
        let trace = TraceDir::open_direct(tmp.path()).unwrap();
        let locs: Vec<_> = trace.kernels().collect();
        let schedules = schedules(&[1, 2]);
        let card = sm86_test_card();
        let params = params();
        let mode = shared_mode(2, 256);
        let mut session = StreamL2ReplaySession::new(&trace, &card, &params, mode, 0, 0).unwrap();

        assert!(matches!(
            session.replay_next(&trace, &locs[1], &schedules, &card, &params, mode),
            Err(ReplaySessionError::LaunchOrderMismatch { position: 0, .. })
        ));
        assert_eq!(session.completed_launches(), 0);

        let wrong_mode = shared_mode(1, 128);
        assert!(matches!(
            session.replay_next(&trace, &locs[0], &schedules, &card, &params, wrong_mode),
            Err(ReplaySessionError::ConfigurationMismatch { .. })
        ));
        assert_eq!(session.completed_launches(), 0);

        let other = tempfile::tempdir().unwrap();
        write_trace(other.path(), &[(1, 1, vec![load(1, BASE, 4)])]);
        let other_trace = TraceDir::open_direct(other.path()).unwrap();
        let other_loc = other_trace.kernels().next().unwrap();
        assert!(matches!(
            session.replay_next(&other_trace, &other_loc, &schedules, &card, &params, mode),
            Err(ReplaySessionError::TraceRootMismatch { .. })
        ));

        let multi = tempfile::tempdir().unwrap();
        write_stream(multi.path(), 0, &[(1, 1, vec![load(1, BASE, 4)])]);
        let root_path = multi.path().join("dynamic_trace.pb");
        let mut root = pb::Trace::decode(std::fs::read(&root_path).unwrap().as_slice()).unwrap();
        let stream_one = pb::CudaStream {
            id: 1,
            kernels: vec![pb::Kernel {
                id: 3,
                name: "k1".into(),
                function_unique_id: 1,
                size_shared_memory: 0,
                number_of_registers: 8,
                shared_memory_base_address: 0,
                local_memory_base_address: 0,
                grid_dim: Some(pb::Dim3d { x: 1, y: 1, z: 1 }),
                block_dim: Some(pb::Dim3d { x: 32, y: 1, z: 1 }),
                cluster_dim: Some(pb::Dim3d { x: 1, y: 1, z: 1 }),
                shared_memory_static_bytes: None,
                shared_memory_dynamic_bytes: None,
                shared_memory_carveout_preference_pct: None,
                driver_occupancy_limit_blocks: None,
            }],
            ordered_cuda_events: Vec::new(),
        };
        root.gpu_device
            .get_mut(&0)
            .unwrap()
            .streams
            .insert(1, stream_one);
        std::fs::write(&root_path, root.encode_to_vec()).unwrap();
        let stream_one_directory = multi.path().join("threadblocks/device_0/stream_1/kernel_3");
        std::fs::create_dir_all(&stream_one_directory).unwrap();
        let stream_one_block = pb::Threadblock {
            block_id: Some(pb::Dim3d { x: 0, y: 0, z: 0 }),
            warps: [(
                0,
                pb::Warp {
                    id: 0,
                    instructions: vec![load(1, BASE, 4)],
                },
            )]
            .into_iter()
            .collect(),
        };
        std::fs::write(
            stream_one_directory.join("d_0_s_1_k_3_0,0,0.pb"),
            stream_one_block.encode_to_vec(),
        )
        .unwrap();
        let multi_trace = TraceDir::open_direct(multi.path()).unwrap();
        let stream_one_loc = multi_trace
            .kernels()
            .find(|loc| loc.stream_id() == 1)
            .unwrap();
        let mut multi_session =
            StreamL2ReplaySession::new(&multi_trace, &card, &params, mode, 0, 0).unwrap();
        assert!(matches!(
            multi_session.replay_next(
                &multi_trace,
                &stream_one_loc,
                &schedules,
                &card,
                &params,
                mode
            ),
            Err(ReplaySessionError::StreamMismatch {
                expected: 0,
                got: 1
            })
        ));
        assert_eq!(multi_session.completed_launches(), 0);
    }

    #[test]
    fn repeated_stream_sessions_are_deterministic_and_exhaustion_is_typed() {
        let tmp = tempfile::tempdir().unwrap();
        write_trace(
            tmp.path(),
            &[
                (1, 1, vec![load(1, BASE, 4)]),
                (2, 2, vec![load(2, BASE, 4)]),
            ],
        );
        let trace = TraceDir::open_direct(tmp.path()).unwrap();
        let locs: Vec<_> = trace.kernels().collect();
        let schedules = schedules(&[1, 2]);
        let card = sm86_test_card();
        let params = params();
        let mode = shared_mode(2, 256);

        let run_once = || {
            let mut session =
                StreamL2ReplaySession::new(&trace, &card, &params, mode, 0, 0).unwrap();
            let results: Vec<_> = locs
                .iter()
                .map(|loc| {
                    let result = full(
                        session
                            .replay_next(&trace, loc, &schedules, &card, &params, mode)
                            .unwrap(),
                    );
                    (
                        result.diagnostic_last_issue_cycles(),
                        *result.l2().unwrap(),
                        result.dram().copied(),
                    )
                })
                .collect();
            assert!(matches!(
                session.replay_next(&trace, &locs[1], &schedules, &card, &params, mode),
                Err(ReplaySessionError::LaunchSequenceComplete {
                    completed_launches: 2
                })
            ));
            (results, session.cumulative_l2_counters())
        };
        assert_eq!(run_once(), run_once());
    }
}

/// Resolve the two special Global routes implemented on the active sm_86 path.
/// The dynamic enum alone is not enough: the static sidecar must prove a
/// Global ATOM/RED family or LDGSTS role before the record enters replay.
fn require_sm86_special_role(
    schedule: &FunctionSchedule,
    arch: ArchId,
    pc: u32,
    route: ReplayMemoryRoute,
    class: pb::MemoryOpClass,
) -> Result<(), ReplayError> {
    if arch != ArchId::Sm86 {
        return Err(ReplayError::UnsupportedMemoryPath { arch, class });
    }
    let slot = schedule
        .by_pc
        .get(&u64::from(pc))
        .ok_or(ReplayError::UnsupportedMemoryPath { arch, class })?;
    let head = opcode_head(&slot.opcode_text);
    let valid = match (route, class) {
        (ReplayMemoryRoute::Global, pb::MemoryOpClass::Atomic) => match slot.mem_role {
            MemRole::BypassLoad(CacheOpToken::Atom) => matches!(head, "ATOM" | "ATOMG"),
            MemRole::Reduction => head == "RED",
            _ => false,
        },
        (ReplayMemoryRoute::Global, pb::MemoryOpClass::AsyncCopy) => {
            matches!(slot.mem_role, MemRole::BypassLoad(CacheOpToken::Ldgsts)) && head == "LDGSTS"
        }
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(ReplayError::UnsupportedMemoryPath { arch, class })
    }
}

fn require_ldgsts_l1_policy(
    schedule: &FunctionSchedule,
    arch: ArchId,
    pc: u32,
) -> Result<LdgstsL1Policy, ReplayError> {
    schedule
        .by_pc
        .get(&u64::from(pc))
        .and_then(|slot| slot.ldgsts_l1_policy)
        .ok_or(ReplayError::UnsupportedMemoryPath {
            arch,
            class: pb::MemoryOpClass::AsyncCopy,
        })
}

/// Return the schedule's memory role while rejecting a dynamic global record
/// whose static sidecar opcode is neither classified nor one of the deliberate
/// untagged load-family heads. A pc absent from the schedule stays neutral here
/// so the issue loop reports its more specific `PcNotInSchedule` error.
fn require_known_global_role(
    schedule: &FunctionSchedule,
    arch: ArchId,
    pc: u32,
) -> Result<MemRole, ReplayError> {
    let Some(slot) = schedule.by_pc.get(&u64::from(pc)) else {
        return Ok(MemRole::None);
    };
    if slot.mem_role == MemRole::None && !is_known_untagged_opcode(&slot.opcode_text) {
        return Err(ReplayError::UnknownGlobalOpcode {
            arch,
            pc,
            opcode: opcode_head(&slot.opcode_text).to_owned(),
        });
    }
    Ok(slot.mem_role)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LocalDirection {
    Load,
    Store,
}

/// Resolve Local traffic only from the joined LDL/STL sidecar role. A missing pc
/// remains neutral so the issue loop retains its dedicated `PcNotInSchedule`
/// diagnostic; every present non-LDL/STL role fails closed.
fn require_known_local_direction(
    schedule: &FunctionSchedule,
    arch: ArchId,
    pc: u32,
) -> Result<Option<LocalDirection>, ReplayError> {
    let Some(slot) = schedule.by_pc.get(&u64::from(pc)) else {
        return Ok(None);
    };
    match (opcode_head(&slot.opcode_text), slot.mem_role) {
        ("LDL", MemRole::EligibleLoad(CacheOpToken::Ldl)) => Ok(Some(LocalDirection::Load)),
        ("STL", MemRole::EligibleLoad(CacheOpToken::Stl)) => Ok(Some(LocalDirection::Store)),
        (opcode, _) => Err(ReplayError::UnknownLocalOpcode {
            arch,
            pc,
            opcode: opcode.to_owned(),
        }),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReplayMemoryRoute {
    Global,
    Local,
    Shared,
    Unsupported,
}

impl ReplayMemoryRoute {
    const fn canonical_space(self) -> pb::MemorySpace {
        match self {
            Self::Global => pb::MemorySpace::Global,
            Self::Local => pb::MemorySpace::Local,
            Self::Shared => pb::MemorySpace::Shared,
            Self::Unsupported => pb::MemorySpace::Unspecified,
        }
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Copy)]
struct ReplayMemoryPreflight<'a> {
    memory: PreflightedMemoryAccess<'a>,
    route: ReplayMemoryRoute,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StaticMemoryRoute {
    Modeled {
        route: ReplayMemoryRoute,
        class: pb::MemoryOpClass,
    },
    UnsupportedClass(pb::MemoryOpClass),
}

fn opcode_has_qualifier(opcode_text: &str, qualifier: &str) -> bool {
    opcode_text
        .split_whitespace()
        .next()
        .unwrap_or("")
        .split('.')
        .skip(1)
        .any(|part| part == qualifier)
}

/// Space-coded SASS heads whose storage class is part of the mnemonic rather
/// than inferred from the dynamic record. ATOM and RED are the legacy Global
/// forms; LDGSTS is routed by its traced Global source operand. Hopper-era REDG,
/// LDGMC, DSMEM, and shared-matrix families are architecture-gated before any
/// record can borrow a modeled route.
fn static_memory_route(
    opcode_text: &str,
    arch: ArchId,
) -> Result<Option<StaticMemoryRoute>, &'static str> {
    let head = opcode_head(opcode_text);
    if matches!(head, "LDS" | "STS" | "LDSM" | "STSM") && opcode_has_qualifier(opcode_text, "MMIO")
    {
        let family = match head {
            "LDS" => "LDS.MMIO",
            "STS" => "STS.MMIO",
            "LDSM" => "LDSM.MMIO",
            "STSM" => "STSM.MMIO",
            _ => unreachable!("match guard admits only shared-memory heads"),
        };
        return match arch {
            ArchId::Sm86 => Err(family),
            ArchId::Sm120 => Ok(Some(StaticMemoryRoute::UnsupportedClass(
                pb::MemoryOpClass::ClusterDsmem,
            ))),
        };
    }
    match head {
        "LDG" | "STG" => Ok(Some(StaticMemoryRoute::Modeled {
            route: ReplayMemoryRoute::Global,
            class: pb::MemoryOpClass::Global,
        })),
        "LDGSTS" => Ok(Some(StaticMemoryRoute::Modeled {
            route: ReplayMemoryRoute::Global,
            class: pb::MemoryOpClass::AsyncCopy,
        })),
        "ATOM" | "ATOMG" | "RED" => Ok(Some(StaticMemoryRoute::Modeled {
            route: ReplayMemoryRoute::Global,
            class: pb::MemoryOpClass::Atomic,
        })),
        "REDG" => match arch {
            ArchId::Sm86 => Err("REDG"),
            ArchId::Sm120 => Ok(Some(StaticMemoryRoute::Modeled {
                route: ReplayMemoryRoute::Global,
                class: pb::MemoryOpClass::Atomic,
            })),
        },
        // Hopper's reducing-load family has no implemented replay semantics on
        // either target. In particular it must never borrow Shared timing when
        // contaminated dynamic metadata claims a Shared route.
        "LDGMC" => Err("LDGMC"),
        "LDS" | "STS" | "LDSM" => Ok(Some(StaticMemoryRoute::Modeled {
            route: ReplayMemoryRoute::Shared,
            class: pb::MemoryOpClass::Shared,
        })),
        "ATOMS" => Ok(Some(StaticMemoryRoute::Modeled {
            route: ReplayMemoryRoute::Shared,
            class: pb::MemoryOpClass::Atomic,
        })),
        "STSM" => match arch {
            ArchId::Sm86 => Err("STSM"),
            ArchId::Sm120 => Ok(Some(StaticMemoryRoute::Modeled {
                route: ReplayMemoryRoute::Shared,
                class: pb::MemoryOpClass::Shared,
            })),
        },
        "STAS" | "REDAS" => match arch {
            ArchId::Sm86 => Err(if head == "STAS" { "STAS" } else { "REDAS" }),
            ArchId::Sm120 => Ok(Some(StaticMemoryRoute::UnsupportedClass(
                pb::MemoryOpClass::ClusterDsmem,
            ))),
        },
        "LDL" | "STL" => Ok(Some(StaticMemoryRoute::Modeled {
            route: ReplayMemoryRoute::Local,
            class: pb::MemoryOpClass::Local,
        })),
        "REDS" => Err("REDS"),
        _ if head.starts_with("ATOM") => Err("ATOM*"),
        _ if head.starts_with("RED") => Err("RED*"),
        _ => Ok(None),
    }
}

/// Reject disagreement between a known space-coded static SASS family and the
/// class/route resolved from the dynamic contract. This runs before allocation
/// identity and classification, so a contradictory record cannot borrow another
/// operation or storage class's timing and traffic semantics.
fn validate_static_memory_route(
    meta: MemoryAccessMeta,
    route: ReplayMemoryRoute,
    schedule: &FunctionSchedule,
    pc: u32,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<(), ReplayError> {
    let Some(slot) = schedule.by_pc.get(&u64::from(pc)) else {
        return Ok(());
    };
    let contextualize = |source| ReplayError::MemoryRoute {
        path: path.to_path_buf(),
        warp_id,
        record_ix,
        class: meta.op_class,
        source,
    };
    let local_role = matches!(
        slot.mem_role,
        MemRole::EligibleLoad(CacheOpToken::Ldl | CacheOpToken::Stl)
    );
    if local_role && route != ReplayMemoryRoute::Local {
        return Err(contextualize(MemoryRouteError::StaticMemoryRouteMismatch {
            expected: pb::MemorySpace::Local,
            got: route.canonical_space(),
        }));
    }
    let expected = static_memory_route(&slot.opcode_text, schedule.arch).map_err(|family| {
        contextualize(MemoryRouteError::StaticMemoryFamilyUnsupported {
            family,
            arch: schedule.arch,
        })
    })?;
    match expected {
        Some(StaticMemoryRoute::Modeled {
            route: expected, ..
        }) if expected != route => {
            Err(contextualize(MemoryRouteError::StaticMemoryRouteMismatch {
                expected: expected.canonical_space(),
                got: route.canonical_space(),
            }))
        }
        Some(StaticMemoryRoute::Modeled {
            class: expected, ..
        }) if expected != meta.op_class => {
            Err(contextualize(MemoryRouteError::StaticMemoryClassMismatch {
                expected,
                got: meta.op_class,
            }))
        }
        Some(StaticMemoryRoute::UnsupportedClass(expected)) if expected != meta.op_class => {
            Err(contextualize(MemoryRouteError::StaticMemoryClassMismatch {
                expected,
                got: meta.op_class,
            }))
        }
        _ => Ok(()),
    }
}

fn validate_replay_access_kind(
    meta: MemoryAccessMeta,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<(), ReplayError> {
    let Some(got) = meta.access_kind else {
        return Ok(()); // additive legacy field: absence remains admissible
    };
    let (valid, expected) = match meta.op_class {
        pb::MemoryOpClass::Global | pb::MemoryOpClass::Shared | pb::MemoryOpClass::Local => (
            matches!(
                got,
                pb::MemoryOpAccessKind::Load | pb::MemoryOpAccessKind::Store
            ),
            "LOAD or STORE",
        ),
        pb::MemoryOpClass::Atomic => (got == pb::MemoryOpAccessKind::Atomic, "ATOMIC"),
        pb::MemoryOpClass::AsyncCopy => (got == pb::MemoryOpAccessKind::AsyncCopy, "ASYNC_COPY"),
        // These routes fail as unsupported before replay. Do not invent a
        // direction contract for a class with no modeled path.
        pb::MemoryOpClass::Unspecified
        | pb::MemoryOpClass::Texture
        | pb::MemoryOpClass::Surface
        | pb::MemoryOpClass::Tma
        | pb::MemoryOpClass::ClusterDsmem
        | pb::MemoryOpClass::Tmem => return Ok(()),
    };
    if valid {
        return Ok(());
    }
    Err(ReplayError::MemoryRoute {
        path: path.to_path_buf(),
        warp_id,
        record_ix,
        class: meta.op_class,
        source: MemoryRouteError::AccessKindInvalid { got, expected },
    })
}

fn validate_static_access_kind(
    meta: MemoryAccessMeta,
    schedule: &FunctionSchedule,
    pc: u32,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<(), ReplayError> {
    let Some(got) = meta.access_kind else {
        return Ok(()); // additive legacy field: no dynamic direction to compare
    };
    // Keep a missing pc neutral so the issue loop preserves its specific
    // PcNotInSchedule diagnostic instead of manufacturing a direction claim.
    let Some(slot) = schedule.by_pc.get(&u64::from(pc)) else {
        return Ok(());
    };
    let head = opcode_head(&slot.opcode_text);
    let expected = if head.starts_with("ATOM") || head.starts_with("RED") {
        Some(pb::MemoryOpAccessKind::Atomic)
    } else {
        match head {
            "LDGSTS" => Some(pb::MemoryOpAccessKind::AsyncCopy),
            "STG" | "STL" | "STS" | "STSM" | "STAS" => Some(pb::MemoryOpAccessKind::Store),
            _ => match slot.mem_role {
                MemRole::EligibleLoad(CacheOpToken::Stl) => Some(pb::MemoryOpAccessKind::Store),
                MemRole::EligibleLoad(_) => Some(pb::MemoryOpAccessKind::Load),
                MemRole::BypassLoad(CacheOpToken::Atom) => Some(pb::MemoryOpAccessKind::Atomic),
                MemRole::BypassLoad(CacheOpToken::Ldgsts) => {
                    Some(pb::MemoryOpAccessKind::AsyncCopy)
                }
                MemRole::BypassLoad(_) => Some(pb::MemoryOpAccessKind::Load),
                MemRole::Store => Some(pb::MemoryOpAccessKind::Store),
                MemRole::Reduction => Some(pb::MemoryOpAccessKind::Atomic),
                MemRole::None if is_known_untagged_opcode(&slot.opcode_text) => {
                    Some(pb::MemoryOpAccessKind::Load)
                }
                MemRole::None => None,
            },
        }
    };
    let Some(expected) = expected else {
        return Ok(()); // the existing unknown-opcode gate owns this diagnostic
    };
    if got == expected {
        return Ok(());
    }
    Err(ReplayError::MemoryRoute {
        path: path.to_path_buf(),
        warp_id,
        record_ix,
        class: meta.op_class,
        source: MemoryRouteError::StaticAccessKindMismatch { got, expected },
    })
}

/// Resolve the replay route from the trace's validated payload-generation
/// contract. Address-carrying v1 records enforce the live producer's exact
/// operand shape; capability-empty legacy traces retain the pre-transition
/// ATOMIC/ASYNC_COPY fallback. GLOBAL retains its documented absent-space
/// fallback in both generations.
#[cfg(test)]
fn replay_memory_route(
    inst: &pb::Instruction,
    class: pb::MemoryOpClass,
    payload_contract: MemoryPayloadContract,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<ReplayMemoryRoute, ReplayError> {
    let contextualize = |source| ReplayError::MemoryRoute {
        path: path.to_path_buf(),
        warp_id,
        record_ix,
        class,
        source,
    };
    let decode_space = |operand: usize, address: &pb::Address| {
        address
            .address_space
            .map(|raw| {
                pb::MemorySpace::try_from(raw).map_err(|_| {
                    contextualize(MemoryRouteError::UnknownAddressSpace { operand, raw })
                })
            })
            .transpose()
    };

    let zero_effective_mask = matches!(
        (inst.active_mask, inst.predicate_mask),
        (Some(active), Some(predicate)) if active & predicate == 0
    );

    match class {
        pb::MemoryOpClass::Unspecified => Ok(ReplayMemoryRoute::Unsupported),
        pb::MemoryOpClass::Shared | pb::MemoryOpClass::Local => {
            if inst.addresses.is_empty() && !zero_effective_mask {
                return Err(contextualize(MemoryRouteError::SharedLocalOperandCount {
                    got: 0,
                }));
            }
            let expected_space = match class {
                pb::MemoryOpClass::Shared => pb::MemorySpace::Shared,
                pb::MemoryOpClass::Local => pb::MemorySpace::Local,
                _ => unreachable!("match arm admits only SHARED or LOCAL"),
            };
            for (operand, address) in inst.addresses.iter().enumerate() {
                if let Some(space) = decode_space(operand, address)? {
                    if space != expected_space {
                        return Err(contextualize(MemoryRouteError::SharedLocalSpaceInvalid {
                            operand,
                            space,
                        }));
                    }
                }
            }
            Ok(match class {
                pb::MemoryOpClass::Shared => ReplayMemoryRoute::Shared,
                pb::MemoryOpClass::Local => ReplayMemoryRoute::Local,
                _ => unreachable!("match arm admits only SHARED or LOCAL"),
            })
        }
        pb::MemoryOpClass::Global => {
            if inst.addresses.is_empty() && !zero_effective_mask {
                return Err(contextualize(MemoryRouteError::GlobalOperandCount {
                    got: 0,
                }));
            }
            for (operand, address) in inst.addresses.iter().enumerate() {
                if let Some(space) = decode_space(operand, address)? {
                    if !matches!(space, pb::MemorySpace::Global | pb::MemorySpace::Generic) {
                        return Err(contextualize(MemoryRouteError::GlobalSpaceInvalid {
                            operand,
                            space,
                        }));
                    }
                }
            }
            Ok(ReplayMemoryRoute::Global)
        }
        pb::MemoryOpClass::Atomic => {
            if payload_contract == MemoryPayloadContract::LegacyExclusion {
                let mut resolved = None;
                for (operand, address) in inst.addresses.iter().enumerate() {
                    let Some(space) = decode_space(operand, address)? else {
                        continue;
                    };
                    let candidate = match space {
                        pb::MemorySpace::Global | pb::MemorySpace::Generic => {
                            ReplayMemoryRoute::Global
                        }
                        pb::MemorySpace::Shared => ReplayMemoryRoute::Shared,
                        pb::MemorySpace::Local => ReplayMemoryRoute::Local,
                        pb::MemorySpace::Unspecified => {
                            return Err(contextualize(MemoryRouteError::AtomicSpaceUnspecified {
                                operand,
                            }));
                        }
                    };
                    match resolved {
                        Some(previous) if previous != candidate => {
                            return Err(contextualize(MemoryRouteError::AtomicSpacesConflict));
                        }
                        None => resolved = Some(candidate),
                        Some(_) => {}
                    }
                }
                return Ok(resolved.unwrap_or(ReplayMemoryRoute::Global));
            }
            // A no-payload zero-active legacy record retains the old global
            // fallback. Every active or address-carrying atomic must carry the
            // producer's single explicit-space memory operand.
            if inst.addresses.is_empty() && zero_effective_mask {
                return Ok(ReplayMemoryRoute::Global);
            }
            if inst.addresses.len() != 1 {
                return Err(contextualize(MemoryRouteError::AtomicOperandCount {
                    got: inst.addresses.len(),
                }));
            }
            let Some(space) = decode_space(0, &inst.addresses[0])? else {
                return Err(contextualize(MemoryRouteError::AtomicSpaceMissing {
                    operand: 0,
                }));
            };
            match space {
                pb::MemorySpace::Global | pb::MemorySpace::Generic => Ok(ReplayMemoryRoute::Global),
                pb::MemorySpace::Shared => Ok(ReplayMemoryRoute::Shared),
                pb::MemorySpace::Local => Ok(ReplayMemoryRoute::Local),
                pb::MemorySpace::Unspecified => {
                    Err(contextualize(MemoryRouteError::AtomicSpaceUnspecified {
                        operand: 0,
                    }))
                }
            }
        }
        pb::MemoryOpClass::AsyncCopy => {
            if payload_contract == MemoryPayloadContract::LegacyExclusion {
                // Legacy payload capability skips the new one-source shape, but
                // an explicit space still cannot contradict ASYNC_COPY's global
                // source. Missing space retains the historical global fallback.
                for (operand, address) in inst.addresses.iter().enumerate() {
                    if let Some(space) = decode_space(operand, address)? {
                        if !matches!(space, pb::MemorySpace::Global | pb::MemorySpace::Generic) {
                            return Err(contextualize(
                                MemoryRouteError::AsyncCopySourceSpaceInvalid { space },
                            ));
                        }
                    }
                }
                return Ok(ReplayMemoryRoute::Global);
            }
            if inst.addresses.len() != 1 {
                return Err(contextualize(MemoryRouteError::AsyncCopySourceCount {
                    got: inst.addresses.len(),
                }));
            }
            let Some(space) = decode_space(0, &inst.addresses[0])? else {
                return Err(contextualize(MemoryRouteError::AsyncCopySourceSpaceMissing));
            };
            if !matches!(space, pb::MemorySpace::Global | pb::MemorySpace::Generic) {
                return Err(contextualize(
                    MemoryRouteError::AsyncCopySourceSpaceInvalid { space },
                ));
            }
            Ok(ReplayMemoryRoute::Global)
        }
        pb::MemoryOpClass::Texture
        | pb::MemoryOpClass::Surface
        | pb::MemoryOpClass::Tma
        | pb::MemoryOpClass::ClusterDsmem
        | pb::MemoryOpClass::Tmem => Ok(ReplayMemoryRoute::Unsupported),
    }
}

#[cfg(test)]
fn preflight_replay_memory<'a>(
    inst: &'a pb::Instruction,
    schedule: &FunctionSchedule,
    payload_contract: MemoryPayloadContract,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<Option<ReplayMemoryPreflight<'a>>, ReplayError> {
    let record = || format!("{} warp={warp_id} inst={record_ix}", path.display());
    let memory = match preflight_memory_access(inst, record) {
        Ok(Some(memory)) => memory,
        Ok(None) => return Ok(None),
        Err(error) => {
            // The reader independently rejects class/space contradictions. Route
            // the same fully-structural-preflighted record through replay's own
            // shape-aware defense so operand-count diagnostics retain precedence
            // over a later conflicting operand.
            if let TraceError::MemoryOpClassAddressSpaceConflict { class, .. } = &error {
                if let Err(route_error) =
                    replay_memory_route(inst, *class, payload_contract, path, warp_id, record_ix)
                {
                    return Err(route_error);
                }
            }
            return Err(error.into());
        }
    };
    let meta = memory.meta();
    let route = replay_memory_route(
        memory.instruction(),
        meta.op_class,
        payload_contract,
        path,
        warp_id,
        record_ix,
    )?;
    validate_replay_access_kind(meta, path, warp_id, record_ix)?;
    // Static direction is meaningful only after the dynamic record identifies
    // this exact schedule. A foreign-function record retains the dedicated
    // MultiFunctionUnsupported diagnostic after structural/route validation.
    if inst.function_unique_id == schedule.function_unique_id {
        validate_static_memory_route(meta, route, schedule, inst.pc, path, warp_id, record_ix)?;
        validate_static_access_kind(meta, schedule, inst.pc, path, warp_id, record_ix)?;
    }
    Ok(Some(ReplayMemoryPreflight { memory, route }))
}

/// Flat-mode classification validates the complete memory wire contract without
/// expanding sectors, then applies the same producer-shape route gate as L1.
///
/// Flat mode preserves the joined static operation role with an empty sector
/// span. No L1 counters exist in this mode; the role remains load-bearing for
/// Store/ATOM/RED timing and unmodeled-tail diagnostics.
#[cfg(test)]
fn classify_flat(
    inst: &pb::Instruction,
    schedule: &FunctionSchedule,
    arch: ArchId,
    path: &std::path::Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<AccessKind, ReplayError> {
    classify_flat_with_contract(
        inst,
        schedule,
        arch,
        MemoryPayloadContract::AddressCarryingV1,
        path,
        warp_id,
        record_ix,
    )
}

#[cfg(test)]
fn classify_flat_with_contract(
    inst: &pb::Instruction,
    schedule: &FunctionSchedule,
    arch: ArchId,
    payload_contract: MemoryPayloadContract,
    path: &std::path::Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<AccessKind, ReplayError> {
    let preflight =
        preflight_replay_memory(inst, schedule, payload_contract, path, warp_id, record_ix)?;
    classify_flat_preflighted(preflight, schedule, arch, path, warp_id, record_ix)
}

/// Flat mode validates the same static operation role as hierarchy replay but
/// deliberately carries an empty sector span. This preserves Store/ATOM/RED
/// semantics for timing and capability diagnostics without decoding addresses.
fn flat_global_kind(role: MemRole) -> AccessKind {
    match role {
        MemRole::EligibleLoad(token) => AccessKind::GlobalEligible {
            off: 0,
            len: 0,
            token,
        },
        MemRole::BypassLoad(token) => AccessKind::GlobalBypass {
            off: 0,
            len: 0,
            token,
            l2_requests: 0,
        },
        MemRole::Store => AccessKind::GlobalStore { off: 0, len: 0 },
        MemRole::Reduction => AccessKind::GlobalReduction {
            off: 0,
            len: 0,
            l2_requests: 0,
        },
        MemRole::None => AccessKind::GlobalUntagged { off: 0, len: 0 },
    }
}

#[derive(Clone, Copy)]
struct CompactReplayMemory<'a> {
    memory: &'a ValidatedMemoryInstruction,
    route: ReplayMemoryRoute,
    pc: u32,
}

fn compact_shared_kind(memory: &ValidatedMemoryInstruction) -> AccessKind {
    AccessKind::Shared {
        wavefronts: u64::from(memory.shared_bank_wavefronts_32x4()),
    }
}

#[cfg(test)]
fn preflighted_shared_kind(
    memory: PreflightedMemoryAccess<'_>,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<AccessKind, ReplayError> {
    let mut word_scratch = Vec::new();
    let wavefronts =
        project_shared_bank_wavefronts_32x4(memory, &mut word_scratch).map_err(|source| {
            ReplayError::SharedBankProjection {
                path: path.to_path_buf(),
                warp_id,
                record_ix,
                source,
            }
        })?;
    Ok(AccessKind::Shared {
        wavefronts: u64::from(wavefronts),
    })
}

fn compact_memory_route_error(error: &ValidatedMemoryRouteError) -> MemoryRouteError {
    match *error {
        ValidatedMemoryRouteError::GlobalOperandCount { got } => {
            MemoryRouteError::GlobalOperandCount { got }
        }
        ValidatedMemoryRouteError::GlobalSpaceInvalid { operand, space } => {
            MemoryRouteError::GlobalSpaceInvalid { operand, space }
        }
        ValidatedMemoryRouteError::SharedLocalOperandCount { got } => {
            MemoryRouteError::SharedLocalOperandCount { got }
        }
        ValidatedMemoryRouteError::SharedLocalSpaceInvalid { operand, space } => {
            MemoryRouteError::SharedLocalSpaceInvalid { operand, space }
        }
        ValidatedMemoryRouteError::AtomicOperandCount { got } => {
            MemoryRouteError::AtomicOperandCount { got }
        }
        ValidatedMemoryRouteError::AtomicSpaceMissing { operand } => {
            MemoryRouteError::AtomicSpaceMissing { operand }
        }
        ValidatedMemoryRouteError::AtomicSpaceUnspecified { operand } => {
            MemoryRouteError::AtomicSpaceUnspecified { operand }
        }
        ValidatedMemoryRouteError::AtomicSpacesConflict => MemoryRouteError::AtomicSpacesConflict,
        ValidatedMemoryRouteError::AsyncCopySourceCount { got } => {
            MemoryRouteError::AsyncCopySourceCount { got }
        }
        ValidatedMemoryRouteError::AsyncCopySourceSpaceMissing => {
            MemoryRouteError::AsyncCopySourceSpaceMissing
        }
        ValidatedMemoryRouteError::AsyncCopySourceSpaceInvalid { space } => {
            MemoryRouteError::AsyncCopySourceSpaceInvalid { space }
        }
        ValidatedMemoryRouteError::AsyncCopyOperandCount { got, expected } => {
            MemoryRouteError::AsyncCopyOperandCount { got, expected }
        }
        ValidatedMemoryRouteError::AsyncCopyDestinationSpaceMissing => {
            MemoryRouteError::AsyncCopyDestinationSpaceMissing
        }
        ValidatedMemoryRouteError::AsyncCopyDestinationSpaceInvalid { space } => {
            MemoryRouteError::AsyncCopyDestinationSpaceInvalid { space }
        }
    }
}

fn preflight_compact_memory<'a>(
    memory: Option<&'a ValidatedMemoryInstruction>,
    warp: &'a ValidatedWarp,
    schedule: &FunctionSchedule,
    pc: u32,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<Option<CompactReplayMemory<'a>>, ReplayError> {
    let Some(memory) = memory else {
        return Ok(None);
    };
    let meta = memory.replay_meta();
    let route = match warp.memory_route(memory) {
        Ok(ValidatedMemoryRoute::Global) => ReplayMemoryRoute::Global,
        Ok(ValidatedMemoryRoute::Local) => ReplayMemoryRoute::Local,
        Ok(ValidatedMemoryRoute::Shared) => ReplayMemoryRoute::Shared,
        Ok(ValidatedMemoryRoute::Unsupported) => ReplayMemoryRoute::Unsupported,
        Err(error) => {
            return Err(ReplayError::MemoryRoute {
                path: path.to_path_buf(),
                warp_id,
                record_ix,
                class: meta.op_class,
                source: compact_memory_route_error(error),
            });
        }
    };
    validate_replay_access_kind(meta, path, warp_id, record_ix)?;
    validate_static_memory_route(meta, route, schedule, pc, path, warp_id, record_ix)?;
    validate_static_access_kind(meta, schedule, pc, path, warp_id, record_ix)?;
    Ok(Some(CompactReplayMemory { memory, route, pc }))
}

fn classify_compact_flat(
    preflight: Option<CompactReplayMemory<'_>>,
    warp: &ValidatedWarp,
    schedule: &FunctionSchedule,
    arch: ArchId,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<AccessKind, ReplayError> {
    let Some(preflight) = preflight else {
        return Ok(AccessKind::NonMemory);
    };
    if !preflight.memory.has_effective_lanes() {
        return Ok(AccessKind::PredicatedOffMemory);
    }
    let meta = preflight.memory.replay_meta();
    match (preflight.route, meta.op_class) {
        (ReplayMemoryRoute::Shared, _) => Ok(compact_shared_kind(preflight.memory)),
        (ReplayMemoryRoute::Local, pb::MemoryOpClass::Local) => Ok(
            match require_known_local_direction(schedule, arch, preflight.pc)? {
                Some(LocalDirection::Load) => AccessKind::LocalLoad { off: 0, len: 0 },
                Some(LocalDirection::Store) => AccessKind::LocalStore { off: 0, len: 0 },
                None => AccessKind::NonMemory,
            },
        ),
        (ReplayMemoryRoute::Local, class) => {
            Err(ReplayError::UnsupportedMemoryPath { arch, class })
        }
        (ReplayMemoryRoute::Unsupported, class) => {
            Err(ReplayError::UnsupportedMemoryPath { arch, class })
        }
        (ReplayMemoryRoute::Global, pb::MemoryOpClass::Global) => {
            let role = require_known_global_role(schedule, arch, preflight.pc)?;
            Ok(flat_global_kind(role))
        }
        (ReplayMemoryRoute::Global, pb::MemoryOpClass::Atomic) => {
            require_sm86_special_role(
                schedule,
                arch,
                preflight.pc,
                preflight.route,
                pb::MemoryOpClass::Atomic,
            )?;
            let role = require_known_global_role(schedule, arch, preflight.pc)?;
            Ok(flat_global_kind(role))
        }
        (ReplayMemoryRoute::Global, pb::MemoryOpClass::AsyncCopy) => {
            require_sm86_special_role(
                schedule,
                arch,
                preflight.pc,
                preflight.route,
                pb::MemoryOpClass::AsyncCopy,
            )?;
            require_known_global_role(schedule, arch, preflight.pc)?;
            let l1_policy = require_ldgsts_l1_policy(schedule, arch, preflight.pc)?;
            let source_active = match warp.sector_plan(preflight.memory) {
                ValidatedSectorPlan::Sectors(sectors) => !sectors.is_empty(),
                ValidatedSectorPlan::Error { error, .. } => {
                    return Err(deferred_sector_error(error, path, warp_id, record_ix));
                }
            };
            let payload_bytes = warp.effective_data_bytes(preflight.memory);
            if source_active != (payload_bytes != 0) {
                return Err(ReplayError::AsyncCopyPayloadMismatch {
                    path: path.to_path_buf(),
                    warp_id,
                    record_ix,
                    source_active,
                    sectors: usize::from(source_active),
                    payload_bytes,
                });
            }
            let completion_work = project_async_copy_completion_work(
                arch,
                l1_policy,
                payload_bytes,
                warp.ldgsts_destination_topology_128b_32x4(preflight.memory),
                path,
                warp_id,
                record_ix,
            )?;
            Ok(AccessKind::GlobalAsyncCopy {
                off: 0,
                len: 0,
                completion_work,
                l1_policy,
            })
        }
        (ReplayMemoryRoute::Global, class) => {
            Err(ReplayError::UnsupportedMemoryPath { arch, class })
        }
    }
}

fn deferred_sector_error(
    error: &ValidatedSectorError,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> ReplayError {
    if let ValidatedSectorError::SectorCountOverflow { sectors } = *error {
        return ReplayError::SectorCountOverflow {
            path: path.to_path_buf(),
            warp_id,
            record_ix,
            sectors,
        };
    }
    let source = match *error {
        ValidatedSectorError::DataWidthInvalid { data_width_bytes } => {
            SectorExpansionError::DataWidthInvalid { data_width_bytes }
        }
        ValidatedSectorError::AddressOverflow {
            lane,
            base,
            data_width_bytes,
        } => SectorExpansionError::AddressOverflow {
            lane,
            base,
            data_width_bytes,
        },
        ValidatedSectorError::UniqueSectorCountOutOfRange { sectors } => {
            SectorExpansionError::UniqueSectorCountOutOfRange { sectors }
        }
        ValidatedSectorError::CardinalityMismatch { expected, produced } => {
            SectorExpansionError::CardinalityMismatch { expected, produced }
        }
        ValidatedSectorError::CapacityUnavailable { buffer, additional } => {
            SectorExpansionError::CapacityUnavailable { buffer, additional }
        }
        ValidatedSectorError::SectorCountOverflow { .. } => unreachable!("handled above"),
    };
    ReplayError::SectorExpansion {
        path: path.to_path_buf(),
        warp_id,
        record_ix,
        source,
    }
}

fn classify_compact_with_l1(
    preflight: Option<CompactReplayMemory<'_>>,
    warp: &ValidatedWarp,
    schedule: &FunctionSchedule,
    arch: ArchId,
    sectors: &mut Vec<i64>,
    async_copy_cohorts: &mut AsyncCopyCohortReplay,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<AccessKind, ReplayError> {
    let Some(preflight) = preflight else {
        return Ok(AccessKind::NonMemory);
    };
    if !preflight.memory.has_effective_lanes() {
        return Ok(AccessKind::PredicatedOffMemory);
    }
    let meta = preflight.memory.replay_meta();
    let sector_plan = warp.sector_plan(preflight.memory);
    if let ValidatedSectorPlan::Error {
        phase: ValidatedSectorErrorPhase::BeforeStaticRole,
        error,
    } = sector_plan
    {
        return Err(deferred_sector_error(error, path, warp_id, record_ix));
    }

    enum StagedHierarchyRole {
        Global(MemRole),
        Local(LocalDirection),
    }
    let staged_role = match (preflight.route, meta.op_class) {
        (ReplayMemoryRoute::Shared, _) => {
            return Ok(compact_shared_kind(preflight.memory));
        }
        (ReplayMemoryRoute::Unsupported, class) => {
            return Err(ReplayError::UnsupportedMemoryPath { arch, class });
        }
        (ReplayMemoryRoute::Local, pb::MemoryOpClass::Local) => {
            let Some(direction) = require_known_local_direction(schedule, arch, preflight.pc)?
            else {
                return Ok(AccessKind::NonMemory);
            };
            StagedHierarchyRole::Local(direction)
        }
        (ReplayMemoryRoute::Local, class) => {
            return Err(ReplayError::UnsupportedMemoryPath { arch, class });
        }
        (ReplayMemoryRoute::Global, pb::MemoryOpClass::Global) => {
            StagedHierarchyRole::Global(require_known_global_role(schedule, arch, preflight.pc)?)
        }
        (
            ReplayMemoryRoute::Global,
            class @ (pb::MemoryOpClass::Atomic | pb::MemoryOpClass::AsyncCopy),
        ) => {
            require_sm86_special_role(schedule, arch, preflight.pc, preflight.route, class)?;
            StagedHierarchyRole::Global(require_known_global_role(schedule, arch, preflight.pc)?)
        }
        (ReplayMemoryRoute::Global, class) => {
            return Err(ReplayError::UnsupportedMemoryPath { arch, class });
        }
    };
    let staged = match sector_plan {
        ValidatedSectorPlan::Sectors(staged) => staged,
        ValidatedSectorPlan::Error {
            phase: ValidatedSectorErrorPhase::AfterStaticRole,
            error,
        } => return Err(deferred_sector_error(error, path, warp_id, record_ix)),
        ValidatedSectorPlan::Error {
            phase: ValidatedSectorErrorPhase::BeforeStaticRole,
            ..
        } => unreachable!("handled above"),
    };
    match staged_role {
        StagedHierarchyRole::Global(_role) if meta.op_class == pb::MemoryOpClass::AsyncCopy => {
            let l1_policy = require_ldgsts_l1_policy(schedule, arch, preflight.pc)?;
            append_staged_async_copy_sectors(
                arch,
                l1_policy,
                warp.effective_data_bytes(preflight.memory),
                warp.ldgsts_destination_topology_128b_32x4(preflight.memory),
                warp.ldgsts_source_cohorts_128b_32x4(preflight.memory),
                warp.ldgsts_source_sector_cohort_masks_128b_32x4(preflight.memory),
                sectors,
                async_copy_cohorts,
                staged,
                path,
                warp_id,
                record_ix,
            )
        }
        StagedHierarchyRole::Global(role) => {
            append_staged_global_sectors(role, sectors, staged, path, warp_id, record_ix)
        }
        StagedHierarchyRole::Local(direction) => {
            append_staged_local_sectors(direction, sectors, staged, path, warp_id, record_ix)
        }
    }
}

#[cfg(test)]
fn classify_flat_preflighted(
    preflight: Option<ReplayMemoryPreflight<'_>>,
    schedule: &FunctionSchedule,
    arch: ArchId,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<AccessKind, ReplayError> {
    let Some(preflight) = preflight else {
        return Ok(AccessKind::NonMemory);
    };
    let inst = preflight.memory.instruction();
    let meta = preflight.memory.meta();
    if meta.and_mask == 0 {
        return Ok(AccessKind::PredicatedOffMemory);
    }
    match (preflight.route, meta.op_class) {
        (ReplayMemoryRoute::Shared, _) => {
            preflighted_shared_kind(preflight.memory, path, warp_id, record_ix)
        }
        (ReplayMemoryRoute::Local, pb::MemoryOpClass::Local) => Ok(
            match require_known_local_direction(schedule, arch, inst.pc)? {
                Some(LocalDirection::Load) => AccessKind::LocalLoad { off: 0, len: 0 },
                Some(LocalDirection::Store) => AccessKind::LocalStore { off: 0, len: 0 },
                None => AccessKind::NonMemory,
            },
        ),
        (ReplayMemoryRoute::Local, class) => {
            Err(ReplayError::UnsupportedMemoryPath { arch, class })
        }
        (ReplayMemoryRoute::Unsupported, class) => {
            Err(ReplayError::UnsupportedMemoryPath { arch, class })
        }
        (ReplayMemoryRoute::Global, pb::MemoryOpClass::Global) => {
            let role = require_known_global_role(schedule, arch, inst.pc)?;
            Ok(flat_global_kind(role))
        }
        (ReplayMemoryRoute::Global, pb::MemoryOpClass::Atomic) => {
            require_sm86_special_role(
                schedule,
                arch,
                inst.pc,
                preflight.route,
                pb::MemoryOpClass::Atomic,
            )?;
            let role = require_known_global_role(schedule, arch, inst.pc)?;
            Ok(flat_global_kind(role))
        }
        (ReplayMemoryRoute::Global, pb::MemoryOpClass::AsyncCopy) => {
            require_sm86_special_role(
                schedule,
                arch,
                inst.pc,
                preflight.route,
                pb::MemoryOpClass::AsyncCopy,
            )?;
            require_known_global_role(schedule, arch, inst.pc)?;
            let l1_policy = require_ldgsts_l1_policy(schedule, arch, inst.pc)?;
            let source_active = meta.and_mask != 0 && !inst.addresses.is_empty();
            let payload_bytes =
                project_effective_data_bytes(preflight.memory).map_err(|source| {
                    ReplayError::EffectiveDataBytes {
                        path: path.to_path_buf(),
                        warp_id,
                        record_ix,
                        source,
                    }
                })?;
            if source_active != (payload_bytes != 0) {
                return Err(ReplayError::AsyncCopyPayloadMismatch {
                    path: path.to_path_buf(),
                    warp_id,
                    record_ix,
                    source_active,
                    sectors: usize::from(source_active),
                    payload_bytes,
                });
            }
            let completion_work = project_async_copy_completion_work(
                arch,
                l1_policy,
                payload_bytes,
                None,
                path,
                warp_id,
                record_ix,
            )?;
            Ok(AccessKind::GlobalAsyncCopy {
                off: 0,
                len: 0,
                completion_work,
                l1_policy,
            })
        }
        (ReplayMemoryRoute::Global, class) => {
            Err(ReplayError::UnsupportedMemoryPath { arch, class })
        }
    }
}

fn checked_staged_global_kind(
    role: MemRole,
    start: usize,
    count: usize,
    l2_requests: u16,
) -> Option<AccessKind> {
    let len = u16::try_from(count).ok()?;
    start.checked_add(count)?;
    Some(match role {
        MemRole::EligibleLoad(token) => AccessKind::GlobalEligible {
            off: start,
            len,
            token,
        },
        MemRole::BypassLoad(token) => AccessKind::GlobalBypass {
            off: start,
            len,
            token,
            l2_requests: if token == CacheOpToken::Atom {
                l2_requests
            } else {
                0
            },
        },
        MemRole::Store => AccessKind::GlobalStore { off: start, len },
        MemRole::Reduction => AccessKind::GlobalReduction {
            off: start,
            len,
            l2_requests,
        },
        MemRole::None => AccessKind::GlobalUntagged { off: start, len },
    })
}

fn project_async_copy_completion_work(
    arch: ArchId,
    l1_policy: LdgstsL1Policy,
    payload_bytes: u64,
    destination_topology: Option<LdgstsDestinationTopology>,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<u32, ReplayError> {
    let throughput = arch
        .ldgsts_completion_throughput()
        .require("ldgsts_completion_throughput")
        .map_err(IssueError::from)?;
    let cache_all_rate = throughput.cache_all_payload_bytes_per_cycle;
    let bypass_rate = throughput.bypass_payload_bytes_per_cycle;
    let units_per_cycle = cache_all_rate.max(bypass_rate);
    if cache_all_rate == 0
        || bypass_rate == 0
        || units_per_cycle % cache_all_rate != 0
        || units_per_cycle % bypass_rate != 0
    {
        return Err(IssueError::InvalidLdgstsCompletionThroughput {
            arch,
            cache_all_bytes_per_cycle: cache_all_rate,
            bypass_bytes_per_cycle: bypass_rate,
        }
        .into());
    }
    let route_rate = match l1_policy {
        LdgstsL1Policy::CacheAtAllLevels => cache_all_rate,
        LdgstsL1Policy::BypassL1 => bypass_rate,
    };
    let units_per_byte = units_per_cycle / route_rate;
    let payload_work = payload_bytes.checked_mul(units_per_byte).ok_or(
        IssueError::TimeMultiplicationOverflow {
            operation: "LDGSTS payload-fallback completion work",
            lhs: payload_bytes,
            rhs: units_per_byte,
        },
    )?;
    if l1_policy == LdgstsL1Policy::BypassL1
        && let Some(topology) = destination_topology
    {
        let payload_work_compact = u32::try_from(payload_work).map_err(|_| {
            ReplayError::AsyncCopyCompletionWorkOutOfRange {
                path: path.to_path_buf(),
                warp_id,
                record_ix,
                completion_work: payload_work,
                max_completion_work: ASYNC_COPY_COMPOSITE_PAYLOAD_WORK_MAX,
            }
        })?;
        // A topology-only copy has executing destination lanes but no source
        // cohort. The compact observation token uses one reserved cohort slot
        // to keep the token non-inactive; the detailed sidecar remains empty.
        let source_cohorts = topology.source_cohorts.max(1);
        return encode_async_copy_payload_and_topology(
            payload_work_compact,
            source_cohorts,
            topology.destination_line_fanout_total,
            topology.cache_all_bank_wavefronts_max,
            topology.bypass_wavefronts_max,
        )
        .ok_or(ReplayError::AsyncCopyCompletionWorkOutOfRange {
            path: path.to_path_buf(),
            warp_id,
            record_ix,
            completion_work: payload_work
                .max(u64::from(topology.destination_line_fanout_total))
                .max(u64::from(topology.cache_all_bank_wavefronts_max))
                .max(u64::from(topology.bypass_wavefronts_max)),
            max_completion_work: ASYNC_COPY_COMPOSITE_PAYLOAD_WORK_MAX,
        });
    }
    let exact_destination = if l1_policy == LdgstsL1Policy::CacheAtAllLevels {
        match destination_topology {
            Some(topology) => {
                let wavefronts = topology.cache_all_bank_wavefronts_total;
                let timing = arch
                    .ldgsts_destination_wavefront_timing()
                    .require("ldgsts_destination_wavefront_timing")
                    .map_err(IssueError::from)?;
                match timing.projection {
                    LdgstsDestinationWavefrontProjection::SourceCohorts128BytesBanks32Words4Bytes => {}
                }
                if timing.cycles_per_wavefront == 0 {
                    return Err(IssueError::InvalidLdgstsDestinationWavefrontTiming {
                        arch,
                        cycles_per_wavefront: timing.cycles_per_wavefront,
                    }
                    .into());
                }
                let destination_cycles = timing
                    .cycles_per_wavefront
                    .checked_mul(u64::from(wavefronts))
                    .ok_or(IssueError::TimeMultiplicationOverflow {
                        operation: "LDGSTS destination-wavefront completion cycles",
                        lhs: timing.cycles_per_wavefront,
                        rhs: u64::from(wavefronts),
                    })?;
                let destination_work = destination_cycles.checked_mul(units_per_cycle).ok_or(
                    IssueError::TimeMultiplicationOverflow {
                        operation: "LDGSTS destination-wavefront completion work",
                        lhs: destination_cycles,
                        rhs: units_per_cycle,
                    },
                )?;
                Some((wavefronts, destination_work))
            }
            None => None,
        }
    } else {
        // Focused bypass probes show a different source/destination coupling.
        // The compact token above retains its trace-derived topology for cold
        // diagnostics, while production timing continues to use payload work
        // until a bypass-specific per-copy mechanism is characterized.
        None
    };
    if payload_work != 0
        && let Some((_wavefronts, destination_work)) = exact_destination
        && destination_work >= payload_work
    {
        let compact = u32::try_from(destination_work).map_err(|_| {
            ReplayError::AsyncCopyCompletionWorkOutOfRange {
                path: path.to_path_buf(),
                warp_id,
                record_ix,
                completion_work: destination_work,
                max_completion_work: ASYNC_COPY_COMPLETION_VALUE_MASK,
            }
        })?;
        return encode_async_copy_destination_work(compact).ok_or(
            ReplayError::AsyncCopyCompletionWorkOutOfRange {
                path: path.to_path_buf(),
                warp_id,
                record_ix,
                completion_work: destination_work,
                max_completion_work: ASYNC_COPY_COMPLETION_VALUE_MASK,
            },
        );
    }
    let completion_work = exact_destination.map_or(payload_work, |(_, destination_work)| {
        payload_work.max(destination_work)
    });
    let compact = u32::try_from(completion_work).map_err(|_| {
        ReplayError::AsyncCopyCompletionWorkOutOfRange {
            path: path.to_path_buf(),
            warp_id,
            record_ix,
            completion_work,
            max_completion_work: ASYNC_COPY_COMPLETION_VALUE_MASK,
        }
    })?;
    encode_async_copy_payload_work(compact).ok_or(ReplayError::AsyncCopyCompletionWorkOutOfRange {
        path: path.to_path_buf(),
        warp_id,
        record_ix,
        completion_work,
        max_completion_work: ASYNC_COPY_COMPLETION_VALUE_MASK,
    })
}

fn checked_staged_async_copy_kind(
    l1_policy: LdgstsL1Policy,
    start: usize,
    count: usize,
    completion_work: u32,
) -> Option<AccessKind> {
    let len = u16::try_from(count).ok()?;
    start.checked_add(count)?;
    Some(AccessKind::GlobalAsyncCopy {
        off: start,
        len,
        completion_work,
        l1_policy,
    })
}

fn checked_staged_local_kind(
    direction: LocalDirection,
    start: usize,
    count: usize,
) -> Option<AccessKind> {
    let len = u16::try_from(count).ok()?;
    start.checked_add(count)?;
    Some(match direction {
        LocalDirection::Load => AccessKind::LocalLoad { off: start, len },
        LocalDirection::Store => AccessKind::LocalStore { off: start, len },
    })
}

fn append_staged_sectors(
    kind: Option<AccessKind>,
    sectors: &mut Vec<i64>,
    sector_scratch: &[i64],
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<AccessKind, ReplayError> {
    let count = sector_scratch.len();
    let overflow = |sectors: usize| ReplayError::SectorCountOverflow {
        path: path.to_path_buf(),
        warp_id,
        record_ix,
        sectors,
    };
    // `AccessKind::off` spans the native `usize` arena domain. Only one
    // record's compact length narrows to `u16`.
    let kind = kind.ok_or_else(|| overflow(count))?;
    sectors
        .try_reserve(count)
        .map_err(|_| ReplayError::SectorExpansion {
            path: path.to_path_buf(),
            warp_id,
            record_ix,
            source: SectorExpansionError::CapacityUnavailable {
                buffer: "warp sector arena",
                additional: count,
            },
        })?;
    sectors.extend_from_slice(sector_scratch);
    Ok(kind)
}

fn append_staged_global_sectors(
    role: MemRole,
    sectors: &mut Vec<i64>,
    sector_scratch: &[i64],
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<AccessKind, ReplayError> {
    let start = sectors.len();
    let l2_requests = if matches!(
        role,
        MemRole::BypassLoad(CacheOpToken::Atom) | MemRole::Reduction
    ) {
        atomic_l2_request_count_128b(sector_scratch)
    } else {
        0
    };
    append_staged_sectors(
        checked_staged_global_kind(role, start, sector_scratch.len(), l2_requests),
        sectors,
        sector_scratch,
        path,
        warp_id,
        record_ix,
    )
}

/// Reconstruct exact L2 atomic request topology from the alias-expanded 32-byte
/// sector stream. Each request carries sectors from one 128-byte line; repeated
/// same-word lanes appear as repeated sector ids, so the line requires the
/// maximum sector multiplicity and distinct lines add.
fn atomic_l2_request_count_128b(sectors: &[i64]) -> u16 {
    let mut by_line: BTreeMap<i64, BTreeMap<i64, u16>> = BTreeMap::new();
    for &sector in sectors {
        let count = by_line
            .entry(sector.div_euclid(4))
            .or_default()
            .entry(sector)
            .or_default();
        *count = count
            .checked_add(1)
            .expect("one warp's bounded sector span cannot overflow u16");
    }
    let requests = by_line.values().fold(0_u32, |total, line| {
        total + u32::from(line.values().copied().max().unwrap_or(0))
    });
    u16::try_from(requests).expect("atomic L2 requests cannot exceed the bounded sector span")
}

fn append_staged_async_copy_sectors(
    arch: ArchId,
    l1_policy: LdgstsL1Policy,
    payload_bytes: u64,
    destination_topology: Option<LdgstsDestinationTopology>,
    source_cohorts: Option<&[LdgstsSourceCohort]>,
    sector_cohort_masks: Option<&[u32]>,
    sectors: &mut Vec<i64>,
    cohort_replay: &mut AsyncCopyCohortReplay,
    sector_scratch: &[i64],
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<AccessKind, ReplayError> {
    let start = sectors.len();
    let count = sector_scratch.len();
    let source_active = count != 0;
    if source_active != (payload_bytes != 0) {
        return Err(ReplayError::AsyncCopyPayloadMismatch {
            path: path.to_path_buf(),
            warp_id,
            record_ix,
            source_active,
            sectors: count,
            payload_bytes,
        });
    }
    let completion_work = project_async_copy_completion_work(
        arch,
        l1_policy,
        payload_bytes,
        destination_topology,
        path,
        warp_id,
        record_ix,
    )?;
    let overflow = |sectors: usize| ReplayError::SectorCountOverflow {
        path: path.to_path_buf(),
        warp_id,
        record_ix,
        sectors,
    };
    let kind = checked_staged_async_copy_kind(l1_policy, start, count, completion_work)
        .ok_or_else(|| overflow(count))?;

    let cohort_projection = match (source_cohorts, sector_cohort_masks) {
        (None, None) => None,
        (Some(cohorts), Some(masks))
            if !source_active && cohorts.is_empty() && masks.is_empty() =>
        {
            None
        }
        (Some(cohorts), Some(masks)) => {
            if masks.len() != count {
                return Err(ReplayError::AsyncCopyCohortProjectionMismatch {
                    path: path.to_path_buf(),
                    warp_id,
                    record_ix,
                    field: "source-sector membership count",
                    expected: count,
                    got: masks.len(),
                });
            }
            if cohorts.is_empty() || cohorts.len() > u32::BITS as usize {
                return Err(ReplayError::AsyncCopyCohortProjectionMismatch {
                    path: path.to_path_buf(),
                    warp_id,
                    record_ix,
                    field: "cohort count in u32 membership domain",
                    expected: usize::from(source_active),
                    got: cohorts.len(),
                });
            }
            Some((cohorts, masks))
        }
        (Some(cohorts), None) => {
            return Err(ReplayError::AsyncCopyCohortProjectionMismatch {
                path: path.to_path_buf(),
                warp_id,
                record_ix,
                field: "paired source-sector cohort masks",
                expected: count,
                got: cohorts.len(),
            });
        }
        (None, Some(masks)) => {
            return Err(ReplayError::AsyncCopyCohortProjectionMismatch {
                path: path.to_path_buf(),
                warp_id,
                record_ix,
                field: "paired source cohort descriptors",
                expected: 1,
                got: masks.len(),
            });
        }
    };

    sectors
        .try_reserve(count)
        .map_err(|_| ReplayError::SectorExpansion {
            path: path.to_path_buf(),
            warp_id,
            record_ix,
            source: SectorExpansionError::CapacityUnavailable {
                buffer: "warp sector arena",
                additional: count,
            },
        })?;
    if let Some((cohorts, masks)) = cohort_projection {
        for (buffer, additional, result) in [
            (
                "span",
                1,
                cohort_replay.spans.try_reserve(1).map_err(|_| ()),
            ),
            (
                "descriptor",
                cohorts.len(),
                cohort_replay
                    .cohorts
                    .try_reserve(cohorts.len())
                    .map_err(|_| ()),
            ),
            (
                "membership-mask",
                masks.len(),
                cohort_replay
                    .sector_membership_masks
                    .try_reserve(masks.len())
                    .map_err(|_| ()),
            ),
        ] {
            result.map_err(|_| ReplayError::AsyncCopyCohortCapacityUnavailable {
                path: path.to_path_buf(),
                warp_id,
                record_ix,
                buffer,
                additional,
            })?;
        }
    }

    sectors.extend_from_slice(sector_scratch);
    if let Some((cohorts, masks)) = cohort_projection {
        let cohort_off = cohort_replay.cohorts.len();
        let membership_off = cohort_replay.sector_membership_masks.len();
        cohort_replay
            .cohorts
            .extend(cohorts.iter().map(|cohort| AsyncCopyCohortDescriptor {
                payload_bytes: cohort.payload_bytes,
                destination_lines: cohort.destination_lines,
                bank_wavefronts: cohort.bank_wavefronts,
                bypass_wavefronts: cohort.bypass_wavefronts,
            }));
        cohort_replay
            .sector_membership_masks
            .extend_from_slice(masks);
        cohort_replay.spans.push(AsyncCopyCohortSpan {
            instruction: record_ix,
            sector_off: start,
            sector_len: u16::try_from(count).expect("checked async-copy replay kind length"),
            cohort_off,
            cohort_len: u8::try_from(cohorts.len())
                .expect("u32 membership domain fits the u8 cohort span"),
            membership_off,
        });
    }
    Ok(kind)
}

fn append_staged_local_sectors(
    direction: LocalDirection,
    sectors: &mut Vec<i64>,
    sector_scratch: &[i64],
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<AccessKind, ReplayError> {
    let start = sectors.len();
    append_staged_sectors(
        checked_staged_local_kind(direction, start, sector_scratch.len()),
        sectors,
        sector_scratch,
        path,
        warp_id,
        record_ix,
    )
}

/// Full classification in L1 mode: wire-contract decode (streaming form),
/// per-instruction unique-sector expansion in reusable private scratch, and
/// the join-time [`MemRole`] split. The warp arena is extended only after
/// decoding, expansion, routing, and index narrowing all succeed.
#[cfg(test)]
fn classify_with_l1(
    inst: &pb::Instruction,
    schedule: &FunctionSchedule,
    arch: ArchId,
    sectors: &mut Vec<i64>,
    sector_scratch: &mut Vec<i64>,
    path: &std::path::Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<AccessKind, ReplayError> {
    classify_with_l1_with_contract(
        inst,
        schedule,
        arch,
        MemoryPayloadContract::AddressCarryingV1,
        sectors,
        sector_scratch,
        path,
        warp_id,
        record_ix,
    )
}

#[cfg(test)]
fn classify_with_l1_with_contract(
    inst: &pb::Instruction,
    schedule: &FunctionSchedule,
    arch: ArchId,
    payload_contract: MemoryPayloadContract,
    sectors: &mut Vec<i64>,
    sector_scratch: &mut Vec<i64>,
    path: &std::path::Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<AccessKind, ReplayError> {
    sector_scratch.clear();
    let preflight =
        preflight_replay_memory(inst, schedule, payload_contract, path, warp_id, record_ix)?;
    classify_with_l1_preflighted(
        preflight,
        schedule,
        arch,
        sectors,
        sector_scratch,
        path,
        warp_id,
        record_ix,
    )
}

#[cfg(test)]
fn classify_with_l1_preflighted(
    preflight: Option<ReplayMemoryPreflight<'_>>,
    schedule: &FunctionSchedule,
    arch: ArchId,
    sectors: &mut Vec<i64>,
    sector_scratch: &mut Vec<i64>,
    path: &std::path::Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<AccessKind, ReplayError> {
    let Some(preflight) = preflight else {
        return Ok(AccessKind::NonMemory);
    };
    let inst = preflight.memory.instruction();
    let meta = preflight.memory.meta();
    if meta.and_mask == 0 {
        return Ok(AccessKind::PredicatedOffMemory);
    }
    let mut sector_plan = SectorExpansionPlan::new();
    let mut sector_error: Option<SectorExpansionError> = None;
    if matches!(
        preflight.route,
        ReplayMemoryRoute::Global | ReplayMemoryRoute::Local
    ) {
        preflight.memory.visit_addresses(|access| {
            if sector_error.is_some() {
                return;
            }
            if let Err(source) = sector_plan.include_record(
                &access.lane_vector,
                (0..access.lane_vector.len())
                    .filter(|&i| access.lane_vector[i] != LANE_INACTIVE_SENTINEL),
                access.data_width_bytes,
            ) {
                sector_error = Some(source);
            }
        });
    }
    if let Some(source) = sector_error {
        return Err(ReplayError::SectorExpansion {
            path: path.to_path_buf(),
            warp_id,
            record_ix,
            source,
        });
    }
    enum StagedHierarchyRole {
        Global(MemRole),
        Local(LocalDirection),
    }

    let staged_role = match (preflight.route, meta.op_class) {
        (ReplayMemoryRoute::Shared, _) => {
            return preflighted_shared_kind(preflight.memory, path, warp_id, record_ix);
        }
        (ReplayMemoryRoute::Unsupported, class) => {
            return Err(ReplayError::UnsupportedMemoryPath { arch, class });
        }
        (ReplayMemoryRoute::Local, pb::MemoryOpClass::Local) => {
            let Some(direction) = require_known_local_direction(schedule, arch, inst.pc)? else {
                return Ok(AccessKind::NonMemory);
            };
            StagedHierarchyRole::Local(direction)
        }
        (ReplayMemoryRoute::Local, class) => {
            return Err(ReplayError::UnsupportedMemoryPath { arch, class });
        }
        (ReplayMemoryRoute::Global, pb::MemoryOpClass::Global) => {
            StagedHierarchyRole::Global(require_known_global_role(schedule, arch, inst.pc)?)
        }
        (
            ReplayMemoryRoute::Global,
            class @ (pb::MemoryOpClass::Atomic | pb::MemoryOpClass::AsyncCopy),
        ) => {
            require_sm86_special_role(schedule, arch, inst.pc, preflight.route, class)?;
            StagedHierarchyRole::Global(require_known_global_role(schedule, arch, inst.pc)?)
        }
        (ReplayMemoryRoute::Global, class) => {
            return Err(ReplayError::UnsupportedMemoryPath { arch, class });
        }
    };
    // A pc absent from the schedule surfaces as the issue loop's typed
    // PcNotInSchedule; classification stays neutral in that case. A present
    // but unrecognized static opcode fails closed instead of becoming a
    // generic untagged access. Static-role resolution follows exact address-
    // expansion errors but precedes sector-count narrowing, matching the
    // established global-route precedence without masking arithmetic faults.
    let prepared = sector_plan
        .prepare()
        .map_err(|source| ReplayError::SectorExpansion {
            path: path.to_path_buf(),
            warp_id,
            record_ix,
            source,
        })?;
    let count = prepared.unique_count();
    if count > usize::from(u16::MAX) {
        return Err(ReplayError::SectorCountOverflow {
            path: path.to_path_buf(),
            warp_id,
            record_ix,
            sectors: count,
        });
    }
    *sector_scratch = prepared
        .expand_bounded(usize::from(u16::MAX))
        .map_err(|source| ReplayError::SectorExpansion {
            path: path.to_path_buf(),
            warp_id,
            record_ix,
            source,
        })?;
    match staged_role {
        StagedHierarchyRole::Global(_role) if meta.op_class == pb::MemoryOpClass::AsyncCopy => {
            let payload_bytes =
                project_effective_data_bytes(preflight.memory).map_err(|source| {
                    ReplayError::EffectiveDataBytes {
                        path: path.to_path_buf(),
                        warp_id,
                        record_ix,
                        source,
                    }
                })?;
            append_staged_async_copy_sectors(
                arch,
                require_ldgsts_l1_policy(schedule, arch, inst.pc)?,
                payload_bytes,
                None,
                None,
                None,
                sectors,
                &mut AsyncCopyCohortReplay::default(),
                sector_scratch,
                path,
                warp_id,
                record_ix,
            )
        }
        StagedHierarchyRole::Global(role) => {
            append_staged_global_sectors(role, sectors, sector_scratch, path, warp_id, record_ix)
        }
        StagedHierarchyRole::Local(direction) => append_staged_local_sectors(
            direction,
            sectors,
            sector_scratch,
            path,
            warp_id,
            record_ix,
        ),
    }
}

#[cfg(test)]
mod aggregation_tests {
    use super::*;

    fn feedback_configuration_inputs() -> (IssueModelParams, ReplayMemMode) {
        let params = IssueModelParams::from_legacy_card_latencies_for_arch(
            ArchId::Sm86,
            34.0,
            244.0,
            423.0,
            23.0,
        )
        .unwrap();
        let mode = ReplayMemMode::L1SharedL2 {
            l1: L1Config {
                arch: ArchId::Sm86,
                line_capacity: 1,
                mshr_count: Some(1),
                hit_latency: params.l1_hit_latency(),
                miss_latency: params.l1_miss_latency(),
            },
            l2: L2Config {
                arch: ArchId::Sm86,
                capacity_bytes: 128,
                line_bytes: 128,
                sector_bytes: 32,
                num_slices: 1,
                associativity: 1,
                replacement: sim_mem::L2Replacement::Lru,
                slice_hash: sim_mem::SliceHash::LineMod,
            },
            dram: DramBw::Unmodeled { arch: ArchId::Sm86 },
        };
        (params, mode)
    }

    fn feedback_candidate(admission_latency_cycles: u32) -> MemoryAdmissionCandidate {
        MemoryAdmissionCandidate {
            candidate_id: 7,
            scope: crate::MemoryServiceCandidateScope::Smsp,
            operation: crate::MemoryServiceCandidateOperation::Store,
            queue_capacity_instructions: 16,
            admission_latency_cycles,
            l2_hit_sector_units: 1,
            l2_hit_sector_cycles: 1,
            l2_miss_sector_units: 1,
            l2_miss_sector_cycles: 10,
            l2_miss_service_width_sectors: 0,
            membar_waits_for_service_completion: false,
            write_completion_latency_cycles: 0,
            write_completion_request_cycles: 0,
            write_completion_region_cycles: 0,
            write_completion_parallel_lanes: 0,
        }
    }

    fn sm(instructions_issued: u64, l1: Option<L1Counters>) -> SmSimResult {
        SmSimResult {
            instructions_issued,
            cycles: 1,
            modeled_local_completion_cycles: None,
            l1,
            unmodeled_tails: TailFlags::default(),
            readiness: crate::ReadinessDiagnostics::default(),
            activity: crate::SmActivity::default(),
        }
    }

    #[test]
    fn record_count_overflow_preserves_context() {
        let path = Path::new("block_7.msgpack");
        assert!(matches!(
            checked_record_count(u64::MAX, path, 3, 11),
            Err(ReplayError::RecordCountOverflow {
                path: got,
                warp_id: 3,
                record_ix: 11,
                accumulated: u64::MAX,
            }) if got == path
        ));
    }

    #[test]
    fn timing_active_feedback_is_part_of_replay_configuration_identity() {
        let (params, mode) = feedback_configuration_inputs();
        let ordinary = ReplayConfiguration::checked(params, mode).unwrap();
        let candidate = feedback_candidate(params.read_barrier_latency() as u32);
        let feedback =
            ReplayConfiguration::checked_with_memory_admission_feedback(params, mode, candidate)
                .unwrap();

        assert_eq!(ordinary.memory_admission_feedback(), None);
        assert_eq!(feedback.memory_admission_feedback(), Some(candidate));
        assert_ne!(ordinary, feedback);

        let mismatched = feedback_candidate(candidate.admission_latency_cycles - 1);
        assert!(matches!(
            ReplayConfiguration::checked_with_memory_admission_feedback(params, mode, mismatched),
            Err(ReplayError::Issue(
                IssueError::MemoryAdmissionFeedbackLatencyMismatch {
                    candidate_id: 7,
                    configured_cycles: 7,
                    issue_model_cycles: 8,
                }
            ))
        ));
    }

    #[test]
    fn system_tail_mismatch_is_typed() {
        let system = TailFlags {
            red_rmw: true,
            ..TailFlags::default()
        };
        let per_sm = TailFlags::default();
        assert!(matches!(
            require_system_tail_consistency(system, per_sm),
            Err(ReplayError::SystemTailMismatch {
                system: got_system,
                per_sm: got_per_sm,
            }) if got_system == system && got_per_sm == per_sm
        ));
    }

    #[test]
    fn instruction_overflow_precedes_count_mismatch() {
        let per_sm = BTreeMap::from([(0, sm(u64::MAX, None)), (1, sm(1, None))]);
        assert!(matches!(
            fold_kernel_result(ArchId::Sm86, per_sm, 0, false, None, None),
            Err(ReplayError::InstructionCountOverflow {
                sm_id: 1,
                accumulated: u64::MAX,
                addend: 1,
            })
        ));
    }

    #[test]
    fn ordinary_count_mismatch_remains_typed() {
        let per_sm = BTreeMap::from([(0, sm(4, None))]);
        assert!(matches!(
            fold_kernel_result(ArchId::Sm86, per_sm, 5, false, None, None),
            Err(ReplayError::CountMismatch {
                issued: 4,
                records: 5,
            })
        ));
    }

    #[test]
    fn terminal_modeled_time_is_the_maximum_modeled_frontier() {
        let mut result = sm(1, None);
        result.cycles = 7;
        let system_dominates = fold_kernel_result(
            ArchId::Sm86,
            BTreeMap::from([(0, result.clone())]),
            1,
            false,
            Some(423),
            None,
        )
        .unwrap();
        assert_eq!(system_dominates.diagnostic_last_issue_cycles, 7);
        assert_eq!(system_dominates.terminal_modeled_event_time, Some(423));

        let issue_dominates = fold_kernel_result(
            ArchId::Sm86,
            BTreeMap::from([(0, result.clone())]),
            1,
            false,
            Some(3),
            None,
        )
        .unwrap();
        assert_eq!(issue_dominates.terminal_modeled_event_time, Some(7));

        let issue_only = fold_kernel_result(
            ArchId::Sm86,
            BTreeMap::from([(0, result)]),
            1,
            false,
            None,
            None,
        )
        .unwrap();
        assert_eq!(issue_only.terminal_modeled_event_time, Some(7));

        let empty =
            fold_kernel_result(ArchId::Sm86, BTreeMap::new(), 0, false, None, None).unwrap();
        assert_eq!(empty.terminal_modeled_event_time, None);
    }

    #[test]
    fn first_l1_snapshot_must_have_matching_provenance() {
        let unbound = BTreeMap::from([(0, sm(1, Some(L1Counters::default())))]);
        assert!(matches!(
            fold_kernel_result(ArchId::Sm86, unbound, 1, true, None, None),
            Err(ReplayError::L1Aggregation {
                sm_id: 0,
                source: L1CounterError::Architecture(
                    sim_arch::ArchitectureProvenanceError::Unbound {
                        component: "L1 counters",
                    }
                ),
            })
        ));

        let wrong_arch = BTreeMap::from([(0, sm(1, Some(L1Counters::for_arch(ArchId::Sm120))))]);
        assert!(matches!(
            fold_kernel_result(ArchId::Sm86, wrong_arch, 1, true, None, None),
            Err(ReplayError::L1Aggregation {
                sm_id: 0,
                source: L1CounterError::Architecture(
                    sim_arch::ArchitectureProvenanceError::Mismatch {
                        component: "L1 counters",
                        expected: ArchId::Sm86,
                        got: ArchId::Sm120,
                    }
                ),
            })
        ));
    }

    #[test]
    fn l1_presence_must_match_replay_mode_for_every_sm() {
        let all_none = BTreeMap::from([(0, sm(1, None))]);
        assert!(matches!(
            fold_kernel_result(ArchId::Sm86, all_none, 1, true, None, None),
            Err(ReplayError::L1PresenceMismatch {
                sm_id: 0,
                expected: true,
                found: false,
            })
        ));

        let all_some = BTreeMap::from([(0, sm(1, Some(L1Counters::for_arch(ArchId::Sm86))))]);
        assert!(matches!(
            fold_kernel_result(ArchId::Sm86, all_some, 1, false, None, None),
            Err(ReplayError::L1PresenceMismatch {
                sm_id: 0,
                expected: false,
                found: true,
            })
        ));

        let missing_second = BTreeMap::from([
            (0, sm(1, Some(L1Counters::for_arch(ArchId::Sm86)))),
            (1, sm(1, None)),
        ]);
        assert!(matches!(
            fold_kernel_result(ArchId::Sm86, missing_second, 2, true, None, None),
            Err(ReplayError::L1PresenceMismatch {
                sm_id: 1,
                expected: true,
                found: false,
            })
        ));

        let unexpected_second = BTreeMap::from([
            (0, sm(1, None)),
            (1, sm(1, Some(L1Counters::for_arch(ArchId::Sm86)))),
        ]);
        assert!(matches!(
            fold_kernel_result(ArchId::Sm86, unexpected_second, 2, false, None, None),
            Err(ReplayError::L1PresenceMismatch {
                sm_id: 1,
                expected: false,
                found: true,
            })
        ));
    }

    #[test]
    fn l1_overflow_keeps_responsible_sm_context() {
        let mut first = L1Counters::for_arch(ArchId::Sm86);
        first.hits = u64::MAX;
        first = first.with_unified_origin(sim_mem::MemoryOrigin::Global);
        let mut second = L1Counters::for_arch(ArchId::Sm86);
        second.hits = 1;
        second = second.with_unified_origin(sim_mem::MemoryOrigin::Global);
        let per_sm = BTreeMap::from([(0, sm(1, Some(first))), (1, sm(1, Some(second)))]);
        assert!(matches!(
            fold_kernel_result(ArchId::Sm86, per_sm, 2, true, None, None),
            Err(ReplayError::L1Aggregation {
                sm_id: 1,
                source: L1CounterError::Overflow {
                    field: sim_mem::L1CounterField::Hits,
                },
            })
        ));
    }
}

#[cfg(test)]
mod space_routing_tests {
    use super::*;
    use crate::control_bits::ControlBits;
    use crate::program::{FunctionSchedule, IssueSlot, OpClass, PipelineClass};
    use std::collections::BTreeMap;
    use std::path::Path;

    /// An ATOMIC-class instruction carrying a single address record whose
    /// per-address space is `space` (None = legacy, no address_space).
    fn atomic_inst(space: Option<pb::MemorySpace>) -> pb::Instruction {
        let mut addrs = vec![LANE_INACTIVE_SENTINEL; 32];
        addrs[0] = 0x1000;
        pb::Instruction {
            pc: 0,
            active_mask: Some(1),
            predicate_mask: Some(1),
            function_unique_id: 1,
            warpid_tb: Some(0),
            vpc: Some(0),
            data_width: Some(4),
            memory_op_class: Some(pb::MemoryOpClass::Atomic as i32),
            memory_op_access_kind: Some(pb::MemoryOpAccessKind::Atomic as i32),
            addresses: vec![pb::Address {
                data_width: 4,
                compression_format: sim_trace::lanes::FORMAT_LIST_ALL,
                addrs,
                address_space: space.map(|s| s as i32),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn global_inst() -> pb::Instruction {
        let mut addrs = vec![LANE_INACTIVE_SENTINEL; 32];
        addrs[0] = 0x1000;
        pb::Instruction {
            pc: 0,
            active_mask: Some(1),
            predicate_mask: Some(1),
            function_unique_id: 1,
            warpid_tb: Some(0),
            vpc: Some(0),
            addresses: vec![pb::Address {
                data_width: 4,
                compression_format: sim_trace::lanes::FORMAT_LIST_ALL,
                addrs,
                address_space: Some(pb::MemorySpace::Global as i32),
                ..Default::default()
            }],
            memory_op_class: Some(pb::MemoryOpClass::Global as i32),
            data_width: Some(4),
            memory_op_access_kind: Some(pb::MemoryOpAccessKind::Load as i32),
            ..Default::default()
        }
    }

    fn spaced_address(space: pb::MemorySpace, lane_zero: i64) -> pb::Address {
        let mut addrs = vec![LANE_INACTIVE_SENTINEL; 32];
        addrs[0] = lane_zero;
        pb::Address {
            data_width: 4,
            compression_format: sim_trace::lanes::FORMAT_LIST_ALL,
            addrs,
            address_space: Some(space as i32),
            ..Default::default()
        }
    }

    fn async_copy_inst(addresses: Vec<pb::Address>) -> pb::Instruction {
        pb::Instruction {
            pc: 0,
            active_mask: Some(1),
            predicate_mask: Some(1),
            function_unique_id: 1,
            addresses,
            memory_op_class: Some(pb::MemoryOpClass::AsyncCopy as i32),
            data_width: Some(4),
            warpid_tb: Some(0),
            vpc: Some(0),
            memory_op_access_kind: Some(pb::MemoryOpAccessKind::AsyncCopy as i32),
            ..Default::default()
        }
    }

    fn schedule(arch: ArchId, role: MemRole, opcode_text: &str) -> FunctionSchedule {
        FunctionSchedule {
            arch,
            function_unique_id: 1,
            kernel_name: "k".into(),
            function_name: "f".into(),
            by_pc: BTreeMap::from([(
                0,
                IssueSlot {
                    control: ControlBits::new(0, false, 7, 7, 0, 0).expect("valid test controls"),
                    op_class: OpClass::Other,
                    async_copy_wait_max_pending_groups: 0,
                    pipeline_class: PipelineClass::DispatchOnly,
                    mem_role: role,
                    ldgsts_l1_policy: matches!(role, MemRole::BypassLoad(CacheOpToken::Ldgsts))
                        .then_some(if opcode_text.split('.').any(|part| part == "BYPASS") {
                            LdgstsL1Policy::BypassL1
                        } else {
                            LdgstsL1Policy::CacheAtAllLevels
                        }),
                    fixed_latency_operands:
                        crate::sass_operands::FixedLatencyOperandStatus::NotApplicable,
                    clocklo_read: false,
                    gpu_membar: false,
                    opcode_text: opcode_text.into(),
                },
            )]),
        }
    }

    fn assert_access_kind_rejected_in_both_modes(
        inst: &pb::Instruction,
        sched: &FunctionSchedule,
        expected: &'static str,
    ) {
        let class = pb::MemoryOpClass::try_from(inst.memory_op_class.unwrap()).unwrap();
        let got = pb::MemoryOpAccessKind::try_from(inst.memory_op_access_kind.unwrap()).unwrap();
        assert!(matches!(
            classify_flat(inst, sched, sched.arch, Path::new("kind.pb"), 5, 7),
            Err(ReplayError::MemoryRoute {
                warp_id: 5,
                record_ix: 7,
                class: got_class,
                source: MemoryRouteError::AccessKindInvalid {
                    got: got_kind,
                    expected: got_expected,
                },
                ..
            }) if got_class == class && got_kind == got && got_expected == expected
        ));

        let mut sectors = vec![9];
        let mut sector_scratch = vec![99];
        assert!(matches!(
            classify_with_l1(
                inst,
                sched,
                sched.arch,
                &mut sectors,
                &mut sector_scratch,
                Path::new("kind.pb"),
                5,
                7,
            ),
            Err(ReplayError::MemoryRoute {
                warp_id: 5,
                record_ix: 7,
                class: got_class,
                source: MemoryRouteError::AccessKindInvalid {
                    got: got_kind,
                    expected: got_expected,
                },
                ..
            }) if got_class == class && got_kind == got && got_expected == expected
        ));
        assert_eq!(sectors, vec![9]);
        assert!(sector_scratch.is_empty());
    }

    fn assert_static_access_kind_rejected_in_both_modes(
        inst: &pb::Instruction,
        sched: &FunctionSchedule,
        expected: pb::MemoryOpAccessKind,
    ) {
        let class = pb::MemoryOpClass::try_from(inst.memory_op_class.unwrap()).unwrap();
        let got = pb::MemoryOpAccessKind::try_from(inst.memory_op_access_kind.unwrap()).unwrap();
        assert!(matches!(
            classify_flat(inst, sched, sched.arch, Path::new("direction.pb"), 6, 8),
            Err(ReplayError::MemoryRoute {
                warp_id: 6,
                record_ix: 8,
                class: got_class,
                source: MemoryRouteError::StaticAccessKindMismatch {
                    got: got_kind,
                    expected: got_expected,
                },
                ..
            }) if got_class == class && got_kind == got && got_expected == expected
        ));

        let mut sectors = vec![10];
        let mut sector_scratch = vec![100];
        assert!(matches!(
            classify_with_l1(
                inst,
                sched,
                sched.arch,
                &mut sectors,
                &mut sector_scratch,
                Path::new("direction.pb"),
                6,
                8,
            ),
            Err(ReplayError::MemoryRoute {
                warp_id: 6,
                record_ix: 8,
                class: got_class,
                source: MemoryRouteError::StaticAccessKindMismatch {
                    got: got_kind,
                    expected: got_expected,
                },
                ..
            }) if got_class == class && got_kind == got && got_expected == expected
        ));
        assert_eq!(sectors, vec![10]);
        assert!(sector_scratch.is_empty());
    }

    fn assert_static_route_rejected_in_both_modes(
        inst: &pb::Instruction,
        sched: &FunctionSchedule,
        expected: pb::MemorySpace,
        got: pb::MemorySpace,
    ) {
        let class = pb::MemoryOpClass::try_from(inst.memory_op_class.unwrap()).unwrap();
        assert!(matches!(
            classify_flat(inst, sched, sched.arch, Path::new("route.pb"), 7, 9),
            Err(ReplayError::MemoryRoute {
                warp_id: 7,
                record_ix: 9,
                class: got_class,
                source: MemoryRouteError::StaticMemoryRouteMismatch {
                    expected: got_expected,
                    got: got_route,
                },
                ..
            }) if got_class == class && got_expected == expected && got_route == got
        ));

        let mut sectors = vec![12];
        let mut sector_scratch = vec![101];
        assert!(matches!(
            classify_with_l1(
                inst,
                sched,
                sched.arch,
                &mut sectors,
                &mut sector_scratch,
                Path::new("route.pb"),
                7,
                9,
            ),
            Err(ReplayError::MemoryRoute {
                warp_id: 7,
                record_ix: 9,
                class: got_class,
                source: MemoryRouteError::StaticMemoryRouteMismatch {
                    expected: got_expected,
                    got: got_route,
                },
                ..
            }) if got_class == class && got_expected == expected && got_route == got
        ));
        assert_eq!(sectors, vec![12]);
        assert!(sector_scratch.is_empty());
    }

    fn assert_static_class_rejected_in_both_modes(
        inst: &pb::Instruction,
        sched: &FunctionSchedule,
        expected: pb::MemoryOpClass,
    ) {
        let got = pb::MemoryOpClass::try_from(inst.memory_op_class.unwrap()).unwrap();
        assert!(matches!(
            classify_flat(inst, sched, sched.arch, Path::new("class.pb"), 8, 10),
            Err(ReplayError::MemoryRoute {
                warp_id: 8,
                record_ix: 10,
                class,
                source: MemoryRouteError::StaticMemoryClassMismatch {
                    expected: got_expected,
                    got: got_class,
                },
                ..
            }) if class == got && got_expected == expected && got_class == got
        ));

        let mut sectors = vec![13];
        let mut sector_scratch = vec![102];
        assert!(matches!(
            classify_with_l1(
                inst,
                sched,
                sched.arch,
                &mut sectors,
                &mut sector_scratch,
                Path::new("class.pb"),
                8,
                10,
            ),
            Err(ReplayError::MemoryRoute {
                warp_id: 8,
                record_ix: 10,
                class,
                source: MemoryRouteError::StaticMemoryClassMismatch {
                    expected: got_expected,
                    got: got_class,
                },
                ..
            }) if class == got && got_expected == expected && got_class == got
        ));
        assert_eq!(sectors, vec![13]);
        assert!(sector_scratch.is_empty());
    }

    fn assert_static_family_unsupported_in_both_modes(
        inst: &pb::Instruction,
        sched: &FunctionSchedule,
        family: &'static str,
    ) {
        let class = pb::MemoryOpClass::try_from(inst.memory_op_class.unwrap()).unwrap();
        assert!(matches!(
            classify_flat(inst, sched, sched.arch, Path::new("family.pb"), 9, 11),
            Err(ReplayError::MemoryRoute {
                warp_id: 9,
                record_ix: 11,
                class: got_class,
                source: MemoryRouteError::StaticMemoryFamilyUnsupported {
                    family: got_family,
                    arch,
                },
                ..
            }) if got_class == class && got_family == family && arch == sched.arch
        ));

        let mut sectors = vec![14];
        let mut sector_scratch = vec![103];
        assert!(matches!(
            classify_with_l1(
                inst,
                sched,
                sched.arch,
                &mut sectors,
                &mut sector_scratch,
                Path::new("family.pb"),
                9,
                11,
            ),
            Err(ReplayError::MemoryRoute {
                warp_id: 9,
                record_ix: 11,
                class: got_class,
                source: MemoryRouteError::StaticMemoryFamilyUnsupported {
                    family: got_family,
                    arch,
                },
                ..
            }) if got_class == class && got_family == family && arch == sched.arch
        ));
        assert_eq!(sectors, vec![14]);
        assert!(sector_scratch.is_empty());
    }

    fn assert_unsupported_memory_path_in_both_modes(
        inst: &pb::Instruction,
        sched: &FunctionSchedule,
        expected_class: pb::MemoryOpClass,
    ) {
        assert!(matches!(
            classify_flat(inst, sched, sched.arch, Path::new("unsupported.pb"), 10, 12),
            Err(ReplayError::UnsupportedMemoryPath { arch, class })
                if arch == sched.arch && class == expected_class
        ));

        let mut sectors = vec![15];
        let mut sector_scratch = vec![104];
        assert!(matches!(
            classify_with_l1(
                inst,
                sched,
                sched.arch,
                &mut sectors,
                &mut sector_scratch,
                Path::new("unsupported.pb"),
                10,
                12,
            ),
            Err(ReplayError::UnsupportedMemoryPath { arch, class })
                if arch == sched.arch && class == expected_class
        ));
        assert_eq!(sectors, vec![15]);
        assert!(sector_scratch.is_empty());
    }

    fn assert_wire_contract_rejected_in_both_modes(
        inst: &pb::Instruction,
        sched: &FunctionSchedule,
        expected_reason: &'static str,
    ) {
        let typed_reason = |result: Result<AccessKind, ReplayError>| match result.unwrap_err() {
            ReplayError::Trace(error) => error
                .typed_reason()
                .unwrap_or_else(|| panic!("wire error has no stable reason: {error:?}")),
            error => panic!("expected a wire-contract rejection, got {error:?}"),
        };
        assert_eq!(
            typed_reason(classify_flat(
                inst,
                sched,
                sched.arch,
                Path::new("wire_contract.pb"),
                11,
                13,
            )),
            expected_reason
        );

        let mut sectors = vec![17];
        let mut sector_scratch = vec![106];
        assert_eq!(
            typed_reason(classify_with_l1(
                inst,
                sched,
                sched.arch,
                &mut sectors,
                &mut sector_scratch,
                Path::new("wire_contract.pb"),
                11,
                13,
            )),
            expected_reason
        );
        assert_eq!(sectors, vec![17]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn flat_and_l1_modes_share_wire_contract_rejections() {
        let sched = schedule(
            ArchId::Sm86,
            MemRole::EligibleLoad(CacheOpToken::LdgECa),
            "LDG.E.CA R0, [R2.64]",
        );
        let cases: Vec<(pb::Instruction, &str)> = vec![
            (
                {
                    let mut inst = global_inst();
                    inst.active_mask = None;
                    inst
                },
                "active_mask_missing",
            ),
            (
                {
                    let mut inst = global_inst();
                    inst.predicate_mask = None;
                    inst
                },
                "predicate_mask_missing",
            ),
            (
                {
                    let mut inst = global_inst();
                    inst.data_width = None;
                    inst
                },
                "data_width_missing",
            ),
            (
                {
                    let mut inst = global_inst();
                    inst.memory_op_class = Some(pb::MemoryOpClass::Unspecified as i32);
                    inst
                },
                "memory_op_class_unspecified",
            ),
            (
                {
                    let mut inst = global_inst();
                    inst.memory_op_class = Some(99);
                    inst
                },
                "unknown_enum_value",
            ),
            (
                {
                    let mut inst = global_inst();
                    inst.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Unspecified as i32);
                    inst
                },
                "memory_op_access_kind_unspecified",
            ),
            (
                {
                    let mut inst = global_inst();
                    inst.memory_op_access_kind = Some(99);
                    inst
                },
                "unknown_enum_value",
            ),
            (
                {
                    let mut inst = global_inst();
                    inst.instruction_cache_op = Some(99);
                    inst
                },
                "unknown_enum_value",
            ),
            (
                {
                    let mut inst = global_inst();
                    inst.addresses[0].address_space = Some(pb::MemorySpace::Unspecified as i32);
                    inst
                },
                "memory_space_unspecified",
            ),
            (
                {
                    let mut inst = global_inst();
                    inst.addresses[0].address_space = Some(99);
                    inst
                },
                "unknown_enum_value",
            ),
        ];

        for (inst, expected_reason) in cases {
            assert_wire_contract_rejected_in_both_modes(&inst, &sched, expected_reason);
        }
    }

    #[test]
    fn sector_expansion_overflow_carries_replay_context() {
        let mut addrs = vec![LANE_INACTIVE_SENTINEL; 32];
        addrs[0] = i64::MAX;
        let mut later_addrs = vec![LANE_INACTIVE_SENTINEL; 32];
        later_addrs[0] = 0;
        let inst = pb::Instruction {
            pc: 0,
            active_mask: Some(1),
            predicate_mask: Some(1),
            function_unique_id: 1,
            warpid_tb: Some(0),
            vpc: Some(0),
            addresses: vec![
                pb::Address {
                    data_width: 2,
                    compression_format: sim_trace::lanes::FORMAT_LIST_ALL,
                    addrs,
                    address_space: Some(pb::MemorySpace::Global as i32),
                    ..Default::default()
                },
                pb::Address {
                    data_width: 2,
                    compression_format: sim_trace::lanes::FORMAT_LIST_ALL,
                    addrs: later_addrs,
                    address_space: Some(pb::MemorySpace::Global as i32),
                    ..Default::default()
                },
            ],
            memory_op_class: Some(pb::MemoryOpClass::Global as i32),
            data_width: Some(2),
            memory_op_access_kind: Some(pb::MemoryOpAccessKind::Load as i32),
            ..Default::default()
        };
        let mut sectors = vec![11];
        let mut sector_scratch = vec![99];
        let err = classify_with_l1(
            &inst,
            &schedule(ArchId::Sm86, MemRole::None, "LDG.E"),
            ArchId::Sm86,
            &mut sectors,
            &mut sector_scratch,
            Path::new("block.pb"),
            7,
            9,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ReplayError::SectorExpansion {
                ref path,
                warp_id: 7,
                record_ix: 9,
                source: SectorExpansionError::AddressOverflow {
                    lane: 0,
                    base: i64::MAX,
                    data_width_bytes: 2,
                },
            } if path == Path::new("block.pb")
        ));
        assert_eq!(sectors, vec![11]);
    }

    #[test]
    fn oversized_data_width_is_rejected_before_sector_expansion() {
        let mut inst = global_inst();
        inst.data_width = Some(i32::MAX);
        inst.addresses[0].data_width = i32::MAX;
        inst.addresses[0].addrs[0] = 0;
        let mut sectors = vec![13];
        let mut sector_scratch = vec![99];

        let err = classify_with_l1(
            &inst,
            &schedule(
                ArchId::Sm86,
                MemRole::EligibleLoad(CacheOpToken::LdgECa),
                "LDG.E.CA",
            ),
            ArchId::Sm86,
            &mut sectors,
            &mut sector_scratch,
            Path::new("wide.pb"),
            8,
            10,
        )
        .unwrap_err();

        assert!(matches!(
            err,
            ReplayError::SectorCountOverflow {
                ref path,
                warp_id: 8,
                record_ix: 10,
                sectors: 67_108_864,
            } if path == Path::new("wide.pb")
        ));
        assert_eq!(sectors, vec![13]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn multi_address_overflow_reports_the_exact_instruction_union() {
        const WIDTH: i32 = 96_000;
        let first = vec![0; 32];
        let second = (0_i64..32)
            .map(|lane| (3_000 + lane * 3_000) * 32)
            .collect();
        let inst = pb::Instruction {
            pc: 0,
            active_mask: Some(u32::MAX),
            predicate_mask: Some(u32::MAX),
            function_unique_id: 1,
            warpid_tb: Some(0),
            vpc: Some(0),
            addresses: vec![
                pb::Address {
                    data_width: WIDTH,
                    compression_format: sim_trace::lanes::FORMAT_LIST_ALL,
                    addrs: first,
                    address_space: Some(pb::MemorySpace::Global as i32),
                    ..Default::default()
                },
                pb::Address {
                    data_width: WIDTH,
                    compression_format: sim_trace::lanes::FORMAT_LIST_ALL,
                    addrs: second,
                    address_space: Some(pb::MemorySpace::Global as i32),
                    ..Default::default()
                },
            ],
            memory_op_class: Some(pb::MemoryOpClass::Global as i32),
            data_width: Some(WIDTH),
            memory_op_access_kind: Some(pb::MemoryOpAccessKind::Load as i32),
            ..Default::default()
        };
        let mut sectors = vec![15];
        let mut sector_scratch = vec![99];

        let err = classify_with_l1(
            &inst,
            &schedule(
                ArchId::Sm86,
                MemRole::EligibleLoad(CacheOpToken::LdgECa),
                "LDG.E.CA",
            ),
            ArchId::Sm86,
            &mut sectors,
            &mut sector_scratch,
            Path::new("multi-wide.pb"),
            9,
            11,
        )
        .unwrap_err();

        assert!(matches!(
            err,
            ReplayError::SectorCountOverflow {
                ref path,
                warp_id: 9,
                record_ix: 11,
                sectors: 99_000,
            } if path == Path::new("multi-wide.pb")
        ));
        assert_eq!(sectors, vec![15]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn oversized_count_does_not_mask_unknown_static_opcode() {
        let mut inst = global_inst();
        inst.data_width = Some(i32::MAX);
        inst.addresses[0].data_width = i32::MAX;
        inst.addresses[0].addrs[0] = 0;
        let mut sectors = vec![17];
        let mut sector_scratch = vec![99];

        assert!(matches!(
            classify_with_l1(
                &inst,
                &schedule(ArchId::Sm86, MemRole::None, "IADD3 R0, R1, R2, RZ"),
                ArchId::Sm86,
                &mut sectors,
                &mut sector_scratch,
                Path::new("wide-opcode.pb"),
                10,
                12,
            ),
            Err(ReplayError::UnknownGlobalOpcode {
                arch: ArchId::Sm86,
                pc: 0,
                ref opcode,
            }) if opcode == "IADD3"
        ));
        assert_eq!(sectors, vec![17]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn oversized_count_does_not_mask_unsupported_special_route() {
        let mut inst = atomic_inst(Some(pb::MemorySpace::Global));
        inst.function_unique_id = 1;
        inst.data_width = Some(i32::MAX);
        inst.addresses[0].data_width = i32::MAX;
        inst.addresses[0].addrs[0] = 0;
        let mut sectors = vec![19];
        let mut sector_scratch = vec![99];

        assert!(matches!(
            classify_with_l1(
                &inst,
                &schedule(ArchId::Sm120, MemRole::None, "ATOMG.E.ADD.STRONG.GPU"),
                ArchId::Sm120,
                &mut sectors,
                &mut sector_scratch,
                Path::new("wide-atomic.pb"),
                11,
                13,
            ),
            Err(ReplayError::UnsupportedMemoryPath {
                arch: ArchId::Sm120,
                class: pb::MemoryOpClass::Atomic,
            })
        ));
        assert_eq!(sectors, vec![19]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn later_trace_error_precedes_sector_callbacks() {
        let mut overflow = vec![LANE_INACTIVE_SENTINEL; 32];
        overflow[0] = i64::MAX;
        let inst = pb::Instruction {
            pc: 0,
            active_mask: Some(1),
            predicate_mask: Some(1),
            function_unique_id: 1,
            warpid_tb: Some(0),
            vpc: Some(0),
            addresses: vec![
                pb::Address {
                    data_width: 2,
                    compression_format: sim_trace::lanes::FORMAT_LIST_ALL,
                    addrs: overflow,
                    address_space: Some(pb::MemorySpace::Global as i32),
                    ..Default::default()
                },
                pb::Address {
                    data_width: 2,
                    compression_format: sim_trace::lanes::FORMAT_LIST_ALL,
                    addrs: Vec::new(),
                    address_space: Some(pb::MemorySpace::Shared as i32),
                    ..Default::default()
                },
            ],
            memory_op_class: Some(pb::MemoryOpClass::Global as i32),
            data_width: Some(2),
            ..Default::default()
        };
        assert!(matches!(
            classify_flat(
                &inst,
                &schedule(ArchId::Sm86, MemRole::None, "LDG.E"),
                ArchId::Sm86,
                Path::new("precedence.pb"),
                2,
                4,
            ),
            Err(ReplayError::Trace(TraceError::Lane {
                source: sim_trace::lanes::LaneError::ListAllLengthInvalid { got: 0 },
                ..
            }))
        ));
        let mut sectors = vec![13];
        let mut sector_scratch = vec![99];
        assert!(matches!(
            classify_with_l1(
                &inst,
                &schedule(ArchId::Sm86, MemRole::None, "LDG.E"),
                ArchId::Sm86,
                &mut sectors,
                &mut sector_scratch,
                Path::new("precedence.pb"),
                2,
                4,
            ),
            Err(ReplayError::Trace(TraceError::Lane {
                source: sim_trace::lanes::LaneError::ListAllLengthInvalid { got: 0 },
                ..
            }))
        ));
        assert_eq!(sectors, vec![13]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn alias_expanded_sectors_reconstruct_exact_l2_atomic_requests() {
        assert_eq!(atomic_l2_request_count_128b(&[]), 0);
        assert_eq!(atomic_l2_request_count_128b(&[0, 1, 2, 3]), 1);
        assert_eq!(atomic_l2_request_count_128b(&[0, 0, 1, 1]), 2);
        assert_eq!(atomic_l2_request_count_128b(&[0, 0, 0, 0]), 4);
        assert_eq!(atomic_l2_request_count_128b(&[0, 4, 8, 12]), 4);
        assert_eq!(atomic_l2_request_count_128b(&[0, 0, 1, 1, 4, 4, 5, 5]), 4);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn staged_sector_start_uses_full_usize_domain() {
        let start = usize::try_from(1_u64 << 32).unwrap();
        assert_eq!(
            checked_staged_global_kind(MemRole::None, start, 1, 0),
            Some(AccessKind::GlobalUntagged { off: start, len: 1 })
        );
        assert_eq!(
            checked_staged_global_kind(MemRole::None, usize::MAX, 1, 0),
            None
        );
    }

    #[test]
    fn staged_sector_count_overflow_precedes_arena_append() {
        let count = usize::from(u16::MAX)
            .checked_add(1)
            .expect("u16 maximum plus one fits usize");
        let scratch = vec![0; count];
        let mut sectors = vec![17];
        let err = append_staged_global_sectors(
            MemRole::None,
            &mut sectors,
            &scratch,
            Path::new("count.pb"),
            3,
            5,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ReplayError::SectorCountOverflow {
                ref path,
                warp_id: 3,
                record_ix: 5,
                sectors,
            } if path == Path::new("count.pb") && sectors == count
        ));
        assert_eq!(sectors, vec![17]);
    }

    #[test]
    fn unknown_static_opcode_leaves_sector_arena_unchanged() {
        let inst = global_inst();
        let mut sectors = vec![19];
        let mut sector_scratch = vec![99];
        let err = classify_with_l1(
            &inst,
            &schedule(ArchId::Sm86, MemRole::None, "IADD3 R0, R1, R2, RZ"),
            ArchId::Sm86,
            &mut sectors,
            &mut sector_scratch,
            Path::new("opcode.pb"),
            4,
            6,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ReplayError::UnknownGlobalOpcode {
                arch: ArchId::Sm86,
                pc: 0,
                ref opcode,
            } if opcode == "IADD3"
        ));
        assert_eq!(sectors, vec![19]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn active_no_payload_global_is_typed_rejected_in_both_modes() {
        let sched = schedule(ArchId::Sm86, MemRole::None, "LDG.E");
        let mut inst = global_inst();
        inst.addresses.clear();

        assert!(matches!(
            classify_flat(
                &inst,
                &sched,
                sched.arch,
                Path::new("empty_global.pb"),
                2,
                4,
            ),
            Err(ReplayError::MemoryRoute {
                warp_id: 2,
                record_ix: 4,
                class: pb::MemoryOpClass::Global,
                source: MemoryRouteError::GlobalOperandCount { got: 0 },
                ..
            })
        ));

        let mut sectors = vec![21];
        let mut sector_scratch = vec![99];
        assert!(matches!(
            classify_with_l1(
                &inst,
                &sched,
                sched.arch,
                &mut sectors,
                &mut sector_scratch,
                Path::new("empty_global.pb"),
                2,
                4,
            ),
            Err(ReplayError::MemoryRoute {
                warp_id: 2,
                record_ix: 4,
                class: pb::MemoryOpClass::Global,
                source: MemoryRouteError::GlobalOperandCount { got: 0 },
                ..
            })
        ));
        assert_eq!(sectors, vec![21]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn zero_active_no_payload_global_becomes_predicated_off_issue() {
        let sched = schedule(ArchId::Sm86, MemRole::None, "LD.E R0, [R2.64]");
        let mut inst = global_inst();
        inst.predicate_mask = Some(0);
        inst.addresses.clear();

        let flat = classify_flat(&inst, &sched, sched.arch, Path::new("zero_global.pb"), 0, 0)
            .expect("zero-active class-only GLOBAL retains its legacy route");
        assert_eq!(flat, AccessKind::PredicatedOffMemory);

        let mut sectors = vec![23];
        let mut sector_scratch = vec![99];
        let l1 = classify_with_l1(
            &inst,
            &sched,
            sched.arch,
            &mut sectors,
            &mut sector_scratch,
            Path::new("zero_global.pb"),
            0,
            0,
        )
        .expect("zero-active class-only GLOBAL retains its L1 route");
        assert_eq!(l1, AccessKind::PredicatedOffMemory);
        assert_eq!(sectors, vec![23]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn active_no_payload_shared_or_local_is_typed_rejected_in_both_modes() {
        for class in [pb::MemoryOpClass::Shared, pb::MemoryOpClass::Local] {
            let sched = schedule(ArchId::Sm86, MemRole::None, "LDS");
            let mut inst = global_inst();
            inst.memory_op_class = Some(class as i32);
            inst.addresses.clear();

            assert!(matches!(
                classify_flat(
                    &inst,
                    &sched,
                    sched.arch,
                    Path::new("empty_shared.pb"),
                    2,
                    5,
                ),
                Err(ReplayError::MemoryRoute {
                    warp_id: 2,
                    record_ix: 5,
                    class: got_class,
                    source: MemoryRouteError::SharedLocalOperandCount { got: 0 },
                    ..
                }) if got_class == class
            ));

            let mut sectors = vec![25];
            let mut sector_scratch = vec![99];
            assert!(matches!(
                classify_with_l1(
                    &inst,
                    &sched,
                    sched.arch,
                    &mut sectors,
                    &mut sector_scratch,
                    Path::new("empty_shared.pb"),
                    2,
                    5,
                ),
                Err(ReplayError::MemoryRoute {
                    warp_id: 2,
                    record_ix: 5,
                    class: got_class,
                    source: MemoryRouteError::SharedLocalOperandCount { got: 0 },
                    ..
                }) if got_class == class
            ));
            assert_eq!(sectors, vec![25]);
            assert!(sector_scratch.is_empty());
        }
    }

    #[test]
    fn valid_ldl_uses_local_sector_route() {
        let sched = schedule(
            ArchId::Sm86,
            MemRole::EligibleLoad(CacheOpToken::Ldl),
            "LDL R0, [R1]",
        );
        let mut inst = global_inst();
        inst.memory_op_class = Some(pb::MemoryOpClass::Local as i32);
        inst.addresses[0].address_space = Some(pb::MemorySpace::Local as i32);

        let flat = classify_flat(&inst, &sched, sched.arch, Path::new("local.pb"), 0, 0)
            .expect("joined LDL has an implemented flat Local route");
        assert!(matches!(flat, AccessKind::LocalLoad { off: 0, len: 0 }));

        let mut sectors = vec![31];
        let mut sector_scratch = vec![99];
        let l1 = classify_with_l1(
            &inst,
            &sched,
            sched.arch,
            &mut sectors,
            &mut sector_scratch,
            Path::new("local.pb"),
            0,
            0,
        )
        .expect("joined LDL has an implemented hierarchy Local route");
        assert!(matches!(l1, AccessKind::LocalLoad { off: 1, len: 1 }));
        assert_eq!(sectors, vec![31, 0x1000 >> 5]);
        assert_eq!(sector_scratch, vec![0x1000 >> 5]);
    }

    #[test]
    fn valid_stl_preserves_store_direction_on_local_sector_route() {
        let sched = schedule(
            ArchId::Sm86,
            MemRole::EligibleLoad(CacheOpToken::Stl),
            "STL [R1], R0",
        );
        let mut inst = global_inst();
        inst.memory_op_class = Some(pb::MemoryOpClass::Local as i32);
        inst.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Store as i32);
        inst.addresses[0].address_space = Some(pb::MemorySpace::Local as i32);

        let flat = classify_flat(&inst, &sched, sched.arch, Path::new("stl.pb"), 0, 0)
            .expect("joined STL has an implemented flat Local route");
        assert!(matches!(flat, AccessKind::LocalStore { off: 0, len: 0 }));

        let mut sectors = Vec::new();
        let mut sector_scratch = Vec::new();
        let l1 = classify_with_l1(
            &inst,
            &sched,
            sched.arch,
            &mut sectors,
            &mut sector_scratch,
            Path::new("stl.pb"),
            0,
            0,
        )
        .expect("joined STL has an implemented hierarchy Local route");
        assert!(matches!(l1, AccessKind::LocalStore { off: 0, len: 1 }));
        assert_eq!(sectors, vec![0x1000 >> 5]);
        assert_eq!(sector_scratch, sectors);
    }

    #[test]
    fn local_route_rejects_non_ldl_stl_static_roles_without_append() {
        let sched = schedule(ArchId::Sm86, MemRole::None, "IADD3 R0, R1, R2, RZ");
        let mut inst = global_inst();
        inst.memory_op_class = Some(pb::MemoryOpClass::Local as i32);
        inst.addresses[0].address_space = Some(pb::MemorySpace::Local as i32);

        assert!(matches!(
            classify_flat(&inst, &sched, sched.arch, Path::new("local_role.pb"), 2, 3),
            Err(ReplayError::UnknownLocalOpcode {
                arch: ArchId::Sm86,
                pc: 0,
                ref opcode,
            }) if opcode == "IADD3"
        ));
        let mut sectors = vec![17];
        let mut sector_scratch = vec![99];
        assert!(matches!(
            classify_with_l1(
                &inst,
                &sched,
                sched.arch,
                &mut sectors,
                &mut sector_scratch,
                Path::new("local_role.pb"),
                2,
                3,
            ),
            Err(ReplayError::UnknownLocalOpcode {
                arch: ArchId::Sm86,
                pc: 0,
                ref opcode,
            }) if opcode == "IADD3"
        ));
        assert_eq!(sectors, vec![17]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn static_ldl_stl_evidence_cannot_escape_local_route() {
        let assert_rejected =
            |inst: &pb::Instruction, sched: &FunctionSchedule, expected_class, got_space| {
                assert!(matches!(
                    classify_flat(
                        inst,
                        sched,
                        sched.arch,
                        Path::new("escaped_local.pb"),
                        3,
                        4,
                    ),
                    Err(ReplayError::MemoryRoute {
                        class,
                        source: MemoryRouteError::StaticMemoryRouteMismatch {
                            expected: pb::MemorySpace::Local,
                            got,
                        },
                        ..
                    }) if class == expected_class && got == got_space
                ));

                let mut sectors = vec![23];
                let mut sector_scratch = vec![99];
                assert!(matches!(
                    classify_with_l1(
                        inst,
                        sched,
                        sched.arch,
                        &mut sectors,
                        &mut sector_scratch,
                        Path::new("escaped_local.pb"),
                        3,
                        4,
                    ),
                    Err(ReplayError::MemoryRoute {
                        class,
                        source: MemoryRouteError::StaticMemoryRouteMismatch {
                            expected: pb::MemorySpace::Local,
                            got,
                        },
                        ..
                    }) if class == expected_class && got == got_space
                ));
                assert_eq!(sectors, vec![23]);
                assert!(sector_scratch.is_empty());
            };

        // The concrete contamination case: STL's EligibleLoad role must never
        // become a GlobalEligible load merely because the dynamic record says
        // GLOBAL/STORE.
        let mut global_stl = global_inst();
        global_stl.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Store as i32);
        assert_rejected(
            &global_stl,
            &schedule(
                ArchId::Sm86,
                MemRole::EligibleLoad(CacheOpToken::Stl),
                "STL [R1], R0",
            ),
            pb::MemoryOpClass::Global,
            pb::MemorySpace::Global,
        );

        // The reciprocal guard is independent of the additive direction field:
        // legacy-absent access_kind cannot let LDL escape through Shared timing.
        let mut shared_ldl = global_inst();
        shared_ldl.memory_op_class = Some(pb::MemoryOpClass::Shared as i32);
        shared_ldl.memory_op_access_kind = None;
        shared_ldl.addresses[0].address_space = Some(pb::MemorySpace::Shared as i32);
        assert_rejected(
            &shared_ldl,
            &schedule(
                ArchId::Sm86,
                MemRole::EligibleLoad(CacheOpToken::Ldl),
                "LDL R0, [R1]",
            ),
            pb::MemoryOpClass::Shared,
            pb::MemorySpace::Shared,
        );
    }

    #[test]
    fn local_address_overflow_precedes_unknown_static_opcode() {
        let mut inst = global_inst();
        inst.memory_op_class = Some(pb::MemoryOpClass::Local as i32);
        inst.data_width = Some(2);
        inst.addresses[0].address_space = Some(pb::MemorySpace::Local as i32);
        inst.addresses[0].data_width = 2;
        inst.addresses[0].addrs[0] = i64::MAX;
        let mut sectors = vec![29];
        let mut sector_scratch = vec![99];

        assert!(matches!(
            classify_with_l1(
                &inst,
                &schedule(ArchId::Sm86, MemRole::None, "IADD3 R0, R1, R2, RZ"),
                ArchId::Sm86,
                &mut sectors,
                &mut sector_scratch,
                Path::new("local_overflow.pb"),
                5,
                6,
            ),
            Err(ReplayError::SectorExpansion {
                warp_id: 5,
                record_ix: 6,
                source: SectorExpansionError::AddressOverflow {
                    lane: 0,
                    base: i64::MAX,
                    data_width_bytes: 2,
                },
                ..
            })
        ));
        assert_eq!(sectors, vec![29]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn unknown_local_opcode_precedes_sector_count_narrowing() {
        let mut inst = global_inst();
        inst.memory_op_class = Some(pb::MemoryOpClass::Local as i32);
        inst.data_width = Some(i32::MAX);
        inst.addresses[0].address_space = Some(pb::MemorySpace::Local as i32);
        inst.addresses[0].data_width = i32::MAX;
        inst.addresses[0].addrs[0] = 0;
        let mut sectors = vec![31];
        let mut sector_scratch = vec![99];

        assert!(matches!(
            classify_with_l1(
                &inst,
                &schedule(ArchId::Sm86, MemRole::None, "IADD3 R0, R1, R2, RZ"),
                ArchId::Sm86,
                &mut sectors,
                &mut sector_scratch,
                Path::new("local_wide_opcode.pb"),
                7,
                8,
            ),
            Err(ReplayError::UnknownLocalOpcode {
                arch: ArchId::Sm86,
                pc: 0,
                ref opcode,
            }) if opcode == "IADD3"
        ));
        assert_eq!(sectors, vec![31]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn zero_active_no_payload_shared_becomes_predicated_off_issue() {
        let sched = schedule(ArchId::Sm86, MemRole::None, "LDS");
        let mut inst = global_inst();
        inst.memory_op_class = Some(pb::MemoryOpClass::Shared as i32);
        inst.predicate_mask = Some(0);
        inst.addresses.clear();

        let flat = classify_flat(&inst, &sched, sched.arch, Path::new("zero_shared.pb"), 0, 0)
            .expect("zero-active class-only SHARED retains its legacy route");
        assert_eq!(flat, AccessKind::PredicatedOffMemory);

        let mut sectors = vec![26];
        let mut sector_scratch = vec![99];
        let l1 = classify_with_l1(
            &inst,
            &sched,
            sched.arch,
            &mut sectors,
            &mut sector_scratch,
            Path::new("zero_shared.pb"),
            0,
            0,
        )
        .expect("zero-active class-only SHARED retains its L1 route");
        assert_eq!(l1, AccessKind::PredicatedOffMemory);
        assert_eq!(sectors, vec![26]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn shared_or_local_class_rejects_explicit_global_space_in_both_modes() {
        for class in [pb::MemoryOpClass::Shared, pb::MemoryOpClass::Local] {
            let sched = schedule(ArchId::Sm86, MemRole::None, "LDS");
            let mut inst = global_inst();
            inst.memory_op_class = Some(class as i32);

            assert!(matches!(
                classify_flat(&inst, &sched, sched.arch, Path::new("space.pb"), 3, 5),
                Err(ReplayError::MemoryRoute {
                    warp_id: 3,
                    record_ix: 5,
                    class: got_class,
                    source: MemoryRouteError::SharedLocalSpaceInvalid {
                        operand: 0,
                        space: pb::MemorySpace::Global,
                    },
                    ..
                }) if got_class == class
            ));

            let mut sectors = vec![27];
            let mut sector_scratch = vec![99];
            assert!(matches!(
                classify_with_l1(
                    &inst,
                    &sched,
                    sched.arch,
                    &mut sectors,
                    &mut sector_scratch,
                    Path::new("space.pb"),
                    3,
                    5,
                ),
                Err(ReplayError::MemoryRoute {
                    warp_id: 3,
                    record_ix: 5,
                    class: got_class,
                    source: MemoryRouteError::SharedLocalSpaceInvalid {
                        operand: 0,
                        space: pb::MemorySpace::Global,
                    },
                    ..
                }) if got_class == class
            ));
            assert_eq!(sectors, vec![27]);
            assert!(sector_scratch.is_empty());
        }
    }

    #[test]
    fn shared_and_local_classes_reject_crossed_spaces() {
        let cases = [
            (pb::MemoryOpClass::Shared, pb::MemorySpace::Local),
            (pb::MemoryOpClass::Local, pb::MemorySpace::Shared),
        ];
        for (case, (class, space)) in cases.into_iter().enumerate() {
            let sched = schedule(ArchId::Sm86, MemRole::None, "LDS");
            let mut inst = global_inst();
            inst.memory_op_class = Some(class as i32);
            inst.addresses[0].address_space = Some(space as i32);

            assert!(matches!(
                classify_flat(
                    &inst,
                    &sched,
                    sched.arch,
                    Path::new("crossed_space.pb"),
                    4,
                    case,
                ),
                Err(ReplayError::MemoryRoute {
                    warp_id: 4,
                    record_ix,
                    class: got_class,
                    source: MemoryRouteError::SharedLocalSpaceInvalid {
                        operand: 0,
                        space: got_space,
                    },
                    ..
                }) if record_ix == case && got_class == class && got_space == space
            ));

            // Exercise the replay boundary directly as a defense independent of
            // the reader's equivalent preflight rejection.
            assert!(matches!(
                replay_memory_route(
                    &inst,
                    class,
                    MemoryPayloadContract::AddressCarryingV1,
                    Path::new("crossed_space.pb"),
                    4,
                    case,
                ),
                Err(ReplayError::MemoryRoute {
                    class: got_class,
                    source: MemoryRouteError::SharedLocalSpaceInvalid {
                        operand: 0,
                        space: got_space,
                    },
                    ..
                }) if got_class == class && got_space == space
            ));
        }
    }

    #[test]
    fn local_atomic_is_never_collapsed_into_shared_timing() {
        let sched = schedule(ArchId::Sm86, MemRole::None, "IADD3 R0, R1, R2, RZ");
        let mut inst = atomic_inst(Some(pb::MemorySpace::Local));
        inst.function_unique_id = 1;

        assert_eq!(
            replay_memory_route(
                &inst,
                pb::MemoryOpClass::Atomic,
                MemoryPayloadContract::AddressCarryingV1,
                Path::new("local_atomic.pb"),
                8,
                10,
            )
            .expect("an explicit Local operand resolves to the Local route"),
            ReplayMemoryRoute::Local
        );
        assert!(matches!(
            classify_flat(
                &inst,
                &sched,
                sched.arch,
                Path::new("local_atomic.pb"),
                8,
                10,
            ),
            Err(ReplayError::UnsupportedMemoryPath {
                arch: ArchId::Sm86,
                class: pb::MemoryOpClass::Atomic,
            })
        ));

        let mut sectors = vec![39];
        let mut sector_scratch = vec![99];
        assert!(matches!(
            classify_with_l1(
                &inst,
                &sched,
                sched.arch,
                &mut sectors,
                &mut sector_scratch,
                Path::new("local_atomic.pb"),
                8,
                10,
            ),
            Err(ReplayError::UnsupportedMemoryPath {
                arch: ArchId::Sm86,
                class: pb::MemoryOpClass::Atomic,
            })
        ));
        assert_eq!(sectors, vec![39]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn legacy_atomic_shared_and_local_operands_conflict() {
        let sched = schedule(ArchId::Sm86, MemRole::None, "IADD3 R0, R1, R2, RZ");
        let mut inst = atomic_inst(Some(pb::MemorySpace::Shared));
        inst.function_unique_id = 1;
        inst.addresses
            .push(spaced_address(pb::MemorySpace::Local, 0x1040));

        assert!(matches!(
            classify_flat_with_contract(
                &inst,
                &sched,
                sched.arch,
                MemoryPayloadContract::LegacyExclusion,
                Path::new("legacy_local_atomic.pb"),
                9,
                11,
            ),
            Err(ReplayError::MemoryRoute {
                class: pb::MemoryOpClass::Atomic,
                source: MemoryRouteError::AtomicSpacesConflict,
                ..
            })
        ));

        let mut sectors = vec![41];
        let mut sector_scratch = vec![99];
        assert!(matches!(
            classify_with_l1_with_contract(
                &inst,
                &sched,
                sched.arch,
                MemoryPayloadContract::LegacyExclusion,
                &mut sectors,
                &mut sector_scratch,
                Path::new("legacy_local_atomic.pb"),
                9,
                11,
            ),
            Err(ReplayError::MemoryRoute {
                class: pb::MemoryOpClass::Atomic,
                source: MemoryRouteError::AtomicSpacesConflict,
                ..
            })
        ));
        assert_eq!(sectors, vec![41]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn zero_active_shared_atomic_becomes_predicated_off_after_route_validation() {
        let mut inst = atomic_inst(Some(pb::MemorySpace::Shared));
        inst.function_unique_id = 1;
        inst.active_mask = Some(1);
        inst.predicate_mask = Some(0);
        inst.data_width = Some(4);
        inst.addresses[0].data_width = 4;
        inst.addresses[0].compression_format = sim_trace::lanes::FORMAT_BASE_DELTA;
        inst.addresses[0].addrs.clear();
        let sched = schedule(
            ArchId::Sm120,
            MemRole::BypassLoad(CacheOpToken::Atom),
            "ATOMS.E.ADD",
        );

        let flat = classify_flat(&inst, &sched, sched.arch, Path::new("zero.pb"), 0, 0)
            .expect("zero-active shared atomic has an implemented flat route");
        assert_eq!(flat, AccessKind::PredicatedOffMemory);

        let mut sectors = vec![23];
        let mut sector_scratch = vec![99];
        let l1 = classify_with_l1(
            &inst,
            &sched,
            sched.arch,
            &mut sectors,
            &mut sector_scratch,
            Path::new("zero.pb"),
            0,
            0,
        )
        .expect("zero-active shared atomic has the same L1 route");
        assert_eq!(l1, AccessKind::PredicatedOffMemory);
        assert_eq!(sectors, vec![23]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn zero_active_no_payload_atomic_becomes_predicated_off_issue() {
        let sched = schedule(
            ArchId::Sm86,
            MemRole::BypassLoad(CacheOpToken::Atom),
            "ATOMG.E.CAS.STRONG.GPU",
        );
        let mut inst = atomic_inst(Some(pb::MemorySpace::Global));
        inst.predicate_mask = Some(0);
        inst.addresses.clear();

        let flat = classify_flat(&inst, &sched, sched.arch, Path::new("zero_atomic.pb"), 0, 0)
            .expect("zero-active no-payload legacy atomic retains its global route");
        assert_eq!(flat, AccessKind::PredicatedOffMemory);

        let mut sectors = vec![29];
        let mut sector_scratch = vec![99];
        let l1 = classify_with_l1(
            &inst,
            &sched,
            sched.arch,
            &mut sectors,
            &mut sector_scratch,
            Path::new("zero_atomic.pb"),
            0,
            0,
        )
        .expect("zero-active no-payload legacy atomic retains its L1 route");
        assert_eq!(l1, AccessKind::PredicatedOffMemory);
        assert_eq!(sectors, vec![29]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn active_no_payload_atomic_is_typed_rejected_in_both_modes() {
        let sched = schedule(
            ArchId::Sm86,
            MemRole::BypassLoad(CacheOpToken::Atom),
            "ATOMG.E.CAS.STRONG.GPU",
        );
        let mut inst = atomic_inst(Some(pb::MemorySpace::Global));
        inst.addresses.clear();

        assert!(matches!(
            classify_flat(
                &inst,
                &sched,
                sched.arch,
                Path::new("empty_atomic.pb"),
                3,
                5
            ),
            Err(ReplayError::MemoryRoute {
                warp_id: 3,
                record_ix: 5,
                class: pb::MemoryOpClass::Atomic,
                source: MemoryRouteError::AtomicOperandCount { got: 0 },
                ..
            })
        ));

        let mut sectors = vec![31];
        let mut sector_scratch = vec![99];
        assert!(matches!(
            classify_with_l1(
                &inst,
                &sched,
                sched.arch,
                &mut sectors,
                &mut sector_scratch,
                Path::new("empty_atomic.pb"),
                3,
                5,
            ),
            Err(ReplayError::MemoryRoute {
                warp_id: 3,
                record_ix: 5,
                class: pb::MemoryOpClass::Atomic,
                source: MemoryRouteError::AtomicOperandCount { got: 0 },
                ..
            })
        ));
        assert_eq!(sectors, vec![31]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn multiple_atomic_operands_are_typed_rejected_in_both_modes() {
        let sched = schedule(
            ArchId::Sm86,
            MemRole::BypassLoad(CacheOpToken::Atom),
            "ATOMG.E.CAS.STRONG.GPU",
        );
        let mut inst = atomic_inst(Some(pb::MemorySpace::Global));
        inst.addresses
            .push(spaced_address(pb::MemorySpace::Global, 0x1040));

        assert!(matches!(
            classify_flat(
                &inst,
                &sched,
                sched.arch,
                Path::new("multi_atomic.pb"),
                4,
                6
            ),
            Err(ReplayError::MemoryRoute {
                warp_id: 4,
                record_ix: 6,
                class: pb::MemoryOpClass::Atomic,
                source: MemoryRouteError::AtomicOperandCount { got: 2 },
                ..
            })
        ));

        let mut sectors = vec![37];
        let mut sector_scratch = vec![99];
        assert!(matches!(
            classify_with_l1(
                &inst,
                &sched,
                sched.arch,
                &mut sectors,
                &mut sector_scratch,
                Path::new("multi_atomic.pb"),
                4,
                6,
            ),
            Err(ReplayError::MemoryRoute {
                warp_id: 4,
                record_ix: 6,
                class: pb::MemoryOpClass::Atomic,
                source: MemoryRouteError::AtomicOperandCount { got: 2 },
                ..
            })
        ));
        assert_eq!(sectors, vec![37]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn explicit_access_kind_must_match_replay_class_in_both_modes() {
        let mut global = global_inst();
        global.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Atomic as i32);
        assert_access_kind_rejected_in_both_modes(
            &global,
            &schedule(ArchId::Sm86, MemRole::None, "LD.E R0, [R2.64]"),
            "LOAD or STORE",
        );

        let mut atomic = atomic_inst(Some(pb::MemorySpace::Global));
        atomic.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Load as i32);
        assert_access_kind_rejected_in_both_modes(
            &atomic,
            &schedule(
                ArchId::Sm86,
                MemRole::BypassLoad(CacheOpToken::Atom),
                "ATOMG.E.CAS.STRONG.GPU",
            ),
            "ATOMIC",
        );

        let mut async_copy = async_copy_inst(vec![spaced_address(pb::MemorySpace::Global, 0x1000)]);
        async_copy.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Atomic as i32);
        assert_access_kind_rejected_in_both_modes(
            &async_copy,
            &schedule(
                ArchId::Sm86,
                MemRole::BypassLoad(CacheOpToken::Ldgsts),
                "LDGSTS.E.128",
            ),
            "ASYNC_COPY",
        );
    }

    #[test]
    fn static_families_reject_class_confusion_without_access_kind() {
        let mut atmg_as_global = global_inst();
        atmg_as_global.memory_op_access_kind = None;
        assert_static_class_rejected_in_both_modes(
            &atmg_as_global,
            &schedule(
                ArchId::Sm86,
                MemRole::BypassLoad(CacheOpToken::Atom),
                "ATOMG.E.ADD.STRONG.GPU",
            ),
            pb::MemoryOpClass::Atomic,
        );

        let mut ldgsts_as_global = global_inst();
        ldgsts_as_global.memory_op_access_kind = None;
        assert_static_class_rejected_in_both_modes(
            &ldgsts_as_global,
            &schedule(
                ArchId::Sm86,
                MemRole::BypassLoad(CacheOpToken::Ldgsts),
                "LDGSTS.E.128 [R1], [R2.64]",
            ),
            pb::MemoryOpClass::AsyncCopy,
        );

        let mut atoms_as_shared = global_inst();
        atoms_as_shared.memory_op_class = Some(pb::MemoryOpClass::Shared as i32);
        atoms_as_shared.memory_op_access_kind = None;
        atoms_as_shared.addresses[0].address_space = Some(pb::MemorySpace::Shared as i32);
        assert_static_class_rejected_in_both_modes(
            &atoms_as_shared,
            &schedule(
                ArchId::Sm120,
                MemRole::BypassLoad(CacheOpToken::Atom),
                "ATOMS.E.ADD",
            ),
            pb::MemoryOpClass::Atomic,
        );

        let mut redg_as_global = global_inst();
        redg_as_global.memory_op_access_kind = None;
        assert_static_class_rejected_in_both_modes(
            &redg_as_global,
            &schedule(
                ArchId::Sm120,
                MemRole::Reduction,
                "REDG.E.ADD.STRONG.GPU [R2.64], R0",
            ),
            pb::MemoryOpClass::Atomic,
        );
    }

    #[test]
    fn explicit_access_kind_must_match_static_direction_in_both_modes() {
        let load_at_store = global_inst();
        assert_static_access_kind_rejected_in_both_modes(
            &load_at_store,
            &schedule(ArchId::Sm86, MemRole::Store, "STG.E [R2.64], R0"),
            pb::MemoryOpAccessKind::Store,
        );

        let mut store_at_load = global_inst();
        store_at_load.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Store as i32);
        assert_static_access_kind_rejected_in_both_modes(
            &store_at_load,
            &schedule(
                ArchId::Sm86,
                MemRole::EligibleLoad(CacheOpToken::LdgECa),
                "LDG.E.CA R0, [R2.64]",
            ),
            pb::MemoryOpAccessKind::Load,
        );

        let mut load_at_shared_store = global_inst();
        load_at_shared_store.memory_op_class = Some(pb::MemoryOpClass::Shared as i32);
        load_at_shared_store.addresses[0].address_space = Some(pb::MemorySpace::Shared as i32);
        assert_static_access_kind_rejected_in_both_modes(
            &load_at_shared_store,
            &schedule(ArchId::Sm86, MemRole::None, "STS [R1], R0"),
            pb::MemoryOpAccessKind::Store,
        );
    }

    #[test]
    fn space_coded_sass_heads_cannot_borrow_another_dynamic_route() {
        let mut shared_atmg = atomic_inst(Some(pb::MemorySpace::Shared));
        shared_atmg.function_unique_id = 1;
        assert_static_route_rejected_in_both_modes(
            &shared_atmg,
            &schedule(
                ArchId::Sm86,
                MemRole::BypassLoad(CacheOpToken::Atom),
                "ATOMG.E.ADD.STRONG.GPU",
            ),
            pb::MemorySpace::Global,
            pb::MemorySpace::Shared,
        );

        let mut global_atoms = atomic_inst(Some(pb::MemorySpace::Global));
        global_atoms.function_unique_id = 1;
        assert_static_route_rejected_in_both_modes(
            &global_atoms,
            &schedule(
                ArchId::Sm86,
                MemRole::BypassLoad(CacheOpToken::Atom),
                "ATOMS.E.ADD",
            ),
            pb::MemorySpace::Shared,
            pb::MemorySpace::Global,
        );

        let mut shared_stg = global_inst();
        shared_stg.memory_op_class = Some(pb::MemoryOpClass::Shared as i32);
        shared_stg.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Store as i32);
        shared_stg.addresses[0].address_space = Some(pb::MemorySpace::Shared as i32);
        assert_static_route_rejected_in_both_modes(
            &shared_stg,
            &schedule(ArchId::Sm86, MemRole::Store, "STG.E [R2.64], R0"),
            pb::MemorySpace::Global,
            pb::MemorySpace::Shared,
        );

        let mut global_sts = global_inst();
        global_sts.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Store as i32);
        assert_static_route_rejected_in_both_modes(
            &global_sts,
            &schedule(ArchId::Sm86, MemRole::None, "STS [R1], R0"),
            pb::MemorySpace::Shared,
            pb::MemorySpace::Global,
        );

        let mut local_ldg = global_inst();
        local_ldg.memory_op_class = Some(pb::MemoryOpClass::Local as i32);
        local_ldg.addresses[0].address_space = Some(pb::MemorySpace::Local as i32);
        assert_static_route_rejected_in_both_modes(
            &local_ldg,
            &schedule(
                ArchId::Sm86,
                MemRole::EligibleLoad(CacheOpToken::LdgECa),
                "LDG.E.CA R0, [R1]",
            ),
            pb::MemorySpace::Global,
            pb::MemorySpace::Local,
        );

        let mut shared_red = atomic_inst(Some(pb::MemorySpace::Shared));
        shared_red.function_unique_id = 1;
        assert_static_route_rejected_in_both_modes(
            &shared_red,
            &schedule(
                ArchId::Sm86,
                MemRole::Reduction,
                "RED.E.ADD.STRONG.GPU [R2.64], R0",
            ),
            pb::MemorySpace::Global,
            pb::MemorySpace::Shared,
        );

        let mut global_reds = atomic_inst(Some(pb::MemorySpace::Global));
        global_reds.function_unique_id = 1;
        assert_static_family_unsupported_in_both_modes(
            &global_reds,
            &schedule(ArchId::Sm86, MemRole::None, "REDS.E.ADD [R1], R0"),
            "REDS",
        );
    }

    #[test]
    fn sm120_dsmem_families_remain_typed_unsupported() {
        let mut redas = atomic_inst(Some(pb::MemorySpace::Global));
        redas.function_unique_id = 1;
        redas.memory_op_class = Some(pb::MemoryOpClass::ClusterDsmem as i32);
        assert_unsupported_memory_path_in_both_modes(
            &redas,
            &schedule(ArchId::Sm120, MemRole::None, "REDAS.E.ADD [R1], R0"),
            pb::MemoryOpClass::ClusterDsmem,
        );

        let mut stas = global_inst();
        stas.memory_op_class = Some(pb::MemoryOpClass::ClusterDsmem as i32);
        stas.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Store as i32);
        assert_unsupported_memory_path_in_both_modes(
            &stas,
            &schedule(ArchId::Sm120, MemRole::None, "STAS [R1], R0"),
            pb::MemoryOpClass::ClusterDsmem,
        );

        let mut lds_mmio = global_inst();
        lds_mmio.memory_op_class = Some(pb::MemoryOpClass::ClusterDsmem as i32);
        assert_unsupported_memory_path_in_both_modes(
            &lds_mmio,
            &schedule(ArchId::Sm120, MemRole::None, "LDS.U.32.MMIO R0, [R1]"),
            pb::MemoryOpClass::ClusterDsmem,
        );
    }

    #[test]
    fn sm120_dsmem_families_reject_modeled_class_contamination() {
        let mut redas_as_atomic = atomic_inst(Some(pb::MemorySpace::Shared));
        redas_as_atomic.function_unique_id = 1;
        assert_static_class_rejected_in_both_modes(
            &redas_as_atomic,
            &schedule(ArchId::Sm120, MemRole::None, "REDAS.E.ADD [R1], R0"),
            pb::MemoryOpClass::ClusterDsmem,
        );

        let mut stas_as_shared = global_inst();
        stas_as_shared.memory_op_class = Some(pb::MemoryOpClass::Shared as i32);
        stas_as_shared.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Store as i32);
        stas_as_shared.addresses[0].address_space = Some(pb::MemorySpace::Shared as i32);
        assert_static_class_rejected_in_both_modes(
            &stas_as_shared,
            &schedule(ArchId::Sm120, MemRole::None, "STAS [R1], R0"),
            pb::MemoryOpClass::ClusterDsmem,
        );

        let mut lds_mmio_as_shared = global_inst();
        lds_mmio_as_shared.memory_op_class = Some(pb::MemoryOpClass::Shared as i32);
        lds_mmio_as_shared.addresses[0].address_space = Some(pb::MemorySpace::Shared as i32);
        assert_static_class_rejected_in_both_modes(
            &lds_mmio_as_shared,
            &schedule(ArchId::Sm120, MemRole::None, "LDS.U.32.MMIO R0, [R1]"),
            pb::MemoryOpClass::ClusterDsmem,
        );
    }

    #[test]
    fn sm86_rejects_post_ampere_static_memory_families() {
        let mut stas = global_inst();
        stas.memory_op_class = Some(pb::MemoryOpClass::Shared as i32);
        stas.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Store as i32);
        stas.addresses[0].address_space = Some(pb::MemorySpace::Shared as i32);
        assert_static_family_unsupported_in_both_modes(
            &stas,
            &schedule(ArchId::Sm86, MemRole::None, "STAS [R1], R0"),
            "STAS",
        );

        let mut redas = atomic_inst(Some(pb::MemorySpace::Shared));
        redas.function_unique_id = 1;
        assert_static_family_unsupported_in_both_modes(
            &redas,
            &schedule(ArchId::Sm86, MemRole::None, "REDAS.E.ADD [R1], R0"),
            "REDAS",
        );

        let mut lds_mmio = global_inst();
        lds_mmio.memory_op_class = Some(pb::MemoryOpClass::Shared as i32);
        lds_mmio.addresses[0].address_space = Some(pb::MemorySpace::Shared as i32);
        assert_static_family_unsupported_in_both_modes(
            &lds_mmio,
            &schedule(ArchId::Sm86, MemRole::None, "LDS.U.32.MMIO R0, [R1]"),
            "LDS.MMIO",
        );

        let mut stsm = global_inst();
        stsm.memory_op_class = Some(pb::MemoryOpClass::Shared as i32);
        stsm.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Store as i32);
        stsm.addresses[0].address_space = Some(pb::MemorySpace::Shared as i32);
        assert_static_family_unsupported_in_both_modes(
            &stsm,
            &schedule(ArchId::Sm86, MemRole::None, "STSM [R1], R0"),
            "STSM",
        );

        let mut redg = atomic_inst(Some(pb::MemorySpace::Global));
        redg.function_unique_id = 1;
        assert_static_family_unsupported_in_both_modes(
            &redg,
            &schedule(
                ArchId::Sm86,
                MemRole::Reduction,
                "REDG.E.ADD.STRONG.GPU [R2.64], R0",
            ),
            "REDG",
        );

        let mut ldgmc_as_shared = global_inst();
        ldgmc_as_shared.memory_op_class = Some(pb::MemoryOpClass::Shared as i32);
        ldgmc_as_shared.addresses[0].address_space = Some(pb::MemorySpace::Shared as i32);
        assert_static_family_unsupported_in_both_modes(
            &ldgmc_as_shared,
            &schedule(ArchId::Sm86, MemRole::None, "LDGMC R0, [R2.64]"),
            "LDGMC",
        );
    }

    #[test]
    fn sm120_ldgmc_and_redg_do_not_gain_sm86_hierarchy_semantics() {
        let ldgmc = global_inst();
        assert_static_family_unsupported_in_both_modes(
            &ldgmc,
            &schedule(ArchId::Sm120, MemRole::None, "LDGMC R0, [R2.64]"),
            "LDGMC",
        );

        let mut redg = atomic_inst(Some(pb::MemorySpace::Global));
        redg.function_unique_id = 1;
        assert_unsupported_memory_path_in_both_modes(
            &redg,
            &schedule(
                ArchId::Sm120,
                MemRole::Reduction,
                "REDG.E.ADD.STRONG.GPU [R2.64], R0",
            ),
            pb::MemoryOpClass::Atomic,
        );
    }

    #[test]
    fn sm120_stsm_uses_shared_timing_only_on_its_supported_architecture() {
        let mut inst = global_inst();
        inst.memory_op_class = Some(pb::MemoryOpClass::Shared as i32);
        inst.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Store as i32);
        inst.addresses[0].address_space = Some(pb::MemorySpace::Shared as i32);
        let sched = schedule(ArchId::Sm120, MemRole::None, "STSM [R1], R0");

        assert!(matches!(
            classify_flat(&inst, &sched, sched.arch, Path::new("stsm.pb"), 0, 0),
            Ok(AccessKind::Shared { .. })
        ));
        let mut sectors = vec![16];
        let mut sector_scratch = vec![105];
        assert!(matches!(
            classify_with_l1(
                &inst,
                &sched,
                sched.arch,
                &mut sectors,
                &mut sector_scratch,
                Path::new("stsm.pb"),
                0,
                0,
            ),
            Ok(AccessKind::Shared { .. })
        ));
        assert_eq!(sectors, vec![16]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn absent_legacy_access_kind_remains_admissible() {
        let sched = schedule(
            ArchId::Sm86,
            MemRole::BypassLoad(CacheOpToken::Ldgsts),
            "LDGSTS.E.128",
        );
        let mut inst = async_copy_inst(vec![spaced_address(pb::MemorySpace::Global, 0x1000)]);
        inst.memory_op_access_kind = None;

        assert!(matches!(
            classify_flat(&inst, &sched, sched.arch, Path::new("legacy_kind.pb"), 0, 0),
            Ok(AccessKind::GlobalAsyncCopy {
                off: 0,
                len: 0,
                completion_work: 4,
                l1_policy: LdgstsL1Policy::CacheAtAllLevels,
            })
        ));
        let mut sectors = vec![11];
        let mut sector_scratch = Vec::new();
        assert!(matches!(
            classify_with_l1(
                &inst,
                &sched,
                sched.arch,
                &mut sectors,
                &mut sector_scratch,
                Path::new("legacy_kind.pb"),
                0,
                0,
            ),
            Ok(AccessKind::GlobalAsyncCopy {
                off: 1,
                len: 1,
                completion_work: 4,
                l1_policy: LdgstsL1Policy::CacheAtAllLevels,
            })
        ));
        assert_eq!(sectors, vec![11, 0x1000 >> 5]);

        let store_sched = schedule(ArchId::Sm86, MemRole::Store, "STG.E [R2.64], R0");
        let mut legacy_global_store = global_inst();
        legacy_global_store.memory_op_access_kind = None;
        assert!(matches!(
            classify_flat(
                &legacy_global_store,
                &store_sched,
                store_sched.arch,
                Path::new("legacy_global_store.pb"),
                0,
                0,
            ),
            Ok(AccessKind::GlobalStore { off: 0, len: 0 })
        ));
        let mut store_sectors = vec![12];
        let mut store_scratch = Vec::new();
        assert!(matches!(
            classify_with_l1(
                &legacy_global_store,
                &store_sched,
                store_sched.arch,
                &mut store_sectors,
                &mut store_scratch,
                Path::new("legacy_global_store.pb"),
                0,
                0,
            ),
            Ok(AccessKind::GlobalStore { off: 1, len: 1 })
        ));
        assert_eq!(store_sectors, vec![12, 0x1000 >> 5]);
    }

    #[test]
    fn capability_empty_legacy_routes_skip_shape_but_validate_explicit_space() {
        let atomic_sched = schedule(
            ArchId::Sm86,
            MemRole::BypassLoad(CacheOpToken::Atom),
            "ATOMG.E.ADD.STRONG.GPU",
        );
        let mut addressed_atomic = atomic_inst(None);
        addressed_atomic.function_unique_id = 1;
        assert!(matches!(
            classify_flat_with_contract(
                &addressed_atomic,
                &atomic_sched,
                atomic_sched.arch,
                MemoryPayloadContract::LegacyExclusion,
                Path::new("legacy_atomic.pb"),
                0,
                0,
            ),
            Ok(AccessKind::GlobalBypass {
                off: 0,
                len: 0,
                token: CacheOpToken::Atom,
                l2_requests: 0,
            })
        ));
        let mut atomic_sectors = vec![20];
        let mut atomic_scratch = vec![99];
        assert!(matches!(
            classify_with_l1_with_contract(
                &addressed_atomic,
                &atomic_sched,
                atomic_sched.arch,
                MemoryPayloadContract::LegacyExclusion,
                &mut atomic_sectors,
                &mut atomic_scratch,
                Path::new("legacy_atomic.pb"),
                0,
                0,
            ),
            Ok(AccessKind::GlobalBypass {
                off: 1,
                len: 1,
                token: CacheOpToken::Atom,
                l2_requests: 1,
            })
        ));
        assert_eq!(atomic_sectors, vec![20, 0x1000 >> 5]);

        let mut addressless_atomic = addressed_atomic.clone();
        addressless_atomic.addresses.clear();
        assert!(
            classify_flat_with_contract(
                &addressless_atomic,
                &atomic_sched,
                atomic_sched.arch,
                MemoryPayloadContract::LegacyExclusion,
                Path::new("legacy_atomic_empty.pb"),
                0,
                0,
            )
            .is_ok()
        );

        let async_sched = schedule(
            ArchId::Sm86,
            MemRole::BypassLoad(CacheOpToken::Ldgsts),
            "LDGSTS.E.128 [R1], [R2.64]",
        );
        let addressless_async = async_copy_inst(Vec::new());
        assert!(
            classify_flat_with_contract(
                &addressless_async,
                &async_sched,
                async_sched.arch,
                MemoryPayloadContract::LegacyExclusion,
                Path::new("legacy_async_empty.pb"),
                0,
                0,
            )
            .is_ok()
        );
        let mut missing_space_async =
            async_copy_inst(vec![spaced_address(pb::MemorySpace::Global, 0x1000)]);
        missing_space_async.addresses[0].address_space = None;
        assert!(
            classify_flat_with_contract(
                &missing_space_async,
                &async_sched,
                async_sched.arch,
                MemoryPayloadContract::LegacyExclusion,
                Path::new("legacy_async_missing_space.pb"),
                0,
                0,
            )
            .is_ok()
        );

        let explicit_shared_async =
            async_copy_inst(vec![spaced_address(pb::MemorySpace::Shared, 0x40)]);
        assert!(matches!(
            classify_flat_with_contract(
                &explicit_shared_async,
                &async_sched,
                async_sched.arch,
                MemoryPayloadContract::LegacyExclusion,
                Path::new("legacy_async_shared.pb"),
                0,
                0,
            ),
            Err(ReplayError::MemoryRoute {
                class: pb::MemoryOpClass::AsyncCopy,
                source: MemoryRouteError::AsyncCopySourceSpaceInvalid {
                    space: pb::MemorySpace::Shared,
                },
                ..
            })
        ));

        let mut async_sectors = vec![21];
        let mut async_scratch = vec![99];
        assert!(matches!(
            classify_with_l1_with_contract(
                &addressless_async,
                &async_sched,
                async_sched.arch,
                MemoryPayloadContract::LegacyExclusion,
                &mut async_sectors,
                &mut async_scratch,
                Path::new("legacy_async_empty.pb"),
                0,
                0,
            ),
            Ok(AccessKind::GlobalAsyncCopy {
                off: 1,
                len: 0,
                completion_work: 0,
                l1_policy: LdgstsL1Policy::CacheAtAllLevels,
            })
        ));
        assert_eq!(async_sectors, vec![21]);
        assert!(async_scratch.is_empty());
    }

    #[test]
    fn async_copy_completion_work_narrowing_is_typed_and_precedes_arena_mutation() {
        let path = Path::new("completion_work_overflow.pb");
        assert_eq!(
            project_async_copy_completion_work(
                ArchId::Sm86,
                LdgstsL1Policy::CacheAtAllLevels,
                u64::from(ASYNC_COPY_COMPLETION_VALUE_MASK),
                None,
                path,
                3,
                7,
            )
            .unwrap(),
            ASYNC_COPY_COMPLETION_VALUE_MASK
        );

        let mut sectors = vec![11];
        let sector_scratch = vec![29];
        let payload_bytes = u64::from(ASYNC_COPY_COMPLETION_VALUE_MASK) + 1;
        let error = append_staged_async_copy_sectors(
            ArchId::Sm86,
            LdgstsL1Policy::CacheAtAllLevels,
            payload_bytes,
            None,
            None,
            None,
            &mut sectors,
            &mut AsyncCopyCohortReplay::default(),
            &sector_scratch,
            path,
            3,
            7,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ReplayError::AsyncCopyCompletionWorkOutOfRange {
                warp_id: 3,
                record_ix: 7,
                completion_work,
                max_completion_work: ASYNC_COPY_COMPLETION_VALUE_MASK,
                ..
            } if completion_work == payload_bytes
        ));
        assert_eq!(sectors, vec![11]);
    }

    #[test]
    fn async_copy_cohort_projection_enters_one_sparse_replay_sidecar_transactionally() {
        let path = Path::new("cohort_projection.pb");
        let topology = LdgstsDestinationTopology {
            source_cohorts: 2,
            destination_line_fanout_total: 3,
            destination_line_fanout_max: 2,
            cache_all_bank_wavefronts_total: 3,
            cache_all_bank_wavefronts_max: 2,
            bypass_wavefronts_total: 4,
            bypass_wavefronts_max: 3,
        };
        let cohorts = [
            LdgstsSourceCohort {
                payload_bytes: 32,
                destination_lines: 1,
                bank_wavefronts: 1,
                bypass_wavefronts: 1,
            },
            LdgstsSourceCohort {
                payload_bytes: 32,
                destination_lines: 2,
                bank_wavefronts: 2,
                bypass_wavefronts: 3,
            },
        ];
        let masks = [0b01, 0b11, 0b10];
        let mut sectors = vec![7];
        let mut sidecar = AsyncCopyCohortReplay::default();
        let kind = append_staged_async_copy_sectors(
            ArchId::Sm86,
            LdgstsL1Policy::BypassL1,
            64,
            Some(topology),
            Some(&cohorts),
            Some(&masks),
            &mut sectors,
            &mut sidecar,
            &[40, 41, 42],
            path,
            3,
            7,
        )
        .unwrap();
        assert!(matches!(
            kind,
            AccessKind::GlobalAsyncCopy {
                off: 1,
                len: 3,
                l1_policy: LdgstsL1Policy::BypassL1,
                ..
            }
        ));
        assert_eq!(sectors, [7, 40, 41, 42]);
        assert_eq!(sidecar.spans.len(), 1);
        let span = sidecar.spans[0];
        assert_eq!(span.instruction, 7);
        assert_eq!(span.sector_off, 1);
        assert_eq!(span.sector_len, 3);
        assert_eq!(sidecar.sector_membership_masks, masks);
        assert_eq!(
            sidecar.cohorts,
            cohorts.map(|cohort| AsyncCopyCohortDescriptor {
                payload_bytes: cohort.payload_bytes,
                destination_lines: cohort.destination_lines,
                bank_wavefronts: cohort.bank_wavefronts,
                bypass_wavefronts: cohort.bypass_wavefronts,
            })
        );

        let before_sectors = sectors.clone();
        let before_spans = sidecar.spans.clone();
        let error = append_staged_async_copy_sectors(
            ArchId::Sm86,
            LdgstsL1Policy::BypassL1,
            64,
            Some(topology),
            Some(&cohorts),
            Some(&masks[..2]),
            &mut sectors,
            &mut sidecar,
            &[50, 51, 52],
            path,
            3,
            8,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ReplayError::AsyncCopyCohortProjectionMismatch {
                field: "source-sector membership count",
                ..
            }
        ));
        assert_eq!(sectors, before_sectors);
        assert_eq!(sidecar.spans, before_spans);
        assert_eq!(sidecar.sector_membership_masks, masks);
    }

    #[test]
    fn zero_fill_async_copy_accepts_empty_source_cohort_projection() {
        let mut sectors = vec![7];
        let mut sidecar = AsyncCopyCohortReplay::default();
        let destination_topology = LdgstsDestinationTopology {
            destination_line_fanout_total: 4,
            destination_line_fanout_max: 4,
            cache_all_bank_wavefronts_total: 4,
            cache_all_bank_wavefronts_max: 4,
            bypass_wavefronts_total: 4,
            bypass_wavefronts_max: 4,
            ..LdgstsDestinationTopology::default()
        };
        let kind = append_staged_async_copy_sectors(
            ArchId::Sm86,
            LdgstsL1Policy::CacheAtAllLevels,
            0,
            Some(destination_topology),
            Some(&[]),
            Some(&[]),
            &mut sectors,
            &mut sidecar,
            &[],
            Path::new("zero_fill.pb"),
            0,
            369,
        )
        .expect("zero-fill carries no global source cohort");
        assert!(matches!(
            kind,
            AccessKind::GlobalAsyncCopy {
                off: 1,
                len: 0,
                completion_work,
                l1_policy: LdgstsL1Policy::CacheAtAllLevels,
            } if completion_work > 0
        ));
        assert_eq!(sectors, [7]);
        assert!(sidecar.spans.is_empty());
        assert!(sidecar.cohorts.is_empty());
        assert!(sidecar.sector_membership_masks.is_empty());

        let mut bypass_sectors = vec![7];
        let mut bypass_sidecar = AsyncCopyCohortReplay::default();
        let bypass_kind = append_staged_async_copy_sectors(
            ArchId::Sm86,
            LdgstsL1Policy::BypassL1,
            0,
            Some(destination_topology),
            Some(&[]),
            Some(&[]),
            &mut bypass_sectors,
            &mut bypass_sidecar,
            &[],
            Path::new("zero_fill_bypass.pb"),
            0,
            370,
        )
        .expect("bypass zero-fill carries a topology-only completion token");
        assert!(matches!(
            bypass_kind,
            AccessKind::GlobalAsyncCopy {
                off: 1,
                len: 0,
                completion_work,
                l1_policy: LdgstsL1Policy::BypassL1,
            } if matches!(
                sim_mem::decode_async_copy_completion(completion_work),
                sim_mem::AsyncCopyCompletion::PayloadAndObservedTopology {
                    payload_work: 0,
                    source_cohorts: 1,
                    destination_lines: 4,
                    max_bank_wavefronts: 4,
                    max_bypass_wavefronts: 4,
                }
            )
        ));
        assert_eq!(bypass_sectors, [7]);
        assert!(bypass_sidecar.spans.is_empty());
        assert!(bypass_sidecar.cohorts.is_empty());
        assert!(bypass_sidecar.sector_membership_masks.is_empty());
    }

    #[test]
    fn async_copy_source_only_routes_global_with_exact_sectors() {
        let sched = schedule(
            ArchId::Sm86,
            MemRole::BypassLoad(CacheOpToken::Ldgsts),
            "LDGSTS.E.128",
        );
        let inst = async_copy_inst(vec![spaced_address(pb::MemorySpace::Global, 0x1000)]);

        let flat = classify_flat(&inst, &sched, sched.arch, Path::new("source.pb"), 0, 0)
            .expect("source-only LDGSTS has an implemented flat route");
        assert!(matches!(
            flat,
            AccessKind::GlobalAsyncCopy {
                off: 0,
                len: 0,
                completion_work: 4,
                l1_policy: LdgstsL1Policy::CacheAtAllLevels,
            }
        ));

        let mut sectors = vec![29];
        let mut sector_scratch = vec![99];
        let l1 = classify_with_l1(
            &inst,
            &sched,
            sched.arch,
            &mut sectors,
            &mut sector_scratch,
            Path::new("source.pb"),
            0,
            0,
        )
        .expect("source-only LDGSTS has an implemented L1 route");
        assert!(matches!(
            l1,
            AccessKind::GlobalAsyncCopy {
                off: 1,
                len: 1,
                completion_work: 4,
                l1_policy: LdgstsL1Policy::CacheAtAllLevels,
            }
        ));
        assert_eq!(sectors, vec![29, 0x1000 >> 5]);
        assert_eq!(sector_scratch, vec![0x1000 >> 5]);

        let bypass_sched = schedule(
            ArchId::Sm86,
            MemRole::BypassLoad(CacheOpToken::Ldgsts),
            "LDGSTS.E.BYPASS.128",
        );
        let mut bypass_sectors = Vec::new();
        let mut bypass_scratch = Vec::new();
        assert!(matches!(
            classify_with_l1(
                &inst,
                &bypass_sched,
                bypass_sched.arch,
                &mut bypass_sectors,
                &mut bypass_scratch,
                Path::new("bypass_source.pb"),
                0,
                0,
            ),
            Ok(AccessKind::GlobalAsyncCopy {
                off: 0,
                len: 1,
                completion_work: 8,
                l1_policy: LdgstsL1Policy::BypassL1,
            })
        ));
        assert_eq!(bypass_sectors, vec![0x1000 >> 5]);
    }

    #[test]
    fn zero_active_async_copy_source_becomes_predicated_off_without_arena_append() {
        let sched = schedule(
            ArchId::Sm86,
            MemRole::BypassLoad(CacheOpToken::Ldgsts),
            "LDGSTS.E.128",
        );
        let mut source = spaced_address(pb::MemorySpace::Global, 0x1000);
        source.compression_format = sim_trace::lanes::FORMAT_BASE_DELTA;
        source.addrs.clear();
        let mut inst = async_copy_inst(vec![source]);
        inst.predicate_mask = Some(0);

        let flat = classify_flat(&inst, &sched, sched.arch, Path::new("zero_source.pb"), 0, 0)
            .expect("zero-active source retains the global route");
        assert_eq!(flat, AccessKind::PredicatedOffMemory);

        let mut sectors = vec![31];
        let mut sector_scratch = vec![99];
        let l1 = classify_with_l1(
            &inst,
            &sched,
            sched.arch,
            &mut sectors,
            &mut sector_scratch,
            Path::new("zero_source.pb"),
            0,
            0,
        )
        .expect("zero-active source retains the L1 global route");
        assert_eq!(l1, AccessKind::PredicatedOffMemory);
        assert_eq!(sectors, vec![31]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn async_copy_rejects_shared_or_mixed_operands_in_both_modes() {
        let sched = schedule(
            ArchId::Sm86,
            MemRole::BypassLoad(CacheOpToken::Ldgsts),
            "LDGSTS.E.128",
        );
        let cases = [
            vec![spaced_address(pb::MemorySpace::Shared, 0x2040)],
            vec![
                spaced_address(pb::MemorySpace::Global, 0x1000),
                spaced_address(pb::MemorySpace::Shared, 0x2040),
            ],
            vec![
                spaced_address(pb::MemorySpace::Shared, 0x2040),
                spaced_address(pb::MemorySpace::Global, 0x1000),
            ],
        ];
        for (case, addresses) in cases.into_iter().enumerate() {
            let inst = async_copy_inst(addresses);
            let flat = classify_flat(
                &inst,
                &sched,
                sched.arch,
                Path::new("invalid_source.pb"),
                2,
                case,
            )
            .unwrap_err();
            assert!(
                matches!(
                    flat,
                    ReplayError::MemoryRoute {
                        warp_id: 2,
                        record_ix,
                        class: pb::MemoryOpClass::AsyncCopy,
                        source: MemoryRouteError::AsyncCopySourceSpaceInvalid {
                            space: pb::MemorySpace::Shared,
                        },
                        ..
                    } if case == 0 && record_ix == case
                ) || matches!(
                    flat,
                    ReplayError::MemoryRoute {
                        warp_id: 2,
                        record_ix,
                        class: pb::MemoryOpClass::AsyncCopy,
                        source: MemoryRouteError::AsyncCopySourceCount { got: 2 },
                        ..
                    } if case > 0 && record_ix == case
                )
            );

            let mut sectors = vec![37];
            let mut sector_scratch = vec![99];
            let l1 = classify_with_l1(
                &inst,
                &sched,
                sched.arch,
                &mut sectors,
                &mut sector_scratch,
                Path::new("invalid_source.pb"),
                2,
                case,
            )
            .unwrap_err();
            assert!(
                matches!(
                    l1,
                    ReplayError::MemoryRoute {
                        warp_id: 2,
                        record_ix,
                        class: pb::MemoryOpClass::AsyncCopy,
                        source: MemoryRouteError::AsyncCopySourceSpaceInvalid {
                            space: pb::MemorySpace::Shared,
                        },
                        ..
                    } if case == 0 && record_ix == case
                ) || matches!(
                    l1,
                    ReplayError::MemoryRoute {
                        warp_id: 2,
                        record_ix,
                        class: pb::MemoryOpClass::AsyncCopy,
                        source: MemoryRouteError::AsyncCopySourceCount { got: 2 },
                        ..
                    } if case > 0 && record_ix == case
                )
            );
            assert_eq!(sectors, vec![37]);
            assert!(sector_scratch.is_empty());
        }
    }

    #[test]
    fn shared_atomic_routes_to_shared_in_flat_and_l1_modes() {
        // A shared atomic (op_class=ATOMIC, address_space=SHARED) must NOT
        // reach the global path, including on the template architecture.
        let sched = schedule(
            ArchId::Sm120,
            MemRole::BypassLoad(CacheOpToken::Atom),
            "ATOMS.E.ADD",
        );
        let mut inst = atomic_inst(Some(pb::MemorySpace::Shared));
        inst.function_unique_id = 1;
        let flat = classify_flat(&inst, &sched, sched.arch, Path::new("t"), 0, 0).unwrap();
        assert!(matches!(flat, AccessKind::Shared { .. }));

        let mut sectors = vec![41];
        let mut sector_scratch = vec![99];
        let l1 = classify_with_l1(
            &inst,
            &sched,
            sched.arch,
            &mut sectors,
            &mut sector_scratch,
            Path::new("t"),
            0,
            0,
        )
        .unwrap();
        assert!(matches!(l1, AccessKind::Shared { .. }));
        assert_eq!(sectors, vec![41]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn global_atomic_requires_sm86_sidecar_role_in_both_modes() {
        let sched = schedule(
            ArchId::Sm86,
            MemRole::BypassLoad(CacheOpToken::Atom),
            "ATOMG.E.CAS.STRONG.GPU",
        );
        let mut inst = atomic_inst(Some(pb::MemorySpace::Global));
        inst.function_unique_id = 1;
        let flat = classify_flat(&inst, &sched, sched.arch, Path::new("t"), 0, 0).unwrap();
        assert!(matches!(
            flat,
            AccessKind::GlobalBypass {
                off: 0,
                len: 0,
                token: CacheOpToken::Atom,
                l2_requests: 0,
            }
        ));

        let mut sectors = vec![43];
        let mut sector_scratch = vec![99];
        let l1 = classify_with_l1(
            &inst,
            &sched,
            sched.arch,
            &mut sectors,
            &mut sector_scratch,
            Path::new("t"),
            0,
            0,
        )
        .unwrap();
        assert!(matches!(
            l1,
            AccessKind::GlobalBypass {
                off: 1,
                len: 1,
                token: CacheOpToken::Atom,
                l2_requests: 1,
            }
        ));
        assert_eq!(sectors, vec![43, 0x1000 >> 5]);
        assert_eq!(sector_scratch, vec![0x1000 >> 5]);
    }

    #[test]
    fn global_reduction_remains_distinct_in_both_modes() {
        let sched = schedule(
            ArchId::Sm86,
            MemRole::Reduction,
            "RED.E.ADD.STRONG.GPU [R2.64], R0",
        );
        let mut inst = atomic_inst(Some(pb::MemorySpace::Global));
        inst.function_unique_id = 1;
        let flat = classify_flat(&inst, &sched, sched.arch, Path::new("red.pb"), 0, 0)
            .expect("implemented reduction has a global atomic route");
        assert!(matches!(
            flat,
            AccessKind::GlobalReduction {
                off: 0,
                len: 0,
                l2_requests: 0,
            }
        ));

        let mut sectors = vec![45];
        let mut sector_scratch = vec![99];
        let l1 = classify_with_l1(
            &inst,
            &sched,
            sched.arch,
            &mut sectors,
            &mut sector_scratch,
            Path::new("red.pb"),
            0,
            0,
        )
        .expect("implemented reduction retains its static reduction role");
        assert!(matches!(
            l1,
            AccessKind::GlobalReduction {
                off: 1,
                len: 1,
                l2_requests: 1,
            }
        ));
        assert_eq!(sectors, vec![45, 0x1000 >> 5]);
        assert_eq!(sector_scratch, vec![0x1000 >> 5]);
    }

    #[test]
    fn addressed_atomic_without_space_is_typed_rejected_in_both_modes() {
        let sched = schedule(
            ArchId::Sm86,
            MemRole::BypassLoad(CacheOpToken::Atom),
            "ATOMG.E.CAS.STRONG.GPU",
        );
        let inst = atomic_inst(None);
        assert!(matches!(
            classify_flat(&inst, &sched, sched.arch, Path::new("t"), 0, 0),
            Err(ReplayError::MemoryRoute {
                class: pb::MemoryOpClass::Atomic,
                source: MemoryRouteError::AtomicSpaceMissing { operand: 0 },
                ..
            })
        ));

        let mut sectors = vec![47];
        let mut sector_scratch = vec![99];
        assert!(matches!(
            classify_with_l1(
                &inst,
                &sched,
                sched.arch,
                &mut sectors,
                &mut sector_scratch,
                Path::new("t"),
                0,
                0,
            ),
            Err(ReplayError::MemoryRoute {
                class: pb::MemoryOpClass::Atomic,
                source: MemoryRouteError::AtomicSpaceMissing { operand: 0 },
                ..
            })
        ));
        assert_eq!(sectors, vec![47]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn flat_global_async_copy_requires_sm86_sidecar_role() {
        let inst = async_copy_inst(vec![spaced_address(pb::MemorySpace::Global, 0x1000)]);
        let sched = schedule(
            ArchId::Sm86,
            MemRole::BypassLoad(CacheOpToken::Ldgsts),
            "LDGSTS.E.128",
        );
        let k = classify_flat(&inst, &sched, sched.arch, Path::new("t"), 0, 0).unwrap();
        assert!(matches!(
            k,
            AccessKind::GlobalAsyncCopy {
                off: 0,
                len: 0,
                completion_work: 4,
                l1_policy: LdgstsL1Policy::CacheAtAllLevels,
            }
        ));
    }

    #[test]
    fn template_global_atomic_is_typed_unsupported() {
        let sched = schedule(
            ArchId::Sm120,
            MemRole::BypassLoad(CacheOpToken::Atom),
            "ATOMG.E.CAS.STRONG.GPU",
        );
        let err = classify_flat(
            &atomic_inst(Some(pb::MemorySpace::Global)),
            &sched,
            sched.arch,
            Path::new("t"),
            0,
            0,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ReplayError::UnsupportedMemoryPath {
                arch: ArchId::Sm120,
                class: pb::MemoryOpClass::Atomic,
            }
        ));
    }

    #[test]
    fn every_unmodeled_memory_class_fails_closed() {
        for class in [
            pb::MemoryOpClass::Texture,
            pb::MemoryOpClass::Surface,
            pb::MemoryOpClass::Tma,
            pb::MemoryOpClass::ClusterDsmem,
            pb::MemoryOpClass::Tmem,
        ] {
            let mut inst = atomic_inst(Some(pb::MemorySpace::Global));
            inst.memory_op_class = Some(class as i32);
            let sched = schedule(ArchId::Sm86, MemRole::None, "IADD3");
            let err = classify_flat(&inst, &sched, sched.arch, Path::new("t"), 0, 0).unwrap_err();
            assert!(matches!(
                err,
                ReplayError::UnsupportedMemoryPath {
                    arch: ArchId::Sm86,
                    class: got,
                } if got == class
            ));
        }
    }

    #[test]
    fn unknown_global_static_opcode_fails_closed() {
        let mut inst = atomic_inst(Some(pb::MemorySpace::Global));
        inst.memory_op_class = Some(pb::MemoryOpClass::Global as i32);
        inst.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Load as i32);
        let sched = schedule(ArchId::Sm86, MemRole::None, "IADD3 R0, R1, R2, RZ");
        let err = classify_flat(&inst, &sched, sched.arch, Path::new("t"), 0, 0).unwrap_err();
        assert!(matches!(
            err,
            ReplayError::UnknownGlobalOpcode {
                arch: ArchId::Sm86,
                pc: 0,
                opcode,
            } if opcode == "IADD3"
        ));
    }

    #[test]
    fn known_untagged_load_opcode_remains_denominator_only() {
        let mut inst = atomic_inst(Some(pb::MemorySpace::Global));
        inst.memory_op_class = Some(pb::MemoryOpClass::Global as i32);
        inst.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Load as i32);
        let sched = schedule(ArchId::Sm86, MemRole::None, "LD.E R0, [R2.64]");
        let k = classify_flat(&inst, &sched, sched.arch, Path::new("t"), 0, 0).unwrap();
        assert!(matches!(k, AccessKind::GlobalUntagged { .. }));
    }
}

#[cfg(test)]
mod replay_identity_tests {
    //! The allocation-identity gate is enforced on the path the simulator
    //! actually consumes (extract_per_sm_blocks), not only in a helper: a
    //! malformed GLOBAL-space global-atomic or LDGSTS-source record fails closed
    //! BEFORE replay classification.
    use super::*;
    use crate::control_bits::ControlBits;
    use crate::program::{FunctionSchedule, IssueSlot, OpClass, PipelineClass};
    use prost::Message;
    use sim_trace::reader::{
        ADDRESS_CARRYING_TRACE_CONTRACT_ID, LDGSTS_DESTINATION_TRACE_CONTRACT_ID,
        SHARED_MEMORY_SPLIT_TRACE_CONTRACT_ID,
    };
    use std::collections::BTreeMap as StdMap;

    #[test]
    fn nonunit_cluster_geometry_is_not_admitted_without_a_model() {
        let kernel = pb::Kernel {
            cluster_dim: Some(pb::Dim3d { x: 2, y: 1, z: 1 }),
            ..Default::default()
        };
        assert!(matches!(
            normalized_cluster_dim(&kernel, ArchId::Sm86),
            Err(ReplayError::ClusterGeometryUnsupported {
                arch: ArchId::Sm86,
                cluster: (2, 1, 1),
            })
        ));
    }

    #[test]
    fn split_shared_memory_total_reaches_architecture_local_residency() {
        let tmp = tempfile::tempdir().unwrap();
        write_trace_with(
            tmp.path(),
            bind_test_allocation(malformed_global_inst(pb::MemoryOpClass::Global)),
        );
        let trace_path = tmp.path().join("dynamic_trace.pb");
        let mut trace = pb::Trace::decode(std::fs::read(&trace_path).unwrap().as_slice()).unwrap();
        trace.trace_contract_id = SHARED_MEMORY_SPLIT_TRACE_CONTRACT_ID.to_owned();
        let kernel = &mut trace
            .gpu_device
            .get_mut(&0)
            .unwrap()
            .streams
            .get_mut(&0)
            .unwrap()
            .kernels[0];
        kernel.size_shared_memory = 6_144;
        kernel.shared_memory_static_bytes = Some(2_048);
        kernel.shared_memory_dynamic_bytes = Some(4_096);
        std::fs::write(&trace_path, trace.encode_to_vec()).unwrap();

        let td = TraceDir::open_direct(tmp.path()).unwrap();
        let loc = td.kernels().next().unwrap();
        let extracted = extract_per_sm_blocks(
            &td,
            &loc,
            &FunctionSchedule {
                arch: ArchId::Sm86,
                function_unique_id: 1,
                kernel_name: "k".into(),
                function_name: "f".into(),
                by_pc: StdMap::new(),
            },
            PlacementConfig::from_arch_card(&sm86_test_card(), ArchId::Sm86).unwrap(),
            false,
        )
        .unwrap();
        assert_eq!(extracted.shared_memory_residency.bytes_per_cta(), 6_144);
        assert_eq!(
            extracted.shared_memory_residency.max_bytes_per_sm(),
            102_400
        );
    }

    #[test]
    fn missing_architecture_local_shared_memory_capacity_fails_closed() {
        let path = Path::new("../../arch_profiles/cards/sm_120-laptop.json");
        let card = ArchCard::load(path).unwrap();
        assert!(matches!(
            PlacementConfig::from_arch_card(&card, ArchId::Sm120),
            Err(ReplayError::ArchitectureCardSharedMemoryCapacity {
                arch: ArchId::Sm120,
                source: CalibError::BlockResearchGap {
                    block: "l1_geometry",
                    ..
                },
            })
        ));
    }

    #[test]
    fn complete_ctas_require_exact_block_local_warps_and_physical_slots() {
        assert!(placement_result(64, &[0, 1]).is_ok());
        assert!(matches!(
            placement_result(64, &[0]).unwrap_err(),
            ReplayError::WarpRosterCardinalityMismatch {
                expected: 2,
                actual: 1,
                ..
            }
        ));
        assert!(matches!(
            placement_result_with_roster(64, 0, &[(0, 0, 0), (2, 2, 1)], &[]).unwrap_err(),
            ReplayError::WarpIdOutOfRange {
                warp_id: 2,
                expected_warps: 2,
                ..
            }
        ));
        assert!(matches!(
            placement_result_with_roster(64, 0, &[(-1, -1, 0), (0, 0, 1)], &[]).unwrap_err(),
            ReplayError::WarpIdOutOfRange {
                warp_id: -1,
                expected_warps: 2,
                ..
            }
        ));
        assert!(matches!(
            placement_result_with_roster(32, 0, &[(0, 1, 0)], &[]).unwrap_err(),
            ReplayError::Trace(TraceError::WarpMapKeyMismatch {
                map_key: 0,
                warp_id: 1,
                ..
            })
        ));
        assert!(matches!(
            placement_result(64, &[0, 0]).unwrap_err(),
            ReplayError::DuplicateWarpSlot { warp_slot: 0, .. }
        ));
        assert!(matches!(
            placement_result_with_empty_warps(64, &[0, 1], &[0]).unwrap_err(),
            ReplayError::WarpSlotMissing { warp_id: 0, .. }
        ));
        assert!(matches!(
            placement_result(32, &[-1]).unwrap_err(),
            ReplayError::WarpSlotOutOfRange {
                warp_slot: -1,
                max_resident_warps: 48,
                ..
            }
        ));
        assert!(placement_result(32, &[47]).is_ok());
        assert!(matches!(
            placement_result(32, &[48]).unwrap_err(),
            ReplayError::WarpSlotOutOfRange {
                warp_slot: 48,
                max_resident_warps: 48,
                ..
            }
        ));
        let oversized_slots = (0..49).collect::<Vec<i32>>();
        assert!(matches!(
            placement_result(1_568, &oversized_slots).unwrap_err(),
            ReplayError::WarpSlotOutOfRange {
                warp_id: 48,
                warp_slot: 48,
                max_resident_warps: 48,
                ..
            }
        ));
    }

    #[test]
    fn card_bounds_traced_and_declared_sm_placement() {
        assert!(placement_result_on_sm(83).is_ok());
        assert!(matches!(
            placement_result_on_sm(84).unwrap_err(),
            ReplayError::SmIdOutOfRange {
                sm_id: 84,
                arch: ArchId::Sm86,
                sku: Sku::Rtx3090Ti,
                sm_count: 84,
                ..
            }
        ));
        assert!(matches!(
            placement_result_on_sm(-1).unwrap_err(),
            ReplayError::SmIdOutOfRange {
                sm_id: -1,
                arch: ArchId::Sm86,
                sku: Sku::Rtx3090Ti,
                sm_count: 84,
                ..
            }
        ));

        #[cfg(feature = "campaign")]
        {
            // Launch 1 is absent from the observed projection. Campaign range
            // validation still traverses the complete declared map.
            assert!(matches!(
                validate_declared_sampling_sm_range([(0, 0), (1, 84)], sm86_test_placement())
                    .unwrap_err(),
                ReplayError::SamplingSmIdOutOfRange {
                    launch_index: 1,
                    sm_id: 84,
                    arch: ArchId::Sm86,
                    sku: Sku::Rtx3090Ti,
                    sm_count: 84,
                }
            ));
        }
    }

    #[test]
    fn architecture_card_must_match_the_replay_schedule() {
        assert!(matches!(
            PlacementConfig::from_arch_card(&sm86_test_card(), ArchId::Sm120),
            Err(ReplayError::ArchitectureMismatch {
                component: "architecture card",
                expected: ArchId::Sm120,
                got: ArchId::Sm86,
            })
        ));
    }

    #[test]
    fn instruction_warp_identity_is_present_and_matches_its_enclosing_warp() {
        let mut missing = malformed_global_inst(pb::MemoryOpClass::Global);
        missing.warpid_tb = None;
        assert!(matches!(
            replay_error(missing),
            ReplayError::Trace(TraceError::WarpidTbMissing { .. })
        ));

        let mut mismatched = bind_test_allocation(malformed_global_inst(pb::MemoryOpClass::Global));
        mismatched.warpid_tb = Some(1);
        assert!(matches!(
            replay_error(mismatched),
            ReplayError::Trace(TraceError::WarpidTbMismatch {
                expected: 0,
                actual: 1,
                ..
            })
        ));
    }

    /// A GLOBAL-space memory instruction of `op_class` with NO allocation_id —
    /// malformed under the address-carrying contract's identity rule.
    fn malformed_global_inst(op_class: pb::MemoryOpClass) -> pb::Instruction {
        let mut addrs = vec![0x7000_0000i64];
        addrs.extend(std::iter::repeat(LANE_INACTIVE_SENTINEL).take(31));
        let access_kind = match op_class {
            pb::MemoryOpClass::Atomic => pb::MemoryOpAccessKind::Atomic,
            pb::MemoryOpClass::AsyncCopy => pb::MemoryOpAccessKind::AsyncCopy,
            _ => pb::MemoryOpAccessKind::Load,
        };
        pb::Instruction {
            pc: 16,
            active_mask: Some(0b1),
            predicate_mask: Some(0b1),
            function_unique_id: 1,
            vpc: Some(16),
            sm_id: Some(0),
            warpid_sm: Some(0),
            warpid_tb: Some(0),
            data_width: Some(4),
            memory_op_class: Some(op_class as i32),
            memory_op_access_kind: Some(access_kind as i32),
            addresses: vec![pb::Address {
                data_width: 4,
                compression_format: 0, // list_all
                addrs,
                allocation_id: None, // malformed: GLOBAL active record, no id
                address_space: Some(pb::MemorySpace::Global as i32),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn bind_test_allocation(mut inst: pb::Instruction) -> pb::Instruction {
        for address in &mut inst.addresses {
            address.allocation_id = Some(1);
        }
        inst
    }

    fn destination_async_copy_inst(transpose_source_pairing: bool) -> pb::Instruction {
        let source = (0_i64..32)
            .map(|lane| {
                let source_index = if transpose_source_pairing {
                    (lane % 4) * 8 + lane / 4
                } else {
                    lane
                };
                0x7000_0000 + source_index * 16
            })
            .collect();
        let destination = (0_i64..32).map(|lane| lane * 16).collect();
        pb::Instruction {
            pc: 16,
            active_mask: Some(u32::MAX),
            predicate_mask: Some(u32::MAX),
            function_unique_id: 1,
            vpc: Some(16),
            sm_id: Some(0),
            warpid_sm: Some(0),
            warpid_tb: Some(0),
            data_width: Some(16),
            memory_op_class: Some(pb::MemoryOpClass::AsyncCopy as i32),
            memory_op_access_kind: Some(pb::MemoryOpAccessKind::AsyncCopy as i32),
            addresses: vec![
                pb::Address {
                    data_width: 16,
                    compression_format: sim_trace::lanes::FORMAT_LIST_ALL,
                    addrs: source,
                    allocation_id: Some(1),
                    address_space: Some(pb::MemorySpace::Global as i32),
                    ..Default::default()
                },
                pb::Address {
                    data_width: 16,
                    compression_format: sim_trace::lanes::FORMAT_LIST_ALL,
                    addrs: destination,
                    allocation_id: None,
                    address_space: Some(pb::MemorySpace::Shared as i32),
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }

    fn destination_async_schedule(l1_policy: LdgstsL1Policy) -> FunctionSchedule {
        FunctionSchedule {
            arch: ArchId::Sm86,
            function_unique_id: 1,
            kernel_name: "k".into(),
            function_name: "f".into(),
            by_pc: StdMap::from([(
                16,
                IssueSlot {
                    control: ControlBits::new(0, false, 7, 7, 0, 0).expect("valid test controls"),
                    op_class: OpClass::Other,
                    async_copy_wait_max_pending_groups: 0,
                    pipeline_class: PipelineClass::DispatchOnly,
                    mem_role: MemRole::BypassLoad(CacheOpToken::Ldgsts),
                    ldgsts_l1_policy: Some(l1_policy),
                    fixed_latency_operands:
                        crate::sass_operands::FixedLatencyOperandStatus::NotApplicable,
                    clocklo_read: false,
                    gpu_membar: false,
                    opcode_text: match l1_policy {
                        LdgstsL1Policy::CacheAtAllLevels => "LDGSTS.E.128 [R1], [R2.64]".into(),
                        LdgstsL1Policy::BypassL1 => "LDGSTS.E.BYPASS.128 [R1], [R2.64]".into(),
                    },
                },
            )]),
        }
    }

    fn write_trace_with(dir: &std::path::Path, inst: pb::Instruction) {
        let kernel = pb::Kernel {
            id: 1,
            name: "k".into(),
            function_unique_id: 1,
            size_shared_memory: 0,
            number_of_registers: 32,
            shared_memory_base_address: 0,
            local_memory_base_address: 0,
            grid_dim: Some(pb::Dim3d { x: 1, y: 1, z: 1 }),
            block_dim: Some(pb::Dim3d { x: 32, y: 1, z: 1 }),
            cluster_dim: Some(pb::Dim3d { x: 1, y: 1, z: 1 }),
            shared_memory_static_bytes: None,
            shared_memory_dynamic_bytes: None,
            shared_memory_carveout_preference_pct: None,
            driver_occupancy_limit_blocks: None,
        };
        let stream = pb::CudaStream {
            id: 0,
            kernels: vec![kernel],
            ordered_cuda_events: vec![],
        };
        let device = pb::GpuDevice {
            id: 0,
            streams: [(0i64, stream)].into_iter().collect(),
        };
        let trace = pb::Trace {
            name: "m".into(),
            binary_version: 86,
            nvbit_version: "1.8".into(),
            accelsim_version: 0,
            is_gathered_registers_values: false,
            gpu_device: [(0i64, device)].into_iter().collect(),
            trace_contract_id: ADDRESS_CARRYING_TRACE_CONTRACT_ID.into(),
            memory_payload_classes: vec![
                pb::MemoryOpClass::Global as i32,
                pb::MemoryOpClass::Shared as i32,
                pb::MemoryOpClass::Local as i32,
                pb::MemoryOpClass::Atomic as i32,
                pb::MemoryOpClass::AsyncCopy as i32,
            ],
        };
        std::fs::write(dir.join("dynamic_trace.pb"), trace.encode_to_vec()).unwrap();
        // A VALID sidecar, so a missing id is AllocationIdMissing (not
        // AllocationsSidecarMissing) — the record itself is what is malformed.
        let extra = dir.join("extra_info");
        std::fs::create_dir_all(&extra).unwrap();
        std::fs::write(
            extra.join("allocations.json"),
            r#"[{"allocation_id":1,"context_id":1,"device_id":0,"base_address":"0x70000000","size_bytes":4096,"alive":true}]"#,
        )
        .unwrap();
        let tb_dir = dir.join("threadblocks/device_0/stream_0/kernel_1");
        std::fs::create_dir_all(&tb_dir).unwrap();
        let warp = pb::Warp {
            id: 0,
            instructions: vec![inst],
        };
        let tb = pb::Threadblock {
            block_id: Some(pb::Dim3d { x: 0, y: 0, z: 0 }),
            warps: [(0i32, warp)].into_iter().collect(),
        };
        std::fs::write(tb_dir.join("d_0_s_0_k_1_0,0,0.pb"), tb.encode_to_vec()).unwrap();
    }

    fn rewrite_as_legacy_payload(dir: &Path) {
        let root_path = dir.join("dynamic_trace.pb");
        let mut root = pb::Trace::decode(std::fs::read(&root_path).unwrap().as_slice()).unwrap();
        root.trace_contract_id.clear();
        root.memory_payload_classes.clear();
        std::fs::write(root_path, root.encode_to_vec()).unwrap();
    }

    fn rewrite_as_ldgsts_destination_payload(dir: &Path) {
        let root_path = dir.join("dynamic_trace.pb");
        let mut root = pb::Trace::decode(std::fs::read(&root_path).unwrap().as_slice()).unwrap();
        root.trace_contract_id = LDGSTS_DESTINATION_TRACE_CONTRACT_ID.to_owned();
        let kernel = &mut root
            .gpu_device
            .get_mut(&0)
            .unwrap()
            .streams
            .get_mut(&0)
            .unwrap()
            .kernels[0];
        kernel.shared_memory_static_bytes = Some(0);
        kernel.shared_memory_dynamic_bytes = Some(0);
        std::fs::write(root_path, root.encode_to_vec()).unwrap();
    }

    fn rewrite_block_warp_roster(
        dir: &std::path::Path,
        block_x: i32,
        sm_id: i32,
        roster: &[(i32, i32, i32)],
    ) {
        let trace_path = dir.join("dynamic_trace.pb");
        let mut trace = pb::Trace::decode(std::fs::read(&trace_path).unwrap().as_slice()).unwrap();
        trace
            .gpu_device
            .get_mut(&0)
            .unwrap()
            .streams
            .get_mut(&0)
            .unwrap()
            .kernels[0]
            .block_dim = Some(pb::Dim3d {
            x: block_x,
            y: 1,
            z: 1,
        });
        std::fs::write(trace_path, trace.encode_to_vec()).unwrap();

        let warps = roster
            .iter()
            .map(|&(map_key, embedded_id, slot)| {
                let instruction = pb::Instruction {
                    pc: 0,
                    active_mask: Some(0),
                    predicate_mask: Some(0),
                    function_unique_id: 1,
                    sm_id: Some(sm_id),
                    warpid_sm: Some(slot),
                    warpid_tb: Some(map_key),
                    vpc: Some(0),
                    ..Default::default()
                };
                (
                    map_key,
                    pb::Warp {
                        id: embedded_id,
                        instructions: vec![instruction],
                    },
                )
            })
            .collect();
        let block = pb::Threadblock {
            block_id: Some(pb::Dim3d { x: 0, y: 0, z: 0 }),
            warps,
        };
        std::fs::write(
            dir.join("threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_0,0,0.pb"),
            block.encode_to_vec(),
        )
        .unwrap();
    }

    fn placement_result_with_roster(
        block_x: i32,
        sm_id: i32,
        roster: &[(i32, i32, i32)],
        empty_warps: &[i32],
    ) -> Result<ExtractedKernel, ReplayError> {
        let tmp = tempfile::tempdir().unwrap();
        write_trace_with(tmp.path(), malformed_global_inst(pb::MemoryOpClass::Global));
        rewrite_block_warp_roster(tmp.path(), block_x, sm_id, roster);
        if !empty_warps.is_empty() {
            let block_path = tmp
                .path()
                .join("threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_0,0,0.pb");
            let mut block =
                pb::Threadblock::decode(std::fs::read(&block_path).unwrap().as_slice()).unwrap();
            for warp_id in empty_warps {
                block
                    .warps
                    .get_mut(warp_id)
                    .expect("test warp exists")
                    .instructions
                    .clear();
            }
            std::fs::write(block_path, block.encode_to_vec()).unwrap();
        }
        let td = TraceDir::open_direct(tmp.path()).map_err(ReplayError::from)?;
        let loc = td.kernels().next().unwrap();
        extract_per_sm_blocks(
            &td,
            &loc,
            &FunctionSchedule {
                arch: ArchId::Sm86,
                function_unique_id: 1,
                kernel_name: "k".into(),
                function_name: "f".into(),
                by_pc: StdMap::new(),
            },
            sm86_test_placement(),
            false,
        )
    }

    fn placement_result_with_empty_warps(
        block_x: i32,
        slots: &[i32],
        empty_warps: &[i32],
    ) -> Result<ExtractedKernel, ReplayError> {
        let roster = slots
            .iter()
            .enumerate()
            .map(|(index, &slot)| {
                let warp_id = i32::try_from(index).unwrap();
                (warp_id, warp_id, slot)
            })
            .collect::<Vec<_>>();
        placement_result_with_roster(block_x, 0, &roster, empty_warps)
    }

    fn placement_result(block_x: i32, slots: &[i32]) -> Result<ExtractedKernel, ReplayError> {
        placement_result_with_empty_warps(block_x, slots, &[])
    }

    fn placement_result_on_sm(sm_id: i32) -> Result<ExtractedKernel, ReplayError> {
        placement_result_with_roster(32, sm_id, &[(0, 0, 0)], &[])
    }

    fn replay_error_with_schedule(inst: pb::Instruction, sched: FunctionSchedule) -> ReplayError {
        let tmp = tempfile::tempdir().unwrap();
        write_trace_with(tmp.path(), inst);
        let td = match TraceDir::open_direct(tmp.path()) {
            Ok(trace) => trace,
            Err(error) => return error.into(),
        };
        let loc = td.kernels().next().unwrap();
        extract_per_sm_blocks(&td, &loc, &sched, sm86_test_placement(), true).unwrap_err()
    }

    fn replay_error(inst: pb::Instruction) -> ReplayError {
        replay_error_with_schedule(
            inst,
            FunctionSchedule {
                arch: ArchId::Sm86,
                function_unique_id: 1,
                kernel_name: "k".into(),
                function_name: "f".into(),
                by_pc: StdMap::new(),
            },
        )
    }

    fn assert_replay_rejects_identity(inst: pb::Instruction) {
        let err = replay_error(inst);
        assert!(
            matches!(
                err,
                ReplayError::Trace(TraceError::AllocationIdMissing { .. })
            ),
            "expected AllocationIdMissing, got {err:?}"
        );
    }

    #[test]
    fn replay_reports_unknown_space_before_missing_identity() {
        let mut inst = malformed_global_inst(pb::MemoryOpClass::Global);
        inst.addresses[0].address_space = Some(99);
        assert!(matches!(
            replay_error(inst),
            ReplayError::Trace(TraceError::UnknownEnumValue {
                field: "address_space",
                raw: 99,
                ..
            })
        ));
    }

    #[test]
    fn replay_reports_later_lane_error_before_earlier_missing_identity() {
        let mut inst = malformed_global_inst(pb::MemoryOpClass::Global);
        let mut malformed_later = inst.addresses[0].clone();
        malformed_later.addrs.clear();
        inst.addresses.push(malformed_later);
        assert!(matches!(
            replay_error(inst),
            ReplayError::Trace(TraceError::Lane {
                source: sim_trace::lanes::LaneError::ListAllLengthInvalid { got: 0 },
                ..
            })
        ));
    }

    #[test]
    fn root_function_join_precedes_replay_allocation_identity() {
        let mut foreign = bind_test_allocation(malformed_global_inst(pb::MemoryOpClass::Global));
        foreign.function_unique_id = 2;
        assert!(matches!(
            replay_error(foreign),
            ReplayError::Trace(TraceError::FunctionUniqueIdMismatch {
                expected: 1,
                actual: 2,
                ..
            })
        ));

        let mut malformed_route =
            bind_test_allocation(malformed_global_inst(pb::MemoryOpClass::Atomic));
        malformed_route.function_unique_id = 2;
        malformed_route
            .addresses
            .push(malformed_route.addresses[0].clone());
        assert!(matches!(
            replay_error(malformed_route),
            ReplayError::Trace(TraceError::FunctionUniqueIdMismatch {
                expected: 1,
                actual: 2,
                ..
            })
        ));
    }

    #[test]
    fn wire_payload_shape_and_root_space_contract_precede_allocation_identity() {
        let mut multi_atomic =
            bind_test_allocation(malformed_global_inst(pb::MemoryOpClass::Atomic));
        multi_atomic
            .addresses
            .push(multi_atomic.addresses[0].clone());
        assert!(matches!(
            replay_error(multi_atomic),
            ReplayError::Trace(TraceError::MemoryPayloadShapeInvalid {
                class: pb::MemoryOpClass::Atomic,
                got: 2,
                ..
            })
        ));

        let shared_with_global_operand = malformed_global_inst(pb::MemoryOpClass::Shared);
        assert!(matches!(
            replay_error(shared_with_global_operand),
            ReplayError::Trace(TraceError::MemoryOpClassAddressSpaceConflict {
                class: pb::MemoryOpClass::Shared,
                space: pb::MemorySpace::Global,
                ..
            })
        ));
    }

    #[test]
    fn replay_access_kind_mismatch_precedes_allocation_identity() {
        let mut inst = bind_test_allocation(malformed_global_inst(pb::MemoryOpClass::AsyncCopy));
        inst.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Atomic as i32);
        assert!(matches!(
            replay_error(inst),
            ReplayError::MemoryRoute {
                class: pb::MemoryOpClass::AsyncCopy,
                source: MemoryRouteError::AccessKindInvalid {
                    got: pb::MemoryOpAccessKind::Atomic,
                    expected: "ASYNC_COPY",
                },
                ..
            }
        ));
    }

    #[test]
    fn replay_static_kind_mismatch_precedes_allocation_identity() {
        let sched = FunctionSchedule {
            arch: ArchId::Sm86,
            function_unique_id: 1,
            kernel_name: "k".into(),
            function_name: "f".into(),
            by_pc: StdMap::from([(
                16,
                IssueSlot {
                    control: ControlBits::new(0, false, 7, 7, 0, 0).expect("valid test controls"),
                    op_class: OpClass::Other,
                    async_copy_wait_max_pending_groups: 0,
                    pipeline_class: PipelineClass::DispatchOnly,
                    mem_role: MemRole::Store,
                    ldgsts_l1_policy: None,
                    fixed_latency_operands:
                        crate::sass_operands::FixedLatencyOperandStatus::NotApplicable,
                    clocklo_read: false,
                    gpu_membar: false,
                    opcode_text: "STG.E [R2.64], R0".into(),
                },
            )]),
        };
        assert!(matches!(
            replay_error_with_schedule(
                bind_test_allocation(malformed_global_inst(pb::MemoryOpClass::Global)),
                sched,
            ),
            ReplayError::MemoryRoute {
                class: pb::MemoryOpClass::Global,
                source: MemoryRouteError::StaticAccessKindMismatch {
                    got: pb::MemoryOpAccessKind::Load,
                    expected: pb::MemoryOpAccessKind::Store,
                },
                ..
            }
        ));
    }

    #[test]
    fn replay_static_class_mismatch_precedes_allocation_identity_without_access_kind() {
        let sched = FunctionSchedule {
            arch: ArchId::Sm86,
            function_unique_id: 1,
            kernel_name: "k".into(),
            function_name: "f".into(),
            by_pc: StdMap::from([(
                16,
                IssueSlot {
                    control: ControlBits::new(0, false, 7, 7, 0, 0).expect("valid test controls"),
                    op_class: OpClass::Other,
                    async_copy_wait_max_pending_groups: 0,
                    pipeline_class: PipelineClass::DispatchOnly,
                    mem_role: MemRole::Reduction,
                    ldgsts_l1_policy: None,
                    fixed_latency_operands:
                        crate::sass_operands::FixedLatencyOperandStatus::NotApplicable,
                    clocklo_read: false,
                    gpu_membar: false,
                    opcode_text: "RED.E.ADD.STRONG.GPU [R2.64], R0".into(),
                },
            )]),
        };
        let mut inst = bind_test_allocation(malformed_global_inst(pb::MemoryOpClass::Global));
        inst.memory_op_access_kind = None;
        assert!(matches!(
            replay_error_with_schedule(inst, sched),
            ReplayError::MemoryRoute {
                class: pb::MemoryOpClass::Global,
                source: MemoryRouteError::StaticMemoryClassMismatch {
                    expected: pb::MemoryOpClass::Atomic,
                    got: pb::MemoryOpClass::Global,
                },
                ..
            }
        ));
    }

    #[test]
    fn replay_static_route_mismatch_precedes_allocation_identity() {
        let sched = FunctionSchedule {
            arch: ArchId::Sm86,
            function_unique_id: 1,
            kernel_name: "k".into(),
            function_name: "f".into(),
            by_pc: StdMap::from([(
                16,
                IssueSlot {
                    control: ControlBits::new(0, false, 7, 7, 0, 0).expect("valid test controls"),
                    op_class: OpClass::Other,
                    async_copy_wait_max_pending_groups: 0,
                    pipeline_class: PipelineClass::DispatchOnly,
                    mem_role: MemRole::BypassLoad(CacheOpToken::Atom),
                    ldgsts_l1_policy: None,
                    fixed_latency_operands:
                        crate::sass_operands::FixedLatencyOperandStatus::NotApplicable,
                    clocklo_read: false,
                    gpu_membar: false,
                    opcode_text: "ATOMS.E.ADD".into(),
                },
            )]),
        };
        assert!(matches!(
            replay_error_with_schedule(
                bind_test_allocation(malformed_global_inst(pb::MemoryOpClass::Atomic)),
                sched,
            ),
            ReplayError::MemoryRoute {
                class: pb::MemoryOpClass::Atomic,
                source: MemoryRouteError::StaticMemoryRouteMismatch {
                    expected: pb::MemorySpace::Shared,
                    got: pb::MemorySpace::Global,
                },
                ..
            }
        ));
    }

    #[test]
    fn deferred_sector_failure_survives_compact_snapshot() {
        let tmp = tempfile::tempdir().unwrap();
        let mut inst = malformed_global_inst(pb::MemoryOpClass::Local);
        inst.addresses[0].address_space = Some(pb::MemorySpace::Local as i32);
        inst.addresses[0].allocation_id = None;
        inst.addresses[0].addrs[0] = i64::MAX;
        write_trace_with(tmp.path(), inst);

        let td =
            TraceDir::open_direct(tmp.path()).expect("overflow is deferred to hierarchy replay");
        let loc = td.kernels().next().unwrap();
        let error = extract_per_sm_blocks(
            &td,
            &loc,
            &FunctionSchedule {
                arch: ArchId::Sm86,
                function_unique_id: 1,
                kernel_name: "k".into(),
                function_name: "f".into(),
                by_pc: StdMap::from([(
                    16,
                    IssueSlot {
                        control: ControlBits::new(0, false, 7, 7, 0, 0)
                            .expect("valid test controls"),
                        op_class: OpClass::Other,
                        async_copy_wait_max_pending_groups: 0,
                        pipeline_class: PipelineClass::DispatchOnly,
                        mem_role: MemRole::EligibleLoad(CacheOpToken::Ldl),
                        ldgsts_l1_policy: None,
                        fixed_latency_operands:
                            crate::sass_operands::FixedLatencyOperandStatus::NotApplicable,
                        clocklo_read: false,
                        gpu_membar: false,
                        opcode_text: "LDL.E".into(),
                    },
                )]),
            },
            sm86_test_placement(),
            true,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ReplayError::SectorExpansion {
                warp_id: 0,
                record_ix: 0,
                source: SectorExpansionError::AddressOverflow {
                    lane: 0,
                    base: i64::MAX,
                    data_width_bytes: 4,
                },
                ..
            }
        ));
    }

    fn replay_async_copy_tail(
        mut inst: pb::Instruction,
        payload_contract: MemoryPayloadContract,
        l1_policy: LdgstsL1Policy,
    ) -> TailFlags {
        inst.sm_id = Some(0);
        inst.warpid_sm = Some(0);

        let tmp = tempfile::tempdir().unwrap();
        write_trace_with(tmp.path(), inst);
        match payload_contract {
            MemoryPayloadContract::LegacyExclusion => rewrite_as_legacy_payload(tmp.path()),
            MemoryPayloadContract::AddressCarryingLdgstsDestinationV3 => {
                rewrite_as_ldgsts_destination_payload(tmp.path());
            }
            MemoryPayloadContract::AddressCarryingV1 => {}
            MemoryPayloadContract::AddressCarryingSharedMemorySplitV2 => {
                panic!("tail fixture does not emit split-v2 roots")
            }
        }

        let td = TraceDir::open_direct(tmp.path()).expect("async-copy trace opens");
        assert_eq!(td.memory_payload_contract(), payload_contract);
        let loc = td.kernels().next().expect("test kernel exists");
        let schedules = BTreeMap::from([(1, destination_async_schedule(l1_policy))]);
        let params = IssueModelParams::from_legacy_card_latencies_for_arch(
            ArchId::Sm86,
            34.0,
            244.0,
            400.0,
            23.0,
        )
        .expect("valid sm_86 test parameters");
        let outcome = replay_trace_kernel(
            &td,
            &loc,
            &schedules,
            &sm86_test_card(),
            &params,
            ReplayMemMode::Flat,
        )
        .expect("async-copy trace replays");
        match outcome {
            KernelReplayOutcome::Full(result) => {
                assert_eq!(result.records_seen(), 1);
                assert_eq!(result.instructions_issued(), 1);
                result.unmodeled_tails()
            }
            #[cfg(feature = "campaign")]
            KernelReplayOutcome::SampledObserved(_) => {
                panic!("complete test trace must produce full coverage")
            }
        }
    }

    #[test]
    fn destination_v3_projects_cache_all_work_and_observes_bypass_destination() {
        for (transpose, l1_policy, expected_work) in [
            (false, LdgstsL1Policy::CacheAtAllLevels, 512),
            (true, LdgstsL1Policy::CacheAtAllLevels, 2_048),
            (true, LdgstsL1Policy::BypassL1, 1_024),
        ] {
            let tmp = tempfile::tempdir().unwrap();
            write_trace_with(tmp.path(), destination_async_copy_inst(transpose));
            rewrite_as_ldgsts_destination_payload(tmp.path());
            let td = TraceDir::open_direct(tmp.path()).expect("destination-v3 trace opens");
            assert_eq!(
                td.memory_payload_contract(),
                MemoryPayloadContract::AddressCarryingLdgstsDestinationV3
            );
            let loc = td.kernels().next().unwrap();
            let schedule = destination_async_schedule(l1_policy);
            let expected_completion = match l1_policy {
                LdgstsL1Policy::CacheAtAllLevels => {
                    encode_async_copy_destination_work(expected_work).unwrap()
                }
                LdgstsL1Policy::BypassL1 => {
                    encode_async_copy_payload_and_topology(expected_work, 4, 16, 4, 8).unwrap()
                }
            };

            for decode_sectors in [false, true] {
                let extracted = extract_per_sm_blocks(
                    &td,
                    &loc,
                    &schedule,
                    sm86_test_placement(),
                    decode_sectors,
                )
                .expect("destination-bearing async copy extracts");
                let warp = &extracted.per_sm_blocks.get(&0).unwrap()[0].warps[0];
                assert!(matches!(
                    warp.insts[0].kind,
                    AccessKind::GlobalAsyncCopy {
                        completion_work,
                        l1_policy: got_policy,
                        ..
                    } if completion_work == expected_completion && got_policy == l1_policy
                ));
                assert_eq!(warp.sectors.len(), if decode_sectors { 16 } else { 0 });
                if decode_sectors {
                    let cohorts = warp
                        .async_copy_cohorts
                        .as_deref()
                        .expect("destination-v3 extraction retains the cold cohort sidecar");
                    assert_eq!(cohorts.spans.len(), 1);
                    assert_eq!(cohorts.spans[0].instruction, 0);
                    assert_eq!(cohorts.spans[0].sector_len, 16);
                    assert_eq!(cohorts.cohorts.len(), 4);
                    assert_eq!(cohorts.sector_membership_masks.len(), 16);
                } else {
                    assert!(warp.async_copy_cohorts.is_none());
                }
            }
        }

        assert_eq!(
            replay_async_copy_tail(
                destination_async_copy_inst(true),
                MemoryPayloadContract::AddressCarryingLdgstsDestinationV3,
                LdgstsL1Policy::CacheAtAllLevels,
            ),
            TailFlags::default()
        );
        assert_eq!(
            replay_async_copy_tail(
                destination_async_copy_inst(true),
                MemoryPayloadContract::AddressCarryingLdgstsDestinationV3,
                LdgstsL1Policy::BypassL1,
            ),
            TailFlags {
                ldgsts_dst_shared_timing: true,
                ..TailFlags::default()
            }
        );
    }

    #[test]
    fn instrumentation_order_survives_wire_compaction_and_flat_extraction() {
        let tmp = tempfile::tempdir().unwrap();
        let mut instruction = destination_async_copy_inst(false);
        instruction.instrumentation_order = Some(1);
        write_trace_with(tmp.path(), instruction);
        rewrite_as_ldgsts_destination_payload(tmp.path());
        let td = TraceDir::open_direct_compact(tmp.path()).expect("ordered trace opens compactly");
        let loc = td.kernels().next().unwrap();
        let schedule = destination_async_schedule(LdgstsL1Policy::CacheAtAllLevels);
        let extracted = extract_per_sm_blocks(&td, &loc, &schedule, sm86_test_placement(), false)
            .expect("ordered instruction extracts");
        let warp = &extracted.per_sm_blocks[&0][0].warps[0];
        assert_eq!(
            warp.async_copy_cohorts
                .as_deref()
                .expect("order allocates the cold async sidecar")
                .instrumentation_order,
            vec![InstrumentationOrderAnchor {
                instruction: 0,
                order: 1,
            }]
        );
        assert_eq!(
            observe_async_instrumentation_order(&extracted.per_sm_blocks, &schedule).unwrap()[&0],
            AsyncInstrumentationOrderDiagnostics {
                selected_events: 1,
                issue_events: 1,
                protocol_contradictions: 1,
                ..AsyncInstrumentationOrderDiagnostics::default()
            }
        );
    }

    #[test]
    fn source_only_async_copy_retains_folded_destination_visibility_tail() {
        let inst = bind_test_allocation(malformed_global_inst(pb::MemoryOpClass::AsyncCopy));
        let tails = replay_async_copy_tail(
            inst,
            MemoryPayloadContract::AddressCarryingV1,
            LdgstsL1Policy::CacheAtAllLevels,
        );
        assert_eq!(
            tails,
            TailFlags {
                ldgsts_dst_shared_timing: true,
                ..TailFlags::default()
            }
        );
    }

    #[test]
    fn legacy_omitted_async_copy_source_retains_folded_visibility_tail() {
        let mut inst = malformed_global_inst(pb::MemoryOpClass::AsyncCopy);
        inst.addresses.clear();
        let tails = replay_async_copy_tail(
            inst,
            MemoryPayloadContract::LegacyExclusion,
            LdgstsL1Policy::CacheAtAllLevels,
        );
        assert_eq!(
            tails,
            TailFlags {
                ldgsts_dst_shared_timing: true,
                ..TailFlags::default()
            }
        );
    }

    #[test]
    fn legacy_route_failure_survives_compact_snapshot() {
        let tmp = tempfile::tempdir().unwrap();
        let mut inst = malformed_global_inst(pb::MemoryOpClass::Global);
        inst.addresses.clear();
        write_trace_with(tmp.path(), inst);
        rewrite_as_legacy_payload(tmp.path());

        let td =
            TraceDir::open_direct(tmp.path()).expect("legacy snapshot remains structurally valid");
        let loc = td.kernels().next().unwrap();
        let error = extract_per_sm_blocks(
            &td,
            &loc,
            &FunctionSchedule {
                arch: ArchId::Sm86,
                function_unique_id: 1,
                kernel_name: "k".into(),
                function_name: "f".into(),
                by_pc: StdMap::new(),
            },
            sm86_test_placement(),
            false,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ReplayError::MemoryRoute {
                class: pb::MemoryOpClass::Global,
                source: MemoryRouteError::GlobalOperandCount { got: 0 },
                ..
            }
        ));
    }

    #[test]
    fn replay_uses_legacy_payload_capability_for_atomic_fallback() {
        let tmp = tempfile::tempdir().unwrap();
        let mut inst = malformed_global_inst(pb::MemoryOpClass::Atomic);
        inst.addresses[0].address_space = None;
        write_trace_with(tmp.path(), inst);

        rewrite_as_legacy_payload(tmp.path());

        let td = TraceDir::open_direct(tmp.path()).unwrap();
        assert_eq!(
            td.memory_payload_contract(),
            MemoryPayloadContract::LegacyExclusion
        );
        let loc = td.kernels().next().unwrap();
        let sched = FunctionSchedule {
            arch: ArchId::Sm86,
            function_unique_id: 1,
            kernel_name: "k".into(),
            function_name: "f".into(),
            by_pc: StdMap::from([(
                16,
                IssueSlot {
                    control: ControlBits::new(0, false, 7, 7, 0, 0).expect("valid test controls"),
                    op_class: OpClass::Other,
                    async_copy_wait_max_pending_groups: 0,
                    pipeline_class: PipelineClass::DispatchOnly,
                    mem_role: MemRole::BypassLoad(CacheOpToken::Atom),
                    ldgsts_l1_policy: None,
                    fixed_latency_operands:
                        crate::sass_operands::FixedLatencyOperandStatus::NotApplicable,
                    clocklo_read: false,
                    gpu_membar: false,
                    opcode_text: "ATOMG.E.ADD.STRONG.GPU".into(),
                },
            )]),
        };
        let extracted = extract_per_sm_blocks(&td, &loc, &sched, sm86_test_placement(), true)
            .expect("legacy capability retains the pre-transition atomic fallback");
        assert_eq!(extracted.records_seen, 1);
        let warp = &extracted.per_sm_blocks[&0][0].warps[0];
        assert!(matches!(
            warp.insts[0].kind,
            AccessKind::GlobalBypass {
                off: 0,
                len: 1,
                token: CacheOpToken::Atom,
                l2_requests: 1,
            }
        ));
        assert_eq!(warp.sectors, vec![0x7000_0000 >> 5]);
    }

    #[test]
    fn replay_rejects_malformed_global_atomic() {
        assert_replay_rejects_identity(malformed_global_inst(pb::MemoryOpClass::Atomic));
    }

    #[test]
    fn replay_rejects_malformed_ldgsts_source() {
        assert_replay_rejects_identity(malformed_global_inst(pb::MemoryOpClass::AsyncCopy));
    }
}

#[cfg(all(test, feature = "campaign"))]
mod completeness_tests {
    //! Reader/replay completeness fail-closed on the path replay ACTUALLY
    //! consumes: an incomplete trace lacking a trace_sampling.json sidecar is
    //! E_INCOMPLETE_TRACE; a subset trace passes iff the sidecar's original_grid
    //! matches the launch gridDim (never rewritten) and records.captured matches
    //! records_seen. A lossless-full sidecar may be present, but never sanctions
    //! incomplete coverage.
    use super::*;
    use crate::control_bits::ControlBits;
    use crate::issue::IssueResourceSource;
    use crate::program::{FunctionSchedule, IssueSlot, OpClass, PipelineClass};
    use prost::Message;
    use std::collections::{BTreeMap as StdMap, BTreeSet as StdSet};

    #[test]
    fn completeness_grid_overflow_is_typed() {
        let wire_max = i32::MAX as u32;
        assert!(matches!(
            checked_grid_total((wire_max, wire_max, wire_max)),
            Err(ReplayError::GridSizeOverflow { grid }) if grid == (wire_max, wire_max, wire_max)
        ));
    }

    /// A minimal non-memory record (no memory_op_class) carrying placement, so it
    /// increments records_seen without engaging the allocation-identity gate.
    fn simple_inst() -> pb::Instruction {
        pb::Instruction {
            pc: 16,
            active_mask: Some(0b1),
            predicate_mask: Some(0b1),
            function_unique_id: 1,
            vpc: Some(16),
            sm_id: Some(0),
            warpid_sm: Some(0),
            warpid_tb: Some(0),
            ..Default::default()
        }
    }

    fn encoded_threadblock(bx: i32) -> Vec<u8> {
        let warp = pb::Warp {
            id: 0,
            instructions: vec![simple_inst()],
        };
        pb::Threadblock {
            block_id: Some(pb::Dim3d { x: bx, y: 0, z: 0 }),
            warps: [(0i32, warp)].into_iter().collect(),
        }
        .encode_to_vec()
    }

    fn write_grid_trace(
        dir: &std::path::Path,
        grid_x: i32,
        grid_y: i32,
        block_xs: &[i32],
        sidecar: Option<&str>,
    ) {
        let kernel = pb::Kernel {
            id: 1,
            name: "_Z1kv".into(),
            function_unique_id: 1,
            size_shared_memory: 0,
            number_of_registers: 32,
            shared_memory_base_address: 0,
            local_memory_base_address: 0,
            grid_dim: Some(pb::Dim3d {
                x: grid_x,
                y: grid_y,
                z: 1,
            }),
            block_dim: Some(pb::Dim3d { x: 32, y: 1, z: 1 }),
            cluster_dim: Some(pb::Dim3d { x: 1, y: 1, z: 1 }),
            shared_memory_static_bytes: None,
            shared_memory_dynamic_bytes: None,
            shared_memory_carveout_preference_pct: None,
            driver_occupancy_limit_blocks: None,
        };
        let stream = pb::CudaStream {
            id: 0,
            kernels: vec![kernel],
            ordered_cuda_events: vec![],
        };
        let device = pb::GpuDevice {
            id: 0,
            streams: [(0i64, stream)].into_iter().collect(),
        };
        let trace = pb::Trace {
            name: "m".into(),
            binary_version: 86,
            nvbit_version: "1.8".into(),
            accelsim_version: 0,
            is_gathered_registers_values: false,
            gpu_device: [(0i64, device)].into_iter().collect(),
            trace_contract_id: String::new(),
            memory_payload_classes: vec![],
        };
        std::fs::write(dir.join("dynamic_trace.pb"), trace.encode_to_vec()).unwrap();
        let tb_dir = dir.join("threadblocks/device_0/stream_0/kernel_1");
        std::fs::create_dir_all(&tb_dir).unwrap();
        for &bx in block_xs {
            std::fs::write(
                tb_dir.join(format!("d_0_s_0_k_1_{bx},0,0.pb")),
                encoded_threadblock(bx),
            )
            .unwrap();
        }
        if let Some(json) = sidecar {
            std::fs::write(dir.join("trace_sampling.json"), json).unwrap();
        }
    }

    // Literal producer fixtures generated and sealed by Python's sidecar
    // implementation.  Consumer tests therefore exercise the real canonical
    // contract instead of manufacturing a Rust-only approximation.
    const PYTHON_SEALED_SUBSET: &str = r#"{"schema_version":2,"cell_id":"c","kernel":{"mangled_name":"_Z1kv","args_hash":"ab"},"method":"wave_segment_stratified_subset","reason":"byte_budget","shape":{"original_grid":{"x":60,"y":1,"z":1},"original_block":{"x":32,"y":1,"z":1},"captured_grid":{"x":60,"y":1,"z":1},"captured_block":{"x":32,"y":1,"z":1},"reduced_shape_variant":null},"gates":{"coop":false,"wave_count":60,"n_measured_waves":40,"tail_measured_waves":2,"atomic_density":0.0,"hotline_share":0.0,"skew_top1pct_share":0.0,"gini":0.0,"per_sm_block_count_cv":0.0,"n_phases":1,"skew_heavy_block_count":1,"certainty_bytes":100,"max_trace_bytes":2000,"certainty_budget_exceeded":false,"verdict":"subset_admissible"},"wave_model":{"reconstruction":"launch_index","per_sm_resident_blocks":[{"sm_id":0,"resident_blocks":1}],"n_waves":60,"segment_len":3,"ghost_depth":1},"per_sm_slot_map":[{"sm_id":0,"block_ids":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59]}],"rng":{"algo":"pcg64","seed":0},"tracer":{"version":5,"features":{"atomic_addr":true,"ldgsts_addr":true},"stats_pass_version":2},"bytes":{"raw":1000,"compressed":200,"codec":"zstd","level":19,"long":27,"dict_id":null},"records":{"expected_full":6000,"captured":12},"hashes":{"selected_block_ids_sha256":"84b7ed43903cca6302ed15cf78ff507c45e575f16d7d5c12c975c5727edc37cf","decompressed_payload_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","per_block_raw_sha256_manifest":"trace-payload-file-manifest-v1\nthreadblocks/device_0/stream_0/kernel_1/b_0.pb\t1000\taaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n","sidecar_sha256":"12b7fcf51dd0d4277c9268b7eda67bca6dec410d36953098f181ef0f7ce7a907"},"label_pairing":"full_shape_label","segments":[{"id":0,"block_ids":[0,1,2]},{"id":19,"block_ids":[57,58,59]},{"id":1,"block_ids":[3,4,5]},{"id":2,"block_ids":[6,7,8]}],"selected_segment_ids":[0,19,1,2],"certainty_segments":[0,19],"force_include_blocks":[0],"strata":[{"id":0,"phase_id":0,"launch_tercile":0,"segment_ids":[1,2],"N_h":6,"n_h":2,"pi":0.3333333333333333,"sector_stddev":2.0,"neyman_weight":12.0}],"selection_reconciliation":{"status":"fixed_point","attempts":1,"max_attempts":3,"capture_stats_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},"selection_budget":{"max_trace_bytes":2000,"basis":"raw_threadblock_pb","certainty_fraction_numerator":2,"certainty_fraction_denominator":5,"certainty_bytes":100},"selection_auxiliary":{"name":"l2_sector_transactions_32b","sector_bytes":32,"scope":"pass2_all_blocks","expected_full":400,"selected":200},"estimator":{"mode":"weighted_full_shape_estimate","targets":["sectors"],"not_recoverable":["cycles"],"convergence":{"status":"converged"}}}"#;

    const PYTHON_SEALED_LOSSLESS: &str = r#"{"schema_version":2,"cell_id":"c","kernel":{"mangled_name":"_Z1kv","args_hash":"deadbeef"},"method":"lossless_compressed_full","reason":"byte_budget","shape":{"original_grid":{"x":60,"y":1,"z":1},"original_block":{"x":32,"y":1,"z":1},"captured_grid":{"x":60,"y":1,"z":1},"captured_block":{"x":32,"y":1,"z":1},"reduced_shape_variant":null},"gates":{"coop":false,"wave_count":60,"n_measured_waves":40,"tail_measured_waves":2,"atomic_density":0.0,"hotline_share":0.0,"skew_top1pct_share":0.0,"gini":0.0,"per_sm_block_count_cv":0.0,"n_phases":1,"skew_heavy_block_count":1,"certainty_bytes":null,"max_trace_bytes":null,"certainty_budget_exceeded":false,"verdict":"subset_admissible"},"wave_model":{"reconstruction":"launch_index","per_sm_resident_blocks":[{"sm_id":0,"resident_blocks":1}],"n_waves":60,"segment_len":3,"ghost_depth":1},"per_sm_slot_map":[{"sm_id":0,"block_ids":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59]}],"rng":{"algo":"pcg64","seed":0},"tracer":{"version":5,"features":{"atomic_addr":true,"ldgsts_addr":true},"stats_pass_version":2},"bytes":{"raw":1000,"compressed":200,"codec":"zstd","level":19,"long":27,"dict_id":null},"records":{"expected_full":60,"captured":60},"hashes":{"selected_block_ids_sha256":"3f3ffcf42accae1591be118baf57efb9b8d7c78028dc566674c809f210ad796a","decompressed_payload_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","per_block_raw_sha256_manifest":"trace-payload-file-manifest-v1\nthreadblocks/device_0/stream_0/kernel_1/b_0.pb\t1000\taaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n","sidecar_sha256":"3c8a94decd2c56db2c72b25aa9b163fcc9d2a1dec13bf075546378c3ec874869"},"label_pairing":"full_shape_label"}"#;

    const SELECTED_BLOCKS: [i32; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 57, 58, 59];
    // Python's canonical manifest builder hashed all 60 exact Prost fixture
    // payloads. Subset tests filter this independent producer fixture by path.
    const PYTHON_FULL_PAYLOAD_MANIFEST: &str =
        include_str!("../tests/data/python_payload_file_manifest_v1.txt");

    fn payload_manifest_for(block_xs: &[i32]) -> (u64, String) {
        let mut selected: StdSet<_> = block_xs
            .iter()
            .map(|block| {
                format!("threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_{block},0,0.pb")
            })
            .collect();
        assert_eq!(selected.len(), block_xs.len());
        let expected: StdSet<_> = (0..60)
            .map(|block| {
                format!("threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_{block},0,0.pb")
            })
            .collect();
        let mut observed = StdSet::new();
        let mut previous = None;
        let mut total_bytes = 0_u64;
        let mut manifest = "trace-payload-file-manifest-v1\n".to_string();
        assert!(PYTHON_FULL_PAYLOAD_MANIFEST.ends_with('\n'));
        assert!(!PYTHON_FULL_PAYLOAD_MANIFEST.ends_with("\n\n"));
        let mut lines = PYTHON_FULL_PAYLOAD_MANIFEST.lines();
        assert_eq!(lines.next(), Some("trace-payload-file-manifest-v1"));
        for line in lines {
            let fields: Vec<_> = line.split('\t').collect();
            assert_eq!(fields.len(), 3, "malformed Python manifest line {line:?}");
            let path = fields[0];
            let size_text = fields[1];
            let hash = fields[2];
            assert!(previous.is_none_or(|prior| prior < path));
            previous = Some(path);
            let size = size_text.parse::<u64>().unwrap();
            assert_eq!(size.to_string(), size_text);
            assert_eq!(hash.len(), 64);
            assert!(
                hash.bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            );
            assert!(
                observed.insert(path.to_string()),
                "duplicate fixture path {path}"
            );
            if selected.remove(path) {
                total_bytes = total_bytes.checked_add(size).unwrap();
                manifest.push_str(line);
                manifest.push('\n');
            }
        }
        assert_eq!(observed, expected, "fixture manifest path roster drift");
        assert!(selected.is_empty(), "fixture manifest omitted {selected:?}");
        (total_bytes, manifest)
    }

    fn bind_payload_sidecar(
        base: &str,
        block_xs: &[i32],
        aggregate_sha256: &str,
        captured_records: u64,
        sidecar_sha256: &str,
    ) -> String {
        let (raw_bytes, manifest) = payload_manifest_for(block_xs);
        let mut value: serde_json::Value = serde_json::from_str(base).unwrap();
        value["bytes"]["raw"] = serde_json::json!(raw_bytes);
        value["records"]["captured"] = serde_json::json!(captured_records);
        value["hashes"]["decompressed_payload_sha256"] = serde_json::json!(aggregate_sha256);
        value["hashes"]["per_block_raw_sha256_manifest"] = serde_json::json!(manifest);
        value["hashes"]["sidecar_sha256"] = serde_json::json!(sidecar_sha256);
        serde_json::to_string(&value).unwrap()
    }

    fn payload_bound_subset() -> String {
        bind_payload_sidecar(
            PYTHON_SEALED_SUBSET,
            &SELECTED_BLOCKS,
            "df85bddbdf77a2f2a75a54d3769476919fcd062c5308b03b1a389c2d3fb21355",
            12,
            "b39d05df6a87ae14cac824c989fd20a49ec303c2171cb7255ae58ad92602aedb",
        )
    }

    fn payload_bound_lossless_subset() -> String {
        bind_payload_sidecar(
            PYTHON_SEALED_LOSSLESS,
            &SELECTED_BLOCKS,
            "df85bddbdf77a2f2a75a54d3769476919fcd062c5308b03b1a389c2d3fb21355",
            60,
            "1b2a0628d819c73cf585e24731547be88fe5822244ec565a4a9dfffc74bf42e4",
        )
    }

    fn payload_bound_lossless_full() -> String {
        let all_blocks: Vec<_> = (0..60).collect();
        bind_payload_sidecar(
            PYTHON_SEALED_LOSSLESS,
            &all_blocks,
            "22203159880c6b50cea0b625110193825b8c8b311894ddff6c8cc4e7be83e115",
            60,
            "a1df48d770a14a29b0d60c5c7ec55c47eec50cf172f4fd92890caf765a810f01",
        )
    }

    fn declare_all_blocks_on_sm(sidecar: String, sm_id: u64, sidecar_sha256: &str) -> String {
        let mut value: serde_json::Value = serde_json::from_str(&sidecar).unwrap();
        value["wave_model"]["per_sm_resident_blocks"][0]["sm_id"] = serde_json::json!(sm_id);
        value["per_sm_slot_map"][0]["sm_id"] = serde_json::json!(sm_id);
        value["hashes"]["sidecar_sha256"] = serde_json::json!(sidecar_sha256);
        serde_json::to_string(&value).unwrap()
    }

    fn payload_bound_lossless_record_drop() -> String {
        let all_blocks: Vec<_> = (0..60).collect();
        bind_payload_sidecar(
            PYTHON_SEALED_LOSSLESS,
            &all_blocks,
            "22203159880c6b50cea0b625110193825b8c8b311894ddff6c8cc4e7be83e115",
            59,
            "86f0f8505cf9ff54d53520c5733c447f357035a1ee9ff6fdd120674e35df2940",
        )
    }

    fn subset_with_captured_mismatch() -> String {
        bind_payload_sidecar(
            PYTHON_SEALED_SUBSET,
            &SELECTED_BLOCKS,
            "df85bddbdf77a2f2a75a54d3769476919fcd062c5308b03b1a389c2d3fb21355",
            13,
            "99ccbd538cad56ab2e004d98a17fe67497d12dd2c9aa4c5417dc62c775c83cc5",
        )
    }

    fn payload_bound_subset_for_different_blocks() -> String {
        const DIFFERENT_BLOCKS: [i32; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 9, 57, 58, 59];
        bind_payload_sidecar(
            PYTHON_SEALED_SUBSET,
            &DIFFERENT_BLOCKS,
            "aaa5e80fd15d9ccc91163b7495f449d7a684c70857f7f7bdb34b38ceb22595f5",
            12,
            "b75758b4ad0da2e37111ac2973d9205264e0045f75d8812d9182f6ff71ad5b3a",
        )
    }

    fn python_order_sensitive_subset() -> String {
        let (raw_bytes, manifest) = payload_manifest_for(&SELECTED_BLOCKS);
        let manifest = manifest
            .replace(
                "threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_0,0,0.pb\t",
                "threadblocks/a.pb\t",
            )
            .replace(
                "threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_1,0,0.pb\t",
                "threadblocks/a/b.pb\t",
            );
        let mut lines = manifest.lines();
        assert_eq!(lines.next(), Some("trace-payload-file-manifest-v1"));
        assert!(lines.next().unwrap().starts_with("threadblocks/a.pb\t"));
        assert!(lines.next().unwrap().starts_with("threadblocks/a/b.pb\t"));

        let mut value: serde_json::Value = serde_json::from_str(PYTHON_SEALED_SUBSET).unwrap();
        value["bytes"]["raw"] = serde_json::json!(raw_bytes);
        value["records"]["captured"] = serde_json::json!(12);
        value["hashes"]["decompressed_payload_sha256"] =
            serde_json::json!("df85bddbdf77a2f2a75a54d3769476919fcd062c5308b03b1a389c2d3fb21355");
        value["hashes"]["per_block_raw_sha256_manifest"] = serde_json::json!(manifest);
        value["hashes"]["sidecar_sha256"] =
            serde_json::json!("bbe0666a9d25ddd1a2c755cea081722444206449977de7e86cfcc0ac6f9e7b3c");
        serde_json::to_string(&value).unwrap()
    }

    fn minimal_fake_sidecar() -> String {
        format!(
            r#"{{"schema_version":2,"method":"wave_segment_stratified_subset","shape":{{"original_grid":{{"x":60,"y":1,"z":1}}}},"records":{{"captured":12}},"hashes":{{"selected_block_ids_sha256":"{a}","decompressed_payload_sha256":"{a}","sidecar_sha256":"{a}"}},"selection_reconciliation":{{"status":"fixed_point","attempts":1,"max_attempts":3,"capture_stats_sha256":"{a}"}}}}"#,
            a = "a".repeat(64)
        )
    }

    fn rewrite_launch_metadata(
        dir: &std::path::Path,
        kernel_name: Option<&str>,
        block_x: Option<i32>,
    ) {
        let path = dir.join("dynamic_trace.pb");
        let mut trace = pb::Trace::decode(std::fs::read(&path).unwrap().as_slice()).unwrap();
        let kernel = &mut trace
            .gpu_device
            .get_mut(&0)
            .unwrap()
            .streams
            .get_mut(&0)
            .unwrap()
            .kernels[0];
        if let Some(name) = kernel_name {
            kernel.name = name.to_string();
        }
        if let Some(x) = block_x {
            kernel.block_dim.as_mut().unwrap().x = x;
        }
        std::fs::write(path, trace.encode_to_vec()).unwrap();
    }

    fn rewrite_trace_binary_version(dir: &std::path::Path, binary_version: i32) {
        let path = dir.join("dynamic_trace.pb");
        let mut trace = pb::Trace::decode(std::fs::read(&path).unwrap().as_slice()).unwrap();
        trace.binary_version = binary_version;
        std::fs::write(path, trace.encode_to_vec()).unwrap();
    }

    fn extract(dir: &std::path::Path) -> Result<ExtractedKernel, ReplayError> {
        let td = TraceDir::open(dir)?;
        let loc = td.kernels().next().unwrap();
        let sched = FunctionSchedule {
            arch: ArchId::Sm86,
            function_unique_id: 1,
            kernel_name: "_Z1kv".into(),
            function_name: "f".into(),
            by_pc: StdMap::new(),
        };
        extract_per_sm_blocks(&td, &loc, &sched, sm86_test_placement(), true)
    }

    fn run(dir: &std::path::Path) -> Result<(), ReplayError> {
        extract(dir).map(|_| ())
    }

    fn replay_outcome_with(
        dir: &std::path::Path,
        mode: ReplayMemMode,
        candidates: &[MemoryServiceCandidate],
        admission_candidates: &[MemoryAdmissionCandidate],
    ) -> Result<KernelReplayOutcome, ReplayError> {
        replay_outcome_with_options(dir, mode, candidates, admission_candidates, None)
    }

    fn replay_outcome_with_options(
        dir: &std::path::Path,
        mode: ReplayMemMode,
        candidates: &[MemoryServiceCandidate],
        admission_candidates: &[MemoryAdmissionCandidate],
        memory_admission_feedback: Option<MemoryAdmissionCandidate>,
    ) -> Result<KernelReplayOutcome, ReplayError> {
        let td = TraceDir::open(dir)?;
        let loc = td.kernels().next().unwrap();
        let schedule = FunctionSchedule {
            arch: ArchId::Sm86,
            function_unique_id: 1,
            kernel_name: "_Z1kv".into(),
            function_name: "f".into(),
            by_pc: StdMap::from([(
                16,
                IssueSlot {
                    control: ControlBits::new(0, false, 7, 7, 0, 0).unwrap(),
                    op_class: OpClass::Other,
                    async_copy_wait_max_pending_groups: 0,
                    pipeline_class: PipelineClass::DispatchOnly,
                    mem_role: MemRole::None,
                    ldgsts_l1_policy: None,
                    fixed_latency_operands:
                        crate::sass_operands::FixedLatencyOperandStatus::NotApplicable,
                    clocklo_read: false,
                    gpu_membar: false,
                    opcode_text: "NOP".into(),
                },
            )]),
        };
        let params = IssueModelParams {
            arch: ArchId::Sm86,
            resource_source: IssueResourceSource::SyntheticTestFixture,
            instruction_front_end: None,
            memory_issue_pipeline: None,
            scalar_execution_pipeline: None,
            issue_ports: 1,
            dual_issue_supported: false,
            ldgsts_l1_hit_return_service: None,
            constant_memory_return_service: None,
            immediate_constant_cache: None,
            l1_hit_latency: 1,
            l1_miss_latency: 1,
            global_flat_latency: 1,
            shared_write_barrier_latency: 1,
            shared_additional_wavefront_latency: 2,
            non_memory_write_barrier_latency: 1,
            read_barrier_latency: 1,
            ldgsts_l1_mshr_floor: std::num::NonZeroU32::new(1).unwrap(),
            ldgsts_completion_throughput: sim_arch::LdgstsCompletionThroughput {
                cache_all_payload_bytes_per_cycle: 64,
                bypass_payload_bytes_per_cycle: 32,
            },
            ldgsts_cache_all_destination_wavefront_cycles: 2,
            ldgsts_global_to_shared_transport_latency: 0,
            ldgsts_cache_all_destination_service_wavefronts: 2,
            ldgsts_cache_all_destination_service_cycles: 3,
            max_resident_warps_per_sm: 48,
            max_resident_blocks_per_sm: u32::MAX,
        };
        replay_kernel_with_schedule_and_l2_and_candidates(
            &td,
            &loc,
            &schedule,
            &sm86_test_card(),
            &params,
            mode,
            None,
            candidates,
            admission_candidates,
            memory_admission_feedback,
        )
    }

    fn replay_outcome(dir: &std::path::Path) -> Result<KernelReplayOutcome, ReplayError> {
        replay_outcome_with(dir, ReplayMemMode::Flat, &[], &[])
    }

    #[test]
    #[cfg(feature = "campaign")]
    fn unadmitted_diagnostics_erase_the_aggregatable_result_type() {
        let tmp = tempfile::tempdir().unwrap();
        write_grid_trace(tmp.path(), 1, 1, &[0], None);

        let diagnostic =
            UnadmittedReplayDiagnostics::from_outcome(replay_outcome(tmp.path()).unwrap());
        assert_eq!(diagnostic.coverage(), UnadmittedCoverage::Full);
        assert_eq!(diagnostic.blocks_observed(), 1);
        assert_eq!(diagnostic.population_blocks(), 1);
        assert_eq!(diagnostic.instructions_issued(), 1);
        assert_eq!(diagnostic.records_seen(), 1);
        assert_eq!(diagnostic.per_sm_count(), Some(1));
        assert_eq!(diagnostic.diagnostic_last_issue_cycles(), Some(1));
    }

    #[test]
    fn direct_replay_joins_trace_binary_version_to_schedule_architecture() {
        let tmp = tempfile::tempdir().unwrap();
        write_grid_trace(tmp.path(), 1, 1, &[0], None);
        rewrite_trace_binary_version(tmp.path(), 120);

        assert!(matches!(
            replay_outcome(tmp.path()),
            Err(ReplayError::ArchitectureMismatch {
                component: "trace binary_version",
                expected: ArchId::Sm86,
                got: ArchId::Sm120,
            })
        ));
    }

    #[test]
    fn direct_replay_rejects_unknown_and_negative_trace_architectures() {
        let tmp = tempfile::tempdir().unwrap();
        write_grid_trace(tmp.path(), 1, 1, &[0], None);
        rewrite_trace_binary_version(tmp.path(), 999);
        assert!(matches!(
            replay_outcome(tmp.path()),
            Err(ReplayError::TraceArchitectureUnknown {
                binary_version: 999,
                ..
            })
        ));

        rewrite_trace_binary_version(tmp.path(), -1);
        assert!(matches!(
            replay_outcome(tmp.path()),
            Err(ReplayError::TraceArchitectureVersionOutOfRange { binary_version: -1 })
        ));
    }

    #[test]
    fn complete_trace_no_sidecar_ok() {
        let tmp = tempfile::tempdir().unwrap();
        write_grid_trace(tmp.path(), 4, 1, &[0, 1, 2, 3], None);
        let extracted = extract(tmp.path()).unwrap();
        let ExtractedCoverage::Full(coverage) = extracted.coverage else {
            panic!("complete unsampled trace must carry full coverage");
        };
        assert_eq!(coverage.blocks_observed(), 4);
        assert_eq!(coverage.launch().grid(), (4, 1, 1));
        assert_eq!(coverage.launch().block(), Some((32, 1, 1)));
        assert!(matches!(
            coverage.source(),
            FullCoverageSource::UnsampledComplete
        ));

        let outcome = replay_outcome(tmp.path()).unwrap();
        let KernelReplayOutcome::Full(full) = outcome else {
            panic!("complete unsampled trace must return full replay");
        };
        assert_eq!(full.coverage().blocks_observed(), 4);
        assert_eq!(full.configuration().arch(), ArchId::Sm86);
        assert_eq!(full.configuration().memory_mode(), ReplayMemMode::Flat);
        assert_eq!(full.co_residency(), CoResidency::Modeled);
        assert_eq!(full.configuration().co_residency(), CoResidency::Modeled);
        assert_eq!(full.records_seen(), 4);
        assert_eq!(full.instructions_issued(), 4);
    }

    #[test]
    fn complete_trace_without_block_dim_is_rejected_by_root_admission() {
        let tmp = tempfile::tempdir().unwrap();
        write_grid_trace(tmp.path(), 4, 1, &[0, 1, 2, 3], None);
        let path = tmp.path().join("dynamic_trace.pb");
        let mut trace = pb::Trace::decode(std::fs::read(&path).unwrap().as_slice()).unwrap();
        trace
            .gpu_device
            .get_mut(&0)
            .unwrap()
            .streams
            .get_mut(&0)
            .unwrap()
            .kernels[0]
            .block_dim = None;
        std::fs::write(path, trace.encode_to_vec()).unwrap();
        assert!(matches!(
            run(tmp.path()),
            Err(ReplayError::Trace(TraceError::LaunchGeometryInvalid { .. }))
        ));
    }

    #[test]
    fn incomplete_trace_no_sidecar_is_e_incomplete_trace() {
        let tmp = tempfile::tempdir().unwrap();
        write_grid_trace(tmp.path(), 4, 1, &[0, 1], None); // 2/4 blocks, no sidecar
        let err = run(tmp.path()).unwrap_err();
        assert!(
            matches!(
                err,
                ReplayError::IncompleteTrace {
                    seen: 2,
                    grid_total: 4
                }
            ),
            "expected IncompleteTrace, got {err:?}"
        );
    }

    #[test]
    fn subset_with_valid_sidecar_ok() {
        let tmp = tempfile::tempdir().unwrap();
        let sidecar = payload_bound_subset();
        write_grid_trace(tmp.path(), 60, 1, &SELECTED_BLOCKS, Some(&sidecar));
        let extracted = extract(tmp.path()).unwrap();
        let ExtractedCoverage::Sampled(coverage) = extracted.coverage else {
            panic!("subset method must remain sampled");
        };
        assert_eq!(coverage.population_blocks(), 60);
        assert_eq!(coverage.blocks_observed(), 12);
        assert_eq!(coverage.launch().grid(), (60, 1, 1));
        assert_eq!(coverage.provenance().selected_block_ids().len(), 12);

        let outcome = replay_outcome(tmp.path()).unwrap();
        let KernelReplayOutcome::SampledObserved(sampled) = outcome else {
            panic!("sampled extraction must not become a full replay result");
        };
        assert_eq!(sampled.coverage().blocks_observed(), 12);
        assert_eq!(sampled.configuration().memory_mode(), ReplayMemMode::Flat);
        assert_eq!(sampled.configuration().co_residency(), CoResidency::Modeled);
        assert_eq!(sampled.counters().arch(), ArchId::Sm86);
        assert_eq!(sampled.counters().instructions_issued(), 12);
        assert_eq!(sampled.counters().records_seen(), 12);
    }

    #[test]
    fn sampled_replay_rejects_memory_candidates_before_discarding_diagnostics() {
        let tmp = tempfile::tempdir().unwrap();
        let sidecar = payload_bound_subset();
        write_grid_trace(tmp.path(), 60, 1, &SELECTED_BLOCKS, Some(&sidecar));
        let mode = ReplayMemMode::L1SharedL2 {
            l1: sim_mem::L1Config {
                arch: ArchId::Sm86,
                line_capacity: 1,
                mshr_count: Some(1),
                hit_latency: 1,
                miss_latency: 1,
            },
            l2: sim_mem::L2Config {
                arch: ArchId::Sm86,
                capacity_bytes: 128,
                line_bytes: 128,
                sector_bytes: 32,
                num_slices: 1,
                associativity: 1,
                replacement: sim_mem::L2Replacement::Lru,
                slice_hash: sim_mem::SliceHash::LineMod,
            },
            dram: DramBw::Unmodeled { arch: ArchId::Sm86 },
        };
        let candidate = MemoryServiceCandidate {
            candidate_id: 1,
            scope: crate::MemoryServiceCandidateScope::Sm,
            operation: crate::MemoryServiceCandidateOperation::Store,
            ideal_wavefront_units: 1,
            ideal_wavefront_cycles: 1,
            unique_line_units: 0,
            unique_line_cycles: 0,
            completion_latency_cycles: 0,
        };

        assert!(matches!(
            replay_outcome_with(tmp.path(), mode, &[candidate], &[]),
            Err(ReplayError::MemoryServiceCandidatesRequireFullCoverage)
        ));

        let admission = MemoryAdmissionCandidate {
            candidate_id: 2,
            scope: crate::MemoryServiceCandidateScope::Sm,
            operation: crate::MemoryServiceCandidateOperation::Store,
            queue_capacity_instructions: 4,
            admission_latency_cycles: 8,
            l2_hit_sector_units: 1,
            l2_hit_sector_cycles: 1,
            l2_miss_sector_units: 1,
            l2_miss_sector_cycles: 10,
            l2_miss_service_width_sectors: 0,
            membar_waits_for_service_completion: false,
            write_completion_latency_cycles: 0,
            write_completion_request_cycles: 0,
            write_completion_region_cycles: 0,
            write_completion_parallel_lanes: 0,
        };
        assert!(matches!(
            replay_outcome_with(tmp.path(), mode, &[], &[admission]),
            Err(ReplayError::MemoryAdmissionCandidatesRequireFullCoverage)
        ));

        let feedback = MemoryAdmissionCandidate {
            admission_latency_cycles: 1,
            ..admission
        };
        assert!(matches!(
            replay_outcome_with_options(tmp.path(), mode, &[], &[], Some(feedback)),
            Err(ReplayError::MemoryAdmissionCandidatesRequireFullCoverage)
        ));
    }

    #[test]
    fn sampled_sidecar_placement_must_match_selected_payload() {
        let tmp = tempfile::tempdir().unwrap();
        let sidecar = declare_all_blocks_on_sm(
            payload_bound_subset(),
            1,
            "0bc32891a99b42b02f0b58a39a10e7c1c78d961467c9616bca6a2bb0486e2ef2",
        );
        write_grid_trace(tmp.path(), 60, 1, &SELECTED_BLOCKS, Some(&sidecar));
        assert!(matches!(
            run(tmp.path()).unwrap_err(),
            ReplayError::SamplingPlacementMismatch {
                launch_index: 0,
                traced_sm_id: 0,
                declared_sm_id: 1,
            }
        ));
    }

    #[test]
    fn sampled_sidecar_sm_range_comes_from_the_architecture_card() {
        let tmp = tempfile::tempdir().unwrap();
        let sidecar = declare_all_blocks_on_sm(
            payload_bound_subset(),
            84,
            "9e5191abfca5604b14fbad15bc2a8b74abcf0ea143c4e9bfa3b1cc8498837006",
        );
        write_grid_trace(tmp.path(), 60, 1, &SELECTED_BLOCKS, Some(&sidecar));
        assert!(matches!(
            run(tmp.path()).unwrap_err(),
            ReplayError::SamplingSmIdOutOfRange {
                launch_index: 0,
                sm_id: 84,
                arch: ArchId::Sm86,
                sku: Sku::Rtx3090Ti,
                sm_count: 84,
            }
        ));
    }

    #[test]
    fn lossless_full_sidecar_cannot_sanction_incomplete_coverage() {
        let tmp = tempfile::tempdir().unwrap();
        let sidecar = payload_bound_lossless_subset();
        write_grid_trace(tmp.path(), 60, 1, &SELECTED_BLOCKS, Some(&sidecar));
        let err = run(tmp.path()).unwrap_err();
        assert!(
            matches!(
                err,
                ReplayError::IncompleteTrace {
                    seen: 12,
                    grid_total: 60
                }
            ),
            "expected IncompleteTrace, got {err:?}"
        );
    }

    #[test]
    fn complete_trace_with_lossless_full_sidecar_ok() {
        let tmp = tempfile::tempdir().unwrap();
        let all_blocks: Vec<i32> = (0..60).collect();
        let sidecar = payload_bound_lossless_full();
        write_grid_trace(tmp.path(), 60, 1, &all_blocks, Some(&sidecar));
        let extracted = extract(tmp.path()).unwrap();
        let ExtractedCoverage::Full(coverage) = extracted.coverage else {
            panic!("lossless full trace must carry full coverage");
        };
        assert_eq!(coverage.blocks_observed(), 60);
        let FullCoverageSource::LosslessCompressedFull(provenance) = coverage.source() else {
            panic!("lossless sidecar provenance must be retained");
        };
        assert_eq!(
            provenance.sidecar_sha256(),
            "a1df48d770a14a29b0d60c5c7ec55c47eec50cf172f4fd92890caf765a810f01"
        );
    }

    #[test]
    fn lossless_sidecar_placement_must_match_every_payload_block() {
        let tmp = tempfile::tempdir().unwrap();
        let all_blocks: Vec<i32> = (0..60).collect();
        let sidecar = declare_all_blocks_on_sm(
            payload_bound_lossless_full(),
            1,
            "b9197fbacad130e04d2f17c06e11c0a6194d568bfa2e9774d2726916ca1e600b",
        );
        write_grid_trace(tmp.path(), 60, 1, &all_blocks, Some(&sidecar));
        assert!(matches!(
            run(tmp.path()).unwrap_err(),
            ReplayError::SamplingPlacementMismatch {
                launch_index: 0,
                traced_sm_id: 0,
                declared_sm_id: 1,
            }
        ));
    }

    #[test]
    fn freshly_sealed_lossless_record_drop_cannot_mint_full_replay() {
        let tmp = tempfile::tempdir().unwrap();
        let all_blocks: Vec<i32> = (0..60).collect();
        let sidecar = payload_bound_lossless_record_drop();
        write_grid_trace(tmp.path(), 60, 1, &all_blocks, Some(&sidecar));
        let err = run(tmp.path()).unwrap_err();
        let ReplayError::Trace(TraceError::SamplingSidecarMalformed { message, .. }) = err else {
            panic!("expected malformed lossless record coverage, got {err:?}");
        };
        assert!(
            message.contains("must equal records.expected_full"),
            "{message}"
        );
    }

    #[test]
    fn subset_with_griddim_rewrite_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let sidecar = payload_bound_subset();
        write_grid_trace(tmp.path(), 61, 1, &SELECTED_BLOCKS, Some(&sidecar));
        let err = run(tmp.path()).unwrap_err();
        assert!(
            matches!(err, ReplayError::GridDimRewritten { .. }),
            "expected GridDimRewritten, got {err:?}"
        );
    }

    #[test]
    fn subset_with_records_captured_mismatch_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let sidecar = subset_with_captured_mismatch();
        write_grid_trace(tmp.path(), 60, 1, &SELECTED_BLOCKS, Some(&sidecar));
        let err = run(tmp.path()).unwrap_err();
        assert!(
            matches!(
                err,
                ReplayError::RecordsCapturedMismatch {
                    records_seen: 12,
                    captured: 13
                }
            ),
            "expected RecordsCapturedMismatch, got {err:?}"
        );
    }

    #[test]
    fn minimal_fake_subset_is_rejected_on_consumer_path() {
        let tmp = tempfile::tempdir().unwrap();
        let sidecar = minimal_fake_sidecar();
        write_grid_trace(tmp.path(), 60, 1, &SELECTED_BLOCKS, Some(&sidecar));
        let err = run(tmp.path()).unwrap_err();
        assert!(matches!(
            err,
            ReplayError::Trace(TraceError::SamplingSidecarMalformed { .. })
        ));
    }

    #[test]
    fn bad_seal_block_hash_and_duplicate_membership_are_rejected_on_consumer_path() {
        let valid = payload_bound_subset();
        let mut duplicate_membership: serde_json::Value = serde_json::from_str(&valid).unwrap();
        duplicate_membership["segments"][2]["block_ids"][0] = serde_json::json!(2);
        let cases = [
            valid.replace(r#""captured":12"#, r#""captured":13"#),
            valid.replace(
                "84b7ed43903cca6302ed15cf78ff507c45e575f16d7d5c12c975c5727edc37cf",
                &"0".repeat(64),
            ),
            serde_json::to_string(&duplicate_membership).unwrap(),
        ];
        for sidecar in cases {
            let tmp = tempfile::tempdir().unwrap();
            write_grid_trace(tmp.path(), 60, 1, &SELECTED_BLOCKS, Some(&sidecar));
            let err = run(tmp.path()).unwrap_err();
            assert!(matches!(
                err,
                ReplayError::Trace(TraceError::SamplingSidecarMalformed { .. })
            ));
        }
    }

    #[test]
    fn sealed_sidecar_for_different_blocks_cannot_sanction_payload() {
        let tmp = tempfile::tempdir().unwrap();
        let sidecar = payload_bound_subset_for_different_blocks();
        let different_blocks = [0, 1, 2, 3, 4, 5, 6, 7, 9, 57, 58, 59];
        write_grid_trace(tmp.path(), 60, 1, &different_blocks, Some(&sidecar));
        let err = run(tmp.path()).unwrap_err();
        assert!(
            matches!(
                err,
                ReplayError::SelectedBlockMembershipMismatch {
                    seen: 12,
                    declared: 12
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn tampered_payload_is_rejected_before_replay() {
        let tmp = tempfile::tempdir().unwrap();
        let sidecar = payload_bound_subset();
        write_grid_trace(tmp.path(), 60, 1, &SELECTED_BLOCKS, Some(&sidecar));
        let path = tmp
            .path()
            .join("threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_0,0,0.pb");
        let mut raw = std::fs::read(&path).unwrap();
        raw[0] ^= 1;
        std::fs::write(path, raw).unwrap();
        assert!(matches!(
            run(tmp.path()).unwrap_err(),
            ReplayError::Trace(TraceError::SamplingSidecarMalformed { .. })
        ));
    }

    #[test]
    fn lossless_sidecar_rejects_count_preserving_payload_tampering() {
        let tmp = tempfile::tempdir().unwrap();
        let all_blocks: Vec<i32> = (0..60).collect();
        let sidecar = payload_bound_lossless_full();
        write_grid_trace(tmp.path(), 60, 1, &all_blocks, Some(&sidecar));
        let path = tmp
            .path()
            .join("threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_0,0,0.pb");
        let mut raw = std::fs::read(&path).unwrap();
        raw[0] ^= 1;
        std::fs::write(path, raw).unwrap();
        let err = run(tmp.path()).unwrap_err();
        assert!(matches!(
            &err,
            ReplayError::Trace(TraceError::SamplingSidecarMalformed { .. })
        ));
        assert!(err.to_string().contains("per_block_raw_sha256_manifest"));
    }

    #[test]
    fn legacy_aggregate_hash_uses_python_manifest_path_order() {
        let tmp = tempfile::tempdir().unwrap();
        let sidecar = python_order_sensitive_subset();
        write_grid_trace(tmp.path(), 60, 1, &SELECTED_BLOCKS, Some(&sidecar));
        let threadblocks = tmp.path().join("threadblocks");
        let original = threadblocks.join("device_0/stream_0/kernel_1");
        std::fs::rename(
            original.join("d_0_s_0_k_1_0,0,0.pb"),
            threadblocks.join("a.pb"),
        )
        .unwrap();
        std::fs::create_dir(threadblocks.join("a")).unwrap();
        std::fs::rename(
            original.join("d_0_s_0_k_1_1,0,0.pb"),
            threadblocks.join("a/b.pb"),
        )
        .unwrap();

        assert!(matches!(
            TraceDir::open(tmp.path()),
            Err(TraceError::ThreadblockPayloadOrphan { .. })
        ));
    }

    #[test]
    fn unchanged_concatenated_bytes_cannot_hide_file_boundary_repartition() {
        let tmp = tempfile::tempdir().unwrap();
        let sidecar = payload_bound_subset();
        write_grid_trace(tmp.path(), 60, 1, &SELECTED_BLOCKS, Some(&sidecar));
        let directory = tmp.path().join("threadblocks/device_0/stream_0/kernel_1");
        let first = directory.join("d_0_s_0_k_1_0,0,0.pb");
        let second = directory.join("d_0_s_0_k_1_1,0,0.pb");
        let original_first = std::fs::read(&first).unwrap();
        let original_second = std::fs::read(&second).unwrap();
        let original_concatenated: Vec<_> = original_first
            .iter()
            .chain(&original_second)
            .copied()
            .collect();

        // Field 2 is the complete protobuf warp-map entry. Move it across the
        // file boundary without changing concatenated bytes; both resulting
        // messages remain decodable, but block 0 loses its recorded warp.
        let boundary = original_first
            .iter()
            .position(|byte| *byte == 0x12)
            .unwrap();
        let repartitioned_first = original_first[..boundary].to_vec();
        let repartitioned_second: Vec<_> = original_first[boundary..]
            .iter()
            .chain(&original_second)
            .copied()
            .collect();
        assert_eq!(
            original_concatenated,
            repartitioned_first
                .iter()
                .chain(&repartitioned_second)
                .copied()
                .collect::<Vec<_>>()
        );
        assert!(pb::Threadblock::decode(repartitioned_first.as_slice()).is_ok());
        assert!(pb::Threadblock::decode(repartitioned_second.as_slice()).is_ok());
        std::fs::write(first, repartitioned_first).unwrap();
        std::fs::write(second, repartitioned_second).unwrap();

        let error = run(tmp.path()).unwrap_err();
        assert!(matches!(
            &error,
            ReplayError::Trace(TraceError::SamplingSidecarMalformed { .. })
        ));
        assert!(
            error.to_string().contains("per_block_raw_sha256_manifest"),
            "{error}"
        );
    }

    #[test]
    fn opened_sidecar_trace_reads_the_verified_payload_snapshot() {
        let tmp = tempfile::tempdir().unwrap();
        let sidecar = payload_bound_subset();
        write_grid_trace(tmp.path(), 60, 1, &SELECTED_BLOCKS, Some(&sidecar));
        let td = TraceDir::open(tmp.path()).unwrap();

        let path = tmp
            .path()
            .join("threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_0,0,0.pb");
        let original = std::fs::read(&path).unwrap();
        let mut replacement = pb::Threadblock::decode(original.as_slice()).unwrap();
        replacement
            .warps
            .get_mut(&0)
            .unwrap()
            .instructions
            .get_mut(0)
            .unwrap()
            .pc = 17;
        let replacement = replacement.encode_to_vec();
        assert_eq!(replacement.len(), original.len());
        std::fs::write(&path, replacement).unwrap();

        let loc = td.kernels().next().unwrap();
        let blocks = td.read_threadblocks_in_launch_order(&loc).unwrap();
        let (_, block_zero) = blocks.iter().find(|(index, _)| *index == 0).unwrap();
        assert_eq!(block_zero.warps[&0].instructions[0].pc, 16);

        let sched = FunctionSchedule {
            arch: ArchId::Sm86,
            function_unique_id: 1,
            kernel_name: "_Z1kv".into(),
            function_name: "f".into(),
            by_pc: StdMap::new(),
        };
        let extracted =
            extract_per_sm_blocks(&td, &loc, &sched, sm86_test_placement(), true).unwrap();
        assert_eq!(extracted.per_sm_blocks[&0][0].warps[0].insts[0].pc, 16);

        assert!(matches!(
            TraceDir::open(tmp.path()),
            Err(TraceError::SamplingSidecarMalformed { .. })
        ));
    }

    #[test]
    fn sidecar_open_rejects_dot_only_pb_payload_name() {
        let tmp = tempfile::tempdir().unwrap();
        let sidecar = payload_bound_subset();
        write_grid_trace(tmp.path(), 60, 1, &SELECTED_BLOCKS, Some(&sidecar));
        let payload = tmp
            .path()
            .join("threadblocks/device_0/stream_0/kernel_1/.pb");
        std::fs::write(payload, encoded_threadblock(0)).unwrap();

        assert!(matches!(
            TraceDir::open(tmp.path()),
            Err(TraceError::ThreadblockPayloadSnapshotInvalid { .. })
        ));
    }

    #[test]
    fn sidecar_open_rejects_pb_named_directories() {
        let tmp = tempfile::tempdir().unwrap();
        let sidecar = payload_bound_subset();
        write_grid_trace(tmp.path(), 60, 1, &SELECTED_BLOCKS, Some(&sidecar));
        let directory = tmp
            .path()
            .join("threadblocks/device_0/stream_0/kernel_1/extra.pb");
        std::fs::create_dir(&directory).unwrap();
        std::fs::write(directory.join("nested.pb"), encoded_threadblock(0)).unwrap();

        assert!(matches!(
            TraceDir::open(tmp.path()),
            Err(TraceError::AnchoredSnapshot(
                sim_trace::AnchoredReadError::NotRegular { .. }
            ))
        ));
    }

    #[cfg(unix)]
    #[test]
    fn sidecar_open_rejects_symlinked_threadblock_payloads() {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::tempdir().unwrap();
        let sidecar = payload_bound_subset();
        write_grid_trace(tmp.path(), 60, 1, &SELECTED_BLOCKS, Some(&sidecar));
        let path = tmp
            .path()
            .join("threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_0,0,0.pb");
        let retained = tmp.path().join("retained-original.bytes");
        std::fs::rename(&path, &retained).unwrap();
        symlink(&retained, &path).unwrap();

        assert!(matches!(
            TraceDir::open(tmp.path()),
            Err(TraceError::AnchoredSnapshot(
                sim_trace::AnchoredReadError::NotRegular { .. }
            ))
        ));
    }

    #[test]
    fn sidecar_kernel_and_block_shape_are_bound_to_launch() {
        let sidecar = payload_bound_subset();

        let wrong_kernel = tempfile::tempdir().unwrap();
        write_grid_trace(wrong_kernel.path(), 60, 1, &SELECTED_BLOCKS, Some(&sidecar));
        rewrite_launch_metadata(wrong_kernel.path(), Some("_Zother"), None);
        assert!(matches!(
            run(wrong_kernel.path()).unwrap_err(),
            ReplayError::SamplingKernelMismatch { .. }
        ));

        let wrong_block = tempfile::tempdir().unwrap();
        write_grid_trace(wrong_block.path(), 60, 1, &SELECTED_BLOCKS, Some(&sidecar));
        rewrite_launch_metadata(wrong_block.path(), None, Some(64));
        assert!(matches!(
            run(wrong_block.path()).unwrap_err(),
            ReplayError::BlockDimRewritten { .. }
        ));
    }

    #[test]
    fn multidim_out_of_grid_cta_is_not_aliased_into_full_coverage() {
        // Grid (2,2,1): CTAs (2,0,0) and (3,0,0) flatten to 2 and 3 (inside
        // [0,4)) but are out of grid on the x axis. They must NOT be accepted as
        // valid blocks (which would falsely satisfy the completeness gate as
        // "4/4 covered"); the shared block_launch_index rejects them typed.
        let tmp = tempfile::tempdir().unwrap();
        write_grid_trace(tmp.path(), 2, 2, &[0, 1, 2, 3], None);
        let err = run(tmp.path()).unwrap_err();
        assert!(
            matches!(
                err,
                ReplayError::Trace(TraceError::BlockIdOutOfRange { .. })
            ),
            "expected BlockIdOutOfRange (not a false full-coverage pass), got {err:?}"
        );
    }
}
