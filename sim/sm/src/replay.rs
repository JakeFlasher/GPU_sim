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
use std::time::Instant;

use serde::Serialize;
use sim_trace::async_sidecars::{ObservationKey as TmaObservationKey, TmaObservation};

use sim_arch::card::{ArchCard, CalibError};
use sim_arch::{ArchId, LdgstsDestinationWavefrontProjection, Sku};
use sim_mem::{L1Config, L1CounterError, L1Counters, SectorExpansionError};
use sim_trace::pb;
use sim_trace::reader::{
    EffectiveDataBytesError, LdgstsDestinationTopology, LdgstsSourceCohort, MemoryAccessMeta,
    MemoryPayloadContract, SharedBankProjectionError, ValidatedMemoryInstruction,
    ValidatedMemoryRoute, ValidatedMemoryRouteError, ValidatedSectorError,
    ValidatedSectorErrorPhase, ValidatedSectorPlan, ValidatedWarp,
};
use sim_trace::{KernelLoc, TraceDir, TraceError, TraceLineage, kernel_grid};
#[cfg(feature = "campaign")]
use sim_trace::{
    LosslessFullProvenance, TraceSamplingMethod, TraceSamplingSidecar, WaveSegmentSubsetProvenance,
};

use sim_mem::{
    ASYNC_COPY_COMPLETION_VALUE_MASK, ASYNC_COPY_COMPOSITE_PAYLOAD_WORK_MAX, CacheOpToken,
    CoherenceConfig, L2Config, L2Counters, L2Model, L2OpClass, LdgstsL1Policy, MemRole,
    encode_async_copy_destination_work, encode_async_copy_payload_and_topology,
    encode_async_copy_payload_work, is_known_untagged_opcode, opcode_head,
};

use crate::causal_frontier::{
    CausalFrontierInstruction, CausalFrontierReport, CausalFrontierResidency,
    summarize_with_residency,
};
#[cfg(feature = "campaign")]
use crate::evidence::AdmittedLaunchEvidence;
use crate::issue::Sm120AlphaConfig;
use crate::issue::{
    AccessKind, AsyncCopyCohortDescriptor, AsyncCopyCohortReplay, AsyncCopyCohortSpan,
    AsyncInstrumentationOrderDiagnostics, BlockReplay, InstrumentationOrderAnchor, IssueError,
    IssueModelParams, IssueResourceSource, MemoryAdmissionCandidate, MemoryServiceCandidate,
    PayloadStoreSite, ReplayInst, SharedMemoryResidency, SmSimResult, TailFlags, WarpReplay,
    MAX_ASYNC_DECISION_SNAPSHOTS, simulate_sm_with_residency, validate_memory_admission_candidates,
    validate_memory_service_candidates,
};

fn append_bounded_async_decision_snapshots(
    metrics: &mut TraceMetrics,
    snapshots: &[crate::AsyncDecisionSnapshot],
    already_dropped: u64,
) -> Result<(), ReplayError> {
    metrics.async_decision_snapshots_dropped = metrics
        .async_decision_snapshots_dropped
        .checked_add(already_dropped)
        .ok_or(IssueError::CounterOverflow {
            counter: "launch async decision snapshots dropped",
        })?;
    let available = MAX_ASYNC_DECISION_SNAPSHOTS
        .saturating_sub(metrics.async_decision_snapshots.len());
    let kept = available.min(snapshots.len());
    metrics
        .async_decision_snapshots
        .extend_from_slice(&snapshots[..kept]);
    metrics.async_decision_snapshots_dropped = metrics
        .async_decision_snapshots_dropped
        .checked_add(u64::try_from(snapshots.len() - kept).map_err(|_| {
            IssueError::CounterOverflow {
                counter: "launch async decision snapshot omission",
            }
        })?)
        .ok_or(IssueError::CounterOverflow {
            counter: "launch async decision snapshots dropped",
        })?;
    Ok(())
}

/// Build a bounded, address-derived TMA source shadow for L2 metric coverage.
///
/// TMA source addresses are present in the native sidecar but are not carried
/// by the compact dynamic memory payload. The shadow therefore runs in a
/// separate L2 model, in sidecar instruction order, and never changes the
/// shared replay cache, timing, DRAM demand, or issue schedule. It is an
/// effective observer projection: ordinary L1/L2 requests and TMA requests
/// are not falsely serialized into one physical stream.
#[derive(Default)]
struct TmaShadowL2Stats {
    requests: u64,
    hits: u64,
    misses: u64,
    unavailable: u64,
    load: [u64; 4],
    store: [u64; 4],
}

fn observe_tma_l2_shadow(
    observations: &BTreeMap<TmaObservationKey, TmaObservation>,
    config: L2Config,
) -> Result<TmaShadowL2Stats, ReplayError> {
    let mut shadow = L2Model::new(config)?;
    let mut stats = TmaShadowL2Stats::default();
    for observation in observations.values() {
        let (sector_ids, class) = if observation.is_load {
            (&observation.source_sector_ids, L2OpClass::Read)
        } else if observation.is_store {
            (&observation.destination_sector_ids, L2OpClass::WriteAtomic)
        } else {
            continue;
        };
        if sector_ids.is_empty() {
            stats.unavailable = stats.unavailable.checked_add(1).ok_or(IssueError::CounterOverflow {
                counter: "TMA shadow L2 unavailable address records",
            })?;
            let lane = if observation.is_load { &mut stats.load } else { &mut stats.store };
            lane[3] = lane[3].checked_add(1).ok_or(IssueError::CounterOverflow {
                counter: "TMA shadow operation-class unavailable records",
            })?;
            continue;
        }
        for &sector_id in sector_ids.iter() {
            let line_id = sector_id.div_euclid(4);
            let sector = u8::try_from(sector_id.rem_euclid(4)).map_err(|_| {
                ReplayError::System(SystemError::L2(
                    sim_mem::L2AccessError::SectorOutOfRange { sector: u8::MAX },
                ))
            })?;
            let outcome = shadow
                .access(line_id, sector, class)
                .map_err(|error| ReplayError::System(SystemError::L2(error)))?;
            stats.requests = stats.requests.checked_add(1).ok_or(IssueError::CounterOverflow {
                counter: "TMA shadow L2 requests",
            })?;
            let lane = if observation.is_load { &mut stats.load } else { &mut stats.store };
            lane[0] = lane[0].checked_add(1).ok_or(IssueError::CounterOverflow {
                counter: "TMA shadow operation-class requests",
            })?;
            if outcome.counted_hit {
                stats.hits = stats.hits.checked_add(1).ok_or(IssueError::CounterOverflow {
                    counter: "TMA shadow L2 hits",
                })?;
                lane[1] = lane[1].checked_add(1).ok_or(IssueError::CounterOverflow {
                    counter: "TMA shadow operation-class hits",
                })?;
            } else {
                stats.misses = stats.misses.checked_add(1).ok_or(IssueError::CounterOverflow {
                    counter: "TMA shadow L2 misses",
                })?;
                lane[2] = lane[2].checked_add(1).ok_or(IssueError::CounterOverflow {
                    counter: "TMA shadow operation-class misses",
                })?;
            }
        }
    }
    Ok(stats)
}
use crate::metrics::TraceMetrics;
use crate::program::{
    FunctionSchedule, OpClass, PipelineClass, sm120_alpha_scalar_control_stall_suppressed,
};
use crate::sass_operands::{FixedLatencyOperandStatus, SassRegister, SassSourceOperand};
use crate::system::{
    DramBw, DramStats, L1PayloadAdmissionEvent, L2SourceSplit, SystemError,
    simulate_kernel_shared_with_residency_and_candidates_with_coherence,
    simulate_kernel_shared_with_residency_and_candidates_with_coherence_and_payload_events,
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
    #[error("generic atomic requires Global or Shared address-space routing, got {got:?}")]
    StaticGenericAtomicRouteInvalid { got: pb::MemorySpace },
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
    AsyncSidecar(#[from] sim_trace::async_sidecars::AsyncSidecarError),
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
    #[error("L1 payload admission events require shared-L2 replay")]
    PayloadAdmissionRequiresSharedL2,
    #[error("L1 payload admission event {index} has no dynamic trace identity")]
    PayloadAdmissionIdentityMissing { index: usize },
    #[error("L1 payload admission event {index} does not match a retained dynamic trace identity")]
    PayloadAdmissionIdentityNotObserved { index: usize },
    #[error(
        "L1 payload admission event {index} targets SM {event_sm}, but the retained dynamic instruction is on SM {trace_sm:?}"
    )]
    PayloadAdmissionSmMismatch {
        index: usize,
        event_sm: i32,
        trace_sm: Option<i32>,
    },
    #[error(
        "L1 payload admission event {index} binds to a non-store or unclassified GLOBAL access ({access_kind:?})"
    )]
    PayloadAdmissionNotStore {
        index: usize,
        access_kind: Option<pb::MemoryOpAccessKind>,
    },
    #[error("L1 payload admission event {index} has no retained physical warp-slot placement")]
    PayloadAdmissionWriterPlacementMissing { index: usize },
    #[error(
        "opt-in coherence service requires a complete decoded topology and lane masks; full_coverage={full_coverage}, decoded={decoded}, unknown_records={unknown_records}, active_masks_missing={active_masks_missing}, predicate_masks_missing={predicate_masks_missing}"
    )]
    CoherenceTopologyIncomplete {
        full_coverage: bool,
        decoded: bool,
        unknown_records: u64,
        active_masks_missing: u64,
        predicate_masks_missing: u64,
    },
    #[error("opt-in dirty writeback service requires complete unsampled launch coverage")]
    WritebackRequiresFullCoverage,
    #[error("fast frontier requires complete unsampled trace coverage")]
    FastFrontierRequiresFullCoverage,
    #[error(
        "fast frontier requires a single launch with no concurrent stream context; launches={launches}"
    )]
    FastFrontierLaunchScopeUnknown { launches: usize },
    #[error(
        "fast frontier requires active and predicate-mask presence on every dynamic record; active_missing={active_missing}, predicate_missing={predicate_missing}"
    )]
    FastFrontierMasksIncomplete {
        active_missing: u64,
        predicate_missing: u64,
    },
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
        "kernel {kernel_id} requires {bytes_per_cta} shared-memory bytes per CTA, exceeding architecture {arch} per-block opt-in limit {max_bytes_per_block}"
    )]
    CtaSharedMemoryBlockLimitExceeded {
        kernel_id: i32,
        arch: ArchId,
        bytes_per_cta: u64,
        max_bytes_per_block: u64,
    },
    #[error(
        "kernel {kernel_id} block has {threads_per_cta} threads, exceeding architecture {arch} per-block thread limit {max_threads_per_block}"
    )]
    CtaThreadCapacityExceeded {
        kernel_id: i32,
        arch: ArchId,
        threads_per_cta: u64,
        max_threads_per_block: u32,
    },
    #[error(
        "{path}: traced sm_id {sm_id} is outside architecture {arch} SKU {sku:?} identifier range 0..{sm_id_upper_bound}"
    )]
    SmIdOutOfRange {
        path: PathBuf,
        sm_id: i32,
        arch: ArchId,
        sku: Sku,
        sm_id_upper_bound: u32,
    },
    #[cfg(feature = "campaign")]
    #[error(
        "trace_sampling per_sm_slot_map block {launch_index} declares sm_id {sm_id} outside architecture {arch} SKU {sku:?} identifier range 0..{sm_id_upper_bound}"
    )]
    SamplingSmIdOutOfRange {
        launch_index: u64,
        sm_id: u64,
        arch: ArchId,
        sku: Sku,
        sm_id_upper_bound: u32,
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
        "TMA opcode {opcode:?} at pc {pc:#x} on architecture {arch} has an unsupported \
         direction; only the source-global/destination-shared load form is modeled"
    )]
    UnsupportedTmaDirection {
        arch: ArchId,
        pc: u32,
        opcode: String,
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
    /// Number of 32-byte sectors in the unique 128-byte instruction-line
    /// envelope.  GCC/ICC fetches an aligned 128-byte line even when the
    /// traced control flow executes only one sector of that line.  This is a
    /// source-qualified traffic observer; it does not alter issue timing.
    compulsory_instruction_line_read_sectors: u64,
    /// Static SASS instruction footprint rounded to aligned 128-byte GCC
    /// fetch lines. This remains separate from the executed-path ledger above
    /// so consumers can retain a bounded cold/static envelope explicitly.
    static_instruction_line_read_sectors: u64,
}

impl FrontEndCounters {
    pub const fn compulsory_instruction_read_sectors(self) -> u64 {
        self.compulsory_instruction_read_sectors
    }

    pub const fn compulsory_instruction_line_read_sectors(self) -> u64 {
        self.compulsory_instruction_line_read_sectors
    }

    pub const fn static_instruction_line_read_sectors(self) -> u64 {
        self.static_instruction_line_read_sectors
    }

    fn add_checked(self, other: Self) -> Option<Self> {
        Some(Self {
            compulsory_instruction_read_sectors: self
                .compulsory_instruction_read_sectors
                .checked_add(other.compulsory_instruction_read_sectors)?,
            compulsory_instruction_line_read_sectors: self
                .compulsory_instruction_line_read_sectors
                .checked_add(other.compulsory_instruction_line_read_sectors)?,
            static_instruction_line_read_sectors: self
                .static_instruction_line_read_sectors
                .checked_add(other.static_instruction_line_read_sectors)?,
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
/// model-local interval summary; every other warp is retained as detailed or
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
    pub macro_dispatches: u64,
    pub maximum_exact_final_cycle: u64,
    pub maximum_summarized_final_cycle: u64,
    pub front_end_static_lines: u64,
    pub front_end_penalty_cycles: u64,
    pub terminal_state_matches: bool,
    /// Full-trace denominator for selective compression. Candidate records
    /// are those inside a scalar segment; reified records are candidates that
    /// could not be collapsed; uncovered records are every prefix/suffix or
    /// noncandidate record left on the exact path.
    pub coverage: FastFrontierCoverage,
    pub refusal_reasons: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct FastFrontierCoverage {
    pub trace_records: u64,
    pub candidate_records: u64,
    pub summarized_records: u64,
    pub reified_records: u64,
    pub uncovered_records: u64,
}

struct FastDynamicWarp<'a> {
    warp: &'a WarpReplay,
    masks: &'a [u32],
    next: usize,
    slot: Option<&'a crate::program::IssueSlot>,
    ready: u64,
    completion: u64,
    waiting_at_barrier: bool,
    barrier_arrival: u64,
    done: bool,
    registers: Vec<(SassRegister, u32, u64)>,
    barriers: [u64; 6],
}

fn fast_cycle_add(base: u64, delta: u64) -> Result<u64, ReplayError> {
    base.checked_add(delta)
        .ok_or_else(|| ReplayError::FastFrontier {
            detail: "fast replay cycle overflow".to_owned(),
        })
}

fn fast_register_overlap(left: SassRegister, right: SassRegister) -> bool {
    if left.file() != right.file() {
        return false;
    }
    let (Some(left), Some(right)) = (
        left.index()
            .map(|i| (u32::from(i), u32::from(left.span_registers()))),
        right
            .index()
            .map(|i| (u32::from(i), u32::from(right.span_registers()))),
    ) else {
        return false;
    };
    left.0 < right.0 + right.1 && right.0 < left.0 + left.1
}

fn fast_register_mask(register: SassRegister, mask: u32) -> u32 {
    match register.file() {
        crate::sass_operands::SassRegisterFile::Uniform
        | crate::sass_operands::SassRegisterFile::UniformPredicate
            if mask != 0 =>
        {
            u32::MAX
        }
        _ => mask,
    }
}

fn fast_pipeline_index(pipeline: PipelineClass) -> usize {
    match pipeline {
        PipelineClass::ScalarAlu { .. } => 0,
        PipelineClass::Tensor { .. } => 1,
        PipelineClass::Sfu { .. } => 2,
        _ => 3,
    }
}

impl FastDynamicWarp<'_> {
    fn ready_at(
        &self,
        dispatch: u64,
        pipes: &[u64; 4],
        pending: &[u64],
        rearm: u64,
    ) -> Option<u64> {
        if self.done || self.waiting_at_barrier {
            return None;
        }
        let slot = self.slot?;
        let mask = self.masks[self.next];
        let mut ready = self.ready.max(dispatch);
        if let Some(guard) = slot.guard_predicate {
            let lanes = fast_register_mask(guard, self.warp.insts[self.next].active_lane_mask);
            for &(prior, prior_lanes, done) in &self.registers {
                if lanes & prior_lanes != 0 && fast_register_overlap(prior, guard) {
                    ready = ready.max(done);
                }
            }
        }
        if mask != 0 {
            ready = ready.max(pipes[fast_pipeline_index(slot.pipeline_class())]);
            let waits =
                slot.control().wait_mask().bits() | slot.ordinary_dependency_wait_mask().bits();
            for (index, &clear) in self.barriers.iter().enumerate() {
                if waits & (1 << index) != 0 {
                    ready = ready.max(clear);
                }
            }
            if let Some(projection) = slot.fixed_latency_operands().projected() {
                let sources = projection
                    .sources()
                    .iter()
                    .filter_map(|source| match source {
                        SassSourceOperand::Register(operand) => Some(operand.register()),
                        _ => None,
                    });
                // RAW and WAW are both completion dependencies. Include predicate
                // inputs, ignore constant registers, and preserve lane identity.
                for register in sources
                    .chain(projection.predicate_sources().iter().map(|o| o.register()))
                    .chain(projection.destinations().iter().map(|o| o.register()))
                {
                    let lanes = fast_register_mask(register, mask);
                    for &(prior, prior_lanes, done) in &self.registers {
                        if lanes & prior_lanes != 0 && fast_register_overlap(prior, register) {
                            ready = ready.max(done);
                        }
                    }
                }
            }
            if let PipelineClass::ScalarAlu {
                in_flight_capacity, ..
            } = slot.pipeline_class()
            {
                // Candidate queries must never remove another candidate's pending
                // completions at a speculative future timestamp.
                while pending.iter().filter(|&&done| done > ready).count()
                    >= in_flight_capacity.max(1) as usize
                {
                    ready = pending.iter().copied().filter(|&done| done > ready).min()?;
                }
            }
            if slot.op_class() == OpClass::BlockBarrier {
                ready = ready.max(rearm);
            }
        }
        Some(ready)
    }
}

/// Event-driven SM120 fast replay. Same-cycle picks see one immutable resource
/// snapshot; idle time jumps to the exact next known readiness. The modulo-four
/// SMSP assignment is the existing alpha assumption, not observed wiring.
fn fast_dynamic_sm_frontier(
    blocks: &[BlockReplay],
    schedule: &FunctionSchedule,
    masks: &BTreeMap<(i64, i32), Vec<u32>>,
    params: &IssueModelParams,
    residency: SharedMemoryResidency,
) -> Result<(u64, u64, u64, u64), ReplayError> {
    struct BlockState {
        first: usize,
        end: usize,
        slots: u64,
        rearm: u64,
    }
    let mut warps = Vec::new();
    let mut block_states = Vec::new();
    for block in blocks {
        let first = warps.len();
        let mut slots = 0u64;
        for warp in &block.warps {
            slots |= 1u64.checked_shl(warp.warp_slot as u32).ok_or_else(|| {
                ReplayError::FastFrontier {
                    detail: "fast warp slot exceeds its resident bitmap".to_owned(),
                }
            })?;
            let lane_masks = masks
                .get(&(block.launch_index, warp.warp_slot))
                .ok_or_else(|| ReplayError::FastFrontier {
                    detail: "fast warp lane masks are missing".to_owned(),
                })?;
            if lane_masks.len() != warp.insts.len() {
                return Err(ReplayError::FastFrontier {
                    detail: "fast warp lane-mask length mismatch".to_owned(),
                });
            }
            let slot = warp
                .insts
                .first()
                .map(|inst| {
                    schedule.issue_slot(u64::from(inst.pc)).ok_or_else(|| {
                        ReplayError::FastFrontier {
                            detail: "fast schedule PC missing".to_owned(),
                        }
                    })
                })
                .transpose()?;
            warps.push(FastDynamicWarp {
                warp,
                masks: lane_masks,
                next: 0,
                slot,
                ready: 0,
                completion: 0,
                waiting_at_barrier: false,
                barrier_arrival: 0,
                done: slot.is_none(),
                registers: Vec::new(),
                barriers: [0; 6],
            });
        }
        block_states.push(BlockState {
            first,
            end: warps.len(),
            slots,
            rearm: 0,
        });
    }
    let mut active: Vec<usize> = Vec::new();
    let mut next_block = 0;
    let mut active_slots = 0u64;
    let mut dispatch = [0u64; 4];
    let mut pipes = [[0u64; 4]; 4];
    let mut pending: [Vec<u64>; 4] = std::array::from_fn(|_| Vec::new());
    let mut held = [None; 4];
    let mut barrier_server = 0u64;
    let barrier_rearm = schedule
        .by_pc
        .values()
        .filter_map(|slot| match slot.pipeline_class() {
            PipelineClass::Barrier {
                generation_rearm_cycles,
                ..
            } => Some(generation_rearm_cycles),
            _ => None,
        })
        .max()
        .unwrap_or(0);
    let mut now = 0u64;
    let mut frontier = 0u64;
    let (mut memory_records, mut async_records, mut barrier_records) = (0u64, 0u64, 0u64);
    while next_block < blocks.len() || !active.is_empty() {
        let mut next_event = None;
        active.retain(|&block| {
            let state = &block_states[block];
            let members = &warps[state.first..state.end];
            if members.iter().all(|w| w.done) {
                let end = members.iter().map(|w| w.completion).max().unwrap_or(now);
                if end <= now {
                    active_slots &= !state.slots;
                    return false;
                }
                next_event = Some(next_event.map_or(end, |prior: u64| prior.min(end)));
            }
            true
        });
        // Admission reuses traced slots only after the prior CTA has retired.
        // Apply the same launch-order, warp, CTA and shared-memory ceilings as
        // full replay; do not admit every traced block simultaneously.
        while next_block < blocks.len() {
            let state = &block_states[next_block];
            let occupied = active_slots | state.slots;
            let shared_fits = (active.len() as u64 + 1)
                .checked_mul(residency.bytes_per_cta())
                .is_some_and(|bytes| bytes <= residency.max_bytes_per_sm());
            if active.len() >= params.max_resident_blocks_per_sm() as usize
                || occupied.count_ones() > params.max_resident_warps_per_sm()
                || active_slots & state.slots != 0
                || !shared_fits
            {
                break;
            }
            for warp in &mut warps[state.first..state.end] {
                warp.ready = now;
                warp.completion = now;
            }
            active_slots = occupied;
            active.push(next_block);
            next_block += 1;
        }
        if active.is_empty() {
            if next_block == blocks.len() {
                break;
            }
            return Err(ReplayError::FastFrontier {
                detail: "fast CTA exceeds resident resource limits".to_owned(),
            });
        }
        for completions in &mut pending {
            completions.retain(|&done| done > now);
        }
        let mut picks = [None; 4];
        for &block in &active {
            let state = &block_states[block];
            for index in state.first..state.end {
                let warp = &warps[index];
                let smsp = warp.warp.warp_slot as usize % 4;
                let Some(ready) =
                    warp.ready_at(dispatch[smsp], &pipes[smsp], &pending[smsp], state.rearm)
                else {
                    continue;
                };
                if ready > now {
                    next_event = Some(next_event.map_or(ready, |prior| prior.min(ready)));
                } else if picks[smsp].is_none()
                    || held[smsp] == Some(index)
                    || (picks[smsp] != held[smsp] && picks[smsp].is_some_and(|prior| index > prior))
                {
                    picks[smsp] = Some(index);
                }
            }
        }
        if picks.iter().all(Option::is_none) {
            now =
                next_event
                    .filter(|&next| next > now)
                    .ok_or_else(|| ReplayError::FastFrontier {
                        detail: "fast replay cannot progress: incomplete CTA barrier or schedule"
                            .to_owned(),
                    })?;
            continue;
        }
        // Every pick below issues at `now`; committing one partition cannot
        // move the timestamp seen by another partition's already selected pick.
        for (smsp, pick) in picks.into_iter().enumerate() {
            let Some(index) = pick else { continue };
            let warp = &mut warps[index];
            let inst = warp.warp.insts[warp.next];
            let slot = warp.slot.ok_or_else(|| ReplayError::FastFrontier {
                detail: "fast schedule PC missing".to_owned(),
            })?;
            let mask = warp.masks[warp.next];
            let pipeline = slot.pipeline_class();
            let control = slot.control();
            let interval = pipeline.initiation_interval_cycles().unwrap_or(1).max(1);
            let latency = match pipeline {
                PipelineClass::ScalarAlu { latency_cycles, .. } => latency_cycles,
                _ => 1,
            };
            let issued_end = fast_cycle_add(now, 1)?;
            dispatch[smsp] = issued_end;
            frontier = frontier.max(issued_end);
            let projection = slot.fixed_latency_operands().projected();
            let observes_register = mask != 0 && projection.is_some_and(|p| p.destinations().iter().any(|d| !d.register().is_constant()));
            let result_ready = crate::issue::data_ready_time(inst.kind, &[], now, params, None, None, observes_register)?;
            let done = if mask == 0 {
                issued_end
            } else {
                if matches!(pipeline, PipelineClass::ScalarAlu { .. }) {
                    fast_cycle_add(now, latency)?
                } else {
                    result_ready.unwrap_or(fast_cycle_add(now, 1)?)
                }
            };
            if mask != 0 {
                if !matches!(
                    pipeline,
                    PipelineClass::Barrier { .. } | PipelineClass::Branch { .. }
                ) {
                    pipes[smsp][fast_pipeline_index(pipeline)] = fast_cycle_add(now, interval)?;
                }
                if matches!(pipeline, PipelineClass::ScalarAlu { .. }) {
                    pending[smsp].push(done);
                }
                // Completed writes no longer constrain future reads. Partial
                // predication overwrites only executing lanes, not the whole register.
                warp.registers.retain(|&(_, _, ready)| ready > now);
                if let Some(projection) = projection {
                    for register in projection.destinations().iter().flat_map(|d| d.register().components()) {
                        let lanes = fast_register_mask(register, mask);
                        for (prior, prior_lanes, _) in &mut warp.registers {
                            if *prior == register {
                                *prior_lanes &= !lanes;
                            }
                        }
                        warp.registers.retain(|&(_, lanes, _)| lanes != 0);
                        warp.registers.push((register, lanes, done));
                    }
                }
            }
            // Share Flat-mode dependency latency arithmetic with full replay.
            // Cache queues and native async visibility are still not modeled here.
            let write = (mask != 0 && slot.op_class() != OpClass::AsyncCopyCommit)
                .then_some(control.write_barrier())
                .flatten();
            if let Some(ready) =
                crate::issue::data_ready_time(inst.kind, &[], now, params, None, write, false)?
            {
                if let Some(barrier) = write {
                    warp.barriers[barrier.index()] = warp.barriers[barrier.index()].max(ready);
                }
            }
            if mask != 0
                && let Some(barrier) = control.read_barrier()
            {
                warp.barriers[barrier.index()] = warp.barriers[barrier.index()]
                    .max(fast_cycle_add(now, params.read_barrier_latency())?);
            }
            let suppress = mask != 0
                && projection.is_some()
                && sm120_alpha_scalar_control_stall_suppressed(slot.opcode_text());
            let stall = if suppress {
                1
            } else {
                u64::from(control.stall()).max(1)
            };
            warp.ready = fast_cycle_add(now, stall.max(if control.yield_flag() { 2 } else { 1 }))?;
            warp.completion = warp.completion.max(done);
            match inst.kind {
                AccessKind::NonMemory | AccessKind::PredicatedOffMemory => {}
                AccessKind::GlobalAsyncCopy { .. } => {
                    memory_records += 1;
                    async_records += 1;
                }
                _ => memory_records += 1,
            }
            warp.next += 1;
            if slot.op_class() == OpClass::BlockBarrier && mask != 0 {
                barrier_records += 1;
                barrier_server = fast_cycle_add(barrier_server.max(now), interval)?;
                warp.barrier_arrival = barrier_server;
                warp.waiting_at_barrier = true;
            } else if warp.next == warp.warp.insts.len() {
                warp.done = true;
            }
            warp.slot = warp
                .warp
                .insts
                .get(warp.next)
                .map(|next| {
                    schedule.issue_slot(u64::from(next.pc)).ok_or_else(|| {
                        ReplayError::FastFrontier {
                            detail: "fast schedule PC missing".to_owned(),
                        }
                    })
                })
                .transpose()?;
            held[smsp] =
                (!warp.done && !warp.waiting_at_barrier && !control.yield_flag()).then_some(index);
        }
        for &block in &active {
            let state = &mut block_states[block];
            let members = &mut warps[state.first..state.end];
            if members.iter().any(|w| w.waiting_at_barrier)
                && members.iter().all(|w| w.done || w.waiting_at_barrier)
            {
                let arrival = members
                    .iter()
                    .filter(|w| w.waiting_at_barrier)
                    .map(|w| w.barrier_arrival)
                    .max()
                    .unwrap_or(now);
                let release = fast_cycle_add(arrival, 1)?;
                state.rearm = fast_cycle_add(arrival, barrier_rearm)?;
                for warp in members.iter_mut().filter(|w| w.waiting_at_barrier) {
                    warp.waiting_at_barrier = false;
                    warp.ready = warp.ready.max(release);
                    warp.completion = warp.completion.max(release);
                    warp.done = warp.next == warp.warp.insts.len();
                }
                frontier = frontier.max(release);
            }
        }
        now = fast_cycle_add(now, 1)?;
    }
    frontier = frontier.max(warps.iter().map(|w| w.completion).max().unwrap_or(0));
    Ok((frontier, memory_records, async_records, barrier_records))
}

fn fast_scalar_segments(
    warp: &WarpReplay,
    schedule: &FunctionSchedule,
) -> Vec<Vec<CausalFrontierInstruction>> {
    let mut segments = Vec::new();
    let mut program = Vec::new();
    let mut prior_destinations: Vec<SassRegister> = Vec::new();
    let mut segment_start = None;

    fn register_overlap(left: SassRegister, right: SassRegister) -> bool {
        if left.file() != right.file() {
            return false;
        }
        let (Some(left_index), Some(right_index)) = (left.index(), right.index()) else {
            return false;
        };
        let left_start = u32::from(left_index);
        let right_start = u32::from(right_index);
        let left_end = left_start + u32::from(left.span_registers());
        let right_end = right_start + u32::from(right.span_registers());
        left_start < right_end && right_start < left_end
    }

    let flush = |segments: &mut Vec<Vec<CausalFrontierInstruction>>,
                 program: &mut Vec<CausalFrontierInstruction>,
                 prior_destinations: &mut Vec<SassRegister>,
                 segment_start: &mut Option<usize>| {
        // A valid entry boundary may begin after an unmodelled prefix. Do not
        // silently discard a safe suffix merely because its local trace index
        // is nonzero; coverage accounting records the prefix separately.
        if program.len() >= 2 && segment_start.is_some() {
            segments.push(std::mem::take(program));
        } else {
            program.clear();
        }
        prior_destinations.clear();
        *segment_start = None;
    };
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
            if control.wait_mask().bits() != 0
                || control.write_barrier().is_some()
                || control.read_barrier().is_some()
            {
                return None;
            }
            let projection = match slot.fixed_latency_operands() {
                FixedLatencyOperandStatus::Projected(projection) => projection,
                // The compact dynamic trace carries no register operands. A
                // static projection is therefore required before a region can
                // be certified; unknown provenance must remain a refusal.
                FixedLatencyOperandStatus::NotApplicable | FixedLatencyOperandStatus::Gap(_) => {
                    return None;
                }
            };
            let mut current_sources = Vec::new();
            for source in projection.sources() {
                if let SassSourceOperand::Register(operand) = source {
                    current_sources.push(operand.register());
                }
            }
            current_sources.extend(
                projection
                    .predicate_sources()
                    .iter()
                    .map(|operand| operand.register()),
            );
            let current_destinations = projection
                .destinations()
                .iter()
                .map(|operand| operand.register())
                .collect::<Vec<_>>();
            if current_sources.iter().any(|source| {
                prior_destinations
                    .iter()
                    .any(|dest| register_overlap(*source, *dest))
            }) || current_destinations.iter().any(|dest| {
                prior_destinations
                    .iter()
                    .any(|prior| register_overlap(*dest, *prior))
            }) {
                return None;
            }
            let sequence = u64::try_from(sequence).ok()?;
            let descriptor = CausalFrontierInstruction::from_schedule(
                schedule,
                u64::from(instruction.pc),
                sequence,
                warp.warp_slot,
                latency_cycles,
                1,
                true,
                false,
            )
            .ok()?;
            Some((descriptor, current_destinations))
        })() {
            Some((descriptor, destinations)) => {
                if program.is_empty() {
                    segment_start = Some(sequence);
                }
                prior_destinations.extend(destinations);
                descriptor
            }
            None => {
                flush(
                    &mut segments,
                    &mut program,
                    &mut prior_destinations,
                    &mut segment_start,
                );
                continue;
            }
        };
        program.push(descriptor);
    }
    flush(
        &mut segments,
        &mut program,
        &mut prior_destinations,
        &mut segment_start,
    );
    segments
}

fn fast_scalar_refusal_reason(warp: &WarpReplay, schedule: &FunctionSchedule) -> &'static str {
    let mut saw_scalar = false;
    let mut saw_unknown_registers = false;
    let mut saw_control_boundary = false;
    for instruction in &warp.insts {
        if !matches!(instruction.kind, AccessKind::NonMemory) {
            continue;
        }
        let Some(slot) = schedule.issue_slot(u64::from(instruction.pc)) else {
            continue;
        };
        if slot.op_class() != OpClass::Other
            || !matches!(slot.mem_role(), MemRole::None)
            || !matches!(slot.pipeline_class(), PipelineClass::ScalarAlu { .. })
        {
            continue;
        }
        saw_scalar = true;
        if slot.fixed_latency_operands().projected().is_none() {
            saw_unknown_registers = true;
        }
        let control = slot.control();
        if control.wait_mask().bits() != 0
            || control.write_barrier().is_some()
            || control.read_barrier().is_some()
        {
            saw_control_boundary = true;
        }
    }
    if saw_unknown_registers {
        "register_provenance_unavailable"
    } else if saw_control_boundary {
        "control_or_dependency_boundary"
    } else if saw_scalar {
        "register_dependency_or_short_region"
    } else {
        "memory_control_or_non_scalar"
    }
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
/// unknown register dependencies, and incomplete/slot-reused residency;
/// callers can use the summary to decide which ranges are safe to summarize
/// while keeping the ordinary replay as fallback.
pub fn fast_frontier_for_trace(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedule: &FunctionSchedule,
    arch_card: &ArchCard,
) -> Result<FastFrontierSummary, ReplayError> {
    let placement = PlacementConfig::from_arch_card(arch_card, schedule.arch)?;
    require_trace_architecture(td, schedule.arch)?;
    let launches = td.kernels().count();
    if launches != 1 {
        return Err(ReplayError::FastFrontierLaunchScopeUnknown { launches });
    }
    let extracted = extract_per_sm_blocks(td, loc, schedule, placement, false, None)?;
    if !matches!(&extracted.coverage, ExtractedCoverage::Full(_)) {
        return Err(ReplayError::FastFrontierRequiresFullCoverage);
    }
    let trace_records = extracted
        .per_sm_blocks
        .values()
        .flat_map(|blocks| blocks.iter())
        .flat_map(|block| block.warps.iter())
        .try_fold(0_u64, |count, warp| {
            count
                .checked_add(u64::try_from(warp.insts.len()).map_err(|_| {
                    ReplayError::FastFrontier {
                        detail: "trace record count exceeds u64".to_owned(),
                    }
                })?)
                .ok_or_else(|| ReplayError::FastFrontier {
                    detail: "trace record count overflows u64".to_owned(),
                })
        })?;
    let active_missing = extracted
        .trace_metrics
        .instructions
        .active_mask_missing_instructions;
    let predicate_missing = extracted
        .trace_metrics
        .instructions
        .predicate_mask_missing_instructions;
    let active_nonfull = extracted
        .trace_metrics
        .instructions
        .active_mask_nonfull_instructions;
    let predicate_nonfull = extracted
        .trace_metrics
        .instructions
        .predicate_mask_nonfull_instructions;
    if active_missing != 0 || predicate_missing != 0 {
        return Err(ReplayError::FastFrontierMasksIncomplete {
            active_missing,
            predicate_missing,
        });
    }
    let mut summary = FastFrontierSummary {
        // Keep the v1 wire token for compatibility. Its semantics are now
        // documented as descriptor-model self-equivalence; native refinement
        // will receive a distinct contract in a later schema revision.
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
        macro_dispatches: 0,
        maximum_exact_final_cycle: 0,
        maximum_summarized_final_cycle: 0,
        front_end_static_lines: 0,
        front_end_penalty_cycles: 0,
        terminal_state_matches: true,
        coverage: FastFrontierCoverage {
            trace_records,
            uncovered_records: trace_records,
            ..FastFrontierCoverage::default()
        },
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
    if active_nonfull != 0 || predicate_nonfull != 0 {
        // The compact descriptor does not retain per-record predicate
        // populations. Preserve the fast-only executable contract, but refuse
        // the whole launch rather than treating predicated scalar work as an
        // all-lane empty-boundary recurrence.
        for blocks in extracted.per_sm_blocks.values() {
            add(
                "blocks_seen",
                &mut summary.blocks_seen,
                u64::try_from(blocks.len()).map_err(|_| ReplayError::FastFrontier {
                    detail: "block count exceeds u64".to_owned(),
                })?,
            )?;
            for block in blocks {
                add(
                    "warps_seen",
                    &mut summary.warps_seen,
                    u64::try_from(block.warps.len()).map_err(|_| ReplayError::FastFrontier {
                        detail: "warp count exceeds u64".to_owned(),
                    })?,
                )?;
                for _ in &block.warps {
                    if active_nonfull != 0 {
                        increment_fast_reason(
                            &mut summary.refusal_reasons,
                            "nonfull_active_population",
                        )?;
                    }
                    if predicate_nonfull != 0 {
                        increment_fast_reason(
                            &mut summary.refusal_reasons,
                            "nonfull_predicate_population",
                        )?;
                    }
                }
            }
        }
        return Ok(summary);
    }
    for blocks in extracted.per_sm_blocks.values() {
        // The causal analyzer's `Complete` premise is the full physical
        // roster for the interval, not the current CTA's local warp list.
        // Gather every traced block on this SM and refuse slot reuse rather
        // than accidentally certifying a segment with an omitted competitor.
        let residents = blocks
            .iter()
            .flat_map(|block| block.warps.iter().map(|warp| warp.warp_slot))
            .collect::<Vec<_>>();
        let roster_has_reuse = {
            let mut sorted = residents.clone();
            sorted.sort_unstable();
            sorted.windows(2).any(|pair| pair[0] == pair[1])
        };
        // The current compact trace has no CTA start/end epochs or incoming
        // pipeline snapshot.  Only one traced warp on one block can therefore
        // satisfy the analyzer's empty-boundary premise; all other rosters
        // remain explicit refusals instead of fresh-state certificates.
        let single_empty_boundary = blocks.len() == 1 && residents.len() == 1;
        for block in blocks {
            add("blocks_seen", &mut summary.blocks_seen, 1)?;
            if roster_has_reuse || !single_empty_boundary {
                let reason = if roster_has_reuse {
                    "resident_roster_slot_reuse"
                } else {
                    "incoming_pipeline_or_residency_unknown"
                };
                for _warp in &block.warps {
                    add("warps_seen", &mut summary.warps_seen, 1)?;
                    increment_fast_reason(&mut summary.refusal_reasons, reason)?;
                }
                continue;
            }
            let mut candidate_segments = Vec::new();
            for warp in &block.warps {
                add("warps_seen", &mut summary.warps_seen, 1)?;
                let warp_segments = fast_scalar_segments(warp, schedule);
                if warp_segments.is_empty() {
                    increment_fast_reason(
                        &mut summary.refusal_reasons,
                        fast_scalar_refusal_reason(warp, schedule),
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
                summary.coverage.candidate_records = summary
                    .coverage
                    .candidate_records
                    .checked_add(segment_len)
                    .ok_or_else(|| ReplayError::FastFrontier {
                        detail: "candidate coverage count overflows u64".to_owned(),
                    })?;
                // A suffix discovered after an unmodelled prefix has no
                // captured entry pipeline/scoreboard state. Retain it in the
                // coverage denominator, but force the exact/model-local
                // refusal path until a typed entry boundary is supplied.
                let closed_entry = segment[0].sequence == 0;
                if !closed_entry {
                    increment_fast_reason(
                        &mut summary.refusal_reasons,
                        "missing_entry_boundary_state",
                    )?;
                }
                let residency = if closed_entry {
                    // The compact trace is not an independent dependency or
                    // effect witness. Residency alone therefore remains an
                    // exact/fallback premise; a separate static/trace witness
                    // is required before a macro can be installed.
                    CausalFrontierResidency::Complete {
                        physical_warps: residents.clone(),
                    }
                } else {
                    CausalFrontierResidency::Unknown
                };
                let report: CausalFrontierReport =
                    summarize_with_residency(std::slice::from_ref(&segment), residency).map_err(
                        |error| ReplayError::FastFrontier {
                            detail: error.to_string(),
                        },
                    )?;
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
                summary.coverage.summarized_records = summary
                    .coverage
                    .summarized_records
                    .checked_add(report.collapsed_instructions)
                    .ok_or_else(|| ReplayError::FastFrontier {
                        detail: "summarized coverage count overflows u64".to_owned(),
                    })?;
                let reified_records = segment_len
                    .checked_sub(report.collapsed_instructions)
                    .ok_or_else(|| ReplayError::FastFrontier {
                        detail: "collapsed coverage exceeds candidate segment".to_owned(),
                    })?;
                summary.coverage.reified_records = summary
                    .coverage
                    .reified_records
                    .checked_add(reified_records)
                    .ok_or_else(|| ReplayError::FastFrontier {
                        detail: "reified coverage count overflows u64".to_owned(),
                    })?;
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
                add(
                    "macro_dispatches",
                    &mut summary.macro_dispatches,
                    report.macro_dispatches,
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
    summary.coverage.uncovered_records = summary
        .coverage
        .trace_records
        .checked_sub(summary.coverage.candidate_records)
        .ok_or_else(|| ReplayError::FastFrontier {
            detail: "candidate coverage exceeds trace record count".to_owned(),
        })?;
    Ok(summary)
}

/// Build the SM120 alpha stage/event prediction without walking the full
/// memory/event queue.  This complementary diagnostic consumes the compact
/// native trace, static schedule classes, dynamic per-warp dependencies,
/// modeled SMSP placement and CTA barriers. Ordinary memory dependencies use
/// the shared Flat-latency arithmetic; cache service and native async visibility
/// remain outside this path.
///
/// Unlike the older SM86 selective-scalar path this function accepts one
/// launch from a multi-launch TraceDir.  Each launch is attributed by its own
/// `KernelLoc`, so the caller can report a complete capture without treating
/// capture-wide order as a timing oracle.
pub fn sm120_fast_stage_for_trace(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedule: &FunctionSchedule,
    arch_card: &ArchCard,
    alpha_config: Sm120AlphaConfig,
) -> Result<FastFrontierSummary, ReplayError> {
    let placement = PlacementConfig::from_arch_card(arch_card, schedule.arch)?;
    require_trace_architecture(td, schedule.arch)?;
    let extracted = extract_per_sm_blocks(td, loc, schedule, placement, false, None)?;
    let params = IssueModelParams::from_card_alpha(ArchId::Sm120, arch_card, alpha_config)?;
    let mut masks = BTreeMap::new();
    for (launch_index, block) in td.validated_threadblocks_in_launch_order(loc)? {
        for warp in block.warps() {
            let Some(slot) = warp.instructions().find_map(|inst| inst.warpid_sm()) else {
                continue;
            };
            let lane_masks = warp
                .instructions()
                .map(|inst| {
                    if inst.predicate_mask_present() {
                        inst.predicate_on_mask()
                    } else if inst.active_mask_present() {
                        inst.active_mask()
                    } else {
                        u32::MAX
                    }
                })
                .collect::<Vec<_>>();
            masks.insert((launch_index, slot), lane_masks);
        }
    }
    let trace_records = extracted.records_seen;
    let mut summary = FastFrontierSummary {
        contract: "sm120_fast_stage_event_v1",
        architecture: schedule.arch.token(),
        blocks_seen: 0,
        warps_seen: 0,
        eligible_warps: 0,
        candidate_regions: 0,
        certified_regions: 0,
        reified_regions: 0,
        candidate_instructions: trace_records,
        collapsed_instructions: 0,
        exact_execution_steps: trace_records,
        summarized_execution_steps: 0,
        macro_dispatches: 0,
        maximum_exact_final_cycle: 0,
        maximum_summarized_final_cycle: 0,
        front_end_static_lines: 0,
        front_end_penalty_cycles: 0,
        // This summary is intentionally not an exact-state certificate.  A
        // false value prevents downstream users from reading it as the old
        // SM86 scalar equivalence contract.
        terminal_state_matches: false,
        coverage: FastFrontierCoverage {
            trace_records,
            candidate_records: trace_records,
            summarized_records: 0,
            reified_records: trace_records,
            uncovered_records: 0,
        },
        refusal_reasons: BTreeMap::new(),
    };

    let mut memory_records = 0_u64;
    let mut async_records = 0_u64;
    let mut barrier_records = 0_u64;
    let mut resource_class_counts: BTreeMap<String, u64> = BTreeMap::new();
    let add = |counter: &'static str, value: &mut u64, amount: u64| {
        *value = value
            .checked_add(amount)
            .ok_or_else(|| ReplayError::FastFrontier {
                detail: format!("{counter} overflows u64"),
            })?;
        Ok::<(), ReplayError>(())
    };

    for blocks in extracted.per_sm_blocks.values() {
        add(
            "blocks_seen",
            &mut summary.blocks_seen,
            u64::try_from(blocks.len()).map_err(|_| ReplayError::FastFrontier {
                detail: "block count exceeds u64".to_owned(),
            })?,
        )?;
        for block in blocks {
            let warp_count = u64::try_from(block.warps.len()).map_err(|_| ReplayError::FastFrontier {
                detail: "warp count exceeds u64".to_owned(),
            })?;
            add(
                "warps_seen",
                &mut summary.warps_seen,
                warp_count,
            )?;
            add(
                "eligible_warps",
                &mut summary.eligible_warps,
                warp_count,
            )?;
            add(
                "candidate_regions",
                &mut summary.candidate_regions,
                warp_count,
            )?;
            for warp in &block.warps {
                for instruction in &warp.insts {
                    let class_name = schedule
                        .issue_slot(u64::from(instruction.pc))
                        .map(|slot| match slot.pipeline_class() {
                            PipelineClass::DispatchOnly => "dispatch",
                            PipelineClass::Branch { .. } => "branch",
                            PipelineClass::Barrier { .. } => "barrier",
                            PipelineClass::Tensor { .. } => "tensor",
                            PipelineClass::Sfu { .. } => "uniform_sfu",
                            PipelineClass::ScalarAlu { .. } => "scalar",
                        })
                        .unwrap_or("unknown");
                    let count = resource_class_counts
                        .entry(class_name.to_owned())
                        .or_default();
                    *count = count.checked_add(1).ok_or_else(|| ReplayError::FastFrontier {
                        detail: "resource-class observation count overflows u64".to_owned(),
                    })?;
                }
            }
        }
        let (frontier, memory, async_count, barriers) = fast_dynamic_sm_frontier(
            blocks,
            schedule,
            &masks,
            &params,
            extracted.shared_memory_residency,
        )?;
        summary.maximum_exact_final_cycle = summary.maximum_exact_final_cycle.max(frontier);
        memory_records = memory_records
            .checked_add(memory)
            .ok_or_else(|| ReplayError::FastFrontier {
                detail: "memory-service observation count overflows u64".to_owned(),
            })?;
        async_records = async_records
            .checked_add(async_count)
            .ok_or_else(|| ReplayError::FastFrontier {
                detail: "async-completion observation count overflows u64".to_owned(),
            })?;
        barrier_records = barrier_records
            .checked_add(barriers)
            .ok_or_else(|| ReplayError::FastFrontier {
                detail: "barrier-visibility observation count overflows u64".to_owned(),
            })?;
    }
    // Charge the footprint actually fetched by this launch, not all cold
    // branches present in the static function table.
    let line_cycles = extracted.front_end.compulsory_instruction_read_sectors;
    if alpha_config.front_end_line_cycles_denominator == 0 {
        return Err(ReplayError::FastFrontier {
            detail: "front-end line denominator is zero".to_owned(),
        });
    }
    let scaled_lines = line_cycles
        .checked_mul(alpha_config.front_end_line_cycles_numerator)
        .and_then(|value| value.checked_add(alpha_config.front_end_line_cycles_denominator - 1))
        .ok_or_else(|| ReplayError::FastFrontier {
            detail: "front-end line penalty overflows u64".to_owned(),
        })?
        / alpha_config.front_end_line_cycles_denominator;
    let front_end_penalty = alpha_config
        .front_end_base_cycles
        .checked_add(scaled_lines)
        .ok_or_else(|| ReplayError::FastFrontier {
            detail: "front-end penalty overflows u64".to_owned(),
        })?;
    summary.front_end_static_lines = line_cycles;
    summary.front_end_penalty_cycles = front_end_penalty;
    summary.maximum_summarized_final_cycle = summary
        .maximum_exact_final_cycle
        .checked_add(front_end_penalty)
        .ok_or_else(|| ReplayError::FastFrontier {
            detail: "front-end-adjusted SM120 stage cycles overflow u64".to_owned(),
        })?;
    let resource_class_total = resource_class_counts.values().try_fold(
        0_u64,
        |total, count| {
            total
                .checked_add(*count)
                .ok_or_else(|| ReplayError::FastFrontier {
                    detail: "resource-class total overflows u64".to_owned(),
                })
        },
    )?;
    summary
        .refusal_reasons
        .insert("stage_resource_classes".to_owned(), resource_class_total);
    summary
        .refusal_reasons
        .insert("memory_service_pending".to_owned(), memory_records);
    summary
        .refusal_reasons
        .insert("async_completion_pending".to_owned(), async_records);
    summary
        .refusal_reasons
        .insert("barrier_visibility_pending".to_owned(), barrier_records);
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
        // The SM-active observation domain excludes post-issue write
        // acknowledgements and service-only writeback/coherence events.  The
        // full modeled terminal frontier remains available through
        // `terminal_modeled_event_time` and the GPC projection; using it here
        // would let a post-exit acknowledgement move an SM-active endpoint.
        let terminal_lower_bound_cycles = self.sm_active_terminal_event_time();
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
    per_sm_instruction_read_sectors: BTreeMap<i32, u64>,
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
        OpClass::BlockBarrier | OpClass::Other | OpClass::MBarrierPhaseQuery => None,
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
    /// Upper bound of the `%smid` identifier domain.  PTX explicitly permits
    /// non-contiguous identifiers, so this is not used as a dense SM count.
    sm_id_upper_bound: u32,
    /// Maximum physical warp slot admitted by the runtime launch contract.
    /// This is deliberately sourced from `runtime_limits` when a card has
    /// that block; it is not an issue-width or scheduler hypothesis.
    max_resident_warps_per_sm: u32,
    /// Launch-envelope block/thread ceilings.  These are retained even when
    /// no issue model exists; a trace can be admitted for structural checks
    /// without inventing scheduler throughput.
    max_resident_blocks_per_sm: u32,
    max_threads_per_sm: u32,
    max_threads_per_block: u32,
    max_shared_bytes_per_sm: u64,
    max_shared_bytes_per_block: u64,
    max_shared_bytes_per_block_optin: u64,
}

impl PlacementConfig {
    fn from_arch_card(card: &ArchCard, expected_arch: ArchId) -> Result<Self, ReplayError> {
        let card_error = |source| ReplayError::ArchitectureCardPlacement { source };
        let arch = card.arch_id().map_err(card_error)?;
        require_replay_arch(expected_arch, arch, "architecture card")?;
        let sku = card.sku_id().map_err(card_error)?;
        let sm_count = card.sm_count().map_err(card_error)?;
        // Placement needs two architecture-local *launch-envelope* facts:
        // physical warp-slot capacity and per-SM shared-memory capacity.  A
        // Blackwell card intentionally leaves L1 geometry as a research gap,
        // so consulting `l1_geometry` here would incorrectly couple placement
        // to an unmeasured cache hypothesis.  Prefer the observed CUDA
        // runtime-limit block when present.  The legacy SM86 card predates
        // that block; retain its validated L1 shared-capacity field only as a
        // compatibility fallback, and never use it to admit SM120.
        let (
            max_resident_warps_per_sm,
            sm_id_upper_bound,
            max_resident_blocks_per_sm,
            max_threads_per_sm,
            max_threads_per_block,
            max_shared_bytes_per_sm,
            max_shared_bytes_per_block,
            max_shared_bytes_per_block_optin,
        ) = match card.require_runtime_limits() {
            Ok(limits) => (
                limits.max_resident_warps_per_sm,
                limits.sm_id_upper_bound,
                limits.max_blocks_per_sm,
                limits.max_threads_per_sm,
                limits.max_threads_per_block,
                limits.max_shared_memory_per_sm_bytes,
                limits.max_shared_memory_per_block_bytes,
                limits.max_shared_memory_per_block_optin_bytes,
            ),
            Err(
                gap @ CalibError::BlockResearchGap {
                    block: "runtime_limits",
                    ..
                },
            ) => {
                if arch != ArchId::Sm86 {
                    return Err(ReplayError::ArchitectureCardSharedMemoryCapacity {
                        arch,
                        source: gap,
                    });
                }
                let l1 = card.require_l1_geometry().map_err(|source| {
                    ReplayError::ArchitectureCardSharedMemoryCapacity { arch, source }
                })?;
                let (legacy_warps, legacy_blocks) = match card.require_occupancy_resources() {
                    Ok(occupancy) => (occupancy.max_resident_warps_per_sm, occupancy.max_blocks),
                    Err(CalibError::BlockResearchGap {
                        block: "occupancy", ..
                    }) => arch
                        .issue_hypotheses()
                        .require("legacy physical warp-slot capacity")
                        .map(|hypotheses| {
                            (
                                hypotheses.max_resident_warps_per_sm,
                                hypotheses.max_resident_blocks_per_sm,
                            )
                        })
                        .map_err(|source| ReplayError::WarpCapacityUnavailable { arch, source })?,
                    Err(source) => {
                        return Err(ReplayError::ArchitectureCardSharedMemoryCapacity {
                            arch,
                            source,
                        });
                    }
                };
                (
                    legacy_warps,
                    sm_count,
                    legacy_blocks,
                    legacy_warps.saturating_mul(32),
                    legacy_warps.saturating_mul(32),
                    l1.max_shared_bytes,
                    l1.max_shared_bytes,
                    l1.max_shared_bytes,
                )
            }
            Err(source) => {
                return Err(ReplayError::ArchitectureCardSharedMemoryCapacity { arch, source });
            }
        };
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
            sm_id_upper_bound,
            max_resident_warps_per_sm,
            max_resident_blocks_per_sm,
            max_threads_per_sm,
            max_threads_per_block,
            max_shared_bytes_per_sm,
            max_shared_bytes_per_block,
            max_shared_bytes_per_block_optin,
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
        if bytes_per_cta > self.max_shared_bytes_per_block_optin {
            return Err(ReplayError::CtaSharedMemoryBlockLimitExceeded {
                kernel_id: kernel.id,
                arch: self.arch,
                bytes_per_cta,
                max_bytes_per_block: self.max_shared_bytes_per_block_optin,
            });
        }
        let block = kernel
            .block_dim
            .as_ref()
            .filter(|dim| dim.x > 0 && dim.y > 0 && dim.z > 0)
            .map(|dim| {
                u64::try_from(dim.x)
                    .ok()
                    .and_then(|x| u64::try_from(dim.y).ok().and_then(|y| x.checked_mul(y)))
                    .and_then(|xy| u64::try_from(dim.z).ok().and_then(|z| xy.checked_mul(z)))
            })
            .flatten();
        if let Some(threads_per_cta) = block {
            if threads_per_cta > u64::from(self.max_threads_per_block) {
                return Err(ReplayError::CtaThreadCapacityExceeded {
                    kernel_id: kernel.id,
                    arch: self.arch,
                    threads_per_cta,
                    max_threads_per_block: self.max_threads_per_block,
                });
            }
        }
        Ok(SharedMemoryResidency::new(
            self.max_shared_bytes_per_sm,
            bytes_per_cta,
        ))
    }

    fn require_traced_sm(self, path: &Path, sm_id: i32) -> Result<(), ReplayError> {
        if sm_id < 0 || u32::try_from(sm_id).unwrap_or(u32::MAX) >= self.sm_id_upper_bound {
            return Err(ReplayError::SmIdOutOfRange {
                path: path.to_path_buf(),
                sm_id,
                arch: self.arch,
                sku: self.sku,
                sm_id_upper_bound: self.sm_id_upper_bound,
            });
        }
        Ok(())
    }

    #[cfg(feature = "campaign")]
    fn require_declared_sm(self, launch_index: u64, sm_id: u64) -> Result<(), ReplayError> {
        if sm_id >= u64::from(self.sm_id_upper_bound) {
            return Err(ReplayError::SamplingSmIdOutOfRange {
                launch_index,
                sm_id,
                arch: self.arch,
                sku: self.sku,
                sm_id_upper_bound: self.sm_id_upper_bound,
            });
        }
        Ok(())
    }
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

fn is_sync_sidecar_opcode(opcode: &str) -> bool {
    opcode.starts_with("SYNCS.") || opcode.starts_with("ARRIVES.")
}

/// Count distinct 32-byte instruction sectors assigned to one traced SM.
/// This is kept separate from the launch-wide front-end count so a later
/// effective observation envelope can use the same scope as active-lifetime
/// IPC. It is metadata only and never contributes replay cycles.
fn per_sm_instruction_sector_set(
    blocks: &[BlockReplay],
    function_unique_id: i32,
) -> BTreeSet<(i32, u32)> {
    let mut sectors = BTreeSet::new();
    for block in blocks {
        for warp in &block.warps {
            for inst in &warp.insts {
                sectors.insert((function_unique_id, inst.pc / 32));
            }
        }
    }
    sectors
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
    tma_shadow_l2: Option<L2Config>,
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
    // Physical slot validation is part of the traced launch envelope.  Do
    // not obtain it from `issue_hypotheses()`: SM120 intentionally has no
    // admitted scheduler hypothesis, while the 5090 runtime card does carry
    // a measured max-resident-warp limit.
    let max_resident_warps = placement.max_resident_warps_per_sm;
    let cluster = normalized_cluster_dim(loc.kernel(), schedule.arch)?;
    // sm_id → blocks (launch index kept for in-SM ordering).
    let mut per_sm_blocks: BTreeMap<i32, Vec<BlockReplay>> = BTreeMap::new();
    let mut seen_indices: BTreeSet<i64> = BTreeSet::new();
    let mut tma_dynamic_keys_seen: BTreeSet<TmaObservationKey> = BTreeSet::new();
    let mut traced_sm_by_launch_index: BTreeMap<i64, i32> = BTreeMap::new();
    let mut records_seen: u64 = 0;
    let mut instruction_sectors = BTreeSet::new();
    let mut instruction_lines = BTreeSet::new();
    let static_instruction_lines = schedule
        .by_pc
        .keys()
        .map(|pc| pc / 128)
        .collect::<BTreeSet<_>>();
    let mut instruction_sectors_by_sm: BTreeMap<i32, BTreeSet<(i32, u32)>> = BTreeMap::new();
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
        // SM120 flat mixed-async launches use the compact classifier but also
        // retain every observed global/local source span. Opaque TMA payload
        // bytes are reported separately and do not manufacture sector ids.
        sectors_decoded: decode_sectors || schedule.arch == ArchId::Sm120,
        ..TraceMetrics::default()
    };
    let sidecar_join_started = Instant::now();
    let observations = if schedule.arch == ArchId::Sm120 {
        td.async_sidecars()?
            .launch(loc.device_id(), loc.stream_id(), loc.kernel().id)
    } else {
        None
    };
    trace_metrics.async_relations = observations
        .filter(|o| !o.tma.is_empty() || !o.sync.is_empty())
        .map(|o| o.relations(cluster.into()));
    trace_metrics.async_obligations = observations
        .filter(|o| !o.tma.is_empty() || !o.sync.is_empty())
        .map(|o| sim_trace::obligations::ObligationFrontier::from_launch(o, cluster.into()).0);
    if let (Some(observations), Some(l2)) = (observations, tma_shadow_l2) {
        let stats = observe_tma_l2_shadow(&observations.tma, l2)?;
        trace_metrics.observe_tma_shadow_l2(
            stats.requests,
            stats.hits,
            stats.misses,
            stats.unavailable,
            stats.load,
            stats.store,
        )?;
    }
    let tma_observations = observations.map(|o| &o.tma);
    let sync_observations = observations.map(|o| &o.sync);
    trace_metrics.sidecar_join_wall_ns =
        u64::try_from(sidecar_join_started.elapsed().as_nanos()).unwrap_or(u64::MAX);

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
        let block_id = tb.block_id();
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
                let constant_addresses = inst.constant_memory_addresses();
                // Only active-lane address vectors contribute to the demand
                // interval. Bank/offset metadata without addresses can be a
                // predicated-off or incomplete record and is retained as
                // typed metadata, never fabricated into traffic.
                let constant_bank = inst.constant_memory_bank_offset().map(|(bank, _)| bank);
                trace_metrics.observe_constant_memory(constant_addresses, constant_bank)?;
                if constant_addresses.is_empty() && inst.constant_memory_metadata_present() {
                    trace_metrics.observe_constant_metadata_only()?;
                }
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
                instruction_lines.insert((inst.function_unique_id(), inst.pc() / 128));
                let is_tma_opcode = schedule.arch == ArchId::Sm120
                    && schedule
                        .issue_slot(u64::from(inst.pc()))
                        .is_some_and(|slot| {
                            matches!(opcode_head(slot.opcode_text()), "UTMALDG" | "UTMASTG" | "UBLKCP")
                        });
                if is_tma_opcode {
                    tma_dynamic_keys_seen.insert(TmaObservationKey {
                        cta: block_id,
                        warp: warp_id,
                        instruction: record_ix,
                        pc: inst.pc(),
                        function: inst.function_unique_id(),
                    });
                }
                if is_tma_opcode && (decode_sectors || preflight.is_none()) {
                    let key = TmaObservationKey {
                        cta: block_id,
                        warp: warp_id,
                        instruction: record_ix,
                        pc: inst.pc(),
                        function: inst.function_unique_id(),
                    };
                    trace_metrics
                        .observe_tma_sidecar_payload(tma_observations.and_then(|o| o.get(&key)))?;
                }
                if schedule.arch == ArchId::Sm120
                    && schedule
                        .issue_slot(u64::from(inst.pc()))
                        .is_some_and(|slot| is_sync_sidecar_opcode(slot.opcode_text()))
                {
                    let key = TmaObservationKey {
                        cta: block_id,
                        warp: warp_id,
                        instruction: record_ix,
                        pc: inst.pc(),
                        function: inst.function_unique_id(),
                    };
                    trace_metrics
                        .observe_sync_sidecar(sync_observations.and_then(|o| o.get(&key)))?;
                }
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
                        block_id,
                        tma_observations,
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
                        inst.pc(),
                        &path,
                        warp_id,
                        record_ix,
                        &mut sectors,
                        block_id,
                        tma_observations,
                    )?
                };
                if schedule.arch == ArchId::Sm86
                    && schedule
                        .issue_slot(u64::from(inst.pc()))
                        .is_some_and(|slot| is_static_constant_global_load(slot.opcode_text()))
                {
                    let span = match kind {
                        AccessKind::GlobalEligible { off, len, .. }
                        | AccessKind::GlobalBypass { off, len, .. }
                        | AccessKind::GlobalUntagged { off, len, .. } => Some((off, len)),
                        _ => None,
                    };
                    if let Some((off, len)) = span {
                        trace_metrics.observe_constant_global_load(
                            block_id,
                            warp_id,
                            &sectors,
                            off,
                            usize::from(len),
                            schedule
                                .issue_slot(u64::from(inst.pc()))
                                .map(|slot| slot.opcode_text())
                                .unwrap_or(""),
                            preflight
                                .as_ref()
                                .is_some_and(|memory| memory.memory.warp_uniform_address()),
                        )?;
                    }
                }
                if let Some((off, len)) = match kind {
                    AccessKind::GlobalBypass {
                        token: CacheOpToken::Atom,
                        off,
                        len,
                        ..
                    }
                    | AccessKind::GlobalEligible {
                        token: CacheOpToken::Atom,
                        off,
                        len,
                        ..
                    } => Some((off, len)),
                    _ => None,
                } {
                    trace_metrics.observe_atomic_conflict(&sectors, off, usize::from(len))?;
                }
                if matches!(kind, AccessKind::LocalLoad { .. } | AccessKind::LocalStore { .. }) {
                    trace_metrics.observe_local_width(inst.data_width(), inst.predicate_on_lanes())?;
                    trace_metrics.observe_local_payload_sectors(
                        kind,
                        inst.data_width(),
                        inst.predicate_on_lanes(),
                    )?;
                    if schedule.arch == ArchId::Sm86 {
                        if let Some(memory) = inst.memory() {
                            trace_metrics.observe_local_canonical_sectors(
                                kind,
                                memory.local_canonical_sectors(),
                            )?;
                            trace_metrics.observe_local_frame256_sectors(
                                kind,
                                memory.local_frame256_sectors(),
                            )?;
                        }
                    }
                }
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
                if let Some(slot) = schedule.issue_slot(u64::from(inst.pc())) {
                    trace_metrics
                        .operand_bank_pressure
                        .observe(slot.fixed_latency_operands(), schedule.arch)?;
                }
                if active_mask_present && predicate_mask_present {
                    trace_metrics.observe_coherence(
                        inst.sm_id(),
                        inst.predicate_on_lanes(),
                        kind,
                        &sectors,
                        inst.instrumentation_order(),
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
                    active_lane_mask: if inst.active_mask_present() {
                        inst.active_mask()
                    } else {
                        u32::MAX
                    },
                    lane_mask: if inst.predicate_mask_present() {
                        inst.predicate_on_mask()
                    } else if inst.active_mask_present() {
                        inst.active_mask()
                    } else {
                        u32::MAX
                    },
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
                warp_id,
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
        let block_replay = BlockReplay {
            launch_index,
            warps,
        };
        instruction_sectors_by_sm
            .entry(sm)
            .or_default()
            .extend(per_sm_instruction_sector_set(
                std::slice::from_ref(&block_replay),
                expected_fuid,
            ));
        per_sm_blocks.entry(sm).or_default().push(block_replay);
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
    if let Some(observations) = observations {
        let mut matched = 0_u64;
        let mut unmatched = 0_u64;
        for (key, observation) in &observations.tma {
            let has_address = (observation.is_load && !observation.source_sector_ids.is_empty())
                || (observation.is_store && !observation.destination_sector_ids.is_empty());
            if !has_address {
                continue;
            }
            if tma_dynamic_keys_seen.contains(key) {
                matched = matched.checked_add(1).ok_or(IssueError::CounterOverflow {
                    counter: "TMA shadow dynamic identity matches",
                })?;
            } else {
                unmatched = unmatched.checked_add(1).ok_or(IssueError::CounterOverflow {
                    counter: "TMA shadow dynamic identity gaps",
                })?;
            }
        }
        trace_metrics.observe_tma_shadow_dynamic_identity(matched, unmatched)?;
    }
    Ok(ExtractedKernel {
        per_sm_blocks,
        per_sm_instruction_read_sectors: instruction_sectors_by_sm
            .into_iter()
            .map(|(sm, sectors)| {
                u64::try_from(sectors.len()).map(|count| (sm, count)).map_err(|_| {
                    IssueError::CounterOverflow {
                        counter: "per-SM compulsory instruction-read sector cardinality",
                    }
                })
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?,
        records_seen,
        coverage,
        shared_memory_residency,
        front_end: FrontEndCounters {
            compulsory_instruction_read_sectors: u64::try_from(instruction_sectors.len()).map_err(
                |_| IssueError::CounterOverflow {
                    counter: "compulsory instruction-read sector cardinality",
                },
            )?,
            compulsory_instruction_line_read_sectors: u64::try_from(instruction_lines.len())
                .map_err(|_| IssueError::CounterOverflow {
                    counter: "compulsory instruction-read line cardinality",
                })?
                .checked_mul(4)
                .ok_or(IssueError::CounterOverflow {
                    counter: "compulsory instruction-read line sector cardinality",
                })?,
            static_instruction_line_read_sectors: u64::try_from(static_instruction_lines.len())
                .map_err(|_| IssueError::CounterOverflow {
                    counter: "static instruction-read line cardinality",
                })?
                .checked_mul(4)
                .ok_or(IssueError::CounterOverflow {
                    counter: "static instruction-read line sector cardinality",
                })?,
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
    Ok(extract_per_sm_blocks(td, loc, schedule, placement, false, None)?.coverage)
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
        coherence: Option<CoherenceConfig>,
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
    #[error("stream-L2 replay does not persist the opt-in coherence directory; use launch scope")]
    CoherenceRequiresLaunchScope,
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
            ReplayMemMode::L1SharedL2 { l2, coherence, .. } => {
                if coherence.is_some() {
                    return Err(ReplaySessionError::CoherenceRequiresLaunchScope);
                }
                l2
            }
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
        ReplayMemMode::L1SharedL2 {
            l1,
            l2,
            dram,
            coherence: _,
        } => {
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
    replay_kernel_with_schedule_and_l2_and_candidates_with_payload_events(
        td,
        loc,
        schedule,
        arch_card,
        params,
        mode,
        retained_l2,
        memory_service_candidates,
        memory_admission_candidates,
        memory_admission_feedback,
        &[],
    )
}

#[allow(clippy::too_many_arguments)]
fn replay_kernel_with_schedule_and_l2_and_candidates_with_payload_events(
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
    payload_events: &[L1PayloadAdmissionEvent],
) -> Result<KernelReplayOutcome, ReplayError> {
    if !payload_events.is_empty() && !matches!(mode, ReplayMemMode::L1SharedL2 { .. }) {
        return Err(ReplayError::PayloadAdmissionRequiresSharedL2);
    }
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
    // Ordinary launches can use the hierarchy classifier in flat mode.
    // Mixed async/TMA launches use the alpha compact classifier to preserve
    // opaque sidecar payloads without borrowing the SM86 completion protocol;
    // that classifier also retains all observed global/local sector spans.
    let alpha_flat_sector_decode = matches!(mode, ReplayMemMode::Flat)
        && params.resource_source() == IssueResourceSource::Sm120Alpha
        && !schedule.by_pc.values().any(|slot| {
            slot.ldgsts_l1_policy().is_some()
                || matches!(
                    slot.op_class(),
                    OpClass::AsyncCopyCommit | OpClass::AsyncCopyWait
                )
                || matches!(opcode_head(slot.opcode_text()), "UTMALDG" | "UTMASTG" | "UBLKCP")
        });
    let decode_sectors = !matches!(mode, ReplayMemMode::Flat) || alpha_flat_sector_decode;
    let ExtractedKernel {
        per_sm_blocks,
        per_sm_instruction_read_sectors,
        records_seen,
        coverage,
        shared_memory_residency,
        front_end,
        mut trace_metrics,
    } = extract_per_sm_blocks(
        td,
        loc,
        schedule,
        placement,
        decode_sectors,
        match mode {
            ReplayMemMode::L1SharedL2 { l2, .. } => Some(l2),
            _ => None,
        },
    )?;
    let instrumentation_order = observe_async_instrumentation_order(&per_sm_blocks, schedule)?;
    #[cfg(feature = "campaign")]
    if matches!(
        mode,
        ReplayMemMode::L1SharedL2 {
            dram: DramBw::CardedWriteback { .. },
            ..
        }
    ) && matches!(&coverage, ExtractedCoverage::Sampled(_))
    {
        return Err(ReplayError::WritebackRequiresFullCoverage);
    }
    if let ReplayMemMode::L1SharedL2 {
        coherence: Some(_), ..
    } = mode
    {
        let full_coverage = matches!(&coverage, ExtractedCoverage::Full(_));
        let active_masks_missing = trace_metrics.instructions.active_mask_missing_instructions;
        let predicate_masks_missing = trace_metrics
            .instructions
            .predicate_mask_missing_instructions;
        if !full_coverage
            || !trace_metrics.sectors_decoded
            || trace_metrics.coherence.unknown_sm_or_line_records != 0
            || active_masks_missing != 0
            || predicate_masks_missing != 0
        {
            return Err(ReplayError::CoherenceTopologyIncomplete {
                full_coverage,
                decoded: trace_metrics.sectors_decoded,
                unknown_records: trace_metrics.coherence.unknown_sm_or_line_records,
                active_masks_missing,
                predicate_masks_missing,
            });
        }
    }
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
                result.activity.front_end_instruction_read_sectors = per_sm_instruction_read_sectors
                    .get(sm)
                    .copied()
                    .unwrap_or(0);
                if async_copy_destination_fallback_sms.contains(sm) {
                    result.unmodeled_tails.ldgsts_dst_shared_timing = true;
                }
                if let Some(observation) = instrumentation_order.get(sm) {
                    result.readiness.async_instrumentation_order = Some(Box::new(*observation));
                }
                per_sm.insert(*sm, result);
            }
            for result in per_sm.values() {
                append_bounded_async_decision_snapshots(
                    &mut trace_metrics,
                    &result.readiness.async_decision_snapshots,
                    result.readiness.async_decision_snapshots_dropped,
                )?;
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
        ReplayMemMode::L1SharedL2 {
            l1,
            l2,
            dram,
            coherence,
        } => {
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
            let mut sys = if payload_events.is_empty() {
                simulate_kernel_shared_with_residency_and_candidates_with_coherence(
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
                    coherence,
                )?
            } else {
                simulate_kernel_shared_with_residency_and_candidates_with_coherence_and_payload_events(
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
                    coherence,
                    payload_events,
                )?
            };
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
            for (sm, sectors) in &per_sm_instruction_read_sectors {
                if let Some(result) = sys.per_sm.get_mut(sm) {
                    result.activity.front_end_instruction_read_sectors = *sectors;
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
            for result in sys.per_sm.values() {
                append_bounded_async_decision_snapshots(
                    &mut trace_metrics,
                    &result.readiness.async_decision_snapshots,
                    result.readiness.async_decision_snapshots_dropped,
                )?;
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
    for result in replay.per_sm.values() {
        if let Some(audit) = result.l1_merge_audit {
            let (global_hits, global_lookups) = result
                .l1
                .map(|counters| {
                    let origin = counters.origin(sim_mem::MemoryOrigin::Global);
                    (origin.hits, origin.lookup_sectors())
                })
                .unwrap_or((0, 0));
            trace_metrics.observe_l1_merge_audit(
                audit.same_request_merges,
                audit.older_pending_merges,
                audit.ordinary_same_request_merges,
                audit.ordinary_older_pending_merges,
                audit.async_same_request_merges,
                audit.async_older_pending_merges,
                audit.ordinary_global_older_pending_merges,
                audit.constant_global_older_pending_merges,
                audit.ordinary_global_same_request_merges,
                audit.constant_global_same_request_merges,
                audit.async_global_same_request_merges,
                audit.ordinary_global_hits,
                audit.async_global_hits,
                audit.ordinary_global_older_sector_pending_merges,
                audit.constant_global_older_sector_pending_merges,
                audit.async_global_older_sector_pending_merges,
                audit.unrequested_subline_miss_sectors,
                audit.wavefront_mask_lookup_hits,
                audit.wavefront_mask_lookup_misses,
                audit.constant_global_hits,
                audit.constant_global_requests,
                global_hits,
                global_lookups,
            )?;
        }
    }
    replay.front_end = front_end;
    replay.trace_metrics = trace_metrics;
    // A drained hypothetical service queue is not a completion relation for
    // native stores/atomics. Keep the system's unknown tails: one completion
    // timestamp or the absence of traced cross-SM sharing cannot prove that
    // all write/visibility obligations were represented.
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

/// Replay one launch with explicit producer-supplied value-bearing L1 events.
/// Each event carries its simulator-cycle timestamp and a residency witness;
/// this API never converts NCU/CUPTI/device time or infers a cycle from payload
/// sequence. Ordinary replay remains unchanged when this entry point is not
/// selected.
pub fn replay_trace_kernel_with_l1_payload_events(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedules: &BTreeMap<i32, FunctionSchedule>,
    arch_card: &ArchCard,
    params: &IssueModelParams,
    mode: ReplayMemMode,
    payload_events: &[L1PayloadAdmissionEvent],
) -> Result<KernelReplayOutcome, ReplayError> {
    if !payload_events.is_empty() && !matches!(mode, ReplayMemMode::L1SharedL2 { .. }) {
        return Err(ReplayError::PayloadAdmissionRequiresSharedL2);
    }
    let schedule = lookup_schedule(schedules, loc)?;
    let locations = td.dynamic_memory_join_locations(loc)?;
    let mut bound_events = payload_events.to_vec();
    for (index, event) in bound_events.iter_mut().enumerate() {
        let Some(identity) = event.identity else {
            return Err(ReplayError::PayloadAdmissionIdentityMissing { index });
        };
        let Some(location) = locations.get(&identity) else {
            return Err(ReplayError::PayloadAdmissionIdentityNotObserved { index });
        };
        if location.sm_id != Some(event.sm_id) {
            return Err(ReplayError::PayloadAdmissionSmMismatch {
                index,
                event_sm: event.sm_id,
                trace_sm: location.sm_id,
            });
        }
        if location.access_kind != Some(pb::MemoryOpAccessKind::Store) {
            return Err(ReplayError::PayloadAdmissionNotStore {
                index,
                access_kind: location.access_kind,
            });
        }
        let Some(warp_slot) = location.warp_slot else {
            return Err(ReplayError::PayloadAdmissionWriterPlacementMissing { index });
        };
        event.writer = Some(PayloadStoreSite {
            launch_index: location.launch_index,
            warp_slot,
            instruction_index: usize::try_from(location.instruction_index)
                .map_err(|_| ReplayError::PayloadAdmissionWriterPlacementMissing { index })?,
        });
    }
    replay_kernel_with_schedule_and_l2_and_candidates_with_payload_events(
        td,
        loc,
        schedule,
        arch_card,
        params,
        mode,
        None,
        &[],
        &[],
        None,
        &bound_events,
    )
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
    let alpha_sm120 = arch == ArchId::Sm120;
    if arch != ArchId::Sm86 && !alpha_sm120 {
        return Err(ReplayError::UnsupportedMemoryPath { arch, class });
    }
    let slot = schedule
        .by_pc
        .get(&u64::from(pc))
        .ok_or(ReplayError::UnsupportedMemoryPath { arch, class })?;
    let head = opcode_head(&slot.opcode_text);
    if alpha_sm120 && class == pb::MemoryOpClass::AsyncCopy && head == "LDGSTS" {
        // Alpha treats an observed async/TMA source as a flat global demand;
        // no SM86 LDGSTS completion or cache route is attached.
        return Ok(());
    }
    let valid = match (route, class) {
        (ReplayMemoryRoute::Global, pb::MemoryOpClass::Atomic) => match slot.mem_role {
            MemRole::BypassLoad(CacheOpToken::Atom) => {
                matches!(head, "ATOM" | "ATOMG") || (alpha_sm120 && head == "REDG")
            }
            MemRole::Reduction => matches!(head, "RED" | "REDG"),
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
        // The SM120 alpha path admits newly observed Blackwell storage-family
        // spellings as untagged global traffic.  They retain their dynamic
        // address/sector stream and flat latency, but do not borrow a guessed
        // cache operator or writeback class.  Ordinary SM120 replay remains
        // fail-closed because this branch is keyed by the alpha-tagged
        // deployment schedule.
        if arch != ArchId::Sm86 {
            return Ok(MemRole::None);
        }
        return Err(ReplayError::UnknownGlobalOpcode {
            arch,
            pc,
            opcode: opcode_head(&slot.opcode_text).to_owned(),
        });
    }
    Ok(slot.mem_role)
}

fn is_static_constant_global_load(opcode_text: &str) -> bool {
    let opcode = opcode_text.split_ascii_whitespace().next().unwrap_or("");
    opcode.split('.').any(|modifier| modifier == "CONSTANT")
        && matches!(opcode_head(opcode_text), "LDG" | "LD" | "LDC" | "LDL")
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StaticMemoryRoute {
    Modeled {
        route: ReplayMemoryRoute,
        class: pb::MemoryOpClass,
    },
    /// Unqualified ATOM/RED use PTX generic addressing. Their dynamic address
    /// operand can resolve to either global or shared memory, so the static
    /// mnemonic supplies the operation class but not one fixed route.
    GenericAtomic {
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

/// Return whether a Blackwell bulk-copy spelling is the modeled
/// global-to-shared producer form.  The qualifier order is part of the
/// instruction contract: `UBLKCP.S.G` names a shared destination and global
/// source.  The legacy `UTMASTG` store form remains a supported effective
/// store-side trace path; it is not confused with the load-side direction
/// check below.
fn is_modeled_tma_load_opcode(opcode_text: &str) -> bool {
    let token = opcode_text
        .split_whitespace()
        .next()
        .unwrap_or("");
    let mut fields = token.split('.');
    match fields.next().unwrap_or("") {
        "UTMALDG" | "UTMASTG" => true,
        "UBLKCP" => {
            matches!(fields.next(), Some("S")) && matches!(fields.next(), Some("G"))
        }
        _ => false,
    }
}

fn is_tma_submission_opcode(opcode_text: &str) -> bool {
    matches!(opcode_head(opcode_text), "UTMALDG" | "UTMASTG" | "UBLKCP")
}

fn validate_tma_load_direction(
    schedule: &FunctionSchedule,
    arch: ArchId,
    pc: u32,
) -> Result<(), ReplayError> {
    let Some(slot) = schedule.by_pc.get(&u64::from(pc)) else {
        return Ok(());
    };
    if is_tma_submission_opcode(&slot.opcode_text)
        && !is_modeled_tma_load_opcode(&slot.opcode_text)
    {
        return Err(ReplayError::UnsupportedTmaDirection {
            arch,
            pc,
            opcode: slot.opcode_text.clone(),
        });
    }
    Ok(())
}

/// Space-coded SASS heads whose storage class is part of the mnemonic rather
/// than inferred from the dynamic record. ATOM and RED are generic, while
/// ATOMG/ATOMS constrain the route; LDGSTS uses its traced Global source. REDG,
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
        "ATOM" | "RED" => Ok(Some(StaticMemoryRoute::GenericAtomic {
            class: pb::MemoryOpClass::Atomic,
        })),
        "ATOMG" => Ok(Some(StaticMemoryRoute::Modeled {
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
    validate_tma_load_direction(schedule, schedule.arch, pc)?;
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
        Some(StaticMemoryRoute::GenericAtomic { class: expected }) => {
            if expected != meta.op_class {
                return Err(contextualize(MemoryRouteError::StaticMemoryClassMismatch {
                    expected,
                    got: meta.op_class,
                }));
            }
            if matches!(route, ReplayMemoryRoute::Global | ReplayMemoryRoute::Shared) {
                Ok(())
            } else {
                Err(contextualize(
                    MemoryRouteError::StaticGenericAtomicRouteInvalid {
                        got: route.canonical_space(),
                    },
                ))
            }
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

/// Flat mode validates the same static operation role as hierarchy replay.  The
/// legacy path deliberately carries an empty sector span, but the SM120 alpha
/// path may retain the already validated NVBit span so its logical global
/// service tail sees the captured request width even when a launch also
/// contains an opaque async/TMA record.
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

fn classify_flat_global_sectors(
    role: MemRole,
    warp: &ValidatedWarp,
    memory: &ValidatedMemoryInstruction,
    sectors: &mut Vec<i64>,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<AccessKind, ReplayError> {
    let sector_scratch = match warp.sector_plan(memory) {
        ValidatedSectorPlan::Sectors(sectors) => sectors,
        ValidatedSectorPlan::Error { error, .. } => {
            return Err(deferred_sector_error(error, path, warp_id, record_ix));
        }
    };
    append_staged_global_sectors(role, sectors, sector_scratch, path, warp_id, record_ix)
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
    pc: u32,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
    sector_arena: &mut Vec<i64>,
    block_id: [u32; 3],
    tma_observations: Option<&BTreeMap<TmaObservationKey, TmaObservation>>,
) -> Result<AccessKind, ReplayError> {
    let Some(preflight) = preflight else {
        if arch == ArchId::Sm120
            && schedule.issue_slot(u64::from(pc)).is_some_and(|slot| {
                let head = opcode_head(&slot.opcode_text);
                matches!(head, "UTMALDG" | "UTMASTG" | "UBLKCP")
            })
        {
            validate_tma_load_direction(schedule, arch, pc)?;
            // TMA operands are carried by tensor-map/mbarrier sidecars rather
            // than ordinary dynamic address operands in the compact trace.
            // Consume the matching per-instruction payload observation when
            // present. A missing sidecar is explicitly unavailable and carries
            // zero logical completion work; it no longer fabricates 8 KiB.
            let key = TmaObservationKey {
                cta: block_id,
                warp: warp_id,
                instruction: record_ix,
                pc,
                function: warp.function_unique_id(),
            };
            let payload_bytes = tma_observations
                .and_then(|observations| observations.get(&key))
                .and_then(|observation| observation.bytes)
                .unwrap_or(0);
            let payload_work = u32::try_from(payload_bytes).map_err(|_| {
                ReplayError::AsyncCopyCompletionWorkOutOfRange {
                    path: path.to_path_buf(),
                    warp_id,
                    record_ix,
                    completion_work: payload_bytes,
                    max_completion_work: ASYNC_COPY_COMPLETION_VALUE_MASK,
                }
            })?;
            let completion_work = encode_async_copy_payload_work(payload_work).ok_or(
                ReplayError::AsyncCopyCompletionWorkOutOfRange {
                    path: path.to_path_buf(),
                    warp_id,
                    record_ix,
                    completion_work: payload_bytes,
                    max_completion_work: ASYNC_COPY_COMPLETION_VALUE_MASK,
                },
            )?;
            return Ok(AccessKind::GlobalAsyncCopy {
                off: 0,
                len: 0,
                completion_work,
                // TMA tensor-map transfers use the distinct logical bypass
                // service arm in full replay. This is a calibration-domain
                // route, not a physical cache policy claim.
                l1_policy: LdgstsL1Policy::BypassL1,
            });
        }
        return Ok(AccessKind::NonMemory);
    };
    if !preflight.memory.has_effective_lanes() {
        return Ok(AccessKind::PredicatedOffMemory);
    }
    let meta = preflight.memory.replay_meta();
    match (preflight.route, meta.op_class) {
        (ReplayMemoryRoute::Shared, _) => Ok(compact_shared_kind(preflight.memory)),
        (ReplayMemoryRoute::Local, pb::MemoryOpClass::Local) => {
            let Some(direction) = require_known_local_direction(schedule, arch, preflight.pc)?
            else {
                return Ok(AccessKind::NonMemory);
            };
            if arch == ArchId::Sm120 {
                let source_sectors = match warp.sector_plan(preflight.memory) {
                    ValidatedSectorPlan::Sectors(sectors) => sectors,
                    ValidatedSectorPlan::Error { error, .. } => {
                        return Err(deferred_sector_error(error, path, warp_id, record_ix));
                    }
                };
                append_staged_local_sectors(
                    direction, sector_arena, source_sectors, path, warp_id, record_ix,
                )
            } else {
                Ok(match direction {
                    LocalDirection::Load => AccessKind::LocalLoad { off: 0, len: 0 },
                    LocalDirection::Store => AccessKind::LocalStore { off: 0, len: 0 },
                })
            }
        }
        (ReplayMemoryRoute::Local, class) => {
            Err(ReplayError::UnsupportedMemoryPath { arch, class })
        }
        (ReplayMemoryRoute::Unsupported, class) => {
            Err(ReplayError::UnsupportedMemoryPath { arch, class })
        }
        (ReplayMemoryRoute::Global, pb::MemoryOpClass::Global) => {
            let role = require_known_global_role(schedule, arch, preflight.pc)?;
            if arch == ArchId::Sm120 {
                classify_flat_global_sectors(
                    role,
                    warp,
                    preflight.memory,
                    sector_arena,
                    path,
                    warp_id,
                    record_ix,
                )
            } else {
                Ok(flat_global_kind(role))
            }
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
            if arch == ArchId::Sm120 {
                classify_flat_global_sectors(
                    role,
                    warp,
                    preflight.memory,
                    sector_arena,
                    path,
                    warp_id,
                    record_ix,
                )
            } else {
                Ok(flat_global_kind(role))
            }
        }
        (ReplayMemoryRoute::Global, pb::MemoryOpClass::AsyncCopy) => {
            if arch == ArchId::Sm120 {
                // Full-replay-only alpha route: preserve an observed async
                // source as a typed completion token instead of collapsing it
                // into an ordinary global instruction.  A compact trace may
                // lack payload/sector evidence; that is zero logical work with
                // an inactive token, not an invented transaction.  This is not
                // a native TMA engine timestamp or queue model.
                let source_sectors = match warp.sector_plan(preflight.memory) {
                    ValidatedSectorPlan::Sectors(sectors) => sectors,
                    ValidatedSectorPlan::Error { error, .. } => {
                        return Err(deferred_sector_error(error, path, warp_id, record_ix));
                    }
                };
                let payload_bytes = warp.effective_data_bytes(preflight.memory);
                let sector_bytes = u64::try_from(source_sectors.len())
                    .ok()
                    .and_then(|count| count.checked_mul(32))
                    .unwrap_or(0);
                let logical_bytes = payload_bytes.max(sector_bytes);
                let completion_work = u32::try_from(logical_bytes)
                    .ok()
                    .and_then(encode_async_copy_payload_work)
                    .ok_or(ReplayError::AsyncCopyCompletionWorkOutOfRange {
                        path: path.to_path_buf(),
                        warp_id,
                        record_ix,
                        completion_work: logical_bytes,
                        max_completion_work: ASYNC_COPY_COMPLETION_VALUE_MASK,
                    })?;
                let kind = checked_staged_async_copy_kind(
                    LdgstsL1Policy::CacheAtAllLevels,
                    sector_arena.len(),
                    source_sectors.len(),
                    completion_work,
                )
                .ok_or(ReplayError::SectorCountOverflow {
                    path: path.to_path_buf(),
                    warp_id,
                    record_ix,
                    sectors: source_sectors.len(),
                })?;
                return append_staged_sectors(
                    Some(kind),
                    sector_arena,
                    source_sectors,
                    path,
                    warp_id,
                    record_ix,
                );
            }
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
    block_id: [u32; 3],
    tma_observations: Option<&BTreeMap<TmaObservationKey, TmaObservation>>,
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
        (ReplayMemoryRoute::Global, class @ pb::MemoryOpClass::Atomic) => {
            require_sm86_special_role(schedule, arch, preflight.pc, preflight.route, class)?;
            StagedHierarchyRole::Global(require_known_global_role(schedule, arch, preflight.pc)?)
        }
        (ReplayMemoryRoute::Global, pb::MemoryOpClass::AsyncCopy) if arch == ArchId::Sm120 => {
            // SM120 async source/destination semantics are admitted only into
            // the explicit alpha cache scenario.  The route remains typed as
            // an async token; it is never downgraded to ordinary global timing.
            StagedHierarchyRole::Global(MemRole::BypassLoad(CacheOpToken::Ldgsts))
        }
        (ReplayMemoryRoute::Global, class @ pb::MemoryOpClass::AsyncCopy) => {
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
            let tma = arch == ArchId::Sm120
                && schedule
                    .issue_slot(u64::from(preflight.pc))
                    .is_some_and(|slot| {
                        matches!(opcode_head(slot.opcode_text()), "UTMALDG" | "UTMASTG" | "UBLKCP")
                    });
            let l1_policy = if tma {
                LdgstsL1Policy::BypassL1
            } else if arch == ArchId::Sm120 {
                let opcode = schedule
                    .issue_slot(u64::from(preflight.pc))
                    .map(|slot| slot.opcode_text())
                    .unwrap_or("");
                if opcode_has_qualifier(opcode, "BYPASS") {
                    LdgstsL1Policy::BypassL1
                } else {
                    LdgstsL1Policy::CacheAtAllLevels
                }
            } else {
                require_ldgsts_l1_policy(schedule, arch, preflight.pc)?
            };
            let completion_work_override = if arch == ArchId::Sm120 {
                let key = TmaObservationKey {
                    cta: block_id,
                    warp: warp_id,
                    instruction: record_ix,
                    pc: preflight.pc,
                    function: warp.function_unique_id(),
                };
                let payload_bytes = if tma {
                    tma_observations
                        .and_then(|observations| observations.get(&key))
                        .and_then(|observation| observation.bytes)
                        .ok_or(ReplayError::AsyncCopyPayloadMismatch {
                            path: path.to_path_buf(),
                            warp_id,
                            record_ix,
                            source_active: true,
                            sectors: staged.len(),
                            payload_bytes: 0,
                        })?
                } else {
                    warp.effective_data_bytes(preflight.memory).max(
                        u64::try_from(staged.len())
                            .unwrap_or(0)
                            .saturating_mul(32),
                    )
                };
                Some(encode_async_copy_payload_work(
                    u32::try_from(payload_bytes).map_err(|_| {
                        ReplayError::AsyncCopyCompletionWorkOutOfRange {
                            path: path.to_path_buf(),
                            warp_id,
                            record_ix,
                            completion_work: payload_bytes,
                            max_completion_work: ASYNC_COPY_COMPLETION_VALUE_MASK,
                        }
                    })?,
                ).ok_or(ReplayError::AsyncCopyCompletionWorkOutOfRange {
                    path: path.to_path_buf(),
                    warp_id,
                    record_ix,
                    completion_work: payload_bytes,
                    max_completion_work: ASYNC_COPY_COMPLETION_VALUE_MASK,
                })?)
            } else {
                None
            };
            append_staged_async_copy_sectors(
                arch,
                l1_policy,
                if tma {
                    0
                } else {
                    warp.effective_data_bytes(preflight.memory).max(
                        u64::try_from(staged.len())
                            .unwrap_or(0)
                            .saturating_mul(32),
                    )
                },
                warp.ldgsts_destination_topology_128b_32x4(preflight.memory),
                warp.ldgsts_source_cohorts_128b_32x4(preflight.memory),
                warp.ldgsts_source_sector_cohort_masks_128b_32x4(preflight.memory),
                sectors,
                async_copy_cohorts,
                staged,
                path,
                warp_id,
                record_ix,
                completion_work_override,
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
    // The flat SM120 alpha path returns before this helper and charges an
    // explicit global-demand approximation.  Any caller that reaches the
    // compact async completion machinery must therefore be an architecture
    // with an admitted LDGSTS protocol; never map SM120 to SM86 here.
    if arch == ArchId::Sm120 {
        return Err(ReplayError::UnsupportedMemoryPath {
            arch,
            class: pb::MemoryOpClass::AsyncCopy,
        });
    }
    let timing_arch = arch;
    let throughput = timing_arch
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
                let timing = timing_arch
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
    completion_work_override: Option<u32>,
) -> Result<AccessKind, ReplayError> {
    let start = sectors.len();
    let count = sector_scratch.len();
    let source_active = count != 0;
    if completion_work_override.is_none() && source_active != (payload_bytes != 0) {
        return Err(ReplayError::AsyncCopyPayloadMismatch {
            path: path.to_path_buf(),
            warp_id,
            record_ix,
            source_active,
            sectors: count,
            payload_bytes,
        });
    }
    let completion_work = completion_work_override.map_or_else(
        || {
            project_async_copy_completion_work(
                arch,
                l1_policy,
                payload_bytes,
                destination_topology,
                path,
                warp_id,
                record_ix,
            )
        },
        Ok,
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
