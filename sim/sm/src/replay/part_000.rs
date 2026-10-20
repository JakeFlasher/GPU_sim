// Lexical implementation section 1/4; included by the parent facade.
use std::collections::{BTreeMap, BTreeSet};

use std::path::{Path, PathBuf};

use std::time::Instant;

use serde::Serialize;

use sim_trace::async_sidecars::{
    LaunchObservations, ObservationKey as TmaObservationKey, TmaObservation,
};

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
    IssueModelParams, IssueResourceSource, MAX_ASYNC_DECISION_SNAPSHOTS, MemoryAdmissionCandidate,
    MemoryServiceCandidate, PayloadStoreSite, ReplayInst, SharedMemoryResidency, SmSimResult,
    TailFlags, WarpReplay, simulate_sm_with_residency, stateful_write_lookup_observer_enabled,
    validate_memory_admission_candidates, validate_memory_service_candidates,
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
    let available =
        MAX_ASYNC_DECISION_SNAPSHOTS.saturating_sub(metrics.async_decision_snapshots.len());
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

/// One address-derived event for a diagnostic joint L2 shadow. `order` is
/// the producer's instrumentation token, never a cycle or engine timestamp.
#[derive(Clone, Copy)]
struct JointL2ShadowEvent {
    order: u64,
    sequence: u64,
    sector_id: i64,
    class: L2OpClass,
    tma_load: bool,
    ordinary_read: bool,
}

#[derive(Default)]
struct JointL2ShadowStats {
    requests: u64,
    hits: u64,
    misses: u64,
    tma_load: [u64; 3],
    ordinary_read: [u64; 3],
    order_unavailable: u64,
    events_dropped: u64,
}

const MAX_JOINT_L2_SHADOW_EVENTS: usize = 2_000_000;

fn observe_joint_l2_shadow(
    mut events: Vec<JointL2ShadowEvent>,
    config: L2Config,
    order_unavailable: u64,
    events_dropped: u64,
) -> Result<JointL2ShadowStats, ReplayError> {
    events.sort_unstable_by_key(|event| (event.order, event.sequence));
    let mut shadow = L2Model::new(config)?;
    let mut stats = JointL2ShadowStats {
        order_unavailable,
        events_dropped,
        ..JointL2ShadowStats::default()
    };
    for event in events {
        let line_id = event.sector_id.div_euclid(4);
        let sector = u8::try_from(event.sector_id.rem_euclid(4)).map_err(|_| {
            ReplayError::System(SystemError::L2(sim_mem::L2AccessError::SectorOutOfRange {
                sector: u8::MAX,
            }))
        })?;
        let outcome = shadow
            .access(line_id, sector, event.class)
            .map_err(|error| ReplayError::System(SystemError::L2(error)))?;
        stats.requests = stats
            .requests
            .checked_add(1)
            .ok_or(IssueError::CounterOverflow {
                counter: "joint TMA/L2 shadow requests",
            })?;
        let hit = u64::from(outcome.counted_hit);
        let miss = u64::from(!outcome.counted_hit);
        stats.hits = stats
            .hits
            .checked_add(hit)
            .ok_or(IssueError::CounterOverflow {
                counter: "joint TMA/L2 shadow hits",
            })?;
        stats.misses = stats
            .misses
            .checked_add(miss)
            .ok_or(IssueError::CounterOverflow {
                counter: "joint TMA/L2 shadow misses",
            })?;
        if event.tma_load {
            stats.tma_load[0] = stats.tma_load[0].saturating_add(1);
            stats.tma_load[1] = stats.tma_load[1].saturating_add(hit);
            stats.tma_load[2] = stats.tma_load[2].saturating_add(miss);
        }
        if event.ordinary_read {
            stats.ordinary_read[0] = stats.ordinary_read[0].saturating_add(1);
            stats.ordinary_read[1] = stats.ordinary_read[1].saturating_add(hit);
            stats.ordinary_read[2] = stats.ordinary_read[2].saturating_add(miss);
        }
    }
    Ok(stats)
}

fn joint_l2_ordinary_span(kind: AccessKind) -> Option<(usize, usize, L2OpClass, bool)> {
    match kind {
        AccessKind::GlobalEligible { off, len, .. }
        | AccessKind::GlobalUntagged { off, len }
        | AccessKind::GlobalAsyncCopy {
            off,
            len,
            l1_policy: LdgstsL1Policy::CacheAtAllLevels,
            ..
        } => Some((off, usize::from(len), L2OpClass::Read, true)),
        AccessKind::GlobalBypass {
            off, len, token, ..
        } => Some((
            off,
            usize::from(len),
            if token == CacheOpToken::Atom {
                L2OpClass::WriteAtomic
            } else {
                L2OpClass::Read
            },
            token != CacheOpToken::Atom,
        )),
        AccessKind::GlobalStore { off, len } | AccessKind::GlobalReduction { off, len, .. } => {
            Some((off, usize::from(len), L2OpClass::WriteAtomic, false))
        }
        AccessKind::GlobalAsyncCopy { .. }
        | AccessKind::LocalLoad { .. }
        | AccessKind::LocalStore { .. }
        | AccessKind::Shared { .. }
        | AccessKind::NonMemory
        | AccessKind::PredicatedOffMemory => None,
    }
}

fn observe_tma_l2_shadow(
    observations: &BTreeMap<TmaObservationKey, TmaObservation>,
    config: L2Config,
) -> Result<TmaShadowL2Stats, ReplayError> {
    observe_tma_l2_shadow_iter(observations.values(), config)
}

fn tma_sidecar_reuse_summary(
    observations: &LaunchObservations,
) -> (
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
) {
    let mut all_sectors = BTreeSet::new();
    let mut all_lines = BTreeSet::new();
    let mut sum_record_unique_sectors = 0_u64;
    let mut sum_record_unique_lines = 0_u64;
    let mut intra_record_duplicate_refs = 0_u64;
    let mut load_sum_sectors = 0_u64;
    let mut store_sum_sectors = 0_u64;
    let mut load_sum_lines = 0_u64;
    let mut store_sum_lines = 0_u64;
    let mut load_sectors = BTreeSet::new();
    let mut store_sectors = BTreeSet::new();
    let mut load_lines = BTreeSet::new();
    let mut store_lines = BTreeSet::new();
    for observation in observations.tma.values() {
        let ids = if observation.is_load {
            &observation.source_sector_ids
        } else if observation.is_store {
            &observation.destination_sector_ids
        } else {
            continue;
        };
        sum_record_unique_sectors =
            sum_record_unique_sectors.saturating_add(observation.diagnostic.unique_sector_count);
        sum_record_unique_lines =
            sum_record_unique_lines.saturating_add(observation.diagnostic.unique_line_count);
        intra_record_duplicate_refs = intra_record_duplicate_refs
            .saturating_add(observation.diagnostic.intra_record_duplicate_sector_refs);
        if observation.is_load {
            load_sum_sectors =
                load_sum_sectors.saturating_add(observation.diagnostic.unique_sector_count);
            load_sum_lines =
                load_sum_lines.saturating_add(observation.diagnostic.unique_line_count);
        } else {
            store_sum_sectors =
                store_sum_sectors.saturating_add(observation.diagnostic.unique_sector_count);
            store_sum_lines =
                store_sum_lines.saturating_add(observation.diagnostic.unique_line_count);
        }
        for &sector in ids.iter() {
            all_sectors.insert(sector);
            all_lines.insert(sector.div_euclid(4));
            if observation.is_load {
                load_sectors.insert(sector);
                load_lines.insert(sector.div_euclid(4));
            } else {
                store_sectors.insert(sector);
                store_lines.insert(sector.div_euclid(4));
            }
        }
    }
    let cross_sectors = load_sectors.intersection(&store_sectors).count();
    let cross_lines = load_lines.intersection(&store_lines).count();
    (
        sum_record_unique_sectors,
        sum_record_unique_lines,
        u64::try_from(all_sectors.len()).unwrap_or(u64::MAX),
        u64::try_from(all_lines.len()).unwrap_or(u64::MAX),
        intra_record_duplicate_refs,
        load_sum_sectors,
        store_sum_sectors,
        load_sum_lines,
        store_sum_lines,
        u64::try_from(load_sectors.len()).unwrap_or(u64::MAX),
        u64::try_from(store_sectors.len()).unwrap_or(u64::MAX),
        u64::try_from(cross_sectors).unwrap_or(u64::MAX),
        u64::try_from(cross_lines).unwrap_or(u64::MAX),
    )
}

fn async_clock_sandwich_summary(
    observations: &LaunchObservations,
    relations: &sim_trace::async_sidecars::AsyncRelations,
) -> (u64, u64, u64, u64) {
    let mut pairs = 0_u64;
    let mut invalid = 0_u64;
    let mut lower_sum = 0_u64;
    let mut upper_sum = 0_u64;
    for relation in &relations.relation_samples {
        let (Some(issue), Some(acquire)) = (
            relation.producer_issue_clock,
            relation.wait_visibility_clock,
        ) else {
            continue;
        };
        if acquire < issue {
            invalid = invalid.saturating_add(1);
            continue;
        }
        let last_false = observations
            .sync
            .get(&relation.wait)
            .and_then(|observation| observation.last_unsuccessful_query_clock);
        if last_false.is_some_and(|clock| clock < issue || clock > acquire) {
            invalid = invalid.saturating_add(1);
            continue;
        }
        let lower = last_false.map_or(0, |clock| clock - issue);
        let upper = acquire - issue;
        pairs = pairs.saturating_add(1);
        lower_sum = lower_sum.saturating_add(lower);
        upper_sum = upper_sum.saturating_add(upper);
    }
    (pairs, invalid, lower_sum, upper_sum)
}

/// Run the same conditional TMA/L2 observer in the sidecar's first-seen
/// stream order.  This is diagnostic only: the lexical keyed-map projection
/// remains the scored/default endpoint because sidecar order is an emission
/// sequence, not a documented engine chronology.
fn observe_tma_l2_shadow_ordered(
    observations: &LaunchObservations,
    config: L2Config,
) -> Result<TmaShadowL2Stats, ReplayError> {
    let iter = if observations.tma_order.is_empty() {
        observations.tma.values().collect::<Vec<_>>()
    } else {
        observations
            .tma_order
            .iter()
            .filter_map(|key| observations.tma.get(key))
            .collect::<Vec<_>>()
    };
    observe_tma_l2_shadow_iter(iter.into_iter(), config)
}

fn observe_tma_l2_shadow_iter<'a>(
    observations: impl Iterator<Item = &'a TmaObservation>,
    config: L2Config,
) -> Result<TmaShadowL2Stats, ReplayError> {
    let mut shadow = L2Model::new(config)?;
    let mut stats = TmaShadowL2Stats::default();
    for observation in observations {
        let (sector_ids, class) = if observation.is_load {
            (&observation.source_sector_ids, L2OpClass::Read)
        } else if observation.is_store {
            (&observation.destination_sector_ids, L2OpClass::WriteAtomic)
        } else {
            continue;
        };
        if sector_ids.is_empty() {
            stats.unavailable =
                stats
                    .unavailable
                    .checked_add(1)
                    .ok_or(IssueError::CounterOverflow {
                        counter: "TMA shadow L2 unavailable address records",
                    })?;
            let lane = if observation.is_load {
                &mut stats.load
            } else {
                &mut stats.store
            };
            lane[3] = lane[3].checked_add(1).ok_or(IssueError::CounterOverflow {
                counter: "TMA shadow operation-class unavailable records",
            })?;
            continue;
        }
        for &sector_id in sector_ids.iter() {
            let line_id = sector_id.div_euclid(4);
            let sector = u8::try_from(sector_id.rem_euclid(4)).map_err(|_| {
                ReplayError::System(SystemError::L2(sim_mem::L2AccessError::SectorOutOfRange {
                    sector: u8::MAX,
                }))
            })?;
            let outcome = shadow
                .access(line_id, sector, class)
                .map_err(|error| ReplayError::System(SystemError::L2(error)))?;
            stats.requests = stats
                .requests
                .checked_add(1)
                .ok_or(IssueError::CounterOverflow {
                    counter: "TMA shadow L2 requests",
                })?;
            let lane = if observation.is_load {
                &mut stats.load
            } else {
                &mut stats.store
            };
            lane[0] = lane[0].checked_add(1).ok_or(IssueError::CounterOverflow {
                counter: "TMA shadow operation-class requests",
            })?;
            if outcome.counted_hit {
                stats.hits = stats
                    .hits
                    .checked_add(1)
                    .ok_or(IssueError::CounterOverflow {
                        counter: "TMA shadow L2 hits",
                    })?;
                lane[1] = lane[1].checked_add(1).ok_or(IssueError::CounterOverflow {
                    counter: "TMA shadow operation-class hits",
                })?;
            } else {
                stats.misses = stats
                    .misses
                    .checked_add(1)
                    .ok_or(IssueError::CounterOverflow {
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
    #[error("core-only diagnostic requires complete unsampled trace coverage")]
    FastFrontierRequiresFullCoverage,
    #[error(
        "core-only diagnostic requires a single launch with no concurrent stream context; launches={launches}"
    )]
    FastFrontierLaunchScopeUnknown { launches: usize },
    #[error(
        "core-only diagnostic requires active and predicate-mask presence on every dynamic record; active_missing={active_missing}, predicate_missing={predicate_missing}"
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
    #[error("core-only diagnostic analysis failed: {detail}")]
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
    /// Distinct 32-byte constant-bank sectors referenced by executed static
    /// SASS instructions. This is an observation-only non-TEX demand term;
    /// it never changes instruction timing or cache state.
    compulsory_constant_read_sectors: u64,
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

    pub const fn compulsory_constant_read_sectors(self) -> u64 {
        self.compulsory_constant_read_sectors
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
            compulsory_constant_read_sectors: self
                .compulsory_constant_read_sectors
                .checked_add(other.compulsory_constant_read_sectors)?,
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
            let observes_register = mask != 0
                && projection
                    .is_some_and(|p| p.destinations().iter().any(|d| !d.register().is_constant()));
            let result_ready = crate::issue::data_ready_time(
                inst.kind,
                &[],
                now,
                params,
                None,
                None,
                observes_register,
            )?;
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
                    for register in projection
                        .destinations()
                        .iter()
                        .flat_map(|d| d.register().components())
                    {
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

