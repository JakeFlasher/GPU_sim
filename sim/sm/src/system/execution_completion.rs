//! Cohesive child mechanism extracted from the parent module.

use super::*;

/// One address-space origin's per-[`L2Source`] observer lanes.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct L2OriginSourceStats {
    pub sectors: [u64; 4],
    pub hits: [u64; 4],
}

/// Gate the opt-in write-admission/completion arm on a source-qualified
/// workload shape. A candidate may be present in the replay configuration for
/// every launch, but it is timing-active only when the static stream contains
/// global STORE/RED work and either a device-scope fence/ERRBAR/CCTL marker or
/// the candidate's explicit terminal-EXIT drain arm. The latter is a bounded
/// no-fence path: it reuses the typed acknowledgement stream rather than
/// claiming a generic store queue or a physical CTA-retirement mechanism.
pub(crate) fn qualifies_store_fence_feedback(
    schedule: &FunctionSchedule,
    candidate: MemoryAdmissionCandidate,
) -> bool {
    let has_global_write = schedule.by_pc.values().any(|slot| {
        matches!(
            (candidate.operation, slot.mem_role()),
            (MemoryServiceCandidateOperation::Store, MemRole::Store)
                | (
                    MemoryServiceCandidateOperation::Reduction,
                    MemRole::Reduction
                )
        )
    });
    let has_device_fence = schedule.by_pc.values().any(|slot| {
        slot.visibility_fence_semantics()
            .is_some_and(|(scope, _)| scope == sim_mem::VisibilityScope::Device)
    });
    let has_errbar_or_cctl = schedule.by_pc.values().any(|slot| {
        let opcode = slot.opcode_text().split_ascii_whitespace().next();
        matches!(opcode, Some("ERRBAR") | Some("CCTL.IVALL"))
    });
    has_global_write
        && (has_device_fence
            || has_errbar_or_cctl
            || ((candidate.exit_waits_for_service_completion
                || candidate.exit_waits_for_admission_drain)
                && !has_device_fence
                && !has_errbar_or_cctl))
}

#[derive(Debug, thiserror::Error)]
pub enum SystemError {
    #[cfg(feature = "reference")]
    #[error("reference trace: {0}")]
    ReferenceTrace(#[from] ReferenceTraceError),
    #[cfg(feature = "reference")]
    #[error("typed power state: {0}")]
    Power(#[from] PowerStateError),
    #[error(transparent)]
    Counters(#[from] SystemCounterError),
    #[error(transparent)]
    Issue(#[from] IssueError),
    #[error(transparent)]
    Engine(#[from] EngineError),
    #[error(transparent)]
    L2PendingReadCapability(#[from] sim_arch::CapabilityError),
    #[error(transparent)]
    L2Snapshot(#[from] sim_mem::L2SnapshotError),
    #[error(transparent)]
    L2(#[from] sim_mem::L2AccessError),
    #[error(transparent)]
    Writeback(#[from] sim_mem::WritebackError),
    #[error(transparent)]
    Coherence(#[from] sim_mem::CoherenceError),
    #[error(transparent)]
    L1Visibility(#[from] sim_mem::L1VisibilityError),
    #[error("logical L1 payload forwarding failed: {0}")]
    PayloadForwarding(String),
    #[error("L1 visibility protocol requires an opt-in coherence configuration")]
    L1VisibilityProtocolRequiresCoherence,
    #[error("resident L1 dirty state requires the carded writeback queue")]
    L1DirtyWritebackRequiresQueue,
    #[error("resident L1 dirty state requires the typed L1 visibility protocol")]
    L1DirtyWritebackRequiresVisibilityProtocol,
    #[error(
        "resident L1 writeback is incompatible with timing candidate/feedback lanes until hidden L2 write admissions are modeled"
    )]
    L1DirtyWritebackCandidateConflict,
    #[error(
        "resident L1/L2 dirty masks overlap on line {line_id}: L1={l1_mask:#06b}, L2={l2_mask:#06b}; payload merge is required"
    )]
    L1L2DirtyOverlap {
        line_id: i64,
        l1_mask: u8,
        l2_mask: u8,
    },
    #[error("L1 payload admission event targets unknown traced SM {sm_id}")]
    L1PayloadAdmissionSmUnknown { sm_id: i32 },
    #[error("L1 payload admission event at t={time} carries no payload records")]
    L1PayloadAdmissionEmpty { time: u64 },
    #[error("L1 payload admission event with dynamic identity has no retained store site")]
    L1PayloadAdmissionWriterMissing,
    #[error(
        "event queue drained with sm {sm_id} unfinished ({outstanding} response(s) recorded \
         outstanding) — request/response wiring invariant broken"
    )]
    Deadlock { sm_id: i32, outstanding: usize },
    #[error(
        "event-loop livelock at t={time}: {idle_wakes} consecutive SM wakes issued nothing, \
         emitted nothing and left the clock frozen — horizon/wake wiring invariant broken"
    )]
    Livelock { time: u64, idle_wakes: u64 },
    #[error("system architecture mismatch for {component}: expected {expected}, got {got}")]
    ArchitectureMismatch {
        component: &'static str,
        expected: ArchId,
        got: ArchId,
    },

    #[error(
        "latency ladder inverted: the DRAM arm ({dram}) prices below the L2-hit arm ({l2_hit}) \
         — the run-ahead causality contract takes the L2-hit arm as the response lower bound"
    )]
    LatencyLadderInverted { l2_hit: u64, dram: u64 },
    #[error(
        "DRAM bandwidth config invalid (bytes_per_sec={bytes_per_sec}, sm_clock_hz={sm_clock_hz}) \
         — zero rates must fail closed, never be clamped into an absurd or inert server"
    )]
    DramBandwidthInvalid {
        bytes_per_sec: u64,
        sm_clock_hz: u64,
    },
    #[error("DRAM fixed-point {operation} overflows u128")]
    DramFixedPointOverflow { operation: &'static str },
    #[error(
        "DRAM completion {completion_fp}/{bytes_per_sec} exceeds the u64 cycle domain after ceiling division"
    )]
    DramCycleOverflow {
        completion_fp: u128,
        bytes_per_sec: u128,
    },
    #[error("SM actor count {count} exceeds the u32 target-id domain")]
    ActorCountOutOfRange { count: usize },
    #[error("event target {target} is outside the {actor_count}-actor system")]
    EventTargetOutOfRange { target: u32, actor_count: usize },
    #[error(
        "{component} due-time mismatch for SM index {sm}: expected event t={expected}, queued head was {got:?}"
    )]
    DueTimeMismatch {
        component: &'static str,
        sm: usize,
        expected: u64,
        got: Option<u64>,
    },
    #[error(
        "{component} response identity mismatch for SM index {sm}: expected {expected:?}, queued head was {got:?}"
    )]
    ResponseIdentityMismatch {
        component: &'static str,
        sm: usize,
        expected: Option<(u64, u64, u64)>,
        got: Option<(u64, u64, u64)>,
    },
    #[error(
        "L2 request timestamp {issue_time} does not match its event timestamp {event_time} for SM index {sm}"
    )]
    RequestTimeMismatch {
        sm: usize,
        issue_time: u64,
        event_time: u64,
    },
    #[error("L2 request for sector {sector_id} has invalid sector mask {sector_mask:#06b}")]
    SectorMaskInvalid { sector_id: i64, sector_mask: u8 },
    #[error(
        "L2 request for {origin:?}-origin anchor sector {sector_id} has incoherent Local mask {local_sector_mask:#06b} within request mask {sector_mask:#06b}"
    )]
    OriginMaskInvalid {
        sector_id: i64,
        origin: MemoryOrigin,
        sector_mask: u8,
        local_sector_mask: u8,
    },
    #[error(
        "L2 request for sector {sector_id} has incoherent source/operation/reply contract: source={request_source:?}, operation={operation:?}, reply={reply:?}"
    )]
    MemoryRequestContractInvalid {
        sector_id: i64,
        request_source: L2Source,
        operation: MemoryOperation,
        reply: ReplyTo,
    },
    #[error(
        "memory-admission observation on SM {sm_id} interleaved group warp={warp} instruction={instruction} before the prior group completed"
    )]
    AdmissionObservationInterleaved {
        sm_id: i32,
        warp: u32,
        instruction: u32,
    },
    #[error(
        "memory-admission observation on SM {sm_id} expected request index {expected}, got {got} for warp={warp} instruction={instruction}"
    )]
    AdmissionObservationSequence {
        sm_id: i32,
        warp: u32,
        instruction: u32,
        expected: u16,
        got: u16,
    },
    #[error(
        "memory-admission observation on SM {sm_id} changed group identity or operation at warp={warp} instruction={instruction}"
    )]
    AdmissionObservationIdentity {
        sm_id: i32,
        warp: u32,
        instruction: u32,
    },
    #[error("memory-admission L2 observation on SM {sm_id} has {hits} hits for {sectors} sectors")]
    AdmissionObservationHits { sm_id: i32, sectors: u64, hits: u64 },
    #[error("event queue drained with an incomplete memory-admission group on SM {sm_id}")]
    AdmissionObservationIncomplete { sm_id: i32 },
    #[error(
        "event queue drained with {pending} modeled write acknowledgement(s) pending on SM {sm_id}"
    )]
    MemoryWriteAckPending { sm_id: i32, pending: usize },
    #[error("event queue drained with modeled write-ack due times still registered")]
    MemoryWriteAckDueNotDrained,
    #[error("event queue drained with {entries} pending response due-time entries")]
    ResponseDueNotDrained { entries: usize },
    #[error("event queue drained with {entries} pending SM wake due-time entries")]
    WakeDueNotDrained { entries: usize },
    #[error("event queue drained with {entries} pending L1 payload-admission due-time entries")]
    PayloadAdmissionDueNotDrained { entries: usize },
    #[error("event queue drained with {entries} pending response-bound entries")]
    PendingBoundsNotDrained { entries: usize },
    #[error("event queue drained with sm {sm_id} still owning {outstanding} response(s)")]
    ResponseNotDrained { sm_id: i32, outstanding: usize },
    #[error(
        "L2 fill schedule mismatch for sector {sector_id}: token={has_token}, ready={has_ready}"
    )]
    FillScheduleMismatch {
        sector_id: i64,
        has_token: bool,
        has_ready: bool,
    },
    #[error("pending response bound {bound} for SM index {sm} is absent")]
    PendingBoundMissing { sm: usize, bound: u64 },
    #[error("response ready time {ready} precedes lower bound {bound} for SM index {sm}")]
    ResponseBeforeBound { sm: usize, ready: u64, bound: u64 },
    #[error(
        "response ready time {response_ready} precedes L2 fill ready time {fill_ready} for sector {sector_id}"
    )]
    ResponseBeforeFill {
        sector_id: i64,
        response_ready: u64,
        fill_ready: u64,
    },
    #[error("pending L2 sector {key:?} has no registered lower-level ready time")]
    PendingL2ReadyMissing { key: L2PendingSectorKey },
    #[error("pending L2 sector {key:?} has no ready-time entry to remove")]
    PendingL2ReadyRemovalMissing { key: L2PendingSectorKey },
    #[error(
        "pending L2 sector {key:?} already has lower-level ready time {ready} during registration"
    )]
    PendingL2ReadyAlreadyRegistered { key: L2PendingSectorKey, ready: u64 },
    #[error(
        "pending L2 sector {key:?} completed at t={got}, but its registered ready time is t={expected}"
    )]
    PendingL2ReadyTimeMismatch {
        key: L2PendingSectorKey,
        expected: u64,
        got: u64,
    },
    #[error("event queue drained with {entries} pending L2 sector ready-time entries")]
    PendingL2ReadyNotDrained { entries: usize },
    #[error(
        "event queue drained with {sectors} dirty writeback sectors and {entries} pending completions"
    )]
    WritebackNotDrained { sectors: u32, entries: usize },
    #[error(
        "event queue drained with {sectors} coherence service sectors and {entries} pending completions"
    )]
    CoherenceNotDrained { sectors: u32, entries: usize },
    #[error("L1 visibility completion at {time} conflicts with an existing SM wake for actor {sm}")]
    L1VisibilityWakeConflict { sm: usize, time: u64 },
    #[error("{counter} counter overflows u64")]
    CounterOverflow { counter: &'static str },
}

#[cfg(feature = "reference")]
#[derive(Debug, thiserror::Error)]
pub enum ReferenceTraceError {
    #[error("I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("JSON encoding failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("record limit {limit} reached")]
    RecordLimit { limit: u64 },
    #[error("SIM_SM_REFERENCE_MAX_RECORDS must be an integer greater than 1, found {value:?}")]
    InvalidLimit { value: String },
    #[error("typed L2 projection failed: {0}")]
    L2Projection(#[from] sim_mem::L2ProjectionError),
    #[error("typed DRAM projection failed: {0}")]
    Dram(#[from] DramStateError),
    #[error("typed power projection failed: {0}")]
    Power(#[from] PowerStateError),
    #[error("typed queue projection failed: {0}")]
    Queue(#[from] QueueStateError),
    #[error("typed boundary structure validation failed: {0}")]
    Boundary(#[from] BoundaryStateError),
    #[error("reference macro witness is invalid: {0}")]
    Witness(String),
}

#[cfg(feature = "reference")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct ReferenceMacroWitness {
    pub(crate) schema: String,
    pub(crate) target: u32,
    pub(crate) start_sequence: u64,
    pub(crate) spec: TypedScalarMacroSpec,
    pub(crate) detailed_internal_effects: Vec<TypedEventTransition>,
    pub(crate) detailed_labels: Vec<String>,
    pub(crate) detailed_exit: TypedEventBoundaryRecord,
}

#[cfg(feature = "reference")]
pub(crate) const DEFAULT_REFERENCE_TRACE_MAX_RECORDS: u64 = 8_192;

#[cfg(feature = "reference")]
pub(crate) fn typed_scalar_macro_spec_from_env() -> Option<TypedScalarMacroSpec> {
    let instruction_count = std::env::var("SIM_SM_REFERENCE_TYPED_SCALAR_COUNT")
        .ok()?
        .parse::<u64>()
        .ok()?;
    let latency_cycles = std::env::var("SIM_SM_REFERENCE_TYPED_SCALAR_LATENCY")
        .ok()?
        .parse::<u64>()
        .ok()?;
    let initiation_interval_cycles = std::env::var("SIM_SM_REFERENCE_TYPED_SCALAR_INTERVAL")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(1);
    let label = std::env::var("SIM_SM_REFERENCE_TYPED_SCALAR_LABEL")
        .unwrap_or_else(|_| "sm_wake".to_owned());
    Some(TypedScalarMacroSpec {
        instruction_count,
        latency_cycles,
        initiation_interval_cycles,
        label,
        issue_times: None,
        terminal_front_end_exit: None,
    })
}

/// Lossless replacements relative to the preceding event record. Equal-length
/// arrays keep their positions; a shape change replaces the entire container.
/// Every state is still projected after every event, independently of which
/// component the event is expected to change.
#[cfg(feature = "reference")]
#[derive(Clone, serde::Serialize)]
#[serde(untagged)]
pub(crate) enum ReferencePathSegment {
    Field(String),
    Index(usize),
}

#[cfg(feature = "reference")]
#[derive(serde::Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub(crate) enum ReferenceOperation<'a> {
    Replace {
        path: Vec<ReferencePathSegment>,
        value: &'a serde_json::Value,
    },
    Splice {
        path: Vec<ReferencePathSegment>,
        start: usize,
        delete_count: usize,
        values: Vec<&'a serde_json::Value>,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct PendingL2ReadyEntry {
    pub(crate) ready: u64,
    #[cfg(feature = "reference")]
    pub(crate) producer: Option<L2RequestKey>,
}

pub(crate) struct L2ResponseCommit {
    pub(crate) ready: u64,
    pub(crate) origin_ix: usize,
    pub(crate) next_delayed: u64,
    pub(crate) next_origin_delayed: u64,
    pub(crate) pending_bounds_removal: PendingRemoval,
}

#[derive(Debug, Clone)]
pub(crate) struct PendingAdmissionInstruction {
    pub(crate) witness: AdmissionRequestWitness,
    pub(crate) operation: MemoryOperation,
    pub(crate) issue_time: u64,
    pub(crate) next_request_index: u16,
    pub(crate) hit_sectors: u64,
    pub(crate) miss_sectors: u64,
    pub(crate) regions: Vec<i64>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CompletedAdmissionInstruction {
    pub(crate) witness: AdmissionRequestWitness,
    pub(crate) operation: MemoryOperation,
    pub(crate) issue_time: u64,
    pub(crate) hit_sectors: u64,
    pub(crate) miss_sectors: u64,
    pub(crate) l2_requests: u64,
    pub(crate) unique_regions: u64,
}

pub(crate) struct L2SystemCommit {
    pub(crate) source_ix: usize,
    pub(crate) source_sectors: u64,
    pub(crate) source_hits: u64,
    pub(crate) origin_source_sectors: [u64; 2],
    pub(crate) origin_source_hits: [u64; 2],
    pub(crate) read_sectors: u64,
    pub(crate) atomic_rmw_read_sectors: u64,
    pub(crate) reduction_rmw_read_sectors: u64,
    pub(crate) origin_read_sectors: [u64; 2],
    pub(crate) topology_stats: DramStats,
    pub(crate) dram_free_fp: Option<u128>,
    pub(crate) read_service: Option<ServicePlan>,
    pub(crate) writeback: Option<ServicePlan>,
    pub(crate) coherence: Option<CoherencePlan>,
    pub(crate) fill_ready: Option<u64>,
    /// Shared-L2 allocation generation associated with a returning response.
    /// The L1 fill generation is a separate identity domain and must not be
    /// used for this join.
    pub(crate) response_l2_allocation_generation: Option<u64>,
    pub(crate) atomic_completion_ready: Option<u64>,
    pub(crate) response: Option<L2ResponseCommit>,
    pub(crate) observed_sectors: u64,
    pub(crate) observed_hits: u64,
    /// Observer-only read/atomic contribution. Store/reduction packetization
    /// is reconstructed from the issuing instruction topology after the event
    /// stream drains, so it never changes modeled L2 state or response time.
    pub(crate) lookup_observer_requests: u64,
    pub(crate) lookup_observer_hits: u64,
    pub(crate) read_lookup_requests: u64,
    pub(crate) read_lookup_hits: u64,
    pub(crate) read_pending_merge_hits: u64,
    pub(crate) store_lookup_requests: u64,
    pub(crate) store_lookup_hits: u64,
    pub(crate) store_packet_lookup_hits: u64,
    pub(crate) store_packet_resident_lookup_hits: u64,
    pub(crate) atomic_lookup_requests: u64,
    pub(crate) atomic_lookup_hits: u64,
    pub(crate) reduction_lookup_requests: u64,
    pub(crate) reduction_lookup_hits: u64,
    /// Source-qualified sectored fill-ownership observer. A pending merge is
    /// a lookup miss that reuses an older allocation's ready frontier; a new
    /// fill owns a new lower-level fetch capability. These values never alter
    /// service timing or cache state.
    pub(crate) pending_merge_sectors: u64,
    pub(crate) new_fill_sectors: u64,
    pub(crate) pending_pressure_samples: u64,
    pub(crate) pending_pressure_sum: u64,
    pub(crate) l2_frontier_samples: u64,
    pub(crate) l2_issue_to_fill_cycles_sum: u64,
    pub(crate) l2_issue_to_fill_cycles_max: u64,
    pub(crate) l2_issue_to_response_cycles_sum: u64,
    pub(crate) l2_issue_to_response_cycles_max: u64,
    pub(crate) l2_service_delay_cycles_sum: u64,
    pub(crate) l2_service_delay_cycles_max: u64,
    pub(crate) client_scope_ix: usize,
    pub(crate) client_scope_requests: u64,
    pub(crate) client_scope_resident_hits: u64,
    pub(crate) client_scope_pending_merges: u64,
    pub(crate) client_scope_fill_misses: u64,
    pub(crate) ltc128b_shadow_promoted_sectors: u64,
}

/// Run every SM's blocks to completion over one shared L2. `l2` is
/// caller-constructed (and caller-drained afterwards) so the geometry
/// resolution stays at the composition root; it must be FRESH per kernel
/// (cold cache, NCU `--cache-control all` parity — same contract as the
/// per-SM L1s). The caller's model is unchanged on every typed error; a
/// successful run commits the fully drained staged L2 as its final operation.
/// `dram` selects the read-bandwidth model ([`DramBw`]):
/// `Unmodeled` is the latency-only ablation / equivalence-differential
/// mode; `Carded` rates are validated non-zero here (typed
/// [`SystemError::DramBandwidthInvalid`], never a clamp).

#[allow(clippy::too_many_arguments)]

/// Drain resident dirty sectors from the per-SM L1s through the explicit
/// writeback queue. The cache plans are committed only after every lower-level
/// service has been admitted and completed, so a capacity or stale-plan error
/// leaves the L1 state dirty rather than silently losing data.
pub(crate) fn flush_l1_dirty_residency(
    cores: &mut [SmCore<'_>],
    l2: &L2Model,
    writeback_queue: Option<&mut WritebackQueue>,
    terminal_event_time: &mut Option<u64>,
    dram_stats: &mut DramStats,
) -> Result<(), SystemError> {
    let mut plans = Vec::new();
    for (ix, core) in cores.iter().enumerate() {
        *dram_stats = dram_stats.with_terminal_l1_dirty_residency(core.l1_dirty_residency())?;
        let plan = core.preview_l1_dirty_flush()?;
        let payloads = if core.payload_observation_enabled() {
            // A store-bound value event enables a stricter terminal contract:
            // every dirty sector must still carry a complete payload before
            // the mask-only timing queue is allowed to clear it. The queue's
            // byte server remains a timing model, not a silicon DRAM receipt.
            core.preview_l1_dirty_payloads()?
        } else {
            Vec::new()
        };
        if !plan.lines().is_empty() {
            plans.push((ix, plan, payloads));
        }
    }
    let l2_dirty = l2.preview_dirty_flush();
    for (_, plan, _) in &plans {
        for l1_line in plan.lines() {
            if let Some(l2_line) = l2_dirty
                .lines()
                .iter()
                .find(|candidate| candidate.line_id == l1_line.line_id)
                && l1_line.dirty_mask & l2_line.dirty_mask != 0
            {
                return Err(SystemError::L1L2DirtyOverlap {
                    line_id: l1_line.line_id,
                    l1_mask: l1_line.dirty_mask,
                    l2_mask: l2_line.dirty_mask,
                });
            }
        }
    }
    if plans.is_empty() {
        return Ok(());
    }
    let Some(writeback_queue) = writeback_queue else {
        return Err(SystemError::L1DirtyWritebackRequiresQueue);
    };
    let capacity = usize::try_from(writeback_queue.capacity_sectors()).map_err(|_| {
        SystemError::CounterOverflow {
            counter: "terminal L1 writeback capacity",
        }
    })?;
    let mut latest_done = (*terminal_event_time).unwrap_or(0);
    let mut batch_issue = latest_done;
    let mut forwarded_payloads = Vec::new();
    for (ix, plan, payloads) in plans {
        let mut offset = 0usize;
        while offset < plan.lines().len() {
            let mut end = offset;
            let mut sectors = 0usize;
            while end < plan.lines().len() {
                let line_sectors = usize::try_from(plan.lines()[end].dirty_mask.count_ones())
                    .map_err(|_| SystemError::CounterOverflow {
                        counter: "terminal L1 dirty sectors",
                    })?;
                if end == offset && line_sectors > capacity {
                    return Err(SystemError::Writeback(
                        sim_mem::WritebackError::CapacityExceeded {
                            capacity: writeback_queue.capacity_sectors(),
                            requested: u32::try_from(line_sectors).unwrap_or(u32::MAX),
                        },
                    ));
                }
                if sectors
                    .checked_add(line_sectors)
                    .is_none_or(|total| total > capacity)
                {
                    break;
                }
                sectors += line_sectors;
                end += 1;
            }
            let requests = plan.lines()[offset..end]
                .iter()
                .map(|line| ServiceRequest::writeback(line.line_id, line.dirty_mask, batch_issue))
                .collect::<Vec<_>>();
            let service_plans = writeback_queue.preview_services(&requests)?;
            writeback_queue.commit_services(&service_plans)?;
            for service in &service_plans {
                let completion =
                    writeback_queue.complete(service.token(), service.service_done_cycle())?;
                latest_done = latest_done.max(completion.service_done_cycle);
                let sectors = u64::from(completion.dirty_mask.count_ones());
                dram_stats.writeback_completed_lines = dram_stats
                    .writeback_completed_lines
                    .checked_add(1)
                    .ok_or(SystemError::CounterOverflow {
                        counter: "completed L1 writeback lines",
                    })?;
                dram_stats.writeback_completed_sectors = dram_stats
                    .writeback_completed_sectors
                    .checked_add(sectors)
                    .ok_or(SystemError::CounterOverflow {
                        counter: "completed L1 writeback sectors",
                    })?;
                let bytes = sectors
                    .checked_mul(32)
                    .ok_or(SystemError::CounterOverflow {
                        counter: "completed L1 writeback bytes",
                    })?;
                dram_stats.writeback_completed_bytes = dram_stats
                    .writeback_completed_bytes
                    .checked_add(bytes)
                    .ok_or(SystemError::CounterOverflow {
                        counter: "completed L1 writeback bytes",
                    })?;
                dram_stats.l1_writeback_completed_lines = dram_stats
                    .l1_writeback_completed_lines
                    .checked_add(1)
                    .ok_or(SystemError::CounterOverflow {
                        counter: "completed L1 writeback lines",
                    })?;
                dram_stats.l1_writeback_completed_sectors = dram_stats
                    .l1_writeback_completed_sectors
                    .checked_add(sectors)
                    .ok_or(SystemError::CounterOverflow {
                        counter: "completed L1 writeback sectors",
                    })?;
                dram_stats.l1_writeback_completed_bytes = dram_stats
                    .l1_writeback_completed_bytes
                    .checked_add(bytes)
                    .ok_or(SystemError::CounterOverflow {
                        counter: "completed L1 writeback bytes",
                    })?;
            }
            batch_issue = latest_done;
            offset = end;
        }
        cores[ix].commit_l1_dirty_flush(plan)?;
        forwarded_payloads.extend(payloads);
    }
    for payload in forwarded_payloads {
        let mut ledger = PayloadForwardingLedgerV1::from_admitted_store(payload)
            .map_err(|error| SystemError::PayloadForwarding(error.to_string()))?;
        ledger
            .commit_l2(payload)
            .map_err(|error| SystemError::PayloadForwarding(error.to_string()))?;
        ledger
            .commit_dram(payload)
            .map_err(|error| SystemError::PayloadForwarding(error.to_string()))?;
        dram_stats.l1_payload_forwarded_sectors = dram_stats
            .l1_payload_forwarded_sectors
            .checked_add(1)
            .ok_or(SystemError::CounterOverflow {
                counter: "logical L1 payload-forwarded sectors",
            })?;
        dram_stats.l1_payload_forwarded_bytes = dram_stats
            .l1_payload_forwarded_bytes
            .checked_add(32)
            .ok_or(SystemError::CounterOverflow {
                counter: "logical L1 payload-forwarded bytes",
            })?;
        dram_stats.l1_payload_forwarding_digest_xor ^= payload.digest64();
    }
    dram_stats.writeback_completion_frontier = Some(
        dram_stats
            .writeback_completion_frontier
            .unwrap_or(0)
            .max(latest_done),
    );
    *terminal_event_time = Some((*terminal_event_time).unwrap_or(0).max(latest_done));
    Ok(())
}

pub(crate) fn replay_horizon(response: Option<u64>, pending_bounds: Option<u64>) -> ReplayHorizon {
    match (response, pending_bounds) {
        (Some(left), Some(right)) => ReplayHorizon::Bounded(left.min(right)),
        (Some(bound), None) | (None, Some(bound)) => ReplayHorizon::Bounded(bound),
        (None, None) => ReplayHorizon::Unbounded,
    }
}

pub(crate) fn batchable_replyless_write(request: MemRequest) -> bool {
    request.reply == ReplyTo::None
        && matches!(
            request.operation,
            MemoryOperation::Store | MemoryOperation::Reduction
        )
}

pub(crate) fn feedback_prices_request(
    candidate: Option<MemoryAdmissionCandidate>,
    request: MemRequest,
) -> bool {
    request.admission_witness.is_some()
        && candidate.is_some_and(|candidate| candidate.matches_operation(request.operation))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn schedule_emitted_requests(
    sm: usize,
    target: TargetId,
    queue: &mut EventQueue<Ev>,
    pending_bounds: &mut PendingBounds,
    scratch: &mut Vec<MemRequest>,
    l2_hit_latency: u64,
    batch_replyless_writes: bool,
    memory_admission_feedback: Option<MemoryAdmissionCandidate>,
) -> Result<(), SystemError> {
    let mut index = 0;
    while index < scratch.len() {
        let request = scratch[index];
        if batch_replyless_writes && batchable_replyless_write(request) {
            let mut end = index + 1;
            while end < scratch.len()
                && batchable_replyless_write(scratch[end])
                && scratch[end].issue_time == request.issue_time
            {
                end += 1;
            }
            if end - index > 1 {
                for batched in &scratch[index..end] {
                    if feedback_prices_request(memory_admission_feedback, *batched) {
                        pending_bounds.insert(sm, batched.issue_time)?;
                    }
                }
                let requests = end - index;
                let sequence_span = NonZeroU64::new(u64::try_from(requests).map_err(|_| {
                    SystemError::CounterOverflow {
                        counter: "L2 access batch sequence span",
                    }
                })?)
                .expect("a batched run contains at least two requests");
                queue.schedule_at_spanning(
                    request.issue_time,
                    target,
                    Ev::L2AccessBatch(scratch[index..end].to_vec().into_boxed_slice()),
                    sequence_span,
                )?;
                index = end;
                continue;
            }
        }
        if request.reply != ReplyTo::None {
            let bound = checked_cycle_add(
                "pending L2-hit response bound",
                request.issue_time,
                l2_hit_latency,
            )?;
            pending_bounds.insert(sm, bound)?;
        }
        // A timing-active write candidate cannot be priced until this future
        // L2 access is processed.  The memory issue pipeline may emit the
        // request several cycles ahead of the actor clock, so retain that
        // access timestamp as a self-affecting bound just like an pending_bounds
        // read response.  Otherwise an unrelated wake can let the SM run past
        // the event that releases its feedback queue or schedules its ack.
        if feedback_prices_request(memory_admission_feedback, request) {
            pending_bounds.insert(sm, request.issue_time)?;
        }
        queue.schedule_at(request.issue_time, target, Ev::L2Access(request))?;
        index += 1;
    }
    scratch.clear();
    Ok(())
}

#[cfg(feature = "reference")]
pub(crate) fn dispatch_reference_macro_event(
    core: &mut SmCore<'_>,
    sm_ix: usize,
    actor_count: usize,
    result: &NativeBoundaryExecutionResult,
    queue: &mut EventQueue<Ev>,
    wake_due: &mut DueTimes,
    power_state: Option<&mut PowerBoundaryState>,
    mut trace: Option<&mut ReferenceTrace>,
) -> Result<bool, SystemError> {
    let contract = match result {
        NativeBoundaryExecutionResult::TypedMacro { contract } => contract,
        NativeBoundaryExecutionResult::ExactReplay { .. } => return Ok(false),
    };
    if contract.validate().is_err() {
        if let Some(trace) = trace.as_deref_mut() {
            trace.note_native_fallback();
        }
        return Ok(false);
    }
    // The scalar actor is allowed to consume only a boundary whose logical
    // effect reservations are immediately after the event that was popped.
    // Otherwise a witness could make a Preserve decision while the real event
    // queue already contains work that would receive an ambiguous sequence.
    let effect_count = u64::try_from(contract.internal_effects.len()).map_err(|_| {
        IssueError::ReferenceMacroUnsupported {
            reason: "typed_macro_effect_count_out_of_range",
        }
    })?;
    let Some(effect_span) = NonZeroU64::new(effect_count) else {
        if let Some(trace) = trace.as_deref_mut() {
            trace.note_native_fallback();
        }
        return Ok(false);
    };
    let expected_effect_start = contract.entry.event.sequence.checked_add(1).ok_or(
        IssueError::ReferenceMacroUnsupported {
            reason: "typed_macro_sequence_overflow",
        },
    )?;
    let expected_exit_sequence = expected_effect_start.checked_add(effect_count).ok_or(
        IssueError::ReferenceMacroUnsupported {
            reason: "typed_macro_sequence_overflow",
        },
    )?;
    if contract.entry.event.payload != TypedEventPayload::SmWake
        || !contract.entry.queue.pending.is_empty()
        || contract.exit.event.sequence != expected_exit_sequence
        || queue.next_sequence() != Some(expected_effect_start)
        || wake_due.next(sm_ix).is_some()
    {
        if let Some(trace) = trace.as_deref_mut() {
            trace.note_native_fallback();
        }
        return Ok(false);
    }
    // Validate and install the actor transition before consuming queue
    // sequence space. The actor performs all descriptor/continuation checks
    // before mutation; an unsupported descriptor therefore falls back without
    // changing either the core or the event queue.
    let native_execution = core.execute_reference_boundary_result(result)?;
    let Some(execution) = native_execution else {
        if let Some(trace) = trace.as_deref_mut() {
            trace.note_native_fallback();
        }
        return Ok(false);
    };
    // Reserve the declared internal effects before the continuation is
    // inserted.  The continuation therefore receives the contract's
    // region-end sequence instead of silently reusing the next free sequence
    // as if the collapsed effects had never existed.
    let reserved_start = queue.reserve_sequence_span(effect_span)?;
    debug_assert_eq!(reserved_start, expected_effect_start);
    if let Some(trace) = trace.as_deref_mut() {
        trace.note_native_dispatch();
    }
    let target = u32::try_from(sm_ix)
        .map_err(|_| SystemError::ActorCountOutOfRange { count: actor_count })?;
    let continuation_seq = queue.schedule_at(execution.end_cycle, TargetId(target), Ev::SmWake)?;
    if continuation_seq != expected_exit_sequence {
        return Err(SystemError::Issue(IssueError::ReferenceMacroUnsupported {
            reason: "typed_macro_continuation_sequence_mismatch",
        }));
    }
    if let Some(power_state) = power_state {
        power_state
            .record_scalar_macro_completion(contract.entry.event.time, execution.end_cycle)?;
    }
    if let Some(trace) = trace.as_deref_mut() {
        trace.expect_region_end(contract.exit.clone());
    }
    wake_due.push(sm_ix, execution.end_cycle)?;
    Ok(true)
}

/// Resume one SM actor: advance to its earliest self-affecting external
/// event — the earlier of its next scheduled response and the lower
/// bound of any request still awaiting L2 pricing — schedule its
/// emitted requests, and re-arm its wake timer if it has one. Returns
/// whether the actor made progress (issued or emitted anything) — the
/// livelock guard's signal.
#[allow(clippy::too_many_arguments)] // driver-internal plumbing, one call shape
pub(crate) fn run_sm(
    ix: usize,
    cores: &mut [SmCore<'_>],
    queue: &mut EventQueue<Ev>,
    response_due: &mut DueTimes,
    write_ack_due: &mut DueTimes,
    wake_due: &mut DueTimes,
    pending_bounds: &mut PendingBounds,
    payload_due: &DueTimes,
    batch_replyless_writes: bool,
    scratch: &mut Vec<MemRequest>,
    l1_visibility_observations: &mut Vec<sim_mem::L1AccessObservation>,
    l1_fence_observations: &mut Vec<sim_mem::L1FenceObservation>,
    l1_fill_visibility_observations: &mut Vec<sim_mem::L1FillVisibleObservation>,
    l1_visibility_queue: Option<&mut L1VisibilityQueue>,
    _sm_ids: &[i32],
    l2_hit_latency: u64,
    memory_admission_feedback: Option<MemoryAdmissionCandidate>,
) -> Result<bool, SystemError> {
    scratch.clear();
    let write_or_pending_bounds = [
        write_ack_due.next(ix),
        pending_bounds.min(ix),
        payload_due.next(ix),
    ]
    .into_iter()
    .flatten()
    .min();
    let initial_visibility_events = cores[ix].take_l1_visibility_events();
    schedule_l1_visibility_events(
        ix,
        initial_visibility_events,
        queue,
        l1_visibility_observations,
        l1_fence_observations,
        l1_fill_visibility_observations,
        cores.len(),
    )?;
    let actor_horizon = replay_horizon(response_due.next(ix), write_or_pending_bounds);
    // The opt-in visibility protocol cannot safely run an actor past an
    // already queued cross-SM invalidation/fill event: its trace-local cache
    // state was generated before that event and cannot be retracted.  Keep
    // the legacy run-ahead path unchanged; strict visibility pays a bounded
    // next-event cap. `SmCore::advance` also yields immediately after the
    // issuing actor produces a same-cycle visibility observation, so those
    // callbacks are no longer hidden behind a later local issue batch.
    let horizon = if l1_visibility_queue.is_some() {
        queue
            .peek_time()
            .filter(|next| *next > queue.now())
            .map_or(actor_horizon, |next| match actor_horizon {
                ReplayHorizon::Bounded(bound) => ReplayHorizon::Bounded(bound.min(next)),
                ReplayHorizon::Unbounded => ReplayHorizon::Bounded(next),
            })
    } else {
        actor_horizon
    };
    let issued_before = cores[ix].instructions_issued();
    let adv = cores[ix].advance(horizon, scratch)?;
    let events = cores[ix].take_l1_visibility_events();
    schedule_l1_visibility_events(
        ix,
        events,
        queue,
        l1_visibility_observations,
        l1_fence_observations,
        l1_fill_visibility_observations,
        cores.len(),
    )?;
    let progress = cores[ix].instructions_issued() > issued_before || !scratch.is_empty();
    let target =
        u32::try_from(ix).map_err(|_| SystemError::ActorCountOutOfRange { count: cores.len() })?;
    schedule_emitted_requests(
        ix,
        TargetId(target),
        queue,
        pending_bounds,
        scratch,
        l2_hit_latency,
        batch_replyless_writes,
        memory_admission_feedback,
    )?;
    if let Advance::Idle {
        wake_at: Some(wake),
    } = adv
    {
        // A cross-SM event may advance the shared queue clock while this
        // actor still has a locally known wake timestamp. Never enqueue a
        // stale wake into the global queue; re-arm at the current event
        // boundary and let `SmCore::advance` catch the actor up on the next
        // pass. This preserves event causality without fabricating service
        // completion time.
        let wake = wake.max(queue.now());
        let covered = wake_due.next(ix).is_some_and(|t| t <= wake);
        if !covered {
            queue.schedule_at(wake, TargetId(target), Ev::SmWake)?;
            wake_due.push(ix, wake)?;
        }
    }
    Ok(progress)
}

pub(crate) fn schedule_l1_visibility_events(
    ix: usize,
    events: Vec<sim_mem::L1VisibilityEvent>,
    queue: &mut EventQueue<Ev>,
    l1_visibility_observations: &mut Vec<sim_mem::L1AccessObservation>,
    l1_fence_observations: &mut Vec<sim_mem::L1FenceObservation>,
    l1_fill_visibility_observations: &mut Vec<sim_mem::L1FillVisibleObservation>,
    actor_count: usize,
) -> Result<(), SystemError> {
    let target = TargetId(
        u32::try_from(ix).map_err(|_| SystemError::ActorCountOutOfRange { count: actor_count })?,
    );
    for event in events {
        match event {
            sim_mem::L1VisibilityEvent::Access(observation) => {
                l1_visibility_observations.push(observation);
                queue.schedule_at(
                    observation.issue_cycle,
                    target,
                    Ev::L1VisibilityAccess(observation),
                )?;
            }
            sim_mem::L1VisibilityEvent::Fence(observation) => {
                l1_fence_observations.push(observation);
                queue.schedule_at(
                    observation.issue_cycle,
                    target,
                    Ev::L1VisibilityFence(observation),
                )?;
            }
            sim_mem::L1VisibilityEvent::FillVisible(observation) => {
                l1_fill_visibility_observations.push(observation);
                queue.schedule_at(
                    observation.visible_cycle,
                    target,
                    Ev::L1FillVisible(observation),
                )?;
            }
            sim_mem::L1VisibilityEvent::InvalidateAll(observation) => {
                queue.schedule_at(
                    observation.issue_cycle,
                    target,
                    Ev::L1InvalidateAll(observation),
                )?;
            }
        }
    }
    Ok(())
}

pub(crate) fn schedule_l1_visibility_completion(
    ix: usize,
    plan: L1VisibilityPlan,
    queue: &mut EventQueue<Ev>,
    pending_bounds: &mut PendingBounds,
    wake_due: &DueTimes,
    sm_ids: &[i32],
    actor_count: usize,
) -> Result<(), SystemError> {
    if !plan.needs_completion() {
        return Ok(());
    }
    let mut affected = vec![ix];
    for target in plan.invalidations() {
        let target_ix = sm_ids
            .iter()
            .position(|candidate| u8::try_from(*candidate).ok() == Some(target.requester))
            .ok_or(SystemError::EventTargetOutOfRange {
                target: u32::from(target.requester),
                actor_count,
            })?;
        if !affected.contains(&target_ix) {
            affected.push(target_ix);
        }
    }
    queue.schedule_at(
        plan.service_done_cycle(),
        TargetId(
            u32::try_from(ix)
                .map_err(|_| SystemError::ActorCountOutOfRange { count: actor_count })?,
        ),
        Ev::L1VisibilityComplete { plan: plan.clone() },
    )?;
    for target_ix in affected {
        pending_bounds.insert(target_ix, plan.service_done_cycle())?;
        if wake_due
            .next(target_ix)
            .is_some_and(|time| time <= plan.service_done_cycle())
        {
            return Err(SystemError::L1VisibilityWakeConflict {
                sm: target_ix,
                time: plan.service_done_cycle(),
            });
        }
    }
    Ok(())
}
