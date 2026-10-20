//! Serialization contracts for typed observations; report-only.

use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AddressDomain {
    VirtualAbsolute,
    AllocationRelative,
    Physical,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TypedMemoryOrigin {
    Global,
    Local,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TypedL2Source {
    L1Fill,
    Bypass,
    Untagged,
    Store,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TypedResponseKind {
    None,
    L1Fill,
    L1SectorFill,
    Warp,
    AsyncCopy,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum L2RequestState {
    Outstanding,
    Completed,
    Stale,
    NonReturning,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L2CounterState {
    pub hits: u64,
    pub misses: u64,
    pub lookup_hits: u64,
    pub lookup_requests: u64,
    pub line_misses: u64,
    pub sector_fill_misses: u64,
    pub write_tag_hits: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L2WayState {
    pub line: CanonicalLineKey,
    pub valid_mask: u8,
    pub dirty_mask: u8,
    pub pending_mask: u8,
    pub recency: i64,
    pub generation: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L2SectorOwnership {
    pub sector: u8,
    pub status: L2SectorStatus,
    pub producer: Option<L2RequestKey>,
    pub consumers: Vec<L2RequestKey>,
    pub ready_at: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L2AllocationState {
    pub line: CanonicalLineKey,
    pub generation: u64,
    pub sectors: Vec<L2SectorOwnership>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum L1FillDueState {
    Timed { cycle: u64 },
    Untimed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L1LineFillState {
    pub line: i64,
    pub due: L1FillDueState,
    pub sequence: u64,
    pub generation: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L1CounterState {
    pub hits: u64,
    pub misses: u64,
    pub merges: u64,
    pub already_requested_merges: u64,
    pub unrequested_pending_merges: u64,
    pub pending_lookup_hits: u64,
    pub stall_events: u64,
    pub line_fills: u64,
    pub line_alloc_misses: u64,
    pub sector_resident_misses: u64,
    pub bypass_sectors: u64,
    pub bypass_atom_sectors: u64,
    pub untagged_sectors: u64,
    pub store_sectors: u64,
    pub store_hits: u64,
    pub store_tag_hits: u64,
    pub write_lookup_hits: u64,
    pub reduction_sectors: u64,
    pub reduction_hits: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AsyncGroupBoundaryState {
    pub sequence: u64,
    pub committed_at: u64,
    pub pending_sources: u32,
    pub source_completion_time: u64,
    pub delivery_work: u64,
    pub destination_wavefronts: u64,
    pub destination_service_queued: bool,
    pub destination_completion_time: u64,
    pub completion_time: u64,
}

/// Per-cohort response state retained for a live bypass `LDGSTS` copy.  The
/// response timestamps are optional while source sectors remain in flight;
/// once a cohort has drained, both endpoints are required to preserve the
/// response envelope used by the diagnostic projection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AsyncCopyCohortBoundaryState {
    pub pending_sources: u32,
    pub first_source_response: Option<u64>,
    pub last_source_response: Option<u64>,
    pub payload_work: u64,
    pub destination_lines: u64,
    pub bank_wavefronts: u64,
    pub bypass_wavefronts: u64,
    pub topology_penalty: u64,
}

/// Lossless identity and response-to-cohort mapping for one live bypass
/// asynchronous-copy instruction.  Group and instruction form the producer
/// identity used by the response path; sector masks preserve which cohorts a
/// delayed source response must decrement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AsyncCopyBypassObservationState {
    pub group: u64,
    pub instruction: u64,
    pub pending_sources: u32,
    pub source_completion_time: u64,
    pub delivery_work: u64,
    pub source_cohorts: u64,
    pub destination_lines: u64,
    pub max_bank_wavefronts: u64,
    pub max_bypass_wavefronts: u64,
    pub sector_off: u64,
    pub sector_membership_masks: Vec<u32>,
    pub cohorts: Vec<AsyncCopyCohortBoundaryState>,
}

pub(crate) fn validate_async_group(
    group: &AsyncGroupBoundaryState,
) -> Result<(), AsyncCopyStateError> {
    if group.source_completion_time > group.completion_time {
        return Err(AsyncCopyStateError::TimestampOrder);
    }
    if group.destination_service_queued {
        if group.pending_sources != 0 {
            return Err(AsyncCopyStateError::DestinationPendingSources);
        }
        if group.destination_wavefronts == 0
            || group.destination_completion_time == 0
            || group.destination_completion_time > group.completion_time
        {
            return Err(AsyncCopyStateError::DestinationService);
        }
    } else if group.destination_completion_time != 0 {
        return Err(AsyncCopyStateError::DestinationService);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryPipelineBoundaryState {
    pub queue_entries_with_dispatch: u32,
    pub local_initiation_interval_cycles: u64,
    pub shared_accept_interval_cycles: u64,
    pub queue_release_times: Vec<Vec<u64>>,
    pub local_completion_ready: Vec<u64>,
    pub shared_accept_ready: u64,
    pub instructions_admitted: u64,
    pub operand_pipeline_delay_cycles: u64,
    pub shared_queue_delay_cycles: u64,
    pub maximum_operand_pipeline_delay_cycles: u64,
    pub maximum_shared_queue_delay_cycles: u64,
    pub maximum_queue_occupancy: u64,
    pub terminal_shared_accept_cycle: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalarPipelineBoundaryState {
    pub latency_cycles: u64,
    pub initiation_interval_cycles: u64,
    pub in_flight_capacity: u32,
    pub release_times: Vec<Vec<u64>>,
    pub next_service: Vec<u64>,
    pub instructions_admitted: u64,
    pub service_busy_cycles: u64,
    pub queue_delay_cycles: u64,
    pub maximum_queue_occupancy: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstructionCacheEntryState {
    pub function_unique_id: i32,
    pub line: u64,
    pub ready_at: u64,
    pub last_access: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstructionStreamLineState {
    pub function_unique_id: i32,
    pub line: u64,
    pub ready_at: u64,
    pub demanded: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum FrontEndStateError {
    #[error("typed instruction front-end shape is invalid")]
    Shape,
    #[error("typed instruction front-end cache has zero ways")]
    CacheWays,
    #[error("typed instruction front-end cache has no sets")]
    CacheShape,
    #[error("typed instruction front-end cache set exceeds its way capacity")]
    CacheCapacity,
    #[error("typed instruction front-end cache has duplicate entries")]
    DuplicateCacheEntry,
    #[error(
        "typed instruction front-end function identity must be positive, found {function_unique_id}"
    )]
    FunctionIdentity { function_unique_id: i32 },
    #[error("typed instruction front-end cache entry is in set {found}, expected set {expected}")]
    CacheSetPlacement { expected: u64, found: usize },
    #[error("typed instruction front-end cache reuses a last-access sequence")]
    DuplicateCacheAccessSequence,
    #[error("typed instruction front-end cache entry has an unreserved access sequence")]
    CacheAccessSequence,
    #[error("typed instruction front-end stream exceeds its declared capacity")]
    StreamCapacity,
    #[error("typed instruction front-end stream lines are not consecutive")]
    StreamOrder,
    #[error("typed instruction front-end stream ready times are not ordered")]
    StreamReadyOrder,
    #[error("typed instruction front-end stream has a demanded non-head line")]
    DemandedStreamTail,
    #[error("typed instruction front-end next-line frontier disagrees with its stream tail")]
    StreamNextLine,
    #[error("typed instruction front-end L0 caches do not share one geometry")]
    L0GeometryMismatch,
    #[error("typed instruction front-end prefetch priority is outside the subcore domain")]
    PrefetchPriority,
    #[error("typed instruction front-end reply timer is before request timer")]
    ReplyTimerBeforeRequest,
    #[error("typed instruction front-end L1 timer and access sequence disagree")]
    L1TimerSequenceMismatch,
    #[error("typed instruction front-end L0 timer and access sequence disagree")]
    L0TimerSequenceMismatch,
    #[error("typed instruction front-end field {field} has unsupported width {value}")]
    UnsupportedWidth { field: &'static str, value: u32 },
    #[error("typed instruction front-end counter {field} expected {expected}, found {found}")]
    CounterMismatch {
        field: &'static str,
        expected: u64,
        found: u64,
    },
    #[error("typed instruction front-end counter arithmetic overflowed")]
    CounterOverflow,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum L2StateError {
    #[error("typed L2 requester uses the invalid u32::MAX SM identity sentinel")]
    InvalidSmIdentity,
    #[error("typed L2 schema {found} is unsupported; expected {expected}")]
    SchemaVersion { found: u32, expected: u32 },
    #[error("typed L2 geometry contains a zero dimension")]
    InvalidGeometry,
    #[error("typed L2 access discipline disagrees with resident allocation generations")]
    DisciplineGenerationMismatch,
    #[error("typed L2 access discipline is missing for an active complete state")]
    DisciplineMissing,
    #[error("typed L2 clock is negative")]
    NegativeClock,
    #[error("typed L2 line identity disagrees with the declared address domain")]
    AddressDomainMismatch,
    #[error("typed L2 root and geometry architecture identities disagree")]
    ArchitectureMismatch,
    #[error("typed L2 geometry uses an unknown replacement policy")]
    UnknownReplacement,
    #[error("typed L2 capacity {found} disagrees with geometry product {expected}")]
    GeometryMismatch { expected: u128, found: u64 },
    #[error("typed L2 lookup hits exceed lookup requests")]
    LookupHitsExceedRequests,
    #[error("typed L2 observer lookup hits exceed observer lookup requests")]
    ObserverLookupHitsExceedRequests,
    #[error("typed L2 miss counter {misses} disagrees with components {components}")]
    MissCompositionMismatch { misses: u64, components: u64 },
    #[error("typed L2 write-tag hits exceed hits")]
    WriteTagHitsExceedHits,
    #[error("typed L2 counter arithmetic overflowed")]
    CounterOverflow,
    #[error("typed L2 set ({slice},{set}) is outside declared geometry")]
    SetOutOfRange { slice: u32, set: u64 },
    #[error("typed L2 set {set} contains too many ways")]
    SetOverfull { set: u64 },
    #[error("typed L2 sets are not in canonical order")]
    UnsortedSets,
    #[error("typed L2 contains duplicate set ({slice},{set})")]
    DuplicateSet { slice: u32, set: u64 },
    #[error("typed L2 set {set} contains duplicate way lines")]
    DuplicateWayLine { set: u64 },
    #[error("typed L2 way {line:?} has neither valid nor pending sectors")]
    EmptyWay { line: CanonicalLineKey },
    #[error("typed L2 {replacement} way {line:?} has invalid recency {recency} at clock {clock}")]
    InvalidRecency {
        line: CanonicalLineKey,
        recency: i64,
        clock: i64,
        replacement: &'static str,
    },
    #[error("typed L2 set ({slice},{set}) contains duplicate recency {recency}")]
    DuplicateRecency { slice: u32, set: u64, recency: i64 },
    #[error("typed L2 contains duplicate resident line/generation identities")]
    DuplicateWayIdentity,
    #[error("typed L2 way contains a sector mask outside the four-sector line")]
    InvalidSectorMask,
    #[error("complete typed L2 way {line:?} has no allocation generation")]
    WayGenerationMissing { line: CanonicalLineKey },
    #[error("typed L2 geometry uses an unknown slice hash")]
    UnknownSliceHash,
    #[error(
        "typed L2 line {line:?} is stored in slice/set ({found_slice},{found_set}), expected ({expected_slice},{expected_set})"
    )]
    LinePlacementMismatch {
        line: CanonicalLineKey,
        found_slice: u32,
        found_set: u64,
        expected_slice: u32,
        expected_set: u64,
    },
    #[error("typed L2 contains duplicate allocation generations")]
    DuplicateAllocation,
    #[error("typed L2 reuses allocation generation {generation} across live lines")]
    DuplicateGeneration { generation: u64 },
    #[error("typed L2 next-generation frontier is not above live generations")]
    GenerationFrontierMismatch,
    #[error("typed L2 allocations are not in canonical line/generation order")]
    UnsortedAllocations,
    #[error("typed L2 allocation contains duplicate or out-of-range sectors")]
    DuplicateAllocationSector,
    #[error("typed L2 allocation sectors are not in ascending order")]
    UnsortedAllocationSectors,
    #[error("typed L2 contains duplicate requester ownership")]
    DuplicateRequestOwner,
    #[error("typed L2 response identity is duplicated")]
    DuplicateResponseIdentity,
    #[error("typed L2 requester ownership is not in canonical order")]
    UnsortedRequestOwners,
    #[error("typed L2 request owner has a negative warp slot {warp_slot}")]
    NegativeRequestWarpSlot { warp_slot: i32 },
    #[error("typed L2 L1 response owner has an invalid sector mask")]
    InvalidRequestOwnerSectorMask,
    #[error("typed L2 L1 response owner uses a different address domain")]
    RequestOwnerAddressDomainMismatch,
    #[error("typed L2 L1 response owner is missing its allocation generation")]
    RequestOwnerGenerationMissing,
    #[error("typed L2 L1 response owner has no matching allocation")]
    RequestOwnerAllocationMissing,
    #[error("typed L2 L1 response owner references sectors outside its allocation")]
    RequestOwnerSectorMismatch,
    #[error("typed L2 opaque response owner carries cache allocation identity")]
    OpaqueRequestOwnerIdentity,
    #[error("typed L2 response owner has no supported response kind")]
    InvalidRequestOwnerKind,
    #[error("typed L2 allocation and resident-way identities disagree")]
    AllocationWayMismatch,
    #[error("typed L2 allocation has no matching resident way")]
    AllocationWithoutWay,
    #[error("typed L2 allocation sector statuses disagree with resident masks")]
    AllocationMaskMismatch {
        line: CanonicalLineKey,
        generation: u64,
    },
    #[error("typed L2 non-pending sector carries requester ownership")]
    NonPendingOwnership {
        line: CanonicalLineKey,
        generation: u64,
        sector: u8,
    },
    #[error("typed L2 non-pending sector carries a ready time without an outstanding fill owner")]
    ReadyTimeOnNonPending {
        line: CanonicalLineKey,
        generation: u64,
        sector: u8,
        ready_at: u64,
    },
    #[error("typed L2 sector consumer is duplicated")]
    DuplicateSectorConsumer,
    #[error("complete typed L2 pending sector references a non-outstanding request owner")]
    PendingConsumerNotOutstanding {
        line: CanonicalLineKey,
        generation: u64,
        sector: u8,
    },
    #[error("typed L2 sector consumers are not in canonical request order")]
    UnsortedSectorConsumers,
    #[error("typed L2 sector ownership references an unknown request")]
    OwnershipReferenceMissing,
    #[error(
        "typed L2 sector ownership references a different line, sector, or allocation generation"
    )]
    OwnershipReferenceMismatch {
        line: CanonicalLineKey,
        generation: u64,
        sector: u8,
    },
    #[error("typed L2 producer is not one of the sector consumers")]
    ProducerNotConsumer,
    #[error("complete typed L2 ownership has a pending sector without a producer")]
    CompletePendingWithoutProducer {
        line: CanonicalLineKey,
        generation: u64,
        sector: u8,
    },
    #[error("complete typed L2 ownership has a pending sector without a ready time")]
    CompletePendingWithoutReadyTime {
        line: CanonicalLineKey,
        generation: u64,
        sector: u8,
    },
    #[error("typed L2 way has pending and valid sectors in the same bit")]
    PendingAndValidOverlap,
    #[error("typed L2 way has dirty sectors that are not valid")]
    DirtyWithoutValid,
    #[error("typed L2 state is marked complete while unknown fields remain")]
    CompleteWithUnknownState,
    #[error("typed L2 state is incomplete without an unknown-state reason")]
    IncompleteWithoutUnknown,
    #[error("typed L2 unknown-state reasons are not in canonical order")]
    UnknownStateOrder,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TypedEventPayload {
    SmWake,
    /// An explicit value-bearing L1 admission supplied by an external event
    /// producer. The typed boundary retains only bounded cardinalities; the
    /// payload bytes and allocation identities remain in the native event
    /// input and are never treated as hardware completion evidence.
    L1PayloadAdmission {
        payload_records: u64,
        witness_lines: u64,
        identity_present: bool,
    },
    L2Access {
        request: TypedMemoryRequest,
    },
    L2AccessBatch {
        requests: Vec<TypedMemoryRequest>,
    },
    L2Fill {
        fill: TypedFillIdentity,
    },
    AtomicCompletion,
    WritebackCompletion {
        token: u64,
        line_id: i64,
        dirty_mask: u8,
        service_done_at: u64,
    },
    CoherenceCompletion {
        token: u64,
        line_id: i64,
        sector_mask: u8,
        service_sectors: u32,
        service_done_at: u64,
    },
    L1VisibilityCompletion {
        token: u64,
        line_id: i64,
        sector_mask: u8,
        service_sectors: u32,
        service_done_at: u64,
    },
    L1VisibilityAccess {
        sequence: u64,
        requester: u8,
        line_id: i64,
        sector_mask: u8,
        is_write: bool,
        l1_resident_after: bool,
        fill_will_reside: bool,
        fill_generation: Option<u32>,
        hit: bool,
        issue_cycle: u64,
    },
    L1VisibilityFence {
        sequence: u64,
        requester: u8,
        pc: u32,
        issue_cycle: u64,
        scope: String,
        order: String,
    },
    L1FillVisible {
        sequence: u64,
        requester: u8,
        line_id: i64,
        sector_mask: u8,
        fill_generation: u32,
        visible_cycle: u64,
    },
    L1InvalidateAll {
        sequence: u64,
        requester: u8,
        issue_cycle: u64,
    },
    MemoryWriteAck {
        warp: u32,
        instruction: u32,
        subcore: u8,
        request_index: u16,
        request_count: u16,
        operation: TypedMemoryOperation,
        issue_at: u64,
        accepted_at: u64,
        service_done_at: u64,
        fence_visible_at: u64,
    },
    MemoryResponse {
        reply: Option<TypedResponseIdentity>,
        fill: Option<TypedFillIdentity>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedResponseIdentity {
    pub warp: u64,
    pub instruction: u64,
    pub sector_index: u64,
    pub response_kind: TypedResponseKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedFillIdentity {
    pub line_id: i64,
    pub generation: u64,
    pub sector_mask: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedAdmissionWitness {
    pub warp: u32,
    pub instruction: u32,
    pub subcore: u8,
    pub request_index: u16,
    pub request_count: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedMemoryRequest {
    pub issue_time: u64,
    pub sector_id: i64,
    pub origin: TypedMemoryOrigin,
    pub sector_mask: u8,
    pub local_sector_mask: u8,
    pub source: TypedL2Source,
    pub operation: TypedMemoryOperation,
    pub constant_load: bool,
    pub constant_deep: bool,
    pub reply: Option<TypedResponseIdentity>,
    pub admission_witness: Option<TypedAdmissionWitness>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedPendingEvent {
    pub time: u64,
    pub sequence: u64,
    /// Current engine phase. Phase zero is the engine's single serialized
    /// class; future multi-phase engines must expose their ordering here.
    pub phase: u8,
    /// Actual engine reservation, including the anchor sequence.
    pub sequence_span: u64,
    pub target: u32,
    pub payload: TypedEventPayload,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueueBoundaryState {
    pub depth: u64,
    pub next_event_time: Option<u64>,
    /// Next unused insertion position; null explicitly denotes exhaustion.
    /// Absence is a malformed snapshot, not an exhausted queue.
    #[serde(deserialize_with = "Option::<u64>::deserialize")]
    pub next_sequence: Option<u64>,
    pub pending: Vec<TypedPendingEvent>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum EventSequenceError {
    #[error("event at sequence {sequence} reserves zero positions")]
    EmptySpan { sequence: u64 },
    #[error("event at sequence {sequence} overflows its {span}-position reservation")]
    SpanOverflow { sequence: u64, span: u64 },
    #[error(
        "event at sequence {sequence} reserves {span} positions for {logical_events} logical events"
    )]
    PayloadSpanMismatch {
        sequence: u64,
        span: u64,
        logical_events: usize,
    },
}

pub(crate) fn sequence_end(sequence: u64, span: u64) -> Result<u64, EventSequenceError> {
    let offset = span
        .checked_sub(1)
        .ok_or(EventSequenceError::EmptySpan { sequence })?;
    sequence
        .checked_add(offset)
        .ok_or(EventSequenceError::SpanOverflow { sequence, span })
}

impl TypedEventPayload {
    pub(crate) fn validate_event_time(&self, event_time: u64) -> Result<(), BoundaryStateError> {
        let check = |payload: &'static str, field: &'static str, found: u64| {
            if found != event_time {
                return Err(BoundaryStateError::PayloadTimeMismatch {
                    payload,
                    field,
                    expected: event_time,
                    found,
                });
            }
            Ok(())
        };
        match self {
            Self::L2Access { request } => {
                request.validate_contract()?;
                check("l2_access", "issue_time", request.issue_time)
            }
            Self::L2AccessBatch { requests } => {
                validate_batch_witnesses(requests)?;
                requests.iter().try_for_each(|request| {
                    request.validate_contract()?;
                    check("l2_access_batch", "issue_time", request.issue_time)
                })
            }
            Self::MemoryWriteAck {
                operation,
                subcore,
                request_index,
                request_count,
                issue_at,
                accepted_at,
                service_done_at,
                fence_visible_at,
                ..
            } => {
                if !matches!(
                    operation,
                    TypedMemoryOperation::Store | TypedMemoryOperation::Reduction
                ) {
                    return Err(BoundaryStateError::MemoryWriteAckOperation {
                        operation: *operation,
                    });
                }
                if *subcore >= 4 || *request_count == 0 || *request_index >= *request_count {
                    return Err(BoundaryStateError::MemoryWriteAckWitness {
                        subcore: *subcore,
                        request_index: *request_index,
                        request_count: *request_count,
                    });
                }
                if accepted_at < issue_at {
                    return Err(BoundaryStateError::PayloadTimestampOrder {
                        payload: "memory_write_ack",
                        earlier_field: "issue_at",
                        earlier: *issue_at,
                        later_field: "accepted_at",
                        later: *accepted_at,
                    });
                }
                if service_done_at < accepted_at {
                    return Err(BoundaryStateError::PayloadTimestampOrder {
                        payload: "memory_write_ack",
                        earlier_field: "accepted_at",
                        earlier: *accepted_at,
                        later_field: "service_done_at",
                        later: *service_done_at,
                    });
                }
                if fence_visible_at < service_done_at {
                    return Err(BoundaryStateError::PayloadTimestampOrder {
                        payload: "memory_write_ack",
                        earlier_field: "service_done_at",
                        earlier: *service_done_at,
                        later_field: "fence_visible_at",
                        later: *fence_visible_at,
                    });
                }
                check("memory_write_ack", "fence_visible_at", *fence_visible_at)
            }
            Self::L2Fill { fill } => validate_fill_identity(fill),
            Self::MemoryResponse { reply, fill } => {
                let Some(reply) = reply else {
                    return Err(BoundaryStateError::MemoryResponseReplyMissing);
                };
                validate_response_identity(reply)?;
                if let Some(fill) = fill {
                    validate_fill_identity(fill)?;
                    if !matches!(
                        reply.response_kind,
                        TypedResponseKind::L1Fill
                            | TypedResponseKind::L1SectorFill
                            | TypedResponseKind::Warp
                            | TypedResponseKind::AsyncCopy
                    ) {
                        return Err(BoundaryStateError::MemoryResponseFillReplyMismatch {
                            response_kind: reply.response_kind,
                        });
                    }
                }
                Ok(())
            }
            Self::SmWake | Self::AtomicCompletion => Ok(()),
            Self::L1PayloadAdmission { .. } => Ok(()),
            Self::WritebackCompletion {
                service_done_at, ..
            } => check("writeback_completion", "service_done_at", *service_done_at),
            Self::CoherenceCompletion {
                service_done_at, ..
            } => check("coherence_completion", "service_done_at", *service_done_at),
            Self::L1VisibilityCompletion {
                service_done_at, ..
            } => check(
                "l1_visibility_completion",
                "service_done_at",
                *service_done_at,
            ),
            Self::L1VisibilityAccess { issue_cycle, .. } => {
                check("l1_visibility_access", "issue_cycle", *issue_cycle)
            }
            Self::L1VisibilityFence { issue_cycle, .. } => {
                check("l1_visibility_fence", "issue_cycle", *issue_cycle)
            }
            Self::L1FillVisible { visible_cycle, .. } => {
                check("l1_fill_visible", "visible_cycle", *visible_cycle)
            }
            Self::L1InvalidateAll { issue_cycle, .. } => {
                check("l1_invalidate_all", "issue_cycle", *issue_cycle)
            }
        }
    }
}

impl TypedEventPayload {
    pub(crate) fn sequence_end(&self, sequence: u64, span: u64) -> Result<u64, EventSequenceError> {
        let end = sequence_end(sequence, span)?;
        let logical_events = match self {
            Self::L2AccessBatch { requests } => requests.len(),
            _ => 1,
        };
        if u64::try_from(logical_events) != Ok(span) {
            return Err(EventSequenceError::PayloadSpanMismatch {
                sequence,
                span,
                logical_events,
            });
        }
        Ok(end)
    }
}

impl QueueBoundaryState {
    pub fn validate(&self) -> Result<(), QueueStateError> {
        if self.depth != u64::try_from(self.pending.len()).unwrap_or(u64::MAX) {
            return Err(QueueStateError::DepthMismatch {
                depth: self.depth,
                pending: self.pending.len(),
            });
        }
        let mut previous = None;
        let mut reservations = BTreeMap::new();
        for event in &self.pending {
            let end = event
                .payload
                .sequence_end(event.sequence, event.sequence_span)?;
            if reservations.insert(event.sequence, end).is_some() {
                return Err(QueueStateError::DuplicateSequence {
                    sequence: event.sequence,
                });
            }
            self.validate_reservation_end(end)?;
            if let Some(previous) = previous
                && (event.time, event.phase, event.sequence) <= previous
            {
                return Err(QueueStateError::Unsorted);
            }
            previous = Some((event.time, event.phase, event.sequence));
        }
        // Timestamp order need not be insertion order. Reservations occupy
        // one shared sequence space across all timestamps; never expand a
        // span into individual positions to check for overlap.
        let mut previous_end = None;
        for (sequence, end) in reservations {
            if let Some(previous_end) = previous_end
                && sequence <= previous_end
            {
                return Err(QueueStateError::OverlappingSpans {
                    sequence,
                    previous_end,
                });
            }
            previous_end = Some(end);
        }
        if self.next_event_time != self.pending.first().map(|event| event.time) {
            return Err(QueueStateError::NextTimeMismatch);
        }
        Ok(())
    }
}

impl QueueBoundaryState {
    pub(crate) fn validate_reservation_end(&self, end: u64) -> Result<(), QueueStateError> {
        if let Some(next_sequence) = self.next_sequence
            && end >= next_sequence
        {
            return Err(QueueStateError::UnreservedSequence { end, next_sequence });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedEventIdentity {
    pub time: u64,
    pub sequence: u64,
    pub sequence_span: u64,
    pub phase: u8,
    pub target: u32,
    pub payload: TypedEventPayload,
}

pub(crate) fn validate_warp_instruction_buffer(
    warp: &WarpBoundaryState,
    front_end: Option<&InstructionFrontEndBoundaryState>,
) -> Result<(), CoreStateError> {
    if warp.done && !warp.instruction_buffer.is_empty() {
        return Err(CoreStateError::DoneWarpInstructionBuffer {
            warp_slot: warp.warp_slot,
        });
    }
    let Some(front_end) = front_end else {
        if !warp.instruction_buffer.is_empty() {
            return Err(CoreStateError::InstructionBufferWithoutFrontEnd {
                warp_slot: warp.warp_slot,
            });
        }
        return Ok(());
    };
    let capacity = usize::try_from(front_end.instruction_buffer_entries).unwrap_or(usize::MAX);
    if warp.instruction_buffer.len() > capacity {
        return Err(CoreStateError::InstructionBufferCapacity {
            warp_slot: warp.warp_slot,
            found: warp.instruction_buffer.len(),
            capacity,
        });
    }
    if warp.front_end_fetch_cursor < warp.next_instruction {
        return Err(CoreStateError::InstructionCursorBeforeNext {
            warp_slot: warp.warp_slot,
            cursor: warp.front_end_fetch_cursor,
            next: warp.next_instruction,
        });
    }
    let mut previous: Option<u64> = None;
    for &(instruction, _) in &warp.instruction_buffer {
        if instruction < warp.next_instruction {
            return Err(CoreStateError::InstructionBufferBeforeNext {
                warp_slot: warp.warp_slot,
                next: warp.next_instruction,
                found: instruction,
            });
        }
        if instruction >= warp.front_end_fetch_cursor {
            return Err(CoreStateError::InstructionBufferBeyondCursor {
                warp_slot: warp.warp_slot,
                cursor: warp.front_end_fetch_cursor,
                found: instruction,
            });
        }
        if let Some(previous) = previous
            && instruction != previous.saturating_add(1)
        {
            return Err(CoreStateError::InstructionBufferOrder {
                warp_slot: warp.warp_slot,
                previous,
                found: instruction,
            });
        }
        previous = Some(instruction);
    }
    Ok(())
}

pub(crate) fn validate_scalar_pipeline(
    state: &ScalarPipelineBoundaryState,
    cycle: u64,
) -> Result<(), CoreStateError> {
    if state.latency_cycles == 0
        || state.initiation_interval_cycles == 0
        || state.in_flight_capacity == 0
        || state.release_times.len() != 4
        || state.next_service.len() != 4
    {
        return Err(CoreStateError::ScalarPipelineShape);
    }
    let mut live_entries = 0_u64;
    let mut maximum_live_entries = 0_u64;
    for (subcore, queue) in state.release_times.iter().enumerate() {
        if queue.iter().any(|release| *release < cycle) {
            return Err(CoreStateError::ScalarPipelineReleaseInPast);
        }
        if queue.windows(2).any(|window| window[1] <= window[0]) {
            return Err(CoreStateError::ScalarPipelineOrder);
        }
        if queue.len() > state.in_flight_capacity as usize {
            return Err(CoreStateError::ScalarPipelineCapacity);
        }
        live_entries = live_entries
            .checked_add(u64::try_from(queue.len()).unwrap_or(u64::MAX))
            .ok_or(CoreStateError::CounterOverflow)?;
        maximum_live_entries =
            maximum_live_entries.max(u64::try_from(queue.len()).unwrap_or(u64::MAX));
        if let Some(last_release) = queue.last() {
            let required_service = last_release
                .checked_add(state.initiation_interval_cycles)
                .ok_or(CoreStateError::CounterOverflow)?;
            if state.next_service[subcore] < required_service {
                return Err(CoreStateError::ScalarPipelineServiceFrontier);
            }
        }
        if state.next_service[subcore] != 0 {
            let first_service = state
                .latency_cycles
                .checked_add(state.initiation_interval_cycles)
                .ok_or(CoreStateError::CounterOverflow)?;
            if state.next_service[subcore] < first_service {
                return Err(CoreStateError::ScalarPipelineServiceOrigin);
            }
        }
    }
    if state.instructions_admitted < live_entries {
        return Err(CoreStateError::ScalarPipelineAdmissionCount);
    }
    // Every native scalar admission advances the selected SMSP's service
    // frontier. Once all release entries have drained, that frontier remains
    // nonzero; an aggregate admitted count with no live entries and four zero
    // frontiers cannot be produced by the executable pipeline.
    if state.instructions_admitted != 0
        && live_entries == 0
        && state.next_service.iter().all(|&ready| ready == 0)
    {
        return Err(CoreStateError::ScalarPipelineHistory);
    }
    if state.maximum_queue_occupancy < maximum_live_entries
        || state.maximum_queue_occupancy > u64::from(state.in_flight_capacity)
        || (state.instructions_admitted != 0 && state.maximum_queue_occupancy == 0)
        || state.maximum_queue_occupancy > state.instructions_admitted
    {
        return Err(CoreStateError::ScalarPipelineOccupancy);
    }
    if state.instructions_admitted == 0
        && (state.next_service.iter().any(|&ready| ready != 0) || state.queue_delay_cycles != 0)
    {
        return Err(CoreStateError::ScalarPipelineDiagnostics);
    }
    let expected_busy = state
        .instructions_admitted
        .checked_mul(state.initiation_interval_cycles)
        .ok_or(CoreStateError::CounterOverflow)?;
    if state.service_busy_cycles != expected_busy {
        return Err(CoreStateError::ScalarPipelineBusyCycles);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CoreStateError {
    #[error("typed core local completion frontier disagrees with CTA-barrier release history")]
    LocalCompletionHistory,
    #[error("typed core SM identity uses the invalid u32::MAX sentinel")]
    InvalidSmIdentity,
    #[error("typed core must expose four SM subcore rosters, found {found}")]
    SubcoreCount { found: usize },
    #[error("typed core unprojected timing-state reasons are not canonical")]
    UnprojectedTimingState,
    #[error("typed core contains a negative warp slot {warp_slot}")]
    NegativeWarpSlot { warp_slot: i32 },
    #[error(
        "typed core warp slot {warp_slot} is listed on SMSP {subcore}, but SM86 maps it to SMSP {expected_subcore}"
    )]
    ResidentWarpWrongSubcore {
        warp_slot: i32,
        subcore: usize,
        expected_subcore: usize,
    },
    #[error("typed core contains duplicate resident warp slot {warp_slot}")]
    DuplicateResidentWarp { warp_slot: i32 },
    #[error("typed core resident warps reuse arrival stamp {arrival}")]
    DuplicateResidentWarpArrival { arrival: u64 },
    #[error("typed core issued count {declared} disagrees with warp cursors {warps}")]
    IssuedCount { declared: u64, warps: u64 },
    #[error("done typed warp {warp_slot} retains instruction-buffer entries")]
    DoneWarpInstructionBuffer { warp_slot: i32 },
    #[error("typed warp {warp_slot} carries instruction-buffer entries without a front-end")]
    InstructionBufferWithoutFrontEnd { warp_slot: i32 },
    #[error(
        "typed warp {warp_slot} instruction buffer has {found} entries, exceeding capacity {capacity}"
    )]
    InstructionBufferCapacity {
        warp_slot: i32,
        found: usize,
        capacity: usize,
    },
    #[error("typed warp {warp_slot} fetch cursor {cursor} is before next instruction {next}")]
    InstructionCursorBeforeNext {
        warp_slot: i32,
        cursor: u64,
        next: u64,
    },
    #[error(
        "typed warp {warp_slot} instruction buffer contains {found} before next instruction {next}"
    )]
    InstructionBufferBeforeNext {
        warp_slot: i32,
        next: u64,
        found: u64,
    },
    #[error(
        "typed warp {warp_slot} instruction buffer contains {found} at or beyond fetch cursor {cursor}"
    )]
    InstructionBufferBeyondCursor {
        warp_slot: i32,
        cursor: u64,
        found: u64,
    },
    #[error(
        "typed warp {warp_slot} instruction buffer is not contiguous after {previous}; found {found}"
    )]
    InstructionBufferOrder {
        warp_slot: i32,
        previous: u64,
        found: u64,
    },
    #[error("typed core resident-warp count {declared} disagrees with roster size {roster}")]
    ResidentWarpCount { declared: u32, roster: usize },
    #[error(
        "typed core resident-warp count {declared} disagrees with active block reservation {reservation}"
    )]
    ResidentWarpReservationCount { declared: u32, reservation: u64 },
    #[error("typed core resident warp slot {warp_slot} has no warp state")]
    ResidentWarpMissing { warp_slot: i32 },
    #[error("typed core unfinished warp slot {warp_slot} is absent from the resident roster")]
    UnfinishedWarpNotResident { warp_slot: i32 },
    #[error(
        "typed core resident roster has {roster} entries but {unfinished_states} unfinished warp states"
    )]
    CurrentRosterCount {
        roster: usize,
        unfinished_states: u64,
    },
    #[error(
        "typed core block {block} declares {declared} live warps but only {admitted} are admitted"
    )]
    BlockWarpCount {
        block: usize,
        declared: u32,
        admitted: u64,
    },
    #[error("typed core block {block} unfinished count exceeds admitted warps")]
    BlockUnfinishedCount { block: usize },
    #[error(
        "typed core block {block} unfinished count {unfinished} exceeds live warp count {warp_count}"
    )]
    BlockUnfinishedExceedsWarpCount {
        block: usize,
        unfinished: u64,
        warp_count: u32,
    },
    #[error("typed core block {block} waiting count exceeds unfinished warps")]
    BlockWaitingCount { block: usize },
    #[error("typed core block {block} warp roster has {roster} entries, expected {declared}")]
    BlockWarpRoster {
        block: usize,
        declared: u64,
        roster: usize,
    },
    #[error(
        "typed core block {block} declares {declared} unfinished warps but has {states} unfinished warp states"
    )]
    BlockWarpStateCount {
        block: usize,
        declared: u64,
        states: u64,
    },
    #[error("typed core block references unknown warp slot {warp_slot}")]
    BlockWarpMissing { warp_slot: i32 },
    #[error("typed core block rosters duplicate warp slot {warp_slot}")]
    DuplicateBlockWarp { warp_slot: i32 },
    #[error("typed core block {block} first-warp frontier expected {expected}, found {found}")]
    BlockFirstWarpOrder {
        block: usize,
        expected: u64,
        found: u64,
    },
    #[error(
        "typed core block {block} warp range ({first_warp}, {admitted_warps}) exceeds {warp_states} warp states"
    )]
    BlockWarpRange {
        block: usize,
        first_warp: u64,
        admitted_warps: u64,
        warp_states: usize,
    },
    #[error("typed core block {block} warp range has a different slot roster")]
    BlockWarpRangeRoster { block: usize },
    #[error("typed core block {block} warp range contains a warp for block {found}")]
    BlockWarpRangeIdentity { block: usize, found: u64 },
    #[error("typed core warp arena ends at {found}, but block ranges cover only {expected} states")]
    WarpArenaTail { expected: u64, found: u64 },
    #[error(
        "typed core block {block} waiting count {declared} disagrees with barrier flags {observed}"
    )]
    BlockBarrierWaitingCount {
        block: usize,
        declared: u64,
        observed: u64,
    },
    #[error("typed core block {block} warp {warp_slot} is done while marked at its CTA barrier")]
    BlockBarrierDoneWarp { block: usize, warp_slot: i32 },
    #[error(
        "typed core block {block} warp {warp_slot} has barrier timestamps without its waiting flag"
    )]
    BlockBarrierTimestampWithoutFlag { block: usize, warp_slot: i32 },
    #[error("typed core block {block} warp {warp_slot} is missing a barrier timestamp")]
    BlockBarrierTimestampMissing { block: usize, warp_slot: i32 },
    #[error(
        "typed core block {block} warp {warp_slot} barrier dispatch {dispatch} is after core cycle {cycle}"
    )]
    BlockBarrierDispatchAfterCycle {
        block: usize,
        warp_slot: i32,
        dispatch: u64,
        cycle: u64,
    },
    #[error(
        "typed core block {block} warp {warp_slot} barrier dispatch {dispatch} is after arrival {arrival}"
    )]
    BlockBarrierTimestampOrder {
        block: usize,
        warp_slot: i32,
        dispatch: u64,
        arrival: u64,
    },
    #[error(
        "typed core block {block} first barrier dispatch expected {expected:?}, found {found:?}"
    )]
    BlockBarrierFirstDispatch {
        block: usize,
        expected: Option<u64>,
        found: Option<u64>,
    },
    #[error(
        "typed core block {block} first barrier arrival expected {expected:?}, found {found:?}"
    )]
    BlockBarrierFirstArrival {
        block: usize,
        expected: Option<u64>,
        found: Option<u64>,
    },
    #[error(
        "typed core block {block} has re-arm frontier {rearm_ready} before any barrier generation"
    )]
    BlockBarrierRearmWithoutGeneration { block: usize, rearm_ready: u64 },
    #[error(
        "typed core active blocks {previous_block} and {block} both retain warp slot {warp_slot}"
    )]
    ActiveBlockWarpSlotCollision {
        warp_slot: i32,
        previous_block: usize,
        block: usize,
    },
    #[error("typed core unfinished warp count {declared} disagrees with blocks {blocks}")]
    UnfinishedWarpCount { declared: u64, blocks: u64 },
    #[error(
        "typed core declares {declared} unfinished warps but exposes {states} unfinished warp states"
    )]
    WarpStateCount { declared: u64, states: u64 },
    #[error(
        "typed core resident shared-memory count {declared} disagrees with active blocks {blocks}"
    )]
    SharedMemoryCount { declared: u64, blocks: u64 },
    #[error("typed core accounting counter overflowed")]
    CounterOverflow,
    #[error("typed memory pipeline shape or configuration is invalid")]
    MemoryPipelineShape,
    #[error("typed memory pipeline release times are not strictly ordered")]
    MemoryPipelineOrder,
    #[error("typed memory pipeline queue exceeds its declared capacity")]
    MemoryPipelineCapacity,
    #[error("typed memory pipeline admitted-count is below live queue entries")]
    MemoryPipelineAdmissionCount,
    #[error("typed memory pipeline has admissions but no local frontier history")]
    MemoryPipelineHistory,
    #[error("typed memory pipeline occupancy diagnostic is inconsistent")]
    MemoryPipelineOccupancy,
    #[error("typed memory pipeline maximum delay exceeds its accumulated delay")]
    MemoryPipelineDelayMaximum,
    #[error("typed memory pipeline has diagnostics despite zero admissions")]
    MemoryPipelineDiagnostics,
    #[error("typed memory pipeline terminal cycle is before a live acceptance")]
    MemoryPipelineTerminalCycle,
    #[error("typed memory pipeline shared service frontier is inconsistent with its releases")]
    MemoryPipelineSharedFrontier,
    #[error("typed memory pipeline local completion frontier exceeds shared acceptance")]
    MemoryPipelineLocalFrontier,
    #[error("typed memory pipeline live queue {subcore} has no local completion frontier")]
    MemoryPipelineLocalFrontierMissing { subcore: usize },
    #[error("typed memory pipeline contains a release time before the core cycle")]
    MemoryPipelineReleaseInPast,
    #[error("typed scalar pipeline shape or configuration is invalid")]
    ScalarPipelineShape,
    #[error("typed scalar pipeline release times are not strictly ordered")]
    ScalarPipelineOrder,
    #[error("typed scalar pipeline queue exceeds its declared capacity")]
    ScalarPipelineCapacity,
    #[error("typed scalar pipeline admitted-count is below live queue entries")]
    ScalarPipelineAdmissionCount,
    #[error("typed scalar pipeline has admissions but no service frontier history")]
    ScalarPipelineHistory,
    #[error("typed scalar pipeline occupancy diagnostic is inconsistent")]
    ScalarPipelineOccupancy,
    #[error("typed scalar pipeline service frontier is behind a live release")]
    ScalarPipelineServiceFrontier,
    #[error("typed scalar pipeline service frontier precedes its first possible service")]
    ScalarPipelineServiceOrigin,
    #[error("typed scalar pipeline busy-cycle diagnostic is inconsistent")]
    ScalarPipelineBusyCycles,
    #[error("typed scalar pipeline has diagnostics despite zero admissions")]
    ScalarPipelineDiagnostics,
    #[error("typed scalar pipeline contains a release time before the core cycle")]
    ScalarPipelineReleaseInPast,
    #[error(transparent)]
    L1(#[from] L1StateError),
    #[error(transparent)]
    FrontEnd(#[from] FrontEndStateError),
    #[error(transparent)]
    AsyncCopy(#[from] AsyncCopyStateError),
    #[error("typed async-copy slot count does not match dynamic warp state")]
    AsyncCopySlots,
    #[error("typed core hold references non-resident warp slot {warp_slot}")]
    HoldWarpMissing { warp_slot: u64 },
    #[error(
        "typed core hold for SMSP {subcore} references warp slot {warp_slot} owned by another SMSP"
    )]
    HoldWarpWrongSubcore { subcore: usize, warp_slot: i32 },
    #[error("typed core contains duplicate warp state for slot {warp_slot}")]
    DuplicateWarpState { block: u64, warp_slot: i32 },
    #[error("typed core warp references unknown block {block}")]
    UnknownWarpBlock { block: u64 },
    #[error("typed core outstanding response count {declared} disagrees with owners {owners}")]
    ResponseCount { declared: u64, owners: usize },
    #[error("typed core response owner belongs to SM {owner}")]
    ResponseOwnerSm { owner: u32 },
    #[error(
        "typed core response owner warp index {response_warp} is outside {warp_states} serialized warp states"
    )]
    ResponseOwnerWarpOutOfRange {
        response_warp: u64,
        warp_states: usize,
    },
    #[error(
        "typed core response owner warp index {response_warp} names slot {state_warp_slot}, but owner carries slot {owner_warp_slot}"
    )]
    ResponseOwnerWarpSlotMismatch {
        response_warp: u64,
        owner_warp_slot: i32,
        state_warp_slot: i32,
    },
    #[error(
        "typed core response owner instruction {instruction} is after warp {response_warp} cursor {next_instruction}"
    )]
    ResponseOwnerInstructionAfterCursor {
        response_warp: u64,
        instruction: u64,
        next_instruction: u64,
    },
    #[error("typed core response owner is not outstanding")]
    ResponseOwnerState,
    #[error("typed core response owner has a negative warp slot {warp_slot}")]
    NegativeResponseOwnerWarpSlot { warp_slot: i32 },
    #[error("typed core contains duplicate response ownership")]
    DuplicateResponseOwner,
    #[error("typed core response owners are not in canonical order")]
    UnsortedResponseOwners,
    #[error(
        "typed core next-block frontier {next_block} disagrees with block states {block_states}"
    )]
    NextBlockMismatch {
        next_block: u64,
        block_states: usize,
    },
}
