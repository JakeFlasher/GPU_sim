// Lexical implementation section 3/3; included by the parent facade.
fn validate_scalar_pipeline(
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

fn validate_block_barrier_state(
    block_index: usize,
    block: &BlockBoundaryState,
    warps: &[WarpBoundaryState],
    cycle: u64,
) -> Result<(), CoreStateError> {
    if block.barrier_generation == 0 && block.barrier_rearm_ready != 0 {
        return Err(CoreStateError::BlockBarrierRearmWithoutGeneration {
            block: block_index,
            rearm_ready: block.barrier_rearm_ready,
        });
    }
    let mut observed_waiting = 0_u64;
    let mut first_dispatch: Option<u64> = None;
    let mut first_arrival: Option<u64> = None;
    for warp in warps {
        if !warp.at_block_barrier {
            if warp.block_barrier_dispatch.is_some() || warp.block_barrier_arrival.is_some() {
                return Err(CoreStateError::BlockBarrierTimestampWithoutFlag {
                    block: block_index,
                    warp_slot: warp.warp_slot,
                });
            }
            continue;
        }
        if warp.done {
            return Err(CoreStateError::BlockBarrierDoneWarp {
                block: block_index,
                warp_slot: warp.warp_slot,
            });
        }
        let dispatch =
            warp.block_barrier_dispatch
                .ok_or(CoreStateError::BlockBarrierTimestampMissing {
                    block: block_index,
                    warp_slot: warp.warp_slot,
                })?;
        let arrival =
            warp.block_barrier_arrival
                .ok_or(CoreStateError::BlockBarrierTimestampMissing {
                    block: block_index,
                    warp_slot: warp.warp_slot,
                })?;
        if dispatch > cycle {
            return Err(CoreStateError::BlockBarrierDispatchAfterCycle {
                block: block_index,
                warp_slot: warp.warp_slot,
                dispatch,
                cycle,
            });
        }
        if arrival < dispatch {
            return Err(CoreStateError::BlockBarrierTimestampOrder {
                block: block_index,
                warp_slot: warp.warp_slot,
                dispatch,
                arrival,
            });
        }
        observed_waiting = observed_waiting
            .checked_add(1)
            .ok_or(CoreStateError::CounterOverflow)?;
        first_dispatch = Some(first_dispatch.map_or(dispatch, |current| current.min(dispatch)));
        first_arrival = Some(first_arrival.map_or(arrival, |current| current.min(arrival)));
    }
    if observed_waiting != block.waiting {
        return Err(CoreStateError::BlockBarrierWaitingCount {
            block: block_index,
            declared: block.waiting,
            observed: observed_waiting,
        });
    }
    if first_dispatch != block.barrier_first_dispatch {
        return Err(CoreStateError::BlockBarrierFirstDispatch {
            block: block_index,
            expected: first_dispatch,
            found: block.barrier_first_dispatch,
        });
    }
    if first_arrival != block.barrier_first_arrival {
        return Err(CoreStateError::BlockBarrierFirstArrival {
            block: block_index,
            expected: first_arrival,
            found: block.barrier_first_arrival,
        });
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedEventBoundaryRecord {
    pub schema_version: u32,
    pub contract: TypedObservationContract,
    pub event: TypedEventIdentity,
    pub queue: QueueBoundaryState,
    pub cores: Vec<TypedCoreBoundaryState>,
    pub shared_l2: L2BoundaryState,
    /// Required shared service state, distinct from per-SM issue frontiers.
    pub dram: DramBoundaryState,
    pub power: PowerBoundaryState,
    pub completeness: Completeness,
}

impl TypedEventBoundaryRecord {
    /// Validate the structural state of one boundary without requiring the
    /// selected contract to be complete. Incomplete contract evidence is a
    /// valid diagnostic result; malformed typed state is an error.
    pub fn validate_structure(&self) -> Result<(), BoundaryStateError> {
        if self.schema_version != TYPED_STATE_SCHEMA_VERSION
            || self.contract.schema_version != TYPED_STATE_SCHEMA_VERSION
        {
            return Err(BoundaryStateError::SchemaVersion);
        }
        if self.event.phase != 0 {
            return Err(BoundaryStateError::UnsupportedPhase {
                phase: self.event.phase,
            });
        }
        self.event.payload.validate_event_time(self.event.time)?;
        if !self.cores.is_empty()
            && usize::try_from(self.event.target)
                .ok()
                .is_none_or(|target| target >= self.cores.len())
        {
            return Err(BoundaryStateError::EventTargetOutOfRange {
                target: self.event.target,
                cores: self.cores.len(),
            });
        }
        self.queue.validate()?;
        let event_end = self.event.sequence_end()?;
        self.queue.validate_reservation_end(event_end)?;
        for pending in &self.queue.pending {
            if !self.cores.is_empty()
                && usize::try_from(pending.target)
                    .ok()
                    .is_none_or(|target| target >= self.cores.len())
            {
                return Err(BoundaryStateError::EventTargetOutOfRange {
                    target: pending.target,
                    cores: self.cores.len(),
                });
            }
            if pending.phase != 0 {
                return Err(BoundaryStateError::UnsupportedPhase {
                    phase: pending.phase,
                });
            }
            pending.payload.validate_event_time(pending.time)?;
            let pending_end = pending
                .payload
                .sequence_end(pending.sequence, pending.sequence_span)?;
            if pending.sequence <= event_end && self.event.sequence <= pending_end {
                return Err(BoundaryStateError::CurrentEventOverlap {
                    sequence: pending.sequence,
                });
            }
            if (pending.time, pending.phase, pending.sequence)
                <= (self.event.time, self.event.phase, self.event.sequence)
            {
                return Err(BoundaryStateError::PendingEventOrder {
                    time: pending.time,
                    sequence: pending.sequence,
                });
            }
        }
        let mut previous_sm_id = None;
        for core in &self.cores {
            // The event timestamp is the global queue frontier. An SM actor
            // may intentionally run ahead to its own response horizon before
            // the queue applies cross-component events, so validate each
            // local clock and state independently rather than imposing a
            // false ordering between the two domains.
            if let Some(previous) = previous_sm_id
                && core.sm_id <= previous
            {
                return Err(BoundaryStateError::CoreIdentityOrder {
                    previous,
                    found: core.sm_id,
                });
            }
            previous_sm_id = Some(core.sm_id);
            core.validate()?
        }
        self.shared_l2.validate()?;
        self.dram.validate()?;
        for allocation in &self.shared_l2.allocations {
            for sector in &allocation.sectors {
                if matches!(sector.status, L2SectorStatus::Pending)
                    && sector
                        .ready_at
                        .is_some_and(|ready_at| ready_at < self.event.time)
                {
                    return Err(BoundaryStateError::L2PendingReadyBeforeBoundary {
                        line: allocation.line.clone(),
                        generation: allocation.generation,
                        sector: sector.sector,
                        ready_at: sector.ready_at.expect("checked above"),
                        event_time: self.event.time,
                    });
                }
            }
        }
        let validate_owner_due = |owner: &L2RequestOwner| {
            owner
                .due
                .filter(|&due| due < self.event.time)
                .map_or(Ok(()), |due| {
                    Err(BoundaryStateError::ResponseDueBeforeBoundary {
                        sm_id: owner.request.sm_id,
                        due,
                        event_time: self.event.time,
                    })
                })
        };
        for owner in &self.shared_l2.request_owners {
            validate_owner_due(owner)?;
        }
        for core in &self.cores {
            for owner in &core.response_owners {
                validate_owner_due(owner)?;
            }
        }
        // A complete boundary must retain the queue event that will consume
        // every known-due response owner. Checking the two projections only
        // independently would allow an owner or a queued response to vanish
        // while all local shapes still validate.
        if !self.cores.is_empty()
            && self.shared_l2.ownership_complete
            && self
                .shared_l2
                .request_owners
                .iter()
                .all(|owner| owner.due.is_some())
        {
            type ResponseQueueKey = (u32, u64, u64, u64, TypedResponseKind, u64);
            let mut owner_counts = BTreeMap::<ResponseQueueKey, u32>::new();
            for owner in &self.shared_l2.request_owners {
                let key = (
                    owner.request.sm_id,
                    owner.request.response_warp,
                    owner.request.instruction,
                    owner.request.sector_index,
                    owner.request.response_kind,
                    owner.due.expect("known-due owner was checked above"),
                );
                *owner_counts.entry(key).or_default() += 1;
            }
            let mut queue_counts = BTreeMap::<ResponseQueueKey, u32>::new();
            for pending in &self.queue.pending {
                let TypedEventPayload::MemoryResponse {
                    reply: Some(reply), ..
                } = &pending.payload
                else {
                    continue;
                };
                let target = usize::try_from(pending.target).map_err(|_| {
                    BoundaryStateError::EventTargetOutOfRange {
                        target: pending.target,
                        cores: self.cores.len(),
                    }
                })?;
                let sm_id = self.cores[target].sm_id;
                let key = (
                    sm_id,
                    reply.warp,
                    reply.instruction,
                    reply.sector_index,
                    reply.response_kind,
                    pending.time,
                );
                *queue_counts.entry(key).or_default() += 1;
            }
            let mut keys = BTreeSet::new();
            keys.extend(owner_counts.keys().copied());
            keys.extend(queue_counts.keys().copied());
            for (sm_id, response_warp, instruction, sector_index, response_kind, due) in keys {
                if owner_counts.get(&(
                    sm_id,
                    response_warp,
                    instruction,
                    sector_index,
                    response_kind,
                    due,
                )) != queue_counts.get(&(
                    sm_id,
                    response_warp,
                    instruction,
                    sector_index,
                    response_kind,
                    due,
                )) {
                    return Err(BoundaryStateError::ResponseOwnerQueueJoin {
                        sm_id,
                        response_warp,
                        instruction,
                        sector_index,
                        response_kind,
                        due,
                    });
                }
            }

            type FillQueueKey = (i64, u64, u64);
            let mut expected_fill_masks = BTreeMap::<FillQueueKey, u8>::new();
            for allocation in &self.shared_l2.allocations {
                let CanonicalLineKey::VirtualAbsolute { line_id } = allocation.line else {
                    continue;
                };
                for sector in &allocation.sectors {
                    if !matches!(sector.status, L2SectorStatus::Pending) {
                        continue;
                    }
                    let Some(ready) = sector.ready_at else {
                        continue;
                    };
                    *expected_fill_masks
                        .entry((line_id, allocation.generation, ready))
                        .or_default() |= 1_u8 << sector.sector;
                }
            }
            let mut queued_fill_masks = BTreeMap::<FillQueueKey, u8>::new();
            for pending in &self.queue.pending {
                let fill = match &pending.payload {
                    TypedEventPayload::L2Fill { fill } => Some(fill),
                    TypedEventPayload::MemoryResponse {
                        fill: Some(fill), ..
                    } => Some(fill),
                    _ => None,
                };
                let Some(fill) = fill else { continue };
                let key = (fill.line_id, fill.generation, pending.time);
                let previous = queued_fill_masks.entry(key).or_default();
                if *previous & fill.sector_mask != 0 {
                    return Err(BoundaryStateError::L2FillQueueJoin {
                        line_id: fill.line_id,
                        generation: fill.generation,
                        ready: pending.time,
                    });
                }
                *previous |= fill.sector_mask;
            }
            let mut fill_keys = BTreeSet::new();
            fill_keys.extend(expected_fill_masks.keys().copied());
            fill_keys.extend(queued_fill_masks.keys().copied());
            for (line_id, generation, ready) in fill_keys {
                if expected_fill_masks.get(&(line_id, generation, ready))
                    != queued_fill_masks.get(&(line_id, generation, ready))
                {
                    return Err(BoundaryStateError::L2FillQueueJoin {
                        line_id,
                        generation,
                        ready,
                    });
                }
            }
        }
        self.power.validate()?;
        if self
            .power
            .domains
            .iter()
            .any(|domain| domain.last_cycle != self.event.time)
        {
            return Err(BoundaryStateError::PowerTimeMismatch {
                event_time: self.event.time,
            });
        }
        let derived = self
            .contract
            .validate(&self.shared_l2, &self.power, &self.cores);
        if self.completeness != derived {
            return Err(BoundaryStateError::CompletenessMismatch {
                declared: self.completeness.clone(),
                derived,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BoundaryStateError {
    #[error("typed event boundary schema version is unsupported")]
    SchemaVersion,
    #[error("typed event phase {phase} is unsupported by the single-phase engine")]
    UnsupportedPhase { phase: u8 },
    #[error(
        "typed {payload} payload field {field} has timestamp {found}, expected event time {expected}"
    )]
    PayloadTimeMismatch {
        payload: &'static str,
        field: &'static str,
        expected: u64,
        found: u64,
    },
    #[error(
        "typed {payload} payload timestamp {earlier_field}={earlier} is after {later_field}={later}"
    )]
    PayloadTimestampOrder {
        payload: &'static str,
        earlier_field: &'static str,
        earlier: u64,
        later_field: &'static str,
        later: u64,
    },
    #[error(
        "typed memory request for sector {sector_id} has invalid masks sector={sector_mask:#06b}, local={local_sector_mask:#06b}"
    )]
    MemoryRequestMask {
        sector_id: i64,
        sector_mask: u8,
        local_sector_mask: u8,
    },
    #[error(
        "typed memory request has incoherent source/operation/reply contract: source={request_source:?}, operation={operation:?}, reply={reply:?}"
    )]
    MemoryRequestContract {
        request_source: TypedL2Source,
        operation: TypedMemoryOperation,
        reply: Option<TypedResponseKind>,
    },
    #[error(
        "typed memory request admission witness has subcore {subcore}, index {request_index}, count {request_count}"
    )]
    MemoryRequestWitness {
        subcore: u8,
        request_index: u16,
        request_count: u16,
    },
    #[error("typed batched memory requests change admission witness group identity")]
    MemoryRequestWitnessGroupMismatch,
    #[error("typed batched memory request witness index expected {expected}, found {found}")]
    MemoryRequestWitnessSequence { expected: u16, found: u16 },
    #[error(
        "typed memory request constant markers are incoherent: source={request_source:?}, operation={operation:?}, constant_load={constant_load}, constant_deep={constant_deep}"
    )]
    MemoryRequestConstantContract {
        request_source: TypedL2Source,
        operation: TypedMemoryOperation,
        constant_load: bool,
        constant_deep: bool,
    },
    #[error(
        "typed memory request sector mask {sector_mask:#06b} is incompatible with {response_kind:?}"
    )]
    MemoryRequestSectorReplyMismatch {
        response_kind: TypedResponseKind,
        sector_mask: u8,
    },
    #[error("typed L2 fill identity has invalid sector mask {sector_mask:#06b}")]
    FillIdentityMask { sector_mask: u8 },
    #[error("typed response identity uses response kind none")]
    ResponseIdentityKindNone,
    #[error("typed memory response event is missing its returning reply identity")]
    MemoryResponseReplyMissing,
    #[error(
        "typed memory response fill is paired with incompatible response kind {response_kind:?}"
    )]
    MemoryResponseFillReplyMismatch { response_kind: TypedResponseKind },
    #[error("typed memory write acknowledgement has unsupported operation {operation:?}")]
    MemoryWriteAckOperation { operation: TypedMemoryOperation },
    #[error(
        "typed memory write acknowledgement witness has subcore {subcore}, index {request_index}, count {request_count}"
    )]
    MemoryWriteAckWitness {
        subcore: u8,
        request_index: u16,
        request_count: u16,
    },
    #[error(transparent)]
    Sequence(#[from] EventSequenceError),
    #[error("pending sequence {sequence} overlaps the current event's reservation")]
    CurrentEventOverlap { sequence: u64 },
    #[error("pending event ({time},{sequence}) is at or before the current event")]
    PendingEventOrder { time: u64, sequence: u64 },
    #[error("typed event target {target} is outside {cores} core snapshots")]
    EventTargetOutOfRange { target: u32, cores: usize },
    #[error("typed core SM identity {found} is not after previous identity {previous}")]
    CoreIdentityOrder { previous: u32, found: u32 },
    #[error("typed power domain time does not match event time {event_time}")]
    PowerTimeMismatch { event_time: u64 },
    #[error(
        "typed response owner for SM {sm_id} is due at {due}, before boundary event time {event_time}"
    )]
    ResponseDueBeforeBoundary {
        sm_id: u32,
        due: u64,
        event_time: u64,
    },
    #[error(
        "typed response owner/queue join is inconsistent for SM {sm_id} warp {response_warp} instruction {instruction} sector {sector_index} kind {response_kind:?} due {due}"
    )]
    ResponseOwnerQueueJoin {
        sm_id: u32,
        response_warp: u64,
        instruction: u64,
        sector_index: u64,
        response_kind: TypedResponseKind,
        due: u64,
    },
    #[error(
        "typed L2 pending-fill/queue join is inconsistent for line {line_id} generation {generation} ready {ready}"
    )]
    L2FillQueueJoin {
        line_id: i64,
        generation: u64,
        ready: u64,
    },
    #[error(
        "typed pending L2 sector {line:?}/{generation}/{sector} is ready at {ready_at}, before boundary event time {event_time}"
    )]
    L2PendingReadyBeforeBoundary {
        line: CanonicalLineKey,
        generation: u64,
        sector: u8,
        ready_at: u64,
        event_time: u64,
    },
    #[error(transparent)]
    Queue(#[from] QueueStateError),
    #[error(transparent)]
    Core(#[from] CoreStateError),
    #[error(transparent)]
    L2(#[from] L2StateError),
    #[error(transparent)]
    Dram(#[from] DramStateError),
    #[error(transparent)]
    Power(#[from] PowerStateError),
    #[error("typed boundary completeness declaration disagrees with its contract and state")]
    CompletenessMismatch {
        declared: Completeness,
        derived: Completeness,
    },
}

