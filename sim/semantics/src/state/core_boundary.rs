//! Semantic submodule; report-only boundary state and serde contracts.

use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedCoreBoundaryState {
    pub sm_id: u32,
    pub cycle: u64,
    pub issued: u64,
    pub resident_warps: u32,
    pub resident_shared_memory_bytes: u64,
    pub unfinished_warps: u64,
    pub next_block: u64,
    pub hold: [Option<u64>; 4],
    pub subcore_warps: Vec<Vec<i32>>,
    pub outstanding_responses: u64,
    pub barrier_pipe_ready: u64,
    pub tensor_pipe_ready: [u64; 4],
    pub ldgsts_destination_pipe_ready: u64,
    pub warp_states: Vec<WarpBoundaryState>,
    pub block_states: Vec<BlockBoundaryState>,
    pub l1: Option<L1BoundaryState>,
    pub front_end: Option<InstructionFrontEndBoundaryState>,
    pub memory_pipeline: Option<MemoryPipelineBoundaryState>,
    pub scalar_pipeline: Option<ScalarPipelineBoundaryState>,
    /// Native maximum CTA-barrier release time across all admitted blocks,
    /// including retired ones. Independent of the issue clock and the memory
    /// pipeline's terminal acceptance: release can lie ahead of `cycle`.
    /// Explicit null denotes no released generation; the field is required.
    #[serde(deserialize_with = "Option::<u64>::deserialize")]
    pub local_completion_time: Option<u64>,
    /// Timing state that affects future issue but is not yet serialized
    /// field-by-field. Exact contracts reify whenever this list is nonempty.
    pub unprojected_timing_state: Vec<String>,
    pub async_copy: Vec<Option<AsyncCopyBoundaryState>>,
    pub response_owners: Vec<L2RequestOwner>,
}

impl TypedCoreBoundaryState {
    pub fn validate(&self) -> Result<(), CoreStateError> {
        if self.sm_id == u32::MAX {
            return Err(CoreStateError::InvalidSmIdentity);
        }
        if self.subcore_warps.len() != 4 {
            return Err(CoreStateError::SubcoreCount {
                found: self.subcore_warps.len(),
            });
        }
        let mut previous_unprojected = None;
        for reason in &self.unprojected_timing_state {
            if reason.is_empty() || previous_unprojected.is_some_and(|previous| reason <= previous)
            {
                return Err(CoreStateError::UnprojectedTimingState);
            }
            previous_unprojected = Some(reason);
        }
        if self.next_block != u64::try_from(self.block_states.len()).unwrap_or(u64::MAX) {
            return Err(CoreStateError::NextBlockMismatch {
                next_block: self.next_block,
                block_states: self.block_states.len(),
            });
        }
        let mut roster = BTreeSet::new();
        for warps in &self.subcore_warps {
            for &warp_slot in warps {
                if warp_slot < 0 {
                    return Err(CoreStateError::NegativeWarpSlot { warp_slot });
                }
                if !roster.insert(warp_slot) {
                    return Err(CoreStateError::DuplicateResidentWarp { warp_slot });
                }
            }
        }
        for (subcore, warp_slot) in self
            .hold
            .iter()
            .enumerate()
            .filter_map(|(subcore, slot)| slot.as_ref().map(|&warp_slot| (subcore, warp_slot)))
        {
            let warp_slot = i32::try_from(warp_slot)
                .map_err(|_| CoreStateError::HoldWarpMissing { warp_slot })?;
            if !roster.contains(&warp_slot) {
                return Err(CoreStateError::HoldWarpMissing {
                    warp_slot: warp_slot as u64,
                });
            }
            if !self.subcore_warps[subcore].contains(&warp_slot) {
                return Err(CoreStateError::HoldWarpWrongSubcore { subcore, warp_slot });
            }
        }
        let mut known_warps = BTreeSet::new();
        for warp in &self.warp_states {
            if warp.warp_slot < 0 {
                return Err(CoreStateError::NegativeWarpSlot {
                    warp_slot: warp.warp_slot,
                });
            }
            validate_warp_instruction_buffer(warp, self.front_end.as_ref())?;
            if usize::try_from(warp.block)
                .ok()
                .is_none_or(|block| block >= self.block_states.len())
            {
                return Err(CoreStateError::UnknownWarpBlock { block: warp.block });
            }
            if !known_warps.insert((warp.block, warp.warp_slot)) {
                return Err(CoreStateError::DuplicateWarpState {
                    block: warp.block,
                    warp_slot: warp.warp_slot,
                });
            }
        }
        let issued_from_warps = self.warp_states.iter().try_fold(0_u64, |sum, warp| {
            sum.checked_add(warp.next_instruction)
                .ok_or(CoreStateError::CounterOverflow)
        })?;
        if issued_from_warps != self.issued {
            return Err(CoreStateError::IssuedCount {
                declared: self.issued,
                warps: issued_from_warps,
            });
        }
        let mut block_warps = BTreeSet::new();
        let mut active_block_slots = BTreeMap::new();
        let mut unfinished_by_block = BTreeMap::new();
        let mut unfinished_state_count = 0_u64;
        let mut resident_warp_reservation = 0_u64;
        let mut expected_first_warp = 0_u64;
        for warp in &self.warp_states {
            if !warp.done {
                unfinished_state_count = unfinished_state_count
                    .checked_add(1)
                    .ok_or(CoreStateError::CounterOverflow)?;
                let count = unfinished_by_block.entry(warp.block).or_insert(0_u64);
                *count = count
                    .checked_add(1)
                    .ok_or(CoreStateError::CounterOverflow)?;
                if !roster.contains(&warp.warp_slot) {
                    return Err(CoreStateError::UnfinishedWarpNotResident {
                        warp_slot: warp.warp_slot,
                    });
                }
            }
        }
        for (block_index, block) in self.block_states.iter().enumerate() {
            if block.unfinished != 0 {
                resident_warp_reservation = resident_warp_reservation
                    .checked_add(u64::from(block.warp_count))
                    .ok_or(CoreStateError::CounterOverflow)?;
            }
            if block.warp_count > u32::try_from(block.admitted_warps).unwrap_or(u32::MAX) {
                return Err(CoreStateError::BlockWarpCount {
                    block: block_index,
                    declared: block.warp_count,
                    admitted: block.admitted_warps,
                });
            }
            if block.unfinished > block.admitted_warps {
                return Err(CoreStateError::BlockUnfinishedCount { block: block_index });
            }
            if block.unfinished > u64::from(block.warp_count) {
                return Err(CoreStateError::BlockUnfinishedExceedsWarpCount {
                    block: block_index,
                    unfinished: block.unfinished,
                    warp_count: block.warp_count,
                });
            }
            if block.waiting > block.unfinished {
                return Err(CoreStateError::BlockWaitingCount { block: block_index });
            }
            if block.warp_slots.len() != usize::try_from(block.admitted_warps).unwrap_or(usize::MAX)
            {
                return Err(CoreStateError::BlockWarpRoster {
                    block: block_index,
                    declared: block.admitted_warps,
                    roster: block.warp_slots.len(),
                });
            }
            for &warp_slot in &block.warp_slots {
                if warp_slot < 0
                    || !known_warps
                        .contains(&(u64::try_from(block_index).unwrap_or(u64::MAX), warp_slot))
                {
                    return Err(CoreStateError::BlockWarpMissing { warp_slot });
                }
                if !block_warps.insert((block_index, warp_slot)) {
                    return Err(CoreStateError::DuplicateBlockWarp { warp_slot });
                }
            }
            if block.first_warp != expected_first_warp {
                return Err(CoreStateError::BlockFirstWarpOrder {
                    block: block_index,
                    expected: expected_first_warp,
                    found: block.first_warp,
                });
            }
            let range_end = block.first_warp.checked_add(block.admitted_warps).ok_or(
                CoreStateError::BlockWarpRange {
                    block: block_index,
                    first_warp: block.first_warp,
                    admitted_warps: block.admitted_warps,
                    warp_states: self.warp_states.len(),
                },
            )?;
            let warp_state_count = u64::try_from(self.warp_states.len()).unwrap_or(u64::MAX);
            if range_end > warp_state_count {
                return Err(CoreStateError::BlockWarpRange {
                    block: block_index,
                    first_warp: block.first_warp,
                    admitted_warps: block.admitted_warps,
                    warp_states: self.warp_states.len(),
                });
            }
            let range_start =
                usize::try_from(block.first_warp).map_err(|_| CoreStateError::BlockWarpRange {
                    block: block_index,
                    first_warp: block.first_warp,
                    admitted_warps: block.admitted_warps,
                    warp_states: self.warp_states.len(),
                })?;
            let range_end =
                usize::try_from(range_end).map_err(|_| CoreStateError::BlockWarpRange {
                    block: block_index,
                    first_warp: block.first_warp,
                    admitted_warps: block.admitted_warps,
                    warp_states: self.warp_states.len(),
                })?;
            let declared_slots = block.warp_slots.iter().copied().collect::<BTreeSet<_>>();
            let range_slots = self.warp_states[range_start..range_end]
                .iter()
                .map(|warp| warp.warp_slot)
                .collect::<BTreeSet<_>>();
            if declared_slots != range_slots {
                return Err(CoreStateError::BlockWarpRangeRoster { block: block_index });
            }
            for warp in &self.warp_states[range_start..range_end] {
                if warp.block != u64::try_from(block_index).unwrap_or(u64::MAX) {
                    return Err(CoreStateError::BlockWarpRangeIdentity {
                        block: block_index,
                        found: warp.block,
                    });
                }
            }
            validate_block_barrier_state(
                block_index,
                block,
                &self.warp_states[range_start..range_end],
                self.cycle,
            )?;
            expected_first_warp = range_end as u64;
            if block.unfinished != 0 {
                for &warp_slot in &block.warp_slots {
                    if let Some(&previous_block) = active_block_slots.get(&warp_slot) {
                        return Err(CoreStateError::ActiveBlockWarpSlotCollision {
                            warp_slot,
                            previous_block,
                            block: block_index,
                        });
                    }
                    active_block_slots.insert(warp_slot, block_index);
                }
            }
            let unfinished_states = unfinished_by_block
                .get(&u64::try_from(block_index).unwrap_or(u64::MAX))
                .copied()
                .unwrap_or(0);
            if unfinished_states != block.unfinished {
                return Err(CoreStateError::BlockWarpStateCount {
                    block: block_index,
                    declared: block.unfinished,
                    states: unfinished_states,
                });
            }
        }
        for (subcore, warps) in self.subcore_warps.iter().enumerate() {
            for &warp_slot in warps {
                let expected_subcore = usize::try_from(warp_slot.rem_euclid(4))
                    .expect("SM86 subcore modulo is within usize");
                if expected_subcore != subcore {
                    return Err(CoreStateError::ResidentWarpWrongSubcore {
                        warp_slot,
                        subcore,
                        expected_subcore,
                    });
                }
            }
        }
        let mut live_arrivals = BTreeSet::new();
        for warp in self.warp_states.iter().filter(|warp| !warp.done) {
            if !live_arrivals.insert(warp.arrival) {
                return Err(CoreStateError::DuplicateResidentWarpArrival {
                    arrival: warp.arrival,
                });
            }
        }
        let warp_state_count = u64::try_from(self.warp_states.len()).unwrap_or(u64::MAX);
        if expected_first_warp != warp_state_count {
            return Err(CoreStateError::WarpArenaTail {
                expected: expected_first_warp,
                found: warp_state_count,
            });
        }
        if u64::from(self.resident_warps) != resident_warp_reservation {
            return Err(CoreStateError::ResidentWarpReservationCount {
                declared: self.resident_warps,
                reservation: resident_warp_reservation,
            });
        }
        let unfinished_sum = self.block_states.iter().try_fold(0_u64, |sum, block| {
            sum.checked_add(block.unfinished)
                .ok_or(CoreStateError::CounterOverflow)
        })?;
        if unfinished_sum != self.unfinished_warps {
            return Err(CoreStateError::UnfinishedWarpCount {
                declared: self.unfinished_warps,
                blocks: unfinished_sum,
            });
        }
        if unfinished_state_count != self.unfinished_warps {
            return Err(CoreStateError::WarpStateCount {
                declared: self.unfinished_warps,
                states: unfinished_state_count,
            });
        }
        let shared_memory_sum = self
            .block_states
            .iter()
            .filter(|block| block.unfinished != 0)
            .try_fold(0_u64, |sum, block| {
                sum.checked_add(block.shared_memory_bytes)
                    .ok_or(CoreStateError::CounterOverflow)
            })?;
        if shared_memory_sum != self.resident_shared_memory_bytes {
            return Err(CoreStateError::SharedMemoryCount {
                declared: self.resident_shared_memory_bytes,
                blocks: shared_memory_sum,
            });
        }
        for &warp_slot in &roster {
            if !known_warps.iter().any(|(_, slot)| *slot == warp_slot) {
                return Err(CoreStateError::ResidentWarpMissing { warp_slot });
            }
        }
        if u64::try_from(roster.len()).unwrap_or(u64::MAX) != unfinished_state_count {
            return Err(CoreStateError::CurrentRosterCount {
                roster: roster.len(),
                unfinished_states: unfinished_state_count,
            });
        }
        if self.outstanding_responses
            != u64::try_from(self.response_owners.len()).unwrap_or(u64::MAX)
        {
            return Err(CoreStateError::ResponseCount {
                declared: self.outstanding_responses,
                owners: self.response_owners.len(),
            });
        }
        let mut previous = None;
        let mut seen = BTreeSet::new();
        for owner in &self.response_owners {
            if owner.request.sm_id != self.sm_id {
                return Err(CoreStateError::ResponseOwnerSm {
                    owner: owner.request.sm_id,
                });
            }
            let response_warp = usize::try_from(owner.request.response_warp).map_err(|_| {
                CoreStateError::ResponseOwnerWarpOutOfRange {
                    response_warp: owner.request.response_warp,
                    warp_states: self.warp_states.len(),
                }
            })?;
            let referenced_warp = self.warp_states.get(response_warp).ok_or(
                CoreStateError::ResponseOwnerWarpOutOfRange {
                    response_warp: owner.request.response_warp,
                    warp_states: self.warp_states.len(),
                },
            )?;
            if referenced_warp.warp_slot != owner.request.warp_slot {
                return Err(CoreStateError::ResponseOwnerWarpSlotMismatch {
                    response_warp: owner.request.response_warp,
                    owner_warp_slot: owner.request.warp_slot,
                    state_warp_slot: referenced_warp.warp_slot,
                });
            }
            if owner.request.instruction >= referenced_warp.next_instruction {
                return Err(CoreStateError::ResponseOwnerInstructionAfterCursor {
                    response_warp: owner.request.response_warp,
                    instruction: owner.request.instruction,
                    next_instruction: referenced_warp.next_instruction,
                });
            }
            if !matches!(owner.state, L2RequestState::Outstanding) {
                return Err(CoreStateError::ResponseOwnerState);
            }
            if owner.request.warp_slot < 0 {
                return Err(CoreStateError::NegativeResponseOwnerWarpSlot {
                    warp_slot: owner.request.warp_slot,
                });
            }
            let key = (
                owner.request.sm_id,
                owner.request.response_warp,
                owner.request.warp_slot,
                owner.request.instruction,
                owner.request.sector_index,
                owner.request.response_kind,
            );
            if !seen.insert(key) {
                return Err(CoreStateError::DuplicateResponseOwner);
            }
            if let Some(previous) = previous
                && key <= previous
            {
                return Err(CoreStateError::UnsortedResponseOwners);
            }
            previous = Some(key);
        }
        if let Some(l1) = &self.l1 {
            l1.validate_at_cycle(self.cycle)?
        }
        if let Some(memory) = &self.memory_pipeline {
            validate_memory_pipeline(memory, self.cycle)?;
        }
        if let Some(scalar) = &self.scalar_pipeline {
            validate_scalar_pipeline(scalar, self.cycle)?;
        }
        if let Some(front_end) = &self.front_end {
            front_end.validate()?
        }
        if !self.async_copy.is_empty() && self.async_copy.len() != self.warp_states.len() {
            return Err(CoreStateError::AsyncCopySlots);
        }
        for state in self.async_copy.iter().flatten() {
            state.validate_at_cycle(self.cycle)?
        }
        let has_released_barrier = self
            .block_states
            .iter()
            .any(|block| block.barrier_generation != 0);
        if has_released_barrier != self.local_completion_time.is_some()
            || self.local_completion_time == Some(0)
        {
            return Err(CoreStateError::LocalCompletionHistory);
        }
        Ok(())
    }
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
            pub(super) type ResponseQueueKey = (u32, u64, u64, u64, TypedResponseKind, u64);
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

            pub(super) type FillQueueKey = (i64, u64, u64);
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
