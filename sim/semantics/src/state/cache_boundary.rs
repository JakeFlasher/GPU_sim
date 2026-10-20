//! Semantic submodule; report-only boundary state and serde contracts.

use super::*;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CanonicalLineKey {
    VirtualAbsolute {
        line_id: i64,
    },
    AllocationRelative {
        allocation_id: u32,
        line_offset: u64,
    },
    Physical {
        line_id: u64,
    },
    Unknown {
        line_id: Option<i64>,
    },
}

impl CanonicalLineKey {
    pub fn domain(&self) -> AddressDomain {
        match self {
            Self::VirtualAbsolute { .. } => AddressDomain::VirtualAbsolute,
            Self::AllocationRelative { .. } => AddressDomain::AllocationRelative,
            Self::Physical { .. } => AddressDomain::Physical,
            Self::Unknown { .. } => AddressDomain::Unknown,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L2BoundaryState {
    pub schema_version: u32,
    pub architecture: String,
    pub geometry: L2GeometryState,
    pub address_domain: AddressDomain,
    pub clock: i64,
    pub bip_counter: u64,
    pub access_discipline: Option<L2AccessDisciplineState>,
    pub counters: L2CounterState,
    pub observer_lookup_hits: u64,
    pub observer_lookup_requests: u64,
    pub next_generation: Option<u64>,
    pub sets: Vec<L2SetState>,
    pub allocations: Vec<L2AllocationState>,
    pub request_owners: Vec<L2RequestOwner>,
    pub ownership_complete: bool,
    pub unknown_state: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L1BoundaryState {
    pub schema_version: u32,
    pub architecture: String,
    pub line_capacity: u64,
    pub mshr_count: Option<u32>,
    pub hit_latency: u64,
    pub miss_latency: u64,
    pub line_order: Vec<(u64, i64)>,
    pub stamp_of: Vec<(i64, u64)>,
    pub sectors: Vec<(i64, u8)>,
    pub line_fills: Vec<L1LineFillState>,
    pub sector_fills: Vec<L1SectorFillState>,
    pub line_generations: Vec<(i64, u32)>,
    pub next_stamp: Option<u64>,
    pub fill_sequence: Option<u64>,
    pub active_lookup_batch: Option<(u64, bool)>,
    pub min_fill_due: Option<u64>,
    pub fill_timing: Option<String>,
    pub counters: L1CounterState,
    pub write_observer_present: bool,
}

impl L1BoundaryState {
    pub fn validate(&self) -> Result<(), L1StateError> {
        if self.schema_version != TYPED_STATE_SCHEMA_VERSION {
            return Err(L1StateError::SchemaVersion);
        }
        if self.architecture.is_empty() {
            return Err(L1StateError::ArchitectureIdentity);
        }
        if self.line_capacity == 0 || self.mshr_count == Some(0) {
            return Err(L1StateError::InvalidGeometry);
        }
        if self.hit_latency >= self.miss_latency {
            return Err(L1StateError::InvalidLatencyOrder);
        }
        // The timing discipline is a latched executable mode, not merely a
        // property of currently live fills. Validate it even for an empty
        // cache; otherwise a malformed snapshot can report a complete state
        // and fail only when the next access consults the latch.
        if !matches!(
            self.fill_timing.as_deref(),
            None | Some("constant") | Some("event")
        ) {
            return Err(L1StateError::FillTimingMismatch);
        }
        if self.line_order.len() > usize::try_from(self.line_capacity).unwrap_or(usize::MAX) {
            return Err(L1StateError::LineCapacityExceeded);
        }
        // A late sector child shares the parent line allocation's MSHR only
        // when both identities carry the same generation. Keep the projected
        // occupancy calculation aligned with the executable L1 model even
        // though the base MSHR value is not a certification gate: access-local
        // LDGSTS floors may legally exceed that configured baseline.
        let _mshr_in_use = self
            .line_fills
            .len()
            .checked_add(
                self.sector_fills
                    .iter()
                    .filter(|fill| {
                        !self.line_fills.iter().any(|parent| {
                            parent.line == fill.line && parent.generation == fill.generation
                        })
                    })
                    .count(),
            )
            .ok_or(L1StateError::MshrCountExceeded)?;
        // `mshr_count` is the card's base limit. Native SM86 cacheable
        // LDGSTS accesses may use an access-local admission floor without
        // changing this serialized configuration field; the typed projection
        // therefore checks shape and occupancy but does not reject that legal
        // per-access policy.
        let miss_composition = self
            .counters
            .line_alloc_misses
            .checked_add(self.counters.sector_resident_misses)
            .and_then(|value| value.checked_add(self.counters.merges))
            .ok_or(L1StateError::CounterOverflow)?;
        if self.counters.misses != miss_composition {
            return Err(L1StateError::MissCompositionMismatch {
                misses: self.counters.misses,
                components: miss_composition,
            });
        }
        if self.counters.pending_lookup_hits > self.counters.merges {
            return Err(L1StateError::PendingLookupHitsExceedMerges);
        }
        if self.counters.bypass_atom_sectors > self.counters.bypass_sectors {
            return Err(L1StateError::BypassAtomsExceedBypass);
        }
        if self.counters.store_hits > self.counters.store_sectors {
            return Err(L1StateError::StoreHitsExceedStores);
        }
        if self.counters.reduction_hits > self.counters.reduction_sectors {
            return Err(L1StateError::ReductionHitsExceedReductions);
        }
        let mut stamps = BTreeMap::new();
        let mut previous_line_order = None;
        for &(stamp, line) in &self.line_order {
            if let Some(previous) = previous_line_order
                && (stamp, line) <= previous
            {
                return Err(L1StateError::UnsortedLineOrder);
            }
            if stamps.insert(stamp, line).is_some() {
                return Err(L1StateError::DuplicateStamp);
            }
            previous_line_order = Some((stamp, line));
        }
        let mut previous_stamp_of = None;
        for &(line, stamp) in &self.stamp_of {
            if let Some(previous) = previous_stamp_of
                && line <= previous
            {
                return Err(L1StateError::UnsortedStampOf);
            }
            if stamps.get(&stamp) != Some(&line) {
                return Err(L1StateError::StampIdentityMismatch);
            }
            previous_stamp_of = Some(line);
        }
        if self.line_order.len() != self.stamp_of.len() {
            return Err(L1StateError::StampCardinalityMismatch);
        }
        let resident_lines = self
            .stamp_of
            .iter()
            .map(|&(line, _)| line)
            .collect::<BTreeSet<_>>();
        if let Some(fill) = self
            .line_fills
            .iter()
            .find(|fill| resident_lines.contains(&fill.line))
        {
            return Err(L1StateError::ResidentLineFillConflict { line: fill.line });
        }
        let mut previous_sector = None;
        for &(line, mask) in &self.sectors {
            if let Some(previous) = previous_sector
                && line <= previous
            {
                return Err(L1StateError::UnsortedSectors);
            }
            if mask == 0 || mask & !0x0f != 0 {
                return Err(L1StateError::InvalidSectorMask);
            }
            previous_sector = Some(line);
        }
        let mut previous_fill = None;
        let mut seen_line_fills = BTreeSet::new();
        for fill in &self.line_fills {
            if self
                .line_generations
                .iter()
                .find(|&&(line, _)| line == fill.line)
                .map(|&(_, generation)| generation)
                != Some(fill.generation)
            {
                return Err(L1StateError::LineFillGenerationMismatch);
            }
            if u64::from(fill.generation) != fill.sequence {
                return Err(L1StateError::LineFillSequenceMismatch);
            }
            let key = (fill.line, fill.generation);
            if !seen_line_fills.insert(key) {
                return Err(L1StateError::DuplicateLineFill);
            }
            if let Some(previous) = previous_fill
                && (fill.line, fill.generation, fill.sequence) <= previous
            {
                return Err(L1StateError::UnsortedLineFills);
            }
            previous_fill = Some((fill.line, fill.generation, fill.sequence));
        }
        let mut previous_sector_fill = None;
        let mut seen_sector_fills = BTreeSet::new();
        let mut fill_sequences = BTreeSet::new();
        for fill in &self.sector_fills {
            if fill.sector >= 4 {
                return Err(L1StateError::InvalidFillSector);
            }
            let key = (fill.line, fill.sector, fill.generation);
            if !seen_sector_fills.insert(key) {
                return Err(L1StateError::DuplicateSectorFill);
            }
            if !fill_sequences.insert(fill.sequence) {
                return Err(L1StateError::DuplicateFillSequence);
            }
            if let Some(previous) = previous_sector_fill
                && (fill.line, fill.sector, fill.generation, fill.sequence) <= previous
            {
                return Err(L1StateError::UnsortedSectorFills);
            }
            previous_sector_fill = Some((fill.line, fill.sector, fill.generation, fill.sequence));
        }
        for fill in &self.line_fills {
            if !fill_sequences.insert(fill.sequence) {
                return Err(L1StateError::DuplicateFillSequence);
            }
        }
        let mut previous_generation = None;
        let mut seen_generations = BTreeSet::new();
        let mut seen_generation_values = BTreeSet::new();
        for &(line, _) in &self.line_generations {
            if !seen_generations.insert(line) {
                return Err(L1StateError::DuplicateGeneration);
            }
            if let Some(previous) = previous_generation
                && line <= previous
            {
                return Err(L1StateError::UnsortedGenerations);
            }
            previous_generation = Some(line);
        }
        for &(_, generation) in &self.line_generations {
            if !seen_generation_values.insert(generation) {
                return Err(L1StateError::DuplicateGenerationValue);
            }
        }
        let lines_with_live_identity = self
            .stamp_of
            .iter()
            .map(|&(line, _)| line)
            .chain(self.line_fills.iter().map(|fill| fill.line))
            .collect::<BTreeSet<_>>();
        for &(line, _) in &self.sectors {
            if !lines_with_live_identity.contains(&line) {
                return Err(L1StateError::SectorLineWithoutAllocation { line });
            }
        }
        for &(line, generation) in &self.line_generations {
            if !lines_with_live_identity.contains(&line) {
                return Err(L1StateError::GenerationWithoutAllocation { line, generation });
            }
        }
        // The fill-timing latch is part of the executable cache identity even
        // after every fill has retired.  A resident line or generation proves
        // that a read path has already selected constant or event timing; an
        // omitted latch would let the next miss choose a mode arbitrarily and
        // therefore cannot certify a resumable boundary.  Store-only empty
        // caches remain valid with no latch because they do not allocate L1
        // lines or fills.
        let has_live_read_allocation = !self.stamp_of.is_empty()
            || !self.sectors.is_empty()
            || !self.line_generations.is_empty();
        if has_live_read_allocation && self.fill_timing.is_none() {
            return Err(L1StateError::FillTimingMissing);
        }
        if let Some(next_stamp) = self.next_stamp
            && self
                .line_order
                .iter()
                .any(|&(stamp, _)| stamp >= next_stamp)
        {
            return Err(L1StateError::NextStampMismatch);
        }
        if let Some(fill_sequence) = self.fill_sequence {
            let line_fill_ahead = self
                .line_fills
                .iter()
                .any(|fill| fill.sequence >= fill_sequence);
            let sector_fill_ahead = self
                .sector_fills
                .iter()
                .any(|fill| fill.sequence >= fill_sequence);
            if line_fill_ahead || sector_fill_ahead {
                return Err(L1StateError::FillSequenceMismatch);
            }
        }
        if let Some((first_fill_sequence, _)) = self.active_lookup_batch
            && first_fill_sequence > self.fill_sequence.unwrap_or(u64::MAX)
        {
            return Err(L1StateError::LookupBatchAheadOfFillSequence);
        }
        if !self.line_fills.is_empty() || !self.sector_fills.is_empty() {
            match self.fill_timing.as_deref() {
                Some("constant")
                    if self
                        .line_fills
                        .iter()
                        .any(|fill| !matches!(fill.due, L1FillDueState::Timed { .. }))
                        || self
                            .sector_fills
                            .iter()
                            .any(|fill| !matches!(fill.due, L1FillDueState::Timed { .. })) =>
                {
                    return Err(L1StateError::FillTimingMismatch);
                }
                Some("event")
                    if self
                        .line_fills
                        .iter()
                        .any(|fill| !matches!(fill.due, L1FillDueState::Untimed))
                        || self
                            .sector_fills
                            .iter()
                            .any(|fill| !matches!(fill.due, L1FillDueState::Untimed)) =>
                {
                    return Err(L1StateError::FillTimingMismatch);
                }
                Some("constant") | Some("event") => {}
                _ => return Err(L1StateError::FillTimingMismatch),
            }
        }
        let timed_min = self
            .line_fills
            .iter()
            .filter_map(|fill| match fill.due {
                L1FillDueState::Timed { cycle } => Some(cycle),
                L1FillDueState::Untimed => None,
            })
            .chain(self.sector_fills.iter().filter_map(|fill| match fill.due {
                L1FillDueState::Timed { cycle } => Some(cycle),
                L1FillDueState::Untimed => None,
            }))
            .min();
        if self.min_fill_due != timed_min {
            return Err(L1StateError::MinimumFillDueMismatch);
        }
        Ok(())
    }

    /// Validate a projected L1 state at the owning SM's current cycle.
    ///
    /// `validate` remains cycle-independent for standalone cache snapshots,
    /// while a typed core boundary must reject a timed MSHR fill that should
    /// already have been retired before that core clock. Event-mode fills are
    /// intentionally untimed and are ordered by the external event queue.
    pub fn validate_at_cycle(&self, cycle: u64) -> Result<(), L1StateError> {
        self.validate()?;
        let due = self
            .line_fills
            .iter()
            .filter_map(|fill| match fill.due {
                L1FillDueState::Timed { cycle } => Some(cycle),
                L1FillDueState::Untimed => None,
            })
            .chain(self.sector_fills.iter().filter_map(|fill| match fill.due {
                L1FillDueState::Timed { cycle } => Some(cycle),
                L1FillDueState::Untimed => None,
            }))
            .find(|&due| due < cycle);
        if let Some(due) = due {
            return Err(L1StateError::FillDueInPast { due, cycle });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AsyncCopyBoundaryState {
    pub open: Option<AsyncGroupBoundaryState>,
    pub committed: Vec<AsyncGroupBoundaryState>,
    pub bypass_observation_count: u64,
    /// Next per-warp group sequence; `None` represents exhausted u64 space.
    pub next_sequence: Option<u64>,
    /// `false` marks a diagnostic projection that does not retain the
    /// response identity/membership details required for exact replay.
    pub bypass_observations_complete: bool,
    pub bypass_observations: Vec<AsyncCopyBypassObservationState>,
}

impl AsyncCopyBoundaryState {
    pub fn validate(&self) -> Result<(), AsyncCopyStateError> {
        let mut previous = None;
        let mut previous_commit = None;
        for group in &self.committed {
            if let Some(previous) = previous
                && group.sequence <= previous
            {
                return Err(AsyncCopyStateError::UnsortedGroups);
            }
            if let Some(previous_commit) = previous_commit
                && group.committed_at < previous_commit
            {
                return Err(AsyncCopyStateError::CommittedTimeOrder);
            }
            validate_async_group(group)?;
            previous = Some(group.sequence);
            previous_commit = Some(group.committed_at);
        }
        if let Some(group) = &self.open {
            validate_async_group(group)?;
            if previous.is_some_and(|sequence| group.sequence <= sequence) {
                return Err(AsyncCopyStateError::OpenGroupOrder);
            }
        }
        let maximum_sequence = self
            .committed
            .iter()
            .map(|group| group.sequence)
            .chain(self.open.iter().map(|group| group.sequence))
            .max();
        if let (Some(next), Some(maximum)) = (self.next_sequence, maximum_sequence)
            && next <= maximum
        {
            return Err(AsyncCopyStateError::NextSequenceMismatch);
        }
        let captured = u64::try_from(self.bypass_observations.len()).unwrap_or(u64::MAX);
        if self.bypass_observations_complete {
            if self.bypass_observation_count != captured {
                return Err(AsyncCopyStateError::BypassObservationCountMismatch {
                    declared: self.bypass_observation_count,
                    captured,
                });
            }
        } else if captured > self.bypass_observation_count {
            return Err(AsyncCopyStateError::BypassObservationCountMismatch {
                declared: self.bypass_observation_count,
                captured,
            });
        }
        if self.bypass_observations_complete
            && let Some(observation) = self.bypass_observations.iter().find(|observation| {
                observation.source_cohorts == 0
                    || observation.sector_membership_masks.is_empty()
                    || observation.cohorts.is_empty()
            })
        {
            return Err(AsyncCopyStateError::BypassObservationDetailMissing {
                group: observation.group,
                instruction: observation.instruction,
            });
        }
        // A partial projection may omit records, but any records it does
        // carry still participate in the executable identity and aggregate
        // contracts. Validate those records even when the completeness flag
        // remains false; otherwise malformed captured observations can hide
        // behind the diagnostic incomplete status.
        self.validate_bypass_observations()?;
        Ok(())
    }

    pub fn validate_at_cycle(&self, cycle: u64) -> Result<(), AsyncCopyStateError> {
        self.validate()?;
        if let Some(group) = self
            .committed
            .iter()
            .find(|group| group.committed_at > cycle)
        {
            return Err(AsyncCopyStateError::CommittedTimeAfterCoreCycle {
                committed_at: group.committed_at,
                cycle,
            });
        }
        if let Some(group) = self.open.as_ref()
            && group.committed_at > cycle
        {
            return Err(AsyncCopyStateError::CommittedTimeAfterCoreCycle {
                committed_at: group.committed_at,
                cycle,
            });
        }
        Ok(())
    }

    pub(super) fn validate_bypass_observations(&self) -> Result<(), AsyncCopyStateError> {
        let mut previous = None;
        for observation in &self.bypass_observations {
            let Some(group) = self
                .committed
                .iter()
                .find(|group| group.sequence == observation.group)
                .or_else(|| {
                    self.open
                        .as_ref()
                        .filter(|group| group.sequence == observation.group)
                })
            else {
                return Err(AsyncCopyStateError::BypassObservationGroupMissing {
                    group: observation.group,
                });
            };
            if u64::from(observation.pending_sources) > u64::from(group.pending_sources) {
                return Err(AsyncCopyStateError::BypassObservationGroupBound {
                    group: observation.group,
                    instruction: observation.instruction,
                    field: "pending_sources",
                    group_value: u64::from(group.pending_sources),
                    observation_value: u64::from(observation.pending_sources),
                });
            }
            if observation.source_completion_time > group.source_completion_time {
                return Err(AsyncCopyStateError::BypassObservationGroupBound {
                    group: observation.group,
                    instruction: observation.instruction,
                    field: "source_completion_time",
                    group_value: group.source_completion_time,
                    observation_value: observation.source_completion_time,
                });
            }
            if observation.delivery_work > group.delivery_work {
                return Err(AsyncCopyStateError::BypassObservationGroupBound {
                    group: observation.group,
                    instruction: observation.instruction,
                    field: "delivery_work",
                    group_value: group.delivery_work,
                    observation_value: observation.delivery_work,
                });
            }
            let key = (observation.group, observation.instruction);
            if let Some(previous) = previous
                && key <= previous
            {
                return Err(AsyncCopyStateError::BypassObservationOrder {
                    group: observation.group,
                    instruction: observation.instruction,
                });
            }
            previous = Some(key);
            if observation.delivery_work == 0 {
                return Err(AsyncCopyStateError::BypassObservationZeroWork {
                    group: observation.group,
                    instruction: observation.instruction,
                });
            }
            if !observation.cohorts.is_empty()
                && observation.source_cohorts
                    != u64::try_from(observation.cohorts.len()).unwrap_or(u64::MAX)
            {
                return Err(AsyncCopyStateError::BypassObservationAggregateMismatch {
                    group: observation.group,
                    instruction: observation.instruction,
                    field: "source_cohorts",
                    expected: u64::try_from(observation.cohorts.len()).unwrap_or(u64::MAX),
                    got: observation.source_cohorts,
                });
            }
            if !observation.sector_membership_masks.is_empty()
                && observation.pending_sources
                    > u32::try_from(observation.sector_membership_masks.len()).unwrap_or(u32::MAX)
            {
                return Err(AsyncCopyStateError::BypassObservationPendingSources {
                    group: observation.group,
                    instruction: observation.instruction,
                });
            }
            let sector_len =
                u64::try_from(observation.sector_membership_masks.len()).unwrap_or(u64::MAX);
            observation.sector_off.checked_add(sector_len).ok_or(
                AsyncCopyStateError::BypassObservationSectorOverflow {
                    group: observation.group,
                    instruction: observation.instruction,
                },
            )?;
            if observation.cohorts.is_empty() {
                if !observation.sector_membership_masks.is_empty() {
                    return Err(AsyncCopyStateError::BypassObservationAggregateMismatch {
                        group: observation.group,
                        instruction: observation.instruction,
                        field: "empty cohort membership",
                        expected: 0,
                        got: u64::try_from(observation.sector_membership_masks.len())
                            .unwrap_or(u64::MAX),
                    });
                }
                continue;
            }
            if observation.sector_membership_masks.is_empty() {
                return Err(AsyncCopyStateError::BypassObservationMembershipMissing {
                    group: observation.group,
                    instruction: observation.instruction,
                });
            }
            let allowed_mask = if observation.cohorts.len() == u32::BITS as usize {
                u32::MAX
            } else if observation.cohorts.len() < u32::BITS as usize {
                (1_u32 << observation.cohorts.len()) - 1
            } else {
                return Err(AsyncCopyStateError::BypassObservationCohortCount {
                    group: observation.group,
                    instruction: observation.instruction,
                });
            };
            let mut memberships = vec![0_u64; observation.cohorts.len()];
            for &mask in &observation.sector_membership_masks {
                if mask == 0 || mask & !allowed_mask != 0 {
                    return Err(AsyncCopyStateError::BypassObservationMembershipInvalid {
                        group: observation.group,
                        instruction: observation.instruction,
                    });
                }
                for (index, membership) in memberships.iter_mut().enumerate() {
                    if mask & (1_u32 << index) != 0 {
                        *membership = membership.checked_add(1).ok_or(
                            AsyncCopyStateError::BypassObservationMembershipOverflow {
                                group: observation.group,
                                instruction: observation.instruction,
                            },
                        )?;
                    }
                }
            }
            let mut payload_work = 0_u64;
            let mut destination_lines = 0_u64;
            let mut max_bank_wavefronts = 0_u64;
            let mut max_bypass_wavefronts = 0_u64;
            for (index, cohort) in observation.cohorts.iter().enumerate() {
                let membership = memberships[index];
                if u64::from(cohort.pending_sources) > membership {
                    return Err(AsyncCopyStateError::BypassObservationCohortPending {
                        group: observation.group,
                        instruction: observation.instruction,
                    });
                }
                let completed = membership - u64::from(cohort.pending_sources);
                match completed {
                    0 if cohort.first_source_response.is_some()
                        || cohort.last_source_response.is_some() =>
                    {
                        return Err(AsyncCopyStateError::BypassObservationResponseTimes {
                            group: observation.group,
                            instruction: observation.instruction,
                        });
                    }
                    0 => {}
                    _ if cohort.first_source_response.is_none()
                        || cohort.last_source_response.is_none() =>
                    {
                        return Err(AsyncCopyStateError::BypassObservationResponseTimes {
                            group: observation.group,
                            instruction: observation.instruction,
                        });
                    }
                    _ => {
                        let first = cohort.first_source_response.expect("checked above");
                        let last = cohort.last_source_response.expect("checked above");
                        if first > last || last > observation.source_completion_time {
                            return Err(AsyncCopyStateError::BypassObservationResponseTimes {
                                group: observation.group,
                                instruction: observation.instruction,
                            });
                        }
                    }
                }
                payload_work = payload_work.checked_add(cohort.payload_work).ok_or(
                    AsyncCopyStateError::BypassObservationAggregateOverflow {
                        group: observation.group,
                        instruction: observation.instruction,
                    },
                )?;
                destination_lines = destination_lines
                    .checked_add(cohort.destination_lines)
                    .ok_or(AsyncCopyStateError::BypassObservationAggregateOverflow {
                        group: observation.group,
                        instruction: observation.instruction,
                    })?;
                max_bank_wavefronts = max_bank_wavefronts.max(cohort.bank_wavefronts);
                max_bypass_wavefronts = max_bypass_wavefronts.max(cohort.bypass_wavefronts);
            }
            if payload_work != observation.delivery_work {
                return Err(AsyncCopyStateError::BypassObservationAggregateMismatch {
                    group: observation.group,
                    instruction: observation.instruction,
                    field: "delivery_work",
                    expected: payload_work,
                    got: observation.delivery_work,
                });
            }
            // The compact observation aggregates source-bearing cohorts and
            // executing source-ignored (zero-fill) destination cohorts.  The
            // sparse boundary sidecar intentionally carries only source
            // cohorts because zero-fill lanes have no source-response event.
            // A source-only sum may therefore be lower, but must never exceed
            // the aggregate topology published in the compact observation.
            if destination_lines > observation.destination_lines
                || max_bank_wavefronts > observation.max_bank_wavefronts
                || max_bypass_wavefronts > observation.max_bypass_wavefronts
            {
                return Err(AsyncCopyStateError::BypassObservationAggregateMismatch {
                    group: observation.group,
                    instruction: observation.instruction,
                    field: "source cohort topology exceeds aggregate",
                    expected: observation.destination_lines,
                    got: destination_lines,
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstructionCacheState {
    pub ways: u64,
    pub access_sequence: u64,
    pub sets: Vec<Vec<InstructionCacheEntryState>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstructionStreamState {
    pub lines: Vec<InstructionStreamLineState>,
    pub next_line: Option<(i32, u64)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstructionFrontEndBoundaryState {
    pub line_bytes: u32,
    pub l0_to_l1_cycles: u64,
    pub l1_to_l0_cycles: u64,
    pub l1_request_ports: u32,
    pub l1_reply_ports: u32,
    pub stream_buffer_lines: u32,
    pub stream_buffers_per_subcore: u32,
    pub prefetches_per_cycle: u32,
    pub instruction_buffer_entries: u32,
    pub fetch_decode_width: u32,
    pub fetch_to_issue_stages: u64,
    pub l0: Vec<InstructionCacheState>,
    pub l1: InstructionCacheState,
    pub streams: Vec<InstructionStreamState>,
    pub l1_request_ready: u64,
    pub l1_reply_ready: u64,
    pub l0_response_ready: Vec<u64>,
    pub prefetch_priority: u64,
    pub cold_fill_latency: u64,
    pub instructions_fetched: u64,
    pub demand_l0_hits: u64,
    pub demand_stream_buffer_hits: u64,
    pub demand_l0_misses: u64,
    pub demand_l1_hits: u64,
    pub demand_l1_misses: u64,
    pub prefetch_requests: u64,
    pub prefetch_l1_hits: u64,
    pub prefetch_l1_misses: u64,
    pub stream_restarts: u64,
    pub maximum_fetch_latency_cycles: u64,
}

impl InstructionCacheState {
    pub(super) fn validate(&self) -> Result<(), FrontEndStateError> {
        if self.ways == 0 || self.sets.is_empty() {
            return Err(FrontEndStateError::CacheShape);
        }
        let mut access_sequences = BTreeSet::new();
        for (set_index, set) in self.sets.iter().enumerate() {
            if set.len() > usize::try_from(self.ways).unwrap_or(usize::MAX) {
                return Err(FrontEndStateError::CacheCapacity);
            }
            let mut keys = BTreeSet::new();
            for entry in set {
                if entry.function_unique_id <= 0 {
                    return Err(FrontEndStateError::FunctionIdentity {
                        function_unique_id: entry.function_unique_id,
                    });
                }
                if !keys.insert((entry.function_unique_id, entry.line)) {
                    return Err(FrontEndStateError::DuplicateCacheEntry);
                }
                let function = u64::from(entry.function_unique_id.unsigned_abs());
                let expected_set = (entry.line ^ function.rotate_left(17))
                    % u64::try_from(self.sets.len()).unwrap_or(u64::MAX);
                if expected_set != u64::try_from(set_index).unwrap_or(u64::MAX) {
                    return Err(FrontEndStateError::CacheSetPlacement {
                        expected: expected_set,
                        found: set_index,
                    });
                }
                if !access_sequences.insert(entry.last_access) {
                    return Err(FrontEndStateError::DuplicateCacheAccessSequence);
                }
                if entry.last_access >= self.access_sequence {
                    return Err(FrontEndStateError::CacheAccessSequence);
                }
            }
        }
        Ok(())
    }
}

impl InstructionStreamState {
    pub(super) fn validate(&self) -> Result<(), FrontEndStateError> {
        for line in &self.lines {
            if line.function_unique_id <= 0 {
                return Err(FrontEndStateError::FunctionIdentity {
                    function_unique_id: line.function_unique_id,
                });
            }
        }
        if let Some((function_unique_id, _)) = self.next_line
            && function_unique_id <= 0
        {
            return Err(FrontEndStateError::FunctionIdentity { function_unique_id });
        }
        for window in self.lines.windows(2) {
            let previous = &window[0];
            let current = &window[1];
            let Some(expected_line) = previous.line.checked_add(1) else {
                return Err(FrontEndStateError::StreamOrder);
            };
            if current.function_unique_id != previous.function_unique_id
                || current.line != expected_line
            {
                return Err(FrontEndStateError::StreamOrder);
            }
            if current.ready_at < previous.ready_at {
                return Err(FrontEndStateError::StreamReadyOrder);
            }
        }
        if self.lines.iter().skip(1).any(|line| line.demanded) {
            return Err(FrontEndStateError::DemandedStreamTail);
        }
        if let Some(last) = self.lines.last() {
            let expected = last
                .line
                .checked_add(1)
                .map(|line| (last.function_unique_id, line));
            if self.next_line != expected {
                return Err(FrontEndStateError::StreamNextLine);
            }
        }
        Ok(())
    }
}

impl InstructionFrontEndBoundaryState {
    pub fn validate(&self) -> Result<(), FrontEndStateError> {
        if self.line_bytes == 0
            || self.l0_to_l1_cycles == 0
            || self.l1_to_l0_cycles == 0
            || self.l1_request_ports == 0
            || self.l1_reply_ports == 0
            || self.stream_buffer_lines == 0
            || self.stream_buffers_per_subcore == 0
            || self.prefetches_per_cycle == 0
            || self.instruction_buffer_entries == 0
            || self.fetch_decode_width == 0
            || self.cold_fill_latency == 0
            || self.fetch_to_issue_stages == 0
            || self.l0.len() != 4
            || self.streams.len() != 4
            || self.l0_response_ready.len() != 4
        {
            return Err(FrontEndStateError::Shape);
        }
        for (field, value) in [
            ("l1_request_ports", self.l1_request_ports),
            ("l1_reply_ports", self.l1_reply_ports),
            (
                "stream_buffers_per_subcore",
                self.stream_buffers_per_subcore,
            ),
            ("prefetches_per_cycle", self.prefetches_per_cycle),
            ("fetch_decode_width", self.fetch_decode_width),
        ] {
            if value != 1 {
                return Err(FrontEndStateError::UnsupportedWidth { field, value });
            }
        }
        if self.l1_reply_ready < self.l1_request_ready {
            return Err(FrontEndStateError::ReplyTimerBeforeRequest);
        }
        let l1_has_access = self.l1.access_sequence != 0;
        if l1_has_access != (self.l1_request_ready != 0 || self.l1_reply_ready != 0) {
            return Err(FrontEndStateError::L1TimerSequenceMismatch);
        }
        if self
            .l0
            .iter()
            .zip(&self.l0_response_ready)
            .any(|(cache, &ready)| (cache.access_sequence != 0) != (ready != 0))
        {
            return Err(FrontEndStateError::L0TimerSequenceMismatch);
        }
        if self
            .l0
            .iter()
            .skip(1)
            .any(|cache| cache.ways != self.l0[0].ways || cache.sets.len() != self.l0[0].sets.len())
        {
            return Err(FrontEndStateError::L0GeometryMismatch);
        }
        if self.prefetch_priority >= self.streams.len() as u64 {
            return Err(FrontEndStateError::PrefetchPriority);
        }
        self.l1.validate()?;
        for cache in &self.l0 {
            cache.validate()?;
        }
        if self
            .streams
            .iter()
            .any(|stream| stream.lines.len() > self.stream_buffer_lines as usize)
        {
            return Err(FrontEndStateError::StreamCapacity);
        }
        for stream in &self.streams {
            stream.validate()?;
        }
        let l0_access_sequence = self.l0.iter().try_fold(0_u64, |sum, cache| {
            sum.checked_add(cache.access_sequence)
                .ok_or(FrontEndStateError::CounterOverflow)
        })?;
        if l0_access_sequence != self.instructions_fetched {
            return Err(FrontEndStateError::CounterMismatch {
                field: "l0_access_sequence",
                expected: self.instructions_fetched,
                found: l0_access_sequence,
            });
        }
        let demand_categories = self
            .demand_l0_hits
            .checked_add(self.demand_stream_buffer_hits)
            .and_then(|value| value.checked_add(self.demand_l0_misses))
            .ok_or(FrontEndStateError::CounterOverflow)?;
        if demand_categories != self.instructions_fetched {
            return Err(FrontEndStateError::CounterMismatch {
                field: "demand_fetch_categories",
                expected: self.instructions_fetched,
                found: demand_categories,
            });
        }
        let l1_accesses = self
            .demand_l0_misses
            .checked_add(self.prefetch_requests)
            .ok_or(FrontEndStateError::CounterOverflow)?;
        if self.l1.access_sequence != l1_accesses {
            return Err(FrontEndStateError::CounterMismatch {
                field: "l1_access_sequence",
                expected: l1_accesses,
                found: self.l1.access_sequence,
            });
        }
        let demand_l1_outcomes = self
            .demand_l1_hits
            .checked_add(self.demand_l1_misses)
            .ok_or(FrontEndStateError::CounterOverflow)?;
        if demand_l1_outcomes != self.demand_l0_misses {
            return Err(FrontEndStateError::CounterMismatch {
                field: "demand_l1_outcomes",
                expected: self.demand_l0_misses,
                found: demand_l1_outcomes,
            });
        }
        let prefetch_l1_outcomes = self
            .prefetch_l1_hits
            .checked_add(self.prefetch_l1_misses)
            .ok_or(FrontEndStateError::CounterOverflow)?;
        if prefetch_l1_outcomes != self.prefetch_requests {
            return Err(FrontEndStateError::CounterMismatch {
                field: "prefetch_l1_outcomes",
                expected: self.prefetch_requests,
                found: prefetch_l1_outcomes,
            });
        }
        Ok(())
    }
}

impl L2BoundaryState {
    pub fn validate(&self) -> Result<(), L2StateError> {
        if self.schema_version != TYPED_STATE_SCHEMA_VERSION {
            return Err(L2StateError::SchemaVersion {
                found: self.schema_version,
                expected: TYPED_STATE_SCHEMA_VERSION,
            });
        }
        if self.architecture.is_empty() || self.geometry.architecture != self.architecture {
            return Err(L2StateError::ArchitectureMismatch);
        }
        if !self.ownership_complete && self.unknown_state.is_empty() {
            return Err(L2StateError::IncompleteWithoutUnknown);
        }
        if self
            .unknown_state
            .windows(2)
            .any(|window| window[1] <= window[0])
        {
            return Err(L2StateError::UnknownStateOrder);
        }
        if !matches!(self.geometry.replacement.as_str(), "lru" | "srrip" | "bip") {
            return Err(L2StateError::UnknownReplacement);
        }
        if self.geometry.capacity_bytes == 0
            || self.geometry.line_bytes == 0
            || self.geometry.sector_bytes == 0
            || self.geometry.sets_per_slice == 0
        {
            return Err(L2StateError::InvalidGeometry);
        }
        if self.geometry.num_slices == 0 || self.geometry.associativity == 0 {
            return Err(L2StateError::InvalidGeometry);
        }
        if self.clock < 0 {
            return Err(L2StateError::NegativeClock);
        }
        if self.address_domain == AddressDomain::VirtualAbsolute
            && !matches!(
                self.geometry.slice_hash.as_str(),
                "line_mod"
                    | "sector_mod"
                    | "xor_folded"
                    | "knuth_mult"
                    | "shift_mod"
                    | "bit_reverse"
            )
        {
            return Err(L2StateError::UnknownSliceHash);
        }
        if !self
            .geometry
            .line_bytes
            .is_multiple_of(self.geometry.sector_bytes)
        {
            return Err(L2StateError::InvalidGeometry);
        }
        let expected_capacity = u128::from(self.geometry.line_bytes)
            .checked_mul(u128::from(self.geometry.num_slices))
            .and_then(|value| value.checked_mul(u128::from(self.geometry.sets_per_slice)))
            .and_then(|value| value.checked_mul(u128::from(self.geometry.associativity)))
            .ok_or(L2StateError::InvalidGeometry)?;
        if expected_capacity != u128::from(self.geometry.capacity_bytes) {
            return Err(L2StateError::GeometryMismatch {
                expected: expected_capacity,
                found: self.geometry.capacity_bytes,
            });
        }
        if self.counters.lookup_hits > self.counters.lookup_requests {
            return Err(L2StateError::LookupHitsExceedRequests);
        }
        if self.observer_lookup_hits > self.observer_lookup_requests {
            return Err(L2StateError::ObserverLookupHitsExceedRequests);
        }
        let miss_components = self
            .counters
            .line_misses
            .checked_add(self.counters.sector_fill_misses)
            .ok_or(L2StateError::CounterOverflow)?;
        if self.counters.misses != miss_components {
            return Err(L2StateError::MissCompositionMismatch {
                misses: self.counters.misses,
                components: miss_components,
            });
        }
        if self.counters.write_tag_hits > self.counters.hits {
            return Err(L2StateError::WriteTagHitsExceedHits);
        }
        let mut previous = None;
        let mut seen_sets = BTreeSet::new();
        let mut live_allocations = BTreeSet::new();
        let mut live_generation_values = BTreeSet::new();
        let mut live_ways = BTreeMap::new();
        for set in &self.sets {
            if set.slice >= self.geometry.num_slices || set.set >= self.geometry.sets_per_slice {
                return Err(L2StateError::SetOutOfRange {
                    slice: set.slice,
                    set: set.set,
                });
            }
            if set.ways.len() > self.geometry.associativity as usize {
                return Err(L2StateError::SetOverfull { set: set.set });
            }
            if let Some(previous) = previous
                && (set.slice, set.set) <= previous
            {
                return Err(L2StateError::UnsortedSets);
            }
            if !seen_sets.insert((set.slice, set.set)) {
                return Err(L2StateError::DuplicateSet {
                    slice: set.slice,
                    set: set.set,
                });
            }
            previous = Some((set.slice, set.set));
            let mut seen_lines = BTreeSet::new();
            let mut seen_recencies = BTreeSet::new();
            for way in &set.ways {
                if way.line.domain() != self.address_domain {
                    return Err(L2StateError::AddressDomainMismatch);
                }
                if way.valid_mask == 0 && way.pending_mask == 0 {
                    return Err(L2StateError::EmptyWay {
                        line: way.line.clone(),
                    });
                }
                match self.geometry.replacement.as_str() {
                    // SM86 SRRIP stores a three-bit RRPV with a maximum of
                    // three.  A forged value changes victim selection and
                    // would fail the native access preflight.
                    "srrip" if !(0..=3).contains(&way.recency) => {
                        return Err(L2StateError::InvalidRecency {
                            line: way.line.clone(),
                            recency: way.recency,
                            clock: self.clock,
                            replacement: "srrip",
                        });
                    }
                    // LRU timestamps are nonnegative clock values; BIP uses
                    // the same monotonically advancing clock but may insert
                    // below the current minimum, so negative BIP recencies
                    // are legal.  Both policies require unique live stamps:
                    // native updates advance the clock for every sector and
                    // BIP's decrement is checked before publication.
                    "lru" if way.recency < 0 || way.recency > self.clock => {
                        return Err(L2StateError::InvalidRecency {
                            line: way.line.clone(),
                            recency: way.recency,
                            clock: self.clock,
                            replacement: "lru",
                        });
                    }
                    "bip" if way.recency > self.clock => {
                        return Err(L2StateError::InvalidRecency {
                            line: way.line.clone(),
                            recency: way.recency,
                            clock: self.clock,
                            replacement: "bip",
                        });
                    }
                    _ => {}
                }
                if !matches!(self.geometry.replacement.as_str(), "srrip")
                    && !seen_recencies.insert(way.recency)
                {
                    return Err(L2StateError::DuplicateRecency {
                        slice: set.slice,
                        set: set.set,
                        recency: way.recency,
                    });
                }
                if (way.valid_mask | way.dirty_mask | way.pending_mask) & !0x0f != 0 {
                    return Err(L2StateError::InvalidSectorMask);
                }
                if way.pending_mask & way.valid_mask != 0 {
                    return Err(L2StateError::PendingAndValidOverlap);
                }
                if way.dirty_mask & !way.valid_mask != 0 {
                    return Err(L2StateError::DirtyWithoutValid);
                }
                // Event-discipline allocations use the generation as part of
                // the fill/response identity. A complete boundary cannot
                // silently drop a resident event-mode way (or any pending
                // way) merely because its generation is absent: the omitted
                // allocation would make a later same-line response
                // ambiguous. Immediate-mode ways may legitimately omit this
                // identity because they have no deferred fill ownership.
                if self.ownership_complete
                    && way.generation.is_none()
                    && (matches!(self.access_discipline, Some(L2AccessDisciplineState::Event))
                        || way.pending_mask != 0)
                {
                    return Err(L2StateError::WayGenerationMissing {
                        line: way.line.clone(),
                    });
                }
                if !seen_lines.insert(way.line.clone()) {
                    return Err(L2StateError::DuplicateWayLine { set: set.set });
                }
                if let Some(generation) = way.generation {
                    let identity = (way.line.clone(), generation);
                    if live_ways.insert(identity.clone(), way).is_some() {
                        return Err(L2StateError::DuplicateWayIdentity);
                    }
                    if !live_generation_values.insert(generation) {
                        return Err(L2StateError::DuplicateGeneration { generation });
                    }
                    live_allocations.insert(identity);
                }
                if let CanonicalLineKey::VirtualAbsolute { line_id } = &way.line {
                    let (expected_slice, expected_set) = expected_l2_placement(
                        *line_id,
                        &self.geometry.slice_hash,
                        self.geometry.num_slices,
                        self.geometry.sets_per_slice,
                    )?;
                    if (set.slice, set.set) != (expected_slice, expected_set) {
                        return Err(L2StateError::LinePlacementMismatch {
                            line: way.line.clone(),
                            found_slice: set.slice,
                            found_set: set.set,
                            expected_slice,
                            expected_set,
                        });
                    }
                }
            }
        }
        let mut seen_allocations = BTreeSet::new();
        let mut seen_allocation_generation_values = BTreeSet::new();
        let mut duplicate_allocation_generation = None;
        let mut previous_allocation = None;
        for allocation in &self.allocations {
            if allocation.line.domain() != self.address_domain {
                return Err(L2StateError::AddressDomainMismatch);
            }
            let key = (allocation.line.clone(), allocation.generation);
            if !seen_allocations.insert(key) {
                return Err(L2StateError::DuplicateAllocation);
            }
            let order = (&allocation.line, allocation.generation);
            if let Some(previous) = previous_allocation
                && order <= previous
            {
                return Err(L2StateError::UnsortedAllocations);
            }
            previous_allocation = Some(order);
            if !seen_allocation_generation_values.insert(allocation.generation) {
                duplicate_allocation_generation = Some(allocation.generation);
            }
            let mut seen_sectors = BTreeSet::new();
            let mut previous_sector = None;
            for sector in &allocation.sectors {
                if sector.sector >= 4 || !seen_sectors.insert(sector.sector) {
                    return Err(L2StateError::DuplicateAllocationSector);
                }
                if let Some(previous) = previous_sector
                    && sector.sector <= previous
                {
                    return Err(L2StateError::UnsortedAllocationSectors);
                }
                previous_sector = Some(sector.sector);
            }
        }
        let maximum_generation = live_allocations
            .iter()
            .map(|(_, generation)| *generation)
            .chain(
                self.allocations
                    .iter()
                    .map(|allocation| allocation.generation),
            )
            .max();
        if let (Some(next_generation), Some(maximum_generation)) =
            (self.next_generation, maximum_generation)
            && next_generation <= maximum_generation
        {
            return Err(L2StateError::GenerationFrontierMismatch);
        }
        if self.ownership_complete {
            let projected_allocations = self
                .allocations
                .iter()
                .map(|allocation| (allocation.line.clone(), allocation.generation))
                .collect::<BTreeSet<_>>();
            if projected_allocations != live_allocations {
                return Err(L2StateError::AllocationWayMismatch);
            }
        }
        let mut seen_response_identities = BTreeSet::new();
        let mut seen_requests = BTreeSet::new();
        let mut previous_request = None;
        for owner in &self.request_owners {
            if owner.request.sm_id == u32::MAX {
                return Err(L2StateError::InvalidSmIdentity);
            }
            let response_identity = (
                owner.request.sm_id,
                owner.request.response_warp,
                owner.request.warp_slot,
                owner.request.instruction,
                owner.request.sector_index,
            );
            if !seen_response_identities.insert(response_identity) {
                return Err(L2StateError::DuplicateResponseIdentity);
            }
            let key = (
                owner.request.sm_id,
                owner.request.response_warp,
                owner.request.warp_slot,
                owner.request.instruction,
                owner.request.sector_index,
                owner.request.response_kind,
            );
            if !seen_requests.insert(key) {
                return Err(L2StateError::DuplicateRequestOwner);
            }
            let order = (
                owner.request.sm_id,
                owner.request.response_warp,
                owner.request.warp_slot,
                owner.request.instruction,
                owner.request.sector_index,
                &owner.request.response_kind,
            );
            if let Some(previous) = previous_request
                && order <= previous
            {
                return Err(L2StateError::UnsortedRequestOwners);
            }
            previous_request = Some(order);
        }
        let owners_by_request = self
            .request_owners
            .iter()
            .map(|owner| (owner.request.clone(), owner))
            .collect::<BTreeMap<_, _>>();
        if self
            .request_owners
            .iter()
            .any(|owner| matches!(owner.request.response_kind, TypedResponseKind::None))
        {
            return Err(L2StateError::InvalidRequestOwnerKind);
        }
        if let Some(generation) = duplicate_allocation_generation {
            return Err(L2StateError::DuplicateGeneration { generation });
        }
        // A response owner is part of the executable state, even when no
        // pending allocation currently references it.  L1 and direct
        // Warp/AsyncCopy replies must retain the allocation identity that
        // priced the response; otherwise an orphan owner can be marked
        // complete and the next response event is absent from the typed
        // transition state. An out-of-range direct-response identity remains
        // explicitly opaque and cannot certify a complete boundary.
        if self.ownership_complete {
            for owner in &self.request_owners {
                if owner.request.warp_slot < 0 {
                    return Err(L2StateError::NegativeRequestWarpSlot {
                        warp_slot: owner.request.warp_slot,
                    });
                }
                match owner.request.response_kind {
                    TypedResponseKind::L1Fill
                    | TypedResponseKind::L1SectorFill
                    | TypedResponseKind::Warp
                    | TypedResponseKind::AsyncCopy => {
                        if matches!(owner.line, CanonicalLineKey::Unknown { .. }) {
                            if !matches!(
                                owner.request.response_kind,
                                TypedResponseKind::Warp | TypedResponseKind::AsyncCopy
                            ) || owner.sector_mask != 0
                                || owner.allocation_generation.is_some()
                            {
                                return Err(L2StateError::OpaqueRequestOwnerIdentity);
                            }
                            continue;
                        }
                        if owner.sector_mask == 0 || owner.sector_mask & !0x0f != 0 {
                            return Err(L2StateError::InvalidRequestOwnerSectorMask);
                        }
                        if owner.line.domain() != self.address_domain {
                            return Err(L2StateError::RequestOwnerAddressDomainMismatch);
                        }
                        let Some(generation) = owner.allocation_generation else {
                            return Err(L2StateError::RequestOwnerGenerationMissing);
                        };
                        let Some(allocation) = self.allocations.iter().find(|allocation| {
                            allocation.line == owner.line && allocation.generation == generation
                        }) else {
                            return Err(L2StateError::RequestOwnerAllocationMissing);
                        };
                        let allocation_mask = allocation
                            .sectors
                            .iter()
                            .fold(0_u8, |mask, sector| mask | (1_u8 << sector.sector));
                        if owner.sector_mask & !allocation_mask != 0 {
                            return Err(L2StateError::RequestOwnerSectorMismatch);
                        }
                    }
                    TypedResponseKind::None => unreachable!("response-owner kind was validated"),
                }
            }
        }
        for allocation in &self.allocations {
            let Some(way) = live_ways.get(&(allocation.line.clone(), allocation.generation)) else {
                return Err(L2StateError::AllocationWithoutWay);
            };
            let expected = (0..4_u8)
                .filter_map(|sector| {
                    let bit = 1_u8 << sector;
                    let status = if way.pending_mask & bit != 0 {
                        L2SectorStatus::Pending
                    } else if way.dirty_mask & bit != 0 {
                        L2SectorStatus::Dirty
                    } else if way.valid_mask & bit != 0 {
                        L2SectorStatus::Valid
                    } else {
                        return None;
                    };
                    Some((sector, status))
                })
                .collect::<Vec<_>>();
            let actual = allocation
                .sectors
                .iter()
                .map(|sector| (sector.sector, sector.status.clone()))
                .collect::<Vec<_>>();
            if actual != expected {
                return Err(L2StateError::AllocationMaskMismatch {
                    line: allocation.line.clone(),
                    generation: allocation.generation,
                });
            }
            for sector in &allocation.sectors {
                let bit = 1_u8 << sector.sector;
                if !matches!(sector.status, L2SectorStatus::Pending) {
                    if sector.producer.is_some() || !sector.consumers.is_empty() {
                        return Err(L2StateError::NonPendingOwnership {
                            line: allocation.line.clone(),
                            generation: allocation.generation,
                            sector: sector.sector,
                        });
                    }
                    if let Some(ready_at) = sector.ready_at {
                        let has_outstanding_fill_owner = self.request_owners.iter().any(|owner| {
                            owner.line == allocation.line
                                && owner.allocation_generation == Some(allocation.generation)
                                && owner.sector_mask & bit != 0
                                && matches!(owner.state, L2RequestState::Outstanding)
                                && matches!(
                                    owner.request.response_kind,
                                    TypedResponseKind::L1Fill
                                        | TypedResponseKind::L1SectorFill
                                        | TypedResponseKind::Warp
                                        | TypedResponseKind::AsyncCopy
                                )
                        });
                        if !has_outstanding_fill_owner {
                            return Err(L2StateError::ReadyTimeOnNonPending {
                                line: allocation.line.clone(),
                                generation: allocation.generation,
                                sector: sector.sector,
                                ready_at,
                            });
                        }
                    }
                    continue;
                }
                let mut previous_consumer = None;
                let mut seen_consumers = BTreeSet::new();
                for consumer in &sector.consumers {
                    if !seen_consumers.insert(consumer.clone()) {
                        return Err(L2StateError::DuplicateSectorConsumer);
                    }
                    if let Some(previous) = previous_consumer
                        && consumer <= previous
                    {
                        return Err(L2StateError::UnsortedSectorConsumers);
                    }
                    previous_consumer = Some(consumer);
                    let Some(owner) = owners_by_request.get(consumer) else {
                        return Err(L2StateError::OwnershipReferenceMissing);
                    };
                    if self.ownership_complete
                        && !matches!(owner.state, L2RequestState::Outstanding)
                    {
                        return Err(L2StateError::PendingConsumerNotOutstanding {
                            line: allocation.line.clone(),
                            generation: allocation.generation,
                            sector: sector.sector,
                        });
                    }
                    if owner.line != allocation.line
                        || owner.allocation_generation != Some(allocation.generation)
                        || owner.sector_mask & bit == 0
                    {
                        return Err(L2StateError::OwnershipReferenceMismatch {
                            line: allocation.line.clone(),
                            generation: allocation.generation,
                            sector: sector.sector,
                        });
                    }
                }
                if let Some(producer) = &sector.producer {
                    if !seen_consumers.contains(producer) {
                        return Err(L2StateError::ProducerNotConsumer);
                    }
                } else if self.ownership_complete {
                    return Err(L2StateError::CompletePendingWithoutProducer {
                        line: allocation.line.clone(),
                        generation: allocation.generation,
                        sector: sector.sector,
                    });
                }
                if self.ownership_complete && sector.ready_at.is_none() {
                    return Err(L2StateError::CompletePendingWithoutReadyTime {
                        line: allocation.line.clone(),
                        generation: allocation.generation,
                        sector: sector.sector,
                    });
                }
            }
        }
        if self.ownership_complete && !self.unknown_state.is_empty() {
            return Err(L2StateError::CompleteWithUnknownState);
        }
        // Allocation generations are part of the executable discipline. The
        // immediate path never creates them, while every resident event-mode
        // line carries one; accepting a contradictory pairing would make a
        // typed boundary impossible to resume without guessing its token
        // identity. Keep this check after the more specific structural checks
        // above so malformed masks/placements retain their precise errors.
        let has_generation_free_way = self
            .sets
            .iter()
            .flat_map(|set| set.ways.iter())
            .any(|way| way.generation.is_none());
        let has_generation_bearing_way = self
            .sets
            .iter()
            .flat_map(|set| set.ways.iter())
            .any(|way| way.generation.is_some());
        if self.ownership_complete {
            let counters_nonzero = [
                self.counters.hits,
                self.counters.misses,
                self.counters.lookup_hits,
                self.counters.lookup_requests,
                self.counters.line_misses,
                self.counters.sector_fill_misses,
                self.counters.write_tag_hits,
                self.observer_lookup_hits,
                self.observer_lookup_requests,
            ]
            .into_iter()
            .any(|value| value != 0);
            let active_state = !self.sets.is_empty()
                || !self.allocations.is_empty()
                || !self.request_owners.is_empty()
                || self.bip_counter != 0
                || self.next_generation.is_some_and(|next| next != 0)
                || counters_nonzero;
            if self.access_discipline.is_none() && active_state {
                return Err(L2StateError::DisciplineMissing);
            }
            match self.access_discipline {
                None | Some(L2AccessDisciplineState::Immediate) if has_generation_bearing_way => {
                    return Err(L2StateError::DisciplineGenerationMismatch);
                }
                Some(L2AccessDisciplineState::Event) if has_generation_free_way => {
                    return Err(L2StateError::DisciplineGenerationMismatch);
                }
                _ => {}
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerBoundaryState {
    pub schema_version: u32,
    pub status: PowerModelStatus,
    pub clock: PowerClockState,
    pub sample_period: Option<u64>,
    pub next_sample: Option<u64>,
    pub domains: Vec<PowerDomainState>,
    pub samples: Vec<PowerSampleState>,
    pub omitted_observations: Vec<String>,
    pub unknown_state: Vec<String>,
}

impl PowerBoundaryState {
    pub fn activity_only(frequency_hz: Option<u64>, sample_period: Option<u64>) -> Self {
        let next_sample = sample_period.filter(|period| *period != 0);
        Self {
            schema_version: TYPED_STATE_SCHEMA_VERSION,
            status: PowerModelStatus::ActivityOnly,
            clock: PowerClockState {
                clock_domain: "sm_clock".to_owned(),
                fixed_frequency_hz: frequency_hz,
                dvfs_modeled: false,
            },
            sample_period,
            next_sample,
            domains: vec![PowerDomainState {
                domain: "modeled_event_system".to_owned(),
                status: PowerDomainStatus::Idle,
                last_cycle: 0,
                clocked_cycles: 0,
                gated_cycles: 0,
                sm_wake_events: 0,
                completion_events: 0,
                memory_events: 0,
                queue_events: 0,
            }],
            samples: Vec::new(),
            omitted_observations: vec![
                "watts".to_owned(),
                "per_instruction_energy".to_owned(),
                "lane_utilization".to_owned(),
                "thermal_and_dvfs_feedback".to_owned(),
            ],
            unknown_state: Vec::new(),
        }
    }

    pub fn advance_to(&mut self, cycle: u64) -> Result<(), PowerStateError> {
        if self.schema_version != TYPED_STATE_SCHEMA_VERSION {
            return Err(PowerStateError::SchemaVersion);
        }
        // Sampling is part of the state transition. Stage the complete update
        // so an overflow or bounded-output refusal never leaves half an
        // integration interval committed.
        let mut next = self.clone();
        next.emit_samples_through(cycle)?;
        next.integrate_to(cycle)?;
        *self = next;
        Ok(())
    }

    pub(super) fn integrate_to(&mut self, cycle: u64) -> Result<(), PowerStateError> {
        for domain in &mut self.domains {
            if cycle < domain.last_cycle {
                return Err(PowerStateError::TimeTravel);
            }
            let delta = cycle - domain.last_cycle;
            match domain.status {
                PowerDomainStatus::Clocked => {
                    domain.clocked_cycles = domain
                        .clocked_cycles
                        .checked_add(delta)
                        .ok_or(PowerStateError::CounterOverflow)?;
                }
                PowerDomainStatus::Gated => {
                    domain.gated_cycles = domain
                        .gated_cycles
                        .checked_add(delta)
                        .ok_or(PowerStateError::CounterOverflow)?;
                }
                PowerDomainStatus::Idle => {}
            }
            domain.last_cycle = cycle;
        }
        Ok(())
    }

    pub fn record_event(&mut self, event: &str) -> Result<(), PowerStateError> {
        if self.schema_version != TYPED_STATE_SCHEMA_VERSION {
            return Err(PowerStateError::SchemaVersion);
        }
        let mut next = self.clone();
        next.record_event_inner(event)?;
        *self = next;
        Ok(())
    }

    /// Apply the narrow scalar-macro power projection used by the reference
    /// event loop.  The macro is accepted only for a closed issue-only region;
    /// this method records its declared clock/event contribution before the
    /// continuation event is observed.  It is not a hardware energy model and
    /// does not manufacture completion tokens.
    pub fn record_scalar_macro_completion(
        &mut self,
        start_cycle: u64,
        end_cycle: u64,
    ) -> Result<(), PowerStateError> {
        if end_cycle < start_cycle {
            return Err(PowerStateError::TimeTravel);
        }
        let mut next = self.clone();
        next.advance_to(start_cycle)?;
        for domain in &mut next.domains {
            if end_cycle < domain.last_cycle {
                return Err(PowerStateError::TimeTravel);
            }
            domain.status = PowerDomainStatus::Clocked;
            domain.queue_events = domain
                .queue_events
                .checked_add(1)
                .ok_or(PowerStateError::CounterOverflow)?;
            domain.sm_wake_events = domain
                .sm_wake_events
                .checked_add(1)
                .ok_or(PowerStateError::CounterOverflow)?;
            domain.last_cycle = end_cycle;
        }
        *self = next;
        Ok(())
    }

    pub(super) fn record_event_inner(&mut self, event: &str) -> Result<(), PowerStateError> {
        let domain = self.domains.first_mut().ok_or(PowerStateError::NoDomains)?;
        domain.status = PowerDomainStatus::Clocked;
        domain.queue_events = domain
            .queue_events
            .checked_add(1)
            .ok_or(PowerStateError::CounterOverflow)?;
        if event.contains("l2")
            || event.contains("memory")
            || event == "mem_response"
            || event == "atomic_completion"
        {
            domain.memory_events = domain
                .memory_events
                .checked_add(1)
                .ok_or(PowerStateError::CounterOverflow)?;
        }
        if event == "sm_wake" {
            domain.sm_wake_events = domain
                .sm_wake_events
                .checked_add(1)
                .ok_or(PowerStateError::CounterOverflow)?;
        }
        if event == "memory_response"
            || event == "l2_fill"
            || event == "atomic_completion"
            || event == "memory_write_ack"
        {
            domain.completion_events = domain
                .completion_events
                .checked_add(1)
                .ok_or(PowerStateError::CounterOverflow)?;
        }
        Ok(())
    }

    pub(super) fn emit_samples_through(&mut self, cycle: u64) -> Result<(), PowerStateError> {
        let Some(period) = self.sample_period else {
            return Ok(());
        };
        if period == 0 {
            return Err(PowerStateError::InvalidSamplePeriod);
        }
        let Some(mut next) = self.next_sample else {
            return Ok(());
        };
        while next <= cycle {
            if self.samples.len() >= 100_000 {
                return Err(PowerStateError::SampleLimit);
            }
            self.integrate_to(next)?;
            let activity_total = self.domains.iter().try_fold(0_u64, |sum, domain| {
                sum.checked_add(domain.clocked_cycles)
                    .ok_or(PowerStateError::CounterOverflow)
            })?;
            self.samples.push(PowerSampleState {
                cycle: next,
                activity_total,
                domain_clocked_cycles: self
                    .domains
                    .iter()
                    .map(|domain| domain.clocked_cycles)
                    .collect(),
                domain_gated_cycles: self
                    .domains
                    .iter()
                    .map(|domain| domain.gated_cycles)
                    .collect(),
            });
            next = next
                .checked_add(period)
                .ok_or(PowerStateError::TimeOverflow)?;
        }
        self.next_sample = Some(next);
        Ok(())
    }

    pub fn validate(&self) -> Result<(), PowerStateError> {
        if self.schema_version != TYPED_STATE_SCHEMA_VERSION {
            return Err(PowerStateError::SchemaVersion);
        }
        if self.domains.is_empty() {
            return Err(PowerStateError::NoDomains);
        }
        if self.clock.clock_domain.is_empty() {
            return Err(PowerStateError::InvalidClockDomain);
        }
        if self.clock.fixed_frequency_hz == Some(0) {
            return Err(PowerStateError::InvalidFrequency);
        }
        if self.clock.dvfs_modeled && self.status != PowerModelStatus::AnalyticCoefficients {
            return Err(PowerStateError::UnknownDvfs);
        }
        if self.status != PowerModelStatus::Unknown && !self.unknown_state.is_empty() {
            return Err(PowerStateError::UnknownState);
        }
        let mut domain_names = BTreeSet::new();
        let mut domain_time = None;
        for domain in &self.domains {
            if domain.domain.is_empty() || !domain_names.insert(&domain.domain) {
                return Err(PowerStateError::DomainIdentity);
            }
            let integrated = domain
                .clocked_cycles
                .checked_add(domain.gated_cycles)
                .ok_or(PowerStateError::CounterOverflow)?;
            if integrated > domain.last_cycle {
                return Err(PowerStateError::IntegralAheadOfClock);
            }
            if let Some(previous) = domain_time
                && previous != domain.last_cycle
            {
                return Err(PowerStateError::DomainTimeMismatch);
            }
            domain_time = Some(domain.last_cycle);
        }
        match (self.sample_period, self.next_sample) {
            (Some(0), _) => return Err(PowerStateError::InvalidSamplePeriod),
            (None, Some(_)) => return Err(PowerStateError::SampleSchedule),
            (Some(_), None) => return Err(PowerStateError::SampleSchedule),
            _ => {}
        }
        if self.sample_period.is_none() && !self.samples.is_empty() {
            return Err(PowerStateError::SampleSchedule);
        }
        let mut previous_sample: Option<&PowerSampleState> = None;
        for sample in &self.samples {
            if sample.domain_clocked_cycles.len() != self.domains.len()
                || sample.domain_gated_cycles.len() != self.domains.len()
            {
                return Err(PowerStateError::SampleDomainCount);
            }
            let calculated_activity =
                sample
                    .domain_clocked_cycles
                    .iter()
                    .try_fold(0_u64, |sum, value| {
                        sum.checked_add(*value)
                            .ok_or(PowerStateError::CounterOverflow)
                    })?;
            if calculated_activity != sample.activity_total {
                return Err(PowerStateError::SampleActivityMismatch);
            }
            if sample
                .domain_clocked_cycles
                .iter()
                .zip(&self.domains)
                .any(|(sampled, domain)| *sampled > domain.clocked_cycles)
                || sample
                    .domain_gated_cycles
                    .iter()
                    .zip(&self.domains)
                    .any(|(sampled, domain)| *sampled > domain.gated_cycles)
            {
                return Err(PowerStateError::SampleAheadOfDomain);
            }
            for (&clocked, &gated) in sample
                .domain_clocked_cycles
                .iter()
                .zip(&sample.domain_gated_cycles)
            {
                if clocked
                    .checked_add(gated)
                    .ok_or(PowerStateError::CounterOverflow)?
                    > sample.cycle
                {
                    return Err(PowerStateError::SampleIntegralAheadOfCycle);
                }
            }
            if let Some(previous) = previous_sample
                && (sample.cycle <= previous.cycle
                    || sample.activity_total < previous.activity_total
                    || sample
                        .domain_clocked_cycles
                        .iter()
                        .zip(&previous.domain_clocked_cycles)
                        .any(|(current, previous)| current < previous)
                    || sample
                        .domain_gated_cycles
                        .iter()
                        .zip(&previous.domain_gated_cycles)
                        .any(|(current, previous)| current < previous))
            {
                return Err(PowerStateError::UnsortedSamples);
            }
            previous_sample = Some(sample);
        }
        if self.samples.iter().any(|sample| {
            self.domains
                .iter()
                .any(|domain| sample.cycle > domain.last_cycle)
        }) {
            return Err(PowerStateError::SampleAheadOfDomainTime);
        }
        if let Some(period) = self.sample_period {
            // `activity_only` starts the schedule at cycle `period` and
            // advances it by exactly one period for every emitted sample.
            // Keeping the complete cadence is part of the resumable state:
            // accepting a later `next_sample` would silently skip an
            // integration interval, while an off-phase sample would shift
            // every future boundary. Run this after per-sample integrity
            // checks so malformed sample payloads retain their precise error.
            let mut expected = period;
            for sample in &self.samples {
                if sample.cycle != expected {
                    return Err(PowerStateError::SampleSchedule);
                }
                expected = expected
                    .checked_add(period)
                    .ok_or(PowerStateError::TimeOverflow)?;
            }
            if self.next_sample != Some(expected) {
                return Err(PowerStateError::SampleSchedule);
            }
        }
        if let (Some(last_sample), Some(next_sample)) = (previous_sample, self.next_sample)
            && next_sample <= last_sample.cycle
        {
            return Err(PowerStateError::SampleSchedule);
        }
        if let (Some(next_sample), Some(last_cycle)) = (self.next_sample, domain_time)
            && next_sample <= last_cycle
        {
            return Err(PowerStateError::SampleSchedule);
        }
        Ok(())
    }
}
