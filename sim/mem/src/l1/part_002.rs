// Lexical implementation section 3/3; included by the parent facade.
impl L1Cache {/// Event-mode eligible access with a caller-characterized lower bound on
    /// finite miss admission. The floor applies only to this access.
    pub fn access_eligible_ev_with_mshr_floor(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
        mshr_floor: NonZeroU32,
    ) -> Result<EvAccess, L1AccessError> {
        self.preflight_latch(FillTiming::Event)?;
        let budget = L1MutationBudget {
            eligible_sectors: 1,
            recency_touches: 1,
            new_fills: 1,
            ..L1MutationBudget::for_origin(origin)
        };
        if self.preflight_mutation_budget(budget).is_err() {
            let mut shadow = self.clone();
            let outcome =
                shadow.access_eligible_event_impl(origin, sector_id, now, Some(mshr_floor))?;
            *self = shadow;
            return Ok(outcome);
        }
        self.access_eligible_event_impl(origin, sector_id, now, Some(mshr_floor))
    }

    /// Event-mode cache mutation when no dependence barrier consumes a hit's
    /// ready time. Miss, merge, stall, fill ownership, and counters are unchanged.
    pub fn access_eligible_ev_unobserved(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
    ) -> Result<EvAccessUnobserved, L1AccessError> {
        self.preflight_latch(FillTiming::Event)?;
        let budget = L1MutationBudget {
            eligible_sectors: 1,
            recency_touches: 1,
            new_fills: 1,
            ..L1MutationBudget::for_origin(origin)
        };
        if self.preflight_mutation_budget(budget).is_err() {
            let mut shadow = self.clone();
            let outcome =
                shadow.access_eligible_event_unobserved_impl(origin, sector_id, now, None)?;
            *self = shadow;
            return Ok(outcome);
        }
        self.access_eligible_event_unobserved_impl(origin, sector_id, now, None)
    }

    /// Unobserved counterpart to [`Self::access_eligible_ev_with_mshr_floor`].
    pub fn access_eligible_ev_unobserved_with_mshr_floor(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
        mshr_floor: NonZeroU32,
    ) -> Result<EvAccessUnobserved, L1AccessError> {
        self.preflight_latch(FillTiming::Event)?;
        let budget = L1MutationBudget {
            eligible_sectors: 1,
            recency_touches: 1,
            new_fills: 1,
            ..L1MutationBudget::for_origin(origin)
        };
        if self.preflight_mutation_budget(budget).is_err() {
            let mut shadow = self.clone();
            let outcome = shadow.access_eligible_event_unobserved_impl(
                origin,
                sector_id,
                now,
                Some(mshr_floor),
            )?;
            *self = shadow;
            return Ok(outcome);
        }
        self.access_eligible_event_unobserved_impl(origin, sector_id, now, Some(mshr_floor))
    }

    /// Complete an event-mode fill AT the response event's time. A
    /// sector fill whose line was evicted while it was in flight is
    /// DROPPED, like the lazy applier. A child may return while its
    /// parent line allocation is still pending; retain its sector bit so
    /// the parent completion can expose the complete allocation. Returns
    /// whether the generation-matched data was retained.
    ///
    /// Deliberate divergence from the constant path: `apply_fills`
    /// breaks SAME-TIMESTAMP ties line-before-sector (a batch-applier
    /// artifact the Python lockstep pins), while event mode completes
    /// same-cycle fills in response order — under eviction pressure the
    /// two orders can pick different LRU victims. Sub-cycle completion
    /// order is below the model's resolution either way; the event
    /// order is the shared-L2 system's single source of truth.
    pub fn complete_fill(&mut self, fill: FillKey) -> Result<bool, L1AccessError> {
        self.preflight_latch(FillTiming::Event)?;
        if !self.is_fill_in_flight(fill) {
            return Err(L1AccessError::FillNotInFlight { fill });
        }
        let will_touch = match fill {
            FillKey::Line { .. } => true,
            FillKey::Sector {
                line, generation, ..
            } => self.line_resident(line) && self.line_generations.get(&line) == Some(&generation),
        };
        if will_touch {
            self.next_stamp.preflight("recency", 1)?;
        }
        self.commit_latch(FillTiming::Event);
        match fill {
            FillKey::Line { line, .. } => {
                self.preflight_dirty_eviction(Some(line))?;
                self.line_fills.remove(&line);
                self.touch_mru(line)?;
                self.evict_beyond_capacity()?;
                Ok(true)
            }
            FillKey::Sector {
                line,
                sector,
                generation,
            } => {
                self.sector_fills.remove(&(line, sector, generation));
                let generation_matches = self.line_resident(line)
                    && self.line_generations.get(&line) == Some(&generation);
                let allocation_matches = self.line_generations.get(&line) == Some(&generation)
                    && (self.line_resident(line) || self.line_fills.contains_key(&line));
                if allocation_matches {
                    *self.sectors.entry(line).or_default() |= 1 << sector;
                    if generation_matches {
                        self.touch_mru(line)?;
                    }
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
        }
    }

    fn probe_write_impl(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
        accounting: WriteAccounting,
    ) -> Result<bool, L1AccessError> {
        let line = sector_id >> 2;
        let sector = (sector_id & 3) as u8;
        self.apply_fills(now)?;
        let tag_resident = self.line_resident(line);
        let hit = tag_resident
            && self
                .sectors
                .get(&line)
                .is_some_and(|mask| mask & (1 << sector) != 0);
        if hit {
            self.next_stamp.preflight("recency", 1)?;
        }
        let deltas: &[(L1CounterField, u64)] = match accounting {
            WriteAccounting::Store => &[
                (L1CounterField::StoreSectors, 1),
                (L1CounterField::StoreTagHits, u64::from(tag_resident)),
                (L1CounterField::StoreHits, u64::from(hit)),
            ],
            WriteAccounting::Reduction => &[
                (L1CounterField::ReductionSectors, 1),
                (L1CounterField::ReductionHits, u64::from(hit)),
            ],
        };
        self.counters = self.counters.add_fields_checked(origin, deltas)?;
        if hit {
            self.touch_mru(line)?;
            if origin == MemoryOrigin::Global
                && accounting == WriteAccounting::Store
                && self.store_policy == L1StorePolicy::ResidentHitWriteBack
            {
                self.mark_dirty(line, 1_u8 << sector)?;
            }
        }
        Ok(hit)
    }

    /// Hit-test a global-class store sector against the current state without
    /// allocating on miss. Under the default policy this is a write-through
    /// no-allocate hypothesis; the explicit resident-hit writeback policy marks
    /// a valid hit dirty for the terminal flush path.
    /// Counts BOTH candidate hit rules — sector-valid (`store_hits`,
    /// the ported-era diagnostic) and the §6.4 write-class line-TAG
    /// rule (`store_tag_hits`) — without changing state behavior (MRU
    /// touch stays on the sector-valid hit only, preserving the L1
    /// state trajectory the scored op_ld exactness was proven on).
    pub fn probe_store(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
    ) -> Result<bool, L1AccessError> {
        let budget = L1MutationBudget {
            store_sectors: 1,
            recency_touches: 1,
            ..L1MutationBudget::for_origin(origin)
        };
        if self.preflight_mutation_budget(budget).is_err()
            || self.preflight_due_fill_recency(now, 1).is_err()
        {
            let mut shadow = self.clone();
            let hit = shadow.probe_write_impl(origin, sector_id, now, WriteAccounting::Store)?;
            *self = shadow;
            return Ok(hit);
        }
        self.probe_write_impl(origin, sector_id, now, WriteAccounting::Store)
    }

    /// Hit-test a RED/REDG sector through the same physical write path as a
    /// store while recording a distinct reduction observer lane. RED therefore
    /// contributes to the unified L1TEX hit rate but never to ordinary STG
    /// attribution; its RMW/writeback/coherence tails remain explicit on the SM
    /// result.
    pub fn probe_reduction(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
    ) -> Result<bool, L1AccessError> {
        let budget = L1MutationBudget {
            reduction_sectors: 1,
            recency_touches: 1,
            ..L1MutationBudget::for_origin(origin)
        };
        if self.preflight_mutation_budget(budget).is_err()
            || self.preflight_due_fill_recency(now, 1).is_err()
        {
            let mut shadow = self.clone();
            let hit =
                shadow.probe_write_impl(origin, sector_id, now, WriteAccounting::Reduction)?;
            *self = shadow;
            return Ok(hit);
        }
        self.probe_write_impl(origin, sector_id, now, WriteAccounting::Reduction)
    }

    /// Record write-class lookup hits derived from the exact instruction's
    /// sector/wavefront topology. This is emission accounting only: the
    /// modeled write-through/no-allocate L1 trajectory remains unchanged.
    pub fn count_write_lookup_hits(
        &mut self,
        origin: MemoryOrigin,
        hits: u64,
    ) -> Result<(), L1AccessError> {
        self.apply_counter_deltas(origin, &[(L1CounterField::WriteLookupHits, hits)])
    }

    /// Observe one complete write instruction through the architecture-local
    /// L1TEX write tag window.  The observer is deliberately independent of
    /// the physical read cache and emits only the number of sector lookups
    /// whose line tag was already resident before this instruction.  A fresh
    /// line is installed after all sectors in the request are classified, so
    /// same-request coalescing is not accidentally reported as a hit.
    pub fn observe_write_lookup_batch(&mut self, sectors: &[i64]) -> Result<u64, L1AccessError> {
        let config = self
            .cfg
            .arch
            .l1_write_lookup_window()
            .require("l1_write_lookup_window")?;
        if config.set_count == 0 || config.tag_slots == 0 {
            return Err(L1AccessError::WriteLookupWindowCapability(
                sim_arch::CapabilityError::ResearchGap {
                    arch: self.cfg.arch,
                    capability: "l1_write_lookup_window",
                    reason: "write-lookup window has zero sets or tag slots",
                },
            ));
        }
        let mut next = self
            .write_lookup_observer
            .as_deref()
            .cloned()
            .unwrap_or_else(|| L1WriteLookupObserver::new(config));
        let hits = next.observe_batch(sectors)?;
        self.write_lookup_observer = Some(Box::new(next));
        Ok(hits)
    }

    /// Manifest-tagged ineligible sectors (the ported bypass counter).
    /// `atom` marks the write-class (ATOM-token) subset — the caller
    /// owns the token, this cache never sees opcodes.
    pub fn count_bypass(
        &mut self,
        origin: MemoryOrigin,
        sectors: u64,
        atom: bool,
    ) -> Result<(), L1AccessError> {
        self.apply_counter_deltas(
            origin,
            &[
                (L1CounterField::BypassSectors, sectors),
                (L1CounterField::BypassAtomSectors, sectors * u64::from(atom)),
            ],
        )
    }

    /// Global-class sectors whose pc the classifier did not tag (v0
    /// unified-denominator extension).
    pub fn count_untagged(
        &mut self,
        origin: MemoryOrigin,
        sectors: u64,
    ) -> Result<(), L1AccessError> {
        self.apply_counter_deltas(origin, &[(L1CounterField::UntaggedSectors, sectors)])
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SectorExpansionError {
    #[error("data_width_bytes={data_width_bytes} is not strictly positive")]
    DataWidthInvalid { data_width_bytes: i32 },
    #[error("active lane index {lane} is outside the 32-lane address vector")]
    LaneIndexOutOfRange { lane: usize },
    #[error(
        "lane {lane} address {base} plus data_width_bytes={data_width_bytes} overflows the signed address domain"
    )]
    AddressOverflow {
        lane: usize,
        base: i64,
        data_width_bytes: i32,
    },
    #[error("unique sector count {sectors} exceeds the configured limit {limit}")]
    UniqueSectorLimitExceeded { limit: usize, sectors: usize },
    #[error("unique sector count {sectors} does not fit the host index domain")]
    UniqueSectorCountOutOfRange { sectors: u128 },
    #[error("sector expansion produced {produced} entries, expected {expected}")]
    CardinalityMismatch { expected: usize, produced: usize },
    #[error("unable to reserve {additional} sector entries for {buffer}")]
    CapacityUnavailable {
        buffer: &'static str,
        additional: usize,
    },
}

/// Validated lane-sector intervals for all address records of one instruction.
/// Planning is range-based: no sector range is enumerated until the exact
/// instruction-wide union is known to fit its consumer's bound.
#[derive(Debug, Default)]
pub struct SectorExpansionPlan {
    intervals: Vec<(i64, i64)>,
}

/// A plan whose exact instruction-wide union cardinality has been computed.
#[derive(Debug)]
pub struct PreparedSectorExpansion {
    intervals: Vec<(i64, i64)>,
    unique_count: usize,
}

fn collect_sector_intervals(
    lane_vector: &[i64; 32],
    active_lanes: impl Iterator<Item = usize>,
    data_width_bytes: i32,
) -> Result<([(i64, i64); 32], usize), SectorExpansionError> {
    if data_width_bytes <= 0 {
        return Err(SectorExpansionError::DataWidthInvalid { data_width_bytes });
    }
    let width_offset = i64::from(data_width_bytes)
        .checked_sub(1)
        .ok_or(SectorExpansionError::DataWidthInvalid { data_width_bytes })?;
    let mut lane_seen = [false; 32];
    let mut intervals = [(0_i64, 0_i64); 32];
    let mut interval_count = 0;
    for lane in active_lanes {
        let base = *lane_vector
            .get(lane)
            .ok_or(SectorExpansionError::LaneIndexOutOfRange { lane })?;
        if lane_seen[lane] {
            continue;
        }
        lane_seen[lane] = true;
        let last_addr =
            base.checked_add(width_offset)
                .ok_or(SectorExpansionError::AddressOverflow {
                    lane,
                    base,
                    data_width_bytes,
                })?;
        intervals[interval_count] = (base >> 5, last_addr >> 5);
        interval_count += 1;
    }
    Ok((intervals, interval_count))
}

impl SectorExpansionPlan {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one address record without enumerating its sector ranges. The plan
    /// is unchanged if validation or fallible reservation fails.
    pub fn include_record(
        &mut self,
        lane_vector: &[i64; 32],
        active_lanes: impl Iterator<Item = usize>,
        data_width_bytes: i32,
    ) -> Result<(), SectorExpansionError> {
        let (intervals, interval_count) =
            collect_sector_intervals(lane_vector, active_lanes, data_width_bytes)?;
        self.intervals.try_reserve(interval_count).map_err(|_| {
            SectorExpansionError::CapacityUnavailable {
                buffer: "sector interval plan",
                additional: interval_count,
            }
        })?;
        self.intervals
            .extend_from_slice(&intervals[..interval_count]);
        Ok(())
    }

    /// Compute the exact union over every included address record in
    /// O(intervals log intervals), still without enumerating sectors.
    pub fn prepare(self) -> Result<PreparedSectorExpansion, SectorExpansionError> {
        let mut sorted = Vec::new();
        sorted.try_reserve(self.intervals.len()).map_err(|_| {
            SectorExpansionError::CapacityUnavailable {
                buffer: "sorted sector intervals",
                additional: self.intervals.len(),
            }
        })?;
        sorted.extend_from_slice(&self.intervals);
        sorted.sort_unstable();

        let mut union_count = 0_u128;
        if let Some(&(initial_first, initial_last)) = sorted.first() {
            let mut first = initial_first;
            let mut last = initial_last;
            for &(next_first, next_last) in &sorted[1..] {
                if next_first <= last.saturating_add(1) {
                    last = last.max(next_last);
                } else {
                    union_count += (i128::from(last) - i128::from(first) + 1).unsigned_abs();
                    first = next_first;
                    last = next_last;
                }
            }
            union_count += (i128::from(last) - i128::from(first) + 1).unsigned_abs();
        }
        let unique_count = usize::try_from(union_count).map_err(|_| {
            SectorExpansionError::UniqueSectorCountOutOfRange {
                sectors: union_count,
            }
        })?;
        Ok(PreparedSectorExpansion {
            intervals: self.intervals,
            unique_count,
        })
    }
}

impl PreparedSectorExpansion {
    pub fn unique_count(&self) -> usize {
        self.unique_count
    }

    /// Expand only after the exact instruction-wide union fits `max_unique`.
    /// Covered ranges are tracked as intervals, so repeated or overlapping
    /// records enumerate each sector exactly once while retaining the original
    /// lane-major, record-major first-seen order.
    pub fn expand_bounded(self, max_unique: usize) -> Result<Vec<i64>, SectorExpansionError> {
        if self.unique_count > max_unique {
            return Err(SectorExpansionError::UniqueSectorLimitExceeded {
                limit: max_unique,
                sectors: self.unique_count,
            });
        }
        let mut staged = Vec::new();
        staged.try_reserve(self.unique_count).map_err(|_| {
            SectorExpansionError::CapacityUnavailable {
                buffer: "sector staging",
                additional: self.unique_count,
            }
        })?;
        let covered_capacity = self.intervals.len().min(self.unique_count);
        let mut covered: Vec<(i64, i64)> = Vec::new();
        covered.try_reserve(covered_capacity).map_err(|_| {
            SectorExpansionError::CapacityUnavailable {
                buffer: "covered sector intervals",
                additional: covered_capacity,
            }
        })?;

        for (first, last) in self.intervals {
            let start = covered.partition_point(|&(_, covered_last)| covered_last < first);
            let mut cursor = Some(first);
            for &(covered_first, covered_last) in &covered[start..] {
                let Some(next) = cursor else {
                    break;
                };
                if covered_first > last {
                    break;
                }
                if next < covered_first {
                    for sector in next..=last.min(covered_first.saturating_sub(1)) {
                        staged.push(sector);
                    }
                }
                cursor = if covered_last >= last {
                    None
                } else {
                    covered_last
                        .checked_add(1)
                        .map(|after_covered| next.max(after_covered))
                };
            }
            if let Some(next) = cursor {
                for sector in next..=last {
                    staged.push(sector);
                }
            }

            let left = covered
                .partition_point(|&(_, covered_last)| covered_last.saturating_add(1) < first);
            let right = left
                + covered[left..]
                    .partition_point(|&(covered_first, _)| covered_first <= last.saturating_add(1));
            if left == right {
                covered.insert(left, (first, last));
            } else {
                let merged = (first.min(covered[left].0), last.max(covered[right - 1].1));
                covered[left] = merged;
                covered.drain(left + 1..right);
            }
        }

        if staged.len() != self.unique_count {
            return Err(SectorExpansionError::CardinalityMismatch {
                expected: self.unique_count,
                produced: staged.len(),
            });
        }
        Ok(staged)
    }
}

