//! L1 visibility and dirty flush completion; mutates only L1 flush state.

use super::*;

impl L1Cache {
    /// Return the observer-only request-boundary merge provenance.
    pub const fn merge_audit(&self) -> L1MergeAudit {
        self.merge_audit
    }
}

impl L1Cache {
    pub(super) fn record_pending_merge_class(
        &mut self,
        origin: MemoryOrigin,
        already_requested: bool,
    ) {
        let lane = if already_requested {
            &mut self.merge_audit.already_requested_merges
        } else {
            &mut self.merge_audit.unrequested_pending_merges
        };
        lane[origin.index()] = lane[origin.index()].saturating_add(1);
    }
}

impl L1Cache {
    pub(super) fn record_merge_audit(
        &mut self,
        origin: MemoryOrigin,
        line: i64,
        sector: u8,
        sequence: u64,
    ) {
        let Some(batch) = self.active_lookup_batch else {
            return;
        };
        let lane = if sequence < batch.first_fill_sequence {
            &mut self.merge_audit.older_pending_merges
        } else {
            &mut self.merge_audit.same_request_merges
        };
        lane[origin.index()] = lane[origin.index()].saturating_add(1);
        let route_lane = if batch.pending_merges_are_hits {
            if sequence < batch.first_fill_sequence {
                &mut self.merge_audit.async_older_pending_merges
            } else {
                &mut self.merge_audit.async_same_request_merges
            }
        } else if sequence < batch.first_fill_sequence {
            &mut self.merge_audit.ordinary_older_pending_merges
        } else {
            &mut self.merge_audit.ordinary_same_request_merges
        };
        route_lane[origin.index()] = route_lane[origin.index()].saturating_add(1);
        let observed_merge_mask =
            self.sectors.get(&line).copied().unwrap_or(0) | (1_u8 << (sector & 3));
        let client = batch.client.index();
        let mask = usize::from(observed_merge_mask & 0x0f);
        let merge_masks = if sequence < batch.first_fill_sequence {
            &mut self.merge_audit.request_masks.older_pending_merge_masks
        } else {
            &mut self.merge_audit.request_masks.same_request_merge_masks
        };
        merge_masks[client][mask] = merge_masks[client][mask].saturating_add(1);
        if origin == MemoryOrigin::Global {
            match batch.client {
                L1LookupClient::OrdinaryGlobal if sequence < batch.first_fill_sequence => {
                    self.merge_audit.ordinary_global_older_pending_merges = self
                        .merge_audit
                        .ordinary_global_older_pending_merges
                        .saturating_add(1);
                }
                L1LookupClient::Constant if sequence < batch.first_fill_sequence => {
                    self.merge_audit.constant_global_older_pending_merges = self
                        .merge_audit
                        .constant_global_older_pending_merges
                        .saturating_add(1);
                }
                L1LookupClient::OrdinaryGlobal => {
                    self.merge_audit.ordinary_global_same_request_merges = self
                        .merge_audit
                        .ordinary_global_same_request_merges
                        .saturating_add(1);
                }
                L1LookupClient::Constant => {
                    self.merge_audit.constant_global_same_request_merges = self
                        .merge_audit
                        .constant_global_same_request_merges
                        .saturating_add(1);
                }
                L1LookupClient::AsyncCopy => {
                    self.merge_audit.async_global_same_request_merges = self
                        .merge_audit
                        .async_global_same_request_merges
                        .saturating_add(1);
                }
                _ => {}
            }
        }
    }
}

impl L1Cache {
    pub(super) fn record_older_sector_pending_merge(
        &mut self,
        origin: MemoryOrigin,
        sequence: u64,
    ) {
        if origin != MemoryOrigin::Global {
            return;
        }
        let Some(batch) = self.active_lookup_batch else {
            return;
        };
        if sequence >= batch.first_fill_sequence {
            return;
        }
        match batch.client {
            L1LookupClient::OrdinaryGlobal => {
                self.merge_audit.ordinary_global_older_sector_pending_merges = self
                    .merge_audit
                    .ordinary_global_older_sector_pending_merges
                    .saturating_add(1);
            }
            L1LookupClient::Constant => {
                self.merge_audit.constant_global_older_sector_pending_merges = self
                    .merge_audit
                    .constant_global_older_sector_pending_merges
                    .saturating_add(1);
            }
            L1LookupClient::AsyncCopy => {
                self.merge_audit.async_global_older_sector_pending_merges = self
                    .merge_audit
                    .async_global_older_sector_pending_merges
                    .saturating_add(1);
            }
            L1LookupClient::Unknown | L1LookupClient::Local => {}
        }
    }
}

impl L1Cache {
    pub(super) fn record_wavefront_mask_lookup(
        &mut self,
        origin: MemoryOrigin,
        line: i64,
        sector: u8,
    ) {
        if origin != MemoryOrigin::Global || sector >= 4 {
            return;
        }
        let bit = 1_u8 << sector;
        let resident_mask = self.sectors.get(&line).copied().unwrap_or(0);
        let pending_mask = self
            .sector_fills
            .keys()
            .filter(|(candidate, _, _)| *candidate == line)
            .fold(0_u8, |mask, (_, pending_sector, _)| {
                mask | (1_u8 << pending_sector)
            });
        // A parent line fill has not produced resident data yet. The sector
        // mask in `sectors` is allocation/request provenance until its line
        // fill retires, so do not classify it as a lookup hit.
        let resident = if self.line_fills.contains_key(&line) {
            0
        } else {
            resident_mask & !pending_mask
        };
        if resident & bit != 0 {
            self.merge_audit.wavefront_mask_lookup_hits = self
                .merge_audit
                .wavefront_mask_lookup_hits
                .saturating_add(1);
        } else {
            self.merge_audit.wavefront_mask_lookup_misses = self
                .merge_audit
                .wavefront_mask_lookup_misses
                .saturating_add(1);
        }
    }
}

impl L1Cache {
    pub const fn store_policy(&self) -> L1StorePolicy {
        self.store_policy
    }
}

impl L1Cache {
    pub fn arch(&self) -> ArchId {
        self.cfg.arch
    }
}

impl L1Cache {
    /// Stable structural projection for bounded differential diagnostics. All
    /// unordered maps are sorted before formatting; this is an observation
    /// surface, not an authority token or a production fast path.
    pub fn boundary_state(&self) -> String {
        let stamp_of = self
            .stamp_of
            .iter()
            .map(|(&line, &stamp)| (line, stamp))
            .collect::<Vec<_>>();
        let mut stamp_of = stamp_of;
        stamp_of.sort_unstable_by_key(|entry| entry.0);
        let sectors = self
            .sectors
            .iter()
            .map(|(&line, &mask)| (line, mask))
            .collect::<Vec<_>>();
        let mut sectors = sectors;
        sectors.sort_unstable_by_key(|entry| entry.0);
        let mut dirty = self
            .dirty
            .iter()
            .map(|(&line, &mask)| (line, mask))
            .collect::<Vec<_>>();
        dirty.sort_unstable_by_key(|entry| entry.0);
        let mut line_fills = self
            .line_fills
            .iter()
            .map(|(&line, &(due, seq, generation))| (line, format!("{due:?}"), seq, generation))
            .collect::<Vec<_>>();
        line_fills.sort_unstable_by_key(|entry| entry.0);
        let mut sector_fills = self
            .sector_fills
            .iter()
            .map(|(&(line, sector, generation), &(due, seq))| {
                (line, sector, generation, format!("{due:?}"), seq)
            })
            .collect::<Vec<_>>();
        sector_fills.sort_unstable_by_key(|entry| (entry.0, entry.1, entry.2));
        let mut generations = self
            .line_generations
            .iter()
            .map(|(&line, &generation)| (line, generation))
            .collect::<Vec<_>>();
        generations.sort_unstable_by_key(|entry| entry.0);
        format!(
            "cfg={:?};by_stamp={:?};stamp_of={:?};sectors={:?};dirty={:?};line_fills={:?};sector_fills={:?};generations={:?};next_stamp={:?};fill_seq={:?};active_lookup={:?};min_fill_due={:?};fill_timing={:?};counters={:?};write_observer={:?}",
            self.cfg,
            self.by_stamp,
            stamp_of,
            sectors,
            dirty,
            line_fills,
            sector_fills,
            generations,
            self.next_stamp,
            self.fill_seq,
            self.active_lookup_batch,
            self.min_fill_due,
            self.fill_timing,
            self.counters,
            self.write_lookup_observer,
        )
    }
}

impl L1Cache {
    pub(super) fn mshr_in_use(&self) -> usize {
        // A late sector child shares only the matching parent allocation's
        // MSHR slot. A zombie child from an evicted generation remains an
        // independent entry while a same-tag line is reallocated.
        self.line_fills.len()
            + self
                .sector_fills
                .keys()
                .filter(|(line, _, generation)| {
                    self.line_fills
                        .get(line)
                        .is_none_or(|&(_, _, parent_generation)| parent_generation != *generation)
                })
                .count()
    }
}

impl L1Cache {
    /// Access-local admission may raise a finite generic approximation to an
    /// independently measured floor without rewriting the cache configuration.
    /// An unlimited ablation remains unlimited.
    pub(super) fn effective_mshr_count(&self, floor: Option<NonZeroU32>) -> Option<usize> {
        self.cfg.mshr_count.map(|configured| {
            let effective = floor.map_or(configured, |floor| configured.max(floor.get()));
            effective as usize
        })
    }
}

impl L1Cache {
    pub(super) fn constant_access_may_retry(
        &self,
        sector_id: i64,
        now: u64,
        floor: Option<NonZeroU32>,
    ) -> bool {
        // A full MSHR set only requires the shadow transaction on the miss
        // path. Hits and merges have no retry loop, so cloning the complete
        // cache for them would turn transient pressure into avoidable work.
        // A fill due at `now` is a mutation boundary: apply_fills runs before
        // classification and may turn a hit into a miss through eviction, so
        // retain the shadow path until that transition is replayed exactly.
        if self.min_fill_due.is_some_and(|due| due <= now) {
            return self
                .effective_mshr_count(floor)
                .is_some_and(|capacity| self.mshr_in_use() >= capacity);
        }
        let line = sector_id >> 2;
        let sector = (sector_id & 3) as u8;
        if self.line_fills.contains_key(&line) {
            return false;
        }
        if let Some(&generation) = self.line_generations.get(&line)
            && self.sector_fills.contains_key(&(line, sector, generation))
        {
            return false;
        }
        if self.line_resident(line)
            && self
                .sectors
                .get(&line)
                .is_some_and(|mask| mask & (1_u8 << sector) != 0)
        {
            return false;
        }
        self.effective_mshr_count(floor)
            .is_some_and(|capacity| self.mshr_in_use() >= capacity)
    }
}

impl L1Cache {
    /// Whether completing this token releases an independently occupied MSHR
    /// slot. A deferred sector child shares its parent's matching generation
    /// slot until that parent returns, so a zombie child from an older
    /// same-tag allocation must not suppress retry drain.
    pub fn fill_releases_mshr(&self, fill: FillKey) -> bool {
        match fill {
            FillKey::Line { line, generation } => {
                !self
                    .sector_fills
                    .keys()
                    .any(|(child_line, _, child_generation)| {
                        *child_line == line && *child_generation == generation
                    })
            }
            FillKey::Sector {
                line, generation, ..
            } => self
                .line_fills
                .get(&line)
                .is_none_or(|&(_, _, parent_generation)| parent_generation != generation),
        }
    }
}

impl L1Cache {
    pub fn begin_lookup_batch_for_client(&mut self, client: L1LookupClient) -> L1LookupBatch {
        let batch = L1LookupBatch {
            first_fill_sequence: self.fill_seq.current(),
            pending_merges_are_hits: false,
            client,
        };
        self.active_lookup_batch = Some(batch);
        batch
    }
}

impl L1Cache {
    /// Start a cache-all LDGSTS source lookup. Independent sm_86 calibration
    /// shows that a merge into an older pending allocation is counted as an
    /// L1TEX lookup hit for this path; same-instruction coalescing is not.
    pub fn begin_async_copy_lookup_batch(&mut self) -> L1LookupBatch {
        let batch = L1LookupBatch {
            first_fill_sequence: self.fill_seq.current(),
            pending_merges_are_hits: true,
            client: L1LookupClient::AsyncCopy,
        };
        self.active_lookup_batch = Some(batch);
        batch
    }
}

impl L1Cache {
    pub(super) fn timing_name(mode: FillTiming) -> &'static str {
        match mode {
            FillTiming::Constant => "constant",
            FillTiming::Event => "event",
        }
    }
}

impl L1Cache {
    pub(super) fn preflight_latch(&self, requested: FillTiming) -> Result<(), L1AccessError> {
        if let Some(latched) = self.fill_timing
            && latched != requested
        {
            return Err(L1AccessError::FillTimingMismatch {
                latched: Self::timing_name(latched),
                requested: Self::timing_name(requested),
            });
        }
        Ok(())
    }
}

impl L1Cache {
    /// Reserve a sector child for a line allocation that is already in flight.
    ///
    /// A same-line access arriving after the original emission batch cannot be
    /// folded into the queued line request.  It therefore owns an independent
    /// sector completion token, while retaining the parent's allocation
    /// generation. The child shares the parent line allocation's MSHR slot
    /// until that parent completes. The requested bit was optimistically
    /// added by the normal line-merge classifier; remove it until this child returns so a later
    /// access cannot observe data before its own L2/DRAM event.
    pub fn reserve_pending_line_sector(
        &mut self,
        fill: FillKey,
        sector: u8,
    ) -> Result<FillKey, L1AccessError> {
        self.preflight_latch(FillTiming::Event)?;
        let FillKey::Line { line, generation } = fill else {
            return Err(L1AccessError::FillNotInFlight { fill });
        };
        if sector >= 4
            || self.line_fills.get(&line).map(|&(_, _, current)| current) != Some(generation)
        {
            return Err(L1AccessError::FillNotInFlight { fill });
        }
        let key = (line, sector, generation);
        if self.sector_fills.contains_key(&key) {
            return Ok(FillKey::Sector {
                line,
                sector,
                generation,
            });
        }
        let (sequence, next_sequence) = self.fill_seq.allocate("fill")?;
        self.fill_seq = next_sequence;
        if let Some(mask) = self.sectors.get_mut(&line) {
            *mask &= !(1_u8 << sector);
            if *mask == 0 {
                self.sectors.remove(&line);
            }
        }
        self.sector_fills.insert(key, (FillDue::Untimed, sequence));
        Ok(FillKey::Sector {
            line,
            sector,
            generation,
        })
    }
}

impl L1Cache {
    pub(super) fn apply_counter_deltas(
        &mut self,
        origin: MemoryOrigin,
        deltas: &[(L1CounterField, u64)],
    ) -> Result<(), L1AccessError> {
        self.counters = self.counters.add_fields_checked(origin, deltas)?;
        Ok(())
    }
}

impl L1Cache {
    /// Prove conservative multi-access mutation budgets without changing cache
    /// state. Counter deltas are staged in caller order so one physical unified
    /// counter snapshot covers every explicit origin lane. Recency and fill
    /// sequences consume the combined physical budget exactly once.
    pub fn preflight_mutation_budgets(
        &self,
        budgets: &[L1MutationBudget],
    ) -> Result<(), L1AccessError> {
        let mut aggregated_counter_values = [[0_u64; 19]; 2];
        for budget in budgets {
            accumulate_preflight_values(
                &mut aggregated_counter_values[budget.origin.index()],
                budget,
            )?;
        }

        let mut counters = self.counters;
        for origin in MemoryOrigin::ALL {
            let values = aggregated_counter_values[origin.index()];
            if values.iter().any(|&value| value != 0) {
                counters = counters.add_fields_checked(origin, &preflight_values_deltas(values))?;
            }
        }

        let mut recency_touches = 0u64;
        let mut new_fills = 0u64;
        for budget in budgets {
            recency_touches = recency_touches.checked_add(budget.recency_touches).ok_or(
                L1AccessError::SequenceOverflow {
                    sequence: "recency",
                    current: self.next_stamp.current(),
                },
            )?;
            new_fills =
                new_fills
                    .checked_add(budget.new_fills)
                    .ok_or(L1AccessError::SequenceOverflow {
                        sequence: "fill",
                        current: self.fill_seq.current(),
                    })?;
        }
        let in_flight =
            u64::try_from(self.mshr_in_use()).map_err(|_| L1AccessError::SequenceOverflow {
                sequence: "recency",
                current: self.next_stamp.current(),
            })?;
        recency_touches =
            recency_touches
                .checked_add(in_flight)
                .ok_or(L1AccessError::SequenceOverflow {
                    sequence: "recency",
                    current: self.next_stamp.current(),
                })?;
        self.next_stamp.preflight("recency", recency_touches)?;
        self.fill_seq.preflight("fill", new_fills)?;
        // FillKey carries a compact u32 generation even though the internal
        // ordering sequence remains u64. A u64 sequence preflight alone is
        // insufficient at 2^32: a later `apply_fills(now)` may mutate this
        // cache before `access_core` discovers that the generation cannot be
        // represented. Reject (and let public wrappers use their shadow
        // transaction) before any due-fill mutation can leak through.
        self.preflight_fill_generation_headroom(new_fills)?;
        Ok(())
    }
}

impl L1Cache {
    /// Conservative, allocation-free ready-time upper bound for an ordered batch
    /// of constant-mode sector accesses. Starting after the latest existing fill
    /// and charging one miss arm per selected sector covers fills created by an
    /// earlier access in the same caller transaction. `None` asks the caller to
    /// classify the batch exactly on a shadow cache.
    pub fn constant_batch_ready_upper_bound(&self, now: u64, sectors: u64) -> Option<u64> {
        if self.preflight_latch(FillTiming::Constant).is_err() {
            return None;
        }
        let hit_ready = Self::checked_time_add("hit visibility", now, self.cfg.hit_latency).ok()?;
        let miss_ready =
            Self::checked_time_add("miss visibility", now, self.cfg.miss_latency).ok()?;
        if sectors == 0 {
            return Some(hit_ready.max(miss_ready));
        }
        let start = self
            .line_fills
            .values()
            .map(|&(due, ..)| due)
            .chain(self.sector_fills.values().map(|&(due, ..)| due))
            .filter_map(FillDue::timed)
            .max()
            .unwrap_or(now)
            .max(now);
        self.cfg
            .miss_latency
            .checked_mul(sectors)
            .and_then(|delta| start.checked_add(delta))
            .map(|batch_miss_ready| batch_miss_ready.max(hit_ready))
    }
}

impl L1Cache {
    pub(super) fn remove_from_set_lru(&mut self, line: i64) {
        let Some(index) = self
            .set_locations
            .get(&line)
            .copied()
            .or_else(|| self.set_index(line))
        else {
            return;
        };
        if let Some(sets) = self.set_lru.as_mut()
            && let Some(set) = sets.get_mut(index)
            && let Some(position) = set.iter().position(|candidate| *candidate == line)
        {
            set.remove(position);
        }
        self.set_locations.remove(&line);
    }
}

impl L1Cache {
    pub(super) fn touch_mru(&mut self, line: i64) -> Result<(), L1AccessError> {
        let (stamp, next_stamp) = self.next_stamp.allocate("recency")?;
        if let Some(old) = self.stamp_of.remove(&line) {
            self.by_stamp.remove(&old);
        }
        self.next_stamp = next_stamp;
        self.by_stamp.insert(stamp, line);
        self.stamp_of.insert(line, stamp);
        if let Some(index) = self.set_index(line)
            && let Some(sets) = self.set_lru.as_mut()
            && let Some(set) = sets.get_mut(index)
        {
            if let Some(position) = set.iter().position(|candidate| *candidate == line) {
                set.remove(position);
            }
            set.push_back(line);
            self.set_locations.insert(line, index);
        }
        Ok(())
    }
}

impl L1Cache {
    pub(super) fn evict_beyond_capacity(&mut self) -> Result<(), L1AccessError> {
        self.preflight_dirty_eviction(None)?;
        if let Some(ways) = self.cfg.ways {
            let limit = usize::try_from(ways).unwrap_or(usize::MAX);
            let victims = self
                .set_lru
                .as_ref()
                .map(|sets| {
                    sets.iter()
                        .flat_map(|set| set.iter().take(set.len().saturating_sub(limit)).copied())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            for line in victims {
                self.remove_resident_line(line);
            }
        }
        while self.by_stamp.len() as u64 > self.cfg.line_capacity {
            let Some((&stamp, &line)) = self.by_stamp.first_key_value() else {
                break;
            };
            debug_assert_eq!(self.stamp_of.get(&line), Some(&stamp));
            self.remove_resident_line(line);
        }
        Ok(())
    }
}

impl L1Cache {
    /// Mark resident sectors dirty for an explicit write-policy experiment.
    /// The default native replay path never calls this method; stores retain
    /// their existing write-through/no-allocate hypothesis until a validated
    /// protocol supplies the call and a completion boundary.
    pub fn mark_dirty(&mut self, line: i64, sector_mask: u8) -> Result<(), L1AccessError> {
        Self::validate_sector_mask(line, sector_mask)?;
        let valid_mask = self.sectors.get(&line).copied().unwrap_or(0);
        if !self.line_resident(line) {
            return Err(L1AccessError::DirtyLineNotResident { line });
        }
        if sector_mask & !valid_mask != 0 {
            return Err(L1AccessError::DirtyMaskInvalid {
                line,
                dirty_mask: sector_mask,
                valid_mask,
            });
        }
        *self.dirty.entry(line).or_default() |= sector_mask;
        // A mask-only mutation carries no replacement bytes. Never retain an
        // older complete payload and present it as this store's value.
        for sector in 0..4_u8 {
            if sector_mask & (1_u8 << sector) != 0 {
                self.dirty_payloads.remove(&(line, sector));
            }
        }
        Ok(())
    }
}

impl L1Cache {
    /// Preflight a complete payload batch without mutating cache state.
    /// Partial records are coalesced by allocation identity and sequence;
    /// every resulting sector must be resident and fully known. A mask-only
    /// dirty sector is refused because a later sidecar cannot prove it is the
    /// same store generation.
    pub fn preview_dirty_payload_admission(
        &self,
        payloads: &[SectorPayloadV1],
    ) -> Result<L1DirtyPayloadAdmissionPlan, L1AccessError> {
        let complete = crate::coalesce_payloads(payloads.to_vec())?;
        let mut expected = Vec::with_capacity(complete.len());
        for payload in &complete {
            let line = payload.line_id();
            let sector = payload.sector();
            if !self.line_resident(line) {
                return Err(L1AccessError::DirtyLineNotResident { line });
            }
            let bit = 1_u8 << sector;
            let valid_mask = self.sectors.get(&line).copied().unwrap_or(0);
            if valid_mask & bit == 0 {
                return Err(L1AccessError::DirtyMaskInvalid {
                    line,
                    dirty_mask: bit,
                    valid_mask,
                });
            }
            let current = self.dirty_payloads.get(&(line, sector)).copied();
            let dirty_mask = self.dirty.get(&line).copied().unwrap_or(0);
            if dirty_mask & bit != 0 && current.is_none() {
                return Err(L1AccessError::DirtyPayloadMissing { line, sector });
            }
            if let Some(previous) = current {
                previous.merge(*payload)?;
            }
            expected.push(L1DirtyPayloadExpectation {
                line,
                sector,
                current,
                valid_mask,
                dirty_mask,
                generation: self.line_generations.get(&line).copied(),
            });
        }
        Ok(L1DirtyPayloadAdmissionPlan {
            payloads: complete.into_boxed_slice(),
            expected: expected.into_boxed_slice(),
        })
    }
}

impl L1Cache {
    /// Attach complete bytes to already-dirty resident sectors after the SM
    /// has checked the exact dynamic writer and its accept time. Unlike the
    /// generic admission API this requires, rather than invents, dirty state.
    /// It cannot validate a dynamic instruction itself; that belongs to the
    /// issue actor. Any failure leaves the whole cache unchanged.
    pub fn attach_observed_store_payloads(
        &mut self,
        payloads: &[SectorPayloadV1],
        witnesses: &[L1PayloadResidencyWitness],
    ) -> Result<(), L1AccessError> {
        self.validate_payload_residency_witnesses(payloads, witnesses)?;
        let complete = crate::coalesce_payloads(payloads.to_vec())?;
        for payload in &complete {
            if self.dirty.get(&payload.line_id()).copied().unwrap_or(0) & (1_u8 << payload.sector())
                == 0
            {
                return Err(L1AccessError::DirtyPayloadResidencyWitnessMismatch {
                    line: payload.line_id(),
                    field: "observed store sector is not dirty",
                });
            }
        }
        let mut next = self.clone();
        for payload in complete {
            next.mark_dirty_payload(payload)?;
        }
        *self = next;
        Ok(())
    }
}

impl L1Cache {
    pub fn preview_dirty_payloads(&self) -> Result<Vec<SectorPayloadV1>, L1AccessError> {
        let mut payloads = Vec::new();
        for (&line, &mask) in &self.dirty {
            for sector in 0..4_u8 {
                if mask & (1 << sector) == 0 {
                    continue;
                }
                let Some(payload) = self.dirty_payloads.get(&(line, sector)).copied() else {
                    return Err(L1AccessError::DirtyPayloadMissing { line, sector });
                };
                if !payload.is_fully_known() {
                    return Err(L1AccessError::DirtyPayloadMissing { line, sector });
                }
                payloads.push(payload);
            }
        }
        payloads.sort_by_key(|payload| (payload.line_id(), payload.sector()));
        Ok(payloads)
    }
}

impl L1Cache {
    /// Return the current dirty-line gauge. This is a state observation, not
    /// completed traffic; callers must use the transactional flush plan below
    /// before claiming a writeback completion.
    pub fn dirty_residency(&self) -> L1DirtyResidency {
        L1DirtyResidency {
            lines: u64::try_from(self.dirty.len()).unwrap_or(u64::MAX),
            sectors: self
                .dirty
                .values()
                .map(|mask| u64::from(mask.count_ones()))
                .sum(),
        }
    }
}

impl L1Cache {
    /// Commit a previously previewed flush after lower-level service has
    /// accepted it. Resident tags remain present but become clean.
    pub fn commit_dirty_flush(&mut self, plan: L1DirtyFlushPlan) -> Result<(), L1AccessError> {
        for line in plan.lines() {
            let current = self.dirty.get(&line.line_id).copied().unwrap_or(0);
            if current != line.dirty_mask {
                return Err(L1AccessError::DirtyFlushStale {
                    line: line.line_id,
                    expected: line.dirty_mask,
                    current,
                });
            }
        }
        for line in plan.lines() {
            self.dirty.remove(&line.line_id);
            for sector in 0..4_u8 {
                if line.dirty_mask & (1 << sector) != 0 {
                    self.dirty_payloads.remove(&(line.line_id, sector));
                }
            }
        }
        Ok(())
    }
}

impl L1Cache {
    /// Explicit cache-control invalidation for a complete L1. This models
    /// only the admitted `CCTL.IVALL`-style operation: pending fills and dirty
    /// sectors refuse because the SM event layer must first account for their
    /// response/writeback ordering.
    pub fn preflight_invalidate_all(&self) -> Result<u64, L1AccessError> {
        if !self.line_fills.is_empty() || !self.sector_fills.is_empty() {
            return Err(L1AccessError::InvalidationAllPendingFill);
        }
        if let Some((&line, &dirty_mask)) = self.dirty.iter().next() {
            return Err(L1AccessError::DirtyResidencyBlocksInvalidation { line, dirty_mask });
        }
        Ok(self
            .sectors
            .values()
            .map(|mask| u64::from(mask.count_ones()))
            .sum())
    }
}

impl L1Cache {
    pub fn invalidate_all(&mut self) -> Result<u64, L1AccessError> {
        let sectors = self.preflight_invalidate_all()?;
        self.by_stamp.clear();
        self.stamp_of.clear();
        if let Some(sets) = self.set_lru.as_mut() {
            for set in sets {
                set.clear();
            }
        }
        self.set_locations.clear();
        self.sectors.clear();
        Ok(sectors)
    }
}

impl L1Cache {
    /// Pending line/sector fills prevent invalidation even when they have no
    /// local timestamp. Event-mode callers must wait for response events,
    /// never mistake the absence of a timed deadline for an empty cache.
    pub fn has_pending_fills(&self) -> bool {
        !self.line_fills.is_empty() || !self.sector_fills.is_empty()
    }
}

impl L1Cache {
    /// Replay due fills in CHRONOLOGICAL order; each fill's effect is
    /// decided by the cache state AT ITS OWN FILL TIME (line-before-sector
    /// only at exactly equal timestamps; same-kind ties in issue order).
    ///
    /// Integer-time note (deliberate divergence from the float-timed
    /// Python walker): the sim's clock is integer cycles, so fills whose
    /// float times would differ inside one cycle land on the same
    /// integer timestamp and resolve by the (kind, issue-order) tie rule
    /// instead of sub-cycle chronology. Bit-parity with Python holds
    /// whenever both sides run integer latencies (proven by the gate's
    /// differential); fractional-latency sub-cycle ordering is below the
    /// engine's resolution by design.
    pub(super) fn apply_fills(&mut self, now: u64) -> Result<(), L1AccessError> {
        let Some(minimum) = self.min_fill_due else {
            return Ok(());
        };
        if now < minimum {
            return Ok(());
        }
        let mut due: Vec<(u64, u8, u64, i64, u8, u32)> = Vec::new();
        for (&line, &(fill_due, seq, generation)) in &self.line_fills {
            if let FillDue::Timed(time) = fill_due
                && time <= now
            {
                due.push((time, 0, seq, line, 0, generation));
            }
        }
        for (&(line, sector, generation), &(fill_due, seq)) in &self.sector_fills {
            if let FillDue::Timed(time) = fill_due
                && time <= now
            {
                due.push((time, 1, seq, line, sector, generation));
            }
        }
        due.sort_unstable();
        let due_count = u64::try_from(due.len()).map_err(|_| L1AccessError::SequenceOverflow {
            sequence: "recency",
            current: self.next_stamp.current(),
        })?;
        if self.next_stamp.preflight("recency", due_count).is_err() {
            // Some due sector fills can be dropped after an earlier fill evicts
            // their line. Near exhaustion, classify those touches exactly on a
            // shadow cache and publish the whole transition only on success.
            let mut shadow = self.clone();
            shadow.commit_due_fills(due)?;
            *self = shadow;
            return Ok(());
        }
        self.commit_due_fills(due)
    }
}

impl L1Cache {
    pub(super) fn access_single(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
        readiness: Readiness,
        mshr_floor: Option<NonZeroU32>,
    ) -> Result<SingleOutcome, L1AccessError> {
        self.apply_fills(now)?;
        match self.access_core(
            origin,
            sector_id,
            FillRequest::Constant { now, readiness },
            mshr_floor,
        )? {
            CoreAccess::Hit { ready } => Ok(SingleOutcome::Done(AccessOutcome::Hit, ready)),
            CoreAccess::LineMerge { pending, .. } | CoreAccess::SectorMerge { pending, .. } => {
                let pending = pending.timed().ok_or(L1AccessError::FillTimingMismatch {
                    latched: Self::timing_name(FillTiming::Event),
                    requested: Self::timing_name(FillTiming::Constant),
                })?;
                Ok(SingleOutcome::Done(AccessOutcome::Merge, Some(pending)))
            }
            CoreAccess::Stalled => Ok(SingleOutcome::Stall {
                retry_at: self.earliest_fill()?,
            }),
            CoreAccess::MissSectorResident { pending, .. } => {
                let ready = pending.timed().ok_or(L1AccessError::FillTimingMismatch {
                    latched: Self::timing_name(FillTiming::Event),
                    requested: Self::timing_name(FillTiming::Constant),
                })?;
                Ok(SingleOutcome::Done(
                    AccessOutcome::MissSectorResident,
                    Some(ready),
                ))
            }
            CoreAccess::MissLineAlloc { pending, .. } => {
                let ready = pending.timed().ok_or(L1AccessError::FillTimingMismatch {
                    latched: Self::timing_name(FillTiming::Event),
                    requested: Self::timing_name(FillTiming::Constant),
                })?;
                Ok(SingleOutcome::Done(
                    AccessOutcome::MissLineAlloc,
                    Some(ready),
                ))
            }
        }
    }
}

impl L1Cache {
    pub(super) fn access_eligible_constant_impl(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
        mshr_floor: Option<NonZeroU32>,
    ) -> Result<(AccessOutcome, u64), L1AccessError> {
        self.commit_latch(FillTiming::Constant);
        let mut t = now;
        loop {
            match self.access_single(origin, sector_id, t, Readiness::Observed, mshr_floor)? {
                SingleOutcome::Done(outcome, Some(ready)) => return Ok((outcome, ready)),
                SingleOutcome::Done(_, None) => {
                    unreachable!("observed constant access omitted its ready time")
                }
                SingleOutcome::Stall { retry_at } => {
                    t = retry_at.max(Self::checked_time_add("stall retry", t, 1)?);
                }
            }
        }
    }
}

impl L1Cache {
    pub(super) fn access_eligible_constant_unobserved_impl(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
        mshr_floor: Option<NonZeroU32>,
    ) -> Result<AccessOutcome, L1AccessError> {
        self.commit_latch(FillTiming::Constant);
        let mut t = now;
        loop {
            match self.access_single(origin, sector_id, t, Readiness::Unobserved, mshr_floor)? {
                SingleOutcome::Done(outcome, _) => return Ok(outcome),
                SingleOutcome::Stall { retry_at } => {
                    t = retry_at.max(Self::checked_time_add("stall retry", t, 1)?);
                }
            }
        }
    }
}

impl L1Cache {
    /// One modeled (manifest-eligible) sector access at time `now`.
    /// MSHR stalls retry internally at the earliest fill time (the ported
    /// walker's retry contract). Returns the outcome and the data-ready
    /// time.
    pub fn access_eligible(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
    ) -> Result<(AccessOutcome, u64), L1AccessError> {
        self.preflight_latch(FillTiming::Constant)?;
        let budget = L1MutationBudget {
            eligible_sectors: 1,
            recency_touches: 1,
            new_fills: 1,
            ..L1MutationBudget::for_origin(origin)
        };
        let fast_path_safe = !self.constant_access_may_retry(sector_id, now, None)
            && self.preflight_mutation_budget(budget).is_ok()
            && self.preflight_constant_access_timing(now).is_ok();
        if !fast_path_safe {
            let mut shadow = self.clone();
            let outcome = shadow.access_eligible_constant_impl(origin, sector_id, now, None)?;
            *self = shadow;
            return Ok(outcome);
        }
        self.access_eligible_constant_impl(origin, sector_id, now, None)
    }
}

impl L1Cache {
    /// Constant-mode eligible access with a caller-characterized lower bound on
    /// finite miss admission. The floor applies only to this access; ordinary
    /// accesses continue to observe [`L1Config::mshr_count`].
    pub fn access_eligible_with_mshr_floor(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
        mshr_floor: NonZeroU32,
    ) -> Result<(AccessOutcome, u64), L1AccessError> {
        self.preflight_latch(FillTiming::Constant)?;
        let budget = L1MutationBudget {
            eligible_sectors: 1,
            recency_touches: 1,
            new_fills: 1,
            ..L1MutationBudget::for_origin(origin)
        };
        let fast_path_safe = !self.constant_access_may_retry(sector_id, now, Some(mshr_floor))
            && self.preflight_mutation_budget(budget).is_ok()
            && self.preflight_constant_access_timing(now).is_ok();
        if !fast_path_safe {
            let mut shadow = self.clone();
            let outcome =
                shadow.access_eligible_constant_impl(origin, sector_id, now, Some(mshr_floor))?;
            *self = shadow;
            return Ok(outcome);
        }
        self.access_eligible_constant_impl(origin, sector_id, now, Some(mshr_floor))
    }
}

impl L1Cache {
    /// Mutate constant-mode cache state for one eligible sector without
    /// observing producer-ready time. Fresh misses still compute their fill due
    /// time; hits and merges skip visibility arithmetic that no dependence
    /// barrier consumes.
    pub fn access_eligible_unobserved(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
    ) -> Result<AccessOutcome, L1AccessError> {
        self.preflight_latch(FillTiming::Constant)?;
        let budget = L1MutationBudget {
            eligible_sectors: 1,
            recency_touches: 1,
            new_fills: 1,
            ..L1MutationBudget::for_origin(origin)
        };
        let fast_path_safe = !self.constant_access_may_retry(sector_id, now, None)
            && self.preflight_mutation_budget(budget).is_ok()
            && self.preflight_constant_access_timing(now).is_ok();
        if !fast_path_safe {
            let mut shadow = self.clone();
            let outcome =
                shadow.access_eligible_constant_unobserved_impl(origin, sector_id, now, None)?;
            *self = shadow;
            return Ok(outcome);
        }
        self.access_eligible_constant_unobserved_impl(origin, sector_id, now, None)
    }
}

impl L1Cache {
    /// Unobserved counterpart to [`Self::access_eligible_with_mshr_floor`].
    pub fn access_eligible_unobserved_with_mshr_floor(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
        mshr_floor: NonZeroU32,
    ) -> Result<AccessOutcome, L1AccessError> {
        self.preflight_latch(FillTiming::Constant)?;
        let budget = L1MutationBudget {
            eligible_sectors: 1,
            recency_touches: 1,
            new_fills: 1,
            ..L1MutationBudget::for_origin(origin)
        };
        let fast_path_safe = !self.constant_access_may_retry(sector_id, now, Some(mshr_floor))
            && self.preflight_mutation_budget(budget).is_ok()
            && self.preflight_constant_access_timing(now).is_ok();
        if !fast_path_safe {
            let mut shadow = self.clone();
            let outcome = shadow.access_eligible_constant_unobserved_impl(
                origin,
                sector_id,
                now,
                Some(mshr_floor),
            )?;
            *self = shadow;
            return Ok(outcome);
        }
        self.access_eligible_constant_unobserved_impl(origin, sector_id, now, Some(mshr_floor))
    }
}

impl L1Cache {
    pub(super) fn access_eligible_event_core(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
        readiness: Readiness,
        mshr_floor: Option<NonZeroU32>,
    ) -> Result<CoreAccess, L1AccessError> {
        // Delay the mode latch until the ladder succeeds.  In particular, an
        // observed hit at `u64::MAX` can fail its ready-time addition; that
        // typed error must not leave a previously fresh cache permanently
        // latched to EVENT.  `access_core` validates all fallible arithmetic
        // before its corresponding mutation, so committing here is the
        // transaction boundary for the mode bit.
        let outcome = self.access_core(
            origin,
            sector_id,
            FillRequest::Event { now, readiness },
            mshr_floor,
        )?;
        self.commit_latch(FillTiming::Event);
        Ok(outcome)
    }
}

impl L1Cache {
    pub(super) fn access_eligible_event_impl(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
        mshr_floor: Option<NonZeroU32>,
    ) -> Result<EvAccess, L1AccessError> {
        let line = sector_id >> 2;
        let sector = (sector_id & 3) as u8;
        Ok(
            match self.access_eligible_event_core(
                origin,
                sector_id,
                now,
                Readiness::Observed,
                mshr_floor,
            )? {
                CoreAccess::Hit { ready: Some(ready) } => EvAccess::Hit { ready },
                CoreAccess::Hit { ready: None } => {
                    unreachable!("observed event access omitted its ready time")
                }
                CoreAccess::LineMerge { generation, .. } => EvAccess::MergedPending {
                    fill: FillKey::Line { line, generation },
                },
                CoreAccess::SectorMerge { generation, .. } => EvAccess::MergedPending {
                    fill: FillKey::Sector {
                        line,
                        sector,
                        generation,
                    },
                },
                CoreAccess::Stalled => EvAccess::Stalled,
                CoreAccess::MissSectorResident { generation, .. } => EvAccess::MissRequest {
                    fill: FillKey::Sector {
                        line,
                        sector,
                        generation,
                    },
                },
                CoreAccess::MissLineAlloc { generation, .. } => EvAccess::MissRequest {
                    fill: FillKey::Line { line, generation },
                },
            },
        )
    }
}

impl L1Cache {
    /// One modeled sector access in EVENT mode (shared-L2 integration):
    /// the fill time is decided by the next level, so misses register an
    /// explicitly UNTIMED in-flight entry. The lazy applier ignores it;
    /// [`Self::complete_fill`] installs it at the response event.
    /// Hit/miss/merge accounting matches [`Self::access_eligible`];
    /// `stall_events` does NOT — the constant
    /// mode's internal retry-at-earliest-fill always succeeds (one stall
    /// per stalled access), while event-mode parking re-counts a stall
    /// on every failed re-attempt after a completion, so `stall_events`
    /// reads "stall ATTEMPTS" in this mode (deliberate: the retry
    /// discipline is the caller's, and no emitted metric consumes the
    /// counter yet).
    ///
    /// The two modes must not be mixed on one cache instance: constant-
    /// mode entries carry real times the event mode would complete out
    /// of order.
    pub fn access_eligible_ev(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
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
            let outcome = shadow.access_eligible_event_impl(origin, sector_id, now, None)?;
            *self = shadow;
            return Ok(outcome);
        }
        self.access_eligible_event_impl(origin, sector_id, now, None)
    }
}

impl L1Cache {
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
}

impl L1Cache {
    pub(super) fn probe_write_impl(
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
}

impl L1Cache {
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
}

impl L1Cache {
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
}

impl L1Cache {
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
}

impl L1Cache {
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
