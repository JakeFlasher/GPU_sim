//! L1 cache state, construction, lookup, fill, and dirty-residency operations; mutates cache state at event timestamps.

use super::*;

#[derive(Debug, Clone)]
pub struct L1Cache {
    pub(super) cfg: L1Config,
    pub(super) store_policy: L1StorePolicy,
    /// Fully-associative LRU line pool: monotonic recency stamps.
    pub(super) by_stamp: BTreeMap<u64, i64>,
    pub(super) stamp_of: HashMap<i64, u64>,
    pub(super) next_stamp: NextSequence,
    /// Per-line fetched-sector bitmask (bit = sector index in line, 0..4).
    /// Also accumulates for in-flight line allocations (the fill services
    /// every merged sector).
    pub(super) sectors: HashMap<i64, u8>,
    /// Explicit opt-in dirty-sector state. The default replay policy does not
    /// mark stores dirty; the resident-hit experiment uses this map and the
    /// transactional flush API below.
    pub(super) dirty: HashMap<i64, u8>,
    /// Optional known-byte payloads for the strict value-bearing flush lane.
    /// Ordinary replay remains mask-only until a producer supplies these.
    pub(super) dirty_payloads: HashMap<(i64, u8), SectorPayloadV1>,
    /// In-flight fills: value = (due state, issue_seq) — the seq keeps
    /// same-timestamp same-kind timed fills in issue order like the Python
    /// stable sort. Event fills remain explicitly untimed. Sector fills are
    /// keyed by line, sector, and allocation generation so an old response
    /// remains representable while a same-tag reallocation has already
    /// issued a new request for that sector.
    pub(super) line_fills: HashMap<i64, (FillDue, u64, u32)>,
    pub(super) sector_fills: HashMap<(i64, u8, u32), (FillDue, u64)>,
    /// Allocation-generation witnesses for event-mode fills.  A sector fill
    /// can outlive the resident line that spawned it; retaining the line's
    /// allocation sequence lets completion reject an old response after the
    /// same tag has been evicted and reallocated (the L2 model already uses
    /// the analogous generation guard).
    pub(super) line_generations: HashMap<i64, u32>,
    pub(super) fill_seq: NextSequence,
    /// Active logical request boundary for the lookup observer only. This does
    /// not participate in tags, MSHR ownership, replacement, or timing.
    pub(super) active_lookup_batch: Option<L1LookupBatch>,
    pub(super) merge_audit: L1MergeAudit,
    /// Exact earliest timed in-flight fill. `None` means there is no timed
    /// fill (empty or event-driven), while `Some(u64::MAX)` is a real due
    /// cycle. `apply_fills` early-outs when `now` has not reached it.
    pub(super) min_fill_due: Option<u64>,
    /// Latched fill-timing discipline (see [`FillTiming`]).
    pub(super) fill_timing: Option<FillTiming>,
    /// Lazily allocated observer-only write tag window.  Keeping it boxed and
    /// optional leaves cold/read-only replays on the compact hot path.
    pub(super) write_lookup_observer: Option<Box<L1WriteLookupObserver>>,
    /// Optional effective set-local LRU state. The global recency maps remain
    /// the canonical typed boundary; this index is an executable projection
    /// used only by the card-backed modulo scenario.
    pub(super) set_lru: Option<Vec<VecDeque<i64>>>,
    /// Actual set chosen by the pressure-bounded unknown-hash scenario.
    /// Fully associative and fixed-index modes leave this empty.
    pub(super) set_locations: HashMap<i64, usize>,
    pub counters: L1Counters,
}

impl L1Cache {
    pub fn new(cfg: L1Config) -> Result<Self, L1ConfigError> {
        if !matches!(cfg.arch, ArchId::Sm86 | ArchId::Sm120) {
            return Err(L1ConfigError::ArchitectureUnsupported { arch: cfg.arch });
        }
        if cfg.line_capacity == 0 {
            return Err(L1ConfigError::CapacityZero);
        }
        if cfg.mshr_count == Some(0) {
            return Err(L1ConfigError::MshrZero);
        }
        if cfg.hit_latency >= cfg.miss_latency {
            return Err(L1ConfigError::LatencyOrderInverted {
                hit: cfg.hit_latency,
                miss: cfg.miss_latency,
            });
        }
        if matches!(
            cfg.set_index,
            L1SetIndex::Modulo | L1SetIndex::XorFold | L1SetIndex::ModuloXorFallback
        ) && (cfg.set_count.is_none_or(|count| count == 0)
            || cfg.ways.is_none_or(|ways| ways == 0))
        {
            return Err(L1ConfigError::SetGeometryInvalid);
        }
        if cfg.set_index == L1SetIndex::FullyAssociative
            && (cfg.set_count.is_some() || cfg.ways.is_some())
        {
            return Err(L1ConfigError::SetGeometryInvalid);
        }
        let arch = cfg.arch;
        Ok(Self {
            cfg,
            store_policy: L1StorePolicy::WriteThroughNoAllocate,
            by_stamp: BTreeMap::new(),
            stamp_of: HashMap::new(),
            next_stamp: NextSequence::Available(0),
            sectors: HashMap::new(),
            dirty: HashMap::new(),
            dirty_payloads: HashMap::new(),
            line_fills: HashMap::new(),
            sector_fills: HashMap::new(),
            line_generations: HashMap::new(),
            fill_seq: NextSequence::Available(0),
            active_lookup_batch: None,
            merge_audit: L1MergeAudit::default(),
            min_fill_due: None,
            fill_timing: None,
            write_lookup_observer: None,
            set_lru: match (cfg.set_index, cfg.set_count) {
                (
                    L1SetIndex::Modulo | L1SetIndex::XorFold | L1SetIndex::ModuloXorFallback,
                    Some(count),
                ) => Some((0..count).map(|_| VecDeque::new()).collect::<Vec<_>>()),
                _ => None,
            },
            set_locations: HashMap::new(),
            counters: L1Counters::for_arch(arch),
        })
    }
}

impl L1Cache {
    pub(super) fn record_request_masks(&mut self, client: L1LookupClient, sectors: &[i64]) {
        let client_index = client.index();
        let mut line_masks: HashMap<i64, u8> = HashMap::new();
        let mut overflow = 0_u64;
        for &sector_id in sectors {
            let line = sector_id >> 2;
            let bit = 1_u8 << ((sector_id & 3) as u8);
            if let Some(mask) = line_masks.get_mut(&line) {
                *mask |= bit;
                continue;
            }
            if line_masks.len() >= L1RequestMaskAudit::MAX_REQUEST_LINES {
                overflow = overflow.saturating_add(1);
                continue;
            }
            line_masks.insert(line, bit);
        }
        let audit = &mut self.merge_audit.request_masks;
        audit.request_count[client_index] = audit.request_count[client_index].saturating_add(1);
        audit.request_mask_overflow[client_index] =
            audit.request_mask_overflow[client_index].saturating_add(u64::from(overflow != 0));
        audit.request_line_count[client_index] =
            audit.request_line_count[client_index].saturating_add(line_masks.len() as u64);
        for mask in line_masks.into_values() {
            let slot = &mut audit.request_masks[client_index][usize::from(mask & 0x0f)];
            *slot = slot.saturating_add(1);
        }
    }
}

impl L1Cache {
    pub(super) fn record_client_hit(&mut self, origin: MemoryOrigin) {
        if origin != MemoryOrigin::Global {
            return;
        }
        let Some(client) = self.active_lookup_batch.map(|batch| batch.client) else {
            return;
        };
        match client {
            L1LookupClient::OrdinaryGlobal => {
                self.merge_audit.ordinary_global_hits =
                    self.merge_audit.ordinary_global_hits.saturating_add(1);
            }
            L1LookupClient::Constant => {
                self.merge_audit.constant_global_hits =
                    self.merge_audit.constant_global_hits.saturating_add(1);
            }
            L1LookupClient::AsyncCopy => {
                self.merge_audit.async_global_hits =
                    self.merge_audit.async_global_hits.saturating_add(1);
            }
            L1LookupClient::Unknown | L1LookupClient::Local => {}
        }
    }
}

impl L1Cache {
    pub(super) fn record_unrequested_subline_miss(
        &mut self,
        origin: MemoryOrigin,
        line: i64,
        sector: u8,
    ) {
        if origin != MemoryOrigin::Global || sector >= 4 {
            return;
        }
        let bit = 1_u8 << sector;
        let line_mask = self.sectors.get(&line).copied().unwrap_or(0);
        let sector_pending = self
            .sector_fills
            .keys()
            .filter(|(candidate, _, _)| *candidate == line)
            .fold(0_u8, |mask, (_, pending_sector, _)| {
                mask | (1_u8 << pending_sector)
            });
        // A first touch of a cold line is an ordinary line-allocation miss,
        // not the disjoint-subline phenomenon. Only audit a sector when the
        // line is already resident or has an in-flight parent/child fill.
        if line_mask == 0
            && sector_pending == 0
            && !self.line_fills.contains_key(&line)
            && !self.line_resident(line)
        {
            return;
        }
        // A parent line fill owns every sector currently admitted into its
        // line mask.  Once the parent is complete, only explicit child sector
        // fills remain pending; all other bits are resident data.
        let known = line_mask | sector_pending;
        if known & bit == 0 {
            self.merge_audit.unrequested_subline_miss_sectors = self
                .merge_audit
                .unrequested_subline_miss_sectors
                .saturating_add(1);
        }
    }
}

impl L1Cache {
    pub(super) fn record_constant_request(&mut self, origin: MemoryOrigin) {
        if origin == MemoryOrigin::Global
            && self
                .active_lookup_batch
                .is_some_and(|batch| batch.client == L1LookupClient::Constant)
        {
            self.merge_audit.constant_global_requests =
                self.merge_audit.constant_global_requests.saturating_add(1);
        }
    }
}

impl L1Cache {
    /// Select the explicit resident-hit writeback experiment. This does not
    /// alter cache geometry or fill timing and is intentionally opt-in.
    pub fn set_store_policy(&mut self, policy: L1StorePolicy) {
        self.store_policy = policy;
    }
}

impl L1Cache {
    /// Return the model-local allocation generation and resident-sector mask
    /// for one line. This is the only residency state the cache model can
    /// supply to a value-bearing admission witness; it is deliberately not a
    /// physical allocation/owner identity and must be joined with an
    /// independently observed producer allocation before payload mutation.
    pub fn resident_payload_state(&self, line: i64) -> Option<(u32, u8)> {
        let generation = self.line_generations.get(&line).copied()?;
        let resident_sector_mask = self.sectors.get(&line).copied().unwrap_or(0);
        self.line_resident(line)
            .then_some((generation, resident_sector_mask))
    }
}

impl L1Cache {
    /// Deterministic typed projection of the modeled read-state L1. The
    /// projection carries fill generations and pending maps directly; no
    /// debug formatting participates in the state contract.
    pub fn typed_boundary_state(&self) -> L1BoundaryState {
        let mut line_order = self
            .by_stamp
            .iter()
            .map(|(&stamp, &line)| (stamp, line))
            .collect::<Vec<_>>();
        line_order.sort_unstable();
        let mut stamp_of = self
            .stamp_of
            .iter()
            .map(|(&line, &stamp)| (line, stamp))
            .collect::<Vec<_>>();
        stamp_of.sort_unstable();
        let mut sectors = self
            .sectors
            .iter()
            .map(|(&line, &mask)| (line, mask))
            .collect::<Vec<_>>();
        sectors.sort_unstable();
        let due = |due: FillDue| match due {
            FillDue::Timed(cycle) => L1FillDueState::Timed { cycle },
            FillDue::Untimed => L1FillDueState::Untimed,
        };
        let mut line_fills = self
            .line_fills
            .iter()
            .map(
                |(&line, &(fill_due, sequence, generation))| L1LineFillState {
                    line,
                    due: due(fill_due),
                    sequence,
                    generation,
                },
            )
            .collect::<Vec<_>>();
        line_fills.sort_by_key(|fill| (fill.line, fill.generation, fill.sequence));
        let mut sector_fills = self
            .sector_fills
            .iter()
            .map(
                |(&(line, sector, generation), &(fill_due, sequence))| L1SectorFillState {
                    line,
                    sector,
                    generation,
                    due: due(fill_due),
                    sequence,
                },
            )
            .collect::<Vec<_>>();
        sector_fills.sort_by_key(|fill| (fill.line, fill.sector, fill.generation, fill.sequence));
        let mut line_generations = self
            .line_generations
            .iter()
            .map(|(&line, &generation)| (line, generation))
            .collect::<Vec<_>>();
        line_generations.sort_unstable();
        let next_stamp = match self.next_stamp {
            NextSequence::Available(value) => Some(value),
            NextSequence::Exhausted => None,
        };
        let fill_sequence = match self.fill_seq {
            NextSequence::Available(value) => Some(value),
            NextSequence::Exhausted => None,
        };
        let active_lookup_batch = self
            .active_lookup_batch
            .map(|batch| (batch.first_fill_sequence, batch.pending_merges_are_hits));
        let fill_timing = self.fill_timing.map(|timing| match timing {
            FillTiming::Constant => "constant".to_owned(),
            FillTiming::Event => "event".to_owned(),
        });
        let counters = self.counters;
        L1BoundaryState {
            schema_version: TYPED_STATE_SCHEMA_VERSION,
            architecture: match self.cfg.arch {
                ArchId::Sm86 => "sm_86".to_owned(),
                ArchId::Sm120 => "sm_120".to_owned(),
            },
            line_capacity: self.cfg.line_capacity,
            mshr_count: self.cfg.mshr_count,
            hit_latency: self.cfg.hit_latency,
            miss_latency: self.cfg.miss_latency,
            line_order,
            stamp_of,
            sectors,
            line_fills,
            sector_fills,
            line_generations,
            next_stamp,
            fill_sequence,
            active_lookup_batch,
            min_fill_due: self.min_fill_due,
            fill_timing,
            counters: L1CounterState {
                hits: counters.hits,
                misses: counters.misses,
                merges: counters.merges,
                already_requested_merges: counters.already_requested_merges,
                unrequested_pending_merges: counters.unrequested_pending_merges,
                pending_lookup_hits: counters.pending_lookup_hits,
                stall_events: counters.stall_events,
                line_fills: counters.line_fills,
                line_alloc_misses: counters.line_alloc_misses,
                sector_resident_misses: counters.sector_resident_misses,
                bypass_sectors: counters.bypass_sectors,
                bypass_atom_sectors: counters.bypass_atom_sectors,
                untagged_sectors: counters.untagged_sectors,
                store_sectors: counters.store_sectors,
                store_hits: counters.store_hits,
                store_tag_hits: counters.store_tag_hits,
                write_lookup_hits: counters.write_lookup_hits,
                reduction_sectors: counters.reduction_sectors,
                reduction_hits: counters.reduction_hits,
            },
            write_observer_present: self.write_lookup_observer.is_some(),
        }
    }
}

impl L1Cache {
    /// Whether `fill` is still owned by this cache's event-mode MSHR state.
    /// Response handlers use this as a fail-closed preflight before consuming
    /// their exact response token or mutating any waiter accounting.
    pub fn is_fill_in_flight(&self, fill: FillKey) -> bool {
        match fill {
            FillKey::Line { line, generation } => self
                .line_fills
                .get(&line)
                .is_some_and(|&(_, _, current)| current == generation),
            FillKey::Sector {
                line,
                sector,
                generation,
            } => self.sector_fills.contains_key(&(line, sector, generation)),
        }
    }
}

impl L1Cache {
    /// Whether any generation of a line-allocation fill is currently in
    /// flight.  This narrow query is used only for validating the companion
    /// line-mask bookkeeping; completion itself must still carry the exact
    /// generation-bearing [`FillKey`].
    pub fn is_line_fill_in_flight(&self, line: i64) -> bool {
        self.line_fills.contains_key(&line)
    }
}

impl L1Cache {
    /// Start an ordinary warp instruction's logical L1 lookup request. Pending
    /// merges remain misses for this observer. The returned token must be
    /// retained if an MSHR-stalled remainder is retried later.
    pub fn begin_lookup_batch(&mut self) -> L1LookupBatch {
        self.begin_lookup_batch_for_client(L1LookupClient::Unknown)
    }
}

impl L1Cache {
    pub fn begin_lookup_batch_for_client_with_sectors(
        &mut self,
        client: L1LookupClient,
        sectors: &[i64],
    ) -> L1LookupBatch {
        let batch = self.begin_lookup_batch_for_client(client);
        self.record_request_masks(client, sectors);
        batch
    }
}

impl L1Cache {
    pub fn begin_async_copy_lookup_batch_with_sectors(&mut self, sectors: &[i64]) -> L1LookupBatch {
        let batch = self.begin_async_copy_lookup_batch();
        self.record_request_masks(L1LookupClient::AsyncCopy, sectors);
        batch
    }
}

impl L1Cache {
    /// Resume the observer boundary of a previously stalled logical request.
    /// Cache behavior is unchanged; only merge attribution consults the token.
    pub fn resume_lookup_batch(&mut self, batch: L1LookupBatch) {
        self.active_lookup_batch = Some(batch);
    }
}

impl L1Cache {
    pub(super) fn preflight_fill_timing(
        stored: FillDue,
        requested: FillTiming,
    ) -> Result<(), L1AccessError> {
        let latched = stored.timing();
        if latched != requested {
            return Err(L1AccessError::FillTimingMismatch {
                latched: Self::timing_name(latched),
                requested: Self::timing_name(requested),
            });
        }
        Ok(())
    }
}

impl L1Cache {
    pub(super) fn commit_latch(&mut self, mode: FillTiming) {
        if self.fill_timing.is_none() {
            self.fill_timing = Some(mode);
        }
    }
}

impl L1Cache {
    pub(crate) fn checked_time_add(
        operation: &'static str,
        base: u64,
        delta: u64,
    ) -> Result<u64, L1AccessError> {
        base.checked_add(delta).ok_or(L1AccessError::TimeOverflow {
            operation,
            base,
            delta,
        })
    }
}

impl L1Cache {
    /// Check recency headroom for fills that will be applied before one
    /// access-level MRU touch.  The ordinary mutation budget only covers the
    /// access itself; due fills are a separate lazy transition and can consume
    /// stamps first.  Count every due fill conservatively (sector fills may be
    /// dropped after an eviction), so a near-exhaustion path is redirected to
    /// the transactional shadow replay instead of publishing a partial fill.
    pub(super) fn preflight_due_fill_recency(
        &self,
        now: u64,
        extra_touches: u64,
    ) -> Result<(), L1AccessError> {
        // The maintained minimum makes the ordinary no-due/future-fill path
        // constant time. Exact counting remains necessary only at a due-fill
        // boundary because an evicted sector fill may be dropped.
        if self.min_fill_due.is_none_or(|minimum| minimum > now) {
            return self.next_stamp.preflight("recency", extra_touches);
        }
        let due_fills = self
            .line_fills
            .values()
            .filter(|&&(due, ..)| due.timed().is_some_and(|time| time <= now))
            .count()
            .checked_add(
                self.sector_fills
                    .values()
                    .filter(|&&(due, _)| due.timed().is_some_and(|time| time <= now))
                    .count(),
            )
            .and_then(|count| u64::try_from(count).ok())
            .ok_or(L1AccessError::SequenceOverflow {
                sequence: "recency",
                current: self.next_stamp.current(),
            })?;
        let uses = due_fills
            .checked_add(extra_touches)
            .ok_or(L1AccessError::SequenceOverflow {
                sequence: "recency",
                current: self.next_stamp.current(),
            })?;
        self.next_stamp.preflight("recency", uses)
    }
}

impl L1Cache {
    pub(super) fn preflight_fill_generation_headroom(
        &self,
        new_fills: u64,
    ) -> Result<(), L1AccessError> {
        let Some(last_offset) = new_fills.checked_sub(1) else {
            return Ok(());
        };
        let current = self.fill_seq.current();
        let generation_limit = u64::from(u32::MAX);
        if current
            .checked_add(last_offset)
            .is_none_or(|last| last > generation_limit)
        {
            return Err(L1AccessError::SequenceOverflow {
                sequence: "fill generation",
                current,
            });
        }
        Ok(())
    }
}

impl L1Cache {
    /// Prove one conservative mutation budget. Callers can fall back to exact
    /// shadow replay only when a near-limit budget is too conservative.
    pub fn preflight_mutation_budget(&self, budget: L1MutationBudget) -> Result<(), L1AccessError> {
        self.preflight_mutation_budgets(std::slice::from_ref(&budget))
    }
}

impl L1Cache {
    pub(super) fn earliest_fill(&self) -> Result<u64, L1AccessError> {
        self.line_fills
            .values()
            .map(|&(due, ..)| due)
            .chain(self.sector_fills.values().map(|&(due, ..)| due))
            .filter_map(FillDue::timed)
            .min()
            .ok_or(L1AccessError::StallWithoutFill)
    }
}

impl L1Cache {
    /// Conservative proof that a constant-mode access and a possible one-entry
    /// MSHR retry have enough timestamp headroom. The cached minimum is the
    /// common path; when fills are already due, find the first still-future fill
    /// that a full MSHR set could retry at. Callers batching several accesses use
    /// a shadow replay when this proof is too conservative.
    pub fn preflight_constant_access_timing(&self, now: u64) -> Result<(), L1AccessError> {
        self.preflight_latch(FillTiming::Constant)?;
        Self::checked_time_add("hit visibility", now, self.cfg.hit_latency)?;
        Self::checked_time_add("miss visibility", now, self.cfg.miss_latency)?;
        let retry_at = if self.mshr_in_use() == 0 {
            None
        } else if let Some(minimum) = self.min_fill_due.filter(|&due| due > now) {
            Some(minimum)
        } else {
            self.line_fills
                .values()
                .map(|&(due, ..)| due)
                .chain(self.sector_fills.values().map(|&(due, ..)| due))
                .filter_map(FillDue::timed)
                .filter(|&time| time > now)
                .min()
        };
        if let Some(retry_at) = retry_at {
            Self::checked_time_add("hit visibility", retry_at, self.cfg.hit_latency)?;
            Self::checked_time_add("miss visibility", retry_at, self.cfg.miss_latency)?;
        }
        self.preflight_due_fill_recency(now, 1)?;
        Ok(())
    }
}

impl L1Cache {
    /// Does the conservative constant-batch ready-time bound fit in u64?
    pub fn constant_batch_timing_has_headroom(&self, now: u64, sectors: u64) -> bool {
        self.constant_batch_ready_upper_bound(now, sectors)
            .is_some()
    }
}

impl L1Cache {
    pub(super) fn set_index(&self, line: i64) -> Option<usize> {
        if let Some(&index) = self.set_locations.get(&line) {
            return Some(index);
        }
        let count = self.cfg.set_count?;
        let modulo = usize::try_from(line.rem_euclid(i64::from(count))).ok()?;
        let folded = usize::try_from(
            (line ^ line.div_euclid(i64::from(count))).rem_euclid(i64::from(count)),
        )
        .ok()?;
        match self.cfg.set_index {
            L1SetIndex::Modulo => Some(modulo),
            L1SetIndex::XorFold => Some(folded),
            L1SetIndex::ModuloXorFallback => {
                let ways = usize::try_from(self.cfg.ways?).ok()?;
                let sets = self.set_lru.as_ref()?;
                let modulo_len = sets.get(modulo).map_or(0, VecDeque::len);
                let folded_len = sets.get(folded).map_or(0, VecDeque::len);
                if modulo_len >= ways && folded_len < modulo_len {
                    Some(folded)
                } else {
                    Some(modulo)
                }
            }
            L1SetIndex::FullyAssociative => None,
        }
    }
}

impl L1Cache {
    pub(super) fn remove_resident_line(&mut self, line: i64) {
        if let Some(stamp) = self.stamp_of.remove(&line) {
            self.by_stamp.remove(&stamp);
        }
        self.remove_from_set_lru(line);
        self.sectors.remove(&line);
        self.line_generations.remove(&line);
    }
}

impl L1Cache {
    pub(super) fn preflight_dirty_eviction(
        &self,
        additional_line: Option<i64>,
    ) -> Result<(), L1AccessError> {
        let capacity = usize::try_from(self.cfg.line_capacity).unwrap_or(usize::MAX);
        let additional = additional_line
            .filter(|line| !self.line_resident(*line))
            .map_or(0, |_| 1);
        let evict_count = self
            .by_stamp
            .len()
            .saturating_add(additional)
            .saturating_sub(capacity);
        for (&_stamp, &line) in self.by_stamp.iter().take(evict_count) {
            if let Some(&dirty_mask) = self.dirty.get(&line) {
                return Err(L1AccessError::DirtyResidencyBlocksEviction { line, dirty_mask });
            }
        }
        if let (Some(ways), Some(sets)) = (self.cfg.ways, self.set_lru.as_ref()) {
            let incoming = additional_line.filter(|line| !self.line_resident(*line));
            let limit = usize::try_from(ways).unwrap_or(usize::MAX);
            for (index, set) in sets.iter().enumerate() {
                let extra = incoming
                    .and_then(|line| self.set_index(line).filter(|candidate| *candidate == index))
                    .map_or(0, |_| 1);
                let overflow = set.len().saturating_add(extra).saturating_sub(limit);
                for &line in set.iter().take(overflow) {
                    if let Some(&dirty_mask) = self.dirty.get(&line) {
                        return Err(L1AccessError::DirtyResidencyBlocksEviction {
                            line,
                            dirty_mask,
                        });
                    }
                }
            }
        }
        Ok(())
    }
}

impl L1Cache {
    pub(super) fn line_resident(&self, line: i64) -> bool {
        self.stamp_of.contains_key(&line)
    }
}

impl L1Cache {
    pub(super) fn validate_sector_mask(line: i64, sector_mask: u8) -> Result<(), L1AccessError> {
        if sector_mask == 0 || sector_mask & !(0b1111) != 0 {
            return Err(L1AccessError::InvalidationMaskInvalid { line, sector_mask });
        }
        Ok(())
    }
}

impl L1Cache {
    /// Mark one resident sector dirty with a complete known-byte payload.
    /// This is the value-bearing companion to the legacy mask-only method.
    pub fn mark_dirty_payload(&mut self, payload: SectorPayloadV1) -> Result<(), L1AccessError> {
        let line = payload.line_id();
        let sector = payload.sector();
        if !self.line_resident(line) {
            return Err(L1AccessError::DirtyLineNotResident { line });
        }
        let valid_mask = self.sectors.get(&line).copied().unwrap_or(0);
        let bit = 1_u8 << sector;
        if valid_mask & bit == 0 {
            return Err(L1AccessError::DirtyMaskInvalid {
                line,
                dirty_mask: bit,
                valid_mask,
            });
        }
        if !payload.is_fully_known() {
            return Err(L1AccessError::DirtyPayloadMissing { line, sector });
        }
        let payload = match self.dirty_payloads.get(&(line, sector)) {
            Some(previous) => previous.merge(payload)?,
            None => payload,
        };
        self.dirty
            .insert(line, self.dirty.get(&line).copied().unwrap_or(0) | bit);
        self.dirty_payloads.insert((line, sector), payload);
        Ok(())
    }
}

impl L1Cache {
    /// Admit payloads only after an external allocation/residency witness has
    /// been checked against the current L1 allocation generation. This is the
    /// native-facing contract; the ordinary sidecar path intentionally does
    /// not synthesize these witnesses.
    pub fn preview_dirty_payload_admission_with_witness(
        &self,
        payloads: &[SectorPayloadV1],
        witnesses: &[L1PayloadResidencyWitness],
    ) -> Result<L1DirtyPayloadAdmissionPlan, L1AccessError> {
        self.validate_payload_residency_witnesses(payloads, witnesses)?;
        self.preview_dirty_payload_admission(payloads)
    }
}

impl L1Cache {
    pub(super) fn validate_payload_residency_witnesses(
        &self,
        payloads: &[SectorPayloadV1],
        witnesses: &[L1PayloadResidencyWitness],
    ) -> Result<(), L1AccessError> {
        let mut by_line = BTreeMap::new();
        for witness in witnesses {
            if witness.allocation_id == 0 || witness.allocation_generation == 0 {
                return Err(L1AccessError::DirtyPayloadResidencyWitnessMismatch {
                    line: witness.line_id,
                    field: "allocation identity must be nonzero",
                });
            }
            if by_line.insert(witness.line_id, *witness).is_some() {
                return Err(L1AccessError::DirtyPayloadResidencyWitnessDuplicate {
                    line: witness.line_id,
                });
            }
        }
        for payload in payloads {
            let line = payload.line_id();
            let witness = by_line
                .get(&line)
                .ok_or(L1AccessError::DirtyPayloadResidencyWitnessMissing { line })?;
            if witness.allocation_id != payload.allocation_id()
                || witness.allocation_generation != payload.allocation_generation()
            {
                return Err(L1AccessError::DirtyPayloadResidencyWitnessMismatch {
                    line,
                    field: "payload allocation identity",
                });
            }
            let actual_generation = self.line_generations.get(&line).copied();
            if actual_generation != Some(witness.resident_generation) {
                return Err(L1AccessError::DirtyPayloadResidencyWitnessMismatch {
                    line,
                    field: "resident allocation generation",
                });
            }
            let actual_mask = self.sectors.get(&line).copied().unwrap_or(0);
            if witness.resident_sector_mask & !actual_mask != 0 {
                return Err(L1AccessError::DirtyPayloadResidencyWitnessMismatch {
                    line,
                    field: "resident sector mask",
                });
            }
            if witness.resident_sector_mask & (1_u8 << payload.sector()) == 0 {
                return Err(L1AccessError::DirtyPayloadResidencyWitnessMismatch {
                    line,
                    field: "payload sector not witnessed resident",
                });
            }
        }
        Ok(())
    }
}

impl L1Cache {
    /// Commit a previously preflighted payload batch atomically. Any state
    /// change after preview causes a typed stale-plan error and leaves this
    /// cache untouched.
    pub fn commit_dirty_payload_admission(
        &mut self,
        plan: L1DirtyPayloadAdmissionPlan,
    ) -> Result<(), L1AccessError> {
        for expected in plan.expected.iter() {
            let current = self
                .dirty_payloads
                .get(&(expected.line, expected.sector))
                .copied();
            let valid_mask = self.sectors.get(&expected.line).copied().unwrap_or(0);
            let dirty_mask = self.dirty.get(&expected.line).copied().unwrap_or(0);
            let generation = self.line_generations.get(&expected.line).copied();
            if current != expected.current
                || valid_mask != expected.valid_mask
                || dirty_mask != expected.dirty_mask
                || generation != expected.generation
            {
                return Err(L1AccessError::DirtyPayloadAdmissionStale {
                    line: expected.line,
                    sector: expected.sector,
                });
            }
        }
        let mut next = self.clone();
        for payload in plan.payloads.iter().copied() {
            next.mark_dirty_payload(payload)?;
        }
        *self = next;
        Ok(())
    }
}

impl L1Cache {
    /// Snapshot dirty sectors without mutating cache state. The plan is
    /// committed only after the lower-level service has accepted the exact
    /// masks, preventing a failed admission from silently clearing dirty data.
    pub fn preview_dirty_flush(&self) -> L1DirtyFlushPlan {
        let mut lines = self
            .dirty
            .iter()
            .map(|(&line_id, &dirty_mask)| L1DirtyLine {
                line_id,
                dirty_mask,
            })
            .collect::<Vec<_>>();
        lines.sort_unstable_by_key(|line| line.line_id);
        L1DirtyFlushPlan {
            lines: lines.into_boxed_slice(),
        }
    }
}

impl L1Cache {
    /// Invalidate resident sectors after the caller has completed any dirty
    /// flush and protocol service. Pending fills are rejected rather than
    /// silently dropped because response tokens need an explicit invalidation
    /// event/ack path in the SM system.
    pub fn invalidate(
        &mut self,
        line: i64,
        sector_mask: u8,
    ) -> Result<L1InvalidationResult, L1AccessError> {
        Self::validate_sector_mask(line, sector_mask)?;
        if self.line_fills.contains_key(&line)
            || self
                .sector_fills
                .keys()
                .any(|(candidate, _, _)| *candidate == line)
        {
            return Err(L1AccessError::InvalidationPendingFill { line });
        }
        let resident_mask = self.sectors.get(&line).copied().unwrap_or(0);
        let invalidated_mask = resident_mask & sector_mask;
        let dirty_mask = self.dirty.get(&line).copied().unwrap_or(0) & invalidated_mask;
        if dirty_mask != 0 {
            return Err(L1AccessError::DirtyResidencyBlocksInvalidation { line, dirty_mask });
        }
        if invalidated_mask != 0 {
            let next = resident_mask & !invalidated_mask;
            if next == 0 {
                self.sectors.remove(&line);
                self.remove_from_set_lru(line);
                if let Some(stamp) = self.stamp_of.remove(&line) {
                    self.by_stamp.remove(&stamp);
                }
            } else {
                self.sectors.insert(line, next);
            }
        }
        Ok(L1InvalidationResult {
            line_id: line,
            requested_mask: sector_mask,
            invalidated_mask,
        })
    }
}

impl L1Cache {
    pub(super) fn commit_due_fills(
        &mut self,
        due: Vec<(u64, u8, u64, i64, u8, u32)>,
    ) -> Result<(), L1AccessError> {
        for (_, kind, _, line, sector, generation) in due {
            if kind == 0 {
                self.line_fills.remove(&line);
                self.touch_mru(line)?;
                self.evict_beyond_capacity()?;
            } else {
                self.sector_fills.remove(&(line, sector, generation));
                let generation_matches = self.line_generations.get(&line) == Some(&generation);
                if self.line_resident(line) && generation_matches {
                    // The sector arrives into its (still-resident) line
                    // and re-references it at MRU.
                    *self.sectors.entry(line).or_default() |= 1 << sector;
                    self.touch_mru(line)?;
                }
                // else: the line was evicted before this fill's own
                // timestamp — the fill is DROPPED.
            }
        }
        self.min_fill_due = self
            .line_fills
            .values()
            .map(|&(due, ..)| due)
            .chain(self.sector_fills.values().map(|&(due, ..)| due))
            .filter_map(FillDue::timed)
            .min();
        Ok(())
    }
}

impl L1Cache {
    /// One classification/state step shared by BOTH fill-timing modes —
    /// the single copy of the ported ladder (line-merge → sector-present
    /// hit → sector-merge → MSHR stall → fresh miss), with all counter
    /// accounting inside. [`FillRequest`] computes a hit timestamp only on a
    /// hit and a fill due state only on a fresh miss, so an unused timing arm
    /// cannot reject a valid merge, stall, or exact-domain result.
    pub(super) fn access_core(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        fill_request: FillRequest,
        mshr_floor: Option<NonZeroU32>,
    ) -> Result<CoreAccess, L1AccessError> {
        let line = sector_id >> 2;
        let sector = (sector_id & 3) as u8;
        let sector_bit = 1_u8 << sector;
        let sector_already_known_before =
            self.sectors.get(&line).copied().unwrap_or(0) & sector_bit != 0;
        self.record_constant_request(origin);
        self.record_wavefront_mask_lookup(origin, line, sector);
        self.record_unrequested_subline_miss(origin, line, sector);
        // A late child of an in-flight line allocation has its own sector
        // completion token. Check it before the parent line MSHR so later
        // accesses to that sector wait for the child response instead of
        // incorrectly riding the parent's earlier response.
        if let Some(&generation) = self.line_generations.get(&line)
            && let Some(&(pending, sequence)) = self.sector_fills.get(&(line, sector, generation))
        {
            Self::preflight_fill_timing(pending, fill_request.timing())?;
            self.record_merge_audit(origin, line, sector, sequence);
            self.record_older_sector_pending_merge(origin, sequence);
            self.record_pending_merge_class(origin, true);
            // An explicit sector child exists, so this request is already
            // represented by the outstanding fill mask.  It remains a timing
            // merge, but is an observer lookup hit on SM120.
            let counters = if self.active_lookup_batch.is_some_and(|batch| {
                batch.pending_merges_are_hits && sequence < batch.first_fill_sequence
            }) {
                self.counters.add_fields_checked(
                    origin,
                    &[
                        (L1CounterField::Merges, 1),
                        (L1CounterField::AlreadyRequestedMerges, 1),
                        (L1CounterField::Misses, 1),
                        (L1CounterField::PendingLookupHits, 1),
                    ],
                )?
            } else {
                self.counters.add_fields_checked(
                    origin,
                    &[
                        (L1CounterField::Merges, 1),
                        (L1CounterField::AlreadyRequestedMerges, 1),
                        (L1CounterField::Misses, 1),
                    ],
                )?
            };
            self.counters = counters;
            return Ok(CoreAccess::SectorMerge {
                pending,
                generation,
            });
        }
        if let Some(&(pending, sequence, generation)) = self.line_fills.get(&line) {
            Self::preflight_fill_timing(pending, fill_request.timing())?;
            self.record_merge_audit(origin, line, sector, sequence);
            self.record_pending_merge_class(origin, sector_already_known_before);
            let counters = if self.active_lookup_batch.is_some_and(|batch| {
                batch.pending_merges_are_hits && sequence < batch.first_fill_sequence
            }) {
                self.counters.add_fields_checked(
                    origin,
                    &[
                        (L1CounterField::Merges, 1),
                        (
                            L1CounterField::AlreadyRequestedMerges,
                            u64::from(sector_already_known_before),
                        ),
                        (
                            L1CounterField::UnrequestedPendingMerges,
                            u64::from(!sector_already_known_before),
                        ),
                        (L1CounterField::Misses, 1),
                        (L1CounterField::PendingLookupHits, 1),
                    ],
                )?
            } else {
                self.counters.add_fields_checked(
                    origin,
                    &[
                        (L1CounterField::Merges, 1),
                        (
                            L1CounterField::AlreadyRequestedMerges,
                            u64::from(sector_already_known_before),
                        ),
                        (
                            L1CounterField::UnrequestedPendingMerges,
                            u64::from(!sector_already_known_before),
                        ),
                        (L1CounterField::Misses, 1),
                    ],
                )?
            };
            self.counters = counters;
            *self.sectors.entry(line).or_default() |= 1 << sector;
            return Ok(CoreAccess::LineMerge {
                pending,
                generation,
            });
        }
        let line_resident = self.line_resident(line);
        // Sector-present HIT testing precedes sector-merge handling.
        let sector_present = line_resident
            && self
                .sectors
                .get(&line)
                .is_some_and(|mask| mask & (1 << sector) != 0);
        if sector_present {
            self.record_client_hit(origin);
            let ready = fill_request.hit_ready(self.cfg.hit_latency)?;
            self.next_stamp.preflight("recency", 1)?;
            let counters = self
                .counters
                .add_fields_checked(origin, &[(L1CounterField::Hits, 1)])?;
            self.counters = counters;
            self.touch_mru(line)?;
            return Ok(CoreAccess::Hit { ready });
        }
        let line_generation = self.line_generations.get(&line).copied();
        if line_resident
            && let Some(generation) = line_generation
            && let Some(&(pending, sequence)) = self.sector_fills.get(&(line, sector, generation))
        {
            Self::preflight_fill_timing(pending, fill_request.timing())?;
            self.record_merge_audit(origin, line, sector, sequence);
            self.record_older_sector_pending_merge(origin, sequence);
            self.record_pending_merge_class(origin, true);
            // The child map is the request mask for this resident-line
            // sector; count the merge as already requested.
            let counters = if self.active_lookup_batch.is_some_and(|batch| {
                batch.pending_merges_are_hits && sequence < batch.first_fill_sequence
            }) {
                self.counters.add_fields_checked(
                    origin,
                    &[
                        (L1CounterField::Merges, 1),
                        (L1CounterField::AlreadyRequestedMerges, 1),
                        (L1CounterField::Misses, 1),
                        (L1CounterField::PendingLookupHits, 1),
                    ],
                )?
            } else {
                self.counters.add_fields_checked(
                    origin,
                    &[
                        (L1CounterField::Merges, 1),
                        (L1CounterField::AlreadyRequestedMerges, 1),
                        (L1CounterField::Misses, 1),
                    ],
                )?
            };
            self.counters = counters;
            return Ok(CoreAccess::SectorMerge {
                pending,
                generation,
            });
        }
        // Miss path: needs an MSHR entry (both fill kinds consume capacity).
        if let Some(cap) = self.effective_mshr_count(mshr_floor)
            && self.mshr_in_use() >= cap
        {
            let occupancy = u64::try_from(self.mshr_in_use()).map_err(|_| {
                L1AccessError::Counter(L1CounterError::PressureOverflow {
                    field: "mshr_stall_occupancy_sum",
                })
            })?;
            let mut counters = self
                .counters
                .add_fields_checked(origin, &[(L1CounterField::StallEvents, 1)])?;
            counters.mshr_stall_occupancy_samples = counters
                .mshr_stall_occupancy_samples
                .checked_add(1)
                .ok_or(L1AccessError::Counter(L1CounterError::PressureOverflow {
                    field: "mshr_stall_occupancy_samples",
                }))?;
            counters.mshr_stall_occupancy_sum = counters
                .mshr_stall_occupancy_sum
                .checked_add(occupancy)
                .ok_or(L1AccessError::Counter(L1CounterError::PressureOverflow {
                    field: "mshr_stall_occupancy_sum",
                }))?;
            self.counters = counters;
            return Ok(CoreAccess::Stalled);
        }
        let fill_due = fill_request.fill_due(self.cfg.miss_latency)?;
        let (seq, next_fill_seq) = self.fill_seq.allocate("fill")?;
        // The allocation sequence is u64. A resident-line sector fill carries
        // the existing line allocation's compact u32 generation in its
        // FillKey, so it must not fail merely because the independent fill
        // ordering sequence has crossed the u32 boundary. A new line
        // allocation still needs the checked conversion below.
        let generation = if line_resident {
            self.line_generations
                .get(&line)
                .copied()
                .or_else(|| u32::try_from(seq).ok())
                .ok_or(L1AccessError::SequenceOverflow {
                    sequence: "fill generation",
                    current: seq,
                })?
        } else {
            u32::try_from(seq).map_err(|_| L1AccessError::SequenceOverflow {
                sequence: "fill generation",
                current: seq,
            })?
        };
        let counters = if line_resident {
            self.counters.add_fields_checked(
                origin,
                &[
                    (L1CounterField::Misses, 1),
                    (L1CounterField::SectorResidentMisses, 1),
                ],
            )?
        } else {
            self.counters.add_fields_checked(
                origin,
                &[
                    (L1CounterField::Misses, 1),
                    (L1CounterField::LineAllocMisses, 1),
                    (L1CounterField::LineFills, 1),
                ],
            )?
        };
        self.fill_seq = next_fill_seq;
        self.counters = counters;
        if let FillDue::Timed(time) = fill_due {
            self.min_fill_due = Some(self.min_fill_due.map_or(time, |due| due.min(time)));
        }
        if line_resident {
            // Partial-line sector fetch: no allocation, no eviction; the
            // sector becomes resident at its FILL time (lazy).
            let generation = self
                .line_generations
                .entry(line)
                .or_insert(generation)
                .to_owned();
            self.sector_fills
                .insert((line, sector, generation), (fill_due, seq));
            // Capture the allocation that owns this resident line.  This is
            // required in event mode because a response may arrive after the
            // line has been evicted and the same tag reallocated.
            Ok(CoreAccess::MissSectorResident {
                pending: fill_due,
                generation,
            })
        } else {
            // The fill sequence is also the allocation-generation witness.
            // It is installed at allocation time (rather than response time)
            // so an older sector response cannot enter a newly allocated line
            // while that line is still in flight.
            self.line_generations.insert(line, generation);
            self.line_fills.insert(line, (fill_due, seq, generation));
            *self.sectors.entry(line).or_default() |= 1 << sector;
            Ok(CoreAccess::MissLineAlloc {
                pending: fill_due,
                generation,
            })
        }
    }
}

impl L1Cache {
    pub(super) fn access_eligible_event_unobserved_impl(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
        mshr_floor: Option<NonZeroU32>,
    ) -> Result<EvAccessUnobserved, L1AccessError> {
        let line = sector_id >> 2;
        let sector = (sector_id & 3) as u8;
        Ok(
            match self.access_eligible_event_core(
                origin,
                sector_id,
                now,
                Readiness::Unobserved,
                mshr_floor,
            )? {
                CoreAccess::Hit { .. } => EvAccessUnobserved::Hit,
                CoreAccess::LineMerge { generation, .. } => EvAccessUnobserved::MergedPending {
                    fill: FillKey::Line { line, generation },
                },
                CoreAccess::SectorMerge { generation, .. } => EvAccessUnobserved::MergedPending {
                    fill: FillKey::Sector {
                        line,
                        sector,
                        generation,
                    },
                },
                CoreAccess::Stalled => EvAccessUnobserved::Stalled,
                CoreAccess::MissSectorResident { generation, .. } => {
                    EvAccessUnobserved::MissRequest {
                        fill: FillKey::Sector {
                            line,
                            sector,
                            generation,
                        },
                    }
                }
                CoreAccess::MissLineAlloc { generation, .. } => EvAccessUnobserved::MissRequest {
                    fill: FillKey::Line { line, generation },
                },
            },
        )
    }
}

impl L1Cache {
    /// Event-mode eligible access with a caller-characterized lower bound on
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
}

impl L1Cache {
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
}

impl L1Cache {
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
}

impl L1Cache {
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
}

impl L1Cache {
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
}
