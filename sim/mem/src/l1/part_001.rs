// Lexical implementation section 2/3; included by the parent facade.
impl L1Cache {pub fn new(cfg: L1Config) -> Result<Self, L1ConfigError> {
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

    /// Return the observer-only request-boundary merge provenance.
    pub const fn merge_audit(&self) -> L1MergeAudit {
        self.merge_audit
    }

    fn record_pending_merge_class(&mut self, origin: MemoryOrigin, already_requested: bool) {
        let lane = if already_requested {
            &mut self.merge_audit.already_requested_merges
        } else {
            &mut self.merge_audit.unrequested_pending_merges
        };
        lane[origin.index()] = lane[origin.index()].saturating_add(1);
    }

    fn record_merge_audit(&mut self, origin: MemoryOrigin, line: i64, sector: u8, sequence: u64) {
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

    fn record_request_masks(&mut self, client: L1LookupClient, sectors: &[i64]) {
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

    fn record_client_hit(&mut self, origin: MemoryOrigin) {
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

    fn record_older_sector_pending_merge(&mut self, origin: MemoryOrigin, sequence: u64) {
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

    fn record_unrequested_subline_miss(&mut self, origin: MemoryOrigin, line: i64, sector: u8) {
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

    fn record_wavefront_mask_lookup(&mut self, origin: MemoryOrigin, line: i64, sector: u8) {
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

    fn record_constant_request(&mut self, origin: MemoryOrigin) {
        if origin == MemoryOrigin::Global
            && self
                .active_lookup_batch
                .is_some_and(|batch| batch.client == L1LookupClient::Constant)
        {
            self.merge_audit.constant_global_requests =
                self.merge_audit.constant_global_requests.saturating_add(1);
        }
    }

    /// Select the explicit resident-hit writeback experiment. This does not
    /// alter cache geometry or fill timing and is intentionally opt-in.
    pub fn set_store_policy(&mut self, policy: L1StorePolicy) {
        self.store_policy = policy;
    }

    pub const fn store_policy(&self) -> L1StorePolicy {
        self.store_policy
    }

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

    pub fn arch(&self) -> ArchId {
        self.cfg.arch
    }

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

    fn mshr_in_use(&self) -> usize {
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

    /// Access-local admission may raise a finite generic approximation to an
    /// independently measured floor without rewriting the cache configuration.
    /// An unlimited ablation remains unlimited.
    fn effective_mshr_count(&self, floor: Option<NonZeroU32>) -> Option<usize> {
        self.cfg.mshr_count.map(|configured| {
            let effective = floor.map_or(configured, |floor| configured.max(floor.get()));
            effective as usize
        })
    }

    fn constant_access_may_retry(
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

    /// Whether any generation of a line-allocation fill is currently in
    /// flight.  This narrow query is used only for validating the companion
    /// line-mask bookkeeping; completion itself must still carry the exact
    /// generation-bearing [`FillKey`].
    pub fn is_line_fill_in_flight(&self, line: i64) -> bool {
        self.line_fills.contains_key(&line)
    }

    /// Start an ordinary warp instruction's logical L1 lookup request. Pending
    /// merges remain misses for this observer. The returned token must be
    /// retained if an MSHR-stalled remainder is retried later.
    pub fn begin_lookup_batch(&mut self) -> L1LookupBatch {
        self.begin_lookup_batch_for_client(L1LookupClient::Unknown)
    }

    pub fn begin_lookup_batch_for_client(&mut self, client: L1LookupClient) -> L1LookupBatch {
        let batch = L1LookupBatch {
            first_fill_sequence: self.fill_seq.current(),
            pending_merges_are_hits: false,
            client,
        };
        self.active_lookup_batch = Some(batch);
        batch
    }

    pub fn begin_lookup_batch_for_client_with_sectors(
        &mut self,
        client: L1LookupClient,
        sectors: &[i64],
    ) -> L1LookupBatch {
        let batch = self.begin_lookup_batch_for_client(client);
        self.record_request_masks(client, sectors);
        batch
    }

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

    pub fn begin_async_copy_lookup_batch_with_sectors(&mut self, sectors: &[i64]) -> L1LookupBatch {
        let batch = self.begin_async_copy_lookup_batch();
        self.record_request_masks(L1LookupClient::AsyncCopy, sectors);
        batch
    }

    /// Resume the observer boundary of a previously stalled logical request.
    /// Cache behavior is unchanged; only merge attribution consults the token.
    pub fn resume_lookup_batch(&mut self, batch: L1LookupBatch) {
        self.active_lookup_batch = Some(batch);
    }

    fn timing_name(mode: FillTiming) -> &'static str {
        match mode {
            FillTiming::Constant => "constant",
            FillTiming::Event => "event",
        }
    }

    fn preflight_fill_timing(stored: FillDue, requested: FillTiming) -> Result<(), L1AccessError> {
        let latched = stored.timing();
        if latched != requested {
            return Err(L1AccessError::FillTimingMismatch {
                latched: Self::timing_name(latched),
                requested: Self::timing_name(requested),
            });
        }
        Ok(())
    }

    fn preflight_latch(&self, requested: FillTiming) -> Result<(), L1AccessError> {
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

    fn commit_latch(&mut self, mode: FillTiming) {
        if self.fill_timing.is_none() {
            self.fill_timing = Some(mode);
        }
    }

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

    fn checked_time_add(
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

    /// Check recency headroom for fills that will be applied before one
    /// access-level MRU touch.  The ordinary mutation budget only covers the
    /// access itself; due fills are a separate lazy transition and can consume
    /// stamps first.  Count every due fill conservatively (sector fills may be
    /// dropped after an eviction), so a near-exhaustion path is redirected to
    /// the transactional shadow replay instead of publishing a partial fill.
    fn preflight_due_fill_recency(
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

    fn apply_counter_deltas(
        &mut self,
        origin: MemoryOrigin,
        deltas: &[(L1CounterField, u64)],
    ) -> Result<(), L1AccessError> {
        self.counters = self.counters.add_fields_checked(origin, deltas)?;
        Ok(())
    }

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

    fn preflight_fill_generation_headroom(&self, new_fills: u64) -> Result<(), L1AccessError> {
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

    /// Prove one conservative mutation budget. Callers can fall back to exact
    /// shadow replay only when a near-limit budget is too conservative.
    pub fn preflight_mutation_budget(&self, budget: L1MutationBudget) -> Result<(), L1AccessError> {
        self.preflight_mutation_budgets(std::slice::from_ref(&budget))
    }

    fn earliest_fill(&self) -> Result<u64, L1AccessError> {
        self.line_fills
            .values()
            .map(|&(due, ..)| due)
            .chain(self.sector_fills.values().map(|&(due, ..)| due))
            .filter_map(FillDue::timed)
            .min()
            .ok_or(L1AccessError::StallWithoutFill)
    }

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

    /// Does the conservative constant-batch ready-time bound fit in u64?
    pub fn constant_batch_timing_has_headroom(&self, now: u64, sectors: u64) -> bool {
        self.constant_batch_ready_upper_bound(now, sectors)
            .is_some()
    }

    fn set_index(&self, line: i64) -> Option<usize> {
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

    fn remove_from_set_lru(&mut self, line: i64) {
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

    fn remove_resident_line(&mut self, line: i64) {
        if let Some(stamp) = self.stamp_of.remove(&line) {
            self.by_stamp.remove(&stamp);
        }
        self.remove_from_set_lru(line);
        self.sectors.remove(&line);
        self.line_generations.remove(&line);
    }

    fn touch_mru(&mut self, line: i64) -> Result<(), L1AccessError> {
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

    fn preflight_dirty_eviction(&self, additional_line: Option<i64>) -> Result<(), L1AccessError> {
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

    fn evict_beyond_capacity(&mut self) -> Result<(), L1AccessError> {
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

    fn line_resident(&self, line: i64) -> bool {
        self.stamp_of.contains_key(&line)
    }

    fn validate_sector_mask(line: i64, sector_mask: u8) -> Result<(), L1AccessError> {
        if sector_mask == 0 || sector_mask & !(0b1111) != 0 {
            return Err(L1AccessError::InvalidationMaskInvalid { line, sector_mask });
        }
        Ok(())
    }

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

    fn validate_payload_residency_witnesses(
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

    /// Pending line/sector fills prevent invalidation even when they have no
    /// local timestamp. Event-mode callers must wait for response events,
    /// never mistake the absence of a timed deadline for an empty cache.
    pub fn has_pending_fills(&self) -> bool {
        !self.line_fills.is_empty() || !self.sector_fills.is_empty()
    }

    fn commit_due_fills(
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
    fn apply_fills(&mut self, now: u64) -> Result<(), L1AccessError> {
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

    /// One classification/state step shared by BOTH fill-timing modes —
    /// the single copy of the ported ladder (line-merge → sector-present
    /// hit → sector-merge → MSHR stall → fresh miss), with all counter
    /// accounting inside. [`FillRequest`] computes a hit timestamp only on a
    /// hit and a fill due state only on a fresh miss, so an unused timing arm
    /// cannot reject a valid merge, stall, or exact-domain result.
    fn access_core(
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

    fn access_single(
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

    fn access_eligible_constant_impl(
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

    fn access_eligible_constant_unobserved_impl(
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

    fn access_eligible_event_core(
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

    fn access_eligible_event_impl(
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

    fn access_eligible_event_unobserved_impl(
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


