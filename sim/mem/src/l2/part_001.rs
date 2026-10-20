// Lexical implementation section 2/2; included by the parent facade.
impl L2Model {
    pub fn new(cfg: L2Config) -> Result<Self, L2BuildError> {
        cfg.validate()?;
        let num_sets = cfg.num_sets_per_slice()?;
        let set_count = checked_set_count(cfg.num_slices, num_sets)?;
        let sets = allocate_set_storage(set_count)?;
        let owner = allocate_l2_model_owner()?;
        let arch = cfg.arch;
        Ok(Self {
            cfg,
            num_sets,
            sets,
            clock: 0,
            bip_counter: 0,
            counters: L2Counters::for_arch(arch),
            observer_lookup_hits: 0,
            observer_lookup_requests: 0,
            owner,
            next_generation: NextAllocationGeneration::Available(0),
            access_discipline: None,
        })
    }

    /// Stable structural projection for bounded differential diagnostics.
    /// Only occupied sets are emitted; empty geometry does not inflate a
    /// reference trace. Process-local owner identities are intentionally
    /// omitted because they do not affect modeled ordering.
    pub fn boundary_state(&self) -> String {
        let occupied = self
            .sets
            .iter()
            .enumerate()
            .filter(|(_, ways)| !ways.is_empty())
            .collect::<Vec<_>>();
        format!(
            "cfg={:?};num_sets={};clock={};bip_counter={};sets={:?};counters={:?};observer_lookup=({},{});next_generation={:?};discipline={:?}",
            self.cfg,
            self.num_sets,
            self.clock,
            self.bip_counter,
            occupied,
            self.counters,
            self.observer_lookup_hits,
            self.observer_lookup_requests,
            self.next_generation,
            self.access_discipline,
        )
    }

    /// Return a deterministic typed projection of the shared cache state.
    ///
    /// The cache model itself knows the line/set/way trajectory but does not
    /// know which SM response token produced or consumed a pending sector.
    /// Those requester edges are joined by `sim-sm` at the system boundary;
    /// this method therefore marks event-discipline ownership incomplete while
    /// still exposing every cache-resident field needed by that join.
    pub fn typed_boundary_state(&self) -> Result<L2BoundaryState, L2ProjectionError> {
        self.validate_counter_architecture()
            .map_err(|_| L2ProjectionError::Overflow)?;
        let sets_per_slice = self.num_sets;
        let mut sets = Vec::new();
        for (set_key, ways) in self.sets.iter().enumerate() {
            if ways.is_empty() {
                continue;
            }
            let set_key = u64::try_from(set_key).map_err(|_| L2ProjectionError::Overflow)?;
            let slice =
                u32::try_from(set_key / sets_per_slice).map_err(|_| L2ProjectionError::Overflow)?;
            let set = set_key % sets_per_slice;
            let mut projected_ways = Vec::with_capacity(ways.len());
            for way in ways {
                let line = CanonicalLineKey::VirtualAbsolute { line_id: way.tag };
                projected_ways.push(L2WayState {
                    line,
                    valid_mask: way.mask,
                    dirty_mask: way.dirty_mask,
                    pending_mask: way.pending_mask,
                    recency: way.recency,
                    generation: way.generation.map(|value| value.0),
                });
            }
            sets.push(L2SetState {
                slice,
                set,
                ways: projected_ways,
            });
        }

        let counters = self.counters;
        let counter_state = L2CounterState {
            hits: counters.hits,
            misses: counters.misses,
            lookup_hits: counters.lookup_hits,
            lookup_requests: counters.lookup_requests,
            line_misses: counters.line_misses,
            sector_fill_misses: counters.sector_fill_misses,
            write_tag_hits: counters.write_tag_hits,
        };
        let mut allocations = Vec::new();
        for set in &sets {
            for way in &set.ways {
                let Some(generation) = way.generation else {
                    continue;
                };
                let mut sectors = Vec::new();
                for sector in 0..L2_SECTORS_PER_LINE as u8 {
                    let bit = 1_u8 << sector;
                    let status = if way.pending_mask & bit != 0 {
                        Some(L2SectorStatus::Pending)
                    } else if way.dirty_mask & bit != 0 {
                        Some(L2SectorStatus::Dirty)
                    } else if way.valid_mask & bit != 0 {
                        Some(L2SectorStatus::Valid)
                    } else {
                        None
                    };
                    if let Some(status) = status {
                        sectors.push(L2SectorOwnership {
                            sector,
                            status,
                            producer: None,
                            consumers: Vec::new(),
                            ready_at: None,
                        });
                    }
                }
                allocations.push(L2AllocationState {
                    line: way.line.clone(),
                    generation,
                    sectors,
                });
            }
        }
        allocations.sort_by(|left, right| {
            left.line
                .cmp(&right.line)
                .then(left.generation.cmp(&right.generation))
        });
        let next_generation = match self.next_generation {
            NextAllocationGeneration::Available(value) => Some(value),
            NextAllocationGeneration::Exhausted => None,
        };
        let access_discipline = self.access_discipline.map(|discipline| match discipline {
            L2AccessDiscipline::Immediate => L2AccessDisciplineState::Immediate,
            L2AccessDiscipline::Event => L2AccessDisciplineState::Event,
        });
        let event_discipline = matches!(self.access_discipline, Some(L2AccessDiscipline::Event));
        let state = L2BoundaryState {
            schema_version: TYPED_STATE_SCHEMA_VERSION,
            architecture: architecture_name(self.cfg.arch).to_owned(),
            geometry: L2GeometryState {
                architecture: architecture_name(self.cfg.arch).to_owned(),
                capacity_bytes: self.cfg.capacity_bytes,
                line_bytes: self.cfg.line_bytes,
                sector_bytes: self.cfg.sector_bytes,
                num_slices: self.cfg.num_slices,
                sets_per_slice,
                associativity: self.cfg.associativity,
                replacement: replacement_name(self.cfg.replacement).to_owned(),
                slice_hash: slice_hash_name(self.cfg.slice_hash).to_owned(),
            },
            address_domain: AddressDomain::VirtualAbsolute,
            clock: self.clock,
            bip_counter: self.bip_counter,
            access_discipline,
            counters: counter_state,
            observer_lookup_hits: self.observer_lookup_hits,
            observer_lookup_requests: self.observer_lookup_requests,
            next_generation,
            sets,
            allocations,
            request_owners: Vec::new(),
            ownership_complete: !event_discipline,
            unknown_state: if event_discipline {
                vec!["requester_ownership_join_pending".to_owned()]
            } else {
                Vec::new()
            },
        };
        state.validate()?;
        Ok(state)
    }

    pub fn config(&self) -> &L2Config {
        &self.cfg
    }

    /// Return the read-only counter snapshot only when its architecture
    /// provenance still matches the owning cache configuration.
    pub fn counters(&self) -> Result<&L2Counters, L2CounterError> {
        self.validate_counter_architecture()?;
        Ok(&self.counters)
    }

    /// Replace the observer-only lookup totals after a shared-system replay has
    /// applied its source/request coalescing rule.  Modeled cache state,
    /// replacement metadata, and the `hits`/`misses` response-pricing counters
    /// are untouched.  The caller supplies cumulative totals so retained
    /// stream sessions remain monotonic across launch boundaries.
    pub fn replace_lookup_observer(
        &mut self,
        lookup_hits: u64,
        lookup_requests: u64,
    ) -> Result<(), L2AccessError> {
        self.validate_counter_architecture()?;
        if lookup_hits > lookup_requests {
            return Err(L2AccessError::Counter(
                L2CounterError::LookupHitsExceedRequests {
                    hits: lookup_hits,
                    requests: lookup_requests,
                },
            ));
        }
        // This API receives cumulative observer totals at a retained
        // multi-launch boundary.  A lower replacement would silently erase
        // history and make the next `checked_delta_since` underflow (or, if
        // consumed directly, misreport a negative launch contribution).
        // Reject both regressions before changing either field.
        if lookup_hits < self.observer_lookup_hits {
            return Err(L2AccessError::Counter(L2CounterError::Underflow {
                field: L2CounterField::LookupHits,
                current: lookup_hits,
                baseline: self.observer_lookup_hits,
            }));
        }
        if lookup_requests < self.observer_lookup_requests {
            return Err(L2AccessError::Counter(L2CounterError::Underflow {
                field: L2CounterField::LookupRequests,
                current: lookup_requests,
                baseline: self.observer_lookup_requests,
            }));
        }
        self.counters.lookup_hits = lookup_hits;
        self.counters.lookup_requests = lookup_requests;
        self.observer_lookup_hits = lookup_hits;
        self.observer_lookup_requests = lookup_requests;
        Ok(())
    }

    /// Record the source-side sectored fill-ownership split for one committed
    /// event access.  This is intentionally separate from the cache state's
    /// physical `hits`/`misses`: a pending merge remains a lookup miss while
    /// reusing an existing lower-level fetch, whereas a new fill owns a new
    /// lower-level fetch capability.  The caller supplies only counts from a
    /// post-commit [`L2EventMaskPreview`], so callback failure cannot publish
    /// a partial observer update.
    pub fn record_fill_ownership_observer(
        &mut self,
        pending_merge_sectors: u64,
        new_fill_sectors: u64,
    ) -> Result<(), L2AccessError> {
        self.validate_counter_architecture()?;
        let pending_merge_total = self
            .counters
            .pending_merge_sectors
            .checked_add(pending_merge_sectors)
            .ok_or(L2AccessError::CounterOverflow {
                field: L2CounterField::PendingMergeSectors,
            })?;
        let new_fill_total = self
            .counters
            .new_fill_sectors
            .checked_add(new_fill_sectors)
            .ok_or(L2AccessError::CounterOverflow {
                field: L2CounterField::NewFillSectors,
            })?;
        self.counters.pending_merge_sectors = pending_merge_total;
        self.counters.new_fill_sectors = new_fill_total;
        Ok(())
    }

    /// Number of event-mode sectors requested but not yet completed. A launch
    /// boundary may carry replacement/valid-sector state only when this is zero.
    pub fn pending_sector_count(&self) -> usize {
        self.sets
            .iter()
            .flat_map(|ways| ways.iter())
            .map(|way| way.pending_mask.count_ones() as usize)
            .sum()
    }

    /// Dirty sectors still resident after the currently committed access
    /// stream. The result is a terminal gauge, not a traffic counter.
    pub fn dirty_residency(&self) -> L2DirtyResidency {
        let mut result = L2DirtyResidency::default();
        for way in self.sets.iter().flatten() {
            if way.dirty_mask != 0 {
                result.lines += 1;
                result.sectors += way.dirty_mask.count_ones() as usize;
            }
        }
        result
    }

    /// Snapshot all resident dirty sectors in deterministic line order. This
    /// is the terminal-flush admission surface; it is distinct from eviction
    /// topology and does not clear any cache state.
    pub fn preview_dirty_flush(&self) -> L2DirtyFlushPlan {
        let mut lines = self
            .sets
            .iter()
            .flatten()
            .filter(|way| way.dirty_mask != 0)
            .map(|way| L2DirtyLine {
                line_id: way.tag,
                dirty_mask: way.dirty_mask,
            })
            .collect::<Vec<_>>();
        lines.sort_unstable_by_key(|line| line.line_id);
        L2DirtyFlushPlan {
            lines: lines.into_boxed_slice(),
        }
    }

    /// Clear exactly the dirty masks accepted by a lower-level writeback
    /// transaction. A stale plan leaves every line unchanged.
    pub fn commit_dirty_flush(&mut self, plan: L2DirtyFlushPlan) -> Result<(), L2AccessError> {
        let mut locations = Vec::with_capacity(plan.lines.len());
        for line in plan.lines() {
            let set_key = self.set_key(line.line_id)?;
            let Some(way) = self.sets[set_key]
                .iter()
                .position(|way| way.tag == line.line_id)
            else {
                return Err(L2AccessError::DirtyFlushStale {
                    line_id: line.line_id,
                    expected: line.dirty_mask,
                    current: 0,
                });
            };
            let current = self.sets[set_key][way].dirty_mask;
            if current != line.dirty_mask {
                return Err(L2AccessError::DirtyFlushStale {
                    line_id: line.line_id,
                    expected: line.dirty_mask,
                    current,
                });
            }
            locations.push((set_key, way));
        }
        for (set_key, way) in locations {
            self.sets[set_key][way].dirty_mask = 0;
        }
        Ok(())
    }

    fn validate_counter_architecture(&self) -> Result<(), L2CounterError> {
        require_counter_arch(self.cfg.arch, &self.counters)
    }

    /// Fallibly copy the complete cache state for a caller-level transaction.
    /// Spare vector capacity is retained so staging does not introduce an
    /// allocation on a path the caller's model had already reserved. No
    /// semantic state in `self` changes if either reservation fails.
    pub fn try_snapshot(&self) -> Result<Self, L2SnapshotError> {
        self.validate_counter_architecture()?;
        let set_count = self.sets.len();
        let set_capacity = self.sets.capacity();
        let mut sets = Vec::new();
        sets.try_reserve_exact(set_capacity).map_err(|source| {
            L2SnapshotError::SetStorageAllocation {
                set_count,
                capacity: set_capacity,
                source,
            }
        })?;
        for (set_key, ways) in self.sets.iter().enumerate() {
            let way_capacity = ways.capacity();
            let mut copied = Vec::new();
            copied.try_reserve_exact(way_capacity).map_err(|source| {
                L2SnapshotError::WayStorageAllocation {
                    set_key,
                    ways: ways.len(),
                    capacity: way_capacity,
                    source,
                }
            })?;
            copied.extend_from_slice(ways);
            sets.push(copied);
        }
        Ok(Self {
            cfg: self.cfg,
            num_sets: self.num_sets,
            sets,
            clock: self.clock,
            bip_counter: self.bip_counter,
            counters: self.counters,
            observer_lookup_hits: self.observer_lookup_hits,
            observer_lookup_requests: self.observer_lookup_requests,
            owner: self.owner,
            next_generation: self.next_generation,
            access_discipline: self.access_discipline,
        })
    }

    fn set_key(&self, line_id: i64) -> Result<usize, L2SetIndexError> {
        if self.cfg.num_slices == 0 {
            return Err(L2SetIndexError::SlicesZero);
        }
        if self.num_sets == 0 {
            return Err(L2SetIndexError::SetsPerSliceZero);
        }
        let signed_sets = i64::try_from(self.num_sets).map_err(|_| {
            L2SetIndexError::SetsPerSliceOutOfSignedRange {
                sets_per_slice: self.num_sets,
            }
        })?;
        let slice = self.cfg.slice_hash.slice_of(line_id, self.cfg.num_slices)?;
        let set_index_signed = line_id
            .div_euclid(i64::from(self.cfg.num_slices))
            .rem_euclid(signed_sets);
        let set_index =
            u64::try_from(set_index_signed).map_err(|_| L2SetIndexError::SetIndexOutOfRange {
                set_index: set_index_signed,
            })?;
        let set_key = u64::from(slice)
            .checked_mul(self.num_sets)
            .and_then(|base| base.checked_add(set_index))
            .ok_or(L2SetIndexError::SetKeyOverflow {
                slice,
                sets_per_slice: self.num_sets,
                set_index,
            })?;
        let set_key =
            usize::try_from(set_key).map_err(|_| L2SetIndexError::SetKeyOutOfRange { set_key })?;
        if set_key >= self.sets.len() {
            return Err(L2SetIndexError::SetKeyOutsideStorage {
                set_key,
                allocated_sets: self.sets.len(),
            });
        }
        Ok(set_key)
    }

    fn preflight_access_discipline(
        &self,
        requested: L2AccessDiscipline,
    ) -> Result<(), L2AccessError> {
        if let Some(latched) = self.access_discipline
            && latched != requested
        {
            return Err(L2AccessError::AccessDisciplineMismatch {
                latched: latched.name(),
                requested: requested.name(),
            });
        }
        Ok(())
    }

    fn prepare_mask(
        &mut self,
        line_id: i64,
        sector_mask: u8,
        class: L2OpClass,
    ) -> Result<PreparedMaskAccess, L2AccessError> {
        validate_sector_mask(sector_mask)?;
        self.validate_counter_architecture()?;
        self.preflight_access_discipline(L2AccessDiscipline::Immediate)?;
        let set_key = self.set_key(line_id)?;
        let associativity = usize::try_from(self.cfg.associativity).map_err(|_| {
            L2AccessError::AssociativityOutOfRange {
                associativity: self.cfg.associativity,
            }
        })?;
        let ways = &mut self.sets[set_key];
        if ways.len() > associativity {
            return Err(L2AccessError::SetOverfull {
                set_key,
                ways: ways.len(),
                associativity,
            });
        }
        validate_srrip_recencies(ways, set_key, self.cfg.replacement)?;
        validate_way_masks(ways, set_key)?;

        let resident = ways.iter().position(|way| way.tag == line_id);
        let mut update = resident.map(|way| PreparedSetUpdate::Update {
            way,
            value: ways[way],
        });
        let mut line_present = resident.is_some();
        let mut valid_mask = resident.map_or(0, |way| ways[way].mask);
        let line_present_before_packet = line_present;
        let mut clock = self.clock;
        let mut bip_counter = self.bip_counter;
        let mut counters = self.counters;
        let slice = usize::try_from(self.cfg.slice_hash.slice_of(line_id, self.cfg.num_slices)?)
            .map_err(|_| L2AccessError::CounterOverflow {
                field: L2CounterField::SliceAccesses,
            })?;
        if slice < counters.slice_accesses.len() {
            counters.slice_accesses[slice] = counters.slice_accesses[slice]
                .checked_add(u64::from(sector_mask.count_ones()))
                .ok_or(L2AccessError::CounterOverflow {
                    field: L2CounterField::SliceAccesses,
                })?;
        }
        let mut counted_mask = 0u8;
        let mut present_mask = 0u8;
        let mut pre_packet_write_tag_hit_mask = 0u8;

        for sector in 0..4u8 {
            let bit = 1u8 << sector;
            if sector_mask & bit == 0 {
                continue;
            }
            clock = clock
                .checked_add(1)
                .ok_or(L2AccessError::ClockOverflow { clock })?;
            let sector_present = line_present && valid_mask & bit != 0;
            let counted_hit = sector_present || (line_present && class == L2OpClass::WriteAtomic);
            // The sector lookup observer is distinct from physical write-tag
            // accounting: a write to a resident line's absent sector remains
            // a sector lookup miss.  Keep `counted_hit` for physical response
            // pricing and expose only present sectors here.
            let lookup_hit = sector_present;

            if !line_present {
                let next_bip_counter =
                    if self.cfg.replacement == L2Replacement::Bip {
                        Some(bip_counter.checked_add(1).ok_or(
                            L2AccessError::BipCounterOverflow {
                                counter: bip_counter,
                            },
                        )?)
                    } else {
                        None
                    };
                update = Some(prepare_install(
                    ways,
                    set_key,
                    line_id,
                    bit,
                    if class == L2OpClass::WriteAtomic {
                        bit
                    } else {
                        0
                    },
                    0,
                    None,
                    associativity,
                    self.cfg.replacement,
                    clock,
                    bip_counter,
                )?);
                let mut delta = L2Counters::for_arch(self.cfg.arch);
                delta.misses = 1;
                delta.line_misses = 1;
                delta.lookup_hits = u64::from(lookup_hit);
                delta.lookup_requests = 1;
                counters.add_checked(&delta)?;
                if let Some(next) = next_bip_counter {
                    bip_counter = next;
                }
            } else {
                let value = update
                    .as_mut()
                    .ok_or(L2AccessError::ReplacementSetEmpty { set_key })?
                    .value_mut();
                value.recency = match self.cfg.replacement {
                    L2Replacement::Srrip => 0,
                    _ => clock,
                };
                value.mask |= bit;
                if class == L2OpClass::WriteAtomic {
                    value.dirty_mask |= bit;
                }
                let mut delta = L2Counters::for_arch(self.cfg.arch);
                delta.lookup_hits = u64::from(lookup_hit);
                delta.lookup_requests = 1;
                if counted_hit {
                    delta.hits = 1;
                    if !sector_present && class == L2OpClass::WriteAtomic {
                        delta.write_tag_hits = 1;
                    }
                } else {
                    delta.misses = 1;
                    delta.sector_fill_misses = 1;
                }
                counters.add_checked(&delta)?;
            }

            if counted_hit {
                counted_mask |= bit;
                if class == L2OpClass::WriteAtomic && line_present_before_packet && !sector_present
                {
                    pre_packet_write_tag_hit_mask |= bit;
                }
            }
            if sector_present {
                present_mask |= bit;
            }
            line_present = true;
            valid_mask |= bit;
        }

        Ok(PreparedMaskAccess {
            set_key,
            update: update.ok_or(L2AccessError::ReplacementSetEmpty { set_key })?,
            clock,
            bip_counter,
            counters,
            next_generation: self.next_generation,
            discipline: L2AccessDiscipline::Immediate,
            preview: L2MaskPreview {
                sectors: sector_mask.count_ones(),
                counted_hits: counted_mask.count_ones(),
                present_sectors: present_mask.count_ones(),
                counted_hit_mask: counted_mask,
                present_mask,
                pre_packet_write_tag_hit_mask,
            },
        })
    }

    fn prepare_event_mask(
        &mut self,
        line_id: i64,
        sector_mask: u8,
        class: L2OpClass,
        pending_read_policy: PendingReadPolicy,
    ) -> Result<
        (
            PreparedMaskAccess,
            L2EventMaskPreview,
            Option<PreparedL2Fill>,
        ),
        L2AccessError,
    > {
        validate_sector_mask(sector_mask)?;
        self.validate_counter_architecture()?;
        self.preflight_access_discipline(L2AccessDiscipline::Event)?;
        let set_key = self.set_key(line_id)?;
        let associativity = usize::try_from(self.cfg.associativity).map_err(|_| {
            L2AccessError::AssociativityOutOfRange {
                associativity: self.cfg.associativity,
            }
        })?;
        let ways = &mut self.sets[set_key];
        if ways.len() > associativity {
            return Err(L2AccessError::SetOverfull {
                set_key,
                ways: ways.len(),
                associativity,
            });
        }
        validate_srrip_recencies(ways, set_key, self.cfg.replacement)?;
        validate_way_masks(ways, set_key)?;

        let resident = ways.iter().position(|way| way.tag == line_id);
        let mut update = resident.map(|way| PreparedSetUpdate::Update {
            way,
            value: ways[way],
        });
        let mut line_present = resident.is_some();
        let mut valid_mask = resident.map_or(0, |way| ways[way].mask);
        let line_present_before_packet = line_present;
        // Preserve the age boundary for this request. `pending_mask` is
        // mutated while the ascending sector walk constructs the new line
        // allocation; using that mutable value to classify a later sector
        // would make a fill opened by this request look like an older fill.
        // The coalescing endpoint is only valid for sectors pending before
        // this access began. Each sector bit is visited once today, but the
        // explicit snapshot keeps that protocol invariant local to the
        // classifier rather than relying on the walk shape.
        let preexisting_pending_mask = resident.map_or(0, |way| ways[way].pending_mask);
        let mut allocation_generation = match resident {
            Some(way) => Some(
                ways[way]
                    .generation
                    .ok_or(L2AccessError::EventAllocationGenerationMissing { line_id })?,
            ),
            None => None,
        };
        let mut next_generation = self.next_generation;
        let mut clock = self.clock;
        let mut bip_counter = self.bip_counter;
        let mut counters = self.counters;
        let slice = usize::try_from(self.cfg.slice_hash.slice_of(line_id, self.cfg.num_slices)?)
            .map_err(|_| L2AccessError::CounterOverflow {
                field: L2CounterField::SliceAccesses,
            })?;
        if slice < counters.slice_accesses.len() {
            counters.slice_accesses[slice] = counters.slice_accesses[slice]
                .checked_add(u64::from(sector_mask.count_ones()))
                .ok_or(L2AccessError::CounterOverflow {
                    field: L2CounterField::SliceAccesses,
                })?;
        }
        let mut counted_mask = 0u8;
        let mut present_mask = 0u8;
        let mut fill_mask = 0u8;
        let mut pending_merge_mask = 0u8;
        let mut pre_packet_write_tag_hit_mask = 0u8;

        for sector in 0..4u8 {
            let bit = 1u8 << sector;
            if sector_mask & bit == 0 {
                continue;
            }
            clock = clock
                .checked_add(1)
                .ok_or(L2AccessError::ClockOverflow { clock })?;
            let sector_present = line_present && valid_mask & bit != 0;
            let sector_pending = line_present && preexisting_pending_mask & bit != 0;
            let counted_hit = sector_present || (line_present && class == L2OpClass::WriteAtomic);
            // Pending read allocations remain lookup misses: they coalesce
            // onto an existing fill, but the requested sector is not yet
            // resident at tag lookup.  SM86 retains the historical effective
            // observer that credits pending sectors, while SM120's
            // operation-read observer is resident-sector scoped.  The generic
            // mixed-client endpoint remains a separate typed residual. Keep
            // this architecture gate local to the observer; `counted_hit`,
            // fill ownership, and cache state remain unchanged.
            let lookup_hit = if self.cfg.arch == ArchId::Sm120 {
                sector_present
            } else {
                sector_present || sector_pending
            };
            if !line_present {
                let next_bip_counter =
                    if self.cfg.replacement == L2Replacement::Bip {
                        Some(bip_counter.checked_add(1).ok_or(
                            L2AccessError::BipCounterOverflow {
                                counter: bip_counter,
                            },
                        )?)
                    } else {
                        None
                    };
                let (generation, next) = next_generation.allocate()?;
                next_generation = next;
                allocation_generation = Some(generation);
                let (initial_valid, initial_dirty, initial_pending) = match class {
                    L2OpClass::Read => (0, 0, bit),
                    L2OpClass::WriteAtomic => (bit, bit, 0),
                };
                update = Some(prepare_install(
                    ways,
                    set_key,
                    line_id,
                    initial_valid,
                    initial_dirty,
                    initial_pending,
                    Some(generation),
                    associativity,
                    self.cfg.replacement,
                    clock,
                    bip_counter,
                )?);
                let mut delta = L2Counters::for_arch(self.cfg.arch);
                delta.misses = 1;
                delta.line_misses = 1;
                delta.lookup_hits = u64::from(lookup_hit);
                delta.lookup_requests = 1;
                counters.add_checked(&delta)?;
                if let Some(next) = next_bip_counter {
                    bip_counter = next;
                }
            } else {
                let value = update
                    .as_mut()
                    .ok_or(L2AccessError::ReplacementSetEmpty { set_key })?
                    .value_mut();
                value.recency = match self.cfg.replacement {
                    L2Replacement::Srrip => 0,
                    _ => clock,
                };
                match class {
                    L2OpClass::Read => {
                        if !sector_present {
                            value.pending_mask |= bit;
                        }
                    }
                    L2OpClass::WriteAtomic => {
                        value.mask |= bit;
                        value.dirty_mask |= bit;
                        value.pending_mask &= !bit;
                    }
                }
                let mut delta = L2Counters::for_arch(self.cfg.arch);
                delta.lookup_hits = u64::from(lookup_hit);
                delta.lookup_requests = 1;
                if counted_hit {
                    delta.hits = 1;
                    if !sector_present && class == L2OpClass::WriteAtomic {
                        delta.write_tag_hits = 1;
                    }
                } else {
                    delta.misses = 1;
                    delta.sector_fill_misses = 1;
                }
                counters.add_checked(&delta)?;
            }

            if counted_hit {
                counted_mask |= bit;
                if class == L2OpClass::WriteAtomic && line_present_before_packet && !sector_present
                {
                    pre_packet_write_tag_hit_mask |= bit;
                }
            }
            if sector_present {
                present_mask |= bit;
            } else if class == L2OpClass::Read {
                if sector_pending
                    && pending_read_policy == PendingReadPolicy::CoalesceByAllocationSector
                {
                    pending_merge_mask |= bit;
                } else {
                    fill_mask |= bit;
                }
            }
            if class == L2OpClass::WriteAtomic {
                valid_mask |= bit;
            }
            line_present = true;
        }

        let generation = allocation_generation
            .ok_or(L2AccessError::EventAllocationGenerationMissing { line_id })?;
        let lookup = L2MaskPreview {
            sectors: sector_mask.count_ones(),
            counted_hits: counted_mask.count_ones(),
            present_sectors: present_mask.count_ones(),
            counted_hit_mask: counted_mask,
            present_mask,
            pre_packet_write_tag_hit_mask,
        };
        let event_preview = L2EventMaskPreview {
            lookup,
            owner: self.owner,
            line_id,
            generation,
            fill_mask,
            pending_merge_mask,
            eviction: update.as_ref().and_then(PreparedSetUpdate::eviction),
        };
        let prepared_fill = (fill_mask != 0).then_some(PreparedL2Fill {
            owner: self.owner,
            line_id,
            generation,
            sector_mask: fill_mask,
        });
        Ok((
            PreparedMaskAccess {
                set_key,
                update: update.ok_or(L2AccessError::ReplacementSetEmpty { set_key })?,
                clock,
                bip_counter,
                counters,
                next_generation,
                discipline: L2AccessDiscipline::Event,
                preview: lookup,
            },
            event_preview,
            prepared_fill,
        ))
    }

    /// Predict an ascending walk of `sector_mask` for one line without
    /// changing recency, replacement, validity, or counters. The local mask
    /// tracks sectors validated by earlier bits in the same request, exactly
    /// matching repeated [`Self::access`] calls in ascending sector order.
    pub fn preview_mask(
        &self,
        line_id: i64,
        sector_mask: u8,
        class: L2OpClass,
    ) -> Result<L2MaskPreview, L2AccessError> {
        validate_sector_mask(sector_mask)?;
        self.validate_counter_architecture()?;
        let set_key = self.set_key(line_id)?;
        let resident = self.sets[set_key].iter().find(|w| w.tag == line_id);
        let mut line_present = resident.is_some();
        let mut valid_mask = resident.map_or(0, |w| w.mask);
        let line_present_before_packet = line_present;
        let mut counted_mask = 0u8;
        let mut present_mask = 0u8;
        let mut pre_packet_write_tag_hit_mask = 0u8;
        for sector in 0..4u8 {
            let bit = 1u8 << sector;
            if sector_mask & bit == 0 {
                continue;
            }
            let sector_present = line_present && valid_mask & bit != 0;
            if sector_present {
                present_mask |= bit;
            }
            if sector_present || (line_present && class == L2OpClass::WriteAtomic) {
                counted_mask |= bit;
            }
            if class == L2OpClass::WriteAtomic && line_present_before_packet && !sector_present {
                pre_packet_write_tag_hit_mask |= bit;
            }
            line_present = true;
            valid_mask |= bit;
        }
        Ok(L2MaskPreview {
            sectors: sector_mask.count_ones(),
            counted_hits: counted_mask.count_ones(),
            present_sectors: present_mask.count_ones(),
            counted_hit_mask: counted_mask,
            present_mask,
            pre_packet_write_tag_hit_mask,
        })
    }

    /// One 32 B sector access in global request order. `line_id` is the
    /// global 128 B line address (`sector_id >> 2` in the L1's sector-id
    /// space); negative ids (wire-admissible) take Python's floored
    /// division/remainder. The STATE trajectory is identical for both
    /// [`L2OpClass`]es — the class only changes how a resident-line /
    /// invalid-sector access is COUNTED (Read: sector-fill miss, the
    /// ported Python rule; WriteAtomic: tag-match hit).
    pub fn access(
        &mut self,
        line_id: i64,
        sector: u8,
        class: L2OpClass,
    ) -> Result<L2Outcome, L2AccessError> {
        if sector >= 4 {
            return Err(L2AccessError::SectorOutOfRange { sector });
        }
        let sector_mask = 1u8 << sector;
        let (preview, ()) = self.access_mask_transactional(line_id, sector_mask, class, |_| {
            Ok::<(), L2AccessError>(())
        })?;
        Ok(L2Outcome {
            counted_hit: preview.counted_hits == 1,
            sector_present: preview.present_sectors == 1,
        })
    }

    /// Prepare one ascending mask walk, run all caller-owned fallible work
    /// against its exact outcome preview, and publish the cache changes only
    /// after the callback succeeds. A preparation or callback error leaves
    /// recency, validity, replacement metadata, and counters unchanged.
    pub fn access_mask_transactional<T, E>(
        &mut self,
        line_id: i64,
        sector_mask: u8,
        class: L2OpClass,
        before_commit: impl FnOnce(L2MaskPreview) -> Result<T, E>,
    ) -> Result<(L2MaskPreview, T), E>
    where
        E: From<L2AccessError>,
    {
        let prepared = self
            .prepare_mask(line_id, sector_mask, class)
            .map_err(E::from)?;
        let preview = prepared.preview;
        let value = before_commit(preview)?;
        prepared.commit(self);
        Ok((preview, value))
    }

    /// Event-discipline counterpart to [`Self::access_mask_transactional`].
    /// Read misses allocate/tag the line and set pending bits but remain absent
    /// from physical-presence and hit accounting until the returned opaque fill
    /// capability is completed. This compatibility surface keeps duplicate
    /// pending reads as distinct lower-level fetches. The callback receives only
    /// a non-completable preview; a fill token is minted after both the callback
    /// and cache commit succeed. Writes validate their sectors immediately.
    pub fn access_mask_event_transactional<T, E>(
        &mut self,
        line_id: i64,
        sector_mask: u8,
        class: L2OpClass,
        before_commit: impl FnOnce(L2EventMaskPreview) -> Result<T, E>,
    ) -> Result<(L2EventMaskAccess, T), E>
    where
        E: From<L2AccessError>,
    {
        self.access_mask_event_with_policy(
            line_id,
            sector_mask,
            class,
            PendingReadPolicy::DistinctFetches,
            before_commit,
        )
    }

    /// Event-mode read access that merges a request for an already-pending sector
    /// onto the matching allocation-generation/sector. The lookup still counts a
    /// miss, but [`L2EventMaskPreview::pending_merge_mask`] identifies sectors
    /// that must reuse the original lower-level ready time instead of issuing a
    /// duplicate fetch. Preparation and callback failure are transactional.
    pub fn access_mask_event_coalesced_transactional<T, E>(
        &mut self,
        line_id: i64,
        sector_mask: u8,
        class: L2OpClass,
        before_commit: impl FnOnce(L2EventMaskPreview) -> Result<T, E>,
    ) -> Result<(L2EventMaskAccess, T), E>
    where
        E: From<L2AccessError>,
    {
        self.access_mask_event_with_policy(
            line_id,
            sector_mask,
            class,
            PendingReadPolicy::CoalesceByAllocationSector,
            before_commit,
        )
    }

    fn access_mask_event_with_policy<T, E>(
        &mut self,
        line_id: i64,
        sector_mask: u8,
        class: L2OpClass,
        pending_read_policy: PendingReadPolicy,
        before_commit: impl FnOnce(L2EventMaskPreview) -> Result<T, E>,
    ) -> Result<(L2EventMaskAccess, T), E>
    where
        E: From<L2AccessError>,
    {
        let (prepared, preview, prepared_fill) = self
            .prepare_event_mask(line_id, sector_mask, class, pending_read_policy)
            .map_err(E::from)?;
        let value = before_commit(preview)?;
        prepared.commit(self);
        let access = L2EventMaskAccess {
            lookup: preview.lookup,
            fill: prepared_fill.map(PreparedL2Fill::issue),
            eviction: preview.eviction,
        };
        Ok((access, value))
    }

    /// Complete every sector authorized by one event read capability.
    pub fn complete_fill(&mut self, token: L2FillToken) -> Result<L2FillCompletion, L2AccessError> {
        self.complete_fill_mask(token, token.sector_mask)
    }

    /// Complete a nonempty subset of one event read capability. Matching
    /// completions are idempotent; eviction or same-tag reallocation returns a
    /// stale no-op and can never resurrect the old allocation.
    pub fn complete_fill_mask(
        &mut self,
        token: L2FillToken,
        completion_mask: u8,
    ) -> Result<L2FillCompletion, L2AccessError> {
        self.validate_counter_architecture()?;
        self.preflight_access_discipline(L2AccessDiscipline::Event)?;
        if token.owner != self.owner {
            return Err(L2AccessError::FillTokenOwnerMismatch);
        }
        if completion_mask == 0
            || completion_mask & !0b1111 != 0
            || completion_mask & !token.sector_mask != 0
        {
            return Err(L2AccessError::FillCompletionMaskInvalid {
                completion_mask,
                token_mask: token.sector_mask,
            });
        }

        let set_key = self.set_key(token.line_id)?;
        validate_way_masks(&self.sets[set_key], set_key)?;
        let Some(way) = self.sets[set_key]
            .iter_mut()
            .find(|way| way.tag == token.line_id)
        else {
            return Ok(L2FillCompletion::Stale);
        };
        match way.generation {
            Some(generation) if generation == token.generation => {}
            Some(_) => return Ok(L2FillCompletion::Stale),
            None => {
                return Err(L2AccessError::EventAllocationGenerationMissing {
                    line_id: token.line_id,
                });
            }
        }
        if way.mask & way.pending_mask != 0 || completion_mask & !(way.mask | way.pending_mask) != 0
        {
            return Err(L2AccessError::FillCompletionStateInvalid {
                line_id: token.line_id,
                completion_mask,
                valid_mask: way.mask,
                pending_mask: way.pending_mask,
            });
        }

        let newly_valid_mask = completion_mask & way.pending_mask;
        let duplicate_mask = completion_mask & way.mask;
        way.mask |= newly_valid_mask;
        way.pending_mask &= !newly_valid_mask;
        Ok(L2FillCompletion::Completed {
            newly_valid_mask,
            duplicate_mask,
        })
    }
}

struct PreparedMaskAccess {
    set_key: usize,
    update: PreparedSetUpdate,
    clock: i64,
    bip_counter: u64,
    counters: L2Counters,
    next_generation: NextAllocationGeneration,
    discipline: L2AccessDiscipline,
    preview: L2MaskPreview,
}

impl PreparedMaskAccess {
    fn commit(self, l2: &mut L2Model) {
        self.update.commit(&mut l2.sets[self.set_key]);
        l2.clock = self.clock;
        l2.bip_counter = self.bip_counter;
        l2.counters = self.counters;
        l2.next_generation = self.next_generation;
        if l2.access_discipline.is_none() {
            l2.access_discipline = Some(self.discipline);
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum PreparedSetUpdate {
    Update {
        way: usize,
        value: Way,
    },
    Insert {
        value: Way,
    },
    Replace {
        victim: usize,
        srrip_age: i64,
        evicted: Way,
        value: Way,
    },
}

impl PreparedSetUpdate {
    fn value_mut(&mut self) -> &mut Way {
        match self {
            Self::Update { value, .. } | Self::Insert { value } | Self::Replace { value, .. } => {
                value
            }
        }
    }

    fn eviction(&self) -> Option<L2Eviction> {
        let Self::Replace { evicted, .. } = self else {
            return None;
        };
        Some(L2Eviction {
            line_id: evicted.tag,
            valid_mask: evicted.mask,
            dirty_mask: evicted.dirty_mask,
            pending_mask: evicted.pending_mask,
        })
    }

    fn commit(self, ways: &mut Vec<Way>) {
        match self {
            Self::Update { way, value } => ways[way] = value,
            Self::Insert { value } => ways.push(value),
            Self::Replace {
                victim,
                srrip_age,
                evicted: _,
                value,
            } => {
                if srrip_age != 0 {
                    for way in ways.iter_mut() {
                        way.recency += srrip_age;
                    }
                }
                ways[victim] = value;
            }
        }
    }
}

fn validate_sector_mask(sector_mask: u8) -> Result<(), L2AccessError> {
    if sector_mask == 0 || sector_mask & !0b1111 != 0 {
        return Err(L2AccessError::SectorMaskInvalid { sector_mask });
    }
    Ok(())
}

fn validate_srrip_recencies(
    ways: &[Way],
    set_key: usize,
    policy: L2Replacement,
) -> Result<(), L2AccessError> {
    if policy == L2Replacement::Srrip {
        for (way, entry) in ways.iter().enumerate() {
            if !(0..=SRRIP_RRPV_MAX).contains(&entry.recency) {
                return Err(L2AccessError::SrripRecencyOutOfRange {
                    set_key,
                    way,
                    recency: entry.recency,
                });
            }
        }
    }
    Ok(())
}

fn validate_way_masks(ways: &[Way], set_key: usize) -> Result<(), L2AccessError> {
    for (way, entry) in ways.iter().enumerate() {
        if entry.dirty_mask & !entry.mask != 0 || entry.dirty_mask & entry.pending_mask != 0 {
            return Err(L2AccessError::DirtyMaskInvalid {
                set_key,
                way,
                valid_mask: entry.mask,
                dirty_mask: entry.dirty_mask,
                pending_mask: entry.pending_mask,
            });
        }
    }
    Ok(())
}

/// `_install_way` port: prepare insertion or replacement without changing
/// semantic set state. A spare way slot is reserved before the caller's
/// pre-commit callback so the later insertion cannot allocate.
#[allow(clippy::too_many_arguments)]
fn prepare_install(
    ways: &mut Vec<Way>,
    set_key: usize,
    tag: i64,
    valid_mask: u8,
    dirty_mask: u8,
    pending_mask: u8,
    generation: Option<L2AllocationGeneration>,
    associativity: usize,
    policy: L2Replacement,
    clock: i64,
    bip_counter: u64,
) -> Result<PreparedSetUpdate, L2AccessError> {
    if ways.len() > associativity {
        return Err(L2AccessError::SetOverfull {
            set_key,
            ways: ways.len(),
            associativity,
        });
    }
    if ways.len() < associativity {
        let recency = insertion_recency(ways, set_key, policy, clock, bip_counter)?;
        ways.try_reserve_exact(1)
            .map_err(|source| L2AccessError::SetStorageAllocation {
                set_key,
                additional: 1,
                source,
            })?;
        return Ok(PreparedSetUpdate::Insert {
            value: Way {
                tag,
                mask: valid_mask,
                dirty_mask,
                pending_mask,
                generation,
                recency,
            },
        });
    }
    if ways.is_empty() {
        return Err(L2AccessError::ReplacementSetEmpty { set_key });
    }

    let (victim, srrip_age) = if policy == L2Replacement::Srrip {
        let maximum = ways
            .iter()
            .map(|way| way.recency)
            .max()
            .ok_or(L2AccessError::ReplacementSetEmpty { set_key })?;
        let age =
            SRRIP_RRPV_MAX
                .checked_sub(maximum)
                .ok_or(L2AccessError::SrripRecencyOutOfRange {
                    set_key,
                    way: 0,
                    recency: maximum,
                })?;
        let victim = ways
            .iter()
            .position(|way| way.recency == maximum)
            .ok_or(L2AccessError::ReplacementSetEmpty { set_key })?;
        (victim, age)
    } else {
        let victim = ways
            .iter()
            .enumerate()
            .min_by_key(|(index, way)| (way.recency, *index))
            .map(|(index, _)| index)
            .ok_or(L2AccessError::ReplacementSetEmpty { set_key })?;
        (victim, 0)
    };
    let recency = insertion_recency(ways, set_key, policy, clock, bip_counter)?;
    let evicted = ways[victim];
    Ok(PreparedSetUpdate::Replace {
        victim,
        srrip_age,
        evicted,
        value: Way {
            tag,
            mask: valid_mask,
            dirty_mask,
            pending_mask,
            generation,
            recency,
        },
    })
}

fn insertion_recency(
    ways: &[Way],
    set_key: usize,
    policy: L2Replacement,
    clock: i64,
    bip_counter: u64,
) -> Result<i64, L2AccessError> {
    match policy {
        L2Replacement::Srrip => Ok(SRRIP_INSERT_RRPV),
        L2Replacement::Bip => {
            let mru = bip_counter.is_multiple_of(BIP_MRU_INSERT_PERIOD);
            if mru || ways.is_empty() {
                Ok(clock)
            } else {
                let minimum = ways
                    .iter()
                    .map(|way| way.recency)
                    .min()
                    .ok_or(L2AccessError::ReplacementSetEmpty { set_key })?;
                minimum
                    .checked_sub(1)
                    .ok_or(L2AccessError::BipRecencyUnderflow { set_key, minimum })
            }
        }
        L2Replacement::Lru => Ok(clock),
    }
}

