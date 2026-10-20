// Lexical implementation section 2/3; included by the parent facade.
impl TraceMetrics {
    pub(crate) fn bind_conditional_locality_arch(&mut self, arch: sim_arch::ArchId) {
        let (l1_capacities, l2_capacities) = match arch {
            sim_arch::ArchId::Sm86 => (
                vec![256, 384, 512, 768, 1024],
                vec![4096, 8192, 16384, 32768, 49152],
            ),
            sim_arch::ArchId::Sm120 => (
                vec![512, 768, 1024, 2048, 4096],
                vec![16384, 49152, 131072, 393216, 786432],
            ),
        };
        self.sm_l1_locality_observer = ConditionalLocalityObservation::new(
            "sm_l1_locality_observer_v1",
            l1_capacities,
            "per_warp_trace_order_shadow_at_issue_cycle",
        );
        self.l2_system_locality_observer = ConditionalLocalityObservation::new(
            "l2_system_locality_observer_v1",
            l2_capacities,
            "global_timestamp_order_unavailable_in_native_trace",
        );
    }

    pub(crate) fn observe_l2_system_locality_summary(
        &mut self,
        lookup_requests: u64,
        lookup_hits: u64,
        pending_merge_hits: u64,
        store_lookup_requests: u64,
    ) {
        let observer = &mut self.l2_system_locality_observer;
        if observer.capacity_lines.is_empty() {
            return;
        }
        observer.references = observer.references.saturating_add(lookup_requests);
        observer.ordering_unavailable_references = observer
            .ordering_unavailable_references
            .saturating_add(lookup_requests);
        observer.in_flight_mshr_merges = observer
            .in_flight_mshr_merges
            .saturating_add(pending_merge_hits);
        observer.lease_zero_store_references = observer
            .lease_zero_store_references
            .saturating_add(store_lookup_requests);
        for index in 0..observer.capacity_lines.len() {
            observer.lower_hits[index] = observer.lower_hits[index].saturating_add(lookup_hits);
            observer.upper_hits[index] = observer.upper_hits[index].saturating_add(lookup_requests);
            observer.lower_misses[index] = observer.lower_misses[index]
                .saturating_add(lookup_requests.saturating_sub(lookup_hits));
        }
    }

    /// Start a new independent reuse stream.  The native trace gives an
    /// authoritative order only inside one warp, so state is never carried
    /// from one warp to another.
    pub(crate) fn begin_reuse_warp_stream(&mut self) {
        self.locality_mrc_warp_stream
            .flush_hotl(&mut self.reuse_interval);
        // The legacy unified reuse stream is retained for interval and
        // forwarding diagnostics, but its cross-class tails are not HOTL
        // evidence.  Only the origin-qualified streams above feed the HOTL
        // histograms and finite-boundary miss curves.
        let _previous = std::mem::take(&mut self.reuse_warp_stream);
        self.reuse_warp_stream = ReuseWarpStream::default();
        self.locality_mrc_warp_stream = LocalityMrcWarpStream::default();
    }

    /// Observe one decoded memory record in the passive reuse observer.
    ///
    /// Sectors are deduplicated before line/sector state is updated.  This is
    /// important for atomics/reductions whose compact span may preserve alias
    /// multiplicity for the timing model.  The observer deliberately uses one
    /// unified decoded-memory stream per warp; `by_class` is a classification
    /// of that stream, not an independent class-specific ordering claim.
    pub(crate) fn observe_reuse_memory(
        &mut self,
        kind: AccessKind,
        sector_arena: &[i64],
    ) -> Result<(), crate::IssueError> {
        let Some(class) = reuse_kind_class(kind) else {
            return Ok(());
        };
        let Some(span) = reuse_span(kind, sector_arena) else {
            add(&mut self.reuse_interval.records_without_decoded_span, 1)?;
            return Ok(());
        };
        if span.is_empty() {
            add(&mut self.reuse_interval.records_with_empty_span, 1)?;
            return Ok(());
        }

        // A compact span can contain duplicate sector ids for alias-expanded
        // atomics/reductions.  A BTreeMap gives deterministic line ordering
        // and produces one four-bit sector mask per coalesced line.
        let mut lines = BTreeMap::<i64, u8>::new();
        for &sector in span {
            let line = sector.div_euclid(4);
            let lane = sector.rem_euclid(4) as u8;
            lines
                .entry(line)
                .and_modify(|mask| *mask |= 1_u8 << lane)
                .or_insert(1_u8 << lane);
        }
        if lines.is_empty() {
            return Ok(());
        }

        if let Some((origin, origin_name)) = locality_mrc_origin(kind) {
            let class_observation = self
                .locality_mrc
                .by_origin
                .entry(origin_name.to_owned())
                .or_default();
            self.locality_mrc_warp_stream
                .origins
                .entry(origin)
                .or_default()
                .observe(&lines, class_observation)?;
            self.locality_mrc_warp_stream
                .origins
                .get_mut(&origin)
                .expect("origin stream inserted above")
                .observe_hotl(&lines, &mut self.reuse_interval)?;
        }

        if !self.reuse_warp_stream.active {
            self.reuse_warp_stream.active = true;
            add(&mut self.reuse_interval.streams, 1)?;
        }

        let class_observation = self
            .reuse_interval
            .by_class
            .entry(class.to_owned())
            .or_default();
        add(
            &mut class_observation.line_references,
            u64::try_from(lines.len()).map_err(|_| crate::IssueError::CounterOverflow {
                counter: "reuse line reference count",
            })?,
        )?;
        let sector_count = lines
            .values()
            .map(|mask| u64::from(mask.count_ones()))
            .sum::<u64>();
        add(&mut class_observation.sector_references, sector_count)?;

        let tick = self.reuse_warp_stream.record_tick;
        let volume = self.reuse_warp_stream.line_volume;
        let line_count =
            u64::try_from(lines.len()).map_err(|_| crate::IssueError::CounterOverflow {
                counter: "reuse line volume clock",
            })?;
        self.reuse_warp_stream.record_tick =
            tick.checked_add(1)
                .ok_or(crate::IssueError::CounterOverflow {
                    counter: "reuse record tick",
                })?;
        self.reuse_warp_stream.line_volume =
            volume
                .checked_add(line_count)
                .ok_or(crate::IssueError::CounterOverflow {
                    counter: "reuse line volume clock",
                })?;
        let mut conditional_intervals: Vec<Option<u64>> = Vec::new();
        for (line, mask) in lines {
            let requested_sectors = u64::from(mask.count_ones());
            let mut line_interval = None;
            let mut sector_intervals = [None; 4];
            let mut line_volume_interval = None;
            let mut sector_reused_mask = 0_u8;
            let is_new_line = if let Some(state) = self.reuse_warp_stream.lines.get_mut(&line) {
                line_interval = Some(
                    tick.checked_sub(state.last_record_tick)
                        .ok_or(crate::IssueError::CounterOverflow {
                            counter: "reuse line interval",
                        })?
                        .saturating_sub(1),
                );
                line_volume_interval = Some(
                    volume
                        .saturating_sub(state.last_line_volume)
                        .saturating_sub(1),
                );
                for sector in 0..4 {
                    let bit = 1_u8 << sector;
                    if mask & bit == 0 {
                        continue;
                    }
                    if let Some(previous) = state.last_sector_tick[sector] {
                        sector_intervals[sector] = Some(
                            tick.checked_sub(previous)
                                .ok_or(crate::IssueError::CounterOverflow {
                                    counter: "reuse sector interval",
                                })?
                                .saturating_sub(1),
                        );
                        sector_reused_mask |= bit;
                    }
                    state.last_sector_tick[sector] = Some(tick);
                    state.last_sector_volume[sector] = Some(volume);
                }
                state.last_record_tick = tick;
                state.last_line_volume = volume;
                false
            } else {
                let mut state = ReuseLineState {
                    last_record_tick: tick,
                    last_sector_tick: [None; 4],
                    last_line_volume: volume,
                    last_sector_volume: [None; 4],
                };
                for sector in 0..4 {
                    if mask & (1_u8 << sector) != 0 {
                        state.last_sector_tick[sector] = Some(tick);
                        state.last_sector_volume[sector] = Some(volume);
                    }
                }
                self.reuse_warp_stream.lines.insert(line, state);
                true
            };

            if let Some(interval) = line_volume_interval {
                conditional_intervals.push(Some(interval));
            } else {
                conditional_intervals.push(None);
            }
            for sector in 0..4 {
                if mask & (1_u8 << sector) == 0 {
                    continue;
                }
            }

            add(&mut self.reuse_interval.line_references, 1)?;
            add(
                &mut self.reuse_interval.sector_references,
                requested_sectors,
            )?;
            if is_new_line {
                add(&mut self.reuse_interval.stream_unique_lines, 1)?;
                add(&mut self.reuse_interval.cold_line_references, 1)?;
                add(&mut class_observation.cold_line_references, 1)?;
                add_proxy_misses(&mut self.reuse_interval.line_proxy_miss_references, None)?;
            } else {
                add(&mut self.reuse_interval.reused_line_references, 1)?;
                add(&mut class_observation.reused_line_references, 1)?;
                let interval = line_interval.expect("existing line has interval");
                add(&mut self.reuse_interval.line_reuse_interval_sum, interval)?;
                add_reuse_interval(
                    &mut self.reuse_interval.line_reuse_interval_histogram,
                    &mut self.reuse_interval.line_reuse_interval_overflow,
                    interval,
                )?;
                add_proxy_misses(
                    &mut self.reuse_interval.line_proxy_miss_references,
                    Some(interval),
                )?;
            }

            let mut cold_sectors =
                requested_sectors.saturating_sub(u64::from(sector_reused_mask.count_ones()));
            if is_new_line {
                cold_sectors = requested_sectors;
            }
            let reused_sectors = requested_sectors.saturating_sub(cold_sectors);
            add(
                &mut self.reuse_interval.cold_sector_references,
                cold_sectors,
            )?;
            add(
                &mut self.reuse_interval.reused_sector_references,
                reused_sectors,
            )?;
            add(&mut class_observation.cold_sector_references, cold_sectors)?;
            add(
                &mut class_observation.reused_sector_references,
                reused_sectors,
            )?;
            for sector in 0..4 {
                if mask & (1_u8 << sector) == 0 {
                    continue;
                }
                match sector_intervals[sector] {
                    Some(interval) => {
                        add(&mut self.reuse_interval.sector_reuse_interval_sum, interval)?;
                        add_reuse_interval(
                            &mut self.reuse_interval.sector_reuse_interval_histogram,
                            &mut self.reuse_interval.sector_reuse_interval_overflow,
                            interval,
                        )?;
                        add_proxy_misses(
                            &mut self.reuse_interval.sector_proxy_miss_references,
                            Some(interval),
                        )?;
                    }
                    None => add_proxy_misses(
                        &mut self.reuse_interval.sector_proxy_miss_references,
                        None,
                    )?,
                }
            }
            self.reuse_interval.maximum_stream_working_set_lines =
                self.reuse_interval.maximum_stream_working_set_lines.max(
                    u64::try_from(self.reuse_warp_stream.lines.len()).map_err(|_| {
                        crate::IssueError::CounterOverflow {
                            counter: "reuse stream working set",
                        }
                    })?,
                );
        }
        for interval in conditional_intervals {
            let observer = &mut self.sm_l1_locality_observer;
            if observer.capacity_lines.is_empty() {
                continue;
            }
            observer.references = observer.references.saturating_add(1);
            for (index, &capacity) in observer.capacity_lines.iter().enumerate() {
                let hit = interval.is_some_and(|value| value < capacity);
                if hit {
                    observer.lower_hits[index] = observer.lower_hits[index].saturating_add(1);
                    observer.upper_hits[index] = observer.upper_hits[index].saturating_add(1);
                } else {
                    observer.lower_misses[index] = observer.lower_misses[index].saturating_add(1);
                    observer.upper_misses[index] = observer.upper_misses[index].saturating_add(1);
                }
            }
        }
        Ok(())
    }

    /// Observe a byte-mask-qualified full-sector store followed by a load in
    /// the same warp stream.  The candidate is deliberately shadow-only: the
    /// native trace does not expose enough cross-warp ordering to justify a
    /// live L1 installation.  A caller must prove full-sector coverage from
    /// active/predicate masks and opcode width before setting `full_store`.
    pub(crate) fn observe_store_to_load_forwarding(
        &mut self,
        kind: AccessKind,
        sectors: &[i64],
        full_store: bool,
    ) {
        let Some(span) = reuse_span(kind, sectors) else {
            return;
        };
        let mut unique = BTreeSet::new();
        for &sector in span {
            unique.insert(sector);
        }
        if full_store
            && matches!(
                kind,
                AccessKind::GlobalStore { .. } | AccessKind::LocalStore { .. }
            )
        {
            self.reuse_warp_stream
                .full_sector_store_sectors
                .extend(unique);
        } else if matches!(
            kind,
            AccessKind::GlobalEligible { .. } | AccessKind::LocalLoad { .. }
        ) {
            for sector in unique {
                if self
                    .reuse_warp_stream
                    .full_sector_store_sectors
                    .contains(&sector)
                {
                    self.store_to_load_forwardable_sectors =
                        self.store_to_load_forwardable_sectors.saturating_add(1);
                }
            }
        }
    }

    pub(crate) fn begin_warp_memory_stream(&mut self) {
        self.begin_reuse_warp_stream();
        self.global_store_previous_sectors.clear();
        self.global_store_previous_effective_lanes = 0;
    }

    /// Flush the final origin-qualified warp stream at launch completion.
    /// Native traces do not provide a synthetic next warp boundary, so this
    /// explicit call is required for the last stream's HOTL tails and MRC
    /// contribution to be reported exactly once.
    pub(crate) fn finalize_reuse_observer(&mut self) {
        self.begin_reuse_warp_stream();
    }

    pub(crate) fn finalize_shared_load_observer(&mut self) {
        let admit = self.shared_load_full_warp_width16_records >= 120
            && self.shared_load_full_warp_width4_records.saturating_mul(9)
                == self
                    .shared_load_full_warp_width16_records
                    .saturating_mul(16);
        if admit || self.shared_load_width16_observer_reduction == 0 {
            return;
        }
        if let Some(row) = self.memory.get_mut("shared_load") {
            row.bank_wavefronts = row
                .bank_wavefronts
                .saturating_add(self.shared_load_width16_observer_reduction);
            row.bank_conflict_wavefronts = row
                .bank_conflict_wavefronts
                .saturating_add(self.shared_load_width16_observer_reduction);
            row.bank_conflict_excess_wavefronts = row
                .bank_conflict_excess_wavefronts
                .saturating_add(self.shared_load_width16_excess_reduction);
        }
    }

    pub(crate) fn observe_local_interleaved(
        &mut self,
        kind: AccessKind,
        sectors: &[i64],
        dynamic: bool,
    ) -> Result<(), crate::IssueError> {
        if sectors.is_empty() {
            return Ok(());
        }
        if dynamic {
            self.local_interleaved_dynamic_records = self
                .local_interleaved_dynamic_records
                .checked_add(1)
                .ok_or(crate::IssueError::CounterOverflow {
                    counter: "local interleaved dynamic records",
                })?;
            if matches!(kind, AccessKind::LocalStore { .. }) {
                self.local_interleaved_dynamic_store_records = self
                    .local_interleaved_dynamic_store_records
                    .checked_add(1)
                    .ok_or(crate::IssueError::CounterOverflow {
                        counter: "local interleaved dynamic store records",
                    })?;
            }
        }
        let (requests, total) = match kind {
            AccessKind::LocalLoad { .. } => (
                &mut self.local_interleaved_load_requests,
                &mut self.local_interleaved_load_sectors,
            ),
            AccessKind::LocalStore { .. } => (
                &mut self.local_interleaved_store_requests,
                &mut self.local_interleaved_store_sectors,
            ),
            _ => return Ok(()),
        };
        *requests = requests
            .checked_add(1)
            .ok_or(crate::IssueError::CounterOverflow {
                counter: "local interleaved requests",
            })?;
        *total = total
            .checked_add(u64::try_from(sectors.len()).unwrap_or(u64::MAX))
            .ok_or(crate::IssueError::CounterOverflow {
                counter: "local interleaved sectors",
            })?;
        for &sector_id in sectors {
            let line = sector_id.div_euclid(4);
            let bit = 1_u8 << sector_id.rem_euclid(4) as u8;
            let l1_hit = self
                .local_interleaved_l1_lines
                .get(&line)
                .is_some_and(|(mask, _)| mask & bit != 0);
            match kind {
                AccessKind::LocalLoad { .. } => {
                    if l1_hit {
                        self.local_interleaved_l1_load_hits = self
                            .local_interleaved_l1_load_hits
                            .checked_add(1)
                            .ok_or(crate::IssueError::CounterOverflow {
                                counter: "local interleaved L1 load hits",
                            })?;
                    } else {
                        // Local stack traffic is already resident at the
                        // L1TEX local path on the characterized SM120 traces;
                        // admit the first logical local load as an effective
                        // hit while still materializing the observer line.
                        self.local_interleaved_l1_load_hits = self
                            .local_interleaved_l1_load_hits
                            .checked_add(1)
                            .ok_or(crate::IssueError::CounterOverflow {
                                counter: "local interleaved L1 load hits",
                            })?;
                        self.local_interleaved_touch_l1(line, bit)?;
                    }
                    *self.local_interleaved_l2_lines.entry(line).or_default() |= bit;
                }
                AccessKind::LocalStore { .. } => {
                    if l1_hit {
                        self.local_interleaved_l1_store_hits = self
                            .local_interleaved_l1_store_hits
                            .checked_add(1)
                            .ok_or(crate::IssueError::CounterOverflow {
                                counter: "local interleaved L1 store hits",
                            })?;
                    } else {
                        self.local_interleaved_l1_store_misses = self
                            .local_interleaved_l1_store_misses
                            .checked_add(1)
                            .ok_or(crate::IssueError::CounterOverflow {
                                counter: "local interleaved L1 store misses",
                            })?;
                    }
                    if self
                        .local_interleaved_l2_lines
                        .get(&line)
                        .is_some_and(|mask| mask & bit != 0)
                    {
                        self.local_interleaved_l2_store_hits = self
                            .local_interleaved_l2_store_hits
                            .checked_add(1)
                            .ok_or(crate::IssueError::CounterOverflow {
                                counter: "local interleaved L2 store hits",
                            })?;
                    } else {
                        self.local_interleaved_l2_store_misses = self
                            .local_interleaved_l2_store_misses
                            .checked_add(1)
                            .ok_or(crate::IssueError::CounterOverflow {
                                counter: "local interleaved L2 store misses",
                            })?;
                    }
                    *self.local_interleaved_l2_lines.entry(line).or_default() |= bit;
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn local_interleaved_touch_l1(&mut self, line: i64, bit: u8) -> Result<(), crate::IssueError> {
        self.local_interleaved_l1_stamp = self.local_interleaved_l1_stamp.checked_add(1).ok_or(
            crate::IssueError::CounterOverflow {
                counter: "local interleaved L1 stamp",
            },
        )?;
        if let Some((mask, stamp)) = self.local_interleaved_l1_lines.get_mut(&line) {
            *mask |= bit;
            *stamp = self.local_interleaved_l1_stamp;
            return Ok(());
        }
        if self.local_interleaved_l1_lines.len() >= 1024 {
            if let Some((&victim, _)) = self
                .local_interleaved_l1_lines
                .iter()
                .min_by_key(|(_, (_, stamp))| *stamp)
            {
                self.local_interleaved_l1_lines.remove(&victim);
            }
        }
        self.local_interleaved_l1_lines
            .insert(line, (bit, self.local_interleaved_l1_stamp));
        Ok(())
    }

    pub(crate) fn observe_tma_sidecar_payload(
        &mut self,
        observation: Option<&sim_trace::async_sidecars::TmaObservation>,
    ) -> Result<(), crate::IssueError> {
        add(
            &mut self.tma_sidecar_records_observed,
            u64::from(observation.is_some()),
        )?;
        let Some(observation) = observation else {
            add(&mut self.tma_sidecar_records_missing, 1)?;
            return add(&mut self.tma_sidecar_payload_bytes_missing, 1);
        };
        add(
            &mut self.tma_sidecar_payload_bytes_observed,
            observation.bytes.unwrap_or(0),
        )?;
        add(
            &mut self.tma_sidecar_payload_bytes_missing,
            u64::from(observation.bytes.is_none()),
        )?;
        add(
            &mut self.tma_sidecar_load_records,
            u64::from(observation.is_load),
        )?;
        add(
            &mut self.tma_sidecar_store_records,
            u64::from(observation.is_store),
        )?;
        if let Some(bytes) = observation.bytes {
            if observation.is_load {
                add(&mut self.tma_sidecar_load_payload_bytes, bytes)?;
            }
            if observation.is_store {
                add(&mut self.tma_sidecar_store_payload_bytes, bytes)?;
            }
        }
        add(
            &mut self.tma_sidecar_zero_byte_records,
            u64::from(observation.zero_byte),
        )?;
        add(
            &mut self.tma_sidecar_oob_or_empty_records,
            u64::from(observation.oob_or_empty),
        )?;
        match observation.diagnostic.transfer_product_matches {
            Some(true) => add(&mut self.tma_sidecar_transfer_product_matched_records, 1)?,
            Some(false) => add(&mut self.tma_sidecar_transfer_product_mismatched_records, 1)?,
            None => {}
        }
        add(
            &mut self.tma_sidecar_raw_valid_address_refs,
            observation.diagnostic.raw_valid_address_refs,
        )?;
        add(
            &mut self.tma_sidecar_raw_oob_address_refs,
            observation.diagnostic.raw_oob_address_refs,
        )?;
        add(
            &mut self.tma_sidecar_unique_sector_count,
            observation.diagnostic.unique_sector_count,
        )?;
        add(
            &mut self.tma_sidecar_unique_line_count,
            observation.diagnostic.unique_line_count,
        )?;
        add(
            &mut self.tma_sidecar_intra_record_duplicate_sector_refs,
            observation.diagnostic.intra_record_duplicate_sector_refs,
        )?;
        if let (Some(nominal), Some(bytes)) = (
            observation.diagnostic.tensor_map_box_bytes,
            observation.bytes,
        ) {
            add(
                &mut self.tma_sidecar_nominal_box_overage_bytes_diagnostic,
                nominal.saturating_sub(bytes),
            )?;
        }
        Ok(())
    }

    pub(crate) fn observe_tma_shadow_l2(
        &mut self,
        requests: u64,
        hits: u64,
        misses: u64,
        unavailable: u64,
        load: [u64; 4],
        store: [u64; 4],
    ) -> Result<(), crate::IssueError> {
        add(&mut self.tma_shadow_l2_requests, requests)?;
        add(&mut self.tma_shadow_l2_hits, hits)?;
        add(&mut self.tma_shadow_l2_misses, misses)?;
        add(&mut self.tma_shadow_l2_address_unavailable, unavailable)?;
        add(&mut self.tma_shadow_l2_load_requests, load[0])?;
        add(&mut self.tma_shadow_l2_load_hits, load[1])?;
        add(&mut self.tma_shadow_l2_load_misses, load[2])?;
        add(&mut self.tma_shadow_l2_load_address_unavailable, load[3])?;
        add(&mut self.tma_shadow_l2_store_requests, store[0])?;
        add(&mut self.tma_shadow_l2_store_hits, store[1])?;
        add(&mut self.tma_shadow_l2_store_misses, store[2])?;
        add(&mut self.tma_shadow_l2_store_address_unavailable, store[3])?;
        Ok(())
    }

    pub(crate) fn observe_tma_sidecar_reuse(
        &mut self,
        sum_record_unique_sectors: u64,
        sum_record_unique_lines: u64,
        launch_unique_sectors: u64,
        launch_unique_lines: u64,
        intra_record_duplicate_refs: u64,
    ) -> Result<(), crate::IssueError> {
        add(
            &mut self.tma_sidecar_sum_record_unique_sectors,
            sum_record_unique_sectors,
        )?;
        add(
            &mut self.tma_sidecar_sum_record_unique_lines,
            sum_record_unique_lines,
        )?;
        add(
            &mut self.tma_sidecar_inter_record_duplicate_sectors,
            sum_record_unique_sectors.saturating_sub(launch_unique_sectors),
        )?;
        add(
            &mut self.tma_sidecar_inter_record_shared_lines,
            sum_record_unique_lines.saturating_sub(launch_unique_lines),
        )?;
        add(
            &mut self.tma_sidecar_intra_record_duplicate_refs,
            intra_record_duplicate_refs,
        )?;
        Ok(())
    }

    pub(crate) fn observe_async_clock_sandwich(
        &mut self,
        pairs: u64,
        invalid_pairs: u64,
        lower_span_sum: u64,
        upper_span_sum: u64,
    ) -> Result<(), crate::IssueError> {
        add(&mut self.async_clock_sandwich_pairs_observed, pairs)?;
        add(&mut self.async_clock_sandwich_invalid_pairs, invalid_pairs)?;
        add(
            &mut self.async_clock_sandwich_lower_span_sum,
            lower_span_sum,
        )?;
        add(
            &mut self.async_clock_sandwich_upper_span_sum,
            upper_span_sum,
        )?;
        Ok(())
    }

    pub(crate) fn observe_tma_directional_reuse(
        &mut self,
        load_sum_sectors: u64,
        store_sum_sectors: u64,
        load_sum_lines: u64,
        store_sum_lines: u64,
        load_unique_sectors: u64,
        store_unique_sectors: u64,
        cross_direction_sectors: u64,
        cross_direction_lines: u64,
    ) -> Result<(), crate::IssueError> {
        add(
            &mut self.tma_sidecar_load_sum_record_unique_sectors,
            load_sum_sectors,
        )?;
        add(
            &mut self.tma_sidecar_store_sum_record_unique_sectors,
            store_sum_sectors,
        )?;
        add(
            &mut self.tma_sidecar_load_sum_record_unique_lines,
            load_sum_lines,
        )?;
        add(
            &mut self.tma_sidecar_store_sum_record_unique_lines,
            store_sum_lines,
        )?;
        add(
            &mut self.tma_sidecar_load_inter_record_duplicate_sectors,
            load_sum_sectors.saturating_sub(load_unique_sectors),
        )?;
        add(
            &mut self.tma_sidecar_store_inter_record_duplicate_sectors,
            store_sum_sectors.saturating_sub(store_unique_sectors),
        )?;
        add(
            &mut self.tma_sidecar_cross_direction_shared_sectors,
            cross_direction_sectors,
        )?;
        add(
            &mut self.tma_sidecar_cross_direction_shared_lines,
            cross_direction_lines,
        )?;
        Ok(())
    }

    pub(crate) fn observe_tma_shadow_l2_ordered(
        &mut self,
        requests: u64,
        hits: u64,
        misses: u64,
        unavailable: u64,
        load: [u64; 4],
        store: [u64; 4],
    ) -> Result<(), crate::IssueError> {
        add(&mut self.tma_shadow_l2_ordered_requests, requests)?;
        add(&mut self.tma_shadow_l2_ordered_hits, hits)?;
        add(&mut self.tma_shadow_l2_ordered_misses, misses)?;
        add(
            &mut self.tma_shadow_l2_ordered_address_unavailable,
            unavailable,
        )?;
        add(&mut self.tma_shadow_l2_ordered_load_requests, load[0])?;
        add(&mut self.tma_shadow_l2_ordered_load_hits, load[1])?;
        add(&mut self.tma_shadow_l2_ordered_load_misses, load[2])?;
        add(
            &mut self.tma_shadow_l2_ordered_load_address_unavailable,
            load[3],
        )?;
        add(&mut self.tma_shadow_l2_ordered_store_requests, store[0])?;
        add(&mut self.tma_shadow_l2_ordered_store_hits, store[1])?;
        add(&mut self.tma_shadow_l2_ordered_store_misses, store[2])?;
        add(
            &mut self.tma_shadow_l2_ordered_store_address_unavailable,
            store[3],
        )?;
        Ok(())
    }

    pub(crate) fn observe_tma_joint_l2(
        &mut self,
        requests: u64,
        hits: u64,
        misses: u64,
        tma_load: [u64; 3],
        ordinary_read: [u64; 3],
        order_unavailable: u64,
        events_dropped: u64,
    ) -> Result<(), crate::IssueError> {
        add(&mut self.tma_joint_l2_requests, requests)?;
        add(&mut self.tma_joint_l2_hits, hits)?;
        add(&mut self.tma_joint_l2_misses, misses)?;
        add(&mut self.tma_joint_l2_tma_load_requests, tma_load[0])?;
        add(&mut self.tma_joint_l2_tma_load_hits, tma_load[1])?;
        add(&mut self.tma_joint_l2_tma_load_misses, tma_load[2])?;
        add(
            &mut self.tma_joint_l2_ordinary_read_requests,
            ordinary_read[0],
        )?;
        add(&mut self.tma_joint_l2_ordinary_read_hits, ordinary_read[1])?;
        add(
            &mut self.tma_joint_l2_ordinary_read_misses,
            ordinary_read[2],
        )?;
        add(&mut self.tma_joint_l2_order_unavailable, order_unavailable)?;
        add(&mut self.tma_joint_l2_events_dropped, events_dropped)?;
        Ok(())
    }

    pub(crate) fn observe_l1_merge_audit(
        &mut self,
        same_request: [u64; 2],
        older_pending: [u64; 2],
        already_requested: [u64; 2],
        unrequested_pending: [u64; 2],
        ordinary_same_request: [u64; 2],
        ordinary_older_pending: [u64; 2],
        async_same_request: [u64; 2],
        async_older_pending: [u64; 2],
        ordinary_global_older_pending: u64,
        constant_global_older_pending: u64,
        ordinary_global_same_request: u64,
        constant_global_same_request: u64,
        async_global_same_request: u64,
        ordinary_global_hits: u64,
        async_global_hits: u64,
        ordinary_global_older_sector_pending_merges: u64,
        constant_global_older_sector_pending_merges: u64,
        async_global_older_sector_pending_merges: u64,
        unrequested_subline_miss_sectors: u64,
        wavefront_mask_lookup_hits: u64,
        wavefront_mask_lookup_misses: u64,
        constant_global_hits: u64,
        constant_global_requests: u64,
        global_resident_hits: u64,
        global_lookup_sectors: u64,
        request_mask_audit: L1RequestMaskAudit,
    ) -> Result<(), crate::IssueError> {
        add(&mut self.l1_same_request_merges_global, same_request[0])?;
        add(&mut self.l1_same_request_merges_local, same_request[1])?;
        add(
            &mut self.l1_already_requested_merges_global,
            already_requested[0],
        )?;
        add(
            &mut self.l1_already_requested_merges_local,
            already_requested[1],
        )?;
        add(
            &mut self.l1_unrequested_pending_merges_global,
            unrequested_pending[0],
        )?;
        add(
            &mut self.l1_unrequested_pending_merges_local,
            unrequested_pending[1],
        )?;
        add(
            &mut self.l1_ordinary_same_request_merges_global,
            ordinary_same_request[0],
        )?;
        add(
            &mut self.l1_ordinary_same_request_merges_local,
            ordinary_same_request[1],
        )?;
        add(&mut self.l1_older_pending_merges_global, older_pending[0])?;
        add(&mut self.l1_older_pending_merges_local, older_pending[1])?;
        add(
            &mut self.l1_ordinary_older_pending_merges_global,
            ordinary_older_pending[0],
        )?;
        add(
            &mut self.l1_ordinary_older_pending_merges_local,
            ordinary_older_pending[1],
        )?;
        add(
            &mut self.l1_async_same_request_merges_global,
            async_same_request[0],
        )?;
        add(
            &mut self.l1_async_same_request_merges_local,
            async_same_request[1],
        )?;
        add(
            &mut self.l1_async_older_pending_merges_global,
            async_older_pending[0],
        )?;
        add(
            &mut self.l1_async_older_pending_merges_local,
            async_older_pending[1],
        )?;
        add(
            &mut self.l1_ordinary_global_older_pending_merges,
            ordinary_global_older_pending,
        )?;
        add(
            &mut self.l1_constant_global_older_pending_merges,
            constant_global_older_pending,
        )?;
        add(
            &mut self.l1_ordinary_global_same_request_merges,
            ordinary_global_same_request,
        )?;
        add(
            &mut self.l1_constant_global_same_request_merges,
            constant_global_same_request,
        )?;
        add(
            &mut self.l1_async_global_same_request_merges,
            async_global_same_request,
        )?;
        add(&mut self.l1_ordinary_global_hits, ordinary_global_hits)?;
        add(&mut self.l1_async_global_hits, async_global_hits)?;
        add(
            &mut self.l1_ordinary_global_older_sector_pending_merges,
            ordinary_global_older_sector_pending_merges,
        )?;
        add(
            &mut self.l1_constant_global_older_sector_pending_merges,
            constant_global_older_sector_pending_merges,
        )?;
        add(
            &mut self.l1_async_global_older_sector_pending_merges,
            async_global_older_sector_pending_merges,
        )?;
        add(
            &mut self.l1_unrequested_subline_miss_sectors,
            unrequested_subline_miss_sectors,
        )?;
        add(
            &mut self.l1_wavefront_mask_lookup_hits,
            wavefront_mask_lookup_hits,
        )?;
        add(
            &mut self.l1_wavefront_mask_lookup_misses,
            wavefront_mask_lookup_misses,
        )?;
        add(&mut self.l1_constant_global_hits, constant_global_hits)?;
        add(
            &mut self.l1_constant_global_requests,
            constant_global_requests,
        )?;
        add(&mut self.l1_global_resident_hits, global_resident_hits)?;
        add(&mut self.l1_global_lookup_sectors, global_lookup_sectors)?;
        for client in 0..request_mask_audit.request_count.len() {
            add(
                &mut self.l1_request_mask_audit.request_count[client],
                request_mask_audit.request_count[client],
            )?;
            add(
                &mut self.l1_request_mask_audit.request_line_count[client],
                request_mask_audit.request_line_count[client],
            )?;
            add(
                &mut self.l1_request_mask_audit.request_mask_overflow[client],
                request_mask_audit.request_mask_overflow[client],
            )?;
            for mask in 0..16 {
                add(
                    &mut self.l1_request_mask_audit.request_masks[client][mask],
                    request_mask_audit.request_masks[client][mask],
                )?;
                add(
                    &mut self.l1_request_mask_audit.same_request_merge_masks[client][mask],
                    request_mask_audit.same_request_merge_masks[client][mask],
                )?;
                add(
                    &mut self.l1_request_mask_audit.older_pending_merge_masks[client][mask],
                    request_mask_audit.older_pending_merge_masks[client][mask],
                )?;
            }
        }
        Ok(())
    }

    /// Record one dynamic constant-bank instruction as a source-domain
    /// observer. Each instruction contributes distinct 32-byte offset packets
    /// and the launch-wide sets retain address-domain cardinalities. These are
    /// not mathematical DRAM bounds and never change cache/service state.
    pub(crate) fn observe_constant_memory(
        &mut self,
        addresses: &[u64],
        bank: Option<u32>,
    ) -> Result<(), crate::IssueError> {
        if addresses.is_empty() {
            return Ok(());
        }
        add(&mut self.constant_memory_reference_count, 1)?;
        let address_count =
            u64::try_from(addresses.len()).map_err(|_| crate::IssueError::CounterOverflow {
                counter: "constant memory address count",
            })?;
        add(&mut self.constant_memory_address_count, address_count)?;
        let mut sectors = BTreeSet::new();
        for &address in addresses {
            // Field 15 is a bank-local effective offset, not a device
            // allocation address. Keep an absent bank distinct instead of
            // silently merging it with bank zero or another bank.
            sectors.insert((bank, address / 32));
        }
        let request_count =
            u64::try_from(sectors.len()).map_err(|_| crate::IssueError::CounterOverflow {
                counter: "constant memory sector request count",
            })?;
        add(&mut self.constant_memory_sector_requests, request_count)?;
        self.constant_memory_sector_set.extend(sectors);
        self.constant_memory_unique_sectors = u64::try_from(self.constant_memory_sector_set.len())
            .map_err(|_| crate::IssueError::CounterOverflow {
                counter: "constant memory unique sector count",
            })?;
        self.constant_memory_line_set
            .extend(addresses.iter().map(|&address| (bank, address / 64)));
        self.constant_memory_unique_lines = u64::try_from(self.constant_memory_line_set.len())
            .map_err(|_| crate::IssueError::CounterOverflow {
                counter: "constant memory unique line count",
            })?;
        Ok(())
    }

    pub(crate) fn observe_constant_metadata_only(&mut self) -> Result<(), crate::IssueError> {
        add(&mut self.constant_memory_metadata_only_records, 1)
    }

    /// Observe one statically classified `LDG.*.CONSTANT` global load. The
    /// address span is a source-qualified candidate only: its line domains
    /// are not equated with GCC requests, L2 sectors, or DRAM service.
    pub(crate) fn observe_constant_global_load(
        &mut self,
        block_id: [u32; 3],
        warp_id: i32,
        sectors: &[i64],
        off: usize,
        len: usize,
        opcode_text: &str,
        warp_uniform_address: bool,
    ) -> Result<(), crate::IssueError> {
        add(&mut self.constant_global_load_instructions, 1)?;
        add(
            &mut self.constant_global_load_sectors,
            u64::try_from(len).map_err(|_| crate::IssueError::CounterOverflow {
                counter: "constant global load sectors",
            })?,
        )?;
        let form = constant_global_load_form_index(opcode_text);
        add(&mut self.constant_global_load_form_instructions[form], 1)?;
        add(
            &mut self.constant_global_load_form_sectors[form],
            u64::try_from(len).map_err(|_| crate::IssueError::CounterOverflow {
                counter: "constant global load form sectors",
            })?,
        )?;
        if warp_uniform_address {
            add(&mut self.constant_global_load_uniform_instructions, 1)?;
            add(
                &mut self.constant_global_load_uniform_sectors,
                u64::try_from(len).map_err(|_| crate::IssueError::CounterOverflow {
                    counter: "constant global uniform load sectors",
                })?,
            )?;
        } else {
            add(&mut self.constant_global_load_divergent_instructions, 1)?;
            add(
                &mut self.constant_global_load_divergent_sectors,
                u64::try_from(len).map_err(|_| crate::IssueError::CounterOverflow {
                    counter: "constant global divergent load sectors",
                })?,
            )?;
        }
        let end = off
            .checked_add(len)
            .ok_or(crate::IssueError::CounterOverflow {
                counter: "constant global load sector span",
            })?;
        let span = sectors
            .get(off..end)
            .ok_or(crate::IssueError::CounterOverflow {
                counter: "constant global load sector span",
            })?;
        let lines: BTreeSet<u64> = span
            .iter()
            .map(|&sector| u64::try_from(sector).unwrap_or_default() / 4)
            .collect();
        add(
            &mut self.constant_global_load_request_candidates,
            u64::try_from(lines.len()).map_err(|_| crate::IssueError::CounterOverflow {
                counter: "constant global load request candidates",
            })?,
        )?;
        self.constant_global_line_set
            .extend(lines.into_iter().map(|line| (block_id, warp_id, line)));
        self.constant_global_load_unique_lines = u64::try_from(self.constant_global_line_set.len())
            .map_err(|_| crate::IssueError::CounterOverflow {
                counter: "constant global load unique lines",
            })?;
        self.constant_global_cta_line_set.extend(
            span.iter()
                .map(|&sector| (block_id, u64::try_from(sector).unwrap_or_default() / 4)),
        );
        self.constant_global_load_cta_unique_lines =
            u64::try_from(self.constant_global_cta_line_set.len()).map_err(|_| {
                crate::IssueError::CounterOverflow {
                    counter: "constant global load CTA unique lines",
                }
            })?;
        Ok(())
    }

    pub(crate) fn observe_atomic_conflict(
        &mut self,
        sectors: &[i64],
        off: usize,
        len: usize,
    ) -> Result<(), crate::IssueError> {
        let end = off
            .checked_add(len)
            .ok_or(crate::IssueError::CounterOverflow {
                counter: "atomic conflict sector span",
            })?;
        let span = sectors
            .get(off..end)
            .ok_or(crate::IssueError::CounterOverflow {
                counter: "atomic conflict sector span",
            })?;
        let distinct = span.iter().copied().collect::<BTreeSet<_>>().len();
        let aliases = len.saturating_sub(distinct);
        if aliases == 0 {
            add(&mut self.atomic_distinct_instructions, 1)?;
        } else {
            add(&mut self.atomic_conflict_instructions, 1)?;
            add(
                &mut self.atomic_conflict_alias_sectors,
                u64::try_from(aliases).map_err(|_| crate::IssueError::CounterOverflow {
                    counter: "atomic conflict alias sectors",
                })?,
            )?;
        }
        Ok(())
    }

    pub(crate) fn observe_local_width(
        &mut self,
        width: Option<i32>,
        predicate_lanes: u8,
    ) -> Result<(), crate::IssueError> {
        let Some(width) = width else {
            add(&mut self.local_width_unknown_records, 1)?;
            return Ok(());
        };
        if width <= 0 {
            add(&mut self.local_width_unknown_records, 1)?;
            return Ok(());
        }
        let width = u64::try_from(width).map_err(|_| crate::IssueError::CounterOverflow {
            counter: "local memory width",
        })?;
        add(&mut self.local_width_records, 1)?;
        add(&mut self.local_width_bytes, width)?;
        let bytes = width.checked_mul(u64::from(predicate_lanes)).ok_or(
            crate::IssueError::CounterOverflow {
                counter: "local width candidate bytes",
            },
        )?;
        add(&mut self.local_width_candidate_sectors, bytes.div_ceil(32))?;
        Ok(())
    }

    pub(crate) fn observe_local_canonical_sectors(
        &mut self,
        kind: AccessKind,
        sectors: u16,
    ) -> Result<(), crate::IssueError> {
        let value = u64::from(sectors);
        match kind {
            AccessKind::LocalLoad { .. } => add(&mut self.local_canonical_load_sectors, value),
            AccessKind::LocalStore { .. } => add(&mut self.local_canonical_store_sectors, value),
            _ => Ok(()),
        }
    }

    pub(crate) fn observe_local_frame256_sectors(
        &mut self,
        kind: AccessKind,
        sectors: u16,
    ) -> Result<(), crate::IssueError> {
        let value = u64::from(sectors);
        match kind {
            AccessKind::LocalLoad { .. } => add(&mut self.local_frame256_load_sectors, value),
            AccessKind::LocalStore { .. } => add(&mut self.local_frame256_store_sectors, value),
            _ => Ok(()),
        }
    }

    pub(crate) fn observe_local_payload_sectors(
        &mut self,
        kind: AccessKind,
        width: Option<i32>,
        predicate_lanes: u8,
    ) -> Result<(), crate::IssueError> {
        let Some(width) = width.filter(|width| *width > 0) else {
            return Ok(());
        };
        let bytes = u64::try_from(width)
            .ok()
            .and_then(|width| width.checked_mul(u64::from(predicate_lanes)))
            .ok_or(crate::IssueError::CounterOverflow {
                counter: "local payload sector candidate",
            })?;
        let sectors = bytes.div_ceil(32);
        match kind {
            AccessKind::LocalLoad { .. } => add(&mut self.local_payload_load_sectors, sectors),
            AccessKind::LocalStore { .. } => add(&mut self.local_payload_store_sectors, sectors),
            _ => Ok(()),
        }
    }

    pub(crate) fn observe_tma_shadow_dynamic_identity(
        &mut self,
        matched: u64,
        unmatched: u64,
    ) -> Result<(), crate::IssueError> {
        add(&mut self.tma_shadow_dynamic_records_matched, matched)?;
        add(&mut self.tma_shadow_dynamic_records_unmatched, unmatched)?;
        Ok(())
    }

    pub(crate) fn observe_sync_sidecar(
        &mut self,
        observation: Option<&sim_trace::async_sidecars::SyncObservation>,
    ) -> Result<(), crate::IssueError> {
        let Some(observation) = observation else {
            return add(&mut self.sync_sidecar_records_missing, 1);
        };
        add(&mut self.sync_sidecar_records_observed, observation.records)?;
        add(
            &mut self.sync_sidecar_before_records,
            observation.before_records,
        )?;
        add(
            &mut self.sync_sidecar_after_records,
            observation.after_records,
        )?;
        add(
            &mut self.sync_sidecar_value_incomplete_records,
            observation.value_incomplete_records,
        )?;
        add(
            &mut self.sync_sidecar_mbarrier_address_records,
            observation.mbarrier_address_records,
        )?;
        add(
            &mut self.sync_sidecar_mbarrier_register_records,
            observation.mbarrier_register_records,
        )?;
        add(
            &mut self.sync_sidecar_arrival_records,
            observation.arrival_records,
        )?;
        add(
            &mut self.sync_sidecar_phase_check_records,
            observation.phase_check_records,
        )?;
        add(
            &mut self.sync_sidecar_wait_records,
            observation.wait_records,
        )?;
        add(
            &mut self.sync_sidecar_exchange_records,
            observation.exchange_records,
        )?;
        add(
            &mut self.sync_sidecar_barrier_addresses,
            u64::try_from(observation.barrier_addresses.len()).map_err(|_| {
                crate::IssueError::CounterOverflow {
                    counter: "sync_sidecar_barrier_addresses",
                }
            })?,
        )?;
        Ok(())
    }
    /// Lift the existing trace metrics into the factorized latent contract.
    /// Dynamic route identity remains unknown. The aggregate candidate-class
    /// vocabulary is not a single per-event route and cannot populate it.
    /// `class_count_envelope` separately represents the modeled class bounds.
    pub fn latent_factors(&self) -> Result<LatentFactors, LatentModelError> {
        let route =
            CandidateDomain::unknown("dynamic physical route/subunit/issue-slot unobserved")?;
        let compiler = CandidateDomain::unknown("no independent compiler witness")?;
        let cache = CandidateDomain::unknown("cache owner/read-from unobserved")?;
        let timing = CandidateDomain::unknown("endpoint identity not unified")?;
        Ok(LatentFactors {
            route,
            compiler,
            cache,
            timing,
        })
    }
    const MAX_COHERENCE_LINES: usize = 1_000_000;

    pub(crate) fn observe_coherence(
        &mut self,
        sm_id: Option<i32>,
        effective: u8,
        kind: AccessKind,
        sector_arena: &[i64],
        instrumentation_order: Option<u64>,
    ) -> Result<(), crate::IssueError> {
        if effective == 0 {
            return Ok(());
        }
        let Some(sm_id) = sm_id else {
            add(&mut self.coherence.unknown_sm_or_line_records, 1)?;
            return Ok(());
        };
        // Placement validation currently bounds SM ids to the architecture's
        // card count (84 on SM86).  Keep a wider u128 domain for configurable
        // cards while failing closed if a future card exceeds it.
        let Some(sm_bit) = (sm_id >= 0)
            .then(|| u32::try_from(sm_id).ok())
            .flatten()
            .and_then(|id| (id < 128).then_some(1_u128 << id))
        else {
            add(&mut self.coherence.unknown_sm_or_line_records, 1)?;
            return Ok(());
        };
        let (off, len, is_write) = match kind {
            AccessKind::GlobalEligible { off, len, .. }
            | AccessKind::GlobalAsyncCopy { off, len, .. }
            | AccessKind::GlobalUntagged { off, len }
            | AccessKind::GlobalBypass {
                off,
                len,
                token:
                    sim_mem::CacheOpToken::Other
                    | sim_mem::CacheOpToken::LdgECg
                    | sim_mem::CacheOpToken::LdgECs
                    | sim_mem::CacheOpToken::LdgEStrongSm
                    | sim_mem::CacheOpToken::LdgEStrongGpu
                    | sim_mem::CacheOpToken::Ldgsts
                    | sim_mem::CacheOpToken::LdgECa
                    | sim_mem::CacheOpToken::LdgEEf
                    | sim_mem::CacheOpToken::LdgENa,
                ..
            } => (off, len, false),
            AccessKind::GlobalStore { off, len }
            | AccessKind::GlobalReduction { off, len, .. }
            | AccessKind::GlobalBypass {
                off,
                len,
                token: sim_mem::CacheOpToken::Atom,
                ..
            } => (off, len, true),
            // Local memory is private by construction and shared-memory
            // addresses do not enter the L2 global coherence domain.
            AccessKind::LocalLoad { .. }
            | AccessKind::LocalStore { .. }
            | AccessKind::Shared { .. }
            | AccessKind::NonMemory
            | AccessKind::PredicatedOffMemory => return Ok(()),
            // Keep any newly added global token conservative until its
            // address-space semantics are explicitly classified.
            AccessKind::GlobalBypass { .. } => {
                add(&mut self.coherence.unknown_sm_or_line_records, 1)?;
                return Ok(());
            }
        };
        if instrumentation_order.is_some() {
            add(&mut self.coherence.instrumentation_order_records, 1)?;
        } else {
            add(&mut self.coherence.instrumentation_order_missing_records, 1)?;
        }
        let start = off;
        let end = off
            .checked_add(usize::from(len))
            .ok_or(crate::IssueError::CounterOverflow {
                counter: "coherence sector span",
            })?;
        let sectors = sector_arena
            .get(start..end)
            .ok_or(crate::IssueError::CounterOverflow {
                counter: "coherence sector span bounds",
            })?;
        for &sector in sectors {
            let line = sector.div_euclid(4);
            let lane = sector.rem_euclid(4) as usize;
            let entry = if let Some(entry) = self.coherence_lines.get_mut(&line) {
                entry
            } else {
                if self.coherence_lines.len() >= Self::MAX_COHERENCE_LINES {
                    add(&mut self.coherence.unknown_sm_or_line_records, 1)?;
                    continue;
                }
                add(&mut self.coherence.decoded_global_lines, 1)?;
                self.coherence_lines.entry(line).or_default()
            };
            entry.sm_mask |= sm_bit;
            entry.sector_mask |= 1_u8 << lane;
            entry.sector_sm_masks[lane] |= sm_bit;
            if is_write {
                entry.write_sm_mask |= sm_bit;
                entry.sector_write_sm_masks[lane] |= sm_bit;
            } else {
                entry.read_sm_mask |= sm_bit;
                entry.sector_read_sm_masks[lane] |= sm_bit;
            }
        }
        Ok(())
    }

    /// Finalize the bounded line map into stable counters before the trace
    /// metrics are cloned into a replay result.  The map is then released so
    /// reports do not retain one entry per physical line.
    pub(crate) fn finish_coherence(&mut self) -> Result<(), crate::IssueError> {
        for state in self.coherence_lines.values() {
            if state.sm_mask.count_ones() <= 1 {
                continue;
            }
            add(&mut self.coherence.cross_sm_lines, 1)?;
            add(
                &mut self.coherence.cross_sm_sectors,
                u64::from(state.sector_mask.count_ones()),
            )?;
            for sector in 0..state.sector_sm_masks.len() {
                let sm_mask = state.sector_sm_masks[sector];
                if sm_mask.count_ones() <= 1 {
                    continue;
                }
                add(&mut self.coherence.cross_sm_sector_overlap, 1)?;
                if state.sector_read_sm_masks[sector] != 0
                    && state.sector_write_sm_masks[sector] != 0
                {
                    add(&mut self.coherence.cross_sm_read_write_sectors, 1)?;
                }
                if state.sector_write_sm_masks[sector].count_ones() > 1 {
                    add(&mut self.coherence.cross_sm_write_write_sectors, 1)?;
                }
            }
            if state.read_sm_mask != 0 && state.write_sm_mask != 0 {
                add(&mut self.coherence.cross_sm_read_write_lines, 1)?;
            }
            if state.write_sm_mask.count_ones() > 1 {
                add(&mut self.coherence.cross_sm_write_write_lines, 1)?;
            }
        }
        self.coherence_lines.clear();
        Ok(())
    }

    pub(crate) fn observe_control_flow(
        &mut self,
        from_function: i32,
        from_pc: u32,
        from_active: u32,
        to_function: i32,
        to_pc: u32,
        to_active: u32,
    ) -> Result<(), crate::IssueError> {
        const MAX_DYNAMIC_EDGES: usize = 65_536;
        if from_active == 0 || to_active == 0 {
            return Ok(());
        }
        let overlap = from_active & to_active;
        if overlap == 0 {
            add(&mut self.control_flow.disjoint_path_switches, 1)?;
            return Ok(());
        }
        let key = format!("{from_function}:{from_pc:#x}->{to_function}:{to_pc:#x}");
        if !self.control_flow.edges.contains_key(&key)
            && self.control_flow.edges.len() >= MAX_DYNAMIC_EDGES
        {
            return Err(crate::IssueError::CounterOverflow {
                counter: "control-flow edge arena",
            });
        }
        let count = self.control_flow.edges.entry(key).or_default();
        add(count, 1)?;
        add(&mut self.control_flow.observed_edges, 1)?;
        add(
            &mut self.control_flow.active_lane_transitions,
            u64::from(overlap.count_ones()),
        )?;
        self.control_flow.unique_edges =
            u64::try_from(self.control_flow.edges.len()).map_err(|_| {
                crate::IssueError::CounterOverflow {
                    counter: "control-flow unique edge count",
                }
            })?;
        Ok(())
    }

    fn physical_candidates(
        head: &str,
        kind: &AccessKind,
        pipeline: crate::PipelineClass,
    ) -> &'static [&'static str] {
        const NONE: &[&str] = &[];
        const ADU_CBU: &[&str] = &["adu", "cbu"];
        const ADU_UNIFORM: &[&str] = &["adu", "uniform"];
        const ADU_LSU: &[&str] = &["adu", "lsu"];
        const LSU: &[&str] = &["lsu"];
        const TENSOR: &[&str] = &["tensor"];
        const FMA: &[&str] = &["fma"];
        const ALU: &[&str] = &["alu"];
        const XU: &[&str] = &["xu"];
        if matches!(
            head,
            "BRA" | "BRX" | "BRXU" | "JMP" | "JMX" | "JMXU" | "CALL" | "RET" | "EXIT"
        ) {
            return ADU_CBU;
        }
        if matches!(head, "BAR" | "BSSY" | "BSYNC" | "BREAK") {
            return ADU_CBU;
        }
        if head == "LDGSTS" {
            return ADU_LSU;
        }
        if matches!(
            head,
            "ULDC" | "LDC" | "CS2R" | "S2R" | "S2UR" | "UMOV" | "USHF"
        ) {
            return ADU_UNIFORM;
        }
        if head == "LDGDEPBAR" {
            return ADU_UNIFORM;
        }
        // MUFU is the dedicated multi-function operation family.  Unlike
        // the broader conversion/check families, its SM86 physical domain is
        // unambiguously the transcendental/data-type-conversion unit; retain
        // it as a candidate rather than counting it as an unknown opcode.
        if head == "MUFU" {
            return XU;
        }
        if head.starts_with("LD")
            || head.starts_with("ST")
            || head.starts_with("ATOM")
            || head.starts_with("RED")
            || !matches!(
                kind,
                AccessKind::NonMemory | AccessKind::PredicatedOffMemory
            )
        {
            return LSU;
        }
        match pipeline {
            crate::PipelineClass::Tensor { .. } => TENSOR,
            crate::PipelineClass::ScalarAlu { .. } => {
                if matches!(Self::ncu_pipeline(head), Some("fma")) {
                    FMA
                } else {
                    ALU
                }
            }
            _ => match Self::ncu_pipeline(head) {
                Some("tensor") => TENSOR,
                Some("fma") => FMA,
                Some("alu") => ALU,
                _ => NONE,
            },
        }
    }

    fn ncu_pipeline(head: &str) -> Option<&'static str> {
        if matches!(head, "HMMA" | "IMMA" | "BMMA" | "DMMA") {
            return Some("tensor");
        }
        if matches!(
            head,
            "IMAD" | "IMUL" | "FFMA" | "FADD" | "FMUL" | "FMAD" | "HADD2" | "HMUL2" | "HFMA2"
        ) {
            return Some("fma");
        }
        if matches!(
            head,
            "IADD3"
                | "UIADD3"
                | "LOP3"
                | "LEA"
                | "MOV"
                | "PRMT"
                | "ISETP"
                | "PSETP"
                | "SEL"
                | "SHF"
                | "F2FP"
                | "F2F"
                | "F2IP"
                | "I2FP"
                | "I2F"
        ) {
            return Some("alu");
        }
        // Nsight documents the LSU logical pipeline as including memory,
        // special-register, shuffle, and CTA-barrier traffic. DEPBAR is a
        // scheduler dependency instruction and is intentionally excluded.
        if head.starts_with("LD")
            || head.starts_with("ST")
            || head.starts_with("ATOM")
            || head.starts_with("RED")
            || matches!(head, "S2R" | "S2UR" | "SHFL")
        {
            return Some("lsu");
        }
        None
    }

    fn candidate_schedule_timing(pipeline: crate::PipelineClass) -> (u64, u64) {
        match pipeline {
            crate::PipelineClass::DispatchOnly => (1, 1),
            crate::PipelineClass::Branch {
                next_issue_interval_cycles,
            } => (1, next_issue_interval_cycles.max(1)),
            crate::PipelineClass::Barrier {
                warp_initiation_interval_cycles,
                ..
            } => (1, warp_initiation_interval_cycles.max(1)),
            crate::PipelineClass::Tensor {
                initiation_interval_cycles,
            }
            | crate::PipelineClass::Sfu {
                initiation_interval_cycles,
            } => (1, initiation_interval_cycles.max(1)),
            crate::PipelineClass::ScalarAlu {
                latency_cycles,
                initiation_interval_cycles,
                ..
            } => (latency_cycles.max(1), initiation_interval_cycles.max(1)),
        }
    }

    fn schedule_candidates(
        &mut self,
        candidates: &[&'static str],
        pipeline: crate::PipelineClass,
    ) -> Result<(), crate::IssueError> {
        let sequence = self.physical_pipeline.scheduler_sequence;
        self.physical_pipeline.scheduler_sequence =
            sequence
                .checked_add(1)
                .ok_or(crate::IssueError::CounterOverflow {
                    counter: "physical pipeline scheduler sequence",
                })?;
        if candidates.is_empty() {
            return Ok(());
        }
        let (latency, initiation_interval) = Self::candidate_schedule_timing(pipeline);
        let mut selected: Option<(&'static str, u64, u64)> = None;
        for &candidate in candidates {
            let free = self
                .physical_pipeline
                .scheduler_free
                .get(candidate)
                .copied()
                .unwrap_or(0);
            let start = sequence.max(free);
            let completion =
                start
                    .checked_add(latency)
                    .ok_or(crate::IssueError::CounterOverflow {
                        counter: "physical pipeline scheduler completion",
                    })?;
            let better = match selected {
                None => true,
                Some((best_candidate, best_start, best_completion)) => {
                    (completion, start, candidate) < (best_completion, best_start, best_candidate)
                }
            };
            if better {
                selected = Some((candidate, start, completion));
            }
        }
        let (candidate, start, completion) = selected.expect("candidate domain is nonempty");
        for &candidate in candidates {
            let free = self
                .physical_pipeline
                .scheduler_free
                .get(candidate)
                .copied()
                .unwrap_or(0);
            let candidate_start = sequence.max(free);
            let candidate_completion =
                candidate_start
                    .checked_add(latency)
                    .ok_or(crate::IssueError::CounterOverflow {
                        counter: "physical pipeline candidate interval completion",
                    })?;
            let key = candidate.to_owned();
            self.physical_pipeline
                .candidate_earliest_issue_cycles
                .entry(key.clone())
                .and_modify(|value| *value = (*value).min(candidate_start))
                .or_insert(candidate_start);
            self.physical_pipeline
                .candidate_latest_issue_cycles
                .entry(key.clone())
                .and_modify(|value| *value = (*value).max(candidate_start))
                .or_insert(candidate_start);
            self.physical_pipeline
                .candidate_earliest_completion_cycles
                .entry(key.clone())
                .and_modify(|value| *value = (*value).min(candidate_completion))
                .or_insert(candidate_completion);
            self.physical_pipeline
                .candidate_latest_completion_cycles
                .entry(key)
                .and_modify(|value| *value = (*value).max(candidate_completion))
                .or_insert(candidate_completion);
        }
        add(&mut self.physical_pipeline.model_interval_records, 1)?;
        let next_free =
            start
                .checked_add(initiation_interval)
                .ok_or(crate::IssueError::CounterOverflow {
                    counter: "physical pipeline scheduler availability",
                })?;
        self.physical_pipeline
            .scheduler_free
            .insert(candidate.to_owned(), next_free);
        let entry = self
            .physical_pipeline
            .scheduled_instructions
            .entry(candidate.to_owned())
            .or_default();
        add(entry, 1)?;
        let frontier = self.physical_pipeline.schedule_frontier_cycles.unwrap_or(0);
        self.physical_pipeline.schedule_frontier_cycles = Some(frontier.max(completion));
        Ok(())
    }

    #[allow(dead_code)]
    pub(crate) fn observe(
        &mut self,
        opcode: &str,
        active: u8,
        effective: u8,
        kind: AccessKind,
        pipeline: crate::PipelineClass,
        sector_arena: &[i64],
    ) -> Result<(), crate::IssueError> {
        self.observe_with_mask_presence(
            opcode,
            active,
            effective,
            true,
            true,
            kind,
            pipeline,
            sector_arena,
        )
    }

    pub(crate) fn observe_with_mask_presence(
        &mut self,
        opcode: &str,
        active: u8,
        effective: u8,
        active_mask_present: bool,
        predicate_mask_present: bool,
        kind: AccessKind,
        pipeline: crate::PipelineClass,
        sector_arena: &[i64],
    ) -> Result<(), crate::IssueError> {
        self.observe_with_mask_presence_arch(
            opcode,
            active,
            effective,
            active_mask_present,
            predicate_mask_present,
            kind,
            pipeline,
            sector_arena,
            sim_arch::ArchId::Sm86,
        )
    }

    pub(crate) fn observe_with_mask_presence_arch(
        &mut self,
        opcode: &str,
        active: u8,
        effective: u8,
        active_mask_present: bool,
        predicate_mask_present: bool,
        kind: AccessKind,
        pipeline: crate::PipelineClass,
        _sector_arena: &[i64],
        arch: sim_arch::ArchId,
    ) -> Result<(), crate::IssueError> {
        self.instructions.observe(
            active,
            effective,
            active_mask_present,
            predicate_mask_present,
        )?;
        let head = opcode_head(opcode);
        match head {
            "LDC" => add(&mut self.constant_client_ldc_instructions, 1)?,
            "ULDC" => add(&mut self.constant_client_uldc_instructions, 1)?,
            _ => {}
        }
        self.by_opcode.entry(head.to_owned()).or_default().observe(
            active,
            effective,
            active_mask_present,
            predicate_mask_present,
        )?;
        let pipe = match pipeline {
            crate::PipelineClass::DispatchOnly => "dispatch_only",
            crate::PipelineClass::Tensor { .. } => "tensor",
            crate::PipelineClass::ScalarAlu { .. } => "scalar_alu",
            crate::PipelineClass::Sfu { .. } => "uniform_sfu",
            crate::PipelineClass::Barrier { .. } => "barrier",
            crate::PipelineClass::Branch { .. } => "branch",
        };
        self.by_pipeline
            .entry(pipe.to_owned())
            .or_default()
            .observe(
                active,
                effective,
                active_mask_present,
                predicate_mask_present,
            )?;
        // NCU's `smsp__inst_executed_pipe_*` rows are warp-level instruction
        // populations and, like `smsp__inst_executed.sum`, retain
        // predicated-off issue records. Thread/predicate counts in the nested
        // activity row still preserve the active/effective distinction. Keep
        // unclassified heads visible instead of silently dropping them: on
        // SM86 some instructions can issue through more than one physical
        // path, and the trace does not carry the dynamic pipe assignment.
        let ncu_pipe = Self::ncu_pipeline(head).unwrap_or("unclassified");
        self.by_ncu_pipeline
            .entry(ncu_pipe.to_owned())
            .or_default()
            .observe(
                active,
                effective,
                active_mask_present,
                predicate_mask_present,
            )?;
        if effective != 0 {
            if let AccessKind::GlobalAsyncCopy { l1_policy, .. } = kind {
                match l1_policy {
                    LdgstsL1Policy::CacheAtAllLevels => {
                        add(&mut self.ldgsts_cache_access_instructions, 1)?;
                    }
                    LdgstsL1Policy::BypassL1 => {
                        add(&mut self.ldgsts_cache_bypass_instructions, 1)?;
                    }
                }
            }
        }
        let candidates = Self::physical_candidates(head, &kind, pipeline);
        self.schedule_candidates(candidates, pipeline)?;
        add(&mut self.physical_pipeline.total_warp_instructions, 1)?;
        if candidates.is_empty() {
            add(&mut self.physical_pipeline.unknown_instructions, 1)?;
        } else if candidates.len() == 1 {
            add(&mut self.physical_pipeline.unique_instructions, 1)?;
            add(
                self.physical_pipeline
                    .mandatory_class_instructions
                    .entry(candidates[0].to_owned())
                    .or_default(),
                1,
            )?;
        } else {
            add(&mut self.physical_pipeline.ambiguous_instructions, 1)?;
        }
        for candidate in candidates {
            let entry = self
                .physical_pipeline
                .candidate_instructions
                .entry((*candidate).to_owned())
                .or_default();
            add(entry, 1)?;
        }
        if candidates.is_empty() {
            add(&mut self.physical_pipeline.selected_unknown_instructions, 1)?;
        } else {
            let selected = candidates
                .iter()
                .min_by_key(|candidate| {
                    (
                        self.physical_pipeline
                            .selected_instructions
                            .get(**candidate)
                            .copied()
                            .unwrap_or(0),
                        **candidate,
                    )
                })
                .expect("nonempty candidate domain");
            let entry = self
                .physical_pipeline
                .selected_instructions
                .entry((*selected).to_owned())
                .or_default();
            add(entry, 1)?;
        }
        // Nsight's SMSP branch counter is the CBU instruction population,
        // excluding only BMOV and BSSY.  The old projection counted only the
        // direct branch heads, which systematically omitted control-flow
        // instructions such as BSYNC, EXIT, WARPSYNC, BREAK, and CONT.  Keep
        // this separate from the SASS branch subset (the `sm__sass_*` metric
        // family) and from the branch-target observer below: this is a
        // warp-instruction cardinality, so predicated-off records still count
        // once when the warp executed the instruction.
        let branch_counter = if arch == sim_arch::ArchId::Sm120 {
            matches!(
                head,
                "BRA"
                    | "BRX"
                    | "BRXU"
                    | "JMP"
                    | "JMX"
                    | "JMXU"
                    | "CALL"
                    | "RET"
                    | "EXIT"
                    | "BSYNC"
                    | "WARPSYNC"
                    | "YIELD"
                    | "BREAK"
                    | "CONT"
            )
        } else {
            matches!(head, "BRA" | "BRX" | "BRXU" | "JMP" | "JMX" | "JMXU")
        };
        if branch_counter {
            add(&mut self.branch_warp_instructions, 1)?;
            add(
                &mut self.branch_mixed_predicate_warp_instructions,
                u64::from(
                    active_mask_present
                        && predicate_mask_present
                        && effective > 0
                        && effective < active,
                ),
            )?;
        }
        if effective == 0 {
            return Ok(());
        }
        if let AccessKind::GlobalStore { off, len } = kind {
            let end =
                off.checked_add(usize::from(len))
                    .ok_or(crate::IssueError::CounterOverflow {
                        counter: "global-store sector span",
                    })?;
            if let Some(span) = _sector_arena.get(off..end) {
                let current = span.iter().copied().collect::<BTreeSet<_>>();
                if effective >= 8 && self.global_store_previous_effective_lanes >= 8 {
                    let duplicate = current
                        .intersection(&self.global_store_previous_sectors)
                        .count() as u64;
                    add(&mut self.global_store_adjacent_duplicate_sectors, duplicate)?;
                }
                self.global_store_previous_sectors = current;
                self.global_store_previous_effective_lanes = effective;
            } else {
                self.global_store_previous_sectors.clear();
                self.global_store_previous_effective_lanes = 0;
            }
        } else {
            self.global_store_previous_sectors.clear();
            self.global_store_previous_effective_lanes = 0;
        }
        if arch == sim_arch::ArchId::Sm120
            && matches!(kind, AccessKind::Shared { .. })
            && head == "LDS"
        {
            let sass_head = opcode.split_ascii_whitespace().next().unwrap_or(opcode);
            let width16 = sass_head.split('.').any(|token| token == "128");
            let width4 = !width16 && !sass_head.split('.').any(|token| token == "64");
            if effective == 32 && width16 {
                add(&mut self.shared_load_full_warp_width16_records, 1)?;
                if let AccessKind::Shared {
                    data_wavefronts,
                    data_ideal_wavefronts,
                    ..
                } = kind
                {
                    if data_wavefronts.saturating_mul(2) == data_ideal_wavefronts {
                        add(
                            &mut self.shared_load_width16_observer_reduction,
                            data_wavefronts,
                        )?;
                        let current_excess = data_wavefronts.saturating_sub(data_ideal_wavefronts);
                        let original_excess = data_wavefronts
                            .saturating_mul(2)
                            .saturating_sub(data_ideal_wavefronts);
                        add(
                            &mut self.shared_load_width16_excess_reduction,
                            original_excess.saturating_sub(current_excess),
                        )?;
                    }
                }
            } else if effective == 32 && width4 {
                add(&mut self.shared_load_full_warp_width4_records, 1)?;
            }
        }
        let (key, sectors, bank_wavefronts) = match kind {
            AccessKind::NonMemory | AccessKind::PredicatedOffMemory => return Ok(()),
            AccessKind::Shared {
                data_wavefronts, ..
            } => (
                // Collective matrix stores are the store-side counterpart
                // of LDSM.  Keeping STSM in the load bucket makes the
                // operation-qualified opcode counters report matrix stores
                // as shared loads and leaves the aggregate store-wavefront
                // observer empty.  Preserve both matrix forms in their
                // direction-specific buckets; the conflict surface below
                // still excludes them from ordinary LDS/STS bank conflicts.
                if matches!(head, "STS" | "STSM") {
                    "shared_store"
                } else if head.starts_with("ATOM") {
                    "shared_atomic"
                } else {
                    "shared_load"
                },
                0,
                data_wavefronts,
            ),
            AccessKind::LocalLoad { len, .. } => ("local_load", u64::from(len), 0),
            AccessKind::LocalStore { len, .. } => ("local_store", u64::from(len), 0),
            AccessKind::GlobalStore { len, .. } => ("global_store", u64::from(len), 0),
            AccessKind::GlobalReduction { len, .. } => (
                "global_reduction",
                // RED is a non-returning, lane-serialized write operation.
                // The validated sector span deliberately preserves repeated
                // aliases: GA10x reports those transaction sectors rather
                // than collapsing them to unique sector IDs (the same
                // convention is used by the atomic projection and by the
                // issue-side alias-aware wavefront counter).
                u64::from(len),
                0,
            ),
            AccessKind::GlobalBypass {
                token: CacheOpToken::Atom,
                off: _,
                len,
                ..
            } => (
                "global_atomic",
                // The validated atomic sector arena repeats each touched
                // sector by its maximum same-word alias multiplicity. This
                // reproduces GA10x's atom-sector surface: coalesced words use
                // unique sectors, while same-word aliases remain serialized.
                u64::from(len),
                0,
            ),
            AccessKind::GlobalAsyncCopy { len, .. } => ("async_global_load", u64::from(len), 0),
            AccessKind::GlobalEligible { len, .. } | AccessKind::GlobalBypass { len, .. } => {
                ("global_load", u64::from(len), 0)
            }
            AccessKind::GlobalUntagged { len, .. } => ("unclassified_global", u64::from(len), 0),
        };
        let row = self.memory.entry(key.to_owned()).or_default();
        add(&mut row.requests, 1)?;
        add(&mut row.sectors, sectors)?;
        if let Some((unique_sectors, unique_lines)) = decoded_sector_stats(kind, _sector_arena) {
            add(&mut row.unique_sectors, unique_sectors)?;
            add(&mut row.unique_lines, unique_lines)?;
            add(
                &mut row.sector_aliases,
                sectors.saturating_sub(unique_sectors),
            )?;
        }
        add(&mut row.bank_wavefronts, bank_wavefronts)?;
        if let AccessKind::Shared {
            data_wavefronts,
            data_ideal_wavefronts,
            ..
        } = kind
        {
            add(&mut row.ideal_bank_wavefronts, data_ideal_wavefronts)?;
            // Nsight's operation-qualified shared-load/store conflict rows
            // do not include matrix loads/stores. Keep those operations in
            // the aggregate wavefront surface, but exclude them from the
            // corresponding conflict numerator.
            let conflict_surface = !matches!(head, "LDSM" | "STSM");
            if conflict_surface {
                add(&mut row.bank_conflict_wavefronts, data_wavefronts)?;
                add(
                    &mut row.ideal_bank_conflict_wavefronts,
                    data_ideal_wavefronts,
                )?;
                add(
                    &mut row.bank_conflict_excess_wavefronts,
                    data_wavefronts.saturating_sub(data_ideal_wavefronts),
                )?;
            }
        }
        if let Some(scope) = memory_scope_key(opcode, kind) {
            let row = self.memory_scope.entry(scope.to_owned()).or_default();
            add(&mut row.requests, 1)?;
            add(&mut row.sectors, sectors)?;
            if let Some((unique_sectors, unique_lines)) = decoded_sector_stats(kind, _sector_arena)
            {
                add(&mut row.unique_sectors, unique_sectors)?;
                add(&mut row.unique_lines, unique_lines)?;
                add(
                    &mut row.sector_aliases,
                    sectors.saturating_sub(unique_sectors),
                )?;
            }
            add(&mut row.bank_wavefronts, bank_wavefronts)?;
        }
        Ok(())
    }
}

/// Half-open, model-relative lifetimes [admission, retirement). No hardware
/// timestamps are used. Retired warps no longer contribute even if their CTA
/// still retains resource allocations. Pending write acknowledgements are a
/// separate completion domain.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct WarpLifetime {
    pub block: i64,
    pub warp_slot: i32,
    pub admission: u64,
    pub retirement: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct SmActivity {
    pub warp_active_cycles: u128,
    pub sm_active_cycles: u64,
    pub max_resident_warps_per_sm: u32,
    pub peak_live_warps: u32,
    pub smsp_active_cycles: [u64; 4],
    pub instructions_per_smsp: [u64; 4],
    pub issue_active_cycles_per_smsp: [u64; 4],
    /// Number of distinct executed instruction sectors assigned to this SM.
    /// This is an observation-side scope anchor for a future per-SM effective
    /// front-end envelope; it does not alter replay timing.
    pub front_end_instruction_read_sectors: u64,
    pub warp_lifetimes: Vec<WarpLifetime>,
}

fn union_length(intervals: &mut [(u64, u64)]) -> u64 {
    intervals.sort_unstable();
    let mut end = 0;
    let mut length: u64 = 0;
    for &(begin, stop) in intervals.iter() {
        length = length.saturating_add(stop.saturating_sub(begin.max(end)));
        end = end.max(stop);
    }
    // Disjoint intervals fit within one u64 timeline; summation cannot overflow.
    length
}

