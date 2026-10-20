// Lexical implementation section 5/8; included by the parent facade.
impl<'a> SmCore<'a> {/// Admit blocks in launch order while all modeled constraints allow: the
    /// resident-block and resident-warp ceilings, the architecture-card
    /// shared-memory capacity, and disjoint physical warp slots across
    /// concurrent CTAs. A colliding later CTA
    /// waits for the earlier CTA to retire; reuse after retirement is valid. A
    /// block whose warps are ALL empty occupies no residency (it can never retire
    /// through the issue path, so counting it would leak resources and block later
    /// admissions forever).
    fn admit_blocks(&mut self, t: u64) -> Result<(), IssueError> {
        while self.next_block < self.blocks.len() {
            let blk = &self.blocks[self.next_block];
            let live_warps = self.block_live_warps[self.next_block];
            let block_shared_memory_bytes = if live_warps == 0 {
                0
            } else {
                self.shared_memory_residency.bytes_per_cta()
            };
            let resident_blocks = self
                .block_states
                .iter()
                .filter(|state| state.unfinished != 0)
                .count() as u32;
            let next_resident_blocks = resident_blocks
                .checked_add(u32::from(live_warps != 0))
                .ok_or(IssueError::CounterOverflow {
                    counter: "resident blocks",
                })?;
            let Some(next_resident_warps) = self.resident_warps.checked_add(live_warps) else {
                // Construction proved this CTA fits alone, so an overflowing sum
                // can only mean it must wait for the current wave to retire.
                if self.resident_warps > 0 {
                    break;
                }
                return Err(IssueError::CounterOverflow {
                    counter: "resident warp",
                });
            };
            let Some(next_resident_shared_memory_bytes) = self
                .resident_shared_memory_bytes
                .checked_add(block_shared_memory_bytes)
            else {
                if self.resident_warps > 0 {
                    break;
                }
                return Err(IssueError::CounterOverflow {
                    counter: "resident shared-memory byte",
                });
            };
            let pending_slots = blk
                .warps
                .iter()
                .filter(|warp| !warp.insts.is_empty())
                .map(|warp| warp.warp_slot)
                .collect::<Vec<_>>();
            let slots_collide = pending_slots
                .iter()
                .any(|slot| self.resident_warp_slots.contains(slot));
            if self.resident_warps > 0
                && (next_resident_blocks > self.params.max_resident_blocks_per_sm
                    || next_resident_warps > self.params.max_resident_warps_per_sm
                    || next_resident_shared_memory_bytes
                        > self.shared_memory_residency.max_bytes_per_sm()
                    || slots_collide)
            {
                break;
            }

            let admitted_warps =
                usize::try_from(live_warps).map_err(|_| IssueError::CounterOverflow {
                    counter: "admitted warp cardinality",
                })?;
            let next_unfinished_warps = checked_usize_counter_add(
                "unfinished warp",
                self.unfinished_warps,
                admitted_warps,
            )?;
            let next_arrival_stamp = checked_u64_counter_add(
                "warp arrival stamp",
                self.arrival_stamp,
                u64::from(live_warps),
            )?;
            let next_block = checked_usize_counter_add("next block", self.next_block, 1)?;

            // Build the complete block state off to the side. A schedule error
            // cannot leave a prefix of the block admitted.
            let block_ix = self.block_states.len();
            let first_warp = self.warps.len();
            let mut subcore_additions = [0usize; SUB_CORES];
            let mut pending_warps = Vec::with_capacity(admitted_warps);
            for w in &blk.warps {
                if w.insts.is_empty() {
                    continue;
                }
                let subcore =
                    usize::try_from(w.warp_slot).map_err(|_| IssueError::WarpSlotOutOfRange {
                        launch_index: blk.launch_index,
                        warp_slot: w.warp_slot,
                        arch: self.params.arch,
                        physical_warp_slots: self.physical_warp_slots,
                    })? % SUB_CORES;
                let pos_in_subcore = checked_usize_counter_add(
                    "sub-core resident warp",
                    self.subcore_warps[subcore].len(),
                    subcore_additions[subcore],
                )?;
                let arrival_offset = u64::try_from(pending_warps.len()).map_err(|_| {
                    IssueError::CounterOverflow {
                        counter: "warp arrival stamp",
                    }
                })?;
                let arrival = checked_u64_counter_add(
                    "warp arrival stamp",
                    self.arrival_stamp,
                    arrival_offset,
                )?;
                let next_slot = lookup_dense(
                    &self.schedule_slots,
                    self.schedule,
                    w.insts[0].pc,
                    blk.launch_index,
                    w.warp_slot,
                )?;
                pending_warps.push((
                    subcore,
                    WarpState {
                        insts: &w.insts,
                        sectors: &w.sectors,
                        async_copy_cohorts: w.async_copy_cohorts.as_deref(),
                        next: 0,
                        front_end_fetch_cursor: 0,
                        instruction_buffer: VecDeque::new(),
                        next_slot: Some(next_slot),
                        current_since: t,
                        admission_time: t,
                        eligible: t,
                        imc_wait_until: 0,
                        ordinary_dependency_wait_mask: next_slot.ordinary_dependency_wait_mask(),
                        barrier_clear: [0; DEP_BARRIERS],
                        barrier_origin: [DependencyOrigin::Unknown; DEP_BARRIERS],
                        barrier_pending: [0; DEP_BARRIERS],
                        register_ready: Vec::new(),
                        block: block_ix,
                        launch_index: blk.launch_index,
                        warp_id: w.warp_id,
                        warp_slot: w.warp_slot,
                        pos_in_subcore,
                        arrival,
                        at_block_barrier: false,
                        cta_barrier_ready: 0,
                        barrier_pipeline_ready: 0,
                        sfu_dependency_ready: 0,
                        branch_pipeline_ready: 0,
                        block_barrier_dispatch: None,
                        block_barrier_arrival: None,
                        clock_first_issue: None,
                        clock_reads_issued: 0,
                        done: false,
                    },
                ));
                subcore_additions[subcore] = checked_usize_counter_add(
                    "sub-core resident warp",
                    subcore_additions[subcore],
                    1,
                )?;
            }

            for (subcore, warp) in pending_warps {
                let ix = self.warps.len();
                self.subcore_warps[subcore].push(ix);
                self.warps.push(warp);
                if self.has_async_copy_protocol {
                    self.async_copies.push(AsyncCopySlot::default());
                }
            }
            debug_assert!(
                if self.has_async_copy_protocol {
                    self.warps.len() == self.async_copies.len()
                } else {
                    self.async_copies.is_empty()
                },
                "async-copy sidecar shape must match its construction-time protocol mode"
            );
            for &slot in &pending_slots {
                let inserted = self.resident_warp_slots.insert(slot);
                debug_assert!(
                    inserted,
                    "admission preflight rejected resident slot collisions"
                );
            }
            self.block_states.push(BlockState {
                unfinished: admitted_warps,
                waiting: 0,
                warp_count: live_warps,
                shared_memory_bytes: block_shared_memory_bytes,
                warp_slots: pending_slots.into_boxed_slice(),
                first_warp,
                admitted_warps,
                barrier_generation: 0,
                barrier_first_dispatch: None,
                barrier_first_arrival: None,
                barrier_rearm_ready: 0,
            });
            self.resident_warps = next_resident_warps;
            self.resident_shared_memory_bytes = next_resident_shared_memory_bytes;
            self.unfinished_warps = next_unfinished_warps;
            self.arrival_stamp = next_arrival_stamp;
            self.next_block = next_block;
        }
        Ok(())
    }

    // (Shared-mode memory helpers are free functions over the
    // destructured `SharedL2State` + warp slice so the borrow of
    // `self.mem` never has to span a `&mut self` call.)

    /// Memory interaction of one issued record in L1Shared mode. Returns
    /// the KNOWN data-ready component (`None` when completion depends only on
    /// in-flight responses); unknown components remain in ordinary-barrier
    /// pending units and/or async-copy-group source units.
    fn mem_step_shared(
        &mut self,
        subcore: usize,
        wix: usize,
        instruction: usize,
        kind: AccessKind,
        constant_load: bool,
        constant_deep: bool,
        client_scope: L2ClientScope,
        l1_observer_store: bool,
        l2_observer_store: bool,
        sectors: &'a [i64],
        write_barrier: Option<DepBarrier>,
        destination_wavefronts: u64,
        memory_time: u64,
        out: &mut Vec<MemRequest>,
    ) -> Result<Option<u64>, IssueError> {
        let t = memory_time;
        let p = self.params;
        let emit_admission_witnesses = self.emit_admission_witnesses;
        let shared_write_lookup_observer = self.shared_write_lookup_runtime.observer.clone();
        // The observer is shared across SM actors, but an instruction's memory
        // step is still a transaction: later request/counter validation may
        // fail after the write stream has been classified.  Prepare only the
        // touched sets and publish the plan after the complete step succeeds;
        // this avoids both leaked tags and an O(total-set-count) clone on every
        // store instruction.
        let mut staged_write_lookup_plan: Option<L1WriteLookupObserverPlan> = None;
        let MemModel::L1Shared(st) = &mut self.mem else {
            unreachable!("mem_step_shared is only dispatched in shared mode")
        };
        let warps = &mut self.warps;
        let async_copies = &mut self.async_copies;
        match kind {
            AccessKind::PredicatedOffMemory => Ok(None),
            AccessKind::NonMemory => {
                if write_barrier.is_some() {
                    p.non_memory_ready(t).map(Some)
                } else {
                    Ok(None)
                }
            }
            AccessKind::Shared { wavefronts, .. } => {
                if write_barrier.is_some() {
                    p.shared_ready(t, wavefronts).map(Some)
                } else {
                    Ok(None)
                }
            }
            AccessKind::GlobalEligible { off, len, .. } | AccessKind::LocalLoad { off, len } => {
                let origin = match kind {
                    AccessKind::GlobalEligible { .. } => MemoryOrigin::Global,
                    AccessKind::LocalLoad { .. } => MemoryOrigin::Local,
                    _ => unreachable!("match arm admits only hierarchy loads"),
                };
                // The hit-pipeline floor applies to the whole dependent load
                // (as in L1Const); misses ride their responses. Line fills
                // emitted at the same issue timestamp (st.line_folds,
                // cleared at batch entry) fold later same-line sectors into
                // their request mask — one L2 transaction per line per
                // timestamp, priced on its worst sector.
                let target = write_barrier.map(FillTarget::Barrier);
                let lookup_client = if origin == MemoryOrigin::Local {
                    L1LookupClient::Local
                } else if constant_load {
                    L1LookupClient::Constant
                } else {
                    L1LookupClient::OrdinaryGlobal
                };
                let request_sectors = &sectors[off..off + usize::from(len)];
                if let Some(observer) = shared_write_lookup_observer.as_ref() {
                    // Reads allocate the same report-only tag presence view
                    // used by stores.  Ignore the returned hit count here;
                    // only the later store instruction consumes it.
                    staged_write_lookup_plan = Some(observer.borrow().plan_batch(request_sectors)?);
                }
                let lookup_batch = st
                    .l1
                    .begin_lookup_batch_for_client_with_sectors(lookup_client, request_sectors);
                let mut known = write_barrier
                    .is_some()
                    .then(|| checked_cycle_add("shared L1 hit-pipeline floor", t, p.l1_hit_latency))
                    .transpose()?;
                let range = off..off + usize::from(len);
                for i in range.clone() {
                    match access_shared(
                        st,
                        warps,
                        wix,
                        target,
                        None,
                        origin,
                        constant_load,
                        constant_deep,
                        client_scope,
                        sectors[i],
                        ResponseId::new(st.response_owner, wix, instruction, i),
                        t,
                        out,
                    ) {
                        Ok(Some(ready)) => {
                            let ready = checked_constant_return_time(
                                ready,
                                constant_deep && target.and_then(FillTarget::barrier).is_some(),
                                &p,
                                "constant L1-hit return latency",
                            )?;
                            if let Some(current) = &mut known {
                                *current = (*current).max(ready);
                            }
                        }
                        Ok(None) => {}
                        Err(SharedAccessError::Stalled) => {
                            // MSHR full: park the remainder; the entry
                            // holds one pending unit until drained.
                            warps[wix].add_pending(wix, write_barrier, 1)?;
                            st.retry.push_back(Retry {
                                warp: wix,
                                instruction,
                                origin,
                                constant_load,
                                constant_deep,
                                client_scope,
                                target,
                                mshr_floor: None,
                                lookup_batch,
                                lo: i,
                                hi: range.end,
                            });
                            break;
                        }
                        Err(SharedAccessError::Issue(source)) => return Err(source),
                    }
                }
                if let Some(plan) = staged_write_lookup_plan.take() {
                    if let Some(observer) = shared_write_lookup_observer.as_ref() {
                        observer.borrow_mut().commit_plan(plan);
                    }
                }
                Ok(known)
            }
            AccessKind::GlobalAsyncCopy {
                off,
                len,
                completion_work,
                l1_policy: LdgstsL1Policy::CacheAtAllLevels,
            } => {
                if st.l1_visibility_requester.is_some() {
                    return Err(IssueError::VisibilityAsyncCopyUnsupported);
                }
                let source_active = completion_work != 0;
                let hit_service_interval = self
                    .params
                    .ldgsts_l1_hit_return_service
                    .map_or(0, |service| service.interval_cycles);
                let mut hit_service_ready =
                    self.shared_write_lookup_runtime.async_l1_hit_service_ready[subcore];
                // The return-port token is per warp-level LDGSTS source
                // instruction, not per 32-byte sector.  The trace `len`
                // already represents the sectors coalesced by one SASS
                // instruction; charging the interval once for every sector
                // multiplies the source frontier by the payload width.
                let mut instruction_hit_service_time = None;
                let range = off..off + usize::from(len);
                let delivery_work = async_copy_completion_work(kind).unwrap_or(0);
                let async_group = if source_active {
                    Some(async_copies[wix].add_pending_async_copy_sources(
                        wix,
                        u32::from(len),
                        delivery_work,
                        destination_wavefronts,
                        None,
                        &p,
                    )?)
                } else {
                    None
                };
                let target = async_group
                    .map(|group| FillTarget::AsyncCopy {
                        group,
                        barrier: write_barrier,
                    })
                    .or_else(|| write_barrier.map(FillTarget::Barrier));
                let request_sectors = &sectors[off..off + usize::from(len)];
                let lookup_batch = st
                    .l1
                    .begin_async_copy_lookup_batch_with_sectors(request_sectors);
                let mut known = (write_barrier.is_some() || source_active)
                    .then(|| {
                        checked_cycle_add(
                            "LDGSTS shared L1 hit-pipeline floor",
                            t,
                            p.l1_hit_latency,
                        )
                    })
                    .transpose()?;
                for i in range.clone() {
                    match access_shared(
                        st,
                        warps,
                        wix,
                        target,
                        Some(p.ldgsts_l1_mshr_floor),
                        MemoryOrigin::Global,
                        false,
                        false,
                        L2ClientScope::AsyncLdgstsCache,
                        sectors[i],
                        ResponseId::new(st.response_owner, wix, instruction, i),
                        t,
                        out,
                    ) {
                        Ok(Some(ready)) => {
                            let ready = if hit_service_interval == 0 {
                                ready
                            } else if let Some(serviced) = instruction_hit_service_time {
                                serviced
                            } else {
                                let serviced = ready.max(hit_service_ready);
                                hit_service_ready = serviced
                                    .checked_add(hit_service_interval)
                                    .ok_or(IssueError::CounterOverflow {
                                        counter: "async L1 hit service timeline",
                                    })?;
                                instruction_hit_service_time = Some(serviced);
                                serviced
                            };
                            if let Some(group) = async_group {
                                async_copies[wix]
                                    .complete_async_copy_sources(wix, group, None, 1, ready, &p)?;
                            }
                            if let Some(current) = &mut known {
                                *current = (*current).max(ready);
                            }
                        }
                        Ok(None) => {}
                        Err(SharedAccessError::Stalled) => {
                            // Async source units were counted at original issue;
                            // only the optional ordinary barrier needs a retry
                            // replacement unit.
                            warps[wix].add_pending(wix, write_barrier, 1)?;
                            st.retry.push_back(Retry {
                                warp: wix,
                                instruction,
                                origin: MemoryOrigin::Global,
                                constant_load: false,
                                constant_deep: false,
                                client_scope: L2ClientScope::AsyncLdgstsCache,
                                target,
                                mshr_floor: Some(p.ldgsts_l1_mshr_floor),
                                lookup_batch,
                                lo: i,
                                hi: range.end,
                            });
                            break;
                        }
                        Err(SharedAccessError::Issue(source)) => return Err(source),
                    }
                }
                if let Some(plan) = staged_write_lookup_plan.take() {
                    if let Some(observer) = shared_write_lookup_observer.as_ref() {
                        observer.borrow_mut().commit_plan(plan);
                    }
                }
                self.shared_write_lookup_runtime.async_l1_hit_service_ready[subcore] =
                    hit_service_ready;
                Ok(known)
            }
            AccessKind::GlobalBypass { off, len, .. }
            | AccessKind::GlobalAsyncCopy {
                off,
                len,
                l1_policy: LdgstsL1Policy::BypassL1,
                ..
            }
            | AccessKind::GlobalUntagged { off, len } => {
                let origin = MemoryOrigin::Global;
                let (source, counted, operation, async_copy) = match kind {
                    // ATOM rides bypass provenance but remains a returning RMW
                    // operation; every other bypass / untagged sector is a read.
                    AccessKind::GlobalBypass { token, .. } => (
                        L2Source::Bypass,
                        true,
                        if token == CacheOpToken::Atom {
                            MemoryOperation::Atomic
                        } else {
                            MemoryOperation::Read
                        },
                        token == CacheOpToken::Ldgsts,
                    ),
                    AccessKind::GlobalAsyncCopy {
                        completion_work, ..
                    } => (
                        L2Source::Bypass,
                        true,
                        MemoryOperation::Read,
                        completion_work != 0,
                    ),
                    AccessKind::GlobalUntagged { .. } => {
                        (L2Source::Untagged, false, MemoryOperation::Read, false)
                    }
                    _ => unreachable!("match arm admits only bypass, async-copy, or untagged"),
                };
                let observer_operation = if l2_observer_store {
                    Some(MemoryOperation::Store)
                } else {
                    None
                };
                let client_scope = match kind {
                    AccessKind::GlobalAsyncCopy { .. } => L2ClientScope::AsyncLdgstsBypass,
                    AccessKind::GlobalBypass {
                        token: CacheOpToken::Atom,
                        ..
                    } => L2ClientScope::Atomic,
                    AccessKind::GlobalBypass {
                        token: CacheOpToken::Ldgsts,
                        ..
                    } => L2ClientScope::AsyncLdgstsBypass,
                    _ => L2ClientScope::Unknown,
                };
                if operation == MemoryOperation::Atomic && st.l1_visibility_requester.is_some() {
                    return Err(IssueError::VisibilityAtomicUnsupported);
                }
                if async_copy && st.l1_visibility_requester.is_some() {
                    return Err(IssueError::VisibilityAsyncCopyUnsupported);
                }
                if counted {
                    st.l1.count_bypass(
                        origin,
                        u64::from(len),
                        operation == MemoryOperation::Atomic,
                    )?;
                } else {
                    st.l1.count_untagged(origin, u64::from(len))?;
                    if l1_observer_store {
                        if let Some(observer) = shared_write_lookup_observer.as_ref() {
                            staged_write_lookup_plan = Some(
                                observer
                                    .borrow()
                                    .plan_batch(&sectors[off..off + usize::from(len)])?,
                            );
                        } else if let Some(hits) = projected_l1_write_lookup_hits(
                            AccessKind::GlobalStore { off, len },
                            sectors,
                        ) {
                            st.l1.count_write_lookup_hits(origin, hits)?;
                        }
                    }
                }
                let async_group = if async_copy {
                    let delivery_work = async_copy_completion_work(kind).unwrap_or(0);
                    let exact_cohorts = warps[wix]
                        .async_copy_cohorts
                        .and_then(|sidecar| sidecar.view(instruction));
                    Some(
                        async_copies[wix].add_pending_async_copy_sources(
                            wix,
                            u32::from(len),
                            delivery_work,
                            destination_wavefronts,
                            Some(
                                bypass_async_copy_observed_topology(
                                    kind,
                                    instruction,
                                    exact_cohorts,
                                )
                                .unwrap_or(BypassAsyncCopyInput {
                                    instruction,
                                    sector_off: off,
                                    sector_len: len,
                                    source_cohorts: 0,
                                    destination_lines: 0,
                                    max_bank_wavefronts: 0,
                                    max_bypass_wavefronts: 0,
                                    cohorts: None,
                                    sector_membership_masks: None,
                                }),
                            ),
                            &p,
                        )?,
                    )
                } else {
                    None
                };
                if len == 0 {
                    // Degenerate records have no source requests. A topology
                    // carrying only executing ignore-src lanes still owns an
                    // async group, so create it before returning; otherwise a
                    // later commit/wait would silently drop destination work.
                    return if write_barrier.is_some() {
                        p.store_visibility_ready(t).map(Some)
                    } else {
                        Ok(None)
                    };
                }
                for (relative, &sector_id) in
                    sectors[off..off + usize::from(len)].iter().enumerate()
                {
                    let i = off + relative;
                    let response_id = ResponseId::new(st.response_owner, wix, instruction, i);
                    let reply = if let Some(group) = async_group {
                        if let Some(barrier) = write_barrier {
                            warps[wix].add_pending(wix, Some(barrier), 1)?;
                        }
                        let reply = ReplyTo::AsyncCopy {
                            warp: wix,
                            group,
                            barrier: write_barrier,
                            response_id,
                        };
                        register_response(st, reply)?;
                        reply
                    } else if let Some(barrier) = write_barrier {
                        let reply = ReplyTo::Warp {
                            warp: wix,
                            barrier,
                            response_id,
                        };
                        warps[wix].add_pending(wix, Some(barrier), 1)?;
                        register_response(st, reply)?;
                        reply
                    } else {
                        ReplyTo::None
                    };
                    out.push(MemRequest {
                        issue_time: t,
                        sector_id,
                        origin,
                        sector_mask: 1 << (sector_id & 3) as u8,
                        local_sector_mask: 0,
                        source,
                        operation,
                        observer_operation,
                        client_scope,
                        constant_load: false,
                        constant_deep: false,
                        reply,
                        admission_witness: None,
                    });
                }
                if let Some(plan) = staged_write_lookup_plan.take() {
                    st.l1.count_write_lookup_hits(origin, plan.hits())?;
                    if let Some(observer) = shared_write_lookup_observer.as_ref() {
                        observer.borrow_mut().commit_plan(plan);
                    }
                }
                Ok(None)
            }
            AccessKind::GlobalStore { off, len }
            | AccessKind::GlobalReduction { off, len, .. }
            | AccessKind::LocalStore { off, len } => {
                let origin = match kind {
                    AccessKind::GlobalStore { .. } | AccessKind::GlobalReduction { .. } => {
                        MemoryOrigin::Global
                    }
                    AccessKind::LocalStore { .. } => MemoryOrigin::Local,
                    _ => unreachable!("match arm admits only hierarchy writes"),
                };
                let reduction = matches!(kind, AccessKind::GlobalReduction { .. });
                let store_sectors = &sectors[off..off + usize::from(len)];
                if shared_write_lookup_observer.is_some()
                    && matches!(kind, AccessKind::GlobalStore { .. })
                {
                    let plan = if let Some(observer) = shared_write_lookup_observer.as_ref() {
                        observer.borrow().plan_batch(store_sectors)?
                    } else {
                        unreachable!("shared observer branch has no observer")
                    };
                    staged_write_lookup_plan = Some(plan);
                } else if let Some(hits) = projected_l1_write_lookup_hits(kind, sectors) {
                    st.l1.count_write_lookup_hits(origin, hits)?;
                }
                for (request_index, &sector_id) in
                    sectors[off..off + usize::from(len)].iter().enumerate()
                {
                    let mut request = if reduction {
                        if st.l1_visibility_requester.is_some() {
                            return Err(IssueError::VisibilityReductionUnsupported);
                        }
                        let hit = st.l1.probe_reduction(origin, sector_id, t)?;
                        if origin == MemoryOrigin::Global {
                            record_global_l1_observation(st, sector_id, true, false, hit, None, t)?;
                        }
                        MemRequest::reduction(t, sector_id)
                    } else {
                        let hit = st.l1.probe_store(origin, sector_id, t)?;
                        let retained_l1_write = origin == MemoryOrigin::Global
                            && hit
                            && st.l1.store_policy() == L1StorePolicy::ResidentHitWriteBack;
                        if origin == MemoryOrigin::Global
                            && let Some(writer) =
                                st.payload_store_writers.as_mut().and_then(|writers| {
                                    writers.get_mut(&(
                                        sector_id.div_euclid(4),
                                        sector_id.rem_euclid(4) as u8,
                                    ))
                                })
                        {
                            *writer = Some(PayloadStoreObservation {
                                site: PayloadStoreSite {
                                    launch_index: warps[wix].launch_index,
                                    warp_slot: warps[wix].warp_slot,
                                    instruction_index: instruction,
                                },
                                accepted_at: t,
                                resident_generation: st
                                    .l1
                                    .resident_payload_state(sector_id.div_euclid(4))
                                    .map(|(generation, _)| generation),
                                retained_dirty: retained_l1_write,
                                attached: false,
                            });
                        }
                        if origin == MemoryOrigin::Global {
                            record_global_l1_observation(
                                st,
                                sector_id,
                                true,
                                retained_l1_write,
                                hit,
                                None,
                                t,
                            )?;
                        }
                        if retained_l1_write {
                            // The resident hit is retained dirty in L1; the
                            // lower L2 mutation occurs only at the explicit
                            // flush boundary, never also on this issue path.
                            continue;
                        }
                        MemRequest::store(t, sector_id, origin)
                    };
                    if origin == MemoryOrigin::Global && emit_admission_witnesses {
                        request.admission_witness = Some(
                            AdmissionRequestWitness::new(
                                u32::try_from(wix).map_err(|_| IssueError::CounterOverflow {
                                    counter: "memory-admission witness warp",
                                })?,
                                u32::try_from(instruction).map_err(|_| {
                                    IssueError::CounterOverflow {
                                        counter: "memory-admission witness instruction",
                                    }
                                })?,
                                u8::try_from(subcore).map_err(|_| IssueError::CounterOverflow {
                                    counter: "memory-admission witness subcore",
                                })?,
                                u16::try_from(request_index).map_err(|_| {
                                    IssueError::CounterOverflow {
                                        counter: "memory-admission witness request index",
                                    }
                                })?,
                                len,
                            )
                            .ok_or(IssueError::CounterOverflow {
                                counter: "memory-admission witness encoding",
                            })?,
                        );
                    }
                    out.push(request);
                }
                // A present write barrier is priced at the L2-hit arm (as in
                // L1Const); stores and reductions do not wait on responses.
                let visibility_ready = if write_barrier.is_some() {
                    p.store_visibility_ready(t).map(Some)?
                } else {
                    None
                };
                if let Some(plan) = staged_write_lookup_plan {
                    st.l1.count_write_lookup_hits(origin, plan.hits())?;
                    if let Some(observer) = shared_write_lookup_observer.as_ref() {
                        observer.borrow_mut().commit_plan(plan);
                    }
                }
                Ok(visibility_ready)
            }
        }
    }

    fn preflight_response_accounting(
        &self,
        st: &SharedL2State,
        reply: ReplyTo,
        now: u64,
    ) -> Result<(), IssueError> {
        let fill = match reply {
            ReplyTo::Warp { warp, barrier, .. } => {
                if warp >= self.warps.len() {
                    return Err(IssueError::ResponseWarpOutOfRange {
                        warp,
                        warps: self.warps.len(),
                    });
                }
                self.warps[warp].preflight_pending_remove(warp, barrier, 1)?;
                return Ok(());
            }
            ReplyTo::AsyncCopy {
                warp,
                group,
                barrier,
                response_id,
            } => {
                if warp >= self.warps.len() {
                    return Err(IssueError::ResponseWarpOutOfRange {
                        warp,
                        warps: self.warps.len(),
                    });
                }
                self.async_copies[warp].preflight_async_copy_source_completion(
                    warp,
                    group,
                    Some(response_id),
                    1,
                    now,
                    &self.params,
                )?;
                if let Some(barrier) = barrier {
                    self.warps[warp].preflight_pending_remove(warp, barrier, 1)?;
                }
                return Ok(());
            }
            ReplyTo::L1Fill { fill, .. } => {
                if !st.l1.is_fill_in_flight(fill) {
                    return Err(IssueError::ResponseFillNotInFlight { fill });
                }
                if let FillKey::Line { line, .. } = fill
                    && !st.fill_masks.contains_key(&line)
                {
                    return Err(IssueError::FillMaskMissing { line });
                }
                fill
            }
            ReplyTo::None => return Err(IssueError::UnexpectedMutationOnlyResponse),
        };

        let mut removals = HashMap::new();
        let mut async_removals = HashMap::new();
        if let Some(waiters) = st.fill_waiters.get(&fill) {
            waiters.try_for_each(|waiter| {
                fill_waiter_ready_time(waiter, now, &self.params)?;
                accumulate_waiter_removals(&mut removals, &mut async_removals, waiter)
            })?;
        }

        let mut projected = HashMap::new();
        for (&(warp, barrier), &remove) in &removals {
            if warp >= self.warps.len() {
                return Err(IssueError::ResponseWarpOutOfRange {
                    warp,
                    warps: self.warps.len(),
                });
            }
            self.warps[warp].preflight_pending_remove(warp, barrier, remove)?;
            let current = self.warps[warp].barrier_pending[barrier.index()];
            projected.insert(
                (warp, barrier),
                current
                    .checked_sub(remove)
                    .ok_or(IssueError::PendingBarrierUnderflow {
                        warp,
                        barrier: barrier.wire(),
                        current,
                        remove,
                    })?,
            );
        }
        let mut projected_async = HashMap::new();
        for (&(warp, group), &remove) in &async_removals {
            project_async_copy_removal(
                &mut projected_async,
                &self.async_copies,
                warp,
                group,
                remove,
                now,
                &self.params,
            )?;
        }

        if st.retry.is_empty() {
            let budget = L1MutationBudget {
                recency_touches: 1,
                ..L1MutationBudget::for_origin(MemoryOrigin::Global)
            };
            if st.l1.preflight_mutation_budget(budget).is_err() {
                let mut shadow = st.l1.clone();
                shadow.complete_fill(fill)?;
            }
            return Ok(());
        }

        let mut budgets = MemoryOrigin::ALL.map(L1MutationBudget::for_origin);
        budgets[MemoryOrigin::Global.index()].recency_touches = 1;
        let mut pending_budget_fits = true;
        let mut has_async_retry = false;
        for retry in &st.retry {
            let budget = &mut budgets[retry.origin.index()];
            if retry.warp >= self.warps.len() {
                return Err(IssueError::ResponseWarpOutOfRange {
                    warp: retry.warp,
                    warps: self.warps.len(),
                });
            }
            has_async_retry |= retry.target.and_then(FillTarget::async_group).is_some();
            let remaining = retry
                .hi
                .checked_sub(retry.lo)
                .ok_or(IssueError::CounterUnderflow {
                    counter: "retry sector cursor",
                })?;
            if retry.hi > self.warps[retry.warp].sectors.len() {
                return Err(IssueError::CounterOverflow {
                    counter: "retry sector cursor",
                });
            }
            let remaining_u64 =
                u64::try_from(remaining).map_err(|_| IssueError::CounterOverflow {
                    counter: "retry L1 budget",
                })?;
            budget.eligible_sectors = checked_u64_counter_add(
                "retry L1 eligible-sector budget",
                budget.eligible_sectors,
                remaining_u64,
            )?;
            budget.recency_touches = checked_u64_counter_add(
                "retry L1 recency budget",
                budget.recency_touches,
                remaining_u64,
            )?;
            budget.new_fills =
                checked_u64_counter_add("retry L1 fill budget", budget.new_fills, remaining_u64)?;
            for sector_index in retry.lo..retry.hi {
                let sector_id = self.warps[retry.warp].sectors[sector_index];
                preflight_line_fill_mask(st, sector_id)?;
                let response_id = ResponseId::new(
                    st.response_owner,
                    retry.warp,
                    retry.instruction,
                    sector_index,
                );
                if let Some(&reply) = st.outstanding_response_ids.get(&response_id) {
                    return Err(IssueError::ResponseTokenAlreadyOutstanding { reply });
                }
            }
            if let Some(barrier) = retry.target.and_then(FillTarget::barrier) {
                let base = self.warps[retry.warp].barrier_pending[barrier.index()];
                let current = projected.entry((retry.warp, barrier)).or_insert(base);
                let before_remove = *current;
                let after_remove =
                    before_remove
                        .checked_sub(1)
                        .ok_or(IssueError::PendingBarrierUnderflow {
                            warp: retry.warp,
                            barrier: barrier.wire(),
                            current: before_remove,
                            remove: 1,
                        })?;
                let additions =
                    u32::try_from(remaining).map_err(|_| IssueError::CounterOverflow {
                        counter: "retry pending-add budget",
                    })?;
                if let Some(after_add) = after_remove.checked_add(additions) {
                    *current = after_add;
                } else {
                    pending_budget_fits = false;
                    *current = after_remove;
                }
            }
        }

        if st.l1.preflight_mutation_budgets(&budgets).is_ok()
            && pending_budget_fits
            && !has_async_retry
        {
            return Ok(());
        }

        // Numeric-boundary slow path: complete and retry on a shadow cache so
        // warm hits and an atomic retry-unit replacement are charged exactly.
        let mut shadow = st.l1.clone();
        shadow.complete_fill(fill)?;
        let mut exact = HashMap::new();
        for (&(warp, barrier), &remove) in &removals {
            let current = self.warps[warp].barrier_pending[barrier.index()];
            exact.insert(
                (warp, barrier),
                current
                    .checked_sub(remove)
                    .ok_or(IssueError::PendingBarrierUnderflow {
                        warp,
                        barrier: barrier.wire(),
                        current,
                        remove,
                    })?,
            );
        }
        let mut exact_async = projected_async;
        'retries: for retry in &st.retry {
            shadow.resume_lookup_batch(retry.lookup_batch);
            let barrier = retry.target.and_then(FillTarget::barrier);
            if let Some(barrier) = barrier {
                let base = self.warps[retry.warp].barrier_pending[barrier.index()];
                let current = exact.entry((retry.warp, barrier)).or_insert(base);
                let before = *current;
                *current = before
                    .checked_sub(1)
                    .ok_or(IssueError::PendingBarrierUnderflow {
                        warp: retry.warp,
                        barrier: barrier.wire(),
                        current: before,
                        remove: 1,
                    })?;
            }
            for sector_index in retry.lo..retry.hi {
                let sector_id = self.warps[retry.warp].sectors[sector_index];
                match access_shared_l1(
                    &mut shadow,
                    retry.origin,
                    sector_id,
                    now,
                    retry.target.is_some(),
                    retry.mshr_floor,
                )? {
                    SharedL1Access::Hit { .. } => {
                        if let Some(group) = retry.target.and_then(FillTarget::async_group) {
                            project_async_copy_removal(
                                &mut exact_async,
                                &self.async_copies,
                                retry.warp,
                                group,
                                1,
                                now,
                                &self.params,
                            )?;
                        }
                    }
                    SharedL1Access::MissRequest { .. } | SharedL1Access::MergedPending { .. } => {
                        if let Some(barrier) = barrier {
                            let current = exact
                                .entry((retry.warp, barrier))
                                .or_insert(self.warps[retry.warp].barrier_pending[barrier.index()]);
                            let before = *current;
                            *current = before.checked_add(1).ok_or(
                                IssueError::PendingBarrierOverflow {
                                    warp: retry.warp,
                                    barrier: barrier.wire(),
                                    current: before,
                                    add: 1,
                                },
                            )?;
                        }
                    }
                    SharedL1Access::Stalled => {
                        if let Some(barrier) = barrier {
                            let current = exact
                                .entry((retry.warp, barrier))
                                .or_insert(self.warps[retry.warp].barrier_pending[barrier.index()]);
                            let before = *current;
                            *current = before.checked_add(1).ok_or(
                                IssueError::PendingBarrierOverflow {
                                    warp: retry.warp,
                                    barrier: barrier.wire(),
                                    current: before,
                                    add: 1,
                                },
                            )?;
                        }
                        break 'retries;
                    }
                }
            }
        }
        Ok(())
    }

    fn preflight_retry_timing(
        &self,
        st: &SharedL2State,
        fill: FillKey,
        now: u64,
    ) -> Result<(), IssueError> {
        if st.retry.is_empty() {
            return Ok(());
        }
        let observes_hit_ready = st
            .retry
            .iter()
            .any(|retry| retry.target.and_then(FillTarget::barrier).is_some());
        let hit_timing_fits = !observes_hit_ready
            || st
                .retry
                .iter()
                .filter(|retry| retry.target.and_then(FillTarget::barrier).is_some())
                .all(|retry| {
                    now.checked_add(self.params.l1_hit_latency)
                        .is_some_and(|hit_ready| {
                            checked_constant_return_time(
                                hit_ready,
                                retry.constant_deep,
                                &self.params,
                                "constant retry L1-hit return latency",
                            )
                            .is_ok()
                        })
                });
        let response_timing_fits = now.checked_add(self.params.l1_miss_latency).is_some();
        if hit_timing_fits && response_timing_fits {
            return Ok(());
        }

        // At the cycle-domain boundary, replay the retry queue on a shadow L1.
        // A hit needs a timestamp only when its retry carries a write barrier;
        // only a fresh miss emits a new self-affecting response. Merges and
        // stalls consume neither timing arm.
        let mut shadow = st.l1.clone();
        shadow.complete_fill(fill)?;
        'retries: for retry in &st.retry {
            shadow.resume_lookup_batch(retry.lookup_batch);
            for sector_index in retry.lo..retry.hi {
                let sector_id = self.warps[retry.warp].sectors[sector_index];
                match access_shared_l1(
                    &mut shadow,
                    retry.origin,
                    sector_id,
                    now,
                    false,
                    retry.mshr_floor,
                )? {
                    SharedL1Access::Hit { .. } => {
                        if retry.target.and_then(FillTarget::barrier).is_some() {
                            let hit_ready = checked_cycle_add(
                                "retry L1 hit visibility",
                                now,
                                self.params.l1_hit_latency,
                            )?;
                            checked_constant_return_time(
                                hit_ready,
                                retry.constant_deep,
                                &self.params,
                                "constant retry L1-hit return latency",
                            )?;
                        }
                    }
                    SharedL1Access::MissRequest { .. } => {
                        checked_cycle_add(
                            "retry self-response horizon",
                            now,
                            self.params.l1_miss_latency,
                        )?;
                    }
                    SharedL1Access::MergedPending { .. } => {}
                    SharedL1Access::Stalled => break 'retries,
                }
            }
        }
        Ok(())
    }

    pub(crate) fn preflight_response(&self, reply: ReplyTo, now: u64) -> Result<(), IssueError> {
        if now < self.t {
            return Err(IssueError::ResponseBeforeClock { now, clock: self.t });
        }
        let st = match &self.mem {
            MemModel::L1Shared(st) => st,
            MemModel::Flat => {
                return Err(IssueError::ResponseModeMismatch { mode: "flat" });
            }
            MemModel::L1Const(_) => {
                return Err(IssueError::ResponseModeMismatch {
                    mode: "constant-L1",
                });
            }
        };
        match reply {
            ReplyTo::Warp { warp, .. } | ReplyTo::AsyncCopy { warp, .. }
                if warp >= self.warps.len() =>
            {
                return Err(IssueError::ResponseWarpOutOfRange {
                    warp,
                    warps: self.warps.len(),
                });
            }
            ReplyTo::None => return Err(IssueError::UnexpectedMutationOnlyResponse),
            _ => {}
        }
        let response_id = reply
            .response_id()
            .ok_or(IssueError::UnexpectedMutationOnlyResponse)?;
        if !st.outstanding_replies.contains(&reply)
            || st.outstanding_response_ids.get(&response_id) != Some(&reply)
        {
            return Err(IssueError::ResponseTokenNotOutstanding { reply });
        }
        if let ReplyTo::Warp { warp, barrier, .. } = reply
            && self.warps[warp].barrier_pending[barrier.index()] == 0
        {
            return Err(IssueError::ResponseBarrierNotPending {
                warp,
                barrier: barrier.wire(),
            });
        }
        self.preflight_response_accounting(st, reply, now)?;
        // A fill completion can drain parked accesses. Validate only the timing
        // branch each exact retry outcome consumes; next-level L2/DRAM pricing
        // is preflighted by the system against the actual L2 outcome.
        if let ReplyTo::L1Fill { fill, .. } = reply {
            self.preflight_retry_timing(st, fill, now)?;
        }
        Ok(())
    }

    /// Apply one memory response at its event time, then drain what it
    /// unblocked (fill waiters, MSHR retry queue — retries may emit new
    /// requests into `out`).
    pub(crate) fn on_response(
        &mut self,
        reply: ReplyTo,
        now: u64,
        out: &mut Vec<MemRequest>,
    ) -> Result<(), IssueError> {
        self.preflight_response(reply, now)?;
        let params = self.params;
        let st = match &mut self.mem {
            MemModel::L1Shared(st) => st,
            MemModel::Flat => {
                return Err(IssueError::ResponseModeMismatch { mode: "flat" });
            }
            MemModel::L1Const(_) => {
                return Err(IssueError::ResponseModeMismatch {
                    mode: "constant-L1",
                });
            }
        };
        let response_id = reply
            .response_id()
            .ok_or(IssueError::UnexpectedMutationOnlyResponse)?;
        if !st.outstanding_replies.remove(&reply)
            || st.outstanding_response_ids.remove(&response_id) != Some(reply)
        {
            return Err(IssueError::ResponseTokenRemovalFailed { reply });
        }
        let response_mask = st.response_masks.get(&response_id).copied();
        st.response_masks.remove(&response_id);
        // New emission batch (the retry drain below may emit).
        st.line_folds.clear();
        let warps = &mut self.warps;
        let async_copies = &mut self.async_copies;
        match reply {
            ReplyTo::L1Fill { fill, .. } => {
                let releases_mshr = st.l1.fill_releases_mshr(fill);
                if let FillKey::Line { line, .. } = fill
                    && st.fill_masks.remove(&line).is_none()
                {
                    return Err(IssueError::FillMaskMissing { line });
                }
                let retained = st.l1.complete_fill(fill)?;
                if let Some(requester) = st.l1_visibility_requester {
                    if !retained {
                        return Err(IssueError::VisibilityFillDropped { fill });
                    }
                    let observed_mask =
                        response_mask.ok_or(IssueError::VisibilityResponseMaskMissing {
                            response: response_id,
                        })?;
                    let (line_id, expected_mask) = match fill {
                        FillKey::Line { line, .. } => (line, 0b1111),
                        FillKey::Sector { line, sector, .. } => (line, 1_u8 << sector),
                    };
                    let sector_mask = match fill {
                        FillKey::Line { .. }
                            if observed_mask != 0 && observed_mask & !0b1111 == 0 =>
                        {
                            observed_mask
                        }
                        FillKey::Sector { .. } if observed_mask == expected_mask => observed_mask,
                        _ => {
                            return Err(IssueError::VisibilityResponseMaskInvalid {
                                response: response_id,
                                expected: expected_mask,
                                got: observed_mask,
                            });
                        }
                    };
                    let generation = fill_generation(fill);
                    let sequence = st.next_l1_visibility_sequence()?;
                    st.l1_fill_visibility_observations
                        .push(sim_mem::L1FillVisibleObservation {
                            sequence,
                            requester,
                            line_id,
                            sector_mask,
                            fill_generation: generation,
                            visible_cycle: now,
                        });
                }
                if let Some(waiters) = st.fill_waiters.remove(&fill) {
                    waiters.try_for_each(|waiter| {
                        let ready = fill_waiter_ready_time(waiter, now, &params)?;
                        if let Some(group) = waiter.target.async_group() {
                            async_copies[waiter.warp].complete_async_copy_sources(
                                waiter.warp,
                                group,
                                None,
                                waiter.units,
                                ready,
                                &params,
                            )?;
                        }
                        if let Some(barrier) = waiter.target.barrier() {
                            warps[waiter.warp].clear_pending(
                                waiter.warp,
                                barrier,
                                waiter.units,
                                ready,
                            )?;
                        }
                        Ok(())
                    })?;
                }
                // Only an independently occupied MSHR completion can drain
                // parked accesses. A late sector child shares its parent
                // line allocation slot and must not cause a speculative
                // retry while that parent remains in flight.
                if releases_mshr {
                    drain_retries(st, warps, async_copies, now, &params, out)?;
                }
            }
            ReplyTo::Warp { warp, barrier, .. } => {
                warps[warp].clear_pending(warp, barrier, 1, now)?;
            }
            ReplyTo::AsyncCopy {
                warp,
                group,
                barrier,
                response_id,
            } => {
                async_copies[warp].complete_async_copy_sources(
                    warp,
                    group,
                    Some(response_id),
                    1,
                    now,
                    &params,
                )?;
                if let Some(barrier) = barrier {
                    warps[warp].clear_pending(warp, barrier, 1, now)?;
                }
            }
            ReplyTo::None => return Err(IssueError::UnexpectedMutationOnlyResponse),
        }
        Ok(())
    }

    fn project_selected_readiness(
        &self,
        subcore: usize,
        warp_index: usize,
        now: u64,
        projected: &mut ReadinessDiagnostics,
    ) -> Result<(), IssueError> {
        let warp = &self.warps[warp_index];
        let slot = warp
            .next_slot
            .ok_or(IssueError::MissingIssueSlot { warp: warp_index })?;
        let kind = warp.insts[warp.next].kind;
        if let Some((class, sectors, wavefronts, unique_lines, atomic_l2_requests)) =
            projected_global_memory_issue(kind, warp.sectors)
        {
            projected
                .global_memory_issue
                .observe(
                    class,
                    sectors,
                    wavefronts,
                    unique_lines,
                    atomic_l2_requests,
                    slot.control.read_barrier.is_some(),
                    now,
                )
                .map_err(|source| match source {
                    ReadinessCounterError::Overflow { field, .. } => {
                        IssueError::CounterOverflow { counter: field }
                    }
                    mismatch @ (ReadinessCounterError::MemoryServiceCandidateMismatch {
                        ..
                    }
                    | ReadinessCounterError::MemoryAdmissionCandidateMismatch {
                        ..
                    }) => IssueError::ReadinessDiagnostic(mismatch),
                })?;
        }
        let base = warp.current_since;
        let mut ready_at = base;
        let mut limiting = None;
        let mut tied = 0;
        let mut async_snapshot = None;
        if self.params.resource_source() == IssueResourceSource::Sm120Alpha {
            let inst = warp.insts[warp.next];
            let ready = issue_fixed_register_ready(
                slot.fixed_latency_operands(),
                inst.effective_lane_mask(),
                inst.active_lane_mask,
                slot.guard_predicate,
                &warp.register_ready,
            );
            for (cause, until) in [
                (ReadinessCause::RegisterRaw, ready.raw),
                (ReadinessCause::RegisterPredicate, ready.predicate),
                (ReadinessCause::RegisterWaw, ready.waw),
            ] {
                observe_readiness_constraint(
                    projected,
                    base,
                    until,
                    cause,
                    &mut ready_at,
                    &mut limiting,
                    &mut tied,
                )?;
            }
        }
        // SM86 RF reservations are Control/Allocate resource observations,
        // not scheduler eligibility. Huerta's Ampere model allows CGGTY issue
        // to continue while the fixed-latency operand-read window reserves its
        // downstream port. Keep the reservation transactional, but do not
        // charge its conflict as a pre-issue readiness cause here.

        if self.instruction_front_end.is_some() {
            let until = warp.instruction_front_end_ready_at().ok_or(
                IssueError::InstructionBufferMissing {
                    warp: warp_index,
                    instruction: warp.next,
                },
            )?;
            observe_readiness_constraint(
                projected,
                base,
                until,
                ReadinessCause::InstructionFrontEnd,
                &mut ready_at,
                &mut limiting,
                &mut tied,
            )?;
        }
        if let Some(memory_pipeline) = self.memory_issue_pipeline.as_ref()
            && self
                .memory_pipeline_slot(u64::from(warp.insts[warp.next].pc))
                .is_some()
            && !matches!(kind, AccessKind::PredicatedOffMemory)
        {
            observe_readiness_constraint(
                projected,
                base,
                memory_pipeline.admission_ready_at(subcore),
                ReadinessCause::MemoryIssueQueue,
                &mut ready_at,
                &mut limiting,
                &mut tied,
            )?;
        }

        if let Some(feedback) = self.memory_admission_feedback.as_ref()
            && feedback.matches_kind(kind)
        {
            let until = feedback
                .admission_ready_at(subcore, kind)
                .expect("selected feedback instruction has a priced queue frontier");
            observe_readiness_constraint(
                projected,
                base,
                until,
                ReadinessCause::MemoryAdmissionQueue,
                &mut ready_at,
                &mut limiting,
                &mut tied,
            )?;
        }
        if let Some(feedback) = self.memory_admission_feedback.as_ref()
            && feedback.candidate.membar_waits_for_service_completion
            && is_gpu_membar_issue(slot)
        {
            let until = feedback
                .completion_ready_at(warp_index, slot)
                .expect("selected MEMBAR has a priced write-completion frontier");
            observe_readiness_constraint(
                projected,
                base,
                until,
                ReadinessCause::MemoryWriteCompletion,
                &mut ready_at,
                &mut limiting,
                &mut tied,
            )?;
        }

        observe_readiness_constraint(
            projected,
            base,
            warp.eligible,
            ReadinessCause::ControlStall,
            &mut ready_at,
            &mut limiting,
            &mut tied,
        )?;
        observe_readiness_constraint(
            projected,
            base,
            warp.cta_barrier_ready,
            ReadinessCause::CtaBarrierRendezvous,
            &mut ready_at,
            &mut limiting,
            &mut tied,
        )?;
        observe_readiness_constraint(
            projected,
            base,
            warp.barrier_pipeline_ready,
            ReadinessCause::PipelineBarrier,
            &mut ready_at,
            &mut limiting,
            &mut tied,
        )?;
        observe_readiness_constraint(
            projected,
            base,
            warp.branch_pipeline_ready,
            ReadinessCause::PipelineBranch,
            &mut ready_at,
            &mut limiting,
            &mut tied,
        )?;

        for barrier in 0..DEP_BARRIERS {
            if !slot.control.wait_mask.contains_index(barrier) {
                continue;
            }
            debug_assert_eq!(
                warp.barrier_pending[barrier], 0,
                "a selected warp cannot have an unresolved dependence counter"
            );
            observe_readiness_constraint(
                projected,
                base,
                warp.barrier_clear[barrier],
                warp.barrier_origin[barrier].into(),
                &mut ready_at,
                &mut limiting,
                &mut tied,
            )?;
        }
        for barrier in 0..DEP_BARRIERS {
            if !warp.ordinary_dependency_wait_mask.contains_index(barrier)
                || slot.control.wait_mask.contains_index(barrier)
            {
                continue;
            }
            debug_assert_eq!(
                warp.barrier_pending[barrier], 0,
                "a selected explicit dependency-barrier wait cannot have an unresolved counter"
            );
            observe_readiness_constraint(
                projected,
                base,
                warp.barrier_clear[barrier],
                warp.barrier_origin[barrier].into(),
                &mut ready_at,
                &mut limiting,
                &mut tied,
            )?;
        }

        if self.has_async_copy_protocol && slot.op_class == OpClass::AsyncCopyCommit {
            let commits_nonempty = self.async_copies[warp_index]
                .state
                .as_deref()
                .and_then(|state| state.open_group)
                .is_some();
            if commits_nonempty {
                projected.async_groups.groups_committed = checked_u64_counter_add(
                    "committed asynchronous-copy group diagnostic",
                    projected.async_groups.groups_committed,
                    1,
                )?;
            }
        }

        if self.has_async_copy_protocol && slot.op_class == OpClass::AsyncCopyWait {
            let prefix = self.async_copies[warp_index]
                .async_wait_prefix_len(slot.async_copy_wait_max_pending_groups);
            let committed_groups_before = self.async_copies[warp_index]
                .state
                .as_deref()
                .map_or(0, |state| state.committed_groups.len());
            let wait_frontier = self.async_copies[warp_index]
                .state
                .as_deref()
                .and_then(|state| {
                    state
                        .committed_groups
                        .iter()
                        .take(prefix)
                        .map(|group| group.completion_time)
                        .max()
                })
                .unwrap_or(0);
            async_snapshot = Some((
                u64::try_from(committed_groups_before).map_err(|_| {
                    IssueError::CounterOverflow {
                        counter: "async-decision committed-group cardinality",
                    }
                })?,
                u64::try_from(prefix).map_err(|_| IssueError::CounterOverflow {
                    counter: "async-decision required-prefix cardinality",
                })?,
                wait_frontier,
            ));
            if let Some(diagnostics) = projected.async_wait_prefix.as_deref_mut() {
                diagnostics.observe(
                    slot.async_copy_wait_max_pending_groups,
                    u64::try_from(committed_groups_before).map_err(|_| {
                        IssueError::CounterOverflow {
                            counter: "async-wait committed-group cardinality diagnostic",
                        }
                    })?,
                    u64::try_from(prefix).map_err(|_| IssueError::CounterOverflow {
                        counter: "async-wait consumed-prefix cardinality diagnostic",
                    })?,
                    wait_frontier.saturating_sub(base),
                )?;
            }
            if let Some(state) = self.async_copies[warp_index].state.as_deref() {
                for group in state.committed_groups.iter().take(prefix) {
                    debug_assert!(
                        group.is_complete_at(now),
                        "a selected async wait requires a complete committed prefix"
                    );
                    let local_completion = group.local_completion_time(&self.params)?;
                    let cause = group.limiting_readiness_cause(&self.params)?;
                    observe_readiness_constraint(
                        projected,
                        base,
                        group.completion_time,
                        cause,
                        &mut ready_at,
                        &mut limiting,
                        &mut tied,
                    )?;

                    projected.async_groups.groups_waited = checked_u64_counter_add(
                        "waited asynchronous-copy group diagnostic",
                        projected.async_groups.groups_waited,
                        1,
                    )?;
                    let source_cycles = group
                        .source_completion_time
                        .saturating_sub(group.committed_at);
                    projected.async_groups.source_cycles = checked_u64_counter_add(
                        "asynchronous-copy committed-to-source diagnostic",
                        projected.async_groups.source_cycles,
                        source_cycles,
                    )?;
                    let local_cycles =
                        local_completion.saturating_sub(group.source_completion_time);
                    projected.async_groups.destination_local_cycles = checked_u64_counter_add(
                        "asynchronous-copy local destination diagnostic",
                        projected.async_groups.destination_local_cycles,
                        local_cycles,
                    )?;
                    let service_cycles = group.completion_time.saturating_sub(local_completion);
                    projected.async_groups.destination_sm_service_cycles = checked_u64_counter_add(
                        "asynchronous-copy SM destination-service diagnostic",
                        projected.async_groups.destination_sm_service_cycles,
                        service_cycles,
                    )?;
                    projected.async_groups.destination_groups_observed = checked_u64_counter_add(
                        "asynchronous-copy destination groups observed",
                        projected.async_groups.destination_groups_observed,
                        u64::from(group.destination_wavefronts != 0),
                    )?;
                    projected.async_groups.destination_wavefronts_observed =
                        checked_u64_counter_add(
                            "asynchronous-copy destination wavefronts observed",
                            projected.async_groups.destination_wavefronts_observed,
                            group.destination_wavefronts,
                        )?;
                    let hidden = group
                        .completion_time
                        .min(base)
                        .saturating_sub(group.committed_at);
                    projected.async_groups.completion_hidden_before_wait_cycles =
                        checked_u64_counter_add(
                            "asynchronous-copy hidden-before-wait diagnostic",
                            projected.async_groups.completion_hidden_before_wait_cycles,
                            hidden,
                        )?;
                    if let Some(observed) = self.async_copies[warp_index].bypass_copy_projection(
                        warp_index,
                        group,
                        &self.params,
                        base,
                    )? {
                        let diagnostics = projected
                            .bypass_async_copies
                            .get_or_insert_with(|| Box::new(BypassAsyncCopyDiagnostics::default()));
                        diagnostics.add_checked(&BypassAsyncCopyDiagnostics {
                            copies_observed: observed.copies,
                            destination_wavefronts_observed: observed.bank_wavefront_upper_bound,
                            bypass_wavefronts_observed: observed.bypass_wavefront_upper_bound,
                            source_cohorts_observed: observed.source_cohorts,
                            destination_lines_observed: observed.destination_lines,
                            max_bank_wavefronts_sum: observed.max_bank_wavefronts_sum,
                            max_bypass_wavefronts_sum: observed.max_bypass_wavefronts_sum,
                            copy_source_spread_cycles: observed.source_spread_cycles,
                            per_copy_completion_cycles: observed.per_copy_completion_cycles,
                            group_aggregate_excess_cycles: observed.group_aggregate_excess_cycles,
                            topology_candidate_penalty_cycles: observed
                                .topology_candidate_penalty_cycles,
                            topology_candidate_completion_cycles: observed
                                .topology_candidate_completion_cycles,
                            topology_candidate_later_cycles: observed
                                .topology_candidate_later_cycles,
                            topology_candidate_earlier_cycles: observed
                                .topology_candidate_earlier_cycles,
                            pipeline_groups_observed: 1,
                            pipeline_q1_completion_cycles: observed.pipeline_q1_completion_cycles,
                            pipeline_q1_exposure_cycles: observed.pipeline_q1_exposure_cycles,
                            pipeline_q2_completion_cycles: observed.pipeline_q2_completion_cycles,
                            pipeline_q2_exposure_cycles: observed.pipeline_q2_exposure_cycles,
                            pipeline_q4_completion_cycles: observed.pipeline_q4_completion_cycles,
                            pipeline_q4_exposure_cycles: observed.pipeline_q4_exposure_cycles,
                            pipeline_q8_completion_cycles: observed.pipeline_q8_completion_cycles,
                            pipeline_q8_exposure_cycles: observed.pipeline_q8_exposure_cycles,
                            pipeline_q16_completion_cycles: observed.pipeline_q16_completion_cycles,
                            pipeline_q16_exposure_cycles: observed.pipeline_q16_exposure_cycles,
                            cohort_groups_observed: observed.cohort_groups_observed,
                            cohort_copies_observed: observed.cohort_copies_observed,
                            cohort_response_memberships_observed: observed
                                .cohort_response_memberships_observed,
                            cohort_response_span_cycles: observed.cohort_response_span_cycles,
                            cohort_topology_penalty_cycles: observed.cohort_topology_penalty_cycles,
                            cohort_candidate_completion_cycles: observed
                                .cohort_candidate_completion_cycles,
                            cohort_candidate_exposure_cycles: observed
                                .cohort_candidate_exposure_cycles,
                            cohort_candidate_later_than_per_copy_cycles: observed
                                .cohort_candidate_later_than_per_copy_cycles,
                            cohort_candidate_earlier_than_per_copy_cycles: observed
                                .cohort_candidate_earlier_than_per_copy_cycles,
                            cohort_candidate_later_than_group_aggregate_cycles: observed
                                .cohort_candidate_later_than_group_aggregate_cycles,
                            cohort_candidate_earlier_than_group_aggregate_cycles: observed
                                .cohort_candidate_earlier_than_group_aggregate_cycles,
                            cohort_candidate_later_than_production_cycles: observed
                                .cohort_candidate_later_than_production_cycles,
                            cohort_candidate_earlier_than_production_cycles: observed
                                .cohort_candidate_earlier_than_production_cycles,
                            cohort_overlap_groups_observed: observed.cohort_overlap_groups_observed,
                            cohort_response_envelope_cycles: observed
                                .cohort_response_envelope_cycles,
                            cohort_topology_prearrival_hidden_cycles: observed
                                .cohort_topology_prearrival_hidden_cycles,
                            cohort_topology_stream_hidden_cycles: observed
                                .cohort_topology_stream_hidden_cycles,
                            cohort_topology_residual_exposure_cycles: observed
                                .cohort_topology_residual_exposure_cycles,
                        })?;
                    }
                    let limited_counter = match cause {
                        ReadinessCause::AsyncSource => {
                            &mut projected.async_groups.source_limited_groups
                        }
                        ReadinessCause::AsyncDestinationLocal => {
                            &mut projected.async_groups.destination_local_limited_groups
                        }
                        ReadinessCause::AsyncDestinationSmService => {
                            &mut projected.async_groups.destination_sm_service_limited_groups
                        }
                        _ => unreachable!("async-group classifier returns an async cause"),
                    };
                    *limited_counter = checked_u64_counter_add(
                        "asynchronous-copy limiting-arm diagnostic",
                        *limited_counter,
                        1,
                    )?;
                }
            }
        }

        if let PipelineClass::Branch {
            next_issue_interval_cycles,
        } = slot.pipeline_class
        {
            projected.control_pipeline.branch_instructions_issued = checked_u64_counter_add(
                "branch-pipeline issued-instruction diagnostic",
                projected.control_pipeline.branch_instructions_issued,
                1,
            )?;
            projected.control_pipeline.branch_resolution_cycles = checked_u64_counter_add(
                "branch-pipeline resolution-cycle diagnostic",
                projected.control_pipeline.branch_resolution_cycles,
                next_issue_interval_cycles,
            )?;
        }

        if let PipelineClass::Barrier {
            warp_initiation_interval_cycles,
            ..
        } = slot.pipeline_class
        {
            projected.barrier_pipeline.instructions_issued = checked_u64_counter_add(
                "CTA-barrier pipeline issued-instruction diagnostic",
                projected.barrier_pipeline.instructions_issued,
                1,
            )?;
            projected.barrier_pipeline.issue_server_busy_cycles = checked_u64_counter_add(
                "CTA-barrier pipeline busy-cycle diagnostic",
                projected.barrier_pipeline.issue_server_busy_cycles,
                warp_initiation_interval_cycles,
            )?;
        }

        if matches!(slot.pipeline_class, PipelineClass::Tensor { .. }) {
            observe_readiness_constraint(
                projected,
                base,
                self.tensor_pipe_ready[subcore],
                ReadinessCause::PipelineTensor,
                &mut ready_at,
                &mut limiting,
                &mut tied,
            )?;
        }
        if matches!(slot.pipeline_class, PipelineClass::Sfu { .. }) {
            observe_readiness_constraint(
                projected,
                base,
                self.sfu_pipe_ready[subcore],
                ReadinessCause::PipelineSfu,
                &mut ready_at,
                &mut limiting,
                &mut tied,
            )?;
            if self.params.arch == ArchId::Sm86 && slot.is_sfu_dependent_chain() {
                observe_readiness_constraint(
                    projected,
                    base,
                    self.warps[warp_index].sfu_dependency_ready,
                    ReadinessCause::PipelineSfu,
                    &mut ready_at,
                    &mut limiting,
                    &mut tied,
                )?;
            }
        }
        if matches!(slot.pipeline_class, PipelineClass::ScalarAlu { .. }) {
            let until = self
                .scalar_execution_pipeline
                .as_ref()
                .map_or(0, |pipeline| pipeline.admission_ready_at(subcore));
            observe_readiness_constraint(
                projected,
                base,
                until,
                ReadinessCause::PipelineAlu,
                &mut ready_at,
                &mut limiting,
                &mut tied,
            )?;
        }

        let effective_ready = ready_at.max(base);
        if let Some((committed_groups_before, required_prefix_len, readiness_cycle)) =
            async_snapshot
        {
            if projected.async_decision_snapshots.len() >= 1_024 {
                projected.async_decision_snapshots_dropped = projected
                    .async_decision_snapshots_dropped
                    .checked_add(1)
                    .ok_or(IssueError::ReadinessDiagnostic(
                        ReadinessCounterError::Overflow {
                            field: "readiness.async_decision_snapshots_dropped",
                            accumulated: u64::MAX,
                            addend: 1,
                        },
                    ))?;
            } else {
                let inst = warp.insts[warp.next];
                projected
                    .async_decision_snapshots
                    .push(AsyncDecisionSnapshot {
                        function_unique_id: self.schedule.function_unique_id(),
                        pc: inst.pc,
                        warp_slot: warp.warp_slot,
                        instruction: u64::try_from(warp.next).unwrap_or(u64::MAX),
                        protocol: "committed_async_group_prefix_wait",
                        committed_groups_before,
                        required_prefix_len,
                        wait_threshold: slot.async_copy_wait_max_pending_groups,
                        readiness_cycle,
                        decision_cycle: effective_ready,
                        issue_cycle: now,
                        observation_scope: "selected_warp_issue_opportunity",
                        semantics: "selected_only_not_blocked_witness",
                        membership: "committed-prefix membership from scalar model state",
                        limiting_cause: limiting.map(readiness_cause_name),
                    });
            }
        }
        if effective_ready > base {
            projected.limiting_cycles.add_cause(
                limiting.expect("a post-baseline ready frontier has a cause"),
                effective_ready - base,
            )?;
        }
        if tied != 0 {
            projected.tied_limiting_causes = checked_u64_counter_add(
                "tied limiting readiness cause diagnostic",
                projected.tied_limiting_causes,
                tied,
            )?;
        }
        if now > effective_ready {
            let delayed = now - effective_ready;
            projected
                .limiting_cycles
                .add_eligible_not_selected(delayed)?;
            projected
                .constraint_exposure_cycles
                .add_eligible_not_selected(delayed)?;
            projected.delayed_ready_issues = checked_u64_counter_add(
                "delayed ready issue diagnostic",
                projected.delayed_ready_issues,
                1,
            )?;
        }
        Ok(())
    }

    /// Run one fetch/decode opportunity per SMSP, then one SM-arbitrated stream
    /// prefetch. The fetch picker mirrors CGGTY: continue the last-issued warp
    /// while its instruction buffer has room, otherwise choose the youngest
    /// resident warp with capacity.
    fn instruction_front_end_cycle(&mut self) -> Result<(), IssueError> {
        let Some(original_state) = self.instruction_front_end.take() else {
            return Ok(());
        };
        let now = self.t;
        let mut original = Some(original_state);
        let capacity = match original
            .as_ref()
            .expect("front-end state owns the active cycle")
            .instruction_buffer_capacity()
        {
            Ok(capacity) => capacity,
            Err(error) => {
                self.instruction_front_end = original;
                return Err(error);
            }
        };
        let fetches: [Option<usize>; SUB_CORES] = std::array::from_fn(|subcore| {
            let can_fetch = |warp: usize| self.warps[warp].can_fetch_instruction(capacity);
            match self.fetch_last_issued[subcore] {
                Some(warp) if can_fetch(warp) => Some(warp),
                _ => self.subcore_warps[subcore]
                    .iter()
                    .copied()
                    .filter(|&warp| can_fetch(warp))
                    .max_by_key(|&warp| (self.warps[warp].arrival, self.warps[warp].warp_id)),
            }
        });
        let use_shadow = original
            .as_ref()
            .expect("front-end state owns the active cycle")
            .needs_transactional_cycle(&fetches, now);
        let mut front_end = if use_shadow {
            original
                .as_ref()
                .expect("shadow cycle retains original front-end state")
                .clone()
        } else {
            original
                .take()
                .expect("normal cycle takes ownership of front-end state")
        };
        front_end.maintain(now);
        let mut fetched = [None; SUB_CORES];
        for (subcore, fetch) in fetches.into_iter().enumerate() {
            let Some(warp) = fetch else { continue };
            let cursor = self.warps[warp].front_end_fetch_cursor;
            let pc = self.warps[warp].insts[cursor].pc;
            let ready_at = match front_end.fetch_instruction(
                subcore,
                self.schedule.function_unique_id,
                pc,
                now,
            ) {
                Ok(ready_at) => ready_at,
                Err(error) => {
                    self.instruction_front_end = if use_shadow {
                        original
                    } else {
                        Some(front_end)
                    };
                    return Err(error);
                }
            };
            if self.fetch_last_issued[subcore] == Some(warp) {
                front_end.diagnostics.fetch_last_issued_continuations = checked_u64_counter_add(
                    "fetch last-issued continuation diagnostic",
                    front_end.diagnostics.fetch_last_issued_continuations,
                    1,
                )?;
                if self.hold[subcore] != Some(warp) {
                    front_end.diagnostics.yield_hold_separations = checked_u64_counter_add(
                        "fetch Yield/issue-hold separation diagnostic",
                        front_end.diagnostics.yield_hold_separations,
                        1,
                    )?;
                }
            }
            fetched[subcore] = Some((warp, ready_at));
        }
        if let Err(error) = front_end.prefetch_cycle(now) {
            self.instruction_front_end = if use_shadow {
                original
            } else {
                Some(front_end)
            };
            return Err(error);
        }
        // Validate every warp-side commit before applying the first one. The
        // checks mirror `push_fetched_instruction`; after they pass, direct
        // field updates cannot fail and preserve the cycle transaction.
        for &(warp, _) in fetched.iter().flatten() {
            let state = &self.warps[warp];
            if !state.can_fetch_instruction(capacity)
                || state.front_end_fetch_cursor.checked_add(1).is_none()
            {
                let error = IssueError::InstructionBufferMissing {
                    warp,
                    instruction: state.front_end_fetch_cursor,
                };
                self.instruction_front_end = if use_shadow {
                    original
                } else {
                    Some(front_end)
                };
                return Err(error);
            }
        }
        for &(warp, ready_at) in fetched.iter().flatten() {
            let state = &mut self.warps[warp];
            let instruction = state.front_end_fetch_cursor;
            state.instruction_buffer.push_back(BufferedInstruction {
                instruction,
                ready_at,
            });
            state.front_end_fetch_cursor = instruction
                .checked_add(1)
                .expect("front-end fetch cursor was preflighted");
        }
        self.instruction_front_end = Some(front_end);
        Ok(())
    }

    fn has_instruction_front_end_work(&self) -> Result<bool, IssueError> {
        let Some(front_end) = self.instruction_front_end.as_ref() else {
            return Ok(false);
        };
        let capacity = front_end.instruction_buffer_capacity()?;
        Ok(front_end.has_prefetch_work()
            || self
                .subcore_warps
                .iter()
                .flatten()
                .copied()
                .any(|warp| self.warps[warp].can_fetch_instruction(capacity)))
    }
}


