//! Control-flow, warp retirement, and cycle advancement; mutates SMSP timing state.

use super::*;

impl<'a> SmCore<'a> {
    /// Consume the native comparator's selected execution path. Exact replay
    /// returns no macro transition; the typed-macro branch performs the native
    /// state mutation and returns its continuation frontier.
    #[cfg(feature = "reference")]
    pub(crate) fn execute_reference_macro_path(
        &mut self,
        path: &NativeBoundaryExecutionPath,
        warp_slot: i32,
        instruction_count: u64,
        initiation_interval_cycles: u64,
        terminal_block_barrier: bool,
        detailed_effects: &[TypedEventTransition],
        terminal_front_end_exit: Option<&InstructionFrontEndBoundaryState>,
    ) -> Result<Option<ReferenceScalarMacroExecution>, IssueError> {
        match path {
            NativeBoundaryExecutionPath::TypedMacro => self
                .execute_reference_scalar_macro(
                    warp_slot,
                    instruction_count,
                    initiation_interval_cycles,
                    terminal_block_barrier,
                    detailed_effects,
                    terminal_front_end_exit,
                )
                .map(Some),
            NativeBoundaryExecutionPath::ExactReplay { .. } => Ok(None),
        }
    }
}

impl<'a> SmCore<'a> {
    #[cfg(feature = "reference")]
    pub(crate) fn reference_issue_effects_since(&self, start: usize) -> &[(u64, u64, u32)] {
        self.reference_issue_effects.get(start..).unwrap_or(&[])
    }
}

impl<'a> SmCore<'a> {
    #[cfg(feature = "reference")]
    pub(crate) fn typed_local_completion_time(&self) -> Option<u64> {
        self.local_completion_time
    }
}

impl<'a> SmCore<'a> {
    #[cfg(feature = "reference")]
    pub(crate) fn typed_unprojected_timing_state(&self) -> Vec<String> {
        let mut reasons = Vec::new();
        if self.sfu_pipe_ready.iter().any(|&ready| ready != 0) {
            // This resource survives CTA retirement/admission, but schema 15
            // does not serialize its availability. Never certify an exact
            // continuation using only the independent tensor frontier.
            reasons.push("sfu_pipeline_state".to_owned());
        }
        if !self.unmodeled_tails.is_empty() {
            // TailFlags are capability boundaries for completion effects that
            // are deliberately outside the modeled event stream. Preserve
            // them as a contract-level refusal instead of allowing a typed
            // boundary to imply complete writeback/coherence/RMW behavior.
            reasons.push("unmodeled_completion_tails".to_owned());
        }
        if let MemModel::L1Shared(state) = &self.mem
            && (!state.fill_waiters.is_empty()
                || !state.fill_masks.is_empty()
                || !state.line_folds.is_empty()
                || !state.retry.is_empty())
        {
            // These shared-L2 side maps select which parked sector range,
            // waiter, and per-fill mask transition runs after a response.
            // L1 tags and response owners alone cannot reconstruct that
            // transaction boundary.
            reasons.push("shared_l2_transient_state".to_owned());
        }
        if self
            .typed_l1_state()
            .is_some_and(|l1| !l1.line_fills.is_empty() || !l1.sector_fills.is_empty())
        {
            // The typed response owner carries L2 allocation identity, while
            // the L1 response handler consumes a private FillKey generation.
            // Without the owner-to-L1-fill edge, a live response cannot be
            // resumed exactly even though the L1 fill timing itself is typed.
            reasons.push("l1_fill_response_identity".to_owned());
        }
        if self
            .warps
            .iter()
            .any(|warp| warp.barrier_pending.iter().any(|&pending| pending != 0))
        {
            // Response owners retain requester/fill identity, but the typed
            // owner schema does not yet carry the dependency-barrier index
            // that each response clears. A live counter is therefore not
            // sufficient to resume the next issue transition exactly.
            reasons.push("dependency_barrier_ownership".to_owned());
        }
        let future_constant_path = self.warps.iter().any(|warp| {
            warp.insts.get(warp.next..).is_some_and(|remaining| {
                remaining
                    .iter()
                    .any(|instruction| self.is_constant_path_pc(u64::from(instruction.pc)))
            })
        });
        if future_constant_path
            && self
                .immediate_constant_cache
                .as_ref()
                .is_some_and(|caches| {
                    caches.iter().any(|cache| {
                        cache.sets.iter().any(|set| !set.is_empty())
                            || !cache.pending.is_empty()
                            || cache.next_service != 0
                            || !cache.diagnostics.is_empty()
                    })
                })
        {
            reasons.push("immediate_constant_cache_state".to_owned());
        }
        if self
            .shared_write_lookup_runtime
            .async_l1_hit_service_ready
            .iter()
            .any(|&ready| ready != 0)
        {
            reasons.push("async_l1_hit_service_state".to_owned());
        }
        if self.result_writebacks.is_some() {
            reasons.push("result_writeback_state".to_owned());
        }
        if !self.memory_service_candidates.is_empty() {
            reasons.push("memory_service_candidates".to_owned());
        }
        if self.memory_admission_feedback.is_some() {
            reasons.push("memory_admission_feedback".to_owned());
        }
        reasons.sort_unstable();
        reasons
    }
}

impl<'a> SmCore<'a> {
    #[cfg(feature = "reference")]
    pub(crate) fn typed_scalar_pipeline_state(&self) -> Option<ScalarPipelineBoundaryState> {
        self.scalar_execution_pipeline.as_ref().map(|state| {
            let diagnostics = state.diagnostics;
            ScalarPipelineBoundaryState {
                latency_cycles: state.config.latency_cycles,
                initiation_interval_cycles: state.config.initiation_interval_cycles,
                in_flight_capacity: state.config.in_flight_capacity,
                release_times: state
                    .release_times
                    .iter()
                    .map(|queue| queue.iter().copied().collect())
                    .collect(),
                next_service: state.next_service.to_vec(),
                instructions_admitted: diagnostics.instructions_admitted,
                service_busy_cycles: diagnostics.service_busy_cycles,
                queue_delay_cycles: diagnostics.queue_delay_cycles,
                maximum_queue_occupancy: diagnostics.maximum_queue_occupancy,
            }
        })
    }
}

impl<'a> SmCore<'a> {
    #[cfg(feature = "reference")]
    pub(crate) fn boundary_snapshot(&self) -> BoundarySnapshot {
        let warp_states = self
            .warps
            .iter()
            .map(|warp| WarpBoundaryState {
                warp_slot: warp.warp_slot,
                next_instruction: u64::try_from(warp.next).unwrap_or(u64::MAX),
                front_end_fetch_cursor: u64::try_from(warp.front_end_fetch_cursor)
                    .unwrap_or(u64::MAX),
                instruction_buffer: warp
                    .instruction_buffer
                    .iter()
                    .map(|buffered| {
                        (
                            u64::try_from(buffered.instruction).unwrap_or(u64::MAX),
                            buffered.ready_at,
                        )
                    })
                    .collect(),
                eligible: warp.eligible,
                imc_wait_until: warp.imc_wait_until,
                barrier_clear: warp.barrier_clear,
                barrier_pending: warp.barrier_pending,
                block: u64::try_from(warp.block).unwrap_or(u64::MAX),
                arrival: warp.arrival,
                at_block_barrier: warp.at_block_barrier,
                cta_barrier_ready: warp.cta_barrier_ready,
                barrier_pipeline_ready: warp.barrier_pipeline_ready,
                branch_pipeline_ready: warp.branch_pipeline_ready,
                block_barrier_dispatch: warp.block_barrier_dispatch,
                block_barrier_arrival: warp.block_barrier_arrival,
                done: warp.done,
            })
            .collect();
        let block_states = self
            .block_states
            .iter()
            .map(|block| BlockBoundaryState {
                unfinished: u64::try_from(block.unfinished).unwrap_or(u64::MAX),
                waiting: u64::try_from(block.waiting).unwrap_or(u64::MAX),
                warp_count: block.warp_count,
                shared_memory_bytes: block.shared_memory_bytes,
                warp_slots: block.warp_slots.to_vec(),
                first_warp: u64::try_from(block.first_warp).unwrap_or(u64::MAX),
                admitted_warps: u64::try_from(block.admitted_warps).unwrap_or(u64::MAX),
                barrier_generation: block.barrier_generation,
                barrier_first_dispatch: block.barrier_first_dispatch,
                barrier_first_arrival: block.barrier_first_arrival,
                barrier_rearm_ready: block.barrier_rearm_ready,
            })
            .collect();
        let hold = self.hold.map(|entry| {
            entry.and_then(|warp| {
                self.warps
                    .get(warp)
                    .map(|state| u64::try_from(state.warp_slot).unwrap_or(0))
            })
        });
        let subcore_warps = self
            .subcore_warps
            .iter()
            .map(|warps| {
                warps
                    .iter()
                    .filter_map(|&warp| self.warps.get(warp).map(|state| state.warp_slot))
                    .collect()
            })
            .collect();
        BoundarySnapshot {
            schema_version: sim_semantics::BOUNDARY_SCHEMA_VERSION,
            cycle: self.t,
            issued: self.issued,
            resident_warps: self.resident_warps,
            resident_shared_memory_bytes: self.resident_shared_memory_bytes,
            unfinished_warps: u64::try_from(self.unfinished_warps).unwrap_or(u64::MAX),
            next_block: u64::try_from(self.next_block).unwrap_or(u64::MAX),
            hold,
            subcore_warps,
            outstanding_responses: u64::try_from(self.outstanding_responses()).unwrap_or(u64::MAX),
            l1_state: match &self.mem {
                MemModel::Flat => None,
                MemModel::L1Const(cache) => Some(cache.boundary_state()),
                MemModel::L1Shared(state) => Some(state.l1.boundary_state()),
            },
            instruction_front_end_state: self
                .instruction_front_end
                .as_ref()
                .map(|state| format!("{state:?}")),
            memory_issue_pipeline_state: self
                .memory_issue_pipeline
                .as_ref()
                .map(|state| format!("{state:?}")),
            scalar_execution_pipeline_state: self
                .scalar_execution_pipeline
                .as_ref()
                .map(|state| format!("{state:?}")),
            async_copy_state: self
                .async_copies
                .iter()
                .map(|slot| slot.state.as_deref().map(|state| format!("{state:?}")))
                .collect(),
            barrier_pipe_ready: self.barrier_pipe_ready,
            tensor_pipe_ready: self.tensor_pipe_ready,
            ldgsts_destination_pipe_ready: self.ldgsts_destination_pipe_ready,
            warp_states,
            block_states,
            omitted_state: vec!["legacy_core_debug_fields_retained".to_owned()],
        }
    }
}

impl<'a> SmCore<'a> {
    /// Retire one warp after its final instruction has become semantically
    /// complete. Ordinary instructions call this immediately; a final
    /// BAR.SYNC calls it from the CTA-release path so the barrier remains a
    /// real synchronization point.
    pub(crate) fn retire_warp(
        &mut self,
        subcore: usize,
        warp_index: usize,
        t: u64,
    ) -> Result<(), IssueError> {
        if self.warps[warp_index].done {
            return Ok(());
        }
        let block = self.warps[warp_index].block;
        let admission_time = self.warps[warp_index].admission_time;
        let warp_slot = self.warps[warp_index].warp_slot;
        let pos = self.warps[warp_index].pos_in_subcore;
        if self.subcore_warps[subcore].get(pos).copied() != Some(warp_index) {
            return Err(IssueError::CounterUnderflow {
                counter: "sub-core resident warp",
            });
        }
        self.warp_lifetimes.push(WarpLifetime {
            block: self.warps[warp_index].launch_index,
            warp_slot,
            admission: admission_time,
            retirement: checked_cycle_add("warp activity end", t, 1)?,
        });
        // A final BAR.SYNC is parked with itself in the CGGTY hold slot while
        // it waits for the CTA rendezvous.  Clear that slot before removing
        // the warp: swap_remove can otherwise leave a stale index that now
        // names a different resident warp (or a done warp), changing the
        // next pick/fetch decision after release.  Ordinary retirement also
        // benefits from the same invariant.
        if self.hold[subcore] == Some(warp_index) {
            self.hold[subcore] = None;
        }
        if self.fetch_last_issued[subcore] == Some(warp_index) {
            self.fetch_last_issued[subcore] = None;
        }
        self.warps[warp_index].done = true;
        self.warps[warp_index].next_slot = None;
        self.warps[warp_index].ordinary_dependency_wait_mask = WaitMask::NONE;
        self.warps[warp_index].at_block_barrier = false;
        self.warps[warp_index].block_barrier_dispatch = None;
        self.warps[warp_index].block_barrier_arrival = None;
        self.unfinished_warps =
            self.unfinished_warps
                .checked_sub(1)
                .ok_or(IssueError::CounterUnderflow {
                    counter: "unfinished warp",
                })?;
        self.block_states[block].unfinished = self.block_states[block]
            .unfinished
            .checked_sub(1)
            .ok_or(IssueError::CounterUnderflow {
                counter: "block unfinished warp",
            })?;

        // Retire from the sub-core pick list (swap-remove + position fix-up)
        // so scans remain bounded by resident warps.
        self.subcore_warps[subcore].swap_remove(pos);
        if let Some(&moved) = self.subcore_warps[subcore].get(pos) {
            self.warps[moved].pos_in_subcore = pos;
        }

        if self.block_states[block].unfinished == 0 {
            let state = &self.block_states[block];
            self.resident_warps = self.resident_warps.checked_sub(state.warp_count).ok_or(
                IssueError::CounterUnderflow {
                    counter: "resident warp",
                },
            )?;
            self.resident_shared_memory_bytes = self
                .resident_shared_memory_bytes
                .checked_sub(state.shared_memory_bytes)
                .ok_or(IssueError::CounterUnderflow {
                    counter: "resident shared-memory byte",
                })?;
            for &warp_slot in &state.warp_slots {
                let removed = self.resident_warp_slots.remove(&warp_slot);
                if !removed {
                    return Err(IssueError::ResidentWarpSlotMissing {
                        launch_index: self.warps[state.first_warp].launch_index,
                        warp_slot,
                    });
                }
            }
            let admit_at = checked_cycle_add("block-wave admission", t, 1)?;
            self.admit_blocks(admit_at)?;
        }
        Ok(())
    }
}

impl<'a> SmCore<'a> {
    /// Run one fetch/decode opportunity per SMSP, then one SM-arbitrated stream
    /// prefetch. The fetch picker mirrors CGGTY: continue the last-issued warp
    /// while its instruction buffer has room, otherwise choose the youngest
    /// resident warp with capacity.
    pub(crate) fn instruction_front_end_cycle(&mut self) -> Result<(), IssueError> {
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
}

impl<'a> SmCore<'a> {
    pub(crate) fn project_async_issue_overlap(
        &self,
        picks: &[Option<usize>; SUB_CORES],
        projected: &mut ReadinessDiagnostics,
    ) -> Result<(), IssueError> {
        if !self.has_async_copy_protocol {
            return Ok(());
        }

        let mut effective_warps = [None; SUB_CORES];
        let mut copies = 0_u64;
        let mut cache_all = 0_u64;
        let mut bypass = 0_u64;
        for (subcore, pick) in picks.iter().enumerate() {
            let Some(warp) = *pick else { continue };
            let kind = self.warps[warp].insts[self.warps[warp].next].kind;
            if async_copy_source_count(kind).is_none() {
                continue;
            }
            effective_warps[subcore] = Some(warp);
            copies += 1;
            match kind {
                AccessKind::GlobalAsyncCopy {
                    l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                    ..
                } => cache_all += 1,
                AccessKind::GlobalAsyncCopy {
                    l1_policy: LdgstsL1Policy::BypassL1,
                    ..
                } => bypass += 1,
                _ => unreachable!("effective async-copy predicate admits only GlobalAsyncCopy"),
            }
        }
        if copies == 0 {
            return Ok(());
        }

        let mut observation = AsyncIssueOverlapDiagnostics::default();
        if copies >= 2 {
            observation.coissue_cycles = 1;
            observation.coissued_copies = copies;
            observation.cache_all_coissued_copies = cache_all;
            observation.bypass_coissued_copies = bypass;
            observation.mixed_policy_coissue_cycles = u64::from(cache_all != 0 && bypass != 0);
            observation.single_acceptor_excess_copies = copies - 1;
            observation.maximum_coissue_width = copies;
        }

        debug_assert_eq!(self.warps.len(), self.async_copies.len());
        let mut committed_total = 0_u64;
        for (warp, slot) in self.async_copies.iter().enumerate() {
            if self.warps[warp].done {
                continue;
            }
            let groups = slot
                .state
                .as_deref()
                .map_or(0, |state| state.committed_groups.len());
            let groups = u64::try_from(groups).map_err(|_| IssueError::CounterOverflow {
                counter: "async issue overlap committed-group population",
            })?;
            committed_total = checked_u64_counter_add(
                "async issue overlap committed-group population",
                committed_total,
                groups,
            )?;
        }
        for warp in effective_warps.into_iter().flatten() {
            let own_groups = self.async_copies[warp]
                .state
                .as_deref()
                .map_or(0, |state| state.committed_groups.len());
            let own_groups =
                u64::try_from(own_groups).map_err(|_| IssueError::CounterOverflow {
                    counter: "async issue overlap own committed-group population",
                })?;
            let other_groups =
                committed_total
                    .checked_sub(own_groups)
                    .ok_or(IssueError::CounterUnderflow {
                        counter: "async issue overlap other-warp committed-group population",
                    })?;
            if other_groups == 0 {
                continue;
            }
            observation.copies_with_other_warp_committed_groups = checked_readiness_add(
                "readiness.async_issue_overlap.copies_with_other_warp_committed_groups",
                observation.copies_with_other_warp_committed_groups,
                1,
            )?;
            observation.other_warp_committed_groups_sum = checked_readiness_add(
                "readiness.async_issue_overlap.other_warp_committed_groups_sum",
                observation.other_warp_committed_groups_sum,
                other_groups,
            )?;
            observation.maximum_other_warp_committed_groups = observation
                .maximum_other_warp_committed_groups
                .max(other_groups);
        }

        if observation.is_empty() {
            return Ok(());
        }
        if let Some(current) = projected.async_issue_overlap.as_deref_mut() {
            current.add_checked(&observation)?;
        } else {
            projected.async_issue_overlap = Some(Box::new(observation));
        }
        Ok(())
    }
}

impl<'a> SmCore<'a> {
    pub(crate) fn project_picked_readiness(
        &self,
        picks: &[Option<usize>; SUB_CORES],
        barrier_arrival_times: &[Option<u64>; SUB_CORES],
        now: u64,
    ) -> Result<ReadinessDiagnostics, IssueError> {
        let mut projected = (*self.readiness).clone();
        for (subcore, pick) in picks.iter().enumerate() {
            if let Some(warp) = *pick {
                self.project_selected_readiness(subcore, warp, now, &mut projected)?;
                let state = &self.warps[warp];
                let slot = state
                    .next_slot
                    .ok_or(IssueError::MissingIssueSlot { warp })?;
                if is_clocklo_issue(slot) {
                    projected.clock_issue.reads_issued = checked_u64_counter_add(
                        "CLOCKLO reads-issued diagnostic",
                        projected.clock_issue.reads_issued,
                        1,
                    )?;
                    state
                        .clock_reads_issued
                        .checked_add(1)
                        .ok_or(IssueError::CounterOverflow {
                            counter: "warp CLOCKLO issue",
                        })?;
                    if let Some(first) = state.clock_first_issue {
                        let interval =
                            now.checked_sub(first).ok_or(IssueError::CounterUnderflow {
                                counter: "warp CLOCKLO issue interval",
                            })?;
                        if state.clock_reads_issued == 1 {
                            projected.clock_issue.warps_with_interval = checked_u64_counter_add(
                                "CLOCKLO warps-with-interval diagnostic",
                                projected.clock_issue.warps_with_interval,
                                1,
                            )?;
                        }
                        projected.clock_issue.maximum_first_to_last_issue_cycles = projected
                            .clock_issue
                            .maximum_first_to_last_issue_cycles
                            .max(interval);
                    }
                    if self
                        .memory_admission_feedback
                        .as_ref()
                        .is_some_and(|feedback| {
                            feedback.candidate.membar_waits_for_service_completion
                        })
                    {
                        match state.clock_reads_issued {
                            1 => {
                                let first = state.clock_first_issue.ok_or(
                                    IssueError::CounterUnderflow {
                                        counter: "first CLOCKLO phase timestamp",
                                    },
                                )?;
                                let interval =
                                    now.checked_sub(first).ok_or(IssueError::CounterUnderflow {
                                        counter: "first-to-second CLOCKLO issue interval",
                                    })?;
                                projected.clock_issue.maximum_first_to_second_issue_cycles =
                                    projected
                                        .clock_issue
                                        .maximum_first_to_second_issue_cycles
                                        .max(interval);
                            }
                            2 => {
                                let second = self
                                    .memory_admission_feedback
                                    .as_ref()
                                    .and_then(|feedback| feedback.second_clock_issue(warp))
                                    .ok_or(IssueError::CounterUnderflow {
                                        counter: "second CLOCKLO phase timestamp",
                                    })?;
                                let interval = now.checked_sub(second).ok_or(
                                    IssueError::CounterUnderflow {
                                        counter: "second-to-third CLOCKLO issue interval",
                                    },
                                )?;
                                projected.clock_issue.warps_with_three_reads =
                                    checked_u64_counter_add(
                                        "CLOCKLO warps-with-three-reads diagnostic",
                                        projected.clock_issue.warps_with_three_reads,
                                        1,
                                    )?;
                                projected.clock_issue.maximum_second_to_third_issue_cycles =
                                    projected
                                        .clock_issue
                                        .maximum_second_to_third_issue_cycles
                                        .max(interval);
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        self.project_async_issue_overlap(picks, &mut projected)?;
        self.project_cta_barrier_diagnostics(picks, barrier_arrival_times, now, &mut projected)?;
        Ok(projected)
    }
}

impl<'a> SmCore<'a> {
    /// Earliest issue time after both warp/control dependencies and the next
    /// instruction's characterized execution resource are available. A missing
    /// slot remains selectable so the established typed invariant error is
    /// preserved at the transaction boundary.
    pub(crate) fn issue_ready_at_impl<const CHECK_ASYNC: bool>(
        &self,
        subcore: usize,
        warp: usize,
        include_immediate_constant: bool,
    ) -> Option<u64> {
        let mut warp_ready = if include_immediate_constant {
            self.warps[warp].ready_at()?
        } else {
            self.warps[warp].ready_at_without_immediate_constant()?
        };
        if self.params.resource_source() == IssueResourceSource::Sm120Alpha
            && let Some(slot) = self.warps[warp].next_slot
        {
            let mask = self.warps[warp].insts[self.warps[warp].next].effective_lane_mask();
            let register_ready = issue_fixed_register_ready(
                slot.fixed_latency_operands(),
                mask,
                self.warps[warp].insts[self.warps[warp].next].active_lane_mask,
                slot.guard_predicate,
                &self.warps[warp].register_ready,
            );
            warp_ready = warp_ready.max(register_ready.latest());
        }
        if self.warps[warp]
            .next_slot
            .is_some_and(IssueSlot::is_cctl_invalidate_all)
            && let MemModel::L1Shared(state) = &self.mem
            && (state.l1.has_pending_fills() || !state.retry.is_empty())
        {
            // Model-local serialization for whole-cache invalidation. Every
            // accepted fill (including an unobserved/no-scoreboard load) and
            // deferred lookup must finish through its normal response path.
            // Returning None blocks only this warp; siblings remain eligible
            // and the existing event loop wakes us on external responses.
            // This is not a hardware CCTL latency or completion-upper claim.
            return None;
        }
        if self.instruction_front_end.is_some() {
            warp_ready = warp_ready.max(self.warps[warp].instruction_front_end_ready_at()?);
        }
        if let Some(memory_pipeline) = self.memory_issue_pipeline.as_ref() {
            let state = &self.warps[warp];
            let pc = u64::from(state.insts[state.next].pc);
            if self.memory_pipeline_slot(pc).is_some()
                && !matches!(
                    state.insts[state.next].kind,
                    AccessKind::PredicatedOffMemory
                )
            {
                warp_ready = warp_ready.max(memory_pipeline.admission_ready_at(subcore));
            }
        }
        if let Some(feedback) = self.memory_admission_feedback.as_ref() {
            let state = &self.warps[warp];
            let admission_ready =
                feedback.admission_ready_at(subcore, state.insts[state.next].kind)?;
            warp_ready = warp_ready.max(admission_ready);
            if let Some(slot) = state.next_slot {
                let completion_ready = feedback.completion_ready_at(warp, slot)?;
                warp_ready = warp_ready.max(completion_ready);
                let terminal_exit = state
                    .next
                    .checked_add(1)
                    .is_some_and(|next| next == state.insts.len());
                let exit_ready = feedback.exit_ready_at(warp, slot, terminal_exit)?;
                warp_ready = warp_ready.max(exit_ready);
            }
        }
        if self.params.arch == ArchId::Sm86
            && self.warps[warp]
                .next_slot
                .is_some_and(IssueSlot::is_sfu_dependent_chain)
        {
            warp_ready = warp_ready.max(self.warps[warp].sfu_dependency_ready);
        }
        if CHECK_ASYNC
            && let Some(IssueSlot {
                op_class: OpClass::AsyncCopyWait,
                async_copy_wait_max_pending_groups,
                ..
            }) = self.warps[warp].next_slot
        {
            let async_ready = self.async_copies[warp]
                .async_wait_clear_time(*async_copy_wait_max_pending_groups)?;
            warp_ready = warp_ready.max(async_ready);
        }
        let pipeline_ready = match self.warps[warp].next_slot.map(|slot| slot.pipeline_class) {
            Some(PipelineClass::Branch { .. }) => 0,
            Some(PipelineClass::Barrier { .. }) => {
                self.block_states[self.warps[warp].block].barrier_rearm_ready
            }
            Some(PipelineClass::Tensor { .. }) => self.tensor_pipe_ready[subcore],
            Some(PipelineClass::Sfu { .. }) => self.sfu_pipe_ready[subcore],
            Some(PipelineClass::ScalarAlu { .. }) => self
                .scalar_execution_pipeline
                .as_ref()
                .map_or(0, |pipeline| pipeline.admission_ready_at(subcore)),
            Some(PipelineClass::DispatchOnly) | None => 0,
        };
        Some(warp_ready.max(pipeline_ready))
    }
}

impl<'a> SmCore<'a> {
    pub(crate) fn scheduler_candidate<const CHECK_ASYNC: bool>(
        &self,
        subcore: usize,
        warp: usize,
        now: u64,
    ) -> bool {
        self.is_issue_ready::<CHECK_ASYNC>(subcore, warp, now)
            || self.l0c_greedy_hold_candidate::<CHECK_ASYNC>(subcore, warp, now)
            || self.l0c_expired_greedy_probe::<CHECK_ASYNC>(subcore, warp, now)
    }
}

impl<'a> SmCore<'a> {
    pub(crate) fn pick_warps<const CHECK_ASYNC: bool>(
        &self,
        now: u64,
    ) -> [Option<usize>; SUB_CORES] {
        std::array::from_fn(|subcore| {
            // CGGTY pick: held warp while its control state and execution
            // resource are ready, else youngest fully-ready warp.
            match self.hold[subcore] {
                Some(w) if self.scheduler_candidate::<CHECK_ASYNC>(subcore, w, now) => Some(w),
                _ => self.subcore_warps[subcore]
                    .iter()
                    .copied()
                    .filter(|&w| self.is_issue_ready::<CHECK_ASYNC>(subcore, w, now))
                    .max_by_key(|&w| (self.warps[w].arrival, self.warps[w].warp_id)),
            }
        })
    }
}

impl<'a> SmCore<'a> {
    pub(crate) fn next_known_issue_time<const CHECK_ASYNC: bool>(&self) -> Option<u64> {
        let mut next_t: Option<u64> = None;
        for (subcore, warps) in self.subcore_warps.iter().enumerate() {
            for &warp in warps {
                if let Some(ready) = self.issue_ready_at::<CHECK_ASYNC>(subcore, warp) {
                    next_t = Some(next_t.map_or(ready, |current| current.min(ready)));
                }
            }
        }
        next_t
    }
}

impl<'a> SmCore<'a> {
    /// Earliest timestamp at which a committed exact-destination group becomes
    /// eligible to enter the aggregate service. Groups with outstanding source
    /// responses have no locally known wake time and remain event-driven.
    pub(crate) fn next_async_copy_destination_source_time(&self) -> Option<u64> {
        self.async_copies
            .iter()
            .filter_map(|slot| slot.state.as_deref())
            .flat_map(|state| state.committed_groups.iter())
            .filter(|group| {
                group.pending_sources == 0
                    && group.destination_wavefronts != 0
                    && !group.destination_service_queued
                    && group.source_completion_time > self.t
            })
            .map(|group| group.source_completion_time)
            .min()
    }
}

impl<'a> SmCore<'a> {
    /// Commit the state transitions that become due while the actor parks at
    /// an external event horizon. Event mode may jump over idle cycles, but
    /// release queues and fixed-clock completions must be retired at the
    /// parked timestamp before typed state is observed.
    pub(crate) fn park_at_horizon(&mut self, bound: u64) {
        if let Some(cache) = self.immediate_constant_cache.as_mut() {
            for state in cache.iter_mut() {
                state.complete_ready(bound);
            }
        }
        if let Some(memory_pipeline) = self.memory_issue_pipeline.as_mut() {
            memory_pipeline.prune(bound);
        }
        if let Some(scalar_pipeline) = self.scalar_execution_pipeline.as_mut() {
            scalar_pipeline.prune(bound);
        }
        self.t = bound;
    }
}

impl<'a> SmCore<'a> {
    /// Run issue cycles strictly BEFORE `horizon`. Emits memory requests
    /// into `out` (L1Shared mode only).
    ///
    /// `horizon` is the earliest time a SELF-affecting external event can
    /// occur — the caller passes the earliest response already scheduled
    /// for this SM (other SMs' events are NOT barriers: cross-SM ordering
    /// lives entirely in the engine queue's `(time, seq)` order on the L2
    /// access stream). Requests emitted DURING the call tighten the cap
    /// to `issue_time + l1_miss_latency` (the L2-hit lower bound of the
    /// latency ladder) — their responses are not scheduled yet, so the
    /// caller cannot know them. The clock never crosses the cap, not even
    /// by skip-ahead: a response arriving at the cap may ready a warp
    /// EARLIER than the next known timer, and the clock is monotonic.
    pub(crate) fn advance(
        &mut self,
        horizon: ReplayHorizon,
        out: &mut Vec<MemRequest>,
    ) -> Result<Advance, IssueError> {
        #[cfg(feature = "reference")]
        {
            self.advance_diagnostics.calls = self.advance_diagnostics.calls.checked_add(1).ok_or(
                IssueError::CounterOverflow {
                    counter: "reference advance calls",
                },
            )?;
        }
        if let Some(bound) = horizon.bounded_time()
            && bound < self.t
        {
            return Err(IssueError::HorizonBeforeClock {
                horizon: bound,
                clock: self.t,
            });
        }
        // New emission batch: line-fill folds are only valid within one
        // call's `out` (the caller schedules it after we return).
        if let MemModel::L1Shared(st) = &mut self.mem {
            st.line_folds.clear();
        }
        let mut horizon = horizon;
        while self.unfinished_warps > 0 {
            if let Some(cache) = self.immediate_constant_cache.as_mut() {
                for state in cache.iter_mut() {
                    state.complete_ready(self.t);
                }
            }
            self.enqueue_ready_async_copy_destinations()?;
            if horizon.reached(self.t) {
                if let Some(bound) = horizon.bounded_time() {
                    // A caller may revisit an already parked horizon after a
                    // same-cycle external event. Retire fixed-clock releases
                    // at that timestamp before exposing the boundary; the
                    // ordinary skip path already performs this via
                    // `park_at_horizon`.
                    self.park_at_horizon(bound);
                }
                return Ok(Advance::Idle {
                    wake_at: Some(self.t),
                });
            }
            if let Some(memory_pipeline) = self.memory_issue_pipeline.as_mut() {
                memory_pipeline.prune(self.t);
            }
            if let Some(scalar_pipeline) = self.scalar_execution_pipeline.as_mut() {
                scalar_pipeline.prune(self.t);
            }
            #[cfg(feature = "reference")]
            {
                self.advance_diagnostics.local_cycle_steps = self
                    .advance_diagnostics
                    .local_cycle_steps
                    .checked_add(1)
                    .ok_or(IssueError::CounterOverflow {
                        counter: "reference local cycle steps",
                    })?;
            }
            self.instruction_front_end_cycle()?;
            // Consume the carded per-SMSP issue width in one logical cycle.
            // A dual-issue card permits a second independently ready warp
            // transaction. Do not erase control stalls or dependency latency
            // to synthesize same-warp pairs: no pairing model is implemented.
            // Each slot has its own preflight/commit boundary; this is not
            // rollback of an already committed first slot on second-slot error.
            let issue_width = self
                .params
                .issue_ports
                .max(1)
                .max(u32::from(self.params.dual_issue_supported).saturating_add(1));
            let visibility_events_before = self.pending_l1_visibility_event_count();
            let mut issued_this_cycle = false;
            for _ in 0..issue_width {
                let before = out.len();
                let issued = self.issue_cycle(out)?;
                issued_this_cycle |= issued;
                // A reply-carrying request emitted this cycle tightens the
                // safe horizon (mutation-only store traffic cannot affect
                // this SM and does not).
                if out[before..].iter().any(|r| r.reply != ReplyTo::None) {
                    let lower_bound = checked_cycle_add(
                        "self-response horizon",
                        self.t,
                        self.params.l1_miss_latency,
                    )?;
                    horizon = horizon.tighten(lower_bound);
                }
                // An IMC miss consumes one scheduler opportunity but emits no
                // instruction pick. Do not replay that same miss across an
                // arbitrarily wide carded issue loop.
                let instruction_pick = self.last_issue_picks.iter().any(Option::is_some);
                if !issued || !instruction_pick {
                    break;
                }
            }
            if issued_this_cycle {
                self.t = checked_cycle_add("issue-clock advance", self.t, 1)?;
                // Visibility observations are consumed by the shared event
                // system, not by the issuing SM itself.  Stop after this
                // issue cycle so the observations at `t` are scheduled and
                // processed before this actor can issue at `t + 1`.
                if self.pending_l1_visibility_event_count() > visibility_events_before {
                    return Ok(Advance::Idle {
                        wake_at: Some(self.t),
                    });
                }
                if self
                    .memory_admission_feedback
                    .as_mut()
                    .is_some_and(|state| state.take_yield_requested())
                {
                    return Ok(Advance::Idle {
                        wake_at: Some(self.t),
                    });
                }
                continue;
            }
            // Nothing ready: skip to the next KNOWN warp/resource timer
            // (resident warps only — retired warps left the sub-core lists;
            // warps waiting on in-flight responses have no computable time and
            // are excluded).
            let issue_or_async_time = if self.has_async_copy_protocol {
                match (
                    self.next_known_issue_time::<true>(),
                    self.next_async_copy_destination_source_time(),
                ) {
                    (Some(issue), Some(source)) => Some(issue.min(source)),
                    (Some(issue), None) => Some(issue),
                    (None, Some(source)) => Some(source),
                    (None, None) => None,
                }
            } else {
                self.next_known_issue_time::<false>()
            };
            let front_end_work = self.has_instruction_front_end_work()?;
            let front_end_time = front_end_work
                .then(|| checked_cycle_add("instruction front-end clock advance", self.t, 1))
                .transpose()?;
            let maintenance_time = self
                .instruction_front_end
                .as_ref()
                .and_then(InstructionFrontEndState::next_maintenance_time);
            let front_end_time = match (front_end_time, maintenance_time) {
                (Some(work), Some(maintenance)) => Some(work.min(maintenance)),
                (work, None) => work,
                (None, maintenance) => maintenance,
            };
            let next_t = match (issue_or_async_time, front_end_time) {
                (Some(issue), Some(front_end)) => Some(issue.min(front_end)),
                (Some(issue), None) => Some(issue),
                (None, Some(front_end)) => Some(front_end),
                (None, None) => None,
            };
            match next_t {
                Some(nt) => {
                    let next_cycle = checked_cycle_add("idle-clock advance", self.t, 1)?;
                    let target = nt.max(next_cycle);
                    // Preserve the next stream-buffer release even if warp
                    // eligibility is later; releasing a line enables prefetch
                    // while the core is otherwise stalled.
                    let can_skip = !self.dense_reference;
                    if let Some(bound) = horizon.bounded_time()
                        && target >= bound
                    {
                        // The dense reference still visits locally idle
                        // cycles strictly before an external horizon. It
                        // must stop at the horizon itself so the queued event
                        // is applied before any work at that timestamp.
                        if self.dense_reference && next_cycle < bound {
                            self.t = next_cycle;
                            continue;
                        }
                        #[cfg(feature = "reference")]
                        if can_skip {
                            let skipped = bound - next_cycle;
                            self.advance_diagnostics.idle_jumps = self
                                .advance_diagnostics
                                .idle_jumps
                                .checked_add(u64::from(skipped != 0))
                                .ok_or(IssueError::CounterOverflow {
                                    counter: "reference idle jumps",
                                })?;
                            self.advance_diagnostics.skipped_cycles = self
                                .advance_diagnostics
                                .skipped_cycles
                                .checked_add(skipped)
                                .ok_or(IssueError::CounterOverflow {
                                    counter: "reference skipped cycles",
                                })?;
                        }
                        // Park AT the horizon instead of jumping past it:
                        // the event there may ready a warp before the
                        // next known timer.
                        self.park_at_horizon(bound);
                        return Ok(Advance::Idle {
                            wake_at: Some(bound),
                        });
                    }
                    #[cfg(feature = "reference")]
                    if can_skip {
                        let skipped = target - next_cycle;
                        self.advance_diagnostics.idle_jumps = self
                            .advance_diagnostics
                            .idle_jumps
                            .checked_add(u64::from(skipped != 0))
                            .ok_or(IssueError::CounterOverflow {
                                counter: "reference idle jumps",
                            })?;
                        self.advance_diagnostics.skipped_cycles = self
                            .advance_diagnostics
                            .skipped_cycles
                            .checked_add(skipped)
                            .ok_or(IssueError::CounterOverflow {
                                counter: "reference skipped cycles",
                            })?;
                    }
                    self.t = if can_skip { target } else { next_cycle };
                }
                None if self.outstanding_responses() > 0
                    || self
                        .memory_admission_feedback
                        .as_ref()
                        .is_some_and(|state| {
                            state.has_pending_bounds() || state.has_pending_write_acks()
                        }) =>
                {
                    if let Some(bound) = horizon.bounded_time() {
                        let next_cycle = checked_cycle_add("idle-clock advance", self.t, 1)?;
                        if self.dense_reference && next_cycle < bound {
                            self.t = next_cycle;
                            continue;
                        }
                        #[cfg(feature = "reference")]
                        if !self.dense_reference {
                            let skipped = bound - next_cycle;
                            self.advance_diagnostics.idle_jumps = self
                                .advance_diagnostics
                                .idle_jumps
                                .checked_add(u64::from(skipped != 0))
                                .ok_or(IssueError::CounterOverflow {
                                    counter: "reference idle jumps",
                                })?;
                            self.advance_diagnostics.skipped_cycles = self
                                .advance_diagnostics
                                .skipped_cycles
                                .checked_add(skipped)
                                .ok_or(IssueError::CounterOverflow {
                                    counter: "reference skipped cycles",
                                })?;
                        }
                        self.park_at_horizon(bound);
                    }
                    return Ok(Advance::Idle { wake_at: None });
                }
                None => {
                    if let MemModel::L1Shared(st) = &self.mem {
                        debug_assert!(
                            st.retry.is_empty(),
                            "MSHR-stalled work implies in-flight fills"
                        );
                    }
                    return Err(IssueError::NoProgress {
                        time: self.t,
                        unfinished: self.unfinished_warps,
                    });
                }
            }
        }
        Ok(Advance::Done)
    }
}

impl<'a> SmCore<'a> {
    pub(crate) fn apply_fixed_result_priority(
        &mut self,
        subcore: usize,
        kind: ResultWritebackKind,
        projected: &mut ReadinessDiagnostics,
    ) -> Result<(), IssueError> {
        let (Some(state), ResultWritebackKind::FixedHadd2 { bank }) =
            (self.result_writebacks.as_mut(), kind)
        else {
            return Ok(());
        };
        let fixed_at = checked_cycle_add(
            "HADD2 result writeback",
            self.t,
            state.timing.hadd2_writeback_cycles,
        )?;
        let (warps, diagnostics) = (&mut self.warps, &mut projected.result_writeback);
        for event in &mut state.pending {
            if event.subcore != subcore as u8
                || event.bank != bank
                || event.writeback_at != fixed_at
            {
                continue;
            }
            event.writeback_at = checked_cycle_add(
                "deferred L1TEX writeback",
                event.writeback_at,
                state.timing.variable_deferral_cycles,
            )?;
            event.visible_at = checked_cycle_add(
                "deferred L1TEX visibility",
                event.visible_at,
                state.timing.variable_deferral_cycles,
            )?;
            warps[event.warp].stamp_clear(
                Some(event.barrier),
                event.visible_at,
                DependencyOrigin::L1Tex,
            );
            diagnostics.fixed_over_l1tex_collisions = checked_u64_counter_add(
                "fixed-over-L1TEX collision diagnostic",
                diagnostics.fixed_over_l1tex_collisions,
                1,
            )?;
            diagnostics.l1tex_deferral_cycles = checked_u64_counter_add(
                "L1TEX writeback deferral-cycle diagnostic",
                diagnostics.l1tex_deferral_cycles,
                state.timing.variable_deferral_cycles,
            )?;
        }
        Ok(())
    }
}
