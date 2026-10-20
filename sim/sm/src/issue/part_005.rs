// Lexical implementation section 6/8; included by the parent facade.
impl<'a> SmCore<'a> {fn project_cta_barrier_diagnostics(
        &self,
        picks: &[Option<usize>; SUB_CORES],
        barrier_arrival_times: &[Option<u64>; SUB_CORES],
        now: u64,
        projected: &mut ReadinessDiagnostics,
    ) -> Result<(), IssueError> {
        #[derive(Clone, Copy)]
        struct BlockDelta {
            block: usize,
            retirements: usize,
            arrivals: usize,
            first_new_arrival: Option<u64>,
            last_new_arrival: Option<u64>,
        }
        let mut deltas: [Option<BlockDelta>; SUB_CORES] = [None; SUB_CORES];
        for (subcore, pick) in picks.iter().enumerate() {
            let Some(warp_index) = *pick else { continue };
            let warp = &self.warps[warp_index];
            let slot = warp
                .next_slot
                .ok_or(IssueError::MissingIssueSlot { warp: warp_index })?;
            let reaches_end = warp.next_issue_retires()?;
            // A final barrier is still a CTA participant. Its retirement is
            // deferred until the generation releases, so it must be counted
            // as an arrival rather than an immediate retirement.
            let arrives = slot.op_class == OpClass::BlockBarrier;
            let retires = reaches_end && !arrives;
            if !retires && !arrives {
                continue;
            }
            let entry_index = deltas
                .iter()
                .position(|entry| entry.is_some_and(|delta| delta.block == warp.block))
                .or_else(|| deltas.iter().position(Option::is_none))
                .ok_or(IssueError::CounterOverflow {
                    counter: "CTA barrier diagnostic block delta",
                })?;
            let entry = deltas[entry_index].get_or_insert(BlockDelta {
                block: warp.block,
                retirements: 0,
                arrivals: 0,
                first_new_arrival: None,
                last_new_arrival: None,
            });
            if retires {
                entry.retirements = checked_usize_counter_add(
                    "CTA barrier diagnostic retirement",
                    entry.retirements,
                    1,
                )?;
            }
            if arrives {
                let arrival = barrier_arrival_times[subcore].unwrap_or(now);
                entry.arrivals =
                    checked_usize_counter_add("CTA barrier diagnostic arrival", entry.arrivals, 1)?;
                entry.first_new_arrival = Some(
                    entry
                        .first_new_arrival
                        .map_or(arrival, |current| current.min(arrival)),
                );
                entry.last_new_arrival = Some(
                    entry
                        .last_new_arrival
                        .map_or(arrival, |current| current.max(arrival)),
                );
            }
        }

        for delta in deltas.into_iter().flatten() {
            let state = &self.block_states[delta.block];
            let unfinished = state.unfinished.checked_sub(delta.retirements).ok_or(
                IssueError::CounterUnderflow {
                    counter: "CTA barrier diagnostic unfinished warp",
                },
            )?;
            let waiting = checked_usize_counter_add(
                "CTA barrier diagnostic waiting warp",
                state.waiting,
                delta.arrivals,
            )?;
            if waiting == 0 || waiting != unfinished {
                continue;
            }

            state
                .barrier_generation
                .checked_add(1)
                .ok_or(IssueError::CounterOverflow {
                    counter: "CTA barrier generation",
                })?;
            let first_issue = state.barrier_first_dispatch.unwrap_or(now);
            let mut last_issue = if delta.arrivals == 0 {
                first_issue
            } else {
                now
            };
            let first_arrival = match (state.barrier_first_arrival, delta.first_new_arrival) {
                (Some(existing), Some(new)) => existing.min(new),
                (Some(existing), None) => existing,
                (None, Some(new)) => new,
                (None, None) => now,
            };
            let mut last_arrival = delta.last_new_arrival.unwrap_or(first_arrival);
            let first = state.first_warp;
            let end = checked_usize_counter_add(
                "CTA barrier diagnostic warp range",
                first,
                state.admitted_warps,
            )?;
            for warp in &self.warps[first..end] {
                if !warp.at_block_barrier {
                    continue;
                }
                let dispatch = warp
                    .block_barrier_dispatch
                    .ok_or(IssueError::CounterUnderflow {
                        counter: "CTA barrier dispatch timestamp",
                    })?;
                let arrival = warp
                    .block_barrier_arrival
                    .ok_or(IssueError::CounterUnderflow {
                        counter: "CTA barrier arrival timestamp",
                    })?;
                last_issue = last_issue.max(dispatch);
                last_arrival = last_arrival.max(arrival);
            }
            let release_at = checked_cycle_add("CTA barrier diagnostic release", last_arrival, 1)?;
            let mut rendezvous = 0u64;
            for warp in &self.warps[first..end] {
                if !warp.at_block_barrier {
                    continue;
                }
                let arrival = warp
                    .block_barrier_arrival
                    .ok_or(IssueError::CounterUnderflow {
                        counter: "CTA barrier arrival timestamp",
                    })?;
                rendezvous = checked_u64_counter_add(
                    "CTA barrier rendezvous warp cycle diagnostic",
                    rendezvous,
                    release_at
                        .checked_sub(arrival)
                        .ok_or(IssueError::CounterUnderflow {
                            counter: "CTA barrier rendezvous interval",
                        })?,
                )?;
            }
            for (subcore, pick) in picks.iter().enumerate() {
                let Some(warp) = *pick else { continue };
                if self.warps[warp].block != delta.block
                    || self.warps[warp].next_issue_retires()?
                    || self.warps[warp]
                        .next_slot
                        .is_none_or(|slot| slot.op_class != OpClass::BlockBarrier)
                {
                    continue;
                }
                let arrival = barrier_arrival_times[subcore].unwrap_or(now);
                rendezvous = checked_u64_counter_add(
                    "CTA barrier rendezvous warp cycle diagnostic",
                    rendezvous,
                    release_at
                        .checked_sub(arrival)
                        .ok_or(IssueError::CounterUnderflow {
                            counter: "CTA barrier new-arrival interval",
                        })?,
                )?;
            }
            let spread =
                last_arrival
                    .checked_sub(first_arrival)
                    .ok_or(IssueError::CounterUnderflow {
                        counter: "CTA barrier arrival spread",
                    })?;
            let issue_span =
                last_issue
                    .checked_sub(first_issue)
                    .ok_or(IssueError::CounterUnderflow {
                        counter: "CTA barrier issue span",
                    })?;

            projected.cta_barriers.generations = checked_u64_counter_add(
                "CTA barrier generation diagnostic",
                projected.cta_barriers.generations,
                1,
            )?;
            projected.cta_barriers.total_arrival_spread_cycles = checked_u64_counter_add(
                "CTA barrier total arrival-spread diagnostic",
                projected.cta_barriers.total_arrival_spread_cycles,
                spread,
            )?;
            projected.cta_barriers.max_arrival_spread_cycles =
                projected.cta_barriers.max_arrival_spread_cycles.max(spread);
            projected.cta_barriers.total_rendezvous_warp_cycles = checked_u64_counter_add(
                "CTA barrier total rendezvous diagnostic",
                projected.cta_barriers.total_rendezvous_warp_cycles,
                rendezvous,
            )?;
            if delta.retirements != 0 {
                projected.cta_barriers.releases_after_participant_retirement =
                    checked_u64_counter_add(
                        "CTA barrier participant-retirement diagnostic",
                        projected.cta_barriers.releases_after_participant_retirement,
                        1,
                    )?;
            }
            projected.barrier_pipeline.generations_completed = checked_u64_counter_add(
                "CTA-barrier pipeline completed-generation diagnostic",
                projected.barrier_pipeline.generations_completed,
                1,
            )?;
            if self
                .barrier_issue
                .is_some_and(|issue| issue.generation_rearm_cycles != 0)
            {
                projected.barrier_pipeline.generation_rearm_events = checked_u64_counter_add(
                    "CTA-barrier pipeline generation-rearm diagnostic",
                    projected.barrier_pipeline.generation_rearm_events,
                    1,
                )?;
            }
            projected.barrier_pipeline.first_issue_timestamp_sum = checked_u64_counter_add(
                "CTA-barrier pipeline first-issue timestamp diagnostic",
                projected.barrier_pipeline.first_issue_timestamp_sum,
                first_issue,
            )?;
            projected.barrier_pipeline.last_issue_timestamp_sum = checked_u64_counter_add(
                "CTA-barrier pipeline last-issue timestamp diagnostic",
                projected.barrier_pipeline.last_issue_timestamp_sum,
                last_issue,
            )?;
            projected
                .barrier_pipeline
                .total_generation_issue_span_cycles = checked_u64_counter_add(
                "CTA-barrier pipeline generation issue-span diagnostic",
                projected
                    .barrier_pipeline
                    .total_generation_issue_span_cycles,
                issue_span,
            )?;
            projected.barrier_pipeline.max_generation_issue_span_cycles = projected
                .barrier_pipeline
                .max_generation_issue_span_cycles
                .max(issue_span);
            projected.barrier_pipeline.first_arrival_timestamp_sum = checked_u64_counter_add(
                "CTA-barrier pipeline first-arrival timestamp diagnostic",
                projected.barrier_pipeline.first_arrival_timestamp_sum,
                first_arrival,
            )?;
            projected.barrier_pipeline.last_arrival_timestamp_sum = checked_u64_counter_add(
                "CTA-barrier pipeline last-arrival timestamp diagnostic",
                projected.barrier_pipeline.last_arrival_timestamp_sum,
                last_arrival,
            )?;
            projected
                .barrier_pipeline
                .total_generation_arrival_span_cycles = checked_u64_counter_add(
                "CTA-barrier pipeline generation arrival-span diagnostic",
                projected
                    .barrier_pipeline
                    .total_generation_arrival_span_cycles,
                spread,
            )?;
            projected
                .barrier_pipeline
                .max_generation_arrival_span_cycles = projected
                .barrier_pipeline
                .max_generation_arrival_span_cycles
                .max(spread);
        }
        Ok(())
    }

    fn project_async_issue_overlap(
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

    fn project_picked_readiness(
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

    fn project_memory_service_candidates(
        &self,
        picks: &[Option<usize>; SUB_CORES],
        now: u64,
    ) -> Result<MemoryServiceCandidateState, IssueError> {
        if self.memory_service_candidates.is_empty() {
            return Ok(self.memory_service_candidates.clone());
        }
        let mut projected = self.memory_service_candidates.clone();
        for (subcore, pick) in picks.iter().enumerate() {
            let Some(warp_index) = *pick else { continue };
            let warp = &self.warps[warp_index];
            if let Some((class, _, ideal_wavefronts, unique_lines, _)) =
                projected_global_memory_issue(warp.insts[warp.next].kind, warp.sectors)
            {
                projected.observe(subcore, class, ideal_wavefronts, unique_lines, now)?;
            }
        }
        Ok(projected)
    }

    fn sm86_register_read_plan_for(
        &self,
        subcore: usize,
        warp: usize,
        now: u64,
    ) -> Result<Sm86RegisterReadPlan, IssueError> {
        let state = &self.warps[warp];
        let slot = state
            .next_slot
            .ok_or(IssueError::MissingIssueSlot { warp })?;
        sm86_register_read_plan(
            slot.fixed_latency_operands(),
            state.block,
            state.warp_slot,
            now,
            self.sm86_rf_port_ready[subcore],
            &self.sm86_rfc[subcore],
        )
    }

    fn apply_sm86_register_read_plan(
        &mut self,
        subcore: usize,
        warp: usize,
        plan: &Sm86RegisterReadPlan,
        projected: &mut ReadinessDiagnostics,
    ) -> Result<(), IssueError> {
        if !plan.modeled {
            if plan.unsupported_shape {
                if let Some(diag) = projected.sm86_register_read.as_deref_mut() {
                    diag.unsupported_shapes = diag.unsupported_shapes.checked_add(1).ok_or(
                        IssueError::CounterOverflow {
                            counter: "readiness.sm86_register_read.unsupported_shapes",
                        },
                    )?;
                }
            }
            return Ok(());
        }
        if plan.unsupported_shape {
            if let Some(diag) = projected.sm86_register_read.as_deref_mut() {
                diag.unsupported_shapes =
                    diag.unsupported_shapes
                        .checked_add(1)
                        .ok_or(IssueError::CounterOverflow {
                            counter: "readiness.sm86_register_read.unsupported_shapes",
                        })?;
            }
            return Ok(());
        }
        self.sm86_rf_port_ready[subcore] = plan.next_port_ready;
        for key in &plan.rfc_consumes {
            if let Some(index) = self.sm86_rfc[subcore].iter().position(|entry| entry == key) {
                self.sm86_rfc[subcore].remove(index);
            }
        }
        for key in &plan.rfc_inserts {
            if let Some(index) = self.sm86_rfc[subcore].iter().position(|entry| entry == key) {
                self.sm86_rfc[subcore].remove(index);
            }
            let bank_entries = self.sm86_rfc[subcore]
                .iter()
                .filter(|entry| entry.bank == key.bank)
                .count();
            if bank_entries >= 3 {
                if let Some(index) = self.sm86_rfc[subcore]
                    .iter()
                    .position(|entry| entry.bank == key.bank)
                {
                    self.sm86_rfc[subcore].remove(index);
                }
            }
            self.sm86_rfc[subcore].push_back(*key);
        }
        if let Some(diag) = projected.sm86_register_read.as_deref_mut() {
            diag.reservations =
                diag.reservations
                    .checked_add(1)
                    .ok_or(IssueError::CounterOverflow {
                        counter: "readiness.sm86_register_read.reservations",
                    })?;
            for bank in 0..2 {
                diag.source_reads[bank] = diag.source_reads[bank]
                    .checked_add(plan.reads[bank])
                    .ok_or(IssueError::CounterOverflow {
                        counter: "readiness.sm86_register_read.source_reads",
                    })?;
            }
            diag.port_conflict_cycles = diag
                .port_conflict_cycles
                .checked_add(plan.port_conflict_cycles)
                .ok_or(IssueError::CounterOverflow {
                    counter: "readiness.sm86_register_read.port_conflict_cycles",
                })?;
            diag.rfc_hits =
                diag.rfc_hits
                    .checked_add(plan.rfc_hits)
                    .ok_or(IssueError::CounterOverflow {
                        counter: "readiness.sm86_register_read.rfc_hits",
                    })?;
            diag.rfc_misses = diag.rfc_misses.checked_add(plan.rfc_misses).ok_or(
                IssueError::CounterOverflow {
                    counter: "readiness.sm86_register_read.rfc_misses",
                },
            )?;
        }
        // A retired warp cannot leave stale compiler-RFC entries behind. The
        // key contains block and traced warp slot, so this is only a bounded
        // hygiene pass for a reused physical slot.
        let _ = warp;
        Ok(())
    }

    /// Earliest issue time after both warp/control dependencies and the next
    /// instruction's characterized execution resource are available. A missing
    /// slot remains selectable so the established typed invariant error is
    /// preserved at the transaction boundary.
    fn issue_ready_at_impl<const CHECK_ASYNC: bool>(
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

    fn issue_ready_at<const CHECK_ASYNC: bool>(&self, subcore: usize, warp: usize) -> Option<u64> {
        self.issue_ready_at_impl::<CHECK_ASYNC>(subcore, warp, true)
    }

    fn issue_ready_at_without_immediate_constant<const CHECK_ASYNC: bool>(
        &self,
        subcore: usize,
        warp: usize,
    ) -> Option<u64> {
        self.issue_ready_at_impl::<CHECK_ASYNC>(subcore, warp, false)
    }

    fn is_issue_ready<const CHECK_ASYNC: bool>(
        &self,
        subcore: usize,
        warp: usize,
        now: u64,
    ) -> bool {
        self.issue_ready_at::<CHECK_ASYNC>(subcore, warp)
            .is_some_and(|ready| ready <= now)
    }

    /// True when an SM86 warp is otherwise ready but its current CBANK line
    /// is not yet available in the private immediate-constant cache.
    fn immediate_constant_blocked(&self, subcore: usize, warp: usize, now: u64) -> bool {
        if self.params.arch != ArchId::Sm86 {
            return false;
        }
        let Some(cache) = self.immediate_constant_cache.as_ref() else {
            return false;
        };
        let state = &self.warps[warp];
        let Some(lines) = self.immediate_constant_lines(state.insts[state.next].pc) else {
            return false;
        };
        if lines.is_empty() {
            return false;
        }
        state.imc_wait_until > now || lines.iter().any(|&line| !cache[subcore].contains(line))
    }

    /// Huerta's bounded L0C miss-switch arm: keep the held (greedy) warp
    /// selected only while ordinary control/dependency/resource readiness is
    /// satisfied and the four-cycle carded window remains.
    fn l0c_greedy_hold_candidate<const CHECK_ASYNC: bool>(
        &self,
        subcore: usize,
        warp: usize,
        now: u64,
    ) -> bool {
        self.params.arch == ArchId::Sm86
            && self.immediate_constant_cache.is_some()
            && self.hold[subcore] == Some(warp)
            && self.immediate_constant_switch_remaining[subcore] > 0
            && self.immediate_constant_blocked(subcore, warp, now)
            && self
                .issue_ready_at_without_immediate_constant::<CHECK_ASYNC>(subcore, warp)
                .is_some_and(|ready| ready <= now)
    }

    /// Once the four-cycle window is exhausted, Huerta still probes the
    /// greedy warp first on each issue-stage pass; only the *switch* becomes
    /// legal after its miss. Keeping this marker distinct from the active
    /// window lets `issue_cycle` stage that miss before scanning siblings.
    fn l0c_expired_greedy_probe<const CHECK_ASYNC: bool>(
        &self,
        subcore: usize,
        warp: usize,
        now: u64,
    ) -> bool {
        self.params.arch == ArchId::Sm86
            && self.immediate_constant_cache.is_some()
            && self.hold[subcore] == Some(warp)
            && self.immediate_constant_switch_remaining[subcore] == 0
            && self.immediate_constant_blocked(subcore, warp, now)
            && self
                .issue_ready_at_without_immediate_constant::<CHECK_ASYNC>(subcore, warp)
                .is_some_and(|ready| ready <= now)
    }

    fn scheduler_candidate<const CHECK_ASYNC: bool>(
        &self,
        subcore: usize,
        warp: usize,
        now: u64,
    ) -> bool {
        self.is_issue_ready::<CHECK_ASYNC>(subcore, warp, now)
            || self.l0c_greedy_hold_candidate::<CHECK_ASYNC>(subcore, warp, now)
            || self.l0c_expired_greedy_probe::<CHECK_ASYNC>(subcore, warp, now)
    }

    fn pick_warps<const CHECK_ASYNC: bool>(&self, now: u64) -> [Option<usize>; SUB_CORES] {
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

    /// Pick in stable SMSP order while reserving the transaction's shared
    /// admission capacity. A candidate rejected by an earlier SMSP's claim is
    /// not a dispatch bubble: the same SMSP may still issue its held/youngest
    /// ready non-candidate (or another candidate whose queue has room).
    fn pick_warps_with_memory_admission<const CHECK_ASYNC: bool>(
        &self,
        now: u64,
    ) -> [Option<usize>; SUB_CORES] {
        let Some(feedback) = self.memory_admission_feedback.as_ref() else {
            return self.pick_warps::<CHECK_ASYNC>(now);
        };
        let mut additions = [0usize; SUB_CORES];
        let mut picks = [None; SUB_CORES];
        for (subcore, pick) in picks.iter_mut().enumerate() {
            let permitted = |warp: usize| {
                if !self.scheduler_candidate::<CHECK_ASYNC>(subcore, warp, now) {
                    return false;
                }
                let state = &self.warps[warp];
                feedback.permits_transaction_addition(
                    subcore,
                    state.insts[state.next].kind,
                    &additions,
                    now,
                )
            };
            let selected = match self.hold[subcore] {
                Some(warp) if permitted(warp) => Some(warp),
                _ => self.subcore_warps[subcore]
                    .iter()
                    .copied()
                    .filter(|&warp| {
                        self.is_issue_ready::<CHECK_ASYNC>(subcore, warp, now) && permitted(warp)
                    })
                    .max_by_key(|&warp| (self.warps[warp].arrival, self.warps[warp].warp_id)),
            };
            if let Some(warp) = selected {
                let state = &self.warps[warp];
                feedback.record_transaction_addition(
                    subcore,
                    state.insts[state.next].kind,
                    &mut additions,
                );
            }
            *pick = selected;
        }
        picks
    }

    /// Assign completion timestamps to barrier arrivals dispatched in this
    /// cycle. SMSP dispatch remains parallel; the one SM-shared service queues
    /// those arrivals in stable subcore order at its measured initiation rate.
    fn project_barrier_arrival_times(
        &self,
        picks: &[Option<usize>; SUB_CORES],
        now: u64,
    ) -> Result<[Option<u64>; SUB_CORES], IssueError> {
        let mut server_ready = self.barrier_pipe_ready;
        let mut arrivals = [None; SUB_CORES];
        for (subcore, pick) in picks.iter().enumerate() {
            let Some(warp) = *pick else { continue };
            let slot = self.warps[warp]
                .next_slot
                .ok_or(IssueError::MissingIssueSlot { warp })?;
            let PipelineClass::Barrier {
                warp_initiation_interval_cycles,
                ..
            } = slot.pipeline_class
            else {
                continue;
            };
            let arrival = checked_cycle_add(
                "CTA-barrier arrival service",
                server_ready.max(now),
                warp_initiation_interval_cycles,
            )?;
            arrivals[subcore] = Some(arrival);
            server_ready = arrival;
        }
        Ok(arrivals)
    }

    fn next_known_issue_time<const CHECK_ASYNC: bool>(&self) -> Option<u64> {
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

    /// Earliest timestamp at which a committed exact-destination group becomes
    /// eligible to enter the aggregate service. Groups with outstanding source
    /// responses have no locally known wake time and remain event-driven.
    fn next_async_copy_destination_source_time(&self) -> Option<u64> {
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

    /// Assign committed, source-complete cache-all groups to the shared
    /// destination service. Projection and timestamp arithmetic finish before
    /// either the pipe or a group is mutated.
    fn enqueue_ready_async_copy_destinations(&mut self) -> Result<(), IssueError> {
        if !self.has_async_copy_protocol {
            return Ok(());
        }
        let mut candidates = Vec::new();
        for (warp, slot) in self.async_copies.iter().enumerate() {
            let Some(state) = slot.state.as_deref() else {
                continue;
            };
            for group in &state.committed_groups {
                if group.pending_sources == 0
                    && group.destination_wavefronts != 0
                    && !group.destination_service_queued
                    && group.source_completion_time <= self.t
                {
                    candidates.push((
                        group.source_completion_time,
                        warp,
                        group.id,
                        group.destination_wavefronts,
                    ));
                }
            }
        }
        candidates
            .sort_unstable_by_key(|&(source_ready, warp, group, _)| (source_ready, warp, group.0));

        let mut projected_pipe = self.ldgsts_destination_pipe_ready;
        let mut assignments = Vec::with_capacity(candidates.len());
        for (source_ready, warp, group, wavefronts) in candidates {
            let service_cycles = self.params.ldgsts_destination_service_cycles(wavefronts)?;
            let start = projected_pipe.max(source_ready);
            let ready = checked_cycle_add(
                "LDGSTS aggregate destination-wavefront service",
                start,
                service_cycles,
            )?;
            projected_pipe = ready;
            assignments.push((warp, group, ready));
        }

        for (warp, group_id, ready) in assignments {
            let group = self.async_copies[warp]
                .async_group_mut(group_id)
                .expect("destination-service candidate remains committed");
            debug_assert!(!group.destination_service_queued);
            group.destination_service_queued = true;
            group.destination_completion_time = ready;
            group.completion_time = group.completion_time.max(ready);
        }
        self.ldgsts_destination_pipe_ready = projected_pipe;
        Ok(())
    }

    /// Commit the state transitions that become due while the actor parks at
    /// an external event horizon. Event mode may jump over idle cycles, but
    /// release queues and fixed-clock completions must be retired at the
    /// parked timestamp before typed state is observed.
    fn park_at_horizon(&mut self, bound: u64) {
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

    fn preflight_issue_timing(
        &self,
        kind: AccessKind,
        constant_deep: bool,
        memory_time: Option<u64>,
        write_barrier: Option<DepBarrier>,
        read_barrier: Option<DepBarrier>,
        stall: u8,
        yield_flag: bool,
        pipeline_class: PipelineClass,
        retires_warp: bool,
        observe_async_copy: bool,
    ) -> Result<(), IssueError> {
        let t = self.t;
        checked_cycle_add("issue-clock advance", t, 1)?;
        match pipeline_class {
            PipelineClass::Branch {
                next_issue_interval_cycles,
            } => {
                checked_cycle_add("branch-pipeline resolution", t, next_issue_interval_cycles)?;
            }
            PipelineClass::Barrier {
                warp_initiation_interval_cycles,
                ..
            } => {
                checked_cycle_add(
                    "CTA-barrier pipeline availability",
                    t,
                    warp_initiation_interval_cycles,
                )?;
            }
            PipelineClass::Tensor {
                initiation_interval_cycles,
            } => {
                checked_cycle_add(
                    "tensor-pipeline availability",
                    t,
                    initiation_interval_cycles,
                )?;
            }
            PipelineClass::Sfu {
                initiation_interval_cycles,
            } => {
                checked_cycle_add(
                    "uniform/SFU pipeline availability",
                    t,
                    initiation_interval_cycles,
                )?;
            }
            PipelineClass::ScalarAlu {
                latency_cycles,
                initiation_interval_cycles,
                ..
            } => {
                checked_cycle_add("scalar execution pipeline latency", t, latency_cycles)?;
                checked_cycle_add(
                    "scalar execution pipeline initiation",
                    t,
                    initiation_interval_cycles,
                )?;
            }
            PipelineClass::DispatchOnly => {}
        }
        if !retires_warp {
            checked_cycle_add("warp stall eligibility", t, u64::from(stall.max(1)))?;
            if yield_flag {
                checked_cycle_add("warp yield eligibility", t, 2)?;
            }
        }
        if !matches!(kind, AccessKind::PredicatedOffMemory) && read_barrier.is_some() {
            checked_cycle_add(
                "read-dependence barrier visibility",
                t,
                self.params.read_barrier_latency,
            )?;
        }

        let needs_ready = write_barrier.is_some() || observe_async_copy;
        match kind {
            AccessKind::NonMemory if needs_ready => {
                self.params.non_memory_ready(t)?;
            }
            AccessKind::Shared { wavefronts, .. } if needs_ready => {
                self.params.shared_ready(t, wavefronts)?;
            }
            AccessKind::NonMemory | AccessKind::PredicatedOffMemory | AccessKind::Shared { .. } => {
            }
            AccessKind::GlobalEligible { .. }
            | AccessKind::LocalLoad { .. }
            | AccessKind::GlobalAsyncCopy {
                l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                ..
            } => match &self.mem {
                MemModel::Flat if needs_ready => {
                    checked_cycle_add(
                        "flat global visibility",
                        t,
                        self.params.global_flat_latency,
                    )?;
                }
                MemModel::L1Const(_) if needs_ready => {
                    checked_cycle_add("L1 hit-pipeline floor", t, self.params.l1_hit_latency)?;
                }
                MemModel::L1Shared(_) if needs_ready => {
                    let hit_ready = checked_cycle_add(
                        "shared L1 hit-pipeline floor",
                        memory_time.unwrap_or(t),
                        self.params.l1_hit_latency,
                    )?;
                    checked_constant_return_time(
                        hit_ready,
                        constant_deep && write_barrier.is_some(),
                        &self.params,
                        "constant L1-hit return latency",
                    )?;
                }
                MemModel::Flat | MemModel::L1Const(_) | MemModel::L1Shared(_) => {}
            },
            AccessKind::GlobalBypass { len, .. }
            | AccessKind::GlobalAsyncCopy {
                len,
                l1_policy: LdgstsL1Policy::BypassL1,
                ..
            }
            | AccessKind::GlobalUntagged { len, .. } => match &self.mem {
                MemModel::Flat if needs_ready => {
                    checked_cycle_add(
                        "flat global visibility",
                        t,
                        self.params.global_flat_latency,
                    )?;
                }
                MemModel::L1Const(_) if needs_ready => {
                    self.params.store_visibility_ready(t)?;
                }
                MemModel::L1Shared(_) if len == 0 && needs_ready => {
                    self.params.store_visibility_ready(t)?;
                }
                MemModel::L1Shared(_) if len > 0 && needs_ready => {
                    checked_cycle_add("self-response horizon", t, self.params.l1_miss_latency)?;
                }
                MemModel::Flat | MemModel::L1Const(_) | MemModel::L1Shared(_) => {}
            },
            AccessKind::GlobalStore { .. }
            | AccessKind::GlobalReduction { .. }
            | AccessKind::LocalStore { .. } => match &self.mem {
                MemModel::Flat if needs_ready => {
                    checked_cycle_add(
                        "flat global visibility",
                        t,
                        self.params.global_flat_latency,
                    )?;
                }
                MemModel::L1Const(_) | MemModel::L1Shared(_) if needs_ready => {
                    self.params.store_visibility_ready(t)?;
                }
                MemModel::Flat | MemModel::L1Const(_) | MemModel::L1Shared(_) => {}
            },
        }
        Ok(())
    }

    fn preflight_issue_memory(
        &self,
        picks: &[Option<usize>; SUB_CORES],
        async_destination_wavefronts: &[u64; SUB_CORES],
        memory_times: &[u64; SUB_CORES],
    ) -> Result<(), IssueError> {
        let (l1, shared) = match &self.mem {
            MemModel::Flat => return Ok(()),
            MemModel::L1Const(l1) => (l1, None),
            MemModel::L1Shared(st) => (&st.l1, Some(st)),
        };
        let mut budgets = MemoryOrigin::ALL.map(L1MutationBudget::for_origin);
        let mut eligible_sectors = 0u64;
        let mut pending_budget_fits = true;
        let mut constant_async_deliveries = [None; SUB_CORES];
        for (subcore, pick) in picks.iter().enumerate() {
            let Some(wix) = *pick else { continue };
            let warp = &self.warps[wix];
            let inst = warp
                .insts
                .get(warp.next)
                .ok_or(IssueError::CounterOverflow {
                    counter: "warp instruction cursor",
                })?;
            let slot = warp
                .next_slot
                .ok_or(IssueError::MissingIssueSlot { warp: wix })?;
            let write_barrier = if self.has_async_copy_protocol {
                ordinary_write_barrier(slot)
            } else {
                slot.control.write_barrier
            };
            let add_budget = |counter: &'static str, value: &mut u64, add: u64| {
                *value = checked_u64_counter_add(counter, *value, add)?;
                Ok::<(), IssueError>(())
            };
            match inst.kind {
                AccessKind::GlobalEligible { off, len, .. }
                | AccessKind::LocalLoad { off, len }
                | AccessKind::GlobalAsyncCopy {
                    off,
                    len,
                    l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                    ..
                } => {
                    let origin = match inst.kind {
                        AccessKind::GlobalEligible { .. } | AccessKind::GlobalAsyncCopy { .. } => {
                            MemoryOrigin::Global
                        }
                        AccessKind::LocalLoad { .. } => MemoryOrigin::Local,
                        _ => unreachable!("match arm admits only hierarchy loads"),
                    };
                    let budget = &mut budgets[origin.index()];
                    let sectors = u64::from(len);
                    add_budget(
                        "L1 eligible-sector budget",
                        &mut budget.eligible_sectors,
                        sectors,
                    )?;
                    add_budget(
                        "combined L1 eligible-sector budget",
                        &mut eligible_sectors,
                        sectors,
                    )?;
                    add_budget("L1 recency budget", &mut budget.recency_touches, sectors)?;
                    add_budget("L1 fill budget", &mut budget.new_fills, sectors)?;
                    if shared.is_none()
                        && let Some(completion_work) = async_copy_completion_work(inst.kind)
                    {
                        debug_assert!(matches!(
                            inst.kind,
                            AccessKind::GlobalAsyncCopy {
                                l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                                ..
                            }
                        ));
                        constant_async_deliveries[subcore] =
                            Some((wix, completion_work, async_destination_wavefronts[subcore]));
                    }
                    if shared.is_some()
                        && warp
                            .preflight_pending_add(wix, write_barrier, u32::from(len))
                            .is_err()
                    {
                        pending_budget_fits = false;
                    }
                    if let Some(st) = shared {
                        let end = off.checked_add(usize::from(len)).ok_or(
                            IssueError::CounterOverflow {
                                counter: "issue sector cursor",
                            },
                        )?;
                        for sector_index in off..end {
                            let sector_id = *warp.sectors.get(sector_index).ok_or(
                                IssueError::CounterOverflow {
                                    counter: "issue sector cursor",
                                },
                            )?;
                            preflight_line_fill_mask(st, sector_id)?;
                            let response_id =
                                ResponseId::new(st.response_owner, wix, warp.next, sector_index);
                            if let Some(&reply) = st.outstanding_response_ids.get(&response_id) {
                                return Err(IssueError::ResponseTokenAlreadyOutstanding { reply });
                            }
                        }
                    }
                }
                AccessKind::GlobalBypass {
                    len, token, off, ..
                } => {
                    let budget = &mut budgets[MemoryOrigin::Global.index()];
                    let sectors = u64::from(len);
                    add_budget(
                        "L1 bypass-sector budget",
                        &mut budget.bypass_sectors,
                        sectors,
                    )?;
                    if token == CacheOpToken::Atom {
                        add_budget(
                            "L1 bypass-atom budget",
                            &mut budget.bypass_atom_sectors,
                            sectors,
                        )?;
                    }
                    if let Some(st) = shared
                        && write_barrier.is_some()
                    {
                        if warp
                            .preflight_pending_add(wix, write_barrier, u32::from(len))
                            .is_err()
                        {
                            pending_budget_fits = false;
                        }
                        for sector_index in off..off + usize::from(len) {
                            let response_id =
                                ResponseId::new(st.response_owner, wix, warp.next, sector_index);
                            if let Some(&reply) = st.outstanding_response_ids.get(&response_id) {
                                return Err(IssueError::ResponseTokenAlreadyOutstanding { reply });
                            }
                        }
                    }
                }
                AccessKind::GlobalAsyncCopy {
                    off,
                    len,
                    completion_work,
                    l1_policy: LdgstsL1Policy::BypassL1,
                    ..
                } => {
                    let budget = &mut budgets[MemoryOrigin::Global.index()];
                    let sectors = u64::from(len);
                    add_budget(
                        "L1 bypass-sector budget",
                        &mut budget.bypass_sectors,
                        sectors,
                    )?;
                    if let Some(st) = shared
                        && (write_barrier.is_some() || completion_work != 0)
                    {
                        if warp
                            .preflight_pending_add(wix, write_barrier, u32::from(len))
                            .is_err()
                        {
                            pending_budget_fits = false;
                        }
                        for sector_index in off..off + usize::from(len) {
                            let response_id =
                                ResponseId::new(st.response_owner, wix, warp.next, sector_index);
                            if let Some(&reply) = st.outstanding_response_ids.get(&response_id) {
                                return Err(IssueError::ResponseTokenAlreadyOutstanding { reply });
                            }
                        }
                    }
                }
                AccessKind::GlobalUntagged { off, len } => {
                    let budget = &mut budgets[MemoryOrigin::Global.index()];
                    let sectors = u64::from(len);
                    add_budget(
                        "L1 untagged-sector budget",
                        &mut budget.untagged_sectors,
                        sectors,
                    )?;
                    if let Some(st) = shared
                        && write_barrier.is_some()
                    {
                        if warp
                            .preflight_pending_add(wix, write_barrier, u32::from(len))
                            .is_err()
                        {
                            pending_budget_fits = false;
                        }
                        for sector_index in off..off + usize::from(len) {
                            let response_id =
                                ResponseId::new(st.response_owner, wix, warp.next, sector_index);
                            if let Some(&reply) = st
                                .outstanding_replies
                                .iter()
                                .find(|reply| reply.response_id() == Some(response_id))
                            {
                                return Err(IssueError::ResponseTokenAlreadyOutstanding { reply });
                            }
                        }
                    }
                }
                AccessKind::GlobalStore { len, .. }
                | AccessKind::GlobalReduction { len, .. }
                | AccessKind::LocalStore { len, .. } => {
                    let origin = match inst.kind {
                        AccessKind::GlobalStore { .. } | AccessKind::GlobalReduction { .. } => {
                            MemoryOrigin::Global
                        }
                        AccessKind::LocalStore { .. } => MemoryOrigin::Local,
                        _ => unreachable!("match arm admits only hierarchy writes"),
                    };
                    let budget = &mut budgets[origin.index()];
                    let sectors = u64::from(len);
                    if matches!(inst.kind, AccessKind::GlobalReduction { .. }) {
                        add_budget(
                            "L1 reduction-sector budget",
                            &mut budget.reduction_sectors,
                            sectors,
                        )?;
                    } else {
                        add_budget("L1 store-sector budget", &mut budget.store_sectors, sectors)?;
                    }
                    add_budget("L1 recency budget", &mut budget.recency_touches, sectors)?;
                }
                AccessKind::NonMemory
                | AccessKind::PredicatedOffMemory
                | AccessKind::Shared { .. } => {}
            }
        }

        let constant_ready_upper = if shared.is_none() {
            l1.constant_batch_ready_upper_bound(self.t, eligible_sectors)
        } else {
            None
        };
        let constant_timing_fits = shared.is_some() || constant_ready_upper.is_some();
        let constant_async_completion_fits = shared.is_some()
            || constant_async_deliveries.iter().flatten().all(
                |&(wix, delivery_work, destination_wavefronts)| {
                    constant_ready_upper.is_some_and(|source_ready| {
                        self.async_copies[wix]
                            .preflight_async_copy_add(
                                wix,
                                0,
                                source_ready,
                                delivery_work,
                                destination_wavefronts,
                                None,
                                &self.params,
                            )
                            .is_ok()
                    })
                },
            );
        let shared_timing_fits = shared.is_none()
            || eligible_sectors == 0
            || (self.t.checked_add(self.params.l1_hit_latency).is_some()
                && self.t.checked_add(self.params.l1_miss_latency).is_some());
        if l1.preflight_mutation_budgets(&budgets).is_ok()
            && pending_budget_fits
            && constant_timing_fits
            && constant_async_completion_fits
            && shared_timing_fits
        {
            return Ok(());
        }

        // Near a numeric boundary, replay this at-most-four-instruction batch
        // on a shadow cache. This avoids rejecting valid warm hits while keeping
        // the ordinary hot path allocation-free.
        let mut shadow = l1.clone();
        let mut same_timestamp_line_folds = HashSet::new();
        for (subcore, pick) in picks.iter().enumerate() {
            let Some(wix) = *pick else { continue };
            let warp = &self.warps[wix];
            let memory_time = memory_times[subcore];
            let inst = warp.insts[warp.next];
            let slot = warp
                .next_slot
                .ok_or(IssueError::MissingIssueSlot { warp: wix })?;
            let write_barrier = if self.has_async_copy_protocol {
                ordinary_write_barrier(slot)
            } else {
                slot.control.write_barrier
            };
            match inst.kind {
                AccessKind::GlobalEligible { off, len, .. }
                | AccessKind::LocalLoad { off, len }
                | AccessKind::GlobalAsyncCopy {
                    off,
                    len,
                    l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                    ..
                } => {
                    let request_sectors = &warp.sectors[off..off + usize::from(len)];
                    if matches!(inst.kind, AccessKind::GlobalAsyncCopy { .. }) {
                        shadow.begin_async_copy_lookup_batch_with_sectors(request_sectors);
                    } else {
                        shadow.begin_lookup_batch_for_client_with_sectors(
                            L1LookupClient::Unknown,
                            request_sectors,
                        );
                    }
                    let origin = match inst.kind {
                        AccessKind::GlobalEligible { .. } | AccessKind::GlobalAsyncCopy { .. } => {
                            MemoryOrigin::Global
                        }
                        AccessKind::LocalLoad { .. } => MemoryOrigin::Local,
                        _ => unreachable!("match arm admits only hierarchy loads"),
                    };
                    let mut pending_add = 0u32;
                    let async_delivery = async_copy_completion_work(inst.kind);
                    let observe_ready = write_barrier.is_some() || async_delivery.is_some();
                    let mut async_source_ready = if shared.is_none() && async_delivery.is_some() {
                        Some(checked_cycle_add(
                            "LDGSTS L1 hit-pipeline floor",
                            memory_time,
                            self.params.l1_hit_latency,
                        )?)
                    } else {
                        None
                    };
                    let mshr_floor = matches!(
                        inst.kind,
                        AccessKind::GlobalAsyncCopy {
                            l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                            ..
                        }
                    )
                    .then_some(self.params.ldgsts_l1_mshr_floor);
                    for &sector_id in &warp.sectors[off..off + usize::from(len)] {
                        if shared.is_some() {
                            match access_shared_l1(
                                &mut shadow,
                                origin,
                                sector_id,
                                memory_time,
                                observe_ready,
                                mshr_floor,
                            )? {
                                SharedL1Access::Hit { .. } => {}
                                SharedL1Access::MissRequest { fill } => {
                                    checked_cycle_add(
                                        "self-response horizon",
                                        memory_time,
                                        self.params.l1_miss_latency,
                                    )?;
                                    if let FillKey::Line { line, .. } = fill {
                                        same_timestamp_line_folds.insert((line, memory_time));
                                    }
                                    pending_add = checked_u32_counter_add(
                                        "issue exact pending-add budget",
                                        pending_add,
                                        1,
                                    )?;
                                }
                                SharedL1Access::MergedPending { fill } => {
                                    if let FillKey::Line { line, .. } = fill
                                        && !same_timestamp_line_folds.contains(&(line, memory_time))
                                    {
                                        shadow.reserve_pending_line_sector(
                                            fill,
                                            (sector_id & 3) as u8,
                                        )?;
                                    }
                                    pending_add = checked_u32_counter_add(
                                        "issue exact pending-add budget",
                                        pending_add,
                                        1,
                                    )?;
                                }
                                SharedL1Access::Stalled => {
                                    pending_add = checked_u32_counter_add(
                                        "issue exact pending-add budget",
                                        pending_add,
                                        1,
                                    )?;
                                    break;
                                }
                            }
                        } else if observe_ready {
                            let (_, ready) = match mshr_floor {
                                Some(floor) => shadow.access_eligible_with_mshr_floor(
                                    origin,
                                    sector_id,
                                    memory_time,
                                    floor,
                                )?,
                                None => shadow.access_eligible(origin, sector_id, memory_time)?,
                            };
                            if let Some(current) = &mut async_source_ready {
                                *current = (*current).max(ready);
                            }
                        } else {
                            match mshr_floor {
                                Some(floor) => {
                                    shadow.access_eligible_unobserved_with_mshr_floor(
                                        origin,
                                        sector_id,
                                        memory_time,
                                        floor,
                                    )?;
                                }
                                None => {
                                    shadow.access_eligible_unobserved(
                                        origin,
                                        sector_id,
                                        memory_time,
                                    )?;
                                }
                            }
                        }
                    }
                    if let (Some(source_ready), Some(delivery_work)) =
                        (async_source_ready, async_delivery)
                    {
                        self.async_copies[wix].preflight_async_copy_add(
                            wix,
                            0,
                            source_ready,
                            delivery_work,
                            async_destination_wavefronts[subcore],
                            None,
                            &self.params,
                        )?;
                    }
                    if shared.is_some() {
                        warp.preflight_pending_add(wix, write_barrier, pending_add)?;
                    }
                }
                AccessKind::GlobalBypass { len, token, .. } => {
                    shadow.count_bypass(
                        MemoryOrigin::Global,
                        u64::from(len),
                        token == CacheOpToken::Atom,
                    )?;
                    if shared.is_some() {
                        warp.preflight_pending_add(wix, write_barrier, u32::from(len))?;
                    }
                }
                AccessKind::GlobalAsyncCopy {
                    len,
                    l1_policy: LdgstsL1Policy::BypassL1,
                    ..
                } => {
                    shadow.count_bypass(MemoryOrigin::Global, u64::from(len), false)?;
                    if shared.is_some() {
                        warp.preflight_pending_add(wix, write_barrier, u32::from(len))?;
                    }
                }
                AccessKind::GlobalUntagged { len, .. } => {
                    shadow.count_untagged(MemoryOrigin::Global, u64::from(len))?;
                    if shared.is_some() {
                        warp.preflight_pending_add(wix, write_barrier, u32::from(len))?;
                    }
                }
                AccessKind::GlobalStore { off, len }
                | AccessKind::GlobalReduction { off, len, .. }
                | AccessKind::LocalStore { off, len } => {
                    let origin = match inst.kind {
                        AccessKind::GlobalStore { .. } | AccessKind::GlobalReduction { .. } => {
                            MemoryOrigin::Global
                        }
                        AccessKind::LocalStore { .. } => MemoryOrigin::Local,
                        _ => unreachable!("match arm admits only hierarchy writes"),
                    };
                    let reduction = matches!(inst.kind, AccessKind::GlobalReduction { .. });
                    let store_sectors = &warp.sectors[off..off + usize::from(len)];
                    if self.shared_write_lookup_runtime.observer.is_some()
                        && matches!(inst.kind, AccessKind::GlobalStore { .. })
                    {
                        let hits = shadow.observe_write_lookup_batch(store_sectors)?;
                        shadow.count_write_lookup_hits(origin, hits)?;
                    } else if let Some(hits) =
                        projected_l1_write_lookup_hits(inst.kind, warp.sectors)
                    {
                        shadow.count_write_lookup_hits(origin, hits)?;
                    }
                    for &sector_id in store_sectors {
                        if reduction {
                            shadow.probe_reduction(origin, sector_id, self.t)?;
                        } else {
                            shadow.probe_store(origin, sector_id, self.t)?;
                        }
                    }
                }
                AccessKind::NonMemory
                | AccessKind::PredicatedOffMemory
                | AccessKind::Shared { .. } => {}
            }
        }
        Ok(())
    }

    /// Check the per-SMSP cacheable-LDGSTS hit-return service timeline before
    /// the issue transaction touches the shared L1 or allocates an async-copy
    /// group.  The ordinary budget preflight proves L1 counters and MSHR
    /// capacity, but the service frontier is a separate counter that is only
    /// advanced for observed hits.  Near `u64::MAX`, replay the selected
    /// memory batch on a private L1 shadow so misses and pending merges do not
    /// cause a false rejection (or allow an overflow after an earlier hit has
    /// already mutated the real cache).
    fn preflight_async_l1_hit_service(
        &self,
        picks: &[Option<usize>; SUB_CORES],
        memory_times: &[u64; SUB_CORES],
    ) -> Result<(), IssueError> {
        let Some(service) = self.params.ldgsts_l1_hit_return_service else {
            return Ok(());
        };
        let interval = service.interval_cycles;
        if interval == 0 {
            return Ok(());
        }

        // Stay allocation-free on the normal path.  The exact shadow is only
        // needed when charging every possible observed instruction could overflow
        // the current service frontier.
        let mut needs_shadow = false;
        for (subcore, pick) in picks.iter().enumerate() {
            let Some(wix) = *pick else { continue };
            let warp = &self.warps[wix];
            let slot = warp
                .next_slot
                .ok_or(IssueError::MissingIssueSlot { warp: wix })?;
            let write_barrier = if self.has_async_copy_protocol {
                ordinary_write_barrier(slot)
            } else {
                slot.control.write_barrier
            };
            let AccessKind::GlobalAsyncCopy {
                len,
                completion_work,
                l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                ..
            } = warp.insts[warp.next].kind
            else {
                continue;
            };
            if write_barrier.is_none() && completion_work == 0 {
                continue;
            }
            if len == 0 {
                continue;
            }
            let Some(hit_ready) = memory_times[subcore].checked_add(self.params.l1_hit_latency)
            else {
                // The ordinary timing preflight normally catches this first;
                // keep this shadow decision conservative if that contract is
                // widened later.
                needs_shadow = true;
                continue;
            };
            let frontier = self.shared_write_lookup_runtime.async_l1_hit_service_ready[subcore];
            if frontier.max(hit_ready).checked_add(interval).is_none() {
                needs_shadow = true;
            }
        }
        if !needs_shadow {
            return Ok(());
        }

        let MemModel::L1Shared(st) = &self.mem else {
            return Ok(());
        };
        let mut shadow = st.l1.clone();
        let mut service_ready = self.shared_write_lookup_runtime.async_l1_hit_service_ready;

        for (subcore, pick) in picks.iter().enumerate() {
            let Some(wix) = *pick else { continue };
            let warp = &self.warps[wix];
            let slot = warp
                .next_slot
                .ok_or(IssueError::MissingIssueSlot { warp: wix })?;
            let kind = warp.insts[warp.next].kind;
            let write_barrier = if self.has_async_copy_protocol {
                ordinary_write_barrier(slot)
            } else {
                slot.control.write_barrier
            };
            let now = memory_times[subcore];
            match kind {
                AccessKind::GlobalEligible { off, len, .. }
                | AccessKind::LocalLoad { off, len }
                | AccessKind::GlobalAsyncCopy {
                    off,
                    len,
                    l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                    ..
                } => {
                    let (origin, mshr_floor, cacheable_async, source_active) = match kind {
                        AccessKind::GlobalEligible { .. } => {
                            (MemoryOrigin::Global, None, false, false)
                        }
                        AccessKind::LocalLoad { .. } => (MemoryOrigin::Local, None, false, false),
                        AccessKind::GlobalAsyncCopy {
                            completion_work, ..
                        } => (
                            MemoryOrigin::Global,
                            Some(self.params.ldgsts_l1_mshr_floor),
                            true,
                            completion_work != 0,
                        ),
                        _ => unreachable!("match arm admits only cacheable L1 accesses"),
                    };
                    let observe_ready = write_barrier.is_some() || source_active;
                    let end =
                        off.checked_add(usize::from(len))
                            .ok_or(IssueError::CounterOverflow {
                                counter: "issue sector cursor",
                            })?;
                    let sectors =
                        warp.sectors
                            .get(off..end)
                            .ok_or(IssueError::CounterOverflow {
                                counter: "issue sector cursor",
                            })?;
                    if cacheable_async {
                        shadow.begin_async_copy_lookup_batch_with_sectors(sectors);
                    } else {
                        shadow.begin_lookup_batch_for_client_with_sectors(
                            L1LookupClient::Unknown,
                            sectors,
                        );
                    }
                    let mut instruction_hit_service_charged = false;
                    for &sector in sectors {
                        let access = access_shared_l1(
                            &mut shadow,
                            origin,
                            sector,
                            now,
                            observe_ready,
                            mshr_floor,
                        )?;
                        if cacheable_async
                            && observe_ready
                            && let SharedL1Access::Hit { ready: Some(ready) } = access
                        {
                            // One source-return service token covers the
                            // whole coalesced LDGSTS instruction.
                            if !instruction_hit_service_charged {
                                if service_ready[subcore] < ready {
                                    service_ready[subcore] = ready;
                                }
                                service_ready[subcore] = service_ready[subcore]
                                    .checked_add(interval)
                                    .ok_or(IssueError::CounterOverflow {
                                        counter: "async L1 hit service timeline",
                                    })?;
                                instruction_hit_service_charged = true;
                            }
                        }
                        if matches!(access, SharedL1Access::Stalled) {
                            break;
                        }
                    }
                }
                AccessKind::GlobalStore { off, len }
                | AccessKind::GlobalReduction { off, len, .. }
                | AccessKind::LocalStore { off, len } => {
                    let end =
                        off.checked_add(usize::from(len))
                            .ok_or(IssueError::CounterOverflow {
                                counter: "issue sector cursor",
                            })?;
                    let sectors =
                        warp.sectors
                            .get(off..end)
                            .ok_or(IssueError::CounterOverflow {
                                counter: "issue sector cursor",
                            })?;
                    let origin = if matches!(kind, AccessKind::LocalStore { .. }) {
                        MemoryOrigin::Local
                    } else {
                        MemoryOrigin::Global
                    };
                    let reduction = matches!(kind, AccessKind::GlobalReduction { .. });
                    for &sector in sectors {
                        if reduction {
                            shadow.probe_reduction(origin, sector, now)?;
                        } else {
                            shadow.probe_store(origin, sector, now)?;
                        }
                    }
                }
                AccessKind::GlobalBypass { .. }
                | AccessKind::GlobalUntagged { .. }
                | AccessKind::GlobalAsyncCopy {
                    l1_policy: LdgstsL1Policy::BypassL1,
                    ..
                }
                | AccessKind::NonMemory
                | AccessKind::PredicatedOffMemory
                | AccessKind::Shared { .. } => {}
            }
        }
        Ok(())
    }
}


