//! Cohesive child mechanism extracted from the parent module.

use super::*;

impl<'a> SmCore<'a> {
    /// Install one closed scalar macro into the native model actor. This
    /// reference-only seam is intentionally narrow: no front-end, memory,
    /// pipeline, barrier, or completion state may cross the boundary. The
    /// typed boundary executor independently validates the corresponding
    /// contract; this method only installs the already-derived state delta and
    /// then leaves `advance` to execute the continuation.
    #[cfg(feature = "reference")]
    pub(crate) fn execute_reference_scalar_macro(
        &mut self,
        warp_slot: i32,
        instruction_count: u64,
        initiation_interval_cycles: u64,
        terminal_block_barrier: bool,
        detailed_effects: &[TypedEventTransition],
        terminal_front_end_exit: Option<&InstructionFrontEndBoundaryState>,
    ) -> Result<ReferenceScalarMacroExecution, IssueError> {
        if instruction_count == 0 || initiation_interval_cycles == 0 {
            return Err(IssueError::ReferenceMacroUnsupported {
                reason: "nonzero_count_and_initiation_interval_required",
            });
        }
        let memory_pipeline_open = self.memory_issue_pipeline.as_ref().is_some_and(|pipeline| {
            pipeline
                .queue_release_times
                .iter()
                .any(|queue| !queue.is_empty())
                || pipeline
                    .local_completion_ready
                    .iter()
                    .any(|&ready| ready > self.t)
                || pipeline.shared_accept_ready > self.t
        });
        let scalar_pipeline_open =
            self.scalar_execution_pipeline
                .as_ref()
                .is_some_and(|pipeline| {
                    pipeline.release_times.iter().any(|queue| !queue.is_empty())
                        || pipeline.next_service.iter().any(|&ready| ready > self.t)
                });
        let constant_cache_open = self
            .immediate_constant_cache
            .as_ref()
            .is_some_and(|caches| {
                caches.iter().any(|cache| {
                    !cache.pending.is_empty()
                        || cache.next_service > self.t
                        || cache.sets.iter().any(|set| !set.is_empty())
                })
            });
        if memory_pipeline_open
            || scalar_pipeline_open
            || constant_cache_open
            || !self.unmodeled_tails.is_empty()
            || self.outstanding_responses() != 0
            || self.block_states.iter().any(|block| block.waiting != 0)
            || self.warps.iter().any(|warp| {
                warp.at_block_barrier
                    || warp.barrier_pending.iter().any(|&pending| pending != 0)
                    || warp.next_slot.is_none() && !warp.done
            })
        {
            return Err(IssueError::ReferenceMacroUnsupported {
                reason: "future_relevant_native_state_is_not_closed",
            });
        }
        let warp_index = self
            .warps
            .iter()
            .position(|warp| warp.warp_slot == warp_slot)
            .ok_or(IssueError::ReferenceMacroUnsupported {
                reason: "warp_slot_not_resident",
            })?;
        let (next, inst_len, launch_index) = {
            let warp = &self.warps[warp_index];
            (warp.next, warp.insts.len(), warp.launch_index)
        };
        let end_index = next
            .checked_add(usize::try_from(instruction_count).map_err(|_| {
                IssueError::ReferenceMacroUnsupported {
                    reason: "instruction_count_out_of_range",
                }
            })?)
            .ok_or(IssueError::ReferenceMacroUnsupported {
                reason: "instruction_count_overflow",
            })?;
        if end_index > inst_len {
            return Err(IssueError::ReferenceMacroUnsupported {
                reason: "macro_exceeds_warp_stream",
            });
        }
        let instruction_count_usize = end_index - next;
        // A comparator-selected production witness carries one detailed issue
        // effect per collapsed instruction.  The model-local helper tests
        // intentionally omit effects and use the scalar recurrence, so keep
        // that compatibility path; whenever effects are supplied, however,
        // reject a truncated or non-monotone span before any actor mutation.
        if terminal_block_barrier && detailed_effects.len() != instruction_count_usize
            || (!detailed_effects.is_empty()
                && (detailed_effects.len() != instruction_count_usize
                    || detailed_effects
                        .windows(2)
                        .any(|window| window[1].event.time < window[0].event.time)))
        {
            return Err(IssueError::ReferenceMacroUnsupported {
                reason: "detailed_effect_span_not_closed",
            });
        }
        if self.instruction_front_end.is_some() && !terminal_block_barrier {
            return Err(IssueError::ReferenceMacroUnsupported {
                reason: "front_end_transition_not_implemented",
            });
        }
        let front_end_transition = if terminal_block_barrier && self.instruction_front_end.is_some()
        {
            let boundary =
                terminal_front_end_exit.ok_or(IssueError::ReferenceMacroUnsupported {
                    reason: "terminal_front_end_exit_missing",
                })?;
            Some(InstructionFrontEndState::from_typed_boundary(boundary)?)
        } else {
            None
        };
        for (offset, instruction) in (next..end_index).enumerate() {
            let slot = lookup_dense(
                &self.schedule_slots,
                self.schedule,
                self.warps[warp_index].insts[instruction].pc,
                launch_index,
                warp_slot,
            )?;
            let terminal_barrier_slot =
                terminal_block_barrier && offset + 2 == instruction_count_usize;
            let terminal_exit_slot =
                terminal_block_barrier && offset + 1 == instruction_count_usize;
            let supported_issue_slot = terminal_barrier_slot
                || terminal_exit_slot
                || matches!(slot.pipeline_class, PipelineClass::ScalarAlu { .. });
            let terminal_sequence_slot = terminal_block_barrier
                && matches!(slot.pipeline_class, PipelineClass::DispatchOnly);
            if !supported_issue_slot && !terminal_sequence_slot
                || !matches!(slot.mem_role, MemRole::None)
                || (!terminal_barrier_slot
                    && !terminal_exit_slot
                    && slot.op_class != OpClass::Other)
                || (terminal_barrier_slot && slot.op_class != OpClass::BlockBarrier)
                || (terminal_exit_slot && !slot.opcode_text().starts_with("EXIT"))
            {
                return Err(IssueError::ReferenceMacroUnsupported {
                    reason: "scalar_macro_descriptor_not_closed",
                });
            }
        }
        // Resolve the detailed continuation slot before mutating the actor.
        // A malformed schedule must take exact fallback without leaving a
        // partially advanced warp behind.
        let next_slot = if end_index < inst_len {
            Some(lookup_dense(
                &self.schedule_slots,
                self.schedule,
                self.warps[warp_index].insts[end_index].pc,
                launch_index,
                warp_slot,
            )?)
        } else {
            None
        };
        let subcore = self.warps[warp_index]
            .warp_slot
            .rem_euclid(SUB_CORES as i32) as usize;
        let block = self.warps[warp_index].block;
        let first_issue = detailed_effects
            .first()
            .map(|effect| effect.event.time)
            .unwrap_or_else(|| self.t.max(self.warps[warp_index].eligible));
        let last_issue = detailed_effects
            .last()
            .map(|effect| effect.event.time)
            .or_else(|| {
                instruction_count
                    .checked_sub(1)
                    .and_then(|count| count.checked_mul(initiation_interval_cycles))
                    .and_then(|offset| first_issue.checked_add(offset))
            })
            .ok_or(IssueError::ReferenceMacroUnsupported {
                reason: "macro_timing_overflow",
            })?;
        let end_cycle = last_issue
            .checked_add(1)
            .ok_or(IssueError::ReferenceMacroUnsupported {
                reason: "macro_end_overflow",
            })?;
        self.warps[warp_index].next = end_index;
        self.warps[warp_index].eligible = if terminal_block_barrier {
            last_issue
        } else {
            end_cycle
        };
        self.warps[warp_index].current_since = end_cycle;
        self.warps[warp_index].next_slot = next_slot;
        if let Some(boundary) = terminal_front_end_exit {
            let _ = boundary;
            self.warps[warp_index].front_end_fetch_cursor = end_index;
            self.warps[warp_index].instruction_buffer.clear();
        }
        self.issued =
            self.issued
                .checked_add(instruction_count)
                .ok_or(IssueError::CounterOverflow {
                    counter: "reference macro issued",
                })?;
        self.instructions_per_smsp[subcore] = self.instructions_per_smsp[subcore]
            .checked_add(instruction_count)
            .ok_or(IssueError::CounterOverflow {
                counter: "reference macro SMSP instructions",
            })?;
        self.issue_active_cycles_per_smsp[subcore] = self.issue_active_cycles_per_smsp[subcore]
            .checked_add(instruction_count)
            .ok_or(IssueError::CounterOverflow {
                counter: "reference macro SMSP active cycles",
            })?;
        self.hold[subcore] = Some(warp_index);
        if let Some(front_end) = front_end_transition {
            self.instruction_front_end = Some(front_end);
        }
        if terminal_block_barrier {
            let barrier_issue = detailed_effects
                .get(detailed_effects.len().saturating_sub(2))
                .map(|effect| effect.event.time)
                .ok_or(IssueError::ReferenceMacroUnsupported {
                    reason: "terminal_barrier_effect_missing",
                })?;
            let barrier_arrival =
                barrier_issue
                    .checked_add(2)
                    .ok_or(IssueError::ReferenceMacroUnsupported {
                        reason: "terminal_barrier_arrival_overflow",
                    })?;
            let barrier_release =
                barrier_arrival
                    .checked_add(1)
                    .ok_or(IssueError::ReferenceMacroUnsupported {
                        reason: "terminal_barrier_release_overflow",
                    })?;
            self.block_states[block].waiting = 0;
            self.block_states[block].barrier_generation = self.block_states[block]
                .barrier_generation
                .checked_add(1)
                .ok_or(IssueError::CounterOverflow {
                    counter: "reference macro barrier generation",
                })?;
            self.block_states[block].barrier_rearm_ready = barrier_arrival;
            self.block_states[block].barrier_first_dispatch = None;
            self.block_states[block].barrier_first_arrival = None;
            self.barrier_pipe_ready = barrier_arrival;
            self.local_completion_time = Some(barrier_release);
            self.warps[warp_index].barrier_clear[0] = detailed_effects
                .get(detailed_effects.len().saturating_sub(4))
                .map(|effect| effect.event.time)
                .ok_or(IssueError::ReferenceMacroUnsupported {
                    reason: "terminal_barrier_clear_effect_missing",
                })?;
            self.warps[warp_index].barrier_pipeline_ready = barrier_release;
            self.warps[warp_index].at_block_barrier = true;
            self.warps[warp_index].next_slot = None;
            // Retire after the terminal EXIT issue.  `retire_warp` records
            // activity through `t + 1`; using the barrier release here would
            // shorten the warp lifetime relative to detailed replay even
            // though the barrier's local completion frontier is earlier.
            self.retire_warp(subcore, warp_index, last_issue)?;
            self.t = end_cycle;
        }
        self.t = end_cycle;
        Ok(ReferenceScalarMacroExecution {
            warp_slot,
            first_issue,
            last_issue,
            end_cycle,
            instruction_count,
        })
    }
}

impl<'a> SmCore<'a> {
    pub(crate) fn preflight_retry_timing(
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
}

impl<'a> SmCore<'a> {
    pub(crate) fn has_instruction_front_end_work(&self) -> Result<bool, IssueError> {
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

impl<'a> SmCore<'a> {
    pub(crate) fn project_cta_barrier_diagnostics(
        &self,
        picks: &[Option<usize>; SUB_CORES],
        barrier_arrival_times: &[Option<u64>; SUB_CORES],
        now: u64,
        projected: &mut ReadinessDiagnostics,
    ) -> Result<(), IssueError> {
        #[derive(Clone, Copy)]
        pub(super) struct BlockDelta {
            pub(super) block: usize,
            pub(super) retirements: usize,
            pub(super) arrivals: usize,
            pub(super) first_new_arrival: Option<u64>,
            pub(super) last_new_arrival: Option<u64>,
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
}

impl<'a> SmCore<'a> {
    pub(crate) fn sm86_register_read_plan_for(
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
}

impl<'a> SmCore<'a> {
    pub(crate) fn apply_sm86_register_read_plan(
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
}

impl<'a> SmCore<'a> {
    pub(crate) fn issue_ready_at<const CHECK_ASYNC: bool>(
        &self,
        subcore: usize,
        warp: usize,
    ) -> Option<u64> {
        self.issue_ready_at_impl::<CHECK_ASYNC>(subcore, warp, true)
    }
}

impl<'a> SmCore<'a> {
    pub(crate) fn is_issue_ready<const CHECK_ASYNC: bool>(
        &self,
        subcore: usize,
        warp: usize,
        now: u64,
    ) -> bool {
        self.issue_ready_at::<CHECK_ASYNC>(subcore, warp)
            .is_some_and(|ready| ready <= now)
    }
}

impl<'a> SmCore<'a> {
    /// Huerta's bounded L0C miss-switch arm: keep the held (greedy) warp
    /// selected only while ordinary control/dependency/resource readiness is
    /// satisfied and the four-cycle carded window remains.
    pub(crate) fn l0c_greedy_hold_candidate<const CHECK_ASYNC: bool>(
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
}

impl<'a> SmCore<'a> {
    /// Once the four-cycle window is exhausted, Huerta still probes the
    /// greedy warp first on each issue-stage pass; only the *switch* becomes
    /// legal after its miss. Keeping this marker distinct from the active
    /// window lets `issue_cycle` stage that miss before scanning siblings.
    pub(crate) fn l0c_expired_greedy_probe<const CHECK_ASYNC: bool>(
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
}

impl<'a> SmCore<'a> {
    /// Assign completion timestamps to barrier arrivals dispatched in this
    /// cycle. SMSP dispatch remains parallel; the one SM-shared service queues
    /// those arrivals in stable subcore order at its measured initiation rate.
    pub(crate) fn project_barrier_arrival_times(
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
}

impl<'a> SmCore<'a> {
    /// Assign committed, source-complete cache-all groups to the shared
    /// destination service. Projection and timestamp arithmetic finish before
    /// either the pipe or a group is mutated.
    pub(crate) fn enqueue_ready_async_copy_destinations(&mut self) -> Result<(), IssueError> {
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
}

impl<'a> SmCore<'a> {
    pub(crate) fn preflight_issue_timing(
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
}

impl<'a> SmCore<'a> {
    /// Validate admissions that a successful retirement will trigger without
    /// mutating the actor.  The commit path in [`retire_warp`] repeats these
    /// already-checked operations; keeping this pass side-effect free preserves
    /// the issue-cycle transaction boundary when a checked counter can fail.
    pub(crate) fn preflight_block_admissions(
        &self,
        mut resident_warps: u32,
        mut unfinished_warps: usize,
        mut resident_shared_memory_bytes: u64,
        mut resident_blocks: u32,
        mut resident_warp_slots: HashSet<i32>,
        mut arrival_stamp: u64,
    ) -> Result<(), IssueError> {
        let mut next_block = self.next_block;
        while next_block < self.blocks.len() {
            let block = &self.blocks[next_block];
            let live_warps = self.block_live_warps[next_block];
            let shared_memory_bytes = if live_warps == 0 {
                0
            } else {
                self.shared_memory_residency.bytes_per_cta()
            };
            let next_resident_blocks = resident_blocks
                .checked_add(u32::from(live_warps != 0))
                .ok_or(IssueError::CounterOverflow {
                    counter: "resident blocks",
                })?;
            let Some(next_resident_warps) = resident_warps.checked_add(live_warps) else {
                if resident_warps > 0 {
                    break;
                }
                return Err(IssueError::CounterOverflow {
                    counter: "resident warp",
                });
            };
            let Some(next_resident_shared_memory_bytes) =
                resident_shared_memory_bytes.checked_add(shared_memory_bytes)
            else {
                if resident_warps > 0 {
                    break;
                }
                return Err(IssueError::CounterOverflow {
                    counter: "resident shared-memory byte",
                });
            };
            let pending_slots = block
                .warps
                .iter()
                .filter(|warp| !warp.insts.is_empty())
                .map(|warp| warp.warp_slot)
                .collect::<Vec<_>>();
            let slots_collide = pending_slots
                .iter()
                .any(|slot| resident_warp_slots.contains(slot));
            if resident_warps > 0
                && (next_resident_blocks > self.params.max_resident_blocks_per_sm
                    || next_resident_warps > self.params.max_resident_warps_per_sm
                    || next_resident_shared_memory_bytes
                        > self.shared_memory_residency.max_bytes_per_sm()
                    || slots_collide)
            {
                break;
            }

            // Keep the preflight in lockstep with `admit_blocks`: this
            // checked addition is otherwise the last fallible counter before
            // the first warp of a successor CTA is appended.
            let admitted_warps =
                usize::try_from(live_warps).map_err(|_| IssueError::CounterOverflow {
                    counter: "admitted warp cardinality",
                })?;
            unfinished_warps = unfinished_warps.checked_add(admitted_warps).ok_or(
                IssueError::CounterOverflow {
                    counter: "unfinished warp",
                },
            )?;

            // Construction validates every dynamic PC and physical slot, but
            // repeat the fallible joins here so this preflight remains a true
            // boundary if an in-memory replay is assembled by a future caller.
            for warp in &block.warps {
                let slot =
                    u32::try_from(warp.warp_slot).map_err(|_| IssueError::WarpSlotOutOfRange {
                        launch_index: block.launch_index,
                        warp_slot: warp.warp_slot,
                        arch: self.params.arch,
                        physical_warp_slots: self.physical_warp_slots,
                    })?;
                if slot >= self.physical_warp_slots {
                    return Err(IssueError::WarpSlotOutOfRange {
                        launch_index: block.launch_index,
                        warp_slot: warp.warp_slot,
                        arch: self.params.arch,
                        physical_warp_slots: self.physical_warp_slots,
                    });
                }
                if !warp.insts.is_empty() {
                    lookup_dense(
                        &self.schedule_slots,
                        self.schedule,
                        warp.insts[0].pc,
                        block.launch_index,
                        warp.warp_slot,
                    )?;
                }
            }
            arrival_stamp = checked_u64_counter_add(
                "warp arrival stamp",
                arrival_stamp,
                u64::from(live_warps),
            )?;
            next_block = checked_usize_counter_add("next block", next_block, 1)?;
            resident_blocks = next_resident_blocks;
            resident_warps = next_resident_warps;
            resident_shared_memory_bytes = next_resident_shared_memory_bytes;
            for slot in pending_slots {
                resident_warp_slots.insert(slot);
            }
        }
        Ok(())
    }
}

impl<'a> SmCore<'a> {
    /// One issue cycle at `self.t` across the four sub-cores. Returns
    /// whether anything issued.
    pub(crate) fn issue_cycle(&mut self, out: &mut Vec<MemRequest>) -> Result<bool, IssueError> {
        let t = self.t;
        self.last_issue_picks = [None; SUB_CORES];
        let mut selected = if self.has_async_copy_protocol {
            self.pick_warps_with_memory_admission::<true>(t)
        } else {
            self.pick_warps_with_memory_admission::<false>(t)
        };
        if matches!(self.mem, MemModel::L1Shared(_))
            && let Some(subcore) = selected.iter().position(|pick| {
                pick.is_some_and(|warp| {
                    self.warps[warp]
                        .next_slot
                        .is_some_and(IssueSlot::is_cctl_invalidate_all)
                })
            })
        {
            // An invalidation is one cache-state boundary. Do not allow a
            // sibling picked in this same transaction to allocate a new fill
            // between the readiness check and invalidate_all().
            let warp = selected[subcore];
            selected = [None; SUB_CORES];
            selected[subcore] = warp;
        }
        let l0c_forced_hold: [bool; SUB_CORES] = std::array::from_fn(|subcore| {
            selected[subcore].is_some_and(|warp| {
                if self.has_async_copy_protocol {
                    self.l0c_greedy_hold_candidate::<true>(subcore, warp, t)
                } else {
                    self.l0c_greedy_hold_candidate::<false>(subcore, warp, t)
                }
            })
        });
        let imc_plan = self.preflight_immediate_constant(selected, t)?;
        let mut picks = imc_plan.picks;
        let effective_picks;
        let mut imc_cache = imc_plan.cache;
        let imc_waits = imc_plan.waits;
        let mut extra_imc_waits = imc_plan.extra_waits;
        let mut imc_miss_observed = imc_plan.miss_observed;
        if let Some(cache) = imc_cache.as_mut() {
            for (subcore, forced) in l0c_forced_hold.iter().copied().enumerate() {
                if forced && imc_waits[subcore].is_some() {
                    cache[subcore].diagnostics.greedy_switch_block_cycles = cache[subcore]
                        .diagnostics
                        .greedy_switch_block_cycles
                        .saturating_add(1);
                }
            }
        }

        // Huerta's issue loop continues through the remaining priority list
        // after a non-greedy warp misses in L0C. The previous Rust path
        // stopped after the first parked candidate, which could create a
        // false dispatch bubble when a younger sibling was also CBANK-bound.
        // Scan only when the four-cycle greedy hold did not apply, preserve
        // every line reservation made by parked candidates, and keep the
        // memory-admission candidate as a final transactional constraint.
        if imc_cache.is_some() {
            for subcore in 0..SUB_CORES {
                let Some(initial) = selected[subcore] else {
                    continue;
                };
                let held_active_window = self.hold[subcore] == Some(initial)
                    && self.immediate_constant_switch_remaining[subcore] > 0;
                if picks[subcore].is_some() || held_active_window || imc_waits[subcore].is_none() {
                    continue;
                }
                let Some(cache) = imc_cache.as_mut() else {
                    break;
                };
                let mut tried = vec![initial];
                loop {
                    let candidate = self.subcore_warps[subcore]
                        .iter()
                        .copied()
                        .filter(|warp| {
                            !tried.contains(warp)
                                && if self.has_async_copy_protocol {
                                    self.issue_ready_at_without_immediate_constant::<true>(
                                        subcore, *warp,
                                    )
                                } else {
                                    self.issue_ready_at_without_immediate_constant::<false>(
                                        subcore, *warp,
                                    )
                                }
                                .is_some_and(|ready| ready <= t)
                        })
                        .max_by_key(|warp| (self.warps[*warp].arrival, self.warps[*warp].warp_id));
                    let Some(candidate) = candidate else { break };
                    tried.push(candidate);

                    if let Some(feedback) = self.memory_admission_feedback.as_ref() {
                        let mut additions = [0usize; SUB_CORES];
                        for (other, pick) in picks.iter().enumerate() {
                            if other == subcore {
                                continue;
                            }
                            if let Some(warp) = *pick {
                                feedback.record_transaction_addition(
                                    other,
                                    self.warps[warp].insts[self.warps[warp].next].kind,
                                    &mut additions,
                                );
                            }
                        }
                        let kind = self.warps[candidate].insts[self.warps[candidate].next].kind;
                        if !feedback.permits_transaction_addition(subcore, kind, &additions, t) {
                            continue;
                        }
                    }

                    cache[subcore].diagnostics.non_greedy_scan_attempts = cache[subcore]
                        .diagnostics
                        .non_greedy_scan_attempts
                        .saturating_add(1);
                    let state = &self.warps[candidate];
                    let lines = self
                        .immediate_constant_lines(state.insts[state.next].pc)
                        .unwrap_or(&[]);
                    let wait = if state.imc_wait_until > t {
                        Some(state.imc_wait_until)
                    } else if lines.is_empty() {
                        None
                    } else {
                        cache[subcore].reserve(lines, t, self.params.l1_miss_latency)?
                    };
                    if let Some(ready) = wait.filter(|ready| *ready > t) {
                        cache[subcore].diagnostics.non_greedy_scan_misses = cache[subcore]
                            .diagnostics
                            .non_greedy_scan_misses
                            .saturating_add(1);
                        extra_imc_waits.push((candidate, ready));
                        imc_miss_observed = true;
                        continue;
                    }
                    cache[subcore].diagnostics.non_greedy_scan_successes = cache[subcore]
                        .diagnostics
                        .non_greedy_scan_successes
                        .saturating_add(1);
                    picks[subcore] = Some(candidate);
                    break;
                }
            }
        }
        effective_picks = picks;
        let memory_pipeline_plan =
            if let Some(memory_pipeline) = self.memory_issue_pipeline.as_ref() {
                let mut selected = [None; SUB_CORES];
                for (subcore, pick) in picks.iter().enumerate() {
                    let Some(warp) = *pick else { continue };
                    let state = &self.warps[warp];
                    if matches!(
                        state.insts[state.next].kind,
                        AccessKind::PredicatedOffMemory
                    ) {
                        continue;
                    }
                    selected[subcore] =
                        self.memory_pipeline_slot(u64::from(state.insts[state.next].pc));
                }
                Some(memory_pipeline.plan(selected, t)?)
            } else {
                None
            };
        let scalar_execution_pipeline_plan =
            if let Some(scalar_pipeline) = self.scalar_execution_pipeline.as_ref() {
                let selected = std::array::from_fn(|subcore| {
                    picks[subcore].and_then(|warp| {
                        let slot = self.warps[warp].next_slot?;
                        matches!(slot.pipeline_class, PipelineClass::ScalarAlu { .. })
                            .then_some(slot.pipeline_class)
                    })
                });
                Some(scalar_pipeline.plan(selected, t)?)
            } else {
                None
            };
        let barrier_arrival_times = self.project_barrier_arrival_times(&picks, t)?;
        let mut async_destination_wavefronts = [0; SUB_CORES];
        for (subcore, pick) in picks.iter().enumerate() {
            let Some(wix) = *pick else { continue };
            let kind = self.warps[wix].insts[self.warps[wix].next].kind;
            async_destination_wavefronts[subcore] =
                async_copy_destination_wavefronts(kind, &self.params)?.unwrap_or(0);
        }

        // A cycle is the caller-visible transaction boundary. Validate every
        // selected instruction's possible timing arms before any sub-core
        // mutates counters, cache state, barriers, requests, or retirement.
        for (subcore, pick) in picks.iter().enumerate() {
            let Some(wix) = *pick else { continue };
            let w = &self.warps[wix];
            if self.instruction_front_end.is_some() {
                w.preflight_buffered_issue(wix)?;
            }
            let slot = w
                .next_slot
                .ok_or(IssueError::MissingIssueSlot { warp: wix })?;
            if slot.is_cctl_invalidate_all()
                && let MemModel::L1Shared(state) = &self.mem
            {
                // Dirty-resident mode still requires a real writeback path;
                // preserve its typed error before any issue-state mutation.
                state.l1.preflight_invalidate_all()?;
            }
            let reaches_end = w.next_issue_retires()?;
            // Final BAR.SYNC participates in the CTA rendezvous and retires
            // only once that generation releases.
            let retires_warp = reaches_end && slot.op_class != OpClass::BlockBarrier;
            let kind = w.insts[w.next].kind;
            let constant_deep = is_constant_load_opcode(slot.opcode_text())
                && self.is_constant_path_pc(u64::from(w.insts[w.next].pc));
            validate_async_copy_completion_work(kind, wix)?;
            let write_barrier = if self.has_async_copy_protocol {
                ordinary_write_barrier(slot)
            } else {
                slot.control.write_barrier
            };
            let observe_async_copy =
                self.has_async_copy_protocol && async_copy_source_count(kind).is_some();
            if observe_async_copy && let Some(delivery_work) = async_copy_completion_work(kind) {
                let source_ready = match &self.mem {
                    MemModel::Flat => checked_cycle_add(
                        "flat async-copy source visibility",
                        self.t,
                        self.params.global_flat_latency,
                    )?,
                    MemModel::L1Const(_) => checked_cycle_add(
                        "constant-L1 async-copy source upper bound",
                        self.t,
                        self.params.l1_miss_latency,
                    )?,
                    MemModel::L1Shared(_) => checked_cycle_add(
                        "shared-L1 async-copy known-source floor",
                        self.t,
                        self.params.l1_hit_latency,
                    )?,
                };
                let pending_add = if matches!(self.mem, MemModel::L1Shared(_)) {
                    u32::from(async_copy_source_count(kind).unwrap_or(0))
                } else {
                    0
                };
                self.async_copies[wix].preflight_async_copy_add(
                    wix,
                    pending_add,
                    source_ready,
                    delivery_work,
                    async_destination_wavefronts[subcore],
                    (matches!(self.mem, MemModel::L1Shared(_))
                        && matches!(
                            kind,
                            AccessKind::GlobalAsyncCopy {
                                l1_policy: LdgstsL1Policy::BypassL1,
                                ..
                            }
                        ))
                    .then_some(
                        bypass_async_copy_observed_topology(
                            kind,
                            w.next,
                            w.async_copy_cohorts
                                .and_then(|sidecar| sidecar.view(w.next)),
                        )
                        .unwrap_or(BypassAsyncCopyInput {
                            instruction: w.next,
                            sector_off: match kind {
                                AccessKind::GlobalAsyncCopy { off, .. } => off,
                                _ => 0,
                            },
                            sector_len: async_copy_source_count(kind).unwrap_or(0),
                            source_cohorts: 0,
                            destination_lines: 0,
                            max_bank_wavefronts: 0,
                            max_bypass_wavefronts: 0,
                            cohorts: None,
                            sector_membership_masks: None,
                        }),
                    ),
                    &self.params,
                )?;
            }
            self.preflight_issue_timing(
                kind,
                constant_deep,
                memory_pipeline_plan
                    .as_ref()
                    .and_then(|plan| plan.issues[subcore])
                    .map(|issue| issue.shared_accept_at),
                write_barrier,
                slot.control.read_barrier,
                if self.params.arch == ArchId::Sm120
                    && sm120_alpha_scalar_control_stall_suppressed(slot.opcode_text())
                {
                    0
                } else {
                    slot.control.stall
                },
                slot.control.yield_flag,
                slot.pipeline_class,
                retires_warp,
                observe_async_copy,
            )?;
        }
        let memory_times = std::array::from_fn(|subcore| {
            memory_pipeline_plan
                .as_ref()
                .and_then(|plan| plan.issues[subcore])
                .map_or(t, |issue| issue.shared_accept_at)
        });
        self.preflight_issue_memory(&picks, &async_destination_wavefronts, &memory_times)?;
        self.preflight_async_l1_hit_service(&picks, &memory_times)?;
        self.preflight_result_writeback_priority(&picks)?;
        let next_issued = self.preflight_issue_accounting(&picks, &barrier_arrival_times)?;
        // Diagnostics are preflighted last so an observational counter can
        // never mask an established execution/cache/domain error. No mutation
        // has occurred yet, so a diagnostic overflow remains transactional.
        let mut projected_readiness =
            self.project_picked_readiness(&picks, &barrier_arrival_times, t)?;
        let projected_memory_service_candidates =
            self.project_memory_service_candidates(&picks, t)?;
        if let Some(plan) = scalar_execution_pipeline_plan.as_ref() {
            projected_readiness.scalar_execution_pipeline = Some(Box::new(plan.next.diagnostics));
        }
        let memory_admission_feedback_plan =
            if let Some(feedback) = self.memory_admission_feedback.as_ref() {
                let mut issues = [None; SUB_CORES];
                for (subcore, pick) in picks.iter().enumerate() {
                    let Some(warp) = *pick else { continue };
                    let state = &self.warps[warp];
                    let slot = state
                        .next_slot
                        .ok_or(IssueError::MissingIssueSlot { warp })?;
                    issues[subcore] = Some(MemoryAdmissionFeedbackIssue {
                        subcore,
                        warp,
                        instruction: state.next,
                        kind: state.insts[state.next].kind,
                        prior_clock_reads: state.clock_reads_issued,
                        clock_read: is_clocklo_issue(slot),
                    });
                }
                Some(feedback.plan_issue(&issues, t)?)
            } else {
                None
            };

        if let Some(state) = self.result_writebacks.as_mut() {
            state.pending.retain(|event| event.writeback_at >= t);
        }

        // Reserve the SM86 RF read-port/RFC state only after every selected
        // instruction has passed the existing transaction preflight. One warp
        // per SMSP is selected, so these plans cannot alias another pick's
        // per-SMSP port timeline.
        let mut sm86_rf_plans: [Option<Sm86RegisterReadPlan>; SUB_CORES] = [None, None, None, None];
        if self.sm86_rf_enabled {
            for (subcore, pick) in picks.iter().enumerate() {
                let Some(warp) = *pick else { continue };
                let plan = self.sm86_register_read_plan_for(subcore, warp, t)?;
                sm86_rf_plans[subcore] = Some(plan);
            }
        }

        // Project the source-backed Huerta L0C miss-switch countdown only
        // after all ordinary preflight checks have passed.  A parked greedy
        // warp consumes one bounded opportunity per observed miss; every
        // effective instruction reloads the carded window for the next
        // instruction.  Sibling selection is therefore enabled only after
        // the four miss opportunities have elapsed.
        let mut projected_immediate_constant_switch = self.immediate_constant_switch_remaining;
        if self.params.arch == ArchId::Sm86
            && let Some(config) = self.params.immediate_constant_cache
        {
            for (subcore, selected_warp) in selected.iter().enumerate() {
                if self.hold[subcore] == *selected_warp
                    && imc_waits[subcore].is_some()
                    && projected_immediate_constant_switch[subcore] > 0
                {
                    projected_immediate_constant_switch[subcore] -= 1;
                }
            }
            for (subcore, pick) in picks.iter().enumerate() {
                if pick.is_some() {
                    projected_immediate_constant_switch[subcore] = config.switch_after_miss_cycles;
                }
            }
        }
        // An IMC miss consumes the SMSP issue opportunity even though no
        // traced instruction is dispatched.  Keep it separate from the exact
        // executed-instruction counter below.
        let mut issued_this_cycle = imc_miss_observed;
        for (sc, pick) in picks.into_iter().enumerate() {
            let Some(wix) = pick else { continue };
            issued_this_cycle = true;
            if let Some(plan) = sm86_rf_plans[sc].as_ref() {
                self.apply_sm86_register_read_plan(sc, wix, plan, &mut projected_readiness)?;
            }

            // Issue exactly one instruction from warp `wix`.
            let (
                blk_ix,
                launch_index,
                warp_slot,
                slot_class,
                pipeline_class,
                scalar_control_stall_suppressed,
                control,
                instruction,
                inst,
                warp_sectors,
                retires_warp,
                result_writeback,
                clock_read,
                constant_load,
                constant_deep,
                cctl_invalidate_all,
                fence_semantics,
            ) = {
                let w = &self.warps[wix];
                let slot = w
                    .next_slot
                    .ok_or(IssueError::MissingIssueSlot { warp: wix })?;
                (
                    w.block,
                    w.launch_index,
                    w.warp_slot,
                    slot.op_class,
                    slot.pipeline_class,
                    self.params.arch == ArchId::Sm120
                        && sm120_alpha_scalar_control_stall_suppressed(slot.opcode_text()),
                    slot.control,
                    w.next,
                    w.insts[w.next],
                    w.sectors,
                    {
                        let reaches_end = w.next_issue_retires()?;
                        reaches_end && slot.op_class != OpClass::BlockBarrier
                    },
                    result_writeback_kind(slot, self.params.arch),
                    is_clocklo_issue(slot),
                    is_constant_load_opcode(slot.opcode_text()),
                    self.is_constant_path_pc(u64::from(w.insts[w.next].pc)),
                    slot.is_cctl_invalidate_all(),
                    slot.visibility_fence_semantics(),
                )
            };
            let produces_dynamic_effects = !matches!(inst.kind, AccessKind::PredicatedOffMemory);
            let write_barrier = if !produces_dynamic_effects
                || (self.has_async_copy_protocol && slot_class == OpClass::AsyncCopyCommit)
            {
                None
            } else {
                control.write_barrier
            };
            let result_writeback = produces_dynamic_effects
                .then_some(result_writeback)
                .flatten();
            let observe_async_copy =
                self.has_async_copy_protocol && async_copy_source_count(inst.kind).is_some();
            let fixed_latency_status = (self.params.resource_source()
                == IssueResourceSource::Sm120Alpha)
                .then(|| {
                    self.warps[wix]
                        .next_slot
                        .map(|current| current.fixed_latency_operands())
                })
                .flatten();
            let observe_register_result = inst.effective_lane_mask() != 0
                && matches!(self.mem, MemModel::Flat)
                && fixed_latency_status
                    .and_then(|s| s.projected())
                    .is_some_and(|p| {
                        p.destinations()
                            .iter()
                            .any(|dst| !dst.register().is_constant())
                    });

            // The instruction has passed its IMC probe and is now dispatched;
            // touch each resident line in the cloned cache state.  A line that
            // was just filled is already present after `complete_ready` at the
            // top of this scheduler iteration.
            if let Some(cache) = imc_cache.as_mut()
                && let Some(lines) = self.immediate_constant_lines(inst.pc)
            {
                for &line in lines {
                    cache[sc].touch(line);
                }
            }
            self.warps[wix].imc_wait_until = 0;

            let memory_pipeline_issue = memory_pipeline_plan
                .as_ref()
                .and_then(|plan| plan.issues[sc]);
            let memory_time = memory_pipeline_issue.map_or(t, |issue| issue.shared_accept_at);
            let sm120_flat_service_completion = if self.sm120_flat_service_enabled {
                let opcode_text = self
                    .schedule
                    .issue_slot(u64::from(inst.pc))
                    .map(|slot| slot.opcode_text())
                    .unwrap_or("UNKNOWN");
                sm120_flat_service_work(inst.kind, opcode_text).map(|(global, work)| {
                    let ready = if global {
                        &mut self.sm120_flat_global_service_ready
                    } else {
                        &mut self.sm120_flat_shared_service_ready
                    };
                    let done = (*ready).max(t).saturating_add(work);
                    *ready = done;
                    done
                })
            } else if self.sm120_flat_shared_medium {
                let wavefronts = match inst.kind {
                    AccessKind::Shared { wavefronts, .. } => wavefronts,
                    _ => 0,
                };
                if wavefronts == 0 {
                    None
                } else {
                    let work = wavefronts
                        .saturating_mul(SM120_FLAT_SHARED_MEDIUM_EXTRA_CYCLES_PER_WAVEFRONT);
                    let done = self
                        .sm120_flat_shared_service_ready
                        .max(t)
                        .saturating_add(work);
                    self.sm120_flat_shared_service_ready = done;
                    Some(done)
                }
            } else if self.sm120_flat_multiwarp_service_enabled {
                let len = match inst.kind {
                    AccessKind::GlobalEligible { len, .. }
                    | AccessKind::GlobalBypass { len, .. }
                    | AccessKind::GlobalUntagged { len, .. }
                    | AccessKind::GlobalStore { len, .. }
                    | AccessKind::GlobalReduction { len, .. } => len,
                    _ => 0,
                };
                if len == 0 {
                    None
                } else {
                    let work = u64::from(len)
                        .saturating_mul(SM120_FLAT_MULTI_WARP_GLOBAL_SERVICE_CYCLES_PER_SECTOR);
                    let done = self
                        .sm120_flat_multiwarp_global_service_ready
                        .max(t)
                        .saturating_add(work)
                        .saturating_add(if self.sm120_flat_multiwarp_dense {
                            SM120_FLAT_MULTI_WARP_DENSE_EXTRA_CYCLES
                        } else if self.sm120_flat_multiwarp_medium {
                            SM120_FLAT_MULTI_WARP_MEDIUM_EXTRA_CYCLES
                        } else {
                            0
                        });
                    self.sm120_flat_multiwarp_global_service_ready = done;
                    Some(done)
                }
            } else {
                None
            };
            let sm120_flat_store_retire_completion = if self.sm120_flat_store_retire_enabled {
                let (len, per_sector) = match inst.kind {
                    AccessKind::GlobalStore { len, .. } => {
                        (u64::from(len), SM120_FLAT_STORE_RETIRE_CYCLES_PER_SECTOR)
                    }
                    AccessKind::GlobalReduction { len, .. } => (
                        u64::from(len),
                        SM120_FLAT_REDUCTION_RETIRE_CYCLES_PER_SECTOR,
                    ),
                    _ => (0, 0),
                };
                (len != 0).then(|| {
                    let done = self
                        .sm120_flat_store_retire_ready
                        .max(t)
                        .saturating_add(len.saturating_mul(per_sector));
                    self.sm120_flat_store_retire_ready = done;
                    done
                })
            } else {
                None
            };
            // Memory hierarchy mutation happens when the local address/operand
            // stages reach the SM-shared acceptor, not when the scheduler first
            // dispatches the instruction.
            let mut data_ready = if matches!(self.mem, MemModel::L1Shared(_)) {
                self.mem_step_shared(
                    sc,
                    wix,
                    instruction,
                    inst.kind,
                    constant_load,
                    constant_deep,
                    match inst.kind {
                        AccessKind::GlobalEligible { .. }
                            if opcode_has_qualifier(
                                self.schedule
                                    .issue_slot(u64::from(inst.pc))
                                    .map(|slot| slot.opcode_text())
                                    .unwrap_or(""),
                                "LTC128B",
                            ) =>
                        {
                            L2ClientScope::Ltc128b
                        }
                        AccessKind::GlobalEligible { .. } if constant_load => {
                            L2ClientScope::ConstantLdg
                        }
                        AccessKind::GlobalEligible { .. } => L2ClientScope::OrdinaryLdg,
                        AccessKind::LocalLoad { .. } => L2ClientScope::Local,
                        AccessKind::GlobalAsyncCopy {
                            l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                            ..
                        } => L2ClientScope::AsyncLdgstsCache,
                        AccessKind::GlobalAsyncCopy { .. } => L2ClientScope::AsyncLdgstsBypass,
                        AccessKind::GlobalStore { .. } | AccessKind::LocalStore { .. } => {
                            L2ClientScope::Store
                        }
                        AccessKind::GlobalReduction { .. } => L2ClientScope::Reduction,
                        AccessKind::GlobalBypass {
                            token: CacheOpToken::Atom,
                            ..
                        } => L2ClientScope::Atomic,
                        _ => L2ClientScope::Unknown,
                    },
                    self.schedule
                        .issue_slot(u64::from(inst.pc))
                        .is_some_and(|slot| {
                            matches!(inst.kind, AccessKind::GlobalUntagged { .. })
                                && opcode_head(slot.opcode_text()) == "ST"
                        }),
                    self.schedule
                        .issue_slot(u64::from(inst.pc))
                        .is_some_and(|slot| {
                            matches!(inst.kind, AccessKind::GlobalUntagged { .. })
                                && opcode_head(slot.opcode_text()) == "ST"
                                && self
                                    .block_states
                                    .get(self.warps[wix].block)
                                    .is_some_and(|state| state.admitted_warps <= 8)
                        }),
                    warp_sectors,
                    write_barrier,
                    async_destination_wavefronts[sc],
                    memory_time,
                    out,
                )?
            } else {
                data_ready_time(
                    inst.kind,
                    warp_sectors,
                    memory_time,
                    &self.params,
                    self.mem.const_l1_mut(),
                    write_barrier,
                    observe_async_copy || observe_register_result,
                )?
            };
            if self.params.resource_source() == IssueResourceSource::Sm120Alpha
                && matches!(self.mem, MemModel::Flat)
                && self.sm120_flat_async_dense
                && let AccessKind::GlobalAsyncCopy {
                    off,
                    len: len @ 1..,
                    l1_policy,
                    ..
                } = inst.kind
            {
                // Retain a bounded recent-source working set for non-bulk
                // cp.async. Repeated tile copies cannot all be cold DRAM
                // accesses. Existing fills carry their own ready timestamp.
                // 128 sectors is a logical 4-KiB window, not inferred L1 size.
                let mut source_ready = t;
                for &sector in &warp_sectors[off..off + usize::from(len)] {
                    let ready = if l1_policy == LdgstsL1Policy::CacheAtAllLevels {
                        if let Some(position) = self
                            .sm120_async_sources
                            .iter()
                            .position(|&(key, _)| key == sector)
                        {
                            let entry = self
                                .sm120_async_sources
                                .remove(position)
                                .expect("located source");
                            self.sm120_async_sources.push_back(entry);
                            entry.1.max(checked_cycle_add(
                                "warm async source",
                                t,
                                self.params.l1_hit_latency,
                            )?)
                        } else {
                            let latency = if self.sm120_flat_async_dense {
                                // Fresh group-depth probes show dense repeated
                                // cp.async source traffic is not a cold-DRAM
                                // ladder: NCU reports only 104/128/168 DRAM
                                // sectors for 64/128/256 groups. Use the
                                // logical L2 arm for this shape-conditioned
                                // source frontier; it is not an L2 geometry
                                // claim.
                                self.params.l1_miss_latency
                            } else {
                                self.params.global_flat_latency
                            };
                            let ready = checked_cycle_add("cold async source", t, latency)?;
                            self.sm120_async_sources.push_back((sector, ready));
                            if self.sm120_async_sources.len() > 128 {
                                self.sm120_async_sources.pop_front();
                            }
                            ready
                        }
                    } else {
                        checked_cycle_add(
                            "bypass async source",
                            t,
                            self.params.global_flat_latency,
                        )?
                    };
                    source_ready = source_ready.max(ready);
                }
                data_ready = Some(source_ready);
            }
            if let Some(service_done) = sm120_flat_service_completion
                && (write_barrier.is_some()
                    || control.read_barrier.is_some()
                    || observe_async_copy
                    || observe_register_result)
            {
                data_ready = Some(data_ready.map_or(service_done, |ready| ready.max(service_done)));
            }
            if let Some(service_done) = sm120_flat_store_retire_completion {
                data_ready = Some(data_ready.map_or(service_done, |ready| ready.max(service_done)));
            }
            // Register results and scoreboard counters must share the same
            // producer completion. A load is not ready at dispatch+1 simply
            // because its consumer lacks a compiler barrier annotation.
            if observe_register_result {
                let completion = match pipeline_class {
                    PipelineClass::ScalarAlu { latency_cycles, .. } => {
                        let base =
                            checked_cycle_add("scalar register completion", t, latency_cycles)?;
                        scalar_execution_pipeline_plan
                            .as_ref()
                            .and_then(|plan| plan.next.release_times[sc].back().copied())
                            .map_or(base, |ready| base.max(ready))
                    }
                    _ => data_ready.unwrap_or(self.params.non_memory_ready(t)?),
                };
                if let Some(status) = fixed_latency_status {
                    self.warps[wix]
                        .register_ready
                        .retain(|entry| entry.ready_at > t);
                    issue_record_fixed_register_writes(
                        &mut self.warps[wix].register_ready,
                        status,
                        inst.effective_lane_mask(),
                        completion,
                    );
                    self.local_completion_time =
                        Some(self.local_completion_time.unwrap_or(0).max(completion));
                }
            }
            if cctl_invalidate_all && matches!(self.mem, MemModel::L1Shared(_)) {
                self.invalidate_l1_all()?;
                if let MemModel::L1Shared(st) = &mut self.mem
                    && let Some(requester) = st.l1_visibility_requester
                {
                    let sequence = st.next_l1_visibility_sequence()?;
                    st.l1_invalidate_all_observations
                        .push(sim_mem::L1InvalidateAllObservation {
                            sequence,
                            requester,
                            issue_cycle: t,
                        });
                }
            }
            if let Some((scope, order)) = fence_semantics
                && let MemModel::L1Shared(st) = &mut self.mem
                && let Some(requester) = st.l1_visibility_requester
            {
                let sequence = st.next_l1_visibility_sequence()?;
                st.l1_fence_observations.push(sim_mem::L1FenceObservation {
                    sequence,
                    requester,
                    pc: inst.pc,
                    issue_cycle: t,
                    scope,
                    order,
                });
            }
            if observe_async_copy && let Some(source_ready) = data_ready {
                // Shared mode already attached the copy's delivery and exact
                // destination work to its pending group before walking source
                // sectors. This second update contributes only the known hit
                // floor; synchronous modes create the group here instead.
                let (delivery_work, destination_wavefronts) =
                    if matches!(self.mem, MemModel::L1Shared(_)) {
                        (0, 0)
                    } else {
                        (
                            if self.sm120_flat_async_dense {
                                0
                            } else {
                                async_copy_completion_work(inst.kind).unwrap_or(0)
                            },
                            async_destination_wavefronts[sc],
                        )
                    };
                self.async_copies[wix].add_known_async_copy(
                    wix,
                    source_ready,
                    delivery_work,
                    destination_wavefronts,
                    &self.params,
                )?;
            }

            match pipeline_class {
                PipelineClass::Branch { .. } => {}
                PipelineClass::Barrier { .. } => {
                    if let Some(arrival) = barrier_arrival_times[sc] {
                        self.barrier_pipe_ready = arrival;
                    }
                }
                PipelineClass::Tensor {
                    initiation_interval_cycles,
                } => {
                    self.tensor_pipe_ready[sc] = checked_cycle_add(
                        "tensor-pipeline availability",
                        t,
                        initiation_interval_cycles,
                    )?;
                }
                PipelineClass::Sfu {
                    initiation_interval_cycles,
                } => {
                    self.sfu_pipe_ready[sc] = checked_cycle_add(
                        "uniform/SFU pipeline availability",
                        t,
                        initiation_interval_cycles,
                    )?;
                    if self.params.arch == ArchId::Sm86
                        && self.warps[wix]
                            .next_slot
                            .is_some_and(IssueSlot::is_sfu_dependent_chain)
                    {
                        self.warps[wix].sfu_dependency_ready = checked_cycle_add(
                            "SM86 dependent SFU result readiness",
                            t,
                            SM86_DEPENDENT_SFU_LATENCY_CYCLES,
                        )?;
                    }
                }
                PipelineClass::ScalarAlu { .. } => {}
                PipelineClass::DispatchOnly => {}
            }
            // Yield semantics per the CGGTY paper: the yield bit means
            // "do not issue this warp in the NEXT cycle" — a one-cycle
            // bubble when no other warp is ready — on top of the stall
            // count. A retiring instruction has no next eligibility to observe.
            if !retires_warp {
                // The retained SM120 high-word control decoder is a
                // transport hypothesis. Dependent-chain calibration
                // shows that applying its scalar stall field on top of the
                // measured family latency double-counts readiness. Keep the
                // field for non-scalar classes, but let the calibrated
                // ScalarAlu pipeline own scalar readiness on SM120.
                let control_stall = if scalar_control_stall_suppressed {
                    0
                } else {
                    control.stall
                };
                let stall = u64::from(control_stall.max(1));
                let mut eligible = checked_cycle_add("warp stall eligibility", t, stall)?;
                if control.yield_flag {
                    eligible = eligible.max(checked_cycle_add("warp yield eligibility", t, 2)?);
                }
                if let Some(service_done) = sm120_flat_store_retire_completion {
                    eligible = eligible.max(service_done);
                }
                if (self.sm120_flat_multiwarp_medium || self.sm120_flat_shared_medium)
                    && let Some(service_done) = sm120_flat_service_completion
                {
                    // Dense multi-warp DRAM calibration exposes admission
                    // backpressure, not merely a terminal drain. Park the
                    // issuing warp at the logical service completion so the
                    // existing scheduler can overlap other warps naturally.
                    eligible = eligible.max(service_done);
                }
                self.warps[wix].eligible = eligible;
            }
            // Dependence barriers are COUNTERS in hardware (producer
            // increments at issue, decrements at write-back; the
            // consumer waits for zero), so with several in-flight
            // producers on one index the barrier clears at the LATEST
            // write-back: accumulate with max, never overwrite — a
            // later fast producer (L1 hit) must not shorten a pending
            // slow one (miss). Unknown write-backs (in-flight responses)
            // are tracked in `barrier_pending`.
            if let Some(dr) = data_ready {
                self.warps[wix].stamp_clear(write_barrier, dr, dependency_origin(inst.kind));
            }
            if let Some(result_writeback) = result_writeback {
                self.register_l1tex_writeback(
                    sc,
                    wix,
                    result_writeback,
                    write_barrier,
                    data_ready,
                )?;
                self.apply_fixed_result_priority(sc, result_writeback, &mut projected_readiness)?;
            }
            if produces_dynamic_effects && let Some(read_barrier) = control.read_barrier {
                let read_clear = memory_pipeline_issue.map_or_else(
                    || {
                        checked_cycle_add(
                            "read-dependence barrier visibility",
                            t,
                            self.params.read_barrier_latency,
                        )
                    },
                    |issue| Ok(issue.operand_ready_at),
                )?;
                let read_clear = sm120_flat_service_completion
                    .map_or(read_clear, |service_done| read_clear.max(service_done));
                self.warps[wix].stamp_clear(
                    Some(read_barrier),
                    read_clear,
                    operand_read_dependency_origin(inst.kind),
                );
            }
            if clock_read {
                let warp = &mut self.warps[wix];
                warp.clock_first_issue.get_or_insert(t);
                warp.clock_reads_issued = warp
                    .clock_reads_issued
                    .checked_add(1)
                    .expect("CLOCKLO issue counter was preflighted");
            }
            if self.has_async_copy_protocol {
                match slot_class {
                    OpClass::AsyncCopyCommit => {
                        let expected_nonempty = self.async_copies[wix]
                            .state
                            .as_deref()
                            .and_then(|state| state.open_group)
                            .is_some();
                        let committed = self.async_copies[wix].commit_async_copy_group(t);
                        debug_assert_eq!(
                            committed, expected_nonempty,
                            "async commit mutation disagrees with its pre-issue state"
                        );
                        if committed && self.sm120_flat_async_service_enabled {
                            let done = self
                                .sm120_flat_async_service_ready
                                .max(t)
                                .saturating_add(SM120_FLAT_ASYNC_GROUP_SERVICE_CYCLES);
                            self.sm120_flat_async_service_ready = done;
                            if let Some(state) = self.async_copies[wix].state.as_deref_mut()
                                && let Some(group) = state.committed_groups.back_mut()
                            {
                                group.destination_completion_time =
                                    group.destination_completion_time.max(done);
                                group.completion_time = group.completion_time.max(done);
                            }
                        }
                        if committed && self.sm120_flat_async_depth_service_per_group != 0 {
                            let done = self
                                .sm120_flat_async_depth_service_ready
                                .max(t)
                                .saturating_add(self.sm120_flat_async_depth_service_per_group);
                            self.sm120_flat_async_depth_service_ready = done;
                            if let Some(state) = self.async_copies[wix].state.as_deref_mut()
                                && let Some(group) = state.committed_groups.back_mut()
                            {
                                group.destination_completion_time =
                                    group.destination_completion_time.max(done);
                                group.completion_time = group.completion_time.max(done);
                            }
                        }
                    }
                    OpClass::AsyncCopyWait => {
                        let max_pending_groups = self.warps[wix]
                            .next_slot
                            .ok_or(IssueError::MissingIssueSlot { warp: wix })?
                            .async_copy_wait_max_pending_groups;
                        self.async_copies[wix].consume_async_wait(max_pending_groups, t);
                    }
                    // `Other` includes explicit SB1..SB5 DEPBAR operand sets.
                    // Readiness was enforced through ordinary barrier state;
                    // they never mutate asynchronous-copy groups.
                    OpClass::BlockBarrier | OpClass::Other | OpClass::MBarrierPhaseQuery => {}
                }
            }

            // Advance the warp's stream.
            if self.instruction_front_end.is_some() {
                self.warps[wix].consume_buffered_issue(wix)?;
            }
            self.warps[wix].next =
                checked_usize_counter_add("warp instruction cursor", self.warps[wix].next, 1)?;
            #[cfg(feature = "reference")]
            self.reference_issue_effects.push((
                self.t,
                u64::try_from(wix).unwrap_or(u64::MAX),
                u32::try_from(instruction).unwrap_or(u32::MAX),
            ));
            let final_block_barrier = slot_class == OpClass::BlockBarrier
                && self.warps[wix].next == self.warps[wix].insts.len();
            if retires_warp {
                self.retire_warp(sc, wix, t)?;
            } else if final_block_barrier {
                // Keep a final barrier warp unfinished and parked until the
                // CTA rendezvous. There is intentionally no next issue slot;
                // the release path retires it after all participants arrive.
                self.warps[wix].next_slot = None;
                self.warps[wix].ordinary_dependency_wait_mask = WaitMask::NONE;
                self.warps[wix].eligible = t;
            } else {
                let next_pc = self.warps[wix].insts[self.warps[wix].next].pc;
                let next_slot = lookup_dense(
                    &self.schedule_slots,
                    self.schedule,
                    next_pc,
                    launch_index,
                    warp_slot,
                )?;
                self.warps[wix].next_slot = Some(next_slot);
                self.warps[wix].ordinary_dependency_wait_mask =
                    next_slot.ordinary_dependency_wait_mask();
                self.warps[wix].current_since = t;
                self.warps[wix].imc_wait_until = 0;
                self.warps[wix].cta_barrier_ready = 0;
                self.warps[wix].barrier_pipeline_ready = 0;
                self.warps[wix].branch_pipeline_ready = 0;
                if matches!(pipeline_class, PipelineClass::Branch { .. })
                    && next_pc != inst.pc.saturating_add(16)
                    && let Some(front_end) = self.instruction_front_end.as_mut()
                {
                    front_end.diagnostics.control_flow_redirects = checked_u64_counter_add(
                        "instruction control-flow redirect observer",
                        front_end.diagnostics.control_flow_redirects,
                        1,
                    )?;
                }
            }
            if !retires_warp
                && let PipelineClass::Branch {
                    next_issue_interval_cycles,
                } = pipeline_class
            {
                self.warps[wix].branch_pipeline_ready =
                    checked_cycle_add("branch-pipeline resolution", t, next_issue_interval_cycles)?;
            }
            // Single hold write: never hold a yielded or finished warp.
            self.hold[sc] = (!control.yield_flag && !self.warps[wix].done).then_some(wix);
            self.fetch_last_issued[sc] = (!self.warps[wix].done).then_some(wix);

            // Block barrier arrival AFTER stream bookkeeping: a finished
            // warp never waits.
            if slot_class == OpClass::BlockBarrier && !self.warps[wix].done {
                let arrival = barrier_arrival_times[sc].unwrap_or(t);
                self.warps[wix].at_block_barrier = true;
                self.warps[wix].block_barrier_dispatch = Some(t);
                self.warps[wix].block_barrier_arrival = Some(arrival);
                self.block_states[blk_ix]
                    .barrier_first_dispatch
                    .get_or_insert(t);
                self.block_states[blk_ix]
                    .barrier_first_arrival
                    .get_or_insert(arrival);
                self.block_states[blk_ix].waiting = checked_usize_counter_add(
                    "block barrier waiter",
                    self.block_states[blk_ix].waiting,
                    1,
                )?;
            }
            // Release check: every unfinished warp of the block waiting.
            // The block's warps are contiguous in `warps` (admission
            // pushes them in one run), so scan only that range.
            let bs = &self.block_states[blk_ix];
            if bs.waiting > 0 && bs.waiting == bs.unfinished {
                let (first, count) = (bs.first_warp, bs.admitted_warps);
                let mut latest_arrival = 0u64;
                let mut second_latest_arrival = 0u64;
                let mut latest_count = 0usize;
                for warp in &self.warps[first..first + count] {
                    if !warp.at_block_barrier {
                        continue;
                    }
                    let arrival = warp
                        .block_barrier_arrival
                        .expect("barrier waiter owns an arrival timestamp");
                    if arrival > latest_arrival {
                        second_latest_arrival = latest_arrival;
                        latest_arrival = arrival;
                        latest_count = 1;
                    } else if arrival == latest_arrival {
                        latest_count += 1;
                    } else if arrival > second_latest_arrival {
                        second_latest_arrival = arrival;
                    }
                }
                self.block_states[blk_ix].waiting = 0;
                self.block_states[blk_ix].barrier_generation = self.block_states[blk_ix]
                    .barrier_generation
                    .checked_add(1)
                    .expect("barrier generation was preflighted");
                self.block_states[blk_ix].barrier_first_dispatch = None;
                self.block_states[blk_ix].barrier_first_arrival = None;
                if let Some(issue) = self.barrier_issue {
                    self.block_states[blk_ix].barrier_rearm_ready = checked_cycle_add(
                        "CTA-barrier generation re-arm",
                        latest_arrival,
                        issue.generation_rearm_cycles,
                    )?;
                }
                let release_at = checked_cycle_add("block-barrier release", latest_arrival, 1)?;
                self.local_completion_time = Some(
                    self.local_completion_time
                        .map_or(release_at, |current| current.max(release_at)),
                );
                // Final BAR.SYNC instructions advanced their warps to the end
                // of stream but deliberately left them unfinished above.
                // Collect those participants before clearing the barrier
                // flags, then retire them through the common residency path.
                let final_barrier_warps = self.warps[first..first + count]
                    .iter()
                    .enumerate()
                    .filter_map(|(offset, warp)| {
                        (warp.at_block_barrier && warp.next == warp.insts.len())
                            .then_some(first + offset)
                    })
                    .collect::<Vec<_>>();
                for w in &mut self.warps[first..first + count] {
                    if w.at_block_barrier {
                        let own_arrival = w
                            .block_barrier_arrival
                            .expect("barrier waiter owns an arrival timestamp");
                        let other_arrival = if count <= 1 {
                            None
                        } else if own_arrival == latest_arrival && latest_count == 1 {
                            Some(second_latest_arrival)
                        } else {
                            Some(latest_arrival)
                        };
                        w.at_block_barrier = false;
                        w.block_barrier_dispatch = None;
                        w.block_barrier_arrival = None;
                        if self.barrier_issue.is_some() {
                            w.barrier_pipeline_ready = checked_cycle_add(
                                "CTA-barrier own-arrival release",
                                own_arrival,
                                1,
                            )?;
                            w.cta_barrier_ready = other_arrival
                                .map(|arrival| {
                                    checked_cycle_add(
                                        "CTA-barrier other-participant rendezvous",
                                        arrival,
                                        1,
                                    )
                                })
                                .transpose()?
                                .unwrap_or(0);
                            debug_assert_eq!(
                                w.barrier_pipeline_ready.max(w.cta_barrier_ready),
                                release_at
                            );
                        } else {
                            w.barrier_pipeline_ready = 0;
                            w.cta_barrier_ready = release_at;
                        }
                    }
                }
                for warp_index in final_barrier_warps {
                    let subcore = self.warps[warp_index]
                        .warp_slot
                        .rem_euclid(SUB_CORES as i32) as usize;
                    // Admission of the next CTA is causally after the barrier
                    // release, not after the last warp's dispatch cycle.
                    self.retire_warp(subcore, warp_index, release_at)?;
                }
            }
        }
        if let Some(plan) = memory_pipeline_plan {
            self.memory_issue_pipeline = Some(plan.next);
        }
        if let Some(plan) = scalar_execution_pipeline_plan {
            self.scalar_execution_pipeline = Some(Box::new(plan.next));
        }
        for (warp, ready) in imc_waits
            .iter()
            .flatten()
            .copied()
            .chain(extra_imc_waits.iter().copied())
        {
            self.warps[warp].imc_wait_until = ready;
        }
        if let Some(cache) = imc_cache.as_mut() {
            for (warp, _) in imc_waits
                .iter()
                .flatten()
                .copied()
                .chain(extra_imc_waits.iter().copied())
            {
                let subcore = self.warps[warp].warp_slot.rem_euclid(SUB_CORES as i32) as usize;
                cache[subcore].diagnostics.miss_wait_events = cache[subcore]
                    .diagnostics
                    .miss_wait_events
                    .saturating_add(1);
                cache[subcore].diagnostics.miss_wait_cycles = cache[subcore]
                    .diagnostics
                    .miss_wait_cycles
                    .saturating_add(1);
            }
        }
        if imc_cache.is_some() {
            self.immediate_constant_switch_remaining = projected_immediate_constant_switch;
            self.immediate_constant_cache = imc_cache;
        }
        // Counts are bounded by next_issued, whose checked increment was
        // preflighted above. Multiple issue ports count only one active cycle.
        for (subcore, pick) in effective_picks.iter().enumerate() {
            if pick.is_some() {
                self.instructions_per_smsp[subcore] = self.instructions_per_smsp[subcore]
                    .checked_add(1)
                    .ok_or(IssueError::CounterOverflow {
                        counter: "instructions per SMSP",
                    })?;
                if self.last_observed_issue[subcore] != Some(t) {
                    self.issue_active_cycles_per_smsp[subcore] = self.issue_active_cycles_per_smsp
                        [subcore]
                        .checked_add(1)
                        .ok_or(IssueError::CounterOverflow {
                            counter: "issue-active cycles per SMSP",
                        })?;
                    self.last_observed_issue[subcore] = Some(t);
                }
            }
        }
        self.issued = next_issued;
        *self.readiness = projected_readiness;
        self.memory_service_candidates = projected_memory_service_candidates;
        if let Some(plan) = memory_admission_feedback_plan {
            self.memory_admission_feedback
                .as_mut()
                .expect("feedback issue plan requires installed feedback state")
                .apply_issue(plan, t);
        }
        if issued_this_cycle {
            self.last_issue_picks = effective_picks;
        }
        Ok(issued_this_cycle)
    }
}
