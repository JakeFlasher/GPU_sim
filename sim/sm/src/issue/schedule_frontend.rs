//! Cohesive child mechanism extracted from the parent module.

use super::*;

impl<'a> SmCore<'a> {
    #[cfg(feature = "reference")]
    pub(crate) fn advance_diagnostics(&self) -> AdvanceDiagnostics {
        self.advance_diagnostics
    }
}

impl<'a> SmCore<'a> {
    /// Consume a comparator-selected typed contract. This remains an opt-in
    /// reference seam; callers must validate the contract and continuation
    /// before invoking it, and exact replay is represented by `Ok(None)`.
    #[cfg(feature = "reference")]
    pub(crate) fn execute_reference_boundary_result(
        &mut self,
        result: &NativeBoundaryExecutionResult,
    ) -> Result<Option<ReferenceScalarMacroExecution>, IssueError> {
        let NativeBoundaryExecutionResult::TypedMacro { contract } = result else {
            return Ok(None);
        };
        let core = contract
            .entry
            .cores
            .first()
            .ok_or(IssueError::ReferenceMacroUnsupported {
                reason: "typed_macro_core_missing",
            })?;
        let warp = core
            .warp_states
            .first()
            .ok_or(IssueError::ReferenceMacroUnsupported {
                reason: "typed_macro_warp_missing",
            })?;
        let count = u64::try_from(contract.internal_effects.len()).map_err(|_| {
            IssueError::ReferenceMacroUnsupported {
                reason: "typed_macro_effect_count_out_of_range",
            }
        })?;
        let interval = contract
            .internal_effects
            .windows(2)
            .next()
            .and_then(|effects| effects[1].event.time.checked_sub(effects[0].event.time))
            .unwrap_or(1)
            .max(1);
        let terminal_block_barrier = contract
            .observable_labels
            .iter()
            .any(|label| label == "terminal_scalar_barrier_v1");
        let terminal_front_end_exit = contract
            .exit
            .cores
            .first()
            .and_then(|core| core.front_end.as_ref());
        self.execute_reference_macro_path(
            &NativeBoundaryExecutionPath::TypedMacro,
            warp.warp_slot,
            count,
            interval,
            terminal_block_barrier,
            &contract.internal_effects,
            terminal_front_end_exit,
        )
    }
}

impl<'a> SmCore<'a> {
    #[cfg(feature = "reference")]
    pub(crate) fn typed_async_copy_states(&self) -> Vec<Option<AsyncCopyBoundaryState>> {
        self.async_copies
            .iter()
            .map(|slot| {
                slot.state.as_ref().map(|state| {
                    let project = |group: AsyncCopyGroup| AsyncGroupBoundaryState {
                        sequence: group.id.0,
                        committed_at: group.committed_at,
                        pending_sources: group.pending_sources,
                        source_completion_time: group.source_completion_time,
                        delivery_work: group.delivery_work,
                        destination_wavefronts: group.destination_wavefronts,
                        destination_service_queued: group.destination_service_queued,
                        destination_completion_time: group.destination_completion_time,
                        completion_time: group.completion_time,
                    };
                    let mut bypass_observations = state
                        .bypass_copies
                        .iter()
                        .map(|copy| AsyncCopyBypassObservationState {
                            group: copy.group.0,
                            instruction: u64::try_from(copy.instruction).unwrap_or(u64::MAX),
                            pending_sources: copy.pending_sources,
                            source_completion_time: copy.source_completion_time,
                            delivery_work: copy.delivery_work,
                            source_cohorts: copy.source_cohorts,
                            destination_lines: copy.destination_lines,
                            max_bank_wavefronts: copy.max_bank_wavefronts,
                            max_bypass_wavefronts: copy.max_bypass_wavefronts,
                            sector_off: u64::try_from(copy.sector_off).unwrap_or(u64::MAX),
                            sector_membership_masks: copy.sector_membership_masks.to_vec(),
                            cohorts: copy
                                .cohorts
                                .iter()
                                .map(|cohort| AsyncCopyCohortBoundaryState {
                                    pending_sources: cohort.pending_sources,
                                    first_source_response: cohort.first_source_response,
                                    last_source_response: cohort.last_source_response,
                                    payload_work: cohort.payload_work,
                                    destination_lines: cohort.destination_lines,
                                    bank_wavefronts: cohort.bank_wavefronts,
                                    bypass_wavefronts: cohort.bypass_wavefronts,
                                    topology_penalty: cohort.topology_penalty,
                                })
                                .collect(),
                        })
                        .collect::<Vec<_>>();
                    bypass_observations
                        .sort_by_key(|observation| (observation.group, observation.instruction));
                    // Aggregate-only fallback captures preserve useful
                    // diagnostics but do not retain the cohort/membership
                    // topology required by an exact transition contract.
                    let bypass_observations_complete =
                        bypass_observations.iter().all(|observation| {
                            observation.source_cohorts != 0
                                && !observation.sector_membership_masks.is_empty()
                                && !observation.cohorts.is_empty()
                        });
                    AsyncCopyBoundaryState {
                        open: state.open_group.map(project),
                        committed: state
                            .committed_groups
                            .iter()
                            .copied()
                            .map(project)
                            .collect(),
                        bypass_observation_count: u64::try_from(state.bypass_copies.len())
                            .unwrap_or(u64::MAX),
                        next_sequence: state.next_sequence,
                        bypass_observations_complete,
                        bypass_observations,
                    }
                })
            })
            .collect()
    }
}

impl<'a> SmCore<'a> {
    #[cfg(feature = "reference")]
    pub(crate) fn typed_front_end_state(&self) -> Option<InstructionFrontEndBoundaryState> {
        let state = self.instruction_front_end.as_ref()?;
        let cache = |cache: &SetAssociativeInstructionCache| InstructionCacheState {
            ways: u64::try_from(cache.ways).unwrap_or(u64::MAX),
            access_sequence: cache.access_sequence,
            sets: cache
                .sets
                .iter()
                .map(|set| {
                    set.iter()
                        .map(|entry| InstructionCacheEntryState {
                            function_unique_id: entry.key.function_unique_id,
                            line: entry.key.line,
                            ready_at: entry.ready_at,
                            last_access: entry.last_access,
                        })
                        .collect()
                })
                .collect(),
        };
        let streams = state
            .streams
            .iter()
            .map(|stream| InstructionStreamState {
                lines: stream
                    .lines
                    .iter()
                    .map(|line| InstructionStreamLineState {
                        function_unique_id: line.key.function_unique_id,
                        line: line.key.line,
                        ready_at: line.ready_at,
                        demanded: line.demanded,
                    })
                    .collect(),
                next_line: stream
                    .next_line
                    .map(|key| (key.function_unique_id, key.line)),
            })
            .collect();
        let diagnostics = state.diagnostics;
        Some(InstructionFrontEndBoundaryState {
            line_bytes: state.config.line_bytes,
            l0_to_l1_cycles: state.config.l0_to_l1_cycles,
            l1_to_l0_cycles: state.config.l1_to_l0_cycles,
            l1_request_ports: state.config.l1_request_ports,
            l1_reply_ports: state.config.l1_reply_ports,
            stream_buffer_lines: state.config.stream_buffer_lines,
            stream_buffers_per_subcore: state.config.stream_buffers_per_subcore,
            prefetches_per_cycle: state.config.prefetches_per_cycle,
            instruction_buffer_entries: state.config.instruction_buffer_entries,
            fetch_decode_width: state.config.fetch_decode_width,
            fetch_to_issue_stages: state.config.fetch_to_issue_stages,
            l0: state.l0.iter().map(cache).collect(),
            l1: cache(&state.l1),
            streams,
            l1_request_ready: state.l1_request_ready,
            l1_reply_ready: state.l1_reply_ready,
            l0_response_ready: state.l0_response_ready.to_vec(),
            prefetch_priority: u64::try_from(state.prefetch_priority).unwrap_or(u64::MAX),
            cold_fill_latency: state.cold_fill_latency,
            instructions_fetched: diagnostics.instructions_fetched,
            demand_l0_hits: diagnostics.demand_l0_hits,
            demand_stream_buffer_hits: diagnostics.demand_stream_buffer_hits,
            demand_l0_misses: diagnostics.demand_l0_misses,
            demand_l1_hits: diagnostics.demand_l1_hits,
            demand_l1_misses: diagnostics.demand_l1_misses,
            prefetch_requests: diagnostics.prefetch_requests,
            prefetch_l1_hits: diagnostics.prefetch_l1_hits,
            prefetch_l1_misses: diagnostics.prefetch_l1_misses,
            stream_restarts: diagnostics.stream_restarts,
            maximum_fetch_latency_cycles: diagnostics.maximum_fetch_latency_cycles,
        })
    }
}

impl<'a> SmCore<'a> {
    pub(crate) fn is_done(&self) -> bool {
        self.unfinished_warps == 0
    }
}

impl<'a> SmCore<'a> {
    /// Instructions issued so far (monotonic; the system driver's
    /// progress signal).
    pub(crate) fn instructions_issued(&self) -> u64 {
        self.issued
    }
}

impl<'a> SmCore<'a> {
    /// Result snapshot; meaningful once [`Self::advance`] returned
    /// [`Advance::Done`].
    pub(crate) fn result(&self) -> SmSimResult {
        let mut readiness = (*self.readiness).clone();
        // The released Huerta scheduler orders a greedy warp followed by
        // descending dynamic warp IDs.  The replay admission path currently
        // stamps warps in the validated block-roster order; retain an explicit
        // audit of any within-CTA inversion instead of silently assuming the
        // two orders are equivalent. This diagnostic never selects timing.
        let mut priority_mismatches = 0_u64;
        for left in 0..self.warps.len() {
            for right in (left + 1)..self.warps.len() {
                let a = &self.warps[left];
                let b = &self.warps[right];
                if a.block != b.block
                    || a.warp_slot.rem_euclid(SUB_CORES as i32)
                        != b.warp_slot.rem_euclid(SUB_CORES as i32)
                {
                    continue;
                }
                let arrival_order = a.arrival.cmp(&b.arrival);
                let dynamic_order = a.warp_id.cmp(&b.warp_id);
                if arrival_order != dynamic_order {
                    priority_mismatches = priority_mismatches.saturating_add(1);
                }
            }
        }
        readiness.youngest_priority_order_mismatches = priority_mismatches;
        if let Some(front_end) = self.instruction_front_end.as_ref() {
            readiness.instruction_front_end = Some(Box::new(front_end.diagnostics));
        }
        if let Some(memory_pipeline) = self.memory_issue_pipeline.as_ref() {
            readiness.memory_issue_pipeline = Some(Box::new(memory_pipeline.diagnostics));
        }
        if let Some(cache) = self.immediate_constant_cache.as_ref() {
            let mut diagnostics = ImmediateConstantCacheDiagnostics::default();
            for state in cache.iter() {
                let addend = state.diagnostics();
                diagnostics.references = diagnostics.references.saturating_add(addend.references);
                diagnostics.lookup_hits =
                    diagnostics.lookup_hits.saturating_add(addend.lookup_hits);
                diagnostics.lookup_misses = diagnostics
                    .lookup_misses
                    .saturating_add(addend.lookup_misses);
                diagnostics.pending_tag_hits = diagnostics
                    .pending_tag_hits
                    .saturating_add(addend.pending_tag_hits);
                diagnostics.tag_misses = diagnostics.tag_misses.saturating_add(addend.tag_misses);
                diagnostics.pending_merges = diagnostics
                    .pending_merges
                    .saturating_add(addend.pending_merges);
                diagnostics.lines_filled =
                    diagnostics.lines_filled.saturating_add(addend.lines_filled);
                diagnostics.miss_wait_events = diagnostics
                    .miss_wait_events
                    .saturating_add(addend.miss_wait_events);
                diagnostics.miss_wait_cycles = diagnostics
                    .miss_wait_cycles
                    .saturating_add(addend.miss_wait_cycles);
                diagnostics.greedy_switch_block_cycles = diagnostics
                    .greedy_switch_block_cycles
                    .saturating_add(addend.greedy_switch_block_cycles);
                diagnostics.non_greedy_scan_attempts = diagnostics
                    .non_greedy_scan_attempts
                    .saturating_add(addend.non_greedy_scan_attempts);
                diagnostics.non_greedy_scan_misses = diagnostics
                    .non_greedy_scan_misses
                    .saturating_add(addend.non_greedy_scan_misses);
                diagnostics.non_greedy_scan_successes = diagnostics
                    .non_greedy_scan_successes
                    .saturating_add(addend.non_greedy_scan_successes);
                diagnostics.maximum_pending_lines = diagnostics
                    .maximum_pending_lines
                    .max(addend.maximum_pending_lines);
            }
            if !diagnostics.is_empty() {
                readiness.immediate_constant_cache = Some(Box::new(diagnostics));
            }
        }
        readiness.memory_service_candidates = self.memory_service_candidates.diagnostics();
        if let Some(feedback) = self.memory_admission_feedback.as_ref() {
            readiness.memory_admission_feedback = Some(Box::new(feedback.diagnostics()));
            let write_ack = feedback.write_ack_diagnostics();
            if !write_ack.is_empty() {
                readiness.memory_write_ack = Some(Box::new(write_ack));
            }
        }
        let pipeline_completion = self
            .memory_issue_pipeline
            .as_ref()
            .filter(|pipeline| !pipeline.diagnostics.is_empty())
            .map(|pipeline| pipeline.diagnostics.terminal_shared_accept_cycle);
        let modeled_local_completion_cycles =
            match (pipeline_completion, self.local_completion_time) {
                (Some(pipeline), Some(local)) => Some(pipeline.max(local)),
                (Some(value), None) | (None, Some(value)) => Some(value),
                (None, None) => None,
            };
        let activity = SmActivity::from_lifetimes(
            self.warp_lifetimes.clone(),
            self.instructions_per_smsp,
            self.issue_active_cycles_per_smsp,
            self.params.max_resident_warps_per_sm,
        );
        SmSimResult {
            instructions_issued: self.issued,
            cycles: self.t,
            modeled_local_completion_cycles,
            l1: self.mem.l1().map(|c| c.counters),
            l1_merge_audit: self.mem.l1().map(L1Cache::merge_audit),
            unmodeled_tails: self.unmodeled_tails,
            readiness,
            activity,
        }
    }
}

impl<'a> SmCore<'a> {
    pub(super) fn project_selected_readiness(
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
}

impl<'a> SmCore<'a> {
    pub(super) fn preflight_issue_accounting(
        &self,
        picks: &[Option<usize>; SUB_CORES],
        barrier_arrival_times: &[Option<u64>; SUB_CORES],
    ) -> Result<u64, IssueError> {
        let issue_add = u64::try_from(picks.iter().flatten().count()).map_err(|_| {
            IssueError::CounterOverflow {
                counter: "issued instruction",
            }
        })?;
        let next_issued = checked_u64_counter_add("issued instruction", self.issued, issue_add)?;
        if next_issued > self.instruction_limit {
            return Err(IssueError::InstructionCountExceedsReplay {
                issued: self.issued,
                add: issue_add,
                limit: self.instruction_limit,
            });
        }

        // (block, ordinary retirements, barrier arrivals, final-barrier
        // arrivals). Final barrier arrivals retire only after the generation
        // releases, so they must not be included in `ordinary retirements`.
        let mut block_deltas: [Option<(usize, usize, usize, usize)>; SUB_CORES] = [None; SUB_CORES];
        let mut released_blocks: [Option<usize>; SUB_CORES] = [None; SUB_CORES];
        let mut total_retirements = 0usize;
        for &wix in picks.iter().flatten() {
            let warp = &self.warps[wix];
            let slot = warp
                .next_slot
                .ok_or(IssueError::MissingIssueSlot { warp: wix })?;
            let reaches_end = warp.next_issue_retires()?;
            let arrives = slot.op_class == OpClass::BlockBarrier;
            let retires = reaches_end && !arrives;
            let entry_index = block_deltas
                .iter()
                .position(|entry| entry.is_some_and(|(block, _, _, _)| block == warp.block))
                .or_else(|| block_deltas.iter().position(Option::is_none))
                .ok_or(IssueError::CounterOverflow {
                    counter: "per-cycle block accounting",
                })?;
            let entry = block_deltas[entry_index].get_or_insert((warp.block, 0, 0, 0));
            if retires {
                entry.1 = checked_usize_counter_add("block retirement", entry.1, 1)?;
                total_retirements =
                    checked_usize_counter_add("unfinished warp retirement", total_retirements, 1)?;
            } else if arrives {
                entry.2 = checked_usize_counter_add("block barrier waiter", entry.2, 1)?;
                if reaches_end {
                    entry.3 = checked_usize_counter_add("final block barrier arrival", entry.3, 1)?;
                }
            }
        }

        self.unfinished_warps.checked_sub(total_retirements).ok_or(
            IssueError::CounterUnderflow {
                counter: "unfinished warp",
            },
        )?;
        let mut residency_released = 0u32;
        let mut shared_memory_released = 0u64;
        for (block, retirements, arrivals, final_arrivals) in block_deltas.into_iter().flatten() {
            let state = &self.block_states[block];
            let unfinished =
                state
                    .unfinished
                    .checked_sub(retirements)
                    .ok_or(IssueError::CounterUnderflow {
                        counter: "block unfinished warp",
                    })?;
            let waiting =
                checked_usize_counter_add("block barrier waiter", state.waiting, arrivals)?;
            if waiting > unfinished {
                return Err(IssueError::CounterOverflow {
                    counter: "block barrier waiter",
                });
            }
            let mut post_release_unfinished = unfinished;
            if waiting > 0 && waiting == unfinished {
                let existing_final_arrivals = self.warps
                    [state.first_warp..state.first_warp + state.admitted_warps]
                    .iter()
                    .filter(|warp| warp.at_block_barrier && warp.next == warp.insts.len())
                    .count();
                let final_retirements = checked_usize_counter_add(
                    "final block barrier retirement",
                    existing_final_arrivals,
                    final_arrivals,
                )?;
                post_release_unfinished = unfinished.checked_sub(final_retirements).ok_or(
                    IssueError::CounterUnderflow {
                        counter: "final block barrier unfinished warp",
                    },
                )?;
                total_retirements = checked_usize_counter_add(
                    "unfinished warp retirement",
                    total_retirements,
                    final_retirements,
                )?;
                let mut last_arrival = self.t;
                for warp in &self.warps[state.first_warp..state.first_warp + state.admitted_warps] {
                    if let Some(arrival) = warp.block_barrier_arrival {
                        last_arrival = last_arrival.max(arrival);
                    }
                }
                for (subcore, pick) in picks.iter().enumerate() {
                    if pick.is_some_and(|warp| self.warps[warp].block == block) {
                        last_arrival =
                            last_arrival.max(barrier_arrival_times[subcore].unwrap_or(self.t));
                    }
                }
                checked_cycle_add("block-barrier release", last_arrival, 1)?;
                if let Some(issue) = self.barrier_issue {
                    checked_cycle_add(
                        "CTA-barrier generation re-arm",
                        last_arrival,
                        issue.generation_rearm_cycles,
                    )?;
                }
            }
            // A block can become empty as part of the barrier release when its
            // final instructions are BAR.SYNC. `post_release_unfinished`
            // accounts for that deferred retirement before releasing residency.
            if post_release_unfinished == 0 {
                for &warp_slot in &state.warp_slots {
                    if !self.resident_warp_slots.contains(&warp_slot) {
                        return Err(IssueError::ResidentWarpSlotMissing {
                            launch_index: self.warps[state.first_warp].launch_index,
                            warp_slot,
                        });
                    }
                }
                let release_index = released_blocks.iter().position(Option::is_none).ok_or(
                    IssueError::CounterOverflow {
                        counter: "CTA barrier diagnostic released block",
                    },
                )?;
                released_blocks[release_index] = Some(block);
                residency_released = checked_u32_counter_add(
                    "resident warp retirement",
                    residency_released,
                    state.warp_count,
                )?;
                shared_memory_released = checked_u64_counter_add(
                    "resident shared-memory retirement",
                    shared_memory_released,
                    state.shared_memory_bytes,
                )?;
            }
        }
        self.unfinished_warps.checked_sub(total_retirements).ok_or(
            IssueError::CounterUnderflow {
                counter: "unfinished warp",
            },
        )?;
        self.resident_warps.checked_sub(residency_released).ok_or(
            IssueError::CounterUnderflow {
                counter: "resident warp",
            },
        )?;
        self.resident_shared_memory_bytes
            .checked_sub(shared_memory_released)
            .ok_or(IssueError::CounterUnderflow {
                counter: "resident shared-memory byte",
            })?;
        if released_blocks.iter().any(Option::is_some) {
            // `retire_warp` admits the next launch only after applying these
            // releases.  Validate that entire admission chain now, while the
            // issue cycle is still transactional; otherwise a deferred counter
            // overflow (for example, the arrival-stamp increment) could retire
            // an earlier warp before returning its error.
            let mut projected_resident_slots = (*self.resident_warp_slots).clone();
            let mut projected_resident_blocks = self
                .block_states
                .iter()
                .filter(|state| state.unfinished != 0)
                .count() as u32;
            for block in released_blocks.into_iter().flatten() {
                projected_resident_blocks = projected_resident_blocks.checked_sub(1).ok_or(
                    IssueError::CounterUnderflow {
                        counter: "resident block",
                    },
                )?;
                for &warp_slot in &self.block_states[block].warp_slots {
                    let removed = projected_resident_slots.remove(&warp_slot);
                    if !removed {
                        return Err(IssueError::ResidentWarpSlotMissing {
                            launch_index: self.warps[self.block_states[block].first_warp]
                                .launch_index,
                            warp_slot,
                        });
                    }
                }
            }
            self.preflight_block_admissions(
                self.resident_warps.checked_sub(residency_released).ok_or(
                    IssueError::CounterUnderflow {
                        counter: "resident warp",
                    },
                )?,
                self.unfinished_warps.checked_sub(total_retirements).ok_or(
                    IssueError::CounterUnderflow {
                        counter: "unfinished warp",
                    },
                )?,
                self.resident_shared_memory_bytes
                    .checked_sub(shared_memory_released)
                    .ok_or(IssueError::CounterUnderflow {
                        counter: "resident shared-memory byte",
                    })?,
                projected_resident_blocks,
                projected_resident_slots,
                self.arrival_stamp,
            )?;
        }
        Ok(next_issued)
    }
}

impl<'a> SmCore<'a> {
    pub(super) fn preflight_result_writeback_priority(
        &self,
        picks: &[Option<usize>; SUB_CORES],
    ) -> Result<(), IssueError> {
        let Some(state) = self.result_writebacks.as_ref() else {
            return Ok(());
        };
        let mut collisions = 0u64;
        let mut deferral_cycles = 0u64;
        for (subcore, pick) in picks.iter().enumerate() {
            let Some(wix) = *pick else { continue };
            let slot = self.warps[wix]
                .next_slot
                .ok_or(IssueError::MissingIssueSlot { warp: wix })?;
            let Some(ResultWritebackKind::FixedHadd2 { bank }) =
                result_writeback_kind(slot, self.params.arch)
            else {
                continue;
            };
            let fixed_at = checked_cycle_add(
                "HADD2 result writeback",
                self.t,
                state.timing.hadd2_writeback_cycles,
            )?;
            for event in &state.pending {
                if event.subcore == subcore as u8
                    && event.bank == bank
                    && event.writeback_at == fixed_at
                {
                    checked_cycle_add(
                        "deferred L1TEX writeback",
                        event.writeback_at,
                        state.timing.variable_deferral_cycles,
                    )?;
                    checked_cycle_add(
                        "deferred L1TEX visibility",
                        event.visible_at,
                        state.timing.variable_deferral_cycles,
                    )?;
                    collisions = checked_u64_counter_add(
                        "fixed-over-L1TEX collision preflight",
                        collisions,
                        1,
                    )?;
                    deferral_cycles = checked_u64_counter_add(
                        "L1TEX writeback deferral-cycle preflight",
                        deferral_cycles,
                        state.timing.variable_deferral_cycles,
                    )?;
                }
            }
        }
        checked_u64_counter_add(
            "fixed-over-L1TEX collision diagnostic",
            self.readiness.result_writeback.fixed_over_l1tex_collisions,
            collisions,
        )?;
        checked_u64_counter_add(
            "L1TEX writeback deferral-cycle diagnostic",
            self.readiness.result_writeback.l1tex_deferral_cycles,
            deferral_cycles,
        )?;
        Ok(())
    }
}
