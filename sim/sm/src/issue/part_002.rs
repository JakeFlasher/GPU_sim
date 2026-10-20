// Lexical implementation section 3/8; included by the parent facade.
impl AsyncCopySlot {
    fn preflight_async_copy_add(
        &self,
        warp: usize,
        pending_add: u32,
        source_ready: u64,
        delivery_work: u64,
        destination_wavefronts: u64,
        bypass_copy: Option<BypassAsyncCopyInput<'_>>,
        params: &IssueModelParams,
    ) -> Result<AsyncCopyGroupId, IssueError> {
        let (id, current, group) = match self.state.as_deref() {
            Some(state) => match state.open_group {
                Some(group) => (group.id, group.pending_sources, group),
                None => {
                    let id = AsyncCopyGroupId(
                        state
                            .next_sequence
                            .ok_or(IssueError::AsyncCopyGroupSequenceExhausted { warp })?,
                    );
                    (id, 0, AsyncCopyGroup::new(id))
                }
            },
            None => {
                let id = AsyncCopyGroupId(0);
                (id, 0, AsyncCopyGroup::new(id))
            }
        };
        current
            .checked_add(pending_add)
            .ok_or(IssueError::AsyncCopyPendingOverflow {
                warp,
                group: id,
                current,
                add: pending_add,
            })?;
        if let Some(copy) = bypass_copy
            && self.state.as_deref().is_some_and(|state| {
                state.bypass_copies.iter().any(|existing| {
                    existing.group == id && existing.instruction == copy.instruction
                })
            })
        {
            return Err(IssueError::AsyncCopyObservationDuplicate {
                warp,
                group: id,
                instruction: copy.instruction,
            });
        }
        if let Some(copy) = bypass_copy {
            build_bypass_cohort_observation(warp, pending_add, delivery_work, copy, params)?;
        }
        group.project_delivery(source_ready, delivery_work, destination_wavefronts, params)?;
        Ok(id)
    }

    fn ensure_open_async_group(&mut self, warp: usize) -> Result<&mut AsyncCopyGroup, IssueError> {
        let state = self
            .state
            .get_or_insert_with(|| Box::new(AsyncCopyState::new()));
        if state.open_group.is_none() {
            let sequence = state
                .next_sequence
                .ok_or(IssueError::AsyncCopyGroupSequenceExhausted { warp })?;
            state.next_sequence = sequence.checked_add(1);
            state.open_group = Some(AsyncCopyGroup::new(AsyncCopyGroupId(sequence)));
        }
        state
            .open_group
            .as_mut()
            .ok_or(IssueError::AsyncCopyGroupSequenceExhausted { warp })
    }

    fn add_known_async_copy(
        &mut self,
        warp: usize,
        source_ready: u64,
        delivery_work: u64,
        destination_wavefronts: u64,
        params: &IssueModelParams,
    ) -> Result<AsyncCopyGroupId, IssueError> {
        let id = self.preflight_async_copy_add(
            warp,
            0,
            source_ready,
            delivery_work,
            destination_wavefronts,
            None,
            params,
        )?;
        let group = self.ensure_open_async_group(warp)?;
        debug_assert_eq!(group.id, id);
        let (source_completion_time, delivery_work, destination_wavefronts, completion_time) =
            group.project_delivery(source_ready, delivery_work, destination_wavefronts, params)?;
        group.source_completion_time = source_completion_time;
        group.delivery_work = delivery_work;
        group.destination_wavefronts = destination_wavefronts;
        group.completion_time = completion_time;
        Ok(id)
    }

    fn add_pending_async_copy_sources(
        &mut self,
        warp: usize,
        add: u32,
        delivery_work: u64,
        destination_wavefronts: u64,
        bypass_copy: Option<BypassAsyncCopyInput<'_>>,
        params: &IssueModelParams,
    ) -> Result<AsyncCopyGroupId, IssueError> {
        let id = self.preflight_async_copy_add(
            warp,
            add,
            0,
            delivery_work,
            destination_wavefronts,
            bypass_copy,
            params,
        )?;
        let group = self.ensure_open_async_group(warp)?;
        debug_assert_eq!(group.id, id);
        let current = group.pending_sources;
        let pending_sources =
            current
                .checked_add(add)
                .ok_or(IssueError::AsyncCopyPendingOverflow {
                    warp,
                    group: id,
                    current,
                    add,
                })?;
        let (
            source_completion_time,
            aggregate_delivery_work,
            destination_wavefronts,
            completion_time,
        ) = group.project_delivery(0, delivery_work, destination_wavefronts, params)?;
        group.pending_sources = pending_sources;
        group.source_completion_time = source_completion_time;
        group.delivery_work = aggregate_delivery_work;
        group.destination_wavefronts = destination_wavefronts;
        group.completion_time = completion_time;
        if let Some(copy) = bypass_copy {
            let (sector_membership_masks, cohorts) =
                build_bypass_cohort_observation(warp, add, delivery_work, copy, params)?;
            if delivery_work != 0 {
                self.state
                    .as_deref_mut()
                    .expect("an observed bypass copy has allocated async-copy state")
                    .bypass_copies
                    .push(BypassAsyncCopyObservation {
                        group: id,
                        instruction: copy.instruction,
                        pending_sources: add,
                        source_completion_time: 0,
                        delivery_work,
                        source_cohorts: copy.source_cohorts,
                        destination_lines: copy.destination_lines,
                        max_bank_wavefronts: copy.max_bank_wavefronts,
                        max_bypass_wavefronts: copy.max_bypass_wavefronts,
                        sector_off: copy.sector_off,
                        sector_membership_masks,
                        cohorts,
                    });
            }
        }
        Ok(id)
    }

    fn async_group(&self, id: AsyncCopyGroupId) -> Option<&AsyncCopyGroup> {
        let state = self.state.as_deref()?;
        state
            .open_group
            .as_ref()
            .filter(|group| group.id == id)
            .or_else(|| state.committed_groups.iter().find(|group| group.id == id))
    }

    fn async_group_mut(&mut self, id: AsyncCopyGroupId) -> Option<&mut AsyncCopyGroup> {
        let state = self.state.as_deref_mut()?;
        if state
            .open_group
            .as_ref()
            .is_some_and(|group| group.id == id)
        {
            return state.open_group.as_mut();
        }
        state
            .committed_groups
            .iter_mut()
            .find(|group| group.id == id)
    }

    fn preflight_async_copy_source_completion(
        &self,
        warp: usize,
        id: AsyncCopyGroupId,
        bypass_response: Option<ResponseId>,
        remove: u32,
        now: u64,
        params: &IssueModelParams,
    ) -> Result<(), IssueError> {
        let group = self
            .async_group(id)
            .ok_or(IssueError::AsyncCopyGroupNotFound { warp, group: id })?;
        group
            .pending_sources
            .checked_sub(remove)
            .ok_or(IssueError::AsyncCopyPendingUnderflow {
                warp,
                group: id,
                current: group.pending_sources,
                remove,
            })?;
        group.project_delivery(now, 0, 0, params)?;
        if let Some(response) = bypass_response {
            let instruction = response.instruction;
            let copy = self
                .state
                .as_deref()
                .and_then(|state| {
                    state
                        .bypass_copies
                        .iter()
                        .find(|copy| copy.group == id && copy.instruction == instruction)
                })
                .ok_or(IssueError::AsyncCopyObservationNotFound {
                    warp,
                    group: id,
                    instruction,
                })?;
            copy.pending_sources.checked_sub(remove).ok_or(
                IssueError::AsyncCopyObservationPendingUnderflow {
                    warp,
                    group: id,
                    instruction,
                    current: copy.pending_sources,
                    remove,
                },
            )?;
            params.ldgsts_completion_ready(
                copy.source_completion_time.max(now),
                copy.delivery_work,
            )?;
            if !copy.cohorts.is_empty() {
                let sector_end = copy
                    .sector_off
                    .checked_add(copy.sector_membership_masks.len())
                    .ok_or(IssueError::CounterOverflow {
                        counter: "bypass cohort membership span",
                    })?;
                let relative = response
                    .sector_index
                    .checked_sub(copy.sector_off)
                    .filter(|&index| index < copy.sector_membership_masks.len())
                    .ok_or(IssueError::AsyncCopyCohortSectorOutOfRange {
                        warp,
                        group: id,
                        instruction,
                        sector_index: response.sector_index,
                        sector_off: copy.sector_off,
                        sector_end,
                    })?;
                let membership = copy.sector_membership_masks[relative];
                for (cohort_index, cohort) in copy.cohorts.iter().enumerate() {
                    if membership & (1_u32 << cohort_index) == 0 {
                        continue;
                    }
                    let pending = cohort.pending_sources.checked_sub(1).ok_or(
                        IssueError::AsyncCopyCohortPendingUnderflow {
                            warp,
                            group: id,
                            instruction,
                            sector_index: response.sector_index,
                            cohort: cohort_index,
                        },
                    )?;
                    let first = cohort
                        .first_source_response
                        .map_or(now, |first| first.min(now));
                    let last = cohort
                        .last_source_response
                        .map_or(now, |last| last.max(now));
                    if pending == 0 {
                        params.ldgsts_completion_ready(last, cohort.payload_work)?;
                        checked_cycle_add(
                            "bypass cohort topology-ready candidate",
                            first,
                            cohort.topology_penalty,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    fn complete_async_copy_sources(
        &mut self,
        warp: usize,
        id: AsyncCopyGroupId,
        bypass_response: Option<ResponseId>,
        remove: u32,
        now: u64,
        params: &IssueModelParams,
    ) -> Result<(), IssueError> {
        self.preflight_async_copy_source_completion(
            warp,
            id,
            bypass_response,
            remove,
            now,
            params,
        )?;
        let group = self
            .async_group_mut(id)
            .ok_or(IssueError::AsyncCopyGroupNotFound { warp, group: id })?;
        let current = group.pending_sources;
        let pending_sources =
            current
                .checked_sub(remove)
                .ok_or(IssueError::AsyncCopyPendingUnderflow {
                    warp,
                    group: id,
                    current,
                    remove,
                })?;
        let (source_completion_time, delivery_work, destination_wavefronts, completion_time) =
            group.project_delivery(now, 0, 0, params)?;
        group.pending_sources = pending_sources;
        group.source_completion_time = source_completion_time;
        group.delivery_work = delivery_work;
        group.destination_wavefronts = destination_wavefronts;
        group.completion_time = completion_time;
        if let Some(response) = bypass_response {
            let instruction = response.instruction;
            let copy = self
                .state
                .as_deref_mut()
                .and_then(|state| {
                    state
                        .bypass_copies
                        .iter_mut()
                        .find(|copy| copy.group == id && copy.instruction == instruction)
                })
                .expect("bypass observation was validated before aggregate mutation");
            copy.pending_sources -= remove;
            copy.source_completion_time = copy.source_completion_time.max(now);
            if !copy.cohorts.is_empty() {
                let relative = response.sector_index - copy.sector_off;
                let membership = copy.sector_membership_masks[relative];
                for (cohort_index, cohort) in copy.cohorts.iter_mut().enumerate() {
                    if membership & (1_u32 << cohort_index) == 0 {
                        continue;
                    }
                    cohort.pending_sources -= 1;
                    cohort.first_source_response = Some(
                        cohort
                            .first_source_response
                            .map_or(now, |first| first.min(now)),
                    );
                    cohort.last_source_response = Some(
                        cohort
                            .last_source_response
                            .map_or(now, |last| last.max(now)),
                    );
                }
            }
        }
        Ok(())
    }

    fn bypass_copy_projection(
        &self,
        warp: usize,
        group: &AsyncCopyGroup,
        params: &IssueModelParams,
        wait_base: u64,
    ) -> Result<Option<BypassAsyncCopyProjection>, IssueError> {
        let Some(state) = self.state.as_deref() else {
            return Ok(None);
        };
        let mut copies = 0_u64;
        let mut bank_wavefront_upper_bound = 0_u64;
        let mut bypass_wavefront_upper_bound = 0_u64;
        let mut source_cohorts = 0_u64;
        let mut destination_lines = 0_u64;
        let mut max_bank_wavefronts_sum = 0_u64;
        let mut max_bypass_wavefronts_sum = 0_u64;
        let mut earliest_source = u64::MAX;
        let mut latest_source = 0_u64;
        let mut per_copy_completion = 0_u64;
        let mut topology_candidate_completion = 0_u64;
        let mut topology_candidate_penalty_cycles = 0_u64;
        let mut pipeline_copies = Vec::new();
        let mut cohort_copies = 0_u64;
        let mut cohort_response_memberships = 0_u64;
        let mut cohort_response_span = 0_u64;
        let mut cohort_topology_penalty = 0_u64;
        let mut cohort_candidate_completion = 0_u64;
        // Each physical response envelope is counted once per copy. Hidden
        // work sums every cohort. Residual exposure instead follows the frozen
        // score's cohort -> copy -> group maximum; cross-group aggregation
        // happens in readiness diagnostics.
        let mut cohort_response_envelope = 0_u64;
        let mut cohort_topology_prearrival_hidden = 0_u64;
        let mut cohort_topology_stream_hidden = 0_u64;
        let mut cohort_topology_residual_exposure = 0_u64;
        for copy in state
            .bypass_copies
            .iter()
            .filter(|copy| copy.group == group.id)
        {
            if copy.pending_sources != 0 {
                return Err(IssueError::AsyncCopyObservationPendingAtWait {
                    warp,
                    group: group.id,
                    instruction: copy.instruction,
                    pending: copy.pending_sources,
                });
            }
            copies = checked_u64_counter_add("observed bypass asynchronous-copy count", copies, 1)?;
            source_cohorts = checked_u64_counter_add(
                "observed bypass source cohorts",
                source_cohorts,
                copy.source_cohorts,
            )?;
            destination_lines = checked_u64_counter_add(
                "observed bypass destination lines",
                destination_lines,
                copy.destination_lines,
            )?;
            max_bank_wavefronts_sum = checked_u64_counter_add(
                "observed bypass maximum bank wavefronts",
                max_bank_wavefronts_sum,
                copy.max_bank_wavefronts,
            )?;
            max_bypass_wavefronts_sum = checked_u64_counter_add(
                "observed bypass maximum projected wavefronts",
                max_bypass_wavefronts_sum,
                copy.max_bypass_wavefronts,
            )?;
            let bank_upper = copy
                .source_cohorts
                .checked_mul(copy.max_bank_wavefronts)
                .ok_or(IssueError::TimeMultiplicationOverflow {
                    operation: "bypass observed bank-wavefront upper bound",
                    lhs: copy.source_cohorts,
                    rhs: copy.max_bank_wavefronts,
                })?;
            bank_wavefront_upper_bound = checked_u64_counter_add(
                "bypass observed bank-wavefront upper bound",
                bank_wavefront_upper_bound,
                bank_upper,
            )?;
            let bypass_upper = copy
                .source_cohorts
                .checked_mul(copy.max_bypass_wavefronts)
                .ok_or(IssueError::TimeMultiplicationOverflow {
                    operation: "bypass observed projected-wavefront upper bound",
                    lhs: copy.source_cohorts,
                    rhs: copy.max_bypass_wavefronts,
                })?;
            bypass_wavefront_upper_bound = checked_u64_counter_add(
                "bypass observed projected-wavefront upper bound",
                bypass_wavefront_upper_bound,
                bypass_upper,
            )?;
            earliest_source = earliest_source.min(copy.source_completion_time);
            latest_source = latest_source.max(copy.source_completion_time);
            let payload_completion =
                params.ldgsts_completion_ready(copy.source_completion_time, copy.delivery_work)?;
            per_copy_completion = per_copy_completion.max(payload_completion);
            // A topology-only ignore-src copy has no source response stream;
            // retain its topology counters, but do not synthesize a bypass
            // candidate penalty from a zero-payload group.
            let topology_penalty = if copy.delivery_work == 0 {
                0
            } else {
                bypass_topology_candidate_penalty(
                    copy.max_bypass_wavefronts,
                    copy.max_bank_wavefronts,
                )?
            };
            topology_candidate_penalty_cycles = checked_u64_counter_add(
                "bypass topology candidate penalty diagnostic",
                topology_candidate_penalty_cycles,
                topology_penalty,
            )?;
            let candidate_completion = checked_cycle_add(
                "bypass topology candidate per-copy completion",
                payload_completion,
                topology_penalty,
            )?;
            topology_candidate_completion = topology_candidate_completion.max(candidate_completion);
            pipeline_copies.push(BypassPipelineCopyCandidate {
                instruction: copy.instruction,
                payload_completion,
                topology_penalty,
            });
            if !copy.cohorts.is_empty() {
                cohort_copies =
                    checked_u64_counter_add("observed bypass cohort copy count", cohort_copies, 1)?;
                let mut cohort_anchor: Option<u64> = None;
                let mut cohort_envelope_end = 0_u64;
                for (cohort_index, cohort) in copy.cohorts.iter().enumerate() {
                    if cohort.pending_sources != 0 {
                        return Err(IssueError::AsyncCopyCohortPendingAtWait {
                            warp,
                            group: group.id,
                            instruction: copy.instruction,
                            cohort: cohort_index,
                            pending: cohort.pending_sources,
                        });
                    }
                    let first = cohort.first_source_response.ok_or(
                        IssueError::AsyncCopyCohortProjectionInvalid {
                            warp,
                            instruction: copy.instruction,
                            field: "cohort first source response",
                            expected: 1,
                            got: 0,
                        },
                    )?;
                    let last = cohort.last_source_response.ok_or(
                        IssueError::AsyncCopyCohortProjectionInvalid {
                            warp,
                            instruction: copy.instruction,
                            field: "cohort last source response",
                            expected: 1,
                            got: 0,
                        },
                    )?;
                    if last < first {
                        return Err(IssueError::AsyncCopyCohortProjectionInvalid {
                            warp,
                            instruction: copy.instruction,
                            field: "cohort response order",
                            expected: first,
                            got: last,
                        });
                    }
                    cohort_anchor = Some(cohort_anchor.map_or(first, |anchor| anchor.min(first)));
                    cohort_envelope_end = cohort_envelope_end.max(last);
                }
                let cohort_anchor = cohort_anchor.expect("active cohort list is nonempty");

                let mut copy_residual_exposure = 0_u64;
                for (cohort_index, cohort) in copy.cohorts.iter().enumerate() {
                    let first = cohort
                        .first_source_response
                        .expect("cohort response presence was validated");
                    let last = cohort.last_source_response.ok_or(
                        IssueError::AsyncCopyCohortProjectionInvalid {
                            warp,
                            instruction: copy.instruction,
                            field: "cohort last source response",
                            expected: 1,
                            got: 0,
                        },
                    )?;
                    let prearrival_window = first - cohort_anchor;
                    let response_stream = last - first;
                    let prearrival_hidden = cohort.topology_penalty.min(prearrival_window);
                    let topology_after_prearrival = cohort.topology_penalty - prearrival_hidden;
                    let stream_hidden = topology_after_prearrival.min(response_stream);
                    let residual_exposure = topology_after_prearrival - stream_hidden;
                    cohort_topology_prearrival_hidden = checked_u64_counter_add(
                        "observed bypass cohort topology prearrival hidden work",
                        cohort_topology_prearrival_hidden,
                        prearrival_hidden,
                    )?;
                    cohort_topology_stream_hidden = checked_u64_counter_add(
                        "observed bypass cohort topology stream hidden work",
                        cohort_topology_stream_hidden,
                        stream_hidden,
                    )?;
                    copy_residual_exposure = copy_residual_exposure.max(residual_exposure);
                    let memberships = copy
                        .sector_membership_masks
                        .iter()
                        .filter(|&&mask| mask & (1_u32 << cohort_index) != 0)
                        .count();
                    cohort_response_memberships = checked_u64_counter_add(
                        "observed bypass cohort response memberships",
                        cohort_response_memberships,
                        u64::try_from(memberships).map_err(|_| IssueError::CounterOverflow {
                            counter: "observed bypass cohort response memberships",
                        })?,
                    )?;
                    cohort_response_span = checked_u64_counter_add(
                        "observed bypass cohort response span",
                        cohort_response_span,
                        last - first,
                    )?;
                    cohort_topology_penalty = checked_u64_counter_add(
                        "observed bypass cohort topology penalty",
                        cohort_topology_penalty,
                        cohort.topology_penalty,
                    )?;
                    let payload_ready =
                        params.ldgsts_completion_ready(last, cohort.payload_work)?;
                    let topology_ready = checked_cycle_add(
                        "bypass cohort topology-ready candidate",
                        first,
                        cohort.topology_penalty,
                    )?;
                    cohort_candidate_completion =
                        cohort_candidate_completion.max(payload_ready.max(topology_ready));
                }
                cohort_response_envelope = checked_u64_counter_add(
                    "observed bypass cohort response envelope",
                    cohort_response_envelope,
                    cohort_envelope_end - cohort_anchor,
                )?;
                cohort_topology_residual_exposure =
                    cohort_topology_residual_exposure.max(copy_residual_exposure);
            }
        }
        if copies == 0 {
            return Ok(None);
        }
        let pipeline_candidate_completion =
            bypass_pipeline_candidate_completions(&mut pipeline_copies)?;
        let aggregate_completion = group.local_completion_time(params)?;
        debug_assert!(
            aggregate_completion >= per_copy_completion,
            "group-aggregate delivery must upper-bound independent per-copy delivery"
        );
        Ok(Some(BypassAsyncCopyProjection {
            copies,
            bank_wavefront_upper_bound,
            bypass_wavefront_upper_bound,
            source_cohorts,
            destination_lines,
            max_bank_wavefronts_sum,
            max_bypass_wavefronts_sum,
            source_spread_cycles: latest_source.saturating_sub(earliest_source),
            per_copy_completion_cycles: per_copy_completion.saturating_sub(group.committed_at),
            group_aggregate_excess_cycles: aggregate_completion.saturating_sub(per_copy_completion),
            topology_candidate_penalty_cycles,
            topology_candidate_completion_cycles: topology_candidate_completion
                .saturating_sub(group.committed_at),
            topology_candidate_later_cycles: topology_candidate_completion
                .saturating_sub(aggregate_completion),
            topology_candidate_earlier_cycles: aggregate_completion
                .saturating_sub(topology_candidate_completion),
            pipeline_q1_completion_cycles: pipeline_candidate_completion[0]
                .saturating_sub(group.committed_at),
            pipeline_q1_exposure_cycles: pipeline_candidate_completion[0].saturating_sub(wait_base),
            pipeline_q2_completion_cycles: pipeline_candidate_completion[1]
                .saturating_sub(group.committed_at),
            pipeline_q2_exposure_cycles: pipeline_candidate_completion[1].saturating_sub(wait_base),
            pipeline_q4_completion_cycles: pipeline_candidate_completion[2]
                .saturating_sub(group.committed_at),
            pipeline_q4_exposure_cycles: pipeline_candidate_completion[2].saturating_sub(wait_base),
            pipeline_q8_completion_cycles: pipeline_candidate_completion[3]
                .saturating_sub(group.committed_at),
            pipeline_q8_exposure_cycles: pipeline_candidate_completion[3].saturating_sub(wait_base),
            pipeline_q16_completion_cycles: pipeline_candidate_completion[4]
                .saturating_sub(group.committed_at),
            pipeline_q16_exposure_cycles: pipeline_candidate_completion[4]
                .saturating_sub(wait_base),
            cohort_groups_observed: u64::from(cohort_copies != 0),
            cohort_copies_observed: cohort_copies,
            cohort_response_memberships_observed: cohort_response_memberships,
            cohort_response_span_cycles: cohort_response_span,
            cohort_topology_penalty_cycles: cohort_topology_penalty,
            cohort_candidate_completion_cycles: cohort_candidate_completion
                .saturating_sub(group.committed_at),
            cohort_candidate_exposure_cycles: cohort_candidate_completion.saturating_sub(wait_base),
            cohort_candidate_later_than_per_copy_cycles: cohort_candidate_completion
                .saturating_sub(per_copy_completion),
            cohort_candidate_earlier_than_per_copy_cycles: per_copy_completion
                .saturating_sub(cohort_candidate_completion),
            cohort_candidate_later_than_group_aggregate_cycles: cohort_candidate_completion
                .saturating_sub(aggregate_completion),
            cohort_candidate_earlier_than_group_aggregate_cycles: aggregate_completion
                .saturating_sub(cohort_candidate_completion),
            cohort_candidate_later_than_production_cycles: cohort_candidate_completion
                .saturating_sub(group.completion_time),
            cohort_candidate_earlier_than_production_cycles: group
                .completion_time
                .saturating_sub(cohort_candidate_completion),
            cohort_overlap_groups_observed: u64::from(cohort_copies != 0),
            cohort_response_envelope_cycles: cohort_response_envelope,
            cohort_topology_prearrival_hidden_cycles: cohort_topology_prearrival_hidden,
            cohort_topology_stream_hidden_cycles: cohort_topology_stream_hidden,
            cohort_topology_residual_exposure_cycles: cohort_topology_residual_exposure,
        }))
    }

    fn commit_async_copy_group(&mut self, now: u64) -> bool {
        let Some(state) = self.state.as_deref_mut() else {
            return false;
        };
        if let Some(mut group) = state.open_group.take() {
            group.committed_at = now;
            state.committed_groups.push_back(group);
            true
        } else {
            false
        }
    }

    fn async_wait_prefix_len(&self, max_pending_groups: u32) -> usize {
        let retained = usize::try_from(max_pending_groups).unwrap_or(usize::MAX);
        self.state.as_deref().map_or(0, |state| {
            state.committed_groups.len().saturating_sub(retained)
        })
    }

    /// Earliest known completion of every group older than the newest `N`.
    /// `None` means at least one required group still has a shared-L2 response
    /// outstanding, so only an external response event can supply a wake time.
    #[cold]
    #[inline(never)]
    fn async_wait_clear_time(&self, max_pending_groups: u32) -> Option<u64> {
        let Some(state) = self.state.as_deref() else {
            return Some(0);
        };
        let mut ready = 0;
        for group in state
            .committed_groups
            .iter()
            .take(self.async_wait_prefix_len(max_pending_groups))
        {
            if group.pending_sources != 0 {
                return None;
            }
            if group.destination_wavefronts != 0 && !group.destination_service_queued {
                return None;
            }
            ready = ready.max(group.completion_time);
        }
        Some(ready)
    }

    fn consume_async_wait(&mut self, max_pending_groups: u32, now: u64) {
        let consume = self.async_wait_prefix_len(max_pending_groups);
        if consume == 0 {
            return;
        }
        let state = self
            .state
            .as_deref_mut()
            .expect("a nonempty wait prefix requires async-copy state");
        debug_assert!(
            state
                .committed_groups
                .iter()
                .take(consume)
                .all(|group| group.is_complete_at(now)),
            "scheduler issued an asynchronous-copy wait before its required prefix completed"
        );
        let consumed_through = state.committed_groups[consume - 1].id.0;
        state
            .bypass_copies
            .retain(|copy| copy.group.0 > consumed_through);
        state.committed_groups.drain(..consume);
    }
}

impl WarpState<'_> {
    fn preflight_pending_add(
        &self,
        warp: usize,
        barrier: Option<DepBarrier>,
        add: u32,
    ) -> Result<(), IssueError> {
        let Some(barrier) = barrier else {
            return Ok(());
        };
        let current = self.barrier_pending[barrier.index()];
        current
            .checked_add(add)
            .ok_or(IssueError::PendingBarrierOverflow {
                warp,
                barrier: barrier.wire(),
                current,
                add,
            })?;
        Ok(())
    }

    fn preflight_pending_remove(
        &self,
        warp: usize,
        barrier: DepBarrier,
        remove: u32,
    ) -> Result<(), IssueError> {
        let current = self.barrier_pending[barrier.index()];
        current
            .checked_sub(remove)
            .ok_or(IssueError::PendingBarrierUnderflow {
                warp,
                barrier: barrier.wire(),
                current,
                remove,
            })?;
        Ok(())
    }

    /// Producer bookkeeping: pending (unknown write-back) units on a
    /// dependence barrier; `None` carries no dependency.
    fn add_pending(
        &mut self,
        warp: usize,
        barrier: Option<DepBarrier>,
        add: u32,
    ) -> Result<(), IssueError> {
        self.preflight_pending_add(warp, barrier, add)?;
        if let Some(barrier) = barrier {
            let pending = &mut self.barrier_pending[barrier.index()];
            let current = *pending;
            *pending = current
                .checked_add(add)
                .ok_or(IssueError::PendingBarrierOverflow {
                    warp,
                    barrier: barrier.wire(),
                    current,
                    add,
                })?;
        }
        Ok(())
    }

    /// Pending units resolved at `now`: decrement AND max-accumulate the
    /// clear stamp (hardware counter semantics — the single copy of the
    /// pairing every response path must preserve).
    fn clear_pending(
        &mut self,
        warp: usize,
        barrier: DepBarrier,
        remove: u32,
        now: u64,
    ) -> Result<(), IssueError> {
        self.preflight_pending_remove(warp, barrier, remove)?;
        let b = barrier.index();
        let current = self.barrier_pending[b];
        self.barrier_pending[b] =
            current
                .checked_sub(remove)
                .ok_or(IssueError::PendingBarrierUnderflow {
                    warp,
                    barrier: barrier.wire(),
                    current,
                    remove,
                })?;
        if now >= self.barrier_clear[b] {
            self.barrier_clear[b] = now;
            self.barrier_origin[b] = DependencyOrigin::L1Tex;
        }
        Ok(())
    }

    /// Release pending units WITHOUT a stamp (a retry entry's own units;
    /// its per-sector waiters carry the stamps).
    fn release_pending(
        &mut self,
        warp: usize,
        barrier: Option<DepBarrier>,
        remove: u32,
    ) -> Result<(), IssueError> {
        let Some(barrier) = barrier else {
            return Ok(());
        };
        self.preflight_pending_remove(warp, barrier, remove)?;
        let pending = &mut self.barrier_pending[barrier.index()];
        let current = *pending;
        *pending = current
            .checked_sub(remove)
            .ok_or(IssueError::PendingBarrierUnderflow {
                warp,
                barrier: barrier.wire(),
                current,
                remove,
            })?;
        Ok(())
    }

    /// Max-accumulate a KNOWN clear stamp on a dependence barrier (counter
    /// semantics: never overwrite); `None` carries no dependency.
    fn stamp_clear(&mut self, barrier: Option<DepBarrier>, at: u64, origin: DependencyOrigin) {
        if let Some(barrier) = barrier {
            let index = barrier.index();
            if at >= self.barrier_clear[index] {
                self.barrier_clear[index] = at;
                self.barrier_origin[index] = origin;
            }
        }
    }

    /// Earliest time the control wait mask and any explicit ordinary `DEPBAR`
    /// operand clear; `None` while a required barrier has responses outstanding
    /// (unknown clear time). SB0 asynchronous-copy waits are handled separately.
    fn wait_clear_time(&self) -> Option<u64> {
        let Some(slot) = self.next_slot else {
            return Some(0);
        };
        let mut t = 0;
        for b in 0..DEP_BARRIERS {
            if slot.control.wait_mask.contains_index(b) {
                if self.barrier_pending[b] > 0 {
                    return None;
                }
                t = t.max(self.barrier_clear[b]);
            }
        }
        for b in 0..DEP_BARRIERS {
            if !self.ordinary_dependency_wait_mask.contains_index(b)
                || slot.control.wait_mask.contains_index(b)
            {
                continue;
            }
            if self.barrier_pending[b] > 0 {
                return None;
            }
            t = t.max(self.barrier_clear[b]);
        }
        Some(t)
    }

    /// Earliest time at which the current instruction is ready when the
    /// immediate-constant (L0C/L1C) response frontier is deliberately
    /// ignored.  Huerta's SM86 scheduler treats that path specially: a
    /// greedy warp may remain the candidate for a bounded miss-switch
    /// window while all ordinary control/dependency frontiers are ready.
    fn ready_at_without_immediate_constant(&self) -> Option<u64> {
        if self.done || self.at_block_barrier {
            return None;
        }
        let dependency_ready = self.wait_clear_time()?;
        Some(
            self.eligible
                .max(dependency_ready)
                .max(self.cta_barrier_ready)
                .max(self.barrier_pipeline_ready)
                .max(self.branch_pipeline_ready),
        )
    }

    fn ready_at(&self) -> Option<u64> {
        self.ready_at_without_immediate_constant()
            .map(|ready| ready.max(self.imc_wait_until))
    }
}

fn async_copy_source_count(kind: AccessKind) -> Option<u16> {
    match kind {
        AccessKind::GlobalAsyncCopy {
            len,
            completion_work,
            ..
        } if !matches!(
            decode_async_copy_completion(completion_work),
            AsyncCopyCompletion::Inactive
        ) =>
        {
            // The full SM120 alpha route may retain no sector-arena span for
            // an opaque TMA/LDGSTS record. A nonzero completion token still
            // carries one logical source transaction; keep the source count
            // bounded without claiming a lane/sector cardinality.
            Some(len.max(1))
        }
        AccessKind::GlobalAsyncCopy { .. }
        | AccessKind::NonMemory
        | AccessKind::PredicatedOffMemory
        | AccessKind::Shared { .. }
        | AccessKind::LocalLoad { .. }
        | AccessKind::LocalStore { .. }
        | AccessKind::GlobalEligible { .. }
        | AccessKind::GlobalBypass { .. }
        | AccessKind::GlobalUntagged { .. }
        | AccessKind::GlobalStore { .. }
        | AccessKind::GlobalReduction { .. } => None,
    }
}

fn async_copy_completion_work(kind: AccessKind) -> Option<u64> {
    match kind {
        AccessKind::GlobalAsyncCopy {
            completion_work, ..
        } => match decode_async_copy_completion(completion_work) {
            AsyncCopyCompletion::Inactive => None,
            AsyncCopyCompletion::PayloadWork(work)
            | AsyncCopyCompletion::ExactDestinationWork(work) => Some(u64::from(work)),
            AsyncCopyCompletion::PayloadAndObservedTopology { payload_work, .. } => {
                Some(u64::from(payload_work))
            }
        },
        AccessKind::NonMemory
        | AccessKind::PredicatedOffMemory
        | AccessKind::Shared { .. }
        | AccessKind::LocalLoad { .. }
        | AccessKind::LocalStore { .. }
        | AccessKind::GlobalEligible { .. }
        | AccessKind::GlobalBypass { .. }
        | AccessKind::GlobalUntagged { .. }
        | AccessKind::GlobalStore { .. }
        | AccessKind::GlobalReduction { .. } => None,
    }
}

fn bypass_async_copy_observed_topology(
    kind: AccessKind,
    instruction: usize,
    exact: Option<AsyncCopyCohortView<'_>>,
) -> Option<BypassAsyncCopyInput<'_>> {
    let AccessKind::GlobalAsyncCopy {
        off,
        len,
        completion_work,
        l1_policy: LdgstsL1Policy::BypassL1,
        ..
    } = kind
    else {
        return None;
    };
    match decode_async_copy_completion(completion_work) {
        AsyncCopyCompletion::PayloadAndObservedTopology {
            source_cohorts,
            destination_lines,
            max_bank_wavefronts,
            max_bypass_wavefronts,
            ..
        } => Some(BypassAsyncCopyInput {
            instruction,
            sector_off: exact.map_or(off, |view| view.sector_off),
            sector_len: exact.map_or(len, |view| view.sector_len),
            source_cohorts: u64::from(source_cohorts),
            destination_lines: u64::from(destination_lines),
            max_bank_wavefronts: u64::from(max_bank_wavefronts),
            max_bypass_wavefronts: u64::from(max_bypass_wavefronts),
            cohorts: exact.map(|view| view.cohorts),
            sector_membership_masks: exact.map(|view| view.sector_membership_masks),
        }),
        AsyncCopyCompletion::Inactive
        | AsyncCopyCompletion::PayloadWork(_)
        | AsyncCopyCompletion::ExactDestinationWork(_) => None,
    }
}

fn async_copy_destination_wavefronts(
    kind: AccessKind,
    params: &IssueModelParams,
) -> Result<Option<u64>, IssueError> {
    let AccessKind::GlobalAsyncCopy {
        completion_work,
        l1_policy: LdgstsL1Policy::CacheAtAllLevels,
        ..
    } = kind
    else {
        return Ok(None);
    };
    let AsyncCopyCompletion::ExactDestinationWork(work) =
        decode_async_copy_completion(completion_work)
    else {
        return Ok(None);
    };
    let work_per_wavefront = params
        .ldgsts_cache_all_destination_wavefront_cycles
        .checked_mul(params.ldgsts_delivery_units_per_cycle())
        .ok_or(IssueError::TimeMultiplicationOverflow {
            operation: "LDGSTS destination work per wavefront",
            lhs: params.ldgsts_cache_all_destination_wavefront_cycles,
            rhs: params.ldgsts_delivery_units_per_cycle(),
        })?;
    let work = u64::from(work);
    if work_per_wavefront == 0 || work % work_per_wavefront != 0 {
        return Err(IssueError::InvalidAsyncCopyDestinationWork {
            completion_work: work,
            work_per_wavefront,
        });
    }
    Ok(Some(work / work_per_wavefront))
}

fn validate_async_copy_completion_work(kind: AccessKind, warp: usize) -> Result<(), IssueError> {
    if let AccessKind::GlobalAsyncCopy {
        len,
        completion_work,
        ..
    } = kind
        && len != 0
        && completion_work == 0
    {
        return Err(IssueError::AsyncCopyCompletionWorkMismatch {
            warp,
            sectors: len,
            completion_work: u64::from(completion_work),
        });
    }
    Ok(())
}

/// `LDGDEPBAR` encodes SB0 in the ordinary write-barrier field, but that bit is
/// the async-group protocol itself rather than a generic 25-cycle producer.
fn ordinary_write_barrier(slot: &IssueSlot) -> Option<DepBarrier> {
    if slot.op_class == OpClass::AsyncCopyCommit {
        None
    } else {
        slot.control.write_barrier
    }
}

/// Data-ready time for one issued record at time `t` in the Flat/L1Const
/// modes (fill times known at issue), driving the L1 state/counters when
/// attached. Producer visibility is computed when either an ordinary write
/// barrier or an async-copy group consumes it; cache accesses still run without
/// either observer because they mutate modeled state.
pub(crate) fn data_ready_time(
    kind: AccessKind,
    sectors: &[i64],
    t: u64,
    params: &IssueModelParams,
    l1: Option<&mut L1Cache>,
    write_barrier: Option<DepBarrier>,
    observe_async_copy: bool,
) -> Result<Option<u64>, IssueError> {
    let needs_ready = write_barrier.is_some() || observe_async_copy;
    match kind {
        AccessKind::PredicatedOffMemory => Ok(None),
        AccessKind::NonMemory => {
            if needs_ready {
                params.non_memory_ready(t).map(Some)
            } else {
                Ok(None)
            }
        }
        AccessKind::Shared { wavefronts, .. } => {
            if needs_ready {
                params.shared_ready(t, wavefronts).map(Some)
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
            match l1 {
                Some(l1) => {
                    let request_sectors = &sectors[off..off + usize::from(len)];
                    l1.begin_lookup_batch_for_client_with_sectors(
                        L1LookupClient::Unknown,
                        request_sectors,
                    );
                    let mut ready = needs_ready
                        .then(|| {
                            checked_cycle_add("L1 hit-pipeline floor", t, params.l1_hit_latency)
                        })
                        .transpose()?;
                    for &sector in &sectors[off..off + usize::from(len)] {
                        if let Some(current) = &mut ready {
                            let (_, sector_ready) = l1.access_eligible(origin, sector, t)?;
                            *current = (*current).max(sector_ready);
                        } else {
                            l1.access_eligible_unobserved(origin, sector, t)?;
                        }
                    }
                    Ok(ready)
                }
                None => {
                    if needs_ready {
                        checked_cycle_add("flat global visibility", t, params.global_flat_latency)
                            .map(Some)
                    } else {
                        Ok(None)
                    }
                }
            }
        }
        AccessKind::GlobalBypass { len, token, .. } => match l1 {
            Some(l1) => {
                l1.count_bypass(
                    MemoryOrigin::Global,
                    u64::from(len),
                    token == CacheOpToken::Atom,
                )?;
                if needs_ready {
                    params.store_visibility_ready(t).map(Some)
                } else {
                    Ok(None)
                }
            }
            None => {
                if needs_ready {
                    checked_cycle_add("flat global visibility", t, params.global_flat_latency)
                        .map(Some)
                } else {
                    Ok(None)
                }
            }
        },
        AccessKind::GlobalAsyncCopy {
            off,
            len,
            l1_policy,
            ..
        } => match (l1, l1_policy) {
            (Some(l1), LdgstsL1Policy::CacheAtAllLevels) => {
                let request_sectors = &sectors[off..off + usize::from(len)];
                l1.begin_async_copy_lookup_batch_with_sectors(request_sectors);
                let mut ready = needs_ready
                    .then(|| {
                        checked_cycle_add("LDGSTS L1 hit-pipeline floor", t, params.l1_hit_latency)
                    })
                    .transpose()?;
                for &sector in &sectors[off..off + usize::from(len)] {
                    if let Some(current) = &mut ready {
                        let (_, sector_ready) = l1.access_eligible_with_mshr_floor(
                            MemoryOrigin::Global,
                            sector,
                            t,
                            params.ldgsts_l1_mshr_floor,
                        )?;
                        *current = (*current).max(sector_ready);
                    } else {
                        l1.access_eligible_unobserved_with_mshr_floor(
                            MemoryOrigin::Global,
                            sector,
                            t,
                            params.ldgsts_l1_mshr_floor,
                        )?;
                    }
                }
                Ok(ready)
            }
            (Some(l1), LdgstsL1Policy::BypassL1) => {
                l1.count_bypass(MemoryOrigin::Global, u64::from(len), false)?;
                if needs_ready {
                    params.store_visibility_ready(t).map(Some)
                } else {
                    Ok(None)
                }
            }
            (None, _) => {
                if needs_ready {
                    checked_cycle_add("flat global visibility", t, params.global_flat_latency)
                        .map(Some)
                } else {
                    Ok(None)
                }
            }
        },
        AccessKind::GlobalUntagged { len, .. } => match l1 {
            Some(l1) => {
                l1.count_untagged(MemoryOrigin::Global, u64::from(len))?;
                if needs_ready {
                    params.store_visibility_ready(t).map(Some)
                } else {
                    Ok(None)
                }
            }
            None => {
                if needs_ready {
                    checked_cycle_add("flat global visibility", t, params.global_flat_latency)
                        .map(Some)
                } else {
                    Ok(None)
                }
            }
        },
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
            match l1 {
                Some(l1) => {
                    if let Some(hits) = projected_l1_write_lookup_hits(kind, sectors) {
                        l1.count_write_lookup_hits(origin, hits)?;
                    }
                    for &sector in &sectors[off..off + usize::from(len)] {
                        if reduction {
                            l1.probe_reduction(origin, sector, t)?;
                        } else {
                            l1.probe_store(origin, sector, t)?;
                        }
                    }
                    if needs_ready {
                        params.store_visibility_ready(t).map(Some)
                    } else {
                        Ok(None)
                    }
                }
                None => {
                    if needs_ready {
                        checked_cycle_add("flat global visibility", t, params.global_flat_latency)
                            .map(Some)
                    } else {
                        Ok(None)
                    }
                }
            }
        }
    }
}

/// Completion identity carried by an eligible L1 access. Ordinary loads only
/// observe a dependence barrier; cacheable asynchronous-copy sources also retain
/// their open group while optionally preserving an ordinary barrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FillTarget {
    Barrier(DepBarrier),
    AsyncCopy {
        group: AsyncCopyGroupId,
        barrier: Option<DepBarrier>,
    },
}

impl FillTarget {
    const fn barrier(self) -> Option<DepBarrier> {
        match self {
            Self::Barrier(barrier) => Some(barrier),
            Self::AsyncCopy { barrier, .. } => barrier,
        }
    }

    const fn async_group(self) -> Option<AsyncCopyGroupId> {
        match self {
            Self::Barrier(_) => None,
            Self::AsyncCopy { group, .. } => Some(group),
        }
    }
}

/// An MSHR-stalled remainder of one instruction's sector list, retried
/// (in FIFO order) whenever a fill completion frees an entry.
struct Retry {
    warp: usize,
    instruction: usize,
    origin: MemoryOrigin,
    /// Whether the stalled logical load uses the SM86 constant-cache return
    /// path. Retries retain the marker so a later fill cannot silently lose
    /// the response delay.
    constant_load: bool,
    /// Whether this constant load uses the measured multi-site operand path.
    /// This remains separate from the broad constant marker above.
    constant_deep: bool,
    client_scope: L2ClientScope,
    /// Completion identity of the stalled instruction. A present ordinary
    /// barrier holds one pending replacement unit until the entry drains;
    /// asynchronous-copy source units were counted at original issue and retain
    /// their group identity across every retry.
    target: Option<FillTarget>,
    /// Access-local finite-admission floor retained from original issue. Only
    /// cacheable LDGSTS retries carry one; ordinary hierarchy traffic uses `None`.
    mshr_floor: Option<NonZeroU32>,
    /// Original logical request boundary for pending-fill lookup attribution.
    lookup_batch: L1LookupBatch,
    /// Unresolved slice `warps[warp].sectors[lo..hi]`.
    lo: usize,
    hi: usize,
}

#[derive(Clone, Copy)]
struct FillWaiter {
    warp: usize,
    target: FillTarget,
    units: u32,
    /// Whether this waiter consumes the measured deep constant-return path.
    /// The marker is per waiter because one folded physical fill can serve
    /// ordinary and constant loads with different return-service latencies.
    constant_deep: bool,
}

/// Completion units cleared when one keyed L1 fill completes. Inline for the
/// dominant single-waiter identity and pre-aggregated by `(warp, target)`, so
/// response preflight only rebuilds counts when distinct targets share a fill.
enum Waiters {
    One(FillWaiter),
    Many(Vec<FillWaiter>),
}

impl Waiters {
    fn add(
        &mut self,
        warp: usize,
        target: FillTarget,
        constant_deep: bool,
    ) -> Result<(), IssueError> {
        let new_waiter = FillWaiter {
            warp,
            target,
            units: 1,
            constant_deep,
        };
        match self {
            Self::One(waiter)
                if waiter.warp == warp
                    && waiter.target == target
                    && waiter.constant_deep == constant_deep =>
            {
                waiter.units = checked_u32_counter_add("fill waiter unit", waiter.units, 1)?;
            }
            Self::One(waiter) => {
                *self = Self::Many(vec![*waiter, new_waiter]);
            }
            Self::Many(waiters) => {
                if let Some(waiter) = waiters.iter_mut().find(|waiter| {
                    waiter.warp == warp
                        && waiter.target == target
                        && waiter.constant_deep == constant_deep
                }) {
                    waiter.units = checked_u32_counter_add("fill waiter unit", waiter.units, 1)?;
                } else {
                    waiters.push(new_waiter);
                }
            }
        }
        Ok(())
    }

    fn try_for_each(
        &self,
        mut visit: impl FnMut(FillWaiter) -> Result<(), IssueError>,
    ) -> Result<(), IssueError> {
        match self {
            Self::One(waiter) => visit(*waiter),
            Self::Many(waiters) => {
                for &waiter in waiters {
                    visit(waiter)?;
                }
                Ok(())
            }
        }
    }
}

fn accumulate_waiter_removals(
    pending_removals: &mut HashMap<(usize, DepBarrier), u32>,
    async_removals: &mut HashMap<(usize, AsyncCopyGroupId), u32>,
    waiter: FillWaiter,
) -> Result<(), IssueError> {
    if let Some(barrier) = waiter.target.barrier() {
        let remove = pending_removals.entry((waiter.warp, barrier)).or_default();
        *remove = checked_u32_counter_add("response pending-remove budget", *remove, waiter.units)?;
    }
    if let Some(group) = waiter.target.async_group() {
        let remove = async_removals.entry((waiter.warp, group)).or_default();
        *remove =
            checked_u32_counter_add("response async-copy remove budget", *remove, waiter.units)?;
    }
    Ok(())
}

fn checked_constant_return_time(
    base: u64,
    constant_deep: bool,
    params: &IssueModelParams,
    operation: &'static str,
) -> Result<u64, IssueError> {
    if !constant_deep {
        return Ok(base);
    }
    let extra = params
        .constant_memory_return_service
        .map_or(0, |service| service.additional_cycles);
    base.checked_add(extra).ok_or(IssueError::TimeOverflow {
        operation,
        base,
        delta: extra,
    })
}

fn fill_waiter_ready_time(
    waiter: FillWaiter,
    response_time: u64,
    params: &IssueModelParams,
) -> Result<u64, IssueError> {
    if !waiter.constant_deep || waiter.target.barrier().is_none() {
        return Ok(response_time);
    }
    checked_constant_return_time(
        response_time,
        true,
        params,
        "constant response return latency",
    )
}

fn project_async_copy_removal(
    projected: &mut HashMap<(usize, AsyncCopyGroupId), u32>,
    async_copies: &[AsyncCopySlot],
    warp: usize,
    group: AsyncCopyGroupId,
    remove: u32,
    now: u64,
    params: &IssueModelParams,
) -> Result<(), IssueError> {
    if warp >= async_copies.len() {
        return Err(IssueError::ResponseWarpOutOfRange {
            warp,
            warps: async_copies.len(),
        });
    }
    let current = if let Some(&current) = projected.get(&(warp, group)) {
        current
    } else {
        async_copies[warp]
            .preflight_async_copy_source_completion(warp, group, None, 0, now, params)?;
        async_copies[warp]
            .async_group(group)
            .ok_or(IssueError::AsyncCopyGroupNotFound { warp, group })?
            .pending_sources
    };
    let next = current
        .checked_sub(remove)
        .ok_or(IssueError::AsyncCopyPendingUnderflow {
            warp,
            group,
            current,
            remove,
        })?;
    projected.insert((warp, group), next);
    Ok(())
}

/// State that EXISTS only in the shared-L2 mode — carrying it inside
/// the [`MemModel`] variant makes a mispaired construction
/// unrepresentable instead of an `expect()` deep in the memory step.
struct SharedL2State {
    l1: L1Cache,
    /// Monotonic tie-breaker for the per-SM visibility stream. Memory cycle
    /// timestamps alone cannot order equal-cycle fences and accesses.
    l1_visibility_sequence: u64,
    /// Complete global eligible L1 observations, including resident hits.
    /// The native system currently drains this only through the explicit
    /// opt-in callback API; the legacy L2-only path remains unchanged.
    l1_visibility_observations: Vec<sim_mem::L1AccessObservation>,
    l1_fence_observations: Vec<sim_mem::L1FenceObservation>,
    l1_fill_visibility_observations: Vec<sim_mem::L1FillVisibleObservation>,
    l1_invalidate_all_observations: Vec<sim_mem::L1InvalidateAllObservation>,
    l1_visibility_requester: Option<u8>,
    /// Only payload-event sectors are tracked; default replay allocates no
    /// value-observation map. Every later store to a watched sector replaces
    /// its writer, including stores with no corresponding payload event.
    payload_store_writers: Option<Box<BTreeMap<(i64, u8), Option<PayloadStoreObservation>>>>,
    response_owner: ResponseOwner,
    /// Exact response-bearing request tokens still expected from the system.
    outstanding_replies: HashSet<ReplyTo>,
    /// Response identities indexed separately so admission checks do not scan
    /// every outstanding token for each sector in a wide instruction.
    outstanding_response_ids: HashMap<ResponseId, ReplyTo>,
    /// Exact sector mask carried by each outstanding response-bearing request.
    /// This stays separate from `fill_masks`, which is the physical L2
    /// exactly-once traffic mask and may include a later child fill.
    response_masks: HashMap<ResponseId, u8>,
    /// Barrier units cleared when the keyed L1 fill completes.
    fill_waiters: HashMap<FillKey, Waiters>,
    /// Per in-flight LINE fill: the sector bits already sent to the L2
    /// (the fill request's mask plus any follow-up fetches). A merge
    /// onto a sector NOT yet in the mask sends a follow-up sector
    /// fetch with its own L1 sector response; a re-touch of a sent
    /// sector sends nothing, like the hardware MSHR. Entries die
    /// with the parent line response. Exactly-once is PER FILL, not
    /// per-address-epoch: if a resident line with an in-flight SECTOR
    /// fill is evicted and re-allocated, the new fill may re-fetch a
    /// sector whose zombie fetch is still outstanding (rare
    /// eviction-window corner, documented at the 2026-07-07 gate).
    fill_masks: HashMap<i64, u8>,
    /// Line fills emitted during the CURRENT timestamped emission batch:
    /// `(line, issue_time) → first index in out`. A merge onto one of these
    /// folds its sector into the fill request's mask. Requests for the same
    /// line at a later local cycle receive an independent sector token; a
    /// whole `advance` call is not a same-cycle packet boundary.
    /// Cleared at every batch entry; indices are only valid because
    /// each batch starts with an empty `out`.
    line_folds: HashMap<(i64, u64), usize>,
    retry: VecDeque<Retry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PayloadStoreSite {
    pub launch_index: i64,
    pub warp_slot: i32,
    pub instruction_index: usize,
}

#[derive(Debug, Clone, Copy)]
struct PayloadStoreObservation {
    site: PayloadStoreSite,
    accepted_at: u64,
    resident_generation: Option<u32>,
    retained_dirty: bool,
    attached: bool,
}

impl SharedL2State {
    fn next_l1_visibility_sequence(&mut self) -> Result<u64, IssueError> {
        let sequence = self.l1_visibility_sequence;
        self.l1_visibility_sequence =
            sequence.checked_add(1).ok_or(IssueError::CounterOverflow {
                counter: "L1 visibility sequence",
            })?;
        Ok(sequence)
    }
}

/// The typed memory model an [`SmCore`] runs under (the public
/// [`SmMemMode`] resolved into owned state).
enum MemModel {
    Flat,
    L1Const(L1Cache),
    L1Shared(SharedL2State),
}

impl MemModel {
    fn l1(&self) -> Option<&L1Cache> {
        match self {
            MemModel::Flat => None,
            MemModel::L1Const(l1) => Some(l1),
            MemModel::L1Shared(st) => Some(&st.l1),
        }
    }

    /// The L1 the CONSTANT-latency data path drives (`None` in Flat and
    /// Shared modes — the shared path never uses `data_ready_time`).
    fn const_l1_mut(&mut self) -> Option<&mut L1Cache> {
        match self {
            MemModel::L1Const(l1) => Some(l1),
            _ => None,
        }
    }

    fn shared_l1_mut(&mut self) -> Option<&mut SharedL2State> {
        match self {
            MemModel::L1Shared(state) => Some(state),
            _ => None,
        }
    }
}

/// Extract the constant-bank line keys used by an instruction's SASS text.
/// Immediate constants are operands of the form `c[bank][offset]` or
/// `cx[bank][offset]`; indexed `LDC`/`LDC.*` operations use a different cache
/// and are intentionally excluded here, while uniform `ULDC`/`ULDC.*`
/// operands remain on the IMC path as observed by NCU.  The parser is deliberately narrow:
/// malformed/non-numeric references are ignored rather than guessed, while
/// the trusted static sidecar remains the source of the instruction itself.
fn immediate_constant_lines(opcode_text: &str, line_bytes: u32) -> Box<[u64]> {
    // `opcode_head` strips width/cache modifiers (for example `ULDC.64`)
    // using the same grammar as the rest of the SASS classifier.  On SM86,
    // the remodeled scheduler probes CBANK operands on both ordinary ALU
    // instructions and uniform `ULDC` forms; NCU's IMC request counter also
    // includes those uniform references.  Only the true indexed `LDC` load
    // uses the separate constant-memory path, so exclude that head while
    // retaining ULDC/ULDC.64 CBANK operands.
    if opcode_head(opcode_text) == "LDC" || line_bytes == 0 {
        return Box::new([]);
    }
    let bytes = opcode_text.as_bytes();
    let mut lines = Vec::new();
    let mut cursor = 0usize;
    while cursor + 3 < bytes.len() {
        let is_c = bytes[cursor] == b'c'
            && (bytes[cursor + 1] == b'['
                || (bytes[cursor + 1] == b'x'
                    && cursor + 2 < bytes.len()
                    && bytes[cursor + 2] == b'['));
        if !is_c {
            cursor += 1;
            continue;
        }
        let first_open = if bytes[cursor + 1] == b'[' {
            cursor + 1
        } else {
            cursor + 2
        };
        let Some(first_close) = bytes[first_open + 1..]
            .iter()
            .position(|&value| value == b']')
            .map(|offset| first_open + 1 + offset)
        else {
            cursor += 1;
            continue;
        };
        let second_open = first_close + 1;
        if second_open >= bytes.len() || bytes[second_open] != b'[' {
            cursor = first_close + 1;
            continue;
        }
        let Some(second_close) = bytes[second_open + 1..]
            .iter()
            .position(|&value| value == b']')
            .map(|offset| second_open + 1 + offset)
        else {
            cursor = second_open + 1;
            continue;
        };
        let parse_literal = |slice: &[u8]| -> Option<u64> {
            let text = std::str::from_utf8(slice).ok()?.trim();
            if text.is_empty() {
                return None;
            }
            if let Some(hex) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
                u64::from_str_radix(hex, 16).ok()
            } else {
                text.parse().ok()
            }
        };
        let bank = parse_literal(&bytes[first_open + 1..first_close]);
        let offset = parse_literal(&bytes[second_open + 1..second_close]);
        if let (Some(bank), Some(offset)) = (bank, offset) {
            let address = bank
                .checked_mul(65_536)
                .and_then(|base| base.checked_add(offset));
            if let Some(address) = address {
                let line = address / u64::from(line_bytes);
                if !lines.contains(&line) {
                    lines.push(line);
                }
            }
        }
        cursor = second_close + 1;
    }
    lines.sort_unstable();
    lines.into_boxed_slice()
}

#[derive(Debug, Clone, Copy)]
struct ImmediateConstantPending {
    line: u64,
    ready_at: u64,
}

#[derive(Debug, Clone)]
struct ImmediateConstantCacheState {
    config: ImmediateConstantCache,
    /// Per-set LRU tags, oldest at the front.  A line key is used as its tag
    /// identity after the set index has been derived; this is sufficient for
    /// the closed constant-address domain and avoids address aliases.
    sets: Vec<VecDeque<u64>>,
    pending: Vec<ImmediateConstantPending>,
    /// The pipelined SM-shared L1C access path admits lines at its measured
    /// initiation interval while the MSHR count bounds lines outstanding.
    next_service: u64,
    diagnostics: ImmediateConstantCacheDiagnostics,
}

impl ImmediateConstantCacheState {
    fn new(config: ImmediateConstantCache) -> Result<Self, IssueError> {
        if config.sets == 0
            || config.line_bytes == 0
            || config.ways == 0
            || config.mshr_entries == 0
            || config.access_queue_entries == 0
            || config.miss_initiation_interval_cycles == 0
            || config.sm_structure_latency_cycles == 0
            || config.l0_to_l1_cycles == 0
            || config.switch_after_miss_cycles == 0
        {
            return Err(IssueError::ImmediateConstantCacheConfigurationInvalid);
        }
        Ok(Self {
            config,
            sets: (0..config.sets).map(|_| VecDeque::new()).collect(),
            pending: Vec::new(),
            next_service: 0,
            diagnostics: ImmediateConstantCacheDiagnostics::default(),
        })
    }

    fn set_index(&self, line: u64) -> usize {
        (line % u64::from(self.config.sets)) as usize
    }

    fn contains(&self, line: u64) -> bool {
        self.sets[self.set_index(line)].contains(&line)
    }

    fn pending_ready(&self, line: u64) -> Option<u64> {
        self.pending
            .iter()
            .find(|entry| entry.line == line)
            .map(|entry| entry.ready_at)
    }

    fn touch(&mut self, line: u64) {
        let set_index = self.set_index(line);
        let set = &mut self.sets[set_index];
        if let Some(position) = set.iter().position(|&tag| tag == line) {
            set.remove(position);
        }
        set.push_back(line);
        while set.len() > self.config.ways as usize {
            set.pop_front();
        }
    }

    fn complete_ready(&mut self, now: u64) {
        let mut ready = Vec::new();
        self.pending.retain(|entry| {
            if entry.ready_at <= now {
                ready.push(entry.line);
                false
            } else {
                true
            }
        });
        // `pending` is issue order. Preserve it when multiple lines have
        // become ready since the last scheduler step: sorting by line address
        // would change same-timestamp fill order and therefore LRU victims.
        // Pending-line identity is unique by `reserve`, so no deduplication is
        // needed here.
        // Diagnostic counters are observational; the execution state itself
        // remains checked. Saturation keeps a pathological trace from turning
        // reporting into a scheduler failure.
        self.diagnostics.lines_filled = self
            .diagnostics
            .lines_filled
            .saturating_add(u64::try_from(ready.len()).unwrap_or(u64::MAX));
        for line in ready {
            self.touch(line);
        }
    }

    /// Reserve every currently missing line referenced by one instruction.
    /// Returns the earliest time at which the instruction may be re-issued;
    /// `None` means all references are already resident.  New misses use one
    /// pipelined L1C service lane; an existing pending line is shared by all
    /// waiting warps on that SMSP.
    fn reserve(
        &mut self,
        lines: &[u64],
        now: u64,
        upper_level_latency: u64,
    ) -> Result<Option<u64>, IssueError> {
        let mut wait_until: Option<u64> = None;
        // `upper_level_latency` is the carded L1/L2 response frontier used by
        // the shared event system.  The 11-stage local structure and its
        // 39-cycle transport are already represented in the NCU-calibrated
        // response frontier for this replay; adding them here would charge
        // the same path twice.  The local fields remain part of the capability
        // identity and diagnostics, while this actor waits on the single
        // externally observable response timestamp.
        let miss_latency = upper_level_latency;
        for &line in lines {
            self.diagnostics.references = self.diagnostics.references.saturating_add(1);
            if self.contains(line) {
                self.diagnostics.lookup_hits = self.diagnostics.lookup_hits.saturating_add(1);
                continue;
            }
            self.diagnostics.lookup_misses = self.diagnostics.lookup_misses.saturating_add(1);
            if let Some(ready) = self.pending_ready(line) {
                self.diagnostics.pending_tag_hits =
                    self.diagnostics.pending_tag_hits.saturating_add(1);
                self.diagnostics.pending_merges = self.diagnostics.pending_merges.saturating_add(1);
                wait_until = Some(wait_until.map_or(ready, |current| current.max(ready)));
                continue;
            }
            if self.pending.len() >= self.config.mshr_entries as usize {
                if let Some(ready) = self.pending.iter().map(|entry| entry.ready_at).min() {
                    wait_until = Some(wait_until.map_or(ready, |current| current.max(ready)));
                }
                continue;
            }
            let start = now.max(self.next_service);
            let ready = start
                .checked_add(miss_latency)
                .ok_or(IssueError::TimeOverflow {
                    operation: "immediate constant-cache line response",
                    base: start,
                    delta: miss_latency,
                })?;
            // The 11-stage L1C structure is a pipelined queue: its one-entry
            // access port accepts a new line each cycle while earlier misses
            // continue through the latency stages.  Serializing the port by
            // the full response latency would turn independent CBANK misses
            // into a bandwidth bottleneck that the released SM86 model does
            // not have.  Keep the response timestamp serialized only by the
            // measured initiation interval (one cycle for the characterized
            // queue), while `ready_at` still carries the full miss latency.
            self.next_service = start
                .checked_add(self.config.miss_initiation_interval_cycles)
                .ok_or(IssueError::TimeOverflow {
                    operation: "immediate constant-cache service initiation",
                    base: start,
                    delta: self.config.miss_initiation_interval_cycles,
                })?;
            self.pending.push(ImmediateConstantPending {
                line,
                ready_at: ready,
            });
            self.diagnostics.tag_misses = self.diagnostics.tag_misses.saturating_add(1);
            self.diagnostics.maximum_pending_lines = self
                .diagnostics
                .maximum_pending_lines
                .max(u64::try_from(self.pending.len()).unwrap_or(u64::MAX));
            wait_until = Some(wait_until.map_or(ready, |current| current.max(ready)));
        }
        Ok(wait_until)
    }

    fn diagnostics(&self) -> ImmediateConstantCacheDiagnostics {
        self.diagnostics
    }
}

#[derive(Debug)]
struct ImmediateConstantPlan {
    picks: [Option<usize>; SUB_CORES],
    cache: Option<Box<[ImmediateConstantCacheState; SUB_CORES]>>,
    waits: [Option<(usize, u64)>; SUB_CORES],
    /// Additional non-greedy candidates that were probed and parked while
    /// scanning past an IMC miss in the same scheduler opportunity.
    extra_waits: Vec<(usize, u64)>,
    miss_observed: bool,
}

/// Resumable single-SM replay actor. See the module docs for the mode
/// semantics and the run-ahead (horizon) contract.
#[derive(Debug)]
struct SharedWriteLookupRuntime {
    observer: Option<Rc<RefCell<L1WriteLookupObserver>>>,
    /// Per-SMSP source-hit return-port frontiers for cacheable LDGSTS.
    async_l1_hit_service_ready: [u64; SUB_CORES],
}

#[cfg(feature = "reference")]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct AdvanceDiagnostics {
    pub calls: u64,
    pub local_cycle_steps: u64,
    pub idle_jumps: u64,
    pub skipped_cycles: u64,
}

#[cfg(feature = "reference")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReferenceScalarMacroExecution {
    pub(crate) warp_slot: i32,
    pub(crate) first_issue: u64,
    pub(crate) last_issue: u64,
    pub(crate) end_cycle: u64,
    pub(crate) instruction_count: u64,
}

pub(crate) struct SmCore<'a> {
    schedule: &'a FunctionSchedule,
    /// Static schedule values in PC order. Contiguous binary search avoids a
    /// pointer-chasing B-tree lookup for every dynamic issue while preserving
    /// the schedule map as the authoritative join/diagnostic surface.
    schedule_slots: Vec<(u64, &'a IssueSlot)>,
    /// Static PC-indexed memory timing table. It is sorted once at admission so
    /// the hot issue loop uses a branch-predictable binary search instead of a
    /// randomized hash lookup for every resident warp.
    memory_pipeline_schedule: Box<[(u64, MemoryPipelineSlot)]>,
    /// Static PCs that enter the measured multi-site constant operand path.
    /// Sorted static PC set; issue-cycle membership is binary-searched to
    /// avoid randomized hashing on every resident-warp readiness check.
    constant_path_pcs: Box<[u64]>,
    /// Pre-parsed immediate-constant line references keyed by static PC.
    /// Sorted static PC -> immediate-constant line table.
    immediate_constant_schedule: Box<[(u32, Box<[u64]>)]>,
    /// Private IMC/L0C state for each SMSP; absent for compact replay inputs.
    immediate_constant_cache: Option<Box<[ImmediateConstantCacheState; SUB_CORES]>>,
    /// Remaining Huerta SM86 L0C miss-switch opportunities for each SMSP.
    /// A successful issue reloads this bounded countdown; while the held
    /// (greedy) warp is otherwise ready but its immediate-constant operand is
    /// not, the scheduler keeps it selected until the countdown expires.
    /// This is intentionally separate from `imc_wait_until`, which is the
    /// response frontier for the specific line request.
    immediate_constant_switch_remaining: [u64; SUB_CORES],
    params: IssueModelParams,
    shared_memory_residency: SharedMemoryResidency,
    physical_warp_slots: u32,
    instruction_front_end: Option<InstructionFrontEndState>,
    memory_issue_pipeline: Option<MemoryIssuePipelineState>,
    mem: MemModel,
    blocks: &'a [BlockReplay],
    block_live_warps: Vec<u32>,
    instruction_limit: u64,
    warps: Vec<WarpState<'a>>,
    /// Cold protocol storage parallel to `warps`; entries allocate only after
    /// an effective asynchronous-copy source is issued.
    async_copies: Vec<AsyncCopySlot>,
    /// Construction-time specialization flag. Kernels without async-copy
    /// sources or controls bypass the protocol path once per scheduler batch,
    /// not once per issued instruction or resident-warp scan.
    has_async_copy_protocol: bool,
    block_states: Vec<BlockState>,
    subcore_warps: [Vec<usize>; SUB_CORES],
    hold: [Option<usize>; SUB_CORES],
    /// Fetch/decode continuation is distinct from issue hold. The SM86
    /// control bit Yield drops issue hold for the next cycle, but Huerta's
    /// fetch/decode model continues the warp issued in the previous cycle
    /// while its instruction buffer has capacity.
    fetch_last_issued: [Option<usize>; SUB_CORES],
    /// Architecture-local CTA-barrier issue resource present when the joined
    /// function contains a characterized barrier instruction.
    barrier_issue: Option<CtaBarrierIssue>,
    /// Earliest cycle at which the one SM-shared barrier server can accept
    /// another warp arrival.
    barrier_pipe_ready: u64,
    /// Earliest issue time of each SMSP's local Tensor Core. This timeline is
    /// kernel-local but survives CTA retirement/admission waves on the SM.
    tensor_pipe_ready: [u64; SUB_CORES],
    /// Independent per-SMSP uniform/SFU initiation timeline.
    sfu_pipe_ready: [u64; SUB_CORES],
    /// Finite scalar INT/FP execution windows, one per SMSP.
    scalar_execution_pipeline: Option<Box<ScalarExecutionPipelineState>>,
    /// Full-mode SM86 effective register-read envelope.  Ports are per SMSP
    /// and bank; the reuse set is compiler-keyed and intentionally not a
    /// physical register-file claim.
    sm86_rf_enabled: bool,
    sm86_rf_port_ready: [[u64; 2]; SUB_CORES],
    sm86_rfc: [VecDeque<Sm86RfcKey>; SUB_CORES],
    /// Cold shared observer/async-return state is boxed to keep the actor's
    /// compact hot structure within the established 4 KiB bound.
    shared_write_lookup_runtime: Box<SharedWriteLookupRuntime>,
    /// Earliest free cycle of the cache-all LDGSTS destination service shared
    /// by every resident warp on this SM.
    ldgsts_destination_pipe_ready: u64,
    /// Pending barrier-observed L1-hit result writebacks. Allocated only when
    /// one joined function contains both the admitted HADD2 producer and the
    /// admitted sm_86 STRONG.SM scalar load form.
    result_writebacks: Option<ResultWritebackState>,
    unmodeled_tails: TailFlags,
    /// Box the cold diagnostic aggregate so adding a new observation surface
    /// cannot inflate the scheduler's hot actor beyond its compact-size
    /// invariant.
    readiness: Box<ReadinessDiagnostics>,
    memory_service_candidates: MemoryServiceCandidateState,
    emit_admission_witnesses: bool,
    memory_admission_feedback: Option<Box<MemoryAdmissionFeedbackState>>,
    /// Effective picks from the previous transaction in the current logical
    /// cycle. Distinguishes dispatch progress from a parked IMC miss.
    last_issue_picks: [Option<usize>; SUB_CORES],
    instructions_per_smsp: [u64; SUB_CORES],
    issue_active_cycles_per_smsp: [u64; SUB_CORES],
    last_observed_issue: [Option<u64>; SUB_CORES],
    warp_lifetimes: Vec<WarpLifetime>,
    issued: u64,
    t: u64,
    /// Latest modeled SM-local completion frontier that can occur after the
    /// last issue (for example, a terminal CTA-barrier release).
    local_completion_time: Option<u64>,
    /// P15 logical route-service availability.  Full replay only; fast
    /// diagnostics never construct or consume this state.
    sm120_flat_service_enabled: bool,
    sm120_flat_global_service_ready: u64,
    sm120_flat_shared_service_ready: u64,
    sm120_flat_multiwarp_service_enabled: bool,
    sm120_flat_multiwarp_global_service_ready: u64,
    sm120_flat_multiwarp_dense: bool,
    sm120_flat_multiwarp_medium: bool,
    sm120_flat_shared_medium: bool,
    sm120_flat_async_dense: bool,
    // Logical recent-source window, not a physical cache geometry.
    sm120_async_sources: VecDeque<(i64, u64)>,
    sm120_flat_async_service_enabled: bool,
    sm120_flat_async_service_ready: u64,
    sm120_flat_async_depth_service_per_group: u64,
    sm120_flat_async_depth_service_ready: u64,
    sm120_flat_store_retire_enabled: bool,
    sm120_flat_store_retire_ready: u64,
    arrival_stamp: u64,
    resident_warps: u32,
    resident_shared_memory_bytes: u64,
    resident_warp_slots: Box<HashSet<i32>>,
    unfinished_warps: usize,
    next_block: usize,
    /// Test/reference-only switch. The production default retains next-event
    /// advancement; the reference lane walks locally idle cycles one at a
    /// time so differential runs can separate clock jumps from state changes.
    dense_reference: bool,
    #[cfg(feature = "reference")]
    advance_diagnostics: AdvanceDiagnostics,
    #[cfg(feature = "reference")]
    reference_issue_effects: Vec<(u64, u64, u32)>,
}

#[cfg(feature = "reference")]
fn reference_mode_from_env() -> Result<bool, IssueError> {
    match std::env::var_os("SIM_SM_REFERENCE_MODE") {
        None => Ok(false),
        Some(mode) if mode == "event" => Ok(false),
        Some(mode) if mode == "dense" => Ok(true),
        Some(mode) => Err(IssueError::ReferenceModeInvalid {
            mode: mode.to_string_lossy().into_owned(),
        }),
    }
}

