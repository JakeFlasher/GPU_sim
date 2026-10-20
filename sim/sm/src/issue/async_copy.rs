//! Async-copy cohorts and destination readiness; mutates SMSP timing and completion state.

use super::*;

/// One cold allocation per warp, indexed sparsely by dynamic instruction.
#[derive(Debug, Default)]
pub(crate) struct AsyncCopyCohortReplay {
    pub(crate) spans: Vec<AsyncCopyCohortSpan>,
    pub(crate) cohorts: Vec<AsyncCopyCohortDescriptor>,
    pub(crate) sector_membership_masks: Vec<u32>,
    /// Sparse selected-instruction callback/channel order. This shares the
    /// cold async sidecar allocation but is never consumed by issue timing.
    pub(crate) instrumentation_order: Vec<InstrumentationOrderAnchor>,
}

impl AsyncCopyCohortReplay {
    pub(super) fn view(&self, instruction: usize) -> Option<AsyncCopyCohortView<'_>> {
        let span = self
            .spans
            .binary_search_by_key(&instruction, |span| span.instruction)
            .ok()
            .map(|index| self.spans[index])?;
        let cohort_end = span.cohort_off.checked_add(usize::from(span.cohort_len))?;
        let membership_end = span
            .membership_off
            .checked_add(usize::from(span.sector_len))?;
        Some(AsyncCopyCohortView {
            sector_off: span.sector_off,
            sector_len: span.sector_len,
            cohorts: self.cohorts.get(span.cohort_off..cohort_end)?,
            sector_membership_masks: self
                .sector_membership_masks
                .get(span.membership_off..membership_end)?,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AsyncCopyGroup {
    pub(super) id: AsyncCopyGroupId,
    /// Issue time of the `LDGDEPBAR` that made this group visible to a
    /// subsequent wait. Zero is also the valid initial-cycle timestamp; the
    /// group cannot be observed here until it is committed.
    pub(super) committed_at: u64,
    /// Source sectors whose shared-L2 response time is not known yet.
    pub(super) pending_sources: u32,
    /// Maximum known lower-level source completion time across flat/constant
    /// copies and shared-L2 responses.
    pub(super) source_completion_time: u64,
    /// Route-weighted effective payload in the architecture capability's common
    /// fixed-point work domain. This is added once per issued copy, never inferred
    /// from or duplicated across its source sectors.
    pub(super) delivery_work: u64,
    /// Exact cache-all destination wavefronts accumulated across this committed
    /// group. Zero retains the payload-only/bypass path.
    pub(super) destination_wavefronts: u64,
    /// The aggregate SM service is assigned once, after commit and after every
    /// source unit has a known completion time.
    pub(super) destination_service_queued: bool,
    /// Latest completion assigned by the aggregate per-SM destination service.
    pub(super) destination_completion_time: u64,
    /// Maximum of the warp-local dependent-completion floor and the aggregate
    /// destination-service completion. Every mutation checks this value before
    /// publishing group state, leaving scheduler readiness infallible.
    pub(super) completion_time: u64,
}

impl AsyncCopyGroup {
    pub(super) const fn new(id: AsyncCopyGroupId) -> Self {
        Self {
            id,
            committed_at: 0,
            pending_sources: 0,
            source_completion_time: 0,
            delivery_work: 0,
            destination_wavefronts: 0,
            destination_service_queued: false,
            destination_completion_time: 0,
            completion_time: 0,
        }
    }

    pub(super) const fn is_complete_at(self, now: u64) -> bool {
        self.pending_sources == 0
            && (self.destination_wavefronts == 0 || self.destination_service_queued)
            && self.completion_time <= now
    }

    pub(super) fn project_delivery(
        self,
        source_ready: u64,
        work_add: u64,
        destination_wavefronts_add: u64,
        params: &IssueModelParams,
    ) -> Result<(u64, u64, u64, u64), IssueError> {
        let delivery_work = checked_u64_counter_add(
            "asynchronous-copy delivery work",
            self.delivery_work,
            work_add,
        )?;
        let source_completion_time = self.source_completion_time.max(source_ready);
        let destination_wavefronts = checked_u64_counter_add(
            "asynchronous-copy destination wavefront",
            self.destination_wavefronts,
            destination_wavefronts_add,
        )?;
        let local_completion_time =
            params.ldgsts_completion_ready(source_completion_time, delivery_work)?;
        let completion_time = local_completion_time.max(self.destination_completion_time);
        Ok((
            source_completion_time,
            delivery_work,
            destination_wavefronts,
            completion_time,
        ))
    }

    pub(super) fn local_completion_time(
        self,
        params: &IssueModelParams,
    ) -> Result<u64, IssueError> {
        params.ldgsts_completion_ready(self.source_completion_time, self.delivery_work)
    }

    pub(super) fn limiting_readiness_cause(
        self,
        params: &IssueModelParams,
    ) -> Result<ReadinessCause, IssueError> {
        let local = self.local_completion_time(params)?;
        Ok(if self.destination_completion_time > local {
            ReadinessCause::AsyncDestinationSmService
        } else if self.delivery_work != 0 || self.destination_wavefronts != 0 {
            ReadinessCause::AsyncDestinationLocal
        } else {
            ReadinessCause::AsyncSource
        })
    }
}

/// Allocated only for warps that issue an effective asynchronous-copy source.
/// Keeping this uncommon state out of every [`WarpState`] preserves the compact
/// scheduler scan used by workloads with no `LDGSTS` traffic.
#[derive(Debug)]
pub(super) struct AsyncCopyState {
    /// Non-bulk `cp.async` copies issued since the most recent commit.
    pub(super) open_group: Option<AsyncCopyGroup>,
    /// Nonempty groups closed by `LDGDEPBAR`, in warp program order.
    pub(super) committed_groups: VecDeque<AsyncCopyGroup>,
    /// Bypass-only per-instruction source/delivery state. Entries share the
    /// owning group's opaque identity and are removed with the waited prefix.
    pub(super) bypass_copies: Vec<BypassAsyncCopyObservation>,
    /// Next process-local sequence for this warp. `None` means the complete
    /// u64 domain has been allocated; the final `u64::MAX` identity remains
    /// valid, but another nonempty group fails closed.
    pub(super) next_sequence: Option<u64>,
}

impl AsyncCopyState {
    pub(super) const fn new() -> Self {
        Self {
            open_group: None,
            committed_groups: VecDeque::new(),
            bypass_copies: Vec::new(),
            next_sequence: Some(0),
        }
    }
}

/// Cold sidecar indexed by warp identity. Keeping even the nullable pointer out
/// of [`WarpState`] preserves that structure's pre-async stride for scheduler
/// scans while retaining direct, deterministic lookup for source responses.
#[derive(Debug, Default)]
pub(super) struct AsyncCopySlot {
    pub(super) state: Option<Box<AsyncCopyState>>,
}

impl AsyncCopySlot {
    pub(super) fn preflight_async_copy_add(
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

    pub(super) fn ensure_open_async_group(
        &mut self,
        warp: usize,
    ) -> Result<&mut AsyncCopyGroup, IssueError> {
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

    pub(super) fn add_known_async_copy(
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

    pub(super) fn add_pending_async_copy_sources(
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

    pub(super) fn async_group(&self, id: AsyncCopyGroupId) -> Option<&AsyncCopyGroup> {
        let state = self.state.as_deref()?;
        state
            .open_group
            .as_ref()
            .filter(|group| group.id == id)
            .or_else(|| state.committed_groups.iter().find(|group| group.id == id))
    }

    pub(super) fn async_group_mut(&mut self, id: AsyncCopyGroupId) -> Option<&mut AsyncCopyGroup> {
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

    pub(super) fn preflight_async_copy_source_completion(
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

    pub(super) fn complete_async_copy_sources(
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

    pub(super) fn bypass_copy_projection(
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

    pub(super) fn commit_async_copy_group(&mut self, now: u64) -> bool {
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

    pub(super) fn async_wait_prefix_len(&self, max_pending_groups: u32) -> usize {
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
    pub(super) fn async_wait_clear_time(&self, max_pending_groups: u32) -> Option<u64> {
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

    pub(super) fn consume_async_wait(&mut self, max_pending_groups: u32, now: u64) {
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
