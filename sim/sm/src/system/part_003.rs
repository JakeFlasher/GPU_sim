// Lexical implementation section 4/4; included by the parent facade.
#[allow(clippy::too_many_arguments)]
fn schedule_emitted_requests(
    sm: usize,
    target: TargetId,
    queue: &mut EventQueue<Ev>,
    pending_bounds: &mut PendingBounds,
    scratch: &mut Vec<MemRequest>,
    l2_hit_latency: u64,
    batch_replyless_writes: bool,
    memory_admission_feedback: Option<MemoryAdmissionCandidate>,
) -> Result<(), SystemError> {
    let mut index = 0;
    while index < scratch.len() {
        let request = scratch[index];
        if batch_replyless_writes && batchable_replyless_write(request) {
            let mut end = index + 1;
            while end < scratch.len()
                && batchable_replyless_write(scratch[end])
                && scratch[end].issue_time == request.issue_time
            {
                end += 1;
            }
            if end - index > 1 {
                for batched in &scratch[index..end] {
                    if feedback_prices_request(memory_admission_feedback, *batched) {
                        pending_bounds.insert(sm, batched.issue_time)?;
                    }
                }
                let requests = end - index;
                let sequence_span = NonZeroU64::new(u64::try_from(requests).map_err(|_| {
                    SystemError::CounterOverflow {
                        counter: "L2 access batch sequence span",
                    }
                })?)
                .expect("a batched run contains at least two requests");
                queue.schedule_at_spanning(
                    request.issue_time,
                    target,
                    Ev::L2AccessBatch(scratch[index..end].to_vec().into_boxed_slice()),
                    sequence_span,
                )?;
                index = end;
                continue;
            }
        }
        if request.reply != ReplyTo::None {
            let bound = checked_cycle_add(
                "pending L2-hit response bound",
                request.issue_time,
                l2_hit_latency,
            )?;
            pending_bounds.insert(sm, bound)?;
        }
        // A timing-active write candidate cannot be priced until this future
        // L2 access is processed.  The memory issue pipeline may emit the
        // request several cycles ahead of the actor clock, so retain that
        // access timestamp as a self-affecting bound just like an pending_bounds
        // read response.  Otherwise an unrelated wake can let the SM run past
        // the event that releases its feedback queue or schedules its ack.
        if feedback_prices_request(memory_admission_feedback, request) {
            pending_bounds.insert(sm, request.issue_time)?;
        }
        queue.schedule_at(request.issue_time, target, Ev::L2Access(request))?;
        index += 1;
    }
    scratch.clear();
    Ok(())
}

#[cfg(feature = "reference")]
fn dispatch_reference_macro_event(
    core: &mut SmCore<'_>,
    sm_ix: usize,
    actor_count: usize,
    result: &NativeBoundaryExecutionResult,
    queue: &mut EventQueue<Ev>,
    wake_due: &mut DueTimes,
    power_state: Option<&mut PowerBoundaryState>,
    mut trace: Option<&mut ReferenceTrace>,
) -> Result<bool, SystemError> {
    let contract = match result {
        NativeBoundaryExecutionResult::TypedMacro { contract } => contract,
        NativeBoundaryExecutionResult::ExactReplay { .. } => return Ok(false),
    };
    if contract.validate().is_err() {
        if let Some(trace) = trace.as_deref_mut() {
            trace.note_native_fallback();
        }
        return Ok(false);
    }
    // The scalar actor is allowed to consume only a boundary whose logical
    // effect reservations are immediately after the event that was popped.
    // Otherwise a witness could make a Preserve decision while the real event
    // queue already contains work that would receive an ambiguous sequence.
    let effect_count = u64::try_from(contract.internal_effects.len()).map_err(|_| {
        IssueError::ReferenceMacroUnsupported {
            reason: "typed_macro_effect_count_out_of_range",
        }
    })?;
    let Some(effect_span) = NonZeroU64::new(effect_count) else {
        if let Some(trace) = trace.as_deref_mut() {
            trace.note_native_fallback();
        }
        return Ok(false);
    };
    let expected_effect_start = contract.entry.event.sequence.checked_add(1).ok_or(
        IssueError::ReferenceMacroUnsupported {
            reason: "typed_macro_sequence_overflow",
        },
    )?;
    let expected_exit_sequence = expected_effect_start.checked_add(effect_count).ok_or(
        IssueError::ReferenceMacroUnsupported {
            reason: "typed_macro_sequence_overflow",
        },
    )?;
    if contract.entry.event.payload != TypedEventPayload::SmWake
        || !contract.entry.queue.pending.is_empty()
        || contract.exit.event.sequence != expected_exit_sequence
        || queue.next_sequence() != Some(expected_effect_start)
        || wake_due.next(sm_ix).is_some()
    {
        if let Some(trace) = trace.as_deref_mut() {
            trace.note_native_fallback();
        }
        return Ok(false);
    }
    // Validate and install the actor transition before consuming queue
    // sequence space. The actor performs all descriptor/continuation checks
    // before mutation; an unsupported descriptor therefore falls back without
    // changing either the core or the event queue.
    let native_execution = core.execute_reference_boundary_result(result)?;
    let Some(execution) = native_execution else {
        if let Some(trace) = trace.as_deref_mut() {
            trace.note_native_fallback();
        }
        return Ok(false);
    };
    // Reserve the declared internal effects before the continuation is
    // inserted.  The continuation therefore receives the contract's
    // region-end sequence instead of silently reusing the next free sequence
    // as if the collapsed effects had never existed.
    let reserved_start = queue.reserve_sequence_span(effect_span)?;
    debug_assert_eq!(reserved_start, expected_effect_start);
    if let Some(trace) = trace.as_deref_mut() {
        trace.note_native_dispatch();
    }
    let target = u32::try_from(sm_ix)
        .map_err(|_| SystemError::ActorCountOutOfRange { count: actor_count })?;
    let continuation_seq = queue.schedule_at(execution.end_cycle, TargetId(target), Ev::SmWake)?;
    if continuation_seq != expected_exit_sequence {
        return Err(SystemError::Issue(IssueError::ReferenceMacroUnsupported {
            reason: "typed_macro_continuation_sequence_mismatch",
        }));
    }
    if let Some(power_state) = power_state {
        power_state
            .record_scalar_macro_completion(contract.entry.event.time, execution.end_cycle)?;
    }
    if let Some(trace) = trace.as_deref_mut() {
        trace.expect_region_end(contract.exit.clone());
    }
    wake_due.push(sm_ix, execution.end_cycle)?;
    Ok(true)
}

/// Resume one SM actor: advance to its earliest self-affecting external
/// event — the earlier of its next scheduled response and the lower
/// bound of any request still awaiting L2 pricing — schedule its
/// emitted requests, and re-arm its wake timer if it has one. Returns
/// whether the actor made progress (issued or emitted anything) — the
/// livelock guard's signal.
#[allow(clippy::too_many_arguments)] // driver-internal plumbing, one call shape
fn run_sm(
    ix: usize,
    cores: &mut [SmCore<'_>],
    queue: &mut EventQueue<Ev>,
    response_due: &mut DueTimes,
    write_ack_due: &mut DueTimes,
    wake_due: &mut DueTimes,
    pending_bounds: &mut PendingBounds,
    payload_due: &DueTimes,
    batch_replyless_writes: bool,
    scratch: &mut Vec<MemRequest>,
    l1_visibility_observations: &mut Vec<sim_mem::L1AccessObservation>,
    l1_fence_observations: &mut Vec<sim_mem::L1FenceObservation>,
    l1_fill_visibility_observations: &mut Vec<sim_mem::L1FillVisibleObservation>,
    l1_visibility_queue: Option<&mut L1VisibilityQueue>,
    _sm_ids: &[i32],
    l2_hit_latency: u64,
    memory_admission_feedback: Option<MemoryAdmissionCandidate>,
) -> Result<bool, SystemError> {
    scratch.clear();
    let write_or_pending_bounds = [
        write_ack_due.next(ix),
        pending_bounds.min(ix),
        payload_due.next(ix),
    ]
    .into_iter()
    .flatten()
    .min();
    let initial_visibility_events = cores[ix].take_l1_visibility_events();
    schedule_l1_visibility_events(
        ix,
        initial_visibility_events,
        queue,
        l1_visibility_observations,
        l1_fence_observations,
        l1_fill_visibility_observations,
        cores.len(),
    )?;
    let actor_horizon = replay_horizon(response_due.next(ix), write_or_pending_bounds);
    // The opt-in visibility protocol cannot safely run an actor past an
    // already queued cross-SM invalidation/fill event: its trace-local cache
    // state was generated before that event and cannot be retracted.  Keep
    // the legacy run-ahead path unchanged; strict visibility pays a bounded
    // next-event cap. `SmCore::advance` also yields immediately after the
    // issuing actor produces a same-cycle visibility observation, so those
    // callbacks are no longer hidden behind a later local issue batch.
    let horizon = if l1_visibility_queue.is_some() {
        queue
            .peek_time()
            .filter(|next| *next > queue.now())
            .map_or(actor_horizon, |next| match actor_horizon {
                ReplayHorizon::Bounded(bound) => ReplayHorizon::Bounded(bound.min(next)),
                ReplayHorizon::Unbounded => ReplayHorizon::Bounded(next),
            })
    } else {
        actor_horizon
    };
    let issued_before = cores[ix].instructions_issued();
    let adv = cores[ix].advance(horizon, scratch)?;
    let events = cores[ix].take_l1_visibility_events();
    schedule_l1_visibility_events(
        ix,
        events,
        queue,
        l1_visibility_observations,
        l1_fence_observations,
        l1_fill_visibility_observations,
        cores.len(),
    )?;
    let progress = cores[ix].instructions_issued() > issued_before || !scratch.is_empty();
    let target =
        u32::try_from(ix).map_err(|_| SystemError::ActorCountOutOfRange { count: cores.len() })?;
    schedule_emitted_requests(
        ix,
        TargetId(target),
        queue,
        pending_bounds,
        scratch,
        l2_hit_latency,
        batch_replyless_writes,
        memory_admission_feedback,
    )?;
    if let Advance::Idle {
        wake_at: Some(wake),
    } = adv
    {
        // A cross-SM event may advance the shared queue clock while this
        // actor still has a locally known wake timestamp. Never enqueue a
        // stale wake into the global queue; re-arm at the current event
        // boundary and let `SmCore::advance` catch the actor up on the next
        // pass. This preserves event causality without fabricating service
        // completion time.
        let wake = wake.max(queue.now());
        let covered = wake_due.next(ix).is_some_and(|t| t <= wake);
        if !covered {
            queue.schedule_at(wake, TargetId(target), Ev::SmWake)?;
            wake_due.push(ix, wake)?;
        }
    }
    Ok(progress)
}

fn schedule_l1_visibility_events(
    ix: usize,
    events: Vec<sim_mem::L1VisibilityEvent>,
    queue: &mut EventQueue<Ev>,
    l1_visibility_observations: &mut Vec<sim_mem::L1AccessObservation>,
    l1_fence_observations: &mut Vec<sim_mem::L1FenceObservation>,
    l1_fill_visibility_observations: &mut Vec<sim_mem::L1FillVisibleObservation>,
    actor_count: usize,
) -> Result<(), SystemError> {
    let target = TargetId(
        u32::try_from(ix).map_err(|_| SystemError::ActorCountOutOfRange { count: actor_count })?,
    );
    for event in events {
        match event {
            sim_mem::L1VisibilityEvent::Access(observation) => {
                l1_visibility_observations.push(observation);
                queue.schedule_at(
                    observation.issue_cycle,
                    target,
                    Ev::L1VisibilityAccess(observation),
                )?;
            }
            sim_mem::L1VisibilityEvent::Fence(observation) => {
                l1_fence_observations.push(observation);
                queue.schedule_at(
                    observation.issue_cycle,
                    target,
                    Ev::L1VisibilityFence(observation),
                )?;
            }
            sim_mem::L1VisibilityEvent::FillVisible(observation) => {
                l1_fill_visibility_observations.push(observation);
                queue.schedule_at(
                    observation.visible_cycle,
                    target,
                    Ev::L1FillVisible(observation),
                )?;
            }
            sim_mem::L1VisibilityEvent::InvalidateAll(observation) => {
                queue.schedule_at(
                    observation.issue_cycle,
                    target,
                    Ev::L1InvalidateAll(observation),
                )?;
            }
        }
    }
    Ok(())
}

fn schedule_l1_visibility_completion(
    ix: usize,
    plan: L1VisibilityPlan,
    queue: &mut EventQueue<Ev>,
    pending_bounds: &mut PendingBounds,
    wake_due: &DueTimes,
    sm_ids: &[i32],
    actor_count: usize,
) -> Result<(), SystemError> {
    if !plan.needs_completion() {
        return Ok(());
    }
    let mut affected = vec![ix];
    for target in plan.invalidations() {
        let target_ix = sm_ids
            .iter()
            .position(|candidate| u8::try_from(*candidate).ok() == Some(target.requester))
            .ok_or(SystemError::EventTargetOutOfRange {
                target: u32::from(target.requester),
                actor_count,
            })?;
        if !affected.contains(&target_ix) {
            affected.push(target_ix);
        }
    }
    queue.schedule_at(
        plan.service_done_cycle(),
        TargetId(
            u32::try_from(ix)
                .map_err(|_| SystemError::ActorCountOutOfRange { count: actor_count })?,
        ),
        Ev::L1VisibilityComplete { plan: plan.clone() },
    )?;
    for target_ix in affected {
        pending_bounds.insert(target_ix, plan.service_done_cycle())?;
        if wake_due
            .next(target_ix)
            .is_some_and(|time| time <= plan.service_done_cycle())
        {
            return Err(SystemError::L1VisibilityWakeConflict {
                sm: target_ix,
                time: plan.service_done_cycle(),
            });
        }
    }
    Ok(())
}

