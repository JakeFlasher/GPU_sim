// Lexical implementation section 3/4; included by the parent facade.
#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_kernel_shared_with_residency_and_candidates_with_coherence_and_payload_events(
    per_sm_blocks: &BTreeMap<i32, Vec<BlockReplay>>,
    schedule: &FunctionSchedule,
    params: &IssueModelParams,
    shared_memory_residency: SharedMemoryResidency,
    l1_config: L1Config,
    l2: &mut L2Model,
    dram: DramBw,
    memory_service_candidates: &[MemoryServiceCandidate],
    memory_admission_candidates: &[MemoryAdmissionCandidate],
    memory_admission_feedback: Option<MemoryAdmissionCandidate>,
    coherence_config: Option<CoherenceConfig>,
    payload_events: &[L1PayloadAdmissionEvent],
) -> Result<SystemResult, SystemError> {
    let arch = schedule.arch;
    require_arch(arch, params.arch, "issue parameters")?;
    require_arch(arch, l1_config.arch, "L1 configuration")?;
    require_arch(arch, l2.config().arch, "L2 configuration")?;
    checked_l2_counters(arch, l2)?;
    require_arch(arch, dram.arch(), "DRAM configuration")?;
    require_l2_pending_read_semantics(arch, l2.config().is_effective())?;
    let memory_admission_feedback = memory_admission_feedback
        .filter(|candidate| qualifies_store_fence_feedback(schedule, *candidate));
    params.validate_l1_timing(l1_config)?;
    // Fail closed on an inverted ladder: every horizon/run-ahead bound
    // below assumes no response can price under the L2-hit arm.
    if params.global_flat_latency < params.l1_miss_latency {
        return Err(SystemError::LatencyLadderInverted {
            l2_hit: params.l1_miss_latency,
            dram: params.global_flat_latency,
        });
    }
    // Fail closed on a zero bandwidth rate: a clamped server would be
    // absurd (1 B/s) or silently inert (0 Hz), and a 1 B/s server arms
    // the fixed-point u64 ceiling on large traces.
    let rate = match dram {
        DramBw::Carded {
            bytes_per_sec,
            sm_clock_hz,
            ..
        }
        | DramBw::CardedWriteback {
            bytes_per_sec,
            sm_clock_hz,
            ..
        } => Some((bytes_per_sec, sm_clock_hz)),
        DramBw::Unmodeled { .. } => None,
    };
    if let Some((bytes_per_sec, sm_clock_hz)) = rate
        && (bytes_per_sec == 0 || sm_clock_hz == 0)
    {
        return Err(SystemError::DramBandwidthInvalid {
            bytes_per_sec,
            sm_clock_hz,
        });
    }
    if u32::try_from(per_sm_blocks.len()).is_err() {
        return Err(SystemError::ActorCountOutOfRange {
            count: per_sm_blocks.len(),
        });
    }
    validate_memory_service_candidates(memory_service_candidates).map_err(IssueError::from)?;
    validate_memory_admission_candidates(memory_admission_candidates).map_err(IssueError::from)?;
    if let Some(candidate) = memory_admission_feedback {
        candidate.validate_feedback(params.read_barrier_latency)?;
    }
    let sm_ids: Vec<i32> = per_sm_blocks.keys().copied().collect();
    // L1TEX tags are private to each SM. Keep one report-only write-lookup
    // observer per SM rather than sharing a tag window across all actors;
    // cross-SM sharing fabricates hits when the same line index is reused on
    // another SM.
    let write_lookup_observers = if stateful_write_lookup_observer_enabled(schedule) {
        Some(
            sm_ids
                .iter()
                .map(|_| {
                    L1WriteLookupObserver::for_arch(arch)
                        .map(|observer| Rc::new(RefCell::new(observer)))
                        .map_err(IssueError::from)
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
    } else {
        None
    };
    let l1_visibility_protocol =
        coherence_config.is_some_and(CoherenceConfig::l1_visibility_protocol);
    let l1_resident_writeback =
        coherence_config.is_some_and(CoherenceConfig::l1_resident_writeback);
    if l1_resident_writeback && !l1_visibility_protocol {
        return Err(SystemError::L1DirtyWritebackRequiresVisibilityProtocol);
    }
    if l1_resident_writeback
        && (writeback_queue_unavailable(dram)
            || !memory_service_candidates.is_empty()
            || !memory_admission_candidates.is_empty()
            || memory_admission_feedback.is_some())
    {
        return Err(if writeback_queue_unavailable(dram) {
            SystemError::L1DirtyWritebackRequiresQueue
        } else {
            SystemError::L1DirtyWritebackCandidateConflict
        });
    }
    let mut cores: Vec<SmCore<'_>> = per_sm_blocks
        .values()
        .enumerate()
        .map(|(sm_ix, blocks)| {
            let mut core = SmCore::new_with_residency_and_observer(
                blocks,
                schedule,
                params,
                shared_memory_residency,
                SmMemMode::L1Shared(l1_config),
                write_lookup_observers
                    .as_ref()
                    .and_then(|observers| observers.get(sm_ix).cloned()),
            )?;
            if l1_visibility_protocol {
                let requester =
                    u8::try_from(sm_ids[sm_ix]).map_err(|_| IssueError::CounterOverflow {
                        counter: "L1 visibility requester domain",
                    })?;
                core.enable_l1_visibility_observations(requester)?;
            }
            if l1_resident_writeback {
                core.enable_l1_resident_writeback()?;
            }
            core.install_memory_service_candidates(memory_service_candidates)?;
            if !memory_admission_candidates.is_empty() {
                core.enable_memory_admission_observations();
            }
            if let Some(candidate) = memory_admission_feedback {
                core.install_memory_admission_feedback(candidate)?;
            }
            Ok::<_, IssueError>(core)
        })
        .collect::<Result<_, _>>()?;
    let observe_admission = !memory_admission_candidates.is_empty();
    let feedback_admission = memory_admission_feedback.is_some();
    let observe_admission_groups = observe_admission || feedback_admission;
    let mut admission_states = observe_admission
        .then(|| {
            (0..cores.len())
                .map(|_| MemoryAdmissionCandidateState::checked(memory_admission_candidates))
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()
        .map_err(IssueError::from)?;
    let mut admission_observers = observe_admission_groups.then(|| {
        (0..cores.len())
            .map(|_| AdmissionInstructionObserver::default())
            .collect::<Vec<_>>()
    });

    let mut queue: EventQueue<Ev> = EventQueue::new();
    // Replyless STORE/RED requests remain batchable even when the typed
    // admission/ack lane is active. The batch event still processes each
    // request independently; retaining it avoids changing unrelated L2
    // ordering merely because the terminal-EXIT arm is enabled.
    #[cfg(feature = "performance-counters")]
    let batch_replyless_writes = std::env::var_os("SIM_RUN_DISABLE_L2_ACCESS_BATCHING").is_none();
    #[cfg(not(feature = "performance-counters"))]
    let batch_replyless_writes = true;
    let mut scratch: Vec<MemRequest> = Vec::new();
    let mut l1_visibility_observations = Vec::new();
    let mut l1_fence_observations = Vec::new();
    let mut l1_fill_visibility_observations = Vec::new();
    // Scheduled-but-unapplied response times PER SM: the only external
    // events that can affect an SM's own state, hence its advance
    // horizon. Other SMs' events are deliberately NOT barriers — cross-SM
    // effects flow only through the L2 access stream, which the queue
    // already orders by (time, seq). (Treating every queued event as a
    // barrier would also livelock: two SMs parked at the same timestamp
    // would bounce off each other's wake events forever.)
    let mut response_due = DueTimes::new(cores.len());
    // Opt-in STORE/RED completions are self-affecting external events: a
    // matching MEMBAR remains pending until the event is processed, and the
    // actor must not run past the earliest pending acknowledgement.
    let mut write_ack_due = DueTimes::new(cores.len());
    // Pending wake times PER SM, deduplicating at the source: a wake is
    // scheduled only when no pending wake at an earlier-or-equal time
    // covers it (an earlier wake re-parks and reschedules if still
    // needed). Without this, every response that wakes an SM which then
    // re-parks at the same future horizon piles up another stale wake —
    // hundreds popping back-to-back at one timestamp on real cells —
    // wasted heap traffic AND an unbounded idle-wake streak the livelock
    // guard could not be told apart from a real spin.
    let mut wake_due = DueTimes::new(cores.len());
    // Producer-supplied payload admissions are self-affecting cache events;
    // keep their due frontier separate from responses, write acknowledgements,
    // and ordinary wakes so an actor cannot run past the mutation.
    let mut payload_due = DueTimes::new(cores.len());
    let mut pending_bounds = PendingBounds((0..cores.len()).map(|_| BTreeMap::new()).collect());
    let mut pending_l2_ready = PendingL2Ready::default();
    let mut l2_by_source = L2SourceSplit::for_arch(arch);
    let mut dram_server = DramServer::new(dram);
    let mut writeback_queue = match dram {
        DramBw::CardedWriteback {
            capacity_sectors,
            bytes_per_sec,
            sm_clock_hz,
            ..
        } => Some(WritebackQueue::new(
            capacity_sectors,
            bytes_per_sec,
            sm_clock_hz,
        )?),
        _ => None,
    };
    let mut coherence_queue = coherence_config.map(CoherenceQueue::new);
    let mut l1_visibility_queue = if l1_visibility_protocol {
        let config = coherence_config.ok_or(SystemError::L1VisibilityProtocolRequiresCoherence)?;
        Some(L1VisibilityQueue::new(L1VisibilityConfig::new(
            config.service_cycles_per_sector(),
            config.capacity_sectors(),
        )?))
    } else {
        None
    };
    let mut dram_stats =
        DramStats::for_arch(arch).with_service_rate(dram.service_rate_bytes_per_sec());
    // Seed actor wakes before producer events.  At equal cycle zero this lets
    // the actor observe `payload_due` and park before the value mutation is
    // applied, rather than allowing a queued payload admission to run ahead
    // of the first modeled issue boundary merely because it was inserted
    // first.  The separate payload horizon still prevents run-ahead at every
    // later timestamp.
    for ix in 0..cores.len() {
        let target = u32::try_from(ix)
            .map_err(|_| SystemError::ActorCountOutOfRange { count: cores.len() })?;
        queue.schedule_at(0, TargetId(target), Ev::SmWake)?;
        wake_due.push(ix, 0)?;
    }
    for event in payload_events {
        if event.payloads.is_empty() {
            return Err(SystemError::L1PayloadAdmissionEmpty { time: event.time });
        }
        let target = sm_ids
            .iter()
            .position(|candidate| *candidate == event.sm_id)
            .ok_or(SystemError::L1PayloadAdmissionSmUnknown { sm_id: event.sm_id })?;
        let target = u32::try_from(target)
            .map_err(|_| SystemError::ActorCountOutOfRange { count: cores.len() })?;
        if event.identity.is_some() && event.writer.is_none() {
            return Err(SystemError::L1PayloadAdmissionWriterMissing);
        }
        cores[target as usize].watch_l1_payload_sectors(&event.payloads);
        queue.schedule_at(
            event.time,
            TargetId(target),
            Ev::L1PayloadAdmission {
                identity_present: event.identity.is_some(),
                writer: event.writer,
                payloads: event.payloads.clone(),
                witnesses: event.witnesses.clone(),
            },
        )?;
        payload_due.push(target as usize, event.time)?;
    }

    let mut spin = SpinGuard::new(cores.len())?;
    let mut terminal_event_time = None;
    let mut sm_active_terminal_event_time = None;
    #[cfg(feature = "performance-counters")]
    let mut performance = std::env::var_os("SIM_RUN_PERFORMANCE_COUNTERS")
        .is_some()
        .then(EventPerformanceCounters::default);
    #[cfg(feature = "reference")]
    let mut reference_trace = ReferenceTrace::from_env()?;
    #[cfg(feature = "reference")]
    let mut power_state = PowerBoundaryState::activity_only(None, None);
    // The caller owns `l2` across this boundary. Complete all replay admission
    // and event-system setup first so their established typed errors keep
    // precedence over snapshot allocation failure. Then replay the entire
    // kernel on a private snapshot and publish it only after event draining,
    // deadlock checks, and result construction all succeed.
    let initial_lookup_observer = *checked_l2_counters(arch, l2)?;
    let mut lookup_observer_hits = 0_u64;
    let mut lookup_observer_requests = 0_u64;
    let mut read_lookup_hits = 0_u64;
    let mut read_pending_merge_hits = 0_u64;
    let mut read_lookup_requests = 0_u64;
    let mut store_lookup_hits = 0_u64;
    let mut store_lookup_requests = 0_u64;
    let mut store_packet_lookup_hits = 0_u64;
    let mut store_packet_resident_lookup_hits = 0_u64;
    let mut atomic_lookup_hits = 0_u64;
    let mut atomic_lookup_requests = 0_u64;
    let mut reduction_lookup_hits = 0_u64;
    let mut reduction_lookup_requests = 0_u64;
    const STORE_PACKET_WINDOW_CYCLES: u64 = 8;
    let mut store_packet_last: HashMap<(usize, i64, u8), u64> = HashMap::new();
    // One traced store instruction can still emit one-sector MemRequests.
    // Remember whether the line tag was present before the first request at
    // one (SM, issue-cycle, line) boundary so later sectors cannot self-credit
    // as write-tag hits on a cold coalesced instruction.
    let mut store_packet_line_tag_state: BTreeMap<(usize, u64, i64), bool> = BTreeMap::new();
    let mut store_packet_candidate_sectors = 0_u64;
    let mut staged_l2 = l2.try_snapshot()?;
    while let Some(ev) = queue.pop()? {
        #[cfg(feature = "reference")]
        let reference_event_name = event_name(&ev.kind);
        #[cfg(feature = "reference")]
        let reference_typed_event_payload = typed_event_payload(&ev.kind);
        let contributes_to_sm_active = !matches!(
            &ev.kind,
            Ev::MemoryWriteAck(_)
                | Ev::L1PayloadAdmission { .. }
                | Ev::L2WritebackComplete { .. }
                | Ev::L2CoherenceComplete { .. }
                | Ev::L1VisibilityComplete { .. }
                | Ev::L1VisibilityAccess(_)
                | Ev::L1VisibilityFence(_)
                | Ev::L1FillVisible(_)
                | Ev::L1InvalidateAll(_)
        );
        #[cfg(feature = "reference")]
        power_state.advance_to(ev.time)?;
        #[cfg(feature = "performance-counters")]
        if let Some(performance) = performance.as_mut() {
            let queue_depth_before_pop =
                queue
                    .len()
                    .checked_add(1)
                    .ok_or(SystemError::CounterOverflow {
                        counter: "performance event-queue depth",
                    })?;
            let logical_events = match &ev.kind {
                Ev::L2AccessBatch(requests) => requests.len(),
                _ => 1,
            };
            performance.observe(&ev.kind, ev.time, queue_depth_before_pop, logical_events)?;
        }
        let sm_ix =
            usize::try_from(ev.target.0).map_err(|_| SystemError::EventTargetOutOfRange {
                target: ev.target.0,
                actor_count: cores.len(),
            })?;
        if sm_ix >= cores.len() {
            return Err(SystemError::EventTargetOutOfRange {
                target: ev.target.0,
                actor_count: cores.len(),
            });
        }
        #[cfg(feature = "reference")]
        if let Some(trace) = reference_trace.as_mut() {
            trace.record_event(
                ev.time,
                ev.seq,
                ev.target.0,
                reference_event_name,
                ev.sequence_span.get(),
                reference_typed_event_payload.clone(),
                &queue,
                &staged_l2,
                &cores,
                &sm_ids,
                &response_due,
                &pending_l2_ready,
                &power_state,
                dram_server.as_ref(),
                "pre",
            )?;
        }
        #[cfg(feature = "reference")]
        let mut reference_native_result = reference_trace
            .as_mut()
            .and_then(ReferenceTrace::take_pending_native_result);
        match ev.kind {
            Ev::SmWake => {
                wake_due.pop_expect("SM wake", sm_ix, ev.time)?;
                #[cfg(feature = "reference")]
                let mut handled_reference_macro = false;
                #[cfg(feature = "reference")]
                if let Some(result) = reference_native_result.take() {
                    let actor_count = cores.len();
                    if dispatch_reference_macro_event(
                        &mut cores[sm_ix],
                        sm_ix,
                        actor_count,
                        &result,
                        &mut queue,
                        &mut wake_due,
                        Some(&mut power_state),
                        reference_trace.as_mut(),
                    )? {
                        handled_reference_macro = true;
                        spin.on_event(ev.time, false)?;
                    }
                }
                #[cfg(not(feature = "reference"))]
                let handled_reference_macro = false;
                if !handled_reference_macro {
                    let progress = run_sm(
                        sm_ix,
                        &mut cores,
                        &mut queue,
                        &mut response_due,
                        &mut write_ack_due,
                        &mut wake_due,
                        &mut pending_bounds,
                        &payload_due,
                        batch_replyless_writes,
                        &mut scratch,
                        &mut l1_visibility_observations,
                        &mut l1_fence_observations,
                        &mut l1_fill_visibility_observations,
                        l1_visibility_queue.as_mut(),
                        &sm_ids,
                        params.l1_miss_latency,
                        memory_admission_feedback,
                    )?;
                    spin.on_event(ev.time, !progress)?;
                }
            }
            Ev::L1PayloadAdmission {
                payloads,
                witnesses,
                writer,
                ..
            } => {
                payload_due.pop_expect("L1 payload admission", sm_ix, ev.time)?;
                if let Some(writer) = writer {
                    cores[sm_ix]
                        .admit_issued_l1_store_payloads(writer, ev.time, &payloads, &witnesses)?;
                } else {
                    cores[sm_ix].admit_l1_dirty_payloads_with_witness(&payloads, &witnesses)?;
                }
                run_sm(
                    sm_ix,
                    &mut cores,
                    &mut queue,
                    &mut response_due,
                    &mut write_ack_due,
                    &mut wake_due,
                    &mut pending_bounds,
                    &payload_due,
                    batch_replyless_writes,
                    &mut scratch,
                    &mut l1_visibility_observations,
                    &mut l1_fence_observations,
                    &mut l1_fill_visibility_observations,
                    l1_visibility_queue.as_mut(),
                    &sm_ids,
                    params.l1_miss_latency,
                    memory_admission_feedback,
                )?;
                spin.on_progress_event(ev.time);
            }
            kind @ (Ev::L2Access(_) | Ev::L2AccessBatch(_)) => {
                let requests = match kind {
                    Ev::L2Access(request) => L2AccessEventRequests::Single([request]),
                    Ev::L2AccessBatch(requests) => L2AccessEventRequests::Batch(requests),
                    _ => unreachable!(),
                };
                let mut process_request = |req: MemRequest| -> Result<(), SystemError> {
                    if req.issue_time != ev.time {
                        return Err(SystemError::RequestTimeMismatch {
                            sm: sm_ix,
                            issue_time: req.issue_time,
                            event_time: ev.time,
                        });
                    }
                    let own_sector_bit = 1u8 << (req.sector_id & 3) as u8;
                    if req.sector_mask == 0
                        || req.sector_mask & !0b1111 != 0
                        || req.sector_mask & own_sector_bit == 0
                    {
                        return Err(SystemError::SectorMaskInvalid {
                            sector_id: req.sector_id,
                            sector_mask: req.sector_mask,
                        });
                    }
                    let anchor_is_local = req.local_sector_mask & own_sector_bit != 0;
                    if req.local_sector_mask & !req.sector_mask != 0
                        || anchor_is_local != (req.origin == MemoryOrigin::Local)
                    {
                        return Err(SystemError::OriginMaskInvalid {
                            sector_id: req.sector_id,
                            origin: req.origin,
                            sector_mask: req.sector_mask,
                            local_sector_mask: req.local_sector_mask,
                        });
                    }
                    if !req.has_valid_contract() {
                        return Err(SystemError::MemoryRequestContractInvalid {
                            sector_id: req.sector_id,
                            request_source: req.source,
                            operation: req.operation,
                            reply: req.reply,
                        });
                    }
                    let feedback_pending_bounds_removal =
                        feedback_prices_request(memory_admission_feedback, req)
                            .then(|| pending_bounds.prepare_remove(sm_ix, req.issue_time))
                            .transpose()?;
                    // Prepare the exact ascending mask walk against a private copy
                    // of its set. The callback completes every fallible timing and
                    // counter calculation before the L2 plan commits; only then can
                    // the API issue a fill token. Event insertion follows
                    // immediately. If insertion fails, the kernel-private L2
                    // snapshot is discarded and the caller-owned model is unchanged.
                    let line = req.sector_id >> 2;
                    let effective_observer_operation = req.effective_observer_operation();
                    let store_packet_group = (req.origin == MemoryOrigin::Global
                        && effective_observer_operation == MemoryOperation::Store)
                        .then_some((sm_ix, req.issue_time, line));
                    // Stage an observer-only per-SM line/sector packet
                    // candidate. The source-sector baseline below remains
                    // one sector per request; a failed transactional access
                    // must not leak coalescer state.
                    let mut store_packet_keys = Vec::new();
                    if req.origin == MemoryOrigin::Global
                        && effective_observer_operation == MemoryOperation::Store
                    {
                        store_packet_last.retain(|(owner, _, _), last| {
                            *owner != sm_ix
                                || req.issue_time.saturating_sub(*last)
                                    <= STORE_PACKET_WINDOW_CYCLES
                        });
                        for sector in 0_u8..4 {
                            if req.sector_mask & (1 << sector) == 0 {
                                continue;
                            }
                            let key = (sm_ix, line, sector);
                            let recent = store_packet_last.get(&key).is_some_and(|last| {
                                req.issue_time.saturating_sub(*last) <= STORE_PACKET_WINDOW_CYCLES
                            });
                            if !recent {
                                store_packet_keys.push(key);
                            }
                        }
                    }
                    let mut store_packet_tag_was_present = None;
                    let (l2_access, commit) = staged_l2.access_mask_event_coalesced_transactional(
                        line,
                        req.sector_mask,
                        req.operation.l2_class(),
                        |preview| -> Result<L2SystemCommit, SystemError> {
                            let lookup = preview.lookup;
                            let merged_ready = pending_l2_ready.merged_ready(preview)?;
                            pending_l2_ready.preflight_register(preview)?;
                            let pending_pressure_sum = u64::try_from(pending_l2_ready.len())
                                .map_err(|_| SystemError::CounterOverflow {
                                    counter: "L2 pending-pressure occupancy",
                                })?;
                            let requested_sectors = u64::from(lookup.sectors);
                            let counted_hits = u64::from(lookup.counted_hits);
                            let pending_merge_sectors =
                                u64::from(preview.pending_merge_mask().count_ones());
                            let new_fill_sectors = u64::from(preview.fill_mask().count_ones());
                            let client_scope_ix = req.client_scope.index();
                            let client_scope_resident_hits =
                                u64::from((lookup.present_mask & req.sector_mask).count_ones());
                            let client_scope_fill_misses = new_fill_sectors;
                            let client_scope_pending_merges = pending_merge_sectors;
                            if req.operation == MemoryOperation::Read {
                                let resident_mask = lookup.present_mask & req.sector_mask;
                                let pending_mask = preview.pending_merge_mask();
                                let fill_mask = preview.fill_mask();
                                debug_assert_eq!(
                                    resident_mask | pending_mask | fill_mask,
                                    req.sector_mask,
                                    "read client-scope masks must cover the request"
                                );
                                debug_assert_eq!(
                                    resident_mask & pending_mask,
                                    0,
                                    "read resident/pending client-scope masks overlap"
                                );
                                debug_assert_eq!(
                                    resident_mask & fill_mask,
                                    0,
                                    "read resident/fill client-scope masks overlap"
                                );
                                debug_assert_eq!(
                                    pending_mask & fill_mask,
                                    0,
                                    "read pending/fill client-scope masks overlap"
                                );
                            } else {
                                debug_assert_eq!(
                                    client_scope_pending_merges, 0,
                                    "write-class client-scope pending arm must be zero"
                                );
                                debug_assert_eq!(
                                    client_scope_fill_misses, 0,
                                    "write-class client-scope fill arm must be zero"
                                );
                            }
                            let ltc128b_shadow_promoted_sectors = if req.client_scope
                                == L2ClientScope::Ltc128b
                                && preview.fill_mask() != 0
                            {
                                u64::from(((!req.sector_mask) & 0x0f).count_ones())
                            } else {
                                0
                            };
                            let store_packet_resident_lookup_hits = if store_packet_group.is_some()
                            {
                                let packet_sector_mask = store_packet_keys
                                    .iter()
                                    .fold(0_u8, |mask, &(_, _, sector)| mask | (1_u8 << sector));
                                u64::from(
                                    (preview.lookup.present_mask & packet_sector_mask).count_ones(),
                                )
                            } else {
                                0
                            };
                            let store_packet_lookup_hits = if let Some(group) = store_packet_group {
                                let tag_was_present = store_packet_line_tag_state.get(&group);
                                let hit_mask = match tag_was_present {
                                    Some(true) => {
                                        preview.lookup.present_mask
                                            | preview.lookup.pre_packet_write_tag_hit_mask
                                    }
                                    Some(false) => 0,
                                    None => {
                                        preview.lookup.present_mask
                                            | preview.lookup.pre_packet_write_tag_hit_mask
                                    }
                                };
                                store_packet_tag_was_present = Some(
                                    store_packet_line_tag_state.get(&group).copied().unwrap_or(
                                        preview.lookup.present_mask != 0
                                            || preview.lookup.pre_packet_write_tag_hit_mask != 0,
                                    ),
                                );
                                u64::from((hit_mask & req.sector_mask).count_ones())
                            } else {
                                0
                            };
                            let observer_operation = effective_observer_operation;
                            let lookup_for_operation =
                                |operation| -> Result<(u64, u64), SystemError> {
                                    Ok(match operation {
                                        MemoryOperation::Read => (
                                            requested_sectors,
                                            counted_hits
                                                .checked_add(u64::from(
                                                    preview.pending_merge_mask().count_ones(),
                                                ))
                                                .ok_or(SystemError::CounterOverflow {
                                                    counter: "L2 read lookup observer hits",
                                                })?,
                                        ),
                                        // ATOM is a returning write-class request,
                                        // but the sector lookup observer still uses
                                        // present sectors. Physical write-tag and RMW
                                        // accounting remain separate from this
                                        // source-unit hit/miss endpoint.
                                        MemoryOperation::Atomic => {
                                            (requested_sectors, u64::from(lookup.present_sectors))
                                        }
                                        // The source-unit TEX sector observer counts
                                        // write-class lookup sectors too.  Keep these
                                        // as per-sector requests; packet counts are a
                                        // separate endpoint and must not be mixed into
                                        // `lts__t_sectors_srcunit_tex`.
                                        // The current source-unit projection retains
                                        // the independently declared write-tag arm;
                                        // write packetization and per-op NCU scope are
                                        // not yet isolated across the scored corpus.
                                        // This is observer-only and never changes L2
                                        // state, DRAM demand, or completion timing.
                                        MemoryOperation::Store => {
                                            if arch == ArchId::Sm86 {
                                                (requested_sectors, requested_sectors)
                                            } else {
                                                (
                                                    requested_sectors,
                                                    u64::from(lookup.present_sectors),
                                                )
                                            }
                                        }
                                        MemoryOperation::Reduction => {
                                            (requested_sectors, u64::from(lookup.present_sectors))
                                        }
                                    })
                                };
                            // Physical lookup totals remain tied to the
                            // actual hierarchy operation. The optional
                            // observer tag only partitions operation-qualified
                            // counters (generic ST.E is the current user).
                            let (lookup_observer_requests, lookup_observer_hits) =
                                lookup_for_operation(effective_observer_operation)?;
                            let (operation_observer_requests, operation_observer_hits) =
                                lookup_for_operation(observer_operation)?;
                            let source_ix = req.source as usize;
                            let origin_masks = [
                                req.sector_mask & !req.local_sector_mask,
                                req.local_sector_mask,
                            ];
                            let origin_requested =
                                origin_masks.map(|mask| u64::from(mask.count_ones()));
                            let origin_hits = origin_masks.map(|mask| {
                                u64::from((mask & lookup.counted_hit_mask).count_ones())
                            });
                            let next_source_sectors = l2_by_source.sectors[source_ix]
                                .checked_add(requested_sectors)
                                .ok_or(SystemCounterError::Overflow {
                                    counter: "L2 source sectors",
                                })?;
                            let next_source_hits = l2_by_source.hits[source_ix]
                                .checked_add(counted_hits)
                                .ok_or(SystemCounterError::Overflow {
                                    counter: "L2 source hits",
                                })?;
                            let mut next_origin_source_sectors = [0u64; 2];
                            let mut next_origin_source_hits = [0u64; 2];
                            for origin in MemoryOrigin::ALL {
                                let origin_ix = origin.index();
                                next_origin_source_sectors[origin_ix] = l2_by_source.origin_sectors
                                    [origin_ix][source_ix]
                                    .checked_add(origin_requested[origin_ix])
                                    .ok_or(SystemCounterError::Overflow {
                                        counter: "L2 origin/source sectors",
                                    })?;
                                next_origin_source_hits[origin_ix] = l2_by_source.origin_hits
                                    [origin_ix][source_ix]
                                    .checked_add(origin_hits[origin_ix])
                                    .ok_or(SystemCounterError::Overflow {
                                        counter: "L2 origin/source hits",
                                    })?;
                            }

                            let fetch_mask = match req.operation {
                                MemoryOperation::Read => preview.fill_mask(),
                                operation if operation.fetches_missing_data() => {
                                    req.sector_mask & !lookup.present_mask
                                }
                                _ => 0,
                            };
                            let fetch_sectors = u64::from(fetch_mask.count_ones());
                            let atomic_rmw_read_sectors =
                                if req.operation == MemoryOperation::Atomic {
                                    fetch_sectors
                                } else {
                                    0
                                };
                            let reduction_rmw_read_sectors =
                                if req.operation == MemoryOperation::Reduction {
                                    fetch_sectors
                                } else {
                                    0
                                };
                            let next_atomic_rmw_read_sectors = dram_stats
                                .atomic_rmw_read_sectors
                                .checked_add(atomic_rmw_read_sectors)
                                .ok_or(SystemError::CounterOverflow {
                                    counter: "DRAM atomic RMW read sectors",
                                })?;
                            let next_reduction_rmw_read_sectors = dram_stats
                                .reduction_rmw_read_sectors
                                .checked_add(reduction_rmw_read_sectors)
                                .ok_or(SystemError::CounterOverflow {
                                    counter: "DRAM reduction RMW read sectors",
                                })?;
                            let origin_fetches = origin_masks
                                .map(|mask| u64::from((mask & fetch_mask).count_ones()));
                            let next_read_sectors = dram_stats
                                .read_sectors
                                .checked_add(fetch_sectors)
                                .ok_or(SystemError::CounterOverflow {
                                    counter: "DRAM read sectors",
                                })?;
                            let mut next_origin_read_sectors = [0u64; 2];
                            for origin in MemoryOrigin::ALL {
                                let origin_ix = origin.index();
                                next_origin_read_sectors[origin_ix] = dram_stats.by_origin
                                    [origin_ix]
                                    .read_sectors
                                    .checked_add(origin_fetches[origin_ix])
                                    .ok_or(SystemError::CounterOverflow {
                                        counter: "DRAM origin read sectors",
                                    })?;
                            }
                            let topology_stats = match preview.eviction() {
                                Some(eviction) => dram_stats.with_l2_eviction(eviction)?,
                                None => dram_stats,
                            }
                            .with_partition_candidate(line, fetch_sectors)?;
                            // Clean replacements remain topology observations;
                            // only a nonzero dirty mask reserves writeback
                            // service capacity.
                            let writeback_eviction = preview
                                .eviction()
                                .filter(|eviction| eviction.dirty_mask != 0);
                            let global_sector_mask = req.sector_mask & !req.local_sector_mask;
                            let coherence_plan = if global_sector_mask != 0 {
                                if let Some(queue) = &coherence_queue {
                                    Some(queue.preview(CoherenceRequest {
                                        line_id: line,
                                        sector_mask: global_sector_mask,
                                        requester: u8::try_from(sm_ids[sm_ix]).map_err(|_| {
                                            SystemError::Coherence(
                                                sim_mem::CoherenceError::RequesterOutOfRange(
                                                    u8::MAX,
                                                ),
                                            )
                                        })?,
                                        is_write: matches!(
                                            req.operation,
                                            MemoryOperation::Store
                                                | MemoryOperation::Atomic
                                                | MemoryOperation::Reduction
                                        ),
                                        issue_cycle: req.issue_time,
                                    })?)
                                } else {
                                    None
                                }
                            } else {
                                None
                            };
                            let service_preview = if fetch_sectors > 0 {
                                let bytes = fetch_sectors.checked_mul(32).ok_or(
                                    SystemError::CounterOverflow {
                                        counter: "DRAM request bytes",
                                    },
                                )?;
                                match (&dram_server, &writeback_queue) {
                                    (Some(server), None) => {
                                        Some(server.preview(req.issue_time, bytes)?)
                                    }
                                    (None, Some(_queue)) => None,
                                    (None, None) => None,
                                    (Some(_), Some(_)) => {
                                        return Err(SystemError::Writeback(
                                            sim_mem::WritebackError::PlanMismatch,
                                        ));
                                    }
                                }
                            } else {
                                None
                            };
                            let (read_service, writeback_service) = if let Some(queue) =
                                &writeback_queue
                            {
                                if let Some(eviction) = writeback_eviction {
                                    // Validate the exact L2 victim masks before
                                    // constructing the combined service reservation.
                                    queue.preview(eviction, req.issue_time)?;
                                }
                                let mut requests = Vec::with_capacity(
                                    usize::from(writeback_eviction.is_some())
                                        + usize::from(fetch_sectors > 0),
                                );
                                if let Some(eviction) = writeback_eviction {
                                    requests.push(ServiceRequest::writeback(
                                        eviction.line_id,
                                        eviction.dirty_mask,
                                        req.issue_time,
                                    ));
                                }
                                if fetch_sectors > 0 {
                                    let bytes = fetch_sectors.checked_mul(32).ok_or(
                                        SystemError::CounterOverflow {
                                            counter: "DRAM request bytes",
                                        },
                                    )?;
                                    requests.push(ServiceRequest::read_line(
                                        bytes,
                                        line,
                                        req.issue_time,
                                    ));
                                }
                                let plans = queue.preview_services(&requests)?;
                                let writeback = if writeback_eviction.is_some() {
                                    Some(*plans.first().ok_or(SystemError::Writeback(
                                        sim_mem::WritebackError::PlanMismatch,
                                    ))?)
                                } else {
                                    None
                                };
                                let read_index = usize::from(writeback_eviction.is_some());
                                let read = if fetch_sectors > 0 {
                                    Some(*plans.get(read_index).ok_or(SystemError::Writeback(
                                        sim_mem::WritebackError::PlanMismatch,
                                    ))?)
                                } else {
                                    None
                                };
                                (read, writeback)
                            } else {
                                (None, None)
                            };
                            let bw_ready = read_service
                                .map(|plan| plan.service_done_cycle())
                                .or_else(|| service_preview.map(|(_, ready)| ready))
                                .unwrap_or(0);

                            let fetch_latency_ready = if fetch_sectors > 0 {
                                let operation = if req.reply != ReplyTo::None {
                                    "DRAM response visibility"
                                } else if req.operation == MemoryOperation::Atomic {
                                    "unobserved ATOM DRAM completion"
                                } else {
                                    "DRAM fill visibility"
                                };
                                Some(checked_cycle_add(
                                    operation,
                                    req.issue_time,
                                    params.global_flat_latency,
                                )?)
                            } else {
                                None
                            };
                            let fill_ready = if preview.fill_mask() != 0 {
                                Some(
                                    fetch_latency_ready
                                        .ok_or(SystemError::FillScheduleMismatch {
                                            sector_id: req.sector_id,
                                            has_token: true,
                                            has_ready: false,
                                        })?
                                        .max(bw_ready),
                                )
                            } else {
                                None
                            };

                            let response = if req.reply != ReplyTo::None {
                                let response_origin_ix = req.origin.index();
                                let bound = checked_cycle_add(
                                    "pending L2-hit response bound",
                                    req.issue_time,
                                    params.l1_miss_latency,
                                )?;
                                let pending_bounds_removal =
                                    pending_bounds.prepare_remove(sm_ix, bound)?;
                                let latency_ready = bound
                                    .max(fetch_latency_ready.unwrap_or(0))
                                    .max(merged_ready.unwrap_or(0))
                                    .max(
                                        coherence_plan.map_or(0, |plan| plan.service_done_cycle()),
                                    );
                                let service_ready = latency_ready.max(bw_ready);
                                // A folded physical fill can serve ordinary
                                // and deep constant waiters together. Their
                                // return-service delta is therefore stamped
                                // per fill waiter by the SM response path;
                                // this event stays at the shared hierarchy
                                // ready time and does not over-delay ordinary
                                // consumers or alter DRAM pricing.
                                if service_ready < bound {
                                    return Err(SystemError::ResponseBeforeBound {
                                        sm: sm_ix,
                                        ready: service_ready,
                                        bound,
                                    });
                                }
                                let delayed = u64::from(service_ready > latency_ready);
                                let next_delayed = dram_stats
                                    .bw_delayed_responses
                                    .checked_add(delayed)
                                    .ok_or(SystemError::CounterOverflow {
                                        counter: "DRAM bandwidth-delayed responses",
                                    })?;
                                let next_origin_delayed = dram_stats.by_origin[response_origin_ix]
                                    .bw_delayed_responses
                                    .checked_add(delayed)
                                    .ok_or(SystemError::CounterOverflow {
                                        counter: "DRAM origin bandwidth-delayed responses",
                                    })?;
                                Some(L2ResponseCommit {
                                    ready: service_ready,
                                    origin_ix: response_origin_ix,
                                    next_delayed,
                                    next_origin_delayed,
                                    pending_bounds_removal,
                                })
                            } else {
                                None
                            };

                            let atomic_completion_ready = if matches!(
                                req.operation,
                                MemoryOperation::Atomic | MemoryOperation::Reduction
                            ) && response.is_none()
                            {
                                let latency_ready = match fetch_latency_ready {
                                    Some(ready) => ready,
                                    None => checked_cycle_add(
                                        if req.operation == MemoryOperation::Reduction {
                                            "unobserved RED L2-hit completion"
                                        } else {
                                            "unobserved ATOM L2-hit completion"
                                        },
                                        req.issue_time,
                                        params.l1_miss_latency,
                                    )?,
                                };
                                Some(latency_ready.max(bw_ready).max(
                                    coherence_plan.map_or(0, |plan| plan.service_done_cycle()),
                                ))
                            } else {
                                None
                            };
                            let service_done = read_service
                                .as_ref()
                                .map(|plan| plan.service_done_cycle())
                                .or_else(|| service_preview.map(|(_, ready)| ready));
                            let l2_frontier_samples = u64::from(
                                fetch_sectors != 0 || response.is_some() || service_done.is_some(),
                            );
                            let issue_to_fill_cycles = fill_ready
                                .map(|ready| ready.saturating_sub(req.issue_time))
                                .unwrap_or(0);
                            let issue_to_response_cycles = response
                                .as_ref()
                                .map(|value| value.ready.saturating_sub(req.issue_time))
                                .unwrap_or(0);
                            let service_delay_cycles = service_done
                                .map(|ready| ready.saturating_sub(req.issue_time))
                                .unwrap_or(0);

                            Ok(L2SystemCommit {
                                source_ix,
                                source_sectors: next_source_sectors,
                                source_hits: next_source_hits,
                                origin_source_sectors: next_origin_source_sectors,
                                origin_source_hits: next_origin_source_hits,
                                read_sectors: next_read_sectors,
                                origin_read_sectors: next_origin_read_sectors,
                                topology_stats,
                                dram_free_fp: service_preview.map(|(done, _)| done),
                                read_service,
                                writeback: writeback_service,
                                coherence: coherence_plan,
                                fill_ready,
                                response_l2_allocation_generation: (req.reply != ReplyTo::None)
                                    .then(|| preview.allocation_generation()),
                                atomic_rmw_read_sectors: next_atomic_rmw_read_sectors,
                                reduction_rmw_read_sectors: next_reduction_rmw_read_sectors,
                                atomic_completion_ready,
                                response,
                                observed_sectors: requested_sectors,
                                observed_hits: counted_hits,
                                lookup_observer_requests,
                                lookup_observer_hits,
                                read_lookup_requests: if matches!(
                                    observer_operation,
                                    MemoryOperation::Read
                                ) {
                                    operation_observer_requests
                                } else {
                                    0
                                },
                                read_lookup_hits: if matches!(
                                    observer_operation,
                                    MemoryOperation::Read
                                ) {
                                    operation_observer_hits
                                } else {
                                    0
                                },
                                read_pending_merge_hits: if matches!(
                                    observer_operation,
                                    MemoryOperation::Read
                                ) {
                                    u64::from(preview.pending_merge_mask().count_ones())
                                } else {
                                    0
                                },
                                store_lookup_requests: if matches!(
                                    observer_operation,
                                    MemoryOperation::Store
                                ) {
                                    operation_observer_requests
                                } else {
                                    0
                                },
                                store_lookup_hits: if matches!(
                                    observer_operation,
                                    MemoryOperation::Store
                                ) {
                                    operation_observer_hits
                                } else {
                                    0
                                },
                                store_packet_lookup_hits,
                                store_packet_resident_lookup_hits,
                                atomic_lookup_requests: if matches!(
                                    observer_operation,
                                    MemoryOperation::Atomic
                                ) {
                                    operation_observer_requests
                                } else {
                                    0
                                },
                                atomic_lookup_hits: if matches!(
                                    observer_operation,
                                    MemoryOperation::Atomic
                                ) {
                                    operation_observer_hits
                                } else {
                                    0
                                },
                                reduction_lookup_requests: if matches!(
                                    observer_operation,
                                    MemoryOperation::Reduction
                                ) {
                                    operation_observer_requests
                                } else {
                                    0
                                },
                                reduction_lookup_hits: if matches!(
                                    observer_operation,
                                    MemoryOperation::Reduction
                                ) {
                                    operation_observer_hits
                                } else {
                                    0
                                },
                                pending_merge_sectors,
                                new_fill_sectors,
                                pending_pressure_samples: 1,
                                pending_pressure_sum,
                                l2_frontier_samples,
                                l2_issue_to_fill_cycles_sum: issue_to_fill_cycles,
                                l2_issue_to_fill_cycles_max: issue_to_fill_cycles,
                                l2_issue_to_response_cycles_sum: issue_to_response_cycles,
                                l2_issue_to_response_cycles_max: issue_to_response_cycles,
                                l2_service_delay_cycles_sum: service_delay_cycles,
                                l2_service_delay_cycles_max: service_delay_cycles,
                                client_scope_ix,
                                client_scope_requests: requested_sectors,
                                client_scope_resident_hits,
                                client_scope_pending_merges,
                                client_scope_fill_misses,
                                ltc128b_shadow_promoted_sectors,
                            })
                        },
                    )?;

                    l2_by_source.client_scope_requests[commit.client_scope_ix] = l2_by_source
                        .client_scope_requests[commit.client_scope_ix]
                        .checked_add(commit.client_scope_requests)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "L2 client-scope requests",
                        })?;
                    l2_by_source.client_scope_resident_hits[commit.client_scope_ix] = l2_by_source
                        .client_scope_resident_hits[commit.client_scope_ix]
                        .checked_add(commit.client_scope_resident_hits)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "L2 client-scope resident hits",
                        })?;
                    l2_by_source.client_scope_pending_merges[commit.client_scope_ix] = l2_by_source
                        .client_scope_pending_merges[commit.client_scope_ix]
                        .checked_add(commit.client_scope_pending_merges)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "L2 client-scope pending merges",
                        })?;
                    l2_by_source.client_scope_fill_misses[commit.client_scope_ix] = l2_by_source
                        .client_scope_fill_misses[commit.client_scope_ix]
                        .checked_add(commit.client_scope_fill_misses)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "L2 client-scope fill misses",
                        })?;
                    l2_by_source.ltc128b_shadow_promoted_sectors = l2_by_source
                        .ltc128b_shadow_promoted_sectors
                        .checked_add(commit.ltc128b_shadow_promoted_sectors)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "L2 LTC128B shadow promoted sectors",
                        })?;

                    for key in &store_packet_keys {
                        store_packet_last.insert(*key, req.issue_time);
                    }
                    if let (Some(group), Some(tag_was_present)) =
                        (store_packet_group, store_packet_tag_was_present)
                    {
                        store_packet_line_tag_state.insert(group, tag_was_present);
                    }
                    store_packet_candidate_sectors = store_packet_candidate_sectors
                        .checked_add(u64::try_from(store_packet_keys.len()).map_err(|_| {
                            SystemError::CounterOverflow {
                                counter: "L2 store packet candidate sectors",
                            }
                        })?)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "L2 store packet candidate sectors",
                        })?;

                    if let Some(writeback_queue) = writeback_queue.as_mut() {
                        let mut service_plans = Vec::with_capacity(2);
                        if let Some(plan) = commit.writeback {
                            service_plans.push(plan);
                        }
                        if let Some(plan) = commit.read_service {
                            service_plans.push(plan);
                        }
                        writeback_queue.commit_services(&service_plans)?;
                    }
                    if let Some(coherence_queue) = coherence_queue.as_mut()
                        && let Some(plan) = commit.coherence
                    {
                        coherence_queue.commit(plan)?;
                    }

                    lookup_observer_requests = lookup_observer_requests
                        .checked_add(commit.lookup_observer_requests)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "L2 lookup observer requests",
                        })?;
                    lookup_observer_hits = lookup_observer_hits
                        .checked_add(commit.lookup_observer_hits)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "L2 lookup observer hits",
                        })?;
                    read_lookup_requests = read_lookup_requests
                        .checked_add(commit.read_lookup_requests)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "L2 read lookup observer requests",
                        })?;
                    read_lookup_hits = read_lookup_hits
                        .checked_add(commit.read_lookup_hits)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "L2 read lookup observer hits",
                        })?;
                    read_pending_merge_hits = read_pending_merge_hits
                        .checked_add(commit.read_pending_merge_hits)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "L2 read pending-merge lookup hits",
                        })?;
                    store_lookup_requests = store_lookup_requests
                        .checked_add(commit.store_lookup_requests)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "L2 store lookup observer requests",
                        })?;
                    store_lookup_hits = store_lookup_hits
                        .checked_add(commit.store_lookup_hits)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "L2 store lookup observer hits",
                        })?;
                    store_packet_lookup_hits = store_packet_lookup_hits
                        .checked_add(commit.store_packet_lookup_hits)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "L2 store packet lookup hits",
                        })?;
                    store_packet_resident_lookup_hits = store_packet_resident_lookup_hits
                        .checked_add(commit.store_packet_resident_lookup_hits)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "L2 resident store packet lookup hits",
                        })?;
                    atomic_lookup_requests = atomic_lookup_requests
                        .checked_add(commit.atomic_lookup_requests)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "L2 atomic lookup observer requests",
                        })?;
                    atomic_lookup_hits = atomic_lookup_hits
                        .checked_add(commit.atomic_lookup_hits)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "L2 atomic lookup observer hits",
                        })?;
                    reduction_lookup_requests = reduction_lookup_requests
                        .checked_add(commit.reduction_lookup_requests)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "L2 reduction lookup observer requests",
                        })?;
                    reduction_lookup_hits = reduction_lookup_hits
                        .checked_add(commit.reduction_lookup_hits)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "L2 reduction lookup observer hits",
                        })?;
                    staged_l2.record_fill_ownership_observer(
                        commit.pending_merge_sectors,
                        commit.new_fill_sectors,
                    )?;
                    l2_by_source.pending_pressure_samples[commit.source_ix] = l2_by_source
                        .pending_pressure_samples[commit.source_ix]
                        .checked_add(commit.pending_pressure_samples)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "L2 source pending-pressure samples",
                        })?;
                    l2_by_source.pending_pressure_sum[commit.source_ix] = l2_by_source
                        .pending_pressure_sum[commit.source_ix]
                        .checked_add(commit.pending_pressure_sum)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "L2 source pending-pressure sum",
                        })?;

                    if observe_admission_groups
                        && let Some(witness) = req.admission_witness
                        && let Some(completed) = admission_observers
                            .as_mut()
                            .expect("enabled admission observer storage")[sm_ix]
                            .observe(
                                sm_ids[sm_ix],
                                witness,
                                req.operation,
                                req.issue_time,
                                commit.observed_sectors,
                                commit.observed_hits,
                                req.sector_id,
                            )?
                    {
                        if observe_admission {
                            admission_states
                                .as_mut()
                                .expect("enabled admission candidate storage")[sm_ix]
                                .observe(
                                    usize::from(completed.witness.subcore()),
                                    usize::try_from(completed.witness.warp()).map_err(|_| {
                                        SystemError::CounterOverflow {
                                            counter: "memory-admission witness warp",
                                        }
                                    })?,
                                    completed.operation,
                                    completed.hit_sectors,
                                    completed.miss_sectors,
                                    completed.issue_time,
                                )?;
                        }
                        if feedback_admission
                            && let Some(ack) = cores[sm_ix].complete_memory_admission_feedback(
                                completed.witness,
                                completed.operation,
                                completed.hit_sectors,
                                completed.miss_sectors,
                                completed.l2_requests,
                                completed.unique_regions,
                                completed.issue_time,
                                ev.time,
                            )?
                        {
                            queue.schedule_at(
                                ack.fence_visible_at,
                                ev.target,
                                Ev::MemoryWriteAck(ack),
                            )?;
                            write_ack_due.push(sm_ix, ack.fence_visible_at)?;
                        }
                    }

                    let has_token = l2_access.fill.is_some();
                    let has_ready = commit.fill_ready.is_some();
                    if has_token != has_ready {
                        return Err(SystemError::FillScheduleMismatch {
                            sector_id: req.sector_id,
                            has_token,
                            has_ready,
                        });
                    }
                    if let (Some(fill), Some(ready)) = (l2_access.fill, commit.fill_ready) {
                        #[cfg(feature = "reference")]
                        {
                            let typed_sm_id = u32::try_from(sm_ids[sm_ix]).unwrap_or(u32::MAX);
                            let producer =
                                cores[sm_ix].typed_l2_request_key(typed_sm_id, req.reply);
                            pending_l2_ready.register_with_producer(fill, ready, producer)?;
                        }
                        #[cfg(not(feature = "reference"))]
                        pending_l2_ready.register(fill, ready)?;
                    }
                    if let Some(plan) = commit.writeback {
                        queue.schedule_at(
                            plan.service_done_cycle(),
                            ev.target,
                            Ev::L2WritebackComplete { plan },
                        )?;
                    }
                    if let Some(plan) = commit.coherence
                        && plan.needs_completion()
                    {
                        queue.schedule_at(
                            plan.service_done_cycle(),
                            ev.target,
                            Ev::L2CoherenceComplete { plan },
                        )?;
                    }
                    // A response that becomes ready with its own new fill carries the
                    // token so installation precedes delivery in one event. A mixed
                    // new/pending read can instead finish its new fetch first and wait
                    // longer for an older pending sector; expose that fill at its true
                    // ready time, then deliver a fill-less response.
                    if let Some(response) = commit.response.as_ref() {
                        let response_fill = match (l2_access.fill, commit.fill_ready) {
                            (Some(fill), Some(fill_ready)) if fill_ready == response.ready => {
                                Some(fill)
                            }
                            (Some(fill), Some(fill_ready)) if fill_ready < response.ready => {
                                queue.schedule_at(fill_ready, ev.target, Ev::L2Fill(fill))?;
                                None
                            }
                            (Some(_), Some(fill_ready)) => {
                                return Err(SystemError::ResponseBeforeFill {
                                    sector_id: req.sector_id,
                                    response_ready: response.ready,
                                    fill_ready,
                                });
                            }
                            (None, None) => None,
                            _ => {
                                return Err(SystemError::FillScheduleMismatch {
                                    sector_id: req.sector_id,
                                    has_token,
                                    has_ready,
                                });
                            }
                        };
                        queue.schedule_at(
                            response.ready,
                            ev.target,
                            Ev::MemResponse {
                                reply: req.reply,
                                fill: response_fill,
                            },
                        )?;
                    } else if let (Some(fill), Some(ready)) = (l2_access.fill, commit.fill_ready) {
                        queue.schedule_at(ready, ev.target, Ev::L2Fill(fill))?;
                    } else if let Some(ready) = commit.atomic_completion_ready {
                        queue.schedule_at(ready, ev.target, Ev::AtomicCompletion)?;
                    }

                    l2_by_source.sectors[commit.source_ix] = commit.source_sectors;
                    l2_by_source.hits[commit.source_ix] = commit.source_hits;
                    for origin in MemoryOrigin::ALL {
                        let origin_ix = origin.index();
                        l2_by_source.origin_sectors[origin_ix][commit.source_ix] =
                            commit.origin_source_sectors[origin_ix];
                        l2_by_source.origin_hits[origin_ix][commit.source_ix] =
                            commit.origin_source_hits[origin_ix];
                    }
                    dram_stats.read_sectors = commit.read_sectors;
                    dram_stats.atomic_rmw_read_sectors = commit.atomic_rmw_read_sectors;
                    dram_stats.reduction_rmw_read_sectors = commit.reduction_rmw_read_sectors;
                    dram_stats.l2_frontier_samples = dram_stats
                        .l2_frontier_samples
                        .checked_add(commit.l2_frontier_samples)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "L2 frontier samples",
                        })?;
                    dram_stats.l2_issue_to_fill_cycles_sum = dram_stats
                        .l2_issue_to_fill_cycles_sum
                        .checked_add(commit.l2_issue_to_fill_cycles_sum)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "L2 issue-to-fill frontier cycles",
                        })?;
                    dram_stats.l2_issue_to_fill_cycles_max = dram_stats
                        .l2_issue_to_fill_cycles_max
                        .max(commit.l2_issue_to_fill_cycles_max);
                    dram_stats.l2_issue_to_response_cycles_sum = dram_stats
                        .l2_issue_to_response_cycles_sum
                        .checked_add(commit.l2_issue_to_response_cycles_sum)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "L2 issue-to-response frontier cycles",
                        })?;
                    dram_stats.l2_issue_to_response_cycles_max = dram_stats
                        .l2_issue_to_response_cycles_max
                        .max(commit.l2_issue_to_response_cycles_max);
                    dram_stats.l2_service_delay_cycles_sum = dram_stats
                        .l2_service_delay_cycles_sum
                        .checked_add(commit.l2_service_delay_cycles_sum)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "L2 service-delay frontier cycles",
                        })?;
                    dram_stats.l2_service_delay_cycles_max = dram_stats
                        .l2_service_delay_cycles_max
                        .max(commit.l2_service_delay_cycles_max);
                    for origin in MemoryOrigin::ALL {
                        let origin_ix = origin.index();
                        dram_stats.by_origin[origin_ix].read_sectors =
                            commit.origin_read_sectors[origin_ix];
                    }
                    dram_stats.l2_evicted_lines = commit.topology_stats.l2_evicted_lines;
                    dram_stats.l2_evicted_valid_sectors =
                        commit.topology_stats.l2_evicted_valid_sectors;
                    dram_stats.l2_evicted_pending_lines =
                        commit.topology_stats.l2_evicted_pending_lines;
                    dram_stats.l2_evicted_pending_sectors =
                        commit.topology_stats.l2_evicted_pending_sectors;
                    dram_stats.l2_dirty_eviction_lines =
                        commit.topology_stats.l2_dirty_eviction_lines;
                    dram_stats.l2_dirty_evicted_sectors =
                        commit.topology_stats.l2_dirty_evicted_sectors;
                    dram_stats.dram_partition_modulo_sectors =
                        commit.topology_stats.dram_partition_modulo_sectors;
                    dram_stats.dram_partition_xor_sectors =
                        commit.topology_stats.dram_partition_xor_sectors;
                    dram_stats.dram_slice_modulo_sectors =
                        commit.topology_stats.dram_slice_modulo_sectors;
                    if let Some(done) = commit.dram_free_fp
                        && let Some(server) = &mut dram_server
                    {
                        server.free_fp = done;
                    }
                    if let Some(response) = commit.response {
                        dram_stats.bw_delayed_responses = response.next_delayed;
                        dram_stats.by_origin[response.origin_ix].bw_delayed_responses =
                            response.next_origin_delayed;
                        response_due.push_response(
                            sm_ix,
                            response.ready,
                            req.reply.response_id().map(|id| id.boundary_identity()),
                            commit.response_l2_allocation_generation,
                        )?;
                        response.pending_bounds_removal.commit(&mut pending_bounds);
                    }
                    if let Some(removal) = feedback_pending_bounds_removal {
                        removal.commit(&mut pending_bounds);
                    }
                    spin.on_progress_event(ev.time);
                    Ok(())
                };
                for request in requests.as_ref().iter().copied() {
                    process_request(request)?;
                }
            }
            Ev::L2Fill(fill) => {
                pending_l2_ready.preflight_remove(fill, ev.time)?;
                let _completion = staged_l2.complete_fill(fill)?;
                pending_l2_ready.commit_remove(fill);
                spin.on_progress_event(ev.time);
            }
            Ev::AtomicCompletion => {
                spin.on_progress_event(ev.time);
            }
            Ev::L2WritebackComplete { plan } => {
                let queue = writeback_queue.as_mut().ok_or(SystemError::Writeback(
                    sim_mem::WritebackError::PlanMismatch,
                ))?;
                let completion = queue.complete(plan.token(), ev.time)?;
                dram_stats.writeback_completed_lines = dram_stats
                    .writeback_completed_lines
                    .checked_add(1)
                    .ok_or(SystemError::CounterOverflow {
                        counter: "completed L2 writeback lines",
                    })?;
                dram_stats.writeback_completed_sectors = dram_stats
                    .writeback_completed_sectors
                    .checked_add(u64::from(completion.dirty_mask.count_ones()))
                    .ok_or(SystemError::CounterOverflow {
                        counter: "completed L2 writeback sectors",
                    })?;
                dram_stats.writeback_completed_bytes = dram_stats
                    .writeback_completed_bytes
                    .checked_add(completion.dirty_mask.count_ones() as u64 * 32)
                    .ok_or(SystemError::CounterOverflow {
                        counter: "completed L2 writeback bytes",
                    })?;
                dram_stats.writeback_completion_frontier = Some(
                    dram_stats
                        .writeback_completion_frontier
                        .unwrap_or(0)
                        .max(ev.time),
                );
                spin.on_progress_event(ev.time);
            }
            Ev::L2CoherenceComplete { plan } => {
                let queue = coherence_queue.as_mut().ok_or(SystemError::Coherence(
                    sim_mem::CoherenceError::ArithmeticOverflow,
                ))?;
                let completion = queue.complete(plan.token(), ev.time)?;
                dram_stats.coherence_completed_transactions = dram_stats
                    .coherence_completed_transactions
                    .checked_add(1)
                    .ok_or(SystemError::CounterOverflow {
                        counter: "completed coherence transactions",
                    })?;
                dram_stats.coherence_completed_sectors = dram_stats
                    .coherence_completed_sectors
                    .checked_add(u64::from(completion.service_sectors))
                    .ok_or(SystemError::CounterOverflow {
                        counter: "completed coherence sectors",
                    })?;
                dram_stats.coherence_completion_frontier = Some(
                    dram_stats
                        .coherence_completion_frontier
                        .unwrap_or(0)
                        .max(ev.time),
                );
                spin.on_progress_event(ev.time);
            }
            Ev::L1VisibilityAccess(observation) => {
                let visibility_queue = l1_visibility_queue
                    .as_mut()
                    .ok_or(SystemError::L1VisibilityProtocolRequiresCoherence)?;
                let plan = visibility_queue.preview_qualified(
                    observation,
                    sim_mem::MemoryVisibilityQualifier::Cacheable,
                )?;
                visibility_queue.commit(plan.clone())?;
                schedule_l1_visibility_completion(
                    sm_ix,
                    plan,
                    &mut queue,
                    &mut pending_bounds,
                    &wake_due,
                    &sm_ids,
                    cores.len(),
                )?;
                spin.on_progress_event(ev.time);
            }
            Ev::L1VisibilityFence(observation) => {
                let visibility_queue = l1_visibility_queue
                    .as_mut()
                    .ok_or(SystemError::L1VisibilityProtocolRequiresCoherence)?;
                visibility_queue.observe_fence(observation)?;
                spin.on_progress_event(ev.time);
            }
            Ev::L1FillVisible(observation) => {
                let visibility_queue = l1_visibility_queue
                    .as_mut()
                    .ok_or(SystemError::L1VisibilityProtocolRequiresCoherence)?;
                visibility_queue.observe_fill_visible(observation)?;
                spin.on_progress_event(ev.time);
            }
            Ev::L1InvalidateAll(observation) => {
                let visibility_queue = l1_visibility_queue
                    .as_mut()
                    .ok_or(SystemError::L1VisibilityProtocolRequiresCoherence)?;
                visibility_queue.observe_invalidate_all(observation)?;
                spin.on_progress_event(ev.time);
            }
            Ev::L1VisibilityComplete { plan } => {
                let visibility_queue = l1_visibility_queue
                    .as_mut()
                    .ok_or(SystemError::L1VisibilityProtocolRequiresCoherence)?;
                let _completion = visibility_queue.complete(plan.token(), ev.time)?;
                let mut affected = vec![
                    sm_ids
                        .iter()
                        .position(|candidate| {
                            u8::try_from(*candidate).ok() == Some(plan.requester())
                        })
                        .ok_or(SystemError::EventTargetOutOfRange {
                            target: u32::from(plan.requester()),
                            actor_count: cores.len(),
                        })?,
                ];
                for target in plan.invalidations() {
                    let target_ix = sm_ids
                        .iter()
                        .position(|candidate| {
                            u8::try_from(*candidate).ok() == Some(target.requester)
                        })
                        .ok_or(SystemError::EventTargetOutOfRange {
                            target: u32::from(target.requester),
                            actor_count: cores.len(),
                        })?;
                    cores[target_ix].invalidate_l1(plan.line_id(), target.sector_mask)?;
                    if !affected.contains(&target_ix) {
                        affected.push(target_ix);
                    }
                }
                for &target_ix in &affected {
                    let bound = plan.service_done_cycle();
                    match pending_bounds.0[target_ix].get(&bound).copied() {
                        Some(1) => {
                            pending_bounds.0[target_ix].remove(&bound);
                        }
                        Some(count) => {
                            pending_bounds.0[target_ix].insert(bound, count - 1);
                        }
                        None => {
                            return Err(SystemError::PendingBoundMissing {
                                sm: target_ix,
                                bound,
                            });
                        }
                    }
                }
                for target_ix in affected {
                    run_sm(
                        target_ix,
                        &mut cores,
                        &mut queue,
                        &mut response_due,
                        &mut write_ack_due,
                        &mut wake_due,
                        &mut pending_bounds,
                        &payload_due,
                        batch_replyless_writes,
                        &mut scratch,
                        &mut l1_visibility_observations,
                        &mut l1_fence_observations,
                        &mut l1_fill_visibility_observations,
                        l1_visibility_queue.as_mut(),
                        &sm_ids,
                        params.l1_miss_latency,
                        memory_admission_feedback,
                    )?;
                }
                spin.on_progress_event(ev.time);
            }
            Ev::MemoryWriteAck(ack) => {
                write_ack_due.pop_expect("modeled write acknowledgement", sm_ix, ev.time)?;
                cores[sm_ix].complete_memory_write_ack(ack, ev.time)?;
                run_sm(
                    sm_ix,
                    &mut cores,
                    &mut queue,
                    &mut response_due,
                    &mut write_ack_due,
                    &mut wake_due,
                    &mut pending_bounds,
                    &payload_due,
                    batch_replyless_writes,
                    &mut scratch,
                    &mut l1_visibility_observations,
                    &mut l1_fence_observations,
                    &mut l1_fill_visibility_observations,
                    l1_visibility_queue.as_mut(),
                    &sm_ids,
                    params.l1_miss_latency,
                    memory_admission_feedback,
                )?;
                spin.on_event(ev.time, false)?;
            }
            Ev::MemResponse { reply, fill } => {
                if let Some(fill) = fill {
                    pending_l2_ready.preflight_remove(fill, ev.time)?;
                }
                cores[sm_ix].preflight_response(reply, ev.time)?;
                if let Some(fill) = fill {
                    // `Stale` is an expected no-op after eviction; delivery of
                    // data to the requester remains valid and must continue.
                    let _completion = staged_l2.complete_fill(fill)?;
                    pending_l2_ready.commit_remove(fill);
                }
                response_due.pop_response_expect(
                    "memory response",
                    sm_ix,
                    ev.time,
                    reply.response_id().map(|id| id.boundary_identity()),
                )?;
                scratch.clear();
                cores[sm_ix].on_response(reply, ev.time, &mut scratch)?;
                // Retries the completion unblocked become L2 accesses at
                // the response time.
                schedule_emitted_requests(
                    sm_ix,
                    ev.target,
                    &mut queue,
                    &mut pending_bounds,
                    &mut scratch,
                    params.l1_miss_latency,
                    batch_replyless_writes,
                    memory_admission_feedback,
                )?;
                run_sm(
                    sm_ix,
                    &mut cores,
                    &mut queue,
                    &mut response_due,
                    &mut write_ack_due,
                    &mut wake_due,
                    &mut pending_bounds,
                    &payload_due,
                    batch_replyless_writes,
                    &mut scratch,
                    &mut l1_visibility_observations,
                    &mut l1_fence_observations,
                    &mut l1_fill_visibility_observations,
                    l1_visibility_queue.as_mut(),
                    &sm_ids,
                    params.l1_miss_latency,
                    memory_admission_feedback,
                )?;
                spin.on_event(ev.time, false)?;
            }
        }
        terminal_event_time = Some(ev.time);
        if contributes_to_sm_active {
            sm_active_terminal_event_time = Some(ev.time);
        }
        #[cfg(feature = "reference")]
        {
            let defer_macro_post = reference_trace
                .as_ref()
                .is_some_and(ReferenceTrace::has_active_native_result);
            if defer_macro_post {
                // The scalar macro already advanced the typed power/core
                // frontier to its region end.  A second snapshot at the
                // original event identity would be a time-travelling
                // pre/post pair; the continuation event is the authoritative
                // region-end boundary and performs the comparison.
                if let Some(trace) = reference_trace.as_mut() {
                    trace.discard_active_native_result();
                }
            } else {
                power_state.record_event(reference_event_name)?;
                if let Some(trace) = reference_trace.as_mut() {
                    trace.record_event(
                        ev.time,
                        ev.seq,
                        ev.target.0,
                        reference_event_name,
                        ev.sequence_span.get(),
                        reference_typed_event_payload,
                        &queue,
                        &staged_l2,
                        &cores,
                        &sm_ids,
                        &response_due,
                        &pending_l2_ready,
                        &power_state,
                        dram_server.as_ref(),
                        "post",
                    )?;
                }
            }
        }
    }

    if pending_l2_ready.len() != 0 {
        return Err(SystemError::PendingL2ReadyNotDrained {
            entries: pending_l2_ready.len(),
        });
    }
    // Resident L1 dirtiness is an explicit policy experiment. Flush it before
    // the shared L2 terminal gauge so the two typed frontiers remain distinct.
    flush_l1_dirty_residency(
        &mut cores,
        &staged_l2,
        writeback_queue.as_mut(),
        &mut terminal_event_time,
        &mut dram_stats,
    )?;
    let terminal_dirty_residency = staged_l2.dirty_residency();
    dram_stats = dram_stats.with_terminal_l2_dirty_residency(terminal_dirty_residency)?;
    // Terminal resident dirty sectors are distinct from eviction victims.
    // The opt-in CardedWriteback path now admits them transactionally only
    // after all SM/L2 events have drained; the default path retains the dirty
    // gauge and its explicit tail flag.
    if let Some(writeback_queue) = writeback_queue.as_mut() {
        let terminal_issue = terminal_event_time.unwrap_or(0);
        let dirty_plan = staged_l2.preview_dirty_flush();
        if !dirty_plan.lines().is_empty() {
            let capacity = usize::try_from(writeback_queue.capacity_sectors()).map_err(|_| {
                SystemError::CounterOverflow {
                    counter: "terminal writeback capacity",
                }
            })?;
            let mut offset = 0usize;
            let mut latest_done = terminal_issue;
            let mut batch_issue = terminal_issue;
            while offset < dirty_plan.lines().len() {
                let mut end = offset;
                let mut sectors = 0usize;
                while end < dirty_plan.lines().len() {
                    let line_sectors = usize::try_from(
                        dirty_plan.lines()[end].dirty_mask.count_ones(),
                    )
                    .map_err(|_| SystemError::CounterOverflow {
                        counter: "terminal dirty sectors",
                    })?;
                    if end == offset && line_sectors > capacity {
                        return Err(SystemError::Writeback(
                            sim_mem::WritebackError::CapacityExceeded {
                                capacity: writeback_queue.capacity_sectors(),
                                requested: u32::try_from(line_sectors).unwrap_or(u32::MAX),
                            },
                        ));
                    }
                    if sectors
                        .checked_add(line_sectors)
                        .is_none_or(|total| total > capacity)
                    {
                        break;
                    }
                    sectors += line_sectors;
                    end += 1;
                }
                let requests = dirty_plan.lines()[offset..end]
                    .iter()
                    .map(|line| {
                        ServiceRequest::writeback(line.line_id, line.dirty_mask, batch_issue)
                    })
                    .collect::<Vec<_>>();
                let service_plans = writeback_queue.preview_services(&requests)?;
                writeback_queue.commit_services(&service_plans)?;
                for service in &service_plans {
                    let completion =
                        writeback_queue.complete(service.token(), service.service_done_cycle())?;
                    latest_done = latest_done.max(completion.service_done_cycle);
                    dram_stats.writeback_completed_lines = dram_stats
                        .writeback_completed_lines
                        .checked_add(1)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "completed L2 writeback lines",
                        })?;
                    dram_stats.writeback_completed_sectors = dram_stats
                        .writeback_completed_sectors
                        .checked_add(u64::from(completion.dirty_mask.count_ones()))
                        .ok_or(SystemError::CounterOverflow {
                            counter: "completed L2 writeback sectors",
                        })?;
                    dram_stats.writeback_completed_bytes = dram_stats
                        .writeback_completed_bytes
                        .checked_add(u64::from(completion.dirty_mask.count_ones()) * 32)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "completed L2 writeback bytes",
                        })?;
                }
                batch_issue = latest_done;
                offset = end;
            }
            staged_l2.commit_dirty_flush(dirty_plan)?;
            dram_stats.writeback_completion_frontier = Some(
                dram_stats
                    .writeback_completion_frontier
                    .unwrap_or(0)
                    .max(latest_done),
            );
            terminal_event_time = Some(terminal_event_time.unwrap_or(0).max(latest_done));
        }
    }
    if let Some(writeback_queue) = writeback_queue.as_ref() {
        let pending_entries = writeback_queue.pending_tokens().count();
        if writeback_queue.in_flight_sectors() != 0 || pending_entries != 0 {
            return Err(SystemError::WritebackNotDrained {
                sectors: writeback_queue.in_flight_sectors(),
                entries: pending_entries,
            });
        }
    }
    if let Some(coherence_queue) = coherence_queue.as_ref() {
        let pending_entries = coherence_queue.pending_tokens().count();
        if coherence_queue.in_flight_sectors() != 0 || pending_entries != 0 {
            return Err(SystemError::CoherenceNotDrained {
                sectors: coherence_queue.in_flight_sectors(),
                entries: pending_entries,
            });
        }
    }
    if !write_ack_due.is_empty() {
        return Err(SystemError::MemoryWriteAckDueNotDrained);
    }
    if !response_due.is_empty() {
        return Err(SystemError::ResponseDueNotDrained {
            entries: response_due.pending_entries(),
        });
    }
    if !wake_due.is_empty() {
        return Err(SystemError::WakeDueNotDrained {
            entries: wake_due.pending_entries(),
        });
    }
    if !payload_due.is_empty() {
        return Err(SystemError::PayloadAdmissionDueNotDrained {
            entries: payload_due.pending_entries(),
        });
    }
    if !pending_bounds.is_empty() {
        return Err(SystemError::PendingBoundsNotDrained {
            entries: pending_bounds.pending_entries(),
        });
    }
    if observe_admission_groups {
        for (ix, observer) in admission_observers
            .as_ref()
            .expect("enabled admission observer storage")
            .iter()
            .enumerate()
        {
            if observer.pending.is_some() {
                return Err(SystemError::AdmissionObservationIncomplete { sm_id: sm_ids[ix] });
            }
        }
    }

    // The L2 state machine already counted every source-unit sector in the
    // operation-specific observer.  Do not add a second packetized write
    // stream here: packet counts belong to a separate requests endpoint and
    // would undercount sector metrics while forcing write lookups to hits.
    let lookup_hits = initial_lookup_observer
        .lookup_hits
        .checked_add(lookup_observer_hits)
        .ok_or(SystemError::CounterOverflow {
            counter: "L2 lookup observer hits",
        })?;
    let lookup_requests = initial_lookup_observer
        .lookup_requests
        .checked_add(lookup_observer_requests)
        .ok_or(SystemError::CounterOverflow {
            counter: "L2 lookup observer requests",
        })?;
    staged_l2.replace_lookup_observer(lookup_hits, lookup_requests)?;
    l2_by_source.read_lookup_requests = read_lookup_requests;
    l2_by_source.read_lookup_hits = read_lookup_hits;
    l2_by_source.read_pending_merge_hits = read_pending_merge_hits;
    l2_by_source.store_lookup_requests = store_lookup_requests;
    l2_by_source.store_lookup_hits = store_lookup_hits;
    l2_by_source.store_packet_lookup_hits = store_packet_lookup_hits;
    l2_by_source.store_packet_resident_lookup_hits = store_packet_resident_lookup_hits;
    l2_by_source.store_packet_candidate_sectors = store_packet_candidate_sectors;
    l2_by_source.atomic_lookup_requests = atomic_lookup_requests;
    l2_by_source.atomic_lookup_hits = atomic_lookup_hits;
    l2_by_source.reduction_lookup_requests = reduction_lookup_requests;
    l2_by_source.reduction_lookup_hits = reduction_lookup_hits;

    let mut per_sm = BTreeMap::new();
    let mut unmodeled_tails = TailFlags::default();
    #[cfg(feature = "reference")]
    let mut advance_diagnostics = AdvanceDiagnostics::default();
    for (ix, core) in cores.iter().enumerate() {
        let pending_write_acks = core.pending_memory_write_acks();
        if pending_write_acks != 0 {
            return Err(SystemError::MemoryWriteAckPending {
                sm_id: sm_ids[ix],
                pending: pending_write_acks,
            });
        }
        if !core.is_done() {
            return Err(SystemError::Deadlock {
                sm_id: sm_ids[ix],
                outstanding: core.outstanding_responses(),
            });
        }
        let outstanding = core.outstanding_responses();
        if outstanding != 0 {
            return Err(SystemError::ResponseNotDrained {
                sm_id: sm_ids[ix],
                outstanding,
            });
        }
        let mut sm_result = core.result();
        if observe_admission {
            sm_result.readiness.memory_admission_candidates = admission_states
                .as_ref()
                .expect("enabled admission candidate storage")[ix]
                .diagnostics();
        }
        unmodeled_tails = unmodeled_tails.union(sm_result.unmodeled_tails);
        per_sm.insert(sm_ids[ix], sm_result);
        #[cfg(feature = "reference")]
        {
            let current = core.advance_diagnostics();
            advance_diagnostics.calls = advance_diagnostics
                .calls
                .checked_add(current.calls)
                .ok_or(SystemError::CounterOverflow {
                    counter: "reference advance calls",
                })?;
            advance_diagnostics.local_cycle_steps = advance_diagnostics
                .local_cycle_steps
                .checked_add(current.local_cycle_steps)
                .ok_or(SystemError::CounterOverflow {
                    counter: "reference local cycle steps",
                })?;
            advance_diagnostics.idle_jumps = advance_diagnostics
                .idle_jumps
                .checked_add(current.idle_jumps)
                .ok_or(SystemError::CounterOverflow {
                    counter: "reference idle jumps",
                })?;
            advance_diagnostics.skipped_cycles = advance_diagnostics
                .skipped_cycles
                .checked_add(current.skipped_cycles)
                .ok_or(SystemError::CounterOverflow {
                    counter: "reference skipped cycles",
                })?;
        }
    }
    let l2_counters = *checked_l2_counters(arch, &staged_l2)?;
    #[cfg(feature = "performance-counters")]
    if let Some(performance) = performance {
        let queue_stats = queue.stats();
        eprintln!(
            "SIM_EVENT_PERF events_processed={} queue_events_processed={} sm_wake={} \
             l1_payload_admission={} l2_access={} l2_access_batches={} l2_access_batched_requests={} l2_fill={} \
             atomic_completion={} writeback_completion={} coherence_completion={} l1_visibility_event={} \
             memory_write_ack={} mem_response={} same_cycle_after_first={} \
             queue_same_cycle_after_first={} maximum_queue_depth={} queue_clock_gap_cycles={} \
             queue_last_clock_gap_cycles={}",
            performance.processed,
            performance.queue_events_processed,
            performance.sm_wake,
            performance.l1_payload_admission,
            performance.l2_access,
            performance.l2_access_batches,
            performance.l2_access_batched_requests,
            performance.l2_fill,
            performance.atomic_completion,
            performance.writeback_completion,
            performance.coherence_completion,
            performance.l1_visibility_event,
            performance.memory_write_ack,
            performance.mem_response,
            performance.same_cycle_after_first,
            performance.queue_same_cycle_after_first,
            performance.maximum_queue_depth,
            queue_stats.clock_gap_cycles,
            queue_stats.last_clock_gap_cycles,
        );
    }
    #[cfg(feature = "reference")]
    if std::env::var_os("SIM_SM_REFERENCE_MODE").is_some() {
        let mode =
            if std::env::var_os("SIM_SM_REFERENCE_MODE").is_some_and(|value| value == "dense") {
                "dense"
            } else {
                "event"
            };
        eprintln!(
            "SIM_REFERENCE_PERF mode={mode} calls={} local_cycle_steps={} idle_jumps={} skipped_cycles={}",
            advance_diagnostics.calls,
            advance_diagnostics.local_cycle_steps,
            advance_diagnostics.idle_jumps,
            advance_diagnostics.skipped_cycles,
        );
    }
    #[cfg(feature = "reference")]
    if let Some(trace) = reference_trace {
        trace.finish(queue.stats(), &power_state)?;
    }
    // The terminal gauge was recorded before any opt-in resident flush above.
    let result = SystemResult {
        arch,
        per_sm,
        l2: l2_counters,
        l2_by_source,
        dram: dram_stats,
        unmodeled_tails,
        terminal_event_time,
        sm_active_terminal_event_time,
        l1_visibility_observations,
        l1_fence_observations,
        l1_fill_visibility_observations,
    };
    *l2 = staged_l2;
    Ok(result)
}

fn replay_horizon(response: Option<u64>, pending_bounds: Option<u64>) -> ReplayHorizon {
    match (response, pending_bounds) {
        (Some(left), Some(right)) => ReplayHorizon::Bounded(left.min(right)),
        (Some(bound), None) | (None, Some(bound)) => ReplayHorizon::Bounded(bound),
        (None, None) => ReplayHorizon::Unbounded,
    }
}

fn batchable_replyless_write(request: MemRequest) -> bool {
    request.reply == ReplyTo::None
        && matches!(
            request.operation,
            MemoryOperation::Store | MemoryOperation::Reduction
        )
}

fn feedback_prices_request(
    candidate: Option<MemoryAdmissionCandidate>,
    request: MemRequest,
) -> bool {
    request.admission_witness.is_some()
        && candidate.is_some_and(|candidate| candidate.matches_operation(request.operation))
}

