// Lexical implementation section 2/4; included by the parent facade.
#[cfg(feature = "reference")]
impl ReferenceTrace {
    fn from_env() -> Result<Option<Self>, ReferenceTraceError> {
        let Some(path) = std::env::var_os("SIM_SM_REFERENCE_TRACE") else {
            return Ok(None);
        };
        Self::from_paths(
            PathBuf::from(path),
            std::env::var_os("SIM_SM_REFERENCE_TYPED_SCALAR_WITNESS").map(PathBuf::from),
        )
    }

    fn from_paths(
        path: PathBuf,
        witness_path: Option<PathBuf>,
    ) -> Result<Option<Self>, ReferenceTraceError> {
        let max_records = match std::env::var("SIM_SM_REFERENCE_MAX_RECORDS") {
            Ok(value) => value
                .parse::<u64>()
                .ok()
                .filter(|&value| value > 1)
                .ok_or(ReferenceTraceError::InvalidLimit { value })?,
            Err(std::env::VarError::NotPresent) => DEFAULT_REFERENCE_TRACE_MAX_RECORDS,
            Err(std::env::VarError::NotUnicode(value)) => {
                return Err(ReferenceTraceError::InvalidLimit {
                    value: value.to_string_lossy().into_owned(),
                });
            }
        };
        let mut trace = Self {
            writer: BufWriter::new(File::create(path)?),
            records: 0,
            max_records,
            previous_event: None,
            issue_effect_cursors: Vec::new(),
            pending_typed_entry: None,
            witness: witness_path
                .map(|path| {
                    let bytes = std::fs::read(path)?;
                    let witness: ReferenceMacroWitness = serde_json::from_slice(&bytes)?;
                    if witness.schema != "reification_native_macro_witness_v1" {
                        return Err(ReferenceTraceError::Witness(
                            "schema must be reification_native_macro_witness_v1".to_owned(),
                        ));
                    }
                    Ok(witness)
                })
                .transpose()?,
            pending_native_result: None,
            active_native_result: None,
            pending_region_end: None,
            native_macro_dispatches: 0,
            native_macro_fallbacks: 0,
        };
        trace.write(&serde_json::json!({
            "kind": "header",
            "schema_version": sim_semantics::BOUNDARY_SCHEMA_VERSION,
            "typed_state_schema_version": TYPED_STATE_SCHEMA_VERSION,
            "typed_l2_power_contract": TypedObservationContract::cycle_exact_power(),
            "typed_root_schema": format!("typed_event_boundary_v{TYPED_STATE_SCHEMA_VERSION}"),
            "reference_record_schema_version": 3,
            "event_encoding": "full_then_delta_v2",
            "mode": std::env::var("SIM_SM_REFERENCE_MODE").unwrap_or_else(|_| "event".to_owned()),
            "state_projection_complete": false,
            "state_projection_incomplete_reason": "legacy_core_debug_projections",
            "observation_contract": "boundary_projection",
            "native_macro_witness": trace.witness.is_some(),
            "native_macro_execution": "comparator_selected_reference_event_loop",
        }))?;
        Ok(Some(trace))
    }

    fn take_pending_native_result(&mut self) -> Option<NativeBoundaryExecutionResult> {
        self.pending_native_result.take()
    }

    fn note_native_fallback(&mut self) {
        self.native_macro_fallbacks = self.native_macro_fallbacks.saturating_add(1);
    }

    fn note_native_dispatch(&mut self) {
        self.native_macro_dispatches = self.native_macro_dispatches.saturating_add(1);
    }

    fn expect_region_end(&mut self, exit: TypedEventBoundaryRecord) {
        self.pending_region_end = Some(exit);
    }

    fn has_active_native_result(&self) -> bool {
        self.active_native_result.is_some()
    }

    fn discard_active_native_result(&mut self) {
        self.active_native_result = None;
    }

    fn native_result_decision(
        result: &NativeBoundaryExecutionResult,
    ) -> sim_semantics::NativeBoundaryRefinementDecision {
        match result {
            NativeBoundaryExecutionResult::TypedMacro { .. } => {
                sim_semantics::NativeBoundaryRefinementDecision::Preserve
            }
            NativeBoundaryExecutionResult::ExactReplay { reason, .. } => {
                sim_semantics::NativeBoundaryRefinementDecision::Reify {
                    reason: reason.clone(),
                }
            }
        }
    }

    fn write<T: serde::Serialize>(&mut self, value: &T) -> Result<(), ReferenceTraceError> {
        if self.records >= self.max_records {
            return Err(ReferenceTraceError::RecordLimit {
                limit: self.max_records,
            });
        }
        serde_json::to_writer(&mut self.writer, value)?;
        self.writer.write_all(b"\n")?;
        self.records += 1;
        Ok(())
    }

    fn write_event(&mut self, value: serde_json::Value) -> Result<(), ReferenceTraceError> {
        // Move the previous state out so the writer can be borrowed without
        // cloning either potentially large boundary.
        if let Some(previous) = self.previous_event.take() {
            let mut changes = Vec::new();
            reference_operations(&previous, &value, &mut Vec::new(), &mut changes);
            #[derive(serde::Serialize)]
            struct Delta<'a> {
                kind: &'static str,
                base_record: u64,
                operations: Vec<ReferenceOperation<'a>>,
            }
            self.write(&Delta {
                kind: "event_delta",
                base_record: self.records - 1,
                operations: changes,
            })?;
        } else {
            self.write(&value)?;
        }
        self.previous_event = Some(value);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn record_event(
        &mut self,
        time: u64,
        sequence: u64,
        target: u32,
        event: &'static str,
        sequence_span: u64,
        typed_payload: TypedEventPayload,
        queue: &EventQueue<Ev>,
        l2: &L2Model,
        cores: &[SmCore<'_>],
        sm_ids: &[i32],
        response_due: &DueTimes,
        pending_l2_ready: &PendingL2Ready,
        power: &PowerBoundaryState,
        dram_server: Option<&DramServer>,
        boundary_phase: &'static str,
    ) -> Result<(), ReferenceTraceError> {
        let snapshots = cores
            .iter()
            .map(SmCore::boundary_snapshot)
            .collect::<Vec<_>>();
        let typed_l1_states = cores.iter().map(SmCore::typed_l1_state).collect::<Vec<_>>();
        let typed_async_copy_states = cores
            .iter()
            .map(SmCore::typed_async_copy_states)
            .collect::<Vec<_>>();
        let typed_memory_pipeline_states = cores
            .iter()
            .map(SmCore::typed_memory_pipeline_state)
            .collect::<Vec<_>>();
        let typed_scalar_pipeline_states = cores
            .iter()
            .map(SmCore::typed_scalar_pipeline_state)
            .collect::<Vec<_>>();
        let typed_front_end_states = cores
            .iter()
            .map(SmCore::typed_front_end_state)
            .collect::<Vec<_>>();
        let pending = queue.pending_projected(|queued| TypedPendingEvent {
            time: queued.time,
            sequence: queued.seq,
            phase: 0,
            sequence_span: queued.sequence_span.get(),
            target: queued.target.0,
            payload: typed_event_payload(&queued.kind),
        });
        let typed_queue = QueueBoundaryState {
            depth: u64::try_from(queue.len()).unwrap_or(u64::MAX),
            next_event_time: queue.peek_time(),
            next_sequence: queue.next_sequence(),
            pending,
        };
        typed_queue.validate()?;
        let mut typed_l2 = l2.typed_boundary_state()?;
        let mut owners_by_core = Vec::with_capacity(cores.len());
        for (ix, core) in cores.iter().enumerate() {
            let sm_id = sm_ids
                .get(ix)
                .copied()
                .and_then(|value| u32::try_from(value).ok())
                .unwrap_or(u32::MAX);
            let due = response_due
                .response_entries(ix)
                .into_iter()
                .map(|((warp, instruction, sector), time, generation)| {
                    (warp, instruction, sector, time, generation)
                })
                .collect::<Vec<_>>();
            let owners = core.typed_l2_request_owners(sm_id, &due);
            typed_l2.request_owners.extend(owners.clone());
            owners_by_core.push(owners);
        }
        let ready_by_sector = pending_l2_ready
            .typed_entries()
            .into_iter()
            .map(|(line, generation, sector, ready, producer)| {
                ((line, generation, sector), (ready, producer))
            })
            .collect::<BTreeMap<_, _>>();
        typed_l2.request_owners.sort_by(|left, right| {
            (
                left.request.sm_id,
                left.request.response_warp,
                left.request.warp_slot,
                left.request.instruction,
                left.request.sector_index,
                &left.request.response_kind,
            )
                .cmp(&(
                    right.request.sm_id,
                    right.request.response_warp,
                    right.request.warp_slot,
                    right.request.instruction,
                    right.request.sector_index,
                    &right.request.response_kind,
                ))
        });
        let mut owners_by_line = BTreeMap::<_, Vec<_>>::new();
        for owner in &typed_l2.request_owners {
            owners_by_line.entry(&owner.line).or_default().push(owner);
        }
        let mut ownership_unknown = Vec::new();
        // A response owner may be observed before the L2 event that prices
        // its request. Record that ambiguity even when no live allocation
        // bucket exists; otherwise the root could claim complete ownership
        // for an orphan L1 response and the structural validator would have
        // to reject an otherwise valid diagnostic boundary.
        for owner in &typed_l2.request_owners {
            if matches!(owner.line, sim_semantics::CanonicalLineKey::Unknown { .. }) {
                continue;
            }
            let Some(generation) = owner.allocation_generation else {
                ownership_unknown.push(format!(
                    "request_allocation_generation_unknown:{:?}",
                    owner.request
                ));
                continue;
            };
            if !typed_l2.allocations.iter().any(|allocation| {
                allocation.line == owner.line && allocation.generation == generation
            }) {
                ownership_unknown.push(format!(
                    "request_allocation_missing:{:?}:{}",
                    owner.line, generation
                ));
            }
        }
        for allocation in &mut typed_l2.allocations {
            let line_owners = owners_by_line.get(&allocation.line);
            if line_owners
                .into_iter()
                .flatten()
                .any(|owner| owner.allocation_generation.is_none())
            {
                ownership_unknown.push(format!(
                    "request_allocation_generation_unknown:{:?}:{}",
                    allocation.line, allocation.generation
                ));
            }
            for sector in &mut allocation.sectors {
                if let sim_semantics::CanonicalLineKey::VirtualAbsolute { line_id } =
                    &allocation.line
                {
                    sector.ready_at = ready_by_sector
                        .get(&(*line_id, allocation.generation, sector.sector))
                        .map(|(ready, _)| *ready);
                }
                if !matches!(sector.status, sim_semantics::L2SectorStatus::Pending) {
                    continue;
                }
                let bit = 1_u8 << sector.sector;
                // Buckets retain the canonical requester order established
                // above. A sector only visits requesters of its own line.
                let consumers = line_owners
                    .into_iter()
                    .flatten()
                    .filter(|owner| {
                        owner.allocation_generation == Some(allocation.generation)
                            && owner.sector_mask & bit != 0
                    })
                    .map(|owner| owner.request.clone())
                    .collect::<Vec<_>>();
                sector.consumers = consumers;
                let producer = match &allocation.line {
                    sim_semantics::CanonicalLineKey::VirtualAbsolute { line_id } => ready_by_sector
                        .get(&(*line_id, allocation.generation, sector.sector))
                        .and_then(|(_, producer)| producer.clone()),
                    _ => None,
                };
                sector.producer = producer
                    .as_ref()
                    .filter(|request| sector.consumers.iter().any(|consumer| consumer == *request))
                    .cloned();
                if producer.is_some() && sector.producer.is_none() {
                    ownership_unknown.push(format!(
                        "pending_sector_producer_not_consumer:{:?}:{}:{}",
                        allocation.line, allocation.generation, sector.sector
                    ));
                }
                if sector.producer.is_none() {
                    ownership_unknown.push(format!(
                        "pending_sector_without_producer:{:?}:{}:{}",
                        allocation.line, allocation.generation, sector.sector
                    ));
                }
            }
        }
        // A pending-ready ledger can outlive its resident allocation after an
        // eviction. Preserve that stale-fill ambiguity explicitly; otherwise
        // a queue fill with no live L2 allocation could still obtain a
        // complete typed projection.
        for (line_id, generation, sector) in ready_by_sector.keys() {
            let represented_pending = typed_l2.allocations.iter().any(|allocation| {
                matches!(
                    allocation.line,
                    sim_semantics::CanonicalLineKey::VirtualAbsolute { line_id: candidate }
                        if candidate == *line_id
                ) && allocation.generation == *generation
                    && allocation.sectors.iter().any(|state| {
                        state.sector == *sector
                            && matches!(state.status, sim_semantics::L2SectorStatus::Pending)
                    })
            });
            if !represented_pending {
                ownership_unknown.push(format!(
                    "pending_fill_allocation_unknown:{}:{}:{}",
                    line_id, generation, sector
                ));
            }
        }
        // Response owners and queued response events are observed through
        // different internal lanes. A due owner without a matching queued
        // response (or vice versa) is unresolved boundary state, not a
        // structural contradiction. Mark the ownership domain incomplete so
        // the typed contract widens/refuses instead of aborting the replay.
        type ResponseQueueKey = (u32, u64, u64, u64, TypedResponseKind, u64);
        let mut owner_counts = BTreeMap::<ResponseQueueKey, u32>::new();
        for owner in &typed_l2.request_owners {
            let Some(due) = owner.due else { continue };
            let key = (
                owner.request.sm_id,
                owner.request.response_warp,
                owner.request.instruction,
                owner.request.sector_index,
                owner.request.response_kind,
                due,
            );
            *owner_counts.entry(key).or_default() += 1;
        }
        let mut queue_counts = BTreeMap::<ResponseQueueKey, u32>::new();
        for pending in &typed_queue.pending {
            let TypedEventPayload::MemoryResponse {
                reply: Some(reply), ..
            } = &pending.payload
            else {
                continue;
            };
            let Some(sm_id) = sm_ids
                .get(usize::try_from(pending.target).unwrap_or(usize::MAX))
                .and_then(|sm_id| u32::try_from(*sm_id).ok())
            else {
                ownership_unknown.push(format!(
                    "response_owner_queue_target_unknown:{}",
                    pending.target
                ));
                continue;
            };
            let key = (
                sm_id,
                reply.warp,
                reply.instruction,
                reply.sector_index,
                reply.response_kind,
                pending.time,
            );
            *queue_counts.entry(key).or_default() += 1;
        }
        let mut response_keys = BTreeSet::new();
        response_keys.extend(owner_counts.keys().copied());
        response_keys.extend(queue_counts.keys().copied());
        for key in response_keys {
            if owner_counts.get(&key) != queue_counts.get(&key) {
                ownership_unknown.push(format!("response_owner_queue_join_unknown:{key:?}"));
            }
        }
        ownership_unknown.sort();
        ownership_unknown.dedup();
        typed_l2.unknown_state = ownership_unknown;
        typed_l2.ownership_complete = typed_l2.unknown_state.is_empty();
        typed_l2
            .validate()
            .map_err(sim_mem::L2ProjectionError::from)?;
        let typed_dram = dram_server
            .map(DramServer::typed_boundary_state)
            .transpose()?
            .unwrap_or(DramBoundaryState::LatencyOnly);
        let typed_cores = snapshots
            .iter()
            .enumerate()
            .map(|(ix, snapshot)| TypedCoreBoundaryState {
                sm_id: sm_ids
                    .get(ix)
                    .copied()
                    .and_then(|value| u32::try_from(value).ok())
                    .unwrap_or(u32::MAX),
                cycle: snapshot.cycle,
                issued: snapshot.issued,
                resident_warps: snapshot.resident_warps,
                resident_shared_memory_bytes: snapshot.resident_shared_memory_bytes,
                unfinished_warps: snapshot.unfinished_warps,
                next_block: snapshot.next_block,
                hold: snapshot.hold,
                subcore_warps: snapshot.subcore_warps.clone(),
                outstanding_responses: snapshot.outstanding_responses,
                barrier_pipe_ready: snapshot.barrier_pipe_ready,
                tensor_pipe_ready: snapshot.tensor_pipe_ready,
                ldgsts_destination_pipe_ready: snapshot.ldgsts_destination_pipe_ready,
                warp_states: snapshot.warp_states.clone(),
                block_states: snapshot.block_states.clone(),
                l1: typed_l1_states[ix].clone(),
                front_end: typed_front_end_states[ix].clone(),
                memory_pipeline: typed_memory_pipeline_states[ix].clone(),
                scalar_pipeline: typed_scalar_pipeline_states[ix].clone(),
                local_completion_time: cores[ix].typed_local_completion_time(),
                unprojected_timing_state: cores[ix].typed_unprojected_timing_state(),
                async_copy: typed_async_copy_states[ix].clone(),
                response_owners: owners_by_core[ix].clone(),
            })
            .collect::<Vec<_>>();
        let typed_l2_power_completeness =
            TypedObservationContract::cycle_exact_power().validate(&typed_l2, power, &typed_cores);
        let typed_boundary = TypedEventBoundaryRecord {
            schema_version: TYPED_STATE_SCHEMA_VERSION,
            contract: TypedObservationContract::cycle_exact_power(),
            event: TypedEventIdentity {
                time,
                sequence,
                sequence_span,
                phase: 0,
                target,
                payload: typed_payload.clone(),
            },
            queue: typed_queue.clone(),
            cores: typed_cores,
            shared_l2: typed_l2.clone(),
            dram: typed_dram,
            power: power.clone(),
            completeness: typed_l2_power_completeness.clone(),
        };
        typed_boundary.validate_structure()?;
        if boundary_phase == "pre"
            && self
                .pending_region_end
                .as_ref()
                .is_some_and(|expected| typed_boundary.event.time >= expected.event.time)
        {
            let expected = self
                .pending_region_end
                .take()
                .expect("pending region-end check was present");
            if typed_boundary.event != expected.event {
                return Err(ReferenceTraceError::Witness(
                    "typed macro continuation event did not reach its declared region end"
                        .to_owned(),
                ));
            }
            if !matches!(
                compare_native_boundary_projection(&expected, &typed_boundary, &expected.contract,),
                sim_semantics::NativeBoundaryRefinementDecision::Preserve
            ) {
                return Err(ReferenceTraceError::Witness(
                    "typed macro continuation boundary disagrees with its declared exit".to_owned(),
                ));
            }
        }
        if self.issue_effect_cursors.len() < cores.len() {
            self.issue_effect_cursors.resize(cores.len(), 0);
        }
        let effect_sequence_base = self
            .pending_typed_entry
            .as_ref()
            .map(|entry| entry.event.sequence)
            .unwrap_or(sequence);
        let mut detailed_internal_effects = Vec::new();
        let mut detailed_effect_index = 0_u64;
        for (ix, core) in cores.iter().enumerate() {
            let start = self.issue_effect_cursors[ix];
            let effects = core.reference_issue_effects_since(start);
            self.issue_effect_cursors[ix] = start.saturating_add(effects.len());
            for &(effect_time, _warp, _instruction) in effects {
                let effect_sequence = effect_sequence_base
                    .checked_add(detailed_effect_index.saturating_add(1))
                    .unwrap_or(u64::MAX);
                detailed_effect_index = detailed_effect_index.saturating_add(1);
                detailed_internal_effects.push(TypedEventTransition {
                    phase: 0,
                    event: TypedEventIdentity {
                        time: effect_time,
                        sequence: effect_sequence,
                        sequence_span: 1,
                        phase: 0,
                        target: sm_ids
                            .get(ix)
                            .and_then(|id| u32::try_from(*id).ok())
                            .unwrap_or(u32::MAX),
                        payload: TypedEventPayload::SmWake,
                    },
                });
            }
        }
        let missing_macro_decision =
            compare_native_region_contract(None, &typed_boundary, &typed_boundary);
        let native_boundary_refinement = if boundary_phase == "pre" {
            self.pending_typed_entry = Some(typed_boundary.clone());
            let witness = self
                .witness
                .clone()
                .filter(|witness| witness.target == target && witness.start_sequence == sequence);
            if let Some(witness) = witness {
                match execute_typed_scalar_macro(
                    &typed_boundary,
                    &witness.spec,
                    &witness.detailed_internal_effects,
                    &witness.detailed_labels,
                    &witness.detailed_exit,
                ) {
                    Ok(result) => {
                        let decision = Self::native_result_decision(&result);
                        self.pending_native_result = Some(result.clone());
                        self.active_native_result = Some(result);
                        decision
                    }
                    Err(error) => sim_semantics::NativeBoundaryRefinementDecision::Reify {
                        reason: sim_semantics::NativeBoundaryReifyReason::SummaryStructure {
                            error: error.to_string(),
                        },
                    },
                }
            } else {
                missing_macro_decision
            }
        } else if boundary_phase == "post" {
            let entry = self.pending_typed_entry.take();
            if let Some(result) = self.active_native_result.take() {
                Self::native_result_decision(&result)
            } else if let (Some(entry), Some(spec)) = (entry, typed_scalar_macro_spec_from_env()) {
                match execute_typed_scalar_macro(
                    &entry,
                    &spec,
                    &detailed_internal_effects,
                    &[event.to_owned()],
                    &typed_boundary,
                ) {
                    Ok(NativeBoundaryExecutionResult::TypedMacro { .. }) => {
                        sim_semantics::NativeBoundaryRefinementDecision::Preserve
                    }
                    Ok(NativeBoundaryExecutionResult::ExactReplay { reason, .. }) => {
                        sim_semantics::NativeBoundaryRefinementDecision::Reify { reason }
                    }
                    Err(error) => sim_semantics::NativeBoundaryRefinementDecision::Reify {
                        reason: sim_semantics::NativeBoundaryReifyReason::SummaryStructure {
                            error: error.to_string(),
                        },
                    },
                }
            } else {
                missing_macro_decision
            }
        } else {
            missing_macro_decision
        };
        let native_boundary_execution = native_boundary_execution_path(&native_boundary_refinement);
        // `typed_boundary` is the canonical state. The old formatted core/L2
        // projections were diagnostic duplicates and made every boundary much
        // larger; the versioned root retains all typed fields and the legacy
        // completeness flag remains in the header for old consumers.
        self.write_event(serde_json::json!({
            "kind": "event",
            "time": time,
            "sequence": sequence,
            "target": target,
            "event": event,
            "typed_boundary": typed_boundary,
            "detailed_internal_effects": detailed_internal_effects,
            "native_boundary_refinement": native_boundary_refinement,
            "native_boundary_execution": native_boundary_execution,
            "boundary_phase": boundary_phase,
        }))
    }

    fn finish(
        mut self,
        queue: sim_engine::QueueStats,
        power: &PowerBoundaryState,
    ) -> Result<(), ReferenceTraceError> {
        if self.pending_region_end.is_some() {
            return Err(ReferenceTraceError::Witness(
                "typed macro continuation ended before its declared region end".to_owned(),
            ));
        }
        // `max_records` bounds data records. The footer is metadata emitted
        // after that bound so a run that ends exactly at the limit remains a
        // valid lossless stream.
        serde_json::to_writer(
            &mut self.writer,
            &serde_json::json!({
                "kind": "footer",
                "records_before_footer": self.records,
                "queue_events_popped": queue.events_popped,
                "queue_clock_gap_cycles": queue.clock_gap_cycles,
                "power_state": power,
                "state_projection_complete": false,
                "native_macro_dispatches": self.native_macro_dispatches,
                "native_macro_fallbacks": self.native_macro_fallbacks,
            }),
        )?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()?;
        Ok(())
    }
}

enum L2AccessEventRequests {
    Single([MemRequest; 1]),
    Batch(Box<[MemRequest]>),
}

impl AsRef<[MemRequest]> for L2AccessEventRequests {
    fn as_ref(&self) -> &[MemRequest] {
        match self {
            Self::Single(request) => request,
            Self::Batch(requests) => requests,
        }
    }
}

#[cfg(feature = "performance-counters")]
#[derive(Debug, Default)]
struct EventPerformanceCounters {
    queue_events_processed: u64,
    processed: u64,
    sm_wake: u64,
    l2_access: u64,
    l2_access_batches: u64,
    l2_access_batched_requests: u64,
    l2_fill: u64,
    atomic_completion: u64,
    writeback_completion: u64,
    coherence_completion: u64,
    l1_visibility_event: u64,
    l1_payload_admission: u64,
    memory_write_ack: u64,
    mem_response: u64,
    same_cycle_after_first: u64,
    queue_same_cycle_after_first: u64,
    maximum_queue_depth: u64,
    previous_time: Option<u64>,
}

#[cfg(feature = "performance-counters")]
impl EventPerformanceCounters {
    fn observe(
        &mut self,
        event: &Ev,
        time: u64,
        queue_depth_before_pop: usize,
        logical_events: usize,
    ) -> Result<(), SystemError> {
        self.queue_events_processed =
            self.queue_events_processed
                .checked_add(1)
                .ok_or(SystemError::CounterOverflow {
                    counter: "performance queue events processed",
                })?;
        let logical_events =
            u64::try_from(logical_events).map_err(|_| SystemError::CounterOverflow {
                counter: "performance logical event multiplicity",
            })?;
        self.processed =
            self.processed
                .checked_add(logical_events)
                .ok_or(SystemError::CounterOverflow {
                    counter: "performance events processed",
                })?;
        match event {
            Ev::L2AccessBatch(_) => {
                self.l2_access_batches =
                    self.l2_access_batches
                        .checked_add(1)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "performance L2 access batches",
                        })?;
                self.l2_access_batched_requests = self
                    .l2_access_batched_requests
                    .checked_add(logical_events)
                    .ok_or(SystemError::CounterOverflow {
                        counter: "performance batched L2 access requests",
                    })?;
                self.l2_access = self.l2_access.checked_add(logical_events).ok_or(
                    SystemError::CounterOverflow {
                        counter: "performance L2 access requests",
                    },
                )?;
            }
            Ev::L2Access(_) => {
                debug_assert_eq!(logical_events, 1);
                self.l2_access =
                    self.l2_access
                        .checked_add(1)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "performance L2 access requests",
                        })?;
            }
            Ev::SmWake
            | Ev::L1PayloadAdmission { .. }
            | Ev::L2Fill(_)
            | Ev::AtomicCompletion
            | Ev::L2WritebackComplete { .. }
            | Ev::L2CoherenceComplete { .. }
            | Ev::L1VisibilityComplete { .. }
            | Ev::L1InvalidateAll(_)
            | Ev::L1VisibilityAccess(_)
            | Ev::L1VisibilityFence(_)
            | Ev::L1FillVisible(_)
            | Ev::MemoryWriteAck(_)
            | Ev::MemResponse { .. } => {
                debug_assert_eq!(logical_events, 1);
                let kind = match event {
                    Ev::SmWake => &mut self.sm_wake,
                    Ev::L1PayloadAdmission { .. } => &mut self.l1_payload_admission,
                    Ev::L2Fill(_) => &mut self.l2_fill,
                    Ev::AtomicCompletion => &mut self.atomic_completion,
                    Ev::L2WritebackComplete { .. } => &mut self.writeback_completion,
                    Ev::L2CoherenceComplete { .. } => &mut self.coherence_completion,
                    Ev::L1VisibilityComplete { .. }
                    | Ev::L1InvalidateAll(_)
                    | Ev::L1VisibilityAccess(_)
                    | Ev::L1VisibilityFence(_)
                    | Ev::L1FillVisible(_) => &mut self.l1_visibility_event,
                    Ev::MemoryWriteAck(_) => &mut self.memory_write_ack,
                    Ev::MemResponse { .. } => &mut self.mem_response,
                    Ev::L2Access(_) | Ev::L2AccessBatch(_) => unreachable!(),
                };
                *kind = kind.checked_add(1).ok_or(SystemError::CounterOverflow {
                    counter: "performance event-kind count",
                })?;
            }
        }
        if self.previous_time == Some(time) {
            self.queue_same_cycle_after_first = self
                .queue_same_cycle_after_first
                .checked_add(1)
                .ok_or(SystemError::CounterOverflow {
                    counter: "performance same-cycle queue events",
                })?;
            self.same_cycle_after_first = self
                .same_cycle_after_first
                .checked_add(logical_events)
                .ok_or(SystemError::CounterOverflow {
                counter: "performance same-cycle events",
            })?;
        } else {
            self.same_cycle_after_first = self
                .same_cycle_after_first
                .checked_add(logical_events.saturating_sub(1))
                .ok_or(SystemError::CounterOverflow {
                    counter: "performance same-cycle events",
                })?;
        }
        self.previous_time = Some(time);
        self.maximum_queue_depth =
            self.maximum_queue_depth
                .max(u64::try_from(queue_depth_before_pop).map_err(|_| {
                    SystemError::CounterOverflow {
                        counter: "performance event-queue depth",
                    }
                })?);
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct PendingL2ReadyEntry {
    ready: u64,
    #[cfg(feature = "reference")]
    producer: Option<L2RequestKey>,
}

#[derive(Default)]
struct PendingL2Ready(HashMap<L2PendingSectorKey, PendingL2ReadyEntry>);

impl PendingL2Ready {
    fn merged_ready(&self, preview: L2EventMaskPreview) -> Result<Option<u64>, SystemError> {
        let mut latest = None;
        for sector in 0..4 {
            let Some(key) = preview.pending_sector_key(sector) else {
                continue;
            };
            let ready = self
                .0
                .get(&key)
                .map(|entry| entry.ready)
                .ok_or(SystemError::PendingL2ReadyMissing { key })?;
            latest = Some(latest.map_or(ready, |current: u64| current.max(ready)));
        }
        Ok(latest)
    }

    fn preflight_register(&self, preview: L2EventMaskPreview) -> Result<(), SystemError> {
        for sector in 0..4 {
            let Some(key) = preview.fill_sector_key(sector) else {
                continue;
            };
            if let Some(entry) = self.0.get(&key) {
                return Err(SystemError::PendingL2ReadyAlreadyRegistered {
                    key,
                    ready: entry.ready,
                });
            }
        }
        Ok(())
    }

    #[cfg(not(feature = "reference"))]
    fn register(&mut self, fill: L2FillToken, ready: u64) -> Result<(), SystemError> {
        self.register_entry(
            fill,
            PendingL2ReadyEntry {
                ready,
                #[cfg(feature = "reference")]
                producer: None,
            },
        )
    }

    #[cfg(feature = "reference")]
    fn register_with_producer(
        &mut self,
        fill: L2FillToken,
        ready: u64,
        producer: Option<L2RequestKey>,
    ) -> Result<(), SystemError> {
        self.register_entry(fill, PendingL2ReadyEntry { ready, producer })
    }

    fn register_entry(
        &mut self,
        fill: L2FillToken,
        entry: PendingL2ReadyEntry,
    ) -> Result<(), SystemError> {
        for sector in 0..4 {
            let Some(key) = fill.pending_sector_key(sector) else {
                continue;
            };
            if let Some(existing) = self.0.get(&key) {
                return Err(SystemError::PendingL2ReadyAlreadyRegistered {
                    key,
                    ready: existing.ready,
                });
            }
        }
        for sector in 0..4 {
            if let Some(key) = fill.pending_sector_key(sector) {
                self.0.insert(key, entry.clone());
            }
        }
        Ok(())
    }

    fn preflight_remove(&self, fill: L2FillToken, now: u64) -> Result<(), SystemError> {
        for sector in 0..4 {
            let Some(key) = fill.pending_sector_key(sector) else {
                continue;
            };
            let ready = self
                .0
                .get(&key)
                .map(|entry| entry.ready)
                .ok_or(SystemError::PendingL2ReadyRemovalMissing { key })?;
            if ready != now {
                return Err(SystemError::PendingL2ReadyTimeMismatch {
                    key,
                    expected: ready,
                    got: now,
                });
            }
        }
        Ok(())
    }

    fn commit_remove(&mut self, fill: L2FillToken) {
        for sector in 0..4 {
            if let Some(key) = fill.pending_sector_key(sector) {
                let removed = self.0.remove(&key);
                debug_assert!(removed.is_some(), "pending L2 removal was preflighted");
            }
        }
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    #[cfg(feature = "reference")]
    fn typed_entries(&self) -> Vec<(i64, u64, u8, u64, Option<L2RequestKey>)> {
        let mut entries = self
            .0
            .iter()
            .map(|(key, entry)| {
                let identity = key.boundary_identity();
                (
                    identity.line_id,
                    identity.generation,
                    identity.sector,
                    entry.ready,
                    entry.producer.clone(),
                )
            })
            .collect::<Vec<_>>();
        entries.sort_unstable();
        entries
    }
}

struct L2ResponseCommit {
    ready: u64,
    origin_ix: usize,
    next_delayed: u64,
    next_origin_delayed: u64,
    pending_bounds_removal: PendingRemoval,
}

#[derive(Debug, Clone)]
struct PendingAdmissionInstruction {
    witness: AdmissionRequestWitness,
    operation: MemoryOperation,
    issue_time: u64,
    next_request_index: u16,
    hit_sectors: u64,
    miss_sectors: u64,
    regions: Vec<i64>,
}

#[derive(Debug, Clone, Copy)]
struct CompletedAdmissionInstruction {
    witness: AdmissionRequestWitness,
    operation: MemoryOperation,
    issue_time: u64,
    hit_sectors: u64,
    miss_sectors: u64,
    l2_requests: u64,
    unique_regions: u64,
}

#[derive(Default)]
struct AdmissionInstructionObserver {
    pending: Option<PendingAdmissionInstruction>,
}

impl AdmissionInstructionObserver {
    #[allow(clippy::too_many_arguments)]
    fn observe(
        &mut self,
        sm_id: i32,
        witness: AdmissionRequestWitness,
        operation: MemoryOperation,
        issue_time: u64,
        sectors: u64,
        hits: u64,
        sector_id: i64,
    ) -> Result<Option<CompletedAdmissionInstruction>, SystemError> {
        if hits > sectors {
            return Err(SystemError::AdmissionObservationHits {
                sm_id,
                sectors,
                hits,
            });
        }
        if witness.request_index() == 0 {
            if self.pending.is_some() {
                return Err(SystemError::AdmissionObservationInterleaved {
                    sm_id,
                    warp: witness.warp(),
                    instruction: witness.instruction(),
                });
            }
            self.pending = Some(PendingAdmissionInstruction {
                witness,
                operation,
                issue_time,
                next_request_index: 0,
                hit_sectors: 0,
                miss_sectors: 0,
                regions: Vec::new(),
            });
        }
        let pending = self
            .pending
            .as_mut()
            .ok_or(SystemError::AdmissionObservationSequence {
                sm_id,
                warp: witness.warp(),
                instruction: witness.instruction(),
                expected: 0,
                got: witness.request_index(),
            })?;
        if pending.witness.warp() != witness.warp()
            || pending.witness.instruction() != witness.instruction()
            || pending.witness.subcore() != witness.subcore()
            || pending.witness.request_count() != witness.request_count()
            || pending.operation != operation
            || pending.issue_time != issue_time
        {
            return Err(SystemError::AdmissionObservationIdentity {
                sm_id,
                warp: witness.warp(),
                instruction: witness.instruction(),
            });
        }
        if pending.next_request_index != witness.request_index() {
            return Err(SystemError::AdmissionObservationSequence {
                sm_id,
                warp: witness.warp(),
                instruction: witness.instruction(),
                expected: pending.next_request_index,
                got: witness.request_index(),
            });
        }
        pending.hit_sectors =
            pending
                .hit_sectors
                .checked_add(hits)
                .ok_or(SystemError::CounterOverflow {
                    counter: "memory-admission observed hit sectors",
                })?;
        pending.miss_sectors = pending.miss_sectors.checked_add(sectors - hits).ok_or(
            SystemError::CounterOverflow {
                counter: "memory-admission observed miss sectors",
            },
        )?;
        let region = sector_id.div_euclid(8);
        if !pending.regions.contains(&region) {
            pending.regions.push(region);
        }
        pending.next_request_index =
            pending
                .next_request_index
                .checked_add(1)
                .ok_or(SystemError::CounterOverflow {
                    counter: "memory-admission request index",
                })?;
        if pending.next_request_index == witness.request_count() {
            let completed = self
                .pending
                .take()
                .expect("completed admission observation owns pending state");
            return Ok(Some(CompletedAdmissionInstruction {
                witness: completed.witness,
                operation: completed.operation,
                issue_time: completed.issue_time,
                hit_sectors: completed.hit_sectors,
                miss_sectors: completed.miss_sectors,
                l2_requests: u64::from(completed.next_request_index),
                unique_regions: u64::try_from(completed.regions.len()).map_err(|_| {
                    SystemError::CounterOverflow {
                        counter: "memory-admission unique completion regions",
                    }
                })?,
            }));
        }
        Ok(None)
    }
}

struct L2SystemCommit {
    source_ix: usize,
    source_sectors: u64,
    source_hits: u64,
    origin_source_sectors: [u64; 2],
    origin_source_hits: [u64; 2],
    read_sectors: u64,
    atomic_rmw_read_sectors: u64,
    reduction_rmw_read_sectors: u64,
    origin_read_sectors: [u64; 2],
    topology_stats: DramStats,
    dram_free_fp: Option<u128>,
    read_service: Option<ServicePlan>,
    writeback: Option<ServicePlan>,
    coherence: Option<CoherencePlan>,
    fill_ready: Option<u64>,
    /// Shared-L2 allocation generation associated with a returning response.
    /// The L1 fill generation is a separate identity domain and must not be
    /// used for this join.
    response_l2_allocation_generation: Option<u64>,
    atomic_completion_ready: Option<u64>,
    response: Option<L2ResponseCommit>,
    observed_sectors: u64,
    observed_hits: u64,
    /// Observer-only read/atomic contribution. Store/reduction packetization
    /// is reconstructed from the issuing instruction topology after the event
    /// stream drains, so it never changes modeled L2 state or response time.
    lookup_observer_requests: u64,
    lookup_observer_hits: u64,
    read_lookup_requests: u64,
    read_lookup_hits: u64,
    read_pending_merge_hits: u64,
    store_lookup_requests: u64,
    store_lookup_hits: u64,
    store_packet_lookup_hits: u64,
    store_packet_resident_lookup_hits: u64,
    atomic_lookup_requests: u64,
    atomic_lookup_hits: u64,
    reduction_lookup_requests: u64,
    reduction_lookup_hits: u64,
    /// Source-qualified sectored fill-ownership observer. A pending merge is
    /// a lookup miss that reuses an older allocation's ready frontier; a new
    /// fill owns a new lower-level fetch capability. These values never alter
    /// service timing or cache state.
    pending_merge_sectors: u64,
    new_fill_sectors: u64,
    pending_pressure_samples: u64,
    pending_pressure_sum: u64,
    l2_frontier_samples: u64,
    l2_issue_to_fill_cycles_sum: u64,
    l2_issue_to_fill_cycles_max: u64,
    l2_issue_to_response_cycles_sum: u64,
    l2_issue_to_response_cycles_max: u64,
    l2_service_delay_cycles_sum: u64,
    l2_service_delay_cycles_max: u64,
    client_scope_ix: usize,
    client_scope_requests: u64,
    client_scope_resident_hits: u64,
    client_scope_pending_merges: u64,
    client_scope_fill_misses: u64,
    ltc128b_shadow_promoted_sectors: u64,
}

/// Aggregate DRAM read-bandwidth model (the §6.3 "DRAM
/// bandwidth-latency" increment). The configured carded rate is a model
/// covariate: the corpus runner defaults to the independently measured
/// 825 GB/s effective-rate hypothesis, while direct `sim-run` without an
/// override falls back to the architecture-card peak. Neither value is a
/// physical controller claim; the model is one G/D/1 byte server over the shared hierarchy
/// read-miss sector stream: each fetching sector demands 32 B, and a
/// reply-carrying request or intrinsically returning ATOM cannot complete
/// before BOTH its latency arm and the server's completion of ITS OWN request's
/// bytes. Late L1 merges carry their own sector response, so the waiting warp
/// clears at that child sector's ready event. Uncongested streams see pure
/// latency (the server only ever ADDS delay).
///
/// Default `Carded` scope remains DRAM READS only — write data validates Store
/// sectors without a fetch, while returning ATOM and non-returning RED charge
/// missing sectors for their internal RMW read. The opt-in `CardedWriteback`
/// variant adds a bounded dirty-sector service lane; coherence ownership and
/// visibility, and RED's value-dependent serialization, remain typed tails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DramBw {
    /// Latency-only (pre-increment behavior): the ablation mode and the
    /// event-vs-const equivalence-differential mode — the constant path
    /// has no bandwidth concept, so equivalence is proven here.
    Unmodeled { arch: ArchId },
    /// Configured carded aggregate service rate; runner/default provenance is
    /// carried by the direct-run configuration rather than inferred here.
    Carded {
        arch: ArchId,
        bytes_per_sec: u64,
        sm_clock_hz: u64,
    },
    /// Opt-in carded service with transactional dirty-sector writeback.
    /// Capacity is measured in 32-byte dirty sectors; this remains a bounded
    /// native replay experiment, not a coherence/ownership protocol.
    CardedWriteback {
        arch: ArchId,
        bytes_per_sec: u64,
        sm_clock_hz: u64,
        capacity_sectors: u32,
    },
}

impl DramBw {
    pub const fn arch(self) -> ArchId {
        match self {
            Self::Unmodeled { arch }
            | Self::Carded { arch, .. }
            | Self::CardedWriteback { arch, .. } => arch,
        }
    }

    pub const fn writeback_capacity_sectors(self) -> Option<u32> {
        match self {
            Self::CardedWriteback {
                capacity_sectors, ..
            } => Some(capacity_sectors),
            _ => None,
        }
    }

    pub const fn service_rate_bytes_per_sec(self) -> Option<u64> {
        match self {
            Self::Carded { bytes_per_sec, .. } | Self::CardedWriteback { bytes_per_sec, .. } => {
                Some(bytes_per_sec)
            }
            Self::Unmodeled { .. } => None,
        }
    }
}

fn writeback_queue_unavailable(dram: DramBw) -> bool {
    !matches!(dram, DramBw::CardedWriteback { .. })
}

/// The G/D/1 byte server. Fixed-point time in units of
/// `cycle × bytes_per_sec` (u128): serving `b` bytes takes
/// `b × sm_clock_hz` fp-units (= b × Hz / Bps cycles), so no division
/// happens until the final ceil back to cycles — bit-exact and
/// deterministic in queue order.
struct DramServer {
    free_fp: u128,
    bytes_per_sec: u128,
    sm_clock_hz: u128,
}

impl DramServer {
    #[cfg(feature = "reference")]
    fn typed_boundary_state(&self) -> Result<DramBoundaryState, DramStateError> {
        DramBoundaryState::from_fixed_point(
            u64::try_from(self.bytes_per_sec).map_err(|_| DramStateError::InvalidRate)?,
            u64::try_from(self.sm_clock_hz).map_err(|_| DramStateError::InvalidRate)?,
            self.free_fp,
        )
    }

    /// The caller (`simulate_kernel_shared`) rejects zero rates with a
    /// typed error before construction — no clamping here (a clamp
    /// would turn a bad card into a silently absurd 1 B/s server and
    /// arm the fixed-point u64 ceiling on large traces; gate finding
    /// 2026-07-07).
    fn new(bw: DramBw) -> Option<Self> {
        match bw {
            DramBw::Unmodeled { .. } => None,
            DramBw::Carded {
                bytes_per_sec,
                sm_clock_hz,
                ..
            } => Some(Self {
                free_fp: 0,
                bytes_per_sec: u128::from(bytes_per_sec),
                sm_clock_hz: u128::from(sm_clock_hz),
            }),
            // The opt-in writeback path owns the shared byte server through
            // `WritebackQueue`; no independent read server is created here.
            DramBw::CardedWriteback { .. } => None,
        }
    }

    /// Compute a service completion without mutating the server. The
    /// caller uses this to preflight the request's exact outcome-dependent
    /// fetch before any L2 or accounting state commits.
    fn preview(&self, arrival: u64, bytes: u64) -> Result<(u128, u64), SystemError> {
        let arrival_fp = u128::from(arrival).checked_mul(self.bytes_per_sec).ok_or(
            SystemError::DramFixedPointOverflow {
                operation: "arrival scaling",
            },
        )?;
        let service_fp = u128::from(bytes).checked_mul(self.sm_clock_hz).ok_or(
            SystemError::DramFixedPointOverflow {
                operation: "service scaling",
            },
        )?;
        let start = arrival_fp.max(self.free_fp);
        let done = start
            .checked_add(service_fp)
            .ok_or(SystemError::DramFixedPointOverflow {
                operation: "service completion",
            })?;
        let ready_fp = done.div_ceil(self.bytes_per_sec);
        let ready = u64::try_from(ready_fp).map_err(|_| SystemError::DramCycleOverflow {
            completion_fp: done,
            bytes_per_sec: self.bytes_per_sec,
        })?;
        Ok((done, ready))
    }
}

/// One address-space origin's DRAM observer lane.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DramOriginStats {
    pub read_sectors: u64,
    pub bw_delayed_responses: u64,
}

/// Driver-side DRAM traffic and timing-neutral lower-level topology observer.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DramStats {
    /// Architecture provenance bound by the shared-L2 system. Unbound
    /// synthetic stats are rejected before aggregation or emission.
    arch: Option<ArchId>,
    pub service_rate_bytes_per_sec: Option<u64>,
    /// 32 B DRAM read fetches DEMANDED by the run (L2-missing sectors
    /// of read operations plus missing sectors of ATOM and RED internal RMW reads).
    /// Counted in BOTH modes: under `Unmodeled` it audits the ablation's demand
    /// with no server attached. Plain stores never enter this modeled
    /// read-fetch count.
    pub read_sectors: u64,
    /// Missing-sector read demand attributable to returning ATOM operations.
    /// This is a traffic observer, not a claim that the value-dependent RMW
    /// serialization is complete.
    pub atomic_rmw_read_sectors: u64,
    /// Missing-sector read demand attributable to non-returning RED/REDG
    /// operations. The reduction result/coherence protocol remains a typed
    /// tail even though its cold-line read arm is charged here.
    pub reduction_rmw_read_sectors: u64,
    /// Responses whose ready time the bandwidth server pushed PAST the
    /// latency arm (0 in uncongested runs and in `Unmodeled` mode).
    /// Late sector follow-ups carry their own completion response and are
    /// therefore included in the same latency-arm accounting as any other
    /// reply-bearing read.
    pub bw_delayed_responses: u64,
    /// L2 replacement topology. In the default read-only path dirty entries
    /// are candidate writeback work; the opt-in writeback variant also reports
    /// separate completion counters below.
    pub l2_evicted_lines: u64,
    pub l2_evicted_valid_sectors: u64,
    pub l2_evicted_pending_lines: u64,
    pub l2_evicted_pending_sectors: u64,
    pub l2_dirty_eviction_lines: u64,
    pub l2_dirty_evicted_sectors: u64,
    /// Opt-in scheduled writeback completion observations, including eviction
    /// victims and the terminal resident-L2 flush in CardedWriteback mode.
    pub writeback_completed_lines: u64,
    pub writeback_completed_sectors: u64,
    pub writeback_completed_bytes: u64,
    /// Subset of the writeback completion stream sourced from resident L1
    /// dirty sectors; L2 eviction/resident flush counters remain separate.
    pub l1_writeback_completed_lines: u64,
    pub l1_writeback_completed_sectors: u64,
    pub l1_writeback_completed_bytes: u64,
    /// Logical payload-forwarding records validated at terminal L1 flush.
    /// These counters prove model-local byte conservation only; they are not
    /// physical DRAM acknowledgements.
    pub l1_payload_forwarded_sectors: u64,
    pub l1_payload_forwarded_bytes: u64,
    pub l1_payload_forwarding_digest_xor: u64,
    pub writeback_completion_frontier: Option<u64>,
    /// Opt-in directory-service completion observations, separate from
    /// writeback and read-traffic counters.
    pub coherence_completed_transactions: u64,
    pub coherence_completed_sectors: u64,
    pub coherence_completion_frontier: Option<u64>,
    /// Maximum terminal dirty residency across the launches aggregated into
    /// this observer. A gauge is never summed as traffic.
    pub maximum_terminal_l2_dirty_lines: u64,
    pub maximum_terminal_l2_dirty_sectors: u64,
    pub maximum_terminal_l1_dirty_lines: u64,
    pub maximum_terminal_l1_dirty_sectors: u64,
    /// Observer-only effective partition/slice pressure candidates. The
    /// mapping uses line-id modulo/XOR hypotheses; it is not physical GA102
    /// swizzling and never feeds the byte server.
    pub dram_partition_modulo_sectors: [u64; 12],
    pub dram_partition_xor_sectors: [u64; 12],
    pub dram_slice_modulo_sectors: [u64; 6],
    /// Timing-neutral per-request frontier observers. Sums are model-domain
    /// intervals, not hardware completion timestamps or queue claims.
    pub l2_frontier_samples: u64,
    pub l2_issue_to_fill_cycles_sum: u64,
    pub l2_issue_to_fill_cycles_max: u64,
    pub l2_issue_to_response_cycles_sum: u64,
    pub l2_issue_to_response_cycles_max: u64,
    pub l2_service_delay_cycles_sum: u64,
    pub l2_service_delay_cycles_max: u64,
    by_origin: [DramOriginStats; 2],
}

impl DramStats {
    pub fn for_arch(arch: ArchId) -> Self {
        Self {
            arch: Some(arch),
            ..Self::default()
        }
    }

    pub fn with_service_rate(mut self, rate: Option<u64>) -> Self {
        self.service_rate_bytes_per_sec = rate;
        self
    }

    pub fn arch(&self) -> Option<ArchId> {
        self.arch
    }

    pub fn origin(&self, origin: MemoryOrigin) -> DramOriginStats {
        self.by_origin[origin.index()]
    }

    fn with_l2_eviction(mut self, eviction: L2Eviction) -> Result<Self, SystemError> {
        macro_rules! add {
            ($field:ident, $value:expr, $name:literal) => {
                self.$field = self
                    .$field
                    .checked_add($value)
                    .ok_or(SystemError::CounterOverflow { counter: $name })?;
            };
        }
        add!(l2_evicted_lines, 1, "L2 evicted lines");
        add!(
            l2_evicted_valid_sectors,
            u64::from(eviction.valid_mask.count_ones()),
            "L2 evicted valid sectors"
        );
        if eviction.pending_mask != 0 {
            add!(l2_evicted_pending_lines, 1, "L2 evicted pending lines");
            add!(
                l2_evicted_pending_sectors,
                u64::from(eviction.pending_mask.count_ones()),
                "L2 evicted pending sectors"
            );
        }
        if eviction.dirty_mask != 0 {
            add!(l2_dirty_eviction_lines, 1, "L2 dirty eviction lines");
            add!(
                l2_dirty_evicted_sectors,
                u64::from(eviction.dirty_mask.count_ones()),
                "L2 dirty evicted sectors"
            );
        }
        Ok(self)
    }

    fn with_terminal_l2_dirty_residency(
        mut self,
        residency: L2DirtyResidency,
    ) -> Result<Self, SystemError> {
        let lines = u64::try_from(residency.lines).map_err(|_| SystemError::CounterOverflow {
            counter: "terminal L2 dirty resident lines",
        })?;
        let sectors =
            u64::try_from(residency.sectors).map_err(|_| SystemError::CounterOverflow {
                counter: "terminal L2 dirty resident sectors",
            })?;
        self.maximum_terminal_l2_dirty_lines = self.maximum_terminal_l2_dirty_lines.max(lines);
        self.maximum_terminal_l2_dirty_sectors =
            self.maximum_terminal_l2_dirty_sectors.max(sectors);
        Ok(self)
    }

    fn with_terminal_l1_dirty_residency(
        mut self,
        residency: sim_mem::L1DirtyResidency,
    ) -> Result<Self, SystemError> {
        self.maximum_terminal_l1_dirty_lines =
            self.maximum_terminal_l1_dirty_lines.max(residency.lines);
        self.maximum_terminal_l1_dirty_sectors = self
            .maximum_terminal_l1_dirty_sectors
            .max(residency.sectors);
        Ok(self)
    }

    fn with_partition_candidate(mut self, line: i64, sectors: u64) -> Result<Self, SystemError> {
        let line = u64::try_from(line).unwrap_or_default();
        let modulo = usize::try_from(line % 12).unwrap_or(0);
        let xor = usize::try_from((line ^ (line >> 7)) % 12).unwrap_or(0);
        let slice = usize::try_from(line % 6).unwrap_or(0);
        self.dram_partition_modulo_sectors[modulo] = self.dram_partition_modulo_sectors[modulo]
            .checked_add(sectors)
            .ok_or(SystemError::CounterOverflow {
                counter: "DRAM modulo partition candidate sectors",
            })?;
        self.dram_partition_xor_sectors[xor] = self.dram_partition_xor_sectors[xor]
            .checked_add(sectors)
            .ok_or(SystemError::CounterOverflow {
                counter: "DRAM XOR partition candidate sectors",
            })?;
        self.dram_slice_modulo_sectors[slice] = self.dram_slice_modulo_sectors[slice]
            .checked_add(sectors)
            .ok_or(SystemError::CounterOverflow {
                counter: "DRAM slice candidate sectors",
            })?;
        Ok(self)
    }

    /// Build a coherent synthetic observer from explicit origin lanes.
    pub fn from_origins(
        arch: ArchId,
        global: DramOriginStats,
        local: DramOriginStats,
    ) -> Result<Self, SystemCounterError> {
        let read_sectors = global.read_sectors.checked_add(local.read_sectors).ok_or(
            SystemCounterError::DramOriginPartitionOverflow {
                counter: "read sectors",
                global: global.read_sectors,
                local: local.read_sectors,
            },
        )?;
        let bw_delayed_responses = global
            .bw_delayed_responses
            .checked_add(local.bw_delayed_responses)
            .ok_or(SystemCounterError::DramOriginPartitionOverflow {
                counter: "bandwidth-delayed responses",
                global: global.bw_delayed_responses,
                local: local.bw_delayed_responses,
            })?;
        Ok(Self {
            arch: Some(arch),
            service_rate_bytes_per_sec: None,
            read_sectors,
            atomic_rmw_read_sectors: 0,
            reduction_rmw_read_sectors: 0,
            bw_delayed_responses,
            l2_evicted_lines: 0,
            l2_evicted_valid_sectors: 0,
            l2_evicted_pending_lines: 0,
            l2_evicted_pending_sectors: 0,
            l2_dirty_eviction_lines: 0,
            l2_dirty_evicted_sectors: 0,
            writeback_completed_lines: 0,
            writeback_completed_sectors: 0,
            writeback_completed_bytes: 0,
            l1_writeback_completed_lines: 0,
            l1_writeback_completed_sectors: 0,
            l1_writeback_completed_bytes: 0,
            l1_payload_forwarded_sectors: 0,
            l1_payload_forwarded_bytes: 0,
            l1_payload_forwarding_digest_xor: 0,
            writeback_completion_frontier: None,
            coherence_completed_transactions: 0,
            coherence_completed_sectors: 0,
            coherence_completion_frontier: None,
            maximum_terminal_l2_dirty_lines: 0,
            maximum_terminal_l2_dirty_sectors: 0,
            maximum_terminal_l1_dirty_lines: 0,
            maximum_terminal_l1_dirty_sectors: 0,
            dram_partition_modulo_sectors: [0; 12],
            dram_partition_xor_sectors: [0; 12],
            dram_slice_modulo_sectors: [0; 6],
            l2_frontier_samples: 0,
            l2_issue_to_fill_cycles_sum: 0,
            l2_issue_to_fill_cycles_max: 0,
            l2_issue_to_response_cycles_sum: 0,
            l2_issue_to_response_cycles_max: 0,
            l2_service_delay_cycles_sum: 0,
            l2_service_delay_cycles_max: 0,
            by_origin: [global, local],
        })
    }

    /// Explicitly attribute an already-built unified synthetic observer to one
    /// origin. Modeled requests always mutate an explicit lane directly.
    pub fn with_unified_origin(mut self, origin: MemoryOrigin) -> Self {
        let unified = DramOriginStats {
            read_sectors: self.read_sectors,
            bw_delayed_responses: self.bw_delayed_responses,
        };
        self.by_origin = [DramOriginStats::default(); 2];
        self.by_origin[origin.index()] = unified;
        self
    }

    pub fn validate_origin_partition(&self) -> Result<(), SystemCounterError> {
        let global = self.origin(MemoryOrigin::Global);
        let local = self.origin(MemoryOrigin::Local);
        let read_sectors = global.read_sectors.checked_add(local.read_sectors).ok_or(
            SystemCounterError::DramOriginPartitionOverflow {
                counter: "read sectors",
                global: global.read_sectors,
                local: local.read_sectors,
            },
        )?;
        if self.read_sectors != read_sectors {
            return Err(SystemCounterError::DramOriginPartitionMismatch {
                counter: "read sectors",
                unified: self.read_sectors,
                global: global.read_sectors,
                local: local.read_sectors,
            });
        }
        let bw_delayed_responses = global
            .bw_delayed_responses
            .checked_add(local.bw_delayed_responses)
            .ok_or(SystemCounterError::DramOriginPartitionOverflow {
                counter: "bandwidth-delayed responses",
                global: global.bw_delayed_responses,
                local: local.bw_delayed_responses,
            })?;
        if self.bw_delayed_responses != bw_delayed_responses {
            return Err(SystemCounterError::DramOriginPartitionMismatch {
                counter: "bandwidth-delayed responses",
                unified: self.bw_delayed_responses,
                global: global.bw_delayed_responses,
                local: local.bw_delayed_responses,
            });
        }
        Ok(())
    }

    pub fn add_checked(&mut self, other: &DramStats) -> Result<(), SystemCounterError> {
        let expected = self
            .arch
            .ok_or(sim_arch::ArchitectureProvenanceError::Unbound {
                component: "DRAM stats",
            })?;
        let got = other
            .arch
            .ok_or(sim_arch::ArchitectureProvenanceError::Unbound {
                component: "DRAM stats",
            })?;
        if expected != got {
            return Err(sim_arch::ArchitectureProvenanceError::Mismatch {
                component: "DRAM stats",
                expected,
                got,
            }
            .into());
        }
        let read_sectors = self.read_sectors.checked_add(other.read_sectors).ok_or(
            SystemCounterError::Overflow {
                counter: "DRAM read sectors",
            },
        )?;
        let atomic_rmw_read_sectors = self
            .atomic_rmw_read_sectors
            .checked_add(other.atomic_rmw_read_sectors)
            .ok_or(SystemCounterError::Overflow {
                counter: "DRAM atomic RMW read sectors",
            })?;
        let reduction_rmw_read_sectors = self
            .reduction_rmw_read_sectors
            .checked_add(other.reduction_rmw_read_sectors)
            .ok_or(SystemCounterError::Overflow {
                counter: "DRAM reduction RMW read sectors",
            })?;
        let bw_delayed_responses = self
            .bw_delayed_responses
            .checked_add(other.bw_delayed_responses)
            .ok_or(SystemCounterError::Overflow {
                counter: "DRAM bandwidth-delayed responses",
            })?;
        macro_rules! add_topology {
            ($field:ident, $name:literal) => {
                self.$field
                    .checked_add(other.$field)
                    .ok_or(SystemCounterError::Overflow { counter: $name })?
            };
        }
        let l2_evicted_lines = add_topology!(l2_evicted_lines, "L2 evicted lines");
        let l2_evicted_valid_sectors =
            add_topology!(l2_evicted_valid_sectors, "L2 evicted valid sectors");
        let l2_evicted_pending_lines =
            add_topology!(l2_evicted_pending_lines, "L2 evicted pending lines");
        let l2_evicted_pending_sectors =
            add_topology!(l2_evicted_pending_sectors, "L2 evicted pending sectors");
        let l2_dirty_eviction_lines =
            add_topology!(l2_dirty_eviction_lines, "L2 dirty eviction lines");
        let l2_dirty_evicted_sectors =
            add_topology!(l2_dirty_evicted_sectors, "L2 dirty evicted sectors");
        let mut dram_partition_modulo_sectors = [0; 12];
        let mut dram_partition_xor_sectors = [0; 12];
        let mut dram_slice_modulo_sectors = [0; 6];
        for index in 0..12 {
            dram_partition_modulo_sectors[index] = self.dram_partition_modulo_sectors[index]
                .checked_add(other.dram_partition_modulo_sectors[index])
                .ok_or(SystemCounterError::Overflow {
                    counter: "DRAM modulo partition candidate sectors",
                })?;
            dram_partition_xor_sectors[index] = self.dram_partition_xor_sectors[index]
                .checked_add(other.dram_partition_xor_sectors[index])
                .ok_or(SystemCounterError::Overflow {
                    counter: "DRAM XOR partition candidate sectors",
                })?;
        }
        for index in 0..6 {
            dram_slice_modulo_sectors[index] = self.dram_slice_modulo_sectors[index]
                .checked_add(other.dram_slice_modulo_sectors[index])
                .ok_or(SystemCounterError::Overflow {
                    counter: "DRAM slice candidate sectors",
                })?;
        }
        macro_rules! add_frontier {
            ($field:ident, $name:literal) => {
                let $field = self
                    .$field
                    .checked_add(other.$field)
                    .ok_or(SystemCounterError::Overflow { counter: $name })?;
            };
        }
        add_frontier!(l2_frontier_samples, "L2 frontier samples");
        add_frontier!(l2_issue_to_fill_cycles_sum, "L2 issue-to-fill cycles");
        let l2_issue_to_fill_cycles_max = self
            .l2_issue_to_fill_cycles_max
            .max(other.l2_issue_to_fill_cycles_max);
        add_frontier!(
            l2_issue_to_response_cycles_sum,
            "L2 issue-to-response cycles"
        );
        let l2_issue_to_response_cycles_max = self
            .l2_issue_to_response_cycles_max
            .max(other.l2_issue_to_response_cycles_max);
        add_frontier!(l2_service_delay_cycles_sum, "L2 service-delay cycles");
        let l2_service_delay_cycles_max = self
            .l2_service_delay_cycles_max
            .max(other.l2_service_delay_cycles_max);
        let mut by_origin = self.by_origin;
        for origin in MemoryOrigin::ALL {
            let origin_ix = origin.index();
            by_origin[origin_ix].read_sectors = self.by_origin[origin_ix]
                .read_sectors
                .checked_add(other.by_origin[origin_ix].read_sectors)
                .ok_or(SystemCounterError::Overflow {
                    counter: "DRAM origin read sectors",
                })?;
            by_origin[origin_ix].bw_delayed_responses = self.by_origin[origin_ix]
                .bw_delayed_responses
                .checked_add(other.by_origin[origin_ix].bw_delayed_responses)
                .ok_or(SystemCounterError::Overflow {
                    counter: "DRAM origin bandwidth-delayed responses",
                })?;
        }
        let next = Self {
            arch: self.arch,
            service_rate_bytes_per_sec: self
                .service_rate_bytes_per_sec
                .or(other.service_rate_bytes_per_sec),
            read_sectors,
            atomic_rmw_read_sectors,
            reduction_rmw_read_sectors,
            bw_delayed_responses,
            l2_evicted_lines,
            l2_evicted_valid_sectors,
            l2_evicted_pending_lines,
            l2_evicted_pending_sectors,
            l2_dirty_eviction_lines,
            l2_dirty_evicted_sectors,
            writeback_completed_lines: self
                .writeback_completed_lines
                .checked_add(other.writeback_completed_lines)
                .ok_or(SystemCounterError::Overflow {
                    counter: "completed L2 writeback lines",
                })?,
            writeback_completed_sectors: self
                .writeback_completed_sectors
                .checked_add(other.writeback_completed_sectors)
                .ok_or(SystemCounterError::Overflow {
                    counter: "completed L2 writeback sectors",
                })?,
            writeback_completed_bytes: self
                .writeback_completed_bytes
                .checked_add(other.writeback_completed_bytes)
                .ok_or(SystemCounterError::Overflow {
                    counter: "completed L2 writeback bytes",
                })?,
            l1_writeback_completed_lines: self
                .l1_writeback_completed_lines
                .checked_add(other.l1_writeback_completed_lines)
                .ok_or(SystemCounterError::Overflow {
                    counter: "completed L1 writeback lines",
                })?,
            l1_writeback_completed_sectors: self
                .l1_writeback_completed_sectors
                .checked_add(other.l1_writeback_completed_sectors)
                .ok_or(SystemCounterError::Overflow {
                    counter: "completed L1 writeback sectors",
                })?,
            l1_writeback_completed_bytes: self
                .l1_writeback_completed_bytes
                .checked_add(other.l1_writeback_completed_bytes)
                .ok_or(SystemCounterError::Overflow {
                    counter: "completed L1 writeback bytes",
                })?,
            l1_payload_forwarded_sectors: self
                .l1_payload_forwarded_sectors
                .checked_add(other.l1_payload_forwarded_sectors)
                .ok_or(SystemCounterError::Overflow {
                    counter: "logical L1 payload-forwarded sectors",
                })?,
            l1_payload_forwarded_bytes: self
                .l1_payload_forwarded_bytes
                .checked_add(other.l1_payload_forwarded_bytes)
                .ok_or(SystemCounterError::Overflow {
                    counter: "logical L1 payload-forwarded bytes",
                })?,
            l1_payload_forwarding_digest_xor: self.l1_payload_forwarding_digest_xor
                ^ other.l1_payload_forwarding_digest_xor,
            writeback_completion_frontier: self
                .writeback_completion_frontier
                .into_iter()
                .chain(other.writeback_completion_frontier)
                .max(),
            coherence_completed_transactions: self
                .coherence_completed_transactions
                .checked_add(other.coherence_completed_transactions)
                .ok_or(SystemCounterError::Overflow {
                    counter: "completed coherence transactions",
                })?,
            coherence_completed_sectors: self
                .coherence_completed_sectors
                .checked_add(other.coherence_completed_sectors)
                .ok_or(SystemCounterError::Overflow {
                    counter: "completed coherence sectors",
                })?,
            coherence_completion_frontier: self
                .coherence_completion_frontier
                .into_iter()
                .chain(other.coherence_completion_frontier)
                .max(),
            maximum_terminal_l2_dirty_lines: self
                .maximum_terminal_l2_dirty_lines
                .max(other.maximum_terminal_l2_dirty_lines),
            maximum_terminal_l2_dirty_sectors: self
                .maximum_terminal_l2_dirty_sectors
                .max(other.maximum_terminal_l2_dirty_sectors),
            maximum_terminal_l1_dirty_lines: self
                .maximum_terminal_l1_dirty_lines
                .max(other.maximum_terminal_l1_dirty_lines),
            maximum_terminal_l1_dirty_sectors: self
                .maximum_terminal_l1_dirty_sectors
                .max(other.maximum_terminal_l1_dirty_sectors),
            dram_partition_modulo_sectors,
            dram_partition_xor_sectors,
            dram_slice_modulo_sectors,
            l2_frontier_samples,
            l2_issue_to_fill_cycles_sum,
            l2_issue_to_fill_cycles_max,
            l2_issue_to_response_cycles_sum,
            l2_issue_to_response_cycles_max,
            l2_service_delay_cycles_sum,
            l2_service_delay_cycles_max,
            by_origin,
        };
        self.validate_origin_partition()?;
        other.validate_origin_partition()?;
        next.validate_origin_partition()?;
        *self = next;
        Ok(())
    }
}

/// Internal result of one kernel launch through the shared-L2 system. Public
/// consumers receive only the configuration-bound replay result.
#[derive(Debug, Clone)]
pub(crate) struct SystemResult {
    /// Deployment architecture validated across schedule and all hierarchy configs.
    pub(crate) arch: ArchId,
    /// Per traced SM id.
    pub(crate) per_sm: BTreeMap<i32, SmSimResult>,
    /// Shared-L2 counters for the whole launch.
    pub(crate) l2: L2Counters,
    /// Per-source L2 sector/hit split (driver-side emission observer).
    pub(crate) l2_by_source: L2SourceSplit,
    /// DRAM read-traffic observer (server engagement diagnostics).
    pub(crate) dram: DramStats,
    /// Union of per-SM unmodeled write-class capability tails.
    pub(crate) unmodeled_tails: TailFlags,
    /// Raw timestamp of the last successfully processed modeled event.
    /// `None` means the queue was empty; `Some(0)` is a real event at time zero.
    /// This is a diagnostic timestamp, not an elapsed-cycle prediction.
    pub(crate) terminal_event_time: Option<u64>,
    /// Last successfully processed modeled event that is not a write
    /// acknowledgement or a writeback/coherence-only service completion. This
    /// preserves an SM-active terminal lower bound when a tail service lane
    /// drains after the final SM issue; the full event frontier above remains
    /// available separately.
    pub(crate) sm_active_terminal_event_time: Option<u64>,
    /// Complete eligible-global L1 access observations captured by the
    /// explicit visibility seam. The default path retains this stream without
    /// changing replay; the explicit visibility-protocol experiment consumes
    /// it through tokenized invalidation events.
    #[allow(dead_code)]
    pub(crate) l1_visibility_observations: Vec<sim_mem::L1AccessObservation>,
    #[allow(dead_code)]
    pub(crate) l1_fence_observations: Vec<sim_mem::L1FenceObservation>,
    #[allow(dead_code)]
    pub(crate) l1_fill_visibility_observations: Vec<sim_mem::L1FillVisibleObservation>,
}

/// Explicit event-time value admission supplied by an external producer.
/// `time` is a simulator-cycle input from that producer; it is never inferred
/// from payload sequence or converted from NCU/CUPTI/device timestamps.
/// Allocation identity comes from the producer, while resident generation and
/// sector mask are checked against the receiving SM's model-local L1 state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct L1PayloadAdmissionEvent {
    pub sm_id: i32,
    pub time: u64,
    /// Optional dynamic trace key. Low-level synthetic system tests may omit
    /// it; the public trace replay adapter requires it and validates it against
    /// the retained launch before scheduling the event.
    pub identity: Option<NativePayloadJoinKey>,
    /// Filled by the public trace replay ingress from the retained dynamic
    /// store location. Synthetic low-level tests may omit it only when
    /// `identity` is also absent.
    pub writer: Option<PayloadStoreSite>,
    pub payloads: Box<[SectorPayloadV1]>,
    pub witnesses: Box<[L1PayloadResidencyWitness]>,
}

/// Run every SM's blocks to completion over one shared L2. `l2` is
/// caller-constructed (and caller-drained afterwards) so the geometry
/// resolution stays at the composition root; it must be FRESH per kernel
/// (cold cache, NCU `--cache-control all` parity — same contract as the
/// per-SM L1s). The caller's model is unchanged on every typed error; a
/// successful run commits the fully drained staged L2 as its final operation.
/// `dram` selects the read-bandwidth model ([`DramBw`]):
/// `Unmodeled` is the latency-only ablation / equivalence-differential
/// mode; `Carded` rates are validated non-zero here (typed
/// [`SystemError::DramBandwidthInvalid`], never a clamp).

#[allow(clippy::too_many_arguments)]

/// Drain resident dirty sectors from the per-SM L1s through the explicit
/// writeback queue. The cache plans are committed only after every lower-level
/// service has been admitted and completed, so a capacity or stale-plan error
/// leaves the L1 state dirty rather than silently losing data.
fn flush_l1_dirty_residency(
    cores: &mut [SmCore<'_>],
    l2: &L2Model,
    writeback_queue: Option<&mut WritebackQueue>,
    terminal_event_time: &mut Option<u64>,
    dram_stats: &mut DramStats,
) -> Result<(), SystemError> {
    let mut plans = Vec::new();
    for (ix, core) in cores.iter().enumerate() {
        *dram_stats = dram_stats.with_terminal_l1_dirty_residency(core.l1_dirty_residency())?;
        let plan = core.preview_l1_dirty_flush()?;
        let payloads = if core.payload_observation_enabled() {
            // A store-bound value event enables a stricter terminal contract:
            // every dirty sector must still carry a complete payload before
            // the mask-only timing queue is allowed to clear it. The queue's
            // byte server remains a timing model, not a silicon DRAM receipt.
            core.preview_l1_dirty_payloads()?
        } else {
            Vec::new()
        };
        if !plan.lines().is_empty() {
            plans.push((ix, plan, payloads));
        }
    }
    let l2_dirty = l2.preview_dirty_flush();
    for (_, plan, _) in &plans {
        for l1_line in plan.lines() {
            if let Some(l2_line) = l2_dirty
                .lines()
                .iter()
                .find(|candidate| candidate.line_id == l1_line.line_id)
                && l1_line.dirty_mask & l2_line.dirty_mask != 0
            {
                return Err(SystemError::L1L2DirtyOverlap {
                    line_id: l1_line.line_id,
                    l1_mask: l1_line.dirty_mask,
                    l2_mask: l2_line.dirty_mask,
                });
            }
        }
    }
    if plans.is_empty() {
        return Ok(());
    }
    let Some(writeback_queue) = writeback_queue else {
        return Err(SystemError::L1DirtyWritebackRequiresQueue);
    };
    let capacity = usize::try_from(writeback_queue.capacity_sectors()).map_err(|_| {
        SystemError::CounterOverflow {
            counter: "terminal L1 writeback capacity",
        }
    })?;
    let mut latest_done = (*terminal_event_time).unwrap_or(0);
    let mut batch_issue = latest_done;
    let mut forwarded_payloads = Vec::new();
    for (ix, plan, payloads) in plans {
        let mut offset = 0usize;
        while offset < plan.lines().len() {
            let mut end = offset;
            let mut sectors = 0usize;
            while end < plan.lines().len() {
                let line_sectors = usize::try_from(plan.lines()[end].dirty_mask.count_ones())
                    .map_err(|_| SystemError::CounterOverflow {
                        counter: "terminal L1 dirty sectors",
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
            let requests = plan.lines()[offset..end]
                .iter()
                .map(|line| ServiceRequest::writeback(line.line_id, line.dirty_mask, batch_issue))
                .collect::<Vec<_>>();
            let service_plans = writeback_queue.preview_services(&requests)?;
            writeback_queue.commit_services(&service_plans)?;
            for service in &service_plans {
                let completion =
                    writeback_queue.complete(service.token(), service.service_done_cycle())?;
                latest_done = latest_done.max(completion.service_done_cycle);
                let sectors = u64::from(completion.dirty_mask.count_ones());
                dram_stats.writeback_completed_lines = dram_stats
                    .writeback_completed_lines
                    .checked_add(1)
                    .ok_or(SystemError::CounterOverflow {
                        counter: "completed L1 writeback lines",
                    })?;
                dram_stats.writeback_completed_sectors = dram_stats
                    .writeback_completed_sectors
                    .checked_add(sectors)
                    .ok_or(SystemError::CounterOverflow {
                        counter: "completed L1 writeback sectors",
                    })?;
                let bytes = sectors
                    .checked_mul(32)
                    .ok_or(SystemError::CounterOverflow {
                        counter: "completed L1 writeback bytes",
                    })?;
                dram_stats.writeback_completed_bytes = dram_stats
                    .writeback_completed_bytes
                    .checked_add(bytes)
                    .ok_or(SystemError::CounterOverflow {
                        counter: "completed L1 writeback bytes",
                    })?;
                dram_stats.l1_writeback_completed_lines = dram_stats
                    .l1_writeback_completed_lines
                    .checked_add(1)
                    .ok_or(SystemError::CounterOverflow {
                        counter: "completed L1 writeback lines",
                    })?;
                dram_stats.l1_writeback_completed_sectors = dram_stats
                    .l1_writeback_completed_sectors
                    .checked_add(sectors)
                    .ok_or(SystemError::CounterOverflow {
                        counter: "completed L1 writeback sectors",
                    })?;
                dram_stats.l1_writeback_completed_bytes = dram_stats
                    .l1_writeback_completed_bytes
                    .checked_add(bytes)
                    .ok_or(SystemError::CounterOverflow {
                        counter: "completed L1 writeback bytes",
                    })?;
            }
            batch_issue = latest_done;
            offset = end;
        }
        cores[ix].commit_l1_dirty_flush(plan)?;
        forwarded_payloads.extend(payloads);
    }
    for payload in forwarded_payloads {
        let mut ledger = PayloadForwardingLedgerV1::from_admitted_store(payload)
            .map_err(|error| SystemError::PayloadForwarding(error.to_string()))?;
        ledger
            .commit_l2(payload)
            .map_err(|error| SystemError::PayloadForwarding(error.to_string()))?;
        ledger
            .commit_dram(payload)
            .map_err(|error| SystemError::PayloadForwarding(error.to_string()))?;
        dram_stats.l1_payload_forwarded_sectors = dram_stats
            .l1_payload_forwarded_sectors
            .checked_add(1)
            .ok_or(SystemError::CounterOverflow {
                counter: "logical L1 payload-forwarded sectors",
            })?;
        dram_stats.l1_payload_forwarded_bytes = dram_stats
            .l1_payload_forwarded_bytes
            .checked_add(32)
            .ok_or(SystemError::CounterOverflow {
                counter: "logical L1 payload-forwarded bytes",
            })?;
        dram_stats.l1_payload_forwarding_digest_xor ^= payload.digest64();
    }
    dram_stats.writeback_completion_frontier = Some(
        dram_stats
            .writeback_completion_frontier
            .unwrap_or(0)
            .max(latest_done),
    );
    *terminal_event_time = Some((*terminal_event_time).unwrap_or(0).max(latest_done));
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_kernel_shared_with_residency_and_candidates_with_coherence(
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
) -> Result<SystemResult, SystemError> {
    simulate_kernel_shared_with_residency_and_candidates_with_coherence_and_payload_events(
        per_sm_blocks,
        schedule,
        params,
        shared_memory_residency,
        l1_config,
        l2,
        dram,
        memory_service_candidates,
        memory_admission_candidates,
        memory_admission_feedback,
        coherence_config,
        &[],
    )
}

