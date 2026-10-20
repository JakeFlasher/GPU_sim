//! Semantic submodule; report-only boundary state and serde contracts.

use super::*;

impl<'a> SmCore<'a> {
    #[inline]
    pub(super) fn memory_pipeline_slot(&self, pc: u64) -> Option<MemoryPipelineSlot> {
        self.memory_pipeline_schedule
            .binary_search_by_key(&pc, |(key, _)| *key)
            .ok()
            .map(|index| self.memory_pipeline_schedule[index].1)
    }
    #[inline]
    pub(super) fn is_constant_path_pc(&self, pc: u64) -> bool {
        self.constant_path_pcs.binary_search(&pc).is_ok()
    }
    #[inline]
    pub(super) fn immediate_constant_lines(&self, pc: u32) -> Option<&[u64]> {
        self.immediate_constant_schedule
            .binary_search_by_key(&pc, |(key, _)| *key)
            .ok()
            .map(|index| self.immediate_constant_schedule[index].1.as_ref())
    }
    pub(crate) fn install_memory_service_candidates(
        &mut self,
        candidates: &[MemoryServiceCandidate],
    ) -> Result<(), IssueError> {
        self.memory_service_candidates = MemoryServiceCandidateState::checked(candidates)?;
        Ok(())
    }
    #[cfg(feature = "reference")]
    pub(crate) fn typed_l2_request_key(&self, sm_id: u32, reply: ReplyTo) -> Option<L2RequestKey> {
        let response_id = reply.response_id()?;
        let response_kind = match reply {
            ReplyTo::L1Fill {
                fill: FillKey::Line { .. },
                ..
            } => TypedResponseKind::L1Fill,
            ReplyTo::L1Fill {
                fill: FillKey::Sector { .. },
                ..
            } => TypedResponseKind::L1SectorFill,
            ReplyTo::Warp { .. } => TypedResponseKind::Warp,
            ReplyTo::AsyncCopy { .. } => TypedResponseKind::AsyncCopy,
            ReplyTo::None => return None,
        };
        let warp_slot = self.warps.get(response_id.warp)?.warp_slot;
        let (response_warp, instruction, sector_index) = response_id.boundary_identity();
        Some(L2RequestKey {
            sm_id,
            response_warp,
            warp_slot,
            instruction,
            sector_index,
            response_kind,
        })
    }
    #[cfg(feature = "reference")]
    pub(crate) fn typed_l2_request_owners(
        &self,
        sm_id: u32,
        response_due: &[(u64, u64, u64, u64, Option<u64>)],
    ) -> Vec<L2RequestOwner> {
        let MemModel::L1Shared(state) = &self.mem else {
            return Vec::new();
        };
        let mut owners = state
            .outstanding_response_ids
            .iter()
            .map(|(response_id, reply)| {
                let direct_identity = self
                    .warps
                    .get(response_id.warp)
                    .and_then(|warp| warp.sectors.get(response_id.sector_index))
                    .map(|sector_id| {
                        (
                            CanonicalLineKey::VirtualAbsolute {
                                line_id: sector_id.div_euclid(4),
                            },
                            1_u8 << sector_id.rem_euclid(4) as u8,
                        )
                    });
                let (line, sector_mask, response_kind) = match reply {
                    ReplyTo::L1Fill { fill, .. } => match fill {
                        FillKey::Line { line, .. } => (
                            CanonicalLineKey::VirtualAbsolute { line_id: *line },
                            state
                                .response_masks
                                .get(response_id)
                                .copied()
                                .or_else(|| state.fill_masks.get(line).copied())
                                .unwrap_or(0),
                            TypedResponseKind::L1Fill,
                        ),
                        FillKey::Sector { line, sector, .. } => (
                            CanonicalLineKey::VirtualAbsolute { line_id: *line },
                            state
                                .response_masks
                                .get(response_id)
                                .copied()
                                .unwrap_or(1_u8 << *sector),
                            TypedResponseKind::L1SectorFill,
                        ),
                    },
                    ReplyTo::Warp { .. } => {
                        let (line, sector_mask) = direct_identity
                            .unwrap_or((CanonicalLineKey::Unknown { line_id: None }, 0));
                        (line, sector_mask, TypedResponseKind::Warp)
                    }
                    ReplyTo::AsyncCopy { .. } => {
                        let (line, sector_mask) = direct_identity
                            .unwrap_or((CanonicalLineKey::Unknown { line_id: None }, 0));
                        (line, sector_mask, TypedResponseKind::AsyncCopy)
                    }
                    ReplyTo::None => (
                        CanonicalLineKey::Unknown { line_id: None },
                        0,
                        TypedResponseKind::None,
                    ),
                };
                let warp_slot = self
                    .warps
                    .get(response_id.warp)
                    .map(|warp| warp.warp_slot)
                    .unwrap_or(-1);
                let (warp, instruction, sector_index) = response_id.boundary_identity();
                let due = response_due.iter().find(
                    |(candidate_warp, candidate_instruction, candidate_sector, _, _)| {
                        (*candidate_warp, *candidate_instruction, *candidate_sector)
                            == (warp, instruction, sector_index)
                    },
                );
                L2RequestOwner {
                    request: L2RequestKey {
                        sm_id,
                        response_warp: warp,
                        warp_slot,
                        instruction,
                        sector_index,
                        response_kind,
                    },
                    line,
                    sector_mask,
                    allocation_generation: due.and_then(|(_, _, _, _, generation)| *generation),
                    state: L2RequestState::Outstanding,
                    due: due.map(|(_, _, _, due, _)| *due),
                }
            })
            .collect::<Vec<_>>();
        owners.sort_by(|left, right| {
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
        owners
    }
    #[cfg(feature = "reference")]
    pub(crate) fn typed_l1_state(&self) -> Option<L1BoundaryState> {
        self.mem.l1().map(L1Cache::typed_boundary_state)
    }
    /// Enable complete eligible-global L1 access observations for an explicit
    /// native visibility experiment. The requester is the independently
    /// retained physical `sm_id`, not an actor-vector index.
    pub(crate) fn enable_l1_visibility_observations(
        &mut self,
        requester: u8,
    ) -> Result<(), IssueError> {
        if requester >= 128 {
            return Err(IssueError::CounterOverflow {
                counter: "L1 visibility requester domain",
            });
        }
        let MemModel::L1Shared(st) = &mut self.mem else {
            return Err(IssueError::ResponseModeMismatch {
                mode: "L1 visibility requires shared-L2",
            });
        };
        st.l1_visibility_requester = Some(requester);
        Ok(())
    }
    pub(crate) fn enable_l1_resident_writeback(&mut self) -> Result<(), IssueError> {
        let Some(state) = self.mem.shared_l1_mut() else {
            return Err(IssueError::ResponseModeMismatch {
                mode: "L1 resident writeback requires shared-L2",
            });
        };
        state
            .l1
            .set_store_policy(L1StorePolicy::ResidentHitWriteBack);
        Ok(())
    }
    pub(crate) fn l1_dirty_residency(&self) -> sim_mem::L1DirtyResidency {
        self.mem
            .l1()
            .map_or_else(sim_mem::L1DirtyResidency::default, L1Cache::dirty_residency)
    }
    pub(crate) fn preview_l1_dirty_flush(&self) -> Result<sim_mem::L1DirtyFlushPlan, IssueError> {
        let MemModel::L1Shared(state) = &self.mem else {
            return Err(IssueError::ResponseModeMismatch {
                mode: "L1 dirty flush requires shared-L2",
            });
        };
        Ok(state.l1.preview_dirty_flush())
    }
    pub(crate) fn preview_l1_dirty_payloads(
        &self,
    ) -> Result<Vec<sim_mem::SectorPayloadV1>, IssueError> {
        let MemModel::L1Shared(state) = &self.mem else {
            return Err(IssueError::ResponseModeMismatch {
                mode: "L1 payload preview requires shared-L2",
            });
        };
        state.l1.preview_dirty_payloads().map_err(IssueError::from)
    }
    pub(crate) fn commit_l1_dirty_flush(
        &mut self,
        plan: sim_mem::L1DirtyFlushPlan,
    ) -> Result<(), IssueError> {
        let Some(state) = self.mem.shared_l1_mut() else {
            return Err(IssueError::ResponseModeMismatch {
                mode: "L1 dirty flush requires shared-L2",
            });
        };
        state.l1.commit_dirty_flush(plan).map_err(IssueError::from)
    }
    pub(crate) fn preview_l1_dirty_payload_admission(
        &self,
        payloads: &[sim_mem::SectorPayloadV1],
        witnesses: &[sim_mem::L1PayloadResidencyWitness],
    ) -> Result<sim_mem::L1DirtyPayloadAdmissionPlan, IssueError> {
        let MemModel::L1Shared(state) = &self.mem else {
            return Err(IssueError::ResponseModeMismatch {
                mode: "L1 payload admission requires shared-L2",
            });
        };
        state
            .l1
            .preview_dirty_payload_admission_with_witness(payloads, witnesses)
            .map_err(IssueError::from)
    }
    pub(crate) fn commit_l1_dirty_payload_admission(
        &mut self,
        plan: sim_mem::L1DirtyPayloadAdmissionPlan,
    ) -> Result<(), IssueError> {
        let MemModel::L1Shared(state) = &mut self.mem else {
            return Err(IssueError::ResponseModeMismatch {
                mode: "L1 payload admission requires shared-L2",
            });
        };
        state
            .l1
            .commit_dirty_payload_admission(plan)
            .map_err(IssueError::from)
    }
    /// Apply one event-time, witness-aware dirty-payload admission as a
    /// single native-facing operation. The underlying preview/commit pair is
    /// still exposed internally so a future event producer can retain the
    /// plan across an explicit scheduling boundary; this convenience path is
    /// atomic with respect to the current SM actor.
    #[allow(dead_code)] // no producer currently supplies this event to replay
    pub(crate) fn admit_l1_dirty_payloads_with_witness(
        &mut self,
        payloads: &[sim_mem::SectorPayloadV1],
        witnesses: &[sim_mem::L1PayloadResidencyWitness],
    ) -> Result<(), IssueError> {
        let plan = self.preview_l1_dirty_payload_admission(payloads, witnesses)?;
        self.commit_l1_dirty_payload_admission(plan)
    }
    pub(crate) fn watch_l1_payload_sectors(&mut self, payloads: &[sim_mem::SectorPayloadV1]) {
        if let MemModel::L1Shared(st) = &mut self.mem {
            let writers = st
                .payload_store_writers
                .get_or_insert_with(Default::default);
            for payload in payloads {
                writers
                    .entry((payload.line_id(), payload.sector()))
                    .or_insert(None);
            }
        }
    }
    /// The public replay path binds the site from the retained trace, never
    /// from a producer's guessed issue cycle or reused physical warp slot.
    pub(crate) fn admit_issued_l1_store_payloads(
        &mut self,
        site: PayloadStoreSite,
        at: u64,
        payloads: &[sim_mem::SectorPayloadV1],
        witnesses: &[sim_mem::L1PayloadResidencyWitness],
    ) -> Result<(), IssueError> {
        let MemModel::L1Shared(st) = &mut self.mem else {
            return Err(IssueError::ResponseModeMismatch {
                mode: "store payload admission requires shared-L2",
            });
        };
        for payload in payloads {
            let line = payload.line_id();
            let sector = payload.sector();
            let refuse = |reason| IssueError::PayloadStoreBinding {
                line,
                sector,
                reason,
            };
            let writer = st
                .payload_store_writers
                .as_ref()
                .and_then(|writers| writers.get(&(line, sector)))
                .and_then(Option::as_ref)
                .ok_or_else(|| refuse("referenced store has not issued"))?;
            if writer.site != site {
                return Err(refuse("current writer is a different dynamic store"));
            }
            if writer.accepted_at > at {
                return Err(refuse("payload event precedes modeled store acceptance"));
            }
            if !writer.retained_dirty {
                return Err(refuse("store did not retain this sector dirty in L1"));
            }
            if writer.attached {
                return Err(refuse(
                    "payload already attached to this dynamic store sector",
                ));
            }
            if st
                .l1
                .resident_payload_state(line)
                .map(|(generation, _)| generation)
                != writer.resident_generation
            {
                return Err(refuse("resident allocation changed after the store"));
            }
        }
        st.l1.attach_observed_store_payloads(payloads, witnesses)?;
        for payload in payloads {
            if let Some(Some(writer)) = st
                .payload_store_writers
                .as_mut()
                .and_then(|writers| writers.get_mut(&(payload.line_id(), payload.sector())))
            {
                writer.attached = true;
            }
        }
        Ok(())
    }
    pub(crate) fn take_l1_visibility_events(&mut self) -> Vec<sim_mem::L1VisibilityEvent> {
        let MemModel::L1Shared(st) = &mut self.mem else {
            return Vec::new();
        };
        let mut events = st
            .l1_visibility_observations
            .drain(..)
            .map(sim_mem::L1VisibilityEvent::Access)
            .chain(
                st.l1_fence_observations
                    .drain(..)
                    .map(sim_mem::L1VisibilityEvent::Fence),
            )
            .chain(
                st.l1_fill_visibility_observations
                    .drain(..)
                    .map(sim_mem::L1VisibilityEvent::FillVisible),
            )
            .chain(
                st.l1_invalidate_all_observations
                    .drain(..)
                    .map(sim_mem::L1VisibilityEvent::InvalidateAll),
            )
            .collect::<Vec<_>>();
        events.sort_unstable_by_key(|event| match event {
            sim_mem::L1VisibilityEvent::Access(observation) => observation.sequence,
            sim_mem::L1VisibilityEvent::Fence(observation) => observation.sequence,
            sim_mem::L1VisibilityEvent::FillVisible(observation) => observation.sequence,
            sim_mem::L1VisibilityEvent::InvalidateAll(observation) => observation.sequence,
        });
        events
    }
    /// Count visibility observations that have been produced by the SM but
    /// not yet transferred to the shared event queue.  The system layer drains
    /// this vector at every actor boundary; `advance` uses the count to stop
    /// immediately after an issue cycle that created a visibility event,
    /// preventing strict L1 replay from batching later issue cycles ahead of
    /// its own same-cycle coherence observation.
    pub(crate) fn pending_l1_visibility_event_count(&self) -> usize {
        let MemModel::L1Shared(st) = &self.mem else {
            return 0;
        };
        st.l1_visibility_observations
            .len()
            .saturating_add(st.l1_fence_observations.len())
            .saturating_add(st.l1_fill_visibility_observations.len())
            .saturating_add(st.l1_invalidate_all_observations.len())
    }
    pub(crate) fn invalidate_l1(
        &mut self,
        line_id: i64,
        sector_mask: u8,
    ) -> Result<(), IssueError> {
        let MemModel::L1Shared(st) = &mut self.mem else {
            return Err(IssueError::ResponseModeMismatch {
                mode: "L1 invalidation requires shared-L2",
            });
        };
        st.l1.invalidate(line_id, sector_mask)?;
        Ok(())
    }
    pub(crate) fn invalidate_l1_all(&mut self) -> Result<u64, IssueError> {
        let MemModel::L1Shared(st) = &mut self.mem else {
            return Err(IssueError::ResponseModeMismatch {
                mode: "L1 invalidation requires shared-L2",
            });
        };
        Ok(st.l1.invalidate_all()?)
    }
    #[cfg(feature = "reference")]
    pub(crate) fn typed_memory_pipeline_state(&self) -> Option<MemoryPipelineBoundaryState> {
        self.memory_issue_pipeline.as_ref().map(|state| {
            let diagnostics = state.diagnostics;
            MemoryPipelineBoundaryState {
                queue_entries_with_dispatch: state.config.queue_entries_with_dispatch,
                local_initiation_interval_cycles: state.config.local_initiation_interval_cycles,
                shared_accept_interval_cycles: state.config.shared_accept_interval_cycles,
                queue_release_times: state
                    .queue_release_times
                    .iter()
                    .map(|queue| queue.iter().copied().collect())
                    .collect(),
                local_completion_ready: state.local_completion_ready.to_vec(),
                shared_accept_ready: state.shared_accept_ready,
                instructions_admitted: diagnostics.instructions_admitted,
                operand_pipeline_delay_cycles: diagnostics.operand_pipeline_delay_cycles,
                shared_queue_delay_cycles: diagnostics.shared_queue_delay_cycles,
                maximum_operand_pipeline_delay_cycles: diagnostics
                    .maximum_operand_pipeline_delay_cycles,
                maximum_shared_queue_delay_cycles: diagnostics.maximum_shared_queue_delay_cycles,
                maximum_queue_occupancy: diagnostics.maximum_queue_occupancy,
                terminal_shared_accept_cycle: diagnostics.terminal_shared_accept_cycle,
            }
        })
    }
    pub(crate) fn enable_memory_admission_observations(&mut self) {
        self.emit_admission_witnesses = true;
    }
    pub(crate) fn install_memory_admission_feedback(
        &mut self,
        candidate: MemoryAdmissionCandidate,
    ) -> Result<(), IssueError> {
        candidate.validate_feedback(self.params.read_barrier_latency)?;
        self.memory_admission_feedback =
            Some(Box::new(MemoryAdmissionFeedbackState::checked(candidate)?));
        self.emit_admission_witnesses = true;
        Ok(())
    }
    pub(crate) fn complete_memory_admission_feedback(
        &mut self,
        witness: AdmissionRequestWitness,
        operation: MemoryOperation,
        hit_sectors: u64,
        miss_sectors: u64,
        l2_requests: u64,
        unique_regions: u64,
        issue_time: u64,
        now: u64,
    ) -> Result<Option<MemoryWriteAck>, IssueError> {
        self.memory_admission_feedback
            .as_mut()
            .ok_or(IssueError::MemoryAdmissionFeedbackReservationMissing {
                warp: witness.warp(),
                instruction: witness.instruction(),
            })?
            .complete(
                witness,
                operation,
                hit_sectors,
                miss_sectors,
                l2_requests,
                unique_regions,
                issue_time,
                now,
            )
    }
    pub(crate) fn complete_memory_write_ack(
        &mut self,
        ack: MemoryWriteAck,
        now: u64,
    ) -> Result<(), IssueError> {
        self.memory_admission_feedback
            .as_mut()
            .ok_or(IssueError::MemoryWriteAckNotPending {
                warp: ack.witness.warp(),
                instruction: ack.witness.instruction(),
            })?
            .acknowledge_write(ack, now)
    }
    pub(crate) fn pending_memory_write_acks(&self) -> usize {
        self.memory_admission_feedback
            .as_deref()
            .map_or(0, MemoryAdmissionFeedbackState::pending_write_ack_count)
    }
    /// Responses the system still owes this SM (diagnostics).
    pub(crate) fn outstanding_responses(&self) -> usize {
        match &self.mem {
            MemModel::L1Shared(st) => st.outstanding_replies.len(),
            _ => 0,
        }
    }
    /// Memory interaction of one issued record in L1Shared mode. Returns
    /// the KNOWN data-ready component (`None` when completion depends only on
    /// in-flight responses); unknown components remain in ordinary-barrier
    /// pending units and/or async-copy-group source units.
    pub(super) fn mem_step_shared(
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
    pub(super) fn preflight_response_accounting(
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
    pub(super) fn project_memory_service_candidates(
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
    pub(super) fn issue_ready_at_without_immediate_constant<const CHECK_ASYNC: bool>(
        &self,
        subcore: usize,
        warp: usize,
    ) -> Option<u64> {
        self.issue_ready_at_impl::<CHECK_ASYNC>(subcore, warp, false)
    }
    /// True when an SM86 warp is otherwise ready but its current CBANK line
    /// is not yet available in the private immediate-constant cache.
    pub(super) fn immediate_constant_blocked(&self, subcore: usize, warp: usize, now: u64) -> bool {
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
    /// Pick in stable SMSP order while reserving the transaction's shared
    /// admission capacity. A candidate rejected by an earlier SMSP's claim is
    /// not a dispatch bubble: the same SMSP may still issue its held/youngest
    /// ready non-candidate (or another candidate whose queue has room).
    pub(super) fn pick_warps_with_memory_admission<const CHECK_ASYNC: bool>(
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
    pub(super) fn preflight_issue_memory(
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
    pub(super) fn preflight_async_l1_hit_service(
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
    pub(super) fn register_l1tex_writeback(
        &mut self,
        subcore: usize,
        warp: usize,
        kind: ResultWritebackKind,
        barrier: Option<DepBarrier>,
        visible_at: Option<u64>,
    ) -> Result<(), IssueError> {
        let l1_hit_latency = self.params.l1_hit_latency;
        let (
            Some(state),
            ResultWritebackKind::L1TexStrongSmU32 { bank },
            Some(barrier),
            Some(visible_at),
        ) = (self.result_writebacks.as_mut(), kind, barrier, visible_at)
        else {
            return Ok(());
        };
        let hit_floor = checked_cycle_add("L1 hit-pipeline floor", self.t, l1_hit_latency)?;
        if visible_at != hit_floor
            || l1_hit_latency < state.timing.l1tex_writeback_to_visibility_cycles
        {
            return Ok(());
        }
        let writeback_at = visible_at - state.timing.l1tex_writeback_to_visibility_cycles;
        state.pending.push(PendingVariableWriteback {
            subcore: subcore as u8,
            bank,
            writeback_at,
            visible_at,
            warp,
            barrier,
        });
        Ok(())
    }
    /// Probe the per-SMSP Immediate Constant Cache for the selected
    /// instructions.  A tag miss consumes an issue attempt but does not
    /// dispatch or advance the warp; the warp is parked until the line's
    /// deterministic GCC/L1C return time.  The state is cloned first so a
    /// later cycle-preflight error cannot leave a partially reserved constant
    /// request behind.
    pub(super) fn preflight_immediate_constant(
        &self,
        picks: [Option<usize>; SUB_CORES],
        now: u64,
    ) -> Result<ImmediateConstantPlan, IssueError> {
        let Some(mut cache) = self.immediate_constant_cache.as_ref().cloned() else {
            return Ok(ImmediateConstantPlan {
                picks,
                cache: None,
                waits: [None; SUB_CORES],
                extra_waits: Vec::new(),
                miss_observed: false,
            });
        };
        let mut effective = picks;
        let mut waits = [None; SUB_CORES];
        let mut miss_observed = false;
        for (subcore, pick) in picks.into_iter().enumerate() {
            let Some(warp) = pick else { continue };
            let state = &self.warps[warp];
            let pc = state.insts[state.next].pc;
            let Some(lines) = self.immediate_constant_lines(pc) else {
                continue;
            };
            if lines.is_empty() {
                continue;
            }
            if state.imc_wait_until > now {
                effective[subcore] = None;
                waits[subcore] = Some((warp, state.imc_wait_until));
                miss_observed = true;
                continue;
            }
            if let Some(ready) = cache[subcore].reserve(lines, now, self.params.l1_miss_latency)?
                && ready > now
            {
                effective[subcore] = None;
                waits[subcore] = Some((warp, ready));
                miss_observed = true;
            }
        }
        Ok(ImmediateConstantPlan {
            picks: effective,
            cache: Some(cache),
            waits,
            extra_waits: Vec::new(),
            miss_observed,
        })
    }
}
