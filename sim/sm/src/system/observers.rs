//! Semantic submodule; report-only boundary state and serde contracts.

use super::*;

/// Per-[`L2Source`] sector/hit split — the driver-side EMISSION observer
/// (the cache model itself no longer counts provenance; gate-3 altitude
/// note c). Indexed by discriminant order.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct L2SourceSplit {
    pub(super) arch: Option<ArchId>,
    pub sectors: [u64; 4],
    pub hits: [u64; 4],
    /// Operation-class lookup observer, separate from the mixed source lanes.
    /// These fields are populated only when the request operation is known;
    /// they do not alter L2 state or timing.
    pub read_lookup_requests: u64,
    pub read_lookup_hits: u64,
    /// Subset of read lookup hits caused only by coalescing with an already
    /// pending fill.  This is retained as a diagnostic so source-qualified
    /// read observers can be compared with and without pending-merge hits;
    /// it does not alter cache state or timing.
    pub read_pending_merge_hits: u64,
    pub store_lookup_requests: u64,
    pub store_lookup_hits: u64,
    /// Observer-only resident hits among the launch-local coalesced store
    /// packet candidate. This is distinct from the physical per-record store
    /// hit lane: repeated writes to one sector may hit the timed L2 model but
    /// are one source-TEX packet at the external client boundary.
    pub store_packet_lookup_hits: u64,
    /// Resident-sector hits among packet candidates, before write-tag
    /// coalescing. This is a report-only lower-bound observer.
    pub store_packet_resident_lookup_hits: u64,
    /// Observer-only per-SM line/sector packet candidate. A second global
    /// store to the same sector within the bounded admission window is
    /// retained as one candidate packet; this never changes the L2 state,
    /// source-sector baseline, DRAM demand, or timing.
    pub store_packet_candidate_sectors: u64,
    /// Passive client-scope shadow counters. They never enter physical
    /// L2/DRAM/timing state or scored hit/miss totals.
    pub client_scope_requests: [u64; 10],
    pub client_scope_resident_hits: [u64; 10],
    pub client_scope_pending_merges: [u64; 10],
    pub client_scope_fill_misses: [u64; 10],
    pub ltc128b_shadow_promoted_sectors: u64,
    pub atomic_lookup_requests: u64,
    pub atomic_lookup_hits: u64,
    pub reduction_lookup_requests: u64,
    pub reduction_lookup_hits: u64,
    /// Source-qualified samples of the number of pending L2 sectors observed
    /// immediately before each event request. This is a PRT/fill-pressure
    /// observer, not a GA102 capacity claim or a timing input.
    pub pending_pressure_samples: [u64; 4],
    /// Sum of pending-sector counts for the corresponding source samples.
    /// Consumers can form a bounded mean pressure without treating a gauge as
    /// additive traffic or selecting a hidden hash.
    pub pending_pressure_sum: [u64; 4],
    pub(super) origin_sectors: [[u64; 4]; 2],
    pub(super) origin_hits: [[u64; 4]; 2],
}

impl L2SourceSplit {
    pub fn for_arch(arch: ArchId) -> Self {
        Self {
            arch: Some(arch),
            ..Self::default()
        }
    }

    pub fn arch(&self) -> Option<ArchId> {
        self.arch
    }

    pub fn source(&self, s: L2Source) -> (u64, u64) {
        (self.sectors[s as usize], self.hits[s as usize])
    }

    pub fn source_for_origin(&self, origin: MemoryOrigin, source: L2Source) -> (u64, u64) {
        let origin_ix = origin.index();
        let source_ix = source as usize;
        (
            self.origin_sectors[origin_ix][source_ix],
            self.origin_hits[origin_ix][source_ix],
        )
    }

    pub fn origin(&self, origin: MemoryOrigin) -> L2OriginSourceStats {
        L2OriginSourceStats {
            sectors: self.origin_sectors[origin.index()],
            hits: self.origin_hits[origin.index()],
        }
    }

    /// Build a coherent synthetic observer from explicit origin lanes.
    pub fn from_origins(
        arch: ArchId,
        global: L2OriginSourceStats,
        local: L2OriginSourceStats,
    ) -> Result<Self, SystemCounterError> {
        let mut split = Self::for_arch(arch);
        split.origin_sectors[MemoryOrigin::Global.index()] = global.sectors;
        split.origin_hits[MemoryOrigin::Global.index()] = global.hits;
        split.origin_sectors[MemoryOrigin::Local.index()] = local.sectors;
        split.origin_hits[MemoryOrigin::Local.index()] = local.hits;
        for source in L2_SOURCES {
            let source_ix = source as usize;
            split.sectors[source_ix] = global.sectors[source_ix]
                .checked_add(local.sectors[source_ix])
                .ok_or(SystemCounterError::L2OriginPartitionOverflow {
                    counter: "sectors",
                    l2_source: source,
                    global: global.sectors[source_ix],
                    local: local.sectors[source_ix],
                })?;
            split.hits[source_ix] = global.hits[source_ix]
                .checked_add(local.hits[source_ix])
                .ok_or(SystemCounterError::L2OriginPartitionOverflow {
                    counter: "hits",
                    l2_source: source,
                    global: global.hits[source_ix],
                    local: local.hits[source_ix],
                })?;
        }
        split.validate_origin_partition()?;
        Ok(split)
    }

    /// Explicitly attribute an already-built unified synthetic observer to one
    /// origin. Modeled requests always mutate an explicit lane directly.
    pub fn with_unified_origin(mut self, origin: MemoryOrigin) -> Self {
        self.origin_sectors = [[0; 4]; 2];
        self.origin_hits = [[0; 4]; 2];
        self.origin_sectors[origin.index()] = self.sectors;
        self.origin_hits[origin.index()] = self.hits;
        self
    }

    pub fn validate_origin_partition(&self) -> Result<(), SystemCounterError> {
        for source in L2_SOURCES {
            let source_ix = source as usize;
            let global_sectors = self.origin_sectors[MemoryOrigin::Global.index()][source_ix];
            let local_sectors = self.origin_sectors[MemoryOrigin::Local.index()][source_ix];
            let global_hits = self.origin_hits[MemoryOrigin::Global.index()][source_ix];
            let local_hits = self.origin_hits[MemoryOrigin::Local.index()][source_ix];
            if global_hits > global_sectors {
                return Err(SystemCounterError::L2OriginHitExceedsSectors {
                    origin: MemoryOrigin::Global,
                    l2_source: source,
                    hits: global_hits,
                    sectors: global_sectors,
                });
            }
            if local_hits > local_sectors {
                return Err(SystemCounterError::L2OriginHitExceedsSectors {
                    origin: MemoryOrigin::Local,
                    l2_source: source,
                    hits: local_hits,
                    sectors: local_sectors,
                });
            }
            let sectors = global_sectors.checked_add(local_sectors).ok_or(
                SystemCounterError::L2OriginPartitionOverflow {
                    counter: "sectors",
                    l2_source: source,
                    global: global_sectors,
                    local: local_sectors,
                },
            )?;
            if self.sectors[source_ix] != sectors {
                return Err(SystemCounterError::L2OriginPartitionMismatch {
                    counter: "sectors",
                    l2_source: source,
                    unified: self.sectors[source_ix],
                    global: global_sectors,
                    local: local_sectors,
                });
            }

            let hits = global_hits.checked_add(local_hits).ok_or(
                SystemCounterError::L2OriginPartitionOverflow {
                    counter: "hits",
                    l2_source: source,
                    global: global_hits,
                    local: local_hits,
                },
            )?;
            if self.hits[source_ix] != hits {
                return Err(SystemCounterError::L2OriginPartitionMismatch {
                    counter: "hits",
                    l2_source: source,
                    unified: self.hits[source_ix],
                    global: global_hits,
                    local: local_hits,
                });
            }
            if self.hits[source_ix] > self.sectors[source_ix] {
                return Err(SystemCounterError::L2SourceHitExceedsSectors {
                    l2_source: source,
                    hits: self.hits[source_ix],
                    sectors: self.sectors[source_ix],
                });
            }
        }
        Ok(())
    }

    pub fn add_checked(&mut self, other: &L2SourceSplit) -> Result<(), SystemCounterError> {
        let expected = self
            .arch
            .ok_or(sim_arch::ArchitectureProvenanceError::Unbound {
                component: "L2 source split",
            })?;
        let got = other
            .arch
            .ok_or(sim_arch::ArchitectureProvenanceError::Unbound {
                component: "L2 source split",
            })?;
        if expected != got {
            return Err(sim_arch::ArchitectureProvenanceError::Mismatch {
                component: "L2 source split",
                expected,
                got,
            }
            .into());
        }
        let mut sectors = [0; 4];
        let mut hits = [0; 4];
        let mut pending_pressure_samples = [0; 4];
        let mut pending_pressure_sum = [0; 4];
        let mut client_scope_requests = [0; 10];
        let mut client_scope_resident_hits = [0; 10];
        let mut client_scope_pending_merges = [0; 10];
        let mut client_scope_fill_misses = [0; 10];
        for i in 0..4 {
            sectors[i] = self.sectors[i].checked_add(other.sectors[i]).ok_or(
                SystemCounterError::Overflow {
                    counter: "L2 source sectors",
                },
            )?;
            hits[i] =
                self.hits[i]
                    .checked_add(other.hits[i])
                    .ok_or(SystemCounterError::Overflow {
                        counter: "L2 source hits",
                    })?;
            pending_pressure_samples[i] = self.pending_pressure_samples[i]
                .checked_add(other.pending_pressure_samples[i])
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 source pending-pressure samples",
                })?;
            pending_pressure_sum[i] = self.pending_pressure_sum[i]
                .checked_add(other.pending_pressure_sum[i])
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 source pending-pressure sum",
                })?;
        }
        for i in 0..10 {
            client_scope_requests[i] = self.client_scope_requests[i]
                .checked_add(other.client_scope_requests[i])
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 client-scope requests",
                })?;
            client_scope_resident_hits[i] = self.client_scope_resident_hits[i]
                .checked_add(other.client_scope_resident_hits[i])
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 client-scope resident hits",
                })?;
            client_scope_pending_merges[i] = self.client_scope_pending_merges[i]
                .checked_add(other.client_scope_pending_merges[i])
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 client-scope pending merges",
                })?;
            client_scope_fill_misses[i] = self.client_scope_fill_misses[i]
                .checked_add(other.client_scope_fill_misses[i])
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 client-scope fill misses",
                })?;
        }
        let mut origin_sectors = [[0; 4]; 2];
        let mut origin_hits = [[0; 4]; 2];
        for origin in MemoryOrigin::ALL {
            let origin_ix = origin.index();
            for source_ix in 0..4 {
                origin_sectors[origin_ix][source_ix] = self.origin_sectors[origin_ix][source_ix]
                    .checked_add(other.origin_sectors[origin_ix][source_ix])
                    .ok_or(SystemCounterError::Overflow {
                        counter: "L2 origin/source sectors",
                    })?;
                origin_hits[origin_ix][source_ix] = self.origin_hits[origin_ix][source_ix]
                    .checked_add(other.origin_hits[origin_ix][source_ix])
                    .ok_or(SystemCounterError::Overflow {
                        counter: "L2 origin/source hits",
                    })?;
            }
        }
        let next = Self {
            arch: self.arch,
            sectors,
            hits,
            read_lookup_requests: self
                .read_lookup_requests
                .checked_add(other.read_lookup_requests)
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 read lookup requests",
                })?,
            read_lookup_hits: self
                .read_lookup_hits
                .checked_add(other.read_lookup_hits)
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 read lookup hits",
                })?,
            read_pending_merge_hits: self
                .read_pending_merge_hits
                .checked_add(other.read_pending_merge_hits)
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 read pending-merge lookup hits",
                })?,
            store_lookup_requests: self
                .store_lookup_requests
                .checked_add(other.store_lookup_requests)
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 store lookup requests",
                })?,
            store_lookup_hits: self
                .store_lookup_hits
                .checked_add(other.store_lookup_hits)
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 store lookup hits",
                })?,
            store_packet_lookup_hits: self
                .store_packet_lookup_hits
                .checked_add(other.store_packet_lookup_hits)
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 store packet lookup hits",
                })?,
            store_packet_resident_lookup_hits: self
                .store_packet_resident_lookup_hits
                .checked_add(other.store_packet_resident_lookup_hits)
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 resident store packet lookup hits",
                })?,
            store_packet_candidate_sectors: self
                .store_packet_candidate_sectors
                .checked_add(other.store_packet_candidate_sectors)
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 store packet candidate sectors",
                })?,
            atomic_lookup_requests: self
                .atomic_lookup_requests
                .checked_add(other.atomic_lookup_requests)
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 atomic lookup requests",
                })?,
            atomic_lookup_hits: self
                .atomic_lookup_hits
                .checked_add(other.atomic_lookup_hits)
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 atomic lookup hits",
                })?,
            reduction_lookup_requests: self
                .reduction_lookup_requests
                .checked_add(other.reduction_lookup_requests)
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 reduction lookup requests",
                })?,
            reduction_lookup_hits: self
                .reduction_lookup_hits
                .checked_add(other.reduction_lookup_hits)
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 reduction lookup hits",
                })?,
            pending_pressure_samples,
            pending_pressure_sum,
            client_scope_requests,
            client_scope_resident_hits,
            client_scope_pending_merges,
            client_scope_fill_misses,
            ltc128b_shadow_promoted_sectors: self
                .ltc128b_shadow_promoted_sectors
                .checked_add(other.ltc128b_shadow_promoted_sectors)
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 LTC128B shadow promoted sectors",
                })?,
            origin_sectors,
            origin_hits,
        };
        self.validate_origin_partition()?;
        other.validate_origin_partition()?;
        next.validate_origin_partition()?;
        *self = next;
        Ok(())
    }
}

#[cfg(feature = "reference")]
pub(super) struct ReferenceTrace {
    pub(super) writer: BufWriter<File>,
    pub(super) records: u64,
    pub(super) max_records: u64,
    pub(super) previous_event: Option<serde_json::Value>,
    pub(super) issue_effect_cursors: Vec<usize>,
    pub(super) pending_typed_entry: Option<TypedEventBoundaryRecord>,
    pub(super) witness: Option<ReferenceMacroWitness>,
    pub(super) pending_native_result: Option<NativeBoundaryExecutionResult>,
    pub(super) active_native_result: Option<NativeBoundaryExecutionResult>,
    pub(super) pending_region_end: Option<TypedEventBoundaryRecord>,
    pub(super) native_macro_dispatches: u64,
    pub(super) native_macro_fallbacks: u64,
}

#[cfg(feature = "reference")]
impl ReferenceTrace {
    pub(super) fn from_env() -> Result<Option<Self>, ReferenceTraceError> {
        let Some(path) = std::env::var_os("SIM_SM_REFERENCE_TRACE") else {
            return Ok(None);
        };
        Self::from_paths(
            PathBuf::from(path),
            std::env::var_os("SIM_SM_REFERENCE_TYPED_SCALAR_WITNESS").map(PathBuf::from),
        )
    }

    pub(super) fn from_paths(
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

    pub(super) fn take_pending_native_result(&mut self) -> Option<NativeBoundaryExecutionResult> {
        self.pending_native_result.take()
    }

    pub(super) fn note_native_fallback(&mut self) {
        self.native_macro_fallbacks = self.native_macro_fallbacks.saturating_add(1);
    }

    pub(super) fn note_native_dispatch(&mut self) {
        self.native_macro_dispatches = self.native_macro_dispatches.saturating_add(1);
    }

    pub(super) fn expect_region_end(&mut self, exit: TypedEventBoundaryRecord) {
        self.pending_region_end = Some(exit);
    }

    pub(super) fn has_active_native_result(&self) -> bool {
        self.active_native_result.is_some()
    }

    pub(super) fn discard_active_native_result(&mut self) {
        self.active_native_result = None;
    }

    pub(super) fn native_result_decision(
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

    pub(super) fn write<T: serde::Serialize>(
        &mut self,
        value: &T,
    ) -> Result<(), ReferenceTraceError> {
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

    pub(super) fn write_event(
        &mut self,
        value: serde_json::Value,
    ) -> Result<(), ReferenceTraceError> {
        // Move the previous state out so the writer can be borrowed without
        // cloning either potentially large boundary.
        if let Some(previous) = self.previous_event.take() {
            let mut changes = Vec::new();
            reference_operations(&previous, &value, &mut Vec::new(), &mut changes);
            #[derive(serde::Serialize)]
            pub(super) struct Delta<'a> {
                pub(super) kind: &'static str,
                pub(super) base_record: u64,
                pub(super) operations: Vec<ReferenceOperation<'a>>,
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
    pub(super) fn record_event(
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
        pub(super) type ResponseQueueKey = (u32, u64, u64, u64, TypedResponseKind, u64);
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

    pub(super) fn finish(
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

pub(super) enum L2AccessEventRequests {
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
pub(super) struct EventPerformanceCounters {
    pub(super) queue_events_processed: u64,
    pub(super) processed: u64,
    pub(super) sm_wake: u64,
    pub(super) l2_access: u64,
    pub(super) l2_access_batches: u64,
    pub(super) l2_access_batched_requests: u64,
    pub(super) l2_fill: u64,
    pub(super) atomic_completion: u64,
    pub(super) writeback_completion: u64,
    pub(super) coherence_completion: u64,
    pub(super) l1_visibility_event: u64,
    pub(super) l1_payload_admission: u64,
    pub(super) memory_write_ack: u64,
    pub(super) mem_response: u64,
    pub(super) same_cycle_after_first: u64,
    pub(super) queue_same_cycle_after_first: u64,
    pub(super) maximum_queue_depth: u64,
    pub(super) previous_time: Option<u64>,
}

#[cfg(feature = "performance-counters")]
impl EventPerformanceCounters {
    pub(super) fn observe(
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

#[derive(Default)]
pub(super) struct AdmissionInstructionObserver {
    pub(super) pending: Option<PendingAdmissionInstruction>,
}

impl AdmissionInstructionObserver {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn observe(
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
