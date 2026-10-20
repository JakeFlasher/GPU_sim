//! Semantic submodule; report-only boundary state and serde contracts.

use super::*;

/// Static register demand projected from admitted SASS operands. Regular-bank
/// counts are retained only for SM86; SM120 records expose projection/gap
/// coverage while physical banks/ports remain unknown. This is a pressure
/// envelope, not a dynamic physical pipe, SMSP, or issue-slot observation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct OperandBankPressure {
    pub projected_records: u64,
    pub generic_projected_records: u64,
    pub gap_records: u64,
    pub not_applicable_records: u64,
    pub source_bank_reads: [u64; 2],
    pub destination_bank_writes: [u64; 2],
}

impl OperandBankPressure {
    pub(crate) fn observe(
        &mut self,
        status: &FixedLatencyOperandStatus,
        arch: sim_arch::ArchId,
    ) -> Result<(), crate::IssueError> {
        let projection = match status {
            FixedLatencyOperandStatus::Projected(projection) => projection,
            FixedLatencyOperandStatus::Gap(_) => {
                add(&mut self.gap_records, 1)?;
                return Ok(());
            }
            FixedLatencyOperandStatus::NotApplicable => {
                add(&mut self.not_applicable_records, 1)?;
                return Ok(());
            }
        };
        add(&mut self.projected_records, 1)?;
        if projection.opcode() == crate::sass_operands::FixedLatencyOpcode::Generic {
            add(&mut self.generic_projected_records, 1)?;
        }
        if arch != sim_arch::ArchId::Sm86 {
            return Ok(());
        }
        let reads = projection.sm86_regular_source_bank_reads();
        let writes = projection.sm86_regular_destination_bank_writes();
        for bank in 0..2 {
            add(&mut self.source_bank_reads[bank], u64::from(reads[bank]))?;
            add(
                &mut self.destination_bank_writes[bank],
                u64::from(writes[bank]),
            )?;
        }
        Ok(())
    }
}

/// Dynamic populations, including predicated-off scheduler records.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct InstructionActivity {
    pub warp_instructions: u64,
    pub active_thread_instructions: u64,
    pub predicate_on_thread_instructions: u64,
    pub predicated_off_warp_instructions: u64,
    /// Dynamic records whose producer omitted the active lane mask. These
    /// are not treated as all-lanes-inactive instructions.
    pub active_mask_missing_instructions: u64,
    /// Dynamic records with a present but non-full active-lane mask. Fast
    /// scalar certification refuses these because the descriptor does not
    /// retain per-record lane populations.
    pub active_mask_nonfull_instructions: u64,
    /// Dynamic records whose producer omitted the predicate mask. Effective
    /// predicate-on populations remain unknown for these records.
    pub predicate_mask_missing_instructions: u64,
    /// Dynamic records whose effective predicate population is not the active
    /// warp population. This remains an observation, not a scalar timing
    /// adjustment; conservative fast certification refuses it.
    pub predicate_mask_nonfull_instructions: u64,
}

/// Trace-local physical-pipeline candidate domains. A dynamic SASS record
/// identifies the opcode and participants, but not the hardware pipe chosen
/// when an instruction has multiple legal paths. `candidate_instructions` is
/// therefore an upper-bound membership count, while unique/ambiguous/unknown
/// populations expose attribution confidence without inventing a pipe sample.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PhysicalPipelineAttribution {
    pub total_warp_instructions: u64,
    pub unique_instructions: u64,
    pub ambiguous_instructions: u64,
    pub unknown_instructions: u64,
    pub candidate_instructions: BTreeMap<String, u64>,
    /// Deterministic model-selected pipe for each record with a nonempty
    /// candidate domain. This is an executable pressure-balancing policy, not
    /// a reconstruction of the hardware's dynamic selector.
    pub selected_instructions: BTreeMap<String, u64>,
    pub selected_unknown_instructions: u64,
    pub selection_policy: &'static str,
    /// A second, timing-aware model-local assignment. It uses candidate
    /// resource availability, initiation interval and nominal latency, but it
    /// is not a hardware route observation.
    #[serde(skip_serializing_if = "is_empty_string_map")]
    pub scheduled_instructions: BTreeMap<String, u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schedule_frontier_cycles: Option<u64>,
    pub schedule_policy: &'static str,
    /// Per-candidate unconstrained bounds from the same model-local
    /// availability state used by the greedy assignment. These are feasible
    /// model ranges, not observations of a silicon route or proof that all
    /// candidates can be selected simultaneously.
    #[serde(skip_serializing_if = "is_empty_string_u64_map")]
    pub candidate_earliest_issue_cycles: BTreeMap<String, u64>,
    #[serde(skip_serializing_if = "is_empty_string_u64_map")]
    pub candidate_latest_issue_cycles: BTreeMap<String, u64>,
    #[serde(skip_serializing_if = "is_empty_string_u64_map")]
    pub candidate_earliest_completion_cycles: BTreeMap<String, u64>,
    #[serde(skip_serializing_if = "is_empty_string_u64_map")]
    pub candidate_latest_completion_cycles: BTreeMap<String, u64>,
    pub model_interval_records: u64,
    /// Singleton-domain class membership, not a hardware route observation.
    /// This remains out of the legacy output and is used only by the opt-in
    /// route-count envelope.
    #[serde(skip)]
    pub(super) mandatory_class_instructions: BTreeMap<String, u64>,
    #[serde(skip)]
    pub(super) scheduler_free: BTreeMap<String, u64>,
    #[serde(skip)]
    pub(super) scheduler_sequence: u64,
}

impl Default for PhysicalPipelineAttribution {
    fn default() -> Self {
        Self {
            total_warp_instructions: 0,
            unique_instructions: 0,
            ambiguous_instructions: 0,
            unknown_instructions: 0,
            candidate_instructions: BTreeMap::new(),
            selected_instructions: BTreeMap::new(),
            selected_unknown_instructions: 0,
            selection_policy: "min_selected_then_lexical_v1",
            scheduled_instructions: BTreeMap::new(),
            schedule_frontier_cycles: None,
            schedule_policy: "candidate_earliest_completion_v1_model_only",
            candidate_earliest_issue_cycles: BTreeMap::new(),
            candidate_latest_issue_cycles: BTreeMap::new(),
            candidate_earliest_completion_cycles: BTreeMap::new(),
            candidate_latest_completion_cycles: BTreeMap::new(),
            model_interval_records: 0,
            mandatory_class_instructions: BTreeMap::new(),
            scheduler_free: BTreeMap::new(),
            scheduler_sequence: 0,
        }
    }
}

impl PhysicalPipelineAttribution {
    /// Aggregate the union of *class* candidates observed in this trace.
    ///
    /// This is intentionally a domain, not a route selection: a trace can
    /// contain both unique and ambiguous instructions, and the union must not
    /// be mistaken for a per-instruction hardware route assignment.
    pub fn aggregate_candidate_class_domain(&self) -> Result<CandidateDomain, LatentModelError> {
        if self.unknown_instructions != 0 {
            return CandidateDomain::unknown(
                "some trace instructions have no candidate class domain",
            );
        }
        if self.candidate_instructions.is_empty() {
            return CandidateDomain::unknown("no dynamic route candidates observed");
        }
        CandidateDomain::from_values(self.candidate_instructions.keys().cloned())
    }

    /// Count ranges over independent class choices. No pressure-balancing
    /// assignment or scored NCU value participates. Unclassified records may
    /// add demand to any named class and can also use classes not listed here.
    pub fn class_count_envelope(&self) -> Result<PipelineClassEnvelope, crate::IssueError> {
        let mut counts = BTreeMap::new();
        for (class, &candidate_count) in &self.candidate_instructions {
            let lower = self
                .mandatory_class_instructions
                .get(class)
                .copied()
                .unwrap_or(0);
            let upper = candidate_count
                .checked_add(self.unknown_instructions)
                .ok_or(crate::IssueError::CounterOverflow {
                    counter: "pipeline class upper count",
                })?;
            if lower > candidate_count || upper > self.total_warp_instructions {
                return Err(crate::IssueError::CounterOverflow {
                    counter: "pipeline class population invariant",
                });
            }
            counts.insert(class.clone(), ClassCountInterval { lower, upper });
        }
        Ok(PipelineClassEnvelope {
            role: "model_class_count_bounds_not_dynamic_physical_pipe_labels",
            evidence: sim_semantics::EvidenceKind::Hypothesis,
            total_warp_instructions: self.total_warp_instructions,
            unclassified_instructions: self.unknown_instructions,
            unlisted_classes_possible: self.unknown_instructions != 0,
            counts,
        })
    }

    /// Convert candidate pipeline classes into an abstract service-demand
    /// envelope.  The result is deliberately separate from `selected` and
    /// `scheduled` diagnostics: it describes feasible resource demand, not a
    /// physical route assignment.
    pub fn resource_demand_envelope(&self) -> Result<ResourceDemandEnvelope, LatentModelError> {
        let mut pools = BTreeMap::new();
        for (pool, &candidate_count) in &self.candidate_instructions {
            let lower = self
                .mandatory_class_instructions
                .get(pool)
                .copied()
                .unwrap_or(0);
            let upper = candidate_count
                .checked_add(self.unknown_instructions)
                .ok_or_else(|| LatentModelError::ResourceDemandOverflow { pool: pool.clone() })?;
            pools.insert(pool.clone(), ResourceDemandInterval::interval(lower, upper));
        }
        ResourceDemandEnvelope::new(
            self.total_warp_instructions,
            self.unknown_instructions,
            pools,
            EvidenceKind::Hypothesis,
        )
    }
}

/// Observe the first subsequent instruction executed by EACH active lane,
/// even when another path executes first. Unresolved lanes of different
/// pending events are disjoint, bounding this state to at most 32 events.
/// This observer does not change the scheduler, trace, or cache state.
#[derive(Default)]
pub(crate) struct BranchSuccessors {
    pub(super) pending: Vec<PendingBranch>,
}

impl BranchSuccessors {
    pub(crate) fn observe(
        &mut self,
        totals: &mut BranchTargets,
        function: i32,
        pc: u32,
        active: u32,
        opcode: &str,
    ) -> Result<(), crate::IssueError> {
        let mut index = 0;
        while index < self.pending.len() {
            let event = &mut self.pending[index];
            let successor_lanes = event.remaining & active;
            if successor_lanes != 0 {
                event.remaining &= !successor_lanes;
                if let Some((_, _, mask)) =
                    event
                        .targets
                        .iter_mut()
                        .find(|(target_function, target_pc, _)| {
                            *target_function == function && *target_pc == pc
                        })
                {
                    *mask |= successor_lanes;
                } else {
                    event.targets.push((function, pc, successor_lanes));
                }
            }
            if event.remaining == 0 {
                let event = self.pending.swap_remove(index);
                add(&mut totals.distinct_successors, event.targets.len() as u64)?;
                let successor_threads: u64 = event
                    .targets
                    .iter()
                    .map(|(_, _, mask)| u64::from(mask.count_ones()))
                    .sum();
                add(&mut totals.successor_active_threads, successor_threads)?;
                if event.targets.len() > 1 {
                    add(&mut totals.divergent_events, 1)?;
                    add(
                        &mut totals.divergent_threads,
                        u64::from(event.active_threads),
                    )?;
                    let minority = event
                        .targets
                        .iter()
                        .map(|(_, _, mask)| mask.count_ones())
                        .min()
                        .unwrap_or(0);
                    add(&mut totals.divergent_minority_threads, u64::from(minority))?;
                } else {
                    add(&mut totals.uniform_events, 1)?;
                    add(&mut totals.uniform_active_threads, successor_threads)?;
                }
            } else {
                index += 1;
            }
        }
        if active != 0
            && matches!(
                opcode_head(opcode),
                "BRA" | "BRX" | "BRXU" | "JMP" | "JMX" | "JMXU" | "CALL" | "RET"
            )
        {
            if self.pending.len() >= 32 {
                return Err(crate::IssueError::CounterOverflow {
                    counter: "branch successor pending events",
                });
            }
            add(&mut totals.events, 1)?;
            self.pending.push(PendingBranch {
                remaining: active,
                active_threads: active.count_ones(),
                targets: Vec::new(),
            });
        }
        Ok(())
    }

    pub(crate) fn finish(self, totals: &mut BranchTargets) -> Result<(), crate::IssueError> {
        add(&mut totals.unresolved_events, self.pending.len() as u64)?;
        let unresolved_threads: u64 = self
            .pending
            .iter()
            .map(|event| u64::from(event.remaining.count_ones()))
            .sum();
        add(&mut totals.unresolved_threads, unresolved_threads)
    }
}

impl InstructionActivity {
    pub(super) fn observe(
        &mut self,
        active: u8,
        effective: u8,
        active_mask_present: bool,
        predicate_mask_present: bool,
    ) -> Result<(), crate::IssueError> {
        add(&mut self.warp_instructions, 1)?;
        if active_mask_present {
            add(&mut self.active_thread_instructions, u64::from(active))?;
            if active != 32 {
                add(&mut self.active_mask_nonfull_instructions, 1)?;
            }
        } else {
            add(&mut self.active_mask_missing_instructions, 1)?;
        }
        if predicate_mask_present && active_mask_present {
            add(
                &mut self.predicate_on_thread_instructions,
                u64::from(effective),
            )?;
            add(
                &mut self.predicated_off_warp_instructions,
                u64::from(effective == 0),
            )?;
            if effective != active {
                add(&mut self.predicate_mask_nonfull_instructions, 1)?;
            }
        } else if !predicate_mask_present {
            add(&mut self.predicate_mask_missing_instructions, 1)?;
        }
        Ok(())
    }
}

impl TraceMetrics {
    /// Observe a byte-mask-qualified full-sector store followed by a load in
    /// the same warp stream.  The candidate is deliberately shadow-only: the
    /// native trace does not expose enough cross-warp ordering to justify a
    /// live L1 installation.  A caller must prove full-sector coverage from
    /// active/predicate masks and opcode width before setting `full_store`.
    pub(crate) fn observe_store_to_load_forwarding(
        &mut self,
        kind: AccessKind,
        sectors: &[i64],
        full_store: bool,
    ) {
        let Some(span) = reuse_span(kind, sectors) else {
            return;
        };
        let mut unique = BTreeSet::new();
        for &sector in span {
            unique.insert(sector);
        }
        if full_store
            && matches!(
                kind,
                AccessKind::GlobalStore { .. } | AccessKind::LocalStore { .. }
            )
        {
            self.reuse_warp_stream
                .full_sector_store_sectors
                .extend(unique);
        } else if matches!(
            kind,
            AccessKind::GlobalEligible { .. } | AccessKind::LocalLoad { .. }
        ) {
            for sector in unique {
                if self
                    .reuse_warp_stream
                    .full_sector_store_sectors
                    .contains(&sector)
                {
                    self.store_to_load_forwardable_sectors =
                        self.store_to_load_forwardable_sectors.saturating_add(1);
                }
            }
        }
    }
    pub(crate) fn begin_warp_memory_stream(&mut self) {
        self.begin_reuse_warp_stream();
        self.global_store_previous_sectors.clear();
        self.global_store_previous_effective_lanes = 0;
    }
    pub(crate) fn finalize_shared_load_observer(&mut self) {
        let admit = self.shared_load_full_warp_width16_records >= 120
            && self.shared_load_full_warp_width4_records.saturating_mul(9)
                == self
                    .shared_load_full_warp_width16_records
                    .saturating_mul(16);
        if admit || self.shared_load_width16_observer_reduction == 0 {
            return;
        }
        if let Some(row) = self.memory.get_mut("shared_load") {
            row.bank_wavefronts = row
                .bank_wavefronts
                .saturating_add(self.shared_load_width16_observer_reduction);
            row.bank_conflict_wavefronts = row
                .bank_conflict_wavefronts
                .saturating_add(self.shared_load_width16_observer_reduction);
            row.bank_conflict_excess_wavefronts = row
                .bank_conflict_excess_wavefronts
                .saturating_add(self.shared_load_width16_excess_reduction);
        }
    }
    pub(crate) fn observe_tma_sidecar_payload(
        &mut self,
        observation: Option<&sim_trace::async_sidecars::TmaObservation>,
    ) -> Result<(), crate::IssueError> {
        add(
            &mut self.tma_sidecar_records_observed,
            u64::from(observation.is_some()),
        )?;
        let Some(observation) = observation else {
            add(&mut self.tma_sidecar_records_missing, 1)?;
            return add(&mut self.tma_sidecar_payload_bytes_missing, 1);
        };
        add(
            &mut self.tma_sidecar_payload_bytes_observed,
            observation.bytes.unwrap_or(0),
        )?;
        add(
            &mut self.tma_sidecar_payload_bytes_missing,
            u64::from(observation.bytes.is_none()),
        )?;
        add(
            &mut self.tma_sidecar_load_records,
            u64::from(observation.is_load),
        )?;
        add(
            &mut self.tma_sidecar_store_records,
            u64::from(observation.is_store),
        )?;
        if let Some(bytes) = observation.bytes {
            if observation.is_load {
                add(&mut self.tma_sidecar_load_payload_bytes, bytes)?;
            }
            if observation.is_store {
                add(&mut self.tma_sidecar_store_payload_bytes, bytes)?;
            }
        }
        add(
            &mut self.tma_sidecar_zero_byte_records,
            u64::from(observation.zero_byte),
        )?;
        add(
            &mut self.tma_sidecar_oob_or_empty_records,
            u64::from(observation.oob_or_empty),
        )?;
        match observation.diagnostic.transfer_product_matches {
            Some(true) => add(&mut self.tma_sidecar_transfer_product_matched_records, 1)?,
            Some(false) => add(&mut self.tma_sidecar_transfer_product_mismatched_records, 1)?,
            None => {}
        }
        add(
            &mut self.tma_sidecar_raw_valid_address_refs,
            observation.diagnostic.raw_valid_address_refs,
        )?;
        add(
            &mut self.tma_sidecar_raw_oob_address_refs,
            observation.diagnostic.raw_oob_address_refs,
        )?;
        add(
            &mut self.tma_sidecar_unique_sector_count,
            observation.diagnostic.unique_sector_count,
        )?;
        add(
            &mut self.tma_sidecar_unique_line_count,
            observation.diagnostic.unique_line_count,
        )?;
        add(
            &mut self.tma_sidecar_intra_record_duplicate_sector_refs,
            observation.diagnostic.intra_record_duplicate_sector_refs,
        )?;
        if let (Some(nominal), Some(bytes)) = (
            observation.diagnostic.tensor_map_box_bytes,
            observation.bytes,
        ) {
            add(
                &mut self.tma_sidecar_nominal_box_overage_bytes_diagnostic,
                nominal.saturating_sub(bytes),
            )?;
        }
        Ok(())
    }
    pub(crate) fn observe_tma_shadow_l2(
        &mut self,
        requests: u64,
        hits: u64,
        misses: u64,
        unavailable: u64,
        load: [u64; 4],
        store: [u64; 4],
    ) -> Result<(), crate::IssueError> {
        add(&mut self.tma_shadow_l2_requests, requests)?;
        add(&mut self.tma_shadow_l2_hits, hits)?;
        add(&mut self.tma_shadow_l2_misses, misses)?;
        add(&mut self.tma_shadow_l2_address_unavailable, unavailable)?;
        add(&mut self.tma_shadow_l2_load_requests, load[0])?;
        add(&mut self.tma_shadow_l2_load_hits, load[1])?;
        add(&mut self.tma_shadow_l2_load_misses, load[2])?;
        add(&mut self.tma_shadow_l2_load_address_unavailable, load[3])?;
        add(&mut self.tma_shadow_l2_store_requests, store[0])?;
        add(&mut self.tma_shadow_l2_store_hits, store[1])?;
        add(&mut self.tma_shadow_l2_store_misses, store[2])?;
        add(&mut self.tma_shadow_l2_store_address_unavailable, store[3])?;
        Ok(())
    }
    pub(crate) fn observe_async_clock_sandwich(
        &mut self,
        pairs: u64,
        invalid_pairs: u64,
        lower_span_sum: u64,
        upper_span_sum: u64,
    ) -> Result<(), crate::IssueError> {
        add(&mut self.async_clock_sandwich_pairs_observed, pairs)?;
        add(&mut self.async_clock_sandwich_invalid_pairs, invalid_pairs)?;
        add(
            &mut self.async_clock_sandwich_lower_span_sum,
            lower_span_sum,
        )?;
        add(
            &mut self.async_clock_sandwich_upper_span_sum,
            upper_span_sum,
        )?;
        Ok(())
    }
    pub(crate) fn observe_tma_shadow_l2_ordered(
        &mut self,
        requests: u64,
        hits: u64,
        misses: u64,
        unavailable: u64,
        load: [u64; 4],
        store: [u64; 4],
    ) -> Result<(), crate::IssueError> {
        add(&mut self.tma_shadow_l2_ordered_requests, requests)?;
        add(&mut self.tma_shadow_l2_ordered_hits, hits)?;
        add(&mut self.tma_shadow_l2_ordered_misses, misses)?;
        add(
            &mut self.tma_shadow_l2_ordered_address_unavailable,
            unavailable,
        )?;
        add(&mut self.tma_shadow_l2_ordered_load_requests, load[0])?;
        add(&mut self.tma_shadow_l2_ordered_load_hits, load[1])?;
        add(&mut self.tma_shadow_l2_ordered_load_misses, load[2])?;
        add(
            &mut self.tma_shadow_l2_ordered_load_address_unavailable,
            load[3],
        )?;
        add(&mut self.tma_shadow_l2_ordered_store_requests, store[0])?;
        add(&mut self.tma_shadow_l2_ordered_store_hits, store[1])?;
        add(&mut self.tma_shadow_l2_ordered_store_misses, store[2])?;
        add(
            &mut self.tma_shadow_l2_ordered_store_address_unavailable,
            store[3],
        )?;
        Ok(())
    }
    pub(crate) fn observe_tma_joint_l2(
        &mut self,
        requests: u64,
        hits: u64,
        misses: u64,
        tma_load: [u64; 3],
        ordinary_read: [u64; 3],
        order_unavailable: u64,
        events_dropped: u64,
    ) -> Result<(), crate::IssueError> {
        add(&mut self.tma_joint_l2_requests, requests)?;
        add(&mut self.tma_joint_l2_hits, hits)?;
        add(&mut self.tma_joint_l2_misses, misses)?;
        add(&mut self.tma_joint_l2_tma_load_requests, tma_load[0])?;
        add(&mut self.tma_joint_l2_tma_load_hits, tma_load[1])?;
        add(&mut self.tma_joint_l2_tma_load_misses, tma_load[2])?;
        add(
            &mut self.tma_joint_l2_ordinary_read_requests,
            ordinary_read[0],
        )?;
        add(&mut self.tma_joint_l2_ordinary_read_hits, ordinary_read[1])?;
        add(
            &mut self.tma_joint_l2_ordinary_read_misses,
            ordinary_read[2],
        )?;
        add(&mut self.tma_joint_l2_order_unavailable, order_unavailable)?;
        add(&mut self.tma_joint_l2_events_dropped, events_dropped)?;
        Ok(())
    }
    pub(crate) fn observe_l1_merge_audit(
        &mut self,
        same_request: [u64; 2],
        older_pending: [u64; 2],
        already_requested: [u64; 2],
        unrequested_pending: [u64; 2],
        ordinary_same_request: [u64; 2],
        ordinary_older_pending: [u64; 2],
        async_same_request: [u64; 2],
        async_older_pending: [u64; 2],
        ordinary_global_older_pending: u64,
        constant_global_older_pending: u64,
        ordinary_global_same_request: u64,
        constant_global_same_request: u64,
        async_global_same_request: u64,
        ordinary_global_hits: u64,
        async_global_hits: u64,
        ordinary_global_older_sector_pending_merges: u64,
        constant_global_older_sector_pending_merges: u64,
        async_global_older_sector_pending_merges: u64,
        unrequested_subline_miss_sectors: u64,
        wavefront_mask_lookup_hits: u64,
        wavefront_mask_lookup_misses: u64,
        constant_global_hits: u64,
        constant_global_requests: u64,
        global_resident_hits: u64,
        global_lookup_sectors: u64,
        request_mask_audit: L1RequestMaskAudit,
    ) -> Result<(), crate::IssueError> {
        add(&mut self.l1_same_request_merges_global, same_request[0])?;
        add(&mut self.l1_same_request_merges_local, same_request[1])?;
        add(
            &mut self.l1_already_requested_merges_global,
            already_requested[0],
        )?;
        add(
            &mut self.l1_already_requested_merges_local,
            already_requested[1],
        )?;
        add(
            &mut self.l1_unrequested_pending_merges_global,
            unrequested_pending[0],
        )?;
        add(
            &mut self.l1_unrequested_pending_merges_local,
            unrequested_pending[1],
        )?;
        add(
            &mut self.l1_ordinary_same_request_merges_global,
            ordinary_same_request[0],
        )?;
        add(
            &mut self.l1_ordinary_same_request_merges_local,
            ordinary_same_request[1],
        )?;
        add(&mut self.l1_older_pending_merges_global, older_pending[0])?;
        add(&mut self.l1_older_pending_merges_local, older_pending[1])?;
        add(
            &mut self.l1_ordinary_older_pending_merges_global,
            ordinary_older_pending[0],
        )?;
        add(
            &mut self.l1_ordinary_older_pending_merges_local,
            ordinary_older_pending[1],
        )?;
        add(
            &mut self.l1_async_same_request_merges_global,
            async_same_request[0],
        )?;
        add(
            &mut self.l1_async_same_request_merges_local,
            async_same_request[1],
        )?;
        add(
            &mut self.l1_async_older_pending_merges_global,
            async_older_pending[0],
        )?;
        add(
            &mut self.l1_async_older_pending_merges_local,
            async_older_pending[1],
        )?;
        add(
            &mut self.l1_ordinary_global_older_pending_merges,
            ordinary_global_older_pending,
        )?;
        add(
            &mut self.l1_constant_global_older_pending_merges,
            constant_global_older_pending,
        )?;
        add(
            &mut self.l1_ordinary_global_same_request_merges,
            ordinary_global_same_request,
        )?;
        add(
            &mut self.l1_constant_global_same_request_merges,
            constant_global_same_request,
        )?;
        add(
            &mut self.l1_async_global_same_request_merges,
            async_global_same_request,
        )?;
        add(&mut self.l1_ordinary_global_hits, ordinary_global_hits)?;
        add(&mut self.l1_async_global_hits, async_global_hits)?;
        add(
            &mut self.l1_ordinary_global_older_sector_pending_merges,
            ordinary_global_older_sector_pending_merges,
        )?;
        add(
            &mut self.l1_constant_global_older_sector_pending_merges,
            constant_global_older_sector_pending_merges,
        )?;
        add(
            &mut self.l1_async_global_older_sector_pending_merges,
            async_global_older_sector_pending_merges,
        )?;
        add(
            &mut self.l1_unrequested_subline_miss_sectors,
            unrequested_subline_miss_sectors,
        )?;
        add(
            &mut self.l1_wavefront_mask_lookup_hits,
            wavefront_mask_lookup_hits,
        )?;
        add(
            &mut self.l1_wavefront_mask_lookup_misses,
            wavefront_mask_lookup_misses,
        )?;
        add(&mut self.l1_constant_global_hits, constant_global_hits)?;
        add(
            &mut self.l1_constant_global_requests,
            constant_global_requests,
        )?;
        add(&mut self.l1_global_resident_hits, global_resident_hits)?;
        add(&mut self.l1_global_lookup_sectors, global_lookup_sectors)?;
        for client in 0..request_mask_audit.request_count.len() {
            add(
                &mut self.l1_request_mask_audit.request_count[client],
                request_mask_audit.request_count[client],
            )?;
            add(
                &mut self.l1_request_mask_audit.request_line_count[client],
                request_mask_audit.request_line_count[client],
            )?;
            add(
                &mut self.l1_request_mask_audit.request_mask_overflow[client],
                request_mask_audit.request_mask_overflow[client],
            )?;
            for mask in 0..16 {
                add(
                    &mut self.l1_request_mask_audit.request_masks[client][mask],
                    request_mask_audit.request_masks[client][mask],
                )?;
                add(
                    &mut self.l1_request_mask_audit.same_request_merge_masks[client][mask],
                    request_mask_audit.same_request_merge_masks[client][mask],
                )?;
                add(
                    &mut self.l1_request_mask_audit.older_pending_merge_masks[client][mask],
                    request_mask_audit.older_pending_merge_masks[client][mask],
                )?;
            }
        }
        Ok(())
    }
    /// Record one dynamic constant-bank instruction as a source-domain
    /// observer. Each instruction contributes distinct 32-byte offset packets
    /// and the launch-wide sets retain address-domain cardinalities. These are
    /// not mathematical DRAM bounds and never change cache/service state.
    pub(crate) fn observe_constant_memory(
        &mut self,
        addresses: &[u64],
        bank: Option<u32>,
    ) -> Result<(), crate::IssueError> {
        if addresses.is_empty() {
            return Ok(());
        }
        add(&mut self.constant_memory_reference_count, 1)?;
        let address_count =
            u64::try_from(addresses.len()).map_err(|_| crate::IssueError::CounterOverflow {
                counter: "constant memory address count",
            })?;
        add(&mut self.constant_memory_address_count, address_count)?;
        let mut sectors = BTreeSet::new();
        for &address in addresses {
            // Field 15 is a bank-local effective offset, not a device
            // allocation address. Keep an absent bank distinct instead of
            // silently merging it with bank zero or another bank.
            sectors.insert((bank, address / 32));
        }
        let request_count =
            u64::try_from(sectors.len()).map_err(|_| crate::IssueError::CounterOverflow {
                counter: "constant memory sector request count",
            })?;
        add(&mut self.constant_memory_sector_requests, request_count)?;
        self.constant_memory_sector_set.extend(sectors);
        self.constant_memory_unique_sectors = u64::try_from(self.constant_memory_sector_set.len())
            .map_err(|_| crate::IssueError::CounterOverflow {
                counter: "constant memory unique sector count",
            })?;
        self.constant_memory_line_set
            .extend(addresses.iter().map(|&address| (bank, address / 64)));
        self.constant_memory_unique_lines = u64::try_from(self.constant_memory_line_set.len())
            .map_err(|_| crate::IssueError::CounterOverflow {
                counter: "constant memory unique line count",
            })?;
        Ok(())
    }
    pub(crate) fn observe_constant_metadata_only(&mut self) -> Result<(), crate::IssueError> {
        add(&mut self.constant_memory_metadata_only_records, 1)
    }
    /// Observe one statically classified `LDG.*.CONSTANT` global load. The
    /// address span is a source-qualified candidate only: its line domains
    /// are not equated with GCC requests, L2 sectors, or DRAM service.
    pub(crate) fn observe_constant_global_load(
        &mut self,
        block_id: [u32; 3],
        warp_id: i32,
        sectors: &[i64],
        off: usize,
        len: usize,
        opcode_text: &str,
        warp_uniform_address: bool,
    ) -> Result<(), crate::IssueError> {
        add(&mut self.constant_global_load_instructions, 1)?;
        add(
            &mut self.constant_global_load_sectors,
            u64::try_from(len).map_err(|_| crate::IssueError::CounterOverflow {
                counter: "constant global load sectors",
            })?,
        )?;
        let form = constant_global_load_form_index(opcode_text);
        add(&mut self.constant_global_load_form_instructions[form], 1)?;
        add(
            &mut self.constant_global_load_form_sectors[form],
            u64::try_from(len).map_err(|_| crate::IssueError::CounterOverflow {
                counter: "constant global load form sectors",
            })?,
        )?;
        if warp_uniform_address {
            add(&mut self.constant_global_load_uniform_instructions, 1)?;
            add(
                &mut self.constant_global_load_uniform_sectors,
                u64::try_from(len).map_err(|_| crate::IssueError::CounterOverflow {
                    counter: "constant global uniform load sectors",
                })?,
            )?;
        } else {
            add(&mut self.constant_global_load_divergent_instructions, 1)?;
            add(
                &mut self.constant_global_load_divergent_sectors,
                u64::try_from(len).map_err(|_| crate::IssueError::CounterOverflow {
                    counter: "constant global divergent load sectors",
                })?,
            )?;
        }
        let end = off
            .checked_add(len)
            .ok_or(crate::IssueError::CounterOverflow {
                counter: "constant global load sector span",
            })?;
        let span = sectors
            .get(off..end)
            .ok_or(crate::IssueError::CounterOverflow {
                counter: "constant global load sector span",
            })?;
        let lines: BTreeSet<u64> = span
            .iter()
            .map(|&sector| u64::try_from(sector).unwrap_or_default() / 4)
            .collect();
        add(
            &mut self.constant_global_load_request_candidates,
            u64::try_from(lines.len()).map_err(|_| crate::IssueError::CounterOverflow {
                counter: "constant global load request candidates",
            })?,
        )?;
        self.constant_global_line_set
            .extend(lines.into_iter().map(|line| (block_id, warp_id, line)));
        self.constant_global_load_unique_lines = u64::try_from(self.constant_global_line_set.len())
            .map_err(|_| crate::IssueError::CounterOverflow {
                counter: "constant global load unique lines",
            })?;
        self.constant_global_cta_line_set.extend(
            span.iter()
                .map(|&sector| (block_id, u64::try_from(sector).unwrap_or_default() / 4)),
        );
        self.constant_global_load_cta_unique_lines =
            u64::try_from(self.constant_global_cta_line_set.len()).map_err(|_| {
                crate::IssueError::CounterOverflow {
                    counter: "constant global load CTA unique lines",
                }
            })?;
        Ok(())
    }
    pub(crate) fn observe_atomic_conflict(
        &mut self,
        sectors: &[i64],
        off: usize,
        len: usize,
    ) -> Result<(), crate::IssueError> {
        let end = off
            .checked_add(len)
            .ok_or(crate::IssueError::CounterOverflow {
                counter: "atomic conflict sector span",
            })?;
        let span = sectors
            .get(off..end)
            .ok_or(crate::IssueError::CounterOverflow {
                counter: "atomic conflict sector span",
            })?;
        let distinct = span.iter().copied().collect::<BTreeSet<_>>().len();
        let aliases = len.saturating_sub(distinct);
        if aliases == 0 {
            add(&mut self.atomic_distinct_instructions, 1)?;
        } else {
            add(&mut self.atomic_conflict_instructions, 1)?;
            add(
                &mut self.atomic_conflict_alias_sectors,
                u64::try_from(aliases).map_err(|_| crate::IssueError::CounterOverflow {
                    counter: "atomic conflict alias sectors",
                })?,
            )?;
        }
        Ok(())
    }
    pub(crate) fn observe_tma_shadow_dynamic_identity(
        &mut self,
        matched: u64,
        unmatched: u64,
    ) -> Result<(), crate::IssueError> {
        add(&mut self.tma_shadow_dynamic_records_matched, matched)?;
        add(&mut self.tma_shadow_dynamic_records_unmatched, unmatched)?;
        Ok(())
    }
    pub(crate) fn observe_sync_sidecar(
        &mut self,
        observation: Option<&sim_trace::async_sidecars::SyncObservation>,
    ) -> Result<(), crate::IssueError> {
        let Some(observation) = observation else {
            return add(&mut self.sync_sidecar_records_missing, 1);
        };
        add(&mut self.sync_sidecar_records_observed, observation.records)?;
        add(
            &mut self.sync_sidecar_before_records,
            observation.before_records,
        )?;
        add(
            &mut self.sync_sidecar_after_records,
            observation.after_records,
        )?;
        add(
            &mut self.sync_sidecar_value_incomplete_records,
            observation.value_incomplete_records,
        )?;
        add(
            &mut self.sync_sidecar_mbarrier_address_records,
            observation.mbarrier_address_records,
        )?;
        add(
            &mut self.sync_sidecar_mbarrier_register_records,
            observation.mbarrier_register_records,
        )?;
        add(
            &mut self.sync_sidecar_arrival_records,
            observation.arrival_records,
        )?;
        add(
            &mut self.sync_sidecar_phase_check_records,
            observation.phase_check_records,
        )?;
        add(
            &mut self.sync_sidecar_wait_records,
            observation.wait_records,
        )?;
        add(
            &mut self.sync_sidecar_exchange_records,
            observation.exchange_records,
        )?;
        add(
            &mut self.sync_sidecar_barrier_addresses,
            u64::try_from(observation.barrier_addresses.len()).map_err(|_| {
                crate::IssueError::CounterOverflow {
                    counter: "sync_sidecar_barrier_addresses",
                }
            })?,
        )?;
        Ok(())
    }
    /// Lift the existing trace metrics into the factorized latent contract.
    /// Dynamic route identity remains unknown. The aggregate candidate-class
    /// vocabulary is not a single per-event route and cannot populate it.
    /// `class_count_envelope` separately represents the modeled class bounds.
    pub fn latent_factors(&self) -> Result<LatentFactors, LatentModelError> {
        let route =
            CandidateDomain::unknown("dynamic physical route/subunit/issue-slot unobserved")?;
        let compiler = CandidateDomain::unknown("no independent compiler witness")?;
        let cache = CandidateDomain::unknown("cache owner/read-from unobserved")?;
        let timing = CandidateDomain::unknown("endpoint identity not unified")?;
        Ok(LatentFactors {
            route,
            compiler,
            cache,
            timing,
        })
    }
    pub(super) const MAX_COHERENCE_LINES: usize = 1_000_000;
    pub(crate) fn observe_coherence(
        &mut self,
        sm_id: Option<i32>,
        effective: u8,
        kind: AccessKind,
        sector_arena: &[i64],
        instrumentation_order: Option<u64>,
    ) -> Result<(), crate::IssueError> {
        if effective == 0 {
            return Ok(());
        }
        let Some(sm_id) = sm_id else {
            add(&mut self.coherence.unknown_sm_or_line_records, 1)?;
            return Ok(());
        };
        // Placement validation currently bounds SM ids to the architecture's
        // card count (84 on SM86).  Keep a wider u128 domain for configurable
        // cards while failing closed if a future card exceeds it.
        let Some(sm_bit) = (sm_id >= 0)
            .then(|| u32::try_from(sm_id).ok())
            .flatten()
            .and_then(|id| (id < 128).then_some(1_u128 << id))
        else {
            add(&mut self.coherence.unknown_sm_or_line_records, 1)?;
            return Ok(());
        };
        let (off, len, is_write) = match kind {
            AccessKind::GlobalEligible { off, len, .. }
            | AccessKind::GlobalAsyncCopy { off, len, .. }
            | AccessKind::GlobalUntagged { off, len }
            | AccessKind::GlobalBypass {
                off,
                len,
                token:
                    sim_mem::CacheOpToken::Other
                    | sim_mem::CacheOpToken::LdgECg
                    | sim_mem::CacheOpToken::LdgECs
                    | sim_mem::CacheOpToken::LdgEStrongSm
                    | sim_mem::CacheOpToken::LdgEStrongGpu
                    | sim_mem::CacheOpToken::Ldgsts
                    | sim_mem::CacheOpToken::LdgECa
                    | sim_mem::CacheOpToken::LdgEEf
                    | sim_mem::CacheOpToken::LdgENa,
                ..
            } => (off, len, false),
            AccessKind::GlobalStore { off, len }
            | AccessKind::GlobalReduction { off, len, .. }
            | AccessKind::GlobalBypass {
                off,
                len,
                token: sim_mem::CacheOpToken::Atom,
                ..
            } => (off, len, true),
            // Local memory is private by construction and shared-memory
            // addresses do not enter the L2 global coherence domain.
            AccessKind::LocalLoad { .. }
            | AccessKind::LocalStore { .. }
            | AccessKind::Shared { .. }
            | AccessKind::NonMemory
            | AccessKind::PredicatedOffMemory => return Ok(()),
            // Keep any newly added global token conservative until its
            // address-space semantics are explicitly classified.
            AccessKind::GlobalBypass { .. } => {
                add(&mut self.coherence.unknown_sm_or_line_records, 1)?;
                return Ok(());
            }
        };
        if instrumentation_order.is_some() {
            add(&mut self.coherence.instrumentation_order_records, 1)?;
        } else {
            add(&mut self.coherence.instrumentation_order_missing_records, 1)?;
        }
        let start = off;
        let end = off
            .checked_add(usize::from(len))
            .ok_or(crate::IssueError::CounterOverflow {
                counter: "coherence sector span",
            })?;
        let sectors = sector_arena
            .get(start..end)
            .ok_or(crate::IssueError::CounterOverflow {
                counter: "coherence sector span bounds",
            })?;
        for &sector in sectors {
            let line = sector.div_euclid(4);
            let lane = sector.rem_euclid(4) as usize;
            let entry = if let Some(entry) = self.coherence_lines.get_mut(&line) {
                entry
            } else {
                if self.coherence_lines.len() >= Self::MAX_COHERENCE_LINES {
                    add(&mut self.coherence.unknown_sm_or_line_records, 1)?;
                    continue;
                }
                add(&mut self.coherence.decoded_global_lines, 1)?;
                self.coherence_lines.entry(line).or_default()
            };
            entry.sm_mask |= sm_bit;
            entry.sector_mask |= 1_u8 << lane;
            entry.sector_sm_masks[lane] |= sm_bit;
            if is_write {
                entry.write_sm_mask |= sm_bit;
                entry.sector_write_sm_masks[lane] |= sm_bit;
            } else {
                entry.read_sm_mask |= sm_bit;
                entry.sector_read_sm_masks[lane] |= sm_bit;
            }
        }
        Ok(())
    }
    /// Finalize the bounded line map into stable counters before the trace
    /// metrics are cloned into a replay result.  The map is then released so
    /// reports do not retain one entry per physical line.
    pub(crate) fn finish_coherence(&mut self) -> Result<(), crate::IssueError> {
        for state in self.coherence_lines.values() {
            if state.sm_mask.count_ones() <= 1 {
                continue;
            }
            add(&mut self.coherence.cross_sm_lines, 1)?;
            add(
                &mut self.coherence.cross_sm_sectors,
                u64::from(state.sector_mask.count_ones()),
            )?;
            for sector in 0..state.sector_sm_masks.len() {
                let sm_mask = state.sector_sm_masks[sector];
                if sm_mask.count_ones() <= 1 {
                    continue;
                }
                add(&mut self.coherence.cross_sm_sector_overlap, 1)?;
                if state.sector_read_sm_masks[sector] != 0
                    && state.sector_write_sm_masks[sector] != 0
                {
                    add(&mut self.coherence.cross_sm_read_write_sectors, 1)?;
                }
                if state.sector_write_sm_masks[sector].count_ones() > 1 {
                    add(&mut self.coherence.cross_sm_write_write_sectors, 1)?;
                }
            }
            if state.read_sm_mask != 0 && state.write_sm_mask != 0 {
                add(&mut self.coherence.cross_sm_read_write_lines, 1)?;
            }
            if state.write_sm_mask.count_ones() > 1 {
                add(&mut self.coherence.cross_sm_write_write_lines, 1)?;
            }
        }
        self.coherence_lines.clear();
        Ok(())
    }
    pub(crate) fn observe_control_flow(
        &mut self,
        from_function: i32,
        from_pc: u32,
        from_active: u32,
        to_function: i32,
        to_pc: u32,
        to_active: u32,
    ) -> Result<(), crate::IssueError> {
        pub(super) const MAX_DYNAMIC_EDGES: usize = 65_536;
        if from_active == 0 || to_active == 0 {
            return Ok(());
        }
        let overlap = from_active & to_active;
        if overlap == 0 {
            add(&mut self.control_flow.disjoint_path_switches, 1)?;
            return Ok(());
        }
        let key = format!("{from_function}:{from_pc:#x}->{to_function}:{to_pc:#x}");
        if !self.control_flow.edges.contains_key(&key)
            && self.control_flow.edges.len() >= MAX_DYNAMIC_EDGES
        {
            return Err(crate::IssueError::CounterOverflow {
                counter: "control-flow edge arena",
            });
        }
        let count = self.control_flow.edges.entry(key).or_default();
        add(count, 1)?;
        add(&mut self.control_flow.observed_edges, 1)?;
        add(
            &mut self.control_flow.active_lane_transitions,
            u64::from(overlap.count_ones()),
        )?;
        self.control_flow.unique_edges =
            u64::try_from(self.control_flow.edges.len()).map_err(|_| {
                crate::IssueError::CounterOverflow {
                    counter: "control-flow unique edge count",
                }
            })?;
        Ok(())
    }
    pub(super) fn physical_candidates(
        head: &str,
        kind: &AccessKind,
        pipeline: crate::PipelineClass,
    ) -> &'static [&'static str] {
        pub(super) const NONE: &[&str] = &[];
        pub(super) const ADU_CBU: &[&str] = &["adu", "cbu"];
        pub(super) const ADU_UNIFORM: &[&str] = &["adu", "uniform"];
        pub(super) const ADU_LSU: &[&str] = &["adu", "lsu"];
        pub(super) const LSU: &[&str] = &["lsu"];
        pub(super) const TENSOR: &[&str] = &["tensor"];
        pub(super) const FMA: &[&str] = &["fma"];
        pub(super) const ALU: &[&str] = &["alu"];
        pub(super) const XU: &[&str] = &["xu"];
        if matches!(
            head,
            "BRA" | "BRX" | "BRXU" | "JMP" | "JMX" | "JMXU" | "CALL" | "RET" | "EXIT"
        ) {
            return ADU_CBU;
        }
        if matches!(head, "BAR" | "BSSY" | "BSYNC" | "BREAK") {
            return ADU_CBU;
        }
        if head == "LDGSTS" {
            return ADU_LSU;
        }
        if matches!(
            head,
            "ULDC" | "LDC" | "CS2R" | "S2R" | "S2UR" | "UMOV" | "USHF"
        ) {
            return ADU_UNIFORM;
        }
        if head == "LDGDEPBAR" {
            return ADU_UNIFORM;
        }
        // MUFU is the dedicated multi-function operation family.  Unlike
        // the broader conversion/check families, its SM86 physical domain is
        // unambiguously the transcendental/data-type-conversion unit; retain
        // it as a candidate rather than counting it as an unknown opcode.
        if head == "MUFU" {
            return XU;
        }
        if head.starts_with("LD")
            || head.starts_with("ST")
            || head.starts_with("ATOM")
            || head.starts_with("RED")
            || !matches!(
                kind,
                AccessKind::NonMemory | AccessKind::PredicatedOffMemory
            )
        {
            return LSU;
        }
        match pipeline {
            crate::PipelineClass::Tensor { .. } => TENSOR,
            crate::PipelineClass::ScalarAlu { .. } => {
                if matches!(Self::ncu_pipeline(head), Some("fma")) {
                    FMA
                } else {
                    ALU
                }
            }
            _ => match Self::ncu_pipeline(head) {
                Some("tensor") => TENSOR,
                Some("fma") => FMA,
                Some("alu") => ALU,
                _ => NONE,
            },
        }
    }
    pub(super) fn ncu_pipeline(head: &str) -> Option<&'static str> {
        if matches!(head, "HMMA" | "IMMA" | "BMMA" | "DMMA") {
            return Some("tensor");
        }
        if matches!(
            head,
            "IMAD" | "IMUL" | "FFMA" | "FADD" | "FMUL" | "FMAD" | "HADD2" | "HMUL2" | "HFMA2"
        ) {
            return Some("fma");
        }
        if matches!(
            head,
            "IADD3"
                | "UIADD3"
                | "LOP3"
                | "LEA"
                | "MOV"
                | "PRMT"
                | "ISETP"
                | "PSETP"
                | "SEL"
                | "SHF"
                | "F2FP"
                | "F2F"
                | "F2IP"
                | "I2FP"
                | "I2F"
        ) {
            return Some("alu");
        }
        // Nsight documents the LSU logical pipeline as including memory,
        // special-register, shuffle, and CTA-barrier traffic. DEPBAR is a
        // scheduler dependency instruction and is intentionally excluded.
        if head.starts_with("LD")
            || head.starts_with("ST")
            || head.starts_with("ATOM")
            || head.starts_with("RED")
            || matches!(head, "S2R" | "S2UR" | "SHFL")
        {
            return Some("lsu");
        }
        None
    }
    pub(super) fn candidate_schedule_timing(pipeline: crate::PipelineClass) -> (u64, u64) {
        match pipeline {
            crate::PipelineClass::DispatchOnly => (1, 1),
            crate::PipelineClass::Branch {
                next_issue_interval_cycles,
            } => (1, next_issue_interval_cycles.max(1)),
            crate::PipelineClass::Barrier {
                warp_initiation_interval_cycles,
                ..
            } => (1, warp_initiation_interval_cycles.max(1)),
            crate::PipelineClass::Tensor {
                initiation_interval_cycles,
            }
            | crate::PipelineClass::Sfu {
                initiation_interval_cycles,
            } => (1, initiation_interval_cycles.max(1)),
            crate::PipelineClass::ScalarAlu {
                latency_cycles,
                initiation_interval_cycles,
                ..
            } => (latency_cycles.max(1), initiation_interval_cycles.max(1)),
        }
    }
    pub(super) fn schedule_candidates(
        &mut self,
        candidates: &[&'static str],
        pipeline: crate::PipelineClass,
    ) -> Result<(), crate::IssueError> {
        let sequence = self.physical_pipeline.scheduler_sequence;
        self.physical_pipeline.scheduler_sequence =
            sequence
                .checked_add(1)
                .ok_or(crate::IssueError::CounterOverflow {
                    counter: "physical pipeline scheduler sequence",
                })?;
        if candidates.is_empty() {
            return Ok(());
        }
        let (latency, initiation_interval) = Self::candidate_schedule_timing(pipeline);
        let mut selected: Option<(&'static str, u64, u64)> = None;
        for &candidate in candidates {
            let free = self
                .physical_pipeline
                .scheduler_free
                .get(candidate)
                .copied()
                .unwrap_or(0);
            let start = sequence.max(free);
            let completion =
                start
                    .checked_add(latency)
                    .ok_or(crate::IssueError::CounterOverflow {
                        counter: "physical pipeline scheduler completion",
                    })?;
            let better = match selected {
                None => true,
                Some((best_candidate, best_start, best_completion)) => {
                    (completion, start, candidate) < (best_completion, best_start, best_candidate)
                }
            };
            if better {
                selected = Some((candidate, start, completion));
            }
        }
        let (candidate, start, completion) = selected.expect("candidate domain is nonempty");
        for &candidate in candidates {
            let free = self
                .physical_pipeline
                .scheduler_free
                .get(candidate)
                .copied()
                .unwrap_or(0);
            let candidate_start = sequence.max(free);
            let candidate_completion =
                candidate_start
                    .checked_add(latency)
                    .ok_or(crate::IssueError::CounterOverflow {
                        counter: "physical pipeline candidate interval completion",
                    })?;
            let key = candidate.to_owned();
            self.physical_pipeline
                .candidate_earliest_issue_cycles
                .entry(key.clone())
                .and_modify(|value| *value = (*value).min(candidate_start))
                .or_insert(candidate_start);
            self.physical_pipeline
                .candidate_latest_issue_cycles
                .entry(key.clone())
                .and_modify(|value| *value = (*value).max(candidate_start))
                .or_insert(candidate_start);
            self.physical_pipeline
                .candidate_earliest_completion_cycles
                .entry(key.clone())
                .and_modify(|value| *value = (*value).min(candidate_completion))
                .or_insert(candidate_completion);
            self.physical_pipeline
                .candidate_latest_completion_cycles
                .entry(key)
                .and_modify(|value| *value = (*value).max(candidate_completion))
                .or_insert(candidate_completion);
        }
        add(&mut self.physical_pipeline.model_interval_records, 1)?;
        let next_free =
            start
                .checked_add(initiation_interval)
                .ok_or(crate::IssueError::CounterOverflow {
                    counter: "physical pipeline scheduler availability",
                })?;
        self.physical_pipeline
            .scheduler_free
            .insert(candidate.to_owned(), next_free);
        let entry = self
            .physical_pipeline
            .scheduled_instructions
            .entry(candidate.to_owned())
            .or_default();
        add(entry, 1)?;
        let frontier = self.physical_pipeline.schedule_frontier_cycles.unwrap_or(0);
        self.physical_pipeline.schedule_frontier_cycles = Some(frontier.max(completion));
        Ok(())
    }
    #[allow(dead_code)]
    pub(crate) fn observe(
        &mut self,
        opcode: &str,
        active: u8,
        effective: u8,
        kind: AccessKind,
        pipeline: crate::PipelineClass,
        sector_arena: &[i64],
    ) -> Result<(), crate::IssueError> {
        self.observe_with_mask_presence(
            opcode,
            active,
            effective,
            true,
            true,
            kind,
            pipeline,
            sector_arena,
        )
    }
    pub(crate) fn observe_with_mask_presence(
        &mut self,
        opcode: &str,
        active: u8,
        effective: u8,
        active_mask_present: bool,
        predicate_mask_present: bool,
        kind: AccessKind,
        pipeline: crate::PipelineClass,
        sector_arena: &[i64],
    ) -> Result<(), crate::IssueError> {
        self.observe_with_mask_presence_arch(
            opcode,
            active,
            effective,
            active_mask_present,
            predicate_mask_present,
            kind,
            pipeline,
            sector_arena,
            sim_arch::ArchId::Sm86,
        )
    }
    pub(crate) fn observe_with_mask_presence_arch(
        &mut self,
        opcode: &str,
        active: u8,
        effective: u8,
        active_mask_present: bool,
        predicate_mask_present: bool,
        kind: AccessKind,
        pipeline: crate::PipelineClass,
        _sector_arena: &[i64],
        arch: sim_arch::ArchId,
    ) -> Result<(), crate::IssueError> {
        self.instructions.observe(
            active,
            effective,
            active_mask_present,
            predicate_mask_present,
        )?;
        let head = opcode_head(opcode);
        match head {
            "LDC" => add(&mut self.constant_client_ldc_instructions, 1)?,
            "ULDC" => add(&mut self.constant_client_uldc_instructions, 1)?,
            _ => {}
        }
        self.by_opcode.entry(head.to_owned()).or_default().observe(
            active,
            effective,
            active_mask_present,
            predicate_mask_present,
        )?;
        let pipe = match pipeline {
            crate::PipelineClass::DispatchOnly => "dispatch_only",
            crate::PipelineClass::Tensor { .. } => "tensor",
            crate::PipelineClass::ScalarAlu { .. } => "scalar_alu",
            crate::PipelineClass::Sfu { .. } => "uniform_sfu",
            crate::PipelineClass::Barrier { .. } => "barrier",
            crate::PipelineClass::Branch { .. } => "branch",
        };
        self.by_pipeline
            .entry(pipe.to_owned())
            .or_default()
            .observe(
                active,
                effective,
                active_mask_present,
                predicate_mask_present,
            )?;
        // NCU's `smsp__inst_executed_pipe_*` rows are warp-level instruction
        // populations and, like `smsp__inst_executed.sum`, retain
        // predicated-off issue records. Thread/predicate counts in the nested
        // activity row still preserve the active/effective distinction. Keep
        // unclassified heads visible instead of silently dropping them: on
        // SM86 some instructions can issue through more than one physical
        // path, and the trace does not carry the dynamic pipe assignment.
        let ncu_pipe = Self::ncu_pipeline(head).unwrap_or("unclassified");
        self.by_ncu_pipeline
            .entry(ncu_pipe.to_owned())
            .or_default()
            .observe(
                active,
                effective,
                active_mask_present,
                predicate_mask_present,
            )?;
        if effective != 0 {
            if let AccessKind::GlobalAsyncCopy { l1_policy, .. } = kind {
                match l1_policy {
                    LdgstsL1Policy::CacheAtAllLevels => {
                        add(&mut self.ldgsts_cache_access_instructions, 1)?;
                    }
                    LdgstsL1Policy::BypassL1 => {
                        add(&mut self.ldgsts_cache_bypass_instructions, 1)?;
                    }
                }
            }
        }
        let candidates = Self::physical_candidates(head, &kind, pipeline);
        self.schedule_candidates(candidates, pipeline)?;
        add(&mut self.physical_pipeline.total_warp_instructions, 1)?;
        if candidates.is_empty() {
            add(&mut self.physical_pipeline.unknown_instructions, 1)?;
        } else if candidates.len() == 1 {
            add(&mut self.physical_pipeline.unique_instructions, 1)?;
            add(
                self.physical_pipeline
                    .mandatory_class_instructions
                    .entry(candidates[0].to_owned())
                    .or_default(),
                1,
            )?;
        } else {
            add(&mut self.physical_pipeline.ambiguous_instructions, 1)?;
        }
        for candidate in candidates {
            let entry = self
                .physical_pipeline
                .candidate_instructions
                .entry((*candidate).to_owned())
                .or_default();
            add(entry, 1)?;
        }
        if candidates.is_empty() {
            add(&mut self.physical_pipeline.selected_unknown_instructions, 1)?;
        } else {
            let selected = candidates
                .iter()
                .min_by_key(|candidate| {
                    (
                        self.physical_pipeline
                            .selected_instructions
                            .get(**candidate)
                            .copied()
                            .unwrap_or(0),
                        **candidate,
                    )
                })
                .expect("nonempty candidate domain");
            let entry = self
                .physical_pipeline
                .selected_instructions
                .entry((*selected).to_owned())
                .or_default();
            add(entry, 1)?;
        }
        // Nsight's SMSP branch counter is the CBU instruction population,
        // excluding only BMOV and BSSY.  The old projection counted only the
        // direct branch heads, which systematically omitted control-flow
        // instructions such as BSYNC, EXIT, WARPSYNC, BREAK, and CONT.  Keep
        // this separate from the SASS branch subset (the `sm__sass_*` metric
        // family) and from the branch-target observer below: this is a
        // warp-instruction cardinality, so predicated-off records still count
        // once when the warp executed the instruction.
        let branch_counter = if arch == sim_arch::ArchId::Sm120 {
            matches!(
                head,
                "BRA"
                    | "BRX"
                    | "BRXU"
                    | "JMP"
                    | "JMX"
                    | "JMXU"
                    | "CALL"
                    | "RET"
                    | "EXIT"
                    | "BSYNC"
                    | "WARPSYNC"
                    | "YIELD"
                    | "BREAK"
                    | "CONT"
            )
        } else {
            matches!(head, "BRA" | "BRX" | "BRXU" | "JMP" | "JMX" | "JMXU")
        };
        if branch_counter {
            add(&mut self.branch_warp_instructions, 1)?;
            add(
                &mut self.branch_mixed_predicate_warp_instructions,
                u64::from(
                    active_mask_present
                        && predicate_mask_present
                        && effective > 0
                        && effective < active,
                ),
            )?;
        }
        if effective == 0 {
            return Ok(());
        }
        if let AccessKind::GlobalStore { off, len } = kind {
            let end =
                off.checked_add(usize::from(len))
                    .ok_or(crate::IssueError::CounterOverflow {
                        counter: "global-store sector span",
                    })?;
            if let Some(span) = _sector_arena.get(off..end) {
                let current = span.iter().copied().collect::<BTreeSet<_>>();
                if effective >= 8 && self.global_store_previous_effective_lanes >= 8 {
                    let duplicate = current
                        .intersection(&self.global_store_previous_sectors)
                        .count() as u64;
                    add(&mut self.global_store_adjacent_duplicate_sectors, duplicate)?;
                }
                self.global_store_previous_sectors = current;
                self.global_store_previous_effective_lanes = effective;
            } else {
                self.global_store_previous_sectors.clear();
                self.global_store_previous_effective_lanes = 0;
            }
        } else {
            self.global_store_previous_sectors.clear();
            self.global_store_previous_effective_lanes = 0;
        }
        if arch == sim_arch::ArchId::Sm120
            && matches!(kind, AccessKind::Shared { .. })
            && head == "LDS"
        {
            let sass_head = opcode.split_ascii_whitespace().next().unwrap_or(opcode);
            let width16 = sass_head.split('.').any(|token| token == "128");
            let width4 = !width16 && !sass_head.split('.').any(|token| token == "64");
            if effective == 32 && width16 {
                add(&mut self.shared_load_full_warp_width16_records, 1)?;
                if let AccessKind::Shared {
                    data_wavefronts,
                    data_ideal_wavefronts,
                    ..
                } = kind
                {
                    if data_wavefronts.saturating_mul(2) == data_ideal_wavefronts {
                        add(
                            &mut self.shared_load_width16_observer_reduction,
                            data_wavefronts,
                        )?;
                        let current_excess = data_wavefronts.saturating_sub(data_ideal_wavefronts);
                        let original_excess = data_wavefronts
                            .saturating_mul(2)
                            .saturating_sub(data_ideal_wavefronts);
                        add(
                            &mut self.shared_load_width16_excess_reduction,
                            original_excess.saturating_sub(current_excess),
                        )?;
                    }
                }
            } else if effective == 32 && width4 {
                add(&mut self.shared_load_full_warp_width4_records, 1)?;
            }
        }
        let (key, sectors, bank_wavefronts) = match kind {
            AccessKind::NonMemory | AccessKind::PredicatedOffMemory => return Ok(()),
            AccessKind::Shared {
                data_wavefronts, ..
            } => (
                // Collective matrix stores are the store-side counterpart
                // of LDSM.  Keeping STSM in the load bucket makes the
                // operation-qualified opcode counters report matrix stores
                // as shared loads and leaves the aggregate store-wavefront
                // observer empty.  Preserve both matrix forms in their
                // direction-specific buckets; the conflict surface below
                // still excludes them from ordinary LDS/STS bank conflicts.
                if matches!(head, "STS" | "STSM") {
                    "shared_store"
                } else if head.starts_with("ATOM") {
                    "shared_atomic"
                } else {
                    "shared_load"
                },
                0,
                data_wavefronts,
            ),
            AccessKind::LocalLoad { len, .. } => ("local_load", u64::from(len), 0),
            AccessKind::LocalStore { len, .. } => ("local_store", u64::from(len), 0),
            AccessKind::GlobalStore { len, .. } => ("global_store", u64::from(len), 0),
            AccessKind::GlobalReduction { len, .. } => (
                "global_reduction",
                // RED is a non-returning, lane-serialized write operation.
                // The validated sector span deliberately preserves repeated
                // aliases: GA10x reports those transaction sectors rather
                // than collapsing them to unique sector IDs (the same
                // convention is used by the atomic projection and by the
                // issue-side alias-aware wavefront counter).
                u64::from(len),
                0,
            ),
            AccessKind::GlobalBypass {
                token: CacheOpToken::Atom,
                off: _,
                len,
                ..
            } => (
                "global_atomic",
                // The validated atomic sector arena repeats each touched
                // sector by its maximum same-word alias multiplicity. This
                // reproduces GA10x's atom-sector surface: coalesced words use
                // unique sectors, while same-word aliases remain serialized.
                u64::from(len),
                0,
            ),
            AccessKind::GlobalAsyncCopy { len, .. } => ("async_global_load", u64::from(len), 0),
            AccessKind::GlobalEligible { len, .. } | AccessKind::GlobalBypass { len, .. } => {
                ("global_load", u64::from(len), 0)
            }
            AccessKind::GlobalUntagged { len, .. } => ("unclassified_global", u64::from(len), 0),
        };
        let row = self.memory.entry(key.to_owned()).or_default();
        add(&mut row.requests, 1)?;
        add(&mut row.sectors, sectors)?;
        if let Some((unique_sectors, unique_lines)) = decoded_sector_stats(kind, _sector_arena) {
            add(&mut row.unique_sectors, unique_sectors)?;
            add(&mut row.unique_lines, unique_lines)?;
            add(
                &mut row.sector_aliases,
                sectors.saturating_sub(unique_sectors),
            )?;
        }
        add(&mut row.bank_wavefronts, bank_wavefronts)?;
        if let AccessKind::Shared {
            data_wavefronts,
            data_ideal_wavefronts,
            ..
        } = kind
        {
            add(&mut row.ideal_bank_wavefronts, data_ideal_wavefronts)?;
            // Nsight's operation-qualified shared-load/store conflict rows
            // do not include matrix loads/stores. Keep those operations in
            // the aggregate wavefront surface, but exclude them from the
            // corresponding conflict numerator.
            let conflict_surface = !matches!(head, "LDSM" | "STSM");
            if conflict_surface {
                add(&mut row.bank_conflict_wavefronts, data_wavefronts)?;
                add(
                    &mut row.ideal_bank_conflict_wavefronts,
                    data_ideal_wavefronts,
                )?;
                add(
                    &mut row.bank_conflict_excess_wavefronts,
                    data_wavefronts.saturating_sub(data_ideal_wavefronts),
                )?;
            }
        }
        if let Some(scope) = memory_scope_key(opcode, kind) {
            let row = self.memory_scope.entry(scope.to_owned()).or_default();
            add(&mut row.requests, 1)?;
            add(&mut row.sectors, sectors)?;
            if let Some((unique_sectors, unique_lines)) = decoded_sector_stats(kind, _sector_arena)
            {
                add(&mut row.unique_sectors, unique_sectors)?;
                add(&mut row.unique_lines, unique_lines)?;
                add(
                    &mut row.sector_aliases,
                    sectors.saturating_sub(unique_sectors),
                )?;
            }
            add(&mut row.bank_wavefronts, bank_wavefronts)?;
        }
        Ok(())
    }
}

/// Half-open, model-relative lifetimes [admission, retirement). No hardware
/// timestamps are used. Retired warps no longer contribute even if their CTA
/// still retains resource allocations. Pending write acknowledgements are a
/// separate completion domain.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct WarpLifetime {
    pub block: i64,
    pub warp_slot: i32,
    pub admission: u64,
    pub retirement: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct SmActivity {
    pub warp_active_cycles: u128,
    pub sm_active_cycles: u64,
    pub max_resident_warps_per_sm: u32,
    pub peak_live_warps: u32,
    pub smsp_active_cycles: [u64; 4],
    pub instructions_per_smsp: [u64; 4],
    pub issue_active_cycles_per_smsp: [u64; 4],
    /// Number of distinct executed instruction sectors assigned to this SM.
    /// This is an observation-side scope anchor for a future per-SM effective
    /// front-end envelope; it does not alter replay timing.
    pub front_end_instruction_read_sectors: u64,
    pub warp_lifetimes: Vec<WarpLifetime>,
}

impl SmActivity {
    pub(crate) fn from_lifetimes(
        warp_lifetimes: Vec<WarpLifetime>,
        instructions_per_smsp: [u64; 4],
        issue_active_cycles_per_smsp: [u64; 4],
        max_resident_warps_per_sm: u32,
    ) -> Self {
        let warp_active_cycles: u128 = warp_lifetimes
            .iter()
            .map(|w| u128::from(w.retirement.saturating_sub(w.admission)))
            .sum();
        let mut intervals: Vec<_> = warp_lifetimes
            .iter()
            .map(|w| (w.admission, w.retirement))
            .collect();
        let sm_active_cycles = union_length(&mut intervals);
        let mut endpoints: Vec<(u64, i32)> = warp_lifetimes
            .iter()
            .flat_map(|w| [(w.admission, 1_i32), (w.retirement, -1_i32)])
            .collect();
        endpoints.sort_unstable_by_key(|&(time, delta)| (time, delta));
        let mut live = 0_i32;
        let mut peak_live_warps = 0_u32;
        for (_, delta) in endpoints {
            live = live.saturating_add(delta);
            peak_live_warps = peak_live_warps.max(u32::try_from(live).unwrap_or(0));
        }
        let smsp_active_cycles = std::array::from_fn(|sc| {
            intervals.clear();
            intervals.extend(
                warp_lifetimes
                    .iter()
                    .filter(|w| w.warp_slot as usize % 4 == sc)
                    .map(|w| (w.admission, w.retirement)),
            );
            union_length(&mut intervals)
        });
        Self {
            warp_active_cycles,
            sm_active_cycles,
            max_resident_warps_per_sm,
            peak_live_warps,
            smsp_active_cycles,
            instructions_per_smsp,
            issue_active_cycles_per_smsp,
            front_end_instruction_read_sectors: 0,
            warp_lifetimes,
        }
    }
}
