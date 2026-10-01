//! Timing-neutral native-replay observations. Counts are not profiler samples.
use std::collections::BTreeMap;

use serde::Serialize;
use sim_mem::{AccessKind, CacheOpToken, opcode_head};
use sim_semantics::{
    CandidateDomain, EvidenceKind, LatentFactors, LatentModelError, ResourceDemandEnvelope,
    ResourceDemandInterval,
};

use crate::sass_operands::FixedLatencyOperandStatus;

/// Static SM86 regular-register bank demand projected from admitted SASS
/// operands. This is a resource-pressure envelope, not a dynamic physical
/// pipe, SMSP, or issue-slot observation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct OperandBankPressure {
    pub projected_records: u64,
    pub gap_records: u64,
    pub source_bank_reads: [u64; 2],
    pub destination_bank_writes: [u64; 2],
}

impl OperandBankPressure {
    pub(crate) fn observe(
        &mut self,
        status: &FixedLatencyOperandStatus,
    ) -> Result<(), crate::IssueError> {
        let projection = match status {
            FixedLatencyOperandStatus::Projected(projection) => projection,
            FixedLatencyOperandStatus::Gap(_) => {
                add(&mut self.gap_records, 1)?;
                return Ok(());
            }
            FixedLatencyOperandStatus::NotApplicable => return Ok(()),
        };
        add(&mut self.projected_records, 1)?;
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

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct MemoryTraffic {
    pub requests: u64,
    pub sectors: u64,
    /// Distinct decoded 32-byte sectors in the request spans.  Atomic and
    /// reduction alias multiplicity is intentionally removed here; `sectors`
    /// retains the serialized transaction surface used by the issue model.
    pub unique_sectors: u64,
    /// Distinct 128-byte lines (four 32-byte sectors) touched by the decoded
    /// request spans.  This is a trace topology diagnostic, not an L2
    /// transaction counter.
    pub unique_lines: u64,
    /// Sector-span entries that repeat a decoded sector, normally same-word
    /// atomic/reduction aliases.  A missing/flat sector arena leaves this at
    /// zero rather than manufacturing an alias count.
    pub sector_aliases: u64,
    /// Shared 32-bank/4-byte-word service work, not NCU sampled stalls.
    pub bank_wavefronts: u64,
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
    mandatory_class_instructions: BTreeMap<String, u64>,
    #[serde(skip)]
    scheduler_free: BTreeMap<String, u64>,
    #[serde(skip)]
    scheduler_sequence: u64,
}

fn is_empty_string_map(value: &BTreeMap<String, u64>) -> bool {
    value.is_empty()
}

fn is_empty_string_u64_map(value: &BTreeMap<String, u64>) -> bool {
    value.is_empty()
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ClassCountInterval {
    pub lower: u64,
    pub upper: u64,
}

/// Independent per-class intervals are not jointly attainable extremes and
/// do not assert how any particular dynamic instruction was physically routed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PipelineClassEnvelope {
    pub role: &'static str,
    pub evidence: sim_semantics::EvidenceKind,
    pub total_warp_instructions: u64,
    pub unclassified_instructions: u64,
    pub unlisted_classes_possible: bool,
    pub counts: BTreeMap<String, ClassCountInterval>,
}

/// Lane-consistent dynamic control-flow graph reconstructed from consecutive
/// trace records. Disjoint active masks are retained as path-switch evidence
/// rather than fabricated as an edge between serialized divergent paths.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ControlFlowGraph {
    pub observed_edges: u64,
    pub unique_edges: u64,
    pub active_lane_transitions: u64,
    pub disjoint_path_switches: u64,
    pub edges: BTreeMap<String, u64>,
}

/// Trace-derived cross-SM line-sharing domain.  The observer is intentionally
/// static/topological: it reports lines whose decoded global accesses are
/// issued by more than one traced SM and whether those accesses contain both
/// read and write classes.  It does not infer temporal ownership, cache
/// invalidations, or completed coherence traffic.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct CoherenceObservation {
    pub decoded_global_lines: u64,
    pub cross_sm_lines: u64,
    pub cross_sm_sectors: u64,
    /// 32-byte sectors touched by more than one traced SM.  This is a
    /// tighter contention domain than `cross_sm_sectors`, which intentionally
    /// includes every sector on a line shared at line granularity.
    pub cross_sm_sector_overlap: u64,
    pub cross_sm_read_write_lines: u64,
    pub cross_sm_write_write_lines: u64,
    /// Per-sector access-class overlap.  These are trace-topology
    /// prerequisites for protocol work, not coherence transactions.
    pub cross_sm_read_write_sectors: u64,
    pub cross_sm_write_write_sectors: u64,
    pub unknown_sm_or_line_records: u64,
    /// Count of global records carrying the producer's warp-local
    /// instrumentation-order token. This never orders different warps/SMs.
    pub instrumentation_order_records: u64,
    pub instrumentation_order_missing_records: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct CoherenceLineState {
    sm_mask: u128,
    read_sm_mask: u128,
    write_sm_mask: u128,
    sector_mask: u8,
    sector_sm_masks: [u128; 4],
    sector_read_sm_masks: [u128; 4],
    sector_write_sm_masks: [u128; 4],
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct TraceMetrics {
    /// Captured launch metadata, not inferred from lane activity or NCU.
    pub launch_resources: Option<LaunchResources>,
    pub instructions: InstructionActivity,
    pub by_opcode: BTreeMap<String, InstructionActivity>,
    pub memory: BTreeMap<String, MemoryTraffic>,
    pub by_pipeline: BTreeMap<String, InstructionActivity>,
    /// NCU-style logical pipe projections. These are deliberately separate
    /// from the issue scheduler's resource classes above.
    pub by_ncu_pipeline: BTreeMap<String, InstructionActivity>,
    /// Conservative physical-pipeline candidate domains. These are not NCU
    /// counters; they quantify how much dynamic pipe attribution the trace
    /// contract can support without value or hardware-selection state.
    pub physical_pipeline: PhysicalPipelineAttribution,
    /// Static operand-bank demand envelope for the admitted SM86 SASS forms;
    /// it is intentionally separate from physical pipeline attribution.
    #[serde(skip)]
    pub operand_bank_pressure: OperandBankPressure,
    pub control_flow: ControlFlowGraph,
    /// Conservative cross-SM sharing domain; not a coherence protocol or
    /// profiler writeback counter.
    pub coherence: CoherenceObservation,
    #[serde(skip)]
    pub(crate) coherence_lines: BTreeMap<i64, CoherenceLineState>,
    pub branch_warp_instructions: u64,
    pub branch_mixed_predicate_warp_instructions: u64,
    /// Actual per-lane successors, independent of branch predicates.
    pub branch_targets: BranchTargets,
    pub sectors_decoded: bool,
}

/// Warp-local control-transfer observations. Counts include fallthrough;
/// EXIT and convergence setup/waits are not target-bearing branch events.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct BranchTargets {
    pub events: u64,
    pub uniform_events: u64,
    pub divergent_events: u64,
    pub distinct_successors: u64,
    /// All active threads in a divergent event, not the minority path size.
    pub divergent_threads: u64,
    /// Sum of active lanes across all resolved successor events.
    pub successor_active_threads: u64,
    /// Sum of the smallest successor-path population for resolved divergent
    /// events. This is a trace-derived minority-path diagnostic, not an NCU
    /// branch counter.
    pub divergent_minority_threads: u64,
    /// Active lanes in resolved uniform successor events.
    pub uniform_active_threads: u64,
    /// At least one active lane had no recorded successor before trace end.
    pub unresolved_events: u64,
    /// Active lanes whose first successor was not observed before trace end.
    pub unresolved_threads: u64,
}

struct PendingBranch {
    remaining: u32,
    active_threads: u32,
    targets: Vec<(i32, u32, u32)>,
}

/// Observe the first subsequent instruction executed by EACH active lane,
/// even when another path executes first. Unresolved lanes of different
/// pending events are disjoint, bounding this state to at most 32 events.
/// This observer does not change the scheduler, trace, or cache state.
#[derive(Default)]
pub(crate) struct BranchSuccessors {
    pending: Vec<PendingBranch>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct LaunchResources {
    pub block_threads: u32,
    pub grid_blocks: u64,
    pub registers_per_thread: u32,
    pub shared_memory_bytes_per_block: u32,
    /// Legacy captures supply only the total; absence must not become zero.
    pub static_shared_memory_bytes: Option<u32>,
    pub dynamic_shared_memory_bytes: Option<u32>,
    pub shared_memory_carveout_preference_pct: Option<i32>,
    pub driver_occupancy_limit_blocks: Option<u32>,
}

fn add(dst: &mut u64, value: u64) -> Result<(), crate::IssueError> {
    *dst = dst
        .checked_add(value)
        .ok_or(crate::IssueError::CounterOverflow {
            counter: "native metric counter",
        })?;
    Ok(())
}

impl InstructionActivity {
    fn observe(
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

/// Return the distinct-sector and distinct-line topology for one decoded
/// request span.  The arena is already validated by the trace reader; this
/// helper stays fail-closed for unit fixtures and legacy flat traces whose
/// arena is intentionally absent.
fn decoded_sector_stats(kind: AccessKind, sector_arena: &[i64]) -> Option<(u64, u64)> {
    let (off, len) = match kind {
        AccessKind::LocalLoad { off, len }
        | AccessKind::LocalStore { off, len }
        | AccessKind::GlobalEligible { off, len, .. }
        | AccessKind::GlobalAsyncCopy { off, len, .. }
        | AccessKind::GlobalBypass { off, len, .. }
        | AccessKind::GlobalUntagged { off, len }
        | AccessKind::GlobalStore { off, len }
        | AccessKind::GlobalReduction { off, len, .. } => (off, usize::from(len)),
        AccessKind::Shared { .. } | AccessKind::NonMemory | AccessKind::PredicatedOffMemory => {
            return Some((0, 0));
        }
    };
    let end = off.checked_add(len)?;
    let span = sector_arena.get(off..end)?;
    let mut sorted = span.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut unique_lines = 0_u64;
    let mut previous_line = None;
    for sector in sorted.iter().copied() {
        let line = sector.div_euclid(4);
        if previous_line != Some(line) {
            unique_lines = unique_lines.saturating_add(1);
            previous_line = Some(line);
        }
    }
    Some((sorted.len() as u64, unique_lines))
}

impl TraceMetrics {
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
    const MAX_COHERENCE_LINES: usize = 1_000_000;

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
                    | sim_mem::CacheOpToken::LdgEEf,
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
        const MAX_DYNAMIC_EDGES: usize = 65_536;
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

    fn physical_candidates(
        head: &str,
        kind: &AccessKind,
        pipeline: crate::PipelineClass,
    ) -> &'static [&'static str] {
        const NONE: &[&str] = &[];
        const ADU_CBU: &[&str] = &["adu", "cbu"];
        const ADU_UNIFORM: &[&str] = &["adu", "uniform"];
        const ADU_LSU: &[&str] = &["adu", "lsu"];
        const LSU: &[&str] = &["lsu"];
        const TENSOR: &[&str] = &["tensor"];
        const FMA: &[&str] = &["fma"];
        const ALU: &[&str] = &["alu"];
        const XU: &[&str] = &["xu"];
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

    fn ncu_pipeline(head: &str) -> Option<&'static str> {
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

    fn candidate_schedule_timing(pipeline: crate::PipelineClass) -> (u64, u64) {
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

    fn schedule_candidates(
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
        _sector_arena: &[i64],
    ) -> Result<(), crate::IssueError> {
        self.instructions.observe(
            active,
            effective,
            active_mask_present,
            predicate_mask_present,
        )?;
        let head = opcode_head(opcode);
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
        if matches!(head, "BRA" | "BRX" | "BRXU" | "JMP" | "JMX" | "JMXU") {
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
        let (key, sectors, bank_wavefronts) = match kind {
            AccessKind::NonMemory | AccessKind::PredicatedOffMemory => return Ok(()),
            AccessKind::Shared { wavefronts } => (
                if head == "STS" {
                    "shared_store"
                } else if head.starts_with("ATOM") {
                    "shared_atomic"
                } else {
                    "shared_load"
                },
                0,
                wavefronts,
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
        add(&mut row.bank_wavefronts, bank_wavefronts)
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
    pub warp_lifetimes: Vec<WarpLifetime>,
}

fn union_length(intervals: &mut [(u64, u64)]) -> u64 {
    intervals.sort_unstable();
    let mut end = 0;
    let mut length: u64 = 0;
    for &(begin, stop) in intervals.iter() {
        length = length.saturating_add(stop.saturating_sub(begin.max(end)));
        end = end.max(stop);
    }
    // Disjoint intervals fit within one u64 timeline; summation cannot overflow.
    length
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
            warp_lifetimes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn branch_successors_resolve_per_lane_targets() {
        let mut successors = BranchSuccessors::default();
        let mut targets = BranchTargets::default();
        successors
            .observe(&mut targets, 1, 0, 0b11, "BRA 0x40")
            .unwrap();
        successors
            .observe(&mut targets, 1, 0x40, 0b01, "IADD3")
            .unwrap();
        successors
            .observe(&mut targets, 1, 0x50, 0b10, "IADD3")
            .unwrap();
        successors.finish(&mut targets).unwrap();
        assert_eq!(targets.events, 1);
        assert_eq!(targets.divergent_events, 1);
        assert_eq!(targets.uniform_events, 0);
        assert_eq!(targets.distinct_successors, 2);
        assert_eq!(targets.divergent_threads, 2);
        assert_eq!(targets.successor_active_threads, 2);
        assert_eq!(targets.divergent_minority_threads, 1);
        assert_eq!(targets.uniform_active_threads, 0);
        assert_eq!(targets.unresolved_events, 0);
        assert_eq!(targets.unresolved_threads, 0);
    }

    #[test]
    fn branch_successors_keep_uniform_and_unresolved_domains_separate() {
        let mut successors = BranchSuccessors::default();
        let mut targets = BranchTargets::default();
        successors
            .observe(&mut targets, 1, 0, 0b11, "BRA 0x40")
            .unwrap();
        successors
            .observe(&mut targets, 1, 0x40, 0b11, "IADD3")
            .unwrap();
        successors
            .observe(&mut targets, 1, 0x50, 0b11, "BRA 0x80")
            .unwrap();
        successors.finish(&mut targets).unwrap();
        assert_eq!(targets.events, 2);
        assert_eq!(targets.uniform_events, 1);
        assert_eq!(targets.divergent_events, 0);
        assert_eq!(targets.successor_active_threads, 2);
        assert_eq!(targets.uniform_active_threads, 2);
        assert_eq!(targets.unresolved_events, 1);
        assert_eq!(targets.unresolved_threads, 2);
    }

    #[test]
    fn occupancy_uses_union_and_integral_not_terminal_max_or_allocated_slots() {
        let a = SmActivity::from_lifetimes(
            vec![
                WarpLifetime {
                    warp_slot: 0,
                    admission: 0,
                    retirement: 10,
                    ..Default::default()
                },
                WarpLifetime {
                    warp_slot: 1,
                    admission: 2,
                    retirement: 6,
                    ..Default::default()
                },
                WarpLifetime {
                    warp_slot: 0,
                    admission: 12,
                    retirement: 15,
                    ..Default::default()
                },
            ],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            48,
        );
        assert_eq!(a.warp_active_cycles, 17);
        assert_eq!(a.sm_active_cycles, 13);
        assert_eq!(a.peak_live_warps, 2);
        assert_eq!(a.smsp_active_cycles, [13, 4, 0, 0]);
    }

    #[test]
    fn predicated_off_memory_has_instruction_but_no_request() {
        let mut m = TraceMetrics::default();
        m.observe(
            "STG.E [R2], R4",
            16,
            0,
            AccessKind::PredicatedOffMemory,
            crate::PipelineClass::DispatchOnly,
            &[],
        )
        .unwrap();
        assert_eq!(m.instructions.warp_instructions, 1);
        assert_eq!(m.instructions.active_thread_instructions, 16);
        assert_eq!(m.instructions.predicate_on_thread_instructions, 0);
        assert!(m.memory.is_empty());
    }

    #[test]
    fn missing_lane_masks_are_not_counted_as_all_lanes_inactive() {
        let mut m = TraceMetrics::default();
        m.observe_with_mask_presence(
            "IADD3",
            0,
            0,
            false,
            false,
            AccessKind::NonMemory,
            crate::PipelineClass::DispatchOnly,
            &[],
        )
        .unwrap();
        assert_eq!(m.instructions.warp_instructions, 1);
        assert_eq!(m.instructions.active_thread_instructions, 0);
        assert_eq!(m.instructions.predicate_on_thread_instructions, 0);
        assert_eq!(m.instructions.predicated_off_warp_instructions, 0);
        assert_eq!(m.instructions.active_mask_missing_instructions, 1);
        assert_eq!(m.instructions.predicate_mask_missing_instructions, 1);
    }

    #[test]
    fn nonfull_lane_populations_are_observed_without_being_silently_full() {
        let mut m = TraceMetrics::default();
        m.observe_with_mask_presence(
            "IADD3",
            16,
            8,
            true,
            true,
            AccessKind::NonMemory,
            crate::PipelineClass::DispatchOnly,
            &[],
        )
        .unwrap();
        assert_eq!(m.instructions.active_mask_nonfull_instructions, 1);
        assert_eq!(m.instructions.predicate_mask_nonfull_instructions, 1);
        assert_eq!(m.instructions.active_mask_missing_instructions, 0);
        assert_eq!(m.instructions.predicate_mask_missing_instructions, 0);
    }

    #[test]
    fn memory_topology_separates_unique_lines_from_atomic_aliases() {
        let mut m = TraceMetrics::default();
        m.observe(
            "ATOM.E.ADD [R2], R4",
            32,
            32,
            AccessKind::GlobalBypass {
                off: 0,
                len: 4,
                token: CacheOpToken::Atom,
                l2_requests: 2,
            },
            crate::PipelineClass::DispatchOnly,
            &[0, 1, 1, 4],
        )
        .unwrap();
        let traffic = &m.memory["global_atomic"];
        assert_eq!(traffic.requests, 1);
        assert_eq!(traffic.sectors, 4);
        assert_eq!(traffic.unique_sectors, 3);
        assert_eq!(traffic.unique_lines, 2);
        assert_eq!(traffic.sector_aliases, 1);
    }

    #[test]
    fn memory_topology_stays_empty_for_flat_sector_arena() {
        let mut m = TraceMetrics::default();
        m.observe(
            "LDG.E R4, [R2]",
            32,
            32,
            AccessKind::GlobalEligible {
                off: 0,
                len: 2,
                token: CacheOpToken::LdgECa,
            },
            crate::PipelineClass::DispatchOnly,
            &[],
        )
        .unwrap();
        let traffic = &m.memory["global_load"];
        assert_eq!(traffic.sectors, 2);
        assert_eq!(traffic.unique_sectors, 0);
        assert_eq!(traffic.unique_lines, 0);
        assert_eq!(traffic.sector_aliases, 0);
    }

    #[test]
    fn ncu_pipeline_projection_retains_predicated_off_issue_records() {
        let mut m = TraceMetrics::default();
        m.observe(
            "IMAD.MOV.U32",
            32,
            32,
            AccessKind::NonMemory,
            crate::PipelineClass::ScalarAlu {
                latency_cycles: 4,
                initiation_interval_cycles: 1,
                in_flight_capacity: 1,
            },
            &[],
        )
        .unwrap();
        m.observe(
            "IMAD.MOV.U32",
            32,
            0,
            AccessKind::PredicatedOffMemory,
            crate::PipelineClass::ScalarAlu {
                latency_cycles: 4,
                initiation_interval_cycles: 1,
                in_flight_capacity: 1,
            },
            &[],
        )
        .unwrap();
        assert_eq!(m.instructions.warp_instructions, 2);
        assert_eq!(m.by_ncu_pipeline["fma"].warp_instructions, 2);
    }

    #[test]
    fn ncu_pipeline_projection_retains_unclassified_opcode_population() {
        let mut m = TraceMetrics::default();
        m.observe(
            "NOP",
            32,
            32,
            AccessKind::NonMemory,
            crate::PipelineClass::DispatchOnly,
            &[],
        )
        .unwrap();
        assert_eq!(m.by_ncu_pipeline["unclassified"].warp_instructions, 1);
    }

    #[test]
    fn logical_pipeline_mapping_distinguishes_funnel_shift_and_warp_shuffle() {
        let mut m = TraceMetrics::default();
        for opcode in ["SHF.R.U32.HI R1, RZ, 1, R2", "SHFL.BFLY R1, R2, 1"] {
            m.observe(
                opcode,
                32,
                32,
                AccessKind::NonMemory,
                crate::PipelineClass::DispatchOnly,
                &[],
            )
            .unwrap();
        }
        assert_eq!(m.by_ncu_pipeline["alu"].warp_instructions, 1);
        assert_eq!(m.by_ncu_pipeline["lsu"].warp_instructions, 1);
    }

    #[test]
    fn physical_pipeline_domains_keep_ambiguous_branch_candidates_explicit() {
        let mut m = TraceMetrics::default();
        m.observe(
            "BRA 0x40",
            32,
            32,
            AccessKind::NonMemory,
            crate::PipelineClass::Branch {
                next_issue_interval_cycles: 1,
            },
            &[],
        )
        .unwrap();
        assert_eq!(m.physical_pipeline.total_warp_instructions, 1);
        assert_eq!(m.physical_pipeline.unique_instructions, 0);
        assert_eq!(m.physical_pipeline.ambiguous_instructions, 1);
        assert_eq!(m.physical_pipeline.unknown_instructions, 0);
        assert_eq!(m.physical_pipeline.candidate_instructions["adu"], 1);
        assert_eq!(m.physical_pipeline.candidate_instructions["cbu"], 1);
        assert_eq!(m.physical_pipeline.selected_instructions["adu"], 1);
        assert_eq!(m.physical_pipeline.selected_unknown_instructions, 0);
        assert_eq!(m.physical_pipeline.scheduled_instructions["adu"], 1);
        assert_eq!(m.physical_pipeline.schedule_frontier_cycles, Some(1));
        assert_eq!(m.physical_pipeline.model_interval_records, 1);
        assert_eq!(
            m.physical_pipeline.candidate_earliest_issue_cycles["adu"],
            0
        );
        assert_eq!(m.physical_pipeline.candidate_latest_issue_cycles["cbu"], 0);
        assert_eq!(
            m.physical_pipeline.candidate_earliest_completion_cycles["adu"],
            1
        );
        assert_eq!(
            m.physical_pipeline.candidate_latest_completion_cycles["cbu"],
            1
        );
        assert_eq!(
            m.physical_pipeline.selection_policy,
            "min_selected_then_lexical_v1"
        );
        assert_eq!(
            m.physical_pipeline.schedule_policy,
            "candidate_earliest_completion_v1_model_only"
        );
    }

    #[test]
    fn operand_bank_pressure_is_a_separate_static_envelope() {
        let control = crate::control_bits::ControlBits::new(0, false, 7, 7, 0, 0).unwrap();
        let projected = crate::sass_operands::project_fixed_latency_operands(
            "FFMA R0, R1, R2, R3",
            control,
            sim_arch::ArchId::Sm86,
        );
        let mut pressure = OperandBankPressure::default();
        pressure.observe(&projected).unwrap();
        assert_eq!(pressure.projected_records, 1);
        assert_eq!(pressure.gap_records, 0);
        assert_eq!(pressure.source_bank_reads, [1, 2]);
        assert_eq!(pressure.destination_bank_writes, [1, 0]);

        let gap = crate::sass_operands::project_fixed_latency_operands(
            "FFMA R0, R1",
            control,
            sim_arch::ArchId::Sm86,
        );
        pressure.observe(&gap).unwrap();
        assert_eq!(pressure.gap_records, 1);
    }

    #[test]
    fn latent_factors_keep_dynamic_routes_unknown_beside_class_bounds() {
        let mut m = TraceMetrics::default();
        m.observe(
            "BRA 0x40",
            32,
            32,
            AccessKind::NonMemory,
            crate::PipelineClass::Branch {
                next_issue_interval_cycles: 1,
            },
            &[],
        )
        .unwrap();
        let factors = m.latent_factors().unwrap();
        assert!(factors.route.is_unknown());
        let envelope = m.physical_pipeline.class_count_envelope().unwrap();
        assert_eq!(
            envelope.counts["adu"],
            ClassCountInterval { lower: 0, upper: 1 }
        );
        assert_eq!(
            envelope.counts["cbu"],
            ClassCountInterval { lower: 0, upper: 1 }
        );
        assert_eq!(envelope.evidence, sim_semantics::EvidenceKind::Hypothesis);
        assert!(factors.compiler.is_unknown());
        assert!(factors.cache.is_unknown());
        assert!(factors.timing.is_unknown());
    }

    #[test]
    fn resource_demand_envelope_keeps_candidate_route_bounds() {
        let mut m = TraceMetrics::default();
        m.observe(
            "BRA 0x40",
            32,
            32,
            AccessKind::NonMemory,
            crate::PipelineClass::Branch {
                next_issue_interval_cycles: 1,
            },
            &[],
        )
        .unwrap();
        let envelope = m.physical_pipeline.resource_demand_envelope().unwrap();
        assert_eq!(envelope.total_events, 1);
        assert_eq!(envelope.unclassified_events, 0);
        assert_eq!(
            envelope.pools["adu"],
            sim_semantics::ResourceDemandInterval::interval(0, 1)
        );
        assert_eq!(
            envelope.pools["cbu"],
            sim_semantics::ResourceDemandInterval::interval(0, 1)
        );
        assert_eq!(envelope.evidence, sim_semantics::EvidenceKind::Hypothesis);
    }

    #[test]
    fn physical_pipeline_domains_keep_exit_in_control_candidate_domain() {
        let mut m = TraceMetrics::default();
        m.observe(
            "EXIT",
            32,
            32,
            AccessKind::NonMemory,
            crate::PipelineClass::Branch {
                next_issue_interval_cycles: 1,
            },
            &[],
        )
        .unwrap();
        assert_eq!(m.physical_pipeline.unknown_instructions, 0);
        assert_eq!(m.physical_pipeline.ambiguous_instructions, 1);
        assert_eq!(m.physical_pipeline.candidate_instructions["adu"], 1);
        assert_eq!(m.physical_pipeline.candidate_instructions["cbu"], 1);
    }

    #[test]
    fn physical_pipeline_domains_classify_mufu_as_xu() {
        let mut m = TraceMetrics::default();
        m.observe(
            "MUFU.SQRT",
            32,
            32,
            AccessKind::NonMemory,
            crate::PipelineClass::Sfu {
                initiation_interval_cycles: 1,
            },
            &[],
        )
        .unwrap();
        assert_eq!(m.physical_pipeline.unique_instructions, 1);
        assert_eq!(m.physical_pipeline.unknown_instructions, 0);
        assert_eq!(m.physical_pipeline.candidate_instructions["xu"], 1);
    }

    #[test]
    fn control_flow_edges_exclude_disjoint_serialized_paths() {
        let mut m = TraceMetrics::default();
        m.observe_control_flow(1, 0x10, 0b11, 1, 0x20, 0b01)
            .unwrap();
        m.observe_control_flow(1, 0x20, 0b01, 1, 0x30, 0b01)
            .unwrap();
        m.observe_control_flow(1, 0x30, 0b01, 1, 0x40, 0b10)
            .unwrap();
        assert_eq!(m.control_flow.observed_edges, 2);
        assert_eq!(m.control_flow.unique_edges, 2);
        assert_eq!(m.control_flow.active_lane_transitions, 2);
        assert_eq!(m.control_flow.disjoint_path_switches, 1);
    }

    #[test]
    fn coherence_domain_counts_cross_sm_line_sharing_without_order_claims() {
        let mut m = TraceMetrics::default();
        let sectors = [0_i64, 1, 4];
        m.observe_coherence(
            Some(0),
            32,
            AccessKind::GlobalEligible {
                off: 0,
                len: 2,
                token: CacheOpToken::LdgECa,
            },
            &sectors,
            None,
        )
        .unwrap();
        m.observe_coherence(
            Some(1),
            32,
            AccessKind::GlobalStore { off: 0, len: 1 },
            &sectors,
            None,
        )
        .unwrap();
        m.observe_coherence(
            Some(2),
            32,
            AccessKind::GlobalBypass {
                off: 0,
                len: 1,
                token: CacheOpToken::Atom,
                l2_requests: 1,
            },
            &sectors,
            None,
        )
        .unwrap();
        m.finish_coherence().unwrap();
        assert_eq!(m.coherence.decoded_global_lines, 1);
        assert_eq!(m.coherence.cross_sm_lines, 1);
        assert_eq!(m.coherence.cross_sm_sectors, 2);
        assert_eq!(m.coherence.cross_sm_sector_overlap, 1);
        assert_eq!(m.coherence.cross_sm_read_write_lines, 1);
        assert_eq!(m.coherence.cross_sm_write_write_lines, 1);
        assert_eq!(m.coherence.cross_sm_read_write_sectors, 1);
        assert_eq!(m.coherence.cross_sm_write_write_sectors, 1);
        assert!(m.coherence_lines.is_empty());
    }

    #[test]
    fn coherence_domain_retains_unknown_sm_records_for_fail_closed_consumers() {
        let mut m = TraceMetrics::default();
        m.observe_coherence(
            None,
            0xff,
            AccessKind::GlobalStore { off: 0, len: 1 },
            &[0],
            None,
        )
        .unwrap();
        assert_eq!(m.coherence.unknown_sm_or_line_records, 1);
        assert_eq!(m.coherence.cross_sm_sector_overlap, 0);
    }

    #[test]
    fn texture_access_is_not_admitted_to_global_coherence_domain() {
        let mut m = TraceMetrics::default();
        m.observe_coherence(
            Some(0),
            32,
            AccessKind::GlobalBypass {
                off: 0,
                len: 1,
                token: CacheOpToken::Tex,
                l2_requests: 1,
            },
            &[0],
            None,
        )
        .unwrap();
        assert_eq!(m.coherence.decoded_global_lines, 0);
        assert_eq!(m.coherence.unknown_sm_or_line_records, 1);
    }
}
