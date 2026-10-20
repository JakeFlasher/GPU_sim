//! Issue, readiness, and pipeline diagnostics; report-only counters.

use super::*;

impl ClockIssueDiagnostics {
    pub(super) const fn is_empty(&self) -> bool {
        self.reads_issued == 0
    }

    pub(super) fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        let next = Self {
            reads_issued: checked_readiness_add(
                "readiness.clock_issue.reads_issued",
                self.reads_issued,
                addend.reads_issued,
            )?,
            warps_with_interval: checked_readiness_add(
                "readiness.clock_issue.warps_with_interval",
                self.warps_with_interval,
                addend.warps_with_interval,
            )?,
            maximum_first_to_last_issue_cycles: self
                .maximum_first_to_last_issue_cycles
                .max(addend.maximum_first_to_last_issue_cycles),
            warps_with_three_reads: checked_readiness_add(
                "readiness.clock_issue.warps_with_three_reads",
                self.warps_with_three_reads,
                addend.warps_with_three_reads,
            )?,
            maximum_first_to_second_issue_cycles: self
                .maximum_first_to_second_issue_cycles
                .max(addend.maximum_first_to_second_issue_cycles),
            maximum_second_to_third_issue_cycles: self
                .maximum_second_to_third_issue_cycles
                .max(addend.maximum_second_to_third_issue_cycles),
        };
        *self = next;
        Ok(())
    }
}

impl ResultWritebackDiagnostics {
    pub const fn is_empty(&self) -> bool {
        self.fixed_over_l1tex_collisions == 0 && self.l1tex_deferral_cycles == 0
    }

    pub(super) fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        let next = Self {
            fixed_over_l1tex_collisions: checked_readiness_add(
                "readiness.result_writeback.fixed_over_l1tex_collisions",
                self.fixed_over_l1tex_collisions,
                addend.fixed_over_l1tex_collisions,
            )?,
            l1tex_deferral_cycles: checked_readiness_add(
                "readiness.result_writeback.l1tex_deferral_cycles",
                self.l1tex_deferral_cycles,
                addend.l1tex_deferral_cycles,
            )?,
        };
        *self = next;
        Ok(())
    }
}

/// Executed instruction-front-end observations for one SM. Demand counters
/// describe dynamic instruction fetches; L1 counters describe line requests
/// after private-L0/stream-buffer filtering. Prefetch counters are kept
/// separate so demand accuracy cannot be inferred from a collapsed total.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct InstructionFrontEndDiagnostics {
    pub instructions_fetched: u64,
    pub demand_l0_hits: u64,
    pub demand_stream_buffer_hits: u64,
    pub demand_l0_misses: u64,
    pub demand_l1_hits: u64,
    pub demand_l1_misses: u64,
    pub prefetch_requests: u64,
    pub prefetch_l1_hits: u64,
    pub prefetch_l1_misses: u64,
    pub stream_restarts: u64,
    pub maximum_fetch_latency_cycles: u64,
    /// Fetches that continued the warp issued in the prior cycle, independent
    /// of whether issue hold was cleared by Yield.
    pub fetch_last_issued_continuations: u64,
    pub yield_hold_separations: u64,
    /// Dynamic non-fall-through control transfers observed by the front end.
    /// This is an observer only; the rejected H26 flush arm is not active.
    pub control_flow_redirects: u64,
}

impl InstructionFrontEndDiagnostics {
    pub const fn is_empty(&self) -> bool {
        self.instructions_fetched == 0
    }

    pub(super) fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        let mut next = *self;
        macro_rules! add {
            ($field:ident) => {
                next.$field = checked_readiness_add(
                    concat!("readiness.instruction_front_end.", stringify!($field)),
                    next.$field,
                    addend.$field,
                )?;
            };
        }
        add!(instructions_fetched);
        add!(demand_l0_hits);
        add!(demand_stream_buffer_hits);
        add!(demand_l0_misses);
        add!(demand_l1_hits);
        add!(demand_l1_misses);
        add!(prefetch_requests);
        add!(prefetch_l1_hits);
        add!(prefetch_l1_misses);
        add!(stream_restarts);
        add!(fetch_last_issued_continuations);
        add!(yield_hold_separations);
        add!(control_flow_redirects);
        next.maximum_fetch_latency_cycles = next
            .maximum_fetch_latency_cycles
            .max(addend.maximum_fetch_latency_cycles);
        *self = next;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct MemoryIssuePipelineDiagnostics {
    pub instructions_admitted: u64,
    /// Dynamic instructions classified into the source-qualified constant
    /// pressure path (`11 + 39` SM86 operand frontier).
    pub constant_pressure_instructions: u64,
    pub operand_pipeline_delay_cycles: u64,
    pub shared_queue_delay_cycles: u64,
    pub maximum_operand_pipeline_delay_cycles: u64,
    pub maximum_shared_queue_delay_cycles: u64,
    pub maximum_queue_occupancy: u64,
    pub terminal_shared_accept_cycle: u64,
}

impl MemoryIssuePipelineDiagnostics {
    pub const fn is_empty(&self) -> bool {
        self.instructions_admitted == 0
    }

    pub(super) fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        let mut next = *self;
        for (field, target, value) in [
            (
                "readiness.memory_issue_pipeline.instructions_admitted",
                &mut next.instructions_admitted,
                addend.instructions_admitted,
            ),
            (
                "readiness.memory_issue_pipeline.constant_pressure_instructions",
                &mut next.constant_pressure_instructions,
                addend.constant_pressure_instructions,
            ),
            (
                "readiness.memory_issue_pipeline.operand_pipeline_delay_cycles",
                &mut next.operand_pipeline_delay_cycles,
                addend.operand_pipeline_delay_cycles,
            ),
            (
                "readiness.memory_issue_pipeline.shared_queue_delay_cycles",
                &mut next.shared_queue_delay_cycles,
                addend.shared_queue_delay_cycles,
            ),
        ] {
            *target = checked_readiness_add(field, *target, value)?;
        }
        next.maximum_operand_pipeline_delay_cycles = next
            .maximum_operand_pipeline_delay_cycles
            .max(addend.maximum_operand_pipeline_delay_cycles);
        next.maximum_shared_queue_delay_cycles = next
            .maximum_shared_queue_delay_cycles
            .max(addend.maximum_shared_queue_delay_cycles);
        next.maximum_queue_occupancy = next
            .maximum_queue_occupancy
            .max(addend.maximum_queue_occupancy);
        next.terminal_shared_accept_cycle = next
            .terminal_shared_accept_cycle
            .max(addend.terminal_shared_accept_cycle);
        *self = next;
        Ok(())
    }
}

/// Causal accounting for the SM86 Immediate Constant Cache (IMC) path. These
/// counters describe simulator line probes and replay stalls; they are
/// observation-only and are never used as a fitted timing correction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ImmediateConstantCacheDiagnostics {
    pub references: u64,
    pub lookup_hits: u64,
    pub lookup_misses: u64,
    /// Lookup misses that found an already outstanding line reservation.
    /// This is the simulator's pending-merge class, not NCU's tag-hit field.
    pub pending_tag_hits: u64,
    /// Lookup misses that allocated a new line request.
    pub tag_misses: u64,
    pub pending_merges: u64,
    pub lines_filled: u64,
    /// Number of selected instructions parked behind an IMC response. This
    /// is an event count, not NCU's overlapping warp-cycle stall metric.
    pub miss_wait_events: u64,
    /// Scheduler issue opportunities consumed by those waits. This is an
    /// SM-side causal count and is not directly comparable to NCU stalls.
    pub miss_wait_cycles: u64,
    /// Cycles for which the source-backed SM86 greedy-warp L0C window
    /// deliberately blocked youngest-warp switching. This is a mechanism
    /// diagnostic, not an elapsed-cycle prediction.
    pub greedy_switch_block_cycles: u64,
    /// Non-greedy candidate probes attempted after a prior candidate missed
    /// in IMC. These counters describe the bounded scan, not hardware warp
    /// sampling.
    pub non_greedy_scan_attempts: u64,
    pub non_greedy_scan_misses: u64,
    pub non_greedy_scan_successes: u64,
    pub maximum_pending_lines: u64,
}

/// Conditional SM86 register-file read envelope. These counters describe the
/// bounded effective model, not a claim about proprietary physical wiring.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct Sm86RegisterReadDiagnostics {
    pub reservations: u64,
    pub source_reads: [u64; 2],
    /// Readiness delay that was actually charged before issue. This is
    /// distinct from a residual same-cycle reservation conflict at dispatch.
    pub applied_wait_cycles: u64,
    pub port_conflict_cycles: u64,
    pub rfc_hits: u64,
    pub rfc_misses: u64,
    pub unsupported_shapes: u64,
}

impl Sm86RegisterReadDiagnostics {
    pub(super) fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        let mut next = *self;
        next.reservations = checked_readiness_add(
            "readiness.sm86_register_read.reservations",
            next.reservations,
            addend.reservations,
        )?;
        for bank in 0..2 {
            next.source_reads[bank] = checked_readiness_add(
                "readiness.sm86_register_read.source_reads",
                next.source_reads[bank],
                addend.source_reads[bank],
            )?;
        }
        next.port_conflict_cycles = checked_readiness_add(
            "readiness.sm86_register_read.port_conflict_cycles",
            next.port_conflict_cycles,
            addend.port_conflict_cycles,
        )?;
        next.applied_wait_cycles = checked_readiness_add(
            "readiness.sm86_register_read.applied_wait_cycles",
            next.applied_wait_cycles,
            addend.applied_wait_cycles,
        )?;
        next.rfc_hits = checked_readiness_add(
            "readiness.sm86_register_read.rfc_hits",
            next.rfc_hits,
            addend.rfc_hits,
        )?;
        next.rfc_misses = checked_readiness_add(
            "readiness.sm86_register_read.rfc_misses",
            next.rfc_misses,
            addend.rfc_misses,
        )?;
        next.unsupported_shapes = checked_readiness_add(
            "readiness.sm86_register_read.unsupported_shapes",
            next.unsupported_shapes,
            addend.unsupported_shapes,
        )?;
        *self = next;
        Ok(())
    }
}

impl ImmediateConstantCacheDiagnostics {
    pub const fn is_empty(&self) -> bool {
        self.references == 0
    }

    pub(super) fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        let mut next = *self;
        macro_rules! add {
            ($field:ident) => {
                next.$field = checked_readiness_add(
                    concat!("readiness.immediate_constant_cache.", stringify!($field)),
                    next.$field,
                    addend.$field,
                )?;
            };
        }
        add!(references);
        add!(lookup_hits);
        add!(lookup_misses);
        add!(pending_tag_hits);
        add!(tag_misses);
        add!(pending_merges);
        add!(lines_filled);
        add!(miss_wait_events);
        add!(miss_wait_cycles);
        add!(greedy_switch_block_cycles);
        add!(non_greedy_scan_attempts);
        add!(non_greedy_scan_misses);
        add!(non_greedy_scan_successes);
        next.maximum_pending_lines = next.maximum_pending_lines.max(addend.maximum_pending_lines);
        *self = next;
        Ok(())
    }
}

/// Deterministic causal readiness summary for one SM replay.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ReadinessDiagnostics {
    pub limiting_cycles: ReadinessCauseCycles,
    pub constraint_exposure_cycles: ReadinessCauseCycles,
    pub delayed_ready_issues: u64,
    pub tied_limiting_causes: u64,
    /// Count of within-CTA resident-warp priority inversions between the
    /// admission/arrival order and the trace's dynamic warp-id order. This is
    /// observational; it never selects timing or fits NCU.
    pub youngest_priority_order_mismatches: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instruction_front_end: Option<Box<InstructionFrontEndDiagnostics>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_issue_pipeline: Option<Box<MemoryIssuePipelineDiagnostics>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub immediate_constant_cache: Option<Box<ImmediateConstantCacheDiagnostics>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sm86_register_read: Option<Box<Sm86RegisterReadDiagnostics>>,
    pub async_groups: AsyncReadinessDiagnostics,
    /// Read-only scalar async decision records. These are emitted only when a
    /// modeled async-group wait is selected; they never participate in issue
    /// selection or completion timing.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub async_decision_snapshots: Vec<AsyncDecisionSnapshot>,
    /// Number of dynamic decision snapshots intentionally omitted after the
    /// bounded diagnostic sample was filled. Dropping diagnostics must never
    /// reject an otherwise valid real-trace replay.
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub async_decision_snapshots_dropped: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scalar_execution_pipeline: Option<Box<ScalarExecutionPipelineDiagnostics>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub async_instrumentation_order: Option<Box<AsyncInstrumentationOrderDiagnostics>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub async_issue_overlap: Option<Box<AsyncIssueOverlapDiagnostics>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub async_wait_prefix: Option<Box<AsyncWaitPrefixDiagnostics>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bypass_async_copies: Option<Box<BypassAsyncCopyDiagnostics>>,
    pub cta_barriers: CtaBarrierDiagnostics,
    #[serde(skip_serializing_if = "BarrierPipelineDiagnostics::is_empty")]
    pub barrier_pipeline: BarrierPipelineDiagnostics,
    #[serde(skip_serializing_if = "ControlPipelineDiagnostics::is_empty")]
    pub control_pipeline: ControlPipelineDiagnostics,
    #[serde(skip_serializing_if = "ResultWritebackDiagnostics::is_empty")]
    pub result_writeback: ResultWritebackDiagnostics,
    #[serde(skip_serializing_if = "ClockIssueDiagnostics::is_empty")]
    pub clock_issue: ClockIssueDiagnostics,
    #[serde(skip_serializing_if = "GlobalMemoryIssueDiagnostics::is_empty")]
    pub global_memory_issue: GlobalMemoryIssueDiagnostics,
    #[serde(skip_serializing_if = "MemoryServiceCandidateDiagnostics::is_empty")]
    pub memory_service_candidates: MemoryServiceCandidateDiagnostics,
    #[serde(skip_serializing_if = "MemoryAdmissionCandidateDiagnostics::is_empty")]
    pub memory_admission_candidates: MemoryAdmissionCandidateDiagnostics,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_admission_feedback: Option<Box<MemoryAdmissionFeedbackDiagnostics>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_write_ack: Option<Box<MemoryWriteAckDiagnostics>>,
}

impl ReadinessDiagnostics {
    /// Checked, transactional aggregation across SMs.
    pub fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        let mut next = self.clone();
        next.limiting_cycles.add_checked(&addend.limiting_cycles)?;
        next.constraint_exposure_cycles
            .add_checked(&addend.constraint_exposure_cycles)?;
        next.delayed_ready_issues = checked_readiness_add(
            "readiness.delayed_ready_issues",
            next.delayed_ready_issues,
            addend.delayed_ready_issues,
        )?;
        next.tied_limiting_causes = checked_readiness_add(
            "readiness.tied_limiting_causes",
            next.tied_limiting_causes,
            addend.tied_limiting_causes,
        )?;
        next.youngest_priority_order_mismatches = checked_readiness_add(
            "readiness.youngest_priority_order_mismatches",
            next.youngest_priority_order_mismatches,
            addend.youngest_priority_order_mismatches,
        )?;
        if let Some(addend_front_end) = addend.instruction_front_end.as_deref() {
            if let Some(current) = next.instruction_front_end.as_deref_mut() {
                current.add_checked(addend_front_end)?;
            } else {
                next.instruction_front_end = Some(Box::new(*addend_front_end));
            }
        }
        if let Some(addend_memory_pipeline) = addend.memory_issue_pipeline.as_deref() {
            if let Some(current) = next.memory_issue_pipeline.as_deref_mut() {
                current.add_checked(addend_memory_pipeline)?;
            } else {
                next.memory_issue_pipeline = Some(Box::new(*addend_memory_pipeline));
            }
        }
        if let Some(addend_immediate_constant) = addend.immediate_constant_cache.as_deref() {
            if let Some(current) = next.immediate_constant_cache.as_deref_mut() {
                current.add_checked(addend_immediate_constant)?;
            } else {
                next.immediate_constant_cache = Some(Box::new(*addend_immediate_constant));
            }
        }
        if let Some(addend_register_read) = addend.sm86_register_read.as_deref() {
            if let Some(current) = next.sm86_register_read.as_deref_mut() {
                current.add_checked(addend_register_read)?;
            } else {
                next.sm86_register_read = Some(Box::new(*addend_register_read));
            }
        }
        next.async_groups.add_checked(&addend.async_groups)?;
        // Snapshots are optional diagnostics.  Keep the serialized report
        // bounded even for very large multi-SM launches; the dropped count
        // preserves the fact that sampling was truncated.
        next.async_decision_snapshots_dropped = checked_readiness_add(
            "readiness.async_decision_snapshots_dropped",
            next.async_decision_snapshots_dropped,
            addend.async_decision_snapshots_dropped,
        )?;
        let available =
            MAX_ASYNC_DECISION_SNAPSHOTS.saturating_sub(next.async_decision_snapshots.len());
        let kept = available.min(addend.async_decision_snapshots.len());
        next.async_decision_snapshots
            .extend_from_slice(&addend.async_decision_snapshots[..kept]);
        let omitted = addend.async_decision_snapshots.len().saturating_sub(kept);
        next.async_decision_snapshots_dropped = checked_readiness_add(
            "readiness.async_decision_snapshots_dropped",
            next.async_decision_snapshots_dropped,
            u64::try_from(omitted).unwrap_or(u64::MAX),
        )?;
        if let Some(addend_scalar) = addend.scalar_execution_pipeline.as_deref() {
            if let Some(current) = next.scalar_execution_pipeline.as_deref_mut() {
                current.add_checked(addend_scalar)?;
            } else {
                next.scalar_execution_pipeline = Some(Box::new(*addend_scalar));
            }
        }
        if let Some(addend_order) = addend.async_instrumentation_order.as_deref() {
            if let Some(current) = next.async_instrumentation_order.as_deref_mut() {
                current.add_checked(addend_order)?;
            } else {
                next.async_instrumentation_order = Some(Box::new(*addend_order));
            }
        }
        if let Some(addend_overlap) = addend.async_issue_overlap.as_deref() {
            if let Some(current) = next.async_issue_overlap.as_deref_mut() {
                current.add_checked(addend_overlap)?;
            } else {
                next.async_issue_overlap = Some(Box::new(*addend_overlap));
            }
        }
        if let Some(addend_waits) = addend.async_wait_prefix.as_deref() {
            if let Some(current) = next.async_wait_prefix.as_deref_mut() {
                current.add_checked(addend_waits)?;
            } else {
                next.async_wait_prefix = Some(Box::new(*addend_waits));
            }
        }
        if let Some(addend_bypass) = addend.bypass_async_copies.as_deref() {
            if let Some(current) = next.bypass_async_copies.as_deref_mut() {
                current.add_checked(addend_bypass)?;
            } else {
                next.bypass_async_copies = Some(Box::new(*addend_bypass));
            }
        }
        next.cta_barriers.add_checked(&addend.cta_barriers)?;
        next.barrier_pipeline
            .add_checked(&addend.barrier_pipeline)?;
        next.control_pipeline
            .add_checked(&addend.control_pipeline)?;
        next.result_writeback
            .add_checked(&addend.result_writeback)?;
        next.clock_issue.add_checked(&addend.clock_issue)?;
        next.global_memory_issue
            .add_checked(&addend.global_memory_issue)?;
        next.memory_service_candidates
            .add_checked(&addend.memory_service_candidates)?;
        next.memory_admission_candidates
            .add_checked(&addend.memory_admission_candidates)?;
        if let Some(addend_feedback) = addend.memory_admission_feedback.as_deref() {
            if let Some(current) = next.memory_admission_feedback.as_deref_mut() {
                current.add_checked(addend_feedback)?;
            } else {
                next.memory_admission_feedback = Some(Box::new(*addend_feedback));
            }
        }
        if let Some(addend_ack) = addend.memory_write_ack.as_deref() {
            if let Some(current) = next.memory_write_ack.as_deref_mut() {
                current.add_checked(addend_ack)?;
            } else {
                next.memory_write_ack = Some(Box::new(*addend_ack));
            }
        }
        *self = next;
        Ok(())
    }

    /// Schema 1 remains byte-compatible without characterized resources;
    /// schema 2 adds the barrier service, schema 3 adds the exact branch
    /// resolution relation, and schema 4 adds an observed fixed/L1TEX
    /// writeback collision. Schema 5 adds an exact CLOCK-boundary issue
    /// diagnostic when the joined program contains measured CLOCKLO reads.
    /// Schema 6 adds scalar-global instruction/sector/ideal-wavefront,
    /// unique-line, exact-atomic-request, and issue-span diagnostics without
    /// activating a memory service. Schema 7 adds results from an explicitly
    /// configured, bounded streaming memory-service candidate set; candidates
    /// remain observational and cannot change issue scheduling. Schema 8
    /// separates scalar-global operand-consumption/read-barrier producers and
    /// their consumer delay from the generic fixed read-counter relation.
    /// Schema 9 adds bounded observational finite admission queues driven by
    /// exact per-instruction L2 hit/miss outcomes. Schema 10 activates one
    /// opt-in feedback-coupled queue that participates in warp readiness.
    /// Schema 11 additionally makes `MEMBAR.SC.GPU` wait on the issuing warp's
    /// matching write-service completion frontier. Schema 12 adds a
    /// non-occupying operation-specific visibility/RMW latency after service.
    /// Schema 13 adds independent request/region completion work over bounded
    /// parallel lanes. Schema 14 adds bypass-only per-copy source/delivery
    /// observations without changing scheduler timing. Schema 15 separates
    /// source-cohort destination-line fanout, bank multiplicity, and a
    /// timing-neutral per-copy topology candidate. Schema 16 adds
    /// timing-neutral pipelined destination candidates over stable per-copy
    /// payload-arrival order for acceptance widths 1, 2, 4, 8, and 16.
    /// Schema 17 joins exact source-sector responses to flexible destination
    /// cohorts and reports a parameter-free source/topology-overlap candidate.
    /// Schema 18 adds a first-response-anchored, timing-neutral destination
    /// reservation-exposure score over those exact cohorts. Schema 19 adds
    /// timing-neutral partial-versus-draining async-wait prefix observations
    /// for programs containing at least one nonzero wait immediate. Schema 20
    /// adds timing-neutral cross-SMSP coissue and sibling-warp committed-group
    /// population observations. Schema 21 adds a timing-free reconstruction of
    /// selected async issue, commit, and wait events in callback/channel order.
    /// Schema 22 makes the existing opt-in STORE/RED write-completion frontier
    /// a drained shared-system event and reports its exact lifecycle. Schema 23
    /// adds the active private-L0/shared-L1 instruction hierarchy, stream-buffer
    /// prefetcher, and three-entry per-warp instruction buffers. Schema 24 adds
    /// the active per-SMSP memory queue/address pipeline and SM-shared acceptor.
    /// Schema 25 adds causal per-SMSP Immediate Constant Cache probe and
    /// replay accounting; these counters remain observation-only.
    /// Schema 26 separates the uniform/SFU readiness counter from the tensor
    /// pipeline counter.
    /// Schema 27 adds the bounded SM86 greedy L0C miss-switch blocking
    /// diagnostic to the immediate-constant accounting. Schema 28 adds the
    /// bounded non-greedy candidate-scan counters.
    /// Schema 29 adds the explicit within-CTA dynamic-youngest priority audit.
    /// Schema 30 adds a dynamic control-flow redirect observer; it does not
    /// activate the rejected branch-stream flush timing arm.
    /// Schema 31 adds the source-qualified constant-pressure memory-pipeline
    /// observer.
    pub fn schema_version(&self) -> u32 {
        // The priority-audit field is serialized for every replay mode, even
        // when no IMC surface is present; retain one unambiguous current
        // schema rather than pretending the additive field is absent.
        return 31;
        #[allow(unreachable_code)]
        let feedback = self.memory_admission_feedback.as_deref();
        if self.immediate_constant_cache.is_some()
            && self
                .immediate_constant_cache
                .as_deref()
                .is_some_and(|cache| {
                    cache.non_greedy_scan_attempts != 0
                        || cache.non_greedy_scan_misses != 0
                        || cache.non_greedy_scan_successes != 0
                })
        {
            28
        } else if self.immediate_constant_cache.is_some() {
            27
        } else if self.limiting_cycles.pipeline_sfu != 0
            || self.constraint_exposure_cycles.pipeline_sfu != 0
        {
            26
        } else if self.immediate_constant_cache.is_some() {
            25
        } else if self.memory_issue_pipeline.is_some() {
            24
        } else if self.instruction_front_end.is_some() {
            23
        } else if self.memory_write_ack.is_some() {
            22
        } else if self.async_instrumentation_order.is_some() {
            21
        } else if self.async_issue_overlap.is_some() {
            20
        } else if self.async_wait_prefix.is_some() {
            19
        } else if self
            .bypass_async_copies
            .as_deref()
            .is_some_and(|diagnostics| diagnostics.cohort_overlap_groups_observed != 0)
        {
            18
        } else if self
            .bypass_async_copies
            .as_deref()
            .is_some_and(|diagnostics| diagnostics.cohort_groups_observed != 0)
        {
            17
        } else if self
            .bypass_async_copies
            .as_deref()
            .is_some_and(|diagnostics| diagnostics.pipeline_groups_observed != 0)
        {
            16
        } else if self
            .bypass_async_copies
            .as_deref()
            .is_some_and(|diagnostics| diagnostics.source_cohorts_observed != 0)
        {
            15
        } else if self.bypass_async_copies.is_some() {
            14
        } else if feedback.is_some_and(|diagnostics| {
            diagnostics.candidate.write_completion_request_cycles != 0
                || diagnostics.candidate.write_completion_region_cycles != 0
        }) {
            13
        } else if feedback
            .is_some_and(|diagnostics| diagnostics.candidate.write_completion_latency_cycles != 0)
        {
            12
        } else if feedback
            .is_some_and(|diagnostics| diagnostics.candidate.membar_waits_for_service_completion)
        {
            11
        } else if feedback.is_some() {
            10
        } else if !self.memory_admission_candidates.is_empty() {
            9
        } else if self.global_memory_issue.has_operand_read_barriers() {
            8
        } else if !self.memory_service_candidates.is_empty() {
            7
        } else if !self.global_memory_issue.is_empty() {
            6
        } else if !self.clock_issue.is_empty() {
            5
        } else if !self.result_writeback.is_empty() {
            4
        } else if !self.control_pipeline.is_empty() {
            3
        } else if !self.barrier_pipeline.is_empty() {
            2
        } else {
            1
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ScalarExecutionPipelineDiagnostics {
    pub instructions_admitted: u64,
    pub service_busy_cycles: u64,
    pub queue_delay_cycles: u64,
    pub maximum_queue_occupancy: u64,
}

impl ScalarExecutionPipelineDiagnostics {
    pub(super) fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        let mut next = *self;
        for (field, target, value) in [
            (
                "readiness.scalar_execution_pipeline.instructions_admitted",
                &mut next.instructions_admitted,
                addend.instructions_admitted,
            ),
            (
                "readiness.scalar_execution_pipeline.service_busy_cycles",
                &mut next.service_busy_cycles,
                addend.service_busy_cycles,
            ),
            (
                "readiness.scalar_execution_pipeline.queue_delay_cycles",
                &mut next.queue_delay_cycles,
                addend.queue_delay_cycles,
            ),
        ] {
            *target = checked_readiness_add(field, *target, value)?;
        }
        next.maximum_queue_occupancy = next
            .maximum_queue_occupancy
            .max(addend.maximum_queue_occupancy);
        *self = next;
        Ok(())
    }
}

#[cfg(feature = "reference")]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct AdvanceDiagnostics {
    pub calls: u64,
    pub local_cycle_steps: u64,
    pub idle_jumps: u64,
    pub skipped_cycles: u64,
}
