// Lexical implementation section 2/4; included by the parent facade.
/// Build a conservative core-only diagnostic report from a real compact trace. The
/// producer bridge intentionally refuses memory, barriers, dynamic pipelines,
/// unknown register dependencies, and incomplete/slot-reused residency;
/// callers can use the summary to decide which ranges are safe to summarize
/// while keeping the ordinary replay as fallback.
pub fn fast_frontier_for_trace(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedule: &FunctionSchedule,
    arch_card: &ArchCard,
) -> Result<FastFrontierSummary, ReplayError> {
    let placement = PlacementConfig::from_arch_card(arch_card, schedule.arch)?;
    require_trace_architecture(td, schedule.arch)?;
    let launches = td.kernels().count();
    if launches != 1 {
        return Err(ReplayError::FastFrontierLaunchScopeUnknown { launches });
    }
    let extracted = extract_per_sm_blocks(td, loc, schedule, placement, false, None)?;
    if !matches!(&extracted.coverage, ExtractedCoverage::Full(_)) {
        return Err(ReplayError::FastFrontierRequiresFullCoverage);
    }
    let trace_records = extracted
        .per_sm_blocks
        .values()
        .flat_map(|blocks| blocks.iter())
        .flat_map(|block| block.warps.iter())
        .try_fold(0_u64, |count, warp| {
            count
                .checked_add(u64::try_from(warp.insts.len()).map_err(|_| {
                    ReplayError::FastFrontier {
                        detail: "trace record count exceeds u64".to_owned(),
                    }
                })?)
                .ok_or_else(|| ReplayError::FastFrontier {
                    detail: "trace record count overflows u64".to_owned(),
                })
        })?;
    let active_missing = extracted
        .trace_metrics
        .instructions
        .active_mask_missing_instructions;
    let predicate_missing = extracted
        .trace_metrics
        .instructions
        .predicate_mask_missing_instructions;
    let active_nonfull = extracted
        .trace_metrics
        .instructions
        .active_mask_nonfull_instructions;
    let predicate_nonfull = extracted
        .trace_metrics
        .instructions
        .predicate_mask_nonfull_instructions;
    if active_missing != 0 || predicate_missing != 0 {
        return Err(ReplayError::FastFrontierMasksIncomplete {
            active_missing,
            predicate_missing,
        });
    }
    let mut summary = FastFrontierSummary {
        // Keep the v1 wire token for compatibility. Its semantics are now
        // documented as descriptor-model self-equivalence; native refinement
        // will receive a distinct contract in a later schema revision.
        contract: "sm86_fast_exact_scalar_regions_v1",
        architecture: schedule.arch.token(),
        blocks_seen: 0,
        warps_seen: 0,
        eligible_warps: 0,
        candidate_regions: 0,
        certified_regions: 0,
        reified_regions: 0,
        candidate_instructions: 0,
        collapsed_instructions: 0,
        exact_execution_steps: 0,
        summarized_execution_steps: 0,
        macro_dispatches: 0,
        maximum_exact_final_cycle: 0,
        maximum_summarized_final_cycle: 0,
        front_end_static_lines: 0,
        front_end_penalty_cycles: 0,
        terminal_state_matches: true,
        coverage: FastFrontierCoverage {
            trace_records,
            uncovered_records: trace_records,
            ..FastFrontierCoverage::default()
        },
        refusal_reasons: BTreeMap::new(),
    };
    let add = |counter: &'static str, value: &mut u64, addend: u64| {
        *value = value
            .checked_add(addend)
            .ok_or_else(|| ReplayError::FastFrontier {
                detail: format!("{counter} overflows u64"),
            })?;
        Ok::<(), ReplayError>(())
    };
    if active_nonfull != 0 || predicate_nonfull != 0 {
        // The compact descriptor does not retain per-record predicate
        // populations. Preserve the fast-only executable contract, but refuse
        // the whole launch rather than treating predicated scalar work as an
        // all-lane empty-boundary recurrence.
        for blocks in extracted.per_sm_blocks.values() {
            add(
                "blocks_seen",
                &mut summary.blocks_seen,
                u64::try_from(blocks.len()).map_err(|_| ReplayError::FastFrontier {
                    detail: "block count exceeds u64".to_owned(),
                })?,
            )?;
            for block in blocks {
                add(
                    "warps_seen",
                    &mut summary.warps_seen,
                    u64::try_from(block.warps.len()).map_err(|_| ReplayError::FastFrontier {
                        detail: "warp count exceeds u64".to_owned(),
                    })?,
                )?;
                for _ in &block.warps {
                    if active_nonfull != 0 {
                        increment_fast_reason(
                            &mut summary.refusal_reasons,
                            "nonfull_active_population",
                        )?;
                    }
                    if predicate_nonfull != 0 {
                        increment_fast_reason(
                            &mut summary.refusal_reasons,
                            "nonfull_predicate_population",
                        )?;
                    }
                }
            }
        }
        return Ok(summary);
    }
    for blocks in extracted.per_sm_blocks.values() {
        // The causal analyzer's `Complete` premise is the full physical
        // roster for the interval, not the current CTA's local warp list.
        // Gather every traced block on this SM and refuse slot reuse rather
        // than accidentally certifying a segment with an omitted competitor.
        let residents = blocks
            .iter()
            .flat_map(|block| block.warps.iter().map(|warp| warp.warp_slot))
            .collect::<Vec<_>>();
        let roster_has_reuse = {
            let mut sorted = residents.clone();
            sorted.sort_unstable();
            sorted.windows(2).any(|pair| pair[0] == pair[1])
        };
        // The current compact trace has no CTA start/end epochs or incoming
        // pipeline snapshot.  Only one traced warp on one block can therefore
        // satisfy the analyzer's empty-boundary premise; all other rosters
        // remain explicit refusals instead of fresh-state certificates.
        let single_empty_boundary = blocks.len() == 1 && residents.len() == 1;
        for block in blocks {
            add("blocks_seen", &mut summary.blocks_seen, 1)?;
            if roster_has_reuse || !single_empty_boundary {
                let reason = if roster_has_reuse {
                    "resident_roster_slot_reuse"
                } else {
                    "incoming_pipeline_or_residency_unknown"
                };
                for _warp in &block.warps {
                    add("warps_seen", &mut summary.warps_seen, 1)?;
                    increment_fast_reason(&mut summary.refusal_reasons, reason)?;
                }
                continue;
            }
            let mut candidate_segments = Vec::new();
            for warp in &block.warps {
                add("warps_seen", &mut summary.warps_seen, 1)?;
                let warp_segments = fast_scalar_segments(warp, schedule);
                if warp_segments.is_empty() {
                    increment_fast_reason(
                        &mut summary.refusal_reasons,
                        fast_scalar_refusal_reason(warp, schedule),
                    )?;
                    continue;
                }
                add("eligible_warps", &mut summary.eligible_warps, 1)?;
                for segment in warp_segments {
                    // A single segment is analyzed at a time so disjoint
                    // segments on the same warp never masquerade as one
                    // contiguous certificate. The complete resident roster
                    // still prevents certification when another resident warp
                    // can compete on that SMSP.
                    candidate_segments.push(segment);
                }
            }
            for segment in candidate_segments {
                let segment_len =
                    u64::try_from(segment.len()).map_err(|_| ReplayError::FastFrontier {
                        detail: "scalar segment length exceeds u64".to_owned(),
                    })?;
                summary.coverage.candidate_records = summary
                    .coverage
                    .candidate_records
                    .checked_add(segment_len)
                    .ok_or_else(|| ReplayError::FastFrontier {
                        detail: "candidate coverage count overflows u64".to_owned(),
                    })?;
                // A suffix discovered after an unmodelled prefix has no
                // captured entry pipeline/scoreboard state. Retain it in the
                // coverage denominator, but force the exact/model-local
                // refusal path until a typed entry boundary is supplied.
                let closed_entry = segment[0].sequence == 0;
                if !closed_entry {
                    increment_fast_reason(
                        &mut summary.refusal_reasons,
                        "missing_entry_boundary_state",
                    )?;
                }
                let residency = if closed_entry {
                    // The compact trace is not an independent dependency or
                    // effect witness. Residency alone therefore remains an
                    // exact/fallback premise; a separate static/trace witness
                    // is required before a macro can be installed.
                    CausalFrontierResidency::Complete {
                        physical_warps: residents.clone(),
                    }
                } else {
                    CausalFrontierResidency::Unknown
                };
                let report: CausalFrontierReport =
                    summarize_with_residency(std::slice::from_ref(&segment), residency).map_err(
                        |error| ReplayError::FastFrontier {
                            detail: error.to_string(),
                        },
                    )?;
                // The generic analyzer only increments candidate_regions after
                // it sees a fully summarizable interval. Keep a producer-side
                // candidate/refusal witness for scalar runs rejected by a
                // control/dependency/occupancy guard, so fast-only output does
                // not confuse "no candidate" with "no scalar work".
                if report.candidate_regions == 0 {
                    add("candidate_regions", &mut summary.candidate_regions, 1)?;
                    add("reified_regions", &mut summary.reified_regions, 1)?;
                    add(
                        "candidate_instructions",
                        &mut summary.candidate_instructions,
                        segment_len,
                    )?;
                    increment_fast_reason(&mut summary.refusal_reasons, "causal_guard_refused")?;
                }
                add(
                    "candidate_regions",
                    &mut summary.candidate_regions,
                    report.candidate_regions,
                )?;
                add(
                    "certified_regions",
                    &mut summary.certified_regions,
                    u64::try_from(report.certified_regions.len()).map_err(|_| {
                        ReplayError::FastFrontier {
                            detail: "certified region count exceeds u64".to_owned(),
                        }
                    })?,
                )?;
                add(
                    "reified_regions",
                    &mut summary.reified_regions,
                    report.reified_regions,
                )?;
                add(
                    "candidate_instructions",
                    &mut summary.candidate_instructions,
                    report.candidate_instructions,
                )?;
                add(
                    "collapsed_instructions",
                    &mut summary.collapsed_instructions,
                    report.collapsed_instructions,
                )?;
                summary.coverage.summarized_records = summary
                    .coverage
                    .summarized_records
                    .checked_add(report.collapsed_instructions)
                    .ok_or_else(|| ReplayError::FastFrontier {
                        detail: "summarized coverage count overflows u64".to_owned(),
                    })?;
                let reified_records = segment_len
                    .checked_sub(report.collapsed_instructions)
                    .ok_or_else(|| ReplayError::FastFrontier {
                        detail: "collapsed coverage exceeds candidate segment".to_owned(),
                    })?;
                summary.coverage.reified_records = summary
                    .coverage
                    .reified_records
                    .checked_add(reified_records)
                    .ok_or_else(|| ReplayError::FastFrontier {
                        detail: "reified coverage count overflows u64".to_owned(),
                    })?;
                add(
                    "exact_execution_steps",
                    &mut summary.exact_execution_steps,
                    report.exact_execution_steps,
                )?;
                add(
                    "summarized_execution_steps",
                    &mut summary.summarized_execution_steps,
                    report.causal_execution_steps,
                )?;
                add(
                    "macro_dispatches",
                    &mut summary.macro_dispatches,
                    report.macro_dispatches,
                )?;
                summary.maximum_exact_final_cycle = summary
                    .maximum_exact_final_cycle
                    .max(report.exact_final_cycle);
                summary.maximum_summarized_final_cycle = summary
                    .maximum_summarized_final_cycle
                    .max(report.causal_final_cycle);
                summary.terminal_state_matches &= report.terminal_state_matches;
                for detail in report.reified_region_details {
                    increment_fast_reason(
                        &mut summary.refusal_reasons,
                        &format!("reified_{:?}", detail.reason),
                    )?;
                }
            }
        }
    }
    summary.coverage.uncovered_records = summary
        .coverage
        .trace_records
        .checked_sub(summary.coverage.candidate_records)
        .ok_or_else(|| ReplayError::FastFrontier {
            detail: "candidate coverage exceeds trace record count".to_owned(),
        })?;
    Ok(summary)
}

/// Build the SM120 alpha stage/event prediction without walking the full
/// memory/event queue.  This complementary diagnostic consumes the compact
/// native trace, static schedule classes, dynamic per-warp dependencies,
/// modeled SMSP placement and CTA barriers. Ordinary memory dependencies use
/// the shared Flat-latency arithmetic; cache service and native async visibility
/// remain outside this path.
///
/// Unlike the older SM86 selective-scalar path this function accepts one
/// launch from a multi-launch TraceDir.  Each launch is attributed by its own
/// `KernelLoc`, so the caller can report a complete capture without treating
/// capture-wide order as a timing oracle.
pub fn sm120_fast_stage_for_trace(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedule: &FunctionSchedule,
    arch_card: &ArchCard,
    alpha_config: Sm120AlphaConfig,
) -> Result<FastFrontierSummary, ReplayError> {
    let placement = PlacementConfig::from_arch_card(arch_card, schedule.arch)?;
    require_trace_architecture(td, schedule.arch)?;
    let extracted = extract_per_sm_blocks(td, loc, schedule, placement, false, None)?;
    let params = IssueModelParams::from_card_alpha(ArchId::Sm120, arch_card, alpha_config)?;
    let mut masks = BTreeMap::new();
    for (launch_index, block) in td.validated_threadblocks_in_launch_order(loc)? {
        for warp in block.warps() {
            let Some(slot) = warp.instructions().find_map(|inst| inst.warpid_sm()) else {
                continue;
            };
            let lane_masks = warp
                .instructions()
                .map(|inst| {
                    if inst.predicate_mask_present() {
                        inst.predicate_on_mask()
                    } else if inst.active_mask_present() {
                        inst.active_mask()
                    } else {
                        u32::MAX
                    }
                })
                .collect::<Vec<_>>();
            masks.insert((launch_index, slot), lane_masks);
        }
    }
    let trace_records = extracted.records_seen;
    let mut summary = FastFrontierSummary {
        contract: "sm120_fast_stage_event_v1",
        architecture: schedule.arch.token(),
        blocks_seen: 0,
        warps_seen: 0,
        eligible_warps: 0,
        candidate_regions: 0,
        certified_regions: 0,
        reified_regions: 0,
        candidate_instructions: trace_records,
        collapsed_instructions: 0,
        exact_execution_steps: trace_records,
        summarized_execution_steps: 0,
        macro_dispatches: 0,
        maximum_exact_final_cycle: 0,
        maximum_summarized_final_cycle: 0,
        front_end_static_lines: 0,
        front_end_penalty_cycles: 0,
        // This summary is intentionally not an exact-state certificate.  A
        // false value prevents downstream users from reading it as the old
        // SM86 scalar equivalence contract.
        terminal_state_matches: false,
        coverage: FastFrontierCoverage {
            trace_records,
            candidate_records: trace_records,
            summarized_records: 0,
            reified_records: trace_records,
            uncovered_records: 0,
        },
        refusal_reasons: BTreeMap::new(),
    };

    let mut memory_records = 0_u64;
    let mut async_records = 0_u64;
    let mut barrier_records = 0_u64;
    let mut resource_class_counts: BTreeMap<String, u64> = BTreeMap::new();
    let add = |counter: &'static str, value: &mut u64, amount: u64| {
        *value = value
            .checked_add(amount)
            .ok_or_else(|| ReplayError::FastFrontier {
                detail: format!("{counter} overflows u64"),
            })?;
        Ok::<(), ReplayError>(())
    };

    for blocks in extracted.per_sm_blocks.values() {
        add(
            "blocks_seen",
            &mut summary.blocks_seen,
            u64::try_from(blocks.len()).map_err(|_| ReplayError::FastFrontier {
                detail: "block count exceeds u64".to_owned(),
            })?,
        )?;
        for block in blocks {
            let warp_count =
                u64::try_from(block.warps.len()).map_err(|_| ReplayError::FastFrontier {
                    detail: "warp count exceeds u64".to_owned(),
                })?;
            add("warps_seen", &mut summary.warps_seen, warp_count)?;
            add("eligible_warps", &mut summary.eligible_warps, warp_count)?;
            add(
                "candidate_regions",
                &mut summary.candidate_regions,
                warp_count,
            )?;
            for warp in &block.warps {
                for instruction in &warp.insts {
                    let class_name = schedule
                        .issue_slot(u64::from(instruction.pc))
                        .map(|slot| match slot.pipeline_class() {
                            PipelineClass::DispatchOnly => "dispatch",
                            PipelineClass::Branch { .. } => "branch",
                            PipelineClass::Barrier { .. } => "barrier",
                            PipelineClass::Tensor { .. } => "tensor",
                            PipelineClass::Sfu { .. } => "uniform_sfu",
                            PipelineClass::ScalarAlu { .. } => "scalar",
                        })
                        .unwrap_or("unknown");
                    let count = resource_class_counts
                        .entry(class_name.to_owned())
                        .or_default();
                    *count = count
                        .checked_add(1)
                        .ok_or_else(|| ReplayError::FastFrontier {
                            detail: "resource-class observation count overflows u64".to_owned(),
                        })?;
                }
            }
        }
        let (frontier, memory, async_count, barriers) = fast_dynamic_sm_frontier(
            blocks,
            schedule,
            &masks,
            &params,
            extracted.shared_memory_residency,
        )?;
        summary.maximum_exact_final_cycle = summary.maximum_exact_final_cycle.max(frontier);
        memory_records =
            memory_records
                .checked_add(memory)
                .ok_or_else(|| ReplayError::FastFrontier {
                    detail: "memory-service observation count overflows u64".to_owned(),
                })?;
        async_records =
            async_records
                .checked_add(async_count)
                .ok_or_else(|| ReplayError::FastFrontier {
                    detail: "async-completion observation count overflows u64".to_owned(),
                })?;
        barrier_records =
            barrier_records
                .checked_add(barriers)
                .ok_or_else(|| ReplayError::FastFrontier {
                    detail: "barrier-visibility observation count overflows u64".to_owned(),
                })?;
    }
    // Charge the footprint actually fetched by this launch, not all cold
    // branches present in the static function table.
    let line_cycles = extracted.front_end.compulsory_instruction_read_sectors;
    if alpha_config.front_end_line_cycles_denominator == 0 {
        return Err(ReplayError::FastFrontier {
            detail: "front-end line denominator is zero".to_owned(),
        });
    }
    let scaled_lines = line_cycles
        .checked_mul(alpha_config.front_end_line_cycles_numerator)
        .and_then(|value| value.checked_add(alpha_config.front_end_line_cycles_denominator - 1))
        .ok_or_else(|| ReplayError::FastFrontier {
            detail: "front-end line penalty overflows u64".to_owned(),
        })?
        / alpha_config.front_end_line_cycles_denominator;
    let front_end_penalty = alpha_config
        .front_end_base_cycles
        .checked_add(scaled_lines)
        .ok_or_else(|| ReplayError::FastFrontier {
            detail: "front-end penalty overflows u64".to_owned(),
        })?;
    summary.front_end_static_lines = line_cycles;
    summary.front_end_penalty_cycles = front_end_penalty;
    summary.maximum_summarized_final_cycle = summary
        .maximum_exact_final_cycle
        .checked_add(front_end_penalty)
        .ok_or_else(|| ReplayError::FastFrontier {
            detail: "front-end-adjusted SM120 stage cycles overflow u64".to_owned(),
        })?;
    let resource_class_total = resource_class_counts
        .values()
        .try_fold(0_u64, |total, count| {
            total
                .checked_add(*count)
                .ok_or_else(|| ReplayError::FastFrontier {
                    detail: "resource-class total overflows u64".to_owned(),
                })
        })?;
    summary
        .refusal_reasons
        .insert("stage_resource_classes".to_owned(), resource_class_total);
    summary
        .refusal_reasons
        .insert("memory_service_pending".to_owned(), memory_records);
    summary
        .refusal_reasons
        .insert("async_completion_pending".to_owned(), async_records);
    summary
        .refusal_reasons
        .insert("barrier_visibility_pending".to_owned(), barrier_records);
    Ok(summary)
}

impl FullKernelReplayResult {
    pub fn coverage(&self) -> &FullLaunchCoverage {
        &self.coverage
    }

    pub fn configuration(&self) -> ReplayConfiguration {
        self.configuration
    }

    pub fn co_residency(&self) -> CoResidency {
        self.configuration.co_residency()
    }

    pub fn arch(&self) -> ArchId {
        self.replay.arch
    }

    pub fn instructions_issued(&self) -> u64 {
        self.replay.instructions_issued
    }

    pub fn records_seen(&self) -> u64 {
        self.replay.records_seen
    }

    pub fn per_sm(&self) -> &BTreeMap<i32, SmSimResult> {
        &self.replay.per_sm
    }

    /// Maximum per-SM issue frontier (`last local issue timestamp + 1`). This
    /// is the unadjusted causal scheduler/event lower bound. Consumers that
    /// need the silicon `sm__cycles_active` observation domain must use the
    /// explicit [`Self::sm_active_observation_frontier`] projection.
    pub fn diagnostic_last_issue_cycles(&self) -> u64 {
        self.replay.diagnostic_last_issue_cycles
    }

    /// Unadjusted issue frontier used by causal replay diagnostics. The value
    /// is kept separate from the architecture-scoped observation projection;
    /// it is never a hardware target or a fitted correction.
    pub fn modeled_issue_lower_bound_cycles(&self) -> u64 {
        self.replay.diagnostic_last_issue_cycles
    }

    /// Raw timestamp of the final successfully processed shared-system event,
    /// before it is combined with the issue frontier.
    pub fn last_processed_modeled_system_event_time(&self) -> Option<u64> {
        self.replay.last_processed_system_event_time
    }

    /// Maximum timestamp covered by the modeled event set: the checked issue
    /// frontier (`last local issue timestamp + 1`) and any successfully
    /// processed shared-system event. This remains an unadjusted lower bound;
    /// the explicit SM-active projection adds the observation boundary.
    pub fn terminal_modeled_event_time(&self) -> Option<u64> {
        self.replay.terminal_modeled_event_time
    }

    /// Unadjusted terminal modeled-event lower bound.
    pub fn modeled_terminal_lower_bound_cycles(&self) -> Option<u64> {
        self.replay.terminal_modeled_event_time
    }

    /// SM-active terminal lower bound with post-issue MemoryWriteAck events
    /// excluded from the observation endpoint.  Acknowledgement events remain
    /// in the full modeled frontier and still affect any dependent issue; this
    /// accessor only keeps the two observation domains distinct.
    pub fn sm_active_terminal_event_time(&self) -> Option<u64> {
        self.replay.sm_active_terminal_event_time
    }

    /// Return the architecture-modeled GPC observation frontier for a full
    /// shared-L2 replay.  `None` is reserved for a non-shared compatibility
    /// mode, an empty frontier, an unsupported architecture capability, or a
    /// checked arithmetic overflow; the physical lower-bound field above is
    /// never replaced or hidden.
    pub fn gpc_observation_frontier(&self) -> Option<GpcObservationFrontier> {
        if !matches!(
            self.configuration.memory_mode,
            ReplayMemMode::L1SharedL2 { .. }
        ) {
            return None;
        }
        let lower_bound_cycles = self.modeled_terminal_lower_bound_cycles()?;
        let grid_blocks = self.coverage.launch().grid_blocks().ok()?;
        let envelope = self
            .arch()
            .gpc_observation_envelope()
            .require("gpc_observation_envelope")
            .ok()?;
        let (envelope_cycles, grid_log2) = envelope.cycles_for_grid(grid_blocks)?;
        let cycles = lower_bound_cycles.checked_add(envelope_cycles)?;
        Some(GpcObservationFrontier {
            lower_bound_cycles,
            envelope_cycles,
            grid_blocks,
            grid_log2,
            cycles,
        })
    }

    /// Return the explicit SM-active physical observation frontier for a
    /// shared-L2 replay. The replay clock remains unadjusted; this projection
    /// adds the architecture-scoped observation boundary while retaining the
    /// causal lower bounds in the same structure.
    pub fn sm_active_observation_frontier(&self) -> Option<SmActiveObservationFrontier> {
        if !matches!(
            self.configuration.memory_mode,
            ReplayMemMode::L1SharedL2 { .. }
        ) {
            return None;
        }
        let issue_lower_bound_cycles = self.modeled_issue_lower_bound_cycles();
        let envelope_cycles = self.configuration.sm_active_activation_cycles();
        if issue_lower_bound_cycles == 0 && self.replay.terminal_modeled_event_time.is_none() {
            return None;
        }
        let issue_cycles = issue_lower_bound_cycles.checked_add(envelope_cycles)?;
        // The SM-active observation domain excludes post-issue write
        // acknowledgements and service-only writeback/coherence events.  The
        // full modeled terminal frontier remains available through
        // `terminal_modeled_event_time` and the GPC projection; using it here
        // would let a post-exit acknowledgement move an SM-active endpoint.
        let terminal_lower_bound_cycles = self.sm_active_terminal_event_time();
        let terminal_cycles =
            terminal_lower_bound_cycles.and_then(|cycles| cycles.checked_add(envelope_cycles));
        Some(SmActiveObservationFrontier {
            issue_lower_bound_cycles,
            issue_cycles,
            terminal_lower_bound_cycles,
            terminal_cycles,
            envelope_cycles,
        })
    }

    /// Typed launch-retirement diagnostic. Consumers must distinguish a
    /// complete modeled frontier from a lower bound with known missing tails.
    pub fn modeled_completion_frontier(&self) -> ModeledCompletionFrontier {
        if self.replay.unmodeled_tails.is_empty() {
            ModeledCompletionFrontier::Complete {
                cycles: self.modeled_terminal_lower_bound_cycles(),
            }
        } else {
            ModeledCompletionFrontier::Incomplete {
                lower_bound_cycles: self.modeled_terminal_lower_bound_cycles(),
                unmodeled_tails: self.replay.unmodeled_tails,
            }
        }
    }

    pub fn l1(&self) -> Option<&L1Counters> {
        self.replay.l1.as_ref()
    }

    pub fn l2(&self) -> Option<&L2Counters> {
        self.replay.l2.as_ref()
    }

    pub fn l2_by_source(&self) -> Option<&L2SourceSplit> {
        self.replay.l2_by_source.as_ref()
    }

    pub fn dram(&self) -> Option<&DramStats> {
        self.replay.dram.as_ref()
    }

    pub fn front_end(&self) -> FrontEndCounters {
        self.replay.front_end
    }

    /// Trace-derived dynamic activity and memory-class counts. These are
    /// independent of NCU and never feed timing configuration.
    pub fn trace_metrics(&self) -> &TraceMetrics {
        &self.replay.trace_metrics
    }

    pub fn unmodeled_tails(&self) -> TailFlags {
        self.replay.unmodeled_tails
    }
}

/// Counter-only observation over a verified sampled block roster. This type is
/// intentionally distinct from the full replay payload and contains no per-SM
/// result, timestamp, cycle count, or timing-result conversion.
#[cfg(feature = "campaign")]
#[derive(Debug, Clone)]
pub struct SampledObservedCounters {
    arch: ArchId,
    instructions_issued: u64,
    records_seen: u64,
    l1: Option<L1Counters>,
    l2: Option<L2Counters>,
    l2_by_source: Option<L2SourceSplit>,
    dram: Option<DramStats>,
    front_end: FrontEndCounters,
    unmodeled_tails: TailFlags,
}

#[cfg(feature = "campaign")]
impl SampledObservedCounters {
    pub fn arch(&self) -> ArchId {
        self.arch
    }

    pub fn instructions_issued(&self) -> u64 {
        self.instructions_issued
    }

    pub fn records_seen(&self) -> u64 {
        self.records_seen
    }

    pub fn l1(&self) -> Option<&L1Counters> {
        self.l1.as_ref()
    }

    pub fn l2(&self) -> Option<&L2Counters> {
        self.l2.as_ref()
    }

    pub fn l2_by_source(&self) -> Option<&L2SourceSplit> {
        self.l2_by_source.as_ref()
    }

    pub fn dram(&self) -> Option<&DramStats> {
        self.dram.as_ref()
    }

    pub fn front_end(&self) -> FrontEndCounters {
        self.front_end
    }

    pub fn unmodeled_tails(&self) -> TailFlags {
        self.unmodeled_tails
    }
}

/// Sampled-observed replay result. No method converts this value into a full or
/// timing-bearing replay result.
#[cfg(feature = "campaign")]
#[derive(Debug, Clone)]
pub struct SampledKernelReplayResult {
    coverage: SampledLaunchCoverage,
    configuration: ReplayConfiguration,
    counters: SampledObservedCounters,
}

#[cfg(feature = "campaign")]
impl SampledKernelReplayResult {
    pub fn coverage(&self) -> &SampledLaunchCoverage {
        &self.coverage
    }

    pub fn configuration(&self) -> ReplayConfiguration {
        self.configuration
    }

    pub fn counters(&self) -> &SampledObservedCounters {
        &self.counters
    }
}

/// Coverage-classified outcome for one kernel launch.
#[derive(Debug, Clone)]
pub enum KernelReplayOutcome {
    Full(FullKernelReplayResult),
    #[cfg(feature = "campaign")]
    SampledObserved(SampledKernelReplayResult),
}

impl KernelReplayOutcome {
    fn l2_counters(&self) -> Option<&L2Counters> {
        match self {
            Self::Full(result) => result.l2(),
            #[cfg(feature = "campaign")]
            Self::SampledObserved(result) => result.counters().l2(),
        }
    }

    /// Consume a coverage-classified result only when it represents the full
    /// launch. Keeping this match inside `sim-sm` makes ordinary callers stable
    /// when Cargo feature unification enables the campaign-only sampled
    /// variant elsewhere in the workspace.
    pub fn into_full(self) -> Option<FullKernelReplayResult> {
        match self {
            Self::Full(result) => Some(result),
            #[cfg(feature = "campaign")]
            Self::SampledObserved(_) => None,
        }
    }
}

/// Coverage classification retained by an explicitly unadmitted diagnostic replay.
#[cfg(feature = "campaign")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnadmittedCoverage {
    Full,
    SampledObserved,
}

/// Read-only diagnostics from raw trace/sidecar inputs. This type deliberately
/// cannot be converted into [`FullKernelReplayResult`] or passed to
/// [`aggregate_full_launch_set`]; only [`replay_kernel`] over
/// [`AdmittedLaunchEvidence`] can mint an aggregatable replay result.
///
/// ```compile_fail
/// use sim_sm::{
///     FullLaunchSetDeclaration, UnadmittedReplayDiagnostics,
///     aggregate_full_launch_set,
/// };
///
/// fn reject(
///     declaration: FullLaunchSetDeclaration,
///     diagnostic: UnadmittedReplayDiagnostics,
/// ) {
///     let _ = aggregate_full_launch_set(declaration, [diagnostic]);
/// }
/// ```
#[cfg(feature = "campaign")]
#[derive(Debug, Clone)]
pub struct UnadmittedReplayDiagnostics {
    coverage: UnadmittedCoverage,
    configuration: ReplayConfiguration,
    arch: ArchId,
    blocks_observed: u64,
    population_blocks: u64,
    instructions_issued: u64,
    records_seen: u64,
    per_sm_count: Option<usize>,
    diagnostic_last_issue_cycles: Option<u64>,
    terminal_modeled_event_time: Option<u64>,
    l1: Option<L1Counters>,
    l2: Option<L2Counters>,
    l2_by_source: Option<L2SourceSplit>,
    dram: Option<DramStats>,
    unmodeled_tails: TailFlags,
}

#[cfg(feature = "campaign")]
impl UnadmittedReplayDiagnostics {
    fn from_outcome(outcome: KernelReplayOutcome) -> Self {
        match outcome {
            KernelReplayOutcome::Full(result) => Self {
                coverage: UnadmittedCoverage::Full,
                configuration: result.configuration(),
                arch: result.arch(),
                blocks_observed: result.coverage().blocks_observed(),
                population_blocks: result.coverage().blocks_observed(),
                instructions_issued: result.instructions_issued(),
                records_seen: result.records_seen(),
                per_sm_count: Some(result.per_sm().len()),
                diagnostic_last_issue_cycles: Some(result.diagnostic_last_issue_cycles()),
                terminal_modeled_event_time: result.terminal_modeled_event_time(),
                l1: result.l1().copied(),
                l2: result.l2().copied(),
                l2_by_source: result.l2_by_source().copied(),
                dram: result.dram().copied(),
                unmodeled_tails: result.unmodeled_tails(),
            },
            KernelReplayOutcome::SampledObserved(result) => {
                let counters = result.counters();
                Self {
                    coverage: UnadmittedCoverage::SampledObserved,
                    configuration: result.configuration(),
                    arch: counters.arch(),
                    blocks_observed: result.coverage().blocks_observed(),
                    population_blocks: result.coverage().population_blocks(),
                    instructions_issued: counters.instructions_issued(),
                    records_seen: counters.records_seen(),
                    per_sm_count: None,
                    diagnostic_last_issue_cycles: None,
                    terminal_modeled_event_time: None,
                    l1: counters.l1().copied(),
                    l2: counters.l2().copied(),
                    l2_by_source: counters.l2_by_source().copied(),
                    dram: counters.dram().copied(),
                    unmodeled_tails: counters.unmodeled_tails(),
                }
            }
        }
    }

    pub fn coverage(&self) -> UnadmittedCoverage {
        self.coverage
    }

    pub fn configuration(&self) -> ReplayConfiguration {
        self.configuration
    }

    pub fn arch(&self) -> ArchId {
        self.arch
    }

    pub fn blocks_observed(&self) -> u64 {
        self.blocks_observed
    }

    pub fn population_blocks(&self) -> u64 {
        self.population_blocks
    }

    pub fn instructions_issued(&self) -> u64 {
        self.instructions_issued
    }

    pub fn records_seen(&self) -> u64 {
        self.records_seen
    }

    pub fn per_sm_count(&self) -> Option<usize> {
        self.per_sm_count
    }

    /// Present only for full-coverage diagnostics. This remains an issue-frontier
    /// diagnostic, never a publication-cycle prediction.
    pub fn diagnostic_last_issue_cycles(&self) -> Option<u64> {
        self.diagnostic_last_issue_cycles
    }

    pub fn terminal_modeled_event_time(&self) -> Option<u64> {
        self.terminal_modeled_event_time
    }

    pub fn l1(&self) -> Option<&L1Counters> {
        self.l1.as_ref()
    }

    pub fn l2(&self) -> Option<&L2Counters> {
        self.l2.as_ref()
    }

    pub fn l2_by_source(&self) -> Option<&L2SourceSplit> {
        self.l2_by_source.as_ref()
    }

    pub fn dram(&self) -> Option<&DramStats> {
        self.dram.as_ref()
    }

    pub fn unmodeled_tails(&self) -> TailFlags {
        self.unmodeled_tails
    }
}

#[derive(Debug)]
pub(crate) enum ExtractedCoverage {
    Full(FullLaunchCoverage),
    #[cfg(feature = "campaign")]
    Sampled(SampledLaunchCoverage),
}

#[derive(Debug)]
struct ExtractedKernel {
    per_sm_blocks: BTreeMap<i32, Vec<BlockReplay>>,
    per_sm_instruction_read_sectors: BTreeMap<i32, u64>,
    records_seen: u64,
    coverage: ExtractedCoverage,
    shared_memory_residency: SharedMemoryResidency,
    front_end: FrontEndCounters,
    trace_metrics: TraceMetrics,
}

#[derive(Debug, Clone, Copy)]
enum OrderedAsyncEventKind {
    Issue { effective: bool },
    Commit,
    Wait { max_pending_groups: u32 },
}

#[derive(Debug, Clone, Copy)]
struct OrderedAsyncEvent {
    order: u64,
    launch_index: i64,
    warp_slot: i32,
    kind: OrderedAsyncEventKind,
}

#[derive(Debug, Clone, Copy, Default)]
struct ObservedAsyncWarpState {
    open_effective_copies: u64,
    committed_groups: u64,
}

fn ordered_async_event_kind(
    schedule: &FunctionSchedule,
    instruction: ReplayInst,
) -> Option<OrderedAsyncEventKind> {
    let slot = schedule
        .by_pc
        .get(&u64::from(instruction.pc))
        .unwrap_or_else(|| unreachable!("extraction already validated the complete schedule join"));
    if slot.ldgsts_l1_policy().is_some() {
        return Some(OrderedAsyncEventKind::Issue {
            effective: matches!(instruction.kind, AccessKind::GlobalAsyncCopy { .. }),
        });
    }
    match slot.op_class() {
        OpClass::AsyncCopyCommit => Some(OrderedAsyncEventKind::Commit),
        OpClass::AsyncCopyWait => Some(OrderedAsyncEventKind::Wait {
            max_pending_groups: slot
                .async_copy_wait_max_pending_groups()
                .expect("async-copy wait slot carries its parsed immediate"),
        }),
        OpClass::BlockBarrier | OpClass::Other | OpClass::MBarrierPhaseQuery => None,
    }
}

fn checked_order_observation_add(
    counter: &'static str,
    lhs: u64,
    rhs: u64,
) -> Result<u64, ReplayError> {
    lhs.checked_add(rhs)
        .ok_or(IssueError::CounterOverflow { counter }.into())
}

/// Validate the opt-in selected-record order contract and reconstruct only the
/// asynchronous-copy group populations it directly observes. The returned map
/// is attached to readiness after simulation; neither this scan nor its result
/// is visible to issue selection or any timing/service state.
fn observe_async_instrumentation_order(
    per_sm_blocks: &BTreeMap<i32, Vec<BlockReplay>>,
    schedule: &FunctionSchedule,
) -> Result<BTreeMap<i32, AsyncInstrumentationOrderDiagnostics>, ReplayError> {
    let observation_enabled = per_sm_blocks.values().flatten().any(|block| {
        block.warps.iter().any(|warp| {
            warp.async_copy_cohorts
                .as_deref()
                .is_some_and(|sidecar| !sidecar.instrumentation_order.is_empty())
        })
    });
    if !observation_enabled {
        return Ok(BTreeMap::new());
    }

    let mut seen_orders = BTreeSet::new();
    let mut events_by_sm: BTreeMap<i32, Vec<OrderedAsyncEvent>> = BTreeMap::new();
    for (&sm_id, blocks) in per_sm_blocks {
        for block in blocks {
            for warp in &block.warps {
                let anchors = warp
                    .async_copy_cohorts
                    .as_deref()
                    .map_or(&[][..], |sidecar| sidecar.instrumentation_order.as_slice());
                let mut anchor_index = 0;
                for (instruction_index, &instruction) in warp.insts.iter().enumerate() {
                    let anchor = anchors
                        .get(anchor_index)
                        .filter(|anchor| anchor.instruction == instruction_index)
                        .copied();
                    if anchor.is_some() {
                        anchor_index += 1;
                    }
                    let selected = ordered_async_event_kind(schedule, instruction);
                    match (selected, anchor) {
                        (Some(kind), Some(anchor)) => {
                            if !seen_orders.insert(anchor.order) {
                                return Err(ReplayError::InstrumentationOrderDuplicate {
                                    order: anchor.order,
                                    sm_id,
                                    launch_index: block.launch_index,
                                    warp_slot: warp.warp_slot,
                                    instruction: instruction_index,
                                });
                            }
                            events_by_sm
                                .entry(sm_id)
                                .or_default()
                                .push(OrderedAsyncEvent {
                                    order: anchor.order,
                                    launch_index: block.launch_index,
                                    warp_slot: warp.warp_slot,
                                    kind,
                                });
                        }
                        (Some(_), None) => {
                            return Err(ReplayError::InstrumentationOrderMissing {
                                sm_id,
                                launch_index: block.launch_index,
                                warp_slot: warp.warp_slot,
                                instruction: instruction_index,
                                pc: instruction.pc,
                            });
                        }
                        (None, Some(anchor)) => {
                            return Err(ReplayError::InstrumentationOrderUnexpected {
                                order: anchor.order,
                                sm_id,
                                launch_index: block.launch_index,
                                warp_slot: warp.warp_slot,
                                instruction: instruction_index,
                                pc: instruction.pc,
                            });
                        }
                        (None, None) => {}
                    }
                }
                debug_assert_eq!(
                    anchor_index,
                    anchors.len(),
                    "compact reader anchor index escaped its instruction stream"
                );
            }
        }
    }

    for (offset, &got) in seen_orders.iter().enumerate() {
        let expected = u64::try_from(offset)
            .ok()
            .and_then(|offset| offset.checked_add(1))
            .ok_or(IssueError::CounterOverflow {
                counter: "instrumentation order density",
            })?;
        if got != expected {
            return Err(ReplayError::InstrumentationOrderNotDense { expected, got });
        }
    }

    let mut observations = BTreeMap::new();
    for (sm_id, mut events) in events_by_sm {
        events.sort_unstable_by_key(|event| event.order);
        let mut states: BTreeMap<(i64, i32), ObservedAsyncWarpState> = BTreeMap::new();
        let mut observation = AsyncInstrumentationOrderDiagnostics::default();
        for event in events {
            observation.selected_events = checked_order_observation_add(
                "async instrumentation selected events",
                observation.selected_events,
                1,
            )?;
            let warp_key = (event.launch_index, event.warp_slot);
            match event.kind {
                OrderedAsyncEventKind::Issue { effective } => {
                    observation.issue_events = checked_order_observation_add(
                        "async instrumentation issue events",
                        observation.issue_events,
                        1,
                    )?;
                    if effective {
                        let committed_total = states.values().try_fold(0_u64, |total, state| {
                            checked_order_observation_add(
                                "async instrumentation committed population",
                                total,
                                state.committed_groups,
                            )
                        })?;
                        let own = states
                            .get(&warp_key)
                            .map_or(0, |state| state.committed_groups);
                        let other = committed_total.checked_sub(own).unwrap_or_else(|| {
                            unreachable!("own groups are included in committed total")
                        });
                        if other != 0 {
                            observation.issue_events_with_other_warp_committed_groups =
                                checked_order_observation_add(
                                    "async instrumentation issues with sibling groups",
                                    observation.issue_events_with_other_warp_committed_groups,
                                    1,
                                )?;
                        }
                        observation.other_warp_committed_groups_sum =
                            checked_order_observation_add(
                                "async instrumentation sibling committed groups",
                                observation.other_warp_committed_groups_sum,
                                other,
                            )?;
                        observation.maximum_other_warp_committed_groups =
                            observation.maximum_other_warp_committed_groups.max(other);
                        let state = states.entry(warp_key).or_default();
                        state.open_effective_copies = checked_order_observation_add(
                            "async instrumentation open copies",
                            state.open_effective_copies,
                            1,
                        )?;
                    }
                }
                OrderedAsyncEventKind::Commit => {
                    observation.commit_events = checked_order_observation_add(
                        "async instrumentation commit events",
                        observation.commit_events,
                        1,
                    )?;
                    let state = states.entry(warp_key).or_default();
                    if state.open_effective_copies == 0 {
                        observation.protocol_contradictions = checked_order_observation_add(
                            "async instrumentation protocol contradictions",
                            observation.protocol_contradictions,
                            1,
                        )?;
                    } else {
                        state.committed_groups = checked_order_observation_add(
                            "async instrumentation committed groups",
                            state.committed_groups,
                            1,
                        )?;
                        state.open_effective_copies = 0;
                    }
                }
                OrderedAsyncEventKind::Wait { max_pending_groups } => {
                    observation.wait_events = checked_order_observation_add(
                        "async instrumentation wait events",
                        observation.wait_events,
                        1,
                    )?;
                    let state = states.entry(warp_key).or_default();
                    if state.open_effective_copies != 0 {
                        observation.protocol_contradictions = checked_order_observation_add(
                            "async instrumentation protocol contradictions",
                            observation.protocol_contradictions,
                            1,
                        )?;
                        state.open_effective_copies = 0;
                    }
                    state.committed_groups =
                        state.committed_groups.min(u64::from(max_pending_groups));
                }
            }
        }
        for state in states.values() {
            observation.protocol_contradictions = checked_order_observation_add(
                "async instrumentation protocol contradictions",
                observation.protocol_contradictions,
                u64::from(state.open_effective_copies != 0)
                    + u64::from(state.committed_groups != 0),
            )?;
        }
        observations.insert(sm_id, observation);
    }
    Ok(observations)
}

/// Card-derived spatial-placement parameters. Only the architecture/SKU-local SM
/// range crosses into replay; no label or target value is a simulator input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PlacementConfig {
    arch: ArchId,
    sku: Sku,
    sm_count: u32,
    /// Upper bound of the `%smid` identifier domain.  PTX explicitly permits
    /// non-contiguous identifiers, so this is not used as a dense SM count.
    sm_id_upper_bound: u32,
    /// Maximum physical warp slot admitted by the runtime launch contract.
    /// This is deliberately sourced from `runtime_limits` when a card has
    /// that block; it is not an issue-width or scheduler hypothesis.
    max_resident_warps_per_sm: u32,
    /// Launch-envelope block/thread ceilings.  These are retained even when
    /// no issue model exists; a trace can be admitted for structural checks
    /// without inventing scheduler throughput.
    max_resident_blocks_per_sm: u32,
    max_threads_per_sm: u32,
    max_threads_per_block: u32,
    max_shared_bytes_per_sm: u64,
    max_shared_bytes_per_block: u64,
    max_shared_bytes_per_block_optin: u64,
}

impl PlacementConfig {
    fn from_arch_card(card: &ArchCard, expected_arch: ArchId) -> Result<Self, ReplayError> {
        let card_error = |source| ReplayError::ArchitectureCardPlacement { source };
        let arch = card.arch_id().map_err(card_error)?;
        require_replay_arch(expected_arch, arch, "architecture card")?;
        let sku = card.sku_id().map_err(card_error)?;
        let sm_count = card.sm_count().map_err(card_error)?;
        // Placement needs two architecture-local *launch-envelope* facts:
        // physical warp-slot capacity and per-SM shared-memory capacity.  A
        // Blackwell card intentionally leaves L1 geometry as a research gap,
        // so consulting `l1_geometry` here would incorrectly couple placement
        // to an unmeasured cache hypothesis.  Prefer the observed CUDA
        // runtime-limit block when present.  The legacy SM86 card predates
        // that block; retain its validated L1 shared-capacity field only as a
        // compatibility fallback, and never use it to admit SM120.
        let (
            max_resident_warps_per_sm,
            sm_id_upper_bound,
            max_resident_blocks_per_sm,
            max_threads_per_sm,
            max_threads_per_block,
            max_shared_bytes_per_sm,
            max_shared_bytes_per_block,
            max_shared_bytes_per_block_optin,
        ) = match card.require_runtime_limits() {
            Ok(limits) => (
                limits.max_resident_warps_per_sm,
                limits.sm_id_upper_bound,
                limits.max_blocks_per_sm,
                limits.max_threads_per_sm,
                limits.max_threads_per_block,
                limits.max_shared_memory_per_sm_bytes,
                limits.max_shared_memory_per_block_bytes,
                limits.max_shared_memory_per_block_optin_bytes,
            ),
            Err(
                gap @ CalibError::BlockResearchGap {
                    block: "runtime_limits",
                    ..
                },
            ) => {
                if arch != ArchId::Sm86 {
                    return Err(ReplayError::ArchitectureCardSharedMemoryCapacity {
                        arch,
                        source: gap,
                    });
                }
                let l1 = card.require_l1_geometry().map_err(|source| {
                    ReplayError::ArchitectureCardSharedMemoryCapacity { arch, source }
                })?;
                let (legacy_warps, legacy_blocks) = match card.require_occupancy_resources() {
                    Ok(occupancy) => (occupancy.max_resident_warps_per_sm, occupancy.max_blocks),
                    Err(CalibError::BlockResearchGap {
                        block: "occupancy", ..
                    }) => arch
                        .issue_hypotheses()
                        .require("legacy physical warp-slot capacity")
                        .map(|hypotheses| {
                            (
                                hypotheses.max_resident_warps_per_sm,
                                hypotheses.max_resident_blocks_per_sm,
                            )
                        })
                        .map_err(|source| ReplayError::WarpCapacityUnavailable { arch, source })?,
                    Err(source) => {
                        return Err(ReplayError::ArchitectureCardSharedMemoryCapacity {
                            arch,
                            source,
                        });
                    }
                };
                (
                    legacy_warps,
                    sm_count,
                    legacy_blocks,
                    legacy_warps.saturating_mul(32),
                    legacy_warps.saturating_mul(32),
                    l1.max_shared_bytes,
                    l1.max_shared_bytes,
                    l1.max_shared_bytes,
                )
            }
            Err(source) => {
                return Err(ReplayError::ArchitectureCardSharedMemoryCapacity { arch, source });
            }
        };
        if max_shared_bytes_per_sm == 0 {
            return Err(ReplayError::SharedMemoryCapacityInvalid {
                arch,
                bytes: max_shared_bytes_per_sm,
            });
        }
        Ok(Self {
            arch,
            sku,
            sm_count,
            sm_id_upper_bound,
            max_resident_warps_per_sm,
            max_resident_blocks_per_sm,
            max_threads_per_sm,
            max_threads_per_block,
            max_shared_bytes_per_sm,
            max_shared_bytes_per_block,
            max_shared_bytes_per_block_optin,
        })
    }

    fn shared_memory_residency(
        self,
        kernel: &pb::Kernel,
    ) -> Result<SharedMemoryResidency, ReplayError> {
        let bytes_per_cta = u64::try_from(kernel.size_shared_memory).map_err(|_| {
            ReplayError::SharedMemoryDemandInvalid {
                kernel_id: kernel.id,
                bytes: kernel.size_shared_memory,
            }
        })?;
        if bytes_per_cta > self.max_shared_bytes_per_sm {
            return Err(ReplayError::CtaSharedMemoryCapacityExceeded {
                kernel_id: kernel.id,
                arch: self.arch,
                bytes_per_cta,
                max_bytes_per_sm: self.max_shared_bytes_per_sm,
            });
        }
        if bytes_per_cta > self.max_shared_bytes_per_block_optin {
            return Err(ReplayError::CtaSharedMemoryBlockLimitExceeded {
                kernel_id: kernel.id,
                arch: self.arch,
                bytes_per_cta,
                max_bytes_per_block: self.max_shared_bytes_per_block_optin,
            });
        }
        let block = kernel
            .block_dim
            .as_ref()
            .filter(|dim| dim.x > 0 && dim.y > 0 && dim.z > 0)
            .map(|dim| {
                u64::try_from(dim.x)
                    .ok()
                    .and_then(|x| u64::try_from(dim.y).ok().and_then(|y| x.checked_mul(y)))
                    .and_then(|xy| u64::try_from(dim.z).ok().and_then(|z| xy.checked_mul(z)))
            })
            .flatten();
        if let Some(threads_per_cta) = block {
            if threads_per_cta > u64::from(self.max_threads_per_block) {
                return Err(ReplayError::CtaThreadCapacityExceeded {
                    kernel_id: kernel.id,
                    arch: self.arch,
                    threads_per_cta,
                    max_threads_per_block: self.max_threads_per_block,
                });
            }
        }
        Ok(SharedMemoryResidency::new(
            self.max_shared_bytes_per_sm,
            bytes_per_cta,
        ))
    }

    fn require_traced_sm(self, path: &Path, sm_id: i32) -> Result<(), ReplayError> {
        if sm_id < 0 || u32::try_from(sm_id).unwrap_or(u32::MAX) >= self.sm_id_upper_bound {
            return Err(ReplayError::SmIdOutOfRange {
                path: path.to_path_buf(),
                sm_id,
                arch: self.arch,
                sku: self.sku,
                sm_id_upper_bound: self.sm_id_upper_bound,
            });
        }
        Ok(())
    }

    #[cfg(feature = "campaign")]
    fn require_declared_sm(self, launch_index: u64, sm_id: u64) -> Result<(), ReplayError> {
        if sm_id >= u64::from(self.sm_id_upper_bound) {
            return Err(ReplayError::SamplingSmIdOutOfRange {
                launch_index,
                sm_id,
                arch: self.arch,
                sku: self.sku,
                sm_id_upper_bound: self.sm_id_upper_bound,
            });
        }
        Ok(())
    }
}

#[cfg(feature = "campaign")]
fn validate_declared_sampling_sm_range(
    declared: impl IntoIterator<Item = (u64, u64)>,
    placement: PlacementConfig,
) -> Result<(), ReplayError> {
    for (launch_index, sm_id) in declared {
        placement.require_declared_sm(launch_index, sm_id)?;
    }
    Ok(())
}

#[cfg(feature = "campaign")]
fn validate_sampling_placement(
    sidecar: &TraceSamplingSidecar,
    traced_sm_by_launch_index: &BTreeMap<i64, i32>,
    placement: PlacementConfig,
) -> Result<(), ReplayError> {
    validate_declared_sampling_sm_range(sidecar.declared_block_placements(), placement)?;
    for (&launch_index, &traced_sm_id) in traced_sm_by_launch_index {
        let declared_sm_id = sidecar
            .declared_sm_for_launch_index(launch_index)
            .ok_or(ReplayError::SamplingPlacementMissing { launch_index })?;
        if u64::try_from(traced_sm_id).ok() != Some(declared_sm_id) {
            return Err(ReplayError::SamplingPlacementMismatch {
                launch_index,
                traced_sm_id,
                declared_sm_id,
            });
        }
    }
    Ok(())
}

fn normalized_cluster_dim(
    kernel: &sim_trace::pb::Kernel,
    arch: ArchId,
) -> Result<(u32, u32, u32), ReplayError> {
    let Some(cluster) = kernel.cluster_dim.as_ref() else {
        return Ok((1, 1, 1));
    };
    let raw = (cluster.x, cluster.y, cluster.z);
    if raw == (0, 0, 0) {
        return Ok((1, 1, 1));
    }
    if raw.0 <= 0 || raw.1 <= 0 || raw.2 <= 0 {
        return Err(ReplayError::ClusterDimInvalid { cluster: raw });
    }
    let cluster = (raw.0 as u32, raw.1 as u32, raw.2 as u32);
    if cluster != (1, 1, 1) {
        return Err(ReplayError::ClusterGeometryUnsupported { arch, cluster });
    }
    Ok(cluster)
}

fn checked_grid_total(grid: (u32, u32, u32)) -> Result<u64, ReplayError> {
    u64::from(grid.0)
        .checked_mul(u64::from(grid.1))
        .and_then(|value| value.checked_mul(u64::from(grid.2)))
        .ok_or(ReplayError::GridSizeOverflow { grid })
}

fn checked_block_warps(block: (u32, u32, u32)) -> Result<u64, ReplayError> {
    const TRACE_WARP_WIDTH: u64 = 32;
    let threads = u64::from(block.0)
        .checked_mul(u64::from(block.1))
        .and_then(|value| value.checked_mul(u64::from(block.2)))
        .ok_or(ReplayError::BlockSizeOverflow { block })?;
    Ok(threads.div_ceil(TRACE_WARP_WIDTH))
}

fn checked_record_count(
    accumulated: u64,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<u64, ReplayError> {
    accumulated
        .checked_add(1)
        .ok_or_else(|| ReplayError::RecordCountOverflow {
            path: path.to_path_buf(),
            warp_id,
            record_ix,
            accumulated,
        })
}

fn checked_collection_len(collection: &'static str, len: usize) -> Result<u64, ReplayError> {
    u64::try_from(len).map_err(|_| ReplayError::CollectionLengthOutOfRange { collection, len })
}

fn is_sync_sidecar_opcode(opcode: &str) -> bool {
    opcode.starts_with("SYNCS.") || opcode.starts_with("ARRIVES.")
}

/// Count distinct 32-byte instruction sectors assigned to one traced SM.
/// This is kept separate from the launch-wide front-end count so a later
/// effective observation envelope can use the same scope as active-lifetime
/// IPC. It is metadata only and never contributes replay cycles.
fn per_sm_instruction_sector_set(
    blocks: &[BlockReplay],
    function_unique_id: i32,
) -> BTreeSet<(i32, u32)> {
    let mut sectors = BTreeSet::new();
    for block in blocks {
        for warp in &block.warps {
            for inst in &warp.insts {
                sectors.insert((function_unique_id, inst.pc / 32));
            }
        }
    }
    sectors
}

/// Extract distinct 32-byte constant-bank sectors from one static SASS
/// instruction. Constant references are retained as source metadata only;
/// malformed/non-numeric operands are ignored rather than guessed.
fn static_constant_sectors(opcode_text: &str) -> Box<[u64]> {
    fn literal(bytes: &[u8]) -> Option<u64> {
        let text = std::str::from_utf8(bytes).ok()?.trim();
        if let Some(hex) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
            u64::from_str_radix(hex, 16).ok()
        } else {
            text.parse().ok()
        }
    }

    let bytes = opcode_text.as_bytes();
    let mut sectors = BTreeSet::new();
    let mut cursor = 0usize;
    while cursor + 3 < bytes.len() {
        let is_constant = bytes[cursor] == b'c'
            && (bytes[cursor + 1] == b'['
                || (bytes[cursor + 1] == b'x'
                    && cursor + 2 < bytes.len()
                    && bytes[cursor + 2] == b'['));
        if !is_constant {
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
        if let (Some(bank), Some(offset)) = (
            literal(&bytes[first_open + 1..first_close]),
            literal(&bytes[second_open + 1..second_close]),
        ) {
            if let Some(address) = bank
                .checked_mul(65_536)
                .and_then(|base| base.checked_add(offset))
            {
                sectors.insert(address / 32);
            }
        }
        cursor = second_close + 1;
    }
    sectors.into_iter().collect()
}

