//! Fast replay session state and completion helpers; active replay orchestration.

use super::*;

/// Build a bounded, address-derived TMA source shadow for L2 metric coverage.
///
/// TMA source addresses are present in the native sidecar but are not carried
/// by the compact dynamic memory payload. The shadow therefore runs in a
/// separate L2 model, in sidecar instruction order, and never changes the
/// shared replay cache, timing, DRAM demand, or issue schedule. It is an
/// effective observer projection: ordinary L1/L2 requests and TMA requests
/// are not falsely serialized into one physical stream.
#[derive(Default)]
pub(crate) struct TmaShadowL2Stats {
    pub(crate) requests: u64,
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    pub(crate) unavailable: u64,
    pub(crate) load: [u64; 4],
    pub(crate) store: [u64; 4],
}

pub(crate) fn observe_tma_l2_shadow(
    observations: &BTreeMap<TmaObservationKey, TmaObservation>,
    config: L2Config,
) -> Result<TmaShadowL2Stats, ReplayError> {
    observe_tma_l2_shadow_iter(observations.values(), config)
}

pub(crate) fn tma_sidecar_reuse_summary(
    observations: &LaunchObservations,
) -> (
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
) {
    let mut all_sectors = BTreeSet::new();
    let mut all_lines = BTreeSet::new();
    let mut sum_record_unique_sectors = 0_u64;
    let mut sum_record_unique_lines = 0_u64;
    let mut intra_record_duplicate_refs = 0_u64;
    let mut load_sum_sectors = 0_u64;
    let mut store_sum_sectors = 0_u64;
    let mut load_sum_lines = 0_u64;
    let mut store_sum_lines = 0_u64;
    let mut load_sectors = BTreeSet::new();
    let mut store_sectors = BTreeSet::new();
    let mut load_lines = BTreeSet::new();
    let mut store_lines = BTreeSet::new();
    for observation in observations.tma.values() {
        let ids = if observation.is_load {
            &observation.source_sector_ids
        } else if observation.is_store {
            &observation.destination_sector_ids
        } else {
            continue;
        };
        sum_record_unique_sectors =
            sum_record_unique_sectors.saturating_add(observation.diagnostic.unique_sector_count);
        sum_record_unique_lines =
            sum_record_unique_lines.saturating_add(observation.diagnostic.unique_line_count);
        intra_record_duplicate_refs = intra_record_duplicate_refs
            .saturating_add(observation.diagnostic.intra_record_duplicate_sector_refs);
        if observation.is_load {
            load_sum_sectors =
                load_sum_sectors.saturating_add(observation.diagnostic.unique_sector_count);
            load_sum_lines =
                load_sum_lines.saturating_add(observation.diagnostic.unique_line_count);
        } else {
            store_sum_sectors =
                store_sum_sectors.saturating_add(observation.diagnostic.unique_sector_count);
            store_sum_lines =
                store_sum_lines.saturating_add(observation.diagnostic.unique_line_count);
        }
        for &sector in ids.iter() {
            all_sectors.insert(sector);
            all_lines.insert(sector.div_euclid(4));
            if observation.is_load {
                load_sectors.insert(sector);
                load_lines.insert(sector.div_euclid(4));
            } else {
                store_sectors.insert(sector);
                store_lines.insert(sector.div_euclid(4));
            }
        }
    }
    let cross_sectors = load_sectors.intersection(&store_sectors).count();
    let cross_lines = load_lines.intersection(&store_lines).count();
    (
        sum_record_unique_sectors,
        sum_record_unique_lines,
        u64::try_from(all_sectors.len()).unwrap_or(u64::MAX),
        u64::try_from(all_lines.len()).unwrap_or(u64::MAX),
        intra_record_duplicate_refs,
        load_sum_sectors,
        store_sum_sectors,
        load_sum_lines,
        store_sum_lines,
        u64::try_from(load_sectors.len()).unwrap_or(u64::MAX),
        u64::try_from(store_sectors.len()).unwrap_or(u64::MAX),
        u64::try_from(cross_sectors).unwrap_or(u64::MAX),
        u64::try_from(cross_lines).unwrap_or(u64::MAX),
    )
}

/// Provenance by which an observed complete launch became full-coverage replay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FullCoverageSource {
    /// No sampling declaration was present; the replay gate observed every CTA.
    UnsampledComplete,
    /// A validated lossless sidecar was present and replay independently observed
    /// every CTA. The sidecar declaration alone never constructs this variant.
    #[cfg(feature = "campaign")]
    LosslessCompressedFull(LosslessFullProvenance),
}

/// Timing-bearing replay payload kept private so coverage cannot be detached or
/// fabricated at the public API boundary.
#[derive(Debug, Clone)]
pub(crate) struct FoldedKernelReplay {
    pub(crate) arch: ArchId,
    pub(crate) instructions_issued: u64,
    pub(crate) records_seen: u64,
    pub(crate) per_sm: BTreeMap<i32, SmSimResult>,
    pub(crate) diagnostic_last_issue_cycles: u64,
    pub(crate) last_processed_system_event_time: Option<u64>,
    pub(crate) terminal_modeled_event_time: Option<u64>,
    pub(crate) sm_active_terminal_event_time: Option<u64>,
    pub(crate) l1: Option<L1Counters>,
    pub(crate) l2: Option<L2Counters>,
    pub(crate) l2_by_source: Option<L2SourceSplit>,
    pub(crate) dram: Option<DramStats>,
    pub(crate) front_end: FrontEndCounters,
    pub(crate) trace_metrics: TraceMetrics,
    pub(crate) unmodeled_tails: TailFlags,
}

/// SM-domain physical observation frontier. The replay's issue and terminal
/// timestamps remain the unadjusted modeled lower bounds; this explicit
/// projection adds only the independently measured SM-active observation
/// boundary and retains both values for auditability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmActiveObservationFrontier {
    pub issue_lower_bound_cycles: u64,
    pub issue_cycles: u64,
    pub terminal_lower_bound_cycles: Option<u64>,
    pub terminal_cycles: Option<u64>,
    pub envelope_cycles: u64,
}

pub(crate) fn fast_cycle_add(base: u64, delta: u64) -> Result<u64, ReplayError> {
    base.checked_add(delta)
        .ok_or_else(|| ReplayError::FastFrontier {
            detail: "fast replay cycle overflow".to_owned(),
        })
}

pub(crate) fn fast_register_mask(register: SassRegister, mask: u32) -> u32 {
    match register.file() {
        crate::sass_operands::SassRegisterFile::Uniform
        | crate::sass_operands::SassRegisterFile::UniformPredicate
            if mask != 0 =>
        {
            u32::MAX
        }
        _ => mask,
    }
}

pub(crate) fn fast_pipeline_index(pipeline: PipelineClass) -> usize {
    match pipeline {
        PipelineClass::ScalarAlu { .. } => 0,
        PipelineClass::Tensor { .. } => 1,
        PipelineClass::Sfu { .. } => 2,
        _ => 3,
    }
}

pub(crate) fn fast_scalar_segments(
    warp: &WarpReplay,
    schedule: &FunctionSchedule,
) -> Vec<Vec<CausalFrontierInstruction>> {
    let mut segments = Vec::new();
    let mut program = Vec::new();
    let mut prior_destinations: Vec<SassRegister> = Vec::new();
    let mut segment_start = None;

    pub(crate) fn register_overlap(left: SassRegister, right: SassRegister) -> bool {
        if left.file() != right.file() {
            return false;
        }
        let (Some(left_index), Some(right_index)) = (left.index(), right.index()) else {
            return false;
        };
        let left_start = u32::from(left_index);
        let right_start = u32::from(right_index);
        let left_end = left_start + u32::from(left.span_registers());
        let right_end = right_start + u32::from(right.span_registers());
        left_start < right_end && right_start < left_end
    }

    let flush = |segments: &mut Vec<Vec<CausalFrontierInstruction>>,
                 program: &mut Vec<CausalFrontierInstruction>,
                 prior_destinations: &mut Vec<SassRegister>,
                 segment_start: &mut Option<usize>| {
        // A valid entry boundary may begin after an unmodelled prefix. Do not
        // silently discard a safe suffix merely because its local trace index
        // is nonzero; coverage accounting records the prefix separately.
        if program.len() >= 2 && segment_start.is_some() {
            segments.push(std::mem::take(program));
        } else {
            program.clear();
        }
        prior_destinations.clear();
        *segment_start = None;
    };
    for (sequence, instruction) in warp.insts.iter().enumerate() {
        let descriptor = match (|| {
            if !matches!(instruction.kind, AccessKind::NonMemory) {
                return None;
            }
            let slot = schedule.issue_slot(u64::from(instruction.pc))?;
            if slot.op_class() != OpClass::Other || !matches!(slot.mem_role(), MemRole::None) {
                return None;
            }
            let PipelineClass::ScalarAlu { latency_cycles, .. } = slot.pipeline_class() else {
                return None;
            };
            let control = slot.control();
            if control.wait_mask().bits() != 0
                || control.write_barrier().is_some()
                || control.read_barrier().is_some()
            {
                return None;
            }
            let projection = match slot.fixed_latency_operands() {
                FixedLatencyOperandStatus::Projected(projection) => projection,
                // The compact dynamic trace carries no register operands. A
                // static projection is therefore required before a region can
                // be certified; unknown provenance must remain a refusal.
                FixedLatencyOperandStatus::NotApplicable | FixedLatencyOperandStatus::Gap(_) => {
                    return None;
                }
            };
            let mut current_sources = Vec::new();
            for source in projection.sources() {
                if let SassSourceOperand::Register(operand) = source {
                    current_sources.push(operand.register());
                }
            }
            current_sources.extend(
                projection
                    .predicate_sources()
                    .iter()
                    .map(|operand| operand.register()),
            );
            let current_destinations = projection
                .destinations()
                .iter()
                .map(|operand| operand.register())
                .collect::<Vec<_>>();
            if current_sources.iter().any(|source| {
                prior_destinations
                    .iter()
                    .any(|dest| register_overlap(*source, *dest))
            }) || current_destinations.iter().any(|dest| {
                prior_destinations
                    .iter()
                    .any(|prior| register_overlap(*dest, *prior))
            }) {
                return None;
            }
            let sequence = u64::try_from(sequence).ok()?;
            let descriptor = CausalFrontierInstruction::from_schedule(
                schedule,
                u64::from(instruction.pc),
                sequence,
                warp.warp_slot,
                latency_cycles,
                1,
                true,
                false,
            )
            .ok()?;
            Some((descriptor, current_destinations))
        })() {
            Some((descriptor, destinations)) => {
                if program.is_empty() {
                    segment_start = Some(sequence);
                }
                prior_destinations.extend(destinations);
                descriptor
            }
            None => {
                flush(
                    &mut segments,
                    &mut program,
                    &mut prior_destinations,
                    &mut segment_start,
                );
                continue;
            }
        };
        program.push(descriptor);
    }
    flush(
        &mut segments,
        &mut program,
        &mut prior_destinations,
        &mut segment_start,
    );
    segments
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

#[derive(Debug, Clone, Copy)]
pub(crate) enum OrderedAsyncEventKind {
    Issue { effective: bool },
    Commit,
    Wait { max_pending_groups: u32 },
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ObservedAsyncWarpState {
    pub(crate) open_effective_copies: u64,
    pub(crate) committed_groups: u64,
}

pub(crate) fn checked_order_observation_add(
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
pub(crate) fn observe_async_instrumentation_order(
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

/// Typed failures for an ordered same-stream replay session. Session errors
/// are separate from one-launch replay failures so callers can distinguish a
/// malformed launch from a launch-chain ownership/order mismatch.
#[derive(Debug, thiserror::Error)]
pub enum ReplaySessionError {
    #[error(transparent)]
    Replay(#[from] ReplayError),
    #[error("stream-L2 cache scope requires L1SharedL2 replay mode, got {mode:?}")]
    SharedL2ModeRequired { mode: ReplayMemMode },
    #[error("stream-L2 replay does not persist the opt-in coherence directory; use launch scope")]
    CoherenceRequiresLaunchScope,
    #[error("trace stream ({device_id}, {stream_id}) contains no kernel launches")]
    EmptyStream { device_id: i64, stream_id: i64 },
    #[error("replay session is bound to trace root {expected}, got {got}")]
    TraceRootMismatch { expected: PathBuf, got: PathBuf },
    #[error("replay session is bound to a different loaded trace lineage")]
    TraceLineageMismatch,
    #[error("replay session is bound to device {expected}, got {got}")]
    DeviceMismatch { expected: i64, got: i64 },
    #[error("replay session is bound to stream {expected}, got {got}")]
    StreamMismatch { expected: i64, got: i64 },
    #[error("replay session configuration mismatch: expected {expected:?}, got {got:?}")]
    ConfigurationMismatch {
        expected: ReplayConfiguration,
        got: ReplayConfiguration,
    },
    #[error(
        "replay session architecture-card placement mismatch: expected sku={expected_sku:?}, sm_count={expected_sm_count}, max_shared_bytes_per_sm={expected_max_shared_bytes_per_sm}; got sku={got_sku:?}, sm_count={got_sm_count}, max_shared_bytes_per_sm={got_max_shared_bytes_per_sm}"
    )]
    ArchitectureCardMismatch {
        expected_sku: Sku,
        expected_sm_count: u32,
        expected_max_shared_bytes_per_sm: u64,
        got_sku: Sku,
        got_sm_count: u32,
        got_max_shared_bytes_per_sm: u64,
    },
    #[error(
        "replay session launch position {position} expects kernel {expected_kernel_id}/function {expected_function_unique_id}, got kernel {got_kernel_id}/function {got_function_unique_id}"
    )]
    LaunchOrderMismatch {
        position: usize,
        expected_kernel_id: i32,
        expected_function_unique_id: i32,
        got_kernel_id: i32,
        got_function_unique_id: i32,
    },
    #[error("replay session already consumed all {completed_launches} launches")]
    LaunchSequenceComplete { completed_launches: usize },
    #[error("stream-L2 replay outcome omitted its per-launch L2 counter delta")]
    LaunchL2CountersMissing,
    #[error(
        "stream-L2 cumulative counters do not reconcile: reported={reported:?}, model={model:?}"
    )]
    CumulativeL2CounterMismatch {
        reported: L2Counters,
        model: L2Counters,
    },
}

pub(crate) fn validate_replay_architecture(
    schedule: &FunctionSchedule,
    params: &IssueModelParams,
    mode: ReplayMemMode,
    memory_admission_feedback: Option<MemoryAdmissionCandidate>,
) -> Result<ReplayConfiguration, ReplayError> {
    require_replay_arch(schedule.arch, params.arch, "issue parameters")?;
    match memory_admission_feedback {
        Some(candidate) => {
            ReplayConfiguration::checked_with_memory_admission_feedback(*params, mode, candidate)
        }
        None => ReplayConfiguration::checked(*params, mode),
    }
}

pub(crate) fn replay_kernel_with_schedule_and_l2(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedule: &FunctionSchedule,
    arch_card: &ArchCard,
    params: &IssueModelParams,
    mode: ReplayMemMode,
    retained_l2: Option<&mut L2Model>,
) -> Result<KernelReplayOutcome, ReplayError> {
    replay_kernel_with_schedule_and_l2_and_candidates(
        td,
        loc,
        schedule,
        arch_card,
        params,
        mode,
        retained_l2,
        &[],
        &[],
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn replay_kernel_with_schedule_and_l2_and_candidates(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedule: &FunctionSchedule,
    arch_card: &ArchCard,
    params: &IssueModelParams,
    mode: ReplayMemMode,
    retained_l2: Option<&mut L2Model>,
    memory_service_candidates: &[MemoryServiceCandidate],
    memory_admission_candidates: &[MemoryAdmissionCandidate],
    memory_admission_feedback: Option<MemoryAdmissionCandidate>,
) -> Result<KernelReplayOutcome, ReplayError> {
    replay_kernel_with_schedule_and_l2_and_candidates_with_payload_events(
        td,
        loc,
        schedule,
        arch_card,
        params,
        mode,
        retained_l2,
        memory_service_candidates,
        memory_admission_candidates,
        memory_admission_feedback,
        &[],
    )
}

pub(crate) fn replay_kernel_with_schedule(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedule: &FunctionSchedule,
    arch_card: &ArchCard,
    params: &IssueModelParams,
    mode: ReplayMemMode,
) -> Result<KernelReplayOutcome, ReplayError> {
    replay_kernel_with_schedule_and_l2(td, loc, schedule, arch_card, params, mode, None)
}

/// Replay the one launch represented by [`AdmittedLaunchEvidence`]. This is the
/// compatibility entry point used by the optional historical campaign runner.
///
/// Caches are fresh per kernel (cold, NCU `--cache-control all` parity).
#[cfg(feature = "campaign")]
pub fn replay_kernel(
    evidence: &AdmittedLaunchEvidence,
    params: &IssueModelParams,
    mode: ReplayMemMode,
) -> Result<KernelReplayOutcome, ReplayError> {
    let identity = evidence.identity();
    let td = evidence.trace_dir();
    let loc = td
        .kernels()
        .find(|loc| {
            loc.device_id() == identity.device_id()
                && loc.stream_id() == identity.stream_id()
                && loc.kernel().id == identity.kernel_id()
                && loc.kernel().function_unique_id == identity.function_unique_id()
        })
        .ok_or(ReplayError::AdmittedLaunchMissing {
            device_id: identity.device_id(),
            stream_id: identity.stream_id(),
            kernel_id: identity.kernel_id(),
        })?;
    replay_kernel_with_schedule(
        td,
        &loc,
        evidence.schedule(),
        evidence.arch_card(),
        params,
        mode,
    )
}

/// Replay one launch directly from loaded trace, schedule, architecture-card,
/// and model inputs. This is the ordinary research entry point; it does not
/// require an evidence manifest or an admission step.
pub fn replay_trace_kernel(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedules: &BTreeMap<i32, FunctionSchedule>,
    arch_card: &ArchCard,
    params: &IssueModelParams,
    mode: ReplayMemMode,
) -> Result<KernelReplayOutcome, ReplayError> {
    let schedule = lookup_schedule(schedules, loc)?;
    replay_kernel_with_schedule(td, loc, schedule, arch_card, params, mode)
}

/// Replay one launch with explicit producer-supplied value-bearing L1 events.
/// Each event carries its simulator-cycle timestamp and a residency witness;
/// this API never converts NCU/CUPTI/device time or infers a cycle from payload
/// sequence. Ordinary replay remains unchanged when this entry point is not
/// selected.
pub fn replay_trace_kernel_with_l1_payload_events(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedules: &BTreeMap<i32, FunctionSchedule>,
    arch_card: &ArchCard,
    params: &IssueModelParams,
    mode: ReplayMemMode,
    payload_events: &[L1PayloadAdmissionEvent],
) -> Result<KernelReplayOutcome, ReplayError> {
    if !payload_events.is_empty() && !matches!(mode, ReplayMemMode::L1SharedL2 { .. }) {
        return Err(ReplayError::PayloadAdmissionRequiresSharedL2);
    }
    let schedule = lookup_schedule(schedules, loc)?;
    let locations = td.dynamic_memory_join_locations(loc)?;
    let mut bound_events = payload_events.to_vec();
    for (index, event) in bound_events.iter_mut().enumerate() {
        let Some(identity) = event.identity else {
            return Err(ReplayError::PayloadAdmissionIdentityMissing { index });
        };
        let Some(location) = locations.get(&identity) else {
            return Err(ReplayError::PayloadAdmissionIdentityNotObserved { index });
        };
        if location.sm_id != Some(event.sm_id) {
            return Err(ReplayError::PayloadAdmissionSmMismatch {
                index,
                event_sm: event.sm_id,
                trace_sm: location.sm_id,
            });
        }
        if location.access_kind != Some(pb::MemoryOpAccessKind::Store) {
            return Err(ReplayError::PayloadAdmissionNotStore {
                index,
                access_kind: location.access_kind,
            });
        }
        let Some(warp_slot) = location.warp_slot else {
            return Err(ReplayError::PayloadAdmissionWriterPlacementMissing { index });
        };
        event.writer = Some(PayloadStoreSite {
            launch_index: location.launch_index,
            warp_slot,
            instruction_index: usize::try_from(location.instruction_index)
                .map_err(|_| ReplayError::PayloadAdmissionWriterPlacementMissing { index })?,
        });
    }
    replay_kernel_with_schedule_and_l2_and_candidates_with_payload_events(
        td,
        loc,
        schedule,
        arch_card,
        params,
        mode,
        None,
        &[],
        &[],
        None,
        &bound_events,
    )
}

/// Replay one launch with one feedback-coupled finite admission candidate.
/// Unlike schema-9 observation, this lane participates in warp readiness and
/// therefore may change issue and cache order. It remains explicit opt-in.
pub fn replay_trace_kernel_with_memory_admission_feedback(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedules: &BTreeMap<i32, FunctionSchedule>,
    arch_card: &ArchCard,
    params: &IssueModelParams,
    mode: ReplayMemMode,
    memory_admission_feedback: MemoryAdmissionCandidate,
) -> Result<KernelReplayOutcome, ReplayError> {
    let schedule = lookup_schedule(schedules, loc)?;
    replay_kernel_with_schedule_and_l2_and_candidates(
        td,
        loc,
        schedule,
        arch_card,
        params,
        mode,
        None,
        &[],
        &[],
        Some(memory_admission_feedback),
    )
}

/// Resolve the two special Global routes implemented on the active sm_86 path.
/// The dynamic enum alone is not enough: the static sidecar must prove a
/// Global ATOM/RED family or LDGSTS role before the record enters replay.
pub(crate) fn require_sm86_special_role(
    schedule: &FunctionSchedule,
    arch: ArchId,
    pc: u32,
    route: ReplayMemoryRoute,
    class: pb::MemoryOpClass,
) -> Result<(), ReplayError> {
    let alpha_sm120 = arch == ArchId::Sm120;
    if arch != ArchId::Sm86 && !alpha_sm120 {
        return Err(ReplayError::UnsupportedMemoryPath { arch, class });
    }
    let slot = schedule
        .by_pc
        .get(&u64::from(pc))
        .ok_or(ReplayError::UnsupportedMemoryPath { arch, class })?;
    let head = opcode_head(&slot.opcode_text);
    if alpha_sm120 && class == pb::MemoryOpClass::AsyncCopy && head == "LDGSTS" {
        // Alpha treats an observed async/TMA source as a flat global demand;
        // no SM86 LDGSTS completion or cache route is attached.
        return Ok(());
    }
    let valid = match (route, class) {
        (ReplayMemoryRoute::Global, pb::MemoryOpClass::Atomic) => match slot.mem_role {
            MemRole::BypassLoad(CacheOpToken::Atom) => {
                matches!(head, "ATOM" | "ATOMG") || (alpha_sm120 && head == "REDG")
            }
            MemRole::Reduction => matches!(head, "RED" | "REDG"),
            _ => false,
        },
        (ReplayMemoryRoute::Global, pb::MemoryOpClass::AsyncCopy) => {
            matches!(slot.mem_role, MemRole::BypassLoad(CacheOpToken::Ldgsts)) && head == "LDGSTS"
        }
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(ReplayError::UnsupportedMemoryPath { arch, class })
    }
}

pub(crate) fn is_static_constant_global_load(opcode_text: &str) -> bool {
    let opcode = opcode_text.split_ascii_whitespace().next().unwrap_or("");
    opcode.split('.').any(|modifier| modifier == "CONSTANT")
        && matches!(opcode_head(opcode_text), "LDG" | "LD" | "LDC" | "LDL")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LocalDirection {
    Load,
    Store,
}

/// Resolve Local traffic only from the joined LDL/STL sidecar role. A missing pc
/// remains neutral so the issue loop retains its dedicated `PcNotInSchedule`
/// diagnostic; every present non-LDL/STL role fails closed.
pub(crate) fn require_known_local_direction(
    schedule: &FunctionSchedule,
    arch: ArchId,
    pc: u32,
) -> Result<Option<LocalDirection>, ReplayError> {
    let Some(slot) = schedule.by_pc.get(&u64::from(pc)) else {
        return Ok(None);
    };
    match (opcode_head(&slot.opcode_text), slot.mem_role) {
        ("LDL", MemRole::EligibleLoad(CacheOpToken::Ldl)) => Ok(Some(LocalDirection::Load)),
        ("STL", MemRole::EligibleLoad(CacheOpToken::Stl)) => Ok(Some(LocalDirection::Store)),
        (opcode, _) => Err(ReplayError::UnknownLocalOpcode {
            arch,
            pc,
            opcode: opcode.to_owned(),
        }),
    }
}

pub(crate) fn opcode_has_qualifier(opcode_text: &str, qualifier: &str) -> bool {
    opcode_text
        .split_whitespace()
        .next()
        .unwrap_or("")
        .split('.')
        .skip(1)
        .any(|part| part == qualifier)
}

/// Return whether a Blackwell bulk-copy spelling is the modeled
/// global-to-shared producer form.  The qualifier order is part of the
/// instruction contract: `UBLKCP.S.G` names a shared destination and global
/// source.  The legacy `UTMASTG` store form remains a supported effective
/// store-side trace path; it is not confused with the load-side direction
/// check below.
pub(crate) fn is_modeled_tma_load_opcode(opcode_text: &str) -> bool {
    let token = opcode_text.split_whitespace().next().unwrap_or("");
    let mut fields = token.split('.');
    match fields.next().unwrap_or("") {
        "UTMALDG" | "UTMASTG" => true,
        "UBLKCP" => matches!(fields.next(), Some("S")) && matches!(fields.next(), Some("G")),
        _ => false,
    }
}

/// Space-coded SASS heads whose storage class is part of the mnemonic rather
/// than inferred from the dynamic record. ATOM and RED are generic, while
/// ATOMG/ATOMS constrain the route; LDGSTS uses its traced Global source. REDG,
/// LDGMC, DSMEM, and shared-matrix families are architecture-gated before any
/// record can borrow a modeled route.
pub(crate) fn static_memory_route(
    opcode_text: &str,
    arch: ArchId,
) -> Result<Option<StaticMemoryRoute>, &'static str> {
    let head = opcode_head(opcode_text);
    if matches!(head, "LDS" | "STS" | "LDSM" | "STSM") && opcode_has_qualifier(opcode_text, "MMIO")
    {
        let family = match head {
            "LDS" => "LDS.MMIO",
            "STS" => "STS.MMIO",
            "LDSM" => "LDSM.MMIO",
            "STSM" => "STSM.MMIO",
            _ => unreachable!("match guard admits only shared-memory heads"),
        };
        return match arch {
            ArchId::Sm86 => Err(family),
            ArchId::Sm120 => Ok(Some(StaticMemoryRoute::UnsupportedClass(
                pb::MemoryOpClass::ClusterDsmem,
            ))),
        };
    }
    match head {
        "LDG" | "STG" => Ok(Some(StaticMemoryRoute::Modeled {
            route: ReplayMemoryRoute::Global,
            class: pb::MemoryOpClass::Global,
        })),
        "LDGSTS" => Ok(Some(StaticMemoryRoute::Modeled {
            route: ReplayMemoryRoute::Global,
            class: pb::MemoryOpClass::AsyncCopy,
        })),
        "ATOM" | "RED" => Ok(Some(StaticMemoryRoute::GenericAtomic {
            class: pb::MemoryOpClass::Atomic,
        })),
        "ATOMG" => Ok(Some(StaticMemoryRoute::Modeled {
            route: ReplayMemoryRoute::Global,
            class: pb::MemoryOpClass::Atomic,
        })),
        "REDG" => match arch {
            ArchId::Sm86 => Err("REDG"),
            ArchId::Sm120 => Ok(Some(StaticMemoryRoute::Modeled {
                route: ReplayMemoryRoute::Global,
                class: pb::MemoryOpClass::Atomic,
            })),
        },
        // Hopper's reducing-load family has no implemented replay semantics on
        // either target. In particular it must never borrow Shared timing when
        // contaminated dynamic metadata claims a Shared route.
        "LDGMC" => Err("LDGMC"),
        "LDS" | "STS" | "LDSM" => Ok(Some(StaticMemoryRoute::Modeled {
            route: ReplayMemoryRoute::Shared,
            class: pb::MemoryOpClass::Shared,
        })),
        "ATOMS" => Ok(Some(StaticMemoryRoute::Modeled {
            route: ReplayMemoryRoute::Shared,
            class: pb::MemoryOpClass::Atomic,
        })),
        "STSM" => match arch {
            ArchId::Sm86 => Err("STSM"),
            ArchId::Sm120 => Ok(Some(StaticMemoryRoute::Modeled {
                route: ReplayMemoryRoute::Shared,
                class: pb::MemoryOpClass::Shared,
            })),
        },
        "STAS" | "REDAS" => match arch {
            ArchId::Sm86 => Err(if head == "STAS" { "STAS" } else { "REDAS" }),
            ArchId::Sm120 => Ok(Some(StaticMemoryRoute::UnsupportedClass(
                pb::MemoryOpClass::ClusterDsmem,
            ))),
        },
        "LDL" | "STL" => Ok(Some(StaticMemoryRoute::Modeled {
            route: ReplayMemoryRoute::Local,
            class: pb::MemoryOpClass::Local,
        })),
        "REDS" => Err("REDS"),
        _ if head.starts_with("ATOM") => Err("ATOM*"),
        _ if head.starts_with("RED") => Err("RED*"),
        _ => Ok(None),
    }
}

pub(crate) fn validate_static_access_kind(
    meta: MemoryAccessMeta,
    schedule: &FunctionSchedule,
    pc: u32,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<(), ReplayError> {
    let Some(got) = meta.access_kind else {
        return Ok(()); // additive legacy field: no dynamic direction to compare
    };
    // Keep a missing pc neutral so the issue loop preserves its specific
    // PcNotInSchedule diagnostic instead of manufacturing a direction claim.
    let Some(slot) = schedule.by_pc.get(&u64::from(pc)) else {
        return Ok(());
    };
    let head = opcode_head(&slot.opcode_text);
    let expected = if head.starts_with("ATOM") || head.starts_with("RED") {
        Some(pb::MemoryOpAccessKind::Atomic)
    } else {
        match head {
            "LDGSTS" => Some(pb::MemoryOpAccessKind::AsyncCopy),
            "STG" | "STL" | "STS" | "STSM" | "STAS" => Some(pb::MemoryOpAccessKind::Store),
            _ => match slot.mem_role {
                MemRole::EligibleLoad(CacheOpToken::Stl) => Some(pb::MemoryOpAccessKind::Store),
                MemRole::EligibleLoad(_) => Some(pb::MemoryOpAccessKind::Load),
                MemRole::BypassLoad(CacheOpToken::Atom) => Some(pb::MemoryOpAccessKind::Atomic),
                MemRole::BypassLoad(CacheOpToken::Ldgsts) => {
                    Some(pb::MemoryOpAccessKind::AsyncCopy)
                }
                MemRole::BypassLoad(_) => Some(pb::MemoryOpAccessKind::Load),
                MemRole::Store => Some(pb::MemoryOpAccessKind::Store),
                MemRole::Reduction => Some(pb::MemoryOpAccessKind::Atomic),
                MemRole::None if is_known_untagged_opcode(&slot.opcode_text) => {
                    Some(pb::MemoryOpAccessKind::Load)
                }
                MemRole::None => None,
            },
        }
    };
    let Some(expected) = expected else {
        return Ok(()); // the existing unknown-opcode gate owns this diagnostic
    };
    if got == expected {
        return Ok(());
    }
    Err(ReplayError::MemoryRoute {
        path: path.to_path_buf(),
        warp_id,
        record_ix,
        class: meta.op_class,
        source: MemoryRouteError::StaticAccessKindMismatch { got, expected },
    })
}

pub(crate) fn preflight_compact_memory<'a>(
    memory: Option<&'a ValidatedMemoryInstruction>,
    warp: &'a ValidatedWarp,
    schedule: &FunctionSchedule,
    pc: u32,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<Option<CompactReplayMemory<'a>>, ReplayError> {
    let Some(memory) = memory else {
        return Ok(None);
    };
    let meta = memory.replay_meta();
    let route = match warp.memory_route(memory) {
        Ok(ValidatedMemoryRoute::Global) => ReplayMemoryRoute::Global,
        Ok(ValidatedMemoryRoute::Local) => ReplayMemoryRoute::Local,
        Ok(ValidatedMemoryRoute::Shared) => ReplayMemoryRoute::Shared,
        Ok(ValidatedMemoryRoute::Unsupported) => ReplayMemoryRoute::Unsupported,
        Err(error) => {
            return Err(ReplayError::MemoryRoute {
                path: path.to_path_buf(),
                warp_id,
                record_ix,
                class: meta.op_class,
                source: compact_memory_route_error(error),
            });
        }
    };
    validate_replay_access_kind(meta, path, warp_id, record_ix)?;
    validate_static_memory_route(meta, route, schedule, pc, path, warp_id, record_ix)?;
    validate_static_access_kind(meta, schedule, pc, path, warp_id, record_ix)?;
    Ok(Some(CompactReplayMemory { memory, route, pc }))
}

pub(crate) fn classify_compact_with_l1(
    preflight: Option<CompactReplayMemory<'_>>,
    warp: &ValidatedWarp,
    schedule: &FunctionSchedule,
    arch: ArchId,
    sectors: &mut Vec<i64>,
    async_copy_cohorts: &mut AsyncCopyCohortReplay,
    block_id: [u32; 3],
    tma_observations: Option<&BTreeMap<TmaObservationKey, TmaObservation>>,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<AccessKind, ReplayError> {
    let Some(preflight) = preflight else {
        return Ok(AccessKind::NonMemory);
    };
    if !preflight.memory.has_effective_lanes() {
        return Ok(AccessKind::PredicatedOffMemory);
    }
    let meta = preflight.memory.replay_meta();
    let sector_plan = warp.sector_plan(preflight.memory);
    if let ValidatedSectorPlan::Error {
        phase: ValidatedSectorErrorPhase::BeforeStaticRole,
        error,
    } = sector_plan
    {
        return Err(deferred_sector_error(error, path, warp_id, record_ix));
    }

    pub(crate) enum StagedHierarchyRole {
        Global(MemRole),
        Local(LocalDirection),
    }
    let staged_role = match (preflight.route, meta.op_class) {
        (ReplayMemoryRoute::Shared, _) => {
            return Ok(compact_shared_kind(preflight.memory, arch));
        }
        (ReplayMemoryRoute::Unsupported, class) => {
            return Err(ReplayError::UnsupportedMemoryPath { arch, class });
        }
        (ReplayMemoryRoute::Local, pb::MemoryOpClass::Local) => {
            let Some(direction) = require_known_local_direction(schedule, arch, preflight.pc)?
            else {
                return Ok(AccessKind::NonMemory);
            };
            StagedHierarchyRole::Local(direction)
        }
        (ReplayMemoryRoute::Local, class) => {
            return Err(ReplayError::UnsupportedMemoryPath { arch, class });
        }
        (ReplayMemoryRoute::Global, pb::MemoryOpClass::Global) => {
            StagedHierarchyRole::Global(require_known_global_role(schedule, arch, preflight.pc)?)
        }
        (ReplayMemoryRoute::Global, class @ pb::MemoryOpClass::Atomic) => {
            require_sm86_special_role(schedule, arch, preflight.pc, preflight.route, class)?;
            StagedHierarchyRole::Global(require_known_global_role(schedule, arch, preflight.pc)?)
        }
        (ReplayMemoryRoute::Global, pb::MemoryOpClass::AsyncCopy) if arch == ArchId::Sm120 => {
            // SM120 async source/destination semantics are admitted only into
            // the explicit alpha cache scenario.  The route remains typed as
            // an async token; it is never downgraded to ordinary global timing.
            StagedHierarchyRole::Global(MemRole::BypassLoad(CacheOpToken::Ldgsts))
        }
        (ReplayMemoryRoute::Global, class @ pb::MemoryOpClass::AsyncCopy) => {
            require_sm86_special_role(schedule, arch, preflight.pc, preflight.route, class)?;
            StagedHierarchyRole::Global(require_known_global_role(schedule, arch, preflight.pc)?)
        }
        (ReplayMemoryRoute::Global, class) => {
            return Err(ReplayError::UnsupportedMemoryPath { arch, class });
        }
    };
    let staged = match sector_plan {
        ValidatedSectorPlan::Sectors(staged) => staged,
        ValidatedSectorPlan::Error {
            phase: ValidatedSectorErrorPhase::AfterStaticRole,
            error,
        } => return Err(deferred_sector_error(error, path, warp_id, record_ix)),
        ValidatedSectorPlan::Error {
            phase: ValidatedSectorErrorPhase::BeforeStaticRole,
            ..
        } => unreachable!("handled above"),
    };
    match staged_role {
        StagedHierarchyRole::Global(_role) if meta.op_class == pb::MemoryOpClass::AsyncCopy => {
            let tma = arch == ArchId::Sm120
                && schedule
                    .issue_slot(u64::from(preflight.pc))
                    .is_some_and(|slot| {
                        matches!(
                            opcode_head(slot.opcode_text()),
                            "UTMALDG" | "UTMASTG" | "UBLKCP"
                        )
                    });
            let l1_policy = if tma {
                LdgstsL1Policy::BypassL1
            } else if arch == ArchId::Sm120 {
                let opcode = schedule
                    .issue_slot(u64::from(preflight.pc))
                    .map(|slot| slot.opcode_text())
                    .unwrap_or("");
                if opcode_has_qualifier(opcode, "BYPASS") {
                    LdgstsL1Policy::BypassL1
                } else {
                    LdgstsL1Policy::CacheAtAllLevels
                }
            } else {
                require_ldgsts_l1_policy(schedule, arch, preflight.pc)?
            };
            let completion_work_override = if arch == ArchId::Sm120 {
                let key = TmaObservationKey {
                    cta: block_id,
                    warp: warp_id,
                    instruction: record_ix,
                    pc: preflight.pc,
                    function: warp.function_unique_id(),
                };
                let payload_bytes = if tma {
                    tma_observations
                        .and_then(|observations| observations.get(&key))
                        .and_then(|observation| observation.bytes)
                        .ok_or(ReplayError::AsyncCopyPayloadMismatch {
                            path: path.to_path_buf(),
                            warp_id,
                            record_ix,
                            source_active: true,
                            sectors: staged.len(),
                            payload_bytes: 0,
                        })?
                } else {
                    warp.effective_data_bytes(preflight.memory)
                        .max(u64::try_from(staged.len()).unwrap_or(0).saturating_mul(32))
                };
                Some(
                    encode_async_copy_payload_work(u32::try_from(payload_bytes).map_err(|_| {
                        ReplayError::AsyncCopyCompletionWorkOutOfRange {
                            path: path.to_path_buf(),
                            warp_id,
                            record_ix,
                            completion_work: payload_bytes,
                            max_completion_work: ASYNC_COPY_COMPLETION_VALUE_MASK,
                        }
                    })?)
                    .ok_or(
                        ReplayError::AsyncCopyCompletionWorkOutOfRange {
                            path: path.to_path_buf(),
                            warp_id,
                            record_ix,
                            completion_work: payload_bytes,
                            max_completion_work: ASYNC_COPY_COMPLETION_VALUE_MASK,
                        },
                    )?,
                )
            } else {
                None
            };
            append_staged_async_copy_sectors(
                arch,
                l1_policy,
                if tma {
                    0
                } else {
                    warp.effective_data_bytes(preflight.memory)
                        .max(u64::try_from(staged.len()).unwrap_or(0).saturating_mul(32))
                },
                warp.ldgsts_destination_topology_128b_32x4(preflight.memory),
                warp.ldgsts_source_cohorts_128b_32x4(preflight.memory),
                warp.ldgsts_source_sector_cohort_masks_128b_32x4(preflight.memory),
                sectors,
                async_copy_cohorts,
                staged,
                path,
                warp_id,
                record_ix,
                completion_work_override,
            )
        }
        StagedHierarchyRole::Global(role) => {
            append_staged_global_sectors(role, sectors, staged, path, warp_id, record_ix)
        }
        StagedHierarchyRole::Local(direction) => {
            append_staged_local_sectors(direction, sectors, staged, path, warp_id, record_ix)
        }
    }
}

pub(crate) fn checked_staged_async_copy_kind(
    l1_policy: LdgstsL1Policy,
    start: usize,
    count: usize,
    completion_work: u32,
) -> Option<AccessKind> {
    let len = u16::try_from(count).ok()?;
    start.checked_add(count)?;
    Some(AccessKind::GlobalAsyncCopy {
        off: start,
        len,
        completion_work,
        l1_policy,
    })
}

pub(crate) fn append_staged_sectors(
    kind: Option<AccessKind>,
    sectors: &mut Vec<i64>,
    sector_scratch: &[i64],
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<AccessKind, ReplayError> {
    let count = sector_scratch.len();
    let overflow = |sectors: usize| ReplayError::SectorCountOverflow {
        path: path.to_path_buf(),
        warp_id,
        record_ix,
        sectors,
    };
    // `AccessKind::off` spans the native `usize` arena domain. Only one
    // record's compact length narrows to `u16`.
    let kind = kind.ok_or_else(|| overflow(count))?;
    sectors
        .try_reserve(count)
        .map_err(|_| ReplayError::SectorExpansion {
            path: path.to_path_buf(),
            warp_id,
            record_ix,
            source: SectorExpansionError::CapacityUnavailable {
                buffer: "warp sector arena",
                additional: count,
            },
        })?;
    sectors.extend_from_slice(sector_scratch);
    Ok(kind)
}

pub(crate) fn append_staged_async_copy_sectors(
    arch: ArchId,
    l1_policy: LdgstsL1Policy,
    payload_bytes: u64,
    destination_topology: Option<LdgstsDestinationTopology>,
    source_cohorts: Option<&[LdgstsSourceCohort]>,
    sector_cohort_masks: Option<&[u32]>,
    sectors: &mut Vec<i64>,
    cohort_replay: &mut AsyncCopyCohortReplay,
    sector_scratch: &[i64],
    path: &Path,
    warp_id: i32,
    record_ix: usize,
    completion_work_override: Option<u32>,
) -> Result<AccessKind, ReplayError> {
    let start = sectors.len();
    let count = sector_scratch.len();
    let source_active = count != 0;
    if completion_work_override.is_none() && source_active != (payload_bytes != 0) {
        return Err(ReplayError::AsyncCopyPayloadMismatch {
            path: path.to_path_buf(),
            warp_id,
            record_ix,
            source_active,
            sectors: count,
            payload_bytes,
        });
    }
    let completion_work = completion_work_override.map_or_else(
        || {
            project_async_copy_completion_work(
                arch,
                l1_policy,
                payload_bytes,
                destination_topology,
                path,
                warp_id,
                record_ix,
            )
        },
        Ok,
    )?;
    let overflow = |sectors: usize| ReplayError::SectorCountOverflow {
        path: path.to_path_buf(),
        warp_id,
        record_ix,
        sectors,
    };
    let kind = checked_staged_async_copy_kind(l1_policy, start, count, completion_work)
        .ok_or_else(|| overflow(count))?;

    let cohort_projection = match (source_cohorts, sector_cohort_masks) {
        (None, None) => None,
        (Some(cohorts), Some(masks))
            if !source_active && cohorts.is_empty() && masks.is_empty() =>
        {
            None
        }
        (Some(cohorts), Some(masks)) => {
            if masks.len() != count {
                return Err(ReplayError::AsyncCopyCohortProjectionMismatch {
                    path: path.to_path_buf(),
                    warp_id,
                    record_ix,
                    field: "source-sector membership count",
                    expected: count,
                    got: masks.len(),
                });
            }
            if cohorts.is_empty() || cohorts.len() > u32::BITS as usize {
                return Err(ReplayError::AsyncCopyCohortProjectionMismatch {
                    path: path.to_path_buf(),
                    warp_id,
                    record_ix,
                    field: "cohort count in u32 membership domain",
                    expected: usize::from(source_active),
                    got: cohorts.len(),
                });
            }
            Some((cohorts, masks))
        }
        (Some(cohorts), None) => {
            return Err(ReplayError::AsyncCopyCohortProjectionMismatch {
                path: path.to_path_buf(),
                warp_id,
                record_ix,
                field: "paired source-sector cohort masks",
                expected: count,
                got: cohorts.len(),
            });
        }
        (None, Some(masks)) => {
            return Err(ReplayError::AsyncCopyCohortProjectionMismatch {
                path: path.to_path_buf(),
                warp_id,
                record_ix,
                field: "paired source cohort descriptors",
                expected: 1,
                got: masks.len(),
            });
        }
    };

    sectors
        .try_reserve(count)
        .map_err(|_| ReplayError::SectorExpansion {
            path: path.to_path_buf(),
            warp_id,
            record_ix,
            source: SectorExpansionError::CapacityUnavailable {
                buffer: "warp sector arena",
                additional: count,
            },
        })?;
    if let Some((cohorts, masks)) = cohort_projection {
        for (buffer, additional, result) in [
            (
                "span",
                1,
                cohort_replay.spans.try_reserve(1).map_err(|_| ()),
            ),
            (
                "descriptor",
                cohorts.len(),
                cohort_replay
                    .cohorts
                    .try_reserve(cohorts.len())
                    .map_err(|_| ()),
            ),
            (
                "membership-mask",
                masks.len(),
                cohort_replay
                    .sector_membership_masks
                    .try_reserve(masks.len())
                    .map_err(|_| ()),
            ),
        ] {
            result.map_err(|_| ReplayError::AsyncCopyCohortCapacityUnavailable {
                path: path.to_path_buf(),
                warp_id,
                record_ix,
                buffer,
                additional,
            })?;
        }
    }

    sectors.extend_from_slice(sector_scratch);
    if let Some((cohorts, masks)) = cohort_projection {
        let cohort_off = cohort_replay.cohorts.len();
        let membership_off = cohort_replay.sector_membership_masks.len();
        cohort_replay
            .cohorts
            .extend(cohorts.iter().map(|cohort| AsyncCopyCohortDescriptor {
                payload_bytes: cohort.payload_bytes,
                destination_lines: cohort.destination_lines,
                bank_wavefronts: cohort.bank_wavefronts,
                bypass_wavefronts: cohort.bypass_wavefronts,
            }));
        cohort_replay
            .sector_membership_masks
            .extend_from_slice(masks);
        cohort_replay.spans.push(AsyncCopyCohortSpan {
            instruction: record_ix,
            sector_off: start,
            sector_len: u16::try_from(count).expect("checked async-copy replay kind length"),
            cohort_off,
            cohort_len: u8::try_from(cohorts.len())
                .expect("u32 membership domain fits the u8 cohort span"),
            membership_off,
        });
    }
    Ok(kind)
}
