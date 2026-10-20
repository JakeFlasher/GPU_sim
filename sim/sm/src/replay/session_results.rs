//! Replay result extraction and launch aggregation; report-only.

use super::*;

#[derive(Default)]
pub(crate) struct JointL2ShadowStats {
    pub(crate) requests: u64,
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    pub(crate) tma_load: [u64; 3],
    pub(crate) ordinary_read: [u64; 3],
    pub(crate) order_unavailable: u64,
    pub(crate) events_dropped: u64,
}

pub(crate) const MAX_JOINT_L2_SHADOW_EVENTS: usize = 2_000_000;

pub(crate) fn joint_l2_ordinary_span(kind: AccessKind) -> Option<(usize, usize, L2OpClass, bool)> {
    match kind {
        AccessKind::GlobalEligible { off, len, .. }
        | AccessKind::GlobalUntagged { off, len }
        | AccessKind::GlobalAsyncCopy {
            off,
            len,
            l1_policy: LdgstsL1Policy::CacheAtAllLevels,
            ..
        } => Some((off, usize::from(len), L2OpClass::Read, true)),
        AccessKind::GlobalBypass {
            off, len, token, ..
        } => Some((
            off,
            usize::from(len),
            if token == CacheOpToken::Atom {
                L2OpClass::WriteAtomic
            } else {
                L2OpClass::Read
            },
            token != CacheOpToken::Atom,
        )),
        AccessKind::GlobalStore { off, len } | AccessKind::GlobalReduction { off, len, .. } => {
            Some((off, usize::from(len), L2OpClass::WriteAtomic, false))
        }
        AccessKind::GlobalAsyncCopy { .. }
        | AccessKind::LocalLoad { .. }
        | AccessKind::LocalStore { .. }
        | AccessKind::Shared { .. }
        | AccessKind::NonMemory
        | AccessKind::PredicatedOffMemory => None,
    }
}

/// Run the same conditional TMA/L2 observer in the sidecar's first-seen
/// stream order.  This is diagnostic only: the lexical keyed-map projection
/// remains the scored/default endpoint because sidecar order is an emission
/// sequence, not a documented engine chronology.
pub(crate) fn observe_tma_l2_shadow_ordered(
    observations: &LaunchObservations,
    config: L2Config,
) -> Result<TmaShadowL2Stats, ReplayError> {
    let iter = if observations.tma_order.is_empty() {
        observations.tma.values().collect::<Vec<_>>()
    } else {
        observations
            .tma_order
            .iter()
            .filter_map(|key| observations.tma.get(key))
            .collect::<Vec<_>>()
    };
    observe_tma_l2_shadow_iter(iter.into_iter(), config)
}

pub(crate) fn observe_tma_l2_shadow_iter<'a>(
    observations: impl Iterator<Item = &'a TmaObservation>,
    config: L2Config,
) -> Result<TmaShadowL2Stats, ReplayError> {
    let mut shadow = L2Model::new(config)?;
    let mut stats = TmaShadowL2Stats::default();
    for observation in observations {
        let (sector_ids, class) = if observation.is_load {
            (&observation.source_sector_ids, L2OpClass::Read)
        } else if observation.is_store {
            (&observation.destination_sector_ids, L2OpClass::WriteAtomic)
        } else {
            continue;
        };
        if sector_ids.is_empty() {
            stats.unavailable =
                stats
                    .unavailable
                    .checked_add(1)
                    .ok_or(IssueError::CounterOverflow {
                        counter: "TMA shadow L2 unavailable address records",
                    })?;
            let lane = if observation.is_load {
                &mut stats.load
            } else {
                &mut stats.store
            };
            lane[3] = lane[3].checked_add(1).ok_or(IssueError::CounterOverflow {
                counter: "TMA shadow operation-class unavailable records",
            })?;
            continue;
        }
        for &sector_id in sector_ids.iter() {
            let line_id = sector_id.div_euclid(4);
            let sector = u8::try_from(sector_id.rem_euclid(4)).map_err(|_| {
                ReplayError::System(SystemError::L2(sim_mem::L2AccessError::SectorOutOfRange {
                    sector: u8::MAX,
                }))
            })?;
            let outcome = shadow
                .access(line_id, sector, class)
                .map_err(|error| ReplayError::System(SystemError::L2(error)))?;
            stats.requests = stats
                .requests
                .checked_add(1)
                .ok_or(IssueError::CounterOverflow {
                    counter: "TMA shadow L2 requests",
                })?;
            let lane = if observation.is_load {
                &mut stats.load
            } else {
                &mut stats.store
            };
            lane[0] = lane[0].checked_add(1).ok_or(IssueError::CounterOverflow {
                counter: "TMA shadow operation-class requests",
            })?;
            if outcome.counted_hit {
                stats.hits = stats
                    .hits
                    .checked_add(1)
                    .ok_or(IssueError::CounterOverflow {
                        counter: "TMA shadow L2 hits",
                    })?;
                lane[1] = lane[1].checked_add(1).ok_or(IssueError::CounterOverflow {
                    counter: "TMA shadow operation-class hits",
                })?;
            } else {
                stats.misses = stats
                    .misses
                    .checked_add(1)
                    .ok_or(IssueError::CounterOverflow {
                        counter: "TMA shadow L2 misses",
                    })?;
                lane[2] = lane[2].checked_add(1).ok_or(IssueError::CounterOverflow {
                    counter: "TMA shadow operation-class misses",
                })?;
            }
        }
    }
    Ok(stats)
}

/// Fast-path evidence produced directly from an admitted SM86 trace. This is
/// deliberately a diagnostic projection: only homogeneous, control-clean,
/// dependency-free scalar regions with no memory role are offered to the
/// model-local interval summary; every other warp is retained as detailed or
/// refused. It does not replace native replay timing or prove silicon behavior.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FastFrontierSummary {
    pub contract: &'static str,
    pub architecture: &'static str,
    pub blocks_seen: u64,
    pub warps_seen: u64,
    pub eligible_warps: u64,
    pub candidate_regions: u64,
    pub certified_regions: u64,
    pub reified_regions: u64,
    pub candidate_instructions: u64,
    pub collapsed_instructions: u64,
    pub exact_execution_steps: u64,
    pub summarized_execution_steps: u64,
    pub macro_dispatches: u64,
    pub maximum_exact_final_cycle: u64,
    pub maximum_summarized_final_cycle: u64,
    pub front_end_static_lines: u64,
    pub front_end_penalty_cycles: u64,
    pub terminal_state_matches: bool,
    /// Full-trace denominator for selective compression. Candidate records
    /// are those inside a scalar segment; reified records are candidates that
    /// could not be collapsed; uncovered records are every prefix/suffix or
    /// noncandidate record left on the exact path.
    pub coverage: FastFrontierCoverage,
    pub refusal_reasons: BTreeMap<String, u64>,
}

pub(crate) fn fast_scalar_refusal_reason(
    warp: &WarpReplay,
    schedule: &FunctionSchedule,
) -> &'static str {
    let mut saw_scalar = false;
    let mut saw_unknown_registers = false;
    let mut saw_control_boundary = false;
    for instruction in &warp.insts {
        if !matches!(instruction.kind, AccessKind::NonMemory) {
            continue;
        }
        let Some(slot) = schedule.issue_slot(u64::from(instruction.pc)) else {
            continue;
        };
        if slot.op_class() != OpClass::Other
            || !matches!(slot.mem_role(), MemRole::None)
            || !matches!(slot.pipeline_class(), PipelineClass::ScalarAlu { .. })
        {
            continue;
        }
        saw_scalar = true;
        if slot.fixed_latency_operands().projected().is_none() {
            saw_unknown_registers = true;
        }
        let control = slot.control();
        if control.wait_mask().bits() != 0
            || control.write_barrier().is_some()
            || control.read_barrier().is_some()
        {
            saw_control_boundary = true;
        }
    }
    if saw_unknown_registers {
        "register_provenance_unavailable"
    } else if saw_control_boundary {
        "control_or_dependency_boundary"
    } else if saw_scalar {
        "register_dependency_or_short_region"
    } else {
        "memory_control_or_non_scalar"
    }
}

pub(crate) fn increment_fast_reason(
    reasons: &mut BTreeMap<String, u64>,
    reason: &str,
) -> Result<(), ReplayError> {
    let value = reasons.entry(reason.to_owned()).or_default();
    *value = value
        .checked_add(1)
        .ok_or_else(|| ReplayError::FastFrontier {
            detail: format!("refusal reason {reason:?} overflows u64"),
        })?;
    Ok(())
}

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

#[cfg(feature = "campaign")]
pub(crate) fn validate_sampling_placement(
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

pub(crate) fn checked_grid_total(grid: (u32, u32, u32)) -> Result<u64, ReplayError> {
    u64::from(grid.0)
        .checked_mul(u64::from(grid.1))
        .and_then(|value| value.checked_mul(u64::from(grid.2)))
        .ok_or(ReplayError::GridSizeOverflow { grid })
}

pub(crate) fn checked_record_count(
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

/// Decode one launch into per-SM block replays on the traced placement.
/// `decode_sectors` selects the L1-mode classification (lane decode +
/// unique-sector expansion) over the flat-mode class check.
pub(crate) fn extract_per_sm_blocks(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedule: &FunctionSchedule,
    placement: PlacementConfig,
    decode_sectors: bool,
    tma_shadow_l2: Option<L2Config>,
) -> Result<ExtractedKernel, ReplayError> {
    let expected_fuid = schedule.function_unique_id;
    let shared_memory_residency = placement.shared_memory_residency(loc.kernel())?;
    let block = loc
        .kernel()
        .block_dim
        .as_ref()
        .filter(|dim| dim.x > 0 && dim.y > 0 && dim.z > 0)
        .map(|dim| (dim.x as u32, dim.y as u32, dim.z as u32));
    let expected_block_warps = block.map(checked_block_warps).transpose()?;
    #[cfg(feature = "campaign")]
    let validate_warp_roster = expected_block_warps.is_some()
        && td
            .sampling_sidecar()
            .is_none_or(|sidecar| Some(sidecar.original_block()) == block);
    #[cfg(not(feature = "campaign"))]
    let validate_warp_roster = expected_block_warps.is_some();
    // Physical slot validation is part of the traced launch envelope.  Do
    // not obtain it from `issue_hypotheses()`: SM120 intentionally has no
    // admitted scheduler hypothesis, while the 5090 runtime card does carry
    // a measured max-resident-warp limit.
    let max_resident_warps = placement.max_resident_warps_per_sm;
    let cluster = normalized_cluster_dim(loc.kernel(), schedule.arch)?;
    // sm_id → blocks (launch index kept for in-SM ordering).
    let mut per_sm_blocks: BTreeMap<i32, Vec<BlockReplay>> = BTreeMap::new();
    let mut seen_indices: BTreeSet<i64> = BTreeSet::new();
    let mut tma_dynamic_keys_seen: BTreeSet<TmaObservationKey> = BTreeSet::new();
    let collect_joint_l2 = tma_shadow_l2.is_some();
    let mut joint_l2_events = Vec::new();
    let mut joint_l2_order_unavailable = 0_u64;
    let mut joint_l2_events_dropped = 0_u64;
    let mut joint_l2_sequence = 0_u64;
    let mut traced_sm_by_launch_index: BTreeMap<i64, i32> = BTreeMap::new();
    let mut records_seen: u64 = 0;
    let mut instruction_sectors = BTreeSet::new();
    let mut constant_sectors = BTreeSet::new();
    let mut instruction_lines = BTreeSet::new();
    let static_instruction_lines = schedule
        .by_pc
        .keys()
        .map(|pc| pc / 128)
        .collect::<BTreeSet<_>>();

    let mut instruction_sectors_by_sm: BTreeMap<i32, BTreeSet<(i32, u32)>> = BTreeMap::new();
    let launch_resources = (loc.kernel().number_of_registers > 0
        || loc.kernel().size_shared_memory > 0
        || loc.kernel().shared_memory_static_bytes.is_some()
        || loc.kernel().shared_memory_dynamic_bytes.is_some()
        || loc.kernel().shared_memory_carveout_preference_pct.is_some()
        || loc.kernel().driver_occupancy_limit_blocks.is_some())
    .then(|| crate::metrics::LaunchResources {
        block_threads: loc
            .kernel()
            .block_dim
            .as_ref()
            .map(|dim| {
                dim.x
                    .max(0)
                    .saturating_mul(dim.y.max(0))
                    .saturating_mul(dim.z.max(0)) as u32
            })
            .unwrap_or(0),
        grid_blocks: loc
            .kernel()
            .grid_dim
            .as_ref()
            .map(|dim| {
                u64::from(dim.x.max(0) as u32)
                    .saturating_mul(u64::from(dim.y.max(0) as u32))
                    .saturating_mul(u64::from(dim.z.max(0) as u32))
            })
            .unwrap_or(0),
        // TraceDir validated nonnegative counts and the split/total sum.
        registers_per_thread: loc.kernel().number_of_registers as u32,
        shared_memory_bytes_per_block: loc.kernel().size_shared_memory as u32,
        static_shared_memory_bytes: loc.kernel().shared_memory_static_bytes,
        dynamic_shared_memory_bytes: loc.kernel().shared_memory_dynamic_bytes,
        shared_memory_carveout_preference_pct: loc.kernel().shared_memory_carveout_preference_pct,
        driver_occupancy_limit_blocks: loc.kernel().driver_occupancy_limit_blocks,
    });
    let mut trace_metrics = TraceMetrics {
        launch_resources,
        // SM120 flat mixed-async launches use the compact classifier but also
        // retain every observed global/local source span. Opaque TMA payload
        // bytes are reported separately and do not manufacture sector ids.
        sectors_decoded: decode_sectors || schedule.arch == ArchId::Sm120,
        ..TraceMetrics::default()
    };
    trace_metrics.bind_conditional_locality_arch(schedule.arch);
    let sidecar_join_started = Instant::now();
    let observations = if schedule.arch == ArchId::Sm120 {
        td.async_sidecars()?
            .launch(loc.device_id(), loc.stream_id(), loc.kernel().id)
    } else {
        None
    };
    trace_metrics.async_relations = observations
        .filter(|o| !o.tma.is_empty() || !o.sync.is_empty())
        .map(|o| o.relations(cluster.into()));
    trace_metrics.async_obligations = observations
        .filter(|o| !o.tma.is_empty() || !o.sync.is_empty())
        .map(|o| sim_trace::obligations::ObligationFrontier::from_launch(o, cluster.into()).0);
    if let Some(observations) = observations {
        let (
            sum_record_unique_sectors,
            sum_record_unique_lines,
            launch_unique_sectors,
            launch_unique_lines,
            intra_record_duplicate_refs,
            load_sum_sectors,
            store_sum_sectors,
            load_sum_lines,
            store_sum_lines,
            load_unique_sectors,
            store_unique_sectors,
            cross_direction_sectors,
            cross_direction_lines,
        ) = tma_sidecar_reuse_summary(observations);
        trace_metrics.observe_tma_sidecar_reuse(
            sum_record_unique_sectors,
            sum_record_unique_lines,
            launch_unique_sectors,
            launch_unique_lines,
            intra_record_duplicate_refs,
        )?;
        trace_metrics.observe_tma_directional_reuse(
            load_sum_sectors,
            store_sum_sectors,
            load_sum_lines,
            store_sum_lines,
            load_unique_sectors,
            store_unique_sectors,
            cross_direction_sectors,
            cross_direction_lines,
        )?;
        if let Some(relations) = trace_metrics.async_relations.as_ref() {
            let (pairs, invalid, lower_sum, upper_sum) =
                async_clock_sandwich_summary(observations, relations);
            trace_metrics.observe_async_clock_sandwich(pairs, invalid, lower_sum, upper_sum)?;
        }
    }
    if let (Some(observations), Some(l2)) = (observations, tma_shadow_l2) {
        let stats = observe_tma_l2_shadow(&observations.tma, l2)?;
        trace_metrics.observe_tma_shadow_l2(
            stats.requests,
            stats.hits,
            stats.misses,
            stats.unavailable,
            stats.load,
            stats.store,
        )?;
        let ordered = observe_tma_l2_shadow_ordered(observations, l2)?;
        trace_metrics.observe_tma_shadow_l2_ordered(
            ordered.requests,
            ordered.hits,
            ordered.misses,
            ordered.unavailable,
            ordered.load,
            ordered.store,
        )?;
    }
    let tma_observations = observations.map(|o| &o.tma);
    let sync_observations = observations.map(|o| &o.sync);
    trace_metrics.sidecar_join_wall_ns =
        u64::try_from(sidecar_join_started.elapsed().as_nanos()).unwrap_or(u64::MAX);

    for path in td.threadblock_paths(loc)? {
        let tb = td.validated_threadblock(&path)?;
        let launch_index = tb.launch_index();
        if !seen_indices.insert(launch_index) {
            return Err(ReplayError::DuplicateLaunchIndex {
                index: launch_index,
            });
        }

        let admitted_block_warps = expected_block_warps.filter(|_| validate_warp_roster);
        if let Some(expected_block_warps) = admitted_block_warps {
            let actual_block_warps =
                checked_collection_len("threadblock warp roster", tb.warps().len())?;
            if actual_block_warps != expected_block_warps {
                return Err(ReplayError::WarpRosterCardinalityMismatch {
                    path,
                    expected: expected_block_warps,
                    actual: actual_block_warps,
                });
            }
        }
        let mut block_sm: Option<i32> = None;
        let block_id = tb.block_id();
        let mut seen_warp_slots = BTreeSet::new();
        let mut warps = Vec::with_capacity(tb.warps().len());
        for warp in tb.warps() {
            let warp_id = warp.map_key();
            if admitted_block_warps.is_some_and(|expected_block_warps| {
                warp_id < 0 || u64::try_from(warp_id).unwrap_or(u64::MAX) >= expected_block_warps
            }) {
                return Err(ReplayError::WarpIdOutOfRange {
                    path,
                    warp_id,
                    expected_warps: admitted_block_warps.expect("checked as present"),
                });
            }
            if warp.warp_id() != warp_id {
                return Err(ReplayError::WarpIdentityMismatch {
                    path,
                    map_key: warp_id,
                    warp_id: warp.warp_id(),
                });
            }
            let mut warp_slot: Option<i32> = None;
            let mut insts = Vec::with_capacity(warp.instructions().len());
            let mut sectors: Vec<i64> = Vec::new();
            let mut async_copy_cohorts = AsyncCopyCohortReplay::default();
            let mut branch_successors = crate::metrics::BranchSuccessors::default();
            let mut previous_control_flow: Option<(i32, u32, u32)> = None;
            trace_metrics.begin_warp_memory_stream();
            for (record_ix, inst) in warp.instructions().enumerate() {
                let record_warp_id = inst.warpid_tb();
                if record_warp_id != warp_id {
                    return Err(ReplayError::InstructionWarpIdMismatch {
                        path,
                        warp_id,
                        record_ix,
                        record_warp_id,
                    });
                }
                records_seen = checked_record_count(records_seen, &path, warp_id, record_ix)?;
                let constant_addresses = inst.constant_memory_addresses();
                // Only active-lane address vectors contribute to the demand
                // interval. Bank/offset metadata without addresses can be a
                // predicated-off or incomplete record and is retained as
                // typed metadata, never fabricated into traffic.
                let constant_bank = inst.constant_memory_bank_offset().map(|(bank, _)| bank);
                trace_metrics.observe_constant_memory(constant_addresses, constant_bank)?;
                if constant_addresses.is_empty() && inst.constant_memory_metadata_present() {
                    trace_metrics.observe_constant_metadata_only()?;
                }
                let preflight = preflight_compact_memory(
                    inst.memory(),
                    warp,
                    schedule,
                    inst.pc(),
                    &path,
                    warp_id,
                    record_ix,
                )?;
                if inst.function_unique_id() != expected_fuid {
                    return Err(ReplayError::MultiFunctionUnsupported {
                        path,
                        warp_id,
                        record_ix,
                        expected: expected_fuid,
                        found: inst.function_unique_id(),
                    });
                }
                instruction_sectors.insert((inst.function_unique_id(), inst.pc() / 32));
                if let Some(slot) = schedule.issue_slot(u64::from(inst.pc())) {
                    constant_sectors.extend(static_constant_sectors(slot.opcode_text()));
                }
                instruction_lines.insert((inst.function_unique_id(), inst.pc() / 128));

                let is_tma_opcode = schedule.arch == ArchId::Sm120
                    && schedule
                        .issue_slot(u64::from(inst.pc()))
                        .is_some_and(|slot| {
                            matches!(
                                opcode_head(slot.opcode_text()),
                                "UTMALDG" | "UTMASTG" | "UBLKCP"
                            )
                        });
                if is_tma_opcode {
                    tma_dynamic_keys_seen.insert(TmaObservationKey {
                        cta: block_id,
                        warp: warp_id,
                        instruction: record_ix,
                        pc: inst.pc(),
                        function: inst.function_unique_id(),
                    });
                }
                if is_tma_opcode && (decode_sectors || preflight.is_none()) {
                    let key = TmaObservationKey {
                        cta: block_id,
                        warp: warp_id,
                        instruction: record_ix,
                        pc: inst.pc(),
                        function: inst.function_unique_id(),
                    };
                    trace_metrics
                        .observe_tma_sidecar_payload(tma_observations.and_then(|o| o.get(&key)))?;
                }
                if schedule.arch == ArchId::Sm120
                    && schedule
                        .issue_slot(u64::from(inst.pc()))
                        .is_some_and(|slot| is_sync_sidecar_opcode(slot.opcode_text()))
                {
                    let key = TmaObservationKey {
                        cta: block_id,
                        warp: warp_id,
                        instruction: record_ix,
                        pc: inst.pc(),
                        function: inst.function_unique_id(),
                    };
                    trace_metrics
                        .observe_sync_sidecar(sync_observations.and_then(|o| o.get(&key)))?;
                }
                if let Some(sm) = inst.sm_id() {
                    match block_sm {
                        None => block_sm = Some(sm),
                        Some(first) if first != sm => {
                            return Err(ReplayError::SmIdInconsistent {
                                path,
                                first,
                                second: sm,
                            });
                        }
                        Some(_) => {}
                    }
                }
                if let Some(slot) = inst.warpid_sm() {
                    match warp_slot {
                        None => warp_slot = Some(slot),
                        Some(first) if first != slot => {
                            return Err(ReplayError::WarpSlotInconsistent {
                                path,
                                warp_id,
                                first,
                                second: slot,
                            });
                        }
                        Some(_) => {}
                    }
                }
                let partial_store_sectors = preflight
                    .as_ref()
                    .map_or(0, |memory| warp.partial_store_sectors(memory.memory));
                let local_interleaved_sectors = preflight.as_ref().map_or(&[][..], |memory| {
                    warp.local_interleaved_sector_plan(memory.memory)
                });
                let local_interleaved_dynamic = preflight
                    .as_ref()
                    .is_some_and(|memory| memory.memory.local_interleaved_is_dynamic());
                let kind = if decode_sectors {
                    classify_compact_with_l1(
                        preflight,
                        warp,
                        schedule,
                        schedule.arch,
                        &mut sectors,
                        &mut async_copy_cohorts,
                        block_id,
                        tma_observations,
                        &path,
                        warp_id,
                        record_ix,
                    )?
                } else {
                    classify_compact_flat(
                        preflight,
                        warp,
                        schedule,
                        schedule.arch,
                        inst.pc(),
                        &path,
                        warp_id,
                        record_ix,
                        &mut sectors,
                        block_id,
                        tma_observations,
                    )?
                };
                if matches!(
                    kind,
                    AccessKind::GlobalStore { .. } | AccessKind::LocalStore { .. }
                ) {
                    trace_metrics.partial_store_sectors = trace_metrics
                        .partial_store_sectors
                        .checked_add(partial_store_sectors)
                        .ok_or(IssueError::CounterOverflow {
                            counter: "partial store sectors",
                        })?;
                }
                if collect_joint_l2 {
                    let order = inst.instrumentation_order();
                    if let Some((off, len, class, ordinary_read)) = joint_l2_ordinary_span(kind) {
                        if len != 0 {
                            let end = off.saturating_add(len);
                            if let Some(span) = sectors.get(off..end) {
                                if let Some(order) = order {
                                    for &sector_id in span {
                                        if joint_l2_events.len() >= MAX_JOINT_L2_SHADOW_EVENTS {
                                            joint_l2_events_dropped =
                                                joint_l2_events_dropped.saturating_add(1);
                                            break;
                                        }
                                        joint_l2_events.push(JointL2ShadowEvent {
                                            order,
                                            sequence: joint_l2_sequence,
                                            sector_id,
                                            class,
                                            tma_load: false,
                                            ordinary_read,
                                        });
                                        joint_l2_sequence = joint_l2_sequence.saturating_add(1);
                                    }
                                } else {
                                    joint_l2_order_unavailable =
                                        joint_l2_order_unavailable.saturating_add(1);
                                }
                            }
                        }
                    }
                    if is_tma_opcode {
                        let key = TmaObservationKey {
                            cta: block_id,
                            warp: warp_id,
                            instruction: record_ix,
                            pc: inst.pc(),
                            function: inst.function_unique_id(),
                        };
                        if let Some(observation) =
                            tma_observations.and_then(|records| records.get(&key))
                        {
                            let ids = if observation.is_load {
                                &observation.source_sector_ids[..]
                            } else if observation.is_store {
                                &observation.destination_sector_ids[..]
                            } else {
                                &[][..]
                            };
                            if ids.is_empty() {
                                joint_l2_order_unavailable =
                                    joint_l2_order_unavailable.saturating_add(1);
                            } else if let Some(order) = order {
                                let class = if observation.is_load {
                                    L2OpClass::Read
                                } else {
                                    L2OpClass::WriteAtomic
                                };
                                for &sector_id in ids.iter() {
                                    if joint_l2_events.len() >= MAX_JOINT_L2_SHADOW_EVENTS {
                                        joint_l2_events_dropped =
                                            joint_l2_events_dropped.saturating_add(1);
                                        break;
                                    }
                                    joint_l2_events.push(JointL2ShadowEvent {
                                        order,
                                        sequence: joint_l2_sequence,
                                        sector_id,
                                        class,
                                        tma_load: observation.is_load,
                                        ordinary_read: false,
                                    });
                                    joint_l2_sequence = joint_l2_sequence.saturating_add(1);
                                }
                            } else {
                                joint_l2_order_unavailable =
                                    joint_l2_order_unavailable.saturating_add(1);
                            }
                        }
                    }
                }
                if schedule.arch == ArchId::Sm86
                    && schedule
                        .issue_slot(u64::from(inst.pc()))
                        .is_some_and(|slot| is_static_constant_global_load(slot.opcode_text()))
                {
                    let span = match kind {
                        AccessKind::GlobalEligible { off, len, .. }
                        | AccessKind::GlobalBypass { off, len, .. }
                        | AccessKind::GlobalUntagged { off, len, .. } => Some((off, len)),
                        _ => None,
                    };
                    if let Some((off, len)) = span {
                        trace_metrics.observe_constant_global_load(
                            block_id,
                            warp_id,
                            &sectors,
                            off,
                            usize::from(len),
                            schedule
                                .issue_slot(u64::from(inst.pc()))
                                .map(|slot| slot.opcode_text())
                                .unwrap_or(""),
                            preflight
                                .as_ref()
                                .is_some_and(|memory| memory.memory.warp_uniform_address()),
                        )?;
                    }
                }
                if let Some((off, len)) = match kind {
                    AccessKind::GlobalBypass {
                        token: CacheOpToken::Atom,
                        off,
                        len,
                        ..
                    }
                    | AccessKind::GlobalEligible {
                        token: CacheOpToken::Atom,
                        off,
                        len,
                        ..
                    } => Some((off, len)),
                    _ => None,
                } {
                    trace_metrics.observe_atomic_conflict(&sectors, off, usize::from(len))?;
                }
                if matches!(
                    kind,
                    AccessKind::LocalLoad { .. } | AccessKind::LocalStore { .. }
                ) {
                    trace_metrics
                        .observe_local_width(inst.data_width(), inst.predicate_on_lanes())?;
                    trace_metrics.observe_local_payload_sectors(
                        kind,
                        inst.data_width(),
                        inst.predicate_on_lanes(),
                    )?;
                    if schedule.arch == ArchId::Sm86 {
                        if let Some(memory) = inst.memory() {
                            trace_metrics.observe_local_canonical_sectors(
                                kind,
                                memory.local_canonical_sectors(),
                            )?;
                            trace_metrics.observe_local_frame256_sectors(
                                kind,
                                memory.local_frame256_sectors(),
                            )?;
                        }
                    }
                }

                let active_mask_present = inst.active_mask_present();
                let predicate_mask_present = inst.predicate_mask_present();
                let (opcode, pipeline) = schedule
                    .issue_slot(u64::from(inst.pc()))
                    .map_or(("UNKNOWN", PipelineClass::DispatchOnly), |slot| {
                        (slot.opcode_text(), slot.pipeline_class())
                    });
                trace_metrics.observe_with_mask_presence_arch(
                    opcode,
                    inst.active_lanes(),
                    inst.predicate_on_lanes(),
                    active_mask_present,
                    predicate_mask_present,
                    kind,
                    pipeline,
                    &sectors,
                    schedule.arch,
                )?;
                // Passive reuse-interval evidence is collected from the same
                // decoded, coalesced span but never feeds cache state, DRAM,
                // or issue timing. The observer resets at each warp because
                // cross-warp arrival order is not present in the trace.
                trace_metrics.observe_reuse_memory(kind, &sectors)?;
                let opcode_head = opcode.split('.').next().unwrap_or(opcode);
                let narrow_store = opcode
                    .split('.')
                    .any(|modifier| matches!(modifier, "U8" | "S8" | "U16" | "S16"));
                let full_sector_store = matches!(opcode_head, "STG" | "STL" | "ST")
                    && !narrow_store
                    && active_mask_present
                    && predicate_mask_present
                    && inst.active_lanes() == 32
                    && inst.predicate_on_lanes() == 32
                    && inst.data_width().is_some_and(|width| width >= 4);
                trace_metrics.observe_store_to_load_forwarding(kind, &sectors, full_sector_store);
                trace_metrics.observe_local_interleaved(
                    kind,
                    local_interleaved_sectors,
                    local_interleaved_dynamic,
                )?;
                if let Some(slot) = schedule.issue_slot(u64::from(inst.pc())) {
                    trace_metrics
                        .operand_bank_pressure
                        .observe(slot.fixed_latency_operands(), schedule.arch)?;
                }
                if active_mask_present && predicate_mask_present {
                    trace_metrics.observe_coherence(
                        inst.sm_id(),
                        inst.predicate_on_lanes(),
                        kind,
                        &sectors,
                        inst.instrumentation_order(),
                    )?;
                }
                if active_mask_present {
                    if let Some((from_function, from_pc, from_active)) = previous_control_flow {
                        trace_metrics.observe_control_flow(
                            from_function,
                            from_pc,
                            from_active,
                            inst.function_unique_id(),
                            inst.pc(),
                            inst.active_mask(),
                        )?;
                    }
                    previous_control_flow =
                        Some((inst.function_unique_id(), inst.pc(), inst.active_mask()));
                    branch_successors.observe(
                        &mut trace_metrics.branch_targets,
                        inst.function_unique_id(),
                        inst.pc(),
                        inst.active_mask(),
                        opcode,
                    )?;
                } else {
                    // Do not connect CFG edges or resolve branch successors
                    // through a record whose lane population is absent.
                    previous_control_flow = None;
                }
                if let Some(order) = inst.instrumentation_order() {
                    async_copy_cohorts
                        .instrumentation_order
                        .push(InstrumentationOrderAnchor {
                            instruction: record_ix,
                            order,
                        });
                }
                insts.push(ReplayInst {
                    pc: inst.pc(),
                    kind,
                    active_lane_mask: if inst.active_mask_present() {
                        inst.active_mask()
                    } else {
                        u32::MAX
                    },
                    lane_mask: if inst.predicate_mask_present() {
                        inst.predicate_on_mask()
                    } else if inst.active_mask_present() {
                        inst.active_mask()
                    } else {
                        u32::MAX
                    },
                });
            }
            branch_successors.finish(&mut trace_metrics.branch_targets)?;
            // Fail-closed traced placement: every roster warp must expose its
            // physical slot. An empty warp has no placement witness and cannot be
            // used to prove either the architecture slot bound or modeled CTA
            // co-residency.
            let Some(warp_slot) = warp_slot else {
                return Err(ReplayError::WarpSlotMissing { path, warp_id });
            };
            if warp_slot < 0 || u32::try_from(warp_slot).unwrap_or(u32::MAX) >= max_resident_warps {
                return Err(ReplayError::WarpSlotOutOfRange {
                    path,
                    warp_id,
                    warp_slot,
                    arch: schedule.arch,
                    max_resident_warps,
                });
            }
            if !seen_warp_slots.insert(warp_slot) {
                return Err(ReplayError::DuplicateWarpSlot { path, warp_slot });
            }
            warps.push(WarpReplay {
                warp_id,
                warp_slot,
                insts,
                sectors,
                async_copy_cohorts: (!async_copy_cohorts.spans.is_empty()
                    || !async_copy_cohorts.instrumentation_order.is_empty())
                .then(|| Box::new(async_copy_cohorts)),
            });
        }
        let Some(sm) = block_sm else {
            return Err(ReplayError::SmIdMissing { path });
        };
        placement.require_traced_sm(&path, sm)?;
        traced_sm_by_launch_index.insert(launch_index, sm);
        let block_replay = BlockReplay {
            launch_index,
            warps,
        };
        instruction_sectors_by_sm
            .entry(sm)
            .or_default()
            .extend(per_sm_instruction_sector_set(
                std::slice::from_ref(&block_replay),
                expected_fuid,
            ));
        per_sm_blocks.entry(sm).or_default().push(block_replay);
    }
    // Completeness gate (fail-closed) on the path replay ACTUALLY consumes: a
    // subset trace (coverage < grid) is admissible ONLY with a valid
    // trace_sampling.json sidecar whose original_grid matches the launch record
    // (gridDim is never rewritten) and whose captured record count matches
    // records_seen. Absent/non-sampling sidecars never waive full coverage.
    let (gx, gy, gz) = kernel_grid(loc.kernel())?;
    let grid = (gx, gy, gz);
    let grid_total = checked_grid_total(grid)?;
    let seen = checked_collection_len("replayed block membership", seen_indices.len())?;
    let launch = ReplayLaunchCoverageBinding {
        device_id: loc.device_id(),
        stream_id: loc.stream_id(),
        kernel_id: loc.kernel().id,
        function_unique_id: loc.kernel().function_unique_id,
        kernel_mangled_name: loc.kernel().name.clone(),
        grid,
        block,
        cluster,
    };
    #[cfg(not(feature = "campaign"))]
    let coverage = {
        if seen != grid_total {
            return Err(ReplayError::IncompleteTrace { seen, grid_total });
        }
        ExtractedCoverage::Full(FullLaunchCoverage {
            launch,
            blocks_observed: seen,
            source: FullCoverageSource::UnsampledComplete,
        })
    };
    #[cfg(feature = "campaign")]
    let coverage = match td.sampling_sidecar() {
        None => {
            if seen != grid_total {
                return Err(ReplayError::IncompleteTrace { seen, grid_total });
            }
            ExtractedCoverage::Full(FullLaunchCoverage {
                launch,
                blocks_observed: seen,
                source: FullCoverageSource::UnsampledComplete,
            })
        }
        Some(sc) => {
            let launch_block = launch.block.ok_or(ReplayError::BlockDimMissing)?;
            if sc.kernel_mangled_name() != loc.kernel().name.as_str() {
                return Err(ReplayError::SamplingKernelMismatch {
                    launch: loc.kernel().name.clone(),
                    sidecar: sc.kernel_mangled_name().to_owned(),
                });
            }
            if sc.original_grid() != grid {
                return Err(ReplayError::GridDimRewritten {
                    launch: grid,
                    sidecar: sc.original_grid(),
                });
            }
            if sc.original_block() != launch_block {
                return Err(ReplayError::BlockDimRewritten {
                    launch: launch_block,
                    sidecar: sc.original_block(),
                });
            }
            let coverage = match sc.method() {
                TraceSamplingMethod::WaveSegmentStratifiedSubset(provenance) => {
                    if provenance.selected_block_ids() != &seen_indices {
                        let declared = checked_collection_len(
                            "trace_sampling selected block membership",
                            provenance.selected_block_ids().len(),
                        )?;
                        return Err(ReplayError::SelectedBlockMembershipMismatch {
                            seen,
                            declared,
                        });
                    }
                    ExtractedCoverage::Sampled(SampledLaunchCoverage {
                        launch,
                        population_blocks: grid_total,
                        blocks_observed: seen,
                        provenance: provenance.clone(),
                    })
                }
                TraceSamplingMethod::LosslessCompressedFull(provenance) => {
                    if seen != grid_total {
                        return Err(ReplayError::IncompleteTrace { seen, grid_total });
                    }
                    ExtractedCoverage::Full(FullLaunchCoverage {
                        launch,
                        blocks_observed: seen,
                        source: FullCoverageSource::LosslessCompressedFull(provenance.clone()),
                    })
                }
            };
            validate_sampling_placement(sc, &traced_sm_by_launch_index, placement)?;
            if records_seen != sc.captured_records() {
                return Err(ReplayError::RecordsCapturedMismatch {
                    records_seen,
                    captured: sc.captured_records(),
                });
            }
            coverage
        }
    };
    for blocks in per_sm_blocks.values_mut() {
        blocks.sort_by_key(|b| b.launch_index);
    }
    trace_metrics.finish_coherence()?;
    if let Some(l2) = tma_shadow_l2 {
        let joint = observe_joint_l2_shadow(
            joint_l2_events,
            l2,
            joint_l2_order_unavailable,
            joint_l2_events_dropped,
        )?;
        trace_metrics.observe_tma_joint_l2(
            joint.requests,
            joint.hits,
            joint.misses,
            joint.tma_load,
            joint.ordinary_read,
            joint.order_unavailable,
            joint.events_dropped,
        )?;
    }
    if let Some(observations) = observations {
        let mut matched = 0_u64;
        let mut unmatched = 0_u64;
        for (key, observation) in &observations.tma {
            let has_address = (observation.is_load && !observation.source_sector_ids.is_empty())
                || (observation.is_store && !observation.destination_sector_ids.is_empty());
            if !has_address {
                continue;
            }
            if tma_dynamic_keys_seen.contains(key) {
                matched = matched.checked_add(1).ok_or(IssueError::CounterOverflow {
                    counter: "TMA shadow dynamic identity matches",
                })?;
            } else {
                unmatched = unmatched
                    .checked_add(1)
                    .ok_or(IssueError::CounterOverflow {
                        counter: "TMA shadow dynamic identity gaps",
                    })?;
            }
        }
        trace_metrics.observe_tma_shadow_dynamic_identity(matched, unmatched)?;
    }
    Ok(ExtractedKernel {
        per_sm_blocks,
        per_sm_instruction_read_sectors: instruction_sectors_by_sm
            .into_iter()
            .map(|(sm, sectors)| {
                u64::try_from(sectors.len())
                    .map(|count| (sm, count))
                    .map_err(|_| IssueError::CounterOverflow {
                        counter: "per-SM compulsory instruction-read sector cardinality",
                    })
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?,
        records_seen,
        coverage,
        shared_memory_residency,
        front_end: FrontEndCounters {
            compulsory_instruction_read_sectors: u64::try_from(instruction_sectors.len()).map_err(
                |_| IssueError::CounterOverflow {
                    counter: "compulsory instruction-read sector cardinality",
                },
            )?,
            compulsory_constant_read_sectors: u64::try_from(constant_sectors.len()).map_err(
                |_| IssueError::CounterOverflow {
                    counter: "compulsory constant-read sector cardinality",
                },
            )?,
            compulsory_instruction_line_read_sectors: u64::try_from(instruction_lines.len())
                .map_err(|_| IssueError::CounterOverflow {
                    counter: "compulsory instruction-read line cardinality",
                })?
                .checked_mul(4)
                .ok_or(IssueError::CounterOverflow {
                    counter: "compulsory instruction-read line sector cardinality",
                })?,
            static_instruction_line_read_sectors: u64::try_from(static_instruction_lines.len())
                .map_err(|_| IssueError::CounterOverflow {
                    counter: "static instruction-read line cardinality",
                })?
                .checked_mul(4)
                .ok_or(IssueError::CounterOverflow {
                    counter: "static instruction-read line sector cardinality",
                })?,
        },
        trace_metrics,
    })
}

pub(crate) fn checked_staged_local_kind(
    direction: LocalDirection,
    start: usize,
    count: usize,
) -> Option<AccessKind> {
    let len = u16::try_from(count).ok()?;
    start.checked_add(count)?;
    Some(match direction {
        LocalDirection::Load => AccessKind::LocalLoad { off: start, len },
        LocalDirection::Store => AccessKind::LocalStore { off: start, len },
    })
}

pub(crate) fn append_staged_global_sectors(
    role: MemRole,
    sectors: &mut Vec<i64>,
    sector_scratch: &[i64],
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<AccessKind, ReplayError> {
    let start = sectors.len();
    let l2_requests = if matches!(
        role,
        MemRole::BypassLoad(CacheOpToken::Atom) | MemRole::Reduction
    ) {
        atomic_l2_request_count_128b(sector_scratch)
    } else {
        0
    };
    append_staged_sectors(
        checked_staged_global_kind(role, start, sector_scratch.len(), l2_requests),
        sectors,
        sector_scratch,
        path,
        warp_id,
        record_ix,
    )
}

pub(crate) fn append_staged_local_sectors(
    direction: LocalDirection,
    sectors: &mut Vec<i64>,
    sector_scratch: &[i64],
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<AccessKind, ReplayError> {
    let start = sectors.len();
    append_staged_sectors(
        checked_staged_local_kind(direction, start, sector_scratch.len()),
        sectors,
        sector_scratch,
        path,
        warp_id,
        record_ix,
    )
}
