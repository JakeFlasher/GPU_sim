//! Trace-stream replay session state; drives active warp streams.

use super::*;

pub(crate) fn append_bounded_async_decision_snapshots(
    metrics: &mut TraceMetrics,
    snapshots: &[crate::AsyncDecisionSnapshot],
    already_dropped: u64,
) -> Result<(), ReplayError> {
    metrics.async_decision_snapshots_dropped = metrics
        .async_decision_snapshots_dropped
        .checked_add(already_dropped)
        .ok_or(IssueError::CounterOverflow {
            counter: "launch async decision snapshots dropped",
        })?;
    let available =
        MAX_ASYNC_DECISION_SNAPSHOTS.saturating_sub(metrics.async_decision_snapshots.len());
    let kept = available.min(snapshots.len());
    metrics
        .async_decision_snapshots
        .extend_from_slice(&snapshots[..kept]);
    metrics.async_decision_snapshots_dropped = metrics
        .async_decision_snapshots_dropped
        .checked_add(u64::try_from(snapshots.len() - kept).map_err(|_| {
            IssueError::CounterOverflow {
                counter: "launch async decision snapshot omission",
            }
        })?)
        .ok_or(IssueError::CounterOverflow {
            counter: "launch async decision snapshots dropped",
        })?;
    Ok(())
}

pub(crate) fn async_clock_sandwich_summary(
    observations: &LaunchObservations,
    relations: &sim_trace::async_sidecars::AsyncRelations,
) -> (u64, u64, u64, u64) {
    let mut pairs = 0_u64;
    let mut invalid = 0_u64;
    let mut lower_sum = 0_u64;
    let mut upper_sum = 0_u64;
    for relation in &relations.relation_samples {
        let (Some(issue), Some(acquire)) = (
            relation.producer_issue_clock,
            relation.wait_visibility_clock,
        ) else {
            continue;
        };
        if acquire < issue {
            invalid = invalid.saturating_add(1);
            continue;
        }
        let last_false = observations
            .sync
            .get(&relation.wait)
            .and_then(|observation| observation.last_unsuccessful_query_clock);
        if last_false.is_some_and(|clock| clock < issue || clock > acquire) {
            invalid = invalid.saturating_add(1);
            continue;
        }
        let lower = last_false.map_or(0, |clock| clock - issue);
        let upper = acquire - issue;
        pairs = pairs.saturating_add(1);
        lower_sum = lower_sum.saturating_add(lower);
        upper_sum = upper_sum.saturating_add(upper);
    }
    (pairs, invalid, lower_sum, upper_sum)
}

#[derive(Debug, thiserror::Error, Clone, Copy, PartialEq, Eq)]
pub enum MemoryRouteError {
    #[error("GLOBAL carries {got} address records; active records require at least one operand")]
    GlobalOperandCount { got: usize },
    #[error("GLOBAL operand {operand} carries non-global address_space {space:?}")]
    GlobalSpaceInvalid {
        operand: usize,
        space: pb::MemorySpace,
    },
    #[error(
        "SHARED/LOCAL carries {got} address records; active records require at least one operand"
    )]
    SharedLocalOperandCount { got: usize },
    #[error("SHARED/LOCAL operand {operand} carries incompatible address_space {space:?}")]
    SharedLocalSpaceInvalid {
        operand: usize,
        space: pb::MemorySpace,
    },
    #[error(
        "ATOMIC carries {got} address records; expected exactly one explicit-space operand, or none only when zero-active"
    )]
    AtomicOperandCount { got: usize },
    #[error("ATOMIC operand {operand} is missing address_space")]
    AtomicSpaceMissing { operand: usize },
    #[error("ATOMIC operand {operand} carries UNSPECIFIED address_space")]
    AtomicSpaceUnspecified { operand: usize },
    #[error("legacy ATOMIC operands carry conflicting address-space routes")]
    AtomicSpacesConflict,
    #[error("ASYNC_COPY carries {got} address records; expected exactly one global source")]
    AsyncCopySourceCount { got: usize },
    #[error("ASYNC_COPY source is missing address_space")]
    AsyncCopySourceSpaceMissing,
    #[error("ASYNC_COPY source carries non-global address_space {space:?}")]
    AsyncCopySourceSpaceInvalid { space: pb::MemorySpace },
    #[error("ASYNC_COPY carries {got} operands; destination-bearing contract expects {expected}")]
    AsyncCopyOperandCount { got: usize, expected: usize },
    #[error("ASYNC_COPY destination is missing address_space")]
    AsyncCopyDestinationSpaceMissing,
    #[error("ASYNC_COPY destination carries non-shared address_space {space:?}")]
    AsyncCopyDestinationSpaceInvalid { space: pb::MemorySpace },
    #[error("memory_op_access_kind {got:?} is incompatible; expected {expected}")]
    AccessKindInvalid {
        got: pb::MemoryOpAccessKind,
        expected: &'static str,
    },
    #[error(
        "memory_op_access_kind {got:?} contradicts the joined static opcode direction {expected:?}"
    )]
    StaticAccessKindMismatch {
        got: pb::MemoryOpAccessKind,
        expected: pb::MemoryOpAccessKind,
    },
    #[error("joined static opcode requires {expected:?} address-space routing, got {got:?}")]
    StaticMemoryRouteMismatch {
        expected: pb::MemorySpace,
        got: pb::MemorySpace,
    },
    #[error("generic atomic requires Global or Shared address-space routing, got {got:?}")]
    StaticGenericAtomicRouteInvalid { got: pb::MemorySpace },
    #[error("joined static opcode requires memory class {expected:?}, got {got:?}")]
    StaticMemoryClassMismatch {
        expected: pb::MemoryOpClass,
        got: pb::MemoryOpClass,
    },
    #[error("static SASS family {family} has no implemented route on architecture {arch}")]
    StaticMemoryFamilyUnsupported { family: &'static str, arch: ArchId },
    #[error("operand {operand} carries address_space value {raw} unknown to the committed schema")]
    UnknownAddressSpace { operand: usize, raw: i32 },
}

/// A separate, architecture-scoped GPC observation frontier.  The underlying
/// modeled event frontier remains available as `lower_bound_cycles`; this
/// projection adds only the independently measured SM86 front-end/semaphore
/// envelope and never consumes an NCU target or the timing crosswalk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpcObservationFrontier {
    pub lower_bound_cycles: u64,
    pub envelope_cycles: u64,
    pub grid_blocks: u64,
    pub grid_log2: u32,
    pub cycles: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct FastFrontierCoverage {
    pub trace_records: u64,
    pub candidate_records: u64,
    pub summarized_records: u64,
    pub reified_records: u64,
    pub uncovered_records: u64,
}

pub(crate) fn fast_register_overlap(left: SassRegister, right: SassRegister) -> bool {
    if left.file() != right.file() {
        return false;
    }
    let (Some(left), Some(right)) = (
        left.index()
            .map(|i| (u32::from(i), u32::from(left.span_registers()))),
        right
            .index()
            .map(|i| (u32::from(i), u32::from(right.span_registers()))),
    ) else {
        return false;
    };
    left.0 < right.0 + right.1 && right.0 < left.0 + left.1
}

/// Event-driven SM120 fast replay. Same-cycle picks see one immutable resource
/// snapshot; idle time jumps to the exact next known readiness. The modulo-four
/// SMSP assignment is the existing alpha assumption, not observed wiring.
pub(crate) fn fast_dynamic_sm_frontier(
    blocks: &[BlockReplay],
    schedule: &FunctionSchedule,
    masks: &BTreeMap<(i64, i32), Vec<u32>>,
    params: &IssueModelParams,
    residency: SharedMemoryResidency,
) -> Result<(u64, u64, u64, u64), ReplayError> {
    pub(crate) struct BlockState {
        pub(crate) first: usize,
        pub(crate) end: usize,
        pub(crate) slots: u64,
        pub(crate) rearm: u64,
    }
    let mut warps = Vec::new();
    let mut block_states = Vec::new();
    for block in blocks {
        let first = warps.len();
        let mut slots = 0u64;
        for warp in &block.warps {
            slots |= 1u64.checked_shl(warp.warp_slot as u32).ok_or_else(|| {
                ReplayError::FastFrontier {
                    detail: "fast warp slot exceeds its resident bitmap".to_owned(),
                }
            })?;
            let lane_masks = masks
                .get(&(block.launch_index, warp.warp_slot))
                .ok_or_else(|| ReplayError::FastFrontier {
                    detail: "fast warp lane masks are missing".to_owned(),
                })?;
            if lane_masks.len() != warp.insts.len() {
                return Err(ReplayError::FastFrontier {
                    detail: "fast warp lane-mask length mismatch".to_owned(),
                });
            }
            let slot = warp
                .insts
                .first()
                .map(|inst| {
                    schedule.issue_slot(u64::from(inst.pc)).ok_or_else(|| {
                        ReplayError::FastFrontier {
                            detail: "fast schedule PC missing".to_owned(),
                        }
                    })
                })
                .transpose()?;
            warps.push(FastDynamicWarp {
                warp,
                masks: lane_masks,
                next: 0,
                slot,
                ready: 0,
                completion: 0,
                waiting_at_barrier: false,
                barrier_arrival: 0,
                done: slot.is_none(),
                registers: Vec::new(),
                barriers: [0; 6],
            });
        }
        block_states.push(BlockState {
            first,
            end: warps.len(),
            slots,
            rearm: 0,
        });
    }
    let mut active: Vec<usize> = Vec::new();
    let mut next_block = 0;
    let mut active_slots = 0u64;
    let mut dispatch = [0u64; 4];
    let mut pipes = [[0u64; 4]; 4];
    let mut pending: [Vec<u64>; 4] = std::array::from_fn(|_| Vec::new());
    let mut held = [None; 4];
    let mut barrier_server = 0u64;
    let barrier_rearm = schedule
        .by_pc
        .values()
        .filter_map(|slot| match slot.pipeline_class() {
            PipelineClass::Barrier {
                generation_rearm_cycles,
                ..
            } => Some(generation_rearm_cycles),
            _ => None,
        })
        .max()
        .unwrap_or(0);
    let mut now = 0u64;
    let mut frontier = 0u64;
    let (mut memory_records, mut async_records, mut barrier_records) = (0u64, 0u64, 0u64);
    while next_block < blocks.len() || !active.is_empty() {
        let mut next_event = None;
        active.retain(|&block| {
            let state = &block_states[block];
            let members = &warps[state.first..state.end];
            if members.iter().all(|w| w.done) {
                let end = members.iter().map(|w| w.completion).max().unwrap_or(now);
                if end <= now {
                    active_slots &= !state.slots;
                    return false;
                }
                next_event = Some(next_event.map_or(end, |prior: u64| prior.min(end)));
            }
            true
        });
        // Admission reuses traced slots only after the prior CTA has retired.
        // Apply the same launch-order, warp, CTA and shared-memory ceilings as
        // full replay; do not admit every traced block simultaneously.
        while next_block < blocks.len() {
            let state = &block_states[next_block];
            let occupied = active_slots | state.slots;
            let shared_fits = (active.len() as u64 + 1)
                .checked_mul(residency.bytes_per_cta())
                .is_some_and(|bytes| bytes <= residency.max_bytes_per_sm());
            if active.len() >= params.max_resident_blocks_per_sm() as usize
                || occupied.count_ones() > params.max_resident_warps_per_sm()
                || active_slots & state.slots != 0
                || !shared_fits
            {
                break;
            }
            for warp in &mut warps[state.first..state.end] {
                warp.ready = now;
                warp.completion = now;
            }
            active_slots = occupied;
            active.push(next_block);
            next_block += 1;
        }
        if active.is_empty() {
            if next_block == blocks.len() {
                break;
            }
            return Err(ReplayError::FastFrontier {
                detail: "fast CTA exceeds resident resource limits".to_owned(),
            });
        }
        for completions in &mut pending {
            completions.retain(|&done| done > now);
        }
        let mut picks = [None; 4];
        for &block in &active {
            let state = &block_states[block];
            for index in state.first..state.end {
                let warp = &warps[index];
                let smsp = warp.warp.warp_slot as usize % 4;
                let Some(ready) =
                    warp.ready_at(dispatch[smsp], &pipes[smsp], &pending[smsp], state.rearm)
                else {
                    continue;
                };
                if ready > now {
                    next_event = Some(next_event.map_or(ready, |prior| prior.min(ready)));
                } else if picks[smsp].is_none()
                    || held[smsp] == Some(index)
                    || (picks[smsp] != held[smsp] && picks[smsp].is_some_and(|prior| index > prior))
                {
                    picks[smsp] = Some(index);
                }
            }
        }
        if picks.iter().all(Option::is_none) {
            now =
                next_event
                    .filter(|&next| next > now)
                    .ok_or_else(|| ReplayError::FastFrontier {
                        detail: "fast replay cannot progress: incomplete CTA barrier or schedule"
                            .to_owned(),
                    })?;
            continue;
        }
        // Every pick below issues at `now`; committing one partition cannot
        // move the timestamp seen by another partition's already selected pick.
        for (smsp, pick) in picks.into_iter().enumerate() {
            let Some(index) = pick else { continue };
            let warp = &mut warps[index];
            let inst = warp.warp.insts[warp.next];
            let slot = warp.slot.ok_or_else(|| ReplayError::FastFrontier {
                detail: "fast schedule PC missing".to_owned(),
            })?;
            let mask = warp.masks[warp.next];
            let pipeline = slot.pipeline_class();
            let control = slot.control();
            let interval = pipeline.initiation_interval_cycles().unwrap_or(1).max(1);
            let latency = match pipeline {
                PipelineClass::ScalarAlu { latency_cycles, .. } => latency_cycles,
                _ => 1,
            };
            let issued_end = fast_cycle_add(now, 1)?;
            dispatch[smsp] = issued_end;
            frontier = frontier.max(issued_end);
            let projection = slot.fixed_latency_operands().projected();
            let observes_register = mask != 0
                && projection
                    .is_some_and(|p| p.destinations().iter().any(|d| !d.register().is_constant()));
            let result_ready = crate::issue::data_ready_time(
                inst.kind,
                &[],
                now,
                params,
                None,
                None,
                observes_register,
            )?;
            let done = if mask == 0 {
                issued_end
            } else {
                if matches!(pipeline, PipelineClass::ScalarAlu { .. }) {
                    fast_cycle_add(now, latency)?
                } else {
                    result_ready.unwrap_or(fast_cycle_add(now, 1)?)
                }
            };
            if mask != 0 {
                if !matches!(
                    pipeline,
                    PipelineClass::Barrier { .. } | PipelineClass::Branch { .. }
                ) {
                    pipes[smsp][fast_pipeline_index(pipeline)] = fast_cycle_add(now, interval)?;
                }
                if matches!(pipeline, PipelineClass::ScalarAlu { .. }) {
                    pending[smsp].push(done);
                }
                // Completed writes no longer constrain future reads. Partial
                // predication overwrites only executing lanes, not the whole register.
                warp.registers.retain(|&(_, _, ready)| ready > now);
                if let Some(projection) = projection {
                    for register in projection
                        .destinations()
                        .iter()
                        .flat_map(|d| d.register().components())
                    {
                        let lanes = fast_register_mask(register, mask);
                        for (prior, prior_lanes, _) in &mut warp.registers {
                            if *prior == register {
                                *prior_lanes &= !lanes;
                            }
                        }
                        warp.registers.retain(|&(_, lanes, _)| lanes != 0);
                        warp.registers.push((register, lanes, done));
                    }
                }
            }
            // Share Flat-mode dependency latency arithmetic with full replay.
            // Cache queues and native async visibility are still not modeled here.
            let write = (mask != 0 && slot.op_class() != OpClass::AsyncCopyCommit)
                .then_some(control.write_barrier())
                .flatten();
            if let Some(ready) =
                crate::issue::data_ready_time(inst.kind, &[], now, params, None, write, false)?
            {
                if let Some(barrier) = write {
                    warp.barriers[barrier.index()] = warp.barriers[barrier.index()].max(ready);
                }
            }
            if mask != 0
                && let Some(barrier) = control.read_barrier()
            {
                warp.barriers[barrier.index()] = warp.barriers[barrier.index()]
                    .max(fast_cycle_add(now, params.read_barrier_latency())?);
            }
            let suppress = mask != 0
                && projection.is_some()
                && sm120_alpha_scalar_control_stall_suppressed(slot.opcode_text());
            let stall = if suppress {
                1
            } else {
                u64::from(control.stall()).max(1)
            };
            warp.ready = fast_cycle_add(now, stall.max(if control.yield_flag() { 2 } else { 1 }))?;
            warp.completion = warp.completion.max(done);
            match inst.kind {
                AccessKind::NonMemory | AccessKind::PredicatedOffMemory => {}
                AccessKind::GlobalAsyncCopy { .. } => {
                    memory_records += 1;
                    async_records += 1;
                }
                _ => memory_records += 1,
            }
            warp.next += 1;
            if slot.op_class() == OpClass::BlockBarrier && mask != 0 {
                barrier_records += 1;
                barrier_server = fast_cycle_add(barrier_server.max(now), interval)?;
                warp.barrier_arrival = barrier_server;
                warp.waiting_at_barrier = true;
            } else if warp.next == warp.warp.insts.len() {
                warp.done = true;
            }
            warp.slot = warp
                .warp
                .insts
                .get(warp.next)
                .map(|next| {
                    schedule.issue_slot(u64::from(next.pc)).ok_or_else(|| {
                        ReplayError::FastFrontier {
                            detail: "fast schedule PC missing".to_owned(),
                        }
                    })
                })
                .transpose()?;
            held[smsp] =
                (!warp.done && !warp.waiting_at_barrier && !control.yield_flag()).then_some(index);
        }
        for &block in &active {
            let state = &mut block_states[block];
            let members = &mut warps[state.first..state.end];
            if members.iter().any(|w| w.waiting_at_barrier)
                && members.iter().all(|w| w.done || w.waiting_at_barrier)
            {
                let arrival = members
                    .iter()
                    .filter(|w| w.waiting_at_barrier)
                    .map(|w| w.barrier_arrival)
                    .max()
                    .unwrap_or(now);
                let release = fast_cycle_add(arrival, 1)?;
                state.rearm = fast_cycle_add(arrival, barrier_rearm)?;
                for warp in members.iter_mut().filter(|w| w.waiting_at_barrier) {
                    warp.waiting_at_barrier = false;
                    warp.ready = warp.ready.max(release);
                    warp.completion = warp.completion.max(release);
                    warp.done = warp.next == warp.warp.insts.len();
                }
                frontier = frontier.max(release);
            }
        }
        now = fast_cycle_add(now, 1)?;
    }
    frontier = frontier.max(warps.iter().map(|w| w.completion).max().unwrap_or(0));
    Ok((frontier, memory_records, async_records, barrier_records))
}

/// Coverage classification retained by an explicitly unadmitted diagnostic replay.
#[cfg(feature = "campaign")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnadmittedCoverage {
    Full,
    SampledObserved,
}

#[derive(Debug)]
pub(crate) enum ExtractedCoverage {
    Full(FullLaunchCoverage),
    #[cfg(feature = "campaign")]
    Sampled(SampledLaunchCoverage),
}

#[derive(Debug)]
pub(crate) struct ExtractedKernel {
    pub(crate) per_sm_blocks: BTreeMap<i32, Vec<BlockReplay>>,
    pub(crate) per_sm_instruction_read_sectors: BTreeMap<i32, u64>,
    pub(crate) records_seen: u64,
    pub(crate) coverage: ExtractedCoverage,
    pub(crate) shared_memory_residency: SharedMemoryResidency,
    pub(crate) front_end: FrontEndCounters,
    pub(crate) trace_metrics: TraceMetrics,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct OrderedAsyncEvent {
    pub(crate) order: u64,
    pub(crate) launch_index: i64,
    pub(crate) warp_slot: i32,
    pub(crate) kind: OrderedAsyncEventKind,
}

pub(crate) fn ordered_async_event_kind(
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

#[cfg(feature = "campaign")]
pub(crate) fn validate_declared_sampling_sm_range(
    declared: impl IntoIterator<Item = (u64, u64)>,
    placement: PlacementConfig,
) -> Result<(), ReplayError> {
    for (launch_index, sm_id) in declared {
        placement.require_declared_sm(launch_index, sm_id)?;
    }
    Ok(())
}

pub(crate) fn normalized_cluster_dim(
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

pub(crate) fn checked_block_warps(block: (u32, u32, u32)) -> Result<u64, ReplayError> {
    pub(crate) const TRACE_WARP_WIDTH: u64 = 32;
    let threads = u64::from(block.0)
        .checked_mul(u64::from(block.1))
        .and_then(|value| value.checked_mul(u64::from(block.2)))
        .ok_or(ReplayError::BlockSizeOverflow { block })?;
    Ok(threads.div_ceil(TRACE_WARP_WIDTH))
}

pub(crate) fn checked_collection_len(
    collection: &'static str,
    len: usize,
) -> Result<u64, ReplayError> {
    u64::try_from(len).map_err(|_| ReplayError::CollectionLengthOutOfRange { collection, len })
}

pub(crate) fn is_sync_sidecar_opcode(opcode: &str) -> bool {
    opcode.starts_with("SYNCS.") || opcode.starts_with("ARRIVES.")
}

/// Count distinct 32-byte instruction sectors assigned to one traced SM.
/// This is kept separate from the launch-wide front-end count so a later
/// effective observation envelope can use the same scope as active-lifetime
/// IPC. It is metadata only and never contributes replay cycles.
pub(crate) fn per_sm_instruction_sector_set(
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

/// Validate every consumed threadblock and classify launch coverage without
/// running the timing model. The campaign feature uses this for its manifest
/// compatibility path.
#[cfg(feature = "campaign")]
pub(crate) fn admit_launch_coverage(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedule: &FunctionSchedule,
    arch_card: &ArchCard,
) -> Result<ExtractedCoverage, ReplayError> {
    let placement = PlacementConfig::from_arch_card(arch_card, schedule.arch)?;
    require_trace_architecture(td, schedule.arch)?;
    Ok(extract_per_sm_blocks(td, loc, schedule, placement, false, None)?.coverage)
}

/// Fold per-SM results into the kernel-level view, enforcing the
/// one-issue-per-record invariant.
pub(crate) fn fold_kernel_result(
    arch: ArchId,
    per_sm: BTreeMap<i32, SmSimResult>,
    records_seen: u64,
    expect_l1: bool,
    last_processed_system_event_time: Option<u64>,
    l2: Option<(L2Counters, L2SourceSplit, DramStats)>,
) -> Result<FoldedKernelReplay, ReplayError> {
    fold_kernel_result_with_sm_active_time(
        arch,
        per_sm,
        records_seen,
        expect_l1,
        last_processed_system_event_time,
        last_processed_system_event_time,
        l2,
    )
}

pub(crate) fn fold_kernel_result_with_sm_active_time(
    arch: ArchId,
    per_sm: BTreeMap<i32, SmSimResult>,
    records_seen: u64,
    expect_l1: bool,
    last_processed_system_event_time: Option<u64>,
    sm_active_terminal_event_time: Option<u64>,
    l2: Option<(L2Counters, L2SourceSplit, DramStats)>,
) -> Result<FoldedKernelReplay, ReplayError> {
    let mut instructions_issued: u64 = 0;
    let mut diagnostic_last_issue_cycles: u64 = 0;
    let mut latest_local_completion_cycles: Option<u64> = None;
    let mut unmodeled_tails = TailFlags::default();
    let mut l1_totals = expect_l1.then(|| L1Counters::for_arch(arch));
    for (&sm_id, result) in &per_sm {
        instructions_issued = instructions_issued
            .checked_add(result.instructions_issued)
            .ok_or(ReplayError::InstructionCountOverflow {
                sm_id,
                accumulated: instructions_issued,
                addend: result.instructions_issued,
            })?;
        diagnostic_last_issue_cycles = diagnostic_last_issue_cycles.max(result.cycles);
        if let Some(completion) = result.modeled_local_completion_cycles {
            latest_local_completion_cycles = Some(
                latest_local_completion_cycles
                    .map_or(completion, |current| current.max(completion)),
            );
        }
        unmodeled_tails = unmodeled_tails.union(result.unmodeled_tails);
        let found_l1 = result.l1.is_some();
        if found_l1 != expect_l1 {
            return Err(ReplayError::L1PresenceMismatch {
                sm_id,
                expected: expect_l1,
                found: found_l1,
            });
        }
        if let (Some(c), Some(total)) = (&result.l1, &mut l1_totals) {
            total
                .add_checked(c)
                .map_err(|source| ReplayError::L1Aggregation { sm_id, source })?;
        }
    }
    if instructions_issued != records_seen {
        return Err(ReplayError::CountMismatch {
            issued: instructions_issued,
            records: records_seen,
        });
    }
    let issue_frontier = (instructions_issued != 0).then_some(diagnostic_last_issue_cycles);
    let terminal_modeled_event_time = [
        issue_frontier,
        last_processed_system_event_time,
        latest_local_completion_cycles,
    ]
    .into_iter()
    .flatten()
    .max();
    let (l2, l2_by_source, dram) = match l2 {
        Some((counters, split, dram)) => (Some(counters), Some(split), Some(dram)),
        None => (None, None, None),
    };
    Ok(FoldedKernelReplay {
        arch,
        instructions_issued,
        records_seen,
        per_sm,
        diagnostic_last_issue_cycles,
        last_processed_system_event_time,
        terminal_modeled_event_time,
        sm_active_terminal_event_time: [
            issue_frontier,
            sm_active_terminal_event_time,
            latest_local_completion_cycles,
        ]
        .into_iter()
        .flatten()
        .max(),
        l1: l1_totals,
        l2,
        l2_by_source,
        dram,
        front_end: FrontEndCounters::default(),
        trace_metrics: TraceMetrics::default(),
        unmodeled_tails,
    })
}

/// Temporal CTA co-residency provenance. The current trace wire carries spatial
/// `sm_id`/`warpid_sm` placement but no block start/end timestamps, so exact
/// hardware co-residency cannot be claimed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoResidency {
    /// Blocks are admitted in launch-index order under the modeled resident-warp
    /// ceiling and architecture-card shared-memory capacity; concurrent CTAs must
    /// also use disjoint traced physical warp slots. This is never a claim that the
    /// hardware used the same overlap.
    Modeled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OrderedSessionLaunch {
    pub(crate) kernel_id: i32,
    pub(crate) function_unique_id: i32,
}

impl StreamL2ReplaySession {
    /// Bind a session to one trace root and one exact device/stream launch
    /// sequence. Construction is accepted only for the shared-L2 replay mode.
    pub fn new(
        td: &TraceDir,
        arch_card: &ArchCard,
        params: &IssueModelParams,
        mode: ReplayMemMode,
        device_id: i64,
        stream_id: i64,
    ) -> Result<Self, ReplaySessionError> {
        let configuration = ReplayConfiguration::checked(*params, mode)?;
        let l2_config = match mode {
            ReplayMemMode::L1SharedL2 { l2, coherence, .. } => {
                if coherence.is_some() {
                    return Err(ReplaySessionError::CoherenceRequiresLaunchScope);
                }
                l2
            }
            _ => return Err(ReplaySessionError::SharedL2ModeRequired { mode }),
        };
        require_trace_architecture(td, configuration.arch())?;
        let placement = PlacementConfig::from_arch_card(arch_card, configuration.arch())?;
        let expected_launches: Vec<_> = td
            .kernels()
            .filter(|loc| loc.device_id() == device_id && loc.stream_id() == stream_id)
            .map(|loc| OrderedSessionLaunch {
                kernel_id: loc.kernel().id,
                function_unique_id: loc.kernel().function_unique_id,
            })
            .collect();
        if expected_launches.is_empty() {
            return Err(ReplaySessionError::EmptyStream {
                device_id,
                stream_id,
            });
        }
        let l2 = L2Model::new(l2_config).map_err(ReplayError::from)?;
        Ok(Self {
            trace_root: td.root().to_path_buf(),
            trace_lineage: td.lineage(),
            configuration,
            placement,
            device_id,
            stream_id,
            expected_launches,
            completed_launches: 0,
            l2,
            cumulative_reported_l2: L2Counters::for_arch(configuration.arch()),
        })
    }
}

impl StreamL2ReplaySession {
    pub fn configuration(&self) -> ReplayConfiguration {
        self.configuration
    }
}

impl StreamL2ReplaySession {
    pub fn device_id(&self) -> i64 {
        self.device_id
    }
}

impl StreamL2ReplaySession {
    pub fn stream_id(&self) -> i64 {
        self.stream_id
    }
}

impl StreamL2ReplaySession {
    pub fn completed_launches(&self) -> usize {
        self.completed_launches
    }
}

impl StreamL2ReplaySession {
    pub fn remaining_launches(&self) -> usize {
        self.expected_launches.len() - self.completed_launches
    }
}

impl StreamL2ReplaySession {
    pub fn is_complete(&self) -> bool {
        self.completed_launches == self.expected_launches.len()
    }
}

impl StreamL2ReplaySession {
    pub fn cumulative_l2_counters(&self) -> L2Counters {
        self.cumulative_reported_l2
    }
}

impl StreamL2ReplaySession {
    pub fn pending_l2_sectors(&self) -> usize {
        self.l2.pending_sector_count()
    }
}

impl StreamL2ReplaySession {
    /// Replay the next launch transactionally. The session cache and cursor are
    /// committed together only after the launch result, drained-state check,
    /// per-launch delta, and cumulative-counter reconciliation all succeed.
    pub fn replay_next(
        &mut self,
        td: &TraceDir,
        loc: &KernelLoc<'_>,
        schedules: &BTreeMap<i32, FunctionSchedule>,
        arch_card: &ArchCard,
        params: &IssueModelParams,
        mode: ReplayMemMode,
    ) -> Result<KernelReplayOutcome, ReplaySessionError> {
        if td.root() != self.trace_root {
            return Err(ReplaySessionError::TraceRootMismatch {
                expected: self.trace_root.clone(),
                got: td.root().to_path_buf(),
            });
        }
        if td.lineage() != self.trace_lineage {
            return Err(ReplaySessionError::TraceLineageMismatch);
        }
        if loc.device_id() != self.device_id {
            return Err(ReplaySessionError::DeviceMismatch {
                expected: self.device_id,
                got: loc.device_id(),
            });
        }
        if loc.stream_id() != self.stream_id {
            return Err(ReplaySessionError::StreamMismatch {
                expected: self.stream_id,
                got: loc.stream_id(),
            });
        }
        let Some(expected_launch) = self.expected_launches.get(self.completed_launches) else {
            return Err(ReplaySessionError::LaunchSequenceComplete {
                completed_launches: self.completed_launches,
            });
        };
        if loc.kernel().id != expected_launch.kernel_id
            || loc.kernel().function_unique_id != expected_launch.function_unique_id
        {
            return Err(ReplaySessionError::LaunchOrderMismatch {
                position: self.completed_launches,
                expected_kernel_id: expected_launch.kernel_id,
                expected_function_unique_id: expected_launch.function_unique_id,
                got_kernel_id: loc.kernel().id,
                got_function_unique_id: loc.kernel().function_unique_id,
            });
        }

        let got_configuration = ReplayConfiguration::checked(*params, mode)?;
        if got_configuration != self.configuration {
            return Err(ReplaySessionError::ConfigurationMismatch {
                expected: self.configuration,
                got: got_configuration,
            });
        }
        let got_placement = PlacementConfig::from_arch_card(arch_card, self.configuration.arch())?;
        if got_placement != self.placement {
            return Err(ReplaySessionError::ArchitectureCardMismatch {
                expected_sku: self.placement.sku,
                expected_sm_count: self.placement.sm_count,
                expected_max_shared_bytes_per_sm: self.placement.max_shared_bytes_per_sm,
                got_sku: got_placement.sku,
                got_sm_count: got_placement.sm_count,
                got_max_shared_bytes_per_sm: got_placement.max_shared_bytes_per_sm,
            });
        }
        if self.l2.pending_sector_count() != 0 {
            return Err(ReplayError::RetainedL2NotDrained {
                pending_sectors: self.l2.pending_sector_count(),
            }
            .into());
        }

        let mut staged_l2 = self.l2.try_snapshot().map_err(ReplayError::from)?;
        let schedule = lookup_schedule(schedules, loc)?;
        let outcome = replay_kernel_with_schedule_and_l2(
            td,
            loc,
            schedule,
            arch_card,
            params,
            mode,
            Some(&mut staged_l2),
        )?;
        if staged_l2.pending_sector_count() != 0 {
            return Err(ReplayError::RetainedL2NotDrained {
                pending_sectors: staged_l2.pending_sector_count(),
            }
            .into());
        }
        let launch_delta = outcome
            .l2_counters()
            .copied()
            .ok_or(ReplaySessionError::LaunchL2CountersMissing)?;
        let mut cumulative_reported_l2 = self.cumulative_reported_l2;
        cumulative_reported_l2
            .add_checked(&launch_delta)
            .map_err(ReplayError::from)?;
        let model_counters = *staged_l2.counters().map_err(ReplayError::from)?;
        if cumulative_reported_l2 != model_counters {
            return Err(ReplaySessionError::CumulativeL2CounterMismatch {
                reported: cumulative_reported_l2,
                model: model_counters,
            });
        }

        self.l2 = staged_l2;
        self.cumulative_reported_l2 = cumulative_reported_l2;
        self.completed_launches += 1;
        Ok(outcome)
    }
}

/// Replay one launch while evaluating a bounded set of observational streaming
/// memory-service hypotheses at the exact issue timestamps. Candidates neither
/// delay issue nor add events; their per-SM diagnostics expose counterfactual
/// service busy time, queueing, and terminal completion. This research surface
/// is intentionally restricted to the full shared-L2 hierarchy mode.
pub fn replay_trace_kernel_with_memory_service_candidates(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedules: &BTreeMap<i32, FunctionSchedule>,
    arch_card: &ArchCard,
    params: &IssueModelParams,
    mode: ReplayMemMode,
    memory_service_candidates: &[MemoryServiceCandidate],
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
        memory_service_candidates,
        &[],
        None,
    )
}

/// Replay one launch while evaluating bounded finite operand-admission queues
/// against exact per-instruction L2 hit/miss outcomes. The evaluator preserves
/// the observed issue and cache order and emits schema-9 diagnostics only; it
/// does not alter scheduler timing or hierarchy state.
pub fn replay_trace_kernel_with_memory_admission_candidates(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedules: &BTreeMap<i32, FunctionSchedule>,
    arch_card: &ArchCard,
    params: &IssueModelParams,
    mode: ReplayMemMode,
    memory_admission_candidates: &[MemoryAdmissionCandidate],
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
        memory_admission_candidates,
        None,
    )
}

/// Replay caller-supplied trace/schedule inputs for debugging only. The returned
/// diagnostics retain counters and directional timestamps but erase the full
/// replay type, so they cannot enter the legacy launch-set aggregation API.
#[cfg(feature = "campaign")]
pub fn replay_unadmitted_diagnostics(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedules: &BTreeMap<i32, FunctionSchedule>,
    arch_card: &ArchCard,
    params: &IssueModelParams,
    mode: ReplayMemMode,
) -> Result<UnadmittedReplayDiagnostics, ReplayError> {
    let schedule = lookup_schedule(schedules, loc)?;
    let outcome = replay_kernel_with_schedule(td, loc, schedule, arch_card, params, mode)?;
    Ok(UnadmittedReplayDiagnostics::from_outcome(outcome))
}

pub(crate) fn require_ldgsts_l1_policy(
    schedule: &FunctionSchedule,
    arch: ArchId,
    pc: u32,
) -> Result<LdgstsL1Policy, ReplayError> {
    schedule
        .by_pc
        .get(&u64::from(pc))
        .and_then(|slot| slot.ldgsts_l1_policy)
        .ok_or(ReplayError::UnsupportedMemoryPath {
            arch,
            class: pb::MemoryOpClass::AsyncCopy,
        })
}

/// Return the schedule's memory role while rejecting a dynamic global record
/// whose static sidecar opcode is neither classified nor one of the deliberate
/// untagged load-family heads. A pc absent from the schedule stays neutral here
/// so the issue loop reports its more specific `PcNotInSchedule` error.
pub(crate) fn require_known_global_role(
    schedule: &FunctionSchedule,
    arch: ArchId,
    pc: u32,
) -> Result<MemRole, ReplayError> {
    let Some(slot) = schedule.by_pc.get(&u64::from(pc)) else {
        return Ok(MemRole::None);
    };
    if slot.mem_role == MemRole::None && !is_known_untagged_opcode(&slot.opcode_text) {
        // The SM120 alpha path admits newly observed Blackwell storage-family
        // spellings as untagged global traffic.  They retain their dynamic
        // address/sector stream and flat latency, but do not borrow a guessed
        // cache operator or writeback class.  Ordinary SM120 replay remains
        // fail-closed because this branch is keyed by the alpha-tagged
        // deployment schedule.
        if arch != ArchId::Sm86 {
            return Ok(MemRole::None);
        }
        return Err(ReplayError::UnknownGlobalOpcode {
            arch,
            pc,
            opcode: opcode_head(&slot.opcode_text).to_owned(),
        });
    }
    Ok(slot.mem_role)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReplayMemoryRoute {
    Global,
    Local,
    Shared,
    Unsupported,
}

impl ReplayMemoryRoute {
    pub(crate) const fn canonical_space(self) -> pb::MemorySpace {
        match self {
            Self::Global => pb::MemorySpace::Global,
            Self::Local => pb::MemorySpace::Local,
            Self::Shared => pb::MemorySpace::Shared,
            Self::Unsupported => pb::MemorySpace::Unspecified,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StaticMemoryRoute {
    Modeled {
        route: ReplayMemoryRoute,
        class: pb::MemoryOpClass,
    },
    /// Unqualified ATOM/RED use PTX generic addressing. Their dynamic address
    /// operand can resolve to either global or shared memory, so the static
    /// mnemonic supplies the operation class but not one fixed route.
    GenericAtomic {
        class: pb::MemoryOpClass,
    },
    UnsupportedClass(pb::MemoryOpClass),
}

pub(crate) fn is_tma_submission_opcode(opcode_text: &str) -> bool {
    matches!(opcode_head(opcode_text), "UTMALDG" | "UTMASTG" | "UBLKCP")
}

pub(crate) fn validate_tma_load_direction(
    schedule: &FunctionSchedule,
    arch: ArchId,
    pc: u32,
) -> Result<(), ReplayError> {
    let Some(slot) = schedule.by_pc.get(&u64::from(pc)) else {
        return Ok(());
    };
    if is_tma_submission_opcode(&slot.opcode_text) && !is_modeled_tma_load_opcode(&slot.opcode_text)
    {
        return Err(ReplayError::UnsupportedTmaDirection {
            arch,
            pc,
            opcode: slot.opcode_text.clone(),
        });
    }
    Ok(())
}

/// Reject disagreement between a known space-coded static SASS family and the
/// class/route resolved from the dynamic contract. This runs before allocation
/// identity and classification, so a contradictory record cannot borrow another
/// operation or storage class's timing and traffic semantics.
pub(crate) fn validate_static_memory_route(
    meta: MemoryAccessMeta,
    route: ReplayMemoryRoute,
    schedule: &FunctionSchedule,
    pc: u32,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<(), ReplayError> {
    let Some(slot) = schedule.by_pc.get(&u64::from(pc)) else {
        return Ok(());
    };
    let contextualize = |source| ReplayError::MemoryRoute {
        path: path.to_path_buf(),
        warp_id,
        record_ix,
        class: meta.op_class,
        source,
    };
    validate_tma_load_direction(schedule, schedule.arch, pc)?;
    let local_role = matches!(
        slot.mem_role,
        MemRole::EligibleLoad(CacheOpToken::Ldl | CacheOpToken::Stl)
    );
    if local_role && route != ReplayMemoryRoute::Local {
        return Err(contextualize(MemoryRouteError::StaticMemoryRouteMismatch {
            expected: pb::MemorySpace::Local,
            got: route.canonical_space(),
        }));
    }
    let expected = static_memory_route(&slot.opcode_text, schedule.arch).map_err(|family| {
        contextualize(MemoryRouteError::StaticMemoryFamilyUnsupported {
            family,
            arch: schedule.arch,
        })
    })?;
    match expected {
        Some(StaticMemoryRoute::Modeled {
            route: expected, ..
        }) if expected != route => {
            Err(contextualize(MemoryRouteError::StaticMemoryRouteMismatch {
                expected: expected.canonical_space(),
                got: route.canonical_space(),
            }))
        }
        Some(StaticMemoryRoute::Modeled {
            class: expected, ..
        }) if expected != meta.op_class => {
            Err(contextualize(MemoryRouteError::StaticMemoryClassMismatch {
                expected,
                got: meta.op_class,
            }))
        }
        Some(StaticMemoryRoute::GenericAtomic { class: expected }) => {
            if expected != meta.op_class {
                return Err(contextualize(MemoryRouteError::StaticMemoryClassMismatch {
                    expected,
                    got: meta.op_class,
                }));
            }
            if matches!(route, ReplayMemoryRoute::Global | ReplayMemoryRoute::Shared) {
                Ok(())
            } else {
                Err(contextualize(
                    MemoryRouteError::StaticGenericAtomicRouteInvalid {
                        got: route.canonical_space(),
                    },
                ))
            }
        }
        Some(StaticMemoryRoute::UnsupportedClass(expected)) if expected != meta.op_class => {
            Err(contextualize(MemoryRouteError::StaticMemoryClassMismatch {
                expected,
                got: meta.op_class,
            }))
        }
        _ => Ok(()),
    }
}

/// Flat mode validates the same static operation role as hierarchy replay.  The
/// legacy path deliberately carries an empty sector span, but the SM120 alpha
/// path may retain the already validated NVBit span so its logical global
/// service tail sees the captured request width even when a launch also
/// contains an opaque async/TMA record.
pub(crate) fn flat_global_kind(role: MemRole) -> AccessKind {
    match role {
        MemRole::EligibleLoad(token) => AccessKind::GlobalEligible {
            off: 0,
            len: 0,
            token,
        },
        MemRole::BypassLoad(token) => AccessKind::GlobalBypass {
            off: 0,
            len: 0,
            token,
            l2_requests: 0,
        },
        MemRole::Store => AccessKind::GlobalStore { off: 0, len: 0 },
        MemRole::Reduction => AccessKind::GlobalReduction {
            off: 0,
            len: 0,
            l2_requests: 0,
        },
        MemRole::None => AccessKind::GlobalUntagged { off: 0, len: 0 },
    }
}

#[derive(Clone, Copy)]
pub(crate) struct CompactReplayMemory<'a> {
    pub(crate) memory: &'a ValidatedMemoryInstruction,
    pub(crate) route: ReplayMemoryRoute,
    pub(crate) pc: u32,
}

pub(crate) fn compact_shared_kind(memory: &ValidatedMemoryInstruction, arch: ArchId) -> AccessKind {
    let data_wavefronts = if arch == ArchId::Sm86 {
        memory.shared_sm86_wavefronts_32x4()
    } else {
        memory.shared_bank_wavefronts_32x4()
    };
    let data_ideal_wavefronts = memory.shared_bank_ideal_wavefronts_32x4();
    AccessKind::Shared {
        // Keep scheduler timing on the established whole-warp projection.
        wavefronts: u64::from(memory.shared_bank_legacy_wavefronts_32x4()),
        ideal_wavefronts: u64::from(memory.shared_bank_legacy_ideal_wavefronts_32x4()),
        // SM86 retains the validated legacy observer; SM120 uses the newer
        // phase-local effective observer. Neither changes issue timing.
        data_wavefronts: u64::from(data_wavefronts),
        data_ideal_wavefronts: u64::from(data_ideal_wavefronts),
    }
}

pub(crate) fn classify_compact_flat(
    preflight: Option<CompactReplayMemory<'_>>,
    warp: &ValidatedWarp,
    schedule: &FunctionSchedule,
    arch: ArchId,
    pc: u32,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
    sector_arena: &mut Vec<i64>,
    block_id: [u32; 3],
    tma_observations: Option<&BTreeMap<TmaObservationKey, TmaObservation>>,
) -> Result<AccessKind, ReplayError> {
    let Some(preflight) = preflight else {
        if arch == ArchId::Sm120
            && schedule.issue_slot(u64::from(pc)).is_some_and(|slot| {
                let head = opcode_head(&slot.opcode_text);
                matches!(head, "UTMALDG" | "UTMASTG" | "UBLKCP")
            })
        {
            validate_tma_load_direction(schedule, arch, pc)?;
            // TMA operands are carried by tensor-map/mbarrier sidecars rather
            // than ordinary dynamic address operands in the compact trace.
            // Consume the matching per-instruction payload observation when
            // present. A missing sidecar is explicitly unavailable and carries
            // zero logical completion work; it no longer fabricates 8 KiB.
            let key = TmaObservationKey {
                cta: block_id,
                warp: warp_id,
                instruction: record_ix,
                pc,
                function: warp.function_unique_id(),
            };
            let payload_bytes = tma_observations
                .and_then(|observations| observations.get(&key))
                .and_then(|observation| observation.bytes)
                .unwrap_or(0);
            let payload_work = u32::try_from(payload_bytes).map_err(|_| {
                ReplayError::AsyncCopyCompletionWorkOutOfRange {
                    path: path.to_path_buf(),
                    warp_id,
                    record_ix,
                    completion_work: payload_bytes,
                    max_completion_work: ASYNC_COPY_COMPLETION_VALUE_MASK,
                }
            })?;
            let completion_work = encode_async_copy_payload_work(payload_work).ok_or(
                ReplayError::AsyncCopyCompletionWorkOutOfRange {
                    path: path.to_path_buf(),
                    warp_id,
                    record_ix,
                    completion_work: payload_bytes,
                    max_completion_work: ASYNC_COPY_COMPLETION_VALUE_MASK,
                },
            )?;
            return Ok(AccessKind::GlobalAsyncCopy {
                off: 0,
                len: 0,
                completion_work,
                // TMA tensor-map transfers use the distinct logical bypass
                // service arm in full replay. This is a calibration-domain
                // route, not a physical cache policy claim.
                l1_policy: LdgstsL1Policy::BypassL1,
            });
        }
        return Ok(AccessKind::NonMemory);
    };
    if !preflight.memory.has_effective_lanes() {
        return Ok(AccessKind::PredicatedOffMemory);
    }
    let meta = preflight.memory.replay_meta();
    match (preflight.route, meta.op_class) {
        (ReplayMemoryRoute::Shared, _) => Ok(compact_shared_kind(preflight.memory, arch)),
        (ReplayMemoryRoute::Local, pb::MemoryOpClass::Local) => {
            let Some(direction) = require_known_local_direction(schedule, arch, preflight.pc)?
            else {
                return Ok(AccessKind::NonMemory);
            };
            if arch == ArchId::Sm120 {
                let source_sectors = match warp.sector_plan(preflight.memory) {
                    ValidatedSectorPlan::Sectors(sectors) => sectors,
                    ValidatedSectorPlan::Error { error, .. } => {
                        return Err(deferred_sector_error(error, path, warp_id, record_ix));
                    }
                };
                append_staged_local_sectors(
                    direction,
                    sector_arena,
                    source_sectors,
                    path,
                    warp_id,
                    record_ix,
                )
            } else {
                Ok(match direction {
                    LocalDirection::Load => AccessKind::LocalLoad { off: 0, len: 0 },
                    LocalDirection::Store => AccessKind::LocalStore { off: 0, len: 0 },
                })
            }
        }
        (ReplayMemoryRoute::Local, class) => {
            Err(ReplayError::UnsupportedMemoryPath { arch, class })
        }
        (ReplayMemoryRoute::Unsupported, class) => {
            Err(ReplayError::UnsupportedMemoryPath { arch, class })
        }
        (ReplayMemoryRoute::Global, pb::MemoryOpClass::Global) => {
            let role = require_known_global_role(schedule, arch, preflight.pc)?;
            if arch == ArchId::Sm120 {
                classify_flat_global_sectors(
                    role,
                    warp,
                    preflight.memory,
                    sector_arena,
                    path,
                    warp_id,
                    record_ix,
                )
            } else {
                Ok(flat_global_kind(role))
            }
        }
        (ReplayMemoryRoute::Global, pb::MemoryOpClass::Atomic) => {
            require_sm86_special_role(
                schedule,
                arch,
                preflight.pc,
                preflight.route,
                pb::MemoryOpClass::Atomic,
            )?;
            let role = require_known_global_role(schedule, arch, preflight.pc)?;
            if arch == ArchId::Sm120 {
                classify_flat_global_sectors(
                    role,
                    warp,
                    preflight.memory,
                    sector_arena,
                    path,
                    warp_id,
                    record_ix,
                )
            } else {
                Ok(flat_global_kind(role))
            }
        }
        (ReplayMemoryRoute::Global, pb::MemoryOpClass::AsyncCopy) => {
            if arch == ArchId::Sm120 {
                // Full-replay-only alpha route: preserve an observed async
                // source as a typed completion token instead of collapsing it
                // into an ordinary global instruction.  A compact trace may
                // lack payload/sector evidence; that is zero logical work with
                // an inactive token, not an invented transaction.  This is not
                // a native TMA engine timestamp or queue model.
                let source_sectors = match warp.sector_plan(preflight.memory) {
                    ValidatedSectorPlan::Sectors(sectors) => sectors,
                    ValidatedSectorPlan::Error { error, .. } => {
                        return Err(deferred_sector_error(error, path, warp_id, record_ix));
                    }
                };
                let payload_bytes = warp.effective_data_bytes(preflight.memory);
                let sector_bytes = u64::try_from(source_sectors.len())
                    .ok()
                    .and_then(|count| count.checked_mul(32))
                    .unwrap_or(0);
                let logical_bytes = payload_bytes.max(sector_bytes);
                let completion_work = u32::try_from(logical_bytes)
                    .ok()
                    .and_then(encode_async_copy_payload_work)
                    .ok_or(ReplayError::AsyncCopyCompletionWorkOutOfRange {
                        path: path.to_path_buf(),
                        warp_id,
                        record_ix,
                        completion_work: logical_bytes,
                        max_completion_work: ASYNC_COPY_COMPLETION_VALUE_MASK,
                    })?;
                let kind = checked_staged_async_copy_kind(
                    LdgstsL1Policy::CacheAtAllLevels,
                    sector_arena.len(),
                    source_sectors.len(),
                    completion_work,
                )
                .ok_or(ReplayError::SectorCountOverflow {
                    path: path.to_path_buf(),
                    warp_id,
                    record_ix,
                    sectors: source_sectors.len(),
                })?;
                return append_staged_sectors(
                    Some(kind),
                    sector_arena,
                    source_sectors,
                    path,
                    warp_id,
                    record_ix,
                );
            }
            require_sm86_special_role(
                schedule,
                arch,
                preflight.pc,
                preflight.route,
                pb::MemoryOpClass::AsyncCopy,
            )?;
            require_known_global_role(schedule, arch, preflight.pc)?;
            let l1_policy = require_ldgsts_l1_policy(schedule, arch, preflight.pc)?;
            let source_active = match warp.sector_plan(preflight.memory) {
                ValidatedSectorPlan::Sectors(sectors) => !sectors.is_empty(),
                ValidatedSectorPlan::Error { error, .. } => {
                    return Err(deferred_sector_error(error, path, warp_id, record_ix));
                }
            };
            let payload_bytes = warp.effective_data_bytes(preflight.memory);
            if source_active != (payload_bytes != 0) {
                return Err(ReplayError::AsyncCopyPayloadMismatch {
                    path: path.to_path_buf(),
                    warp_id,
                    record_ix,
                    source_active,
                    sectors: usize::from(source_active),
                    payload_bytes,
                });
            }
            let completion_work = project_async_copy_completion_work(
                arch,
                l1_policy,
                payload_bytes,
                warp.ldgsts_destination_topology_128b_32x4(preflight.memory),
                path,
                warp_id,
                record_ix,
            )?;
            Ok(AccessKind::GlobalAsyncCopy {
                off: 0,
                len: 0,
                completion_work,
                l1_policy,
            })
        }
        (ReplayMemoryRoute::Global, class) => {
            Err(ReplayError::UnsupportedMemoryPath { arch, class })
        }
    }
}

/// Reconstruct exact L2 atomic request topology from the alias-expanded 32-byte
/// sector stream. Each request carries sectors from one 128-byte line; repeated
/// same-word lanes appear as repeated sector ids, so the line requires the
/// maximum sector multiplicity and distinct lines add.
pub(crate) fn atomic_l2_request_count_128b(sectors: &[i64]) -> u16 {
    let mut by_line: BTreeMap<i64, BTreeMap<i64, u16>> = BTreeMap::new();
    for &sector in sectors {
        let count = by_line
            .entry(sector.div_euclid(4))
            .or_default()
            .entry(sector)
            .or_default();
        *count = count
            .checked_add(1)
            .expect("one warp's bounded sector span cannot overflow u16");
    }
    let requests = by_line.values().fold(0_u32, |total, line| {
        total + u32::from(line.values().copied().max().unwrap_or(0))
    });
    u16::try_from(requests).expect("atomic L2 requests cannot exceed the bounded sector span")
}
