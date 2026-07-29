//! Trace → SM replay driver: flattens one traced kernel launch into per-SM
//! block replays using the REAL hardware placement recorded in the trace
//! (per-instruction `sm_id` / `warpid_sm`), joins each dynamic record to
//! the issue schedule (and, when an L1 setup is supplied, to the cache-op
//! manifest + per-record unique-sector expansion), and runs the issue
//! model on every SM.
//!
//! Placement policy v0: the traced placement IS the model — each block
//! runs on the SM the hardware ran it on, each warp on its physical warp
//! slot. Traces without `sm_id` are a typed error (a synthetic placement
//! model is a future knob, not a silent fallback).
//!
//! Memory-record classification (L1 mode): a global-class record takes
//! the [`MemRole`] its pc's SIDECAR OPCODE was classified to at join
//! time (`sim_mem::cache_op` port of the Python token rules) — eligible
//! load → modeled L1 sectors; bypass load → counted bypass; store head →
//! hit-tested without allocating; everything else → unified-denominator
//! count only. Sector expansion follows the `l1_trace_adapter` contract:
//! per-lane `addr>>5 ..= (addr+width-1)>>5`, deduplicated per INSTRUCTION
//! in first-seen order (active lanes are the non-sentinel slots — the
//! canonical positional guarantee).
//!
//! v0 limitation (fail-closed, not silent): every dynamic record of a
//! launch must carry the launch's own `function_unique_id` — traced device
//! function CALLs would need per-segment schedules and are rejected typed
//! until a trace exercises them (none of the 17 M1 traces does).
//!
//! Memory discipline: blocks are decoded ONE at a time and compacted
//! immediately (12-byte records + a per-warp sector arena); the prost tree
//! is dropped before the next block loads.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use sim_arch::ArchId;
use sim_mem::{L1Config, L1CounterError, L1Counters, SectorExpansionError, SectorExpansionPlan};
use sim_trace::lanes::LANE_INACTIVE_SENTINEL;
use sim_trace::pb;
use sim_trace::reader::{
    MemoryAccessMeta, MemoryPayloadContract, PreflightedMemoryAccess,
    check_preflighted_allocation_identity, preflight_memory_access,
};
use sim_trace::{
    KernelLoc, TraceDir, TraceError, block_launch_index, kernel_grid, read_threadblock,
};

use sim_mem::{
    CacheOpToken, L2Config, L2Counters, L2Model, MemRole, is_known_untagged_opcode, opcode_head,
};

use crate::issue::{
    AccessKind, BlockReplay, IssueError, IssueModelParams, ReplayInst, SmSimResult, TailFlags,
    WarpReplay, simulate_sm,
};
use crate::program::FunctionSchedule;
use crate::system::{DramBw, DramStats, L2SourceSplit, SystemError, simulate_kernel_shared};

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
    #[error("joined static opcode requires memory class {expected:?}, got {got:?}")]
    StaticMemoryClassMismatch {
        expected: pb::MemoryOpClass,
        got: pb::MemoryOpClass,
    },
    #[error("static SASS family {family} has no reviewed route on architecture {arch}")]
    StaticMemoryFamilyUnsupported { family: &'static str, arch: ArchId },
    #[error("operand {operand} carries address_space value {raw} unknown to the committed schema")]
    UnknownAddressSpace { operand: usize, raw: i32 },
}

#[derive(Debug, thiserror::Error)]
pub enum ReplayError {
    #[error(transparent)]
    Trace(#[from] TraceError),
    #[error(transparent)]
    Issue(#[from] IssueError),
    #[error(transparent)]
    System(#[from] SystemError),
    #[error("L2 config invalid: {0}")]
    L2Config(#[from] sim_mem::L2ConfigError),
    #[error("L2 construction failed: {0}")]
    L2Build(#[from] sim_mem::L2BuildError),
    #[error("SM {sm_id} L1 counter aggregation failed: {source}")]
    L1Aggregation {
        sm_id: i32,
        #[source]
        source: L1CounterError,
    },
    #[error("SM {sm_id} L1 counter presence mismatch: expected={expected}, found={found}")]
    L1PresenceMismatch {
        sm_id: i32,
        expected: bool,
        found: bool,
    },
    #[error("replay architecture mismatch for {component}: expected {expected}, got {got}")]
    ArchitectureMismatch {
        component: &'static str,
        expected: ArchId,
        got: ArchId,
    },

    #[error(
        "memory class {class:?} has no reviewed replay path for architecture {arch}; the \
         simulator refuses to route it through generic global timing"
    )]
    UnsupportedMemoryPath {
        arch: ArchId,
        class: pb::MemoryOpClass,
    },
    #[error(
        "{path}: warp {warp_id} record {record_ix}: {class:?} operand routing is invalid: {source}"
    )]
    MemoryRoute {
        path: PathBuf,
        warp_id: i32,
        record_ix: usize,
        class: pb::MemoryOpClass,
        #[source]
        source: MemoryRouteError,
    },
    #[error(
        "global memory record at pc {pc:#x} on architecture {arch} has unknown static opcode head {opcode:?}"
    )]
    UnknownGlobalOpcode {
        arch: ArchId,
        pc: u32,
        opcode: String,
    },
    #[error(
        "local memory record at pc {pc:#x} on architecture {arch} has static opcode head {opcode:?}; expected an admitted LDL or STL role"
    )]
    UnknownLocalOpcode {
        arch: ArchId,
        pc: u32,
        opcode: String,
    },
    #[error("no issue schedule for function_unique_id {function_unique_id}")]
    ScheduleMissing { function_unique_id: i32 },
    #[error(
        "{path}: warp {warp_id} record {record_ix} carries function_unique_id {found}, launch \
         kernel is {expected} — multi-function replay is not supported in v0"
    )]
    MultiFunctionUnsupported {
        path: PathBuf,
        warp_id: i32,
        record_ix: usize,
        expected: i32,
        found: i32,
    },
    #[error("{path}: no instruction record carries sm_id — v0 replay requires traced placement")]
    SmIdMissing { path: PathBuf },
    #[error(
        "{path}: sm_id changes within the block ({first} then {second}) — blocks never migrate"
    )]
    SmIdInconsistent {
        path: PathBuf,
        first: i32,
        second: i32,
    },
    #[error("{path}: warp {warp_id} warpid_sm changes within the warp ({first} then {second})")]
    WarpSlotInconsistent {
        path: PathBuf,
        warp_id: i32,
        first: i32,
        second: i32,
    },
    #[error(
        "{path}: warp {warp_id} has no warpid_sm on any record — v0 replay requires the traced \
         physical warp slot (sub-core assignment), like sm_id"
    )]
    WarpSlotMissing { path: PathBuf, warp_id: i32 },
    #[error(
        "{path}: warp {warp_id} record {record_ix}: {sectors} unique sectors in one record \
         overflows the u16 arena index"
    )]
    SectorCountOverflow {
        path: PathBuf,
        warp_id: i32,
        record_ix: usize,
        sectors: usize,
    },
    #[error("{path}: warp {warp_id} record {record_ix}: sector expansion failed: {source}")]
    SectorExpansion {
        path: PathBuf,
        warp_id: i32,
        record_ix: usize,
        #[source]
        source: SectorExpansionError,
    },
    #[error(
        "{path}: warp {warp_id} record {record_ix}: record count overflows u64 at {accumulated}"
    )]
    RecordCountOverflow {
        path: PathBuf,
        warp_id: i32,
        record_ix: usize,
        accumulated: u64,
    },
    #[error("collection '{collection}' length {len} does not fit u64")]
    CollectionLengthOutOfRange {
        collection: &'static str,
        len: usize,
    },
    #[error("SM {sm_id} instruction aggregation {accumulated} + {addend} overflows u64")]
    InstructionCountOverflow {
        sm_id: i32,
        accumulated: u64,
        addend: u64,
    },
    #[error("duplicate block launch index {index} across threadblock files")]
    DuplicateLaunchIndex { index: i64 },
    #[error(
        "issued {issued} != records {records} — replay invariant broken (every traced record \
         must issue exactly once)"
    )]
    CountMismatch { issued: u64, records: u64 },
    #[error("kernel launch grid {grid:?} overflows the u64 block-count domain")]
    GridSizeOverflow { grid: (u32, u32, u32) },
    #[error(
        "E_INCOMPLETE_TRACE: {seen}/{grid_total} launch blocks present and no \
         trace_sampling.json sidecar — an incomplete trace must carry a sampling sidecar"
    )]
    IncompleteTrace { seen: u64, grid_total: u64 },
    #[error(
        "trace_sampling sidecar original_grid {sidecar:?} != launch gridDim {launch:?} \
         (a subset must NEVER rewrite gridDim)"
    )]
    GridDimRewritten {
        launch: (u32, u32, u32),
        sidecar: (u32, u32, u32),
    },
    #[error("records_seen {records_seen} != trace_sampling sidecar records.captured {captured}")]
    RecordsCapturedMismatch { records_seen: u64, captured: u64 },
    #[error(
        "trace_sampling sidecar selected block membership does not match replayed threadblocks \
         ({seen} seen, {declared} declared)"
    )]
    SelectedBlockMembershipMismatch { seen: u64, declared: u64 },
    #[error("trace_sampling sidecar kernel {sidecar:?} != launch kernel {launch:?}")]
    SamplingKernelMismatch { launch: String, sidecar: String },
    #[error("kernel launch block_dim is absent or degenerate")]
    BlockDimMissing,
    #[error("trace_sampling sidecar original_block {sidecar:?} != launch blockDim {launch:?}")]
    BlockDimRewritten {
        launch: (u32, u32, u32),
        sidecar: (u32, u32, u32),
    },
}

/// Replay outcome for one kernel launch.
#[derive(Debug, Clone)]
pub struct KernelReplayResult {
    /// Deployment architecture validated across schedule and replay configuration.
    pub arch: ArchId,
    /// Exact by construction: one issue per traced record (invariant-checked
    /// against `records_seen`).
    pub instructions_issued: u64,
    pub records_seen: u64,
    /// Per traced SM id.
    pub per_sm: BTreeMap<i32, SmSimResult>,
    /// Maximum per-SM issue frontier (`last local issue timestamp + 1`).
    /// This is an issue-only diagnostic, not an NCU cycle prediction.
    pub diagnostic_last_issue_cycles: u64,
    /// Raw timestamp of the last successfully processed shared-system event.
    /// Modes without the shared event system report `None`. This diagnostic is
    /// deliberately not converted into elapsed cycles or a publication metric.
    pub terminal_modeled_event_time: Option<u64>,
    /// Kernel-level L1 counters (sum over per-SM caches) in L1 mode.
    pub l1: Option<L1Counters>,
    /// Shared-L2 counters (the `replay_kernel_shared` path only).
    pub l2: Option<L2Counters>,
    /// Per-source L2 split from the driver-side emission observer.
    pub l2_by_source: Option<L2SourceSplit>,
    /// DRAM read-traffic observer (shared mode only).
    pub dram: Option<DramStats>,
    /// Union of per-SM unmodeled write-class capability tails.
    pub unmodeled_tails: TailFlags,
}

fn checked_grid_total(grid: (u32, u32, u32)) -> Result<u64, ReplayError> {
    u64::from(grid.0)
        .checked_mul(u64::from(grid.1))
        .and_then(|value| value.checked_mul(u64::from(grid.2)))
        .ok_or(ReplayError::GridSizeOverflow { grid })
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

/// Decode one launch into per-SM block replays on the traced placement.
/// `decode_sectors` selects the L1-mode classification (lane decode +
/// unique-sector expansion) over the flat-mode class check.
fn extract_per_sm_blocks(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedule: &FunctionSchedule,
    decode_sectors: bool,
) -> Result<(BTreeMap<i32, Vec<BlockReplay>>, u64), ReplayError> {
    let expected_fuid = schedule.function_unique_id;
    // sm_id → blocks (launch index kept for in-SM ordering).
    let mut per_sm_blocks: BTreeMap<i32, Vec<BlockReplay>> = BTreeMap::new();
    let mut seen_indices: BTreeSet<i64> = BTreeSet::new();
    let mut records_seen: u64 = 0;

    for path in td.threadblock_paths(loc)? {
        let tb = read_threadblock(&path)?;
        let launch_index = block_launch_index(loc.kernel, &tb, &path)?;
        if !seen_indices.insert(launch_index) {
            return Err(ReplayError::DuplicateLaunchIndex {
                index: launch_index,
            });
        }

        let mut block_sm: Option<i32> = None;
        let mut warps = Vec::with_capacity(tb.warps.len());
        for (&warp_id, warp) in &tb.warps {
            let mut warp_slot: Option<i32> = None;
            let mut insts = Vec::with_capacity(warp.instructions.len());
            let mut sectors: Vec<i64> = Vec::new();
            let mut sector_scratch: Vec<i64> = Vec::new();
            for (record_ix, inst) in warp.instructions.iter().enumerate() {
                records_seen = checked_record_count(records_seen, &path, warp_id, record_ix)?;
                // Complete wire and producer-shape preflight on the path the
                // simulator consumes. Structural errors govern route errors;
                // route errors govern the function-identity diagnostic; only a
                // matching function may then prove each GLOBAL/GENERIC active
                // record's resolvable, in-range allocation id. The immutable
                // token is reused by both later passes, so this ordering neither
                // preflight nor allocates an intermediate payload vector.
                if decode_sectors {
                    sector_scratch.clear();
                }
                let preflight = preflight_replay_memory(
                    inst,
                    schedule,
                    td.memory_payload_contract(),
                    &path,
                    warp_id,
                    record_ix,
                )?;
                if inst.function_unique_id != expected_fuid {
                    return Err(ReplayError::MultiFunctionUnsupported {
                        path,
                        warp_id,
                        record_ix,
                        expected: expected_fuid,
                        found: inst.function_unique_id,
                    });
                }
                if let Some(preflight) = preflight {
                    check_preflighted_allocation_identity(
                        preflight.memory,
                        td.allocations(),
                        || format!("{} warp={warp_id} inst={record_ix}", path.display()),
                    )?;
                }
                if let Some(sm) = inst.sm_id {
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
                if let Some(slot) = inst.warpid_sm {
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
                let kind = if decode_sectors {
                    classify_with_l1_preflighted(
                        preflight,
                        schedule,
                        schedule.arch,
                        &mut sectors,
                        &mut sector_scratch,
                        &path,
                        warp_id,
                        record_ix,
                    )?
                } else {
                    classify_flat_preflighted(preflight, schedule, schedule.arch)?
                };
                insts.push(ReplayInst { pc: inst.pc, kind });
            }
            // Fail-closed traced placement: a warp that issued records
            // without ever exposing its physical slot is a typed error,
            // exactly like a block without sm_id (a synthetic placement
            // model is a future knob, never a silent fallback). Warps
            // with NO records carry no placement and issue nothing.
            let warp_slot = match warp_slot {
                Some(slot) => slot,
                None if insts.is_empty() => warp_id,
                None => {
                    return Err(ReplayError::WarpSlotMissing { path, warp_id });
                }
            };
            warps.push(WarpReplay {
                warp_slot,
                insts,
                sectors,
            });
        }
        let Some(sm) = block_sm else {
            return Err(ReplayError::SmIdMissing { path });
        };
        per_sm_blocks.entry(sm).or_default().push(BlockReplay {
            launch_index,
            warps,
        });
    }
    // Completeness gate (fail-closed) on the path replay ACTUALLY consumes: a
    // subset trace (coverage < grid) is admissible ONLY with a valid
    // trace_sampling.json sidecar whose original_grid matches the launch record
    // (gridDim is never rewritten) and whose captured record count matches
    // records_seen. Absent/non-sampling sidecars never waive full coverage.
    let (gx, gy, gz) = kernel_grid(loc.kernel)?;
    let grid_total = checked_grid_total((gx, gy, gz))?;
    let seen = checked_collection_len("replayed block membership", seen_indices.len())?;
    match td.sampling_sidecar() {
        None => {
            if seen < grid_total {
                return Err(ReplayError::IncompleteTrace { seen, grid_total });
            }
        }
        Some(sc) => {
            let block = loc
                .kernel
                .block_dim
                .as_ref()
                .filter(|dim| dim.x > 0 && dim.y > 0 && dim.z > 0)
                .ok_or(ReplayError::BlockDimMissing)?;
            let launch_block = (block.x as u32, block.y as u32, block.z as u32);
            if sc.kernel_mangled_name != loc.kernel.name {
                return Err(ReplayError::SamplingKernelMismatch {
                    launch: loc.kernel.name.clone(),
                    sidecar: sc.kernel_mangled_name.clone(),
                });
            }
            if sc.original_grid != (gx, gy, gz) {
                return Err(ReplayError::GridDimRewritten {
                    launch: (gx, gy, gz),
                    sidecar: sc.original_grid,
                });
            }
            if sc.original_block != launch_block {
                return Err(ReplayError::BlockDimRewritten {
                    launch: launch_block,
                    sidecar: sc.original_block,
                });
            }
            if sc.sanctions_incomplete && sc.selected_block_ids.as_ref() != Some(&seen_indices) {
                let declared = match &sc.selected_block_ids {
                    Some(blocks) => checked_collection_len(
                        "trace_sampling selected block membership",
                        blocks.len(),
                    )?,
                    None => 0,
                };
                return Err(ReplayError::SelectedBlockMembershipMismatch { seen, declared });
            }
            if seen < grid_total && !sc.sanctions_incomplete {
                return Err(ReplayError::IncompleteTrace { seen, grid_total });
            }
            if records_seen != sc.captured_records {
                return Err(ReplayError::RecordsCapturedMismatch {
                    records_seen,
                    captured: sc.captured_records,
                });
            }
        }
    }
    for blocks in per_sm_blocks.values_mut() {
        blocks.sort_by_key(|b| b.launch_index);
    }
    Ok((per_sm_blocks, records_seen))
}

/// Fold per-SM results into the kernel-level view, enforcing the
/// one-issue-per-record invariant.
fn fold_kernel_result(
    arch: ArchId,
    per_sm: BTreeMap<i32, SmSimResult>,
    records_seen: u64,
    expect_l1: bool,
    terminal_modeled_event_time: Option<u64>,
    l2: Option<(L2Counters, L2SourceSplit, DramStats)>,
) -> Result<KernelReplayResult, ReplayError> {
    let mut instructions_issued: u64 = 0;
    let mut diagnostic_last_issue_cycles: u64 = 0;
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
    let (l2, l2_by_source, dram) = match l2 {
        Some((counters, split, dram)) => (Some(counters), Some(split), Some(dram)),
        None => (None, None, None),
    };
    Ok(KernelReplayResult {
        arch,
        instructions_issued,
        records_seen,
        per_sm,
        diagnostic_last_issue_cycles,
        terminal_modeled_event_time,
        l1: l1_totals,
        l2,
        l2_by_source,
        dram,
        unmodeled_tails,
    })
}

fn lookup_schedule<'a>(
    schedules: &'a BTreeMap<i32, FunctionSchedule>,
    loc: &KernelLoc<'_>,
) -> Result<&'a FunctionSchedule, ReplayError> {
    let expected_fuid = loc.kernel.function_unique_id;
    schedules
        .get(&expected_fuid)
        .ok_or(ReplayError::ScheduleMissing {
            function_unique_id: expected_fuid,
        })
}

/// Memory-hierarchy mode for one kernel replay — the ONE typed selector
/// every harness goes through (a pair of optional configs would let a
/// plumbing slip degrade a cell to the wrong mode silently).
#[derive(Debug, Clone, Copy)]
pub enum ReplayMemMode {
    /// No cache model (flat global latency).
    Flat,
    /// Per-SM L1s with constant fill latency below them (the pre-L2
    /// lockstep/ablation surface).
    L1Const(L1Config),
    /// Per-SM L1s over ONE shared L2 (the full hierarchy): L1 misses
    /// priced L2-hit vs DRAM by the shared state through the event
    /// system.
    L1SharedL2 {
        l1: L1Config,
        l2: L2Config,
        dram: DramBw,
    },
}

fn require_replay_arch(
    expected: ArchId,
    got: ArchId,
    component: &'static str,
) -> Result<(), ReplayError> {
    if got != expected {
        return Err(ReplayError::ArchitectureMismatch {
            component,
            expected,
            got,
        });
    }
    Ok(())
}

fn validate_replay_architecture(
    schedule: &FunctionSchedule,
    params: &IssueModelParams,
    mode: ReplayMemMode,
) -> Result<(), ReplayError> {
    let arch = schedule.arch;
    require_replay_arch(arch, params.arch, "issue parameters")?;
    match mode {
        ReplayMemMode::Flat => {}
        ReplayMemMode::L1Const(l1) => {
            require_replay_arch(arch, l1.arch, "L1 configuration")?;
        }
        ReplayMemMode::L1SharedL2 { l1, l2, dram } => {
            require_replay_arch(arch, l1.arch, "L1 configuration")?;
            require_replay_arch(arch, l2.arch, "L2 configuration")?;
            require_replay_arch(arch, dram.arch(), "DRAM configuration")?;
        }
    }
    Ok(())
}

/// Replay one kernel launch on its traced placement under `mode`.
/// Caches are FRESH per kernel (cold, NCU `--cache-control all` parity).
pub fn replay_kernel(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedules: &BTreeMap<i32, FunctionSchedule>,
    params: &IssueModelParams,
    mode: ReplayMemMode,
) -> Result<KernelReplayResult, ReplayError> {
    let schedule = lookup_schedule(schedules, loc)?;
    validate_replay_architecture(schedule, params, mode)?;
    let decode_sectors = !matches!(mode, ReplayMemMode::Flat);
    let (per_sm_blocks, records_seen) = extract_per_sm_blocks(td, loc, schedule, decode_sectors)?;
    match mode {
        ReplayMemMode::Flat | ReplayMemMode::L1Const(_) => {
            let l1_config = match mode {
                ReplayMemMode::L1Const(cfg) => Some(cfg),
                _ => None,
            };
            let mut per_sm = BTreeMap::new();
            for (sm, blocks) in &per_sm_blocks {
                per_sm.insert(*sm, simulate_sm(blocks, schedule, params, l1_config)?);
            }
            fold_kernel_result(
                schedule.arch,
                per_sm,
                records_seen,
                l1_config.is_some(),
                None,
                None,
            )
        }
        ReplayMemMode::L1SharedL2 { l1, l2, dram } => {
            let mut l2 = L2Model::new(l2)?;
            let sys = simulate_kernel_shared(&per_sm_blocks, schedule, params, l1, &mut l2, dram)?;
            require_replay_arch(schedule.arch, sys.arch, "system result")?;
            fold_kernel_result(
                schedule.arch,
                sys.per_sm,
                records_seen,
                true,
                sys.terminal_event_time,
                Some((sys.l2, sys.l2_by_source, sys.dram)),
            )
        }
    }
}

/// Resolve the two special Global routes reviewed on the active sm_86 path.
/// The dynamic enum alone is not enough: the static sidecar must prove a
/// Global ATOM/RED family or LDGSTS role before the record enters replay.
fn require_sm86_special_role(
    schedule: &FunctionSchedule,
    arch: ArchId,
    pc: u32,
    route: ReplayMemoryRoute,
    class: pb::MemoryOpClass,
) -> Result<(), ReplayError> {
    if arch != ArchId::Sm86 {
        return Err(ReplayError::UnsupportedMemoryPath { arch, class });
    }
    let slot = schedule
        .by_pc
        .get(&u64::from(pc))
        .ok_or(ReplayError::UnsupportedMemoryPath { arch, class })?;
    let head = opcode_head(&slot.opcode_text);
    let valid = match (route, class) {
        (ReplayMemoryRoute::Global, pb::MemoryOpClass::Atomic) => match slot.mem_role {
            MemRole::BypassLoad(CacheOpToken::Atom) => matches!(head, "ATOM" | "ATOMG"),
            MemRole::Reduction => head == "RED",
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

/// Return the schedule's memory role while rejecting a dynamic global record
/// whose static sidecar opcode is neither classified nor one of the deliberate
/// untagged load-family heads. A pc absent from the schedule stays neutral here
/// so the issue loop reports its more specific `PcNotInSchedule` error.
fn require_known_global_role(
    schedule: &FunctionSchedule,
    arch: ArchId,
    pc: u32,
) -> Result<MemRole, ReplayError> {
    let Some(slot) = schedule.by_pc.get(&u64::from(pc)) else {
        return Ok(MemRole::None);
    };
    if slot.mem_role == MemRole::None && !is_known_untagged_opcode(&slot.opcode_text) {
        return Err(ReplayError::UnknownGlobalOpcode {
            arch,
            pc,
            opcode: opcode_head(&slot.opcode_text).to_owned(),
        });
    }
    Ok(slot.mem_role)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LocalDirection {
    Load,
    Store,
}

/// Resolve Local traffic only from the joined LDL/STL sidecar role. A missing pc
/// remains neutral so the issue loop retains its dedicated `PcNotInSchedule`
/// diagnostic; every present non-LDL/STL role fails closed.
fn require_known_local_direction(
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReplayMemoryRoute {
    Global,
    Local,
    Shared,
    Unsupported,
}

impl ReplayMemoryRoute {
    const fn canonical_space(self) -> pb::MemorySpace {
        match self {
            Self::Global => pb::MemorySpace::Global,
            Self::Local => pb::MemorySpace::Local,
            Self::Shared => pb::MemorySpace::Shared,
            Self::Unsupported => pb::MemorySpace::Unspecified,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ReplayMemoryPreflight<'a> {
    memory: PreflightedMemoryAccess<'a>,
    route: ReplayMemoryRoute,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StaticMemoryRoute {
    Modeled {
        route: ReplayMemoryRoute,
        class: pb::MemoryOpClass,
    },
    UnsupportedClass(pb::MemoryOpClass),
}

fn opcode_has_qualifier(opcode_text: &str, qualifier: &str) -> bool {
    opcode_text
        .split_whitespace()
        .next()
        .unwrap_or("")
        .split('.')
        .skip(1)
        .any(|part| part == qualifier)
}

/// Space-coded SASS heads whose storage class is part of the mnemonic rather
/// than inferred from the dynamic record. ATOM and RED are the legacy Global
/// forms; LDGSTS is routed by its traced Global source operand. Hopper-era REDG,
/// LDGMC, DSMEM, and shared-matrix families are architecture-gated before any
/// record can borrow a modeled route.
fn static_memory_route(
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
        "ATOM" | "ATOMG" | "RED" => Ok(Some(StaticMemoryRoute::Modeled {
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
        // Hopper's reducing-load family has no reviewed replay semantics on
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

/// Reject disagreement between a known space-coded static SASS family and the
/// class/route resolved from the dynamic contract. This runs before allocation
/// identity and classification, so a contradictory record cannot borrow another
/// operation or storage class's timing and traffic semantics.
fn validate_static_memory_route(
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
        Some(StaticMemoryRoute::UnsupportedClass(expected)) if expected != meta.op_class => {
            Err(contextualize(MemoryRouteError::StaticMemoryClassMismatch {
                expected,
                got: meta.op_class,
            }))
        }
        _ => Ok(()),
    }
}

fn validate_replay_access_kind(
    meta: MemoryAccessMeta,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<(), ReplayError> {
    let Some(got) = meta.access_kind else {
        return Ok(()); // additive legacy field: absence remains admissible
    };
    let (valid, expected) = match meta.op_class {
        pb::MemoryOpClass::Global | pb::MemoryOpClass::Shared | pb::MemoryOpClass::Local => (
            matches!(
                got,
                pb::MemoryOpAccessKind::Load | pb::MemoryOpAccessKind::Store
            ),
            "LOAD or STORE",
        ),
        pb::MemoryOpClass::Atomic => (got == pb::MemoryOpAccessKind::Atomic, "ATOMIC"),
        pb::MemoryOpClass::AsyncCopy => (got == pb::MemoryOpAccessKind::AsyncCopy, "ASYNC_COPY"),
        // These routes fail as unsupported before replay. Do not invent a
        // direction contract for a class with no modeled path.
        pb::MemoryOpClass::Unspecified
        | pb::MemoryOpClass::Texture
        | pb::MemoryOpClass::Surface
        | pb::MemoryOpClass::Tma
        | pb::MemoryOpClass::ClusterDsmem
        | pb::MemoryOpClass::Tmem => return Ok(()),
    };
    if valid {
        return Ok(());
    }
    Err(ReplayError::MemoryRoute {
        path: path.to_path_buf(),
        warp_id,
        record_ix,
        class: meta.op_class,
        source: MemoryRouteError::AccessKindInvalid { got, expected },
    })
}

fn validate_static_access_kind(
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

/// Resolve the replay route from the trace's validated payload-generation
/// contract. Address-carrying v1 records enforce the live producer's exact
/// operand shape; capability-empty legacy traces retain the pre-transition
/// ATOMIC/ASYNC_COPY fallback. GLOBAL retains its documented absent-space
/// fallback in both generations.
fn replay_memory_route(
    inst: &pb::Instruction,
    class: pb::MemoryOpClass,
    payload_contract: MemoryPayloadContract,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<ReplayMemoryRoute, ReplayError> {
    let contextualize = |source| ReplayError::MemoryRoute {
        path: path.to_path_buf(),
        warp_id,
        record_ix,
        class,
        source,
    };
    let decode_space = |operand: usize, address: &pb::Address| {
        address
            .address_space
            .map(|raw| {
                pb::MemorySpace::try_from(raw).map_err(|_| {
                    contextualize(MemoryRouteError::UnknownAddressSpace { operand, raw })
                })
            })
            .transpose()
    };

    let zero_effective_mask = matches!(
        (inst.active_mask, inst.predicate_mask),
        (Some(active), Some(predicate)) if active & predicate == 0
    );

    match class {
        pb::MemoryOpClass::Unspecified => Ok(ReplayMemoryRoute::Unsupported),
        pb::MemoryOpClass::Shared | pb::MemoryOpClass::Local => {
            if inst.addresses.is_empty() && !zero_effective_mask {
                return Err(contextualize(MemoryRouteError::SharedLocalOperandCount {
                    got: 0,
                }));
            }
            let expected_space = match class {
                pb::MemoryOpClass::Shared => pb::MemorySpace::Shared,
                pb::MemoryOpClass::Local => pb::MemorySpace::Local,
                _ => unreachable!("match arm admits only SHARED or LOCAL"),
            };
            for (operand, address) in inst.addresses.iter().enumerate() {
                if let Some(space) = decode_space(operand, address)? {
                    if space != expected_space {
                        return Err(contextualize(MemoryRouteError::SharedLocalSpaceInvalid {
                            operand,
                            space,
                        }));
                    }
                }
            }
            Ok(match class {
                pb::MemoryOpClass::Shared => ReplayMemoryRoute::Shared,
                pb::MemoryOpClass::Local => ReplayMemoryRoute::Local,
                _ => unreachable!("match arm admits only SHARED or LOCAL"),
            })
        }
        pb::MemoryOpClass::Global => {
            if inst.addresses.is_empty() && !zero_effective_mask {
                return Err(contextualize(MemoryRouteError::GlobalOperandCount {
                    got: 0,
                }));
            }
            for (operand, address) in inst.addresses.iter().enumerate() {
                if let Some(space) = decode_space(operand, address)? {
                    if !matches!(space, pb::MemorySpace::Global | pb::MemorySpace::Generic) {
                        return Err(contextualize(MemoryRouteError::GlobalSpaceInvalid {
                            operand,
                            space,
                        }));
                    }
                }
            }
            Ok(ReplayMemoryRoute::Global)
        }
        pb::MemoryOpClass::Atomic => {
            if payload_contract == MemoryPayloadContract::LegacyExclusion {
                let mut resolved = None;
                for (operand, address) in inst.addresses.iter().enumerate() {
                    let Some(space) = decode_space(operand, address)? else {
                        continue;
                    };
                    let candidate = match space {
                        pb::MemorySpace::Global | pb::MemorySpace::Generic => {
                            ReplayMemoryRoute::Global
                        }
                        pb::MemorySpace::Shared => ReplayMemoryRoute::Shared,
                        pb::MemorySpace::Local => ReplayMemoryRoute::Local,
                        pb::MemorySpace::Unspecified => {
                            return Err(contextualize(MemoryRouteError::AtomicSpaceUnspecified {
                                operand,
                            }));
                        }
                    };
                    match resolved {
                        Some(previous) if previous != candidate => {
                            return Err(contextualize(MemoryRouteError::AtomicSpacesConflict));
                        }
                        None => resolved = Some(candidate),
                        Some(_) => {}
                    }
                }
                return Ok(resolved.unwrap_or(ReplayMemoryRoute::Global));
            }
            // A no-payload zero-active legacy record retains the old global
            // fallback. Every active or address-carrying atomic must carry the
            // producer's single explicit-space memory operand.
            if inst.addresses.is_empty() && zero_effective_mask {
                return Ok(ReplayMemoryRoute::Global);
            }
            if inst.addresses.len() != 1 {
                return Err(contextualize(MemoryRouteError::AtomicOperandCount {
                    got: inst.addresses.len(),
                }));
            }
            let Some(space) = decode_space(0, &inst.addresses[0])? else {
                return Err(contextualize(MemoryRouteError::AtomicSpaceMissing {
                    operand: 0,
                }));
            };
            match space {
                pb::MemorySpace::Global | pb::MemorySpace::Generic => Ok(ReplayMemoryRoute::Global),
                pb::MemorySpace::Shared => Ok(ReplayMemoryRoute::Shared),
                pb::MemorySpace::Local => Ok(ReplayMemoryRoute::Local),
                pb::MemorySpace::Unspecified => {
                    Err(contextualize(MemoryRouteError::AtomicSpaceUnspecified {
                        operand: 0,
                    }))
                }
            }
        }
        pb::MemoryOpClass::AsyncCopy => {
            if payload_contract == MemoryPayloadContract::LegacyExclusion {
                // Legacy payload capability skips the new one-source shape, but
                // an explicit space still cannot contradict ASYNC_COPY's global
                // source. Missing space retains the historical global fallback.
                for (operand, address) in inst.addresses.iter().enumerate() {
                    if let Some(space) = decode_space(operand, address)? {
                        if !matches!(space, pb::MemorySpace::Global | pb::MemorySpace::Generic) {
                            return Err(contextualize(
                                MemoryRouteError::AsyncCopySourceSpaceInvalid { space },
                            ));
                        }
                    }
                }
                return Ok(ReplayMemoryRoute::Global);
            }
            if inst.addresses.len() != 1 {
                return Err(contextualize(MemoryRouteError::AsyncCopySourceCount {
                    got: inst.addresses.len(),
                }));
            }
            let Some(space) = decode_space(0, &inst.addresses[0])? else {
                return Err(contextualize(MemoryRouteError::AsyncCopySourceSpaceMissing));
            };
            if !matches!(space, pb::MemorySpace::Global | pb::MemorySpace::Generic) {
                return Err(contextualize(
                    MemoryRouteError::AsyncCopySourceSpaceInvalid { space },
                ));
            }
            Ok(ReplayMemoryRoute::Global)
        }
        pb::MemoryOpClass::Texture
        | pb::MemoryOpClass::Surface
        | pb::MemoryOpClass::Tma
        | pb::MemoryOpClass::ClusterDsmem
        | pb::MemoryOpClass::Tmem => Ok(ReplayMemoryRoute::Unsupported),
    }
}

fn preflight_replay_memory<'a>(
    inst: &'a pb::Instruction,
    schedule: &FunctionSchedule,
    payload_contract: MemoryPayloadContract,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<Option<ReplayMemoryPreflight<'a>>, ReplayError> {
    let record = || format!("{} warp={warp_id} inst={record_ix}", path.display());
    let memory = match preflight_memory_access(inst, record) {
        Ok(Some(memory)) => memory,
        Ok(None) => return Ok(None),
        Err(error) => {
            // The reader independently rejects class/space contradictions. Route
            // the same fully-structural-preflighted record through replay's own
            // shape-aware defense so operand-count diagnostics retain precedence
            // over a later conflicting operand.
            if let TraceError::MemoryOpClassAddressSpaceConflict { class, .. } = &error {
                if let Err(route_error) =
                    replay_memory_route(inst, *class, payload_contract, path, warp_id, record_ix)
                {
                    return Err(route_error);
                }
            }
            return Err(error.into());
        }
    };
    let meta = memory.meta();
    let route = replay_memory_route(
        memory.instruction(),
        meta.op_class,
        payload_contract,
        path,
        warp_id,
        record_ix,
    )?;
    validate_replay_access_kind(meta, path, warp_id, record_ix)?;
    // Static direction is meaningful only after the dynamic record identifies
    // this exact schedule. A foreign-function record retains the dedicated
    // MultiFunctionUnsupported diagnostic after structural/route validation.
    if inst.function_unique_id == schedule.function_unique_id {
        validate_static_memory_route(meta, route, schedule, inst.pc, path, warp_id, record_ix)?;
        validate_static_access_kind(meta, schedule, inst.pc, path, warp_id, record_ix)?;
    }
    Ok(Some(ReplayMemoryPreflight { memory, route }))
}

/// Flat-mode classification validates the complete memory wire contract without
/// expanding sectors, then applies the same producer-shape route gate as L1.
///
/// Flat mode preserves the joined static operation role with an empty sector
/// span. No L1 counters exist in this mode; the role remains load-bearing for
/// Store/ATOM/RED timing and unmodeled-tail diagnostics.
#[cfg(test)]
fn classify_flat(
    inst: &pb::Instruction,
    schedule: &FunctionSchedule,
    arch: ArchId,
    path: &std::path::Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<AccessKind, ReplayError> {
    classify_flat_with_contract(
        inst,
        schedule,
        arch,
        MemoryPayloadContract::AddressCarryingV1,
        path,
        warp_id,
        record_ix,
    )
}

#[cfg(test)]
fn classify_flat_with_contract(
    inst: &pb::Instruction,
    schedule: &FunctionSchedule,
    arch: ArchId,
    payload_contract: MemoryPayloadContract,
    path: &std::path::Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<AccessKind, ReplayError> {
    let preflight =
        preflight_replay_memory(inst, schedule, payload_contract, path, warp_id, record_ix)?;
    classify_flat_preflighted(preflight, schedule, arch)
}

/// Flat mode validates the same static operation role as hierarchy replay but
/// deliberately carries an empty sector span. This preserves Store/ATOM/RED
/// semantics for timing and capability diagnostics without decoding addresses.
fn flat_global_kind(role: MemRole) -> AccessKind {
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
        },
        MemRole::Store => AccessKind::GlobalStore { off: 0, len: 0 },
        MemRole::Reduction => AccessKind::GlobalReduction { off: 0, len: 0 },
        MemRole::None => AccessKind::GlobalUntagged { off: 0, len: 0 },
    }
}

fn classify_flat_preflighted(
    preflight: Option<ReplayMemoryPreflight<'_>>,
    schedule: &FunctionSchedule,
    arch: ArchId,
) -> Result<AccessKind, ReplayError> {
    let Some(preflight) = preflight else {
        return Ok(AccessKind::NonMemory);
    };
    let inst = preflight.memory.instruction();
    let meta = preflight.memory.meta();
    match (preflight.route, meta.op_class) {
        (ReplayMemoryRoute::Shared, _) => Ok(AccessKind::Shared),
        (ReplayMemoryRoute::Local, pb::MemoryOpClass::Local) => Ok(
            match require_known_local_direction(schedule, arch, inst.pc)? {
                Some(LocalDirection::Load) => AccessKind::LocalLoad { off: 0, len: 0 },
                Some(LocalDirection::Store) => AccessKind::LocalStore { off: 0, len: 0 },
                None => AccessKind::NonMemory,
            },
        ),
        (ReplayMemoryRoute::Local, class) => {
            Err(ReplayError::UnsupportedMemoryPath { arch, class })
        }
        (ReplayMemoryRoute::Unsupported, class) => {
            Err(ReplayError::UnsupportedMemoryPath { arch, class })
        }
        (ReplayMemoryRoute::Global, pb::MemoryOpClass::Global) => {
            let role = require_known_global_role(schedule, arch, inst.pc)?;
            Ok(flat_global_kind(role))
        }
        (
            ReplayMemoryRoute::Global,
            class @ (pb::MemoryOpClass::Atomic | pb::MemoryOpClass::AsyncCopy),
        ) => {
            require_sm86_special_role(schedule, arch, inst.pc, preflight.route, class)?;
            let role = require_known_global_role(schedule, arch, inst.pc)?;
            Ok(flat_global_kind(role))
        }
        (ReplayMemoryRoute::Global, class) => {
            Err(ReplayError::UnsupportedMemoryPath { arch, class })
        }
    }
}

fn checked_staged_global_kind(role: MemRole, start: usize, count: usize) -> Option<AccessKind> {
    let len = u16::try_from(count).ok()?;
    start.checked_add(count)?;
    Some(match role {
        MemRole::EligibleLoad(token) => AccessKind::GlobalEligible {
            off: start,
            len,
            token,
        },
        MemRole::BypassLoad(token) => AccessKind::GlobalBypass {
            off: start,
            len,
            token,
        },
        MemRole::Store => AccessKind::GlobalStore { off: start, len },
        MemRole::Reduction => AccessKind::GlobalReduction { off: start, len },
        MemRole::None => AccessKind::GlobalUntagged { off: start, len },
    })
}

fn checked_staged_local_kind(
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

fn append_staged_sectors(
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

fn append_staged_global_sectors(
    role: MemRole,
    sectors: &mut Vec<i64>,
    sector_scratch: &[i64],
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<AccessKind, ReplayError> {
    let start = sectors.len();
    append_staged_sectors(
        checked_staged_global_kind(role, start, sector_scratch.len()),
        sectors,
        sector_scratch,
        path,
        warp_id,
        record_ix,
    )
}

fn append_staged_local_sectors(
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

/// Full classification in L1 mode: wire-contract decode (streaming form),
/// per-instruction unique-sector expansion in reusable private scratch, and
/// the join-time [`MemRole`] split. The warp arena is extended only after
/// decoding, expansion, routing, and index narrowing all succeed.
#[cfg(test)]
fn classify_with_l1(
    inst: &pb::Instruction,
    schedule: &FunctionSchedule,
    arch: ArchId,
    sectors: &mut Vec<i64>,
    sector_scratch: &mut Vec<i64>,
    path: &std::path::Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<AccessKind, ReplayError> {
    classify_with_l1_with_contract(
        inst,
        schedule,
        arch,
        MemoryPayloadContract::AddressCarryingV1,
        sectors,
        sector_scratch,
        path,
        warp_id,
        record_ix,
    )
}

#[cfg(test)]
fn classify_with_l1_with_contract(
    inst: &pb::Instruction,
    schedule: &FunctionSchedule,
    arch: ArchId,
    payload_contract: MemoryPayloadContract,
    sectors: &mut Vec<i64>,
    sector_scratch: &mut Vec<i64>,
    path: &std::path::Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<AccessKind, ReplayError> {
    sector_scratch.clear();
    let preflight =
        preflight_replay_memory(inst, schedule, payload_contract, path, warp_id, record_ix)?;
    classify_with_l1_preflighted(
        preflight,
        schedule,
        arch,
        sectors,
        sector_scratch,
        path,
        warp_id,
        record_ix,
    )
}

fn classify_with_l1_preflighted(
    preflight: Option<ReplayMemoryPreflight<'_>>,
    schedule: &FunctionSchedule,
    arch: ArchId,
    sectors: &mut Vec<i64>,
    sector_scratch: &mut Vec<i64>,
    path: &std::path::Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<AccessKind, ReplayError> {
    let Some(preflight) = preflight else {
        return Ok(AccessKind::NonMemory);
    };
    let inst = preflight.memory.instruction();
    let meta = preflight.memory.meta();
    let mut sector_plan = SectorExpansionPlan::new();
    let mut sector_error: Option<SectorExpansionError> = None;
    if matches!(
        preflight.route,
        ReplayMemoryRoute::Global | ReplayMemoryRoute::Local
    ) {
        preflight.memory.visit_addresses(|access| {
            if sector_error.is_some() {
                return;
            }
            if let Err(source) = sector_plan.include_record(
                &access.lane_vector,
                (0..access.lane_vector.len())
                    .filter(|&i| access.lane_vector[i] != LANE_INACTIVE_SENTINEL),
                access.data_width_bytes,
            ) {
                sector_error = Some(source);
            }
        });
    }
    if let Some(source) = sector_error {
        return Err(ReplayError::SectorExpansion {
            path: path.to_path_buf(),
            warp_id,
            record_ix,
            source,
        });
    }
    enum StagedHierarchyRole {
        Global(MemRole),
        Local(LocalDirection),
    }

    let staged_role = match (preflight.route, meta.op_class) {
        (ReplayMemoryRoute::Shared, _) => return Ok(AccessKind::Shared),
        (ReplayMemoryRoute::Unsupported, class) => {
            return Err(ReplayError::UnsupportedMemoryPath { arch, class });
        }
        (ReplayMemoryRoute::Local, pb::MemoryOpClass::Local) => {
            let Some(direction) = require_known_local_direction(schedule, arch, inst.pc)? else {
                return Ok(AccessKind::NonMemory);
            };
            StagedHierarchyRole::Local(direction)
        }
        (ReplayMemoryRoute::Local, class) => {
            return Err(ReplayError::UnsupportedMemoryPath { arch, class });
        }
        (ReplayMemoryRoute::Global, pb::MemoryOpClass::Global) => {
            StagedHierarchyRole::Global(require_known_global_role(schedule, arch, inst.pc)?)
        }
        (
            ReplayMemoryRoute::Global,
            class @ (pb::MemoryOpClass::Atomic | pb::MemoryOpClass::AsyncCopy),
        ) => {
            require_sm86_special_role(schedule, arch, inst.pc, preflight.route, class)?;
            StagedHierarchyRole::Global(require_known_global_role(schedule, arch, inst.pc)?)
        }
        (ReplayMemoryRoute::Global, class) => {
            return Err(ReplayError::UnsupportedMemoryPath { arch, class });
        }
    };
    // A pc absent from the schedule surfaces as the issue loop's typed
    // PcNotInSchedule; classification stays neutral in that case. A present
    // but unrecognized static opcode fails closed instead of becoming a
    // generic untagged access. Static-role resolution follows exact address-
    // expansion errors but precedes sector-count narrowing, matching the
    // established global-route precedence without masking arithmetic faults.
    let prepared = sector_plan
        .prepare()
        .map_err(|source| ReplayError::SectorExpansion {
            path: path.to_path_buf(),
            warp_id,
            record_ix,
            source,
        })?;
    let count = prepared.unique_count();
    if count > usize::from(u16::MAX) {
        return Err(ReplayError::SectorCountOverflow {
            path: path.to_path_buf(),
            warp_id,
            record_ix,
            sectors: count,
        });
    }
    *sector_scratch = prepared
        .expand_bounded(usize::from(u16::MAX))
        .map_err(|source| ReplayError::SectorExpansion {
            path: path.to_path_buf(),
            warp_id,
            record_ix,
            source,
        })?;
    match staged_role {
        StagedHierarchyRole::Global(role) => {
            append_staged_global_sectors(role, sectors, sector_scratch, path, warp_id, record_ix)
        }
        StagedHierarchyRole::Local(direction) => append_staged_local_sectors(
            direction,
            sectors,
            sector_scratch,
            path,
            warp_id,
            record_ix,
        ),
    }
}

#[cfg(test)]
mod aggregation_tests {
    use super::*;

    fn sm(instructions_issued: u64, l1: Option<L1Counters>) -> SmSimResult {
        SmSimResult {
            instructions_issued,
            cycles: 1,
            l1,
            unmodeled_tails: TailFlags::default(),
        }
    }

    #[test]
    fn record_count_overflow_preserves_context() {
        let path = Path::new("block_7.msgpack");
        assert!(matches!(
            checked_record_count(u64::MAX, path, 3, 11),
            Err(ReplayError::RecordCountOverflow {
                path: got,
                warp_id: 3,
                record_ix: 11,
                accumulated: u64::MAX,
            }) if got == path
        ));
    }

    #[test]
    fn instruction_overflow_precedes_count_mismatch() {
        let per_sm = BTreeMap::from([(0, sm(u64::MAX, None)), (1, sm(1, None))]);
        assert!(matches!(
            fold_kernel_result(ArchId::Sm86, per_sm, 0, false, None, None),
            Err(ReplayError::InstructionCountOverflow {
                sm_id: 1,
                accumulated: u64::MAX,
                addend: 1,
            })
        ));
    }

    #[test]
    fn ordinary_count_mismatch_remains_typed() {
        let per_sm = BTreeMap::from([(0, sm(4, None))]);
        assert!(matches!(
            fold_kernel_result(ArchId::Sm86, per_sm, 5, false, None, None),
            Err(ReplayError::CountMismatch {
                issued: 4,
                records: 5,
            })
        ));
    }

    #[test]
    fn issue_and_terminal_event_diagnostics_remain_independent() {
        let mut result = sm(1, None);
        result.cycles = 7;
        let folded = fold_kernel_result(
            ArchId::Sm86,
            BTreeMap::from([(0, result)]),
            1,
            false,
            Some(423),
            None,
        )
        .unwrap();
        assert_eq!(folded.diagnostic_last_issue_cycles, 7);
        assert_eq!(folded.terminal_modeled_event_time, Some(423));
    }

    #[test]
    fn first_l1_snapshot_must_have_matching_provenance() {
        let unbound = BTreeMap::from([(0, sm(1, Some(L1Counters::default())))]);
        assert!(matches!(
            fold_kernel_result(ArchId::Sm86, unbound, 1, true, None, None),
            Err(ReplayError::L1Aggregation {
                sm_id: 0,
                source: L1CounterError::Architecture(
                    sim_arch::ArchitectureProvenanceError::Unbound {
                        component: "L1 counters",
                    }
                ),
            })
        ));

        let wrong_arch = BTreeMap::from([(0, sm(1, Some(L1Counters::for_arch(ArchId::Sm120))))]);
        assert!(matches!(
            fold_kernel_result(ArchId::Sm86, wrong_arch, 1, true, None, None),
            Err(ReplayError::L1Aggregation {
                sm_id: 0,
                source: L1CounterError::Architecture(
                    sim_arch::ArchitectureProvenanceError::Mismatch {
                        component: "L1 counters",
                        expected: ArchId::Sm86,
                        got: ArchId::Sm120,
                    }
                ),
            })
        ));
    }

    #[test]
    fn l1_presence_must_match_replay_mode_for_every_sm() {
        let all_none = BTreeMap::from([(0, sm(1, None))]);
        assert!(matches!(
            fold_kernel_result(ArchId::Sm86, all_none, 1, true, None, None),
            Err(ReplayError::L1PresenceMismatch {
                sm_id: 0,
                expected: true,
                found: false,
            })
        ));

        let all_some = BTreeMap::from([(0, sm(1, Some(L1Counters::for_arch(ArchId::Sm86))))]);
        assert!(matches!(
            fold_kernel_result(ArchId::Sm86, all_some, 1, false, None, None),
            Err(ReplayError::L1PresenceMismatch {
                sm_id: 0,
                expected: false,
                found: true,
            })
        ));

        let missing_second = BTreeMap::from([
            (0, sm(1, Some(L1Counters::for_arch(ArchId::Sm86)))),
            (1, sm(1, None)),
        ]);
        assert!(matches!(
            fold_kernel_result(ArchId::Sm86, missing_second, 2, true, None, None),
            Err(ReplayError::L1PresenceMismatch {
                sm_id: 1,
                expected: true,
                found: false,
            })
        ));

        let unexpected_second = BTreeMap::from([
            (0, sm(1, None)),
            (1, sm(1, Some(L1Counters::for_arch(ArchId::Sm86)))),
        ]);
        assert!(matches!(
            fold_kernel_result(ArchId::Sm86, unexpected_second, 2, false, None, None),
            Err(ReplayError::L1PresenceMismatch {
                sm_id: 1,
                expected: false,
                found: true,
            })
        ));
    }

    #[test]
    fn l1_overflow_keeps_responsible_sm_context() {
        let mut first = L1Counters::for_arch(ArchId::Sm86);
        first.hits = u64::MAX;
        first = first.with_unified_origin(sim_mem::MemoryOrigin::Global);
        let mut second = L1Counters::for_arch(ArchId::Sm86);
        second.hits = 1;
        second = second.with_unified_origin(sim_mem::MemoryOrigin::Global);
        let per_sm = BTreeMap::from([(0, sm(1, Some(first))), (1, sm(1, Some(second)))]);
        assert!(matches!(
            fold_kernel_result(ArchId::Sm86, per_sm, 2, true, None, None),
            Err(ReplayError::L1Aggregation {
                sm_id: 1,
                source: L1CounterError::Overflow {
                    field: sim_mem::L1CounterField::Hits,
                },
            })
        ));
    }
}

#[cfg(test)]
mod space_routing_tests {
    use super::*;
    use crate::control_bits::ControlBits;
    use crate::program::{FunctionSchedule, IssueSlot, OpClass};
    use std::collections::BTreeMap;
    use std::path::Path;

    /// An ATOMIC-class instruction carrying a single address record whose
    /// per-address space is `space` (None = legacy, no address_space).
    fn atomic_inst(space: Option<pb::MemorySpace>) -> pb::Instruction {
        let mut addrs = vec![LANE_INACTIVE_SENTINEL; 32];
        addrs[0] = 0x1000;
        pb::Instruction {
            active_mask: Some(1),
            predicate_mask: Some(1),
            data_width: Some(4),
            memory_op_class: Some(pb::MemoryOpClass::Atomic as i32),
            memory_op_access_kind: Some(pb::MemoryOpAccessKind::Atomic as i32),
            addresses: vec![pb::Address {
                data_width: 4,
                compression_format: sim_trace::lanes::FORMAT_LIST_ALL,
                addrs,
                address_space: space.map(|s| s as i32),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn global_inst() -> pb::Instruction {
        let mut addrs = vec![LANE_INACTIVE_SENTINEL; 32];
        addrs[0] = 0x1000;
        pb::Instruction {
            pc: 0,
            active_mask: Some(1),
            predicate_mask: Some(1),
            function_unique_id: 1,
            addresses: vec![pb::Address {
                data_width: 4,
                compression_format: sim_trace::lanes::FORMAT_LIST_ALL,
                addrs,
                address_space: Some(pb::MemorySpace::Global as i32),
                ..Default::default()
            }],
            memory_op_class: Some(pb::MemoryOpClass::Global as i32),
            data_width: Some(4),
            memory_op_access_kind: Some(pb::MemoryOpAccessKind::Load as i32),
            ..Default::default()
        }
    }

    fn spaced_address(space: pb::MemorySpace, lane_zero: i64) -> pb::Address {
        let mut addrs = vec![LANE_INACTIVE_SENTINEL; 32];
        addrs[0] = lane_zero;
        pb::Address {
            data_width: 4,
            compression_format: sim_trace::lanes::FORMAT_LIST_ALL,
            addrs,
            address_space: Some(space as i32),
            ..Default::default()
        }
    }

    fn async_copy_inst(addresses: Vec<pb::Address>) -> pb::Instruction {
        pb::Instruction {
            pc: 0,
            active_mask: Some(1),
            predicate_mask: Some(1),
            function_unique_id: 1,
            addresses,
            memory_op_class: Some(pb::MemoryOpClass::AsyncCopy as i32),
            data_width: Some(4),
            memory_op_access_kind: Some(pb::MemoryOpAccessKind::AsyncCopy as i32),
            ..Default::default()
        }
    }

    fn schedule(arch: ArchId, role: MemRole, opcode_text: &str) -> FunctionSchedule {
        FunctionSchedule {
            arch,
            function_unique_id: 1,
            kernel_name: "k".into(),
            function_name: "f".into(),
            by_pc: BTreeMap::from([(
                0,
                IssueSlot {
                    control: ControlBits::new(0, false, 7, 7, 0, 0).expect("valid test controls"),
                    op_class: OpClass::Other,
                    mem_role: role,
                    opcode_text: opcode_text.into(),
                },
            )]),
        }
    }

    fn assert_access_kind_rejected_in_both_modes(
        inst: &pb::Instruction,
        sched: &FunctionSchedule,
        expected: &'static str,
    ) {
        let class = pb::MemoryOpClass::try_from(inst.memory_op_class.unwrap()).unwrap();
        let got = pb::MemoryOpAccessKind::try_from(inst.memory_op_access_kind.unwrap()).unwrap();
        assert!(matches!(
            classify_flat(inst, sched, sched.arch, Path::new("kind.pb"), 5, 7),
            Err(ReplayError::MemoryRoute {
                warp_id: 5,
                record_ix: 7,
                class: got_class,
                source: MemoryRouteError::AccessKindInvalid {
                    got: got_kind,
                    expected: got_expected,
                },
                ..
            }) if got_class == class && got_kind == got && got_expected == expected
        ));

        let mut sectors = vec![9];
        let mut sector_scratch = vec![99];
        assert!(matches!(
            classify_with_l1(
                inst,
                sched,
                sched.arch,
                &mut sectors,
                &mut sector_scratch,
                Path::new("kind.pb"),
                5,
                7,
            ),
            Err(ReplayError::MemoryRoute {
                warp_id: 5,
                record_ix: 7,
                class: got_class,
                source: MemoryRouteError::AccessKindInvalid {
                    got: got_kind,
                    expected: got_expected,
                },
                ..
            }) if got_class == class && got_kind == got && got_expected == expected
        ));
        assert_eq!(sectors, vec![9]);
        assert!(sector_scratch.is_empty());
    }

    fn assert_static_access_kind_rejected_in_both_modes(
        inst: &pb::Instruction,
        sched: &FunctionSchedule,
        expected: pb::MemoryOpAccessKind,
    ) {
        let class = pb::MemoryOpClass::try_from(inst.memory_op_class.unwrap()).unwrap();
        let got = pb::MemoryOpAccessKind::try_from(inst.memory_op_access_kind.unwrap()).unwrap();
        assert!(matches!(
            classify_flat(inst, sched, sched.arch, Path::new("direction.pb"), 6, 8),
            Err(ReplayError::MemoryRoute {
                warp_id: 6,
                record_ix: 8,
                class: got_class,
                source: MemoryRouteError::StaticAccessKindMismatch {
                    got: got_kind,
                    expected: got_expected,
                },
                ..
            }) if got_class == class && got_kind == got && got_expected == expected
        ));

        let mut sectors = vec![10];
        let mut sector_scratch = vec![100];
        assert!(matches!(
            classify_with_l1(
                inst,
                sched,
                sched.arch,
                &mut sectors,
                &mut sector_scratch,
                Path::new("direction.pb"),
                6,
                8,
            ),
            Err(ReplayError::MemoryRoute {
                warp_id: 6,
                record_ix: 8,
                class: got_class,
                source: MemoryRouteError::StaticAccessKindMismatch {
                    got: got_kind,
                    expected: got_expected,
                },
                ..
            }) if got_class == class && got_kind == got && got_expected == expected
        ));
        assert_eq!(sectors, vec![10]);
        assert!(sector_scratch.is_empty());
    }

    fn assert_static_route_rejected_in_both_modes(
        inst: &pb::Instruction,
        sched: &FunctionSchedule,
        expected: pb::MemorySpace,
        got: pb::MemorySpace,
    ) {
        let class = pb::MemoryOpClass::try_from(inst.memory_op_class.unwrap()).unwrap();
        assert!(matches!(
            classify_flat(inst, sched, sched.arch, Path::new("route.pb"), 7, 9),
            Err(ReplayError::MemoryRoute {
                warp_id: 7,
                record_ix: 9,
                class: got_class,
                source: MemoryRouteError::StaticMemoryRouteMismatch {
                    expected: got_expected,
                    got: got_route,
                },
                ..
            }) if got_class == class && got_expected == expected && got_route == got
        ));

        let mut sectors = vec![12];
        let mut sector_scratch = vec![101];
        assert!(matches!(
            classify_with_l1(
                inst,
                sched,
                sched.arch,
                &mut sectors,
                &mut sector_scratch,
                Path::new("route.pb"),
                7,
                9,
            ),
            Err(ReplayError::MemoryRoute {
                warp_id: 7,
                record_ix: 9,
                class: got_class,
                source: MemoryRouteError::StaticMemoryRouteMismatch {
                    expected: got_expected,
                    got: got_route,
                },
                ..
            }) if got_class == class && got_expected == expected && got_route == got
        ));
        assert_eq!(sectors, vec![12]);
        assert!(sector_scratch.is_empty());
    }

    fn assert_static_class_rejected_in_both_modes(
        inst: &pb::Instruction,
        sched: &FunctionSchedule,
        expected: pb::MemoryOpClass,
    ) {
        let got = pb::MemoryOpClass::try_from(inst.memory_op_class.unwrap()).unwrap();
        assert!(matches!(
            classify_flat(inst, sched, sched.arch, Path::new("class.pb"), 8, 10),
            Err(ReplayError::MemoryRoute {
                warp_id: 8,
                record_ix: 10,
                class,
                source: MemoryRouteError::StaticMemoryClassMismatch {
                    expected: got_expected,
                    got: got_class,
                },
                ..
            }) if class == got && got_expected == expected && got_class == got
        ));

        let mut sectors = vec![13];
        let mut sector_scratch = vec![102];
        assert!(matches!(
            classify_with_l1(
                inst,
                sched,
                sched.arch,
                &mut sectors,
                &mut sector_scratch,
                Path::new("class.pb"),
                8,
                10,
            ),
            Err(ReplayError::MemoryRoute {
                warp_id: 8,
                record_ix: 10,
                class,
                source: MemoryRouteError::StaticMemoryClassMismatch {
                    expected: got_expected,
                    got: got_class,
                },
                ..
            }) if class == got && got_expected == expected && got_class == got
        ));
        assert_eq!(sectors, vec![13]);
        assert!(sector_scratch.is_empty());
    }

    fn assert_static_family_unsupported_in_both_modes(
        inst: &pb::Instruction,
        sched: &FunctionSchedule,
        family: &'static str,
    ) {
        let class = pb::MemoryOpClass::try_from(inst.memory_op_class.unwrap()).unwrap();
        assert!(matches!(
            classify_flat(inst, sched, sched.arch, Path::new("family.pb"), 9, 11),
            Err(ReplayError::MemoryRoute {
                warp_id: 9,
                record_ix: 11,
                class: got_class,
                source: MemoryRouteError::StaticMemoryFamilyUnsupported {
                    family: got_family,
                    arch,
                },
                ..
            }) if got_class == class && got_family == family && arch == sched.arch
        ));

        let mut sectors = vec![14];
        let mut sector_scratch = vec![103];
        assert!(matches!(
            classify_with_l1(
                inst,
                sched,
                sched.arch,
                &mut sectors,
                &mut sector_scratch,
                Path::new("family.pb"),
                9,
                11,
            ),
            Err(ReplayError::MemoryRoute {
                warp_id: 9,
                record_ix: 11,
                class: got_class,
                source: MemoryRouteError::StaticMemoryFamilyUnsupported {
                    family: got_family,
                    arch,
                },
                ..
            }) if got_class == class && got_family == family && arch == sched.arch
        ));
        assert_eq!(sectors, vec![14]);
        assert!(sector_scratch.is_empty());
    }

    fn assert_unsupported_memory_path_in_both_modes(
        inst: &pb::Instruction,
        sched: &FunctionSchedule,
        expected_class: pb::MemoryOpClass,
    ) {
        assert!(matches!(
            classify_flat(inst, sched, sched.arch, Path::new("unsupported.pb"), 10, 12),
            Err(ReplayError::UnsupportedMemoryPath { arch, class })
                if arch == sched.arch && class == expected_class
        ));

        let mut sectors = vec![15];
        let mut sector_scratch = vec![104];
        assert!(matches!(
            classify_with_l1(
                inst,
                sched,
                sched.arch,
                &mut sectors,
                &mut sector_scratch,
                Path::new("unsupported.pb"),
                10,
                12,
            ),
            Err(ReplayError::UnsupportedMemoryPath { arch, class })
                if arch == sched.arch && class == expected_class
        ));
        assert_eq!(sectors, vec![15]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn sector_expansion_overflow_carries_replay_context() {
        let mut addrs = vec![LANE_INACTIVE_SENTINEL; 32];
        addrs[0] = i64::MAX;
        let mut later_addrs = vec![LANE_INACTIVE_SENTINEL; 32];
        later_addrs[0] = 0;
        let inst = pb::Instruction {
            pc: 0,
            active_mask: Some(1),
            predicate_mask: Some(1),
            function_unique_id: 1,
            addresses: vec![
                pb::Address {
                    data_width: 2,
                    compression_format: sim_trace::lanes::FORMAT_LIST_ALL,
                    addrs,
                    address_space: Some(pb::MemorySpace::Global as i32),
                    ..Default::default()
                },
                pb::Address {
                    data_width: 2,
                    compression_format: sim_trace::lanes::FORMAT_LIST_ALL,
                    addrs: later_addrs,
                    address_space: Some(pb::MemorySpace::Global as i32),
                    ..Default::default()
                },
            ],
            memory_op_class: Some(pb::MemoryOpClass::Global as i32),
            data_width: Some(2),
            memory_op_access_kind: Some(pb::MemoryOpAccessKind::Load as i32),
            ..Default::default()
        };
        let mut sectors = vec![11];
        let mut sector_scratch = vec![99];
        let err = classify_with_l1(
            &inst,
            &schedule(ArchId::Sm86, MemRole::None, "LDG.E"),
            ArchId::Sm86,
            &mut sectors,
            &mut sector_scratch,
            Path::new("block.pb"),
            7,
            9,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ReplayError::SectorExpansion {
                ref path,
                warp_id: 7,
                record_ix: 9,
                source: SectorExpansionError::AddressOverflow {
                    lane: 0,
                    base: i64::MAX,
                    data_width_bytes: 2,
                },
            } if path == Path::new("block.pb")
        ));
        assert_eq!(sectors, vec![11]);
    }

    #[test]
    fn oversized_data_width_is_rejected_before_sector_expansion() {
        let mut inst = global_inst();
        inst.data_width = Some(i32::MAX);
        inst.addresses[0].data_width = i32::MAX;
        inst.addresses[0].addrs[0] = 0;
        let mut sectors = vec![13];
        let mut sector_scratch = vec![99];

        let err = classify_with_l1(
            &inst,
            &schedule(
                ArchId::Sm86,
                MemRole::EligibleLoad(CacheOpToken::LdgECa),
                "LDG.E.CA",
            ),
            ArchId::Sm86,
            &mut sectors,
            &mut sector_scratch,
            Path::new("wide.pb"),
            8,
            10,
        )
        .unwrap_err();

        assert!(matches!(
            err,
            ReplayError::SectorCountOverflow {
                ref path,
                warp_id: 8,
                record_ix: 10,
                sectors: 67_108_864,
            } if path == Path::new("wide.pb")
        ));
        assert_eq!(sectors, vec![13]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn multi_address_overflow_reports_the_exact_instruction_union() {
        const WIDTH: i32 = 96_000;
        let first = vec![0; 32];
        let second = (0_i64..32)
            .map(|lane| (3_000 + lane * 3_000) * 32)
            .collect();
        let inst = pb::Instruction {
            pc: 0,
            active_mask: Some(u32::MAX),
            predicate_mask: Some(u32::MAX),
            function_unique_id: 1,
            addresses: vec![
                pb::Address {
                    data_width: WIDTH,
                    compression_format: sim_trace::lanes::FORMAT_LIST_ALL,
                    addrs: first,
                    address_space: Some(pb::MemorySpace::Global as i32),
                    ..Default::default()
                },
                pb::Address {
                    data_width: WIDTH,
                    compression_format: sim_trace::lanes::FORMAT_LIST_ALL,
                    addrs: second,
                    address_space: Some(pb::MemorySpace::Global as i32),
                    ..Default::default()
                },
            ],
            memory_op_class: Some(pb::MemoryOpClass::Global as i32),
            data_width: Some(WIDTH),
            memory_op_access_kind: Some(pb::MemoryOpAccessKind::Load as i32),
            ..Default::default()
        };
        let mut sectors = vec![15];
        let mut sector_scratch = vec![99];

        let err = classify_with_l1(
            &inst,
            &schedule(
                ArchId::Sm86,
                MemRole::EligibleLoad(CacheOpToken::LdgECa),
                "LDG.E.CA",
            ),
            ArchId::Sm86,
            &mut sectors,
            &mut sector_scratch,
            Path::new("multi-wide.pb"),
            9,
            11,
        )
        .unwrap_err();

        assert!(matches!(
            err,
            ReplayError::SectorCountOverflow {
                ref path,
                warp_id: 9,
                record_ix: 11,
                sectors: 99_000,
            } if path == Path::new("multi-wide.pb")
        ));
        assert_eq!(sectors, vec![15]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn oversized_count_does_not_mask_unknown_static_opcode() {
        let mut inst = global_inst();
        inst.data_width = Some(i32::MAX);
        inst.addresses[0].data_width = i32::MAX;
        inst.addresses[0].addrs[0] = 0;
        let mut sectors = vec![17];
        let mut sector_scratch = vec![99];

        assert!(matches!(
            classify_with_l1(
                &inst,
                &schedule(ArchId::Sm86, MemRole::None, "IADD3 R0, R1, R2, RZ"),
                ArchId::Sm86,
                &mut sectors,
                &mut sector_scratch,
                Path::new("wide-opcode.pb"),
                10,
                12,
            ),
            Err(ReplayError::UnknownGlobalOpcode {
                arch: ArchId::Sm86,
                pc: 0,
                ref opcode,
            }) if opcode == "IADD3"
        ));
        assert_eq!(sectors, vec![17]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn oversized_count_does_not_mask_unsupported_special_route() {
        let mut inst = atomic_inst(Some(pb::MemorySpace::Global));
        inst.function_unique_id = 1;
        inst.data_width = Some(i32::MAX);
        inst.addresses[0].data_width = i32::MAX;
        inst.addresses[0].addrs[0] = 0;
        let mut sectors = vec![19];
        let mut sector_scratch = vec![99];

        assert!(matches!(
            classify_with_l1(
                &inst,
                &schedule(ArchId::Sm120, MemRole::None, "ATOMG.E.ADD.STRONG.GPU"),
                ArchId::Sm120,
                &mut sectors,
                &mut sector_scratch,
                Path::new("wide-atomic.pb"),
                11,
                13,
            ),
            Err(ReplayError::UnsupportedMemoryPath {
                arch: ArchId::Sm120,
                class: pb::MemoryOpClass::Atomic,
            })
        ));
        assert_eq!(sectors, vec![19]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn later_trace_error_precedes_sector_callbacks() {
        let mut overflow = vec![LANE_INACTIVE_SENTINEL; 32];
        overflow[0] = i64::MAX;
        let inst = pb::Instruction {
            pc: 0,
            active_mask: Some(1),
            predicate_mask: Some(1),
            function_unique_id: 1,
            addresses: vec![
                pb::Address {
                    data_width: 2,
                    compression_format: sim_trace::lanes::FORMAT_LIST_ALL,
                    addrs: overflow,
                    address_space: Some(pb::MemorySpace::Global as i32),
                    ..Default::default()
                },
                pb::Address {
                    data_width: 2,
                    compression_format: sim_trace::lanes::FORMAT_LIST_ALL,
                    addrs: Vec::new(),
                    address_space: Some(pb::MemorySpace::Shared as i32),
                    ..Default::default()
                },
            ],
            memory_op_class: Some(pb::MemoryOpClass::Global as i32),
            data_width: Some(2),
            ..Default::default()
        };
        assert!(matches!(
            classify_flat(
                &inst,
                &schedule(ArchId::Sm86, MemRole::None, "LDG.E"),
                ArchId::Sm86,
                Path::new("precedence.pb"),
                2,
                4,
            ),
            Err(ReplayError::Trace(TraceError::Lane {
                source: sim_trace::lanes::LaneError::ListAllLengthInvalid { got: 0 },
                ..
            }))
        ));
        let mut sectors = vec![13];
        let mut sector_scratch = vec![99];
        assert!(matches!(
            classify_with_l1(
                &inst,
                &schedule(ArchId::Sm86, MemRole::None, "LDG.E"),
                ArchId::Sm86,
                &mut sectors,
                &mut sector_scratch,
                Path::new("precedence.pb"),
                2,
                4,
            ),
            Err(ReplayError::Trace(TraceError::Lane {
                source: sim_trace::lanes::LaneError::ListAllLengthInvalid { got: 0 },
                ..
            }))
        ));
        assert_eq!(sectors, vec![13]);
        assert!(sector_scratch.is_empty());
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn staged_sector_start_uses_full_usize_domain() {
        let start = usize::try_from(1_u64 << 32).unwrap();
        assert_eq!(
            checked_staged_global_kind(MemRole::None, start, 1),
            Some(AccessKind::GlobalUntagged { off: start, len: 1 })
        );
        assert_eq!(
            checked_staged_global_kind(MemRole::None, usize::MAX, 1),
            None
        );
    }

    #[test]
    fn staged_sector_count_overflow_precedes_arena_append() {
        let count = usize::from(u16::MAX)
            .checked_add(1)
            .expect("u16 maximum plus one fits usize");
        let scratch = vec![0; count];
        let mut sectors = vec![17];
        let err = append_staged_global_sectors(
            MemRole::None,
            &mut sectors,
            &scratch,
            Path::new("count.pb"),
            3,
            5,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ReplayError::SectorCountOverflow {
                ref path,
                warp_id: 3,
                record_ix: 5,
                sectors,
            } if path == Path::new("count.pb") && sectors == count
        ));
        assert_eq!(sectors, vec![17]);
    }

    #[test]
    fn unknown_static_opcode_leaves_sector_arena_unchanged() {
        let inst = global_inst();
        let mut sectors = vec![19];
        let mut sector_scratch = vec![99];
        let err = classify_with_l1(
            &inst,
            &schedule(ArchId::Sm86, MemRole::None, "IADD3 R0, R1, R2, RZ"),
            ArchId::Sm86,
            &mut sectors,
            &mut sector_scratch,
            Path::new("opcode.pb"),
            4,
            6,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ReplayError::UnknownGlobalOpcode {
                arch: ArchId::Sm86,
                pc: 0,
                ref opcode,
            } if opcode == "IADD3"
        ));
        assert_eq!(sectors, vec![19]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn active_no_payload_global_is_typed_rejected_in_both_modes() {
        let sched = schedule(ArchId::Sm86, MemRole::None, "LDG.E");
        let mut inst = global_inst();
        inst.addresses.clear();

        assert!(matches!(
            classify_flat(
                &inst,
                &sched,
                sched.arch,
                Path::new("empty_global.pb"),
                2,
                4,
            ),
            Err(ReplayError::MemoryRoute {
                warp_id: 2,
                record_ix: 4,
                class: pb::MemoryOpClass::Global,
                source: MemoryRouteError::GlobalOperandCount { got: 0 },
                ..
            })
        ));

        let mut sectors = vec![21];
        let mut sector_scratch = vec![99];
        assert!(matches!(
            classify_with_l1(
                &inst,
                &sched,
                sched.arch,
                &mut sectors,
                &mut sector_scratch,
                Path::new("empty_global.pb"),
                2,
                4,
            ),
            Err(ReplayError::MemoryRoute {
                warp_id: 2,
                record_ix: 4,
                class: pb::MemoryOpClass::Global,
                source: MemoryRouteError::GlobalOperandCount { got: 0 },
                ..
            })
        ));
        assert_eq!(sectors, vec![21]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn zero_active_no_payload_global_retains_legacy_route() {
        let sched = schedule(ArchId::Sm86, MemRole::None, "LD.E R0, [R2.64]");
        let mut inst = global_inst();
        inst.predicate_mask = Some(0);
        inst.addresses.clear();

        let flat = classify_flat(&inst, &sched, sched.arch, Path::new("zero_global.pb"), 0, 0)
            .expect("zero-active class-only GLOBAL retains its legacy route");
        assert!(matches!(flat, AccessKind::GlobalUntagged { .. }));

        let mut sectors = vec![23];
        let mut sector_scratch = vec![99];
        let l1 = classify_with_l1(
            &inst,
            &sched,
            sched.arch,
            &mut sectors,
            &mut sector_scratch,
            Path::new("zero_global.pb"),
            0,
            0,
        )
        .expect("zero-active class-only GLOBAL retains its L1 route");
        assert!(matches!(l1, AccessKind::GlobalUntagged { off: 1, len: 0 }));
        assert_eq!(sectors, vec![23]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn active_no_payload_shared_or_local_is_typed_rejected_in_both_modes() {
        for class in [pb::MemoryOpClass::Shared, pb::MemoryOpClass::Local] {
            let sched = schedule(ArchId::Sm86, MemRole::None, "LDS");
            let mut inst = global_inst();
            inst.memory_op_class = Some(class as i32);
            inst.addresses.clear();

            assert!(matches!(
                classify_flat(
                    &inst,
                    &sched,
                    sched.arch,
                    Path::new("empty_shared.pb"),
                    2,
                    5,
                ),
                Err(ReplayError::MemoryRoute {
                    warp_id: 2,
                    record_ix: 5,
                    class: got_class,
                    source: MemoryRouteError::SharedLocalOperandCount { got: 0 },
                    ..
                }) if got_class == class
            ));

            let mut sectors = vec![25];
            let mut sector_scratch = vec![99];
            assert!(matches!(
                classify_with_l1(
                    &inst,
                    &sched,
                    sched.arch,
                    &mut sectors,
                    &mut sector_scratch,
                    Path::new("empty_shared.pb"),
                    2,
                    5,
                ),
                Err(ReplayError::MemoryRoute {
                    warp_id: 2,
                    record_ix: 5,
                    class: got_class,
                    source: MemoryRouteError::SharedLocalOperandCount { got: 0 },
                    ..
                }) if got_class == class
            ));
            assert_eq!(sectors, vec![25]);
            assert!(sector_scratch.is_empty());
        }
    }

    #[test]
    fn valid_ldl_uses_local_sector_route() {
        let sched = schedule(
            ArchId::Sm86,
            MemRole::EligibleLoad(CacheOpToken::Ldl),
            "LDL R0, [R1]",
        );
        let mut inst = global_inst();
        inst.memory_op_class = Some(pb::MemoryOpClass::Local as i32);
        inst.addresses[0].address_space = Some(pb::MemorySpace::Local as i32);

        let flat = classify_flat(&inst, &sched, sched.arch, Path::new("local.pb"), 0, 0)
            .expect("joined LDL has a reviewed flat Local route");
        assert!(matches!(flat, AccessKind::LocalLoad { off: 0, len: 0 }));

        let mut sectors = vec![31];
        let mut sector_scratch = vec![99];
        let l1 = classify_with_l1(
            &inst,
            &sched,
            sched.arch,
            &mut sectors,
            &mut sector_scratch,
            Path::new("local.pb"),
            0,
            0,
        )
        .expect("joined LDL has a reviewed hierarchy Local route");
        assert!(matches!(l1, AccessKind::LocalLoad { off: 1, len: 1 }));
        assert_eq!(sectors, vec![31, 0x1000 >> 5]);
        assert_eq!(sector_scratch, vec![0x1000 >> 5]);
    }

    #[test]
    fn valid_stl_preserves_store_direction_on_local_sector_route() {
        let sched = schedule(
            ArchId::Sm86,
            MemRole::EligibleLoad(CacheOpToken::Stl),
            "STL [R1], R0",
        );
        let mut inst = global_inst();
        inst.memory_op_class = Some(pb::MemoryOpClass::Local as i32);
        inst.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Store as i32);
        inst.addresses[0].address_space = Some(pb::MemorySpace::Local as i32);

        let flat = classify_flat(&inst, &sched, sched.arch, Path::new("stl.pb"), 0, 0)
            .expect("joined STL has a reviewed flat Local route");
        assert!(matches!(flat, AccessKind::LocalStore { off: 0, len: 0 }));

        let mut sectors = Vec::new();
        let mut sector_scratch = Vec::new();
        let l1 = classify_with_l1(
            &inst,
            &sched,
            sched.arch,
            &mut sectors,
            &mut sector_scratch,
            Path::new("stl.pb"),
            0,
            0,
        )
        .expect("joined STL has a reviewed hierarchy Local route");
        assert!(matches!(l1, AccessKind::LocalStore { off: 0, len: 1 }));
        assert_eq!(sectors, vec![0x1000 >> 5]);
        assert_eq!(sector_scratch, sectors);
    }

    #[test]
    fn local_route_rejects_non_ldl_stl_static_roles_without_append() {
        let sched = schedule(ArchId::Sm86, MemRole::None, "IADD3 R0, R1, R2, RZ");
        let mut inst = global_inst();
        inst.memory_op_class = Some(pb::MemoryOpClass::Local as i32);
        inst.addresses[0].address_space = Some(pb::MemorySpace::Local as i32);

        assert!(matches!(
            classify_flat(&inst, &sched, sched.arch, Path::new("local_role.pb"), 2, 3),
            Err(ReplayError::UnknownLocalOpcode {
                arch: ArchId::Sm86,
                pc: 0,
                ref opcode,
            }) if opcode == "IADD3"
        ));
        let mut sectors = vec![17];
        let mut sector_scratch = vec![99];
        assert!(matches!(
            classify_with_l1(
                &inst,
                &sched,
                sched.arch,
                &mut sectors,
                &mut sector_scratch,
                Path::new("local_role.pb"),
                2,
                3,
            ),
            Err(ReplayError::UnknownLocalOpcode {
                arch: ArchId::Sm86,
                pc: 0,
                ref opcode,
            }) if opcode == "IADD3"
        ));
        assert_eq!(sectors, vec![17]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn static_ldl_stl_evidence_cannot_escape_local_route() {
        let assert_rejected =
            |inst: &pb::Instruction, sched: &FunctionSchedule, expected_class, got_space| {
                assert!(matches!(
                    classify_flat(
                        inst,
                        sched,
                        sched.arch,
                        Path::new("escaped_local.pb"),
                        3,
                        4,
                    ),
                    Err(ReplayError::MemoryRoute {
                        class,
                        source: MemoryRouteError::StaticMemoryRouteMismatch {
                            expected: pb::MemorySpace::Local,
                            got,
                        },
                        ..
                    }) if class == expected_class && got == got_space
                ));

                let mut sectors = vec![23];
                let mut sector_scratch = vec![99];
                assert!(matches!(
                    classify_with_l1(
                        inst,
                        sched,
                        sched.arch,
                        &mut sectors,
                        &mut sector_scratch,
                        Path::new("escaped_local.pb"),
                        3,
                        4,
                    ),
                    Err(ReplayError::MemoryRoute {
                        class,
                        source: MemoryRouteError::StaticMemoryRouteMismatch {
                            expected: pb::MemorySpace::Local,
                            got,
                        },
                        ..
                    }) if class == expected_class && got == got_space
                ));
                assert_eq!(sectors, vec![23]);
                assert!(sector_scratch.is_empty());
            };

        // The concrete contamination case: STL's EligibleLoad role must never
        // become a GlobalEligible load merely because the dynamic record says
        // GLOBAL/STORE.
        let mut global_stl = global_inst();
        global_stl.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Store as i32);
        assert_rejected(
            &global_stl,
            &schedule(
                ArchId::Sm86,
                MemRole::EligibleLoad(CacheOpToken::Stl),
                "STL [R1], R0",
            ),
            pb::MemoryOpClass::Global,
            pb::MemorySpace::Global,
        );

        // The reciprocal guard is independent of the additive direction field:
        // legacy-absent access_kind cannot let LDL escape through Shared timing.
        let mut shared_ldl = global_inst();
        shared_ldl.memory_op_class = Some(pb::MemoryOpClass::Shared as i32);
        shared_ldl.memory_op_access_kind = None;
        shared_ldl.addresses[0].address_space = Some(pb::MemorySpace::Shared as i32);
        assert_rejected(
            &shared_ldl,
            &schedule(
                ArchId::Sm86,
                MemRole::EligibleLoad(CacheOpToken::Ldl),
                "LDL R0, [R1]",
            ),
            pb::MemoryOpClass::Shared,
            pb::MemorySpace::Shared,
        );
    }

    #[test]
    fn local_address_overflow_precedes_unknown_static_opcode() {
        let mut inst = global_inst();
        inst.memory_op_class = Some(pb::MemoryOpClass::Local as i32);
        inst.data_width = Some(2);
        inst.addresses[0].address_space = Some(pb::MemorySpace::Local as i32);
        inst.addresses[0].data_width = 2;
        inst.addresses[0].addrs[0] = i64::MAX;
        let mut sectors = vec![29];
        let mut sector_scratch = vec![99];

        assert!(matches!(
            classify_with_l1(
                &inst,
                &schedule(ArchId::Sm86, MemRole::None, "IADD3 R0, R1, R2, RZ"),
                ArchId::Sm86,
                &mut sectors,
                &mut sector_scratch,
                Path::new("local_overflow.pb"),
                5,
                6,
            ),
            Err(ReplayError::SectorExpansion {
                warp_id: 5,
                record_ix: 6,
                source: SectorExpansionError::AddressOverflow {
                    lane: 0,
                    base: i64::MAX,
                    data_width_bytes: 2,
                },
                ..
            })
        ));
        assert_eq!(sectors, vec![29]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn unknown_local_opcode_precedes_sector_count_narrowing() {
        let mut inst = global_inst();
        inst.memory_op_class = Some(pb::MemoryOpClass::Local as i32);
        inst.data_width = Some(i32::MAX);
        inst.addresses[0].address_space = Some(pb::MemorySpace::Local as i32);
        inst.addresses[0].data_width = i32::MAX;
        inst.addresses[0].addrs[0] = 0;
        let mut sectors = vec![31];
        let mut sector_scratch = vec![99];

        assert!(matches!(
            classify_with_l1(
                &inst,
                &schedule(ArchId::Sm86, MemRole::None, "IADD3 R0, R1, R2, RZ"),
                ArchId::Sm86,
                &mut sectors,
                &mut sector_scratch,
                Path::new("local_wide_opcode.pb"),
                7,
                8,
            ),
            Err(ReplayError::UnknownLocalOpcode {
                arch: ArchId::Sm86,
                pc: 0,
                ref opcode,
            }) if opcode == "IADD3"
        ));
        assert_eq!(sectors, vec![31]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn zero_active_no_payload_shared_retains_legacy_route() {
        let sched = schedule(ArchId::Sm86, MemRole::None, "LDS");
        let mut inst = global_inst();
        inst.memory_op_class = Some(pb::MemoryOpClass::Shared as i32);
        inst.predicate_mask = Some(0);
        inst.addresses.clear();

        let flat = classify_flat(&inst, &sched, sched.arch, Path::new("zero_shared.pb"), 0, 0)
            .expect("zero-active class-only SHARED retains its legacy route");
        assert!(matches!(flat, AccessKind::Shared));

        let mut sectors = vec![26];
        let mut sector_scratch = vec![99];
        let l1 = classify_with_l1(
            &inst,
            &sched,
            sched.arch,
            &mut sectors,
            &mut sector_scratch,
            Path::new("zero_shared.pb"),
            0,
            0,
        )
        .expect("zero-active class-only SHARED retains its L1 route");
        assert!(matches!(l1, AccessKind::Shared));
        assert_eq!(sectors, vec![26]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn shared_or_local_class_rejects_explicit_global_space_in_both_modes() {
        for class in [pb::MemoryOpClass::Shared, pb::MemoryOpClass::Local] {
            let sched = schedule(ArchId::Sm86, MemRole::None, "LDS");
            let mut inst = global_inst();
            inst.memory_op_class = Some(class as i32);

            assert!(matches!(
                classify_flat(&inst, &sched, sched.arch, Path::new("space.pb"), 3, 5),
                Err(ReplayError::MemoryRoute {
                    warp_id: 3,
                    record_ix: 5,
                    class: got_class,
                    source: MemoryRouteError::SharedLocalSpaceInvalid {
                        operand: 0,
                        space: pb::MemorySpace::Global,
                    },
                    ..
                }) if got_class == class
            ));

            let mut sectors = vec![27];
            let mut sector_scratch = vec![99];
            assert!(matches!(
                classify_with_l1(
                    &inst,
                    &sched,
                    sched.arch,
                    &mut sectors,
                    &mut sector_scratch,
                    Path::new("space.pb"),
                    3,
                    5,
                ),
                Err(ReplayError::MemoryRoute {
                    warp_id: 3,
                    record_ix: 5,
                    class: got_class,
                    source: MemoryRouteError::SharedLocalSpaceInvalid {
                        operand: 0,
                        space: pb::MemorySpace::Global,
                    },
                    ..
                }) if got_class == class
            ));
            assert_eq!(sectors, vec![27]);
            assert!(sector_scratch.is_empty());
        }
    }

    #[test]
    fn shared_and_local_classes_reject_crossed_spaces() {
        let cases = [
            (pb::MemoryOpClass::Shared, pb::MemorySpace::Local),
            (pb::MemoryOpClass::Local, pb::MemorySpace::Shared),
        ];
        for (case, (class, space)) in cases.into_iter().enumerate() {
            let sched = schedule(ArchId::Sm86, MemRole::None, "LDS");
            let mut inst = global_inst();
            inst.memory_op_class = Some(class as i32);
            inst.addresses[0].address_space = Some(space as i32);

            assert!(matches!(
                classify_flat(
                    &inst,
                    &sched,
                    sched.arch,
                    Path::new("crossed_space.pb"),
                    4,
                    case,
                ),
                Err(ReplayError::MemoryRoute {
                    warp_id: 4,
                    record_ix,
                    class: got_class,
                    source: MemoryRouteError::SharedLocalSpaceInvalid {
                        operand: 0,
                        space: got_space,
                    },
                    ..
                }) if record_ix == case && got_class == class && got_space == space
            ));

            // Exercise the replay boundary directly as a defense independent of
            // the reader's equivalent preflight rejection.
            assert!(matches!(
                replay_memory_route(
                    &inst,
                    class,
                    MemoryPayloadContract::AddressCarryingV1,
                    Path::new("crossed_space.pb"),
                    4,
                    case,
                ),
                Err(ReplayError::MemoryRoute {
                    class: got_class,
                    source: MemoryRouteError::SharedLocalSpaceInvalid {
                        operand: 0,
                        space: got_space,
                    },
                    ..
                }) if got_class == class && got_space == space
            ));
        }
    }

    #[test]
    fn local_atomic_is_never_collapsed_into_shared_timing() {
        let sched = schedule(ArchId::Sm86, MemRole::None, "IADD3 R0, R1, R2, RZ");
        let mut inst = atomic_inst(Some(pb::MemorySpace::Local));
        inst.function_unique_id = 1;

        assert_eq!(
            replay_memory_route(
                &inst,
                pb::MemoryOpClass::Atomic,
                MemoryPayloadContract::AddressCarryingV1,
                Path::new("local_atomic.pb"),
                8,
                10,
            )
            .expect("an explicit Local operand resolves to the Local route"),
            ReplayMemoryRoute::Local
        );
        assert!(matches!(
            classify_flat(
                &inst,
                &sched,
                sched.arch,
                Path::new("local_atomic.pb"),
                8,
                10,
            ),
            Err(ReplayError::UnsupportedMemoryPath {
                arch: ArchId::Sm86,
                class: pb::MemoryOpClass::Atomic,
            })
        ));

        let mut sectors = vec![39];
        let mut sector_scratch = vec![99];
        assert!(matches!(
            classify_with_l1(
                &inst,
                &sched,
                sched.arch,
                &mut sectors,
                &mut sector_scratch,
                Path::new("local_atomic.pb"),
                8,
                10,
            ),
            Err(ReplayError::UnsupportedMemoryPath {
                arch: ArchId::Sm86,
                class: pb::MemoryOpClass::Atomic,
            })
        ));
        assert_eq!(sectors, vec![39]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn legacy_atomic_shared_and_local_operands_conflict() {
        let sched = schedule(ArchId::Sm86, MemRole::None, "IADD3 R0, R1, R2, RZ");
        let mut inst = atomic_inst(Some(pb::MemorySpace::Shared));
        inst.function_unique_id = 1;
        inst.addresses
            .push(spaced_address(pb::MemorySpace::Local, 0x1040));

        assert!(matches!(
            classify_flat_with_contract(
                &inst,
                &sched,
                sched.arch,
                MemoryPayloadContract::LegacyExclusion,
                Path::new("legacy_local_atomic.pb"),
                9,
                11,
            ),
            Err(ReplayError::MemoryRoute {
                class: pb::MemoryOpClass::Atomic,
                source: MemoryRouteError::AtomicSpacesConflict,
                ..
            })
        ));

        let mut sectors = vec![41];
        let mut sector_scratch = vec![99];
        assert!(matches!(
            classify_with_l1_with_contract(
                &inst,
                &sched,
                sched.arch,
                MemoryPayloadContract::LegacyExclusion,
                &mut sectors,
                &mut sector_scratch,
                Path::new("legacy_local_atomic.pb"),
                9,
                11,
            ),
            Err(ReplayError::MemoryRoute {
                class: pb::MemoryOpClass::Atomic,
                source: MemoryRouteError::AtomicSpacesConflict,
                ..
            })
        ));
        assert_eq!(sectors, vec![41]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn zero_active_shared_atomic_routes_shared_in_flat_and_l1_modes() {
        let mut inst = atomic_inst(Some(pb::MemorySpace::Shared));
        inst.function_unique_id = 1;
        inst.active_mask = Some(1);
        inst.predicate_mask = Some(0);
        inst.data_width = Some(4);
        inst.addresses[0].data_width = 4;
        inst.addresses[0].compression_format = sim_trace::lanes::FORMAT_BASE_DELTA;
        inst.addresses[0].addrs.clear();
        let sched = schedule(
            ArchId::Sm120,
            MemRole::BypassLoad(CacheOpToken::Atom),
            "ATOMS.E.ADD",
        );

        let flat = classify_flat(&inst, &sched, sched.arch, Path::new("zero.pb"), 0, 0)
            .expect("zero-active shared atomic has a reviewed flat route");
        assert!(matches!(flat, AccessKind::Shared));

        let mut sectors = vec![23];
        let mut sector_scratch = vec![99];
        let l1 = classify_with_l1(
            &inst,
            &sched,
            sched.arch,
            &mut sectors,
            &mut sector_scratch,
            Path::new("zero.pb"),
            0,
            0,
        )
        .expect("zero-active shared atomic has the same L1 route");
        assert!(matches!(l1, AccessKind::Shared));
        assert_eq!(sectors, vec![23]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn zero_active_no_payload_atomic_retains_legacy_global_route() {
        let sched = schedule(
            ArchId::Sm86,
            MemRole::BypassLoad(CacheOpToken::Atom),
            "ATOMG.E.CAS.STRONG.GPU",
        );
        let mut inst = atomic_inst(Some(pb::MemorySpace::Global));
        inst.predicate_mask = Some(0);
        inst.addresses.clear();

        let flat = classify_flat(&inst, &sched, sched.arch, Path::new("zero_atomic.pb"), 0, 0)
            .expect("zero-active no-payload legacy atomic retains its global route");
        assert!(matches!(
            flat,
            AccessKind::GlobalBypass {
                off: 0,
                len: 0,
                token: CacheOpToken::Atom,
            }
        ));

        let mut sectors = vec![29];
        let mut sector_scratch = vec![99];
        let l1 = classify_with_l1(
            &inst,
            &sched,
            sched.arch,
            &mut sectors,
            &mut sector_scratch,
            Path::new("zero_atomic.pb"),
            0,
            0,
        )
        .expect("zero-active no-payload legacy atomic retains its L1 route");
        assert!(matches!(
            l1,
            AccessKind::GlobalBypass {
                off: 1,
                len: 0,
                token: CacheOpToken::Atom,
            }
        ));
        assert_eq!(sectors, vec![29]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn active_no_payload_atomic_is_typed_rejected_in_both_modes() {
        let sched = schedule(
            ArchId::Sm86,
            MemRole::BypassLoad(CacheOpToken::Atom),
            "ATOMG.E.CAS.STRONG.GPU",
        );
        let mut inst = atomic_inst(Some(pb::MemorySpace::Global));
        inst.addresses.clear();

        assert!(matches!(
            classify_flat(
                &inst,
                &sched,
                sched.arch,
                Path::new("empty_atomic.pb"),
                3,
                5
            ),
            Err(ReplayError::MemoryRoute {
                warp_id: 3,
                record_ix: 5,
                class: pb::MemoryOpClass::Atomic,
                source: MemoryRouteError::AtomicOperandCount { got: 0 },
                ..
            })
        ));

        let mut sectors = vec![31];
        let mut sector_scratch = vec![99];
        assert!(matches!(
            classify_with_l1(
                &inst,
                &sched,
                sched.arch,
                &mut sectors,
                &mut sector_scratch,
                Path::new("empty_atomic.pb"),
                3,
                5,
            ),
            Err(ReplayError::MemoryRoute {
                warp_id: 3,
                record_ix: 5,
                class: pb::MemoryOpClass::Atomic,
                source: MemoryRouteError::AtomicOperandCount { got: 0 },
                ..
            })
        ));
        assert_eq!(sectors, vec![31]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn multiple_atomic_operands_are_typed_rejected_in_both_modes() {
        let sched = schedule(
            ArchId::Sm86,
            MemRole::BypassLoad(CacheOpToken::Atom),
            "ATOMG.E.CAS.STRONG.GPU",
        );
        let mut inst = atomic_inst(Some(pb::MemorySpace::Global));
        inst.addresses
            .push(spaced_address(pb::MemorySpace::Global, 0x1040));

        assert!(matches!(
            classify_flat(
                &inst,
                &sched,
                sched.arch,
                Path::new("multi_atomic.pb"),
                4,
                6
            ),
            Err(ReplayError::MemoryRoute {
                warp_id: 4,
                record_ix: 6,
                class: pb::MemoryOpClass::Atomic,
                source: MemoryRouteError::AtomicOperandCount { got: 2 },
                ..
            })
        ));

        let mut sectors = vec![37];
        let mut sector_scratch = vec![99];
        assert!(matches!(
            classify_with_l1(
                &inst,
                &sched,
                sched.arch,
                &mut sectors,
                &mut sector_scratch,
                Path::new("multi_atomic.pb"),
                4,
                6,
            ),
            Err(ReplayError::MemoryRoute {
                warp_id: 4,
                record_ix: 6,
                class: pb::MemoryOpClass::Atomic,
                source: MemoryRouteError::AtomicOperandCount { got: 2 },
                ..
            })
        ));
        assert_eq!(sectors, vec![37]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn explicit_access_kind_must_match_replay_class_in_both_modes() {
        let mut global = global_inst();
        global.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Atomic as i32);
        assert_access_kind_rejected_in_both_modes(
            &global,
            &schedule(ArchId::Sm86, MemRole::None, "LD.E R0, [R2.64]"),
            "LOAD or STORE",
        );

        let mut atomic = atomic_inst(Some(pb::MemorySpace::Global));
        atomic.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Load as i32);
        assert_access_kind_rejected_in_both_modes(
            &atomic,
            &schedule(
                ArchId::Sm86,
                MemRole::BypassLoad(CacheOpToken::Atom),
                "ATOMG.E.CAS.STRONG.GPU",
            ),
            "ATOMIC",
        );

        let mut async_copy = async_copy_inst(vec![spaced_address(pb::MemorySpace::Global, 0x1000)]);
        async_copy.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Atomic as i32);
        assert_access_kind_rejected_in_both_modes(
            &async_copy,
            &schedule(
                ArchId::Sm86,
                MemRole::BypassLoad(CacheOpToken::Ldgsts),
                "LDGSTS.E.128",
            ),
            "ASYNC_COPY",
        );
    }

    #[test]
    fn static_families_reject_class_confusion_without_access_kind() {
        let mut atmg_as_global = global_inst();
        atmg_as_global.memory_op_access_kind = None;
        assert_static_class_rejected_in_both_modes(
            &atmg_as_global,
            &schedule(
                ArchId::Sm86,
                MemRole::BypassLoad(CacheOpToken::Atom),
                "ATOMG.E.ADD.STRONG.GPU",
            ),
            pb::MemoryOpClass::Atomic,
        );

        let mut ldgsts_as_global = global_inst();
        ldgsts_as_global.memory_op_access_kind = None;
        assert_static_class_rejected_in_both_modes(
            &ldgsts_as_global,
            &schedule(
                ArchId::Sm86,
                MemRole::BypassLoad(CacheOpToken::Ldgsts),
                "LDGSTS.E.128 [R1], [R2.64]",
            ),
            pb::MemoryOpClass::AsyncCopy,
        );

        let mut atoms_as_shared = global_inst();
        atoms_as_shared.memory_op_class = Some(pb::MemoryOpClass::Shared as i32);
        atoms_as_shared.memory_op_access_kind = None;
        atoms_as_shared.addresses[0].address_space = Some(pb::MemorySpace::Shared as i32);
        assert_static_class_rejected_in_both_modes(
            &atoms_as_shared,
            &schedule(
                ArchId::Sm120,
                MemRole::BypassLoad(CacheOpToken::Atom),
                "ATOMS.E.ADD",
            ),
            pb::MemoryOpClass::Atomic,
        );

        let mut redg_as_global = global_inst();
        redg_as_global.memory_op_access_kind = None;
        assert_static_class_rejected_in_both_modes(
            &redg_as_global,
            &schedule(
                ArchId::Sm120,
                MemRole::Reduction,
                "REDG.E.ADD.STRONG.GPU [R2.64], R0",
            ),
            pb::MemoryOpClass::Atomic,
        );
    }

    #[test]
    fn explicit_access_kind_must_match_static_direction_in_both_modes() {
        let load_at_store = global_inst();
        assert_static_access_kind_rejected_in_both_modes(
            &load_at_store,
            &schedule(ArchId::Sm86, MemRole::Store, "STG.E [R2.64], R0"),
            pb::MemoryOpAccessKind::Store,
        );

        let mut store_at_load = global_inst();
        store_at_load.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Store as i32);
        assert_static_access_kind_rejected_in_both_modes(
            &store_at_load,
            &schedule(
                ArchId::Sm86,
                MemRole::EligibleLoad(CacheOpToken::LdgECa),
                "LDG.E.CA R0, [R2.64]",
            ),
            pb::MemoryOpAccessKind::Load,
        );

        let mut load_at_shared_store = global_inst();
        load_at_shared_store.memory_op_class = Some(pb::MemoryOpClass::Shared as i32);
        load_at_shared_store.addresses[0].address_space = Some(pb::MemorySpace::Shared as i32);
        assert_static_access_kind_rejected_in_both_modes(
            &load_at_shared_store,
            &schedule(ArchId::Sm86, MemRole::None, "STS [R1], R0"),
            pb::MemoryOpAccessKind::Store,
        );
    }

    #[test]
    fn space_coded_sass_heads_cannot_borrow_another_dynamic_route() {
        let mut shared_atmg = atomic_inst(Some(pb::MemorySpace::Shared));
        shared_atmg.function_unique_id = 1;
        assert_static_route_rejected_in_both_modes(
            &shared_atmg,
            &schedule(
                ArchId::Sm86,
                MemRole::BypassLoad(CacheOpToken::Atom),
                "ATOMG.E.ADD.STRONG.GPU",
            ),
            pb::MemorySpace::Global,
            pb::MemorySpace::Shared,
        );

        let mut global_atoms = atomic_inst(Some(pb::MemorySpace::Global));
        global_atoms.function_unique_id = 1;
        assert_static_route_rejected_in_both_modes(
            &global_atoms,
            &schedule(
                ArchId::Sm86,
                MemRole::BypassLoad(CacheOpToken::Atom),
                "ATOMS.E.ADD",
            ),
            pb::MemorySpace::Shared,
            pb::MemorySpace::Global,
        );

        let mut shared_stg = global_inst();
        shared_stg.memory_op_class = Some(pb::MemoryOpClass::Shared as i32);
        shared_stg.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Store as i32);
        shared_stg.addresses[0].address_space = Some(pb::MemorySpace::Shared as i32);
        assert_static_route_rejected_in_both_modes(
            &shared_stg,
            &schedule(ArchId::Sm86, MemRole::Store, "STG.E [R2.64], R0"),
            pb::MemorySpace::Global,
            pb::MemorySpace::Shared,
        );

        let mut global_sts = global_inst();
        global_sts.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Store as i32);
        assert_static_route_rejected_in_both_modes(
            &global_sts,
            &schedule(ArchId::Sm86, MemRole::None, "STS [R1], R0"),
            pb::MemorySpace::Shared,
            pb::MemorySpace::Global,
        );

        let mut local_ldg = global_inst();
        local_ldg.memory_op_class = Some(pb::MemoryOpClass::Local as i32);
        local_ldg.addresses[0].address_space = Some(pb::MemorySpace::Local as i32);
        assert_static_route_rejected_in_both_modes(
            &local_ldg,
            &schedule(
                ArchId::Sm86,
                MemRole::EligibleLoad(CacheOpToken::LdgECa),
                "LDG.E.CA R0, [R1]",
            ),
            pb::MemorySpace::Global,
            pb::MemorySpace::Local,
        );

        let mut shared_red = atomic_inst(Some(pb::MemorySpace::Shared));
        shared_red.function_unique_id = 1;
        assert_static_route_rejected_in_both_modes(
            &shared_red,
            &schedule(
                ArchId::Sm86,
                MemRole::Reduction,
                "RED.E.ADD.STRONG.GPU [R2.64], R0",
            ),
            pb::MemorySpace::Global,
            pb::MemorySpace::Shared,
        );

        let mut global_reds = atomic_inst(Some(pb::MemorySpace::Global));
        global_reds.function_unique_id = 1;
        assert_static_family_unsupported_in_both_modes(
            &global_reds,
            &schedule(ArchId::Sm86, MemRole::None, "REDS.E.ADD [R1], R0"),
            "REDS",
        );
    }

    #[test]
    fn sm120_dsmem_families_remain_typed_unsupported() {
        let mut redas = atomic_inst(Some(pb::MemorySpace::Global));
        redas.function_unique_id = 1;
        redas.memory_op_class = Some(pb::MemoryOpClass::ClusterDsmem as i32);
        assert_unsupported_memory_path_in_both_modes(
            &redas,
            &schedule(ArchId::Sm120, MemRole::None, "REDAS.E.ADD [R1], R0"),
            pb::MemoryOpClass::ClusterDsmem,
        );

        let mut stas = global_inst();
        stas.memory_op_class = Some(pb::MemoryOpClass::ClusterDsmem as i32);
        stas.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Store as i32);
        assert_unsupported_memory_path_in_both_modes(
            &stas,
            &schedule(ArchId::Sm120, MemRole::None, "STAS [R1], R0"),
            pb::MemoryOpClass::ClusterDsmem,
        );

        let mut lds_mmio = global_inst();
        lds_mmio.memory_op_class = Some(pb::MemoryOpClass::ClusterDsmem as i32);
        assert_unsupported_memory_path_in_both_modes(
            &lds_mmio,
            &schedule(ArchId::Sm120, MemRole::None, "LDS.U.32.MMIO R0, [R1]"),
            pb::MemoryOpClass::ClusterDsmem,
        );
    }

    #[test]
    fn sm120_dsmem_families_reject_modeled_class_contamination() {
        let mut redas_as_atomic = atomic_inst(Some(pb::MemorySpace::Shared));
        redas_as_atomic.function_unique_id = 1;
        assert_static_class_rejected_in_both_modes(
            &redas_as_atomic,
            &schedule(ArchId::Sm120, MemRole::None, "REDAS.E.ADD [R1], R0"),
            pb::MemoryOpClass::ClusterDsmem,
        );

        let mut stas_as_shared = global_inst();
        stas_as_shared.memory_op_class = Some(pb::MemoryOpClass::Shared as i32);
        stas_as_shared.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Store as i32);
        stas_as_shared.addresses[0].address_space = Some(pb::MemorySpace::Shared as i32);
        assert_static_class_rejected_in_both_modes(
            &stas_as_shared,
            &schedule(ArchId::Sm120, MemRole::None, "STAS [R1], R0"),
            pb::MemoryOpClass::ClusterDsmem,
        );

        let mut lds_mmio_as_shared = global_inst();
        lds_mmio_as_shared.memory_op_class = Some(pb::MemoryOpClass::Shared as i32);
        lds_mmio_as_shared.addresses[0].address_space = Some(pb::MemorySpace::Shared as i32);
        assert_static_class_rejected_in_both_modes(
            &lds_mmio_as_shared,
            &schedule(ArchId::Sm120, MemRole::None, "LDS.U.32.MMIO R0, [R1]"),
            pb::MemoryOpClass::ClusterDsmem,
        );
    }

    #[test]
    fn sm86_rejects_post_ampere_static_memory_families() {
        let mut stas = global_inst();
        stas.memory_op_class = Some(pb::MemoryOpClass::Shared as i32);
        stas.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Store as i32);
        stas.addresses[0].address_space = Some(pb::MemorySpace::Shared as i32);
        assert_static_family_unsupported_in_both_modes(
            &stas,
            &schedule(ArchId::Sm86, MemRole::None, "STAS [R1], R0"),
            "STAS",
        );

        let mut redas = atomic_inst(Some(pb::MemorySpace::Shared));
        redas.function_unique_id = 1;
        assert_static_family_unsupported_in_both_modes(
            &redas,
            &schedule(ArchId::Sm86, MemRole::None, "REDAS.E.ADD [R1], R0"),
            "REDAS",
        );

        let mut lds_mmio = global_inst();
        lds_mmio.memory_op_class = Some(pb::MemoryOpClass::Shared as i32);
        lds_mmio.addresses[0].address_space = Some(pb::MemorySpace::Shared as i32);
        assert_static_family_unsupported_in_both_modes(
            &lds_mmio,
            &schedule(ArchId::Sm86, MemRole::None, "LDS.U.32.MMIO R0, [R1]"),
            "LDS.MMIO",
        );

        let mut stsm = global_inst();
        stsm.memory_op_class = Some(pb::MemoryOpClass::Shared as i32);
        stsm.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Store as i32);
        stsm.addresses[0].address_space = Some(pb::MemorySpace::Shared as i32);
        assert_static_family_unsupported_in_both_modes(
            &stsm,
            &schedule(ArchId::Sm86, MemRole::None, "STSM [R1], R0"),
            "STSM",
        );

        let mut redg = atomic_inst(Some(pb::MemorySpace::Global));
        redg.function_unique_id = 1;
        assert_static_family_unsupported_in_both_modes(
            &redg,
            &schedule(
                ArchId::Sm86,
                MemRole::Reduction,
                "REDG.E.ADD.STRONG.GPU [R2.64], R0",
            ),
            "REDG",
        );

        let mut ldgmc_as_shared = global_inst();
        ldgmc_as_shared.memory_op_class = Some(pb::MemoryOpClass::Shared as i32);
        ldgmc_as_shared.addresses[0].address_space = Some(pb::MemorySpace::Shared as i32);
        assert_static_family_unsupported_in_both_modes(
            &ldgmc_as_shared,
            &schedule(ArchId::Sm86, MemRole::None, "LDGMC R0, [R2.64]"),
            "LDGMC",
        );
    }

    #[test]
    fn sm120_ldgmc_and_redg_do_not_gain_sm86_hierarchy_semantics() {
        let ldgmc = global_inst();
        assert_static_family_unsupported_in_both_modes(
            &ldgmc,
            &schedule(ArchId::Sm120, MemRole::None, "LDGMC R0, [R2.64]"),
            "LDGMC",
        );

        let mut redg = atomic_inst(Some(pb::MemorySpace::Global));
        redg.function_unique_id = 1;
        assert_unsupported_memory_path_in_both_modes(
            &redg,
            &schedule(
                ArchId::Sm120,
                MemRole::Reduction,
                "REDG.E.ADD.STRONG.GPU [R2.64], R0",
            ),
            pb::MemoryOpClass::Atomic,
        );
    }

    #[test]
    fn sm120_stsm_uses_shared_timing_only_on_its_supported_architecture() {
        let mut inst = global_inst();
        inst.memory_op_class = Some(pb::MemoryOpClass::Shared as i32);
        inst.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Store as i32);
        inst.addresses[0].address_space = Some(pb::MemorySpace::Shared as i32);
        let sched = schedule(ArchId::Sm120, MemRole::None, "STSM [R1], R0");

        assert!(matches!(
            classify_flat(&inst, &sched, sched.arch, Path::new("stsm.pb"), 0, 0),
            Ok(AccessKind::Shared)
        ));
        let mut sectors = vec![16];
        let mut sector_scratch = vec![105];
        assert!(matches!(
            classify_with_l1(
                &inst,
                &sched,
                sched.arch,
                &mut sectors,
                &mut sector_scratch,
                Path::new("stsm.pb"),
                0,
                0,
            ),
            Ok(AccessKind::Shared)
        ));
        assert_eq!(sectors, vec![16]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn absent_legacy_access_kind_remains_admissible() {
        let sched = schedule(
            ArchId::Sm86,
            MemRole::BypassLoad(CacheOpToken::Ldgsts),
            "LDGSTS.E.128",
        );
        let mut inst = async_copy_inst(vec![spaced_address(pb::MemorySpace::Global, 0x1000)]);
        inst.memory_op_access_kind = None;

        assert!(matches!(
            classify_flat(&inst, &sched, sched.arch, Path::new("legacy_kind.pb"), 0, 0),
            Ok(AccessKind::GlobalBypass {
                off: 0,
                len: 0,
                token: CacheOpToken::Ldgsts,
            })
        ));
        let mut sectors = vec![11];
        let mut sector_scratch = Vec::new();
        assert!(matches!(
            classify_with_l1(
                &inst,
                &sched,
                sched.arch,
                &mut sectors,
                &mut sector_scratch,
                Path::new("legacy_kind.pb"),
                0,
                0,
            ),
            Ok(AccessKind::GlobalBypass {
                off: 1,
                len: 1,
                token: CacheOpToken::Ldgsts,
            })
        ));
        assert_eq!(sectors, vec![11, 0x1000 >> 5]);

        let store_sched = schedule(ArchId::Sm86, MemRole::Store, "STG.E [R2.64], R0");
        let mut legacy_global_store = global_inst();
        legacy_global_store.memory_op_access_kind = None;
        assert!(matches!(
            classify_flat(
                &legacy_global_store,
                &store_sched,
                store_sched.arch,
                Path::new("legacy_global_store.pb"),
                0,
                0,
            ),
            Ok(AccessKind::GlobalStore { off: 0, len: 0 })
        ));
        let mut store_sectors = vec![12];
        let mut store_scratch = Vec::new();
        assert!(matches!(
            classify_with_l1(
                &legacy_global_store,
                &store_sched,
                store_sched.arch,
                &mut store_sectors,
                &mut store_scratch,
                Path::new("legacy_global_store.pb"),
                0,
                0,
            ),
            Ok(AccessKind::GlobalStore { off: 1, len: 1 })
        ));
        assert_eq!(store_sectors, vec![12, 0x1000 >> 5]);
    }

    #[test]
    fn capability_empty_legacy_routes_skip_shape_but_validate_explicit_space() {
        let atomic_sched = schedule(
            ArchId::Sm86,
            MemRole::BypassLoad(CacheOpToken::Atom),
            "ATOMG.E.ADD.STRONG.GPU",
        );
        let mut addressed_atomic = atomic_inst(None);
        addressed_atomic.function_unique_id = 1;
        assert!(matches!(
            classify_flat_with_contract(
                &addressed_atomic,
                &atomic_sched,
                atomic_sched.arch,
                MemoryPayloadContract::LegacyExclusion,
                Path::new("legacy_atomic.pb"),
                0,
                0,
            ),
            Ok(AccessKind::GlobalBypass {
                off: 0,
                len: 0,
                token: CacheOpToken::Atom,
            })
        ));
        let mut atomic_sectors = vec![20];
        let mut atomic_scratch = vec![99];
        assert!(matches!(
            classify_with_l1_with_contract(
                &addressed_atomic,
                &atomic_sched,
                atomic_sched.arch,
                MemoryPayloadContract::LegacyExclusion,
                &mut atomic_sectors,
                &mut atomic_scratch,
                Path::new("legacy_atomic.pb"),
                0,
                0,
            ),
            Ok(AccessKind::GlobalBypass {
                off: 1,
                len: 1,
                token: CacheOpToken::Atom,
            })
        ));
        assert_eq!(atomic_sectors, vec![20, 0x1000 >> 5]);

        let mut addressless_atomic = addressed_atomic.clone();
        addressless_atomic.addresses.clear();
        assert!(
            classify_flat_with_contract(
                &addressless_atomic,
                &atomic_sched,
                atomic_sched.arch,
                MemoryPayloadContract::LegacyExclusion,
                Path::new("legacy_atomic_empty.pb"),
                0,
                0,
            )
            .is_ok()
        );

        let async_sched = schedule(
            ArchId::Sm86,
            MemRole::BypassLoad(CacheOpToken::Ldgsts),
            "LDGSTS.E.128 [R1], [R2.64]",
        );
        let addressless_async = async_copy_inst(Vec::new());
        assert!(
            classify_flat_with_contract(
                &addressless_async,
                &async_sched,
                async_sched.arch,
                MemoryPayloadContract::LegacyExclusion,
                Path::new("legacy_async_empty.pb"),
                0,
                0,
            )
            .is_ok()
        );
        let mut missing_space_async =
            async_copy_inst(vec![spaced_address(pb::MemorySpace::Global, 0x1000)]);
        missing_space_async.addresses[0].address_space = None;
        assert!(
            classify_flat_with_contract(
                &missing_space_async,
                &async_sched,
                async_sched.arch,
                MemoryPayloadContract::LegacyExclusion,
                Path::new("legacy_async_missing_space.pb"),
                0,
                0,
            )
            .is_ok()
        );

        let explicit_shared_async =
            async_copy_inst(vec![spaced_address(pb::MemorySpace::Shared, 0x40)]);
        assert!(matches!(
            classify_flat_with_contract(
                &explicit_shared_async,
                &async_sched,
                async_sched.arch,
                MemoryPayloadContract::LegacyExclusion,
                Path::new("legacy_async_shared.pb"),
                0,
                0,
            ),
            Err(ReplayError::MemoryRoute {
                class: pb::MemoryOpClass::AsyncCopy,
                source: MemoryRouteError::AsyncCopySourceSpaceInvalid {
                    space: pb::MemorySpace::Shared,
                },
                ..
            })
        ));

        let mut async_sectors = vec![21];
        let mut async_scratch = vec![99];
        assert!(matches!(
            classify_with_l1_with_contract(
                &addressless_async,
                &async_sched,
                async_sched.arch,
                MemoryPayloadContract::LegacyExclusion,
                &mut async_sectors,
                &mut async_scratch,
                Path::new("legacy_async_empty.pb"),
                0,
                0,
            ),
            Ok(AccessKind::GlobalBypass {
                off: 1,
                len: 0,
                token: CacheOpToken::Ldgsts,
            })
        ));
        assert_eq!(async_sectors, vec![21]);
        assert!(async_scratch.is_empty());
    }

    #[test]
    fn async_copy_source_only_routes_global_with_exact_sectors() {
        let sched = schedule(
            ArchId::Sm86,
            MemRole::BypassLoad(CacheOpToken::Ldgsts),
            "LDGSTS.E.128",
        );
        let inst = async_copy_inst(vec![spaced_address(pb::MemorySpace::Global, 0x1000)]);

        let flat = classify_flat(&inst, &sched, sched.arch, Path::new("source.pb"), 0, 0)
            .expect("source-only LDGSTS has a reviewed flat route");
        assert!(matches!(
            flat,
            AccessKind::GlobalBypass {
                off: 0,
                len: 0,
                token: CacheOpToken::Ldgsts,
            }
        ));

        let mut sectors = vec![29];
        let mut sector_scratch = vec![99];
        let l1 = classify_with_l1(
            &inst,
            &sched,
            sched.arch,
            &mut sectors,
            &mut sector_scratch,
            Path::new("source.pb"),
            0,
            0,
        )
        .expect("source-only LDGSTS has a reviewed L1 route");
        assert!(matches!(
            l1,
            AccessKind::GlobalBypass {
                off: 1,
                len: 1,
                token: CacheOpToken::Ldgsts,
            }
        ));
        assert_eq!(sectors, vec![29, 0x1000 >> 5]);
        assert_eq!(sector_scratch, vec![0x1000 >> 5]);
    }

    #[test]
    fn zero_active_async_copy_source_stays_global_without_arena_append() {
        let sched = schedule(
            ArchId::Sm86,
            MemRole::BypassLoad(CacheOpToken::Ldgsts),
            "LDGSTS.E.128",
        );
        let mut source = spaced_address(pb::MemorySpace::Global, 0x1000);
        source.compression_format = sim_trace::lanes::FORMAT_BASE_DELTA;
        source.addrs.clear();
        let mut inst = async_copy_inst(vec![source]);
        inst.predicate_mask = Some(0);

        let flat = classify_flat(&inst, &sched, sched.arch, Path::new("zero_source.pb"), 0, 0)
            .expect("zero-active source retains the global route");
        assert!(matches!(
            flat,
            AccessKind::GlobalBypass {
                off: 0,
                len: 0,
                token: CacheOpToken::Ldgsts,
            }
        ));

        let mut sectors = vec![31];
        let mut sector_scratch = vec![99];
        let l1 = classify_with_l1(
            &inst,
            &sched,
            sched.arch,
            &mut sectors,
            &mut sector_scratch,
            Path::new("zero_source.pb"),
            0,
            0,
        )
        .expect("zero-active source retains the L1 global route");
        assert!(matches!(
            l1,
            AccessKind::GlobalBypass {
                off: 1,
                len: 0,
                token: CacheOpToken::Ldgsts,
            }
        ));
        assert_eq!(sectors, vec![31]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn async_copy_rejects_shared_or_mixed_operands_in_both_modes() {
        let sched = schedule(
            ArchId::Sm86,
            MemRole::BypassLoad(CacheOpToken::Ldgsts),
            "LDGSTS.E.128",
        );
        let cases = [
            vec![spaced_address(pb::MemorySpace::Shared, 0x2040)],
            vec![
                spaced_address(pb::MemorySpace::Global, 0x1000),
                spaced_address(pb::MemorySpace::Shared, 0x2040),
            ],
            vec![
                spaced_address(pb::MemorySpace::Shared, 0x2040),
                spaced_address(pb::MemorySpace::Global, 0x1000),
            ],
        ];
        for (case, addresses) in cases.into_iter().enumerate() {
            let inst = async_copy_inst(addresses);
            let flat = classify_flat(
                &inst,
                &sched,
                sched.arch,
                Path::new("invalid_source.pb"),
                2,
                case,
            )
            .unwrap_err();
            assert!(
                matches!(
                    flat,
                    ReplayError::MemoryRoute {
                        warp_id: 2,
                        record_ix,
                        class: pb::MemoryOpClass::AsyncCopy,
                        source: MemoryRouteError::AsyncCopySourceSpaceInvalid {
                            space: pb::MemorySpace::Shared,
                        },
                        ..
                    } if case == 0 && record_ix == case
                ) || matches!(
                    flat,
                    ReplayError::MemoryRoute {
                        warp_id: 2,
                        record_ix,
                        class: pb::MemoryOpClass::AsyncCopy,
                        source: MemoryRouteError::AsyncCopySourceCount { got: 2 },
                        ..
                    } if case > 0 && record_ix == case
                )
            );

            let mut sectors = vec![37];
            let mut sector_scratch = vec![99];
            let l1 = classify_with_l1(
                &inst,
                &sched,
                sched.arch,
                &mut sectors,
                &mut sector_scratch,
                Path::new("invalid_source.pb"),
                2,
                case,
            )
            .unwrap_err();
            assert!(
                matches!(
                    l1,
                    ReplayError::MemoryRoute {
                        warp_id: 2,
                        record_ix,
                        class: pb::MemoryOpClass::AsyncCopy,
                        source: MemoryRouteError::AsyncCopySourceSpaceInvalid {
                            space: pb::MemorySpace::Shared,
                        },
                        ..
                    } if case == 0 && record_ix == case
                ) || matches!(
                    l1,
                    ReplayError::MemoryRoute {
                        warp_id: 2,
                        record_ix,
                        class: pb::MemoryOpClass::AsyncCopy,
                        source: MemoryRouteError::AsyncCopySourceCount { got: 2 },
                        ..
                    } if case > 0 && record_ix == case
                )
            );
            assert_eq!(sectors, vec![37]);
            assert!(sector_scratch.is_empty());
        }
    }

    #[test]
    fn shared_atomic_routes_to_shared_in_flat_and_l1_modes() {
        // A shared atomic (op_class=ATOMIC, address_space=SHARED) must NOT
        // reach the global path, including on the template architecture.
        let sched = schedule(
            ArchId::Sm120,
            MemRole::BypassLoad(CacheOpToken::Atom),
            "ATOMS.E.ADD",
        );
        let mut inst = atomic_inst(Some(pb::MemorySpace::Shared));
        inst.function_unique_id = 1;
        let flat = classify_flat(&inst, &sched, sched.arch, Path::new("t"), 0, 0).unwrap();
        assert!(matches!(flat, AccessKind::Shared));

        let mut sectors = vec![41];
        let mut sector_scratch = vec![99];
        let l1 = classify_with_l1(
            &inst,
            &sched,
            sched.arch,
            &mut sectors,
            &mut sector_scratch,
            Path::new("t"),
            0,
            0,
        )
        .unwrap();
        assert!(matches!(l1, AccessKind::Shared));
        assert_eq!(sectors, vec![41]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn global_atomic_requires_reviewed_sm86_sidecar_role_in_both_modes() {
        let sched = schedule(
            ArchId::Sm86,
            MemRole::BypassLoad(CacheOpToken::Atom),
            "ATOMG.E.CAS.STRONG.GPU",
        );
        let mut inst = atomic_inst(Some(pb::MemorySpace::Global));
        inst.function_unique_id = 1;
        let flat = classify_flat(&inst, &sched, sched.arch, Path::new("t"), 0, 0).unwrap();
        assert!(matches!(
            flat,
            AccessKind::GlobalBypass {
                off: 0,
                len: 0,
                token: CacheOpToken::Atom,
            }
        ));

        let mut sectors = vec![43];
        let mut sector_scratch = vec![99];
        let l1 = classify_with_l1(
            &inst,
            &sched,
            sched.arch,
            &mut sectors,
            &mut sector_scratch,
            Path::new("t"),
            0,
            0,
        )
        .unwrap();
        assert!(matches!(
            l1,
            AccessKind::GlobalBypass {
                off: 1,
                len: 1,
                token: CacheOpToken::Atom,
            }
        ));
        assert_eq!(sectors, vec![43, 0x1000 >> 5]);
        assert_eq!(sector_scratch, vec![0x1000 >> 5]);
    }

    #[test]
    fn global_reduction_remains_distinct_in_both_modes() {
        let sched = schedule(
            ArchId::Sm86,
            MemRole::Reduction,
            "RED.E.ADD.STRONG.GPU [R2.64], R0",
        );
        let mut inst = atomic_inst(Some(pb::MemorySpace::Global));
        inst.function_unique_id = 1;
        let flat = classify_flat(&inst, &sched, sched.arch, Path::new("red.pb"), 0, 0)
            .expect("reviewed reduction has a global atomic route");
        assert!(matches!(
            flat,
            AccessKind::GlobalReduction { off: 0, len: 0 }
        ));

        let mut sectors = vec![45];
        let mut sector_scratch = vec![99];
        let l1 = classify_with_l1(
            &inst,
            &sched,
            sched.arch,
            &mut sectors,
            &mut sector_scratch,
            Path::new("red.pb"),
            0,
            0,
        )
        .expect("reviewed reduction retains its static reduction role");
        assert!(matches!(l1, AccessKind::GlobalReduction { off: 1, len: 1 }));
        assert_eq!(sectors, vec![45, 0x1000 >> 5]);
        assert_eq!(sector_scratch, vec![0x1000 >> 5]);
    }

    #[test]
    fn addressed_atomic_without_space_is_typed_rejected_in_both_modes() {
        let sched = schedule(
            ArchId::Sm86,
            MemRole::BypassLoad(CacheOpToken::Atom),
            "ATOMG.E.CAS.STRONG.GPU",
        );
        let inst = atomic_inst(None);
        assert!(matches!(
            classify_flat(&inst, &sched, sched.arch, Path::new("t"), 0, 0),
            Err(ReplayError::MemoryRoute {
                class: pb::MemoryOpClass::Atomic,
                source: MemoryRouteError::AtomicSpaceMissing { operand: 0 },
                ..
            })
        ));

        let mut sectors = vec![47];
        let mut sector_scratch = vec![99];
        assert!(matches!(
            classify_with_l1(
                &inst,
                &sched,
                sched.arch,
                &mut sectors,
                &mut sector_scratch,
                Path::new("t"),
                0,
                0,
            ),
            Err(ReplayError::MemoryRoute {
                class: pb::MemoryOpClass::Atomic,
                source: MemoryRouteError::AtomicSpaceMissing { operand: 0 },
                ..
            })
        ));
        assert_eq!(sectors, vec![47]);
        assert!(sector_scratch.is_empty());
    }

    #[test]
    fn flat_global_async_copy_requires_reviewed_sm86_sidecar_role() {
        let inst = async_copy_inst(vec![spaced_address(pb::MemorySpace::Global, 0x1000)]);
        let sched = schedule(
            ArchId::Sm86,
            MemRole::BypassLoad(CacheOpToken::Ldgsts),
            "LDGSTS.E.128",
        );
        let k = classify_flat(&inst, &sched, sched.arch, Path::new("t"), 0, 0).unwrap();
        assert!(matches!(
            k,
            AccessKind::GlobalBypass {
                off: 0,
                len: 0,
                token: CacheOpToken::Ldgsts,
            }
        ));
    }

    #[test]
    fn template_global_atomic_is_typed_unsupported() {
        let sched = schedule(
            ArchId::Sm120,
            MemRole::BypassLoad(CacheOpToken::Atom),
            "ATOMG.E.CAS.STRONG.GPU",
        );
        let err = classify_flat(
            &atomic_inst(Some(pb::MemorySpace::Global)),
            &sched,
            sched.arch,
            Path::new("t"),
            0,
            0,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ReplayError::UnsupportedMemoryPath {
                arch: ArchId::Sm120,
                class: pb::MemoryOpClass::Atomic,
            }
        ));
    }

    #[test]
    fn every_unmodeled_memory_class_fails_closed() {
        for class in [
            pb::MemoryOpClass::Texture,
            pb::MemoryOpClass::Surface,
            pb::MemoryOpClass::Tma,
            pb::MemoryOpClass::ClusterDsmem,
            pb::MemoryOpClass::Tmem,
        ] {
            let mut inst = atomic_inst(Some(pb::MemorySpace::Global));
            inst.memory_op_class = Some(class as i32);
            let sched = schedule(ArchId::Sm86, MemRole::None, "IADD3");
            let err = classify_flat(&inst, &sched, sched.arch, Path::new("t"), 0, 0).unwrap_err();
            assert!(matches!(
                err,
                ReplayError::UnsupportedMemoryPath {
                    arch: ArchId::Sm86,
                    class: got,
                } if got == class
            ));
        }
    }

    #[test]
    fn unknown_global_static_opcode_fails_closed() {
        let mut inst = atomic_inst(Some(pb::MemorySpace::Global));
        inst.memory_op_class = Some(pb::MemoryOpClass::Global as i32);
        inst.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Load as i32);
        let sched = schedule(ArchId::Sm86, MemRole::None, "IADD3 R0, R1, R2, RZ");
        let err = classify_flat(&inst, &sched, sched.arch, Path::new("t"), 0, 0).unwrap_err();
        assert!(matches!(
            err,
            ReplayError::UnknownGlobalOpcode {
                arch: ArchId::Sm86,
                pc: 0,
                opcode,
            } if opcode == "IADD3"
        ));
    }

    #[test]
    fn known_untagged_load_opcode_remains_denominator_only() {
        let mut inst = atomic_inst(Some(pb::MemorySpace::Global));
        inst.memory_op_class = Some(pb::MemoryOpClass::Global as i32);
        inst.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Load as i32);
        let sched = schedule(ArchId::Sm86, MemRole::None, "LD.E R0, [R2.64]");
        let k = classify_flat(&inst, &sched, sched.arch, Path::new("t"), 0, 0).unwrap();
        assert!(matches!(k, AccessKind::GlobalUntagged { .. }));
    }
}

#[cfg(test)]
mod replay_identity_tests {
    //! The allocation-identity gate is enforced on the path the simulator
    //! actually consumes (extract_per_sm_blocks), not only in a helper: a
    //! malformed GLOBAL-space global-atomic or LDGSTS-source record fails closed
    //! BEFORE replay classification.
    use super::*;
    use crate::control_bits::ControlBits;
    use crate::program::{FunctionSchedule, IssueSlot, OpClass};
    use prost::Message;
    use sim_trace::reader::ADDRESS_CARRYING_TRACE_CONTRACT_ID;
    use std::collections::BTreeMap as StdMap;

    /// A GLOBAL-space memory instruction of `op_class` with NO allocation_id —
    /// malformed under the address-carrying contract's identity rule.
    fn malformed_global_inst(op_class: pb::MemoryOpClass) -> pb::Instruction {
        let mut addrs = vec![0x7000_0000i64];
        addrs.extend(std::iter::repeat(LANE_INACTIVE_SENTINEL).take(31));
        let access_kind = match op_class {
            pb::MemoryOpClass::Atomic => pb::MemoryOpAccessKind::Atomic,
            pb::MemoryOpClass::AsyncCopy => pb::MemoryOpAccessKind::AsyncCopy,
            _ => pb::MemoryOpAccessKind::Load,
        };
        pb::Instruction {
            pc: 16,
            active_mask: Some(0b1),
            predicate_mask: Some(0b1),
            function_unique_id: 1,
            vpc: Some(16),
            sm_id: Some(0),
            warpid_sm: Some(0),
            warpid_tb: Some(0),
            data_width: Some(4),
            memory_op_class: Some(op_class as i32),
            memory_op_access_kind: Some(access_kind as i32),
            addresses: vec![pb::Address {
                data_width: 4,
                compression_format: 0, // list_all
                addrs,
                allocation_id: None, // malformed: GLOBAL active record, no id
                address_space: Some(pb::MemorySpace::Global as i32),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn write_trace_with(dir: &std::path::Path, inst: pb::Instruction) {
        let kernel = pb::Kernel {
            id: 1,
            name: "k".into(),
            function_unique_id: 1,
            size_shared_memory: 0,
            number_of_registers: 32,
            shared_memory_base_address: 0,
            local_memory_base_address: 0,
            grid_dim: Some(pb::Dim3d { x: 1, y: 1, z: 1 }),
            block_dim: Some(pb::Dim3d { x: 32, y: 1, z: 1 }),
            cluster_dim: Some(pb::Dim3d { x: 1, y: 1, z: 1 }),
        };
        let stream = pb::CudaStream {
            id: 0,
            kernels: vec![kernel],
            ordered_cuda_events: vec![],
        };
        let device = pb::GpuDevice {
            id: 0,
            streams: [(0i64, stream)].into_iter().collect(),
        };
        let trace = pb::Trace {
            name: "m".into(),
            binary_version: 1,
            nvbit_version: "1.8".into(),
            accelsim_version: 0,
            is_gathered_registers_values: false,
            gpu_device: [(0i64, device)].into_iter().collect(),
            trace_contract_id: ADDRESS_CARRYING_TRACE_CONTRACT_ID.into(),
            memory_payload_classes: vec![
                pb::MemoryOpClass::Global as i32,
                pb::MemoryOpClass::Shared as i32,
                pb::MemoryOpClass::Local as i32,
                pb::MemoryOpClass::Atomic as i32,
                pb::MemoryOpClass::AsyncCopy as i32,
            ],
        };
        std::fs::write(dir.join("dynamic_trace.pb"), trace.encode_to_vec()).unwrap();
        // A VALID sidecar, so a missing id is AllocationIdMissing (not
        // AllocationsSidecarMissing) — the record itself is what is malformed.
        let extra = dir.join("extra_info");
        std::fs::create_dir_all(&extra).unwrap();
        std::fs::write(
            extra.join("allocations.json"),
            r#"[{"allocation_id":1,"context_id":1,"device_id":0,"base_address":"0x70000000","size_bytes":4096,"alive":true}]"#,
        )
        .unwrap();
        let tb_dir = dir.join("threadblocks/device_0/stream_0/kernel_1");
        std::fs::create_dir_all(&tb_dir).unwrap();
        let warp = pb::Warp {
            id: 0,
            instructions: vec![inst],
        };
        let tb = pb::Threadblock {
            block_id: Some(pb::Dim3d { x: 0, y: 0, z: 0 }),
            warps: [(0i32, warp)].into_iter().collect(),
        };
        std::fs::write(tb_dir.join("d_0_s_0_k_1_0,0,0.pb"), tb.encode_to_vec()).unwrap();
    }

    fn replay_error_with_schedule(inst: pb::Instruction, sched: FunctionSchedule) -> ReplayError {
        let tmp = tempfile::tempdir().unwrap();
        write_trace_with(tmp.path(), inst);
        let td = TraceDir::open(tmp.path()).unwrap();
        let loc = td.kernels().next().unwrap();
        extract_per_sm_blocks(&td, &loc, &sched, true).unwrap_err()
    }

    fn replay_error(inst: pb::Instruction) -> ReplayError {
        replay_error_with_schedule(
            inst,
            FunctionSchedule {
                arch: ArchId::Sm86,
                function_unique_id: 1,
                kernel_name: "k".into(),
                function_name: "f".into(),
                by_pc: StdMap::new(),
            },
        )
    }

    fn assert_replay_rejects_identity(inst: pb::Instruction) {
        let err = replay_error(inst);
        assert!(
            matches!(
                err,
                ReplayError::Trace(TraceError::AllocationIdMissing { .. })
            ),
            "expected AllocationIdMissing, got {err:?}"
        );
    }

    #[test]
    fn replay_reports_unknown_space_before_missing_identity() {
        let mut inst = malformed_global_inst(pb::MemoryOpClass::Global);
        inst.addresses[0].address_space = Some(99);
        assert!(matches!(
            replay_error(inst),
            ReplayError::Trace(TraceError::UnknownEnumValue {
                field: "address_space",
                raw: 99,
                ..
            })
        ));
    }

    #[test]
    fn replay_reports_later_lane_error_before_earlier_missing_identity() {
        let mut inst = malformed_global_inst(pb::MemoryOpClass::Global);
        let mut malformed_later = inst.addresses[0].clone();
        malformed_later.addrs.clear();
        inst.addresses.push(malformed_later);
        assert!(matches!(
            replay_error(inst),
            ReplayError::Trace(TraceError::Lane {
                source: sim_trace::lanes::LaneError::ListAllLengthInvalid { got: 0 },
                ..
            })
        ));
    }

    #[test]
    fn foreign_function_precedes_allocation_identity_after_route_validation() {
        let mut foreign = malformed_global_inst(pb::MemoryOpClass::Global);
        foreign.function_unique_id = 2;
        assert!(matches!(
            replay_error(foreign),
            ReplayError::MultiFunctionUnsupported {
                expected: 1,
                found: 2,
                ..
            }
        ));

        let mut malformed_route = malformed_global_inst(pb::MemoryOpClass::Atomic);
        malformed_route.function_unique_id = 2;
        malformed_route
            .addresses
            .push(malformed_route.addresses[0].clone());
        assert!(matches!(
            replay_error(malformed_route),
            ReplayError::MemoryRoute {
                class: pb::MemoryOpClass::Atomic,
                source: MemoryRouteError::AtomicOperandCount { got: 2 },
                ..
            }
        ));
    }

    #[test]
    fn replay_route_shape_precedes_allocation_identity() {
        let mut multi_atomic = malformed_global_inst(pb::MemoryOpClass::Atomic);
        multi_atomic
            .addresses
            .push(multi_atomic.addresses[0].clone());
        assert!(matches!(
            replay_error(multi_atomic),
            ReplayError::MemoryRoute {
                class: pb::MemoryOpClass::Atomic,
                source: MemoryRouteError::AtomicOperandCount { got: 2 },
                ..
            }
        ));

        let shared_with_global_operand = malformed_global_inst(pb::MemoryOpClass::Shared);
        assert!(matches!(
            replay_error(shared_with_global_operand),
            ReplayError::MemoryRoute {
                class: pb::MemoryOpClass::Shared,
                source: MemoryRouteError::SharedLocalSpaceInvalid {
                    operand: 0,
                    space: pb::MemorySpace::Global,
                },
                ..
            }
        ));
    }

    #[test]
    fn replay_access_kind_mismatch_precedes_allocation_identity() {
        let mut inst = malformed_global_inst(pb::MemoryOpClass::AsyncCopy);
        inst.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Atomic as i32);
        assert!(matches!(
            replay_error(inst),
            ReplayError::MemoryRoute {
                class: pb::MemoryOpClass::AsyncCopy,
                source: MemoryRouteError::AccessKindInvalid {
                    got: pb::MemoryOpAccessKind::Atomic,
                    expected: "ASYNC_COPY",
                },
                ..
            }
        ));
    }

    #[test]
    fn replay_static_kind_mismatch_precedes_allocation_identity() {
        let sched = FunctionSchedule {
            arch: ArchId::Sm86,
            function_unique_id: 1,
            kernel_name: "k".into(),
            function_name: "f".into(),
            by_pc: StdMap::from([(
                16,
                IssueSlot {
                    control: ControlBits::new(0, false, 7, 7, 0, 0).expect("valid test controls"),
                    op_class: OpClass::Other,
                    mem_role: MemRole::Store,
                    opcode_text: "STG.E [R2.64], R0".into(),
                },
            )]),
        };
        assert!(matches!(
            replay_error_with_schedule(malformed_global_inst(pb::MemoryOpClass::Global), sched,),
            ReplayError::MemoryRoute {
                class: pb::MemoryOpClass::Global,
                source: MemoryRouteError::StaticAccessKindMismatch {
                    got: pb::MemoryOpAccessKind::Load,
                    expected: pb::MemoryOpAccessKind::Store,
                },
                ..
            }
        ));
    }

    #[test]
    fn replay_static_class_mismatch_precedes_allocation_identity_without_access_kind() {
        let sched = FunctionSchedule {
            arch: ArchId::Sm86,
            function_unique_id: 1,
            kernel_name: "k".into(),
            function_name: "f".into(),
            by_pc: StdMap::from([(
                16,
                IssueSlot {
                    control: ControlBits::new(0, false, 7, 7, 0, 0).expect("valid test controls"),
                    op_class: OpClass::Other,
                    mem_role: MemRole::Reduction,
                    opcode_text: "RED.E.ADD.STRONG.GPU [R2.64], R0".into(),
                },
            )]),
        };
        let mut inst = malformed_global_inst(pb::MemoryOpClass::Global);
        inst.memory_op_access_kind = None;
        assert!(matches!(
            replay_error_with_schedule(inst, sched),
            ReplayError::MemoryRoute {
                class: pb::MemoryOpClass::Global,
                source: MemoryRouteError::StaticMemoryClassMismatch {
                    expected: pb::MemoryOpClass::Atomic,
                    got: pb::MemoryOpClass::Global,
                },
                ..
            }
        ));
    }

    #[test]
    fn replay_static_route_mismatch_precedes_allocation_identity() {
        let sched = FunctionSchedule {
            arch: ArchId::Sm86,
            function_unique_id: 1,
            kernel_name: "k".into(),
            function_name: "f".into(),
            by_pc: StdMap::from([(
                16,
                IssueSlot {
                    control: ControlBits::new(0, false, 7, 7, 0, 0).expect("valid test controls"),
                    op_class: OpClass::Other,
                    mem_role: MemRole::BypassLoad(CacheOpToken::Atom),
                    opcode_text: "ATOMS.E.ADD".into(),
                },
            )]),
        };
        assert!(matches!(
            replay_error_with_schedule(malformed_global_inst(pb::MemoryOpClass::Atomic), sched,),
            ReplayError::MemoryRoute {
                class: pb::MemoryOpClass::Atomic,
                source: MemoryRouteError::StaticMemoryRouteMismatch {
                    expected: pb::MemorySpace::Shared,
                    got: pb::MemorySpace::Global,
                },
                ..
            }
        ));
    }

    #[test]
    fn replay_uses_legacy_payload_capability_for_atomic_fallback() {
        let tmp = tempfile::tempdir().unwrap();
        let mut inst = malformed_global_inst(pb::MemoryOpClass::Atomic);
        inst.addresses[0].address_space = None;
        write_trace_with(tmp.path(), inst);

        let root_path = tmp.path().join("dynamic_trace.pb");
        let mut root = pb::Trace::decode(std::fs::read(&root_path).unwrap().as_slice()).unwrap();
        root.trace_contract_id.clear();
        root.memory_payload_classes.clear();
        std::fs::write(&root_path, root.encode_to_vec()).unwrap();

        let td = TraceDir::open(tmp.path()).unwrap();
        assert_eq!(
            td.memory_payload_contract(),
            MemoryPayloadContract::LegacyExclusion
        );
        let loc = td.kernels().next().unwrap();
        let sched = FunctionSchedule {
            arch: ArchId::Sm86,
            function_unique_id: 1,
            kernel_name: "k".into(),
            function_name: "f".into(),
            by_pc: StdMap::from([(
                16,
                IssueSlot {
                    control: ControlBits::new(0, false, 7, 7, 0, 0).expect("valid test controls"),
                    op_class: OpClass::Other,
                    mem_role: MemRole::BypassLoad(CacheOpToken::Atom),
                    opcode_text: "ATOMG.E.ADD.STRONG.GPU".into(),
                },
            )]),
        };
        let (blocks, records) = extract_per_sm_blocks(&td, &loc, &sched, true)
            .expect("legacy capability retains the pre-transition atomic fallback");
        assert_eq!(records, 1);
        let warp = &blocks[&0][0].warps[0];
        assert!(matches!(
            warp.insts[0].kind,
            AccessKind::GlobalBypass {
                off: 0,
                len: 1,
                token: CacheOpToken::Atom,
            }
        ));
        assert_eq!(warp.sectors, vec![0x7000_0000 >> 5]);
    }

    #[test]
    fn replay_rejects_malformed_global_atomic() {
        assert_replay_rejects_identity(malformed_global_inst(pb::MemoryOpClass::Atomic));
    }

    #[test]
    fn replay_rejects_malformed_ldgsts_source() {
        assert_replay_rejects_identity(malformed_global_inst(pb::MemoryOpClass::AsyncCopy));
    }
}

#[cfg(test)]
mod completeness_tests {
    //! Reader/replay completeness fail-closed on the path replay ACTUALLY
    //! consumes: an incomplete trace lacking a trace_sampling.json sidecar is
    //! E_INCOMPLETE_TRACE; a subset trace passes iff the sidecar's original_grid
    //! matches the launch gridDim (never rewritten) and records.captured matches
    //! records_seen. A lossless-full sidecar may be present, but never sanctions
    //! incomplete coverage.
    use super::*;
    use crate::program::FunctionSchedule;
    use prost::Message;
    use std::collections::BTreeMap as StdMap;

    #[test]
    fn completeness_grid_overflow_is_typed() {
        let wire_max = i32::MAX as u32;
        assert!(matches!(
            checked_grid_total((wire_max, wire_max, wire_max)),
            Err(ReplayError::GridSizeOverflow { grid }) if grid == (wire_max, wire_max, wire_max)
        ));
    }

    /// A minimal non-memory record (no memory_op_class) carrying placement, so it
    /// increments records_seen without engaging the allocation-identity gate.
    fn simple_inst() -> pb::Instruction {
        pb::Instruction {
            pc: 16,
            active_mask: Some(0b1),
            predicate_mask: Some(0b1),
            function_unique_id: 1,
            vpc: Some(16),
            sm_id: Some(0),
            warpid_sm: Some(0),
            warpid_tb: Some(0),
            ..Default::default()
        }
    }

    fn write_grid_trace(
        dir: &std::path::Path,
        grid_x: i32,
        grid_y: i32,
        block_xs: &[i32],
        sidecar: Option<&str>,
    ) {
        let kernel = pb::Kernel {
            id: 1,
            name: "_Z1kv".into(),
            function_unique_id: 1,
            size_shared_memory: 0,
            number_of_registers: 32,
            shared_memory_base_address: 0,
            local_memory_base_address: 0,
            grid_dim: Some(pb::Dim3d {
                x: grid_x,
                y: grid_y,
                z: 1,
            }),
            block_dim: Some(pb::Dim3d { x: 32, y: 1, z: 1 }),
            cluster_dim: Some(pb::Dim3d { x: 1, y: 1, z: 1 }),
        };
        let stream = pb::CudaStream {
            id: 0,
            kernels: vec![kernel],
            ordered_cuda_events: vec![],
        };
        let device = pb::GpuDevice {
            id: 0,
            streams: [(0i64, stream)].into_iter().collect(),
        };
        let trace = pb::Trace {
            name: "m".into(),
            binary_version: 1,
            nvbit_version: "1.8".into(),
            accelsim_version: 0,
            is_gathered_registers_values: false,
            gpu_device: [(0i64, device)].into_iter().collect(),
            trace_contract_id: String::new(),
            memory_payload_classes: vec![],
        };
        std::fs::write(dir.join("dynamic_trace.pb"), trace.encode_to_vec()).unwrap();
        let tb_dir = dir.join("threadblocks/device_0/stream_0/kernel_1");
        std::fs::create_dir_all(&tb_dir).unwrap();
        for &bx in block_xs {
            let warp = pb::Warp {
                id: 0,
                instructions: vec![simple_inst()],
            };
            let tb = pb::Threadblock {
                block_id: Some(pb::Dim3d { x: bx, y: 0, z: 0 }),
                warps: [(0i32, warp)].into_iter().collect(),
            };
            std::fs::write(
                tb_dir.join(format!("d_0_s_0_k_1_{bx},0,0.pb")),
                tb.encode_to_vec(),
            )
            .unwrap();
        }
        if let Some(json) = sidecar {
            std::fs::write(dir.join("trace_sampling.json"), json).unwrap();
        }
    }

    // Literal producer fixtures generated and sealed by Python's sidecar
    // implementation.  Consumer tests therefore exercise the real canonical
    // contract instead of manufacturing a Rust-only approximation.
    const PYTHON_SEALED_SUBSET: &str = r#"{"schema_version":2,"cell_id":"c","kernel":{"mangled_name":"_Z1kv","args_hash":"ab"},"method":"wave_segment_stratified_subset","reason":"byte_budget","shape":{"original_grid":{"x":60,"y":1,"z":1},"original_block":{"x":32,"y":1,"z":1},"captured_grid":{"x":60,"y":1,"z":1},"captured_block":{"x":32,"y":1,"z":1},"reduced_shape_variant":null},"gates":{"coop":false,"wave_count":60,"n_measured_waves":40,"tail_measured_waves":2,"atomic_density":0.0,"hotline_share":0.0,"skew_top1pct_share":0.0,"gini":0.0,"per_sm_block_count_cv":0.0,"n_phases":1,"skew_heavy_block_count":1,"certainty_bytes":100,"max_trace_bytes":2000,"certainty_budget_exceeded":false,"verdict":"subset_admissible"},"wave_model":{"reconstruction":"launch_index","per_sm_resident_blocks":[{"sm_id":0,"resident_blocks":1}],"n_waves":60,"segment_len":3,"ghost_depth":1},"per_sm_slot_map":[{"sm_id":0,"block_ids":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59]}],"rng":{"algo":"pcg64","seed":0},"tracer":{"version":5,"features":{"atomic_addr":true,"ldgsts_addr":true},"stats_pass_version":2},"bytes":{"raw":310,"compressed":200,"codec":"zstd","level":19,"long":27,"dict_id":null},"records":{"expected_full":6000,"captured":12},"hashes":{"selected_block_ids_sha256":"84b7ed43903cca6302ed15cf78ff507c45e575f16d7d5c12c975c5727edc37cf","decompressed_payload_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","sidecar_sha256":"331103782174538149454c35590a56994e29ffe65aa4614378df607cf993b4d9"},"label_pairing":"full_shape_label","segments":[{"id":0,"block_ids":[0,1,2]},{"id":19,"block_ids":[57,58,59]},{"id":1,"block_ids":[3,4,5]},{"id":2,"block_ids":[6,7,8]}],"selected_segment_ids":[0,19,1,2],"certainty_segments":[0,19],"force_include_blocks":[0],"strata":[{"id":0,"phase_id":0,"launch_tercile":0,"segment_ids":[1,2],"N_h":6,"n_h":2,"pi":0.3333333333333333,"sector_stddev":2.0,"neyman_weight":12.0}],"selection_reconciliation":{"status":"fixed_point","attempts":1,"max_attempts":3,"capture_stats_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},"selection_budget":{"max_trace_bytes":2000,"basis":"raw_threadblock_pb","certainty_fraction_numerator":2,"certainty_fraction_denominator":5,"certainty_bytes":100},"selection_auxiliary":{"name":"l2_sector_transactions_32b","sector_bytes":32,"scope":"pass2_all_blocks","expected_full":400,"selected":200},"estimator":{"mode":"weighted_full_shape_estimate","targets":["sectors"],"not_recoverable":["cycles"],"convergence":{"status":"converged"}}}"#;

    const PYTHON_SEALED_LOSSLESS: &str = r#"{"schema_version":2,"cell_id":"c","kernel":{"mangled_name":"_Z1kv","args_hash":"deadbeef"},"method":"lossless_compressed_full","reason":"byte_budget","shape":{"original_grid":{"x":60,"y":1,"z":1},"original_block":{"x":32,"y":1,"z":1},"captured_grid":{"x":60,"y":1,"z":1},"captured_block":{"x":32,"y":1,"z":1},"reduced_shape_variant":null},"gates":{"coop":false,"wave_count":60,"n_measured_waves":40,"tail_measured_waves":2,"atomic_density":0.0,"hotline_share":0.0,"skew_top1pct_share":0.0,"gini":0.0,"per_sm_block_count_cv":0.0,"n_phases":1,"skew_heavy_block_count":1,"certainty_bytes":null,"max_trace_bytes":null,"certainty_budget_exceeded":false,"verdict":"subset_admissible"},"wave_model":{"reconstruction":"launch_index","per_sm_resident_blocks":[{"sm_id":0,"resident_blocks":1}],"n_waves":60,"segment_len":3,"ghost_depth":1},"per_sm_slot_map":[{"sm_id":0,"block_ids":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59]}],"rng":{"algo":"pcg64","seed":0},"tracer":{"version":5,"features":{"atomic_addr":true,"ldgsts_addr":true},"stats_pass_version":1},"bytes":{"raw":1000,"compressed":200,"codec":"zstd","level":19,"long":27,"dict_id":null},"records":{"expected_full":6000,"captured":60},"hashes":{"selected_block_ids_sha256":"3f3ffcf42accae1591be118baf57efb9b8d7c78028dc566674c809f210ad796a","decompressed_payload_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","sidecar_sha256":"657fd9dc2f3f0a826dfb4bb030028ea32d315af1d37b54b4ff8bc9891f54aac8"},"label_pairing":"full_shape_label"}"#;

    const SELECTED_BLOCKS: [i32; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 57, 58, 59];

    fn payload_bound_subset() -> String {
        PYTHON_SEALED_SUBSET
            .replace(
                &"a".repeat(64),
                "df85bddbdf77a2f2a75a54d3769476919fcd062c5308b03b1a389c2d3fb21355",
            )
            .replace(
                "331103782174538149454c35590a56994e29ffe65aa4614378df607cf993b4d9",
                "322b93482bd788380a65a23061ea283c7fd695de4dabe2a8126e72f474ed64b9",
            )
    }

    fn subset_with_captured_mismatch() -> String {
        payload_bound_subset()
            .replace(r#""captured":12"#, r#""captured":13"#)
            .replace(
                "322b93482bd788380a65a23061ea283c7fd695de4dabe2a8126e72f474ed64b9",
                "8ae92ee1c1d0e1ecdfd4b834d562f8983b5c191611298573c474e7b92fb7ff48",
            )
    }

    fn payload_bound_subset_for_different_blocks() -> String {
        payload_bound_subset()
            .replace(
                "df85bddbdf77a2f2a75a54d3769476919fcd062c5308b03b1a389c2d3fb21355",
                "aaa5e80fd15d9ccc91163b7495f449d7a684c70857f7f7bdb34b38ceb22595f5",
            )
            .replace(
                "322b93482bd788380a65a23061ea283c7fd695de4dabe2a8126e72f474ed64b9",
                "b22df41d037e55bd9568e2f436bf7225717e4e6ce6be701617643fed8996894e",
            )
    }

    fn minimal_fake_sidecar() -> String {
        format!(
            r#"{{"schema_version":2,"method":"wave_segment_stratified_subset","shape":{{"original_grid":{{"x":60,"y":1,"z":1}}}},"records":{{"captured":12}},"hashes":{{"selected_block_ids_sha256":"{a}","decompressed_payload_sha256":"{a}","sidecar_sha256":"{a}"}},"selection_reconciliation":{{"status":"fixed_point","attempts":1,"max_attempts":3,"capture_stats_sha256":"{a}"}}}}"#,
            a = "a".repeat(64)
        )
    }

    fn rewrite_launch_metadata(
        dir: &std::path::Path,
        kernel_name: Option<&str>,
        block_x: Option<i32>,
    ) {
        let path = dir.join("dynamic_trace.pb");
        let mut trace = pb::Trace::decode(std::fs::read(&path).unwrap().as_slice()).unwrap();
        let kernel = &mut trace
            .gpu_device
            .get_mut(&0)
            .unwrap()
            .streams
            .get_mut(&0)
            .unwrap()
            .kernels[0];
        if let Some(name) = kernel_name {
            kernel.name = name.to_string();
        }
        if let Some(x) = block_x {
            kernel.block_dim.as_mut().unwrap().x = x;
        }
        std::fs::write(path, trace.encode_to_vec()).unwrap();
    }

    fn run(dir: &std::path::Path) -> Result<(), ReplayError> {
        let td = TraceDir::open(dir)?;
        let loc = td.kernels().next().unwrap();
        let sched = FunctionSchedule {
            arch: ArchId::Sm86,
            function_unique_id: 1,
            kernel_name: "_Z1kv".into(),
            function_name: "f".into(),
            by_pc: StdMap::new(),
        };
        extract_per_sm_blocks(&td, &loc, &sched, true).map(|_| ())
    }

    #[test]
    fn complete_trace_no_sidecar_ok() {
        let tmp = tempfile::tempdir().unwrap();
        write_grid_trace(tmp.path(), 4, 1, &[0, 1, 2, 3], None);
        let result = run(tmp.path());
        assert!(result.is_ok(), "{result:?}");
    }

    #[test]
    fn complete_legacy_trace_without_block_dim_needs_no_sampling_binding() {
        let tmp = tempfile::tempdir().unwrap();
        write_grid_trace(tmp.path(), 4, 1, &[0, 1, 2, 3], None);
        let path = tmp.path().join("dynamic_trace.pb");
        let mut trace = pb::Trace::decode(std::fs::read(&path).unwrap().as_slice()).unwrap();
        trace
            .gpu_device
            .get_mut(&0)
            .unwrap()
            .streams
            .get_mut(&0)
            .unwrap()
            .kernels[0]
            .block_dim = None;
        std::fs::write(path, trace.encode_to_vec()).unwrap();
        let result = run(tmp.path());
        assert!(result.is_ok(), "{result:?}");
    }

    #[test]
    fn incomplete_trace_no_sidecar_is_e_incomplete_trace() {
        let tmp = tempfile::tempdir().unwrap();
        write_grid_trace(tmp.path(), 4, 1, &[0, 1], None); // 2/4 blocks, no sidecar
        let err = run(tmp.path()).unwrap_err();
        assert!(
            matches!(
                err,
                ReplayError::IncompleteTrace {
                    seen: 2,
                    grid_total: 4
                }
            ),
            "expected IncompleteTrace, got {err:?}"
        );
    }

    #[test]
    fn subset_with_valid_sidecar_ok() {
        let tmp = tempfile::tempdir().unwrap();
        let sidecar = payload_bound_subset();
        write_grid_trace(tmp.path(), 60, 1, &SELECTED_BLOCKS, Some(&sidecar));
        let result = run(tmp.path());
        assert!(result.is_ok(), "{result:?}");
    }

    #[test]
    fn lossless_full_sidecar_cannot_sanction_incomplete_coverage() {
        let tmp = tempfile::tempdir().unwrap();
        write_grid_trace(
            tmp.path(),
            60,
            1,
            &SELECTED_BLOCKS,
            Some(PYTHON_SEALED_LOSSLESS),
        );
        let err = run(tmp.path()).unwrap_err();
        assert!(
            matches!(
                err,
                ReplayError::IncompleteTrace {
                    seen: 12,
                    grid_total: 60
                }
            ),
            "expected IncompleteTrace, got {err:?}"
        );
    }

    #[test]
    fn complete_trace_with_lossless_full_sidecar_ok() {
        let tmp = tempfile::tempdir().unwrap();
        let all_blocks: Vec<i32> = (0..60).collect();
        write_grid_trace(tmp.path(), 60, 1, &all_blocks, Some(PYTHON_SEALED_LOSSLESS));
        assert!(run(tmp.path()).is_ok());
    }

    #[test]
    fn subset_with_griddim_rewrite_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let sidecar = payload_bound_subset();
        write_grid_trace(tmp.path(), 61, 1, &SELECTED_BLOCKS, Some(&sidecar));
        let err = run(tmp.path()).unwrap_err();
        assert!(
            matches!(err, ReplayError::GridDimRewritten { .. }),
            "expected GridDimRewritten, got {err:?}"
        );
    }

    #[test]
    fn subset_with_records_captured_mismatch_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let sidecar = subset_with_captured_mismatch();
        write_grid_trace(tmp.path(), 60, 1, &SELECTED_BLOCKS, Some(&sidecar));
        let err = run(tmp.path()).unwrap_err();
        assert!(
            matches!(
                err,
                ReplayError::RecordsCapturedMismatch {
                    records_seen: 12,
                    captured: 13
                }
            ),
            "expected RecordsCapturedMismatch, got {err:?}"
        );
    }

    #[test]
    fn minimal_fake_subset_is_rejected_on_consumer_path() {
        let tmp = tempfile::tempdir().unwrap();
        let sidecar = minimal_fake_sidecar();
        write_grid_trace(tmp.path(), 60, 1, &SELECTED_BLOCKS, Some(&sidecar));
        let err = run(tmp.path()).unwrap_err();
        assert!(matches!(
            err,
            ReplayError::Trace(TraceError::SamplingSidecarMalformed { .. })
        ));
    }

    #[test]
    fn bad_seal_block_hash_and_duplicate_membership_are_rejected_on_consumer_path() {
        let valid = payload_bound_subset();
        let cases = [
            valid.replace(r#""captured":12"#, r#""captured":13"#),
            valid.replace(
                "84b7ed43903cca6302ed15cf78ff507c45e575f16d7d5c12c975c5727edc37cf",
                &"0".repeat(64),
            ),
            valid.replace(
                r#"{"id":1,"block_ids":[3,4,5]}"#,
                r#"{"id":1,"block_ids":[2,4,5]}"#,
            ),
        ];
        for sidecar in cases {
            let tmp = tempfile::tempdir().unwrap();
            write_grid_trace(tmp.path(), 60, 1, &SELECTED_BLOCKS, Some(&sidecar));
            let err = run(tmp.path()).unwrap_err();
            assert!(matches!(
                err,
                ReplayError::Trace(TraceError::SamplingSidecarMalformed { .. })
            ));
        }
    }

    #[test]
    fn sealed_sidecar_for_different_blocks_cannot_sanction_payload() {
        let tmp = tempfile::tempdir().unwrap();
        let sidecar = payload_bound_subset_for_different_blocks();
        let different_blocks = [0, 1, 2, 3, 4, 5, 6, 7, 9, 57, 58, 59];
        write_grid_trace(tmp.path(), 60, 1, &different_blocks, Some(&sidecar));
        let err = run(tmp.path()).unwrap_err();
        assert!(
            matches!(
                err,
                ReplayError::SelectedBlockMembershipMismatch {
                    seen: 12,
                    declared: 12
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn tampered_payload_is_rejected_before_replay() {
        let tmp = tempfile::tempdir().unwrap();
        let sidecar = payload_bound_subset();
        write_grid_trace(tmp.path(), 60, 1, &SELECTED_BLOCKS, Some(&sidecar));
        let path = tmp
            .path()
            .join("threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_0,0,0.pb");
        let mut raw = std::fs::read(&path).unwrap();
        raw[0] ^= 1;
        std::fs::write(path, raw).unwrap();
        assert!(matches!(
            run(tmp.path()).unwrap_err(),
            ReplayError::Trace(TraceError::SamplingSidecarMalformed { .. })
        ));
    }

    #[test]
    fn sidecar_kernel_and_block_shape_are_bound_to_launch() {
        let sidecar = payload_bound_subset();

        let wrong_kernel = tempfile::tempdir().unwrap();
        write_grid_trace(wrong_kernel.path(), 60, 1, &SELECTED_BLOCKS, Some(&sidecar));
        rewrite_launch_metadata(wrong_kernel.path(), Some("_Zother"), None);
        assert!(matches!(
            run(wrong_kernel.path()).unwrap_err(),
            ReplayError::SamplingKernelMismatch { .. }
        ));

        let wrong_block = tempfile::tempdir().unwrap();
        write_grid_trace(wrong_block.path(), 60, 1, &SELECTED_BLOCKS, Some(&sidecar));
        rewrite_launch_metadata(wrong_block.path(), None, Some(64));
        assert!(matches!(
            run(wrong_block.path()).unwrap_err(),
            ReplayError::BlockDimRewritten { .. }
        ));
    }

    #[test]
    fn multidim_out_of_grid_cta_is_not_aliased_into_full_coverage() {
        // Grid (2,2,1): CTAs (2,0,0) and (3,0,0) flatten to 2 and 3 (inside
        // [0,4)) but are out of grid on the x axis. They must NOT be accepted as
        // valid blocks (which would falsely satisfy the completeness gate as
        // "4/4 covered"); the shared block_launch_index rejects them typed.
        let tmp = tempfile::tempdir().unwrap();
        write_grid_trace(tmp.path(), 2, 2, &[0, 1, 2, 3], None);
        let err = run(tmp.path()).unwrap_err();
        assert!(
            matches!(
                err,
                ReplayError::Trace(TraceError::BlockIdOutOfRange { .. })
            ),
            "expected BlockIdOutOfRange (not a false full-coverage pass), got {err:?}"
        );
    }
}
