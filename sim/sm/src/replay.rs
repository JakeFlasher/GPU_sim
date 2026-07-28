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
use sim_mem::{L1Config, L1CounterError, L1Counters, SectorExpansionError, push_unique_sectors};
use sim_trace::lanes::LANE_INACTIVE_SENTINEL;
use sim_trace::pb;
use sim_trace::{
    KernelLoc, TraceDir, TraceError, block_launch_index, check_allocation_identity, kernel_grid,
    read_threadblock, visit_memory_access,
};

use sim_mem::{
    CacheOpToken, L2Config, L2Counters, L2Model, MemRole, is_known_untagged_opcode, opcode_head,
};

use crate::issue::{
    AccessKind, BlockReplay, IssueError, IssueModelParams, ReplayInst, SmSimResult, WarpReplay,
    simulate_sm,
};
use crate::program::FunctionSchedule;
use crate::system::{DramBw, DramStats, L2SourceSplit, SystemError, simulate_kernel_shared};

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
        "global memory record at pc {pc:#x} on architecture {arch} has unknown static opcode head {opcode:?}"
    )]
    UnknownGlobalOpcode {
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
    /// Directional v0 proxy for `gpc__cycles_elapsed.max`.
    pub max_cycles: u64,
    /// Kernel-level L1 counters (sum over per-SM caches) in L1 mode.
    pub l1: Option<L1Counters>,
    /// Shared-L2 counters (the `replay_kernel_shared` path only).
    pub l2: Option<L2Counters>,
    /// Per-source L2 split from the driver-side emission observer.
    pub l2_by_source: Option<L2SourceSplit>,
    /// DRAM read-traffic observer (shared mode only).
    pub dram: Option<DramStats>,
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
            for (record_ix, inst) in warp.instructions.iter().enumerate() {
                records_seen = checked_record_count(records_seen, &path, warp_id, record_ix)?;
                // Allocation-identity gate on the path the simulator actually
                // consumes: a GLOBAL/GENERIC-space active record (ordinary global,
                // global atomic, LDGSTS source) must carry a resolvable, in-range
                // allocation id — parity with the Python validator and the
                // TraceDir reader. Fail-closed BEFORE classification so a
                // malformed record never reaches the L1/L2 model.
                check_allocation_identity(inst, td.allocations(), || {
                    format!("{} warp={warp_id} inst={record_ix}", path.display())
                })?;
                if inst.function_unique_id != expected_fuid {
                    return Err(ReplayError::MultiFunctionUnsupported {
                        path,
                        warp_id,
                        record_ix,
                        expected: expected_fuid,
                        found: inst.function_unique_id,
                    });
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
                    classify_with_l1(
                        inst,
                        schedule,
                        schedule.arch,
                        &mut sectors,
                        &path,
                        warp_id,
                        record_ix,
                    )?
                } else {
                    classify_flat(inst, schedule, schedule.arch, &path, warp_id, record_ix)?
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
    l2: Option<(L2Counters, L2SourceSplit, DramStats)>,
) -> Result<KernelReplayResult, ReplayError> {
    let mut instructions_issued: u64 = 0;
    let mut max_cycles: u64 = 0;
    let mut l1_totals = expect_l1.then(|| L1Counters::for_arch(arch));
    for (&sm_id, result) in &per_sm {
        instructions_issued = instructions_issued
            .checked_add(result.instructions_issued)
            .ok_or(ReplayError::InstructionCountOverflow {
                sm_id,
                accumulated: instructions_issued,
                addend: result.instructions_issued,
            })?;
        max_cycles = max_cycles.max(result.cycles);
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
        max_cycles,
        l1: l1_totals,
        l2,
        l2_by_source,
        dram,
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
                Some((sys.l2, sys.l2_by_source, sys.dram)),
            )
        }
    }
}

/// Resolve the two special global classes that are reviewed on the active
/// sm_86 path. The dynamic enum alone is not enough: the static sidecar must
/// prove the matching cache-op token before the record enters replay.
fn require_sm86_special_role(
    schedule: &FunctionSchedule,
    arch: ArchId,
    pc: u32,
    class: pb::MemoryOpClass,
) -> Result<CacheOpToken, ReplayError> {
    if arch != ArchId::Sm86 {
        return Err(ReplayError::UnsupportedMemoryPath { arch, class });
    }
    let role = schedule
        .by_pc
        .get(&u64::from(pc))
        .map(|slot| slot.mem_role)
        .ok_or(ReplayError::UnsupportedMemoryPath { arch, class })?;
    let expected = match class {
        pb::MemoryOpClass::Atomic => CacheOpToken::Atom,
        pb::MemoryOpClass::AsyncCopy => CacheOpToken::Ldgsts,
        _ => return Err(ReplayError::UnsupportedMemoryPath { arch, class }),
    };
    if matches!(role, MemRole::BypassLoad(token) if token == expected) {
        Ok(expected)
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

/// path rejects via `visit_memory_access` are rejected here with the
/// SAME typed diagnoses (a malformed record must never pass one mode
/// and fail the other).
///
/// Note: flat mode reuses `GlobalUntagged { off: 0, len: 0 }` as its
/// "global-class, sectors not decoded" encoding; `AccessKind` consumers
/// must not read flat-mode replays as untagged-pc statistics (no L1
/// counters exist in this mode).
fn classify_flat(
    inst: &pb::Instruction,
    schedule: &FunctionSchedule,
    arch: ArchId,
    path: &std::path::Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<AccessKind, ReplayError> {
    let record = || format!("{} warp={warp_id} inst={record_ix}", path.display());
    let Some(raw) = inst.memory_op_class else {
        // Memory-ness contract: address payloads without a memory class
        // are a typed violation (mirrors the reader).
        if inst.addresses.is_empty() {
            return Ok(AccessKind::NonMemory);
        }
        return Err(TraceError::MemoryOpClassMissing {
            record: record(),
            n_addresses: inst.addresses.len(),
        }
        .into());
    };
    let class = pb::MemoryOpClass::try_from(raw).map_err(|_| TraceError::UnknownEnumValue {
        record: record(),
        field: "memory_op_class",
        raw,
    })?;
    match class {
        pb::MemoryOpClass::Unspecified => {
            // Contract violation, NOT schema skew — same diagnosis as
            // the reader path.
            Err(TraceError::MemoryOpClassUnspecified { record: record() }.into())
        }
        pb::MemoryOpClass::Shared | pb::MemoryOpClass::Local => Ok(AccessKind::Shared),
        pb::MemoryOpClass::Global => {
            // Route a shared-space record out of the global path even in
            // flat mode where sectors are not decoded. Per-address space is
            // authoritative when present; absent falls back to global.
            let shared_space = inst.addresses.iter().any(|a| {
                matches!(
                    a.address_space
                        .and_then(|r| pb::MemorySpace::try_from(r).ok()),
                    Some(pb::MemorySpace::Shared) | Some(pb::MemorySpace::Local)
                )
            });
            if shared_space {
                Ok(AccessKind::Shared)
            } else {
                require_known_global_role(schedule, arch, inst.pc)?;
                Ok(AccessKind::GlobalUntagged { off: 0, len: 0 })
            }
        }
        pb::MemoryOpClass::Atomic | pb::MemoryOpClass::AsyncCopy => {
            let shared_space = inst.addresses.iter().any(|a| {
                matches!(
                    a.address_space
                        .and_then(|r| pb::MemorySpace::try_from(r).ok()),
                    Some(pb::MemorySpace::Shared) | Some(pb::MemorySpace::Local)
                )
            });
            if shared_space {
                return Ok(AccessKind::Shared);
            }
            require_sm86_special_role(schedule, arch, inst.pc, class)?;
            Ok(AccessKind::GlobalUntagged { off: 0, len: 0 })
        }
        pb::MemoryOpClass::Texture
        | pb::MemoryOpClass::Surface
        | pb::MemoryOpClass::Tma
        | pb::MemoryOpClass::ClusterDsmem
        | pb::MemoryOpClass::Tmem => Err(ReplayError::UnsupportedMemoryPath { arch, class }),
    }
}

/// Full classification in L1 mode: wire-contract decode (streaming form),
/// per-instruction unique-sector expansion into the warp arena, and the
/// join-time [`MemRole`] split.
fn classify_with_l1(
    inst: &pb::Instruction,
    schedule: &FunctionSchedule,
    arch: ArchId,
    sectors: &mut Vec<i64>,
    path: &std::path::Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<AccessKind, ReplayError> {
    let start = sectors.len();
    let record = || format!("{} warp={warp_id} inst={record_ix}", path.display());
    let mut effective_space: Option<pb::MemorySpace> = None;
    let mut sector_error: Option<SectorExpansionError> = None;
    let visit_result = visit_memory_access(inst, record, |a| {
        if sector_error.is_some() {
            return;
        }
        if effective_space.is_none() {
            effective_space = a.address_space;
        }
        if let Err(source) = push_unique_sectors(
            &a.lane_vector,
            (0..a.lane_vector.len()).filter(|&i| a.lane_vector[i] != LANE_INACTIVE_SENTINEL),
            a.data_width_bytes,
            sectors,
            start,
        ) {
            sectors.truncate(start);
            sector_error = Some(source);
        }
    });
    if let Some(source) = sector_error {
        return Err(ReplayError::SectorExpansion {
            path: path.to_path_buf(),
            warp_id,
            record_ix,
            source,
        });
    }
    let meta = match visit_result {
        Ok(meta) => meta,
        Err(source) => {
            sectors.truncate(start);
            return Err(source.into());
        }
    };
    let Some(meta) = meta else {
        return Ok(AccessKind::NonMemory);
    };
    // Route by EFFECTIVE per-address space, not the instruction-level op_class:
    // a shared atomic (ATOMS) carries op_class=ATOMIC but address_space=SHARED,
    // and must not enter the global L1TEX/L2 sector stream. The per-address
    // space is authoritative when present; absent (legacy traces) falls back to
    // the instruction-level Shared class.
    let shared_space = matches!(
        effective_space,
        Some(pb::MemorySpace::Shared) | Some(pb::MemorySpace::Local)
    );
    if shared_space {
        sectors.truncate(start);
        return match meta.op_class {
            pb::MemoryOpClass::Unspecified => {
                Err(TraceError::MemoryOpClassUnspecified { record: record() }.into())
            }
            pb::MemoryOpClass::Shared
            | pb::MemoryOpClass::Local
            | pb::MemoryOpClass::Global
            | pb::MemoryOpClass::Atomic
            | pb::MemoryOpClass::AsyncCopy => Ok(AccessKind::Shared),
            class => Err(ReplayError::UnsupportedMemoryPath { arch, class }),
        };
    }
    match meta.op_class {
        pb::MemoryOpClass::Unspecified => {
            sectors.truncate(start);
            return Err(TraceError::MemoryOpClassUnspecified { record: record() }.into());
        }
        pb::MemoryOpClass::Shared | pb::MemoryOpClass::Local => {
            sectors.truncate(start);
            return Ok(AccessKind::Shared);
        }
        pb::MemoryOpClass::Atomic | pb::MemoryOpClass::AsyncCopy => {
            require_sm86_special_role(schedule, arch, inst.pc, meta.op_class)?;
        }
        pb::MemoryOpClass::Texture
        | pb::MemoryOpClass::Surface
        | pb::MemoryOpClass::Tma
        | pb::MemoryOpClass::ClusterDsmem
        | pb::MemoryOpClass::Tmem => {
            sectors.truncate(start);
            return Err(ReplayError::UnsupportedMemoryPath {
                arch,
                class: meta.op_class,
            });
        }
        pb::MemoryOpClass::Global => {}
    }
    let count = sectors.len() - start;
    let overflow = |sectors: usize| ReplayError::SectorCountOverflow {
        path: path.to_path_buf(),
        warp_id,
        record_ix,
        sectors,
    };
    let off = u32::try_from(start).map_err(|_| overflow(start))?;
    let len = u16::try_from(count).map_err(|_| overflow(count))?;
    // A pc absent from the schedule surfaces as the issue loop's typed
    // PcNotInSchedule; classification stays neutral in that case. A present
    // but unrecognized static opcode fails closed instead of becoming a
    // generic untagged global access.
    let role = require_known_global_role(schedule, arch, inst.pc)?;
    Ok(match role {
        MemRole::EligibleLoad(token) => AccessKind::GlobalEligible { off, len, token },
        MemRole::BypassLoad(token) => AccessKind::GlobalBypass { off, len, token },
        MemRole::Store => AccessKind::GlobalStore { off, len },
        MemRole::None => AccessKind::GlobalUntagged { off, len },
    })
}

#[cfg(test)]
mod aggregation_tests {
    use super::*;

    fn sm(instructions_issued: u64, l1: Option<L1Counters>) -> SmSimResult {
        SmSimResult {
            instructions_issued,
            cycles: 1,
            l1,
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
            fold_kernel_result(ArchId::Sm86, per_sm, 0, false, None),
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
            fold_kernel_result(ArchId::Sm86, per_sm, 5, false, None),
            Err(ReplayError::CountMismatch {
                issued: 4,
                records: 5,
            })
        ));
    }

    #[test]
    fn first_l1_snapshot_must_have_matching_provenance() {
        let unbound = BTreeMap::from([(0, sm(1, Some(L1Counters::default())))]);
        assert!(matches!(
            fold_kernel_result(ArchId::Sm86, unbound, 1, true, None),
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
            fold_kernel_result(ArchId::Sm86, wrong_arch, 1, true, None),
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
            fold_kernel_result(ArchId::Sm86, all_none, 1, true, None),
            Err(ReplayError::L1PresenceMismatch {
                sm_id: 0,
                expected: true,
                found: false,
            })
        ));

        let all_some = BTreeMap::from([(0, sm(1, Some(L1Counters::for_arch(ArchId::Sm86))))]);
        assert!(matches!(
            fold_kernel_result(ArchId::Sm86, all_some, 1, false, None),
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
            fold_kernel_result(ArchId::Sm86, missing_second, 2, true, None),
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
            fold_kernel_result(ArchId::Sm86, unexpected_second, 2, false, None),
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
        let mut second = L1Counters::for_arch(ArchId::Sm86);
        second.hits = 1;
        let per_sm = BTreeMap::from([(0, sm(1, Some(first))), (1, sm(1, Some(second)))]);
        assert!(matches!(
            fold_kernel_result(ArchId::Sm86, per_sm, 2, true, None),
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
        let mut a = pb::Address::default();
        a.addrs = vec![0x1000];
        a.address_space = space.map(|s| s as i32);
        pb::Instruction {
            memory_op_class: Some(pb::MemoryOpClass::Atomic as i32),
            addresses: vec![a],
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
                    control: ControlBits {
                        stall: 0,
                        yield_flag: false,
                        write_barrier: 7,
                        read_barrier: 7,
                        wait_mask: 0,
                        reuse: 0,
                    },
                    op_class: OpClass::Other,
                    mem_role: role,
                    opcode_text: opcode_text.into(),
                },
            )]),
        }
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
        let err = classify_with_l1(
            &inst,
            &schedule(ArchId::Sm86, MemRole::None, "LDG.E"),
            ArchId::Sm86,
            &mut sectors,
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
    fn sector_expansion_error_precedes_later_trace_error() {
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
                    address_space: Some(pb::MemorySpace::Global as i32),
                    ..Default::default()
                },
            ],
            memory_op_class: Some(pb::MemoryOpClass::Global as i32),
            data_width: Some(2),
            ..Default::default()
        };
        let mut sectors = vec![13];
        assert!(matches!(
            classify_with_l1(
                &inst,
                &schedule(ArchId::Sm86, MemRole::None, "LDG.E"),
                ArchId::Sm86,
                &mut sectors,
                Path::new("precedence.pb"),
                2,
                4,
            ),
            Err(ReplayError::SectorExpansion {
                source: SectorExpansionError::AddressOverflow {
                    lane: 0,
                    base: i64::MAX,
                    data_width_bytes: 2,
                },
                ..
            })
        ));
        assert_eq!(sectors, vec![13]);
    }

    #[test]
    fn flat_shared_atomic_routes_to_shared_not_global() {
        // A shared atomic (op_class=ATOMIC, address_space=SHARED) must NOT
        // reach the global path, including on the template architecture.
        let sched = schedule(ArchId::Sm120, MemRole::None, "ATOMS.E.ADD");
        let k = classify_flat(
            &atomic_inst(Some(pb::MemorySpace::Shared)),
            &sched,
            sched.arch,
            Path::new("t"),
            0,
            0,
        )
        .unwrap();
        assert!(matches!(k, AccessKind::Shared));
    }

    #[test]
    fn flat_global_atomic_requires_reviewed_sm86_sidecar_role() {
        let sched = schedule(
            ArchId::Sm86,
            MemRole::BypassLoad(CacheOpToken::Atom),
            "ATOMG.E.CAS.STRONG.GPU",
        );
        let k = classify_flat(
            &atomic_inst(Some(pb::MemorySpace::Global)),
            &sched,
            sched.arch,
            Path::new("t"),
            0,
            0,
        )
        .unwrap();
        assert!(matches!(k, AccessKind::GlobalUntagged { .. }));
    }

    #[test]
    fn flat_legacy_atomic_without_space_uses_reviewed_global_path() {
        let sched = schedule(
            ArchId::Sm86,
            MemRole::BypassLoad(CacheOpToken::Atom),
            "ATOMG.E.CAS.STRONG.GPU",
        );
        let k =
            classify_flat(&atomic_inst(None), &sched, sched.arch, Path::new("t"), 0, 0).unwrap();
        assert!(matches!(k, AccessKind::GlobalUntagged { .. }));
    }

    #[test]
    fn flat_global_async_copy_requires_reviewed_sm86_sidecar_role() {
        let mut inst = atomic_inst(Some(pb::MemorySpace::Global));
        inst.memory_op_class = Some(pb::MemoryOpClass::AsyncCopy as i32);
        let sched = schedule(
            ArchId::Sm86,
            MemRole::BypassLoad(CacheOpToken::Ldgsts),
            "LDGSTS.E.128",
        );
        let k = classify_flat(&inst, &sched, sched.arch, Path::new("t"), 0, 0).unwrap();
        assert!(matches!(k, AccessKind::GlobalUntagged { .. }));
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
    use crate::program::FunctionSchedule;
    use prost::Message;
    use std::collections::BTreeMap as StdMap;

    /// A GLOBAL-space memory instruction of `op_class` with NO allocation_id —
    /// malformed under the address-carrying contract's identity rule.
    fn malformed_global_inst(op_class: pb::MemoryOpClass) -> pb::Instruction {
        let mut addrs = vec![0x7000_0000i64];
        addrs.extend(std::iter::repeat(LANE_INACTIVE_SENTINEL).take(31));
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
            memory_op_access_kind: Some(pb::MemoryOpAccessKind::Atomic as i32),
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
            trace_contract_id: String::new(),
            memory_payload_classes: vec![],
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

    fn assert_replay_rejects_identity(inst: pb::Instruction) {
        let tmp = tempfile::tempdir().unwrap();
        write_trace_with(tmp.path(), inst);
        let td = TraceDir::open(tmp.path()).unwrap();
        let loc = td.kernels().next().unwrap();
        let sched = FunctionSchedule {
            arch: ArchId::Sm86,
            function_unique_id: 1,
            kernel_name: "k".into(),
            function_name: "f".into(),
            by_pc: StdMap::new(),
        };
        let err = extract_per_sm_blocks(&td, &loc, &sched, true).unwrap_err();
        assert!(
            matches!(
                err,
                ReplayError::Trace(TraceError::AllocationIdMissing { .. })
            ),
            "expected AllocationIdMissing, got {err:?}"
        );
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
            data_width: Some(4),
            memory_op_class: None,
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
    const PYTHON_SEALED_SUBSET: &str = r#"{"schema_version":2,"cell_id":"c","kernel":{"mangled_name":"_Z1kv","args_hash":"ab"},"method":"wave_segment_stratified_subset","reason":"byte_budget","shape":{"original_grid":{"x":60,"y":1,"z":1},"original_block":{"x":32,"y":1,"z":1},"captured_grid":{"x":60,"y":1,"z":1},"captured_block":{"x":32,"y":1,"z":1},"reduced_shape_variant":null},"gates":{"coop":false,"wave_count":60,"n_measured_waves":40,"tail_measured_waves":2,"atomic_density":0.0,"hotline_share":0.0,"skew_top1pct_share":0.0,"gini":0.0,"per_sm_block_count_cv":0.0,"n_phases":1,"skew_heavy_block_count":1,"certainty_bytes":100,"max_trace_bytes":2000,"certainty_budget_exceeded":false,"verdict":"subset_admissible"},"wave_model":{"reconstruction":"launch_index","per_sm_resident_blocks":[{"sm_id":0,"resident_blocks":1}],"n_waves":60,"segment_len":3,"ghost_depth":1},"per_sm_slot_map":[{"sm_id":0,"block_ids":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59]}],"rng":{"algo":"pcg64","seed":0},"tracer":{"version":5,"features":{"atomic_addr":true,"ldgsts_addr":true},"stats_pass_version":2},"bytes":{"raw":334,"compressed":200,"codec":"zstd","level":19,"long":27,"dict_id":null},"records":{"expected_full":6000,"captured":12},"hashes":{"selected_block_ids_sha256":"84b7ed43903cca6302ed15cf78ff507c45e575f16d7d5c12c975c5727edc37cf","decompressed_payload_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","sidecar_sha256":"331103782174538149454c35590a56994e29ffe65aa4614378df607cf993b4d9"},"label_pairing":"full_shape_label","segments":[{"id":0,"block_ids":[0,1,2]},{"id":19,"block_ids":[57,58,59]},{"id":1,"block_ids":[3,4,5]},{"id":2,"block_ids":[6,7,8]}],"selected_segment_ids":[0,19,1,2],"certainty_segments":[0,19],"force_include_blocks":[0],"strata":[{"id":0,"phase_id":0,"launch_tercile":0,"segment_ids":[1,2],"N_h":6,"n_h":2,"pi":0.3333333333333333,"sector_stddev":2.0,"neyman_weight":12.0}],"selection_reconciliation":{"status":"fixed_point","attempts":1,"max_attempts":3,"capture_stats_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},"selection_budget":{"max_trace_bytes":2000,"basis":"raw_threadblock_pb","certainty_fraction_numerator":2,"certainty_fraction_denominator":5,"certainty_bytes":100},"selection_auxiliary":{"name":"l2_sector_transactions_32b","sector_bytes":32,"scope":"pass2_all_blocks","expected_full":400,"selected":200},"estimator":{"mode":"weighted_full_shape_estimate","targets":["sectors"],"not_recoverable":["cycles"],"convergence":{"status":"converged"}}}"#;

    const PYTHON_SEALED_LOSSLESS: &str = r#"{"schema_version":2,"cell_id":"c","kernel":{"mangled_name":"_Z1kv","args_hash":"deadbeef"},"method":"lossless_compressed_full","reason":"byte_budget","shape":{"original_grid":{"x":60,"y":1,"z":1},"original_block":{"x":32,"y":1,"z":1},"captured_grid":{"x":60,"y":1,"z":1},"captured_block":{"x":32,"y":1,"z":1},"reduced_shape_variant":null},"gates":{"coop":false,"wave_count":60,"n_measured_waves":40,"tail_measured_waves":2,"atomic_density":0.0,"hotline_share":0.0,"skew_top1pct_share":0.0,"gini":0.0,"per_sm_block_count_cv":0.0,"n_phases":1,"skew_heavy_block_count":1,"certainty_bytes":null,"max_trace_bytes":null,"certainty_budget_exceeded":false,"verdict":"subset_admissible"},"wave_model":{"reconstruction":"launch_index","per_sm_resident_blocks":[{"sm_id":0,"resident_blocks":1}],"n_waves":60,"segment_len":3,"ghost_depth":1},"per_sm_slot_map":[{"sm_id":0,"block_ids":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59]}],"rng":{"algo":"pcg64","seed":0},"tracer":{"version":5,"features":{"atomic_addr":true,"ldgsts_addr":true},"stats_pass_version":1},"bytes":{"raw":1000,"compressed":200,"codec":"zstd","level":19,"long":27,"dict_id":null},"records":{"expected_full":6000,"captured":60},"hashes":{"selected_block_ids_sha256":"3f3ffcf42accae1591be118baf57efb9b8d7c78028dc566674c809f210ad796a","decompressed_payload_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","sidecar_sha256":"657fd9dc2f3f0a826dfb4bb030028ea32d315af1d37b54b4ff8bc9891f54aac8"},"label_pairing":"full_shape_label"}"#;

    const SELECTED_BLOCKS: [i32; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 57, 58, 59];

    fn payload_bound_subset() -> String {
        PYTHON_SEALED_SUBSET
            .replace(
                &"a".repeat(64),
                "f63b3f9ad203aaf13dd1bb4a15c89a89150d6b454956b730d7fa5f19ce9c2017",
            )
            .replace(
                "331103782174538149454c35590a56994e29ffe65aa4614378df607cf993b4d9",
                "fce059fb41f8aa35ca5eccfd52fca3ebde04523a45b825998c5d077613584ae9",
            )
    }

    fn subset_with_captured_mismatch() -> String {
        payload_bound_subset()
            .replace(r#""captured":12"#, r#""captured":13"#)
            .replace(
                "fce059fb41f8aa35ca5eccfd52fca3ebde04523a45b825998c5d077613584ae9",
                "d7d309d03ffea9ab4331da42d75419124f082182d29e6a0a13029bb402dec91e",
            )
    }

    fn payload_bound_subset_for_different_blocks() -> String {
        payload_bound_subset()
            .replace(
                "f63b3f9ad203aaf13dd1bb4a15c89a89150d6b454956b730d7fa5f19ce9c2017",
                "9fac314545a5d68513cb0b65765fa27df886df2f477cda2ff8eceb9ef67364e8",
            )
            .replace(
                "fce059fb41f8aa35ca5eccfd52fca3ebde04523a45b825998c5d077613584ae9",
                "bfe2f708fb586c7b5b303027172f91495ab8e23827ba471ddc9f675222aa18a1",
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
