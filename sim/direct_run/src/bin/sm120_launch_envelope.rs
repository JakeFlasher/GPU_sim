//! Structural SM120 launch-envelope inspector.
//!
//! This deliberately does not construct `IssueModelParams`, read a control
//! sidecar, or run native timing. It validates only architecture-card launch
//! limits against a real compact NVBit trace and emits placement evidence.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::path::PathBuf;

use serde::Serialize;
use sim_arch::card::{ArchCard, CalibError};
use sim_trace::{TraceDir, TraceError, kernel_grid};

#[derive(Debug, thiserror::Error)]
enum Error {
    #[error("usage: sm120-launch-envelope --trace DIR --arch-card FILE")]
    Usage,
    #[error("missing option value for {0}")]
    MissingValue(&'static str),
    #[error("unknown option {0}")]
    Unknown(String),
    #[error("architecture card: {0}")]
    Card(#[from] CalibError),
    #[error("trace: {0}")]
    Trace(#[from] TraceError),
    #[error("trace has no execution metadata")]
    MissingExecInfo,
    #[error("trace architecture {found} does not match card {expected}")]
    ArchitectureMismatch { found: u32, expected: u32 },
    #[error("kernel {kernel_id} has invalid block dimension {block:?}")]
    InvalidBlock {
        kernel_id: i32,
        block: (i32, i32, i32),
    },
    #[error("kernel {kernel_id} has negative shared-memory demand {bytes}")]
    NegativeShared { kernel_id: i32, bytes: i32 },
    #[error(
        "kernel {kernel_id} CTA has {threads} threads, exceeding runtime max_threads_per_block={limit}"
    )]
    ThreadsPerBlock {
        kernel_id: i32,
        threads: u64,
        limit: u32,
    },
    #[error(
        "kernel {kernel_id} CTA has {bytes} shared bytes, exceeding runtime opt-in block limit={limit}"
    )]
    SharedPerBlock {
        kernel_id: i32,
        bytes: u64,
        limit: u64,
    },
    #[error(
        "kernel {kernel_id} CTA has {bytes} shared bytes, exceeding runtime per-SM limit={limit}"
    )]
    SharedPerSm {
        kernel_id: i32,
        bytes: u64,
        limit: u64,
    },
    #[error(
        "kernel {kernel_id} block {block} has traced sm_id {sm_id} outside identifier bound {bound}"
    )]
    SmId {
        kernel_id: i32,
        block: i64,
        sm_id: i32,
        bound: u32,
    },
    #[error(
        "kernel {kernel_id} block {block} has traced warp slot {slot} outside identifier bound {bound}"
    )]
    WarpSlot {
        kernel_id: i32,
        block: i64,
        slot: i32,
        bound: u32,
    },
    #[error("kernel {kernel_id} block {block} changes sm_id from {first} to {second}")]
    BlockSmMismatch {
        kernel_id: i32,
        block: i64,
        first: i32,
        second: i32,
    },
    #[error("kernel {kernel_id} block {block} has no traced sm_id")]
    MissingBlockSm { kernel_id: i32, block: i64 },
    #[error("kernel {kernel_id} block {block} warp {warp} has no traced warpid_sm")]
    MissingWarpSlot {
        kernel_id: i32,
        block: i64,
        warp: i32,
    },
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    status: &'static str,
    note: &'static str,
    arch: String,
    sku: String,
    sm_count: u32,
    sm_id_upper_bound: u32,
    max_resident_warps_per_sm: u32,
    max_blocks_per_sm: u32,
    max_threads_per_sm: u32,
    max_threads_per_block: u32,
    max_shared_memory_per_sm_bytes: u64,
    max_shared_memory_per_block_bytes: u64,
    max_shared_memory_per_block_optin_bytes: u64,
    launches: Vec<LaunchReport>,
}

#[derive(Debug, Serialize)]
struct LaunchReport {
    device_id: i64,
    stream_id: i64,
    kernel_id: i32,
    function_unique_id: i32,
    kernel_name: String,
    grid: [u32; 3],
    block: [u32; 3],
    grid_blocks: u64,
    blocks_observed: u64,
    full_block_coverage: bool,
    shared_memory_bytes_per_block: u64,
    threads_per_block: u64,
    distinct_sm_ids: u64,
    min_sm_id: Option<i32>,
    max_sm_id: Option<i32>,
    distinct_sm_ids_are_dense: bool,
    distinct_warp_slots: u64,
    max_warp_slot: Option<i32>,
    max_warps_in_one_cta: u64,
    blocks_by_sm_observed: BTreeMap<i32, u64>,
    blocks_by_sm_is_residency: bool,
    records_seen: u64,
    records_missing_sm_id: u64,
    records_missing_warp_slot: u64,
}

fn parse_args() -> Result<(PathBuf, PathBuf), Error> {
    let mut trace = None;
    let mut card = None;
    let mut args = env::args_os().skip(1);
    while let Some(arg) = args.next() {
        let arg = arg.to_string_lossy();
        match arg.as_ref() {
            "--trace" => {
                trace = Some(
                    args.next()
                        .map(PathBuf::from)
                        .ok_or(Error::MissingValue("--trace"))?,
                )
            }
            "--arch-card" => {
                card = Some(
                    args.next()
                        .map(PathBuf::from)
                        .ok_or(Error::MissingValue("--arch-card"))?,
                )
            }
            "--help" | "-h" => return Err(Error::Usage),
            other => return Err(Error::Unknown(other.to_owned())),
        }
    }
    trace.zip(card).ok_or(Error::Usage)
}

fn product3(d: [u32; 3]) -> u64 {
    u64::from(d[0]) * u64::from(d[1]) * u64::from(d[2])
}

fn run(trace_root: PathBuf, card_path: PathBuf) -> Result<Report, Error> {
    let card = ArchCard::load(&card_path)?;
    let arch = card.arch_id()?;
    let sku = card.sku_id()?;
    let sm_count = card.sm_count()?;
    let limits = card.require_runtime_limits()?;
    let trace = TraceDir::open_direct_compact(trace_root)?;
    let exec = trace.exec_info().ok_or(Error::MissingExecInfo)?;
    for kernel in exec.kernels.values() {
        if kernel.architecture_version != arch.numeric() {
            return Err(Error::ArchitectureMismatch {
                found: kernel.architecture_version,
                expected: arch.numeric(),
            });
        }
    }
    let mut launches = Vec::new();
    for loc in trace.kernels() {
        let kernel = loc.kernel();
        let grid_tuple = kernel_grid(kernel)?;
        let grid = [grid_tuple.0, grid_tuple.1, grid_tuple.2];
        let raw_block =
            kernel
                .block_dim
                .as_ref()
                .map(|d| (d.x, d.y, d.z))
                .ok_or(Error::InvalidBlock {
                    kernel_id: kernel.id,
                    block: (0, 0, 0),
                })?;
        if raw_block.0 <= 0 || raw_block.1 <= 0 || raw_block.2 <= 0 {
            return Err(Error::InvalidBlock {
                kernel_id: kernel.id,
                block: raw_block,
            });
        }
        let block = [raw_block.0 as u32, raw_block.1 as u32, raw_block.2 as u32];
        let threads = product3(block);
        if threads > u64::from(limits.max_threads_per_block) {
            return Err(Error::ThreadsPerBlock {
                kernel_id: kernel.id,
                threads,
                limit: limits.max_threads_per_block,
            });
        }
        if kernel.size_shared_memory < 0 {
            return Err(Error::NegativeShared {
                kernel_id: kernel.id,
                bytes: kernel.size_shared_memory,
            });
        }
        let shared = kernel.size_shared_memory as u64;
        if shared > limits.max_shared_memory_per_block_optin_bytes {
            return Err(Error::SharedPerBlock {
                kernel_id: kernel.id,
                bytes: shared,
                limit: limits.max_shared_memory_per_block_optin_bytes,
            });
        }
        if shared > limits.max_shared_memory_per_sm_bytes {
            return Err(Error::SharedPerSm {
                kernel_id: kernel.id,
                bytes: shared,
                limit: limits.max_shared_memory_per_sm_bytes,
            });
        }
        let blocks = trace.validated_threadblocks_in_launch_order(&loc)?;
        let mut sm_ids = BTreeSet::new();
        let mut slots = BTreeSet::new();
        let mut blocks_by_sm = BTreeMap::<i32, u64>::new();
        let mut max_warps = 0_u64;
        let mut records = 0_u64;
        let mut missing_sm = 0_u64;
        let mut missing_slot = 0_u64;
        for (launch_index, block) in &blocks {
            max_warps = max_warps.max(block.warps().len() as u64);
            let mut block_sm = None;
            for warp in block.warps() {
                let mut warp_slot = None;
                for instruction in warp.instructions() {
                    records += 1;
                    match instruction.sm_id() {
                        Some(sm) => {
                            if sm < 0 || sm as u32 >= limits.sm_id_upper_bound {
                                return Err(Error::SmId {
                                    kernel_id: kernel.id,
                                    block: *launch_index,
                                    sm_id: sm,
                                    bound: limits.sm_id_upper_bound,
                                });
                            }
                            match block_sm {
                                None => block_sm = Some(sm),
                                Some(first) if first != sm => {
                                    return Err(Error::BlockSmMismatch {
                                        kernel_id: kernel.id,
                                        block: *launch_index,
                                        first,
                                        second: sm,
                                    });
                                }
                                Some(_) => {}
                            }
                            sm_ids.insert(sm);
                        }
                        None => missing_sm += 1,
                    }
                    match instruction.warpid_sm() {
                        Some(slot) => {
                            if slot < 0 || slot as u32 >= limits.warp_id_upper_bound {
                                return Err(Error::WarpSlot {
                                    kernel_id: kernel.id,
                                    block: *launch_index,
                                    slot,
                                    bound: limits.warp_id_upper_bound,
                                });
                            }
                            warp_slot = Some(slot);
                            slots.insert(slot);
                        }
                        None => missing_slot += 1,
                    }
                }
                if warp_slot.is_none() {
                    return Err(Error::MissingWarpSlot {
                        kernel_id: kernel.id,
                        block: *launch_index,
                        warp: warp.warp_id(),
                    });
                }
            }
            let Some(sm) = block_sm else {
                return Err(Error::MissingBlockSm {
                    kernel_id: kernel.id,
                    block: *launch_index,
                });
            };
            *blocks_by_sm.entry(sm).or_default() += 1;
        }
        let dense = match (sm_ids.iter().next(), sm_ids.iter().next_back()) {
            (Some(first), Some(last)) => sm_ids.iter().copied().eq(*first..=*last),
            _ => true,
        };
        launches.push(LaunchReport {
            device_id: loc.device_id(),
            stream_id: loc.stream_id(),
            kernel_id: kernel.id,
            function_unique_id: kernel.function_unique_id,
            kernel_name: kernel.name.clone(),
            grid,
            block,
            grid_blocks: product3(grid),
            blocks_observed: blocks.len() as u64,
            full_block_coverage: blocks.len() as u64 == product3(grid),
            shared_memory_bytes_per_block: shared,
            threads_per_block: threads,
            distinct_sm_ids: sm_ids.len() as u64,
            min_sm_id: sm_ids.iter().next().copied(),
            max_sm_id: sm_ids.iter().next_back().copied(),
            distinct_sm_ids_are_dense: dense,
            distinct_warp_slots: slots.len() as u64,
            max_warp_slot: slots.iter().next_back().copied(),
            max_warps_in_one_cta: max_warps,
            blocks_by_sm_observed: blocks_by_sm,
            blocks_by_sm_is_residency: false,
            records_seen: records,
            records_missing_sm_id: missing_sm,
            records_missing_warp_slot: missing_slot,
        });
    }
    Ok(Report {
        schema_version: 1,
        status: "placement_envelope_only",
        note: "runtime launch limits and traced identifiers only; no SM120 issue, cache, completion, or timing model",
        arch: arch.token().to_owned(),
        sku: sku.token().to_owned(),
        sm_count,
        sm_id_upper_bound: limits.sm_id_upper_bound,
        max_resident_warps_per_sm: limits.max_resident_warps_per_sm,
        max_blocks_per_sm: limits.max_blocks_per_sm,
        max_threads_per_sm: limits.max_threads_per_sm,
        max_threads_per_block: limits.max_threads_per_block,
        max_shared_memory_per_sm_bytes: limits.max_shared_memory_per_sm_bytes,
        max_shared_memory_per_block_bytes: limits.max_shared_memory_per_block_bytes,
        max_shared_memory_per_block_optin_bytes: limits.max_shared_memory_per_block_optin_bytes,
        launches,
    })
}

fn main() -> std::process::ExitCode {
    match parse_args().and_then(|(trace, card)| run(trace, card)) {
        Ok(report) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&report).expect("report JSON")
            );
            std::process::ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::from(2)
        }
    }
}
