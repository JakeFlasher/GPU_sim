//! SMSP issue model (design §6.2): per-warp program replay where issue
//! timing is driven by the compiler's own dependence encoding — the SASS
//! control bits (stall counts, dependence-barrier set/wait, yield) — not by
//! simulated scoreboards. This is the 13–17% cycles-MAPE lever the UPC
//! MICRO'25 modern-core work identified [F6].
//!
//! Model shape:
//! - 4 sub-cores (SMSPs) per SM; a warp lives on sub-core
//!   `warp_slot % 4` (the physical `warpid_sm` slot when traced).
//! - CGGTY-style pick per sub-core per cycle: keep issuing from the held
//!   warp while it stays ready (greedy; a set yield bit drops the hold),
//!   otherwise switch to the YOUNGEST ready warp (largest arrival stamp).
//! - Each issued instruction makes its warp eligible again at
//!   `t + max(stall, 1)`; a set write/read dependence barrier is stamped
//!   with a clear time; an instruction whose control word carries a wait
//!   mask cannot issue until every masked barrier has cleared.
//! - `BAR.*` (classified by the sidecar's opcode text) is a block-wide
//!   barrier: generation semantics — it releases when every unfinished
//!   warp of the block is waiting at it (warps that already finished do
//!   not block release, matching Volta+ exited-thread semantics).
//! - Block residency: blocks are admitted to the SM in launch order while
//!   the resident warp count stays within `max_resident_warps_per_sm`;
//!   a finished block frees its slots (wave execution).
//!
//! Memory timing comes in three modes ([`SmMemMode`]):
//! - **Flat**: no cache model; global-class barriers clear at the flat
//!   cold-DRAM latency (the pre-L1 v0 behavior).
//! - **L1Const**: a per-SM sectored L1 probed AT ISSUE TIME with fill
//!   times known at issue (constant L2-hit latency below the L1) — the
//!   sim-mem L1 v0 behavior, kept as the lockstep-differential surface.
//! - **L1Shared**: the request/response INVERSION for the shared L2
//!   (design §6.3): an L1 miss's fill time is NOT knowable at issue —
//!   the shared L2's state when the request arrives decides L2-hit vs
//!   DRAM pricing. Misses emit [`MemRequest`]s; the warp's dependence
//!   barrier tracks PENDING response counts alongside the known clear
//!   stamp (hardware counter semantics extended to unknown write-backs),
//!   and [`SmCore::on_response`] completes fills / clears pending units
//!   at response-event time. MSHR-full accesses park in a retry queue
//!   drained on the next completion (the event-mode analogue of the
//!   constant mode's retry-at-earliest-fill; the warp itself keeps
//!   issuing until a wait mask stops it, like the hardware's LSU replay
//!   path).
//!
//! [`SmCore`] is a resumable actor: `advance(horizon)` runs issue cycles
//! strictly BEFORE `horizon` (the event system passes the next queued
//! event's time so no cross-SM effect is overrun; a self-emitted request
//! additionally caps the run-ahead at the earliest time its response
//! could arrive). [`simulate_sm`] is the synchronous wrapper for the
//! Flat/L1Const modes (horizon = ∞, no requests possible).
//!
//! The instruction COUNT produced by replay is placement- and
//! timing-invariant: every traced record issues exactly once, so
//! `instructions_issued` is exact regardless of the cycle model's fidelity.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};

use sim_arch::ArchId;
use sim_mem::{
    CacheOpToken, EvAccess, FillKey, L1Cache, L1Config, L1Counters, L2OpClass, L2Source,
};

use crate::program::{FunctionSchedule, IssueSlot, OpClass};

pub use sim_mem::AccessKind;

/// One dynamic instruction, compacted for replay.
#[derive(Debug, Clone, Copy)]
pub struct ReplayInst {
    /// Function-relative pc — the join key into [`FunctionSchedule`].
    pub pc: u32,
    pub kind: AccessKind,
}

/// One traced warp of a block: physical slot + its dynamic stream + the
/// sector arena its records' `off/len` payloads index.
#[derive(Debug, Default)]
pub struct WarpReplay {
    /// Physical warp slot on the SM (`warpid_sm` when traced); sub-core
    /// assignment is `warp_slot % 4`.
    pub warp_slot: i32,
    pub insts: Vec<ReplayInst>,
    pub sectors: Vec<i64>,
}

/// One traced block assigned to this SM, in launch order.
#[derive(Debug)]
pub struct BlockReplay {
    pub launch_index: i64,
    pub warps: Vec<WarpReplay>,
}

/// Timing parameters. Card-derived values come from the pchase static
/// card / sim-calib (the caller maps the fields — sim-sm deliberately
/// depends on neither); the rest are documented hypothesis knobs pending
/// Phase-1D calibration extensions.
#[derive(Debug, Clone, Copy)]
pub struct IssueModelParams {
    /// Deployment architecture whose card and issue hypotheses produced these parameters.
    pub arch: ArchId,
    /// L1 hit data-ready latency (card `l1_hit_latency_cycles`, ceiled).
    pub l1_hit_latency: u64,
    /// End-to-end L1-miss/L2-HIT data-ready latency (card
    /// `l2_hit_latency_cycles`, ceiled). In L1Const mode every miss is
    /// priced at this (the L2-hit assumption); in L1Shared mode it is
    /// the L2-hit arm of the ladder AND the lower bound on any response.
    pub l1_miss_latency: u64,
    /// End-to-end L1-miss/L2-MISS data-ready latency (card
    /// `dram_latency_cycles`, ceiled): the DRAM arm of the ladder in
    /// L1Shared mode, and the flat global barrier latency in Flat mode.
    pub global_flat_latency: u64,
    /// Shared-class barrier latency (card `shared_latency_cycles`).
    pub shared_write_barrier_latency: u64,
    /// Write-barrier clear latency for non-memory producers (S2R, MUFU,
    /// I2F, HMMA & co). Hypothesis knob: ~20–25 cycles is the
    /// special-register/SFU class in Abdelkhalik et al. (HPEC'22, A100);
    /// not yet microbenched on this project's cards.
    pub non_memory_write_barrier_latency: u64,
    /// Read-dependence barrier clear latency (operand consumption).
    /// Hypothesis knob: source operands are read a few cycles after issue.
    pub read_barrier_latency: u64,
    /// Resident-warp ceiling per SM. The reviewed sm_86 hypothesis is 48;
    /// template architectures return a typed capability gap until their
    /// occupancy evidence is carded.
    pub max_resident_warps_per_sm: u32,
}

fn checked_latency(name: &'static str, value: f64) -> Result<u64, IssueError> {
    // `u64::MAX as f64` is rounded to 2^64.  Treat it as an exclusive
    // boundary: accepting equality would let the float→u64 cast saturate.
    if !value.is_finite() || value <= 0.0 || value >= u64::MAX as f64 {
        return Err(IssueError::InvalidLatency { name, value });
    }
    let rounded = value.ceil();
    if rounded <= 0.0 || rounded >= u64::MAX as f64 {
        return Err(IssueError::InvalidLatency { name, value });
    }
    Ok(rounded as u64)
}

impl IssueModelParams {
    /// Resolve card latencies plus reviewed architecture-local issue
    /// hypotheses.  A template architecture with missing scheduler facts
    /// returns a typed capability error instead of inheriting sm_86 knobs.
    pub fn from_card_latencies_for_arch(
        arch: ArchId,
        l1_hit_cycles: f64,
        l2_hit_cycles: f64,
        dram_cycles: f64,
        shared_cycles: f64,
    ) -> Result<Self, IssueError> {
        let hypotheses = arch.issue_hypotheses().require("issue_hypotheses")?;
        Ok(Self {
            arch,
            l1_hit_latency: checked_latency("l1_hit_cycles", l1_hit_cycles)?,
            l1_miss_latency: checked_latency("l2_hit_cycles", l2_hit_cycles)?,
            global_flat_latency: checked_latency("dram_cycles", dram_cycles)?,
            shared_write_barrier_latency: checked_latency("shared_cycles", shared_cycles)?,
            non_memory_write_barrier_latency: hypotheses.non_memory_write_barrier_latency,
            read_barrier_latency: hypotheses.read_barrier_latency,
            max_resident_warps_per_sm: hypotheses.max_resident_warps_per_sm,
        })
    }

    /// Non-memory producer visibility (one copy for BOTH memory modes).
    fn non_memory_ready(&self, t: u64) -> u64 {
        t + self.non_memory_write_barrier_latency
    }

    /// Shared-memory visibility (one copy for BOTH memory modes).
    fn shared_ready(&self, t: u64) -> u64 {
        t + self.shared_write_barrier_latency
    }

    /// Store / degenerate-record / L1-bypass visibility: the L2-hit arm.
    /// The single copy both data paths must price identically — the
    /// Const-vs-Shared ablation delta depends on it.
    fn store_visibility_ready(&self, t: u64) -> u64 {
        t + self.l1_miss_latency
    }

    /// L1 configuration sharing THIS param set's latencies (single
    /// source — the hit/miss numbers must never exist in two copies
    /// that can drift), with geometry from the pchase card resolvers.
    pub fn l1_config(&self, line_capacity: u64, mshr_count: u32) -> L1Config {
        L1Config {
            arch: self.arch,
            line_capacity,
            mshr_count: Some(mshr_count),
            hit_latency: self.l1_hit_latency,
            miss_latency: self.l1_miss_latency,
        }
    }
}

/// Memory-model mode for one SM replay.
#[derive(Debug, Clone, Copy)]
pub enum SmMemMode {
    /// No cache model: flat global latency.
    Flat,
    /// Per-SM L1 with constant (L2-hit) fill latency below it.
    L1Const(L1Config),
    /// Per-SM L1 over the SHARED L2: misses become requests/responses.
    L1Shared(L1Config),
}

#[derive(Debug, thiserror::Error)]
pub enum IssueError {
    #[error("issue model is unsupported for deployment architecture {arch}")]
    ArchitectureUnsupported { arch: ArchId },
    #[error(
        "pc {pc} (block {launch_index}, warp slot {warp_slot}) not in the issue schedule of \
         {function_name:?} — dynamic trace and static table disagree"
    )]
    PcNotInSchedule {
        pc: u32,
        launch_index: i64,
        warp_slot: i32,
        function_name: String,
    },
    #[error(
        "no schedulable warp, no pending timer and no outstanding response at t={time} with \
         {unfinished} warp(s) unfinished — issue-model invariant broken"
    )]
    NoProgress { time: u64, unfinished: usize },
    #[error("issue architecture mismatch for {component}: expected {expected}, got {got}")]
    ArchitectureMismatch {
        component: &'static str,
        expected: ArchId,
        got: ArchId,
    },
    #[error("{name} latency must be finite, positive, and fit in u64 (got {value})")]
    InvalidLatency { name: &'static str, value: f64 },

    #[error("issue capability unavailable: {0}")]
    Capability(#[from] sim_arch::CapabilityError),
    #[error("L1 configuration is invalid: {0}")]
    L1Config(#[from] sim_mem::L1ConfigError),
}

/// Result of replaying one SM's blocks to completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmSimResult {
    /// Exact: one per traced record fed in.
    pub instructions_issued: u64,
    /// Directional v0 cycle count (time of last issue + 1; the response
    /// tail of the final loads is not counted — write-back completion is
    /// not an issue event).
    pub cycles: u64,
    /// This SM's L1 counters when an L1 model was attached.
    pub l1: Option<L1Counters>,
}

/// A memory request the SM emits in L1Shared mode. The system layer
/// schedules the L2 access at `issue_time` (the engine's global
/// `(time, seq)` order supplies the cross-SM interleaving) and routes the
/// response per `reply`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemRequest {
    pub issue_time: u64,
    /// Global 32 B sector id (`byte_address >> 5`).
    pub sector_id: i64,
    /// Sectors of `sector_id`'s 128 B line carried by THIS request
    /// (bit = sector index in line; always contains `sector_id`'s own
    /// bit). More than one bit only on L1 LINE-fill requests whose
    /// emission batch merged additional sectors onto the fill — NCU
    /// counts each 32 B sector at the LTS, so a multi-sector line fill
    /// must reach the L2 as that many sector accesses, not one request
    /// (§6.4 slice 2; witness accounting lives in the progress log,
    /// never in sim sources — Rule 1). The L2 handler walks the mask
    /// in ascending sector order and prices the reply on the WORST
    /// sector outcome (the fill completes when its slowest sector
    /// arrives).
    pub sector_mask: u8,
    pub source: L2Source,
    /// Behavioral L2 hit rule for this sector (§6.4): stores/atomics
    /// tag-match, loads need the sector — set from the record's token
    /// at emission, orthogonal to the provenance-only `source`.
    pub class: L2OpClass,
    pub reply: ReplyTo,
}

/// What the response to a [`MemRequest`] must do at arrival.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplyTo {
    /// Complete this L1 fill (installing the line/sector) and clear the
    /// fill's registered barrier waiters.
    L1Fill(FillKey),
    /// Clear one pending unit on (warp, barrier) directly — bypass and
    /// untagged loads that never touch the modeled L1.
    Warp { warp: u32, barrier: u8 },
    /// Mutation-only access (stores): no response event needed.
    None,
}

impl MemRequest {
    /// An L1 fill request (line alloc or resident-sector fetch): Read
    /// class by construction — nothing that fills the modeled L1 is a
    /// write (stores hit-test without allocating, atomics bypass). The
    /// typed constructors keep the (source, class, reply) coupling in
    /// one place so an invalid combination is unwritable at emission
    /// sites (gate finding 2026-07-07).
    fn l1_fill(issue_time: u64, sector_id: i64, fill: FillKey) -> Self {
        Self {
            issue_time,
            sector_id,
            sector_mask: 1 << (sector_id & 3) as u8,
            source: L2Source::L1Fill,
            class: L2OpClass::Read,
            reply: ReplyTo::L1Fill(fill),
        }
    }

    /// A follow-up sector fetch for a merge onto an older in-flight
    /// fill: same provenance and class as the fill, no reply (the
    /// waiter rides the original fill's response).
    fn l1_followup_fetch(issue_time: u64, sector_id: i64) -> Self {
        Self {
            issue_time,
            sector_id,
            sector_mask: 1 << (sector_id & 3) as u8,
            source: L2Source::L1Fill,
            class: L2OpClass::Read,
            reply: ReplyTo::None,
        }
    }

    /// A global store sector (STG/RED): write-class tag-match rule at
    /// the L2, mutation-only (no response).
    fn store(issue_time: u64, sector_id: i64) -> Self {
        Self {
            issue_time,
            sector_id,
            sector_mask: 1 << (sector_id & 3) as u8,
            source: L2Source::Store,
            class: L2OpClass::WriteAtomic,
            reply: ReplyTo::None,
        }
    }
}

/// Outcome of one [`SmCore::advance`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Advance {
    /// The SM cannot act strictly before `wake_at` (if `Some`); with
    /// `None`, only a memory response can unblock it.
    Idle { wake_at: Option<u64> },
    /// All warps of all blocks retired.
    Done,
}

const SUB_CORES: usize = 4;
const DEP_BARRIERS: usize = 6;

struct WarpState<'a> {
    insts: &'a [ReplayInst],
    sectors: &'a [i64],
    next: usize,
    next_slot: Option<&'a IssueSlot>,
    eligible: u64,
    barrier_clear: [u64; DEP_BARRIERS],
    /// Outstanding UNKNOWN write-backs per barrier index (L1Shared mode):
    /// hardware barriers are counters, and a response still in flight is
    /// an increment whose decrement time is unknown — a wait on the index
    /// cannot pass, and the warp has no computable ready time, until
    /// every pending unit resolves.
    barrier_pending: [u32; DEP_BARRIERS],
    block: usize,
    launch_index: i64,
    warp_slot: i32,
    /// Index into its sub-core's pick list while live (retired warps are
    /// swap-removed from the list; the swapped-in warp's position is
    /// fixed up). A warp is only ever picked from its own sub-core, so
    /// the retiring sub-core index is the loop's `sc`.
    pos_in_subcore: usize,
    arrival: u64,
    at_block_barrier: bool,
    done: bool,
}

struct BlockState {
    unfinished: usize,
    waiting: usize,
    warp_count: u32,
    /// This block's warps occupy `warps[first_warp..first_warp +
    /// admitted_warps]` (admission pushes them contiguously), so barrier
    /// release scans only the block's own range.
    first_warp: usize,
    admitted_warps: usize,
}

impl WarpState<'_> {
    /// Producer bookkeeping: one pending (unknown write-back) unit on a
    /// valid barrier index; no-op on the invalid sentinel.
    fn add_pending(&mut self, barrier: u8) {
        if usize::from(barrier) < DEP_BARRIERS {
            self.barrier_pending[usize::from(barrier)] += 1;
        }
    }

    /// One pending unit resolved at `now`: decrement AND max-accumulate
    /// the clear stamp (hardware counter semantics — the single copy of
    /// the pairing every response path must preserve).
    fn clear_pending(&mut self, barrier: u8, now: u64) {
        let b = usize::from(barrier);
        self.barrier_pending[b] = self.barrier_pending[b]
            .checked_sub(1)
            .expect("barrier pending-unit underflow");
        self.barrier_clear[b] = self.barrier_clear[b].max(now);
    }

    /// Release a pending unit WITHOUT a stamp (a retry entry's own unit;
    /// its per-sector waiters carry the stamps). No-op on the sentinel.
    fn release_pending(&mut self, barrier: u8) {
        if usize::from(barrier) < DEP_BARRIERS {
            self.barrier_pending[usize::from(barrier)] = self.barrier_pending[usize::from(barrier)]
                .checked_sub(1)
                .expect("retry entry without a pending unit");
        }
    }

    /// Max-accumulate a KNOWN clear stamp on a valid barrier index
    /// (counter semantics: never overwrite); no-op on the sentinel.
    fn stamp_clear(&mut self, barrier: u8, at: u64) {
        if usize::from(barrier) < DEP_BARRIERS {
            let clear = &mut self.barrier_clear[usize::from(barrier)];
            *clear = (*clear).max(at);
        }
    }

    /// Earliest time the pending wait mask clears; `None` while any
    /// masked barrier has responses outstanding (unknown clear time).
    fn wait_clear_time(&self) -> Option<u64> {
        let Some(slot) = self.next_slot else {
            return Some(0);
        };
        let mut t = 0;
        for b in 0..DEP_BARRIERS {
            if slot.control.wait_mask & (1 << b) != 0 {
                if self.barrier_pending[b] > 0 {
                    return None;
                }
                t = t.max(self.barrier_clear[b]);
            }
        }
        Some(t)
    }

    fn ready_at(&self) -> Option<u64> {
        if self.done || self.at_block_barrier {
            return None;
        }
        self.wait_clear_time().map(|w| self.eligible.max(w))
    }

    fn is_ready(&self, t: u64) -> bool {
        self.ready_at().is_some_and(|r| r <= t)
    }
}

/// Data-ready time for one issued record at time `t` in the Flat/L1Const
/// modes (fill times known at issue), driving the L1 state/counters when
/// attached.
fn data_ready_time(
    kind: AccessKind,
    sectors: &[i64],
    t: u64,
    params: &IssueModelParams,
    l1: Option<&mut L1Cache>,
) -> u64 {
    match kind {
        AccessKind::NonMemory => params.non_memory_ready(t),
        AccessKind::Shared => params.shared_ready(t),
        AccessKind::GlobalEligible { off, len, .. } => match l1 {
            Some(l1) => {
                let mut ready = t + params.l1_hit_latency;
                for &sector in &sectors[off as usize..off as usize + usize::from(len)] {
                    let (_, r) = l1.access_eligible(sector, t);
                    ready = ready.max(r);
                }
                ready
            }
            None => t + params.global_flat_latency,
        },
        AccessKind::GlobalBypass { len, token, .. } => match l1 {
            Some(l1) => {
                l1.count_bypass(u64::from(len), token == CacheOpToken::Atom);
                params.store_visibility_ready(t)
            }
            None => t + params.global_flat_latency,
        },
        AccessKind::GlobalUntagged { len, .. } => match l1 {
            Some(l1) => {
                l1.count_untagged(u64::from(len));
                params.store_visibility_ready(t)
            }
            None => t + params.global_flat_latency,
        },
        AccessKind::GlobalStore { off, len } => match l1 {
            Some(l1) => {
                for &sector in &sectors[off as usize..off as usize + usize::from(len)] {
                    l1.probe_store(sector, t);
                }
                params.store_visibility_ready(t)
            }
            None => t + params.global_flat_latency,
        },
    }
}

/// An MSHR-stalled remainder of one instruction's sector list, retried
/// (in FIFO order) whenever a fill completion frees an entry.
struct Retry {
    warp: usize,
    /// The stalled instruction's RAW write-barrier index (the control
    /// word's own invalid-sentinel domain, validated by the same
    /// `< DEP_BARRIERS` check as everywhere else — no second sentinel
    /// encoding). When valid, the entry holds ONE pending unit until
    /// fully drained.
    barrier: u8,
    /// Unresolved slice `warps[warp].sectors[lo..hi]`.
    lo: usize,
    hi: usize,
}

/// Barrier units cleared when one keyed L1 fill completes. Inline for
/// the dominant single-waiter case — a `Vec` per miss is pure allocator
/// churn on 100M-record cells.
enum Waiters {
    One(u32, u8),
    Many(Vec<(u32, u8)>),
}

/// State that EXISTS only in the shared-L2 mode — carrying it inside
/// the [`MemModel`] variant makes a mispaired construction
/// unrepresentable instead of an `expect()` deep in the memory step.
struct SharedL2State {
    l1: L1Cache,
    /// Responses still expected from the system.
    outstanding: u64,
    /// Barrier units cleared when the keyed L1 fill completes.
    fill_waiters: HashMap<FillKey, Waiters>,
    /// Per in-flight LINE fill: the sector bits already sent to the L2
    /// (the fill request's mask plus any follow-up fetches). A merge
    /// onto a sector NOT yet in the mask sends a follow-up sector
    /// fetch (`ReplyTo::None` — the waiter still rides the original
    /// fill's response, the documented merge timing); a re-touch of a
    /// sent sector sends nothing, like the hardware MSHR. Entries die
    /// with the fill response. Exactly-once is PER FILL, not
    /// per-address-epoch: if a resident line with an in-flight SECTOR
    /// fill is evicted and re-allocated, the new fill may re-fetch a
    /// sector whose zombie fetch is still outstanding (rare
    /// eviction-window corner, documented at the 2026-07-07 gate).
    fill_masks: HashMap<i64, u8>,
    /// Line fills emitted during the CURRENT emission batch (one
    /// `advance` or `on_response` call — the window whose requests are
    /// still unscheduled in the caller's `out`): line id → index in
    /// `out`. A merge onto one of these FOLDS its sector into the fill
    /// request's mask, so the sector participates in worst-of-mask
    /// pricing regardless of which warp/instruction/retry-entry of the
    /// batch touched it (the MSHR accumulates sectors until the SM
    /// yields — gate finding 2026-07-07: per-instruction folding made
    /// the priced arm depend on batch boundaries instead of physics).
    /// Cleared at every batch entry; indices are only valid because
    /// each batch starts with an empty `out`.
    line_folds: Vec<(i64, usize)>,
    retry: VecDeque<Retry>,
}

/// The typed memory model an [`SmCore`] runs under (the public
/// [`SmMemMode`] resolved into owned state).
enum MemModel {
    Flat,
    L1Const(L1Cache),
    L1Shared(SharedL2State),
}

impl MemModel {
    fn l1(&self) -> Option<&L1Cache> {
        match self {
            MemModel::Flat => None,
            MemModel::L1Const(l1) => Some(l1),
            MemModel::L1Shared(st) => Some(&st.l1),
        }
    }

    /// The L1 the CONSTANT-latency data path drives (`None` in Flat and
    /// Shared modes — the shared path never uses `data_ready_time`).
    fn const_l1_mut(&mut self) -> Option<&mut L1Cache> {
        match self {
            MemModel::L1Const(l1) => Some(l1),
            _ => None,
        }
    }
}

/// Resumable single-SM replay actor. See the module docs for the mode
/// semantics and the run-ahead (horizon) contract.
pub struct SmCore<'a> {
    schedule: &'a FunctionSchedule,
    params: IssueModelParams,
    mem: MemModel,
    blocks: &'a [BlockReplay],
    warps: Vec<WarpState<'a>>,
    block_states: Vec<BlockState>,
    subcore_warps: [Vec<usize>; SUB_CORES],
    hold: [Option<usize>; SUB_CORES],
    issued: u64,
    t: u64,
    arrival_stamp: u64,
    resident_warps: u32,
    unfinished_warps: usize,
    next_block: usize,
}

impl<'a> SmCore<'a> {
    /// Build the actor and admit the initial block wave. `blocks` must be
    /// this SM's blocks in launch order; every instruction's pc must
    /// resolve in `schedule` (single-function launches — the replay
    /// driver enforces that before calling). Cache modes attach a FRESH
    /// per-SM L1 (cold cache per kernel, NCU `--cache-control all`
    /// parity).
    pub fn new(
        blocks: &'a [BlockReplay],
        schedule: &'a FunctionSchedule,
        params: &IssueModelParams,
        mode: SmMemMode,
    ) -> Result<Self, IssueError> {
        if params.arch != ArchId::Sm86 {
            return Err(IssueError::ArchitectureUnsupported { arch: params.arch });
        }
        if params.arch != schedule.arch {
            return Err(IssueError::ArchitectureMismatch {
                component: "issue parameters",
                expected: schedule.arch,
                got: params.arch,
            });
        }
        let l1_arch = match mode {
            SmMemMode::Flat => None,
            SmMemMode::L1Const(cfg) | SmMemMode::L1Shared(cfg) => Some(cfg.arch),
        };
        if let Some(got) = l1_arch
            && got != schedule.arch
        {
            return Err(IssueError::ArchitectureMismatch {
                component: "L1 configuration",
                expected: schedule.arch,
                got,
            });
        }
        let mem = match mode {
            SmMemMode::Flat => MemModel::Flat,
            SmMemMode::L1Const(cfg) => MemModel::L1Const(L1Cache::new(cfg)?),
            SmMemMode::L1Shared(cfg) => MemModel::L1Shared(SharedL2State {
                l1: L1Cache::new(cfg)?,
                outstanding: 0,
                fill_waiters: HashMap::new(),
                fill_masks: HashMap::new(),
                line_folds: Vec::new(),
                retry: VecDeque::new(),
            }),
        };
        let mut core = Self {
            schedule,
            params: *params,
            mem,
            blocks,
            warps: Vec::new(),
            block_states: Vec::new(),
            subcore_warps: Default::default(),
            hold: [None; SUB_CORES],
            issued: 0,
            t: 0,
            arrival_stamp: 0,
            resident_warps: 0,
            unfinished_warps: 0,
            next_block: 0,
        };
        core.admit_blocks(0)?;
        Ok(core)
    }

    /// The SM's local clock (time of the next unissued cycle).
    pub fn now(&self) -> u64 {
        self.t
    }

    pub fn is_done(&self) -> bool {
        self.unfinished_warps == 0
    }

    /// The attached L1's counters (any cache mode).
    pub fn l1_counters(&self) -> Option<&L1Counters> {
        self.mem.l1().map(|c| &c.counters)
    }

    /// Responses the system still owes this SM (diagnostics).
    pub fn outstanding_responses(&self) -> u64 {
        match &self.mem {
            MemModel::L1Shared(st) => st.outstanding,
            _ => 0,
        }
    }

    /// Instructions issued so far (monotonic; the system driver's
    /// progress signal).
    pub fn instructions_issued(&self) -> u64 {
        self.issued
    }

    /// Result snapshot; meaningful once [`Self::advance`] returned
    /// [`Advance::Done`].
    pub fn result(&self) -> SmSimResult {
        SmSimResult {
            instructions_issued: self.issued,
            cycles: self.t,
            l1: self.mem.l1().map(|c| c.counters),
        }
    }

    /// Admit blocks in launch order while the warp ceiling allows. A block
    /// larger than the ceiling is admitted alone rather than deadlocking;
    /// a block whose warps are ALL empty occupies no residency (it can
    /// never retire through the issue path, so counting it would leak
    /// warp slots and block later admissions forever).
    fn admit_blocks(&mut self, t: u64) -> Result<(), IssueError> {
        while self.next_block < self.blocks.len() {
            let blk = &self.blocks[self.next_block];
            let live_warps = blk.warps.iter().filter(|w| !w.insts.is_empty()).count() as u32;
            if self.resident_warps > 0
                && self.resident_warps + live_warps > self.params.max_resident_warps_per_sm
            {
                break;
            }
            let block_ix = self.block_states.len();
            let first_warp = self.warps.len();
            let mut unfinished = 0usize;
            for w in &blk.warps {
                // Empty traced warps issue nothing: skip entirely
                // (no residency, no sub-core slot).
                if w.insts.is_empty() {
                    continue;
                }
                let subcore = (w.warp_slot.rem_euclid(SUB_CORES as i32)) as usize;
                let ws = WarpState {
                    insts: &w.insts,
                    sectors: &w.sectors,
                    next: 0,
                    next_slot: Some(lookup(
                        self.schedule,
                        w.insts[0].pc,
                        blk.launch_index,
                        w.warp_slot,
                    )?),
                    eligible: t,
                    barrier_clear: [0; DEP_BARRIERS],
                    barrier_pending: [0; DEP_BARRIERS],
                    block: block_ix,
                    launch_index: blk.launch_index,
                    warp_slot: w.warp_slot,
                    pos_in_subcore: self.subcore_warps[subcore].len(),
                    arrival: self.arrival_stamp,
                    at_block_barrier: false,
                    done: false,
                };
                self.arrival_stamp += 1;
                unfinished += 1;
                let ix = self.warps.len();
                self.subcore_warps[subcore].push(ix);
                self.warps.push(ws);
            }
            self.block_states.push(BlockState {
                unfinished,
                waiting: 0,
                warp_count: live_warps,
                first_warp,
                admitted_warps: unfinished,
            });
            self.resident_warps += live_warps;
            self.unfinished_warps += unfinished;
            self.next_block += 1;
        }
        Ok(())
    }

    // (Shared-mode memory helpers are free functions over the
    // destructured `SharedL2State` + warp slice so the borrow of
    // `self.mem` never has to span a `&mut self` call.)

    /// Memory interaction of one issued record in L1Shared mode. Returns
    /// the KNOWN data-ready component (`None` when the record's write
    /// visibility depends only on in-flight responses); unknown
    /// components are tracked as barrier-pending units.
    fn mem_step_shared(
        &mut self,
        wix: usize,
        kind: AccessKind,
        sectors: &'a [i64],
        write_barrier: u8,
        out: &mut Vec<MemRequest>,
    ) -> Option<u64> {
        let t = self.t;
        let p = self.params;
        let MemModel::L1Shared(st) = &mut self.mem else {
            unreachable!("mem_step_shared is only dispatched in shared mode")
        };
        let warps = &mut self.warps;
        match kind {
            AccessKind::NonMemory => Some(p.non_memory_ready(t)),
            AccessKind::Shared => Some(p.shared_ready(t)),
            AccessKind::GlobalEligible { off, len, .. } => {
                // The hit-pipeline floor applies to the whole load (as in
                // L1Const); misses ride their responses. Line fills
                // emitted anywhere in THIS emission batch (st.line_folds,
                // cleared at advance entry) fold later same-line sectors
                // into their request mask — one L2 transaction per line
                // per batch, priced on its worst sector.
                let mut known = t + p.l1_hit_latency;
                let range = off as usize..off as usize + usize::from(len);
                for i in range.clone() {
                    match access_shared(st, warps, wix, write_barrier, sectors[i], t, out) {
                        Ok(Some(ready)) => known = known.max(ready),
                        Ok(None) => {}
                        Err(()) => {
                            // MSHR full: park the remainder; the entry
                            // holds one pending unit until drained.
                            warps[wix].add_pending(write_barrier);
                            st.retry.push_back(Retry {
                                warp: wix,
                                barrier: write_barrier,
                                lo: i,
                                hi: range.end,
                            });
                            break;
                        }
                    }
                }
                Some(known)
            }
            AccessKind::GlobalBypass { off, len, .. } | AccessKind::GlobalUntagged { off, len } => {
                let (source, counted, class) = match kind {
                    // ATOM rides bypass provenance but is write-class at
                    // the L2 (§6.4 tag-match rule); every other bypass /
                    // untagged sector is a load needing the data.
                    AccessKind::GlobalBypass { token, .. } => (
                        L2Source::Bypass,
                        true,
                        if token == CacheOpToken::Atom {
                            L2OpClass::WriteAtomic
                        } else {
                            L2OpClass::Read
                        },
                    ),
                    _ => (L2Source::Untagged, false, L2OpClass::Read),
                };
                if counted {
                    st.l1
                        .count_bypass(u64::from(len), class == L2OpClass::WriteAtomic);
                } else {
                    st.l1.count_untagged(u64::from(len));
                }
                if len == 0 {
                    // Degenerate record (fully predicated-off lanes are
                    // legal on the wire): no sectors, no requests — but
                    // the write-barrier contract must match L1Const,
                    // which stamps these at the L2-hit arm.
                    return Some(p.store_visibility_ready(t));
                }
                let barrier_valid = usize::from(write_barrier) < DEP_BARRIERS;
                for &sector_id in &sectors[off as usize..off as usize + usize::from(len)] {
                    let reply = if barrier_valid {
                        warps[wix].add_pending(write_barrier);
                        st.outstanding += 1;
                        ReplyTo::Warp {
                            warp: wix as u32,
                            barrier: write_barrier,
                        }
                    } else {
                        ReplyTo::None
                    };
                    out.push(MemRequest {
                        issue_time: t,
                        sector_id,
                        sector_mask: 1 << (sector_id & 3) as u8,
                        source,
                        class,
                        reply,
                    });
                }
                None
            }
            AccessKind::GlobalStore { off, len } => {
                for &sector_id in &sectors[off as usize..off as usize + usize::from(len)] {
                    st.l1.probe_store(sector_id, t);
                    out.push(MemRequest::store(t, sector_id));
                }
                // v0: store visibility priced at the L2-hit arm (as in
                // L1Const); stores do not wait on responses.
                Some(p.store_visibility_ready(t))
            }
        }
    }

    /// Apply one memory response at its event time, then drain what it
    /// unblocked (fill waiters, MSHR retry queue — retries may emit new
    /// requests into `out`).
    pub fn on_response(&mut self, reply: ReplyTo, now: u64, out: &mut Vec<MemRequest>) {
        let MemModel::L1Shared(st) = &mut self.mem else {
            unreachable!("responses only exist in shared mode")
        };
        // New emission batch (the retry drain below may emit).
        st.line_folds.clear();
        let warps = &mut self.warps;
        match reply {
            ReplyTo::L1Fill(fill) => {
                st.outstanding = st
                    .outstanding
                    .checked_sub(1)
                    .expect("response without an outstanding request");
                if let FillKey::Line(line) = fill {
                    st.fill_masks.remove(&line);
                }
                st.l1.complete_fill(fill);
                if let Some(waiters) = st.fill_waiters.remove(&fill) {
                    match waiters {
                        Waiters::One(w, b) => warps[w as usize].clear_pending(b, now),
                        Waiters::Many(list) => {
                            for (w, b) in list {
                                warps[w as usize].clear_pending(b, now);
                            }
                        }
                    }
                }
                // The completed fill freed an MSHR entry: retry parked
                // accesses (FIFO) until one stalls again.
                drain_retries(st, warps, now, out);
            }
            ReplyTo::Warp { warp, barrier } => {
                st.outstanding = st
                    .outstanding
                    .checked_sub(1)
                    .expect("response without an outstanding request");
                warps[warp as usize].clear_pending(barrier, now);
            }
            ReplyTo::None => unreachable!("mutation-only requests get no response"),
        }
    }

    /// Run issue cycles strictly BEFORE `horizon`. Emits memory requests
    /// into `out` (L1Shared mode only).
    ///
    /// `horizon` is the earliest time a SELF-affecting external event can
    /// occur — the caller passes the earliest response already scheduled
    /// for this SM (other SMs' events are NOT barriers: cross-SM ordering
    /// lives entirely in the engine queue's `(time, seq)` order on the L2
    /// access stream). Requests emitted DURING the call tighten the cap
    /// to `issue_time + l1_miss_latency` (the L2-hit lower bound of the
    /// latency ladder) — their responses are not scheduled yet, so the
    /// caller cannot know them. The clock never crosses the cap, not even
    /// by skip-ahead: a response arriving at the cap may ready a warp
    /// EARLIER than the next known timer, and the clock is monotonic.
    pub fn advance(
        &mut self,
        horizon: u64,
        out: &mut Vec<MemRequest>,
    ) -> Result<Advance, IssueError> {
        // New emission batch: line-fill folds are only valid within one
        // call's `out` (the caller schedules it after we return).
        if let MemModel::L1Shared(st) = &mut self.mem {
            st.line_folds.clear();
        }
        let mut h = horizon;
        while self.unfinished_warps > 0 {
            if self.t >= h {
                return Ok(Advance::Idle {
                    wake_at: Some(self.t),
                });
            }
            let before = out.len();
            let issued_this_cycle = self.issue_cycle(out)?;
            // A reply-carrying request emitted this cycle tightens the
            // safe horizon (mutation-only store traffic cannot affect
            // this SM and does not).
            if out[before..].iter().any(|r| r.reply != ReplyTo::None) {
                h = h.min(self.t + self.params.l1_miss_latency);
            }
            if issued_this_cycle {
                self.t += 1;
                continue;
            }
            // Nothing ready: skip to the next KNOWN timer (resident warps
            // only — retired warps left the sub-core lists; warps waiting
            // on in-flight responses have no computable time and are
            // excluded).
            let next_t = self
                .subcore_warps
                .iter()
                .flatten()
                .filter_map(|&w| self.warps[w].ready_at())
                .min();
            match next_t {
                Some(nt) => {
                    let target = nt.max(self.t + 1);
                    if target >= h {
                        // Park AT the horizon instead of jumping past it:
                        // the event there may ready a warp before the
                        // next known timer.
                        return Ok(Advance::Idle { wake_at: Some(h) });
                    }
                    self.t = target;
                }
                None if self.outstanding_responses() > 0 => {
                    return Ok(Advance::Idle { wake_at: None });
                }
                None => {
                    if let MemModel::L1Shared(st) = &self.mem {
                        debug_assert!(
                            st.retry.is_empty(),
                            "MSHR-stalled work implies in-flight fills"
                        );
                    }
                    return Err(IssueError::NoProgress {
                        time: self.t,
                        unfinished: self.unfinished_warps,
                    });
                }
            }
        }
        Ok(Advance::Done)
    }

    /// One issue cycle at `self.t` across the four sub-cores. Returns
    /// whether anything issued.
    fn issue_cycle(&mut self, out: &mut Vec<MemRequest>) -> Result<bool, IssueError> {
        let t = self.t;
        let mut issued_this_cycle = false;
        for sc in 0..SUB_CORES {
            // CGGTY pick: held warp while ready, else youngest ready.
            let pick = match self.hold[sc] {
                Some(w) if self.warps[w].is_ready(t) => Some(w),
                _ => self.subcore_warps[sc]
                    .iter()
                    .copied()
                    .filter(|&w| self.warps[w].is_ready(t))
                    .max_by_key(|&w| self.warps[w].arrival),
            };
            let Some(wix) = pick else { continue };
            issued_this_cycle = true;

            // Issue exactly one instruction from warp `wix`.
            let (blk_ix, launch_index, warp_slot, slot_class, control, inst, warp_sectors) = {
                let w = &self.warps[wix];
                let slot = w.next_slot.expect("ready warp has a pending slot");
                (
                    w.block,
                    w.launch_index,
                    w.warp_slot,
                    slot.op_class,
                    slot.control,
                    w.insts[w.next],
                    w.sectors,
                )
            };
            self.issued += 1;

            // Memory-model interaction happens at issue time regardless of
            // whether a dependence barrier consumes the ready time.
            let data_ready = if matches!(self.mem, MemModel::L1Shared(_)) {
                self.mem_step_shared(wix, inst.kind, warp_sectors, control.write_barrier, out)
            } else {
                Some(data_ready_time(
                    inst.kind,
                    warp_sectors,
                    t,
                    &self.params,
                    self.mem.const_l1_mut(),
                ))
            };

            // Yield semantics per the CGGTY paper: the yield bit means
            // "do not issue this warp in the NEXT cycle" — a one-cycle
            // bubble when no other warp is ready — on top of the stall
            // count.
            let stall = u64::from(control.stall.max(1));
            let mut eligible = t + stall;
            if control.yield_flag {
                eligible = eligible.max(t + 2);
            }
            self.warps[wix].eligible = eligible;
            // Dependence barriers are COUNTERS in hardware (producer
            // increments at issue, decrements at write-back; the
            // consumer waits for zero), so with several in-flight
            // producers on one index the barrier clears at the LATEST
            // write-back: accumulate with max, never overwrite — a
            // later fast producer (L1 hit) must not shorten a pending
            // slow one (miss). Unknown write-backs (in-flight responses)
            // are tracked in `barrier_pending`.
            if let Some(dr) = data_ready {
                self.warps[wix].stamp_clear(control.write_barrier, dr);
            }
            self.warps[wix].stamp_clear(control.read_barrier, t + self.params.read_barrier_latency);

            // Advance the warp's stream.
            self.warps[wix].next += 1;
            if self.warps[wix].next >= self.warps[wix].insts.len() {
                self.warps[wix].done = true;
                self.warps[wix].next_slot = None;
                self.unfinished_warps -= 1;
                self.block_states[blk_ix].unfinished -= 1;
                // Retire from the sub-core pick list (swap-remove +
                // position fix-up) so pick/skip-ahead scans stay
                // bounded by RESIDENT warps, not warps-ever-admitted.
                let pos = self.warps[wix].pos_in_subcore;
                self.subcore_warps[sc].swap_remove(pos);
                if let Some(&moved) = self.subcore_warps[sc].get(pos) {
                    self.warps[moved].pos_in_subcore = pos;
                }
                if self.block_states[blk_ix].unfinished == 0 {
                    self.resident_warps -= self.block_states[blk_ix].warp_count;
                    self.admit_blocks(t + 1)?;
                }
            } else {
                let next_pc = self.warps[wix].insts[self.warps[wix].next].pc;
                self.warps[wix].next_slot =
                    Some(lookup(self.schedule, next_pc, launch_index, warp_slot)?);
            }
            // Single hold write: never hold a yielded or finished warp.
            self.hold[sc] = (!control.yield_flag && !self.warps[wix].done).then_some(wix);

            // Block barrier arrival AFTER stream bookkeeping: a finished
            // warp never waits.
            if slot_class == OpClass::BlockBarrier && !self.warps[wix].done {
                self.warps[wix].at_block_barrier = true;
                self.block_states[blk_ix].waiting += 1;
            }
            // Release check: every unfinished warp of the block waiting.
            // The block's warps are contiguous in `warps` (admission
            // pushes them in one run), so scan only that range.
            let bs = &self.block_states[blk_ix];
            if bs.waiting > 0 && bs.waiting == bs.unfinished {
                let (first, count) = (bs.first_warp, bs.admitted_warps);
                self.block_states[blk_ix].waiting = 0;
                for w in &mut self.warps[first..first + count] {
                    if w.at_block_barrier {
                        w.at_block_barrier = false;
                        w.eligible = w.eligible.max(t + 1);
                    }
                }
            }
        }
        Ok(issued_this_cycle)
    }
}

/// Register a barrier unit that clears when `fill` completes (no-op on
/// an invalid barrier index — the fill itself still completes).
fn add_fill_waiter(
    st: &mut SharedL2State,
    warps: &mut [WarpState<'_>],
    fill: FillKey,
    warp: usize,
    barrier: u8,
) {
    if usize::from(barrier) >= DEP_BARRIERS {
        return;
    }
    warps[warp].barrier_pending[usize::from(barrier)] += 1;
    match st.fill_waiters.entry(fill) {
        Entry::Vacant(v) => {
            v.insert(Waiters::One(warp as u32, barrier));
        }
        Entry::Occupied(mut o) => match o.get_mut() {
            Waiters::Many(list) => list.push((warp as u32, barrier)),
            slot => {
                let Waiters::One(w0, b0) = *slot else {
                    unreachable!("non-Many slot is One")
                };
                *slot = Waiters::Many(vec![(w0, b0), (warp as u32, barrier)]);
            }
        },
    }
}

/// Emission surface of one access BATCH (one instruction's sector
/// One eligible sector access in L1Shared mode at time `now`, emitting a
/// request on a fresh miss. Returns `Ok(known_ready)` for hits,
/// `Ok(None)` when the data waits on a response, `Err(())` on an MSHR
/// stall (caller parks the remainder).
///
/// A sector merging onto a line fill emitted during the CURRENT
/// emission batch (`st.line_folds` — one `advance`/`on_response` call)
/// folds its bit into that request's `sector_mask`, so one L2
/// transaction carries the batch's sectors and worst-of-mask pricing
/// covers every warp/instruction/retry-entry of the batch. A merge onto
/// an OLDER in-flight fill emits a follow-up `ReplyTo::None` sector
/// fetch instead — the L2 must see every fetched sector exactly once
/// per fill (NCU sector accounting + L2 state fidelity), while the
/// waiter's timing stays bound to the original fill response (a
/// documented approximation: the follow-up's bytes may complete after
/// the waiter cleared, and same-timestamp queue order may process a
/// follow-up after its parent fill's response).
#[allow(clippy::result_unit_err)]
fn access_shared(
    st: &mut SharedL2State,
    warps: &mut [WarpState<'_>],
    wix: usize,
    barrier: u8,
    sector_id: i64,
    now: u64,
    out: &mut Vec<MemRequest>,
) -> Result<Option<u64>, ()> {
    let sbit = 1u8 << (sector_id & 3) as u8;
    match st.l1.access_eligible_ev(sector_id, now) {
        EvAccess::Hit { ready } => Ok(Some(ready)),
        EvAccess::MissRequest { fill } => {
            out.push(MemRequest::l1_fill(now, sector_id, fill));
            st.outstanding += 1;
            if let FillKey::Line(line) = fill {
                st.fill_masks.insert(line, sbit);
                st.line_folds.push((line, out.len() - 1));
            }
            add_fill_waiter(st, warps, fill, wix, barrier);
            Ok(None)
        }
        EvAccess::MergedPending { fill } => {
            if let FillKey::Line(line) = fill {
                // Pairing invariant (proven at the 2026-07-07 gate): a
                // line-pending merge always has a live fill_masks entry
                // (created at MissRequest, removed at the response that
                // also completes the fill). Fail LOUD if a future
                // change breaks it — a fabricated empty mask would
                // silently re-fetch already-sent sectors.
                let sent = st
                    .fill_masks
                    .get_mut(&line)
                    .expect("line-pending merge without a fill_masks entry");
                if *sent & sbit == 0 {
                    *sent |= sbit;
                    if let Some(&(_, ix)) = st.line_folds.iter().find(|&&(l, _)| l == line) {
                        out[ix].sector_mask |= sbit;
                    } else {
                        out.push(MemRequest::l1_followup_fetch(now, sector_id));
                    }
                }
                // Re-touch of an already-sent sector: MSHR merge, no
                // new fetch (matches NCU's xbar sector accounting).
            }
            // Sector-fill merges need nothing: that sector's fetch was
            // sent with its own request.
            add_fill_waiter(st, warps, fill, wix, barrier);
            Ok(None)
        }
        EvAccess::Stalled => Err(()),
    }
}

fn drain_retries(
    st: &mut SharedL2State,
    warps: &mut [WarpState<'_>],
    now: u64,
    out: &mut Vec<MemRequest>,
) {
    while let Some(mut r) = st.retry.pop_front() {
        let sectors = warps[r.warp].sectors;
        // The whole drain is ONE emission batch (st.line_folds, cleared
        // by on_response): a line fill emitted by an earlier retry
        // entry folds later entries' same-line sectors — same-timestamp
        // requests must not split one line transaction (gate finding).
        while r.lo < r.hi {
            match access_shared(st, warps, r.warp, r.barrier, sectors[r.lo], now, out) {
                Ok(Some(ready)) => warps[r.warp].stamp_clear(r.barrier, ready),
                Ok(None) => {}
                Err(()) => {
                    // Still full: keep the entry (and its pending unit)
                    // parked; nothing behind it can proceed either.
                    st.retry.push_front(r);
                    return;
                }
            }
            r.lo += 1;
        }
        // Entry fully drained: release its pending unit (new waiters
        // were registered above, so the barrier never transiently reads
        // clear while work remains).
        warps[r.warp].release_pending(r.barrier);
    }
}

fn lookup(
    schedule: &FunctionSchedule,
    pc: u32,
    launch_index: i64,
    warp_slot: i32,
) -> Result<&IssueSlot, IssueError> {
    schedule
        .by_pc
        .get(&u64::from(pc))
        .ok_or(IssueError::PcNotInSchedule {
            pc,
            launch_index,
            warp_slot,
            function_name: schedule.function_name.clone(),
        })
}

/// Replay one SM's blocks to completion under the v0 issue model
/// (synchronous Flat/L1Const modes; the shared-L2 mode runs through
/// [`SmCore`] + the event system). `l1_config`, when supplied, attaches
/// a FRESH per-SM L1 (cold cache per kernel, NCU `--cache-control all`
/// parity).
pub fn simulate_sm(
    blocks: &[BlockReplay],
    schedule: &FunctionSchedule,
    params: &IssueModelParams,
    l1_config: Option<L1Config>,
) -> Result<SmSimResult, IssueError> {
    let mode = match l1_config {
        Some(cfg) => SmMemMode::L1Const(cfg),
        None => SmMemMode::Flat,
    };
    let mut core = SmCore::new(blocks, schedule, params, mode)?;
    let mut out = Vec::new();
    match core.advance(u64::MAX, &mut out)? {
        Advance::Done => {}
        Advance::Idle { .. } => unreachable!("synchronous modes emit no requests"),
    }
    debug_assert!(out.is_empty(), "synchronous modes emit no requests");
    Ok(core.result())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::collections::BTreeMap;

    use sim_mem::CacheOpToken;

    use super::*;
    use crate::control_bits::{BARRIER_NONE, ControlBits};
    use crate::program::{FunctionSchedule, IssueSlot, OpClass};

    fn cb(stall: u8) -> ControlBits {
        ControlBits {
            stall,
            yield_flag: false,
            write_barrier: BARRIER_NONE,
            read_barrier: BARRIER_NONE,
            wait_mask: 0,
            reuse: 0,
        }
    }

    fn slot(control: ControlBits, op_class: OpClass, text: &str) -> IssueSlot {
        IssueSlot {
            control,
            op_class,
            mem_role: sim_mem::MemRole::None,
            opcode_text: text.to_owned(),
        }
    }

    fn schedule(slots: Vec<(u64, IssueSlot)>) -> FunctionSchedule {
        FunctionSchedule {
            arch: ArchId::Sm86,
            function_unique_id: 1,
            kernel_name: "k___0".to_owned(),
            function_name: "k".to_owned(),
            by_pc: slots.into_iter().collect::<BTreeMap<_, _>>(),
        }
    }

    fn params() -> IssueModelParams {
        IssueModelParams {
            arch: ArchId::Sm86,
            l1_hit_latency: 34,
            l1_miss_latency: 244,
            global_flat_latency: 400,
            shared_write_barrier_latency: 23,
            non_memory_write_barrier_latency: 25,
            read_barrier_latency: 8,
            max_resident_warps_per_sm: 48,
        }
    }

    #[test]
    fn latency_conversion_rejects_u64_boundary_before_cast() {
        assert!(checked_latency("boundary", u64::MAX as f64).is_err());
        assert!(checked_latency("infinite", f64::INFINITY).is_err());
        assert!(checked_latency("negative", -1.0).is_err());
        assert_eq!(checked_latency("fractional", 12.01).unwrap(), 13);
    }

    fn l1_config() -> L1Config {
        L1Config {
            arch: ArchId::Sm86,
            line_capacity: 768,
            mshr_count: Some(32),
            hit_latency: 34,
            miss_latency: 244,
        }
    }

    fn eligible(off: u32, len: u16) -> AccessKind {
        AccessKind::GlobalEligible {
            off,
            len,
            token: CacheOpToken::LdgECa,
        }
    }

    fn inst(pc: u32) -> ReplayInst {
        ReplayInst {
            pc,
            kind: AccessKind::NonMemory,
        }
    }

    fn warp(slot_ix: i32, insts: Vec<ReplayInst>) -> WarpReplay {
        WarpReplay {
            warp_slot: slot_ix,
            insts,
            sectors: Vec::new(),
        }
    }

    #[test]
    fn single_warp_stall_timing() {
        // Three instructions, stalls 1,1,1: issues at t=0,1,2 → cycles 3.
        let sched = schedule(vec![
            (0, slot(cb(1), OpClass::Other, "IADD3")),
            (16, slot(cb(1), OpClass::Other, "IADD3")),
            (32, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0), inst(16), inst(32)])],
        }];
        let r = simulate_sm(&blocks, &sched, &params(), None).unwrap();
        assert_eq!(r.instructions_issued, 3);
        assert_eq!(r.cycles, 3);
        assert!(r.l1.is_none());
    }

    #[test]
    fn stall_counts_delay_reissue() {
        // stalls 4,4: issues at t=0,4 → cycles 5.
        let sched = schedule(vec![
            (0, slot(cb(4), OpClass::Other, "FFMA")),
            (16, slot(cb(4), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0), inst(16)])],
        }];
        let r = simulate_sm(&blocks, &sched, &params(), None).unwrap();
        assert_eq!(r.instructions_issued, 2);
        assert_eq!(r.cycles, 5);
    }

    #[test]
    fn flat_write_barrier_wait_blocks_consumer_without_l1() {
        // inst0 (global load, no L1 attached) sets barrier 0; inst1 waits.
        // Barrier clears at the flat latency 400 → inst1 at t=400.
        let mut load = cb(1);
        load.write_barrier = 0;
        let mut consumer = cb(1);
        consumer.wait_mask = 0b000001;
        let sched = schedule(vec![
            (0, slot(load, OpClass::Other, "LDG.E")),
            (16, slot(consumer, OpClass::Other, "FFMA")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: eligible(0, 1),
                    },
                    inst(16),
                ],
                sectors: vec![0],
            }],
        }];
        let r = simulate_sm(&blocks, &sched, &params(), None).unwrap();
        assert_eq!(r.instructions_issued, 2);
        assert_eq!(r.cycles, 401);
    }

    #[test]
    fn l1_hit_shortens_the_dependent_load_chain() {
        // Two dependent loads of the SAME sector + consumers. With the L1
        // attached, the first load misses (ready at 244) and the second
        // HITS (ready at +34); without it both cost the flat 400.
        let mut load = cb(1);
        load.write_barrier = 0;
        let mut consumer = cb(1);
        consumer.wait_mask = 0b000001;
        let sched = schedule(vec![
            (0, slot(load, OpClass::Other, "LDG.E")),
            (16, slot(consumer, OpClass::Other, "FFMA")),
            (32, slot(load, OpClass::Other, "LDG.E")),
            (48, slot(consumer, OpClass::Other, "EXIT")),
        ]);
        let mk = || {
            vec![BlockReplay {
                launch_index: 0,
                warps: vec![WarpReplay {
                    warp_slot: 0,
                    insts: vec![
                        ReplayInst {
                            pc: 0,
                            kind: eligible(0, 1),
                        },
                        inst(16),
                        ReplayInst {
                            pc: 32,
                            kind: eligible(1, 1),
                        },
                        inst(48),
                    ],
                    sectors: vec![7, 7],
                }],
            }]
        };
        let with_l1 = simulate_sm(&mk(), &sched, &params(), Some(l1_config())).unwrap();
        let without = simulate_sm(&mk(), &sched, &params(), None).unwrap();
        let c = with_l1.l1.unwrap();
        assert_eq!(c.hits, 1);
        assert_eq!(c.misses, 1);
        assert_eq!(c.line_alloc_misses, 1);
        // miss chain: barrier clears 244, consumer at 244; hit chain: the
        // second load issues at 245, hit ready 245+34; EXIT at 279.
        assert!(with_l1.cycles < without.cycles);
        assert_eq!(without.cycles, 802);
    }

    #[test]
    fn bypass_untagged_and_store_sectors_are_counted() {
        let sched = schedule(vec![
            (0, slot(cb(1), OpClass::Other, "LDG.E.CG")),
            (16, slot(cb(1), OpClass::Other, "LDG.E")),
            (32, slot(cb(1), OpClass::Other, "STG.E")),
            (48, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalBypass {
                            off: 0,
                            len: 2,
                            token: CacheOpToken::LdgECg,
                        },
                    },
                    ReplayInst {
                        pc: 16,
                        kind: AccessKind::GlobalUntagged { off: 2, len: 1 },
                    },
                    ReplayInst {
                        pc: 32,
                        kind: AccessKind::GlobalStore { off: 3, len: 2 },
                    },
                    inst(48),
                ],
                sectors: vec![0, 1, 2, 3, 4],
            }],
        }];
        let r = simulate_sm(&blocks, &sched, &params(), Some(l1_config())).unwrap();
        let c = r.l1.unwrap();
        assert_eq!(c.bypass_sectors, 2);
        assert_eq!(c.untagged_sectors, 1);
        assert_eq!(c.store_sectors, 2);
        assert_eq!(c.store_hits, 0, "stores never allocate");
        assert_eq!(c.hits + c.misses, 0, "no modeled accesses in this stream");
    }

    #[test]
    fn yield_bit_excludes_warp_next_cycle() {
        // CGGTY paper: yield = "do not issue this warp next cycle"; with
        // no other warp ready that is a one-cycle bubble. stall 1 +
        // yield → second instruction issues at t=2, cycles=3.
        let mut y = cb(1);
        y.yield_flag = true;
        let sched = schedule(vec![
            (0, slot(y, OpClass::Other, "IADD3")),
            (16, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0), inst(16)])],
        }];
        let r = simulate_sm(&blocks, &sched, &params(), None).unwrap();
        assert_eq!(r.cycles, 3);
    }

    #[test]
    fn shared_barrier_index_accumulates_like_a_counter() {
        // Two producers on barrier 0: a slow global (flat 400) then a
        // fast non-memory (25). The consumer waits on bit 0 and must
        // wait for the SLOW producer — hardware barriers are counters,
        // a later fast producer must not shorten the pending wait.
        let mut slow = cb(1);
        slow.write_barrier = 0;
        let mut fast = cb(1);
        fast.write_barrier = 0;
        let mut consumer = cb(1);
        consumer.wait_mask = 0b000001;
        let sched = schedule(vec![
            (0, slot(slow, OpClass::Other, "LDG.E")),
            (16, slot(fast, OpClass::Other, "S2R")),
            (32, slot(consumer, OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: eligible(0, 1),
                    },
                    inst(16),
                    inst(32),
                ],
                sectors: vec![0],
            }],
        }];
        let r = simulate_sm(&blocks, &sched, &params(), None).unwrap();
        // Consumer waits until t=400 (slow), NOT t=1+25 (fast overwrite).
        assert_eq!(r.cycles, 401);
    }

    #[test]
    fn all_empty_block_does_not_leak_residency() {
        // Block 0's warps are all traced-empty; block 1 must still be
        // admitted and replayed even under a tight residency cap.
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
        let blocks = vec![
            BlockReplay {
                launch_index: 0,
                warps: (0..4).map(|s| warp(s, vec![])).collect(),
            },
            BlockReplay {
                launch_index: 1,
                warps: vec![warp(0, vec![inst(0)])],
            },
        ];
        let mut p = params();
        p.max_resident_warps_per_sm = 4;
        let r = simulate_sm(&blocks, &sched, &p, None).unwrap();
        assert_eq!(r.instructions_issued, 1);
    }

    #[test]
    fn greedy_hold_then_youngest_switch() {
        // Two warps on the same sub-core (slots 0 and 4); everyone ready:
        // one issue per cycle on the shared sub-core → 4 cycles.
        let sched = schedule(vec![
            (0, slot(cb(1), OpClass::Other, "IADD3")),
            (16, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![
                warp(0, vec![inst(0), inst(16)]),
                warp(4, vec![inst(0), inst(16)]),
            ],
        }];
        let r = simulate_sm(&blocks, &sched, &params(), None).unwrap();
        assert_eq!(r.instructions_issued, 4);
        assert_eq!(r.cycles, 4);
    }

    #[test]
    fn warps_on_distinct_subcores_issue_in_parallel() {
        let sched = schedule(vec![
            (0, slot(cb(1), OpClass::Other, "IADD3")),
            (16, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: (0..4).map(|s| warp(s, vec![inst(0), inst(16)])).collect(),
        }];
        let r = simulate_sm(&blocks, &sched, &params(), None).unwrap();
        assert_eq!(r.instructions_issued, 8);
        assert_eq!(r.cycles, 2);
    }

    #[test]
    fn block_barrier_synchronizes_block() {
        // Warp 0 reaches BAR early and must wait for warp 1, which has an
        // extra pre-barrier instruction with a 12-cycle stall.
        let sched = schedule(vec![
            (0, slot(cb(1), OpClass::Other, "IADD3")),
            (8, slot(cb(12), OpClass::Other, "FFMA")),
            (
                16,
                slot(cb(1), OpClass::BlockBarrier, "BAR.SYNC.DEFER_BLOCKING"),
            ),
            (32, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![
                warp(0, vec![inst(0), inst(16), inst(32)]),
                warp(1, vec![inst(0), inst(8), inst(16), inst(32)]),
            ],
        }];
        let r = simulate_sm(&blocks, &sched, &params(), None).unwrap();
        assert_eq!(r.instructions_issued, 7);
        assert!(r.cycles >= 15, "cycles={}", r.cycles);
    }

    #[test]
    fn finished_warp_does_not_block_barrier() {
        let sched = schedule(vec![
            (0, slot(cb(1), OpClass::Other, "IADD3")),
            (
                16,
                slot(cb(1), OpClass::BlockBarrier, "BAR.SYNC.DEFER_BLOCKING"),
            ),
            (32, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![
                warp(0, vec![inst(0), inst(16), inst(32)]),
                warp(1, vec![inst(0)]),
            ],
        }];
        let r = simulate_sm(&blocks, &sched, &params(), None).unwrap();
        assert_eq!(r.instructions_issued, 4);
    }

    #[test]
    fn residency_cap_forces_waves() {
        // Block 0 on sub-cores 0/1, block 1 on sub-cores 2/3 so the
        // uncapped run can use all four sub-cores at once.
        let sched = schedule(vec![
            (0, slot(cb(1), OpClass::Other, "IADD3")),
            (16, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let mk_block = |ix: i64, slot0: i32| BlockReplay {
            launch_index: ix,
            warps: vec![
                warp(slot0, vec![inst(0), inst(16)]),
                warp(slot0 + 1, vec![inst(0), inst(16)]),
            ],
        };
        let blocks = vec![mk_block(0, 0), mk_block(1, 2)];
        let mut p = params();
        p.max_resident_warps_per_sm = 2;
        let r = simulate_sm(&blocks, &sched, &p, None).unwrap();
        assert_eq!(r.instructions_issued, 8);
        assert!(r.cycles >= 4, "cycles={}", r.cycles);
        let r2 = simulate_sm(&blocks, &sched, &params(), None).unwrap();
        assert_eq!(r2.cycles, 2);
    }

    #[test]
    fn unknown_pc_is_typed_error() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "IADD3"))]);
        let blocks = vec![BlockReplay {
            launch_index: 7,
            warps: vec![warp(3, vec![inst(0), inst(999)])],
        }];
        let err = simulate_sm(&blocks, &sched, &params(), None).unwrap_err();
        assert!(matches!(err, IssueError::PcNotInSchedule { pc: 999, .. }));
    }

    // ------------------------------------------------------------------
    // SmCore event-mode (L1Shared) tests: a manual driver stands in for
    // the system layer.
    // ------------------------------------------------------------------

    fn shared_core<'a>(
        blocks: &'a [BlockReplay],
        sched: &'a FunctionSchedule,
        p: &IssueModelParams,
    ) -> SmCore<'a> {
        SmCore::new(blocks, sched, p, SmMemMode::L1Shared(l1_config())).unwrap()
    }

    #[test]
    fn shared_mode_miss_blocks_consumer_until_response() {
        let mut load = cb(1);
        load.write_barrier = 0;
        let mut consumer = cb(1);
        consumer.wait_mask = 0b000001;
        let sched = schedule(vec![
            (0, slot(load, OpClass::Other, "LDG.E")),
            (16, slot(consumer, OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: eligible(0, 1),
                    },
                    inst(16),
                ],
                sectors: vec![0],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut out = Vec::new();
        // The load misses → request emitted; the consumer's wait has a
        // PENDING barrier → no computable timer → blocked on response.
        let a = core.advance(u64::MAX, &mut out).unwrap();
        assert_eq!(a, Advance::Idle { wake_at: None });
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].issue_time, 0);
        assert_eq!(out[0].source, L2Source::L1Fill);
        let ReplyTo::L1Fill(fill) = out[0].reply else {
            panic!("expected an L1 fill reply");
        };
        // Response at the DRAM arm (423): consumer issues at 423.
        let mut out2 = Vec::new();
        core.on_response(ReplyTo::L1Fill(fill), 423, &mut out2);
        assert!(out2.is_empty());
        assert_eq!(core.advance(u64::MAX, &mut out2).unwrap(), Advance::Done);
        let r = core.result();
        assert_eq!(r.instructions_issued, 2);
        assert_eq!(r.cycles, 424);
    }

    #[test]
    fn shared_mode_horizon_caps_run_ahead_after_a_request() {
        // Independent instructions after a miss: the SM may run ahead,
        // but never past issue_time + l1_miss_latency (the earliest
        // possible response).
        let mut load = cb(1);
        load.write_barrier = 0;
        let sched = schedule(vec![
            (0, slot(load, OpClass::Other, "LDG.E")),
            (16, slot(cb(15), OpClass::Other, "IADD3")),
            (32, slot(cb(15), OpClass::Other, "IADD3")),
            (48, slot(cb(15), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: eligible(0, 1),
                    },
                    inst(16),
                    inst(32),
                    inst(48),
                ],
                sectors: vec![0],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut out = Vec::new();
        // 4 independent issues would finish at t=46 — well inside the
        // 244-cycle response bound, so the run completes...
        assert_eq!(core.advance(u64::MAX, &mut out).unwrap(), Advance::Done);
        assert_eq!(out.len(), 1);
        // ...but a tight EXTERNAL horizon yields first.
        let blocks2 = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: eligible(0, 1),
                    },
                    inst(16),
                    inst(32),
                    inst(48),
                ],
                sectors: vec![0],
            }],
        }];
        let mut core2 = shared_core(&blocks2, &sched, &params());
        let mut out2 = Vec::new();
        let a = core2.advance(10, &mut out2).unwrap();
        assert_eq!(
            a,
            Advance::Idle { wake_at: Some(10) },
            "the next known timer (16) lies past the horizon: park AT the \
             horizon, where an event may ready an earlier warp"
        );
        assert_eq!(core2.advance(u64::MAX, &mut out2).unwrap(), Advance::Done);
    }

    #[test]
    fn shared_mode_merged_waiters_all_clear_on_one_response() {
        // Two warps load the SAME sector; warp 1 merges into warp 0's
        // fill. Both consumers wait; ONE response clears both.
        let mut load = cb(1);
        load.write_barrier = 0;
        let mut consumer = cb(1);
        consumer.wait_mask = 0b000001;
        let sched = schedule(vec![
            (0, slot(load, OpClass::Other, "LDG.E")),
            (16, slot(consumer, OpClass::Other, "EXIT")),
        ]);
        let mk_warp = |slot_ix: i32| WarpReplay {
            warp_slot: slot_ix,
            insts: vec![
                ReplayInst {
                    pc: 0,
                    kind: eligible(0, 1),
                },
                inst(16),
            ],
            sectors: vec![9],
        };
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![mk_warp(0), mk_warp(1)],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut out = Vec::new();
        assert_eq!(
            core.advance(u64::MAX, &mut out).unwrap(),
            Advance::Idle { wake_at: None }
        );
        assert_eq!(out.len(), 1, "second warp merged, no second request");
        let c = core.l1_counters().unwrap();
        assert_eq!((c.misses, c.merges), (2, 1));
        core.on_response(out[0].reply, 300, &mut out);
        assert_eq!(core.advance(u64::MAX, &mut out).unwrap(), Advance::Done);
        // Both consumers issued at 300 (distinct sub-cores) → cycles 301.
        assert_eq!(core.result().cycles, 301);
    }

    #[test]
    fn shared_mode_mshr_stall_parks_and_retries_on_completion() {
        // MSHR capacity 1: the second warp's DIFFERENT-line load stalls,
        // parks, and retries when the first fill completes.
        let mut load = cb(1);
        load.write_barrier = 0;
        let mut consumer = cb(1);
        consumer.wait_mask = 0b000001;
        let sched = schedule(vec![
            (0, slot(load, OpClass::Other, "LDG.E")),
            (16, slot(consumer, OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![
                WarpReplay {
                    warp_slot: 0,
                    insts: vec![
                        ReplayInst {
                            pc: 0,
                            kind: eligible(0, 1),
                        },
                        inst(16),
                    ],
                    sectors: vec![0],
                },
                WarpReplay {
                    warp_slot: 1,
                    insts: vec![
                        ReplayInst {
                            pc: 0,
                            kind: eligible(0, 1),
                        },
                        inst(16),
                    ],
                    sectors: vec![8],
                },
            ],
        }];
        let mut cfg = l1_config();
        cfg.mshr_count = Some(1);
        let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::L1Shared(cfg)).unwrap();
        let mut out = Vec::new();
        assert_eq!(
            core.advance(u64::MAX, &mut out).unwrap(),
            Advance::Idle { wake_at: None }
        );
        assert_eq!(out.len(), 1, "second load stalled, not requested");
        assert_eq!(core.l1_counters().unwrap().stall_events, 1);
        // First fill completes → retry emits the parked request at 244.
        let first = out[0].reply;
        out.clear();
        core.on_response(first, 244, &mut out);
        assert_eq!(out.len(), 1, "retry sent the parked sector");
        assert_eq!(out[0].issue_time, 244);
        assert_eq!(
            core.advance(u64::MAX, &mut out).unwrap(),
            Advance::Idle { wake_at: None }
        );
        let second = out[0].reply;
        core.on_response(second, 488, &mut out);
        assert_eq!(core.advance(u64::MAX, &mut out).unwrap(), Advance::Done);
        let r = core.result();
        assert_eq!(r.instructions_issued, 4);
        // Warp 1's consumer issues at 488 → cycles 489.
        assert_eq!(r.cycles, 489);
    }

    #[test]
    fn shared_mode_bypass_load_waits_on_direct_reply() {
        let mut load = cb(1);
        load.write_barrier = 2;
        let mut consumer = cb(1);
        consumer.wait_mask = 0b000100;
        let sched = schedule(vec![
            (0, slot(load, OpClass::Other, "LDG.E.CG")),
            (16, slot(consumer, OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalBypass {
                            off: 0,
                            len: 2,
                            token: CacheOpToken::LdgECg,
                        },
                    },
                    inst(16),
                ],
                sectors: vec![40, 41],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut out = Vec::new();
        assert_eq!(
            core.advance(u64::MAX, &mut out).unwrap(),
            Advance::Idle { wake_at: None }
        );
        assert_eq!(out.len(), 2, "one request per bypass sector");
        assert!(out.iter().all(|r| r.source == L2Source::Bypass));
        assert!(
            out.iter().all(|r| r.class == L2OpClass::Read),
            ".cg bypass loads are read-class at the L2"
        );
        assert!(matches!(
            out[0].reply,
            ReplyTo::Warp {
                warp: 0,
                barrier: 2
            }
        ));
        // Both responses must land before the wait clears.
        let mut sink = Vec::new();
        core.on_response(out[0].reply, 244, &mut sink);
        assert_eq!(
            core.advance(u64::MAX, &mut sink).unwrap(),
            Advance::Idle { wake_at: None },
            "one of two pending units still outstanding"
        );
        core.on_response(out[1].reply, 423, &mut sink);
        assert_eq!(core.advance(u64::MAX, &mut sink).unwrap(), Advance::Done);
        assert_eq!(core.result().cycles, 424);
    }

    #[test]
    fn shared_mode_store_emits_mutation_only_requests() {
        let sched = schedule(vec![
            (0, slot(cb(1), OpClass::Other, "STG.E")),
            (16, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalStore { off: 0, len: 2 },
                    },
                    inst(16),
                ],
                sectors: vec![100, 101],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut out = Vec::new();
        assert_eq!(core.advance(u64::MAX, &mut out).unwrap(), Advance::Done);
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|r| r.source == L2Source::Store
            && r.class == L2OpClass::WriteAtomic
            && r.reply == ReplyTo::None));
    }

    /// §6.4 slice 2 — one instruction's cold 4-sector line miss reaches
    /// the L2 as ONE request CARRYING ALL FOUR SECTORS (the merged
    /// sectors fold into the fill's mask; NCU counts 4 LTS sectors),
    /// with a single response completing every waiter.
    #[test]
    fn shared_mode_line_fill_folds_batch_sectors_into_one_request() {
        let mut load = cb(1);
        load.write_barrier = 2;
        let mut consumer = cb(1);
        consumer.wait_mask = 0b000100;
        let sched = schedule(vec![
            (0, slot(load, OpClass::Other, "LDG.E")),
            (16, slot(consumer, OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalEligible {
                            off: 0,
                            len: 4,
                            token: CacheOpToken::LdgECa,
                        },
                    },
                    inst(16),
                ],
                sectors: vec![40, 41, 42, 43],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut out = Vec::new();
        assert_eq!(
            core.advance(u64::MAX, &mut out).unwrap(),
            Advance::Idle { wake_at: None }
        );
        assert_eq!(out.len(), 1, "one L2 transaction for the whole line");
        assert_eq!(out[0].sector_mask, 0b1111);
        assert_eq!(out[0].reply, ReplyTo::L1Fill(FillKey::Line(10)));
        let mut sink = Vec::new();
        core.on_response(out[0].reply, 244, &mut sink);
        assert_eq!(core.advance(u64::MAX, &mut sink).unwrap(), Advance::Done);
        assert_eq!(core.result().cycles, 245);
    }

    /// §6.4 slice 2 — a LATER instruction merging onto an in-flight
    /// line fill: a NEW sector sends a follow-up `ReplyTo::None` fetch
    /// (the L2 must see it once), a re-touched sector sends nothing
    /// (MSHR merge), and every waiter still rides the ORIGINAL fill's
    /// one response.
    #[test]
    fn shared_mode_same_batch_cross_instruction_merge_folds() {
        // Gate fix 2026-07-07: fold scope is the EMISSION BATCH (one
        // advance call), not one instruction — a later instruction's
        // merge onto an earlier instruction's line fill folds into that
        // request's mask (so worst-of-mask pricing covers it) instead
        // of splitting off a follow-up fetch.
        let mut load_a = cb(1);
        load_a.write_barrier = 2;
        let mut load_b = cb(1);
        load_b.write_barrier = 2;
        let mut consumer = cb(1);
        consumer.wait_mask = 0b000100;
        let sched = schedule(vec![
            (0, slot(load_a, OpClass::Other, "LDG.E")),
            (16, slot(load_b, OpClass::Other, "LDG.E")),
            (32, slot(consumer, OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalEligible {
                            off: 0,
                            len: 1,
                            token: CacheOpToken::LdgECa,
                        },
                    },
                    ReplayInst {
                        pc: 16,
                        kind: AccessKind::GlobalEligible {
                            off: 1,
                            len: 2,
                            token: CacheOpToken::LdgECa,
                        },
                    },
                    inst(32),
                ],
                // A: sector 40. B: sector 41 (folds) + 40 (re-touch).
                sectors: vec![40, 41, 40],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut out = Vec::new();
        assert_eq!(
            core.advance(u64::MAX, &mut out).unwrap(),
            Advance::Idle { wake_at: None }
        );
        assert_eq!(out.len(), 1, "one line transaction for the whole batch");
        assert_eq!(out[0].sector_mask, 0b11, "B's sector folded into A's fill");
        assert_eq!(out[0].reply, ReplyTo::L1Fill(FillKey::Line(10)));
        let mut sink = Vec::new();
        core.on_response(out[0].reply, 244, &mut sink);
        assert_eq!(
            core.advance(u64::MAX, &mut sink).unwrap(),
            Advance::Done,
            "one response clears both instructions' waiters"
        );
    }

    /// A merge onto a fill emitted in an EARLIER batch (the request
    /// already left with the SM's previous yield) cannot fold — it
    /// sends exactly one follow-up `ReplyTo::None` fetch, and the
    /// waiter still rides the original fill's response.
    #[test]
    fn shared_mode_cross_batch_merge_sends_followup_fetch_once() {
        let mut load_m = cb(1);
        load_m.write_barrier = 2;
        let mut load_l = cb(1);
        load_l.write_barrier = 3;
        // Merging load: gated on barrier 2 (fill M's return) so it
        // issues in a LATER advance call while fill L is still in
        // flight; its own waiter registers on barrier 2.
        let mut load_merge = cb(1);
        load_merge.write_barrier = 2;
        load_merge.wait_mask = 0b000100;
        let mut consumer = cb(1);
        consumer.wait_mask = 0b001100;
        let sched = schedule(vec![
            (0, slot(load_m, OpClass::Other, "LDG.E")),
            (16, slot(load_l, OpClass::Other, "LDG.E")),
            (32, slot(load_merge, OpClass::Other, "LDG.E")),
            (48, slot(consumer, OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalEligible {
                            off: 0,
                            len: 1,
                            token: CacheOpToken::LdgECa,
                        },
                    },
                    ReplayInst {
                        pc: 16,
                        kind: AccessKind::GlobalEligible {
                            off: 1,
                            len: 1,
                            token: CacheOpToken::LdgECa,
                        },
                    },
                    ReplayInst {
                        pc: 32,
                        kind: AccessKind::GlobalEligible {
                            off: 2,
                            len: 1,
                            token: CacheOpToken::LdgECa,
                        },
                    },
                    inst(48),
                ],
                // M: sector 80 (line 20). L: sector 40 (line 10).
                // Merge: sector 41 (line 10, new sector).
                sectors: vec![80, 40, 41],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut out = Vec::new();
        assert_eq!(
            core.advance(u64::MAX, &mut out).unwrap(),
            Advance::Idle { wake_at: None }
        );
        assert_eq!(out.len(), 2, "two fills; the merging load is gated");
        assert_eq!(out[0].reply, ReplyTo::L1Fill(FillKey::Line(20)));
        assert_eq!(out[1].reply, ReplyTo::L1Fill(FillKey::Line(10)));
        assert_eq!(out[1].sector_mask, 0b01);
        // Fill M returns; the merging load issues in a NEW batch while
        // fill L is still in flight.
        let mut out2 = Vec::new();
        core.on_response(out[0].reply, 423, &mut out2);
        assert_eq!(
            core.advance(u64::MAX, &mut out2).unwrap(),
            Advance::Idle { wake_at: None }
        );
        assert_eq!(out2.len(), 1, "exactly one follow-up fetch");
        assert_eq!(out2[0].sector_mask, 0b10, "only the new sector");
        assert_eq!(out2[0].reply, ReplyTo::None, "no second response");
        assert_eq!(out2[0].source, L2Source::L1Fill);
        // Fill L's one response clears BOTH its own waiter (barrier 3)
        // and the merged load's waiter (barrier 2).
        let mut sink = Vec::new();
        core.on_response(out[1].reply, 846, &mut sink);
        assert_eq!(core.advance(u64::MAX, &mut sink).unwrap(), Advance::Done);
    }

    /// ATOM records ride BYPASS provenance (they never stage into the
    /// modeled L1) but carry the §6.4 WRITE class at the L2: the source
    /// says who sent the sector, the class says how it hits.
    #[test]
    fn shared_mode_atom_is_bypass_provenance_write_class() {
        let mut atom = cb(1);
        atom.write_barrier = 2;
        let mut consumer = cb(1);
        consumer.wait_mask = 0b000100;
        let sched = schedule(vec![
            (0, slot(atom, OpClass::Other, "ATOMG.E.ADD.STRONG.GPU")),
            (16, slot(consumer, OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalBypass {
                            off: 0,
                            len: 1,
                            token: CacheOpToken::Atom,
                        },
                    },
                    inst(16),
                ],
                sectors: vec![40],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut out = Vec::new();
        assert_eq!(
            core.advance(u64::MAX, &mut out).unwrap(),
            Advance::Idle { wake_at: None }
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].source, L2Source::Bypass);
        assert_eq!(out[0].class, L2OpClass::WriteAtomic);
        assert!(
            matches!(
                out[0].reply,
                ReplyTo::Warp {
                    warp: 0,
                    barrier: 2
                }
            ),
            "the atomic's return value still waits on the reply"
        );
    }
}
