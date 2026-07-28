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
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};

use sim_arch::ArchId;
use sim_mem::{
    CacheOpToken, EvAccess, EvAccessUnobserved, FillKey, L1AccessError, L1Cache, L1Config,
    L1Counters, L1MutationBudget, L2OpClass, L2Source,
};

use crate::control_bits::{ControlDomainError, DepBarrier};
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

pub(crate) fn checked_cycle_add(
    operation: &'static str,
    base: u64,
    delta: u64,
) -> Result<u64, IssueError> {
    base.checked_add(delta).ok_or(IssueError::TimeOverflow {
        operation,
        base,
        delta,
    })
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
    fn non_memory_ready(&self, t: u64) -> Result<u64, IssueError> {
        checked_cycle_add(
            "non-memory write-barrier visibility",
            t,
            self.non_memory_write_barrier_latency,
        )
    }

    /// Shared-memory visibility (one copy for BOTH memory modes).
    fn shared_ready(&self, t: u64) -> Result<u64, IssueError> {
        checked_cycle_add(
            "shared-memory write-barrier visibility",
            t,
            self.shared_write_barrier_latency,
        )
    }

    /// Store / degenerate-record / L1-bypass visibility: the L2-hit arm.
    /// The single copy both data paths must price identically — the
    /// Const-vs-Shared ablation delta depends on it.
    fn store_visibility_ready(&self, t: u64) -> Result<u64, IssueError> {
        checked_cycle_add(
            "store/bypass write-barrier visibility",
            t,
            self.l1_miss_latency,
        )
    }

    /// Validate that an attached L1 uses this parameter set's single
    /// timing authority. This runs at the system composition root as well
    /// as per-core construction so an empty SM map cannot skip the check.
    pub(crate) fn validate_l1_timing(&self, cfg: L1Config) -> Result<(), IssueError> {
        if cfg.hit_latency != self.l1_hit_latency {
            return Err(IssueError::L1TimingMismatch {
                field: "hit",
                expected: self.l1_hit_latency,
                got: cfg.hit_latency,
            });
        }
        if cfg.miss_latency != self.l1_miss_latency {
            return Err(IssueError::L1TimingMismatch {
                field: "miss",
                expected: self.l1_miss_latency,
                got: cfg.miss_latency,
            });
        }
        Ok(())
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
        "pc {pc} (block {launch_index}, warp slot {warp_slot}) has invalid issue control: {source}"
    )]
    ControlDomain {
        pc: u32,
        launch_index: i64,
        warp_slot: i32,
        #[source]
        source: ControlDomainError,
    },
    #[error(
        "pc {pc} (block {launch_index}, warp slot {warp_slot}) sector span start {off} + length {len} overflows usize"
    )]
    SectorSpanOverflow {
        pc: u32,
        launch_index: i64,
        warp_slot: i32,
        off: usize,
        len: u16,
    },
    #[error(
        "pc {pc} (block {launch_index}, warp slot {warp_slot}) sector span {off}..{end} exceeds arena length {sectors}"
    )]
    SectorSpanOutOfBounds {
        pc: u32,
        launch_index: i64,
        warp_slot: i32,
        off: usize,
        len: u16,
        end: usize,
        sectors: usize,
    },
    #[error("block {launch_index} has {warps} live warps, which does not fit in u32")]
    LiveWarpCountOutOfRange { launch_index: i64, warps: usize },
    #[error("warp {warp} was selected without a pending issue slot")]
    MissingIssueSlot { warp: usize },
    #[error("{counter} counter overflow")]
    CounterOverflow { counter: &'static str },
    #[error("{counter} counter underflow")]
    CounterUnderflow { counter: &'static str },
    #[error("issuing {add} instruction(s) from count {issued} exceeds replay cardinality {limit}")]
    InstructionCountExceedsReplay { issued: u64, add: u64, limit: u64 },
    #[error(
        "warp {warp} dependence barrier {barrier} pending count {current} + {add} overflows u32"
    )]
    PendingBarrierOverflow {
        warp: usize,
        barrier: u8,
        current: u32,
        add: u32,
    },
    #[error(
        "warp {warp} dependence barrier {barrier} pending count {current} cannot release {remove} unit(s)"
    )]
    PendingBarrierUnderflow {
        warp: usize,
        barrier: u8,
        current: u32,
        remove: u32,
    },
    #[error("response-owner identity domain is exhausted")]
    ResponseOwnerExhausted,
    #[error("memory responses are invalid for the {mode} memory mode")]
    ResponseModeMismatch { mode: &'static str },
    #[error("memory response fill {fill:?} is not in flight in the L1")]
    ResponseFillNotInFlight { fill: FillKey },
    #[error("line fill {line} has no live sector-mask bookkeeping")]
    FillMaskMissing { line: i64 },
    #[error("line {line} has sector-mask bookkeeping without an in-flight fill")]
    FillMaskWithoutFill { line: i64 },
    #[error("response-bearing request token {reply:?} is already outstanding")]
    ResponseTokenAlreadyOutstanding { reply: ReplyTo },
    #[error("memory response token {reply:?} disappeared after successful preflight")]
    ResponseTokenRemovalFailed { reply: ReplyTo },
    #[error("memory response targets warp index {warp}, but this SM has {warps} admitted warps")]
    ResponseWarpOutOfRange { warp: usize, warps: usize },
    #[error("memory response token {reply:?} was not emitted or is no longer outstanding")]
    ResponseTokenNotOutstanding { reply: ReplyTo },
    #[error(
        "memory response targets warp {warp} dependence barrier {barrier}, which is not pending"
    )]
    ResponseBarrierNotPending { warp: usize, barrier: u8 },
    #[error("mutation-only memory request cannot carry a response event")]
    UnexpectedMutationOnlyResponse,
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
    #[error("L1 {field} latency {got} does not match issue parameter {expected}")]
    L1TimingMismatch {
        field: &'static str,
        expected: u64,
        got: u64,
    },
    #[error("{name} latency must be finite, positive, and fit in u64 (got {value})")]
    InvalidLatency { name: &'static str, value: f64 },
    #[error("{operation}: cycle addition {base} + {delta} overflows u64")]
    TimeOverflow {
        operation: &'static str,
        base: u64,
        delta: u64,
    },
    #[error("memory response at t={now} precedes the SM clock t={clock}")]
    ResponseBeforeClock { now: u64, clock: u64 },
    #[error("advance horizon t={horizon} precedes the SM clock t={clock}")]
    HorizonBeforeClock { horizon: u64, clock: u64 },

    #[error("issue capability unavailable: {0}")]
    Capability(#[from] sim_arch::CapabilityError),
    #[error("L1 access failed: {0}")]
    L1Access(#[from] L1AccessError),
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

/// Process-local identity of one live shared-L2 actor. It is deliberately
/// opaque and never enters simulator outputs; its only role is to prevent a
/// response capability emitted by one actor or replay generation from being
/// accepted by another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ResponseOwner(u64);

static NEXT_RESPONSE_OWNER: AtomicU64 = AtomicU64::new(1);

fn allocate_response_owner_from(next: &AtomicU64) -> Result<ResponseOwner, IssueError> {
    next.fetch_update(
        Ordering::Relaxed,
        Ordering::Relaxed,
        |current| match current {
            0 => None,
            u64::MAX => Some(0),
            _ => current.checked_add(1),
        },
    )
    .map(ResponseOwner)
    .map_err(|_| IssueError::ResponseOwnerExhausted)
}

fn allocate_response_owner() -> Result<ResponseOwner, IssueError> {
    allocate_response_owner_from(&NEXT_RESPONSE_OWNER)
}

/// Stable identity of one response-bearing request. Its fields are private:
/// callers may route a token emitted in [`MemRequest::reply`], but cannot mint
/// an identity for a request the SM never emitted. The owner component is
/// process-unique for the lifetime of the process and rejects cross-core and
/// stale-replay token swaps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResponseId {
    owner: ResponseOwner,
    warp: usize,
    instruction: usize,
    sector_index: usize,
}

impl ResponseId {
    const fn new(
        owner: ResponseOwner,
        warp: usize,
        instruction: usize,
        sector_index: usize,
    ) -> Self {
        Self {
            owner,
            warp,
            instruction,
            sector_index,
        }
    }
}

/// What the response to a [`MemRequest`] must do at arrival.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReplyTo {
    /// Complete this L1 fill (installing the line/sector) and clear the
    /// fill's registered barrier waiters.
    L1Fill {
        fill: FillKey,
        response_id: ResponseId,
    },
    /// Clear one pending unit on (warp, barrier) directly — bypass and
    /// untagged loads that never touch the modeled L1.
    Warp {
        warp: usize,
        barrier: DepBarrier,
        response_id: ResponseId,
    },
    /// Mutation-only access (stores): no response event needed.
    None,
}

impl ReplyTo {
    const fn response_id(self) -> Option<ResponseId> {
        match self {
            Self::L1Fill { response_id, .. } | Self::Warp { response_id, .. } => Some(response_id),
            Self::None => None,
        }
    }
}

impl MemRequest {
    /// An L1 fill request (line alloc or resident-sector fetch): Read
    /// class by construction — nothing that fills the modeled L1 is a
    /// write (stores hit-test without allocating, atomics bypass). The
    /// typed constructors keep the (source, class, reply) coupling in
    /// one place so an invalid combination is unwritable at emission
    /// sites (gate finding 2026-07-07).
    fn l1_fill(issue_time: u64, sector_id: i64, fill: FillKey, response_id: ResponseId) -> Self {
        Self {
            issue_time,
            sector_id,
            sector_mask: 1 << (sector_id & 3) as u8,
            source: L2Source::L1Fill,
            class: L2OpClass::Read,
            reply: ReplyTo::L1Fill { fill, response_id },
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

/// Exclusive limit for one [`SmCore::advance`] call. A bounded horizon may be
/// any cycle, including `u64::MAX`; unbounded replay is represented by its own
/// variant instead of stealing a valid timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayHorizon {
    Bounded(u64),
    Unbounded,
}

impl ReplayHorizon {
    fn bounded_time(self) -> Option<u64> {
        match self {
            Self::Bounded(time) => Some(time),
            Self::Unbounded => None,
        }
    }

    fn reached(self, now: u64) -> bool {
        self.bounded_time().is_some_and(|bound| now >= bound)
    }

    fn tighten(self, bound: u64) -> Self {
        match self {
            Self::Bounded(current) => Self::Bounded(current.min(bound)),
            Self::Unbounded => Self::Bounded(bound),
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

fn checked_u64_counter_add(
    counter: &'static str,
    current: u64,
    add: u64,
) -> Result<u64, IssueError> {
    current
        .checked_add(add)
        .ok_or(IssueError::CounterOverflow { counter })
}

fn checked_usize_counter_add(
    counter: &'static str,
    current: usize,
    add: usize,
) -> Result<usize, IssueError> {
    current
        .checked_add(add)
        .ok_or(IssueError::CounterOverflow { counter })
}

fn checked_u32_counter_add(
    counter: &'static str,
    current: u32,
    add: u32,
) -> Result<u32, IssueError> {
    current
        .checked_add(add)
        .ok_or(IssueError::CounterOverflow { counter })
}

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
    fn next_issue_retires(&self) -> Result<bool, IssueError> {
        let next = checked_usize_counter_add("warp instruction cursor", self.next, 1)?;
        Ok(next == self.insts.len())
    }

    fn preflight_pending_add(
        &self,
        warp: usize,
        barrier: Option<DepBarrier>,
        add: u32,
    ) -> Result<(), IssueError> {
        let Some(barrier) = barrier else {
            return Ok(());
        };
        let current = self.barrier_pending[barrier.index()];
        current
            .checked_add(add)
            .ok_or(IssueError::PendingBarrierOverflow {
                warp,
                barrier: barrier.wire(),
                current,
                add,
            })?;
        Ok(())
    }

    fn preflight_pending_remove(
        &self,
        warp: usize,
        barrier: DepBarrier,
        remove: u32,
    ) -> Result<(), IssueError> {
        let current = self.barrier_pending[barrier.index()];
        current
            .checked_sub(remove)
            .ok_or(IssueError::PendingBarrierUnderflow {
                warp,
                barrier: barrier.wire(),
                current,
                remove,
            })?;
        Ok(())
    }

    /// Producer bookkeeping: pending (unknown write-back) units on a
    /// dependence barrier; `None` carries no dependency.
    fn add_pending(
        &mut self,
        warp: usize,
        barrier: Option<DepBarrier>,
        add: u32,
    ) -> Result<(), IssueError> {
        self.preflight_pending_add(warp, barrier, add)?;
        if let Some(barrier) = barrier {
            let pending = &mut self.barrier_pending[barrier.index()];
            let current = *pending;
            *pending = current
                .checked_add(add)
                .ok_or(IssueError::PendingBarrierOverflow {
                    warp,
                    barrier: barrier.wire(),
                    current,
                    add,
                })?;
        }
        Ok(())
    }

    /// Pending units resolved at `now`: decrement AND max-accumulate the
    /// clear stamp (hardware counter semantics — the single copy of the
    /// pairing every response path must preserve).
    fn clear_pending(
        &mut self,
        warp: usize,
        barrier: DepBarrier,
        remove: u32,
        now: u64,
    ) -> Result<(), IssueError> {
        self.preflight_pending_remove(warp, barrier, remove)?;
        let b = barrier.index();
        let current = self.barrier_pending[b];
        self.barrier_pending[b] =
            current
                .checked_sub(remove)
                .ok_or(IssueError::PendingBarrierUnderflow {
                    warp,
                    barrier: barrier.wire(),
                    current,
                    remove,
                })?;
        self.barrier_clear[b] = self.barrier_clear[b].max(now);
        Ok(())
    }

    /// Release pending units WITHOUT a stamp (a retry entry's own units;
    /// its per-sector waiters carry the stamps).
    fn release_pending(
        &mut self,
        warp: usize,
        barrier: Option<DepBarrier>,
        remove: u32,
    ) -> Result<(), IssueError> {
        let Some(barrier) = barrier else {
            return Ok(());
        };
        self.preflight_pending_remove(warp, barrier, remove)?;
        let pending = &mut self.barrier_pending[barrier.index()];
        let current = *pending;
        *pending = current
            .checked_sub(remove)
            .ok_or(IssueError::PendingBarrierUnderflow {
                warp,
                barrier: barrier.wire(),
                current,
                remove,
            })?;
        Ok(())
    }

    /// Max-accumulate a KNOWN clear stamp on a dependence barrier (counter
    /// semantics: never overwrite); `None` carries no dependency.
    fn stamp_clear(&mut self, barrier: Option<DepBarrier>, at: u64) {
        if let Some(barrier) = barrier {
            let clear = &mut self.barrier_clear[barrier.index()];
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
            if slot.control.wait_mask.contains_index(b) {
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
/// attached. Producer visibility is computed only when a write barrier consumes
/// it; cache accesses still run without one because they mutate modeled state.
fn data_ready_time(
    kind: AccessKind,
    sectors: &[i64],
    t: u64,
    params: &IssueModelParams,
    l1: Option<&mut L1Cache>,
    write_barrier: Option<DepBarrier>,
) -> Result<Option<u64>, IssueError> {
    let needs_ready = write_barrier.is_some();
    match kind {
        AccessKind::NonMemory => {
            if needs_ready {
                params.non_memory_ready(t).map(Some)
            } else {
                Ok(None)
            }
        }
        AccessKind::Shared => {
            if needs_ready {
                params.shared_ready(t).map(Some)
            } else {
                Ok(None)
            }
        }
        AccessKind::GlobalEligible { off, len, .. } => match l1 {
            Some(l1) => {
                let mut ready = needs_ready
                    .then(|| checked_cycle_add("L1 hit-pipeline floor", t, params.l1_hit_latency))
                    .transpose()?;
                for &sector in &sectors[off..off + usize::from(len)] {
                    if let Some(current) = &mut ready {
                        let (_, sector_ready) = l1.access_eligible(sector, t)?;
                        *current = (*current).max(sector_ready);
                    } else {
                        l1.access_eligible_unobserved(sector, t)?;
                    }
                }
                Ok(ready)
            }
            None => {
                if needs_ready {
                    checked_cycle_add("flat global visibility", t, params.global_flat_latency)
                        .map(Some)
                } else {
                    Ok(None)
                }
            }
        },
        AccessKind::GlobalBypass { len, token, .. } => match l1 {
            Some(l1) => {
                l1.count_bypass(u64::from(len), token == CacheOpToken::Atom)?;
                if needs_ready {
                    params.store_visibility_ready(t).map(Some)
                } else {
                    Ok(None)
                }
            }
            None => {
                if needs_ready {
                    checked_cycle_add("flat global visibility", t, params.global_flat_latency)
                        .map(Some)
                } else {
                    Ok(None)
                }
            }
        },
        AccessKind::GlobalUntagged { len, .. } => match l1 {
            Some(l1) => {
                l1.count_untagged(u64::from(len))?;
                if needs_ready {
                    params.store_visibility_ready(t).map(Some)
                } else {
                    Ok(None)
                }
            }
            None => {
                if needs_ready {
                    checked_cycle_add("flat global visibility", t, params.global_flat_latency)
                        .map(Some)
                } else {
                    Ok(None)
                }
            }
        },
        AccessKind::GlobalStore { off, len } => match l1 {
            Some(l1) => {
                for &sector in &sectors[off..off + usize::from(len)] {
                    l1.probe_store(sector, t)?;
                }
                if needs_ready {
                    params.store_visibility_ready(t).map(Some)
                } else {
                    Ok(None)
                }
            }
            None => {
                if needs_ready {
                    checked_cycle_add("flat global visibility", t, params.global_flat_latency)
                        .map(Some)
                } else {
                    Ok(None)
                }
            }
        },
    }
}

/// An MSHR-stalled remainder of one instruction's sector list, retried
/// (in FIFO order) whenever a fill completion frees an entry.
struct Retry {
    warp: usize,
    instruction: usize,
    /// The stalled instruction's optional write-dependence barrier. When
    /// present, the entry holds one pending unit until fully drained.
    barrier: Option<DepBarrier>,
    /// Unresolved slice `warps[warp].sectors[lo..hi]`.
    lo: usize,
    hi: usize,
}

#[derive(Clone, Copy)]
struct FillWaiter {
    warp: usize,
    barrier: DepBarrier,
    units: u32,
}

/// Barrier units cleared when one keyed L1 fill completes. Inline for
/// the dominant single-waiter identity and pre-aggregated by (warp, barrier),
/// so response preflight never rebuilds ordinary waiter counts in a heap map.
enum Waiters {
    One(FillWaiter),
    Many(Vec<FillWaiter>),
}

impl Waiters {
    fn add(&mut self, warp: usize, barrier: DepBarrier) -> Result<(), IssueError> {
        let new_waiter = FillWaiter {
            warp,
            barrier,
            units: 1,
        };
        match self {
            Self::One(waiter) if waiter.warp == warp && waiter.barrier == barrier => {
                waiter.units = checked_u32_counter_add("fill waiter unit", waiter.units, 1)?;
            }
            Self::One(waiter) => {
                *self = Self::Many(vec![*waiter, new_waiter]);
            }
            Self::Many(waiters) => {
                if let Some(waiter) = waiters
                    .iter_mut()
                    .find(|waiter| waiter.warp == warp && waiter.barrier == barrier)
                {
                    waiter.units = checked_u32_counter_add("fill waiter unit", waiter.units, 1)?;
                } else {
                    waiters.push(new_waiter);
                }
            }
        }
        Ok(())
    }

    fn try_for_each(
        &self,
        mut visit: impl FnMut(FillWaiter) -> Result<(), IssueError>,
    ) -> Result<(), IssueError> {
        match self {
            Self::One(waiter) => visit(*waiter),
            Self::Many(waiters) => {
                for &waiter in waiters {
                    visit(waiter)?;
                }
                Ok(())
            }
        }
    }
}

fn accumulate_pending_removal(
    removals: &mut HashMap<(usize, DepBarrier), u32>,
    waiter: FillWaiter,
) -> Result<(), IssueError> {
    let remove = removals.entry((waiter.warp, waiter.barrier)).or_default();
    *remove = checked_u32_counter_add("response pending-remove budget", *remove, waiter.units)?;
    Ok(())
}

/// State that EXISTS only in the shared-L2 mode — carrying it inside
/// the [`MemModel`] variant makes a mispaired construction
/// unrepresentable instead of an `expect()` deep in the memory step.
struct SharedL2State {
    l1: L1Cache,
    response_owner: ResponseOwner,
    /// Exact response-bearing request tokens still expected from the system.
    outstanding_replies: HashSet<ReplyTo>,
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
    block_live_warps: Vec<u32>,
    instruction_limit: u64,
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
        // Validate the complete dynamic/static join and every actor-cardinality
        // conversion before constructing mutable replay state. A later block
        // admission must not discover an unknown PC or overflow after earlier
        // blocks have already committed instructions.
        let mut block_live_warps = Vec::with_capacity(blocks.len());
        let mut instruction_limit = 0u64;
        let mut arrival_limit = 0u64;
        for block in blocks {
            let live_warp_count = block
                .warps
                .iter()
                .filter(|warp| !warp.insts.is_empty())
                .count();
            let live_warps = u32::try_from(live_warp_count).map_err(|_| {
                IssueError::LiveWarpCountOutOfRange {
                    launch_index: block.launch_index,
                    warps: live_warp_count,
                }
            })?;
            block_live_warps.push(live_warps);
            arrival_limit = checked_u64_counter_add(
                "warp arrival stamp",
                arrival_limit,
                u64::from(live_warps),
            )?;
            for warp in &block.warps {
                let warp_instructions =
                    u64::try_from(warp.insts.len()).map_err(|_| IssueError::CounterOverflow {
                        counter: "replay instruction cardinality",
                    })?;
                instruction_limit = checked_u64_counter_add(
                    "replay instruction cardinality",
                    instruction_limit,
                    warp_instructions,
                )?;
                for inst in &warp.insts {
                    let slot = lookup(schedule, inst.pc, block.launch_index, warp.warp_slot)?;
                    slot.control
                        .validate()
                        .map_err(|source| IssueError::ControlDomain {
                            pc: inst.pc,
                            launch_index: block.launch_index,
                            warp_slot: warp.warp_slot,
                            source,
                        })?;
                    validate_sector_span(
                        inst.kind,
                        warp.sectors.len(),
                        inst.pc,
                        block.launch_index,
                        warp.warp_slot,
                    )?;
                }
            }
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
        if let SmMemMode::L1Const(cfg) | SmMemMode::L1Shared(cfg) = mode {
            params.validate_l1_timing(cfg)?;
        }
        let mem = match mode {
            SmMemMode::Flat => MemModel::Flat,
            SmMemMode::L1Const(cfg) => MemModel::L1Const(L1Cache::new(cfg)?),
            SmMemMode::L1Shared(cfg) => MemModel::L1Shared(SharedL2State {
                l1: L1Cache::new(cfg)?,
                response_owner: allocate_response_owner()?,
                outstanding_replies: HashSet::new(),
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
            block_live_warps,
            instruction_limit,
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
    pub fn outstanding_responses(&self) -> usize {
        match &self.mem {
            MemModel::L1Shared(st) => st.outstanding_replies.len(),
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
            let live_warps = self.block_live_warps[self.next_block];
            let Some(next_resident_warps) = self.resident_warps.checked_add(live_warps) else {
                // Any overflowing sum is above every u32 residency ceiling,
                // so this block cannot join a non-empty wave.
                if self.resident_warps > 0 {
                    break;
                }
                return Err(IssueError::CounterOverflow {
                    counter: "resident warp",
                });
            };
            if self.resident_warps > 0
                && next_resident_warps > self.params.max_resident_warps_per_sm
            {
                break;
            }

            let admitted_warps =
                usize::try_from(live_warps).map_err(|_| IssueError::CounterOverflow {
                    counter: "admitted warp cardinality",
                })?;
            let next_unfinished_warps = checked_usize_counter_add(
                "unfinished warp",
                self.unfinished_warps,
                admitted_warps,
            )?;
            let next_arrival_stamp = checked_u64_counter_add(
                "warp arrival stamp",
                self.arrival_stamp,
                u64::from(live_warps),
            )?;
            let next_block = checked_usize_counter_add("next block", self.next_block, 1)?;

            // Build the complete block state off to the side. A schedule error
            // cannot leave a prefix of the block admitted.
            let block_ix = self.block_states.len();
            let first_warp = self.warps.len();
            let mut subcore_additions = [0usize; SUB_CORES];
            let mut pending_warps = Vec::with_capacity(admitted_warps);
            for w in &blk.warps {
                if w.insts.is_empty() {
                    continue;
                }
                let subcore = (w.warp_slot.rem_euclid(SUB_CORES as i32)) as usize;
                let pos_in_subcore = checked_usize_counter_add(
                    "sub-core resident warp",
                    self.subcore_warps[subcore].len(),
                    subcore_additions[subcore],
                )?;
                let arrival_offset = u64::try_from(pending_warps.len()).map_err(|_| {
                    IssueError::CounterOverflow {
                        counter: "warp arrival stamp",
                    }
                })?;
                let arrival = checked_u64_counter_add(
                    "warp arrival stamp",
                    self.arrival_stamp,
                    arrival_offset,
                )?;
                pending_warps.push((
                    subcore,
                    WarpState {
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
                        pos_in_subcore,
                        arrival,
                        at_block_barrier: false,
                        done: false,
                    },
                ));
                subcore_additions[subcore] = checked_usize_counter_add(
                    "sub-core resident warp",
                    subcore_additions[subcore],
                    1,
                )?;
            }

            for (subcore, warp) in pending_warps {
                let ix = self.warps.len();
                self.subcore_warps[subcore].push(ix);
                self.warps.push(warp);
            }
            self.block_states.push(BlockState {
                unfinished: admitted_warps,
                waiting: 0,
                warp_count: live_warps,
                first_warp,
                admitted_warps,
            });
            self.resident_warps = next_resident_warps;
            self.unfinished_warps = next_unfinished_warps;
            self.arrival_stamp = next_arrival_stamp;
            self.next_block = next_block;
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
        instruction: usize,
        kind: AccessKind,
        sectors: &'a [i64],
        write_barrier: Option<DepBarrier>,
        out: &mut Vec<MemRequest>,
    ) -> Result<Option<u64>, IssueError> {
        let t = self.t;
        let p = self.params;
        let MemModel::L1Shared(st) = &mut self.mem else {
            unreachable!("mem_step_shared is only dispatched in shared mode")
        };
        let warps = &mut self.warps;
        match kind {
            AccessKind::NonMemory => {
                if write_barrier.is_some() {
                    p.non_memory_ready(t).map(Some)
                } else {
                    Ok(None)
                }
            }
            AccessKind::Shared => {
                if write_barrier.is_some() {
                    p.shared_ready(t).map(Some)
                } else {
                    Ok(None)
                }
            }
            AccessKind::GlobalEligible { off, len, .. } => {
                // The hit-pipeline floor applies to the whole dependent load
                // (as in L1Const); misses ride their responses. Line fills
                // emitted anywhere in THIS emission batch (st.line_folds,
                // cleared at advance entry) fold later same-line sectors
                // into their request mask — one L2 transaction per line
                // per batch, priced on its worst sector.
                let mut known = write_barrier
                    .is_some()
                    .then(|| checked_cycle_add("shared L1 hit-pipeline floor", t, p.l1_hit_latency))
                    .transpose()?;
                let range = off..off + usize::from(len);
                for i in range.clone() {
                    match access_shared(
                        st,
                        warps,
                        wix,
                        write_barrier,
                        sectors[i],
                        ResponseId::new(st.response_owner, wix, instruction, i),
                        t,
                        out,
                    ) {
                        Ok(Some(ready)) => {
                            if let Some(current) = &mut known {
                                *current = (*current).max(ready);
                            }
                        }
                        Ok(None) => {}
                        Err(SharedAccessError::Stalled) => {
                            // MSHR full: park the remainder; the entry
                            // holds one pending unit until drained.
                            warps[wix].add_pending(wix, write_barrier, 1)?;
                            st.retry.push_back(Retry {
                                warp: wix,
                                instruction,
                                barrier: write_barrier,
                                lo: i,
                                hi: range.end,
                            });
                            break;
                        }
                        Err(SharedAccessError::Issue(source)) => return Err(source),
                    }
                }
                Ok(known)
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
                        .count_bypass(u64::from(len), class == L2OpClass::WriteAtomic)?;
                } else {
                    st.l1.count_untagged(u64::from(len))?;
                }
                if len == 0 {
                    // Degenerate record (fully predicated-off lanes are
                    // legal on the wire): no sectors, no requests. Only a
                    // present write barrier consumes the L2-hit visibility.
                    return if write_barrier.is_some() {
                        p.store_visibility_ready(t).map(Some)
                    } else {
                        Ok(None)
                    };
                }
                for i in off..off + usize::from(len) {
                    let sector_id = sectors[i];
                    let reply = if let Some(barrier) = write_barrier {
                        let reply = ReplyTo::Warp {
                            warp: wix,
                            barrier,
                            response_id: ResponseId::new(st.response_owner, wix, instruction, i),
                        };
                        warps[wix].add_pending(wix, Some(barrier), 1)?;
                        register_response(st, reply)?;
                        reply
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
                Ok(None)
            }
            AccessKind::GlobalStore { off, len } => {
                for &sector_id in &sectors[off..off + usize::from(len)] {
                    st.l1.probe_store(sector_id, t)?;
                    out.push(MemRequest::store(t, sector_id));
                }
                // v0: a present store write barrier is priced at the L2-hit
                // arm (as in L1Const); stores do not wait on responses.
                if write_barrier.is_some() {
                    p.store_visibility_ready(t).map(Some)
                } else {
                    Ok(None)
                }
            }
        }
    }

    fn preflight_response_accounting(
        &self,
        st: &SharedL2State,
        reply: ReplyTo,
        now: u64,
    ) -> Result<(), IssueError> {
        let fill = match reply {
            ReplyTo::Warp { warp, barrier, .. } => {
                if warp >= self.warps.len() {
                    return Err(IssueError::ResponseWarpOutOfRange {
                        warp,
                        warps: self.warps.len(),
                    });
                }
                self.warps[warp].preflight_pending_remove(warp, barrier, 1)?;
                return Ok(());
            }
            ReplyTo::L1Fill { fill, .. } => {
                if !st.l1.is_fill_in_flight(fill) {
                    return Err(IssueError::ResponseFillNotInFlight { fill });
                }
                if let FillKey::Line(line) = fill
                    && !st.fill_masks.contains_key(&line)
                {
                    return Err(IssueError::FillMaskMissing { line });
                }
                fill
            }
            ReplyTo::None => return Err(IssueError::UnexpectedMutationOnlyResponse),
        };

        if st.retry.is_empty() {
            if let Some(waiters) = st.fill_waiters.get(&fill) {
                waiters.try_for_each(|waiter| {
                    if waiter.warp >= self.warps.len() {
                        return Err(IssueError::ResponseWarpOutOfRange {
                            warp: waiter.warp,
                            warps: self.warps.len(),
                        });
                    }
                    self.warps[waiter.warp].preflight_pending_remove(
                        waiter.warp,
                        waiter.barrier,
                        waiter.units,
                    )
                })?;
            }
            let budget = L1MutationBudget {
                recency_touches: 1,
                ..L1MutationBudget::default()
            };
            if st.l1.preflight_mutation_budget(budget).is_err() {
                let mut shadow = st.l1.clone();
                shadow.complete_fill(fill)?;
            }
            return Ok(());
        }

        let mut removals = HashMap::new();
        if let Some(waiters) = st.fill_waiters.get(&fill) {
            waiters.try_for_each(|waiter| accumulate_pending_removal(&mut removals, waiter))?;
        }

        let mut projected = HashMap::new();
        for (&(warp, barrier), &remove) in &removals {
            if warp >= self.warps.len() {
                return Err(IssueError::ResponseWarpOutOfRange {
                    warp,
                    warps: self.warps.len(),
                });
            }
            self.warps[warp].preflight_pending_remove(warp, barrier, remove)?;
            let current = self.warps[warp].barrier_pending[barrier.index()];
            projected.insert(
                (warp, barrier),
                current
                    .checked_sub(remove)
                    .ok_or(IssueError::PendingBarrierUnderflow {
                        warp,
                        barrier: barrier.wire(),
                        current,
                        remove,
                    })?,
            );
        }

        let mut budget = L1MutationBudget {
            recency_touches: 1,
            ..L1MutationBudget::default()
        };
        let mut pending_budget_fits = true;
        for retry in &st.retry {
            if retry.warp >= self.warps.len() {
                return Err(IssueError::ResponseWarpOutOfRange {
                    warp: retry.warp,
                    warps: self.warps.len(),
                });
            }
            let remaining = retry
                .hi
                .checked_sub(retry.lo)
                .ok_or(IssueError::CounterUnderflow {
                    counter: "retry sector cursor",
                })?;
            if retry.hi > self.warps[retry.warp].sectors.len() {
                return Err(IssueError::CounterOverflow {
                    counter: "retry sector cursor",
                });
            }
            let remaining_u64 =
                u64::try_from(remaining).map_err(|_| IssueError::CounterOverflow {
                    counter: "retry L1 budget",
                })?;
            budget.eligible_sectors = checked_u64_counter_add(
                "retry L1 eligible-sector budget",
                budget.eligible_sectors,
                remaining_u64,
            )?;
            budget.recency_touches = checked_u64_counter_add(
                "retry L1 recency budget",
                budget.recency_touches,
                remaining_u64,
            )?;
            budget.new_fills =
                checked_u64_counter_add("retry L1 fill budget", budget.new_fills, remaining_u64)?;
            for sector_index in retry.lo..retry.hi {
                let sector_id = self.warps[retry.warp].sectors[sector_index];
                preflight_line_fill_mask(st, sector_id)?;
                let response_id = ResponseId::new(
                    st.response_owner,
                    retry.warp,
                    retry.instruction,
                    sector_index,
                );
                if let Some(&reply) = st
                    .outstanding_replies
                    .iter()
                    .find(|reply| reply.response_id() == Some(response_id))
                {
                    return Err(IssueError::ResponseTokenAlreadyOutstanding { reply });
                }
            }
            if let Some(barrier) = retry.barrier {
                let base = self.warps[retry.warp].barrier_pending[barrier.index()];
                let current = projected.entry((retry.warp, barrier)).or_insert(base);
                let before_remove = *current;
                let after_remove =
                    before_remove
                        .checked_sub(1)
                        .ok_or(IssueError::PendingBarrierUnderflow {
                            warp: retry.warp,
                            barrier: barrier.wire(),
                            current: before_remove,
                            remove: 1,
                        })?;
                let additions =
                    u32::try_from(remaining).map_err(|_| IssueError::CounterOverflow {
                        counter: "retry pending-add budget",
                    })?;
                if let Some(after_add) = after_remove.checked_add(additions) {
                    *current = after_add;
                } else {
                    pending_budget_fits = false;
                    *current = after_remove;
                }
            }
        }

        if st.l1.preflight_mutation_budget(budget).is_ok() && pending_budget_fits {
            return Ok(());
        }

        // Numeric-boundary slow path: complete and retry on a shadow cache so
        // warm hits and an atomic retry-unit replacement are charged exactly.
        let mut shadow = st.l1.clone();
        shadow.complete_fill(fill)?;
        let mut exact = HashMap::new();
        for (&(warp, barrier), &remove) in &removals {
            let current = self.warps[warp].barrier_pending[barrier.index()];
            exact.insert(
                (warp, barrier),
                current
                    .checked_sub(remove)
                    .ok_or(IssueError::PendingBarrierUnderflow {
                        warp,
                        barrier: barrier.wire(),
                        current,
                        remove,
                    })?,
            );
        }
        'retries: for retry in &st.retry {
            if let Some(barrier) = retry.barrier {
                let base = self.warps[retry.warp].barrier_pending[barrier.index()];
                let current = exact.entry((retry.warp, barrier)).or_insert(base);
                let before = *current;
                *current = before
                    .checked_sub(1)
                    .ok_or(IssueError::PendingBarrierUnderflow {
                        warp: retry.warp,
                        barrier: barrier.wire(),
                        current: before,
                        remove: 1,
                    })?;
            }
            for sector_index in retry.lo..retry.hi {
                let sector_id = self.warps[retry.warp].sectors[sector_index];
                match access_shared_l1(&mut shadow, sector_id, now, retry.barrier.is_some())? {
                    SharedL1Access::Hit { .. } => {}
                    SharedL1Access::MissRequest { .. } | SharedL1Access::MergedPending { .. } => {
                        if let Some(barrier) = retry.barrier {
                            let current = exact
                                .entry((retry.warp, barrier))
                                .or_insert(self.warps[retry.warp].barrier_pending[barrier.index()]);
                            let before = *current;
                            *current = before.checked_add(1).ok_or(
                                IssueError::PendingBarrierOverflow {
                                    warp: retry.warp,
                                    barrier: barrier.wire(),
                                    current: before,
                                    add: 1,
                                },
                            )?;
                        }
                    }
                    SharedL1Access::Stalled => {
                        if let Some(barrier) = retry.barrier {
                            let current = exact
                                .entry((retry.warp, barrier))
                                .or_insert(self.warps[retry.warp].barrier_pending[barrier.index()]);
                            let before = *current;
                            *current = before.checked_add(1).ok_or(
                                IssueError::PendingBarrierOverflow {
                                    warp: retry.warp,
                                    barrier: barrier.wire(),
                                    current: before,
                                    add: 1,
                                },
                            )?;
                        }
                        break 'retries;
                    }
                }
            }
        }
        Ok(())
    }

    fn preflight_retry_timing(
        &self,
        st: &SharedL2State,
        fill: FillKey,
        now: u64,
    ) -> Result<(), IssueError> {
        if st.retry.is_empty() {
            return Ok(());
        }
        let observes_hit_ready = st.retry.iter().any(|retry| retry.barrier.is_some());
        let hit_timing_fits =
            !observes_hit_ready || now.checked_add(self.params.l1_hit_latency).is_some();
        let response_timing_fits = now.checked_add(self.params.l1_miss_latency).is_some();
        if hit_timing_fits && response_timing_fits {
            return Ok(());
        }

        // At the cycle-domain boundary, replay the retry queue on a shadow L1.
        // A hit needs a timestamp only when its retry carries a write barrier;
        // only a fresh miss emits a new self-affecting response. Merges and
        // stalls consume neither timing arm.
        let mut shadow = st.l1.clone();
        shadow.complete_fill(fill)?;
        'retries: for retry in &st.retry {
            for sector_index in retry.lo..retry.hi {
                let sector_id = self.warps[retry.warp].sectors[sector_index];
                match access_shared_l1(&mut shadow, sector_id, now, false)? {
                    SharedL1Access::Hit { .. } => {
                        if retry.barrier.is_some() {
                            checked_cycle_add(
                                "retry L1 hit visibility",
                                now,
                                self.params.l1_hit_latency,
                            )?;
                        }
                    }
                    SharedL1Access::MissRequest { .. } => {
                        checked_cycle_add(
                            "retry self-response horizon",
                            now,
                            self.params.l1_miss_latency,
                        )?;
                    }
                    SharedL1Access::MergedPending { .. } => {}
                    SharedL1Access::Stalled => break 'retries,
                }
            }
        }
        Ok(())
    }

    pub(crate) fn preflight_response(&self, reply: ReplyTo, now: u64) -> Result<(), IssueError> {
        if now < self.t {
            return Err(IssueError::ResponseBeforeClock { now, clock: self.t });
        }
        let st = match &self.mem {
            MemModel::L1Shared(st) => st,
            MemModel::Flat => {
                return Err(IssueError::ResponseModeMismatch { mode: "flat" });
            }
            MemModel::L1Const(_) => {
                return Err(IssueError::ResponseModeMismatch {
                    mode: "constant-L1",
                });
            }
        };
        match reply {
            ReplyTo::Warp { warp, .. } if warp >= self.warps.len() => {
                return Err(IssueError::ResponseWarpOutOfRange {
                    warp,
                    warps: self.warps.len(),
                });
            }
            ReplyTo::None => return Err(IssueError::UnexpectedMutationOnlyResponse),
            _ => {}
        }
        if !st.outstanding_replies.contains(&reply) {
            return Err(IssueError::ResponseTokenNotOutstanding { reply });
        }
        if let ReplyTo::Warp { warp, barrier, .. } = reply
            && self.warps[warp].barrier_pending[barrier.index()] == 0
        {
            return Err(IssueError::ResponseBarrierNotPending {
                warp,
                barrier: barrier.wire(),
            });
        }
        self.preflight_response_accounting(st, reply, now)?;
        // A fill completion can drain parked accesses. Validate only the timing
        // branch each exact retry outcome consumes; next-level L2/DRAM pricing
        // is preflighted by the system against the actual L2 outcome.
        if let ReplyTo::L1Fill { fill, .. } = reply {
            self.preflight_retry_timing(st, fill, now)?;
        }
        Ok(())
    }

    /// Apply one memory response at its event time, then drain what it
    /// unblocked (fill waiters, MSHR retry queue — retries may emit new
    /// requests into `out`).
    pub fn on_response(
        &mut self,
        reply: ReplyTo,
        now: u64,
        out: &mut Vec<MemRequest>,
    ) -> Result<(), IssueError> {
        self.preflight_response(reply, now)?;
        let st = match &mut self.mem {
            MemModel::L1Shared(st) => st,
            MemModel::Flat => {
                return Err(IssueError::ResponseModeMismatch { mode: "flat" });
            }
            MemModel::L1Const(_) => {
                return Err(IssueError::ResponseModeMismatch {
                    mode: "constant-L1",
                });
            }
        };
        if !st.outstanding_replies.remove(&reply) {
            return Err(IssueError::ResponseTokenRemovalFailed { reply });
        }
        // New emission batch (the retry drain below may emit).
        st.line_folds.clear();
        let warps = &mut self.warps;
        match reply {
            ReplyTo::L1Fill { fill, .. } => {
                if let FillKey::Line(line) = fill
                    && st.fill_masks.remove(&line).is_none()
                {
                    return Err(IssueError::FillMaskMissing { line });
                }
                st.l1.complete_fill(fill)?;
                if let Some(waiters) = st.fill_waiters.remove(&fill) {
                    waiters.try_for_each(|waiter| {
                        warps[waiter.warp].clear_pending(
                            waiter.warp,
                            waiter.barrier,
                            waiter.units,
                            now,
                        )
                    })?;
                }
                // The completed fill freed an MSHR entry: retry parked
                // accesses (FIFO) until one stalls again.
                drain_retries(st, warps, now, out)?;
            }
            ReplyTo::Warp { warp, barrier, .. } => {
                warps[warp].clear_pending(warp, barrier, 1, now)?;
            }
            ReplyTo::None => return Err(IssueError::UnexpectedMutationOnlyResponse),
        }
        Ok(())
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
        horizon: ReplayHorizon,
        out: &mut Vec<MemRequest>,
    ) -> Result<Advance, IssueError> {
        if let Some(bound) = horizon.bounded_time()
            && bound < self.t
        {
            return Err(IssueError::HorizonBeforeClock {
                horizon: bound,
                clock: self.t,
            });
        }
        // New emission batch: line-fill folds are only valid within one
        // call's `out` (the caller schedules it after we return).
        if let MemModel::L1Shared(st) = &mut self.mem {
            st.line_folds.clear();
        }
        let mut horizon = horizon;
        while self.unfinished_warps > 0 {
            if horizon.reached(self.t) {
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
                let lower_bound = checked_cycle_add(
                    "self-response horizon",
                    self.t,
                    self.params.l1_miss_latency,
                )?;
                horizon = horizon.tighten(lower_bound);
            }
            if issued_this_cycle {
                self.t = checked_cycle_add("issue-clock advance", self.t, 1)?;
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
                    let next_cycle = checked_cycle_add("idle-clock advance", self.t, 1)?;
                    let target = nt.max(next_cycle);
                    if let Some(bound) = horizon.bounded_time()
                        && target >= bound
                    {
                        // Park AT the horizon instead of jumping past it:
                        // the event there may ready a warp before the
                        // next known timer.
                        return Ok(Advance::Idle {
                            wake_at: Some(bound),
                        });
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

    fn preflight_issue_timing(
        &self,
        kind: AccessKind,
        write_barrier: Option<DepBarrier>,
        read_barrier: Option<DepBarrier>,
        stall: u8,
        yield_flag: bool,
        retires_warp: bool,
    ) -> Result<(), IssueError> {
        let t = self.t;
        checked_cycle_add("issue-clock advance", t, 1)?;
        if !retires_warp {
            checked_cycle_add("warp stall eligibility", t, u64::from(stall.max(1)))?;
            if yield_flag {
                checked_cycle_add("warp yield eligibility", t, 2)?;
            }
        }
        if read_barrier.is_some() {
            checked_cycle_add(
                "read-dependence barrier visibility",
                t,
                self.params.read_barrier_latency,
            )?;
        }

        let needs_ready = write_barrier.is_some();
        match kind {
            AccessKind::NonMemory if needs_ready => {
                self.params.non_memory_ready(t)?;
            }
            AccessKind::Shared if needs_ready => {
                self.params.shared_ready(t)?;
            }
            AccessKind::NonMemory | AccessKind::Shared => {}
            AccessKind::GlobalEligible { .. } => match &self.mem {
                MemModel::Flat if needs_ready => {
                    checked_cycle_add(
                        "flat global visibility",
                        t,
                        self.params.global_flat_latency,
                    )?;
                }
                MemModel::L1Const(_) if needs_ready => {
                    checked_cycle_add("L1 hit-pipeline floor", t, self.params.l1_hit_latency)?;
                }
                MemModel::L1Shared(_) if needs_ready => {
                    checked_cycle_add(
                        "shared L1 hit-pipeline floor",
                        t,
                        self.params.l1_hit_latency,
                    )?;
                }
                MemModel::Flat | MemModel::L1Const(_) | MemModel::L1Shared(_) => {}
            },
            AccessKind::GlobalBypass { len, .. } | AccessKind::GlobalUntagged { len, .. } => {
                match &self.mem {
                    MemModel::Flat if needs_ready => {
                        checked_cycle_add(
                            "flat global visibility",
                            t,
                            self.params.global_flat_latency,
                        )?;
                    }
                    MemModel::L1Const(_) if needs_ready => {
                        self.params.store_visibility_ready(t)?;
                    }
                    MemModel::L1Shared(_) if len == 0 && needs_ready => {
                        self.params.store_visibility_ready(t)?;
                    }
                    MemModel::L1Shared(_) if len > 0 && needs_ready => {
                        checked_cycle_add("self-response horizon", t, self.params.l1_miss_latency)?;
                    }
                    MemModel::Flat | MemModel::L1Const(_) | MemModel::L1Shared(_) => {}
                }
            }
            AccessKind::GlobalStore { .. } => match &self.mem {
                MemModel::Flat if needs_ready => {
                    checked_cycle_add(
                        "flat global visibility",
                        t,
                        self.params.global_flat_latency,
                    )?;
                }
                MemModel::L1Const(_) | MemModel::L1Shared(_) if needs_ready => {
                    self.params.store_visibility_ready(t)?;
                }
                MemModel::Flat | MemModel::L1Const(_) | MemModel::L1Shared(_) => {}
            },
        }
        Ok(())
    }

    fn preflight_issue_memory(&self, picks: &[Option<usize>; SUB_CORES]) -> Result<(), IssueError> {
        let (l1, shared) = match &self.mem {
            MemModel::Flat => return Ok(()),
            MemModel::L1Const(l1) => (l1, None),
            MemModel::L1Shared(st) => (&st.l1, Some(st)),
        };
        let mut budget = L1MutationBudget::default();
        let mut pending_budget_fits = true;
        for &wix in picks.iter().flatten() {
            let warp = &self.warps[wix];
            let inst = warp
                .insts
                .get(warp.next)
                .ok_or(IssueError::CounterOverflow {
                    counter: "warp instruction cursor",
                })?;
            let slot = warp
                .next_slot
                .ok_or(IssueError::MissingIssueSlot { warp: wix })?;
            let add_budget = |counter: &'static str, value: &mut u64, add: u64| {
                *value = checked_u64_counter_add(counter, *value, add)?;
                Ok::<(), IssueError>(())
            };
            match inst.kind {
                AccessKind::GlobalEligible { off, len, .. } => {
                    let sectors = u64::from(len);
                    add_budget(
                        "L1 eligible-sector budget",
                        &mut budget.eligible_sectors,
                        sectors,
                    )?;
                    add_budget("L1 recency budget", &mut budget.recency_touches, sectors)?;
                    add_budget("L1 fill budget", &mut budget.new_fills, sectors)?;
                    if shared.is_some()
                        && warp
                            .preflight_pending_add(wix, slot.control.write_barrier, u32::from(len))
                            .is_err()
                    {
                        pending_budget_fits = false;
                    }
                    if let Some(st) = shared {
                        let end = off.checked_add(usize::from(len)).ok_or(
                            IssueError::CounterOverflow {
                                counter: "issue sector cursor",
                            },
                        )?;
                        for sector_index in off..end {
                            let sector_id = *warp.sectors.get(sector_index).ok_or(
                                IssueError::CounterOverflow {
                                    counter: "issue sector cursor",
                                },
                            )?;
                            preflight_line_fill_mask(st, sector_id)?;
                            let response_id =
                                ResponseId::new(st.response_owner, wix, warp.next, sector_index);
                            if let Some(&reply) = st
                                .outstanding_replies
                                .iter()
                                .find(|reply| reply.response_id() == Some(response_id))
                            {
                                return Err(IssueError::ResponseTokenAlreadyOutstanding { reply });
                            }
                        }
                    }
                }
                AccessKind::GlobalBypass {
                    len, token, off, ..
                } => {
                    let sectors = u64::from(len);
                    add_budget(
                        "L1 bypass-sector budget",
                        &mut budget.bypass_sectors,
                        sectors,
                    )?;
                    if token == CacheOpToken::Atom {
                        add_budget(
                            "L1 bypass-atom budget",
                            &mut budget.bypass_atom_sectors,
                            sectors,
                        )?;
                    }
                    if let Some(st) = shared
                        && slot.control.write_barrier.is_some()
                    {
                        if warp
                            .preflight_pending_add(wix, slot.control.write_barrier, u32::from(len))
                            .is_err()
                        {
                            pending_budget_fits = false;
                        }
                        for sector_index in off..off + usize::from(len) {
                            let response_id =
                                ResponseId::new(st.response_owner, wix, warp.next, sector_index);
                            if let Some(&reply) = st
                                .outstanding_replies
                                .iter()
                                .find(|reply| reply.response_id() == Some(response_id))
                            {
                                return Err(IssueError::ResponseTokenAlreadyOutstanding { reply });
                            }
                        }
                    }
                }
                AccessKind::GlobalUntagged { off, len } => {
                    let sectors = u64::from(len);
                    add_budget(
                        "L1 untagged-sector budget",
                        &mut budget.untagged_sectors,
                        sectors,
                    )?;
                    if let Some(st) = shared
                        && slot.control.write_barrier.is_some()
                    {
                        if warp
                            .preflight_pending_add(wix, slot.control.write_barrier, u32::from(len))
                            .is_err()
                        {
                            pending_budget_fits = false;
                        }
                        for sector_index in off..off + usize::from(len) {
                            let response_id =
                                ResponseId::new(st.response_owner, wix, warp.next, sector_index);
                            if let Some(&reply) = st
                                .outstanding_replies
                                .iter()
                                .find(|reply| reply.response_id() == Some(response_id))
                            {
                                return Err(IssueError::ResponseTokenAlreadyOutstanding { reply });
                            }
                        }
                    }
                }
                AccessKind::GlobalStore { len, .. } => {
                    let sectors = u64::from(len);
                    add_budget("L1 store-sector budget", &mut budget.store_sectors, sectors)?;
                    add_budget("L1 recency budget", &mut budget.recency_touches, sectors)?;
                }
                AccessKind::NonMemory | AccessKind::Shared => {}
            }
        }

        let constant_timing_fits = shared.is_some()
            || l1.constant_batch_timing_has_headroom(self.t, budget.eligible_sectors);
        let shared_timing_fits = shared.is_none()
            || budget.eligible_sectors == 0
            || (self.t.checked_add(self.params.l1_hit_latency).is_some()
                && self.t.checked_add(self.params.l1_miss_latency).is_some());
        if l1.preflight_mutation_budget(budget).is_ok()
            && pending_budget_fits
            && constant_timing_fits
            && shared_timing_fits
        {
            return Ok(());
        }

        // Near a numeric boundary, replay this at-most-four-instruction batch
        // on a shadow cache. This avoids rejecting valid warm hits while keeping
        // the ordinary hot path allocation-free.
        let mut shadow = l1.clone();
        for &wix in picks.iter().flatten() {
            let warp = &self.warps[wix];
            let inst = warp.insts[warp.next];
            let slot = warp
                .next_slot
                .ok_or(IssueError::MissingIssueSlot { warp: wix })?;
            match inst.kind {
                AccessKind::GlobalEligible { off, len, .. } => {
                    let mut pending_add = 0u32;
                    for &sector_id in &warp.sectors[off..off + usize::from(len)] {
                        if shared.is_some() {
                            match access_shared_l1(
                                &mut shadow,
                                sector_id,
                                self.t,
                                slot.control.write_barrier.is_some(),
                            )? {
                                SharedL1Access::Hit { .. } => {}
                                SharedL1Access::MissRequest { .. } => {
                                    checked_cycle_add(
                                        "self-response horizon",
                                        self.t,
                                        self.params.l1_miss_latency,
                                    )?;
                                    pending_add = checked_u32_counter_add(
                                        "issue exact pending-add budget",
                                        pending_add,
                                        1,
                                    )?;
                                }
                                SharedL1Access::MergedPending { .. } => {
                                    pending_add = checked_u32_counter_add(
                                        "issue exact pending-add budget",
                                        pending_add,
                                        1,
                                    )?;
                                }
                                SharedL1Access::Stalled => {
                                    pending_add = checked_u32_counter_add(
                                        "issue exact pending-add budget",
                                        pending_add,
                                        1,
                                    )?;
                                    break;
                                }
                            }
                        } else if slot.control.write_barrier.is_some() {
                            shadow.access_eligible(sector_id, self.t)?;
                        } else {
                            shadow.access_eligible_unobserved(sector_id, self.t)?;
                        }
                    }
                    if shared.is_some() {
                        warp.preflight_pending_add(wix, slot.control.write_barrier, pending_add)?;
                    }
                }
                AccessKind::GlobalBypass { len, token, .. } => {
                    shadow.count_bypass(u64::from(len), token == CacheOpToken::Atom)?;
                    if shared.is_some() {
                        warp.preflight_pending_add(
                            wix,
                            slot.control.write_barrier,
                            u32::from(len),
                        )?;
                    }
                }
                AccessKind::GlobalUntagged { len, .. } => {
                    shadow.count_untagged(u64::from(len))?;
                    if shared.is_some() {
                        warp.preflight_pending_add(
                            wix,
                            slot.control.write_barrier,
                            u32::from(len),
                        )?;
                    }
                }
                AccessKind::GlobalStore { off, len } => {
                    for &sector_id in &warp.sectors[off..off + usize::from(len)] {
                        shadow.probe_store(sector_id, self.t)?;
                    }
                }
                AccessKind::NonMemory | AccessKind::Shared => {}
            }
        }
        Ok(())
    }

    fn preflight_issue_accounting(
        &self,
        picks: &[Option<usize>; SUB_CORES],
    ) -> Result<u64, IssueError> {
        let issue_add = u64::try_from(picks.iter().flatten().count()).map_err(|_| {
            IssueError::CounterOverflow {
                counter: "issued instruction",
            }
        })?;
        let next_issued = checked_u64_counter_add("issued instruction", self.issued, issue_add)?;
        if next_issued > self.instruction_limit {
            return Err(IssueError::InstructionCountExceedsReplay {
                issued: self.issued,
                add: issue_add,
                limit: self.instruction_limit,
            });
        }

        let mut block_deltas: [Option<(usize, usize, usize)>; SUB_CORES] = [None; SUB_CORES];
        let mut total_retirements = 0usize;
        for &wix in picks.iter().flatten() {
            let warp = &self.warps[wix];
            let _inst = warp
                .insts
                .get(warp.next)
                .ok_or(IssueError::CounterOverflow {
                    counter: "warp instruction cursor",
                })?;
            let retires = warp.next_issue_retires()?;
            let slot = warp
                .next_slot
                .ok_or(IssueError::MissingIssueSlot { warp: wix })?;
            let entry_index = block_deltas
                .iter()
                .position(|entry| entry.is_some_and(|(block, _, _)| block == warp.block))
                .or_else(|| block_deltas.iter().position(Option::is_none))
                .ok_or(IssueError::CounterOverflow {
                    counter: "per-cycle block accounting",
                })?;
            let entry = block_deltas[entry_index].get_or_insert((warp.block, 0, 0));
            if retires {
                entry.1 = checked_usize_counter_add("block retirement", entry.1, 1)?;
                total_retirements =
                    checked_usize_counter_add("unfinished warp retirement", total_retirements, 1)?;
            } else if slot.op_class == OpClass::BlockBarrier {
                entry.2 = checked_usize_counter_add("block barrier waiter", entry.2, 1)?;
            }
        }

        self.unfinished_warps.checked_sub(total_retirements).ok_or(
            IssueError::CounterUnderflow {
                counter: "unfinished warp",
            },
        )?;
        let mut residency_released = 0u32;
        for (block, retirements, arrivals) in block_deltas.into_iter().flatten() {
            let state = &self.block_states[block];
            let unfinished =
                state
                    .unfinished
                    .checked_sub(retirements)
                    .ok_or(IssueError::CounterUnderflow {
                        counter: "block unfinished warp",
                    })?;
            let waiting =
                checked_usize_counter_add("block barrier waiter", state.waiting, arrivals)?;
            if waiting > unfinished {
                return Err(IssueError::CounterOverflow {
                    counter: "block barrier waiter",
                });
            }
            if unfinished == 0 {
                residency_released = checked_u32_counter_add(
                    "resident warp retirement",
                    residency_released,
                    state.warp_count,
                )?;
            }
        }
        self.resident_warps.checked_sub(residency_released).ok_or(
            IssueError::CounterUnderflow {
                counter: "resident warp",
            },
        )?;
        Ok(next_issued)
    }

    /// One issue cycle at `self.t` across the four sub-cores. Returns
    /// whether anything issued.
    fn issue_cycle(&mut self, out: &mut Vec<MemRequest>) -> Result<bool, IssueError> {
        let t = self.t;
        let picks: [Option<usize>; SUB_CORES] = std::array::from_fn(|sc| {
            // CGGTY pick: held warp while ready, else youngest ready.
            match self.hold[sc] {
                Some(w) if self.warps[w].is_ready(t) => Some(w),
                _ => self.subcore_warps[sc]
                    .iter()
                    .copied()
                    .filter(|&w| self.warps[w].is_ready(t))
                    .max_by_key(|&w| self.warps[w].arrival),
            }
        });

        // A cycle is the caller-visible transaction boundary. Validate every
        // selected instruction's possible timing arms before any sub-core
        // mutates counters, cache state, barriers, requests, or retirement.
        for wix in picks.iter().flatten().copied() {
            let w = &self.warps[wix];
            let slot = w
                .next_slot
                .ok_or(IssueError::MissingIssueSlot { warp: wix })?;
            let retires_warp = w.next_issue_retires()?;
            self.preflight_issue_timing(
                w.insts[w.next].kind,
                slot.control.write_barrier,
                slot.control.read_barrier,
                slot.control.stall,
                slot.control.yield_flag,
                retires_warp,
            )?;
        }
        self.preflight_issue_memory(&picks)?;
        let next_issued = self.preflight_issue_accounting(&picks)?;

        let mut issued_this_cycle = false;
        for (sc, pick) in picks.into_iter().enumerate() {
            let Some(wix) = pick else { continue };
            issued_this_cycle = true;

            // Issue exactly one instruction from warp `wix`.
            let (
                blk_ix,
                launch_index,
                warp_slot,
                slot_class,
                control,
                instruction,
                inst,
                warp_sectors,
                retires_warp,
            ) = {
                let w = &self.warps[wix];
                let slot = w
                    .next_slot
                    .ok_or(IssueError::MissingIssueSlot { warp: wix })?;
                (
                    w.block,
                    w.launch_index,
                    w.warp_slot,
                    slot.op_class,
                    slot.control,
                    w.next,
                    w.insts[w.next],
                    w.sectors,
                    w.next_issue_retires()?,
                )
            };

            // Memory-model interaction happens at issue time regardless of
            // whether a dependence barrier consumes the ready time.
            let data_ready = if matches!(self.mem, MemModel::L1Shared(_)) {
                self.mem_step_shared(
                    wix,
                    instruction,
                    inst.kind,
                    warp_sectors,
                    control.write_barrier,
                    out,
                )?
            } else {
                data_ready_time(
                    inst.kind,
                    warp_sectors,
                    t,
                    &self.params,
                    self.mem.const_l1_mut(),
                    control.write_barrier,
                )?
            };

            // Yield semantics per the CGGTY paper: the yield bit means
            // "do not issue this warp in the NEXT cycle" — a one-cycle
            // bubble when no other warp is ready — on top of the stall
            // count. A retiring instruction has no next eligibility to observe.
            if !retires_warp {
                let stall = u64::from(control.stall.max(1));
                let mut eligible = checked_cycle_add("warp stall eligibility", t, stall)?;
                if control.yield_flag {
                    eligible = eligible.max(checked_cycle_add("warp yield eligibility", t, 2)?);
                }
                self.warps[wix].eligible = eligible;
            }
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
            if let Some(read_barrier) = control.read_barrier {
                let read_clear = checked_cycle_add(
                    "read-dependence barrier visibility",
                    t,
                    self.params.read_barrier_latency,
                )?;
                self.warps[wix].stamp_clear(Some(read_barrier), read_clear);
            }

            // Advance the warp's stream.
            self.warps[wix].next =
                checked_usize_counter_add("warp instruction cursor", self.warps[wix].next, 1)?;
            if retires_warp {
                self.warps[wix].done = true;
                self.warps[wix].next_slot = None;
                self.unfinished_warps =
                    self.unfinished_warps
                        .checked_sub(1)
                        .ok_or(IssueError::CounterUnderflow {
                            counter: "unfinished warp",
                        })?;
                self.block_states[blk_ix].unfinished = self.block_states[blk_ix]
                    .unfinished
                    .checked_sub(1)
                    .ok_or(IssueError::CounterUnderflow {
                        counter: "block unfinished warp",
                    })?;
                // Retire from the sub-core pick list (swap-remove +
                // position fix-up) so pick/skip-ahead scans stay
                // bounded by RESIDENT warps, not warps-ever-admitted.
                let pos = self.warps[wix].pos_in_subcore;
                self.subcore_warps[sc].swap_remove(pos);
                if let Some(&moved) = self.subcore_warps[sc].get(pos) {
                    self.warps[moved].pos_in_subcore = pos;
                }
                if self.block_states[blk_ix].unfinished == 0 {
                    self.resident_warps = self
                        .resident_warps
                        .checked_sub(self.block_states[blk_ix].warp_count)
                        .ok_or(IssueError::CounterUnderflow {
                            counter: "resident warp",
                        })?;
                    let admit_at = checked_cycle_add("block-wave admission", t, 1)?;
                    self.admit_blocks(admit_at)?;
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
                self.block_states[blk_ix].waiting = checked_usize_counter_add(
                    "block barrier waiter",
                    self.block_states[blk_ix].waiting,
                    1,
                )?;
            }
            // Release check: every unfinished warp of the block waiting.
            // The block's warps are contiguous in `warps` (admission
            // pushes them in one run), so scan only that range.
            let bs = &self.block_states[blk_ix];
            if bs.waiting > 0 && bs.waiting == bs.unfinished {
                let (first, count) = (bs.first_warp, bs.admitted_warps);
                self.block_states[blk_ix].waiting = 0;
                let release_at = checked_cycle_add("block-barrier release", t, 1)?;
                for w in &mut self.warps[first..first + count] {
                    if w.at_block_barrier {
                        w.at_block_barrier = false;
                        w.eligible = w.eligible.max(release_at);
                    }
                }
            }
        }
        self.issued = next_issued;
        Ok(issued_this_cycle)
    }
}

/// Register a barrier unit that clears when `fill` completes; `None`
/// carries no dependency.
fn add_fill_waiter(
    st: &mut SharedL2State,
    warps: &mut [WarpState<'_>],
    fill: FillKey,
    warp: usize,
    barrier: Option<DepBarrier>,
) -> Result<(), IssueError> {
    let Some(barrier) = barrier else {
        return Ok(());
    };
    warps[warp].add_pending(warp, Some(barrier), 1)?;
    match st.fill_waiters.entry(fill) {
        Entry::Vacant(v) => {
            v.insert(Waiters::One(FillWaiter {
                warp,
                barrier,
                units: 1,
            }));
        }
        Entry::Occupied(mut o) => o.get_mut().add(warp, barrier)?,
    }
    Ok(())
}

fn preflight_line_fill_mask(st: &SharedL2State, sector_id: i64) -> Result<(), IssueError> {
    let line = sector_id >> 2;
    let fill_in_flight = st.l1.is_fill_in_flight(FillKey::Line(line));
    let has_mask = st.fill_masks.contains_key(&line);
    match (fill_in_flight, has_mask) {
        (true, false) => Err(IssueError::FillMaskMissing { line }),
        (false, true) => Err(IssueError::FillMaskWithoutFill { line }),
        _ => Ok(()),
    }
}

fn register_response(st: &mut SharedL2State, reply: ReplyTo) -> Result<(), IssueError> {
    if !st.outstanding_replies.insert(reply) {
        return Err(IssueError::ResponseTokenAlreadyOutstanding { reply });
    }
    Ok(())
}

enum SharedL1Access {
    Hit { ready: Option<u64> },
    MissRequest { fill: FillKey },
    MergedPending { fill: FillKey },
    Stalled,
}

fn access_shared_l1(
    l1: &mut L1Cache,
    sector_id: i64,
    now: u64,
    observe_ready: bool,
) -> Result<SharedL1Access, L1AccessError> {
    if observe_ready {
        return Ok(match l1.access_eligible_ev(sector_id, now)? {
            EvAccess::Hit { ready } => SharedL1Access::Hit { ready: Some(ready) },
            EvAccess::MissRequest { fill } => SharedL1Access::MissRequest { fill },
            EvAccess::MergedPending { fill } => SharedL1Access::MergedPending { fill },
            EvAccess::Stalled => SharedL1Access::Stalled,
        });
    }
    Ok(match l1.access_eligible_ev_unobserved(sector_id, now)? {
        EvAccessUnobserved::Hit => SharedL1Access::Hit { ready: None },
        EvAccessUnobserved::MissRequest { fill } => SharedL1Access::MissRequest { fill },
        EvAccessUnobserved::MergedPending { fill } => SharedL1Access::MergedPending { fill },
        EvAccessUnobserved::Stalled => SharedL1Access::Stalled,
    })
}

enum SharedAccessError {
    Stalled,
    Issue(IssueError),
}

impl From<IssueError> for SharedAccessError {
    fn from(source: IssueError) -> Self {
        Self::Issue(source)
    }
}

/// Emission surface of one access BATCH (one instruction's sector
/// One eligible sector access in L1Shared mode at time `now`, emitting a
/// request on a fresh miss. Returns `Ok(known_ready)` for hits,
/// `Ok(None)` when the data waits on a response, and
/// `Err(SharedAccessError::Stalled)` on an MSHR stall (caller parks the
/// remainder). Accounting failures remain typed.
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
fn access_shared(
    st: &mut SharedL2State,
    warps: &mut [WarpState<'_>],
    wix: usize,
    barrier: Option<DepBarrier>,
    sector_id: i64,
    response_id: ResponseId,
    now: u64,
    out: &mut Vec<MemRequest>,
) -> Result<Option<u64>, SharedAccessError> {
    let sbit = 1u8 << (sector_id & 3) as u8;
    match access_shared_l1(&mut st.l1, sector_id, now, barrier.is_some())
        .map_err(IssueError::from)?
    {
        SharedL1Access::Hit { ready } => Ok(ready),
        SharedL1Access::MissRequest { fill } => {
            let request = MemRequest::l1_fill(now, sector_id, fill, response_id);
            register_response(st, request.reply)?;
            out.push(request);
            if let FillKey::Line(line) = fill {
                st.fill_masks.insert(line, sbit);
                st.line_folds.push((line, out.len() - 1));
            }
            add_fill_waiter(st, warps, fill, wix, barrier)?;
            Ok(None)
        }
        SharedL1Access::MergedPending { fill } => {
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
                    .ok_or(SharedAccessError::Issue(IssueError::FillMaskMissing {
                        line,
                    }))?;
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
            add_fill_waiter(st, warps, fill, wix, barrier)?;
            Ok(None)
        }
        SharedL1Access::Stalled => Err(SharedAccessError::Stalled),
    }
}

fn drain_retries(
    st: &mut SharedL2State,
    warps: &mut [WarpState<'_>],
    now: u64,
    out: &mut Vec<MemRequest>,
) -> Result<(), IssueError> {
    while let Some(mut r) = st.retry.pop_front() {
        let sectors = warps[r.warp].sectors;
        // Replace the retry entry's unit atomically with whatever waiter
        // units this attempt creates. No warp can observe pending state while
        // a response handler runs; releasing first preserves u32::MAX as a
        // valid final count instead of requiring a nonexistent transient slot.
        warps[r.warp].release_pending(r.warp, r.barrier, 1)?;
        // The whole drain is ONE emission batch (st.line_folds, cleared
        // by on_response): a line fill emitted by an earlier retry
        // entry folds later entries' same-line sectors — same-timestamp
        // requests must not split one line transaction (gate finding).
        while r.lo < r.hi {
            match access_shared(
                st,
                warps,
                r.warp,
                r.barrier,
                sectors[r.lo],
                ResponseId::new(st.response_owner, r.warp, r.instruction, r.lo),
                now,
                out,
            ) {
                Ok(Some(ready)) => warps[r.warp].stamp_clear(r.barrier, ready),
                Ok(None) => {}
                Err(SharedAccessError::Stalled) => {
                    // Still full: restore the retry entry's replacement unit
                    // and park it; nothing behind it can proceed either.
                    warps[r.warp].add_pending(r.warp, r.barrier, 1)?;
                    st.retry.push_front(r);
                    return Ok(());
                }
                Err(SharedAccessError::Issue(source)) => return Err(source),
            }
            r.lo = r.lo.checked_add(1).ok_or(IssueError::CounterOverflow {
                counter: "retry sector cursor",
            })?;
        }
    }
    Ok(())
}

fn validate_sector_span(
    kind: AccessKind,
    sectors: usize,
    pc: u32,
    launch_index: i64,
    warp_slot: i32,
) -> Result<(), IssueError> {
    let (off, len) = match kind {
        AccessKind::GlobalEligible { off, len, .. }
        | AccessKind::GlobalBypass { off, len, .. }
        | AccessKind::GlobalUntagged { off, len }
        | AccessKind::GlobalStore { off, len } => (off, len),
        AccessKind::NonMemory | AccessKind::Shared => return Ok(()),
    };
    let end = off
        .checked_add(usize::from(len))
        .ok_or(IssueError::SectorSpanOverflow {
            pc,
            launch_index,
            warp_slot,
            off,
            len,
        })?;
    if end > sectors {
        return Err(IssueError::SectorSpanOutOfBounds {
            pc,
            launch_index,
            warp_slot,
            off,
            len,
            end,
            sectors,
        });
    }
    Ok(())
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
    match core.advance(ReplayHorizon::Unbounded, &mut out)? {
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

    use sim_mem::{AccessOutcome, CacheOpToken, L1CounterError, L1CounterField};

    use super::*;
    use crate::control_bits::{ControlBits, DepBarrier, WaitMask};
    use crate::program::{FunctionSchedule, IssueSlot, OpClass};

    fn cb(stall: u8) -> ControlBits {
        ControlBits::new(stall, false, 7, 7, 0, 0).expect("valid test controls")
    }

    fn barrier(value: u8) -> DepBarrier {
        DepBarrier::new(value).expect("valid test barrier")
    }

    fn wait_mask(bits: u8) -> WaitMask {
        WaitMask::new(bits).expect("valid test wait mask")
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

    #[test]
    fn maximum_response_owner_is_allocated_before_exhaustion() {
        let next = AtomicU64::new(u64::MAX);
        assert_eq!(
            allocate_response_owner_from(&next).unwrap(),
            ResponseOwner(u64::MAX)
        );
        assert_eq!(next.load(Ordering::Relaxed), 0);
        assert!(matches!(
            allocate_response_owner_from(&next),
            Err(IssueError::ResponseOwnerExhausted)
        ));
        assert_eq!(next.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn past_horizon_is_rejected_before_mutation() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0)])],
        }];
        let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
        core.t = 1;
        let mut out = Vec::new();
        assert!(matches!(
            core.advance(ReplayHorizon::Bounded(0), &mut out),
            Err(IssueError::HorizonBeforeClock {
                horizon: 0,
                clock: 1,
            })
        ));
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), 0);
        assert_eq!(core.warps[0].next, 0);
    }

    fn construction_error(kind: AccessKind, sectors: Vec<i64>) -> IssueError {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDG.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 7,
            warps: vec![WarpReplay {
                warp_slot: 3,
                insts: vec![ReplayInst { pc: 0, kind }],
                sectors,
            }],
        }];
        let parameters = params();
        match SmCore::new(&blocks, &sched, &parameters, SmMemMode::Flat) {
            Err(error) => error,
            Ok(_) => panic!("invalid sector span constructed replay state"),
        }
    }

    #[test]
    fn every_memory_sector_span_is_validated_before_core_construction() {
        let kinds = [
            AccessKind::GlobalEligible {
                off: 1,
                len: 1,
                token: CacheOpToken::LdgECa,
            },
            AccessKind::GlobalBypass {
                off: 1,
                len: 1,
                token: CacheOpToken::LdgECg,
            },
            AccessKind::GlobalUntagged { off: 1, len: 1 },
            AccessKind::GlobalStore { off: 1, len: 1 },
        ];
        for kind in kinds {
            assert!(matches!(
                construction_error(kind, vec![40]),
                IssueError::SectorSpanOutOfBounds {
                    pc: 0,
                    launch_index: 7,
                    warp_slot: 3,
                    off: 1,
                    len: 1,
                    end: 2,
                    sectors: 1,
                }
            ));
        }
    }

    #[test]
    fn sector_span_overflow_and_empty_out_of_bounds_are_typed() {
        assert!(matches!(
            construction_error(
                AccessKind::GlobalStore {
                    off: usize::MAX,
                    len: 1,
                },
                Vec::new(),
            ),
            IssueError::SectorSpanOverflow {
                pc: 0,
                launch_index: 7,
                warp_slot: 3,
                off: usize::MAX,
                len: 1,
            }
        ));
        assert!(matches!(
            construction_error(AccessKind::GlobalUntagged { off: 1, len: 0 }, Vec::new(),),
            IssueError::SectorSpanOutOfBounds {
                off: 1,
                len: 0,
                end: 1,
                sectors: 0,
                ..
            }
        ));
    }

    #[test]
    fn empty_sector_span_at_arena_end_is_valid() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LD.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: AccessKind::GlobalUntagged { off: 0, len: 0 },
                }],
                sectors: Vec::new(),
            }],
        }];
        SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat)
            .expect("empty span at the arena end is valid");
    }

    #[test]
    fn invalid_public_control_fields_fail_before_core_construction() {
        let blocks = vec![BlockReplay {
            launch_index: 7,
            warps: vec![warp(3, vec![inst(0)])],
        }];

        let mut invalid_stall = cb(1);
        invalid_stall.stall = 16;
        let stall_schedule = schedule(vec![(0, slot(invalid_stall, OpClass::Other, "IADD3"))]);
        let Err(stall_error) = SmCore::new(&blocks, &stall_schedule, &params(), SmMemMode::Flat)
        else {
            panic!("invalid stall constructed replay state");
        };
        assert!(matches!(
            stall_error,
            IssueError::ControlDomain {
                pc: 0,
                launch_index: 7,
                warp_slot: 3,
                source: ControlDomainError::StallOutOfRange { stall: 16 },
            }
        ));

        let mut invalid_reuse = cb(1);
        invalid_reuse.reuse = 16;
        let reuse_schedule = schedule(vec![(0, slot(invalid_reuse, OpClass::Other, "IADD3"))]);
        let Err(reuse_error) = SmCore::new(&blocks, &reuse_schedule, &params(), SmMemMode::Flat)
        else {
            panic!("invalid reuse mask constructed replay state");
        };
        assert!(matches!(
            reuse_error,
            IssueError::ControlDomain {
                pc: 0,
                launch_index: 7,
                warp_slot: 3,
                source: ControlDomainError::ReuseOutOfRange { reuse: 16 },
            }
        ));
    }

    #[test]
    fn producer_latency_overflow_is_typed() {
        let mut control = cb(1);
        control.write_barrier = Some(barrier(0));
        let sched = schedule(vec![(0, slot(control, OpClass::Other, "EXIT"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0)])],
        }];
        let mut p = params();
        p.non_memory_write_barrier_latency = u64::MAX;
        let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
        core.t = 1;
        let mut out = Vec::new();
        assert!(matches!(
            core.advance(ReplayHorizon::Unbounded, &mut out),
            Err(IssueError::TimeOverflow {
                operation: "non-memory write-barrier visibility",
                base: 1,
                delta: u64::MAX,
            })
        ));
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), 0);
        assert_eq!(core.warps[0].next, 0);
    }

    #[test]
    fn absent_dependency_barriers_skip_unused_visibility_timing() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDS"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(
                0,
                vec![ReplayInst {
                    pc: 0,
                    kind: AccessKind::Shared,
                }],
            )],
        }];
        let mut p = params();
        p.shared_write_barrier_latency = u64::MAX;
        p.read_barrier_latency = u64::MAX;
        let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
        core.t = u64::MAX - 1;
        let mut out = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Done
        );
        assert!(out.is_empty());
        assert_eq!(core.result().cycles, u64::MAX);
        assert_eq!(core.warps[0].barrier_clear, [0; DEP_BARRIERS]);
    }

    #[test]
    fn present_read_barrier_overflow_remains_typed_and_nonmutating() {
        let mut control = cb(1);
        control.read_barrier = Some(barrier(0));
        let sched = schedule(vec![(0, slot(control, OpClass::Other, "LDS"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(
                0,
                vec![ReplayInst {
                    pc: 0,
                    kind: AccessKind::Shared,
                }],
            )],
        }];
        let mut p = params();
        p.read_barrier_latency = 2;
        let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
        core.t = u64::MAX - 1;
        let mut out = Vec::new();

        assert!(matches!(
            core.advance(ReplayHorizon::Unbounded, &mut out),
            Err(IssueError::TimeOverflow {
                operation: "read-dependence barrier visibility",
                base,
                delta: 2,
            }) if base == u64::MAX - 1
        ));
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), 0);
        assert_eq!(core.warps[0].next, 0);
        assert_eq!(core.warps[0].barrier_clear, [0; DEP_BARRIERS]);
    }

    #[test]
    fn warm_l1_hit_ignores_an_overflowing_unused_miss_arm() {
        let mut control = cb(1);
        control.write_barrier = Some(barrier(0));
        let sched = schedule(vec![(0, slot(control, OpClass::Other, "LDG.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: eligible(0, 1),
                }],
                sectors: vec![0],
            }],
        }];
        let mut p = params();
        p.l1_hit_latency = 1;
        p.l1_miss_latency = 2;

        for shared in [false, true] {
            let mode = if shared {
                SmMemMode::L1Shared(p.l1_config(8, 1))
            } else {
                SmMemMode::L1Const(p.l1_config(8, 1))
            };
            let mut core = SmCore::new(&blocks, &sched, &p, mode).unwrap();
            match &mut core.mem {
                MemModel::L1Const(l1) => {
                    assert!(matches!(
                        l1.access_eligible(0, 0).unwrap(),
                        (AccessOutcome::MissLineAlloc, 2)
                    ));
                    assert!(matches!(
                        l1.access_eligible(0, 2).unwrap(),
                        (AccessOutcome::Hit, 3)
                    ));
                }
                MemModel::L1Shared(st) => {
                    let EvAccess::MissRequest { fill } = st.l1.access_eligible_ev(0, 0).unwrap()
                    else {
                        panic!("expected cold prefill miss");
                    };
                    st.l1.complete_fill(fill).unwrap();
                }
                MemModel::Flat => panic!("expected an L1 mode"),
            }
            core.t = u64::MAX - 1;
            let mut out = Vec::new();

            assert_eq!(
                core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
                Advance::Done
            );
            assert!(out.is_empty());
            assert_eq!(core.result().cycles, u64::MAX);
            assert_eq!(core.warps[0].barrier_clear[0], u64::MAX);
        }
    }

    #[test]
    fn unbarriered_warm_l1_hit_skips_unused_hit_visibility() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDG.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: eligible(0, 1),
                }],
                sectors: vec![0],
            }],
        }];
        let mut p = params();
        p.l1_hit_latency = 2;
        p.l1_miss_latency = 3;

        for shared in [false, true] {
            let mode = if shared {
                SmMemMode::L1Shared(p.l1_config(8, 1))
            } else {
                SmMemMode::L1Const(p.l1_config(8, 1))
            };
            let mut core = SmCore::new(&blocks, &sched, &p, mode).unwrap();
            match &mut core.mem {
                MemModel::L1Const(l1) => {
                    assert!(matches!(
                        l1.access_eligible(0, 0).unwrap(),
                        (AccessOutcome::MissLineAlloc, 3)
                    ));
                    assert!(matches!(
                        l1.access_eligible(0, 3).unwrap(),
                        (AccessOutcome::Hit, 5)
                    ));
                }
                MemModel::L1Shared(st) => {
                    let EvAccess::MissRequest { fill } = st.l1.access_eligible_ev(0, 0).unwrap()
                    else {
                        panic!("expected cold prefill miss");
                    };
                    st.l1.complete_fill(fill).unwrap();
                }
                MemModel::Flat => panic!("expected an L1 mode"),
            }
            core.t = u64::MAX - 1;
            let mut out = Vec::new();

            assert_eq!(
                core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
                Advance::Done
            );
            assert!(out.is_empty());
            assert_eq!(core.result().cycles, u64::MAX);
            assert_eq!(core.warps[0].barrier_clear, [0; DEP_BARRIERS]);
        }
    }

    #[test]
    fn cold_shared_l1_miss_overflow_remains_typed_and_nonmutating() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDG.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: eligible(0, 1),
                }],
                sectors: vec![0],
            }],
        }];
        let mut p = params();
        p.l1_hit_latency = 1;
        p.l1_miss_latency = 2;
        let mut core =
            SmCore::new(&blocks, &sched, &p, SmMemMode::L1Shared(p.l1_config(8, 1))).unwrap();
        core.t = u64::MAX - 1;
        let before = *core.l1_counters().unwrap();
        let mut out = Vec::new();

        assert!(matches!(
            core.advance(ReplayHorizon::Unbounded, &mut out),
            Err(IssueError::TimeOverflow {
                operation: "self-response horizon",
                base,
                delta: 2,
            }) if base == u64::MAX - 1
        ));
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), 0);
        assert_eq!(core.warps[0].next, 0);
        assert_eq!(core.outstanding_responses(), 0);
        assert_eq!(*core.l1_counters().unwrap(), before);
    }

    #[test]
    fn exact_maximum_barrier_time_is_valid() {
        let mut control = cb(1);
        control.write_barrier = Some(barrier(0));
        let sched = schedule(vec![(0, slot(control, OpClass::Other, "EXIT"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0)])],
        }];
        let mut p = params();
        p.non_memory_write_barrier_latency = u64::MAX - 1;
        let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
        core.t = 1;
        let mut out = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Done
        );
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), 1);
        assert_eq!(core.warps[0].barrier_clear[0], u64::MAX);
        assert_eq!(core.result().cycles, 2);
    }

    #[test]
    fn final_issue_can_advance_the_clock_to_exact_maximum() {
        for (stall, yield_flag) in [(15, false), (1, true)] {
            let mut control = cb(stall);
            control.yield_flag = yield_flag;
            let sched = schedule(vec![(0, slot(control, OpClass::Other, "EXIT"))]);
            let blocks = vec![BlockReplay {
                launch_index: 0,
                warps: vec![warp(0, vec![inst(0)])],
            }];
            let mut p = params();
            p.non_memory_write_barrier_latency = 1;
            p.read_barrier_latency = 1;
            let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
            core.t = u64::MAX - 1;
            let mut out = Vec::new();

            assert_eq!(
                core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
                Advance::Done
            );
            assert!(out.is_empty());
            assert_eq!(core.result().instructions_issued, 1);
            assert_eq!(core.result().cycles, u64::MAX);
        }
    }

    #[test]
    fn nonterminal_issue_still_requires_post_issue_eligibility() {
        for (stall, yield_flag, expected_operation, expected_delta) in [
            (15, false, "warp stall eligibility", 15),
            (1, true, "warp yield eligibility", 2),
        ] {
            let mut control = cb(stall);
            control.yield_flag = yield_flag;
            let sched = schedule(vec![
                (0, slot(control, OpClass::Other, "IADD3")),
                (16, slot(cb(1), OpClass::Other, "EXIT")),
            ]);
            let blocks = vec![BlockReplay {
                launch_index: 0,
                warps: vec![warp(0, vec![inst(0), inst(16)])],
            }];
            let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
            core.t = u64::MAX - 1;
            let mut out = Vec::new();

            assert!(matches!(
                core.advance(ReplayHorizon::Unbounded, &mut out),
                Err(IssueError::TimeOverflow {
                    operation,
                    base,
                    delta,
                }) if operation == expected_operation
                    && base == u64::MAX - 1
                    && delta == expected_delta
            ));
            assert!(out.is_empty());
            assert_eq!(core.instructions_issued(), 0);
            assert_eq!(core.warps[0].next, 0);
        }
    }

    #[test]
    fn maximum_bounded_horizon_is_distinct_from_unbounded_replay() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0)])],
        }];
        let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
        core.t = u64::MAX;
        let mut out = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Bounded(u64::MAX), &mut out)
                .unwrap(),
            Advance::Idle {
                wake_at: Some(u64::MAX),
            }
        );
        assert!(matches!(
            core.advance(ReplayHorizon::Unbounded, &mut out),
            Err(IssueError::TimeOverflow {
                operation: "issue-clock advance",
                base: u64::MAX,
                delta: 1,
            })
        ));
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), 0);
        assert_eq!(core.warps[0].next, 0);
    }

    #[test]
    fn self_response_horizon_overflow_is_typed() {
        let mut control = cb(1);
        control.write_barrier = Some(barrier(0));
        let sched = schedule(vec![(0, slot(control, OpClass::Other, "LDG.E.CG"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: AccessKind::GlobalBypass {
                        off: 0,
                        len: 1,
                        token: CacheOpToken::LdgECg,
                    },
                }],
                sectors: vec![0],
            }],
        }];
        let mut p = params();
        p.l1_miss_latency = u64::MAX;
        let mut core = SmCore::new(
            &blocks,
            &sched,
            &p,
            SmMemMode::L1Shared(p.l1_config(768, 32)),
        )
        .unwrap();
        core.t = 1;
        let mut out = Vec::new();
        assert!(matches!(
            core.advance(ReplayHorizon::Unbounded, &mut out),
            Err(IssueError::TimeOverflow {
                operation: "self-response horizon",
                base: 1,
                delta: u64::MAX,
            })
        ));
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), 0);
        assert_eq!(core.warps[0].next, 0);
        assert_eq!(core.outstanding_responses(), 0);
        assert_eq!(core.l1_counters().unwrap().bypass_sectors, 0);
    }

    #[test]
    fn l1_timing_must_match_issue_parameters() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0)])],
        }];
        let p = params();
        let mut cfg = p.l1_config(768, 32);
        cfg.miss_latency += 1;
        assert!(matches!(
            SmCore::new(&blocks, &sched, &p, SmMemMode::L1Shared(cfg)),
            Err(IssueError::L1TimingMismatch {
                field: "miss",
                expected: 244,
                got: 245,
            })
        ));
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

    fn eligible(off: usize, len: u16) -> AccessKind {
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
        load.write_barrier = Some(barrier(0));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000001);
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
        load.write_barrier = Some(barrier(0));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000001);
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
        slow.write_barrier = Some(barrier(0));
        let mut fast = cb(1);
        fast.write_barrier = Some(barrier(0));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000001);
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
    fn issue_count_overflow_rejects_the_whole_cycle_before_mutation() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0)])],
        }];
        let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
        core.issued = u64::MAX;
        let before_unfinished = core.unfinished_warps;
        let before_resident = core.resident_warps;
        let mut out = Vec::new();

        assert!(matches!(
            core.advance(ReplayHorizon::Unbounded, &mut out),
            Err(IssueError::CounterOverflow {
                counter: "issued instruction"
            })
        ));
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), u64::MAX);
        assert_eq!(core.warps[0].next, 0);
        assert!(!core.warps[0].done);
        assert_eq!(core.unfinished_warps, before_unfinished);
        assert_eq!(core.resident_warps, before_resident);
    }

    #[test]
    fn pending_overflow_rejects_issue_before_cache_or_request_mutation() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let sched = schedule(vec![(0, slot(load, OpClass::Other, "LDG.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: eligible(0, 1),
                }],
                sectors: vec![0],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        core.warps[0].barrier_pending[0] = u32::MAX;
        let before_l1 = *core.l1_counters().unwrap();
        let mut out = Vec::new();

        assert!(matches!(
            core.advance(ReplayHorizon::Unbounded, &mut out),
            Err(IssueError::PendingBarrierOverflow {
                warp: 0,
                barrier: 0,
                current: u32::MAX,
                add: 1,
            })
        ));
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), 0);
        assert_eq!(core.warps[0].next, 0);
        assert_eq!(core.warps[0].barrier_pending[0], u32::MAX);
        assert_eq!(core.outstanding_responses(), 0);
        assert_eq!(*core.l1_counters().unwrap(), before_l1);
    }

    #[test]
    fn warm_hit_accepts_maximum_pending_count_without_a_new_unit() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let sched = schedule(vec![(0, slot(load, OpClass::Other, "LDG.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: eligible(0, 1),
                }],
                sectors: vec![0],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let MemModel::L1Shared(st) = &mut core.mem else {
            panic!("expected shared mode");
        };
        let EvAccess::MissRequest { fill } = st.l1.access_eligible_ev(0, 0).unwrap() else {
            panic!("expected cold prefill miss");
        };
        st.l1.complete_fill(fill).unwrap();
        core.warps[0].barrier_pending[0] = u32::MAX;
        let before = *core.l1_counters().unwrap();
        let mut out = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Done
        );
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), 1);
        assert_eq!(core.warps[0].next, 1);
        assert_eq!(core.warps[0].barrier_pending[0], u32::MAX);
        let after = *core.l1_counters().unwrap();
        assert_eq!(after.hits, before.hits + 1);
        assert_eq!(after.misses, before.misses);
        assert_eq!(core.outstanding_responses(), 0);
    }

    #[test]
    fn multi_subcore_l1_overflow_rejects_the_whole_issue_cycle() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "STG.E"))]);
        let store = || ReplayInst {
            pc: 0,
            kind: AccessKind::GlobalStore { off: 0, len: 1 },
        };
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![
                WarpReplay {
                    warp_slot: 0,
                    insts: vec![store()],
                    sectors: vec![0],
                },
                WarpReplay {
                    warp_slot: 1,
                    insts: vec![store()],
                    sectors: vec![4],
                },
            ],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let MemModel::L1Shared(st) = &mut core.mem else {
            panic!("expected shared mode");
        };
        st.l1.counters.store_sectors = u64::MAX - 1;
        let before = *core.l1_counters().unwrap();
        let mut out = Vec::new();

        assert!(matches!(
            core.advance(ReplayHorizon::Unbounded, &mut out),
            Err(IssueError::L1Access(L1AccessError::Counter(
                L1CounterError::Overflow {
                    field: L1CounterField::StoreSectors,
                }
            )))
        ));
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), 0);
        assert_eq!(core.warps[0].next, 0);
        assert_eq!(core.warps[1].next, 0);
        assert_eq!(*core.l1_counters().unwrap(), before);
    }

    #[test]
    fn constant_l1_retry_time_error_rejects_all_subcores_before_mutation() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let sched = schedule(vec![(0, slot(load, OpClass::Other, "LDG.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![
                WarpReplay {
                    warp_slot: 0,
                    insts: vec![ReplayInst {
                        pc: 0,
                        kind: eligible(0, 1),
                    }],
                    sectors: vec![0],
                },
                WarpReplay {
                    warp_slot: 1,
                    insts: vec![ReplayInst {
                        pc: 0,
                        kind: eligible(0, 1),
                    }],
                    sectors: vec![8],
                },
            ],
        }];
        let mut p = params();
        p.l1_hit_latency = 2;
        p.l1_miss_latency = 10;
        let cfg = p.l1_config(8, 1);
        let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::L1Const(cfg)).unwrap();
        let MemModel::L1Const(l1) = &mut core.mem else {
            panic!("expected constant L1 mode");
        };
        assert!(matches!(
            l1.access_eligible(0, u64::MAX - 15).unwrap(),
            (AccessOutcome::MissLineAlloc, ready) if ready == u64::MAX - 5
        ));
        core.t = u64::MAX - 14;
        let before_l1 = *core.l1_counters().unwrap();
        let mut out = Vec::new();

        assert!(matches!(
            core.advance(ReplayHorizon::Unbounded, &mut out),
            Err(IssueError::L1Access(L1AccessError::TimeOverflow {
                operation: "miss visibility",
                base,
                delta: 10,
            })) if base == u64::MAX - 5
        ));
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), 0);
        assert_eq!(core.warps[0].next, 0);
        assert_eq!(core.warps[1].next, 0);
        assert_eq!(*core.l1_counters().unwrap(), before_l1);
        let MemModel::L1Const(l1) = &core.mem else {
            panic!("expected constant L1 mode");
        };
        assert!(l1.is_fill_in_flight(FillKey::Line(0)));
    }

    #[test]
    fn same_cycle_fill_chain_overflow_rejects_all_subcores_before_mutation() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let sched = schedule(vec![(0, slot(load, OpClass::Other, "LDG.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![
                WarpReplay {
                    warp_slot: 0,
                    insts: vec![ReplayInst {
                        pc: 0,
                        kind: eligible(0, 1),
                    }],
                    sectors: vec![0],
                },
                WarpReplay {
                    warp_slot: 1,
                    insts: vec![ReplayInst {
                        pc: 0,
                        kind: eligible(0, 1),
                    }],
                    sectors: vec![8],
                },
            ],
        }];
        let mut p = params();
        p.l1_hit_latency = 2;
        p.l1_miss_latency = 10;
        let cfg = p.l1_config(8, 1);
        let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::L1Const(cfg)).unwrap();
        core.t = u64::MAX - 15;
        let before_l1 = *core.l1_counters().unwrap();
        let mut out = Vec::new();

        assert!(matches!(
            core.advance(ReplayHorizon::Unbounded, &mut out),
            Err(IssueError::L1Access(L1AccessError::TimeOverflow {
                operation: "miss visibility",
                base,
                delta: 10,
            })) if base == u64::MAX - 5
        ));
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), 0);
        assert_eq!(core.warps[0].next, 0);
        assert_eq!(core.warps[1].next, 0);
        assert_eq!(*core.l1_counters().unwrap(), before_l1);
        let MemModel::L1Const(l1) = &core.mem else {
            panic!("expected constant L1 mode");
        };
        assert!(!l1.is_fill_in_flight(FillKey::Line(0)));
        assert!(!l1.is_fill_in_flight(FillKey::Line(2)));
    }

    #[test]
    fn response_tokens_reject_cross_core_and_stale_replay_swaps() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let sched = schedule(vec![(0, slot(load, OpClass::Other, "LDG.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: eligible(0, 1),
                }],
                sectors: vec![0],
            }],
        }];
        let mut first = shared_core(&blocks, &sched, &params());
        let mut second = shared_core(&blocks, &sched, &params());
        let mut first_requests = Vec::new();
        let mut second_requests = Vec::new();
        first
            .advance(ReplayHorizon::Unbounded, &mut first_requests)
            .unwrap();
        second
            .advance(ReplayHorizon::Unbounded, &mut second_requests)
            .unwrap();
        let first_reply = first_requests[0].reply;
        let second_reply = second_requests[0].reply;
        assert_ne!(first_reply, second_reply);
        let second_before = *second.l1_counters().unwrap();
        let mut emitted = Vec::new();
        assert!(matches!(
            second.on_response(first_reply, 244, &mut emitted),
            Err(IssueError::ResponseTokenNotOutstanding { reply }) if reply == first_reply
        ));
        assert!(emitted.is_empty());
        assert_eq!(second.outstanding_responses(), 1);
        assert_eq!(*second.l1_counters().unwrap(), second_before);

        drop(first);
        let mut next_generation = shared_core(&blocks, &sched, &params());
        let mut next_requests = Vec::new();
        next_generation
            .advance(ReplayHorizon::Unbounded, &mut next_requests)
            .unwrap();
        assert_ne!(first_reply, next_requests[0].reply);
        assert!(matches!(
            next_generation.on_response(first_reply, 244, &mut emitted),
            Err(IssueError::ResponseTokenNotOutstanding { reply }) if reply == first_reply
        ));
        assert_eq!(next_generation.outstanding_responses(), 1);
    }

    #[test]
    fn response_to_non_shared_mode_is_typed_and_nonmutating() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0)])],
        }];
        let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
        let mut out = Vec::new();

        assert!(matches!(
            core.on_response(ReplyTo::None, 0, &mut out),
            Err(IssueError::ResponseModeMismatch { mode: "flat" })
        ));
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), 0);
        assert_eq!(core.warps[0].next, 0);
    }

    #[test]
    fn fill_waiter_underflow_rejects_response_before_any_mutation() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(1);
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
        let mut requests = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut requests)
                .unwrap(),
            Advance::Idle { wake_at: None }
        );
        let reply = requests[0].reply;
        let ReplyTo::L1Fill { fill, .. } = reply else {
            panic!("expected fill response");
        };
        core.warps[0].barrier_pending[0] = 0;
        let before_l1 = *core.l1_counters().unwrap();
        let mut emitted = Vec::new();

        assert!(matches!(
            core.on_response(reply, 244, &mut emitted),
            Err(IssueError::PendingBarrierUnderflow {
                warp: 0,
                barrier: 0,
                current: 0,
                remove: 1,
            })
        ));
        assert!(emitted.is_empty());
        assert_eq!(core.outstanding_responses(), 1);
        assert_eq!(*core.l1_counters().unwrap(), before_l1);
        let MemModel::L1Shared(st) = &core.mem else {
            panic!("expected shared mode");
        };
        assert!(st.l1.is_fill_in_flight(fill));
        assert!(st.fill_waiters.contains_key(&fill));
        if let FillKey::Line(line) = fill {
            assert!(st.fill_masks.contains_key(&line));
        }
    }

    #[test]
    fn retry_l1_overflow_preserves_the_entire_response_transaction() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(1);
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
                sectors: vec![0, 8],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut requests = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut requests)
                .unwrap(),
            Advance::Idle { wake_at: None }
        );
        let reply = requests[0].reply;
        let ReplyTo::L1Fill { fill, .. } = reply else {
            panic!("expected fill response");
        };
        core.warps[0].barrier_pending[0] = 2;
        let MemModel::L1Shared(st) = &mut core.mem else {
            panic!("expected shared mode");
        };
        st.retry.push_back(Retry {
            warp: 0,
            instruction: 0,
            barrier: Some(barrier(0)),
            lo: 1,
            hi: 2,
        });
        st.l1.counters.misses = u64::MAX;
        let before_l1 = *core.l1_counters().unwrap();
        let mut emitted = Vec::new();

        assert!(matches!(
            core.on_response(reply, 244, &mut emitted),
            Err(IssueError::L1Access(L1AccessError::Counter(
                L1CounterError::Overflow {
                    field: L1CounterField::Misses,
                }
            )))
        ));
        assert!(emitted.is_empty());
        assert_eq!(core.outstanding_responses(), 1);
        assert_eq!(core.warps[0].barrier_pending[0], 2);
        assert_eq!(*core.l1_counters().unwrap(), before_l1);
        let MemModel::L1Shared(st) = &core.mem else {
            panic!("expected shared mode");
        };
        assert!(st.l1.is_fill_in_flight(fill));
        assert!(st.fill_waiters.contains_key(&fill));
        assert_eq!(st.retry.len(), 1);
        let retry = st.retry.front().unwrap();
        assert_eq!(retry.warp, 0);
        assert_eq!(retry.lo, 1);
        assert_eq!(retry.hi, 2);
        if let FillKey::Line(line) = fill {
            assert!(st.fill_masks.contains_key(&line));
        }
    }

    #[test]
    fn retry_replacement_accepts_exact_maximum_pending_count() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(1);
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
                sectors: vec![0, 8, 12],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut requests = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut requests)
                .unwrap(),
            Advance::Idle { wake_at: None }
        );
        let reply = requests[0].reply;
        core.warps[0].barrier_pending[0] = u32::MAX;
        let MemModel::L1Shared(st) = &mut core.mem else {
            panic!("expected shared mode");
        };
        st.retry.push_back(Retry {
            warp: 0,
            instruction: 0,
            barrier: Some(barrier(0)),
            lo: 1,
            hi: 3,
        });
        let mut emitted = Vec::new();

        core.on_response(reply, 244, &mut emitted).unwrap();
        assert_eq!(emitted.len(), 2);
        assert_eq!(core.outstanding_responses(), 2);
        assert_eq!(core.warps[0].barrier_pending[0], u32::MAX);
        let MemModel::L1Shared(st) = &core.mem else {
            panic!("expected shared mode");
        };
        assert!(st.retry.is_empty());
    }

    #[test]
    fn admission_counter_failure_does_not_partially_admit_a_block() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
        let blocks = vec![
            BlockReplay {
                launch_index: 0,
                warps: vec![warp(0, vec![inst(0)])],
            },
            BlockReplay {
                launch_index: 1,
                warps: vec![warp(1, vec![inst(0)])],
            },
        ];
        let mut p = params();
        p.max_resident_warps_per_sm = 1;
        let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
        assert_eq!(core.next_block, 1);
        core.resident_warps = 0;
        core.arrival_stamp = u64::MAX;
        let before_warps = core.warps.len();
        let before_blocks = core.block_states.len();
        let before_subcores = core.subcore_warps.each_ref().map(Vec::len);

        assert!(matches!(
            core.admit_blocks(1),
            Err(IssueError::CounterOverflow {
                counter: "warp arrival stamp"
            })
        ));
        assert_eq!(core.warps.len(), before_warps);
        assert_eq!(core.block_states.len(), before_blocks);
        assert_eq!(core.subcore_warps.each_ref().map(Vec::len), before_subcores);
        assert_eq!(core.next_block, 1);
        assert_eq!(core.arrival_stamp, u64::MAX);
    }

    #[test]
    fn exact_max_accounting_total_is_valid_when_nothing_is_added() {
        assert_eq!(
            checked_u64_counter_add("test total", u64::MAX, 0).unwrap(),
            u64::MAX
        );
    }

    #[test]
    fn shared_mode_miss_blocks_consumer_until_response() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000001);
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
        let a = core.advance(ReplayHorizon::Unbounded, &mut out).unwrap();
        assert_eq!(a, Advance::Idle { wake_at: None });
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].issue_time, 0);
        assert_eq!(out[0].source, L2Source::L1Fill);
        let reply = out[0].reply;
        let ReplyTo::L1Fill { response_id, .. } = reply else {
            panic!("expected an L1 fill response token");
        };
        let mut out2 = Vec::new();
        assert!(matches!(
            core.on_response(
                ReplyTo::Warp {
                    warp: 0,
                    barrier: barrier(0),
                    response_id,
                },
                423,
                &mut out2,
            ),
            Err(IssueError::ResponseTokenNotOutstanding { .. })
        ));
        assert!(matches!(
            core.on_response(
                ReplyTo::L1Fill {
                    fill: FillKey::Line(999),
                    response_id,
                },
                423,
                &mut out2,
            ),
            Err(IssueError::ResponseTokenNotOutstanding { .. })
        ));
        assert!(out2.is_empty());
        assert_eq!(core.outstanding_responses(), 1);
        assert_eq!(core.warps[0].barrier_pending[0], 1);
        // Response at the DRAM arm (423): consumer issues at 423.
        core.on_response(reply, 423, &mut out2).unwrap();
        assert!(out2.is_empty());
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out2).unwrap(),
            Advance::Done
        );
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
        load.write_barrier = Some(barrier(0));
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
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Done
        );
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
        let a = core2
            .advance(ReplayHorizon::Bounded(10), &mut out2)
            .unwrap();
        assert_eq!(
            a,
            Advance::Idle { wake_at: Some(10) },
            "the next known timer (16) lies past the horizon: park AT the \
             horizon, where an event may ready an earlier warp"
        );
        assert_eq!(
            core2.advance(ReplayHorizon::Unbounded, &mut out2).unwrap(),
            Advance::Done
        );
    }

    #[test]
    fn shared_mode_merged_waiters_all_clear_on_one_response() {
        // Two warps load the SAME sector; warp 1 merges into warp 0's
        // fill. Both consumers wait; ONE response clears both.
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000001);
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
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Idle { wake_at: None }
        );
        assert_eq!(out.len(), 1, "second warp merged, no second request");
        let c = core.l1_counters().unwrap();
        assert_eq!((c.misses, c.merges), (2, 1));
        core.on_response(out[0].reply, 300, &mut out).unwrap();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Done
        );
        // Both consumers issued at 300 (distinct sub-cores) → cycles 301.
        assert_eq!(core.result().cycles, 301);
    }

    #[test]
    fn unbarriered_retry_hit_skips_unused_timing_arms() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDG.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: eligible(0, 1),
                }],
                sectors: vec![0, 8],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let MemModel::L1Shared(st) = &mut core.mem else {
            panic!("expected shared mode");
        };
        let EvAccess::MissRequest { fill } = st.l1.access_eligible_ev(8, 0).unwrap() else {
            panic!("expected cold prefill miss");
        };
        st.l1.complete_fill(fill).unwrap();

        let mut out = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Done
        );
        assert_eq!(out.len(), 1, "cold issued load still owns an L1 fill");
        let reply = out.pop().unwrap().reply;
        let MemModel::L1Shared(st) = &mut core.mem else {
            panic!("expected shared mode");
        };
        st.retry.push_back(Retry {
            warp: 0,
            instruction: 0,
            barrier: None,
            lo: 1,
            hi: 2,
        });
        let hits_before = st.l1.counters.hits;

        core.on_response(reply, u64::MAX - 1, &mut out).unwrap();
        assert!(out.is_empty());
        assert_eq!(core.outstanding_responses(), 0);
        assert_eq!(core.l1_counters().unwrap().hits, hits_before + 1);
        let MemModel::L1Shared(st) = &core.mem else {
            panic!("expected shared mode");
        };
        assert!(st.retry.is_empty());
        assert_eq!(core.warps[0].barrier_clear, [0; DEP_BARRIERS]);
    }

    #[test]
    fn shared_mode_mshr_stall_parks_and_retries_on_completion() {
        // MSHR capacity 1: the second warp's DIFFERENT-line load stalls,
        // parks, and retries when the first fill completes.
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000001);
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
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Idle { wake_at: None }
        );
        assert_eq!(out.len(), 1, "second load stalled, not requested");
        assert_eq!(core.l1_counters().unwrap().stall_events, 1);
        // First fill completes → retry emits the parked request at 244.
        let first = out[0].reply;
        out.clear();
        assert!(matches!(
            core.on_response(first, u64::MAX - 1, &mut out),
            Err(IssueError::TimeOverflow {
                operation: "retry self-response horizon",
                base,
                delta: 244,
            }) if base == u64::MAX - 1
        ));
        assert!(out.is_empty());
        assert_eq!(core.outstanding_responses(), 1);
        let MemModel::L1Shared(st) = &core.mem else {
            panic!("expected shared mode");
        };
        assert_eq!(st.retry.len(), 1);
        assert_eq!(core.warps[1].barrier_pending[0], 1);

        core.on_response(first, 244, &mut out).unwrap();
        assert_eq!(out.len(), 1, "retry sent the parked sector");
        assert_eq!(out[0].issue_time, 244);
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Idle { wake_at: None }
        );
        let second = out[0].reply;
        core.on_response(second, 488, &mut out).unwrap();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Done
        );
        let r = core.result();
        assert_eq!(r.instructions_issued, 4);
        // Warp 1's consumer issues at 488 → cycles 489.
        assert_eq!(r.cycles, 489);
    }

    #[test]
    fn shared_mode_bypass_load_waits_on_direct_reply() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(2));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000100);
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
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
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
                barrier: got,
                ..
            } if got == barrier(2)
        ));
        // Both responses must land before the wait clears.
        let mut sink = Vec::new();
        core.on_response(out[0].reply, 244, &mut sink).unwrap();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut sink).unwrap(),
            Advance::Idle { wake_at: None },
            "one of two pending units still outstanding"
        );
        core.on_response(out[1].reply, 423, &mut sink).unwrap();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut sink).unwrap(),
            Advance::Done
        );
        assert_eq!(core.result().cycles, 424);
    }

    #[test]
    fn invalid_response_times_are_rejected_before_mutation() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(2));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000100);
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
                            len: 1,
                            token: CacheOpToken::LdgECg,
                        },
                    },
                    inst(16),
                ],
                sectors: vec![40],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut requests = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut requests)
                .unwrap(),
            Advance::Idle { wake_at: None }
        );
        assert_eq!(core.outstanding_responses(), 1);
        assert_eq!(core.warps[0].barrier_pending[2], 1);
        let ReplyTo::Warp { response_id, .. } = requests[0].reply else {
            panic!("expected a direct warp response token");
        };

        let mut retries = Vec::new();
        assert_eq!(core.now(), 1);
        assert!(matches!(
            core.on_response(requests[0].reply, 0, &mut retries),
            Err(IssueError::ResponseBeforeClock { now: 0, clock: 1 })
        ));
        assert!(matches!(
            core.on_response(
                ReplyTo::Warp {
                    warp: 1,
                    barrier: barrier(2),
                    response_id,
                },
                1,
                &mut retries,
            ),
            Err(IssueError::ResponseWarpOutOfRange { warp: 1, warps: 1 })
        ));
        assert!(matches!(
            core.on_response(
                ReplyTo::Warp {
                    warp: 0,
                    barrier: barrier(1),
                    response_id,
                },
                1,
                &mut retries,
            ),
            Err(IssueError::ResponseTokenNotOutstanding { .. })
        ));
        assert!(matches!(
            core.on_response(ReplyTo::None, 1, &mut retries),
            Err(IssueError::UnexpectedMutationOnlyResponse)
        ));
        assert!(retries.is_empty());
        assert_eq!(core.now(), 1);
        assert_eq!(core.outstanding_responses(), 1);
        assert_eq!(core.warps[0].barrier_pending[2], 1);
        assert_eq!(core.warps[0].barrier_clear[2], 0);

        core.on_response(requests[0].reply, 244, &mut retries)
            .unwrap();
        assert!(matches!(
            core.on_response(requests[0].reply, 244, &mut retries),
            Err(IssueError::ResponseTokenNotOutstanding { .. })
        ));
        assert!(retries.is_empty());
        assert_eq!(core.now(), 1);
        assert_eq!(core.outstanding_responses(), 0);
        assert_eq!(core.warps[0].barrier_pending[2], 0);
        assert_eq!(core.warps[0].barrier_clear[2], 244);
    }

    #[test]
    fn valid_response_at_exact_maximum_clears_its_barrier() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(2));
        let sched = schedule(vec![(0, slot(load, OpClass::Other, "LDG.E.CG"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: AccessKind::GlobalBypass {
                        off: 0,
                        len: 1,
                        token: CacheOpToken::LdgECg,
                    },
                }],
                sectors: vec![40],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut requests = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut requests)
                .unwrap(),
            Advance::Done
        );
        assert_eq!(requests.len(), 1);
        assert_eq!(core.warps[0].barrier_pending[2], 1);
        let mut retries = Vec::new();

        core.on_response(requests[0].reply, u64::MAX, &mut retries)
            .unwrap();

        assert!(retries.is_empty());
        assert_eq!(core.outstanding_responses(), 0);
        assert_eq!(core.warps[0].barrier_pending[2], 0);
        assert_eq!(core.warps[0].barrier_clear[2], u64::MAX);
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
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Done
        );
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
        load.write_barrier = Some(barrier(2));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000100);
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
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Idle { wake_at: None }
        );
        assert_eq!(out.len(), 1, "one L2 transaction for the whole line");
        assert_eq!(out[0].sector_mask, 0b1111);
        assert!(matches!(
            out[0].reply,
            ReplyTo::L1Fill {
                fill: FillKey::Line(10),
                ..
            }
        ));
        let mut sink = Vec::new();
        core.on_response(out[0].reply, 244, &mut sink).unwrap();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut sink).unwrap(),
            Advance::Done
        );
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
        load_a.write_barrier = Some(barrier(2));
        let mut load_b = cb(1);
        load_b.write_barrier = Some(barrier(2));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000100);
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
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Idle { wake_at: None }
        );
        assert_eq!(out.len(), 1, "one line transaction for the whole batch");
        assert_eq!(out[0].sector_mask, 0b11, "B's sector folded into A's fill");
        assert!(matches!(
            out[0].reply,
            ReplyTo::L1Fill {
                fill: FillKey::Line(10),
                ..
            }
        ));
        let mut sink = Vec::new();
        core.on_response(out[0].reply, 244, &mut sink).unwrap();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut sink).unwrap(),
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
        load_m.write_barrier = Some(barrier(2));
        let mut load_l = cb(1);
        load_l.write_barrier = Some(barrier(3));
        // Merging load: gated on barrier 2 (fill M's return) so it
        // issues in a LATER advance call while fill L is still in
        // flight; its own waiter registers on barrier 2.
        let mut load_merge = cb(1);
        load_merge.write_barrier = Some(barrier(2));
        load_merge.wait_mask = wait_mask(0b000100);
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b001100);
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
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Idle { wake_at: None }
        );
        assert_eq!(out.len(), 2, "two fills; the merging load is gated");
        assert!(matches!(
            out[0].reply,
            ReplyTo::L1Fill {
                fill: FillKey::Line(20),
                ..
            }
        ));
        assert!(matches!(
            out[1].reply,
            ReplyTo::L1Fill {
                fill: FillKey::Line(10),
                ..
            }
        ));
        assert_eq!(out[1].sector_mask, 0b01);
        // Fill M returns; the merging load issues in a NEW batch while
        // fill L is still in flight.
        let mut out2 = Vec::new();
        core.on_response(out[0].reply, 423, &mut out2).unwrap();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out2).unwrap(),
            Advance::Idle { wake_at: None }
        );
        assert_eq!(out2.len(), 1, "exactly one follow-up fetch");
        assert_eq!(out2[0].sector_mask, 0b10, "only the new sector");
        assert_eq!(out2[0].reply, ReplyTo::None, "no second response");
        assert_eq!(out2[0].source, L2Source::L1Fill);
        // Fill L's one response clears BOTH its own waiter (barrier 3)
        // and the merged load's waiter (barrier 2).
        let mut sink = Vec::new();
        core.on_response(out[1].reply, 846, &mut sink).unwrap();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut sink).unwrap(),
            Advance::Done
        );
    }

    /// ATOM records ride BYPASS provenance (they never stage into the
    /// modeled L1) but carry the §6.4 WRITE class at the L2: the source
    /// says who sent the sector, the class says how it hits.
    #[test]
    fn shared_mode_atom_is_bypass_provenance_write_class() {
        let mut atom = cb(1);
        atom.write_barrier = Some(barrier(2));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000100);
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
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
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
                    barrier: got,
                    ..
                } if got == barrier(2)
            ),
            "the atomic's return value still waits on the reply"
        );
    }
}
