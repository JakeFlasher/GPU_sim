//! SMSP issue model (design §6.2): per-warp program replay where issue
//! timing is driven by the compiler's own dependence encoding — the SASS
//! control bits (stall counts, dependence-barrier set/wait, yield) — not by
//! simulated scoreboards. This is the 13–17% cycles-MAPE lever the UPC
//! MICRO'25 modern-core work identified `F6`.
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
//! - Admitted tensor instructions additionally wait for their SMSP-local
//!   Tensor Core initiation timeline. Other pipelines remain dispatch-only,
//!   so independent non-tensor work may issue while the tensor pipe is busy.
//! - Exact predicated stall-5/yield branches carry architecture-local
//!   readiness: ten cycles for the characterized forward construction and
//!   three target-resolution cycles for ordinary uniform backward loops. A
//!   separate compact-loop classifier admits the measured nine-cycle
//!   next-issue handoff when a fresh predicate immediately feeds a backward
//!   edge no more than 64 bytes away; longer/other forms retain the ordinary
//!   control path.
//! - `BAR.*` (classified by the sidecar's opcode text) is a block-wide
//!   barrier: generation semantics — it releases when every unfinished
//!   warp of the block is waiting at it (warps that already finished do
//!   not block release, matching Volta+ exited-thread semantics).
//! - Block residency: blocks are admitted to the SM in launch order while
//!   resident warps and resident shared-memory bytes stay within their
//!   architecture-local ceilings and concurrent CTAs use disjoint physical warp
//!   slots; a finished block frees both resources for a later CTA (wave execution).
//! - Memory dispatch uses the measured Ampere five-entry SMSP queue (four
//!   queued instructions plus the dispatch latch), operation/register-width
//!   WAR timing, one local address/operand acceptance every four cycles, and
//!   one SM-shared acceptance every two cycles. Cache mutation occurs at that
//!   shared-accept timestamp rather than scheduler dispatch.
//!
//! Memory timing comes in three modes ([`SmMemMode`]):
//! - **Flat**: no cache model; global-class barriers clear at the flat
//!   cold-DRAM latency (the pre-L1 v0 behavior).
//! - **L1Const**: a per-SM sectored L1 probed at memory-pipeline shared
//!   acceptance with fill times known there (constant L2-hit latency below the L1) — the
//!   sim-mem L1 v0 behavior, kept as the lockstep-differential surface.
//! - **L1Shared**: the request/response INVERSION for the shared L2
//!   (design §6.3): an L1 miss's fill time is NOT knowable at issue —
//!   the shared L2's state when the request arrives decides L2-hit vs
//!   DRAM pricing. Misses emit [`MemRequest`]s; ordinary load barriers
//!   and non-bulk async-copy groups retain typed completion targets on
//!   hits, fills, and MSHR retries. [`SmCore::on_response`] completes the
//!   fill and every target at response-event time. An explicit
//!   `LDGSTS.E.BYPASS` instead keeps the direct-L2 async response route.
//!   MSHR-full accesses park in a retry queue drained on the next
//!   completion (the event-mode analogue of the constant mode's
//!   retry-at-earliest-fill; the warp itself keeps issuing until a wait
//!   mask or async-group wait stops it, like the hardware's LSU replay
//!   path).
//!
//! [`SmCore`] is a resumable actor: `advance(horizon)` runs issue cycles
//! strictly BEFORE `horizon` (the event system passes the next queued
//! event's time so no cross-SM effect is overrun; a self-emitted request
//! additionally caps the run-ahead at the earliest time its response
//! could arrive). [`simulate_sm_with_residency`] is the synchronous wrapper for
//! the Flat/L1Const modes (horizon = ∞, no requests possible).
//!
//! The instruction COUNT produced by replay is placement- and
//! timing-invariant: every traced record issues exactly once, so
//! `instructions_issued` is exact regardless of the cycle model's fidelity.
//! The timing-active shared-L2 replay retains an unadjusted causal clock.  A
//! separate architecture-scoped SM-active observation projection can add its
//! measured launch boundary without contaminating event timestamps.

use std::cell::RefCell;
use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::num::{NonZeroU32, NonZeroU128};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::Serialize;
use sim_arch::card::CalibError;
use sim_arch::{
    ArchId, CtaBarrierIssue, FixedL1TexWritebackPriority, ImmediateConstantCache,
    InstructionFrontEnd, MemoryIssuePipeline, ScalarExecutionPipeline,
};
use sim_mem::{
    AsyncCopyCompletion, CacheOpToken, EvAccess, EvAccessUnobserved, FillKey, L1AccessError,
    L1Cache, L1Config, L1Counters, L1LookupBatch, L1MutationBudget, L1StorePolicy,
    L1WriteLookupObserver, L1WriteLookupObserverPlan, L2Source, LdgstsL1Policy, MemRole,
    MemoryOrigin, decode_async_copy_completion, opcode_head,
};
#[cfg(feature = "reference")]
use sim_semantics::{
    AsyncCopyBoundaryState, AsyncCopyBypassObservationState, AsyncCopyCohortBoundaryState,
    AsyncGroupBoundaryState, BlockBoundaryState, BoundarySnapshot, CanonicalLineKey,
    InstructionCacheEntryState, InstructionCacheState, InstructionFrontEndBoundaryState,
    InstructionStreamLineState, InstructionStreamState, L1BoundaryState, L2RequestKey,
    L2RequestOwner, L2RequestState, MemoryPipelineBoundaryState, ScalarPipelineBoundaryState,
    TypedEventTransition, TypedResponseKind, WarpBoundaryState,
};
#[cfg(feature = "reference")]
use sim_semantics::{NativeBoundaryExecutionPath, NativeBoundaryExecutionResult};

use crate::control_bits::{ControlDomainError, DepBarrier, WaitMask};
use crate::metrics::{SmActivity, WarpLifetime};
use crate::program::{
    sm120_alpha_scalar_control_stall_suppressed, FunctionSchedule, IssueSlot, OpClass,
    PipelineClass,
};
use crate::sass_operands::{
    FixedLatencyOpcode, FixedLatencyOperandStatus, SassRegister, SassRegisterFile,
    SassSourceOperand,
};

pub use sim_mem::AccessKind;

mod config;
pub use config::{IssueModelParams, IssueResourceSource, Sm120AlphaConfig};
pub(crate) use config::{checked_cycle_add, SharedMemoryResidency};

mod memory_types;
pub use memory_types::{MemoryOperation, TailFlags};

mod readiness;
pub use readiness::{
    AsyncInstrumentationOrderDiagnostics, AsyncIssueOverlapDiagnostics,
    AsyncReadinessDiagnostics, AsyncWaitPrefixDiagnostics, BarrierPipelineDiagnostics,
    BypassAsyncCopyDiagnostics, CtaBarrierDiagnostics, ControlPipelineDiagnostics,
    ClockIssueDiagnostics, GlobalMemoryIssueDiagnostics, ReadinessCauseCycles,
    ReadinessCounterError, ResultWritebackDiagnostics,
};
use readiness::{
    checked_readiness_add, dependency_origin, observe_readiness_constraint,
    operand_read_dependency_origin, projected_global_memory_issue, projected_l1_write_lookup_hits,
};

mod candidates;
pub use candidates::{
    MemoryAdmissionCandidate, MemoryAdmissionCandidateDiagnostics, MemoryAdmissionCandidateError,
    MemoryAdmissionCandidateResult, MemoryAdmissionFeedbackDiagnostics, MemoryServiceCandidate,
    MemoryServiceCandidateDiagnostics, MemoryServiceCandidateError, MemoryServiceCandidateOperation,
    MemoryServiceCandidateResult, MemoryServiceCandidateScope, MemoryWriteAckDiagnostics,
    MAX_MEMORY_ADMISSION_CANDIDATES, MAX_MEMORY_ADMISSION_QUEUE_CAPACITY,
    MAX_MEMORY_ADMISSION_SERVICE_WIDTH_SECTORS, MAX_MEMORY_SERVICE_CANDIDATES,
};
pub(crate) use candidates::{
    validate_memory_admission_candidates, validate_memory_service_candidates,
    MemoryAdmissionCandidateState,
};
use candidates::{MemoryAdmissionFeedbackState, MemoryServiceCandidateState, MemoryAdmissionFeedbackIssue};

/// One dynamic instruction, compacted for replay.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ReplayInst {
    /// Function-relative pc — the join key into [`FunctionSchedule`].
    pub(crate) pc: u32,
    pub(crate) kind: AccessKind,
    /// Effective executing lanes for the dynamic instruction. Native traces
    /// carry both masks; legacy synthetic fixtures remain all-lane.
    pub(crate) lane_mask: u32,
}

impl ReplayInst {
    pub(crate) fn effective_lane_mask(self) -> u32 {
        self.lane_mask
    }
}

/// One traced warp of a block: physical slot + its dynamic stream + the
/// sector arena its records' `off/len` payloads index.
#[derive(Debug, Default)]
pub(crate) struct WarpReplay {
    /// Physical warp slot on the SM (`warpid_sm` when traced); sub-core
    /// assignment is `warp_slot % 4`.
    pub(crate) warp_slot: i32,
    pub(crate) insts: Vec<ReplayInst>,
    pub(crate) sectors: Vec<i64>,
    /// Sparse, cold LDGSTS source-cohort projection. Only destination-bearing
    /// records allocate this arena; the hot [`ReplayInst`] stays unchanged.
    pub(crate) async_copy_cohorts: Option<Box<AsyncCopyCohortReplay>>,
}

/// Exact trace-derived topology of one flexible 128-byte LDGSTS source cohort.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AsyncCopyCohortDescriptor {
    pub(crate) payload_bytes: u32,
    pub(crate) destination_lines: u32,
    pub(crate) bank_wavefronts: u32,
    pub(crate) bypass_wavefronts: u32,
}

/// Sparse arena span for one dynamic LDGSTS record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AsyncCopyCohortSpan {
    pub(crate) instruction: usize,
    pub(crate) sector_off: usize,
    pub(crate) sector_len: u16,
    pub(crate) cohort_off: usize,
    pub(crate) cohort_len: u8,
    pub(crate) membership_off: usize,
}

/// One cold allocation per warp, indexed sparsely by dynamic instruction.
#[derive(Debug, Default)]
pub(crate) struct AsyncCopyCohortReplay {
    pub(crate) spans: Vec<AsyncCopyCohortSpan>,
    pub(crate) cohorts: Vec<AsyncCopyCohortDescriptor>,
    pub(crate) sector_membership_masks: Vec<u32>,
    /// Sparse selected-instruction callback/channel order. This shares the
    /// cold async sidecar allocation but is never consumed by issue timing.
    pub(crate) instrumentation_order: Vec<InstrumentationOrderAnchor>,
}

/// One selected dynamic instruction's 1-based callback/channel order token.
/// `instruction` indexes the owning [`WarpReplay::insts`] stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct InstrumentationOrderAnchor {
    pub(crate) instruction: usize,
    pub(crate) order: u64,
}

#[derive(Debug, Clone, Copy)]
struct AsyncCopyCohortView<'a> {
    sector_off: usize,
    sector_len: u16,
    cohorts: &'a [AsyncCopyCohortDescriptor],
    sector_membership_masks: &'a [u32],
}

impl AsyncCopyCohortReplay {
    fn view(&self, instruction: usize) -> Option<AsyncCopyCohortView<'_>> {
        let span = self
            .spans
            .binary_search_by_key(&instruction, |span| span.instruction)
            .ok()
            .map(|index| self.spans[index])?;
        let cohort_end = span.cohort_off.checked_add(usize::from(span.cohort_len))?;
        let membership_end = span
            .membership_off
            .checked_add(usize::from(span.sector_len))?;
        Some(AsyncCopyCohortView {
            sector_off: span.sector_off,
            sector_len: span.sector_len,
            cohorts: self.cohorts.get(span.cohort_off..cohort_end)?,
            sector_membership_masks: self
                .sector_membership_masks
                .get(span.membership_off..membership_end)?,
        })
    }
}

/// One traced block assigned to this SM, in launch order.
#[derive(Debug)]
pub(crate) struct BlockReplay {
    pub(crate) launch_index: i64,
    pub(crate) warps: Vec<WarpReplay>,
}


/// Memory-model mode for one SM replay.
#[derive(Debug, Clone, Copy)]
pub(crate) enum SmMemMode {
    /// No cache model: flat global latency.
    Flat,
    /// Per-SM L1 with constant (L2-hit) fill latency below it.
    L1Const(L1Config),
    /// Per-SM L1 over the SHARED L2: misses become requests/responses.
    L1Shared(L1Config),
}

#[derive(Debug, thiserror::Error)]
pub enum IssueError {
    #[error("hypothetical issue-port override must be in 1..=4; got {issue_ports}")]
    InvalidHypotheticalIssuePorts { issue_ports: u32 },
    #[error("a hypothetical issue-port override was already applied")]
    HypotheticalOverrideAlreadyApplied,
    #[error("issue model is unsupported for deployment architecture {arch}")]
    ArchitectureUnsupported { arch: ArchId },
    #[cfg(feature = "reference")]
    #[error("SIM_SM_REFERENCE_MODE must be `event` or `dense`, found {mode:?}")]
    ReferenceModeInvalid { mode: String },
    #[cfg(feature = "reference")]
    #[error("reference scalar macro unsupported: {reason}")]
    ReferenceMacroUnsupported { reason: &'static str },
    #[error(
        "one function declares inconsistent CTA-barrier pipeline configurations: expected {expected:?}, got {got:?}"
    )]
    BarrierPipelineConfigurationMismatch {
        expected: CtaBarrierIssue,
        got: CtaBarrierIssue,
    },
    #[error(
        "carded issue resource {field}={got} is incompatible with the consumed SM86 topology; expected {expected} for {arch}"
    )]
    CardedResourceMismatch {
        arch: ArchId,
        field: &'static str,
        expected: u64,
        got: u64,
    },
    #[error("architecture-default mismatch for {arch}: {field} is {got}, expected {expected}")]
    ArchitectureDefaultMismatch {
        arch: ArchId,
        field: &'static str,
        expected: u64,
        got: u64,
    },
    #[error(
        "architecture {arch} has invalid cacheable-LDGSTS L1 miss-admission floor {floor}; expected a positive value"
    )]
    InvalidLdgstsL1MshrFloor { arch: ArchId, floor: u32 },
    #[error(
        "architecture {arch} has invalid LDGSTS completion throughput: cache-all={cache_all_bytes_per_cycle} B/cycle, bypass={bypass_bytes_per_cycle} B/cycle; rates must be positive and divide their maximum"
    )]
    InvalidLdgstsCompletionThroughput {
        arch: ArchId,
        cache_all_bytes_per_cycle: u64,
        bypass_bytes_per_cycle: u64,
    },
    #[error(
        "architecture {arch} has invalid cache-all LDGSTS destination timing: {cycles_per_wavefront} cycles per wavefront; expected a positive value"
    )]
    InvalidLdgstsDestinationWavefrontTiming {
        arch: ArchId,
        cycles_per_wavefront: u64,
    },
    #[error(
        "architecture {arch} has invalid cache-all LDGSTS destination throughput: {service_wavefronts} wavefronts per {service_cycles} SM cycles; both terms must be positive"
    )]
    InvalidLdgstsDestinationWavefrontThroughput {
        arch: ArchId,
        service_wavefronts: u64,
        service_cycles: u64,
    },
    #[error(
        "cache-all LDGSTS exact destination work {completion_work} is not an integral multiple of {work_per_wavefront} work units per wavefront"
    )]
    InvalidAsyncCopyDestinationWork {
        completion_work: u64,
        work_per_wavefront: u64,
    },
    #[error(
        "warp {warp} asynchronous-copy source activity is inconsistent: sectors={sectors}, completion_work={completion_work}"
    )]
    AsyncCopyCompletionWorkMismatch {
        warp: usize,
        sectors: u16,
        completion_work: u64,
    },
    #[error("issue-card validation failed: {0}")]
    Calibration(#[from] CalibError),
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
    #[error(
        "pc {pc} (block {launch_index}, warp slot {warp_slot}) has incoherent atomic topology: sector transactions={sectors}, L2 requests={l2_requests}"
    )]
    AtomicRequestTopologyInvalid {
        pc: u32,
        launch_index: i64,
        warp_slot: i32,
        sectors: u16,
        l2_requests: u16,
    },
    #[error("block {launch_index} has {warps} live warps, which does not fit in u32")]
    LiveWarpCountOutOfRange { launch_index: i64, warps: usize },
    #[error(
        "block {launch_index} has {live_warps} live warps, exceeding architecture {arch} resident-warp limit {max_resident_warps}"
    )]
    BlockResidencyLimitExceeded {
        launch_index: i64,
        live_warps: u32,
        arch: ArchId,
        max_resident_warps: u32,
    },
    #[error(
        "block {launch_index} exceeds architecture {arch} resident-block limit {max_resident_blocks}"
    )]
    BlockCountResidencyLimitExceeded {
        launch_index: i64,
        arch: ArchId,
        max_resident_blocks: u32,
    },
    #[error(
        "architecture {arch} has invalid shared-memory capacity {max_bytes_per_sm} bytes per SM"
    )]
    SharedMemoryCapacityInvalid { arch: ArchId, max_bytes_per_sm: u64 },
    #[error(
        "block {launch_index} requires {bytes_per_cta} shared-memory bytes, exceeding architecture {arch} per-SM capacity {max_bytes_per_sm}"
    )]
    BlockSharedMemoryLimitExceeded {
        launch_index: i64,
        bytes_per_cta: u64,
        arch: ArchId,
        max_bytes_per_sm: u64,
    },
    #[error(
        "block {launch_index} physical warp slot {warp_slot} is outside architecture {arch} range 0..{physical_warp_slots}"
    )]
    WarpSlotOutOfRange {
        launch_index: i64,
        warp_slot: i32,
        arch: ArchId,
        physical_warp_slots: u32,
    },
    #[error("block {launch_index} assigns physical warp slot {warp_slot} more than once")]
    DuplicateWarpSlot { launch_index: i64, warp_slot: i32 },
    #[error(
        "block {launch_index} released physical warp slot {warp_slot}, but that slot is not resident"
    )]
    ResidentWarpSlotMissing { launch_index: i64, warp_slot: i32 },
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
    #[error("warp {warp} exhausted the asynchronous-copy group sequence domain")]
    AsyncCopyGroupSequenceExhausted { warp: usize },
    #[error(
        "warp {warp} asynchronous-copy group {group:?} pending source count {current} + {add} overflows u32"
    )]
    AsyncCopyPendingOverflow {
        warp: usize,
        group: AsyncCopyGroupId,
        current: u32,
        add: u32,
    },
    #[error(
        "warp {warp} asynchronous-copy group {group:?} pending source count {current} cannot release {remove} unit(s)"
    )]
    AsyncCopyPendingUnderflow {
        warp: usize,
        group: AsyncCopyGroupId,
        current: u32,
        remove: u32,
    },
    #[error("warp {warp} has no live asynchronous-copy group {group:?}")]
    AsyncCopyGroupNotFound {
        warp: usize,
        group: AsyncCopyGroupId,
    },
    #[error(
        "warp {warp} asynchronous-copy group {group:?} already observes bypass copy instruction {instruction}"
    )]
    AsyncCopyObservationDuplicate {
        warp: usize,
        group: AsyncCopyGroupId,
        instruction: usize,
    },
    #[error(
        "warp {warp} asynchronous-copy group {group:?} has no observed bypass copy instruction {instruction}"
    )]
    AsyncCopyObservationNotFound {
        warp: usize,
        group: AsyncCopyGroupId,
        instruction: usize,
    },
    #[error(
        "warp {warp} asynchronous-copy group {group:?} bypass copy instruction {instruction} pending source count {current} cannot release {remove} unit(s)"
    )]
    AsyncCopyObservationPendingUnderflow {
        warp: usize,
        group: AsyncCopyGroupId,
        instruction: usize,
        current: u32,
        remove: u32,
    },
    #[error(
        "warp {warp} asynchronous-copy group {group:?} bypass copy instruction {instruction} still has {pending} source unit(s) at wait"
    )]
    AsyncCopyObservationPendingAtWait {
        warp: usize,
        group: AsyncCopyGroupId,
        instruction: usize,
        pending: u32,
    },
    #[error(
        "warp {warp} bypass-copy instruction {instruction} has invalid cohort projection {field}: expected {expected}, got {got}"
    )]
    AsyncCopyCohortProjectionInvalid {
        warp: usize,
        instruction: usize,
        field: &'static str,
        expected: u64,
        got: u64,
    },
    #[error(
        "warp {warp} asynchronous-copy group {group:?} bypass copy instruction {instruction} source sector {sector_index} is outside cohort-membership span {sector_off}..{sector_end}"
    )]
    AsyncCopyCohortSectorOutOfRange {
        warp: usize,
        group: AsyncCopyGroupId,
        instruction: usize,
        sector_index: usize,
        sector_off: usize,
        sector_end: usize,
    },
    #[error(
        "warp {warp} asynchronous-copy group {group:?} bypass copy instruction {instruction} cohort {cohort} has no pending membership for source sector {sector_index}"
    )]
    AsyncCopyCohortPendingUnderflow {
        warp: usize,
        group: AsyncCopyGroupId,
        instruction: usize,
        sector_index: usize,
        cohort: usize,
    },
    #[error(
        "warp {warp} asynchronous-copy group {group:?} bypass copy instruction {instruction} cohort {cohort} still has {pending} source membership(s) at wait"
    )]
    AsyncCopyCohortPendingAtWait {
        warp: usize,
        group: AsyncCopyGroupId,
        instruction: usize,
        cohort: usize,
        pending: u32,
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
    #[error("L1 visibility fill response {response:?} has no exact sector-mask witness")]
    VisibilityResponseMaskMissing { response: ResponseId },
    #[error(
        "L1 visibility fill response {response:?} has invalid sector mask {got:#06b} (expected {expected:#06b})"
    )]
    VisibilityResponseMaskInvalid {
        response: ResponseId,
        expected: u8,
        got: u8,
    },
    #[error("L1 visibility fill {fill:?} was dropped by an allocation-generation mismatch")]
    VisibilityFillDropped { fill: FillKey },
    #[error("L1 visibility protocol does not admit bypass atomic/RMW observation")]
    VisibilityAtomicUnsupported,
    #[error("L1 visibility protocol does not admit reduction/RMW observation")]
    VisibilityReductionUnsupported,
    #[error("L1 payload for line {line} sector {sector} cannot attach to its store: {reason}")]
    PayloadStoreBinding {
        line: i64,
        sector: u8,
        reason: &'static str,
    },
    #[error("L1 visibility protocol does not admit asynchronous-copy observation")]
    VisibilityAsyncCopyUnsupported,
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
    #[error("{operation}: cycle multiplication {lhs} * {rhs} overflows u64")]
    TimeMultiplicationOverflow {
        operation: &'static str,
        lhs: u64,
        rhs: u64,
    },
    #[error("memory response at t={now} precedes the SM clock t={clock}")]
    ResponseBeforeClock { now: u64, clock: u64 },
    #[error("advance horizon t={horizon} precedes the SM clock t={clock}")]
    HorizonBeforeClock { horizon: u64, clock: u64 },

    #[error("issue capability unavailable: {0}")]
    Capability(#[from] sim_arch::CapabilityError),
    #[error(
        "instruction front-end component {component} has invalid {field}={value}; the value must be positive and representable"
    )]
    InstructionFrontEndGeometryInvalid {
        component: &'static str,
        field: &'static str,
        value: u64,
    },
    #[error("instruction front-end width {field}={value} is not implemented (only one is modeled)")]
    InstructionFrontEndWidthUnsupported { field: &'static str, value: u32 },
    #[error("instruction stream order changed: expected function/line {expected:?}, got {got:?}")]
    InstructionStreamOrder {
        expected: Option<(i32, u64)>,
        got: (i32, u64),
    },
    #[error("warp {warp} instruction buffer is missing dynamic instruction {instruction} at issue")]
    InstructionBufferMissing { warp: usize, instruction: usize },
    #[error(
        "warp {warp} instruction buffer order changed: expected dynamic instruction {expected}, got {got}"
    )]
    InstructionBufferOrder {
        warp: usize,
        expected: usize,
        got: usize,
    },
    #[error("memory issue-pipeline configuration has invalid {field}={value}")]
    MemoryIssuePipelineConfigurationInvalid { field: &'static str, value: u64 },
    #[error("memory issue-pipeline queue on subcore {subcore} is full at t={time}")]
    MemoryIssuePipelineQueueFull { subcore: usize, time: u64 },
    #[error(
        "scalar execution pipeline configuration is invalid: latency={latency}, initiation={initiation}, capacity={capacity}"
    )]
    ScalarExecutionPipelineConfigurationInvalid {
        latency: u64,
        initiation: u64,
        capacity: u32,
    },
    #[error("scalar execution pipeline queue on subcore {subcore} is full at t={time}")]
    ScalarExecutionPipelineQueueFull { subcore: usize, time: u64 },
    #[error("immediate constant-cache configuration is invalid")]
    ImmediateConstantCacheConfigurationInvalid,
    #[error("L1 access failed: {0}")]
    L1Access(#[from] L1AccessError),
    #[error("L1 configuration is invalid: {0}")]
    L1Config(#[from] sim_mem::L1ConfigError),
    #[error("memory-service candidate configuration is invalid: {0}")]
    MemoryServiceCandidate(#[from] MemoryServiceCandidateError),
    #[error("memory-admission candidate configuration is invalid: {0}")]
    MemoryAdmissionCandidate(#[from] MemoryAdmissionCandidateError),
    #[error("feedback admission queue on subcore {subcore} is full at t={time}")]
    MemoryAdmissionFeedbackQueueFull { subcore: usize, time: u64 },
    #[error(
        "feedback admission candidate {candidate_id} declares operand-visibility latency {configured_cycles} cycles, but the active issue model uses {issue_model_cycles} cycles"
    )]
    MemoryAdmissionFeedbackLatencyMismatch {
        candidate_id: u8,
        configured_cycles: u32,
        issue_model_cycles: u64,
    },
    #[error("feedback admission reservation is absent for warp {warp} instruction {instruction}")]
    MemoryAdmissionFeedbackReservationMissing { warp: u32, instruction: u32 },
    #[error(
        "feedback admission reservation was already priced for warp {warp} instruction {instruction}"
    )]
    MemoryAdmissionFeedbackAlreadyPriced { warp: u32, instruction: u32 },
    #[error(
        "modeled write acknowledgement is already pending for warp {warp} instruction {instruction}"
    )]
    MemoryWriteAckAlreadyPending { warp: u32, instruction: u32 },
    #[error(
        "modeled write acknowledgement is not pending for warp {warp} instruction {instruction}"
    )]
    MemoryWriteAckNotPending { warp: u32, instruction: u32 },
    #[error(
        "modeled write acknowledgement payload changed for warp {warp} instruction {instruction}"
    )]
    MemoryWriteAckMismatch { warp: u32, instruction: u32 },
    #[error(
        "modeled write acknowledgement for warp {warp} instruction {instruction} arrived at t={got}, expected t={expected}"
    )]
    MemoryWriteAckTimeMismatch {
        warp: u32,
        instruction: u32,
        expected: u64,
        got: u64,
    },
    #[error("issue diagnostic update failed: {0}")]
    ReadinessDiagnostic(#[from] ReadinessCounterError),
}




impl ClockIssueDiagnostics {
    const fn is_empty(&self) -> bool {
        self.reads_issued == 0
    }

    fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
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

    fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
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
}

impl InstructionFrontEndDiagnostics {
    pub const fn is_empty(&self) -> bool {
        self.instructions_fetched == 0
    }

    fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
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

    fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        let mut next = *self;
        for (field, target, value) in [
            (
                "readiness.memory_issue_pipeline.instructions_admitted",
                &mut next.instructions_admitted,
                addend.instructions_admitted,
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
    pub maximum_pending_lines: u64,
}

impl ImmediateConstantCacheDiagnostics {
    pub const fn is_empty(&self) -> bool {
        self.references == 0
    }

    fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instruction_front_end: Option<Box<InstructionFrontEndDiagnostics>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_issue_pipeline: Option<Box<MemoryIssuePipelineDiagnostics>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub immediate_constant_cache: Option<Box<ImmediateConstantCacheDiagnostics>>,
    pub async_groups: AsyncReadinessDiagnostics,
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
        next.async_groups.add_checked(&addend.async_groups)?;
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
    pub fn schema_version(&self) -> u32 {
        let feedback = self.memory_admission_feedback.as_deref();
        if self.limiting_cycles.pipeline_sfu != 0
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

/// Result of replaying one SM's blocks to completion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmSimResult {
    /// Exact: one per traced record fed in.
    pub instructions_issued: u64,
    /// Unadjusted modeled SM clock frontier (time of last issue + 1). An
    /// explicit observation projection may add an architecture-scoped launch
    /// boundary; that boundary is not hidden in this causal timestamp. The
    /// response tail of final loads is not counted here.
    pub cycles: u64,
    /// Latest completion of an active SM-local pipeline after scheduler issue.
    /// This remains separate from `cycles`, the issue frontier.
    pub modeled_local_completion_cycles: Option<u64>,
    /// This SM's L1 counters when an L1 model was attached.
    pub l1: Option<L1Counters>,
    /// Explicit capability gaps for write-class tails present in this replay.
    pub unmodeled_tails: TailFlags,
    /// Simulator-internal causal readiness diagnostics. These are not NCU
    /// sample counts and their overlapping exposure fields are not elapsed
    /// kernel cycles.
    pub readiness: ReadinessDiagnostics,
    /// Model-relative active-warp integral and per-SMSP issue observations.
    pub activity: SmActivity,
}

// P15 full-replay service envelope.  These are logical route-service terms
// inferred from independent width/stride shapes; they are not cache geometry,
// partition hash, MSHR count or native completion timestamps.  The service is
// admitted only for a single-warp dense launch so concurrent-warp overlap is
// not charged twice by this small alpha arm.
const SM120_FLAT_SERVICE_SECTOR_THRESHOLD: u64 = 512;
const SM120_FLAT_GLOBAL_32B_SERVICE_CYCLES_PER_SECTOR: u64 = 14;
const SM120_FLAT_GLOBAL_64B_SERVICE_CYCLES_PER_SECTOR: u64 = 10;
const SM120_FLAT_GLOBAL_128B_SERVICE_CYCLES_PER_SECTOR: u64 = 8;
const SM120_FLAT_SHARED_SERVICE_CYCLES_PER_WAVEFRONT: u64 = 7;
const SM120_FLAT_MULTI_WARP_GLOBAL_SERVICE_CYCLES_PER_SECTOR: u64 = 2;
const SM120_FLAT_MULTI_WARP_DENSE_EXTRA_CYCLES: u64 = 7;
const SM120_FLAT_MULTI_WARP_MEDIUM_EXTRA_CYCLES: u64 = 40;
const SM120_FLAT_SHARED_MEDIUM_EXTRA_CYCLES_PER_WAVEFRONT: u64 = 42;
const SM120_FLAT_ASYNC_GROUP_SERVICE_CYCLES: u64 = 1_250;
const SM120_FLAT_STORE_RETIRE_CYCLES_PER_SECTOR: u64 = 19;
const SM120_FLAT_REDUCTION_RETIRE_CYCLES_PER_SECTOR: u64 = 17;

fn sm120_flat_global_service_rate(opcode_text: &str) -> u64 {
    let opcode = opcode_text.split_ascii_whitespace().next().unwrap_or(opcode_text);
    if opcode.contains(".128") {
        SM120_FLAT_GLOBAL_128B_SERVICE_CYCLES_PER_SECTOR
    } else if opcode.contains(".64") {
        SM120_FLAT_GLOBAL_64B_SERVICE_CYCLES_PER_SECTOR
    } else {
        SM120_FLAT_GLOBAL_32B_SERVICE_CYCLES_PER_SECTOR
    }
}

fn sm120_flat_service_work(kind: AccessKind, opcode_text: &str) -> Option<(bool, u64)> {
    match kind {
        AccessKind::GlobalEligible { len, .. }
        | AccessKind::GlobalBypass { len, .. }
        | AccessKind::GlobalUntagged { len, .. }
        | AccessKind::GlobalStore { len, .. }
        | AccessKind::GlobalReduction { len, .. } if len != 0 => Some((
            true,
            u64::from(len).saturating_mul(sm120_flat_global_service_rate(opcode_text)),
        )),
        AccessKind::Shared { wavefronts } if wavefronts != 0 => Some((
            false,
            wavefronts.saturating_mul(SM120_FLAT_SHARED_SERVICE_CYCLES_PER_WAVEFRONT),
        )),
        _ => None,
    }
}

/// Group witness for one sector request belonging to an effective scalar
/// STORE/RED warp instruction. The system consumes these witnesses only to
/// reconstruct exact per-instruction L2 hit/miss work for opt-in admission
/// candidates; they do not affect hierarchy identity or ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AdmissionRequestWitness(NonZeroU128);

impl AdmissionRequestWitness {
    const VALID: u128 = 1 << 127;
    const U16_MASK: u128 = u16::MAX as u128;
    const U32_MASK: u128 = u32::MAX as u128;

    pub(crate) const fn new(
        warp: u32,
        instruction: u32,
        subcore: u8,
        request_index: u16,
        request_count: u16,
    ) -> Option<Self> {
        if subcore >= SUB_CORES as u8 || request_count == 0 || request_index >= request_count {
            return None;
        }
        let packed = Self::VALID
            | warp as u128
            | ((instruction as u128) << 32)
            | ((subcore as u128) << 64)
            | ((request_index as u128) << 66)
            | ((request_count as u128) << 82);
        match NonZeroU128::new(packed) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    pub(crate) const fn warp(self) -> u32 {
        (self.0.get() & Self::U32_MASK) as u32
    }

    pub(crate) const fn instruction(self) -> u32 {
        ((self.0.get() >> 32) & Self::U32_MASK) as u32
    }

    pub(crate) const fn subcore(self) -> u8 {
        ((self.0.get() >> 64) & 0b11) as u8
    }

    pub(crate) const fn request_index(self) -> u16 {
        ((self.0.get() >> 66) & Self::U16_MASK) as u16
    }

    pub(crate) const fn request_count(self) -> u16 {
        ((self.0.get() >> 82) & Self::U16_MASK) as u16
    }
}

/// Exact event payload for one opt-in modeled STORE/RED completion. The
/// admission witness names the instruction group; the shared-system target
/// names the owning SM, so identical warp/instruction indices on different SMs
/// cannot exchange acknowledgements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MemoryWriteAck {
    pub(crate) witness: AdmissionRequestWitness,
    pub(crate) operation: MemoryOperation,
    pub(crate) issue_at: u64,
    pub(crate) accepted_at: u64,
    pub(crate) service_done_at: u64,
    pub(crate) fence_visible_at: u64,
}

/// A memory request the SM emits in L1Shared mode. The system layer
/// schedules the L2 access at `issue_time` (the engine's global
/// `(time, seq)` order supplies the cross-SM interleaving) and routes the
/// response per `reply`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemRequest {
    pub issue_time: u64,
    /// Physical 32 B hierarchy sector id (`byte_address >> 5`).
    pub sector_id: i64,
    /// Address-space provenance of the anchor sector (`sector_id`). For folded
    /// line fills, [`Self::local_sector_mask`] carries exact per-sector
    /// provenance while this field anchors response-delay attribution. Neither
    /// field participates in L1/L2 tags, fill keys, response identity, or event
    /// ordering.
    pub origin: MemoryOrigin,
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
    /// Subset of `sector_mask` attributed to Local address-space traffic.
    /// Global provenance is the complement within `sector_mask`. Keeping the
    /// split on the folded physical request preserves one worst-of-mask reply
    /// while allowing L2 hits and DRAM fetches to be accounted per origin.
    pub local_sector_mask: u8,
    pub source: L2Source,
    /// Semantic memory operation. The shared system derives the L2 hit rule
    /// and DRAM-fetch requirement from this value and rejects incoherent
    /// `(source, operation, reply)` combinations before hierarchy mutation.
    pub operation: MemoryOperation,
    /// Static SASS marker for a regular-register constant load.  Constant
    /// loads still use the normal global allocation/cache state, but their
    /// response may cross a distinct SM-side return path on SM86.
    pub(crate) constant_load: bool,
    /// Static pressure-class marker. The broad constant marker is retained
    /// for provenance; only this narrower marker is allowed to price the
    /// measured multi-site return-service delay. Response timing is attached
    /// to each `FillWaiter`, so this request-level copy never prices a folded
    /// fill.
    pub(crate) constant_deep: bool,
    pub reply: ReplyTo,
    pub(crate) admission_witness: Option<AdmissionRequestWitness>,
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

    pub(crate) const fn boundary_identity(self) -> (u64, u64, u64) {
        (
            self.warp as u64,
            self.instruction as u64,
            self.sector_index as u64,
        )
    }
}

/// What the response to one modeled memory request must do at arrival.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReplyTo {
    /// Complete this L1 fill (installing the line/sector) and resolve its
    /// registered ordinary-barrier and/or async-copy-group targets.
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
    /// Complete one global-source sector in a warp's non-bulk async-copy
    /// group. The group may still be open or may have been committed while the
    /// request was in flight.
    AsyncCopy {
        warp: usize,
        group: AsyncCopyGroupId,
        /// An unusual static form may also expose the same completion through an
        /// ordinary dependence barrier; both counters clear atomically.
        barrier: Option<DepBarrier>,
        response_id: ResponseId,
    },
    /// No SM dependency consumes returned data. Stores and reductions are
    /// intrinsically non-returning; an ATOM may also use this when its returned
    /// value has no dependence barrier, while retaining `Atomic` request
    /// semantics and its modeled RMW fetch.
    None,
}

impl ReplyTo {
    pub(crate) const fn response_id(self) -> Option<ResponseId> {
        match self {
            Self::L1Fill { response_id, .. }
            | Self::Warp { response_id, .. }
            | Self::AsyncCopy { response_id, .. } => Some(response_id),
            Self::None => None,
        }
    }
}

impl MemRequest {
    const fn local_mask(origin: MemoryOrigin, sector_bit: u8) -> u8 {
        match origin {
            MemoryOrigin::Global => 0,
            MemoryOrigin::Local => sector_bit,
        }
    }

    fn fold_sector_origin(
        &mut self,
        origin: MemoryOrigin,
        sector_bit: u8,
        constant_load: bool,
        constant_deep: bool,
    ) {
        self.sector_mask |= sector_bit;
        if origin == MemoryOrigin::Local {
            self.local_sector_mask |= sector_bit;
        }
        self.constant_load |= constant_load;
        self.constant_deep |= constant_deep;
    }

    /// An L1 fill request (line alloc or resident-sector fetch): Read
    /// class by construction — nothing that fills the modeled L1 is a
    /// write (stores hit-test without allocating, atomics bypass). The
    /// typed constructors keep the (source, operation, reply) coupling in
    /// one place so an invalid combination is unwritable at emission
    /// sites (gate finding 2026-07-07).
    fn l1_fill(
        issue_time: u64,
        sector_id: i64,
        origin: MemoryOrigin,
        fill: FillKey,
        response_id: ResponseId,
        constant_load: bool,
        constant_deep: bool,
    ) -> Self {
        let sector_bit = 1 << (sector_id & 3) as u8;
        Self {
            issue_time,
            sector_id,
            origin,
            sector_mask: sector_bit,
            local_sector_mask: Self::local_mask(origin, sector_bit),
            source: L2Source::L1Fill,
            operation: MemoryOperation::Read,
            constant_load,
            constant_deep,
            reply: ReplyTo::L1Fill { fill, response_id },
            admission_witness: None,
        }
    }

    /// A follow-up sector fetch for a merge onto an older in-flight
    /// fill. It carries its own L1 sector-fill response token: sectored
    /// MSHRs complete per returned 32-byte child, so the late sector cannot
    /// clear its waiter when the parent's line child returns.
    fn l1_followup_fetch(
        issue_time: u64,
        sector_id: i64,
        origin: MemoryOrigin,
        fill: FillKey,
        response_id: ResponseId,
        constant_load: bool,
        constant_deep: bool,
    ) -> Self {
        let sector_bit = 1 << (sector_id & 3) as u8;
        Self {
            issue_time,
            sector_id,
            origin,
            sector_mask: sector_bit,
            local_sector_mask: Self::local_mask(origin, sector_bit),
            source: L2Source::L1Fill,
            operation: MemoryOperation::Read,
            constant_load,
            constant_deep,
            reply: ReplyTo::L1Fill { fill, response_id },
            admission_witness: None,
        }
    }

    /// A plain hierarchy store sector (STG/STL): write-class tag-match rule
    /// at the L2, intrinsically non-returning.
    fn store(issue_time: u64, sector_id: i64, origin: MemoryOrigin) -> Self {
        Self::write(issue_time, sector_id, origin, MemoryOperation::Store)
    }

    /// A non-returning hierarchy reduction sector (RED/REDG). Its write data
    /// validates the sector immediately; the event model separately charges
    /// a missing-sector RMW read when the L2 line is cold. The remaining
    /// reduction/coherence semantics stay exposed in [`TailFlags`].
    fn reduction(issue_time: u64, sector_id: i64) -> Self {
        Self::write(
            issue_time,
            sector_id,
            MemoryOrigin::Global,
            MemoryOperation::Reduction,
        )
    }

    fn write(
        issue_time: u64,
        sector_id: i64,
        origin: MemoryOrigin,
        operation: MemoryOperation,
    ) -> Self {
        let sector_bit = 1 << (sector_id & 3) as u8;
        Self {
            issue_time,
            sector_id,
            origin,
            sector_mask: sector_bit,
            local_sector_mask: Self::local_mask(origin, sector_bit),
            source: L2Source::Store,
            operation,
            constant_load: false,
            constant_deep: false,
            reply: ReplyTo::None,
            admission_witness: None,
        }
    }

    pub(crate) fn has_valid_contract(self) -> bool {
        let base = matches!(
            (self.source, self.operation, self.reply),
            (
                L2Source::L1Fill,
                MemoryOperation::Read,
                ReplyTo::L1Fill { .. } | ReplyTo::None
            ) | (
                L2Source::Bypass,
                MemoryOperation::Read,
                ReplyTo::Warp { .. } | ReplyTo::AsyncCopy { .. } | ReplyTo::None,
            ) | (
                L2Source::Untagged,
                MemoryOperation::Read,
                ReplyTo::Warp { .. } | ReplyTo::None
            ) | (
                L2Source::Bypass,
                MemoryOperation::Atomic,
                ReplyTo::Warp { .. } | ReplyTo::None
            ) | (
                L2Source::Store,
                MemoryOperation::Store | MemoryOperation::Reduction,
                ReplyTo::None,
            )
        );
        let admission = match self.admission_witness {
            None => true,
            Some(witness) => {
                self.source == L2Source::Store
                    && self.origin == MemoryOrigin::Global
                    && matches!(
                        self.operation,
                        MemoryOperation::Store | MemoryOperation::Reduction
                    )
                    && witness.request_count() != 0
                    && witness.request_index() < witness.request_count()
                    && usize::from(witness.subcore()) < SUB_CORES
            }
        };
        base && admission
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct InstructionLineKey {
    function_unique_id: i32,
    line: u64,
}

impl InstructionLineKey {
    fn following(self) -> Option<Self> {
        Some(Self {
            function_unique_id: self.function_unique_id,
            line: self.line.checked_add(1)?,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct InstructionCacheEntry {
    key: InstructionLineKey,
    ready_at: u64,
    last_access: u64,
}

/// Small deterministic set-associative tag store for the instruction path.
/// A pending line remains a tag hit with its future fill timestamp, matching
/// the merge behavior needed when many warps execute one code region.
#[derive(Debug, Clone)]
struct SetAssociativeInstructionCache {
    sets: Vec<Vec<InstructionCacheEntry>>,
    ways: usize,
    access_sequence: u64,
}

impl SetAssociativeInstructionCache {
    fn new(component: &'static str, set_count: u32, ways: u32) -> Result<Self, IssueError> {
        if set_count == 0 {
            return Err(IssueError::InstructionFrontEndGeometryInvalid {
                component,
                field: "sets",
                value: u64::from(set_count),
            });
        }
        if ways == 0 {
            return Err(IssueError::InstructionFrontEndGeometryInvalid {
                component,
                field: "ways",
                value: u64::from(ways),
            });
        }
        let set_count = usize::try_from(set_count).map_err(|_| {
            IssueError::InstructionFrontEndGeometryInvalid {
                component,
                field: "sets",
                value: u64::from(set_count),
            }
        })?;
        let ways =
            usize::try_from(ways).map_err(|_| IssueError::InstructionFrontEndGeometryInvalid {
                component,
                field: "ways",
                value: u64::from(ways),
            })?;
        Ok(Self {
            sets: (0..set_count).map(|_| Vec::with_capacity(ways)).collect(),
            ways,
            access_sequence: 0,
        })
    }

    fn set_index(&self, key: InstructionLineKey) -> usize {
        let function = u64::from(key.function_unique_id.unsigned_abs());
        usize::try_from((key.line ^ function.rotate_left(17)) % self.sets.len() as u64)
            .expect("modulo by usize-sized set count fits usize")
    }

    fn next_access_sequence(&mut self) -> Result<u64, IssueError> {
        let sequence = self.access_sequence;
        self.access_sequence =
            self.access_sequence
                .checked_add(1)
                .ok_or(IssueError::CounterOverflow {
                    counter: "instruction-cache access sequence",
                })?;
        Ok(sequence)
    }

    fn probe(&mut self, key: InstructionLineKey) -> Result<Option<u64>, IssueError> {
        let set = self.set_index(key);
        let Some(index) = self.sets[set].iter().position(|entry| entry.key == key) else {
            return Ok(None);
        };
        let sequence = self.next_access_sequence()?;
        let entry = &mut self.sets[set][index];
        entry.last_access = sequence;
        Ok(Some(entry.ready_at))
    }

    fn reserve(&mut self, key: InstructionLineKey, ready_at: u64) -> Result<(), IssueError> {
        let sequence = self.next_access_sequence()?;
        let set = self.set_index(key);
        if let Some(entry) = self.sets[set].iter_mut().find(|entry| entry.key == key) {
            entry.ready_at = entry.ready_at.min(ready_at);
            entry.last_access = sequence;
            return Ok(());
        }
        let entries = &mut self.sets[set];
        if entries.len() == self.ways {
            let victim = entries
                .iter()
                .enumerate()
                .min_by_key(|(index, entry)| (entry.last_access, *index))
                .map(|(index, _)| index)
                .ok_or(IssueError::InstructionFrontEndGeometryInvalid {
                    component: "instruction_cache",
                    field: "full-set victim cardinality",
                    value: 0,
                })?;
            entries.swap_remove(victim);
        }
        entries.push(InstructionCacheEntry {
            key,
            ready_at,
            last_access: sequence,
        });
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct StreamBufferLine {
    key: InstructionLineKey,
    ready_at: u64,
    demanded: bool,
}

#[derive(Debug, Default, Clone)]
struct InstructionStreamBuffer {
    lines: VecDeque<StreamBufferLine>,
    next_line: Option<InstructionLineKey>,
}

impl InstructionStreamBuffer {
    fn maintain(&mut self, now: u64) {
        if self
            .lines
            .front()
            .is_some_and(|line| line.demanded && line.ready_at <= now)
        {
            self.lines.pop_front();
        }
    }

    fn demand(&mut self, key: InstructionLineKey) -> Option<u64> {
        let line = self.lines.front_mut()?;
        if line.key != key {
            return None;
        }
        line.demanded = true;
        Some(line.ready_at)
    }

    fn can_restart(&self) -> bool {
        self.lines.front().is_none_or(|line| !line.demanded)
    }

    fn restart_after(&mut self, key: InstructionLineKey) {
        if !self.can_restart() {
            return;
        }
        self.lines.clear();
        self.next_line = key.following();
    }

    fn can_prefetch(&self, capacity: usize) -> bool {
        self.next_line.is_some() && self.lines.len() < capacity
    }

    fn next_prefetch(&self) -> Option<InstructionLineKey> {
        self.next_line
    }

    fn push_prefetch(&mut self, key: InstructionLineKey, ready_at: u64) -> Result<(), IssueError> {
        if self.next_line != Some(key) {
            return Err(IssueError::InstructionStreamOrder {
                expected: self
                    .next_line
                    .map(|line| (line.function_unique_id, line.line)),
                got: (key.function_unique_id, key.line),
            });
        }
        self.lines.push_back(StreamBufferLine {
            key,
            ready_at,
            demanded: false,
        });
        self.next_line = key.following();
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
enum InstructionL1RequestKind {
    Demand,
    Prefetch,
}

/// One SM's active instruction hierarchy. Lower-level cold fills use the
/// independently carded dependent-DRAM latency; L0/L1 geometry and transport
/// come from the architecture capability above.
#[derive(Debug, Clone)]
struct InstructionFrontEndState {
    config: InstructionFrontEnd,
    l0: [SetAssociativeInstructionCache; SUB_CORES],
    l1: SetAssociativeInstructionCache,
    streams: [InstructionStreamBuffer; SUB_CORES],
    l1_request_ready: u64,
    l1_reply_ready: u64,
    l0_response_ready: [u64; SUB_CORES],
    prefetch_priority: usize,
    cold_fill_latency: u64,
    diagnostics: InstructionFrontEndDiagnostics,
}

impl InstructionFrontEndState {
    fn new(config: InstructionFrontEnd, cold_fill_latency: u64) -> Result<Self, IssueError> {
        for (field, value) in [
            ("line_bytes", u64::from(config.line_bytes)),
            ("l0_to_l1_cycles", config.l0_to_l1_cycles),
            ("l1_to_l0_cycles", config.l1_to_l0_cycles),
            ("l1_request_ports", u64::from(config.l1_request_ports)),
            ("l1_reply_ports", u64::from(config.l1_reply_ports)),
            ("stream_buffer_lines", u64::from(config.stream_buffer_lines)),
            (
                "stream_buffers_per_subcore",
                u64::from(config.stream_buffers_per_subcore),
            ),
            (
                "prefetches_per_cycle",
                u64::from(config.prefetches_per_cycle),
            ),
            (
                "instruction_buffer_entries",
                u64::from(config.instruction_buffer_entries),
            ),
            ("fetch_decode_width", u64::from(config.fetch_decode_width)),
            ("fetch_to_issue_stages", config.fetch_to_issue_stages),
            ("cold_fill_latency", cold_fill_latency),
        ] {
            if value == 0 {
                return Err(IssueError::InstructionFrontEndGeometryInvalid {
                    component: "instruction_front_end",
                    field,
                    value,
                });
            }
        }
        for (field, value) in [
            ("l1_request_ports", config.l1_request_ports),
            ("l1_reply_ports", config.l1_reply_ports),
            (
                "stream_buffers_per_subcore",
                config.stream_buffers_per_subcore,
            ),
            ("prefetches_per_cycle", config.prefetches_per_cycle),
            ("fetch_decode_width", config.fetch_decode_width),
        ] {
            if value != 1 {
                return Err(IssueError::InstructionFrontEndWidthUnsupported { field, value });
            }
        }
        let l0 = [
            SetAssociativeInstructionCache::new("L0I", config.l0_sets, config.l0_ways)?,
            SetAssociativeInstructionCache::new("L0I", config.l0_sets, config.l0_ways)?,
            SetAssociativeInstructionCache::new("L0I", config.l0_sets, config.l0_ways)?,
            SetAssociativeInstructionCache::new("L0I", config.l0_sets, config.l0_ways)?,
        ];
        Ok(Self {
            config,
            l0,
            l1: SetAssociativeInstructionCache::new("L1I", config.l1_sets, config.l1_ways)?,
            streams: std::array::from_fn(|_| InstructionStreamBuffer::default()),
            l1_request_ready: 0,
            l1_reply_ready: 0,
            l0_response_ready: [0; SUB_CORES],
            prefetch_priority: 0,
            cold_fill_latency,
            diagnostics: InstructionFrontEndDiagnostics::default(),
        })
    }

    #[cfg(feature = "reference")]
    fn from_typed_boundary(
        boundary: &InstructionFrontEndBoundaryState,
    ) -> Result<Self, IssueError> {
        boundary
            .validate()
            .map_err(|_| IssueError::ReferenceMacroUnsupported {
                reason: "typed_front_end_exit_invalid",
            })?;
        let cache =
            |state: &InstructionCacheState| -> Result<SetAssociativeInstructionCache, IssueError> {
                let ways = usize::try_from(state.ways).map_err(|_| {
                    IssueError::ReferenceMacroUnsupported {
                        reason: "typed_front_end_cache_ways_out_of_range",
                    }
                })?;
                let sets = state
                    .sets
                    .iter()
                    .map(|entries| {
                        entries
                            .iter()
                            .map(|entry| InstructionCacheEntry {
                                key: InstructionLineKey {
                                    function_unique_id: entry.function_unique_id,
                                    line: entry.line,
                                },
                                ready_at: entry.ready_at,
                                last_access: entry.last_access,
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                Ok(SetAssociativeInstructionCache {
                    sets,
                    ways,
                    access_sequence: state.access_sequence,
                })
            };
        let l0 = boundary
            .l0
            .iter()
            .map(cache)
            .collect::<Result<Vec<_>, _>>()?;
        let l0: [SetAssociativeInstructionCache; SUB_CORES] =
            l0.try_into()
                .map_err(|_| IssueError::ReferenceMacroUnsupported {
                    reason: "typed_front_end_l0_shape_invalid",
                })?;
        let streams = boundary
            .streams
            .iter()
            .map(|stream| InstructionStreamBuffer {
                lines: stream
                    .lines
                    .iter()
                    .map(|line| StreamBufferLine {
                        key: InstructionLineKey {
                            function_unique_id: line.function_unique_id,
                            line: line.line,
                        },
                        ready_at: line.ready_at,
                        demanded: line.demanded,
                    })
                    .collect(),
                next_line: stream
                    .next_line
                    .map(|(function_unique_id, line)| InstructionLineKey {
                        function_unique_id,
                        line,
                    }),
            })
            .collect::<Vec<_>>();
        let streams: [InstructionStreamBuffer; SUB_CORES] =
            streams
                .try_into()
                .map_err(|_| IssueError::ReferenceMacroUnsupported {
                    reason: "typed_front_end_stream_shape_invalid",
                })?;
        let config = InstructionFrontEnd {
            line_bytes: boundary.line_bytes,
            l0_sets: u32::try_from(boundary.l0.first().map_or(0, |cache| cache.sets.len()))
                .map_err(|_| IssueError::ReferenceMacroUnsupported {
                    reason: "typed_front_end_l0_sets_out_of_range",
                })?,
            l0_ways: u32::try_from(boundary.l0.first().map_or(0, |cache| cache.ways)).map_err(
                |_| IssueError::ReferenceMacroUnsupported {
                    reason: "typed_front_end_l0_ways_out_of_range",
                },
            )?,
            l1_sets: u32::try_from(boundary.l1.sets.len()).map_err(|_| {
                IssueError::ReferenceMacroUnsupported {
                    reason: "typed_front_end_l1_sets_out_of_range",
                }
            })?,
            l1_ways: u32::try_from(boundary.l1.ways).map_err(|_| {
                IssueError::ReferenceMacroUnsupported {
                    reason: "typed_front_end_l1_ways_out_of_range",
                }
            })?,
            l0_to_l1_cycles: boundary.l0_to_l1_cycles,
            l1_to_l0_cycles: boundary.l1_to_l0_cycles,
            l1_request_ports: boundary.l1_request_ports,
            l1_reply_ports: boundary.l1_reply_ports,
            stream_buffer_lines: boundary.stream_buffer_lines,
            stream_buffers_per_subcore: boundary.stream_buffers_per_subcore,
            prefetches_per_cycle: boundary.prefetches_per_cycle,
            instruction_buffer_entries: boundary.instruction_buffer_entries,
            fetch_decode_width: boundary.fetch_decode_width,
            fetch_to_issue_stages: boundary.fetch_to_issue_stages,
        };
        Ok(Self {
            config,
            l0,
            l1: cache(&boundary.l1)?,
            streams,
            l1_request_ready: boundary.l1_request_ready,
            l1_reply_ready: boundary.l1_reply_ready,
            l0_response_ready: boundary.l0_response_ready.clone().try_into().map_err(|_| {
                IssueError::ReferenceMacroUnsupported {
                    reason: "typed_front_end_response_shape_invalid",
                }
            })?,
            prefetch_priority: usize::try_from(boundary.prefetch_priority).map_err(|_| {
                IssueError::ReferenceMacroUnsupported {
                    reason: "typed_front_end_prefetch_priority_invalid",
                }
            })?,
            cold_fill_latency: boundary.cold_fill_latency,
            diagnostics: InstructionFrontEndDiagnostics {
                instructions_fetched: boundary.instructions_fetched,
                demand_l0_hits: boundary.demand_l0_hits,
                demand_stream_buffer_hits: boundary.demand_stream_buffer_hits,
                demand_l0_misses: boundary.demand_l0_misses,
                demand_l1_hits: boundary.demand_l1_hits,
                demand_l1_misses: boundary.demand_l1_misses,
                prefetch_requests: boundary.prefetch_requests,
                prefetch_l1_hits: boundary.prefetch_l1_hits,
                prefetch_l1_misses: boundary.prefetch_l1_misses,
                stream_restarts: boundary.stream_restarts,
                maximum_fetch_latency_cycles: boundary.maximum_fetch_latency_cycles,
            },
        })
    }

    fn instruction_buffer_capacity(&self) -> Result<usize, IssueError> {
        usize::try_from(self.config.instruction_buffer_entries).map_err(|_| {
            IssueError::InstructionFrontEndGeometryInvalid {
                component: "instruction_front_end",
                field: "instruction_buffer_entries",
                value: u64::from(self.config.instruction_buffer_entries),
            }
        })
    }

    fn line_key(&self, function_unique_id: i32, pc: u32) -> InstructionLineKey {
        InstructionLineKey {
            function_unique_id,
            line: u64::from(pc) / u64::from(self.config.line_bytes),
        }
    }

    fn access_l1(
        &mut self,
        key: InstructionLineKey,
        now: u64,
        kind: InstructionL1RequestKind,
    ) -> Result<u64, IssueError> {
        let request_depart = now.max(self.l1_request_ready);
        self.l1_request_ready = checked_cycle_add(
            "instruction L1 request-port availability",
            request_depart,
            1,
        )?;
        let request_arrival = checked_cycle_add(
            "instruction L0-to-L1 transport",
            request_depart,
            self.config.l0_to_l1_cycles,
        )?;
        let line_ready = if let Some(ready_at) = self.l1.probe(key)? {
            match kind {
                InstructionL1RequestKind::Demand => {
                    self.diagnostics.demand_l1_hits = checked_u64_counter_add(
                        "instruction demand L1 hit",
                        self.diagnostics.demand_l1_hits,
                        1,
                    )?;
                }
                InstructionL1RequestKind::Prefetch => {
                    self.diagnostics.prefetch_l1_hits = checked_u64_counter_add(
                        "instruction prefetch L1 hit",
                        self.diagnostics.prefetch_l1_hits,
                        1,
                    )?;
                }
            }
            request_arrival.max(ready_at)
        } else {
            match kind {
                InstructionL1RequestKind::Demand => {
                    self.diagnostics.demand_l1_misses = checked_u64_counter_add(
                        "instruction demand L1 miss",
                        self.diagnostics.demand_l1_misses,
                        1,
                    )?;
                }
                InstructionL1RequestKind::Prefetch => {
                    self.diagnostics.prefetch_l1_misses = checked_u64_counter_add(
                        "instruction prefetch L1 miss",
                        self.diagnostics.prefetch_l1_misses,
                        1,
                    )?;
                }
            }
            let ready_at = checked_cycle_add(
                "instruction L1 cold fill",
                request_arrival,
                self.cold_fill_latency,
            )?;
            self.l1.reserve(key, ready_at)?;
            ready_at
        };
        let reply_depart = line_ready.max(self.l1_reply_ready);
        self.l1_reply_ready =
            checked_cycle_add("instruction L1 reply-port availability", reply_depart, 1)?;
        checked_cycle_add(
            "instruction L1-to-L0 transport",
            reply_depart,
            self.config.l1_to_l0_cycles,
        )
    }

    fn fetch_instruction(
        &mut self,
        subcore: usize,
        function_unique_id: i32,
        pc: u32,
        now: u64,
    ) -> Result<u64, IssueError> {
        let key = self.line_key(function_unique_id, pc);
        self.diagnostics.instructions_fetched = checked_u64_counter_add(
            "instruction front-end fetch",
            self.diagnostics.instructions_fetched,
            1,
        )?;
        let line_ready = if let Some(ready_at) = self.l0[subcore].probe(key)? {
            self.diagnostics.demand_l0_hits = checked_u64_counter_add(
                "instruction demand L0 hit",
                self.diagnostics.demand_l0_hits,
                1,
            )?;
            ready_at
        } else if let Some(ready_at) = self.streams[subcore].demand(key) {
            self.diagnostics.demand_stream_buffer_hits = checked_u64_counter_add(
                "instruction stream-buffer hit",
                self.diagnostics.demand_stream_buffer_hits,
                1,
            )?;
            self.l0[subcore].reserve(key, ready_at)?;
            ready_at
        } else {
            self.diagnostics.demand_l0_misses = checked_u64_counter_add(
                "instruction demand L0 miss",
                self.diagnostics.demand_l0_misses,
                1,
            )?;
            let ready_at = self.access_l1(key, now, InstructionL1RequestKind::Demand)?;
            self.l0[subcore].reserve(key, ready_at)?;
            if self.streams[subcore].can_restart() {
                self.streams[subcore].restart_after(key);
                self.diagnostics.stream_restarts = checked_u64_counter_add(
                    "instruction stream restart",
                    self.diagnostics.stream_restarts,
                    1,
                )?;
            }
            ready_at
        };
        let response_at = now.max(line_ready).max(self.l0_response_ready[subcore]);
        self.l0_response_ready[subcore] =
            checked_cycle_add("instruction L0 response-port availability", response_at, 1)?;
        let ready_at = checked_cycle_add(
            "instruction fetch-to-issue pipeline",
            response_at,
            self.config.fetch_to_issue_stages,
        )?;
        self.diagnostics.maximum_fetch_latency_cycles = self
            .diagnostics
            .maximum_fetch_latency_cycles
            .max(ready_at.saturating_sub(now));
        Ok(ready_at)
    }

    fn maintain(&mut self, now: u64) {
        for stream in &mut self.streams {
            stream.maintain(now);
        }
    }

    /// A demanded head can free a prefetch slot before any warp is eligible
    /// to issue. It remains a local wake source even when all instruction and
    /// stream buffers are currently full.
    fn next_maintenance_time(&self) -> Option<u64> {
        self.streams
            .iter()
            .filter_map(|stream| stream.lines.front())
            .filter(|line| line.demanded)
            .map(|line| line.ready_at)
            .min()
    }

    fn has_prefetch_work(&self) -> bool {
        let capacity = self.config.stream_buffer_lines as usize;
        self.streams
            .iter()
            .any(|stream| stream.can_prefetch(capacity))
    }

    /// Keep the normal fetch path allocation-free.  A shadow cycle is needed
    /// only near checked counter, cache-sequence, or timestamp boundaries,
    /// where a later fetch error could otherwise leave an earlier fetch
    /// partially committed.
    fn needs_transactional_cycle(&self, fetches: &[Option<usize>; SUB_CORES], now: u64) -> bool {
        let fetch_count = fetches.iter().flatten().count();
        let prefetch_count = usize::from(self.has_prefetch_work());
        let access_count = fetch_count.saturating_add(prefetch_count);
        let fetches_u64 = u64::try_from(fetch_count).unwrap_or(u64::MAX);
        let accesses_u64 = u64::try_from(access_count).unwrap_or(u64::MAX);
        let sequence_increments = accesses_u64.saturating_mul(2);
        let counter_overflows = |value: u64, amount: u64| value.checked_add(amount).is_none();

        if counter_overflows(self.diagnostics.instructions_fetched, fetches_u64)
            || counter_overflows(self.diagnostics.demand_l0_hits, fetches_u64)
            || counter_overflows(self.diagnostics.demand_stream_buffer_hits, fetches_u64)
            || counter_overflows(self.diagnostics.demand_l0_misses, fetches_u64)
            || counter_overflows(self.diagnostics.demand_l1_hits, accesses_u64)
            || counter_overflows(self.diagnostics.demand_l1_misses, accesses_u64)
            || counter_overflows(
                self.diagnostics.prefetch_requests,
                u64::try_from(prefetch_count).unwrap_or(u64::MAX),
            )
            || counter_overflows(
                self.diagnostics.prefetch_l1_hits,
                u64::try_from(prefetch_count).unwrap_or(u64::MAX),
            )
            || counter_overflows(
                self.diagnostics.prefetch_l1_misses,
                u64::try_from(prefetch_count).unwrap_or(u64::MAX),
            )
            || counter_overflows(self.diagnostics.stream_restarts, fetches_u64)
            || self
                .l1
                .access_sequence
                .checked_add(sequence_increments)
                .is_none()
            || self.l1_request_ready.checked_add(accesses_u64).is_none()
            || self.l1_reply_ready.checked_add(accesses_u64).is_none()
        {
            return true;
        }
        for (subcore, fetch) in fetches.iter().enumerate() {
            if fetch.is_some()
                && (self.l0[subcore].access_sequence.checked_add(2).is_none()
                    || self.l0_response_ready[subcore].checked_add(1).is_none())
            {
                return true;
            }
        }
        let mut maximum_delay = 0_u64;
        for delta in [
            1,
            self.config.l0_to_l1_cycles,
            self.cold_fill_latency,
            1,
            self.config.l1_to_l0_cycles,
            1,
            self.config.fetch_to_issue_stages,
        ] {
            let Some(next) = maximum_delay.checked_add(delta) else {
                return true;
            };
            maximum_delay = next;
        }
        now.checked_add(maximum_delay).is_none()
            || self.l1_request_ready.checked_add(maximum_delay).is_none()
            || self.l1_reply_ready.checked_add(maximum_delay).is_none()
    }

    fn prefetch_cycle(&mut self, now: u64) -> Result<(), IssueError> {
        let capacity = usize::try_from(self.config.stream_buffer_lines).map_err(|_| {
            IssueError::InstructionFrontEndGeometryInvalid {
                component: "instruction_front_end",
                field: "stream_buffer_lines",
                value: u64::from(self.config.stream_buffer_lines),
            }
        })?;
        let selected = (0..SUB_CORES)
            .map(|offset| (self.prefetch_priority + offset) % SUB_CORES)
            .find(|&subcore| self.streams[subcore].can_prefetch(capacity));
        let Some(subcore) = selected else {
            return Ok(());
        };
        let key =
            self.streams[subcore]
                .next_prefetch()
                .ok_or(IssueError::InstructionStreamOrder {
                    expected: None,
                    got: (0, 0),
                })?;
        self.diagnostics.prefetch_requests = checked_u64_counter_add(
            "instruction prefetch request",
            self.diagnostics.prefetch_requests,
            1,
        )?;
        let ready_at = self.access_l1(key, now, InstructionL1RequestKind::Prefetch)?;
        self.streams[subcore].push_prefetch(key, ready_at)?;
        self.prefetch_priority = (subcore + 1) % SUB_CORES;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct BufferedInstruction {
    instruction: usize,
    ready_at: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MemoryOperandWidth {
    Bits32,
    Bits64,
    Bits128,
}

fn memory_operand_width(opcode: &str) -> MemoryOperandWidth {
    if opcode.split('.').any(|modifier| modifier == "128") {
        MemoryOperandWidth::Bits128
    } else if opcode.split('.').any(|modifier| modifier == "64") {
        MemoryOperandWidth::Bits64
    } else {
        MemoryOperandWidth::Bits32
    }
}

fn memory_address_is_uniform(opcode_text: &str) -> bool {
    opcode_text
        .split_once('[')
        .and_then(|(_, tail)| tail.split_once(']'))
        .is_some_and(|(address, _)| {
            address
                .split(|character: char| !character.is_ascii_alphanumeric())
                .any(|token| token.starts_with("UR"))
        })
}

/// Identify regular-register constant-cache-qualified loads from static SASS.
/// They continue through the normal global/L1/L2 allocation path; the marker
/// only selects the SM86 constant-return service when the shared system prices
/// the response.  Keeping this classification textual avoids treating dynamic
/// trace metadata or measured NCU values as simulator inputs.
fn is_constant_load_opcode(opcode_text: &str) -> bool {
    let opcode = opcode_text.split_ascii_whitespace().next().unwrap_or("");
    opcode.split('.').any(|modifier| modifier == "CONSTANT")
        && matches!(opcode_head(opcode_text), "LDG" | "LD" | "LDC" | "LDL")
}

fn is_u16_opcode(opcode_text: &str) -> bool {
    opcode_text
        .split_ascii_whitespace()
        .next()
        .is_some_and(|opcode| opcode.split('.').any(|modifier| modifier == "U16"))
}

/// Count distinct static constant-load request streams. The released SM86
/// memory model has a five-entry local memory dispatch window (four queue
/// entries plus its dispatch latch); a stream count beyond that window is the
/// conservative point at which the L0C/GCC transport can remain occupied.
/// This is derived solely from SASS shape, never from a workload id or an NCU
/// target.
fn constant_stream_count(schedule: &FunctionSchedule) -> usize {
    schedule
        .by_pc
        .values()
        .filter(|slot| is_constant_load_opcode(slot.opcode_text()))
        .count()
}

fn deep_constant_load(
    slot: &IssueSlot,
    stream_count: usize,
    queue_entries_with_dispatch: u32,
) -> bool {
    let pressure_threshold = usize::try_from(queue_entries_with_dispatch)
        .unwrap_or(usize::MAX)
        .saturating_add(1);
    let stream_pressure = stream_count > pressure_threshold;
    let halfword_pressure = is_u16_opcode(slot.opcode_text())
        && stream_count >= pressure_threshold
        && slot
            .control
            .write_barrier
            .is_some_and(|barrier| barrier.wire() > 2);
    is_constant_load_opcode(slot.opcode_text())
        && ((!is_u16_opcode(slot.opcode_text()) && stream_pressure) || halfword_pressure)
}

fn is_u16_global_store_opcode(opcode_text: &str) -> bool {
    opcode_head(opcode_text) == "STG"
        && opcode_text
            .split_ascii_whitespace()
            .next()
            .is_some_and(|opcode| opcode.split('.').any(|modifier| modifier == "U16"))
}

/// Select the observer-only write-tag window for the narrow memory shape for
/// which it is defined: a global U16 store stream paired with constant-space
/// global loads and no shared-memory traffic.  Other kernels retain the
/// topology-only write observer until their own independent cache policy is
/// characterized; this prevents a partial policy from silently changing
/// unrelated read/write rows.
pub(crate) fn stateful_write_lookup_observer_enabled(schedule: &FunctionSchedule) -> bool {
    let mut has_constant_load = false;
    let mut has_u16_store = false;
    let mut has_other_global_store = false;
    let mut has_shared_memory = false;
    for slot in schedule.by_pc.values() {
        let head = opcode_head(slot.opcode_text());
        has_constant_load |= is_constant_load_opcode(slot.opcode_text());
        has_u16_store |= is_u16_global_store_opcode(slot.opcode_text());
        if matches!(head, "STG" | "STL") && !is_u16_global_store_opcode(slot.opcode_text()) {
            has_other_global_store = true;
        }
        has_shared_memory |= matches!(head, "LDS" | "LDSM" | "STS" | "STSM");
    }
    has_constant_load && has_u16_store && !has_other_global_store && !has_shared_memory
}

#[derive(Debug, Clone, Copy)]
struct MemoryPipelineSlot {
    operand_ready_latency: u64,
}

fn memory_pipeline_slot(
    slot: &IssueSlot,
    config: MemoryIssuePipeline,
    deep_constant: bool,
) -> Option<MemoryPipelineSlot> {
    let head = opcode_head(slot.opcode_text());
    let uniform = memory_address_is_uniform(slot.opcode_text());
    let width = memory_operand_width(
        slot.opcode_text()
            .split_ascii_whitespace()
            .next()
            .unwrap_or(""),
    );
    let store_latency = |shared: bool| match (shared, uniform, width) {
        (false, true, MemoryOperandWidth::Bits32) => config.global_store_uniform_32_war_cycles,
        (false, true, MemoryOperandWidth::Bits64) => config.global_store_uniform_64_war_cycles,
        (false, true, MemoryOperandWidth::Bits128) => config.global_store_uniform_128_war_cycles,
        (false, false, MemoryOperandWidth::Bits32) => config.global_store_regular_32_war_cycles,
        (false, false, MemoryOperandWidth::Bits64) => config.global_store_regular_64_war_cycles,
        (false, false, MemoryOperandWidth::Bits128) => config.global_store_regular_128_war_cycles,
        (true, true, MemoryOperandWidth::Bits32) => config.shared_store_uniform_32_war_cycles,
        (true, true, MemoryOperandWidth::Bits64) => config.shared_store_uniform_64_war_cycles,
        (true, true, MemoryOperandWidth::Bits128) => config.shared_store_uniform_128_war_cycles,
        (true, false, MemoryOperandWidth::Bits32) => config.shared_store_regular_32_war_cycles,
        (true, false, MemoryOperandWidth::Bits64) => config.shared_store_regular_64_war_cycles,
        (true, false, MemoryOperandWidth::Bits128) => config.shared_store_regular_128_war_cycles,
    };
    let operand_ready_latency = match head {
        "LDGSTS" => config.ldgsts_war_cycles,
        "LDG" | "LD" | "LDC" | "LDL" => {
            if deep_constant {
                config.constant_load_war_cycles
            } else if uniform {
                config.global_load_uniform_war_cycles
            } else {
                config.global_load_regular_war_cycles
            }
        }
        "STG" | "STL" => store_latency(false),
        "LDS" | "LDSM" => config.shared_load_war_cycles,
        "STS" => store_latency(true),
        _ => return None,
    };
    Some(MemoryPipelineSlot {
        operand_ready_latency,
    })
}

#[derive(Debug, Clone, Copy, Default)]
struct MemoryPipelineIssue {
    operand_ready_at: u64,
    shared_accept_at: u64,
}

#[derive(Debug, Clone)]
struct MemoryIssuePipelineState {
    config: MemoryIssuePipeline,
    queue_release_times: [VecDeque<u64>; SUB_CORES],
    local_completion_ready: [u64; SUB_CORES],
    shared_accept_ready: u64,
    diagnostics: MemoryIssuePipelineDiagnostics,
}

#[derive(Debug, Clone)]
struct MemoryIssuePipelinePlan {
    next: MemoryIssuePipelineState,
    issues: [Option<MemoryPipelineIssue>; SUB_CORES],
}

/// Scheduler-visible occupancy for the architecture's scalar INT/FP
/// execution stream.  Dispatch remains one instruction per SMSP cycle; an
/// entry reserves a future execution slot and leaves the window when that
/// slot starts.  This captures finite pipelined service without turning the
/// model into a register scoreboard or a per-cycle global scan.
/// Research-only scalar resource candidate. It is intentionally not wired
/// into production IssueModelParams; the issue-port campaign found that a
/// single generic queue does not transfer across independent and mixed
/// workloads. Keeping the implementation here makes future probes
/// reproducible without silently changing the deployed lane.
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct ScalarExecutionPipelineState {
    config: ScalarExecutionPipeline,
    release_times: [VecDeque<u64>; SUB_CORES],
    next_service: [u64; SUB_CORES],
    diagnostics: ScalarExecutionPipelineDiagnostics,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ScalarExecutionPipelineDiagnostics {
    pub instructions_admitted: u64,
    pub service_busy_cycles: u64,
    pub queue_delay_cycles: u64,
    pub maximum_queue_occupancy: u64,
}

impl ScalarExecutionPipelineDiagnostics {
    fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
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

#[derive(Debug, Clone)]
struct ScalarExecutionPipelinePlan {
    next: ScalarExecutionPipelineState,
}

#[allow(dead_code)]
impl ScalarExecutionPipelineState {
    fn new(config: ScalarExecutionPipeline) -> Result<Self, IssueError> {
        if config.latency_cycles == 0
            || config.initiation_interval_cycles == 0
            || config.in_flight_capacity == 0
        {
            return Err(IssueError::ScalarExecutionPipelineConfigurationInvalid {
                latency: config.latency_cycles,
                initiation: config.initiation_interval_cycles,
                capacity: config.in_flight_capacity,
            });
        }
        Ok(Self {
            config,
            release_times: std::array::from_fn(|_| VecDeque::new()),
            next_service: [0; SUB_CORES],
            diagnostics: ScalarExecutionPipelineDiagnostics::default(),
        })
    }

    fn prune(&mut self, now: u64) {
        for queue in &mut self.release_times {
            while queue.front().is_some_and(|&release| release <= now) {
                queue.pop_front();
            }
        }
    }

    fn admission_ready_at(&self, subcore: usize) -> u64 {
        let queue = &self.release_times[subcore];
        if queue.len() < self.config.in_flight_capacity as usize {
            0
        } else {
            queue.front().copied().unwrap_or(0)
        }
    }

    fn plan(
        &self,
        selected: [Option<PipelineClass>; SUB_CORES],
        now: u64,
    ) -> Result<ScalarExecutionPipelinePlan, IssueError> {
        let mut next = self.clone();
        for (subcore, slot) in selected.into_iter().enumerate() {
            let Some(PipelineClass::ScalarAlu {
                latency_cycles,
                initiation_interval_cycles,
                ..
            }) = slot
            else {
                continue;
            };
            if next.release_times[subcore].len() >= self.config.in_flight_capacity as usize {
                return Err(IssueError::ScalarExecutionPipelineQueueFull { subcore, time: now });
            }
            let ready_for_service = checked_cycle_add(
                "scalar execution pipeline latency",
                now,
                latency_cycles,
            )?;
            let service_start = ready_for_service.max(next.next_service[subcore]);
            next.next_service[subcore] = checked_cycle_add(
                "scalar execution pipeline initiation",
                service_start,
                initiation_interval_cycles,
            )?;
            next.release_times[subcore].push_back(service_start);
            let diagnostics = &mut next.diagnostics;
            diagnostics.instructions_admitted = checked_u64_counter_add(
                "scalar execution pipeline admitted instruction",
                diagnostics.instructions_admitted,
                1,
            )?;
            diagnostics.service_busy_cycles = checked_u64_counter_add(
                "scalar execution pipeline service work",
                diagnostics.service_busy_cycles,
                initiation_interval_cycles,
            )?;
            diagnostics.queue_delay_cycles = checked_u64_counter_add(
                "scalar execution pipeline queue delay",
                diagnostics.queue_delay_cycles,
                service_start.saturating_sub(ready_for_service),
            )?;
            diagnostics.maximum_queue_occupancy = diagnostics.maximum_queue_occupancy.max(
                u64::try_from(next.release_times[subcore].len()).map_err(|_| {
                    IssueError::CounterOverflow {
                        counter: "scalar execution pipeline queue occupancy",
                    }
                })?,
            );
        }
        Ok(ScalarExecutionPipelinePlan { next })
    }
}

impl MemoryIssuePipelineState {
    fn new(config: MemoryIssuePipeline) -> Result<Self, IssueError> {
        if config.queue_entries_with_dispatch == 0 {
            return Err(IssueError::MemoryIssuePipelineConfigurationInvalid {
                field: "queue_entries_with_dispatch",
                value: 0,
            });
        }
        for (field, value) in [
            (
                "local_initiation_interval_cycles",
                config.local_initiation_interval_cycles,
            ),
            (
                "shared_accept_interval_cycles",
                config.shared_accept_interval_cycles,
            ),
        ] {
            if value == 0 {
                return Err(IssueError::MemoryIssuePipelineConfigurationInvalid { field, value });
            }
        }
        Ok(Self {
            config,
            queue_release_times: std::array::from_fn(|_| VecDeque::new()),
            local_completion_ready: [0; SUB_CORES],
            shared_accept_ready: 0,
            diagnostics: MemoryIssuePipelineDiagnostics::default(),
        })
    }

    fn prune(&mut self, now: u64) {
        for queue in &mut self.queue_release_times {
            while queue.front().is_some_and(|&release| release <= now) {
                queue.pop_front();
            }
        }
    }

    fn admission_ready_at(&self, subcore: usize) -> u64 {
        let queue = &self.queue_release_times[subcore];
        if queue.len() < self.config.queue_entries_with_dispatch as usize {
            0
        } else {
            queue.front().copied().unwrap_or(0)
        }
    }

    fn plan(
        &self,
        selected: [Option<MemoryPipelineSlot>; SUB_CORES],
        now: u64,
    ) -> Result<MemoryIssuePipelinePlan, IssueError> {
        let mut next = self.clone();
        let mut issues = [None; SUB_CORES];
        for (subcore, slot) in selected.into_iter().enumerate() {
            let Some(slot) = slot else { continue };
            if next.queue_release_times[subcore].len()
                >= next.config.queue_entries_with_dispatch as usize
            {
                return Err(IssueError::MemoryIssuePipelineQueueFull { subcore, time: now });
            }
            let nominal_operand_ready = checked_cycle_add(
                "memory operand/address readiness",
                now,
                slot.operand_ready_latency,
            )?;
            let operand_ready_at = nominal_operand_ready.max(next.local_completion_ready[subcore]);
            next.local_completion_ready[subcore] = checked_cycle_add(
                "memory local initiation interval",
                operand_ready_at,
                next.config.local_initiation_interval_cycles,
            )?;
            let shared_accept_at = operand_ready_at.max(next.shared_accept_ready);
            next.shared_accept_ready = checked_cycle_add(
                "memory SM-shared acceptance interval",
                shared_accept_at,
                next.config.shared_accept_interval_cycles,
            )?;
            next.queue_release_times[subcore].push_back(shared_accept_at);
            let diagnostics = &mut next.diagnostics;
            diagnostics.instructions_admitted = checked_u64_counter_add(
                "memory-pipeline admitted instruction",
                diagnostics.instructions_admitted,
                1,
            )?;
            diagnostics.operand_pipeline_delay_cycles = checked_u64_counter_add(
                "memory-pipeline operand delay",
                diagnostics.operand_pipeline_delay_cycles,
                operand_ready_at - nominal_operand_ready,
            )?;
            diagnostics.shared_queue_delay_cycles = checked_u64_counter_add(
                "memory-pipeline shared queue delay",
                diagnostics.shared_queue_delay_cycles,
                shared_accept_at - operand_ready_at,
            )?;
            diagnostics.maximum_operand_pipeline_delay_cycles = diagnostics
                .maximum_operand_pipeline_delay_cycles
                .max(operand_ready_at - nominal_operand_ready);
            diagnostics.maximum_shared_queue_delay_cycles = diagnostics
                .maximum_shared_queue_delay_cycles
                .max(shared_accept_at - operand_ready_at);
            diagnostics.maximum_queue_occupancy = diagnostics.maximum_queue_occupancy.max(
                u64::try_from(next.queue_release_times[subcore].len()).map_err(|_| {
                    IssueError::CounterOverflow {
                        counter: "memory-pipeline queue occupancy",
                    }
                })?,
            );
            diagnostics.terminal_shared_accept_cycle = diagnostics
                .terminal_shared_accept_cycle
                .max(shared_accept_at);
            issues[subcore] = Some(MemoryPipelineIssue {
                operand_ready_at,
                shared_accept_at,
            });
        }
        Ok(MemoryIssuePipelinePlan { next, issues })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResultWritebackKind {
    FixedHadd2 { bank: u8 },
    L1TexStrongSmU32 { bank: u8 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PendingVariableWriteback {
    subcore: u8,
    bank: u8,
    writeback_at: u64,
    visible_at: u64,
    warp: usize,
    barrier: DepBarrier,
}

#[derive(Debug)]
struct ResultWritebackState {
    timing: FixedL1TexWritebackPriority,
    pending: Vec<PendingVariableWriteback>,
}

fn scalar_regular_destination_bank(opcode_text: &str) -> Option<u8> {
    let (_, operands) = opcode_text.trim().split_once(char::is_whitespace)?;
    let destination = operands.split(',').next()?.trim();
    let index = destination.strip_prefix('R')?.parse::<u16>().ok()?;
    Some((index & 1) as u8)
}

fn is_clocklo_issue(slot: &IssueSlot) -> bool {
    slot.is_clocklo_read()
}

fn is_gpu_membar_issue(slot: &IssueSlot) -> bool {
    slot.is_gpu_membar()
}

fn result_writeback_kind(slot: &IssueSlot, arch: ArchId) -> Option<ResultWritebackKind> {
    if arch != ArchId::Sm86 {
        return None;
    }
    if let FixedLatencyOperandStatus::Projected(projection) = slot.fixed_latency_operands()
        && projection.opcode() == FixedLatencyOpcode::Hadd2
        && let [destination] = projection.destinations()
        && let Some(bank) = destination.register().sm86_regular_bank()
    {
        return Some(ResultWritebackKind::FixedHadd2 { bank });
    }
    if slot.mem_role() == MemRole::EligibleLoad(CacheOpToken::LdgEStrongSm)
        && slot.opcode_text().split_whitespace().next() == Some("LDG.E.STRONG.SM")
    {
        return scalar_regular_destination_bank(slot.opcode_text())
            .map(|bank| ResultWritebackKind::L1TexStrongSmU32 { bank });
    }
    None
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum DependencyOrigin {
    Read,
    GlobalLoadOperandRead,
    GlobalStoreOperandRead,
    GlobalAtomicOperandRead,
    GlobalReductionOperandRead,
    NonMemoryWrite,
    SharedWrite,
    L1Tex,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ReadinessCause {
    InstructionFrontEnd,
    DependenceCounterRead,
    DependenceCounterGlobalLoadOperandRead,
    DependenceCounterGlobalStoreOperandRead,
    DependenceCounterGlobalAtomicOperandRead,
    DependenceCounterGlobalReductionOperandRead,
    DependenceCounterNonMemoryWrite,
    DependenceCounterSharedWrite,
    DependenceCounterL1Tex,
    DependenceCounterUnknownOrigin,
    AsyncSource,
    AsyncDestinationLocal,
    AsyncDestinationSmService,
    CtaBarrierRendezvous,
    PipelineBarrier,
    PipelineBranch,
    PipelineTensor,
    PipelineSfu,
    PipelineAlu,
    MemoryAdmissionQueue,
    MemoryIssueQueue,
    MemoryWriteCompletion,
    ControlStall,
}

impl From<DependencyOrigin> for ReadinessCause {
    fn from(origin: DependencyOrigin) -> Self {
        match origin {
            DependencyOrigin::Read => Self::DependenceCounterRead,
            DependencyOrigin::GlobalLoadOperandRead => Self::DependenceCounterGlobalLoadOperandRead,
            DependencyOrigin::GlobalStoreOperandRead => {
                Self::DependenceCounterGlobalStoreOperandRead
            }
            DependencyOrigin::GlobalAtomicOperandRead => {
                Self::DependenceCounterGlobalAtomicOperandRead
            }
            DependencyOrigin::GlobalReductionOperandRead => {
                Self::DependenceCounterGlobalReductionOperandRead
            }
            DependencyOrigin::NonMemoryWrite => Self::DependenceCounterNonMemoryWrite,
            DependencyOrigin::SharedWrite => Self::DependenceCounterSharedWrite,
            DependencyOrigin::L1Tex => Self::DependenceCounterL1Tex,
            DependencyOrigin::Unknown => Self::DependenceCounterUnknownOrigin,
        }
    }
}

impl ReadinessCauseCycles {
    fn add_cause(&mut self, cause: ReadinessCause, delta: u64) -> Result<(), IssueError> {
        let (field, counter): (&'static str, &mut u64) = match cause {
            ReadinessCause::InstructionFrontEnd => (
                "instruction front-end readiness",
                &mut self.instruction_front_end,
            ),
            ReadinessCause::ControlStall => ("control stall readiness", &mut self.control_stall),
            ReadinessCause::DependenceCounterRead => (
                "read-dependence counter readiness",
                &mut self.dependence_counter_read,
            ),
            ReadinessCause::DependenceCounterGlobalLoadOperandRead => (
                "global-load operand-read counter readiness",
                &mut self.dependence_counter_global_load_operand_read,
            ),
            ReadinessCause::DependenceCounterGlobalStoreOperandRead => (
                "global-store operand-read counter readiness",
                &mut self.dependence_counter_global_store_operand_read,
            ),
            ReadinessCause::DependenceCounterGlobalAtomicOperandRead => (
                "global-atomic operand-read counter readiness",
                &mut self.dependence_counter_global_atomic_operand_read,
            ),
            ReadinessCause::DependenceCounterGlobalReductionOperandRead => (
                "global-reduction operand-read counter readiness",
                &mut self.dependence_counter_global_reduction_operand_read,
            ),
            ReadinessCause::DependenceCounterNonMemoryWrite => (
                "non-memory write-dependence counter readiness",
                &mut self.dependence_counter_non_memory_write,
            ),
            ReadinessCause::DependenceCounterSharedWrite => (
                "shared write-dependence counter readiness",
                &mut self.dependence_counter_shared_write,
            ),
            ReadinessCause::DependenceCounterL1Tex => (
                "L1TEX-dependence counter readiness",
                &mut self.dependence_counter_l1tex,
            ),
            ReadinessCause::DependenceCounterUnknownOrigin => (
                "unknown-origin dependence counter readiness",
                &mut self.dependence_counter_unknown_origin,
            ),
            ReadinessCause::AsyncSource => {
                ("asynchronous-copy source readiness", &mut self.async_source)
            }
            ReadinessCause::AsyncDestinationLocal => (
                "asynchronous-copy local destination readiness",
                &mut self.async_destination_local,
            ),
            ReadinessCause::AsyncDestinationSmService => (
                "asynchronous-copy SM destination-service readiness",
                &mut self.async_destination_sm_service,
            ),
            ReadinessCause::CtaBarrierRendezvous => (
                "CTA barrier rendezvous readiness",
                &mut self.cta_barrier_rendezvous,
            ),
            ReadinessCause::PipelineBarrier => {
                ("CTA-barrier pipeline readiness", &mut self.pipeline_barrier)
            }
            ReadinessCause::PipelineBranch => {
                ("branch-resolution readiness", &mut self.pipeline_branch)
            }
            ReadinessCause::PipelineTensor => {
                ("tensor-pipeline readiness", &mut self.pipeline_tensor)
            }
            ReadinessCause::PipelineSfu => {
                ("uniform/SFU-pipeline readiness", &mut self.pipeline_sfu)
            }
            ReadinessCause::PipelineAlu => {
                ("scalar-ALU pipeline readiness", &mut self.pipeline_alu)
            }
            ReadinessCause::MemoryAdmissionQueue => (
                "memory-admission queue readiness",
                &mut self.memory_admission_queue,
            ),
            ReadinessCause::MemoryIssueQueue => (
                "memory issue-pipeline queue readiness",
                &mut self.memory_issue_queue,
            ),
            ReadinessCause::MemoryWriteCompletion => (
                "memory write-completion readiness",
                &mut self.memory_write_completion,
            ),
        };
        *counter = checked_u64_counter_add(field, *counter, delta)?;
        Ok(())
    }

    fn add_eligible_not_selected(&mut self, delta: u64) -> Result<(), IssueError> {
        self.eligible_not_selected = checked_u64_counter_add(
            "eligible-not-selected readiness",
            self.eligible_not_selected,
            delta,
        )?;
        Ok(())
    }
}

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

/// Warp-local identity of one nonempty asynchronous-copy group. The numeric
/// sequence is intentionally opaque outside the SM actor; shared-memory replies
/// may route an identity emitted by the actor but cannot construct one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AsyncCopyGroupId(u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AsyncCopyGroup {
    id: AsyncCopyGroupId,
    /// Issue time of the `LDGDEPBAR` that made this group visible to a
    /// subsequent wait. Zero is also the valid initial-cycle timestamp; the
    /// group cannot be observed here until it is committed.
    committed_at: u64,
    /// Source sectors whose shared-L2 response time is not known yet.
    pending_sources: u32,
    /// Maximum known lower-level source completion time across flat/constant
    /// copies and shared-L2 responses.
    source_completion_time: u64,
    /// Route-weighted effective payload in the architecture capability's common
    /// fixed-point work domain. This is added once per issued copy, never inferred
    /// from or duplicated across its source sectors.
    delivery_work: u64,
    /// Exact cache-all destination wavefronts accumulated across this committed
    /// group. Zero retains the payload-only/bypass path.
    destination_wavefronts: u64,
    /// The aggregate SM service is assigned once, after commit and after every
    /// source unit has a known completion time.
    destination_service_queued: bool,
    /// Latest completion assigned by the aggregate per-SM destination service.
    destination_completion_time: u64,
    /// Maximum of the warp-local dependent-completion floor and the aggregate
    /// destination-service completion. Every mutation checks this value before
    /// publishing group state, leaving scheduler readiness infallible.
    completion_time: u64,
}

/// One source-bearing bypass LDGSTS instruction retained inside its
/// asynchronous-copy group. Topology-only zero-payload copies remain in the
/// group protocol but do not enter this source-response observation sidecar.
/// The production readiness path continues to use [`AsyncCopyGroup`]'s
/// established aggregate fields; this sidecar preserves the event ordering
/// needed to evaluate a future per-copy delivery model without changing time.
#[derive(Debug, Clone, PartialEq, Eq)]
struct BypassAsyncCopyObservation {
    group: AsyncCopyGroupId,
    instruction: usize,
    pending_sources: u32,
    source_completion_time: u64,
    delivery_work: u64,
    source_cohorts: u64,
    destination_lines: u64,
    max_bank_wavefronts: u64,
    max_bypass_wavefronts: u64,
    sector_off: usize,
    sector_membership_masks: Box<[u32]>,
    cohorts: Box<[BypassAsyncCopyCohortObservation]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BypassAsyncCopyInput<'a> {
    instruction: usize,
    sector_off: usize,
    sector_len: u16,
    source_cohorts: u64,
    destination_lines: u64,
    max_bank_wavefronts: u64,
    max_bypass_wavefronts: u64,
    cohorts: Option<&'a [AsyncCopyCohortDescriptor]>,
    sector_membership_masks: Option<&'a [u32]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BypassAsyncCopyCohortObservation {
    pending_sources: u32,
    first_source_response: Option<u64>,
    last_source_response: Option<u64>,
    payload_work: u64,
    destination_lines: u64,
    bank_wavefronts: u64,
    bypass_wavefronts: u64,
    topology_penalty: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct BypassAsyncCopyProjection {
    copies: u64,
    bank_wavefront_upper_bound: u64,
    bypass_wavefront_upper_bound: u64,
    source_cohorts: u64,
    destination_lines: u64,
    max_bank_wavefronts_sum: u64,
    max_bypass_wavefronts_sum: u64,
    source_spread_cycles: u64,
    per_copy_completion_cycles: u64,
    group_aggregate_excess_cycles: u64,
    topology_candidate_penalty_cycles: u64,
    topology_candidate_completion_cycles: u64,
    topology_candidate_later_cycles: u64,
    topology_candidate_earlier_cycles: u64,
    pipeline_q1_completion_cycles: u64,
    pipeline_q1_exposure_cycles: u64,
    pipeline_q2_completion_cycles: u64,
    pipeline_q2_exposure_cycles: u64,
    pipeline_q4_completion_cycles: u64,
    pipeline_q4_exposure_cycles: u64,
    pipeline_q8_completion_cycles: u64,
    pipeline_q8_exposure_cycles: u64,
    pipeline_q16_completion_cycles: u64,
    pipeline_q16_exposure_cycles: u64,
    cohort_groups_observed: u64,
    cohort_copies_observed: u64,
    cohort_response_memberships_observed: u64,
    cohort_response_span_cycles: u64,
    cohort_topology_penalty_cycles: u64,
    cohort_candidate_completion_cycles: u64,
    cohort_candidate_exposure_cycles: u64,
    cohort_candidate_later_than_per_copy_cycles: u64,
    cohort_candidate_earlier_than_per_copy_cycles: u64,
    cohort_candidate_later_than_group_aggregate_cycles: u64,
    cohort_candidate_earlier_than_group_aggregate_cycles: u64,
    cohort_candidate_later_than_production_cycles: u64,
    cohort_candidate_earlier_than_production_cycles: u64,
    cohort_overlap_groups_observed: u64,
    cohort_response_envelope_cycles: u64,
    cohort_topology_prearrival_hidden_cycles: u64,
    cohort_topology_stream_hidden_cycles: u64,
    cohort_topology_residual_exposure_cycles: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BypassPipelineCopyCandidate {
    instruction: usize,
    payload_completion: u64,
    topology_penalty: u64,
}

// Timing-neutral local-latency candidate selected on conflict/severe and
// tested on skew. Readiness never consumes these coefficients; schema 15
// reports the resulting frontier so failed depth transfer remains visible.
const BYPASS_TOPOLOGY_CANDIDATE_WAVEFRONT_CYCLES: u64 = 4;
const BYPASS_TOPOLOGY_CANDIDATE_BANK_CYCLES: u64 = 2;
const BYPASS_PIPELINE_CANDIDATE_Q: [u64; 5] = [1, 2, 4, 8, 16];

fn bypass_topology_candidate_penalty(
    bypass_wavefronts: u64,
    bank_wavefronts: u64,
) -> Result<u64, IssueError> {
    if bypass_wavefronts == 0 || bank_wavefronts == 0 {
        return Ok(0);
    }
    let wavefront_penalty = bypass_wavefronts
        .saturating_sub(1)
        .checked_mul(BYPASS_TOPOLOGY_CANDIDATE_WAVEFRONT_CYCLES)
        .ok_or(IssueError::TimeMultiplicationOverflow {
            operation: "bypass topology candidate wavefront penalty",
            lhs: bypass_wavefronts.saturating_sub(1),
            rhs: BYPASS_TOPOLOGY_CANDIDATE_WAVEFRONT_CYCLES,
        })?;
    let bank_penalty = bank_wavefronts
        .saturating_sub(1)
        .checked_mul(BYPASS_TOPOLOGY_CANDIDATE_BANK_CYCLES)
        .ok_or(IssueError::TimeMultiplicationOverflow {
            operation: "bypass topology candidate bank penalty",
            lhs: bank_wavefronts.saturating_sub(1),
            rhs: BYPASS_TOPOLOGY_CANDIDATE_BANK_CYCLES,
        })?;
    checked_cycle_add(
        "bypass topology candidate local penalty",
        wavefront_penalty,
        bank_penalty,
    )
}

fn bypass_pipeline_candidate_completion(
    copies: &[BypassPipelineCopyCandidate],
    q: u64,
) -> Result<u64, IssueError> {
    debug_assert!(q != 0, "bypass pipeline acceptance width must be nonzero");
    let mut next_accept = 0_u64;
    let mut group_completion = 0_u64;
    for copy in copies {
        let start = copy.payload_completion.max(next_accept);
        let finish = checked_cycle_add(
            "bypass pipeline candidate per-copy completion",
            start,
            copy.topology_penalty,
        )?;
        let accept_interval = copy.topology_penalty / q + u64::from(copy.topology_penalty % q != 0);
        next_accept = checked_cycle_add(
            "bypass pipeline candidate next acceptance",
            start,
            accept_interval,
        )?;
        group_completion = group_completion.max(finish);
    }
    Ok(group_completion)
}

fn bypass_pipeline_candidate_completions(
    copies: &mut [BypassPipelineCopyCandidate],
) -> Result<[u64; BYPASS_PIPELINE_CANDIDATE_Q.len()], IssueError> {
    copies.sort_by_key(|copy| (copy.payload_completion, copy.instruction));
    let mut completions = [0_u64; BYPASS_PIPELINE_CANDIDATE_Q.len()];
    for (completion, q) in completions.iter_mut().zip(BYPASS_PIPELINE_CANDIDATE_Q) {
        *completion = bypass_pipeline_candidate_completion(copies, q)?;
    }
    Ok(completions)
}

impl AsyncCopyGroup {
    const fn new(id: AsyncCopyGroupId) -> Self {
        Self {
            id,
            committed_at: 0,
            pending_sources: 0,
            source_completion_time: 0,
            delivery_work: 0,
            destination_wavefronts: 0,
            destination_service_queued: false,
            destination_completion_time: 0,
            completion_time: 0,
        }
    }

    const fn is_complete_at(self, now: u64) -> bool {
        self.pending_sources == 0
            && (self.destination_wavefronts == 0 || self.destination_service_queued)
            && self.completion_time <= now
    }

    fn project_delivery(
        self,
        source_ready: u64,
        work_add: u64,
        destination_wavefronts_add: u64,
        params: &IssueModelParams,
    ) -> Result<(u64, u64, u64, u64), IssueError> {
        let delivery_work = checked_u64_counter_add(
            "asynchronous-copy delivery work",
            self.delivery_work,
            work_add,
        )?;
        let source_completion_time = self.source_completion_time.max(source_ready);
        let destination_wavefronts = checked_u64_counter_add(
            "asynchronous-copy destination wavefront",
            self.destination_wavefronts,
            destination_wavefronts_add,
        )?;
        let local_completion_time =
            params.ldgsts_completion_ready(source_completion_time, delivery_work)?;
        let completion_time = local_completion_time.max(self.destination_completion_time);
        Ok((
            source_completion_time,
            delivery_work,
            destination_wavefronts,
            completion_time,
        ))
    }

    fn local_completion_time(self, params: &IssueModelParams) -> Result<u64, IssueError> {
        params.ldgsts_completion_ready(self.source_completion_time, self.delivery_work)
    }

    fn limiting_readiness_cause(
        self,
        params: &IssueModelParams,
    ) -> Result<ReadinessCause, IssueError> {
        let local = self.local_completion_time(params)?;
        Ok(if self.destination_completion_time > local {
            ReadinessCause::AsyncDestinationSmService
        } else if self.delivery_work != 0 || self.destination_wavefronts != 0 {
            ReadinessCause::AsyncDestinationLocal
        } else {
            ReadinessCause::AsyncSource
        })
    }
}

/// Allocated only for warps that issue an effective asynchronous-copy source.
/// Keeping this uncommon state out of every [`WarpState`] preserves the compact
/// scheduler scan used by workloads with no `LDGSTS` traffic.
#[derive(Debug)]
struct AsyncCopyState {
    /// Non-bulk `cp.async` copies issued since the most recent commit.
    open_group: Option<AsyncCopyGroup>,
    /// Nonempty groups closed by `LDGDEPBAR`, in warp program order.
    committed_groups: VecDeque<AsyncCopyGroup>,
    /// Bypass-only per-instruction source/delivery state. Entries share the
    /// owning group's opaque identity and are removed with the waited prefix.
    bypass_copies: Vec<BypassAsyncCopyObservation>,
    /// Next process-local sequence for this warp. `None` means the complete
    /// u64 domain has been allocated; the final `u64::MAX` identity remains
    /// valid, but another nonempty group fails closed.
    next_sequence: Option<u64>,
}

impl AsyncCopyState {
    const fn new() -> Self {
        Self {
            open_group: None,
            committed_groups: VecDeque::new(),
            bypass_copies: Vec::new(),
            next_sequence: Some(0),
        }
    }
}

/// Cold sidecar indexed by warp identity. Keeping even the nullable pointer out
/// of [`WarpState`] preserves that structure's pre-async stride for scheduler
/// scans while retaining direct, deterministic lookup for source responses.
#[derive(Debug, Default)]
struct AsyncCopySlot {
    state: Option<Box<AsyncCopyState>>,
}

struct WarpState<'a> {
    insts: &'a [ReplayInst],
    sectors: &'a [i64],
    async_copy_cohorts: Option<&'a AsyncCopyCohortReplay>,
    next: usize,
    /// Dynamic trace cursor already requested by the fetch scheduler and the
    /// three-entry instruction buffer (including in-flight fetches).
    front_end_fetch_cursor: usize,
    instruction_buffer: VecDeque<BufferedInstruction>,
    next_slot: Option<&'a IssueSlot>,
    /// Issue/admission timestamp at which `next_slot` became the current
    /// instruction. Causal readiness delays are measured from this frontier.
    current_since: u64,
    admission_time: u64,
    eligible: u64,
    /// Earliest cycle at which the current instruction may be re-issued after
    /// an immediate-constant-cache miss. Zero means no pending IMC request.
    imc_wait_until: u64,
    /// Compiler-emitted `DEPBAR.LE SB1..SB5, 0x0[, {..}]` operands for the
    /// current instruction. This is cached when `next_slot` changes so
    /// scheduler scans use the ordinary barrier counter/clear path without
    /// reparsing SASS.
    ordinary_dependency_wait_mask: WaitMask,
    barrier_clear: [u64; DEP_BARRIERS],
    barrier_origin: [DependencyOrigin; DEP_BARRIERS],
    /// Outstanding UNKNOWN write-backs per barrier index (L1Shared mode):
    /// hardware barriers are counters, and a response still in flight is
    /// an increment whose decrement time is unknown — a wait on the index
    /// cannot pass, and the warp has no computable ready time, until
    /// every pending unit resolves.
    barrier_pending: [u32; DEP_BARRIERS],
    /// Full replay's dynamic fixed-latency register recurrence. Native
    /// extraction currently carries no operand values in `ReplayInst`, so
    /// alpha uses the executing warp mask conservatively (all lanes for
    /// legacy fixtures); fast replay refines this with its compact masks.
    register_ready: Vec<RegisterReadyEntry>,
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
    /// CTA-release arm for the current post-barrier instruction. This is kept
    /// separate from compiler-control eligibility so attribution cannot confuse
    /// a CTA rendezvous with a SASS stall count.
    cta_barrier_ready: u64,
    /// Completion of this warp's own arrival through the SM-shared barrier
    /// service. The CTA arm above carries only other-participant rendezvous.
    barrier_pipeline_ready: u64,
    branch_pipeline_ready: u64,
    block_barrier_dispatch: Option<u64>,
    block_barrier_arrival: Option<u64>,
    clock_first_issue: Option<u64>,
    clock_reads_issued: u64,
    done: bool,
}

#[derive(Debug, Clone, Copy)]
struct RegisterReadyEntry {
    register: SassRegister,
    lane_mask: u32,
    ready_at: u64,
}

fn issue_register_overlap(left: SassRegister, right: SassRegister) -> bool {
    if left.file() != right.file() { return false; }
    let (Some(left_index), Some(right_index)) = (left.index(), right.index()) else { return false; };
    u32::from(left_index) < u32::from(right_index) + u32::from(right.span_registers())
        && u32::from(right_index) < u32::from(left_index) + u32::from(left.span_registers())
}

fn issue_register_lane_mask(register: SassRegister, mask: u32) -> u32 {
    match register.file() {
        SassRegisterFile::Uniform | SassRegisterFile::UniformPredicate if mask != 0 => u32::MAX,
        _ => mask,
    }
}

fn issue_fixed_register_ready(
    status: &FixedLatencyOperandStatus,
    mask: u32,
    entries: &[RegisterReadyEntry],
) -> u64 {
    let FixedLatencyOperandStatus::Projected(projection) = status else { return 0; };
    let sources = projection.sources().iter().filter_map(|source| match source {
        SassSourceOperand::Register(operand) => Some(operand.register()),
        SassSourceOperand::Immediate | SassSourceOperand::ConstantBank => None,
    }).chain(projection.predicate_sources().iter().map(|operand| operand.register()))
      .chain(projection.destinations().iter().map(|operand| operand.register()));
    let mut ready = 0;
    for register in sources {
        let lanes = issue_register_lane_mask(register, mask);
        for entry in entries {
            if lanes & entry.lane_mask != 0 && issue_register_overlap(register, entry.register) {
                ready = ready.max(entry.ready_at);
            }
        }
    }
    ready
}

fn issue_record_fixed_register_writes(
    entries: &mut Vec<RegisterReadyEntry>,
    status: &FixedLatencyOperandStatus,
    mask: u32,
    ready_at: u64,
) {
    let FixedLatencyOperandStatus::Projected(projection) = status else { return; };
    for destination in projection.destinations() {
        let register = destination.register();
        if register.is_constant() { continue; }
        let lanes = issue_register_lane_mask(register, mask);
        for entry in entries.iter_mut().filter(|entry| entry.register == register) {
            entry.lane_mask &= !lanes;
        }
        entries.retain(|entry| entry.lane_mask != 0);
        if lanes != 0 { entries.push(RegisterReadyEntry { register, lane_mask: lanes, ready_at }); }
    }
}

struct BlockState {
    unfinished: usize,
    waiting: usize,
    warp_count: u32,
    shared_memory_bytes: u64,
    /// Physical slots held while this block is resident. The trace does not carry
    /// CTA lifetime, so modeled overlap may admit another block only when its slot
    /// set is disjoint from every currently resident block.
    warp_slots: Box<[i32]>,
    /// This block's warps occupy `warps[first_warp..first_warp +
    /// admitted_warps]` (admission pushes them contiguously), so barrier
    /// release scans only the block's own range.
    first_warp: usize,
    admitted_warps: usize,
    barrier_generation: u64,
    barrier_first_dispatch: Option<u64>,
    barrier_first_arrival: Option<u64>,
    /// Earliest issue time of this CTA's next barrier generation. It gates
    /// barrier instructions only; ordinary work after rendezvous is unaffected.
    barrier_rearm_ready: u64,
}

impl WarpState<'_> {
    fn can_fetch_instruction(&self, capacity: usize) -> bool {
        !self.done
            && self.front_end_fetch_cursor < self.insts.len()
            && self.instruction_buffer.len() < capacity
    }

    fn instruction_front_end_ready_at(&self) -> Option<u64> {
        let buffered = self.instruction_buffer.front()?;
        (buffered.instruction == self.next).then_some(buffered.ready_at)
    }

    fn preflight_buffered_issue(&self, warp: usize) -> Result<(), IssueError> {
        let Some(buffered) = self.instruction_buffer.front() else {
            return Err(IssueError::InstructionBufferMissing {
                warp,
                instruction: self.next,
            });
        };
        if buffered.instruction != self.next {
            return Err(IssueError::InstructionBufferOrder {
                warp,
                expected: self.next,
                got: buffered.instruction,
            });
        }
        Ok(())
    }

    fn consume_buffered_issue(&mut self, warp: usize) -> Result<(), IssueError> {
        self.preflight_buffered_issue(warp)?;
        self.instruction_buffer.pop_front();
        Ok(())
    }

    fn next_issue_retires(&self) -> Result<bool, IssueError> {
        let next = checked_usize_counter_add("warp instruction cursor", self.next, 1)?;
        Ok(next == self.insts.len())
    }
}

fn invalid_cohort_projection(
    warp: usize,
    instruction: usize,
    field: &'static str,
    expected: u64,
    got: u64,
) -> IssueError {
    IssueError::AsyncCopyCohortProjectionInvalid {
        warp,
        instruction,
        field,
        expected,
        got,
    }
}

fn build_bypass_cohort_observation(
    warp: usize,
    pending_add: u32,
    delivery_work: u64,
    input: BypassAsyncCopyInput<'_>,
    params: &IssueModelParams,
) -> Result<(Box<[u32]>, Box<[BypassAsyncCopyCohortObservation]>), IssueError> {
    let (Some(descriptors), Some(masks)) = (input.cohorts, input.sector_membership_masks) else {
        if input.cohorts.is_some() != input.sector_membership_masks.is_some() {
            return Err(invalid_cohort_projection(
                warp,
                input.instruction,
                "paired cohort descriptor/membership presence",
                1,
                0,
            ));
        }
        return Ok((Box::new([]), Box::new([])));
    };
    let sector_len = usize::from(input.sector_len);
    if masks.len() != sector_len {
        return Err(invalid_cohort_projection(
            warp,
            input.instruction,
            "source-sector membership count",
            u64::from(input.sector_len),
            u64::try_from(masks.len()).unwrap_or(u64::MAX),
        ));
    }
    if u32::from(input.sector_len) != pending_add {
        return Err(invalid_cohort_projection(
            warp,
            input.instruction,
            "pending source-sector count",
            u64::from(input.sector_len),
            u64::from(pending_add),
        ));
    }
    if descriptors.is_empty() || descriptors.len() > u32::BITS as usize {
        return Err(invalid_cohort_projection(
            warp,
            input.instruction,
            "cohort count in u32 membership domain",
            input.source_cohorts,
            u64::try_from(descriptors.len()).unwrap_or(u64::MAX),
        ));
    }
    let descriptor_count = u64::try_from(descriptors.len()).unwrap_or(u64::MAX);
    if descriptor_count != input.source_cohorts {
        return Err(invalid_cohort_projection(
            warp,
            input.instruction,
            "source cohort count",
            input.source_cohorts,
            descriptor_count,
        ));
    }

    let allowed_mask = if descriptors.len() == u32::BITS as usize {
        u32::MAX
    } else {
        (1_u32 << descriptors.len()) - 1
    };
    let mut pending = vec![0_u32; descriptors.len()];
    for &mask in masks {
        if mask == 0 || mask & !allowed_mask != 0 {
            return Err(invalid_cohort_projection(
                warp,
                input.instruction,
                "source-sector cohort mask",
                u64::from(allowed_mask),
                u64::from(mask),
            ));
        }
        for (cohort, count) in pending.iter_mut().enumerate() {
            if mask & (1_u32 << cohort) != 0 {
                *count = count.checked_add(1).ok_or(IssueError::CounterOverflow {
                    counter: "bypass cohort pending source membership",
                })?;
            }
        }
    }

    let common_rate = params.ldgsts_delivery_units_per_cycle();
    let bypass_rate = params
        .ldgsts_completion_throughput
        .bypass_payload_bytes_per_cycle;
    debug_assert!(bypass_rate != 0 && common_rate.is_multiple_of(bypass_rate));
    let work_per_byte = common_rate / bypass_rate;
    let mut payload_work_sum = 0_u64;
    let mut destination_lines_sum = 0_u64;
    let mut max_bank_wavefronts = 0_u64;
    let mut max_bypass_wavefronts = 0_u64;
    let mut cohorts = Vec::with_capacity(descriptors.len());
    for (&descriptor, pending_sources) in descriptors.iter().zip(pending) {
        if pending_sources == 0 {
            return Err(invalid_cohort_projection(
                warp,
                input.instruction,
                "cohort source-sector membership",
                1,
                0,
            ));
        }
        if descriptor.payload_bytes == 0
            || descriptor.destination_lines == 0
            || descriptor.bank_wavefronts == 0
            || descriptor.bypass_wavefronts == 0
        {
            return Err(invalid_cohort_projection(
                warp,
                input.instruction,
                "nonzero active cohort descriptor",
                1,
                0,
            ));
        }
        let payload_work = u64::from(descriptor.payload_bytes)
            .checked_mul(work_per_byte)
            .ok_or(IssueError::TimeMultiplicationOverflow {
                operation: "bypass cohort payload work",
                lhs: u64::from(descriptor.payload_bytes),
                rhs: work_per_byte,
            })?;
        payload_work_sum = checked_u64_counter_add(
            "bypass cohort payload work sum",
            payload_work_sum,
            payload_work,
        )?;
        destination_lines_sum = checked_u64_counter_add(
            "bypass cohort destination line sum",
            destination_lines_sum,
            u64::from(descriptor.destination_lines),
        )?;
        max_bank_wavefronts = max_bank_wavefronts.max(u64::from(descriptor.bank_wavefronts));
        max_bypass_wavefronts = max_bypass_wavefronts.max(u64::from(descriptor.bypass_wavefronts));
        cohorts.push(BypassAsyncCopyCohortObservation {
            pending_sources,
            first_source_response: None,
            last_source_response: None,
            payload_work,
            destination_lines: u64::from(descriptor.destination_lines),
            bank_wavefronts: u64::from(descriptor.bank_wavefronts),
            bypass_wavefronts: u64::from(descriptor.bypass_wavefronts),
            topology_penalty: bypass_topology_candidate_penalty(
                u64::from(descriptor.bypass_wavefronts),
                u64::from(descriptor.bank_wavefronts),
            )?,
        });
    }
    if delivery_work != payload_work_sum {
        return Err(invalid_cohort_projection(
            warp,
            input.instruction,
            "exact cohort payload work",
            delivery_work,
            payload_work_sum,
        ));
    }
    // The compact aggregate also includes executing source-ignored (zero-fill)
    // destination lanes.  Those lanes deliberately have no global source
    // response and therefore do not appear in the sparse source-cohort sidecar.
    // Require the detailed source-bearing projection to fit inside the compact
    // aggregate instead of requiring equality that would reject valid mixed
    // source/zero-fill instructions.  The residual remains topology-only and
    // is not converted into a completion timestamp.
    for (field, aggregate, source_only) in [
        (
            "source cohort destination line sum exceeds aggregate",
            input.destination_lines,
            destination_lines_sum,
        ),
        (
            "source cohort maximum bank wavefronts exceeds aggregate",
            input.max_bank_wavefronts,
            max_bank_wavefronts,
        ),
        (
            "source cohort maximum bypass wavefronts exceeds aggregate",
            input.max_bypass_wavefronts,
            max_bypass_wavefronts,
        ),
    ] {
        if source_only > aggregate {
            return Err(invalid_cohort_projection(
                warp,
                input.instruction,
                field,
                aggregate,
                source_only,
            ));
        }
    }
    Ok((masks.into(), cohorts.into_boxed_slice()))
}

impl AsyncCopySlot {
    fn preflight_async_copy_add(
        &self,
        warp: usize,
        pending_add: u32,
        source_ready: u64,
        delivery_work: u64,
        destination_wavefronts: u64,
        bypass_copy: Option<BypassAsyncCopyInput<'_>>,
        params: &IssueModelParams,
    ) -> Result<AsyncCopyGroupId, IssueError> {
        let (id, current, group) = match self.state.as_deref() {
            Some(state) => match state.open_group {
                Some(group) => (group.id, group.pending_sources, group),
                None => {
                    let id = AsyncCopyGroupId(
                        state
                            .next_sequence
                            .ok_or(IssueError::AsyncCopyGroupSequenceExhausted { warp })?,
                    );
                    (id, 0, AsyncCopyGroup::new(id))
                }
            },
            None => {
                let id = AsyncCopyGroupId(0);
                (id, 0, AsyncCopyGroup::new(id))
            }
        };
        current
            .checked_add(pending_add)
            .ok_or(IssueError::AsyncCopyPendingOverflow {
                warp,
                group: id,
                current,
                add: pending_add,
            })?;
        if let Some(copy) = bypass_copy
            && self.state.as_deref().is_some_and(|state| {
                state.bypass_copies.iter().any(|existing| {
                    existing.group == id && existing.instruction == copy.instruction
                })
            })
        {
            return Err(IssueError::AsyncCopyObservationDuplicate {
                warp,
                group: id,
                instruction: copy.instruction,
            });
        }
        if let Some(copy) = bypass_copy {
            build_bypass_cohort_observation(warp, pending_add, delivery_work, copy, params)?;
        }
        group.project_delivery(source_ready, delivery_work, destination_wavefronts, params)?;
        Ok(id)
    }

    fn ensure_open_async_group(&mut self, warp: usize) -> Result<&mut AsyncCopyGroup, IssueError> {
        let state = self
            .state
            .get_or_insert_with(|| Box::new(AsyncCopyState::new()));
        if state.open_group.is_none() {
            let sequence = state
                .next_sequence
                .ok_or(IssueError::AsyncCopyGroupSequenceExhausted { warp })?;
            state.next_sequence = sequence.checked_add(1);
            state.open_group = Some(AsyncCopyGroup::new(AsyncCopyGroupId(sequence)));
        }
        state
            .open_group
            .as_mut()
            .ok_or(IssueError::AsyncCopyGroupSequenceExhausted { warp })
    }

    fn add_known_async_copy(
        &mut self,
        warp: usize,
        source_ready: u64,
        delivery_work: u64,
        destination_wavefronts: u64,
        params: &IssueModelParams,
    ) -> Result<AsyncCopyGroupId, IssueError> {
        let id = self.preflight_async_copy_add(
            warp,
            0,
            source_ready,
            delivery_work,
            destination_wavefronts,
            None,
            params,
        )?;
        let group = self.ensure_open_async_group(warp)?;
        debug_assert_eq!(group.id, id);
        let (source_completion_time, delivery_work, destination_wavefronts, completion_time) =
            group.project_delivery(source_ready, delivery_work, destination_wavefronts, params)?;
        group.source_completion_time = source_completion_time;
        group.delivery_work = delivery_work;
        group.destination_wavefronts = destination_wavefronts;
        group.completion_time = completion_time;
        Ok(id)
    }

    fn add_pending_async_copy_sources(
        &mut self,
        warp: usize,
        add: u32,
        delivery_work: u64,
        destination_wavefronts: u64,
        bypass_copy: Option<BypassAsyncCopyInput<'_>>,
        params: &IssueModelParams,
    ) -> Result<AsyncCopyGroupId, IssueError> {
        let id = self.preflight_async_copy_add(
            warp,
            add,
            0,
            delivery_work,
            destination_wavefronts,
            bypass_copy,
            params,
        )?;
        let group = self.ensure_open_async_group(warp)?;
        debug_assert_eq!(group.id, id);
        let current = group.pending_sources;
        let pending_sources =
            current
                .checked_add(add)
                .ok_or(IssueError::AsyncCopyPendingOverflow {
                    warp,
                    group: id,
                    current,
                    add,
                })?;
        let (
            source_completion_time,
            aggregate_delivery_work,
            destination_wavefronts,
            completion_time,
        ) = group.project_delivery(0, delivery_work, destination_wavefronts, params)?;
        group.pending_sources = pending_sources;
        group.source_completion_time = source_completion_time;
        group.delivery_work = aggregate_delivery_work;
        group.destination_wavefronts = destination_wavefronts;
        group.completion_time = completion_time;
        if let Some(copy) = bypass_copy {
            let (sector_membership_masks, cohorts) =
                build_bypass_cohort_observation(warp, add, delivery_work, copy, params)?;
            if delivery_work != 0 {
                self.state
                    .as_deref_mut()
                    .expect("an observed bypass copy has allocated async-copy state")
                    .bypass_copies
                    .push(BypassAsyncCopyObservation {
                        group: id,
                        instruction: copy.instruction,
                        pending_sources: add,
                        source_completion_time: 0,
                        delivery_work,
                        source_cohorts: copy.source_cohorts,
                        destination_lines: copy.destination_lines,
                        max_bank_wavefronts: copy.max_bank_wavefronts,
                        max_bypass_wavefronts: copy.max_bypass_wavefronts,
                        sector_off: copy.sector_off,
                        sector_membership_masks,
                        cohorts,
                    });
            }
        }
        Ok(id)
    }

    fn async_group(&self, id: AsyncCopyGroupId) -> Option<&AsyncCopyGroup> {
        let state = self.state.as_deref()?;
        state
            .open_group
            .as_ref()
            .filter(|group| group.id == id)
            .or_else(|| state.committed_groups.iter().find(|group| group.id == id))
    }

    fn async_group_mut(&mut self, id: AsyncCopyGroupId) -> Option<&mut AsyncCopyGroup> {
        let state = self.state.as_deref_mut()?;
        if state
            .open_group
            .as_ref()
            .is_some_and(|group| group.id == id)
        {
            return state.open_group.as_mut();
        }
        state
            .committed_groups
            .iter_mut()
            .find(|group| group.id == id)
    }

    fn preflight_async_copy_source_completion(
        &self,
        warp: usize,
        id: AsyncCopyGroupId,
        bypass_response: Option<ResponseId>,
        remove: u32,
        now: u64,
        params: &IssueModelParams,
    ) -> Result<(), IssueError> {
        let group = self
            .async_group(id)
            .ok_or(IssueError::AsyncCopyGroupNotFound { warp, group: id })?;
        group
            .pending_sources
            .checked_sub(remove)
            .ok_or(IssueError::AsyncCopyPendingUnderflow {
                warp,
                group: id,
                current: group.pending_sources,
                remove,
            })?;
        group.project_delivery(now, 0, 0, params)?;
        if let Some(response) = bypass_response {
            let instruction = response.instruction;
            let copy = self
                .state
                .as_deref()
                .and_then(|state| {
                    state
                        .bypass_copies
                        .iter()
                        .find(|copy| copy.group == id && copy.instruction == instruction)
                })
                .ok_or(IssueError::AsyncCopyObservationNotFound {
                    warp,
                    group: id,
                    instruction,
                })?;
            copy.pending_sources.checked_sub(remove).ok_or(
                IssueError::AsyncCopyObservationPendingUnderflow {
                    warp,
                    group: id,
                    instruction,
                    current: copy.pending_sources,
                    remove,
                },
            )?;
            params.ldgsts_completion_ready(
                copy.source_completion_time.max(now),
                copy.delivery_work,
            )?;
            if !copy.cohorts.is_empty() {
                let sector_end = copy
                    .sector_off
                    .checked_add(copy.sector_membership_masks.len())
                    .ok_or(IssueError::CounterOverflow {
                        counter: "bypass cohort membership span",
                    })?;
                let relative = response
                    .sector_index
                    .checked_sub(copy.sector_off)
                    .filter(|&index| index < copy.sector_membership_masks.len())
                    .ok_or(IssueError::AsyncCopyCohortSectorOutOfRange {
                        warp,
                        group: id,
                        instruction,
                        sector_index: response.sector_index,
                        sector_off: copy.sector_off,
                        sector_end,
                    })?;
                let membership = copy.sector_membership_masks[relative];
                for (cohort_index, cohort) in copy.cohorts.iter().enumerate() {
                    if membership & (1_u32 << cohort_index) == 0 {
                        continue;
                    }
                    let pending = cohort.pending_sources.checked_sub(1).ok_or(
                        IssueError::AsyncCopyCohortPendingUnderflow {
                            warp,
                            group: id,
                            instruction,
                            sector_index: response.sector_index,
                            cohort: cohort_index,
                        },
                    )?;
                    let first = cohort
                        .first_source_response
                        .map_or(now, |first| first.min(now));
                    let last = cohort
                        .last_source_response
                        .map_or(now, |last| last.max(now));
                    if pending == 0 {
                        params.ldgsts_completion_ready(last, cohort.payload_work)?;
                        checked_cycle_add(
                            "bypass cohort topology-ready candidate",
                            first,
                            cohort.topology_penalty,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    fn complete_async_copy_sources(
        &mut self,
        warp: usize,
        id: AsyncCopyGroupId,
        bypass_response: Option<ResponseId>,
        remove: u32,
        now: u64,
        params: &IssueModelParams,
    ) -> Result<(), IssueError> {
        self.preflight_async_copy_source_completion(
            warp,
            id,
            bypass_response,
            remove,
            now,
            params,
        )?;
        let group = self
            .async_group_mut(id)
            .ok_or(IssueError::AsyncCopyGroupNotFound { warp, group: id })?;
        let current = group.pending_sources;
        let pending_sources =
            current
                .checked_sub(remove)
                .ok_or(IssueError::AsyncCopyPendingUnderflow {
                    warp,
                    group: id,
                    current,
                    remove,
                })?;
        let (source_completion_time, delivery_work, destination_wavefronts, completion_time) =
            group.project_delivery(now, 0, 0, params)?;
        group.pending_sources = pending_sources;
        group.source_completion_time = source_completion_time;
        group.delivery_work = delivery_work;
        group.destination_wavefronts = destination_wavefronts;
        group.completion_time = completion_time;
        if let Some(response) = bypass_response {
            let instruction = response.instruction;
            let copy = self
                .state
                .as_deref_mut()
                .and_then(|state| {
                    state
                        .bypass_copies
                        .iter_mut()
                        .find(|copy| copy.group == id && copy.instruction == instruction)
                })
                .expect("bypass observation was validated before aggregate mutation");
            copy.pending_sources -= remove;
            copy.source_completion_time = copy.source_completion_time.max(now);
            if !copy.cohorts.is_empty() {
                let relative = response.sector_index - copy.sector_off;
                let membership = copy.sector_membership_masks[relative];
                for (cohort_index, cohort) in copy.cohorts.iter_mut().enumerate() {
                    if membership & (1_u32 << cohort_index) == 0 {
                        continue;
                    }
                    cohort.pending_sources -= 1;
                    cohort.first_source_response = Some(
                        cohort
                            .first_source_response
                            .map_or(now, |first| first.min(now)),
                    );
                    cohort.last_source_response = Some(
                        cohort
                            .last_source_response
                            .map_or(now, |last| last.max(now)),
                    );
                }
            }
        }
        Ok(())
    }

    fn bypass_copy_projection(
        &self,
        warp: usize,
        group: &AsyncCopyGroup,
        params: &IssueModelParams,
        wait_base: u64,
    ) -> Result<Option<BypassAsyncCopyProjection>, IssueError> {
        let Some(state) = self.state.as_deref() else {
            return Ok(None);
        };
        let mut copies = 0_u64;
        let mut bank_wavefront_upper_bound = 0_u64;
        let mut bypass_wavefront_upper_bound = 0_u64;
        let mut source_cohorts = 0_u64;
        let mut destination_lines = 0_u64;
        let mut max_bank_wavefronts_sum = 0_u64;
        let mut max_bypass_wavefronts_sum = 0_u64;
        let mut earliest_source = u64::MAX;
        let mut latest_source = 0_u64;
        let mut per_copy_completion = 0_u64;
        let mut topology_candidate_completion = 0_u64;
        let mut topology_candidate_penalty_cycles = 0_u64;
        let mut pipeline_copies = Vec::new();
        let mut cohort_copies = 0_u64;
        let mut cohort_response_memberships = 0_u64;
        let mut cohort_response_span = 0_u64;
        let mut cohort_topology_penalty = 0_u64;
        let mut cohort_candidate_completion = 0_u64;
        // Each physical response envelope is counted once per copy. Hidden
        // work sums every cohort. Residual exposure instead follows the frozen
        // score's cohort -> copy -> group maximum; cross-group aggregation
        // happens in readiness diagnostics.
        let mut cohort_response_envelope = 0_u64;
        let mut cohort_topology_prearrival_hidden = 0_u64;
        let mut cohort_topology_stream_hidden = 0_u64;
        let mut cohort_topology_residual_exposure = 0_u64;
        for copy in state
            .bypass_copies
            .iter()
            .filter(|copy| copy.group == group.id)
        {
            if copy.pending_sources != 0 {
                return Err(IssueError::AsyncCopyObservationPendingAtWait {
                    warp,
                    group: group.id,
                    instruction: copy.instruction,
                    pending: copy.pending_sources,
                });
            }
            copies = checked_u64_counter_add("observed bypass asynchronous-copy count", copies, 1)?;
            source_cohorts = checked_u64_counter_add(
                "observed bypass source cohorts",
                source_cohorts,
                copy.source_cohorts,
            )?;
            destination_lines = checked_u64_counter_add(
                "observed bypass destination lines",
                destination_lines,
                copy.destination_lines,
            )?;
            max_bank_wavefronts_sum = checked_u64_counter_add(
                "observed bypass maximum bank wavefronts",
                max_bank_wavefronts_sum,
                copy.max_bank_wavefronts,
            )?;
            max_bypass_wavefronts_sum = checked_u64_counter_add(
                "observed bypass maximum projected wavefronts",
                max_bypass_wavefronts_sum,
                copy.max_bypass_wavefronts,
            )?;
            let bank_upper = copy
                .source_cohorts
                .checked_mul(copy.max_bank_wavefronts)
                .ok_or(IssueError::TimeMultiplicationOverflow {
                    operation: "bypass observed bank-wavefront upper bound",
                    lhs: copy.source_cohorts,
                    rhs: copy.max_bank_wavefronts,
                })?;
            bank_wavefront_upper_bound = checked_u64_counter_add(
                "bypass observed bank-wavefront upper bound",
                bank_wavefront_upper_bound,
                bank_upper,
            )?;
            let bypass_upper = copy
                .source_cohorts
                .checked_mul(copy.max_bypass_wavefronts)
                .ok_or(IssueError::TimeMultiplicationOverflow {
                    operation: "bypass observed projected-wavefront upper bound",
                    lhs: copy.source_cohorts,
                    rhs: copy.max_bypass_wavefronts,
                })?;
            bypass_wavefront_upper_bound = checked_u64_counter_add(
                "bypass observed projected-wavefront upper bound",
                bypass_wavefront_upper_bound,
                bypass_upper,
            )?;
            earliest_source = earliest_source.min(copy.source_completion_time);
            latest_source = latest_source.max(copy.source_completion_time);
            let payload_completion =
                params.ldgsts_completion_ready(copy.source_completion_time, copy.delivery_work)?;
            per_copy_completion = per_copy_completion.max(payload_completion);
            // A topology-only ignore-src copy has no source response stream;
            // retain its topology counters, but do not synthesize a bypass
            // candidate penalty from a zero-payload group.
            let topology_penalty = if copy.delivery_work == 0 {
                0
            } else {
                bypass_topology_candidate_penalty(
                    copy.max_bypass_wavefronts,
                    copy.max_bank_wavefronts,
                )?
            };
            topology_candidate_penalty_cycles = checked_u64_counter_add(
                "bypass topology candidate penalty diagnostic",
                topology_candidate_penalty_cycles,
                topology_penalty,
            )?;
            let candidate_completion = checked_cycle_add(
                "bypass topology candidate per-copy completion",
                payload_completion,
                topology_penalty,
            )?;
            topology_candidate_completion = topology_candidate_completion.max(candidate_completion);
            pipeline_copies.push(BypassPipelineCopyCandidate {
                instruction: copy.instruction,
                payload_completion,
                topology_penalty,
            });
            if !copy.cohorts.is_empty() {
                cohort_copies =
                    checked_u64_counter_add("observed bypass cohort copy count", cohort_copies, 1)?;
                let mut cohort_anchor: Option<u64> = None;
                let mut cohort_envelope_end = 0_u64;
                for (cohort_index, cohort) in copy.cohorts.iter().enumerate() {
                    if cohort.pending_sources != 0 {
                        return Err(IssueError::AsyncCopyCohortPendingAtWait {
                            warp,
                            group: group.id,
                            instruction: copy.instruction,
                            cohort: cohort_index,
                            pending: cohort.pending_sources,
                        });
                    }
                    let first = cohort.first_source_response.ok_or(
                        IssueError::AsyncCopyCohortProjectionInvalid {
                            warp,
                            instruction: copy.instruction,
                            field: "cohort first source response",
                            expected: 1,
                            got: 0,
                        },
                    )?;
                    let last = cohort.last_source_response.ok_or(
                        IssueError::AsyncCopyCohortProjectionInvalid {
                            warp,
                            instruction: copy.instruction,
                            field: "cohort last source response",
                            expected: 1,
                            got: 0,
                        },
                    )?;
                    if last < first {
                        return Err(IssueError::AsyncCopyCohortProjectionInvalid {
                            warp,
                            instruction: copy.instruction,
                            field: "cohort response order",
                            expected: first,
                            got: last,
                        });
                    }
                    cohort_anchor = Some(cohort_anchor.map_or(first, |anchor| anchor.min(first)));
                    cohort_envelope_end = cohort_envelope_end.max(last);
                }
                let cohort_anchor = cohort_anchor.expect("active cohort list is nonempty");

                let mut copy_residual_exposure = 0_u64;
                for (cohort_index, cohort) in copy.cohorts.iter().enumerate() {
                    let first = cohort
                        .first_source_response
                        .expect("cohort response presence was validated");
                    let last = cohort.last_source_response.ok_or(
                        IssueError::AsyncCopyCohortProjectionInvalid {
                            warp,
                            instruction: copy.instruction,
                            field: "cohort last source response",
                            expected: 1,
                            got: 0,
                        },
                    )?;
                    let prearrival_window = first - cohort_anchor;
                    let response_stream = last - first;
                    let prearrival_hidden = cohort.topology_penalty.min(prearrival_window);
                    let topology_after_prearrival = cohort.topology_penalty - prearrival_hidden;
                    let stream_hidden = topology_after_prearrival.min(response_stream);
                    let residual_exposure = topology_after_prearrival - stream_hidden;
                    cohort_topology_prearrival_hidden = checked_u64_counter_add(
                        "observed bypass cohort topology prearrival hidden work",
                        cohort_topology_prearrival_hidden,
                        prearrival_hidden,
                    )?;
                    cohort_topology_stream_hidden = checked_u64_counter_add(
                        "observed bypass cohort topology stream hidden work",
                        cohort_topology_stream_hidden,
                        stream_hidden,
                    )?;
                    copy_residual_exposure = copy_residual_exposure.max(residual_exposure);
                    let memberships = copy
                        .sector_membership_masks
                        .iter()
                        .filter(|&&mask| mask & (1_u32 << cohort_index) != 0)
                        .count();
                    cohort_response_memberships = checked_u64_counter_add(
                        "observed bypass cohort response memberships",
                        cohort_response_memberships,
                        u64::try_from(memberships).map_err(|_| IssueError::CounterOverflow {
                            counter: "observed bypass cohort response memberships",
                        })?,
                    )?;
                    cohort_response_span = checked_u64_counter_add(
                        "observed bypass cohort response span",
                        cohort_response_span,
                        last - first,
                    )?;
                    cohort_topology_penalty = checked_u64_counter_add(
                        "observed bypass cohort topology penalty",
                        cohort_topology_penalty,
                        cohort.topology_penalty,
                    )?;
                    let payload_ready =
                        params.ldgsts_completion_ready(last, cohort.payload_work)?;
                    let topology_ready = checked_cycle_add(
                        "bypass cohort topology-ready candidate",
                        first,
                        cohort.topology_penalty,
                    )?;
                    cohort_candidate_completion =
                        cohort_candidate_completion.max(payload_ready.max(topology_ready));
                }
                cohort_response_envelope = checked_u64_counter_add(
                    "observed bypass cohort response envelope",
                    cohort_response_envelope,
                    cohort_envelope_end - cohort_anchor,
                )?;
                cohort_topology_residual_exposure =
                    cohort_topology_residual_exposure.max(copy_residual_exposure);
            }
        }
        if copies == 0 {
            return Ok(None);
        }
        let pipeline_candidate_completion =
            bypass_pipeline_candidate_completions(&mut pipeline_copies)?;
        let aggregate_completion = group.local_completion_time(params)?;
        debug_assert!(
            aggregate_completion >= per_copy_completion,
            "group-aggregate delivery must upper-bound independent per-copy delivery"
        );
        Ok(Some(BypassAsyncCopyProjection {
            copies,
            bank_wavefront_upper_bound,
            bypass_wavefront_upper_bound,
            source_cohorts,
            destination_lines,
            max_bank_wavefronts_sum,
            max_bypass_wavefronts_sum,
            source_spread_cycles: latest_source.saturating_sub(earliest_source),
            per_copy_completion_cycles: per_copy_completion.saturating_sub(group.committed_at),
            group_aggregate_excess_cycles: aggregate_completion.saturating_sub(per_copy_completion),
            topology_candidate_penalty_cycles,
            topology_candidate_completion_cycles: topology_candidate_completion
                .saturating_sub(group.committed_at),
            topology_candidate_later_cycles: topology_candidate_completion
                .saturating_sub(aggregate_completion),
            topology_candidate_earlier_cycles: aggregate_completion
                .saturating_sub(topology_candidate_completion),
            pipeline_q1_completion_cycles: pipeline_candidate_completion[0]
                .saturating_sub(group.committed_at),
            pipeline_q1_exposure_cycles: pipeline_candidate_completion[0].saturating_sub(wait_base),
            pipeline_q2_completion_cycles: pipeline_candidate_completion[1]
                .saturating_sub(group.committed_at),
            pipeline_q2_exposure_cycles: pipeline_candidate_completion[1].saturating_sub(wait_base),
            pipeline_q4_completion_cycles: pipeline_candidate_completion[2]
                .saturating_sub(group.committed_at),
            pipeline_q4_exposure_cycles: pipeline_candidate_completion[2].saturating_sub(wait_base),
            pipeline_q8_completion_cycles: pipeline_candidate_completion[3]
                .saturating_sub(group.committed_at),
            pipeline_q8_exposure_cycles: pipeline_candidate_completion[3].saturating_sub(wait_base),
            pipeline_q16_completion_cycles: pipeline_candidate_completion[4]
                .saturating_sub(group.committed_at),
            pipeline_q16_exposure_cycles: pipeline_candidate_completion[4]
                .saturating_sub(wait_base),
            cohort_groups_observed: u64::from(cohort_copies != 0),
            cohort_copies_observed: cohort_copies,
            cohort_response_memberships_observed: cohort_response_memberships,
            cohort_response_span_cycles: cohort_response_span,
            cohort_topology_penalty_cycles: cohort_topology_penalty,
            cohort_candidate_completion_cycles: cohort_candidate_completion
                .saturating_sub(group.committed_at),
            cohort_candidate_exposure_cycles: cohort_candidate_completion.saturating_sub(wait_base),
            cohort_candidate_later_than_per_copy_cycles: cohort_candidate_completion
                .saturating_sub(per_copy_completion),
            cohort_candidate_earlier_than_per_copy_cycles: per_copy_completion
                .saturating_sub(cohort_candidate_completion),
            cohort_candidate_later_than_group_aggregate_cycles: cohort_candidate_completion
                .saturating_sub(aggregate_completion),
            cohort_candidate_earlier_than_group_aggregate_cycles: aggregate_completion
                .saturating_sub(cohort_candidate_completion),
            cohort_candidate_later_than_production_cycles: cohort_candidate_completion
                .saturating_sub(group.completion_time),
            cohort_candidate_earlier_than_production_cycles: group
                .completion_time
                .saturating_sub(cohort_candidate_completion),
            cohort_overlap_groups_observed: u64::from(cohort_copies != 0),
            cohort_response_envelope_cycles: cohort_response_envelope,
            cohort_topology_prearrival_hidden_cycles: cohort_topology_prearrival_hidden,
            cohort_topology_stream_hidden_cycles: cohort_topology_stream_hidden,
            cohort_topology_residual_exposure_cycles: cohort_topology_residual_exposure,
        }))
    }

    fn commit_async_copy_group(&mut self, now: u64) -> bool {
        let Some(state) = self.state.as_deref_mut() else {
            return false;
        };
        if let Some(mut group) = state.open_group.take() {
            group.committed_at = now;
            state.committed_groups.push_back(group);
            true
        } else {
            false
        }
    }

    fn async_wait_prefix_len(&self, max_pending_groups: u32) -> usize {
        let retained = usize::try_from(max_pending_groups).unwrap_or(usize::MAX);
        self.state.as_deref().map_or(0, |state| {
            state.committed_groups.len().saturating_sub(retained)
        })
    }

    /// Earliest known completion of every group older than the newest `N`.
    /// `None` means at least one required group still has a shared-L2 response
    /// outstanding, so only an external response event can supply a wake time.
    #[cold]
    #[inline(never)]
    fn async_wait_clear_time(&self, max_pending_groups: u32) -> Option<u64> {
        let Some(state) = self.state.as_deref() else {
            return Some(0);
        };
        let mut ready = 0;
        for group in state
            .committed_groups
            .iter()
            .take(self.async_wait_prefix_len(max_pending_groups))
        {
            if group.pending_sources != 0 {
                return None;
            }
            if group.destination_wavefronts != 0 && !group.destination_service_queued {
                return None;
            }
            ready = ready.max(group.completion_time);
        }
        Some(ready)
    }

    fn consume_async_wait(&mut self, max_pending_groups: u32, now: u64) {
        let consume = self.async_wait_prefix_len(max_pending_groups);
        if consume == 0 {
            return;
        }
        let state = self
            .state
            .as_deref_mut()
            .expect("a nonempty wait prefix requires async-copy state");
        debug_assert!(
            state
                .committed_groups
                .iter()
                .take(consume)
                .all(|group| group.is_complete_at(now)),
            "scheduler issued an asynchronous-copy wait before its required prefix completed"
        );
        let consumed_through = state.committed_groups[consume - 1].id.0;
        state
            .bypass_copies
            .retain(|copy| copy.group.0 > consumed_through);
        state.committed_groups.drain(..consume);
    }
}

impl WarpState<'_> {
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
        if now >= self.barrier_clear[b] {
            self.barrier_clear[b] = now;
            self.barrier_origin[b] = DependencyOrigin::L1Tex;
        }
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
    fn stamp_clear(&mut self, barrier: Option<DepBarrier>, at: u64, origin: DependencyOrigin) {
        if let Some(barrier) = barrier {
            let index = barrier.index();
            if at >= self.barrier_clear[index] {
                self.barrier_clear[index] = at;
                self.barrier_origin[index] = origin;
            }
        }
    }

    /// Earliest time the control wait mask and any explicit ordinary `DEPBAR`
    /// operand clear; `None` while a required barrier has responses outstanding
    /// (unknown clear time). SB0 asynchronous-copy waits are handled separately.
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
        for b in 0..DEP_BARRIERS {
            if !self.ordinary_dependency_wait_mask.contains_index(b)
                || slot.control.wait_mask.contains_index(b)
            {
                continue;
            }
            if self.barrier_pending[b] > 0 {
                return None;
            }
            t = t.max(self.barrier_clear[b]);
        }
        Some(t)
    }

    fn ready_at(&self) -> Option<u64> {
        if self.done || self.at_block_barrier {
            return None;
        }
        let dependency_ready = self.wait_clear_time()?;
        Some(
            self.eligible
                .max(dependency_ready)
                .max(self.cta_barrier_ready)
                .max(self.barrier_pipeline_ready)
                .max(self.branch_pipeline_ready)
                .max(self.imc_wait_until),
        )
    }
}

fn async_copy_source_count(kind: AccessKind) -> Option<u16> {
    match kind {
        AccessKind::GlobalAsyncCopy {
            len,
            completion_work,
            ..
        } if !matches!(
            decode_async_copy_completion(completion_work),
            AsyncCopyCompletion::Inactive
        ) =>
        {
            // The full SM120 alpha route may retain no sector-arena span for
            // an opaque TMA/LDGSTS record. A nonzero completion token still
            // carries one logical source transaction; keep the source count
            // bounded without claiming a lane/sector cardinality.
            Some(len.max(1))
        }
        AccessKind::GlobalAsyncCopy { .. }
        | AccessKind::NonMemory
        | AccessKind::PredicatedOffMemory
        | AccessKind::Shared { .. }
        | AccessKind::LocalLoad { .. }
        | AccessKind::LocalStore { .. }
        | AccessKind::GlobalEligible { .. }
        | AccessKind::GlobalBypass { .. }
        | AccessKind::GlobalUntagged { .. }
        | AccessKind::GlobalStore { .. }
        | AccessKind::GlobalReduction { .. } => None,
    }
}

fn async_copy_completion_work(kind: AccessKind) -> Option<u64> {
    match kind {
        AccessKind::GlobalAsyncCopy {
            completion_work, ..
        } => match decode_async_copy_completion(completion_work) {
            AsyncCopyCompletion::Inactive => None,
            AsyncCopyCompletion::PayloadWork(work)
            | AsyncCopyCompletion::ExactDestinationWork(work) => Some(u64::from(work)),
            AsyncCopyCompletion::PayloadAndObservedTopology { payload_work, .. } => {
                Some(u64::from(payload_work))
            }
        },
        AccessKind::NonMemory
        | AccessKind::PredicatedOffMemory
        | AccessKind::Shared { .. }
        | AccessKind::LocalLoad { .. }
        | AccessKind::LocalStore { .. }
        | AccessKind::GlobalEligible { .. }
        | AccessKind::GlobalBypass { .. }
        | AccessKind::GlobalUntagged { .. }
        | AccessKind::GlobalStore { .. }
        | AccessKind::GlobalReduction { .. } => None,
    }
}

fn bypass_async_copy_observed_topology(
    kind: AccessKind,
    instruction: usize,
    exact: Option<AsyncCopyCohortView<'_>>,
) -> Option<BypassAsyncCopyInput<'_>> {
    let AccessKind::GlobalAsyncCopy {
        off,
        len,
        completion_work,
        l1_policy: LdgstsL1Policy::BypassL1,
        ..
    } = kind
    else {
        return None;
    };
    match decode_async_copy_completion(completion_work) {
        AsyncCopyCompletion::PayloadAndObservedTopology {
            source_cohorts,
            destination_lines,
            max_bank_wavefronts,
            max_bypass_wavefronts,
            ..
        } => Some(BypassAsyncCopyInput {
            instruction,
            sector_off: exact.map_or(off, |view| view.sector_off),
            sector_len: exact.map_or(len, |view| view.sector_len),
            source_cohorts: u64::from(source_cohorts),
            destination_lines: u64::from(destination_lines),
            max_bank_wavefronts: u64::from(max_bank_wavefronts),
            max_bypass_wavefronts: u64::from(max_bypass_wavefronts),
            cohorts: exact.map(|view| view.cohorts),
            sector_membership_masks: exact.map(|view| view.sector_membership_masks),
        }),
        AsyncCopyCompletion::Inactive
        | AsyncCopyCompletion::PayloadWork(_)
        | AsyncCopyCompletion::ExactDestinationWork(_) => None,
    }
}

fn async_copy_destination_wavefronts(
    kind: AccessKind,
    params: &IssueModelParams,
) -> Result<Option<u64>, IssueError> {
    let AccessKind::GlobalAsyncCopy {
        completion_work,
        l1_policy: LdgstsL1Policy::CacheAtAllLevels,
        ..
    } = kind
    else {
        return Ok(None);
    };
    let AsyncCopyCompletion::ExactDestinationWork(work) =
        decode_async_copy_completion(completion_work)
    else {
        return Ok(None);
    };
    let work_per_wavefront = params
        .ldgsts_cache_all_destination_wavefront_cycles
        .checked_mul(params.ldgsts_delivery_units_per_cycle())
        .ok_or(IssueError::TimeMultiplicationOverflow {
            operation: "LDGSTS destination work per wavefront",
            lhs: params.ldgsts_cache_all_destination_wavefront_cycles,
            rhs: params.ldgsts_delivery_units_per_cycle(),
        })?;
    let work = u64::from(work);
    if work_per_wavefront == 0 || work % work_per_wavefront != 0 {
        return Err(IssueError::InvalidAsyncCopyDestinationWork {
            completion_work: work,
            work_per_wavefront,
        });
    }
    Ok(Some(work / work_per_wavefront))
}

fn validate_async_copy_completion_work(kind: AccessKind, warp: usize) -> Result<(), IssueError> {
    if let AccessKind::GlobalAsyncCopy {
        len,
        completion_work,
        ..
    } = kind
        && len != 0
        && completion_work == 0
    {
        return Err(IssueError::AsyncCopyCompletionWorkMismatch {
            warp,
            sectors: len,
            completion_work: u64::from(completion_work),
        });
    }
    Ok(())
}

/// `LDGDEPBAR` encodes SB0 in the ordinary write-barrier field, but that bit is
/// the async-group protocol itself rather than a generic 25-cycle producer.
fn ordinary_write_barrier(slot: &IssueSlot) -> Option<DepBarrier> {
    if slot.op_class == OpClass::AsyncCopyCommit {
        None
    } else {
        slot.control.write_barrier
    }
}

/// Data-ready time for one issued record at time `t` in the Flat/L1Const
/// modes (fill times known at issue), driving the L1 state/counters when
/// attached. Producer visibility is computed when either an ordinary write
/// barrier or an async-copy group consumes it; cache accesses still run without
/// either observer because they mutate modeled state.
pub(crate) fn data_ready_time(
    kind: AccessKind,
    sectors: &[i64],
    t: u64,
    params: &IssueModelParams,
    l1: Option<&mut L1Cache>,
    write_barrier: Option<DepBarrier>,
    observe_async_copy: bool,
) -> Result<Option<u64>, IssueError> {
    let needs_ready = write_barrier.is_some() || observe_async_copy;
    match kind {
        AccessKind::PredicatedOffMemory => Ok(None),
        AccessKind::NonMemory => {
            if needs_ready {
                params.non_memory_ready(t).map(Some)
            } else {
                Ok(None)
            }
        }
        AccessKind::Shared { wavefronts } => {
            if needs_ready {
                params.shared_ready(t, wavefronts).map(Some)
            } else {
                Ok(None)
            }
        }
        AccessKind::GlobalEligible { off, len, .. } | AccessKind::LocalLoad { off, len } => {
            let origin = match kind {
                AccessKind::GlobalEligible { .. } => MemoryOrigin::Global,
                AccessKind::LocalLoad { .. } => MemoryOrigin::Local,
                _ => unreachable!("match arm admits only hierarchy loads"),
            };
            match l1 {
                Some(l1) => {
                    l1.begin_lookup_batch();
                    let mut ready = needs_ready
                        .then(|| {
                            checked_cycle_add("L1 hit-pipeline floor", t, params.l1_hit_latency)
                        })
                        .transpose()?;
                    for &sector in &sectors[off..off + usize::from(len)] {
                        if let Some(current) = &mut ready {
                            let (_, sector_ready) = l1.access_eligible(origin, sector, t)?;
                            *current = (*current).max(sector_ready);
                        } else {
                            l1.access_eligible_unobserved(origin, sector, t)?;
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
            }
        }
        AccessKind::GlobalBypass { len, token, .. } => match l1 {
            Some(l1) => {
                l1.count_bypass(
                    MemoryOrigin::Global,
                    u64::from(len),
                    token == CacheOpToken::Atom,
                )?;
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
        AccessKind::GlobalAsyncCopy {
            off,
            len,
            l1_policy,
            ..
        } => match (l1, l1_policy) {
            (Some(l1), LdgstsL1Policy::CacheAtAllLevels) => {
                l1.begin_async_copy_lookup_batch();
                let mut ready = needs_ready
                    .then(|| {
                        checked_cycle_add("LDGSTS L1 hit-pipeline floor", t, params.l1_hit_latency)
                    })
                    .transpose()?;
                for &sector in &sectors[off..off + usize::from(len)] {
                    if let Some(current) = &mut ready {
                        let (_, sector_ready) = l1.access_eligible_with_mshr_floor(
                            MemoryOrigin::Global,
                            sector,
                            t,
                            params.ldgsts_l1_mshr_floor,
                        )?;
                        *current = (*current).max(sector_ready);
                    } else {
                        l1.access_eligible_unobserved_with_mshr_floor(
                            MemoryOrigin::Global,
                            sector,
                            t,
                            params.ldgsts_l1_mshr_floor,
                        )?;
                    }
                }
                Ok(ready)
            }
            (Some(l1), LdgstsL1Policy::BypassL1) => {
                l1.count_bypass(MemoryOrigin::Global, u64::from(len), false)?;
                if needs_ready {
                    params.store_visibility_ready(t).map(Some)
                } else {
                    Ok(None)
                }
            }
            (None, _) => {
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
                l1.count_untagged(MemoryOrigin::Global, u64::from(len))?;
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
        AccessKind::GlobalStore { off, len }
        | AccessKind::GlobalReduction { off, len, .. }
        | AccessKind::LocalStore { off, len } => {
            let origin = match kind {
                AccessKind::GlobalStore { .. } | AccessKind::GlobalReduction { .. } => {
                    MemoryOrigin::Global
                }
                AccessKind::LocalStore { .. } => MemoryOrigin::Local,
                _ => unreachable!("match arm admits only hierarchy writes"),
            };
            let reduction = matches!(kind, AccessKind::GlobalReduction { .. });
            match l1 {
                Some(l1) => {
                    if let Some(hits) = projected_l1_write_lookup_hits(kind, sectors) {
                        l1.count_write_lookup_hits(origin, hits)?;
                    }
                    for &sector in &sectors[off..off + usize::from(len)] {
                        if reduction {
                            l1.probe_reduction(origin, sector, t)?;
                        } else {
                            l1.probe_store(origin, sector, t)?;
                        }
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
            }
        }
    }
}

/// Completion identity carried by an eligible L1 access. Ordinary loads only
/// observe a dependence barrier; cacheable asynchronous-copy sources also retain
/// their open group while optionally preserving an ordinary barrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FillTarget {
    Barrier(DepBarrier),
    AsyncCopy {
        group: AsyncCopyGroupId,
        barrier: Option<DepBarrier>,
    },
}

impl FillTarget {
    const fn barrier(self) -> Option<DepBarrier> {
        match self {
            Self::Barrier(barrier) => Some(barrier),
            Self::AsyncCopy { barrier, .. } => barrier,
        }
    }

    const fn async_group(self) -> Option<AsyncCopyGroupId> {
        match self {
            Self::Barrier(_) => None,
            Self::AsyncCopy { group, .. } => Some(group),
        }
    }
}

/// An MSHR-stalled remainder of one instruction's sector list, retried
/// (in FIFO order) whenever a fill completion frees an entry.
struct Retry {
    warp: usize,
    instruction: usize,
    origin: MemoryOrigin,
    /// Whether the stalled logical load uses the SM86 constant-cache return
    /// path. Retries retain the marker so a later fill cannot silently lose
    /// the response delay.
    constant_load: bool,
    /// Whether this constant load uses the measured multi-site operand path.
    /// This remains separate from the broad constant marker above.
    constant_deep: bool,
    /// Completion identity of the stalled instruction. A present ordinary
    /// barrier holds one pending replacement unit until the entry drains;
    /// asynchronous-copy source units were counted at original issue and retain
    /// their group identity across every retry.
    target: Option<FillTarget>,
    /// Access-local finite-admission floor retained from original issue. Only
    /// cacheable LDGSTS retries carry one; ordinary hierarchy traffic uses `None`.
    mshr_floor: Option<NonZeroU32>,
    /// Original logical request boundary for pending-fill lookup attribution.
    lookup_batch: L1LookupBatch,
    /// Unresolved slice `warps[warp].sectors[lo..hi]`.
    lo: usize,
    hi: usize,
}

#[derive(Clone, Copy)]
struct FillWaiter {
    warp: usize,
    target: FillTarget,
    units: u32,
    /// Whether this waiter consumes the measured deep constant-return path.
    /// The marker is per waiter because one folded physical fill can serve
    /// ordinary and constant loads with different return-service latencies.
    constant_deep: bool,
}

/// Completion units cleared when one keyed L1 fill completes. Inline for the
/// dominant single-waiter identity and pre-aggregated by `(warp, target)`, so
/// response preflight only rebuilds counts when distinct targets share a fill.
enum Waiters {
    One(FillWaiter),
    Many(Vec<FillWaiter>),
}

impl Waiters {
    fn add(
        &mut self,
        warp: usize,
        target: FillTarget,
        constant_deep: bool,
    ) -> Result<(), IssueError> {
        let new_waiter = FillWaiter {
            warp,
            target,
            units: 1,
            constant_deep,
        };
        match self {
            Self::One(waiter)
                if waiter.warp == warp
                    && waiter.target == target
                    && waiter.constant_deep == constant_deep =>
            {
                waiter.units = checked_u32_counter_add("fill waiter unit", waiter.units, 1)?;
            }
            Self::One(waiter) => {
                *self = Self::Many(vec![*waiter, new_waiter]);
            }
            Self::Many(waiters) => {
                if let Some(waiter) = waiters.iter_mut().find(|waiter| {
                    waiter.warp == warp
                        && waiter.target == target
                        && waiter.constant_deep == constant_deep
                }) {
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

fn accumulate_waiter_removals(
    pending_removals: &mut HashMap<(usize, DepBarrier), u32>,
    async_removals: &mut HashMap<(usize, AsyncCopyGroupId), u32>,
    waiter: FillWaiter,
) -> Result<(), IssueError> {
    if let Some(barrier) = waiter.target.barrier() {
        let remove = pending_removals.entry((waiter.warp, barrier)).or_default();
        *remove = checked_u32_counter_add("response pending-remove budget", *remove, waiter.units)?;
    }
    if let Some(group) = waiter.target.async_group() {
        let remove = async_removals.entry((waiter.warp, group)).or_default();
        *remove =
            checked_u32_counter_add("response async-copy remove budget", *remove, waiter.units)?;
    }
    Ok(())
}

fn checked_constant_return_time(
    base: u64,
    constant_deep: bool,
    params: &IssueModelParams,
    operation: &'static str,
) -> Result<u64, IssueError> {
    if !constant_deep {
        return Ok(base);
    }
    let extra = params
        .constant_memory_return_service
        .map_or(0, |service| service.additional_cycles);
    base.checked_add(extra).ok_or(IssueError::TimeOverflow {
        operation,
        base,
        delta: extra,
    })
}

fn fill_waiter_ready_time(
    waiter: FillWaiter,
    response_time: u64,
    params: &IssueModelParams,
) -> Result<u64, IssueError> {
    if !waiter.constant_deep || waiter.target.barrier().is_none() {
        return Ok(response_time);
    }
    checked_constant_return_time(
        response_time,
        true,
        params,
        "constant response return latency",
    )
}

fn project_async_copy_removal(
    projected: &mut HashMap<(usize, AsyncCopyGroupId), u32>,
    async_copies: &[AsyncCopySlot],
    warp: usize,
    group: AsyncCopyGroupId,
    remove: u32,
    now: u64,
    params: &IssueModelParams,
) -> Result<(), IssueError> {
    if warp >= async_copies.len() {
        return Err(IssueError::ResponseWarpOutOfRange {
            warp,
            warps: async_copies.len(),
        });
    }
    let current = if let Some(&current) = projected.get(&(warp, group)) {
        current
    } else {
        async_copies[warp]
            .preflight_async_copy_source_completion(warp, group, None, 0, now, params)?;
        async_copies[warp]
            .async_group(group)
            .ok_or(IssueError::AsyncCopyGroupNotFound { warp, group })?
            .pending_sources
    };
    let next = current
        .checked_sub(remove)
        .ok_or(IssueError::AsyncCopyPendingUnderflow {
            warp,
            group,
            current,
            remove,
        })?;
    projected.insert((warp, group), next);
    Ok(())
}

/// State that EXISTS only in the shared-L2 mode — carrying it inside
/// the [`MemModel`] variant makes a mispaired construction
/// unrepresentable instead of an `expect()` deep in the memory step.
struct SharedL2State {
    l1: L1Cache,
    /// Monotonic tie-breaker for the per-SM visibility stream. Memory cycle
    /// timestamps alone cannot order equal-cycle fences and accesses.
    l1_visibility_sequence: u64,
    /// Complete global eligible L1 observations, including resident hits.
    /// The native system currently drains this only through the explicit
    /// opt-in callback API; the legacy L2-only path remains unchanged.
    l1_visibility_observations: Vec<sim_mem::L1AccessObservation>,
    l1_fence_observations: Vec<sim_mem::L1FenceObservation>,
    l1_fill_visibility_observations: Vec<sim_mem::L1FillVisibleObservation>,
    l1_invalidate_all_observations: Vec<sim_mem::L1InvalidateAllObservation>,
    l1_visibility_requester: Option<u8>,
    /// Only payload-event sectors are tracked; default replay allocates no
    /// value-observation map. Every later store to a watched sector replaces
    /// its writer, including stores with no corresponding payload event.
    payload_store_writers: Option<Box<BTreeMap<(i64, u8), Option<PayloadStoreObservation>>>>,
    response_owner: ResponseOwner,
    /// Exact response-bearing request tokens still expected from the system.
    outstanding_replies: HashSet<ReplyTo>,
    /// Response identities indexed separately so admission checks do not scan
    /// every outstanding token for each sector in a wide instruction.
    outstanding_response_ids: HashMap<ResponseId, ReplyTo>,
    /// Exact sector mask carried by each outstanding response-bearing request.
    /// This stays separate from `fill_masks`, which is the physical L2
    /// exactly-once traffic mask and may include a later child fill.
    response_masks: HashMap<ResponseId, u8>,
    /// Barrier units cleared when the keyed L1 fill completes.
    fill_waiters: HashMap<FillKey, Waiters>,
    /// Per in-flight LINE fill: the sector bits already sent to the L2
    /// (the fill request's mask plus any follow-up fetches). A merge
    /// onto a sector NOT yet in the mask sends a follow-up sector
    /// fetch with its own L1 sector response; a re-touch of a sent
    /// sector sends nothing, like the hardware MSHR. Entries die
    /// with the parent line response. Exactly-once is PER FILL, not
    /// per-address-epoch: if a resident line with an in-flight SECTOR
    /// fill is evicted and re-allocated, the new fill may re-fetch a
    /// sector whose zombie fetch is still outstanding (rare
    /// eviction-window corner, documented at the 2026-07-07 gate).
    fill_masks: HashMap<i64, u8>,
    /// Line fills emitted during the CURRENT timestamped emission batch:
    /// `(line, issue_time) → first index in out`. A merge onto one of these
    /// folds its sector into the fill request's mask. Requests for the same
    /// line at a later local cycle receive an independent sector token; a
    /// whole `advance` call is not a same-cycle packet boundary.
    /// Cleared at every batch entry; indices are only valid because
    /// each batch starts with an empty `out`.
    line_folds: HashMap<(i64, u64), usize>,
    retry: VecDeque<Retry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PayloadStoreSite {
    pub launch_index: i64,
    pub warp_slot: i32,
    pub instruction_index: usize,
}

#[derive(Debug, Clone, Copy)]
struct PayloadStoreObservation {
    site: PayloadStoreSite,
    accepted_at: u64,
    resident_generation: Option<u32>,
    retained_dirty: bool,
    attached: bool,
}

impl SharedL2State {
    fn next_l1_visibility_sequence(&mut self) -> Result<u64, IssueError> {
        let sequence = self.l1_visibility_sequence;
        self.l1_visibility_sequence =
            sequence.checked_add(1).ok_or(IssueError::CounterOverflow {
                counter: "L1 visibility sequence",
            })?;
        Ok(sequence)
    }
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

    fn shared_l1_mut(&mut self) -> Option<&mut SharedL2State> {
        match self {
            MemModel::L1Shared(state) => Some(state),
            _ => None,
        }
    }
}

/// Extract the constant-bank line keys used by an instruction's SASS text.
/// Immediate constants are operands of the form `c[bank][offset]` or
/// `cx[bank][offset]`; indexed `LDC`/`LDC.*` operations use a different cache
/// and are intentionally excluded here, while uniform `ULDC`/`ULDC.*`
/// operands remain on the IMC path as observed by NCU.  The parser is deliberately narrow:
/// malformed/non-numeric references are ignored rather than guessed, while
/// the trusted static sidecar remains the source of the instruction itself.
fn immediate_constant_lines(opcode_text: &str, line_bytes: u32) -> Box<[u64]> {
    // `opcode_head` strips width/cache modifiers (for example `ULDC.64`)
    // using the same grammar as the rest of the SASS classifier.  On SM86,
    // the remodeled scheduler probes CBANK operands on both ordinary ALU
    // instructions and uniform `ULDC` forms; NCU's IMC request counter also
    // includes those uniform references.  Only the true indexed `LDC` load
    // uses the separate constant-memory path, so exclude that head while
    // retaining ULDC/ULDC.64 CBANK operands.
    if opcode_head(opcode_text) == "LDC" || line_bytes == 0 {
        return Box::new([]);
    }
    let bytes = opcode_text.as_bytes();
    let mut lines = Vec::new();
    let mut cursor = 0usize;
    while cursor + 3 < bytes.len() {
        let is_c = bytes[cursor] == b'c'
            && (bytes[cursor + 1] == b'['
                || (bytes[cursor + 1] == b'x'
                    && cursor + 2 < bytes.len()
                    && bytes[cursor + 2] == b'['));
        if !is_c {
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
        let parse_literal = |slice: &[u8]| -> Option<u64> {
            let text = std::str::from_utf8(slice).ok()?.trim();
            if text.is_empty() {
                return None;
            }
            if let Some(hex) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
                u64::from_str_radix(hex, 16).ok()
            } else {
                text.parse().ok()
            }
        };
        let bank = parse_literal(&bytes[first_open + 1..first_close]);
        let offset = parse_literal(&bytes[second_open + 1..second_close]);
        if let (Some(bank), Some(offset)) = (bank, offset) {
            let address = bank
                .checked_mul(65_536)
                .and_then(|base| base.checked_add(offset));
            if let Some(address) = address {
                let line = address / u64::from(line_bytes);
                if !lines.contains(&line) {
                    lines.push(line);
                }
            }
        }
        cursor = second_close + 1;
    }
    lines.sort_unstable();
    lines.into_boxed_slice()
}

#[derive(Debug, Clone, Copy)]
struct ImmediateConstantPending {
    line: u64,
    ready_at: u64,
}

#[derive(Debug, Clone)]
struct ImmediateConstantCacheState {
    config: ImmediateConstantCache,
    /// Per-set LRU tags, oldest at the front.  A line key is used as its tag
    /// identity after the set index has been derived; this is sufficient for
    /// the closed constant-address domain and avoids address aliases.
    sets: Vec<VecDeque<u64>>,
    pending: Vec<ImmediateConstantPending>,
    /// The pipelined SM-shared L1C access path admits lines at its measured
    /// initiation interval while the MSHR count bounds lines outstanding.
    next_service: u64,
    diagnostics: ImmediateConstantCacheDiagnostics,
}

impl ImmediateConstantCacheState {
    fn new(config: ImmediateConstantCache) -> Result<Self, IssueError> {
        if config.sets == 0
            || config.line_bytes == 0
            || config.ways == 0
            || config.mshr_entries == 0
            || config.access_queue_entries == 0
            || config.miss_initiation_interval_cycles == 0
            || config.sm_structure_latency_cycles == 0
            || config.l0_to_l1_cycles == 0
            || config.switch_after_miss_cycles == 0
        {
            return Err(IssueError::ImmediateConstantCacheConfigurationInvalid);
        }
        Ok(Self {
            config,
            sets: (0..config.sets).map(|_| VecDeque::new()).collect(),
            pending: Vec::new(),
            next_service: 0,
            diagnostics: ImmediateConstantCacheDiagnostics::default(),
        })
    }

    fn set_index(&self, line: u64) -> usize {
        (line % u64::from(self.config.sets)) as usize
    }

    fn contains(&self, line: u64) -> bool {
        self.sets[self.set_index(line)].contains(&line)
    }

    fn pending_ready(&self, line: u64) -> Option<u64> {
        self.pending
            .iter()
            .find(|entry| entry.line == line)
            .map(|entry| entry.ready_at)
    }

    fn touch(&mut self, line: u64) {
        let set_index = self.set_index(line);
        let set = &mut self.sets[set_index];
        if let Some(position) = set.iter().position(|&tag| tag == line) {
            set.remove(position);
        }
        set.push_back(line);
        while set.len() > self.config.ways as usize {
            set.pop_front();
        }
    }

    fn complete_ready(&mut self, now: u64) {
        let mut ready = Vec::new();
        self.pending.retain(|entry| {
            if entry.ready_at <= now {
                ready.push(entry.line);
                false
            } else {
                true
            }
        });
        // `pending` is issue order. Preserve it when multiple lines have
        // become ready since the last scheduler step: sorting by line address
        // would change same-timestamp fill order and therefore LRU victims.
        // Pending-line identity is unique by `reserve`, so no deduplication is
        // needed here.
        // Diagnostic counters are observational; the execution state itself
        // remains checked. Saturation keeps a pathological trace from turning
        // reporting into a scheduler failure.
        self.diagnostics.lines_filled = self
            .diagnostics
            .lines_filled
            .saturating_add(u64::try_from(ready.len()).unwrap_or(u64::MAX));
        for line in ready {
            self.touch(line);
        }
    }

    /// Reserve every currently missing line referenced by one instruction.
    /// Returns the earliest time at which the instruction may be re-issued;
    /// `None` means all references are already resident.  New misses use one
    /// pipelined L1C service lane; an existing pending line is shared by all
    /// waiting warps on that SMSP.
    fn reserve(
        &mut self,
        lines: &[u64],
        now: u64,
        upper_level_latency: u64,
    ) -> Result<Option<u64>, IssueError> {
        let mut wait_until: Option<u64> = None;
        // `upper_level_latency` is the carded L1/L2 response frontier used by
        // the shared event system.  The 11-stage local structure and its
        // 39-cycle transport are already represented in the NCU-calibrated
        // response frontier for this replay; adding them here would charge
        // the same path twice.  The local fields remain part of the capability
        // identity and diagnostics, while this actor waits on the single
        // externally observable response timestamp.
        let miss_latency = upper_level_latency;
        for &line in lines {
            self.diagnostics.references = self.diagnostics.references.saturating_add(1);
            if self.contains(line) {
                self.diagnostics.lookup_hits = self.diagnostics.lookup_hits.saturating_add(1);
                continue;
            }
            self.diagnostics.lookup_misses = self.diagnostics.lookup_misses.saturating_add(1);
            if let Some(ready) = self.pending_ready(line) {
                self.diagnostics.pending_tag_hits =
                    self.diagnostics.pending_tag_hits.saturating_add(1);
                self.diagnostics.pending_merges = self.diagnostics.pending_merges.saturating_add(1);
                wait_until = Some(wait_until.map_or(ready, |current| current.max(ready)));
                continue;
            }
            if self.pending.len() >= self.config.mshr_entries as usize {
                if let Some(ready) = self.pending.iter().map(|entry| entry.ready_at).min() {
                    wait_until = Some(wait_until.map_or(ready, |current| current.max(ready)));
                }
                continue;
            }
            let start = now.max(self.next_service);
            let ready = start
                .checked_add(miss_latency)
                .ok_or(IssueError::TimeOverflow {
                    operation: "immediate constant-cache line response",
                    base: start,
                    delta: miss_latency,
                })?;
            // The 11-stage L1C structure is a pipelined queue: its one-entry
            // access port accepts a new line each cycle while earlier misses
            // continue through the latency stages.  Serializing the port by
            // the full response latency would turn independent CBANK misses
            // into a bandwidth bottleneck that the released SM86 model does
            // not have.  Keep the response timestamp serialized only by the
            // measured initiation interval (one cycle for the characterized
            // queue), while `ready_at` still carries the full miss latency.
            self.next_service = start
                .checked_add(self.config.miss_initiation_interval_cycles)
                .ok_or(IssueError::TimeOverflow {
                    operation: "immediate constant-cache service initiation",
                    base: start,
                    delta: self.config.miss_initiation_interval_cycles,
                })?;
            self.pending.push(ImmediateConstantPending {
                line,
                ready_at: ready,
            });
            self.diagnostics.tag_misses = self.diagnostics.tag_misses.saturating_add(1);
            self.diagnostics.maximum_pending_lines = self
                .diagnostics
                .maximum_pending_lines
                .max(u64::try_from(self.pending.len()).unwrap_or(u64::MAX));
            wait_until = Some(wait_until.map_or(ready, |current| current.max(ready)));
        }
        Ok(wait_until)
    }

    fn diagnostics(&self) -> ImmediateConstantCacheDiagnostics {
        self.diagnostics
    }
}

#[derive(Debug)]
struct ImmediateConstantPlan {
    picks: [Option<usize>; SUB_CORES],
    cache: Option<Box<[ImmediateConstantCacheState; SUB_CORES]>>,
    waits: [Option<(usize, u64)>; SUB_CORES],
    miss_observed: bool,
}

/// Resumable single-SM replay actor. See the module docs for the mode
/// semantics and the run-ahead (horizon) contract.
#[derive(Debug)]
struct SharedWriteLookupRuntime {
    observer: Option<Rc<RefCell<L1WriteLookupObserver>>>,
    /// Per-SMSP source-hit return-port frontiers for cacheable LDGSTS.
    async_l1_hit_service_ready: [u64; SUB_CORES],
}

#[cfg(feature = "reference")]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct AdvanceDiagnostics {
    pub calls: u64,
    pub local_cycle_steps: u64,
    pub idle_jumps: u64,
    pub skipped_cycles: u64,
}

#[cfg(feature = "reference")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReferenceScalarMacroExecution {
    pub(crate) warp_slot: i32,
    pub(crate) first_issue: u64,
    pub(crate) last_issue: u64,
    pub(crate) end_cycle: u64,
    pub(crate) instruction_count: u64,
}

pub(crate) struct SmCore<'a> {
    schedule: &'a FunctionSchedule,
    /// Static schedule values in PC order. Contiguous binary search avoids a
    /// pointer-chasing B-tree lookup for every dynamic issue while preserving
    /// the schedule map as the authoritative join/diagnostic surface.
    schedule_slots: Vec<(u64, &'a IssueSlot)>,
    /// Static PC-indexed memory timing table. It is sorted once at admission so
    /// the hot issue loop uses a branch-predictable binary search instead of a
    /// randomized hash lookup for every resident warp.
    memory_pipeline_schedule: Box<[(u64, MemoryPipelineSlot)]>,
    /// Static PCs that enter the measured multi-site constant operand path.
    /// Sorted static PC set; issue-cycle membership is binary-searched to
    /// avoid randomized hashing on every resident-warp readiness check.
    constant_path_pcs: Box<[u64]>,
    /// Pre-parsed immediate-constant line references keyed by static PC.
    /// Sorted static PC -> immediate-constant line table.
    immediate_constant_schedule: Box<[(u32, Box<[u64]>)]>,
    /// Private IMC/L0C state for each SMSP; absent for synthetic fixtures.
    immediate_constant_cache: Option<Box<[ImmediateConstantCacheState; SUB_CORES]>>,
    params: IssueModelParams,
    shared_memory_residency: SharedMemoryResidency,
    physical_warp_slots: u32,
    instruction_front_end: Option<InstructionFrontEndState>,
    memory_issue_pipeline: Option<MemoryIssuePipelineState>,
    mem: MemModel,
    blocks: &'a [BlockReplay],
    block_live_warps: Vec<u32>,
    instruction_limit: u64,
    warps: Vec<WarpState<'a>>,
    /// Cold protocol storage parallel to `warps`; entries allocate only after
    /// an effective asynchronous-copy source is issued.
    async_copies: Vec<AsyncCopySlot>,
    /// Construction-time specialization flag. Kernels without async-copy
    /// sources or controls bypass the protocol path once per scheduler batch,
    /// not once per issued instruction or resident-warp scan.
    has_async_copy_protocol: bool,
    block_states: Vec<BlockState>,
    subcore_warps: [Vec<usize>; SUB_CORES],
    hold: [Option<usize>; SUB_CORES],
    /// Architecture-local CTA-barrier issue resource present when the joined
    /// function contains a characterized barrier instruction.
    barrier_issue: Option<CtaBarrierIssue>,
    /// Earliest cycle at which the one SM-shared barrier server can accept
    /// another warp arrival.
    barrier_pipe_ready: u64,
    /// Earliest issue time of each SMSP's local Tensor Core. This timeline is
    /// kernel-local but survives CTA retirement/admission waves on the SM.
    tensor_pipe_ready: [u64; SUB_CORES],
    /// Independent per-SMSP uniform/SFU initiation timeline.
    sfu_pipe_ready: [u64; SUB_CORES],
    /// Finite scalar INT/FP execution windows, one per SMSP.
    scalar_execution_pipeline: Option<Box<ScalarExecutionPipelineState>>,
    /// Cold shared observer/async-return state is boxed to keep the actor's
    /// compact hot structure within the established 4 KiB bound.
    shared_write_lookup_runtime: Box<SharedWriteLookupRuntime>,
    /// Earliest free cycle of the cache-all LDGSTS destination service shared
    /// by every resident warp on this SM.
    ldgsts_destination_pipe_ready: u64,
    /// Pending barrier-observed L1-hit result writebacks. Allocated only when
    /// one joined function contains both the admitted HADD2 producer and the
    /// admitted sm_86 STRONG.SM scalar load form.
    result_writebacks: Option<ResultWritebackState>,
    unmodeled_tails: TailFlags,
    /// Box the cold diagnostic aggregate so adding a new observation surface
    /// cannot inflate the scheduler's hot actor beyond its compact-size
    /// invariant.
    readiness: Box<ReadinessDiagnostics>,
    memory_service_candidates: MemoryServiceCandidateState,
    emit_admission_witnesses: bool,
    memory_admission_feedback: Option<Box<MemoryAdmissionFeedbackState>>,
    /// Effective picks from the previous transaction in the current logical
    /// cycle. Distinguishes dispatch progress from a parked IMC miss.
    last_issue_picks: [Option<usize>; SUB_CORES],
    instructions_per_smsp: [u64; SUB_CORES],
    issue_active_cycles_per_smsp: [u64; SUB_CORES],
    last_observed_issue: [Option<u64>; SUB_CORES],
    warp_lifetimes: Vec<WarpLifetime>,
    issued: u64,
    t: u64,
    /// Latest modeled SM-local completion frontier that can occur after the
    /// last issue (for example, a terminal CTA-barrier release).
    local_completion_time: Option<u64>,
    /// P15 logical route-service availability.  Full replay only; fast
    /// diagnostics never construct or consume this state.
    sm120_flat_service_enabled: bool,
    sm120_flat_global_service_ready: u64,
    sm120_flat_shared_service_ready: u64,
    sm120_flat_multiwarp_service_enabled: bool,
    sm120_flat_multiwarp_global_service_ready: u64,
    sm120_flat_multiwarp_dense: bool,
    sm120_flat_multiwarp_medium: bool,
    sm120_flat_shared_medium: bool,
    sm120_flat_async_dense: bool,
    // Logical recent-source window, not a physical cache geometry.
    sm120_async_sources: VecDeque<(i64, u64)>,
    sm120_flat_async_service_enabled: bool,
    sm120_flat_async_service_ready: u64,
    sm120_flat_async_depth_service_per_group: u64,
    sm120_flat_async_depth_service_ready: u64,
    sm120_flat_store_retire_enabled: bool,
    sm120_flat_store_retire_ready: u64,
    arrival_stamp: u64,
    resident_warps: u32,
    resident_shared_memory_bytes: u64,
    resident_warp_slots: Box<HashSet<i32>>,
    unfinished_warps: usize,
    next_block: usize,
    /// Test/reference-only switch. The production default retains next-event
    /// advancement; the reference lane walks locally idle cycles one at a
    /// time so differential runs can separate clock jumps from state changes.
    dense_reference: bool,
    #[cfg(feature = "reference")]
    advance_diagnostics: AdvanceDiagnostics,
    #[cfg(feature = "reference")]
    reference_issue_effects: Vec<(u64, u64, u32)>,
}

#[cfg(feature = "reference")]
fn reference_mode_from_env() -> Result<bool, IssueError> {
    match std::env::var_os("SIM_SM_REFERENCE_MODE") {
        None => Ok(false),
        Some(mode) if mode == "event" => Ok(false),
        Some(mode) if mode == "dense" => Ok(true),
        Some(mode) => Err(IssueError::ReferenceModeInvalid {
            mode: mode.to_string_lossy().into_owned(),
        }),
    }
}

impl<'a> SmCore<'a> {
    #[inline]
    fn memory_pipeline_slot(&self, pc: u64) -> Option<MemoryPipelineSlot> {
        self.memory_pipeline_schedule
            .binary_search_by_key(&pc, |(key, _)| *key)
            .ok()
            .map(|index| self.memory_pipeline_schedule[index].1)
    }

    #[inline]
    fn is_constant_path_pc(&self, pc: u64) -> bool {
        self.constant_path_pcs.binary_search(&pc).is_ok()
    }

    #[inline]
    fn immediate_constant_lines(&self, pc: u32) -> Option<&[u64]> {
        self.immediate_constant_schedule
            .binary_search_by_key(&pc, |(key, _)| *key)
            .ok()
            .map(|index| self.immediate_constant_schedule[index].1.as_ref())
    }

    /// Build the actor and admit the initial block wave. `blocks` must be
    /// this SM's blocks in launch order; every instruction's pc must
    /// resolve in `schedule` (single-function launches — the replay
    /// driver enforces that before calling). Cache modes attach a FRESH
    /// per-SM L1 (cold cache per kernel, NCU `--cache-control all`
    /// parity).
    pub(crate) fn new_with_residency(
        blocks: &'a [BlockReplay],
        schedule: &'a FunctionSchedule,
        params: &IssueModelParams,
        shared_memory_residency: SharedMemoryResidency,
        mode: SmMemMode,
    ) -> Result<Self, IssueError> {
        Self::new_with_residency_and_observer(
            blocks,
            schedule,
            params,
            shared_memory_residency,
            mode,
            None,
        )
    }

    pub(crate) fn new_with_residency_and_observer(
        blocks: &'a [BlockReplay],
        schedule: &'a FunctionSchedule,
        params: &IssueModelParams,
        shared_memory_residency: SharedMemoryResidency,
        mode: SmMemMode,
        write_lookup_observer: Option<Rc<RefCell<L1WriteLookupObserver>>>,
    ) -> Result<Self, IssueError> {
        params.validate_resource_source()?;
        if shared_memory_residency.max_bytes_per_sm() == 0 {
            return Err(IssueError::SharedMemoryCapacityInvalid {
                arch: params.arch,
                max_bytes_per_sm: shared_memory_residency.max_bytes_per_sm(),
            });
        }
        if params.arch != ArchId::Sm86
            && !matches!(params.resource_source, IssueResourceSource::Sm120Alpha)
        {
            return Err(IssueError::ArchitectureUnsupported { arch: params.arch });
        }
        if params.arch != schedule.arch {
            return Err(IssueError::ArchitectureMismatch {
                component: "issue parameters",
                expected: schedule.arch,
                got: params.arch,
            });
        }
        let physical_warp_slots = if matches!(params.resource_source, IssueResourceSource::Sm120Alpha)
        {
            params.max_resident_warps_per_sm
        } else {
            params
                .arch
                .issue_hypotheses()
                .require("physical_warp_slots")?
                .max_resident_warps_per_sm
        };
        let schedule_slots = schedule
            .by_pc
            .iter()
            .map(|(&pc, slot)| (pc, slot))
            .collect::<Vec<_>>();
        let constant_streams = constant_stream_count(schedule);
        let constant_pressure_threshold = params
            .memory_issue_pipeline
            .map_or(u32::MAX, |config| config.queue_entries_with_dispatch);
        let mut constant_path_pcs = schedule
            .by_pc
            .iter()
            .filter_map(|(&pc, slot)| {
                deep_constant_load(slot, constant_streams, constant_pressure_threshold)
                    .then_some(pc)
            })
            .collect::<Vec<_>>();
        constant_path_pcs.sort_unstable();
        let immediate_constant_config = matches!(mode, SmMemMode::L1Shared(_))
            .then_some(params.immediate_constant_cache)
            .flatten();
        let immediate_constant_schedule =
            immediate_constant_config.map_or_else(Vec::new, |config| {
                schedule
                    .by_pc
                    .iter()
                    .filter_map(|(&pc, slot)| {
                        let lines = immediate_constant_lines(slot.opcode_text(), config.line_bytes);
                        (!lines.is_empty()).then_some((pc as u32, lines))
                    })
                    .collect::<Vec<_>>()
            });
        let immediate_constant_cache: Option<Box<[ImmediateConstantCacheState; SUB_CORES]>> =
            immediate_constant_config
                .map(|config| {
                    Ok::<Box<[ImmediateConstantCacheState; SUB_CORES]>, IssueError>(Box::new([
                        ImmediateConstantCacheState::new(config)?,
                        ImmediateConstantCacheState::new(config)?,
                        ImmediateConstantCacheState::new(config)?,
                        ImmediateConstantCacheState::new(config)?,
                    ]))
                })
                .transpose()?;
        let mut memory_pipeline_schedule = Vec::new();
        if let Some(config) = params.memory_issue_pipeline {
            for (&pc, slot) in &schedule.by_pc {
                if let Some(memory_slot) =
                    memory_pipeline_slot(slot, config, constant_path_pcs.binary_search(&pc).is_ok())
                {
                    memory_pipeline_schedule.push((pc, memory_slot));
                }
            }
        }
        // Validate the complete dynamic/static join and every actor-cardinality
        // conversion before constructing mutable replay state. A later block
        // admission must not discover an unknown PC or overflow after earlier
        // blocks have already committed instructions.
        let mut block_live_warps = Vec::with_capacity(blocks.len());
        let mut instruction_limit = 0u64;
        let mut arrival_limit = 0u64;
        let mut sm120_flat_global_sectors = 0u64;
        let mut sm120_flat_shared_wavefronts = 0u64;
        let mut sm120_flat_live_warps = 0u32;
        let mut sm120_flat_async_group_count = 0u64;
        let mut sm120_flat_atomic_present = false;
        let has_async_copy_protocol = schedule.by_pc.values().any(|slot| {
            matches!(
                slot.op_class,
                OpClass::AsyncCopyCommit | OpClass::AsyncCopyWait
            ) || slot.mem_role == sim_mem::MemRole::BypassLoad(CacheOpToken::Ldgsts)
        });
        let has_nonzero_async_wait = schedule.by_pc.values().any(|slot| {
            slot.op_class == OpClass::AsyncCopyWait && slot.async_copy_wait_max_pending_groups != 0
        });
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
            sm120_flat_live_warps = sm120_flat_live_warps.saturating_add(live_warps);
            for warp in &block.warps {
                for inst in &warp.insts {
                    if schedule
                        .issue_slot(u64::from(inst.pc))
                        .is_some_and(|slot| opcode_head(slot.opcode_text()) == "LDGDEPBAR")
                    {
                        sm120_flat_async_group_count = sm120_flat_async_group_count.saturating_add(1);
                    }
                    match inst.kind {
                        AccessKind::GlobalBypass { token: CacheOpToken::Atom, len, .. } => {
                            sm120_flat_atomic_present = true;
                            sm120_flat_global_sectors = sm120_flat_global_sectors.saturating_add(u64::from(len));
                        }
                        AccessKind::GlobalReduction { len, .. } => {
                            sm120_flat_global_sectors = sm120_flat_global_sectors.saturating_add(u64::from(len));
                        }
                        AccessKind::GlobalEligible { len, .. }
                        | AccessKind::GlobalBypass { len, .. }
                        | AccessKind::GlobalUntagged { len, .. }
                        | AccessKind::GlobalStore { len, .. } => {
                            sm120_flat_global_sectors = sm120_flat_global_sectors.saturating_add(u64::from(len));
                        }
                        AccessKind::Shared { wavefronts } => {
                            sm120_flat_shared_wavefronts = sm120_flat_shared_wavefronts.saturating_add(wavefronts);
                        }
                        AccessKind::NonMemory
                        | AccessKind::PredicatedOffMemory
                        | AccessKind::LocalLoad { .. }
                        | AccessKind::LocalStore { .. }
                        | AccessKind::GlobalAsyncCopy { .. } => {}
                    }
                }
            }
            if live_warps > params.max_resident_warps_per_sm {
                return Err(IssueError::BlockResidencyLimitExceeded {
                    launch_index: block.launch_index,
                    live_warps,
                    arch: params.arch,
                    max_resident_warps: params.max_resident_warps_per_sm,
                });
            }
            if live_warps != 0 && params.max_resident_blocks_per_sm == 0 {
                return Err(IssueError::BlockCountResidencyLimitExceeded {
                    launch_index: block.launch_index,
                    arch: params.arch,
                    max_resident_blocks: params.max_resident_blocks_per_sm,
                });
            }
            if live_warps != 0
                && shared_memory_residency.bytes_per_cta()
                    > shared_memory_residency.max_bytes_per_sm()
            {
                return Err(IssueError::BlockSharedMemoryLimitExceeded {
                    launch_index: block.launch_index,
                    bytes_per_cta: shared_memory_residency.bytes_per_cta(),
                    arch: params.arch,
                    max_bytes_per_sm: shared_memory_residency.max_bytes_per_sm(),
                });
            }
            let mut block_slots = HashSet::with_capacity(block.warps.len());
            for warp in &block.warps {
                let slot =
                    u32::try_from(warp.warp_slot).map_err(|_| IssueError::WarpSlotOutOfRange {
                        launch_index: block.launch_index,
                        warp_slot: warp.warp_slot,
                        arch: params.arch,
                        physical_warp_slots,
                    })?;
                if slot >= physical_warp_slots {
                    return Err(IssueError::WarpSlotOutOfRange {
                        launch_index: block.launch_index,
                        warp_slot: warp.warp_slot,
                        arch: params.arch,
                        physical_warp_slots,
                    });
                }
                if !block_slots.insert(warp.warp_slot) {
                    return Err(IssueError::DuplicateWarpSlot {
                        launch_index: block.launch_index,
                        warp_slot: warp.warp_slot,
                    });
                }
            }
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
                    let slot = lookup_dense(
                        &schedule_slots,
                        schedule,
                        inst.pc,
                        block.launch_index,
                        warp.warp_slot,
                    )?;
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
        let unmodeled_tails = blocks
            .iter()
            .flat_map(|block| &block.warps)
            .flat_map(|warp| &warp.insts)
            .fold(TailFlags::default(), |flags, inst| {
                flags.union(TailFlags::for_access(inst.kind))
            });
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
                l1_visibility_sequence: 0,
                l1_visibility_observations: Vec::new(),
                l1_fence_observations: Vec::new(),
                l1_fill_visibility_observations: Vec::new(),
                l1_invalidate_all_observations: Vec::new(),
                l1_visibility_requester: None,
                payload_store_writers: None,
                response_owner: allocate_response_owner()?,
                outstanding_replies: HashSet::new(),
                outstanding_response_ids: HashMap::new(),
                response_masks: HashMap::new(),
                fill_waiters: HashMap::new(),
                fill_masks: HashMap::new(),
                line_folds: HashMap::new(),
                retry: VecDeque::new(),
            }),
        };
        let disable_reference_front_end = {
            #[cfg(feature = "reference")]
            {
                std::env::var_os("SIM_SM_REFERENCE_DISABLE_FRONTEND").is_some()
            }
            #[cfg(not(feature = "reference"))]
            {
                false
            }
        };
        let instruction_front_end = if disable_reference_front_end {
            None
        } else {
            params
                .instruction_front_end
                .map(|config| InstructionFrontEndState::new(config, params.l1_miss_latency))
                .transpose()?
        };
        let memory_issue_pipeline = params
            .memory_issue_pipeline
            .map(MemoryIssuePipelineState::new)
            .transpose()?;
        let scalar_execution_pipeline = params
            .scalar_execution_pipeline
            .map(ScalarExecutionPipelineState::new)
            .transpose()?
            .map(Box::new);
        let stateful_write_lookup_observer = stateful_write_lookup_observer_enabled(schedule);
        let write_lookup_observer = if stateful_write_lookup_observer {
            match write_lookup_observer {
                Some(observer) => Some(observer),
                None => Some(Rc::new(RefCell::new(
                    L1WriteLookupObserver::for_arch(params.arch).map_err(IssueError::from)?,
                ))),
            }
        } else {
            None
        };
        let mut barrier_issue = None;
        for slot in schedule.by_pc.values() {
            let PipelineClass::Barrier {
                warp_initiation_interval_cycles,
                generation_rearm_cycles,
            } = slot.pipeline_class
            else {
                continue;
            };
            let declared = CtaBarrierIssue {
                warp_initiation_interval_cycles,
                generation_rearm_cycles,
            };
            if let Some(expected) = barrier_issue
                && expected != declared
            {
                return Err(IssueError::BarrierPipelineConfigurationMismatch {
                    expected,
                    got: declared,
                });
            }
            barrier_issue = Some(declared);
        }
        let has_fixed_writeback = schedule.by_pc.values().any(|slot| {
            matches!(
                result_writeback_kind(slot, params.arch),
                Some(ResultWritebackKind::FixedHadd2 { .. })
            )
        });
        let has_variable_writeback = schedule.by_pc.values().any(|slot| {
            matches!(
                result_writeback_kind(slot, params.arch),
                Some(ResultWritebackKind::L1TexStrongSmU32 { .. })
            )
        });
        let result_writebacks = if has_fixed_writeback && has_variable_writeback {
            Some(ResultWritebackState {
                timing: params
                    .arch
                    .fixed_l1tex_writeback_priority()
                    .require("fixed_l1tex_writeback_priority")?,
                pending: Vec::new(),
            })
        } else {
            None
        };
        let readiness = ReadinessDiagnostics {
            async_wait_prefix: has_nonzero_async_wait
                .then(|| Box::new(AsyncWaitPrefixDiagnostics::default())),
            ..ReadinessDiagnostics::default()
        };
        #[cfg(feature = "reference")]
        let dense_reference = reference_mode_from_env()?;
        #[cfg(not(feature = "reference"))]
        let dense_reference = false;
        let mut core = Self {
            schedule,
            schedule_slots,
            memory_pipeline_schedule: memory_pipeline_schedule.into_boxed_slice(),
            constant_path_pcs: constant_path_pcs.into_boxed_slice(),
            immediate_constant_schedule: immediate_constant_schedule.into_boxed_slice(),
            immediate_constant_cache,
            params: *params,
            shared_memory_residency,
            physical_warp_slots,
            instruction_front_end,
            memory_issue_pipeline,
            mem,
            blocks,
            block_live_warps,
            instruction_limit,
            warps: Vec::new(),
            async_copies: Vec::new(),
            has_async_copy_protocol,
            block_states: Vec::new(),
            subcore_warps: Default::default(),
            hold: [None; SUB_CORES],
            barrier_issue,
            barrier_pipe_ready: 0,
            tensor_pipe_ready: [0; SUB_CORES],
            sfu_pipe_ready: [0; SUB_CORES],
            scalar_execution_pipeline,
            shared_write_lookup_runtime: Box::new(SharedWriteLookupRuntime {
                observer: write_lookup_observer,
                async_l1_hit_service_ready: [0; SUB_CORES],
            }),
            ldgsts_destination_pipe_ready: 0,
            result_writebacks,
            unmodeled_tails,
            readiness: Box::new(readiness),
            memory_service_candidates: MemoryServiceCandidateState::default(),
            emit_admission_witnesses: false,
            memory_admission_feedback: None,
            last_issue_picks: [None; SUB_CORES],
            instructions_per_smsp: [0; SUB_CORES],
            issue_active_cycles_per_smsp: [0; SUB_CORES],
            last_observed_issue: [None; SUB_CORES],
            warp_lifetimes: Vec::new(),
            issued: 0,
            // Keep the scheduler/event clock anchored at zero.  The
            // architecture-scoped SM-active observation boundary is an
            // explicit output projection (see `sm_active_observation_frontier`)
            // rather than hidden in every readiness timestamp and cumulative
            // diagnostic.  This preserves the causal lower-bound semantics of
            // the issue model and prevents a fixed observation interval from
            // being charged once per resident SM or per event.
            t: 0,
            local_completion_time: None,
            sm120_flat_service_enabled: params.resource_source() == IssueResourceSource::Sm120Alpha
                && matches!(mode, SmMemMode::Flat)
                && sm120_flat_live_warps == 1
                && (sm120_flat_global_sectors >= SM120_FLAT_SERVICE_SECTOR_THRESHOLD
                    || sm120_flat_shared_wavefronts >= SM120_FLAT_SERVICE_SECTOR_THRESHOLD),
            sm120_flat_global_service_ready: 0,
            sm120_flat_shared_service_ready: 0,
            sm120_flat_multiwarp_service_enabled: params.resource_source() == IssueResourceSource::Sm120Alpha
                && matches!(mode, SmMemMode::Flat)
                && sm120_flat_live_warps > 1
                && sm120_flat_global_sectors >= SM120_FLAT_SERVICE_SECTOR_THRESHOLD,
            sm120_flat_multiwarp_global_service_ready: 0,
            sm120_flat_multiwarp_dense: params.resource_source() == IssueResourceSource::Sm120Alpha
                && matches!(mode, SmMemMode::Flat)
                && sm120_flat_live_warps > 1
                && sm120_flat_global_sectors >= 8 * SM120_FLAT_SERVICE_SECTOR_THRESHOLD,
            sm120_flat_multiwarp_medium: params.resource_source() == IssueResourceSource::Sm120Alpha
                && matches!(mode, SmMemMode::Flat)
                && sm120_flat_live_warps > 1
                && sm120_flat_global_sectors >= SM120_FLAT_SERVICE_SECTOR_THRESHOLD,
            sm120_flat_shared_medium: params.resource_source() == IssueResourceSource::Sm120Alpha
                && matches!(mode, SmMemMode::Flat)
                && sm120_flat_live_warps == 1
                && sm120_flat_shared_wavefronts >= 128,
            sm120_async_sources: VecDeque::new(),
            sm120_flat_async_dense: params.resource_source() == IssueResourceSource::Sm120Alpha
                && matches!(mode, SmMemMode::Flat)
                && sm120_flat_async_group_count > 8,
            sm120_flat_async_service_enabled: params.resource_source() == IssueResourceSource::Sm120Alpha
                && matches!(mode, SmMemMode::Flat)
                && sm120_flat_async_group_count <= 8
                && schedule.by_pc.values().any(|slot| opcode_head(slot.opcode_text()) == "LDGDEPBAR"),
            sm120_flat_async_service_ready: 0,
            // Fresh q7 group-depth probes are a logical overlap surface:
            // depth 1 needs no extra arm, depth 2 needs ~44 cycles/group,
            // and depth 4 needs ~13 cycles/group. This is explicitly shape
            // conditioned and never presented as a native queue width.
            sm120_flat_async_depth_service_per_group: if sm120_flat_async_group_count <= 64 {
                0
            } else if sm120_flat_async_group_count <= 128 {
                44
            } else if sm120_flat_async_group_count > 128 {
                13
            } else {
                0
            },
            sm120_flat_async_depth_service_ready: 0,
            sm120_flat_store_retire_enabled: params.resource_source() == IssueResourceSource::Sm120Alpha
                && matches!(mode, SmMemMode::Flat)
                && !sm120_flat_atomic_present
                && sm120_flat_live_warps == 1
                && (128..SM120_FLAT_SERVICE_SECTOR_THRESHOLD).contains(&sm120_flat_global_sectors),
            sm120_flat_store_retire_ready: 0,
            arrival_stamp: 0,
            resident_warps: 0,
            resident_shared_memory_bytes: 0,
            resident_warp_slots: Box::new(HashSet::new()),
            unfinished_warps: 0,
            next_block: 0,
            dense_reference,
            #[cfg(feature = "reference")]
            advance_diagnostics: AdvanceDiagnostics::default(),
            #[cfg(feature = "reference")]
            reference_issue_effects: Vec::new(),
        };
        core.admit_blocks(core.t)?;
        Ok(core)
    }

    pub(crate) fn install_memory_service_candidates(
        &mut self,
        candidates: &[MemoryServiceCandidate],
    ) -> Result<(), IssueError> {
        self.memory_service_candidates = MemoryServiceCandidateState::checked(candidates)?;
        Ok(())
    }

    #[cfg(feature = "reference")]
    pub(crate) fn advance_diagnostics(&self) -> AdvanceDiagnostics {
        self.advance_diagnostics
    }

    /// Install one closed scalar macro into the native model actor. This
    /// reference-only seam is intentionally narrow: no front-end, memory,
    /// pipeline, barrier, or completion state may cross the boundary. The
    /// typed boundary executor independently validates the corresponding
    /// contract; this method only installs the already-derived state delta and
    /// then leaves `advance` to execute the continuation.
    #[cfg(feature = "reference")]
    pub(crate) fn execute_reference_scalar_macro(
        &mut self,
        warp_slot: i32,
        instruction_count: u64,
        initiation_interval_cycles: u64,
        terminal_block_barrier: bool,
        detailed_effects: &[TypedEventTransition],
        terminal_front_end_exit: Option<&InstructionFrontEndBoundaryState>,
    ) -> Result<ReferenceScalarMacroExecution, IssueError> {
        if instruction_count == 0 || initiation_interval_cycles == 0 {
            return Err(IssueError::ReferenceMacroUnsupported {
                reason: "nonzero_count_and_initiation_interval_required",
            });
        }
        let memory_pipeline_open = self.memory_issue_pipeline.as_ref().is_some_and(|pipeline| {
            pipeline
                .queue_release_times
                .iter()
                .any(|queue| !queue.is_empty())
                || pipeline
                    .local_completion_ready
                    .iter()
                    .any(|&ready| ready > self.t)
                || pipeline.shared_accept_ready > self.t
        });
        let scalar_pipeline_open =
            self.scalar_execution_pipeline
                .as_ref()
                .is_some_and(|pipeline| {
                    pipeline.release_times.iter().any(|queue| !queue.is_empty())
                        || pipeline.next_service.iter().any(|&ready| ready > self.t)
                });
        let constant_cache_open = self
            .immediate_constant_cache
            .as_ref()
            .is_some_and(|caches| {
                caches.iter().any(|cache| {
                    !cache.pending.is_empty()
                        || cache.next_service > self.t
                        || cache.sets.iter().any(|set| !set.is_empty())
                })
            });
        if memory_pipeline_open
            || scalar_pipeline_open
            || constant_cache_open
            || !self.unmodeled_tails.is_empty()
            || self.outstanding_responses() != 0
            || self.block_states.iter().any(|block| block.waiting != 0)
            || self.warps.iter().any(|warp| {
                warp.at_block_barrier
                    || warp.barrier_pending.iter().any(|&pending| pending != 0)
                    || warp.next_slot.is_none() && !warp.done
            })
        {
            return Err(IssueError::ReferenceMacroUnsupported {
                reason: "future_relevant_native_state_is_not_closed",
            });
        }
        let warp_index = self
            .warps
            .iter()
            .position(|warp| warp.warp_slot == warp_slot)
            .ok_or(IssueError::ReferenceMacroUnsupported {
                reason: "warp_slot_not_resident",
            })?;
        let (next, inst_len, launch_index) = {
            let warp = &self.warps[warp_index];
            (warp.next, warp.insts.len(), warp.launch_index)
        };
        let end_index = next
            .checked_add(usize::try_from(instruction_count).map_err(|_| {
                IssueError::ReferenceMacroUnsupported {
                    reason: "instruction_count_out_of_range",
                }
            })?)
            .ok_or(IssueError::ReferenceMacroUnsupported {
                reason: "instruction_count_overflow",
            })?;
        if end_index > inst_len {
            return Err(IssueError::ReferenceMacroUnsupported {
                reason: "macro_exceeds_warp_stream",
            });
        }
        let instruction_count_usize = end_index - next;
        // A comparator-selected production witness carries one detailed issue
        // effect per collapsed instruction.  The model-local helper tests
        // intentionally omit effects and use the scalar recurrence, so keep
        // that compatibility path; whenever effects are supplied, however,
        // reject a truncated or non-monotone span before any actor mutation.
        if terminal_block_barrier && detailed_effects.len() != instruction_count_usize
            || (!detailed_effects.is_empty()
                && (detailed_effects.len() != instruction_count_usize
                    || detailed_effects
                        .windows(2)
                        .any(|window| window[1].event.time < window[0].event.time)))
        {
            return Err(IssueError::ReferenceMacroUnsupported {
                reason: "detailed_effect_span_not_closed",
            });
        }
        if self.instruction_front_end.is_some() && !terminal_block_barrier {
            return Err(IssueError::ReferenceMacroUnsupported {
                reason: "front_end_transition_not_implemented",
            });
        }
        let front_end_transition = if terminal_block_barrier && self.instruction_front_end.is_some()
        {
            let boundary =
                terminal_front_end_exit.ok_or(IssueError::ReferenceMacroUnsupported {
                    reason: "terminal_front_end_exit_missing",
                })?;
            Some(InstructionFrontEndState::from_typed_boundary(boundary)?)
        } else {
            None
        };
        for (offset, instruction) in (next..end_index).enumerate() {
            let slot = lookup_dense(
                &self.schedule_slots,
                self.schedule,
                self.warps[warp_index].insts[instruction].pc,
                launch_index,
                warp_slot,
            )?;
            let terminal_barrier_slot =
                terminal_block_barrier && offset + 2 == instruction_count_usize;
            let terminal_exit_slot =
                terminal_block_barrier && offset + 1 == instruction_count_usize;
            let supported_issue_slot = terminal_barrier_slot
                || terminal_exit_slot
                || matches!(slot.pipeline_class, PipelineClass::ScalarAlu { .. });
            let terminal_sequence_slot = terminal_block_barrier
                && matches!(slot.pipeline_class, PipelineClass::DispatchOnly);
            if !supported_issue_slot && !terminal_sequence_slot
                || !matches!(slot.mem_role, MemRole::None)
                || (!terminal_barrier_slot
                    && !terminal_exit_slot
                    && slot.op_class != OpClass::Other)
                || (terminal_barrier_slot && slot.op_class != OpClass::BlockBarrier)
                || (terminal_exit_slot && !slot.opcode_text().starts_with("EXIT"))
            {
                return Err(IssueError::ReferenceMacroUnsupported {
                    reason: "scalar_macro_descriptor_not_closed",
                });
            }
        }
        // Resolve the detailed continuation slot before mutating the actor.
        // A malformed schedule must take exact fallback without leaving a
        // partially advanced warp behind.
        let next_slot = if end_index < inst_len {
            Some(lookup_dense(
                &self.schedule_slots,
                self.schedule,
                self.warps[warp_index].insts[end_index].pc,
                launch_index,
                warp_slot,
            )?)
        } else {
            None
        };
        let subcore = self.warps[warp_index]
            .warp_slot
            .rem_euclid(SUB_CORES as i32) as usize;
        let block = self.warps[warp_index].block;
        let first_issue = detailed_effects
            .first()
            .map(|effect| effect.event.time)
            .unwrap_or_else(|| self.t.max(self.warps[warp_index].eligible));
        let last_issue = detailed_effects
            .last()
            .map(|effect| effect.event.time)
            .or_else(|| {
                instruction_count
                    .checked_sub(1)
                    .and_then(|count| count.checked_mul(initiation_interval_cycles))
                    .and_then(|offset| first_issue.checked_add(offset))
            })
            .ok_or(IssueError::ReferenceMacroUnsupported {
                reason: "macro_timing_overflow",
            })?;
        let end_cycle = last_issue
            .checked_add(1)
            .ok_or(IssueError::ReferenceMacroUnsupported {
                reason: "macro_end_overflow",
            })?;
        self.warps[warp_index].next = end_index;
        self.warps[warp_index].eligible = if terminal_block_barrier {
            last_issue
        } else {
            end_cycle
        };
        self.warps[warp_index].current_since = end_cycle;
        self.warps[warp_index].next_slot = next_slot;
        if let Some(boundary) = terminal_front_end_exit {
            let _ = boundary;
            self.warps[warp_index].front_end_fetch_cursor = end_index;
            self.warps[warp_index].instruction_buffer.clear();
        }
        self.issued =
            self.issued
                .checked_add(instruction_count)
                .ok_or(IssueError::CounterOverflow {
                    counter: "reference macro issued",
                })?;
        self.instructions_per_smsp[subcore] = self.instructions_per_smsp[subcore]
            .checked_add(instruction_count)
            .ok_or(IssueError::CounterOverflow {
                counter: "reference macro SMSP instructions",
            })?;
        self.issue_active_cycles_per_smsp[subcore] = self.issue_active_cycles_per_smsp[subcore]
            .checked_add(instruction_count)
            .ok_or(IssueError::CounterOverflow {
                counter: "reference macro SMSP active cycles",
            })?;
        self.hold[subcore] = Some(warp_index);
        if let Some(front_end) = front_end_transition {
            self.instruction_front_end = Some(front_end);
        }
        if terminal_block_barrier {
            let barrier_issue = detailed_effects
                .get(detailed_effects.len().saturating_sub(2))
                .map(|effect| effect.event.time)
                .ok_or(IssueError::ReferenceMacroUnsupported {
                    reason: "terminal_barrier_effect_missing",
                })?;
            let barrier_arrival =
                barrier_issue
                    .checked_add(2)
                    .ok_or(IssueError::ReferenceMacroUnsupported {
                        reason: "terminal_barrier_arrival_overflow",
                    })?;
            let barrier_release =
                barrier_arrival
                    .checked_add(1)
                    .ok_or(IssueError::ReferenceMacroUnsupported {
                        reason: "terminal_barrier_release_overflow",
                    })?;
            self.block_states[block].waiting = 0;
            self.block_states[block].barrier_generation = self.block_states[block]
                .barrier_generation
                .checked_add(1)
                .ok_or(IssueError::CounterOverflow {
                    counter: "reference macro barrier generation",
                })?;
            self.block_states[block].barrier_rearm_ready = barrier_arrival;
            self.block_states[block].barrier_first_dispatch = None;
            self.block_states[block].barrier_first_arrival = None;
            self.barrier_pipe_ready = barrier_arrival;
            self.local_completion_time = Some(barrier_release);
            self.warps[warp_index].barrier_clear[0] = detailed_effects
                .get(detailed_effects.len().saturating_sub(4))
                .map(|effect| effect.event.time)
                .ok_or(IssueError::ReferenceMacroUnsupported {
                    reason: "terminal_barrier_clear_effect_missing",
                })?;
            self.warps[warp_index].barrier_pipeline_ready = barrier_release;
            self.warps[warp_index].at_block_barrier = true;
            self.warps[warp_index].next_slot = None;
            // Retire after the terminal EXIT issue.  `retire_warp` records
            // activity through `t + 1`; using the barrier release here would
            // shorten the warp lifetime relative to detailed replay even
            // though the barrier's local completion frontier is earlier.
            self.retire_warp(subcore, warp_index, last_issue)?;
            self.t = end_cycle;
        }
        self.t = end_cycle;
        Ok(ReferenceScalarMacroExecution {
            warp_slot,
            first_issue,
            last_issue,
            end_cycle,
            instruction_count,
        })
    }

    /// Consume the native comparator's selected execution path. Exact replay
    /// returns no macro transition; the typed-macro branch performs the native
    /// state mutation and returns its continuation frontier.
    #[cfg(feature = "reference")]
    pub(crate) fn execute_reference_macro_path(
        &mut self,
        path: &NativeBoundaryExecutionPath,
        warp_slot: i32,
        instruction_count: u64,
        initiation_interval_cycles: u64,
        terminal_block_barrier: bool,
        detailed_effects: &[TypedEventTransition],
        terminal_front_end_exit: Option<&InstructionFrontEndBoundaryState>,
    ) -> Result<Option<ReferenceScalarMacroExecution>, IssueError> {
        match path {
            NativeBoundaryExecutionPath::TypedMacro => self
                .execute_reference_scalar_macro(
                    warp_slot,
                    instruction_count,
                    initiation_interval_cycles,
                    terminal_block_barrier,
                    detailed_effects,
                    terminal_front_end_exit,
                )
                .map(Some),
            NativeBoundaryExecutionPath::ExactReplay { .. } => Ok(None),
        }
    }

    /// Consume a comparator-selected typed contract. This remains an opt-in
    /// reference seam; callers must validate the contract and continuation
    /// before invoking it, and exact replay is represented by `Ok(None)`.
    #[cfg(feature = "reference")]
    pub(crate) fn execute_reference_boundary_result(
        &mut self,
        result: &NativeBoundaryExecutionResult,
    ) -> Result<Option<ReferenceScalarMacroExecution>, IssueError> {
        let NativeBoundaryExecutionResult::TypedMacro { contract } = result else {
            return Ok(None);
        };
        let core = contract
            .entry
            .cores
            .first()
            .ok_or(IssueError::ReferenceMacroUnsupported {
                reason: "typed_macro_core_missing",
            })?;
        let warp = core
            .warp_states
            .first()
            .ok_or(IssueError::ReferenceMacroUnsupported {
                reason: "typed_macro_warp_missing",
            })?;
        let count = u64::try_from(contract.internal_effects.len()).map_err(|_| {
            IssueError::ReferenceMacroUnsupported {
                reason: "typed_macro_effect_count_out_of_range",
            }
        })?;
        let interval = contract
            .internal_effects
            .windows(2)
            .next()
            .and_then(|effects| effects[1].event.time.checked_sub(effects[0].event.time))
            .unwrap_or(1)
            .max(1);
        let terminal_block_barrier = contract
            .observable_labels
            .iter()
            .any(|label| label == "terminal_scalar_barrier_v1");
        let terminal_front_end_exit = contract
            .exit
            .cores
            .first()
            .and_then(|core| core.front_end.as_ref());
        self.execute_reference_macro_path(
            &NativeBoundaryExecutionPath::TypedMacro,
            warp.warp_slot,
            count,
            interval,
            terminal_block_barrier,
            &contract.internal_effects,
            terminal_front_end_exit,
        )
    }

    #[cfg(feature = "reference")]
    pub(crate) fn reference_issue_effects_since(&self, start: usize) -> &[(u64, u64, u32)] {
        self.reference_issue_effects.get(start..).unwrap_or(&[])
    }

    #[cfg(feature = "reference")]
    pub(crate) fn typed_l2_request_key(&self, sm_id: u32, reply: ReplyTo) -> Option<L2RequestKey> {
        let response_id = reply.response_id()?;
        let response_kind = match reply {
            ReplyTo::L1Fill {
                fill: FillKey::Line { .. },
                ..
            } => TypedResponseKind::L1Fill,
            ReplyTo::L1Fill {
                fill: FillKey::Sector { .. },
                ..
            } => TypedResponseKind::L1SectorFill,
            ReplyTo::Warp { .. } => TypedResponseKind::Warp,
            ReplyTo::AsyncCopy { .. } => TypedResponseKind::AsyncCopy,
            ReplyTo::None => return None,
        };
        let warp_slot = self.warps.get(response_id.warp)?.warp_slot;
        let (response_warp, instruction, sector_index) = response_id.boundary_identity();
        Some(L2RequestKey {
            sm_id,
            response_warp,
            warp_slot,
            instruction,
            sector_index,
            response_kind,
        })
    }

    #[cfg(feature = "reference")]
    pub(crate) fn typed_l2_request_owners(
        &self,
        sm_id: u32,
        response_due: &[(u64, u64, u64, u64, Option<u64>)],
    ) -> Vec<L2RequestOwner> {
        let MemModel::L1Shared(state) = &self.mem else {
            return Vec::new();
        };
        let mut owners = state
            .outstanding_response_ids
            .iter()
            .map(|(response_id, reply)| {
                let direct_identity = self
                    .warps
                    .get(response_id.warp)
                    .and_then(|warp| warp.sectors.get(response_id.sector_index))
                    .map(|sector_id| {
                        (
                            CanonicalLineKey::VirtualAbsolute {
                                line_id: sector_id.div_euclid(4),
                            },
                            1_u8 << sector_id.rem_euclid(4) as u8,
                        )
                    });
                let (line, sector_mask, response_kind) = match reply {
                    ReplyTo::L1Fill { fill, .. } => match fill {
                        FillKey::Line { line, .. } => (
                            CanonicalLineKey::VirtualAbsolute { line_id: *line },
                            state
                                .response_masks
                                .get(response_id)
                                .copied()
                                .or_else(|| state.fill_masks.get(line).copied())
                                .unwrap_or(0),
                            TypedResponseKind::L1Fill,
                        ),
                        FillKey::Sector { line, sector, .. } => (
                            CanonicalLineKey::VirtualAbsolute { line_id: *line },
                            state
                                .response_masks
                                .get(response_id)
                                .copied()
                                .unwrap_or(1_u8 << *sector),
                            TypedResponseKind::L1SectorFill,
                        ),
                    },
                    ReplyTo::Warp { .. } => {
                        let (line, sector_mask) = direct_identity
                            .unwrap_or((CanonicalLineKey::Unknown { line_id: None }, 0));
                        (line, sector_mask, TypedResponseKind::Warp)
                    }
                    ReplyTo::AsyncCopy { .. } => {
                        let (line, sector_mask) = direct_identity
                            .unwrap_or((CanonicalLineKey::Unknown { line_id: None }, 0));
                        (line, sector_mask, TypedResponseKind::AsyncCopy)
                    }
                    ReplyTo::None => (
                        CanonicalLineKey::Unknown { line_id: None },
                        0,
                        TypedResponseKind::None,
                    ),
                };
                let warp_slot = self
                    .warps
                    .get(response_id.warp)
                    .map(|warp| warp.warp_slot)
                    .unwrap_or(-1);
                let (warp, instruction, sector_index) = response_id.boundary_identity();
                let due = response_due.iter().find(
                    |(candidate_warp, candidate_instruction, candidate_sector, _, _)| {
                        (*candidate_warp, *candidate_instruction, *candidate_sector)
                            == (warp, instruction, sector_index)
                    },
                );
                L2RequestOwner {
                    request: L2RequestKey {
                        sm_id,
                        response_warp: warp,
                        warp_slot,
                        instruction,
                        sector_index,
                        response_kind,
                    },
                    line,
                    sector_mask,
                    allocation_generation: due.and_then(|(_, _, _, _, generation)| *generation),
                    state: L2RequestState::Outstanding,
                    due: due.map(|(_, _, _, due, _)| *due),
                }
            })
            .collect::<Vec<_>>();
        owners.sort_by(|left, right| {
            (
                left.request.sm_id,
                left.request.response_warp,
                left.request.warp_slot,
                left.request.instruction,
                left.request.sector_index,
                &left.request.response_kind,
            )
                .cmp(&(
                    right.request.sm_id,
                    right.request.response_warp,
                    right.request.warp_slot,
                    right.request.instruction,
                    right.request.sector_index,
                    &right.request.response_kind,
                ))
        });
        owners
    }

    #[cfg(feature = "reference")]
    pub(crate) fn typed_l1_state(&self) -> Option<L1BoundaryState> {
        self.mem.l1().map(L1Cache::typed_boundary_state)
    }

    /// Enable complete eligible-global L1 access observations for an explicit
    /// native visibility experiment. The requester is the independently
    /// retained physical `sm_id`, not an actor-vector index.
    pub(crate) fn enable_l1_visibility_observations(
        &mut self,
        requester: u8,
    ) -> Result<(), IssueError> {
        if requester >= 128 {
            return Err(IssueError::CounterOverflow {
                counter: "L1 visibility requester domain",
            });
        }
        let MemModel::L1Shared(st) = &mut self.mem else {
            return Err(IssueError::ResponseModeMismatch {
                mode: "L1 visibility requires shared-L2",
            });
        };
        st.l1_visibility_requester = Some(requester);
        Ok(())
    }

    pub(crate) fn enable_l1_resident_writeback(&mut self) -> Result<(), IssueError> {
        let Some(state) = self.mem.shared_l1_mut() else {
            return Err(IssueError::ResponseModeMismatch {
                mode: "L1 resident writeback requires shared-L2",
            });
        };
        state
            .l1
            .set_store_policy(L1StorePolicy::ResidentHitWriteBack);
        Ok(())
    }

    pub(crate) fn l1_dirty_residency(&self) -> sim_mem::L1DirtyResidency {
        self.mem
            .l1()
            .map_or_else(sim_mem::L1DirtyResidency::default, L1Cache::dirty_residency)
    }

    pub(crate) fn preview_l1_dirty_flush(&self) -> Result<sim_mem::L1DirtyFlushPlan, IssueError> {
        let MemModel::L1Shared(state) = &self.mem else {
            return Err(IssueError::ResponseModeMismatch {
                mode: "L1 dirty flush requires shared-L2",
            });
        };
        Ok(state.l1.preview_dirty_flush())
    }

    pub(crate) fn payload_observation_enabled(&self) -> bool {
        matches!(&self.mem, MemModel::L1Shared(state) if state.payload_store_writers.is_some())
    }

    pub(crate) fn preview_l1_dirty_payloads(
        &self,
    ) -> Result<Vec<sim_mem::SectorPayloadV1>, IssueError> {
        let MemModel::L1Shared(state) = &self.mem else {
            return Err(IssueError::ResponseModeMismatch {
                mode: "L1 payload preview requires shared-L2",
            });
        };
        state.l1.preview_dirty_payloads().map_err(IssueError::from)
    }

    pub(crate) fn commit_l1_dirty_flush(
        &mut self,
        plan: sim_mem::L1DirtyFlushPlan,
    ) -> Result<(), IssueError> {
        let Some(state) = self.mem.shared_l1_mut() else {
            return Err(IssueError::ResponseModeMismatch {
                mode: "L1 dirty flush requires shared-L2",
            });
        };
        state.l1.commit_dirty_flush(plan).map_err(IssueError::from)
    }

    pub(crate) fn preview_l1_dirty_payload_admission(
        &self,
        payloads: &[sim_mem::SectorPayloadV1],
        witnesses: &[sim_mem::L1PayloadResidencyWitness],
    ) -> Result<sim_mem::L1DirtyPayloadAdmissionPlan, IssueError> {
        let MemModel::L1Shared(state) = &self.mem else {
            return Err(IssueError::ResponseModeMismatch {
                mode: "L1 payload admission requires shared-L2",
            });
        };
        state
            .l1
            .preview_dirty_payload_admission_with_witness(payloads, witnesses)
            .map_err(IssueError::from)
    }

    pub(crate) fn commit_l1_dirty_payload_admission(
        &mut self,
        plan: sim_mem::L1DirtyPayloadAdmissionPlan,
    ) -> Result<(), IssueError> {
        let MemModel::L1Shared(state) = &mut self.mem else {
            return Err(IssueError::ResponseModeMismatch {
                mode: "L1 payload admission requires shared-L2",
            });
        };
        state
            .l1
            .commit_dirty_payload_admission(plan)
            .map_err(IssueError::from)
    }

    /// Apply one event-time, witness-aware dirty-payload admission as a
    /// single native-facing operation. The underlying preview/commit pair is
    /// still exposed internally so a future event producer can retain the
    /// plan across an explicit scheduling boundary; this convenience path is
    /// atomic with respect to the current SM actor.
    #[allow(dead_code)] // no producer currently supplies this event to replay
    pub(crate) fn admit_l1_dirty_payloads_with_witness(
        &mut self,
        payloads: &[sim_mem::SectorPayloadV1],
        witnesses: &[sim_mem::L1PayloadResidencyWitness],
    ) -> Result<(), IssueError> {
        let plan = self.preview_l1_dirty_payload_admission(payloads, witnesses)?;
        self.commit_l1_dirty_payload_admission(plan)
    }

    pub(crate) fn watch_l1_payload_sectors(&mut self, payloads: &[sim_mem::SectorPayloadV1]) {
        if let MemModel::L1Shared(st) = &mut self.mem {
            let writers = st
                .payload_store_writers
                .get_or_insert_with(Default::default);
            for payload in payloads {
                writers
                    .entry((payload.line_id(), payload.sector()))
                    .or_insert(None);
            }
        }
    }

    /// The public replay path binds the site from the retained trace, never
    /// from a producer's guessed issue cycle or reused physical warp slot.
    pub(crate) fn admit_issued_l1_store_payloads(
        &mut self,
        site: PayloadStoreSite,
        at: u64,
        payloads: &[sim_mem::SectorPayloadV1],
        witnesses: &[sim_mem::L1PayloadResidencyWitness],
    ) -> Result<(), IssueError> {
        let MemModel::L1Shared(st) = &mut self.mem else {
            return Err(IssueError::ResponseModeMismatch {
                mode: "store payload admission requires shared-L2",
            });
        };
        for payload in payloads {
            let line = payload.line_id();
            let sector = payload.sector();
            let refuse = |reason| IssueError::PayloadStoreBinding {
                line,
                sector,
                reason,
            };
            let writer = st
                .payload_store_writers
                .as_ref()
                .and_then(|writers| writers.get(&(line, sector)))
                .and_then(Option::as_ref)
                .ok_or_else(|| refuse("referenced store has not issued"))?;
            if writer.site != site {
                return Err(refuse("current writer is a different dynamic store"));
            }
            if writer.accepted_at > at {
                return Err(refuse("payload event precedes modeled store acceptance"));
            }
            if !writer.retained_dirty {
                return Err(refuse("store did not retain this sector dirty in L1"));
            }
            if writer.attached {
                return Err(refuse(
                    "payload already attached to this dynamic store sector",
                ));
            }
            if st
                .l1
                .resident_payload_state(line)
                .map(|(generation, _)| generation)
                != writer.resident_generation
            {
                return Err(refuse("resident allocation changed after the store"));
            }
        }
        st.l1.attach_observed_store_payloads(payloads, witnesses)?;
        for payload in payloads {
            if let Some(Some(writer)) = st
                .payload_store_writers
                .as_mut()
                .and_then(|writers| writers.get_mut(&(payload.line_id(), payload.sector())))
            {
                writer.attached = true;
            }
        }
        Ok(())
    }

    pub(crate) fn take_l1_visibility_events(&mut self) -> Vec<sim_mem::L1VisibilityEvent> {
        let MemModel::L1Shared(st) = &mut self.mem else {
            return Vec::new();
        };
        let mut events = st
            .l1_visibility_observations
            .drain(..)
            .map(sim_mem::L1VisibilityEvent::Access)
            .chain(
                st.l1_fence_observations
                    .drain(..)
                    .map(sim_mem::L1VisibilityEvent::Fence),
            )
            .chain(
                st.l1_fill_visibility_observations
                    .drain(..)
                    .map(sim_mem::L1VisibilityEvent::FillVisible),
            )
            .chain(
                st.l1_invalidate_all_observations
                    .drain(..)
                    .map(sim_mem::L1VisibilityEvent::InvalidateAll),
            )
            .collect::<Vec<_>>();
        events.sort_unstable_by_key(|event| match event {
            sim_mem::L1VisibilityEvent::Access(observation) => observation.sequence,
            sim_mem::L1VisibilityEvent::Fence(observation) => observation.sequence,
            sim_mem::L1VisibilityEvent::FillVisible(observation) => observation.sequence,
            sim_mem::L1VisibilityEvent::InvalidateAll(observation) => observation.sequence,
        });
        events
    }

    /// Count visibility observations that have been produced by the SM but
    /// not yet transferred to the shared event queue.  The system layer drains
    /// this vector at every actor boundary; `advance` uses the count to stop
    /// immediately after an issue cycle that created a visibility event,
    /// preventing strict L1 replay from batching later issue cycles ahead of
    /// its own same-cycle coherence observation.
    pub(crate) fn pending_l1_visibility_event_count(&self) -> usize {
        let MemModel::L1Shared(st) = &self.mem else {
            return 0;
        };
        st.l1_visibility_observations
            .len()
            .saturating_add(st.l1_fence_observations.len())
            .saturating_add(st.l1_fill_visibility_observations.len())
            .saturating_add(st.l1_invalidate_all_observations.len())
    }

    pub(crate) fn invalidate_l1(
        &mut self,
        line_id: i64,
        sector_mask: u8,
    ) -> Result<(), IssueError> {
        let MemModel::L1Shared(st) = &mut self.mem else {
            return Err(IssueError::ResponseModeMismatch {
                mode: "L1 invalidation requires shared-L2",
            });
        };
        st.l1.invalidate(line_id, sector_mask)?;
        Ok(())
    }

    pub(crate) fn invalidate_l1_all(&mut self) -> Result<u64, IssueError> {
        let MemModel::L1Shared(st) = &mut self.mem else {
            return Err(IssueError::ResponseModeMismatch {
                mode: "L1 invalidation requires shared-L2",
            });
        };
        Ok(st.l1.invalidate_all()?)
    }

    #[cfg(feature = "reference")]
    pub(crate) fn typed_local_completion_time(&self) -> Option<u64> {
        self.local_completion_time
    }

    #[cfg(feature = "reference")]
    pub(crate) fn typed_unprojected_timing_state(&self) -> Vec<String> {
        let mut reasons = Vec::new();
        if self.sfu_pipe_ready.iter().any(|&ready| ready != 0) {
            // This resource survives CTA retirement/admission, but schema 15
            // does not serialize its availability. Never certify an exact
            // continuation using only the independent tensor frontier.
            reasons.push("sfu_pipeline_state".to_owned());
        }
        if !self.unmodeled_tails.is_empty() {
            // TailFlags are capability boundaries for completion effects that
            // are deliberately outside the modeled event stream. Preserve
            // them as a contract-level refusal instead of allowing a typed
            // boundary to imply complete writeback/coherence/RMW behavior.
            reasons.push("unmodeled_completion_tails".to_owned());
        }
        if let MemModel::L1Shared(state) = &self.mem
            && (!state.fill_waiters.is_empty()
                || !state.fill_masks.is_empty()
                || !state.line_folds.is_empty()
                || !state.retry.is_empty())
        {
            // These shared-L2 side maps select which parked sector range,
            // waiter, and per-fill mask transition runs after a response.
            // L1 tags and response owners alone cannot reconstruct that
            // transaction boundary.
            reasons.push("shared_l2_transient_state".to_owned());
        }
        if self
            .typed_l1_state()
            .is_some_and(|l1| !l1.line_fills.is_empty() || !l1.sector_fills.is_empty())
        {
            // The typed response owner carries L2 allocation identity, while
            // the L1 response handler consumes a private FillKey generation.
            // Without the owner-to-L1-fill edge, a live response cannot be
            // resumed exactly even though the L1 fill timing itself is typed.
            reasons.push("l1_fill_response_identity".to_owned());
        }
        if self
            .warps
            .iter()
            .any(|warp| warp.barrier_pending.iter().any(|&pending| pending != 0))
        {
            // Response owners retain requester/fill identity, but the typed
            // owner schema does not yet carry the dependency-barrier index
            // that each response clears. A live counter is therefore not
            // sufficient to resume the next issue transition exactly.
            reasons.push("dependency_barrier_ownership".to_owned());
        }
        let future_constant_path = self.warps.iter().any(|warp| {
            warp.insts.get(warp.next..).is_some_and(|remaining| {
                remaining
                    .iter()
                    .any(|instruction| self.is_constant_path_pc(u64::from(instruction.pc)))
            })
        });
        if future_constant_path
            && self
                .immediate_constant_cache
                .as_ref()
                .is_some_and(|caches| {
                    caches.iter().any(|cache| {
                        cache.sets.iter().any(|set| !set.is_empty())
                            || !cache.pending.is_empty()
                            || cache.next_service != 0
                            || !cache.diagnostics.is_empty()
                    })
                })
        {
            reasons.push("immediate_constant_cache_state".to_owned());
        }
        if self
            .shared_write_lookup_runtime
            .async_l1_hit_service_ready
            .iter()
            .any(|&ready| ready != 0)
        {
            reasons.push("async_l1_hit_service_state".to_owned());
        }
        if self.result_writebacks.is_some() {
            reasons.push("result_writeback_state".to_owned());
        }
        if !self.memory_service_candidates.is_empty() {
            reasons.push("memory_service_candidates".to_owned());
        }
        if self.memory_admission_feedback.is_some() {
            reasons.push("memory_admission_feedback".to_owned());
        }
        reasons.sort_unstable();
        reasons
    }

    #[cfg(feature = "reference")]
    pub(crate) fn typed_async_copy_states(&self) -> Vec<Option<AsyncCopyBoundaryState>> {
        self.async_copies
            .iter()
            .map(|slot| {
                slot.state.as_ref().map(|state| {
                    let project = |group: AsyncCopyGroup| AsyncGroupBoundaryState {
                        sequence: group.id.0,
                        committed_at: group.committed_at,
                        pending_sources: group.pending_sources,
                        source_completion_time: group.source_completion_time,
                        delivery_work: group.delivery_work,
                        destination_wavefronts: group.destination_wavefronts,
                        destination_service_queued: group.destination_service_queued,
                        destination_completion_time: group.destination_completion_time,
                        completion_time: group.completion_time,
                    };
                    let mut bypass_observations = state
                        .bypass_copies
                        .iter()
                        .map(|copy| AsyncCopyBypassObservationState {
                            group: copy.group.0,
                            instruction: u64::try_from(copy.instruction).unwrap_or(u64::MAX),
                            pending_sources: copy.pending_sources,
                            source_completion_time: copy.source_completion_time,
                            delivery_work: copy.delivery_work,
                            source_cohorts: copy.source_cohorts,
                            destination_lines: copy.destination_lines,
                            max_bank_wavefronts: copy.max_bank_wavefronts,
                            max_bypass_wavefronts: copy.max_bypass_wavefronts,
                            sector_off: u64::try_from(copy.sector_off).unwrap_or(u64::MAX),
                            sector_membership_masks: copy.sector_membership_masks.to_vec(),
                            cohorts: copy
                                .cohorts
                                .iter()
                                .map(|cohort| AsyncCopyCohortBoundaryState {
                                    pending_sources: cohort.pending_sources,
                                    first_source_response: cohort.first_source_response,
                                    last_source_response: cohort.last_source_response,
                                    payload_work: cohort.payload_work,
                                    destination_lines: cohort.destination_lines,
                                    bank_wavefronts: cohort.bank_wavefronts,
                                    bypass_wavefronts: cohort.bypass_wavefronts,
                                    topology_penalty: cohort.topology_penalty,
                                })
                                .collect(),
                        })
                        .collect::<Vec<_>>();
                    bypass_observations
                        .sort_by_key(|observation| (observation.group, observation.instruction));
                    // Aggregate-only fallback captures preserve useful
                    // diagnostics but do not retain the cohort/membership
                    // topology required by an exact transition contract.
                    let bypass_observations_complete =
                        bypass_observations.iter().all(|observation| {
                            observation.source_cohorts != 0
                                && !observation.sector_membership_masks.is_empty()
                                && !observation.cohorts.is_empty()
                        });
                    AsyncCopyBoundaryState {
                        open: state.open_group.map(project),
                        committed: state
                            .committed_groups
                            .iter()
                            .copied()
                            .map(project)
                            .collect(),
                        bypass_observation_count: u64::try_from(state.bypass_copies.len())
                            .unwrap_or(u64::MAX),
                        next_sequence: state.next_sequence,
                        bypass_observations_complete,
                        bypass_observations,
                    }
                })
            })
            .collect()
    }

    #[cfg(feature = "reference")]
    pub(crate) fn typed_memory_pipeline_state(&self) -> Option<MemoryPipelineBoundaryState> {
        self.memory_issue_pipeline.as_ref().map(|state| {
            let diagnostics = state.diagnostics;
            MemoryPipelineBoundaryState {
                queue_entries_with_dispatch: state.config.queue_entries_with_dispatch,
                local_initiation_interval_cycles: state.config.local_initiation_interval_cycles,
                shared_accept_interval_cycles: state.config.shared_accept_interval_cycles,
                queue_release_times: state
                    .queue_release_times
                    .iter()
                    .map(|queue| queue.iter().copied().collect())
                    .collect(),
                local_completion_ready: state.local_completion_ready.to_vec(),
                shared_accept_ready: state.shared_accept_ready,
                instructions_admitted: diagnostics.instructions_admitted,
                operand_pipeline_delay_cycles: diagnostics.operand_pipeline_delay_cycles,
                shared_queue_delay_cycles: diagnostics.shared_queue_delay_cycles,
                maximum_operand_pipeline_delay_cycles: diagnostics
                    .maximum_operand_pipeline_delay_cycles,
                maximum_shared_queue_delay_cycles: diagnostics.maximum_shared_queue_delay_cycles,
                maximum_queue_occupancy: diagnostics.maximum_queue_occupancy,
                terminal_shared_accept_cycle: diagnostics.terminal_shared_accept_cycle,
            }
        })
    }

    #[cfg(feature = "reference")]
    pub(crate) fn typed_scalar_pipeline_state(&self) -> Option<ScalarPipelineBoundaryState> {
        self.scalar_execution_pipeline.as_ref().map(|state| {
            let diagnostics = state.diagnostics;
            ScalarPipelineBoundaryState {
                latency_cycles: state.config.latency_cycles,
                initiation_interval_cycles: state.config.initiation_interval_cycles,
                in_flight_capacity: state.config.in_flight_capacity,
                release_times: state
                    .release_times
                    .iter()
                    .map(|queue| queue.iter().copied().collect())
                    .collect(),
                next_service: state.next_service.to_vec(),
                instructions_admitted: diagnostics.instructions_admitted,
                service_busy_cycles: diagnostics.service_busy_cycles,
                queue_delay_cycles: diagnostics.queue_delay_cycles,
                maximum_queue_occupancy: diagnostics.maximum_queue_occupancy,
            }
        })
    }

    #[cfg(feature = "reference")]
    pub(crate) fn typed_front_end_state(&self) -> Option<InstructionFrontEndBoundaryState> {
        let state = self.instruction_front_end.as_ref()?;
        let cache = |cache: &SetAssociativeInstructionCache| InstructionCacheState {
            ways: u64::try_from(cache.ways).unwrap_or(u64::MAX),
            access_sequence: cache.access_sequence,
            sets: cache
                .sets
                .iter()
                .map(|set| {
                    set.iter()
                        .map(|entry| InstructionCacheEntryState {
                            function_unique_id: entry.key.function_unique_id,
                            line: entry.key.line,
                            ready_at: entry.ready_at,
                            last_access: entry.last_access,
                        })
                        .collect()
                })
                .collect(),
        };
        let streams = state
            .streams
            .iter()
            .map(|stream| InstructionStreamState {
                lines: stream
                    .lines
                    .iter()
                    .map(|line| InstructionStreamLineState {
                        function_unique_id: line.key.function_unique_id,
                        line: line.key.line,
                        ready_at: line.ready_at,
                        demanded: line.demanded,
                    })
                    .collect(),
                next_line: stream
                    .next_line
                    .map(|key| (key.function_unique_id, key.line)),
            })
            .collect();
        let diagnostics = state.diagnostics;
        Some(InstructionFrontEndBoundaryState {
            line_bytes: state.config.line_bytes,
            l0_to_l1_cycles: state.config.l0_to_l1_cycles,
            l1_to_l0_cycles: state.config.l1_to_l0_cycles,
            l1_request_ports: state.config.l1_request_ports,
            l1_reply_ports: state.config.l1_reply_ports,
            stream_buffer_lines: state.config.stream_buffer_lines,
            stream_buffers_per_subcore: state.config.stream_buffers_per_subcore,
            prefetches_per_cycle: state.config.prefetches_per_cycle,
            instruction_buffer_entries: state.config.instruction_buffer_entries,
            fetch_decode_width: state.config.fetch_decode_width,
            fetch_to_issue_stages: state.config.fetch_to_issue_stages,
            l0: state.l0.iter().map(cache).collect(),
            l1: cache(&state.l1),
            streams,
            l1_request_ready: state.l1_request_ready,
            l1_reply_ready: state.l1_reply_ready,
            l0_response_ready: state.l0_response_ready.to_vec(),
            prefetch_priority: u64::try_from(state.prefetch_priority).unwrap_or(u64::MAX),
            cold_fill_latency: state.cold_fill_latency,
            instructions_fetched: diagnostics.instructions_fetched,
            demand_l0_hits: diagnostics.demand_l0_hits,
            demand_stream_buffer_hits: diagnostics.demand_stream_buffer_hits,
            demand_l0_misses: diagnostics.demand_l0_misses,
            demand_l1_hits: diagnostics.demand_l1_hits,
            demand_l1_misses: diagnostics.demand_l1_misses,
            prefetch_requests: diagnostics.prefetch_requests,
            prefetch_l1_hits: diagnostics.prefetch_l1_hits,
            prefetch_l1_misses: diagnostics.prefetch_l1_misses,
            stream_restarts: diagnostics.stream_restarts,
            maximum_fetch_latency_cycles: diagnostics.maximum_fetch_latency_cycles,
        })
    }

    #[cfg(feature = "reference")]
    pub(crate) fn boundary_snapshot(&self) -> BoundarySnapshot {
        let warp_states = self
            .warps
            .iter()
            .map(|warp| WarpBoundaryState {
                warp_slot: warp.warp_slot,
                next_instruction: u64::try_from(warp.next).unwrap_or(u64::MAX),
                front_end_fetch_cursor: u64::try_from(warp.front_end_fetch_cursor)
                    .unwrap_or(u64::MAX),
                instruction_buffer: warp
                    .instruction_buffer
                    .iter()
                    .map(|buffered| {
                        (
                            u64::try_from(buffered.instruction).unwrap_or(u64::MAX),
                            buffered.ready_at,
                        )
                    })
                    .collect(),
                eligible: warp.eligible,
                imc_wait_until: warp.imc_wait_until,
                barrier_clear: warp.barrier_clear,
                barrier_pending: warp.barrier_pending,
                block: u64::try_from(warp.block).unwrap_or(u64::MAX),
                arrival: warp.arrival,
                at_block_barrier: warp.at_block_barrier,
                cta_barrier_ready: warp.cta_barrier_ready,
                barrier_pipeline_ready: warp.barrier_pipeline_ready,
                branch_pipeline_ready: warp.branch_pipeline_ready,
                block_barrier_dispatch: warp.block_barrier_dispatch,
                block_barrier_arrival: warp.block_barrier_arrival,
                done: warp.done,
            })
            .collect();
        let block_states = self
            .block_states
            .iter()
            .map(|block| BlockBoundaryState {
                unfinished: u64::try_from(block.unfinished).unwrap_or(u64::MAX),
                waiting: u64::try_from(block.waiting).unwrap_or(u64::MAX),
                warp_count: block.warp_count,
                shared_memory_bytes: block.shared_memory_bytes,
                warp_slots: block.warp_slots.to_vec(),
                first_warp: u64::try_from(block.first_warp).unwrap_or(u64::MAX),
                admitted_warps: u64::try_from(block.admitted_warps).unwrap_or(u64::MAX),
                barrier_generation: block.barrier_generation,
                barrier_first_dispatch: block.barrier_first_dispatch,
                barrier_first_arrival: block.barrier_first_arrival,
                barrier_rearm_ready: block.barrier_rearm_ready,
            })
            .collect();
        let hold = self.hold.map(|entry| {
            entry.and_then(|warp| {
                self.warps
                    .get(warp)
                    .map(|state| u64::try_from(state.warp_slot).unwrap_or(0))
            })
        });
        let subcore_warps = self
            .subcore_warps
            .iter()
            .map(|warps| {
                warps
                    .iter()
                    .filter_map(|&warp| self.warps.get(warp).map(|state| state.warp_slot))
                    .collect()
            })
            .collect();
        BoundarySnapshot {
            schema_version: sim_semantics::BOUNDARY_SCHEMA_VERSION,
            cycle: self.t,
            issued: self.issued,
            resident_warps: self.resident_warps,
            resident_shared_memory_bytes: self.resident_shared_memory_bytes,
            unfinished_warps: u64::try_from(self.unfinished_warps).unwrap_or(u64::MAX),
            next_block: u64::try_from(self.next_block).unwrap_or(u64::MAX),
            hold,
            subcore_warps,
            outstanding_responses: u64::try_from(self.outstanding_responses()).unwrap_or(u64::MAX),
            l1_state: match &self.mem {
                MemModel::Flat => None,
                MemModel::L1Const(cache) => Some(cache.boundary_state()),
                MemModel::L1Shared(state) => Some(state.l1.boundary_state()),
            },
            instruction_front_end_state: self
                .instruction_front_end
                .as_ref()
                .map(|state| format!("{state:?}")),
            memory_issue_pipeline_state: self
                .memory_issue_pipeline
                .as_ref()
                .map(|state| format!("{state:?}")),
            scalar_execution_pipeline_state: self
                .scalar_execution_pipeline
                .as_ref()
                .map(|state| format!("{state:?}")),
            async_copy_state: self
                .async_copies
                .iter()
                .map(|slot| slot.state.as_deref().map(|state| format!("{state:?}")))
                .collect(),
            barrier_pipe_ready: self.barrier_pipe_ready,
            tensor_pipe_ready: self.tensor_pipe_ready,
            ldgsts_destination_pipe_ready: self.ldgsts_destination_pipe_ready,
            warp_states,
            block_states,
            omitted_state: vec!["legacy_core_debug_fields_retained".to_owned()],
        }
    }

    pub(crate) fn enable_memory_admission_observations(&mut self) {
        self.emit_admission_witnesses = true;
    }

    pub(crate) fn install_memory_admission_feedback(
        &mut self,
        candidate: MemoryAdmissionCandidate,
    ) -> Result<(), IssueError> {
        candidate.validate_feedback(self.params.read_barrier_latency)?;
        self.memory_admission_feedback =
            Some(Box::new(MemoryAdmissionFeedbackState::checked(candidate)?));
        self.emit_admission_witnesses = true;
        Ok(())
    }

    pub(crate) fn complete_memory_admission_feedback(
        &mut self,
        witness: AdmissionRequestWitness,
        operation: MemoryOperation,
        hit_sectors: u64,
        miss_sectors: u64,
        l2_requests: u64,
        unique_regions: u64,
        issue_time: u64,
    ) -> Result<Option<MemoryWriteAck>, IssueError> {
        self.memory_admission_feedback
            .as_mut()
            .ok_or(IssueError::MemoryAdmissionFeedbackReservationMissing {
                warp: witness.warp(),
                instruction: witness.instruction(),
            })?
            .complete(
                witness,
                operation,
                hit_sectors,
                miss_sectors,
                l2_requests,
                unique_regions,
                issue_time,
            )
    }

    pub(crate) fn complete_memory_write_ack(
        &mut self,
        ack: MemoryWriteAck,
        now: u64,
    ) -> Result<(), IssueError> {
        self.memory_admission_feedback
            .as_mut()
            .ok_or(IssueError::MemoryWriteAckNotPending {
                warp: ack.witness.warp(),
                instruction: ack.witness.instruction(),
            })?
            .acknowledge_write(ack, now)
    }

    pub(crate) fn pending_memory_write_acks(&self) -> usize {
        self.memory_admission_feedback
            .as_deref()
            .map_or(0, MemoryAdmissionFeedbackState::pending_write_ack_count)
    }

    pub(crate) fn is_done(&self) -> bool {
        self.unfinished_warps == 0
    }

    /// Responses the system still owes this SM (diagnostics).
    pub(crate) fn outstanding_responses(&self) -> usize {
        match &self.mem {
            MemModel::L1Shared(st) => st.outstanding_replies.len(),
            _ => 0,
        }
    }

    /// Instructions issued so far (monotonic; the system driver's
    /// progress signal).
    pub(crate) fn instructions_issued(&self) -> u64 {
        self.issued
    }

    /// Result snapshot; meaningful once [`Self::advance`] returned
    /// [`Advance::Done`].
    pub(crate) fn result(&self) -> SmSimResult {
        let mut readiness = (*self.readiness).clone();
        if let Some(front_end) = self.instruction_front_end.as_ref() {
            readiness.instruction_front_end = Some(Box::new(front_end.diagnostics));
        }
        if let Some(memory_pipeline) = self.memory_issue_pipeline.as_ref() {
            readiness.memory_issue_pipeline = Some(Box::new(memory_pipeline.diagnostics));
        }
        if let Some(cache) = self.immediate_constant_cache.as_ref() {
            let mut diagnostics = ImmediateConstantCacheDiagnostics::default();
            for state in cache.iter() {
                let addend = state.diagnostics();
                diagnostics.references = diagnostics.references.saturating_add(addend.references);
                diagnostics.lookup_hits =
                    diagnostics.lookup_hits.saturating_add(addend.lookup_hits);
                diagnostics.lookup_misses = diagnostics
                    .lookup_misses
                    .saturating_add(addend.lookup_misses);
                diagnostics.pending_tag_hits = diagnostics
                    .pending_tag_hits
                    .saturating_add(addend.pending_tag_hits);
                diagnostics.tag_misses = diagnostics.tag_misses.saturating_add(addend.tag_misses);
                diagnostics.pending_merges = diagnostics
                    .pending_merges
                    .saturating_add(addend.pending_merges);
                diagnostics.lines_filled =
                    diagnostics.lines_filled.saturating_add(addend.lines_filled);
                diagnostics.miss_wait_events = diagnostics
                    .miss_wait_events
                    .saturating_add(addend.miss_wait_events);
                diagnostics.miss_wait_cycles = diagnostics
                    .miss_wait_cycles
                    .saturating_add(addend.miss_wait_cycles);
                diagnostics.maximum_pending_lines = diagnostics
                    .maximum_pending_lines
                    .max(addend.maximum_pending_lines);
            }
            if !diagnostics.is_empty() {
                readiness.immediate_constant_cache = Some(Box::new(diagnostics));
            }
        }
        readiness.memory_service_candidates = self.memory_service_candidates.diagnostics();
        if let Some(feedback) = self.memory_admission_feedback.as_ref() {
            readiness.memory_admission_feedback = Some(Box::new(feedback.diagnostics()));
            let write_ack = feedback.write_ack_diagnostics();
            if !write_ack.is_empty() {
                readiness.memory_write_ack = Some(Box::new(write_ack));
            }
        }
        let pipeline_completion = self
            .memory_issue_pipeline
            .as_ref()
            .filter(|pipeline| !pipeline.diagnostics.is_empty())
            .map(|pipeline| pipeline.diagnostics.terminal_shared_accept_cycle);
        let modeled_local_completion_cycles =
            match (pipeline_completion, self.local_completion_time) {
                (Some(pipeline), Some(local)) => Some(pipeline.max(local)),
                (Some(value), None) | (None, Some(value)) => Some(value),
                (None, None) => None,
            };
        let activity = SmActivity::from_lifetimes(
            self.warp_lifetimes.clone(),
            self.instructions_per_smsp,
            self.issue_active_cycles_per_smsp,
            self.params.max_resident_warps_per_sm,
        );
        SmSimResult {
            instructions_issued: self.issued,
            cycles: self.t,
            modeled_local_completion_cycles,
            l1: self.mem.l1().map(|c| c.counters),
            unmodeled_tails: self.unmodeled_tails,
            readiness,
            activity,
        }
    }

    /// Retire one warp after its final instruction has become semantically
    /// complete. Ordinary instructions call this immediately; a final
    /// BAR.SYNC calls it from the CTA-release path so the barrier remains a
    /// real synchronization point.
    fn retire_warp(&mut self, subcore: usize, warp_index: usize, t: u64) -> Result<(), IssueError> {
        if self.warps[warp_index].done {
            return Ok(());
        }
        let block = self.warps[warp_index].block;
        let admission_time = self.warps[warp_index].admission_time;
        let warp_slot = self.warps[warp_index].warp_slot;
        let pos = self.warps[warp_index].pos_in_subcore;
        if self.subcore_warps[subcore].get(pos).copied() != Some(warp_index) {
            return Err(IssueError::CounterUnderflow {
                counter: "sub-core resident warp",
            });
        }
        self.warp_lifetimes.push(WarpLifetime {
            block: self.warps[warp_index].launch_index,
            warp_slot,
            admission: admission_time,
            retirement: checked_cycle_add("warp activity end", t, 1)?,
        });
        // A final BAR.SYNC is parked with itself in the CGGTY hold slot while
        // it waits for the CTA rendezvous.  Clear that slot before removing
        // the warp: swap_remove can otherwise leave a stale index that now
        // names a different resident warp (or a done warp), changing the
        // next pick/fetch decision after release.  Ordinary retirement also
        // benefits from the same invariant.
        if self.hold[subcore] == Some(warp_index) {
            self.hold[subcore] = None;
        }
        self.warps[warp_index].done = true;
        self.warps[warp_index].next_slot = None;
        self.warps[warp_index].ordinary_dependency_wait_mask = WaitMask::NONE;
        self.warps[warp_index].at_block_barrier = false;
        self.warps[warp_index].block_barrier_dispatch = None;
        self.warps[warp_index].block_barrier_arrival = None;
        self.unfinished_warps =
            self.unfinished_warps
                .checked_sub(1)
                .ok_or(IssueError::CounterUnderflow {
                    counter: "unfinished warp",
                })?;
        self.block_states[block].unfinished = self.block_states[block]
            .unfinished
            .checked_sub(1)
            .ok_or(IssueError::CounterUnderflow {
                counter: "block unfinished warp",
            })?;

        // Retire from the sub-core pick list (swap-remove + position fix-up)
        // so scans remain bounded by resident warps.
        self.subcore_warps[subcore].swap_remove(pos);
        if let Some(&moved) = self.subcore_warps[subcore].get(pos) {
            self.warps[moved].pos_in_subcore = pos;
        }

        if self.block_states[block].unfinished == 0 {
            let state = &self.block_states[block];
            self.resident_warps = self.resident_warps.checked_sub(state.warp_count).ok_or(
                IssueError::CounterUnderflow {
                    counter: "resident warp",
                },
            )?;
            self.resident_shared_memory_bytes = self
                .resident_shared_memory_bytes
                .checked_sub(state.shared_memory_bytes)
                .ok_or(IssueError::CounterUnderflow {
                    counter: "resident shared-memory byte",
                })?;
            for &warp_slot in &state.warp_slots {
                let removed = self.resident_warp_slots.remove(&warp_slot);
                if !removed {
                    return Err(IssueError::ResidentWarpSlotMissing {
                        launch_index: self.warps[state.first_warp].launch_index,
                        warp_slot,
                    });
                }
            }
            let admit_at = checked_cycle_add("block-wave admission", t, 1)?;
            self.admit_blocks(admit_at)?;
        }
        Ok(())
    }

    /// Admit blocks in launch order while all modeled constraints allow: the
    /// resident-block and resident-warp ceilings, the architecture-card
    /// shared-memory capacity, and disjoint physical warp slots across
    /// concurrent CTAs. A colliding later CTA
    /// waits for the earlier CTA to retire; reuse after retirement is valid. A
    /// block whose warps are ALL empty occupies no residency (it can never retire
    /// through the issue path, so counting it would leak resources and block later
    /// admissions forever).
    fn admit_blocks(&mut self, t: u64) -> Result<(), IssueError> {
        while self.next_block < self.blocks.len() {
            let blk = &self.blocks[self.next_block];
            let live_warps = self.block_live_warps[self.next_block];
            let block_shared_memory_bytes = if live_warps == 0 {
                0
            } else {
                self.shared_memory_residency.bytes_per_cta()
            };
            let resident_blocks = self
                .block_states
                .iter()
                .filter(|state| state.unfinished != 0)
                .count() as u32;
            let next_resident_blocks = resident_blocks
                .checked_add(u32::from(live_warps != 0))
                .ok_or(IssueError::CounterOverflow {
                    counter: "resident blocks",
                })?;
            let Some(next_resident_warps) = self.resident_warps.checked_add(live_warps) else {
                // Construction proved this CTA fits alone, so an overflowing sum
                // can only mean it must wait for the current wave to retire.
                if self.resident_warps > 0 {
                    break;
                }
                return Err(IssueError::CounterOverflow {
                    counter: "resident warp",
                });
            };
            let Some(next_resident_shared_memory_bytes) = self
                .resident_shared_memory_bytes
                .checked_add(block_shared_memory_bytes)
            else {
                if self.resident_warps > 0 {
                    break;
                }
                return Err(IssueError::CounterOverflow {
                    counter: "resident shared-memory byte",
                });
            };
            let pending_slots = blk
                .warps
                .iter()
                .filter(|warp| !warp.insts.is_empty())
                .map(|warp| warp.warp_slot)
                .collect::<Vec<_>>();
            let slots_collide = pending_slots
                .iter()
                .any(|slot| self.resident_warp_slots.contains(slot));
            if self.resident_warps > 0
                && (next_resident_blocks > self.params.max_resident_blocks_per_sm
                    || next_resident_warps > self.params.max_resident_warps_per_sm
                    || next_resident_shared_memory_bytes
                        > self.shared_memory_residency.max_bytes_per_sm()
                    || slots_collide)
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
                let subcore =
                    usize::try_from(w.warp_slot).map_err(|_| IssueError::WarpSlotOutOfRange {
                        launch_index: blk.launch_index,
                        warp_slot: w.warp_slot,
                        arch: self.params.arch,
                        physical_warp_slots: self.physical_warp_slots,
                    })? % SUB_CORES;
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
                let next_slot = lookup_dense(
                    &self.schedule_slots,
                    self.schedule,
                    w.insts[0].pc,
                    blk.launch_index,
                    w.warp_slot,
                )?;
                pending_warps.push((
                    subcore,
                    WarpState {
                        insts: &w.insts,
                        sectors: &w.sectors,
                        async_copy_cohorts: w.async_copy_cohorts.as_deref(),
                        next: 0,
                        front_end_fetch_cursor: 0,
                        instruction_buffer: VecDeque::new(),
                        next_slot: Some(next_slot),
                        current_since: t,
                        admission_time: t,
                        eligible: t,
                        imc_wait_until: 0,
                        ordinary_dependency_wait_mask: next_slot.ordinary_dependency_wait_mask(),
                        barrier_clear: [0; DEP_BARRIERS],
                        barrier_origin: [DependencyOrigin::Unknown; DEP_BARRIERS],
                        barrier_pending: [0; DEP_BARRIERS],
                        register_ready: Vec::new(),
                        block: block_ix,
                        launch_index: blk.launch_index,
                        warp_slot: w.warp_slot,
                        pos_in_subcore,
                        arrival,
                        at_block_barrier: false,
                        cta_barrier_ready: 0,
                        barrier_pipeline_ready: 0,
                        branch_pipeline_ready: 0,
                        block_barrier_dispatch: None,
                        block_barrier_arrival: None,
                        clock_first_issue: None,
                        clock_reads_issued: 0,
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
                if self.has_async_copy_protocol {
                    self.async_copies.push(AsyncCopySlot::default());
                }
            }
            debug_assert!(
                if self.has_async_copy_protocol {
                    self.warps.len() == self.async_copies.len()
                } else {
                    self.async_copies.is_empty()
                },
                "async-copy sidecar shape must match its construction-time protocol mode"
            );
            for &slot in &pending_slots {
                let inserted = self.resident_warp_slots.insert(slot);
                debug_assert!(
                    inserted,
                    "admission preflight rejected resident slot collisions"
                );
            }
            self.block_states.push(BlockState {
                unfinished: admitted_warps,
                waiting: 0,
                warp_count: live_warps,
                shared_memory_bytes: block_shared_memory_bytes,
                warp_slots: pending_slots.into_boxed_slice(),
                first_warp,
                admitted_warps,
                barrier_generation: 0,
                barrier_first_dispatch: None,
                barrier_first_arrival: None,
                barrier_rearm_ready: 0,
            });
            self.resident_warps = next_resident_warps;
            self.resident_shared_memory_bytes = next_resident_shared_memory_bytes;
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
    /// the KNOWN data-ready component (`None` when completion depends only on
    /// in-flight responses); unknown components remain in ordinary-barrier
    /// pending units and/or async-copy-group source units.
    fn mem_step_shared(
        &mut self,
        subcore: usize,
        wix: usize,
        instruction: usize,
        kind: AccessKind,
        constant_load: bool,
        constant_deep: bool,
        sectors: &'a [i64],
        write_barrier: Option<DepBarrier>,
        destination_wavefronts: u64,
        memory_time: u64,
        out: &mut Vec<MemRequest>,
    ) -> Result<Option<u64>, IssueError> {
        let t = memory_time;
        let p = self.params;
        let emit_admission_witnesses = self.emit_admission_witnesses;
        let shared_write_lookup_observer = self.shared_write_lookup_runtime.observer.clone();
        // The observer is shared across SM actors, but an instruction's memory
        // step is still a transaction: later request/counter validation may
        // fail after the write stream has been classified.  Prepare only the
        // touched sets and publish the plan after the complete step succeeds;
        // this avoids both leaked tags and an O(total-set-count) clone on every
        // store instruction.
        let mut staged_write_lookup_plan: Option<L1WriteLookupObserverPlan> = None;
        let MemModel::L1Shared(st) = &mut self.mem else {
            unreachable!("mem_step_shared is only dispatched in shared mode")
        };
        let warps = &mut self.warps;
        let async_copies = &mut self.async_copies;
        match kind {
            AccessKind::PredicatedOffMemory => Ok(None),
            AccessKind::NonMemory => {
                if write_barrier.is_some() {
                    p.non_memory_ready(t).map(Some)
                } else {
                    Ok(None)
                }
            }
            AccessKind::Shared { wavefronts } => {
                if write_barrier.is_some() {
                    p.shared_ready(t, wavefronts).map(Some)
                } else {
                    Ok(None)
                }
            }
            AccessKind::GlobalEligible { off, len, .. } | AccessKind::LocalLoad { off, len } => {
                let origin = match kind {
                    AccessKind::GlobalEligible { .. } => MemoryOrigin::Global,
                    AccessKind::LocalLoad { .. } => MemoryOrigin::Local,
                    _ => unreachable!("match arm admits only hierarchy loads"),
                };
                // The hit-pipeline floor applies to the whole dependent load
                // (as in L1Const); misses ride their responses. Line fills
                // emitted at the same issue timestamp (st.line_folds,
                // cleared at batch entry) fold later same-line sectors into
                // their request mask — one L2 transaction per line per
                // timestamp, priced on its worst sector.
                let target = write_barrier.map(FillTarget::Barrier);
                let lookup_batch = st.l1.begin_lookup_batch();
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
                        target,
                        None,
                        origin,
                        constant_load,
                        constant_deep,
                        sectors[i],
                        ResponseId::new(st.response_owner, wix, instruction, i),
                        t,
                        out,
                    ) {
                        Ok(Some(ready)) => {
                            let ready = checked_constant_return_time(
                                ready,
                                constant_deep && target.and_then(FillTarget::barrier).is_some(),
                                &p,
                                "constant L1-hit return latency",
                            )?;
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
                                origin,
                                constant_load,
                                constant_deep,
                                target,
                                mshr_floor: None,
                                lookup_batch,
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
            AccessKind::GlobalAsyncCopy {
                off,
                len,
                completion_work,
                l1_policy: LdgstsL1Policy::CacheAtAllLevels,
            } => {
                if st.l1_visibility_requester.is_some() {
                    return Err(IssueError::VisibilityAsyncCopyUnsupported);
                }
                let source_active = completion_work != 0;
                let hit_service_interval = self
                    .params
                    .ldgsts_l1_hit_return_service
                    .map_or(0, |service| service.interval_cycles);
                let mut hit_service_ready =
                    self.shared_write_lookup_runtime.async_l1_hit_service_ready[subcore];
                // The return-port token is per warp-level LDGSTS source
                // instruction, not per 32-byte sector.  The trace `len`
                // already represents the sectors coalesced by one SASS
                // instruction; charging the interval once for every sector
                // multiplies the source frontier by the payload width.
                let mut instruction_hit_service_time = None;
                let range = off..off + usize::from(len);
                let delivery_work = async_copy_completion_work(kind).unwrap_or(0);
                let async_group = if source_active {
                    Some(async_copies[wix].add_pending_async_copy_sources(
                        wix,
                        u32::from(len),
                        delivery_work,
                        destination_wavefronts,
                        None,
                        &p,
                    )?)
                } else {
                    None
                };
                let target = async_group
                    .map(|group| FillTarget::AsyncCopy {
                        group,
                        barrier: write_barrier,
                    })
                    .or_else(|| write_barrier.map(FillTarget::Barrier));
                let lookup_batch = st.l1.begin_async_copy_lookup_batch();
                let mut known = (write_barrier.is_some() || source_active)
                    .then(|| {
                        checked_cycle_add(
                            "LDGSTS shared L1 hit-pipeline floor",
                            t,
                            p.l1_hit_latency,
                        )
                    })
                    .transpose()?;
                for i in range.clone() {
                    match access_shared(
                        st,
                        warps,
                        wix,
                        target,
                        Some(p.ldgsts_l1_mshr_floor),
                        MemoryOrigin::Global,
                        false,
                        false,
                        sectors[i],
                        ResponseId::new(st.response_owner, wix, instruction, i),
                        t,
                        out,
                    ) {
                        Ok(Some(ready)) => {
                            let ready = if hit_service_interval == 0 {
                                ready
                            } else if let Some(serviced) = instruction_hit_service_time {
                                serviced
                            } else {
                                let serviced = ready.max(hit_service_ready);
                                hit_service_ready = serviced
                                    .checked_add(hit_service_interval)
                                    .ok_or(IssueError::CounterOverflow {
                                        counter: "async L1 hit service timeline",
                                    })?;
                                instruction_hit_service_time = Some(serviced);
                                serviced
                            };
                            if let Some(group) = async_group {
                                async_copies[wix]
                                    .complete_async_copy_sources(wix, group, None, 1, ready, &p)?;
                            }
                            if let Some(current) = &mut known {
                                *current = (*current).max(ready);
                            }
                        }
                        Ok(None) => {}
                        Err(SharedAccessError::Stalled) => {
                            // Async source units were counted at original issue;
                            // only the optional ordinary barrier needs a retry
                            // replacement unit.
                            warps[wix].add_pending(wix, write_barrier, 1)?;
                            st.retry.push_back(Retry {
                                warp: wix,
                                instruction,
                                origin: MemoryOrigin::Global,
                                constant_load: false,
                                constant_deep: false,
                                target,
                                mshr_floor: Some(p.ldgsts_l1_mshr_floor),
                                lookup_batch,
                                lo: i,
                                hi: range.end,
                            });
                            break;
                        }
                        Err(SharedAccessError::Issue(source)) => return Err(source),
                    }
                }
                self.shared_write_lookup_runtime.async_l1_hit_service_ready[subcore] =
                    hit_service_ready;
                Ok(known)
            }
            AccessKind::GlobalBypass { off, len, .. }
            | AccessKind::GlobalAsyncCopy {
                off,
                len,
                l1_policy: LdgstsL1Policy::BypassL1,
                ..
            }
            | AccessKind::GlobalUntagged { off, len } => {
                let origin = MemoryOrigin::Global;
                let (source, counted, operation, async_copy) = match kind {
                    // ATOM rides bypass provenance but remains a returning RMW
                    // operation; every other bypass / untagged sector is a read.
                    AccessKind::GlobalBypass { token, .. } => (
                        L2Source::Bypass,
                        true,
                        if token == CacheOpToken::Atom {
                            MemoryOperation::Atomic
                        } else {
                            MemoryOperation::Read
                        },
                        token == CacheOpToken::Ldgsts,
                    ),
                    AccessKind::GlobalAsyncCopy {
                        completion_work, ..
                    } => (
                        L2Source::Bypass,
                        true,
                        MemoryOperation::Read,
                        completion_work != 0,
                    ),
                    AccessKind::GlobalUntagged { .. } => {
                        (L2Source::Untagged, false, MemoryOperation::Read, false)
                    }
                    _ => unreachable!("match arm admits only bypass, async-copy, or untagged"),
                };
                if operation == MemoryOperation::Atomic && st.l1_visibility_requester.is_some() {
                    return Err(IssueError::VisibilityAtomicUnsupported);
                }
                if async_copy && st.l1_visibility_requester.is_some() {
                    return Err(IssueError::VisibilityAsyncCopyUnsupported);
                }
                if counted {
                    st.l1.count_bypass(
                        origin,
                        u64::from(len),
                        operation == MemoryOperation::Atomic,
                    )?;
                } else {
                    st.l1.count_untagged(origin, u64::from(len))?;
                }
                let async_group = if async_copy {
                    let delivery_work = async_copy_completion_work(kind).unwrap_or(0);
                    let exact_cohorts = warps[wix]
                        .async_copy_cohorts
                        .and_then(|sidecar| sidecar.view(instruction));
                    Some(
                        async_copies[wix].add_pending_async_copy_sources(
                            wix,
                            u32::from(len),
                            delivery_work,
                            destination_wavefronts,
                            Some(
                                bypass_async_copy_observed_topology(
                                    kind,
                                    instruction,
                                    exact_cohorts,
                                )
                                .unwrap_or(BypassAsyncCopyInput {
                                    instruction,
                                    sector_off: off,
                                    sector_len: len,
                                    source_cohorts: 0,
                                    destination_lines: 0,
                                    max_bank_wavefronts: 0,
                                    max_bypass_wavefronts: 0,
                                    cohorts: None,
                                    sector_membership_masks: None,
                                }),
                            ),
                            &p,
                        )?,
                    )
                } else {
                    None
                };
                if len == 0 {
                    // Degenerate records have no source requests. A topology
                    // carrying only executing ignore-src lanes still owns an
                    // async group, so create it before returning; otherwise a
                    // later commit/wait would silently drop destination work.
                    return if write_barrier.is_some() {
                        p.store_visibility_ready(t).map(Some)
                    } else {
                        Ok(None)
                    };
                }
                for (relative, &sector_id) in
                    sectors[off..off + usize::from(len)].iter().enumerate()
                {
                    let i = off + relative;
                    let response_id = ResponseId::new(st.response_owner, wix, instruction, i);
                    let reply = if let Some(group) = async_group {
                        if let Some(barrier) = write_barrier {
                            warps[wix].add_pending(wix, Some(barrier), 1)?;
                        }
                        let reply = ReplyTo::AsyncCopy {
                            warp: wix,
                            group,
                            barrier: write_barrier,
                            response_id,
                        };
                        register_response(st, reply)?;
                        reply
                    } else if let Some(barrier) = write_barrier {
                        let reply = ReplyTo::Warp {
                            warp: wix,
                            barrier,
                            response_id,
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
                        origin,
                        sector_mask: 1 << (sector_id & 3) as u8,
                        local_sector_mask: 0,
                        source,
                        operation,
                        constant_load: false,
                        constant_deep: false,
                        reply,
                        admission_witness: None,
                    });
                }
                Ok(None)
            }
            AccessKind::GlobalStore { off, len }
            | AccessKind::GlobalReduction { off, len, .. }
            | AccessKind::LocalStore { off, len } => {
                let origin = match kind {
                    AccessKind::GlobalStore { .. } | AccessKind::GlobalReduction { .. } => {
                        MemoryOrigin::Global
                    }
                    AccessKind::LocalStore { .. } => MemoryOrigin::Local,
                    _ => unreachable!("match arm admits only hierarchy writes"),
                };
                let reduction = matches!(kind, AccessKind::GlobalReduction { .. });
                let store_sectors = &sectors[off..off + usize::from(len)];
                if shared_write_lookup_observer.is_some()
                    && matches!(kind, AccessKind::GlobalStore { .. })
                {
                    let plan = if let Some(observer) = shared_write_lookup_observer.as_ref() {
                        observer.borrow().plan_batch(store_sectors)?
                    } else {
                        unreachable!("shared observer branch has no observer")
                    };
                    staged_write_lookup_plan = Some(plan);
                } else if let Some(hits) = projected_l1_write_lookup_hits(kind, sectors) {
                    st.l1.count_write_lookup_hits(origin, hits)?;
                }
                for (request_index, &sector_id) in
                    sectors[off..off + usize::from(len)].iter().enumerate()
                {
                    let mut request = if reduction {
                        if st.l1_visibility_requester.is_some() {
                            return Err(IssueError::VisibilityReductionUnsupported);
                        }
                        let hit = st.l1.probe_reduction(origin, sector_id, t)?;
                        if origin == MemoryOrigin::Global {
                            record_global_l1_observation(st, sector_id, true, false, hit, None, t)?;
                        }
                        MemRequest::reduction(t, sector_id)
                    } else {
                        let hit = st.l1.probe_store(origin, sector_id, t)?;
                        let retained_l1_write = origin == MemoryOrigin::Global
                            && hit
                            && st.l1.store_policy() == L1StorePolicy::ResidentHitWriteBack;
                        if origin == MemoryOrigin::Global
                            && let Some(writer) =
                                st.payload_store_writers.as_mut().and_then(|writers| {
                                    writers.get_mut(&(
                                        sector_id.div_euclid(4),
                                        sector_id.rem_euclid(4) as u8,
                                    ))
                                })
                        {
                            *writer = Some(PayloadStoreObservation {
                                site: PayloadStoreSite {
                                    launch_index: warps[wix].launch_index,
                                    warp_slot: warps[wix].warp_slot,
                                    instruction_index: instruction,
                                },
                                accepted_at: t,
                                resident_generation: st
                                    .l1
                                    .resident_payload_state(sector_id.div_euclid(4))
                                    .map(|(generation, _)| generation),
                                retained_dirty: retained_l1_write,
                                attached: false,
                            });
                        }
                        if origin == MemoryOrigin::Global {
                            record_global_l1_observation(
                                st,
                                sector_id,
                                true,
                                retained_l1_write,
                                hit,
                                None,
                                t,
                            )?;
                        }
                        if retained_l1_write {
                            // The resident hit is retained dirty in L1; the
                            // lower L2 mutation occurs only at the explicit
                            // flush boundary, never also on this issue path.
                            continue;
                        }
                        MemRequest::store(t, sector_id, origin)
                    };
                    if origin == MemoryOrigin::Global && emit_admission_witnesses {
                        request.admission_witness = Some(
                            AdmissionRequestWitness::new(
                                u32::try_from(wix).map_err(|_| IssueError::CounterOverflow {
                                    counter: "memory-admission witness warp",
                                })?,
                                u32::try_from(instruction).map_err(|_| {
                                    IssueError::CounterOverflow {
                                        counter: "memory-admission witness instruction",
                                    }
                                })?,
                                u8::try_from(subcore).map_err(|_| IssueError::CounterOverflow {
                                    counter: "memory-admission witness subcore",
                                })?,
                                u16::try_from(request_index).map_err(|_| {
                                    IssueError::CounterOverflow {
                                        counter: "memory-admission witness request index",
                                    }
                                })?,
                                len,
                            )
                            .ok_or(IssueError::CounterOverflow {
                                counter: "memory-admission witness encoding",
                            })?,
                        );
                    }
                    out.push(request);
                }
                // A present write barrier is priced at the L2-hit arm (as in
                // L1Const); stores and reductions do not wait on responses.
                let visibility_ready = if write_barrier.is_some() {
                    p.store_visibility_ready(t).map(Some)?
                } else {
                    None
                };
                if let Some(plan) = staged_write_lookup_plan {
                    st.l1.count_write_lookup_hits(origin, plan.hits())?;
                    if let Some(observer) = shared_write_lookup_observer.as_ref() {
                        observer.borrow_mut().commit_plan(plan);
                    }
                }
                Ok(visibility_ready)
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
            ReplyTo::AsyncCopy {
                warp,
                group,
                barrier,
                response_id,
            } => {
                if warp >= self.warps.len() {
                    return Err(IssueError::ResponseWarpOutOfRange {
                        warp,
                        warps: self.warps.len(),
                    });
                }
                self.async_copies[warp].preflight_async_copy_source_completion(
                    warp,
                    group,
                    Some(response_id),
                    1,
                    now,
                    &self.params,
                )?;
                if let Some(barrier) = barrier {
                    self.warps[warp].preflight_pending_remove(warp, barrier, 1)?;
                }
                return Ok(());
            }
            ReplyTo::L1Fill { fill, .. } => {
                if !st.l1.is_fill_in_flight(fill) {
                    return Err(IssueError::ResponseFillNotInFlight { fill });
                }
                if let FillKey::Line { line, .. } = fill
                    && !st.fill_masks.contains_key(&line)
                {
                    return Err(IssueError::FillMaskMissing { line });
                }
                fill
            }
            ReplyTo::None => return Err(IssueError::UnexpectedMutationOnlyResponse),
        };

        let mut removals = HashMap::new();
        let mut async_removals = HashMap::new();
        if let Some(waiters) = st.fill_waiters.get(&fill) {
            waiters.try_for_each(|waiter| {
                fill_waiter_ready_time(waiter, now, &self.params)?;
                accumulate_waiter_removals(&mut removals, &mut async_removals, waiter)
            })?;
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
        let mut projected_async = HashMap::new();
        for (&(warp, group), &remove) in &async_removals {
            project_async_copy_removal(
                &mut projected_async,
                &self.async_copies,
                warp,
                group,
                remove,
                now,
                &self.params,
            )?;
        }

        if st.retry.is_empty() {
            let budget = L1MutationBudget {
                recency_touches: 1,
                ..L1MutationBudget::for_origin(MemoryOrigin::Global)
            };
            if st.l1.preflight_mutation_budget(budget).is_err() {
                let mut shadow = st.l1.clone();
                shadow.complete_fill(fill)?;
            }
            return Ok(());
        }

        let mut budgets = MemoryOrigin::ALL.map(L1MutationBudget::for_origin);
        budgets[MemoryOrigin::Global.index()].recency_touches = 1;
        let mut pending_budget_fits = true;
        let mut has_async_retry = false;
        for retry in &st.retry {
            let budget = &mut budgets[retry.origin.index()];
            if retry.warp >= self.warps.len() {
                return Err(IssueError::ResponseWarpOutOfRange {
                    warp: retry.warp,
                    warps: self.warps.len(),
                });
            }
            has_async_retry |= retry.target.and_then(FillTarget::async_group).is_some();
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
                if let Some(&reply) = st.outstanding_response_ids.get(&response_id) {
                    return Err(IssueError::ResponseTokenAlreadyOutstanding { reply });
                }
            }
            if let Some(barrier) = retry.target.and_then(FillTarget::barrier) {
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

        if st.l1.preflight_mutation_budgets(&budgets).is_ok()
            && pending_budget_fits
            && !has_async_retry
        {
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
        let mut exact_async = projected_async;
        'retries: for retry in &st.retry {
            shadow.resume_lookup_batch(retry.lookup_batch);
            let barrier = retry.target.and_then(FillTarget::barrier);
            if let Some(barrier) = barrier {
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
                match access_shared_l1(
                    &mut shadow,
                    retry.origin,
                    sector_id,
                    now,
                    retry.target.is_some(),
                    retry.mshr_floor,
                )? {
                    SharedL1Access::Hit { .. } => {
                        if let Some(group) = retry.target.and_then(FillTarget::async_group) {
                            project_async_copy_removal(
                                &mut exact_async,
                                &self.async_copies,
                                retry.warp,
                                group,
                                1,
                                now,
                                &self.params,
                            )?;
                        }
                    }
                    SharedL1Access::MissRequest { .. } | SharedL1Access::MergedPending { .. } => {
                        if let Some(barrier) = barrier {
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
                        if let Some(barrier) = barrier {
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
        let observes_hit_ready = st
            .retry
            .iter()
            .any(|retry| retry.target.and_then(FillTarget::barrier).is_some());
        let hit_timing_fits = !observes_hit_ready
            || st
                .retry
                .iter()
                .filter(|retry| retry.target.and_then(FillTarget::barrier).is_some())
                .all(|retry| {
                    now.checked_add(self.params.l1_hit_latency)
                        .is_some_and(|hit_ready| {
                            checked_constant_return_time(
                                hit_ready,
                                retry.constant_deep,
                                &self.params,
                                "constant retry L1-hit return latency",
                            )
                            .is_ok()
                        })
                });
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
            shadow.resume_lookup_batch(retry.lookup_batch);
            for sector_index in retry.lo..retry.hi {
                let sector_id = self.warps[retry.warp].sectors[sector_index];
                match access_shared_l1(
                    &mut shadow,
                    retry.origin,
                    sector_id,
                    now,
                    false,
                    retry.mshr_floor,
                )? {
                    SharedL1Access::Hit { .. } => {
                        if retry.target.and_then(FillTarget::barrier).is_some() {
                            let hit_ready = checked_cycle_add(
                                "retry L1 hit visibility",
                                now,
                                self.params.l1_hit_latency,
                            )?;
                            checked_constant_return_time(
                                hit_ready,
                                retry.constant_deep,
                                &self.params,
                                "constant retry L1-hit return latency",
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
            ReplyTo::Warp { warp, .. } | ReplyTo::AsyncCopy { warp, .. }
                if warp >= self.warps.len() =>
            {
                return Err(IssueError::ResponseWarpOutOfRange {
                    warp,
                    warps: self.warps.len(),
                });
            }
            ReplyTo::None => return Err(IssueError::UnexpectedMutationOnlyResponse),
            _ => {}
        }
        let response_id = reply
            .response_id()
            .ok_or(IssueError::UnexpectedMutationOnlyResponse)?;
        if !st.outstanding_replies.contains(&reply)
            || st.outstanding_response_ids.get(&response_id) != Some(&reply)
        {
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
    pub(crate) fn on_response(
        &mut self,
        reply: ReplyTo,
        now: u64,
        out: &mut Vec<MemRequest>,
    ) -> Result<(), IssueError> {
        self.preflight_response(reply, now)?;
        let params = self.params;
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
        let response_id = reply
            .response_id()
            .ok_or(IssueError::UnexpectedMutationOnlyResponse)?;
        if !st.outstanding_replies.remove(&reply)
            || st.outstanding_response_ids.remove(&response_id) != Some(reply)
        {
            return Err(IssueError::ResponseTokenRemovalFailed { reply });
        }
        let response_mask = st.response_masks.get(&response_id).copied();
        st.response_masks.remove(&response_id);
        // New emission batch (the retry drain below may emit).
        st.line_folds.clear();
        let warps = &mut self.warps;
        let async_copies = &mut self.async_copies;
        match reply {
            ReplyTo::L1Fill { fill, .. } => {
                let releases_mshr = st.l1.fill_releases_mshr(fill);
                if let FillKey::Line { line, .. } = fill
                    && st.fill_masks.remove(&line).is_none()
                {
                    return Err(IssueError::FillMaskMissing { line });
                }
                let retained = st.l1.complete_fill(fill)?;
                if let Some(requester) = st.l1_visibility_requester {
                    if !retained {
                        return Err(IssueError::VisibilityFillDropped { fill });
                    }
                    let observed_mask =
                        response_mask.ok_or(IssueError::VisibilityResponseMaskMissing {
                            response: response_id,
                        })?;
                    let (line_id, expected_mask) = match fill {
                        FillKey::Line { line, .. } => (line, 0b1111),
                        FillKey::Sector { line, sector, .. } => (line, 1_u8 << sector),
                    };
                    let sector_mask = match fill {
                        FillKey::Line { .. }
                            if observed_mask != 0 && observed_mask & !0b1111 == 0 =>
                        {
                            observed_mask
                        }
                        FillKey::Sector { .. } if observed_mask == expected_mask => observed_mask,
                        _ => {
                            return Err(IssueError::VisibilityResponseMaskInvalid {
                                response: response_id,
                                expected: expected_mask,
                                got: observed_mask,
                            });
                        }
                    };
                    let generation = fill_generation(fill);
                    let sequence = st.next_l1_visibility_sequence()?;
                    st.l1_fill_visibility_observations
                        .push(sim_mem::L1FillVisibleObservation {
                            sequence,
                            requester,
                            line_id,
                            sector_mask,
                            fill_generation: generation,
                            visible_cycle: now,
                        });
                }
                if let Some(waiters) = st.fill_waiters.remove(&fill) {
                    waiters.try_for_each(|waiter| {
                        let ready = fill_waiter_ready_time(waiter, now, &params)?;
                        if let Some(group) = waiter.target.async_group() {
                            async_copies[waiter.warp].complete_async_copy_sources(
                                waiter.warp,
                                group,
                                None,
                                waiter.units,
                                ready,
                                &params,
                            )?;
                        }
                        if let Some(barrier) = waiter.target.barrier() {
                            warps[waiter.warp].clear_pending(
                                waiter.warp,
                                barrier,
                                waiter.units,
                                ready,
                            )?;
                        }
                        Ok(())
                    })?;
                }
                // Only an independently occupied MSHR completion can drain
                // parked accesses. A late sector child shares its parent
                // line allocation slot and must not cause a speculative
                // retry while that parent remains in flight.
                if releases_mshr {
                    drain_retries(st, warps, async_copies, now, &params, out)?;
                }
            }
            ReplyTo::Warp { warp, barrier, .. } => {
                warps[warp].clear_pending(warp, barrier, 1, now)?;
            }
            ReplyTo::AsyncCopy {
                warp,
                group,
                barrier,
                response_id,
            } => {
                async_copies[warp].complete_async_copy_sources(
                    warp,
                    group,
                    Some(response_id),
                    1,
                    now,
                    &params,
                )?;
                if let Some(barrier) = barrier {
                    warps[warp].clear_pending(warp, barrier, 1, now)?;
                }
            }
            ReplyTo::None => return Err(IssueError::UnexpectedMutationOnlyResponse),
        }
        Ok(())
    }

    fn project_selected_readiness(
        &self,
        subcore: usize,
        warp_index: usize,
        now: u64,
        projected: &mut ReadinessDiagnostics,
    ) -> Result<(), IssueError> {
        let warp = &self.warps[warp_index];
        let slot = warp
            .next_slot
            .ok_or(IssueError::MissingIssueSlot { warp: warp_index })?;
        let kind = warp.insts[warp.next].kind;
        if let Some((class, sectors, wavefronts, unique_lines, atomic_l2_requests)) =
            projected_global_memory_issue(kind, warp.sectors)
        {
            projected
                .global_memory_issue
                .observe(
                    class,
                    sectors,
                    wavefronts,
                    unique_lines,
                    atomic_l2_requests,
                    slot.control.read_barrier.is_some(),
                    now,
                )
                .map_err(|source| match source {
                    ReadinessCounterError::Overflow { field, .. } => {
                        IssueError::CounterOverflow { counter: field }
                    }
                    mismatch @ (ReadinessCounterError::MemoryServiceCandidateMismatch {
                        ..
                    }
                    | ReadinessCounterError::MemoryAdmissionCandidateMismatch {
                        ..
                    }) => IssueError::ReadinessDiagnostic(mismatch),
                })?;
        }
        let base = warp.current_since;
        let mut ready_at = base;
        let mut limiting = None;
        let mut tied = 0;

        if self.instruction_front_end.is_some() {
            let until = warp.instruction_front_end_ready_at().ok_or(
                IssueError::InstructionBufferMissing {
                    warp: warp_index,
                    instruction: warp.next,
                },
            )?;
            observe_readiness_constraint(
                projected,
                base,
                until,
                ReadinessCause::InstructionFrontEnd,
                &mut ready_at,
                &mut limiting,
                &mut tied,
            )?;
        }
        if let Some(memory_pipeline) = self.memory_issue_pipeline.as_ref()
            && self
                .memory_pipeline_slot(u64::from(warp.insts[warp.next].pc))
                .is_some()
            && !matches!(kind, AccessKind::PredicatedOffMemory)
        {
            observe_readiness_constraint(
                projected,
                base,
                memory_pipeline.admission_ready_at(subcore),
                ReadinessCause::MemoryIssueQueue,
                &mut ready_at,
                &mut limiting,
                &mut tied,
            )?;
        }

        if let Some(feedback) = self.memory_admission_feedback.as_ref()
            && feedback.matches_kind(kind)
        {
            let until = feedback
                .admission_ready_at(subcore, kind)
                .expect("selected feedback instruction has a priced queue frontier");
            observe_readiness_constraint(
                projected,
                base,
                until,
                ReadinessCause::MemoryAdmissionQueue,
                &mut ready_at,
                &mut limiting,
                &mut tied,
            )?;
        }
        if let Some(feedback) = self.memory_admission_feedback.as_ref()
            && feedback.candidate.membar_waits_for_service_completion
            && is_gpu_membar_issue(slot)
        {
            let until = feedback
                .completion_ready_at(warp_index, slot)
                .expect("selected MEMBAR has a priced write-completion frontier");
            observe_readiness_constraint(
                projected,
                base,
                until,
                ReadinessCause::MemoryWriteCompletion,
                &mut ready_at,
                &mut limiting,
                &mut tied,
            )?;
        }

        observe_readiness_constraint(
            projected,
            base,
            warp.eligible,
            ReadinessCause::ControlStall,
            &mut ready_at,
            &mut limiting,
            &mut tied,
        )?;
        observe_readiness_constraint(
            projected,
            base,
            warp.cta_barrier_ready,
            ReadinessCause::CtaBarrierRendezvous,
            &mut ready_at,
            &mut limiting,
            &mut tied,
        )?;
        observe_readiness_constraint(
            projected,
            base,
            warp.barrier_pipeline_ready,
            ReadinessCause::PipelineBarrier,
            &mut ready_at,
            &mut limiting,
            &mut tied,
        )?;
        observe_readiness_constraint(
            projected,
            base,
            warp.branch_pipeline_ready,
            ReadinessCause::PipelineBranch,
            &mut ready_at,
            &mut limiting,
            &mut tied,
        )?;

        for barrier in 0..DEP_BARRIERS {
            if !slot.control.wait_mask.contains_index(barrier) {
                continue;
            }
            debug_assert_eq!(
                warp.barrier_pending[barrier], 0,
                "a selected warp cannot have an unresolved dependence counter"
            );
            observe_readiness_constraint(
                projected,
                base,
                warp.barrier_clear[barrier],
                warp.barrier_origin[barrier].into(),
                &mut ready_at,
                &mut limiting,
                &mut tied,
            )?;
        }
        for barrier in 0..DEP_BARRIERS {
            if !warp.ordinary_dependency_wait_mask.contains_index(barrier)
                || slot.control.wait_mask.contains_index(barrier)
            {
                continue;
            }
            debug_assert_eq!(
                warp.barrier_pending[barrier], 0,
                "a selected explicit dependency-barrier wait cannot have an unresolved counter"
            );
            observe_readiness_constraint(
                projected,
                base,
                warp.barrier_clear[barrier],
                warp.barrier_origin[barrier].into(),
                &mut ready_at,
                &mut limiting,
                &mut tied,
            )?;
        }

        if self.has_async_copy_protocol && slot.op_class == OpClass::AsyncCopyCommit {
            let commits_nonempty = self.async_copies[warp_index]
                .state
                .as_deref()
                .and_then(|state| state.open_group)
                .is_some();
            if commits_nonempty {
                projected.async_groups.groups_committed = checked_u64_counter_add(
                    "committed asynchronous-copy group diagnostic",
                    projected.async_groups.groups_committed,
                    1,
                )?;
            }
        }

        if self.has_async_copy_protocol && slot.op_class == OpClass::AsyncCopyWait {
            let prefix = self.async_copies[warp_index]
                .async_wait_prefix_len(slot.async_copy_wait_max_pending_groups);
            let committed_groups_before = self.async_copies[warp_index]
                .state
                .as_deref()
                .map_or(0, |state| state.committed_groups.len());
            let wait_frontier = self.async_copies[warp_index]
                .state
                .as_deref()
                .and_then(|state| {
                    state
                        .committed_groups
                        .iter()
                        .take(prefix)
                        .map(|group| group.completion_time)
                        .max()
                })
                .unwrap_or(0);
            if let Some(diagnostics) = projected.async_wait_prefix.as_deref_mut() {
                diagnostics.observe(
                    slot.async_copy_wait_max_pending_groups,
                    u64::try_from(committed_groups_before).map_err(|_| {
                        IssueError::CounterOverflow {
                            counter: "async-wait committed-group cardinality diagnostic",
                        }
                    })?,
                    u64::try_from(prefix).map_err(|_| IssueError::CounterOverflow {
                        counter: "async-wait consumed-prefix cardinality diagnostic",
                    })?,
                    wait_frontier.saturating_sub(base),
                )?;
            }
            if let Some(state) = self.async_copies[warp_index].state.as_deref() {
                for group in state.committed_groups.iter().take(prefix) {
                    debug_assert!(
                        group.is_complete_at(now),
                        "a selected async wait requires a complete committed prefix"
                    );
                    let local_completion = group.local_completion_time(&self.params)?;
                    let cause = group.limiting_readiness_cause(&self.params)?;
                    observe_readiness_constraint(
                        projected,
                        base,
                        group.completion_time,
                        cause,
                        &mut ready_at,
                        &mut limiting,
                        &mut tied,
                    )?;

                    projected.async_groups.groups_waited = checked_u64_counter_add(
                        "waited asynchronous-copy group diagnostic",
                        projected.async_groups.groups_waited,
                        1,
                    )?;
                    let source_cycles = group
                        .source_completion_time
                        .saturating_sub(group.committed_at);
                    projected.async_groups.source_cycles = checked_u64_counter_add(
                        "asynchronous-copy committed-to-source diagnostic",
                        projected.async_groups.source_cycles,
                        source_cycles,
                    )?;
                    let local_cycles =
                        local_completion.saturating_sub(group.source_completion_time);
                    projected.async_groups.destination_local_cycles = checked_u64_counter_add(
                        "asynchronous-copy local destination diagnostic",
                        projected.async_groups.destination_local_cycles,
                        local_cycles,
                    )?;
                    let service_cycles = group.completion_time.saturating_sub(local_completion);
                    projected.async_groups.destination_sm_service_cycles = checked_u64_counter_add(
                        "asynchronous-copy SM destination-service diagnostic",
                        projected.async_groups.destination_sm_service_cycles,
                        service_cycles,
                    )?;
                    projected.async_groups.destination_groups_observed = checked_u64_counter_add(
                        "asynchronous-copy destination groups observed",
                        projected.async_groups.destination_groups_observed,
                        u64::from(group.destination_wavefronts != 0),
                    )?;
                    projected.async_groups.destination_wavefronts_observed =
                        checked_u64_counter_add(
                            "asynchronous-copy destination wavefronts observed",
                            projected.async_groups.destination_wavefronts_observed,
                            group.destination_wavefronts,
                        )?;
                    let hidden = group
                        .completion_time
                        .min(base)
                        .saturating_sub(group.committed_at);
                    projected.async_groups.completion_hidden_before_wait_cycles =
                        checked_u64_counter_add(
                            "asynchronous-copy hidden-before-wait diagnostic",
                            projected.async_groups.completion_hidden_before_wait_cycles,
                            hidden,
                        )?;
                    if let Some(observed) = self.async_copies[warp_index].bypass_copy_projection(
                        warp_index,
                        group,
                        &self.params,
                        base,
                    )? {
                        let diagnostics = projected
                            .bypass_async_copies
                            .get_or_insert_with(|| Box::new(BypassAsyncCopyDiagnostics::default()));
                        diagnostics.add_checked(&BypassAsyncCopyDiagnostics {
                            copies_observed: observed.copies,
                            destination_wavefronts_observed: observed.bank_wavefront_upper_bound,
                            bypass_wavefronts_observed: observed.bypass_wavefront_upper_bound,
                            source_cohorts_observed: observed.source_cohorts,
                            destination_lines_observed: observed.destination_lines,
                            max_bank_wavefronts_sum: observed.max_bank_wavefronts_sum,
                            max_bypass_wavefronts_sum: observed.max_bypass_wavefronts_sum,
                            copy_source_spread_cycles: observed.source_spread_cycles,
                            per_copy_completion_cycles: observed.per_copy_completion_cycles,
                            group_aggregate_excess_cycles: observed.group_aggregate_excess_cycles,
                            topology_candidate_penalty_cycles: observed
                                .topology_candidate_penalty_cycles,
                            topology_candidate_completion_cycles: observed
                                .topology_candidate_completion_cycles,
                            topology_candidate_later_cycles: observed
                                .topology_candidate_later_cycles,
                            topology_candidate_earlier_cycles: observed
                                .topology_candidate_earlier_cycles,
                            pipeline_groups_observed: 1,
                            pipeline_q1_completion_cycles: observed.pipeline_q1_completion_cycles,
                            pipeline_q1_exposure_cycles: observed.pipeline_q1_exposure_cycles,
                            pipeline_q2_completion_cycles: observed.pipeline_q2_completion_cycles,
                            pipeline_q2_exposure_cycles: observed.pipeline_q2_exposure_cycles,
                            pipeline_q4_completion_cycles: observed.pipeline_q4_completion_cycles,
                            pipeline_q4_exposure_cycles: observed.pipeline_q4_exposure_cycles,
                            pipeline_q8_completion_cycles: observed.pipeline_q8_completion_cycles,
                            pipeline_q8_exposure_cycles: observed.pipeline_q8_exposure_cycles,
                            pipeline_q16_completion_cycles: observed.pipeline_q16_completion_cycles,
                            pipeline_q16_exposure_cycles: observed.pipeline_q16_exposure_cycles,
                            cohort_groups_observed: observed.cohort_groups_observed,
                            cohort_copies_observed: observed.cohort_copies_observed,
                            cohort_response_memberships_observed: observed
                                .cohort_response_memberships_observed,
                            cohort_response_span_cycles: observed.cohort_response_span_cycles,
                            cohort_topology_penalty_cycles: observed.cohort_topology_penalty_cycles,
                            cohort_candidate_completion_cycles: observed
                                .cohort_candidate_completion_cycles,
                            cohort_candidate_exposure_cycles: observed
                                .cohort_candidate_exposure_cycles,
                            cohort_candidate_later_than_per_copy_cycles: observed
                                .cohort_candidate_later_than_per_copy_cycles,
                            cohort_candidate_earlier_than_per_copy_cycles: observed
                                .cohort_candidate_earlier_than_per_copy_cycles,
                            cohort_candidate_later_than_group_aggregate_cycles: observed
                                .cohort_candidate_later_than_group_aggregate_cycles,
                            cohort_candidate_earlier_than_group_aggregate_cycles: observed
                                .cohort_candidate_earlier_than_group_aggregate_cycles,
                            cohort_candidate_later_than_production_cycles: observed
                                .cohort_candidate_later_than_production_cycles,
                            cohort_candidate_earlier_than_production_cycles: observed
                                .cohort_candidate_earlier_than_production_cycles,
                            cohort_overlap_groups_observed: observed.cohort_overlap_groups_observed,
                            cohort_response_envelope_cycles: observed
                                .cohort_response_envelope_cycles,
                            cohort_topology_prearrival_hidden_cycles: observed
                                .cohort_topology_prearrival_hidden_cycles,
                            cohort_topology_stream_hidden_cycles: observed
                                .cohort_topology_stream_hidden_cycles,
                            cohort_topology_residual_exposure_cycles: observed
                                .cohort_topology_residual_exposure_cycles,
                        })?;
                    }
                    let limited_counter = match cause {
                        ReadinessCause::AsyncSource => {
                            &mut projected.async_groups.source_limited_groups
                        }
                        ReadinessCause::AsyncDestinationLocal => {
                            &mut projected.async_groups.destination_local_limited_groups
                        }
                        ReadinessCause::AsyncDestinationSmService => {
                            &mut projected.async_groups.destination_sm_service_limited_groups
                        }
                        _ => unreachable!("async-group classifier returns an async cause"),
                    };
                    *limited_counter = checked_u64_counter_add(
                        "asynchronous-copy limiting-arm diagnostic",
                        *limited_counter,
                        1,
                    )?;
                }
            }
        }

        if let PipelineClass::Branch {
            next_issue_interval_cycles,
        } = slot.pipeline_class
        {
            projected.control_pipeline.branch_instructions_issued = checked_u64_counter_add(
                "branch-pipeline issued-instruction diagnostic",
                projected.control_pipeline.branch_instructions_issued,
                1,
            )?;
            projected.control_pipeline.branch_resolution_cycles = checked_u64_counter_add(
                "branch-pipeline resolution-cycle diagnostic",
                projected.control_pipeline.branch_resolution_cycles,
                next_issue_interval_cycles,
            )?;
        }

        if let PipelineClass::Barrier {
            warp_initiation_interval_cycles,
            ..
        } = slot.pipeline_class
        {
            projected.barrier_pipeline.instructions_issued = checked_u64_counter_add(
                "CTA-barrier pipeline issued-instruction diagnostic",
                projected.barrier_pipeline.instructions_issued,
                1,
            )?;
            projected.barrier_pipeline.issue_server_busy_cycles = checked_u64_counter_add(
                "CTA-barrier pipeline busy-cycle diagnostic",
                projected.barrier_pipeline.issue_server_busy_cycles,
                warp_initiation_interval_cycles,
            )?;
        }

        if matches!(slot.pipeline_class, PipelineClass::Tensor { .. }) {
            observe_readiness_constraint(
                projected,
                base,
                self.tensor_pipe_ready[subcore],
                ReadinessCause::PipelineTensor,
                &mut ready_at,
                &mut limiting,
                &mut tied,
            )?;
        }
        if matches!(slot.pipeline_class, PipelineClass::Sfu { .. }) {
            observe_readiness_constraint(
                projected,
                base,
                self.sfu_pipe_ready[subcore],
                ReadinessCause::PipelineSfu,
                &mut ready_at,
                &mut limiting,
                &mut tied,
            )?;
        }
        if matches!(slot.pipeline_class, PipelineClass::ScalarAlu { .. }) {
            let until = self
                .scalar_execution_pipeline
                .as_ref()
                .map_or(0, |pipeline| pipeline.admission_ready_at(subcore));
            observe_readiness_constraint(
                projected,
                base,
                until,
                ReadinessCause::PipelineAlu,
                &mut ready_at,
                &mut limiting,
                &mut tied,
            )?;
        }

        let effective_ready = ready_at.max(base);
        if effective_ready > base {
            projected.limiting_cycles.add_cause(
                limiting.expect("a post-baseline ready frontier has a cause"),
                effective_ready - base,
            )?;
        }
        if tied != 0 {
            projected.tied_limiting_causes = checked_u64_counter_add(
                "tied limiting readiness cause diagnostic",
                projected.tied_limiting_causes,
                tied,
            )?;
        }
        if now > effective_ready {
            let delayed = now - effective_ready;
            projected
                .limiting_cycles
                .add_eligible_not_selected(delayed)?;
            projected
                .constraint_exposure_cycles
                .add_eligible_not_selected(delayed)?;
            projected.delayed_ready_issues = checked_u64_counter_add(
                "delayed ready issue diagnostic",
                projected.delayed_ready_issues,
                1,
            )?;
        }
        Ok(())
    }

    /// Run one fetch/decode opportunity per SMSP, then one SM-arbitrated stream
    /// prefetch. The fetch picker mirrors CGGTY: continue the last-issued warp
    /// while its instruction buffer has room, otherwise choose the youngest
    /// resident warp with capacity.
    fn instruction_front_end_cycle(&mut self) -> Result<(), IssueError> {
        let Some(original_state) = self.instruction_front_end.take() else {
            return Ok(());
        };
        let now = self.t;
        let mut original = Some(original_state);
        let capacity = match original
            .as_ref()
            .expect("front-end state owns the active cycle")
            .instruction_buffer_capacity()
        {
            Ok(capacity) => capacity,
            Err(error) => {
                self.instruction_front_end = original;
                return Err(error);
            }
        };
        let fetches: [Option<usize>; SUB_CORES] = std::array::from_fn(|subcore| {
            let can_fetch = |warp: usize| self.warps[warp].can_fetch_instruction(capacity);
            match self.hold[subcore] {
                Some(warp) if can_fetch(warp) => Some(warp),
                _ => self.subcore_warps[subcore]
                    .iter()
                    .copied()
                    .filter(|&warp| can_fetch(warp))
                    .max_by_key(|&warp| self.warps[warp].arrival),
            }
        });
        let use_shadow = original
            .as_ref()
            .expect("front-end state owns the active cycle")
            .needs_transactional_cycle(&fetches, now);
        let mut front_end = if use_shadow {
            original
                .as_ref()
                .expect("shadow cycle retains original front-end state")
                .clone()
        } else {
            original
                .take()
                .expect("normal cycle takes ownership of front-end state")
        };
        front_end.maintain(now);
        let mut fetched = [None; SUB_CORES];
        for (subcore, fetch) in fetches.into_iter().enumerate() {
            let Some(warp) = fetch else { continue };
            let cursor = self.warps[warp].front_end_fetch_cursor;
            let pc = self.warps[warp].insts[cursor].pc;
            let ready_at = match front_end.fetch_instruction(
                subcore,
                self.schedule.function_unique_id,
                pc,
                now,
            ) {
                Ok(ready_at) => ready_at,
                Err(error) => {
                    self.instruction_front_end = if use_shadow {
                        original
                    } else {
                        Some(front_end)
                    };
                    return Err(error);
                }
            };
            fetched[subcore] = Some((warp, ready_at));
        }
        if let Err(error) = front_end.prefetch_cycle(now) {
            self.instruction_front_end = if use_shadow {
                original
            } else {
                Some(front_end)
            };
            return Err(error);
        }
        // Validate every warp-side commit before applying the first one. The
        // checks mirror `push_fetched_instruction`; after they pass, direct
        // field updates cannot fail and preserve the cycle transaction.
        for &(warp, _) in fetched.iter().flatten() {
            let state = &self.warps[warp];
            if !state.can_fetch_instruction(capacity)
                || state.front_end_fetch_cursor.checked_add(1).is_none()
            {
                let error = IssueError::InstructionBufferMissing {
                    warp,
                    instruction: state.front_end_fetch_cursor,
                };
                self.instruction_front_end = if use_shadow {
                    original
                } else {
                    Some(front_end)
                };
                return Err(error);
            }
        }
        for &(warp, ready_at) in fetched.iter().flatten() {
            let state = &mut self.warps[warp];
            let instruction = state.front_end_fetch_cursor;
            state.instruction_buffer.push_back(BufferedInstruction {
                instruction,
                ready_at,
            });
            state.front_end_fetch_cursor = instruction
                .checked_add(1)
                .expect("front-end fetch cursor was preflighted");
        }
        self.instruction_front_end = Some(front_end);
        Ok(())
    }

    fn has_instruction_front_end_work(&self) -> Result<bool, IssueError> {
        let Some(front_end) = self.instruction_front_end.as_ref() else {
            return Ok(false);
        };
        let capacity = front_end.instruction_buffer_capacity()?;
        Ok(front_end.has_prefetch_work()
            || self
                .subcore_warps
                .iter()
                .flatten()
                .copied()
                .any(|warp| self.warps[warp].can_fetch_instruction(capacity)))
    }

    fn project_cta_barrier_diagnostics(
        &self,
        picks: &[Option<usize>; SUB_CORES],
        barrier_arrival_times: &[Option<u64>; SUB_CORES],
        now: u64,
        projected: &mut ReadinessDiagnostics,
    ) -> Result<(), IssueError> {
        #[derive(Clone, Copy)]
        struct BlockDelta {
            block: usize,
            retirements: usize,
            arrivals: usize,
            first_new_arrival: Option<u64>,
            last_new_arrival: Option<u64>,
        }
        let mut deltas: [Option<BlockDelta>; SUB_CORES] = [None; SUB_CORES];
        for (subcore, pick) in picks.iter().enumerate() {
            let Some(warp_index) = *pick else { continue };
            let warp = &self.warps[warp_index];
            let slot = warp
                .next_slot
                .ok_or(IssueError::MissingIssueSlot { warp: warp_index })?;
            let reaches_end = warp.next_issue_retires()?;
            // A final barrier is still a CTA participant. Its retirement is
            // deferred until the generation releases, so it must be counted
            // as an arrival rather than an immediate retirement.
            let arrives = slot.op_class == OpClass::BlockBarrier;
            let retires = reaches_end && !arrives;
            if !retires && !arrives {
                continue;
            }
            let entry_index = deltas
                .iter()
                .position(|entry| entry.is_some_and(|delta| delta.block == warp.block))
                .or_else(|| deltas.iter().position(Option::is_none))
                .ok_or(IssueError::CounterOverflow {
                    counter: "CTA barrier diagnostic block delta",
                })?;
            let entry = deltas[entry_index].get_or_insert(BlockDelta {
                block: warp.block,
                retirements: 0,
                arrivals: 0,
                first_new_arrival: None,
                last_new_arrival: None,
            });
            if retires {
                entry.retirements = checked_usize_counter_add(
                    "CTA barrier diagnostic retirement",
                    entry.retirements,
                    1,
                )?;
            }
            if arrives {
                let arrival = barrier_arrival_times[subcore].unwrap_or(now);
                entry.arrivals =
                    checked_usize_counter_add("CTA barrier diagnostic arrival", entry.arrivals, 1)?;
                entry.first_new_arrival = Some(
                    entry
                        .first_new_arrival
                        .map_or(arrival, |current| current.min(arrival)),
                );
                entry.last_new_arrival = Some(
                    entry
                        .last_new_arrival
                        .map_or(arrival, |current| current.max(arrival)),
                );
            }
        }

        for delta in deltas.into_iter().flatten() {
            let state = &self.block_states[delta.block];
            let unfinished = state.unfinished.checked_sub(delta.retirements).ok_or(
                IssueError::CounterUnderflow {
                    counter: "CTA barrier diagnostic unfinished warp",
                },
            )?;
            let waiting = checked_usize_counter_add(
                "CTA barrier diagnostic waiting warp",
                state.waiting,
                delta.arrivals,
            )?;
            if waiting == 0 || waiting != unfinished {
                continue;
            }

            state
                .barrier_generation
                .checked_add(1)
                .ok_or(IssueError::CounterOverflow {
                    counter: "CTA barrier generation",
                })?;
            let first_issue = state.barrier_first_dispatch.unwrap_or(now);
            let mut last_issue = if delta.arrivals == 0 {
                first_issue
            } else {
                now
            };
            let first_arrival = match (state.barrier_first_arrival, delta.first_new_arrival) {
                (Some(existing), Some(new)) => existing.min(new),
                (Some(existing), None) => existing,
                (None, Some(new)) => new,
                (None, None) => now,
            };
            let mut last_arrival = delta.last_new_arrival.unwrap_or(first_arrival);
            let first = state.first_warp;
            let end = checked_usize_counter_add(
                "CTA barrier diagnostic warp range",
                first,
                state.admitted_warps,
            )?;
            for warp in &self.warps[first..end] {
                if !warp.at_block_barrier {
                    continue;
                }
                let dispatch = warp
                    .block_barrier_dispatch
                    .ok_or(IssueError::CounterUnderflow {
                        counter: "CTA barrier dispatch timestamp",
                    })?;
                let arrival = warp
                    .block_barrier_arrival
                    .ok_or(IssueError::CounterUnderflow {
                        counter: "CTA barrier arrival timestamp",
                    })?;
                last_issue = last_issue.max(dispatch);
                last_arrival = last_arrival.max(arrival);
            }
            let release_at = checked_cycle_add("CTA barrier diagnostic release", last_arrival, 1)?;
            let mut rendezvous = 0u64;
            for warp in &self.warps[first..end] {
                if !warp.at_block_barrier {
                    continue;
                }
                let arrival = warp
                    .block_barrier_arrival
                    .ok_or(IssueError::CounterUnderflow {
                        counter: "CTA barrier arrival timestamp",
                    })?;
                rendezvous = checked_u64_counter_add(
                    "CTA barrier rendezvous warp cycle diagnostic",
                    rendezvous,
                    release_at
                        .checked_sub(arrival)
                        .ok_or(IssueError::CounterUnderflow {
                            counter: "CTA barrier rendezvous interval",
                        })?,
                )?;
            }
            for (subcore, pick) in picks.iter().enumerate() {
                let Some(warp) = *pick else { continue };
                if self.warps[warp].block != delta.block
                    || self.warps[warp].next_issue_retires()?
                    || self.warps[warp]
                        .next_slot
                        .is_none_or(|slot| slot.op_class != OpClass::BlockBarrier)
                {
                    continue;
                }
                let arrival = barrier_arrival_times[subcore].unwrap_or(now);
                rendezvous = checked_u64_counter_add(
                    "CTA barrier rendezvous warp cycle diagnostic",
                    rendezvous,
                    release_at
                        .checked_sub(arrival)
                        .ok_or(IssueError::CounterUnderflow {
                            counter: "CTA barrier new-arrival interval",
                        })?,
                )?;
            }
            let spread =
                last_arrival
                    .checked_sub(first_arrival)
                    .ok_or(IssueError::CounterUnderflow {
                        counter: "CTA barrier arrival spread",
                    })?;
            let issue_span =
                last_issue
                    .checked_sub(first_issue)
                    .ok_or(IssueError::CounterUnderflow {
                        counter: "CTA barrier issue span",
                    })?;

            projected.cta_barriers.generations = checked_u64_counter_add(
                "CTA barrier generation diagnostic",
                projected.cta_barriers.generations,
                1,
            )?;
            projected.cta_barriers.total_arrival_spread_cycles = checked_u64_counter_add(
                "CTA barrier total arrival-spread diagnostic",
                projected.cta_barriers.total_arrival_spread_cycles,
                spread,
            )?;
            projected.cta_barriers.max_arrival_spread_cycles =
                projected.cta_barriers.max_arrival_spread_cycles.max(spread);
            projected.cta_barriers.total_rendezvous_warp_cycles = checked_u64_counter_add(
                "CTA barrier total rendezvous diagnostic",
                projected.cta_barriers.total_rendezvous_warp_cycles,
                rendezvous,
            )?;
            if delta.retirements != 0 {
                projected.cta_barriers.releases_after_participant_retirement =
                    checked_u64_counter_add(
                        "CTA barrier participant-retirement diagnostic",
                        projected.cta_barriers.releases_after_participant_retirement,
                        1,
                    )?;
            }
            projected.barrier_pipeline.generations_completed = checked_u64_counter_add(
                "CTA-barrier pipeline completed-generation diagnostic",
                projected.barrier_pipeline.generations_completed,
                1,
            )?;
            if self
                .barrier_issue
                .is_some_and(|issue| issue.generation_rearm_cycles != 0)
            {
                projected.barrier_pipeline.generation_rearm_events = checked_u64_counter_add(
                    "CTA-barrier pipeline generation-rearm diagnostic",
                    projected.barrier_pipeline.generation_rearm_events,
                    1,
                )?;
            }
            projected.barrier_pipeline.first_issue_timestamp_sum = checked_u64_counter_add(
                "CTA-barrier pipeline first-issue timestamp diagnostic",
                projected.barrier_pipeline.first_issue_timestamp_sum,
                first_issue,
            )?;
            projected.barrier_pipeline.last_issue_timestamp_sum = checked_u64_counter_add(
                "CTA-barrier pipeline last-issue timestamp diagnostic",
                projected.barrier_pipeline.last_issue_timestamp_sum,
                last_issue,
            )?;
            projected
                .barrier_pipeline
                .total_generation_issue_span_cycles = checked_u64_counter_add(
                "CTA-barrier pipeline generation issue-span diagnostic",
                projected
                    .barrier_pipeline
                    .total_generation_issue_span_cycles,
                issue_span,
            )?;
            projected.barrier_pipeline.max_generation_issue_span_cycles = projected
                .barrier_pipeline
                .max_generation_issue_span_cycles
                .max(issue_span);
            projected.barrier_pipeline.first_arrival_timestamp_sum = checked_u64_counter_add(
                "CTA-barrier pipeline first-arrival timestamp diagnostic",
                projected.barrier_pipeline.first_arrival_timestamp_sum,
                first_arrival,
            )?;
            projected.barrier_pipeline.last_arrival_timestamp_sum = checked_u64_counter_add(
                "CTA-barrier pipeline last-arrival timestamp diagnostic",
                projected.barrier_pipeline.last_arrival_timestamp_sum,
                last_arrival,
            )?;
            projected
                .barrier_pipeline
                .total_generation_arrival_span_cycles = checked_u64_counter_add(
                "CTA-barrier pipeline generation arrival-span diagnostic",
                projected
                    .barrier_pipeline
                    .total_generation_arrival_span_cycles,
                spread,
            )?;
            projected
                .barrier_pipeline
                .max_generation_arrival_span_cycles = projected
                .barrier_pipeline
                .max_generation_arrival_span_cycles
                .max(spread);
        }
        Ok(())
    }

    fn project_async_issue_overlap(
        &self,
        picks: &[Option<usize>; SUB_CORES],
        projected: &mut ReadinessDiagnostics,
    ) -> Result<(), IssueError> {
        if !self.has_async_copy_protocol {
            return Ok(());
        }

        let mut effective_warps = [None; SUB_CORES];
        let mut copies = 0_u64;
        let mut cache_all = 0_u64;
        let mut bypass = 0_u64;
        for (subcore, pick) in picks.iter().enumerate() {
            let Some(warp) = *pick else { continue };
            let kind = self.warps[warp].insts[self.warps[warp].next].kind;
            if async_copy_source_count(kind).is_none() {
                continue;
            }
            effective_warps[subcore] = Some(warp);
            copies += 1;
            match kind {
                AccessKind::GlobalAsyncCopy {
                    l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                    ..
                } => cache_all += 1,
                AccessKind::GlobalAsyncCopy {
                    l1_policy: LdgstsL1Policy::BypassL1,
                    ..
                } => bypass += 1,
                _ => unreachable!("effective async-copy predicate admits only GlobalAsyncCopy"),
            }
        }
        if copies == 0 {
            return Ok(());
        }

        let mut observation = AsyncIssueOverlapDiagnostics::default();
        if copies >= 2 {
            observation.coissue_cycles = 1;
            observation.coissued_copies = copies;
            observation.cache_all_coissued_copies = cache_all;
            observation.bypass_coissued_copies = bypass;
            observation.mixed_policy_coissue_cycles = u64::from(cache_all != 0 && bypass != 0);
            observation.single_acceptor_excess_copies = copies - 1;
            observation.maximum_coissue_width = copies;
        }

        debug_assert_eq!(self.warps.len(), self.async_copies.len());
        let mut committed_total = 0_u64;
        for (warp, slot) in self.async_copies.iter().enumerate() {
            if self.warps[warp].done {
                continue;
            }
            let groups = slot
                .state
                .as_deref()
                .map_or(0, |state| state.committed_groups.len());
            let groups = u64::try_from(groups).map_err(|_| IssueError::CounterOverflow {
                counter: "async issue overlap committed-group population",
            })?;
            committed_total = checked_u64_counter_add(
                "async issue overlap committed-group population",
                committed_total,
                groups,
            )?;
        }
        for warp in effective_warps.into_iter().flatten() {
            let own_groups = self.async_copies[warp]
                .state
                .as_deref()
                .map_or(0, |state| state.committed_groups.len());
            let own_groups =
                u64::try_from(own_groups).map_err(|_| IssueError::CounterOverflow {
                    counter: "async issue overlap own committed-group population",
                })?;
            let other_groups =
                committed_total
                    .checked_sub(own_groups)
                    .ok_or(IssueError::CounterUnderflow {
                        counter: "async issue overlap other-warp committed-group population",
                    })?;
            if other_groups == 0 {
                continue;
            }
            observation.copies_with_other_warp_committed_groups = checked_readiness_add(
                "readiness.async_issue_overlap.copies_with_other_warp_committed_groups",
                observation.copies_with_other_warp_committed_groups,
                1,
            )?;
            observation.other_warp_committed_groups_sum = checked_readiness_add(
                "readiness.async_issue_overlap.other_warp_committed_groups_sum",
                observation.other_warp_committed_groups_sum,
                other_groups,
            )?;
            observation.maximum_other_warp_committed_groups = observation
                .maximum_other_warp_committed_groups
                .max(other_groups);
        }

        if observation.is_empty() {
            return Ok(());
        }
        if let Some(current) = projected.async_issue_overlap.as_deref_mut() {
            current.add_checked(&observation)?;
        } else {
            projected.async_issue_overlap = Some(Box::new(observation));
        }
        Ok(())
    }

    fn project_picked_readiness(
        &self,
        picks: &[Option<usize>; SUB_CORES],
        barrier_arrival_times: &[Option<u64>; SUB_CORES],
        now: u64,
    ) -> Result<ReadinessDiagnostics, IssueError> {
        let mut projected = (*self.readiness).clone();
        for (subcore, pick) in picks.iter().enumerate() {
            if let Some(warp) = *pick {
                self.project_selected_readiness(subcore, warp, now, &mut projected)?;
                let state = &self.warps[warp];
                let slot = state
                    .next_slot
                    .ok_or(IssueError::MissingIssueSlot { warp })?;
                if is_clocklo_issue(slot) {
                    projected.clock_issue.reads_issued = checked_u64_counter_add(
                        "CLOCKLO reads-issued diagnostic",
                        projected.clock_issue.reads_issued,
                        1,
                    )?;
                    state
                        .clock_reads_issued
                        .checked_add(1)
                        .ok_or(IssueError::CounterOverflow {
                            counter: "warp CLOCKLO issue",
                        })?;
                    if let Some(first) = state.clock_first_issue {
                        let interval =
                            now.checked_sub(first).ok_or(IssueError::CounterUnderflow {
                                counter: "warp CLOCKLO issue interval",
                            })?;
                        if state.clock_reads_issued == 1 {
                            projected.clock_issue.warps_with_interval = checked_u64_counter_add(
                                "CLOCKLO warps-with-interval diagnostic",
                                projected.clock_issue.warps_with_interval,
                                1,
                            )?;
                        }
                        projected.clock_issue.maximum_first_to_last_issue_cycles = projected
                            .clock_issue
                            .maximum_first_to_last_issue_cycles
                            .max(interval);
                    }
                    if self
                        .memory_admission_feedback
                        .as_ref()
                        .is_some_and(|feedback| {
                            feedback.candidate.membar_waits_for_service_completion
                        })
                    {
                        match state.clock_reads_issued {
                            1 => {
                                let first = state.clock_first_issue.ok_or(
                                    IssueError::CounterUnderflow {
                                        counter: "first CLOCKLO phase timestamp",
                                    },
                                )?;
                                let interval =
                                    now.checked_sub(first).ok_or(IssueError::CounterUnderflow {
                                        counter: "first-to-second CLOCKLO issue interval",
                                    })?;
                                projected.clock_issue.maximum_first_to_second_issue_cycles =
                                    projected
                                        .clock_issue
                                        .maximum_first_to_second_issue_cycles
                                        .max(interval);
                            }
                            2 => {
                                let second = self
                                    .memory_admission_feedback
                                    .as_ref()
                                    .and_then(|feedback| feedback.second_clock_issue(warp))
                                    .ok_or(IssueError::CounterUnderflow {
                                        counter: "second CLOCKLO phase timestamp",
                                    })?;
                                let interval = now.checked_sub(second).ok_or(
                                    IssueError::CounterUnderflow {
                                        counter: "second-to-third CLOCKLO issue interval",
                                    },
                                )?;
                                projected.clock_issue.warps_with_three_reads =
                                    checked_u64_counter_add(
                                        "CLOCKLO warps-with-three-reads diagnostic",
                                        projected.clock_issue.warps_with_three_reads,
                                        1,
                                    )?;
                                projected.clock_issue.maximum_second_to_third_issue_cycles =
                                    projected
                                        .clock_issue
                                        .maximum_second_to_third_issue_cycles
                                        .max(interval);
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        self.project_async_issue_overlap(picks, &mut projected)?;
        self.project_cta_barrier_diagnostics(picks, barrier_arrival_times, now, &mut projected)?;
        Ok(projected)
    }

    fn project_memory_service_candidates(
        &self,
        picks: &[Option<usize>; SUB_CORES],
        now: u64,
    ) -> Result<MemoryServiceCandidateState, IssueError> {
        if self.memory_service_candidates.is_empty() {
            return Ok(self.memory_service_candidates.clone());
        }
        let mut projected = self.memory_service_candidates.clone();
        for (subcore, pick) in picks.iter().enumerate() {
            let Some(warp_index) = *pick else { continue };
            let warp = &self.warps[warp_index];
            if let Some((class, _, ideal_wavefronts, unique_lines, _)) =
                projected_global_memory_issue(warp.insts[warp.next].kind, warp.sectors)
            {
                projected.observe(subcore, class, ideal_wavefronts, unique_lines, now)?;
            }
        }
        Ok(projected)
    }

    /// Earliest issue time after both warp/control dependencies and the next
    /// instruction's characterized execution resource are available. A missing
    /// slot remains selectable so the established typed invariant error is
    /// preserved at the transaction boundary.
    fn issue_ready_at<const CHECK_ASYNC: bool>(&self, subcore: usize, warp: usize) -> Option<u64> {
        let mut warp_ready = self.warps[warp].ready_at()?;
        if self.params.resource_source() == IssueResourceSource::Sm120Alpha
            && let Some(slot) = self.warps[warp].next_slot {
            let mask = self.warps[warp].insts[self.warps[warp].next].effective_lane_mask();
            warp_ready = warp_ready.max(issue_fixed_register_ready(
                slot.fixed_latency_operands(),
                mask,
                &self.warps[warp].register_ready,
            ));
        }
        if self.warps[warp]
            .next_slot
            .is_some_and(IssueSlot::is_cctl_invalidate_all)
            && let MemModel::L1Shared(state) = &self.mem
            && (state.l1.has_pending_fills() || !state.retry.is_empty())
        {
            // Model-local serialization for whole-cache invalidation. Every
            // accepted fill (including an unobserved/no-scoreboard load) and
            // deferred lookup must finish through its normal response path.
            // Returning None blocks only this warp; siblings remain eligible
            // and the existing event loop wakes us on external responses.
            // This is not a hardware CCTL latency or completion-upper claim.
            return None;
        }
        if self.instruction_front_end.is_some() {
            warp_ready = warp_ready.max(self.warps[warp].instruction_front_end_ready_at()?);
        }
        if let Some(memory_pipeline) = self.memory_issue_pipeline.as_ref() {
            let state = &self.warps[warp];
            let pc = u64::from(state.insts[state.next].pc);
            if self.memory_pipeline_slot(pc).is_some()
                && !matches!(
                    state.insts[state.next].kind,
                    AccessKind::PredicatedOffMemory
                )
            {
                warp_ready = warp_ready.max(memory_pipeline.admission_ready_at(subcore));
            }
        }
        if let Some(feedback) = self.memory_admission_feedback.as_ref() {
            let state = &self.warps[warp];
            let admission_ready =
                feedback.admission_ready_at(subcore, state.insts[state.next].kind)?;
            warp_ready = warp_ready.max(admission_ready);
            if let Some(slot) = state.next_slot {
                let completion_ready = feedback.completion_ready_at(warp, slot)?;
                warp_ready = warp_ready.max(completion_ready);
            }
        }
        if CHECK_ASYNC
            && let Some(IssueSlot {
                op_class: OpClass::AsyncCopyWait,
                async_copy_wait_max_pending_groups,
                ..
            }) = self.warps[warp].next_slot
        {
            let async_ready = self.async_copies[warp]
                .async_wait_clear_time(*async_copy_wait_max_pending_groups)?;
            warp_ready = warp_ready.max(async_ready);
        }
        let pipeline_ready = match self.warps[warp].next_slot.map(|slot| slot.pipeline_class) {
            Some(PipelineClass::Branch { .. }) => 0,
            Some(PipelineClass::Barrier { .. }) => {
                self.block_states[self.warps[warp].block].barrier_rearm_ready
            }
            Some(PipelineClass::Tensor { .. }) => self.tensor_pipe_ready[subcore],
            Some(PipelineClass::Sfu { .. }) => self.sfu_pipe_ready[subcore],
            Some(PipelineClass::ScalarAlu { .. }) => self
                .scalar_execution_pipeline
                .as_ref()
                .map_or(0, |pipeline| pipeline.admission_ready_at(subcore)),
            Some(PipelineClass::DispatchOnly) | None => 0,
        };
        Some(warp_ready.max(pipeline_ready))
    }

    fn is_issue_ready<const CHECK_ASYNC: bool>(
        &self,
        subcore: usize,
        warp: usize,
        now: u64,
    ) -> bool {
        self.issue_ready_at::<CHECK_ASYNC>(subcore, warp)
            .is_some_and(|ready| ready <= now)
    }

    fn pick_warps<const CHECK_ASYNC: bool>(&self, now: u64) -> [Option<usize>; SUB_CORES] {
        std::array::from_fn(|subcore| {
            // CGGTY pick: held warp while its control state and execution
            // resource are ready, else youngest fully-ready warp.
            match self.hold[subcore] {
                Some(w) if self.is_issue_ready::<CHECK_ASYNC>(subcore, w, now) => Some(w),
                _ => self.subcore_warps[subcore]
                    .iter()
                    .copied()
                    .filter(|&w| self.is_issue_ready::<CHECK_ASYNC>(subcore, w, now))
                    .max_by_key(|&w| self.warps[w].arrival),
            }
        })
    }

    /// Pick in stable SMSP order while reserving the transaction's shared
    /// admission capacity. A candidate rejected by an earlier SMSP's claim is
    /// not a dispatch bubble: the same SMSP may still issue its held/youngest
    /// ready non-candidate (or another candidate whose queue has room).
    fn pick_warps_with_memory_admission<const CHECK_ASYNC: bool>(
        &self,
        now: u64,
    ) -> [Option<usize>; SUB_CORES] {
        let Some(feedback) = self.memory_admission_feedback.as_ref() else {
            return self.pick_warps::<CHECK_ASYNC>(now);
        };
        let mut additions = [0usize; SUB_CORES];
        let mut picks = [None; SUB_CORES];
        for (subcore, pick) in picks.iter_mut().enumerate() {
            let permitted = |warp: usize| {
                if !self.is_issue_ready::<CHECK_ASYNC>(subcore, warp, now) {
                    return false;
                }
                let state = &self.warps[warp];
                feedback.permits_transaction_addition(
                    subcore,
                    state.insts[state.next].kind,
                    &additions,
                    now,
                )
            };
            let selected = match self.hold[subcore] {
                Some(warp) if permitted(warp) => Some(warp),
                _ => self.subcore_warps[subcore]
                    .iter()
                    .copied()
                    .filter(|&warp| permitted(warp))
                    .max_by_key(|&warp| self.warps[warp].arrival),
            };
            if let Some(warp) = selected {
                let state = &self.warps[warp];
                feedback.record_transaction_addition(
                    subcore,
                    state.insts[state.next].kind,
                    &mut additions,
                );
            }
            *pick = selected;
        }
        picks
    }

    /// Assign completion timestamps to barrier arrivals dispatched in this
    /// cycle. SMSP dispatch remains parallel; the one SM-shared service queues
    /// those arrivals in stable subcore order at its measured initiation rate.
    fn project_barrier_arrival_times(
        &self,
        picks: &[Option<usize>; SUB_CORES],
        now: u64,
    ) -> Result<[Option<u64>; SUB_CORES], IssueError> {
        let mut server_ready = self.barrier_pipe_ready;
        let mut arrivals = [None; SUB_CORES];
        for (subcore, pick) in picks.iter().enumerate() {
            let Some(warp) = *pick else { continue };
            let slot = self.warps[warp]
                .next_slot
                .ok_or(IssueError::MissingIssueSlot { warp })?;
            let PipelineClass::Barrier {
                warp_initiation_interval_cycles,
                ..
            } = slot.pipeline_class
            else {
                continue;
            };
            let arrival = checked_cycle_add(
                "CTA-barrier arrival service",
                server_ready.max(now),
                warp_initiation_interval_cycles,
            )?;
            arrivals[subcore] = Some(arrival);
            server_ready = arrival;
        }
        Ok(arrivals)
    }

    fn next_known_issue_time<const CHECK_ASYNC: bool>(&self) -> Option<u64> {
        let mut next_t: Option<u64> = None;
        for (subcore, warps) in self.subcore_warps.iter().enumerate() {
            for &warp in warps {
                if let Some(ready) = self.issue_ready_at::<CHECK_ASYNC>(subcore, warp) {
                    next_t = Some(next_t.map_or(ready, |current| current.min(ready)));
                }
            }
        }
        next_t
    }

    /// Earliest timestamp at which a committed exact-destination group becomes
    /// eligible to enter the aggregate service. Groups with outstanding source
    /// responses have no locally known wake time and remain event-driven.
    fn next_async_copy_destination_source_time(&self) -> Option<u64> {
        self.async_copies
            .iter()
            .filter_map(|slot| slot.state.as_deref())
            .flat_map(|state| state.committed_groups.iter())
            .filter(|group| {
                group.pending_sources == 0
                    && group.destination_wavefronts != 0
                    && !group.destination_service_queued
                    && group.source_completion_time > self.t
            })
            .map(|group| group.source_completion_time)
            .min()
    }

    /// Assign committed, source-complete cache-all groups to the shared
    /// destination service. Projection and timestamp arithmetic finish before
    /// either the pipe or a group is mutated.
    fn enqueue_ready_async_copy_destinations(&mut self) -> Result<(), IssueError> {
        if !self.has_async_copy_protocol {
            return Ok(());
        }
        let mut candidates = Vec::new();
        for (warp, slot) in self.async_copies.iter().enumerate() {
            let Some(state) = slot.state.as_deref() else {
                continue;
            };
            for group in &state.committed_groups {
                if group.pending_sources == 0
                    && group.destination_wavefronts != 0
                    && !group.destination_service_queued
                    && group.source_completion_time <= self.t
                {
                    candidates.push((
                        group.source_completion_time,
                        warp,
                        group.id,
                        group.destination_wavefronts,
                    ));
                }
            }
        }
        candidates
            .sort_unstable_by_key(|&(source_ready, warp, group, _)| (source_ready, warp, group.0));

        let mut projected_pipe = self.ldgsts_destination_pipe_ready;
        let mut assignments = Vec::with_capacity(candidates.len());
        for (source_ready, warp, group, wavefronts) in candidates {
            let service_cycles = self.params.ldgsts_destination_service_cycles(wavefronts)?;
            let start = projected_pipe.max(source_ready);
            let ready = checked_cycle_add(
                "LDGSTS aggregate destination-wavefront service",
                start,
                service_cycles,
            )?;
            projected_pipe = ready;
            assignments.push((warp, group, ready));
        }

        for (warp, group_id, ready) in assignments {
            let group = self.async_copies[warp]
                .async_group_mut(group_id)
                .expect("destination-service candidate remains committed");
            debug_assert!(!group.destination_service_queued);
            group.destination_service_queued = true;
            group.destination_completion_time = ready;
            group.completion_time = group.completion_time.max(ready);
        }
        self.ldgsts_destination_pipe_ready = projected_pipe;
        Ok(())
    }

    /// Commit the state transitions that become due while the actor parks at
    /// an external event horizon. Event mode may jump over idle cycles, but
    /// release queues and fixed-clock completions must be retired at the
    /// parked timestamp before typed state is observed.
    fn park_at_horizon(&mut self, bound: u64) {
        if let Some(cache) = self.immediate_constant_cache.as_mut() {
            for state in cache.iter_mut() {
                state.complete_ready(bound);
            }
        }
        if let Some(memory_pipeline) = self.memory_issue_pipeline.as_mut() {
            memory_pipeline.prune(bound);
        }
        if let Some(scalar_pipeline) = self.scalar_execution_pipeline.as_mut() {
            scalar_pipeline.prune(bound);
        }
        self.t = bound;
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
    pub(crate) fn advance(
        &mut self,
        horizon: ReplayHorizon,
        out: &mut Vec<MemRequest>,
    ) -> Result<Advance, IssueError> {
        #[cfg(feature = "reference")]
        {
            self.advance_diagnostics.calls = self.advance_diagnostics.calls.checked_add(1).ok_or(
                IssueError::CounterOverflow {
                    counter: "reference advance calls",
                },
            )?;
        }
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
            if let Some(cache) = self.immediate_constant_cache.as_mut() {
                for state in cache.iter_mut() {
                    state.complete_ready(self.t);
                }
            }
            self.enqueue_ready_async_copy_destinations()?;
            if horizon.reached(self.t) {
                if let Some(bound) = horizon.bounded_time() {
                    // A caller may revisit an already parked horizon after a
                    // same-cycle external event. Retire fixed-clock releases
                    // at that timestamp before exposing the boundary; the
                    // ordinary skip path already performs this via
                    // `park_at_horizon`.
                    self.park_at_horizon(bound);
                }
                return Ok(Advance::Idle {
                    wake_at: Some(self.t),
                });
            }
            if let Some(memory_pipeline) = self.memory_issue_pipeline.as_mut() {
                memory_pipeline.prune(self.t);
            }
            if let Some(scalar_pipeline) = self.scalar_execution_pipeline.as_mut() {
                scalar_pipeline.prune(self.t);
            }
            #[cfg(feature = "reference")]
            {
                self.advance_diagnostics.local_cycle_steps = self
                    .advance_diagnostics
                    .local_cycle_steps
                    .checked_add(1)
                    .ok_or(IssueError::CounterOverflow {
                        counter: "reference local cycle steps",
                    })?;
            }
            self.instruction_front_end_cycle()?;
            // Consume the carded per-SMSP issue width in one logical cycle.
            // A dual-issue card permits a second independently ready warp
            // transaction. Do not erase control stalls or dependency latency
            // to synthesize same-warp pairs: no pairing model is implemented.
            // Each slot has its own preflight/commit boundary; this is not
            // rollback of an already committed first slot on second-slot error.
            let issue_width = self
                .params
                .issue_ports
                .max(1)
                .max(u32::from(self.params.dual_issue_supported).saturating_add(1));
            let visibility_events_before = self.pending_l1_visibility_event_count();
            let mut issued_this_cycle = false;
            for _ in 0..issue_width {
                let before = out.len();
                let issued = self.issue_cycle(out)?;
                issued_this_cycle |= issued;
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
                // An IMC miss consumes one scheduler opportunity but emits no
                // instruction pick. Do not replay that same miss across an
                // arbitrarily wide carded issue loop.
                let instruction_pick = self.last_issue_picks.iter().any(Option::is_some);
                if !issued || !instruction_pick {
                    break;
                }
            }
            if issued_this_cycle {
                self.t = checked_cycle_add("issue-clock advance", self.t, 1)?;
                // Visibility observations are consumed by the shared event
                // system, not by the issuing SM itself.  Stop after this
                // issue cycle so the observations at `t` are scheduled and
                // processed before this actor can issue at `t + 1`.
                if self.pending_l1_visibility_event_count() > visibility_events_before {
                    return Ok(Advance::Idle {
                        wake_at: Some(self.t),
                    });
                }
                if self
                    .memory_admission_feedback
                    .as_mut()
                    .is_some_and(|state| state.take_yield_requested())
                {
                    return Ok(Advance::Idle {
                        wake_at: Some(self.t),
                    });
                }
                continue;
            }
            // Nothing ready: skip to the next KNOWN warp/resource timer
            // (resident warps only — retired warps left the sub-core lists;
            // warps waiting on in-flight responses have no computable time and
            // are excluded).
            let issue_or_async_time = if self.has_async_copy_protocol {
                match (
                    self.next_known_issue_time::<true>(),
                    self.next_async_copy_destination_source_time(),
                ) {
                    (Some(issue), Some(source)) => Some(issue.min(source)),
                    (Some(issue), None) => Some(issue),
                    (None, Some(source)) => Some(source),
                    (None, None) => None,
                }
            } else {
                self.next_known_issue_time::<false>()
            };
            let front_end_work = self.has_instruction_front_end_work()?;
            let front_end_time = front_end_work
                .then(|| checked_cycle_add("instruction front-end clock advance", self.t, 1))
                .transpose()?;
            let maintenance_time = self
                .instruction_front_end
                .as_ref()
                .and_then(InstructionFrontEndState::next_maintenance_time);
            let front_end_time = match (front_end_time, maintenance_time) {
                (Some(work), Some(maintenance)) => Some(work.min(maintenance)),
                (work, None) => work,
                (None, maintenance) => maintenance,
            };
            let next_t = match (issue_or_async_time, front_end_time) {
                (Some(issue), Some(front_end)) => Some(issue.min(front_end)),
                (Some(issue), None) => Some(issue),
                (None, Some(front_end)) => Some(front_end),
                (None, None) => None,
            };
            match next_t {
                Some(nt) => {
                    let next_cycle = checked_cycle_add("idle-clock advance", self.t, 1)?;
                    let target = nt.max(next_cycle);
                    // Preserve the next stream-buffer release even if warp
                    // eligibility is later; releasing a line enables prefetch
                    // while the core is otherwise stalled.
                    let can_skip = !self.dense_reference;
                    if let Some(bound) = horizon.bounded_time()
                        && target >= bound
                    {
                        // The dense reference still visits locally idle
                        // cycles strictly before an external horizon. It
                        // must stop at the horizon itself so the queued event
                        // is applied before any work at that timestamp.
                        if self.dense_reference && next_cycle < bound {
                            self.t = next_cycle;
                            continue;
                        }
                        #[cfg(feature = "reference")]
                        if can_skip {
                            let skipped = bound - next_cycle;
                            self.advance_diagnostics.idle_jumps = self
                                .advance_diagnostics
                                .idle_jumps
                                .checked_add(u64::from(skipped != 0))
                                .ok_or(IssueError::CounterOverflow {
                                    counter: "reference idle jumps",
                                })?;
                            self.advance_diagnostics.skipped_cycles = self
                                .advance_diagnostics
                                .skipped_cycles
                                .checked_add(skipped)
                                .ok_or(IssueError::CounterOverflow {
                                    counter: "reference skipped cycles",
                                })?;
                        }
                        // Park AT the horizon instead of jumping past it:
                        // the event there may ready a warp before the
                        // next known timer.
                        self.park_at_horizon(bound);
                        return Ok(Advance::Idle {
                            wake_at: Some(bound),
                        });
                    }
                    #[cfg(feature = "reference")]
                    if can_skip {
                        let skipped = target - next_cycle;
                        self.advance_diagnostics.idle_jumps = self
                            .advance_diagnostics
                            .idle_jumps
                            .checked_add(u64::from(skipped != 0))
                            .ok_or(IssueError::CounterOverflow {
                                counter: "reference idle jumps",
                            })?;
                        self.advance_diagnostics.skipped_cycles = self
                            .advance_diagnostics
                            .skipped_cycles
                            .checked_add(skipped)
                            .ok_or(IssueError::CounterOverflow {
                                counter: "reference skipped cycles",
                            })?;
                    }
                    self.t = if can_skip { target } else { next_cycle };
                }
                None if self.outstanding_responses() > 0
                    || self
                        .memory_admission_feedback
                        .as_ref()
                        .is_some_and(|state| {
                            state.has_unpriced() || state.has_pending_write_acks()
                        }) =>
                {
                    if let Some(bound) = horizon.bounded_time() {
                        let next_cycle = checked_cycle_add("idle-clock advance", self.t, 1)?;
                        if self.dense_reference && next_cycle < bound {
                            self.t = next_cycle;
                            continue;
                        }
                        #[cfg(feature = "reference")]
                        if !self.dense_reference {
                            let skipped = bound - next_cycle;
                            self.advance_diagnostics.idle_jumps = self
                                .advance_diagnostics
                                .idle_jumps
                                .checked_add(u64::from(skipped != 0))
                                .ok_or(IssueError::CounterOverflow {
                                    counter: "reference idle jumps",
                                })?;
                            self.advance_diagnostics.skipped_cycles = self
                                .advance_diagnostics
                                .skipped_cycles
                                .checked_add(skipped)
                                .ok_or(IssueError::CounterOverflow {
                                    counter: "reference skipped cycles",
                                })?;
                        }
                        self.park_at_horizon(bound);
                    }
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
        constant_deep: bool,
        memory_time: Option<u64>,
        write_barrier: Option<DepBarrier>,
        read_barrier: Option<DepBarrier>,
        stall: u8,
        yield_flag: bool,
        pipeline_class: PipelineClass,
        retires_warp: bool,
        observe_async_copy: bool,
    ) -> Result<(), IssueError> {
        let t = self.t;
        checked_cycle_add("issue-clock advance", t, 1)?;
        match pipeline_class {
            PipelineClass::Branch {
                next_issue_interval_cycles,
            } => {
                checked_cycle_add("branch-pipeline resolution", t, next_issue_interval_cycles)?;
            }
            PipelineClass::Barrier {
                warp_initiation_interval_cycles,
                ..
            } => {
                checked_cycle_add(
                    "CTA-barrier pipeline availability",
                    t,
                    warp_initiation_interval_cycles,
                )?;
            }
            PipelineClass::Tensor {
                initiation_interval_cycles,
            } => {
                checked_cycle_add(
                    "tensor-pipeline availability",
                    t,
                    initiation_interval_cycles,
                )?;
            }
            PipelineClass::Sfu {
                initiation_interval_cycles,
            } => {
                checked_cycle_add(
                    "uniform/SFU pipeline availability",
                    t,
                    initiation_interval_cycles,
                )?;
            }
            PipelineClass::ScalarAlu {
                latency_cycles,
                initiation_interval_cycles,
                ..
            } => {
                checked_cycle_add("scalar execution pipeline latency", t, latency_cycles)?;
                checked_cycle_add(
                    "scalar execution pipeline initiation",
                    t,
                    initiation_interval_cycles,
                )?;
            }
            PipelineClass::DispatchOnly => {}
        }
        if !retires_warp {
            checked_cycle_add("warp stall eligibility", t, u64::from(stall.max(1)))?;
            if yield_flag {
                checked_cycle_add("warp yield eligibility", t, 2)?;
            }
        }
        if !matches!(kind, AccessKind::PredicatedOffMemory) && read_barrier.is_some() {
            checked_cycle_add(
                "read-dependence barrier visibility",
                t,
                self.params.read_barrier_latency,
            )?;
        }

        let needs_ready = write_barrier.is_some() || observe_async_copy;
        match kind {
            AccessKind::NonMemory if needs_ready => {
                self.params.non_memory_ready(t)?;
            }
            AccessKind::Shared { wavefronts } if needs_ready => {
                self.params.shared_ready(t, wavefronts)?;
            }
            AccessKind::NonMemory | AccessKind::PredicatedOffMemory | AccessKind::Shared { .. } => {
            }
            AccessKind::GlobalEligible { .. }
            | AccessKind::LocalLoad { .. }
            | AccessKind::GlobalAsyncCopy {
                l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                ..
            } => match &self.mem {
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
                    let hit_ready = checked_cycle_add(
                        "shared L1 hit-pipeline floor",
                        memory_time.unwrap_or(t),
                        self.params.l1_hit_latency,
                    )?;
                    checked_constant_return_time(
                        hit_ready,
                        constant_deep && write_barrier.is_some(),
                        &self.params,
                        "constant L1-hit return latency",
                    )?;
                }
                MemModel::Flat | MemModel::L1Const(_) | MemModel::L1Shared(_) => {}
            },
            AccessKind::GlobalBypass { len, .. }
            | AccessKind::GlobalAsyncCopy {
                len,
                l1_policy: LdgstsL1Policy::BypassL1,
                ..
            }
            | AccessKind::GlobalUntagged { len, .. } => match &self.mem {
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
            },
            AccessKind::GlobalStore { .. }
            | AccessKind::GlobalReduction { .. }
            | AccessKind::LocalStore { .. } => match &self.mem {
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

    fn preflight_issue_memory(
        &self,
        picks: &[Option<usize>; SUB_CORES],
        async_destination_wavefronts: &[u64; SUB_CORES],
        memory_times: &[u64; SUB_CORES],
    ) -> Result<(), IssueError> {
        let (l1, shared) = match &self.mem {
            MemModel::Flat => return Ok(()),
            MemModel::L1Const(l1) => (l1, None),
            MemModel::L1Shared(st) => (&st.l1, Some(st)),
        };
        let mut budgets = MemoryOrigin::ALL.map(L1MutationBudget::for_origin);
        let mut eligible_sectors = 0u64;
        let mut pending_budget_fits = true;
        let mut constant_async_deliveries = [None; SUB_CORES];
        for (subcore, pick) in picks.iter().enumerate() {
            let Some(wix) = *pick else { continue };
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
            let write_barrier = if self.has_async_copy_protocol {
                ordinary_write_barrier(slot)
            } else {
                slot.control.write_barrier
            };
            let add_budget = |counter: &'static str, value: &mut u64, add: u64| {
                *value = checked_u64_counter_add(counter, *value, add)?;
                Ok::<(), IssueError>(())
            };
            match inst.kind {
                AccessKind::GlobalEligible { off, len, .. }
                | AccessKind::LocalLoad { off, len }
                | AccessKind::GlobalAsyncCopy {
                    off,
                    len,
                    l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                    ..
                } => {
                    let origin = match inst.kind {
                        AccessKind::GlobalEligible { .. } | AccessKind::GlobalAsyncCopy { .. } => {
                            MemoryOrigin::Global
                        }
                        AccessKind::LocalLoad { .. } => MemoryOrigin::Local,
                        _ => unreachable!("match arm admits only hierarchy loads"),
                    };
                    let budget = &mut budgets[origin.index()];
                    let sectors = u64::from(len);
                    add_budget(
                        "L1 eligible-sector budget",
                        &mut budget.eligible_sectors,
                        sectors,
                    )?;
                    add_budget(
                        "combined L1 eligible-sector budget",
                        &mut eligible_sectors,
                        sectors,
                    )?;
                    add_budget("L1 recency budget", &mut budget.recency_touches, sectors)?;
                    add_budget("L1 fill budget", &mut budget.new_fills, sectors)?;
                    if shared.is_none()
                        && let Some(completion_work) = async_copy_completion_work(inst.kind)
                    {
                        debug_assert!(matches!(
                            inst.kind,
                            AccessKind::GlobalAsyncCopy {
                                l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                                ..
                            }
                        ));
                        constant_async_deliveries[subcore] =
                            Some((wix, completion_work, async_destination_wavefronts[subcore]));
                    }
                    if shared.is_some()
                        && warp
                            .preflight_pending_add(wix, write_barrier, u32::from(len))
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
                            if let Some(&reply) = st.outstanding_response_ids.get(&response_id) {
                                return Err(IssueError::ResponseTokenAlreadyOutstanding { reply });
                            }
                        }
                    }
                }
                AccessKind::GlobalBypass {
                    len, token, off, ..
                } => {
                    let budget = &mut budgets[MemoryOrigin::Global.index()];
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
                        && write_barrier.is_some()
                    {
                        if warp
                            .preflight_pending_add(wix, write_barrier, u32::from(len))
                            .is_err()
                        {
                            pending_budget_fits = false;
                        }
                        for sector_index in off..off + usize::from(len) {
                            let response_id =
                                ResponseId::new(st.response_owner, wix, warp.next, sector_index);
                            if let Some(&reply) = st.outstanding_response_ids.get(&response_id) {
                                return Err(IssueError::ResponseTokenAlreadyOutstanding { reply });
                            }
                        }
                    }
                }
                AccessKind::GlobalAsyncCopy {
                    off,
                    len,
                    completion_work,
                    l1_policy: LdgstsL1Policy::BypassL1,
                    ..
                } => {
                    let budget = &mut budgets[MemoryOrigin::Global.index()];
                    let sectors = u64::from(len);
                    add_budget(
                        "L1 bypass-sector budget",
                        &mut budget.bypass_sectors,
                        sectors,
                    )?;
                    if let Some(st) = shared
                        && (write_barrier.is_some() || completion_work != 0)
                    {
                        if warp
                            .preflight_pending_add(wix, write_barrier, u32::from(len))
                            .is_err()
                        {
                            pending_budget_fits = false;
                        }
                        for sector_index in off..off + usize::from(len) {
                            let response_id =
                                ResponseId::new(st.response_owner, wix, warp.next, sector_index);
                            if let Some(&reply) = st.outstanding_response_ids.get(&response_id) {
                                return Err(IssueError::ResponseTokenAlreadyOutstanding { reply });
                            }
                        }
                    }
                }
                AccessKind::GlobalUntagged { off, len } => {
                    let budget = &mut budgets[MemoryOrigin::Global.index()];
                    let sectors = u64::from(len);
                    add_budget(
                        "L1 untagged-sector budget",
                        &mut budget.untagged_sectors,
                        sectors,
                    )?;
                    if let Some(st) = shared
                        && write_barrier.is_some()
                    {
                        if warp
                            .preflight_pending_add(wix, write_barrier, u32::from(len))
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
                AccessKind::GlobalStore { len, .. }
                | AccessKind::GlobalReduction { len, .. }
                | AccessKind::LocalStore { len, .. } => {
                    let origin = match inst.kind {
                        AccessKind::GlobalStore { .. } | AccessKind::GlobalReduction { .. } => {
                            MemoryOrigin::Global
                        }
                        AccessKind::LocalStore { .. } => MemoryOrigin::Local,
                        _ => unreachable!("match arm admits only hierarchy writes"),
                    };
                    let budget = &mut budgets[origin.index()];
                    let sectors = u64::from(len);
                    if matches!(inst.kind, AccessKind::GlobalReduction { .. }) {
                        add_budget(
                            "L1 reduction-sector budget",
                            &mut budget.reduction_sectors,
                            sectors,
                        )?;
                    } else {
                        add_budget("L1 store-sector budget", &mut budget.store_sectors, sectors)?;
                    }
                    add_budget("L1 recency budget", &mut budget.recency_touches, sectors)?;
                }
                AccessKind::NonMemory
                | AccessKind::PredicatedOffMemory
                | AccessKind::Shared { .. } => {}
            }
        }

        let constant_ready_upper = if shared.is_none() {
            l1.constant_batch_ready_upper_bound(self.t, eligible_sectors)
        } else {
            None
        };
        let constant_timing_fits = shared.is_some() || constant_ready_upper.is_some();
        let constant_async_completion_fits = shared.is_some()
            || constant_async_deliveries.iter().flatten().all(
                |&(wix, delivery_work, destination_wavefronts)| {
                    constant_ready_upper.is_some_and(|source_ready| {
                        self.async_copies[wix]
                            .preflight_async_copy_add(
                                wix,
                                0,
                                source_ready,
                                delivery_work,
                                destination_wavefronts,
                                None,
                                &self.params,
                            )
                            .is_ok()
                    })
                },
            );
        let shared_timing_fits = shared.is_none()
            || eligible_sectors == 0
            || (self.t.checked_add(self.params.l1_hit_latency).is_some()
                && self.t.checked_add(self.params.l1_miss_latency).is_some());
        if l1.preflight_mutation_budgets(&budgets).is_ok()
            && pending_budget_fits
            && constant_timing_fits
            && constant_async_completion_fits
            && shared_timing_fits
        {
            return Ok(());
        }

        // Near a numeric boundary, replay this at-most-four-instruction batch
        // on a shadow cache. This avoids rejecting valid warm hits while keeping
        // the ordinary hot path allocation-free.
        let mut shadow = l1.clone();
        let mut same_timestamp_line_folds = HashSet::new();
        for (subcore, pick) in picks.iter().enumerate() {
            let Some(wix) = *pick else { continue };
            let warp = &self.warps[wix];
            let memory_time = memory_times[subcore];
            let inst = warp.insts[warp.next];
            let slot = warp
                .next_slot
                .ok_or(IssueError::MissingIssueSlot { warp: wix })?;
            let write_barrier = if self.has_async_copy_protocol {
                ordinary_write_barrier(slot)
            } else {
                slot.control.write_barrier
            };
            match inst.kind {
                AccessKind::GlobalEligible { off, len, .. }
                | AccessKind::LocalLoad { off, len }
                | AccessKind::GlobalAsyncCopy {
                    off,
                    len,
                    l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                    ..
                } => {
                    if matches!(inst.kind, AccessKind::GlobalAsyncCopy { .. }) {
                        shadow.begin_async_copy_lookup_batch();
                    } else {
                        shadow.begin_lookup_batch();
                    }
                    let origin = match inst.kind {
                        AccessKind::GlobalEligible { .. } | AccessKind::GlobalAsyncCopy { .. } => {
                            MemoryOrigin::Global
                        }
                        AccessKind::LocalLoad { .. } => MemoryOrigin::Local,
                        _ => unreachable!("match arm admits only hierarchy loads"),
                    };
                    let mut pending_add = 0u32;
                    let async_delivery = async_copy_completion_work(inst.kind);
                    let observe_ready = write_barrier.is_some() || async_delivery.is_some();
                    let mut async_source_ready = if shared.is_none() && async_delivery.is_some() {
                        Some(checked_cycle_add(
                            "LDGSTS L1 hit-pipeline floor",
                            memory_time,
                            self.params.l1_hit_latency,
                        )?)
                    } else {
                        None
                    };
                    let mshr_floor = matches!(
                        inst.kind,
                        AccessKind::GlobalAsyncCopy {
                            l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                            ..
                        }
                    )
                    .then_some(self.params.ldgsts_l1_mshr_floor);
                    for &sector_id in &warp.sectors[off..off + usize::from(len)] {
                        if shared.is_some() {
                            match access_shared_l1(
                                &mut shadow,
                                origin,
                                sector_id,
                                memory_time,
                                observe_ready,
                                mshr_floor,
                            )? {
                                SharedL1Access::Hit { .. } => {}
                                SharedL1Access::MissRequest { fill } => {
                                    checked_cycle_add(
                                        "self-response horizon",
                                        memory_time,
                                        self.params.l1_miss_latency,
                                    )?;
                                    if let FillKey::Line { line, .. } = fill {
                                        same_timestamp_line_folds.insert((line, memory_time));
                                    }
                                    pending_add = checked_u32_counter_add(
                                        "issue exact pending-add budget",
                                        pending_add,
                                        1,
                                    )?;
                                }
                                SharedL1Access::MergedPending { fill } => {
                                    if let FillKey::Line { line, .. } = fill
                                        && !same_timestamp_line_folds.contains(&(line, memory_time))
                                    {
                                        shadow.reserve_pending_line_sector(
                                            fill,
                                            (sector_id & 3) as u8,
                                        )?;
                                    }
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
                        } else if observe_ready {
                            let (_, ready) = match mshr_floor {
                                Some(floor) => shadow.access_eligible_with_mshr_floor(
                                    origin,
                                    sector_id,
                                    memory_time,
                                    floor,
                                )?,
                                None => shadow.access_eligible(origin, sector_id, memory_time)?,
                            };
                            if let Some(current) = &mut async_source_ready {
                                *current = (*current).max(ready);
                            }
                        } else {
                            match mshr_floor {
                                Some(floor) => {
                                    shadow.access_eligible_unobserved_with_mshr_floor(
                                        origin,
                                        sector_id,
                                        memory_time,
                                        floor,
                                    )?;
                                }
                                None => {
                                    shadow.access_eligible_unobserved(
                                        origin,
                                        sector_id,
                                        memory_time,
                                    )?;
                                }
                            }
                        }
                    }
                    if let (Some(source_ready), Some(delivery_work)) =
                        (async_source_ready, async_delivery)
                    {
                        self.async_copies[wix].preflight_async_copy_add(
                            wix,
                            0,
                            source_ready,
                            delivery_work,
                            async_destination_wavefronts[subcore],
                            None,
                            &self.params,
                        )?;
                    }
                    if shared.is_some() {
                        warp.preflight_pending_add(wix, write_barrier, pending_add)?;
                    }
                }
                AccessKind::GlobalBypass { len, token, .. } => {
                    shadow.count_bypass(
                        MemoryOrigin::Global,
                        u64::from(len),
                        token == CacheOpToken::Atom,
                    )?;
                    if shared.is_some() {
                        warp.preflight_pending_add(wix, write_barrier, u32::from(len))?;
                    }
                }
                AccessKind::GlobalAsyncCopy {
                    len,
                    l1_policy: LdgstsL1Policy::BypassL1,
                    ..
                } => {
                    shadow.count_bypass(MemoryOrigin::Global, u64::from(len), false)?;
                    if shared.is_some() {
                        warp.preflight_pending_add(wix, write_barrier, u32::from(len))?;
                    }
                }
                AccessKind::GlobalUntagged { len, .. } => {
                    shadow.count_untagged(MemoryOrigin::Global, u64::from(len))?;
                    if shared.is_some() {
                        warp.preflight_pending_add(wix, write_barrier, u32::from(len))?;
                    }
                }
                AccessKind::GlobalStore { off, len }
                | AccessKind::GlobalReduction { off, len, .. }
                | AccessKind::LocalStore { off, len } => {
                    let origin = match inst.kind {
                        AccessKind::GlobalStore { .. } | AccessKind::GlobalReduction { .. } => {
                            MemoryOrigin::Global
                        }
                        AccessKind::LocalStore { .. } => MemoryOrigin::Local,
                        _ => unreachable!("match arm admits only hierarchy writes"),
                    };
                    let reduction = matches!(inst.kind, AccessKind::GlobalReduction { .. });
                    let store_sectors = &warp.sectors[off..off + usize::from(len)];
                    if self.shared_write_lookup_runtime.observer.is_some()
                        && matches!(inst.kind, AccessKind::GlobalStore { .. })
                    {
                        let hits = shadow.observe_write_lookup_batch(store_sectors)?;
                        shadow.count_write_lookup_hits(origin, hits)?;
                    } else if let Some(hits) =
                        projected_l1_write_lookup_hits(inst.kind, warp.sectors)
                    {
                        shadow.count_write_lookup_hits(origin, hits)?;
                    }
                    for &sector_id in store_sectors {
                        if reduction {
                            shadow.probe_reduction(origin, sector_id, self.t)?;
                        } else {
                            shadow.probe_store(origin, sector_id, self.t)?;
                        }
                    }
                }
                AccessKind::NonMemory
                | AccessKind::PredicatedOffMemory
                | AccessKind::Shared { .. } => {}
            }
        }
        Ok(())
    }

    /// Check the per-SMSP cacheable-LDGSTS hit-return service timeline before
    /// the issue transaction touches the shared L1 or allocates an async-copy
    /// group.  The ordinary budget preflight proves L1 counters and MSHR
    /// capacity, but the service frontier is a separate counter that is only
    /// advanced for observed hits.  Near `u64::MAX`, replay the selected
    /// memory batch on a private L1 shadow so misses and pending merges do not
    /// cause a false rejection (or allow an overflow after an earlier hit has
    /// already mutated the real cache).
    fn preflight_async_l1_hit_service(
        &self,
        picks: &[Option<usize>; SUB_CORES],
        memory_times: &[u64; SUB_CORES],
    ) -> Result<(), IssueError> {
        let Some(service) = self.params.ldgsts_l1_hit_return_service else {
            return Ok(());
        };
        let interval = service.interval_cycles;
        if interval == 0 {
            return Ok(());
        }

        // Stay allocation-free on the normal path.  The exact shadow is only
        // needed when charging every possible observed instruction could overflow
        // the current service frontier.
        let mut needs_shadow = false;
        for (subcore, pick) in picks.iter().enumerate() {
            let Some(wix) = *pick else { continue };
            let warp = &self.warps[wix];
            let slot = warp
                .next_slot
                .ok_or(IssueError::MissingIssueSlot { warp: wix })?;
            let write_barrier = if self.has_async_copy_protocol {
                ordinary_write_barrier(slot)
            } else {
                slot.control.write_barrier
            };
            let AccessKind::GlobalAsyncCopy {
                len,
                completion_work,
                l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                ..
            } = warp.insts[warp.next].kind
            else {
                continue;
            };
            if write_barrier.is_none() && completion_work == 0 {
                continue;
            }
            if len == 0 {
                continue;
            }
            let Some(hit_ready) = memory_times[subcore].checked_add(self.params.l1_hit_latency)
            else {
                // The ordinary timing preflight normally catches this first;
                // keep this shadow decision conservative if that contract is
                // widened later.
                needs_shadow = true;
                continue;
            };
            let frontier = self.shared_write_lookup_runtime.async_l1_hit_service_ready[subcore];
            if frontier.max(hit_ready).checked_add(interval).is_none() {
                needs_shadow = true;
            }
        }
        if !needs_shadow {
            return Ok(());
        }

        let MemModel::L1Shared(st) = &self.mem else {
            return Ok(());
        };
        let mut shadow = st.l1.clone();
        let mut service_ready = self.shared_write_lookup_runtime.async_l1_hit_service_ready;

        for (subcore, pick) in picks.iter().enumerate() {
            let Some(wix) = *pick else { continue };
            let warp = &self.warps[wix];
            let slot = warp
                .next_slot
                .ok_or(IssueError::MissingIssueSlot { warp: wix })?;
            let kind = warp.insts[warp.next].kind;
            let write_barrier = if self.has_async_copy_protocol {
                ordinary_write_barrier(slot)
            } else {
                slot.control.write_barrier
            };
            let now = memory_times[subcore];
            match kind {
                AccessKind::GlobalEligible { off, len, .. }
                | AccessKind::LocalLoad { off, len }
                | AccessKind::GlobalAsyncCopy {
                    off,
                    len,
                    l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                    ..
                } => {
                    let (origin, mshr_floor, cacheable_async, source_active) = match kind {
                        AccessKind::GlobalEligible { .. } => {
                            (MemoryOrigin::Global, None, false, false)
                        }
                        AccessKind::LocalLoad { .. } => (MemoryOrigin::Local, None, false, false),
                        AccessKind::GlobalAsyncCopy {
                            completion_work, ..
                        } => (
                            MemoryOrigin::Global,
                            Some(self.params.ldgsts_l1_mshr_floor),
                            true,
                            completion_work != 0,
                        ),
                        _ => unreachable!("match arm admits only cacheable L1 accesses"),
                    };
                    let observe_ready = write_barrier.is_some() || source_active;
                    if cacheable_async {
                        shadow.begin_async_copy_lookup_batch();
                    } else {
                        shadow.begin_lookup_batch();
                    }
                    let end =
                        off.checked_add(usize::from(len))
                            .ok_or(IssueError::CounterOverflow {
                                counter: "issue sector cursor",
                            })?;
                    let sectors =
                        warp.sectors
                            .get(off..end)
                            .ok_or(IssueError::CounterOverflow {
                                counter: "issue sector cursor",
                            })?;
                    let mut instruction_hit_service_charged = false;
                    for &sector in sectors {
                        let access = access_shared_l1(
                            &mut shadow,
                            origin,
                            sector,
                            now,
                            observe_ready,
                            mshr_floor,
                        )?;
                        if cacheable_async
                            && observe_ready
                            && let SharedL1Access::Hit { ready: Some(ready) } = access
                        {
                            // One source-return service token covers the
                            // whole coalesced LDGSTS instruction.
                            if !instruction_hit_service_charged {
                                if service_ready[subcore] < ready {
                                    service_ready[subcore] = ready;
                                }
                                service_ready[subcore] = service_ready[subcore]
                                    .checked_add(interval)
                                    .ok_or(IssueError::CounterOverflow {
                                        counter: "async L1 hit service timeline",
                                    })?;
                                instruction_hit_service_charged = true;
                            }
                        }
                        if matches!(access, SharedL1Access::Stalled) {
                            break;
                        }
                    }
                }
                AccessKind::GlobalStore { off, len }
                | AccessKind::GlobalReduction { off, len, .. }
                | AccessKind::LocalStore { off, len } => {
                    let end =
                        off.checked_add(usize::from(len))
                            .ok_or(IssueError::CounterOverflow {
                                counter: "issue sector cursor",
                            })?;
                    let sectors =
                        warp.sectors
                            .get(off..end)
                            .ok_or(IssueError::CounterOverflow {
                                counter: "issue sector cursor",
                            })?;
                    let origin = if matches!(kind, AccessKind::LocalStore { .. }) {
                        MemoryOrigin::Local
                    } else {
                        MemoryOrigin::Global
                    };
                    let reduction = matches!(kind, AccessKind::GlobalReduction { .. });
                    for &sector in sectors {
                        if reduction {
                            shadow.probe_reduction(origin, sector, now)?;
                        } else {
                            shadow.probe_store(origin, sector, now)?;
                        }
                    }
                }
                AccessKind::GlobalBypass { .. }
                | AccessKind::GlobalUntagged { .. }
                | AccessKind::GlobalAsyncCopy {
                    l1_policy: LdgstsL1Policy::BypassL1,
                    ..
                }
                | AccessKind::NonMemory
                | AccessKind::PredicatedOffMemory
                | AccessKind::Shared { .. } => {}
            }
        }
        Ok(())
    }

    fn preflight_issue_accounting(
        &self,
        picks: &[Option<usize>; SUB_CORES],
        barrier_arrival_times: &[Option<u64>; SUB_CORES],
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

        // (block, ordinary retirements, barrier arrivals, final-barrier
        // arrivals). Final barrier arrivals retire only after the generation
        // releases, so they must not be included in `ordinary retirements`.
        let mut block_deltas: [Option<(usize, usize, usize, usize)>; SUB_CORES] = [None; SUB_CORES];
        let mut released_blocks: [Option<usize>; SUB_CORES] = [None; SUB_CORES];
        let mut total_retirements = 0usize;
        for &wix in picks.iter().flatten() {
            let warp = &self.warps[wix];
            let slot = warp
                .next_slot
                .ok_or(IssueError::MissingIssueSlot { warp: wix })?;
            let reaches_end = warp.next_issue_retires()?;
            let arrives = slot.op_class == OpClass::BlockBarrier;
            let retires = reaches_end && !arrives;
            let entry_index = block_deltas
                .iter()
                .position(|entry| entry.is_some_and(|(block, _, _, _)| block == warp.block))
                .or_else(|| block_deltas.iter().position(Option::is_none))
                .ok_or(IssueError::CounterOverflow {
                    counter: "per-cycle block accounting",
                })?;
            let entry = block_deltas[entry_index].get_or_insert((warp.block, 0, 0, 0));
            if retires {
                entry.1 = checked_usize_counter_add("block retirement", entry.1, 1)?;
                total_retirements =
                    checked_usize_counter_add("unfinished warp retirement", total_retirements, 1)?;
            } else if arrives {
                entry.2 = checked_usize_counter_add("block barrier waiter", entry.2, 1)?;
                if reaches_end {
                    entry.3 = checked_usize_counter_add("final block barrier arrival", entry.3, 1)?;
                }
            }
        }

        self.unfinished_warps.checked_sub(total_retirements).ok_or(
            IssueError::CounterUnderflow {
                counter: "unfinished warp",
            },
        )?;
        let mut residency_released = 0u32;
        let mut shared_memory_released = 0u64;
        for (block, retirements, arrivals, final_arrivals) in block_deltas.into_iter().flatten() {
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
            let mut post_release_unfinished = unfinished;
            if waiting > 0 && waiting == unfinished {
                let existing_final_arrivals = self.warps
                    [state.first_warp..state.first_warp + state.admitted_warps]
                    .iter()
                    .filter(|warp| warp.at_block_barrier && warp.next == warp.insts.len())
                    .count();
                let final_retirements = checked_usize_counter_add(
                    "final block barrier retirement",
                    existing_final_arrivals,
                    final_arrivals,
                )?;
                post_release_unfinished = unfinished.checked_sub(final_retirements).ok_or(
                    IssueError::CounterUnderflow {
                        counter: "final block barrier unfinished warp",
                    },
                )?;
                total_retirements = checked_usize_counter_add(
                    "unfinished warp retirement",
                    total_retirements,
                    final_retirements,
                )?;
                let mut last_arrival = self.t;
                for warp in &self.warps[state.first_warp..state.first_warp + state.admitted_warps] {
                    if let Some(arrival) = warp.block_barrier_arrival {
                        last_arrival = last_arrival.max(arrival);
                    }
                }
                for (subcore, pick) in picks.iter().enumerate() {
                    if pick.is_some_and(|warp| self.warps[warp].block == block) {
                        last_arrival =
                            last_arrival.max(barrier_arrival_times[subcore].unwrap_or(self.t));
                    }
                }
                checked_cycle_add("block-barrier release", last_arrival, 1)?;
                if let Some(issue) = self.barrier_issue {
                    checked_cycle_add(
                        "CTA-barrier generation re-arm",
                        last_arrival,
                        issue.generation_rearm_cycles,
                    )?;
                }
            }
            // A block can become empty as part of the barrier release when its
            // final instructions are BAR.SYNC. `post_release_unfinished`
            // accounts for that deferred retirement before releasing residency.
            if post_release_unfinished == 0 {
                for &warp_slot in &state.warp_slots {
                    if !self.resident_warp_slots.contains(&warp_slot) {
                        return Err(IssueError::ResidentWarpSlotMissing {
                            launch_index: self.warps[state.first_warp].launch_index,
                            warp_slot,
                        });
                    }
                }
                let release_index = released_blocks.iter().position(Option::is_none).ok_or(
                    IssueError::CounterOverflow {
                        counter: "CTA barrier diagnostic released block",
                    },
                )?;
                released_blocks[release_index] = Some(block);
                residency_released = checked_u32_counter_add(
                    "resident warp retirement",
                    residency_released,
                    state.warp_count,
                )?;
                shared_memory_released = checked_u64_counter_add(
                    "resident shared-memory retirement",
                    shared_memory_released,
                    state.shared_memory_bytes,
                )?;
            }
        }
        self.unfinished_warps.checked_sub(total_retirements).ok_or(
            IssueError::CounterUnderflow {
                counter: "unfinished warp",
            },
        )?;
        self.resident_warps.checked_sub(residency_released).ok_or(
            IssueError::CounterUnderflow {
                counter: "resident warp",
            },
        )?;
        self.resident_shared_memory_bytes
            .checked_sub(shared_memory_released)
            .ok_or(IssueError::CounterUnderflow {
                counter: "resident shared-memory byte",
            })?;
        if released_blocks.iter().any(Option::is_some) {
            // `retire_warp` admits the next launch only after applying these
            // releases.  Validate that entire admission chain now, while the
            // issue cycle is still transactional; otherwise a deferred counter
            // overflow (for example, the arrival-stamp increment) could retire
            // an earlier warp before returning its error.
            let mut projected_resident_slots = (*self.resident_warp_slots).clone();
            let mut projected_resident_blocks = self
                .block_states
                .iter()
                .filter(|state| state.unfinished != 0)
                .count() as u32;
            for block in released_blocks.into_iter().flatten() {
                projected_resident_blocks = projected_resident_blocks.checked_sub(1).ok_or(
                    IssueError::CounterUnderflow {
                        counter: "resident block",
                    },
                )?;
                for &warp_slot in &self.block_states[block].warp_slots {
                    let removed = projected_resident_slots.remove(&warp_slot);
                    if !removed {
                        return Err(IssueError::ResidentWarpSlotMissing {
                            launch_index: self.warps[self.block_states[block].first_warp]
                                .launch_index,
                            warp_slot,
                        });
                    }
                }
            }
            self.preflight_block_admissions(
                self.resident_warps.checked_sub(residency_released).ok_or(
                    IssueError::CounterUnderflow {
                        counter: "resident warp",
                    },
                )?,
                self.unfinished_warps.checked_sub(total_retirements).ok_or(
                    IssueError::CounterUnderflow {
                        counter: "unfinished warp",
                    },
                )?,
                self.resident_shared_memory_bytes
                    .checked_sub(shared_memory_released)
                    .ok_or(IssueError::CounterUnderflow {
                        counter: "resident shared-memory byte",
                    })?,
                projected_resident_blocks,
                projected_resident_slots,
                self.arrival_stamp,
            )?;
        }
        Ok(next_issued)
    }

    /// Validate admissions that a successful retirement will trigger without
    /// mutating the actor.  The commit path in [`retire_warp`] repeats these
    /// already-checked operations; keeping this pass side-effect free preserves
    /// the issue-cycle transaction boundary when a checked counter can fail.
    fn preflight_block_admissions(
        &self,
        mut resident_warps: u32,
        mut unfinished_warps: usize,
        mut resident_shared_memory_bytes: u64,
        mut resident_blocks: u32,
        mut resident_warp_slots: HashSet<i32>,
        mut arrival_stamp: u64,
    ) -> Result<(), IssueError> {
        let mut next_block = self.next_block;
        while next_block < self.blocks.len() {
            let block = &self.blocks[next_block];
            let live_warps = self.block_live_warps[next_block];
            let shared_memory_bytes = if live_warps == 0 {
                0
            } else {
                self.shared_memory_residency.bytes_per_cta()
            };
            let next_resident_blocks = resident_blocks
                .checked_add(u32::from(live_warps != 0))
                .ok_or(IssueError::CounterOverflow {
                    counter: "resident blocks",
                })?;
            let Some(next_resident_warps) = resident_warps.checked_add(live_warps) else {
                if resident_warps > 0 {
                    break;
                }
                return Err(IssueError::CounterOverflow {
                    counter: "resident warp",
                });
            };
            let Some(next_resident_shared_memory_bytes) =
                resident_shared_memory_bytes.checked_add(shared_memory_bytes)
            else {
                if resident_warps > 0 {
                    break;
                }
                return Err(IssueError::CounterOverflow {
                    counter: "resident shared-memory byte",
                });
            };
            let pending_slots = block
                .warps
                .iter()
                .filter(|warp| !warp.insts.is_empty())
                .map(|warp| warp.warp_slot)
                .collect::<Vec<_>>();
            let slots_collide = pending_slots
                .iter()
                .any(|slot| resident_warp_slots.contains(slot));
            if resident_warps > 0
                && (next_resident_blocks > self.params.max_resident_blocks_per_sm
                    || next_resident_warps > self.params.max_resident_warps_per_sm
                    || next_resident_shared_memory_bytes
                        > self.shared_memory_residency.max_bytes_per_sm()
                    || slots_collide)
            {
                break;
            }

            // Keep the preflight in lockstep with `admit_blocks`: this
            // checked addition is otherwise the last fallible counter before
            // the first warp of a successor CTA is appended.
            let admitted_warps =
                usize::try_from(live_warps).map_err(|_| IssueError::CounterOverflow {
                    counter: "admitted warp cardinality",
                })?;
            unfinished_warps = unfinished_warps.checked_add(admitted_warps).ok_or(
                IssueError::CounterOverflow {
                    counter: "unfinished warp",
                },
            )?;

            // Construction validates every dynamic PC and physical slot, but
            // repeat the fallible joins here so this preflight remains a true
            // boundary if an in-memory replay is assembled by a future caller.
            for warp in &block.warps {
                let slot =
                    u32::try_from(warp.warp_slot).map_err(|_| IssueError::WarpSlotOutOfRange {
                        launch_index: block.launch_index,
                        warp_slot: warp.warp_slot,
                        arch: self.params.arch,
                        physical_warp_slots: self.physical_warp_slots,
                    })?;
                if slot >= self.physical_warp_slots {
                    return Err(IssueError::WarpSlotOutOfRange {
                        launch_index: block.launch_index,
                        warp_slot: warp.warp_slot,
                        arch: self.params.arch,
                        physical_warp_slots: self.physical_warp_slots,
                    });
                }
                if !warp.insts.is_empty() {
                    lookup_dense(
                        &self.schedule_slots,
                        self.schedule,
                        warp.insts[0].pc,
                        block.launch_index,
                        warp.warp_slot,
                    )?;
                }
            }
            arrival_stamp = checked_u64_counter_add(
                "warp arrival stamp",
                arrival_stamp,
                u64::from(live_warps),
            )?;
            next_block = checked_usize_counter_add("next block", next_block, 1)?;
            resident_blocks = next_resident_blocks;
            resident_warps = next_resident_warps;
            resident_shared_memory_bytes = next_resident_shared_memory_bytes;
            for slot in pending_slots {
                resident_warp_slots.insert(slot);
            }
        }
        Ok(())
    }

    fn preflight_result_writeback_priority(
        &self,
        picks: &[Option<usize>; SUB_CORES],
    ) -> Result<(), IssueError> {
        let Some(state) = self.result_writebacks.as_ref() else {
            return Ok(());
        };
        let mut collisions = 0u64;
        let mut deferral_cycles = 0u64;
        for (subcore, pick) in picks.iter().enumerate() {
            let Some(wix) = *pick else { continue };
            let slot = self.warps[wix]
                .next_slot
                .ok_or(IssueError::MissingIssueSlot { warp: wix })?;
            let Some(ResultWritebackKind::FixedHadd2 { bank }) =
                result_writeback_kind(slot, self.params.arch)
            else {
                continue;
            };
            let fixed_at = checked_cycle_add(
                "HADD2 result writeback",
                self.t,
                state.timing.hadd2_writeback_cycles,
            )?;
            for event in &state.pending {
                if event.subcore == subcore as u8
                    && event.bank == bank
                    && event.writeback_at == fixed_at
                {
                    checked_cycle_add(
                        "deferred L1TEX writeback",
                        event.writeback_at,
                        state.timing.variable_deferral_cycles,
                    )?;
                    checked_cycle_add(
                        "deferred L1TEX visibility",
                        event.visible_at,
                        state.timing.variable_deferral_cycles,
                    )?;
                    collisions = checked_u64_counter_add(
                        "fixed-over-L1TEX collision preflight",
                        collisions,
                        1,
                    )?;
                    deferral_cycles = checked_u64_counter_add(
                        "L1TEX writeback deferral-cycle preflight",
                        deferral_cycles,
                        state.timing.variable_deferral_cycles,
                    )?;
                }
            }
        }
        checked_u64_counter_add(
            "fixed-over-L1TEX collision diagnostic",
            self.readiness.result_writeback.fixed_over_l1tex_collisions,
            collisions,
        )?;
        checked_u64_counter_add(
            "L1TEX writeback deferral-cycle diagnostic",
            self.readiness.result_writeback.l1tex_deferral_cycles,
            deferral_cycles,
        )?;
        Ok(())
    }

    fn register_l1tex_writeback(
        &mut self,
        subcore: usize,
        warp: usize,
        kind: ResultWritebackKind,
        barrier: Option<DepBarrier>,
        visible_at: Option<u64>,
    ) -> Result<(), IssueError> {
        let l1_hit_latency = self.params.l1_hit_latency;
        let (
            Some(state),
            ResultWritebackKind::L1TexStrongSmU32 { bank },
            Some(barrier),
            Some(visible_at),
        ) = (self.result_writebacks.as_mut(), kind, barrier, visible_at)
        else {
            return Ok(());
        };
        let hit_floor = checked_cycle_add("L1 hit-pipeline floor", self.t, l1_hit_latency)?;
        if visible_at != hit_floor
            || l1_hit_latency < state.timing.l1tex_writeback_to_visibility_cycles
        {
            return Ok(());
        }
        let writeback_at = visible_at - state.timing.l1tex_writeback_to_visibility_cycles;
        state.pending.push(PendingVariableWriteback {
            subcore: subcore as u8,
            bank,
            writeback_at,
            visible_at,
            warp,
            barrier,
        });
        Ok(())
    }

    fn apply_fixed_result_priority(
        &mut self,
        subcore: usize,
        kind: ResultWritebackKind,
        projected: &mut ReadinessDiagnostics,
    ) -> Result<(), IssueError> {
        let (Some(state), ResultWritebackKind::FixedHadd2 { bank }) =
            (self.result_writebacks.as_mut(), kind)
        else {
            return Ok(());
        };
        let fixed_at = checked_cycle_add(
            "HADD2 result writeback",
            self.t,
            state.timing.hadd2_writeback_cycles,
        )?;
        let (warps, diagnostics) = (&mut self.warps, &mut projected.result_writeback);
        for event in &mut state.pending {
            if event.subcore != subcore as u8
                || event.bank != bank
                || event.writeback_at != fixed_at
            {
                continue;
            }
            event.writeback_at = checked_cycle_add(
                "deferred L1TEX writeback",
                event.writeback_at,
                state.timing.variable_deferral_cycles,
            )?;
            event.visible_at = checked_cycle_add(
                "deferred L1TEX visibility",
                event.visible_at,
                state.timing.variable_deferral_cycles,
            )?;
            warps[event.warp].stamp_clear(
                Some(event.barrier),
                event.visible_at,
                DependencyOrigin::L1Tex,
            );
            diagnostics.fixed_over_l1tex_collisions = checked_u64_counter_add(
                "fixed-over-L1TEX collision diagnostic",
                diagnostics.fixed_over_l1tex_collisions,
                1,
            )?;
            diagnostics.l1tex_deferral_cycles = checked_u64_counter_add(
                "L1TEX writeback deferral-cycle diagnostic",
                diagnostics.l1tex_deferral_cycles,
                state.timing.variable_deferral_cycles,
            )?;
        }
        Ok(())
    }

    /// Probe the per-SMSP Immediate Constant Cache for the selected
    /// instructions.  A tag miss consumes an issue attempt but does not
    /// dispatch or advance the warp; the warp is parked until the line's
    /// deterministic GCC/L1C return time.  The state is cloned first so a
    /// later cycle-preflight error cannot leave a partially reserved constant
    /// request behind.
    fn preflight_immediate_constant(
        &self,
        picks: [Option<usize>; SUB_CORES],
        now: u64,
    ) -> Result<ImmediateConstantPlan, IssueError> {
        let Some(mut cache) = self.immediate_constant_cache.as_ref().cloned() else {
            return Ok(ImmediateConstantPlan {
                picks,
                cache: None,
                waits: [None; SUB_CORES],
                miss_observed: false,
            });
        };
        let mut effective = picks;
        let mut waits = [None; SUB_CORES];
        let mut miss_observed = false;
        for (subcore, pick) in picks.into_iter().enumerate() {
            let Some(warp) = pick else { continue };
            let state = &self.warps[warp];
            let pc = state.insts[state.next].pc;
            let Some(lines) = self.immediate_constant_lines(pc) else {
                continue;
            };
            if lines.is_empty() {
                continue;
            }
            if state.imc_wait_until > now {
                effective[subcore] = None;
                waits[subcore] = Some((warp, state.imc_wait_until));
                miss_observed = true;
                continue;
            }
            if let Some(ready) = cache[subcore].reserve(lines, now, self.params.l1_miss_latency)?
                && ready > now
            {
                effective[subcore] = None;
                waits[subcore] = Some((warp, ready));
                miss_observed = true;
            }
        }
        Ok(ImmediateConstantPlan {
            picks: effective,
            cache: Some(cache),
            waits,
            miss_observed,
        })
    }

    /// One issue cycle at `self.t` across the four sub-cores. Returns
    /// whether anything issued.
    fn issue_cycle(&mut self, out: &mut Vec<MemRequest>) -> Result<bool, IssueError> {
        let t = self.t;
        self.last_issue_picks = [None; SUB_CORES];
        let mut selected = if self.has_async_copy_protocol {
            self.pick_warps_with_memory_admission::<true>(t)
        } else {
            self.pick_warps_with_memory_admission::<false>(t)
        };
        if matches!(self.mem, MemModel::L1Shared(_))
            && let Some(subcore) = selected.iter().position(|pick| {
                pick.is_some_and(|warp| {
                    self.warps[warp]
                        .next_slot
                        .is_some_and(IssueSlot::is_cctl_invalidate_all)
                })
            })
        {
            // An invalidation is one cache-state boundary. Do not allow a
            // sibling picked in this same transaction to allocate a new fill
            // between the readiness check and invalidate_all().
            let warp = selected[subcore];
            selected = [None; SUB_CORES];
            selected[subcore] = warp;
        }
        let imc_plan = self.preflight_immediate_constant(selected, t)?;
        let picks = imc_plan.picks;
        let effective_picks = picks;
        let mut imc_cache = imc_plan.cache;
        let imc_waits = imc_plan.waits;
        let imc_miss_observed = imc_plan.miss_observed;
        let memory_pipeline_plan =
            if let Some(memory_pipeline) = self.memory_issue_pipeline.as_ref() {
                let mut selected = [None; SUB_CORES];
                for (subcore, pick) in picks.iter().enumerate() {
                    let Some(warp) = *pick else { continue };
                    let state = &self.warps[warp];
                    if matches!(
                        state.insts[state.next].kind,
                        AccessKind::PredicatedOffMemory
                    ) {
                        continue;
                    }
                    selected[subcore] =
                        self.memory_pipeline_slot(u64::from(state.insts[state.next].pc));
                }
                Some(memory_pipeline.plan(selected, t)?)
            } else {
                None
            };
        let scalar_execution_pipeline_plan =
            if let Some(scalar_pipeline) = self.scalar_execution_pipeline.as_ref() {
                let selected = std::array::from_fn(|subcore| {
                    picks[subcore].and_then(|warp| {
                        let slot = self.warps[warp].next_slot?;
                        matches!(slot.pipeline_class, PipelineClass::ScalarAlu { .. })
                            .then_some(slot.pipeline_class)
                    })
                });
                Some(scalar_pipeline.plan(selected, t)?)
            } else {
                None
            };
        let barrier_arrival_times = self.project_barrier_arrival_times(&picks, t)?;
        let mut async_destination_wavefronts = [0; SUB_CORES];
        for (subcore, pick) in picks.iter().enumerate() {
            let Some(wix) = *pick else { continue };
            let kind = self.warps[wix].insts[self.warps[wix].next].kind;
            async_destination_wavefronts[subcore] =
                async_copy_destination_wavefronts(kind, &self.params)?.unwrap_or(0);
        }

        // A cycle is the caller-visible transaction boundary. Validate every
        // selected instruction's possible timing arms before any sub-core
        // mutates counters, cache state, barriers, requests, or retirement.
        for (subcore, pick) in picks.iter().enumerate() {
            let Some(wix) = *pick else { continue };
            let w = &self.warps[wix];
            if self.instruction_front_end.is_some() {
                w.preflight_buffered_issue(wix)?;
            }
            let slot = w
                .next_slot
                .ok_or(IssueError::MissingIssueSlot { warp: wix })?;
            if slot.is_cctl_invalidate_all()
                && let MemModel::L1Shared(state) = &self.mem
            {
                // Dirty-resident mode still requires a real writeback path;
                // preserve its typed error before any issue-state mutation.
                state.l1.preflight_invalidate_all()?;
            }
            let reaches_end = w.next_issue_retires()?;
            // Final BAR.SYNC participates in the CTA rendezvous and retires
            // only once that generation releases.
            let retires_warp = reaches_end && slot.op_class != OpClass::BlockBarrier;
            let kind = w.insts[w.next].kind;
            let constant_deep = is_constant_load_opcode(slot.opcode_text())
                && self.is_constant_path_pc(u64::from(w.insts[w.next].pc));
            validate_async_copy_completion_work(kind, wix)?;
            let write_barrier = if self.has_async_copy_protocol {
                ordinary_write_barrier(slot)
            } else {
                slot.control.write_barrier
            };
            let observe_async_copy =
                self.has_async_copy_protocol && async_copy_source_count(kind).is_some();
            if observe_async_copy && let Some(delivery_work) = async_copy_completion_work(kind) {
                let source_ready = match &self.mem {
                    MemModel::Flat => checked_cycle_add(
                        "flat async-copy source visibility",
                        self.t,
                        self.params.global_flat_latency,
                    )?,
                    MemModel::L1Const(_) => checked_cycle_add(
                        "constant-L1 async-copy source upper bound",
                        self.t,
                        self.params.l1_miss_latency,
                    )?,
                    MemModel::L1Shared(_) => checked_cycle_add(
                        "shared-L1 async-copy known-source floor",
                        self.t,
                        self.params.l1_hit_latency,
                    )?,
                };
                let pending_add = if matches!(self.mem, MemModel::L1Shared(_)) {
                    u32::from(async_copy_source_count(kind).unwrap_or(0))
                } else {
                    0
                };
                self.async_copies[wix].preflight_async_copy_add(
                    wix,
                    pending_add,
                    source_ready,
                    delivery_work,
                    async_destination_wavefronts[subcore],
                    (matches!(self.mem, MemModel::L1Shared(_))
                        && matches!(
                            kind,
                            AccessKind::GlobalAsyncCopy {
                                l1_policy: LdgstsL1Policy::BypassL1,
                                ..
                            }
                        ))
                    .then_some(
                        bypass_async_copy_observed_topology(
                            kind,
                            w.next,
                            w.async_copy_cohorts
                                .and_then(|sidecar| sidecar.view(w.next)),
                        )
                        .unwrap_or(BypassAsyncCopyInput {
                            instruction: w.next,
                            sector_off: match kind {
                                AccessKind::GlobalAsyncCopy { off, .. } => off,
                                _ => 0,
                            },
                            sector_len: async_copy_source_count(kind).unwrap_or(0),
                            source_cohorts: 0,
                            destination_lines: 0,
                            max_bank_wavefronts: 0,
                            max_bypass_wavefronts: 0,
                            cohorts: None,
                            sector_membership_masks: None,
                        }),
                    ),
                    &self.params,
                )?;
            }
            self.preflight_issue_timing(
                kind,
                constant_deep,
                memory_pipeline_plan
                    .as_ref()
                    .and_then(|plan| plan.issues[subcore])
                    .map(|issue| issue.shared_accept_at),
                write_barrier,
                slot.control.read_barrier,
                if self.params.arch == ArchId::Sm120
                    && sm120_alpha_scalar_control_stall_suppressed(slot.opcode_text())
                {
                    0
                } else {
                    slot.control.stall
                },
                slot.control.yield_flag,
                slot.pipeline_class,
                retires_warp,
                observe_async_copy,
            )?;
        }
        let memory_times = std::array::from_fn(|subcore| {
            memory_pipeline_plan
                .as_ref()
                .and_then(|plan| plan.issues[subcore])
                .map_or(t, |issue| issue.shared_accept_at)
        });
        self.preflight_issue_memory(&picks, &async_destination_wavefronts, &memory_times)?;
        self.preflight_async_l1_hit_service(&picks, &memory_times)?;
        self.preflight_result_writeback_priority(&picks)?;
        let next_issued = self.preflight_issue_accounting(&picks, &barrier_arrival_times)?;
        // Diagnostics are preflighted last so an observational counter can
        // never mask an established execution/cache/domain error. No mutation
        // has occurred yet, so a diagnostic overflow remains transactional.
        let mut projected_readiness =
            self.project_picked_readiness(&picks, &barrier_arrival_times, t)?;
        let projected_memory_service_candidates =
            self.project_memory_service_candidates(&picks, t)?;
        if let Some(plan) = scalar_execution_pipeline_plan.as_ref() {
            projected_readiness.scalar_execution_pipeline = Some(Box::new(plan.next.diagnostics));
        }
        let memory_admission_feedback_plan =
            if let Some(feedback) = self.memory_admission_feedback.as_ref() {
                let mut issues = [None; SUB_CORES];
                for (subcore, pick) in picks.iter().enumerate() {
                    let Some(warp) = *pick else { continue };
                    let state = &self.warps[warp];
                    let slot = state
                        .next_slot
                        .ok_or(IssueError::MissingIssueSlot { warp })?;
                    issues[subcore] = Some(MemoryAdmissionFeedbackIssue {
                        subcore,
                        warp,
                        instruction: state.next,
                        kind: state.insts[state.next].kind,
                        prior_clock_reads: state.clock_reads_issued,
                        clock_read: is_clocklo_issue(slot),
                    });
                }
                Some(feedback.plan_issue(&issues, t)?)
            } else {
                None
            };

        if let Some(state) = self.result_writebacks.as_mut() {
            state.pending.retain(|event| event.writeback_at >= t);
        }

        // An IMC miss consumes the SMSP issue opportunity even though no
        // traced instruction is dispatched.  Keep it separate from the exact
        // executed-instruction counter below.
        let mut issued_this_cycle = imc_miss_observed;
        for (sc, pick) in picks.into_iter().enumerate() {
            let Some(wix) = pick else { continue };
            issued_this_cycle = true;

            // Issue exactly one instruction from warp `wix`.
            let (
                blk_ix,
                launch_index,
                warp_slot,
                slot_class,
                pipeline_class,
                scalar_control_stall_suppressed,
                control,
                instruction,
                inst,
                warp_sectors,
                retires_warp,
                result_writeback,
                clock_read,
                constant_load,
                constant_deep,
                cctl_invalidate_all,
                fence_semantics,
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
                    slot.pipeline_class,
                    self.params.arch == ArchId::Sm120
                        && sm120_alpha_scalar_control_stall_suppressed(slot.opcode_text()),
                    slot.control,
                    w.next,
                    w.insts[w.next],
                    w.sectors,
                    {
                        let reaches_end = w.next_issue_retires()?;
                        reaches_end && slot.op_class != OpClass::BlockBarrier
                    },
                    result_writeback_kind(slot, self.params.arch),
                    is_clocklo_issue(slot),
                    is_constant_load_opcode(slot.opcode_text()),
                    self.is_constant_path_pc(u64::from(w.insts[w.next].pc)),
                    slot.is_cctl_invalidate_all(),
                    slot.visibility_fence_semantics(),
                )
            };
            let produces_dynamic_effects = !matches!(inst.kind, AccessKind::PredicatedOffMemory);
            let write_barrier = if !produces_dynamic_effects
                || (self.has_async_copy_protocol && slot_class == OpClass::AsyncCopyCommit)
            {
                None
            } else {
                control.write_barrier
            };
            let result_writeback = produces_dynamic_effects
                .then_some(result_writeback)
                .flatten();
            let observe_async_copy =
                self.has_async_copy_protocol && async_copy_source_count(inst.kind).is_some();
            let fixed_latency_status = (self.params.resource_source() == IssueResourceSource::Sm120Alpha)
                .then(|| self.warps[wix]
                .next_slot
                .map(|current| current.fixed_latency_operands())).flatten();
            let fixed_register_ready_at = match pipeline_class {
                PipelineClass::ScalarAlu { latency_cycles, .. } => {
                    checked_cycle_add("fixed-register completion", t, latency_cycles)?
                }
                _ => checked_cycle_add("fixed-register completion", t, 1)?,
            };
            self.warps[wix].register_ready.retain(|entry| entry.ready_at > t);
            if let Some(status) = fixed_latency_status {
                issue_record_fixed_register_writes(
                    &mut self.warps[wix].register_ready,
                    status,
                    inst.effective_lane_mask(),
                    fixed_register_ready_at,
                );
            }

            // The instruction has passed its IMC probe and is now dispatched;
            // touch each resident line in the cloned cache state.  A line that
            // was just filled is already present after `complete_ready` at the
            // top of this scheduler iteration.
            if let Some(cache) = imc_cache.as_mut()
                && let Some(lines) = self.immediate_constant_lines(inst.pc)
            {
                for &line in lines {
                    cache[sc].touch(line);
                }
            }
            self.warps[wix].imc_wait_until = 0;

            let memory_pipeline_issue = memory_pipeline_plan
                .as_ref()
                .and_then(|plan| plan.issues[sc]);
            let memory_time = memory_pipeline_issue.map_or(t, |issue| issue.shared_accept_at);
            let sm120_flat_service_completion = if self.sm120_flat_service_enabled {
                let opcode_text = self
                    .schedule
                    .issue_slot(u64::from(inst.pc))
                    .map(|slot| slot.opcode_text())
                    .unwrap_or("UNKNOWN");
                sm120_flat_service_work(inst.kind, opcode_text).map(|(global, work)| {
                    let ready = if global {
                        &mut self.sm120_flat_global_service_ready
                    } else {
                        &mut self.sm120_flat_shared_service_ready
                    };
                    let done = (*ready).max(t).saturating_add(work);
                    *ready = done;
                    done
                })
            } else if self.sm120_flat_shared_medium {
                let wavefronts = match inst.kind {
                    AccessKind::Shared { wavefronts } => wavefronts,
                    _ => 0,
                };
                if wavefronts == 0 {
                    None
                } else {
                    let work = wavefronts
                        .saturating_mul(SM120_FLAT_SHARED_MEDIUM_EXTRA_CYCLES_PER_WAVEFRONT);
                    let done = self
                        .sm120_flat_shared_service_ready
                        .max(t)
                        .saturating_add(work);
                    self.sm120_flat_shared_service_ready = done;
                    Some(done)
                }
            } else if self.sm120_flat_multiwarp_service_enabled {
                let len = match inst.kind {
                    AccessKind::GlobalEligible { len, .. }
                    | AccessKind::GlobalBypass { len, .. }
                    | AccessKind::GlobalUntagged { len, .. }
                    | AccessKind::GlobalStore { len, .. }
                    | AccessKind::GlobalReduction { len, .. } => len,
                    _ => 0,
                };
                if len == 0 {
                    None
                } else {
                    let work = u64::from(len)
                        .saturating_mul(SM120_FLAT_MULTI_WARP_GLOBAL_SERVICE_CYCLES_PER_SECTOR);
                    let done = self
                        .sm120_flat_multiwarp_global_service_ready
                        .max(t)
                        .saturating_add(work)
                        .saturating_add(if self.sm120_flat_multiwarp_dense {
                            SM120_FLAT_MULTI_WARP_DENSE_EXTRA_CYCLES
                        } else if self.sm120_flat_multiwarp_medium {
                            SM120_FLAT_MULTI_WARP_MEDIUM_EXTRA_CYCLES
                        } else {
                            0
                        });
                    self.sm120_flat_multiwarp_global_service_ready = done;
                    Some(done)
                }
            } else {
                None
            };
            let sm120_flat_store_retire_completion = if self.sm120_flat_store_retire_enabled {
                let (len, per_sector) = match inst.kind {
                    AccessKind::GlobalStore { len, .. } =>
                        (u64::from(len), SM120_FLAT_STORE_RETIRE_CYCLES_PER_SECTOR),
                    AccessKind::GlobalReduction { len, .. } =>
                        (u64::from(len), SM120_FLAT_REDUCTION_RETIRE_CYCLES_PER_SECTOR),
                    _ => (0, 0),
                };
                (len != 0).then(|| {
                    let done = self.sm120_flat_store_retire_ready
                        .max(t)
                        .saturating_add(len.saturating_mul(per_sector));
                    self.sm120_flat_store_retire_ready = done;
                    done
                })
            } else { None };
            // Memory hierarchy mutation happens when the local address/operand
            // stages reach the SM-shared acceptor, not when the scheduler first
            // dispatches the instruction.
            let mut data_ready = if matches!(self.mem, MemModel::L1Shared(_)) {
                self.mem_step_shared(
                    sc,
                    wix,
                    instruction,
                    inst.kind,
                    constant_load,
                    constant_deep,
                    warp_sectors,
                    write_barrier,
                    async_destination_wavefronts[sc],
                    memory_time,
                    out,
                )?
            } else {
                data_ready_time(
                    inst.kind,
                    warp_sectors,
                    memory_time,
                    &self.params,
                    self.mem.const_l1_mut(),
                    write_barrier,
                    observe_async_copy,
                )?
            };
            if self.params.resource_source() == IssueResourceSource::Sm120Alpha
                && matches!(self.mem, MemModel::Flat)
                && self.sm120_flat_async_dense
                && let AccessKind::GlobalAsyncCopy { off, len: len @ 1.., l1_policy, .. } = inst.kind
            {
                // Retain a bounded recent-source working set for non-bulk
                // cp.async. Repeated tile copies cannot all be cold DRAM
                // accesses. Existing fills carry their own ready timestamp.
                // 128 sectors is a logical 4-KiB window, not inferred L1 size.
                let mut source_ready = t;
                for &sector in &warp_sectors[off..off + usize::from(len)] {
                    let ready = if l1_policy == LdgstsL1Policy::CacheAtAllLevels {
                        if let Some(position) = self.sm120_async_sources.iter().position(|&(key, _)| key == sector) {
                            let entry = self.sm120_async_sources.remove(position).expect("located source");
                            self.sm120_async_sources.push_back(entry);
                            entry.1.max(checked_cycle_add("warm async source", t, self.params.l1_hit_latency)?)
                        } else {
                            let latency = if self.sm120_flat_async_dense {
                                // Fresh group-depth probes show dense repeated
                                // cp.async source traffic is not a cold-DRAM
                                // ladder: NCU reports only 104/128/168 DRAM
                                // sectors for 64/128/256 groups. Use the
                                // logical L2 arm for this shape-conditioned
                                // source frontier; it is not an L2 geometry
                                // claim.
                                self.params.l1_miss_latency
                            } else {
                                self.params.global_flat_latency
                            };
                            let ready = checked_cycle_add("cold async source", t, latency)?;
                            self.sm120_async_sources.push_back((sector, ready));
                            if self.sm120_async_sources.len() > 128 { self.sm120_async_sources.pop_front(); }
                            ready
                        }
                    } else { checked_cycle_add("bypass async source", t, self.params.global_flat_latency)? };
                    source_ready = source_ready.max(ready);
                }
                data_ready = Some(source_ready);
            }
            if let Some(service_done) = sm120_flat_service_completion
                && (write_barrier.is_some()
                    || control.read_barrier.is_some()
                    || observe_async_copy)
            {
                data_ready = Some(data_ready.map_or(service_done, |ready| ready.max(service_done)));
            }
            if let Some(service_done) = sm120_flat_store_retire_completion {
                data_ready = Some(data_ready.map_or(service_done, |ready| ready.max(service_done)));
            }
            if cctl_invalidate_all && matches!(self.mem, MemModel::L1Shared(_)) {
                self.invalidate_l1_all()?;
                if let MemModel::L1Shared(st) = &mut self.mem
                    && let Some(requester) = st.l1_visibility_requester
                {
                    let sequence = st.next_l1_visibility_sequence()?;
                    st.l1_invalidate_all_observations
                        .push(sim_mem::L1InvalidateAllObservation {
                            sequence,
                            requester,
                            issue_cycle: t,
                        });
                }
            }
            if let Some((scope, order)) = fence_semantics
                && let MemModel::L1Shared(st) = &mut self.mem
                && let Some(requester) = st.l1_visibility_requester
            {
                let sequence = st.next_l1_visibility_sequence()?;
                st.l1_fence_observations.push(sim_mem::L1FenceObservation {
                    sequence,
                    requester,
                    pc: inst.pc,
                    issue_cycle: t,
                    scope,
                    order,
                });
            }
            if observe_async_copy && let Some(source_ready) = data_ready {
                // Shared mode already attached the copy's delivery and exact
                // destination work to its pending group before walking source
                // sectors. This second update contributes only the known hit
                // floor; synchronous modes create the group here instead.
                let (delivery_work, destination_wavefronts) =
                    if matches!(self.mem, MemModel::L1Shared(_)) {
                        (0, 0)
                    } else {
                        (
                            if self.sm120_flat_async_dense { 0 } else {
                                async_copy_completion_work(inst.kind).unwrap_or(0)
                            },
                            async_destination_wavefronts[sc],
                        )
                    };
                self.async_copies[wix].add_known_async_copy(
                    wix,
                    source_ready,
                    delivery_work,
                    destination_wavefronts,
                    &self.params,
                )?;
            }

            match pipeline_class {
                PipelineClass::Branch { .. } => {}
                PipelineClass::Barrier { .. } => {
                    if let Some(arrival) = barrier_arrival_times[sc] {
                        self.barrier_pipe_ready = arrival;
                    }
                }
                PipelineClass::Tensor {
                    initiation_interval_cycles,
                } => {
                    self.tensor_pipe_ready[sc] = checked_cycle_add(
                        "tensor-pipeline availability",
                        t,
                        initiation_interval_cycles,
                    )?;
                }
                PipelineClass::Sfu {
                    initiation_interval_cycles,
                } => {
                    self.sfu_pipe_ready[sc] = checked_cycle_add(
                        "uniform/SFU pipeline availability",
                        t,
                        initiation_interval_cycles,
                    )?;
                }
                PipelineClass::ScalarAlu { .. } => {}
                PipelineClass::DispatchOnly => {}
            }
            // Yield semantics per the CGGTY paper: the yield bit means
            // "do not issue this warp in the NEXT cycle" — a one-cycle
            // bubble when no other warp is ready — on top of the stall
            // count. A retiring instruction has no next eligibility to observe.
            if !retires_warp {
                // The retained SM120 high-word control decoder is a
                // transport hypothesis. P9's dependent-chain calibration
                // shows that applying its scalar stall field on top of the
                // measured family latency double-counts readiness. Keep the
                // field for non-scalar classes, but let the calibrated
                // ScalarAlu pipeline own scalar readiness on SM120.
                let control_stall = if scalar_control_stall_suppressed {
                    0
                } else {
                    control.stall
                };
                let stall = u64::from(control_stall.max(1));
                let mut eligible = checked_cycle_add("warp stall eligibility", t, stall)?;
                if control.yield_flag {
                    eligible = eligible.max(checked_cycle_add("warp yield eligibility", t, 2)?);
                }
                if let Some(service_done) = sm120_flat_store_retire_completion {
                    eligible = eligible.max(service_done);
                }
                if (self.sm120_flat_multiwarp_medium || self.sm120_flat_shared_medium)
                    && let Some(service_done) = sm120_flat_service_completion
                {
                    // Dense multi-warp DRAM calibration exposes admission
                    // backpressure, not merely a terminal drain. Park the
                    // issuing warp at the logical service completion so the
                    // existing scheduler can overlap other warps naturally.
                    eligible = eligible.max(service_done);
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
                self.warps[wix].stamp_clear(write_barrier, dr, dependency_origin(inst.kind));
            }
            if let Some(result_writeback) = result_writeback {
                self.register_l1tex_writeback(
                    sc,
                    wix,
                    result_writeback,
                    write_barrier,
                    data_ready,
                )?;
                self.apply_fixed_result_priority(sc, result_writeback, &mut projected_readiness)?;
            }
            if produces_dynamic_effects && let Some(read_barrier) = control.read_barrier {
                let read_clear = memory_pipeline_issue.map_or_else(
                    || {
                        checked_cycle_add(
                            "read-dependence barrier visibility",
                            t,
                            self.params.read_barrier_latency,
                        )
                    },
                    |issue| Ok(issue.operand_ready_at),
                )?;
                let read_clear = sm120_flat_service_completion
                    .map_or(read_clear, |service_done| read_clear.max(service_done));
                self.warps[wix].stamp_clear(
                    Some(read_barrier),
                    read_clear,
                    operand_read_dependency_origin(inst.kind),
                );
            }
            if clock_read {
                let warp = &mut self.warps[wix];
                warp.clock_first_issue.get_or_insert(t);
                warp.clock_reads_issued = warp
                    .clock_reads_issued
                    .checked_add(1)
                    .expect("CLOCKLO issue counter was preflighted");
            }
            if self.has_async_copy_protocol {
                match slot_class {
                    OpClass::AsyncCopyCommit => {
                        let expected_nonempty = self.async_copies[wix]
                            .state
                            .as_deref()
                            .and_then(|state| state.open_group)
                            .is_some();
                        let committed = self.async_copies[wix].commit_async_copy_group(t);
                        debug_assert_eq!(
                            committed, expected_nonempty,
                            "async commit mutation disagrees with its pre-issue state"
                        );
                        if committed && self.sm120_flat_async_service_enabled {
                            let done = self.sm120_flat_async_service_ready
                                .max(t)
                                .saturating_add(SM120_FLAT_ASYNC_GROUP_SERVICE_CYCLES);
                            self.sm120_flat_async_service_ready = done;
                            if let Some(state) = self.async_copies[wix].state.as_deref_mut()
                                && let Some(group) = state.committed_groups.back_mut()
                            {
                                group.destination_completion_time =
                                    group.destination_completion_time.max(done);
                                group.completion_time = group.completion_time.max(done);
                            }
                        }
                        if committed && self.sm120_flat_async_depth_service_per_group != 0 {
                            let done = self.sm120_flat_async_depth_service_ready
                                .max(t)
                                .saturating_add(self.sm120_flat_async_depth_service_per_group);
                            self.sm120_flat_async_depth_service_ready = done;
                            if let Some(state) = self.async_copies[wix].state.as_deref_mut()
                                && let Some(group) = state.committed_groups.back_mut()
                            {
                                group.destination_completion_time =
                                    group.destination_completion_time.max(done);
                                group.completion_time = group.completion_time.max(done);
                            }
                        }

                    }
                    OpClass::AsyncCopyWait => {
                        let max_pending_groups = self.warps[wix]
                            .next_slot
                            .ok_or(IssueError::MissingIssueSlot { warp: wix })?
                            .async_copy_wait_max_pending_groups;
                        self.async_copies[wix].consume_async_wait(max_pending_groups, t);
                    }
                    // `Other` includes explicit SB1..SB5 DEPBAR operand sets.
                    // Readiness was enforced through ordinary barrier state;
                    // they never mutate asynchronous-copy groups.
                    OpClass::BlockBarrier | OpClass::Other => {}
                }
            }

            // Advance the warp's stream.
            if self.instruction_front_end.is_some() {
                self.warps[wix].consume_buffered_issue(wix)?;
            }
            self.warps[wix].next =
                checked_usize_counter_add("warp instruction cursor", self.warps[wix].next, 1)?;
            #[cfg(feature = "reference")]
            self.reference_issue_effects.push((
                self.t,
                u64::try_from(wix).unwrap_or(u64::MAX),
                u32::try_from(instruction).unwrap_or(u32::MAX),
            ));
            let final_block_barrier = slot_class == OpClass::BlockBarrier
                && self.warps[wix].next == self.warps[wix].insts.len();
            if retires_warp {
                self.retire_warp(sc, wix, t)?;
            } else if final_block_barrier {
                // Keep a final barrier warp unfinished and parked until the
                // CTA rendezvous. There is intentionally no next issue slot;
                // the release path retires it after all participants arrive.
                self.warps[wix].next_slot = None;
                self.warps[wix].ordinary_dependency_wait_mask = WaitMask::NONE;
                self.warps[wix].eligible = t;
            } else {
                let next_pc = self.warps[wix].insts[self.warps[wix].next].pc;
                let next_slot = lookup_dense(
                    &self.schedule_slots,
                    self.schedule,
                    next_pc,
                    launch_index,
                    warp_slot,
                )?;
                self.warps[wix].next_slot = Some(next_slot);
                self.warps[wix].ordinary_dependency_wait_mask =
                    next_slot.ordinary_dependency_wait_mask();
                self.warps[wix].current_since = t;
                self.warps[wix].imc_wait_until = 0;
                self.warps[wix].cta_barrier_ready = 0;
                self.warps[wix].barrier_pipeline_ready = 0;
                self.warps[wix].branch_pipeline_ready = 0;
            }
            if !retires_warp
                && let PipelineClass::Branch {
                    next_issue_interval_cycles,
                } = pipeline_class
            {
                self.warps[wix].branch_pipeline_ready =
                    checked_cycle_add("branch-pipeline resolution", t, next_issue_interval_cycles)?;
            }
            // Single hold write: never hold a yielded or finished warp.
            self.hold[sc] = (!control.yield_flag && !self.warps[wix].done).then_some(wix);

            // Block barrier arrival AFTER stream bookkeeping: a finished
            // warp never waits.
            if slot_class == OpClass::BlockBarrier && !self.warps[wix].done {
                let arrival = barrier_arrival_times[sc].unwrap_or(t);
                self.warps[wix].at_block_barrier = true;
                self.warps[wix].block_barrier_dispatch = Some(t);
                self.warps[wix].block_barrier_arrival = Some(arrival);
                self.block_states[blk_ix]
                    .barrier_first_dispatch
                    .get_or_insert(t);
                self.block_states[blk_ix]
                    .barrier_first_arrival
                    .get_or_insert(arrival);
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
                let mut latest_arrival = 0u64;
                let mut second_latest_arrival = 0u64;
                let mut latest_count = 0usize;
                for warp in &self.warps[first..first + count] {
                    if !warp.at_block_barrier {
                        continue;
                    }
                    let arrival = warp
                        .block_barrier_arrival
                        .expect("barrier waiter owns an arrival timestamp");
                    if arrival > latest_arrival {
                        second_latest_arrival = latest_arrival;
                        latest_arrival = arrival;
                        latest_count = 1;
                    } else if arrival == latest_arrival {
                        latest_count += 1;
                    } else if arrival > second_latest_arrival {
                        second_latest_arrival = arrival;
                    }
                }
                self.block_states[blk_ix].waiting = 0;
                self.block_states[blk_ix].barrier_generation = self.block_states[blk_ix]
                    .barrier_generation
                    .checked_add(1)
                    .expect("barrier generation was preflighted");
                self.block_states[blk_ix].barrier_first_dispatch = None;
                self.block_states[blk_ix].barrier_first_arrival = None;
                if let Some(issue) = self.barrier_issue {
                    self.block_states[blk_ix].barrier_rearm_ready = checked_cycle_add(
                        "CTA-barrier generation re-arm",
                        latest_arrival,
                        issue.generation_rearm_cycles,
                    )?;
                }
                let release_at = checked_cycle_add("block-barrier release", latest_arrival, 1)?;
                self.local_completion_time = Some(
                    self.local_completion_time
                        .map_or(release_at, |current| current.max(release_at)),
                );
                // Final BAR.SYNC instructions advanced their warps to the end
                // of stream but deliberately left them unfinished above.
                // Collect those participants before clearing the barrier
                // flags, then retire them through the common residency path.
                let final_barrier_warps = self.warps[first..first + count]
                    .iter()
                    .enumerate()
                    .filter_map(|(offset, warp)| {
                        (warp.at_block_barrier && warp.next == warp.insts.len())
                            .then_some(first + offset)
                    })
                    .collect::<Vec<_>>();
                for w in &mut self.warps[first..first + count] {
                    if w.at_block_barrier {
                        let own_arrival = w
                            .block_barrier_arrival
                            .expect("barrier waiter owns an arrival timestamp");
                        let other_arrival = if count <= 1 {
                            None
                        } else if own_arrival == latest_arrival && latest_count == 1 {
                            Some(second_latest_arrival)
                        } else {
                            Some(latest_arrival)
                        };
                        w.at_block_barrier = false;
                        w.block_barrier_dispatch = None;
                        w.block_barrier_arrival = None;
                        if self.barrier_issue.is_some() {
                            w.barrier_pipeline_ready = checked_cycle_add(
                                "CTA-barrier own-arrival release",
                                own_arrival,
                                1,
                            )?;
                            w.cta_barrier_ready = other_arrival
                                .map(|arrival| {
                                    checked_cycle_add(
                                        "CTA-barrier other-participant rendezvous",
                                        arrival,
                                        1,
                                    )
                                })
                                .transpose()?
                                .unwrap_or(0);
                            debug_assert_eq!(
                                w.barrier_pipeline_ready.max(w.cta_barrier_ready),
                                release_at
                            );
                        } else {
                            w.barrier_pipeline_ready = 0;
                            w.cta_barrier_ready = release_at;
                        }
                    }
                }
                for warp_index in final_barrier_warps {
                    let subcore = self.warps[warp_index]
                        .warp_slot
                        .rem_euclid(SUB_CORES as i32) as usize;
                    // Admission of the next CTA is causally after the barrier
                    // release, not after the last warp's dispatch cycle.
                    self.retire_warp(subcore, warp_index, release_at)?;
                }
            }
        }
        if let Some(plan) = memory_pipeline_plan {
            self.memory_issue_pipeline = Some(plan.next);
        }
        if let Some(plan) = scalar_execution_pipeline_plan {
            self.scalar_execution_pipeline = Some(Box::new(plan.next));
        }
        for &(warp, ready) in imc_waits.iter().flatten() {
            self.warps[warp].imc_wait_until = ready;
        }
        if let Some(cache) = imc_cache.as_mut() {
            for wait in imc_waits.iter().flatten() {
                let subcore = self.warps[wait.0].warp_slot.rem_euclid(SUB_CORES as i32) as usize;
                cache[subcore].diagnostics.miss_wait_events = cache[subcore]
                    .diagnostics
                    .miss_wait_events
                    .saturating_add(1);
                cache[subcore].diagnostics.miss_wait_cycles = cache[subcore]
                    .diagnostics
                    .miss_wait_cycles
                    .saturating_add(1);
            }
        }
        if imc_cache.is_some() {
            self.immediate_constant_cache = imc_cache;
        }
        // Counts are bounded by next_issued, whose checked increment was
        // preflighted above. Multiple issue ports count only one active cycle.
        for (subcore, pick) in effective_picks.iter().enumerate() {
            if pick.is_some() {
                self.instructions_per_smsp[subcore] = self.instructions_per_smsp[subcore]
                    .checked_add(1)
                    .ok_or(IssueError::CounterOverflow {
                        counter: "instructions per SMSP",
                    })?;
                if self.last_observed_issue[subcore] != Some(t) {
                    self.issue_active_cycles_per_smsp[subcore] = self.issue_active_cycles_per_smsp
                        [subcore]
                        .checked_add(1)
                        .ok_or(IssueError::CounterOverflow {
                            counter: "issue-active cycles per SMSP",
                        })?;
                    self.last_observed_issue[subcore] = Some(t);
                }
            }
        }
        self.issued = next_issued;
        *self.readiness = projected_readiness;
        self.memory_service_candidates = projected_memory_service_candidates;
        if let Some(plan) = memory_admission_feedback_plan {
            self.memory_admission_feedback
                .as_mut()
                .expect("feedback issue plan requires installed feedback state")
                .apply_issue(plan, t);
        }
        if issued_this_cycle {
            self.last_issue_picks = effective_picks;
        }
        Ok(issued_this_cycle)
    }
}

/// Register a completion target that clears when `fill` completes. The
/// asynchronous-copy group already owns its pending source unit; an optional
/// ordinary barrier is counted here alongside ordinary load barriers.
fn add_fill_waiter(
    st: &mut SharedL2State,
    warps: &mut [WarpState<'_>],
    fill: FillKey,
    warp: usize,
    target: Option<FillTarget>,
    constant_deep: bool,
) -> Result<(), IssueError> {
    let Some(target) = target else {
        return Ok(());
    };
    warps[warp].add_pending(warp, target.barrier(), 1)?;
    match st.fill_waiters.entry(fill) {
        Entry::Vacant(v) => {
            v.insert(Waiters::One(FillWaiter {
                warp,
                target,
                units: 1,
                constant_deep,
            }));
        }
        Entry::Occupied(mut o) => o.get_mut().add(warp, target, constant_deep)?,
    }
    Ok(())
}

fn preflight_line_fill_mask(st: &SharedL2State, sector_id: i64) -> Result<(), IssueError> {
    let line = sector_id >> 2;
    let fill_in_flight = st.l1.is_line_fill_in_flight(line);
    let has_mask = st.fill_masks.contains_key(&line);
    match (fill_in_flight, has_mask) {
        (true, false) => Err(IssueError::FillMaskMissing { line }),
        (false, true) => Err(IssueError::FillMaskWithoutFill { line }),
        _ => Ok(()),
    }
}

fn register_response(st: &mut SharedL2State, reply: ReplyTo) -> Result<(), IssueError> {
    let response_id = reply
        .response_id()
        .ok_or(IssueError::UnexpectedMutationOnlyResponse)?;
    if let Some(&existing) = st.outstanding_response_ids.get(&response_id) {
        return Err(IssueError::ResponseTokenAlreadyOutstanding { reply: existing });
    }
    if !st.outstanding_replies.insert(reply) {
        return Err(IssueError::ResponseTokenAlreadyOutstanding { reply });
    }
    st.outstanding_response_ids.insert(response_id, reply);
    Ok(())
}

enum SharedL1Access {
    Hit { ready: Option<u64> },
    MissRequest { fill: FillKey },
    MergedPending { fill: FillKey },
    Stalled,
}

fn record_global_l1_observation(
    st: &mut SharedL2State,
    sector_id: i64,
    is_write: bool,
    l1_resident_after: bool,
    hit: bool,
    fill_generation: Option<u32>,
    issue_cycle: u64,
) -> Result<(), IssueError> {
    let Some(requester) = st.l1_visibility_requester else {
        return Ok(());
    };
    let sequence = st.next_l1_visibility_sequence()?;
    st.l1_visibility_observations
        .push(sim_mem::L1AccessObservation {
            sequence,
            line_id: sector_id >> 2,
            sector_mask: 1_u8 << (sector_id & 3) as u8,
            requester,
            is_write,
            l1_resident_after,
            fill_will_reside: fill_generation.is_some(),
            fill_generation,
            hit,
            issue_cycle,
        });
    Ok(())
}

fn fill_generation(fill: FillKey) -> u32 {
    match fill {
        FillKey::Line { generation, .. } | FillKey::Sector { generation, .. } => generation,
    }
}

fn access_shared_l1(
    l1: &mut L1Cache,
    origin: MemoryOrigin,
    sector_id: i64,
    now: u64,
    observe_ready: bool,
    mshr_floor: Option<NonZeroU32>,
) -> Result<SharedL1Access, L1AccessError> {
    if observe_ready {
        let access = match mshr_floor {
            Some(floor) => l1.access_eligible_ev_with_mshr_floor(origin, sector_id, now, floor)?,
            None => l1.access_eligible_ev(origin, sector_id, now)?,
        };
        return Ok(match access {
            EvAccess::Hit { ready } => SharedL1Access::Hit { ready: Some(ready) },
            EvAccess::MissRequest { fill } => SharedL1Access::MissRequest { fill },
            EvAccess::MergedPending { fill } => SharedL1Access::MergedPending { fill },
            EvAccess::Stalled => SharedL1Access::Stalled,
        });
    }
    let access = match mshr_floor {
        Some(floor) => {
            l1.access_eligible_ev_unobserved_with_mshr_floor(origin, sector_id, now, floor)?
        }
        None => l1.access_eligible_ev_unobserved(origin, sector_id, now)?,
    };
    Ok(match access {
        EvAccessUnobserved::Hit => SharedL1Access::Hit { ready: None },
        EvAccessUnobserved::MissRequest { fill } => SharedL1Access::MissRequest { fill },
        EvAccessUnobserved::MergedPending { fill } => SharedL1Access::MergedPending { fill },
        EvAccessUnobserved::Stalled => SharedL1Access::Stalled,
    })
}

#[derive(Debug)]
enum SharedAccessError {
    Stalled,
    Issue(IssueError),
}

impl From<IssueError> for SharedAccessError {
    fn from(source: IssueError) -> Self {
        Self::Issue(source)
    }
}

/// One eligible sector access in L1Shared mode at time `now`, emitting a
/// request on a fresh miss. Returns `Ok(known_ready)` for hits,
/// `Ok(None)` when the data waits on a response, and
/// `Err(SharedAccessError::Stalled)` on an MSHR stall (caller parks the
/// remainder). Accounting failures remain typed.
///
/// A sector merging onto a line fill emitted during the CURRENT
/// emission batch (`st.line_folds`, keyed by line and issue timestamp)
/// folds its bit into that request's `sector_mask`, so one L2
/// transaction carries same-cycle sectors and worst-of-mask pricing
/// covers every warp/instruction/retry-entry at that timestamp. A merge onto
/// an OLDER in-flight fill emits a follow-up sector fetch with an independent
/// `FillKey::Sector` response; the L2 must see every fetched sector exactly
/// once per fill and the waiter clears only at that sector's ready event.
fn access_shared(
    st: &mut SharedL2State,
    warps: &mut [WarpState<'_>],
    wix: usize,
    target: Option<FillTarget>,
    mshr_floor: Option<NonZeroU32>,
    origin: MemoryOrigin,
    constant_load: bool,
    constant_deep: bool,
    sector_id: i64,
    response_id: ResponseId,
    now: u64,
    out: &mut Vec<MemRequest>,
) -> Result<Option<u64>, SharedAccessError> {
    let sbit = 1u8 << (sector_id & 3) as u8;
    match access_shared_l1(
        &mut st.l1,
        origin,
        sector_id,
        now,
        target.is_some(),
        mshr_floor,
    )
    .map_err(IssueError::from)?
    {
        SharedL1Access::Hit { ready } => {
            if origin == MemoryOrigin::Global {
                record_global_l1_observation(st, sector_id, false, true, true, None, now)?;
            }
            Ok(ready)
        }
        SharedL1Access::MissRequest { fill } => {
            if origin == MemoryOrigin::Global {
                // The directory must not publish ownership before the L1 fill
                // response is visible. A later fill-visible event is required
                // to make this sector resident in the visibility protocol.
                record_global_l1_observation(
                    st,
                    sector_id,
                    false,
                    false,
                    false,
                    Some(fill_generation(fill)),
                    now,
                )?;
            }
            let request = MemRequest::l1_fill(
                now,
                sector_id,
                origin,
                fill,
                response_id,
                constant_load,
                constant_deep,
            );
            register_response(st, request.reply)?;
            st.response_masks.insert(response_id, sbit);
            out.push(request);
            if let FillKey::Line { line, .. } = fill {
                st.fill_masks.insert(line, sbit);
                st.line_folds.entry((line, now)).or_insert(out.len() - 1);
            }
            add_fill_waiter(st, warps, fill, wix, target, constant_deep)?;
            Ok(None)
        }
        SharedL1Access::MergedPending { fill } => {
            if origin == MemoryOrigin::Global {
                record_global_l1_observation(
                    st,
                    sector_id,
                    false,
                    false,
                    false,
                    Some(fill_generation(fill)),
                    now,
                )?;
            }
            if let FillKey::Line { line, .. } = fill {
                // Pairing invariant (proven at the 2026-07-07 gate): a
                // line-pending merge always has a live fill_masks entry
                // (created at MissRequest, removed at the response that
                // also completes the fill). Fail LOUD if a future
                // change breaks it — a fabricated empty mask would
                // silently re-fetch already-sent sectors.
                let already_sent = st.fill_masks.get(&line).ok_or(SharedAccessError::Issue(
                    IssueError::FillMaskMissing { line },
                ))? & sbit
                    != 0;
                if !already_sent {
                    if let Some(&ix) = st.line_folds.get(&(line, now)) {
                        *st.fill_masks
                            .get_mut(&line)
                            .expect("fill mask checked above") |= sbit;
                        out[ix].fold_sector_origin(origin, sbit, constant_load, constant_deep);
                        if let Some(response_id) = out[ix].reply.response_id() {
                            *st.response_masks.entry(response_id).or_insert(0) |= sbit;
                        }
                    } else {
                        let sector_fill = st
                            .l1
                            .reserve_pending_line_sector(fill, (sector_id & 3) as u8)
                            .map_err(IssueError::from)?;
                        *st.fill_masks
                            .get_mut(&line)
                            .expect("fill mask checked above") |= sbit;
                        let request = MemRequest::l1_followup_fetch(
                            now,
                            sector_id,
                            origin,
                            sector_fill,
                            response_id,
                            constant_load,
                            constant_deep,
                        );
                        register_response(st, request.reply)?;
                        st.response_masks.insert(response_id, sbit);
                        out.push(request);
                        add_fill_waiter(st, warps, sector_fill, wix, target, constant_deep)?;
                        return Ok(None);
                    }
                }
                // Re-touch of an already-sent sector: MSHR merge, no
                // new fetch (matches NCU's xbar sector accounting).
            }
            // Sector-fill merges need nothing: that sector's fetch was
            // sent with its own request.
            add_fill_waiter(st, warps, fill, wix, target, constant_deep)?;
            Ok(None)
        }
        SharedL1Access::Stalled => Err(SharedAccessError::Stalled),
    }
}

fn drain_retries(
    st: &mut SharedL2State,
    warps: &mut [WarpState<'_>],
    async_copies: &mut [AsyncCopySlot],
    now: u64,
    params: &IssueModelParams,
    out: &mut Vec<MemRequest>,
) -> Result<(), IssueError> {
    while let Some(mut r) = st.retry.pop_front() {
        st.l1.resume_lookup_batch(r.lookup_batch);
        let sectors = warps[r.warp].sectors;
        let barrier = r.target.and_then(FillTarget::barrier);
        // Replace the retry entry's ordinary-barrier unit atomically with
        // whatever waiter units this attempt creates. Async-copy source units
        // remain owned by their group until a hit or fill completion resolves
        // each sector.
        warps[r.warp].release_pending(r.warp, barrier, 1)?;
        // The drain is one emission batch, but folds remain timestamp scoped:
        // a line fill emitted by an earlier retry entry only folds later
        // same-timestamp sectors; later-cycle sectors receive child tokens.
        while r.lo < r.hi {
            match access_shared(
                st,
                warps,
                r.warp,
                r.target,
                r.mshr_floor,
                r.origin,
                r.constant_load,
                r.constant_deep,
                sectors[r.lo],
                ResponseId::new(st.response_owner, r.warp, r.instruction, r.lo),
                now,
                out,
            ) {
                Ok(Some(ready)) => {
                    let ready = checked_constant_return_time(
                        ready,
                        r.constant_deep && barrier.is_some(),
                        params,
                        "constant retry L1-hit return latency",
                    )?;
                    if let Some(group) = r.target.and_then(FillTarget::async_group) {
                        async_copies[r.warp]
                            .complete_async_copy_sources(r.warp, group, None, 1, ready, params)?;
                    }
                    warps[r.warp].stamp_clear(barrier, ready, DependencyOrigin::L1Tex);
                }
                Ok(None) => {}
                Err(SharedAccessError::Stalled) => {
                    // Still full: restore the retry entry's replacement unit
                    // and park it; nothing behind it can proceed either.
                    warps[r.warp].add_pending(r.warp, barrier, 1)?;
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
    let atomic_topology = match kind {
        AccessKind::GlobalBypass {
            len,
            token: CacheOpToken::Atom,
            l2_requests,
            ..
        }
        | AccessKind::GlobalReduction {
            len, l2_requests, ..
        } => Some((len, l2_requests, true)),
        AccessKind::GlobalBypass {
            len, l2_requests, ..
        } if l2_requests != 0 => Some((len, l2_requests, false)),
        _ => None,
    };
    if let Some((sector_transactions, l2_requests, is_atomic)) = atomic_topology
        && (!is_atomic
            || (sector_transactions == 0) != (l2_requests == 0)
            || l2_requests > sector_transactions)
    {
        return Err(IssueError::AtomicRequestTopologyInvalid {
            pc,
            launch_index,
            warp_slot,
            sectors: sector_transactions,
            l2_requests,
        });
    }
    let (off, len) = match kind {
        AccessKind::GlobalEligible { off, len, .. }
        | AccessKind::GlobalBypass { off, len, .. }
        | AccessKind::GlobalAsyncCopy { off, len, .. }
        | AccessKind::GlobalUntagged { off, len }
        | AccessKind::GlobalStore { off, len }
        | AccessKind::GlobalReduction { off, len, .. }
        | AccessKind::LocalLoad { off, len }
        | AccessKind::LocalStore { off, len } => (off, len),
        AccessKind::NonMemory | AccessKind::PredicatedOffMemory | AccessKind::Shared { .. } => {
            return Ok(());
        }
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

fn lookup_dense<'a>(
    schedule_slots: &[(u64, &'a IssueSlot)],
    schedule: &FunctionSchedule,
    pc: u32,
    launch_index: i64,
    warp_slot: i32,
) -> Result<&'a IssueSlot, IssueError> {
    schedule_slots
        .binary_search_by_key(&u64::from(pc), |(slot_pc, _)| *slot_pc)
        .map(|index| schedule_slots[index].1)
        .map_err(|_| IssueError::PcNotInSchedule {
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
pub(crate) fn simulate_sm_with_residency(
    blocks: &[BlockReplay],
    schedule: &FunctionSchedule,
    params: &IssueModelParams,
    shared_memory_residency: SharedMemoryResidency,
    l1_config: Option<L1Config>,
) -> Result<SmSimResult, IssueError> {
    let mode = match l1_config {
        Some(cfg) => SmMemMode::L1Const(cfg),
        None => SmMemMode::Flat,
    };
    let mut core =
        SmCore::new_with_residency(blocks, schedule, params, shared_memory_residency, mode)?;
    let mut out = Vec::new();
    match core.advance(ReplayHorizon::Unbounded, &mut out)? {
        Advance::Done => {}
        Advance::Idle { .. } => unreachable!("synchronous modes emit no requests"),
    }
    debug_assert!(out.is_empty(), "synchronous modes emit no requests");
    Ok(core.result())
}

