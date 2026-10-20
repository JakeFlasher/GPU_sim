// Lexical implementation section 1/8; included by the parent facade.
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
    L1Cache, L1Config, L1Counters, L1LookupBatch, L1LookupClient, L1MergeAudit, L1MutationBudget,
    L1StorePolicy, L1WriteLookupObserver, L1WriteLookupObserverPlan, L2Source, LdgstsL1Policy,
    MemRole, MemoryOrigin, decode_async_copy_completion, opcode_head,
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
    FunctionSchedule, IssueSlot, OpClass, PipelineClass,
    sm120_alpha_scalar_control_stall_suppressed,
};

use crate::sass_operands::{
    FixedLatencyOpcode, FixedLatencyOperandStatus, SassRegister, SassRegisterFile,
    SassSourceOperand,
};

pub(crate) fn opcode_has_qualifier(opcode_text: &str, qualifier: &str) -> bool {
    opcode_text
        .split_ascii_whitespace()
        .next()
        .is_some_and(|opcode| opcode.split('.').any(|part| part == qualifier))
}

pub use sim_mem::AccessKind;

pub(crate) const MAX_ASYNC_DECISION_SNAPSHOTS: usize = 1_024;

pub use config::{IssueModelParams, IssueResourceSource, Sm120AlphaConfig};

pub(crate) use config::{SharedMemoryResidency, checked_cycle_add};

pub use memory_types::{MemoryOperation, TailFlags};

pub use readiness::{
    AsyncInstrumentationOrderDiagnostics, AsyncIssueOverlapDiagnostics, AsyncReadinessDiagnostics,
    AsyncWaitPrefixDiagnostics, BarrierPipelineDiagnostics, BypassAsyncCopyDiagnostics,
    ClockIssueDiagnostics, ControlPipelineDiagnostics, CtaBarrierDiagnostics,
    GlobalMemoryIssueDiagnostics, ReadinessCauseCycles, ReadinessCounterError,
    ResultWritebackDiagnostics,
};

use readiness::{
    checked_readiness_add, dependency_origin, is_zero_u64, observe_readiness_constraint,
    operand_read_dependency_origin, projected_global_memory_issue, projected_l1_write_lookup_hits,
};

pub use candidates::{
    MAX_MEMORY_ADMISSION_CANDIDATES, MAX_MEMORY_ADMISSION_QUEUE_CAPACITY,
    MAX_MEMORY_ADMISSION_SERVICE_WIDTH_SECTORS, MAX_MEMORY_SERVICE_CANDIDATES,
    MemoryAdmissionCandidate, MemoryAdmissionCandidateDiagnostics, MemoryAdmissionCandidateError,
    MemoryAdmissionCandidateResult, MemoryAdmissionFeedbackDiagnostics, MemoryServiceCandidate,
    MemoryServiceCandidateDiagnostics, MemoryServiceCandidateError,
    MemoryServiceCandidateOperation, MemoryServiceCandidateResult, MemoryServiceCandidateScope,
    MemoryWriteAckDiagnostics,
};

pub(crate) use candidates::{
    MemoryAdmissionCandidateState, validate_memory_admission_candidates,
    validate_memory_service_candidates,
};

use candidates::{
    MemoryAdmissionFeedbackIssue, MemoryAdmissionFeedbackState, MemoryServiceCandidateState,
};

/// One dynamic instruction, compacted for replay.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ReplayInst {
    /// Function-relative pc — the join key into [`FunctionSchedule`].
    pub(crate) pc: u32,
    pub(crate) kind: AccessKind,
    /// Effective executing lanes for the dynamic instruction. Native traces
    /// carry both masks; legacy captures remain all-lane.
    pub(crate) lane_mask: u32,
    /// Active lanes also evaluate the guard when its outcome is false.
    pub(crate) active_lane_mask: u32,
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
    /// Block-local dynamic warp identity from the trace roster. Huerta's
    /// youngest-warp comparator uses this identity; retaining it separately
    /// from the physical SM slot makes the scheduler priority auditable.
    pub(crate) warp_id: i32,
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
    #[error("SM120 effective L1 timing override is invalid: hit={hit}, miss={miss}")]
    InvalidL1TimingOverride { hit: u64, miss: u64 },
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
    /// Fetches that continued the warp issued in the prior cycle, independent
    /// of whether issue hold was cleared by Yield.
    pub fetch_last_issued_continuations: u64,
    pub yield_hold_separations: u64,
    /// Dynamic non-fall-through control transfers observed by the front end.
    /// This is an observer only; the rejected H26 flush arm is not active.
    pub control_flow_redirects: u64,
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
        add!(fetch_last_issued_continuations);
        add!(yield_hold_separations);
        add!(control_flow_redirects);
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
    /// Dynamic instructions classified into the source-qualified constant
    /// pressure path (`11 + 39` SM86 operand frontier).
    pub constant_pressure_instructions: u64,
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
                "readiness.memory_issue_pipeline.constant_pressure_instructions",
                &mut next.constant_pressure_instructions,
                addend.constant_pressure_instructions,
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
    /// Cycles for which the source-backed SM86 greedy-warp L0C window
    /// deliberately blocked youngest-warp switching. This is a mechanism
    /// diagnostic, not an elapsed-cycle prediction.
    pub greedy_switch_block_cycles: u64,
    /// Non-greedy candidate probes attempted after a prior candidate missed
    /// in IMC. These counters describe the bounded scan, not hardware warp
    /// sampling.
    pub non_greedy_scan_attempts: u64,
    pub non_greedy_scan_misses: u64,
    pub non_greedy_scan_successes: u64,
    pub maximum_pending_lines: u64,
}

/// Conditional SM86 register-file read envelope. These counters describe the
/// bounded effective model, not a claim about proprietary physical wiring.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct Sm86RegisterReadDiagnostics {
    pub reservations: u64,
    pub source_reads: [u64; 2],
    /// Readiness delay that was actually charged before issue. This is
    /// distinct from a residual same-cycle reservation conflict at dispatch.
    pub applied_wait_cycles: u64,
    pub port_conflict_cycles: u64,
    pub rfc_hits: u64,
    pub rfc_misses: u64,
    pub unsupported_shapes: u64,
}

impl Sm86RegisterReadDiagnostics {
    fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        let mut next = *self;
        next.reservations = checked_readiness_add(
            "readiness.sm86_register_read.reservations",
            next.reservations,
            addend.reservations,
        )?;
        for bank in 0..2 {
            next.source_reads[bank] = checked_readiness_add(
                "readiness.sm86_register_read.source_reads",
                next.source_reads[bank],
                addend.source_reads[bank],
            )?;
        }
        next.port_conflict_cycles = checked_readiness_add(
            "readiness.sm86_register_read.port_conflict_cycles",
            next.port_conflict_cycles,
            addend.port_conflict_cycles,
        )?;
        next.applied_wait_cycles = checked_readiness_add(
            "readiness.sm86_register_read.applied_wait_cycles",
            next.applied_wait_cycles,
            addend.applied_wait_cycles,
        )?;
        next.rfc_hits = checked_readiness_add(
            "readiness.sm86_register_read.rfc_hits",
            next.rfc_hits,
            addend.rfc_hits,
        )?;
        next.rfc_misses = checked_readiness_add(
            "readiness.sm86_register_read.rfc_misses",
            next.rfc_misses,
            addend.rfc_misses,
        )?;
        next.unsupported_shapes = checked_readiness_add(
            "readiness.sm86_register_read.unsupported_shapes",
            next.unsupported_shapes,
            addend.unsupported_shapes,
        )?;
        *self = next;
        Ok(())
    }
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
        add!(greedy_switch_block_cycles);
        add!(non_greedy_scan_attempts);
        add!(non_greedy_scan_misses);
        add!(non_greedy_scan_successes);
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
    /// Count of within-CTA resident-warp priority inversions between the
    /// admission/arrival order and the trace's dynamic warp-id order. This is
    /// observational; it never selects timing or fits NCU.
    pub youngest_priority_order_mismatches: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instruction_front_end: Option<Box<InstructionFrontEndDiagnostics>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_issue_pipeline: Option<Box<MemoryIssuePipelineDiagnostics>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub immediate_constant_cache: Option<Box<ImmediateConstantCacheDiagnostics>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sm86_register_read: Option<Box<Sm86RegisterReadDiagnostics>>,
    pub async_groups: AsyncReadinessDiagnostics,
    /// Read-only scalar async decision records. These are emitted only when a
    /// modeled async-group wait is selected; they never participate in issue
    /// selection or completion timing.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub async_decision_snapshots: Vec<AsyncDecisionSnapshot>,
    /// Number of dynamic decision snapshots intentionally omitted after the
    /// bounded diagnostic sample was filled. Dropping diagnostics must never
    /// reject an otherwise valid real-trace replay.
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub async_decision_snapshots_dropped: u64,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AsyncDecisionSnapshot {
    pub function_unique_id: i32,
    pub pc: u32,
    pub warp_slot: i32,
    pub instruction: u64,
    pub protocol: &'static str,
    pub committed_groups_before: u64,
    pub required_prefix_len: u64,
    pub wait_threshold: u32,
    pub readiness_cycle: u64,
    pub decision_cycle: u64,
    pub issue_cycle: u64,
    /// This diagnostic is emitted only for a warp selected by the current
    /// arbitration pass.  It is not a census of blocked candidates and does
    /// not prove that the async prefix was the cause preventing another warp
    /// from issuing.
    pub observation_scope: &'static str,
    /// Explicit semantic disposition prevents consumers from treating the
    /// selected-only snapshot as the stronger same-state blocked witness that
    /// would require instrumentation before arbitration.
    pub semantics: &'static str,
    pub membership: &'static str,
    pub limiting_cause: Option<&'static str>,
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
        next.youngest_priority_order_mismatches = checked_readiness_add(
            "readiness.youngest_priority_order_mismatches",
            next.youngest_priority_order_mismatches,
            addend.youngest_priority_order_mismatches,
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
        if let Some(addend_register_read) = addend.sm86_register_read.as_deref() {
            if let Some(current) = next.sm86_register_read.as_deref_mut() {
                current.add_checked(addend_register_read)?;
            } else {
                next.sm86_register_read = Some(Box::new(*addend_register_read));
            }
        }
        next.async_groups.add_checked(&addend.async_groups)?;
        // Snapshots are optional diagnostics.  Keep the serialized report
        // bounded even for very large multi-SM launches; the dropped count
        // preserves the fact that sampling was truncated.
        next.async_decision_snapshots_dropped = checked_readiness_add(
            "readiness.async_decision_snapshots_dropped",
            next.async_decision_snapshots_dropped,
            addend.async_decision_snapshots_dropped,
        )?;
        let available =
            MAX_ASYNC_DECISION_SNAPSHOTS.saturating_sub(next.async_decision_snapshots.len());
        let kept = available.min(addend.async_decision_snapshots.len());
        next.async_decision_snapshots
            .extend_from_slice(&addend.async_decision_snapshots[..kept]);
        let omitted = addend.async_decision_snapshots.len().saturating_sub(kept);
        next.async_decision_snapshots_dropped = checked_readiness_add(
            "readiness.async_decision_snapshots_dropped",
            next.async_decision_snapshots_dropped,
            u64::try_from(omitted).unwrap_or(u64::MAX),
        )?;
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
    /// Schema 27 adds the bounded SM86 greedy L0C miss-switch blocking
    /// diagnostic to the immediate-constant accounting. Schema 28 adds the
    /// bounded non-greedy candidate-scan counters.
    /// Schema 29 adds the explicit within-CTA dynamic-youngest priority audit.
    /// Schema 30 adds a dynamic control-flow redirect observer; it does not
    /// activate the rejected branch-stream flush timing arm.
    /// Schema 31 adds the source-qualified constant-pressure memory-pipeline
    /// observer.
    pub fn schema_version(&self) -> u32 {
        // The priority-audit field is serialized for every replay mode, even
        // when no IMC surface is present; retain one unambiguous current
        // schema rather than pretending the additive field is absent.
        return 31;
        #[allow(unreachable_code)]
        let feedback = self.memory_admission_feedback.as_deref();
        if self.immediate_constant_cache.is_some()
            && self
                .immediate_constant_cache
                .as_deref()
                .is_some_and(|cache| {
                    cache.non_greedy_scan_attempts != 0
                        || cache.non_greedy_scan_misses != 0
                        || cache.non_greedy_scan_successes != 0
                })
        {
            28
        } else if self.immediate_constant_cache.is_some() {
            27
        } else if self.limiting_cycles.pipeline_sfu != 0
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
    /// Read-only request-boundary provenance for L1 merge classification.
    pub l1_merge_audit: Option<L1MergeAudit>,
    /// Explicit capability gaps for write-class tails present in this replay.
    pub unmodeled_tails: TailFlags,
    /// Simulator-internal causal readiness diagnostics. These are not NCU
    /// sample counts and their overlapping exposure fields are not elapsed
    /// kernel cycles.
    pub readiness: ReadinessDiagnostics,
    /// Model-relative active-warp integral and per-SMSP issue observations.
    pub activity: SmActivity,
}

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
    let opcode = opcode_text
        .split_ascii_whitespace()
        .next()
        .unwrap_or(opcode_text);
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
        | AccessKind::GlobalReduction { len, .. }
            if len != 0 =>
        {
            Some((
                true,
                u64::from(len).saturating_mul(sm120_flat_global_service_rate(opcode_text)),
            ))
        }
        AccessKind::Shared { wavefronts, .. } if wavefronts != 0 => Some((
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

/// Static client-scope tag for the passive L2 scope observer. It never
/// changes the physical hierarchy route, cache state, DRAM, or timing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum L2ClientScope {
    OrdinaryLdg,
    ConstantLdg,
    Ltc128b,
    AsyncLdgstsCache,
    AsyncLdgstsBypass,
    Local,
    Store,
    Atomic,
    Reduction,
    Unknown,
}

impl L2ClientScope {
    pub(crate) const fn index(self) -> usize {
        match self {
            Self::OrdinaryLdg => 0,
            Self::ConstantLdg => 1,
            Self::Ltc128b => 2,
            Self::AsyncLdgstsCache => 3,
            Self::AsyncLdgstsBypass => 4,
            Self::Local => 5,
            Self::Store => 6,
            Self::Atomic => 7,
            Self::Reduction => 8,
            Self::Unknown => 9,
        }
    }
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
    /// Optional operation-class observer tag. This never changes the
    /// physical hierarchy operation, response timing, or DRAM fetch policy;
    /// it only lets static generic-store forms use the operation-qualified
    /// lookup counters without changing their established read route.
    pub(crate) observer_operation: Option<MemoryOperation>,
    pub(crate) client_scope: L2ClientScope,
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

impl MemRequest {
    /// Operation used only by operation-qualified observer counters.  The
    /// physical hierarchy continues to use `operation`; generic `ST.E` keeps
    /// its established untagged physical route while the observer sees it as
    /// a store.
    pub fn effective_observer_operation(&self) -> MemoryOperation {
        self.observer_operation.unwrap_or(self.operation)
    }
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
        client_scope: L2ClientScope,
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
            observer_operation: None,
            client_scope,
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
        client_scope: L2ClientScope,
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
            observer_operation: None,
            client_scope,
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
            observer_operation: None,
            client_scope: match (origin, operation) {
                (MemoryOrigin::Local, MemoryOperation::Store) => L2ClientScope::Local,
                (MemoryOrigin::Global, MemoryOperation::Store) => L2ClientScope::Store,
                (_, MemoryOperation::Atomic) => L2ClientScope::Atomic,
                (_, MemoryOperation::Reduction) => L2ClientScope::Reduction,
                _ => L2ClientScope::Unknown,
            },
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

