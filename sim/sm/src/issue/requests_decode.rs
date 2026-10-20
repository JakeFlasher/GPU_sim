//! Cohesive child mechanism extracted from the parent module.

use super::*;

pub(crate) fn opcode_has_qualifier(opcode_text: &str, qualifier: &str) -> bool {
    opcode_text
        .split_ascii_whitespace()
        .next()
        .is_some_and(|opcode| opcode.split('.').any(|part| part == qualifier))
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

/// One selected dynamic instruction's 1-based callback/channel order token.
/// `instruction` indexes the owning [`WarpReplay::insts`] stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct InstrumentationOrderAnchor {
    pub(crate) instruction: usize,
    pub(crate) order: u64,
}

/// One traced block assigned to this SM, in launch order.
#[derive(Debug)]
pub(crate) struct BlockReplay {
    pub(crate) launch_index: i64,
    pub(crate) warps: Vec<WarpReplay>,
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

/// Exclusive limit for one [`SmCore::advance`] call. A bounded horizon may be
/// any cycle, including `u64::MAX`; unbounded replay is represented by its own
/// variant instead of stealing a valid timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayHorizon {
    Bounded(u64),
    Unbounded,
}

impl ReplayHorizon {
    pub(crate) fn bounded_time(self) -> Option<u64> {
        match self {
            Self::Bounded(time) => Some(time),
            Self::Unbounded => None,
        }
    }
}

impl ReplayHorizon {
    pub(crate) fn reached(self, now: u64) -> bool {
        self.bounded_time().is_some_and(|bound| now >= bound)
    }
}

impl ReplayHorizon {
    pub(crate) fn tighten(self, bound: u64) -> Self {
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

pub(crate) const SUB_CORES: usize = 4;

pub(crate) const DEP_BARRIERS: usize = 6;

#[derive(Debug, Clone, Copy)]
pub(crate) struct InstructionCacheEntry {
    pub(crate) key: InstructionLineKey,
    pub(crate) ready_at: u64,
    pub(crate) last_access: u64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct StreamBufferLine {
    pub(crate) key: InstructionLineKey,
    pub(crate) ready_at: u64,
    pub(crate) demanded: bool,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum InstructionL1RequestKind {
    Demand,
    Prefetch,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct BufferedInstruction {
    pub(crate) instruction: usize,
    pub(crate) ready_at: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MemoryOperandWidth {
    Bits32,
    Bits64,
    Bits128,
}

pub(crate) fn memory_operand_width(opcode: &str) -> MemoryOperandWidth {
    if opcode.split('.').any(|modifier| modifier == "128") {
        MemoryOperandWidth::Bits128
    } else if opcode.split('.').any(|modifier| modifier == "64") {
        MemoryOperandWidth::Bits64
    } else {
        MemoryOperandWidth::Bits32
    }
}

pub(crate) fn memory_address_is_uniform(opcode_text: &str) -> bool {
    opcode_text
        .split_once('[')
        .and_then(|(_, tail)| tail.split_once(']'))
        .is_some_and(|(address, _)| {
            address
                .split(|character: char| !character.is_ascii_alphanumeric())
                .any(|token| token.starts_with("UR"))
        })
}

pub(crate) fn is_u16_opcode(opcode_text: &str) -> bool {
    opcode_text
        .split_ascii_whitespace()
        .next()
        .is_some_and(|opcode| opcode.split('.').any(|modifier| modifier == "U16"))
}

pub(crate) fn deep_constant_load(
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

#[derive(Debug, Clone, Copy)]
pub(crate) struct MemoryPipelineSlot {
    pub(crate) operand_ready_latency: u64,
    pub(crate) constant_pressure: bool,
}

pub(crate) fn memory_pipeline_slot(
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
        constant_pressure: deep_constant,
    })
}

pub(crate) fn is_gpu_membar_issue(slot: &IssueSlot) -> bool {
    slot.is_gpu_membar()
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum DependencyOrigin {
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

pub(crate) fn readiness_cause_name(cause: ReadinessCause) -> &'static str {
    match cause {
        ReadinessCause::AsyncSource => "async_source",
        ReadinessCause::AsyncDestinationLocal => "async_destination_local",
        ReadinessCause::AsyncDestinationSmService => "async_destination_sm_service",
        _ => "other_readiness_cause",
    }
}

pub(crate) fn checked_usize_counter_add(
    counter: &'static str,
    current: usize,
    add: usize,
) -> Result<usize, IssueError> {
    current
        .checked_add(add)
        .ok_or(IssueError::CounterOverflow { counter })
}

/// Warp-local identity of one nonempty asynchronous-copy group. The numeric
/// sequence is intentionally opaque outside the SM actor; shared-memory replies
/// may route an identity emitted by the actor but cannot construct one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AsyncCopyGroupId(pub(crate) u64);

/// One source-bearing bypass LDGSTS instruction retained inside its
/// asynchronous-copy group. Topology-only zero-payload copies remain in the
/// group protocol but do not enter this source-response observation sidecar.
/// The production readiness path continues to use [`AsyncCopyGroup`]'s
/// established aggregate fields; this sidecar preserves the event ordering
/// needed to evaluate a future per-copy delivery model without changing time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BypassAsyncCopyObservation {
    pub(crate) group: AsyncCopyGroupId,
    pub(crate) instruction: usize,
    pub(crate) pending_sources: u32,
    pub(crate) source_completion_time: u64,
    pub(crate) delivery_work: u64,
    pub(crate) source_cohorts: u64,
    pub(crate) destination_lines: u64,
    pub(crate) max_bank_wavefronts: u64,
    pub(crate) max_bypass_wavefronts: u64,
    pub(crate) sector_off: usize,
    pub(crate) sector_membership_masks: Box<[u32]>,
    pub(crate) cohorts: Box<[BypassAsyncCopyCohortObservation]>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct BypassAsyncCopyProjection {
    pub(crate) copies: u64,
    pub(crate) bank_wavefront_upper_bound: u64,
    pub(crate) bypass_wavefront_upper_bound: u64,
    pub(crate) source_cohorts: u64,
    pub(crate) destination_lines: u64,
    pub(crate) max_bank_wavefronts_sum: u64,
    pub(crate) max_bypass_wavefronts_sum: u64,
    pub(crate) source_spread_cycles: u64,
    pub(crate) per_copy_completion_cycles: u64,
    pub(crate) group_aggregate_excess_cycles: u64,
    pub(crate) topology_candidate_penalty_cycles: u64,
    pub(crate) topology_candidate_completion_cycles: u64,
    pub(crate) topology_candidate_later_cycles: u64,
    pub(crate) topology_candidate_earlier_cycles: u64,
    pub(crate) pipeline_q1_completion_cycles: u64,
    pub(crate) pipeline_q1_exposure_cycles: u64,
    pub(crate) pipeline_q2_completion_cycles: u64,
    pub(crate) pipeline_q2_exposure_cycles: u64,
    pub(crate) pipeline_q4_completion_cycles: u64,
    pub(crate) pipeline_q4_exposure_cycles: u64,
    pub(crate) pipeline_q8_completion_cycles: u64,
    pub(crate) pipeline_q8_exposure_cycles: u64,
    pub(crate) pipeline_q16_completion_cycles: u64,
    pub(crate) pipeline_q16_exposure_cycles: u64,
    pub(crate) cohort_groups_observed: u64,
    pub(crate) cohort_copies_observed: u64,
    pub(crate) cohort_response_memberships_observed: u64,
    pub(crate) cohort_response_span_cycles: u64,
    pub(crate) cohort_topology_penalty_cycles: u64,
    pub(crate) cohort_candidate_completion_cycles: u64,
    pub(crate) cohort_candidate_exposure_cycles: u64,
    pub(crate) cohort_candidate_later_than_per_copy_cycles: u64,
    pub(crate) cohort_candidate_earlier_than_per_copy_cycles: u64,
    pub(crate) cohort_candidate_later_than_group_aggregate_cycles: u64,
    pub(crate) cohort_candidate_earlier_than_group_aggregate_cycles: u64,
    pub(crate) cohort_candidate_later_than_production_cycles: u64,
    pub(crate) cohort_candidate_earlier_than_production_cycles: u64,
    pub(crate) cohort_overlap_groups_observed: u64,
    pub(crate) cohort_response_envelope_cycles: u64,
    pub(crate) cohort_topology_prearrival_hidden_cycles: u64,
    pub(crate) cohort_topology_stream_hidden_cycles: u64,
    pub(crate) cohort_topology_residual_exposure_cycles: u64,
}

pub(crate) fn bypass_pipeline_candidate_completion(
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

#[derive(Debug, Clone, Default)]
pub(crate) struct Sm86RegisterReadPlan {
    pub(crate) modeled: bool,
    pub(crate) next_port_ready: [u64; 2],
    pub(crate) reads: [u64; 2],
    pub(crate) port_conflict_cycles: u64,
    pub(crate) rfc_hits: u64,
    pub(crate) rfc_misses: u64,
    pub(crate) rfc_consumes: Vec<Sm86RfcKey>,
    pub(crate) rfc_inserts: Vec<Sm86RfcKey>,
    pub(crate) unsupported_shape: bool,
}

pub(crate) fn issue_register_overlap(left: SassRegister, right: SassRegister) -> bool {
    if left.file() != right.file() {
        return false;
    }
    let (Some(left_index), Some(right_index)) = (left.index(), right.index()) else {
        return false;
    };
    u32::from(left_index) < u32::from(right_index) + u32::from(right.span_registers())
        && u32::from(right_index) < u32::from(left_index) + u32::from(left.span_registers())
}

pub(crate) fn issue_register_lane_mask(register: SassRegister, mask: u32) -> u32 {
    match register.file() {
        SassRegisterFile::Uniform | SassRegisterFile::UniformPredicate if mask != 0 => u32::MAX,
        _ => mask,
    }
}

pub(crate) fn issue_fixed_register_ready(
    status: &FixedLatencyOperandStatus,
    mask: u32,
    active_mask: u32,
    guard: Option<SassRegister>,
    entries: &[RegisterReadyEntry],
) -> FixedRegisterReady {
    let mut ready = FixedRegisterReady::default();
    if let Some(register) = guard {
        let lanes = issue_register_lane_mask(register, active_mask);
        for entry in entries {
            if lanes & entry.lane_mask != 0 && issue_register_overlap(register, entry.register) {
                ready.predicate = ready.predicate.max(entry.ready_at);
            }
        }
    }
    let FixedLatencyOperandStatus::Projected(projection) = status else {
        return ready;
    };
    for source in projection.sources() {
        let SassSourceOperand::Register(operand) = source else {
            continue;
        };
        let register = operand.register();
        let lanes = issue_register_lane_mask(register, mask);
        for entry in entries {
            if lanes & entry.lane_mask != 0 && issue_register_overlap(register, entry.register) {
                ready.raw = ready.raw.max(entry.ready_at);
            }
        }
    }
    for operand in projection.predicate_sources() {
        let register = operand.register();
        let lanes = issue_register_lane_mask(register, mask);
        for entry in entries {
            if lanes & entry.lane_mask != 0 && issue_register_overlap(register, entry.register) {
                ready.predicate = ready.predicate.max(entry.ready_at);
            }
        }
    }
    for operand in projection.destinations() {
        let register = operand.register();
        let lanes = issue_register_lane_mask(register, mask);
        for entry in entries {
            if lanes & entry.lane_mask != 0 && issue_register_overlap(register, entry.register) {
                ready.waw = ready.waw.max(entry.ready_at);
            }
        }
    }
    ready
}

pub(crate) fn issue_record_fixed_register_writes(
    entries: &mut Vec<RegisterReadyEntry>,
    status: &FixedLatencyOperandStatus,
    mask: u32,
    ready_at: u64,
) {
    let FixedLatencyOperandStatus::Projected(projection) = status else {
        return;
    };
    for register in projection
        .destinations()
        .iter()
        .flat_map(|dst| dst.register().components())
    {
        let lanes = issue_register_lane_mask(register, mask);
        for entry in entries
            .iter_mut()
            .filter(|entry| issue_register_overlap(register, entry.register))
        {
            entry.lane_mask &= !lanes;
        }
        entries.retain(|entry| entry.lane_mask != 0);
        if lanes != 0 {
            entries.push(RegisterReadyEntry {
                register,
                lane_mask: lanes,
                ready_at,
            });
        }
    }
}

pub(crate) struct BlockState {
    pub(crate) unfinished: usize,
    pub(crate) waiting: usize,
    pub(crate) warp_count: u32,
    pub(crate) shared_memory_bytes: u64,
    /// Physical slots held while this block is resident. The trace does not carry
    /// CTA lifetime, so modeled overlap may admit another block only when its slot
    /// set is disjoint from every currently resident block.
    pub(crate) warp_slots: Box<[i32]>,
    /// This block's warps occupy `warps[first_warp..first_warp +
    /// admitted_warps]` (admission pushes them contiguously), so barrier
    /// release scans only the block's own range.
    pub(crate) first_warp: usize,
    pub(crate) admitted_warps: usize,
    pub(crate) barrier_generation: u64,
    pub(crate) barrier_first_dispatch: Option<u64>,
    pub(crate) barrier_first_arrival: Option<u64>,
    /// Earliest issue time of this CTA's next barrier generation. It gates
    /// barrier instructions only; ordinary work after rendezvous is unaffected.
    pub(crate) barrier_rearm_ready: u64,
}

pub(crate) fn invalid_cohort_projection(
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

pub(crate) fn build_bypass_cohort_observation(
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

pub(crate) fn async_copy_source_count(kind: AccessKind) -> Option<u16> {
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

pub(crate) fn bypass_async_copy_observed_topology(
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

pub(crate) fn validate_async_copy_completion_work(
    kind: AccessKind,
    warp: usize,
) -> Result<(), IssueError> {
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

/// An MSHR-stalled remainder of one instruction's sector list, retried
/// (in FIFO order) whenever a fill completion frees an entry.
pub(crate) struct Retry {
    pub(crate) warp: usize,
    pub(crate) instruction: usize,
    pub(crate) origin: MemoryOrigin,
    /// Whether the stalled logical load uses the SM86 constant-cache return
    /// path. Retries retain the marker so a later fill cannot silently lose
    /// the response delay.
    pub(crate) constant_load: bool,
    /// Whether this constant load uses the measured multi-site operand path.
    /// This remains separate from the broad constant marker above.
    pub(crate) constant_deep: bool,
    pub(crate) client_scope: L2ClientScope,
    /// Completion identity of the stalled instruction. A present ordinary
    /// barrier holds one pending replacement unit until the entry drains;
    /// asynchronous-copy source units were counted at original issue and retain
    /// their group identity across every retry.
    pub(crate) target: Option<FillTarget>,
    /// Access-local finite-admission floor retained from original issue. Only
    /// cacheable LDGSTS retries carry one; ordinary hierarchy traffic uses `None`.
    pub(crate) mshr_floor: Option<NonZeroU32>,
    /// Original logical request boundary for pending-fill lookup attribution.
    pub(crate) lookup_batch: L1LookupBatch,
    /// Unresolved slice `warps[warp].sectors[lo..hi]`.
    pub(crate) lo: usize,
    pub(crate) hi: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct FillWaiter {
    pub(crate) warp: usize,
    pub(crate) target: FillTarget,
    pub(crate) units: u32,
    /// Whether this waiter consumes the measured deep constant-return path.
    /// The marker is per waiter because one folded physical fill can serve
    /// ordinary and constant loads with different return-service latencies.
    pub(crate) constant_deep: bool,
}

pub(crate) fn accumulate_waiter_removals(
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

pub(crate) fn checked_constant_return_time(
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

pub(crate) fn fill_waiter_ready_time(
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

pub(crate) fn project_async_copy_removal(
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PayloadStoreSite {
    pub launch_index: i64,
    pub warp_slot: i32,
    pub instruction_index: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PayloadStoreObservation {
    pub(crate) site: PayloadStoreSite,
    pub(crate) accepted_at: u64,
    pub(crate) resident_generation: Option<u32>,
    pub(crate) retained_dirty: bool,
    pub(crate) attached: bool,
}

/// Extract the constant-bank line keys used by an instruction's SASS text.
/// Immediate constants are operands of the form `c[bank][offset]` or
/// `cx[bank][offset]`; indexed `LDC`/`LDC.*` operations use a different cache
/// and are intentionally excluded here, while uniform `ULDC`/`ULDC.*`
/// operands remain on the IMC path as observed by NCU.  The parser is deliberately narrow:
/// malformed/non-numeric references are ignored rather than guessed, while
/// the trusted static sidecar remains the source of the instruction itself.
pub(crate) fn immediate_constant_lines(opcode_text: &str, line_bytes: u32) -> Box<[u64]> {
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

#[cfg(feature = "reference")]
pub(crate) fn reference_mode_from_env() -> Result<bool, IssueError> {
    match std::env::var_os("SIM_SM_REFERENCE_MODE") {
        None => Ok(false),
        Some(mode) if mode == "event" => Ok(false),
        Some(mode) if mode == "dense" => Ok(true),
        Some(mode) => Err(IssueError::ReferenceModeInvalid {
            mode: mode.to_string_lossy().into_owned(),
        }),
    }
}

/// Register a completion target that clears when `fill` completes. The
/// asynchronous-copy group already owns its pending source unit; an optional
/// ordinary barrier is counted here alongside ordinary load barriers.
pub(crate) fn add_fill_waiter(
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

pub(crate) fn preflight_line_fill_mask(
    st: &SharedL2State,
    sector_id: i64,
) -> Result<(), IssueError> {
    let line = sector_id >> 2;
    let fill_in_flight = st.l1.is_line_fill_in_flight(line);
    let has_mask = st.fill_masks.contains_key(&line);
    match (fill_in_flight, has_mask) {
        (true, false) => Err(IssueError::FillMaskMissing { line }),
        (false, true) => Err(IssueError::FillMaskWithoutFill { line }),
        _ => Ok(()),
    }
}

pub(crate) fn register_response(st: &mut SharedL2State, reply: ReplyTo) -> Result<(), IssueError> {
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

pub(crate) enum SharedL1Access {
    Hit { ready: Option<u64> },
    MissRequest { fill: FillKey },
    MergedPending { fill: FillKey },
    Stalled,
}

pub(crate) fn record_global_l1_observation(
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

pub(crate) fn fill_generation(fill: FillKey) -> u32 {
    match fill {
        FillKey::Line { generation, .. } | FillKey::Sector { generation, .. } => generation,
    }
}

pub(crate) fn access_shared_l1(
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

impl From<IssueError> for SharedAccessError {
    fn from(source: IssueError) -> Self {
        Self::Issue(source)
    }
}

pub(crate) fn drain_retries(
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
                r.client_scope,
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

pub(crate) fn validate_sector_span(
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

pub(crate) fn lookup_dense<'a>(
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
