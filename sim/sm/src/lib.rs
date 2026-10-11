//! sim-sm: the SM (streaming multiprocessor) timing model.
//!
//! Phase-2 slices:
//! - [`control_bits`] — fail-closed ingestion of the Phase-1E per-function
//!   control-bits sidecars (stall counts, dependence barriers, yield,
//!   reuse), the issue model's dependence source instead of simulated
//!   scoreboards (design §6.2).
//! - [`program`] — the (function, pc) join of trace static metadata
//!   (`sim_trace::ExecInfo`) with the sidecar into replayable issue
//!   schedules.
//! - `issue` — the v0 SMSP issue model: 4 sub-cores, CGGTY-style pick,
//!   control-bits-driven stall/barrier timing, block-wide `BAR` sync;
//!   `issue::SmCore` is the resumable actor form for the event system.
//! - `system` — the kernel-level event system: SM actors over the
//!   SHARED L2 through the deterministic engine queue (the §6.3
//!   request/response inversion).
//! - [`replay`] — the trace→SM driver replaying a kernel launch on its
//!   traced hardware placement (`sm_id` / `warpid_sm`).
//! - [`causal_frontier`] — an opt-in descriptor-model executor over explicit
//!   dynamic descriptors; it can install a narrow macro transition, resumes
//!   detailed continuation, and refuses unknown dependency/completion state.
//!
//! Tensor Core initiation contention is modeled per SMSP for the admitted
//! `sm_86` matrix-instruction forms. CTA barrier instructions dispatch through
//! their SMSPs and feed one SM-shared arrival service at two cycles per warp;
//! own-arrival completion and other-participant rendezvous remain distinct.
//! The measured predicated forward-BRA form also carries an architecture-local
//! ten-cycle minimum next-issue frontier; uniform backward targets carry their
//! typed three-cycle resolution diagnostic. Address-carrying global atomics
//! retain alias-expanded sectors and exact L2 request topology. CLOCK-bracketed
//! probes expose exact first-to-last CLOCKLO issue intervals in readiness schema
//! 5. Schema 6 adds scalar-global instruction/sector counts, alias-aware ideal
//! L1TEX wavefronts, unique 128-byte lines, exact atomic requests, and issue
//! spans without changing scheduling. Schema 7 carries bounded opt-in streaming
//! service candidates; schema 8 distinguishes scalar-global operand-read
//! barrier producers and consumer delay. Predicated-off memory records remain
//! issued scheduler slots but create no traffic, writeback, or dependency
//! producer. Schema 9 evaluates bounded SM/per-SMSP operand-admission queues
//! and a shared FIFO service from exact per-instruction L2 hit/miss outcomes.
//! Schema 10 runs one opt-in feedback candidate that parks full-queue warps,
//! yields for exact L2 pricing, and reschedules subsequent issue/cache order.
//! Ordinary replay and schemas 1–9 remain unchanged.
//! Direct launch replay also has a separate opt-in dirty-sector writeback
//! service (`--dirty-writeback-capacity-sectors`); it shares the carded byte
//! server with read fetches but does not model coherence ownership.
//! For the exact sm_86 scalar STRONG.SM/HADD2 forms, an SMSP-local same-bank
//! writeback collision grants HADD2 and delays the barrier-observed load result
//! by one cycle; readiness schema 4 reports that event. Queue capacity and
//! other fixed/variable forms remain explicit fidelity boundaries.
//! Other execution pipes remain explicit dispatch-only fidelity boundaries
//! (design §6.2).
//!
//! Non-bulk asynchronous copies retain the source-cache policy encoded by
//! `sm_86` SASS: `LDGSTS.E` participates in the modeled L1/L2 hierarchy, while
//! an explicit `LDGSTS.E.BYPASS` uses the direct-L2 path. L1 hits, fills, and
//! MSHR retries preserve async-group completion identity so `DEPBAR.LE SB0, N`
//! observes the source completion. On finite caches, cacheable `LDGSTS.E` accesses
//! also use the independently bounded `sm_86` floor of 248 concurrent line fills;
//! ordinary hierarchy traffic retains the unresolved 32-entry working
//! approximation, and an unlimited ablation stays unlimited. In the shared L2,
//! matching pending `sm_86` TEX reads remain lookup misses but reuse the original
//! allocation-generation/sector fetch and ready time instead of issuing duplicate
//! DRAM traffic. No finite L2 MSHR depth is claimed. `sm_120` returns typed research
//! gaps rather than inheriting either Ampere semantic.
//!
//! Low-level shared-system results are intentionally not part of the public API:
//!
//! ```compile_fail
//! use sim_sm::SystemResult;
//! ```

pub mod causal_frontier;
pub mod control_bits;
pub mod custom;
#[cfg(feature = "campaign")]
pub mod evidence;
mod issue;
pub mod metrics;
pub mod program;
pub mod replay;
pub mod sass_operands;
mod system;

pub use causal_frontier::{
    CausalFrontierError, CausalFrontierEvidence, CausalFrontierInstruction, CausalFrontierReport,
    CausalFrontierResidency, CausalFrontierWitness, CausalFrontierWitnessOrigin, MacroEffect,
    MacroTransition, ModelDispatch, ModelExecutionSnapshot, ModelExecutor,
    summarize as summarize_causal_frontier,
    summarize_with_residency as summarize_causal_frontier_with_residency,
};
pub use control_bits::{
    BARRIER_NONE, ControlBits, ControlBitsError, ControlDomainError, ControlSidecar, DepBarrier,
    FunctionControl, InstructionControl, WaitMask,
};
#[cfg(feature = "campaign")]
pub use evidence::{
    AdmittedLaunchEvidence, BinaryDigestGap, CanonicalTraceBundlePath, LaunchEvidenceIdentity,
    ManifestArtifactToken, ManifestError, ReplayAuthenticatedLaunchIdentity,
    ThreadblockPayloadSetIdentity, TraceBundleComponentIdentity, TraceBundleComponents,
    TraceBundleIdentity, admit_launch_evidence, admit_launch_evidence_anchored,
};
pub use issue::{
    AccessKind, AsyncInstrumentationOrderDiagnostics, AsyncIssueOverlapDiagnostics,
    AsyncReadinessDiagnostics, BarrierPipelineDiagnostics, ClockIssueDiagnostics,
    ControlPipelineDiagnostics, CtaBarrierDiagnostics, GlobalMemoryIssueDiagnostics, IssueError,
    IssueModelParams, IssueResourceSource, Sm120AlphaConfig, MAX_MEMORY_ADMISSION_CANDIDATES,
    MAX_MEMORY_ADMISSION_QUEUE_CAPACITY, MAX_MEMORY_ADMISSION_SERVICE_WIDTH_SECTORS,
    MAX_MEMORY_SERVICE_CANDIDATES, MemoryAdmissionCandidate, MemoryAdmissionCandidateDiagnostics,
    MemoryAdmissionCandidateError, MemoryAdmissionCandidateResult,
    MemoryAdmissionFeedbackDiagnostics, MemoryOperation, MemoryServiceCandidate,
    MemoryServiceCandidateDiagnostics, MemoryServiceCandidateError,
    MemoryServiceCandidateOperation, MemoryServiceCandidateResult, MemoryServiceCandidateScope,
    MemoryWriteAckDiagnostics, PayloadStoreSite, ReadinessCauseCycles, ReadinessCounterError,
    ReadinessDiagnostics, ReplyTo, ResultWritebackDiagnostics, SmSimResult, TailFlags,
};
pub use metrics::{
    ClassCountInterval, ControlFlowGraph, InstructionActivity, MemoryTraffic, OperandBankPressure,
    PhysicalPipelineAttribution, PipelineClassEnvelope, SmActivity, TraceMetrics, WarpLifetime,
};
pub use program::{
    FunctionSchedule, IssueSlot, OpClass, PipelineClass, ProgramError, build_schedules,
    build_schedules_sm120_alpha, build_schedules_with_pipeline_arch,
};
pub use replay::{
    CoResidency, FastFrontierCoverage, FastFrontierSummary, FrontEndCounters,
    FullCounterEstimatorProvenance, FullCoverageSource, FullKernelReplayResult, FullLaunchCoverage,
    FullLaunchSetCounters, FullLaunchSetDeclaration, FullLaunchSetDeclarationError,
    FullLaunchTimingDiagnostic, GpcObservationFrontier, KernelReplayOutcome,
    LaunchSetAggregationError, LaunchSetCounterComponent, LaunchSetScalarCounter, LaunchSetScope,
    LaunchSetScopeError, ModeledCompletionFrontier, ReplayConfiguration, ReplayError,
    ReplayLaunchCoverageBinding, ReplayLaunchCoverageBindingError, ReplayMemMode,
    ReplaySessionError, SmActiveObservationFrontier, StreamL2ReplaySession,
    aggregate_full_launch_set, fast_frontier_for_trace, sm120_fast_stage_for_trace,
    replay_trace_kernel,
    replay_trace_kernel_with_l1_payload_events,
    replay_trace_kernel_with_memory_admission_candidates,
    replay_trace_kernel_with_memory_admission_feedback,
    replay_trace_kernel_with_memory_service_candidates,
};
#[cfg(feature = "campaign")]
pub use replay::{
    SampledKernelReplayResult, SampledLaunchCoverage, SampledObservedCounters, UnadmittedCoverage,
    UnadmittedReplayDiagnostics, replay_kernel, replay_unadmitted_diagnostics,
};
pub use sass_operands::{
    FixedLatencyFamily, FixedLatencyOpcode, FixedLatencyOperandGap, FixedLatencyOperandGapReason,
    FixedLatencyOperandProjection, FixedLatencyOperandStatus, SassRegister, SassRegisterFile,
    SassRegisterOperand, SassSourceOperand, project_fixed_latency_operands,
};
pub use sim_mem::MemRole;
pub use system::{
    DramBw, DramOriginStats, DramStats, L1PayloadAdmissionEvent, L2OriginSourceStats,
    L2SourceSplit, SystemCounterError, SystemError,
};
