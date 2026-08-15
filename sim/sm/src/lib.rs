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
//!
//! Tensor Core initiation contention is modeled per SMSP for the admitted
//! `sm_86` matrix-instruction forms. Other execution pipes remain explicit
//! dispatch-only fidelity boundaries (design §6.2).
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

pub mod control_bits;
#[cfg(feature = "campaign")]
pub mod evidence;
mod issue;
pub mod program;
pub mod replay;
mod system;

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
    AccessKind, IssueError, IssueModelParams, IssueResourceSource, MemoryOperation, ReplyTo,
    SmSimResult, TailFlags,
};
pub use program::{
    FunctionSchedule, IssueSlot, OpClass, PipelineClass, ProgramError, build_schedules,
};
pub use replay::{
    CoResidency, FullCounterEstimatorProvenance, FullCoverageSource, FullKernelReplayResult,
    FullLaunchCoverage, FullLaunchSetCounters, FullLaunchSetDeclaration,
    FullLaunchSetDeclarationError, FullLaunchTimingDiagnostic, KernelReplayOutcome,
    LaunchSetAggregationError, LaunchSetCounterComponent, LaunchSetScalarCounter, LaunchSetScope,
    LaunchSetScopeError, ReplayConfiguration, ReplayError, ReplayLaunchCoverageBinding,
    ReplayLaunchCoverageBindingError, ReplayMemMode, aggregate_full_launch_set,
    replay_trace_kernel,
};
#[cfg(feature = "campaign")]
pub use replay::{
    SampledKernelReplayResult, SampledLaunchCoverage, SampledObservedCounters, UnadmittedCoverage,
    UnadmittedReplayDiagnostics, replay_kernel, replay_unadmitted_diagnostics,
};
pub use sim_mem::MemRole;
pub use system::{
    DramBw, DramOriginStats, DramStats, L2OriginSourceStats, L2SourceSplit, SystemCounterError,
    SystemError,
};
