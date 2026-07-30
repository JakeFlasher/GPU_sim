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
//! Pipe occupancy models land in later increments (design §6.2).
//!
//! Low-level shared-system results are intentionally not part of the public API:
//!
//! ```compile_fail
//! use sim_sm::SystemResult;
//! ```

pub mod control_bits;
pub mod evidence;
mod issue;
pub mod program;
pub mod replay;
mod system;

pub use control_bits::{
    BARRIER_NONE, ControlBits, ControlBitsError, ControlDomainError, ControlSidecar, DepBarrier,
    FunctionControl, InstructionControl, WaitMask,
};
pub use evidence::{
    AdmittedLaunchEvidence, BinaryDigestGap, LaunchEvidenceIdentity, ManifestError,
    admit_launch_evidence,
};
pub use issue::{
    AccessKind, IssueError, IssueModelParams, MemoryOperation, ReplyTo, SmSimResult, TailFlags,
};
pub use program::{FunctionSchedule, IssueSlot, OpClass, ProgramError, build_schedules};
pub use replay::{
    CoResidency, FullCounterEstimatorProvenance, FullCoverageSource, FullKernelReplayResult,
    FullLaunchCoverage, FullLaunchSetCounters, FullLaunchSetDeclaration,
    FullLaunchSetDeclarationError, FullLaunchTimingDiagnostic, KernelReplayOutcome,
    LaunchSetAggregationError, LaunchSetCounterComponent, LaunchSetScalarCounter, LaunchSetScope,
    LaunchSetScopeError, ReplayConfiguration, ReplayError, ReplayLaunchCoverageBinding,
    ReplayLaunchCoverageBindingError, ReplayMemMode, SampledKernelReplayResult,
    SampledLaunchCoverage, SampledObservedCounters, UnadmittedCoverage,
    UnadmittedReplayDiagnostics, aggregate_full_launch_set, replay_kernel,
    replay_unadmitted_diagnostics,
};
pub use sim_mem::MemRole;
pub use system::{
    DramBw, DramOriginStats, DramStats, L2OriginSourceStats, L2SourceSplit, SystemCounterError,
    SystemError,
};
