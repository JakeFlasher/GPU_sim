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
//! - [`issue`] — the v0 SMSP issue model: 4 sub-cores, CGGTY-style pick,
//!   control-bits-driven stall/barrier timing, block-wide `BAR` sync;
//!   [`issue::SmCore`] is the resumable actor form for the event system.
//! - [`system`] — the kernel-level event system: SM actors over the
//!   SHARED L2 through the deterministic engine queue (the §6.3
//!   request/response inversion).
//! - [`replay`] — the trace→SM driver replaying a kernel launch on its
//!   traced hardware placement (`sm_id` / `warpid_sm`).
//!
//! Pipe occupancy models land in later increments (design §6.2).

pub mod control_bits;
pub mod issue;
pub mod program;
pub mod replay;
pub mod system;

pub use control_bits::{
    BARRIER_NONE, ControlBits, ControlBitsError, ControlSidecar, FunctionControl,
    InstructionControl,
};
pub use issue::{
    AccessKind, Advance, BlockReplay, IssueError, IssueModelParams, MemRequest, ReplayInst,
    ReplyTo, SmCore, SmMemMode, SmSimResult, WarpReplay, simulate_sm,
};
pub use program::{FunctionSchedule, IssueSlot, OpClass, ProgramError, build_schedules};
pub use replay::{KernelReplayResult, ReplayError, ReplayMemMode, replay_kernel};
pub use sim_mem::MemRole;
pub use system::{
    DramBw, DramStats, L2SourceSplit, SystemError, SystemResult, simulate_kernel_shared,
};
