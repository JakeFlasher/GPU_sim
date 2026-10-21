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

pub use sim_mem::AccessKind;

#[path = "issue/config.rs"]
mod config;
pub use config::{IssueModelParams, IssueResourceSource, Sm120AlphaConfig};

pub(crate) use config::{SharedMemoryResidency, checked_cycle_add};

#[path = "issue/memory_types.rs"]
mod memory_types;
pub use memory_types::{MemoryOperation, TailFlags};

#[path = "issue/readiness.rs"]
mod readiness;
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

#[path = "issue/candidates.rs"]
mod candidates;
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

#[path = "issue/async_copy.rs"]
mod async_copy;
#[path = "issue/core_init.rs"]
mod core_init;
#[path = "issue/core_mem.rs"]
mod core_mem;
#[path = "issue/core_state.rs"]
mod core_state;
#[path = "issue/diagnostics.rs"]
mod diagnostics;
#[path = "issue/frontend.rs"]
mod frontend;
#[path = "issue/pipelines.rs"]
mod pipelines;
#[path = "issue/request_types.rs"]
mod request_types;
#[path = "issue/schedule_barriers.rs"]
mod schedule_barriers;
#[path = "issue/schedule_control.rs"]
mod schedule_control;
#[path = "issue/schedule_frontend.rs"]
mod schedule_frontend;
#[path = "issue/warp_state.rs"]
mod warp_state;
#[allow(unused_imports)]
pub(crate) use self::async_copy::*;
pub use self::core_state::*;
#[allow(unused_imports)]
pub(crate) use self::core_state::*;
pub use self::diagnostics::*;
#[allow(unused_imports)]
pub(crate) use self::diagnostics::*;
#[allow(unused_imports)]
pub(crate) use self::frontend::*;
#[allow(unused_imports)]
pub(crate) use self::pipelines::*;
pub use self::request_types::*;
#[allow(unused_imports)]
pub(crate) use self::request_types::*;
#[allow(unused_imports)]
pub(crate) use self::warp_state::*;
