//! Trace → SM replay driver: flattens one traced kernel launch into per-SM
//! block replays using the REAL hardware placement recorded in the trace
//! (per-instruction `sm_id` / `warpid_sm`), joins each dynamic record to
//! the issue schedule (and, when an L1 setup is supplied, to the cache-op
//! manifest + per-record unique-sector expansion), and runs the issue
//! model on every SM.
//!
//! Placement policy v0: spatial placement is traced — each block runs on the
//! SM the hardware ran it on, and each warp uses its observed physical slot.
//! Traces without `sm_id` are a typed error (a synthetic placement model is a
//! future knob, not a silent fallback). Exact temporal block co-residency is not
//! present on the wire; launch-order overlap under the resident-warp and
//! architecture-card shared-memory ceilings plus disjoint traced physical slots is
//! explicitly typed as [`CoResidency::Modeled`].
//!
//! Memory-record classification (L1 mode): a global-class record takes
//! the [`MemRole`] its pc's SIDECAR OPCODE was classified to at join
//! time (`sim_mem::cache_op` port of the Python token rules) — eligible
//! load → modeled L1 sectors; bypass load → counted bypass; store head →
//! hit-tested without allocating; everything else → unified-denominator
//! count only. Sector expansion follows the `l1_trace_adapter` contract:
//! per-lane `addr>>5 ..= (addr+width-1)>>5`, deduplicated per INSTRUCTION
//! in first-seen order (active lanes are the non-sentinel slots — the
//! canonical positional guarantee).
//!
//! v0 limitation (fail-closed, not silent): every dynamic record of a
//! launch must carry the launch's own `function_unique_id` — traced device
//! function CALLs would need per-segment schedules and are rejected typed
//! until a trace exercises them.
//!
//! Memory discipline: threadblock payloads are decoded and semantically
//! validated once when `TraceDir` captures the immutable snapshot. The reader
//! retains compact scheduler/memory metadata plus a per-warp sector arena and
//! discards the address-heavy Prost tree. Extraction never performs a second
//! wire walk or lane-address reconstruction.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Serialize;
use sim_trace::async_sidecars::{
    LaunchObservations, ObservationKey as TmaObservationKey, TmaObservation,
};

use sim_arch::card::{ArchCard, CalibError};
use sim_arch::{ArchId, LdgstsDestinationWavefrontProjection, Sku};
use sim_mem::{L1Config, L1CounterError, L1Counters, SectorExpansionError};
use sim_trace::pb;
use sim_trace::reader::{
    EffectiveDataBytesError, LdgstsDestinationTopology, LdgstsSourceCohort, MemoryAccessMeta,
    MemoryPayloadContract, SharedBankProjectionError, ValidatedMemoryInstruction,
    ValidatedMemoryRoute, ValidatedMemoryRouteError, ValidatedSectorError,
    ValidatedSectorErrorPhase, ValidatedSectorPlan, ValidatedWarp,
};
use sim_trace::{KernelLoc, TraceDir, TraceError, TraceLineage, kernel_grid};
#[cfg(feature = "campaign")]
use sim_trace::{
    LosslessFullProvenance, TraceSamplingMethod, TraceSamplingSidecar, WaveSegmentSubsetProvenance,
};

use sim_mem::{
    ASYNC_COPY_COMPLETION_VALUE_MASK, ASYNC_COPY_COMPOSITE_PAYLOAD_WORK_MAX, CacheOpToken,
    CoherenceConfig, L2Config, L2Counters, L2Model, L2OpClass, LdgstsL1Policy, MemRole,
    encode_async_copy_destination_work, encode_async_copy_payload_and_topology,
    encode_async_copy_payload_work, is_known_untagged_opcode, opcode_head,
};

use crate::causal_frontier::{
    CausalFrontierInstruction, CausalFrontierReport, CausalFrontierResidency,
    summarize_with_residency,
};
#[cfg(feature = "campaign")]
use crate::evidence::AdmittedLaunchEvidence;
use crate::issue::Sm120AlphaConfig;
use crate::issue::{
    AccessKind, AsyncCopyCohortDescriptor, AsyncCopyCohortReplay, AsyncCopyCohortSpan,
    AsyncInstrumentationOrderDiagnostics, BlockReplay, InstrumentationOrderAnchor, IssueError,
    IssueModelParams, IssueResourceSource, MAX_ASYNC_DECISION_SNAPSHOTS, MemoryAdmissionCandidate,
    MemoryServiceCandidate, PayloadStoreSite, ReplayInst, SharedMemoryResidency, SmSimResult,
    TailFlags, WarpReplay, simulate_sm_with_residency, stateful_write_lookup_observer_enabled,
    validate_memory_admission_candidates, validate_memory_service_candidates,
};

use crate::metrics::TraceMetrics;

use crate::program::{
    FunctionSchedule, OpClass, PipelineClass, sm120_alpha_scalar_control_stall_suppressed,
};

use crate::sass_operands::{FixedLatencyOperandStatus, SassRegister, SassSourceOperand};

use crate::system::{
    DramBw, DramStats, L1PayloadAdmissionEvent, L2SourceSplit, SystemError,
    simulate_kernel_shared_with_residency_and_candidates_with_coherence,
    simulate_kernel_shared_with_residency_and_candidates_with_coherence_and_payload_events,
};

#[path = "replay/aggregation.rs"]
mod aggregation;
pub use aggregation::{
    FullCounterEstimatorProvenance, FullLaunchSetCounters, FullLaunchSetDeclaration,
    FullLaunchSetDeclarationError, FullLaunchTimingDiagnostic, LaunchSetAggregationError,
    LaunchSetCounterComponent, LaunchSetScalarCounter, LaunchSetScope, LaunchSetScopeError,
    ReplayLaunchCoverageBindingError, aggregate_full_launch_set,
};

#[path = "replay/coverage.rs"]
mod coverage;
#[path = "replay/fast_warp.rs"]
mod fast_warp;
#[path = "replay/placement.rs"]
mod placement;
#[path = "replay/results.rs"]
mod results;
#[path = "replay/session.rs"]
mod session;
pub use self::coverage::*;
pub(crate) use self::coverage::*;
pub use self::fast_warp::*;
pub(crate) use self::fast_warp::*;
pub use self::placement::*;
pub(crate) use self::placement::*;
pub use self::results::*;
pub(crate) use self::results::*;
pub use self::session::*;
pub(crate) use self::session::*;
