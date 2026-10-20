//! Direct composition of trace loading, scheduling, and GPU replay.
//!
//! The runner accepts ordinary input paths. It deliberately has no admission
//! manifest, artifact digest, receipt, publication, or approval layer.

pub mod emit;
#[path = "metrics.rs"]
mod metrics;
pub mod ncu;
#[path = "prepared.rs"]
mod prepared;
#[path = "resources.rs"]
mod resources;
pub use prepared::{
    CoverageAdmission, KernelKey, KernelSelector, PreparedLaunchIdentity, PreparedReplay,
};

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Instant;

use crate::emit::{
    EmitOutcome, EmittedMetricKey, FullLaunchPredictions, MetricValue, Unavailable, Unit,
    async_global_l1_hit_exclusion, emit_launch, global_read_resident_hit_adjustment,
    global_resident_hit_adjustment, l1_write_hit_terms,
};
use serde::{Deserialize, Serialize};
use sim_arch::ArchId;
use sim_arch::card::{ArchCard, CalibError};
use sim_arch::spec::{ArchitectureSpec, ResourceKind};
use sim_mem::{
    CoherenceConfig, L1Config, L1Counters, L1SetIndex, L2_WRITE_ALLOCATE_POLICY, L2Config,
    L2Counters, L2Replacement, L2Source, MemoryOrigin, NativePayloadRecordV1,
    PayloadForwardingLedgerV1, PchaseCard, SectorPayloadV1, SliceHash, coalesce_payloads,
    enrich_native_payload_records, join_native_payload_records,
    load_native_payload_sidecar_records,
};
use sim_semantics::{
    CacheCandidateSummary, CacheWitnessDomains, ClockDomain, CompletionRequirement,
    EndpointDescriptor, FrontierBound, FrontierVector, LatentFactors, LatentModelError,
    NcuObserverProjection, ResourceServiceProjection, ServiceComposition, ServiceOverlapMode,
    ServiceQueueProjection, ServiceRateProfile, ServiceStageRole,
};
use sim_sm::{
    ControlSidecar, DramBw, DramStats, FastFrontierSummary, FullKernelReplayResult,
    GpcObservationFrontier, IssueModelParams, L1PayloadAdmissionEvent, L2SourceSplit,
    MAX_MEMORY_ADMISSION_CANDIDATES, MAX_MEMORY_SERVICE_CANDIDATES, MemoryAdmissionCandidate,
    MemoryServiceCandidate, ModeledCompletionFrontier, ReadinessDiagnostics, ReplayMemMode,
    Sm120AlphaConfig, SmActiveObservationFrontier, StreamL2ReplaySession, TailFlags, TraceMetrics,
    build_schedules, replay_trace_kernel, replay_trace_kernel_with_l1_payload_events,
    replay_trace_kernel_with_memory_admission_candidates,
    replay_trace_kernel_with_memory_admission_feedback,
    replay_trace_kernel_with_memory_service_candidates,
};
use sim_trace::{AnchoredDir, ExecInfo, TraceDir, kernel_grid, strict_json_from_slice};

#[path = "reports.rs"]
mod reports;
#[path = "runner.rs"]
mod runner;
pub use self::reports::*;
pub(crate) use self::reports::*;
pub use self::runner::*;
pub(crate) use self::runner::*;
