//! Shared post-replay metric projection facade. Child modules are real Rust modules; all outputs are report-only.
//! Unit-explicit SM86 metrics derived after replay, never from NCU values.
//! Existing campaign emissions keep their compatibility contract. This surface
//! supplies additional ratios and explicitly distinguishes approximations/gaps.
use super::RunError;
use super::resources::LaunchResourceObservation;
use crate::emit::async_global_l1_hit_exclusion;
use serde::Serialize;
use sim_arch::ArchId;
use sim_arch::card::ArchCard;
use sim_mem::MemoryOrigin;
use sim_sm::{FullKernelReplayResult, TraceMetrics};
use sim_sm::{
    GlobalMemoryIssueDiagnostics, ReadinessCauseCycles, ReadinessDiagnostics, Sm120AlphaConfig,
};
use std::collections::BTreeMap;

#[path = "metrics/l1.rs"]
mod l1;
#[path = "metrics/l2_dram.rs"]
mod l2_dram;
#[path = "metrics/launch_occupancy.rs"]
mod launch_occupancy;
#[path = "metrics/pipeline_ipc.rs"]
mod pipeline_ipc;
pub use self::l1::*;
pub(crate) use self::l1::*;
pub use self::l2_dram::*;
pub(crate) use self::l2_dram::*;
pub use self::launch_occupancy::*;
pub(crate) use self::launch_occupancy::*;
pub use self::pipeline_ipc::*;
pub(crate) use self::pipeline_ipc::*;

pub(super) struct Sm86MetricState<'a> {
    pub(super) result: &'a FullKernelReplayResult,
    pub(super) external: Option<&'a LaunchResourceObservation>,
    pub(super) card: &'a ArchCard,
    pub(super) sm_count: u32,
    pub(super) active_sm_count: u32,
    pub(super) clock: Option<u64>,
    pub(super) metrics: BTreeMap<String, Metric>,
}
