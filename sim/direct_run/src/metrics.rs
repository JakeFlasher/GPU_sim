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

include!("metrics/l1.rs");
include!("metrics/l2_dram.rs");
include!("metrics/pipeline_ipc.rs");
