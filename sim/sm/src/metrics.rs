//! Timing-neutral native-replay observations. Counts are not profiler samples.
use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use sim_mem::{AccessKind, CacheOpToken, L1RequestMaskAudit, LdgstsL1Policy, opcode_head};
use sim_semantics::{
    CandidateDomain, EvidenceKind, LatentFactors, LatentModelError, ResourceDemandEnvelope,
    ResourceDemandInterval,
};

use crate::sass_operands::FixedLatencyOperandStatus;

#[cfg(test)]
#[path = "metrics/tests.rs"]
mod tests;

#[path = "metrics/locality_mrc.rs"]
mod locality_mrc;
#[path = "metrics/pipeline_activity.rs"]
mod pipeline_activity;
pub use self::locality_mrc::*;
pub(crate) use self::locality_mrc::*;
pub use self::pipeline_activity::*;
pub(crate) use self::pipeline_activity::*;
