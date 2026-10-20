//! IPC, pipeline, and architecture-specific metric projections. Report-only.

use super::*;

#[path = "pipeline_core.rs"]
mod pipeline_core;
#[path = "pipeline_sm86.rs"]
mod pipeline_sm86;
#[path = "pipeline_tail.rs"]
mod pipeline_tail;
pub(crate) use self::pipeline_core::*;
pub(crate) use self::pipeline_sm86::*;
pub(crate) use self::pipeline_tail::*;
