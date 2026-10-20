//! SM86 tail and completion projections; report-only and timing-neutral.

use super::*;

#[path = "pipeline_sm86_c_a.rs"]
mod pipeline_sm86_c_a;
#[path = "pipeline_sm86_c_b.rs"]
mod pipeline_sm86_c_b;
pub(super) use self::pipeline_sm86_c_a::*;
pub(super) use self::pipeline_sm86_c_b::*;
