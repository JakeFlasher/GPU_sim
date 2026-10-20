//! SM86 pipeline-projection facade; report-only and timing/cache state is untouched.

use super::*;

#[path = "pipeline_sm86_a.rs"]
mod pipeline_sm86_a;
#[path = "pipeline_sm86_b.rs"]
mod pipeline_sm86_b;
#[path = "pipeline_sm86_c.rs"]
mod pipeline_sm86_c;
pub(super) use self::pipeline_sm86_a::*;
pub(super) use self::pipeline_sm86_b::*;
pub(super) use self::pipeline_sm86_c::*;
