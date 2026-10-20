//! L1 dirty-data and visibility flush facade; mutates cache state at completion boundaries.

use super::*;

#[path = "core_flush_a.rs"]
mod core_flush_a;
#[path = "core_flush_b.rs"]
mod core_flush_b;
pub(super) use self::core_flush_a::*;
pub(super) use self::core_flush_b::*;
