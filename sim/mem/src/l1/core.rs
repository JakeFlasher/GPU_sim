//! L1 mutation facade for sector tags, fills, and flushes; mutates cache state at event timestamps.

use super::*;

#[path = "core_access.rs"]
mod core_access;
#[path = "core_flush.rs"]
mod core_flush;
pub use self::core_access::*;
pub(super) use self::core_flush::*;
