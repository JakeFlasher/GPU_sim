//! Semantic submodule; report-only boundary state and serde contracts.

use super::*;

#[path = "core_access.rs"]
mod core_access;
#[path = "core_flush.rs"]
mod core_flush;
pub use self::core_access::*;
pub(super) use self::core_flush::*;
