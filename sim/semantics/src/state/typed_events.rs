//! Semantic submodule; report-only boundary state and serde contracts.

use super::*;

#[path = "typed_events_serialization.rs"]
mod typed_events_serialization;
#[path = "typed_events_validation.rs"]
mod typed_events_validation;
pub use self::typed_events_serialization::*;
pub use self::typed_events_validation::*;
