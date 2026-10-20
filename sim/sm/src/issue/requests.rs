//! Semantic submodule; report-only boundary state and serde contracts.

use super::*;

#[path = "requests_decode.rs"]
mod requests_decode;
#[path = "requests_validation.rs"]
mod requests_validation;
pub use self::requests_decode::*;
pub use self::requests_validation::*;
