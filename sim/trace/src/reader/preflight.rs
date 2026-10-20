//! Trace preflight facade; child modules validate payload, allocation, and semantics.

use super::*;

mod preflight_access;

mod preflight_allocation;

mod preflight_validation;
pub use self::preflight_access::*;
pub use self::preflight_allocation::*;
pub use self::preflight_validation::*;
