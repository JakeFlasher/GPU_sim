//! Typed boundary state shared by the detailed and accelerated executors.
//!
//! These records are deliberately model-relative.  They describe the state
//! that the simulator can transform and compare; they do not claim that a
//! virtual trace address is a physical cache address or that activity counts
//! are measured watts.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::boundary::{BlockBoundaryState, WarpBoundaryState};
use crate::dram::{DramBoundaryState, DramStateError};

#[path = "state/cache_boundary.rs"]
mod cache_boundary;
#[path = "state/core_boundary.rs"]
mod core_boundary;
#[path = "state/typed_events.rs"]
mod typed_events;
pub use self::cache_boundary::*;
pub(crate) use self::cache_boundary::*;
pub use self::core_boundary::*;
pub(crate) use self::core_boundary::*;
pub use self::typed_events::*;
pub(crate) use self::typed_events::*;
