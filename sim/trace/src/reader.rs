//! Trace-directory reader for the NVBit tracer's on-disk layout:
//!
//! ```text
//! <trace_dir>/dynamic_trace.pb                    # dynamic_trace.Trace
//! <trace_dir>/threadblocks/device_<d>/stream_<s>/kernel_<k>/*.pb
//!                                                 # dynamic_trace.threadblock
//! <trace_dir>/extra_info/allocations.json         # allocation-identity registry
//! <trace_dir>/extra_info/enhanced_execution_info.json # static launch metadata
//! <trace_dir>/trace_sampling.json                 # sampling-provenance sidecar
//! ```
//!
//! [`TraceDir::open_direct`] reads ordinary replay inputs and ignores campaign
//! provenance. With the explicit `campaign` feature, [`TraceDir::open`] retains
//! the historical sealed-campaign behavior for the opt-in campaign skill. Both
//! loaders validate protobuf and simulator semantics before exposing replay
//! inputs.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use prost::Message;
use serde::de::{DeserializeSeed, Error as _, MapAccess, SeqAccess, Visitor};

#[cfg(feature = "campaign")]
use crate::anchored::FileIdentity;
use crate::anchored::{AnchoredDir, AnchoredFile, AnchoredReadError, FileGenerationIdentity};
#[cfg(feature = "campaign")]
use crate::digest::{Sha256, Sha256Digest, canonical_json};
use crate::exec_info::ExecInfo;
use crate::lanes::{self, LaneError};
use crate::pb;
use crate::wire_preflight::{self, WirePreflightError};

#[path = "reader/compact.rs"]
mod compact;
#[path = "reader/directory.rs"]
mod directory;
#[path = "reader/error.rs"]
mod error;
#[path = "reader/preflight_access.rs"]
mod preflight_access;
#[path = "reader/preflight_allocation.rs"]
mod preflight_allocation;
#[path = "reader/preflight_validation.rs"]
mod preflight_validation;
#[path = "reader/sampling.rs"]
mod sampling;
#[path = "reader/snapshot.rs"]
mod snapshot;
pub use self::compact::*;
pub use self::directory::*;
pub use self::error::*;
pub use self::preflight_access::*;
pub use self::preflight_allocation::*;
pub use self::preflight_validation::*;
pub use self::snapshot::*;
