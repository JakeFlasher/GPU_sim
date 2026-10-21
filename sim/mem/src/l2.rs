//! Shared L2 slice model — a lockstep port of the sectored set-associative
//! replay core in `the native L2 replay model`
//! (`replay_access_stream` + `_install_way`), the engine behind the
//! Phase-1 L2 predictor. Like the L1 port, only the CACHE STATE MACHINE
//! is taken: the Python module replayed a timeless trace-order stream,
//! while the sim drives this model with the globally (time, seq)-ordered
//! request stream the event engine produces — the ordering the flat
//! replay stood in for.
//!
//! Ported semantics (32 B sectors in 128 B lines) — these bullets
//! describe the READ class, the rule the Python engine implements; the
//! §6.4 [`L2OpClass::WriteAtomic`] class additionally hits on line-TAG
//! residency alone (same state trajectory, different counting — see
//! `L2OpClass`; the parity corpus pins the Read class only):
//! - a Read-class sector access HITS iff its line tag is resident in
//!   the addressed set AND that sector's valid bit is set;
//! - a resident line whose requested sector is not yet valid is a
//!   sector-fill miss for reads. The immediate API sets the bit during the
//!   access to preserve Python parity; the event API keeps it pending until an
//!   explicit generation-bound completion. Writes validate immediately in both
//!   disciplines because they carry the sector data;
//! - a non-resident line allocates a way, evicting per policy when the
//!   set is full (LRU / SRRIP 2-bit RRPV / deterministic BIP 1-in-32);
//! - slice/set addressing: `slice = hash(line_id)`,
//!   `set = (line_id // num_slices) % num_sets_per_slice`, with the six
//!   pre-registered slice-hash candidates ported bit-exactly.
//!
//! Rule 1 note: the Phase-1 calibrator SELECTED a config for sm_86 by
//! fitting against local NCU labels (capacity_scale, hash, policy). That
//! selection is a label-fitted artifact and is deliberately NOT consumed
//! here. The sim's geometry comes from the arch card's vendor-documented
//! `l2_geometry` (capacity, line/sector, slice count); associativity,
//! slice hash and replacement policy resolve through the deployment
//! architecture's `sim-arch` capability. These remain documented hypotheses
//! to be tested by microbenchmarks—never by same-cell label fits—and an
//! architecture without a model receives a typed capability error.
//!
//! Fill timing: the immediate API remains an occupancy-only Python-parity
//! surface. A separate event discipline tracks pending sectors and makes them
//! valid only through opaque, allocation-generation-bound completion tokens.
//! Pending reads always remain lookup misses. The compatibility event API mints
//! a distinct fetch, while the coalesced event API splits newly fetched sectors
//! from allocation/sector-matched pending merges so the system can reuse the
//! original ready time. This is visibility/ABA state, not an invented finite L2
//! MSHR capacity.

use std::sync::atomic::{AtomicU64, Ordering};

use serde::Serialize;
use sim_arch::{ArchError, ArchId, CapabilityError, L2Replacement as ArchL2Replacement};
use sim_semantics::{
    AddressDomain, CanonicalLineKey, L2AccessDisciplineState, L2AllocationState, L2BoundaryState,
    L2CounterState, L2GeometryState, L2SectorOwnership, L2SectorStatus, L2SetState, L2StateError,
    L2WayState, TYPED_STATE_SCHEMA_VERSION,
};

#[path = "l2/config_counters.rs"]
mod config_counters;
#[path = "l2/core.rs"]
mod core;
pub use self::config_counters::*;
pub use self::core::*;
