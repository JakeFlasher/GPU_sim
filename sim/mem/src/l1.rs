//! Sectored L1 cache core — a lockstep port of the cache state machine in
//! `the native L1 replay model` (`_access_once` +
//! `_apply_fills`), the model whose semantics the Phase-1 L1 predictor
//! validated. The Python module's warp-pool SCHEDULER is deliberately NOT
//! ported: it was a latency-proportional stand-in for warp interleaving,
//! and the sim's SM issue model provides the real interleaving (traced
//! placement + control-bits timing). This module is only the cache.
//!
//! Ported semantics (two-layer, 32 B sector / 128 B line):
//! - hit decision is sector-with-line-allocation: a hit needs the LINE
//!   resident in the (fully-associative, LRU) line pool AND the touched
//!   SECTOR already fetched into it;
//! - two distinguishable in-flight fill kinds, both consuming MSHR
//!   capacity: line-ALLOCATION fills keyed by line (any access to the line
//!   meanwhile MERGES by line — counts as a miss, ready at the fill, and
//!   the fill services every sector the merged accesses touched), and
//!   resident-line SECTOR fills keyed by (line, sector) (only the SAME
//!   missing sector merges; the line's other resident sectors still HIT);
//! - fills apply lazily in CHRONOLOGICAL order (line before sector only at
//!   exactly equal timestamps; same-kind ties in issue order): an in-flight
//!   line is not in the LRU until its fill time, and a sector fill whose
//!   line was evicted mid-flight is DROPPED (sector residency dies with
//!   the line);
//! - when every MSHR entry is busy the access STALLS and retries when the
//!   earliest fill (of either kind) frees an entry;
//! - the miss stream splits into `line_alloc_misses` (fresh allocation +
//!   the eviction it forces) and `sector_resident_misses` (partial-line
//!   sector fetch); invariant: `misses == line_alloc + sector_resident +
//!   merges`.
//!
//! v0 EXTENSIONS beyond the ported model (kept in separate counters so the
//! ported semantics stay auditable): Global/Local STORE and Global RED sectors
//! are hit-TESTED against the same state without allocating on miss (NCU's
//! unified `l1tex__t_sector_hit_rate.pct` covers write-class sectors; the L1 is
//! write-through no-allocate for hierarchy writes in this v0 hypothesis), and
//! bypass/untagged sector counting for the unified-denominator emission. RED
//! uses a distinct observer lane and never enters ordinary STG attribution.
//! L1TEX lookup accounting remains separate from modeled cache hit state: cache-all
//! LDGSTS merges into allocations from older warp instructions are observed as
//! lookup hits, while same-instruction sector coalescing remains a miss.

use serde::Serialize;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::num::NonZeroU32;

use sim_arch::{ArchId, L1WriteLookupWindow};
use sim_semantics::{
    L1BoundaryState, L1CounterState, L1FillDueState, L1LineFillState, L1SectorFillState,
    TYPED_STATE_SCHEMA_VERSION,
};

use crate::MemoryOrigin;
use crate::SectorPayloadV1;

#[path = "l1/accessors.rs"]
mod accessors;
#[path = "l1/cache_ops.rs"]
mod cache_ops;
#[path = "l1/cache_state.rs"]
mod cache_state;
#[path = "l1/config_counters.rs"]
mod config_counters;
pub use self::accessors::*;
pub use self::cache_state::*;
pub use self::config_counters::*;
