//! Sectored L1 cache core — a lockstep port of the cache state machine in
//! `tools/cache_predictor/l1_warp_pool_walker.py` (`_access_once` +
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
//! ported semantics stay auditable): global-class STORE sectors are
//! hit-TESTED against the same state without allocating on miss (NCU's
//! unified `l1tex__t_sector_hit_rate.pct` counts store sectors; the L1 is
//! write-through no-allocate for global stores in this v0 hypothesis), and
//! bypass/untagged sector counting for the unified-denominator emission.

use std::collections::{BTreeMap, HashMap};

use sim_arch::ArchId;

use crate::MemoryOrigin;

pub const SECTOR_BYTES: u32 = 32;
pub const SECTORS_PER_LINE: u32 = 4;
pub const LINE_BYTES: u32 = SECTOR_BYTES * SECTORS_PER_LINE;

/// Configuration (card-resolved by `pchase_card`; latencies are integer
/// cycles — the caller ceils the card's fractional cycle values).
#[derive(Debug, Clone, Copy)]
pub struct L1Config {
    /// Deployment architecture whose geometry/latency hypotheses produced this config.
    pub arch: ArchId,
    pub line_capacity: u64,
    /// `None` = unlimited (ablation only; the card/documented default is
    /// always finite in production).
    pub mshr_count: Option<u32>,
    pub hit_latency: u64,
    pub miss_latency: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum L1ConfigError {
    #[error("L1 cache geometry is unsupported for deployment architecture {arch}")]
    ArchitectureUnsupported { arch: ArchId },
    #[error("line_capacity must be positive")]
    CapacityZero,
    #[error("mshr_count must be positive when finite")]
    MshrZero,
    #[error("hit latency {hit} must be strictly below miss latency {miss}")]
    LatencyOrderInverted { hit: u64, miss: u64 },
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum L1AccessError {
    #[error(transparent)]
    Counter(#[from] L1CounterError),
    #[error("L1 fill-timing mode is {latched}, but {requested} mode was requested")]
    FillTimingMismatch {
        latched: &'static str,
        requested: &'static str,
    },
    #[error("L1 fill {fill:?} is not in flight")]
    FillNotInFlight { fill: FillKey },
    #[error("L1 {sequence} sequence is exhausted at {current}")]
    SequenceOverflow {
        sequence: &'static str,
        current: u64,
    },
    #[error("L1 {operation} time addition {base} + {delta} overflows u64")]
    TimeOverflow {
        operation: &'static str,
        base: u64,
        delta: u64,
    },
    #[error("L1 stalled without an in-flight fill")]
    StallWithoutFill,
}

/// Conservative mutation budget used by callers to prove that a multi-access
/// batch cannot exhaust L1 counters or internal sequences before committing it.
/// The origin is mandatory so preflight covers the same accounting lane as the
/// later mutations; there is intentionally no implicit Global default.
#[derive(Debug, Clone, Copy)]
pub struct L1MutationBudget {
    pub origin: MemoryOrigin,
    pub eligible_sectors: u64,
    pub bypass_sectors: u64,
    pub bypass_atom_sectors: u64,
    pub untagged_sectors: u64,
    pub store_sectors: u64,
    pub recency_touches: u64,
    pub new_fills: u64,
}

impl L1MutationBudget {
    pub const fn for_origin(origin: MemoryOrigin) -> Self {
        Self {
            origin,
            eligible_sectors: 0,
            bypass_sectors: 0,
            bypass_atom_sectors: 0,
            untagged_sectors: 0,
            store_sectors: 0,
            recency_touches: 0,
            new_fills: 0,
        }
    }
}

/// Outcome of one eligible sector access (after any MSHR stall retries).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessOutcome {
    Hit,
    MissLineAlloc,
    MissSectorResident,
    Merge,
}

/// One origin's accounting lane. It mirrors the unified fields in
/// [`L1Counters`] but carries no architecture of its own; provenance remains
/// bound once on the enclosing physical-cache snapshot.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct L1OriginCounters {
    pub hits: u64,
    pub misses: u64,
    pub merges: u64,
    pub stall_events: u64,
    pub line_fills: u64,
    pub line_alloc_misses: u64,
    pub sector_resident_misses: u64,
    pub bypass_sectors: u64,
    pub bypass_atom_sectors: u64,
    pub untagged_sectors: u64,
    pub store_sectors: u64,
    pub store_hits: u64,
    pub store_tag_hits: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L1Counters {
    /// Architecture provenance bound by the owning [`L1Cache`]. `None` is
    /// reserved for synthetic/unbound snapshots and is rejected at emission
    /// or aggregation boundaries.
    arch: Option<ArchId>,
    /// Ported counters (modeled, manifest-eligible accesses only).
    pub hits: u64,
    pub misses: u64,
    pub merges: u64,
    pub stall_events: u64,
    pub line_fills: u64,
    pub line_alloc_misses: u64,
    pub sector_resident_misses: u64,
    /// Manifest-tagged ineligible sectors (the ported bypass counter).
    pub bypass_sectors: u64,
    /// Subset of `bypass_sectors` whose cache-op token is ATOM (the
    /// write-class bypass stream). Kept as a subset counter so the
    /// emission layer can fold the READ-class bypass stream (LDGSTS)
    /// into the global-load sector surface without pulling atomics in
    /// with it — NCU books those under op_atom/op_red, not op_ld.
    pub bypass_atom_sectors: u64,
    /// v0 extensions for the unified-denominator emission.
    pub untagged_sectors: u64,
    pub store_sectors: u64,
    pub store_hits: u64,
    /// §6.4 write-class rule at the L1 probe: store sectors whose LINE
    /// tag was resident regardless of the sector's valid bit (a
    /// superset of `store_hits`). Counted alongside — the sector-valid
    /// `store_hits` stays the v0 unified diagnostic until l1tex store
    /// accounting is scored against labels and one rule wins.
    pub store_tag_hits: u64,
    /// Accounting provenance only. Cache tags, fills, MSHRs, recency, and
    /// replacement remain unified physical state.
    by_origin: [L1OriginCounters; 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L1CounterField {
    Hits,
    Misses,
    Merges,
    StallEvents,
    LineFills,
    LineAllocMisses,
    SectorResidentMisses,
    BypassSectors,
    BypassAtomSectors,
    UntaggedSectors,
    StoreSectors,
    StoreHits,
    StoreTagHits,
}

impl std::fmt::Display for L1CounterField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Hits => "hits",
            Self::Misses => "misses",
            Self::Merges => "merges",
            Self::StallEvents => "stall_events",
            Self::LineFills => "line_fills",
            Self::LineAllocMisses => "line_alloc_misses",
            Self::SectorResidentMisses => "sector_resident_misses",
            Self::BypassSectors => "bypass_sectors",
            Self::BypassAtomSectors => "bypass_atom_sectors",
            Self::UntaggedSectors => "untagged_sectors",
            Self::StoreSectors => "store_sectors",
            Self::StoreHits => "store_hits",
            Self::StoreTagHits => "store_tag_hits",
        })
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum L1CounterError {
    #[error(transparent)]
    Architecture(#[from] sim_arch::ArchitectureProvenanceError),
    #[error("L1 counter '{field}' overflows u64 during aggregation")]
    Overflow { field: L1CounterField },
    #[error("L1 {origin:?} origin counter '{field}' overflows u64 during aggregation")]
    OriginOverflow {
        origin: MemoryOrigin,
        field: L1CounterField,
    },
    #[error("L1 origin partition for '{field}' overflows u64: Global={global}, Local={local}")]
    OriginPartitionOverflow {
        field: L1CounterField,
        global: u64,
        local: u64,
    },
    #[error(
        "L1 origin partition mismatch for '{field}': unified={unified}, Global={global}, Local={local}"
    )]
    OriginPartitionMismatch {
        field: L1CounterField,
        unified: u64,
        global: u64,
        local: u64,
    },
}

fn checked_counter_add(
    field: L1CounterField,
    left: u64,
    right: u64,
) -> Result<u64, L1CounterError> {
    left.checked_add(right)
        .ok_or(L1CounterError::Overflow { field })
}

fn origin_counter_mut(counters: &mut L1OriginCounters, field: L1CounterField) -> &mut u64 {
    match field {
        L1CounterField::Hits => &mut counters.hits,
        L1CounterField::Misses => &mut counters.misses,
        L1CounterField::Merges => &mut counters.merges,
        L1CounterField::StallEvents => &mut counters.stall_events,
        L1CounterField::LineFills => &mut counters.line_fills,
        L1CounterField::LineAllocMisses => &mut counters.line_alloc_misses,
        L1CounterField::SectorResidentMisses => &mut counters.sector_resident_misses,
        L1CounterField::BypassSectors => &mut counters.bypass_sectors,
        L1CounterField::BypassAtomSectors => &mut counters.bypass_atom_sectors,
        L1CounterField::UntaggedSectors => &mut counters.untagged_sectors,
        L1CounterField::StoreSectors => &mut counters.store_sectors,
        L1CounterField::StoreHits => &mut counters.store_hits,
        L1CounterField::StoreTagHits => &mut counters.store_tag_hits,
    }
}

fn add_origin_fields_checked(
    counters: &L1OriginCounters,
    origin: MemoryOrigin,
    deltas: &[(L1CounterField, u64)],
) -> Result<L1OriginCounters, L1CounterError> {
    let mut next = *counters;
    for &(field, delta) in deltas {
        let counter = origin_counter_mut(&mut next, field);
        *counter = counter
            .checked_add(delta)
            .ok_or(L1CounterError::OriginOverflow { origin, field })?;
    }
    Ok(next)
}

fn origin_deltas(counters: &L1OriginCounters) -> [(L1CounterField, u64); 13] {
    [
        (L1CounterField::Hits, counters.hits),
        (L1CounterField::Misses, counters.misses),
        (L1CounterField::Merges, counters.merges),
        (L1CounterField::StallEvents, counters.stall_events),
        (L1CounterField::LineFills, counters.line_fills),
        (L1CounterField::LineAllocMisses, counters.line_alloc_misses),
        (
            L1CounterField::SectorResidentMisses,
            counters.sector_resident_misses,
        ),
        (L1CounterField::BypassSectors, counters.bypass_sectors),
        (
            L1CounterField::BypassAtomSectors,
            counters.bypass_atom_sectors,
        ),
        (L1CounterField::UntaggedSectors, counters.untagged_sectors),
        (L1CounterField::StoreSectors, counters.store_sectors),
        (L1CounterField::StoreHits, counters.store_hits),
        (L1CounterField::StoreTagHits, counters.store_tag_hits),
    ]
}

fn unified_deltas(counters: &L1Counters) -> [(L1CounterField, u64); 13] {
    [
        (L1CounterField::Hits, counters.hits),
        (L1CounterField::Misses, counters.misses),
        (L1CounterField::Merges, counters.merges),
        (L1CounterField::StallEvents, counters.stall_events),
        (L1CounterField::LineFills, counters.line_fills),
        (L1CounterField::LineAllocMisses, counters.line_alloc_misses),
        (
            L1CounterField::SectorResidentMisses,
            counters.sector_resident_misses,
        ),
        (L1CounterField::BypassSectors, counters.bypass_sectors),
        (
            L1CounterField::BypassAtomSectors,
            counters.bypass_atom_sectors,
        ),
        (L1CounterField::UntaggedSectors, counters.untagged_sectors),
        (L1CounterField::StoreSectors, counters.store_sectors),
        (L1CounterField::StoreHits, counters.store_hits),
        (L1CounterField::StoreTagHits, counters.store_tag_hits),
    ]
}

impl Default for L1Counters {
    fn default() -> Self {
        Self {
            arch: None,
            hits: 0,
            misses: 0,
            merges: 0,
            stall_events: 0,
            line_fills: 0,
            line_alloc_misses: 0,
            sector_resident_misses: 0,
            bypass_sectors: 0,
            bypass_atom_sectors: 0,
            untagged_sectors: 0,
            store_sectors: 0,
            store_hits: 0,
            store_tag_hits: 0,
            by_origin: [L1OriginCounters::default(); 2],
        }
    }
}

impl L1Counters {
    pub fn for_arch(arch: ArchId) -> Self {
        Self {
            arch: Some(arch),
            ..Self::default()
        }
    }

    pub fn arch(&self) -> Option<ArchId> {
        self.arch
    }

    pub fn origin(&self, origin: MemoryOrigin) -> L1OriginCounters {
        self.by_origin[origin.index()]
    }

    /// Verify that every unified observer equals the checked sum of its two
    /// accounting lanes. This never examines or partitions physical cache state.
    pub fn validate_origin_partition(&self) -> Result<(), L1CounterError> {
        let global = self.origin(MemoryOrigin::Global);
        let local = self.origin(MemoryOrigin::Local);
        for ((field, unified), (_, global), (_, local)) in unified_deltas(self)
            .into_iter()
            .zip(origin_deltas(&global))
            .zip(origin_deltas(&local))
            .map(|((unified, global), local)| (unified, global, local))
        {
            let partition =
                global
                    .checked_add(local)
                    .ok_or(L1CounterError::OriginPartitionOverflow {
                        field,
                        global,
                        local,
                    })?;
            if unified != partition {
                return Err(L1CounterError::OriginPartitionMismatch {
                    field,
                    unified,
                    global,
                    local,
                });
            }
        }
        Ok(())
    }

    /// Build a coherent synthetic snapshot from explicit origin lanes.
    pub fn from_origins(
        arch: ArchId,
        global: L1OriginCounters,
        local: L1OriginCounters,
    ) -> Result<Self, L1CounterError> {
        let counters = Self::for_arch(arch)
            .add_fields_checked(MemoryOrigin::Global, &origin_deltas(&global))?;
        counters.add_fields_checked(MemoryOrigin::Local, &origin_deltas(&local))
    }

    /// Explicitly attribute an already-built unified snapshot to one origin.
    /// This is for synthetic composition/tests that construct raw counters;
    /// modeled cache traffic must use the origin-bearing mutation APIs instead.
    pub fn with_unified_origin(mut self, origin: MemoryOrigin) -> Self {
        let unified = L1OriginCounters {
            hits: self.hits,
            misses: self.misses,
            merges: self.merges,
            stall_events: self.stall_events,
            line_fills: self.line_fills,
            line_alloc_misses: self.line_alloc_misses,
            sector_resident_misses: self.sector_resident_misses,
            bypass_sectors: self.bypass_sectors,
            bypass_atom_sectors: self.bypass_atom_sectors,
            untagged_sectors: self.untagged_sectors,
            store_sectors: self.store_sectors,
            store_hits: self.store_hits,
            store_tag_hits: self.store_tag_hits,
        };
        self.by_origin = [L1OriginCounters::default(); 2];
        self.by_origin[origin.index()] = unified;
        self
    }

    fn add_fields_checked(
        &self,
        origin: MemoryOrigin,
        deltas: &[(L1CounterField, u64)],
    ) -> Result<Self, L1CounterError> {
        let mut next = *self;
        for &(field, delta) in deltas {
            let counter = match field {
                L1CounterField::Hits => &mut next.hits,
                L1CounterField::Misses => &mut next.misses,
                L1CounterField::Merges => &mut next.merges,
                L1CounterField::StallEvents => &mut next.stall_events,
                L1CounterField::LineFills => &mut next.line_fills,
                L1CounterField::LineAllocMisses => &mut next.line_alloc_misses,
                L1CounterField::SectorResidentMisses => &mut next.sector_resident_misses,
                L1CounterField::BypassSectors => &mut next.bypass_sectors,
                L1CounterField::BypassAtomSectors => &mut next.bypass_atom_sectors,
                L1CounterField::UntaggedSectors => &mut next.untagged_sectors,
                L1CounterField::StoreSectors => &mut next.store_sectors,
                L1CounterField::StoreHits => &mut next.store_hits,
                L1CounterField::StoreTagHits => &mut next.store_tag_hits,
            };
            *counter = checked_counter_add(field, *counter, delta)?;
        }
        next.by_origin[origin.index()] =
            add_origin_fields_checked(&self.by_origin[origin.index()], origin, deltas)?;
        Ok(next)
    }

    pub fn add_checked(&mut self, other: &L1Counters) -> Result<(), L1CounterError> {
        let expected = self
            .arch
            .ok_or(sim_arch::ArchitectureProvenanceError::Unbound {
                component: "L1 counters",
            })?;
        let got = other
            .arch
            .ok_or(sim_arch::ArchitectureProvenanceError::Unbound {
                component: "L1 counters",
            })?;
        if expected != got {
            return Err(sim_arch::ArchitectureProvenanceError::Mismatch {
                component: "L1 counters",
                expected,
                got,
            }
            .into());
        }

        let hits = checked_counter_add(L1CounterField::Hits, self.hits, other.hits)?;
        let misses = checked_counter_add(L1CounterField::Misses, self.misses, other.misses)?;
        let merges = checked_counter_add(L1CounterField::Merges, self.merges, other.merges)?;
        let stall_events = checked_counter_add(
            L1CounterField::StallEvents,
            self.stall_events,
            other.stall_events,
        )?;
        let line_fills =
            checked_counter_add(L1CounterField::LineFills, self.line_fills, other.line_fills)?;
        let line_alloc_misses = checked_counter_add(
            L1CounterField::LineAllocMisses,
            self.line_alloc_misses,
            other.line_alloc_misses,
        )?;
        let sector_resident_misses = checked_counter_add(
            L1CounterField::SectorResidentMisses,
            self.sector_resident_misses,
            other.sector_resident_misses,
        )?;
        let bypass_sectors = checked_counter_add(
            L1CounterField::BypassSectors,
            self.bypass_sectors,
            other.bypass_sectors,
        )?;
        let bypass_atom_sectors = checked_counter_add(
            L1CounterField::BypassAtomSectors,
            self.bypass_atom_sectors,
            other.bypass_atom_sectors,
        )?;
        let untagged_sectors = checked_counter_add(
            L1CounterField::UntaggedSectors,
            self.untagged_sectors,
            other.untagged_sectors,
        )?;
        let store_sectors = checked_counter_add(
            L1CounterField::StoreSectors,
            self.store_sectors,
            other.store_sectors,
        )?;
        let store_hits =
            checked_counter_add(L1CounterField::StoreHits, self.store_hits, other.store_hits)?;
        let store_tag_hits = checked_counter_add(
            L1CounterField::StoreTagHits,
            self.store_tag_hits,
            other.store_tag_hits,
        )?;
        let mut by_origin = self.by_origin;
        for origin in MemoryOrigin::ALL {
            by_origin[origin.index()] = add_origin_fields_checked(
                &self.by_origin[origin.index()],
                origin,
                &origin_deltas(&other.by_origin[origin.index()]),
            )?;
        }

        let next = Self {
            arch: self.arch,
            hits,
            misses,
            merges,
            stall_events,
            line_fills,
            line_alloc_misses,
            sector_resident_misses,
            bypass_sectors,
            bypass_atom_sectors,
            untagged_sectors,
            store_sectors,
            store_hits,
            store_tag_hits,
            by_origin,
        };
        // Arithmetic precedence stays unified first, then origin lanes. Only
        // after those operations fit do public aggregation boundaries reject a
        // malformed input partition (including two defects that would cancel).
        self.validate_origin_partition()?;
        other.validate_origin_partition()?;
        next.validate_origin_partition()?;
        *self = next;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NextSequence {
    Available(u64),
    Exhausted,
}

impl NextSequence {
    fn current(self) -> u64 {
        match self {
            Self::Available(current) => current,
            Self::Exhausted => u64::MAX,
        }
    }

    fn preflight(self, sequence: &'static str, uses: u64) -> Result<(), L1AccessError> {
        let Some(last_offset) = uses.checked_sub(1) else {
            return Ok(());
        };
        match self {
            Self::Available(current) if current.checked_add(last_offset).is_some() => Ok(()),
            Self::Available(current) => Err(L1AccessError::SequenceOverflow { sequence, current }),
            Self::Exhausted => Err(L1AccessError::SequenceOverflow {
                sequence,
                current: u64::MAX,
            }),
        }
    }

    fn allocate(self, sequence: &'static str) -> Result<(u64, Self), L1AccessError> {
        match self {
            Self::Available(current) => Ok((
                current,
                current
                    .checked_add(1)
                    .map_or(Self::Exhausted, Self::Available),
            )),
            Self::Exhausted => Err(L1AccessError::SequenceOverflow {
                sequence,
                current: u64::MAX,
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FillDue {
    /// Constant-latency fill due at this exact cycle. `u64::MAX` is valid.
    Timed(u64),
    /// Event-driven fill whose completion is owned by the next level.
    Untimed,
}

impl FillDue {
    fn timed(self) -> Option<u64> {
        match self {
            Self::Timed(time) => Some(time),
            Self::Untimed => None,
        }
    }

    fn timing(self) -> FillTiming {
        match self {
            Self::Timed(_) => FillTiming::Constant,
            Self::Untimed => FillTiming::Event,
        }
    }
}

enum SingleOutcome {
    Done(AccessOutcome, Option<u64>),
    Stall { retry_at: u64 },
}

/// The one shared classification result of [`L1Cache::access_core`];
/// each fill-timing mode maps it to its own public outcome type.
enum CoreAccess {
    Hit {
        ready: Option<u64>,
    },
    /// Merged into an in-flight line allocation. Constant mode carries the
    /// exact due cycle; event mode carries [`FillDue::Untimed`].
    LineMerge {
        pending: FillDue,
    },
    /// Merged into an in-flight same-sector fetch.
    SectorMerge {
        pending: FillDue,
    },
    Stalled,
    MissLineAlloc {
        pending: FillDue,
    },
    MissSectorResident {
        pending: FillDue,
    },
}

/// Which fill-timing discipline an [`L1Cache`] instance runs under —
/// latched on first use so the two families can never mix on one
/// instance (they share the in-flight entry maps).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FillTiming {
    /// Fill times known at issue (`now + miss_latency`), applied lazily.
    Constant,
    /// Fill times decided by the next level; entries remain explicitly
    /// untimed until [`L1Cache::complete_fill`].
    Event,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Readiness {
    Observed,
    Unobserved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FillRequest {
    Constant { now: u64, readiness: Readiness },
    Event { now: u64, readiness: Readiness },
}

impl FillRequest {
    fn timing(self) -> FillTiming {
        match self {
            Self::Constant { .. } => FillTiming::Constant,
            Self::Event { .. } => FillTiming::Event,
        }
    }

    fn hit_ready(self, hit_latency: u64) -> Result<Option<u64>, L1AccessError> {
        match self {
            Self::Constant {
                now,
                readiness: Readiness::Observed,
            } => L1Cache::checked_time_add("hit visibility", now, hit_latency).map(Some),
            Self::Event {
                now,
                readiness: Readiness::Observed,
            } => L1Cache::checked_time_add("event hit visibility", now, hit_latency).map(Some),
            Self::Constant {
                readiness: Readiness::Unobserved,
                ..
            }
            | Self::Event {
                readiness: Readiness::Unobserved,
                ..
            } => Ok(None),
        }
    }

    fn fill_due(self, miss_latency: u64) -> Result<FillDue, L1AccessError> {
        match self {
            Self::Constant { now, .. } => {
                L1Cache::checked_time_add("miss visibility", now, miss_latency).map(FillDue::Timed)
            }
            Self::Event { .. } => Ok(FillDue::Untimed),
        }
    }
}

/// Identifies one in-flight fill for the event-driven mode: the key the
/// next level's RESPONSE must complete. Line-allocation fills are keyed
/// by line; resident-sector fills by (line, sector) — the same two MSHR
/// kinds as the constant-latency mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FillKey {
    Line(i64),
    Sector(i64, u8),
}

/// Outcome of one eligible sector access in EVENT mode: fill times are
/// unknown at issue (the shared L2 decides them), so misses surface as
/// requests and the caller wires completions back via [`L1Cache::complete_fill`].
/// The fill request fetches the ACCESSED 32 B sector; sectors that merge
/// into an outstanding line allocation are delivered to the L2 exactly
/// once per fill BY THE CALLER (the §6.4 slice-2 machinery in sim-sm:
/// same-batch merges fold into the fill request's sector mask, later
/// merges send one follow-up fetch — the pre-slice-2 undercount is
/// FIXED there, not here). The ported L1 still installs the whole
/// merged mask at one fill event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvAccess {
    /// Sector hit: data ready at a known time.
    Hit { ready: u64 },
    /// Fresh miss: send one request to the next level; the response must
    /// call [`L1Cache::complete_fill`] with this key.
    MissRequest { fill: FillKey },
    /// Merged into an ALREADY outstanding fill: no new request; the
    /// merging warp waits on the same key's completion.
    MergedPending { fill: FillKey },
    /// Every MSHR entry is busy: the caller parks the access and retries
    /// after the next completion frees an entry (the event-mode analogue
    /// of the constant mode's retry-at-earliest-fill).
    Stalled,
}

/// Event-mode cache outcome when no dependence barrier consumes hit-ready time.
/// Cache state and fill ownership are identical to [`EvAccess`], but a hit does
/// not perform otherwise-unused timestamp arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvAccessUnobserved {
    Hit,
    MissRequest { fill: FillKey },
    MergedPending { fill: FillKey },
    Stalled,
}

#[derive(Clone)]
pub struct L1Cache {
    cfg: L1Config,
    /// Fully-associative LRU line pool: monotonic recency stamps.
    by_stamp: BTreeMap<u64, i64>,
    stamp_of: HashMap<i64, u64>,
    next_stamp: NextSequence,
    /// Per-line fetched-sector bitmask (bit = sector index in line, 0..4).
    /// Also accumulates for in-flight line allocations (the fill services
    /// every merged sector).
    sectors: HashMap<i64, u8>,
    /// In-flight fills: value = (due state, issue_seq) — the seq keeps
    /// same-timestamp same-kind timed fills in issue order like the Python
    /// stable sort. Event fills remain explicitly untimed.
    line_fills: HashMap<i64, (FillDue, u64)>,
    sector_fills: HashMap<(i64, u8), (FillDue, u64)>,
    fill_seq: NextSequence,
    /// Exact earliest timed in-flight fill. `None` means there is no timed
    /// fill (empty or event-driven), while `Some(u64::MAX)` is a real due
    /// cycle. `apply_fills` early-outs when `now` has not reached it.
    min_fill_due: Option<u64>,
    /// Latched fill-timing discipline (see [`FillTiming`]).
    fill_timing: Option<FillTiming>,
    pub counters: L1Counters,
}

impl L1Cache {
    pub fn new(cfg: L1Config) -> Result<Self, L1ConfigError> {
        if cfg.arch != ArchId::Sm86 {
            return Err(L1ConfigError::ArchitectureUnsupported { arch: cfg.arch });
        }
        if cfg.line_capacity == 0 {
            return Err(L1ConfigError::CapacityZero);
        }
        if cfg.mshr_count == Some(0) {
            return Err(L1ConfigError::MshrZero);
        }
        if cfg.hit_latency >= cfg.miss_latency {
            return Err(L1ConfigError::LatencyOrderInverted {
                hit: cfg.hit_latency,
                miss: cfg.miss_latency,
            });
        }
        let arch = cfg.arch;
        Ok(Self {
            cfg,
            by_stamp: BTreeMap::new(),
            stamp_of: HashMap::new(),
            next_stamp: NextSequence::Available(0),
            sectors: HashMap::new(),
            line_fills: HashMap::new(),
            sector_fills: HashMap::new(),
            fill_seq: NextSequence::Available(0),
            min_fill_due: None,
            fill_timing: None,
            counters: L1Counters::for_arch(arch),
        })
    }

    pub fn arch(&self) -> ArchId {
        self.cfg.arch
    }

    fn mshr_in_use(&self) -> usize {
        self.line_fills.len() + self.sector_fills.len()
    }

    /// Whether `fill` is still owned by this cache's event-mode MSHR state.
    /// Response handlers use this as a fail-closed preflight before consuming
    /// their exact response token or mutating any waiter accounting.
    pub fn is_fill_in_flight(&self, fill: FillKey) -> bool {
        match fill {
            FillKey::Line(line) => self.line_fills.contains_key(&line),
            FillKey::Sector(line, sector) => self.sector_fills.contains_key(&(line, sector)),
        }
    }

    fn timing_name(mode: FillTiming) -> &'static str {
        match mode {
            FillTiming::Constant => "constant",
            FillTiming::Event => "event",
        }
    }

    fn preflight_fill_timing(stored: FillDue, requested: FillTiming) -> Result<(), L1AccessError> {
        let latched = stored.timing();
        if latched != requested {
            return Err(L1AccessError::FillTimingMismatch {
                latched: Self::timing_name(latched),
                requested: Self::timing_name(requested),
            });
        }
        Ok(())
    }

    fn preflight_latch(&self, requested: FillTiming) -> Result<(), L1AccessError> {
        if let Some(latched) = self.fill_timing
            && latched != requested
        {
            return Err(L1AccessError::FillTimingMismatch {
                latched: Self::timing_name(latched),
                requested: Self::timing_name(requested),
            });
        }
        Ok(())
    }

    fn commit_latch(&mut self, mode: FillTiming) {
        if self.fill_timing.is_none() {
            self.fill_timing = Some(mode);
        }
    }

    fn checked_time_add(
        operation: &'static str,
        base: u64,
        delta: u64,
    ) -> Result<u64, L1AccessError> {
        base.checked_add(delta).ok_or(L1AccessError::TimeOverflow {
            operation,
            base,
            delta,
        })
    }

    fn apply_counter_deltas(
        &mut self,
        origin: MemoryOrigin,
        deltas: &[(L1CounterField, u64)],
    ) -> Result<(), L1AccessError> {
        self.counters = self.counters.add_fields_checked(origin, deltas)?;
        Ok(())
    }

    /// Prove a conservative multi-access mutation budget without changing
    /// cache state. Callers can fall back to exact shadow replay only when a
    /// near-limit budget is too conservative.
    pub fn preflight_mutation_budget(&self, budget: L1MutationBudget) -> Result<(), L1AccessError> {
        self.counters.add_fields_checked(
            budget.origin,
            &[
                (L1CounterField::Hits, budget.eligible_sectors),
                (L1CounterField::Misses, budget.eligible_sectors),
                (L1CounterField::Merges, budget.eligible_sectors),
                (L1CounterField::StallEvents, budget.eligible_sectors),
                (L1CounterField::LineFills, budget.eligible_sectors),
                (L1CounterField::LineAllocMisses, budget.eligible_sectors),
                (
                    L1CounterField::SectorResidentMisses,
                    budget.eligible_sectors,
                ),
                (L1CounterField::BypassSectors, budget.bypass_sectors),
                (
                    L1CounterField::BypassAtomSectors,
                    budget.bypass_atom_sectors,
                ),
                (L1CounterField::UntaggedSectors, budget.untagged_sectors),
                (L1CounterField::StoreSectors, budget.store_sectors),
                (L1CounterField::StoreHits, budget.store_sectors),
                (L1CounterField::StoreTagHits, budget.store_sectors),
            ],
        )?;
        let in_flight =
            u64::try_from(self.mshr_in_use()).map_err(|_| L1AccessError::SequenceOverflow {
                sequence: "recency",
                current: self.next_stamp.current(),
            })?;
        let recency_touches = budget.recency_touches.checked_add(in_flight).ok_or(
            L1AccessError::SequenceOverflow {
                sequence: "recency",
                current: self.next_stamp.current(),
            },
        )?;
        self.next_stamp.preflight("recency", recency_touches)?;
        self.fill_seq.preflight("fill", budget.new_fills)?;
        Ok(())
    }

    fn earliest_fill(&self) -> Result<u64, L1AccessError> {
        self.line_fills
            .values()
            .chain(self.sector_fills.values())
            .filter_map(|&(due, _)| due.timed())
            .min()
            .ok_or(L1AccessError::StallWithoutFill)
    }

    /// Conservative proof that a constant-mode access and a possible one-entry
    /// MSHR retry have enough timestamp headroom. The cached minimum is the
    /// common path; when fills are already due, find the first still-future fill
    /// that a full MSHR set could retry at. Callers batching several accesses use
    /// a shadow replay when this proof is too conservative.
    pub fn preflight_constant_access_timing(&self, now: u64) -> Result<(), L1AccessError> {
        self.preflight_latch(FillTiming::Constant)?;
        Self::checked_time_add("hit visibility", now, self.cfg.hit_latency)?;
        Self::checked_time_add("miss visibility", now, self.cfg.miss_latency)?;
        let retry_at = if self.mshr_in_use() == 0 {
            None
        } else if let Some(minimum) = self.min_fill_due.filter(|&due| due > now) {
            Some(minimum)
        } else {
            self.line_fills
                .values()
                .chain(self.sector_fills.values())
                .filter_map(|&(due, _)| due.timed())
                .filter(|&time| time > now)
                .min()
        };
        if let Some(retry_at) = retry_at {
            Self::checked_time_add("hit visibility", retry_at, self.cfg.hit_latency)?;
            Self::checked_time_add("miss visibility", retry_at, self.cfg.miss_latency)?;
        }
        Ok(())
    }

    /// Conservative, allocation-free headroom proof for an ordered batch of
    /// constant-mode sector accesses. Starting after the latest existing fill
    /// and charging one miss arm per selected sector covers fills created by an
    /// earlier access in the same caller transaction. `false` asks the caller to
    /// classify the batch exactly on a shadow cache.
    pub fn constant_batch_timing_has_headroom(&self, now: u64, sectors: u64) -> bool {
        if self.preflight_latch(FillTiming::Constant).is_err()
            || Self::checked_time_add("hit visibility", now, self.cfg.hit_latency).is_err()
            || Self::checked_time_add("miss visibility", now, self.cfg.miss_latency).is_err()
        {
            return false;
        }
        if sectors == 0 {
            return true;
        }
        let start = self
            .line_fills
            .values()
            .chain(self.sector_fills.values())
            .filter_map(|&(due, _)| due.timed())
            .max()
            .unwrap_or(now)
            .max(now);
        self.cfg
            .miss_latency
            .checked_mul(sectors)
            .and_then(|delta| start.checked_add(delta))
            .is_some()
    }

    fn touch_mru(&mut self, line: i64) -> Result<(), L1AccessError> {
        let (stamp, next_stamp) = self.next_stamp.allocate("recency")?;
        if let Some(old) = self.stamp_of.remove(&line) {
            self.by_stamp.remove(&old);
        }
        self.next_stamp = next_stamp;
        self.by_stamp.insert(stamp, line);
        self.stamp_of.insert(line, stamp);
        Ok(())
    }

    fn evict_beyond_capacity(&mut self) {
        while self.by_stamp.len() as u64 > self.cfg.line_capacity {
            let Some((&stamp, &line)) = self.by_stamp.first_key_value() else {
                break;
            };
            self.by_stamp.remove(&stamp);
            self.stamp_of.remove(&line);
            // Partial-line eviction: sector residency dies with the line.
            self.sectors.remove(&line);
        }
    }

    fn line_resident(&self, line: i64) -> bool {
        self.stamp_of.contains_key(&line)
    }

    fn commit_due_fills(&mut self, due: Vec<(u64, u8, u64, i64, u8)>) -> Result<(), L1AccessError> {
        for (_, kind, _, line, sector) in due {
            if kind == 0 {
                self.line_fills.remove(&line);
                self.touch_mru(line)?;
                self.evict_beyond_capacity();
            } else {
                self.sector_fills.remove(&(line, sector));
                if self.line_resident(line) {
                    // The sector arrives into its (still-resident) line
                    // and re-references it at MRU.
                    *self.sectors.entry(line).or_default() |= 1 << sector;
                    self.touch_mru(line)?;
                }
                // else: the line was evicted before this fill's own
                // timestamp — the fill is DROPPED.
            }
        }
        self.min_fill_due = self
            .line_fills
            .values()
            .chain(self.sector_fills.values())
            .filter_map(|&(due, _)| due.timed())
            .min();
        Ok(())
    }

    /// Replay due fills in CHRONOLOGICAL order; each fill's effect is
    /// decided by the cache state AT ITS OWN FILL TIME (line-before-sector
    /// only at exactly equal timestamps; same-kind ties in issue order).
    ///
    /// Integer-time note (deliberate divergence from the float-timed
    /// Python walker): the sim's clock is integer cycles, so fills whose
    /// float times would differ inside one cycle land on the same
    /// integer timestamp and resolve by the (kind, issue-order) tie rule
    /// instead of sub-cycle chronology. Bit-parity with Python holds
    /// whenever both sides run integer latencies (proven by the gate's
    /// differential); fractional-latency sub-cycle ordering is below the
    /// engine's resolution by design.
    fn apply_fills(&mut self, now: u64) -> Result<(), L1AccessError> {
        let Some(minimum) = self.min_fill_due else {
            return Ok(());
        };
        if now < minimum {
            return Ok(());
        }
        let mut due: Vec<(u64, u8, u64, i64, u8)> = Vec::new();
        for (&line, &(fill_due, seq)) in &self.line_fills {
            if let FillDue::Timed(time) = fill_due
                && time <= now
            {
                due.push((time, 0, seq, line, 0));
            }
        }
        for (&(line, sector), &(fill_due, seq)) in &self.sector_fills {
            if let FillDue::Timed(time) = fill_due
                && time <= now
            {
                due.push((time, 1, seq, line, sector));
            }
        }
        due.sort_unstable();
        let due_count = u64::try_from(due.len()).map_err(|_| L1AccessError::SequenceOverflow {
            sequence: "recency",
            current: self.next_stamp.current(),
        })?;
        if self.next_stamp.preflight("recency", due_count).is_err() {
            // Some due sector fills can be dropped after an earlier fill evicts
            // their line. Near exhaustion, classify those touches exactly on a
            // shadow cache and publish the whole transition only on success.
            let mut shadow = self.clone();
            shadow.commit_due_fills(due)?;
            *self = shadow;
            return Ok(());
        }
        self.commit_due_fills(due)
    }

    /// One classification/state step shared by BOTH fill-timing modes —
    /// the single copy of the ported ladder (line-merge → sector-present
    /// hit → sector-merge → MSHR stall → fresh miss), with all counter
    /// accounting inside. [`FillRequest`] computes a hit timestamp only on a
    /// hit and a fill due state only on a fresh miss, so an unused timing arm
    /// cannot reject a valid merge, stall, or exact-domain result.
    fn access_core(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        fill_request: FillRequest,
    ) -> Result<CoreAccess, L1AccessError> {
        let line = sector_id >> 2;
        let sector = (sector_id & 3) as u8;
        if let Some(&(pending, _)) = self.line_fills.get(&line) {
            Self::preflight_fill_timing(pending, fill_request.timing())?;
            let counters = self.counters.add_fields_checked(
                origin,
                &[(L1CounterField::Merges, 1), (L1CounterField::Misses, 1)],
            )?;
            self.counters = counters;
            *self.sectors.entry(line).or_default() |= 1 << sector;
            return Ok(CoreAccess::LineMerge { pending });
        }
        let line_resident = self.line_resident(line);
        // Sector-present HIT testing precedes sector-merge handling.
        let sector_present = line_resident
            && self
                .sectors
                .get(&line)
                .is_some_and(|mask| mask & (1 << sector) != 0);
        if sector_present {
            let ready = fill_request.hit_ready(self.cfg.hit_latency)?;
            self.next_stamp.preflight("recency", 1)?;
            let counters = self
                .counters
                .add_fields_checked(origin, &[(L1CounterField::Hits, 1)])?;
            self.counters = counters;
            self.touch_mru(line)?;
            return Ok(CoreAccess::Hit { ready });
        }
        if line_resident && let Some(&(pending, _)) = self.sector_fills.get(&(line, sector)) {
            Self::preflight_fill_timing(pending, fill_request.timing())?;
            let counters = self.counters.add_fields_checked(
                origin,
                &[(L1CounterField::Merges, 1), (L1CounterField::Misses, 1)],
            )?;
            self.counters = counters;
            return Ok(CoreAccess::SectorMerge { pending });
        }
        // Miss path: needs an MSHR entry (both fill kinds consume capacity).
        if let Some(cap) = self.cfg.mshr_count
            && self.mshr_in_use() >= cap as usize
        {
            self.apply_counter_deltas(origin, &[(L1CounterField::StallEvents, 1)])?;
            return Ok(CoreAccess::Stalled);
        }
        let fill_due = fill_request.fill_due(self.cfg.miss_latency)?;
        let (seq, next_fill_seq) = self.fill_seq.allocate("fill")?;
        let counters = if line_resident {
            self.counters.add_fields_checked(
                origin,
                &[
                    (L1CounterField::Misses, 1),
                    (L1CounterField::SectorResidentMisses, 1),
                ],
            )?
        } else {
            self.counters.add_fields_checked(
                origin,
                &[
                    (L1CounterField::Misses, 1),
                    (L1CounterField::LineAllocMisses, 1),
                    (L1CounterField::LineFills, 1),
                ],
            )?
        };
        self.fill_seq = next_fill_seq;
        self.counters = counters;
        if let FillDue::Timed(time) = fill_due {
            self.min_fill_due = Some(self.min_fill_due.map_or(time, |due| due.min(time)));
        }
        if line_resident {
            // Partial-line sector fetch: no allocation, no eviction; the
            // sector becomes resident at its FILL time (lazy).
            self.sector_fills.insert((line, sector), (fill_due, seq));
            Ok(CoreAccess::MissSectorResident { pending: fill_due })
        } else {
            self.line_fills.insert(line, (fill_due, seq));
            *self.sectors.entry(line).or_default() |= 1 << sector;
            Ok(CoreAccess::MissLineAlloc { pending: fill_due })
        }
    }

    fn access_single(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
        readiness: Readiness,
    ) -> Result<SingleOutcome, L1AccessError> {
        self.apply_fills(now)?;
        match self.access_core(origin, sector_id, FillRequest::Constant { now, readiness })? {
            CoreAccess::Hit { ready } => Ok(SingleOutcome::Done(AccessOutcome::Hit, ready)),
            CoreAccess::LineMerge { pending } | CoreAccess::SectorMerge { pending } => {
                let pending = pending.timed().ok_or(L1AccessError::FillTimingMismatch {
                    latched: Self::timing_name(FillTiming::Event),
                    requested: Self::timing_name(FillTiming::Constant),
                })?;
                Ok(SingleOutcome::Done(AccessOutcome::Merge, Some(pending)))
            }
            CoreAccess::Stalled => Ok(SingleOutcome::Stall {
                retry_at: self.earliest_fill()?,
            }),
            CoreAccess::MissSectorResident { pending } => {
                let ready = pending.timed().ok_or(L1AccessError::FillTimingMismatch {
                    latched: Self::timing_name(FillTiming::Event),
                    requested: Self::timing_name(FillTiming::Constant),
                })?;
                Ok(SingleOutcome::Done(
                    AccessOutcome::MissSectorResident,
                    Some(ready),
                ))
            }
            CoreAccess::MissLineAlloc { pending } => {
                let ready = pending.timed().ok_or(L1AccessError::FillTimingMismatch {
                    latched: Self::timing_name(FillTiming::Event),
                    requested: Self::timing_name(FillTiming::Constant),
                })?;
                Ok(SingleOutcome::Done(
                    AccessOutcome::MissLineAlloc,
                    Some(ready),
                ))
            }
        }
    }

    fn access_eligible_constant_impl(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
    ) -> Result<(AccessOutcome, u64), L1AccessError> {
        self.commit_latch(FillTiming::Constant);
        let mut t = now;
        loop {
            match self.access_single(origin, sector_id, t, Readiness::Observed)? {
                SingleOutcome::Done(outcome, Some(ready)) => return Ok((outcome, ready)),
                SingleOutcome::Done(_, None) => {
                    unreachable!("observed constant access omitted its ready time")
                }
                SingleOutcome::Stall { retry_at } => {
                    t = retry_at.max(Self::checked_time_add("stall retry", t, 1)?);
                }
            }
        }
    }

    fn access_eligible_constant_unobserved_impl(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
    ) -> Result<AccessOutcome, L1AccessError> {
        self.commit_latch(FillTiming::Constant);
        let mut t = now;
        loop {
            match self.access_single(origin, sector_id, t, Readiness::Unobserved)? {
                SingleOutcome::Done(outcome, _) => return Ok(outcome),
                SingleOutcome::Stall { retry_at } => {
                    t = retry_at.max(Self::checked_time_add("stall retry", t, 1)?);
                }
            }
        }
    }

    /// One modeled (manifest-eligible) sector access at time `now`.
    /// MSHR stalls retry internally at the earliest fill time (the ported
    /// walker's retry contract). Returns the outcome and the data-ready
    /// time.
    pub fn access_eligible(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
    ) -> Result<(AccessOutcome, u64), L1AccessError> {
        self.preflight_latch(FillTiming::Constant)?;
        let budget = L1MutationBudget {
            eligible_sectors: 1,
            recency_touches: 1,
            new_fills: 1,
            ..L1MutationBudget::for_origin(origin)
        };
        let fast_path_safe = self.preflight_mutation_budget(budget).is_ok()
            && self.preflight_constant_access_timing(now).is_ok();
        if !fast_path_safe {
            let mut shadow = self.clone();
            let outcome = shadow.access_eligible_constant_impl(origin, sector_id, now)?;
            *self = shadow;
            return Ok(outcome);
        }
        self.access_eligible_constant_impl(origin, sector_id, now)
    }

    /// Mutate constant-mode cache state for one eligible sector without
    /// observing producer-ready time. Fresh misses still compute their fill due
    /// time; hits and merges skip visibility arithmetic that no dependence
    /// barrier consumes.
    pub fn access_eligible_unobserved(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
    ) -> Result<AccessOutcome, L1AccessError> {
        self.preflight_latch(FillTiming::Constant)?;
        let budget = L1MutationBudget {
            eligible_sectors: 1,
            recency_touches: 1,
            new_fills: 1,
            ..L1MutationBudget::for_origin(origin)
        };
        let fast_path_safe = self.preflight_mutation_budget(budget).is_ok()
            && self.preflight_constant_access_timing(now).is_ok();
        if !fast_path_safe {
            let mut shadow = self.clone();
            let outcome =
                shadow.access_eligible_constant_unobserved_impl(origin, sector_id, now)?;
            *self = shadow;
            return Ok(outcome);
        }
        self.access_eligible_constant_unobserved_impl(origin, sector_id, now)
    }

    fn access_eligible_event_core(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
        readiness: Readiness,
    ) -> Result<CoreAccess, L1AccessError> {
        self.commit_latch(FillTiming::Event);
        self.access_core(origin, sector_id, FillRequest::Event { now, readiness })
    }

    fn access_eligible_event_impl(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
    ) -> Result<EvAccess, L1AccessError> {
        let line = sector_id >> 2;
        let sector = (sector_id & 3) as u8;
        Ok(
            match self.access_eligible_event_core(origin, sector_id, now, Readiness::Observed)? {
                CoreAccess::Hit { ready: Some(ready) } => EvAccess::Hit { ready },
                CoreAccess::Hit { ready: None } => {
                    unreachable!("observed event access omitted its ready time")
                }
                CoreAccess::LineMerge { .. } => EvAccess::MergedPending {
                    fill: FillKey::Line(line),
                },
                CoreAccess::SectorMerge { .. } => EvAccess::MergedPending {
                    fill: FillKey::Sector(line, sector),
                },
                CoreAccess::Stalled => EvAccess::Stalled,
                CoreAccess::MissSectorResident { .. } => EvAccess::MissRequest {
                    fill: FillKey::Sector(line, sector),
                },
                CoreAccess::MissLineAlloc { .. } => EvAccess::MissRequest {
                    fill: FillKey::Line(line),
                },
            },
        )
    }

    fn access_eligible_event_unobserved_impl(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
    ) -> Result<EvAccessUnobserved, L1AccessError> {
        let line = sector_id >> 2;
        let sector = (sector_id & 3) as u8;
        Ok(
            match self.access_eligible_event_core(origin, sector_id, now, Readiness::Unobserved)? {
                CoreAccess::Hit { .. } => EvAccessUnobserved::Hit,
                CoreAccess::LineMerge { .. } => EvAccessUnobserved::MergedPending {
                    fill: FillKey::Line(line),
                },
                CoreAccess::SectorMerge { .. } => EvAccessUnobserved::MergedPending {
                    fill: FillKey::Sector(line, sector),
                },
                CoreAccess::Stalled => EvAccessUnobserved::Stalled,
                CoreAccess::MissSectorResident { .. } => EvAccessUnobserved::MissRequest {
                    fill: FillKey::Sector(line, sector),
                },
                CoreAccess::MissLineAlloc { .. } => EvAccessUnobserved::MissRequest {
                    fill: FillKey::Line(line),
                },
            },
        )
    }

    /// One modeled sector access in EVENT mode (shared-L2 integration):
    /// the fill time is decided by the next level, so misses register an
    /// explicitly UNTIMED in-flight entry. The lazy applier ignores it;
    /// [`Self::complete_fill`] installs it at the response event.
    /// Hit/miss/merge accounting matches [`Self::access_eligible`];
    /// `stall_events` does NOT — the constant
    /// mode's internal retry-at-earliest-fill always succeeds (one stall
    /// per stalled access), while event-mode parking re-counts a stall
    /// on every failed re-attempt after a completion, so `stall_events`
    /// reads "stall ATTEMPTS" in this mode (deliberate: the retry
    /// discipline is the caller's, and no emitted metric consumes the
    /// counter yet).
    ///
    /// The two modes must not be mixed on one cache instance: constant-
    /// mode entries carry real times the event mode would complete out
    /// of order.
    pub fn access_eligible_ev(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
    ) -> Result<EvAccess, L1AccessError> {
        self.preflight_latch(FillTiming::Event)?;
        let budget = L1MutationBudget {
            eligible_sectors: 1,
            recency_touches: 1,
            new_fills: 1,
            ..L1MutationBudget::for_origin(origin)
        };
        if self.preflight_mutation_budget(budget).is_err() {
            let mut shadow = self.clone();
            let outcome = shadow.access_eligible_event_impl(origin, sector_id, now)?;
            *self = shadow;
            return Ok(outcome);
        }
        self.access_eligible_event_impl(origin, sector_id, now)
    }

    /// Event-mode cache mutation when no dependence barrier consumes a hit's
    /// ready time. Miss, merge, stall, fill ownership, and counters are unchanged.
    pub fn access_eligible_ev_unobserved(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
    ) -> Result<EvAccessUnobserved, L1AccessError> {
        self.preflight_latch(FillTiming::Event)?;
        let budget = L1MutationBudget {
            eligible_sectors: 1,
            recency_touches: 1,
            new_fills: 1,
            ..L1MutationBudget::for_origin(origin)
        };
        if self.preflight_mutation_budget(budget).is_err() {
            let mut shadow = self.clone();
            let outcome = shadow.access_eligible_event_unobserved_impl(origin, sector_id, now)?;
            *self = shadow;
            return Ok(outcome);
        }
        self.access_eligible_event_unobserved_impl(origin, sector_id, now)
    }

    /// Complete an event-mode fill AT the response event's time. A
    /// sector fill whose line was evicted while it was in flight is
    /// DROPPED, like the lazy applier; returns whether the fill
    /// installed.
    ///
    /// Deliberate divergence from the constant path: `apply_fills`
    /// breaks SAME-TIMESTAMP ties line-before-sector (a batch-applier
    /// artifact the Python lockstep pins), while event mode completes
    /// same-cycle fills in response order — under eviction pressure the
    /// two orders can pick different LRU victims. Sub-cycle completion
    /// order is below the model's resolution either way; the event
    /// order is the shared-L2 system's single source of truth.
    pub fn complete_fill(&mut self, fill: FillKey) -> Result<bool, L1AccessError> {
        self.preflight_latch(FillTiming::Event)?;
        if !self.is_fill_in_flight(fill) {
            return Err(L1AccessError::FillNotInFlight { fill });
        }
        let will_touch = match fill {
            FillKey::Line(_) => true,
            FillKey::Sector(line, _) => self.line_resident(line),
        };
        if will_touch {
            self.next_stamp.preflight("recency", 1)?;
        }
        self.commit_latch(FillTiming::Event);
        match fill {
            FillKey::Line(line) => {
                self.line_fills.remove(&line);
                self.touch_mru(line)?;
                self.evict_beyond_capacity();
                Ok(true)
            }
            FillKey::Sector(line, sector) => {
                self.sector_fills.remove(&(line, sector));
                if self.line_resident(line) {
                    *self.sectors.entry(line).or_default() |= 1 << sector;
                    self.touch_mru(line)?;
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
        }
    }

    fn probe_store_impl(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
    ) -> Result<bool, L1AccessError> {
        let line = sector_id >> 2;
        let sector = (sector_id & 3) as u8;
        self.apply_fills(now)?;
        let tag_resident = self.line_resident(line);
        let hit = tag_resident
            && self
                .sectors
                .get(&line)
                .is_some_and(|mask| mask & (1 << sector) != 0);
        if hit {
            self.next_stamp.preflight("recency", 1)?;
        }
        let counters = self.counters.add_fields_checked(
            origin,
            &[
                (L1CounterField::StoreSectors, 1),
                (L1CounterField::StoreTagHits, u64::from(tag_resident)),
                (L1CounterField::StoreHits, u64::from(hit)),
            ],
        )?;
        self.counters = counters;
        if hit {
            self.touch_mru(line)?;
        }
        Ok(hit)
    }

    /// v0 extension: hit-TEST a global-class store sector against the
    /// current state without allocating on miss (write-through
    /// no-allocate hypothesis). A store hit re-references the line.
    /// Counts BOTH candidate hit rules — sector-valid (`store_hits`,
    /// the ported-era diagnostic) and the §6.4 write-class line-TAG
    /// rule (`store_tag_hits`) — without changing state behavior (MRU
    /// touch stays on the sector-valid hit only, preserving the L1
    /// state trajectory the scored op_ld exactness was proven on).
    pub fn probe_store(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
    ) -> Result<bool, L1AccessError> {
        let budget = L1MutationBudget {
            store_sectors: 1,
            recency_touches: 1,
            ..L1MutationBudget::for_origin(origin)
        };
        if self.preflight_mutation_budget(budget).is_err() {
            let mut shadow = self.clone();
            let hit = shadow.probe_store_impl(origin, sector_id, now)?;
            *self = shadow;
            return Ok(hit);
        }
        self.probe_store_impl(origin, sector_id, now)
    }

    /// Manifest-tagged ineligible sectors (the ported bypass counter).
    /// `atom` marks the write-class (ATOM-token) subset — the caller
    /// owns the token, this cache never sees opcodes.
    pub fn count_bypass(
        &mut self,
        origin: MemoryOrigin,
        sectors: u64,
        atom: bool,
    ) -> Result<(), L1AccessError> {
        self.apply_counter_deltas(
            origin,
            &[
                (L1CounterField::BypassSectors, sectors),
                (L1CounterField::BypassAtomSectors, sectors * u64::from(atom)),
            ],
        )
    }

    /// Global-class sectors whose pc the classifier did not tag (v0
    /// unified-denominator extension).
    pub fn count_untagged(
        &mut self,
        origin: MemoryOrigin,
        sectors: u64,
    ) -> Result<(), L1AccessError> {
        self.apply_counter_deltas(origin, &[(L1CounterField::UntaggedSectors, sectors)])
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SectorExpansionError {
    #[error("data_width_bytes={data_width_bytes} is not strictly positive")]
    DataWidthInvalid { data_width_bytes: i32 },
    #[error("active lane index {lane} is outside the 32-lane address vector")]
    LaneIndexOutOfRange { lane: usize },
    #[error(
        "lane {lane} address {base} plus data_width_bytes={data_width_bytes} overflows the signed address domain"
    )]
    AddressOverflow {
        lane: usize,
        base: i64,
        data_width_bytes: i32,
    },
    #[error("unique sector count {sectors} exceeds the configured limit {limit}")]
    UniqueSectorLimitExceeded { limit: usize, sectors: usize },
    #[error("unique sector count {sectors} does not fit the host index domain")]
    UniqueSectorCountOutOfRange { sectors: u128 },
    #[error("sector expansion produced {produced} entries, expected {expected}")]
    CardinalityMismatch { expected: usize, produced: usize },
    #[error("unable to reserve {additional} sector entries for {buffer}")]
    CapacityUnavailable {
        buffer: &'static str,
        additional: usize,
    },
}

/// Validated lane-sector intervals for all address records of one instruction.
/// Planning is range-based: no sector range is enumerated until the exact
/// instruction-wide union is known to fit its consumer's bound.
#[derive(Debug, Default)]
pub struct SectorExpansionPlan {
    intervals: Vec<(i64, i64)>,
}

/// A plan whose exact instruction-wide union cardinality has been computed.
#[derive(Debug)]
pub struct PreparedSectorExpansion {
    intervals: Vec<(i64, i64)>,
    unique_count: usize,
}

fn collect_sector_intervals(
    lane_vector: &[i64; 32],
    active_lanes: impl Iterator<Item = usize>,
    data_width_bytes: i32,
) -> Result<([(i64, i64); 32], usize), SectorExpansionError> {
    if data_width_bytes <= 0 {
        return Err(SectorExpansionError::DataWidthInvalid { data_width_bytes });
    }
    let width_offset = i64::from(data_width_bytes)
        .checked_sub(1)
        .ok_or(SectorExpansionError::DataWidthInvalid { data_width_bytes })?;
    let mut lane_seen = [false; 32];
    let mut intervals = [(0_i64, 0_i64); 32];
    let mut interval_count = 0;
    for lane in active_lanes {
        let base = *lane_vector
            .get(lane)
            .ok_or(SectorExpansionError::LaneIndexOutOfRange { lane })?;
        if lane_seen[lane] {
            continue;
        }
        lane_seen[lane] = true;
        let last_addr =
            base.checked_add(width_offset)
                .ok_or(SectorExpansionError::AddressOverflow {
                    lane,
                    base,
                    data_width_bytes,
                })?;
        intervals[interval_count] = (base >> 5, last_addr >> 5);
        interval_count += 1;
    }
    Ok((intervals, interval_count))
}

impl SectorExpansionPlan {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one address record without enumerating its sector ranges. The plan
    /// is unchanged if validation or fallible reservation fails.
    pub fn include_record(
        &mut self,
        lane_vector: &[i64; 32],
        active_lanes: impl Iterator<Item = usize>,
        data_width_bytes: i32,
    ) -> Result<(), SectorExpansionError> {
        let (intervals, interval_count) =
            collect_sector_intervals(lane_vector, active_lanes, data_width_bytes)?;
        self.intervals.try_reserve(interval_count).map_err(|_| {
            SectorExpansionError::CapacityUnavailable {
                buffer: "sector interval plan",
                additional: interval_count,
            }
        })?;
        self.intervals
            .extend_from_slice(&intervals[..interval_count]);
        Ok(())
    }

    /// Compute the exact union over every included address record in
    /// O(intervals log intervals), still without enumerating sectors.
    pub fn prepare(self) -> Result<PreparedSectorExpansion, SectorExpansionError> {
        let mut sorted = Vec::new();
        sorted.try_reserve(self.intervals.len()).map_err(|_| {
            SectorExpansionError::CapacityUnavailable {
                buffer: "sorted sector intervals",
                additional: self.intervals.len(),
            }
        })?;
        sorted.extend_from_slice(&self.intervals);
        sorted.sort_unstable();

        let mut union_count = 0_u128;
        if let Some(&(initial_first, initial_last)) = sorted.first() {
            let mut first = initial_first;
            let mut last = initial_last;
            for &(next_first, next_last) in &sorted[1..] {
                if next_first <= last.saturating_add(1) {
                    last = last.max(next_last);
                } else {
                    union_count += (i128::from(last) - i128::from(first) + 1).unsigned_abs();
                    first = next_first;
                    last = next_last;
                }
            }
            union_count += (i128::from(last) - i128::from(first) + 1).unsigned_abs();
        }
        let unique_count = usize::try_from(union_count).map_err(|_| {
            SectorExpansionError::UniqueSectorCountOutOfRange {
                sectors: union_count,
            }
        })?;
        Ok(PreparedSectorExpansion {
            intervals: self.intervals,
            unique_count,
        })
    }
}

impl PreparedSectorExpansion {
    pub fn unique_count(&self) -> usize {
        self.unique_count
    }

    /// Expand only after the exact instruction-wide union fits `max_unique`.
    /// Covered ranges are tracked as intervals, so repeated or overlapping
    /// records enumerate each sector exactly once while retaining the original
    /// lane-major, record-major first-seen order.
    pub fn expand_bounded(self, max_unique: usize) -> Result<Vec<i64>, SectorExpansionError> {
        if self.unique_count > max_unique {
            return Err(SectorExpansionError::UniqueSectorLimitExceeded {
                limit: max_unique,
                sectors: self.unique_count,
            });
        }
        let mut staged = Vec::new();
        staged.try_reserve(self.unique_count).map_err(|_| {
            SectorExpansionError::CapacityUnavailable {
                buffer: "sector staging",
                additional: self.unique_count,
            }
        })?;
        let covered_capacity = self.intervals.len().min(self.unique_count);
        let mut covered: Vec<(i64, i64)> = Vec::new();
        covered.try_reserve(covered_capacity).map_err(|_| {
            SectorExpansionError::CapacityUnavailable {
                buffer: "covered sector intervals",
                additional: covered_capacity,
            }
        })?;

        for (first, last) in self.intervals {
            let start = covered.partition_point(|&(_, covered_last)| covered_last < first);
            let mut cursor = Some(first);
            for &(covered_first, covered_last) in &covered[start..] {
                let Some(next) = cursor else {
                    break;
                };
                if covered_first > last {
                    break;
                }
                if next < covered_first {
                    for sector in next..=last.min(covered_first.saturating_sub(1)) {
                        staged.push(sector);
                    }
                }
                cursor = if covered_last >= last {
                    None
                } else {
                    covered_last
                        .checked_add(1)
                        .map(|after_covered| next.max(after_covered))
                };
            }
            if let Some(next) = cursor {
                for sector in next..=last {
                    staged.push(sector);
                }
            }

            let left = covered
                .partition_point(|&(_, covered_last)| covered_last.saturating_add(1) < first);
            let right = left
                + covered[left..]
                    .partition_point(|&(covered_first, _)| covered_first <= last.saturating_add(1));
            if left == right {
                covered.insert(left, (first, last));
            } else {
                let merged = (first.min(covered[left].0), last.max(covered[right - 1].1));
                covered[left] = merged;
                covered.drain(left + 1..right);
            }
        }

        if staged.len() != self.unique_count {
            return Err(SectorExpansionError::CardinalityMismatch {
                expected: self.unique_count,
                produced: staged.len(),
            });
        }
        Ok(staged)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn access(l1: &mut L1Cache, sector: i64, now: u64) -> (AccessOutcome, u64) {
        l1.access_eligible(MemoryOrigin::Global, sector, now)
            .unwrap()
    }

    fn access_ev(l1: &mut L1Cache, sector: i64, now: u64) -> EvAccess {
        l1.access_eligible_ev(MemoryOrigin::Global, sector, now)
            .unwrap()
    }

    fn complete(l1: &mut L1Cache, fill: FillKey) -> bool {
        l1.complete_fill(fill).unwrap()
    }

    fn store(l1: &mut L1Cache, sector: i64, now: u64) -> bool {
        l1.probe_store(MemoryOrigin::Global, sector, now).unwrap()
    }

    fn cache(capacity: u64, mshr: Option<u32>) -> L1Cache {
        L1Cache::new(L1Config {
            arch: ArchId::Sm86,
            line_capacity: capacity,
            mshr_count: mshr,
            hit_latency: 34,
            miss_latency: 244,
        })
        .unwrap()
    }

    fn assert_cache_state_eq(actual: &L1Cache, expected: &L1Cache) {
        assert_eq!(actual.cfg.arch, expected.cfg.arch);
        assert_eq!(actual.cfg.line_capacity, expected.cfg.line_capacity);
        assert_eq!(actual.cfg.mshr_count, expected.cfg.mshr_count);
        assert_eq!(actual.cfg.hit_latency, expected.cfg.hit_latency);
        assert_eq!(actual.cfg.miss_latency, expected.cfg.miss_latency);
        assert_eq!(actual.by_stamp, expected.by_stamp);
        assert_eq!(actual.stamp_of, expected.stamp_of);
        assert_eq!(actual.next_stamp, expected.next_stamp);
        assert_eq!(actual.sectors, expected.sectors);
        assert_eq!(actual.line_fills, expected.line_fills);
        assert_eq!(actual.sector_fills, expected.sector_fills);
        assert_eq!(actual.fill_seq, expected.fill_seq);
        assert_eq!(actual.min_fill_due, expected.min_fill_due);
        assert_eq!(actual.fill_timing, expected.fill_timing);
        assert_eq!(actual.counters, expected.counters);
    }

    fn counters_invariant(c: &L1Counters) {
        assert_eq!(
            c.misses,
            c.line_alloc_misses + c.sector_resident_misses + c.merges,
            "miss == line_alloc + sector_resident + merge"
        );
    }

    fn assert_origin_partition(c: &L1Counters) {
        let global = c.origin(MemoryOrigin::Global);
        let local = c.origin(MemoryOrigin::Local);
        macro_rules! partition {
            ($field:ident) => {
                assert_eq!(
                    c.$field,
                    global.$field + local.$field,
                    concat!(stringify!($field), " origin partition")
                );
            };
        }
        partition!(hits);
        partition!(misses);
        partition!(merges);
        partition!(stall_events);
        partition!(line_fills);
        partition!(line_alloc_misses);
        partition!(sector_resident_misses);
        partition!(bypass_sectors);
        partition!(bypass_atom_sectors);
        partition!(untagged_sectors);
        partition!(store_sectors);
        partition!(store_hits);
        partition!(store_tag_hits);
        for (name, lane) in [("Global", global), ("Local", local)] {
            assert_eq!(
                lane.misses,
                lane.line_alloc_misses + lane.sector_resident_misses + lane.merges,
                "{name} miss partition"
            );
            assert!(
                lane.bypass_atom_sectors <= lane.bypass_sectors,
                "{name} bypass-ATOM subset"
            );
        }
        assert!(c.bypass_atom_sectors <= c.bypass_sectors);
    }

    #[test]
    fn sector_with_line_allocation_hit_decision() {
        let mut l1 = cache(8, None);
        // Sector 0 of line 0: line-alloc miss; fills at t=244.
        let (o, ready) = access(&mut l1, 0, 0);
        assert_eq!(o, AccessOutcome::MissLineAlloc);
        assert_eq!(ready, 244);
        // After the fill: same sector hits, but sector 1 of the SAME
        // (now-resident) line is a sector-resident miss, not a hit.
        let (o, _) = access(&mut l1, 0, 300);
        assert_eq!(o, AccessOutcome::Hit);
        let (o, _) = access(&mut l1, 1, 300);
        assert_eq!(o, AccessOutcome::MissSectorResident);
        counters_invariant(&l1.counters);
    }

    #[test]
    fn line_merge_while_allocation_in_flight() {
        let mut l1 = cache(8, None);
        let (_, fill) = access(&mut l1, 0, 0);
        // Different sector of the same line while the allocation is in
        // flight: merge (counts as a miss), ready at the SAME fill time,
        // and the fill services the merged sector too.
        let (o, ready) = access(&mut l1, 1, 10);
        assert_eq!(o, AccessOutcome::Merge);
        assert_eq!(ready, fill);
        let (o, _) = access(&mut l1, 1, 250);
        assert_eq!(
            o,
            AccessOutcome::Hit,
            "merged sector serviced by the line fill"
        );
        counters_invariant(&l1.counters);
    }

    #[test]
    fn lazy_fill_is_not_a_hit_before_fill_time() {
        let mut l1 = cache(8, None);
        access(&mut l1, 0, 0);
        // Same sector before the fill time: merge, never an LRU hit.
        let (o, _) = access(&mut l1, 0, 100);
        assert_eq!(o, AccessOutcome::Merge);
    }

    #[test]
    fn unobserved_hits_skip_unused_visibility_arithmetic() {
        let cfg = L1Config {
            arch: ArchId::Sm86,
            line_capacity: 8,
            mshr_count: Some(1),
            hit_latency: 2,
            miss_latency: 3,
        };

        let mut constant = L1Cache::new(cfg).unwrap();
        assert_eq!(
            constant
                .access_eligible_unobserved(MemoryOrigin::Global, 0, 0)
                .unwrap(),
            AccessOutcome::MissLineAlloc
        );
        assert_eq!(
            constant
                .access_eligible_unobserved(MemoryOrigin::Global, 0, 3)
                .unwrap(),
            AccessOutcome::Hit
        );
        let before_observed = constant.clone();
        assert!(matches!(
            constant.access_eligible(MemoryOrigin::Global,0, u64::MAX - 1),
            Err(L1AccessError::TimeOverflow {
                operation: "hit visibility",
                base,
                delta: 2,
            }) if base == u64::MAX - 1
        ));
        assert_cache_state_eq(&constant, &before_observed);
        assert_eq!(
            constant
                .access_eligible_unobserved(MemoryOrigin::Global, 0, u64::MAX - 1)
                .unwrap(),
            AccessOutcome::Hit
        );

        let mut event = L1Cache::new(cfg).unwrap();
        let EvAccess::MissRequest { fill } = event
            .access_eligible_ev(MemoryOrigin::Global, 0, 0)
            .unwrap()
        else {
            panic!("expected event-mode prefill miss");
        };
        event.complete_fill(fill).unwrap();
        let before_observed = event.clone();
        assert!(matches!(
            event.access_eligible_ev(MemoryOrigin::Global,0, u64::MAX - 1),
            Err(L1AccessError::TimeOverflow {
                operation: "event hit visibility",
                base,
                delta: 2,
            }) if base == u64::MAX - 1
        ));
        assert_cache_state_eq(&event, &before_observed);
        assert_eq!(
            event
                .access_eligible_ev_unobserved(MemoryOrigin::Global, 0, u64::MAX - 1)
                .unwrap(),
            EvAccessUnobserved::Hit
        );
    }

    #[test]
    fn unobserved_constant_miss_still_requires_a_fill_due_time() {
        let cfg = L1Config {
            arch: ArchId::Sm86,
            line_capacity: 8,
            mshr_count: Some(1),
            hit_latency: 1,
            miss_latency: 2,
        };
        let mut l1 = L1Cache::new(cfg).unwrap();
        let before = l1.clone();

        assert!(matches!(
            l1.access_eligible_unobserved(MemoryOrigin::Global,0, u64::MAX - 1),
            Err(L1AccessError::TimeOverflow {
                operation: "miss visibility",
                base,
                delta: 2,
            }) if base == u64::MAX - 1
        ));
        assert_cache_state_eq(&l1, &before);
    }

    #[test]
    fn exact_maximum_is_a_real_timed_fill_and_overflow_stays_typed() {
        let config = L1Config {
            arch: ArchId::Sm86,
            line_capacity: 8,
            mshr_count: Some(1),
            hit_latency: 0,
            miss_latency: u64::MAX,
        };
        let mut l1 = L1Cache::new(config).unwrap();
        assert_eq!(
            l1.access_eligible(MemoryOrigin::Global, 0, 0).unwrap(),
            (AccessOutcome::MissLineAlloc, u64::MAX)
        );
        assert_eq!(l1.min_fill_due, Some(u64::MAX));
        assert_eq!(l1.line_fills.get(&0), Some(&(FillDue::Timed(u64::MAX), 0)));

        l1.apply_fills(u64::MAX - 1).unwrap();
        assert!(l1.is_fill_in_flight(FillKey::Line(0)));
        assert!(!l1.line_resident(0));
        l1.apply_fills(u64::MAX).unwrap();
        assert!(!l1.is_fill_in_flight(FillKey::Line(0)));
        assert!(l1.line_resident(0));
        assert_eq!(l1.min_fill_due, None);

        let mut overflow = L1Cache::new(config).unwrap();
        let before = overflow.clone();
        assert_eq!(
            overflow.access_eligible(MemoryOrigin::Global, 0, 1),
            Err(L1AccessError::TimeOverflow {
                operation: "miss visibility",
                base: 1,
                delta: u64::MAX,
            })
        );
        assert_cache_state_eq(&overflow, &before);
    }

    #[test]
    fn constant_merge_ignores_an_overflowing_unused_miss_arm() {
        let mut l1 = L1Cache::new(L1Config {
            arch: ArchId::Sm86,
            line_capacity: 8,
            mshr_count: Some(1),
            hit_latency: 1,
            miss_latency: 2,
        })
        .unwrap();
        assert_eq!(
            l1.access_eligible(MemoryOrigin::Global, 0, u64::MAX - 2)
                .unwrap(),
            (AccessOutcome::MissLineAlloc, u64::MAX)
        );

        assert_eq!(
            l1.access_eligible(MemoryOrigin::Global, 1, u64::MAX - 1)
                .unwrap(),
            (AccessOutcome::Merge, u64::MAX)
        );
        assert!(l1.is_fill_in_flight(FillKey::Line(0)));
        assert_eq!(l1.counters.misses, 2);
        assert_eq!(l1.counters.merges, 1);
    }

    #[test]
    fn event_nonhit_paths_ignore_an_overflowing_unused_hit_arm() {
        let mut l1 = L1Cache::new(L1Config {
            arch: ArchId::Sm86,
            line_capacity: 8,
            mshr_count: Some(1),
            hit_latency: 1,
            miss_latency: 2,
        })
        .unwrap();
        assert_eq!(
            l1.access_eligible_ev(MemoryOrigin::Global, 0, u64::MAX)
                .unwrap(),
            EvAccess::MissRequest {
                fill: FillKey::Line(0),
            }
        );
        assert_eq!(
            l1.access_eligible_ev(MemoryOrigin::Global, 1, u64::MAX)
                .unwrap(),
            EvAccess::MergedPending {
                fill: FillKey::Line(0),
            }
        );
        assert_eq!(
            l1.access_eligible_ev(MemoryOrigin::Global, 4, u64::MAX)
                .unwrap(),
            EvAccess::Stalled
        );

        assert!(l1.complete_fill(FillKey::Line(0)).unwrap());
        let before = l1.clone();
        assert_eq!(
            l1.access_eligible_ev(MemoryOrigin::Global, 0, u64::MAX),
            Err(L1AccessError::TimeOverflow {
                operation: "event hit visibility",
                base: u64::MAX,
                delta: 1,
            })
        );
        assert_cache_state_eq(&l1, &before);
    }

    #[test]
    fn event_fill_remains_untimed_at_maximum_cycle() {
        let mut l1 = cache(8, Some(1));
        assert_eq!(
            access_ev(&mut l1, 0, 0),
            EvAccess::MissRequest {
                fill: FillKey::Line(0),
            }
        );
        assert_eq!(l1.min_fill_due, None);
        assert_eq!(l1.line_fills.get(&0), Some(&(FillDue::Untimed, 0)));
        let before = l1.clone();

        l1.apply_fills(u64::MAX).unwrap();

        assert_cache_state_eq(&l1, &before);
        assert!(l1.is_fill_in_flight(FillKey::Line(0)));
        assert!(!l1.line_resident(0));
    }

    #[test]
    fn maximum_recency_stamp_is_used_before_typed_exhaustion() {
        let mut l1 = cache(8, None);
        l1.by_stamp.insert(0, 0);
        l1.stamp_of.insert(0, 0);
        l1.sectors.insert(0, 1);
        l1.next_stamp = NextSequence::Available(u64::MAX);

        assert_eq!(
            l1.access_eligible_ev(MemoryOrigin::Global, 0, 0),
            Ok(EvAccess::Hit { ready: 34 })
        );
        assert_eq!(l1.next_stamp, NextSequence::Exhausted);
        assert_eq!(l1.stamp_of.get(&0), Some(&u64::MAX));
        let before = l1.clone();

        assert_eq!(
            l1.access_eligible_ev(MemoryOrigin::Global, 0, 1),
            Err(L1AccessError::SequenceOverflow {
                sequence: "recency",
                current: u64::MAX,
            })
        );
        assert_cache_state_eq(&l1, &before);
    }

    #[test]
    fn maximum_fill_sequence_is_used_before_typed_exhaustion() {
        let mut l1 = cache(8, None);
        l1.fill_seq = NextSequence::Available(u64::MAX);

        assert_eq!(
            l1.access_eligible_ev(MemoryOrigin::Global, 0, 0),
            Ok(EvAccess::MissRequest {
                fill: FillKey::Line(0),
            })
        );
        assert_eq!(l1.fill_seq, NextSequence::Exhausted);
        assert_eq!(l1.line_fills.get(&0), Some(&(FillDue::Untimed, u64::MAX)));
        let before = l1.clone();

        assert_eq!(
            l1.access_eligible_ev(MemoryOrigin::Global, 4, 1),
            Err(L1AccessError::SequenceOverflow {
                sequence: "fill",
                current: u64::MAX,
            })
        );
        assert_cache_state_eq(&l1, &before);
    }

    #[test]
    fn same_missing_sector_on_resident_line_merges() {
        let mut l1 = cache(8, None);
        access(&mut l1, 0, 0);
        l1.apply_fills(244).unwrap();
        // Sector 1 fetch in flight on the resident line...
        let (o, fill) = access(&mut l1, 1, 250);
        assert_eq!(o, AccessOutcome::MissSectorResident);
        // ...same sector merges; OTHER resident sector still hits.
        let (o, ready) = access(&mut l1, 1, 260);
        assert_eq!(o, AccessOutcome::Merge);
        assert_eq!(ready, fill);
        let (o, _) = access(&mut l1, 0, 260);
        assert_eq!(
            o,
            AccessOutcome::Hit,
            "sector fetch must not shadow its line"
        );
        counters_invariant(&l1.counters);
    }

    #[test]
    fn eviction_drops_sector_residency_and_inflight_sector_fill() {
        let mut l1 = cache(1, None);
        // Line 0 resident.
        access(&mut l1, 0, 0);
        l1.apply_fills(244).unwrap();
        // Sector 1 of line 0 in flight (fills at 250+244=494)...
        access(&mut l1, 1, 250);
        // ...line 1 allocation evicts line 0 (capacity 1) at its own fill
        // time 251+244=495 > 494. The sector fill applies FIRST (chrono
        // order) into the still-resident line, then the line fill evicts.
        access(&mut l1, 4, 251);
        l1.apply_fills(1000).unwrap();
        // Line 0 evicted: sector 0 must MISS as a fresh line allocation.
        let (o, _) = access(&mut l1, 0, 1000);
        assert_eq!(o, AccessOutcome::MissLineAlloc);
        counters_invariant(&l1.counters);
    }

    #[test]
    fn sector_fill_dropped_when_line_evicted_before_its_timestamp() {
        let mut l1 = cache(1, None);
        access(&mut l1, 0, 0); // line 0
        l1.apply_fills(244).unwrap();
        // Line 1 alloc at t=245 → fills 489, evicting line 0 at 489.
        access(&mut l1, 4, 245);
        // Sector fetch into line 0 at t=246 → would fill at 490, AFTER
        // the eviction: must be dropped.
        let (o, _) = access(&mut l1, 1, 246);
        assert_eq!(o, AccessOutcome::MissSectorResident);
        l1.apply_fills(10_000).unwrap();
        // Line 0 fully gone; re-access = fresh allocation, and the
        // dropped sector never resurrects.
        let (o, _) = access(&mut l1, 1, 10_000);
        assert_eq!(o, AccessOutcome::MissLineAlloc);
        counters_invariant(&l1.counters);
    }

    #[test]
    fn mshr_full_stalls_and_retries() {
        let mut l1 = cache(8, Some(1));
        let (_, fill) = access(&mut l1, 0, 0); // occupies the only MSHR
        // A different line at t=1 stalls until the fill frees the entry,
        // then misses: ready = fill_time + miss_latency.
        let (o, ready) = access(&mut l1, 8, 1);
        assert_eq!(o, AccessOutcome::MissLineAlloc);
        assert_eq!(ready, fill + 244);
        assert_eq!(l1.counters.stall_events, 1);
        counters_invariant(&l1.counters);
    }

    #[test]
    fn lru_eviction_order() {
        let mut l1 = cache(2, None);
        access(&mut l1, 0, 0); // line 0
        access(&mut l1, 4, 1); // line 1
        l1.apply_fills(300).unwrap();
        access(&mut l1, 0, 300); // touch line 0 → MRU
        // Line 2 evicts the LRU line 1 (not line 0).
        access(&mut l1, 8, 301);
        l1.apply_fills(1000).unwrap();
        let (o, _) = access(&mut l1, 0, 1000);
        assert_eq!(o, AccessOutcome::Hit);
        let (o, _) = access(&mut l1, 4, 1001);
        assert_eq!(o, AccessOutcome::MissLineAlloc);
        counters_invariant(&l1.counters);
    }

    #[test]
    fn origins_partition_counters_but_share_one_physical_cache() {
        let mut l1 = cache(8, None);
        assert_eq!(
            l1.access_eligible(MemoryOrigin::Global, 0, 0).unwrap().0,
            AccessOutcome::MissLineAlloc
        );
        assert_eq!(
            l1.access_eligible(MemoryOrigin::Local, 0, 300).unwrap().0,
            AccessOutcome::Hit,
            "Local must hit the line installed by Global traffic"
        );
        assert!(l1.probe_store(MemoryOrigin::Local, 0, 300).unwrap());
        l1.count_bypass(MemoryOrigin::Global, 3, false).unwrap();
        l1.count_bypass(MemoryOrigin::Global, 2, true).unwrap();
        l1.count_untagged(MemoryOrigin::Local, 4).unwrap();

        let global = l1.counters.origin(MemoryOrigin::Global);
        let local = l1.counters.origin(MemoryOrigin::Local);
        assert_eq!(global.line_alloc_misses, 1);
        assert_eq!(global.misses, 1);
        assert_eq!(global.bypass_sectors, 5);
        assert_eq!(global.bypass_atom_sectors, 2);
        assert_eq!(local.hits, 1);
        assert_eq!(local.store_sectors, 1);
        assert_eq!(local.store_hits, 1);
        assert_eq!(local.untagged_sectors, 4);
        assert_origin_partition(&l1.counters);
    }

    #[test]
    fn store_probe_tests_without_allocating() {
        let mut l1 = cache(8, None);
        assert!(!store(&mut l1, 0, 0), "cold store misses");
        // The store must NOT have allocated: a load still line-alloc
        // misses.
        let (o, _) = access(&mut l1, 0, 10);
        assert_eq!(o, AccessOutcome::MissLineAlloc);
        l1.apply_fills(300).unwrap();
        assert!(store(&mut l1, 0, 300), "store hits the resident sector");
        // A store to a DIFFERENT sector of the resident line: sector-
        // valid rule misses, §6.4 line-tag rule hits.
        assert!(!store(&mut l1, 1, 300), "sector-valid rule still misses");
        assert_eq!(l1.counters.store_sectors, 3);
        assert_eq!(l1.counters.store_hits, 1);
        assert_eq!(
            l1.counters.store_tag_hits, 2,
            "tag rule counts the resident-line probes (both post-fill)"
        );
    }

    #[test]
    fn unique_sector_expansion_dedups_in_first_seen_order() {
        let mut lanes = [i64::MIN; 32];
        // 4 active lanes, 4B apart starting at 0: all in sector 0.
        for (i, l) in lanes.iter_mut().enumerate().take(4) {
            *l = (i * 4) as i64;
        }
        // Lane 4 crosses sectors 3→4 (base 127, width 4).
        lanes[4] = 127;
        let mut plan = SectorExpansionPlan::new();
        plan.include_record(&lanes, 0..5, 4).unwrap();
        // A second address record of the SAME instruction dedups against
        // the instruction's whole sector union.
        plan.include_record(&lanes, 0..5, 4).unwrap();
        let prepared = plan.prepare().unwrap();
        assert_eq!(prepared.unique_count(), 3);
        assert_eq!(prepared.expand_bounded(usize::MAX).unwrap(), vec![0, 3, 4]);
    }

    #[test]
    fn overlapping_records_preserve_first_seen_sector_order() {
        let mut later_first = [i64::MIN; 32];
        later_first[0] = 10 * i64::from(SECTOR_BYTES);
        let mut earlier_overlap = [i64::MIN; 32];
        earlier_overlap[0] = 8 * i64::from(SECTOR_BYTES);
        let mut plan = SectorExpansionPlan::new();
        plan.include_record(&later_first, 0..1, 96).unwrap();
        plan.include_record(&earlier_overlap, 0..1, 128).unwrap();

        let prepared = plan.prepare().unwrap();
        assert_eq!(prepared.unique_count(), 5);
        assert_eq!(prepared.expand_bounded(5).unwrap(), vec![10, 11, 12, 8, 9]);
    }

    #[test]
    fn oversized_sector_union_is_rejected_before_expansion() {
        let mut lanes = [i64::MIN; 32];
        lanes[0] = 0;
        let mut plan = SectorExpansionPlan::new();
        plan.include_record(&lanes, 0..1, i32::MAX).unwrap();
        let prepared = plan.prepare().unwrap();
        assert_eq!(prepared.unique_count(), 67_108_864);

        assert_eq!(
            prepared.expand_bounded(usize::from(u16::MAX)),
            Err(SectorExpansionError::UniqueSectorLimitExceeded {
                limit: usize::from(u16::MAX),
                sectors: 67_108_864,
            })
        );
    }

    #[test]
    fn bounded_sector_expansion_uses_union_not_interval_sum() {
        let mut lanes = [i64::MIN; 32];
        lanes[0] = 0;
        lanes[1] = 1;
        let mut plan = SectorExpansionPlan::new();
        plan.include_record(&lanes, 0..2, 64).unwrap();
        let prepared = plan.prepare().unwrap();

        assert_eq!(prepared.unique_count(), 3);
        assert_eq!(prepared.expand_bounded(3).unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn accumulated_sector_limit_failure_is_nonmutating() {
        let mut first = [i64::MIN; 32];
        first[0] = 0;
        let mut second = [i64::MIN; 32];
        second[0] = 64;
        let mut plan = SectorExpansionPlan::new();
        plan.include_record(&first, 0..1, 64).unwrap();
        plan.include_record(&second, 0..1, 64).unwrap();
        let prepared = plan.prepare().unwrap();

        assert_eq!(prepared.unique_count(), 4);
        assert_eq!(
            prepared.expand_bounded(3),
            Err(SectorExpansionError::UniqueSectorLimitExceeded {
                limit: 3,
                sectors: 4,
            })
        );
    }

    #[test]
    fn high_address_sector_expansion_overflow_is_typed() {
        let mut lanes = [i64::MIN; 32];
        lanes[0] = 0;
        lanes[1] = i64::MAX;
        let mut plan = SectorExpansionPlan::new();
        assert_eq!(
            plan.include_record(&lanes, 0..2, 2),
            Err(SectorExpansionError::AddressOverflow {
                lane: 1,
                base: i64::MAX,
                data_width_bytes: 2,
            })
        );
        assert!(plan.intervals.is_empty());
    }

    #[test]
    fn nonpositive_sector_width_is_rejected() {
        let lanes = [0; 32];
        let mut plan = SectorExpansionPlan::new();
        assert_eq!(
            plan.include_record(&lanes, 0..1, 0),
            Err(SectorExpansionError::DataWidthInvalid {
                data_width_bytes: 0,
            })
        );
        assert!(plan.intervals.is_empty());
    }

    #[test]
    fn malformed_sector_expansion_indices_are_typed_and_nonmutating() {
        let lanes = [0; 32];
        let mut plan = SectorExpansionPlan::new();
        plan.include_record(&lanes, 0..1, 1).unwrap();
        let before = plan.intervals.clone();

        assert_eq!(
            plan.include_record(&lanes, std::iter::once(32), 1),
            Err(SectorExpansionError::LaneIndexOutOfRange { lane: 32 })
        );
        assert_eq!(plan.intervals, before);
    }

    #[test]
    fn mixing_fill_timing_modes_is_typed_and_nonmutating() {
        let mut l1 = cache(8, None);
        access(&mut l1, 0, 0); // latches Constant
        let before = l1.counters;
        assert_eq!(
            l1.access_eligible_ev(MemoryOrigin::Global, 4, 1),
            Err(L1AccessError::FillTimingMismatch {
                latched: "constant",
                requested: "event",
            })
        );
        assert_eq!(l1.counters, before);
    }

    #[test]
    fn constant_retry_error_rolls_back_fill_stall_and_mode_state() {
        let mut l1 = cache(8, Some(1));
        access(&mut l1, 0, 0);
        l1.counters.misses = u64::MAX;
        let before = l1.clone();

        assert_eq!(
            l1.access_eligible(MemoryOrigin::Global, 8, 1),
            Err(L1AccessError::Counter(L1CounterError::Overflow {
                field: L1CounterField::Misses,
            }))
        );
        assert_cache_state_eq(&l1, &before);
    }

    #[test]
    fn event_counter_error_does_not_latch_mode_or_create_a_fill() {
        let mut l1 = cache(8, Some(1));
        l1.counters.misses = u64::MAX;
        let before = l1.clone();

        assert_eq!(
            l1.access_eligible_ev(MemoryOrigin::Global, 0, 0),
            Err(L1AccessError::Counter(L1CounterError::Overflow {
                field: L1CounterField::Misses,
            }))
        );
        assert_cache_state_eq(&l1, &before);
    }

    #[test]
    fn exact_slow_path_accepts_an_unrelated_maximum_counter() {
        let mut l1 = cache(8, Some(1));
        l1.counters.hits = u64::MAX;

        assert_eq!(
            l1.access_eligible_ev(MemoryOrigin::Global, 0, 0).unwrap(),
            EvAccess::MissRequest {
                fill: FillKey::Line(0),
            }
        );
        assert_eq!(l1.counters.hits, u64::MAX);
        assert_eq!(l1.counters.misses, 1);
        assert!(l1.is_fill_in_flight(FillKey::Line(0)));
    }

    #[test]
    fn store_counter_error_does_not_apply_a_due_fill() {
        let mut l1 = cache(8, Some(1));
        access(&mut l1, 0, 0);
        l1.counters.store_sectors = u64::MAX;
        let before = l1.clone();

        assert_eq!(
            l1.probe_store(MemoryOrigin::Global, 4, 300),
            Err(L1AccessError::Counter(L1CounterError::Overflow {
                field: L1CounterField::StoreSectors,
            }))
        );
        assert_cache_state_eq(&l1, &before);
    }

    #[test]
    fn missing_fill_completion_is_typed_and_nonmutating() {
        let mut l1 = cache(8, Some(4));
        let before = l1.counters;
        assert_eq!(
            l1.complete_fill(FillKey::Line(7)),
            Err(L1AccessError::FillNotInFlight {
                fill: FillKey::Line(7),
            })
        );
        assert_eq!(l1.counters, before);
        assert_eq!(l1.fill_timing, None);
        assert!(l1.line_fills.is_empty());
        assert!(l1.sector_fills.is_empty());
    }

    #[test]
    fn ev_mode_miss_merge_completion_then_hit() {
        let mut l1 = cache(8, Some(4));
        // Fresh line miss → request keyed by line.
        let a = access_ev(&mut l1, 0, 0);
        assert_eq!(
            a,
            EvAccess::MissRequest {
                fill: FillKey::Line(0)
            }
        );
        // Another sector of the same line merges into the SAME fill.
        let b = access_ev(&mut l1, 1, 5);
        assert_eq!(
            b,
            EvAccess::MergedPending {
                fill: FillKey::Line(0)
            }
        );
        // Response installs line + both merged sectors; hits follow.
        assert!(complete(&mut l1, FillKey::Line(0)));
        assert!(matches!(
            access_ev(&mut l1, 0, 400),
            EvAccess::Hit { ready: 434 }
        ));
        assert!(matches!(access_ev(&mut l1, 1, 400), EvAccess::Hit { .. }));
        // Sector 2 on the now-resident line: sector-resident request, and
        // the same missing sector merges while in flight.
        assert_eq!(
            access_ev(&mut l1, 2, 401),
            EvAccess::MissRequest {
                fill: FillKey::Sector(0, 2)
            }
        );
        assert_eq!(
            access_ev(&mut l1, 2, 402),
            EvAccess::MergedPending {
                fill: FillKey::Sector(0, 2)
            }
        );
        assert!(complete(&mut l1, FillKey::Sector(0, 2)));
        assert!(matches!(access_ev(&mut l1, 2, 900), EvAccess::Hit { .. }));
        assert_eq!(l1.counters.hits, 3);
        assert_eq!(l1.counters.misses, 4);
        assert_eq!(l1.counters.merges, 2);
        assert_eq!(l1.counters.line_alloc_misses, 1);
        assert_eq!(l1.counters.sector_resident_misses, 1);
        counters_invariant(&l1.counters);
    }

    #[test]
    fn ev_mode_sector_fill_dropped_when_line_evicted_in_flight() {
        let mut l1 = cache(1, None);
        access_ev(&mut l1, 0, 0); // line 0 alloc
        complete(&mut l1, FillKey::Line(0));
        // Sector fetch in flight on line 0...
        access_ev(&mut l1, 1, 10);
        // ...line 1 allocation completes first and evicts line 0
        // (capacity 1).
        access_ev(&mut l1, 4, 11);
        assert!(complete(&mut l1, FillKey::Line(1)));
        // The sector response then finds its line gone: DROPPED.
        assert!(!complete(&mut l1, FillKey::Sector(0, 1)));
        assert_eq!(
            access_ev(&mut l1, 1, 500),
            EvAccess::MissRequest {
                fill: FillKey::Line(0)
            },
            "dropped sector never resurrects; the line itself was evicted"
        );
        counters_invariant(&l1.counters);
    }

    #[test]
    fn ev_mode_mshr_full_stalls_until_completion_frees_an_entry() {
        let mut l1 = cache(8, Some(1));
        access_ev(&mut l1, 0, 0); // occupies the only MSHR
        assert_eq!(access_ev(&mut l1, 8, 1), EvAccess::Stalled);
        assert_eq!(l1.counters.stall_events, 1);
        complete(&mut l1, FillKey::Line(0));
        assert_eq!(
            access_ev(&mut l1, 8, 2),
            EvAccess::MissRequest {
                fill: FillKey::Line(2)
            }
        );
        counters_invariant(&l1.counters);
    }

    #[test]
    fn counter_aggregation_preserves_both_origin_lanes() {
        let mut global_cache = cache(8, None);
        global_cache
            .access_eligible(MemoryOrigin::Global, 0, 0)
            .unwrap();
        let mut local_cache = cache(8, None);
        local_cache
            .access_eligible(MemoryOrigin::Local, 4, 0)
            .unwrap();

        let mut total = global_cache.counters;
        total.add_checked(&local_cache.counters).unwrap();
        assert_eq!(total.misses, 2);
        assert_eq!(total.origin(MemoryOrigin::Global).misses, 1);
        assert_eq!(total.origin(MemoryOrigin::Local).misses, 1);
        assert_origin_partition(&total);
    }

    #[test]
    fn unified_counter_overflow_precedes_origin_lane_overflow() {
        let mut counters = L1Counters::for_arch(ArchId::Sm86);
        counters.hits = u64::MAX;
        counters.by_origin[MemoryOrigin::Local.index()].hits = u64::MAX;
        assert_eq!(
            counters.add_fields_checked(MemoryOrigin::Local, &[(L1CounterField::Hits, 1)]),
            Err(L1CounterError::Overflow {
                field: L1CounterField::Hits,
            })
        );

        counters.hits = 0;
        assert_eq!(
            counters.add_fields_checked(MemoryOrigin::Local, &[(L1CounterField::Hits, 1)]),
            Err(L1CounterError::OriginOverflow {
                origin: MemoryOrigin::Local,
                field: L1CounterField::Hits,
            })
        );
    }

    #[test]
    fn counter_aggregation_rejects_mixed_architecture() {
        let mut sm86 = L1Counters::for_arch(ArchId::Sm86);
        let sm120 = L1Counters::for_arch(ArchId::Sm120);
        assert!(matches!(
            sm86.add_checked(&sm120),
            Err(L1CounterError::Architecture(
                sim_arch::ArchitectureProvenanceError::Mismatch {
                    component: "L1 counters",
                    expected: ArchId::Sm86,
                    got: ArchId::Sm120,
                }
            ))
        ));
    }

    #[test]
    fn counter_aggregation_overflow_is_named_and_transactional() {
        type Setter = fn(&mut L1Counters, u64);
        let cases: [(L1CounterField, Setter); 13] = [
            (L1CounterField::Hits, |c, v| c.hits = v),
            (L1CounterField::Misses, |c, v| c.misses = v),
            (L1CounterField::Merges, |c, v| c.merges = v),
            (L1CounterField::StallEvents, |c, v| c.stall_events = v),
            (L1CounterField::LineFills, |c, v| c.line_fills = v),
            (L1CounterField::LineAllocMisses, |c, v| {
                c.line_alloc_misses = v
            }),
            (L1CounterField::SectorResidentMisses, |c, v| {
                c.sector_resident_misses = v
            }),
            (L1CounterField::BypassSectors, |c, v| c.bypass_sectors = v),
            (L1CounterField::BypassAtomSectors, |c, v| {
                c.bypass_atom_sectors = v
            }),
            (L1CounterField::UntaggedSectors, |c, v| {
                c.untagged_sectors = v
            }),
            (L1CounterField::StoreSectors, |c, v| c.store_sectors = v),
            (L1CounterField::StoreHits, |c, v| c.store_hits = v),
            (L1CounterField::StoreTagHits, |c, v| c.store_tag_hits = v),
        ];
        for (field, set) in cases {
            let mut total = L1Counters::for_arch(ArchId::Sm86);
            total.hits = 7;
            set(&mut total, u64::MAX);
            let before = total;
            let mut addend = L1Counters::for_arch(ArchId::Sm86);
            addend.misses = 11;
            set(&mut addend, 1);
            assert!(matches!(
                total.add_checked(&addend),
                Err(L1CounterError::Overflow { field: got }) if got == field
            ));
            assert_eq!(total, before, "overflow in {field} partially committed");
        }
    }

    #[test]
    fn counter_aggregation_rejects_incoherent_origin_partition_transactionally() {
        let mut total = L1Counters::for_arch(ArchId::Sm86);
        total.hits = 1;
        let addend = L1Counters::for_arch(ArchId::Sm86);
        let before = total;

        assert_eq!(
            total.add_checked(&addend),
            Err(L1CounterError::OriginPartitionMismatch {
                field: L1CounterField::Hits,
                unified: 1,
                global: 0,
                local: 0,
            })
        );
        assert_eq!(total, before);
    }

    #[test]
    fn counter_aggregation_commits_every_field() {
        let mut total = L1Counters::for_arch(ArchId::Sm86);
        total.hits = 1;
        total.misses = 2;
        total.merges = 3;
        total.stall_events = 4;
        total.line_fills = 5;
        total.line_alloc_misses = 6;
        total.sector_resident_misses = 7;
        total.bypass_sectors = 8;
        total.bypass_atom_sectors = 9;
        total.untagged_sectors = 10;
        total.store_sectors = 11;
        total.store_hits = 12;
        total.store_tag_hits = 13;
        total = total.with_unified_origin(MemoryOrigin::Global);

        let mut addend = L1Counters::for_arch(ArchId::Sm86);
        addend.hits = 20;
        addend.misses = 21;
        addend.merges = 22;
        addend.stall_events = 23;
        addend.line_fills = 24;
        addend.line_alloc_misses = 25;
        addend.sector_resident_misses = 26;
        addend.bypass_sectors = 27;
        addend.bypass_atom_sectors = 28;
        addend.untagged_sectors = 29;
        addend.store_sectors = 30;
        addend.store_hits = 31;
        addend.store_tag_hits = 32;
        addend = addend.with_unified_origin(MemoryOrigin::Local);

        total.add_checked(&addend).unwrap();
        assert_eq!(total.arch(), Some(ArchId::Sm86));
        assert_eq!(total.hits, 21);
        assert_eq!(total.misses, 23);
        assert_eq!(total.merges, 25);
        assert_eq!(total.stall_events, 27);
        assert_eq!(total.line_fills, 29);
        assert_eq!(total.line_alloc_misses, 31);
        assert_eq!(total.sector_resident_misses, 33);
        assert_eq!(total.bypass_sectors, 35);
        assert_eq!(total.bypass_atom_sectors, 37);
        assert_eq!(total.untagged_sectors, 39);
        assert_eq!(total.store_sectors, 41);
        assert_eq!(total.store_hits, 43);
        assert_eq!(total.store_tag_hits, 45);
    }

    #[test]
    fn bypass_atom_subset_counter_tracks_only_the_atom_stream() {
        let mut l1 = cache(8, None);
        l1.count_bypass(MemoryOrigin::Global, 3, false).unwrap();
        l1.count_bypass(MemoryOrigin::Global, 2, true).unwrap();
        assert_eq!(l1.counters.bypass_sectors, 5);
        assert_eq!(l1.counters.bypass_atom_sectors, 2);
    }
}
