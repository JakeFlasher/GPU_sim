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
//! L1TEX lookup accounting remains separate from physical hit state: cache-all
//! LDGSTS merges into allocations from older warp instructions are observed as
//! lookup hits, while same-instruction sector coalescing remains a miss.

use std::collections::{BTreeMap, HashMap};
use std::num::NonZeroU32;

use sim_arch::{ArchId, L1WriteLookupWindow};
use sim_semantics::{
    L1BoundaryState, L1CounterState, L1FillDueState, L1LineFillState, L1SectorFillState,
    TYPED_STATE_SCHEMA_VERSION,
};

use crate::MemoryOrigin;
use crate::SectorPayloadV1;

pub const SECTOR_BYTES: u32 = 32;
pub const SECTORS_PER_LINE: u32 = 4;
pub const LINE_BYTES: u32 = SECTOR_BYTES * SECTORS_PER_LINE;

/// Explicit store-state policy for the bounded L1 model. The default retains
/// the existing write-through/no-allocate hypothesis. `ResidentHitWriteBack`
/// is a separate experiment: a store marks only an already-valid resident
/// sector dirty, so it never invents a write-allocate fill or a missing value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum L1StorePolicy {
    #[default]
    WriteThroughNoAllocate,
    ResidentHitWriteBack,
}

/// Configuration (card-resolved by `pchase_card` for SM86, or an explicit
/// effective SM120 observation). Latencies are integer cycles — the caller
/// ceils fractional source observations. SM120 remains opt-in: accepting its
/// typed configuration here does not invent geometry or wire flat replay to a
/// guessed cache model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

impl L1Config {
    /// Explicit SM120 fully-associative LRU scenario, not physical geometry.
    /// Capacity, timing and miss concurrency are caller-supplied hypotheses.
    /// `None` disables MSHR pressure; it does not represent an unknown count.
    pub const fn effective_sm120(
        line_capacity: u64,
        mshr_count: Option<u32>,
        hit_latency: u64,
        miss_latency: u64,
    ) -> Self {
        Self {
            arch: ArchId::Sm120,
            line_capacity,
            mshr_count,
            hit_latency,
            miss_latency,
        }
    }
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
    #[error("L1 write-lookup window capability is unavailable: {0}")]
    WriteLookupWindowCapability(#[from] sim_arch::CapabilityError),
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum L1AccessError {
    #[error(transparent)]
    Counter(#[from] L1CounterError),
    #[error("L1 write-lookup window capability is unavailable: {0}")]
    WriteLookupWindowCapability(#[from] sim_arch::CapabilityError),
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
    #[error("L1 line {line} sector mask {sector_mask:#06b} is invalid")]
    InvalidationMaskInvalid { line: i64, sector_mask: u8 },
    #[error("L1 line {line} invalidation overlaps a pending fill")]
    InvalidationPendingFill { line: i64 },
    #[error("L1-wide invalidation overlaps one or more pending fills")]
    InvalidationAllPendingFill,
    #[error("L1 line {line} has dirty sectors {dirty_mask:#06b}; flush them before invalidation")]
    DirtyResidencyBlocksInvalidation { line: i64, dirty_mask: u8 },
    #[error("L1 line {line} has dirty sectors {dirty_mask:#06b}; flush them before eviction")]
    DirtyResidencyBlocksEviction { line: i64, dirty_mask: u8 },
    #[error("L1 line {line} is not resident")]
    DirtyLineNotResident { line: i64 },
    #[error(
        "L1 line {line} dirty mask {dirty_mask:#06b} is not a resident-sector subset {valid_mask:#06b}"
    )]
    DirtyMaskInvalid {
        line: i64,
        dirty_mask: u8,
        valid_mask: u8,
    },
    #[error(
        "L1 dirty flush plan is stale for line {line}: expected {expected:#06b}, current {current:#06b}"
    )]
    DirtyFlushStale {
        line: i64,
        expected: u8,
        current: u8,
    },
    #[error("L1 line {line} sector {sector} has no complete dirty payload")]
    DirtyPayloadMissing { line: i64, sector: u8 },
    #[error("L1 payload admission plan is stale for line {line} sector {sector}")]
    DirtyPayloadAdmissionStale { line: i64, sector: u8 },
    #[error("L1 payload residency witness is missing for line {line}")]
    DirtyPayloadResidencyWitnessMissing { line: i64 },
    #[error("L1 payload residency witness is duplicated for line {line}")]
    DirtyPayloadResidencyWitnessDuplicate { line: i64 },
    #[error("L1 payload residency witness mismatch for line {line}: {field}")]
    DirtyPayloadResidencyWitnessMismatch { line: i64, field: &'static str },
    #[error(transparent)]
    DirtyPayload(#[from] crate::PayloadError),
}

/// One L1 line's dirty sectors.  Dirty state is an explicit opt-in seam;
/// existing native replay remains write-through/no-allocate until a validated
/// store policy and protocol boundary are supplied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L1DirtyLine {
    pub line_id: i64,
    pub dirty_mask: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct L1DirtyFlushPlan {
    lines: Box<[L1DirtyLine]>,
}

impl L1DirtyFlushPlan {
    pub fn lines(&self) -> &[L1DirtyLine] {
        &self.lines
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct L1DirtyPayloadExpectation {
    line: i64,
    sector: u8,
    current: Option<SectorPayloadV1>,
    valid_mask: u8,
    dirty_mask: u8,
    generation: Option<u32>,
}

/// Transactional admission of complete value-bearing dirty sectors. The
/// caller must already possess a residency/allocation witness; this plan
/// checks the current L1 residency and preserves the payload allocation
/// identity, but it does not infer a hardware allocation from a line tag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct L1DirtyPayloadAdmissionPlan {
    payloads: Box<[SectorPayloadV1]>,
    expected: Box<[L1DirtyPayloadExpectation]>,
}

/// External witness required before value-bearing payloads may be associated
/// with a resident L1 allocation. `resident_generation` is the L1 allocation
/// generation observed by the caller; allocation identity/generation belongs
/// to the value producer. Neither side is inferred from a line tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L1PayloadResidencyWitness {
    pub line_id: i64,
    pub allocation_id: u64,
    pub allocation_generation: u64,
    pub resident_generation: u32,
    pub resident_sector_mask: u8,
}

impl L1DirtyPayloadAdmissionPlan {
    pub fn payloads(&self) -> &[SectorPayloadV1] {
        &self.payloads
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L1InvalidationResult {
    pub line_id: i64,
    pub requested_mask: u8,
    pub invalidated_mask: u8,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct L1DirtyResidency {
    pub lines: u64,
    pub sectors: u64,
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
    pub reduction_sectors: u64,
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
            reduction_sectors: 0,
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

/// Opaque identity of one logical memory request's L1 lookup batch. A pending
/// fill whose allocation sequence predates this token is an NCU lookup hit;
/// merges into a fill created by this same request are sector coalescing and
/// remain lookup misses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L1LookupBatch {
    first_fill_sequence: u64,
    pending_merges_are_hits: bool,
}

/// One origin's accounting lane. It mirrors the unified fields in
/// [`L1Counters`] but carries no architecture of its own; provenance remains
/// bound once on the enclosing physical-cache snapshot.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct L1OriginCounters {
    pub hits: u64,
    pub misses: u64,
    pub merges: u64,
    /// Misses that merged into a fill allocated by an earlier logical memory
    /// request. Same-request sector coalescing is deliberately excluded.
    pub pending_lookup_hits: u64,
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
    /// Trace-topology lookup hits for write-class sectors. This observer uses
    /// 64 B lookup allocation within each ideal L1TEX wavefront and does not
    /// mutate cache tags, fills, replacement, or timing.
    pub write_lookup_hits: u64,
    /// RED/REDG sectors and sector-valid hits, kept separate from ordinary STG.
    pub reduction_sectors: u64,
    pub reduction_hits: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct L1Counters {
    /// Architecture provenance bound by the owning [`L1Cache`]. `None` is
    /// reserved for synthetic/unbound snapshots and is rejected at emission
    /// or aggregation boundaries.
    arch: Option<ArchId>,
    /// Ported counters (modeled, manifest-eligible accesses only).
    pub hits: u64,
    pub misses: u64,
    pub merges: u64,
    pub pending_lookup_hits: u64,
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
    pub write_lookup_hits: u64,
    /// RED/REDG sectors and sector-valid hits. They contribute to the unified
    /// L1TEX observer but never to the plain global-store surface.
    pub reduction_sectors: u64,
    pub reduction_hits: u64,
    /// Accounting provenance only. Cache tags, fills, MSHRs, recency, and
    /// replacement remain unified physical state.
    by_origin: [L1OriginCounters; 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L1CounterField {
    Hits,
    Misses,
    Merges,
    PendingLookupHits,
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
    WriteLookupHits,
    ReductionSectors,
    ReductionHits,
}

impl std::fmt::Display for L1CounterField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Hits => "hits",
            Self::Misses => "misses",
            Self::Merges => "merges",
            Self::PendingLookupHits => "pending_lookup_hits",
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
            Self::WriteLookupHits => "write_lookup_hits",
            Self::ReductionSectors => "reduction_sectors",
            Self::ReductionHits => "reduction_hits",
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

// Keep the preflight projection in one fixed-width representation.  A replay
// batch commonly contains many small budgets from one warp instruction; the
// old implementation copied the complete counter snapshot once per budget.
// Aggregating these deltas by origin preserves the same checked arithmetic and
// provenance lanes while reducing that work to at most two projections.
const PREFLIGHT_COUNTER_FIELDS: [L1CounterField; 17] = [
    L1CounterField::Hits,
    L1CounterField::Misses,
    L1CounterField::Merges,
    L1CounterField::PendingLookupHits,
    L1CounterField::StallEvents,
    L1CounterField::LineFills,
    L1CounterField::LineAllocMisses,
    L1CounterField::SectorResidentMisses,
    L1CounterField::BypassSectors,
    L1CounterField::BypassAtomSectors,
    L1CounterField::UntaggedSectors,
    L1CounterField::StoreSectors,
    L1CounterField::StoreHits,
    L1CounterField::StoreTagHits,
    L1CounterField::WriteLookupHits,
    L1CounterField::ReductionSectors,
    L1CounterField::ReductionHits,
];

fn preflight_budget_deltas(
    budget: &L1MutationBudget,
) -> Result<[(L1CounterField, u64); 17], L1AccessError> {
    let write_lookup_hits = budget
        .store_sectors
        .checked_add(budget.reduction_sectors)
        .ok_or(L1AccessError::Counter(L1CounterError::Overflow {
            field: L1CounterField::WriteLookupHits,
        }))?;
    Ok([
        (L1CounterField::Hits, budget.eligible_sectors),
        (L1CounterField::Misses, budget.eligible_sectors),
        (L1CounterField::Merges, budget.eligible_sectors),
        (L1CounterField::PendingLookupHits, budget.eligible_sectors),
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
        (L1CounterField::WriteLookupHits, write_lookup_hits),
        (L1CounterField::ReductionSectors, budget.reduction_sectors),
        (L1CounterField::ReductionHits, budget.reduction_sectors),
    ])
}

fn preflight_values_deltas(values: [u64; 17]) -> [(L1CounterField, u64); 17] {
    std::array::from_fn(|index| (PREFLIGHT_COUNTER_FIELDS[index], values[index]))
}

fn accumulate_preflight_values(
    values: &mut [u64; 17],
    budget: &L1MutationBudget,
) -> Result<(), L1AccessError> {
    for (index, (field, delta)) in preflight_budget_deltas(budget)?.into_iter().enumerate() {
        values[index] = values[index]
            .checked_add(delta)
            .ok_or(L1AccessError::Counter(L1CounterError::Overflow { field }))?;
    }
    Ok(())
}

fn origin_counter_mut(counters: &mut L1OriginCounters, field: L1CounterField) -> &mut u64 {
    match field {
        L1CounterField::Hits => &mut counters.hits,
        L1CounterField::Misses => &mut counters.misses,
        L1CounterField::Merges => &mut counters.merges,
        L1CounterField::PendingLookupHits => &mut counters.pending_lookup_hits,
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
        L1CounterField::WriteLookupHits => &mut counters.write_lookup_hits,
        L1CounterField::ReductionSectors => &mut counters.reduction_sectors,
        L1CounterField::ReductionHits => &mut counters.reduction_hits,
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

fn origin_deltas(counters: &L1OriginCounters) -> [(L1CounterField, u64); 17] {
    [
        (L1CounterField::Hits, counters.hits),
        (L1CounterField::Misses, counters.misses),
        (L1CounterField::Merges, counters.merges),
        (
            L1CounterField::PendingLookupHits,
            counters.pending_lookup_hits,
        ),
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
        (L1CounterField::WriteLookupHits, counters.write_lookup_hits),
        (L1CounterField::ReductionSectors, counters.reduction_sectors),
        (L1CounterField::ReductionHits, counters.reduction_hits),
    ]
}

fn unified_deltas(counters: &L1Counters) -> [(L1CounterField, u64); 17] {
    [
        (L1CounterField::Hits, counters.hits),
        (L1CounterField::Misses, counters.misses),
        (L1CounterField::Merges, counters.merges),
        (
            L1CounterField::PendingLookupHits,
            counters.pending_lookup_hits,
        ),
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
        (L1CounterField::WriteLookupHits, counters.write_lookup_hits),
        (L1CounterField::ReductionSectors, counters.reduction_sectors),
        (L1CounterField::ReductionHits, counters.reduction_hits),
    ]
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
            pending_lookup_hits: self.pending_lookup_hits,
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
            write_lookup_hits: self.write_lookup_hits,
            reduction_sectors: self.reduction_sectors,
            reduction_hits: self.reduction_hits,
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
                L1CounterField::PendingLookupHits => &mut next.pending_lookup_hits,
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
                L1CounterField::WriteLookupHits => &mut next.write_lookup_hits,
                L1CounterField::ReductionSectors => &mut next.reduction_sectors,
                L1CounterField::ReductionHits => &mut next.reduction_hits,
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
        let pending_lookup_hits = checked_counter_add(
            L1CounterField::PendingLookupHits,
            self.pending_lookup_hits,
            other.pending_lookup_hits,
        )?;
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
        let write_lookup_hits = checked_counter_add(
            L1CounterField::WriteLookupHits,
            self.write_lookup_hits,
            other.write_lookup_hits,
        )?;
        let reduction_sectors = checked_counter_add(
            L1CounterField::ReductionSectors,
            self.reduction_sectors,
            other.reduction_sectors,
        )?;
        let reduction_hits = checked_counter_add(
            L1CounterField::ReductionHits,
            self.reduction_hits,
            other.reduction_hits,
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
            pending_lookup_hits,
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
            write_lookup_hits,
            reduction_sectors,
            reduction_hits,
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
        generation: u32,
    },
    /// Merged into an in-flight same-sector fetch.
    SectorMerge {
        pending: FillDue,
        generation: u32,
    },
    Stalled,
    MissLineAlloc {
        pending: FillDue,
        generation: u32,
    },
    MissSectorResident {
        pending: FillDue,
        generation: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WriteAccounting {
    Store,
    Reduction,
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
    /// A line-allocation fill. `generation` is the fill-sequence identity of
    /// the allocation, so a delayed response from an older same-tag
    /// allocation cannot complete a newer one.
    Line { line: i64, generation: u32 },
    /// A resident-sector fill. The generation identifies the line allocation
    /// that owned the sector when the request was issued.
    Sector {
        line: i64,
        sector: u8,
        generation: u32,
    },
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

/// Observer-only tag window for Ampere L1TEX write lookups.  It is separate
/// from the physical read-state cache: the latter remains the validated
/// fully-associative replay approximation, while this state models the
/// write-allocate request stream used by the NCU `op_st_lookup_*` counters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct L1WriteLookupObserver {
    config: L1WriteLookupWindow,
    /// Each set is kept in LRU-to-MRU order.  A store instruction evaluates
    /// all sectors of a line against the pre-request tag, then touches it once
    /// (same-request sector coalescing therefore remains a miss).
    sets: Vec<Vec<i64>>,
}

/// Fallible-free state transition prepared for one write-observer batch.  The
/// plan clones only sets touched by the batch (rather than the complete
/// observer), so callers can defer publication across a larger instruction
/// transaction without turning every store into an O(total-set-count) clone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct L1WriteLookupObserverPlan {
    hits: u64,
    sets: Vec<(usize, Vec<i64>)>,
}

impl L1WriteLookupObserverPlan {
    pub const fn hits(&self) -> u64 {
        self.hits
    }
}

impl L1WriteLookupObserver {
    pub fn for_arch(arch: ArchId) -> Result<Self, L1AccessError> {
        let config = arch
            .l1_write_lookup_window()
            .require("l1_write_lookup_window")?;
        if config.set_count == 0 || config.tag_slots == 0 {
            return Err(L1AccessError::WriteLookupWindowCapability(
                sim_arch::CapabilityError::ResearchGap {
                    arch,
                    capability: "l1_write_lookup_window",
                    reason: "write-lookup window has zero sets or tag slots",
                },
            ));
        }
        Ok(Self {
            config,
            sets: (0..config.set_count as usize).map(|_| Vec::new()).collect(),
        })
    }

    fn new(config: L1WriteLookupWindow) -> Self {
        Self {
            config,
            sets: (0..config.set_count as usize).map(|_| Vec::new()).collect(),
        }
    }

    pub fn plan_batch(&self, sectors: &[i64]) -> Result<L1WriteLookupObserverPlan, L1AccessError> {
        let mut groups: Vec<(i64, u8)> = Vec::new();
        let mut group_indices: HashMap<i64, usize> = HashMap::new();
        for &sector in sectors {
            let line = sector >> 2;
            let bit = 1_u8 << (sector & 3) as u8;
            if let Some(&index) = group_indices.get(&line) {
                groups[index].1 |= bit;
            } else {
                group_indices.insert(line, groups.len());
                groups.push((line, bit));
            }
        }
        let mut hits = 0_u64;
        let mut staged_sets: Vec<(usize, Vec<i64>)> = Vec::new();
        let mut staged_set_indices: HashMap<usize, usize> = HashMap::new();
        for (line, mask) in groups {
            let set_count = i64::from(self.config.set_count);
            let set_index = line.rem_euclid(set_count) as usize;
            let staged_index = if let Some(&index) = staged_set_indices.get(&set_index) {
                index
            } else {
                let index = staged_sets.len();
                staged_set_indices.insert(set_index, index);
                staged_sets.push((set_index, self.sets[set_index].clone()));
                index
            };
            let set = &mut staged_sets[staged_index].1;
            let tag_hit = set.contains(&line);
            if tag_hit {
                hits = hits.checked_add(u64::from(mask.count_ones())).ok_or(
                    L1AccessError::Counter(L1CounterError::Overflow {
                        field: L1CounterField::WriteLookupHits,
                    }),
                )?;
                if let Some(position) = set.iter().position(|candidate| *candidate == line) {
                    set.remove(position);
                }
            }
            if !tag_hit && set.len() >= self.config.tag_slots as usize {
                set.remove(0);
            }
            set.push(line);
        }
        Ok(L1WriteLookupObserverPlan {
            hits,
            sets: staged_sets,
        })
    }

    pub fn commit_plan(&mut self, plan: L1WriteLookupObserverPlan) {
        for (set_index, set) in plan.sets {
            self.sets[set_index] = set;
        }
    }

    pub fn observe_batch(&mut self, sectors: &[i64]) -> Result<u64, L1AccessError> {
        let plan = self.plan_batch(sectors)?;
        let hits = plan.hits;
        self.commit_plan(plan);
        Ok(hits)
    }
}

#[derive(Debug, Clone)]
pub struct L1Cache {
    cfg: L1Config,
    store_policy: L1StorePolicy,
    /// Fully-associative LRU line pool: monotonic recency stamps.
    by_stamp: BTreeMap<u64, i64>,
    stamp_of: HashMap<i64, u64>,
    next_stamp: NextSequence,
    /// Per-line fetched-sector bitmask (bit = sector index in line, 0..4).
    /// Also accumulates for in-flight line allocations (the fill services
    /// every merged sector).
    sectors: HashMap<i64, u8>,
    /// Explicit opt-in dirty-sector state. The default replay policy does not
    /// mark stores dirty; the resident-hit experiment uses this map and the
    /// transactional flush API below.
    dirty: HashMap<i64, u8>,
    /// Optional known-byte payloads for the strict value-bearing flush lane.
    /// Ordinary replay remains mask-only until a producer supplies these.
    dirty_payloads: HashMap<(i64, u8), SectorPayloadV1>,
    /// In-flight fills: value = (due state, issue_seq) — the seq keeps
    /// same-timestamp same-kind timed fills in issue order like the Python
    /// stable sort. Event fills remain explicitly untimed. Sector fills are
    /// keyed by line, sector, and allocation generation so an old response
    /// remains representable while a same-tag reallocation has already
    /// issued a new request for that sector.
    line_fills: HashMap<i64, (FillDue, u64, u32)>,
    sector_fills: HashMap<(i64, u8, u32), (FillDue, u64)>,
    /// Allocation-generation witnesses for event-mode fills.  A sector fill
    /// can outlive the resident line that spawned it; retaining the line's
    /// allocation sequence lets completion reject an old response after the
    /// same tag has been evicted and reallocated (the L2 model already uses
    /// the analogous generation guard).
    line_generations: HashMap<i64, u32>,
    fill_seq: NextSequence,
    /// Active logical request boundary for the lookup observer only. This does
    /// not participate in tags, MSHR ownership, replacement, or timing.
    active_lookup_batch: Option<L1LookupBatch>,
    /// Exact earliest timed in-flight fill. `None` means there is no timed
    /// fill (empty or event-driven), while `Some(u64::MAX)` is a real due
    /// cycle. `apply_fills` early-outs when `now` has not reached it.
    min_fill_due: Option<u64>,
    /// Latched fill-timing discipline (see [`FillTiming`]).
    fill_timing: Option<FillTiming>,
    /// Lazily allocated observer-only write tag window.  Keeping it boxed and
    /// optional leaves cold/read-only replays on the compact hot path.
    write_lookup_observer: Option<Box<L1WriteLookupObserver>>,
    pub counters: L1Counters,
}

impl L1Cache {
    pub fn new(cfg: L1Config) -> Result<Self, L1ConfigError> {
        if !matches!(cfg.arch, ArchId::Sm86 | ArchId::Sm120) {
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
            store_policy: L1StorePolicy::WriteThroughNoAllocate,
            by_stamp: BTreeMap::new(),
            stamp_of: HashMap::new(),
            next_stamp: NextSequence::Available(0),
            sectors: HashMap::new(),
            dirty: HashMap::new(),
            dirty_payloads: HashMap::new(),
            line_fills: HashMap::new(),
            sector_fills: HashMap::new(),
            line_generations: HashMap::new(),
            fill_seq: NextSequence::Available(0),
            active_lookup_batch: None,
            min_fill_due: None,
            fill_timing: None,
            write_lookup_observer: None,
            counters: L1Counters::for_arch(arch),
        })
    }

    /// Select the explicit resident-hit writeback experiment. This does not
    /// alter cache geometry or fill timing and is intentionally opt-in.
    pub fn set_store_policy(&mut self, policy: L1StorePolicy) {
        self.store_policy = policy;
    }

    pub const fn store_policy(&self) -> L1StorePolicy {
        self.store_policy
    }

    /// Return the model-local allocation generation and resident-sector mask
    /// for one line. This is the only residency state the cache model can
    /// supply to a value-bearing admission witness; it is deliberately not a
    /// physical allocation/owner identity and must be joined with an
    /// independently observed producer allocation before payload mutation.
    pub fn resident_payload_state(&self, line: i64) -> Option<(u32, u8)> {
        let generation = self.line_generations.get(&line).copied()?;
        let resident_sector_mask = self.sectors.get(&line).copied().unwrap_or(0);
        self.line_resident(line)
            .then_some((generation, resident_sector_mask))
    }

    pub fn arch(&self) -> ArchId {
        self.cfg.arch
    }

    /// Stable structural projection for bounded differential diagnostics. All
    /// unordered maps are sorted before formatting; this is an observation
    /// surface, not an authority token or a production fast path.
    pub fn boundary_state(&self) -> String {
        let stamp_of = self
            .stamp_of
            .iter()
            .map(|(&line, &stamp)| (line, stamp))
            .collect::<Vec<_>>();
        let mut stamp_of = stamp_of;
        stamp_of.sort_unstable_by_key(|entry| entry.0);
        let sectors = self
            .sectors
            .iter()
            .map(|(&line, &mask)| (line, mask))
            .collect::<Vec<_>>();
        let mut sectors = sectors;
        sectors.sort_unstable_by_key(|entry| entry.0);
        let mut dirty = self
            .dirty
            .iter()
            .map(|(&line, &mask)| (line, mask))
            .collect::<Vec<_>>();
        dirty.sort_unstable_by_key(|entry| entry.0);
        let mut line_fills = self
            .line_fills
            .iter()
            .map(|(&line, &(due, seq, generation))| (line, format!("{due:?}"), seq, generation))
            .collect::<Vec<_>>();
        line_fills.sort_unstable_by_key(|entry| entry.0);
        let mut sector_fills = self
            .sector_fills
            .iter()
            .map(|(&(line, sector, generation), &(due, seq))| {
                (line, sector, generation, format!("{due:?}"), seq)
            })
            .collect::<Vec<_>>();
        sector_fills.sort_unstable_by_key(|entry| (entry.0, entry.1, entry.2));
        let mut generations = self
            .line_generations
            .iter()
            .map(|(&line, &generation)| (line, generation))
            .collect::<Vec<_>>();
        generations.sort_unstable_by_key(|entry| entry.0);
        format!(
            "cfg={:?};by_stamp={:?};stamp_of={:?};sectors={:?};dirty={:?};line_fills={:?};sector_fills={:?};generations={:?};next_stamp={:?};fill_seq={:?};active_lookup={:?};min_fill_due={:?};fill_timing={:?};counters={:?};write_observer={:?}",
            self.cfg,
            self.by_stamp,
            stamp_of,
            sectors,
            dirty,
            line_fills,
            sector_fills,
            generations,
            self.next_stamp,
            self.fill_seq,
            self.active_lookup_batch,
            self.min_fill_due,
            self.fill_timing,
            self.counters,
            self.write_lookup_observer,
        )
    }

    /// Deterministic typed projection of the physical read-state L1. The
    /// projection carries fill generations and pending maps directly; no
    /// debug formatting participates in the state contract.
    pub fn typed_boundary_state(&self) -> L1BoundaryState {
        let mut line_order = self
            .by_stamp
            .iter()
            .map(|(&stamp, &line)| (stamp, line))
            .collect::<Vec<_>>();
        line_order.sort_unstable();
        let mut stamp_of = self
            .stamp_of
            .iter()
            .map(|(&line, &stamp)| (line, stamp))
            .collect::<Vec<_>>();
        stamp_of.sort_unstable();
        let mut sectors = self
            .sectors
            .iter()
            .map(|(&line, &mask)| (line, mask))
            .collect::<Vec<_>>();
        sectors.sort_unstable();
        let due = |due: FillDue| match due {
            FillDue::Timed(cycle) => L1FillDueState::Timed { cycle },
            FillDue::Untimed => L1FillDueState::Untimed,
        };
        let mut line_fills = self
            .line_fills
            .iter()
            .map(
                |(&line, &(fill_due, sequence, generation))| L1LineFillState {
                    line,
                    due: due(fill_due),
                    sequence,
                    generation,
                },
            )
            .collect::<Vec<_>>();
        line_fills.sort_by_key(|fill| (fill.line, fill.generation, fill.sequence));
        let mut sector_fills = self
            .sector_fills
            .iter()
            .map(
                |(&(line, sector, generation), &(fill_due, sequence))| L1SectorFillState {
                    line,
                    sector,
                    generation,
                    due: due(fill_due),
                    sequence,
                },
            )
            .collect::<Vec<_>>();
        sector_fills.sort_by_key(|fill| (fill.line, fill.sector, fill.generation, fill.sequence));
        let mut line_generations = self
            .line_generations
            .iter()
            .map(|(&line, &generation)| (line, generation))
            .collect::<Vec<_>>();
        line_generations.sort_unstable();
        let next_stamp = match self.next_stamp {
            NextSequence::Available(value) => Some(value),
            NextSequence::Exhausted => None,
        };
        let fill_sequence = match self.fill_seq {
            NextSequence::Available(value) => Some(value),
            NextSequence::Exhausted => None,
        };
        let active_lookup_batch = self
            .active_lookup_batch
            .map(|batch| (batch.first_fill_sequence, batch.pending_merges_are_hits));
        let fill_timing = self.fill_timing.map(|timing| match timing {
            FillTiming::Constant => "constant".to_owned(),
            FillTiming::Event => "event".to_owned(),
        });
        let counters = self.counters;
        L1BoundaryState {
            schema_version: TYPED_STATE_SCHEMA_VERSION,
            architecture: match self.cfg.arch {
                ArchId::Sm86 => "sm_86".to_owned(),
                ArchId::Sm120 => "sm_120".to_owned(),
            },
            line_capacity: self.cfg.line_capacity,
            mshr_count: self.cfg.mshr_count,
            hit_latency: self.cfg.hit_latency,
            miss_latency: self.cfg.miss_latency,
            line_order,
            stamp_of,
            sectors,
            line_fills,
            sector_fills,
            line_generations,
            next_stamp,
            fill_sequence,
            active_lookup_batch,
            min_fill_due: self.min_fill_due,
            fill_timing,
            counters: L1CounterState {
                hits: counters.hits,
                misses: counters.misses,
                merges: counters.merges,
                pending_lookup_hits: counters.pending_lookup_hits,
                stall_events: counters.stall_events,
                line_fills: counters.line_fills,
                line_alloc_misses: counters.line_alloc_misses,
                sector_resident_misses: counters.sector_resident_misses,
                bypass_sectors: counters.bypass_sectors,
                bypass_atom_sectors: counters.bypass_atom_sectors,
                untagged_sectors: counters.untagged_sectors,
                store_sectors: counters.store_sectors,
                store_hits: counters.store_hits,
                store_tag_hits: counters.store_tag_hits,
                write_lookup_hits: counters.write_lookup_hits,
                reduction_sectors: counters.reduction_sectors,
                reduction_hits: counters.reduction_hits,
            },
            write_observer_present: self.write_lookup_observer.is_some(),
        }
    }

    fn mshr_in_use(&self) -> usize {
        // A late sector child shares only the matching parent allocation's
        // MSHR slot. A zombie child from an evicted generation remains an
        // independent entry while a same-tag line is reallocated.
        self.line_fills.len()
            + self
                .sector_fills
                .keys()
                .filter(|(line, _, generation)| {
                    self.line_fills
                        .get(line)
                        .is_none_or(|&(_, _, parent_generation)| parent_generation != *generation)
                })
                .count()
    }

    /// Access-local admission may raise a finite generic approximation to an
    /// independently measured floor without rewriting the cache configuration.
    /// An unlimited ablation remains unlimited.
    fn effective_mshr_count(&self, floor: Option<NonZeroU32>) -> Option<usize> {
        self.cfg.mshr_count.map(|configured| {
            let effective = floor.map_or(configured, |floor| configured.max(floor.get()));
            effective as usize
        })
    }

    fn constant_access_may_retry(
        &self,
        sector_id: i64,
        now: u64,
        floor: Option<NonZeroU32>,
    ) -> bool {
        // A full MSHR set only requires the shadow transaction on the miss
        // path. Hits and merges have no retry loop, so cloning the complete
        // cache for them would turn transient pressure into avoidable work.
        // A fill due at `now` is a mutation boundary: apply_fills runs before
        // classification and may turn a hit into a miss through eviction, so
        // retain the shadow path until that transition is replayed exactly.
        if self.min_fill_due.is_some_and(|due| due <= now) {
            return self
                .effective_mshr_count(floor)
                .is_some_and(|capacity| self.mshr_in_use() >= capacity);
        }
        let line = sector_id >> 2;
        let sector = (sector_id & 3) as u8;
        if self.line_fills.contains_key(&line) {
            return false;
        }
        if let Some(&generation) = self.line_generations.get(&line)
            && self.sector_fills.contains_key(&(line, sector, generation))
        {
            return false;
        }
        if self.line_resident(line)
            && self
                .sectors
                .get(&line)
                .is_some_and(|mask| mask & (1_u8 << sector) != 0)
        {
            return false;
        }
        self.effective_mshr_count(floor)
            .is_some_and(|capacity| self.mshr_in_use() >= capacity)
    }

    /// Whether `fill` is still owned by this cache's event-mode MSHR state.
    /// Response handlers use this as a fail-closed preflight before consuming
    /// their exact response token or mutating any waiter accounting.
    pub fn is_fill_in_flight(&self, fill: FillKey) -> bool {
        match fill {
            FillKey::Line { line, generation } => self
                .line_fills
                .get(&line)
                .is_some_and(|&(_, _, current)| current == generation),
            FillKey::Sector {
                line,
                sector,
                generation,
            } => self.sector_fills.contains_key(&(line, sector, generation)),
        }
    }

    /// Whether completing this token releases an independently occupied MSHR
    /// slot. A deferred sector child shares its parent's matching generation
    /// slot until that parent returns, so a zombie child from an older
    /// same-tag allocation must not suppress retry drain.
    pub fn fill_releases_mshr(&self, fill: FillKey) -> bool {
        match fill {
            FillKey::Line { line, generation } => {
                !self
                    .sector_fills
                    .keys()
                    .any(|(child_line, _, child_generation)| {
                        *child_line == line && *child_generation == generation
                    })
            }
            FillKey::Sector {
                line, generation, ..
            } => self
                .line_fills
                .get(&line)
                .is_none_or(|&(_, _, parent_generation)| parent_generation != generation),
        }
    }

    /// Whether any generation of a line-allocation fill is currently in
    /// flight.  This narrow query is used only for validating the companion
    /// line-mask bookkeeping; completion itself must still carry the exact
    /// generation-bearing [`FillKey`].
    pub fn is_line_fill_in_flight(&self, line: i64) -> bool {
        self.line_fills.contains_key(&line)
    }

    /// Start an ordinary warp instruction's logical L1 lookup request. Pending
    /// merges remain misses for this observer. The returned token must be
    /// retained if an MSHR-stalled remainder is retried later.
    pub fn begin_lookup_batch(&mut self) -> L1LookupBatch {
        let batch = L1LookupBatch {
            first_fill_sequence: self.fill_seq.current(),
            pending_merges_are_hits: false,
        };
        self.active_lookup_batch = Some(batch);
        batch
    }

    /// Start a cache-all LDGSTS source lookup. Independent sm_86 calibration
    /// shows that a merge into an older pending allocation is counted as an
    /// L1TEX lookup hit for this path; same-instruction coalescing is not.
    pub fn begin_async_copy_lookup_batch(&mut self) -> L1LookupBatch {
        let batch = L1LookupBatch {
            first_fill_sequence: self.fill_seq.current(),
            pending_merges_are_hits: true,
        };
        self.active_lookup_batch = Some(batch);
        batch
    }

    /// Resume the observer boundary of a previously stalled logical request.
    /// Cache behavior is unchanged; only merge attribution consults the token.
    pub fn resume_lookup_batch(&mut self, batch: L1LookupBatch) {
        self.active_lookup_batch = Some(batch);
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

    /// Reserve a sector child for a line allocation that is already in flight.
    ///
    /// A same-line access arriving after the original emission batch cannot be
    /// folded into the queued line request.  It therefore owns an independent
    /// sector completion token, while retaining the parent's allocation
    /// generation. The child shares the parent line allocation's MSHR slot
    /// until that parent completes. The requested bit was optimistically
    /// added by the normal line-merge classifier; remove it until this child returns so a later
    /// access cannot observe data before its own L2/DRAM event.
    pub fn reserve_pending_line_sector(
        &mut self,
        fill: FillKey,
        sector: u8,
    ) -> Result<FillKey, L1AccessError> {
        self.preflight_latch(FillTiming::Event)?;
        let FillKey::Line { line, generation } = fill else {
            return Err(L1AccessError::FillNotInFlight { fill });
        };
        if sector >= 4
            || self.line_fills.get(&line).map(|&(_, _, current)| current) != Some(generation)
        {
            return Err(L1AccessError::FillNotInFlight { fill });
        }
        let key = (line, sector, generation);
        if self.sector_fills.contains_key(&key) {
            return Ok(FillKey::Sector {
                line,
                sector,
                generation,
            });
        }
        let (sequence, next_sequence) = self.fill_seq.allocate("fill")?;
        self.fill_seq = next_sequence;
        if let Some(mask) = self.sectors.get_mut(&line) {
            *mask &= !(1_u8 << sector);
            if *mask == 0 {
                self.sectors.remove(&line);
            }
        }
        self.sector_fills.insert(key, (FillDue::Untimed, sequence));
        Ok(FillKey::Sector {
            line,
            sector,
            generation,
        })
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

    /// Check recency headroom for fills that will be applied before one
    /// access-level MRU touch.  The ordinary mutation budget only covers the
    /// access itself; due fills are a separate lazy transition and can consume
    /// stamps first.  Count every due fill conservatively (sector fills may be
    /// dropped after an eviction), so a near-exhaustion path is redirected to
    /// the transactional shadow replay instead of publishing a partial fill.
    fn preflight_due_fill_recency(
        &self,
        now: u64,
        extra_touches: u64,
    ) -> Result<(), L1AccessError> {
        // The maintained minimum makes the ordinary no-due/future-fill path
        // constant time. Exact counting remains necessary only at a due-fill
        // boundary because an evicted sector fill may be dropped.
        if self.min_fill_due.is_none_or(|minimum| minimum > now) {
            return self.next_stamp.preflight("recency", extra_touches);
        }
        let due_fills = self
            .line_fills
            .values()
            .filter(|&&(due, ..)| due.timed().is_some_and(|time| time <= now))
            .count()
            .checked_add(
                self.sector_fills
                    .values()
                    .filter(|&&(due, _)| due.timed().is_some_and(|time| time <= now))
                    .count(),
            )
            .and_then(|count| u64::try_from(count).ok())
            .ok_or(L1AccessError::SequenceOverflow {
                sequence: "recency",
                current: self.next_stamp.current(),
            })?;
        let uses = due_fills
            .checked_add(extra_touches)
            .ok_or(L1AccessError::SequenceOverflow {
                sequence: "recency",
                current: self.next_stamp.current(),
            })?;
        self.next_stamp.preflight("recency", uses)
    }

    fn apply_counter_deltas(
        &mut self,
        origin: MemoryOrigin,
        deltas: &[(L1CounterField, u64)],
    ) -> Result<(), L1AccessError> {
        self.counters = self.counters.add_fields_checked(origin, deltas)?;
        Ok(())
    }

    /// Prove conservative multi-access mutation budgets without changing cache
    /// state. Counter deltas are staged in caller order so one physical unified
    /// counter snapshot covers every explicit origin lane. Recency and fill
    /// sequences consume the combined physical budget exactly once.
    pub fn preflight_mutation_budgets(
        &self,
        budgets: &[L1MutationBudget],
    ) -> Result<(), L1AccessError> {
        let mut aggregated_counter_values = [[0_u64; 17]; 2];
        for budget in budgets {
            accumulate_preflight_values(
                &mut aggregated_counter_values[budget.origin.index()],
                budget,
            )?;
        }

        let mut counters = self.counters;
        for origin in MemoryOrigin::ALL {
            let values = aggregated_counter_values[origin.index()];
            if values.iter().any(|&value| value != 0) {
                counters = counters.add_fields_checked(origin, &preflight_values_deltas(values))?;
            }
        }

        let mut recency_touches = 0u64;
        let mut new_fills = 0u64;
        for budget in budgets {
            recency_touches = recency_touches.checked_add(budget.recency_touches).ok_or(
                L1AccessError::SequenceOverflow {
                    sequence: "recency",
                    current: self.next_stamp.current(),
                },
            )?;
            new_fills =
                new_fills
                    .checked_add(budget.new_fills)
                    .ok_or(L1AccessError::SequenceOverflow {
                        sequence: "fill",
                        current: self.fill_seq.current(),
                    })?;
        }
        let in_flight =
            u64::try_from(self.mshr_in_use()).map_err(|_| L1AccessError::SequenceOverflow {
                sequence: "recency",
                current: self.next_stamp.current(),
            })?;
        recency_touches =
            recency_touches
                .checked_add(in_flight)
                .ok_or(L1AccessError::SequenceOverflow {
                    sequence: "recency",
                    current: self.next_stamp.current(),
                })?;
        self.next_stamp.preflight("recency", recency_touches)?;
        self.fill_seq.preflight("fill", new_fills)?;
        // FillKey carries a compact u32 generation even though the internal
        // ordering sequence remains u64. A u64 sequence preflight alone is
        // insufficient at 2^32: a later `apply_fills(now)` may mutate this
        // cache before `access_core` discovers that the generation cannot be
        // represented. Reject (and let public wrappers use their shadow
        // transaction) before any due-fill mutation can leak through.
        self.preflight_fill_generation_headroom(new_fills)?;
        Ok(())
    }

    fn preflight_fill_generation_headroom(&self, new_fills: u64) -> Result<(), L1AccessError> {
        let Some(last_offset) = new_fills.checked_sub(1) else {
            return Ok(());
        };
        let current = self.fill_seq.current();
        let generation_limit = u64::from(u32::MAX);
        if current
            .checked_add(last_offset)
            .is_none_or(|last| last > generation_limit)
        {
            return Err(L1AccessError::SequenceOverflow {
                sequence: "fill generation",
                current,
            });
        }
        Ok(())
    }

    /// Prove one conservative mutation budget. Callers can fall back to exact
    /// shadow replay only when a near-limit budget is too conservative.
    pub fn preflight_mutation_budget(&self, budget: L1MutationBudget) -> Result<(), L1AccessError> {
        self.preflight_mutation_budgets(std::slice::from_ref(&budget))
    }

    fn earliest_fill(&self) -> Result<u64, L1AccessError> {
        self.line_fills
            .values()
            .map(|&(due, ..)| due)
            .chain(self.sector_fills.values().map(|&(due, ..)| due))
            .filter_map(FillDue::timed)
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
                .map(|&(due, ..)| due)
                .chain(self.sector_fills.values().map(|&(due, ..)| due))
                .filter_map(FillDue::timed)
                .filter(|&time| time > now)
                .min()
        };
        if let Some(retry_at) = retry_at {
            Self::checked_time_add("hit visibility", retry_at, self.cfg.hit_latency)?;
            Self::checked_time_add("miss visibility", retry_at, self.cfg.miss_latency)?;
        }
        self.preflight_due_fill_recency(now, 1)?;
        Ok(())
    }

    /// Conservative, allocation-free ready-time upper bound for an ordered batch
    /// of constant-mode sector accesses. Starting after the latest existing fill
    /// and charging one miss arm per selected sector covers fills created by an
    /// earlier access in the same caller transaction. `None` asks the caller to
    /// classify the batch exactly on a shadow cache.
    pub fn constant_batch_ready_upper_bound(&self, now: u64, sectors: u64) -> Option<u64> {
        if self.preflight_latch(FillTiming::Constant).is_err() {
            return None;
        }
        let hit_ready = Self::checked_time_add("hit visibility", now, self.cfg.hit_latency).ok()?;
        let miss_ready =
            Self::checked_time_add("miss visibility", now, self.cfg.miss_latency).ok()?;
        if sectors == 0 {
            return Some(hit_ready.max(miss_ready));
        }
        let start = self
            .line_fills
            .values()
            .map(|&(due, ..)| due)
            .chain(self.sector_fills.values().map(|&(due, ..)| due))
            .filter_map(FillDue::timed)
            .max()
            .unwrap_or(now)
            .max(now);
        self.cfg
            .miss_latency
            .checked_mul(sectors)
            .and_then(|delta| start.checked_add(delta))
            .map(|batch_miss_ready| batch_miss_ready.max(hit_ready))
    }

    /// Does the conservative constant-batch ready-time bound fit in u64?
    pub fn constant_batch_timing_has_headroom(&self, now: u64, sectors: u64) -> bool {
        self.constant_batch_ready_upper_bound(now, sectors)
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

    fn preflight_dirty_eviction(&self, additional_line: Option<i64>) -> Result<(), L1AccessError> {
        let capacity = usize::try_from(self.cfg.line_capacity).unwrap_or(usize::MAX);
        let additional = additional_line
            .filter(|line| !self.line_resident(*line))
            .map_or(0, |_| 1);
        let evict_count = self
            .by_stamp
            .len()
            .saturating_add(additional)
            .saturating_sub(capacity);
        for (&_stamp, &line) in self.by_stamp.iter().take(evict_count) {
            if let Some(&dirty_mask) = self.dirty.get(&line) {
                return Err(L1AccessError::DirtyResidencyBlocksEviction { line, dirty_mask });
            }
        }
        Ok(())
    }

    fn evict_beyond_capacity(&mut self) -> Result<(), L1AccessError> {
        self.preflight_dirty_eviction(None)?;
        while self.by_stamp.len() as u64 > self.cfg.line_capacity {
            let Some((&stamp, &line)) = self.by_stamp.first_key_value() else {
                break;
            };
            self.by_stamp.remove(&stamp);
            self.stamp_of.remove(&line);
            // Partial-line eviction: sector residency dies with the line.
            self.sectors.remove(&line);
            // Any later sector-fill response belongs to this old allocation;
            // removing the current witness makes same-tag reallocation an
            // explicit generation mismatch instead of allowing stale data to
            // resurrect into the new line.
            self.line_generations.remove(&line);
        }
        Ok(())
    }

    fn line_resident(&self, line: i64) -> bool {
        self.stamp_of.contains_key(&line)
    }

    fn validate_sector_mask(line: i64, sector_mask: u8) -> Result<(), L1AccessError> {
        if sector_mask == 0 || sector_mask & !(0b1111) != 0 {
            return Err(L1AccessError::InvalidationMaskInvalid { line, sector_mask });
        }
        Ok(())
    }

    /// Mark resident sectors dirty for an explicit write-policy experiment.
    /// The default native replay path never calls this method; stores retain
    /// their existing write-through/no-allocate hypothesis until a validated
    /// protocol supplies the call and a completion boundary.
    pub fn mark_dirty(&mut self, line: i64, sector_mask: u8) -> Result<(), L1AccessError> {
        Self::validate_sector_mask(line, sector_mask)?;
        let valid_mask = self.sectors.get(&line).copied().unwrap_or(0);
        if !self.line_resident(line) {
            return Err(L1AccessError::DirtyLineNotResident { line });
        }
        if sector_mask & !valid_mask != 0 {
            return Err(L1AccessError::DirtyMaskInvalid {
                line,
                dirty_mask: sector_mask,
                valid_mask,
            });
        }
        *self.dirty.entry(line).or_default() |= sector_mask;
        // A mask-only mutation carries no replacement bytes. Never retain an
        // older complete payload and present it as this store's value.
        for sector in 0..4_u8 {
            if sector_mask & (1_u8 << sector) != 0 {
                self.dirty_payloads.remove(&(line, sector));
            }
        }
        Ok(())
    }

    /// Mark one resident sector dirty with a complete known-byte payload.
    /// This is the value-bearing companion to the legacy mask-only method.
    pub fn mark_dirty_payload(&mut self, payload: SectorPayloadV1) -> Result<(), L1AccessError> {
        let line = payload.line_id();
        let sector = payload.sector();
        if !self.line_resident(line) {
            return Err(L1AccessError::DirtyLineNotResident { line });
        }
        let valid_mask = self.sectors.get(&line).copied().unwrap_or(0);
        let bit = 1_u8 << sector;
        if valid_mask & bit == 0 {
            return Err(L1AccessError::DirtyMaskInvalid {
                line,
                dirty_mask: bit,
                valid_mask,
            });
        }
        if !payload.is_fully_known() {
            return Err(L1AccessError::DirtyPayloadMissing { line, sector });
        }
        let payload = match self.dirty_payloads.get(&(line, sector)) {
            Some(previous) => previous.merge(payload)?,
            None => payload,
        };
        self.dirty
            .insert(line, self.dirty.get(&line).copied().unwrap_or(0) | bit);
        self.dirty_payloads.insert((line, sector), payload);
        Ok(())
    }

    /// Preflight a complete payload batch without mutating cache state.
    /// Partial records are coalesced by allocation identity and sequence;
    /// every resulting sector must be resident and fully known. A mask-only
    /// dirty sector is refused because a later sidecar cannot prove it is the
    /// same store generation.
    pub fn preview_dirty_payload_admission(
        &self,
        payloads: &[SectorPayloadV1],
    ) -> Result<L1DirtyPayloadAdmissionPlan, L1AccessError> {
        let complete = crate::coalesce_payloads(payloads.to_vec())?;
        let mut expected = Vec::with_capacity(complete.len());
        for payload in &complete {
            let line = payload.line_id();
            let sector = payload.sector();
            if !self.line_resident(line) {
                return Err(L1AccessError::DirtyLineNotResident { line });
            }
            let bit = 1_u8 << sector;
            let valid_mask = self.sectors.get(&line).copied().unwrap_or(0);
            if valid_mask & bit == 0 {
                return Err(L1AccessError::DirtyMaskInvalid {
                    line,
                    dirty_mask: bit,
                    valid_mask,
                });
            }
            let current = self.dirty_payloads.get(&(line, sector)).copied();
            let dirty_mask = self.dirty.get(&line).copied().unwrap_or(0);
            if dirty_mask & bit != 0 && current.is_none() {
                return Err(L1AccessError::DirtyPayloadMissing { line, sector });
            }
            if let Some(previous) = current {
                previous.merge(*payload)?;
            }
            expected.push(L1DirtyPayloadExpectation {
                line,
                sector,
                current,
                valid_mask,
                dirty_mask,
                generation: self.line_generations.get(&line).copied(),
            });
        }
        Ok(L1DirtyPayloadAdmissionPlan {
            payloads: complete.into_boxed_slice(),
            expected: expected.into_boxed_slice(),
        })
    }

    /// Admit payloads only after an external allocation/residency witness has
    /// been checked against the current L1 allocation generation. This is the
    /// native-facing contract; the ordinary sidecar path intentionally does
    /// not synthesize these witnesses.
    pub fn preview_dirty_payload_admission_with_witness(
        &self,
        payloads: &[SectorPayloadV1],
        witnesses: &[L1PayloadResidencyWitness],
    ) -> Result<L1DirtyPayloadAdmissionPlan, L1AccessError> {
        self.validate_payload_residency_witnesses(payloads, witnesses)?;
        self.preview_dirty_payload_admission(payloads)
    }

    fn validate_payload_residency_witnesses(
        &self,
        payloads: &[SectorPayloadV1],
        witnesses: &[L1PayloadResidencyWitness],
    ) -> Result<(), L1AccessError> {
        let mut by_line = BTreeMap::new();
        for witness in witnesses {
            if witness.allocation_id == 0 || witness.allocation_generation == 0 {
                return Err(L1AccessError::DirtyPayloadResidencyWitnessMismatch {
                    line: witness.line_id,
                    field: "allocation identity must be nonzero",
                });
            }
            if by_line.insert(witness.line_id, *witness).is_some() {
                return Err(L1AccessError::DirtyPayloadResidencyWitnessDuplicate {
                    line: witness.line_id,
                });
            }
        }
        for payload in payloads {
            let line = payload.line_id();
            let witness = by_line
                .get(&line)
                .ok_or(L1AccessError::DirtyPayloadResidencyWitnessMissing { line })?;
            if witness.allocation_id != payload.allocation_id()
                || witness.allocation_generation != payload.allocation_generation()
            {
                return Err(L1AccessError::DirtyPayloadResidencyWitnessMismatch {
                    line,
                    field: "payload allocation identity",
                });
            }
            let actual_generation = self.line_generations.get(&line).copied();
            if actual_generation != Some(witness.resident_generation) {
                return Err(L1AccessError::DirtyPayloadResidencyWitnessMismatch {
                    line,
                    field: "resident allocation generation",
                });
            }
            let actual_mask = self.sectors.get(&line).copied().unwrap_or(0);
            if witness.resident_sector_mask & !actual_mask != 0 {
                return Err(L1AccessError::DirtyPayloadResidencyWitnessMismatch {
                    line,
                    field: "resident sector mask",
                });
            }
            if witness.resident_sector_mask & (1_u8 << payload.sector()) == 0 {
                return Err(L1AccessError::DirtyPayloadResidencyWitnessMismatch {
                    line,
                    field: "payload sector not witnessed resident",
                });
            }
        }
        Ok(())
    }

    /// Attach complete bytes to already-dirty resident sectors after the SM
    /// has checked the exact dynamic writer and its accept time. Unlike the
    /// generic admission API this requires, rather than invents, dirty state.
    /// It cannot validate a dynamic instruction itself; that belongs to the
    /// issue actor. Any failure leaves the whole cache unchanged.
    pub fn attach_observed_store_payloads(
        &mut self,
        payloads: &[SectorPayloadV1],
        witnesses: &[L1PayloadResidencyWitness],
    ) -> Result<(), L1AccessError> {
        self.validate_payload_residency_witnesses(payloads, witnesses)?;
        let complete = crate::coalesce_payloads(payloads.to_vec())?;
        for payload in &complete {
            if self.dirty.get(&payload.line_id()).copied().unwrap_or(0) & (1_u8 << payload.sector())
                == 0
            {
                return Err(L1AccessError::DirtyPayloadResidencyWitnessMismatch {
                    line: payload.line_id(),
                    field: "observed store sector is not dirty",
                });
            }
        }
        let mut next = self.clone();
        for payload in complete {
            next.mark_dirty_payload(payload)?;
        }
        *self = next;
        Ok(())
    }

    /// Commit a previously preflighted payload batch atomically. Any state
    /// change after preview causes a typed stale-plan error and leaves this
    /// cache untouched.
    pub fn commit_dirty_payload_admission(
        &mut self,
        plan: L1DirtyPayloadAdmissionPlan,
    ) -> Result<(), L1AccessError> {
        for expected in plan.expected.iter() {
            let current = self
                .dirty_payloads
                .get(&(expected.line, expected.sector))
                .copied();
            let valid_mask = self.sectors.get(&expected.line).copied().unwrap_or(0);
            let dirty_mask = self.dirty.get(&expected.line).copied().unwrap_or(0);
            let generation = self.line_generations.get(&expected.line).copied();
            if current != expected.current
                || valid_mask != expected.valid_mask
                || dirty_mask != expected.dirty_mask
                || generation != expected.generation
            {
                return Err(L1AccessError::DirtyPayloadAdmissionStale {
                    line: expected.line,
                    sector: expected.sector,
                });
            }
        }
        let mut next = self.clone();
        for payload in plan.payloads.iter().copied() {
            next.mark_dirty_payload(payload)?;
        }
        *self = next;
        Ok(())
    }

    pub fn preview_dirty_payloads(&self) -> Result<Vec<SectorPayloadV1>, L1AccessError> {
        let mut payloads = Vec::new();
        for (&line, &mask) in &self.dirty {
            for sector in 0..4_u8 {
                if mask & (1 << sector) == 0 {
                    continue;
                }
                let Some(payload) = self.dirty_payloads.get(&(line, sector)).copied() else {
                    return Err(L1AccessError::DirtyPayloadMissing { line, sector });
                };
                if !payload.is_fully_known() {
                    return Err(L1AccessError::DirtyPayloadMissing { line, sector });
                }
                payloads.push(payload);
            }
        }
        payloads.sort_by_key(|payload| (payload.line_id(), payload.sector()));
        Ok(payloads)
    }

    /// Return the current dirty-line gauge. This is a state observation, not
    /// completed traffic; callers must use the transactional flush plan below
    /// before claiming a writeback completion.
    pub fn dirty_residency(&self) -> L1DirtyResidency {
        L1DirtyResidency {
            lines: u64::try_from(self.dirty.len()).unwrap_or(u64::MAX),
            sectors: self
                .dirty
                .values()
                .map(|mask| u64::from(mask.count_ones()))
                .sum(),
        }
    }

    /// Snapshot dirty sectors without mutating cache state. The plan is
    /// committed only after the lower-level service has accepted the exact
    /// masks, preventing a failed admission from silently clearing dirty data.
    pub fn preview_dirty_flush(&self) -> L1DirtyFlushPlan {
        let mut lines = self
            .dirty
            .iter()
            .map(|(&line_id, &dirty_mask)| L1DirtyLine {
                line_id,
                dirty_mask,
            })
            .collect::<Vec<_>>();
        lines.sort_unstable_by_key(|line| line.line_id);
        L1DirtyFlushPlan {
            lines: lines.into_boxed_slice(),
        }
    }

    /// Commit a previously previewed flush after lower-level service has
    /// accepted it. Resident tags remain present but become clean.
    pub fn commit_dirty_flush(&mut self, plan: L1DirtyFlushPlan) -> Result<(), L1AccessError> {
        for line in plan.lines() {
            let current = self.dirty.get(&line.line_id).copied().unwrap_or(0);
            if current != line.dirty_mask {
                return Err(L1AccessError::DirtyFlushStale {
                    line: line.line_id,
                    expected: line.dirty_mask,
                    current,
                });
            }
        }
        for line in plan.lines() {
            self.dirty.remove(&line.line_id);
            for sector in 0..4_u8 {
                if line.dirty_mask & (1 << sector) != 0 {
                    self.dirty_payloads.remove(&(line.line_id, sector));
                }
            }
        }
        Ok(())
    }

    /// Invalidate resident sectors after the caller has completed any dirty
    /// flush and protocol service. Pending fills are rejected rather than
    /// silently dropped because response tokens need an explicit invalidation
    /// event/ack path in the SM system.
    pub fn invalidate(
        &mut self,
        line: i64,
        sector_mask: u8,
    ) -> Result<L1InvalidationResult, L1AccessError> {
        Self::validate_sector_mask(line, sector_mask)?;
        if self.line_fills.contains_key(&line)
            || self
                .sector_fills
                .keys()
                .any(|(candidate, _, _)| *candidate == line)
        {
            return Err(L1AccessError::InvalidationPendingFill { line });
        }
        let resident_mask = self.sectors.get(&line).copied().unwrap_or(0);
        let invalidated_mask = resident_mask & sector_mask;
        let dirty_mask = self.dirty.get(&line).copied().unwrap_or(0) & invalidated_mask;
        if dirty_mask != 0 {
            return Err(L1AccessError::DirtyResidencyBlocksInvalidation { line, dirty_mask });
        }
        if invalidated_mask != 0 {
            let next = resident_mask & !invalidated_mask;
            if next == 0 {
                self.sectors.remove(&line);
                if let Some(stamp) = self.stamp_of.remove(&line) {
                    self.by_stamp.remove(&stamp);
                }
            } else {
                self.sectors.insert(line, next);
            }
        }
        Ok(L1InvalidationResult {
            line_id: line,
            requested_mask: sector_mask,
            invalidated_mask,
        })
    }

    /// Explicit cache-control invalidation for a complete L1. This models
    /// only the admitted `CCTL.IVALL`-style operation: pending fills and dirty
    /// sectors refuse because the SM event layer must first account for their
    /// response/writeback ordering.
    pub fn preflight_invalidate_all(&self) -> Result<u64, L1AccessError> {
        if !self.line_fills.is_empty() || !self.sector_fills.is_empty() {
            return Err(L1AccessError::InvalidationAllPendingFill);
        }
        if let Some((&line, &dirty_mask)) = self.dirty.iter().next() {
            return Err(L1AccessError::DirtyResidencyBlocksInvalidation { line, dirty_mask });
        }
        Ok(self
            .sectors
            .values()
            .map(|mask| u64::from(mask.count_ones()))
            .sum())
    }

    pub fn invalidate_all(&mut self) -> Result<u64, L1AccessError> {
        let sectors = self.preflight_invalidate_all()?;
        self.by_stamp.clear();
        self.stamp_of.clear();
        self.sectors.clear();
        Ok(sectors)
    }

    /// Pending line/sector fills prevent invalidation even when they have no
    /// local timestamp. Event-mode callers must wait for response events,
    /// never mistake the absence of a timed deadline for an empty cache.
    pub fn has_pending_fills(&self) -> bool {
        !self.line_fills.is_empty() || !self.sector_fills.is_empty()
    }

    fn commit_due_fills(
        &mut self,
        due: Vec<(u64, u8, u64, i64, u8, u32)>,
    ) -> Result<(), L1AccessError> {
        for (_, kind, _, line, sector, generation) in due {
            if kind == 0 {
                self.line_fills.remove(&line);
                self.touch_mru(line)?;
                self.evict_beyond_capacity()?;
            } else {
                self.sector_fills.remove(&(line, sector, generation));
                let generation_matches = self.line_generations.get(&line) == Some(&generation);
                if self.line_resident(line) && generation_matches {
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
            .map(|&(due, ..)| due)
            .chain(self.sector_fills.values().map(|&(due, ..)| due))
            .filter_map(FillDue::timed)
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
        let mut due: Vec<(u64, u8, u64, i64, u8, u32)> = Vec::new();
        for (&line, &(fill_due, seq, generation)) in &self.line_fills {
            if let FillDue::Timed(time) = fill_due
                && time <= now
            {
                due.push((time, 0, seq, line, 0, generation));
            }
        }
        for (&(line, sector, generation), &(fill_due, seq)) in &self.sector_fills {
            if let FillDue::Timed(time) = fill_due
                && time <= now
            {
                due.push((time, 1, seq, line, sector, generation));
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
        mshr_floor: Option<NonZeroU32>,
    ) -> Result<CoreAccess, L1AccessError> {
        let line = sector_id >> 2;
        let sector = (sector_id & 3) as u8;
        // A late child of an in-flight line allocation has its own sector
        // completion token. Check it before the parent line MSHR so later
        // accesses to that sector wait for the child response instead of
        // incorrectly riding the parent's earlier response.
        if let Some(&generation) = self.line_generations.get(&line)
            && let Some(&(pending, sequence)) = self.sector_fills.get(&(line, sector, generation))
        {
            Self::preflight_fill_timing(pending, fill_request.timing())?;
            let counters = if self.active_lookup_batch.is_some_and(|batch| {
                batch.pending_merges_are_hits && sequence < batch.first_fill_sequence
            }) {
                self.counters.add_fields_checked(
                    origin,
                    &[
                        (L1CounterField::Merges, 1),
                        (L1CounterField::Misses, 1),
                        (L1CounterField::PendingLookupHits, 1),
                    ],
                )?
            } else {
                self.counters.add_fields_checked(
                    origin,
                    &[(L1CounterField::Merges, 1), (L1CounterField::Misses, 1)],
                )?
            };
            self.counters = counters;
            return Ok(CoreAccess::SectorMerge {
                pending,
                generation,
            });
        }
        if let Some(&(pending, sequence, generation)) = self.line_fills.get(&line) {
            Self::preflight_fill_timing(pending, fill_request.timing())?;
            let counters = if self.active_lookup_batch.is_some_and(|batch| {
                batch.pending_merges_are_hits && sequence < batch.first_fill_sequence
            }) {
                self.counters.add_fields_checked(
                    origin,
                    &[
                        (L1CounterField::Merges, 1),
                        (L1CounterField::Misses, 1),
                        (L1CounterField::PendingLookupHits, 1),
                    ],
                )?
            } else {
                self.counters.add_fields_checked(
                    origin,
                    &[(L1CounterField::Merges, 1), (L1CounterField::Misses, 1)],
                )?
            };
            self.counters = counters;
            *self.sectors.entry(line).or_default() |= 1 << sector;
            return Ok(CoreAccess::LineMerge {
                pending,
                generation,
            });
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
        let line_generation = self.line_generations.get(&line).copied();
        if line_resident
            && let Some(generation) = line_generation
            && let Some(&(pending, sequence)) = self.sector_fills.get(&(line, sector, generation))
        {
            Self::preflight_fill_timing(pending, fill_request.timing())?;
            let counters = if self.active_lookup_batch.is_some_and(|batch| {
                batch.pending_merges_are_hits && sequence < batch.first_fill_sequence
            }) {
                self.counters.add_fields_checked(
                    origin,
                    &[
                        (L1CounterField::Merges, 1),
                        (L1CounterField::Misses, 1),
                        (L1CounterField::PendingLookupHits, 1),
                    ],
                )?
            } else {
                self.counters.add_fields_checked(
                    origin,
                    &[(L1CounterField::Merges, 1), (L1CounterField::Misses, 1)],
                )?
            };
            self.counters = counters;
            return Ok(CoreAccess::SectorMerge {
                pending,
                generation,
            });
        }
        // Miss path: needs an MSHR entry (both fill kinds consume capacity).
        if let Some(cap) = self.effective_mshr_count(mshr_floor)
            && self.mshr_in_use() >= cap
        {
            self.apply_counter_deltas(origin, &[(L1CounterField::StallEvents, 1)])?;
            return Ok(CoreAccess::Stalled);
        }
        let fill_due = fill_request.fill_due(self.cfg.miss_latency)?;
        let (seq, next_fill_seq) = self.fill_seq.allocate("fill")?;
        // The allocation sequence is u64. A resident-line sector fill carries
        // the existing line allocation's compact u32 generation in its
        // FillKey, so it must not fail merely because the independent fill
        // ordering sequence has crossed the u32 boundary. A new line
        // allocation still needs the checked conversion below.
        let generation = if line_resident {
            self.line_generations
                .get(&line)
                .copied()
                .or_else(|| u32::try_from(seq).ok())
                .ok_or(L1AccessError::SequenceOverflow {
                    sequence: "fill generation",
                    current: seq,
                })?
        } else {
            u32::try_from(seq).map_err(|_| L1AccessError::SequenceOverflow {
                sequence: "fill generation",
                current: seq,
            })?
        };
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
            let generation = self
                .line_generations
                .entry(line)
                .or_insert(generation)
                .to_owned();
            self.sector_fills
                .insert((line, sector, generation), (fill_due, seq));
            // Capture the allocation that owns this resident line.  This is
            // required in event mode because a response may arrive after the
            // line has been evicted and the same tag reallocated.
            Ok(CoreAccess::MissSectorResident {
                pending: fill_due,
                generation,
            })
        } else {
            // The fill sequence is also the allocation-generation witness.
            // It is installed at allocation time (rather than response time)
            // so an older sector response cannot enter a newly allocated line
            // while that line is still in flight.
            self.line_generations.insert(line, generation);
            self.line_fills.insert(line, (fill_due, seq, generation));
            *self.sectors.entry(line).or_default() |= 1 << sector;
            Ok(CoreAccess::MissLineAlloc {
                pending: fill_due,
                generation,
            })
        }
    }

    fn access_single(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
        readiness: Readiness,
        mshr_floor: Option<NonZeroU32>,
    ) -> Result<SingleOutcome, L1AccessError> {
        self.apply_fills(now)?;
        match self.access_core(
            origin,
            sector_id,
            FillRequest::Constant { now, readiness },
            mshr_floor,
        )? {
            CoreAccess::Hit { ready } => Ok(SingleOutcome::Done(AccessOutcome::Hit, ready)),
            CoreAccess::LineMerge { pending, .. } | CoreAccess::SectorMerge { pending, .. } => {
                let pending = pending.timed().ok_or(L1AccessError::FillTimingMismatch {
                    latched: Self::timing_name(FillTiming::Event),
                    requested: Self::timing_name(FillTiming::Constant),
                })?;
                Ok(SingleOutcome::Done(AccessOutcome::Merge, Some(pending)))
            }
            CoreAccess::Stalled => Ok(SingleOutcome::Stall {
                retry_at: self.earliest_fill()?,
            }),
            CoreAccess::MissSectorResident { pending, .. } => {
                let ready = pending.timed().ok_or(L1AccessError::FillTimingMismatch {
                    latched: Self::timing_name(FillTiming::Event),
                    requested: Self::timing_name(FillTiming::Constant),
                })?;
                Ok(SingleOutcome::Done(
                    AccessOutcome::MissSectorResident,
                    Some(ready),
                ))
            }
            CoreAccess::MissLineAlloc { pending, .. } => {
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
        mshr_floor: Option<NonZeroU32>,
    ) -> Result<(AccessOutcome, u64), L1AccessError> {
        self.commit_latch(FillTiming::Constant);
        let mut t = now;
        loop {
            match self.access_single(origin, sector_id, t, Readiness::Observed, mshr_floor)? {
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
        mshr_floor: Option<NonZeroU32>,
    ) -> Result<AccessOutcome, L1AccessError> {
        self.commit_latch(FillTiming::Constant);
        let mut t = now;
        loop {
            match self.access_single(origin, sector_id, t, Readiness::Unobserved, mshr_floor)? {
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
        let fast_path_safe = !self.constant_access_may_retry(sector_id, now, None)
            && self.preflight_mutation_budget(budget).is_ok()
            && self.preflight_constant_access_timing(now).is_ok();
        if !fast_path_safe {
            let mut shadow = self.clone();
            let outcome = shadow.access_eligible_constant_impl(origin, sector_id, now, None)?;
            *self = shadow;
            return Ok(outcome);
        }
        self.access_eligible_constant_impl(origin, sector_id, now, None)
    }

    /// Constant-mode eligible access with a caller-characterized lower bound on
    /// finite miss admission. The floor applies only to this access; ordinary
    /// accesses continue to observe [`L1Config::mshr_count`].
    pub fn access_eligible_with_mshr_floor(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
        mshr_floor: NonZeroU32,
    ) -> Result<(AccessOutcome, u64), L1AccessError> {
        self.preflight_latch(FillTiming::Constant)?;
        let budget = L1MutationBudget {
            eligible_sectors: 1,
            recency_touches: 1,
            new_fills: 1,
            ..L1MutationBudget::for_origin(origin)
        };
        let fast_path_safe = !self.constant_access_may_retry(sector_id, now, Some(mshr_floor))
            && self.preflight_mutation_budget(budget).is_ok()
            && self.preflight_constant_access_timing(now).is_ok();
        if !fast_path_safe {
            let mut shadow = self.clone();
            let outcome =
                shadow.access_eligible_constant_impl(origin, sector_id, now, Some(mshr_floor))?;
            *self = shadow;
            return Ok(outcome);
        }
        self.access_eligible_constant_impl(origin, sector_id, now, Some(mshr_floor))
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
        let fast_path_safe = !self.constant_access_may_retry(sector_id, now, None)
            && self.preflight_mutation_budget(budget).is_ok()
            && self.preflight_constant_access_timing(now).is_ok();
        if !fast_path_safe {
            let mut shadow = self.clone();
            let outcome =
                shadow.access_eligible_constant_unobserved_impl(origin, sector_id, now, None)?;
            *self = shadow;
            return Ok(outcome);
        }
        self.access_eligible_constant_unobserved_impl(origin, sector_id, now, None)
    }

    /// Unobserved counterpart to [`Self::access_eligible_with_mshr_floor`].
    pub fn access_eligible_unobserved_with_mshr_floor(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
        mshr_floor: NonZeroU32,
    ) -> Result<AccessOutcome, L1AccessError> {
        self.preflight_latch(FillTiming::Constant)?;
        let budget = L1MutationBudget {
            eligible_sectors: 1,
            recency_touches: 1,
            new_fills: 1,
            ..L1MutationBudget::for_origin(origin)
        };
        let fast_path_safe = !self.constant_access_may_retry(sector_id, now, Some(mshr_floor))
            && self.preflight_mutation_budget(budget).is_ok()
            && self.preflight_constant_access_timing(now).is_ok();
        if !fast_path_safe {
            let mut shadow = self.clone();
            let outcome = shadow.access_eligible_constant_unobserved_impl(
                origin,
                sector_id,
                now,
                Some(mshr_floor),
            )?;
            *self = shadow;
            return Ok(outcome);
        }
        self.access_eligible_constant_unobserved_impl(origin, sector_id, now, Some(mshr_floor))
    }

    fn access_eligible_event_core(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
        readiness: Readiness,
        mshr_floor: Option<NonZeroU32>,
    ) -> Result<CoreAccess, L1AccessError> {
        // Delay the mode latch until the ladder succeeds.  In particular, an
        // observed hit at `u64::MAX` can fail its ready-time addition; that
        // typed error must not leave a previously fresh cache permanently
        // latched to EVENT.  `access_core` validates all fallible arithmetic
        // before its corresponding mutation, so committing here is the
        // transaction boundary for the mode bit.
        let outcome = self.access_core(
            origin,
            sector_id,
            FillRequest::Event { now, readiness },
            mshr_floor,
        )?;
        self.commit_latch(FillTiming::Event);
        Ok(outcome)
    }

    fn access_eligible_event_impl(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
        mshr_floor: Option<NonZeroU32>,
    ) -> Result<EvAccess, L1AccessError> {
        let line = sector_id >> 2;
        let sector = (sector_id & 3) as u8;
        Ok(
            match self.access_eligible_event_core(
                origin,
                sector_id,
                now,
                Readiness::Observed,
                mshr_floor,
            )? {
                CoreAccess::Hit { ready: Some(ready) } => EvAccess::Hit { ready },
                CoreAccess::Hit { ready: None } => {
                    unreachable!("observed event access omitted its ready time")
                }
                CoreAccess::LineMerge { generation, .. } => EvAccess::MergedPending {
                    fill: FillKey::Line { line, generation },
                },
                CoreAccess::SectorMerge { generation, .. } => EvAccess::MergedPending {
                    fill: FillKey::Sector {
                        line,
                        sector,
                        generation,
                    },
                },
                CoreAccess::Stalled => EvAccess::Stalled,
                CoreAccess::MissSectorResident { generation, .. } => EvAccess::MissRequest {
                    fill: FillKey::Sector {
                        line,
                        sector,
                        generation,
                    },
                },
                CoreAccess::MissLineAlloc { generation, .. } => EvAccess::MissRequest {
                    fill: FillKey::Line { line, generation },
                },
            },
        )
    }

    fn access_eligible_event_unobserved_impl(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
        mshr_floor: Option<NonZeroU32>,
    ) -> Result<EvAccessUnobserved, L1AccessError> {
        let line = sector_id >> 2;
        let sector = (sector_id & 3) as u8;
        Ok(
            match self.access_eligible_event_core(
                origin,
                sector_id,
                now,
                Readiness::Unobserved,
                mshr_floor,
            )? {
                CoreAccess::Hit { .. } => EvAccessUnobserved::Hit,
                CoreAccess::LineMerge { generation, .. } => EvAccessUnobserved::MergedPending {
                    fill: FillKey::Line { line, generation },
                },
                CoreAccess::SectorMerge { generation, .. } => EvAccessUnobserved::MergedPending {
                    fill: FillKey::Sector {
                        line,
                        sector,
                        generation,
                    },
                },
                CoreAccess::Stalled => EvAccessUnobserved::Stalled,
                CoreAccess::MissSectorResident { generation, .. } => {
                    EvAccessUnobserved::MissRequest {
                        fill: FillKey::Sector {
                            line,
                            sector,
                            generation,
                        },
                    }
                }
                CoreAccess::MissLineAlloc { generation, .. } => EvAccessUnobserved::MissRequest {
                    fill: FillKey::Line { line, generation },
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
            let outcome = shadow.access_eligible_event_impl(origin, sector_id, now, None)?;
            *self = shadow;
            return Ok(outcome);
        }
        self.access_eligible_event_impl(origin, sector_id, now, None)
    }

    /// Event-mode eligible access with a caller-characterized lower bound on
    /// finite miss admission. The floor applies only to this access.
    pub fn access_eligible_ev_with_mshr_floor(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
        mshr_floor: NonZeroU32,
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
            let outcome =
                shadow.access_eligible_event_impl(origin, sector_id, now, Some(mshr_floor))?;
            *self = shadow;
            return Ok(outcome);
        }
        self.access_eligible_event_impl(origin, sector_id, now, Some(mshr_floor))
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
            let outcome =
                shadow.access_eligible_event_unobserved_impl(origin, sector_id, now, None)?;
            *self = shadow;
            return Ok(outcome);
        }
        self.access_eligible_event_unobserved_impl(origin, sector_id, now, None)
    }

    /// Unobserved counterpart to [`Self::access_eligible_ev_with_mshr_floor`].
    pub fn access_eligible_ev_unobserved_with_mshr_floor(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
        mshr_floor: NonZeroU32,
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
            let outcome = shadow.access_eligible_event_unobserved_impl(
                origin,
                sector_id,
                now,
                Some(mshr_floor),
            )?;
            *self = shadow;
            return Ok(outcome);
        }
        self.access_eligible_event_unobserved_impl(origin, sector_id, now, Some(mshr_floor))
    }

    /// Complete an event-mode fill AT the response event's time. A
    /// sector fill whose line was evicted while it was in flight is
    /// DROPPED, like the lazy applier. A child may return while its
    /// parent line allocation is still pending; retain its sector bit so
    /// the parent completion can expose the complete allocation. Returns
    /// whether the generation-matched data was retained.
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
            FillKey::Line { .. } => true,
            FillKey::Sector {
                line, generation, ..
            } => self.line_resident(line) && self.line_generations.get(&line) == Some(&generation),
        };
        if will_touch {
            self.next_stamp.preflight("recency", 1)?;
        }
        self.commit_latch(FillTiming::Event);
        match fill {
            FillKey::Line { line, .. } => {
                self.preflight_dirty_eviction(Some(line))?;
                self.line_fills.remove(&line);
                self.touch_mru(line)?;
                self.evict_beyond_capacity()?;
                Ok(true)
            }
            FillKey::Sector {
                line,
                sector,
                generation,
            } => {
                self.sector_fills.remove(&(line, sector, generation));
                let generation_matches = self.line_resident(line)
                    && self.line_generations.get(&line) == Some(&generation);
                let allocation_matches = self.line_generations.get(&line) == Some(&generation)
                    && (self.line_resident(line) || self.line_fills.contains_key(&line));
                if allocation_matches {
                    *self.sectors.entry(line).or_default() |= 1 << sector;
                    if generation_matches {
                        self.touch_mru(line)?;
                    }
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
        }
    }

    fn probe_write_impl(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
        accounting: WriteAccounting,
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
        let deltas: &[(L1CounterField, u64)] = match accounting {
            WriteAccounting::Store => &[
                (L1CounterField::StoreSectors, 1),
                (L1CounterField::StoreTagHits, u64::from(tag_resident)),
                (L1CounterField::StoreHits, u64::from(hit)),
            ],
            WriteAccounting::Reduction => &[
                (L1CounterField::ReductionSectors, 1),
                (L1CounterField::ReductionHits, u64::from(hit)),
            ],
        };
        self.counters = self.counters.add_fields_checked(origin, deltas)?;
        if hit {
            self.touch_mru(line)?;
            if origin == MemoryOrigin::Global
                && accounting == WriteAccounting::Store
                && self.store_policy == L1StorePolicy::ResidentHitWriteBack
            {
                self.mark_dirty(line, 1_u8 << sector)?;
            }
        }
        Ok(hit)
    }

    /// Hit-test a global-class store sector against the current state without
    /// allocating on miss. Under the default policy this is a write-through
    /// no-allocate hypothesis; the explicit resident-hit writeback policy marks
    /// a valid hit dirty for the terminal flush path.
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
        if self.preflight_mutation_budget(budget).is_err()
            || self.preflight_due_fill_recency(now, 1).is_err()
        {
            let mut shadow = self.clone();
            let hit = shadow.probe_write_impl(origin, sector_id, now, WriteAccounting::Store)?;
            *self = shadow;
            return Ok(hit);
        }
        self.probe_write_impl(origin, sector_id, now, WriteAccounting::Store)
    }

    /// Hit-test a RED/REDG sector through the same physical write path as a
    /// store while recording a distinct reduction observer lane. RED therefore
    /// contributes to the unified L1TEX hit rate but never to ordinary STG
    /// attribution; its RMW/writeback/coherence tails remain explicit on the SM
    /// result.
    pub fn probe_reduction(
        &mut self,
        origin: MemoryOrigin,
        sector_id: i64,
        now: u64,
    ) -> Result<bool, L1AccessError> {
        let budget = L1MutationBudget {
            reduction_sectors: 1,
            recency_touches: 1,
            ..L1MutationBudget::for_origin(origin)
        };
        if self.preflight_mutation_budget(budget).is_err()
            || self.preflight_due_fill_recency(now, 1).is_err()
        {
            let mut shadow = self.clone();
            let hit =
                shadow.probe_write_impl(origin, sector_id, now, WriteAccounting::Reduction)?;
            *self = shadow;
            return Ok(hit);
        }
        self.probe_write_impl(origin, sector_id, now, WriteAccounting::Reduction)
    }

    /// Record write-class lookup hits derived from the exact instruction's
    /// sector/wavefront topology. This is emission accounting only: the
    /// physical write-through/no-allocate L1 trajectory remains unchanged.
    pub fn count_write_lookup_hits(
        &mut self,
        origin: MemoryOrigin,
        hits: u64,
    ) -> Result<(), L1AccessError> {
        self.apply_counter_deltas(origin, &[(L1CounterField::WriteLookupHits, hits)])
    }

    /// Observe one complete write instruction through the architecture-local
    /// L1TEX write tag window.  The observer is deliberately independent of
    /// the physical read cache and emits only the number of sector lookups
    /// whose line tag was already resident before this instruction.  A fresh
    /// line is installed after all sectors in the request are classified, so
    /// same-request coalescing is not accidentally reported as a hit.
    pub fn observe_write_lookup_batch(&mut self, sectors: &[i64]) -> Result<u64, L1AccessError> {
        let config = self
            .cfg
            .arch
            .l1_write_lookup_window()
            .require("l1_write_lookup_window")?;
        if config.set_count == 0 || config.tag_slots == 0 {
            return Err(L1AccessError::WriteLookupWindowCapability(
                sim_arch::CapabilityError::ResearchGap {
                    arch: self.cfg.arch,
                    capability: "l1_write_lookup_window",
                    reason: "write-lookup window has zero sets or tag slots",
                },
            ));
        }
        let mut next = self
            .write_lookup_observer
            .as_deref()
            .cloned()
            .unwrap_or_else(|| L1WriteLookupObserver::new(config));
        let hits = next.observe_batch(sectors)?;
        self.write_lookup_observer = Some(Box::new(next));
        Ok(hits)
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
