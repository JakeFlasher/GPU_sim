// Lexical implementation section 1/3; included by the parent facade.
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
    /// Optional card-backed set count for the effective SM86 state arm. `None`
    /// retains the historical fully-associative sensitivity model.
    pub set_count: Option<u32>,
    /// Optional per-set way bound. The physical hash is intentionally not
    /// guessed; the current effective scenario uses a documented modulo index.
    pub ways: Option<u32>,
    pub set_index: L1SetIndex,
}

/// Effective L1 set-index scenario. `Modulo` is a transparent address-domain
/// sensitivity arm; it is not a claim that the proprietary GA102 hash is known.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum L1SetIndex {
    #[default]
    FullyAssociative,
    Modulo,
    /// Bounded unknown-hash sensitivity: fold the next set-index bit group
    /// into the low set bits. This is a candidate address-domain transform,
    /// not a recovered GA102 physical hash.
    XorFold,
    /// Pressure-bounded unknown-hash envelope. Use the card modulo candidate
    /// until its set is full, then use the folded candidate only when it has
    /// strictly more room. This is a deterministic effective scenario, not a
    /// workload/NCU-fitted physical hash selector.
    ModuloXorFallback,
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
            set_count: None,
            ways: None,
            set_index: L1SetIndex::FullyAssociative,
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
    #[error("L1 set geometry must provide both positive set_count and ways")]
    SetGeometryInvalid,
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
    client: L1LookupClient,
}

/// Fixed-size, read-only audit of the sector masks presented to one logical
/// lookup request. It is diagnostic only: it never changes cache state,
/// replacement, MSHR ownership, or replay timing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct L1RequestMaskAudit {
    pub request_count: [u64; 5],
    pub request_line_count: [u64; 5],
    pub request_masks: [[u64; 16]; 5],
    pub same_request_merge_masks: [[u64; 16]; 5],
    pub older_pending_merge_masks: [[u64; 16]; 5],
    pub request_mask_overflow: [u64; 5],
}

impl L1RequestMaskAudit {
    pub const MAX_REQUEST_LINES: usize = 1024;
}

/// Read-only provenance for L1 lookup merges. These counters describe
/// the request boundary observed by the cache model; they do not alter tags,
/// fills, MSHR ownership, replacement, or timing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct L1MergeAudit {
    pub same_request_merges: [u64; 2],
    pub older_pending_merges: [u64; 2],
    pub already_requested_merges: [u64; 2],
    pub unrequested_pending_merges: [u64; 2],
    pub ordinary_same_request_merges: [u64; 2],
    pub ordinary_older_pending_merges: [u64; 2],
    pub async_same_request_merges: [u64; 2],
    pub async_older_pending_merges: [u64; 2],
    pub ordinary_global_older_pending_merges: u64,
    pub constant_global_older_pending_merges: u64,
    /// Same-request ordinary-global merges partitioned by static client.
    /// These observer fields let the post-replay endpoint selector handle
    /// mixed constant/plain launches without changing cache state or timing.
    pub ordinary_global_same_request_merges: u64,
    pub constant_global_same_request_merges: u64,
    pub async_global_same_request_merges: u64,
    /// Resident-sector lookup hits partitioned by the static lookup client.
    /// These are observer-only provenance counters; cache state and timing
    /// remain unified.  Constant hits are retained in the legacy field below
    /// for compatibility with the existing SM86 endpoint.
    pub ordinary_global_hits: u64,
    pub async_global_hits: u64,
    /// Older pending merges into a partial-line sector fill.  Nsight's
    /// sector lookup endpoint keeps these as misses even when the 128-byte
    /// tag is already resident; retaining the owner kind prevents the
    /// aggregate merge formula from fabricating a tag hit for U16 half-line
    /// accesses.
    pub ordinary_global_older_sector_pending_merges: u64,
    pub constant_global_older_sector_pending_merges: u64,
    pub async_global_older_sector_pending_merges: u64,
    pub unrequested_subline_miss_sectors: u64,
    pub wavefront_mask_lookup_hits: u64,
    pub wavefront_mask_lookup_misses: u64,
    pub constant_global_hits: u64,
    pub constant_global_requests: u64,
    pub request_masks: L1RequestMaskAudit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L1LookupClient {
    Unknown,
    OrdinaryGlobal,
    Constant,
    Local,
    AsyncCopy,
}

impl L1LookupClient {
    pub const ALL: [Self; 5] = [
        Self::Unknown,
        Self::OrdinaryGlobal,
        Self::Constant,
        Self::Local,
        Self::AsyncCopy,
    ];
    pub const fn index(self) -> usize {
        match self {
            Self::Unknown => 0,
            Self::OrdinaryGlobal => 1,
            Self::Constant => 2,
            Self::Local => 3,
            Self::AsyncCopy => 4,
        }
    }
}

/// One origin's accounting lane. It mirrors the unified fields in
/// [`L1Counters`] but carries no architecture of its own; provenance remains
/// bound once on the enclosing physical-cache snapshot.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct L1OriginCounters {
    pub hits: u64,
    pub misses: u64,
    pub merges: u64,
    /// Pending sectors already present in the active request mask.
    pub already_requested_merges: u64,
    /// Pending sectors first introduced by this request while a line fill is
    /// in flight; these remain lookup misses.
    pub unrequested_pending_merges: u64,
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

impl L1OriginCounters {
    /// Effective L1TEX tag-lookup-hit projection for sector metrics.
    ///
    /// The timing cache keeps all pending sectors as `merges` because they
    /// share one in-flight fill and must not become independent timing hits.
    /// The profiler lookup surface is a bounded tag observation: a fresh line
    /// allocation contributes one tag miss, while the remaining sectors that
    /// merge onto that allocation are projected as lookup hits. This remains
    /// an explicit effective observer and does not alter tags, fills, MSHRs,
    /// replacement, or replay timing.
    /// Architecture-scoped L1TEX lookup projection. SM120 retains the
    /// original pending-merge/four-sector observer; SM86 uses the card-backed
    /// line-allocation observer. Neither branch changes cache state or timing.
    pub const fn effective_lookup_hits_for_arch(&self, arch: ArchId) -> u64 {
        let projected = if matches!(arch, ArchId::Sm120) {
            self.hits.saturating_add(self.already_requested_merges)
        } else {
            self.hits
                .saturating_add(self.merges)
                .saturating_sub(self.line_alloc_misses)
        };
        let denominator = self.lookup_sectors();
        if projected < denominator {
            projected
        } else {
            denominator
        }
    }

    /// Backward-compatible SM86/default projection for unbound origin callers.
    pub const fn effective_lookup_hits(&self) -> u64 {
        self.effective_lookup_hits_for_arch(ArchId::Sm86)
    }

    /// Total read-sector lookups represented by this origin lane.
    pub const fn lookup_sectors(&self) -> u64 {
        self.hits.saturating_add(self.misses)
    }
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
    pub already_requested_merges: u64,
    pub unrequested_pending_merges: u64,
    pub pending_lookup_hits: u64,
    pub stall_events: u64,
    /// Number of miss attempts observed while the finite L1 MSHR set was
    /// full. These pressure fields are unified cache observers and are not
    /// partitioned into global/local traffic lanes.
    pub mshr_stall_occupancy_samples: u64,
    /// Sum of MSHR entries in use at those full-set stall observations.
    pub mshr_stall_occupancy_sum: u64,
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
    /// §6.4 write-class line-tag observer used by the SM86 compatibility
    /// surface. It is a superset of sector-valid `store_hits`; SM120's
    /// unified sector-hit projection uses the effective model's resident
    /// sector counters instead because no SM120 write-lookup capability is
    /// established.
    pub store_tag_hits: u64,
    pub write_lookup_hits: u64,
    /// RED/REDG sectors and sector-valid hits. They contribute to the unified
    /// L1TEX observer but never to the plain global-store surface.
    pub reduction_sectors: u64,
    pub reduction_hits: u64,
    /// Accounting provenance only. Cache tags, fills, MSHRs, recency, and
    /// replacement remain unified modeled cache state.
    by_origin: [L1OriginCounters; 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L1CounterField {
    Hits,
    Misses,
    Merges,
    AlreadyRequestedMerges,
    UnrequestedPendingMerges,
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
            Self::AlreadyRequestedMerges => "already_requested_merges",
            Self::UnrequestedPendingMerges => "unrequested_pending_merges",
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
    #[error("L1 pressure observer '{field}' overflows u64")]
    PressureOverflow { field: &'static str },
}

fn checked_counter_add(
    field: L1CounterField,
    left: u64,
    right: u64,
) -> Result<u64, L1CounterError> {
    left.checked_add(right)
        .ok_or(L1CounterError::Overflow { field })
}

const PREFLIGHT_COUNTER_FIELDS: [L1CounterField; 19] = [
    L1CounterField::Hits,
    L1CounterField::Misses,
    L1CounterField::Merges,
    L1CounterField::AlreadyRequestedMerges,
    L1CounterField::UnrequestedPendingMerges,
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
) -> Result<[(L1CounterField, u64); 19], L1AccessError> {
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
        (L1CounterField::AlreadyRequestedMerges, 0),
        (L1CounterField::UnrequestedPendingMerges, 0),
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

fn preflight_values_deltas(values: [u64; 19]) -> [(L1CounterField, u64); 19] {
    std::array::from_fn(|index| (PREFLIGHT_COUNTER_FIELDS[index], values[index]))
}

fn accumulate_preflight_values(
    values: &mut [u64; 19],
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
        L1CounterField::AlreadyRequestedMerges => &mut counters.already_requested_merges,
        L1CounterField::UnrequestedPendingMerges => &mut counters.unrequested_pending_merges,
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

fn origin_deltas(counters: &L1OriginCounters) -> [(L1CounterField, u64); 19] {
    [
        (L1CounterField::Hits, counters.hits),
        (L1CounterField::Misses, counters.misses),
        (L1CounterField::Merges, counters.merges),
        (
            L1CounterField::AlreadyRequestedMerges,
            counters.already_requested_merges,
        ),
        (
            L1CounterField::UnrequestedPendingMerges,
            counters.unrequested_pending_merges,
        ),
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

fn unified_deltas(counters: &L1Counters) -> [(L1CounterField, u64); 19] {
    [
        (L1CounterField::Hits, counters.hits),
        (L1CounterField::Misses, counters.misses),
        (L1CounterField::Merges, counters.merges),
        (
            L1CounterField::AlreadyRequestedMerges,
            counters.already_requested_merges,
        ),
        (
            L1CounterField::UnrequestedPendingMerges,
            counters.unrequested_pending_merges,
        ),
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
    /// accounting lanes. This never examines or partitions modeled cache state.
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
            already_requested_merges: self.already_requested_merges,
            unrequested_pending_merges: self.unrequested_pending_merges,
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
                L1CounterField::AlreadyRequestedMerges => &mut next.already_requested_merges,
                L1CounterField::UnrequestedPendingMerges => &mut next.unrequested_pending_merges,
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
        let already_requested_merges = checked_counter_add(
            L1CounterField::AlreadyRequestedMerges,
            self.already_requested_merges,
            other.already_requested_merges,
        )?;
        let unrequested_pending_merges = checked_counter_add(
            L1CounterField::UnrequestedPendingMerges,
            self.unrequested_pending_merges,
            other.unrequested_pending_merges,
        )?;
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
        let mshr_stall_occupancy_samples = self
            .mshr_stall_occupancy_samples
            .checked_add(other.mshr_stall_occupancy_samples)
            .ok_or(L1CounterError::PressureOverflow {
                field: "mshr_stall_occupancy_samples",
            })?;
        let mshr_stall_occupancy_sum = self
            .mshr_stall_occupancy_sum
            .checked_add(other.mshr_stall_occupancy_sum)
            .ok_or(L1CounterError::PressureOverflow {
                field: "mshr_stall_occupancy_sum",
            })?;
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
            already_requested_merges,
            unrequested_pending_merges,
            pending_lookup_hits,
            stall_events,
            mshr_stall_occupancy_samples,
            mshr_stall_occupancy_sum,
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
/// from the modeled read-state cache: the latter remains the validated
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
    merge_audit: L1MergeAudit,
    /// Exact earliest timed in-flight fill. `None` means there is no timed
    /// fill (empty or event-driven), while `Some(u64::MAX)` is a real due
    /// cycle. `apply_fills` early-outs when `now` has not reached it.
    min_fill_due: Option<u64>,
    /// Latched fill-timing discipline (see [`FillTiming`]).
    fill_timing: Option<FillTiming>,
    /// Lazily allocated observer-only write tag window.  Keeping it boxed and
    /// optional leaves cold/read-only replays on the compact hot path.
    write_lookup_observer: Option<Box<L1WriteLookupObserver>>,
    /// Optional effective set-local LRU state. The global recency maps remain
    /// the canonical typed boundary; this index is an executable projection
    /// used only by the card-backed modulo scenario.
    set_lru: Option<Vec<VecDeque<i64>>>,
    /// Actual set chosen by the pressure-bounded unknown-hash scenario.
    /// Fully associative and fixed-index modes leave this empty.
    set_locations: HashMap<i64, usize>,
    pub counters: L1Counters,
}

