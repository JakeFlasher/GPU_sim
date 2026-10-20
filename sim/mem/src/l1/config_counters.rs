//! Semantic submodule; report-only boundary state and serde contracts.

use super::*;

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
    pub(super) lines: Box<[L1DirtyLine]>,
}

impl L1DirtyFlushPlan {
    pub fn lines(&self) -> &[L1DirtyLine] {
        &self.lines
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct L1DirtyPayloadExpectation {
    pub(super) line: i64,
    pub(super) sector: u8,
    pub(super) current: Option<SectorPayloadV1>,
    pub(super) valid_mask: u8,
    pub(super) dirty_mask: u8,
    pub(super) generation: Option<u32>,
}

/// Transactional admission of complete value-bearing dirty sectors. The
/// caller must already possess a residency/allocation witness; this plan
/// checks the current L1 residency and preserves the payload allocation
/// identity, but it does not infer a hardware allocation from a line tag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct L1DirtyPayloadAdmissionPlan {
    pub(super) payloads: Box<[SectorPayloadV1]>,
    pub(super) expected: Box<[L1DirtyPayloadExpectation]>,
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

/// Opaque identity of one logical memory request's L1 lookup batch. A pending
/// fill whose allocation sequence predates this token is an NCU lookup hit;
/// merges into a fill created by this same request are sector coalescing and
/// remain lookup misses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L1LookupBatch {
    pub(super) first_fill_sequence: u64,
    pub(super) pending_merges_are_hits: bool,
    pub(super) client: L1LookupClient,
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
    pub(super) arch: Option<ArchId>,
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
    pub(super) by_origin: [L1OriginCounters; 2],
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

    pub(super) fn add_fields_checked(
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

/// Observer-only tag window for Ampere L1TEX write lookups.  It is separate
/// from the modeled read-state cache: the latter remains the validated
/// fully-associative replay approximation, while this state models the
/// write-allocate request stream used by the NCU `op_st_lookup_*` counters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct L1WriteLookupObserver {
    pub(super) config: L1WriteLookupWindow,
    /// Each set is kept in LRU-to-MRU order.  A store instruction evaluates
    /// all sectors of a line against the pre-request tag, then touches it once
    /// (same-request sector coalescing therefore remains a miss).
    pub(super) sets: Vec<Vec<i64>>,
}

/// Fallible-free state transition prepared for one write-observer batch.  The
/// plan clones only sets touched by the batch (rather than the complete
/// observer), so callers can defer publication across a larger instruction
/// transaction without turning every store into an O(total-set-count) clone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct L1WriteLookupObserverPlan {
    pub(super) hits: u64,
    pub(super) sets: Vec<(usize, Vec<i64>)>,
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

    pub(super) fn new(config: L1WriteLookupWindow) -> Self {
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
    pub(super) intervals: Vec<(i64, i64)>,
}

/// A plan whose exact instruction-wide union cardinality has been computed.
#[derive(Debug)]
pub struct PreparedSectorExpansion {
    pub(super) intervals: Vec<(i64, i64)>,
    pub(super) unique_count: usize,
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
