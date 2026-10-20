//! Semantic submodule; report-only boundary state and serde contracts.

use super::*;

pub const SECTOR_BYTES: u32 = 32;

pub const SECTORS_PER_LINE: u32 = 4;

pub const LINE_BYTES: u32 = SECTOR_BYTES * SECTORS_PER_LINE;

/// Outcome of one eligible sector access (after any MSHR stall retries).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessOutcome {
    Hit,
    MissLineAlloc,
    MissSectorResident,
    Merge,
}

pub(super) fn checked_counter_add(
    field: L1CounterField,
    left: u64,
    right: u64,
) -> Result<u64, L1CounterError> {
    left.checked_add(right)
        .ok_or(L1CounterError::Overflow { field })
}

pub(super) const PREFLIGHT_COUNTER_FIELDS: [L1CounterField; 19] = [
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

pub(super) fn preflight_budget_deltas(
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

pub(super) fn preflight_values_deltas(values: [u64; 19]) -> [(L1CounterField, u64); 19] {
    std::array::from_fn(|index| (PREFLIGHT_COUNTER_FIELDS[index], values[index]))
}

pub(super) fn accumulate_preflight_values(
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

pub(super) fn origin_counter_mut(
    counters: &mut L1OriginCounters,
    field: L1CounterField,
) -> &mut u64 {
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

pub(super) fn add_origin_fields_checked(
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

pub(super) fn origin_deltas(counters: &L1OriginCounters) -> [(L1CounterField, u64); 19] {
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

pub(super) fn unified_deltas(counters: &L1Counters) -> [(L1CounterField, u64); 19] {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NextSequence {
    Available(u64),
    Exhausted,
}

impl NextSequence {
    pub(super) fn current(self) -> u64 {
        match self {
            Self::Available(current) => current,
            Self::Exhausted => u64::MAX,
        }
    }

    pub(super) fn preflight(self, sequence: &'static str, uses: u64) -> Result<(), L1AccessError> {
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

    pub(super) fn allocate(self, sequence: &'static str) -> Result<(u64, Self), L1AccessError> {
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
pub(super) enum FillDue {
    /// Constant-latency fill due at this exact cycle. `u64::MAX` is valid.
    Timed(u64),
    /// Event-driven fill whose completion is owned by the next level.
    Untimed,
}

impl FillDue {
    pub(super) fn timed(self) -> Option<u64> {
        match self {
            Self::Timed(time) => Some(time),
            Self::Untimed => None,
        }
    }

    pub(super) fn timing(self) -> FillTiming {
        match self {
            Self::Timed(_) => FillTiming::Constant,
            Self::Untimed => FillTiming::Event,
        }
    }
}

pub(super) enum SingleOutcome {
    Done(AccessOutcome, Option<u64>),
    Stall { retry_at: u64 },
}

/// The one shared classification result of [`L1Cache::access_core`];
/// each fill-timing mode maps it to its own public outcome type.
pub(super) enum CoreAccess {
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
pub(super) enum WriteAccounting {
    Store,
    Reduction,
}

/// Which fill-timing discipline an [`L1Cache`] instance runs under —
/// latched on first use so the two families can never mix on one
/// instance (they share the in-flight entry maps).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FillTiming {
    /// Fill times known at issue (`now + miss_latency`), applied lazily.
    Constant,
    /// Fill times decided by the next level; entries remain explicitly
    /// untimed until [`L1Cache::complete_fill`].
    Event,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Readiness {
    Observed,
    Unobserved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FillRequest {
    Constant { now: u64, readiness: Readiness },
    Event { now: u64, readiness: Readiness },
}

impl FillRequest {
    pub(super) fn timing(self) -> FillTiming {
        match self {
            Self::Constant { .. } => FillTiming::Constant,
            Self::Event { .. } => FillTiming::Event,
        }
    }

    pub(super) fn hit_ready(self, hit_latency: u64) -> Result<Option<u64>, L1AccessError> {
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

    pub(super) fn fill_due(self, miss_latency: u64) -> Result<FillDue, L1AccessError> {
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

pub(super) fn collect_sector_intervals(
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
