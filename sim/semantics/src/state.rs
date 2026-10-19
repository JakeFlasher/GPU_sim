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

/// Version 15 adds a required shared DRAM service state, preserving its exact
/// fractional-cycle frontier and distinguishing carded bandwidth from the
/// latency-only ablation. Version 14 added each core's cumulative local
/// completion frontier (explicit null means no released CTA barrier).
/// It keeps the lossless payloads from
/// version 3, widens dynamic requester sector indices to u64, and records
/// hidden logical sequence spans and the explicit serialized phase used by
/// batched queue events. It also preserves the shared-L2 allocation
/// generation associated with each outstanding response, so a delayed
/// response from an evicted same-line allocation cannot be joined to a newer
/// allocation. Older version-2 payloads are not structurally
/// compatible because they carried only a batch count and response kind. It
/// also carries the identity and cohort response mapping for each live bypass
/// asynchronous-copy observation; a count-only projection cannot resume an
/// `AsyncCopy` response without guessing its producer.
pub const TYPED_STATE_SCHEMA_VERSION: u32 = 15;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AddressDomain {
    VirtualAbsolute,
    AllocationRelative,
    Physical,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TypedMemoryOrigin {
    Global,
    Local,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TypedL2Source {
    L1Fill,
    Bypass,
    Untagged,
    Store,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TypedMemoryOperation {
    Read,
    Store,
    Atomic,
    Reduction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TypedResponseKind {
    None,
    L1Fill,
    L1SectorFill,
    Warp,
    AsyncCopy,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CanonicalLineKey {
    VirtualAbsolute {
        line_id: i64,
    },
    AllocationRelative {
        allocation_id: u32,
        line_offset: u64,
    },
    Physical {
        line_id: u64,
    },
    Unknown {
        line_id: Option<i64>,
    },
}

impl CanonicalLineKey {
    pub fn domain(&self) -> AddressDomain {
        match self {
            Self::VirtualAbsolute { .. } => AddressDomain::VirtualAbsolute,
            Self::AllocationRelative { .. } => AddressDomain::AllocationRelative,
            Self::Physical { .. } => AddressDomain::Physical,
            Self::Unknown { .. } => AddressDomain::Unknown,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum L2AccessDisciplineState {
    Immediate,
    Event,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum L2SectorStatus {
    Valid,
    Pending,
    Dirty,
    Stale,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum L2RequestState {
    Outstanding,
    Completed,
    Stale,
    NonReturning,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L2GeometryState {
    pub architecture: String,
    pub capacity_bytes: u64,
    pub line_bytes: u32,
    pub sector_bytes: u32,
    pub num_slices: u32,
    pub sets_per_slice: u64,
    pub associativity: u32,
    pub replacement: String,
    pub slice_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L2CounterState {
    pub hits: u64,
    pub misses: u64,
    pub lookup_hits: u64,
    pub lookup_requests: u64,
    pub line_misses: u64,
    pub sector_fill_misses: u64,
    pub write_tag_hits: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L2WayState {
    pub line: CanonicalLineKey,
    pub valid_mask: u8,
    pub dirty_mask: u8,
    pub pending_mask: u8,
    pub recency: i64,
    pub generation: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L2SetState {
    pub slice: u32,
    pub set: u64,
    pub ways: Vec<L2WayState>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L2RequestKey {
    pub sm_id: u32,
    /// Dynamic response warp identity. This remains distinct from
    /// `warp_slot`, which can be reused after a warp retires while its
    /// response is still outstanding.
    pub response_warp: u64,
    pub warp_slot: i32,
    pub instruction: u64,
    /// Dynamic response indices originate in the host `usize` domain. Keep
    /// the typed boundary lossless instead of narrowing a large trace to u32.
    pub sector_index: u64,
    pub response_kind: TypedResponseKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L2SectorOwnership {
    pub sector: u8,
    pub status: L2SectorStatus,
    pub producer: Option<L2RequestKey>,
    pub consumers: Vec<L2RequestKey>,
    pub ready_at: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L2AllocationState {
    pub line: CanonicalLineKey,
    pub generation: u64,
    pub sectors: Vec<L2SectorOwnership>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L2RequestOwner {
    pub request: L2RequestKey,
    pub line: CanonicalLineKey,
    pub sector_mask: u8,
    /// Shared-L2 allocation generation that priced this response. This is a
    /// different identity domain from the L1 fill generation carried by the
    /// response kind. `None` means the response did not traverse a typed L2
    /// allocation and cannot be a pending-sector consumer.
    pub allocation_generation: Option<u64>,
    pub state: L2RequestState,
    pub due: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L2BoundaryState {
    pub schema_version: u32,
    pub architecture: String,
    pub geometry: L2GeometryState,
    pub address_domain: AddressDomain,
    pub clock: i64,
    pub bip_counter: u64,
    pub access_discipline: Option<L2AccessDisciplineState>,
    pub counters: L2CounterState,
    pub observer_lookup_hits: u64,
    pub observer_lookup_requests: u64,
    pub next_generation: Option<u64>,
    pub sets: Vec<L2SetState>,
    pub allocations: Vec<L2AllocationState>,
    pub request_owners: Vec<L2RequestOwner>,
    pub ownership_complete: bool,
    pub unknown_state: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum L1FillDueState {
    Timed { cycle: u64 },
    Untimed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L1LineFillState {
    pub line: i64,
    pub due: L1FillDueState,
    pub sequence: u64,
    pub generation: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L1SectorFillState {
    pub line: i64,
    pub sector: u8,
    pub generation: u32,
    pub due: L1FillDueState,
    pub sequence: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L1CounterState {
    pub hits: u64,
    pub misses: u64,
    pub merges: u64,
    pub already_requested_merges: u64,
    pub unrequested_pending_merges: u64,
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
    pub write_lookup_hits: u64,
    pub reduction_sectors: u64,
    pub reduction_hits: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L1BoundaryState {
    pub schema_version: u32,
    pub architecture: String,
    pub line_capacity: u64,
    pub mshr_count: Option<u32>,
    pub hit_latency: u64,
    pub miss_latency: u64,
    pub line_order: Vec<(u64, i64)>,
    pub stamp_of: Vec<(i64, u64)>,
    pub sectors: Vec<(i64, u8)>,
    pub line_fills: Vec<L1LineFillState>,
    pub sector_fills: Vec<L1SectorFillState>,
    pub line_generations: Vec<(i64, u32)>,
    pub next_stamp: Option<u64>,
    pub fill_sequence: Option<u64>,
    pub active_lookup_batch: Option<(u64, bool)>,
    pub min_fill_due: Option<u64>,
    pub fill_timing: Option<String>,
    pub counters: L1CounterState,
    pub write_observer_present: bool,
}

impl L1BoundaryState {
    pub fn validate(&self) -> Result<(), L1StateError> {
        if self.schema_version != TYPED_STATE_SCHEMA_VERSION {
            return Err(L1StateError::SchemaVersion);
        }
        if self.architecture.is_empty() {
            return Err(L1StateError::ArchitectureIdentity);
        }
        if self.line_capacity == 0 || self.mshr_count == Some(0) {
            return Err(L1StateError::InvalidGeometry);
        }
        if self.hit_latency >= self.miss_latency {
            return Err(L1StateError::InvalidLatencyOrder);
        }
        // The timing discipline is a latched executable mode, not merely a
        // property of currently live fills. Validate it even for an empty
        // cache; otherwise a malformed snapshot can report a complete state
        // and fail only when the next access consults the latch.
        if !matches!(
            self.fill_timing.as_deref(),
            None | Some("constant") | Some("event")
        ) {
            return Err(L1StateError::FillTimingMismatch);
        }
        if self.line_order.len() > usize::try_from(self.line_capacity).unwrap_or(usize::MAX) {
            return Err(L1StateError::LineCapacityExceeded);
        }
        // A late sector child shares the parent line allocation's MSHR only
        // when both identities carry the same generation. Keep the projected
        // occupancy calculation aligned with the executable L1 model even
        // though the base MSHR value is not a certification gate: access-local
        // LDGSTS floors may legally exceed that configured baseline.
        let _mshr_in_use = self
            .line_fills
            .len()
            .checked_add(
                self.sector_fills
                    .iter()
                    .filter(|fill| {
                        !self.line_fills.iter().any(|parent| {
                            parent.line == fill.line && parent.generation == fill.generation
                        })
                    })
                    .count(),
            )
            .ok_or(L1StateError::MshrCountExceeded)?;
        // `mshr_count` is the card's base limit. Native SM86 cacheable
        // LDGSTS accesses may use an access-local admission floor without
        // changing this serialized configuration field; the typed projection
        // therefore checks shape and occupancy but does not reject that legal
        // per-access policy.
        let miss_composition = self
            .counters
            .line_alloc_misses
            .checked_add(self.counters.sector_resident_misses)
            .and_then(|value| value.checked_add(self.counters.merges))
            .ok_or(L1StateError::CounterOverflow)?;
        if self.counters.misses != miss_composition {
            return Err(L1StateError::MissCompositionMismatch {
                misses: self.counters.misses,
                components: miss_composition,
            });
        }
        if self.counters.pending_lookup_hits > self.counters.merges {
            return Err(L1StateError::PendingLookupHitsExceedMerges);
        }
        if self.counters.bypass_atom_sectors > self.counters.bypass_sectors {
            return Err(L1StateError::BypassAtomsExceedBypass);
        }
        if self.counters.store_hits > self.counters.store_sectors {
            return Err(L1StateError::StoreHitsExceedStores);
        }
        if self.counters.reduction_hits > self.counters.reduction_sectors {
            return Err(L1StateError::ReductionHitsExceedReductions);
        }
        let mut stamps = BTreeMap::new();
        let mut previous_line_order = None;
        for &(stamp, line) in &self.line_order {
            if let Some(previous) = previous_line_order
                && (stamp, line) <= previous
            {
                return Err(L1StateError::UnsortedLineOrder);
            }
            if stamps.insert(stamp, line).is_some() {
                return Err(L1StateError::DuplicateStamp);
            }
            previous_line_order = Some((stamp, line));
        }
        let mut previous_stamp_of = None;
        for &(line, stamp) in &self.stamp_of {
            if let Some(previous) = previous_stamp_of
                && line <= previous
            {
                return Err(L1StateError::UnsortedStampOf);
            }
            if stamps.get(&stamp) != Some(&line) {
                return Err(L1StateError::StampIdentityMismatch);
            }
            previous_stamp_of = Some(line);
        }
        if self.line_order.len() != self.stamp_of.len() {
            return Err(L1StateError::StampCardinalityMismatch);
        }
        let resident_lines = self
            .stamp_of
            .iter()
            .map(|&(line, _)| line)
            .collect::<BTreeSet<_>>();
        if let Some(fill) = self
            .line_fills
            .iter()
            .find(|fill| resident_lines.contains(&fill.line))
        {
            return Err(L1StateError::ResidentLineFillConflict { line: fill.line });
        }
        let mut previous_sector = None;
        for &(line, mask) in &self.sectors {
            if let Some(previous) = previous_sector
                && line <= previous
            {
                return Err(L1StateError::UnsortedSectors);
            }
            if mask == 0 || mask & !0x0f != 0 {
                return Err(L1StateError::InvalidSectorMask);
            }
            previous_sector = Some(line);
        }
        let mut previous_fill = None;
        let mut seen_line_fills = BTreeSet::new();
        for fill in &self.line_fills {
            if self
                .line_generations
                .iter()
                .find(|&&(line, _)| line == fill.line)
                .map(|&(_, generation)| generation)
                != Some(fill.generation)
            {
                return Err(L1StateError::LineFillGenerationMismatch);
            }
            if u64::from(fill.generation) != fill.sequence {
                return Err(L1StateError::LineFillSequenceMismatch);
            }
            let key = (fill.line, fill.generation);
            if !seen_line_fills.insert(key) {
                return Err(L1StateError::DuplicateLineFill);
            }
            if let Some(previous) = previous_fill
                && (fill.line, fill.generation, fill.sequence) <= previous
            {
                return Err(L1StateError::UnsortedLineFills);
            }
            previous_fill = Some((fill.line, fill.generation, fill.sequence));
        }
        let mut previous_sector_fill = None;
        let mut seen_sector_fills = BTreeSet::new();
        let mut fill_sequences = BTreeSet::new();
        for fill in &self.sector_fills {
            if fill.sector >= 4 {
                return Err(L1StateError::InvalidFillSector);
            }
            let key = (fill.line, fill.sector, fill.generation);
            if !seen_sector_fills.insert(key) {
                return Err(L1StateError::DuplicateSectorFill);
            }
            if !fill_sequences.insert(fill.sequence) {
                return Err(L1StateError::DuplicateFillSequence);
            }
            if let Some(previous) = previous_sector_fill
                && (fill.line, fill.sector, fill.generation, fill.sequence) <= previous
            {
                return Err(L1StateError::UnsortedSectorFills);
            }
            previous_sector_fill = Some((fill.line, fill.sector, fill.generation, fill.sequence));
        }
        for fill in &self.line_fills {
            if !fill_sequences.insert(fill.sequence) {
                return Err(L1StateError::DuplicateFillSequence);
            }
        }
        let mut previous_generation = None;
        let mut seen_generations = BTreeSet::new();
        let mut seen_generation_values = BTreeSet::new();
        for &(line, _) in &self.line_generations {
            if !seen_generations.insert(line) {
                return Err(L1StateError::DuplicateGeneration);
            }
            if let Some(previous) = previous_generation
                && line <= previous
            {
                return Err(L1StateError::UnsortedGenerations);
            }
            previous_generation = Some(line);
        }
        for &(_, generation) in &self.line_generations {
            if !seen_generation_values.insert(generation) {
                return Err(L1StateError::DuplicateGenerationValue);
            }
        }
        let lines_with_live_identity = self
            .stamp_of
            .iter()
            .map(|&(line, _)| line)
            .chain(self.line_fills.iter().map(|fill| fill.line))
            .collect::<BTreeSet<_>>();
        for &(line, _) in &self.sectors {
            if !lines_with_live_identity.contains(&line) {
                return Err(L1StateError::SectorLineWithoutAllocation { line });
            }
        }
        for &(line, generation) in &self.line_generations {
            if !lines_with_live_identity.contains(&line) {
                return Err(L1StateError::GenerationWithoutAllocation { line, generation });
            }
        }
        // The fill-timing latch is part of the executable cache identity even
        // after every fill has retired.  A resident line or generation proves
        // that a read path has already selected constant or event timing; an
        // omitted latch would let the next miss choose a mode arbitrarily and
        // therefore cannot certify a resumable boundary.  Store-only empty
        // caches remain valid with no latch because they do not allocate L1
        // lines or fills.
        let has_live_read_allocation = !self.stamp_of.is_empty()
            || !self.sectors.is_empty()
            || !self.line_generations.is_empty();
        if has_live_read_allocation && self.fill_timing.is_none() {
            return Err(L1StateError::FillTimingMissing);
        }
        if let Some(next_stamp) = self.next_stamp
            && self
                .line_order
                .iter()
                .any(|&(stamp, _)| stamp >= next_stamp)
        {
            return Err(L1StateError::NextStampMismatch);
        }
        if let Some(fill_sequence) = self.fill_sequence {
            let line_fill_ahead = self
                .line_fills
                .iter()
                .any(|fill| fill.sequence >= fill_sequence);
            let sector_fill_ahead = self
                .sector_fills
                .iter()
                .any(|fill| fill.sequence >= fill_sequence);
            if line_fill_ahead || sector_fill_ahead {
                return Err(L1StateError::FillSequenceMismatch);
            }
        }
        if let Some((first_fill_sequence, _)) = self.active_lookup_batch
            && first_fill_sequence > self.fill_sequence.unwrap_or(u64::MAX)
        {
            return Err(L1StateError::LookupBatchAheadOfFillSequence);
        }
        if !self.line_fills.is_empty() || !self.sector_fills.is_empty() {
            match self.fill_timing.as_deref() {
                Some("constant")
                    if self
                        .line_fills
                        .iter()
                        .any(|fill| !matches!(fill.due, L1FillDueState::Timed { .. }))
                        || self
                            .sector_fills
                            .iter()
                            .any(|fill| !matches!(fill.due, L1FillDueState::Timed { .. })) =>
                {
                    return Err(L1StateError::FillTimingMismatch);
                }
                Some("event")
                    if self
                        .line_fills
                        .iter()
                        .any(|fill| !matches!(fill.due, L1FillDueState::Untimed))
                        || self
                            .sector_fills
                            .iter()
                            .any(|fill| !matches!(fill.due, L1FillDueState::Untimed)) =>
                {
                    return Err(L1StateError::FillTimingMismatch);
                }
                Some("constant") | Some("event") => {}
                _ => return Err(L1StateError::FillTimingMismatch),
            }
        }
        let timed_min = self
            .line_fills
            .iter()
            .filter_map(|fill| match fill.due {
                L1FillDueState::Timed { cycle } => Some(cycle),
                L1FillDueState::Untimed => None,
            })
            .chain(self.sector_fills.iter().filter_map(|fill| match fill.due {
                L1FillDueState::Timed { cycle } => Some(cycle),
                L1FillDueState::Untimed => None,
            }))
            .min();
        if self.min_fill_due != timed_min {
            return Err(L1StateError::MinimumFillDueMismatch);
        }
        Ok(())
    }

    /// Validate a projected L1 state at the owning SM's current cycle.
    ///
    /// `validate` remains cycle-independent for standalone cache snapshots,
    /// while a typed core boundary must reject a timed MSHR fill that should
    /// already have been retired before that core clock. Event-mode fills are
    /// intentionally untimed and are ordered by the external event queue.
    pub fn validate_at_cycle(&self, cycle: u64) -> Result<(), L1StateError> {
        self.validate()?;
        let due = self
            .line_fills
            .iter()
            .filter_map(|fill| match fill.due {
                L1FillDueState::Timed { cycle } => Some(cycle),
                L1FillDueState::Untimed => None,
            })
            .chain(self.sector_fills.iter().filter_map(|fill| match fill.due {
                L1FillDueState::Timed { cycle } => Some(cycle),
                L1FillDueState::Untimed => None,
            }))
            .find(|&due| due < cycle);
        if let Some(due) = due {
            return Err(L1StateError::FillDueInPast { due, cycle });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum L1StateError {
    #[error("typed L1 schema is unsupported")]
    SchemaVersion,
    #[error("typed L1 architecture identity must be non-empty")]
    ArchitectureIdentity,
    #[error("typed L1 geometry has a zero capacity or MSHR count")]
    InvalidGeometry,
    #[error("typed L1 hit latency must be lower than miss latency")]
    InvalidLatencyOrder,
    #[error("typed L1 resident lines exceed the declared capacity")]
    LineCapacityExceeded,
    #[error("typed L1 line {line} is both resident and in-flight")]
    ResidentLineFillConflict { line: i64 },
    #[error("typed L1 in-flight fills exceed the declared MSHR count")]
    MshrCountExceeded,
    #[error("typed L1 line order is not canonical")]
    UnsortedLineOrder,
    #[error("typed L1 contains duplicate stamps")]
    DuplicateStamp,
    #[error("typed L1 stamp map is not an inverse of line order")]
    StampIdentityMismatch,
    #[error("typed L1 stamp map is not canonical")]
    UnsortedStampOf,
    #[error("typed L1 stamp maps have different cardinalities")]
    StampCardinalityMismatch,
    #[error("typed L1 sectors are not canonical")]
    UnsortedSectors,
    #[error("typed L1 sector mask is invalid")]
    InvalidSectorMask,
    #[error("typed L1 sector state has no resident line or line allocation for line {line}")]
    SectorLineWithoutAllocation { line: i64 },
    #[error(
        "typed L1 generation {generation} has no resident line or line allocation for line {line}"
    )]
    GenerationWithoutAllocation { line: i64, generation: u32 },
    #[error("typed L1 contains duplicate line fills")]
    DuplicateLineFill,
    #[error("typed L1 line fills are not canonical")]
    UnsortedLineFills,
    #[error("typed L1 fill sector is outside the four-sector line")]
    InvalidFillSector,
    #[error("typed L1 contains duplicate sector fills")]
    DuplicateSectorFill,
    #[error("typed L1 sector fills are not canonical")]
    UnsortedSectorFills,
    #[error("typed L1 line generations are not canonical")]
    UnsortedGenerations,
    #[error("typed L1 contains duplicate line generations")]
    DuplicateGeneration,
    #[error("typed L1 reuses one live allocation generation across lines")]
    DuplicateGenerationValue,
    #[error("typed L1 line fill generation disagrees with its live line generation")]
    LineFillGenerationMismatch,
    #[error("typed L1 line fill sequence disagrees with its allocation generation")]
    LineFillSequenceMismatch,
    #[error("typed L1 next stamp is below a live stamp")]
    NextStampMismatch,
    #[error("typed L1 fill sequence is below a live fill")]
    FillSequenceMismatch,
    #[error("typed L1 reuses one fill sequence across live fills")]
    DuplicateFillSequence,
    #[error("typed L1 fill timing discipline disagrees with live fill due states")]
    FillTimingMismatch,
    #[error("typed L1 live read allocation has no fill timing discipline")]
    FillTimingMissing,
    #[error("typed L1 minimum fill due time disagrees with live fills")]
    MinimumFillDueMismatch,
    #[error("typed L1 miss counter {misses} disagrees with components {components}")]
    MissCompositionMismatch { misses: u64, components: u64 },
    #[error("typed L1 pending lookup hits exceed merges")]
    PendingLookupHitsExceedMerges,
    #[error("typed L1 atomic bypass sectors exceed bypass sectors")]
    BypassAtomsExceedBypass,
    #[error("typed L1 store hits exceed store sectors")]
    StoreHitsExceedStores,
    #[error("typed L1 reduction hits exceed reduction sectors")]
    ReductionHitsExceedReductions,
    #[error("typed L1 counter arithmetic overflowed")]
    CounterOverflow,
    #[error("typed L1 lookup batch starts after the fill sequence frontier")]
    LookupBatchAheadOfFillSequence,
    #[error("typed L1 timed fill due at {due} is before core cycle {cycle}")]
    FillDueInPast { due: u64, cycle: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AsyncGroupBoundaryState {
    pub sequence: u64,
    pub committed_at: u64,
    pub pending_sources: u32,
    pub source_completion_time: u64,
    pub delivery_work: u64,
    pub destination_wavefronts: u64,
    pub destination_service_queued: bool,
    pub destination_completion_time: u64,
    pub completion_time: u64,
}

/// Per-cohort response state retained for a live bypass `LDGSTS` copy.  The
/// response timestamps are optional while source sectors remain in flight;
/// once a cohort has drained, both endpoints are required to preserve the
/// response envelope used by the diagnostic projection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AsyncCopyCohortBoundaryState {
    pub pending_sources: u32,
    pub first_source_response: Option<u64>,
    pub last_source_response: Option<u64>,
    pub payload_work: u64,
    pub destination_lines: u64,
    pub bank_wavefronts: u64,
    pub bypass_wavefronts: u64,
    pub topology_penalty: u64,
}

/// Lossless identity and response-to-cohort mapping for one live bypass
/// asynchronous-copy instruction.  Group and instruction form the producer
/// identity used by the response path; sector masks preserve which cohorts a
/// delayed source response must decrement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AsyncCopyBypassObservationState {
    pub group: u64,
    pub instruction: u64,
    pub pending_sources: u32,
    pub source_completion_time: u64,
    pub delivery_work: u64,
    pub source_cohorts: u64,
    pub destination_lines: u64,
    pub max_bank_wavefronts: u64,
    pub max_bypass_wavefronts: u64,
    pub sector_off: u64,
    pub sector_membership_masks: Vec<u32>,
    pub cohorts: Vec<AsyncCopyCohortBoundaryState>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AsyncCopyBoundaryState {
    pub open: Option<AsyncGroupBoundaryState>,
    pub committed: Vec<AsyncGroupBoundaryState>,
    pub bypass_observation_count: u64,
    /// Next per-warp group sequence; `None` represents exhausted u64 space.
    pub next_sequence: Option<u64>,
    /// `false` marks a diagnostic projection that does not retain the
    /// response identity/membership details required for exact replay.
    pub bypass_observations_complete: bool,
    pub bypass_observations: Vec<AsyncCopyBypassObservationState>,
}

impl AsyncCopyBoundaryState {
    pub fn validate(&self) -> Result<(), AsyncCopyStateError> {
        let mut previous = None;
        let mut previous_commit = None;
        for group in &self.committed {
            if let Some(previous) = previous
                && group.sequence <= previous
            {
                return Err(AsyncCopyStateError::UnsortedGroups);
            }
            if let Some(previous_commit) = previous_commit
                && group.committed_at < previous_commit
            {
                return Err(AsyncCopyStateError::CommittedTimeOrder);
            }
            validate_async_group(group)?;
            previous = Some(group.sequence);
            previous_commit = Some(group.committed_at);
        }
        if let Some(group) = &self.open {
            validate_async_group(group)?;
            if previous.is_some_and(|sequence| group.sequence <= sequence) {
                return Err(AsyncCopyStateError::OpenGroupOrder);
            }
        }
        let maximum_sequence = self
            .committed
            .iter()
            .map(|group| group.sequence)
            .chain(self.open.iter().map(|group| group.sequence))
            .max();
        if let (Some(next), Some(maximum)) = (self.next_sequence, maximum_sequence)
            && next <= maximum
        {
            return Err(AsyncCopyStateError::NextSequenceMismatch);
        }
        let captured = u64::try_from(self.bypass_observations.len()).unwrap_or(u64::MAX);
        if self.bypass_observations_complete {
            if self.bypass_observation_count != captured {
                return Err(AsyncCopyStateError::BypassObservationCountMismatch {
                    declared: self.bypass_observation_count,
                    captured,
                });
            }
        } else if captured > self.bypass_observation_count {
            return Err(AsyncCopyStateError::BypassObservationCountMismatch {
                declared: self.bypass_observation_count,
                captured,
            });
        }
        if self.bypass_observations_complete
            && let Some(observation) = self.bypass_observations.iter().find(|observation| {
                observation.source_cohorts == 0
                    || observation.sector_membership_masks.is_empty()
                    || observation.cohorts.is_empty()
            })
        {
            return Err(AsyncCopyStateError::BypassObservationDetailMissing {
                group: observation.group,
                instruction: observation.instruction,
            });
        }
        // A partial projection may omit records, but any records it does
        // carry still participate in the executable identity and aggregate
        // contracts. Validate those records even when the completeness flag
        // remains false; otherwise malformed captured observations can hide
        // behind the diagnostic incomplete status.
        self.validate_bypass_observations()?;
        Ok(())
    }

    pub fn validate_at_cycle(&self, cycle: u64) -> Result<(), AsyncCopyStateError> {
        self.validate()?;
        if let Some(group) = self
            .committed
            .iter()
            .find(|group| group.committed_at > cycle)
        {
            return Err(AsyncCopyStateError::CommittedTimeAfterCoreCycle {
                committed_at: group.committed_at,
                cycle,
            });
        }
        if let Some(group) = self.open.as_ref()
            && group.committed_at > cycle
        {
            return Err(AsyncCopyStateError::CommittedTimeAfterCoreCycle {
                committed_at: group.committed_at,
                cycle,
            });
        }
        Ok(())
    }

    fn validate_bypass_observations(&self) -> Result<(), AsyncCopyStateError> {
        let mut previous = None;
        for observation in &self.bypass_observations {
            let Some(group) = self
                .committed
                .iter()
                .find(|group| group.sequence == observation.group)
                .or_else(|| {
                    self.open
                        .as_ref()
                        .filter(|group| group.sequence == observation.group)
                })
            else {
                return Err(AsyncCopyStateError::BypassObservationGroupMissing {
                    group: observation.group,
                });
            };
            if u64::from(observation.pending_sources) > u64::from(group.pending_sources) {
                return Err(AsyncCopyStateError::BypassObservationGroupBound {
                    group: observation.group,
                    instruction: observation.instruction,
                    field: "pending_sources",
                    group_value: u64::from(group.pending_sources),
                    observation_value: u64::from(observation.pending_sources),
                });
            }
            if observation.source_completion_time > group.source_completion_time {
                return Err(AsyncCopyStateError::BypassObservationGroupBound {
                    group: observation.group,
                    instruction: observation.instruction,
                    field: "source_completion_time",
                    group_value: group.source_completion_time,
                    observation_value: observation.source_completion_time,
                });
            }
            if observation.delivery_work > group.delivery_work {
                return Err(AsyncCopyStateError::BypassObservationGroupBound {
                    group: observation.group,
                    instruction: observation.instruction,
                    field: "delivery_work",
                    group_value: group.delivery_work,
                    observation_value: observation.delivery_work,
                });
            }
            let key = (observation.group, observation.instruction);
            if let Some(previous) = previous
                && key <= previous
            {
                return Err(AsyncCopyStateError::BypassObservationOrder {
                    group: observation.group,
                    instruction: observation.instruction,
                });
            }
            previous = Some(key);
            if observation.delivery_work == 0 {
                return Err(AsyncCopyStateError::BypassObservationZeroWork {
                    group: observation.group,
                    instruction: observation.instruction,
                });
            }
            if !observation.cohorts.is_empty()
                && observation.source_cohorts
                    != u64::try_from(observation.cohorts.len()).unwrap_or(u64::MAX)
            {
                return Err(AsyncCopyStateError::BypassObservationAggregateMismatch {
                    group: observation.group,
                    instruction: observation.instruction,
                    field: "source_cohorts",
                    expected: u64::try_from(observation.cohorts.len()).unwrap_or(u64::MAX),
                    got: observation.source_cohorts,
                });
            }
            if !observation.sector_membership_masks.is_empty()
                && observation.pending_sources
                    > u32::try_from(observation.sector_membership_masks.len()).unwrap_or(u32::MAX)
            {
                return Err(AsyncCopyStateError::BypassObservationPendingSources {
                    group: observation.group,
                    instruction: observation.instruction,
                });
            }
            let sector_len =
                u64::try_from(observation.sector_membership_masks.len()).unwrap_or(u64::MAX);
            observation.sector_off.checked_add(sector_len).ok_or(
                AsyncCopyStateError::BypassObservationSectorOverflow {
                    group: observation.group,
                    instruction: observation.instruction,
                },
            )?;
            if observation.cohorts.is_empty() {
                if !observation.sector_membership_masks.is_empty() {
                    return Err(AsyncCopyStateError::BypassObservationAggregateMismatch {
                        group: observation.group,
                        instruction: observation.instruction,
                        field: "empty cohort membership",
                        expected: 0,
                        got: u64::try_from(observation.sector_membership_masks.len())
                            .unwrap_or(u64::MAX),
                    });
                }
                continue;
            }
            if observation.sector_membership_masks.is_empty() {
                return Err(AsyncCopyStateError::BypassObservationMembershipMissing {
                    group: observation.group,
                    instruction: observation.instruction,
                });
            }
            let allowed_mask = if observation.cohorts.len() == u32::BITS as usize {
                u32::MAX
            } else if observation.cohorts.len() < u32::BITS as usize {
                (1_u32 << observation.cohorts.len()) - 1
            } else {
                return Err(AsyncCopyStateError::BypassObservationCohortCount {
                    group: observation.group,
                    instruction: observation.instruction,
                });
            };
            let mut memberships = vec![0_u64; observation.cohorts.len()];
            for &mask in &observation.sector_membership_masks {
                if mask == 0 || mask & !allowed_mask != 0 {
                    return Err(AsyncCopyStateError::BypassObservationMembershipInvalid {
                        group: observation.group,
                        instruction: observation.instruction,
                    });
                }
                for (index, membership) in memberships.iter_mut().enumerate() {
                    if mask & (1_u32 << index) != 0 {
                        *membership = membership.checked_add(1).ok_or(
                            AsyncCopyStateError::BypassObservationMembershipOverflow {
                                group: observation.group,
                                instruction: observation.instruction,
                            },
                        )?;
                    }
                }
            }
            let mut payload_work = 0_u64;
            let mut destination_lines = 0_u64;
            let mut max_bank_wavefronts = 0_u64;
            let mut max_bypass_wavefronts = 0_u64;
            for (index, cohort) in observation.cohorts.iter().enumerate() {
                let membership = memberships[index];
                if u64::from(cohort.pending_sources) > membership {
                    return Err(AsyncCopyStateError::BypassObservationCohortPending {
                        group: observation.group,
                        instruction: observation.instruction,
                    });
                }
                let completed = membership - u64::from(cohort.pending_sources);
                match completed {
                    0 if cohort.first_source_response.is_some()
                        || cohort.last_source_response.is_some() =>
                    {
                        return Err(AsyncCopyStateError::BypassObservationResponseTimes {
                            group: observation.group,
                            instruction: observation.instruction,
                        });
                    }
                    0 => {}
                    _ if cohort.first_source_response.is_none()
                        || cohort.last_source_response.is_none() =>
                    {
                        return Err(AsyncCopyStateError::BypassObservationResponseTimes {
                            group: observation.group,
                            instruction: observation.instruction,
                        });
                    }
                    _ => {
                        let first = cohort.first_source_response.expect("checked above");
                        let last = cohort.last_source_response.expect("checked above");
                        if first > last || last > observation.source_completion_time {
                            return Err(AsyncCopyStateError::BypassObservationResponseTimes {
                                group: observation.group,
                                instruction: observation.instruction,
                            });
                        }
                    }
                }
                payload_work = payload_work.checked_add(cohort.payload_work).ok_or(
                    AsyncCopyStateError::BypassObservationAggregateOverflow {
                        group: observation.group,
                        instruction: observation.instruction,
                    },
                )?;
                destination_lines = destination_lines
                    .checked_add(cohort.destination_lines)
                    .ok_or(AsyncCopyStateError::BypassObservationAggregateOverflow {
                        group: observation.group,
                        instruction: observation.instruction,
                    })?;
                max_bank_wavefronts = max_bank_wavefronts.max(cohort.bank_wavefronts);
                max_bypass_wavefronts = max_bypass_wavefronts.max(cohort.bypass_wavefronts);
            }
            if payload_work != observation.delivery_work {
                return Err(AsyncCopyStateError::BypassObservationAggregateMismatch {
                    group: observation.group,
                    instruction: observation.instruction,
                    field: "delivery_work",
                    expected: payload_work,
                    got: observation.delivery_work,
                });
            }
            // The compact observation aggregates source-bearing cohorts and
            // executing source-ignored (zero-fill) destination cohorts.  The
            // sparse boundary sidecar intentionally carries only source
            // cohorts because zero-fill lanes have no source-response event.
            // A source-only sum may therefore be lower, but must never exceed
            // the aggregate topology published in the compact observation.
            if destination_lines > observation.destination_lines
                || max_bank_wavefronts > observation.max_bank_wavefronts
                || max_bypass_wavefronts > observation.max_bypass_wavefronts
            {
                return Err(AsyncCopyStateError::BypassObservationAggregateMismatch {
                    group: observation.group,
                    instruction: observation.instruction,
                    field: "source cohort topology exceeds aggregate",
                    expected: observation.destination_lines,
                    got: destination_lines,
                });
            }
        }
        Ok(())
    }
}

fn validate_async_group(group: &AsyncGroupBoundaryState) -> Result<(), AsyncCopyStateError> {
    if group.source_completion_time > group.completion_time {
        return Err(AsyncCopyStateError::TimestampOrder);
    }
    if group.destination_service_queued {
        if group.pending_sources != 0 {
            return Err(AsyncCopyStateError::DestinationPendingSources);
        }
        if group.destination_wavefronts == 0
            || group.destination_completion_time == 0
            || group.destination_completion_time > group.completion_time
        {
            return Err(AsyncCopyStateError::DestinationService);
        }
    } else if group.destination_completion_time != 0 {
        return Err(AsyncCopyStateError::DestinationService);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AsyncCopyStateError {
    #[error("typed async-copy committed groups are not strictly ordered")]
    UnsortedGroups,
    #[error("typed async-copy committed groups are not ordered by commit time")]
    CommittedTimeOrder,
    #[error("typed async-copy committed time {committed_at} is after core cycle {cycle}")]
    CommittedTimeAfterCoreCycle { committed_at: u64, cycle: u64 },
    #[error("typed async-copy open group is older than a committed group")]
    OpenGroupOrder,
    #[error("typed async-copy group timestamps are not ordered")]
    TimestampOrder,
    #[error("typed async-copy destination service state is inconsistent")]
    DestinationService,
    #[error("typed async-copy destination service was queued before all sources completed")]
    DestinationPendingSources,
    #[error("typed async-copy next sequence does not follow its live groups")]
    NextSequenceMismatch,
    #[error(
        "typed async-copy bypass observation count {declared} disagrees with captured {captured}"
    )]
    BypassObservationCountMismatch { declared: u64, captured: u64 },
    #[error("typed async-copy bypass observation belongs to missing group {group}")]
    BypassObservationGroupMissing { group: u64 },
    #[error(
        "typed async-copy bypass observation {group}/{instruction} is marked complete without cohort detail"
    )]
    BypassObservationDetailMissing { group: u64, instruction: u64 },
    #[error(
        "typed async-copy bypass observation {group}/{instruction} exceeds its group {field} bound: group={group_value}, observation={observation_value}"
    )]
    BypassObservationGroupBound {
        group: u64,
        instruction: u64,
        field: &'static str,
        group_value: u64,
        observation_value: u64,
    },
    #[error(
        "typed async-copy bypass observations are not in canonical order at group {group}, instruction {instruction}"
    )]
    BypassObservationOrder { group: u64, instruction: u64 },
    #[error("typed async-copy bypass observation {group}/{instruction} has zero delivery work")]
    BypassObservationZeroWork { group: u64, instruction: u64 },
    #[error(
        "typed async-copy bypass observation {group}/{instruction} has inconsistent cohort count"
    )]
    BypassObservationCohortCount { group: u64, instruction: u64 },
    #[error(
        "typed async-copy bypass observation {group}/{instruction} has missing sector membership"
    )]
    BypassObservationMembershipMissing { group: u64, instruction: u64 },
    #[error(
        "typed async-copy bypass observation {group}/{instruction} has invalid sector membership"
    )]
    BypassObservationMembershipInvalid { group: u64, instruction: u64 },
    #[error("typed async-copy bypass observation {group}/{instruction} membership overflows")]
    BypassObservationMembershipOverflow { group: u64, instruction: u64 },
    #[error(
        "typed async-copy bypass observation {group}/{instruction} has inconsistent pending sources"
    )]
    BypassObservationPendingSources { group: u64, instruction: u64 },
    #[error("typed async-copy bypass observation {group}/{instruction} sector span overflows")]
    BypassObservationSectorOverflow { group: u64, instruction: u64 },
    #[error(
        "typed async-copy bypass observation {group}/{instruction} has inconsistent cohort pending state"
    )]
    BypassObservationCohortPending { group: u64, instruction: u64 },
    #[error(
        "typed async-copy bypass observation {group}/{instruction} has inconsistent response timestamps"
    )]
    BypassObservationResponseTimes { group: u64, instruction: u64 },
    #[error(
        "typed async-copy bypass observation {group}/{instruction} aggregate arithmetic overflows"
    )]
    BypassObservationAggregateOverflow { group: u64, instruction: u64 },
    #[error(
        "typed async-copy bypass observation {group}/{instruction} aggregate {field} is inconsistent"
    )]
    BypassObservationAggregateMismatch {
        group: u64,
        instruction: u64,
        field: &'static str,
        expected: u64,
        got: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryPipelineBoundaryState {
    pub queue_entries_with_dispatch: u32,
    pub local_initiation_interval_cycles: u64,
    pub shared_accept_interval_cycles: u64,
    pub queue_release_times: Vec<Vec<u64>>,
    pub local_completion_ready: Vec<u64>,
    pub shared_accept_ready: u64,
    pub instructions_admitted: u64,
    pub operand_pipeline_delay_cycles: u64,
    pub shared_queue_delay_cycles: u64,
    pub maximum_operand_pipeline_delay_cycles: u64,
    pub maximum_shared_queue_delay_cycles: u64,
    pub maximum_queue_occupancy: u64,
    pub terminal_shared_accept_cycle: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalarPipelineBoundaryState {
    pub latency_cycles: u64,
    pub initiation_interval_cycles: u64,
    pub in_flight_capacity: u32,
    pub release_times: Vec<Vec<u64>>,
    pub next_service: Vec<u64>,
    pub instructions_admitted: u64,
    pub service_busy_cycles: u64,
    pub queue_delay_cycles: u64,
    pub maximum_queue_occupancy: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstructionCacheEntryState {
    pub function_unique_id: i32,
    pub line: u64,
    pub ready_at: u64,
    pub last_access: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstructionCacheState {
    pub ways: u64,
    pub access_sequence: u64,
    pub sets: Vec<Vec<InstructionCacheEntryState>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstructionStreamLineState {
    pub function_unique_id: i32,
    pub line: u64,
    pub ready_at: u64,
    pub demanded: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstructionStreamState {
    pub lines: Vec<InstructionStreamLineState>,
    pub next_line: Option<(i32, u64)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstructionFrontEndBoundaryState {
    pub line_bytes: u32,
    pub l0_to_l1_cycles: u64,
    pub l1_to_l0_cycles: u64,
    pub l1_request_ports: u32,
    pub l1_reply_ports: u32,
    pub stream_buffer_lines: u32,
    pub stream_buffers_per_subcore: u32,
    pub prefetches_per_cycle: u32,
    pub instruction_buffer_entries: u32,
    pub fetch_decode_width: u32,
    pub fetch_to_issue_stages: u64,
    pub l0: Vec<InstructionCacheState>,
    pub l1: InstructionCacheState,
    pub streams: Vec<InstructionStreamState>,
    pub l1_request_ready: u64,
    pub l1_reply_ready: u64,
    pub l0_response_ready: Vec<u64>,
    pub prefetch_priority: u64,
    pub cold_fill_latency: u64,
    pub instructions_fetched: u64,
    pub demand_l0_hits: u64,
    pub demand_stream_buffer_hits: u64,
    pub demand_l0_misses: u64,
    pub demand_l1_hits: u64,
    pub demand_l1_misses: u64,
    pub prefetch_requests: u64,
    pub prefetch_l1_hits: u64,
    pub prefetch_l1_misses: u64,
    pub stream_restarts: u64,
    pub maximum_fetch_latency_cycles: u64,
}

impl InstructionCacheState {
    fn validate(&self) -> Result<(), FrontEndStateError> {
        if self.ways == 0 || self.sets.is_empty() {
            return Err(FrontEndStateError::CacheShape);
        }
        let mut access_sequences = BTreeSet::new();
        for (set_index, set) in self.sets.iter().enumerate() {
            if set.len() > usize::try_from(self.ways).unwrap_or(usize::MAX) {
                return Err(FrontEndStateError::CacheCapacity);
            }
            let mut keys = BTreeSet::new();
            for entry in set {
                if entry.function_unique_id <= 0 {
                    return Err(FrontEndStateError::FunctionIdentity {
                        function_unique_id: entry.function_unique_id,
                    });
                }
                if !keys.insert((entry.function_unique_id, entry.line)) {
                    return Err(FrontEndStateError::DuplicateCacheEntry);
                }
                let function = u64::from(entry.function_unique_id.unsigned_abs());
                let expected_set = (entry.line ^ function.rotate_left(17))
                    % u64::try_from(self.sets.len()).unwrap_or(u64::MAX);
                if expected_set != u64::try_from(set_index).unwrap_or(u64::MAX) {
                    return Err(FrontEndStateError::CacheSetPlacement {
                        expected: expected_set,
                        found: set_index,
                    });
                }
                if !access_sequences.insert(entry.last_access) {
                    return Err(FrontEndStateError::DuplicateCacheAccessSequence);
                }
                if entry.last_access >= self.access_sequence {
                    return Err(FrontEndStateError::CacheAccessSequence);
                }
            }
        }
        Ok(())
    }
}

impl InstructionStreamState {
    fn validate(&self) -> Result<(), FrontEndStateError> {
        for line in &self.lines {
            if line.function_unique_id <= 0 {
                return Err(FrontEndStateError::FunctionIdentity {
                    function_unique_id: line.function_unique_id,
                });
            }
        }
        if let Some((function_unique_id, _)) = self.next_line
            && function_unique_id <= 0
        {
            return Err(FrontEndStateError::FunctionIdentity { function_unique_id });
        }
        for window in self.lines.windows(2) {
            let previous = &window[0];
            let current = &window[1];
            let Some(expected_line) = previous.line.checked_add(1) else {
                return Err(FrontEndStateError::StreamOrder);
            };
            if current.function_unique_id != previous.function_unique_id
                || current.line != expected_line
            {
                return Err(FrontEndStateError::StreamOrder);
            }
            if current.ready_at < previous.ready_at {
                return Err(FrontEndStateError::StreamReadyOrder);
            }
        }
        if self.lines.iter().skip(1).any(|line| line.demanded) {
            return Err(FrontEndStateError::DemandedStreamTail);
        }
        if let Some(last) = self.lines.last() {
            let expected = last
                .line
                .checked_add(1)
                .map(|line| (last.function_unique_id, line));
            if self.next_line != expected {
                return Err(FrontEndStateError::StreamNextLine);
            }
        }
        Ok(())
    }
}

impl InstructionFrontEndBoundaryState {
    pub fn validate(&self) -> Result<(), FrontEndStateError> {
        if self.line_bytes == 0
            || self.l0_to_l1_cycles == 0
            || self.l1_to_l0_cycles == 0
            || self.l1_request_ports == 0
            || self.l1_reply_ports == 0
            || self.stream_buffer_lines == 0
            || self.stream_buffers_per_subcore == 0
            || self.prefetches_per_cycle == 0
            || self.instruction_buffer_entries == 0
            || self.fetch_decode_width == 0
            || self.cold_fill_latency == 0
            || self.fetch_to_issue_stages == 0
            || self.l0.len() != 4
            || self.streams.len() != 4
            || self.l0_response_ready.len() != 4
        {
            return Err(FrontEndStateError::Shape);
        }
        for (field, value) in [
            ("l1_request_ports", self.l1_request_ports),
            ("l1_reply_ports", self.l1_reply_ports),
            (
                "stream_buffers_per_subcore",
                self.stream_buffers_per_subcore,
            ),
            ("prefetches_per_cycle", self.prefetches_per_cycle),
            ("fetch_decode_width", self.fetch_decode_width),
        ] {
            if value != 1 {
                return Err(FrontEndStateError::UnsupportedWidth { field, value });
            }
        }
        if self.l1_reply_ready < self.l1_request_ready {
            return Err(FrontEndStateError::ReplyTimerBeforeRequest);
        }
        let l1_has_access = self.l1.access_sequence != 0;
        if l1_has_access != (self.l1_request_ready != 0 || self.l1_reply_ready != 0) {
            return Err(FrontEndStateError::L1TimerSequenceMismatch);
        }
        if self
            .l0
            .iter()
            .zip(&self.l0_response_ready)
            .any(|(cache, &ready)| (cache.access_sequence != 0) != (ready != 0))
        {
            return Err(FrontEndStateError::L0TimerSequenceMismatch);
        }
        if self
            .l0
            .iter()
            .skip(1)
            .any(|cache| cache.ways != self.l0[0].ways || cache.sets.len() != self.l0[0].sets.len())
        {
            return Err(FrontEndStateError::L0GeometryMismatch);
        }
        if self.prefetch_priority >= self.streams.len() as u64 {
            return Err(FrontEndStateError::PrefetchPriority);
        }
        self.l1.validate()?;
        for cache in &self.l0 {
            cache.validate()?;
        }
        if self
            .streams
            .iter()
            .any(|stream| stream.lines.len() > self.stream_buffer_lines as usize)
        {
            return Err(FrontEndStateError::StreamCapacity);
        }
        for stream in &self.streams {
            stream.validate()?;
        }
        let l0_access_sequence = self.l0.iter().try_fold(0_u64, |sum, cache| {
            sum.checked_add(cache.access_sequence)
                .ok_or(FrontEndStateError::CounterOverflow)
        })?;
        if l0_access_sequence != self.instructions_fetched {
            return Err(FrontEndStateError::CounterMismatch {
                field: "l0_access_sequence",
                expected: self.instructions_fetched,
                found: l0_access_sequence,
            });
        }
        let demand_categories = self
            .demand_l0_hits
            .checked_add(self.demand_stream_buffer_hits)
            .and_then(|value| value.checked_add(self.demand_l0_misses))
            .ok_or(FrontEndStateError::CounterOverflow)?;
        if demand_categories != self.instructions_fetched {
            return Err(FrontEndStateError::CounterMismatch {
                field: "demand_fetch_categories",
                expected: self.instructions_fetched,
                found: demand_categories,
            });
        }
        let l1_accesses = self
            .demand_l0_misses
            .checked_add(self.prefetch_requests)
            .ok_or(FrontEndStateError::CounterOverflow)?;
        if self.l1.access_sequence != l1_accesses {
            return Err(FrontEndStateError::CounterMismatch {
                field: "l1_access_sequence",
                expected: l1_accesses,
                found: self.l1.access_sequence,
            });
        }
        let demand_l1_outcomes = self
            .demand_l1_hits
            .checked_add(self.demand_l1_misses)
            .ok_or(FrontEndStateError::CounterOverflow)?;
        if demand_l1_outcomes != self.demand_l0_misses {
            return Err(FrontEndStateError::CounterMismatch {
                field: "demand_l1_outcomes",
                expected: self.demand_l0_misses,
                found: demand_l1_outcomes,
            });
        }
        let prefetch_l1_outcomes = self
            .prefetch_l1_hits
            .checked_add(self.prefetch_l1_misses)
            .ok_or(FrontEndStateError::CounterOverflow)?;
        if prefetch_l1_outcomes != self.prefetch_requests {
            return Err(FrontEndStateError::CounterMismatch {
                field: "prefetch_l1_outcomes",
                expected: self.prefetch_requests,
                found: prefetch_l1_outcomes,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum FrontEndStateError {
    #[error("typed instruction front-end shape is invalid")]
    Shape,
    #[error("typed instruction front-end cache has zero ways")]
    CacheWays,
    #[error("typed instruction front-end cache has no sets")]
    CacheShape,
    #[error("typed instruction front-end cache set exceeds its way capacity")]
    CacheCapacity,
    #[error("typed instruction front-end cache has duplicate entries")]
    DuplicateCacheEntry,
    #[error(
        "typed instruction front-end function identity must be positive, found {function_unique_id}"
    )]
    FunctionIdentity { function_unique_id: i32 },
    #[error("typed instruction front-end cache entry is in set {found}, expected set {expected}")]
    CacheSetPlacement { expected: u64, found: usize },
    #[error("typed instruction front-end cache reuses a last-access sequence")]
    DuplicateCacheAccessSequence,
    #[error("typed instruction front-end cache entry has an unreserved access sequence")]
    CacheAccessSequence,
    #[error("typed instruction front-end stream exceeds its declared capacity")]
    StreamCapacity,
    #[error("typed instruction front-end stream lines are not consecutive")]
    StreamOrder,
    #[error("typed instruction front-end stream ready times are not ordered")]
    StreamReadyOrder,
    #[error("typed instruction front-end stream has a demanded non-head line")]
    DemandedStreamTail,
    #[error("typed instruction front-end next-line frontier disagrees with its stream tail")]
    StreamNextLine,
    #[error("typed instruction front-end L0 caches do not share one geometry")]
    L0GeometryMismatch,
    #[error("typed instruction front-end prefetch priority is outside the subcore domain")]
    PrefetchPriority,
    #[error("typed instruction front-end reply timer is before request timer")]
    ReplyTimerBeforeRequest,
    #[error("typed instruction front-end L1 timer and access sequence disagree")]
    L1TimerSequenceMismatch,
    #[error("typed instruction front-end L0 timer and access sequence disagree")]
    L0TimerSequenceMismatch,
    #[error("typed instruction front-end field {field} has unsupported width {value}")]
    UnsupportedWidth { field: &'static str, value: u32 },
    #[error("typed instruction front-end counter {field} expected {expected}, found {found}")]
    CounterMismatch {
        field: &'static str,
        expected: u64,
        found: u64,
    },
    #[error("typed instruction front-end counter arithmetic overflowed")]
    CounterOverflow,
}

impl L2BoundaryState {
    pub fn validate(&self) -> Result<(), L2StateError> {
        if self.schema_version != TYPED_STATE_SCHEMA_VERSION {
            return Err(L2StateError::SchemaVersion {
                found: self.schema_version,
                expected: TYPED_STATE_SCHEMA_VERSION,
            });
        }
        if self.architecture.is_empty() || self.geometry.architecture != self.architecture {
            return Err(L2StateError::ArchitectureMismatch);
        }
        if !self.ownership_complete && self.unknown_state.is_empty() {
            return Err(L2StateError::IncompleteWithoutUnknown);
        }
        if self
            .unknown_state
            .windows(2)
            .any(|window| window[1] <= window[0])
        {
            return Err(L2StateError::UnknownStateOrder);
        }
        if !matches!(self.geometry.replacement.as_str(), "lru" | "srrip" | "bip") {
            return Err(L2StateError::UnknownReplacement);
        }
        if self.geometry.capacity_bytes == 0
            || self.geometry.line_bytes == 0
            || self.geometry.sector_bytes == 0
            || self.geometry.sets_per_slice == 0
        {
            return Err(L2StateError::InvalidGeometry);
        }
        if self.geometry.num_slices == 0 || self.geometry.associativity == 0 {
            return Err(L2StateError::InvalidGeometry);
        }
        if self.clock < 0 {
            return Err(L2StateError::NegativeClock);
        }
        if self.address_domain == AddressDomain::VirtualAbsolute
            && !matches!(
                self.geometry.slice_hash.as_str(),
                "line_mod"
                    | "sector_mod"
                    | "xor_folded"
                    | "knuth_mult"
                    | "shift_mod"
                    | "bit_reverse"
            )
        {
            return Err(L2StateError::UnknownSliceHash);
        }
        if !self
            .geometry
            .line_bytes
            .is_multiple_of(self.geometry.sector_bytes)
        {
            return Err(L2StateError::InvalidGeometry);
        }
        let expected_capacity = u128::from(self.geometry.line_bytes)
            .checked_mul(u128::from(self.geometry.num_slices))
            .and_then(|value| value.checked_mul(u128::from(self.geometry.sets_per_slice)))
            .and_then(|value| value.checked_mul(u128::from(self.geometry.associativity)))
            .ok_or(L2StateError::InvalidGeometry)?;
        if expected_capacity != u128::from(self.geometry.capacity_bytes) {
            return Err(L2StateError::GeometryMismatch {
                expected: expected_capacity,
                found: self.geometry.capacity_bytes,
            });
        }
        if self.counters.lookup_hits > self.counters.lookup_requests {
            return Err(L2StateError::LookupHitsExceedRequests);
        }
        if self.observer_lookup_hits > self.observer_lookup_requests {
            return Err(L2StateError::ObserverLookupHitsExceedRequests);
        }
        let miss_components = self
            .counters
            .line_misses
            .checked_add(self.counters.sector_fill_misses)
            .ok_or(L2StateError::CounterOverflow)?;
        if self.counters.misses != miss_components {
            return Err(L2StateError::MissCompositionMismatch {
                misses: self.counters.misses,
                components: miss_components,
            });
        }
        if self.counters.write_tag_hits > self.counters.hits {
            return Err(L2StateError::WriteTagHitsExceedHits);
        }
        let mut previous = None;
        let mut seen_sets = BTreeSet::new();
        let mut live_allocations = BTreeSet::new();
        let mut live_generation_values = BTreeSet::new();
        let mut live_ways = BTreeMap::new();
        for set in &self.sets {
            if set.slice >= self.geometry.num_slices || set.set >= self.geometry.sets_per_slice {
                return Err(L2StateError::SetOutOfRange {
                    slice: set.slice,
                    set: set.set,
                });
            }
            if set.ways.len() > self.geometry.associativity as usize {
                return Err(L2StateError::SetOverfull { set: set.set });
            }
            if let Some(previous) = previous
                && (set.slice, set.set) <= previous
            {
                return Err(L2StateError::UnsortedSets);
            }
            if !seen_sets.insert((set.slice, set.set)) {
                return Err(L2StateError::DuplicateSet {
                    slice: set.slice,
                    set: set.set,
                });
            }
            previous = Some((set.slice, set.set));
            let mut seen_lines = BTreeSet::new();
            let mut seen_recencies = BTreeSet::new();
            for way in &set.ways {
                if way.line.domain() != self.address_domain {
                    return Err(L2StateError::AddressDomainMismatch);
                }
                if way.valid_mask == 0 && way.pending_mask == 0 {
                    return Err(L2StateError::EmptyWay {
                        line: way.line.clone(),
                    });
                }
                match self.geometry.replacement.as_str() {
                    // SM86 SRRIP stores a three-bit RRPV with a maximum of
                    // three.  A forged value changes victim selection and
                    // would fail the native access preflight.
                    "srrip" if !(0..=3).contains(&way.recency) => {
                        return Err(L2StateError::InvalidRecency {
                            line: way.line.clone(),
                            recency: way.recency,
                            clock: self.clock,
                            replacement: "srrip",
                        });
                    }
                    // LRU timestamps are nonnegative clock values; BIP uses
                    // the same monotonically advancing clock but may insert
                    // below the current minimum, so negative BIP recencies
                    // are legal.  Both policies require unique live stamps:
                    // native updates advance the clock for every sector and
                    // BIP's decrement is checked before publication.
                    "lru" if way.recency < 0 || way.recency > self.clock => {
                        return Err(L2StateError::InvalidRecency {
                            line: way.line.clone(),
                            recency: way.recency,
                            clock: self.clock,
                            replacement: "lru",
                        });
                    }
                    "bip" if way.recency > self.clock => {
                        return Err(L2StateError::InvalidRecency {
                            line: way.line.clone(),
                            recency: way.recency,
                            clock: self.clock,
                            replacement: "bip",
                        });
                    }
                    _ => {}
                }
                if !matches!(self.geometry.replacement.as_str(), "srrip")
                    && !seen_recencies.insert(way.recency)
                {
                    return Err(L2StateError::DuplicateRecency {
                        slice: set.slice,
                        set: set.set,
                        recency: way.recency,
                    });
                }
                if (way.valid_mask | way.dirty_mask | way.pending_mask) & !0x0f != 0 {
                    return Err(L2StateError::InvalidSectorMask);
                }
                if way.pending_mask & way.valid_mask != 0 {
                    return Err(L2StateError::PendingAndValidOverlap);
                }
                if way.dirty_mask & !way.valid_mask != 0 {
                    return Err(L2StateError::DirtyWithoutValid);
                }
                // Event-discipline allocations use the generation as part of
                // the fill/response identity. A complete boundary cannot
                // silently drop a resident event-mode way (or any pending
                // way) merely because its generation is absent: the omitted
                // allocation would make a later same-line response
                // ambiguous. Immediate-mode ways may legitimately omit this
                // identity because they have no deferred fill ownership.
                if self.ownership_complete
                    && way.generation.is_none()
                    && (matches!(self.access_discipline, Some(L2AccessDisciplineState::Event))
                        || way.pending_mask != 0)
                {
                    return Err(L2StateError::WayGenerationMissing {
                        line: way.line.clone(),
                    });
                }
                if !seen_lines.insert(way.line.clone()) {
                    return Err(L2StateError::DuplicateWayLine { set: set.set });
                }
                if let Some(generation) = way.generation {
                    let identity = (way.line.clone(), generation);
                    if live_ways.insert(identity.clone(), way).is_some() {
                        return Err(L2StateError::DuplicateWayIdentity);
                    }
                    if !live_generation_values.insert(generation) {
                        return Err(L2StateError::DuplicateGeneration { generation });
                    }
                    live_allocations.insert(identity);
                }
                if let CanonicalLineKey::VirtualAbsolute { line_id } = &way.line {
                    let (expected_slice, expected_set) = expected_l2_placement(
                        *line_id,
                        &self.geometry.slice_hash,
                        self.geometry.num_slices,
                        self.geometry.sets_per_slice,
                    )?;
                    if (set.slice, set.set) != (expected_slice, expected_set) {
                        return Err(L2StateError::LinePlacementMismatch {
                            line: way.line.clone(),
                            found_slice: set.slice,
                            found_set: set.set,
                            expected_slice,
                            expected_set,
                        });
                    }
                }
            }
        }
        let mut seen_allocations = BTreeSet::new();
        let mut seen_allocation_generation_values = BTreeSet::new();
        let mut duplicate_allocation_generation = None;
        let mut previous_allocation = None;
        for allocation in &self.allocations {
            if allocation.line.domain() != self.address_domain {
                return Err(L2StateError::AddressDomainMismatch);
            }
            let key = (allocation.line.clone(), allocation.generation);
            if !seen_allocations.insert(key) {
                return Err(L2StateError::DuplicateAllocation);
            }
            let order = (&allocation.line, allocation.generation);
            if let Some(previous) = previous_allocation
                && order <= previous
            {
                return Err(L2StateError::UnsortedAllocations);
            }
            previous_allocation = Some(order);
            if !seen_allocation_generation_values.insert(allocation.generation) {
                duplicate_allocation_generation = Some(allocation.generation);
            }
            let mut seen_sectors = BTreeSet::new();
            let mut previous_sector = None;
            for sector in &allocation.sectors {
                if sector.sector >= 4 || !seen_sectors.insert(sector.sector) {
                    return Err(L2StateError::DuplicateAllocationSector);
                }
                if let Some(previous) = previous_sector
                    && sector.sector <= previous
                {
                    return Err(L2StateError::UnsortedAllocationSectors);
                }
                previous_sector = Some(sector.sector);
            }
        }
        let maximum_generation = live_allocations
            .iter()
            .map(|(_, generation)| *generation)
            .chain(
                self.allocations
                    .iter()
                    .map(|allocation| allocation.generation),
            )
            .max();
        if let (Some(next_generation), Some(maximum_generation)) =
            (self.next_generation, maximum_generation)
            && next_generation <= maximum_generation
        {
            return Err(L2StateError::GenerationFrontierMismatch);
        }
        if self.ownership_complete {
            let projected_allocations = self
                .allocations
                .iter()
                .map(|allocation| (allocation.line.clone(), allocation.generation))
                .collect::<BTreeSet<_>>();
            if projected_allocations != live_allocations {
                return Err(L2StateError::AllocationWayMismatch);
            }
        }
        let mut seen_response_identities = BTreeSet::new();
        let mut seen_requests = BTreeSet::new();
        let mut previous_request = None;
        for owner in &self.request_owners {
            if owner.request.sm_id == u32::MAX {
                return Err(L2StateError::InvalidSmIdentity);
            }
            let response_identity = (
                owner.request.sm_id,
                owner.request.response_warp,
                owner.request.warp_slot,
                owner.request.instruction,
                owner.request.sector_index,
            );
            if !seen_response_identities.insert(response_identity) {
                return Err(L2StateError::DuplicateResponseIdentity);
            }
            let key = (
                owner.request.sm_id,
                owner.request.response_warp,
                owner.request.warp_slot,
                owner.request.instruction,
                owner.request.sector_index,
                owner.request.response_kind,
            );
            if !seen_requests.insert(key) {
                return Err(L2StateError::DuplicateRequestOwner);
            }
            let order = (
                owner.request.sm_id,
                owner.request.response_warp,
                owner.request.warp_slot,
                owner.request.instruction,
                owner.request.sector_index,
                &owner.request.response_kind,
            );
            if let Some(previous) = previous_request
                && order <= previous
            {
                return Err(L2StateError::UnsortedRequestOwners);
            }
            previous_request = Some(order);
        }
        let owners_by_request = self
            .request_owners
            .iter()
            .map(|owner| (owner.request.clone(), owner))
            .collect::<BTreeMap<_, _>>();
        if self
            .request_owners
            .iter()
            .any(|owner| matches!(owner.request.response_kind, TypedResponseKind::None))
        {
            return Err(L2StateError::InvalidRequestOwnerKind);
        }
        if let Some(generation) = duplicate_allocation_generation {
            return Err(L2StateError::DuplicateGeneration { generation });
        }
        // A response owner is part of the executable state, even when no
        // pending allocation currently references it.  L1 and direct
        // Warp/AsyncCopy replies must retain the allocation identity that
        // priced the response; otherwise an orphan owner can be marked
        // complete and the next response event is absent from the typed
        // transition state. An out-of-range direct-response identity remains
        // explicitly opaque and cannot certify a complete boundary.
        if self.ownership_complete {
            for owner in &self.request_owners {
                if owner.request.warp_slot < 0 {
                    return Err(L2StateError::NegativeRequestWarpSlot {
                        warp_slot: owner.request.warp_slot,
                    });
                }
                match owner.request.response_kind {
                    TypedResponseKind::L1Fill
                    | TypedResponseKind::L1SectorFill
                    | TypedResponseKind::Warp
                    | TypedResponseKind::AsyncCopy => {
                        if matches!(owner.line, CanonicalLineKey::Unknown { .. }) {
                            if !matches!(
                                owner.request.response_kind,
                                TypedResponseKind::Warp | TypedResponseKind::AsyncCopy
                            ) || owner.sector_mask != 0
                                || owner.allocation_generation.is_some()
                            {
                                return Err(L2StateError::OpaqueRequestOwnerIdentity);
                            }
                            continue;
                        }
                        if owner.sector_mask == 0 || owner.sector_mask & !0x0f != 0 {
                            return Err(L2StateError::InvalidRequestOwnerSectorMask);
                        }
                        if owner.line.domain() != self.address_domain {
                            return Err(L2StateError::RequestOwnerAddressDomainMismatch);
                        }
                        let Some(generation) = owner.allocation_generation else {
                            return Err(L2StateError::RequestOwnerGenerationMissing);
                        };
                        let Some(allocation) = self.allocations.iter().find(|allocation| {
                            allocation.line == owner.line && allocation.generation == generation
                        }) else {
                            return Err(L2StateError::RequestOwnerAllocationMissing);
                        };
                        let allocation_mask = allocation
                            .sectors
                            .iter()
                            .fold(0_u8, |mask, sector| mask | (1_u8 << sector.sector));
                        if owner.sector_mask & !allocation_mask != 0 {
                            return Err(L2StateError::RequestOwnerSectorMismatch);
                        }
                    }
                    TypedResponseKind::None => unreachable!("response-owner kind was validated"),
                }
            }
        }
        for allocation in &self.allocations {
            let Some(way) = live_ways.get(&(allocation.line.clone(), allocation.generation)) else {
                return Err(L2StateError::AllocationWithoutWay);
            };
            let expected = (0..4_u8)
                .filter_map(|sector| {
                    let bit = 1_u8 << sector;
                    let status = if way.pending_mask & bit != 0 {
                        L2SectorStatus::Pending
                    } else if way.dirty_mask & bit != 0 {
                        L2SectorStatus::Dirty
                    } else if way.valid_mask & bit != 0 {
                        L2SectorStatus::Valid
                    } else {
                        return None;
                    };
                    Some((sector, status))
                })
                .collect::<Vec<_>>();
            let actual = allocation
                .sectors
                .iter()
                .map(|sector| (sector.sector, sector.status.clone()))
                .collect::<Vec<_>>();
            if actual != expected {
                return Err(L2StateError::AllocationMaskMismatch {
                    line: allocation.line.clone(),
                    generation: allocation.generation,
                });
            }
            for sector in &allocation.sectors {
                let bit = 1_u8 << sector.sector;
                if !matches!(sector.status, L2SectorStatus::Pending) {
                    if sector.producer.is_some() || !sector.consumers.is_empty() {
                        return Err(L2StateError::NonPendingOwnership {
                            line: allocation.line.clone(),
                            generation: allocation.generation,
                            sector: sector.sector,
                        });
                    }
                    if let Some(ready_at) = sector.ready_at {
                        let has_outstanding_fill_owner = self.request_owners.iter().any(|owner| {
                            owner.line == allocation.line
                                && owner.allocation_generation == Some(allocation.generation)
                                && owner.sector_mask & bit != 0
                                && matches!(owner.state, L2RequestState::Outstanding)
                                && matches!(
                                    owner.request.response_kind,
                                    TypedResponseKind::L1Fill
                                        | TypedResponseKind::L1SectorFill
                                        | TypedResponseKind::Warp
                                        | TypedResponseKind::AsyncCopy
                                )
                        });
                        if !has_outstanding_fill_owner {
                            return Err(L2StateError::ReadyTimeOnNonPending {
                                line: allocation.line.clone(),
                                generation: allocation.generation,
                                sector: sector.sector,
                                ready_at,
                            });
                        }
                    }
                    continue;
                }
                let mut previous_consumer = None;
                let mut seen_consumers = BTreeSet::new();
                for consumer in &sector.consumers {
                    if !seen_consumers.insert(consumer.clone()) {
                        return Err(L2StateError::DuplicateSectorConsumer);
                    }
                    if let Some(previous) = previous_consumer
                        && consumer <= previous
                    {
                        return Err(L2StateError::UnsortedSectorConsumers);
                    }
                    previous_consumer = Some(consumer);
                    let Some(owner) = owners_by_request.get(consumer) else {
                        return Err(L2StateError::OwnershipReferenceMissing);
                    };
                    if self.ownership_complete
                        && !matches!(owner.state, L2RequestState::Outstanding)
                    {
                        return Err(L2StateError::PendingConsumerNotOutstanding {
                            line: allocation.line.clone(),
                            generation: allocation.generation,
                            sector: sector.sector,
                        });
                    }
                    if owner.line != allocation.line
                        || owner.allocation_generation != Some(allocation.generation)
                        || owner.sector_mask & bit == 0
                    {
                        return Err(L2StateError::OwnershipReferenceMismatch {
                            line: allocation.line.clone(),
                            generation: allocation.generation,
                            sector: sector.sector,
                        });
                    }
                }
                if let Some(producer) = &sector.producer {
                    if !seen_consumers.contains(producer) {
                        return Err(L2StateError::ProducerNotConsumer);
                    }
                } else if self.ownership_complete {
                    return Err(L2StateError::CompletePendingWithoutProducer {
                        line: allocation.line.clone(),
                        generation: allocation.generation,
                        sector: sector.sector,
                    });
                }
                if self.ownership_complete && sector.ready_at.is_none() {
                    return Err(L2StateError::CompletePendingWithoutReadyTime {
                        line: allocation.line.clone(),
                        generation: allocation.generation,
                        sector: sector.sector,
                    });
                }
            }
        }
        if self.ownership_complete && !self.unknown_state.is_empty() {
            return Err(L2StateError::CompleteWithUnknownState);
        }
        // Allocation generations are part of the executable discipline. The
        // immediate path never creates them, while every resident event-mode
        // line carries one; accepting a contradictory pairing would make a
        // typed boundary impossible to resume without guessing its token
        // identity. Keep this check after the more specific structural checks
        // above so malformed masks/placements retain their precise errors.
        let has_generation_free_way = self
            .sets
            .iter()
            .flat_map(|set| set.ways.iter())
            .any(|way| way.generation.is_none());
        let has_generation_bearing_way = self
            .sets
            .iter()
            .flat_map(|set| set.ways.iter())
            .any(|way| way.generation.is_some());
        if self.ownership_complete {
            let counters_nonzero = [
                self.counters.hits,
                self.counters.misses,
                self.counters.lookup_hits,
                self.counters.lookup_requests,
                self.counters.line_misses,
                self.counters.sector_fill_misses,
                self.counters.write_tag_hits,
                self.observer_lookup_hits,
                self.observer_lookup_requests,
            ]
            .into_iter()
            .any(|value| value != 0);
            let active_state = !self.sets.is_empty()
                || !self.allocations.is_empty()
                || !self.request_owners.is_empty()
                || self.bip_counter != 0
                || self.next_generation.is_some_and(|next| next != 0)
                || counters_nonzero;
            if self.access_discipline.is_none() && active_state {
                return Err(L2StateError::DisciplineMissing);
            }
            match self.access_discipline {
                None | Some(L2AccessDisciplineState::Immediate) if has_generation_bearing_way => {
                    return Err(L2StateError::DisciplineGenerationMismatch);
                }
                Some(L2AccessDisciplineState::Event) if has_generation_free_way => {
                    return Err(L2StateError::DisciplineGenerationMismatch);
                }
                _ => {}
            }
        }
        Ok(())
    }
}

fn expected_l2_placement(
    line_id: i64,
    slice_hash: &str,
    num_slices: u32,
    sets_per_slice: u64,
) -> Result<(u32, u64), L2StateError> {
    if num_slices == 0 || sets_per_slice == 0 {
        return Err(L2StateError::InvalidGeometry);
    }
    let slices = i128::from(num_slices);
    let sets = i128::from(sets_per_slice);
    let slice = match slice_hash {
        "line_mod" => i128::from(line_id).rem_euclid(slices),
        "sector_mod" => (i128::from(line_id).rem_euclid(slices) * 4).rem_euclid(slices),
        "xor_folded" => i128::from(line_id ^ (line_id >> 5) ^ (line_id >> 11)).rem_euclid(slices),
        "knuth_mult" => {
            i128::from(line_id.wrapping_mul(2_654_435_761) & 0xffff_ffff).rem_euclid(slices)
        }
        "shift_mod" => i128::from(line_id >> 1).rem_euclid(slices),
        "bit_reverse" => i128::from((line_id as u16).reverse_bits()).rem_euclid(slices),
        _ => return Err(L2StateError::UnknownSliceHash),
    };
    let set = i128::from(line_id).div_euclid(slices).rem_euclid(sets);
    Ok((
        u32::try_from(slice).map_err(|_| L2StateError::InvalidGeometry)?,
        u64::try_from(set).map_err(|_| L2StateError::InvalidGeometry)?,
    ))
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum L2StateError {
    #[error("typed L2 requester uses the invalid u32::MAX SM identity sentinel")]
    InvalidSmIdentity,
    #[error("typed L2 schema {found} is unsupported; expected {expected}")]
    SchemaVersion { found: u32, expected: u32 },
    #[error("typed L2 geometry contains a zero dimension")]
    InvalidGeometry,
    #[error("typed L2 access discipline disagrees with resident allocation generations")]
    DisciplineGenerationMismatch,
    #[error("typed L2 access discipline is missing for an active complete state")]
    DisciplineMissing,
    #[error("typed L2 clock is negative")]
    NegativeClock,
    #[error("typed L2 line identity disagrees with the declared address domain")]
    AddressDomainMismatch,
    #[error("typed L2 root and geometry architecture identities disagree")]
    ArchitectureMismatch,
    #[error("typed L2 geometry uses an unknown replacement policy")]
    UnknownReplacement,
    #[error("typed L2 capacity {found} disagrees with geometry product {expected}")]
    GeometryMismatch { expected: u128, found: u64 },
    #[error("typed L2 lookup hits exceed lookup requests")]
    LookupHitsExceedRequests,
    #[error("typed L2 observer lookup hits exceed observer lookup requests")]
    ObserverLookupHitsExceedRequests,
    #[error("typed L2 miss counter {misses} disagrees with components {components}")]
    MissCompositionMismatch { misses: u64, components: u64 },
    #[error("typed L2 write-tag hits exceed hits")]
    WriteTagHitsExceedHits,
    #[error("typed L2 counter arithmetic overflowed")]
    CounterOverflow,
    #[error("typed L2 set ({slice},{set}) is outside declared geometry")]
    SetOutOfRange { slice: u32, set: u64 },
    #[error("typed L2 set {set} contains too many ways")]
    SetOverfull { set: u64 },
    #[error("typed L2 sets are not in canonical order")]
    UnsortedSets,
    #[error("typed L2 contains duplicate set ({slice},{set})")]
    DuplicateSet { slice: u32, set: u64 },
    #[error("typed L2 set {set} contains duplicate way lines")]
    DuplicateWayLine { set: u64 },
    #[error("typed L2 way {line:?} has neither valid nor pending sectors")]
    EmptyWay { line: CanonicalLineKey },
    #[error("typed L2 {replacement} way {line:?} has invalid recency {recency} at clock {clock}")]
    InvalidRecency {
        line: CanonicalLineKey,
        recency: i64,
        clock: i64,
        replacement: &'static str,
    },
    #[error("typed L2 set ({slice},{set}) contains duplicate recency {recency}")]
    DuplicateRecency { slice: u32, set: u64, recency: i64 },
    #[error("typed L2 contains duplicate resident line/generation identities")]
    DuplicateWayIdentity,
    #[error("typed L2 way contains a sector mask outside the four-sector line")]
    InvalidSectorMask,
    #[error("complete typed L2 way {line:?} has no allocation generation")]
    WayGenerationMissing { line: CanonicalLineKey },
    #[error("typed L2 geometry uses an unknown slice hash")]
    UnknownSliceHash,
    #[error(
        "typed L2 line {line:?} is stored in slice/set ({found_slice},{found_set}), expected ({expected_slice},{expected_set})"
    )]
    LinePlacementMismatch {
        line: CanonicalLineKey,
        found_slice: u32,
        found_set: u64,
        expected_slice: u32,
        expected_set: u64,
    },
    #[error("typed L2 contains duplicate allocation generations")]
    DuplicateAllocation,
    #[error("typed L2 reuses allocation generation {generation} across live lines")]
    DuplicateGeneration { generation: u64 },
    #[error("typed L2 next-generation frontier is not above live generations")]
    GenerationFrontierMismatch,
    #[error("typed L2 allocations are not in canonical line/generation order")]
    UnsortedAllocations,
    #[error("typed L2 allocation contains duplicate or out-of-range sectors")]
    DuplicateAllocationSector,
    #[error("typed L2 allocation sectors are not in ascending order")]
    UnsortedAllocationSectors,
    #[error("typed L2 contains duplicate requester ownership")]
    DuplicateRequestOwner,
    #[error("typed L2 response identity is duplicated")]
    DuplicateResponseIdentity,
    #[error("typed L2 requester ownership is not in canonical order")]
    UnsortedRequestOwners,
    #[error("typed L2 request owner has a negative warp slot {warp_slot}")]
    NegativeRequestWarpSlot { warp_slot: i32 },
    #[error("typed L2 L1 response owner has an invalid sector mask")]
    InvalidRequestOwnerSectorMask,
    #[error("typed L2 L1 response owner uses a different address domain")]
    RequestOwnerAddressDomainMismatch,
    #[error("typed L2 L1 response owner is missing its allocation generation")]
    RequestOwnerGenerationMissing,
    #[error("typed L2 L1 response owner has no matching allocation")]
    RequestOwnerAllocationMissing,
    #[error("typed L2 L1 response owner references sectors outside its allocation")]
    RequestOwnerSectorMismatch,
    #[error("typed L2 opaque response owner carries cache allocation identity")]
    OpaqueRequestOwnerIdentity,
    #[error("typed L2 response owner has no supported response kind")]
    InvalidRequestOwnerKind,
    #[error("typed L2 allocation and resident-way identities disagree")]
    AllocationWayMismatch,
    #[error("typed L2 allocation has no matching resident way")]
    AllocationWithoutWay,
    #[error("typed L2 allocation sector statuses disagree with resident masks")]
    AllocationMaskMismatch {
        line: CanonicalLineKey,
        generation: u64,
    },
    #[error("typed L2 non-pending sector carries requester ownership")]
    NonPendingOwnership {
        line: CanonicalLineKey,
        generation: u64,
        sector: u8,
    },
    #[error("typed L2 non-pending sector carries a ready time without an outstanding fill owner")]
    ReadyTimeOnNonPending {
        line: CanonicalLineKey,
        generation: u64,
        sector: u8,
        ready_at: u64,
    },
    #[error("typed L2 sector consumer is duplicated")]
    DuplicateSectorConsumer,
    #[error("complete typed L2 pending sector references a non-outstanding request owner")]
    PendingConsumerNotOutstanding {
        line: CanonicalLineKey,
        generation: u64,
        sector: u8,
    },
    #[error("typed L2 sector consumers are not in canonical request order")]
    UnsortedSectorConsumers,
    #[error("typed L2 sector ownership references an unknown request")]
    OwnershipReferenceMissing,
    #[error(
        "typed L2 sector ownership references a different line, sector, or allocation generation"
    )]
    OwnershipReferenceMismatch {
        line: CanonicalLineKey,
        generation: u64,
        sector: u8,
    },
    #[error("typed L2 producer is not one of the sector consumers")]
    ProducerNotConsumer,
    #[error("complete typed L2 ownership has a pending sector without a producer")]
    CompletePendingWithoutProducer {
        line: CanonicalLineKey,
        generation: u64,
        sector: u8,
    },
    #[error("complete typed L2 ownership has a pending sector without a ready time")]
    CompletePendingWithoutReadyTime {
        line: CanonicalLineKey,
        generation: u64,
        sector: u8,
    },
    #[error("typed L2 way has pending and valid sectors in the same bit")]
    PendingAndValidOverlap,
    #[error("typed L2 way has dirty sectors that are not valid")]
    DirtyWithoutValid,
    #[error("typed L2 state is marked complete while unknown fields remain")]
    CompleteWithUnknownState,
    #[error("typed L2 state is incomplete without an unknown-state reason")]
    IncompleteWithoutUnknown,
    #[error("typed L2 unknown-state reasons are not in canonical order")]
    UnknownStateOrder,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PowerModelStatus {
    Omitted,
    ActivityOnly,
    AnalyticCoefficients,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PowerDomainStatus {
    Clocked,
    Idle,
    Gated,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerClockState {
    pub clock_domain: String,
    pub fixed_frequency_hz: Option<u64>,
    pub dvfs_modeled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerDomainState {
    pub domain: String,
    pub status: PowerDomainStatus,
    pub last_cycle: u64,
    pub clocked_cycles: u64,
    pub gated_cycles: u64,
    pub sm_wake_events: u64,
    pub completion_events: u64,
    pub memory_events: u64,
    pub queue_events: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerSampleState {
    pub cycle: u64,
    pub activity_total: u64,
    pub domain_clocked_cycles: Vec<u64>,
    pub domain_gated_cycles: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerBoundaryState {
    pub schema_version: u32,
    pub status: PowerModelStatus,
    pub clock: PowerClockState,
    pub sample_period: Option<u64>,
    pub next_sample: Option<u64>,
    pub domains: Vec<PowerDomainState>,
    pub samples: Vec<PowerSampleState>,
    pub omitted_observations: Vec<String>,
    pub unknown_state: Vec<String>,
}

impl PowerBoundaryState {
    pub fn activity_only(frequency_hz: Option<u64>, sample_period: Option<u64>) -> Self {
        let next_sample = sample_period.filter(|period| *period != 0);
        Self {
            schema_version: TYPED_STATE_SCHEMA_VERSION,
            status: PowerModelStatus::ActivityOnly,
            clock: PowerClockState {
                clock_domain: "sm_clock".to_owned(),
                fixed_frequency_hz: frequency_hz,
                dvfs_modeled: false,
            },
            sample_period,
            next_sample,
            domains: vec![PowerDomainState {
                domain: "modeled_event_system".to_owned(),
                status: PowerDomainStatus::Idle,
                last_cycle: 0,
                clocked_cycles: 0,
                gated_cycles: 0,
                sm_wake_events: 0,
                completion_events: 0,
                memory_events: 0,
                queue_events: 0,
            }],
            samples: Vec::new(),
            omitted_observations: vec![
                "watts".to_owned(),
                "per_instruction_energy".to_owned(),
                "lane_utilization".to_owned(),
                "thermal_and_dvfs_feedback".to_owned(),
            ],
            unknown_state: Vec::new(),
        }
    }

    pub fn advance_to(&mut self, cycle: u64) -> Result<(), PowerStateError> {
        if self.schema_version != TYPED_STATE_SCHEMA_VERSION {
            return Err(PowerStateError::SchemaVersion);
        }
        // Sampling is part of the state transition. Stage the complete update
        // so an overflow or bounded-output refusal never leaves half an
        // integration interval committed.
        let mut next = self.clone();
        next.emit_samples_through(cycle)?;
        next.integrate_to(cycle)?;
        *self = next;
        Ok(())
    }

    fn integrate_to(&mut self, cycle: u64) -> Result<(), PowerStateError> {
        for domain in &mut self.domains {
            if cycle < domain.last_cycle {
                return Err(PowerStateError::TimeTravel);
            }
            let delta = cycle - domain.last_cycle;
            match domain.status {
                PowerDomainStatus::Clocked => {
                    domain.clocked_cycles = domain
                        .clocked_cycles
                        .checked_add(delta)
                        .ok_or(PowerStateError::CounterOverflow)?;
                }
                PowerDomainStatus::Gated => {
                    domain.gated_cycles = domain
                        .gated_cycles
                        .checked_add(delta)
                        .ok_or(PowerStateError::CounterOverflow)?;
                }
                PowerDomainStatus::Idle => {}
            }
            domain.last_cycle = cycle;
        }
        Ok(())
    }

    pub fn record_event(&mut self, event: &str) -> Result<(), PowerStateError> {
        if self.schema_version != TYPED_STATE_SCHEMA_VERSION {
            return Err(PowerStateError::SchemaVersion);
        }
        let mut next = self.clone();
        next.record_event_inner(event)?;
        *self = next;
        Ok(())
    }

    /// Apply the narrow scalar-macro power projection used by the reference
    /// event loop.  The macro is accepted only for a closed issue-only region;
    /// this method records its declared clock/event contribution before the
    /// continuation event is observed.  It is not a hardware energy model and
    /// does not manufacture completion tokens.
    pub fn record_scalar_macro_completion(
        &mut self,
        start_cycle: u64,
        end_cycle: u64,
    ) -> Result<(), PowerStateError> {
        if end_cycle < start_cycle {
            return Err(PowerStateError::TimeTravel);
        }
        let mut next = self.clone();
        next.advance_to(start_cycle)?;
        for domain in &mut next.domains {
            if end_cycle < domain.last_cycle {
                return Err(PowerStateError::TimeTravel);
            }
            domain.status = PowerDomainStatus::Clocked;
            domain.queue_events = domain
                .queue_events
                .checked_add(1)
                .ok_or(PowerStateError::CounterOverflow)?;
            domain.sm_wake_events = domain
                .sm_wake_events
                .checked_add(1)
                .ok_or(PowerStateError::CounterOverflow)?;
            domain.last_cycle = end_cycle;
        }
        *self = next;
        Ok(())
    }

    fn record_event_inner(&mut self, event: &str) -> Result<(), PowerStateError> {
        let domain = self.domains.first_mut().ok_or(PowerStateError::NoDomains)?;
        domain.status = PowerDomainStatus::Clocked;
        domain.queue_events = domain
            .queue_events
            .checked_add(1)
            .ok_or(PowerStateError::CounterOverflow)?;
        if event.contains("l2")
            || event.contains("memory")
            || event == "mem_response"
            || event == "atomic_completion"
        {
            domain.memory_events = domain
                .memory_events
                .checked_add(1)
                .ok_or(PowerStateError::CounterOverflow)?;
        }
        if event == "sm_wake" {
            domain.sm_wake_events = domain
                .sm_wake_events
                .checked_add(1)
                .ok_or(PowerStateError::CounterOverflow)?;
        }
        if event == "memory_response"
            || event == "l2_fill"
            || event == "atomic_completion"
            || event == "memory_write_ack"
        {
            domain.completion_events = domain
                .completion_events
                .checked_add(1)
                .ok_or(PowerStateError::CounterOverflow)?;
        }
        Ok(())
    }

    fn emit_samples_through(&mut self, cycle: u64) -> Result<(), PowerStateError> {
        let Some(period) = self.sample_period else {
            return Ok(());
        };
        if period == 0 {
            return Err(PowerStateError::InvalidSamplePeriod);
        }
        let Some(mut next) = self.next_sample else {
            return Ok(());
        };
        while next <= cycle {
            if self.samples.len() >= 100_000 {
                return Err(PowerStateError::SampleLimit);
            }
            self.integrate_to(next)?;
            let activity_total = self.domains.iter().try_fold(0_u64, |sum, domain| {
                sum.checked_add(domain.clocked_cycles)
                    .ok_or(PowerStateError::CounterOverflow)
            })?;
            self.samples.push(PowerSampleState {
                cycle: next,
                activity_total,
                domain_clocked_cycles: self
                    .domains
                    .iter()
                    .map(|domain| domain.clocked_cycles)
                    .collect(),
                domain_gated_cycles: self
                    .domains
                    .iter()
                    .map(|domain| domain.gated_cycles)
                    .collect(),
            });
            next = next
                .checked_add(period)
                .ok_or(PowerStateError::TimeOverflow)?;
        }
        self.next_sample = Some(next);
        Ok(())
    }

    pub fn validate(&self) -> Result<(), PowerStateError> {
        if self.schema_version != TYPED_STATE_SCHEMA_VERSION {
            return Err(PowerStateError::SchemaVersion);
        }
        if self.domains.is_empty() {
            return Err(PowerStateError::NoDomains);
        }
        if self.clock.clock_domain.is_empty() {
            return Err(PowerStateError::InvalidClockDomain);
        }
        if self.clock.fixed_frequency_hz == Some(0) {
            return Err(PowerStateError::InvalidFrequency);
        }
        if self.clock.dvfs_modeled && self.status != PowerModelStatus::AnalyticCoefficients {
            return Err(PowerStateError::UnknownDvfs);
        }
        if self.status != PowerModelStatus::Unknown && !self.unknown_state.is_empty() {
            return Err(PowerStateError::UnknownState);
        }
        let mut domain_names = BTreeSet::new();
        let mut domain_time = None;
        for domain in &self.domains {
            if domain.domain.is_empty() || !domain_names.insert(&domain.domain) {
                return Err(PowerStateError::DomainIdentity);
            }
            let integrated = domain
                .clocked_cycles
                .checked_add(domain.gated_cycles)
                .ok_or(PowerStateError::CounterOverflow)?;
            if integrated > domain.last_cycle {
                return Err(PowerStateError::IntegralAheadOfClock);
            }
            if let Some(previous) = domain_time
                && previous != domain.last_cycle
            {
                return Err(PowerStateError::DomainTimeMismatch);
            }
            domain_time = Some(domain.last_cycle);
        }
        match (self.sample_period, self.next_sample) {
            (Some(0), _) => return Err(PowerStateError::InvalidSamplePeriod),
            (None, Some(_)) => return Err(PowerStateError::SampleSchedule),
            (Some(_), None) => return Err(PowerStateError::SampleSchedule),
            _ => {}
        }
        if self.sample_period.is_none() && !self.samples.is_empty() {
            return Err(PowerStateError::SampleSchedule);
        }
        let mut previous_sample: Option<&PowerSampleState> = None;
        for sample in &self.samples {
            if sample.domain_clocked_cycles.len() != self.domains.len()
                || sample.domain_gated_cycles.len() != self.domains.len()
            {
                return Err(PowerStateError::SampleDomainCount);
            }
            let calculated_activity =
                sample
                    .domain_clocked_cycles
                    .iter()
                    .try_fold(0_u64, |sum, value| {
                        sum.checked_add(*value)
                            .ok_or(PowerStateError::CounterOverflow)
                    })?;
            if calculated_activity != sample.activity_total {
                return Err(PowerStateError::SampleActivityMismatch);
            }
            if sample
                .domain_clocked_cycles
                .iter()
                .zip(&self.domains)
                .any(|(sampled, domain)| *sampled > domain.clocked_cycles)
                || sample
                    .domain_gated_cycles
                    .iter()
                    .zip(&self.domains)
                    .any(|(sampled, domain)| *sampled > domain.gated_cycles)
            {
                return Err(PowerStateError::SampleAheadOfDomain);
            }
            for (&clocked, &gated) in sample
                .domain_clocked_cycles
                .iter()
                .zip(&sample.domain_gated_cycles)
            {
                if clocked
                    .checked_add(gated)
                    .ok_or(PowerStateError::CounterOverflow)?
                    > sample.cycle
                {
                    return Err(PowerStateError::SampleIntegralAheadOfCycle);
                }
            }
            if let Some(previous) = previous_sample
                && (sample.cycle <= previous.cycle
                    || sample.activity_total < previous.activity_total
                    || sample
                        .domain_clocked_cycles
                        .iter()
                        .zip(&previous.domain_clocked_cycles)
                        .any(|(current, previous)| current < previous)
                    || sample
                        .domain_gated_cycles
                        .iter()
                        .zip(&previous.domain_gated_cycles)
                        .any(|(current, previous)| current < previous))
            {
                return Err(PowerStateError::UnsortedSamples);
            }
            previous_sample = Some(sample);
        }
        if self.samples.iter().any(|sample| {
            self.domains
                .iter()
                .any(|domain| sample.cycle > domain.last_cycle)
        }) {
            return Err(PowerStateError::SampleAheadOfDomainTime);
        }
        if let Some(period) = self.sample_period {
            // `activity_only` starts the schedule at cycle `period` and
            // advances it by exactly one period for every emitted sample.
            // Keeping the complete cadence is part of the resumable state:
            // accepting a later `next_sample` would silently skip an
            // integration interval, while an off-phase sample would shift
            // every future boundary. Run this after per-sample integrity
            // checks so malformed sample payloads retain their precise error.
            let mut expected = period;
            for sample in &self.samples {
                if sample.cycle != expected {
                    return Err(PowerStateError::SampleSchedule);
                }
                expected = expected
                    .checked_add(period)
                    .ok_or(PowerStateError::TimeOverflow)?;
            }
            if self.next_sample != Some(expected) {
                return Err(PowerStateError::SampleSchedule);
            }
        }
        if let (Some(last_sample), Some(next_sample)) = (previous_sample, self.next_sample)
            && next_sample <= last_sample.cycle
        {
            return Err(PowerStateError::SampleSchedule);
        }
        if let (Some(next_sample), Some(last_cycle)) = (self.next_sample, domain_time)
            && next_sample <= last_cycle
        {
            return Err(PowerStateError::SampleSchedule);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PowerStateError {
    #[error("typed power schema is unsupported")]
    SchemaVersion,
    #[error("power state moved backwards in time")]
    TimeTravel,
    #[error("power activity counter overflowed")]
    CounterOverflow,
    #[error("power sample period must be non-zero")]
    InvalidSamplePeriod,
    #[error("power sample time overflowed")]
    TimeOverflow,
    #[error("power sample output exceeds the bounded 100000-sample limit")]
    SampleLimit,
    #[error("power state has no domains")]
    NoDomains,
    #[error("power clock domain name must be non-empty")]
    InvalidClockDomain,
    #[error("power domain names must be non-empty and unique")]
    DomainIdentity,
    #[error("fixed power frequency must be non-zero when provided")]
    InvalidFrequency,
    #[error("power sample schedule is inconsistent with its period")]
    SampleSchedule,
    #[error("power samples are not strictly ordered or activity decreased")]
    UnsortedSamples,
    #[error("power sample domain vectors do not match the domain count")]
    SampleDomainCount,
    #[error("power sample activity total disagrees with its domain integrals")]
    SampleActivityMismatch,
    #[error("power sample is ahead of the corresponding domain integral")]
    SampleAheadOfDomain,
    #[error("power domain integrals are ahead of their elapsed cycle")]
    IntegralAheadOfClock,
    #[error("power domains do not share one boundary timestamp")]
    DomainTimeMismatch,
    #[error("power sample integrals are ahead of their sample cycle")]
    SampleIntegralAheadOfCycle,
    #[error("power sample time is ahead of the corresponding domain clock")]
    SampleAheadOfDomainTime,
    #[error("DVFS state is not represented by the selected power model")]
    UnknownDvfs,
    #[error("power state contains unknown fields")]
    UnknownState,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Completeness {
    Complete,
    Incomplete { unknown: Vec<String> },
    Refused { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TypedContractKind {
    BoundaryDiagnosticV2,
    CycleExactMemoryV2,
    CycleExactPowerV2,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedObservationContract {
    pub schema_version: u32,
    pub kind: TypedContractKind,
    pub address_domain: AddressDomain,
    pub power: PowerModelStatus,
}

impl TypedObservationContract {
    pub fn diagnostic() -> Self {
        Self {
            schema_version: TYPED_STATE_SCHEMA_VERSION,
            kind: TypedContractKind::BoundaryDiagnosticV2,
            address_domain: AddressDomain::Unknown,
            power: PowerModelStatus::Omitted,
        }
    }

    pub fn cycle_exact_memory() -> Self {
        Self {
            schema_version: TYPED_STATE_SCHEMA_VERSION,
            kind: TypedContractKind::CycleExactMemoryV2,
            address_domain: AddressDomain::VirtualAbsolute,
            power: PowerModelStatus::Omitted,
        }
    }

    pub fn cycle_exact_power() -> Self {
        Self {
            schema_version: TYPED_STATE_SCHEMA_VERSION,
            kind: TypedContractKind::CycleExactPowerV2,
            address_domain: AddressDomain::VirtualAbsolute,
            power: PowerModelStatus::ActivityOnly,
        }
    }

    pub fn validate(
        &self,
        l2: &L2BoundaryState,
        power: &PowerBoundaryState,
        cores: &[TypedCoreBoundaryState],
    ) -> Completeness {
        if self.schema_version != TYPED_STATE_SCHEMA_VERSION {
            return Completeness::Refused {
                reason: "typed_contract_schema_version".to_owned(),
            };
        }
        match self.kind {
            TypedContractKind::CycleExactMemoryV2 if self.power != PowerModelStatus::Omitted => {
                return Completeness::Refused {
                    reason: "memory_contract_power_mismatch".to_owned(),
                };
            }
            TypedContractKind::CycleExactPowerV2
                if self.power != PowerModelStatus::ActivityOnly =>
            {
                return Completeness::Refused {
                    reason: "power_contract_status_mismatch".to_owned(),
                };
            }
            _ => {}
        }
        if let Err(error) = l2.validate() {
            return Completeness::Refused {
                reason: error.to_string(),
            };
        }
        if let Err(error) = power.validate() {
            return Completeness::Refused {
                reason: error.to_string(),
            };
        }
        let mut core_ids = BTreeSet::new();
        let mut previous_core_id = None;
        for core in cores {
            if let Err(error) = core.validate() {
                return Completeness::Refused {
                    reason: error.to_string(),
                };
            }
            if !core_ids.insert(core.sm_id) {
                return Completeness::Refused {
                    reason: "duplicate_typed_core_sm_identity".to_owned(),
                };
            }
            if previous_core_id.is_some_and(|previous| core.sm_id <= previous) {
                return Completeness::Refused {
                    reason: "typed_core_identity_order".to_owned(),
                };
            }
            previous_core_id = Some(core.sm_id);
        }
        // The current typed core, queue, and causal-frontier contracts model
        // the four-SMSP SM86 topology. Keep diagnostic projections
        // architecture-neutral, but never let an SM120 or unknown L2 card
        // identity borrow this exactness claim until its own typed schema and
        // transition rules exist.
        if !matches!(self.kind, TypedContractKind::BoundaryDiagnosticV2)
            && l2.architecture != "sm_86"
        {
            return Completeness::Refused {
                reason: "sm86_exact_contract_requires_sm_86".to_owned(),
            };
        }
        if !l2.ownership_complete {
            return Completeness::Incomplete {
                unknown: l2.unknown_state.clone(),
            };
        }
        if !matches!(self.kind, TypedContractKind::BoundaryDiagnosticV2) {
            let mut unknown = cores
                .iter()
                .flat_map(|core| core.unprojected_timing_state.iter().cloned())
                .collect::<BTreeSet<_>>();
            if cores.iter().any(|core| {
                core.warp_states
                    .iter()
                    .any(|warp| warp.barrier_pending.iter().any(|&pending| pending != 0))
            }) {
                // The response-owner projection has no dependency-barrier
                // destination. Do not trust a producer-supplied marker list
                // to hide this live transition state.
                unknown.insert("dependency_barrier_ownership".to_owned());
            }
            if !unknown.is_empty() {
                return Completeness::Incomplete {
                    unknown: unknown.into_iter().collect(),
                };
            }
        }
        // The shared projection and each SM projection duplicate live
        // response ownership at different scopes. A complete boundary must
        // carry the same canonical owner records in both places; otherwise a
        // response can be present in L2 while absent from the core (or the
        // reverse), and the next transition cannot be reconstructed exactly.
        let mut shared_owners = l2.request_owners.clone();
        let mut core_owners = cores
            .iter()
            .flat_map(|core| core.response_owners.iter().cloned())
            .collect::<Vec<_>>();
        let owner_order = |left: &L2RequestOwner, right: &L2RequestOwner| {
            (
                &left.request,
                &left.line,
                left.sector_mask,
                left.allocation_generation,
                &left.state,
                left.due,
            )
                .cmp(&(
                    &right.request,
                    &right.line,
                    right.sector_mask,
                    right.allocation_generation,
                    &right.state,
                    right.due,
                ))
        };
        shared_owners.sort_by(owner_order);
        core_owners.sort_by(owner_order);
        if shared_owners != core_owners {
            return Completeness::Incomplete {
                unknown: vec!["l2_core_response_owner_join".to_owned()],
            };
        }
        // The diagnostic contract may intentionally carry only a shared-L2
        // projection, but cycle-exact contracts promise per-SM issue,
        // frontier, and resource state. An empty roster therefore cannot be
        // certified as an exact boundary even when the shared cache and
        // response-owner join are otherwise complete.
        if !matches!(self.kind, TypedContractKind::BoundaryDiagnosticV2) && cores.is_empty() {
            return Completeness::Incomplete {
                unknown: vec!["core_state_missing".to_owned()],
            };
        }
        for state in cores
            .iter()
            .flat_map(|core| core.async_copy.iter().flatten())
        {
            if let Err(error) = state.validate() {
                return Completeness::Refused {
                    reason: error.to_string(),
                };
            }
        }
        if cores.iter().any(|core| {
            core.async_copy.iter().flatten().any(|state| {
                !state.bypass_observations_complete
                    || state.bypass_observation_count
                        != u64::try_from(state.bypass_observations.len()).unwrap_or(u64::MAX)
            })
        }) {
            return Completeness::Incomplete {
                unknown: vec!["async_bypass_observation_state".to_owned()],
            };
        }
        if cores
            .iter()
            .any(|core| core.l1.as_ref().is_some_and(|l1| l1.write_observer_present))
        {
            return Completeness::Incomplete {
                unknown: vec!["l1_write_lookup_observer_state".to_owned()],
            };
        }
        if l2.request_owners.iter().any(|owner| {
            matches!(
                owner.request.response_kind,
                TypedResponseKind::Warp | TypedResponseKind::AsyncCopy
            ) && matches!(owner.line, CanonicalLineKey::Unknown { .. })
        }) {
            return Completeness::Incomplete {
                unknown: vec!["opaque_response_allocation_identity".to_owned()],
            };
        }
        if l2
            .request_owners
            .iter()
            .any(|owner| matches!(owner.state, L2RequestState::Outstanding) && owner.due.is_none())
        {
            return Completeness::Incomplete {
                unknown: vec!["request_due_time_unknown".to_owned()],
            };
        }
        if self.kind != TypedContractKind::BoundaryDiagnosticV2
            && l2.address_domain != self.address_domain
        {
            return Completeness::Refused {
                reason: "address_domain_mismatch".to_owned(),
            };
        }
        if self.power != PowerModelStatus::Omitted {
            if power.status != self.power {
                return Completeness::Incomplete {
                    unknown: vec!["power_model_status".to_owned()],
                };
            }
            if let Err(error) = power.validate() {
                return Completeness::Refused {
                    reason: error.to_string(),
                };
            }
        }
        Completeness::Complete
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TypedEventPayload {
    SmWake,
    /// An explicit value-bearing L1 admission supplied by an external event
    /// producer. The typed boundary retains only bounded cardinalities; the
    /// payload bytes and allocation identities remain in the native event
    /// input and are never treated as hardware completion evidence.
    L1PayloadAdmission {
        payload_records: u64,
        witness_lines: u64,
        identity_present: bool,
    },
    L2Access {
        request: TypedMemoryRequest,
    },
    L2AccessBatch {
        requests: Vec<TypedMemoryRequest>,
    },
    L2Fill {
        fill: TypedFillIdentity,
    },
    AtomicCompletion,
    WritebackCompletion {
        token: u64,
        line_id: i64,
        dirty_mask: u8,
        service_done_at: u64,
    },
    CoherenceCompletion {
        token: u64,
        line_id: i64,
        sector_mask: u8,
        service_sectors: u32,
        service_done_at: u64,
    },
    L1VisibilityCompletion {
        token: u64,
        line_id: i64,
        sector_mask: u8,
        service_sectors: u32,
        service_done_at: u64,
    },
    L1VisibilityAccess {
        sequence: u64,
        requester: u8,
        line_id: i64,
        sector_mask: u8,
        is_write: bool,
        l1_resident_after: bool,
        fill_will_reside: bool,
        fill_generation: Option<u32>,
        hit: bool,
        issue_cycle: u64,
    },
    L1VisibilityFence {
        sequence: u64,
        requester: u8,
        pc: u32,
        issue_cycle: u64,
        scope: String,
        order: String,
    },
    L1FillVisible {
        sequence: u64,
        requester: u8,
        line_id: i64,
        sector_mask: u8,
        fill_generation: u32,
        visible_cycle: u64,
    },
    L1InvalidateAll {
        sequence: u64,
        requester: u8,
        issue_cycle: u64,
    },
    MemoryWriteAck {
        warp: u32,
        instruction: u32,
        subcore: u8,
        request_index: u16,
        request_count: u16,
        operation: TypedMemoryOperation,
        issue_at: u64,
        accepted_at: u64,
        service_done_at: u64,
        fence_visible_at: u64,
    },
    MemoryResponse {
        reply: Option<TypedResponseIdentity>,
        fill: Option<TypedFillIdentity>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedResponseIdentity {
    pub warp: u64,
    pub instruction: u64,
    pub sector_index: u64,
    pub response_kind: TypedResponseKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedFillIdentity {
    pub line_id: i64,
    pub generation: u64,
    pub sector_mask: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedAdmissionWitness {
    pub warp: u32,
    pub instruction: u32,
    pub subcore: u8,
    pub request_index: u16,
    pub request_count: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedMemoryRequest {
    pub issue_time: u64,
    pub sector_id: i64,
    pub origin: TypedMemoryOrigin,
    pub sector_mask: u8,
    pub local_sector_mask: u8,
    pub source: TypedL2Source,
    pub operation: TypedMemoryOperation,
    pub constant_load: bool,
    pub constant_deep: bool,
    pub reply: Option<TypedResponseIdentity>,
    pub admission_witness: Option<TypedAdmissionWitness>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedPendingEvent {
    pub time: u64,
    pub sequence: u64,
    /// Current engine phase. Phase zero is the engine's single serialized
    /// class; future multi-phase engines must expose their ordering here.
    pub phase: u8,
    /// Actual engine reservation, including the anchor sequence.
    pub sequence_span: u64,
    pub target: u32,
    pub payload: TypedEventPayload,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueueBoundaryState {
    pub depth: u64,
    pub next_event_time: Option<u64>,
    /// Next unused insertion position; null explicitly denotes exhaustion.
    /// Absence is a malformed snapshot, not an exhausted queue.
    #[serde(deserialize_with = "Option::<u64>::deserialize")]
    pub next_sequence: Option<u64>,
    pub pending: Vec<TypedPendingEvent>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum EventSequenceError {
    #[error("event at sequence {sequence} reserves zero positions")]
    EmptySpan { sequence: u64 },
    #[error("event at sequence {sequence} overflows its {span}-position reservation")]
    SpanOverflow { sequence: u64, span: u64 },
    #[error(
        "event at sequence {sequence} reserves {span} positions for {logical_events} logical events"
    )]
    PayloadSpanMismatch {
        sequence: u64,
        span: u64,
        logical_events: usize,
    },
}

fn sequence_end(sequence: u64, span: u64) -> Result<u64, EventSequenceError> {
    let offset = span
        .checked_sub(1)
        .ok_or(EventSequenceError::EmptySpan { sequence })?;
    sequence
        .checked_add(offset)
        .ok_or(EventSequenceError::SpanOverflow { sequence, span })
}

impl TypedEventPayload {
    fn validate_event_time(&self, event_time: u64) -> Result<(), BoundaryStateError> {
        let check = |payload: &'static str, field: &'static str, found: u64| {
            if found != event_time {
                return Err(BoundaryStateError::PayloadTimeMismatch {
                    payload,
                    field,
                    expected: event_time,
                    found,
                });
            }
            Ok(())
        };
        match self {
            Self::L2Access { request } => {
                request.validate_contract()?;
                check("l2_access", "issue_time", request.issue_time)
            }
            Self::L2AccessBatch { requests } => {
                validate_batch_witnesses(requests)?;
                requests.iter().try_for_each(|request| {
                    request.validate_contract()?;
                    check("l2_access_batch", "issue_time", request.issue_time)
                })
            }
            Self::MemoryWriteAck {
                operation,
                subcore,
                request_index,
                request_count,
                issue_at,
                accepted_at,
                service_done_at,
                fence_visible_at,
                ..
            } => {
                if !matches!(
                    operation,
                    TypedMemoryOperation::Store | TypedMemoryOperation::Reduction
                ) {
                    return Err(BoundaryStateError::MemoryWriteAckOperation {
                        operation: *operation,
                    });
                }
                if *subcore >= 4 || *request_count == 0 || *request_index >= *request_count {
                    return Err(BoundaryStateError::MemoryWriteAckWitness {
                        subcore: *subcore,
                        request_index: *request_index,
                        request_count: *request_count,
                    });
                }
                if accepted_at < issue_at {
                    return Err(BoundaryStateError::PayloadTimestampOrder {
                        payload: "memory_write_ack",
                        earlier_field: "issue_at",
                        earlier: *issue_at,
                        later_field: "accepted_at",
                        later: *accepted_at,
                    });
                }
                if service_done_at < accepted_at {
                    return Err(BoundaryStateError::PayloadTimestampOrder {
                        payload: "memory_write_ack",
                        earlier_field: "accepted_at",
                        earlier: *accepted_at,
                        later_field: "service_done_at",
                        later: *service_done_at,
                    });
                }
                if fence_visible_at < service_done_at {
                    return Err(BoundaryStateError::PayloadTimestampOrder {
                        payload: "memory_write_ack",
                        earlier_field: "service_done_at",
                        earlier: *service_done_at,
                        later_field: "fence_visible_at",
                        later: *fence_visible_at,
                    });
                }
                check("memory_write_ack", "fence_visible_at", *fence_visible_at)
            }
            Self::L2Fill { fill } => validate_fill_identity(fill),
            Self::MemoryResponse { reply, fill } => {
                let Some(reply) = reply else {
                    return Err(BoundaryStateError::MemoryResponseReplyMissing);
                };
                validate_response_identity(reply)?;
                if let Some(fill) = fill {
                    validate_fill_identity(fill)?;
                    if !matches!(
                        reply.response_kind,
                        TypedResponseKind::L1Fill
                            | TypedResponseKind::L1SectorFill
                            | TypedResponseKind::Warp
                            | TypedResponseKind::AsyncCopy
                    ) {
                        return Err(BoundaryStateError::MemoryResponseFillReplyMismatch {
                            response_kind: reply.response_kind,
                        });
                    }
                }
                Ok(())
            }
            Self::SmWake | Self::AtomicCompletion => Ok(()),
            Self::L1PayloadAdmission { .. } => Ok(()),
            Self::WritebackCompletion {
                service_done_at, ..
            } => check("writeback_completion", "service_done_at", *service_done_at),
            Self::CoherenceCompletion {
                service_done_at, ..
            } => check("coherence_completion", "service_done_at", *service_done_at),
            Self::L1VisibilityCompletion {
                service_done_at, ..
            } => check(
                "l1_visibility_completion",
                "service_done_at",
                *service_done_at,
            ),
            Self::L1VisibilityAccess { issue_cycle, .. } => {
                check("l1_visibility_access", "issue_cycle", *issue_cycle)
            }
            Self::L1VisibilityFence { issue_cycle, .. } => {
                check("l1_visibility_fence", "issue_cycle", *issue_cycle)
            }
            Self::L1FillVisible { visible_cycle, .. } => {
                check("l1_fill_visible", "visible_cycle", *visible_cycle)
            }
            Self::L1InvalidateAll { issue_cycle, .. } => {
                check("l1_invalidate_all", "issue_cycle", *issue_cycle)
            }
        }
    }

    fn sequence_end(&self, sequence: u64, span: u64) -> Result<u64, EventSequenceError> {
        let end = sequence_end(sequence, span)?;
        let logical_events = match self {
            Self::L2AccessBatch { requests } => requests.len(),
            _ => 1,
        };
        if u64::try_from(logical_events) != Ok(span) {
            return Err(EventSequenceError::PayloadSpanMismatch {
                sequence,
                span,
                logical_events,
            });
        }
        Ok(end)
    }
}

fn validate_fill_identity(fill: &TypedFillIdentity) -> Result<(), BoundaryStateError> {
    if fill.sector_mask == 0 || fill.sector_mask & !0x0f != 0 {
        return Err(BoundaryStateError::FillIdentityMask {
            sector_mask: fill.sector_mask,
        });
    }
    Ok(())
}

fn validate_response_identity(response: &TypedResponseIdentity) -> Result<(), BoundaryStateError> {
    if matches!(response.response_kind, TypedResponseKind::None) {
        return Err(BoundaryStateError::ResponseIdentityKindNone);
    }
    Ok(())
}

impl TypedMemoryRequest {
    fn validate_contract(&self) -> Result<(), BoundaryStateError> {
        let own_sector_bit = 1_u8 << (self.sector_id & 3) as u8;
        if self.sector_mask == 0
            || self.sector_mask & !0x0f != 0
            || self.sector_mask & own_sector_bit == 0
            || self.local_sector_mask & !self.sector_mask != 0
            || (self.local_sector_mask & own_sector_bit != 0)
                != matches!(self.origin, TypedMemoryOrigin::Local)
        {
            return Err(BoundaryStateError::MemoryRequestMask {
                sector_id: self.sector_id,
                sector_mask: self.sector_mask,
                local_sector_mask: self.local_sector_mask,
            });
        }
        if (self.constant_load || self.constant_deep)
            && (!self.constant_load
                || !matches!(self.source, TypedL2Source::L1Fill)
                || !matches!(self.operation, TypedMemoryOperation::Read))
        {
            return Err(BoundaryStateError::MemoryRequestConstantContract {
                request_source: self.source,
                operation: self.operation,
                constant_load: self.constant_load,
                constant_deep: self.constant_deep,
            });
        }
        let reply = self.reply.as_ref().map(|reply| reply.response_kind);
        if matches!(reply, Some(TypedResponseKind::L1SectorFill))
            && self.sector_mask.count_ones() != 1
        {
            return Err(BoundaryStateError::MemoryRequestSectorReplyMismatch {
                response_kind: TypedResponseKind::L1SectorFill,
                sector_mask: self.sector_mask,
            });
        }
        let valid_contract = matches!(
            (self.source, self.operation, reply),
            (TypedL2Source::L1Fill, TypedMemoryOperation::Read, None)
                | (
                    TypedL2Source::L1Fill,
                    TypedMemoryOperation::Read,
                    Some(TypedResponseKind::L1Fill | TypedResponseKind::L1SectorFill),
                )
                | (
                    TypedL2Source::Bypass,
                    TypedMemoryOperation::Read,
                    None | Some(TypedResponseKind::Warp | TypedResponseKind::AsyncCopy),
                )
                | (
                    TypedL2Source::Untagged,
                    TypedMemoryOperation::Read,
                    None | Some(TypedResponseKind::Warp),
                )
                | (
                    TypedL2Source::Bypass,
                    TypedMemoryOperation::Atomic,
                    None | Some(TypedResponseKind::Warp),
                )
                | (
                    TypedL2Source::Store,
                    TypedMemoryOperation::Store | TypedMemoryOperation::Reduction,
                    None,
                )
        );
        if !valid_contract {
            return Err(BoundaryStateError::MemoryRequestContract {
                request_source: self.source,
                operation: self.operation,
                reply,
            });
        }
        if let Some(witness) = &self.admission_witness
            && (witness.subcore >= 4
                || witness.request_count == 0
                || witness.request_index >= witness.request_count
                || !matches!(
                    (self.source, self.origin, self.operation),
                    (
                        TypedL2Source::Store,
                        TypedMemoryOrigin::Global,
                        TypedMemoryOperation::Store | TypedMemoryOperation::Reduction
                    )
                ))
        {
            return Err(BoundaryStateError::MemoryRequestWitness {
                subcore: witness.subcore,
                request_index: witness.request_index,
                request_count: witness.request_count,
            });
        }
        Ok(())
    }
}

fn validate_batch_witnesses(requests: &[TypedMemoryRequest]) -> Result<(), BoundaryStateError> {
    // The native observer carries one pending instruction across events, but
    // closes it as soon as request_count is reached.  The scheduler can then
    // place the next instruction's index-zero request in the same timestamped
    // batch.  Keep the active group's expected index so we accept those legal
    // group transitions while still rejecting interleaving and skips.
    let mut current: Option<(TypedAdmissionWitness, TypedMemoryOperation, u64, u16)> = None;
    let mut interrupted_incomplete_group = false;
    for request in requests {
        let Some(witness) = request.admission_witness else {
            // A witness sequence describes every request emitted by one
            // instruction. If it interrupts a group whose advertised count
            // has not been observed, retain that fact: otherwise a forged
            // batch could hide a gap between indices and restart at an
            // arbitrary witness index.
            if let Some((previous_witness, _, _, expected)) = current
                && expected != previous_witness.request_count
            {
                interrupted_incomplete_group = true;
            }
            continue;
        };
        if interrupted_incomplete_group {
            return Err(BoundaryStateError::MemoryRequestWitnessGroupMismatch);
        }
        let Some((previous_witness, previous_operation, previous_issue_time, expected)) = current
        else {
            let next = witness.request_index.checked_add(1).ok_or(
                BoundaryStateError::MemoryRequestWitnessSequence {
                    expected: u16::MAX,
                    found: witness.request_index,
                },
            )?;
            current = Some((witness, request.operation, request.issue_time, next));
            continue;
        };

        let same_group = previous_witness.warp == witness.warp
            && previous_witness.instruction == witness.instruction
            && previous_witness.subcore == witness.subcore
            && previous_witness.request_count == witness.request_count
            && previous_operation == request.operation
            && previous_issue_time == request.issue_time;
        if same_group {
            if witness.request_index != expected {
                return Err(BoundaryStateError::MemoryRequestWitnessSequence {
                    expected,
                    found: witness.request_index,
                });
            }
            let next = expected.checked_add(1).ok_or(
                BoundaryStateError::MemoryRequestWitnessSequence {
                    expected: u16::MAX,
                    found: witness.request_index,
                },
            )?;
            current = Some((witness, request.operation, request.issue_time, next));
            continue;
        }

        if expected != previous_witness.request_count {
            return Err(BoundaryStateError::MemoryRequestWitnessGroupMismatch);
        }
        if witness.request_index != 0 {
            return Err(BoundaryStateError::MemoryRequestWitnessSequence {
                expected: 0,
                found: witness.request_index,
            });
        }
        current = Some((witness, request.operation, request.issue_time, 1));
    }
    Ok(())
}

impl QueueBoundaryState {
    pub fn validate(&self) -> Result<(), QueueStateError> {
        if self.depth != u64::try_from(self.pending.len()).unwrap_or(u64::MAX) {
            return Err(QueueStateError::DepthMismatch {
                depth: self.depth,
                pending: self.pending.len(),
            });
        }
        let mut previous = None;
        let mut reservations = BTreeMap::new();
        for event in &self.pending {
            let end = event
                .payload
                .sequence_end(event.sequence, event.sequence_span)?;
            if reservations.insert(event.sequence, end).is_some() {
                return Err(QueueStateError::DuplicateSequence {
                    sequence: event.sequence,
                });
            }
            self.validate_reservation_end(end)?;
            if let Some(previous) = previous
                && (event.time, event.phase, event.sequence) <= previous
            {
                return Err(QueueStateError::Unsorted);
            }
            previous = Some((event.time, event.phase, event.sequence));
        }
        // Timestamp order need not be insertion order. Reservations occupy
        // one shared sequence space across all timestamps; never expand a
        // span into individual positions to check for overlap.
        let mut previous_end = None;
        for (sequence, end) in reservations {
            if let Some(previous_end) = previous_end
                && sequence <= previous_end
            {
                return Err(QueueStateError::OverlappingSpans {
                    sequence,
                    previous_end,
                });
            }
            previous_end = Some(end);
        }
        if self.next_event_time != self.pending.first().map(|event| event.time) {
            return Err(QueueStateError::NextTimeMismatch);
        }
        Ok(())
    }

    fn validate_reservation_end(&self, end: u64) -> Result<(), QueueStateError> {
        if let Some(next_sequence) = self.next_sequence
            && end >= next_sequence
        {
            return Err(QueueStateError::UnreservedSequence { end, next_sequence });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum QueueStateError {
    #[error(transparent)]
    Sequence(#[from] EventSequenceError),
    #[error("typed queue depth {depth} differs from pending event count {pending}")]
    DepthMismatch { depth: u64, pending: usize },
    #[error("typed queue contains duplicate sequence {sequence}")]
    DuplicateSequence { sequence: u64 },
    #[error("typed queue events are not in (time,sequence) order")]
    Unsorted,
    #[error("typed queue next-event time disagrees with its pending head")]
    NextTimeMismatch,
    #[error("event sequence {sequence} overlaps a reservation ending at {previous_end}")]
    OverlappingSpans { sequence: u64, previous_end: u64 },
    #[error("reservation ending at {end} reaches the next unused sequence {next_sequence}")]
    UnreservedSequence { end: u64, next_sequence: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedEventIdentity {
    pub time: u64,
    pub sequence: u64,
    pub sequence_span: u64,
    pub phase: u8,
    pub target: u32,
    pub payload: TypedEventPayload,
}

impl TypedEventIdentity {
    pub fn sequence_end(&self) -> Result<u64, EventSequenceError> {
        self.payload.sequence_end(self.sequence, self.sequence_span)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedCoreBoundaryState {
    pub sm_id: u32,
    pub cycle: u64,
    pub issued: u64,
    pub resident_warps: u32,
    pub resident_shared_memory_bytes: u64,
    pub unfinished_warps: u64,
    pub next_block: u64,
    pub hold: [Option<u64>; 4],
    pub subcore_warps: Vec<Vec<i32>>,
    pub outstanding_responses: u64,
    pub barrier_pipe_ready: u64,
    pub tensor_pipe_ready: [u64; 4],
    pub ldgsts_destination_pipe_ready: u64,
    pub warp_states: Vec<WarpBoundaryState>,
    pub block_states: Vec<BlockBoundaryState>,
    pub l1: Option<L1BoundaryState>,
    pub front_end: Option<InstructionFrontEndBoundaryState>,
    pub memory_pipeline: Option<MemoryPipelineBoundaryState>,
    pub scalar_pipeline: Option<ScalarPipelineBoundaryState>,
    /// Native maximum CTA-barrier release time across all admitted blocks,
    /// including retired ones. Independent of the issue clock and the memory
    /// pipeline's terminal acceptance: release can lie ahead of `cycle`.
    /// Explicit null denotes no released generation; the field is required.
    #[serde(deserialize_with = "Option::<u64>::deserialize")]
    pub local_completion_time: Option<u64>,
    /// Timing state that affects future issue but is not yet serialized
    /// field-by-field. Exact contracts reify whenever this list is nonempty.
    pub unprojected_timing_state: Vec<String>,
    pub async_copy: Vec<Option<AsyncCopyBoundaryState>>,
    pub response_owners: Vec<L2RequestOwner>,
}

impl TypedCoreBoundaryState {
    pub fn validate(&self) -> Result<(), CoreStateError> {
        if self.sm_id == u32::MAX {
            return Err(CoreStateError::InvalidSmIdentity);
        }
        if self.subcore_warps.len() != 4 {
            return Err(CoreStateError::SubcoreCount {
                found: self.subcore_warps.len(),
            });
        }
        let mut previous_unprojected = None;
        for reason in &self.unprojected_timing_state {
            if reason.is_empty() || previous_unprojected.is_some_and(|previous| reason <= previous)
            {
                return Err(CoreStateError::UnprojectedTimingState);
            }
            previous_unprojected = Some(reason);
        }
        if self.next_block != u64::try_from(self.block_states.len()).unwrap_or(u64::MAX) {
            return Err(CoreStateError::NextBlockMismatch {
                next_block: self.next_block,
                block_states: self.block_states.len(),
            });
        }
        let mut roster = BTreeSet::new();
        for warps in &self.subcore_warps {
            for &warp_slot in warps {
                if warp_slot < 0 {
                    return Err(CoreStateError::NegativeWarpSlot { warp_slot });
                }
                if !roster.insert(warp_slot) {
                    return Err(CoreStateError::DuplicateResidentWarp { warp_slot });
                }
            }
        }
        for (subcore, warp_slot) in self
            .hold
            .iter()
            .enumerate()
            .filter_map(|(subcore, slot)| slot.as_ref().map(|&warp_slot| (subcore, warp_slot)))
        {
            let warp_slot = i32::try_from(warp_slot)
                .map_err(|_| CoreStateError::HoldWarpMissing { warp_slot })?;
            if !roster.contains(&warp_slot) {
                return Err(CoreStateError::HoldWarpMissing {
                    warp_slot: warp_slot as u64,
                });
            }
            if !self.subcore_warps[subcore].contains(&warp_slot) {
                return Err(CoreStateError::HoldWarpWrongSubcore { subcore, warp_slot });
            }
        }
        let mut known_warps = BTreeSet::new();
        for warp in &self.warp_states {
            if warp.warp_slot < 0 {
                return Err(CoreStateError::NegativeWarpSlot {
                    warp_slot: warp.warp_slot,
                });
            }
            validate_warp_instruction_buffer(warp, self.front_end.as_ref())?;
            if usize::try_from(warp.block)
                .ok()
                .is_none_or(|block| block >= self.block_states.len())
            {
                return Err(CoreStateError::UnknownWarpBlock { block: warp.block });
            }
            if !known_warps.insert((warp.block, warp.warp_slot)) {
                return Err(CoreStateError::DuplicateWarpState {
                    block: warp.block,
                    warp_slot: warp.warp_slot,
                });
            }
        }
        let issued_from_warps = self.warp_states.iter().try_fold(0_u64, |sum, warp| {
            sum.checked_add(warp.next_instruction)
                .ok_or(CoreStateError::CounterOverflow)
        })?;
        if issued_from_warps != self.issued {
            return Err(CoreStateError::IssuedCount {
                declared: self.issued,
                warps: issued_from_warps,
            });
        }
        let mut block_warps = BTreeSet::new();
        let mut active_block_slots = BTreeMap::new();
        let mut unfinished_by_block = BTreeMap::new();
        let mut unfinished_state_count = 0_u64;
        let mut resident_warp_reservation = 0_u64;
        let mut expected_first_warp = 0_u64;
        for warp in &self.warp_states {
            if !warp.done {
                unfinished_state_count = unfinished_state_count
                    .checked_add(1)
                    .ok_or(CoreStateError::CounterOverflow)?;
                let count = unfinished_by_block.entry(warp.block).or_insert(0_u64);
                *count = count
                    .checked_add(1)
                    .ok_or(CoreStateError::CounterOverflow)?;
                if !roster.contains(&warp.warp_slot) {
                    return Err(CoreStateError::UnfinishedWarpNotResident {
                        warp_slot: warp.warp_slot,
                    });
                }
            }
        }
        for (block_index, block) in self.block_states.iter().enumerate() {
            if block.unfinished != 0 {
                resident_warp_reservation = resident_warp_reservation
                    .checked_add(u64::from(block.warp_count))
                    .ok_or(CoreStateError::CounterOverflow)?;
            }
            if block.warp_count > u32::try_from(block.admitted_warps).unwrap_or(u32::MAX) {
                return Err(CoreStateError::BlockWarpCount {
                    block: block_index,
                    declared: block.warp_count,
                    admitted: block.admitted_warps,
                });
            }
            if block.unfinished > block.admitted_warps {
                return Err(CoreStateError::BlockUnfinishedCount { block: block_index });
            }
            if block.unfinished > u64::from(block.warp_count) {
                return Err(CoreStateError::BlockUnfinishedExceedsWarpCount {
                    block: block_index,
                    unfinished: block.unfinished,
                    warp_count: block.warp_count,
                });
            }
            if block.waiting > block.unfinished {
                return Err(CoreStateError::BlockWaitingCount { block: block_index });
            }
            if block.warp_slots.len() != usize::try_from(block.admitted_warps).unwrap_or(usize::MAX)
            {
                return Err(CoreStateError::BlockWarpRoster {
                    block: block_index,
                    declared: block.admitted_warps,
                    roster: block.warp_slots.len(),
                });
            }
            for &warp_slot in &block.warp_slots {
                if warp_slot < 0
                    || !known_warps
                        .contains(&(u64::try_from(block_index).unwrap_or(u64::MAX), warp_slot))
                {
                    return Err(CoreStateError::BlockWarpMissing { warp_slot });
                }
                if !block_warps.insert((block_index, warp_slot)) {
                    return Err(CoreStateError::DuplicateBlockWarp { warp_slot });
                }
            }
            if block.first_warp != expected_first_warp {
                return Err(CoreStateError::BlockFirstWarpOrder {
                    block: block_index,
                    expected: expected_first_warp,
                    found: block.first_warp,
                });
            }
            let range_end = block.first_warp.checked_add(block.admitted_warps).ok_or(
                CoreStateError::BlockWarpRange {
                    block: block_index,
                    first_warp: block.first_warp,
                    admitted_warps: block.admitted_warps,
                    warp_states: self.warp_states.len(),
                },
            )?;
            let warp_state_count = u64::try_from(self.warp_states.len()).unwrap_or(u64::MAX);
            if range_end > warp_state_count {
                return Err(CoreStateError::BlockWarpRange {
                    block: block_index,
                    first_warp: block.first_warp,
                    admitted_warps: block.admitted_warps,
                    warp_states: self.warp_states.len(),
                });
            }
            let range_start =
                usize::try_from(block.first_warp).map_err(|_| CoreStateError::BlockWarpRange {
                    block: block_index,
                    first_warp: block.first_warp,
                    admitted_warps: block.admitted_warps,
                    warp_states: self.warp_states.len(),
                })?;
            let range_end =
                usize::try_from(range_end).map_err(|_| CoreStateError::BlockWarpRange {
                    block: block_index,
                    first_warp: block.first_warp,
                    admitted_warps: block.admitted_warps,
                    warp_states: self.warp_states.len(),
                })?;
            let declared_slots = block.warp_slots.iter().copied().collect::<BTreeSet<_>>();
            let range_slots = self.warp_states[range_start..range_end]
                .iter()
                .map(|warp| warp.warp_slot)
                .collect::<BTreeSet<_>>();
            if declared_slots != range_slots {
                return Err(CoreStateError::BlockWarpRangeRoster { block: block_index });
            }
            for warp in &self.warp_states[range_start..range_end] {
                if warp.block != u64::try_from(block_index).unwrap_or(u64::MAX) {
                    return Err(CoreStateError::BlockWarpRangeIdentity {
                        block: block_index,
                        found: warp.block,
                    });
                }
            }
            validate_block_barrier_state(
                block_index,
                block,
                &self.warp_states[range_start..range_end],
                self.cycle,
            )?;
            expected_first_warp = range_end as u64;
            if block.unfinished != 0 {
                for &warp_slot in &block.warp_slots {
                    if let Some(&previous_block) = active_block_slots.get(&warp_slot) {
                        return Err(CoreStateError::ActiveBlockWarpSlotCollision {
                            warp_slot,
                            previous_block,
                            block: block_index,
                        });
                    }
                    active_block_slots.insert(warp_slot, block_index);
                }
            }
            let unfinished_states = unfinished_by_block
                .get(&u64::try_from(block_index).unwrap_or(u64::MAX))
                .copied()
                .unwrap_or(0);
            if unfinished_states != block.unfinished {
                return Err(CoreStateError::BlockWarpStateCount {
                    block: block_index,
                    declared: block.unfinished,
                    states: unfinished_states,
                });
            }
        }
        for (subcore, warps) in self.subcore_warps.iter().enumerate() {
            for &warp_slot in warps {
                let expected_subcore = usize::try_from(warp_slot.rem_euclid(4))
                    .expect("SM86 subcore modulo is within usize");
                if expected_subcore != subcore {
                    return Err(CoreStateError::ResidentWarpWrongSubcore {
                        warp_slot,
                        subcore,
                        expected_subcore,
                    });
                }
            }
        }
        let mut live_arrivals = BTreeSet::new();
        for warp in self.warp_states.iter().filter(|warp| !warp.done) {
            if !live_arrivals.insert(warp.arrival) {
                return Err(CoreStateError::DuplicateResidentWarpArrival {
                    arrival: warp.arrival,
                });
            }
        }
        let warp_state_count = u64::try_from(self.warp_states.len()).unwrap_or(u64::MAX);
        if expected_first_warp != warp_state_count {
            return Err(CoreStateError::WarpArenaTail {
                expected: expected_first_warp,
                found: warp_state_count,
            });
        }
        if u64::from(self.resident_warps) != resident_warp_reservation {
            return Err(CoreStateError::ResidentWarpReservationCount {
                declared: self.resident_warps,
                reservation: resident_warp_reservation,
            });
        }
        let unfinished_sum = self.block_states.iter().try_fold(0_u64, |sum, block| {
            sum.checked_add(block.unfinished)
                .ok_or(CoreStateError::CounterOverflow)
        })?;
        if unfinished_sum != self.unfinished_warps {
            return Err(CoreStateError::UnfinishedWarpCount {
                declared: self.unfinished_warps,
                blocks: unfinished_sum,
            });
        }
        if unfinished_state_count != self.unfinished_warps {
            return Err(CoreStateError::WarpStateCount {
                declared: self.unfinished_warps,
                states: unfinished_state_count,
            });
        }
        let shared_memory_sum = self
            .block_states
            .iter()
            .filter(|block| block.unfinished != 0)
            .try_fold(0_u64, |sum, block| {
                sum.checked_add(block.shared_memory_bytes)
                    .ok_or(CoreStateError::CounterOverflow)
            })?;
        if shared_memory_sum != self.resident_shared_memory_bytes {
            return Err(CoreStateError::SharedMemoryCount {
                declared: self.resident_shared_memory_bytes,
                blocks: shared_memory_sum,
            });
        }
        for &warp_slot in &roster {
            if !known_warps.iter().any(|(_, slot)| *slot == warp_slot) {
                return Err(CoreStateError::ResidentWarpMissing { warp_slot });
            }
        }
        if u64::try_from(roster.len()).unwrap_or(u64::MAX) != unfinished_state_count {
            return Err(CoreStateError::CurrentRosterCount {
                roster: roster.len(),
                unfinished_states: unfinished_state_count,
            });
        }
        if self.outstanding_responses
            != u64::try_from(self.response_owners.len()).unwrap_or(u64::MAX)
        {
            return Err(CoreStateError::ResponseCount {
                declared: self.outstanding_responses,
                owners: self.response_owners.len(),
            });
        }
        let mut previous = None;
        let mut seen = BTreeSet::new();
        for owner in &self.response_owners {
            if owner.request.sm_id != self.sm_id {
                return Err(CoreStateError::ResponseOwnerSm {
                    owner: owner.request.sm_id,
                });
            }
            let response_warp = usize::try_from(owner.request.response_warp).map_err(|_| {
                CoreStateError::ResponseOwnerWarpOutOfRange {
                    response_warp: owner.request.response_warp,
                    warp_states: self.warp_states.len(),
                }
            })?;
            let referenced_warp = self.warp_states.get(response_warp).ok_or(
                CoreStateError::ResponseOwnerWarpOutOfRange {
                    response_warp: owner.request.response_warp,
                    warp_states: self.warp_states.len(),
                },
            )?;
            if referenced_warp.warp_slot != owner.request.warp_slot {
                return Err(CoreStateError::ResponseOwnerWarpSlotMismatch {
                    response_warp: owner.request.response_warp,
                    owner_warp_slot: owner.request.warp_slot,
                    state_warp_slot: referenced_warp.warp_slot,
                });
            }
            if owner.request.instruction >= referenced_warp.next_instruction {
                return Err(CoreStateError::ResponseOwnerInstructionAfterCursor {
                    response_warp: owner.request.response_warp,
                    instruction: owner.request.instruction,
                    next_instruction: referenced_warp.next_instruction,
                });
            }
            if !matches!(owner.state, L2RequestState::Outstanding) {
                return Err(CoreStateError::ResponseOwnerState);
            }
            if owner.request.warp_slot < 0 {
                return Err(CoreStateError::NegativeResponseOwnerWarpSlot {
                    warp_slot: owner.request.warp_slot,
                });
            }
            let key = (
                owner.request.sm_id,
                owner.request.response_warp,
                owner.request.warp_slot,
                owner.request.instruction,
                owner.request.sector_index,
                owner.request.response_kind,
            );
            if !seen.insert(key) {
                return Err(CoreStateError::DuplicateResponseOwner);
            }
            if let Some(previous) = previous
                && key <= previous
            {
                return Err(CoreStateError::UnsortedResponseOwners);
            }
            previous = Some(key);
        }
        if let Some(l1) = &self.l1 {
            l1.validate_at_cycle(self.cycle)?
        }
        if let Some(memory) = &self.memory_pipeline {
            validate_memory_pipeline(memory, self.cycle)?;
        }
        if let Some(scalar) = &self.scalar_pipeline {
            validate_scalar_pipeline(scalar, self.cycle)?;
        }
        if let Some(front_end) = &self.front_end {
            front_end.validate()?
        }
        if !self.async_copy.is_empty() && self.async_copy.len() != self.warp_states.len() {
            return Err(CoreStateError::AsyncCopySlots);
        }
        for state in self.async_copy.iter().flatten() {
            state.validate_at_cycle(self.cycle)?
        }
        let has_released_barrier = self
            .block_states
            .iter()
            .any(|block| block.barrier_generation != 0);
        if has_released_barrier != self.local_completion_time.is_some()
            || self.local_completion_time == Some(0)
        {
            return Err(CoreStateError::LocalCompletionHistory);
        }
        Ok(())
    }
}

fn validate_warp_instruction_buffer(
    warp: &WarpBoundaryState,
    front_end: Option<&InstructionFrontEndBoundaryState>,
) -> Result<(), CoreStateError> {
    if warp.done && !warp.instruction_buffer.is_empty() {
        return Err(CoreStateError::DoneWarpInstructionBuffer {
            warp_slot: warp.warp_slot,
        });
    }
    let Some(front_end) = front_end else {
        if !warp.instruction_buffer.is_empty() {
            return Err(CoreStateError::InstructionBufferWithoutFrontEnd {
                warp_slot: warp.warp_slot,
            });
        }
        return Ok(());
    };
    let capacity = usize::try_from(front_end.instruction_buffer_entries).unwrap_or(usize::MAX);
    if warp.instruction_buffer.len() > capacity {
        return Err(CoreStateError::InstructionBufferCapacity {
            warp_slot: warp.warp_slot,
            found: warp.instruction_buffer.len(),
            capacity,
        });
    }
    if warp.front_end_fetch_cursor < warp.next_instruction {
        return Err(CoreStateError::InstructionCursorBeforeNext {
            warp_slot: warp.warp_slot,
            cursor: warp.front_end_fetch_cursor,
            next: warp.next_instruction,
        });
    }
    let mut previous: Option<u64> = None;
    for &(instruction, _) in &warp.instruction_buffer {
        if instruction < warp.next_instruction {
            return Err(CoreStateError::InstructionBufferBeforeNext {
                warp_slot: warp.warp_slot,
                next: warp.next_instruction,
                found: instruction,
            });
        }
        if instruction >= warp.front_end_fetch_cursor {
            return Err(CoreStateError::InstructionBufferBeyondCursor {
                warp_slot: warp.warp_slot,
                cursor: warp.front_end_fetch_cursor,
                found: instruction,
            });
        }
        if let Some(previous) = previous
            && instruction != previous.saturating_add(1)
        {
            return Err(CoreStateError::InstructionBufferOrder {
                warp_slot: warp.warp_slot,
                previous,
                found: instruction,
            });
        }
        previous = Some(instruction);
    }
    Ok(())
}

fn validate_memory_pipeline(
    state: &MemoryPipelineBoundaryState,
    cycle: u64,
) -> Result<(), CoreStateError> {
    if state.queue_entries_with_dispatch == 0
        || state.local_initiation_interval_cycles == 0
        || state.shared_accept_interval_cycles == 0
        || state.queue_release_times.len() != 4
        || state.local_completion_ready.len() != 4
    {
        return Err(CoreStateError::MemoryPipelineShape);
    }
    let mut live_entries = 0_u64;
    let mut maximum_live_entries = 0_u64;
    let mut maximum_release = None;
    for (subcore, queue) in state.queue_release_times.iter().enumerate() {
        if queue.iter().any(|release| *release < cycle) {
            return Err(CoreStateError::MemoryPipelineReleaseInPast);
        }
        if !queue.is_empty() && state.local_completion_ready[subcore] == 0 {
            return Err(CoreStateError::MemoryPipelineLocalFrontierMissing { subcore });
        }
        if queue.windows(2).any(|window| window[1] <= window[0]) {
            return Err(CoreStateError::MemoryPipelineOrder);
        }
        if queue.len() > state.queue_entries_with_dispatch as usize {
            return Err(CoreStateError::MemoryPipelineCapacity);
        }
        live_entries = live_entries
            .checked_add(u64::try_from(queue.len()).unwrap_or(u64::MAX))
            .ok_or(CoreStateError::CounterOverflow)?;
        maximum_live_entries =
            maximum_live_entries.max(u64::try_from(queue.len()).unwrap_or(u64::MAX));
        maximum_release = queue.iter().copied().max().max(maximum_release);
    }
    if state.instructions_admitted < live_entries {
        return Err(CoreStateError::MemoryPipelineAdmissionCount);
    }
    // Every native memory admission advances the selected SMSP's local
    // completion frontier. Once all queue entries drain, that historical
    // frontier remains nonzero; aggregate admissions with no live entries and
    // four zero local frontiers cannot be produced by the executable plan.
    if state.instructions_admitted != 0
        && live_entries == 0
        && state.local_completion_ready.iter().all(|&ready| ready == 0)
    {
        return Err(CoreStateError::MemoryPipelineHistory);
    }
    if state.maximum_queue_occupancy < maximum_live_entries
        || state.maximum_queue_occupancy > u64::from(state.queue_entries_with_dispatch)
        || (state.instructions_admitted != 0 && state.maximum_queue_occupancy == 0)
        || state.maximum_queue_occupancy > state.instructions_admitted
    {
        return Err(CoreStateError::MemoryPipelineOccupancy);
    }
    if state.maximum_operand_pipeline_delay_cycles > state.operand_pipeline_delay_cycles
        || state.maximum_shared_queue_delay_cycles > state.shared_queue_delay_cycles
    {
        return Err(CoreStateError::MemoryPipelineDelayMaximum);
    }
    if state.instructions_admitted == 0
        && (state.local_completion_ready.iter().any(|&ready| ready != 0)
            || state.terminal_shared_accept_cycle != 0
            || state.operand_pipeline_delay_cycles != 0
            || state.shared_queue_delay_cycles != 0
            || state.maximum_operand_pipeline_delay_cycles != 0
            || state.maximum_shared_queue_delay_cycles != 0)
    {
        return Err(CoreStateError::MemoryPipelineDiagnostics);
    }
    // This is an inequality bound, not a scheduled timestamp. Shared-port
    // contention can move the terminal acceptance past every local operand
    // completion, so this sum may exceed u64 even when all native frontiers
    // fit. Keep the comparison exact without rejecting that valid state.
    let local_frontier_limit = u128::from(state.terminal_shared_accept_cycle)
        + u128::from(state.local_initiation_interval_cycles);
    if state
        .local_completion_ready
        .iter()
        .any(|&ready| u128::from(ready) > local_frontier_limit)
    {
        return Err(CoreStateError::MemoryPipelineLocalFrontier);
    }
    if maximum_release.is_some_and(|release| state.terminal_shared_accept_cycle < release) {
        return Err(CoreStateError::MemoryPipelineTerminalCycle);
    }
    let expected_ready = if state.instructions_admitted == 0 {
        0
    } else {
        state
            .terminal_shared_accept_cycle
            .checked_add(state.shared_accept_interval_cycles)
            .ok_or(CoreStateError::CounterOverflow)?
    };
    if state.shared_accept_ready != expected_ready {
        return Err(CoreStateError::MemoryPipelineSharedFrontier);
    }
    if let Some(maximum_release) = maximum_release {
        let live_expected = maximum_release
            .checked_add(state.shared_accept_interval_cycles)
            .ok_or(CoreStateError::CounterOverflow)?;
        if state.shared_accept_ready < live_expected {
            return Err(CoreStateError::MemoryPipelineSharedFrontier);
        }
    }
    Ok(())
}

fn validate_scalar_pipeline(
    state: &ScalarPipelineBoundaryState,
    cycle: u64,
) -> Result<(), CoreStateError> {
    if state.latency_cycles == 0
        || state.initiation_interval_cycles == 0
        || state.in_flight_capacity == 0
        || state.release_times.len() != 4
        || state.next_service.len() != 4
    {
        return Err(CoreStateError::ScalarPipelineShape);
    }
    let mut live_entries = 0_u64;
    let mut maximum_live_entries = 0_u64;
    for (subcore, queue) in state.release_times.iter().enumerate() {
        if queue.iter().any(|release| *release < cycle) {
            return Err(CoreStateError::ScalarPipelineReleaseInPast);
        }
        if queue.windows(2).any(|window| window[1] <= window[0]) {
            return Err(CoreStateError::ScalarPipelineOrder);
        }
        if queue.len() > state.in_flight_capacity as usize {
            return Err(CoreStateError::ScalarPipelineCapacity);
        }
        live_entries = live_entries
            .checked_add(u64::try_from(queue.len()).unwrap_or(u64::MAX))
            .ok_or(CoreStateError::CounterOverflow)?;
        maximum_live_entries =
            maximum_live_entries.max(u64::try_from(queue.len()).unwrap_or(u64::MAX));
        if let Some(last_release) = queue.last() {
            let required_service = last_release
                .checked_add(state.initiation_interval_cycles)
                .ok_or(CoreStateError::CounterOverflow)?;
            if state.next_service[subcore] < required_service {
                return Err(CoreStateError::ScalarPipelineServiceFrontier);
            }
        }
        if state.next_service[subcore] != 0 {
            let first_service = state
                .latency_cycles
                .checked_add(state.initiation_interval_cycles)
                .ok_or(CoreStateError::CounterOverflow)?;
            if state.next_service[subcore] < first_service {
                return Err(CoreStateError::ScalarPipelineServiceOrigin);
            }
        }
    }
    if state.instructions_admitted < live_entries {
        return Err(CoreStateError::ScalarPipelineAdmissionCount);
    }
    // Every native scalar admission advances the selected SMSP's service
    // frontier. Once all release entries have drained, that frontier remains
    // nonzero; an aggregate admitted count with no live entries and four zero
    // frontiers cannot be produced by the executable pipeline.
    if state.instructions_admitted != 0
        && live_entries == 0
        && state.next_service.iter().all(|&ready| ready == 0)
    {
        return Err(CoreStateError::ScalarPipelineHistory);
    }
    if state.maximum_queue_occupancy < maximum_live_entries
        || state.maximum_queue_occupancy > u64::from(state.in_flight_capacity)
        || (state.instructions_admitted != 0 && state.maximum_queue_occupancy == 0)
        || state.maximum_queue_occupancy > state.instructions_admitted
    {
        return Err(CoreStateError::ScalarPipelineOccupancy);
    }
    if state.instructions_admitted == 0
        && (state.next_service.iter().any(|&ready| ready != 0) || state.queue_delay_cycles != 0)
    {
        return Err(CoreStateError::ScalarPipelineDiagnostics);
    }
    let expected_busy = state
        .instructions_admitted
        .checked_mul(state.initiation_interval_cycles)
        .ok_or(CoreStateError::CounterOverflow)?;
    if state.service_busy_cycles != expected_busy {
        return Err(CoreStateError::ScalarPipelineBusyCycles);
    }
    Ok(())
}

fn validate_block_barrier_state(
    block_index: usize,
    block: &BlockBoundaryState,
    warps: &[WarpBoundaryState],
    cycle: u64,
) -> Result<(), CoreStateError> {
    if block.barrier_generation == 0 && block.barrier_rearm_ready != 0 {
        return Err(CoreStateError::BlockBarrierRearmWithoutGeneration {
            block: block_index,
            rearm_ready: block.barrier_rearm_ready,
        });
    }
    let mut observed_waiting = 0_u64;
    let mut first_dispatch: Option<u64> = None;
    let mut first_arrival: Option<u64> = None;
    for warp in warps {
        if !warp.at_block_barrier {
            if warp.block_barrier_dispatch.is_some() || warp.block_barrier_arrival.is_some() {
                return Err(CoreStateError::BlockBarrierTimestampWithoutFlag {
                    block: block_index,
                    warp_slot: warp.warp_slot,
                });
            }
            continue;
        }
        if warp.done {
            return Err(CoreStateError::BlockBarrierDoneWarp {
                block: block_index,
                warp_slot: warp.warp_slot,
            });
        }
        let dispatch =
            warp.block_barrier_dispatch
                .ok_or(CoreStateError::BlockBarrierTimestampMissing {
                    block: block_index,
                    warp_slot: warp.warp_slot,
                })?;
        let arrival =
            warp.block_barrier_arrival
                .ok_or(CoreStateError::BlockBarrierTimestampMissing {
                    block: block_index,
                    warp_slot: warp.warp_slot,
                })?;
        if dispatch > cycle {
            return Err(CoreStateError::BlockBarrierDispatchAfterCycle {
                block: block_index,
                warp_slot: warp.warp_slot,
                dispatch,
                cycle,
            });
        }
        if arrival < dispatch {
            return Err(CoreStateError::BlockBarrierTimestampOrder {
                block: block_index,
                warp_slot: warp.warp_slot,
                dispatch,
                arrival,
            });
        }
        observed_waiting = observed_waiting
            .checked_add(1)
            .ok_or(CoreStateError::CounterOverflow)?;
        first_dispatch = Some(first_dispatch.map_or(dispatch, |current| current.min(dispatch)));
        first_arrival = Some(first_arrival.map_or(arrival, |current| current.min(arrival)));
    }
    if observed_waiting != block.waiting {
        return Err(CoreStateError::BlockBarrierWaitingCount {
            block: block_index,
            declared: block.waiting,
            observed: observed_waiting,
        });
    }
    if first_dispatch != block.barrier_first_dispatch {
        return Err(CoreStateError::BlockBarrierFirstDispatch {
            block: block_index,
            expected: first_dispatch,
            found: block.barrier_first_dispatch,
        });
    }
    if first_arrival != block.barrier_first_arrival {
        return Err(CoreStateError::BlockBarrierFirstArrival {
            block: block_index,
            expected: first_arrival,
            found: block.barrier_first_arrival,
        });
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CoreStateError {
    #[error("typed core local completion frontier disagrees with CTA-barrier release history")]
    LocalCompletionHistory,
    #[error("typed core SM identity uses the invalid u32::MAX sentinel")]
    InvalidSmIdentity,
    #[error("typed core must expose four SM subcore rosters, found {found}")]
    SubcoreCount { found: usize },
    #[error("typed core unprojected timing-state reasons are not canonical")]
    UnprojectedTimingState,
    #[error("typed core contains a negative warp slot {warp_slot}")]
    NegativeWarpSlot { warp_slot: i32 },
    #[error(
        "typed core warp slot {warp_slot} is listed on SMSP {subcore}, but SM86 maps it to SMSP {expected_subcore}"
    )]
    ResidentWarpWrongSubcore {
        warp_slot: i32,
        subcore: usize,
        expected_subcore: usize,
    },
    #[error("typed core contains duplicate resident warp slot {warp_slot}")]
    DuplicateResidentWarp { warp_slot: i32 },
    #[error("typed core resident warps reuse arrival stamp {arrival}")]
    DuplicateResidentWarpArrival { arrival: u64 },
    #[error("typed core issued count {declared} disagrees with warp cursors {warps}")]
    IssuedCount { declared: u64, warps: u64 },
    #[error("done typed warp {warp_slot} retains instruction-buffer entries")]
    DoneWarpInstructionBuffer { warp_slot: i32 },
    #[error("typed warp {warp_slot} carries instruction-buffer entries without a front-end")]
    InstructionBufferWithoutFrontEnd { warp_slot: i32 },
    #[error(
        "typed warp {warp_slot} instruction buffer has {found} entries, exceeding capacity {capacity}"
    )]
    InstructionBufferCapacity {
        warp_slot: i32,
        found: usize,
        capacity: usize,
    },
    #[error("typed warp {warp_slot} fetch cursor {cursor} is before next instruction {next}")]
    InstructionCursorBeforeNext {
        warp_slot: i32,
        cursor: u64,
        next: u64,
    },
    #[error(
        "typed warp {warp_slot} instruction buffer contains {found} before next instruction {next}"
    )]
    InstructionBufferBeforeNext {
        warp_slot: i32,
        next: u64,
        found: u64,
    },
    #[error(
        "typed warp {warp_slot} instruction buffer contains {found} at or beyond fetch cursor {cursor}"
    )]
    InstructionBufferBeyondCursor {
        warp_slot: i32,
        cursor: u64,
        found: u64,
    },
    #[error(
        "typed warp {warp_slot} instruction buffer is not contiguous after {previous}; found {found}"
    )]
    InstructionBufferOrder {
        warp_slot: i32,
        previous: u64,
        found: u64,
    },
    #[error("typed core resident-warp count {declared} disagrees with roster size {roster}")]
    ResidentWarpCount { declared: u32, roster: usize },
    #[error(
        "typed core resident-warp count {declared} disagrees with active block reservation {reservation}"
    )]
    ResidentWarpReservationCount { declared: u32, reservation: u64 },
    #[error("typed core resident warp slot {warp_slot} has no warp state")]
    ResidentWarpMissing { warp_slot: i32 },
    #[error("typed core unfinished warp slot {warp_slot} is absent from the resident roster")]
    UnfinishedWarpNotResident { warp_slot: i32 },
    #[error(
        "typed core resident roster has {roster} entries but {unfinished_states} unfinished warp states"
    )]
    CurrentRosterCount {
        roster: usize,
        unfinished_states: u64,
    },
    #[error(
        "typed core block {block} declares {declared} live warps but only {admitted} are admitted"
    )]
    BlockWarpCount {
        block: usize,
        declared: u32,
        admitted: u64,
    },
    #[error("typed core block {block} unfinished count exceeds admitted warps")]
    BlockUnfinishedCount { block: usize },
    #[error(
        "typed core block {block} unfinished count {unfinished} exceeds live warp count {warp_count}"
    )]
    BlockUnfinishedExceedsWarpCount {
        block: usize,
        unfinished: u64,
        warp_count: u32,
    },
    #[error("typed core block {block} waiting count exceeds unfinished warps")]
    BlockWaitingCount { block: usize },
    #[error("typed core block {block} warp roster has {roster} entries, expected {declared}")]
    BlockWarpRoster {
        block: usize,
        declared: u64,
        roster: usize,
    },
    #[error(
        "typed core block {block} declares {declared} unfinished warps but has {states} unfinished warp states"
    )]
    BlockWarpStateCount {
        block: usize,
        declared: u64,
        states: u64,
    },
    #[error("typed core block references unknown warp slot {warp_slot}")]
    BlockWarpMissing { warp_slot: i32 },
    #[error("typed core block rosters duplicate warp slot {warp_slot}")]
    DuplicateBlockWarp { warp_slot: i32 },
    #[error("typed core block {block} first-warp frontier expected {expected}, found {found}")]
    BlockFirstWarpOrder {
        block: usize,
        expected: u64,
        found: u64,
    },
    #[error(
        "typed core block {block} warp range ({first_warp}, {admitted_warps}) exceeds {warp_states} warp states"
    )]
    BlockWarpRange {
        block: usize,
        first_warp: u64,
        admitted_warps: u64,
        warp_states: usize,
    },
    #[error("typed core block {block} warp range has a different slot roster")]
    BlockWarpRangeRoster { block: usize },
    #[error("typed core block {block} warp range contains a warp for block {found}")]
    BlockWarpRangeIdentity { block: usize, found: u64 },
    #[error("typed core warp arena ends at {found}, but block ranges cover only {expected} states")]
    WarpArenaTail { expected: u64, found: u64 },
    #[error(
        "typed core block {block} waiting count {declared} disagrees with barrier flags {observed}"
    )]
    BlockBarrierWaitingCount {
        block: usize,
        declared: u64,
        observed: u64,
    },
    #[error("typed core block {block} warp {warp_slot} is done while marked at its CTA barrier")]
    BlockBarrierDoneWarp { block: usize, warp_slot: i32 },
    #[error(
        "typed core block {block} warp {warp_slot} has barrier timestamps without its waiting flag"
    )]
    BlockBarrierTimestampWithoutFlag { block: usize, warp_slot: i32 },
    #[error("typed core block {block} warp {warp_slot} is missing a barrier timestamp")]
    BlockBarrierTimestampMissing { block: usize, warp_slot: i32 },
    #[error(
        "typed core block {block} warp {warp_slot} barrier dispatch {dispatch} is after core cycle {cycle}"
    )]
    BlockBarrierDispatchAfterCycle {
        block: usize,
        warp_slot: i32,
        dispatch: u64,
        cycle: u64,
    },
    #[error(
        "typed core block {block} warp {warp_slot} barrier dispatch {dispatch} is after arrival {arrival}"
    )]
    BlockBarrierTimestampOrder {
        block: usize,
        warp_slot: i32,
        dispatch: u64,
        arrival: u64,
    },
    #[error(
        "typed core block {block} first barrier dispatch expected {expected:?}, found {found:?}"
    )]
    BlockBarrierFirstDispatch {
        block: usize,
        expected: Option<u64>,
        found: Option<u64>,
    },
    #[error(
        "typed core block {block} first barrier arrival expected {expected:?}, found {found:?}"
    )]
    BlockBarrierFirstArrival {
        block: usize,
        expected: Option<u64>,
        found: Option<u64>,
    },
    #[error(
        "typed core block {block} has re-arm frontier {rearm_ready} before any barrier generation"
    )]
    BlockBarrierRearmWithoutGeneration { block: usize, rearm_ready: u64 },
    #[error(
        "typed core active blocks {previous_block} and {block} both retain warp slot {warp_slot}"
    )]
    ActiveBlockWarpSlotCollision {
        warp_slot: i32,
        previous_block: usize,
        block: usize,
    },
    #[error("typed core unfinished warp count {declared} disagrees with blocks {blocks}")]
    UnfinishedWarpCount { declared: u64, blocks: u64 },
    #[error(
        "typed core declares {declared} unfinished warps but exposes {states} unfinished warp states"
    )]
    WarpStateCount { declared: u64, states: u64 },
    #[error(
        "typed core resident shared-memory count {declared} disagrees with active blocks {blocks}"
    )]
    SharedMemoryCount { declared: u64, blocks: u64 },
    #[error("typed core accounting counter overflowed")]
    CounterOverflow,
    #[error("typed memory pipeline shape or configuration is invalid")]
    MemoryPipelineShape,
    #[error("typed memory pipeline release times are not strictly ordered")]
    MemoryPipelineOrder,
    #[error("typed memory pipeline queue exceeds its declared capacity")]
    MemoryPipelineCapacity,
    #[error("typed memory pipeline admitted-count is below live queue entries")]
    MemoryPipelineAdmissionCount,
    #[error("typed memory pipeline has admissions but no local frontier history")]
    MemoryPipelineHistory,
    #[error("typed memory pipeline occupancy diagnostic is inconsistent")]
    MemoryPipelineOccupancy,
    #[error("typed memory pipeline maximum delay exceeds its accumulated delay")]
    MemoryPipelineDelayMaximum,
    #[error("typed memory pipeline has diagnostics despite zero admissions")]
    MemoryPipelineDiagnostics,
    #[error("typed memory pipeline terminal cycle is before a live acceptance")]
    MemoryPipelineTerminalCycle,
    #[error("typed memory pipeline shared service frontier is inconsistent with its releases")]
    MemoryPipelineSharedFrontier,
    #[error("typed memory pipeline local completion frontier exceeds shared acceptance")]
    MemoryPipelineLocalFrontier,
    #[error("typed memory pipeline live queue {subcore} has no local completion frontier")]
    MemoryPipelineLocalFrontierMissing { subcore: usize },
    #[error("typed memory pipeline contains a release time before the core cycle")]
    MemoryPipelineReleaseInPast,
    #[error("typed scalar pipeline shape or configuration is invalid")]
    ScalarPipelineShape,
    #[error("typed scalar pipeline release times are not strictly ordered")]
    ScalarPipelineOrder,
    #[error("typed scalar pipeline queue exceeds its declared capacity")]
    ScalarPipelineCapacity,
    #[error("typed scalar pipeline admitted-count is below live queue entries")]
    ScalarPipelineAdmissionCount,
    #[error("typed scalar pipeline has admissions but no service frontier history")]
    ScalarPipelineHistory,
    #[error("typed scalar pipeline occupancy diagnostic is inconsistent")]
    ScalarPipelineOccupancy,
    #[error("typed scalar pipeline service frontier is behind a live release")]
    ScalarPipelineServiceFrontier,
    #[error("typed scalar pipeline service frontier precedes its first possible service")]
    ScalarPipelineServiceOrigin,
    #[error("typed scalar pipeline busy-cycle diagnostic is inconsistent")]
    ScalarPipelineBusyCycles,
    #[error("typed scalar pipeline has diagnostics despite zero admissions")]
    ScalarPipelineDiagnostics,
    #[error("typed scalar pipeline contains a release time before the core cycle")]
    ScalarPipelineReleaseInPast,
    #[error(transparent)]
    L1(#[from] L1StateError),
    #[error(transparent)]
    FrontEnd(#[from] FrontEndStateError),
    #[error(transparent)]
    AsyncCopy(#[from] AsyncCopyStateError),
    #[error("typed async-copy slot count does not match dynamic warp state")]
    AsyncCopySlots,
    #[error("typed core hold references non-resident warp slot {warp_slot}")]
    HoldWarpMissing { warp_slot: u64 },
    #[error(
        "typed core hold for SMSP {subcore} references warp slot {warp_slot} owned by another SMSP"
    )]
    HoldWarpWrongSubcore { subcore: usize, warp_slot: i32 },
    #[error("typed core contains duplicate warp state for slot {warp_slot}")]
    DuplicateWarpState { block: u64, warp_slot: i32 },
    #[error("typed core warp references unknown block {block}")]
    UnknownWarpBlock { block: u64 },
    #[error("typed core outstanding response count {declared} disagrees with owners {owners}")]
    ResponseCount { declared: u64, owners: usize },
    #[error("typed core response owner belongs to SM {owner}")]
    ResponseOwnerSm { owner: u32 },
    #[error(
        "typed core response owner warp index {response_warp} is outside {warp_states} serialized warp states"
    )]
    ResponseOwnerWarpOutOfRange {
        response_warp: u64,
        warp_states: usize,
    },
    #[error(
        "typed core response owner warp index {response_warp} names slot {state_warp_slot}, but owner carries slot {owner_warp_slot}"
    )]
    ResponseOwnerWarpSlotMismatch {
        response_warp: u64,
        owner_warp_slot: i32,
        state_warp_slot: i32,
    },
    #[error(
        "typed core response owner instruction {instruction} is after warp {response_warp} cursor {next_instruction}"
    )]
    ResponseOwnerInstructionAfterCursor {
        response_warp: u64,
        instruction: u64,
        next_instruction: u64,
    },
    #[error("typed core response owner is not outstanding")]
    ResponseOwnerState,
    #[error("typed core response owner has a negative warp slot {warp_slot}")]
    NegativeResponseOwnerWarpSlot { warp_slot: i32 },
    #[error("typed core contains duplicate response ownership")]
    DuplicateResponseOwner,
    #[error("typed core response owners are not in canonical order")]
    UnsortedResponseOwners,
    #[error(
        "typed core next-block frontier {next_block} disagrees with block states {block_states}"
    )]
    NextBlockMismatch {
        next_block: u64,
        block_states: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedEventBoundaryRecord {
    pub schema_version: u32,
    pub contract: TypedObservationContract,
    pub event: TypedEventIdentity,
    pub queue: QueueBoundaryState,
    pub cores: Vec<TypedCoreBoundaryState>,
    pub shared_l2: L2BoundaryState,
    /// Required shared service state, distinct from per-SM issue frontiers.
    pub dram: DramBoundaryState,
    pub power: PowerBoundaryState,
    pub completeness: Completeness,
}

impl TypedEventBoundaryRecord {
    /// Validate the structural state of one boundary without requiring the
    /// selected contract to be complete. Incomplete contract evidence is a
    /// valid diagnostic result; malformed typed state is an error.
    pub fn validate_structure(&self) -> Result<(), BoundaryStateError> {
        if self.schema_version != TYPED_STATE_SCHEMA_VERSION
            || self.contract.schema_version != TYPED_STATE_SCHEMA_VERSION
        {
            return Err(BoundaryStateError::SchemaVersion);
        }
        if self.event.phase != 0 {
            return Err(BoundaryStateError::UnsupportedPhase {
                phase: self.event.phase,
            });
        }
        self.event.payload.validate_event_time(self.event.time)?;
        if !self.cores.is_empty()
            && usize::try_from(self.event.target)
                .ok()
                .is_none_or(|target| target >= self.cores.len())
        {
            return Err(BoundaryStateError::EventTargetOutOfRange {
                target: self.event.target,
                cores: self.cores.len(),
            });
        }
        self.queue.validate()?;
        let event_end = self.event.sequence_end()?;
        self.queue.validate_reservation_end(event_end)?;
        for pending in &self.queue.pending {
            if !self.cores.is_empty()
                && usize::try_from(pending.target)
                    .ok()
                    .is_none_or(|target| target >= self.cores.len())
            {
                return Err(BoundaryStateError::EventTargetOutOfRange {
                    target: pending.target,
                    cores: self.cores.len(),
                });
            }
            if pending.phase != 0 {
                return Err(BoundaryStateError::UnsupportedPhase {
                    phase: pending.phase,
                });
            }
            pending.payload.validate_event_time(pending.time)?;
            let pending_end = pending
                .payload
                .sequence_end(pending.sequence, pending.sequence_span)?;
            if pending.sequence <= event_end && self.event.sequence <= pending_end {
                return Err(BoundaryStateError::CurrentEventOverlap {
                    sequence: pending.sequence,
                });
            }
            if (pending.time, pending.phase, pending.sequence)
                <= (self.event.time, self.event.phase, self.event.sequence)
            {
                return Err(BoundaryStateError::PendingEventOrder {
                    time: pending.time,
                    sequence: pending.sequence,
                });
            }
        }
        let mut previous_sm_id = None;
        for core in &self.cores {
            // The event timestamp is the global queue frontier. An SM actor
            // may intentionally run ahead to its own response horizon before
            // the queue applies cross-component events, so validate each
            // local clock and state independently rather than imposing a
            // false ordering between the two domains.
            if let Some(previous) = previous_sm_id
                && core.sm_id <= previous
            {
                return Err(BoundaryStateError::CoreIdentityOrder {
                    previous,
                    found: core.sm_id,
                });
            }
            previous_sm_id = Some(core.sm_id);
            core.validate()?
        }
        self.shared_l2.validate()?;
        self.dram.validate()?;
        for allocation in &self.shared_l2.allocations {
            for sector in &allocation.sectors {
                if matches!(sector.status, L2SectorStatus::Pending)
                    && sector
                        .ready_at
                        .is_some_and(|ready_at| ready_at < self.event.time)
                {
                    return Err(BoundaryStateError::L2PendingReadyBeforeBoundary {
                        line: allocation.line.clone(),
                        generation: allocation.generation,
                        sector: sector.sector,
                        ready_at: sector.ready_at.expect("checked above"),
                        event_time: self.event.time,
                    });
                }
            }
        }
        let validate_owner_due = |owner: &L2RequestOwner| {
            owner
                .due
                .filter(|&due| due < self.event.time)
                .map_or(Ok(()), |due| {
                    Err(BoundaryStateError::ResponseDueBeforeBoundary {
                        sm_id: owner.request.sm_id,
                        due,
                        event_time: self.event.time,
                    })
                })
        };
        for owner in &self.shared_l2.request_owners {
            validate_owner_due(owner)?;
        }
        for core in &self.cores {
            for owner in &core.response_owners {
                validate_owner_due(owner)?;
            }
        }
        // A complete boundary must retain the queue event that will consume
        // every known-due response owner. Checking the two projections only
        // independently would allow an owner or a queued response to vanish
        // while all local shapes still validate.
        if !self.cores.is_empty()
            && self.shared_l2.ownership_complete
            && self
                .shared_l2
                .request_owners
                .iter()
                .all(|owner| owner.due.is_some())
        {
            type ResponseQueueKey = (u32, u64, u64, u64, TypedResponseKind, u64);
            let mut owner_counts = BTreeMap::<ResponseQueueKey, u32>::new();
            for owner in &self.shared_l2.request_owners {
                let key = (
                    owner.request.sm_id,
                    owner.request.response_warp,
                    owner.request.instruction,
                    owner.request.sector_index,
                    owner.request.response_kind,
                    owner.due.expect("known-due owner was checked above"),
                );
                *owner_counts.entry(key).or_default() += 1;
            }
            let mut queue_counts = BTreeMap::<ResponseQueueKey, u32>::new();
            for pending in &self.queue.pending {
                let TypedEventPayload::MemoryResponse {
                    reply: Some(reply), ..
                } = &pending.payload
                else {
                    continue;
                };
                let target = usize::try_from(pending.target).map_err(|_| {
                    BoundaryStateError::EventTargetOutOfRange {
                        target: pending.target,
                        cores: self.cores.len(),
                    }
                })?;
                let sm_id = self.cores[target].sm_id;
                let key = (
                    sm_id,
                    reply.warp,
                    reply.instruction,
                    reply.sector_index,
                    reply.response_kind,
                    pending.time,
                );
                *queue_counts.entry(key).or_default() += 1;
            }
            let mut keys = BTreeSet::new();
            keys.extend(owner_counts.keys().copied());
            keys.extend(queue_counts.keys().copied());
            for (sm_id, response_warp, instruction, sector_index, response_kind, due) in keys {
                if owner_counts.get(&(
                    sm_id,
                    response_warp,
                    instruction,
                    sector_index,
                    response_kind,
                    due,
                )) != queue_counts.get(&(
                    sm_id,
                    response_warp,
                    instruction,
                    sector_index,
                    response_kind,
                    due,
                )) {
                    return Err(BoundaryStateError::ResponseOwnerQueueJoin {
                        sm_id,
                        response_warp,
                        instruction,
                        sector_index,
                        response_kind,
                        due,
                    });
                }
            }

            type FillQueueKey = (i64, u64, u64);
            let mut expected_fill_masks = BTreeMap::<FillQueueKey, u8>::new();
            for allocation in &self.shared_l2.allocations {
                let CanonicalLineKey::VirtualAbsolute { line_id } = allocation.line else {
                    continue;
                };
                for sector in &allocation.sectors {
                    if !matches!(sector.status, L2SectorStatus::Pending) {
                        continue;
                    }
                    let Some(ready) = sector.ready_at else {
                        continue;
                    };
                    *expected_fill_masks
                        .entry((line_id, allocation.generation, ready))
                        .or_default() |= 1_u8 << sector.sector;
                }
            }
            let mut queued_fill_masks = BTreeMap::<FillQueueKey, u8>::new();
            for pending in &self.queue.pending {
                let fill = match &pending.payload {
                    TypedEventPayload::L2Fill { fill } => Some(fill),
                    TypedEventPayload::MemoryResponse {
                        fill: Some(fill), ..
                    } => Some(fill),
                    _ => None,
                };
                let Some(fill) = fill else { continue };
                let key = (fill.line_id, fill.generation, pending.time);
                let previous = queued_fill_masks.entry(key).or_default();
                if *previous & fill.sector_mask != 0 {
                    return Err(BoundaryStateError::L2FillQueueJoin {
                        line_id: fill.line_id,
                        generation: fill.generation,
                        ready: pending.time,
                    });
                }
                *previous |= fill.sector_mask;
            }
            let mut fill_keys = BTreeSet::new();
            fill_keys.extend(expected_fill_masks.keys().copied());
            fill_keys.extend(queued_fill_masks.keys().copied());
            for (line_id, generation, ready) in fill_keys {
                if expected_fill_masks.get(&(line_id, generation, ready))
                    != queued_fill_masks.get(&(line_id, generation, ready))
                {
                    return Err(BoundaryStateError::L2FillQueueJoin {
                        line_id,
                        generation,
                        ready,
                    });
                }
            }
        }
        self.power.validate()?;
        if self
            .power
            .domains
            .iter()
            .any(|domain| domain.last_cycle != self.event.time)
        {
            return Err(BoundaryStateError::PowerTimeMismatch {
                event_time: self.event.time,
            });
        }
        let derived = self
            .contract
            .validate(&self.shared_l2, &self.power, &self.cores);
        if self.completeness != derived {
            return Err(BoundaryStateError::CompletenessMismatch {
                declared: self.completeness.clone(),
                derived,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BoundaryStateError {
    #[error("typed event boundary schema version is unsupported")]
    SchemaVersion,
    #[error("typed event phase {phase} is unsupported by the single-phase engine")]
    UnsupportedPhase { phase: u8 },
    #[error(
        "typed {payload} payload field {field} has timestamp {found}, expected event time {expected}"
    )]
    PayloadTimeMismatch {
        payload: &'static str,
        field: &'static str,
        expected: u64,
        found: u64,
    },
    #[error(
        "typed {payload} payload timestamp {earlier_field}={earlier} is after {later_field}={later}"
    )]
    PayloadTimestampOrder {
        payload: &'static str,
        earlier_field: &'static str,
        earlier: u64,
        later_field: &'static str,
        later: u64,
    },
    #[error(
        "typed memory request for sector {sector_id} has invalid masks sector={sector_mask:#06b}, local={local_sector_mask:#06b}"
    )]
    MemoryRequestMask {
        sector_id: i64,
        sector_mask: u8,
        local_sector_mask: u8,
    },
    #[error(
        "typed memory request has incoherent source/operation/reply contract: source={request_source:?}, operation={operation:?}, reply={reply:?}"
    )]
    MemoryRequestContract {
        request_source: TypedL2Source,
        operation: TypedMemoryOperation,
        reply: Option<TypedResponseKind>,
    },
    #[error(
        "typed memory request admission witness has subcore {subcore}, index {request_index}, count {request_count}"
    )]
    MemoryRequestWitness {
        subcore: u8,
        request_index: u16,
        request_count: u16,
    },
    #[error("typed batched memory requests change admission witness group identity")]
    MemoryRequestWitnessGroupMismatch,
    #[error("typed batched memory request witness index expected {expected}, found {found}")]
    MemoryRequestWitnessSequence { expected: u16, found: u16 },
    #[error(
        "typed memory request constant markers are incoherent: source={request_source:?}, operation={operation:?}, constant_load={constant_load}, constant_deep={constant_deep}"
    )]
    MemoryRequestConstantContract {
        request_source: TypedL2Source,
        operation: TypedMemoryOperation,
        constant_load: bool,
        constant_deep: bool,
    },
    #[error(
        "typed memory request sector mask {sector_mask:#06b} is incompatible with {response_kind:?}"
    )]
    MemoryRequestSectorReplyMismatch {
        response_kind: TypedResponseKind,
        sector_mask: u8,
    },
    #[error("typed L2 fill identity has invalid sector mask {sector_mask:#06b}")]
    FillIdentityMask { sector_mask: u8 },
    #[error("typed response identity uses response kind none")]
    ResponseIdentityKindNone,
    #[error("typed memory response event is missing its returning reply identity")]
    MemoryResponseReplyMissing,
    #[error(
        "typed memory response fill is paired with incompatible response kind {response_kind:?}"
    )]
    MemoryResponseFillReplyMismatch { response_kind: TypedResponseKind },
    #[error("typed memory write acknowledgement has unsupported operation {operation:?}")]
    MemoryWriteAckOperation { operation: TypedMemoryOperation },
    #[error(
        "typed memory write acknowledgement witness has subcore {subcore}, index {request_index}, count {request_count}"
    )]
    MemoryWriteAckWitness {
        subcore: u8,
        request_index: u16,
        request_count: u16,
    },
    #[error(transparent)]
    Sequence(#[from] EventSequenceError),
    #[error("pending sequence {sequence} overlaps the current event's reservation")]
    CurrentEventOverlap { sequence: u64 },
    #[error("pending event ({time},{sequence}) is at or before the current event")]
    PendingEventOrder { time: u64, sequence: u64 },
    #[error("typed event target {target} is outside {cores} core snapshots")]
    EventTargetOutOfRange { target: u32, cores: usize },
    #[error("typed core SM identity {found} is not after previous identity {previous}")]
    CoreIdentityOrder { previous: u32, found: u32 },
    #[error("typed power domain time does not match event time {event_time}")]
    PowerTimeMismatch { event_time: u64 },
    #[error(
        "typed response owner for SM {sm_id} is due at {due}, before boundary event time {event_time}"
    )]
    ResponseDueBeforeBoundary {
        sm_id: u32,
        due: u64,
        event_time: u64,
    },
    #[error(
        "typed response owner/queue join is inconsistent for SM {sm_id} warp {response_warp} instruction {instruction} sector {sector_index} kind {response_kind:?} due {due}"
    )]
    ResponseOwnerQueueJoin {
        sm_id: u32,
        response_warp: u64,
        instruction: u64,
        sector_index: u64,
        response_kind: TypedResponseKind,
        due: u64,
    },
    #[error(
        "typed L2 pending-fill/queue join is inconsistent for line {line_id} generation {generation} ready {ready}"
    )]
    L2FillQueueJoin {
        line_id: i64,
        generation: u64,
        ready: u64,
    },
    #[error(
        "typed pending L2 sector {line:?}/{generation}/{sector} is ready at {ready_at}, before boundary event time {event_time}"
    )]
    L2PendingReadyBeforeBoundary {
        line: CanonicalLineKey,
        generation: u64,
        sector: u8,
        ready_at: u64,
        event_time: u64,
    },
    #[error(transparent)]
    Queue(#[from] QueueStateError),
    #[error(transparent)]
    Core(#[from] CoreStateError),
    #[error(transparent)]
    L2(#[from] L2StateError),
    #[error(transparent)]
    Dram(#[from] DramStateError),
    #[error(transparent)]
    Power(#[from] PowerStateError),
    #[error("typed boundary completeness declaration disagrees with its contract and state")]
    CompletenessMismatch {
        declared: Completeness,
        derived: Completeness,
    },
}
