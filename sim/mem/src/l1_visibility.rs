//! Opt-in stateful L1 visibility/ownership directory.
//!
//! This queue is the missing state contract between per-SM L1 accesses and
//! the existing ordered-L2 coherence experiment.  It consumes an observation
//! for *every* eligible global L1 access, including resident hits; a caller
//! that only forwards L2 misses cannot establish invalidation correctness.
//!
//! The queue is deliberately separate from [`crate::coherence::CoherenceQueue`.
//! It does not silently attach to native replay, does not infer memory-order
//! scopes, and does not flush dirty data.  It provides a transactional
//! per-line/per-sector ownership transition with explicit invalidation targets
//! so the SM event layer can later schedule targeted invalidation callbacks.
//! Missing observations remain a caller-side refusal, not a cold-start guess.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};

const SECTOR_MASK: u8 = 0b1111;
const MAX_LINES: usize = 1_000_000;
static NEXT_QUEUE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct SectorState {
    owner: Option<u8>,
    sharers: u128,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct LineState {
    sectors: [SectorState; 4],
}

/// Reuses the bounded service assumptions of the ordered-L2 directory, but
/// keeps the L1 visibility queue a distinct opt-in mechanism.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L1VisibilityConfig {
    service_cycles_per_sector: u64,
    capacity_sectors: u32,
}

impl L1VisibilityConfig {
    pub const fn new(
        service_cycles_per_sector: u64,
        capacity_sectors: u32,
    ) -> Result<Self, L1VisibilityError> {
        if service_cycles_per_sector == 0 {
            return Err(L1VisibilityError::ServiceCyclesZero);
        }
        if capacity_sectors == 0 {
            return Err(L1VisibilityError::CapacityZero);
        }
        Ok(Self {
            service_cycles_per_sector,
            capacity_sectors,
        })
    }

    pub const fn service_cycles_per_sector(self) -> u64 {
        self.service_cycles_per_sector
    }

    pub const fn capacity_sectors(self) -> u32 {
        self.capacity_sectors
    }
}

/// One complete observation from one SM's L1 access path. `is_write` means
/// that the access requires exclusive ownership (store/atomic/reduction).
/// `hit` is retained for evidence and does not change protocol semantics: a
/// resident hit is precisely the access that the old L2-only queue could not
/// observe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L1AccessObservation {
    /// Monotonic order within one SM's typed visibility stream. It is not a
    /// timestamp substitute: equal-cycle events remain ordered explicitly.
    pub sequence: u64,
    pub line_id: i64,
    pub sector_mask: u8,
    pub requester: u8,
    pub is_write: bool,
    /// Whether the access is expected to leave a line resident in this L1.
    /// SM86's current store hypothesis is write-through/no-allocate, so a
    /// store can invalidate remote readers without becoming an L1 owner.
    pub l1_resident_after: bool,
    /// A cacheable read miss reserves an L1 allocation but is not visible
    /// until its matching fill callback.  Bypass/nonresident reads keep this
    /// false and must not create a phantom sharer or pending fill.
    pub fill_will_reside: bool,
    /// Allocation identity carried by a cacheable miss and repeated by its
    /// fill-visible callback.  A generation is mandatory when
    /// `fill_will_reside` is true; stale callbacks are rejected.
    pub fill_generation: Option<u32>,
    pub hit: bool,
    pub issue_cycle: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisibilityScope {
    Block,
    Device,
    System,
}

impl VisibilityScope {
    const fn index(self) -> usize {
        match self {
            Self::Block => 0,
            Self::Device => 1,
            Self::System => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisibilityOrder {
    Acquire,
    Release,
    AcquireRelease,
    SequentiallyConsistent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L1FenceObservation {
    /// Same per-SM ordered stream as [`L1AccessObservation::sequence`].
    pub sequence: u64,
    pub requester: u8,
    pub pc: u32,
    pub issue_cycle: u64,
    pub scope: VisibilityScope,
    pub order: VisibilityOrder,
}

/// Retained per-requester fence frontiers. These are protocol state, not a
/// claim that the proprietary hardware fence implementation has been
/// reconstructed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct L1FenceState {
    pub release: [Option<u64>; 3],
    pub acquire: [Option<u64>; 3],
}

/// A cacheable read fill became resident after its lower-level response. An
/// issued miss must not publish L1 ownership until this event is observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L1FillVisibleObservation {
    pub sequence: u64,
    pub requester: u8,
    pub line_id: i64,
    pub sector_mask: u8,
    pub fill_generation: u32,
    pub visible_cycle: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L1InvalidateAllObservation {
    pub sequence: u64,
    pub requester: u8,
    pub issue_cycle: u64,
}

/// Ordered L1 visibility input. Keeping fences and accesses in one stream
/// prevents a later consumer from sorting two separately collected vectors by
/// cycle and silently reversing equal-cycle memory-order events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L1VisibilityEvent {
    Access(L1AccessObservation),
    Fence(L1FenceObservation),
    FillVisible(L1FillVisibleObservation),
    InvalidateAll(L1InvalidateAllObservation),
}

/// One L1 that must be invalidated before a remote writer becomes visible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L1InvalidationTarget {
    pub requester: u8,
    pub sector_mask: u8,
}

/// A transactionally preflighted visibility transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct L1VisibilityPlan {
    queue_id: u64,
    token: u64,
    request: L1AccessObservation,
    service_sectors: u32,
    service_start_cycle: u64,
    service_done_cycle: u64,
    expected_epoch: u64,
    next_line: LineState,
    invalidations: Box<[L1InvalidationTarget]>,
}

impl L1VisibilityPlan {
    pub const fn token(&self) -> u64 {
        self.token
    }

    pub const fn line_id(&self) -> i64 {
        self.request.line_id
    }

    pub const fn requester(&self) -> u8 {
        self.request.requester
    }

    pub const fn sector_mask(&self) -> u8 {
        self.request.sector_mask
    }

    pub const fn service_sectors(&self) -> u32 {
        self.service_sectors
    }

    pub const fn service_start_cycle(&self) -> u64 {
        self.service_start_cycle
    }

    pub const fn service_done_cycle(&self) -> u64 {
        self.service_done_cycle
    }

    pub const fn needs_completion(&self) -> bool {
        self.service_sectors != 0
    }

    pub fn invalidations(&self) -> &[L1InvalidationTarget] {
        &self.invalidations
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L1VisibilityCompletion {
    pub token: u64,
    pub line_id: i64,
    pub sector_mask: u8,
    pub service_sectors: u32,
    pub service_done_cycle: u64,
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum L1VisibilityError {
    #[error("L1 visibility service cycles must be positive")]
    ServiceCyclesZero,
    #[error("L1 visibility capacity must be positive")]
    CapacityZero,
    #[error("L1 visibility requester {0} is outside the 0..128 domain")]
    RequesterOutOfRange(u8),
    #[error("L1 visibility sector mask {0:#06b} is invalid")]
    SectorMaskInvalid(u8),
    #[error("L1 visibility service capacity {capacity} exceeded by {requested} sectors")]
    CapacityExceeded { capacity: u32, requested: u32 },
    #[error("L1 visibility line limit {0} reached")]
    LineLimit(usize),
    #[error("L1 visibility request time {requested} regresses event frontier {frontier}")]
    TimeRegression { requested: u64, frontier: u64 },
    #[error("L1 visibility line {line_id} has pending invalidation through cycle {ready_cycle}")]
    LinePending { line_id: i64, ready_cycle: u64 },
    #[error("L1 visibility transaction is stale (expected epoch {expected}, current {current})")]
    StalePlan { expected: u64, current: u64 },
    #[error("L1 visibility token {0} is unknown or already completed")]
    UnknownToken(u64),
    #[error("L1 visibility token {token} completed at {at}, before service time {ready}")]
    EarlyCompletion { token: u64, at: u64, ready: u64 },
    #[error("L1 visibility arithmetic overflow")]
    ArithmeticOverflow,
    #[error(
        "L1 visibility sequence for requester {requester} regresses: requested {requested}, previous {previous}"
    )]
    SequenceRegression {
        requester: u8,
        requested: u64,
        previous: u64,
    },
    #[error("L1 fill visibility for requester {requester} line {line_id} conflicts with an owner")]
    FillVisibilityConflict { requester: u8, line_id: i64 },
    #[error("L1 visibility read miss for requester {requester} line {line_id} has no pending fill")]
    FillVisibilityMissing { requester: u8, line_id: i64 },
    #[error("L1 visibility line {line_id} has a pending fill from another requester")]
    PendingFillConflict { line_id: i64 },
    #[error("L1 visibility cacheable miss has no allocation generation")]
    FillGenerationMissing,
    #[error(
        "L1 visibility fill generation mismatch for requester {requester} line {line_id}: expected {expected}, got {got}"
    )]
    FillGenerationMismatch {
        requester: u8,
        line_id: i64,
        expected: u32,
        got: u32,
    },
    #[error("L1 visibility residency metadata is inconsistent")]
    ResidencyMetadataInvalid,
    #[error(
        "L1 visibility read for requester {requester} line {line_id} requires an explicit volatile/proxy/atomic qualifier"
    )]
    VisibilityQualifierMissing { requester: u8, line_id: i64 },
}

/// Transactional L1 ownership/invalidation directory.
#[derive(Debug, PartialEq, Eq)]
pub struct L1VisibilityQueue {
    queue_id: u64,
    config: L1VisibilityConfig,
    free_cycle: u64,
    event_frontier: u64,
    in_flight_sectors: u32,
    next_token: u64,
    epoch: u64,
    lines: BTreeMap<i64, LineState>,
    pending: BTreeMap<u64, L1VisibilityPlan>,
    pending_fills: BTreeMap<(u8, i64), (u8, u32)>,
    pending_fill_lines: BTreeMap<(u8, i64), LineState>,
    last_sequences: [Option<u64>; 128],
    fence_states: [L1FenceState; 128],
}

impl L1VisibilityQueue {
    pub fn new(config: L1VisibilityConfig) -> Self {
        Self {
            queue_id: NEXT_QUEUE_ID.fetch_add(1, Ordering::Relaxed),
            config,
            free_cycle: 0,
            event_frontier: 0,
            in_flight_sectors: 0,
            next_token: 0,
            epoch: 0,
            lines: BTreeMap::new(),
            pending: BTreeMap::new(),
            pending_fills: BTreeMap::new(),
            pending_fill_lines: BTreeMap::new(),
            last_sequences: [None; 128],
            fence_states: [L1FenceState::default(); 128],
        }
    }

    pub const fn config(&self) -> L1VisibilityConfig {
        self.config
    }

    pub const fn in_flight_sectors(&self) -> u32 {
        self.in_flight_sectors
    }

    pub fn pending_tokens(&self) -> impl Iterator<Item = u64> + '_ {
        self.pending.keys().copied()
    }

    pub fn fence_state(&self, requester: u8) -> Result<L1FenceState, L1VisibilityError> {
        if requester >= 128 {
            return Err(L1VisibilityError::RequesterOutOfRange(requester));
        }
        Ok(self.fence_states[usize::from(requester)])
    }

    /// Consume a fence in the same ordered stream as L1 accesses. The state
    /// is intentionally explicit even where no service cycle is charged: a
    /// future release/acquire implementation must not infer ordering from
    /// timestamps after this boundary has been crossed.
    pub fn observe_fence(&mut self, fence: L1FenceObservation) -> Result<(), L1VisibilityError> {
        if fence.requester >= 128 {
            return Err(L1VisibilityError::RequesterOutOfRange(fence.requester));
        }
        if let Some(previous) = self.last_sequences[usize::from(fence.requester)]
            && fence.sequence <= previous
        {
            return Err(L1VisibilityError::SequenceRegression {
                requester: fence.requester,
                requested: fence.sequence,
                previous,
            });
        }
        if fence.issue_cycle < self.event_frontier {
            return Err(L1VisibilityError::TimeRegression {
                requested: fence.issue_cycle,
                frontier: self.event_frontier,
            });
        }
        let state = &mut self.fence_states[usize::from(fence.requester)];
        let scope = fence.scope.index();
        let update = |slot: &mut Option<u64>, cycle: u64| {
            *slot = Some(slot.map_or(cycle, |previous| previous.max(cycle)));
        };
        match fence.order {
            VisibilityOrder::Acquire => update(&mut state.acquire[scope], fence.issue_cycle),
            VisibilityOrder::Release => update(&mut state.release[scope], fence.issue_cycle),
            VisibilityOrder::AcquireRelease => {
                update(&mut state.acquire[scope], fence.issue_cycle);
                update(&mut state.release[scope], fence.issue_cycle);
            }
            VisibilityOrder::SequentiallyConsistent => {
                update(&mut state.acquire[2], fence.issue_cycle);
                update(&mut state.release[2], fence.issue_cycle);
            }
        }
        self.last_sequences[usize::from(fence.requester)] = Some(fence.sequence);
        self.event_frontier = fence.issue_cycle;
        self.epoch = self
            .epoch
            .checked_add(1)
            .ok_or(L1VisibilityError::ArithmeticOverflow)?;
        Ok(())
    }

    pub fn preview(
        &self,
        request: L1AccessObservation,
    ) -> Result<L1VisibilityPlan, L1VisibilityError> {
        validate_request(request)?;
        if let Some(previous) = self.last_sequences[usize::from(request.requester)]
            && request.sequence <= previous
        {
            return Err(L1VisibilityError::SequenceRegression {
                requester: request.requester,
                requested: request.sequence,
                previous,
            });
        }
        if request.issue_cycle < self.event_frontier {
            return Err(L1VisibilityError::TimeRegression {
                requested: request.issue_cycle,
                frontier: self.event_frontier,
            });
        }
        if let Some(pending) = self
            .pending
            .values()
            .find(|pending| pending.line_id() == request.line_id)
        {
            return Err(L1VisibilityError::LinePending {
                line_id: request.line_id,
                ready_cycle: pending.service_done_cycle,
            });
        }
        if let Some((pending_requester, _)) = self
            .pending_fills
            .keys()
            .find(|(_, line)| *line == request.line_id)
            && *pending_requester != request.requester
        {
            return Err(L1VisibilityError::PendingFillConflict {
                line_id: request.line_id,
            });
        }
        let pending_same_request = self
            .pending_fills
            .contains_key(&(request.requester, request.line_id));
        if request.fill_will_reside {
            let key = (request.requester, request.line_id);
            if let Some((_, expected_generation)) = self.pending_fills.get(&key) {
                let got = request
                    .fill_generation
                    .ok_or(L1VisibilityError::FillGenerationMissing)?;
                if *expected_generation != got {
                    return Err(L1VisibilityError::FillGenerationMismatch {
                        requester: request.requester,
                        line_id: request.line_id,
                        expected: *expected_generation,
                        got,
                    });
                }
                // Same-request sectors may merge into one generation-bearing
                // line fill.  The pending line snapshot below becomes the
                // post-fill state; the callback still publishes it only once
                // the lower response carries the exact aggregate mask.
            }
        } else if pending_same_request {
            return Err(L1VisibilityError::LinePending {
                line_id: request.line_id,
                ready_cycle: request.issue_cycle,
            });
        }
        let line = if pending_same_request {
            self.pending_fill_lines
                .get(&(request.requester, request.line_id))
                .copied()
                .ok_or(L1VisibilityError::FillVisibilityMissing {
                    requester: request.requester,
                    line_id: request.line_id,
                })?
        } else {
            self.lines
                .get(&request.line_id)
                .copied()
                .unwrap_or_default()
        };
        if !self.lines.contains_key(&request.line_id) && self.lines.len() >= MAX_LINES {
            return Err(L1VisibilityError::LineLimit(MAX_LINES));
        }
        let mut next_line = line;
        let requester_bit = 1_u128 << request.requester;
        let mut service_sectors = 0_u32;
        let mut target_masks: BTreeMap<u8, u8> = BTreeMap::new();

        for sector in 0..4_u8 {
            let bit = 1_u8 << sector;
            if request.sector_mask & bit == 0 {
                continue;
            }
            let state = &mut next_line.sectors[usize::from(sector)];
            let mut targets = BTreeSet::new();
            let remote_owner_read =
                !request.is_write && state.owner.is_some_and(|owner| owner != request.requester);
            if request.is_write
                && let Some(owner) = state.owner.filter(|owner| *owner != request.requester)
            {
                targets.insert(owner);
            }
            if request.is_write {
                for sharer in 0..128_u8 {
                    if sharer != request.requester && state.sharers & (1_u128 << sharer) != 0 {
                        targets.insert(sharer);
                    }
                }
            }
            if request.is_write {
                if !targets.is_empty() {
                    service_sectors = service_sectors
                        .checked_add(1)
                        .ok_or(L1VisibilityError::ArithmeticOverflow)?;
                    for target in targets {
                        *target_masks.entry(target).or_default() |= bit;
                    }
                }
                *state = if request.l1_resident_after {
                    SectorState {
                        owner: Some(request.requester),
                        sharers: 0,
                    }
                } else {
                    SectorState::default()
                };
            } else if state.owner == Some(request.requester) {
                // A read by the exclusive owner does not need a transfer.
            } else {
                if remote_owner_read || !targets.is_empty() {
                    service_sectors = service_sectors
                        .checked_add(1)
                        .ok_or(L1VisibilityError::ArithmeticOverflow)?;
                    for target in targets {
                        *target_masks.entry(target).or_default() |= bit;
                    }
                }
                // A cacheable miss computes the post-fill sharer state in
                // `next_line`, but the commit remains pending until the
                // generation-matched FillVisible event.  A bypass/nonresident
                // read must leave the directory untouched; treating it as a
                // sharer creates a false invalidation on the next writer.
                if request.l1_resident_after || request.fill_will_reside {
                    if let Some(owner) = state.owner.take() {
                        state.sharers |= 1_u128 << owner;
                    }
                    state.sharers |= requester_bit;
                }
            }
        }

        let requested_capacity = self
            .in_flight_sectors
            .checked_add(service_sectors)
            .ok_or(L1VisibilityError::ArithmeticOverflow)?;
        if requested_capacity > self.config.capacity_sectors {
            return Err(L1VisibilityError::CapacityExceeded {
                capacity: self.config.capacity_sectors,
                requested: requested_capacity,
            });
        }
        let service_start_cycle = if service_sectors == 0 {
            request.issue_cycle
        } else {
            self.free_cycle.max(request.issue_cycle)
        };
        let duration = u64::from(service_sectors)
            .checked_mul(self.config.service_cycles_per_sector)
            .ok_or(L1VisibilityError::ArithmeticOverflow)?;
        let service_done_cycle = service_start_cycle
            .checked_add(duration)
            .ok_or(L1VisibilityError::ArithmeticOverflow)?;
        let invalidations = target_masks
            .into_iter()
            .map(|(requester, sector_mask)| L1InvalidationTarget {
                requester,
                sector_mask,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Ok(L1VisibilityPlan {
            queue_id: self.queue_id,
            token: self.next_token,
            request,
            service_sectors,
            service_start_cycle,
            service_done_cycle,
            expected_epoch: self.epoch,
            next_line,
            invalidations,
        })
    }

    /// Preview a read with an explicit visibility qualifier. A plain
    /// cacheable read cannot silently consume another requester's resident
    /// ownership as if it were a synchronized observation; callers must
    /// provide volatile/proxy/atomic evidence when the line is remotely held.
    pub fn preview_qualified(
        &self,
        request: L1AccessObservation,
        qualifier: crate::memory_order::MemoryVisibilityQualifier,
    ) -> Result<L1VisibilityPlan, L1VisibilityError> {
        validate_request(request)?;
        if !request.is_write
            && matches!(
                qualifier,
                crate::memory_order::MemoryVisibilityQualifier::Cacheable
            )
            && self.lines.get(&request.line_id).is_some_and(|line| {
                line.sectors.iter().enumerate().any(|(index, sector)| {
                    request.sector_mask & (1_u8 << index) != 0
                        && (sector.owner.is_some_and(|owner| owner != request.requester)
                            || (sector.sharers & !(1_u128 << request.requester)) != 0)
                })
            })
        {
            return Err(L1VisibilityError::VisibilityQualifierMissing {
                requester: request.requester,
                line_id: request.line_id,
            });
        }
        self.preview(request)
    }

    pub fn commit(&mut self, plan: L1VisibilityPlan) -> Result<(), L1VisibilityError> {
        if plan.queue_id != self.queue_id || plan.expected_epoch != self.epoch {
            return Err(L1VisibilityError::StalePlan {
                expected: plan.expected_epoch,
                current: self.epoch,
            });
        }
        if plan.token != self.next_token || self.preview(plan.request)? != plan {
            return Err(L1VisibilityError::StalePlan {
                expected: plan.expected_epoch,
                current: self.epoch,
            });
        }
        // Check all generation-bearing state before mutating any queue
        // structure.  A failed commit must be observationally transactional,
        // including when a caller presents a stale allocation generation.
        if plan.request.fill_will_reside {
            let generation = plan
                .request
                .fill_generation
                .ok_or(L1VisibilityError::FillGenerationMissing)?;
            if let Some((_, expected_generation)) = self
                .pending_fills
                .get(&(plan.request.requester, plan.request.line_id))
                && *expected_generation != generation
            {
                return Err(L1VisibilityError::FillGenerationMismatch {
                    requester: plan.request.requester,
                    line_id: plan.request.line_id,
                    expected: *expected_generation,
                    got: generation,
                });
            }
        }
        let next_token = self
            .next_token
            .checked_add(1)
            .ok_or(L1VisibilityError::ArithmeticOverflow)?;
        let next_epoch = self
            .epoch
            .checked_add(1)
            .ok_or(L1VisibilityError::ArithmeticOverflow)?;
        if plan.needs_completion() {
            self.free_cycle = plan.service_done_cycle;
            self.pending.insert(plan.token, plan.clone());
        } else if !plan.request.fill_will_reside {
            self.lines.insert(plan.line_id(), plan.next_line);
        }
        if plan.request.fill_will_reside {
            let key = (plan.request.requester, plan.request.line_id);
            let generation = plan
                .request
                .fill_generation
                .ok_or(L1VisibilityError::FillGenerationMissing)?;
            let entry = self.pending_fills.entry(key).or_insert((0, generation));
            entry.0 |= plan.request.sector_mask;
            self.pending_fill_lines.insert(key, plan.next_line);
        }
        self.event_frontier = plan.request.issue_cycle;
        self.last_sequences[usize::from(plan.request.requester)] = Some(plan.request.sequence);
        self.in_flight_sectors = self
            .in_flight_sectors
            .checked_add(plan.service_sectors)
            .ok_or(L1VisibilityError::ArithmeticOverflow)?;
        self.next_token = next_token;
        self.epoch = next_epoch;
        Ok(())
    }

    pub fn complete(
        &mut self,
        token: u64,
        at_cycle: u64,
    ) -> Result<L1VisibilityCompletion, L1VisibilityError> {
        let pending = self
            .pending
            .remove(&token)
            .ok_or(L1VisibilityError::UnknownToken(token))?;
        if at_cycle < pending.service_done_cycle {
            self.pending.insert(token, pending.clone());
            return Err(L1VisibilityError::EarlyCompletion {
                token,
                at: at_cycle,
                ready: pending.service_done_cycle,
            });
        }
        if at_cycle < self.event_frontier {
            self.pending.insert(token, pending.clone());
            return Err(L1VisibilityError::TimeRegression {
                requested: at_cycle,
                frontier: self.event_frontier,
            });
        }
        // A cacheable read miss may have a visibility-service completion that
        // precedes the lower-level data response.  Service completion alone
        // must not publish an L1 sharer; the generation-matched FillVisible
        // callback is the sole publication edge.  If the callback arrived
        // first, inserting the same final state is harmless and unnecessary.
        if !pending.request.fill_will_reside {
            self.lines.insert(pending.line_id(), pending.next_line);
        }
        self.in_flight_sectors = self
            .in_flight_sectors
            .checked_sub(pending.service_sectors)
            .ok_or(L1VisibilityError::ArithmeticOverflow)?;
        self.event_frontier = self.event_frontier.max(at_cycle);
        self.epoch = self
            .epoch
            .checked_add(1)
            .ok_or(L1VisibilityError::ArithmeticOverflow)?;
        Ok(L1VisibilityCompletion {
            token,
            line_id: pending.line_id(),
            sector_mask: pending.sector_mask(),
            service_sectors: pending.service_sectors,
            service_done_cycle: pending.service_done_cycle,
        })
    }

    pub fn observe_fill_visible(
        &mut self,
        fill: L1FillVisibleObservation,
    ) -> Result<(), L1VisibilityError> {
        if fill.requester >= 128 {
            return Err(L1VisibilityError::RequesterOutOfRange(fill.requester));
        }
        if fill.sector_mask == 0 || fill.sector_mask & !SECTOR_MASK != 0 {
            return Err(L1VisibilityError::SectorMaskInvalid(fill.sector_mask));
        }
        if let Some(previous) = self.last_sequences[usize::from(fill.requester)]
            && fill.sequence <= previous
        {
            return Err(L1VisibilityError::SequenceRegression {
                requester: fill.requester,
                requested: fill.sequence,
                previous,
            });
        }
        if fill.visible_cycle < self.event_frontier {
            return Err(L1VisibilityError::TimeRegression {
                requested: fill.visible_cycle,
                frontier: self.event_frontier,
            });
        }
        let key = (fill.requester, fill.line_id);
        let (pending, expected_generation) = self.pending_fills.get(&key).copied().ok_or(
            L1VisibilityError::FillVisibilityMissing {
                requester: fill.requester,
                line_id: fill.line_id,
            },
        )?;
        if fill.fill_generation != expected_generation {
            return Err(L1VisibilityError::FillGenerationMismatch {
                requester: fill.requester,
                line_id: fill.line_id,
                expected: expected_generation,
                got: fill.fill_generation,
            });
        }
        if fill.sector_mask & !pending != 0 {
            return Err(L1VisibilityError::FillVisibilityMissing {
                requester: fill.requester,
                line_id: fill.line_id,
            });
        }
        let key = (fill.requester, fill.line_id);
        let pending_line = self.pending_fill_lines.get_mut(&key).ok_or(
            L1VisibilityError::FillVisibilityMissing {
                requester: fill.requester,
                line_id: fill.line_id,
            },
        )?;
        let state = pending_line;
        for sector in 0..4_u8 {
            let bit = 1_u8 << sector;
            if fill.sector_mask & bit == 0 {
                continue;
            }
            let sector_state = &mut state.sectors[usize::from(sector)];
            sector_state.owner = None;
            sector_state.sharers |= 1_u128 << fill.requester;
        }
        let remaining = pending & !fill.sector_mask;
        if remaining == 0 {
            self.pending_fills.remove(&key);
            let visible_line = self
                .pending_fill_lines
                .remove(&key)
                .expect("pending fill line remains until final sector visible");
            self.lines.insert(fill.line_id, visible_line);
        } else {
            self.pending_fills
                .insert(key, (remaining, expected_generation));
        }
        self.last_sequences[usize::from(fill.requester)] = Some(fill.sequence);
        self.event_frontier = fill.visible_cycle;
        self.epoch = self
            .epoch
            .checked_add(1)
            .ok_or(L1VisibilityError::ArithmeticOverflow)?;
        Ok(())
    }

    pub fn observe_invalidate_all(
        &mut self,
        invalidation: L1InvalidateAllObservation,
    ) -> Result<(), L1VisibilityError> {
        if invalidation.requester >= 128 {
            return Err(L1VisibilityError::RequesterOutOfRange(
                invalidation.requester,
            ));
        }
        if let Some(previous) = self.last_sequences[usize::from(invalidation.requester)]
            && invalidation.sequence <= previous
        {
            return Err(L1VisibilityError::SequenceRegression {
                requester: invalidation.requester,
                requested: invalidation.sequence,
                previous,
            });
        }
        if invalidation.issue_cycle < self.event_frontier {
            return Err(L1VisibilityError::TimeRegression {
                requested: invalidation.issue_cycle,
                frontier: self.event_frontier,
            });
        }
        let bit = 1_u128 << invalidation.requester;
        for line in self.lines.values_mut() {
            for sector in &mut line.sectors {
                if sector.owner == Some(invalidation.requester) {
                    sector.owner = None;
                }
                sector.sharers &= !bit;
            }
        }
        self.last_sequences[usize::from(invalidation.requester)] = Some(invalidation.sequence);
        self.event_frontier = invalidation.issue_cycle;
        self.epoch = self
            .epoch
            .checked_add(1)
            .ok_or(L1VisibilityError::ArithmeticOverflow)?;
        Ok(())
    }
}

fn validate_request(request: L1AccessObservation) -> Result<(), L1VisibilityError> {
    if request.requester >= 128 {
        return Err(L1VisibilityError::RequesterOutOfRange(request.requester));
    }
    if request.sector_mask == 0 || request.sector_mask & !SECTOR_MASK != 0 {
        return Err(L1VisibilityError::SectorMaskInvalid(request.sector_mask));
    }
    match (request.fill_will_reside, request.fill_generation) {
        (true, Some(_)) if !request.is_write && !request.hit && !request.l1_resident_after => {}
        (true, _) => return Err(L1VisibilityError::ResidencyMetadataInvalid),
        (false, Some(_)) => return Err(L1VisibilityError::ResidencyMetadataInvalid),
        (false, None) => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> L1VisibilityConfig {
        L1VisibilityConfig::new(2, 8).expect("valid config")
    }

    fn read(line_id: i64, requester: u8, issue_cycle: u64) -> L1AccessObservation {
        L1AccessObservation {
            sequence: issue_cycle,
            line_id,
            sector_mask: 1,
            requester,
            is_write: false,
            l1_resident_after: true,
            fill_will_reside: false,
            fill_generation: None,
            hit: true,
            issue_cycle,
        }
    }

    #[test]
    fn remote_write_plans_targeted_l1_invalidation() {
        let mut queue = L1VisibilityQueue::new(config());
        let first = queue.preview(read(4, 0, 0)).unwrap();
        queue.commit(first).unwrap();
        let second = queue.preview(read(4, 1, 0)).unwrap();
        queue.commit(second).unwrap();
        let write = queue
            .preview(L1AccessObservation {
                sequence: 0,
                line_id: 4,
                sector_mask: 1,
                requester: 2,
                is_write: true,
                l1_resident_after: false,
                fill_will_reside: false,
                fill_generation: None,
                hit: false,
                issue_cycle: 1,
            })
            .unwrap();
        assert_eq!(write.service_sectors(), 1);
        assert_eq!(
            write.invalidations(),
            &[
                L1InvalidationTarget {
                    requester: 0,
                    sector_mask: 1,
                },
                L1InvalidationTarget {
                    requester: 1,
                    sector_mask: 1,
                }
            ]
        );
        queue.commit(write.clone()).unwrap();
        assert!(matches!(
            queue.preview(read(4, 0, 1)),
            Err(L1VisibilityError::LinePending { .. })
        ));
        assert_eq!(
            queue.complete(write.token(), write.service_done_cycle()),
            Ok(L1VisibilityCompletion {
                token: write.token(),
                line_id: 4,
                sector_mask: 1,
                service_sectors: 1,
                service_done_cycle: write.service_done_cycle(),
            })
        );
        let owner_read = queue
            .preview(read(4, 2, write.service_done_cycle()))
            .unwrap();
        assert_eq!(owner_read.service_sectors(), 0);
    }

    #[test]
    fn cacheable_remote_read_requires_explicit_visibility_qualifier() {
        let mut queue = L1VisibilityQueue::new(config());
        queue.commit(queue.preview(read(4, 0, 0)).unwrap()).unwrap();
        let remote = read(4, 1, 1);
        assert_eq!(
            queue.preview_qualified(
                remote,
                crate::memory_order::MemoryVisibilityQualifier::Cacheable,
            ),
            Err(L1VisibilityError::VisibilityQualifierMissing {
                requester: 1,
                line_id: 4,
            })
        );
        queue
            .preview_qualified(
                remote,
                crate::memory_order::MemoryVisibilityQualifier::Volatile,
            )
            .unwrap();
    }

    #[test]
    fn read_read_sharing_is_free_and_outcome_is_visible_immediately() {
        let mut queue = L1VisibilityQueue::new(config());
        queue.commit(queue.preview(read(1, 0, 0)).unwrap()).unwrap();
        let second = queue.preview(read(1, 1, 0)).unwrap();
        assert_eq!(second.service_sectors(), 0);
        assert!(second.invalidations().is_empty());
    }

    #[test]
    fn remote_owner_read_downgrades_to_sharing_without_invalidating_owner() {
        let mut queue = L1VisibilityQueue::new(config());
        let owner = L1AccessObservation {
            sequence: 0,
            line_id: 2,
            sector_mask: 1,
            requester: 0,
            is_write: true,
            l1_resident_after: true,
            fill_will_reside: false,
            fill_generation: None,
            hit: false,
            issue_cycle: 0,
        };
        queue.commit(queue.preview(owner).unwrap()).unwrap();
        let remote = queue.preview(read(2, 1, 1)).unwrap();
        assert_eq!(remote.service_sectors(), 1);
        assert!(remote.invalidations().is_empty());
        queue.commit(remote.clone()).unwrap();
        queue
            .complete(remote.token(), remote.service_done_cycle())
            .unwrap();
        let owner_again = queue
            .preview(read(2, 0, remote.service_done_cycle()))
            .unwrap();
        assert_eq!(owner_again.service_sectors(), 0);
        assert!(owner_again.invalidations().is_empty());
    }

    #[test]
    fn remote_owner_fill_and_service_order_have_same_sharing_result() {
        fn run(fill_first: bool) -> (u32, u32) {
            let mut queue = L1VisibilityQueue::new(config());
            let owner = L1AccessObservation {
                sequence: 0,
                line_id: 21,
                sector_mask: 1,
                requester: 0,
                is_write: true,
                l1_resident_after: true,
                fill_will_reside: false,
                fill_generation: None,
                hit: false,
                issue_cycle: 0,
            };
            queue.commit(queue.preview(owner).unwrap()).unwrap();
            let miss = L1AccessObservation {
                sequence: 0,
                line_id: 21,
                sector_mask: 1,
                requester: 1,
                is_write: false,
                l1_resident_after: false,
                fill_will_reside: true,
                fill_generation: Some(11),
                hit: false,
                issue_cycle: 1,
            };
            let plan = queue.preview(miss).unwrap();
            assert!(plan.invalidations().is_empty());
            queue.commit(plan.clone()).unwrap();
            let visible = L1FillVisibleObservation {
                sequence: 1,
                requester: 1,
                line_id: 21,
                sector_mask: 1,
                fill_generation: 11,
                visible_cycle: if fill_first { 2 } else { 4 },
            };
            if fill_first {
                queue.observe_fill_visible(visible).unwrap();
                queue
                    .complete(plan.token(), plan.service_done_cycle())
                    .unwrap();
            } else {
                queue
                    .complete(plan.token(), plan.service_done_cycle())
                    .unwrap();
                assert!(matches!(
                    queue.preview(L1AccessObservation {
                        sequence: 1,
                        line_id: 21,
                        sector_mask: 1,
                        requester: 1,
                        is_write: false,
                        l1_resident_after: true,
                        fill_will_reside: false,
                        fill_generation: None,
                        hit: true,
                        issue_cycle: 4,
                    }),
                    Err(L1VisibilityError::LinePending { .. })
                ));
                queue.observe_fill_visible(visible).unwrap();
            }
            let owner_read = queue
                .preview(L1AccessObservation {
                    sequence: 1,
                    line_id: 21,
                    sector_mask: 1,
                    requester: 0,
                    is_write: false,
                    l1_resident_after: true,
                    fill_will_reside: false,
                    fill_generation: None,
                    hit: true,
                    issue_cycle: 5,
                })
                .unwrap();
            let reader_read = queue
                .preview(L1AccessObservation {
                    sequence: 2,
                    line_id: 21,
                    sector_mask: 1,
                    requester: 1,
                    is_write: false,
                    l1_resident_after: true,
                    fill_will_reside: false,
                    fill_generation: None,
                    hit: true,
                    issue_cycle: 5,
                })
                .unwrap();
            (owner_read.service_sectors(), reader_read.service_sectors())
        }

        assert_eq!(run(true), (0, 0));
        assert_eq!(run(false), (0, 0));
    }

    #[test]
    fn fill_visibility_is_the_publication_edge_after_service_completion() {
        let mut queue = L1VisibilityQueue::new(config());
        let miss = L1AccessObservation {
            sequence: 0,
            line_id: 31,
            sector_mask: 1,
            requester: 0,
            is_write: false,
            l1_resident_after: false,
            fill_will_reside: true,
            fill_generation: Some(3),
            hit: false,
            issue_cycle: 0,
        };
        let plan = queue.preview(miss).unwrap();
        assert_eq!(plan.service_sectors(), 0);
        queue.commit(plan).unwrap();
        let writer = L1AccessObservation {
            sequence: 0,
            line_id: 31,
            sector_mask: 1,
            requester: 1,
            is_write: true,
            l1_resident_after: false,
            fill_will_reside: false,
            fill_generation: None,
            hit: false,
            issue_cycle: 1,
        };
        assert!(matches!(
            queue.preview(writer),
            Err(L1VisibilityError::PendingFillConflict { .. })
        ));
        assert!(matches!(
            queue.preview(L1AccessObservation {
                sequence: 1,
                line_id: 31,
                sector_mask: 1,
                requester: 0,
                is_write: false,
                l1_resident_after: true,
                fill_will_reside: false,
                fill_generation: None,
                hit: true,
                issue_cycle: 1,
            }),
            Err(L1VisibilityError::LinePending { .. })
        ));
        queue
            .observe_fill_visible(L1FillVisibleObservation {
                sequence: 1,
                requester: 0,
                line_id: 31,
                sector_mask: 1,
                fill_generation: 3,
                visible_cycle: 2,
            })
            .unwrap();
        let writer = L1AccessObservation {
            issue_cycle: 3,
            ..writer
        };
        let writer_plan = queue.preview(writer).unwrap();
        assert_eq!(writer_plan.service_sectors(), 1);
        assert_eq!(
            writer_plan.invalidations(),
            &[L1InvalidationTarget {
                requester: 0,
                sector_mask: 1,
            }]
        );
    }

    #[test]
    fn nonresident_read_does_not_publish_an_l1_sharer() {
        let mut queue = L1VisibilityQueue::new(config());
        let bypass = L1AccessObservation {
            sequence: 0,
            line_id: 9,
            sector_mask: 1,
            requester: 0,
            is_write: false,
            l1_resident_after: false,
            fill_will_reside: false,
            fill_generation: None,
            hit: false,
            issue_cycle: 0,
        };
        queue.commit(queue.preview(bypass).unwrap()).unwrap();
        let write = L1AccessObservation {
            sequence: 0,
            line_id: 9,
            sector_mask: 1,
            requester: 1,
            is_write: true,
            l1_resident_after: false,
            fill_will_reside: false,
            fill_generation: None,
            hit: false,
            issue_cycle: 1,
        };
        // A bypass/nonresident read is not a fill.  A remote write therefore
        // sees no phantom sharer and needs no invalidation service.
        assert_eq!(queue.preview(write).unwrap().service_sectors(), 0);
    }

    #[test]
    fn cacheable_miss_requires_matching_fill_generation_before_visibility() {
        let mut queue = L1VisibilityQueue::new(config());
        let miss = L1AccessObservation {
            sequence: 0,
            line_id: 12,
            sector_mask: 1,
            requester: 0,
            is_write: false,
            l1_resident_after: false,
            fill_will_reside: true,
            fill_generation: Some(7),
            hit: false,
            issue_cycle: 0,
        };
        queue.commit(queue.preview(miss).unwrap()).unwrap();
        let write = L1AccessObservation {
            sequence: 0,
            line_id: 12,
            sector_mask: 1,
            requester: 1,
            is_write: true,
            l1_resident_after: false,
            fill_will_reside: false,
            fill_generation: None,
            hit: false,
            issue_cycle: 1,
        };
        assert!(matches!(
            queue.preview(write),
            Err(L1VisibilityError::PendingFillConflict { .. })
        ));
        assert!(matches!(
            queue.observe_fill_visible(L1FillVisibleObservation {
                sequence: 1,
                requester: 0,
                line_id: 12,
                sector_mask: 1,
                fill_generation: 8,
                visible_cycle: 2,
            }),
            Err(L1VisibilityError::FillGenerationMismatch { .. })
        ));
        queue
            .observe_fill_visible(L1FillVisibleObservation {
                sequence: 1,
                requester: 0,
                line_id: 12,
                sector_mask: 1,
                fill_generation: 7,
                visible_cycle: 2,
            })
            .unwrap();
        let write = L1AccessObservation {
            issue_cycle: 3,
            sequence: 1,
            ..write
        };
        assert_eq!(queue.preview(write).unwrap().service_sectors(), 1);
    }

    #[test]
    fn ordered_fence_state_is_consumed_and_sequence_regressions_refuse() {
        let mut queue = L1VisibilityQueue::new(config());
        queue
            .observe_fence(L1FenceObservation {
                sequence: 0,
                requester: 0,
                pc: 16,
                issue_cycle: 2,
                scope: VisibilityScope::Device,
                order: VisibilityOrder::Release,
            })
            .unwrap();
        let state = queue.fence_state(0).unwrap();
        assert_eq!(state.release[1], Some(2));
        assert_eq!(state.acquire[1], None);
        assert!(matches!(
            queue.observe_fence(L1FenceObservation {
                sequence: 0,
                requester: 0,
                pc: 20,
                issue_cycle: 3,
                scope: VisibilityScope::Device,
                order: VisibilityOrder::Acquire,
            }),
            Err(L1VisibilityError::SequenceRegression { .. })
        ));
    }

    #[test]
    fn invalidate_all_removes_requester_from_directory_without_touching_peers() {
        let mut queue = L1VisibilityQueue::new(config());
        queue
            .commit(queue.preview(read(40, 0, 0)).unwrap())
            .unwrap();
        queue
            .commit(queue.preview(read(40, 1, 1)).unwrap())
            .unwrap();
        queue
            .observe_invalidate_all(L1InvalidateAllObservation {
                sequence: 1,
                requester: 0,
                issue_cycle: 2,
            })
            .unwrap();
        let peer_write = queue
            .preview(L1AccessObservation {
                sequence: 2,
                line_id: 40,
                sector_mask: 1,
                requester: 2,
                is_write: true,
                l1_resident_after: false,
                fill_will_reside: false,
                fill_generation: None,
                hit: false,
                issue_cycle: 3,
            })
            .unwrap();
        assert_eq!(peer_write.service_sectors(), 1);
        assert_eq!(
            peer_write.invalidations(),
            &[L1InvalidationTarget {
                requester: 1,
                sector_mask: 1,
            }]
        );
    }
}
