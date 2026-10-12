//! Bounded trace-observable coherence/visibility service.
//!
//! This is deliberately a directory-style mechanism seam, not an NVIDIA
//! protocol reconstruction. It consumes ordered `(SM, line, sector, read/write)`
//! accesses that are already present in a complete native L2 event stream.
//! Read/read sharing is free; a read that follows another SM's write, or a
//! write that invalidates another SM's readers/writer, consumes one configured
//! service cycle per affected sector. Unknown producer identity must be
//! rejected by the caller before this queue is enabled. The directory starts
//! cold and has no API for importing resident L1/L2 ownership; callers must
//! therefore treat a trace with hidden initial state as outside this seam.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

const SECTOR_MASK: u8 = 0b1111;
const MAX_DIRECTORY_LINES: usize = 1_000_000;
static NEXT_QUEUE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct SectorState {
    owner: Option<u8>,
    sharers: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct LineState {
    sectors: [SectorState; 4],
}

/// Explicit opt-in coherence service parameters. Values are hypotheses for a
/// bounded model and are not architecture-card measurements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoherenceConfig {
    service_cycles_per_sector: u64,
    capacity_sectors: u32,
    /// Enables the separate L1 observation/invalidation event path.  This is
    /// deliberately part of the typed replay configuration rather than an
    /// environment variable so a result can be reproduced from its inputs.
    l1_visibility_protocol: bool,
    /// Enables resident-hit L1 dirty tracking for the explicit terminal
    /// writeback experiment. It never implies write-allocate on a cold store.
    l1_resident_writeback: bool,
}

impl CoherenceConfig {
    pub fn new(
        service_cycles_per_sector: u64,
        capacity_sectors: u32,
    ) -> Result<Self, CoherenceError> {
        if service_cycles_per_sector == 0 {
            return Err(CoherenceError::ServiceCyclesZero);
        }
        if capacity_sectors == 0 {
            return Err(CoherenceError::CapacityZero);
        }
        Ok(Self {
            service_cycles_per_sector,
            capacity_sectors,
            l1_visibility_protocol: false,
            l1_resident_writeback: false,
        })
    }

    /// Enable the bounded L1 visibility experiment for this launch.  The
    /// directory timing values remain model hypotheses; this flag only selects
    /// the explicit observation/invalidation event path.
    pub const fn with_l1_visibility_protocol(self, enabled: bool) -> Self {
        Self {
            l1_visibility_protocol: enabled,
            ..self
        }
    }

    pub const fn service_cycles_per_sector(self) -> u64 {
        self.service_cycles_per_sector
    }

    pub const fn capacity_sectors(self) -> u32 {
        self.capacity_sectors
    }

    pub const fn l1_visibility_protocol(self) -> bool {
        self.l1_visibility_protocol
    }

    /// Enable resident-hit L1 dirty tracking. The lower writeback queue is
    /// still required by the replay configuration; this flag only selects the
    /// cache-state policy.
    pub const fn with_l1_resident_writeback(self, enabled: bool) -> Self {
        Self {
            l1_resident_writeback: enabled,
            ..self
        }
    }

    pub const fn l1_resident_writeback(self) -> bool {
        self.l1_resident_writeback
    }
}

/// Ordered L2 access used by the directory. `sector_mask` is line-relative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoherenceRequest {
    pub line_id: i64,
    pub sector_mask: u8,
    pub requester: u8,
    pub is_write: bool,
    pub issue_cycle: u64,
}

/// A preflighted directory transition. Private state prevents changing the
/// requester, transition count or timing between L2 and directory commits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoherencePlan {
    owner_id: u64,
    token: u64,
    request: CoherenceRequest,
    service_sectors: u32,
    service_start_cycle: u64,
    service_done_cycle: u64,
    expected_epoch: u64,
    next_line: LineState,
}

impl CoherencePlan {
    pub const fn token(self) -> u64 {
        self.token
    }

    pub const fn request_line_id(self) -> i64 {
        self.request.line_id
    }

    pub const fn request_sector_mask(self) -> u8 {
        self.request.sector_mask
    }

    pub const fn service_sectors(self) -> u32 {
        self.service_sectors
    }

    pub const fn service_start_cycle(self) -> u64 {
        self.service_start_cycle
    }

    pub const fn service_done_cycle(self) -> u64 {
        self.service_done_cycle
    }

    pub const fn needs_completion(self) -> bool {
        self.service_sectors != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoherenceCompletion {
    pub token: u64,
    pub line_id: i64,
    pub sector_mask: u8,
    pub service_sectors: u32,
    pub service_done_cycle: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PendingCompletion {
    plan: CoherencePlan,
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum CoherenceError {
    #[error("coherence service cycles must be positive")]
    ServiceCyclesZero,
    #[error("coherence capacity must be positive")]
    CapacityZero,
    #[error("coherence requester {0} is outside the 0..128 domain")]
    RequesterOutOfRange(u8),
    #[error("coherence sector mask {0:#06b} is invalid")]
    SectorMaskInvalid(u8),
    #[error("coherence service capacity {capacity} exceeded by {requested} sectors")]
    CapacityExceeded { capacity: u32, requested: u32 },
    #[error("coherence directory line limit {0} reached")]
    DirectoryLineLimit(usize),
    #[error("coherence request time {requested} regresses event frontier {frontier}")]
    TimeRegression { requested: u64, frontier: u64 },
    #[error("coherence transaction is stale (expected epoch {expected}, current {current})")]
    StalePlan { expected: u64, current: u64 },
    #[error(
        "coherence line {line_id} has a pending visibility transfer through cycle {ready_cycle}"
    )]
    LineVisibilityPending { line_id: i64, ready_cycle: u64 },
    #[error("coherence token {0} is unknown or already completed")]
    UnknownToken(u64),
    #[error("coherence token {token} completed at {at}, before service time {ready}")]
    EarlyCompletion { token: u64, at: u64, ready: u64 },
    #[error("coherence arithmetic overflow")]
    ArithmeticOverflow,
}

/// Bounded directory and serialized coherence-service queue.
#[derive(Debug, PartialEq, Eq)]
pub struct CoherenceQueue {
    owner_id: u64,
    config: CoherenceConfig,
    free_cycle: u64,
    event_frontier: u64,
    in_flight_sectors: u32,
    next_token: u64,
    epoch: u64,
    lines: BTreeMap<i64, LineState>,
    pending: BTreeMap<u64, PendingCompletion>,
}

impl CoherenceQueue {
    pub fn new(config: CoherenceConfig) -> Self {
        let owner_id = NEXT_QUEUE_ID.fetch_add(1, Ordering::Relaxed);
        Self {
            owner_id,
            config,
            free_cycle: 0,
            event_frontier: 0,
            in_flight_sectors: 0,
            next_token: 0,
            epoch: 0,
            lines: BTreeMap::new(),
            pending: BTreeMap::new(),
        }
    }

    pub const fn config(&self) -> CoherenceConfig {
        self.config
    }

    pub const fn in_flight_sectors(&self) -> u32 {
        self.in_flight_sectors
    }

    pub fn pending_tokens(&self) -> impl Iterator<Item = u64> + '_ {
        self.pending.keys().copied()
    }

    pub fn preview(&self, request: CoherenceRequest) -> Result<CoherencePlan, CoherenceError> {
        validate_request(request)?;
        if request.issue_cycle < self.event_frontier {
            return Err(CoherenceError::TimeRegression {
                requested: request.issue_cycle,
                frontier: self.event_frontier,
            });
        }
        if let Some(pending) = self
            .pending
            .values()
            .find(|pending| pending.plan.request.line_id == request.line_id)
        {
            // The directory deliberately refuses to expose an admission-time
            // reservation as if it were already visible.  The completion event
            // must retire first, at which point `complete` publishes the
            // transition into `lines`.
            return Err(CoherenceError::LineVisibilityPending {
                line_id: request.line_id,
                ready_cycle: pending.plan.service_done_cycle,
            });
        }
        let line = self
            .lines
            .get(&request.line_id)
            .copied()
            .unwrap_or_default();
        if !self.lines.contains_key(&request.line_id) && self.lines.len() >= MAX_DIRECTORY_LINES {
            return Err(CoherenceError::DirectoryLineLimit(MAX_DIRECTORY_LINES));
        }
        let mut next_line = line;
        let requester_bit = 1_u128 << request.requester;
        let mut service_sectors = 0_u32;
        for sector in 0..4 {
            if request.sector_mask & (1 << sector) == 0 {
                continue;
            }
            let state = &mut next_line.sectors[sector as usize];
            if request.is_write {
                let remote_owner = state.owner.is_some_and(|owner| owner != request.requester);
                let remote_readers = (state.sharers & !requester_bit).count_ones();
                if remote_owner || remote_readers != 0 {
                    service_sectors = service_sectors
                        .checked_add(1)
                        .ok_or(CoherenceError::ArithmeticOverflow)?;
                }
                *state = SectorState {
                    owner: Some(request.requester),
                    sharers: 0,
                };
            } else {
                if state.owner.is_some_and(|owner| owner != request.requester) {
                    service_sectors = service_sectors
                        .checked_add(1)
                        .ok_or(CoherenceError::ArithmeticOverflow)?;
                    let owner_bit = 1_u128 << state.owner.expect("owner was checked");
                    state.owner = None;
                    state.sharers |= owner_bit;
                }
                state.sharers |= requester_bit;
            }
        }
        let requested_capacity = self
            .in_flight_sectors
            .checked_add(service_sectors)
            .ok_or(CoherenceError::ArithmeticOverflow)?;
        if requested_capacity > self.config.capacity_sectors {
            return Err(CoherenceError::CapacityExceeded {
                capacity: self.config.capacity_sectors,
                requested: requested_capacity,
            });
        }
        // A zero-service request is immediately visible at its issue cycle.
        // It must not inherit the serialized service lane's free cycle, or a
        // read/read access to an unrelated line would create global
        // head-of-line delay in response timing.
        let service_start_cycle = if service_sectors == 0 {
            request.issue_cycle
        } else {
            self.free_cycle.max(request.issue_cycle)
        };
        let service_duration = u64::from(service_sectors)
            .checked_mul(self.config.service_cycles_per_sector)
            .ok_or(CoherenceError::ArithmeticOverflow)?;
        let service_done_cycle = service_start_cycle
            .checked_add(service_duration)
            .ok_or(CoherenceError::ArithmeticOverflow)?;
        Ok(CoherencePlan {
            owner_id: self.owner_id,
            token: self.next_token,
            request,
            service_sectors,
            service_start_cycle,
            service_done_cycle,
            expected_epoch: self.epoch,
            next_line,
        })
    }

    pub fn commit(&mut self, plan: CoherencePlan) -> Result<(), CoherenceError> {
        if plan.owner_id != self.owner_id {
            return Err(CoherenceError::StalePlan {
                expected: self.epoch,
                current: self.epoch,
            });
        }
        if plan.expected_epoch != self.epoch {
            return Err(CoherenceError::StalePlan {
                expected: plan.expected_epoch,
                current: self.epoch,
            });
        }
        if plan.token != self.next_token {
            return Err(CoherenceError::StalePlan {
                expected: plan.expected_epoch,
                current: self.epoch,
            });
        }
        let expected = self.preview(plan.request)?;
        if expected != plan {
            return Err(CoherenceError::StalePlan {
                expected: plan.expected_epoch,
                current: self.epoch,
            });
        }
        let next_token = self
            .next_token
            .checked_add(1)
            .ok_or(CoherenceError::ArithmeticOverflow)?;
        let next_epoch = self
            .epoch
            .checked_add(1)
            .ok_or(CoherenceError::ArithmeticOverflow)?;
        if plan.needs_completion() {
            // Keep the old visible state until the scheduled completion.  The
            // plan's `next_line` is committed by `complete`, so a later access
            // cannot observe ownership before the transfer is visible.
            self.free_cycle = plan.service_done_cycle;
        } else {
            self.lines.insert(plan.request.line_id, plan.next_line);
        }
        self.event_frontier = plan.request.issue_cycle;
        self.in_flight_sectors = self
            .in_flight_sectors
            .checked_add(plan.service_sectors)
            .ok_or(CoherenceError::ArithmeticOverflow)?;
        if plan.needs_completion() {
            self.pending.insert(plan.token, PendingCompletion { plan });
        }
        self.next_token = next_token;
        self.epoch = next_epoch;
        Ok(())
    }

    pub fn complete(
        &mut self,
        token: u64,
        at_cycle: u64,
    ) -> Result<CoherenceCompletion, CoherenceError> {
        let pending = *self
            .pending
            .get(&token)
            .ok_or(CoherenceError::UnknownToken(token))?;
        if at_cycle < pending.plan.service_done_cycle {
            return Err(CoherenceError::EarlyCompletion {
                token,
                at: at_cycle,
                ready: pending.plan.service_done_cycle,
            });
        }
        if at_cycle < self.event_frontier {
            return Err(CoherenceError::TimeRegression {
                requested: at_cycle,
                frontier: self.event_frontier,
            });
        }
        let next_epoch = self
            .epoch
            .checked_add(1)
            .ok_or(CoherenceError::ArithmeticOverflow)?;
        self.pending.remove(&token);
        self.lines
            .insert(pending.plan.request.line_id, pending.plan.next_line);
        self.in_flight_sectors = self
            .in_flight_sectors
            .checked_sub(pending.plan.service_sectors)
            .ok_or(CoherenceError::ArithmeticOverflow)?;
        self.event_frontier = self.event_frontier.max(at_cycle);
        self.epoch = next_epoch;
        Ok(CoherenceCompletion {
            token,
            line_id: pending.plan.request.line_id,
            sector_mask: pending.plan.request.sector_mask,
            service_sectors: pending.plan.service_sectors,
            service_done_cycle: pending.plan.service_done_cycle,
        })
    }
}

fn validate_request(request: CoherenceRequest) -> Result<(), CoherenceError> {
    if request.requester >= 128 {
        return Err(CoherenceError::RequesterOutOfRange(request.requester));
    }
    if request.sector_mask == 0 || request.sector_mask & !SECTOR_MASK != 0 {
        return Err(CoherenceError::SectorMaskInvalid(request.sector_mask));
    }
    Ok(())
}
