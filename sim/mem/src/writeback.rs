//! Transactional dirty-sector writeback and byte-service reservations.
//!
//! The native SM86 replay currently exposes exact L2 eviction masks but does
//! not expose a trace-complete writeback/coherence protocol. This module is a
//! reusable mechanism seam: it prices dirty sectors (and optional read
//! reservations) on one deterministic byte server, enforces finite dirty
//! in-flight capacity, and makes same-line fetches wait for the latest
//! submitted writeback. It deliberately does not claim ownership,
//! visibility scope, or physical SM86 completion semantics.

use crate::SectorPayloadV1;
use crate::l2::L2Eviction;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

const SECTOR_BYTES: u64 = 32;
const SECTOR_MASK: u8 = 0b1111;
static NEXT_QUEUE_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_BATCH_ID: AtomicU64 = AtomicU64::new(1);

/// A byte-server reservation request.
///
/// Use [`ServiceRequest::read`] for a normal byte reservation and
/// [`ServiceRequest::writeback`] for a dirty-sector reservation. Requests are
/// intentionally immutable; validation and token assignment happen only in
/// [`WritebackQueue::preview_services`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServiceRequest {
    bytes: u64,
    issue_cycle: u64,
    not_before_cycle: u64,
    line_id: Option<i64>,
    dirty_mask: u8,
}

impl ServiceRequest {
    /// Build a read reservation. Reads do not consume dirty writeback capacity
    /// and do not create a same-line writeback hazard.
    pub const fn read(bytes: u64, issue_cycle: u64) -> Self {
        Self {
            bytes,
            issue_cycle,
            not_before_cycle: issue_cycle,
            line_id: None,
            dirty_mask: 0,
        }
    }

    /// Build a read reservation whose fetch arrival must wait for the latest
    /// writeback of `line_id`. Reads do not consume dirty-sector capacity.
    pub const fn read_line(bytes: u64, line_id: i64, issue_cycle: u64) -> Self {
        Self::read_line_after(bytes, line_id, issue_cycle, issue_cycle)
    }

    /// Build a read reservation for `line_id` with an explicit lower bound on
    /// service start. This is useful when a caller has already preflighted a
    /// cache visibility/fill dependency; the request remains chronologically
    /// ordered by `issue_cycle`, while the byte server waits until
    /// `not_before_cycle`.
    pub const fn read_line_after(
        bytes: u64,
        line_id: i64,
        issue_cycle: u64,
        not_before_cycle: u64,
    ) -> Self {
        Self {
            bytes,
            issue_cycle,
            not_before_cycle,
            line_id: Some(line_id),
            dirty_mask: 0,
        }
    }

    /// Build a dirty-sector writeback reservation. The queue validates the
    /// four-sector mask before producing a plan.
    pub const fn writeback(line_id: i64, dirty_mask: u8, issue_cycle: u64) -> Self {
        Self::writeback_after(line_id, dirty_mask, issue_cycle, issue_cycle)
    }

    /// Build a dirty-sector writeback reservation with an explicit service
    /// lower bound. The lower bound is not a second event and does not advance
    /// the queue's chronological frontier.
    pub const fn writeback_after(
        line_id: i64,
        dirty_mask: u8,
        issue_cycle: u64,
        not_before_cycle: u64,
    ) -> Self {
        Self {
            bytes: (dirty_mask.count_ones() as u64) * SECTOR_BYTES,
            issue_cycle,
            not_before_cycle,
            line_id: Some(line_id),
            dirty_mask,
        }
    }

    pub const fn bytes(self) -> u64 {
        self.bytes
    }

    pub const fn issue_cycle(self) -> u64 {
        self.issue_cycle
    }

    pub const fn not_before_cycle(self) -> u64 {
        self.not_before_cycle
    }
}

/// A preflighted byte-server reservation.
///
/// All state-bearing fields are private so callers cannot change line identity,
/// byte count, token, or timing between cache preflight and commit. The
/// read-only accessors are sufficient to schedule a surrounding event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServicePlan {
    owner_id: u64,
    batch_id: u64,
    batch_len: u64,
    batch_index: u64,
    token: u64,
    line_id: Option<i64>,
    dirty_mask: u8,
    bytes: u64,
    issue_cycle: u64,
    not_before_cycle: u64,
    service_start_cycle: u64,
    service_done_cycle: u64,
    expected_epoch: u64,
    done_fp: u128,
}

impl ServicePlan {
    pub const fn token(self) -> u64 {
        self.token
    }

    pub const fn line_id(self) -> Option<i64> {
        self.line_id
    }

    pub const fn dirty_mask(self) -> u8 {
        self.dirty_mask
    }

    pub const fn dirty_sectors(self) -> u32 {
        self.dirty_mask.count_ones()
    }

    pub const fn is_writeback(self) -> bool {
        self.dirty_mask != 0
    }

    pub const fn bytes(self) -> u64 {
        self.bytes
    }

    pub const fn issue_cycle(self) -> u64 {
        self.issue_cycle
    }

    pub const fn not_before_cycle(self) -> u64 {
        self.not_before_cycle
    }

    pub const fn service_start_cycle(self) -> u64 {
        self.service_start_cycle
    }

    pub const fn service_done_cycle(self) -> u64 {
        self.service_done_cycle
    }
}

/// A writeback request after transactional timing/capacity preflight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WritebackPlan {
    service: ServicePlan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PayloadWritebackPlan {
    service: WritebackPlan,
    payloads: Box<[SectorPayloadV1]>,
}

impl PayloadWritebackPlan {
    pub const fn service_plan(&self) -> WritebackPlan {
        self.service
    }

    pub fn payloads(&self) -> &[SectorPayloadV1] {
        &self.payloads
    }

    pub const fn line_id(&self) -> i64 {
        self.service.line_id()
    }

    pub const fn dirty_mask(&self) -> u8 {
        self.service.dirty_mask()
    }
}

impl WritebackPlan {
    pub const fn token(self) -> u64 {
        self.service.token()
    }

    pub const fn line_id(self) -> i64 {
        match self.service.line_id() {
            Some(line_id) => line_id,
            None => 0,
        }
    }

    pub const fn dirty_mask(self) -> u8 {
        self.service.dirty_mask()
    }

    pub const fn dirty_sectors(self) -> u32 {
        self.service.dirty_sectors()
    }

    pub const fn bytes(self) -> u64 {
        self.service.bytes()
    }

    pub const fn issue_cycle(self) -> u64 {
        self.service.issue_cycle()
    }

    pub const fn service_start_cycle(self) -> u64 {
        self.service.service_start_cycle()
    }

    pub const fn service_done_cycle(self) -> u64 {
        self.service.service_done_cycle()
    }

    pub const fn not_before_cycle(self) -> u64 {
        self.service.not_before_cycle()
    }

    /// Expose the immutable service reservation for an atomic mixed read /
    /// writeback commit. The returned plan carries its private provenance and
    /// dirty-sector metadata; it cannot be forged with a struct literal.
    pub const fn service_plan(self) -> ServicePlan {
        self.service
    }
}

/// A completed writeback acknowledgement. Completion is intentionally a
/// separate operation from planning/commit so a caller can schedule it in its
/// own event queue after committing the surrounding cache transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WritebackCompletion {
    pub token: u64,
    pub line_id: i64,
    pub dirty_mask: u8,
    pub service_done_cycle: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PendingWriteback {
    plan: ServicePlan,
}

/// Deterministic, bounded shared byte-server for dirty L2 sector writebacks.
///
/// `capacity_sectors` is a queue capacity in 32-byte beats. The service rate
/// uses fixed-point `cycle * bytes_per_sec` units, matching the native DRAM
/// read observer's arithmetic without introducing floating point.
/// A queue is intentionally not `Clone`: each plan belongs to one mutable
/// queue, and copying its owner/epoch would let plans cross divergent states.
#[derive(Debug, PartialEq, Eq)]
pub struct WritebackQueue {
    queue_id: u64,
    capacity_sectors: u32,
    bytes_per_sec: u128,
    sm_clock_hz: u128,
    free_fp: u128,
    in_flight_sectors: u32,
    next_token: u64,
    epoch: u64,
    /// The greatest accepted issue/access/completion cycle. Reservations may
    /// queue behind the service frontier, but an old issue/fetch can never be
    /// admitted after the frontier has advanced.
    event_frontier_cycle: u64,
    pending: BTreeMap<u64, PendingWriteback>,
    /// Latest submitted completion per line. A later request supersedes the
    /// entry; an older acknowledgement must never erase a newer hazard.
    latest_by_line: BTreeMap<i64, (u64, u128)>,
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum WritebackError {
    #[error("writeback capacity must be positive")]
    CapacityZero,
    #[error("writeback byte rate and SM clock must be positive")]
    RateZero,
    #[error("writeback queue identity space is exhausted")]
    QueueIdExhausted,
    #[error("service reservation identity space is exhausted")]
    ReservationIdExhausted,
    #[error("writeback fixed-point arithmetic overflow")]
    ArithmeticOverflow,
    #[error("service request must reserve at least one byte")]
    RequestBytesZero,
    #[error("dirty mask {dirty_mask:#06b} is outside the four-sector line")]
    DirtyMaskInvalid { dirty_mask: u8 },
    #[error("dirty mask {dirty_mask:#06b} is not contained in valid mask {valid_mask:#06b}")]
    DirtyNotValid { dirty_mask: u8, valid_mask: u8 },
    #[error("dirty mask {dirty_mask:#06b} overlaps pending mask {pending_mask:#06b}")]
    DirtyPendingOverlap { dirty_mask: u8, pending_mask: u8 },
    #[error("writeback line identity is missing for dirty mask {dirty_mask:#06b}")]
    DirtyLineMissing { dirty_mask: u8 },
    #[error("writeback capacity {capacity} sectors exceeded by {requested} sectors")]
    CapacityExceeded { capacity: u32, requested: u32 },
    #[error("writeback transaction is stale (expected epoch {expected}, current {current})")]
    StalePlan { expected: u64, current: u64 },
    #[error("writeback plan belongs to queue {expected}, not queue {actual}")]
    ForeignPlan { expected: u64, actual: u64 },
    #[error("writeback plan does not match current queue timing state")]
    PlanMismatch,
    #[error("writeback token {0} is unknown or already completed")]
    UnknownToken(u64),
    #[error("writeback token {token} completed at {at}, before service time {ready}")]
    EarlyCompletion { token: u64, at: u64, ready: u64 },
    #[error("event time {requested} regresses queue frontier {frontier}")]
    TimeRegression { requested: u64, frontier: u64 },
    #[error("payload writeback has identity mismatch")]
    PayloadIdentityMismatch,
    #[error("payload writeback coverage {got:#06b} does not equal dirty mask {expected:#06b}")]
    PayloadCoverageMismatch { expected: u8, got: u8 },
    #[error("payload writeback has no complete payload for line {line_id} sector {sector}")]
    PayloadMissing { line_id: i64, sector: u8 },
}

impl WritebackQueue {
    pub fn new(
        capacity_sectors: u32,
        bytes_per_sec: u64,
        sm_clock_hz: u64,
    ) -> Result<Self, WritebackError> {
        if capacity_sectors == 0 {
            return Err(WritebackError::CapacityZero);
        }
        if bytes_per_sec == 0 || sm_clock_hz == 0 {
            return Err(WritebackError::RateZero);
        }
        let queue_id = allocate_id(&NEXT_QUEUE_ID).ok_or(WritebackError::QueueIdExhausted)?;
        Ok(Self {
            queue_id,
            capacity_sectors,
            bytes_per_sec: u128::from(bytes_per_sec),
            sm_clock_hz: u128::from(sm_clock_hz),
            free_fp: 0,
            in_flight_sectors: 0,
            next_token: 0,
            epoch: 0,
            event_frontier_cycle: 0,
            pending: BTreeMap::new(),
            latest_by_line: BTreeMap::new(),
        })
    }

    pub const fn capacity_sectors(&self) -> u32 {
        self.capacity_sectors
    }

    pub const fn in_flight_sectors(&self) -> u32 {
        self.in_flight_sectors
    }

    pub const fn event_frontier_cycle(&self) -> u64 {
        self.event_frontier_cycle
    }

    pub fn pending_tokens(&self) -> impl Iterator<Item = u64> + '_ {
        self.pending.keys().copied()
    }

    /// Return the earliest cycle at which a same-line fetch may be admitted.
    /// This is a hazard lower bound, not a claim about cache-tag visibility.
    /// The query is an ordered access event: it advances the event frontier and
    /// invalidates preflight plans, while leaving byte-server reservations
    /// unchanged.
    pub fn fetch_arrival(&mut self, line_id: i64, issue_cycle: u64) -> Result<u64, WritebackError> {
        self.check_event_time(issue_cycle)?;
        let next_epoch = self.next_epoch()?;
        let issue_fp = self.scale_cycle(issue_cycle)?;
        let arrival_fp = self
            .latest_by_line
            .get(&line_id)
            .map_or(issue_fp, |(_, ready)| issue_fp.max(*ready));
        let arrival = self.ceil_cycle(arrival_fp)?;
        self.event_frontier_cycle = issue_cycle;
        self.epoch = next_epoch;
        Ok(arrival)
    }

    /// Preview one ordinary read reservation. Reads share the same fixed-point
    /// byte server as dirty writebacks but do not consume dirty-sector capacity
    /// or create a writeback completion token.
    pub fn preview_read(
        &self,
        bytes: u64,
        issue_cycle: u64,
    ) -> Result<ServicePlan, WritebackError> {
        self.preview_service(bytes, issue_cycle)
    }

    /// Generic alias for [`Self::preview_read`].
    pub fn preview_service(
        &self,
        bytes: u64,
        issue_cycle: u64,
    ) -> Result<ServicePlan, WritebackError> {
        let requests = [ServiceRequest::read(bytes, issue_cycle)];
        let mut plans = self.preview_services(&requests)?;
        Ok(plans.remove(0))
    }

    /// Preview several reads/writebacks as one ordered reservation transaction.
    /// Token and epoch exhaustion are checked here, before a caller commits its
    /// external cache mutation. The returned plans can be atomically committed
    /// with [`Self::commit_services`].
    pub fn preview_services(
        &self,
        requests: &[ServiceRequest],
    ) -> Result<Vec<ServicePlan>, WritebackError> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        let request_count =
            u64::try_from(requests.len()).map_err(|_| WritebackError::ArithmeticOverflow)?;
        self.next_token
            .checked_add(request_count)
            .ok_or(WritebackError::ArithmeticOverflow)?;
        self.epoch
            .checked_add(1)
            .ok_or(WritebackError::ArithmeticOverflow)?;
        let batch_id = allocate_id(&NEXT_BATCH_ID).ok_or(WritebackError::ReservationIdExhausted)?;
        self.preview_services_from_state(requests, batch_id)
    }

    /// Preflight a dirty eviction without mutating queue state. The caller
    /// may perform this before a cache transaction and call [`Self::commit`]
    /// immediately after that transaction commits.
    pub fn preview(
        &self,
        eviction: L2Eviction,
        issue_cycle: u64,
    ) -> Result<Option<WritebackPlan>, WritebackError> {
        validate_eviction(eviction)?;
        self.check_event_time(issue_cycle)?;
        if eviction.dirty_mask == 0 {
            return Ok(None);
        }
        let requests = [ServiceRequest::writeback(
            eviction.line_id,
            eviction.dirty_mask,
            issue_cycle,
        )];
        let plan = self
            .preview_services(&requests)?
            .into_iter()
            .next()
            .ok_or(WritebackError::PlanMismatch)?;
        Ok(Some(WritebackPlan { service: plan }))
    }

    /// Preview a value-bearing writeback. Every dirty sector must have one
    /// complete payload with the same line identity and a matching sector
    /// bit; the queue itself remains mask/timing compatible with legacy
    /// callers until this plan is explicitly used by a payload-aware owner.
    pub fn preview_payload(
        &self,
        eviction: L2Eviction,
        issue_cycle: u64,
        payloads: &[SectorPayloadV1],
    ) -> Result<PayloadWritebackPlan, WritebackError> {
        let service =
            self.preview(eviction, issue_cycle)?
                .ok_or(WritebackError::DirtyLineMissing {
                    dirty_mask: eviction.dirty_mask,
                })?;
        let mut covered = 0_u8;
        for payload in payloads {
            if payload.line_id() != eviction.line_id
                || payload.sector() >= 4
                || covered & (1_u8 << payload.sector()) != 0
                || payloads.first().is_some_and(|first| {
                    first.allocation_id() != payload.allocation_id()
                        || first.allocation_generation() != payload.allocation_generation()
                })
            {
                return Err(WritebackError::PayloadIdentityMismatch);
            }
            if !payload.is_fully_known() {
                return Err(WritebackError::PayloadMissing {
                    line_id: eviction.line_id,
                    sector: payload.sector(),
                });
            }
            covered |= 1_u8 << payload.sector();
        }
        if covered != eviction.dirty_mask {
            return Err(WritebackError::PayloadCoverageMismatch {
                expected: eviction.dirty_mask,
                got: covered,
            });
        }
        Ok(PayloadWritebackPlan {
            service,
            payloads: payloads.to_vec().into_boxed_slice(),
        })
    }

    /// Commit several preflighted read/writeback reservations transactionally.
    /// Every plan is validated against a virtual copy of the current queue; no
    /// state is changed until all tokens, epochs, timing, masks, capacity, and
    /// overflow checks have passed.
    pub fn commit_services(&mut self, plans: &[ServicePlan]) -> Result<(), WritebackError> {
        if plans.is_empty() {
            return Ok(());
        }
        let request_count =
            u64::try_from(plans.len()).map_err(|_| WritebackError::ArithmeticOverflow)?;
        let batch_id = plans[0].batch_id;
        for (index, plan) in plans.iter().enumerate() {
            if plan.owner_id != self.queue_id {
                return Err(WritebackError::ForeignPlan {
                    expected: self.queue_id,
                    actual: plan.owner_id,
                });
            }
            if plan.expected_epoch != self.epoch {
                return Err(WritebackError::StalePlan {
                    expected: plan.expected_epoch,
                    current: self.epoch,
                });
            }
            if plan.batch_id != batch_id
                || plan.batch_len != request_count
                || plan.batch_index
                    != u64::try_from(index).map_err(|_| WritebackError::ArithmeticOverflow)?
            {
                return Err(WritebackError::PlanMismatch);
            }
        }
        let next_token = self
            .next_token
            .checked_add(request_count)
            .ok_or(WritebackError::ArithmeticOverflow)?;
        let next_epoch = self.next_epoch()?;
        let requests = plans
            .iter()
            .map(|plan| ServiceRequest {
                bytes: plan.bytes,
                issue_cycle: plan.issue_cycle,
                not_before_cycle: plan.not_before_cycle,
                line_id: plan.line_id,
                dirty_mask: plan.dirty_mask,
            })
            .collect::<Vec<_>>();
        let expected = self.preview_services_from_state(&requests, batch_id)?;
        if expected.len() != plans.len()
            || expected
                .iter()
                .zip(plans)
                .any(|(expected, actual)| expected != actual)
        {
            return Err(WritebackError::PlanMismatch);
        }

        let final_plan = plans.last().ok_or(WritebackError::PlanMismatch)?;
        let dirty_sectors = plans.iter().try_fold(0_u32, |total, plan| {
            total
                .checked_add(plan.dirty_sectors())
                .ok_or(WritebackError::ArithmeticOverflow)
        })?;
        let next_in_flight_sectors = self
            .in_flight_sectors
            .checked_add(dirty_sectors)
            .ok_or(WritebackError::ArithmeticOverflow)?;
        // Derive infallible insertion records before changing queue state.
        let writebacks = plans
            .iter()
            .filter(|plan| plan.is_writeback())
            .map(|plan| {
                let line_id = plan.line_id.ok_or(WritebackError::DirtyLineMissing {
                    dirty_mask: plan.dirty_mask,
                })?;
                if self.pending.contains_key(&plan.token) {
                    return Err(WritebackError::PlanMismatch);
                }
                Ok((line_id, *plan))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // All fallible checks are complete. Allocation failure follows Rust's
        // allocator abort behavior, not a recoverable partial-commit error.
        self.free_fp = final_plan.done_fp;
        self.in_flight_sectors = next_in_flight_sectors;
        for (line_id, plan) in writebacks {
            self.latest_by_line
                .insert(line_id, (plan.token, plan.done_fp));
            self.pending.insert(plan.token, PendingWriteback { plan });
        }
        self.next_token = next_token;
        self.event_frontier_cycle = plans
            .iter()
            .map(|plan| plan.issue_cycle)
            .max()
            .unwrap_or(self.event_frontier_cycle);
        self.epoch = next_epoch;
        Ok(())
    }

    /// Commit one generic service reservation.
    pub fn commit_service(&mut self, plan: ServicePlan) -> Result<(), WritebackError> {
        self.commit_services(&[plan])
    }

    /// Commit a writeback plan produced by [`Self::preview`].
    pub fn commit(&mut self, plan: WritebackPlan) -> Result<(), WritebackError> {
        self.commit_service(plan.service)
    }

    /// Convenience transaction for callers that do not need to coordinate a
    /// separate cache commit. Cache integrations should use `preview` and
    /// `commit` explicitly so a failed capacity check cannot mutate L2 state.
    pub fn submit(
        &mut self,
        eviction: L2Eviction,
        issue_cycle: u64,
    ) -> Result<Option<WritebackPlan>, WritebackError> {
        let Some(plan) = self.preview(eviction, issue_cycle)? else {
            return Ok(None);
        };
        self.commit(plan)?;
        Ok(Some(plan))
    }

    /// Apply a scheduled completion acknowledgement. The latest same-line
    /// hazard is removed only when its own token completes. Tokens may be
    /// acknowledged out of service order, but all acknowledgements must carry
    /// the current or a later event cycle; retroactive completion is rejected.
    pub fn complete(
        &mut self,
        token: u64,
        at_cycle: u64,
    ) -> Result<WritebackCompletion, WritebackError> {
        let pending = *self
            .pending
            .get(&token)
            .ok_or(WritebackError::UnknownToken(token))?;
        if at_cycle < pending.plan.service_done_cycle {
            return Err(WritebackError::EarlyCompletion {
                token,
                at: at_cycle,
                ready: pending.plan.service_done_cycle,
            });
        }
        self.check_event_time(at_cycle)?;
        let next_epoch = self.next_epoch()?;
        let dirty_sectors = pending.plan.dirty_sectors();
        let next_in_flight = self
            .in_flight_sectors
            .checked_sub(dirty_sectors)
            .ok_or(WritebackError::ArithmeticOverflow)?;
        let line_id = pending
            .plan
            .line_id
            .ok_or(WritebackError::DirtyLineMissing {
                dirty_mask: pending.plan.dirty_mask,
            })?;
        let remove_latest = self
            .latest_by_line
            .get(&line_id)
            .is_some_and(|(latest, _)| *latest == token);

        // All fallible checks are complete before mutation.
        self.pending.remove(&token);
        self.in_flight_sectors = next_in_flight;
        if remove_latest {
            self.latest_by_line.remove(&line_id);
        }
        self.event_frontier_cycle = at_cycle;
        self.epoch = next_epoch;
        Ok(WritebackCompletion {
            token,
            line_id,
            dirty_mask: pending.plan.dirty_mask,
            service_done_cycle: pending.plan.service_done_cycle,
        })
    }

    fn preview_services_from_state(
        &self,
        requests: &[ServiceRequest],
        batch_id: u64,
    ) -> Result<Vec<ServicePlan>, WritebackError> {
        let mut free_fp = self.free_fp;
        let mut in_flight_sectors = self.in_flight_sectors;
        let mut event_frontier_cycle = self.event_frontier_cycle;
        // Only dirty lines changed by this batch need virtual state. Avoid an
        // O(number of pending lines) clone for every individual read request.
        let mut batch_latest_by_line = BTreeMap::<i64, u128>::new();
        let batch_len =
            u64::try_from(requests.len()).map_err(|_| WritebackError::ArithmeticOverflow)?;
        let mut plans = Vec::with_capacity(requests.len());
        for (index, request) in requests.iter().copied().enumerate() {
            validate_request(request)?;
            if request.issue_cycle < event_frontier_cycle {
                return Err(WritebackError::TimeRegression {
                    requested: request.issue_cycle,
                    frontier: event_frontier_cycle,
                });
            }
            let dirty_sectors = request.dirty_mask.count_ones();
            let requested_capacity = in_flight_sectors
                .checked_add(dirty_sectors)
                .ok_or(WritebackError::ArithmeticOverflow)?;
            if requested_capacity > self.capacity_sectors {
                return Err(WritebackError::CapacityExceeded {
                    capacity: self.capacity_sectors,
                    requested: requested_capacity,
                });
            }
            let issue_fp = self.scale_cycle(request.issue_cycle)?;
            let not_before_fp = self.scale_cycle(request.not_before_cycle)?;
            let line_ready_fp = request
                .line_id
                .and_then(|line_id| {
                    batch_latest_by_line
                        .get(&line_id)
                        .copied()
                        .or_else(|| self.latest_by_line.get(&line_id).map(|(_, ready)| *ready))
                })
                .unwrap_or(0);
            let service_start_fp = free_fp.max(issue_fp).max(not_before_fp).max(line_ready_fp);
            let service_fp = u128::from(request.bytes)
                .checked_mul(self.sm_clock_hz)
                .ok_or(WritebackError::ArithmeticOverflow)?;
            let done_fp = service_start_fp
                .checked_add(service_fp)
                .ok_or(WritebackError::ArithmeticOverflow)?;
            let service_start_cycle = self.ceil_cycle(service_start_fp)?;
            let service_done_cycle = self.ceil_cycle(done_fp)?;
            let offset = u64::try_from(index).map_err(|_| WritebackError::ArithmeticOverflow)?;
            let token = self
                .next_token
                .checked_add(offset)
                .ok_or(WritebackError::ArithmeticOverflow)?;
            let plan = ServicePlan {
                owner_id: self.queue_id,
                batch_id,
                batch_len,
                batch_index: offset,
                token,
                line_id: request.line_id,
                dirty_mask: request.dirty_mask,
                bytes: request.bytes,
                issue_cycle: request.issue_cycle,
                not_before_cycle: request.not_before_cycle,
                service_start_cycle,
                service_done_cycle,
                expected_epoch: self.epoch,
                done_fp,
            };
            plans.push(plan);
            free_fp = done_fp;
            in_flight_sectors = requested_capacity;
            if request.dirty_mask != 0 {
                match request.line_id {
                    Some(line_id) => {
                        batch_latest_by_line.insert(line_id, done_fp);
                    }
                    None => unreachable!("dirty requests are validated with a line identity"),
                }
            }
            event_frontier_cycle = request.issue_cycle;
        }
        Ok(plans)
    }

    fn check_event_time(&self, cycle: u64) -> Result<(), WritebackError> {
        if cycle < self.event_frontier_cycle {
            return Err(WritebackError::TimeRegression {
                requested: cycle,
                frontier: self.event_frontier_cycle,
            });
        }
        Ok(())
    }

    fn next_epoch(&self) -> Result<u64, WritebackError> {
        self.epoch
            .checked_add(1)
            .ok_or(WritebackError::ArithmeticOverflow)
    }

    fn scale_cycle(&self, cycle: u64) -> Result<u128, WritebackError> {
        u128::from(cycle)
            .checked_mul(self.bytes_per_sec)
            .ok_or(WritebackError::ArithmeticOverflow)
    }

    fn ceil_cycle(&self, fixed: u128) -> Result<u64, WritebackError> {
        let cycle = fixed.div_ceil(self.bytes_per_sec);
        u64::try_from(cycle).map_err(|_| WritebackError::ArithmeticOverflow)
    }
}

fn allocate_id(counter: &AtomicU64) -> Option<u64> {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .ok()
}

fn validate_request(request: ServiceRequest) -> Result<(), WritebackError> {
    if request.bytes == 0 {
        return Err(WritebackError::RequestBytesZero);
    }
    if request.dirty_mask & !SECTOR_MASK != 0 {
        return Err(WritebackError::DirtyMaskInvalid {
            dirty_mask: request.dirty_mask,
        });
    }
    if request.dirty_mask != 0 && request.line_id.is_none() {
        return Err(WritebackError::DirtyLineMissing {
            dirty_mask: request.dirty_mask,
        });
    }
    if request.dirty_mask != 0
        && request.bytes != u64::from(request.dirty_mask.count_ones()) * SECTOR_BYTES
    {
        return Err(WritebackError::PlanMismatch);
    }
    Ok(())
}

fn validate_eviction(eviction: L2Eviction) -> Result<(), WritebackError> {
    if eviction.dirty_mask & !SECTOR_MASK != 0 {
        return Err(WritebackError::DirtyMaskInvalid {
            dirty_mask: eviction.dirty_mask,
        });
    }
    if eviction.valid_mask & !SECTOR_MASK != 0 || eviction.pending_mask & !SECTOR_MASK != 0 {
        return Err(WritebackError::DirtyMaskInvalid {
            dirty_mask: eviction.valid_mask | eviction.pending_mask,
        });
    }
    if eviction.dirty_mask & !eviction.valid_mask != 0 {
        return Err(WritebackError::DirtyNotValid {
            dirty_mask: eviction.dirty_mask,
            valid_mask: eviction.valid_mask,
        });
    }
    if eviction.dirty_mask & eviction.pending_mask != 0 {
        return Err(WritebackError::DirtyPendingOverlap {
            dirty_mask: eviction.dirty_mask,
            pending_mask: eviction.pending_mask,
        });
    }
    Ok(())
}
