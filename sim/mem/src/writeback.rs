//! Transactional dirty-sector writeback service for bounded memory studies.
//!
//! The native SM86 replay currently exposes exact L2 eviction masks but does
//! not expose a trace-complete writeback/coherence protocol.  This module is a
//! reusable mechanism seam: it prices dirty sectors on a deterministic shared
//! byte server, enforces finite in-flight capacity, and makes same-line
//! fetches wait for the latest submitted writeback.  It deliberately does not
//! claim ownership, visibility scope, or physical SM86 completion semantics.

use crate::l2::L2Eviction;
use std::collections::BTreeMap;

const SECTOR_BYTES: u64 = 32;
const SECTOR_MASK: u8 = 0b1111;

/// A writeback request after transactional timing/capacity preflight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WritebackPlan {
    pub token: u64,
    pub line_id: i64,
    pub dirty_mask: u8,
    pub bytes: u64,
    pub issue_cycle: u64,
    pub service_start_cycle: u64,
    pub service_done_cycle: u64,
    expected_epoch: u64,
    done_fp: u128,
}

impl WritebackPlan {
    pub const fn dirty_sectors(self) -> u32 {
        self.dirty_mask.count_ones()
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
    plan: WritebackPlan,
}

/// Deterministic, bounded shared byte-server for dirty L2 sector writebacks.
///
/// `capacity_sectors` is a queue capacity in 32-byte beats.  The service rate
/// uses fixed-point `cycle * bytes_per_sec` units, matching the native DRAM
/// read observer's arithmetic without introducing floating point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WritebackQueue {
    capacity_sectors: u32,
    bytes_per_sec: u128,
    sm_clock_hz: u128,
    free_fp: u128,
    in_flight_sectors: u32,
    next_token: u64,
    epoch: u64,
    pending: BTreeMap<u64, PendingWriteback>,
    /// Latest submitted completion per line.  A later request supersedes the
    /// entry; an older acknowledgement must never erase a newer hazard.
    latest_by_line: BTreeMap<i64, (u64, u128)>,
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum WritebackError {
    #[error("writeback capacity must be positive")]
    CapacityZero,
    #[error("writeback byte rate and SM clock must be positive")]
    RateZero,
    #[error("writeback fixed-point arithmetic overflow")]
    ArithmeticOverflow,
    #[error("dirty mask {dirty_mask:#06b} is outside the four-sector line")]
    DirtyMaskInvalid { dirty_mask: u8 },
    #[error("dirty mask {dirty_mask:#06b} is not contained in valid mask {valid_mask:#06b}")]
    DirtyNotValid { dirty_mask: u8, valid_mask: u8 },
    #[error("dirty mask {dirty_mask:#06b} overlaps pending mask {pending_mask:#06b}")]
    DirtyPendingOverlap { dirty_mask: u8, pending_mask: u8 },
    #[error("writeback queue capacity {capacity} sectors exceeded by {requested} sectors")]
    CapacityExceeded { capacity: u32, requested: u32 },
    #[error("writeback transaction is stale (expected epoch {expected}, current {current})")]
    StalePlan { expected: u64, current: u64 },
    #[error("writeback plan does not match current queue timing state")]
    PlanMismatch,
    #[error("writeback token {0} is unknown or already completed")]
    UnknownToken(u64),
    #[error("writeback token {token} completed at {at}, before service time {ready}")]
    EarlyCompletion { token: u64, at: u64, ready: u64 },
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
        Ok(Self {
            capacity_sectors,
            bytes_per_sec: u128::from(bytes_per_sec),
            sm_clock_hz: u128::from(sm_clock_hz),
            free_fp: 0,
            in_flight_sectors: 0,
            next_token: 0,
            epoch: 0,
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

    pub fn pending_tokens(&self) -> impl Iterator<Item = u64> + '_ {
        self.pending.keys().copied()
    }

    /// Return the earliest cycle at which a same-line fetch may be admitted.
    /// This is a hazard lower bound, not a claim about cache-tag visibility.
    pub fn fetch_arrival(&self, line_id: i64, issue_cycle: u64) -> Result<u64, WritebackError> {
        let issue_fp = self.scale_cycle(issue_cycle)?;
        let arrival_fp = self
            .latest_by_line
            .get(&line_id)
            .map_or(issue_fp, |(_, ready)| issue_fp.max(*ready));
        self.ceil_cycle(arrival_fp)
    }

    /// Preflight a dirty eviction without mutating queue state.  The caller
    /// may perform this before a cache transaction and call [`Self::commit`]
    /// immediately after that transaction commits.
    pub fn preview(
        &self,
        eviction: L2Eviction,
        issue_cycle: u64,
    ) -> Result<Option<WritebackPlan>, WritebackError> {
        validate_eviction(eviction)?;
        let dirty_sectors = eviction.dirty_mask.count_ones();
        if dirty_sectors == 0 {
            return Ok(None);
        }
        let requested = self
            .in_flight_sectors
            .checked_add(dirty_sectors)
            .ok_or(WritebackError::ArithmeticOverflow)?;
        if requested > self.capacity_sectors {
            return Err(WritebackError::CapacityExceeded {
                capacity: self.capacity_sectors,
                requested,
            });
        }
        let bytes = u64::from(dirty_sectors)
            .checked_mul(SECTOR_BYTES)
            .ok_or(WritebackError::ArithmeticOverflow)?;
        let issue_fp = self.scale_cycle(issue_cycle)?;
        let line_ready_fp = self
            .latest_by_line
            .get(&eviction.line_id)
            .map_or(0, |(_, ready)| *ready);
        let service_start_fp = self.free_fp.max(issue_fp).max(line_ready_fp);
        let service_fp = u128::from(bytes)
            .checked_mul(self.sm_clock_hz)
            .ok_or(WritebackError::ArithmeticOverflow)?;
        let done_fp = service_start_fp
            .checked_add(service_fp)
            .ok_or(WritebackError::ArithmeticOverflow)?;
        let service_start_cycle = self.ceil_cycle(service_start_fp)?;
        let service_done_cycle = self.ceil_cycle(done_fp)?;
        let token = self.next_token;
        let expected_epoch = self.epoch;
        Ok(Some(WritebackPlan {
            token,
            line_id: eviction.line_id,
            dirty_mask: eviction.dirty_mask,
            bytes,
            issue_cycle,
            service_start_cycle,
            service_done_cycle,
            expected_epoch,
            done_fp,
        }))
    }

    /// Commit a plan produced by [`Self::preview`].  Epoch checking makes a
    /// plan unusable after another queue mutation, preventing capacity and
    /// same-line reservations from being silently double-counted.
    pub fn commit(&mut self, plan: WritebackPlan) -> Result<(), WritebackError> {
        if plan.expected_epoch != self.epoch {
            return Err(WritebackError::StalePlan {
                expected: plan.expected_epoch,
                current: self.epoch,
            });
        }
        if self.pending.contains_key(&plan.token) || plan.token != self.next_token {
            return Err(WritebackError::StalePlan {
                expected: plan.expected_epoch,
                current: self.epoch,
            });
        }
        let expected = self.preview(
            L2Eviction {
                line_id: plan.line_id,
                valid_mask: plan.dirty_mask,
                dirty_mask: plan.dirty_mask,
                pending_mask: 0,
            },
            plan.issue_cycle,
        )?;
        if expected != Some(plan) {
            return Err(WritebackError::PlanMismatch);
        }
        let sectors = plan.dirty_sectors();
        let requested = self
            .in_flight_sectors
            .checked_add(sectors)
            .ok_or(WritebackError::ArithmeticOverflow)?;
        if requested > self.capacity_sectors {
            return Err(WritebackError::CapacityExceeded {
                capacity: self.capacity_sectors,
                requested,
            });
        }
        let next_token = self
            .next_token
            .checked_add(1)
            .ok_or(WritebackError::ArithmeticOverflow)?;
        let next_epoch = self
            .epoch
            .checked_add(1)
            .ok_or(WritebackError::ArithmeticOverflow)?;
        self.free_fp = plan.done_fp;
        self.in_flight_sectors = requested;
        self.latest_by_line
            .insert(plan.line_id, (plan.token, plan.done_fp));
        self.pending.insert(plan.token, PendingWriteback { plan });
        self.next_token = next_token;
        self.epoch = next_epoch;
        Ok(())
    }

    /// Convenience transaction for callers that do not need to coordinate a
    /// separate cache commit.  Cache integrations should use `preview` and
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

    /// Apply a scheduled completion acknowledgement.  The latest same-line
    /// hazard is removed only when its own token completes.
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
        let next_epoch = self
            .epoch
            .checked_add(1)
            .ok_or(WritebackError::ArithmeticOverflow)?;
        self.pending.remove(&token);
        self.in_flight_sectors = self
            .in_flight_sectors
            .checked_sub(pending.plan.dirty_sectors())
            .ok_or(WritebackError::ArithmeticOverflow)?;
        if self
            .latest_by_line
            .get(&pending.plan.line_id)
            .is_some_and(|(latest, _)| *latest == token)
        {
            self.latest_by_line.remove(&pending.plan.line_id);
        }
        self.epoch = next_epoch;
        Ok(WritebackCompletion {
            token,
            line_id: pending.plan.line_id,
            dirty_mask: pending.plan.dirty_mask,
            service_done_cycle: pending.plan.service_done_cycle,
        })
    }

    fn scale_cycle(&self, cycle: u64) -> Result<u128, WritebackError> {
        u128::from(cycle)
            .checked_mul(self.bytes_per_sec)
            .ok_or(WritebackError::ArithmeticOverflow)
    }

    fn ceil_cycle(&self, fixed: u128) -> Result<u64, WritebackError> {
        let cycle = fixed
            .checked_add(self.bytes_per_sec - 1)
            .ok_or(WritebackError::ArithmeticOverflow)?
            .checked_div(self.bytes_per_sec)
            .ok_or(WritebackError::ArithmeticOverflow)?;
        u64::try_from(cycle).map_err(|_| WritebackError::ArithmeticOverflow)
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn eviction(line_id: i64, valid_mask: u8, dirty_mask: u8) -> L2Eviction {
        L2Eviction {
            line_id,
            valid_mask,
            dirty_mask,
            pending_mask: 0,
        }
    }

    #[test]
    fn clean_eviction_is_a_noop_and_dirty_service_is_tokenized() {
        let mut queue = WritebackQueue::new(4, 2048, 32).expect("queue should initialize");
        assert_eq!(queue.submit(eviction(1, 0b0011, 0), 0), Ok(None));
        let plan = queue
            .submit(eviction(2, 0b0101, 0b0101), 3)
            .expect("dirty eviction should submit")
            .expect("dirty plan should exist");
        assert_eq!(plan.bytes, 64);
        assert_eq!(plan.service_start_cycle, 3);
        assert_eq!(plan.service_done_cycle, 4);
        assert_eq!(queue.in_flight_sectors(), 2);
        let completion = queue
            .complete(plan.token, plan.service_done_cycle)
            .expect("completion should be accepted");
        assert_eq!(completion.line_id, 2);
        assert_eq!(queue.in_flight_sectors(), 0);
    }

    #[test]
    fn capacity_and_invalid_masks_fail_without_mutating_queue() {
        let mut queue = WritebackQueue::new(1, 2048, 32).expect("queue should initialize");
        let before = queue.clone();
        assert!(matches!(
            queue.submit(eviction(1, 0b0011, 0b0011), 0),
            Err(WritebackError::CapacityExceeded { .. })
        ));
        assert_eq!(queue, before);
        assert!(matches!(
            queue.submit(
                L2Eviction {
                    line_id: 1,
                    valid_mask: 0b0011,
                    dirty_mask: 0b0001,
                    pending_mask: 0b0001,
                },
                0,
            ),
            Err(WritebackError::DirtyPendingOverlap { .. })
        ));
        assert_eq!(queue, before);
    }

    #[test]
    fn same_line_hazard_survives_old_completion_and_stale_plans_reject() {
        let mut queue = WritebackQueue::new(4, 2048, 32).expect("queue should initialize");
        let first = queue
            .submit(eviction(7, 0b0001, 0b0001), 0)
            .expect("first submit should succeed")
            .expect("first plan should exist");
        assert!(
            queue.fetch_arrival(7, 0).expect("arrival should compute") >= first.service_done_cycle
        );
        let stale = queue
            .preview(eviction(8, 0b0001, 0b0001), 0)
            .expect("preview should succeed")
            .expect("preview should exist");
        let second = queue
            .submit(eviction(7, 0b0010, 0b0010), 1)
            .expect("second submit should succeed")
            .expect("second plan should exist");
        assert!(matches!(
            queue.commit(stale),
            Err(WritebackError::StalePlan { .. })
        ));
        queue
            .complete(first.token, first.service_done_cycle)
            .expect("first completion should succeed");
        assert!(
            queue.fetch_arrival(7, 1).expect("arrival should compute") >= second.service_done_cycle
        );
        assert!(matches!(
            queue.complete(second.token, second.service_done_cycle - 1),
            Err(WritebackError::EarlyCompletion { .. })
        ));
        assert_eq!(queue.in_flight_sectors(), 1);
        queue
            .complete(second.token, second.service_done_cycle)
            .expect("second completion should succeed");
        assert_eq!(queue.fetch_arrival(7, 1), Ok(1));
    }

    #[test]
    fn forged_public_plan_fields_are_rejected_transactionally() {
        let mut queue = WritebackQueue::new(2, 2048, 32).expect("queue should initialize");
        let plan = queue
            .preview(eviction(9, 0b0001, 0b0001), 0)
            .expect("preview should succeed")
            .expect("plan should exist");
        let before = queue.clone();
        let mut forged = plan;
        forged.bytes += 32;
        assert_eq!(queue.commit(forged), Err(WritebackError::PlanMismatch));
        assert_eq!(queue, before);
    }
}
