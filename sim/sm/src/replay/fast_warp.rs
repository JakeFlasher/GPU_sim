//! Fast-path warp replay helpers; drives the fast execution path without changing full-mode cache semantics.

use super::*;

pub(super) struct FastDynamicWarp<'a> {
    pub(super) warp: &'a WarpReplay,
    pub(super) masks: &'a [u32],
    pub(super) next: usize,
    pub(super) slot: Option<&'a crate::program::IssueSlot>,
    pub(super) ready: u64,
    pub(super) completion: u64,
    pub(super) waiting_at_barrier: bool,
    pub(super) barrier_arrival: u64,
    pub(super) done: bool,
    pub(super) registers: Vec<(SassRegister, u32, u64)>,
    pub(super) barriers: [u64; 6],
}

impl FastDynamicWarp<'_> {
    pub(super) fn ready_at(
        &self,
        dispatch: u64,
        pipes: &[u64; 4],
        pending: &[u64],
        rearm: u64,
    ) -> Option<u64> {
        if self.done || self.waiting_at_barrier {
            return None;
        }
        let slot = self.slot?;
        let mask = self.masks[self.next];
        let mut ready = self.ready.max(dispatch);
        if let Some(guard) = slot.guard_predicate {
            let lanes = fast_register_mask(guard, self.warp.insts[self.next].active_lane_mask);
            for &(prior, prior_lanes, done) in &self.registers {
                if lanes & prior_lanes != 0 && fast_register_overlap(prior, guard) {
                    ready = ready.max(done);
                }
            }
        }
        if mask != 0 {
            ready = ready.max(pipes[fast_pipeline_index(slot.pipeline_class())]);
            let waits =
                slot.control().wait_mask().bits() | slot.ordinary_dependency_wait_mask().bits();
            for (index, &clear) in self.barriers.iter().enumerate() {
                if waits & (1 << index) != 0 {
                    ready = ready.max(clear);
                }
            }
            if let Some(projection) = slot.fixed_latency_operands().projected() {
                let sources = projection
                    .sources()
                    .iter()
                    .filter_map(|source| match source {
                        SassSourceOperand::Register(operand) => Some(operand.register()),
                        _ => None,
                    });
                // RAW and WAW are both completion dependencies. Include predicate
                // inputs, ignore constant registers, and preserve lane identity.
                for register in sources
                    .chain(projection.predicate_sources().iter().map(|o| o.register()))
                    .chain(projection.destinations().iter().map(|o| o.register()))
                {
                    let lanes = fast_register_mask(register, mask);
                    for &(prior, prior_lanes, done) in &self.registers {
                        if lanes & prior_lanes != 0 && fast_register_overlap(prior, register) {
                            ready = ready.max(done);
                        }
                    }
                }
            }
            if let PipelineClass::ScalarAlu {
                in_flight_capacity, ..
            } = slot.pipeline_class()
            {
                // Candidate queries must never remove another candidate's pending
                // completions at a speculative future timestamp.
                while pending.iter().filter(|&&done| done > ready).count()
                    >= in_flight_capacity.max(1) as usize
                {
                    ready = pending.iter().copied().filter(|&done| done > ready).min()?;
                }
            }
            if slot.op_class() == OpClass::BlockBarrier {
                ready = ready.max(rearm);
            }
        }
        Some(ready)
    }
}
