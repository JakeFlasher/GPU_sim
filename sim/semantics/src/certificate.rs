//! Exact summaries for a narrow descriptor model: one independent lane group,
//! one homogeneous pipeline, and no incoming in-flight work on that pipeline.
//! Producer-supplied dependence/effect facts remain assumptions of this model;
//! these summaries do not establish silicon accuracy or general program safety.

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionEvidence {
    pub sequence: u64,
    pub pipeline_key: u8,
    pub latency_cycles: u64,
    pub initiation_interval_cycles: u64,
    pub in_flight_capacity: u32,
    pub completion_units: u32,
    pub control_clean: bool,
    pub dependency_free: bool,
    pub external_side_effect: bool,
}

impl ExecutionEvidence {
    pub const fn locally_summarizable(self) -> bool {
        self.control_clean
            && self.dependency_free
            && !self.external_side_effect
            && self.completion_units == 1
            && self.latency_cycles != 0
            && self.initiation_interval_cycles != 0
            && self.in_flight_capacity != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReifyReason {
    EmptyInterval,
    NonContiguousSequence,
    MixedPipeline,
    MixedTiming,
    MissingTiming,
    ControlEffect,
    Dependency,
    ExternalSideEffect,
    CompletionMultiplicity,
    OccupancyNotExclusive,
    IncomingPipelineWork,
    ArithmeticOverflow,
}

/// Constructed only by checking a descriptor interval. Timing is relative to
/// its first issue. Applying this summary requires an empty pipeline at entry;
/// callers must retain the projected pending completions for later work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct IntervalCertificate {
    first_sequence: u64,
    last_sequence: u64,
    instruction_count: u64,
    pipeline_key: u8,
    latency_cycles: u64,
    initiation_interval_cycles: u64,
    in_flight_capacity: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntervalProjection {
    pub first_issue: u64,
    pub last_issue: u64,
    pub next_issue: u64,
    pub dispatch_ready: u64,
    pub last_completion: u64,
    /// State immediately after the last issue, including its new completion.
    pub pending_completions: Vec<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "decision")]
pub enum FidelityDecision {
    Summary { certificate: IntervalCertificate },
    Reify { reason: ReifyReason },
}

impl IntervalCertificate {
    pub const fn first_sequence(self) -> u64 {
        self.first_sequence
    }
    pub const fn last_sequence(self) -> u64 {
        self.last_sequence
    }
    pub const fn instruction_count(self) -> u64 {
        self.instruction_count
    }

    // For capacity C, latency L, and initiation interval I:
    // issue(k) = floor(k/C) * max(L, C*I) + (k mod C)*I.
    // This solves t[k] = max(t[k-1]+I, t[k-C]+L) for an empty pipeline.
    fn issue_offset(self, index: u64) -> u128 {
        let capacity = u128::from(self.in_flight_capacity);
        let interval = u128::from(self.initiation_interval_cycles);
        let period = u128::from(self.latency_cycles).max(capacity * interval);
        u128::from(index) / capacity * period + (u128::from(index) % capacity) * interval
    }

    pub fn project(self, first_issue: u64) -> Result<IntervalProjection, ReifyReason> {
        let narrow = |time: u128| u64::try_from(time).map_err(|_| ReifyReason::ArithmeticOverflow);
        let last = self.issue_offset(self.instruction_count - 1) + u128::from(first_issue);
        let last_issue = narrow(last)?;
        let next_issue = narrow(last + u128::from(self.initiation_interval_cycles))?;
        let dispatch_ready = narrow(last + 1)?;
        let last_completion = narrow(last + u128::from(self.latency_cycles))?;
        let pending_count = self
            .instruction_count
            .min(u64::from(self.in_flight_capacity))
            .min(
                self.latency_cycles
                    .div_ceil(self.initiation_interval_cycles),
            );
        let tail_start = self.instruction_count - pending_count;
        let mut pending_completions = Vec::new();
        for index in tail_start..self.instruction_count {
            let done = narrow(
                u128::from(first_issue)
                    + self.issue_offset(index)
                    + u128::from(self.latency_cycles),
            )?;
            if done > last_issue {
                pending_completions.push(done);
            }
        }
        Ok(IntervalProjection {
            first_issue,
            last_issue,
            next_issue,
            dispatch_ready,
            last_completion,
            pending_completions,
        })
    }
}

pub fn certify_interval(
    instructions: impl IntoIterator<Item = ExecutionEvidence>,
    active_occupants: u32,
) -> FidelityDecision {
    let result = build_certificate(instructions, active_occupants);
    match result {
        Ok(certificate) => FidelityDecision::Summary { certificate },
        Err(reason) => FidelityDecision::Reify { reason },
    }
}

fn build_certificate(
    instructions: impl IntoIterator<Item = ExecutionEvidence>,
    active_occupants: u32,
) -> Result<IntervalCertificate, ReifyReason> {
    let mut instructions = instructions.into_iter();
    let first = instructions.next().ok_or(ReifyReason::EmptyInterval)?;
    if active_occupants != 1 {
        return Err(ReifyReason::OccupancyNotExclusive);
    }
    check_instruction(first)?;
    let mut count = 1u64;
    let mut previous = first.sequence;
    for instruction in instructions {
        check_instruction(instruction)?;
        if previous.checked_add(1) != Some(instruction.sequence) {
            return Err(ReifyReason::NonContiguousSequence);
        }
        if instruction.pipeline_key != first.pipeline_key {
            return Err(ReifyReason::MixedPipeline);
        }
        if instruction.latency_cycles != first.latency_cycles
            || instruction.initiation_interval_cycles != first.initiation_interval_cycles
            || instruction.in_flight_capacity != first.in_flight_capacity
        {
            return Err(ReifyReason::MixedTiming);
        }
        count = count
            .checked_add(1)
            .ok_or(ReifyReason::ArithmeticOverflow)?;
        previous = instruction.sequence;
    }
    let certificate = IntervalCertificate {
        first_sequence: first.sequence,
        last_sequence: previous,
        instruction_count: count,
        pipeline_key: first.pipeline_key,
        latency_cycles: first.latency_cycles,
        initiation_interval_cycles: first.initiation_interval_cycles,
        in_flight_capacity: first.in_flight_capacity,
    };
    // Check the arithmetic without allocating a pending-completion vector.
    let last = certificate.issue_offset(count - 1);
    if last + u128::from(first.latency_cycles.max(first.initiation_interval_cycles))
        > u128::from(u64::MAX)
    {
        return Err(ReifyReason::ArithmeticOverflow);
    }
    Ok(certificate)
}

fn check_instruction(instruction: ExecutionEvidence) -> Result<(), ReifyReason> {
    if instruction.initiation_interval_cycles == 0
        || instruction.in_flight_capacity == 0
        || instruction.latency_cycles == 0
    {
        return Err(ReifyReason::MissingTiming);
    }
    if !instruction.control_clean {
        return Err(ReifyReason::ControlEffect);
    }
    if !instruction.dependency_free {
        return Err(ReifyReason::Dependency);
    }
    if instruction.external_side_effect {
        return Err(ReifyReason::ExternalSideEffect);
    }
    if instruction.completion_units != 1 {
        return Err(ReifyReason::CompletionMultiplicity);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn evidence(sequence: u64) -> ExecutionEvidence {
        ExecutionEvidence {
            sequence,
            pipeline_key: 1,
            latency_cycles: 4,
            initiation_interval_cycles: 2,
            in_flight_capacity: 8,
            completion_units: 1,
            control_clean: true,
            dependency_free: true,
            external_side_effect: false,
        }
    }
    #[test]
    fn mixed_timing_and_maximum_sequence_do_not_get_certificates() {
        for field in 0..3 {
            let mut second = evidence(1);
            match field {
                0 => second.latency_cycles = 1,
                1 => second.initiation_interval_cycles = 1,
                _ => second.in_flight_capacity = 1,
            }
            assert_eq!(
                certify_interval([evidence(0), second], 1),
                FidelityDecision::Reify {
                    reason: ReifyReason::MixedTiming
                }
            );
        }
        assert_eq!(
            certify_interval([evidence(u64::MAX), evidence(u64::MAX)], 1),
            FidelityDecision::Reify {
                reason: ReifyReason::NonContiguousSequence
            }
        );
    }
    #[test]
    fn missing_dependency_or_shared_occupancy_reifies() {
        let mut dependent = evidence(0);
        dependent.dependency_free = false;
        assert_eq!(
            certify_interval([dependent, evidence(1)], 1),
            FidelityDecision::Reify {
                reason: ReifyReason::Dependency
            }
        );
        assert_eq!(
            certify_interval([evidence(0), evidence(1)], 2),
            FidelityDecision::Reify {
                reason: ReifyReason::OccupancyNotExclusive
            }
        );
    }
    #[test]
    fn projection_matches_independent_cycle_tick_with_capacity_pressure() {
        for count in 1..=16u64 {
            for capacity in 1..=4u32 {
                for interval in 1..=4u64 {
                    for latency in 1..=16u64 {
                        let descriptors = (0..count).map(|sequence| ExecutionEvidence {
                            latency_cycles: latency,
                            initiation_interval_cycles: interval,
                            in_flight_capacity: capacity,
                            ..evidence(sequence)
                        });
                        let FidelityDecision::Summary { certificate } =
                            certify_interval(descriptors, 1)
                        else {
                            panic!("valid interval")
                        };
                        // Deliberately cycle-stepped reference: it does not use the closed form.
                        let start = 7;
                        let mut now = start;
                        let mut next_issue = start;
                        let mut pending = Vec::new();
                        let mut issued = 0;
                        loop {
                            pending.retain(|done| *done > now);
                            if now >= next_issue && pending.len() < capacity as usize {
                                issued += 1;
                                pending.push(now + latency);
                                next_issue = now + interval;
                                if issued == count {
                                    break;
                                }
                            }
                            now += 1;
                        }
                        let projected = certificate.project(start).unwrap();
                        assert_eq!(
                            projected,
                            IntervalProjection {
                                first_issue: start,
                                last_issue: now,
                                next_issue,
                                dispatch_ready: now + 1,
                                last_completion: now + latency,
                                pending_completions: pending,
                            },
                            "n={count} capacity={capacity} interval={interval} latency={latency}"
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn projection_reports_clock_overflow_without_saturating() {
        let FidelityDecision::Summary { certificate } = certify_interval([evidence(0)], 1) else {
            panic!("valid")
        };
        assert_eq!(
            certificate.project(u64::MAX),
            Err(ReifyReason::ArithmeticOverflow)
        );
    }
}
