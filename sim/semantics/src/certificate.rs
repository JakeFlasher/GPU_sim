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
    UnknownResidency,
    OccupancyNotExclusive,
    IncomingPipelineWork,
    /// Per-warp sequence ordinals are not a globally comparable issue order.
    /// Without an independent scheduler/arbitration witness, a multi-warp
    /// interval must remain on the exact path.
    CrossWarpOrderUnknown,
    /// No independently derived dependency/effect witness covers the
    /// candidate interval; producer-supplied booleans are insufficient.
    MissingIndependentEvidence,
    /// The certificate does not bind to the descriptor region presented to
    /// the executor.  A caller must not install a summary derived for a
    /// different pipeline, timing tuple, or sequence span.
    CertificateBinding,
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
    pub const fn pipeline_key(self) -> u8 {
        self.pipeline_key
    }
    pub const fn latency_cycles(self) -> u64 {
        self.latency_cycles
    }
    pub const fn initiation_interval_cycles(self) -> u64 {
        self.initiation_interval_cycles
    }
    pub const fn in_flight_capacity(self) -> u32 {
        self.in_flight_capacity
    }

    /// Materialize the exact issue/completion frontier for every instruction
    /// in this certificate. This is intentionally separate from the compact
    /// pending-completion projection: a cycle-accuracy checker must be able to
    /// compare intermediate observations, not only the terminal resource state.
    pub fn issue_completion_times(self, first_issue: u64) -> Result<Vec<(u64, u64)>, ReifyReason> {
        let narrow = |time: u128| u64::try_from(time).map_err(|_| ReifyReason::ArithmeticOverflow);
        let mut observations = Vec::with_capacity(
            usize::try_from(self.instruction_count).map_err(|_| ReifyReason::ArithmeticOverflow)?,
        );
        for index in 0..self.instruction_count {
            let issue = narrow(
                u128::from(first_issue)
                    .checked_add(self.issue_offset(index)?)
                    .ok_or(ReifyReason::ArithmeticOverflow)?,
            )?;
            let completion = narrow(
                u128::from(issue)
                    .checked_add(u128::from(self.latency_cycles))
                    .ok_or(ReifyReason::ArithmeticOverflow)?,
            )?;
            observations.push((issue, completion));
        }
        Ok(observations)
    }

    // For capacity C, latency L, and initiation interval I:
    // issue(k) = floor(k/C) * max(L, C*I) + (k mod C)*I.
    // This solves t[k] = max(t[k-1]+I, t[k-C]+L) for an empty pipeline.
    fn issue_offset(self, index: u64) -> Result<u128, ReifyReason> {
        let capacity = u128::from(self.in_flight_capacity);
        let interval = u128::from(self.initiation_interval_cycles);
        let period = u128::from(self.latency_cycles).max(
            capacity
                .checked_mul(interval)
                .ok_or(ReifyReason::ArithmeticOverflow)?,
        );
        let full_periods = u128::from(index) / capacity;
        let in_period = u128::from(index) % capacity;
        full_periods
            .checked_mul(period)
            .and_then(|offset| {
                in_period
                    .checked_mul(interval)
                    .and_then(|tail| offset.checked_add(tail))
            })
            .ok_or(ReifyReason::ArithmeticOverflow)
    }

    pub fn project(self, first_issue: u64) -> Result<IntervalProjection, ReifyReason> {
        let narrow = |time: u128| u64::try_from(time).map_err(|_| ReifyReason::ArithmeticOverflow);
        let last = self
            .issue_offset(self.instruction_count - 1)?
            .checked_add(u128::from(first_issue))
            .ok_or(ReifyReason::ArithmeticOverflow)?;
        let last_issue = narrow(last)?;
        let next_issue = narrow(
            last.checked_add(u128::from(self.initiation_interval_cycles))
                .ok_or(ReifyReason::ArithmeticOverflow)?,
        )?;
        let dispatch_ready = narrow(last.checked_add(1).ok_or(ReifyReason::ArithmeticOverflow)?)?;
        let last_completion = narrow(
            last.checked_add(u128::from(self.latency_cycles))
                .ok_or(ReifyReason::ArithmeticOverflow)?,
        )?;
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
                    .checked_add(self.issue_offset(index)?)
                    .and_then(|value| value.checked_add(u128::from(self.latency_cycles)))
                    .ok_or(ReifyReason::ArithmeticOverflow)?,
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
    let last = certificate.issue_offset(count - 1)?;
    if last
        .checked_add(u128::from(
            first.latency_cycles.max(first.initiation_interval_cycles),
        ))
        .ok_or(ReifyReason::ArithmeticOverflow)?
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
