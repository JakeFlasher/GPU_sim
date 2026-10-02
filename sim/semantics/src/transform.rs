//! Contract-aware typed boundary transformations.
//!
//! A certificate must derive its exit from the entry using a known transition
//! rule. Structural validity and equality with a supplied exit do not prove a
//! transition. The identity rule implemented here consumes no cycles or events
//! and leaves every field unchanged. Advancing native SM execution requires a
//! different rule and its corresponding checker.

use serde::{Deserialize, Serialize};

use crate::{
    Completeness, PowerDomainStatus, TypedEventBoundaryRecord, TypedEventIdentity,
    TypedEventPayload, TypedObservationContract, TypedPendingEvent,
};

pub const TRANSFORM_SCHEMA_VERSION: u32 = 2;
const IDENTITY_TRANSFORMER: &str = "identity_boundary_v2";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransformInterval {
    pub start_cycle: u64,
    pub end_cycle: u64,
    pub start_sequence: u64,
    pub end_sequence: u64,
    /// Sequence monotonicity is an optional producer witness. Event time and
    /// insertion sequence are independent coordinates when an event can be
    /// inserted after an earlier timestamp has already been scheduled.
    #[serde(default)]
    pub sequence_ordered: bool,
}

impl TransformInterval {
    pub fn validate(self) -> Result<(), TransformIntervalError> {
        if self.end_cycle < self.start_cycle
            || (self.sequence_ordered && self.end_sequence < self.start_sequence)
        {
            return Err(TransformIntervalError::Reversed);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalInputFrontier {
    pub next_event_time: Option<u64>,
    pub closed_world: bool,
    pub unknown: Vec<String>,
}

impl ExternalInputFrontier {
    fn validate(self, interval: TransformInterval) -> Result<(), TransformReifyReason> {
        if !self.closed_world || !self.unknown.is_empty() {
            return Err(TransformReifyReason::OpenExternalInput);
        }
        if self
            .next_event_time
            .is_some_and(|time| time <= interval.end_cycle)
        {
            return Err(TransformReifyReason::ExternalInputInsideInterval);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransformInput {
    pub entry: TypedEventBoundaryRecord,
    /// Candidate exit to check, never the source of a computed transition.
    pub exit: TypedEventBoundaryRecord,
    pub external_frontier: ExternalInputFrontier,
    pub contract: TypedObservationContract,
    pub interval: TransformInterval,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedEventTransition {
    pub phase: u8,
    pub event: TypedEventIdentity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationDelta {
    pub elapsed_cycles: u64,
    pub emitted_event_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SummaryProof {
    pub schema_version: u32,
    pub transformer: String,
    pub interval: TransformInterval,
    pub contract: TypedObservationContract,
    pub external_frontier: ExternalInputFrontier,
    pub entry: TypedEventBoundaryRecord,
    pub exit: TypedEventBoundaryRecord,
    pub emitted_events: Vec<TypedEventTransition>,
    pub observation_delta: ObservationDelta,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransformerOutput {
    pub exit: TypedEventBoundaryRecord,
    pub emitted_events: Vec<TypedEventTransition>,
    pub observation_delta: ObservationDelta,
    pub proof: SummaryProof,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "decision")]
pub enum TransformerDecision {
    Certify(Box<TransformerOutput>),
    Reify { reason: TransformReifyReason },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeBoundarySide {
    Summary,
    Detailed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum NativeBoundaryReifyReason {
    SummaryStructure {
        error: String,
    },
    DetailedStructure {
        error: String,
    },
    ContractMismatch {
        side: NativeBoundarySide,
    },
    Incomplete {
        side: NativeBoundarySide,
        unknown: Vec<String>,
    },
    Refused {
        side: NativeBoundarySide,
        reason: String,
    },
    ProjectionMismatch,
    InternalEffectMismatch {
        expected: usize,
        observed: usize,
        first_difference: Option<String>,
    },
    ObservableLabelMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "decision")]
pub enum NativeBoundaryRefinementDecision {
    Preserve,
    Reify { reason: NativeBoundaryReifyReason },
}

/// Execution path selected from a native boundary decision. The current
/// native replay writer only records this plan; it never replaces `SmCore`
/// execution without a complete macro implementation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "path")]
pub enum NativeBoundaryExecutionPath {
    TypedMacro,
    ExactReplay { reason: NativeBoundaryReifyReason },
}

/// Result of consuming a native-boundary decision. The macro branch carries
/// the independently derived transition contract; the exact branch carries
/// the detailed exit that must remain authoritative.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "execution")]
pub enum NativeBoundaryExecutionResult {
    TypedMacro {
        contract: TypedRegionContract,
    },
    ExactReplay {
        reason: NativeBoundaryReifyReason,
        detailed_exit: TypedEventBoundaryRecord,
    },
}

pub fn native_boundary_execution_path(
    decision: &NativeBoundaryRefinementDecision,
) -> NativeBoundaryExecutionPath {
    match decision {
        NativeBoundaryRefinementDecision::Preserve => NativeBoundaryExecutionPath::TypedMacro,
        NativeBoundaryRefinementDecision::Reify { reason } => {
            NativeBoundaryExecutionPath::ExactReplay {
                reason: reason.clone(),
            }
        }
    }
}

/// State-complete region contract. A scalar cycle recurrence is not enough to
/// construct this value: callers must supply both typed boundaries, all
/// internal effects, and every pending completion token.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedRegionContract {
    pub schema_version: u32,
    pub contract: TypedObservationContract,
    pub interval: TransformInterval,
    pub entry: TypedEventBoundaryRecord,
    pub internal_effects: Vec<TypedEventTransition>,
    pub pending_completion_tokens: Vec<TypedPendingEvent>,
    pub observable_labels: Vec<String>,
    pub exit: TypedEventBoundaryRecord,
}

/// Narrow typed transition used to exercise the native-boundary contract. It
/// deliberately accepts only a single closed warp with no front-end/cache/
/// response state, so its exit can be derived without copying a detailed exit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedScalarMacroSpec {
    pub instruction_count: u64,
    pub latency_cycles: u64,
    #[serde(default = "default_macro_interval")]
    pub initiation_interval_cycles: u64,
    pub label: String,
}

fn default_macro_interval() -> u64 {
    1
}

pub fn derive_typed_scalar_macro_region(
    entry: &TypedEventBoundaryRecord,
    spec: &TypedScalarMacroSpec,
) -> Result<TypedRegionContract, RegionContractError> {
    if spec.instruction_count == 0 || spec.latency_cycles == 0 || spec.label.trim().is_empty() {
        return Err(RegionContractError::UnsupportedScalarMacro {
            reason: "nonzero_instruction_count_latency_and_label_required".to_owned(),
        });
    }
    entry
        .validate_structure()
        .map_err(|error| RegionContractError::Boundary {
            side: NativeBoundarySide::Summary,
            error: error.to_string(),
        })?;
    let pending_macro_safe = entry.queue.pending.iter().all(|pending| {
        matches!(
            pending.payload,
            TypedEventPayload::SmWake | TypedEventPayload::AtomicCompletion
        )
    });
    if entry.completeness != Completeness::Complete || entry.cores.len() != 1 || !pending_macro_safe
    {
        return Err(RegionContractError::UnsupportedScalarMacro {
            reason: "entry_has_incomplete_state_or_unsafe_pending_events".to_owned(),
        });
    }
    if spec.initiation_interval_cycles == 0 {
        return Err(RegionContractError::UnsupportedScalarMacro {
            reason: "initiation_interval_must_be_nonzero".to_owned(),
        });
    }
    let first_issue = entry
        .cores
        .first()
        .and_then(|core| core.warp_states.first())
        .map(|warp| entry.event.time.max(warp.eligible))
        .unwrap_or(entry.event.time);
    let mut exit = entry.clone();
    let last_issue_offset = spec
        .instruction_count
        .checked_sub(1)
        .and_then(|count| count.checked_mul(spec.initiation_interval_cycles))
        .ok_or(RegionContractError::ScalarMacroArithmeticOverflow)?;
    let duration = first_issue
        .checked_sub(entry.event.time)
        .and_then(|offset| offset.checked_add(last_issue_offset))
        .and_then(|offset| offset.checked_add(1))
        .ok_or(RegionContractError::ScalarMacroArithmeticOverflow)?;
    let end_time = entry
        .event
        .time
        .checked_add(duration)
        .ok_or(RegionContractError::ScalarMacroArithmeticOverflow)?;
    let end_sequence = entry
        .event
        .sequence
        .checked_add(spec.instruction_count + 1)
        .ok_or(RegionContractError::ScalarMacroArithmeticOverflow)?;
    let core = exit
        .cores
        .first_mut()
        .ok_or(RegionContractError::UnsupportedScalarMacro {
            reason: "core_missing".to_owned(),
        })?;
    let l1_open = core.l1.as_ref().is_some_and(|l1| {
        !l1.line_fills.is_empty()
            || !l1.sector_fills.is_empty()
            || l1.active_lookup_batch.is_some()
            || l1.min_fill_due.is_some()
    });
    let memory_pipeline_open = core.memory_pipeline.as_ref().is_some_and(|pipeline| {
        pipeline
            .queue_release_times
            .iter()
            .any(|queue| !queue.is_empty())
            || pipeline
                .local_completion_ready
                .iter()
                .any(|&ready| ready > core.cycle)
            || pipeline.shared_accept_ready > core.cycle
    });
    let scalar_pipeline_open = core.scalar_pipeline.as_ref().is_some_and(|pipeline| {
        pipeline.release_times.iter().any(|queue| !queue.is_empty())
            || pipeline
                .next_service
                .iter()
                .any(|&ready| ready > core.cycle)
    });
    if core.warp_states.len() != 1
        || core.warp_states[0].done
        || core.warp_states[0]
            .barrier_pending
            .iter()
            .any(|&pending| pending != 0)
        || core.warp_states[0].at_block_barrier
        || core.warp_states[0].block_barrier_dispatch.is_some()
        || core.warp_states[0].block_barrier_arrival.is_some()
        || core.barrier_pipe_ready != 0
        || core.block_states.iter().any(|block| block.waiting != 0)
        || core.outstanding_responses != 0
        || !core.response_owners.is_empty()
        || l1_open
        || memory_pipeline_open
        || scalar_pipeline_open
        || core.async_copy.iter().any(Option::is_some)
    {
        return Err(RegionContractError::UnsupportedScalarMacro {
            reason: "future_relevant_warp_or_response_state_is_not_closed".to_owned(),
        });
    }
    let buffer_count = usize::try_from(spec.instruction_count)
        .map_err(|_| RegionContractError::ScalarMacroArithmeticOverflow)?;
    if core.front_end.is_some() {
        if core.warp_states[0].instruction_buffer.len() < buffer_count {
            return Err(RegionContractError::UnsupportedScalarMacro {
                reason: "instruction_buffer_does_not_close_macro_entry".to_owned(),
            });
        }
        let expected_start = core.warp_states[0].next_instruction;
        if !core.warp_states[0].instruction_buffer[..buffer_count]
            .iter()
            .enumerate()
            .all(|(index, &(instruction, _))| instruction == expected_start + index as u64)
        {
            return Err(RegionContractError::UnsupportedScalarMacro {
                reason: "instruction_buffer_not_contiguous_at_macro_entry".to_owned(),
            });
        }
        core.warp_states[0].instruction_buffer.drain(..buffer_count);
    } else if !core.warp_states[0].instruction_buffer.is_empty() {
        return Err(RegionContractError::UnsupportedScalarMacro {
            reason: "instruction_buffer_present_without_front_end_state".to_owned(),
        });
    }
    core.issued = core
        .issued
        .checked_add(spec.instruction_count)
        .ok_or(RegionContractError::ScalarMacroArithmeticOverflow)?;
    core.cycle = end_time;
    core.warp_states[0].next_instruction = core.warp_states[0]
        .next_instruction
        .checked_add(spec.instruction_count)
        .ok_or(RegionContractError::ScalarMacroArithmeticOverflow)?;
    core.warp_states[0].front_end_fetch_cursor = core.warp_states[0]
        .front_end_fetch_cursor
        .checked_add(spec.instruction_count)
        .ok_or(RegionContractError::ScalarMacroArithmeticOverflow)?;
    core.warp_states[0].eligible = end_time;
    for domain in &mut exit.power.domains {
        domain.status = PowerDomainStatus::Clocked;
        domain.clocked_cycles = domain
            .clocked_cycles
            .checked_add(duration)
            .ok_or(RegionContractError::ScalarMacroArithmeticOverflow)?;
        domain.completion_events = domain
            .completion_events
            .checked_add(spec.instruction_count)
            .ok_or(RegionContractError::ScalarMacroArithmeticOverflow)?;
        domain.last_cycle = end_time;
    }
    exit.event = TypedEventIdentity {
        time: end_time,
        sequence: end_sequence,
        sequence_span: 1,
        phase: entry.event.phase,
        target: entry.event.target,
        payload: TypedEventPayload::SmWake,
    };
    exit.queue.next_event_time = exit.queue.pending.first().map(|pending| pending.time);
    let next_sequence = end_sequence
        .checked_add(1)
        .ok_or(RegionContractError::ScalarMacroArithmeticOverflow)?;
    exit.queue.next_sequence = Some(exit.queue.next_sequence.unwrap_or(0).max(next_sequence));
    let mut effects = Vec::with_capacity(
        usize::try_from(spec.instruction_count)
            .map_err(|_| RegionContractError::ScalarMacroArithmeticOverflow)?,
    );
    for index in 1..=spec.instruction_count {
        effects.push(TypedEventTransition {
            phase: entry.event.phase,
            event: TypedEventIdentity {
                time: entry
                    .event
                    .time
                    .max(first_issue)
                    .checked_add(
                        index
                            .checked_sub(1)
                            .and_then(|offset| offset.checked_mul(spec.initiation_interval_cycles))
                            .ok_or(RegionContractError::ScalarMacroArithmeticOverflow)?,
                    )
                    .ok_or(RegionContractError::ScalarMacroArithmeticOverflow)?,
                sequence: entry
                    .event
                    .sequence
                    .checked_add(index)
                    .ok_or(RegionContractError::ScalarMacroArithmeticOverflow)?,
                sequence_span: 1,
                phase: entry.event.phase,
                target: entry.event.target,
                payload: TypedEventPayload::SmWake,
            },
        });
    }
    TypedRegionContract::new(
        entry.contract.clone(),
        TransformInterval {
            start_cycle: entry.event.time,
            end_cycle: end_time,
            start_sequence: entry.event.sequence,
            end_sequence,
            sequence_ordered: true,
        },
        entry.clone(),
        effects,
        exit.queue.pending.clone(),
        vec![spec.label.clone()],
        exit,
    )
}

/// Execute the typed scalar macro seam against an independently produced
/// detailed continuation. The returned path is what a native dispatcher would
/// consume; callers must use `ExactReplay` for every rejected contract.
pub fn execute_typed_scalar_macro(
    entry: &TypedEventBoundaryRecord,
    spec: &TypedScalarMacroSpec,
    detailed_effects: &[TypedEventTransition],
    detailed_labels: &[String],
    detailed_exit: &TypedEventBoundaryRecord,
) -> Result<NativeBoundaryExecutionResult, RegionContractError> {
    let summary = derive_typed_scalar_macro_region(entry, spec)?;
    let decision = compare_native_region_contract_with_trace(
        Some(&summary),
        entry,
        detailed_effects,
        detailed_labels,
        detailed_exit,
    );
    Ok(match decision {
        NativeBoundaryRefinementDecision::Preserve => {
            NativeBoundaryExecutionResult::TypedMacro { contract: summary }
        }
        NativeBoundaryRefinementDecision::Reify { reason } => {
            NativeBoundaryExecutionResult::ExactReplay {
                reason,
                detailed_exit: detailed_exit.clone(),
            }
        }
    })
}

impl TypedRegionContract {
    pub fn new(
        contract: TypedObservationContract,
        interval: TransformInterval,
        entry: TypedEventBoundaryRecord,
        internal_effects: Vec<TypedEventTransition>,
        pending_completion_tokens: Vec<TypedPendingEvent>,
        observable_labels: Vec<String>,
        exit: TypedEventBoundaryRecord,
    ) -> Result<Self, RegionContractError> {
        let region = Self {
            schema_version: TRANSFORM_SCHEMA_VERSION,
            contract,
            interval,
            entry,
            internal_effects,
            pending_completion_tokens,
            observable_labels,
            exit,
        };
        region.validate()?;
        Ok(region)
    }

    pub fn validate(&self) -> Result<(), RegionContractError> {
        if self.schema_version != TRANSFORM_SCHEMA_VERSION {
            return Err(RegionContractError::SchemaVersion);
        }
        self.interval
            .validate()
            .map_err(|_| RegionContractError::IntervalReversed)?;
        self.entry
            .validate_structure()
            .map_err(|error| RegionContractError::Boundary {
                side: NativeBoundarySide::Summary,
                error: error.to_string(),
            })?;
        self.exit
            .validate_structure()
            .map_err(|error| RegionContractError::Boundary {
                side: NativeBoundarySide::Detailed,
                error: error.to_string(),
            })?;
        if self.entry.contract != self.contract {
            return Err(RegionContractError::ContractMismatch {
                side: NativeBoundarySide::Summary,
            });
        }
        if self.exit.contract != self.contract {
            return Err(RegionContractError::ContractMismatch {
                side: NativeBoundarySide::Detailed,
            });
        }
        if (self.entry.event.time, self.entry.event.sequence)
            != (self.interval.start_cycle, self.interval.start_sequence)
        {
            return Err(RegionContractError::EntryOutsideInterval);
        }
        if (self.exit.event.time, self.exit.event.sequence)
            != (self.interval.end_cycle, self.interval.end_sequence)
        {
            return Err(RegionContractError::ExitOutsideInterval);
        }
        let mut previous = None;
        for effect in &self.internal_effects {
            effect
                .event
                .sequence_end()
                .map_err(|error| RegionContractError::EffectPayload {
                    sequence: effect.event.sequence,
                    error: error.to_string(),
                })?;
            let key = (effect.event.time, effect.phase, effect.event.sequence);
            if effect.phase != effect.event.phase {
                return Err(RegionContractError::EffectPhaseMismatch {
                    sequence: effect.event.sequence,
                });
            }
            let sequence_in_interval = if self.interval.sequence_ordered {
                effect.event.sequence >= self.interval.start_sequence
                    && effect.event.sequence <= self.interval.end_sequence
            } else {
                let lower = self.interval.start_sequence.min(self.interval.end_sequence);
                let upper = self.interval.start_sequence.max(self.interval.end_sequence);
                effect.event.sequence >= lower && effect.event.sequence <= upper
            };
            if effect.event.time < self.interval.start_cycle
                || effect.event.time > self.interval.end_cycle
                || !sequence_in_interval
            {
                return Err(RegionContractError::EffectOutsideInterval {
                    sequence: effect.event.sequence,
                });
            }
            let out_of_order = previous.is_some_and(|old| {
                if self.interval.sequence_ordered {
                    key < old
                } else {
                    (effect.event.time, effect.phase) < (old.0, old.1)
                        || (effect.event.time, effect.phase) == (old.0, old.1)
                            && effect.event.sequence < old.2
                }
            });
            if out_of_order {
                return Err(RegionContractError::EffectOrder {
                    sequence: effect.event.sequence,
                });
            }
            previous = Some(key);
        }
        if self.pending_completion_tokens != self.exit.queue.pending {
            return Err(RegionContractError::PendingTokenMismatch);
        }
        Ok(())
    }

    pub fn from_summary_proof(proof: &SummaryProof) -> Result<Self, RegionContractError> {
        Self::new(
            proof.contract.clone(),
            proof.interval,
            proof.entry.clone(),
            proof.emitted_events.clone(),
            proof.exit.queue.pending.clone(),
            Vec::new(),
            proof.exit.clone(),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RegionContractError {
    #[error("region contract schema version is unsupported")]
    SchemaVersion,
    #[error("region contract interval is reversed")]
    IntervalReversed,
    #[error("{side:?} region boundary is structurally invalid: {error}")]
    Boundary {
        side: NativeBoundarySide,
        error: String,
    },
    #[error("{side:?} region boundary uses a different typed observation contract")]
    ContractMismatch { side: NativeBoundarySide },
    #[error("region entry boundary is outside the declared interval")]
    EntryOutsideInterval,
    #[error("region exit boundary is outside the declared interval")]
    ExitOutsideInterval,
    #[error("region effect phase disagrees with its typed event phase at sequence {sequence}")]
    EffectPhaseMismatch { sequence: u64 },
    #[error("region effect at sequence {sequence} is outside the declared interval")]
    EffectOutsideInterval { sequence: u64 },
    #[error("region internal effects are not in engine order at sequence {sequence}")]
    EffectOrder { sequence: u64 },
    #[error("region effect payload is malformed at sequence {sequence}: {error}")]
    EffectPayload { sequence: u64, error: String },
    #[error("region pending completion tokens do not equal the exit queue")]
    PendingTokenMismatch,
    #[error("typed scalar macro is unsupported: {reason}")]
    UnsupportedScalarMacro { reason: String },
    #[error("typed scalar macro arithmetic overflowed")]
    ScalarMacroArithmeticOverflow,
}

fn boundary_completeness(
    boundary: &TypedEventBoundaryRecord,
    contract: &TypedObservationContract,
    side: NativeBoundarySide,
) -> Result<(), NativeBoundaryReifyReason> {
    if boundary.contract != *contract {
        return Err(NativeBoundaryReifyReason::ContractMismatch { side });
    }
    match &boundary.completeness {
        Completeness::Complete => {}
        Completeness::Incomplete { unknown } => {
            return Err(NativeBoundaryReifyReason::Incomplete {
                side,
                unknown: unknown.clone(),
            });
        }
        Completeness::Refused { reason } => {
            return Err(NativeBoundaryReifyReason::Refused {
                side,
                reason: reason.clone(),
            });
        }
    }
    match contract.validate(&boundary.shared_l2, &boundary.power, &boundary.cores) {
        Completeness::Complete => Ok(()),
        Completeness::Incomplete { unknown } => {
            Err(NativeBoundaryReifyReason::Incomplete { side, unknown })
        }
        Completeness::Refused { reason } => {
            Err(NativeBoundaryReifyReason::Refused { side, reason })
        }
    }
}

pub fn compare_native_boundary_projection(
    summary_exit: &TypedEventBoundaryRecord,
    detailed_exit: &TypedEventBoundaryRecord,
    contract: &TypedObservationContract,
) -> NativeBoundaryRefinementDecision {
    if let Err(error) = summary_exit.validate_structure() {
        return NativeBoundaryRefinementDecision::Reify {
            reason: NativeBoundaryReifyReason::SummaryStructure {
                error: error.to_string(),
            },
        };
    }
    if let Err(error) = detailed_exit.validate_structure() {
        return NativeBoundaryRefinementDecision::Reify {
            reason: NativeBoundaryReifyReason::DetailedStructure {
                error: error.to_string(),
            },
        };
    }
    if let Err(reason) = boundary_completeness(summary_exit, contract, NativeBoundarySide::Summary)
    {
        return NativeBoundaryRefinementDecision::Reify { reason };
    }
    if let Err(reason) =
        boundary_completeness(detailed_exit, contract, NativeBoundarySide::Detailed)
    {
        return NativeBoundaryRefinementDecision::Reify { reason };
    }
    if summary_exit == detailed_exit {
        NativeBoundaryRefinementDecision::Preserve
    } else {
        NativeBoundaryRefinementDecision::Reify {
            reason: NativeBoundaryReifyReason::ProjectionMismatch,
        }
    }
}

pub fn compare_native_region_contract(
    summary: Option<&TypedRegionContract>,
    detailed_entry: &TypedEventBoundaryRecord,
    detailed_exit: &TypedEventBoundaryRecord,
) -> NativeBoundaryRefinementDecision {
    compare_native_region_contract_with_trace(summary, detailed_entry, &[], &[], detailed_exit)
}

/// Compare a macro contract against the independently observed internal event
/// span and labels as well as both typed boundaries. Endpoint equality alone
/// is not transition refinement.
pub fn compare_native_region_contract_with_trace(
    summary: Option<&TypedRegionContract>,
    detailed_entry: &TypedEventBoundaryRecord,
    detailed_effects: &[TypedEventTransition],
    detailed_labels: &[String],
    detailed_exit: &TypedEventBoundaryRecord,
) -> NativeBoundaryRefinementDecision {
    let Some(summary) = summary else {
        return NativeBoundaryRefinementDecision::Reify {
            reason: NativeBoundaryReifyReason::SummaryStructure {
                error: "model_macro_typed_region_contract_missing".to_owned(),
            },
        };
    };
    if let Err(error) = summary.validate() {
        return NativeBoundaryRefinementDecision::Reify {
            reason: NativeBoundaryReifyReason::SummaryStructure {
                error: error.to_string(),
            },
        };
    }
    if let NativeBoundaryRefinementDecision::Reify { reason } =
        compare_native_boundary_projection(&summary.entry, detailed_entry, &summary.contract)
    {
        return NativeBoundaryRefinementDecision::Reify { reason };
    }
    if summary.internal_effects != detailed_effects {
        let first_difference = summary
            .internal_effects
            .iter()
            .zip(detailed_effects)
            .position(|(expected, observed)| expected != observed)
            .map(|index| {
                format!(
                    "effect_index={index}:expected={:?}:observed={:?}",
                    summary.internal_effects[index], detailed_effects[index]
                )
            });
        return NativeBoundaryRefinementDecision::Reify {
            reason: NativeBoundaryReifyReason::InternalEffectMismatch {
                expected: summary.internal_effects.len(),
                observed: detailed_effects.len(),
                first_difference,
            },
        };
    }
    if summary.observable_labels != detailed_labels {
        return NativeBoundaryRefinementDecision::Reify {
            reason: NativeBoundaryReifyReason::ObservableLabelMismatch,
        };
    }
    compare_native_boundary_projection(&summary.exit, detailed_exit, &summary.contract)
}

pub trait StateTransformer {
    fn decide(&self, input: TransformInput) -> Result<TransformerDecision, TransformError>;
}

/// The mathematical identity: no time advance, no event, and no state change.
/// An interval containing work reifies until a rule derives all of its effects.
#[derive(Debug, Clone, Copy, Default)]
pub struct IdentityBoundaryTransformer;

impl StateTransformer for IdentityBoundaryTransformer {
    fn decide(&self, input: TransformInput) -> Result<TransformerDecision, TransformError> {
        input
            .interval
            .validate()
            .map_err(|_| TransformError::IntervalReversed)?;
        input.entry.validate_structure()?;
        input.exit.validate_structure()?;
        if input.entry.contract != input.contract || input.exit.contract != input.contract {
            return Ok(TransformerDecision::Reify {
                reason: TransformReifyReason::ContractMismatch,
            });
        }
        if !matches!(
            input.contract.validate(
                &input.entry.shared_l2,
                &input.entry.power,
                &input.entry.cores
            ),
            Completeness::Complete
        ) || input.entry.completeness != Completeness::Complete
        {
            return Ok(TransformerDecision::Reify {
                reason: TransformReifyReason::IncompleteEntry,
            });
        }
        if let Err(reason) = input.external_frontier.clone().validate(input.interval) {
            return Ok(TransformerDecision::Reify { reason });
        }
        if input.entry.event.time != input.interval.start_cycle
            || input.entry.event.sequence != input.interval.start_sequence
        {
            return Ok(TransformerDecision::Reify {
                reason: TransformReifyReason::EntryOutsideInterval,
            });
        }
        if input.exit.event.time != input.interval.end_cycle
            || input.exit.event.sequence != input.interval.end_sequence
        {
            return Ok(TransformerDecision::Reify {
                reason: TransformReifyReason::ExitOutsideInterval,
            });
        }
        if !matches!(
            input
                .contract
                .validate(&input.exit.shared_l2, &input.exit.power, &input.exit.cores),
            Completeness::Complete
        ) || input.exit.completeness != Completeness::Complete
        {
            return Ok(TransformerDecision::Reify {
                reason: TransformReifyReason::IncompleteExit,
            });
        }
        if input.interval.start_cycle != input.interval.end_cycle
            || input.interval.start_sequence != input.interval.end_sequence
        {
            return Ok(TransformerDecision::Reify {
                reason: TransformReifyReason::UnsupportedTransition,
            });
        }
        // Derive the output from the entry. Accepting a supplied exit here
        // would permit invented work whenever the endpoints looked valid.
        let exit = input.entry.clone();
        if exit != input.exit {
            return Ok(TransformerDecision::Reify {
                reason: TransformReifyReason::IdentityStateMismatch,
            });
        }
        let emitted_events = Vec::new();
        let observation_delta = ObservationDelta {
            elapsed_cycles: 0,
            emitted_event_count: 0,
        };
        let proof = SummaryProof {
            schema_version: TRANSFORM_SCHEMA_VERSION,
            transformer: IDENTITY_TRANSFORMER.to_owned(),
            interval: input.interval,
            contract: input.contract,
            external_frontier: input.external_frontier,
            entry: input.entry,
            exit: exit.clone(),
            emitted_events: emitted_events.clone(),
            observation_delta,
        };
        Ok(TransformerDecision::Certify(Box::new(TransformerOutput {
            exit,
            emitted_events,
            observation_delta,
            proof,
        })))
    }
}

/// Check the implemented transition rule from the independently observed
/// entry, then compare with the independently executed exit and external
/// frontier and event transitions. A proof's embedded entry/frontier/events
/// are evidence to compare, never the authority that defines the run.
pub fn check_proof(
    proof: &SummaryProof,
    detailed_entry: &TypedEventBoundaryRecord,
    detailed_exit: &TypedEventBoundaryRecord,
    observed_frontier: &ExternalInputFrontier,
    observed_events: &[TypedEventTransition],
) -> Result<(), TransformCheckError> {
    if proof.schema_version != TRANSFORM_SCHEMA_VERSION {
        return Err(TransformCheckError::SchemaVersion);
    }
    if proof.transformer != IDENTITY_TRANSFORMER {
        return Err(TransformCheckError::UnsupportedTransformer);
    }
    proof
        .interval
        .validate()
        .map_err(|_| TransformCheckError::IntervalMismatch)?;
    if proof.entry.event.time != proof.interval.start_cycle
        || proof.entry.event.sequence != proof.interval.start_sequence
        || proof.exit.event.time != proof.interval.end_cycle
        || proof.exit.event.sequence != proof.interval.end_sequence
    {
        return Err(TransformCheckError::IntervalMismatch);
    }
    if proof.observation_delta.elapsed_cycles
        != proof.interval.end_cycle - proof.interval.start_cycle
    {
        return Err(TransformCheckError::ObservationDeltaMismatch);
    }
    if proof
        .external_frontier
        .clone()
        .validate(proof.interval)
        .is_err()
    {
        return Err(TransformCheckError::ExternalFrontierMismatch);
    }
    proof.entry.validate_structure()?;
    proof.exit.validate_structure()?;
    detailed_entry.validate_structure()?;
    detailed_exit.validate_structure()?;
    if &proof.entry != detailed_entry {
        return Err(TransformCheckError::EntryMismatch);
    }
    if &proof.external_frontier != observed_frontier {
        return Err(TransformCheckError::FrontierEvidenceMismatch);
    }
    if proof.entry.contract != proof.contract || proof.entry.completeness != Completeness::Complete
    {
        return Err(TransformCheckError::ContractMismatch);
    }
    if proof.contract.validate(
        &proof.entry.shared_l2,
        &proof.entry.power,
        &proof.entry.cores,
    ) != Completeness::Complete
    {
        return Err(TransformCheckError::ContractMismatch);
    }
    // Recompute the identity rule rather than trusting proof.exit or the
    // reference argument. Agreement between two supplied values is insufficient.
    if proof.interval.start_cycle != proof.interval.end_cycle
        || proof.interval.start_sequence != proof.interval.end_sequence
        || proof.exit != proof.entry
    {
        return Err(TransformCheckError::TransitionMismatch);
    }
    if proof.emitted_events.as_slice() != observed_events {
        return Err(TransformCheckError::EventTraceMismatch);
    }
    if proof.observation_delta.emitted_event_count != proof.emitted_events.len() as u64 {
        return Err(TransformCheckError::EventCountMismatch);
    }
    if !proof.emitted_events.is_empty() {
        return Err(TransformCheckError::TransitionMismatch);
    }
    TypedRegionContract::from_summary_proof(proof).map_err(TransformCheckError::RegionContract)?;
    if let NativeBoundaryRefinementDecision::Reify { reason } =
        compare_native_boundary_projection(&proof.exit, detailed_exit, &proof.contract)
    {
        return Err(TransformCheckError::NativeBoundaryReify { reason });
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransformReifyReason {
    ContractMismatch,
    UnsupportedTransition,
    IdentityStateMismatch,
    EntryOutsideInterval,
    IncompleteEntry,
    IncompleteExit,
    OpenExternalInput,
    ExternalInputInsideInterval,
    ExitOutsideInterval,
}

#[derive(Debug, thiserror::Error)]
pub enum TransformError {
    #[error("transform interval is reversed")]
    IntervalReversed,
    #[error("typed boundary structure is invalid: {0}")]
    Boundary(#[from] crate::BoundaryStateError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TransformIntervalError {
    #[error("transform interval is reversed")]
    Reversed,
}

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum TransformCheckError {
    #[error("summary proof schema version is unsupported")]
    SchemaVersion,
    #[error("summary proof names a transformer without an implemented checking rule")]
    UnsupportedTransformer,
    #[error("summary exit cannot be derived from its entry by the declared rule")]
    TransitionMismatch,
    #[error("summary proof interval does not cover its entry/exit events")]
    IntervalMismatch,
    #[error("summary proof external frontier is open or intersects its interval")]
    ExternalFrontierMismatch,
    #[error("summary proof contract is incomplete for its entry or exit")]
    ContractMismatch,
    #[error("summary proof observation delta does not match its interval")]
    ObservationDeltaMismatch,
    #[error(transparent)]
    Boundary(#[from] crate::BoundaryStateError),
    #[error("summary proof exit does not equal the detailed exit boundary")]
    ExitMismatch,
    #[error("native boundary refinement requires exact replay: {reason:?}")]
    NativeBoundaryReify { reason: NativeBoundaryReifyReason },
    #[error("typed region contract is invalid: {0}")]
    RegionContract(#[from] RegionContractError),
    #[error("summary proof entry does not equal the independently observed entry boundary")]
    EntryMismatch,
    #[error("summary proof frontier does not equal the independently observed frontier")]
    FrontierEvidenceMismatch,
    #[error("summary proof emitted-event count disagrees with its observation delta")]
    EventCountMismatch,
    #[error("summary proof emitted events differ from the independently observed transition trace")]
    EventTraceMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AddressDomain, BlockBoundaryState, L2BoundaryState, L2CounterState, L2GeometryState,
        PowerBoundaryState, QueueBoundaryState, TypedContractKind, TypedCoreBoundaryState,
        TypedEventPayload, WarpBoundaryState,
    };

    fn boundary(time: u64, sequence: u64, complete: bool) -> TypedEventBoundaryRecord {
        let l2 = L2BoundaryState {
            schema_version: crate::TYPED_STATE_SCHEMA_VERSION,
            architecture: "sm_86".to_owned(),
            geometry: L2GeometryState {
                architecture: "sm_86".to_owned(),
                capacity_bytes: 128,
                line_bytes: 128,
                sector_bytes: 32,
                num_slices: 1,
                sets_per_slice: 1,
                associativity: 1,
                replacement: "lru".to_owned(),
                slice_hash: "line_mod".to_owned(),
            },
            address_domain: AddressDomain::VirtualAbsolute,
            clock: i64::try_from(time).unwrap_or(i64::MAX),
            bip_counter: 0,
            access_discipline: None,
            counters: L2CounterState {
                hits: 0,
                misses: 0,
                lookup_hits: 0,
                lookup_requests: 0,
                line_misses: 0,
                sector_fill_misses: 0,
                write_tag_hits: 0,
            },
            observer_lookup_hits: 0,
            observer_lookup_requests: 0,
            next_generation: None,
            sets: Vec::new(),
            allocations: Vec::new(),
            request_owners: Vec::new(),
            ownership_complete: complete,
            unknown_state: if complete {
                Vec::new()
            } else {
                vec!["unknown".to_owned()]
            },
        };
        let mut power = PowerBoundaryState::activity_only(None, None);
        power.domains[0].last_cycle = time;
        TypedEventBoundaryRecord {
            schema_version: crate::TYPED_STATE_SCHEMA_VERSION,
            contract: TypedObservationContract {
                schema_version: crate::TYPED_STATE_SCHEMA_VERSION,
                kind: TypedContractKind::CycleExactPowerV2,
                address_domain: AddressDomain::VirtualAbsolute,
                power: crate::PowerModelStatus::ActivityOnly,
            },
            event: TypedEventIdentity {
                time,
                sequence,
                sequence_span: 1,
                phase: 0,
                target: 0,
                payload: TypedEventPayload::SmWake,
            },
            queue: QueueBoundaryState {
                depth: 0,
                next_event_time: None,
                next_sequence: sequence.checked_add(1),
                pending: Vec::new(),
            },
            cores: vec![TypedCoreBoundaryState {
                sm_id: 0,
                cycle: time,
                issued: 0,
                resident_warps: 0,
                resident_shared_memory_bytes: 0,
                unfinished_warps: 0,
                next_block: 0,
                hold: [None; 4],
                subcore_warps: vec![Vec::new(); 4],
                outstanding_responses: 0,
                barrier_pipe_ready: 0,
                tensor_pipe_ready: [0; 4],
                ldgsts_destination_pipe_ready: 0,
                warp_states: Vec::new(),
                block_states: Vec::new(),
                l1: None,
                front_end: None,
                memory_pipeline: None,
                scalar_pipeline: None,
                local_completion_time: None,
                unprojected_timing_state: Vec::new(),
                async_copy: Vec::new(),
                response_owners: Vec::new(),
            }],
            shared_l2: l2,
            dram: crate::DramBoundaryState::LatencyOnly,
            power,
            completeness: if complete {
                Completeness::Complete
            } else {
                Completeness::Incomplete {
                    unknown: vec!["unknown".to_owned()],
                }
            },
        }
    }

    fn scalar_entry_boundary() -> TypedEventBoundaryRecord {
        let mut value = boundary(0, 0, true);
        let core = &mut value.cores[0];
        core.resident_warps = 1;
        core.unfinished_warps = 1;
        core.next_block = 1;
        core.subcore_warps = vec![vec![0], Vec::new(), Vec::new(), Vec::new()];
        core.warp_states = vec![WarpBoundaryState {
            warp_slot: 0,
            next_instruction: 0,
            front_end_fetch_cursor: 0,
            instruction_buffer: Vec::new(),
            eligible: 0,
            imc_wait_until: 0,
            barrier_clear: [0; 6],
            barrier_pending: [0; 6],
            block: 0,
            arrival: 0,
            at_block_barrier: false,
            cta_barrier_ready: 0,
            barrier_pipeline_ready: 0,
            branch_pipeline_ready: 0,
            block_barrier_dispatch: None,
            block_barrier_arrival: None,
            done: false,
        }];
        core.block_states = vec![BlockBoundaryState {
            unfinished: 1,
            waiting: 0,
            warp_count: 1,
            shared_memory_bytes: 0,
            warp_slots: vec![0],
            first_warp: 0,
            admitted_warps: 1,
            barrier_generation: 0,
            barrier_first_dispatch: None,
            barrier_first_arrival: None,
            barrier_rearm_ready: 0,
        }];
        core.async_copy = vec![None];
        value
            .validate_structure()
            .expect("scalar entry should be closed");
        value
    }

    #[test]
    fn identity_transformer_reifies_incomplete_entry_without_guessing() {
        let input = TransformInput {
            entry: boundary(0, 0, false),
            exit: boundary(1, 1, true),
            external_frontier: ExternalInputFrontier {
                next_event_time: None,
                closed_world: true,
                unknown: Vec::new(),
            },
            contract: TypedObservationContract::cycle_exact_power(),
            interval: TransformInterval {
                start_cycle: 0,
                end_cycle: 1,
                start_sequence: 0,
                end_sequence: 1,
                sequence_ordered: false,
            },
        };
        assert_eq!(
            IdentityBoundaryTransformer
                .decide(input)
                .expect("test fixture should be valid"),
            TransformerDecision::Reify {
                reason: TransformReifyReason::IncompleteEntry
            }
        );
    }

    #[test]
    fn interval_keeps_time_and_insertion_sequence_as_separate_coordinates() {
        let reordered = TransformInterval {
            start_cycle: 5,
            end_cycle: 10,
            start_sequence: 9,
            end_sequence: 3,
            sequence_ordered: false,
        };
        assert!(reordered.validate().is_ok());
        assert!(
            TransformInterval {
                sequence_ordered: true,
                ..reordered
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn same_cycle_effects_still_require_engine_insertion_order() {
        let entry = boundary(5, 9, true);
        let exit = boundary(10, 3, true);
        let mut later = entry.event.clone();
        later.sequence = 8;
        let result = TypedRegionContract::new(
            TypedObservationContract::cycle_exact_power(),
            TransformInterval {
                start_cycle: 5,
                end_cycle: 10,
                start_sequence: 9,
                end_sequence: 3,
                sequence_ordered: false,
            },
            entry.clone(),
            vec![
                TypedEventTransition {
                    phase: 0,
                    event: entry.event,
                },
                TypedEventTransition {
                    phase: 0,
                    event: later,
                },
            ],
            Vec::new(),
            Vec::new(),
            exit,
        );
        assert!(matches!(
            result,
            Err(RegionContractError::EffectOrder { .. })
        ));
    }

    #[test]
    fn checker_rejects_a_corrupted_exit_state() {
        let mut proof = SummaryProof {
            schema_version: TRANSFORM_SCHEMA_VERSION,
            transformer: IDENTITY_TRANSFORMER.to_owned(),
            interval: TransformInterval {
                start_cycle: 0,
                end_cycle: 1,
                start_sequence: 0,
                end_sequence: 1,
                sequence_ordered: false,
            },
            contract: TypedObservationContract::cycle_exact_power(),
            external_frontier: ExternalInputFrontier {
                next_event_time: None,
                closed_world: true,
                unknown: Vec::new(),
            },
            entry: boundary(0, 0, true),
            exit: boundary(1, 1, true),
            emitted_events: vec![TypedEventTransition {
                phase: 0,
                event: boundary(1, 1, true).event,
            }],
            observation_delta: ObservationDelta {
                elapsed_cycles: 1,
                emitted_event_count: 1,
            },
        };
        proof.exit.power.domains[0].clocked_cycles = 1;
        assert!(matches!(
            check_proof(
                &proof,
                &boundary(0, 0, true),
                &boundary(1, 1, true),
                &ExternalInputFrontier {
                    next_event_time: None,
                    closed_world: true,
                    unknown: Vec::new(),
                },
                proof.emitted_events.as_slice(),
            ),
            Err(TransformCheckError::TransitionMismatch)
        ));
    }

    #[test]
    fn identity_transformer_emits_a_checkable_complete_proof() {
        let input = TransformInput {
            entry: boundary(0, 0, true),
            exit: boundary(0, 0, true),
            external_frontier: ExternalInputFrontier {
                next_event_time: None,
                closed_world: true,
                unknown: Vec::new(),
            },
            contract: TypedObservationContract::cycle_exact_power(),
            interval: TransformInterval {
                start_cycle: 0,
                end_cycle: 0,
                start_sequence: 0,
                end_sequence: 0,
                sequence_ordered: false,
            },
        };
        let TransformerDecision::Certify(output) = IdentityBoundaryTransformer
            .decide(input)
            .expect("test fixture should be valid")
        else {
            panic!("complete typed boundaries should certify");
        };
        assert_eq!(
            check_proof(
                &output.proof,
                &output.proof.entry,
                &output.exit,
                &output.proof.external_frontier,
                &[],
            ),
            Ok(())
        );
    }

    #[test]
    fn native_boundary_projection_preserves_equal_complete_state() {
        let value = boundary(0, 0, true);
        assert_eq!(
            compare_native_boundary_projection(
                &value,
                &value,
                &TypedObservationContract::cycle_exact_power(),
            ),
            NativeBoundaryRefinementDecision::Preserve
        );
    }

    #[test]
    fn native_boundary_projection_reifies_unknown_detailed_state() {
        let summary = boundary(0, 0, true);
        let mut detailed = summary.clone();
        detailed.shared_l2.ownership_complete = false;
        detailed.shared_l2.unknown_state = vec!["unmodeled_completion_tails".to_owned()];
        detailed.completeness = Completeness::Incomplete {
            unknown: vec!["unmodeled_completion_tails".to_owned()],
        };
        assert!(matches!(
            compare_native_boundary_projection(
                &summary,
                &detailed,
                &TypedObservationContract::cycle_exact_power(),
            ),
            NativeBoundaryRefinementDecision::Reify {
                reason: NativeBoundaryReifyReason::Incomplete {
                    side: NativeBoundarySide::Detailed,
                    ..
                }
            }
        ));
    }

    #[test]
    fn missing_typed_macro_region_forces_exact_replay() {
        let value = boundary(0, 0, true);
        assert_eq!(
            compare_native_region_contract(None, &value, &value),
            NativeBoundaryRefinementDecision::Reify {
                reason: NativeBoundaryReifyReason::SummaryStructure {
                    error: "model_macro_typed_region_contract_missing".to_owned(),
                },
            }
        );
    }

    #[test]
    fn malformed_internal_effect_payload_forces_exact_replay() {
        let entry = boundary(0, 0, true);
        let mut malformed = boundary(1, 1, true).event;
        malformed.sequence_span = 0;
        let result = TypedRegionContract::new(
            TypedObservationContract::cycle_exact_power(),
            TransformInterval {
                start_cycle: 0,
                end_cycle: 1,
                start_sequence: 0,
                end_sequence: 1,
                sequence_ordered: false,
            },
            entry,
            vec![TypedEventTransition {
                phase: 0,
                event: malformed,
            }],
            Vec::new(),
            Vec::new(),
            boundary(1, 1, true),
        );
        assert!(matches!(
            result,
            Err(RegionContractError::EffectPayload { .. })
        ));
    }

    #[test]
    fn native_region_comparator_checks_internal_effects_and_labels() {
        let entry = boundary(0, 0, true);
        let exit = boundary(1, 1, true);
        let effect = TypedEventTransition {
            phase: 0,
            event: exit.event.clone(),
        };
        let region = TypedRegionContract::new(
            TypedObservationContract::cycle_exact_power(),
            TransformInterval {
                start_cycle: 0,
                end_cycle: 1,
                start_sequence: 0,
                end_sequence: 1,
                sequence_ordered: false,
            },
            entry.clone(),
            vec![effect.clone()],
            Vec::new(),
            vec!["scalar_completion".to_owned()],
            exit.clone(),
        )
        .unwrap();
        assert!(matches!(
            compare_native_region_contract_with_trace(
                Some(&region),
                &entry,
                &[],
                &["scalar_completion".to_owned()],
                &exit,
            ),
            NativeBoundaryRefinementDecision::Reify {
                reason: NativeBoundaryReifyReason::InternalEffectMismatch { .. }
            }
        ));
        assert_eq!(
            compare_native_region_contract_with_trace(
                Some(&region),
                &entry,
                &[effect],
                &["scalar_completion".to_owned()],
                &exit,
            ),
            NativeBoundaryRefinementDecision::Preserve
        );
        assert!(matches!(
            native_boundary_execution_path(&NativeBoundaryRefinementDecision::Reify {
                reason: NativeBoundaryReifyReason::ProjectionMismatch,
            }),
            NativeBoundaryExecutionPath::ExactReplay { .. }
        ));
        assert_eq!(
            native_boundary_execution_path(&NativeBoundaryRefinementDecision::Preserve),
            NativeBoundaryExecutionPath::TypedMacro
        );
    }

    #[test]
    fn typed_scalar_macro_derives_exit_and_effect_span_independently() {
        let entry = scalar_entry_boundary();
        let region = derive_typed_scalar_macro_region(
            &entry,
            &TypedScalarMacroSpec {
                instruction_count: 2,
                latency_cycles: 3,
                initiation_interval_cycles: 3,
                label: "scalar_macro_v1".to_owned(),
            },
        )
        .expect("closed scalar entry should derive a typed transition");
        assert_eq!(region.internal_effects.len(), 2);
        assert_eq!(region.exit.event.time, 4);
        assert_eq!(region.exit.event.sequence, 3);
        assert_eq!(region.exit.cores[0].issued, 2);
        assert_eq!(region.exit.cores[0].warp_states[0].next_instruction, 2);
        assert!(matches!(
            compare_native_region_contract_with_trace(
                Some(&region),
                &region.entry,
                &region.internal_effects,
                &region.observable_labels,
                &region.exit,
            ),
            NativeBoundaryRefinementDecision::Preserve
        ));
        assert!(matches!(
            execute_typed_scalar_macro(
                &entry,
                &TypedScalarMacroSpec {
                    instruction_count: 2,
                    latency_cycles: 3,
                    initiation_interval_cycles: 3,
                    label: "scalar_macro_v1".to_owned(),
                },
                &region.internal_effects,
                &region.observable_labels,
                &region.exit,
            )
            .unwrap(),
            NativeBoundaryExecutionResult::TypedMacro { contract: _ }
        ));
        assert!(matches!(
            execute_typed_scalar_macro(
                &entry,
                &TypedScalarMacroSpec {
                    instruction_count: 2,
                    latency_cycles: 3,
                    initiation_interval_cycles: 3,
                    label: "scalar_macro_v1".to_owned(),
                },
                &[],
                &region.observable_labels,
                &region.exit,
            )
            .unwrap(),
            NativeBoundaryExecutionResult::ExactReplay { .. }
        ));
    }

    #[test]
    fn typed_scalar_macro_refuses_pending_response_barrier_and_unknown_tail_entries() {
        let spec = TypedScalarMacroSpec {
            instruction_count: 1,
            latency_cycles: 2,
            initiation_interval_cycles: 2,
            label: "scalar_macro_v1".to_owned(),
        };
        let mut pending = scalar_entry_boundary();
        pending.queue.depth = 1;
        pending.queue.next_event_time = Some(0);
        pending.queue.next_sequence = Some(2);
        pending.queue.pending.push(TypedPendingEvent {
            time: 0,
            sequence: 1,
            phase: 0,
            sequence_span: 1,
            target: 0,
            payload: TypedEventPayload::MemoryResponse {
                reply: Some(crate::TypedResponseIdentity {
                    warp: 0,
                    instruction: 0,
                    sector_index: 0,
                    response_kind: crate::TypedResponseKind::Warp,
                }),
                fill: None,
            },
        });
        let pending_result = derive_typed_scalar_macro_region(&pending, &spec);
        assert!(matches!(
            pending_result,
            Err(RegionContractError::Boundary { .. })
                | Err(RegionContractError::UnsupportedScalarMacro { .. })
        ));

        let mut barrier = scalar_entry_boundary();
        barrier.cores[0].warp_states[0].barrier_pending[0] = 1;
        assert!(matches!(
            derive_typed_scalar_macro_region(&barrier, &spec),
            Err(RegionContractError::Boundary { .. })
                | Err(RegionContractError::UnsupportedScalarMacro { .. })
        ));

        let mut unknown = scalar_entry_boundary();
        unknown.completeness = Completeness::Incomplete {
            unknown: vec!["unmodeled_completion_tails".to_owned()],
        };
        assert!(matches!(
            derive_typed_scalar_macro_region(&unknown, &spec),
            Err(RegionContractError::Boundary { .. })
                | Err(RegionContractError::UnsupportedScalarMacro { .. })
        ));
    }

    #[test]
    fn checker_requires_independent_entry_and_frontier_evidence() {
        let input = TransformInput {
            entry: boundary(0, 0, true),
            exit: boundary(0, 0, true),
            external_frontier: ExternalInputFrontier {
                next_event_time: None,
                closed_world: true,
                unknown: Vec::new(),
            },
            contract: TypedObservationContract::cycle_exact_power(),
            interval: TransformInterval {
                start_cycle: 0,
                end_cycle: 0,
                start_sequence: 0,
                end_sequence: 0,
                sequence_ordered: false,
            },
        };
        let TransformerDecision::Certify(output) = IdentityBoundaryTransformer
            .decide(input)
            .expect("test fixture should be valid")
        else {
            panic!("zero-length identity should certify");
        };
        let mut actual_entry = output.proof.entry.clone();
        actual_entry.cores[0].barrier_pipe_ready = 1;
        assert!(matches!(
            check_proof(
                &output.proof,
                &actual_entry,
                &output.exit,
                &output.proof.external_frontier,
                &[],
            ),
            Err(TransformCheckError::EntryMismatch)
        ));
        let observed_frontier = ExternalInputFrontier {
            next_event_time: None,
            closed_world: false,
            unknown: vec!["unobserved_dma".to_owned()],
        };
        assert!(matches!(
            check_proof(
                &output.proof,
                &output.proof.entry,
                &output.exit,
                &observed_frontier,
                &[],
            ),
            Err(TransformCheckError::FrontierEvidenceMismatch)
        ));
        let omitted_event = TypedEventTransition {
            phase: 0,
            event: output.proof.entry.event.clone(),
        };
        assert!(matches!(
            check_proof(
                &output.proof,
                &output.proof.entry,
                &output.exit,
                &output.proof.external_frontier,
                &[omitted_event],
            ),
            Err(TransformCheckError::EventTraceMismatch)
        ));
    }

    #[test]
    fn same_cycle_external_frontier_reifies_without_phase_guessing() {
        let input = TransformInput {
            entry: boundary(0, 0, true),
            exit: boundary(1, 1, true),
            external_frontier: ExternalInputFrontier {
                next_event_time: Some(1),
                closed_world: true,
                unknown: Vec::new(),
            },
            contract: TypedObservationContract::cycle_exact_power(),
            interval: TransformInterval {
                start_cycle: 0,
                end_cycle: 1,
                start_sequence: 0,
                end_sequence: 1,
                sequence_ordered: false,
            },
        };
        assert_eq!(
            IdentityBoundaryTransformer
                .decide(input)
                .expect("test fixture should be valid"),
            TransformerDecision::Reify {
                reason: TransformReifyReason::ExternalInputInsideInterval
            }
        );
    }

    #[test]
    fn identity_cannot_certify_arbitrary_work_from_a_supplied_exit() {
        let mut exit = boundary(1, 1, true);
        exit.cores[0].barrier_pipe_ready = 1;
        let input = TransformInput {
            entry: boundary(0, 0, true),
            exit,
            external_frontier: ExternalInputFrontier {
                next_event_time: None,
                closed_world: true,
                unknown: Vec::new(),
            },
            contract: TypedObservationContract::cycle_exact_power(),
            interval: TransformInterval {
                start_cycle: 0,
                end_cycle: 1,
                start_sequence: 0,
                end_sequence: 1,
                sequence_ordered: false,
            },
        };
        assert!(matches!(
            IdentityBoundaryTransformer
                .decide(input)
                .expect("valid input"),
            TransformerDecision::Reify { .. }
        ));
    }
}
