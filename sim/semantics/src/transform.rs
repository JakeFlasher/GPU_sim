//! Contract-aware typed boundary transformations.
//!
//! A certificate must derive its exit from the entry using a known transition
//! rule. Structural validity and equality with a supplied exit do not prove a
//! transition. The identity rule implemented here consumes no cycles or events
//! and leaves every field unchanged. Advancing native SM execution requires a
//! different rule and its corresponding checker.

use serde::{Deserialize, Serialize};

use crate::{Completeness, TypedEventBoundaryRecord, TypedEventIdentity, TypedObservationContract};

pub const TRANSFORM_SCHEMA_VERSION: u32 = 2;
const IDENTITY_TRANSFORMER: &str = "identity_boundary_v2";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransformInterval {
    pub start_cycle: u64,
    pub end_cycle: u64,
    pub start_sequence: u64,
    pub end_sequence: u64,
}

impl TransformInterval {
    pub fn validate(self) -> Result<(), TransformIntervalError> {
        if self.end_cycle < self.start_cycle || self.end_sequence < self.start_sequence {
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
    if proof.entry.contract != proof.contract
        || proof.exit.contract != proof.contract
        || detailed_exit.contract != proof.contract
        || proof.entry.completeness != Completeness::Complete
        || proof.exit.completeness != Completeness::Complete
    {
        return Err(TransformCheckError::ContractMismatch);
    }
    if proof.contract.validate(
        &proof.entry.shared_l2,
        &proof.entry.power,
        &proof.entry.cores,
    ) != Completeness::Complete
        || proof
            .contract
            .validate(&proof.exit.shared_l2, &proof.exit.power, &proof.exit.cores)
            != Completeness::Complete
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
    if &proof.exit != detailed_exit {
        return Err(TransformCheckError::ExitMismatch);
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
        AddressDomain, L2BoundaryState, L2CounterState, L2GeometryState, PowerBoundaryState,
        QueueBoundaryState, TypedContractKind, TypedCoreBoundaryState, TypedEventPayload,
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
    fn checker_rejects_a_corrupted_exit_state() {
        let mut proof = SummaryProof {
            schema_version: TRANSFORM_SCHEMA_VERSION,
            transformer: IDENTITY_TRANSFORMER.to_owned(),
            interval: TransformInterval {
                start_cycle: 0,
                end_cycle: 1,
                start_sequence: 0,
                end_sequence: 1,
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
