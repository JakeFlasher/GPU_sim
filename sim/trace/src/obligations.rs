//! Evidence-separated asynchronous obligation analysis.
//!
//! This module is deliberately an analysis projection, not a replay timing
//! backend. It reconstructs obligations visible in real sidecars, keeps
//! generation/lifetime alternatives correlated, and reports what a consumer
//! frontier establishes without inventing an engine clock. A failed phase
//! query proves `arrival_balance_nonzero OR transaction_balance_nonzero`; it
//! does not prove either individual literal is the unique blocker.

use crate::async_sidecars::{
    BarrierOperation, LaunchObservations, ObservationKey, PhaseAssociation, PhaseQueryKind,
    PhaseReadinessObservation,
};
use serde::Serialize;
use std::collections::BTreeMap;

/// A bounded analysis budget. This is a software precision limit, never a
/// claimed hardware capacity.
pub const FRONTIER_ALTERNATIVE_BUDGET: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FrontierCertainty {
    ObservedSatisfaction,
    NecessaryObligation,
    PossibleObligation,
    Unresolved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PollingView {
    CapturedProgram,
    QualifiedSynchronization,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FrontierPosition {
    BeforeQueryIssue,
    AfterQueryResult,
    BeforePollLoopExit,
    BeforeAcquireEndpoint,
    BeforeDependentUse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisPerspective {
    CapturedPrefix,
    RetrospectiveObservation,
    ReplayScenario,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RequirementStatus {
    Satisfied,
    Unsatisfied,
    Undetermined,
    InterpretationConflict,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FrontierDerivation {
    EarlierAccounting,
    CurrentQuery,
    JointInference,
    RetrospectiveEvidence,
    NoResult,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockerLiteral {
    ArrivalBalanceNonzero,
    TransactionBalanceNonzero,
    AsyncGroupPrefixOutstanding,
    ProducerMembershipUnresolved,
    BarrierLifetimeUnresolved,
    QueryResultUnavailable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "operator", content = "literals")]
pub enum BlockerExpression {
    AnyOf(Vec<BlockerLiteral>),
    AllOf(Vec<BlockerLiteral>),
    Unknown(&'static str),
}

/// A deliberately joint alternative set. The producer and phase vectors are
/// not an implicit Cartesian product and no pairing is selected from file
/// order or parity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProducerPhaseCorrelation {
    pub pairing: ProducerPhasePairing,
    pub association: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ProducerPhasePairing {
    Exact(Vec<ProducerPhasePair>),
    UnresolvedJoint {
        producer_candidates: Vec<ObservationKey>,
        phase_candidates: Vec<u64>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProducerPhasePair {
    pub producer: ObservationKey,
    pub phase: u64,
    pub guard: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AccountingEvent {
    pub instruction: ObservationKey,
    pub operation: BarrierOperation,
    pub lane_mask: u32,
    pub transaction_operand_bytes: Option<u64>,
    pub arrival_count_operand: Option<u64>,
    pub pending_arrival_delta: Option<i64>,
    pub next_phase_arrival_delta: Option<i64>,
    pub transaction_delta_bytes: Option<i64>,
    pub effect_basis: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeferredArrivalObligation {
    pub marker: ObservationKey,
    pub source_candidates: Vec<ObservationKey>,
    pub source_membership: &'static str,
    pub effect_at_marker: &'static str,
    pub discharge_basis: &'static str,
}

/// One destination-barrier ledger entry. Balances are observed net deltas
/// from captured forms, not absolute hardware state: initial counts and
/// hidden engine accounting may be absent from a trace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BarrierLedger {
    pub barrier_cta: [u32; 3],
    pub barrier_offset: u32,
    pub barrier_address_raw: Option<u32>,
    pub barrier_cluster_rank: Option<u32>,
    pub lifetime: &'static str,
    pub generation: Option<u64>,
    pub operations: Vec<BarrierOperation>,
    pub observed_pending_arrival_delta: i64,
    pub observed_next_phase_arrival_delta: i64,
    pub observed_transaction_delta_bytes: i64,
    pub transaction_balance_is_signed: bool,
    pub producer_candidates: Vec<ObservationKey>,
    pub requested_phases: Vec<u64>,
    pub producer_phase_correlations: Vec<ProducerPhaseCorrelation>,
    pub accounting_events: Vec<AccountingEvent>,
    pub deferred_arrivals: Vec<DeferredArrivalObligation>,
    pub producer_lane_mask: u32,
    pub executed_lane_mask: u32,
    pub arrival_lane_mask: u32,
    pub transaction_lane_mask: u32,
    pub query_lane_mask: u32,
    pub exchange_token_records: u64,
    pub exchange_token_lane_mask: u32,
    pub producer_membership: &'static str,
    pub phase_query_instructions: u64,
    pub successful_query_instructions: u64,
    pub unsuccessful_query_instructions: u64,
    pub async_copy_arrivals_separate: u64,
    pub interpretation_conflicts: u64,
    /// Raw identity or operand coverage is incomplete; no exact witness is
    /// emitted from this ledger even when its offset-only candidate set is
    /// useful for diagnostics.
    pub identity_or_operand_precision_lost: bool,
    pub precision_lost: bool,
}

/// A causal consumer frontier at one observed query/result position.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ConsumerFrontier {
    pub consumer: ObservationKey,
    pub barrier_cta: Option<[u32; 3]>,
    pub barrier_offset: Option<u32>,
    pub barrier_address_raw: Option<u32>,
    pub barrier_cluster_rank: Option<u32>,
    pub requested_phase: Option<u64>,
    pub requested_phases: Vec<u64>,
    pub query_kind: Option<PhaseQueryKind>,
    pub query_evaluation_observed: bool,
    pub query_evaluation_instruction: usize,
    pub query_result_lane_mask: u32,
    pub query_result_mixed: bool,
    pub callback_sample_clock: Option<u64>,
    pub callback_phase_state: &'static str,
    pub position: FrontierPosition,
    pub perspective: AnalysisPerspective,
    pub derivation: FrontierDerivation,
    pub requirement_expression: &'static str,
    pub pre_result_status: RequirementStatus,
    pub post_result_status: RequirementStatus,
    pub pre_result_blocker_expression: Option<BlockerExpression>,
    pub requirement_status: RequirementStatus,
    pub mandatory_blockers: Vec<BlockerLiteral>,
    pub possible_blockers: Vec<BlockerLiteral>,
    pub blocker_expression: Option<BlockerExpression>,
    pub supporting_events: Vec<ObservationKey>,
    pub unresolved_reasons: Vec<&'static str>,
    pub captured_polling: bool,
    pub qualified_synchronization_applied: bool,
    pub engine_completion_clock: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ObligationFrontier {
    pub polling_view: PollingView,
    pub tma_load_obligations: u64,
    pub tma_store_obligations: u64,
    pub expected_arrival_obligations: u64,
    pub expected_transaction_bytes: u64,
    pub captured_query_instructions: u64,
    pub captured_successful_queries: u64,
    pub captured_unsuccessful_queries: u64,
    pub captured_polling_is_not_counterfactual: bool,
    pub qualified_synchronization_available: bool,
    pub qualified_synchronization_applied: bool,
    pub qualified_synchronization_reason: &'static str,
    pub qualified_phase_observations: u64,
    pub protocol_scoped_phase_candidates: u64,
    pub qualified_satisfied_generations: u64,
    pub possible_unresolved_generations: u64,
    /// Protocol requirements, not individually identified blockers.
    pub mandatory_obligations: Vec<&'static str>,
    pub possible_obligations: Vec<&'static str>,
    pub unresolved_obligations: Vec<&'static str>,
    pub ledgers: Vec<BarrierLedger>,
    pub consumer_frontiers: Vec<ConsumerFrontier>,
    pub consumer_obligations: Vec<ConsumerObligation>,
    pub frontier_alternative_budget: usize,
    pub frontier_precision_lost: bool,
    pub interpretation_conflict_frontiers: u64,
    pub pre_result_frontiers: u64,
    pub post_result_narrowed_frontiers: u64,
    pub current_query_only_frontiers: u64,
    pub deferred_arrival_obligations: u64,
    pub engine_completion_clock: &'static str,
    pub model_timing_input: &'static str,
}

/// One observed consumer-facing relation. Multiple records can refer to the
/// same producer contribution; bytes are never summed once per observer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ConsumerObligation {
    pub producer: ObservationKey,
    pub query: ObservationKey,
    pub barrier_cta: [u32; 3],
    pub barrier_offset: u32,
    pub barrier_address_raw: Option<u32>,
    pub barrier_cluster_rank: Option<u32>,
    pub payload_bytes: u64,
    pub producer_association: FrontierCertainty,
    /// These fields mean “required by the architectural phase expression”,
    /// not “currently the unique unsatisfied blocker”.
    pub software_arrival: FrontierCertainty,
    pub transaction_bytes: FrontierCertainty,
    pub phase_satisfaction: FrontierCertainty,
    pub query_clock_basis: &'static str,
    pub engine_completion_clock: &'static str,
}

type Barrier = ([u32; 3], u32, Option<u32>, Option<u32>);

fn add_i64(target: &mut i64, value: i64, precision_lost: &mut bool) {
    if let Some(sum) = target.checked_add(value) {
        *target = sum;
    } else {
        *precision_lost = true;
    }
}

fn push_candidate(
    candidates: &mut Vec<ObservationKey>,
    key: ObservationKey,
    precision_lost: &mut bool,
) {
    if candidates.contains(&key) {
        return;
    }
    if candidates.len() >= FRONTIER_ALTERNATIVE_BUDGET {
        *precision_lost = true;
        return;
    }
    candidates.push(key);
}

fn signed_count(value: Option<u64>) -> Option<i64> {
    value.and_then(|value| {
        i64::try_from(value)
            .ok()
            .and_then(|value| value.checked_neg())
    })
}

fn accounting_effect(
    operation: BarrierOperation,
    observation: &crate::async_sidecars::SyncObservation,
) -> (Option<i64>, Option<i64>, Option<i64>, &'static str) {
    match operation {
        BarrierOperation::ArriveExpectTransaction => (
            Some(-1),
            None,
            observation.transaction_delta_bytes,
            "arrive plus transaction expectation; exact participant multiplicity retained separately",
        ),
        BarrierOperation::ArriveOnly => (
            Some(-1),
            None,
            None,
            "ordinary arrival; lane mask retained without converting it to a count",
        ),
        BarrierOperation::ArrivalDrop => (
            signed_count(observation.arrival_count_operand),
            signed_count(observation.arrival_count_operand),
            None,
            "arrival/drop; current and next-phase effects remain separate; missing count unresolved",
        ),
        BarrierOperation::CountedArrival => (
            signed_count(observation.arrival_count_operand),
            None,
            None,
            "counted arrival; missing count remains unresolved",
        ),
        BarrierOperation::ExpectTransactionOnly => (
            None,
            None,
            observation.transaction_delta_bytes,
            "transaction expectation only",
        ),
        BarrierOperation::CompleteTransaction => (
            None,
            None,
            observation.transaction_delta_bytes,
            "transaction accounting decrement; signed balance is not clamped",
        ),
        _ => (
            None,
            None,
            None,
            "no ordinary transaction-barrier ledger effect",
        ),
    }
}

fn new_ledger(
    cta: [u32; 3],
    offset: u32,
    raw: Option<u32>,
    cluster_rank: Option<u32>,
    producer: bool,
) -> BarrierLedger {
    BarrierLedger {
        barrier_cta: cta,
        barrier_offset: offset,
        barrier_address_raw: raw,
        barrier_cluster_rank: cluster_rank,
        lifetime: "initialization_unobserved",
        generation: None,
        operations: Vec::new(),
        observed_pending_arrival_delta: 0,
        observed_next_phase_arrival_delta: 0,
        observed_transaction_delta_bytes: 0,
        transaction_balance_is_signed: true,
        producer_candidates: Vec::new(),
        requested_phases: Vec::new(),
        producer_phase_correlations: Vec::new(),
        accounting_events: Vec::new(),
        deferred_arrivals: Vec::new(),
        producer_lane_mask: 0,
        executed_lane_mask: 0,
        arrival_lane_mask: 0,
        transaction_lane_mask: 0,
        query_lane_mask: 0,
        exchange_token_records: 0,
        exchange_token_lane_mask: 0,
        producer_membership: if producer {
            "candidate_set; generation assignment unresolved"
        } else {
            "no destination-qualified producer observed"
        },
        phase_query_instructions: 0,
        successful_query_instructions: 0,
        unsuccessful_query_instructions: 0,
        async_copy_arrivals_separate: 0,
        interpretation_conflicts: 0,
        identity_or_operand_precision_lost: false,
        precision_lost: false,
    }
}

fn build_ledgers(observations: &LaunchObservations) -> Vec<BarrierLedger> {
    let mut ledgers: BTreeMap<Barrier, BarrierLedger> = BTreeMap::new();
    for (&key, observation) in &observations.tma {
        let Some(offset) = observation.destination_barrier else {
            continue;
        };
        let identity = (
            key.cta,
            offset,
            observation.destination_barrier_raw,
            observation.destination_cluster_rank,
        );
        let entry = ledgers
            .entry(identity)
            .or_insert_with(|| new_ledger(key.cta, offset, identity.2, identity.3, true));
        if identity.2.is_none() {
            entry.identity_or_operand_precision_lost = true;
            entry.precision_lost = true;
        }
        push_candidate(
            &mut entry.producer_candidates,
            key,
            &mut entry.precision_lost,
        );
        entry.producer_lane_mask |= observation.executed_lane_mask.unwrap_or(0);
        entry.executed_lane_mask |= observation.executed_lane_mask.unwrap_or(0);
    }
    for (&key, observation) in &observations.sync {
        let Some(offset) = (observation.barrier_addresses.len() == 1).then(|| {
            *observation
                .barrier_addresses
                .first()
                .expect("set length checked")
        }) else {
            continue;
        };
        let identity = (
            key.cta,
            offset,
            observation.barrier_address_raw,
            observation.barrier_cluster_rank,
        );
        let entry = ledgers
            .entry(identity)
            .or_insert_with(|| new_ledger(key.cta, offset, identity.2, identity.3, false));
        if identity.2.is_none() {
            entry.identity_or_operand_precision_lost = true;
            entry.precision_lost = true;
        }
        if observation.exchange_records != 0 {
            entry.lifetime = "exchange_witness_observed; initialization_or_reuse_not_decoded";
        }
        entry.executed_lane_mask |= observation.executed_lane_mask;
        entry.arrival_lane_mask |= observation.arrival_lane_mask;
        entry.transaction_lane_mask |= observation.transaction_lane_mask;
        entry.query_lane_mask |= observation.query_lane_mask;
        entry.exchange_token_records = entry
            .exchange_token_records
            .saturating_add(observation.exchange_token_records);
        entry.exchange_token_lane_mask |= observation.exchange_token_lane_mask;
        for phase in &observation.requested_phases {
            if entry.requested_phases.len() >= FRONTIER_ALTERNATIVE_BUDGET {
                entry.precision_lost = true;
                break;
            }
            if !entry.requested_phases.contains(phase) {
                entry.requested_phases.push(*phase);
            }
        }
        if observation.interpretation_conflict {
            entry.interpretation_conflicts = entry.interpretation_conflicts.saturating_add(1);
            entry.precision_lost = true;
        }
        if observation.operand_precision_lost {
            entry.identity_or_operand_precision_lost = true;
            entry.precision_lost = true;
        }
        for operation in &observation.barrier_operations {
            if !entry.operations.contains(operation) {
                entry.operations.push(*operation);
            }
            let (pending_delta, next_phase_delta, transaction_delta, effect_basis) =
                accounting_effect(*operation, observation);
            entry.accounting_events.push(AccountingEvent {
                instruction: key,
                operation: *operation,
                lane_mask: observation.executed_lane_mask,
                transaction_operand_bytes: observation.transaction_operand_bytes,
                arrival_count_operand: observation.arrival_count_operand,
                pending_arrival_delta: pending_delta,
                next_phase_arrival_delta: next_phase_delta,
                transaction_delta_bytes: transaction_delta,
                effect_basis,
            });
            if (pending_delta.is_none()
                && matches!(
                    operation,
                    BarrierOperation::CountedArrival
                        | BarrierOperation::ArrivalDrop
                        | BarrierOperation::UnknownArrivalForm
                ))
                || (matches!(
                    operation,
                    BarrierOperation::ExpectTransactionOnly
                        | BarrierOperation::CompleteTransaction
                        | BarrierOperation::ArriveExpectTransaction
                ) && observation.transaction_operand_bytes.is_some()
                    && transaction_delta.is_none())
            {
                entry.precision_lost = true;
            }
            if let Some(delta) = pending_delta {
                add_i64(
                    &mut entry.observed_pending_arrival_delta,
                    delta,
                    &mut entry.precision_lost,
                );
            }
            if let Some(delta) = next_phase_delta {
                add_i64(
                    &mut entry.observed_next_phase_arrival_delta,
                    delta,
                    &mut entry.precision_lost,
                );
            }
            if let Some(delta) = transaction_delta {
                add_i64(
                    &mut entry.observed_transaction_delta_bytes,
                    delta,
                    &mut entry.precision_lost,
                );
            }
            if *operation == BarrierOperation::AsyncCopyBarrierArrival {
                entry.async_copy_arrivals_separate =
                    entry.async_copy_arrivals_separate.saturating_add(1);
                let mut source_candidates = Vec::new();
                for (&source_key, source) in &observations.tma {
                    if source_key.cta == key.cta
                        && source_key.warp == key.warp
                        && source_key.instruction < key.instruction
                        && source.is_load
                    {
                        push_candidate(
                            &mut source_candidates,
                            source_key,
                            &mut entry.precision_lost,
                        );
                    }
                }
                entry.deferred_arrivals.push(DeferredArrivalObligation {
                    marker: key,
                    source_candidates,
                    source_membership: "prior same-warp TMA candidates only; LDGSTS static membership may be absent",
                    effect_at_marker: "deferred arrival registered; no arrival balance change at marker issue",
                    discharge_basis: "model-relative prior async-copy completion, if represented",
                });
            }
        }
        if observation.is_phase_query {
            entry.phase_query_instructions = entry.phase_query_instructions.saturating_add(1);
            entry.successful_query_instructions = entry
                .successful_query_instructions
                .saturating_add(u64::from(observation.successful_wait_lanes != 0));
            entry.unsuccessful_query_instructions = entry
                .unsuccessful_query_instructions
                .saturating_add(u64::from(observation.unsuccessful_wait_lanes != 0));
        }
    }
    for entry in ledgers.values_mut() {
        entry.operations.sort();
        entry.producer_candidates.sort();
        entry.requested_phases.sort();
        if !entry.producer_candidates.is_empty() || !entry.requested_phases.is_empty() {
            entry
                .producer_phase_correlations
                .push(ProducerPhaseCorrelation {
                    pairing: ProducerPhasePairing::UnresolvedJoint {
                        producer_candidates: entry.producer_candidates.clone(),
                        phase_candidates: entry.requested_phases.clone(),
                    },
                    association: "joint alternatives; producer-to-phase pairing unresolved",
                });
        }
        if entry.producer_candidates.len() == 1 {
            entry.producer_membership =
                "one destination candidate; phase generation still unresolved";
        }
    }
    ledgers.into_values().collect()
}

fn blocker_expression_for_failed_query(
    has_barrier: bool,
) -> (
    Vec<BlockerLiteral>,
    Vec<BlockerLiteral>,
    Option<BlockerExpression>,
    Vec<&'static str>,
) {
    if has_barrier {
        (
            Vec::new(),
            vec![
                BlockerLiteral::ArrivalBalanceNonzero,
                BlockerLiteral::TransactionBalanceNonzero,
            ],
            Some(BlockerExpression::AnyOf(vec![
                BlockerLiteral::ArrivalBalanceNonzero,
                BlockerLiteral::TransactionBalanceNonzero,
            ])),
            vec![
                "failed query proves the phase requirement is unsatisfied, but not which conjunct failed",
            ],
        )
    } else {
        (
            Vec::new(),
            vec![BlockerLiteral::ProducerMembershipUnresolved],
            Some(BlockerExpression::Unknown("phase_identity_unresolved")),
            vec!["barrier identity is unavailable or ambiguous at the query frontier"],
        )
    }
}

fn build_frontiers(
    observations: &LaunchObservations,
    phase: &[PhaseReadinessObservation],
    ledgers: &[BarrierLedger],
) -> Vec<ConsumerFrontier> {
    let mut exact_success: BTreeMap<ObservationKey, Vec<&PhaseReadinessObservation>> =
        BTreeMap::new();
    for observation in phase {
        if observation.association != PhaseAssociation::EstablishedWitness {
            continue;
        }
        exact_success
            .entry(observation.query)
            .or_default()
            .push(observation);
    }
    let mut result = Vec::new();
    for (&key, observation) in &observations.sync {
        if !observation.is_phase_query {
            continue;
        }
        let barrier = (observation.barrier_addresses.len() == 1).then(|| {
            *observation
                .barrier_addresses
                .first()
                .expect("set length checked")
        });
        let barrier_raw = observation.barrier_address_raw;
        let barrier_cluster_rank = observation.barrier_cluster_rank;
        let barrier_cta = barrier.map(|_| key.cta);
        let ledger_precision_lost = barrier.and_then(|offset| {
            ledgers
                .iter()
                .find(|ledger| {
                    ledger.barrier_cta == key.cta
                        && ledger.barrier_offset == offset
                        && ledger.barrier_address_raw == barrier_raw
                        && ledger.barrier_cluster_rank == barrier_cluster_rank
                })
                .map(|ledger| ledger.precision_lost)
        }) == Some(true);
        let has_success = observation.successful_wait_lanes != 0;
        let has_failure = observation.unsuccessful_wait_lanes != 0;
        let callback_sample_clock = if observation.query_result_mixed {
            None
        } else if has_failure {
            observation.last_unsuccessful_query_clock
        } else if has_success {
            observation.first_successful_query_clock
        } else {
            observation.query_clock
        };
        let callback_phase_state = if callback_sample_clock.is_some() {
            "not_established_at_after_call_sample"
        } else {
            "no_callback_clock"
        };
        let pre_result_blocker_expression = Some(BlockerExpression::Unknown(
            "captured_prefix_does_not_decode_initial_phase_balance",
        ));
        if has_success {
            if let Some(matches) = exact_success.get(&key) {
                for matched in matches {
                    let association_uncertain = ledger_precision_lost
                        || observation.interpretation_conflict
                        || observation.operand_precision_lost;
                    result.push(ConsumerFrontier {
                        consumer: key,
                        barrier_cta: Some(matched.barrier_cta),
                        barrier_offset: Some(matched.barrier_offset),
                        barrier_address_raw: matched.barrier_address_raw,
                        barrier_cluster_rank: matched.destination_cluster_rank,
                        requested_phase: observation.requested_phase,
                        requested_phases: observation.requested_phases.iter().copied().collect(),
                        query_kind: observation.query_kind,
                        query_evaluation_observed: observation.query_result_evaluated,
                        query_evaluation_instruction: key.instruction,
                        query_result_lane_mask: observation.query_result_lane_mask,
                        query_result_mixed: observation.query_result_mixed,
                        callback_sample_clock,
                        callback_phase_state,
                        position: FrontierPosition::AfterQueryResult,
                        perspective: AnalysisPerspective::RetrospectiveObservation,
                        derivation: FrontierDerivation::RetrospectiveEvidence,
                        requirement_expression: "pending_arrivals == 0 AND transaction_balance == 0",
                        requirement_status: if observation.interpretation_conflict {
                            RequirementStatus::InterpretationConflict
                        } else if has_failure || association_uncertain {
                            RequirementStatus::Undetermined
                        } else {
                            RequirementStatus::Satisfied
                        },
                        pre_result_status: RequirementStatus::Undetermined,
                        post_result_status: if observation.interpretation_conflict {
                            RequirementStatus::InterpretationConflict
                        } else if has_failure || association_uncertain {
                            RequirementStatus::Undetermined
                        } else {
                            RequirementStatus::Satisfied
                        },
                        pre_result_blocker_expression: pre_result_blocker_expression.clone(),
                        mandatory_blockers: Vec::new(),
                        possible_blockers: if has_failure {
                            vec![
                                BlockerLiteral::ArrivalBalanceNonzero,
                                BlockerLiteral::TransactionBalanceNonzero,
                            ]
                        } else if association_uncertain {
                            vec![BlockerLiteral::ProducerMembershipUnresolved]
                        } else {
                            Vec::new()
                        },
                        blocker_expression: if has_failure {
                            Some(BlockerExpression::AnyOf(vec![
                                BlockerLiteral::ArrivalBalanceNonzero,
                                BlockerLiteral::TransactionBalanceNonzero,
                            ]))
                        } else if association_uncertain {
                            Some(BlockerExpression::Unknown(
                                "producer_membership_or_interpretation_unresolved",
                            ))
                        } else {
                            None
                        },
                        supporting_events: vec![matched.producer, key],
                        unresolved_reasons: if observation.interpretation_conflict {
                            vec!["incompatible opcode/query forms share one dynamic key"]
                        } else if observation.operand_precision_lost {
                            vec!["signed operand delta exceeds representable ledger precision"]
                        } else if ledger_precision_lost {
                            vec!["ledger alternatives widened or lost precision; exact association is not decisive"]
                        } else if has_failure {
                            vec!["mixed lane result: successful and unsuccessful query lanes are aggregated at one instruction key"]
                        } else {
                            vec!["producer association uses complete captured-launch records"]
                        },
                        captured_polling: observation.query_kind == Some(PhaseQueryKind::TryWait),
                        qualified_synchronization_applied: false,
                        engine_completion_clock: "unavailable",
                    });
                }
            } else {
                result.push(ConsumerFrontier {
                    consumer: key,
                    barrier_cta,
                    barrier_offset: barrier,
                    barrier_address_raw: barrier_raw,
                    barrier_cluster_rank,
                    requested_phase: observation.requested_phase,
                    requested_phases: observation.requested_phases.iter().copied().collect(),
                    query_kind: observation.query_kind,
                    query_evaluation_observed: observation.query_result_evaluated,
                    query_evaluation_instruction: key.instruction,
                    query_result_lane_mask: observation.query_result_lane_mask,
                    query_result_mixed: observation.query_result_mixed,
                    callback_sample_clock,
                    callback_phase_state,
                    position: FrontierPosition::AfterQueryResult,
                    perspective: AnalysisPerspective::CapturedPrefix,
                    derivation: FrontierDerivation::CurrentQuery,
                    requirement_expression: "pending_arrivals == 0 AND transaction_balance == 0",
                    requirement_status: if observation.interpretation_conflict {
                        RequirementStatus::InterpretationConflict
                    } else {
                        RequirementStatus::Undetermined
                    },
                    pre_result_status: RequirementStatus::Undetermined,
                    post_result_status: if observation.interpretation_conflict {
                        RequirementStatus::InterpretationConflict
                    } else {
                        RequirementStatus::Undetermined
                    },
                    pre_result_blocker_expression: pre_result_blocker_expression.clone(),
                    mandatory_blockers: Vec::new(),
                    possible_blockers: vec![BlockerLiteral::ProducerMembershipUnresolved],
                    blocker_expression: Some(BlockerExpression::Unknown(if observation.interpretation_conflict {
                        "incompatible opcode/query forms share one dynamic key"
                    } else {
                        "successful query has no exact producer/cohort association"
                    })),
                    supporting_events: vec![key],
                    unresolved_reasons: if observation.interpretation_conflict {
                        vec!["incompatible opcode/query forms share one dynamic key"]
                    } else {
                        vec!["successful query is observed, but generation/cohort membership is unresolved"]
                    },
                    captured_polling: observation.query_kind == Some(PhaseQueryKind::TryWait),
                    qualified_synchronization_applied: false,
                    engine_completion_clock: "unavailable",
                });
            }
        } else if has_failure {
            let (mandatory, possible, expression, mut reasons) =
                blocker_expression_for_failed_query(barrier.is_some());
            if observation.query_kind == Some(PhaseQueryKind::TryWait) {
                reasons.push(
                    "TRYWAIT may suspend until the qualified phase condition; a false/early return is not a nonblocking-test witness",
                );
            }
            if observation.interpretation_conflict {
                reasons.push("incompatible opcode/query forms share one dynamic key");
            }
            if observation.operand_precision_lost {
                reasons.push("signed operand delta exceeds representable ledger precision");
            }
            if ledger_precision_lost {
                reasons.push("ledger alternatives widened or lost precision");
            }
            result.push(ConsumerFrontier {
                consumer: key,
                barrier_cta,
                barrier_offset: barrier,
                barrier_address_raw: barrier_raw,
                barrier_cluster_rank,
                requested_phase: observation.requested_phase,
                requested_phases: observation.requested_phases.iter().copied().collect(),
                query_kind: observation.query_kind,
                query_evaluation_observed: observation.query_result_evaluated,
                query_evaluation_instruction: key.instruction,
                query_result_lane_mask: observation.query_result_lane_mask,
                query_result_mixed: observation.query_result_mixed,
                callback_sample_clock,
                callback_phase_state,
                position: FrontierPosition::AfterQueryResult,
                perspective: AnalysisPerspective::CapturedPrefix,
                derivation: FrontierDerivation::CurrentQuery,
                requirement_expression: "pending_arrivals == 0 AND transaction_balance == 0",
                requirement_status: if observation.interpretation_conflict {
                    RequirementStatus::InterpretationConflict
                } else {
                    RequirementStatus::Unsatisfied
                },
                pre_result_status: RequirementStatus::Undetermined,
                post_result_status: if observation.interpretation_conflict {
                    RequirementStatus::InterpretationConflict
                } else {
                    RequirementStatus::Unsatisfied
                },
                pre_result_blocker_expression: pre_result_blocker_expression.clone(),
                mandatory_blockers: mandatory,
                possible_blockers: possible,
                blocker_expression: expression,
                supporting_events: vec![key],
                unresolved_reasons: reasons,
                // A failed test is one observation, not proof that the
                // captured program retried.  TRYWAIT is potentially
                // suspending and is tracked separately from a nonblocking
                // PHASECHK/TESTWAIT query; adjacent retries require separate
                // dynamic keys.
                captured_polling: observation.query_kind == Some(PhaseQueryKind::TryWait),
                qualified_synchronization_applied: false,
                engine_completion_clock: "unavailable",
            });
        } else {
            result.push(ConsumerFrontier {
                consumer: key,
                barrier_cta,
                barrier_offset: barrier,
                barrier_address_raw: barrier_raw,
                barrier_cluster_rank,
                requested_phase: observation.requested_phase,
                requested_phases: observation.requested_phases.iter().copied().collect(),
                query_kind: observation.query_kind,
                query_evaluation_observed: observation.query_result_evaluated,
                query_evaluation_instruction: key.instruction,
                query_result_lane_mask: observation.query_result_lane_mask,
                query_result_mixed: observation.query_result_mixed,
                callback_sample_clock,
                callback_phase_state,
                position: FrontierPosition::AfterQueryResult,
                perspective: AnalysisPerspective::CapturedPrefix,
                derivation: FrontierDerivation::NoResult,
                requirement_expression: "pending_arrivals == 0 AND transaction_balance == 0",
                requirement_status: if observation.interpretation_conflict {
                    RequirementStatus::InterpretationConflict
                } else {
                    RequirementStatus::Undetermined
                },
                pre_result_status: RequirementStatus::Undetermined,
                post_result_status: if observation.interpretation_conflict {
                    RequirementStatus::InterpretationConflict
                } else {
                    RequirementStatus::Undetermined
                },
                pre_result_blocker_expression: pre_result_blocker_expression.clone(),
                mandatory_blockers: Vec::new(),
                possible_blockers: vec![BlockerLiteral::QueryResultUnavailable],
                blocker_expression: Some(BlockerExpression::Unknown(
                    if observation.interpretation_conflict {
                        "incompatible opcode/query forms share one dynamic key"
                    } else {
                        "query result unavailable in sidecar prefix"
                    },
                )),
                supporting_events: vec![key],
                unresolved_reasons: if observation.interpretation_conflict {
                    vec!["incompatible opcode/query forms share one dynamic key"]
                } else if observation.operand_precision_lost {
                    vec!["signed operand delta exceeds representable ledger precision"]
                } else {
                    vec!["query operands or result mask is incomplete"]
                },
                captured_polling: observation.query_kind == Some(PhaseQueryKind::TryWait),
                qualified_synchronization_applied: false,
                engine_completion_clock: "unavailable",
            });
        }
    }
    result
}

impl ObligationFrontier {
    /// Project one real launch into a typed ledger and prefix-sensitive
    /// frontiers. This method never rewrites the captured polling loop and
    /// never uses a later successful observation to resolve an earlier failed
    /// prefix.
    pub fn from_launch(
        observations: &LaunchObservations,
        cluster: [u32; 3],
    ) -> (Self, Vec<PhaseReadinessObservation>) {
        let phase = observations.phase_readiness_observations(cluster);
        let relations = observations.relations(cluster);
        let ledgers = build_ledgers(observations);
        let consumer_frontiers = build_frontiers(observations, &phase, &ledgers);
        let expected_arrival_obligations = observations
            .sync
            .values()
            .filter(|observation| observation.is_arrival)
            .count() as u64;
        let expected_transaction_bytes = observations
            .sync
            .values()
            .filter_map(|observation| observation.expected_tx_bytes)
            .fold(0_u64, |sum, value| sum.saturating_add(value));
        let captured_query_instructions = observations
            .sync
            .values()
            .filter(|observation| observation.is_phase_query)
            .count() as u64;
        let captured_successful_queries = observations
            .sync
            .values()
            .filter(|observation| observation.successful_wait_lanes != 0)
            .count() as u64;
        let captured_unsuccessful_queries = observations
            .sync
            .values()
            .filter(|observation| observation.unsuccessful_wait_lanes != 0)
            .count() as u64;
        let qualified_satisfied_generations = phase
            .iter()
            .filter(|observation| observation.association == PhaseAssociation::EstablishedWitness)
            .count() as u64;
        let protocol_scoped_phase_candidates = phase
            .iter()
            .filter(|observation| {
                observation.association == PhaseAssociation::ProtocolScopedCandidate
            })
            .count() as u64;
        let tma_load_obligations = observations
            .tma
            .values()
            .filter(|observation| observation.is_load)
            .count() as u64;
        let possible_unresolved_generations =
            tma_load_obligations.saturating_sub(qualified_satisfied_generations);
        let mandatory_obligations =
            if expected_arrival_obligations != 0 || expected_transaction_bytes != 0 {
                vec!["phase_requires_arrival_and_transaction_accounting"]
            } else {
                Vec::new()
            };
        let mut possible_obligations = Vec::new();
        if qualified_satisfied_generations != 0 {
            possible_obligations.push("associated_phase_satisfaction");
        }
        if relations.one_shot_relations != 0 {
            possible_obligations.push("ordered_producer_consumer_relation");
        }
        let mut unresolved_obligations = vec!["engine_completion_clock"];
        if possible_unresolved_generations != 0 {
            unresolved_obligations.push("producer_cohort_or_generation_membership");
        }
        if captured_unsuccessful_queries != 0 {
            unresolved_obligations.push("counterfactual_polling_schedule");
        }
        let consumer_obligations = phase
            .iter()
            .filter(|observation| observation.association == PhaseAssociation::EstablishedWitness)
            .map(|observation| ConsumerObligation {
                producer: observation.producer,
                query: observation.query,
                barrier_cta: observation.barrier_cta,
                barrier_offset: observation.barrier_offset,
                barrier_address_raw: observation.barrier_address_raw,
                barrier_cluster_rank: observation.destination_cluster_rank,
                payload_bytes: observation.payload_bytes,
                producer_association: FrontierCertainty::ObservedSatisfaction,
                software_arrival: FrontierCertainty::PossibleObligation,
                transaction_bytes: FrontierCertainty::PossibleObligation,
                phase_satisfaction: FrontierCertainty::ObservedSatisfaction,
                query_clock_basis: observation.window_basis,
                engine_completion_clock: "unavailable",
            })
            .collect();
        let frontier_precision_lost = ledgers.iter().any(|ledger| ledger.precision_lost);
        let interpretation_conflict_frontiers = consumer_frontiers
            .iter()
            .filter(|frontier| {
                frontier.requirement_status == RequirementStatus::InterpretationConflict
            })
            .count() as u64;
        let pre_result_frontiers = consumer_frontiers.len() as u64;
        let post_result_narrowed_frontiers = consumer_frontiers
            .iter()
            .filter(|frontier| frontier.pre_result_status != frontier.post_result_status)
            .count() as u64;
        let current_query_only_frontiers = consumer_frontiers
            .iter()
            .filter(|frontier| {
                frontier.derivation == FrontierDerivation::CurrentQuery
                    && frontier.pre_result_status == RequirementStatus::Undetermined
            })
            .count() as u64;
        let deferred_arrival_obligations = ledgers
            .iter()
            .map(|ledger| ledger.deferred_arrivals.len() as u64)
            .sum();
        (
            Self {
                polling_view: PollingView::CapturedProgram,
                tma_load_obligations,
                tma_store_obligations: observations
                    .tma
                    .values()
                    .filter(|observation| observation.is_store)
                    .count() as u64,
                expected_arrival_obligations,
                expected_transaction_bytes,
                captured_query_instructions,
                captured_successful_queries,
                captured_unsuccessful_queries,
                captured_polling_is_not_counterfactual: true,
                qualified_synchronization_available: qualified_satisfied_generations != 0,
                qualified_synchronization_applied: false,
                qualified_synchronization_reason: "diagnostic frontier only; no counterfactual polling loop was substituted",
                qualified_phase_observations: qualified_satisfied_generations,
                protocol_scoped_phase_candidates,
                qualified_satisfied_generations,
                possible_unresolved_generations,
                mandatory_obligations,
                possible_obligations,
                unresolved_obligations,
                ledgers,
                consumer_frontiers,
                consumer_obligations,
                frontier_alternative_budget: FRONTIER_ALTERNATIVE_BUDGET,
                frontier_precision_lost,
                interpretation_conflict_frontiers,
                pre_result_frontiers,
                post_result_narrowed_frontiers,
                current_query_only_frontiers,
                deferred_arrival_obligations,
                engine_completion_clock: "unavailable: no engine-originated per-generation timestamp",
                model_timing_input: "none: frontier is evidence/model diagnostics, not replay timing",
            },
            phase,
        )
    }
}
