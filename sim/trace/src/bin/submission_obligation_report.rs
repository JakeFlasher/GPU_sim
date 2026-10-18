//! Report typed submission/dependency obligations from one real SM120 capture.
//!
//! This is a retrospective evidence projection. It does not alter replay,
//! consume NCU values, infer physical queues, or synthesize completion clocks.

use serde::Serialize;
use sim_trace::TraceDir;
use sim_trace::submission::{SubmissionClass, SubmissionSummary, load_metadata};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Debug, Default, Serialize)]
struct DynamicObligations {
    launches: u64,
    tma_records: u64,
    sync_instruction_records: u64,
    tma_load_records: u64,
    tma_store_records: u64,
    arrival_instruction_records: u64,
    arrive_expect_transaction_records: u64,
    arrive_only_records: u64,
    complete_transaction_records: u64,
    async_copy_barrier_arrival_records: u64,
    signed_transaction_delta_bytes: i64,
    executed_lane_union: u32,
    arrival_lane_union: u32,
    transaction_lane_union: u32,
    query_lane_union: u32,
    exchange_token_records: u64,
    exchange_token_lane_union: u32,
    expected_transaction_bytes: u64,
    phase_query_instructions: u64,
    successful_query_instructions: u64,
    unsuccessful_query_instructions: u64,
    captured_polling_query_instructions: u64,
    phase_readiness_observations: u64,
    ordered_relations: u64,
    tma_generation_relations: u64,
    tma_exact_payload_cohorts: u64,
    tma_payload_accounted_relations: u64,
    tma_phase_readiness_relations: u64,
    unresolved_generation_barriers: u64,
    unresolved_payload_barriers: u64,
    query_evaluation_observations: u64,
    pre_result_frontiers: u64,
    post_result_narrowed_frontiers: u64,
    current_query_only_frontiers: u64,
    deferred_arrival_obligations: u64,
    accounting_events: u64,
}

#[derive(Debug, Default, Serialize)]
struct StaticDynamicJoin {
    dynamic_records: u64,
    matched_records: u64,
    unmatched_records: u64,
    ambiguous_records: u64,
    matched_class_counts: BTreeMap<SubmissionClass, u64>,
    join_key: &'static str,
}

#[derive(Debug, Default, Serialize)]
struct ProtocolRouting {
    native_phase_query_instruction_records: u64,
    alpha_static_async_wait_classification: u64,
    native_deferred_arrival_instruction_records: u64,
    alpha_static_other_classification: u64,
    structural_mismatch_candidates: u64,
    /// This report intentionally has no runtime alpha schedule.  Counts are
    /// static/dynamic join candidates only, never exercised issue decisions.
    classification_scope: &'static str,
    source_rule: &'static str,
}

#[derive(Debug, Serialize)]
struct Report {
    capture: String,
    static_submission: SubmissionSummary,
    dynamic_obligations: DynamicObligations,
    static_dynamic_join: StaticDynamicJoin,
    protocol_routing: ProtocolRouting,
    typed_absences: BTreeMap<&'static str, &'static str>,
    interpretation_scope: &'static str,
}

fn add_dynamic(
    report: &mut DynamicObligations,
    join: &mut StaticDynamicJoin,
    static_submission: &SubmissionSummary,
    trace: &TraceDir,
    routing: &mut ProtocolRouting,
) -> Result<(), Box<dyn std::error::Error>> {
    let contains_opcode = |opcodes: &[String], prefix: &str| {
        opcodes.iter().any(|opcode| {
            opcode.split_ascii_whitespace().any(|token| {
                token
                    .trim_start_matches(|c: char| c == '@' || c == '!')
                    .starts_with(prefix)
            })
        })
    };
    let sidecars = trace.async_sidecars()?;
    let mut by_pc: BTreeMap<(i32, u64), Vec<SubmissionClass>> = BTreeMap::new();
    let mut opcode_by_pc: BTreeMap<(i32, u64), Vec<String>> = BTreeMap::new();
    for instruction in &static_submission.instructions {
        if let Some(function_id) = instruction.function_id {
            by_pc
                .entry((function_id, instruction.pc))
                .or_default()
                .push(instruction.class);
            opcode_by_pc
                .entry((function_id, instruction.pc))
                .or_default()
                .push(instruction.opcode.clone());
        }
    }
    let mut join_record = |function_id: i32, pc: u32| {
        join.dynamic_records = join.dynamic_records.saturating_add(1);
        let matches = by_pc.get(&(function_id, u64::from(pc)));
        match matches {
            Some(classes) if classes.len() == 1 => {
                join.matched_records = join.matched_records.saturating_add(1);
                *join.matched_class_counts.entry(classes[0]).or_default() += 1;
            }
            Some(_) => join.ambiguous_records = join.ambiguous_records.saturating_add(1),
            None => join.unmatched_records = join.unmatched_records.saturating_add(1),
        }
    };
    for loc in trace.kernels() {
        let Some(observations) = sidecars.launch(loc.device_id(), loc.stream_id(), loc.kernel().id)
        else {
            continue;
        };
        report.launches = report.launches.saturating_add(1);
        report.tma_records = report
            .tma_records
            .saturating_add(observations.tma.len() as u64);
        for key in observations.tma.keys() {
            join_record(key.function, key.pc);
        }
        report.sync_instruction_records = report
            .sync_instruction_records
            .saturating_add(observations.sync.len() as u64);
        for key in observations.sync.keys() {
            join_record(key.function, key.pc);
        }
        for observation in observations.tma.values() {
            if observation.is_load {
                report.tma_load_records = report.tma_load_records.saturating_add(1);
            }
            if observation.is_store {
                report.tma_store_records = report.tma_store_records.saturating_add(1);
            }
            report.executed_lane_union |= observation.executed_lane_mask.unwrap_or(0);
        }
        for (&key, observation) in &observations.sync {
            let static_opcodes = opcode_by_pc.get(&(key.function, u64::from(key.pc)));
            if observation.is_phase_query {
                routing.native_phase_query_instruction_records = routing
                    .native_phase_query_instruction_records
                    .saturating_add(1);
                if static_opcodes.is_some_and(|opcodes| contains_opcode(opcodes, "SYNCS.PHASECHK"))
                {
                    // `alpha_op_class` currently routes every PHASECHK to
                    // AsyncCopyWait. This is a structural mismatch candidate
                    // because PHASECHK is a transaction-barrier query, not a
                    // DEPBAR committed-group wait. It is not yet a replay bug.
                    routing.alpha_static_async_wait_classification = routing
                        .alpha_static_async_wait_classification
                        .saturating_add(1);
                    routing.structural_mismatch_candidates =
                        routing.structural_mismatch_candidates.saturating_add(1);
                }
            }
            if observation.is_async_copy_arrival {
                routing.native_deferred_arrival_instruction_records = routing
                    .native_deferred_arrival_instruction_records
                    .saturating_add(1);
                if static_opcodes
                    .is_some_and(|opcodes| contains_opcode(opcodes, "ARRIVES.LDGSTSBAR"))
                {
                    // `alpha_op_class` falls through to Other for ARRIVES;
                    // the deferred copy-arrival effect is not represented by
                    // an ordinary arrival or async-group event here.
                    routing.alpha_static_other_classification =
                        routing.alpha_static_other_classification.saturating_add(1);
                    routing.structural_mismatch_candidates =
                        routing.structural_mismatch_candidates.saturating_add(1);
                }
            }
            report.executed_lane_union |= observation.executed_lane_mask;
            report.arrival_lane_union |= observation.arrival_lane_mask;
            report.transaction_lane_union |= observation.transaction_lane_mask;
            report.query_lane_union |= observation.query_lane_mask;
            report.exchange_token_records = report
                .exchange_token_records
                .saturating_add(observation.exchange_token_records);
            report.exchange_token_lane_union |= observation.exchange_token_lane_mask;
            report.arrival_instruction_records = report
                .arrival_instruction_records
                .saturating_add(u64::from(observation.is_arrival));
            for operation in &observation.barrier_operations {
                use sim_trace::async_sidecars::BarrierOperation;
                match operation {
                    BarrierOperation::ArriveExpectTransaction => {
                        report.arrive_expect_transaction_records =
                            report.arrive_expect_transaction_records.saturating_add(1)
                    }
                    BarrierOperation::ArriveOnly => {
                        report.arrive_only_records = report.arrive_only_records.saturating_add(1)
                    }
                    BarrierOperation::CompleteTransaction => {
                        report.complete_transaction_records =
                            report.complete_transaction_records.saturating_add(1)
                    }
                    BarrierOperation::AsyncCopyBarrierArrival => {
                        report.async_copy_barrier_arrival_records =
                            report.async_copy_barrier_arrival_records.saturating_add(1)
                    }
                    _ => {}
                }
            }
            if let Some(delta) = observation.transaction_delta_bytes {
                report.signed_transaction_delta_bytes =
                    report.signed_transaction_delta_bytes.saturating_add(delta);
            }
            report.expected_transaction_bytes = report
                .expected_transaction_bytes
                .saturating_add(observation.expected_tx_bytes.unwrap_or(0));
            report.phase_query_instructions = report
                .phase_query_instructions
                .saturating_add(u64::from(observation.is_phase_query));
            report.successful_query_instructions = report
                .successful_query_instructions
                .saturating_add(u64::from(observation.successful_wait_lanes != 0));
            report.unsuccessful_query_instructions = report
                .unsuccessful_query_instructions
                .saturating_add(u64::from(observation.unsuccessful_wait_lanes != 0));
            // This is an observation of what the captured program actually
            // polled. It is not a counterfactual synchronization abstraction.
            report.captured_polling_query_instructions = report
                .captured_polling_query_instructions
                .saturating_add(u64::from(
                    observation.query_kind
                        == Some(sim_trace::async_sidecars::PhaseQueryKind::TryWait),
                ));
        }
        let cluster = loc
            .kernel()
            .cluster_dim
            .as_ref()
            .filter(|dim| dim.x > 0 && dim.y > 0 && dim.z > 0)
            .map(|dim| [dim.x as u32, dim.y as u32, dim.z as u32])
            .unwrap_or([1, 1, 1]);
        let relations = observations.relations(cluster);
        report.ordered_relations = report
            .ordered_relations
            .saturating_add(relations.one_shot_relations);
        report.tma_generation_relations = report
            .tma_generation_relations
            .saturating_add(relations.generation_relations.len() as u64);
        report.tma_exact_payload_cohorts = report.tma_exact_payload_cohorts.saturating_add(
            relations
                .generation_relations
                .iter()
                .filter(|relation| {
                    relation.association
                        == sim_trace::async_sidecars::TmaGenerationAssociation::ExactPayloadCohort
                })
                .count() as u64,
        );
        report.tma_payload_accounted_relations =
            report.tma_payload_accounted_relations.saturating_add(
                relations
                    .generation_relations
                    .iter()
                    .filter(|relation| {
                        relation.service_state
                            == sim_trace::async_sidecars::TmaServiceState::PayloadAccounted
                    })
                    .count() as u64,
            );
        report.tma_phase_readiness_relations = report.tma_phase_readiness_relations.saturating_add(
            relations
                .generation_relations
                .iter()
                .filter(|relation| {
                    relation.service_state
                        == sim_trace::async_sidecars::TmaServiceState::PhaseReadinessObserved
                })
                .count() as u64,
        );
        report.unresolved_generation_barriers = report
            .unresolved_generation_barriers
            .saturating_add(relations.unresolved_generation_barriers);
        report.unresolved_payload_barriers = report
            .unresolved_payload_barriers
            .saturating_add(relations.unresolved_payload_barriers);
        report.phase_readiness_observations = report
            .phase_readiness_observations
            .saturating_add(observations.phase_readiness_observations(cluster).len() as u64);
        let (frontier, _) =
            sim_trace::obligations::ObligationFrontier::from_launch(observations, cluster);
        report.query_evaluation_observations = report.query_evaluation_observations.saturating_add(
            frontier
                .consumer_frontiers
                .iter()
                .filter(|frontier| frontier.query_evaluation_observed)
                .count() as u64,
        );
        report.pre_result_frontiers = report
            .pre_result_frontiers
            .saturating_add(frontier.pre_result_frontiers);
        report.post_result_narrowed_frontiers = report
            .post_result_narrowed_frontiers
            .saturating_add(frontier.post_result_narrowed_frontiers);
        report.current_query_only_frontiers = report
            .current_query_only_frontiers
            .saturating_add(frontier.current_query_only_frontiers);
        report.deferred_arrival_obligations = report
            .deferred_arrival_obligations
            .saturating_add(frontier.deferred_arrival_obligations);
        report.accounting_events = report.accounting_events.saturating_add(
            frontier
                .ledgers
                .iter()
                .map(|ledger| ledger.accounting_events.len() as u64)
                .sum::<u64>(),
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or("usage: submission_obligation_report CAPTURE")?,
    );
    if std::env::args_os().nth(2).is_some() {
        return Err("expected one capture directory".into());
    }
    let trace = TraceDir::open_direct_compact(&root)?;
    let static_submission = load_metadata(&root)?;
    let mut dynamic_obligations = DynamicObligations::default();
    let mut static_dynamic_join = StaticDynamicJoin {
        join_key: "(function_id, pc)",
        ..StaticDynamicJoin::default()
    };
    let mut protocol_routing = ProtocolRouting {
        classification_scope: "static_opcode_join_candidate; runtime_handler_not_observed",
        source_rule: "source metadata contains PHASECHK/ARRIVES.LDGSTSBAR; alpha schedule and handler effects are not executed by this report, so counts are candidate inventory only",
        ..ProtocolRouting::default()
    };
    add_dynamic(
        &mut dynamic_obligations,
        &mut static_dynamic_join,
        &static_submission,
        &trace,
        &mut protocol_routing,
    )?;
    let mut typed_absences = BTreeMap::new();
    typed_absences.insert(
        "engine_completion_clock",
        "unavailable: no engine-originated per-generation timestamp in the collected contract",
    );
    typed_absences.insert(
        "physical_queue_capacity",
        "unavailable: static scoreboard/queue metadata is not a measured capacity",
    );
    typed_absences.insert(
        "counterfactual_polling_schedule",
        "unavailable unless source semantics prove a qualified synchronization abstraction",
    );
    let report = Report {
        capture: root.display().to_string(),
        static_submission,
        dynamic_obligations,
        static_dynamic_join,
        protocol_routing,
        typed_absences,
        interpretation_scope: "real-trace static/dynamic obligation projection; no replay timing input",
    };
    serde_json::to_writer_pretty(std::io::stdout().lock(), &report)?;
    println!();
    Ok(())
}
