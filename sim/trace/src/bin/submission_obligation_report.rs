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
    unresolved_generation_barriers: u64,
    unresolved_payload_barriers: u64,
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

#[derive(Debug, Serialize)]
struct Report {
    capture: String,
    static_submission: SubmissionSummary,
    dynamic_obligations: DynamicObligations,
    static_dynamic_join: StaticDynamicJoin,
    typed_absences: BTreeMap<&'static str, &'static str>,
    interpretation_scope: &'static str,
}

fn add_dynamic(
    report: &mut DynamicObligations,
    join: &mut StaticDynamicJoin,
    static_submission: &SubmissionSummary,
    trace: &TraceDir,
) -> Result<(), Box<dyn std::error::Error>> {
    let sidecars = trace.async_sidecars()?;
    let mut by_pc: BTreeMap<(i32, u64), Vec<SubmissionClass>> = BTreeMap::new();
    for instruction in &static_submission.instructions {
        if let Some(function_id) = instruction.function_id {
            by_pc
                .entry((function_id, instruction.pc))
                .or_default()
                .push(instruction.class);
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
        for observation in observations.sync.values() {
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
                .saturating_add(u64::from(observation.is_phase_query));
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
        report.unresolved_generation_barriers = report
            .unresolved_generation_barriers
            .saturating_add(relations.unresolved_generation_barriers);
        report.unresolved_payload_barriers = report
            .unresolved_payload_barriers
            .saturating_add(relations.unresolved_payload_barriers);
        report.phase_readiness_observations = report
            .phase_readiness_observations
            .saturating_add(observations.phase_readiness_observations(cluster).len() as u64);
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
    add_dynamic(
        &mut dynamic_obligations,
        &mut static_dynamic_join,
        &static_submission,
        &trace,
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
        typed_absences,
        interpretation_scope: "real-trace static/dynamic obligation projection; no replay timing input",
    };
    serde_json::to_writer_pretty(std::io::stdout().lock(), &report)?;
    println!();
    Ok(())
}
