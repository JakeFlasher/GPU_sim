//! Compare the evidence-frontier diagnosis with the scalar async dependency
//! rule used by `sim-sm` without pretending that a scalar per-query endpoint
//! was recorded. This reads one existing real capture and never replays it.

use serde::Serialize;
use sim_trace::TraceDir;
use sim_trace::obligations::{ObligationFrontier, RequirementStatus};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Debug, Default, Serialize)]
struct FrontierCounts {
    satisfied: u64,
    unsatisfied: u64,
    undetermined: u64,
    interpretation_conflict: u64,
    failed_query_disjunctions: u64,
}

#[derive(Debug, Serialize)]
struct LaunchReport {
    device_id: i64,
    stream_id: i64,
    kernel_id: i32,
    frontier: ObligationFrontier,
    counts: FrontierCounts,
}

#[derive(Debug, Serialize)]
struct Report {
    capture: String,
    scalar_model_snapshot: &'static str,
    scalar_rule: BTreeMap<&'static str, &'static str>,
    launches: Vec<LaunchReport>,
    comparison_scope: &'static str,
}

fn count_frontier(frontier: &ObligationFrontier) -> FrontierCounts {
    let mut counts = FrontierCounts::default();
    for consumer in &frontier.consumer_frontiers {
        match consumer.requirement_status {
            RequirementStatus::Satisfied => counts.satisfied += 1,
            RequirementStatus::Unsatisfied => counts.unsatisfied += 1,
            RequirementStatus::Undetermined => counts.undetermined += 1,
            RequirementStatus::InterpretationConflict => counts.interpretation_conflict += 1,
        }
        if consumer.blocker_expression.as_ref().is_some_and(|expression| {
            matches!(
                expression,
                sim_trace::obligations::BlockerExpression::AnyOf(literals)
                    if literals.len() == 2
                        && literals.contains(&sim_trace::obligations::BlockerLiteral::ArrivalBalanceNonzero)
                        && literals.contains(&sim_trace::obligations::BlockerLiteral::TransactionBalanceNonzero)
            )
        }) {
            counts.failed_query_disjunctions += 1;
        }
    }
    counts
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or("usage: obligation_causal_report CAPTURE")?,
    );
    if std::env::args_os().nth(2).is_some() {
        return Err("expected one capture directory".into());
    }
    let trace = TraceDir::open_direct_compact(&root)?;
    let sidecars = trace.async_sidecars()?;
    let mut launches = Vec::new();
    for loc in trace.kernels() {
        let Some(observations) = sidecars.launch(loc.device_id(), loc.stream_id(), loc.kernel().id)
        else {
            continue;
        };
        let cluster = loc
            .kernel()
            .cluster_dim
            .as_ref()
            .filter(|dim| dim.x > 0 && dim.y > 0 && dim.z > 0)
            .map(|dim| [dim.x as u32, dim.y as u32, dim.z as u32])
            .unwrap_or([1, 1, 1]);
        let (frontier, _) = ObligationFrontier::from_launch(observations, cluster);
        let counts = count_frontier(&frontier);
        launches.push(LaunchReport {
            device_id: loc.device_id(),
            stream_id: loc.stream_id(),
            kernel_id: loc.kernel().id,
            frontier,
            counts,
        });
    }
    let mut scalar_rule = BTreeMap::new();
    scalar_rule.insert(
        "commit",
        "AsyncCopyCommit closes the open group and may add configured SM120 flat service",
    );
    scalar_rule.insert(
        "wait_prefix",
        "AsyncCopyWait consumes the oldest committed prefix selected by max_pending_groups",
    );
    scalar_rule.insert(
        "readiness",
        "wait readiness uses model-relative group completion_time and limiting_readiness_cause",
    );
    scalar_rule.insert(
        "evidence_boundary",
        "the scalar completion_time is not an engine-originated per-generation clock",
    );
    let report = Report {
        capture: root.display().to_string(),
        scalar_model_snapshot: "unavailable: no per-query scalar group snapshot is emitted by existing outputs",
        scalar_rule,
        launches,
        comparison_scope: "same-trace source-rule comparison; no replay, NCU fitting, or timing correction",
    };
    serde_json::to_writer_pretty(std::io::stdout().lock(), &report)?;
    println!();
    Ok(())
}
