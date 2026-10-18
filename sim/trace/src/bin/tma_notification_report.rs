//! Compact report for the bounded SM120 phase-readiness observation seam.
//!
//! The report consumes one real NVBit capture and never feeds clocks into
//! replay.  Query roles are supplied by the caller because relaxed and acquire
//! mbarrier queries can share the same lowered SASS mnemonic.

use serde::Serialize;
use sim_trace::TraceDir;
use std::path::PathBuf;

#[derive(Debug, Serialize)]
struct LaunchReport {
    device_id: i64,
    stream_id: i64,
    kernel_id: i32,
    tma_records: usize,
    sync_records: usize,
    phase_readiness_observations: usize,
    relaxed_observations: usize,
    acquire_observations: usize,
    completion_clock: &'static str,
    observations: Vec<ObservationReport>,
}

#[derive(Debug, Serialize)]
struct ObservationReport {
    #[serde(flatten)]
    observation: sim_trace::async_sidecars::PhaseReadinessObservation,
    query_role: &'static str,
    query_semantics: &'static str,
    query_clock_basis: &'static str,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let root = PathBuf::from(
        args.next()
            .ok_or("usage: tma_notification_report CAPTURE RELAXED_WARP ACQUIRE_WARP")?,
    );
    let relaxed_warp: i32 = args
        .next()
        .ok_or("missing relaxed observer warp")?
        .to_str()
        .ok_or("relaxed warp is not UTF-8")?
        .parse()?;
    let acquire_warp: i32 = args
        .next()
        .ok_or("missing acquire consumer warp")?
        .to_str()
        .ok_or("acquire warp is not UTF-8")?
        .parse()?;
    if args.next().is_some() {
        return Err("expected capture and two role warps".into());
    }
    let trace = TraceDir::open_direct_compact(&root)?;
    let sidecars = trace.async_sidecars()?;
    let mut launches = Vec::new();
    for loc in trace.kernels() {
        let Some(observations) = sidecars.launch(loc.device_id(), loc.stream_id(), loc.kernel().id)
        else {
            continue;
        };
        let tma_records = observations.tma.len();
        let sync_records = observations.sync.len();
        let cluster = loc
            .kernel()
            .cluster_dim
            .as_ref()
            .filter(|dim| dim.x > 0 && dim.y > 0 && dim.z > 0)
            .map(|dim| [dim.x as u32, dim.y as u32, dim.z as u32])
            .unwrap_or([1, 1, 1]);
        let all = observations.phase_readiness_observations(cluster);
        let all_count = all.len();
        let relaxed_observations = all
            .iter()
            .filter(|observation| observation.query.warp == relaxed_warp)
            .count();
        let acquire_observations = all
            .iter()
            .filter(|observation| observation.query.warp == acquire_warp)
            .count();
        let observations = all
            .into_iter()
            .map(|observation| {
                let (query_role, query_semantics) = if observation.query.warp == relaxed_warp {
                    ("relaxed_observer", "relaxed")
                } else if observation.query.warp == acquire_warp {
                    ("acquire_consumer", "acquire")
                } else {
                    ("unclassified", "unknown")
                };
                ObservationReport {
                    query_clock_basis: observation.window_basis,
                    observation,
                    query_role,
                    query_semantics,
                }
            })
            .collect();
        launches.push(LaunchReport {
            device_id: loc.device_id(),
            stream_id: loc.stream_id(),
            kernel_id: loc.kernel().id,
            tma_records,
            sync_records,
            phase_readiness_observations: all_count,
            relaxed_observations,
            acquire_observations,
            completion_clock: "unavailable: no engine-originated timestamp in NVBit sidecar",
            observations,
        });
    }
    serde_json::to_writer_pretty(std::io::stdout().lock(), &launches)?;
    println!();
    Ok(())
}
