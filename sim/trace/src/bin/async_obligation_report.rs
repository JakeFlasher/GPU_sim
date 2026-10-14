//! Report captured versus qualified asynchronous-obligation frontiers.
//!
//! This tool reads an existing real capture only. It does not replay, tune,
//! synthesize a wait schedule, or provide a completion timestamp.

use serde::Serialize;
use sim_trace::obligations::ObligationFrontier;
use sim_trace::TraceDir;
use std::path::PathBuf;

#[derive(Debug, Serialize)]
struct LaunchReport {
    device_id: i64,
    stream_id: i64,
    kernel_id: i32,
    frontier: ObligationFrontier,
    phase_observation_count: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or("usage: async_obligation_report CAPTURE")?,
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
        let (frontier, phase) = ObligationFrontier::from_launch(observations, cluster);
        launches.push(LaunchReport {
            device_id: loc.device_id(),
            stream_id: loc.stream_id(),
            kernel_id: loc.kernel().id,
            frontier,
            phase_observation_count: phase.len(),
        });
    }
    serde_json::to_writer_pretty(std::io::stdout().lock(), &launches)?;
    println!();
    Ok(())
}
