//! Report native TMA/synchronization relations from one real capture.
//! This is an evidence surface only; it never supplies replay timing.
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
    relations: sim_trace::async_sidecars::AsyncRelations,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let root = PathBuf::from(args.next().ok_or("usage: async_relation_report CAPTURE")?);
    if args.next().is_some() {
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
        launches.push(LaunchReport {
            device_id: loc.device_id(),
            stream_id: loc.stream_id(),
            kernel_id: loc.kernel().id,
            tma_records: observations.tma.len(),
            sync_records: observations.sync.len(),
            relations: observations.relations(cluster),
        });
    }
    serde_json::to_writer_pretty(std::io::stdout().lock(), &launches)?;
    println!();
    Ok(())
}
