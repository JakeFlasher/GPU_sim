#![allow(clippy::unwrap_used)] // test binary: unwrap is an assertion
//! Machine-local smoke test against a real captured trace.
//!
//! A local captured trace can be supplied through the environment:
//!
//! ```sh
//! SIM_TRACE_SMOKE_DIR=<abs path to a trace_<shape> dir> \
//!   cargo test --release -p sim-trace --test machine_local_smoke \
//!   real_trace_full_walk -- --ignored --exact --nocapture
//! ```
//!
//! The test is ignored by the ordinary workspace suite because capture data is
//! machine-local. Run it directly whenever a suitable trace is available.

use sim_trace::{TraceDir, visit_memory_access};

#[test]
#[ignore = "requires a machine-local captured trace"]
fn real_trace_full_walk() {
    let dir = std::env::var("SIM_TRACE_SMOKE_DIR")
        .expect("SIM_TRACE_SMOKE_DIR must point to a captured trace");
    let td = TraceDir::open_direct(&dir).expect("open trace dir");
    let mut kernels = 0usize;
    let mut blocks = 0usize;
    let mut insts = 0usize;
    let mut mem_insts = 0usize;
    for loc in td.kernels().collect::<Vec<_>>() {
        kernels += 1;
        // Launch-order read exercises block_id validation (presence, grid
        // range, duplicates) on every real threadblock.
        let ordered = td
            .read_threadblocks_in_launch_order(&loc)
            .expect("launch-order threadblock read");
        assert!(
            ordered.windows(2).all(|w| w[0].0 < w[1].0),
            "launch order must be strictly increasing"
        );
        for (index, tb) in &ordered {
            blocks += 1;
            for (warp_id, warp) in &tb.warps {
                for (i, inst) in warp.instructions.iter().enumerate() {
                    insts += 1;
                    let rec = || format!("block={index} warp={warp_id} inst={i}");
                    // Streaming form: the hot-path API the SM/mem models use.
                    let meta = visit_memory_access(inst, rec, |_a| {}).expect("reader contract");
                    if meta.is_some() {
                        mem_insts += 1;
                    }
                }
            }
        }
    }
    eprintln!(
        "smoke: {kernels} kernel(s), {blocks} threadblock(s), {insts} instruction(s), \
         {mem_insts} memory instruction(s)"
    );
    assert!(kernels > 0 && blocks > 0 && insts > 0);
}
