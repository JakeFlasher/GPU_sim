#![allow(clippy::unwrap_used)] // test binary: unwrap is an assertion
//! Machine-local smoke test against a real captured trace.
//!
//! The M1 trace evidence is gitignored machine-local data, so this test takes
//! its input from the environment instead of a hard-coded path:
//!
//! ```sh
//! SIM_TRACE_SMOKE_DIR=<abs path to a trace_<shape> dir> \
//!   cargo test --release -p sim-trace --test machine_local_smoke \
//!   real_trace_full_walk -- --ignored --exact --nocapture
//! ```
//!
//! The test is ignored by the ordinary workspace suite and is executed by the
//! authoritative hardware gate. That gate supplies the environment; a direct
//! ignored-test invocation without it fails rather than returning green.

use sim_trace::{TraceDir, visit_memory_access};

#[test]
#[ignore = "requires machine-local M1 evidence; run tools/dev/hw_evidence_gate.sh"]
fn real_trace_full_walk() {
    let dir = std::env::var("SIM_TRACE_SMOKE_DIR")
        .expect("SIM_TRACE_SMOKE_DIR must be set by tools/dev/hw_evidence_gate.sh");
    let td = TraceDir::open(&dir).expect("open trace dir");
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
