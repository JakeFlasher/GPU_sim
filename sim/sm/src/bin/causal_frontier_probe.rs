//! Small executable integration probe for the merged causal-frontier module.
//!
//! The production replay remains the source of simulator timing. This probe
//! keeps the descriptor producer explicit and exercises the native analyzer's
//! collapse/refusal behavior in the same release build used by campaigns.

use sim_sm::{CausalFrontierInstruction, PipelineClass, summarize_causal_frontier};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scalar = PipelineClass::ScalarAlu {
        latency_cycles: 1,
        initiation_interval_cycles: 1,
        in_flight_capacity: 8,
    };
    let mut first = Vec::new();
    for sequence in 0..4 {
        first.push(CausalFrontierInstruction {
            sequence,
            physical_warp: 0,
            pipeline: scalar,
            latency_cycles: 1,
            initiation_interval_cycles: 1,
            in_flight_capacity: 8,
            completion_units: 1,
            control_clean: true,
            dependency_free: true,
            external_side_effect: false,
        });
    }
    let mut second = first.clone();
    for instruction in &mut second {
        instruction.physical_warp = 1;
    }
    let mut third = first.clone();
    for instruction in &mut third {
        instruction.physical_warp = 5;
    }
    // A side-effecting tail is kept explicit and cannot become a candidate
    // region. The two programs on SMSP 1 force their safe regions to reify.
    let mut refused = third[0];
    refused.sequence = 4;
    refused.external_side_effect = true;
    third.push(refused);
    let report = summarize_causal_frontier(&[first, second, third])?;
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}
