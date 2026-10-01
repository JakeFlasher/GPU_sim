//! Small executable integration probe for the merged causal-frontier module.
//!
//! The production replay remains the source of simulator timing. This probe
//! keeps the descriptor producer explicit and exercises the native analyzer's
//! collapse/refusal behavior in the same release build used by campaigns.

use sim_sm::{
    CausalFrontierEvidence, CausalFrontierInstruction, CausalFrontierResidency, PipelineClass,
    summarize_causal_frontier, summarize_causal_frontier_with_residency,
};

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
    let programs = [first, second, third];
    let single_warp = (0..16)
        .map(|sequence| CausalFrontierInstruction {
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
        })
        .collect::<Vec<_>>();
    let model_local_single_warp = summarize_causal_frontier_with_residency(
        std::slice::from_ref(&single_warp),
        CausalFrontierResidency::CompleteWithEvidence {
            physical_warps: vec![0],
            evidence: vec![CausalFrontierEvidence {
                physical_warp: 0,
                first_sequence: 0,
                last_sequence: 15,
            }],
        },
    )?;
    let report = summarize_causal_frontier_with_residency(
        &programs,
        CausalFrontierResidency::Complete {
            physical_warps: vec![0, 1, 5],
        },
    )?;
    let unknown = summarize_causal_frontier(&programs)?;
    // Warp 4 owns SMSP 0 too, even though it has no candidate instructions.
    let extra_resident = summarize_causal_frontier_with_residency(
        &programs,
        CausalFrontierResidency::Complete {
            physical_warps: vec![0, 1, 4, 5],
        },
    )?;
    let mut output = serde_json::to_value(report)?;
    output["residency_cases"] = serde_json::json!({
        "unknown": unknown,
        "extra_resident": extra_resident,
    });
    output["model_local_single_warp"] = serde_json::to_value(model_local_single_warp)?;
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
