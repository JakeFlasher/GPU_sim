//! Bounded PhaseFuse analyzer scaling probe.
//!
//! This generates explicit public causal-frontier descriptors and measures
//! descriptor construction, analyzer wall time, and the analyzer's own
//! planner telemetry. It is intentionally not a native trace or timing
//! prediction. A separate wrapper can collect process RSS for each case.

use std::collections::BTreeMap;
use std::time::Instant;

use sim_sm::{
    CausalFrontierInstruction, CausalFrontierResidency, PipelineClass,
    summarize_causal_frontier_with_residency,
};

#[derive(Debug, Clone, Copy)]
struct Options {
    n: usize,
    region_len: usize,
    complete_residency: bool,
    shuffled: bool,
    seed: u64,
}

fn usage() -> &'static str {
    "phasefuse-scale --n N --region-len {1,2,8,64} \
     --residency {complete,unknown} --order {sorted,shuffled} --seed S"
}

fn parse_options() -> Result<Options, String> {
    let mut args = std::env::args().skip(1);
    let mut n = None;
    let mut region_len = None;
    let mut complete_residency = None;
    let mut shuffled = None;
    let mut seed = 0_u64;
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}; usage: {}", usage()))?;
        match flag.as_str() {
            "--n" => n = Some(value.parse::<usize>().map_err(|_| "invalid --n")?),
            "--region-len" => {
                let parsed = value.parse::<usize>().map_err(|_| "invalid --region-len")?;
                if !matches!(parsed, 1 | 2 | 8 | 64) {
                    return Err("--region-len must be one of 1,2,8,64".to_owned());
                }
                region_len = Some(parsed);
            }
            "--residency" => {
                complete_residency = Some(match value.as_str() {
                    "complete" => true,
                    "unknown" => false,
                    _ => return Err("--residency must be complete or unknown".to_owned()),
                });
            }
            "--order" => {
                shuffled = Some(match value.as_str() {
                    "sorted" => false,
                    "shuffled" => true,
                    _ => return Err("--order must be sorted or shuffled".to_owned()),
                });
            }
            "--seed" => seed = value.parse::<u64>().map_err(|_| "invalid --seed")?,
            _ => return Err(format!("unknown option {flag}; usage: {}", usage())),
        }
    }
    let n = n.ok_or_else(|| format!("--n is required; usage: {}", usage()))?;
    if n == 0 || n > 1_000_000 {
        return Err("--n must be in 1..=1000000".to_owned());
    }
    Ok(Options {
        n,
        region_len: region_len.ok_or_else(|| "--region-len is required".to_owned())?,
        complete_residency: complete_residency
            .ok_or_else(|| "--residency is required".to_owned())?,
        shuffled: shuffled.ok_or_else(|| "--order is required".to_owned())?,
        seed,
    })
}

fn shuffle<T>(values: &mut [T], mut state: u64) {
    for index in (1..values.len()).rev() {
        state ^= state << 7;
        state ^= state >> 9;
        values.swap(index, (state as usize) % (index + 1));
    }
}

fn program(
    count: usize,
    physical_warp: i32,
    region_len: usize,
    shuffled: bool,
    seed: u64,
) -> Vec<CausalFrontierInstruction> {
    let scalar = PipelineClass::ScalarAlu {
        latency_cycles: 1,
        initiation_interval_cycles: 1,
        in_flight_capacity: 8,
    };
    let mut result = Vec::with_capacity(count);
    let mut sequence = 0_u64;
    while result.len() < count {
        let remaining = count - result.len();
        let run = remaining.min(region_len);
        for _ in 0..run {
            result.push(CausalFrontierInstruction {
                sequence,
                physical_warp,
                pipeline: scalar,
                latency_cycles: 1,
                initiation_interval_cycles: 1,
                in_flight_capacity: 8,
                completion_units: 1,
                control_clean: true,
                dependency_free: true,
                external_side_effect: false,
            });
            sequence += 1;
        }
        if result.len() < count {
            result.push(CausalFrontierInstruction {
                sequence,
                physical_warp,
                pipeline: scalar,
                latency_cycles: 1,
                initiation_interval_cycles: 1,
                in_flight_capacity: 8,
                completion_units: 1,
                control_clean: true,
                dependency_free: true,
                external_side_effect: true,
            });
            sequence += 1;
        }
    }
    if shuffled {
        shuffle(&mut result, seed ^ physical_warp as u64);
    }
    result
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = parse_options().map_err(|error| format!("{error}; usage: {}", usage()))?;
    let build_started = Instant::now();
    let base = options.n / 4;
    let remainder = options.n % 4;
    let programs = (0..4)
        .map(|warp| {
            let count = base + usize::from((warp as usize) < remainder);
            program(
                count,
                warp,
                options.region_len,
                options.shuffled,
                options.seed,
            )
        })
        .collect::<Vec<_>>();
    let build_input_ns = build_started.elapsed().as_nanos();
    let residency = if options.complete_residency {
        CausalFrontierResidency::Complete {
            physical_warps: vec![0, 1, 2, 3],
        }
    } else {
        CausalFrontierResidency::Unknown
    };
    let api_started = Instant::now();
    let report = summarize_causal_frontier_with_residency(&programs, residency)?;
    let api_wall_ns = api_started.elapsed().as_nanos();
    let mut refusal_reasons = BTreeMap::<String, u64>::new();
    for detail in &report.reified_region_details {
        *refusal_reasons
            .entry(format!("{:?}", detail.reason))
            .or_default() += 1;
    }
    let residency_name = if options.complete_residency {
        "complete"
    } else {
        "unknown"
    };
    let order_name = if options.shuffled {
        "shuffled"
    } else {
        "sorted"
    };
    let case_key = format!(
        "n{}-r{}-{}-{}-seed{}",
        options.n, options.region_len, residency_name, order_name, options.seed
    );
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "schema": "phasefuse_scale_v1",
            "case_key": case_key,
            "seed": options.seed,
            "n_records": options.n,
            "program_count": programs.len(),
            "region_len": options.region_len,
            "residency": residency_name,
            "input_order": order_name,
            "build_input_ns": build_input_ns,
            "api_wall_ns": api_wall_ns,
            "planner_time_ns": report.planner_time_ns,
            "exact_execution_steps": report.exact_execution_steps,
            "causal_execution_steps": report.causal_execution_steps,
            "exact_final_cycle": report.exact_final_cycle,
            "causal_final_cycle": report.causal_final_cycle,
            "candidate_regions": report.candidate_regions,
            "collapsed_regions": report.collapsed_regions,
            "reified_regions": report.reified_regions,
            "candidate_instructions": report.candidate_instructions,
            "collapsed_instructions": report.collapsed_instructions,
            "terminal_state_matches": report.terminal_state_matches,
            "refusal_reasons": refusal_reasons,
        }))?
    );
    Ok(())
}
