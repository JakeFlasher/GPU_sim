use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::json;
use sim_sm::{
    ControlSidecar, FixedLatencyOperandStatus, SassSourceOperand, project_fixed_latency_operands,
};

fn usage(program: &str) -> ! {
    eprintln!("usage: {program} --out OUTPUT.json SIDECAR.json [SIDECAR.json ...]");
    std::process::exit(2);
}

fn increment(map: &mut BTreeMap<String, u64>, key: impl Into<String>) {
    let value = map.entry(key.into()).or_default();
    *value = value.saturating_add(1);
}

fn main() {
    let mut arguments = std::env::args();
    let program = arguments
        .next()
        .unwrap_or_else(|| "operand_projection_audit".to_owned());
    let remaining = arguments.collect::<Vec<_>>();
    if remaining.len() < 3 || remaining[0] != "--out" {
        usage(&program);
    }
    let output = PathBuf::from(&remaining[1]);
    let inputs = &remaining[2..];

    let mut total_records = 0u64;
    let mut projected = 0u64;
    let mut not_applicable = 0u64;
    let mut gaps = 0u64;
    let mut opcodes = BTreeMap::new();
    let mut families = BTreeMap::new();
    let mut reuse_masks = BTreeMap::new();
    let mut source_bank_demands = BTreeMap::new();
    let mut destination_bank_demands = BTreeMap::new();
    let mut gap_examples = Vec::new();
    let mut per_file = Vec::new();
    let mut per_function = Vec::new();

    for input in inputs {
        let path = Path::new(input);
        let sidecar = match ControlSidecar::load(path, "sm_86") {
            Ok(sidecar) => sidecar,
            Err(error) => {
                eprintln!("{}: {error}", path.display());
                std::process::exit(1);
            }
        };
        let mut file_records = 0u64;
        let mut file_projected = 0u64;
        let mut file_gaps = 0u64;
        for (function, controls) in &sidecar.functions {
            let mut function_projected = 0u64;
            let mut function_not_applicable = 0u64;
            let mut function_gaps = 0u64;
            let mut function_opcodes = BTreeMap::new();
            let mut function_reuse_masks = BTreeMap::new();
            let mut function_source_bank_demands = BTreeMap::new();
            let mut function_destination_bank_demands = BTreeMap::new();
            let mut function_instructions = Vec::new();
            for (&pc, instruction) in &controls.by_pc {
                total_records = total_records.saturating_add(1);
                file_records = file_records.saturating_add(1);
                match project_fixed_latency_operands(
                    &instruction.opcode_text,
                    instruction.control,
                    sim_arch::ArchId::Sm86,
                ) {
                    FixedLatencyOperandStatus::NotApplicable => {
                        not_applicable = not_applicable.saturating_add(1);
                        function_not_applicable = function_not_applicable.saturating_add(1);
                    }
                    FixedLatencyOperandStatus::Projected(projection) => {
                        projected = projected.saturating_add(1);
                        file_projected = file_projected.saturating_add(1);
                        function_projected = function_projected.saturating_add(1);
                        increment(&mut opcodes, format!("{:?}", projection.opcode()));
                        increment(&mut function_opcodes, format!("{:?}", projection.opcode()));
                        increment(&mut families, format!("{:?}", projection.family()));
                        increment(&mut reuse_masks, format!("0x{:x}", projection.reuse_mask()));
                        increment(
                            &mut function_reuse_masks,
                            format!("0x{:x}", projection.reuse_mask()),
                        );
                        increment(
                            &mut source_bank_demands,
                            format!("{:?}", projection.sm86_regular_source_bank_reads()),
                        );
                        increment(
                            &mut function_source_bank_demands,
                            format!("{:?}", projection.sm86_regular_source_bank_reads()),
                        );
                        increment(
                            &mut destination_bank_demands,
                            format!("{:?}", projection.sm86_regular_destination_bank_writes()),
                        );
                        increment(
                            &mut function_destination_bank_demands,
                            format!("{:?}", projection.sm86_regular_destination_bank_writes()),
                        );
                        function_instructions.push(json!({
                            "pc": pc,
                            "opcode_text": instruction.opcode_text,
                            "opcode": format!("{:?}", projection.opcode()),
                            "reuse_mask": projection.reuse_mask(),
                            "source_bank_reads": projection.sm86_regular_source_bank_reads(),
                            "destination_bank_writes": projection.sm86_regular_destination_bank_writes(),
                            "destinations": projection
                                .destinations()
                                .iter()
                                .map(|operand| format!("{:?}", operand))
                                .collect::<Vec<_>>(),
                            "sources": projection
                                .sources()
                                .iter()
                                .map(|operand| format!("{:?}", operand))
                                .collect::<Vec<_>>(),
                            "predicate_sources": projection
                                .predicate_sources()
                                .iter()
                                .map(|operand| format!("{:?}", operand))
                                .collect::<Vec<_>>(),
                            "control": {
                                "stall": instruction.control.stall,
                                "yield_flag": instruction.control.yield_flag,
                            },
                        }));
                        // A projection preserves one slot for every data
                        // source, including immediate/constant operands.
                        let register_sources = projection
                            .sources()
                            .iter()
                            .filter(|source| matches!(source, SassSourceOperand::Register(_)))
                            .count();
                        if register_sources > projection.sources().len() {
                            unreachable!("register subset cannot exceed sources");
                        }
                    }
                    FixedLatencyOperandStatus::Gap(gap) => {
                        gaps = gaps.saturating_add(1);
                        file_gaps = file_gaps.saturating_add(1);
                        function_gaps = function_gaps.saturating_add(1);
                        if gap_examples.len() < 32 {
                            gap_examples.push(json!({
                                "path": input,
                                "function": function,
                                "pc": pc,
                                "opcode_text": instruction.opcode_text,
                                "gap": format!("{:?}", gap),
                            }));
                        }
                    }
                }
            }
            per_function.push(json!({
                "path": input,
                "function": function,
                "records": controls.by_pc.len(),
                "projected": function_projected,
                "not_applicable": function_not_applicable,
                "gaps": function_gaps,
                "opcodes": function_opcodes,
                "reuse_masks": function_reuse_masks,
                "source_bank_demands": function_source_bank_demands,
                "destination_bank_demands": function_destination_bank_demands,
                "instructions": function_instructions,
            }));
        }
        per_file.push(json!({
            "path": input,
            "records": file_records,
            "projected": file_projected,
            "gaps": file_gaps,
        }));
    }

    let document = json!({
        "schema_version": 1,
        "architecture": "sm_86",
        "files": inputs.len(),
        "records": total_records,
        "projected": projected,
        "not_applicable": not_applicable,
        "gaps": gaps,
        "opcodes": opcodes,
        "families": families,
        "reuse_masks": reuse_masks,
        "source_bank_demands": source_bank_demands,
        "destination_bank_demands": destination_bank_demands,
        "gap_examples": gap_examples,
        "per_file": per_file,
        "per_function": per_function,
    });
    let bytes = match serde_json::to_vec_pretty(&document) {
        Ok(mut bytes) => {
            bytes.push(b'\n');
            bytes
        }
        Err(error) => {
            eprintln!("could not serialize audit: {error}");
            std::process::exit(1);
        }
    };
    if let Some(parent) = output.parent()
        && let Err(error) = std::fs::create_dir_all(parent)
    {
        eprintln!("could not create {}: {error}", parent.display());
        std::process::exit(1);
    }
    if let Err(error) = std::fs::write(&output, bytes) {
        eprintln!("could not write {}: {error}", output.display());
        std::process::exit(1);
    }
    println!("{}", output.display());
    if gaps != 0 {
        std::process::exit(1);
    }
}
