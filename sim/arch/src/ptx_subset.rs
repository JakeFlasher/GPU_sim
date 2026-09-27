//! A deliberately small PTX-like source frontend for `simt_its_v1`.
//!
//! This is a bounded compiler seam, not CUDA/PTX compatibility. It lowers a
//! textual subset into the existing functional ITS instruction enum and
//! rejects unsupported ABI, memory-model, predication, vector and call forms.

use crate::its::{ItsError, ItsInstruction, ItsProgram};
use std::collections::BTreeMap;

pub const PTX_SUBSET_CONTRACT: &str = "ptx_subset_v1";
pub const MAX_SOURCE_BYTES: usize = 1024 * 1024;
pub const MAX_SOURCE_LINES: usize = 100_000;
pub const MAX_SOURCE_LINE_BYTES: usize = 16 * 1024;
pub const MAX_INITIAL_MEMORY_WORDS: usize = 262_144;
const MAX_INSTRUCTIONS: usize = 4096;

#[derive(Debug, thiserror::Error)]
pub enum PtxSubsetError {
    #[error("PTX subset resource limit: {0}")]
    ResourceLimit(String),
    #[error("PTX subset line {line}: {detail}")]
    Invalid { line: usize, detail: String },
    #[error("PTX subset label {0:?} is undefined")]
    UndefinedLabel(String),
    #[error("PTX subset label {0:?} is duplicated")]
    DuplicateLabel(String),
    #[error("PTX subset cannot lower an ITS program: {0}")]
    Program(#[from] ItsError),
}

#[derive(Debug)]
enum PendingInstruction {
    Ready(ItsInstruction),
    Branch { pred: Option<u8>, label: String },
}

/// Compile the admitted PTX-like subset into one bounded `simt_its_v1`
/// program. The `architecture` string must match the supplied ITS spec.
pub fn compile_subset(
    source: &str,
    architecture: impl Into<String>,
    max_steps: u64,
) -> Result<ItsProgram, PtxSubsetError> {
    compile_subset_with_memory(source, architecture, max_steps, BTreeMap::new())
}

/// Compile the subset with an explicit initial u32 memory image.  Keeping the
/// input image outside the source makes input-varying control-flow tests
/// reproducible without mutating generated program files.
pub fn compile_subset_with_memory(
    source: &str,
    architecture: impl Into<String>,
    max_steps: u64,
    initial_memory: BTreeMap<u32, u32>,
) -> Result<ItsProgram, PtxSubsetError> {
    if source.len() > MAX_SOURCE_BYTES {
        return Err(PtxSubsetError::ResourceLimit(format!(
            "source is {} bytes, limit is {MAX_SOURCE_BYTES}",
            source.len()
        )));
    }
    if initial_memory.len() > MAX_INITIAL_MEMORY_WORDS {
        return Err(PtxSubsetError::ResourceLimit(format!(
            "initial memory has {} words, limit is {MAX_INITIAL_MEMORY_WORDS}",
            initial_memory.len()
        )));
    }
    let mut labels = BTreeMap::new();
    let mut pending = Vec::new();
    let mut entry_declarations = 0_u32;
    for (line_index, raw_line) in source.lines().enumerate() {
        let line_number = line_index + 1;
        if line_number > MAX_SOURCE_LINES {
            return Err(PtxSubsetError::ResourceLimit(format!(
                "source exceeds {MAX_SOURCE_LINES} lines"
            )));
        }
        if raw_line.len() > MAX_SOURCE_LINE_BYTES {
            return Err(PtxSubsetError::ResourceLimit(format!(
                "line {line_number} exceeds {MAX_SOURCE_LINE_BYTES} bytes"
            )));
        }
        let line = raw_line.split("//").next().unwrap_or("").trim();
        if line.is_empty() || matches!(line, "{" | "}" | ")" | "(") {
            continue;
        }
        if line.starts_with('.') {
            // Only the entry declaration is syntax accepted by this frontend.
            // Silently dropping arbitrary PTX directives would turn ABI,
            // address-size, and memory-model mismatches into false execution.
            if line.starts_with(".visible .entry ") || line.starts_with(".entry ") {
                entry_declarations = entry_declarations.saturating_add(1);
                if entry_declarations > 1 {
                    return Err(invalid(
                        line_number,
                        "only one .entry declaration is admitted",
                    ));
                }
                continue;
            }
            return Err(invalid(
                line_number,
                "unsupported PTX directive; only .entry declarations are admitted",
            ));
        }
        if let Some(label) = line.strip_suffix(':') {
            let label = label.trim().to_owned();
            if label.is_empty() {
                return Err(invalid(line_number, "empty label"));
            }
            if labels.insert(label.clone(), pending.len()).is_some() {
                return Err(PtxSubsetError::DuplicateLabel(label));
            }
            continue;
        }
        if pending.len() >= MAX_INSTRUCTIONS {
            return Err(PtxSubsetError::ResourceLimit(format!(
                "instruction count exceeds {MAX_INSTRUCTIONS}"
            )));
        }
        pending.push(parse_line(line_number, line)?);
    }

    let mut instructions = Vec::with_capacity(pending.len() + 1);
    for item in pending {
        match item {
            PendingInstruction::Ready(instruction) => instructions.push(instruction),
            PendingInstruction::Branch { pred, label } => {
                let target = labels
                    .get(&label)
                    .copied()
                    .ok_or_else(|| PtxSubsetError::UndefinedLabel(label.clone()))?;
                let target = u32::try_from(target)
                    .map_err(|_| invalid(0, "label target exceeds u32 PC domain"))?;
                instructions.push(match pred {
                    Some(pred) => ItsInstruction::BranchIf { pred, target },
                    None => ItsInstruction::Branch { target },
                });
            }
        }
    }
    if !matches!(instructions.last(), Some(ItsInstruction::Halt)) {
        instructions.push(ItsInstruction::Halt);
    }
    ItsProgram::from_instructions_with_memory(architecture, max_steps, instructions, initial_memory)
        .map_err(PtxSubsetError::Program)
}

fn invalid(line: usize, detail: impl Into<String>) -> PtxSubsetError {
    PtxSubsetError::Invalid {
        line,
        detail: detail.into(),
    }
}

fn parse_line(line_number: usize, source: &str) -> Result<PendingInstruction, PtxSubsetError> {
    let mut text = source.trim().trim_end_matches(';').trim().to_owned();
    let mut predicate = None;
    if let Some(rest) = text.strip_prefix('@') {
        let (pred, remainder) = rest.split_once(char::is_whitespace).ok_or_else(|| {
            invalid(
                line_number,
                "predication must be followed by an instruction",
            )
        })?;
        if pred.starts_with('!') {
            return Err(invalid(line_number, "negated predicates are not admitted"));
        }
        predicate = Some(parse_predicate(line_number, pred)?);
        text = remainder.trim().to_owned();
    }
    let tokens = text
        .replace([',', '[', ']'], " ")
        .split_whitespace()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let Some(op) = tokens.first().map(String::as_str) else {
        return Err(invalid(line_number, "empty instruction"));
    };
    if predicate.is_some() && !op.starts_with("bra") {
        return Err(invalid(
            line_number,
            "only predicated branch instructions are admitted",
        ));
    }
    if predicate.is_some() && op == "bra.uni" {
        return Err(invalid(
            line_number,
            "bra.uni is uniform and cannot carry a lane-varying predicate",
        ));
    }
    match op {
        "mov.u32" => {
            require_arity(line_number, &tokens, 3)?;
            if tokens[2] != "%laneid" {
                return Err(invalid(
                    line_number,
                    "mov.u32 only accepts %laneid as source",
                ));
            }
            Ok(PendingInstruction::Ready(ItsInstruction::SetLaneId {
                dst: parse_register(line_number, &tokens[1])?,
            }))
        }
        "add.s32" => {
            require_arity(line_number, &tokens, 4)?;
            Ok(PendingInstruction::Ready(ItsInstruction::AddImm {
                dst: parse_register(line_number, &tokens[1])?,
                src: parse_register(line_number, &tokens[2])?,
                imm: parse_i32(line_number, &tokens[3])?,
            }))
        }
        "mul.lo.u32" => {
            require_arity(line_number, &tokens, 4)?;
            Ok(PendingInstruction::Ready(ItsInstruction::MulImm {
                dst: parse_register(line_number, &tokens[1])?,
                src: parse_register(line_number, &tokens[2])?,
                imm: parse_u32(line_number, &tokens[3])?,
            }))
        }
        "setp.lt.s32" => {
            require_arity(line_number, &tokens, 4)?;
            Ok(PendingInstruction::Ready(ItsInstruction::CmpLtImm {
                dst: parse_predicate(line_number, &tokens[1])?,
                src: parse_register(line_number, &tokens[2])?,
                imm: parse_i32(line_number, &tokens[3])?,
            }))
        }
        "ld.global.u32" => {
            require_arity(line_number, &tokens, 3)?;
            Ok(PendingInstruction::Ready(ItsInstruction::LoadIndirect {
                dst: parse_register(line_number, &tokens[1])?,
                address: parse_register(line_number, &tokens[2])?,
            }))
        }
        "st.global.u32" => {
            require_arity(line_number, &tokens, 3)?;
            Ok(PendingInstruction::Ready(ItsInstruction::StoreIndirect {
                address: parse_register(line_number, &tokens[1])?,
                src: parse_register(line_number, &tokens[2])?,
            }))
        }
        "bra" | "bra.uni" => {
            require_arity(line_number, &tokens, 2)?;
            Ok(PendingInstruction::Branch {
                pred: predicate,
                label: tokens[1].clone(),
            })
        }
        "exit" => {
            require_arity(line_number, &tokens, 1)?;
            Ok(PendingInstruction::Ready(ItsInstruction::Halt))
        }
        _ => Err(invalid(
            line_number,
            format!("unsupported opcode {op:?}; use the documented ptx_subset_v1 subset"),
        )),
    }
}

fn require_arity(line: usize, tokens: &[String], expected: usize) -> Result<(), PtxSubsetError> {
    if tokens.len() == expected {
        Ok(())
    } else {
        Err(invalid(
            line,
            format!("expected {expected} tokens, got {}", tokens.len()),
        ))
    }
}

fn parse_register(line: usize, token: &str) -> Result<u8, PtxSubsetError> {
    token
        .strip_prefix("%r")
        .ok_or_else(|| invalid(line, format!("expected %rN register, got {token:?}")))?
        .parse::<u8>()
        .map_err(|_| invalid(line, format!("invalid register {token:?}")))
}

fn parse_predicate(line: usize, token: &str) -> Result<u8, PtxSubsetError> {
    token
        .strip_prefix("%p")
        .ok_or_else(|| invalid(line, format!("expected %pN predicate, got {token:?}")))?
        .parse::<u8>()
        .map_err(|_| invalid(line, format!("invalid predicate {token:?}")))
}

fn parse_i32(line: usize, token: &str) -> Result<i32, PtxSubsetError> {
    if let Some(hex) = token
        .strip_prefix("0x")
        .or_else(|| token.strip_prefix("0X"))
    {
        return u32::from_str_radix(hex, 16)
            .map(|value| value as i32)
            .map_err(|_| invalid(line, format!("expected signed immediate, got {token:?}")));
    }
    token
        .parse::<i32>()
        .map_err(|_| invalid(line, format!("expected signed immediate, got {token:?}")))
}

fn parse_u32(line: usize, token: &str) -> Result<u32, PtxSubsetError> {
    if let Some(hex) = token
        .strip_prefix("0x")
        .or_else(|| token.strip_prefix("0X"))
    {
        return u32::from_str_radix(hex, 16)
            .map_err(|_| invalid(line, format!("expected unsigned immediate, got {token:?}")));
    }
    token
        .parse::<u32>()
        .map_err(|_| invalid(line, format!("expected unsigned immediate, got {token:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::its::{ItsError, ItsExecutor, ItsInstruction};
    use crate::spec::ArchitectureSpec;
    use std::path::Path;

    #[test]
    fn compiles_value_dependent_subset_and_resolves_labels() {
        let source = r#"
            .visible .entry demo() {
            mov.u32 %r0, %laneid;
            mul.lo.u32 %r1, %r0, 4;
            ld.global.u32 %r2, [%r1];
            setp.lt.s32 %p0, %r2, 10;
            @%p0 bra fast;
            add.s32 %r2, %r2, 100;
            bra done;
        fast:
            add.s32 %r2, %r2, 200;
        done:
            st.global.u32 [%r1], %r2;
            exit;
            }
        "#;
        let program = compile_subset(source, "probe", 32).expect("subset should compile");
        assert_eq!(program.instructions().len(), 10);
        assert!(matches!(
            program.instructions()[4],
            ItsInstruction::BranchIf { target: 7, .. }
        ));
        assert!(matches!(
            program.instructions().last(),
            Some(ItsInstruction::Halt)
        ));
    }

    #[test]
    fn rejects_unsupported_predication_and_opcode() {
        assert!(matches!(
            compile_subset("@!%p0 bra done; done: exit;", "probe", 8),
            Err(PtxSubsetError::Invalid { .. })
        ));
        assert!(matches!(
            compile_subset("atom.global.add.u32 %r0, [%r1], %r2; exit;", "probe", 8),
            Err(PtxSubsetError::Invalid { .. })
        ));
        assert!(matches!(
            compile_subset(".address_size 64\nexit;", "probe", 8),
            Err(PtxSubsetError::Invalid { .. })
        ));
        assert!(matches!(
            compile_subset("ret;", "probe", 8),
            Err(PtxSubsetError::Invalid { .. })
        ));
        assert!(matches!(
            compile_subset(".entry a()\n.entry b()\nexit;", "probe", 8),
            Err(PtxSubsetError::Invalid { .. })
        ));
        assert!(matches!(
            compile_subset("mov.u32 %r32, %laneid;\nexit;", "probe", 8),
            Err(PtxSubsetError::Program(ItsError::InvalidProgram(_)))
        ));
        assert!(matches!(
            compile_subset("setp.lt.s32 %p8, %r0, 1;\nexit;", "probe", 8),
            Err(PtxSubsetError::Program(ItsError::InvalidProgram(_)))
        ));
    }

    #[test]
    fn compiles_executes_and_varies_signed_memory_inputs() {
        let source = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../arch_profiles/programs/simt_its_subset.ptx"
        ));
        let spec = ArchitectureSpec::parse(
            Path::new("spec.json"),
            br#"
            {
              "schema_version": 1,
              "architecture": "probe",
              "isa": "simt_its_v1",
              "instruction_set": ["set_lane_id", "mul_imm", "load_indirect", "cmp_lt_imm", "branch_if", "add_imm", "branch", "store_indirect", "halt"],
              "execution": {"simt_width": 4, "max_resident_groups": 1},
              "resources": [{"name":"alu","kind":"compute","issue_width":1,"latency_cycles":1}],
              "evidence": "unvalidated"
            }
            "#,
        )
        .expect("spec should parse");
        let all_true = compile_subset_with_memory(
            source,
            "probe",
            64,
            [(0, 1), (4, 2), (8, 3), (12, 4)].into_iter().collect(),
        )
        .expect("all-true source should compile");
        let mixed = compile_subset_with_memory(
            source,
            "probe",
            64,
            [(0, u32::MAX), (4, 12), (8, 3), (12, 14)]
                .into_iter()
                .collect(),
        )
        .expect("mixed source should compile");
        let all_true = ItsExecutor::new(&spec, &all_true)
            .expect("all-true program should be admitted")
            .run_traced()
            .expect("all-true execution should complete");
        let mixed = ItsExecutor::new(&spec, &mixed)
            .expect("mixed program should be admitted")
            .run_traced()
            .expect("mixed execution should complete");
        assert_eq!(
            all_true.memory,
            [(0, 201), (4, 202), (8, 203), (12, 204)]
                .into_iter()
                .collect()
        );
        // 0xffff_ffff is -1 under setp.lt.s32 and therefore takes the fast
        // path; this catches accidental unsigned comparison in the executor.
        assert_eq!(
            mixed.memory,
            [(0, 199), (4, 112), (8, 203), (12, 114)]
                .into_iter()
                .collect()
        );
        let branch = mixed
            .dynamic_trace
            .as_ref()
            .expect("trace should be present")
            .iter()
            .find(|record| matches!(record.instruction, ItsInstruction::BranchIf { .. }))
            .expect("branch should be present");
        assert_eq!(
            branch.successor_masks,
            [(5, 0b1010), (7, 0b0101)].into_iter().collect()
        );
    }
}
