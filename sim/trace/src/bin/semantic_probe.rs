use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::Read;
use std::path::Path;
use std::process::ExitCode;

use serde::Deserialize;
use sim_semantics::{
    ControlEvidence, InstructionClassification, NativeInstructionClassifier, OperationKind,
};
use sim_trace::TraceDir;
use sim_trace::semantic::{sm86_opcode_head, visit_trace, write_roundtrip};

#[derive(Debug, Deserialize)]
struct ControlSidecar {
    arch: String,
    functions: BTreeMap<String, Vec<ControlInstruction>>,
}
#[derive(Debug, Deserialize)]
struct ControlInstruction {
    pc: u64,
    opcode_text: String,
    status: String,
    control: RawControl,
}
#[derive(Debug, Deserialize)]
struct RawControl {
    stall: u8,
    yield_flag: u8,
    write_barrier: u8,
    read_barrier: u8,
    wait_mask: u8,
    reuse: u8,
}
struct SidecarClassifier {
    by_key: BTreeMap<(i32, u64), InstructionClassification>,
}
impl SidecarClassifier {
    fn load(trace: &TraceDir, path: &Path) -> Result<Self, String> {
        const MAX_CONTROL_BYTES: u64 = 64 * 1024 * 1024;
        let mut file = fs::File::open(path).map_err(|e| e.to_string())?;
        let mut raw = Vec::new();
        file.by_ref()
            .take(MAX_CONTROL_BYTES.saturating_add(1))
            .read_to_end(&mut raw)
            .map_err(|e| e.to_string())?;
        if raw.len() as u64 > MAX_CONTROL_BYTES {
            return Err(format!("control sidecar exceeds {MAX_CONTROL_BYTES} bytes"));
        }
        let sidecar: ControlSidecar = sim_trace::strict_json_from_slice(&raw)
            .map_err(|e| format!("could not parse control sidecar: {e}"))?;
        if sidecar.arch != "sm_86" {
            return Err(format!(
                "control sidecar architecture {:?} is not sm_86",
                sidecar.arch
            ));
        }
        let exec = trace.exec_info().ok_or("trace lacks execution metadata")?;
        let mut by_key = BTreeMap::new();
        let mut joined_functions = BTreeMap::new();
        for (function_id, kernel) in &exec.kernels {
            let Some(entries) = sidecar.functions.get(&kernel.function_name) else {
                return Err(format!(
                    "control sidecar is missing function {:?}",
                    kernel.function_name
                ));
            };
            let mut observed_pcs = std::collections::BTreeSet::new();
            for entry in entries {
                if entry.status != "decoded" {
                    return Err(format!(
                        "control instruction at pc {} is not decoded",
                        entry.pc
                    ));
                }
                if !observed_pcs.insert(entry.pc) {
                    return Err(format!(
                        "control sidecar repeats pc {} in function {:?}",
                        entry.pc, kernel.function_name
                    ));
                }
                let yield_flag = match entry.control.yield_flag {
                    0 => false,
                    1 => true,
                    value => {
                        return Err(format!(
                            "control yield_flag {value} at pc {} is not 0 or 1",
                            entry.pc
                        ));
                    }
                };
                by_key.insert(
                    (*function_id, entry.pc),
                    InstructionClassification {
                        opcode_text: entry.opcode_text.clone(),
                        operation: classify_opcode(&entry.opcode_text),
                        control: Some(ControlEvidence {
                            stall: entry.control.stall,
                            yield_flag,
                            write_barrier: entry.control.write_barrier,
                            read_barrier: entry.control.read_barrier,
                            wait_mask: entry.control.wait_mask,
                            reuse: entry.control.reuse,
                        }),
                    },
                );
            }
            if observed_pcs != kernel.static_pcs {
                return Err(format!(
                    "control sidecar pc set disagrees with execution metadata for {:?}",
                    kernel.function_name
                ));
            }
            joined_functions.insert(kernel.function_name.clone(), ());
        }
        if sidecar
            .functions
            .keys()
            .any(|name| !joined_functions.contains_key(name))
        {
            return Err(
                "control sidecar contains a function absent from execution metadata".to_owned(),
            );
        }
        Ok(Self { by_key })
    }
}
impl NativeInstructionClassifier for SidecarClassifier {
    fn classify(&self, function_unique_id: i32, pc: u64) -> Option<InstructionClassification> {
        self.by_key.get(&(function_unique_id, pc)).cloned()
    }
}
// This classifies known SM86 mnemonic families only. It does not establish
// modifier semantics, dependencies, execution latency, or skipping eligibility.
fn classify_opcode(text: &str) -> OperationKind {
    match sm86_opcode_head(text) {
        "HMMA" | "IMMA" | "BMMA" => OperationKind::Matrix,
        "BAR" | "MEMBAR" | "DEPBAR" | "LDGDEPBAR" | "WARPSYNC" => OperationKind::Synchronization,
        "BRA" | "BRX" | "JMP" | "EXIT" | "RET" | "CALL" | "BSSY" | "BSYNC" | "B2R" | "BREAK" => {
            OperationKind::BranchControl
        }
        "LD" | "LDC" | "ULDC" | "LDG" | "LDGSTS" | "LDS" | "LDSM" | "LDL" | "ST" | "STG"
        | "STS" | "STL" | "ATOM" | "ATOMG" | "ATOMS" | "RED" | "TEX" => OperationKind::Memory,
        "F2FP" | "F2I" | "FADD" | "FCHK" | "FFMA" | "FMNMX" | "FMUL" | "FSEL" | "FSETP"
        | "HFMA2" | "HADD2" | "HMUL2" | "HSETP2" | "I2F" | "I2FP" | "IABS" | "IADD3" | "IMAD"
        | "ISETP" | "LEA" | "LOP3" | "MUFU" | "PLOP3" | "PRMT" | "SEL" | "SHF" | "UIADD3"
        | "UIMAD" | "UISETP" | "ULEA" | "ULOP3" | "USHF" | "POPC" | "FLO" => {
            OperationKind::Arithmetic
        }
        "MOV" | "UMOV" | "R2UR" => OperationKind::RegisterMove,
        "CS2R" | "S2R" | "S2UR" => OperationKind::SpecialRegister,
        "SHFL" | "VOTE" => OperationKind::LaneExchange,
        "CCTL" => OperationKind::CacheControl,
        "YIELD" => OperationKind::SchedulingHint,
        "NOP" => OperationKind::NoOp,
        _ => OperationKind::Unknown,
    }
}

fn main() -> ExitCode {
    let mut args = env::args_os().skip(1);
    let Some(root) = args.next() else {
        eprintln!("usage: semantic-probe TRACE_ROOT [CONTROL_SIDECAR_JSON]");
        return ExitCode::from(2);
    };
    let trace = match TraceDir::open_direct(root) {
        Ok(value) => value,
        Err(e) => {
            eprintln!("semantic-probe: trace: {e}");
            return ExitCode::from(1);
        }
    };
    let classifier = match args.next() {
        Some(path) => match SidecarClassifier::load(&trace, path.as_ref()) {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("semantic-probe: {e}");
                return ExitCode::from(1);
            }
        },
        None => None,
    };
    let roundtrip = match args.next() {
        None => None,
        Some(flag) if flag == "--roundtrip" => match args.next() {
            Some(path) if args.next().is_none() => Some(path),
            _ => {
                eprintln!(
                    "usage: semantic-probe TRACE_ROOT [CONTROL_SIDECAR_JSON] [--roundtrip OUT_DIR]"
                );
                return ExitCode::from(2);
            }
        },
        Some(_) => {
            eprintln!(
                "usage: semantic-probe TRACE_ROOT [CONTROL_SIDECAR_JSON] [--roundtrip OUT_DIR]"
            );
            return ExitCode::from(2);
        }
    };
    let summary = match visit_trace(
        &trace,
        classifier
            .as_ref()
            .map(|c| c as &dyn NativeInstructionClassifier),
        |_| Ok(()),
    ) {
        Ok(summary) => summary,
        Err(e) => {
            eprintln!("semantic-probe: semantic adapter: {e}");
            return ExitCode::from(1);
        }
    };
    if let Some(destination) = roundtrip {
        if let Err(error) = write_roundtrip(
            &trace,
            destination.as_ref(),
            classifier
                .as_ref()
                .map(|classifier| classifier as &dyn NativeInstructionClassifier),
        ) {
            eprintln!("semantic-probe: roundtrip: {error}");
            return ExitCode::from(1);
        }
    }
    match serde_json::to_string_pretty(&summary) {
        Ok(doc) => {
            println!("{doc}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("semantic-probe: output: {e}");
            ExitCode::from(1)
        }
    }
}
