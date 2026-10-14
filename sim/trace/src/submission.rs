//! Typed submission/dependency evidence from real SM120 static metadata.
//!
//! This module deliberately stops at instruction-family and encoded-control
//! obligations.  It does not turn a virtual queue, scoreboard id, or control
//! field into a physical port, capacity, service latency, or engine-completion
//! timestamp.  The input is the NVBit-native `native_static_metadata.json`
//! emitted beside a real capture; no synthetic fixture is accepted or needed.

use crate::async_sidecars::{BarrierOperation, barrier_operation};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

const MAX_METADATA_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Debug, Clone, thiserror::Error)]
#[error("SM120 submission metadata {path}: {message}")]
pub struct SubmissionError {
    pub path: PathBuf,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SubmissionClass {
    TmaLoad,
    TmaStore,
    BulkCopy,
    AsyncCopy,
    AsyncGroupCommit,
    AsyncGroupWait,
    BarrierArrival,
    BarrierQuery,
    BarrierExchange,
    ScoreboardControl,
    Other,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ControlEvidence {
    pub available: bool,
    pub control_word_hex: Option<String>,
    pub stall_count: Option<u8>,
    pub yield_bit: Option<bool>,
    pub new_read_barrier_id: Option<u8>,
    pub new_write_barrier_id: Option<u8>,
    pub wait_barrier_mask: Option<u8>,
    pub source: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SubmissionInstruction {
    pub function_id: Option<i32>,
    pub kernel_name: String,
    pub pc: u64,
    pub opcode: String,
    pub class: SubmissionClass,
    /// Opcode-form barrier semantics, when the visible spelling identifies
    /// one.  This remains architectural obligation metadata; it is not a
    /// physical queue or engine assignment.
    pub barrier_operation: Option<BarrierOperation>,
    pub control: ControlEvidence,
    /// This is a semantic interpretation of the visible instruction family,
    /// not a claim about a physical unit.  It stays explicit in the report so
    /// callers cannot confuse static metadata with hardware timing evidence.
    pub interpretation: &'static str,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct SubmissionSummary {
    pub metadata_available: bool,
    pub metadata_bytes: u64,
    pub kernel_count: u64,
    pub instruction_count: u64,
    pub control_available_instructions: u64,
    pub class_counts: BTreeMap<SubmissionClass, u64>,
    pub instructions: Vec<SubmissionInstruction>,
}

#[derive(Debug, Deserialize)]
struct StaticMetadata {
    kernels: Option<Vec<StaticKernel>>,
}

#[derive(Debug, Deserialize)]
struct StaticKernel {
    kernel_name: Option<String>,
    unique_function_id: Option<i32>,
    instructions: Option<Vec<StaticInstruction>>,
}

#[derive(Debug, Deserialize)]
struct StaticInstruction {
    pc_num_dec: Option<u64>,
    opcode: Option<String>,
    op_code: Option<String>,
    sass: Option<String>,
    control_word_hex: Option<String>,
    stall_count: Option<u8>,
    yield_bit: Option<u8>,
    new_read_barrier_id: Option<u8>,
    new_write_barrier_id: Option<u8>,
    wait_barrier_mask: Option<u8>,
    control_fields_available: Option<bool>,
    control_decode_source: Option<String>,
    control_fields: Option<StaticControlFields>,
}

#[derive(Debug, Deserialize)]
struct StaticControlFields {
    stall_count: Option<u8>,
    is_yield: Option<bool>,
    id_new_read_barrier: Option<u8>,
    id_new_write_barrier: Option<u8>,
    wait_barrier_bits: Option<u8>,
}

fn head(opcode: &str) -> &str {
    opcode
        .trim()
        .trim_start_matches(|c: char| c == '@' || c == '!')
        .split(|c: char| c.is_ascii_whitespace() || c == ';')
        .next()
        .unwrap_or("")
}

fn classify(opcode: &str, control: &ControlEvidence) -> (SubmissionClass, &'static str) {
    let op = head(opcode);
    if op.starts_with("SYNCS.ARRIVE") || op.starts_with("ARRIVES") {
        return (
            SubmissionClass::BarrierArrival,
            "software/barrier arrival obligation",
        );
    }
    if op.starts_with("SYNCS.PHASECHK")
        || op.starts_with("SYNCS.TRYWAIT")
        || op.starts_with("SYNCS.TESTWAIT")
    {
        return (
            SubmissionClass::BarrierQuery,
            "phase/query observation site",
        );
    }
    if op.starts_with("SYNCS.EXCH") {
        return (
            SubmissionClass::BarrierExchange,
            "barrier state exchange/lifetime site",
        );
    }
    if op.starts_with("UBLKCP") {
        return (
            SubmissionClass::BulkCopy,
            "Blackwell bulk-copy submission; completion remains separate",
        );
    }
    if op.starts_with("UTMALDG") {
        return (
            SubmissionClass::TmaLoad,
            "TMA load submission; engine completion unavailable",
        );
    }
    if op.starts_with("UTMASTG") {
        return (
            SubmissionClass::TmaStore,
            "TMA store submission; engine completion unavailable",
        );
    }
    if op.starts_with("LDGSTS") {
        return (SubmissionClass::AsyncCopy, "LDGSTS async-copy submission");
    }
    if op.starts_with("LDGDEPBAR") {
        return (SubmissionClass::AsyncGroupCommit, "async-copy group commit");
    }
    if op.starts_with("DEPBAR") {
        return (
            SubmissionClass::AsyncGroupWait,
            "async-copy/group wait; membership is separate",
        );
    }
    if control.wait_barrier_mask.unwrap_or(0) != 0
        || control.new_read_barrier_id.is_some_and(|value| value != 7)
        || control.new_write_barrier_id.is_some_and(|value| value != 7)
    {
        return (
            SubmissionClass::ScoreboardControl,
            "encoded dependency/scoreboard control",
        );
    }
    (
        SubmissionClass::Other,
        "no typed async/scoreboard class inferred",
    )
}

fn control(instruction: &StaticInstruction) -> ControlEvidence {
    let fields = instruction.control_fields.as_ref();
    let available = instruction.control_fields_available == Some(true)
        || instruction.control_word_hex.is_some()
        || fields.is_some();
    ControlEvidence {
        available,
        control_word_hex: instruction.control_word_hex.clone(),
        stall_count: fields
            .and_then(|x| x.stall_count)
            .or(instruction.stall_count),
        yield_bit: fields
            .and_then(|x| x.is_yield)
            .or_else(|| instruction.yield_bit.map(|x| x != 0)),
        new_read_barrier_id: fields
            .and_then(|x| x.id_new_read_barrier)
            .or(instruction.new_read_barrier_id),
        new_write_barrier_id: fields
            .and_then(|x| x.id_new_write_barrier)
            .or(instruction.new_write_barrier_id),
        wait_barrier_mask: fields
            .and_then(|x| x.wait_barrier_bits)
            .or(instruction.wait_barrier_mask),
        source: instruction.control_decode_source.clone(),
    }
}

/// Parse a real capture's static metadata. The report keeps every instruction
/// so downstream analysis can join dynamic TMA/synchronization keys by
/// `(function_id, pc)` without guessing from opcode text alone.
pub fn load_metadata(root: &Path) -> Result<SubmissionSummary, SubmissionError> {
    let path = root.join("extra_info/native_static_metadata.json");
    let bytes = fs::read(&path).map_err(|error| SubmissionError {
        path: path.clone(),
        message: error.to_string(),
    })?;
    if bytes.len() as u64 > MAX_METADATA_BYTES {
        return Err(SubmissionError {
            path,
            message: format!("metadata exceeds {MAX_METADATA_BYTES} bytes"),
        });
    }
    let metadata: StaticMetadata =
        serde_json::from_slice(&bytes).map_err(|error| SubmissionError {
            path: path.clone(),
            message: error.to_string(),
        })?;
    let mut summary = SubmissionSummary {
        metadata_available: true,
        metadata_bytes: bytes.len() as u64,
        ..SubmissionSummary::default()
    };
    for kernel in metadata.kernels.unwrap_or_default() {
        summary.kernel_count = summary.kernel_count.saturating_add(1);
        let name = kernel.kernel_name.unwrap_or_else(|| "<unknown>".to_owned());
        for instruction in kernel.instructions.unwrap_or_default() {
            summary.instruction_count = summary.instruction_count.saturating_add(1);
            let control = control(&instruction);
            if control.available {
                summary.control_available_instructions =
                    summary.control_available_instructions.saturating_add(1);
            }
            let opcode = instruction
                .sass
                .or(instruction.opcode)
                .or(instruction.op_code)
                .unwrap_or_else(|| "<unknown>".to_owned());
            let (class, interpretation) = classify(&opcode, &control);
            let operation = barrier_operation(&opcode);
            *summary.class_counts.entry(class).or_default() += 1;
            summary.instructions.push(SubmissionInstruction {
                function_id: kernel.unique_function_id,
                kernel_name: name.clone(),
                pc: instruction.pc_num_dec.unwrap_or(0),
                opcode,
                class,
                barrier_operation: operation,
                control,
                interpretation,
            });
        }
    }
    Ok(summary)
}
