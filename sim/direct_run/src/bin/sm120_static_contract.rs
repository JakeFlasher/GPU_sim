//! Static SM120 metadata ↔ NVBit trace contract inspector.
//!
//! This is a metadata/operand admission tool, not an issue or timing model.
//! It proves that the retained static SASS/control export joins the dynamic
//! trace's `(function_unique_id, pc)` surface without silently dropping
//! records. Unknown control/operand fields remain counts in the report.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sim_arch::ArchId;
use sim_arch::card::{ArchCard, CalibError};
use sim_trace::{TraceDir, TraceError};

const MAX_METADATA_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
enum Error {
    #[error("usage: sm120-static-contract --trace DIR --arch-card FILE")]
    Usage,
    #[error("missing option value for {0}")]
    MissingValue(&'static str),
    #[error("unknown option {0}")]
    Unknown(String),
    #[error("architecture card: {0}")]
    Card(#[from] CalibError),
    #[error("trace: {0}")]
    Trace(#[from] TraceError),
    #[error("trace has no execution metadata")]
    MissingExecInfo,
    #[error("metadata {path} is {observed} bytes, exceeding limit {limit}")]
    MetadataTooLarge {
        path: PathBuf,
        observed: u64,
        limit: u64,
    },
    #[error("metadata {path}: {source}")]
    MetadataParse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("metadata kind {got:?} is not native_static_sm120_v3")]
    MetadataKind { got: String },
    #[error("architecture card {got} is not the SM120 static-contract target")]
    CardArchitecture { got: ArchId },
    #[error("metadata control_latency_claims must be false")]
    MetadataLatencyClaim,
    #[error("metadata kernel {kernel_id} has architecture {got}, expected 120")]
    KernelArchitecture { kernel_id: i32, got: u32 },
    #[error("metadata kernel {kernel_id} is missing from enhanced execution info")]
    KernelMissing { kernel_id: i32 },
    #[error("metadata contains duplicate function_unique_id {kernel_id}")]
    DuplicateKernel { kernel_id: i32 },
    #[error("metadata kernel {kernel_id} has non-positive function_unique_id")]
    InvalidKernel { kernel_id: i32 },
    #[error("metadata kernel {kernel_id} has duplicate static PC {pc:#x}")]
    DuplicateStaticPc { kernel_id: i32, pc: u64 },
    #[error("enhanced kernel {kernel_id} is missing from static metadata")]
    EnhancedKernelMissing { kernel_id: i32 },
    #[error("metadata kernel {kernel_id} has no dynamic trace records")]
    DynamicKernelMissing { kernel_id: i32 },
    #[error("sidecar {path}: {source}")]
    SidecarWrite {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("sidecar serialization: {source}")]
    SidecarSerialize {
        #[source]
        source: serde_json::Error,
    },
    #[error("kernel {kernel_id} static PC sets disagree: metadata={metadata}, exec={exec}")]
    StaticPcMismatch {
        kernel_id: i32,
        metadata: usize,
        exec: usize,
    },
    #[error("kernel {kernel_id} dynamic PC {pc:#x} is absent from static metadata")]
    DynamicPcMissing { kernel_id: i32, pc: u32 },
}

#[derive(Debug, Deserialize)]
struct Metadata {
    metadata_kind: String,
    static_exporter_version: u32,
    control_latency_claims: bool,
    kernels: Vec<MetadataKernel>,
}

#[derive(Debug, Deserialize)]
struct MetadataKernel {
    kernel_name: String,
    unique_function_id: i32,
    architecture_version: u32,
    instructions: Vec<MetadataInstruction>,
}

#[derive(Debug, Deserialize)]
struct MetadataInstruction {
    pc_num_dec: u64,
    #[serde(default)]
    op_code: String,
    #[serde(default)]
    control_fields_available: bool,
    #[serde(default)]
    register_metadata_available: bool,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    status: &'static str,
    note: &'static str,
    metadata_kind: String,
    static_exporter_version: u32,
    trace_arch: u32,
    card_arch: String,
    kernels: Vec<KernelReport>,
}

#[derive(Debug, Serialize)]
struct KernelReport {
    function_unique_id: i32,
    kernel_name: String,
    static_instruction_count: u64,
    dynamic_instruction_count: u64,
    dynamic_pc_count: u64,
    static_dynamic_pc_join: bool,
    control_fields_available: u64,
    control_fields_missing: u64,
    register_metadata_available: u64,
    register_metadata_missing: u64,
    opcode_histogram: BTreeMap<String, u64>,
}

#[derive(Debug, Serialize)]
struct OpaqueControlSidecar {
    schema_version: u32,
    contract: &'static str,
    semantics: &'static str,
    timing_fields_are_uninterpreted: bool,
    source_metadata: serde_json::Value,
}

fn parse_args() -> Result<(PathBuf, PathBuf, Option<PathBuf>), Error> {
    let mut trace = None;
    let mut card = None;
    let mut out = None;
    let mut args = env::args_os().skip(1);
    while let Some(arg) = args.next() {
        let arg = arg.to_string_lossy();
        match arg.as_ref() {
            "--trace" => {
                trace = Some(
                    args.next()
                        .map(PathBuf::from)
                        .ok_or(Error::MissingValue("--trace"))?,
                )
            }
            "--arch-card" => {
                card = Some(
                    args.next()
                        .map(PathBuf::from)
                        .ok_or(Error::MissingValue("--arch-card"))?,
                )
            }
            "--out" => {
                out = Some(
                    args.next()
                        .map(PathBuf::from)
                        .ok_or(Error::MissingValue("--out"))?,
                )
            }
            "--help" | "-h" => return Err(Error::Usage),
            other => return Err(Error::Unknown(other.to_owned())),
        }
    }
    trace
        .zip(card)
        .map(|(trace, card)| (trace, card, out))
        .ok_or(Error::Usage)
}

fn strip_instance_suffix(name: &str) -> &str {
    let Some(pos) = name.rfind("___") else {
        return name;
    };
    let suffix = &name[pos + 3..];
    if !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit()) {
        &name[..pos]
    } else {
        name
    }
}

fn read_metadata(path: &Path) -> Result<(Metadata, serde_json::Value), Error> {
    let file = fs::File::open(path).map_err(|source| Error::MetadataParse {
        path: path.to_path_buf(),
        source: serde_json::Error::io(source),
    })?;
    let mut bytes = Vec::new();
    file.take(MAX_METADATA_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|source| Error::MetadataParse {
            path: path.to_path_buf(),
            source: serde_json::Error::io(source),
        })?;
    if bytes.len() as u64 > MAX_METADATA_BYTES {
        return Err(Error::MetadataTooLarge {
            path: path.to_path_buf(),
            observed: bytes.len() as u64,
            limit: MAX_METADATA_BYTES,
        });
    }
    let raw: serde_json::Value =
        sim_trace::strict_json_from_slice(&bytes).map_err(|source| Error::MetadataParse {
            path: path.to_path_buf(),
            source,
        })?;
    let metadata = serde_json::from_value(raw.clone()).map_err(|source| Error::MetadataParse {
        path: path.to_path_buf(),
        source,
    })?;
    Ok((metadata, raw))
}

fn run(
    trace_root: PathBuf,
    card_path: PathBuf,
    sidecar_path: Option<PathBuf>,
) -> Result<Report, Error> {
    let card = ArchCard::load(&card_path)?;
    let card_arch = card.arch_id()?;
    if card_arch != ArchId::Sm120 {
        return Err(Error::CardArchitecture { got: card_arch });
    }
    let trace = TraceDir::open_direct_compact(&trace_root)?;
    let exec = trace.exec_info().ok_or(Error::MissingExecInfo)?;
    let metadata_path = trace_root.join("extra_info/native_static_metadata.json");
    let (metadata, raw_metadata) = read_metadata(&metadata_path)?;
    if metadata.metadata_kind != "native_static_sm120_v3" {
        return Err(Error::MetadataKind {
            got: metadata.metadata_kind,
        });
    }
    if metadata.static_exporter_version != 3 || metadata.control_latency_claims {
        return Err(Error::MetadataLatencyClaim);
    }
    let mut by_id = BTreeMap::new();
    for kernel in metadata.kernels {
        if kernel.unique_function_id <= 0 {
            return Err(Error::InvalidKernel {
                kernel_id: kernel.unique_function_id,
            });
        }
        if kernel.architecture_version != 120 {
            return Err(Error::KernelArchitecture {
                kernel_id: kernel.unique_function_id,
                got: kernel.architecture_version,
            });
        }
        let function_id = kernel.unique_function_id;
        if by_id.insert(function_id, kernel).is_some() {
            return Err(Error::DuplicateKernel {
                kernel_id: function_id,
            });
        }
    }
    for (&function_id, kernel) in &exec.kernels {
        if kernel.architecture_version != 120 {
            return Err(Error::KernelArchitecture {
                kernel_id: function_id,
                got: kernel.architecture_version,
            });
        }
    }
    if by_id.keys().any(|id| !exec.kernels.contains_key(id)) {
        let kernel_id = *by_id
            .keys()
            .find(|id| !exec.kernels.contains_key(id))
            .expect("set contains an absent metadata id");
        return Err(Error::EnhancedKernelMissing { kernel_id });
    }
    let mut dynamic = BTreeMap::<i32, (u64, BTreeSet<u32>)>::new();
    for loc in trace.kernels() {
        for (_, block) in trace.validated_threadblocks_in_launch_order(&loc)? {
            for warp in block.warps() {
                for instruction in warp.instructions() {
                    let entry = dynamic
                        .entry(instruction.function_unique_id())
                        .or_insert((0, BTreeSet::new()));
                    entry.0 += 1;
                    entry.1.insert(instruction.pc());
                }
            }
        }
    }
    let mut reports = Vec::new();
    for (&function_id, kernel) in &exec.kernels {
        let static_kernel = by_id.get(&function_id).ok_or(Error::KernelMissing {
            kernel_id: function_id,
        })?;
        if strip_instance_suffix(&static_kernel.kernel_name) != kernel.function_name {
            return Err(Error::KernelMissing {
                kernel_id: function_id,
            });
        }
        let mut metadata_pcs = BTreeSet::new();
        for instruction in &static_kernel.instructions {
            if !metadata_pcs.insert(instruction.pc_num_dec) {
                return Err(Error::DuplicateStaticPc {
                    kernel_id: function_id,
                    pc: instruction.pc_num_dec,
                });
            }
        }
        if metadata_pcs != kernel.static_pcs {
            return Err(Error::StaticPcMismatch {
                kernel_id: function_id,
                metadata: metadata_pcs.len(),
                exec: kernel.static_pcs.len(),
            });
        }
        let (dynamic_count, dynamic_pcs) = dynamic
            .get(&function_id)
            .cloned()
            .unwrap_or((0, BTreeSet::new()));
        if dynamic_count == 0 {
            return Err(Error::DynamicKernelMissing {
                kernel_id: function_id,
            });
        }
        for &pc in &dynamic_pcs {
            if !metadata_pcs.contains(&u64::from(pc)) {
                return Err(Error::DynamicPcMissing {
                    kernel_id: function_id,
                    pc,
                });
            }
        }
        let mut opcodes = BTreeMap::new();
        let mut control_yes = 0;
        let mut register_yes = 0;
        for instruction in &static_kernel.instructions {
            *opcodes.entry(instruction.op_code.clone()).or_default() += 1;
            control_yes += u64::from(instruction.control_fields_available);
            register_yes += u64::from(instruction.register_metadata_available);
        }
        reports.push(KernelReport {
            function_unique_id: function_id,
            kernel_name: static_kernel.kernel_name.clone(),
            static_instruction_count: metadata_pcs.len() as u64,
            dynamic_instruction_count: dynamic_count,
            dynamic_pc_count: dynamic_pcs.len() as u64,
            static_dynamic_pc_join: true,
            control_fields_available: control_yes,
            control_fields_missing: metadata_pcs.len() as u64 - control_yes,
            register_metadata_available: register_yes,
            register_metadata_missing: metadata_pcs.len() as u64 - register_yes,
            opcode_histogram: opcodes,
        });
    }
    if dynamic.keys().any(|id| !exec.kernels.contains_key(id)) {
        let kernel_id = *dynamic
            .keys()
            .find(|id| !exec.kernels.contains_key(id))
            .expect("set contains an absent dynamic id");
        return Err(Error::EnhancedKernelMissing { kernel_id });
    }
    if let Some(path) = sidecar_path {
        let sidecar = OpaqueControlSidecar {
            schema_version: 1,
            contract: "sm120_static_control_opaque_v1",
            semantics: "raw_static_metadata_only; control_and_operand_fields_are_uninterpreted; register_metadata_available_false_means_register_file_metadata_absent_not_operand_list_absent",
            timing_fields_are_uninterpreted: true,
            source_metadata: raw_metadata,
        };
        let bytes = serde_json::to_vec_pretty(&sidecar)
            .map_err(|source| Error::SidecarSerialize { source })?;
        let mut output = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|source| Error::SidecarWrite { path: path.clone(), source })?;
        output
            .write_all(&bytes)
            .map_err(|source| Error::SidecarWrite { path, source })?;
    }
    Ok(Report {
        schema_version: 1,
        status: "static_contract_only",
        note: "static metadata and dynamic PC/control/operand admission only; register metadata availability is register-file metadata only and does not imply operands are absent; no SM120 issue, cache, completion, or timing model",
        metadata_kind: "native_static_sm120_v3".to_owned(),
        static_exporter_version: metadata.static_exporter_version,
        trace_arch: 120,
        card_arch: card_arch.token().to_owned(),
        kernels: reports,
    })
}

fn main() -> std::process::ExitCode {
    match parse_args().and_then(|(trace, card, sidecar)| run(trace, card, sidecar)) {
        Ok(report) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&report).expect("report JSON")
            );
            std::process::ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(tag: &str) -> PathBuf {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        std::env::temp_dir().join(format!("sm120_static_contract_{tag}_{nonce}.json"))
    }

    #[test]
    fn strict_parser_rejects_duplicate_keys() {
        let path = temp_path("duplicate");
        fs::write(&path, br#"{"metadata_kind":"native_static_sm120_v3","metadata_kind":"evil"}"#).unwrap();
        assert!(matches!(read_metadata(&path), Err(Error::MetadataParse { .. })));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn version_three_is_required() {
        let path = temp_path("version");
        fs::write(&path, br#"{"metadata_kind":"native_static_sm120_v3","static_exporter_version":4,"control_latency_claims":false,"kernels":[]}"#).unwrap();
        let (metadata, _) = read_metadata(&path).unwrap();
        assert_ne!(metadata.static_exporter_version, 3);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn raw_value_preserves_unknown_fields_semantically() {
        let path = temp_path("unknown");
        fs::write(&path, br#"{"metadata_kind":"native_static_sm120_v3","static_exporter_version":3,"control_latency_claims":false,"kernels":[],"future":{"opaque":true}}"#).unwrap();
        let (_, raw) = read_metadata(&path).unwrap();
        assert_eq!(raw["future"]["opaque"], true);
        let _ = fs::remove_file(path);
    }
}
