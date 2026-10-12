//! Loader for `extra_info/enhanced_execution_info.json` — the NVBit
//! tracer's static per-kernel metadata that ships alongside every trace dir.
//!
//! The simulator consumes ONLY the join surface:
//! `unique_function_id` → kernel name / sidecar function name + the static
//! pc table. This is the key that connects a dynamic trace record
//! (`instruction.function_unique_id`, `instruction.pc`) to the Phase-1E
//! control-bits sidecar (`sim_sm::ControlSidecar`, keyed by function name
//! and pc).
//!
//! Join facts, verified on all 17 M1 traces (2026-07-06):
//! - every traced instruction's `function_unique_id` matches a kernel's
//!   `unique_function_id` here;
//! - `kernel_name` minus its trailing `___<digits>` instance suffix equals
//!   the sidecar's function key BYTE-EXACTLY (C++ mangled names included);
//! - the static pc set here equals the sidecar's pc set exactly, and every
//!   dynamic traced pc is a member (pcs are function-relative; `vpc - pc`
//!   is the constant function base address).
//!
//! Deliberately NOT parsed: the per-instruction `control_bits` block and
//! `op_code`. The embedded control bits are untrustworthy — compared against
//! the Phase-1E sidecars (which re-decode the binary's actual encoding under
//! a transcription guard), only 88 of 5,736 static instructions agreed
//! (stall counts mostly 0, yield never set; NVBit records
//! `encoded_instruction: []`, so it never saw the raw words). The sidecar is
//! the ONLY admissible control-bits source. `op_code` is NVBit's own
//! normalization and disagrees with nvdisasm's head (e.g. NVBit "MOV" where
//! SASS reads "IMAD.MOV.U32"), so opcode classification must come from the
//! sidecar's `opcode_text`, never from here.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::json::strict_json_from_slice;
use serde::Deserialize;

/// Path of the exec-info file inside a trace dir.
const EXEC_INFO_RELPATH: &str = "extra_info/enhanced_execution_info.json";
/// Full modern workload captures may carry >64 MiB of static launch metadata.
/// Keep the bound finite but large enough for the retained SM120 corpus.
const MAX_EXEC_INFO_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum ExecInfoError {
    #[error("failed to read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("exec info {path} is {observed} bytes, exceeding implementation limit {limit}")]
    ResourceLimit {
        path: PathBuf,
        observed: u64,
        limit: u64,
    },
    #[error("failed to parse {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("exec info has no kernels")]
    EmptyKernels,
    #[error("kernel {kernel_name:?}: empty static instruction table")]
    EmptyInstructions { kernel_name: String },
    #[error("kernel {kernel_name:?}: unique_function_id {id} is not positive")]
    InvalidFunctionId { kernel_name: String, id: i32 },
    #[error("duplicate unique_function_id {id} (kernels {first:?} and {second:?})")]
    DuplicateFunctionId {
        id: i32,
        first: String,
        second: String,
    },
    #[error("kernel {kernel_name:?}: duplicate static pc {pc}")]
    DuplicatePc { kernel_name: String, pc: u64 },
    #[error(
        "kernel {kernel_name:?}: pc transcription mismatch: \
         pc_string_hex {hex:?} != pc_num_dec {dec}"
    )]
    PcTranscriptionMismatch {
        kernel_name: String,
        hex: String,
        dec: u64,
    },
}

/// Static metadata for one kernel: the (function id → name, pc table) join
/// surface.
#[derive(Debug, Clone)]
pub struct KernelStaticInfo {
    /// Raw NVBit kernel name, including the trailing `___<digits>` instance
    /// suffix (e.g. `layernorm_fwd_bf16_sm80_kernel___0`).
    pub kernel_name: String,
    /// `kernel_name` with the trailing `___<digits>` suffix stripped — the
    /// control-bits sidecar's function key. Equal to `kernel_name` when no
    /// suffix is present.
    pub function_name: String,
    /// SM architecture version as NVBit reports it (86 for sm_86).
    pub architecture_version: u32,
    /// Function-relative pcs of every static instruction.
    pub static_pcs: BTreeSet<u64>,
}

/// Parsed exec info: `unique_function_id` → static kernel metadata.
#[derive(Debug, Clone)]
pub struct ExecInfo {
    pub benchmark_name: String,
    pub kernels: BTreeMap<i32, KernelStaticInfo>,
}

#[derive(Deserialize)]
struct RawExecInfo {
    benchmark_name: String,
    kernels: Vec<RawKernel>,
}

#[derive(Deserialize)]
struct RawKernel {
    kernel_name: String,
    unique_function_id: i32,
    architecture_version: u32,
    instructions: Vec<RawInst>,
}

#[derive(Deserialize)]
struct RawInst {
    pc_string_hex: String,
    pc_num_dec: u64,
}

/// Strip a trailing `___<digits>` NVBit instance suffix, if present.
///
/// Only a suffix where every character after the final `___` is a digit is
/// stripped — mangled C++ names may legitimately contain `___` elsewhere.
fn strip_instance_suffix(kernel_name: &str) -> &str {
    if let Some(pos) = kernel_name.rfind("___") {
        let tail = &kernel_name[pos + 3..];
        if !tail.is_empty() && tail.bytes().all(|b| b.is_ascii_digit()) {
            return &kernel_name[..pos];
        }
    }
    kernel_name
}

impl ExecInfo {
    /// Load and validate `<trace_dir>/extra_info/enhanced_execution_info.json`.
    pub fn load_from_trace_dir(trace_dir: &Path) -> Result<Self, ExecInfoError> {
        let path = trace_dir.join(EXEC_INFO_RELPATH);
        let raw = read_bounded_exec_info(&path, MAX_EXEC_INFO_BYTES)?;
        Self::parse(&path, &raw)
    }

    /// Parse already-snapshotted exec-info bytes. The caller remains responsible
    /// for binding the raw digest before invoking this semantic validator.
    pub fn parse(path: &Path, raw: &[u8]) -> Result<Self, ExecInfoError> {
        let parsed: RawExecInfo =
            strict_json_from_slice(raw).map_err(|source| ExecInfoError::Parse {
                path: path.to_path_buf(),
                source,
            })?;
        Self::from_raw(parsed)
    }

    fn from_raw(raw: RawExecInfo) -> Result<Self, ExecInfoError> {
        if raw.kernels.is_empty() {
            return Err(ExecInfoError::EmptyKernels);
        }
        let mut kernels: BTreeMap<i32, KernelStaticInfo> = BTreeMap::new();
        for k in raw.kernels {
            if k.unique_function_id <= 0 {
                return Err(ExecInfoError::InvalidFunctionId {
                    kernel_name: k.kernel_name,
                    id: k.unique_function_id,
                });
            }
            if k.instructions.is_empty() {
                return Err(ExecInfoError::EmptyInstructions {
                    kernel_name: k.kernel_name,
                });
            }
            let mut static_pcs = BTreeSet::new();
            for inst in &k.instructions {
                let hex_value = u64::from_str_radix(&inst.pc_string_hex, 16);
                if hex_value != Ok(inst.pc_num_dec) {
                    return Err(ExecInfoError::PcTranscriptionMismatch {
                        kernel_name: k.kernel_name,
                        hex: inst.pc_string_hex.clone(),
                        dec: inst.pc_num_dec,
                    });
                }
                if !static_pcs.insert(inst.pc_num_dec) {
                    return Err(ExecInfoError::DuplicatePc {
                        kernel_name: k.kernel_name,
                        pc: inst.pc_num_dec,
                    });
                }
            }
            let function_name = strip_instance_suffix(&k.kernel_name).to_owned();
            let info = KernelStaticInfo {
                function_name,
                architecture_version: k.architecture_version,
                static_pcs,
                kernel_name: k.kernel_name,
            };
            match kernels.entry(k.unique_function_id) {
                std::collections::btree_map::Entry::Occupied(e) => {
                    return Err(ExecInfoError::DuplicateFunctionId {
                        id: k.unique_function_id,
                        first: e.get().kernel_name.clone(),
                        second: info.kernel_name,
                    });
                }
                std::collections::btree_map::Entry::Vacant(v) => {
                    v.insert(info);
                }
            }
        }
        Ok(Self {
            benchmark_name: raw.benchmark_name,
            kernels,
        })
    }
}

fn read_bounded_exec_info(path: &Path, limit: u64) -> Result<Vec<u8>, ExecInfoError> {
    let file = std::fs::File::open(path).map_err(|source| ExecInfoError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut raw = Vec::new();
    file.take(limit.saturating_add(1))
        .read_to_end(&mut raw)
        .map_err(|source| ExecInfoError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let observed = u64::try_from(raw.len()).unwrap_or(u64::MAX);
    if observed > limit {
        return Err(ExecInfoError::ResourceLimit {
            path: path.to_path_buf(),
            observed,
            limit,
        });
    }
    Ok(raw)
}
