//! Native TMA and synchronization observations, owned by one loaded trace.
//! Neither profiler measurements nor hardware completion timestamps live here.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use serde::Deserialize;

// Runtime resource bounds, not corpus-selection or measurement gates.
const MAX_LINE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_INSTRUCTION_KEYS: usize = 4_000_000;

#[derive(Debug, Clone, thiserror::Error)]
#[error("native async sidecar {path}: {message}")]
pub struct AsyncSidecarError {
    pub path: PathBuf,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ObservationKey {
    pub cta: [u32; 3],
    pub warp: i32,
    pub instruction: usize,
    pub pc: u32,
    pub function: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct TmaObservation {
    pub bytes: Option<u64>,
    pub is_load: bool,
    pub is_store: bool,
    pub zero_byte: bool,
    pub oob_or_empty: bool,
}

#[derive(Debug, Default)]
pub struct SyncObservation {
    pub records: u64,
    pub before_records: u64,
    pub after_records: u64,
    pub value_incomplete_records: u64,
    pub mbarrier_address_records: u64,
    pub mbarrier_register_records: u64,
    pub arrival_records: u64,
    pub phase_check_records: u64,
    pub wait_records: u64,
    pub exchange_records: u64,
    // Distinct within ONE instruction. Summing the lengths is an occurrence
    // count, not distinct barrier objects or addresses across a whole launch.
    pub barrier_addresses: BTreeSet<u32>,
}

#[derive(Debug, Default)]
pub struct LaunchObservations {
    pub tma: BTreeMap<ObservationKey, TmaObservation>,
    pub sync: BTreeMap<ObservationKey, SyncObservation>,
}

#[derive(Debug, Default)]
pub struct AsyncSidecars {
    launches: BTreeMap<(i64, i64, i32), LaunchObservations>,
}

// Do not flatten a serde map here: it would materialize the enormous address
// arrays. Unused JSON fields are streamed/skipped without constructing Values.
#[derive(Deserialize)]
struct Row {
    device_id: i64,
    kernel_id: i32,
    stream_id: Option<i64>,
    threadblock: Option<String>,
    cta: [u32; 3],
    warp_id: i32,
    warp_instruction_index: usize,
    function_id: i32,
    pc: u32,
    opcode: String,
    byte_count: Option<u64>,
    zero_byte_disposition: Option<String>,
    operand_kind: Option<String>,
    phase: Option<String>,
    value_complete: Option<bool>,
    barrier_address_present: Option<bool>,
    barrier_address_u32: Option<u32>,
}

impl AsyncSidecars {
    pub fn launch(&self, device: i64, stream: i64, kernel: i32) -> Option<&LaunchObservations> {
        self.launches.get(&(device, stream, kernel))
    }

    pub(crate) fn load(
        root: &Path,
        streams: &BTreeMap<(i64, i32), BTreeSet<i64>>,
    ) -> Result<Self, AsyncSidecarError> {
        let mut result = Self::default();
        let mut keys = 0;
        for (name, tma) in [("tma_trace.jsonl", true), ("sync_trace.jsonl", false)] {
            let path = root.join("extra_info").join(name);
            let error = |message: String| AsyncSidecarError {
                path: path.clone(),
                message,
            };
            let file = match File::open(&path) {
                Ok(file) => file,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => return Err(error(e.to_string())),
            };
            let mut reader = BufReader::new(file);
            let mut line = Vec::new();
            let mut number = 0_u64;
            loop {
                line.clear();
                let read = (&mut reader)
                    .take(MAX_LINE_BYTES + 1)
                    .read_until(b'\n', &mut line)
                    .map_err(|e| error(e.to_string()))?;
                if read == 0 {
                    break;
                }
                number += 1;
                if read as u64 > MAX_LINE_BYTES {
                    return Err(error(format!(
                        "line {number} exceeds {MAX_LINE_BYTES} bytes"
                    )));
                }
                let row: Row = serde_json::from_slice(&line)
                    .map_err(|e| error(format!("line {number}: {e}")))?;
                let launch_streams = streams.get(&(row.device_id, row.kernel_id));
                let stream = row
                    .stream_id
                    .or_else(|| {
                        // Producer identity: d_<device>_s_<stream>_k_<kernel>_<CTA>.
                        let tb = row.threadblock.as_deref()?;
                        let prefix = format!("d_{}_s_", row.device_id);
                        let (stream, rest) = tb.strip_prefix(&prefix)?.split_once("_k_")?;
                        rest.strip_prefix(&format!("{}_", row.kernel_id))?;
                        stream.parse().ok()
                    })
                    .or_else(|| launch_streams.filter(|s| s.len() == 1)?.first().copied())
                    .ok_or_else(|| {
                        error(format!(
                            "line {number} has ambiguous/missing stream identity"
                        ))
                    })?;
                if !launch_streams.is_some_and(|s| s.contains(&stream)) {
                    return Err(error(format!(
                        "line {number} refers to an absent traced launch"
                    )));
                }
                let key = ObservationKey {
                    cta: row.cta,
                    warp: row.warp_id,
                    instruction: row.warp_instruction_index,
                    pc: row.pc,
                    function: row.function_id,
                };
                let launch = result
                    .launches
                    .entry((row.device_id, stream, row.kernel_id))
                    .or_default();
                let is_new = if tma {
                    !launch.tma.contains_key(&key)
                } else {
                    !launch.sync.contains_key(&key)
                };
                if is_new {
                    keys += 1;
                    if keys > MAX_INSTRUCTION_KEYS {
                        return Err(error(format!(
                            "instruction index exceeds {MAX_INSTRUCTION_KEYS} keys"
                        )));
                    }
                }
                if tma {
                    let head = row.opcode.split('.').next().unwrap_or("");
                    if !matches!(head, "UTMALDG" | "UTMASTG") {
                        return Err(error(format!(
                            "line {number} unsupported TMA opcode {}",
                            row.opcode
                        )));
                    }
                    let obs = TmaObservation {
                        bytes: row.byte_count,
                        is_load: head == "UTMALDG",
                        is_store: head == "UTMASTG",
                        zero_byte: row.byte_count == Some(0),
                        oob_or_empty: row.zero_byte_disposition.as_deref()
                            == Some("oob_or_empty_transfer"),
                    };
                    if launch.tma.insert(key, obs).is_some() {
                        return Err(error(format!(
                            "line {number} duplicates TMA instruction identity"
                        )));
                    }
                } else {
                    let obs = launch.sync.entry(key).or_default();
                    let increment = |counter: &mut u64, amount: u64| {
                        *counter = counter.checked_add(amount).ok_or_else(|| {
                            error(format!("line {number} observation count overflow"))
                        })?;
                        Ok::<(), AsyncSidecarError>(())
                    };
                    increment(&mut obs.records, 1)?;
                    increment(
                        &mut obs.before_records,
                        u64::from(row.phase.as_deref() == Some("before")),
                    )?;
                    increment(
                        &mut obs.after_records,
                        u64::from(row.phase.as_deref() == Some("after")),
                    )?;
                    increment(
                        &mut obs.value_incomplete_records,
                        u64::from(row.value_complete != Some(true)),
                    )?;
                    increment(
                        &mut obs.mbarrier_address_records,
                        u64::from(row.operand_kind.as_deref() == Some("mbarrier_address")),
                    )?;
                    increment(
                        &mut obs.mbarrier_register_records,
                        u64::from(row.operand_kind.as_deref() == Some("mbarrier_register")),
                    )?;
                    increment(
                        &mut obs.arrival_records,
                        u64::from(
                            row.opcode.starts_with("SYNCS.ARRIVE")
                                || row.opcode.starts_with("ARRIVES."),
                        ),
                    )?;
                    increment(
                        &mut obs.phase_check_records,
                        u64::from(row.opcode.starts_with("SYNCS.PHASECHK")),
                    )?;
                    increment(
                        &mut obs.wait_records,
                        u64::from(row.opcode.contains("TRYWAIT")),
                    )?;
                    increment(
                        &mut obs.exchange_records,
                        u64::from(row.opcode.starts_with("SYNCS.EXCH")),
                    )?;
                    if row.barrier_address_present == Some(true)
                        && let Some(address) = row.barrier_address_u32
                    {
                        obs.barrier_addresses.insert(address);
                    }
                }
            }
        }
        Ok(result)
    }
}
