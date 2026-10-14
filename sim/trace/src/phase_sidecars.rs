//! Optional NVBit `%clock64` phase-record sidecars.
//!
//! These records are independent timing observations.  They are never folded
//! into replay cycles, NCU targets, or cache/scheduler constants.  A missing
//! phase sidecar is a normal unavailable state; malformed present data fails
//! closed so a partial record cannot look like a complete interval.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::Serialize;

const RECORD_BYTES: usize = 56;
const MAX_FILE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_FILES: usize = 4096;
const MAX_TOTAL_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug, Clone, thiserror::Error)]
#[error("phase-clock sidecar {path}: {message}")]
pub struct PhaseSidecarError {
    pub path: PathBuf,
    pub message: String,
}

/// One wire record emitted by `phase_clock_record_t` in the NVBit tracer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PhaseClockRecord {
    pub launch_index: u32,
    pub cta: [i32; 3],
    pub sm_id: i32,
    pub warp: i32,
    pub phase_id: u32,
    pub phase_kind: u32,
    pub clock_role: u32,
    pub clock64: u64,
    pub active_mask: u32,
}

/// A unique open/close interval for one launch/CTA/warp/phase scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PhaseClockInterval {
    pub repeat_index: u32,
    pub active_mask: u32,
    pub launch_index: u32,
    pub cta: [i32; 3],
    pub sm_id: i32,
    pub warp: i32,
    pub phase_id: u32,
    pub phase_kind: u32,
    pub open_clock: u64,
    pub close_clock: u64,
    pub cycles: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct PhaseClockEvidence {
    pub available: bool,
    pub files: u64,
    pub records: u64,
    pub launches: BTreeSet<u32>,
    pub intervals: Vec<PhaseClockInterval>,
    pub ambiguous_scopes: u64,
    pub invalid_order_scopes: u64,
}

impl PhaseClockEvidence {
    /// Load all `phase_records_<repeat>_<launch>.bin` leaves in one capture.
    /// No file is required; the empty result is the explicit unavailable state.
    pub fn load(root: &Path) -> Result<Self, PhaseSidecarError> {
        let mut paths = Vec::new();
        let entries = match fs::read_dir(root) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Self::default())
            }
            Err(error) => {
                return Err(PhaseSidecarError {
                    path: root.to_owned(),
                    message: error.to_string(),
                })
            }
        };
        for entry in entries {
            let entry = entry.map_err(|error| PhaseSidecarError {
                path: root.to_owned(),
                message: error.to_string(),
            })?;
            let path = entry.path();
            let name = path.file_name().and_then(|value| value.to_str());
            if name.is_some_and(|value| {
                value.starts_with("phase_records_") && value.ends_with(".bin")
            }) {
                if paths.len() == MAX_FILES {
                    return Err(PhaseSidecarError { path, message: "too many phase files".to_owned() });
                }
                paths.push(path);
            }
        }
        paths.sort();
        let mut evidence = Self {
            available: !paths.is_empty(),
            files: u64::try_from(paths.len()).map_err(|_| PhaseSidecarError {
                path: root.to_owned(),
                message: "phase file count exceeds u64".to_owned(),
            })?,
            ..Self::default()
        };
        let mut scopes: BTreeMap<PhaseScope, PhasePair> = BTreeMap::new();
        let mut total_bytes = 0_u64;
        for path in paths {
            let error = |message: String| PhaseSidecarError { path: path.clone(), message };
            let (repeat_index, launch_index) = file_scope(&path)
                .ok_or_else(|| error("expected phase_records_<repeat>_<launch>.bin".to_owned()))?;
            let file = File::open(&path).map_err(|e| error(e.to_string()))?;
            let mut bytes = Vec::new();
            file.take(MAX_FILE_BYTES + 1).read_to_end(&mut bytes)
                .map_err(|e| error(e.to_string()))?;
            total_bytes = total_bytes.checked_add(bytes.len() as u64)
                .ok_or_else(|| error("phase byte count overflow".to_owned()))?;
            if bytes.len() as u64 > MAX_FILE_BYTES || total_bytes > MAX_TOTAL_BYTES {
                return Err(error("phase data exceeds byte limit".to_owned()));
            }
            if bytes.len() % RECORD_BYTES != 0 {
                return Err(PhaseSidecarError {
                    path,
                    message: format!(
                        "length {} is not a multiple of phase record size {RECORD_BYTES}",
                        bytes.len()
                    ),
                });
            }
            for chunk in bytes.chunks_exact(RECORD_BYTES) {
                let record = decode_record(chunk).ok_or_else(|| PhaseSidecarError {
                    path: path.clone(),
                    message: "phase record has an invalid fixed-width layout".to_owned(),
                })?;
                if record.launch_index != launch_index || record.clock_role > 1
                    || record.phase_kind > 3 || record.cta.iter().any(|v| *v < 0)
                    || record.sm_id < 0 || !(0..32).contains(&record.warp)
                    || record.active_mask == 0 {
                    return Err(error("invalid phase scope, role, kind or lane mask".to_owned()));
                }
                evidence.records = evidence.records.checked_add(1).ok_or_else(|| {
                    PhaseSidecarError {
                        path: path.clone(),
                        message: "phase record count overflows u64".to_owned(),
                    }
                })?;
                evidence.launches.insert(record.launch_index);
                let pair = scopes.entry(record.scope(repeat_index)).or_default();
                let slot = if record.clock_role == 0 { &mut pair.open } else { &mut pair.close };
                if slot.is_some() { pair.duplicate = true; } else { *slot = Some(record); }
            }
        }
        for (scope, pair) in scopes {
            let Some((open, close)) = pair.open.zip(pair.close).filter(|_| !pair.duplicate)
            else {
                evidence.ambiguous_scopes += 1;
                continue;
            };
            if close.clock64 < open.clock64 {
                evidence.invalid_order_scopes = evidence
                    .invalid_order_scopes
                    .checked_add(1)
                    .ok_or_else(|| PhaseSidecarError {
                        path: root.to_owned(),
                        message: "invalid phase scope count overflows u64".to_owned(),
                    })?;
                continue;
            }
            evidence.intervals.push(PhaseClockInterval {
                repeat_index: scope.repeat_index,
                active_mask: scope.active_mask,
                launch_index: scope.launch_index,
                cta: scope.cta,
                sm_id: scope.sm_id,
                warp: scope.warp,
                phase_id: scope.phase_id,
                phase_kind: scope.phase_kind,
                open_clock: open.clock64,
                close_clock: close.clock64,
                cycles: close.clock64 - open.clock64,
            });
        }
        Ok(evidence)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct PhaseScope {
    repeat_index: u32,
    active_mask: u32,
    launch_index: u32,
    cta: [i32; 3],
    sm_id: i32,
    warp: i32,
    phase_id: u32,
    phase_kind: u32,
}

impl PhaseClockRecord {
    fn scope(self, repeat_index: u32) -> PhaseScope {
        PhaseScope {
            repeat_index,
            active_mask: self.active_mask,
            launch_index: self.launch_index,
            cta: self.cta,
            sm_id: self.sm_id,
            warp: self.warp,
            phase_id: self.phase_id,
            phase_kind: self.phase_kind,
        }
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    bytes.get(offset..offset + 4)?.try_into().ok().map(u32::from_le_bytes)
}

fn read_i32(bytes: &[u8], offset: usize) -> Option<i32> {
    read_u32(bytes, offset).map(|value| i32::from_le_bytes(value.to_le_bytes()))
}

fn read_u64(bytes: &[u8], offset: usize) -> Option<u64> {
    bytes.get(offset..offset + 8)?.try_into().ok().map(u64::from_le_bytes)
}

fn decode_record(bytes: &[u8]) -> Option<PhaseClockRecord> {
    (bytes.len() == RECORD_BYTES).then_some(())?;
    Some(PhaseClockRecord {
        launch_index: read_u32(bytes, 0)?,
        cta: [read_i32(bytes, 4)?, read_i32(bytes, 8)?, read_i32(bytes, 12)?],
        sm_id: read_i32(bytes, 16)?,
        warp: read_i32(bytes, 20)?,
        phase_id: read_u32(bytes, 24)?,
        phase_kind: read_u32(bytes, 28)?,
        clock_role: read_u32(bytes, 32)?,
        clock64: read_u64(bytes, 40)?,
        active_mask: read_u32(bytes, 48)?,
    })
}

#[derive(Default)]
struct PhasePair {
    open: Option<PhaseClockRecord>,
    close: Option<PhaseClockRecord>,
    duplicate: bool,
}

fn file_scope(path: &Path) -> Option<(u32, u32)> {
    let name = path.file_name()?.to_str()?;
    let (repeat, launch) = name.strip_prefix("phase_records_")?.strip_suffix(".bin")?.split_once('_')?;
    Some((repeat.parse().ok()?, launch.parse().ok()?))
}
