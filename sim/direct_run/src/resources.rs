//! Optional CUPTI launch-resource observations.
//!
//! This is deliberately separate from NCU labels and from replay inputs. CUPTI
//! reports the driver's selected shared-memory configuration (`sharedMemoryExecuted`),
//! whereas a CUDA carveout attribute is only a preference. The records are
//! joined by exact kernel name, device/stream, geometry, and checked resources.
//! Cross-run correlation IDs are not NVBit launch IDs. Repeated equal shapes
//! are safe only when their resource observations agree; otherwise reject the
//! ambiguity rather than rely on CUPTI activity-buffer order.

use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::io::Read;
use std::path::Path;

use super::RunError;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LaunchResourceObservation {
    #[serde(rename = "record")]
    _record: String,
    pub kernel_name: String,
    pub device: u32,
    pub stream: u64,
    pub correlation_id: u64,
    /// False for interchangeable repeated observations: do not pretend that
    /// an arbitrary external correlation ID identifies a particular launch.
    #[serde(skip)]
    pub unique_correlation: bool,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub registers_per_thread: u32,
    pub static_shared_memory_bytes: u32,
    pub dynamic_shared_memory_bytes: u32,
    pub shared_memory_config_bytes: u32,
    pub carveout_requested: u8,
    pub carveout_pct: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct LaunchKey {
    device: u32,
    stream: u64,
    kernel_name: String,
    grid: [u32; 3],
    block: [u32; 3],
}

fn kernel_name_without_instance_suffix(name: &str) -> &str {
    name.rfind("___")
        .filter(|&index| {
            !name[index + 3..].is_empty()
                && name[index + 3..].bytes().all(|byte| byte.is_ascii_digit())
        })
        .map_or(name, |index| &name[..index])
}

impl LaunchResourceObservation {
    fn key(&self) -> LaunchKey {
        LaunchKey {
            device: self.device,
            stream: self.stream,
            // CUPTI reports the native symbol, not an NVBit instance name.
            // A native symbol may itself legitimately end in `___<digits>`.
            kernel_name: self.kernel_name.clone(),
            grid: self.grid,
            block: self.block,
        }
    }

    fn same_resources(&self, other: &Self) -> bool {
        self.registers_per_thread == other.registers_per_thread
            && self.static_shared_memory_bytes == other.static_shared_memory_bytes
            && self.dynamic_shared_memory_bytes == other.dynamic_shared_memory_bytes
            && self.shared_memory_config_bytes == other.shared_memory_config_bytes
            && self.carveout_requested == other.carveout_requested
            && self.carveout_pct == other.carveout_pct
    }

    pub(crate) fn validate_trace_resources(
        &self,
        trace: Option<&sim_sm::metrics::LaunchResources>,
    ) -> Result<(), RunError> {
        let trace = trace.ok_or_else(|| {
            RunError::new(
                "launch resources",
                "trace lacks launch-resource metadata for checking the CUPTI join",
            )
        })?;
        let mismatch = if self.registers_per_thread != trace.registers_per_thread {
            Some("registers_per_thread")
        } else if u64::from(self.static_shared_memory_bytes)
            + u64::from(self.dynamic_shared_memory_bytes)
            != u64::from(trace.shared_memory_bytes_per_block)
        {
            Some("shared_memory_bytes_per_block")
        } else if trace
            .static_shared_memory_bytes
            .is_some_and(|value| value != self.static_shared_memory_bytes)
        {
            Some("static_shared_memory_bytes")
        } else if trace
            .dynamic_shared_memory_bytes
            .is_some_and(|value| value != self.dynamic_shared_memory_bytes)
        {
            Some("dynamic_shared_memory_bytes")
        } else if trace
            .shared_memory_carveout_preference_pct
            .filter(|value| *value >= 0)
            .zip(self.carveout_pct)
            .is_some_and(|(captured, observed)| captured as u32 != observed)
        {
            Some("shared_memory_carveout_preference_pct")
        } else {
            None
        };
        if let Some(field) = mismatch {
            return Err(RunError::new(
                "launch resources",
                format!("CUPTI {field} disagrees with captured trace resources"),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
pub(crate) struct ResourceObservations {
    by_key: BTreeMap<LaunchKey, VecDeque<LaunchResourceObservation>>,
}

impl ResourceObservations {
    pub(crate) fn load(path: &Path) -> Result<Self, RunError> {
        const MAX_BYTES: u64 = 64 * 1024 * 1024;
        let mut text = String::new();
        std::fs::File::open(path)
            .and_then(|file| file.take(MAX_BYTES + 1).read_to_string(&mut text))
            .map_err(|error| {
                RunError::new(
                    "launch resources",
                    format!("cannot read {}: {error}", path.display()),
                )
            })?;
        if text.len() as u64 > MAX_BYTES {
            return Err(RunError::new(
                "launch resources",
                "CUPTI resource file exceeds 64 MiB",
            ));
        }
        let mut observations = Self::default();
        let mut summary_seen = false;
        let mut record_count = 0_u64;
        let mut correlations = BTreeSet::new();
        for (line_number, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            if summary_seen {
                return Err(RunError::new(
                    "launch resources",
                    "records follow the CUPTI summary",
                ));
            }
            let value: serde_json::Value = sim_trace::strict_json_from_slice(line.as_bytes())
                .map_err(|error| {
                    RunError::new(
                        "launch resources",
                        format!(
                            "{} line {} is invalid JSON: {error}",
                            path.display(),
                            line_number + 1
                        ),
                    )
                })?;
            match value.get("record").and_then(serde_json::Value::as_str) {
                Some("summary") => {
                    let launches = value
                        .get("launches")
                        .and_then(serde_json::Value::as_u64)
                        .ok_or_else(|| {
                            RunError::new(
                                "launch resources",
                                "CUPTI summary lacks an integer launches count",
                            )
                        })?;
                    let dropped = value
                        .get("dropped")
                        .and_then(serde_json::Value::as_u64)
                        .ok_or_else(|| {
                            RunError::new(
                                "launch resources",
                                "CUPTI summary lacks an integer dropped count",
                            )
                        })?;
                    if dropped != 0 {
                        return Err(RunError::new(
                            "launch resources",
                            format!("{dropped} CUPTI records were dropped"),
                        ));
                    }
                    if launches != record_count {
                        return Err(RunError::new(
                            "launch resources",
                            "CUPTI summary launch count differs from resource records",
                        ));
                    }
                    summary_seen = true;
                    continue;
                }
                Some("launch_resources") => {}
                Some("tool") => continue,
                _ => {
                    return Err(RunError::new(
                        "launch resources",
                        "unknown CUPTI resource record type",
                    ));
                }
            }
            let mut observation: LaunchResourceObservation = serde_json::from_value(value)
                .map_err(|error| {
                    RunError::new(
                        "launch resources",
                        format!(
                            "{} line {} has invalid launch-resource schema: {error}",
                            path.display(),
                            line_number + 1
                        ),
                    )
                })?;
            if observation.grid.contains(&0) || observation.block.contains(&0) {
                return Err(RunError::new(
                    "launch resources",
                    format!(
                        "{} line {} contains invalid zero launch geometry",
                        path.display(),
                        line_number + 1
                    ),
                ));
            }
            if observation.carveout_requested > 1
                || observation.carveout_pct.is_some_and(|pct| pct > 100)
                || (observation.carveout_requested == 1 && observation.carveout_pct.is_none())
                || (observation.carveout_requested == 0
                    && observation.carveout_pct.is_some_and(|pct| pct != 0))
            {
                return Err(RunError::new(
                    "launch resources",
                    format!(
                        "{} line {} contains invalid carveout metadata",
                        path.display(),
                        line_number + 1
                    ),
                ));
            }
            if observation.carveout_requested == 0 {
                observation.carveout_pct = None;
            }
            if !correlations.insert((
                observation.device,
                observation.stream,
                observation.correlation_id,
            )) {
                return Err(RunError::new(
                    "launch resources",
                    "duplicate CUPTI correlation identity",
                ));
            }
            record_count = record_count.checked_add(1).ok_or_else(|| {
                RunError::new("launch resources", "CUPTI launch-resource count overflow")
            })?;
            let records = observations.by_key.entry(observation.key()).or_default();
            if let Some(first) = records.front() {
                if !first.same_resources(&observation) {
                    return Err(RunError::new(
                        "launch resources",
                        "ambiguous repeated launch shape has different CUPTI resources; cross-run correlation is unavailable",
                    ));
                }
                // Only the first record can still carry a unique identity;
                // every later insertion is already marked non-unique. Avoid
                // rescanning an arbitrarily long repeated-launch sequence.
                records
                    .front_mut()
                    .expect("checked nonempty")
                    .unique_correlation = false;
                observation.unique_correlation = false;
            } else {
                observation.unique_correlation = true;
            }
            records.push_back(observation);
        }
        if observations.by_key.is_empty() {
            return Err(RunError::new(
                "launch resources",
                format!("{} contains no launch_resources records", path.display()),
            ));
        }
        if !summary_seen {
            return Err(RunError::new(
                "launch resources",
                "missing CUPTI summary: capture may be truncated",
            ));
        }
        Ok(observations)
    }

    pub(crate) fn take(
        &mut self,
        device: i64,
        stream: i64,
        kernel_name: &str,
        grid: (u32, u32, u32),
        block: Option<(u32, u32, u32)>,
    ) -> Result<LaunchResourceObservation, RunError> {
        let device = u32::try_from(device)
            .map_err(|_| RunError::new("launch resources", "device id is negative"))?;
        let stream = u64::try_from(stream)
            .map_err(|_| RunError::new("launch resources", "stream id is negative"))?;
        let block = block
            .ok_or_else(|| RunError::new("launch resources", "launch has no block geometry"))?;
        let exact_key = LaunchKey {
            device,
            stream,
            kernel_name: kernel_name.to_owned(),
            grid: [grid.0, grid.1, grid.2],
            block: [block.0, block.1, block.2],
        };
        // Prefer an exact native symbol.  A legitimate CUDA symbol may end
        // in `___<digits>`; only fall back to removing the NVBit instance
        // suffix when the exact key is absent (e.g. `kernel___0`).
        let key = if self.by_key.contains_key(&exact_key) {
            exact_key
        } else {
            LaunchKey {
                device,
                stream,
                kernel_name: kernel_name_without_instance_suffix(kernel_name).to_owned(),
                grid: [grid.0, grid.1, grid.2],
                block: [block.0, block.1, block.2],
            }
        };
        let records = self.by_key.get_mut(&key).ok_or_else(|| {
            RunError::new(
                "launch resources",
                format!("no CUPTI resource record matches kernel {kernel_name:?} device={device} stream={stream}"),
            )
        })?;
        if records.is_empty() {
            return Err(RunError::new(
                "launch resources",
                format!("CUPTI resource records for kernel {kernel_name:?} are exhausted"),
            ));
        }
        // Repeated observations have identical resources, checked at load.
        // Their order cannot affect the values or imply correlation identity.
        Ok(records.pop_front().expect("checked nonempty"))
    }

    pub(crate) fn finish(&self) -> Result<(), RunError> {
        let remaining: usize = self.by_key.values().map(VecDeque::len).sum();
        if remaining != 0 {
            return Err(RunError::new(
                "launch resources",
                format!("{remaining} CUPTI launch-resource records were not joined"),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn joins_native_symbol_without_stripping_its_numeric_suffix() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "{{\"record\":\"launch_resources\",\"kernel_name\":\"k___7\",\"device\":0,\"stream\":7,\"correlation_id\":3,\"grid\":[16,32,1],\"block\":[32,1,1],\"registers_per_thread\":36,\"static_shared_memory_bytes\":768,\"dynamic_shared_memory_bytes\":0,\"shared_memory_config_bytes\":32768,\"carveout_requested\":0,\"carveout_pct\":null}}"
        )
        .unwrap();
        writeln!(
            file,
            "{{\"record\":\"summary\",\"launches\":1,\"dropped\":0}}"
        )
        .unwrap();
        let mut observations = ResourceObservations::load(file.path()).unwrap();
        let record = observations
            .take(0, 7, "k___7___0", (16, 32, 1), Some((32, 1, 1)))
            .unwrap();
        assert_eq!(record.shared_memory_config_bytes, 32_768);
        assert_eq!(record.carveout_pct, None);
        let mut trace = sim_sm::metrics::LaunchResources {
            block_threads: 32,
            grid_blocks: 512,
            registers_per_thread: 36,
            shared_memory_bytes_per_block: 768,
            static_shared_memory_bytes: Some(768),
            dynamic_shared_memory_bytes: Some(0),
            shared_memory_carveout_preference_pct: None,
            driver_occupancy_limit_blocks: Some(16),
        };
        record.validate_trace_resources(Some(&trace)).unwrap();
        assert!(record.validate_trace_resources(None).is_err());
        trace.registers_per_thread = 37;
        assert!(
            record
                .validate_trace_resources(Some(&trace))
                .unwrap_err()
                .to_string()
                .contains("registers_per_thread")
        );
        trace.registers_per_thread = 36;
        trace.static_shared_memory_bytes = Some(512);
        trace.dynamic_shared_memory_bytes = Some(256);
        assert!(
            record
                .validate_trace_resources(Some(&trace))
                .unwrap_err()
                .to_string()
                .contains("static_shared_memory_bytes")
        );
        observations.finish().unwrap();

        // The same spelling can be a legitimate native symbol, not an NVBit
        // instance suffix. Exact-key preference must keep that record joinable.
        let mut exact = ResourceObservations::load(file.path()).unwrap();
        exact
            .take(0, 7, "k___7", (16, 32, 1), Some((32, 1, 1)))
            .unwrap();
        exact.finish().unwrap();
    }

    #[test]
    fn rejects_dropped_cupti_records() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "{{\"record\":\"launch_resources\",\"kernel_name\":\"k\",\"device\":0,\"stream\":7,\"correlation_id\":3,\"grid\":[1,1,1],\"block\":[32,1,1],\"registers_per_thread\":1,\"static_shared_memory_bytes\":0,\"dynamic_shared_memory_bytes\":0,\"shared_memory_config_bytes\":16384,\"carveout_requested\":0,\"carveout_pct\":null}}").unwrap();
        writeln!(
            file,
            "{{\"record\":\"summary\",\"launches\":1,\"dropped\":1}}"
        )
        .unwrap();
        let error = ResourceObservations::load(file.path()).unwrap_err();
        assert!(error.to_string().contains("dropped"));
    }

    #[test]
    fn rejects_inconsistent_carveout_presence_marker() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "{{\"record\":\"launch_resources\",\"kernel_name\":\"k\",\"device\":0,\"stream\":7,\"correlation_id\":3,\"grid\":[1,1,1],\"block\":[32,1,1],\"registers_per_thread\":1,\"static_shared_memory_bytes\":0,\"dynamic_shared_memory_bytes\":0,\"shared_memory_config_bytes\":16384,\"carveout_requested\":0,\"carveout_pct\":50}}").unwrap();
        writeln!(
            file,
            "{{\"record\":\"summary\",\"launches\":1,\"dropped\":0}}"
        )
        .unwrap();
        let error = ResourceObservations::load(file.path()).unwrap_err();
        assert!(error.to_string().contains("carveout"));
    }

    #[test]
    fn repeated_shape_does_not_claim_cupti_correlation_identity() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        for correlation in [3_u64, 4_u64] {
            writeln!(
                file,
                "{{\"record\":\"launch_resources\",\"kernel_name\":\"k\",\"device\":0,\"stream\":7,\"correlation_id\":{correlation},\"grid\":[1,1,1],\"block\":[32,1,1],\"registers_per_thread\":1,\"static_shared_memory_bytes\":0,\"dynamic_shared_memory_bytes\":0,\"shared_memory_config_bytes\":16384,\"carveout_requested\":0,\"carveout_pct\":0}}"
            )
            .unwrap();
        }
        writeln!(
            file,
            "{{\"record\":\"summary\",\"launches\":2,\"dropped\":0}}"
        )
        .unwrap();
        let mut observations = ResourceObservations::load(file.path()).unwrap();
        assert!(
            !observations
                .take(0, 7, "k", (1, 1, 1), Some((32, 1, 1)))
                .unwrap()
                .unique_correlation
        );
        assert!(
            !observations
                .take(0, 7, "k", (1, 1, 1), Some((32, 1, 1)))
                .unwrap()
                .unique_correlation
        );
        observations.finish().unwrap();
    }
}
