//! Allocation and manifest preflight validation; report-only.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct DynamicMemoryJoinLocation {
    pub key: DynamicMemoryJoinIdentity,
    pub block_id: [u32; 3],
    pub warp: u32,
    pub instruction_index: u32,
    pub launch_index: i64,
    pub warp_slot: Option<i32>,
    /// Trace-declared direction for this dynamic GLOBAL occurrence.  A
    /// payload admission may bind only to `Store`; `None` is retained for
    /// legacy traces and must fail closed at the value-bearing ingress.
    pub access_kind: Option<pb::MemoryOpAccessKind>,
    /// Trace-observed SM placement for the dynamic instruction. This is a
    /// placement identity, not a physical pipe/SMSP/slot observation.
    pub sm_id: Option<i32>,
}

pub const ADDRESS_CARRYING_TRACE_CONTRACT_ID: &str = "sm86_addr_carry_v1";

pub const SHARED_MEMORY_SPLIT_TRACE_CONTRACT_ID: &str = "sm86_addr_carry_shmem_split_v2";

pub const LDGSTS_DESTINATION_TRACE_CONTRACT_ID: &str = "sm86_addr_carry_ldgsts_dst_v3";

pub const SM120_LDGSTS_DESTINATION_TRACE_CONTRACT_ID: &str = "sm120_addr_carry_ldgsts_dst_v3";

pub(crate) const ADDRESS_CARRYING_PAYLOAD_CLASSES: [pb::MemoryOpClass; 5] = [
    pb::MemoryOpClass::Global,
    pb::MemoryOpClass::Shared,
    pb::MemoryOpClass::Local,
    pb::MemoryOpClass::Atomic,
    pb::MemoryOpClass::AsyncCopy,
];

#[cfg(feature = "campaign")]
pub(crate) struct VerifiedSamplingPayload {
    pub(crate) by_path: BTreeMap<PathBuf, Vec<u8>>,
    pub(crate) identity: ThreadblockPayloadIdentity,
}

pub(crate) fn map_threadblock_tree_read_error(
    root: &Path,
    source: AnchoredReadError,
    _preserve_manifest_artifact_role: bool,
) -> TraceError {
    match source {
        AnchoredReadError::Io { path, source }
            if path == root.join("threadblocks")
                && source.kind() == std::io::ErrorKind::NotFound =>
        {
            TraceError::TraceThreadblocksEmpty { dir: path }
        }
        #[cfg(feature = "campaign")]
        other if _preserve_manifest_artifact_role => {
            map_manifest_snapshot_read_error("threadblock_payload", other)
        }
        other => TraceError::AnchoredSnapshot(other),
    }
}

pub(crate) fn map_snapshot_read_error(
    artifact: &'static str,
    source: AnchoredReadError,
) -> TraceError {
    map_snapshot_read_error_with(artifact, source, TraceError::AnchoredSnapshot)
}

#[cfg(feature = "campaign")]
pub(crate) fn map_manifest_snapshot_read_error(
    artifact: &'static str,
    source: AnchoredReadError,
) -> TraceError {
    map_snapshot_read_error_with(artifact, source, |source| {
        TraceError::SnapshotArtifactRead { artifact, source }
    })
}

#[cfg(feature = "campaign")]
pub(crate) fn capture_optional_snapshot_artifact(
    reader: &AnchoredDir,
    relative: &Path,
    expected: Option<Option<u64>>,
    artifact: &'static str,
    maximum_size: u64,
) -> Result<Option<AnchoredFile>, TraceError> {
    let map_read_error = |source| {
        if expected.is_some() {
            map_manifest_snapshot_read_error(artifact, source)
        } else {
            map_snapshot_read_error(artifact, source)
        }
    };
    match expected {
        None => reader
            .read_optional_bounded_regular(relative, maximum_size)
            .map_err(map_read_error),
        Some(Some(size)) => {
            if size > maximum_size {
                return Err(TraceError::TraceArtifactResourceLimitExceeded {
                    path: reader.root().join(relative),
                    artifact,
                    resource: "bytes",
                    observed: size,
                    limit: maximum_size,
                });
            }
            reader
                .read_optional_exact_regular(relative, size)
                .map_err(map_read_error)?
                .ok_or(TraceError::SnapshotArtifactPresenceMismatch {
                    artifact,
                    expected: true,
                    actual: false,
                })
                .map(Some)
        }
        Some(None) => {
            if reader
                .probe_optional_regular(relative)
                .map_err(map_read_error)?
                .is_some()
            {
                return Err(TraceError::SnapshotArtifactPresenceMismatch {
                    artifact,
                    expected: false,
                    actual: true,
                });
            }
            Ok(None)
        }
    }
}

pub(crate) fn sidecar_byte_limit(
    path: &Path,
    artifact: &'static str,
    raw: &[u8],
    maximum_bytes: u64,
) -> Result<(), TraceError> {
    let observed = u64::try_from(raw.len()).unwrap_or(u64::MAX);
    if observed > maximum_bytes {
        return Err(TraceError::TraceArtifactResourceLimitExceeded {
            path: path.to_path_buf(),
            artifact,
            resource: "bytes",
            observed,
            limit: maximum_bytes,
        });
    }
    Ok(())
}

pub(crate) fn map_json_preflight_limit(
    path: &Path,
    artifact: &'static str,
    error: JsonResourceLimit,
) -> TraceError {
    TraceError::TraceArtifactResourceLimitExceeded {
        path: path.to_path_buf(),
        artifact,
        resource: error.resource,
        observed: error.observed,
        limit: error.limit,
    }
}

pub(crate) fn parse_ascii_signed_decimal(text: &str) -> Option<i128> {
    let digits = text.strip_prefix('-').unwrap_or(text);
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    text.parse().ok()
}

pub(crate) fn parse_trace_stats(path: &Path, raw: &[u8]) -> Result<TraceStats, TraceError> {
    if raw.is_empty() {
        return Err(TraceError::StatsEmpty {
            path: path.to_path_buf(),
        });
    }

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(raw);
    let headers = reader
        .headers()
        .map_err(|error| TraceError::StatsSchemaInvalid {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?
        .iter()
        .map(str::trim)
        .collect::<Vec<_>>();
    let expected_index = headers
        .iter()
        .position(|header| *header == "total_insts")
        .ok_or_else(|| TraceError::StatsSchemaInvalid {
            path: path.to_path_buf(),
            message: "stats.csv lacks total_insts".to_owned(),
        })?;
    let admitted_index = headers
        .iter()
        .position(|header| *header == "total_reported_insts")
        .ok_or_else(|| TraceError::StatsSchemaInvalid {
            path: path.to_path_buf(),
            message: "stats.csv lacks total_reported_insts".to_owned(),
        })?;

    let mut rows = 0_u64;
    let mut expected_total = 0_u64;
    let mut admitted_total = 0_u64;
    for (record_index, record) in reader.records().enumerate() {
        let record = record.map_err(|error| TraceError::StatsSchemaInvalid {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
        if record.iter().all(|cell| cell.trim().is_empty()) {
            continue;
        }
        let row = u64::try_from(record_index)
            .ok()
            .and_then(|index| index.checked_add(2))
            .ok_or(TraceError::TraceCountOverflow {
                counter: "stats row number",
            })?;
        let expected_text =
            record
                .get(expected_index)
                .ok_or_else(|| TraceError::StatsSchemaInvalid {
                    path: path.to_path_buf(),
                    message: format!("row {row} is truncated"),
                })?;
        let admitted_text =
            record
                .get(admitted_index)
                .ok_or_else(|| TraceError::StatsSchemaInvalid {
                    path: path.to_path_buf(),
                    message: format!("row {row} is truncated"),
                })?;
        let expected_signed =
            parse_ascii_signed_decimal(expected_text.trim()).ok_or_else(|| {
                TraceError::StatsSchemaInvalid {
                    path: path.to_path_buf(),
                    message: format!("row {row} has a non-integer or out-of-domain total_insts"),
                }
            })?;
        let admitted_signed =
            parse_ascii_signed_decimal(admitted_text.trim()).ok_or_else(|| {
                TraceError::StatsSchemaInvalid {
                    path: path.to_path_buf(),
                    message: format!(
                        "row {row} has a non-integer or out-of-domain total_reported_insts"
                    ),
                }
            })?;
        if expected_signed < 0 || admitted_signed < 0 || admitted_signed > expected_signed {
            return Err(TraceError::StatsCountersInvalid {
                path: path.to_path_buf(),
                row,
                expected: expected_signed,
                admitted: admitted_signed,
            });
        }
        let expected =
            u64::try_from(expected_signed).map_err(|_| TraceError::StatsSchemaInvalid {
                path: path.to_path_buf(),
                message: format!("row {row} total_insts exceeds u64"),
            })?;
        let admitted =
            u64::try_from(admitted_signed).map_err(|_| TraceError::StatsSchemaInvalid {
                path: path.to_path_buf(),
                message: format!("row {row} total_reported_insts exceeds u64"),
            })?;
        expected_total =
            expected_total
                .checked_add(expected)
                .ok_or(TraceError::TraceCountOverflow {
                    counter: "stats expected total",
                })?;
        admitted_total =
            admitted_total
                .checked_add(admitted)
                .ok_or(TraceError::TraceCountOverflow {
                    counter: "stats admitted total",
                })?;
        rows = rows.checked_add(1).ok_or(TraceError::TraceCountOverflow {
            counter: "stats row count",
        })?;
    }
    if rows == 0 || expected_total == 0 || admitted_total == 0 {
        return Err(TraceError::TracePayloadEmpty {
            path: path.to_path_buf(),
            rows,
            expected: expected_total,
            admitted: admitted_total,
        });
    }
    Ok(TraceStats {
        row_count: rows,
        expected_record_count: expected_total,
        admitted_record_count: admitted_total,
    })
}

pub(crate) fn compact_memory_route(
    instruction: &pb::Instruction,
    class: pb::MemoryOpClass,
    payload_contract: MemoryPayloadContract,
    and_mask: u32,
) -> Result<ValidatedMemoryRoute, ValidatedMemoryRouteError> {
    let space = |address: &pb::Address| {
        address
            .address_space
            .map(|raw| match pb::MemorySpace::try_from(raw) {
                Ok(space) => space,
                Err(_) => unreachable!("validated address carried an unknown space"),
            })
    };
    let zero_effective_mask = and_mask == 0;
    match class {
        pb::MemoryOpClass::Unspecified => Ok(ValidatedMemoryRoute::Unsupported),
        pb::MemoryOpClass::Shared | pb::MemoryOpClass::Local => {
            if instruction.addresses.is_empty() && !zero_effective_mask {
                return Err(ValidatedMemoryRouteError::SharedLocalOperandCount { got: 0 });
            }
            let expected = if class == pb::MemoryOpClass::Shared {
                pb::MemorySpace::Shared
            } else {
                pb::MemorySpace::Local
            };
            for (operand, address) in instruction.addresses.iter().enumerate() {
                let Some(space) = space(address) else {
                    continue;
                };
                if space != expected {
                    return Err(ValidatedMemoryRouteError::SharedLocalSpaceInvalid {
                        operand,
                        space,
                    });
                }
            }
            Ok(if class == pb::MemoryOpClass::Shared {
                ValidatedMemoryRoute::Shared
            } else {
                ValidatedMemoryRoute::Local
            })
        }
        pb::MemoryOpClass::Global => {
            if instruction.addresses.is_empty() && !zero_effective_mask {
                return Err(ValidatedMemoryRouteError::GlobalOperandCount { got: 0 });
            }
            for (operand, address) in instruction.addresses.iter().enumerate() {
                let Some(space) = space(address) else {
                    continue;
                };
                if !matches!(space, pb::MemorySpace::Global | pb::MemorySpace::Generic) {
                    return Err(ValidatedMemoryRouteError::GlobalSpaceInvalid { operand, space });
                }
            }
            Ok(ValidatedMemoryRoute::Global)
        }
        pb::MemoryOpClass::Atomic => {
            if payload_contract == MemoryPayloadContract::LegacyExclusion {
                let mut resolved = None;
                for (operand, address) in instruction.addresses.iter().enumerate() {
                    let Some(space) = space(address) else {
                        continue;
                    };
                    let candidate = match space {
                        pb::MemorySpace::Global | pb::MemorySpace::Generic => {
                            ValidatedMemoryRoute::Global
                        }
                        pb::MemorySpace::Shared => ValidatedMemoryRoute::Shared,
                        pb::MemorySpace::Local => ValidatedMemoryRoute::Local,
                        pb::MemorySpace::Unspecified => {
                            return Err(ValidatedMemoryRouteError::AtomicSpaceUnspecified {
                                operand,
                            });
                        }
                    };
                    match resolved {
                        Some(previous) if previous != candidate => {
                            return Err(ValidatedMemoryRouteError::AtomicSpacesConflict);
                        }
                        None => resolved = Some(candidate),
                        Some(_) => {}
                    }
                }
                return Ok(resolved.unwrap_or(ValidatedMemoryRoute::Global));
            }
            if instruction.addresses.is_empty() && zero_effective_mask {
                return Ok(ValidatedMemoryRoute::Global);
            }
            if instruction.addresses.len() != 1 {
                return Err(ValidatedMemoryRouteError::AtomicOperandCount {
                    got: instruction.addresses.len(),
                });
            }
            let Some(space) = space(&instruction.addresses[0]) else {
                return Err(ValidatedMemoryRouteError::AtomicSpaceMissing { operand: 0 });
            };
            match space {
                pb::MemorySpace::Global | pb::MemorySpace::Generic => {
                    Ok(ValidatedMemoryRoute::Global)
                }
                pb::MemorySpace::Shared => Ok(ValidatedMemoryRoute::Shared),
                pb::MemorySpace::Local => Ok(ValidatedMemoryRoute::Local),
                pb::MemorySpace::Unspecified => {
                    Err(ValidatedMemoryRouteError::AtomicSpaceUnspecified { operand: 0 })
                }
            }
        }
        pb::MemoryOpClass::AsyncCopy => {
            if payload_contract == MemoryPayloadContract::LegacyExclusion {
                for address in &instruction.addresses {
                    let Some(space) = space(address) else {
                        continue;
                    };
                    if !matches!(space, pb::MemorySpace::Global | pb::MemorySpace::Generic) {
                        return Err(ValidatedMemoryRouteError::AsyncCopySourceSpaceInvalid {
                            space,
                        });
                    }
                }
                return Ok(ValidatedMemoryRoute::Global);
            }
            if payload_contract.carries_ldgsts_destination() {
                if instruction.addresses.len() != 2 {
                    return Err(ValidatedMemoryRouteError::AsyncCopyOperandCount {
                        got: instruction.addresses.len(),
                        expected: 2,
                    });
                }
                let Some(source_space) = space(&instruction.addresses[0]) else {
                    return Err(ValidatedMemoryRouteError::AsyncCopySourceSpaceMissing);
                };
                if !matches!(
                    source_space,
                    pb::MemorySpace::Global | pb::MemorySpace::Generic
                ) {
                    return Err(ValidatedMemoryRouteError::AsyncCopySourceSpaceInvalid {
                        space: source_space,
                    });
                }
                let Some(destination_space) = space(&instruction.addresses[1]) else {
                    return Err(ValidatedMemoryRouteError::AsyncCopyDestinationSpaceMissing);
                };
                if destination_space != pb::MemorySpace::Shared {
                    return Err(
                        ValidatedMemoryRouteError::AsyncCopyDestinationSpaceInvalid {
                            space: destination_space,
                        },
                    );
                }
                return Ok(ValidatedMemoryRoute::Global);
            }
            if instruction.addresses.len() != 1 {
                return Err(ValidatedMemoryRouteError::AsyncCopySourceCount {
                    got: instruction.addresses.len(),
                });
            }
            let Some(space) = space(&instruction.addresses[0]) else {
                return Err(ValidatedMemoryRouteError::AsyncCopySourceSpaceMissing);
            };
            if !matches!(space, pb::MemorySpace::Global | pb::MemorySpace::Generic) {
                return Err(ValidatedMemoryRouteError::AsyncCopySourceSpaceInvalid { space });
            }
            Ok(ValidatedMemoryRoute::Global)
        }
        pb::MemoryOpClass::Texture
        | pb::MemoryOpClass::Surface
        | pb::MemoryOpClass::Tma
        | pb::MemoryOpClass::ClusterDsmem
        | pb::MemoryOpClass::Tmem => Ok(ValidatedMemoryRoute::Unsupported),
    }
}

pub(crate) fn validate_trace_semantics(
    root: &Path,
    trace: &pb::Trace,
    payload_paths: impl IntoIterator<Item = PathBuf>,
    memory_payload_contract: MemoryPayloadContract,
    mut validate_payload: impl FnMut(&Path, &pb::Kernel) -> Result<ValidatedPayload, TraceError>,
) -> Result<
    (
        TraceValidationSummary,
        BTreeMap<PathBuf, ValidatedThreadblock>,
    ),
    TraceError,
> {
    let payload_paths = payload_paths.into_iter().collect::<BTreeSet<_>>();
    let mut launch_ids = BTreeSet::new();
    let mut consumed_payloads = BTreeSet::new();
    let mut validated_payload = BTreeMap::new();
    let mut payloads_by_parent: BTreeMap<PathBuf, Vec<PathBuf>> = BTreeMap::new();
    for path in &payload_paths {
        let parent =
            path.parent()
                .ok_or_else(|| TraceError::ThreadblockPayloadSnapshotInvalid {
                    root: root.to_path_buf(),
                    message: format!("payload path has no launch parent: {}", path.display()),
                })?;
        payloads_by_parent
            .entry(parent.to_path_buf())
            .or_default()
            .push(path.clone());
    }
    let mut launch_count = 0_u64;
    let mut threadblock_count = 0_u64;
    let mut instruction_count = 0_u64;

    for (&device_key, device) in &trace.gpu_device {
        if u32::try_from(device_key).ok() != Some(device.id) {
            return Err(TraceError::DeviceMapKeyMismatch {
                map_key: device_key,
                embedded_id: device.id,
            });
        }
        for (&stream_key, stream) in &device.streams {
            if stream_key != stream.id {
                return Err(TraceError::StreamMapKeyMismatch {
                    map_key: stream_key,
                    embedded_id: stream.id,
                });
            }
            for kernel in &stream.kernels {
                if kernel.id <= 0 {
                    return Err(TraceError::LaunchIdentityInvalid {
                        device_id: device_key,
                        stream_id: stream_key,
                        kernel_id: kernel.id,
                    });
                }
                if !launch_ids.insert((device_key, stream_key, kernel.id)) {
                    return Err(TraceError::LaunchIdentityDuplicate {
                        device_id: device_key,
                        stream_id: stream_key,
                        kernel_id: kernel.id,
                    });
                }
                launch_count =
                    launch_count
                        .checked_add(1)
                        .ok_or(TraceError::TraceCountOverflow {
                            counter: "launch count",
                        })?;
                if kernel.function_unique_id <= 0 {
                    return Err(TraceError::FunctionUniqueIdInvalid {
                        record: format!(
                            "root launch device={device_key} stream={stream_key} kernel={}",
                            kernel.id
                        ),
                        got: kernel.function_unique_id,
                    });
                }
                if kernel.number_of_registers < 0 {
                    return Err(TraceError::TraceLaunchRegistersInvalid {
                        device_id: device_key,
                        stream_id: stream_key,
                        kernel_id: kernel.id,
                        registers: kernel.number_of_registers,
                    });
                }
                if kernel
                    .shared_memory_carveout_preference_pct
                    .is_some_and(|preference| !(-1..=100).contains(&preference))
                {
                    return Err(TraceError::TraceLaunchCarveoutInvalid {
                        device_id: device_key,
                        stream_id: stream_key,
                        kernel_id: kernel.id,
                        preference: kernel
                            .shared_memory_carveout_preference_pct
                            .expect("checked as present"),
                    });
                }
                if kernel
                    .driver_occupancy_limit_blocks
                    .is_some_and(|blocks| blocks == 0)
                {
                    return Err(TraceError::TraceLaunchOccupancyInvalid {
                        device_id: device_key,
                        stream_id: stream_key,
                        kernel_id: kernel.id,
                        blocks: kernel
                            .driver_occupancy_limit_blocks
                            .expect("checked as present"),
                    });
                }
                if !positive_geometry(kernel.grid_dim.as_ref()) {
                    return Err(TraceError::LaunchGeometryInvalid {
                        device_id: device_key,
                        stream_id: stream_key,
                        kernel_id: kernel.id,
                        field: "grid_dim",
                    });
                }
                if !positive_geometry(kernel.block_dim.as_ref()) {
                    return Err(TraceError::LaunchGeometryInvalid {
                        device_id: device_key,
                        stream_id: stream_key,
                        kernel_id: kernel.id,
                        field: "block_dim",
                    });
                }
                validate_shared_memory_split(
                    device_key,
                    stream_key,
                    kernel,
                    memory_payload_contract,
                )?;

                let launch_dir = root
                    .join("threadblocks")
                    .join(format!("device_{device_key}"))
                    .join(format!("stream_{stream_key}"))
                    .join(format!("kernel_{}", kernel.id));
                let Some(paths) = payloads_by_parent.get(launch_dir.as_path()) else {
                    return Err(TraceError::ThreadblocksMissing { dir: launch_dir });
                };
                let mut block_ids = BTreeSet::new();
                for path in paths {
                    let validated = validate_payload(path, kernel)?;
                    if !block_ids.insert(validated.block_index) {
                        return Err(TraceError::BlockIdDuplicate {
                            path: path.to_path_buf(),
                            index: validated.block_index,
                        });
                    }
                    validated_payload.insert(path.clone(), validated.block);
                    consumed_payloads.insert(path.clone());
                    threadblock_count =
                        threadblock_count
                            .checked_add(1)
                            .ok_or(TraceError::TraceCountOverflow {
                                counter: "threadblock count",
                            })?;
                    instruction_count = instruction_count.checked_add(validated.records).ok_or(
                        TraceError::TraceCountOverflow {
                            counter: "instruction count",
                        },
                    )?;
                }
            }
        }
    }
    if launch_count == 0 {
        return Err(TraceError::TraceLaunchesEmpty);
    }
    if let Some(path) = payload_paths
        .iter()
        .find(|path| !consumed_payloads.contains(path.as_path()))
    {
        return Err(TraceError::ThreadblockPayloadOrphan { path: path.clone() });
    }
    Ok((
        TraceValidationSummary {
            launch_count,
            threadblock_count,
            instruction_count,
        },
        validated_payload,
    ))
}

#[cfg(feature = "campaign")]
pub(crate) fn validate_subset_producer_contract(
    root: &serde_json::Map<String, serde_json::Value>,
    original_grid: (u32, u32, u32),
    original_block: (u32, u32, u32),
    captured_block: (u32, u32, u32),
    selected_segments: &BTreeSet<u64>,
    certainty_segments: &BTreeSet<u64>,
    segment_blocks: &BTreeMap<u64, BTreeSet<u64>>,
) -> Result<ValidatedPerSmSlotMap, String> {
    let reason = string(required(root, "reason", "sampled subset")?, "reason")?;
    if ![
        "byte_budget",
        "walltime_budget",
        "gate_forced",
        "not_breaching",
    ]
    .contains(&reason)
    {
        return Err("reason is not a schema-v2 value".to_string());
    }
    if original_block != captured_block {
        return Err(
            "shape.captured_block must equal shape.original_block for a subset".to_string(),
        );
    }
    let shape = object(required(root, "shape", "sampled subset")?, "shape")?;
    if !shape
        .get("reduced_shape_variant")
        .is_some_and(serde_json::Value::is_null)
    {
        return Err("sampled subset requires reduced_shape_variant=null".to_string());
    }

    let gates = object(required(root, "gates", "sampled subset")?, "gates")?;
    pub(crate) const GATE_FIELDS: &[&str] = &[
        "coop",
        "wave_count",
        "n_measured_waves",
        "tail_measured_waves",
        "atomic_density",
        "hotline_share",
        "skew_top1pct_share",
        "gini",
        "per_sm_block_count_cv",
        "n_phases",
        "skew_heavy_block_count",
        "certainty_bytes",
        "max_trace_bytes",
        "certainty_budget_exceeded",
        "verdict",
    ];
    exact_keys(gates, GATE_FIELDS, GATE_FIELDS, "gates")?;
    if boolean(required(gates, "coop", "gates")?, "gates.coop")? {
        return Err("a cooperative launch cannot sanction a sampled subset".to_string());
    }
    let gate_wave_count = unsigned(required(gates, "wave_count", "gates")?, "gates.wave_count")?;
    let gate_measured = unsigned(
        required(gates, "n_measured_waves", "gates")?,
        "gates.n_measured_waves",
    )?;
    let gate_tail_measured = unsigned(
        required(gates, "tail_measured_waves", "gates")?,
        "gates.tail_measured_waves",
    )?;
    for field in [
        "atomic_density",
        "hotline_share",
        "skew_top1pct_share",
        "gini",
        "per_sm_block_count_cv",
    ] {
        nonnegative_number(required(gates, field, "gates")?, &format!("gates.{field}"))?;
    }
    positive(required(gates, "n_phases", "gates")?, "gates.n_phases")?;
    let skew_heavy_count = unsigned(
        required(gates, "skew_heavy_block_count", "gates")?,
        "gates.skew_heavy_block_count",
    )?;
    let gate_certainty_bytes = unsigned(
        required(gates, "certainty_bytes", "gates")?,
        "gates.certainty_bytes",
    )?;
    let gate_max_bytes = positive(
        required(gates, "max_trace_bytes", "gates")?,
        "gates.max_trace_bytes",
    )?;
    if boolean(
        required(gates, "certainty_budget_exceeded", "gates")?,
        "gates.certainty_budget_exceeded",
    )? {
        return Err("a certainty-budget-exceeded gate cannot sanction a subset".to_string());
    }
    if string(required(gates, "verdict", "gates")?, "gates.verdict")? != "subset_admissible" {
        return Err("a sampled subset requires gates.verdict=subset_admissible".to_string());
    }

    let tracer = object(required(root, "tracer", "sampled subset")?, "tracer")?;
    exact_keys(
        tracer,
        &["version", "features", "stats_pass_version"],
        &["version", "features", "stats_pass_version"],
        "tracer",
    )?;
    if unsigned(required(tracer, "version", "tracer")?, "tracer.version")? != 5
        || unsigned(
            required(tracer, "stats_pass_version", "tracer")?,
            "tracer.stats_pass_version",
        )? != 2
    {
        return Err("sampled subset requires tracer v5 and stats pass v2".to_string());
    }
    let features = object(required(tracer, "features", "tracer")?, "tracer.features")?;
    exact_keys(
        features,
        &["atomic_addr", "ldgsts_addr"],
        &["atomic_addr", "ldgsts_addr"],
        "tracer.features",
    )?;
    if !boolean(
        required(features, "atomic_addr", "tracer.features")?,
        "tracer.features.atomic_addr",
    )? || !boolean(
        required(features, "ldgsts_addr", "tracer.features")?,
        "tracer.features.ldgsts_addr",
    )? {
        return Err("sampled subset requires atomic_addr and ldgsts_addr features".to_string());
    }

    let rng = object(required(root, "rng", "sampled subset")?, "rng")?;
    exact_keys(rng, &["algo", "seed"], &["algo", "seed"], "rng")?;
    if string(required(rng, "algo", "rng")?, "rng.algo")? != "pcg64"
        || required(rng, "seed", "rng")?.as_i64().is_none()
    {
        return Err("sampled subset requires pcg64 with an integer seed".to_string());
    }
    if string(
        required(root, "label_pairing", "sampled subset")?,
        "label_pairing",
    )? != "full_shape_label"
    {
        return Err("sampled subset requires full_shape_label pairing".to_string());
    }

    let wave = object(
        required(root, "wave_model", "sampled subset")?,
        "wave_model",
    )?;
    pub(crate) const WAVE_FIELDS: &[&str] = &[
        "reconstruction",
        "per_sm_resident_blocks",
        "n_waves",
        "segment_len",
        "ghost_depth",
    ];
    exact_keys(wave, WAVE_FIELDS, WAVE_FIELDS, "wave_model")?;
    if string(
        required(wave, "reconstruction", "wave_model")?,
        "wave_model.reconstruction",
    )? != "launch_index"
    {
        return Err("wave_model.reconstruction must be launch_index".to_string());
    }
    let declared_n_waves = unsigned(
        required(wave, "n_waves", "wave_model")?,
        "wave_model.n_waves",
    )?;
    let segment_len = positive(
        required(wave, "segment_len", "wave_model")?,
        "wave_model.segment_len",
    )?;
    let ghost_depth = unsigned(
        required(wave, "ghost_depth", "wave_model")?,
        "wave_model.ghost_depth",
    )?;
    if segment_len != 3 || ghost_depth != 1 {
        return Err(
            "current schema-v2 producer requires segment_len=3 and ghost_depth=1".to_string(),
        );
    }

    let residency_values = required(wave, "per_sm_resident_blocks", "wave_model")?
        .as_array()
        .filter(|values| !values.is_empty())
        .ok_or_else(|| "wave_model.per_sm_resident_blocks must be a nonempty array".to_string())?;
    let mut resident_by_sm = BTreeMap::new();
    for (index, value) in residency_values.iter().enumerate() {
        let context = format!("wave_model.per_sm_resident_blocks[{index}]");
        let row = object(value, &context)?;
        exact_keys(
            row,
            &["sm_id", "resident_blocks"],
            &["sm_id", "resident_blocks"],
            &context,
        )?;
        let sm = unsigned(
            required(row, "sm_id", &context)?,
            &format!("{context}.sm_id"),
        )?;
        let resident = positive(
            required(row, "resident_blocks", &context)?,
            &format!("{context}.resident_blocks"),
        )?;
        if resident_by_sm.insert(sm, resident).is_some() {
            return Err(format!("duplicate resident-block entry for sm {sm}"));
        }
    }

    let slot_values = required(root, "per_sm_slot_map", "sampled subset")?
        .as_array()
        .filter(|values| !values.is_empty())
        .ok_or_else(|| "per_sm_slot_map must be a nonempty array".to_string())?;
    let mut roster_by_sm: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    let mut roster = BTreeSet::new();
    for (index, value) in slot_values.iter().enumerate() {
        let context = format!("per_sm_slot_map[{index}]");
        let row = object(value, &context)?;
        exact_keys(
            row,
            &["sm_id", "block_ids"],
            &["sm_id", "block_ids"],
            &context,
        )?;
        let sm = unsigned(
            required(row, "sm_id", &context)?,
            &format!("{context}.sm_id"),
        )?;
        let values = required(row, "block_ids", &context)?
            .as_array()
            .filter(|values| !values.is_empty())
            .ok_or_else(|| format!("{context}.block_ids must be a nonempty array"))?;
        let mut block_ids = Vec::with_capacity(values.len());
        for value in values {
            let block = unsigned(value, &format!("{context}.block_ids entry"))?;
            if block > i64::MAX as u64 {
                return Err(format!(
                    "sampled block id {block} exceeds the replay index range"
                ));
            }
            if block_ids.last().is_some_and(|previous| *previous >= block) {
                return Err(format!(
                    "{context}.block_ids must be strictly launch-index sorted"
                ));
            }
            if !roster.insert(block) {
                return Err(format!(
                    "block {block} occurs more than once in per_sm_slot_map"
                ));
            }
            block_ids.push(block);
        }
        if roster_by_sm.insert(sm, block_ids).is_some() {
            return Err(format!("duplicate per_sm_slot_map entry for sm {sm}"));
        }
    }
    if resident_by_sm.keys().ne(roster_by_sm.keys()) {
        return Err("resident-block SM keys must exactly equal per_sm_slot_map keys".to_string());
    }
    let grid_total = u64::from(original_grid.0)
        .checked_mul(u64::from(original_grid.1))
        .and_then(|value| value.checked_mul(u64::from(original_grid.2)))
        .ok_or_else(|| "original grid size overflows u64".to_string())?;
    if roster.len() as u64 != grid_total || roster.iter().copied().ne(0..grid_total) {
        return Err("per_sm_slot_map must exhaust launch indices 0..original_grid".to_string());
    }

    let reconstructed_n_waves = roster_by_sm
        .iter()
        .map(|(sm, blocks)| {
            let resident = resident_by_sm[sm];
            (blocks.len() as u64).div_ceil(resident)
        })
        .max()
        .unwrap_or(0);
    if reconstructed_n_waves != declared_n_waves || gate_wave_count != declared_n_waves {
        return Err("wave count disagrees with reconstructed per-SM cohorts".to_string());
    }
    let complete_waves = roster_by_sm
        .iter()
        .map(|(sm, blocks)| blocks.len() as u64 / resident_by_sm[sm])
        .min()
        .unwrap_or(0);
    let complete_cutoff = (complete_waves / segment_len) * segment_len;
    let residual_tail = reconstructed_n_waves > complete_cutoff;
    let tail_wave_count = if residual_tail {
        reconstructed_n_waves - complete_cutoff
    } else {
        segment_len.min(complete_cutoff)
    };
    // Deliberate semantic clamp: the Python producer defines ghost removal as
    // max(0, waves - ghosts), so the saturating subtractions here are contract
    // arithmetic rather than overflow recovery. The ghost product itself must
    // still be checked.
    let expected_tail_measured = tail_wave_count.saturating_sub(ghost_depth);
    let segment_count = complete_cutoff / segment_len + u64::from(residual_tail);
    let ghost_waves = ghost_depth
        .checked_mul(segment_count)
        .ok_or_else(|| "ghost-wave accounting overflows u64".to_string())?;
    let expected_measured = reconstructed_n_waves.saturating_sub(ghost_waves);
    if gate_tail_measured != expected_tail_measured || gate_measured != expected_measured {
        return Err("measured-wave gates disagree with reconstructed ghost accounting".to_string());
    }

    let mut expected_segments: BTreeMap<u64, BTreeSet<u64>> = BTreeMap::new();
    for (sm, blocks) in &roster_by_sm {
        let resident = resident_by_sm[sm];
        for (sequence, block) in blocks.iter().copied().enumerate() {
            let wave_index = sequence as u64 / resident;
            let segment_id = if wave_index < complete_cutoff {
                wave_index / segment_len
            } else {
                complete_cutoff / segment_len
            };
            expected_segments
                .entry(segment_id)
                .or_default()
                .insert(block);
        }
    }
    for (segment_id, blocks) in segment_blocks {
        if expected_segments.get(segment_id) != Some(blocks) {
            return Err(format!(
                "segment {segment_id} is not a complete reconstructed wave cohort"
            ));
        }
    }
    let cold_segment = *expected_segments
        .keys()
        .next()
        .ok_or_else(|| "wave reconstruction produced no segments".to_string())?;
    let candidate_tail = complete_cutoff / segment_len;
    let tail_segment = if expected_segments.contains_key(&candidate_tail) {
        candidate_tail
    } else {
        *expected_segments
            .keys()
            .next_back()
            .expect("nonempty above")
    };
    if !certainty_segments.contains(&cold_segment) || !certainty_segments.contains(&tail_segment) {
        return Err(
            "certainty_segments must contain reconstructed cold and tail cohorts".to_string(),
        );
    }

    let force_blocks = parse_unique_ids(
        required(root, "force_include_blocks", "sampled subset")?,
        "force_include_blocks",
        true,
    )?;
    if force_blocks.len() as u64 != skew_heavy_count {
        return Err("skew_heavy_block_count disagrees with force_include_blocks".to_string());
    }
    let certainty_blocks: BTreeSet<u64> = certainty_segments
        .iter()
        .flat_map(|segment| segment_blocks[segment].iter().copied())
        .collect();
    if !force_blocks.is_subset(&certainty_blocks) {
        return Err("force_include_blocks must be contained in certainty segments".to_string());
    }

    let strata_values = required(root, "strata", "sampled subset")?
        .as_array()
        .filter(|values| !values.is_empty())
        .ok_or_else(|| "strata must be a nonempty array".to_string())?;
    let mut stratum_ids = BTreeSet::new();
    let mut stratum_keys = BTreeSet::new();
    let mut sampled_ordinary = BTreeSet::new();
    pub(crate) const STRATUM_FIELDS: &[&str] = &[
        "id",
        "phase_id",
        "launch_tercile",
        "segment_ids",
        "N_h",
        "n_h",
        "pi",
        "sector_stddev",
        "neyman_weight",
    ];
    for (index, value) in strata_values.iter().enumerate() {
        let context = format!("strata[{index}]");
        let stratum = object(value, &context)?;
        exact_keys(stratum, STRATUM_FIELDS, STRATUM_FIELDS, &context)?;
        let id = unsigned(required(stratum, "id", &context)?, &format!("{context}.id"))?;
        let phase = unsigned(
            required(stratum, "phase_id", &context)?,
            &format!("{context}.phase_id"),
        )?;
        let tercile = unsigned(
            required(stratum, "launch_tercile", &context)?,
            &format!("{context}.launch_tercile"),
        )?;
        if tercile > 2 {
            return Err(format!("{context} id does not encode phase/launch tercile"));
        }
        let encoded_id = phase
            .checked_mul(3)
            .and_then(|value| value.checked_add(tercile))
            .ok_or_else(|| format!("{context} phase/launch tercile encoding overflows u64"))?;
        if id != encoded_id {
            return Err(format!("{context} id does not encode phase/launch tercile"));
        }
        if !stratum_ids.insert(id) || !stratum_keys.insert((phase, tercile)) {
            return Err(format!("{context} duplicates a stratum id or key"));
        }
        let ids = parse_unique_ids(
            required(stratum, "segment_ids", &context)?,
            &format!("{context}.segment_ids"),
            false,
        )?;
        if ids.len() < 2 {
            return Err(format!(
                "{context}.segment_ids must contain at least two ids"
            ));
        }
        let population = positive(
            required(stratum, "N_h", &context)?,
            &format!("{context}.N_h"),
        )?;
        let sample = positive(
            required(stratum, "n_h", &context)?,
            &format!("{context}.n_h"),
        )?;
        let ids_len = u64::try_from(ids.len())
            .map_err(|_| format!("{context}.segment_ids cardinality exceeds u64"))?;
        if sample < 2 || sample > population || !sampling_cardinalities_match(sample, ids_len) {
            return Err(format!("{context} has inconsistent N_h/n_h/segment_ids"));
        }
        let pi = nonnegative_number(required(stratum, "pi", &context)?, &format!("{context}.pi"))?;
        let stddev = nonnegative_number(
            required(stratum, "sector_stddev", &context)?,
            &format!("{context}.sector_stddev"),
        )?;
        let weight = nonnegative_number(
            required(stratum, "neyman_weight", &context)?,
            &format!("{context}.neyman_weight"),
        )?;
        if pi <= 0.0 || pi > 1.0 {
            return Err(format!("{context} has inconsistent design provenance"));
        }
        let population_f64 = sampling_u64_as_f64(population);
        let expected_pi = sampling_u64_as_f64(sample) / population_f64;
        // An absolute floor would admit arbitrarily wrong probabilities as N_h
        // grows. Keep only a relative allowance for cross-runtime float spelling.
        if !relative_close_enough(pi, expected_pi) {
            return Err(format!("{context} has inconsistent design provenance"));
        }
        let expected_weight = population_f64 * stddev;
        if !expected_weight.is_finite() {
            return Err(format!("{context} derived neyman_weight is not finite"));
        }
        if !close_enough(weight, expected_weight) {
            return Err(format!("{context} has inconsistent design provenance"));
        }
        if ids.iter().any(|id| !sampled_ordinary.insert(*id)) {
            return Err("ordinary segment occurs in multiple strata".to_string());
        }
    }
    if !sampled_ordinary.is_disjoint(certainty_segments) {
        return Err("certainty segments cannot also be sampled ordinary segments".to_string());
    }
    let selected_partition: BTreeSet<u64> = sampled_ordinary
        .union(certainty_segments)
        .copied()
        .collect();
    if &selected_partition != selected_segments {
        return Err(
            "strata sampled ids plus certainty ids must partition selected ids".to_string(),
        );
    }

    let budget = object(
        required(root, "selection_budget", "sampled subset")?,
        "selection_budget",
    )?;
    pub(crate) const BUDGET_FIELDS: &[&str] = &[
        "max_trace_bytes",
        "basis",
        "certainty_fraction_numerator",
        "certainty_fraction_denominator",
        "certainty_bytes",
    ];
    exact_keys(budget, BUDGET_FIELDS, BUDGET_FIELDS, "selection_budget")?;
    let max_bytes = positive(
        required(budget, "max_trace_bytes", "selection_budget")?,
        "selection_budget.max_trace_bytes",
    )?;
    let certainty_bytes = unsigned(
        required(budget, "certainty_bytes", "selection_budget")?,
        "selection_budget.certainty_bytes",
    )?;
    if string(
        required(budget, "basis", "selection_budget")?,
        "selection_budget.basis",
    )? != "raw_threadblock_pb"
        || unsigned(
            required(budget, "certainty_fraction_numerator", "selection_budget")?,
            "selection_budget.certainty_fraction_numerator",
        )? != 2
        || unsigned(
            required(budget, "certainty_fraction_denominator", "selection_budget")?,
            "selection_budget.certainty_fraction_denominator",
        )? != 5
        || max_bytes != gate_max_bytes
        || certainty_bytes != gate_certainty_bytes
        || 5_u128 * u128::from(certainty_bytes) > 2_u128 * u128::from(max_bytes)
    {
        return Err("selection budget disagrees with the exact raw-byte 2/5 contract".to_string());
    }

    let auxiliary = object(
        required(root, "selection_auxiliary", "sampled subset")?,
        "selection_auxiliary",
    )?;
    pub(crate) const AUX_FIELDS: &[&str] =
        &["name", "sector_bytes", "scope", "expected_full", "selected"];
    exact_keys(auxiliary, AUX_FIELDS, AUX_FIELDS, "selection_auxiliary")?;
    let expected_aux = unsigned(
        required(auxiliary, "expected_full", "selection_auxiliary")?,
        "selection_auxiliary.expected_full",
    )?;
    let selected_aux = unsigned(
        required(auxiliary, "selected", "selection_auxiliary")?,
        "selection_auxiliary.selected",
    )?;
    if string(
        required(auxiliary, "name", "selection_auxiliary")?,
        "selection_auxiliary.name",
    )? != "l2_sector_transactions_32b"
        || unsigned(
            required(auxiliary, "sector_bytes", "selection_auxiliary")?,
            "selection_auxiliary.sector_bytes",
        )? != 32
        || string(
            required(auxiliary, "scope", "selection_auxiliary")?,
            "selection_auxiliary.scope",
        )? != "pass2_all_blocks"
        || selected_aux > expected_aux
    {
        return Err("selection auxiliary envelope is inconsistent".to_string());
    }

    let bytes = object(required(root, "bytes", "sampled subset")?, "bytes")?;
    pub(crate) const BYTE_FIELDS: &[&str] =
        &["raw", "compressed", "codec", "level", "long", "dict_id"];
    exact_keys(bytes, BYTE_FIELDS, BYTE_FIELDS, "bytes")?;
    let raw_bytes = unsigned(required(bytes, "raw", "bytes")?, "bytes.raw")?;
    unsigned(required(bytes, "compressed", "bytes")?, "bytes.compressed")?;
    if raw_bytes > max_bytes
        || string(required(bytes, "codec", "bytes")?, "bytes.codec")? != "zstd"
        || required(bytes, "level", "bytes")?.as_i64() != Some(19)
        || required(bytes, "long", "bytes")?.as_i64() != Some(27)
        || !required(bytes, "dict_id", "bytes")?.is_null()
    {
        return Err("bytes envelope is not the current bounded zstd contract".to_string());
    }

    let estimator = object(required(root, "estimator", "sampled subset")?, "estimator")?;
    exact_keys(
        estimator,
        &["mode", "targets", "not_recoverable", "convergence"],
        &[
            "mode",
            "form",
            "ci",
            "targets",
            "not_recoverable",
            "convergence",
        ],
        "estimator",
    )?;
    if string(required(estimator, "mode", "estimator")?, "estimator.mode")?
        != "weighted_full_shape_estimate"
    {
        return Err("subset estimator mode must be weighted_full_shape_estimate".to_string());
    }
    let targets = required(estimator, "targets", "estimator")?
        .as_array()
        .filter(|values| !values.is_empty())
        .ok_or_else(|| "estimator.targets must be a nonempty array".to_string())?;
    for target in targets {
        string(target, "estimator.targets entry")?;
    }
    string_array(
        required(estimator, "not_recoverable", "estimator")?,
        "estimator.not_recoverable",
    )?;
    if let Some(form) = estimator.get("form") {
        string(form, "estimator.form")?;
    }
    if let Some(ci_value) = estimator.get("ci") {
        let ci = object(ci_value, "estimator.ci")?;
        exact_keys(ci, &[], &["method", "B", "alpha"], "estimator.ci")?;
        if let Some(method) = ci.get("method") {
            string(method, "estimator.ci.method")?;
        }
        if let Some(samples) = ci.get("B") {
            positive(samples, "estimator.ci.B")?;
        }
        if let Some(alpha) = ci.get("alpha") {
            nonnegative_number(alpha, "estimator.ci.alpha")?;
        }
    }
    let convergence = object(
        required(estimator, "convergence", "estimator")?,
        "estimator.convergence",
    )?;
    exact_keys(
        convergence,
        &["status"],
        &["status", "per_target"],
        "estimator.convergence",
    )?;
    let status = string(
        required(convergence, "status", "estimator.convergence")?,
        "estimator.convergence.status",
    )?;
    if !["converged", "budget_limited_wide_ci", "unavailable"].contains(&status) {
        return Err("estimator convergence status is invalid".to_string());
    }
    if convergence
        .get("per_target")
        .is_some_and(|value| !value.is_object())
    {
        return Err("estimator.convergence.per_target must be an object".to_string());
    }

    let records = object(required(root, "records", "sampled subset")?, "records")?;
    let expected_records = unsigned(
        required(records, "expected_full", "records")?,
        "records.expected_full",
    )?;
    let captured_records = unsigned(
        required(records, "captured", "records")?,
        "records.captured",
    )?;
    if captured_records > expected_records {
        return Err("records.captured cannot exceed records.expected_full".to_string());
    }
    ValidatedPerSmSlotMap::retain(&roster_by_sm, grid_total)
}

pub(crate) fn project_ldgsts_source_cohorts_128b_32x4(
    preflight: PreflightedMemoryAccess<'_>,
    source_sectors: &[i64],
    word_scratch: &mut Vec<i64>,
) -> Result<ProjectedLdgstsSourceCohorts, LdgstsDestinationProjectionError> {
    let mut source_index = None;
    let mut source_count = 0_usize;
    let mut destination_index = None;
    let mut destination_count = 0_usize;
    for (operand, address) in preflight.instruction().addresses.iter().enumerate() {
        let space = address
            .address_space
            .and_then(|raw| pb::MemorySpace::try_from(raw).ok());
        match space {
            Some(pb::MemorySpace::Global | pb::MemorySpace::Generic) => {
                source_count += 1;
                source_index.get_or_insert(operand);
            }
            Some(pb::MemorySpace::Shared) => {
                destination_count += 1;
                destination_index.get_or_insert(operand);
            }
            Some(pb::MemorySpace::Unspecified | pb::MemorySpace::Local) | None => {}
        }
    }
    if source_count != 1 {
        return Err(LdgstsDestinationProjectionError::SourceOperandCount { got: source_count });
    }
    if destination_count != 1 {
        return Err(LdgstsDestinationProjectionError::DestinationOperandCount {
            got: destination_count,
        });
    }
    let source = preflight
        .decoded_address(source_index.expect("count proved source index"))
        .expect("source index came from instruction addresses");
    let destination = preflight
        .decoded_address(destination_index.expect("count proved destination index"))
        .expect("destination index came from instruction addresses");
    if source.data_width_bytes != destination.data_width_bytes {
        return Err(LdgstsDestinationProjectionError::OperandWidthMismatch {
            source_bytes: source.data_width_bytes,
            destination_bytes: destination.data_width_bytes,
        });
    }
    let width = u64::try_from(source.data_width_bytes)
        .unwrap_or_else(|_| unreachable!("preflight admitted a non-positive operand width"));
    let width_u32 = u32::try_from(width)
        .unwrap_or_else(|_| unreachable!("one source transfer width escaped u32"));
    let instruction_mask = preflight.meta().and_mask;

    pub(crate) const EMPTY_TRANSFER: LdgstsLaneTransfer = LdgstsLaneTransfer {
        source_start: 0,
        source_end: 0,
        source_first_sector: 0,
        source_last_sector: 0,
        destination_first_word: 0,
        destination_last_word: 0,
        lane: 0,
        cohort: 0,
    };
    let mut transfers = [EMPTY_TRANSFER; lanes::LANES_PER_WARP];
    let mut transfer_count = 0_usize;
    let mut zero_fill_words = Vec::new();
    let mut zero_fill_transfers = 0_u32;
    let mut projected_words = 0_u64;
    for lane in 0..lanes::LANES_PER_WARP {
        let source_address = source.lane_vector[lane];
        let destination_address = destination.lane_vector[lane];
        if width > 128 {
            return Err(
                LdgstsDestinationProjectionError::SourceTransferExceedsCohort {
                    lane,
                    width_bytes: width,
                },
            );
        }
        if destination_address == lanes::LANE_INACTIVE_SENTINEL
            && (instruction_mask >> lane) & 1 == 1
        {
            return Err(LdgstsDestinationProjectionError::DestinationLaneMissing { lane });
        }
        if destination_address == lanes::LANE_INACTIVE_SENTINEL {
            continue;
        }
        if source_address == lanes::LANE_INACTIVE_SENTINEL {
            // cp.async ignore-src lanes have no global source cohort, but an
            // executing lane still performs destination zero-fill work. Keep
            // that work in a separate zero-payload cohort so source sectors
            // remain empty while async completion stays nonzero.
            let destination_width_offset = i64::try_from(width - 1)
                .unwrap_or_else(|_| unreachable!("128-byte transfer width escaped i64"));
            let destination_last_byte = destination_address
                .checked_add(destination_width_offset)
                .ok_or(
                LdgstsDestinationProjectionError::DestinationIntervalOverflow {
                    lane,
                    base: destination_address,
                    width_bytes: width,
                },
            )?;
            let first_word = destination_address.div_euclid(4);
            let last_word = destination_last_byte.div_euclid(4);
            let words = u64::try_from(last_word - first_word + 1).unwrap_or_else(|_| {
                unreachable!("checked destination interval produced negative words")
            });
            projected_words = projected_words.checked_add(words).ok_or(
                LdgstsDestinationProjectionError::InstructionWordLimit {
                    words: u64::MAX,
                    limit: MAX_SHARED_WORDS_PER_INSTRUCTION,
                },
            )?;
            if projected_words > MAX_SHARED_WORDS_PER_INSTRUCTION {
                return Err(LdgstsDestinationProjectionError::InstructionWordLimit {
                    words: projected_words,
                    limit: MAX_SHARED_WORDS_PER_INSTRUCTION,
                });
            }
            let additional = usize::try_from(words)
                .unwrap_or_else(|_| unreachable!("bounded destination words escaped usize"));
            zero_fill_words.try_reserve(additional).map_err(|_| {
                LdgstsDestinationProjectionError::CapacityUnavailable { additional }
            })?;
            zero_fill_words.extend(first_word..=last_word);
            zero_fill_transfers = zero_fill_transfers.checked_add(1).ok_or(
                LdgstsDestinationProjectionError::InstructionWordLimit {
                    words: u64::MAX,
                    limit: MAX_SHARED_WORDS_PER_INSTRUCTION,
                },
            )?;
            continue;
        }
        let source_start = source_address as u64;
        let source_end = source_start.checked_add(width).ok_or(
            LdgstsDestinationProjectionError::SourceIntervalOverflow {
                lane,
                base: source_start,
                width_bytes: width,
            },
        )?;
        let destination_width_offset = i64::try_from(width - 1)
            .unwrap_or_else(|_| unreachable!("128-byte transfer width escaped i64"));
        let destination_last_byte = destination_address
            .checked_add(destination_width_offset)
            .ok_or(
                LdgstsDestinationProjectionError::DestinationIntervalOverflow {
                    lane,
                    base: destination_address,
                    width_bytes: width,
                },
            )?;
        transfers[transfer_count] = LdgstsLaneTransfer {
            source_start,
            source_end,
            source_first_sector: source_address >> 5,
            source_last_sector: ((source_end - 1) as i64) >> 5,
            destination_first_word: destination_address.div_euclid(4),
            destination_last_word: destination_last_byte.div_euclid(4),
            lane: lane as u8,
            cohort: 0,
        };
        transfer_count += 1;
    }
    transfers[..transfer_count].sort_unstable_by_key(|transfer| {
        (transfer.source_start, transfer.source_end, transfer.lane)
    });

    word_scratch.clear();
    let mut cohort_start = None;
    let mut cohort_active_transfers = 0_u32;
    let mut cohort_payload_bytes = 0_u32;
    let mut cohort_ordinal = 0_u8;
    let mut cohorts = Vec::new();
    for transfer in &mut transfers[..transfer_count] {
        if cohort_start.is_some_and(|start| transfer.source_end - start > 128) {
            finish_ldgsts_destination_cohort(
                word_scratch,
                cohort_active_transfers,
                cohort_payload_bytes,
                &mut cohorts,
            )?;
            cohort_start = None;
            cohort_active_transfers = 0;
            cohort_payload_bytes = 0;
            cohort_ordinal = cohort_ordinal
                .checked_add(1)
                .ok_or(LdgstsDestinationProjectionError::SourceCohortCountOverflow)?;
        }
        cohort_start.get_or_insert(transfer.source_start);
        transfer.cohort = cohort_ordinal;
        cohort_active_transfers = cohort_active_transfers
            .checked_add(1)
            .expect("one source cohort cannot exceed the 32-lane warp");
        cohort_payload_bytes = cohort_payload_bytes
            .checked_add(width_u32)
            .expect("one source cohort payload cannot exceed 4096 bytes");
        let words =
            u64::try_from(transfer.destination_last_word - transfer.destination_first_word + 1)
                .unwrap_or_else(|_| {
                    unreachable!("checked destination interval produced negative words")
                });
        projected_words = projected_words.checked_add(words).ok_or(
            LdgstsDestinationProjectionError::InstructionWordLimit {
                words: u64::MAX,
                limit: MAX_SHARED_WORDS_PER_INSTRUCTION,
            },
        )?;
        if projected_words > MAX_SHARED_WORDS_PER_INSTRUCTION {
            return Err(LdgstsDestinationProjectionError::InstructionWordLimit {
                words: projected_words,
                limit: MAX_SHARED_WORDS_PER_INSTRUCTION,
            });
        }
        let additional = usize::try_from(words)
            .unwrap_or_else(|_| unreachable!("bounded destination words escaped usize"));
        word_scratch
            .try_reserve(additional)
            .map_err(|_| LdgstsDestinationProjectionError::CapacityUnavailable { additional })?;
        word_scratch.extend(transfer.destination_first_word..=transfer.destination_last_word);
    }
    finish_ldgsts_destination_cohort(
        word_scratch,
        cohort_active_transfers,
        cohort_payload_bytes,
        &mut cohorts,
    )?;

    let mut topology = recompose_ldgsts_destination_topology(&cohorts)?;
    if zero_fill_transfers != 0 {
        let mut zero_cohorts = Vec::new();
        finish_ldgsts_destination_cohort(
            &mut zero_fill_words,
            zero_fill_transfers,
            0,
            &mut zero_cohorts,
        )?;
        let zero = recompose_ldgsts_destination_topology(&zero_cohorts)?;
        topology.destination_line_fanout_total = topology
            .destination_line_fanout_total
            .checked_add(zero.destination_line_fanout_total)
            .ok_or(LdgstsDestinationProjectionError::DestinationLineFanoutOverflow)?;
        topology.destination_line_fanout_max = topology
            .destination_line_fanout_max
            .max(zero.destination_line_fanout_max);
        topology.cache_all_bank_wavefronts_total = topology
            .cache_all_bank_wavefronts_total
            .checked_add(zero.cache_all_bank_wavefronts_total)
            .ok_or(LdgstsDestinationProjectionError::WavefrontCountOverflow)?;
        topology.cache_all_bank_wavefronts_max = topology
            .cache_all_bank_wavefronts_max
            .max(zero.cache_all_bank_wavefronts_max);
        topology.bypass_wavefronts_total = topology
            .bypass_wavefronts_total
            .checked_add(zero.bypass_wavefronts_total)
            .ok_or(LdgstsDestinationProjectionError::BypassWavefrontCountOverflow)?;
        topology.bypass_wavefronts_max = topology
            .bypass_wavefronts_max
            .max(zero.bypass_wavefronts_max);
    }
    debug_assert_eq!(
        cohorts
            .iter()
            .map(|cohort| u64::from(cohort.payload_bytes))
            .sum::<u64>(),
        u64::from(width_u32) * transfer_count as u64
    );

    let mut source_sector_cohort_masks = Vec::new();
    source_sector_cohort_masks
        .try_reserve(source_sectors.len())
        .map_err(|_| LdgstsDestinationProjectionError::CapacityUnavailable {
            additional: source_sectors.len(),
        })?;
    for &sector in source_sectors {
        let mut mask = 0_u32;
        for transfer in &transfers[..transfer_count] {
            if sector >= transfer.source_first_sector && sector <= transfer.source_last_sector {
                mask |= 1_u32
                    .checked_shl(u32::from(transfer.cohort))
                    .unwrap_or_else(|| unreachable!("32-lane cohort ordinal escaped u32 mask"));
            }
        }
        debug_assert_ne!(mask, 0, "compact source sector must belong to a cohort");
        source_sector_cohort_masks.push(mask);
    }
    Ok(ProjectedLdgstsSourceCohorts {
        topology,
        cohorts,
        source_sector_cohort_masks,
    })
}

/// Enforce the Python-equivalent allocation identity on one instruction against
/// the loaded registry (`None` = no sidecar). The complete memory wire record is
/// preflighted before the first identity diagnosis, so an unknown enum or malformed
/// later lane vector cannot be masked by an earlier missing allocation id. A
/// GLOBAL/GENERIC-space record with active lanes must carry a non-zero
/// `allocation_id` in the registry containing every active lane address, or
/// the producer's explicit `unresolved_module_global` disposition. The latter
/// retains captured virtual addresses without claiming allocator bounds or
/// lifetime identity. SHARED/LOCAL records (e.g. ATOMS) are exempt.
pub fn check_allocation_identity(
    inst: &pb::Instruction,
    registry: Option<&AllocationRegistry>,
    record: impl Fn() -> String,
) -> Result<(), TraceError> {
    let Some(preflight) = preflight_memory_access(inst, &record)? else {
        return Ok(());
    };
    check_preflighted_allocation_identity(preflight, registry, record)
}

/// Apply the complete per-instruction contract and enclosing warp-identity
/// contract to every record in one decoded threadblock. Iteration follows sorted
/// protobuf map keys and then instruction order, matching the full-snapshot
/// admission walk.
pub fn validate_threadblock_contract(
    threadblock: &pb::Threadblock,
    label: impl AsRef<str>,
    registry: Option<&AllocationRegistry>,
    payload_contract: MemoryPayloadContract,
    expected_function_unique_id: i32,
) -> Result<(), TraceError> {
    let label = label.as_ref();
    for (&warp_id, warp) in &threadblock.warps {
        if warp.id != warp_id {
            return Err(TraceError::WarpMapKeyMismatch {
                threadblock: label.to_owned(),
                map_key: warp_id,
                warp_id: warp.id,
            });
        }
        let mut previous_instrumentation_order = None;
        for (instruction_index, instruction) in warp.instructions.iter().enumerate() {
            let record = || format!("{label} warp={warp_id} inst={instruction_index}");
            let preflight = preflight_memory_access(instruction, record)?;
            let Some(instruction_warp_id) = instruction.warpid_tb else {
                return Err(TraceError::WarpidTbMissing { record: record() });
            };
            if instruction_warp_id != warp_id {
                return Err(TraceError::WarpidTbMismatch {
                    record: record(),
                    expected: warp_id,
                    actual: instruction_warp_id,
                });
            }
            if instruction.function_unique_id != expected_function_unique_id {
                return Err(TraceError::FunctionUniqueIdMismatch {
                    record: record(),
                    expected: expected_function_unique_id,
                    actual: instruction.function_unique_id,
                });
            }
            if let Some(current) = instruction.instrumentation_order {
                if current == 0 {
                    return Err(TraceError::InstrumentationOrderZero { record: record() });
                }
                for (present, identity) in [
                    (instruction.sm_id.is_some(), "sm_id"),
                    (instruction.warpid_sm.is_some(), "warpid_sm"),
                ] {
                    if !present {
                        return Err(TraceError::InstrumentationOrderIdentityMissing {
                            record: record(),
                            identity,
                        });
                    }
                }
                if let Some(previous) = previous_instrumentation_order
                    && current <= previous
                {
                    return Err(TraceError::InstrumentationOrderNotIncreasing {
                        record: record(),
                        previous,
                        current,
                    });
                }
                previous_instrumentation_order = Some(current);
            }
            let Some(preflight) = preflight else {
                continue;
            };
            if matches!(
                payload_contract,
                MemoryPayloadContract::AddressCarryingV1
                    | MemoryPayloadContract::AddressCarryingSharedMemorySplitV2
                    | MemoryPayloadContract::AddressCarryingLdgstsDestinationV3
            ) {
                validate_address_carrying_shape(preflight, payload_contract, &record)?;
            }
            check_preflighted_allocation_identity(preflight, registry, record)?;
        }
    }
    Ok(())
}
