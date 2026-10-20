//! Cohesive child mechanism extracted from the parent module.

use super::*;

/// Canonical key emitted by the validated trace for a per-lane global memory
/// record. The alias keeps the trace projection and value-sidecar join on one
/// identity type without exposing cache or timing semantics here.
pub type DynamicMemoryJoinIdentity = sim_mem::NativePayloadJoinKey;

#[cfg(feature = "campaign")]
pub(crate) const CAMPAIGN_THREADBLOCK_FILE_MAX_BYTES: u64 = 768 * MIB;

pub(crate) const DIRECT_THREADBLOCK_FILE_MAX_COUNT: u64 = 1_000_000;

pub(crate) const ALLOCATION_ENTRY_MAX_COUNT: u64 = 131_072;

#[cfg(feature = "campaign")]
pub(crate) const TRACE_SAMPLING_JSON_MAX_NODES: u64 = 1_000_000;

impl<'a> KernelLoc<'a> {
    pub fn device_id(self) -> i64 {
        self.device_id
    }
}

impl<'a> KernelLoc<'a> {
    pub fn stream_id(self) -> i64 {
        self.stream_id
    }
}

impl<'a> KernelLoc<'a> {
    pub fn kernel(self) -> &'a pb::Kernel {
        self.kernel
    }
}

pub(crate) const ADDRESS_CARRYING_PAYLOAD_CAPABILITY_VECTOR: [i32; 5] = [1, 2, 3, 6, 8];

pub(crate) fn classify_memory_payload_contract(
    trace: &pb::Trace,
    path: &Path,
) -> Result<MemoryPayloadContract, TraceError> {
    if trace.trace_contract_id.is_empty() {
        if trace.memory_payload_classes.is_empty() {
            return Ok(MemoryPayloadContract::LegacyExclusion);
        }
        return Err(TraceError::MemoryPayloadCapabilitiesInvalid {
            path: path.to_path_buf(),
            contract_id: trace.trace_contract_id.clone(),
            got: trace.memory_payload_classes.clone(),
        });
    }
    let contract = match trace.trace_contract_id.as_str() {
        ADDRESS_CARRYING_TRACE_CONTRACT_ID => MemoryPayloadContract::AddressCarryingV1,
        SHARED_MEMORY_SPLIT_TRACE_CONTRACT_ID => {
            MemoryPayloadContract::AddressCarryingSharedMemorySplitV2
        }
        LDGSTS_DESTINATION_TRACE_CONTRACT_ID | SM120_LDGSTS_DESTINATION_TRACE_CONTRACT_ID => {
            MemoryPayloadContract::AddressCarryingLdgstsDestinationV3
        }
        _ => {
            return Err(TraceError::TraceContractUnsupported {
                path: path.to_path_buf(),
                got: trace.trace_contract_id.clone(),
            });
        }
    };

    // V2 and V3 use the repeated enum field as an ordered capability vector,
    // not a set. Keep v1's historical set-based admission below unchanged.
    if contract.carries_shared_memory_split()
        && !trace
            .memory_payload_classes
            .iter()
            .copied()
            .eq(ADDRESS_CARRYING_PAYLOAD_CAPABILITY_VECTOR)
    {
        return Err(TraceError::MemoryPayloadCapabilitiesInvalid {
            path: path.to_path_buf(),
            contract_id: trace.trace_contract_id.clone(),
            got: trace.memory_payload_classes.clone(),
        });
    }

    let expected: BTreeSet<_> = ADDRESS_CARRYING_PAYLOAD_CLASSES.into_iter().collect();
    let mut found = BTreeSet::new();
    for &raw in &trace.memory_payload_classes {
        let Ok(class) = pb::MemoryOpClass::try_from(raw) else {
            return Err(TraceError::MemoryPayloadCapabilitiesInvalid {
                path: path.to_path_buf(),
                contract_id: trace.trace_contract_id.clone(),
                got: trace.memory_payload_classes.clone(),
            });
        };
        if !found.insert(class) {
            return Err(TraceError::MemoryPayloadCapabilitiesInvalid {
                path: path.to_path_buf(),
                contract_id: trace.trace_contract_id.clone(),
                got: trace.memory_payload_classes.clone(),
            });
        }
    }
    if found != expected {
        return Err(TraceError::MemoryPayloadCapabilitiesInvalid {
            path: path.to_path_buf(),
            contract_id: trace.trace_contract_id.clone(),
            got: trace.memory_payload_classes.clone(),
        });
    }
    Ok(contract)
}

impl<'de> DeserializeSeed<'de> for JsonValueSeed<'_> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if self.top_level_sequence_entry {
            self.counter.bump_top_level_sequence_entries()?;
        }
        self.counter.bump_nodes()?;
        deserializer.deserialize_any(JsonValueVisitor {
            counter: self.counter,
            root: self.root,
        })
    }
}

pub(crate) fn preflight_json_shape(
    raw: &[u8],
    maximum_nodes: u64,
    maximum_top_level_sequence_entries: Option<u64>,
) -> Result<JsonShape, JsonPreflightError> {
    let mut counter = JsonShapeCounter {
        nodes: 0,
        top_level_sequence_entries: 0,
        maximum_nodes,
        maximum_top_level_sequence_entries,
        exceeded: None,
    };
    let mut deserializer = serde_json::Deserializer::from_slice(raw);
    let parsed = JsonValueSeed {
        counter: &mut counter,
        root: true,
        top_level_sequence_entry: false,
    }
    .deserialize(&mut deserializer);
    if let Some(exceeded) = counter.exceeded {
        return Err(JsonPreflightError::ResourceLimit(exceeded));
    }
    parsed.map_err(|error| JsonPreflightError::Malformed(error.to_string()))?;
    deserializer
        .end()
        .map_err(|error| JsonPreflightError::Malformed(error.to_string()))?;
    Ok(JsonShape {
        nodes: counter.nodes,
        top_level_sequence_entries: counter.top_level_sequence_entries,
    })
}

#[cfg(feature = "campaign")]
pub(crate) fn insert_snapshot_file_identity(
    identities: &mut BTreeMap<FileIdentity, (PathBuf, FileGenerationIdentity)>,
    artifact: &'static str,
    preserve_manifest_artifact_role: bool,
    generation: FileGenerationIdentity,
    path: PathBuf,
) -> Result<(), TraceError> {
    if let Some((first, _)) =
        identities.insert(generation.file_identity(), (path.clone(), generation))
    {
        let source = AnchoredReadError::HardlinkAlias {
            first,
            second: path,
        };
        return Err(if preserve_manifest_artifact_role {
            TraceError::SnapshotArtifactRead { artifact, source }
        } else {
            TraceError::AnchoredSnapshot(source)
        });
    }
    Ok(())
}

pub(crate) fn positive_geometry(dim: Option<&pb::Dim3d>) -> bool {
    dim.is_some_and(|dim| dim.x > 0 && dim.y > 0 && dim.z > 0)
}

pub(crate) fn validate_shared_memory_split(
    device_id: i64,
    stream_id: i64,
    kernel: &pb::Kernel,
    contract: MemoryPayloadContract,
) -> Result<(), TraceError> {
    if kernel.size_shared_memory < 0 {
        return Err(TraceError::TraceLaunchSharedMemoryInvalid {
            device_id,
            stream_id,
            kernel_id: kernel.id,
            combined: kernel.size_shared_memory,
        });
    }

    match contract {
        MemoryPayloadContract::LegacyExclusion | MemoryPayloadContract::AddressCarryingV1 => {
            if kernel.shared_memory_static_bytes.is_some()
                || kernel.shared_memory_dynamic_bytes.is_some()
            {
                return Err(TraceError::SharedMemorySplitUnadvertised {
                    device_id,
                    stream_id,
                    kernel_id: kernel.id,
                });
            }
            if kernel.size_shared_memory != 0 {
                return Err(TraceError::SharedMemorySplitUnavailable {
                    device_id,
                    stream_id,
                    kernel_id: kernel.id,
                });
            }
        }
        MemoryPayloadContract::AddressCarryingSharedMemorySplitV2
        | MemoryPayloadContract::AddressCarryingLdgstsDestinationV3 => {
            let (Some(static_bytes), Some(dynamic_bytes)) = (
                kernel.shared_memory_static_bytes,
                kernel.shared_memory_dynamic_bytes,
            ) else {
                return Err(TraceError::SharedMemorySplitMissing {
                    device_id,
                    stream_id,
                    kernel_id: kernel.id,
                });
            };
            let sum = u64::from(static_bytes) + u64::from(dynamic_bytes);
            if sum != kernel.size_shared_memory as u64 {
                return Err(TraceError::SharedMemorySplitMismatch {
                    device_id,
                    stream_id,
                    kernel_id: kernel.id,
                    combined: kernel.size_shared_memory,
                    static_bytes,
                    dynamic_bytes,
                });
            }
        }
    }
    Ok(())
}

/// Exact topology and payload work retained for one flexible 128-byte LDGSTS
/// source cohort. Cohorts are ordered by their smallest uncovered source
/// address, matching [`project_ldgsts_source_cohort_topology_128b_32x4`].
/// Timing policy remains outside the trace reader.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LdgstsSourceCohort {
    /// Sum of active source-transfer widths assigned to this cohort.
    pub payload_bytes: u32,
    /// Distinct 128-byte shared destination lines touched by this cohort.
    pub destination_lines: u32,
    /// Maximum distinct 4-byte-word multiplicity on any shared bank.
    pub bank_wavefronts: u32,
    /// Cohort-local bypass projection:
    /// `min(active transfers, destination_lines * bank_wavefronts)`.
    pub bypass_wavefronts: u32,
}

pub(crate) fn local_canonical_sector_count(
    preflight: PreflightedMemoryAccess<'_>,
    fixed_lane_stride: Option<i64>,
) -> u16 {
    let mut sectors = std::collections::BTreeSet::new();
    preflight.visit_addresses(|access| {
        let width = i64::from(access.data_width_bytes);
        // Characterized SM86 local records advance adjacent lane-private
        // slots by the transfer width itself (4B scalar, 16B vector, ...).
        // This endpoint candidate remains outside replay state/timing.
        let lane_stride = fixed_lane_stride.unwrap_or_else(|| width.max(1));
        for (lane, &base) in access.lane_vector.iter().enumerate() {
            if base == lanes::LANE_INACTIVE_SENTINEL {
                continue;
            }
            let Some(lane_offset) = i64::try_from(lane)
                .ok()
                .and_then(|lane| lane.checked_mul(lane_stride))
            else {
                return;
            };
            let Some(first_byte) = base.checked_add(lane_offset) else {
                return;
            };
            let Some(last_byte) = first_byte.checked_add(width.saturating_sub(1)) else {
                return;
            };
            let first_sector = first_byte >> 5;
            let last_sector = last_byte >> 5;
            for sector in first_sector..=last_sector {
                sectors.insert(sector);
            }
        }
    });
    u16::try_from(sectors.len()).unwrap_or(u16::MAX)
}

pub(crate) fn warp_uniform_address(preflight: PreflightedMemoryAccess<'_>) -> bool {
    let mut first: Option<(i64, i32)> = None;
    let mut active = 0_u32;
    let mut uniform = true;
    preflight.visit_addresses(|access| {
        for lane in 0..lanes::LANES_PER_WARP {
            if (access.lane_mask >> lane) & 1 == 0 {
                continue;
            }
            let address = access.lane_vector[lane];
            if address == lanes::LANE_INACTIVE_SENTINEL {
                continue;
            }
            active = active.saturating_add(1);
            if let Some(expected) = first {
                if expected != (address, access.data_width_bytes) {
                    uniform = false;
                }
            } else {
                first = Some((address, access.data_width_bytes));
            }
        }
    });
    active != 0 && uniform
}

/// Project one address-carrying global atomic warp instruction onto the
/// 32-byte sector transactions observed at L1TEX/LTS on sm_86.
///
/// Distinct atomic words in one sector share a transaction, while lanes that
/// alias the same atomic word serialize. Consequently each touched sector is
/// repeated by the largest exact-address multiplicity among lanes touching
/// that sector. This retains ordinary coalescing (`lane_word`: four sectors),
/// distinct-line traffic (`lane_line`: 32 sectors), and same-word
/// serialization (`warp_word`: one sector repeated 32 times) without retaining
/// the full lane vector after trace validation.
pub(crate) fn compact_atomic_sector_transaction_vector_32b(
    preflight: PreflightedMemoryAccess<'_>,
) -> Result<Vec<i64>, (ValidatedSectorErrorPhase, ValidatedSectorError)> {
    let unique = compact_sector_vector_for_operand(preflight, None, false)?;
    if unique.is_empty() {
        return Ok(unique);
    }

    let mut intervals = Vec::new();
    let mut early_error = None;
    preflight.visit_addresses(|access| {
        if early_error.is_some() {
            return;
        }
        if access.data_width_bytes <= 0 {
            early_error = Some(ValidatedSectorError::DataWidthInvalid {
                data_width_bytes: access.data_width_bytes,
            });
            return;
        }
        let width_offset = i64::from(access.data_width_bytes) - 1;
        for (lane, &base) in access.lane_vector.iter().enumerate() {
            if base == lanes::LANE_INACTIVE_SENTINEL {
                continue;
            }
            let Some(last_address) = base.checked_add(width_offset) else {
                early_error = Some(ValidatedSectorError::AddressOverflow {
                    lane,
                    base,
                    data_width_bytes: access.data_width_bytes,
                });
                return;
            };
            if intervals.try_reserve(1).is_err() {
                early_error = Some(ValidatedSectorError::CapacityUnavailable {
                    buffer: "atomic address intervals",
                    additional: 1,
                });
                return;
            }
            intervals.push((base, base >> 5, last_address >> 5));
        }
    });
    if let Some(error) = early_error {
        return Err((ValidatedSectorErrorPhase::BeforeStaticRole, error));
    }
    intervals.sort_unstable();

    let mut multiplicities = Vec::new();
    multiplicities.try_reserve(unique.len()).map_err(|_| {
        (
            ValidatedSectorErrorPhase::AfterStaticRole,
            ValidatedSectorError::CapacityUnavailable {
                buffer: "atomic sector multiplicities",
                additional: unique.len(),
            },
        )
    })?;
    let mut transaction_count = 0_usize;
    for &sector in &unique {
        let mut previous_base = None;
        let mut aliases = 0_usize;
        let mut maximum_aliases = 0_usize;
        for &(base, first_sector, last_sector) in &intervals {
            if sector < first_sector || sector > last_sector {
                continue;
            }
            if previous_base == Some(base) {
                aliases += 1;
            } else {
                previous_base = Some(base);
                aliases = 1;
            }
            maximum_aliases = maximum_aliases.max(aliases);
        }
        debug_assert!(
            maximum_aliases > 0,
            "unique atomic sector has no lane interval"
        );
        transaction_count = transaction_count.checked_add(maximum_aliases).ok_or((
            ValidatedSectorErrorPhase::AfterStaticRole,
            ValidatedSectorError::SectorCountOverflow {
                sectors: usize::MAX,
            },
        ))?;
        if transaction_count > usize::from(u16::MAX) {
            return Err((
                ValidatedSectorErrorPhase::AfterStaticRole,
                ValidatedSectorError::SectorCountOverflow {
                    sectors: transaction_count,
                },
            ));
        }
        multiplicities.push(maximum_aliases);
    }

    let mut staged = Vec::new();
    staged.try_reserve(transaction_count).map_err(|_| {
        (
            ValidatedSectorErrorPhase::AfterStaticRole,
            ValidatedSectorError::CapacityUnavailable {
                buffer: "atomic sector transaction staging",
                additional: transaction_count,
            },
        )
    })?;
    for (&sector, &multiplicity) in unique.iter().zip(&multiplicities) {
        staged.extend(std::iter::repeat_n(sector, multiplicity));
    }
    Ok(staged)
}

pub(crate) fn validate_threadblock_payload(
    raw: &[u8],
    path: &Path,
    kernel: &pb::Kernel,
    allocations: Option<&AllocationRegistry>,
    memory_payload_contract: MemoryPayloadContract,
) -> Result<ValidatedPayload, TraceError> {
    if raw.is_empty() {
        return Err(TraceError::ThreadblockRecordsEmpty {
            path: path.to_path_buf(),
        });
    }
    let threadblock = decode_threadblock(raw, path)?;
    let block_index = block_launch_index(kernel, &threadblock, path)?;
    let records = threadblock.warps.values().try_fold(0_u64, |total, warp| {
        let count =
            u64::try_from(warp.instructions.len()).map_err(|_| TraceError::TraceCountOverflow {
                counter: "threadblock instruction count",
            })?;
        total
            .checked_add(count)
            .ok_or(TraceError::TraceCountOverflow {
                counter: "threadblock instruction count",
            })
    })?;
    if records == 0 {
        return Err(TraceError::ThreadblockRecordsEmpty {
            path: path.to_path_buf(),
        });
    }
    let label = path.to_string_lossy();
    validate_threadblock_contract(
        &threadblock,
        &label,
        allocations,
        memory_payload_contract,
        kernel.function_unique_id,
    )?;
    let block = ValidatedThreadblock::from_message(
        block_index,
        threadblock,
        memory_payload_contract,
        &label,
    )?;
    Ok(ValidatedPayload {
        block_index,
        records,
        block,
    })
}

/// Flattened launch index of one threadblock from the proto's canonical
/// `block_id` (CUDA row-major: `x + y*gx + z*gx*gy` — the same rule as the
/// Python consumer `l1_production_helper._flatten_block_id`). Fail-closed
/// on missing `block_id` and out-of-grid coordinates; duplicate detection
/// is the caller's job (it needs the full block set).
/// The validated `(gx, gy, gz)` grid of a launch record — the same presence +
/// strictly-positive check `block_launch_index` applies. Used by the reader/replay
/// completeness gate to compute `grid_total` and cross-check `gridDim` against a
/// trace-sampling sidecar's `original_grid`.
pub fn kernel_grid(kernel: &pb::Kernel) -> Result<(u32, u32, u32), TraceError> {
    let g = kernel
        .grid_dim
        .as_ref()
        .filter(|g| g.x > 0 && g.y > 0 && g.z > 0)
        .ok_or(TraceError::GridDimMissing {
            kernel_id: kernel.id,
        })?;
    // The filter guarantees strictly-positive i32 dims, so the u32 cast is safe.
    Ok((g.x as u32, g.y as u32, g.z as u32))
}

/// One canonical entry from `hashes.per_block_raw_sha256_manifest`.
#[cfg(feature = "campaign")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PayloadFileManifestEntry {
    pub(crate) size_bytes: u64,
    pub(crate) sha256: String,
}

#[cfg(feature = "campaign")]
pub(crate) const PAYLOAD_FILE_MANIFEST_PREFIX: &str = "trace-payload-file-manifest-v1\n";

pub(crate) fn safe_payload_manifest_path(path: &str) -> bool {
    if path.is_empty()
        || !path.is_ascii()
        || path
            .bytes()
            .any(|byte| !(b'!'..=b'~').contains(&byte) || byte == b'\\')
        || !path.ends_with(".pb")
    {
        return false;
    }
    let Some(file_name) = path.rsplit('/').next() else {
        return false;
    };
    let Some(file_stem) = file_name.strip_suffix(".pb") else {
        return false;
    };
    if file_stem.bytes().all(|byte| byte == b'.') {
        return false;
    }
    let mut parts = path.split('/');
    if parts.next() != Some("threadblocks") {
        return false;
    }
    let mut saw_payload_component = false;
    for part in parts {
        if part.is_empty() || part == "." || part == ".." {
            return false;
        }
        saw_payload_component = true;
    }
    saw_payload_component
}

#[cfg(feature = "campaign")]
pub(crate) fn sha256_hex(input: &[u8]) -> String {
    Sha256Digest::of(input).to_string()
}

#[cfg(feature = "campaign")]
pub(crate) fn exact_keys(
    object: &serde_json::Map<String, serde_json::Value>,
    required: &[&str],
    allowed: &[&str],
    context: &str,
) -> Result<(), String> {
    for key in required {
        if !object.contains_key(*key) {
            return Err(format!("{context} is missing required field {key}"));
        }
    }
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!("{context} contains unknown field {key}"));
        }
    }
    Ok(())
}

#[cfg(feature = "campaign")]
pub(crate) fn object<'a>(
    value: &'a serde_json::Value,
    context: &str,
) -> Result<&'a serde_json::Map<String, serde_json::Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{context} must be an object"))
}

#[cfg(feature = "campaign")]
pub(crate) fn required<'a>(
    object: &'a serde_json::Map<String, serde_json::Value>,
    field: &str,
    context: &str,
) -> Result<&'a serde_json::Value, String> {
    object
        .get(field)
        .ok_or_else(|| format!("{context} is missing required field {field}"))
}

#[cfg(feature = "campaign")]
pub(crate) fn parse_dim3(
    value: &serde_json::Value,
    context: &str,
) -> Result<(u32, u32, u32), String> {
    let dim = object(value, context)?;
    exact_keys(dim, &["x", "y", "z"], &["x", "y", "z"], context)?;
    Ok((
        positive_u32(required(dim, "x", context)?, &format!("{context}.x"))?,
        positive_u32(required(dim, "y", context)?, &format!("{context}.y"))?,
        positive_u32(required(dim, "z", context)?, &format!("{context}.z"))?,
    ))
}

#[cfg(feature = "campaign")]
pub(crate) fn parse_unique_ids(
    value: &serde_json::Value,
    context: &str,
    allow_empty: bool,
) -> Result<BTreeSet<u64>, String> {
    let values = value
        .as_array()
        .ok_or_else(|| format!("{context} must be an array"))?;
    if !allow_empty && values.is_empty() {
        return Err(format!("{context} must not be empty"));
    }
    let mut ids = BTreeSet::new();
    for value in values {
        let id = value
            .as_u64()
            .ok_or_else(|| format!("{context} entries must be nonnegative integers"))?;
        if !ids.insert(id) {
            return Err(format!("{context} contains duplicate id {id}"));
        }
    }
    Ok(ids)
}

#[cfg(feature = "campaign")]
pub(crate) fn boolean(value: &serde_json::Value, context: &str) -> Result<bool, String> {
    value
        .as_bool()
        .ok_or_else(|| format!("{context} must be boolean"))
}

#[cfg(feature = "campaign")]
pub(crate) fn nonnegative_number(value: &serde_json::Value, context: &str) -> Result<f64, String> {
    value
        .as_f64()
        .filter(|value| value.is_finite() && *value >= 0.0)
        .ok_or_else(|| format!("{context} must be a finite nonnegative number"))
}

#[cfg(feature = "campaign")]
pub(crate) fn sampling_cardinalities_match(sample: u64, ids_len: u64) -> bool {
    sample == ids_len
}

/// Convert the complete `u64` domain through two exactly representable `u32`
/// limbs. The final addition performs the one unavoidable binary64 rounding;
/// no unchecked integer-to-float narrowing hides which source bits participate.
#[cfg(feature = "campaign")]
pub(crate) fn sampling_u64_as_f64(value: u64) -> f64 {
    let bytes = value.to_be_bytes();
    let high = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let low = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    f64::from(high) * 4_294_967_296.0 + f64::from(low)
}

#[cfg(feature = "campaign")]
pub(crate) fn validate_lossless_producer_contract(
    root: &serde_json::Map<String, serde_json::Value>,
    original_grid: (u32, u32, u32),
    original_block: (u32, u32, u32),
    captured_grid: (u32, u32, u32),
    captured_block: (u32, u32, u32),
    selected_block_ids_sha256: &str,
) -> Result<(u64, ValidatedPerSmSlotMap), String> {
    pub(crate) const LOSSLESS_FIELDS: &[&str] = &[
        "schema_version",
        "cell_id",
        "kernel",
        "method",
        "reason",
        "shape",
        "gates",
        "wave_model",
        "per_sm_slot_map",
        "rng",
        "label_pairing",
        "tracer",
        "bytes",
        "records",
        "hashes",
    ];
    exact_keys(
        root,
        LOSSLESS_FIELDS,
        LOSSLESS_FIELDS,
        "lossless compressed sidecar",
    )?;

    let reason = string(
        required(root, "reason", "lossless compressed sidecar")?,
        "reason",
    )?;
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
    if original_grid != captured_grid || original_block != captured_block {
        return Err("lossless compression cannot rewrite the captured launch shape".to_string());
    }
    let shape = object(
        required(root, "shape", "lossless compressed sidecar")?,
        "shape",
    )?;
    if !shape
        .get("reduced_shape_variant")
        .is_some_and(serde_json::Value::is_null)
    {
        return Err("lossless compression requires reduced_shape_variant=null".to_string());
    }
    if string(
        required(root, "label_pairing", "lossless compressed sidecar")?,
        "label_pairing",
    )? != "full_shape_label"
    {
        return Err("lossless compression requires full_shape_label pairing".to_string());
    }

    let gates = object(
        required(root, "gates", "lossless compressed sidecar")?,
        "gates",
    )?;
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
    boolean(required(gates, "coop", "gates")?, "gates.coop")?;
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
    unsigned(
        required(gates, "skew_heavy_block_count", "gates")?,
        "gates.skew_heavy_block_count",
    )?;
    let certainty_bytes = match required(gates, "certainty_bytes", "gates")? {
        serde_json::Value::Null => None,
        value => Some(unsigned(value, "gates.certainty_bytes")?),
    };
    let max_trace_bytes = match required(gates, "max_trace_bytes", "gates")? {
        serde_json::Value::Null => None,
        value => Some(positive(value, "gates.max_trace_bytes")?),
    };
    let budget_exceeded = boolean(
        required(gates, "certainty_budget_exceeded", "gates")?,
        "gates.certainty_budget_exceeded",
    )?;
    match certainty_bytes {
        Some(certainty) => {
            let maximum = max_trace_bytes.ok_or_else(|| {
                "gates.certainty_bytes requires gates.max_trace_bytes".to_string()
            })?;
            let expected = 5_u128 * u128::from(certainty) > 2_u128 * u128::from(maximum);
            if budget_exceeded != expected {
                return Err(
                    "gates.certainty_budget_exceeded disagrees with the exact 2/5 rule".to_string(),
                );
            }
        }
        None if budget_exceeded => {
            return Err(
                "gates.certainty_budget_exceeded cannot be true without byte evidence".to_string(),
            );
        }
        None => {}
    }
    let verdict = string(required(gates, "verdict", "gates")?, "gates.verdict")?;
    if ![
        "subset_admissible",
        "coop_forbid",
        "wave_forbid",
        "tail_measurement_forbid",
        "atomic_forbid",
        "skew_to_reduced_shape",
        "certainty_budget_to_reduced_shape",
        "homogeneity_to_reduced_shape",
    ]
    .contains(&verdict)
    {
        return Err("gates.verdict is not a schema-v2 value".to_string());
    }

    let tracer = object(
        required(root, "tracer", "lossless compressed sidecar")?,
        "tracer",
    )?;
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
        return Err("lossless compression requires tracer v5 and stats pass v2".to_string());
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
        return Err(
            "lossless compression requires atomic_addr and ldgsts_addr features".to_string(),
        );
    }

    let rng = object(required(root, "rng", "lossless compressed sidecar")?, "rng")?;
    exact_keys(rng, &["algo", "seed"], &["algo", "seed"], "rng")?;
    string(required(rng, "algo", "rng")?, "rng.algo")?;
    if required(rng, "seed", "rng")?.as_i64().is_none() {
        return Err("rng.seed must be an integer".to_string());
    }

    let bytes = object(
        required(root, "bytes", "lossless compressed sidecar")?,
        "bytes",
    )?;
    pub(crate) const BYTE_FIELDS: &[&str] =
        &["raw", "compressed", "codec", "level", "long", "dict_id"];
    exact_keys(bytes, &["raw", "compressed", "codec"], BYTE_FIELDS, "bytes")?;
    let raw_bytes = unsigned(required(bytes, "raw", "bytes")?, "bytes.raw")?;
    unsigned(required(bytes, "compressed", "bytes")?, "bytes.compressed")?;
    string(required(bytes, "codec", "bytes")?, "bytes.codec")?;
    for field in ["level", "long"] {
        if bytes
            .get(field)
            .is_some_and(|value| value.as_i64().is_none() && value.as_u64().is_none())
        {
            return Err(format!("bytes.{field} must be an integer"));
        }
    }
    if bytes.get("dict_id").is_some_and(|value| {
        !value.is_null() && value.as_i64().is_none() && value.as_u64().is_none()
    }) {
        return Err("bytes.dict_id must be an integer or null".to_string());
    }

    let wave = object(
        required(root, "wave_model", "lossless compressed sidecar")?,
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

    let slot_values = required(root, "per_sm_slot_map", "lossless compressed sidecar")?
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
                    "lossless block id {block} exceeds the replay index range"
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
    let roster_len = u64::try_from(roster.len())
        .map_err(|_| "lossless block roster cardinality exceeds u64".to_string())?;
    if roster_len != grid_total || roster.iter().copied().ne(0..grid_total) {
        return Err("per_sm_slot_map must exhaust launch indices 0..original_grid".to_string());
    }
    let reconstructed_n_waves = roster_by_sm
        .iter()
        .map(|(sm, blocks)| {
            u64::try_from(blocks.len())
                .map(|count| count.div_ceil(resident_by_sm[sm]))
                .map_err(|_| "per-SM block roster cardinality exceeds u64".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap_or(0);
    if reconstructed_n_waves != declared_n_waves || gate_wave_count != declared_n_waves {
        return Err("wave count disagrees with reconstructed per-SM cohorts".to_string());
    }
    let complete_waves = roster_by_sm
        .iter()
        .map(|(sm, blocks)| {
            u64::try_from(blocks.len())
                .map(|count| count / resident_by_sm[sm])
                .map_err(|_| "per-SM block roster cardinality exceeds u64".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .min()
        .unwrap_or(0);
    let complete_cutoff = (complete_waves / segment_len) * segment_len;
    let residual_tail = reconstructed_n_waves > complete_cutoff;
    let tail_wave_count = if residual_tail {
        reconstructed_n_waves - complete_cutoff
    } else {
        segment_len.min(complete_cutoff)
    };
    let expected_tail_measured = tail_wave_count.saturating_sub(ghost_depth);
    let segment_count = (complete_cutoff / segment_len)
        .checked_add(u64::from(residual_tail))
        .ok_or_else(|| "lossless segment count overflows u64".to_string())?;
    let ghost_waves = ghost_depth
        .checked_mul(segment_count)
        .ok_or_else(|| "lossless ghost-wave accounting overflows u64".to_string())?;
    let expected_measured = reconstructed_n_waves.saturating_sub(ghost_waves);
    if gate_tail_measured != expected_tail_measured || gate_measured != expected_measured {
        return Err("measured-wave gates disagree with reconstructed ghost accounting".to_string());
    }

    let sorted_blocks: Vec<u64> = roster.into_iter().collect();
    let block_json = serde_json::to_string(&sorted_blocks)
        .map_err(|error| format!("could not canonicalize lossless block ids: {error}"))?;
    if selected_block_ids_sha256 != sha256_hex(block_json.as_bytes()) {
        return Err("selected_block_ids_sha256 does not match the full launch roster".to_string());
    }

    let records = object(
        required(root, "records", "lossless compressed sidecar")?,
        "records",
    )?;
    let expected_records = unsigned(
        required(records, "expected_full", "records")?,
        "records.expected_full",
    )?;
    let captured_records = unsigned(
        required(records, "captured", "records")?,
        "records.captured",
    )?;
    if captured_records != expected_records {
        return Err("lossless records.captured must equal records.expected_full".to_string());
    }
    Ok((
        raw_bytes,
        ValidatedPerSmSlotMap::retain(&roster_by_sm, grid_total)?,
    ))
}

/// Whole-warp shared service projection retained for scheduler timing. This
/// deliberately excludes the phase-local observer refinement so changing the
/// L1TEX counter model cannot change SM issue or cycle state.
pub(crate) fn project_shared_bank_wavefronts_32x4_legacy_with_policy(
    preflight: PreflightedMemoryAccess<'_>,
    word_scratch: &mut Vec<i64>,
    dedup_atomic: bool,
) -> Result<u32, SharedBankProjectionError> {
    word_scratch.clear();
    let mut projected_words = 0_u64;
    let mut failure = None;
    preflight.visit_addresses(|access| {
        if failure.is_some() {
            return;
        }
        let width = i128::from(access.data_width_bytes);
        for (lane, &address) in access.lane_vector.iter().enumerate() {
            if address == lanes::LANE_INACTIVE_SENTINEL {
                continue;
            }
            let first_word = i128::from(address).div_euclid(SHARED_BANK_WORD_BYTES_32X4);
            let last_byte = i128::from(address) + width - 1;
            let last_word = last_byte.div_euclid(SHARED_BANK_WORD_BYTES_32X4);
            let words = u64::try_from(last_word - first_word + 1)
                .unwrap_or_else(|_| unreachable!("positive data width produced a negative span"));
            if words > MAX_SHARED_WORDS_PER_LANE {
                failure = Some(SharedBankProjectionError::LaneWordLimit {
                    lane,
                    words,
                    limit: MAX_SHARED_WORDS_PER_LANE,
                });
                return;
            }
            let Some(next_projected_words) = projected_words.checked_add(words) else {
                failure = Some(SharedBankProjectionError::InstructionWordLimit {
                    words: u64::MAX,
                    limit: MAX_SHARED_WORDS_PER_INSTRUCTION,
                });
                return;
            };
            if next_projected_words > MAX_SHARED_WORDS_PER_INSTRUCTION {
                failure = Some(SharedBankProjectionError::InstructionWordLimit {
                    words: next_projected_words,
                    limit: MAX_SHARED_WORDS_PER_INSTRUCTION,
                });
                return;
            }
            let additional = usize::try_from(words)
                .unwrap_or_else(|_| unreachable!("bounded word span escaped usize"));
            if word_scratch.try_reserve(additional).is_err() {
                failure = Some(SharedBankProjectionError::CapacityUnavailable { additional });
                return;
            }
            for word in first_word..=last_word {
                word_scratch.push(
                    i64::try_from(word)
                        .unwrap_or_else(|_| unreachable!("i64 byte address word escaped i64")),
                );
            }
            projected_words = next_projected_words;
        }
    });
    if let Some(failure) = failure {
        return Err(failure);
    }
    let is_shared_atomic = preflight.meta().access_kind == Some(pb::MemoryOpAccessKind::Atomic)
        || preflight.meta().op_class == pb::MemoryOpClass::Atomic;
    if dedup_atomic || !is_shared_atomic {
        word_scratch.sort_unstable();
        word_scratch.dedup();
    }
    let mut bank_words = [0_u64; SHARED_BANK_COUNT_32X4 as usize];
    for &word in word_scratch.iter() {
        let bank = word.rem_euclid(SHARED_BANK_COUNT_32X4) as usize;
        bank_words[bank] = bank_words[bank]
            .checked_add(1)
            .ok_or(SharedBankProjectionError::CountOverflow { bank: bank as u8 })?;
    }
    u32::try_from(bank_words.into_iter().max().unwrap_or(0))
        .map_err(|_| SharedBankProjectionError::CountOverflow { bank: 0 })
}

/// Whole-warp shared service projection retained for SM120 scheduler timing.
pub fn project_shared_bank_wavefronts_32x4_legacy(
    preflight: PreflightedMemoryAccess<'_>,
    word_scratch: &mut Vec<i64>,
) -> Result<u32, SharedBankProjectionError> {
    project_shared_bank_wavefronts_32x4_legacy_with_policy(preflight, word_scratch, false)
}

/// SM86 compatibility projection from the original branch. It deduplicates
/// broadcasts for every shared record, including records whose producer
/// metadata carries an atomic class.
pub fn project_shared_bank_wavefronts_32x4_sm86(
    preflight: PreflightedMemoryAccess<'_>,
    word_scratch: &mut Vec<i64>,
) -> Result<u32, SharedBankProjectionError> {
    project_shared_bank_wavefronts_32x4_legacy_with_policy(preflight, word_scratch, true)
}

/// Whole-warp width-only ideal paired with the scheduler projection.
pub fn project_shared_bank_ideal_wavefronts_32x4_legacy(
    preflight: PreflightedMemoryAccess<'_>,
) -> Result<u32, SharedBankProjectionError> {
    let mut ideal_words = 0_u64;
    let mut failure = None;
    preflight.visit_addresses(|access| {
        if failure.is_some() {
            return;
        }
        if access.data_width_bytes <= 0 {
            failure = Some(SharedBankProjectionError::LaneWordLimit {
                lane: 0,
                words: 0,
                limit: MAX_SHARED_WORDS_PER_LANE,
            });
            return;
        }
        let width = u64::try_from(access.data_width_bytes)
            .unwrap_or_else(|_| unreachable!("validated width escaped u64"));
        let words = width
            .checked_add((SHARED_BANK_WORD_BYTES_32X4 as u64) - 1)
            .and_then(|value| value.checked_div(SHARED_BANK_WORD_BYTES_32X4 as u64))
            .unwrap_or(u64::MAX);
        if words > MAX_SHARED_WORDS_PER_LANE {
            failure = Some(SharedBankProjectionError::LaneWordLimit {
                lane: 0,
                words,
                limit: MAX_SHARED_WORDS_PER_LANE,
            });
            return;
        }
        ideal_words = ideal_words.max(words);
    });
    if let Some(failure) = failure {
        return Err(failure);
    }
    u32::try_from(ideal_words).map_err(|_| SharedBankProjectionError::CountOverflow { bank: 0 })
}

/// Compute the ideal shared-memory wavefront count for one instruction under
/// the same 32-bank/4-byte-word contract as
/// [`project_shared_bank_wavefronts_32x4`].  The ideal count is the maximum
/// number of 4-byte words any active lane must transfer.  It therefore
/// preserves the unavoidable multi-wavefront cost of a wide per-lane access
/// (for example, an 8-byte load needs at least two wavefronts) while ignoring
/// inter-lane bank conflicts.  Nsight's shared-memory bank-conflict surface
/// measures excess wavefronts above this ideal, not excess above one request.
pub fn project_shared_bank_ideal_wavefronts_32x4(
    preflight: PreflightedMemoryAccess<'_>,
) -> Result<u32, SharedBankProjectionError> {
    let mut ideal_words = 0_u64;
    let mut max_data_width_bytes = 0_i32;
    let mut active_lane_mask = 0_u32;
    let mut failure = None;
    preflight.visit_addresses(|access| {
        if failure.is_some() {
            return;
        }
        max_data_width_bytes = max_data_width_bytes.max(access.data_width_bytes);
        for (lane, &address) in access.lane_vector.iter().enumerate() {
            if address != lanes::LANE_INACTIVE_SENTINEL {
                active_lane_mask |= 1_u32 << lane;
            }
        }
        if access.data_width_bytes <= 0 {
            failure = Some(SharedBankProjectionError::LaneWordLimit {
                lane: 0,
                words: 0,
                limit: MAX_SHARED_WORDS_PER_LANE,
            });
            return;
        }
        let width = u64::try_from(access.data_width_bytes).unwrap_or_else(|_| unreachable!());
        let words = width
            .checked_add((SHARED_BANK_WORD_BYTES_32X4 as u64) - 1)
            .and_then(|value| value.checked_div(SHARED_BANK_WORD_BYTES_32X4 as u64))
            .unwrap_or(u64::MAX);
        if words > MAX_SHARED_WORDS_PER_LANE {
            failure = Some(SharedBankProjectionError::LaneWordLimit {
                lane: 0,
                words,
                limit: MAX_SHARED_WORDS_PER_LANE,
            });
            return;
        }
        ideal_words = ideal_words.max(words);
    });
    if let Some(failure) = failure {
        return Err(failure);
    }

    // Match the phased service model used by
    // `project_shared_bank_wavefronts_32x4`: one ideal wavefront per active
    // half-warp (64-bit) or quarter-warp (128-bit).  The <=32-bit path and
    // widths above 128 bits retain the historical width-only ideal.
    let lanes_per_phase = match max_data_width_bytes {
        5..=8 => 16_u32,
        9..=16 => 8_u32,
        _ => 0_u32,
    };
    if lanes_per_phase != 0 {
        let mut phases = 0_u32;
        let phase_count = 32_u32 / lanes_per_phase;
        for phase in 0..phase_count {
            let phase_mask = ((1_u32 << lanes_per_phase) - 1) << (phase * lanes_per_phase);
            if active_lane_mask & phase_mask != 0 {
                phases = phases.saturating_add(1);
            }
        }
        return Ok(phases);
    }
    Ok(u32::try_from(ideal_words).unwrap_or_else(|_| unreachable!()))
}

pub(crate) fn recompose_ldgsts_destination_topology(
    cohorts: &[LdgstsSourceCohort],
) -> Result<LdgstsDestinationTopology, LdgstsDestinationProjectionError> {
    let mut topology = LdgstsDestinationTopology {
        source_cohorts: u32::try_from(cohorts.len())
            .map_err(|_| LdgstsDestinationProjectionError::SourceCohortCountOverflow)?,
        ..LdgstsDestinationTopology::default()
    };
    for cohort in cohorts {
        topology.destination_line_fanout_total = topology
            .destination_line_fanout_total
            .checked_add(cohort.destination_lines)
            .ok_or(LdgstsDestinationProjectionError::DestinationLineFanoutOverflow)?;
        topology.destination_line_fanout_max = topology
            .destination_line_fanout_max
            .max(cohort.destination_lines);
        topology.cache_all_bank_wavefronts_total = topology
            .cache_all_bank_wavefronts_total
            .checked_add(cohort.bank_wavefronts)
            .ok_or(LdgstsDestinationProjectionError::WavefrontCountOverflow)?;
        topology.cache_all_bank_wavefronts_max = topology
            .cache_all_bank_wavefronts_max
            .max(cohort.bank_wavefronts);
        topology.bypass_wavefronts_total = topology
            .bypass_wavefronts_total
            .checked_add(cohort.bypass_wavefronts)
            .ok_or(LdgstsDestinationProjectionError::BypassWavefrontCountOverflow)?;
        topology.bypass_wavefronts_max =
            topology.bypass_wavefronts_max.max(cohort.bypass_wavefronts);
    }
    Ok(topology)
}

/// Pair each active lane's LDGSTS source and destination transfer, form
/// flexible source cohorts whose complete byte span is at most 128 bytes, and
/// retain destination-line fanout separately from 32-bank/4-byte-word
/// multiplicity. The bypass projection is derived inside each cohort before
/// aggregation; it is an observational topology coordinate, not a timing rule.
/// Cohorts are anchored at the smallest uncovered source address rather than at
/// absolute 128-byte cache-line boundaries.
pub fn project_ldgsts_source_cohort_topology_128b_32x4(
    preflight: PreflightedMemoryAccess<'_>,
    word_scratch: &mut Vec<i64>,
) -> Result<LdgstsDestinationTopology, LdgstsDestinationProjectionError> {
    Ok(project_ldgsts_source_cohorts_128b_32x4(preflight, &[], word_scratch)?.topology)
}

/// A memory instruction decoded to what the memory hierarchy consumes —
/// the ALLOCATING convenience form (one heap `Vec` per memory record).
/// Hot walk paths (millions of records per kernel) use
/// [`visit_memory_access`] instead, which streams [`AddressAccess`] values
/// through a sink with zero per-record heap allocation.
#[derive(Debug, Clone)]
pub struct MemoryAccess {
    pub op_class: pb::MemoryOpClass,
    /// See [`MemoryAccessMeta::access_kind`].
    pub access_kind: Option<pb::MemoryOpAccessKind>,
    /// See [`MemoryAccessMeta::cache_op`].
    pub cache_op: Option<pb::InstructionCacheOp>,
    /// See [`MemoryAccessMeta::and_mask`].
    pub and_mask: u32,
    /// One decoded record per wire address record (empty when and_mask==0
    /// or the producer emitted no address payload).
    pub accesses: Vec<AddressAccess>,
}

pub(crate) fn memory_space_is_compatible(class: pb::MemoryOpClass, space: pb::MemorySpace) -> bool {
    match class {
        pb::MemoryOpClass::Global => {
            matches!(space, pb::MemorySpace::Global | pb::MemorySpace::Generic)
        }
        pb::MemoryOpClass::Shared => space == pb::MemorySpace::Shared,
        pb::MemoryOpClass::Local => space == pb::MemorySpace::Local,
        pb::MemoryOpClass::Atomic => matches!(
            space,
            pb::MemorySpace::Global
                | pb::MemorySpace::Generic
                | pb::MemorySpace::Shared
                | pb::MemorySpace::Local
        ),
        // The generic structural pass admits the v3 destination space; the
        // enclosing payload-contract shape gate subsequently requires the exact
        // ordered (GLOBAL|GENERIC source, SHARED destination) pair. Historical
        // source-only contracts retain their stricter one-operand shape there.
        pb::MemoryOpClass::AsyncCopy => matches!(
            space,
            pb::MemorySpace::Global | pb::MemorySpace::Generic | pb::MemorySpace::Shared
        ),
        pb::MemoryOpClass::Unspecified
        | pb::MemoryOpClass::Texture
        | pb::MemoryOpClass::Surface
        | pb::MemoryOpClass::Tma
        | pb::MemoryOpClass::ClusterDsmem
        | pb::MemoryOpClass::Tmem => true,
    }
}

/// Validate one traced instruction's complete memory wire contract without
/// invoking a consumer callback. A successful token borrows the immutable
/// instruction and may later stream its address records without another full
/// preflight.
///
/// "Memory instruction" uses the canonical Python predicate: address records OR
/// a present `memory_op_class`. Data-width alone does not reclassify a non-memory
/// record; access-kind or cache-op metadata on such a record is rejected as a
/// producer contradiction. Every instruction must carry `warpid_tb`, a
/// positive function id, both masks, and a low-32-consistent `vpc`. Memory records
/// additionally require a specified class and positive instruction width; unknown
/// enum values are typed schema skew, explicit memory-space UNSPECIFIED is rejected,
/// and every address record must reconstruct positionally. Non-memory instructions
/// return `Ok(None)` after their universal fields have been validated.
pub fn preflight_memory_access<'a>(
    inst: &'a pb::Instruction,
    record: impl Fn() -> String,
) -> Result<Option<PreflightedMemoryAccess<'a>>, TraceError> {
    let n_addresses = inst.addresses.len();
    // Canonical Python predicate: data_width and additive provenance fields do
    // not by themselves reclassify an otherwise non-memory instruction. The
    // additive fields are nevertheless contradictory on that record and are
    // rejected after their enum values receive complete wire validation.
    let looks_like_memory = n_addresses > 0 || inst.memory_op_class.is_some();

    // Keep the Python validator's fail-fast precedence: memory width first,
    // followed by the fields required on every instruction.
    let inst_width = if looks_like_memory {
        let Some(inst_width) = inst.data_width else {
            return Err(TraceError::DataWidthMissing { record: record() });
        };
        if inst_width <= 0 {
            return Err(TraceError::DataWidthInvalid {
                record: record(),
                got: inst_width,
            });
        }
        inst_width
    } else {
        0
    };

    if inst.warpid_tb.is_none() {
        return Err(TraceError::WarpidTbMissing { record: record() });
    }
    if inst.function_unique_id <= 0 {
        return Err(TraceError::FunctionUniqueIdInvalid {
            record: record(),
            got: inst.function_unique_id,
        });
    }
    let Some(active_mask) = inst.active_mask else {
        return Err(TraceError::ActiveMaskMissing { record: record() });
    };
    let Some(predicate_mask) = inst.predicate_mask else {
        return Err(TraceError::PredicateMaskMissing { record: record() });
    };
    let Some(vpc) = inst.vpc else {
        return Err(TraceError::VpcMissing { record: record() });
    };
    let vpc_low_32 = vpc as u32;
    if vpc_low_32 != inst.pc {
        return Err(TraceError::VpcInconsistentWithPc {
            record: record(),
            vpc_low_32,
            pc: inst.pc,
        });
    }

    if !looks_like_memory {
        let access_kind = match inst.memory_op_access_kind {
            None => None,
            Some(raw) => {
                let kind: pb::MemoryOpAccessKind =
                    known_enum(raw, "memory_op_access_kind", &record)?;
                if kind == pb::MemoryOpAccessKind::Unspecified {
                    return Err(TraceError::MemoryOpAccessKindUnspecified { record: record() });
                }
                Some(kind)
            }
        };
        let cache_op = match inst.instruction_cache_op {
            None => None,
            Some(raw) => {
                let op: pb::InstructionCacheOp = known_enum(raw, "instruction_cache_op", &record)?;
                if op == pb::InstructionCacheOp::Unspecified {
                    return Err(TraceError::InstructionCacheOpUnspecified { record: record() });
                }
                Some(op)
            }
        };
        if access_kind.is_some() {
            return Err(TraceError::MemoryMetadataOnNonMemoryInstruction {
                record: record(),
                field: "memory_op_access_kind",
            });
        }
        if cache_op.is_some() {
            return Err(TraceError::MemoryMetadataOnNonMemoryInstruction {
                record: record(),
                field: "instruction_cache_op",
            });
        }
        return Ok(None);
    }

    let Some(op_class_raw) = inst.memory_op_class else {
        return Err(TraceError::MemoryOpClassMissing {
            record: record(),
            n_addresses,
        });
    };
    let op_class: pb::MemoryOpClass = known_enum(op_class_raw, "memory_op_class", &record)?;
    if op_class == pb::MemoryOpClass::Unspecified {
        return Err(TraceError::MemoryOpClassUnspecified { record: record() });
    }

    let and_mask = active_mask & predicate_mask;
    // Validate every address, including zero-mask records. Those stream no
    // callbacks, but malformed lane payloads and unknown enums remain failures.
    for addr in &inst.addresses {
        decode_address_access(addr, and_mask, inst_width, &record)?;
    }
    // Structural preflight above intentionally completes before this semantic
    // compatibility gate. Explicit UNSPECIFIED is never a legacy fallback;
    // absence alone retains the class-derived behavior.
    for (operand, addr) in inst.addresses.iter().enumerate() {
        let Some(raw) = addr.address_space else {
            continue;
        };
        let space: pb::MemorySpace = known_enum(raw, "address_space", &record)?;
        if space == pb::MemorySpace::Unspecified {
            return Err(TraceError::MemorySpaceUnspecified {
                record: record(),
                operand,
            });
        }
        if !memory_space_is_compatible(op_class, space) {
            return Err(TraceError::MemoryOpClassAddressSpaceConflict {
                record: record(),
                operand,
                class: op_class,
                space,
            });
        }
    }

    let access_kind = match inst.memory_op_access_kind {
        None => None,
        Some(raw) => {
            let kind: pb::MemoryOpAccessKind = known_enum(raw, "memory_op_access_kind", &record)?;
            if kind == pb::MemoryOpAccessKind::Unspecified {
                return Err(TraceError::MemoryOpAccessKindUnspecified { record: record() });
            }
            Some(kind)
        }
    };
    let cache_op = match inst.instruction_cache_op {
        None => None,
        Some(raw) => {
            let op: pb::InstructionCacheOp = known_enum(raw, "instruction_cache_op", &record)?;
            if op == pb::InstructionCacheOp::Unspecified {
                return Err(TraceError::InstructionCacheOpUnspecified { record: record() });
            }
            Some(op)
        }
    };

    Ok(Some(PreflightedMemoryAccess {
        inst,
        meta: MemoryAccessMeta {
            op_class,
            access_kind,
            cache_op,
            and_mask,
        },
        inst_width,
    }))
}

/// Whether an effective per-address space requires global allocation identity.
/// A record's own `address_space` is authoritative (GLOBAL/GENERIC require it,
/// SHARED/LOCAL are exempt); absent falls back to the instruction-level GLOBAL
/// class — exactly the Python gate's rule.
pub(crate) fn record_requires_identity(
    space: Option<pb::MemorySpace>,
    inst_class: Option<pb::MemoryOpClass>,
) -> bool {
    match space {
        Some(pb::MemorySpace::Global) | Some(pb::MemorySpace::Generic) => true,
        Some(_) => false,
        None => inst_class == Some(pb::MemoryOpClass::Global),
    }
}

pub(crate) fn validate_address_carrying_shape(
    preflight: PreflightedMemoryAccess<'_>,
    contract: MemoryPayloadContract,
    record: &impl Fn() -> String,
) -> Result<(), TraceError> {
    let meta = preflight.meta();
    let instruction = preflight.instruction();
    if !ADDRESS_CARRYING_PAYLOAD_CLASSES.contains(&meta.op_class) {
        return Err(TraceError::MemoryPayloadClassUndeclared {
            record: record(),
            class: meta.op_class,
        });
    }

    let operand_count = instruction.addresses.len();
    let invalid = |detail| {
        Err(TraceError::MemoryPayloadShapeInvalid {
            record: record(),
            class: meta.op_class,
            detail,
            got: operand_count,
        })
    };
    match meta.op_class {
        pb::MemoryOpClass::Global | pb::MemoryOpClass::Shared | pb::MemoryOpClass::Local => {
            if meta.and_mask != 0 && operand_count == 0 {
                return invalid("an active record requires at least one operand");
            }
        }
        pb::MemoryOpClass::Atomic => {
            if meta.and_mask == 0 && operand_count == 0 {
                return Ok(());
            }
            if operand_count != 1 {
                return invalid("expected exactly one operand, except zero-active may carry none");
            }
            if instruction.addresses[0].address_space.is_none() {
                return invalid("the sole operand requires an explicit concrete address_space");
            }
        }
        pb::MemoryOpClass::AsyncCopy => {
            if contract.carries_ldgsts_destination() {
                if operand_count != 2 {
                    return invalid(
                        "expected exactly an ordered GLOBAL/GENERIC source and SHARED destination",
                    );
                }
                let source_space = instruction.addresses[0]
                    .address_space
                    .and_then(|raw| pb::MemorySpace::try_from(raw).ok());
                if !matches!(
                    source_space,
                    Some(pb::MemorySpace::Global | pb::MemorySpace::Generic)
                ) {
                    return invalid("operand 0 requires explicit GLOBAL or GENERIC source space");
                }
                let destination_space = instruction.addresses[1]
                    .address_space
                    .and_then(|raw| pb::MemorySpace::try_from(raw).ok());
                if destination_space != Some(pb::MemorySpace::Shared) {
                    return invalid("operand 1 requires explicit SHARED destination space");
                }
            } else {
                if operand_count != 1 {
                    return invalid("expected exactly one source operand");
                }
                let source_space = instruction.addresses[0]
                    .address_space
                    .and_then(|raw| pb::MemorySpace::try_from(raw).ok());
                if !matches!(
                    source_space,
                    Some(pb::MemorySpace::Global | pb::MemorySpace::Generic)
                ) {
                    return invalid("the source operand requires explicit GLOBAL or GENERIC space");
                }
            }
        }
        pb::MemoryOpClass::Unspecified
        | pb::MemoryOpClass::Texture
        | pb::MemoryOpClass::Surface
        | pb::MemoryOpClass::Tma
        | pb::MemoryOpClass::ClusterDsmem
        | pb::MemoryOpClass::Tmem => {
            unreachable!("undeclared classes return before shape validation")
        }
    }
    Ok(())
}

/// Allocating convenience wrapper over [`visit_memory_access`]: collects the
/// streamed records into a `Vec`. Identical contract and typed errors —
/// implemented on top of the streaming core so the two forms cannot drift.
pub fn memory_access(
    inst: &pb::Instruction,
    record: impl Fn() -> String,
) -> Result<Option<MemoryAccess>, TraceError> {
    let mut accesses = Vec::new();
    let meta = visit_memory_access(inst, record, |a| accesses.push(*a))?;
    Ok(meta.map(|m| MemoryAccess {
        op_class: m.op_class,
        access_kind: m.access_kind,
        cache_op: m.cache_op,
        and_mask: m.and_mask,
        accesses,
    }))
}
