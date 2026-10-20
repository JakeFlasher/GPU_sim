//! Cohesive child mechanism extracted from the parent module.

use super::*;

pub(crate) const MIB: u64 = 1024 * 1024;

pub(crate) const DYNAMIC_TRACE_PB_MAX_BYTES: u64 = 1024 * MIB;

pub(crate) const ALLOCATIONS_JSON_MAX_BYTES: u64 = 16 * MIB;

#[cfg(feature = "campaign")]
pub(crate) const TRACE_SAMPLING_JSON_MAX_BYTES: u64 = 64 * MIB;

pub(crate) const STATS_CSV_MAX_BYTES: u64 = 64 * MIB;

pub(crate) const DIRECT_THREADBLOCK_TOTAL_MAX_BYTES: u64 = 32 * 1024 * MIB;

pub(crate) const ALLOCATIONS_JSON_MAX_NODES: u64 = 1 + 7 * ALLOCATION_ENTRY_MAX_COUNT;

pub(crate) fn map_wire_preflight_error(
    path: &Path,
    message_type: &'static str,
    error: WirePreflightError,
) -> TraceError {
    match error {
        WirePreflightError::Malformed(detail) => TraceError::ProtobufWireMalformed {
            path: path.to_path_buf(),
            message_type,
            detail,
        },
        WirePreflightError::ResourceLimit {
            resource,
            observed,
            limit,
        } => TraceError::ProtobufDecodedResourceLimitExceeded {
            path: path.to_path_buf(),
            message_type,
            resource,
            observed,
            limit,
        },
    }
}

pub(crate) fn map_snapshot_read_error_with(
    artifact: &'static str,
    source: AnchoredReadError,
    anchored: impl FnOnce(AnchoredReadError) -> TraceError,
) -> TraceError {
    match source {
        AnchoredReadError::SizeMismatch {
            expected, actual, ..
        } => TraceError::SnapshotArtifactSizeMismatch {
            artifact,
            expected,
            actual,
        },
        AnchoredReadError::SizeLimitExceeded {
            path,
            limit,
            actual,
        } => TraceError::TraceArtifactResourceLimitExceeded {
            path,
            artifact,
            resource: "bytes",
            observed: actual,
            limit,
        },
        other => anchored(other),
    }
}

pub(crate) const DIRECT_EXEC_INFO_RELATIVE: &str = "extra_info/enhanced_execution_info.json";

/// Match the exec-info loader bound for full-workload direct replay.
pub(crate) const DIRECT_EXEC_INFO_MAX_BYTES: u64 = 256 * 1024 * 1024;

pub(crate) fn read_direct_exec_info(
    reader: &AnchoredDir,
) -> Result<Option<(ExecInfo, FileGenerationIdentity)>, TraceError> {
    let relative = Path::new(DIRECT_EXEC_INFO_RELATIVE);
    let path = reader.root().join(relative);
    let file = match reader
        .read_optional_bounded_regular(relative, DIRECT_EXEC_INFO_MAX_BYTES)
        .map_err(TraceError::AnchoredSnapshot)?
    {
        Some(file) => file,
        None => return Ok(None),
    };
    let generation = file.generation_identity();
    let raw = file.into_raw();
    let exec = ExecInfo::parse(&path, &raw).map_err(TraceError::ExecInfo)?;
    let actual = reader
        .inspect_regular_generation(relative)
        .map_err(TraceError::AnchoredSnapshot)?;
    if actual != generation {
        return Err(TraceError::ExecInfoChanged { path });
    }
    Ok(Some((exec, generation)))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct JsonShape {
    pub(crate) nodes: u64,
    pub(crate) top_level_sequence_entries: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct JsonResourceLimit {
    pub(crate) resource: &'static str,
    pub(crate) observed: u64,
    pub(crate) limit: u64,
}

#[derive(Debug)]
pub(crate) enum JsonPreflightError {
    Malformed(String),
    ResourceLimit(JsonResourceLimit),
}

impl<'de> Visitor<'de> for JsonValueVisitor<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a JSON value")
    }

    fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_i64<E>(self, _value: i64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_u64<E>(self, _value: u64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_str<E>(self, _value: &str) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_string<E>(self, _value: String) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        JsonValueSeed {
            counter: self.counter,
            root: false,
            top_level_sequence_entry: false,
        }
        .deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        while sequence
            .next_element_seed(JsonValueSeed {
                counter: self.counter,
                root: false,
                top_level_sequence_entry: self.root,
            })?
            .is_some()
        {}
        Ok(())
    }

    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut keys = BTreeSet::new();
        while let Some(key) = object.next_key::<String>()? {
            if !keys.insert(key.clone()) {
                return Err(A::Error::custom(format!(
                    "duplicate JSON object key {key:?}"
                )));
            }
            object.next_value_seed(JsonValueSeed {
                counter: self.counter,
                root: false,
                top_level_sequence_entry: false,
            })?;
        }
        Ok(())
    }
}

/// Failure while reducing active lane widths to exact effective payload bytes.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum EffectiveDataBytesError {
    #[error("effective memory payload byte count overflows u64")]
    Overflow,
}

/// Failure while reducing explicit shared addresses to the bounded projection
/// used by the sm_86 bank model. This is an implementation resource boundary,
/// not a wire-contract reinterpretation.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum SharedBankProjectionError {
    #[error(
        "lane {lane} spans {words} 4-byte words, exceeding the per-lane projection limit {limit}"
    )]
    LaneWordLimit { lane: usize, words: u64, limit: u64 },
    #[error(
        "instruction projects {words} 4-byte words, exceeding the per-instruction limit {limit}"
    )]
    InstructionWordLimit { words: u64, limit: u64 },
    #[error("projection scratch could not reserve {additional} additional words")]
    CapacityUnavailable { additional: usize },
    #[error("bank {bank} wavefront count overflowed u64")]
    CountOverflow { bank: u8 },
}

/// Compact per-instruction LDGSTS source/destination topology derived only
/// from the lane-paired trace operands. Timing policy remains outside the
/// trace reader.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LdgstsDestinationTopology {
    /// Number of flexible 128-byte source cohorts formed by the active lane
    /// transfers. Zero is the exact result for a predicated-off instruction.
    pub source_cohorts: u32,
    /// Sum of distinct 128-byte destination lines touched by source-coupled
    /// cohorts and executing source-ignored zero-fill lanes.
    pub destination_line_fanout_total: u32,
    /// Largest distinct destination-line fanout of any one source cohort.
    pub destination_line_fanout_max: u32,
    /// Existing cache-all projection: the sum, over source cohorts and
    /// zero-fill destination cohorts, of the maximum distinct 4-byte-word
    /// multiplicity on any of 32 shared banks.
    pub cache_all_bank_wavefronts_total: u32,
    /// Largest cache-all bank-word multiplicity of any destination cohort.
    pub cache_all_bank_wavefronts_max: u32,
    /// Trace-derived bypass projection. For each destination cohort this adds
    /// `min(active transfers, destination lines * bank wavefronts)` before
    /// summing, so line fanout and bank multiplicity are not collapsed before
    /// their cohort-local interaction is observed.
    pub bypass_wavefronts_total: u32,
    /// Largest cohort-local contribution to `bypass_wavefronts_total`.
    pub bypass_wavefronts_max: u32,
}

/// Failure while pairing one LDGSTS global-source transfer with its
/// lane-corresponding shared destination and reducing flexible 128-byte source
/// cohorts to destination-line and shared-bank topology.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum LdgstsDestinationProjectionError {
    #[error("expected exactly one GLOBAL/GENERIC source operand, got {got}")]
    SourceOperandCount { got: usize },
    #[error("expected exactly one SHARED destination operand, got {got}")]
    DestinationOperandCount { got: usize },
    #[error("lane {lane} has an active source but no corresponding shared destination")]
    DestinationLaneMissing { lane: usize },
    #[error(
        "source and destination operand widths differ: source={source_bytes} bytes, destination={destination_bytes} bytes"
    )]
    OperandWidthMismatch {
        source_bytes: i32,
        destination_bytes: i32,
    },
    #[error("lane {lane} source transfer width {width_bytes} exceeds one 128-byte cohort")]
    SourceTransferExceedsCohort { lane: usize, width_bytes: u64 },
    #[error(
        "lane {lane} source interval overflows the u64 address domain: base=0x{base:x}, width={width_bytes}"
    )]
    SourceIntervalOverflow {
        lane: usize,
        base: u64,
        width_bytes: u64,
    },
    #[error(
        "lane {lane} destination interval overflows the i64 address domain: base={base}, width={width_bytes}"
    )]
    DestinationIntervalOverflow {
        lane: usize,
        base: i64,
        width_bytes: u64,
    },
    #[error(
        "instruction projects {words} destination words, exceeding the per-instruction limit {limit}"
    )]
    InstructionWordLimit { words: u64, limit: u64 },
    #[error("destination projection scratch could not reserve {additional} additional words")]
    CapacityUnavailable { additional: usize },
    #[error("destination bank {bank} wavefront count overflowed u32")]
    BankWavefrontCountOverflow { bank: u8 },
    #[error("instruction destination wavefront sum overflowed u32")]
    WavefrontCountOverflow,
    #[error("instruction source-cohort count overflowed u32")]
    SourceCohortCountOverflow,
    #[error("instruction destination-line fanout sum overflowed u32")]
    DestinationLineFanoutOverflow,
    #[error(
        "cohort bypass projection overflowed u32: destination-line fanout {destination_lines} * bank wavefronts {bank_wavefronts}"
    )]
    BypassWavefrontProductOverflow {
        destination_lines: u32,
        bank_wavefronts: u32,
    },
    #[error("instruction bypass destination-wavefront sum overflowed u32")]
    BypassWavefrontCountOverflow,
    #[error("LDGSTS topology field {field}={value} exceeds compact limit {limit}")]
    CompactTopologyFieldOutOfRange {
        field: &'static str,
        value: u32,
        limit: u8,
    },
}

pub(crate) fn compact_sector_vector_for_operand(
    preflight: PreflightedMemoryAccess<'_>,
    selected_operand: Option<usize>,
    local_interleave: bool,
) -> Result<Vec<i64>, (ValidatedSectorErrorPhase, ValidatedSectorError)> {
    let mut intervals = Vec::new();
    let mut early_error = None;
    preflight.visit_addresses_enumerated(|operand, access| {
        if early_error.is_some() || selected_operand.is_some_and(|selected| selected != operand) {
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
            let base = if local_interleave {
                let word = base.div_euclid(4);
                let byte = base.rem_euclid(4);
                let words_per_lane = (access.data_width_bytes.max(4) + 3) / 4;
                let Some(lane_word) = word
                    .checked_mul(i64::try_from(lanes::LANES_PER_WARP).unwrap_or(32))
                    .and_then(|value| {
                        value.checked_add(
                            i64::try_from(lane)
                                .unwrap_or(0)
                                .checked_mul(i64::from(words_per_lane))?,
                        )
                    })
                else {
                    early_error = Some(ValidatedSectorError::AddressOverflow {
                        lane,
                        base,
                        data_width_bytes: access.data_width_bytes,
                    });
                    return;
                };
                let Some(expanded) = lane_word
                    .checked_mul(4)
                    .and_then(|value| value.checked_add(byte))
                else {
                    early_error = Some(ValidatedSectorError::AddressOverflow {
                        lane,
                        base,
                        data_width_bytes: access.data_width_bytes,
                    });
                    return;
                };
                expanded
            } else {
                base
            };
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
                    buffer: "sector interval plan",
                    additional: 1,
                });
                return;
            }
            intervals.push((base >> 5, last_address >> 5));
        }
    });
    if let Some(error) = early_error {
        return Err((ValidatedSectorErrorPhase::BeforeStaticRole, error));
    }

    let mut sorted = Vec::new();
    sorted.try_reserve(intervals.len()).map_err(|_| {
        (
            ValidatedSectorErrorPhase::AfterStaticRole,
            ValidatedSectorError::CapacityUnavailable {
                buffer: "sorted sector intervals",
                additional: intervals.len(),
            },
        )
    })?;
    sorted.extend_from_slice(&intervals);
    sorted.sort_unstable();
    let mut union_count = 0_u128;
    if let Some(&(initial_first, initial_last)) = sorted.first() {
        let mut first = initial_first;
        let mut last = initial_last;
        for &(next_first, next_last) in &sorted[1..] {
            if next_first <= last.saturating_add(1) {
                last = last.max(next_last);
            } else {
                union_count += (i128::from(last) - i128::from(first) + 1).unsigned_abs();
                first = next_first;
                last = next_last;
            }
        }
        union_count += (i128::from(last) - i128::from(first) + 1).unsigned_abs();
    }
    let unique_count = usize::try_from(union_count).map_err(|_| {
        (
            ValidatedSectorErrorPhase::AfterStaticRole,
            ValidatedSectorError::UniqueSectorCountOutOfRange {
                sectors: union_count,
            },
        )
    })?;
    if unique_count > usize::from(u16::MAX) {
        return Err((
            ValidatedSectorErrorPhase::AfterStaticRole,
            ValidatedSectorError::SectorCountOverflow {
                sectors: unique_count,
            },
        ));
    }

    // The union count above proves this expansion is bounded by u16::MAX.
    // Preserve first-seen interval/sector order with a set-backed walk instead
    // of repeatedly searching and draining the growing covered-interval list.
    let mut staged = Vec::new();
    staged.try_reserve(unique_count).map_err(|_| {
        (
            ValidatedSectorErrorPhase::AfterStaticRole,
            ValidatedSectorError::CapacityUnavailable {
                buffer: "sector staging",
                additional: unique_count,
            },
        )
    })?;
    let mut seen = HashSet::new();
    seen.try_reserve(unique_count).map_err(|_| {
        (
            ValidatedSectorErrorPhase::AfterStaticRole,
            ValidatedSectorError::CapacityUnavailable {
                buffer: "sector membership set",
                additional: unique_count,
            },
        )
    })?;
    for (first, last) in intervals {
        let mut sector = first;
        loop {
            if seen.insert(sector) {
                staged.push(sector);
            }
            if sector == last {
                break;
            }
            sector = sector
                .checked_add(1)
                .unwrap_or_else(|| unreachable!("validated interval end escaped i64"));
        }
    }
    if staged.len() != unique_count {
        return Err((
            ValidatedSectorErrorPhase::AfterStaticRole,
            ValidatedSectorError::CardinalityMismatch {
                expected: unique_count,
                produced: staged.len(),
            },
        ));
    }
    Ok(staged)
}

pub(crate) fn validate_trace_snapshot_semantics(
    root: &Path,
    trace: &pb::Trace,
    payload: &BTreeMap<PathBuf, Vec<u8>>,
    allocations: Option<&AllocationRegistry>,
    memory_payload_contract: MemoryPayloadContract,
) -> Result<
    (
        TraceValidationSummary,
        BTreeMap<PathBuf, ValidatedThreadblock>,
    ),
    TraceError,
> {
    validate_trace_semantics(
        root,
        trace,
        payload.keys().cloned(),
        memory_payload_contract,
        |path, kernel| {
            let raw =
                payload
                    .get(path)
                    .ok_or_else(|| TraceError::SamplingPayloadSnapshotMissing {
                        path: path.to_path_buf(),
                    })?;
            validate_threadblock_payload(raw, path, kernel, allocations, memory_payload_contract)
        },
    )
}

#[cfg(feature = "campaign")]
pub(crate) fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(feature = "campaign")]
pub(crate) fn parse_payload_file_manifest(
    manifest: &str,
) -> Result<BTreeMap<String, PayloadFileManifestEntry>, String> {
    let body = manifest
        .strip_prefix(PAYLOAD_FILE_MANIFEST_PREFIX)
        .and_then(|value| value.strip_suffix('\n'))
        .ok_or_else(|| {
            "per_block_raw_sha256_manifest must use the canonical v1 header and trailing newline"
                .to_string()
        })?;
    if body.is_empty() {
        return Err("per_block_raw_sha256_manifest must contain at least one .pb file".to_string());
    }

    let mut entries = BTreeMap::new();
    let mut previous_path: Option<&str> = None;
    for line in body.split('\n') {
        let mut fields = line.split('\t');
        let path = fields.next().unwrap_or_default();
        let size_text = fields.next().ok_or_else(|| {
            "per_block_raw_sha256_manifest entries must be path<TAB>size<TAB>sha256".to_string()
        })?;
        let sha256 = fields.next().ok_or_else(|| {
            "per_block_raw_sha256_manifest entries must be path<TAB>size<TAB>sha256".to_string()
        })?;
        if fields.next().is_some() {
            return Err(
                "per_block_raw_sha256_manifest entries must be path<TAB>size<TAB>sha256"
                    .to_string(),
            );
        }
        if !safe_payload_manifest_path(path) {
            return Err(format!(
                "per_block_raw_sha256_manifest path is not a canonical threadblocks .pb path: {path:?}"
            ));
        }
        if previous_path.is_some_and(|previous| path <= previous) {
            return Err(
                "per_block_raw_sha256_manifest paths must be unique and byte-sorted".to_string(),
            );
        }
        if size_text.is_empty()
            || !size_text.bytes().all(|byte| byte.is_ascii_digit())
            || (size_text.len() > 1 && size_text.starts_with('0'))
        {
            return Err(format!(
                "per_block_raw_sha256_manifest size is not canonical decimal: {size_text:?}"
            ));
        }
        let size_bytes = size_text.parse::<u64>().map_err(|_| {
            format!("per_block_raw_sha256_manifest size exceeds u64: {size_text:?}")
        })?;
        if !is_lower_hex_sha256(sha256) {
            return Err(
                "per_block_raw_sha256_manifest hashes must be lowercase SHA-256".to_string(),
            );
        }
        previous_path = Some(path);
        entries.insert(
            path.to_string(),
            PayloadFileManifestEntry {
                size_bytes,
                sha256: sha256.to_string(),
            },
        );
    }
    Ok(entries)
}

#[cfg(feature = "campaign")]
pub(crate) fn positive_u32(value: &serde_json::Value, context: &str) -> Result<u32, String> {
    let value = value
        .as_u64()
        .and_then(|raw| u32::try_from(raw).ok())
        .filter(|raw| *raw > 0)
        .ok_or_else(|| format!("{context} must be a positive u32"))?;
    Ok(value)
}

#[cfg(feature = "campaign")]
pub(crate) fn canonical_sidecar_sha256(sidecar: &serde_json::Value) -> Result<String, String> {
    let mut unsigned = sidecar.clone();
    let root = unsigned
        .as_object_mut()
        .ok_or_else(|| "trace_sampling.json root must be an object".to_string())?;
    let hashes = root
        .get_mut("hashes")
        .and_then(serde_json::Value::as_object_mut)
        .ok_or_else(|| "hashes must be an object".to_string())?;
    if hashes.remove("sidecar_sha256").is_none() {
        return Err("hashes is missing required field sidecar_sha256".to_string());
    }
    let canonical = canonical_json(&unsigned).map_err(|error| error.to_string())?;
    Ok(sha256_hex(canonical.as_bytes()))
}

#[cfg(feature = "campaign")]
pub(crate) fn unsigned(value: &serde_json::Value, context: &str) -> Result<u64, String> {
    value
        .as_u64()
        .ok_or_else(|| format!("{context} must be a nonnegative integer"))
}

#[cfg(feature = "campaign")]
pub(crate) fn positive(value: &serde_json::Value, context: &str) -> Result<u64, String> {
    unsigned(value, context).and_then(|value| {
        if value == 0 {
            Err(format!("{context} must be positive"))
        } else {
            Ok(value)
        }
    })
}

#[cfg(feature = "campaign")]
pub(crate) fn string<'a>(value: &'a serde_json::Value, context: &str) -> Result<&'a str, String> {
    value
        .as_str()
        .ok_or_else(|| format!("{context} must be a string"))
}

#[cfg(feature = "campaign")]
pub(crate) fn string_array(value: &serde_json::Value, context: &str) -> Result<(), String> {
    let values = value
        .as_array()
        .ok_or_else(|| format!("{context} must be an array"))?;
    for value in values {
        string(value, &format!("{context} entry"))?;
    }
    Ok(())
}

#[cfg(feature = "campaign")]
pub(crate) fn relative_close_enough(left: f64, right: f64) -> bool {
    left.is_finite()
        && right.is_finite()
        && (left - right).abs() <= 1e-12 * left.abs().max(right.abs())
}

#[cfg(feature = "campaign")]
pub(crate) fn close_enough(left: f64, right: f64) -> bool {
    left.is_finite()
        && right.is_finite()
        && (left - right).abs() <= 1e-12_f64.max(1e-12 * left.abs().max(right.abs()))
}

#[cfg(feature = "campaign")]
pub(crate) fn validate_sampling_payload_snapshot(
    root: &Path,
    sidecar: &TraceSamplingSidecar,
    snapshot: &VerifiedSamplingPayload,
) -> Result<(), TraceError> {
    let sidecar_path = root.join("trace_sampling.json");
    let malformed = |message: String| TraceError::SamplingSidecarMalformed {
        path: sidecar_path.clone(),
        message,
    };
    let mut actual_entries = BTreeMap::new();
    for (path, raw) in &snapshot.by_path {
        let relative = path
            .strip_prefix(root)
            .ok()
            .and_then(Path::to_str)
            .ok_or_else(|| {
                malformed(format!(
                    "snapshotted payload path is not canonical under the trace root: {}",
                    path.display()
                ))
            })?;
        actual_entries.insert(
            relative,
            (
                u64::try_from(raw.len())
                    .map_err(|_| malformed(format!("payload {relative:?} length exceeds u64")))?,
                Sha256Digest::of(raw),
            ),
        );
    }
    let actual_paths: BTreeSet<_> = actual_entries.keys().copied().collect();
    let declared_paths: BTreeSet<_> = sidecar
        .payload_file_manifest
        .keys()
        .map(String::as_str)
        .collect();
    if actual_paths != declared_paths {
        let missing: Vec<_> = declared_paths.difference(&actual_paths).take(4).collect();
        let unexpected: Vec<_> = actual_paths.difference(&declared_paths).take(4).collect();
        return Err(malformed(format!(
            "per_block_raw_sha256_manifest path set disagrees with threadblock payload: missing={missing:?}, unexpected={unexpected:?}"
        )));
    }
    for (relative, (size, digest)) in actual_entries {
        let declared = &sidecar.payload_file_manifest[relative];
        if size != declared.size_bytes {
            return Err(malformed(format!(
                "per_block_raw_sha256_manifest size for {relative:?} is {}, actual {size}",
                declared.size_bytes
            )));
        }
        if digest.to_string() != declared.sha256 {
            return Err(malformed(format!(
                "per_block_raw_sha256_manifest hash for {relative:?} does not match raw payload (actual {digest})"
            )));
        }
    }
    if snapshot.identity.raw_bytes != sidecar.raw_payload_bytes {
        return Err(malformed(format!(
            "bytes.raw {} does not match threadblock payload bytes {}",
            sidecar.raw_payload_bytes, snapshot.identity.raw_bytes
        )));
    }
    if snapshot.identity.sorted_path_stream_sha256.to_string()
        != sidecar.decompressed_payload_sha256
    {
        return Err(malformed(format!(
            "decompressed_payload_sha256 does not match sorted threadblock bytes (actual {})",
            snapshot.identity.sorted_path_stream_sha256
        )));
    }
    Ok(())
}

pub fn block_launch_index(
    kernel: &pb::Kernel,
    tb: &pb::Threadblock,
    path: &Path,
) -> Result<i64, TraceError> {
    let grid = kernel
        .grid_dim
        .as_ref()
        .filter(|g| g.x > 0 && g.y > 0 && g.z > 0)
        .ok_or(TraceError::GridDimMissing {
            kernel_id: kernel.id,
        })?;
    let (gx, gy, gz) = (i64::from(grid.x), i64::from(grid.y), i64::from(grid.z));
    let grid_plane = gx
        .checked_mul(gy)
        .ok_or(TraceError::GridArithmeticOverflow {
            kernel_id: kernel.id,
            operation: "grid plane size",
        })?;
    let grid_total = grid_plane
        .checked_mul(gz)
        .ok_or(TraceError::GridArithmeticOverflow {
            kernel_id: kernel.id,
            operation: "grid volume",
        })?;
    let Some(bid) = tb.block_id.as_ref() else {
        return Err(TraceError::BlockIdMissing {
            path: path.to_path_buf(),
        });
    };
    let (x, y, z) = (bid.x, bid.y, bid.z);
    // Reject invalid coordinates before signed launch-index arithmetic. A malformed
    // out-of-axis coordinate must remain a range error even when its flattened
    // value exceeds i64; i128 covers every i32-coordinate/i32-grid product.
    if x < 0 || y < 0 || z < 0 || i64::from(x) >= gx || i64::from(y) >= gy || i64::from(z) >= gz {
        let index =
            i128::from(x) + i128::from(y) * i128::from(gx) + i128::from(z) * i128::from(grid_plane);
        return Err(TraceError::BlockIdOutOfRange {
            path: path.to_path_buf(),
            x,
            y,
            z,
            index,
            grid: grid_total,
        });
    }
    let y_offset = i64::from(y)
        .checked_mul(gx)
        .ok_or(TraceError::GridArithmeticOverflow {
            kernel_id: kernel.id,
            operation: "block y offset",
        })?;
    let z_offset =
        i64::from(z)
            .checked_mul(grid_plane)
            .ok_or(TraceError::GridArithmeticOverflow {
                kernel_id: kernel.id,
                operation: "block z offset",
            })?;
    let index = i64::from(x)
        .checked_add(y_offset)
        .and_then(|value| value.checked_add(z_offset))
        .ok_or(TraceError::GridArithmeticOverflow {
            kernel_id: kernel.id,
            operation: "flattened block index",
        })?;
    // The per-axis checks above prevent an invalid CTA from flattening into a
    // valid launch index. Retain the flattened range check as a fail-closed
    // invariant over the checked arithmetic.
    if index >= grid_total {
        return Err(TraceError::BlockIdOutOfRange {
            path: path.to_path_buf(),
            x,
            y,
            z,
            index: i128::from(index),
            grid: grid_total,
        });
    }
    Ok(index)
}

/// Read one per-threadblock `.pb` file.
pub(crate) fn decode_threadblock(raw: &[u8], path: &Path) -> Result<pb::Threadblock, TraceError> {
    wire_preflight::preflight_threadblock(raw)
        .map_err(|error| map_wire_preflight_error(path, "dynamic_trace.threadblock", error))?;
    pb::Threadblock::decode(raw).map_err(|source| TraceError::Decode {
        path: path.to_path_buf(),
        message_type: "dynamic_trace.threadblock",
        source,
    })
}

/// One wire address record decoded for the memory hierarchy: the
/// reconstructed 32-lane vector plus the per-lane access width in bytes.
/// Width resolution mirrors the canonical Python consumer
/// (`l1_production_helper`): the address record's own `data_width` wins
/// when strictly positive, else the instruction-level width.
#[derive(Debug, Clone, Copy)]
pub struct AddressAccess {
    pub lane_vector: [i64; lanes::LANES_PER_WARP],
    /// Operand-specific effective lanes. An absent wire field falls back to
    /// the instruction mask. Async-copy sources narrow this for ignore-src
    /// zero-fill lanes without hiding the SHARED destination writes.
    pub lane_mask: u32,
    pub data_width_bytes: i32,
    /// Per-address memory space (address-carrying contract). Absent on legacy
    /// pre-transition traces (the field is unset), where consumers fall back to
    /// the instruction-level [`MemoryAccessMeta::op_class`]. A record carrying
    /// an explicit space lets a consumer route the global source of an atomic /
    /// async-copy to the L2 stream while exempting a shared destination.
    pub address_space: Option<pb::MemorySpace>,
    /// Backing global allocation id (absent/0 = unresolved). The identity gate
    /// requires a non-zero id resolving to a live sidecar allocation for a
    /// GLOBAL/GENERIC-space active record; SHARED/LOCAL are exempt.
    pub allocation_id: Option<u32>,
}

/// The instruction-level memory identity, without the per-record payloads —
/// what [`visit_memory_access`] returns after streaming the payloads into
/// the caller's sink.
#[derive(Debug, Clone, Copy)]
pub struct MemoryAccessMeta {
    pub op_class: pb::MemoryOpClass,
    /// Absent on legacy traces (pre-tag-12); consumers route absent to their
    /// UNSPECIFIED bucket, mirroring the Python consumers.
    pub access_kind: Option<pb::MemoryOpAccessKind>,
    /// Absent or UNSPECIFIED means "tracer did not classify the cache
    /// operator" — downstream L1 modeling must treat that as typed missing
    /// provenance, not as a default operator. (A wire value UNKNOWN to the
    /// committed schema is a typed reader error instead — schema skew.)
    pub cache_op: Option<pb::InstructionCacheOp>,
    /// `active_mask & predicate_mask`. May be 0: a predicated-off memory
    /// instruction still ISSUES (the SM model sees it); it just touches no
    /// lanes — the sink receives no records in that case, mirroring the
    /// canonical consumers, which skip lane reconstruction on zero masks.
    pub and_mask: u32,
}

/// A successful full structural preflight tied to one immutable instruction.
/// Consumers may apply semantic gates to [`Self::meta`] before invoking
/// [`Self::visit_addresses`], which preserves structural-error precedence without
/// decoding the complete wire record again.
#[derive(Debug, Clone, Copy)]
pub struct PreflightedMemoryAccess<'a> {
    pub(crate) inst: &'a pb::Instruction,
    pub(crate) meta: MemoryAccessMeta,
    pub(crate) inst_width: i32,
}

impl<'a> PreflightedMemoryAccess<'a> {
    pub fn instruction(self) -> &'a pb::Instruction {
        self.inst
    }
}

impl<'a> PreflightedMemoryAccess<'a> {
    pub fn meta(self) -> MemoryAccessMeta {
        self.meta
    }
}

impl<'a> PreflightedMemoryAccess<'a> {
    /// Reconstruct each already-validated address into stack storage in wire
    /// order. The instruction borrow prevents mutation between preflight and
    /// this pass, making a second decode failure an unreachable invariant.
    pub fn visit_addresses<F>(self, mut sink: F)
    where
        F: FnMut(&AddressAccess),
    {
        self.visit_addresses_enumerated(|_, access| sink(access));
    }
}

impl<'a> PreflightedMemoryAccess<'a> {
    fn visit_addresses_enumerated<F>(self, mut sink: F)
    where
        F: FnMut(usize, &AddressAccess),
    {
        if self.meta.and_mask == 0 {
            return;
        }
        let no_error_context = || String::new();
        for (operand, addr) in self.inst.addresses.iter().enumerate() {
            let access = match decode_address_access(
                addr,
                self.meta.and_mask,
                self.inst_width,
                &no_error_context,
            ) {
                Ok(access) => access,
                Err(_) => unreachable!("preflighted immutable address failed a second decode"),
            };
            sink(operand, &access);
        }
    }
}

impl<'a> PreflightedMemoryAccess<'a> {
    pub(crate) fn decoded_address(self, operand: usize) -> Option<AddressAccess> {
        let addr = self.inst.addresses.get(operand)?;
        let no_error_context = || String::new();
        Some(
            decode_address_access(addr, self.meta.and_mask, self.inst_width, &no_error_context)
                .unwrap_or_else(|_| {
                    unreachable!("preflighted immutable address failed a second decode")
                }),
        )
    }
}

/// Sum each decoded address operand's per-lane width over its effective lanes.
/// This preserves payload volume independently of source-sector coalescing.
pub fn project_effective_data_bytes(
    preflight: PreflightedMemoryAccess<'_>,
) -> Result<u64, EffectiveDataBytesError> {
    let mut bytes = 0_u64;
    let mut overflowed = false;
    preflight.visit_addresses(|access| {
        if overflowed {
            return;
        }
        match effective_data_bytes_for_access(access) {
            Ok(operand_bytes) => match bytes.checked_add(operand_bytes) {
                Some(total) => bytes = total,
                None => overflowed = true,
            },
            Err(EffectiveDataBytesError::Overflow) => overflowed = true,
        }
    });
    if overflowed {
        Err(EffectiveDataBytesError::Overflow)
    } else {
        Ok(bytes)
    }
}

/// Count sectors whose decoded byte coverage is partial for one global store
/// instruction. This is a trace-side RMW-demand bound: it does not assert that
/// the line missed in L2, and callers must intersect it with their modeled
/// store-miss population before charging any lower-level read.
pub fn project_partial_store_sectors(
    preflight: PreflightedMemoryAccess<'_>,
) -> Result<u64, EffectiveDataBytesError> {
    let mut coverage = BTreeMap::<i64, u32>::new();
    let mut overflowed = false;
    preflight.visit_addresses(|access| {
        if overflowed {
            return;
        }
        let width = match u64::try_from(access.data_width_bytes) {
            Ok(width) => width,
            Err(_) => {
                overflowed = true;
                return;
            }
        };
        for &address in access.lane_vector.iter() {
            if address == lanes::LANE_INACTIVE_SENTINEL {
                continue;
            }
            for byte in 0..width {
                let byte_address = match i128::from(address).checked_add(i128::from(byte)) {
                    Some(value) => value,
                    None => {
                        overflowed = true;
                        break;
                    }
                };
                let sector = byte_address.div_euclid(32);
                let offset = byte_address.rem_euclid(32) as u32;
                let Ok(sector) = i64::try_from(sector) else {
                    overflowed = true;
                    break;
                };
                let entry = coverage.entry(sector).or_default();
                *entry |= 1_u32 << offset;
            }
            if overflowed {
                break;
            }
        }
    });
    if overflowed {
        return Err(EffectiveDataBytesError::Overflow);
    }
    Ok(coverage.values().filter(|mask| **mask != u32::MAX).count() as u64)
}

pub(crate) fn project_operand_effective_data_bytes(
    preflight: PreflightedMemoryAccess<'_>,
    operand: usize,
) -> Result<u64, EffectiveDataBytesError> {
    let access = preflight
        .decoded_address(operand)
        .unwrap_or_else(|| unreachable!("validated async-copy source operand disappeared"));
    effective_data_bytes_for_access(&access)
}

pub(crate) fn effective_data_bytes_for_access(
    access: &AddressAccess,
) -> Result<u64, EffectiveDataBytesError> {
    let active_lanes = access
        .lane_vector
        .iter()
        .filter(|&&address| address != lanes::LANE_INACTIVE_SENTINEL)
        .count() as u64;
    let width =
        u64::try_from(access.data_width_bytes).map_err(|_| EffectiveDataBytesError::Overflow)?;
    width
        .checked_mul(active_lanes)
        .ok_or(EffectiveDataBytesError::Overflow)
}

pub(crate) const SHARED_BANK_COUNT_32X4: i64 = 32;

pub(crate) const SHARED_BANK_WORD_BYTES_32X4: i128 = 4;

pub(crate) const MAX_SHARED_WORDS_PER_LANE: u64 = 1_024;

pub(crate) const MAX_SHARED_WORDS_PER_INSTRUCTION: u64 = 65_536;

/// Reduce effective shared addresses to the exact bank-wavefront observable for
/// 32 banks with 4-byte words.
///
/// Ordinary loads/stores use the broadcast rule: repeated references to one
/// 32-bit word are serviced by one bank word.  Shared atomics are different:
/// each active lane performs an independent read-modify-write, so even lanes
/// targeting the same word must be serialized by the bank's atomic unit.  The
/// access-kind metadata is part of the trace contract and lets us keep those
/// two observers separate without treating global atomics as shared traffic
/// (this function is only called after the route has resolved to `Shared`).
pub fn project_shared_bank_wavefronts_32x4(
    preflight: PreflightedMemoryAccess<'_>,
    word_scratch: &mut Vec<i64>,
) -> Result<u32, SharedBankProjectionError> {
    word_scratch.clear();
    let mut projected_words = 0_u64;
    let mut max_data_width_bytes = 0_i32;
    let mut failure = None;
    preflight.visit_addresses(|access| {
        if failure.is_some() {
            return;
        }
        max_data_width_bytes = max_data_width_bytes.max(access.data_width_bytes);
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
    if !is_shared_atomic {
        // SM120 services a 64-bit shared access in two independent 16-lane
        // phases and a 128-bit access in four independent 8-lane phases.
        // Broadcast collapse is local to each phase; deduplicating the whole
        // warp would incorrectly let one phase satisfy another.  Keep widths
        // above 128 bits on the historical whole-warp path because their
        // packetization is not represented by this compact contract.
        let lanes_per_phase = match max_data_width_bytes {
            5..=8 => 16_usize,
            9..=16 => 8_usize,
            _ => 0_usize,
        };
        let pack_sparse_active_loads = lanes_per_phase != 0
            && max_data_width_bytes <= 8
            && preflight.meta().access_kind == Some(pb::MemoryOpAccessKind::Load);
        if lanes_per_phase != 0 {
            // A whole-warp broadcast is still one service word even when the
            // operand is wide.  CUTLASS epilogues in particular issue a
            // vector load/store with every lane naming the same shared
            // fragment; forcing that through four quarter-warps would turn a
            // true broadcast into four fabricated wavefronts.  Detect this
            // source-level topology and retain the historical whole-warp
            // collapse for it.  Nonuniform lane spans use the phase model.
            let mut lane_spans: [Vec<(i128, i128)>; 32] = std::array::from_fn(|_| Vec::new());
            preflight.visit_addresses(|access| {
                let width = i128::from(access.data_width_bytes);
                for (lane, &address) in access.lane_vector.iter().enumerate() {
                    if address == lanes::LANE_INACTIVE_SENTINEL {
                        continue;
                    }
                    let first_word = i128::from(address).div_euclid(SHARED_BANK_WORD_BYTES_32X4);
                    let last_byte = i128::from(address) + width - 1;
                    let last_word = last_byte.div_euclid(SHARED_BANK_WORD_BYTES_32X4);
                    lane_spans[lane].push((first_word, last_word));
                }
            });
            let broadcast_signature = lane_spans.iter().find(|spans| !spans.is_empty());
            let all_active_lanes_broadcast = broadcast_signature.is_some()
                && lane_spans
                    .iter()
                    .filter(|spans| !spans.is_empty())
                    .all(|spans| Some(spans) == broadcast_signature);
            if all_active_lanes_broadcast {
                // Fall through to the existing whole-warp sort/dedup path.
                // This preserves the measured broadcast behavior while the
                // phased path below handles genuinely lane-varying accesses.
            } else {
                // Load requests packetize active lane spans by payload
                // capacity, not by physical lane-number windows. Stores
                // retain the established fixed half/quarter-warp packets;
                // their baseline is already exact on the real corpus.
                let active_lanes: Vec<usize> = lane_spans
                    .iter()
                    .enumerate()
                    .filter_map(|(lane, spans)| (!spans.is_empty()).then_some(lane))
                    .collect();
                let mut total_wavefronts = 0_u64;
                let phase_count = 32_usize / lanes_per_phase;
                for phase in 0..phase_count {
                    word_scratch.clear();
                    preflight.visit_addresses(|access| {
                        let width = i128::from(access.data_width_bytes);
                        for (lane, &address) in access.lane_vector.iter().enumerate() {
                            let in_phase = if pack_sparse_active_loads {
                                active_lanes
                                    .chunks(lanes_per_phase)
                                    .nth(phase)
                                    .is_some_and(|phase_lanes| phase_lanes.contains(&lane))
                            } else {
                                let lane_lo = phase * lanes_per_phase;
                                lane >= lane_lo && lane < lane_lo + lanes_per_phase
                            };
                            if !in_phase || address == lanes::LANE_INACTIVE_SENTINEL {
                                continue;
                            }
                            let first_word =
                                i128::from(address).div_euclid(SHARED_BANK_WORD_BYTES_32X4);
                            let last_byte = i128::from(address) + width - 1;
                            let last_word = last_byte.div_euclid(SHARED_BANK_WORD_BYTES_32X4);
                            for word in first_word..=last_word {
                                word_scratch.push(i64::try_from(word).unwrap_or_else(|_| {
                                    unreachable!("i64 byte address word escaped i64")
                                }));
                            }
                        }
                    });
                    if word_scratch.is_empty() {
                        continue;
                    }
                    word_scratch.sort_unstable();
                    word_scratch.dedup();
                    let mut bank_words = [0_u64; SHARED_BANK_COUNT_32X4 as usize];
                    let mut bank_overflow = None;
                    for &word in word_scratch.iter() {
                        let bank = word.rem_euclid(SHARED_BANK_COUNT_32X4) as usize;
                        let Some(next) = bank_words[bank].checked_add(1) else {
                            bank_overflow = Some(bank as u8);
                            break;
                        };
                        bank_words[bank] = next;
                    }
                    if let Some(bank) = bank_overflow {
                        return Err(SharedBankProjectionError::CountOverflow { bank });
                    }
                    total_wavefronts = total_wavefronts
                        .checked_add(bank_words.into_iter().max().unwrap_or(0))
                        .ok_or(SharedBankProjectionError::CountOverflow { bank: 0 })?;
                }
                // On SM120, a conflict-free full-warp `LDS.128` packet is
                // observed as two shared-data wavefronts rather than the
                // four quarter-warp scheduler phases implied by the byte
                // width.  Keep this correction on the decoupled observer
                // only; scheduler timing retains the legacy projection.
                let observed_wavefronts = if max_data_width_bytes == 16
                    && preflight.meta().access_kind == Some(pb::MemoryOpAccessKind::Load)
                    && active_lanes.len() == lanes::LANES_PER_WARP
                    && total_wavefronts == 4
                {
                    total_wavefronts / 2
                } else {
                    total_wavefronts
                };
                return u32::try_from(observed_wavefronts)
                    .map_err(|_| SharedBankProjectionError::CountOverflow { bank: 0 });
            }
        }

        // Shared LDS/STS broadcasts collapse repeated references to one
        // 32-bit word.  Keep this legacy path byte-for-byte identical for
        // ordinary shared accesses.
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
    let wavefronts = bank_words.into_iter().max().unwrap_or(0);
    Ok(
        u32::try_from(wavefronts)
            .unwrap_or_else(|_| unreachable!("bounded projection escaped u32")),
    )
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct LdgstsLaneTransfer {
    pub(crate) source_start: u64,
    pub(crate) source_end: u64,
    pub(crate) source_first_sector: i64,
    pub(crate) source_last_sector: i64,
    pub(crate) destination_first_word: i64,
    pub(crate) destination_last_word: i64,
    pub(crate) lane: u8,
    pub(crate) cohort: u8,
}

pub(crate) struct ProjectedLdgstsSourceCohorts {
    pub(crate) topology: LdgstsDestinationTopology,
    pub(crate) cohorts: Vec<LdgstsSourceCohort>,
    pub(crate) source_sector_cohort_masks: Vec<u32>,
}

pub(crate) const LDGSTS_DESTINATION_LINE_WORDS_128B: i64 = 128 / 4;

pub(crate) fn finish_ldgsts_destination_cohort(
    word_scratch: &mut Vec<i64>,
    active_transfers: u32,
    payload_bytes: u32,
    cohorts: &mut Vec<LdgstsSourceCohort>,
) -> Result<(), LdgstsDestinationProjectionError> {
    if active_transfers == 0 {
        debug_assert!(word_scratch.is_empty());
        debug_assert_eq!(payload_bytes, 0);
        return Ok(());
    }
    word_scratch.sort_unstable();
    word_scratch.dedup();
    let mut bank_words = [0_u32; SHARED_BANK_COUNT_32X4 as usize];
    let mut destination_lines = 0_u32;
    let mut previous_line = None;
    for &word in word_scratch.iter() {
        let line = word.div_euclid(LDGSTS_DESTINATION_LINE_WORDS_128B);
        if previous_line != Some(line) {
            destination_lines = destination_lines
                .checked_add(1)
                .ok_or(LdgstsDestinationProjectionError::DestinationLineFanoutOverflow)?;
            previous_line = Some(line);
        }
        let bank = word.rem_euclid(SHARED_BANK_COUNT_32X4) as usize;
        bank_words[bank] = bank_words[bank].checked_add(1).ok_or(
            LdgstsDestinationProjectionError::BankWavefrontCountOverflow { bank: bank as u8 },
        )?;
    }
    let bank_wavefronts = bank_words.into_iter().max().unwrap_or(0);
    let bypass_product = destination_lines.checked_mul(bank_wavefronts).ok_or(
        LdgstsDestinationProjectionError::BypassWavefrontProductOverflow {
            destination_lines,
            bank_wavefronts,
        },
    )?;
    let bypass_wavefronts = active_transfers.min(bypass_product);
    cohorts
        .try_reserve(1)
        .map_err(|_| LdgstsDestinationProjectionError::CapacityUnavailable { additional: 1 })?;
    cohorts.push(LdgstsSourceCohort {
        payload_bytes,
        destination_lines,
        bank_wavefronts,
        bypass_wavefronts,
    });
    word_scratch.clear();
    Ok(())
}

/// Compatibility wrapper for the established cache-all bank-wavefront scalar.
pub fn project_ldgsts_source_cohort_wavefronts_128b_32x4(
    preflight: PreflightedMemoryAccess<'_>,
    word_scratch: &mut Vec<i64>,
) -> Result<u32, LdgstsDestinationProjectionError> {
    Ok(
        project_ldgsts_source_cohort_topology_128b_32x4(preflight, word_scratch)?
            .cache_all_bank_wavefronts_total,
    )
}

pub(crate) fn known_enum<T: TryFrom<i32>>(
    raw: i32,
    field: &'static str,
    record: &impl Fn() -> String,
) -> Result<T, TraceError> {
    T::try_from(raw).map_err(|_| TraceError::UnknownEnumValue {
        record: record(),
        field,
        raw,
    })
}

pub(crate) fn decode_address_access(
    addr: &pb::Address,
    instruction_mask: u32,
    inst_width: i32,
    record: &impl Fn() -> String,
) -> Result<AddressAccess, TraceError> {
    let lane_mask = addr.lane_mask.unwrap_or(instruction_mask);
    if lane_mask & !instruction_mask != 0 {
        return Err(TraceError::AddressLaneMaskOutsideInstructionMask {
            record: record(),
            lane_mask,
            instruction_mask,
        });
    }
    // Match the Python gate's aggregate mask diagnosis before positional
    // reconstruction. A malformed list_all length has higher precedence there;
    // every other per-slot disagreement remains lane_position_mismatch.
    if addr.compression_format != lanes::FORMAT_LIST_ALL
        || addr.addrs.len() == lanes::LANES_PER_WARP
    {
        let real_count = if addr.compression_format == lanes::FORMAT_LIST_ALL {
            addr.addrs
                .iter()
                .filter(|&&slot| slot != lanes::LANE_INACTIVE_SENTINEL)
                .count()
        } else {
            addr.addrs.len()
        };
        if (lane_mask == 0 && real_count > 0)
            || (lane_mask != 0
                && addr.compression_format == lanes::FORMAT_LIST_ALL
                && real_count == 0)
        {
            return Err(TraceError::LaneMaskInconsistent { record: record() });
        }
    }

    let lane_vector =
        lanes::reconstruct_lane_vector(addr, lane_mask).map_err(|source| TraceError::Lane {
            record: record(),
            source,
        })?;
    let data_width_bytes = if addr.data_width > 0 {
        addr.data_width
    } else {
        inst_width
    };
    let address_space = match addr.address_space {
        None => None,
        Some(raw) => Some(known_enum::<pb::MemorySpace>(raw, "address_space", record)?),
    };
    Ok(AddressAccess {
        lane_vector,
        lane_mask,
        data_width_bytes,
        address_space,
        allocation_id: addr.allocation_id,
    })
}

/// Interpret one traced instruction per the wire contract, streaming each
/// decoded address record into `sink` — the zero-per-record-allocation form for
/// hot walk paths. Full preflight completes before the first callback, so an
/// error never exposes a partial callback prefix. The sink borrows a
/// stack-resident [`AddressAccess`]; copy it out (`Copy`) only if it must outlive
/// the call.
pub fn visit_memory_access<F>(
    inst: &pb::Instruction,
    record: impl Fn() -> String,
    sink: F,
) -> Result<Option<MemoryAccessMeta>, TraceError>
where
    F: FnMut(&AddressAccess),
{
    let Some(preflight) = preflight_memory_access(inst, record)? else {
        return Ok(None);
    };
    let meta = preflight.meta();
    preflight.visit_addresses(sink);
    Ok(Some(meta))
}

/// One backing allocation's device VA range.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AllocRange {
    pub(crate) base: u64,
    pub(crate) size: u64,
}

/// The `<trace_dir>/extra_info/allocations.json` registry — the same sidecar the
/// Python `trace_contract` validator reads (an array of
/// `{allocation_id, base_address (hex string), size_bytes, ...}`). Used to
/// enforce the address-carrying contract's allocation identity in the Rust
/// reader, at parity with the Python gate.
#[derive(Debug, Clone, Default)]
pub struct AllocationRegistry {
    pub(crate) by_id: BTreeMap<u32, AllocRange>,
}

/// The EXACT six-field allocation-sidecar entry, at parity with the Python
/// validator's shape gate (`the trace contract`: all six fields required,
/// no extras). `deny_unknown_fields` + all-required non-Option fields make serde
/// reject a missing field, an unexpected key, a negative/boolean `allocation_id`
/// or `size_bytes`, and a non-bool `alive`. The explicit VALUE checks (positive
/// id/size, 0x-prefixed base, duplicate id) run in `from_str` after.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawAllocation {
    pub(crate) allocation_id: u32,
    // context_id / device_id / alive are producer metadata not consumed by the
    // range check (Python does not gate on `alive` — a freed-at-trace-end
    // allocation was valid during the trace), but each MUST be present and
    // correctly typed for exact-shape parity. `device_id` may be 0 or negative.
    #[allow(dead_code)]
    pub(crate) context_id: i64,
    #[allow(dead_code)]
    pub(crate) device_id: i64,
    pub(crate) base_address: String,
    pub(crate) size_bytes: u64,
    #[allow(dead_code)]
    pub(crate) alive: bool,
}

pub(crate) fn parse_hex_u64(s: &str) -> Option<u64> {
    // Canonical cross-runtime grammar: an exact 0x/0X prefix followed by one or
    // more ASCII hex digits. In particular, Python-style digit separators are
    // not wire syntax.
    let digits = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X"))?;
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    u64::from_str_radix(digits, 16).ok()
}

impl AllocationRegistry {
    /// Return an address's byte offset within a validated allocation.
    ///
    /// This exposes allocation-relative identity without exposing or inventing
    /// a physical address. `None` means the id is unknown, the subtraction
    /// underflows, or the address is at/after the allocation end.
    pub fn relative_byte_offset(&self, allocation_id: u32, address: u64) -> Option<u64> {
        let range = self.by_id.get(&allocation_id)?;
        let offset = address.checked_sub(range.base)?;
        (offset < range.size).then_some(offset)
    }
}

impl AllocationRegistry {
    /// Return a compact 32-byte sector's offset within a sector-aligned
    /// allocation.
    ///
    /// Global CUDA allocations are sector aligned in admitted traces. A
    /// malformed/non-aligned synthetic registry fails closed rather than
    /// silently rounding its base.
    pub fn relative_sector_offset(&self, allocation_id: u32, absolute_sector: i64) -> Option<u64> {
        let range = self.by_id.get(&allocation_id)?;
        if range.base & 31 != 0 {
            return None;
        }
        let sector = u64::try_from(absolute_sector).ok()?;
        let base_sector = range.base >> 5;
        let offset = sector.checked_sub(base_sector)?;
        let byte_offset = offset.checked_mul(32)?;
        (byte_offset < range.size).then_some(offset)
    }
}

impl AllocationRegistry {
    /// Load from `<root>/extra_info/allocations.json`. `Ok(None)` if the file is
    /// absent (legacy traces, or traces with no GLOBAL/GENERIC active-lane record
    /// requiring identity). `Err(AllocationsSidecarMalformed)` on bad JSON, a
    /// non-hex `base_address`, or a duplicate `allocation_id`.
    pub fn load(root: &Path) -> Result<Option<Self>, TraceError> {
        let path = root.join("extra_info").join("allocations.json");
        let reader = AnchoredDir::open(root)?;
        let raw = match reader
            .read_optional_bounded_regular(
                Path::new("extra_info/allocations.json"),
                ALLOCATIONS_JSON_MAX_BYTES,
            )
            .map_err(|source| map_snapshot_read_error("allocations_json", source))?
        {
            Some(file) => file.into_raw(),
            None => return Ok(None),
        };
        Self::parse(&path, &raw).map(Some)
    }
}

impl AllocationRegistry {
    pub(crate) fn parse(path: &Path, raw: &[u8]) -> Result<Self, TraceError> {
        Self::parse_with_limits(
            path,
            raw,
            ALLOCATIONS_JSON_MAX_BYTES,
            ALLOCATIONS_JSON_MAX_NODES,
            ALLOCATION_ENTRY_MAX_COUNT,
        )
    }
}

impl AllocationRegistry {
    pub(crate) fn parse_with_limits(
        path: &Path,
        raw: &[u8],
        maximum_bytes: u64,
        maximum_nodes: u64,
        maximum_entries: u64,
    ) -> Result<Self, TraceError> {
        sidecar_byte_limit(path, "allocations_json", raw, maximum_bytes)?;
        match preflight_json_shape(raw, maximum_nodes, Some(maximum_entries)) {
            Ok(_) => {}
            Err(JsonPreflightError::Malformed(message)) => {
                return Err(TraceError::AllocationsSidecarMalformed {
                    path: path.to_path_buf(),
                    message,
                });
            }
            Err(JsonPreflightError::ResourceLimit(error)) => {
                return Err(map_json_preflight_limit(path, "allocations_json", error));
            }
        }
        let entries: Vec<RawAllocation> =
            serde_json::from_slice(raw).map_err(|e| TraceError::AllocationsSidecarMalformed {
                path: path.to_path_buf(),
                message: e.to_string(),
            })?;
        let mut by_id = BTreeMap::new();
        for e in entries {
            // Python parity: allocation ids start at 1 (0 is the unresolved
            // sentinel) and every allocation has a positive size.
            if e.allocation_id == 0 {
                return Err(TraceError::AllocationsSidecarMalformed {
                    path: path.to_path_buf(),
                    message: "allocation_id must be positive (0 is the unresolved sentinel)"
                        .to_string(),
                });
            }
            if e.size_bytes == 0 {
                return Err(TraceError::AllocationsSidecarMalformed {
                    path: path.to_path_buf(),
                    message: format!("allocation_id {} has size_bytes == 0", e.allocation_id),
                });
            }
            let base = parse_hex_u64(&e.base_address).ok_or_else(|| {
                TraceError::AllocationsSidecarMalformed {
                    path: path.to_path_buf(),
                    message: format!(
                        "allocation_id {} has a non-0x-hex base_address {:?}",
                        e.allocation_id, e.base_address
                    ),
                }
            })?;
            if by_id
                .insert(
                    e.allocation_id,
                    AllocRange {
                        base,
                        size: e.size_bytes,
                    },
                )
                .is_some()
            {
                return Err(TraceError::AllocationIdDuplicate {
                    path: path.to_path_buf(),
                    allocation_id: e.allocation_id,
                });
            }
        }
        Ok(Self { by_id })
    }
}

/// Apply allocation identity to an instruction whose complete wire structure was
/// already validated. This split lets replay enforce its producer-shape route
/// before identity without rerunning fallible structural preflight or allocating an
/// intermediate decoded-payload vector; immutable address records are revisited by
/// the allocation-free visitor.
pub fn check_preflighted_allocation_identity(
    preflight: PreflightedMemoryAccess<'_>,
    registry: Option<&AllocationRegistry>,
    record: impl Fn() -> String,
) -> Result<(), TraceError> {
    let meta = preflight.meta();
    let inst_class = Some(meta.op_class);
    for addr in &preflight.instruction().addresses {
        let address_space = match addr.address_space {
            None => None,
            Some(raw) => Some(known_enum::<pb::MemorySpace>(
                raw,
                "address_space",
                &record,
            )?),
        };
        if !record_requires_identity(address_space, inst_class) {
            continue;
        }

        let access = decode_address_access(addr, meta.and_mask, preflight.inst_width, &record)?;
        let allocation_id = addr.allocation_id.unwrap_or(0);
        // Module/library-owned globals need not appear in cuMemAlloc hooks.
        // Honor the producer's explicit disposition and keep the recorded VA;
        // it is not proof of symbol bounds or lifetime, and must never become
        // a fabricated allocator id. Unannotated or unresolved_global records
        // still fail closed. No scored NCU information enters this decision.
        let module_global = match addr.allocation_identity_status.as_deref() {
            None | Some("resolved") => false,
            Some("unresolved_module_global") => true,
            Some(status) => {
                return Err(TraceError::AllocationIdentityStatusInvalid {
                    record: record(),
                    status: status.to_owned(),
                });
            }
        };
        if module_global {
            if allocation_id != 0 {
                return Err(TraceError::AllocationIdentityStatusConflict {
                    record: record(),
                    allocation_id,
                });
            }
            if access.lane_mask != 0 {
                continue;
            }
        }
        if allocation_id == 0 {
            if access.lane_mask != 0 {
                return Err(TraceError::AllocationIdMissing { record: record() });
            }
            continue;
        }
        let Some(registry) = registry else {
            return Err(TraceError::AllocationsSidecarMissing { record: record() });
        };
        let Some(range) = registry.by_id.get(&allocation_id) else {
            return Err(TraceError::AllocationIdUnknown {
                record: record(),
                allocation_id,
            });
        };
        if access.lane_mask == 0 {
            continue;
        }

        let base = u128::from(range.base);
        let end = base + u128::from(range.size);
        for (lane, &slot) in access.lane_vector.iter().enumerate() {
            if slot == lanes::LANE_INACTIVE_SENTINEL {
                continue;
            }
            let lane_addr = slot as u64;
            let lane_addr_wide = u128::from(lane_addr);
            if !(base <= lane_addr_wide && lane_addr_wide < end) {
                return Err(TraceError::LaneOutsideAllocationRange {
                    record: record(),
                    lane,
                    lane_addr,
                    allocation_id,
                    base: range.base,
                    end,
                });
            }
        }
    }
    Ok(())
}
