//! Trace sampling provenance and sidecar contracts; report-only.

use super::*;

/// Method-specific provenance retained from a supported v2 sampling sidecar.
/// Unsupported schema methods never construct this enum.
#[cfg(feature = "campaign")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceSamplingMethod {
    /// Sealed lossless-compression provenance. Replay must independently observe
    /// full launch coverage; this variant never sanctions incompleteness.
    LosslessCompressedFull(LosslessFullProvenance),
    /// A deeply validated fixed-point subset envelope. The selected membership
    /// and seals remain bound as one variant instead of a boolean plus optional
    /// fields that callers could combine inconsistently.
    WaveSegmentStratifiedSubset(WaveSegmentSubsetProvenance),
}

#[cfg(feature = "campaign")]
impl TraceSamplingMethod {
    pub fn selected_block_ids(&self) -> Option<&BTreeSet<i64>> {
        match self {
            Self::LosslessCompressedFull(_) => None,
            Self::WaveSegmentStratifiedSubset(provenance) => Some(&provenance.selected_block_ids),
        }
    }

    pub fn sanctions_incomplete(&self) -> bool {
        matches!(self, Self::WaveSegmentStratifiedSubset(_))
    }
}

/// Hash identity retained from a sealed lossless-compression sidecar. This
/// declaration remains distinct from the launch-bound full-coverage witness
/// that replay establishes from the decoded block roster.
#[cfg(feature = "campaign")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LosslessFullProvenance {
    pub(super) selected_block_ids_sha256: String,
    pub(super) sidecar_sha256: String,
}

#[cfg(feature = "campaign")]
impl LosslessFullProvenance {
    pub fn selected_block_ids_sha256(&self) -> &str {
        &self.selected_block_ids_sha256
    }

    pub fn sidecar_sha256(&self) -> &str {
        &self.sidecar_sha256
    }
}

/// Identity retained from a Python-canonical, deeply validated subset sidecar.
/// Payload bytes are additionally verified by [`TraceDir::open`] before callers
/// can obtain this provenance through [`TraceDir::sampling_sidecar`].
#[cfg(feature = "campaign")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaveSegmentSubsetProvenance {
    pub(super) selected_block_ids: BTreeSet<i64>,
    pub(super) selected_block_ids_sha256: String,
    pub(super) sidecar_sha256: String,
}

#[cfg(feature = "campaign")]
impl WaveSegmentSubsetProvenance {
    pub fn selected_block_ids(&self) -> &BTreeSet<i64> {
        &self.selected_block_ids
    }

    pub fn selected_block_ids_sha256(&self) -> &str {
        &self.selected_block_ids_sha256
    }

    pub fn sidecar_sha256(&self) -> &str {
        &self.sidecar_sha256
    }
}

/// Reader-side view of a supported `<root>/trace_sampling.json` v2 sidecar.
/// Fields are private so method-specific invariants can only be constructed by
/// the closed parser. A sealed lossless declaration is not itself a
/// full-coverage witness. Launch-bound full versus sampled coverage is
/// established later by replay after observing block membership and record counts.
#[cfg(feature = "campaign")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceSamplingSidecar {
    pub(super) cell_id: String,
    pub(super) kernel_mangled_name: String,
    pub(super) kernel_args_hash: String,
    pub(super) original_grid: (u32, u32, u32),
    pub(super) original_block: (u32, u32, u32),
    pub(super) captured_grid: (u32, u32, u32),
    pub(super) captured_block: (u32, u32, u32),
    pub(super) expected_full_records: u64,
    pub(super) captured_records: u64,
    pub(super) decompressed_payload_sha256: String,
    pub(super) payload_file_manifest: BTreeMap<String, PayloadFileManifestEntry>,
    pub(super) raw_payload_bytes: u64,
    pub(super) per_sm_slot_map: ValidatedPerSmSlotMap,
    pub(super) method: TraceSamplingMethod,
}

#[cfg(feature = "campaign")]
impl TraceSamplingSidecar {
    pub fn cell_id(&self) -> &str {
        &self.cell_id
    }

    pub fn kernel_mangled_name(&self) -> &str {
        &self.kernel_mangled_name
    }

    pub fn kernel_args_hash(&self) -> &str {
        &self.kernel_args_hash
    }

    pub fn original_grid(&self) -> (u32, u32, u32) {
        self.original_grid
    }

    pub fn original_block(&self) -> (u32, u32, u32) {
        self.original_block
    }

    pub fn captured_grid(&self) -> (u32, u32, u32) {
        self.captured_grid
    }

    pub fn captured_block(&self) -> (u32, u32, u32) {
        self.captured_block
    }

    pub fn expected_full_records(&self) -> u64 {
        self.expected_full_records
    }

    pub fn captured_records(&self) -> u64 {
        self.captured_records
    }

    pub fn decompressed_payload_sha256(&self) -> &str {
        &self.decompressed_payload_sha256
    }

    pub fn raw_payload_bytes(&self) -> u64 {
        self.raw_payload_bytes
    }

    /// Iterate the sidecar's validated full-launch spatial placement in launch
    /// index order. The map says where each block was observed, not which blocks
    /// were resident concurrently.
    pub fn declared_block_placements(&self) -> impl ExactSizeIterator<Item = (u64, u64)> + '_ {
        self.per_sm_slot_map
            .sm_by_launch_index
            .iter()
            .copied()
            .enumerate()
            .map(|(launch_index, sm_id)| (launch_index as u64, sm_id))
    }

    pub fn declared_sm_for_launch_index(&self, launch_index: i64) -> Option<u64> {
        usize::try_from(launch_index)
            .ok()
            .and_then(|index| self.per_sm_slot_map.sm_by_launch_index.get(index))
            .copied()
    }

    pub fn method(&self) -> &TraceSamplingMethod {
        &self.method
    }
}

#[cfg(feature = "campaign")]
impl TraceSamplingSidecar {
    /// Parse `<root>/trace_sampling.json` if present (`Ok(None)` if absent).
    /// This internal stage validates and retains the JSON envelope; callers only
    /// receive it through [`TraceDir::sampling_sidecar`] after `TraceDir::open`
    /// has also bound the decompressed threadblock bytes.

    pub(super) fn parse(path: &Path, raw: &[u8]) -> Result<Self, TraceError> {
        Self::parse_with_limits(
            path,
            raw,
            TRACE_SAMPLING_JSON_MAX_BYTES,
            TRACE_SAMPLING_JSON_MAX_NODES,
        )
    }

    pub(super) fn parse_with_limits(
        path: &Path,
        raw: &[u8],
        maximum_bytes: u64,
        maximum_nodes: u64,
    ) -> Result<Self, TraceError> {
        sidecar_byte_limit(path, "trace_sampling_json", raw, maximum_bytes)?;
        match preflight_json_shape(raw, maximum_nodes, None) {
            Ok(_) => {}
            Err(JsonPreflightError::Malformed(message)) => {
                return Err(TraceError::SamplingSidecarMalformed {
                    path: path.to_path_buf(),
                    message,
                });
            }
            Err(JsonPreflightError::ResourceLimit(error)) => {
                return Err(map_json_preflight_limit(path, "trace_sampling_json", error));
            }
        }
        let parsed: serde_json::Value =
            serde_json::from_slice(raw).map_err(|e| TraceError::SamplingSidecarMalformed {
                path: path.to_path_buf(),
                message: e.to_string(),
            })?;
        let malformed = |message: String| TraceError::SamplingSidecarMalformed {
            path: path.to_path_buf(),
            message,
        };
        macro_rules! get {
            ($object:expr, $field:expr, $context:expr) => {
                required($object, $field, $context).map_err(&malformed)?
            };
        }
        let root = object(&parsed, "trace_sampling.json root").map_err(&malformed)?;
        pub(super) const TOP_LEVEL_FIELDS: &[&str] = &[
            "schema_version",
            "cell_id",
            "kernel",
            "method",
            "reason",
            "shape",
            "gates",
            "wave_model",
            "segments",
            "selected_segment_ids",
            "certainty_segments",
            "force_include_blocks",
            "per_sm_slot_map",
            "strata",
            "selection_reconciliation",
            "selection_budget",
            "selection_auxiliary",
            "rng",
            "estimator",
            "label_pairing",
            "tracer",
            "bytes",
            "records",
            "hashes",
        ];
        pub(super) const BASE_REQUIRED: &[&str] = &[
            "schema_version",
            "cell_id",
            "kernel",
            "method",
            "reason",
            "shape",
            "gates",
            "records",
            "tracer",
            "hashes",
        ];
        exact_keys(
            root,
            BASE_REQUIRED,
            TOP_LEVEL_FIELDS,
            "trace_sampling.json root",
        )
        .map_err(&malformed)?;
        if get!(root, "schema_version", "trace_sampling.json root").as_u64() != Some(2) {
            return Err(malformed("schema_version must be 2".to_string()));
        }
        let method = get!(root, "method", "trace_sampling.json root")
            .as_str()
            .ok_or_else(|| malformed("method must be a string".to_string()))?;

        let cell_id = get!(root, "cell_id", "trace_sampling.json root")
            .as_str()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| malformed("cell_id must be a non-empty string".to_string()))?
            .to_string();
        let kernel = object(get!(root, "kernel", "trace_sampling.json root"), "kernel")
            .map_err(&malformed)?;
        exact_keys(
            kernel,
            &["mangled_name", "args_hash"],
            &["mangled_name", "args_hash"],
            "kernel",
        )
        .map_err(&malformed)?;
        let kernel_mangled_name = get!(kernel, "mangled_name", "kernel")
            .as_str()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| malformed("kernel.mangled_name must be nonempty".to_string()))?
            .to_string();
        let kernel_args_hash = get!(kernel, "args_hash", "kernel")
            .as_str()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| malformed("kernel.args_hash must be nonempty".to_string()))?
            .to_string();
        get!(root, "reason", "trace_sampling.json root")
            .as_str()
            .ok_or_else(|| malformed("reason must be a string".to_string()))?;
        object(get!(root, "gates", "trace_sampling.json root"), "gates").map_err(&malformed)?;
        object(get!(root, "tracer", "trace_sampling.json root"), "tracer").map_err(&malformed)?;

        let shape =
            object(get!(root, "shape", "trace_sampling.json root"), "shape").map_err(&malformed)?;
        exact_keys(
            shape,
            &[
                "original_grid",
                "original_block",
                "captured_grid",
                "captured_block",
            ],
            &[
                "original_grid",
                "original_block",
                "captured_grid",
                "captured_block",
                "reduced_shape_variant",
            ],
            "shape",
        )
        .map_err(&malformed)?;
        let original_grid =
            parse_dim3(get!(shape, "original_grid", "shape"), "shape.original_grid")
                .map_err(&malformed)?;
        let captured_grid =
            parse_dim3(get!(shape, "captured_grid", "shape"), "shape.captured_grid")
                .map_err(&malformed)?;
        let original_block = parse_dim3(
            get!(shape, "original_block", "shape"),
            "shape.original_block",
        )
        .map_err(&malformed)?;
        let captured_block = parse_dim3(
            get!(shape, "captured_block", "shape"),
            "shape.captured_block",
        )
        .map_err(&malformed)?;

        let records = object(get!(root, "records", "trace_sampling.json root"), "records")
            .map_err(&malformed)?;
        exact_keys(
            records,
            &["expected_full", "captured"],
            &["expected_full", "captured"],
            "records",
        )
        .map_err(&malformed)?;
        let expected_full_records = get!(records, "expected_full", "records")
            .as_u64()
            .ok_or_else(|| malformed("records.expected_full must be a u64".to_string()))?;
        let captured_records = get!(records, "captured", "records")
            .as_u64()
            .ok_or_else(|| malformed("records.captured must be a u64".to_string()))?;

        let hashes = object(get!(root, "hashes", "trace_sampling.json root"), "hashes")
            .map_err(&malformed)?;
        exact_keys(
            hashes,
            &[
                "selected_block_ids_sha256",
                "decompressed_payload_sha256",
                "per_block_raw_sha256_manifest",
                "sidecar_sha256",
            ],
            &[
                "selected_block_ids_sha256",
                "decompressed_payload_sha256",
                "per_block_raw_sha256_manifest",
                "sidecar_sha256",
            ],
            "hashes",
        )
        .map_err(&malformed)?;
        for field in [
            "selected_block_ids_sha256",
            "decompressed_payload_sha256",
            "sidecar_sha256",
        ] {
            let value = get!(hashes, field, "hashes")
                .as_str()
                .ok_or_else(|| malformed(format!("hashes.{field} must be a string")))?;
            if !is_lower_hex_sha256(value) {
                return Err(malformed(format!(
                    "hashes.{field} must be lowercase SHA-256"
                )));
            }
        }
        let selected_block_ids_sha256 = get!(hashes, "selected_block_ids_sha256", "hashes")
            .as_str()
            .expect("hash string validated above")
            .to_string();
        let decompressed_payload_sha256 = get!(hashes, "decompressed_payload_sha256", "hashes")
            .as_str()
            .expect("hash string validated above")
            .to_string();
        let payload_file_manifest = parse_payload_file_manifest(
            get!(hashes, "per_block_raw_sha256_manifest", "hashes")
                .as_str()
                .ok_or_else(|| {
                    malformed("hashes.per_block_raw_sha256_manifest must be a string".to_string())
                })?,
        )
        .map_err(&malformed)?;
        let sidecar_sha256 = get!(hashes, "sidecar_sha256", "hashes")
            .as_str()
            .expect("hash string validated above")
            .to_string();
        let expected_seal = canonical_sidecar_sha256(&parsed).map_err(&malformed)?;
        if sidecar_sha256 != expected_seal {
            return Err(malformed(
                "hashes.sidecar_sha256 does not match Python-canonical sidecar".to_string(),
            ));
        }

        let (method, raw_payload_bytes, per_sm_slot_map) = match method {
            "wave_segment_stratified_subset" => {
                pub(super) const SUBSET_REQUIRED: &[&str] = &[
                    "wave_model",
                    "segments",
                    "selected_segment_ids",
                    "certainty_segments",
                    "force_include_blocks",
                    "per_sm_slot_map",
                    "strata",
                    "selection_reconciliation",
                    "selection_budget",
                    "selection_auxiliary",
                    "rng",
                    "estimator",
                    "label_pairing",
                    "bytes",
                ];
                exact_keys(root, SUBSET_REQUIRED, TOP_LEVEL_FIELDS, "sampled subset")
                    .map_err(&malformed)?;
                for field in [
                    "wave_model",
                    "selection_budget",
                    "selection_auxiliary",
                    "rng",
                    "estimator",
                    "bytes",
                ] {
                    object(get!(root, field, "sampled subset"), field).map_err(&malformed)?;
                }
                for field in ["force_include_blocks", "per_sm_slot_map", "strata"] {
                    get!(root, field, "sampled subset")
                        .as_array()
                        .ok_or_else(|| malformed(format!("{field} must be an array")))?;
                }
                get!(root, "label_pairing", "sampled subset")
                    .as_str()
                    .ok_or_else(|| malformed("label_pairing must be a string".to_string()))?;
                if captured_grid != original_grid {
                    return Err(malformed(
                        "shape.captured_grid must equal shape.original_grid for a subset"
                            .to_string(),
                    ));
                }

                let reconciliation = object(
                    get!(root, "selection_reconciliation", "sampled subset"),
                    "selection_reconciliation",
                )
                .map_err(&malformed)?;
                exact_keys(
                    reconciliation,
                    &["status", "attempts", "max_attempts", "capture_stats_sha256"],
                    &["status", "attempts", "max_attempts", "capture_stats_sha256"],
                    "selection_reconciliation",
                )
                .map_err(&malformed)?;
                let status = get!(reconciliation, "status", "selection_reconciliation").as_str();
                let attempts =
                    get!(reconciliation, "attempts", "selection_reconciliation").as_u64();
                let max_attempts =
                    get!(reconciliation, "max_attempts", "selection_reconciliation").as_u64();
                let capture_hash = get!(
                    reconciliation,
                    "capture_stats_sha256",
                    "selection_reconciliation"
                )
                .as_str();
                if status != Some("fixed_point")
                    || attempts.is_none_or(|value| value == 0)
                    || max_attempts.is_none_or(|value| value == 0)
                    || attempts > max_attempts
                    || capture_hash.is_none_or(|value| !is_lower_hex_sha256(value))
                {
                    return Err(malformed(
                        "invalid fixed-point reconciliation envelope".to_string(),
                    ));
                }

                let selected_segments = parse_unique_ids(
                    get!(root, "selected_segment_ids", "sampled subset"),
                    "selected_segment_ids",
                    false,
                )
                .map_err(&malformed)?;
                let certainty_segments = parse_unique_ids(
                    get!(root, "certainty_segments", "sampled subset"),
                    "certainty_segments",
                    false,
                )
                .map_err(&malformed)?;
                if !certainty_segments.is_subset(&selected_segments) {
                    return Err(malformed(
                        "certainty_segments must be a subset of selected_segment_ids".to_string(),
                    ));
                }

                let segment_values = get!(root, "segments", "sampled subset")
                    .as_array()
                    .ok_or_else(|| malformed("segments must be an array".to_string()))?;
                let mut segment_ids = BTreeSet::new();
                let mut segment_blocks_by_id = BTreeMap::new();
                let mut blocks = BTreeSet::new();
                for (index, segment_value) in segment_values.iter().enumerate() {
                    let context = format!("segments[{index}]");
                    let segment = object(segment_value, &context).map_err(&malformed)?;
                    exact_keys(
                        segment,
                        &["id", "block_ids"],
                        &["id", "block_ids"],
                        &context,
                    )
                    .map_err(&malformed)?;
                    let segment_id = get!(segment, "id", &context).as_u64().ok_or_else(|| {
                        malformed(format!("{context}.id must be a nonnegative integer"))
                    })?;
                    if !segment_ids.insert(segment_id) {
                        return Err(malformed(format!(
                            "segments contains duplicate id {segment_id}"
                        )));
                    }
                    let blocks_for_segment = parse_unique_ids(
                        get!(segment, "block_ids", &context),
                        &format!("{context}.block_ids"),
                        false,
                    )
                    .map_err(&malformed)?;
                    for block_id in &blocks_for_segment {
                        if *block_id > i64::MAX as u64 {
                            return Err(malformed(format!(
                                "selected block id {block_id} exceeds the replay index range"
                            )));
                        }
                        if !blocks.insert(*block_id) {
                            return Err(malformed(format!(
                                "selected block id {block_id} occurs in multiple segments"
                            )));
                        }
                    }
                    segment_blocks_by_id.insert(segment_id, blocks_for_segment);
                }
                if segment_ids != selected_segments {
                    return Err(malformed(
                        "segments ids must exactly equal selected_segment_ids".to_string(),
                    ));
                }

                let per_sm_slot_map = validate_subset_producer_contract(
                    root,
                    original_grid,
                    original_block,
                    captured_block,
                    &selected_segments,
                    &certainty_segments,
                    &segment_blocks_by_id,
                )
                .map_err(&malformed)?;

                let sorted_blocks: Vec<u64> = blocks.iter().copied().collect();
                let block_json = serde_json::to_string(&sorted_blocks).map_err(|error| {
                    malformed(format!(
                        "could not canonicalize selected block ids: {error}"
                    ))
                })?;
                let expected_block_hash = sha256_hex(block_json.as_bytes());
                if selected_block_ids_sha256 != expected_block_hash {
                    return Err(malformed(
                        "selected_block_ids_sha256 does not match segment membership".to_string(),
                    ));
                }

                (
                    TraceSamplingMethod::WaveSegmentStratifiedSubset(WaveSegmentSubsetProvenance {
                        selected_block_ids: blocks.into_iter().map(|value| value as i64).collect(),
                        selected_block_ids_sha256,
                        sidecar_sha256,
                    }),
                    unsigned(
                        required(
                            object(get!(root, "bytes", "sampled subset"), "bytes")
                                .map_err(&malformed)?,
                            "raw",
                            "bytes",
                        )
                        .map_err(&malformed)?,
                        "bytes.raw",
                    )
                    .map_err(&malformed)?,
                    per_sm_slot_map,
                )
            }
            // A lossless-compression declaration is sealed and payload-bound,
            // but replay must still observe complete launch coverage itself.
            "lossless_compressed_full" => {
                let (raw, per_sm_slot_map) = validate_lossless_producer_contract(
                    root,
                    original_grid,
                    original_block,
                    captured_grid,
                    captured_block,
                    &selected_block_ids_sha256,
                )
                .map_err(&malformed)?;
                (
                    TraceSamplingMethod::LosslessCompressedFull(LosslessFullProvenance {
                        selected_block_ids_sha256,
                        sidecar_sha256,
                    }),
                    raw,
                    per_sm_slot_map,
                )
            }
            _ => {
                return Err(malformed(
                    "method must be a sampled subset or lossless compressed full".to_string(),
                ));
            }
        };
        let manifest_bytes = payload_file_manifest
            .values()
            .try_fold(0_u64, |total, entry| {
                total.checked_add(entry.size_bytes).ok_or_else(|| {
                    malformed("per_block_raw_sha256_manifest byte total overflows u64".to_string())
                })
            })?;
        if manifest_bytes != raw_payload_bytes {
            return Err(malformed(format!(
                "per_block_raw_sha256_manifest byte total {manifest_bytes} disagrees with bytes.raw {raw_payload_bytes}"
            )));
        }
        Ok(TraceSamplingSidecar {
            cell_id,
            kernel_mangled_name,
            kernel_args_hash,
            original_grid,
            original_block,
            captured_grid,
            captured_block,
            expected_full_records,
            captured_records,
            decompressed_payload_sha256,
            payload_file_manifest,
            raw_payload_bytes,
            per_sm_slot_map,
            method,
        })
    }
}
