//! Trace artifact snapshots and JSON-shape limits; report-only.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPayloadContract {
    LegacyExclusion,
    AddressCarryingV1,
    AddressCarryingSharedMemorySplitV2,
    AddressCarryingLdgstsDestinationV3,
}

impl MemoryPayloadContract {
    pub(super) const fn carries_atomic_addresses(self) -> bool {
        !matches!(self, Self::LegacyExclusion)
    }

    pub(super) const fn carries_shared_memory_split(self) -> bool {
        matches!(
            self,
            Self::AddressCarryingSharedMemorySplitV2 | Self::AddressCarryingLdgstsDestinationV3
        )
    }

    pub(super) const fn carries_ldgsts_destination(self) -> bool {
        matches!(self, Self::AddressCarryingLdgstsDestinationV3)
    }
}

/// Raw-byte identity of one snapshotted trace artifact.
#[cfg(feature = "campaign")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArtifactDigestIdentity {
    pub(super) size_bytes: u64,
    pub(super) sha256: Sha256Digest,
}

#[cfg(feature = "campaign")]
impl ArtifactDigestIdentity {
    pub(super) fn from_bytes(bytes: &[u8], path: &Path) -> Result<Self, TraceError> {
        let size_bytes = u64::try_from(bytes.len()).map_err(|_| {
            TraceError::ThreadblockPayloadSnapshotInvalid {
                root: path.to_path_buf(),
                message: "artifact byte length exceeds u64".to_owned(),
            }
        })?;
        Ok(Self {
            size_bytes,
            sha256: Sha256Digest::of(bytes),
        })
    }

    pub fn size_bytes(self) -> u64 {
        self.size_bytes
    }

    pub fn sha256(self) -> Sha256Digest {
        self.sha256
    }
}

/// Deterministic identity of every threadblock payload file in a trace snapshot.
#[cfg(feature = "campaign")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThreadblockPayloadIdentity {
    pub(super) file_count: u64,
    pub(super) raw_bytes: u64,
    pub(super) sorted_path_manifest_sha256: Sha256Digest,
    pub(super) sorted_path_stream_sha256: Sha256Digest,
}

#[cfg(feature = "campaign")]
impl ThreadblockPayloadIdentity {
    pub fn file_count(&self) -> u64 {
        self.file_count
    }

    pub fn raw_bytes(&self) -> u64 {
        self.raw_bytes
    }

    pub fn sorted_path_manifest_sha256(&self) -> Sha256Digest {
        self.sorted_path_manifest_sha256
    }

    pub fn sorted_path_stream_sha256(&self) -> Sha256Digest {
        self.sorted_path_stream_sha256
    }
}

/// Raw identities captured before any trace artifact is decoded.
#[cfg(feature = "campaign")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceArtifactIdentity {
    pub(super) dynamic_trace_pb: ArtifactDigestIdentity,
    pub(super) allocations_json: Option<ArtifactDigestIdentity>,
    pub(super) trace_sampling_json: Option<ArtifactDigestIdentity>,
    pub(super) stats_csv: Option<ArtifactDigestIdentity>,
    pub(super) threadblock_payload: ThreadblockPayloadIdentity,
}

#[cfg(feature = "campaign")]
impl TraceArtifactIdentity {
    pub fn dynamic_trace_pb(&self) -> ArtifactDigestIdentity {
        self.dynamic_trace_pb
    }

    pub fn allocations_json(&self) -> Option<ArtifactDigestIdentity> {
        self.allocations_json
    }

    pub fn trace_sampling_json(&self) -> Option<ArtifactDigestIdentity> {
        self.trace_sampling_json
    }

    pub fn stats_csv(&self) -> Option<ArtifactDigestIdentity> {
        self.stats_csv
    }

    pub fn threadblock_payload(&self) -> &ThreadblockPayloadIdentity {
        &self.threadblock_payload
    }
}

/// Declared byte/cardinality bounds used before a manifest snapshot reads data.
#[cfg(feature = "campaign")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceSnapshotLimits {
    pub(super) dynamic_trace_pb_size: u64,
    pub(super) allocations_json_size: Option<u64>,
    pub(super) trace_sampling_json_size: Option<u64>,
    pub(super) stats_csv_size: Option<u64>,
    pub(super) threadblock_file_count: u64,
    pub(super) threadblock_raw_bytes: u64,
}

#[cfg(feature = "campaign")]
impl TraceSnapshotLimits {
    pub fn new(
        dynamic_trace_pb_size: u64,
        allocations_json_size: Option<u64>,
        trace_sampling_json_size: Option<u64>,
        stats_csv_size: Option<u64>,
        threadblock_file_count: u64,
        threadblock_raw_bytes: u64,
    ) -> Self {
        Self {
            dynamic_trace_pb_size,
            allocations_json_size,
            trace_sampling_json_size,
            stats_csv_size,
            threadblock_file_count,
            threadblock_raw_bytes,
        }
    }
}

/// One-read snapshot used by manifest admission. Artifact identities are
/// available before parsing, and [`TraceDir::from_snapshot`] consumes these exact
/// bytes so a hash-then-reopen substitution cannot change replay inputs.
#[cfg(feature = "campaign")]
pub struct TraceDirSnapshot {
    pub(super) root: PathBuf,
    pub(super) dynamic_trace_pb: Vec<u8>,
    pub(super) allocations_json: Option<Vec<u8>>,
    pub(super) trace_sampling_json: Option<Vec<u8>>,
    pub(super) stats_csv: Option<Vec<u8>>,
    pub(super) threadblock_payload: VerifiedSamplingPayload,
    pub(super) identity: TraceArtifactIdentity,
    pub(super) file_identities: BTreeMap<FileIdentity, (PathBuf, FileGenerationIdentity)>,
}

#[cfg(feature = "campaign")]
impl TraceDirSnapshot {
    pub fn capture(root: impl Into<PathBuf>) -> Result<Self, TraceError> {
        let root = root.into();
        let reader = AnchoredDir::open(&root)?;
        Self::capture_from_anchored(&reader, Path::new(""), None)
    }

    pub fn capture_anchored(
        reader: &AnchoredDir,
        root_relative: &Path,
        limits: TraceSnapshotLimits,
    ) -> Result<Self, TraceError> {
        Self::capture_from_anchored(reader, root_relative, Some(limits))
    }

    pub(super) fn capture_from_anchored(
        reader: &AnchoredDir,
        root_relative: &Path,
        limits: Option<TraceSnapshotLimits>,
    ) -> Result<Self, TraceError> {
        let root = reader.root().join(root_relative);
        let preserve_manifest_artifact_role = limits.is_some();
        let dynamic_relative = root_relative.join("dynamic_trace.pb");
        let dynamic_file = match limits {
            Some(limits) => reader
                .read_exact_regular(&dynamic_relative, limits.dynamic_trace_pb_size)
                .map_err(|source| map_manifest_snapshot_read_error("dynamic_trace_pb", source))?,
            None => reader.read_bounded_regular(&dynamic_relative, DYNAMIC_TRACE_PB_MAX_BYTES)?,
        };
        let dynamic_path = root.join("dynamic_trace.pb");
        let mut file_identities = BTreeMap::new();
        insert_snapshot_file_identity(
            &mut file_identities,
            "dynamic_trace_pb",
            preserve_manifest_artifact_role,
            dynamic_file.generation_identity(),
            dynamic_path.clone(),
        )?;
        let dynamic_trace_pb = dynamic_file.into_raw();
        // Root-byte failures precede payload-tree failures in both validators.
        // Keep this byte-level check at snapshot capture so an empty root cannot
        // be masked by an independently empty/missing payload tree.
        if dynamic_trace_pb.is_empty() {
            return Err(TraceError::TraceRootEmpty { path: dynamic_path });
        }

        let allocations_relative = root_relative.join("extra_info/allocations.json");
        let allocations_path = root.join("extra_info/allocations.json");
        let allocations_file = capture_optional_snapshot_artifact(
            reader,
            &allocations_relative,
            limits.map(|limits| limits.allocations_json_size),
            "allocations_json",
            ALLOCATIONS_JSON_MAX_BYTES,
        )?;
        if let Some(file) = &allocations_file {
            insert_snapshot_file_identity(
                &mut file_identities,
                "allocations_json",
                preserve_manifest_artifact_role,
                file.generation_identity(),
                allocations_path.clone(),
            )?;
        }
        let allocations_json = allocations_file.map(AnchoredFile::into_raw);

        let sampling_relative = root_relative.join("trace_sampling.json");
        let sampling_path = root.join("trace_sampling.json");
        let sampling_file = capture_optional_snapshot_artifact(
            reader,
            &sampling_relative,
            limits.map(|limits| limits.trace_sampling_json_size),
            "trace_sampling_json",
            TRACE_SAMPLING_JSON_MAX_BYTES,
        )?;
        if let Some(file) = &sampling_file {
            insert_snapshot_file_identity(
                &mut file_identities,
                "trace_sampling_json",
                preserve_manifest_artifact_role,
                file.generation_identity(),
                sampling_path.clone(),
            )?;
        }
        let trace_sampling_json = sampling_file.map(AnchoredFile::into_raw);

        let stats_relative = root_relative.join("stats.csv");
        let stats_path = root.join("stats.csv");
        let stats_file = capture_optional_snapshot_artifact(
            reader,
            &stats_relative,
            limits.map(|limits| limits.stats_csv_size),
            "stats_csv",
            STATS_CSV_MAX_BYTES,
        )?;
        if let Some(file) = &stats_file {
            insert_snapshot_file_identity(
                &mut file_identities,
                "stats_csv",
                preserve_manifest_artifact_role,
                file.generation_identity(),
                stats_path.clone(),
            )?;
        }
        let stats_csv = stats_file.map(AnchoredFile::into_raw);

        let (maximum_files, maximum_bytes) = limits.map_or(
            (
                DIRECT_THREADBLOCK_FILE_MAX_COUNT,
                CAMPAIGN_THREADBLOCK_FILE_MAX_BYTES,
            ),
            |limits| (limits.threadblock_file_count, limits.threadblock_raw_bytes),
        );
        let maximum_file_bytes = limits.map_or(CAMPAIGN_THREADBLOCK_FILE_MAX_BYTES, |limits| {
            limits
                .threadblock_raw_bytes
                .min(CAMPAIGN_THREADBLOCK_FILE_MAX_BYTES)
        });
        let tree_relative = root_relative.join("threadblocks");
        let mut tree = reader
            .read_pb_tree_with_file_limit(
                &tree_relative,
                maximum_files,
                maximum_bytes,
                maximum_file_bytes,
            )
            .map_err(|source| {
                map_threadblock_tree_read_error(&root, source, preserve_manifest_artifact_role)
            })?;
        let actual_file_count = u64::try_from(tree.files.len()).map_err(|_| {
            TraceError::SnapshotArtifactCardinalityMismatch {
                artifact: "threadblock_payload",
                expected: maximum_files,
                actual: u64::MAX,
            }
        })?;
        if let Some(limits) = limits {
            if actual_file_count != limits.threadblock_file_count {
                return Err(TraceError::SnapshotArtifactCardinalityMismatch {
                    artifact: "threadblock_payload",
                    expected: limits.threadblock_file_count,
                    actual: actual_file_count,
                });
            }
            if tree.bytes != limits.threadblock_raw_bytes {
                return Err(TraceError::SnapshotArtifactSizeMismatch {
                    artifact: "threadblock_payload",
                    expected: limits.threadblock_raw_bytes,
                    actual: tree.bytes,
                });
            }
        }
        if tree.files.is_empty() {
            return Err(TraceError::TraceThreadblocksEmpty {
                dir: root.join("threadblocks"),
            });
        }
        // Python's producer hashes slash-separated manifest path spellings, not
        // component-wise `Path` order (`a.pb` sorts before `a/b.pb` there).
        // Re-establish that exact order after the anchored tree walk.
        tree.files
            .sort_by(|left, right| left.relative.to_str().cmp(&right.relative.to_str()));
        let mut raw_bytes = 0_u64;
        let mut stream_digest = Sha256::new();
        let mut manifest = String::from("trace-payload-file-manifest-v1\n");
        let mut by_path = BTreeMap::new();
        for entry in tree.files {
            let relative = entry.relative.strip_prefix(root_relative).map_err(|_| {
                TraceError::ThreadblockPayloadSnapshotInvalid {
                    root: root.clone(),
                    message: format!(
                        "descriptor-anchored payload path is outside trace root: {}",
                        entry.relative.display()
                    ),
                }
            })?;
            let relative =
                relative
                    .to_str()
                    .ok_or_else(|| TraceError::ThreadblockPayloadSnapshotInvalid {
                        root: root.clone(),
                        message: format!("payload path is not UTF-8: {}", entry.relative.display()),
                    })?;
            if !safe_payload_manifest_path(relative) {
                return Err(TraceError::ThreadblockPayloadSnapshotInvalid {
                    root: root.clone(),
                    message: format!(
                        "payload path is not canonical for the per-file manifest: {relative:?}"
                    ),
                });
            }
            let path = root.join(relative);
            insert_snapshot_file_identity(
                &mut file_identities,
                "threadblock_payload",
                preserve_manifest_artifact_role,
                entry.file.generation_identity(),
                path.clone(),
            )?;
            let raw = entry.file.into_raw();
            let size = u64::try_from(raw.len()).map_err(|_| {
                TraceError::ThreadblockPayloadSnapshotInvalid {
                    root: root.clone(),
                    message: format!("payload {} length exceeds u64", path.display()),
                }
            })?;
            raw_bytes = raw_bytes.checked_add(size).ok_or_else(|| {
                TraceError::ThreadblockPayloadSnapshotInvalid {
                    root: root.clone(),
                    message: "payload byte count overflows u64".to_owned(),
                }
            })?;
            let file_digest = Sha256Digest::of(&raw);
            manifest.push_str(&format!("{relative}\t{size}\t{file_digest}\n"));
            stream_digest.update(&raw);
            by_path.insert(path, raw);
        }
        let threadblock_payload = VerifiedSamplingPayload {
            by_path,
            identity: ThreadblockPayloadIdentity {
                file_count: actual_file_count,
                raw_bytes,
                sorted_path_manifest_sha256: Sha256Digest::of(manifest.as_bytes()),
                sorted_path_stream_sha256: stream_digest.finish_digest(),
            },
        };
        let identity = TraceArtifactIdentity {
            dynamic_trace_pb: ArtifactDigestIdentity::from_bytes(&dynamic_trace_pb, &dynamic_path)?,
            allocations_json: allocations_json
                .as_deref()
                .map(|raw| ArtifactDigestIdentity::from_bytes(raw, &allocations_path))
                .transpose()?,
            trace_sampling_json: trace_sampling_json
                .as_deref()
                .map(|raw| ArtifactDigestIdentity::from_bytes(raw, &sampling_path))
                .transpose()?,
            stats_csv: stats_csv
                .as_deref()
                .map(|raw| ArtifactDigestIdentity::from_bytes(raw, &stats_path))
                .transpose()?,
            threadblock_payload: threadblock_payload.identity.clone(),
        };
        Ok(Self {
            root,
            dynamic_trace_pb,
            allocations_json,
            trace_sampling_json,
            stats_csv,
            threadblock_payload,
            identity,
            file_identities,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn artifact_identity(&self) -> &TraceArtifactIdentity {
        &self.identity
    }

    pub fn file_identities(&self) -> impl Iterator<Item = (FileIdentity, &Path)> {
        self.file_identities
            .iter()
            .map(|(identity, (path, _))| (*identity, path.as_path()))
    }

    /// Every file generation whose exact bytes entered this snapshot.
    pub fn file_generations(&self) -> impl Iterator<Item = (FileGenerationIdentity, &Path)> {
        self.file_identities
            .values()
            .map(|(path, generation)| (*generation, path.as_path()))
    }
}

pub(super) struct JsonShapeCounter {
    pub(super) nodes: u64,
    pub(super) top_level_sequence_entries: u64,
    pub(super) maximum_nodes: u64,
    pub(super) maximum_top_level_sequence_entries: Option<u64>,
    pub(super) exceeded: Option<JsonResourceLimit>,
}

impl JsonShapeCounter {
    pub(super) fn bump_nodes<E: serde::de::Error>(&mut self) -> Result<(), E> {
        self.nodes = self.nodes.saturating_add(1);
        if self.nodes > self.maximum_nodes {
            let exceeded = JsonResourceLimit {
                resource: "json_nodes",
                observed: self.nodes,
                limit: self.maximum_nodes,
            };
            self.exceeded = Some(exceeded);
            return Err(E::custom("JSON node limit exceeded"));
        }
        Ok(())
    }

    pub(super) fn bump_top_level_sequence_entries<E: serde::de::Error>(&mut self) -> Result<(), E> {
        self.top_level_sequence_entries = self.top_level_sequence_entries.saturating_add(1);
        if let Some(limit) = self.maximum_top_level_sequence_entries
            && self.top_level_sequence_entries > limit
        {
            let exceeded = JsonResourceLimit {
                resource: "top_level_entries",
                observed: self.top_level_sequence_entries,
                limit,
            };
            self.exceeded = Some(exceeded);
            return Err(E::custom("JSON top-level entry limit exceeded"));
        }
        Ok(())
    }
}

pub(super) struct JsonValueSeed<'a> {
    pub(super) counter: &'a mut JsonShapeCounter,
    pub(super) root: bool,
    pub(super) top_level_sequence_entry: bool,
}

pub(super) struct JsonValueVisitor<'a> {
    pub(super) counter: &'a mut JsonShapeCounter,
    pub(super) root: bool,
}
