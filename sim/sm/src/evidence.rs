//! Manifest-only admission for one launch's transitive simulator evidence.
//!
//! This boundary snapshots and hashes raw artifacts before parsing them, proves
//! the existing trace/exec/control/card/coverage joins, and only then constructs
//! a canonical JSON identity seal. The traced workload binary is deliberately a
//! typed gap: the current trace contract has no producer-bound binary digest.

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU64;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use sim_arch::card::{ArchCard, CalibError};
use sim_arch::{ArchError, ArchId, Sku};
use sim_mem::{PchaseCard, PchaseCardError};
use sim_trace::reader::MemoryPayloadContract;
use sim_trace::{
    AnchoredDir, ArtifactDigestIdentity, CanonicalJsonError, ExecInfo, ExecInfoError,
    FileGenerationIdentity, FileIdentity, Sha256Digest, TraceArtifactIdentity, TraceDir,
    TraceDirSnapshot, TraceError, TraceSnapshotLimits, canonical_json, strict_json_from_slice,
};

use crate::control_bits::{ControlBitsError, ControlSidecar};
use crate::program::{FunctionSchedule, ProgramError, build_schedules};
use crate::replay::{
    ExtractedCoverage, FullCoverageSource, ReplayError, ReplayLaunchCoverageBinding,
    ReplayLaunchCoverageBindingError, admit_launch_coverage,
};

const MANIFEST_SCHEMA_VERSION: u32 = 2;
const WIRE_VALIDATOR_CONTRACT_ID: &str = "sim_trace_wire_v1";
const TRACE_BUNDLE_DIGEST_CONTRACT_ID: &str = "sim-sm.trace-bundle-identity.v1";
const TRACE_BUNDLE_ID_PREFIX: &str = "trace-bundle-v1-";
const MAX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
const MAX_REGULAR_ARTIFACT_BYTES: u64 = 64 * 1024 * 1024;
const MAX_THREADBLOCK_PAYLOAD_FILES: u64 = 1_000_000;
const MAX_THREADBLOCK_PAYLOAD_BYTES: u64 = 768 * 1024 * 1024;
const MAX_EVIDENCE_SNAPSHOT_BYTES: u64 = 1024 * 1024 * 1024;

/// Closed artifact tokens carried through launch-evidence admission failures.
///
/// Keeping these as an enum, rather than dispatching on rendered paths or error
/// strings, lets downstream error serializers classify card, configuration, and
/// trace failures exhaustively.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ManifestArtifactToken {
    LaunchEvidenceManifest,
    DynamicTracePb,
    ThreadblockPayload,
    ThreadblockPayloadSortedPathManifest,
    ThreadblockPayloadSortedPathStream,
    AllocationsJson,
    TraceSamplingJson,
    StatsCsv,
    EnhancedExecutionInfoJson,
    ControlBitsSidecar,
    ArchCard,
    PchaseCard,
}

impl ManifestArtifactToken {
    pub const fn token(self) -> &'static str {
        match self {
            Self::LaunchEvidenceManifest => "launch_evidence_manifest",
            Self::DynamicTracePb => "dynamic_trace_pb",
            Self::ThreadblockPayload => "threadblock_payload",
            Self::ThreadblockPayloadSortedPathManifest => {
                "threadblock_payload.sorted_path_manifest"
            }
            Self::ThreadblockPayloadSortedPathStream => "threadblock_payload.sorted_path_stream",
            Self::AllocationsJson => "allocations_json",
            Self::TraceSamplingJson => "trace_sampling_json",
            Self::StatsCsv => "stats_csv",
            Self::EnhancedExecutionInfoJson => "enhanced_execution_info_json",
            Self::ControlBitsSidecar => "control_bits_sidecar",
            Self::ArchCard => "arch_card",
            Self::PchaseCard => "pchase_card",
        }
    }

    fn from_trace_snapshot_token(token: &str) -> Option<Self> {
        match token {
            "dynamic_trace_pb" => Some(Self::DynamicTracePb),
            "threadblock_payload" => Some(Self::ThreadblockPayload),
            "allocations_json" => Some(Self::AllocationsJson),
            "trace_sampling_json" => Some(Self::TraceSamplingJson),
            "stats_csv" => Some(Self::StatsCsv),
            _ => None,
        }
    }
}

impl std::fmt::Display for ManifestArtifactToken {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.token())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    #[error("could not parse launch-evidence manifest {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("unsupported launch-evidence manifest schema_version {got}; expected {expected}")]
    SchemaVersionUnsupported { got: u32, expected: u32 },
    #[error("manifest artifact {artifact} has a non-canonical relative path {path:?}")]
    ArtifactPathInvalid {
        artifact: ManifestArtifactToken,
        path: String,
    },
    #[error(
        "manifest artifact path edge {edge} for {artifact} is invalid: expected {expected}, got {actual}"
    )]
    ArtifactPathMismatch {
        artifact: ManifestArtifactToken,
        edge: &'static str,
        expected: PathBuf,
        actual: PathBuf,
    },
    #[error("descriptor-anchored root-manifest read failed: {0}")]
    Anchored(#[from] sim_trace::AnchoredReadError),
    #[error("descriptor-anchored artifact read failed for {artifact}: {source}")]
    ArtifactRead {
        artifact: ManifestArtifactToken,
        #[source]
        source: sim_trace::AnchoredReadError,
    },
    #[error(
        "retained artifact generation for {artifact} at {path} could not be rechecked: {source}"
    )]
    ArtifactGenerationRead {
        artifact: ManifestArtifactToken,
        path: PathBuf,
        #[source]
        source: sim_trace::AnchoredReadError,
    },
    #[error(
        "retained artifact generation for {artifact} at {path} changed: expected {expected:?}, got {actual:?}"
    )]
    ArtifactGenerationMismatch {
        artifact: ManifestArtifactToken,
        path: PathBuf,
        expected: FileGenerationIdentity,
        actual: FileGenerationIdentity,
    },
    #[error("manifest artifact {artifact} duplicates another artifact path at {path}")]
    ArtifactPathDuplicate {
        artifact: ManifestArtifactToken,
        path: PathBuf,
    },
    #[error("manifest artifact {artifact} aliases the same inode as {first} at {second}")]
    ArtifactFileAlias {
        artifact: ManifestArtifactToken,
        first: PathBuf,
        second: PathBuf,
    },
    #[error("artifact {artifact} byte size mismatch: manifest {expected}, snapshot {actual}")]
    ArtifactSizeMismatch {
        artifact: ManifestArtifactToken,
        expected: u64,
        actual: u64,
    },
    #[error("artifact {artifact} SHA-256 mismatch: manifest {expected}, snapshot {actual}")]
    ArtifactDigestMismatch {
        artifact: ManifestArtifactToken,
        expected: Sha256Digest,
        actual: Sha256Digest,
    },
    #[error(
        "artifact {artifact} presence mismatch: manifest declares {expected}, snapshot has {actual}"
    )]
    ArtifactPresenceMismatch {
        artifact: ManifestArtifactToken,
        expected: bool,
        actual: bool,
    },
    #[error("artifact {artifact} file count mismatch: manifest {expected}, snapshot {actual}")]
    ArtifactCardinalityMismatch {
        artifact: ManifestArtifactToken,
        expected: u64,
        actual: u64,
    },
    #[error(
        "artifact {artifact} declares {declared} {resource}, exceeding implementation limit {limit}"
    )]
    ArtifactResourceLimitExceeded {
        artifact: ManifestArtifactToken,
        resource: &'static str,
        declared: u64,
        limit: u64,
    },
    #[error(
        "launch-evidence manifest declares {declared} aggregate artifact bytes, exceeding implementation limit {limit}"
    )]
    SnapshotResourceLimitExceeded { declared: u64, limit: u64 },
    #[error("trace-bundle path must be a nonempty canonical relative path; got {path:?}")]
    TraceBundlePathInvalid { path: String },
    #[error("trace-bundle field {field} must be positive; got {actual}")]
    TraceBundleNonpositive { field: &'static str, actual: u64 },
    #[error(
        "trace-bundle total size overflow while adding {component}: accumulated {accumulated}, addend {addend}"
    )]
    TraceBundleSizeOverflow {
        component: &'static str,
        accumulated: u64,
        addend: u64,
    },
    #[error("trace-bundle count edge {edge} mismatch: expected {expected}, got {actual}")]
    TraceBundleCountMismatch {
        edge: &'static str,
        expected: u64,
        actual: u64,
    },
    #[error("trace-bundle wire-version contract token is invalid: {value:?}")]
    TraceBundleWireVersionInvalid { value: String },
    #[error("replay-authenticated launch field {field} value {value} is outside {expected_domain}")]
    ReplayIdentityValueOutOfRange {
        field: &'static str,
        value: i64,
        expected_domain: &'static str,
    },
    #[error("trace contract/capability admission failed: {source}")]
    CapabilityMissing {
        #[source]
        source: TraceError,
    },
    #[error("trace admission failed: {0}")]
    Trace(TraceError),
    #[error("exec-info admission failed: {0}")]
    ExecInfo(#[from] ExecInfoError),
    #[error("control-sidecar admission failed: {0}")]
    Control(#[from] ControlBitsError),
    #[error("exec/control schedule join failed: {0}")]
    Program(#[from] ProgramError),
    #[error("architecture-card admission failed: {0}")]
    ArchCard(#[from] CalibError),
    #[error("P-chase-card admission failed: {0}")]
    PchaseCard(#[from] PchaseCardError),
    #[error("architecture identity is invalid: {0}")]
    Architecture(#[from] ArchError),
    #[error("coverage admission failed: {0}")]
    Coverage(#[from] ReplayError),
    #[error("required identity field {field} is absent or empty")]
    IdentityFieldMissing { field: &'static str },
    #[error(
        "identity edge for artifact {artifact} {edge} mismatch: expected {expected:?}, got {actual:?}"
    )]
    IdentityMismatch {
        artifact: ManifestArtifactToken,
        edge: &'static str,
        expected: String,
        actual: String,
    },
    #[error(
        "launch selector ({device_id}, {stream_id}, {kernel_id}) does not identify a root launch"
    )]
    LaunchMissing {
        device_id: i64,
        stream_id: i64,
        kernel_id: i32,
    },
    #[error(
        "launch selector ({device_id}, {stream_id}, {kernel_id}) is ambiguous in the root trace"
    )]
    LaunchAmbiguous {
        device_id: i64,
        stream_id: i64,
        kernel_id: i32,
    },
    #[error("identity edge {edge} is unverifiable: {reason}")]
    EdgeUnverifiable {
        edge: &'static str,
        reason: &'static str,
    },
    #[error("could not serialize launch evidence identity: {0}")]
    IdentitySerialization(#[from] serde_json::Error),
    #[error("could not canonicalize launch evidence identity: {0}")]
    IdentityCanonicalization(#[from] CanonicalJsonError),
}

#[derive(Debug, Deserialize)]
struct ManifestVersionEnvelope {
    schema_version: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LaunchEvidenceManifest {
    #[serde(rename = "schema_version")]
    _schema_version: u32,
    artifacts: ManifestArtifacts,
    launch_selector: ManifestLaunchSelector,
    architecture: ManifestArchitecture,
    binary_digest: ManifestBinaryDigest,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestArtifacts {
    dynamic_trace_pb: ManifestArtifact,
    threadblock_payload: ManifestThreadblockPayload,
    allocations_json: ManifestOptionalArtifact,
    trace_sampling_json: ManifestOptionalArtifact,
    stats_csv: ManifestArtifact,
    enhanced_execution_info_json: ManifestArtifact,
    control_bits_sidecar: ManifestArtifact,
    arch_card: ManifestArtifact,
    pchase_card: ManifestArtifact,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestArtifact {
    path: String,
    size_bytes: u64,
    sha256: Sha256Digest,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestThreadblockPayload {
    path: String,
    file_count: u64,
    raw_bytes: u64,
    sorted_path_manifest_sha256: Sha256Digest,
    sorted_path_stream_sha256: Sha256Digest,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
enum ManifestOptionalArtifact {
    Present { artifact: ManifestArtifact },
    Absent,
}

impl ManifestOptionalArtifact {
    fn artifact(&self) -> Option<&ManifestArtifact> {
        match self {
            Self::Present { artifact } => Some(artifact),
            Self::Absent => None,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestLaunchSelector {
    device_id: i64,
    stream_id: i64,
    kernel_id: i32,
    function_unique_id: i32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestArchitecture {
    arch_id: String,
    sku: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
enum ManifestBinaryDigest {
    Value { sha256: Sha256Digest },
    TypedGap { gap: BinaryDigestGap },
}

/// Closed reason why one admitted launch lacks a traced-workload binary digest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BinaryDigestGap {
    TraceContractDoesNotEmitBinaryDigest,
}

/// Canonical trace-root path relative to the launch-evidence manifest directory.
///
/// Construction is private to admission, so consumers cannot mint a path that
/// bypasses the manifest's canonical-relative-path checks.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct CanonicalTraceBundlePath(String);

impl CanonicalTraceBundlePath {
    fn checked(raw: &str) -> Result<Self, ManifestError> {
        let path = Path::new(raw);
        if raw.is_empty()
            || raw.starts_with('/')
            || raw.ends_with('/')
            || raw.contains("//")
            || raw
                .split('/')
                .any(|component| matches!(component, "." | ".."))
            || !raw.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'/' | b'-')
            })
            || path.is_absolute()
            || path
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
        {
            return Err(ManifestError::TraceBundlePathInvalid {
                path: raw.to_owned(),
            });
        }
        Ok(Self(raw.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn as_path(&self) -> &Path {
        Path::new(&self.0)
    }
}

/// Positive byte size and raw SHA-256 for one replay-consumed bundle component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct TraceBundleComponentIdentity {
    size_bytes: NonZeroU64,
    sha256: Sha256Digest,
}

impl TraceBundleComponentIdentity {
    fn checked(
        field: &'static str,
        size_bytes: u64,
        sha256: Sha256Digest,
    ) -> Result<Self, ManifestError> {
        Ok(Self {
            size_bytes: require_positive_trace_bundle_value(field, size_bytes)?,
            sha256,
        })
    }

    pub fn size_bytes(self) -> u64 {
        self.size_bytes.get()
    }

    pub fn sha256(self) -> Sha256Digest {
        self.sha256
    }
}

/// Transitive identity of every snapshotted threadblock protobuf payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ThreadblockPayloadSetIdentity {
    file_count: NonZeroU64,
    total_size_bytes: NonZeroU64,
    manifest_sha256: Sha256Digest,
    content_sha256: Sha256Digest,
}

impl ThreadblockPayloadSetIdentity {
    fn checked(
        file_count: u64,
        total_size_bytes: u64,
        manifest_sha256: Sha256Digest,
        content_sha256: Sha256Digest,
    ) -> Result<Self, ManifestError> {
        Ok(Self {
            file_count: require_positive_trace_bundle_value(
                "components.threadblock_payloads.file_count",
                file_count,
            )?,
            total_size_bytes: require_positive_trace_bundle_value(
                "components.threadblock_payloads.total_size_bytes",
                total_size_bytes,
            )?,
            manifest_sha256,
            content_sha256,
        })
    }

    pub fn file_count(&self) -> u64 {
        self.file_count.get()
    }

    pub fn total_size_bytes(&self) -> u64 {
        self.total_size_bytes.get()
    }

    pub fn manifest_sha256(&self) -> Sha256Digest {
        self.manifest_sha256
    }

    pub fn content_sha256(&self) -> Sha256Digest {
        self.content_sha256
    }
}

/// The five component identities required by `sim_result.v2.trace_identity`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TraceBundleComponents {
    root_dynamic_trace: TraceBundleComponentIdentity,
    threadblock_payloads: ThreadblockPayloadSetIdentity,
    execution_metadata: TraceBundleComponentIdentity,
    control_bits_sidecar: TraceBundleComponentIdentity,
    stats: TraceBundleComponentIdentity,
}

impl TraceBundleComponents {
    pub fn root_dynamic_trace(&self) -> TraceBundleComponentIdentity {
        self.root_dynamic_trace
    }

    pub fn threadblock_payloads(&self) -> &ThreadblockPayloadSetIdentity {
        &self.threadblock_payloads
    }

    pub fn execution_metadata(&self) -> TraceBundleComponentIdentity {
        self.execution_metadata
    }

    pub fn control_bits_sidecar(&self) -> TraceBundleComponentIdentity {
        self.control_bits_sidecar
    }

    pub fn stats(&self) -> TraceBundleComponentIdentity {
        self.stats
    }
}

#[derive(Serialize)]
struct TraceBundleDigestPreimage<'a> {
    digest_contract_id: &'static str,
    wire_version: &'a str,
    total_size_bytes: NonZeroU64,
    record_count: NonZeroU64,
    instruction_count: NonZeroU64,
    components: &'a TraceBundleComponents,
}

/// Immutable authenticated projection matching `sim_result.v2.trace_identity`.
///
/// The content-addressed bundle SHA uses a domain-separated canonical-JSON
/// preimage and commits all five replay-consumed component identities plus the
/// validated wire/count contract. The bundle path is deliberately relocation
/// metadata and is not part of the content digest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TraceBundleIdentity {
    bundle_id: String,
    bundle_path: CanonicalTraceBundlePath,
    bundle_sha256: Sha256Digest,
    wire_version: String,
    total_size_bytes: NonZeroU64,
    record_count: NonZeroU64,
    instruction_count: NonZeroU64,
    components: TraceBundleComponents,
}

impl TraceBundleIdentity {
    #[allow(clippy::too_many_arguments)]
    fn checked(
        bundle_path: CanonicalTraceBundlePath,
        wire_version: &str,
        validation_launch_count: u64,
        validation_threadblock_count: u64,
        validation_instruction_count: u64,
        stats_admitted_record_count: u64,
        components: TraceBundleComponents,
    ) -> Result<Self, ManifestError> {
        if wire_version.is_empty() || !wire_version.bytes().all(|byte| byte.is_ascii_graphic()) {
            return Err(ManifestError::TraceBundleWireVersionInvalid {
                value: wire_version.to_owned(),
            });
        }
        require_positive_trace_bundle_value(
            "wire_admission.launch_count",
            validation_launch_count,
        )?;
        let validated_threadblocks = require_positive_trace_bundle_value(
            "wire_admission.threadblock_count",
            validation_threadblock_count,
        )?;
        if validated_threadblocks.get() != components.threadblock_payloads.file_count() {
            return Err(ManifestError::TraceBundleCountMismatch {
                edge: "snapshotted payload file count->validated threadblock count",
                expected: components.threadblock_payloads.file_count(),
                actual: validated_threadblocks.get(),
            });
        }
        let instruction_count =
            require_positive_trace_bundle_value("instruction_count", validation_instruction_count)?;
        let record_count =
            require_positive_trace_bundle_value("record_count", stats_admitted_record_count)?;
        if record_count != instruction_count {
            return Err(ManifestError::TraceBundleCountMismatch {
                edge: "stats admitted record count->validated instruction count",
                expected: instruction_count.get(),
                actual: record_count.get(),
            });
        }

        let mut total_size_bytes = 0_u64;
        for (component, addend) in [
            (
                "root_dynamic_trace",
                components.root_dynamic_trace.size_bytes(),
            ),
            (
                "threadblock_payloads",
                components.threadblock_payloads.total_size_bytes(),
            ),
            (
                "execution_metadata",
                components.execution_metadata.size_bytes(),
            ),
            (
                "control_bits_sidecar",
                components.control_bits_sidecar.size_bytes(),
            ),
            ("stats", components.stats.size_bytes()),
        ] {
            total_size_bytes = total_size_bytes.checked_add(addend).ok_or(
                ManifestError::TraceBundleSizeOverflow {
                    component,
                    accumulated: total_size_bytes,
                    addend,
                },
            )?;
        }
        let total_size_bytes =
            require_positive_trace_bundle_value("total_size_bytes", total_size_bytes)?;
        let preimage = TraceBundleDigestPreimage {
            digest_contract_id: TRACE_BUNDLE_DIGEST_CONTRACT_ID,
            wire_version,
            total_size_bytes,
            record_count,
            instruction_count,
            components: &components,
        };
        let preimage = serde_json::to_value(preimage)?;
        let preimage = canonical_json(&preimage)?;
        let bundle_sha256 = Sha256Digest::of(preimage.as_bytes());
        let bundle_id = format!("{TRACE_BUNDLE_ID_PREFIX}{bundle_sha256}");
        Ok(Self {
            bundle_id,
            bundle_path,
            bundle_sha256,
            wire_version: wire_version.to_owned(),
            total_size_bytes,
            record_count,
            instruction_count,
            components,
        })
    }

    pub fn bundle_id(&self) -> &str {
        &self.bundle_id
    }

    pub fn bundle_path(&self) -> &CanonicalTraceBundlePath {
        &self.bundle_path
    }

    pub fn bundle_sha256(&self) -> Sha256Digest {
        self.bundle_sha256
    }

    pub fn wire_version(&self) -> &str {
        &self.wire_version
    }

    pub fn total_size_bytes(&self) -> u64 {
        self.total_size_bytes.get()
    }

    pub fn record_count(&self) -> u64 {
        self.record_count.get()
    }

    pub fn instruction_count(&self) -> u64 {
        self.instruction_count.get()
    }

    pub fn components(&self) -> &TraceBundleComponents {
        &self.components
    }
}

/// Replay-visible launch fields authenticated by trace, metadata, and coverage admission.
///
/// The unsigned device/stream domains match the reusable v2 launch-key schema;
/// cohort-only workload/execution/ordinal assertions are intentionally absent.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ReplayAuthenticatedLaunchIdentity {
    kernel_mangled_name: String,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    device_id: u32,
    stream_id: u64,
    binary_architecture: ArchId,
}

impl ReplayAuthenticatedLaunchIdentity {
    fn checked(
        binding: &ReplayLaunchCoverageBinding,
        binary_architecture: ArchId,
    ) -> Result<Self, ManifestError> {
        let device_id = u32::try_from(binding.device_id()).map_err(|_| {
            ManifestError::ReplayIdentityValueOutOfRange {
                field: "device_id",
                value: binding.device_id(),
                expected_domain: "uint32",
            }
        })?;
        let stream_id = u64::try_from(binding.stream_id()).map_err(|_| {
            ManifestError::ReplayIdentityValueOutOfRange {
                field: "stream_id",
                value: binding.stream_id(),
                expected_domain: "uint64",
            }
        })?;
        let block = binding.block().ok_or(ManifestError::IdentityFieldMissing {
            field: "dynamic_trace.kernel.block_dim",
        })?;
        Ok(Self {
            kernel_mangled_name: binding.kernel_mangled_name().to_owned(),
            grid: binding.grid(),
            block,
            device_id,
            stream_id,
            binary_architecture,
        })
    }

    pub fn kernel_mangled_name(&self) -> &str {
        &self.kernel_mangled_name
    }

    pub fn grid(&self) -> (u32, u32, u32) {
        self.grid
    }

    pub fn block(&self) -> (u32, u32, u32) {
        self.block
    }

    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    pub fn stream_id(&self) -> u64 {
        self.stream_id
    }

    pub fn binary_architecture(&self) -> ArchId {
        self.binary_architecture
    }
}

fn require_positive_trace_bundle_value(
    field: &'static str,
    actual: u64,
) -> Result<NonZeroU64, ManifestError> {
    NonZeroU64::new(actual).ok_or(ManifestError::TraceBundleNonpositive { field, actual })
}

#[derive(Debug, Clone, Serialize)]
#[serde(deny_unknown_fields)]
struct SealedArtifact {
    size_bytes: u64,
    sha256: Sha256Digest,
}

impl From<ArtifactDigestIdentity> for SealedArtifact {
    fn from(identity: ArtifactDigestIdentity) -> Self {
        Self {
            size_bytes: identity.size_bytes(),
            sha256: identity.sha256(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(deny_unknown_fields)]
struct LaunchEvidencePayload {
    schema_version: u32,
    cell_id: Option<String>,
    kernel: KernelIdentity,
    launch: LaunchIdentity,
    geometry: GeometryIdentity,
    trace_contract_id: String,
    memory_payload_classes: BTreeSet<String>,
    wire_admission: WireAdmissionIdentity,
    artifact_digests: ArtifactDigests,
    arch: ArchitectureIdentity,
    binary: BinaryIdentity,
    coverage: CoverageIdentity,
}

#[derive(Debug, Clone, Serialize)]
#[serde(deny_unknown_fields)]
struct KernelIdentity {
    mangled_name: String,
    args_hash: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(deny_unknown_fields)]
struct LaunchIdentity {
    device_id: i64,
    stream_id: i64,
    kernel_id: i32,
    function_unique_id: i32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(deny_unknown_fields)]
struct GeometryIdentity {
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    cluster_dim: (u32, u32, u32),
}

#[derive(Debug, Clone, Serialize)]
#[serde(deny_unknown_fields)]
struct WireAdmissionIdentity {
    validator_contract_id: String,
    launch_count: u64,
    threadblock_count: u64,
    instruction_count: u64,
    stats_row_count: u64,
    stats_expected_record_count: u64,
    stats_admitted_record_count: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(deny_unknown_fields)]
struct ArtifactDigests {
    dynamic_trace_pb: SealedArtifact,
    threadblock_payload_file_count: u64,
    threadblock_payload_raw_bytes: u64,
    threadblock_payload_manifest_sha256: Sha256Digest,
    threadblock_payload_sha256: Sha256Digest,
    allocations_json: Option<SealedArtifact>,
    trace_sampling_json: Option<SealedArtifact>,
    trace_sampling_json_seal: Option<Sha256Digest>,
    stats_csv: SealedArtifact,
    enhanced_execution_info_json: SealedArtifact,
    control_bits_sidecar: SealedArtifact,
}

#[derive(Debug, Clone, Serialize)]
#[serde(deny_unknown_fields)]
struct ArchitectureIdentity {
    arch_id: String,
    sku: String,
    arch_card: SealedArtifact,
    pchase_card: SealedArtifact,
}

#[derive(Debug, Clone, Serialize)]
#[serde(deny_unknown_fields)]
struct BinaryIdentity {
    name: String,
    digest: BinaryDigestIdentity,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
enum BinaryDigestIdentity {
    TypedGap { gap: BinaryDigestGap },
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum CoverageIdentity {
    ExactFull {
        blocks_observed: u64,
    },
    LosslessStored {
        blocks_observed: u64,
        sidecar_sha256: Sha256Digest,
    },
    EstimatedSampled {
        population_blocks: u64,
        blocks_observed: u64,
        selected_block_ids: Vec<i64>,
        selected_block_ids_sha256: Sha256Digest,
        sidecar_sha256: Sha256Digest,
    },
}

/// Canonical, SHA-256-sealed identity constructed only after transitive admission.
#[derive(Debug, Clone)]
pub struct LaunchEvidenceIdentity {
    payload: LaunchEvidencePayload,
    trace_bundle: TraceBundleIdentity,
    replay_launch: ReplayAuthenticatedLaunchIdentity,
    seal: Sha256Digest,
    canonical_json: String,
}

impl LaunchEvidenceIdentity {
    fn seal(
        payload: LaunchEvidencePayload,
        trace_bundle: TraceBundleIdentity,
        replay_launch: ReplayAuthenticatedLaunchIdentity,
    ) -> Result<Self, ManifestError> {
        let unsigned = serde_json::to_value(&payload)?;
        let unsigned_canonical = canonical_json(&unsigned)?;
        let seal = Sha256Digest::of(unsigned_canonical.as_bytes());
        let mut sealed = unsigned;
        let object = sealed
            .as_object_mut()
            .ok_or(ManifestError::EdgeUnverifiable {
                edge: "launch_evidence_identity",
                reason: "serialized identity root is not an object",
            })?;
        object.insert(
            "identity_sha256".to_owned(),
            serde_json::Value::String(seal.to_string()),
        );
        let canonical_json = canonical_json(&sealed)?;
        Ok(Self {
            payload,
            trace_bundle,
            replay_launch,
            seal,
            canonical_json,
        })
    }

    pub fn seal_sha256(&self) -> Sha256Digest {
        self.seal
    }

    pub fn canonical_json(&self) -> &str {
        &self.canonical_json
    }

    /// Authenticated trace-bundle projection ready for `sim_result.v2` emission.
    pub fn trace_bundle_identity(&self) -> &TraceBundleIdentity {
        &self.trace_bundle
    }

    /// Replay-visible subset of the v2 launch key, with unsigned wire domains.
    pub fn replay_authenticated_launch(&self) -> &ReplayAuthenticatedLaunchIdentity {
        &self.replay_launch
    }

    pub fn arch(&self) -> Result<ArchId, ArchError> {
        ArchId::parse_deploy(&self.payload.arch.arch_id)
    }

    pub fn sku(&self) -> Result<Sku, ArchError> {
        Sku::parse(&self.payload.arch.sku)
    }

    pub fn device_id(&self) -> i64 {
        self.payload.launch.device_id
    }

    pub fn stream_id(&self) -> i64 {
        self.payload.launch.stream_id
    }

    pub fn kernel_id(&self) -> i32 {
        self.payload.launch.kernel_id
    }

    pub fn function_unique_id(&self) -> i32 {
        self.payload.launch.function_unique_id
    }

    pub fn arch_card_sha256(&self) -> Sha256Digest {
        self.payload.arch.arch_card.sha256
    }

    pub fn pchase_card_sha256(&self) -> Sha256Digest {
        self.payload.arch.pchase_card.sha256
    }

    /// Reconstruct the declaration-side launch identity sealed by admission.
    /// This does not create an observed coverage witness.
    pub fn launch_binding(
        &self,
    ) -> Result<ReplayLaunchCoverageBinding, ReplayLaunchCoverageBindingError> {
        ReplayLaunchCoverageBinding::checked(
            self.payload.launch.device_id,
            self.payload.launch.stream_id,
            self.payload.launch.kernel_id,
            self.payload.launch.function_unique_id,
            self.payload.kernel.mangled_name.clone(),
            self.payload.geometry.grid_dim,
            Some(self.payload.geometry.block_dim),
            self.payload.geometry.cluster_dim,
        )
    }

    pub fn binary_digest_gap(&self) -> BinaryDigestGap {
        match self.payload.binary.digest {
            BinaryDigestIdentity::TypedGap { gap } => gap,
        }
    }
}

struct RetainedArtifactGeneration {
    artifact: ManifestArtifactToken,
    relative_path: PathBuf,
    expected: FileGenerationIdentity,
}

/// Exact optional-presence and threadblock-tree claims retained from admission.
/// Re-capturing through the retained ancestor descriptor detects names that did
/// not exist during admission as well as roster substitutions that preserve
/// aggregate count and byte totals.
struct RetainedTraceInventory {
    root_relative: PathBuf,
    limits: TraceSnapshotLimits,
    expected: TraceArtifactIdentity,
}

impl RetainedTraceInventory {
    fn reanchor(&mut self, prefix: &Path) {
        self.root_relative = prefix.join(&self.root_relative);
    }

    fn recheck(&self, base: &AnchoredDir) -> Result<(), ManifestError> {
        let trace_root = base.root().join(&self.root_relative);
        let snapshot = TraceDirSnapshot::capture_anchored(base, &self.root_relative, self.limits)
            .map_err(|source| map_trace_snapshot_error(&trace_root, source))?;
        verify_trace_artifact_identity(&self.expected, snapshot.artifact_identity())
    }
}

/// Retained directory capability plus every file generation consumed by admission
/// and the closed trace inventory, including explicit absence claims.
struct RetainedEvidenceGeneration {
    base: AnchoredDir,
    artifacts: Vec<RetainedArtifactGeneration>,
    trace_inventory: RetainedTraceInventory,
}

impl RetainedEvidenceGeneration {
    fn capture(
        reader: &AnchoredDir,
        identities: BTreeMap<FileIdentity, ObservedArtifactGeneration>,
        trace_inventory: RetainedTraceInventory,
    ) -> Result<Self, ManifestError> {
        let mut artifacts = identities
            .into_values()
            .map(|observed| {
                let relative_path = observed
                    .path
                    .strip_prefix(reader.root())
                    .map_err(|_| ManifestError::EdgeUnverifiable {
                        edge: "retained_artifact_generation",
                        reason: "admitted artifact path is outside its retained evidence base",
                    })?
                    .to_path_buf();
                Ok(RetainedArtifactGeneration {
                    artifact: observed.artifact,
                    relative_path,
                    expected: observed.generation,
                })
            })
            .collect::<Result<Vec<_>, ManifestError>>()?;
        artifacts.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
        let base = reader.descendant(Path::new(""))?;
        Ok(Self {
            base,
            artifacts,
            trace_inventory,
        })
    }

    fn reanchor(mut self, reader: &AnchoredDir, prefix: &Path) -> Result<Self, ManifestError> {
        for artifact in &mut self.artifacts {
            artifact.relative_path = prefix.join(&artifact.relative_path);
        }
        self.trace_inventory.reanchor(prefix);
        self.base = reader.descendant(Path::new(""))?;
        Ok(self)
    }

    fn recheck(&self) -> Result<(), ManifestError> {
        for artifact in &self.artifacts {
            let actual = self
                .base
                .inspect_regular_generation(&artifact.relative_path)
                .map_err(|source| ManifestError::ArtifactGenerationRead {
                    artifact: artifact.artifact,
                    path: artifact.relative_path.clone(),
                    source,
                })?;
            if actual != artifact.expected {
                return Err(ManifestError::ArtifactGenerationMismatch {
                    artifact: artifact.artifact,
                    path: artifact.relative_path.clone(),
                    expected: artifact.expected,
                    actual,
                });
            }
        }
        self.trace_inventory.recheck(&self.base)
    }
}

/// Exact snapshotted inputs paired with their self-sealed launch identity.
pub struct AdmittedLaunchEvidence {
    identity: LaunchEvidenceIdentity,
    trace_dir: TraceDir,
    schedules: BTreeMap<i32, FunctionSchedule>,
    arch_card: ArchCard,
    pchase_card: PchaseCard,
    retained_generation: RetainedEvidenceGeneration,
}

impl AdmittedLaunchEvidence {
    pub fn identity(&self) -> &LaunchEvidenceIdentity {
        &self.identity
    }

    pub fn trace_dir(&self) -> &TraceDir {
        &self.trace_dir
    }

    pub fn schedule(&self) -> &FunctionSchedule {
        &self.schedules[&self.identity.payload.launch.function_unique_id]
    }

    pub fn schedules(&self) -> &BTreeMap<i32, FunctionSchedule> {
        &self.schedules
    }

    pub fn arch_card(&self) -> &ArchCard {
        &self.arch_card
    }

    pub fn pchase_card(&self) -> &PchaseCard {
        &self.pchase_card
    }

    /// Recheck that every named artifact still denotes the admitted generation.
    ///
    /// Replay consumes immutable snapshotted bytes, while publication additionally
    /// requires these retained name-to-generation edges to remain intact.
    pub fn recheck_retained_generation(&self) -> Result<(), ManifestError> {
        self.retained_generation.recheck()
    }
}

fn resolve_manifest_path(
    base: &Path,
    artifact: ManifestArtifactToken,
    raw: &str,
) -> Result<PathBuf, ManifestError> {
    if raw.is_empty()
        || raw.starts_with('/')
        || raw.ends_with('/')
        || raw.contains("//")
        || raw
            .split('/')
            .any(|component| matches!(component, "." | ".."))
        || !raw
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'/' | b'-'))
    {
        return Err(ManifestError::ArtifactPathInvalid {
            artifact,
            path: raw.to_owned(),
        });
    }
    let relative = Path::new(raw);
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(ManifestError::ArtifactPathInvalid {
            artifact,
            path: raw.to_owned(),
        });
    }
    Ok(base.join(relative))
}

fn require_path(
    artifact: ManifestArtifactToken,
    edge: &'static str,
    expected: PathBuf,
    actual: &Path,
) -> Result<(), ManifestError> {
    if actual != expected {
        return Err(ManifestError::ArtifactPathMismatch {
            artifact,
            edge,
            expected,
            actual: actual.to_path_buf(),
        });
    }
    Ok(())
}

fn require_resource_limit(
    artifact: ManifestArtifactToken,
    resource: &'static str,
    declared: u64,
    limit: u64,
) -> Result<(), ManifestError> {
    if declared > limit {
        return Err(ManifestError::ArtifactResourceLimitExceeded {
            artifact,
            resource,
            declared,
            limit,
        });
    }
    Ok(())
}

fn add_snapshot_bytes(total: &mut u64, bytes: u64) -> Result<(), ManifestError> {
    *total = total
        .checked_add(bytes)
        .ok_or(ManifestError::SnapshotResourceLimitExceeded {
            declared: u64::MAX,
            limit: MAX_EVIDENCE_SNAPSHOT_BYTES,
        })?;
    if *total > MAX_EVIDENCE_SNAPSHOT_BYTES {
        return Err(ManifestError::SnapshotResourceLimitExceeded {
            declared: *total,
            limit: MAX_EVIDENCE_SNAPSHOT_BYTES,
        });
    }
    Ok(())
}

fn validate_manifest_resource_limits(artifacts: &ManifestArtifacts) -> Result<(), ManifestError> {
    require_resource_limit(
        ManifestArtifactToken::ThreadblockPayload,
        "files",
        artifacts.threadblock_payload.file_count,
        MAX_THREADBLOCK_PAYLOAD_FILES,
    )?;
    require_resource_limit(
        ManifestArtifactToken::ThreadblockPayload,
        "bytes",
        artifacts.threadblock_payload.raw_bytes,
        MAX_THREADBLOCK_PAYLOAD_BYTES,
    )?;

    let mut total = artifacts.threadblock_payload.raw_bytes;
    if total > MAX_EVIDENCE_SNAPSHOT_BYTES {
        return Err(ManifestError::SnapshotResourceLimitExceeded {
            declared: total,
            limit: MAX_EVIDENCE_SNAPSHOT_BYTES,
        });
    }
    for (token, artifact) in [
        (
            ManifestArtifactToken::DynamicTracePb,
            &artifacts.dynamic_trace_pb,
        ),
        (ManifestArtifactToken::StatsCsv, &artifacts.stats_csv),
        (
            ManifestArtifactToken::EnhancedExecutionInfoJson,
            &artifacts.enhanced_execution_info_json,
        ),
        (
            ManifestArtifactToken::ControlBitsSidecar,
            &artifacts.control_bits_sidecar,
        ),
        (ManifestArtifactToken::ArchCard, &artifacts.arch_card),
        (ManifestArtifactToken::PchaseCard, &artifacts.pchase_card),
    ] {
        require_resource_limit(
            token,
            "bytes",
            artifact.size_bytes,
            MAX_REGULAR_ARTIFACT_BYTES,
        )?;
        add_snapshot_bytes(&mut total, artifact.size_bytes)?;
    }
    for (token, artifact) in [
        (
            ManifestArtifactToken::AllocationsJson,
            artifacts.allocations_json.artifact(),
        ),
        (
            ManifestArtifactToken::TraceSamplingJson,
            artifacts.trace_sampling_json.artifact(),
        ),
    ] {
        let Some(artifact) = artifact else {
            continue;
        };
        require_resource_limit(
            token,
            "bytes",
            artifact.size_bytes,
            MAX_REGULAR_ARTIFACT_BYTES,
        )?;
        add_snapshot_bytes(&mut total, artifact.size_bytes)?;
    }
    Ok(())
}

fn verify_artifact(
    artifact: ManifestArtifactToken,
    expected: &ManifestArtifact,
    actual: ArtifactDigestIdentity,
) -> Result<(), ManifestError> {
    if expected.size_bytes != actual.size_bytes() {
        return Err(ManifestError::ArtifactSizeMismatch {
            artifact,
            expected: expected.size_bytes,
            actual: actual.size_bytes(),
        });
    }
    if expected.sha256 != actual.sha256() {
        return Err(ManifestError::ArtifactDigestMismatch {
            artifact,
            expected: expected.sha256,
            actual: actual.sha256(),
        });
    }
    Ok(())
}

fn verify_optional_artifact(
    artifact: ManifestArtifactToken,
    expected: &ManifestOptionalArtifact,
    actual: Option<ArtifactDigestIdentity>,
) -> Result<(), ManifestError> {
    match (expected.artifact(), actual) {
        (None, None) => Ok(()),
        (Some(expected), Some(actual)) => verify_artifact(artifact, expected, actual),
        (expected, actual) => Err(ManifestError::ArtifactPresenceMismatch {
            artifact,
            expected: expected.is_some(),
            actual: actual.is_some(),
        }),
    }
}

fn verify_retained_digest(
    artifact: ManifestArtifactToken,
    expected: ArtifactDigestIdentity,
    actual: ArtifactDigestIdentity,
) -> Result<(), ManifestError> {
    if expected.size_bytes() != actual.size_bytes() {
        return Err(ManifestError::ArtifactSizeMismatch {
            artifact,
            expected: expected.size_bytes(),
            actual: actual.size_bytes(),
        });
    }
    if expected.sha256() != actual.sha256() {
        return Err(ManifestError::ArtifactDigestMismatch {
            artifact,
            expected: expected.sha256(),
            actual: actual.sha256(),
        });
    }
    Ok(())
}

fn verify_retained_optional_digest(
    artifact: ManifestArtifactToken,
    expected: Option<ArtifactDigestIdentity>,
    actual: Option<ArtifactDigestIdentity>,
) -> Result<(), ManifestError> {
    match (expected, actual) {
        (None, None) => Ok(()),
        (Some(expected), Some(actual)) => verify_retained_digest(artifact, expected, actual),
        (expected, actual) => Err(ManifestError::ArtifactPresenceMismatch {
            artifact,
            expected: expected.is_some(),
            actual: actual.is_some(),
        }),
    }
}

fn verify_trace_artifact_identity(
    expected: &TraceArtifactIdentity,
    actual: &TraceArtifactIdentity,
) -> Result<(), ManifestError> {
    verify_retained_digest(
        ManifestArtifactToken::DynamicTracePb,
        expected.dynamic_trace_pb(),
        actual.dynamic_trace_pb(),
    )?;
    verify_retained_optional_digest(
        ManifestArtifactToken::AllocationsJson,
        expected.allocations_json(),
        actual.allocations_json(),
    )?;
    verify_retained_optional_digest(
        ManifestArtifactToken::TraceSamplingJson,
        expected.trace_sampling_json(),
        actual.trace_sampling_json(),
    )?;
    verify_retained_optional_digest(
        ManifestArtifactToken::StatsCsv,
        expected.stats_csv(),
        actual.stats_csv(),
    )?;

    let expected_payload = expected.threadblock_payload();
    let actual_payload = actual.threadblock_payload();
    if expected_payload.file_count() != actual_payload.file_count() {
        return Err(ManifestError::ArtifactCardinalityMismatch {
            artifact: ManifestArtifactToken::ThreadblockPayload,
            expected: expected_payload.file_count(),
            actual: actual_payload.file_count(),
        });
    }
    if expected_payload.raw_bytes() != actual_payload.raw_bytes() {
        return Err(ManifestError::ArtifactSizeMismatch {
            artifact: ManifestArtifactToken::ThreadblockPayload,
            expected: expected_payload.raw_bytes(),
            actual: actual_payload.raw_bytes(),
        });
    }
    if expected_payload.sorted_path_manifest_sha256()
        != actual_payload.sorted_path_manifest_sha256()
    {
        return Err(ManifestError::ArtifactDigestMismatch {
            artifact: ManifestArtifactToken::ThreadblockPayloadSortedPathManifest,
            expected: expected_payload.sorted_path_manifest_sha256(),
            actual: actual_payload.sorted_path_manifest_sha256(),
        });
    }
    if expected_payload.sorted_path_stream_sha256() != actual_payload.sorted_path_stream_sha256() {
        return Err(ManifestError::ArtifactDigestMismatch {
            artifact: ManifestArtifactToken::ThreadblockPayloadSortedPathStream,
            expected: expected_payload.sorted_path_stream_sha256(),
            actual: actual_payload.sorted_path_stream_sha256(),
        });
    }
    Ok(())
}

struct ObservedArtifactGeneration {
    artifact: ManifestArtifactToken,
    path: PathBuf,
    generation: FileGenerationIdentity,
}

fn read_manifest_artifact(
    reader: &AnchoredDir,
    base: &Path,
    file_identities: &mut BTreeMap<FileIdentity, ObservedArtifactGeneration>,
    artifact: ManifestArtifactToken,
    expected: &ManifestArtifact,
) -> Result<(PathBuf, Vec<u8>, ReadArtifactIdentity), ManifestError> {
    let path = resolve_manifest_path(base, artifact, &expected.path)?;
    let file = reader
        .read_exact_regular(Path::new(&expected.path), expected.size_bytes)
        .map_err(|source| match source {
            sim_trace::AnchoredReadError::SizeMismatch {
                expected, actual, ..
            } => ManifestError::ArtifactSizeMismatch {
                artifact,
                expected,
                actual,
            },
            other => ManifestError::ArtifactRead {
                artifact,
                source: other,
            },
        })?;
    insert_manifest_file_identity(
        file_identities,
        artifact,
        file.generation_identity(),
        path.clone(),
    )?;
    let raw = file.into_raw();
    let actual = ReadArtifactIdentity {
        size_bytes: u64::try_from(raw.len()).map_err(|_| ManifestError::ArtifactSizeMismatch {
            artifact,
            expected: expected.size_bytes,
            actual: u64::MAX,
        })?,
        sha256: Sha256Digest::of(&raw),
    };
    if expected.sha256 != actual.sha256 {
        return Err(ManifestError::ArtifactDigestMismatch {
            artifact,
            expected: expected.sha256,
            actual: actual.sha256,
        });
    }
    Ok((path, raw, actual))
}

fn trace_snapshot_artifact_token(
    trace_root: &Path,
    path: &Path,
) -> Result<ManifestArtifactToken, ManifestError> {
    for (expected, artifact) in [
        (
            trace_root.join("dynamic_trace.pb"),
            ManifestArtifactToken::DynamicTracePb,
        ),
        (
            trace_root.join("extra_info/allocations.json"),
            ManifestArtifactToken::AllocationsJson,
        ),
        (
            trace_root.join("trace_sampling.json"),
            ManifestArtifactToken::TraceSamplingJson,
        ),
        (
            trace_root.join("stats.csv"),
            ManifestArtifactToken::StatsCsv,
        ),
    ] {
        if path == expected {
            return Ok(artifact);
        }
    }
    if path.starts_with(trace_root.join("threadblocks")) {
        return Ok(ManifestArtifactToken::ThreadblockPayload);
    }
    Err(ManifestError::EdgeUnverifiable {
        edge: "trace_snapshot_artifact_generation",
        reason: "snapshotted trace artifact path has no closed manifest role",
    })
}

fn insert_manifest_file_identity(
    identities: &mut BTreeMap<FileIdentity, ObservedArtifactGeneration>,
    artifact: ManifestArtifactToken,
    generation: FileGenerationIdentity,
    path: PathBuf,
) -> Result<(), ManifestError> {
    if let Some(first) = identities.insert(
        generation.file_identity(),
        ObservedArtifactGeneration {
            artifact,
            path: path.clone(),
            generation,
        },
    ) {
        return Err(ManifestError::ArtifactFileAlias {
            artifact,
            first: first.path,
            second: path,
        });
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct ReadArtifactIdentity {
    size_bytes: u64,
    sha256: Sha256Digest,
}

impl From<ReadArtifactIdentity> for SealedArtifact {
    fn from(identity: ReadArtifactIdentity) -> Self {
        Self {
            size_bytes: identity.size_bytes,
            sha256: identity.sha256,
        }
    }
}

fn parse_trace_snapshot(
    trace_root: &Path,
    snapshot: TraceDirSnapshot,
) -> Result<TraceDir, ManifestError> {
    TraceDir::from_snapshot(snapshot).map_err(|source| match source {
        TraceError::TraceContractUnsupported { .. } => ManifestError::CapabilityMissing { source },
        other => map_trace_snapshot_error(trace_root, other),
    })
}

fn map_trace_snapshot_error(trace_root: &Path, source: TraceError) -> ManifestError {
    match source {
        TraceError::SnapshotArtifactPresenceMismatch {
            artifact,
            expected,
            actual,
        } => match ManifestArtifactToken::from_trace_snapshot_token(artifact) {
            Some(artifact) => ManifestError::ArtifactPresenceMismatch {
                artifact,
                expected,
                actual,
            },
            None => ManifestError::Trace(TraceError::SnapshotArtifactPresenceMismatch {
                artifact,
                expected,
                actual,
            }),
        },
        TraceError::SnapshotArtifactSizeMismatch {
            artifact,
            expected,
            actual,
        } => match ManifestArtifactToken::from_trace_snapshot_token(artifact) {
            Some(artifact) => ManifestError::ArtifactSizeMismatch {
                artifact,
                expected,
                actual,
            },
            None => ManifestError::Trace(TraceError::SnapshotArtifactSizeMismatch {
                artifact,
                expected,
                actual,
            }),
        },
        TraceError::SnapshotArtifactCardinalityMismatch {
            artifact,
            expected,
            actual,
        } => match ManifestArtifactToken::from_trace_snapshot_token(artifact) {
            Some(artifact) => ManifestError::ArtifactCardinalityMismatch {
                artifact,
                expected,
                actual,
            },
            None => ManifestError::Trace(TraceError::SnapshotArtifactCardinalityMismatch {
                artifact,
                expected,
                actual,
            }),
        },
        TraceError::TraceArtifactResourceLimitExceeded {
            path,
            artifact,
            resource,
            observed,
            limit,
        } => match ManifestArtifactToken::from_trace_snapshot_token(artifact) {
            Some(artifact) => ManifestError::ArtifactResourceLimitExceeded {
                artifact,
                resource,
                declared: observed,
                limit,
            },
            None => ManifestError::Trace(TraceError::TraceArtifactResourceLimitExceeded {
                path,
                artifact,
                resource,
                observed,
                limit,
            }),
        },
        TraceError::SnapshotArtifactRead { artifact, source } => {
            match ManifestArtifactToken::from_trace_snapshot_token(artifact) {
                Some(artifact) => match source {
                    sim_trace::AnchoredReadError::HardlinkAlias { first, second } => {
                        ManifestError::ArtifactFileAlias {
                            artifact,
                            first,
                            second,
                        }
                    }
                    source => ManifestError::ArtifactRead { artifact, source },
                },
                None => ManifestError::Trace(TraceError::SnapshotArtifactRead { artifact, source }),
            }
        }
        TraceError::AnchoredSnapshot(sim_trace::AnchoredReadError::HardlinkAlias {
            first,
            second,
        }) => match trace_snapshot_artifact_token(trace_root, &second) {
            Ok(artifact) => ManifestError::ArtifactFileAlias {
                artifact,
                first,
                second,
            },
            Err(_) => ManifestError::Trace(TraceError::AnchoredSnapshot(
                sim_trace::AnchoredReadError::HardlinkAlias { first, second },
            )),
        },
        other => ManifestError::Trace(other),
    }
}

fn identity_mismatch(
    artifact: ManifestArtifactToken,
    edge: &'static str,
    expected: impl ToString,
    actual: impl ToString,
) -> ManifestError {
    ManifestError::IdentityMismatch {
        artifact,
        edge,
        expected: expected.to_string(),
        actual: actual.to_string(),
    }
}

fn parse_required_sha256(
    artifact: ManifestArtifactToken,
    field: &'static str,
    value: &str,
) -> Result<Sha256Digest, ManifestError> {
    Sha256Digest::parse(value).map_err(|_| ManifestError::IdentityMismatch {
        artifact,
        edge: field,
        expected: "canonical lowercase SHA-256".to_owned(),
        actual: value.to_owned(),
    })
}

fn manifest_base(path: &Path) -> PathBuf {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf()
}

/// Admit one strict launch-evidence manifest and return only snapshotted inputs.
pub fn admit_launch_evidence(
    manifest_path: impl AsRef<Path>,
) -> Result<AdmittedLaunchEvidence, ManifestError> {
    let manifest_path = manifest_path.as_ref();
    let base = manifest_base(manifest_path);
    let manifest_name = manifest_path.file_name().ok_or_else(|| {
        ManifestError::Anchored(sim_trace::AnchoredReadError::PathInvalid {
            path: manifest_path.to_path_buf(),
        })
    })?;
    let reader = AnchoredDir::open(&base)?;
    admit_launch_evidence_anchored(&reader, Path::new(manifest_name))
}

/// Admit launch evidence relative to an already retained ancestor anchor.
///
/// The returned evidence retains a duplicate of this ancestor capability, not
/// only a child-directory descriptor, so final publication rechecks traverse the
/// same run-manifest-relative names that were admitted.
pub fn admit_launch_evidence_anchored(
    reader: &AnchoredDir,
    manifest_relative: &Path,
) -> Result<AdmittedLaunchEvidence, ManifestError> {
    let components = manifest_relative.components().collect::<Vec<_>>();
    if components.is_empty()
        || components
            .iter()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(ManifestError::Anchored(
            sim_trace::AnchoredReadError::PathInvalid {
                path: manifest_relative.to_path_buf(),
            },
        ));
    }
    let manifest_name = manifest_relative.file_name().ok_or_else(|| {
        ManifestError::Anchored(sim_trace::AnchoredReadError::PathInvalid {
            path: manifest_relative.to_path_buf(),
        })
    })?;
    let parent = manifest_relative.parent().unwrap_or_else(|| Path::new(""));
    if parent.as_os_str().is_empty() {
        return admit_launch_evidence_in_base(
            reader,
            Path::new(manifest_name),
            reader,
            Path::new(""),
        );
    }

    let evidence_reader = reader.descendant(parent)?;
    admit_launch_evidence_in_base(&evidence_reader, Path::new(manifest_name), reader, parent)
}

fn admit_launch_evidence_in_base(
    reader: &AnchoredDir,
    manifest_name: &Path,
    retained_root: &AnchoredDir,
    retained_prefix: &Path,
) -> Result<AdmittedLaunchEvidence, ManifestError> {
    let mut components = manifest_name.components();
    if !matches!(components.next(), Some(Component::Normal(_)))
        || components.next().is_some()
        || manifest_name.file_name() != Some(manifest_name.as_os_str())
    {
        return Err(ManifestError::Anchored(
            sim_trace::AnchoredReadError::PathInvalid {
                path: manifest_name.to_path_buf(),
            },
        ));
    }
    let base = reader.root().to_path_buf();
    let manifest_path = base.join(manifest_name);
    let manifest_file = reader.read_bounded_regular(manifest_name, MAX_MANIFEST_BYTES)?;
    let manifest_file_generation = manifest_file.generation_identity();
    let manifest_raw = manifest_file.into_raw();
    let envelope: ManifestVersionEnvelope =
        strict_json_from_slice(&manifest_raw).map_err(|source| ManifestError::Parse {
            path: manifest_path.to_path_buf(),
            source,
        })?;
    if envelope.schema_version != MANIFEST_SCHEMA_VERSION {
        return Err(ManifestError::SchemaVersionUnsupported {
            got: envelope.schema_version,
            expected: MANIFEST_SCHEMA_VERSION,
        });
    }
    let manifest: LaunchEvidenceManifest =
        strict_json_from_slice(&manifest_raw).map_err(|source| ManifestError::Parse {
            path: manifest_path.to_path_buf(),
            source,
        })?;
    let mut file_identities = BTreeMap::new();
    insert_manifest_file_identity(
        &mut file_identities,
        ManifestArtifactToken::LaunchEvidenceManifest,
        manifest_file_generation,
        manifest_path.to_path_buf(),
    )?;
    let artifacts = &manifest.artifacts;
    // Manifest sizes are claims, not resource authority. Reject independently
    // bounded per-artifact/cardinality and aggregate costs before any declared
    // size reaches a reserve/read path.
    validate_manifest_resource_limits(artifacts)?;

    let dynamic_path = resolve_manifest_path(
        &base,
        ManifestArtifactToken::DynamicTracePb,
        &artifacts.dynamic_trace_pb.path,
    )?;
    require_path(
        ManifestArtifactToken::DynamicTracePb,
        "dynamic_trace_pb.filename",
        dynamic_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("dynamic_trace.pb"),
        &dynamic_path,
    )?;
    let trace_root = dynamic_path
        .parent()
        .ok_or(ManifestError::IdentityFieldMissing {
            field: "dynamic_trace_pb parent",
        })?
        .to_path_buf();
    let payload_path = resolve_manifest_path(
        &base,
        ManifestArtifactToken::ThreadblockPayload,
        &artifacts.threadblock_payload.path,
    )?;
    require_path(
        ManifestArtifactToken::ThreadblockPayload,
        "threadblock_payload.trace_root",
        trace_root.join("threadblocks"),
        &payload_path,
    )?;
    let exec_path = resolve_manifest_path(
        &base,
        ManifestArtifactToken::EnhancedExecutionInfoJson,
        &artifacts.enhanced_execution_info_json.path,
    )?;
    require_path(
        ManifestArtifactToken::EnhancedExecutionInfoJson,
        "enhanced_execution_info_json.trace_root",
        trace_root.join("extra_info/enhanced_execution_info.json"),
        &exec_path,
    )?;
    if let Some(artifact) = artifacts.allocations_json.artifact() {
        let path = resolve_manifest_path(
            &base,
            ManifestArtifactToken::AllocationsJson,
            &artifact.path,
        )?;
        require_path(
            ManifestArtifactToken::AllocationsJson,
            "allocations_json.trace_root",
            trace_root.join("extra_info/allocations.json"),
            &path,
        )?;
    }
    if let Some(artifact) = artifacts.trace_sampling_json.artifact() {
        let path = resolve_manifest_path(
            &base,
            ManifestArtifactToken::TraceSamplingJson,
            &artifact.path,
        )?;
        require_path(
            ManifestArtifactToken::TraceSamplingJson,
            "trace_sampling_json.trace_root",
            trace_root.join("trace_sampling.json"),
            &path,
        )?;
    }
    let stats_path = resolve_manifest_path(
        &base,
        ManifestArtifactToken::StatsCsv,
        &artifacts.stats_csv.path,
    )?;
    require_path(
        ManifestArtifactToken::StatsCsv,
        "stats_csv.trace_root",
        trace_root.join("stats.csv"),
        &stats_path,
    )?;

    let control_path = resolve_manifest_path(
        &base,
        ManifestArtifactToken::ControlBitsSidecar,
        &artifacts.control_bits_sidecar.path,
    )?;
    let arch_card_path = resolve_manifest_path(
        &base,
        ManifestArtifactToken::ArchCard,
        &artifacts.arch_card.path,
    )?;
    let pchase_card_path = resolve_manifest_path(
        &base,
        ManifestArtifactToken::PchaseCard,
        &artifacts.pchase_card.path,
    )?;
    let mut distinct_paths = BTreeSet::new();
    for (artifact, path) in [
        (ManifestArtifactToken::DynamicTracePb, dynamic_path.clone()),
        (ManifestArtifactToken::StatsCsv, stats_path.clone()),
        (
            ManifestArtifactToken::EnhancedExecutionInfoJson,
            exec_path.clone(),
        ),
        (
            ManifestArtifactToken::ControlBitsSidecar,
            control_path.clone(),
        ),
        (ManifestArtifactToken::ArchCard, arch_card_path.clone()),
        (ManifestArtifactToken::PchaseCard, pchase_card_path.clone()),
    ] {
        if !distinct_paths.insert(path.clone()) {
            return Err(ManifestError::ArtifactPathDuplicate { artifact, path });
        }
    }
    for (token, optional) in [
        (
            ManifestArtifactToken::AllocationsJson,
            artifacts.allocations_json.artifact(),
        ),
        (
            ManifestArtifactToken::TraceSamplingJson,
            artifacts.trace_sampling_json.artifact(),
        ),
    ]
    .into_iter()
    .filter_map(|(token, artifact)| artifact.map(|artifact| (token, artifact)))
    {
        let path = resolve_manifest_path(&base, token, &optional.path)?;
        if !distinct_paths.insert(path.clone()) {
            return Err(ManifestError::ArtifactPathDuplicate {
                artifact: token,
                path,
            });
        }
    }

    let trace_root_relative = Path::new(&artifacts.dynamic_trace_pb.path).parent().ok_or(
        ManifestError::IdentityFieldMissing {
            field: "dynamic_trace_pb relative parent",
        },
    )?;
    let trace_bundle_path =
        CanonicalTraceBundlePath::checked(trace_root_relative.to_str().ok_or_else(|| {
            ManifestError::TraceBundlePathInvalid {
                path: trace_root_relative.to_string_lossy().into_owned(),
            }
        })?)?;
    let snapshot_limits = TraceSnapshotLimits::new(
        artifacts.dynamic_trace_pb.size_bytes,
        artifacts
            .allocations_json
            .artifact()
            .map(|artifact| artifact.size_bytes),
        artifacts
            .trace_sampling_json
            .artifact()
            .map(|artifact| artifact.size_bytes),
        Some(artifacts.stats_csv.size_bytes),
        artifacts.threadblock_payload.file_count,
        artifacts.threadblock_payload.raw_bytes,
    );
    let snapshot =
        TraceDirSnapshot::capture_anchored(&reader, trace_root_relative, snapshot_limits)
            .map_err(|source| map_trace_snapshot_error(&trace_root, source))?;
    for (generation, path) in snapshot.file_generations() {
        insert_manifest_file_identity(
            &mut file_identities,
            trace_snapshot_artifact_token(&trace_root, path)?,
            generation,
            path.to_path_buf(),
        )?;
    }
    let trace_artifacts = snapshot.artifact_identity();
    verify_artifact(
        ManifestArtifactToken::DynamicTracePb,
        &artifacts.dynamic_trace_pb,
        trace_artifacts.dynamic_trace_pb(),
    )?;
    verify_optional_artifact(
        ManifestArtifactToken::AllocationsJson,
        &artifacts.allocations_json,
        trace_artifacts.allocations_json(),
    )?;
    verify_optional_artifact(
        ManifestArtifactToken::TraceSamplingJson,
        &artifacts.trace_sampling_json,
        trace_artifacts.trace_sampling_json(),
    )?;
    let stats_identity =
        trace_artifacts
            .stats_csv()
            .ok_or(ManifestError::ArtifactPresenceMismatch {
                artifact: ManifestArtifactToken::StatsCsv,
                expected: true,
                actual: false,
            })?;
    verify_artifact(
        ManifestArtifactToken::StatsCsv,
        &artifacts.stats_csv,
        stats_identity,
    )?;
    let expected_payload = &artifacts.threadblock_payload;
    let actual_payload = trace_artifacts.threadblock_payload();
    if expected_payload.file_count != actual_payload.file_count() {
        return Err(ManifestError::ArtifactCardinalityMismatch {
            artifact: ManifestArtifactToken::ThreadblockPayload,
            expected: expected_payload.file_count,
            actual: actual_payload.file_count(),
        });
    }
    if expected_payload.raw_bytes != actual_payload.raw_bytes() {
        return Err(ManifestError::ArtifactSizeMismatch {
            artifact: ManifestArtifactToken::ThreadblockPayload,
            expected: expected_payload.raw_bytes,
            actual: actual_payload.raw_bytes(),
        });
    }
    if expected_payload.sorted_path_manifest_sha256 != actual_payload.sorted_path_manifest_sha256()
    {
        return Err(ManifestError::ArtifactDigestMismatch {
            artifact: ManifestArtifactToken::ThreadblockPayloadSortedPathManifest,
            expected: expected_payload.sorted_path_manifest_sha256,
            actual: actual_payload.sorted_path_manifest_sha256(),
        });
    }
    if expected_payload.sorted_path_stream_sha256 != actual_payload.sorted_path_stream_sha256() {
        return Err(ManifestError::ArtifactDigestMismatch {
            artifact: ManifestArtifactToken::ThreadblockPayloadSortedPathStream,
            expected: expected_payload.sorted_path_stream_sha256,
            actual: actual_payload.sorted_path_stream_sha256(),
        });
    }
    let trace_identity = trace_artifacts.clone();
    let trace_dir = parse_trace_snapshot(&trace_root, snapshot)?;
    if !matches!(
        trace_dir.memory_payload_contract(),
        MemoryPayloadContract::AddressCarryingV1
            | MemoryPayloadContract::AddressCarryingSharedMemorySplitV2
    ) {
        return Err(ManifestError::EdgeUnverifiable {
            edge: "trace_contract_id",
            reason: "legacy exclusion traces cannot mint launch evidence identity",
        });
    }
    let wire_summary = trace_dir.validation_summary();
    let trace_stats = trace_dir.stats().ok_or(ManifestError::EdgeUnverifiable {
        edge: "stats_csv",
        reason: "Q1009 launch admission requires snapshotted non-vacuous producer counters",
    })?;

    let (exec_path, exec_raw, exec_digest) = read_manifest_artifact(
        &reader,
        &base,
        &mut file_identities,
        ManifestArtifactToken::EnhancedExecutionInfoJson,
        &artifacts.enhanced_execution_info_json,
    )?;
    let exec = ExecInfo::parse(&exec_path, &exec_raw)?;
    if exec.benchmark_name.is_empty() {
        return Err(ManifestError::IdentityFieldMissing {
            field: "enhanced_execution_info_json.benchmark_name",
        });
    }
    let expected_control_stem = exec
        .benchmark_name
        .strip_suffix("_run")
        .unwrap_or(&exec.benchmark_name);
    let expected_control_name = format!("{expected_control_stem}.control_bits.json");
    if control_path.file_name().and_then(|name| name.to_str()) != Some(&expected_control_name) {
        return Err(identity_mismatch(
            ManifestArtifactToken::ControlBitsSidecar,
            "benchmark_name->control_bits_sidecar_stem",
            expected_control_name,
            control_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("<non-utf8>"),
        ));
    }

    let expected_arch = ArchId::parse_deploy(&manifest.architecture.arch_id)?;
    let expected_sku = Sku::parse(&manifest.architecture.sku)?;
    if expected_sku.arch() != expected_arch {
        return Err(identity_mismatch(
            ManifestArtifactToken::DynamicTracePb,
            "manifest sku->arch",
            expected_arch,
            expected_sku.arch(),
        ));
    }
    let (_, control_raw, control_digest) = read_manifest_artifact(
        &reader,
        &base,
        &mut file_identities,
        ManifestArtifactToken::ControlBitsSidecar,
        &artifacts.control_bits_sidecar,
    )?;
    let control = ControlSidecar::parse_for_arch(&control_path, &control_raw, expected_arch)?;
    if control.binary.is_empty() {
        return Err(ManifestError::IdentityFieldMissing {
            field: "control_bits_sidecar.binary",
        });
    }
    // A canonical sidecar name is the only producer-side join token available
    // while the trace contract carries no binary digest.  Bind its embedded
    // binary basename to the execution metadata before allowing the sidecar
    // to contribute scheduling controls; otherwise a resealed sidecar from a
    // different executable can pass the filename and digest checks.
    let actual_binary = Path::new(&control.binary)
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .ok_or_else(|| ManifestError::IdentityMismatch {
            artifact: ManifestArtifactToken::ControlBitsSidecar,
            edge: "benchmark_name->control_bits_sidecar.binary",
            expected: format!("{:?} or {expected_control_stem:?}", exec.benchmark_name),
            actual: control.binary.clone(),
        })?;
    if actual_binary != exec.benchmark_name && actual_binary != expected_control_stem {
        return Err(identity_mismatch(
            ManifestArtifactToken::ControlBitsSidecar,
            "benchmark_name->control_bits_sidecar.binary",
            format!("{:?} or {expected_control_stem:?}", exec.benchmark_name),
            actual_binary,
        ));
    }
    let schedules = build_schedules(&exec, &control)?;

    let (_, arch_card_raw, arch_card_digest) = read_manifest_artifact(
        &reader,
        &base,
        &mut file_identities,
        ManifestArtifactToken::ArchCard,
        &artifacts.arch_card,
    )?;
    let arch_card = ArchCard::parse(&arch_card_path, &arch_card_raw)?;
    let card_arch = arch_card.arch_id()?;
    let card_sku = arch_card.sku_id()?;
    if card_arch != expected_arch {
        return Err(identity_mismatch(
            ManifestArtifactToken::ArchCard,
            "manifest arch->architecture card",
            expected_arch,
            card_arch,
        ));
    }
    if card_sku != expected_sku {
        return Err(identity_mismatch(
            ManifestArtifactToken::ArchCard,
            "manifest sku->architecture card",
            expected_sku.token(),
            card_sku.token(),
        ));
    }
    let (_, pchase_card_raw, pchase_card_digest) = read_manifest_artifact(
        &reader,
        &base,
        &mut file_identities,
        ManifestArtifactToken::PchaseCard,
        &artifacts.pchase_card,
    )?;
    let pchase_card = PchaseCard::parse(&pchase_card_path, &pchase_card_raw)?;
    if pchase_card.arch() != expected_arch {
        return Err(identity_mismatch(
            ManifestArtifactToken::PchaseCard,
            "manifest arch->P-chase card",
            expected_arch,
            pchase_card.arch(),
        ));
    }
    let trace_arch_version = u32::try_from(trace_dir.trace().binary_version).map_err(|_| {
        ManifestError::IdentityMismatch {
            artifact: ManifestArtifactToken::DynamicTracePb,
            edge: "trace binary_version->architecture",
            expected: expected_arch.numeric().to_string(),
            actual: trace_dir.trace().binary_version.to_string(),
        }
    })?;
    let trace_arch = ArchId::from_numeric(trace_arch_version)?;
    if trace_arch != expected_arch {
        return Err(identity_mismatch(
            ManifestArtifactToken::DynamicTracePb,
            "trace binary_version->manifest arch",
            expected_arch,
            trace_arch,
        ));
    }

    let selector = &manifest.launch_selector;
    let mut matching = trace_dir
        .kernels()
        .filter(|loc| {
            loc.device_id() == selector.device_id
                && loc.stream_id() == selector.stream_id
                && loc.kernel().id == selector.kernel_id
        })
        .collect::<Vec<_>>();
    if matching.is_empty() {
        return Err(ManifestError::LaunchMissing {
            device_id: selector.device_id,
            stream_id: selector.stream_id,
            kernel_id: selector.kernel_id,
        });
    }
    if matching.len() != 1 {
        return Err(ManifestError::LaunchAmbiguous {
            device_id: selector.device_id,
            stream_id: selector.stream_id,
            kernel_id: selector.kernel_id,
        });
    }
    let loc = matching.pop().expect("nonempty launch match was checked");
    if loc.kernel().function_unique_id != selector.function_unique_id {
        return Err(identity_mismatch(
            ManifestArtifactToken::DynamicTracePb,
            "manifest function_unique_id->root launch",
            selector.function_unique_id,
            loc.kernel().function_unique_id,
        ));
    }
    if loc.kernel().name.is_empty() {
        return Err(ManifestError::IdentityFieldMissing {
            field: "dynamic_trace.kernel.name",
        });
    }
    let exec_kernel = exec.kernels.get(&selector.function_unique_id).ok_or(
        ManifestError::IdentityFieldMissing {
            field: "enhanced_execution_info_json function_unique_id",
        },
    )?;
    if exec_kernel.kernel_name != loc.kernel().name {
        return Err(identity_mismatch(
            ManifestArtifactToken::EnhancedExecutionInfoJson,
            "root kernel name->exec-info kernel name",
            &loc.kernel().name,
            &exec_kernel.kernel_name,
        ));
    }
    let schedule =
        schedules
            .get(&selector.function_unique_id)
            .ok_or(ManifestError::IdentityFieldMissing {
                field: "joined function schedule",
            })?;
    if schedule.arch != expected_arch {
        return Err(identity_mismatch(
            ManifestArtifactToken::ControlBitsSidecar,
            "schedule arch->manifest arch",
            expected_arch,
            schedule.arch,
        ));
    }

    let coverage = admit_launch_coverage(&trace_dir, &loc, schedule, &arch_card)?;
    let launch_binding = match &coverage {
        ExtractedCoverage::Full(full) => full.launch().clone(),
        ExtractedCoverage::Sampled(sampled) => sampled.launch().clone(),
    };
    let block_dim = launch_binding
        .block()
        .ok_or(ManifestError::IdentityFieldMissing {
            field: "dynamic_trace.kernel.block_dim",
        })?;
    let replay_launch = ReplayAuthenticatedLaunchIdentity::checked(&launch_binding, expected_arch)?;
    let coverage_identity = match coverage {
        ExtractedCoverage::Full(full) => match full.source() {
            FullCoverageSource::UnsampledComplete => CoverageIdentity::ExactFull {
                blocks_observed: full.blocks_observed(),
            },
            FullCoverageSource::LosslessCompressedFull(provenance) => {
                CoverageIdentity::LosslessStored {
                    blocks_observed: full.blocks_observed(),
                    sidecar_sha256: parse_required_sha256(
                        ManifestArtifactToken::TraceSamplingJson,
                        "trace_sampling_json.sidecar_sha256",
                        provenance.sidecar_sha256(),
                    )?,
                }
            }
        },
        ExtractedCoverage::Sampled(sampled) => CoverageIdentity::EstimatedSampled {
            population_blocks: sampled.population_blocks(),
            blocks_observed: sampled.blocks_observed(),
            selected_block_ids: sampled
                .provenance()
                .selected_block_ids()
                .iter()
                .copied()
                .collect(),
            selected_block_ids_sha256: parse_required_sha256(
                ManifestArtifactToken::TraceSamplingJson,
                "trace_sampling_json.selected_block_ids_sha256",
                sampled.provenance().selected_block_ids_sha256(),
            )?,
            sidecar_sha256: parse_required_sha256(
                ManifestArtifactToken::TraceSamplingJson,
                "trace_sampling_json.sidecar_sha256",
                sampled.provenance().sidecar_sha256(),
            )?,
        },
    };

    let (cell_id, args_hash, sampling_seal) = match trace_dir.sampling_sidecar() {
        None => (None, None, None),
        Some(sidecar) => {
            if sidecar.cell_id().is_empty() {
                return Err(ManifestError::IdentityFieldMissing {
                    field: "trace_sampling_json.cell_id",
                });
            }
            if sidecar.kernel_args_hash().is_empty() {
                return Err(ManifestError::IdentityFieldMissing {
                    field: "trace_sampling_json.kernel.args_hash",
                });
            }
            let seal = match sidecar.method() {
                sim_trace::TraceSamplingMethod::LosslessCompressedFull(provenance) => {
                    provenance.sidecar_sha256()
                }
                sim_trace::TraceSamplingMethod::WaveSegmentStratifiedSubset(provenance) => {
                    provenance.sidecar_sha256()
                }
            };
            (
                Some(sidecar.cell_id().to_owned()),
                Some(sidecar.kernel_args_hash().to_owned()),
                Some(parse_required_sha256(
                    ManifestArtifactToken::TraceSamplingJson,
                    "trace_sampling_json.sidecar_sha256",
                    seal,
                )?),
            )
        }
    };

    let binary_digest = match manifest.binary_digest {
        ManifestBinaryDigest::TypedGap { gap } => {
            if !matches!(
                trace_dir.memory_payload_contract(),
                MemoryPayloadContract::AddressCarryingV1
                    | MemoryPayloadContract::AddressCarryingSharedMemorySplitV2
            ) {
                return Err(ManifestError::EdgeUnverifiable {
                    edge: "binary.digest",
                    reason: "typed binary-digest gap is valid only for an address-carrying trace contract",
                });
            }
            BinaryDigestIdentity::TypedGap { gap }
        }
        ManifestBinaryDigest::Value { sha256 } => {
            let _ = sha256;
            return Err(ManifestError::EdgeUnverifiable {
                edge: "binary.digest",
                reason: "the current trace producer emits no traced-workload binary digest",
            });
        }
    };

    let trace_bundle_components = TraceBundleComponents {
        root_dynamic_trace: TraceBundleComponentIdentity::checked(
            "components.root_dynamic_trace.size_bytes",
            trace_identity.dynamic_trace_pb().size_bytes(),
            trace_identity.dynamic_trace_pb().sha256(),
        )?,
        threadblock_payloads: ThreadblockPayloadSetIdentity::checked(
            trace_identity.threadblock_payload().file_count(),
            trace_identity.threadblock_payload().raw_bytes(),
            trace_identity
                .threadblock_payload()
                .sorted_path_manifest_sha256(),
            trace_identity
                .threadblock_payload()
                .sorted_path_stream_sha256(),
        )?,
        execution_metadata: TraceBundleComponentIdentity::checked(
            "components.execution_metadata.size_bytes",
            exec_digest.size_bytes,
            exec_digest.sha256,
        )?,
        control_bits_sidecar: TraceBundleComponentIdentity::checked(
            "components.control_bits_sidecar.size_bytes",
            control_digest.size_bytes,
            control_digest.sha256,
        )?,
        stats: TraceBundleComponentIdentity::checked(
            "components.stats.size_bytes",
            stats_identity.size_bytes(),
            stats_identity.sha256(),
        )?,
    };
    let trace_bundle = TraceBundleIdentity::checked(
        trace_bundle_path,
        WIRE_VALIDATOR_CONTRACT_ID,
        wire_summary.launch_count(),
        wire_summary.threadblock_count(),
        wire_summary.instruction_count(),
        trace_stats.admitted_record_count(),
        trace_bundle_components,
    )?;

    let memory_payload_classes = ["global", "shared", "local", "atomic", "async_copy"]
        .into_iter()
        .map(str::to_owned)
        .collect();
    let payload = LaunchEvidencePayload {
        schema_version: MANIFEST_SCHEMA_VERSION,
        cell_id,
        kernel: KernelIdentity {
            mangled_name: loc.kernel().name.clone(),
            args_hash,
        },
        launch: LaunchIdentity {
            device_id: launch_binding.device_id(),
            stream_id: launch_binding.stream_id(),
            kernel_id: launch_binding.kernel_id(),
            function_unique_id: launch_binding.function_unique_id(),
        },
        geometry: GeometryIdentity {
            grid_dim: launch_binding.grid(),
            block_dim,
            cluster_dim: launch_binding.cluster(),
        },
        trace_contract_id: trace_dir.trace().trace_contract_id.clone(),
        memory_payload_classes,
        wire_admission: WireAdmissionIdentity {
            validator_contract_id: WIRE_VALIDATOR_CONTRACT_ID.to_owned(),
            launch_count: wire_summary.launch_count(),
            threadblock_count: wire_summary.threadblock_count(),
            instruction_count: wire_summary.instruction_count(),
            stats_row_count: trace_stats.row_count(),
            stats_expected_record_count: trace_stats.expected_record_count(),
            stats_admitted_record_count: trace_stats.admitted_record_count(),
        },
        artifact_digests: ArtifactDigests {
            dynamic_trace_pb: trace_identity.dynamic_trace_pb().into(),
            threadblock_payload_file_count: trace_identity.threadblock_payload().file_count(),
            threadblock_payload_raw_bytes: trace_identity.threadblock_payload().raw_bytes(),
            threadblock_payload_manifest_sha256: trace_identity
                .threadblock_payload()
                .sorted_path_manifest_sha256(),
            threadblock_payload_sha256: trace_identity
                .threadblock_payload()
                .sorted_path_stream_sha256(),
            allocations_json: trace_identity.allocations_json().map(Into::into),
            trace_sampling_json: trace_identity.trace_sampling_json().map(Into::into),
            trace_sampling_json_seal: sampling_seal,
            stats_csv: stats_identity.into(),
            enhanced_execution_info_json: exec_digest.into(),
            control_bits_sidecar: control_digest.into(),
        },
        arch: ArchitectureIdentity {
            arch_id: expected_arch.token().to_owned(),
            sku: expected_sku.token().to_owned(),
            arch_card: arch_card_digest.into(),
            pchase_card: pchase_card_digest.into(),
        },
        binary: BinaryIdentity {
            name: control.binary,
            digest: binary_digest,
        },
        coverage: coverage_identity,
    };
    let identity = LaunchEvidenceIdentity::seal(payload, trace_bundle, replay_launch)?;
    let retained_generation = RetainedEvidenceGeneration::capture(
        reader,
        file_identities,
        RetainedTraceInventory {
            root_relative: trace_root_relative.to_path_buf(),
            limits: snapshot_limits,
            expected: trace_identity,
        },
    )?
    .reanchor(retained_root, retained_prefix)?;
    Ok(AdmittedLaunchEvidence {
        identity,
        trace_dir,
        schedules,
        arch_card,
        pchase_card,
        retained_generation,
    })
}
