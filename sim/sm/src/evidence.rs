//! Manifest-only admission for one launch's transitive simulator evidence.
//!
//! This boundary snapshots and hashes raw artifacts before parsing them, proves
//! the existing trace/exec/control/card/coverage joins, and only then constructs
//! a canonical JSON identity seal. The traced workload binary is deliberately a
//! typed gap: the current trace contract has no producer-bound binary digest.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use sim_arch::{ArchError, ArchId, Sku};
use sim_calib::{ArchCard, CalibError};
use sim_mem::{PchaseCard, PchaseCardError};
use sim_trace::reader::{ADDRESS_CARRYING_TRACE_CONTRACT_ID, MemoryPayloadContract};
use sim_trace::{
    AnchoredDir, ArtifactDigestIdentity, CanonicalJsonError, ExecInfo, ExecInfoError, FileIdentity,
    Sha256Digest, TraceDir, TraceDirSnapshot, TraceError, TraceSnapshotLimits, canonical_json,
};

use crate::control_bits::{ControlBitsError, ControlSidecar};
use crate::program::{FunctionSchedule, ProgramError, build_schedules};
use crate::replay::{ExtractedCoverage, FullCoverageSource, ReplayError, admit_launch_coverage};

const MANIFEST_SCHEMA_VERSION: u32 = 2;
const WIRE_VALIDATOR_CONTRACT_ID: &str = "sim_trace_wire_v1";
const MAX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
const MAX_REGULAR_ARTIFACT_BYTES: u64 = 64 * 1024 * 1024;
const MAX_THREADBLOCK_PAYLOAD_FILES: u64 = 1_000_000;
const MAX_THREADBLOCK_PAYLOAD_BYTES: u64 = 768 * 1024 * 1024;
const MAX_EVIDENCE_SNAPSHOT_BYTES: u64 = 1024 * 1024 * 1024;

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
        artifact: &'static str,
        path: String,
    },
    #[error("manifest artifact path edge {edge} is invalid: expected {expected}, got {actual}")]
    ArtifactPathMismatch {
        edge: &'static str,
        expected: PathBuf,
        actual: PathBuf,
    },
    #[error("descriptor-anchored evidence read failed: {0}")]
    Anchored(#[from] sim_trace::AnchoredReadError),
    #[error("manifest artifact paths alias one another at {path}")]
    ArtifactPathDuplicate { path: PathBuf },
    #[error("manifest artifacts {first} and {second} alias the same inode")]
    ArtifactFileAlias { first: PathBuf, second: PathBuf },
    #[error("artifact {artifact} byte size mismatch: manifest {expected}, snapshot {actual}")]
    ArtifactSizeMismatch {
        artifact: &'static str,
        expected: u64,
        actual: u64,
    },
    #[error("artifact {artifact} SHA-256 mismatch: manifest {expected}, snapshot {actual}")]
    ArtifactDigestMismatch {
        artifact: &'static str,
        expected: Sha256Digest,
        actual: Sha256Digest,
    },
    #[error(
        "artifact {artifact} presence mismatch: manifest declares {expected}, snapshot has {actual}"
    )]
    ArtifactPresenceMismatch {
        artifact: &'static str,
        expected: bool,
        actual: bool,
    },
    #[error("artifact {artifact} file count mismatch: manifest {expected}, snapshot {actual}")]
    ArtifactCardinalityMismatch {
        artifact: &'static str,
        expected: u64,
        actual: u64,
    },
    #[error(
        "artifact {artifact} declares {declared} {resource}, exceeding implementation limit {limit}"
    )]
    ArtifactResourceLimitExceeded {
        artifact: &'static str,
        resource: &'static str,
        declared: u64,
        limit: u64,
    },
    #[error(
        "launch-evidence manifest declares {declared} aggregate artifact bytes, exceeding implementation limit {limit}"
    )]
    SnapshotResourceLimitExceeded { declared: u64, limit: u64 },
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
    #[error("identity edge {edge} mismatch: expected {expected:?}, got {actual:?}")]
    IdentityMismatch {
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
    seal: Sha256Digest,
    canonical_json: String,
}

impl LaunchEvidenceIdentity {
    fn seal(payload: LaunchEvidencePayload) -> Result<Self, ManifestError> {
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

    pub fn binary_digest_gap(&self) -> BinaryDigestGap {
        match self.payload.binary.digest {
            BinaryDigestIdentity::TypedGap { gap } => gap,
        }
    }
}

/// Exact snapshotted inputs paired with their self-sealed launch identity.
pub struct AdmittedLaunchEvidence {
    identity: LaunchEvidenceIdentity,
    trace_dir: TraceDir,
    schedules: BTreeMap<i32, FunctionSchedule>,
    arch_card: ArchCard,
    pchase_card: PchaseCard,
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
}

fn resolve_manifest_path(
    base: &Path,
    artifact: &'static str,
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

fn require_path(edge: &'static str, expected: PathBuf, actual: &Path) -> Result<(), ManifestError> {
    if actual != expected {
        return Err(ManifestError::ArtifactPathMismatch {
            edge,
            expected,
            actual: actual.to_path_buf(),
        });
    }
    Ok(())
}

fn require_resource_limit(
    artifact: &'static str,
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
        "threadblock_payload",
        "files",
        artifacts.threadblock_payload.file_count,
        MAX_THREADBLOCK_PAYLOAD_FILES,
    )?;
    require_resource_limit(
        "threadblock_payload",
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
    for (name, artifact) in [
        ("dynamic_trace_pb", &artifacts.dynamic_trace_pb),
        ("stats_csv", &artifacts.stats_csv),
        (
            "enhanced_execution_info_json",
            &artifacts.enhanced_execution_info_json,
        ),
        ("control_bits_sidecar", &artifacts.control_bits_sidecar),
        ("arch_card", &artifacts.arch_card),
        ("pchase_card", &artifacts.pchase_card),
    ] {
        require_resource_limit(
            name,
            "bytes",
            artifact.size_bytes,
            MAX_REGULAR_ARTIFACT_BYTES,
        )?;
        add_snapshot_bytes(&mut total, artifact.size_bytes)?;
    }
    for (name, artifact) in [
        ("allocations_json", artifacts.allocations_json.artifact()),
        (
            "trace_sampling_json",
            artifacts.trace_sampling_json.artifact(),
        ),
    ] {
        let Some(artifact) = artifact else {
            continue;
        };
        require_resource_limit(
            name,
            "bytes",
            artifact.size_bytes,
            MAX_REGULAR_ARTIFACT_BYTES,
        )?;
        add_snapshot_bytes(&mut total, artifact.size_bytes)?;
    }
    Ok(())
}

fn verify_artifact(
    artifact: &'static str,
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
    artifact: &'static str,
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

fn read_manifest_artifact(
    reader: &AnchoredDir,
    base: &Path,
    file_identities: &mut BTreeMap<FileIdentity, PathBuf>,
    artifact: &'static str,
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
            other => ManifestError::Anchored(other),
        })?;
    insert_manifest_file_identity(file_identities, file.identity(), path.clone())?;
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

fn insert_manifest_file_identity(
    identities: &mut BTreeMap<FileIdentity, PathBuf>,
    identity: FileIdentity,
    path: PathBuf,
) -> Result<(), ManifestError> {
    if let Some(first) = identities.insert(identity, path.clone()) {
        return Err(ManifestError::ArtifactFileAlias {
            first,
            second: path,
        });
    }
    Ok(())
}

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

fn parse_trace_snapshot(snapshot: TraceDirSnapshot) -> Result<TraceDir, ManifestError> {
    TraceDir::from_snapshot(snapshot).map_err(|source| match source {
        TraceError::TraceContractUnsupported { .. }
        | TraceError::MemoryPayloadCapabilitiesInvalid { .. } => {
            ManifestError::CapabilityMissing { source }
        }
        other => ManifestError::Trace(other),
    })
}

fn map_trace_snapshot_error(source: TraceError) -> ManifestError {
    match source {
        TraceError::SnapshotArtifactPresenceMismatch {
            artifact,
            expected,
            actual,
        } => ManifestError::ArtifactPresenceMismatch {
            artifact,
            expected,
            actual,
        },
        TraceError::SnapshotArtifactSizeMismatch {
            artifact,
            expected,
            actual,
        } => ManifestError::ArtifactSizeMismatch {
            artifact,
            expected,
            actual,
        },
        TraceError::SnapshotArtifactCardinalityMismatch {
            artifact,
            expected,
            actual,
        } => ManifestError::ArtifactCardinalityMismatch {
            artifact,
            expected,
            actual,
        },
        TraceError::AnchoredSnapshot(sim_trace::AnchoredReadError::HardlinkAlias {
            first,
            second,
        }) => ManifestError::ArtifactFileAlias { first, second },
        other => ManifestError::Trace(other),
    }
}

fn identity_mismatch(
    edge: &'static str,
    expected: impl ToString,
    actual: impl ToString,
) -> ManifestError {
    ManifestError::IdentityMismatch {
        edge,
        expected: expected.to_string(),
        actual: actual.to_string(),
    }
}

fn parse_required_sha256(field: &'static str, value: &str) -> Result<Sha256Digest, ManifestError> {
    Sha256Digest::parse(value).map_err(|_| ManifestError::IdentityMismatch {
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
    let manifest_file =
        reader.read_bounded_regular(Path::new(manifest_name), MAX_MANIFEST_BYTES)?;
    let manifest_file_identity = manifest_file.identity();
    let manifest_raw = manifest_file.into_raw();
    let envelope: ManifestVersionEnvelope =
        serde_json::from_slice(&manifest_raw).map_err(|source| ManifestError::Parse {
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
        serde_json::from_slice(&manifest_raw).map_err(|source| ManifestError::Parse {
            path: manifest_path.to_path_buf(),
            source,
        })?;
    let mut file_identities = BTreeMap::new();
    insert_manifest_file_identity(
        &mut file_identities,
        manifest_file_identity,
        manifest_path.to_path_buf(),
    )?;
    let artifacts = &manifest.artifacts;
    // Manifest sizes are claims, not resource authority. Reject independently
    // bounded per-artifact/cardinality and aggregate costs before any declared
    // size reaches a reserve/read path.
    validate_manifest_resource_limits(artifacts)?;

    let dynamic_path =
        resolve_manifest_path(&base, "dynamic_trace_pb", &artifacts.dynamic_trace_pb.path)?;
    require_path(
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
        "threadblock_payload",
        &artifacts.threadblock_payload.path,
    )?;
    require_path(
        "threadblock_payload.trace_root",
        trace_root.join("threadblocks"),
        &payload_path,
    )?;
    let exec_path = resolve_manifest_path(
        &base,
        "enhanced_execution_info_json",
        &artifacts.enhanced_execution_info_json.path,
    )?;
    require_path(
        "enhanced_execution_info_json.trace_root",
        trace_root.join("extra_info/enhanced_execution_info.json"),
        &exec_path,
    )?;
    if let Some(artifact) = artifacts.allocations_json.artifact() {
        let path = resolve_manifest_path(&base, "allocations_json", &artifact.path)?;
        require_path(
            "allocations_json.trace_root",
            trace_root.join("extra_info/allocations.json"),
            &path,
        )?;
    }
    if let Some(artifact) = artifacts.trace_sampling_json.artifact() {
        let path = resolve_manifest_path(&base, "trace_sampling_json", &artifact.path)?;
        require_path(
            "trace_sampling_json.trace_root",
            trace_root.join("trace_sampling.json"),
            &path,
        )?;
    }
    let stats_path = resolve_manifest_path(&base, "stats_csv", &artifacts.stats_csv.path)?;
    require_path(
        "stats_csv.trace_root",
        trace_root.join("stats.csv"),
        &stats_path,
    )?;

    let control_path = resolve_manifest_path(
        &base,
        "control_bits_sidecar",
        &artifacts.control_bits_sidecar.path,
    )?;
    let arch_card_path = resolve_manifest_path(&base, "arch_card", &artifacts.arch_card.path)?;
    let pchase_card_path =
        resolve_manifest_path(&base, "pchase_card", &artifacts.pchase_card.path)?;
    let mut distinct_paths = BTreeSet::new();
    for path in [
        dynamic_path.clone(),
        stats_path.clone(),
        exec_path.clone(),
        control_path.clone(),
        arch_card_path.clone(),
        pchase_card_path.clone(),
    ] {
        if !distinct_paths.insert(path.clone()) {
            return Err(ManifestError::ArtifactPathDuplicate { path });
        }
    }
    for optional in [
        artifacts.allocations_json.artifact(),
        artifacts.trace_sampling_json.artifact(),
    ]
    .into_iter()
    .flatten()
    {
        let path = resolve_manifest_path(&base, "optional_artifact", &optional.path)?;
        if !distinct_paths.insert(path.clone()) {
            return Err(ManifestError::ArtifactPathDuplicate { path });
        }
    }

    let trace_root_relative = Path::new(&artifacts.dynamic_trace_pb.path).parent().ok_or(
        ManifestError::IdentityFieldMissing {
            field: "dynamic_trace_pb relative parent",
        },
    )?;
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
            .map_err(map_trace_snapshot_error)?;
    for (identity, path) in snapshot.file_identities() {
        insert_manifest_file_identity(&mut file_identities, identity, path.to_path_buf())?;
    }
    let trace_artifacts = snapshot.artifact_identity();
    verify_artifact(
        "dynamic_trace_pb",
        &artifacts.dynamic_trace_pb,
        trace_artifacts.dynamic_trace_pb(),
    )?;
    verify_optional_artifact(
        "allocations_json",
        &artifacts.allocations_json,
        trace_artifacts.allocations_json(),
    )?;
    verify_optional_artifact(
        "trace_sampling_json",
        &artifacts.trace_sampling_json,
        trace_artifacts.trace_sampling_json(),
    )?;
    let stats_identity =
        trace_artifacts
            .stats_csv()
            .ok_or(ManifestError::ArtifactPresenceMismatch {
                artifact: "stats_csv",
                expected: true,
                actual: false,
            })?;
    verify_artifact("stats_csv", &artifacts.stats_csv, stats_identity)?;
    let expected_payload = &artifacts.threadblock_payload;
    let actual_payload = trace_artifacts.threadblock_payload();
    if expected_payload.file_count != actual_payload.file_count() {
        return Err(ManifestError::ArtifactCardinalityMismatch {
            artifact: "threadblock_payload",
            expected: expected_payload.file_count,
            actual: actual_payload.file_count(),
        });
    }
    if expected_payload.raw_bytes != actual_payload.raw_bytes() {
        return Err(ManifestError::ArtifactSizeMismatch {
            artifact: "threadblock_payload",
            expected: expected_payload.raw_bytes,
            actual: actual_payload.raw_bytes(),
        });
    }
    if expected_payload.sorted_path_manifest_sha256 != actual_payload.sorted_path_manifest_sha256()
    {
        return Err(ManifestError::ArtifactDigestMismatch {
            artifact: "threadblock_payload.sorted_path_manifest",
            expected: expected_payload.sorted_path_manifest_sha256,
            actual: actual_payload.sorted_path_manifest_sha256(),
        });
    }
    if expected_payload.sorted_path_stream_sha256 != actual_payload.sorted_path_stream_sha256() {
        return Err(ManifestError::ArtifactDigestMismatch {
            artifact: "threadblock_payload.sorted_path_stream",
            expected: expected_payload.sorted_path_stream_sha256,
            actual: actual_payload.sorted_path_stream_sha256(),
        });
    }
    let trace_identity = trace_artifacts.clone();
    let trace_dir = parse_trace_snapshot(snapshot)?;
    if trace_dir.memory_payload_contract() != MemoryPayloadContract::AddressCarryingV1 {
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
        "enhanced_execution_info_json",
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
            "manifest sku->arch",
            expected_arch,
            expected_sku.arch(),
        ));
    }
    let (_, control_raw, control_digest) = read_manifest_artifact(
        &reader,
        &base,
        &mut file_identities,
        "control_bits_sidecar",
        &artifacts.control_bits_sidecar,
    )?;
    let control = ControlSidecar::parse_for_arch(&control_path, &control_raw, expected_arch)?;
    if control.binary.is_empty() {
        return Err(ManifestError::IdentityFieldMissing {
            field: "control_bits_sidecar.binary",
        });
    }
    let schedules = build_schedules(&exec, &control)?;

    let (_, arch_card_raw, arch_card_digest) = read_manifest_artifact(
        &reader,
        &base,
        &mut file_identities,
        "arch_card",
        &artifacts.arch_card,
    )?;
    let arch_card = ArchCard::parse(&arch_card_path, &arch_card_raw)?;
    let card_arch = arch_card.arch_id()?;
    let card_sku = arch_card.sku_id()?;
    if card_arch != expected_arch {
        return Err(identity_mismatch(
            "manifest arch->architecture card",
            expected_arch,
            card_arch,
        ));
    }
    if card_sku != expected_sku {
        return Err(identity_mismatch(
            "manifest sku->architecture card",
            expected_sku.token(),
            card_sku.token(),
        ));
    }
    let (_, pchase_card_raw, pchase_card_digest) = read_manifest_artifact(
        &reader,
        &base,
        &mut file_identities,
        "pchase_card",
        &artifacts.pchase_card,
    )?;
    let pchase_card = PchaseCard::parse(&pchase_card_path, &pchase_card_raw)?;
    if pchase_card.arch() != expected_arch {
        return Err(identity_mismatch(
            "manifest arch->P-chase card",
            expected_arch,
            pchase_card.arch(),
        ));
    }
    let trace_arch_version = u32::try_from(trace_dir.trace().binary_version).map_err(|_| {
        ManifestError::IdentityMismatch {
            edge: "trace binary_version->architecture",
            expected: expected_arch.numeric().to_string(),
            actual: trace_dir.trace().binary_version.to_string(),
        }
    })?;
    let trace_arch = ArchId::from_numeric(trace_arch_version)?;
    if trace_arch != expected_arch {
        return Err(identity_mismatch(
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
    let coverage_identity = match coverage {
        ExtractedCoverage::Full(full) => match full.source() {
            FullCoverageSource::UnsampledComplete => CoverageIdentity::ExactFull {
                blocks_observed: full.blocks_observed(),
            },
            FullCoverageSource::LosslessCompressedFull(provenance) => {
                CoverageIdentity::LosslessStored {
                    blocks_observed: full.blocks_observed(),
                    sidecar_sha256: parse_required_sha256(
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
                "trace_sampling_json.selected_block_ids_sha256",
                sampled.provenance().selected_block_ids_sha256(),
            )?,
            sidecar_sha256: parse_required_sha256(
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
                    "trace_sampling_json.sidecar_sha256",
                    seal,
                )?),
            )
        }
    };

    let binary_digest = match manifest.binary_digest {
        ManifestBinaryDigest::TypedGap { gap } => {
            if trace_dir.trace().trace_contract_id != ADDRESS_CARRYING_TRACE_CONTRACT_ID {
                return Err(ManifestError::EdgeUnverifiable {
                    edge: "binary.digest",
                    reason: "typed legacy gap is valid only for the current address-carrying trace contract",
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
    let identity = LaunchEvidenceIdentity::seal(payload)?;
    Ok(AdmittedLaunchEvidence {
        identity,
        trace_dir,
        schedules,
        arch_card,
        pchase_card,
    })
}
