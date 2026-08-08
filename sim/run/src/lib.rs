//! Manifest-only production composition for admitted simulator launches.
//!
//! The runner accepts no command-line overrides, environment-selected modes, or
//! target values. Every replay input is admitted through `sim-sm`, exact launch
//! sets use the checked aggregation algebra, and binding cycles remain the typed
//! `required_unimplemented` emission produced by [`emit`].

pub mod emit;
mod build_provenance {
    include!(concat!(env!("OUT_DIR"), "/build_provenance.rs"));
}
mod runner_v2;
pub(crate) mod wire;

#[cfg(test)]
use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::path::{Component, Path};

#[cfg(test)]
use crate::emit::emit_all;
use serde::{Deserialize, Deserializer, Serialize, de};
use sim_arch::ArchId;
#[cfg(test)]
use sim_arch::Sku;
use sim_arch::card::CalibError;
use sim_mem::{
    L1Config, L2Config, L2ConfigError, L2Replacement, PchaseCardError, SliceHash, WalkGeometry,
};
#[cfg(test)]
use sim_mem::{L1Counters, L1OriginCounters, L2Counters, L2Source, MemoryOrigin};
use sim_sm::{
    AdmittedLaunchEvidence, DramBw, IssueError, ManifestArtifactToken, ManifestError,
    ReplayConfiguration, ReplayError, ReplayMemMode,
};
#[cfg(test)]
use sim_sm::{
    DramOriginStats, DramStats, FullCoverageSource, FullKernelReplayResult, FullLaunchCoverage,
    FullLaunchSetCounters, FullLaunchSetDeclaration, KernelReplayOutcome, L2OriginSourceStats,
    L2SourceSplit, ReplayLaunchCoverageBinding, TailFlags, admit_launch_evidence,
    aggregate_full_launch_set, replay_kernel,
};
#[cfg(test)]
use sim_trace::{AnchoredDir, canonical_json};
use sim_trace::{AnchoredReadError, Sha256Digest, TraceError};

#[cfg(test)]
const RUN_MANIFEST_SCHEMA_VERSION: u32 = 1;
const MAX_RUN_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
#[cfg(test)]
const MAX_LAUNCHES: usize = 4096;
#[cfg(test)]
const RESULT_SCHEMA_VERSION: &str = "sim_result.v1";

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
struct CanonicalRelativePath(String);

impl CanonicalRelativePath {
    fn checked(raw: String) -> Result<Self, String> {
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
        {
            return Err(format!("not a canonical relative path: {raw:?}"));
        }
        let path = Path::new(&raw);
        if path.is_absolute()
            || path
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
        {
            return Err(format!("not a canonical relative path: {raw:?}"));
        }
        Ok(Self(raw))
    }

    fn as_str(&self) -> &str {
        &self.0
    }

    #[cfg(test)]
    fn as_path(&self) -> &Path {
        Path::new(&self.0)
    }
}

impl<'de> Deserialize<'de> for CanonicalRelativePath {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Self::checked(raw).map_err(de::Error::custom)
    }
}

#[cfg(test)]
#[derive(Debug, Deserialize)]
struct ManifestVersionEnvelope {
    schema_version: u32,
}

#[cfg(test)]
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SimInputManifest {
    schema_version: u32,
    scope: String,
    architecture: ManifestArchitecture,
    configuration_evidence: ManifestConfigurationEvidence,
    launches: Vec<ManifestLaunch>,
    replay_mode: ManifestReplayMode,
}

#[cfg(test)]
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestArchitecture {
    arch_id: String,
    sku: String,
}

#[cfg(test)]
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestConfigurationEvidence {
    arch_card_sha256: Sha256Digest,
    pchase_card_sha256: Sha256Digest,
}

#[cfg(test)]
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestLaunch {
    launch_evidence_manifest_path: CanonicalRelativePath,
    expected_identity_sha256: Sha256Digest,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
enum ManifestReplayMode {
    Flat,
    L1Const,
    L1SharedL2 {
        dram_bandwidth: ManifestDramBandwidth,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ManifestDramBandwidth {
    Carded,
    Unmodeled,
}

/// Closed simulator failure surface. The private kind has exactly one variant
/// for each `sim_error.v2` schema arm, so adding a reachable failure requires an
/// explicit serialization decision rather than falling through an internal arm.
#[derive(Debug)]
pub struct RunError {
    kind: RunErrorKind,
}

#[derive(Debug, thiserror::Error)]
enum RunErrorKind {
    #[error("command-line usage rejected: {detail}")]
    CliUsage {
        detail: String,
        provided_positional_argument_count: u32,
    },
    #[error("manifest admission failed: {detail}")]
    ManifestAdmission {
        detail: String,
        prerequisite_missing: bool,
    },
    #[error("trace admission failed: {detail}")]
    TraceAdmission {
        detail: String,
        prerequisite_missing: bool,
    },
    #[error("architecture-card admission failed: {detail}")]
    ArchCardAdmission {
        detail: String,
        prerequisite_missing: bool,
    },
    #[error("configuration resolution failed: {detail}")]
    ConfigurationResolution {
        detail: String,
        prerequisite_missing: bool,
    },
    #[error("replay failed: {detail}")]
    Replay { detail: String },
    #[error("emission failed: {detail}")]
    Emission { detail: String },
    #[error("result serialization failed: {detail}")]
    ResultSerialization { detail: String },
    #[error("output write failed: {detail}")]
    OutputWrite { detail: String },
}

impl std::fmt::Display for RunError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.kind.fmt(formatter)
    }
}

impl std::error::Error for RunError {}

impl RunError {
    fn new(kind: RunErrorKind) -> Self {
        Self { kind }
    }

    pub fn cli_usage(detail: impl Into<String>, provided_positional_argument_count: u32) -> Self {
        Self::new(RunErrorKind::CliUsage {
            detail: detail.into(),
            provided_positional_argument_count,
        })
    }

    pub fn output_write(detail: impl Into<String>) -> Self {
        Self::new(RunErrorKind::OutputWrite {
            detail: detail.into(),
        })
    }

    fn manifest_admission(detail: impl Into<String>, prerequisite_missing: bool) -> Self {
        Self::new(RunErrorKind::ManifestAdmission {
            detail: detail.into(),
            prerequisite_missing,
        })
    }

    fn trace_admission(detail: impl Into<String>, prerequisite_missing: bool) -> Self {
        Self::new(RunErrorKind::TraceAdmission {
            detail: detail.into(),
            prerequisite_missing,
        })
    }

    fn arch_card_admission(detail: impl Into<String>, prerequisite_missing: bool) -> Self {
        Self::new(RunErrorKind::ArchCardAdmission {
            detail: detail.into(),
            prerequisite_missing,
        })
    }

    fn configuration_resolution(detail: impl Into<String>, prerequisite_missing: bool) -> Self {
        Self::new(RunErrorKind::ConfigurationResolution {
            detail: detail.into(),
            prerequisite_missing,
        })
    }

    fn replay(detail: impl Into<String>) -> Self {
        Self::new(RunErrorKind::Replay {
            detail: detail.into(),
        })
    }

    fn emission(detail: impl Into<String>) -> Self {
        Self::new(RunErrorKind::Emission {
            detail: detail.into(),
        })
    }

    fn result_serialization(detail: impl Into<String>) -> Self {
        Self::new(RunErrorKind::ResultSerialization {
            detail: detail.into(),
        })
    }

    pub fn failure_type(&self) -> &'static str {
        match &self.kind {
            RunErrorKind::CliUsage { .. } => "CLI_USAGE_FAILED",
            RunErrorKind::ManifestAdmission { .. } => "MANIFEST_ADMISSION_FAILED",
            RunErrorKind::TraceAdmission { .. } => "TRACE_ADMISSION_FAILED",
            RunErrorKind::ArchCardAdmission { .. } => "ARCH_CARD_ADMISSION_FAILED",
            RunErrorKind::ConfigurationResolution { .. } => "CONFIGURATION_RESOLUTION_FAILED",
            RunErrorKind::Replay { .. } => "REPLAY_FAILED",
            RunErrorKind::Emission { .. } => "EMISSION_FAILED",
            RunErrorKind::ResultSerialization { .. } => "RESULT_SERIALIZATION_FAILED",
            RunErrorKind::OutputWrite { .. } => "OUTPUT_WRITE_FAILED",
        }
    }

    pub fn stage(&self) -> &'static str {
        match &self.kind {
            RunErrorKind::CliUsage { .. } => "cli_usage",
            RunErrorKind::ManifestAdmission { .. } => "manifest_admission",
            RunErrorKind::TraceAdmission { .. } => "trace_admission",
            RunErrorKind::ArchCardAdmission { .. } => "arch_card_admission",
            RunErrorKind::ConfigurationResolution { .. } => "configuration_resolution",
            RunErrorKind::Replay { .. } => "replay",
            RunErrorKind::Emission { .. } => "emission",
            RunErrorKind::ResultSerialization { .. } => "result_serialization",
            RunErrorKind::OutputWrite { .. } => "output_write",
        }
    }

    pub const fn exit_code(&self) -> u8 {
        match &self.kind {
            RunErrorKind::CliUsage { .. } => 2,
            RunErrorKind::ManifestAdmission {
                prerequisite_missing: true,
                ..
            }
            | RunErrorKind::TraceAdmission {
                prerequisite_missing: true,
                ..
            }
            | RunErrorKind::ArchCardAdmission {
                prerequisite_missing: true,
                ..
            }
            | RunErrorKind::ConfigurationResolution {
                prerequisite_missing: true,
                ..
            } => 2,
            RunErrorKind::ManifestAdmission { .. }
            | RunErrorKind::TraceAdmission { .. }
            | RunErrorKind::ArchCardAdmission { .. }
            | RunErrorKind::ConfigurationResolution { .. }
            | RunErrorKind::Replay { .. }
            | RunErrorKind::Emission { .. }
            | RunErrorKind::ResultSerialization { .. }
            | RunErrorKind::OutputWrite { .. } => 1,
        }
    }

    pub fn canonical_json(&self) -> Result<String, ErrorEncodingError> {
        encode_error_v2(self)
    }

    /// Prevalidated, sealed last-resort document used only when normal typed
    /// error encoding fails. It is deliberately a constant so the fallback
    /// cannot recurse through the serializer it is replacing.
    pub const fn canonical_serialization_fallback() -> &'static str {
        r#"{"error":{"detail":"could not serialize typed error document","failure_type":"RESULT_SERIALIZATION_FAILED","stage":"result_serialization"},"error_sha256":"cd91b02214fcb7c3d542dd0c0b49610c95597745d7bc5f1b03c97dc1a385adfc","exit_status":{"exit_code":1,"status":"failure"},"schema_version":"sim_error.v2"}"#
    }
}

#[derive(Debug, thiserror::Error)]
#[error("could not encode sim_error.v2: {detail}")]
pub struct ErrorEncodingError {
    detail: String,
}

impl ErrorEncodingError {
    fn new(detail: impl std::fmt::Display) -> Self {
        Self {
            detail: detail.to_string(),
        }
    }
}

fn encode_error_v2(error: &RunError) -> Result<String, ErrorEncodingError> {
    use crate::wire::{SimErrorKindV2, SimErrorV2};

    let detail = error.to_string();
    let kind = match &error.kind {
        RunErrorKind::CliUsage {
            provided_positional_argument_count,
            ..
        } => SimErrorKindV2::cli_usage(detail, *provided_positional_argument_count),
        RunErrorKind::ManifestAdmission { .. } => SimErrorKindV2::manifest_admission(detail, None),
        RunErrorKind::TraceAdmission { .. } => SimErrorKindV2::trace_admission(detail, None),
        RunErrorKind::ArchCardAdmission { .. } => SimErrorKindV2::arch_card_admission(detail, None),
        RunErrorKind::ConfigurationResolution { .. } => {
            SimErrorKindV2::configuration_resolution(detail, None)
        }
        RunErrorKind::Replay { .. } => SimErrorKindV2::replay(detail, None),
        RunErrorKind::Emission { .. } => SimErrorKindV2::emission(detail, None),
        RunErrorKind::ResultSerialization { .. } => {
            SimErrorKindV2::result_serialization(detail, None)
        }
        RunErrorKind::OutputWrite { .. } => SimErrorKindV2::output_write(detail, None),
    }
    .map_err(ErrorEncodingError::new)?;
    let exit_code = std::num::NonZeroU8::new(error.exit_code())
        .ok_or_else(|| ErrorEncodingError::new("failure exit code was zero"))?;
    let placeholder = Sha256Digest::of(b"sim-run.error-v2.unsealed-placeholder");
    let provisional = SimErrorV2::new(placeholder, None, kind.clone(), exit_code);
    let provisional_value = serde_json::to_value(&provisional).map_err(ErrorEncodingError::new)?;
    let unsigned_value = error_value_without_seal(provisional_value)?;
    let unsigned_canonical =
        sim_trace::canonical_json_utf8(&unsigned_value).map_err(ErrorEncodingError::new)?;
    let error_sha256 = Sha256Digest::of(unsigned_canonical.as_bytes());

    let sealed = SimErrorV2::new(error_sha256, None, kind, exit_code);
    let sealed_value = serde_json::to_value(&sealed).map_err(ErrorEncodingError::new)?;
    let revalidated_unsigned = error_value_without_seal(sealed_value.clone())?;
    let revalidated_canonical =
        sim_trace::canonical_json_utf8(&revalidated_unsigned).map_err(ErrorEncodingError::new)?;
    if revalidated_canonical != unsigned_canonical
        || Sha256Digest::of(revalidated_canonical.as_bytes()) != error_sha256
    {
        return Err(ErrorEncodingError::new(
            "sim_error.v2 seal preimage changed during checked construction",
        ));
    }

    let canonical_json =
        sim_trace::canonical_json_utf8(&sealed_value).map_err(ErrorEncodingError::new)?;
    let reparsed: serde_json::Value =
        serde_json::from_str(&canonical_json).map_err(ErrorEncodingError::new)?;
    if sim_trace::canonical_json_utf8(&reparsed).map_err(ErrorEncodingError::new)? != canonical_json
    {
        return Err(ErrorEncodingError::new(
            "sim_error.v2 final document did not revalidate as canonical UTF-8 JSON",
        ));
    }
    Ok(canonical_json)
}

fn error_value_without_seal(
    mut value: serde_json::Value,
) -> Result<serde_json::Value, ErrorEncodingError> {
    let object = value.as_object_mut().ok_or_else(|| {
        ErrorEncodingError::new("serialized error document root is not an object")
    })?;
    let removed = object
        .remove("error_sha256")
        .ok_or_else(|| ErrorEncodingError::new("serialized error document has no seal member"))?;
    if !removed.is_string() {
        return Err(ErrorEncodingError::new(
            "serialized error document seal is not a string",
        ));
    }
    Ok(value)
}

fn anchored_error_is_missing(error: &AnchoredReadError) -> bool {
    match error {
        AnchoredReadError::RetainedDescriptorUnavailable { .. } => true,
        AnchoredReadError::Io { source, .. } => source.kind() == std::io::ErrorKind::NotFound,
        _ => false,
    }
}

fn trace_error_is_missing(error: &TraceError) -> bool {
    match error {
        TraceError::Io { source, .. } => source.kind() == std::io::ErrorKind::NotFound,
        TraceError::AnchoredSnapshot(source) => anchored_error_is_missing(source),
        TraceError::SnapshotArtifactPresenceMismatch {
            expected: true,
            actual: false,
            ..
        }
        | TraceError::TraceThreadblocksEmpty { .. }
        | TraceError::ThreadblocksMissing { .. }
        | TraceError::AllocationsSidecarMissing { .. }
        | TraceError::SamplingPayloadSnapshotMissing { .. } => true,
        _ => false,
    }
}

/// One sealed canonical simulator result.
#[derive(Debug, Clone)]
pub struct SimRunOutput {
    canonical_json: String,
    result_sha256: Sha256Digest,
}

impl SimRunOutput {
    pub fn canonical_json(&self) -> &str {
        &self.canonical_json
    }

    pub fn result_sha256(&self) -> Sha256Digest {
        self.result_sha256
    }
}

#[cfg(test)]
struct SealedDocument {
    canonical_json: String,
    seal: Sha256Digest,
}

fn result_serialization_error(error: impl std::fmt::Display) -> RunError {
    RunError::result_serialization(error.to_string())
}

#[cfg(test)]
fn seal_document(
    mut value: serde_json::Value,
    field: &'static str,
) -> Result<SealedDocument, RunError> {
    let unsigned = canonical_json(&value).map_err(result_serialization_error)?;
    let seal = Sha256Digest::of(unsigned.as_bytes());
    let object = value
        .as_object_mut()
        .ok_or_else(|| result_serialization_error("serialized document root is not an object"))?;
    object.insert(
        field.to_owned(),
        serde_json::Value::String(seal.to_string()),
    );
    let canonical_json = canonical_json(&value).map_err(result_serialization_error)?;
    Ok(SealedDocument {
        canonical_json,
        seal,
    })
}

fn manifest_base(path: &Path) -> &Path {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    }
}

fn manifest_rejected(path: &Path, reason: impl std::fmt::Display) -> RunError {
    RunError::manifest_admission(
        format!("run manifest {} was rejected: {reason}", path.display()),
        false,
    )
}

fn manifest_read_error(path: &Path, source: AnchoredReadError) -> RunError {
    let prerequisite_missing = anchored_error_is_missing(&source);
    RunError::manifest_admission(
        format!("could not read run manifest {}: {source}", path.display()),
        prerequisite_missing,
    )
}

fn manifest_serialization_error(path: &Path, error: impl std::fmt::Display) -> RunError {
    manifest_rejected(path, format!("canonical manifest encoding failed: {error}"))
}

fn require_canonical_manifest_path(path: &Path) -> Result<(), RunError> {
    let raw = path
        .to_str()
        .ok_or_else(|| manifest_rejected(path, "manifest path must be UTF-8"))?;
    let has_dot_component = raw
        .split('/')
        .any(|component| matches!(component, "." | ".."));
    if raw.is_empty() || raw == "/" || raw.ends_with('/') || raw.contains("//") || has_dot_component
    {
        return Err(manifest_rejected(
            path,
            "manifest path must use one literal canonical spelling",
        ));
    }
    if path
        .components()
        .any(|component| !matches!(component, Component::RootDir | Component::Normal(_)))
    {
        return Err(manifest_rejected(
            path,
            "manifest path contains a non-canonical component",
        ));
    }
    Ok(())
}

#[cfg(test)]
fn read_manifest(path: &Path) -> Result<(SimInputManifest, Sha256Digest), RunError> {
    require_canonical_manifest_path(path)?;
    let base = manifest_base(path);
    let name = path
        .file_name()
        .ok_or_else(|| manifest_rejected(path, "manifest path has no file name"))?;
    let reader = AnchoredDir::open(base.to_path_buf())
        .map_err(|source| manifest_read_error(path, source))?;
    let file = reader
        .read_bounded_regular(Path::new(name), MAX_RUN_MANIFEST_BYTES)
        .map_err(|source| manifest_read_error(path, source))?;
    let raw = file.into_raw();
    let envelope: ManifestVersionEnvelope =
        serde_json::from_slice(&raw).map_err(|source| manifest_rejected(path, source))?;
    if envelope.schema_version != RUN_MANIFEST_SCHEMA_VERSION {
        return Err(manifest_rejected(
            path,
            format!(
                "unsupported schema_version {}; expected {}",
                envelope.schema_version, RUN_MANIFEST_SCHEMA_VERSION
            ),
        ));
    }
    let manifest: SimInputManifest =
        serde_json::from_slice(&raw).map_err(|source| manifest_rejected(path, source))?;
    if manifest.launches.is_empty() {
        return Err(manifest_rejected(path, "launches must be nonempty"));
    }
    if manifest.launches.len() > MAX_LAUNCHES {
        return Err(manifest_rejected(
            path,
            format!(
                "launch count {} exceeds implementation limit {MAX_LAUNCHES}",
                manifest.launches.len()
            ),
        ));
    }
    let mut paths = BTreeSet::new();
    for launch in &manifest.launches {
        if !paths.insert(launch.launch_evidence_manifest_path.as_str()) {
            return Err(manifest_rejected(
                path,
                format!(
                    "duplicate launch_evidence_manifest_path {:?}",
                    launch.launch_evidence_manifest_path.as_str()
                ),
            ));
        }
    }
    let semantic = serde_json::to_value(&manifest)
        .map_err(|error| manifest_serialization_error(path, error))?;
    let canonical =
        canonical_json(&semantic).map_err(|error| manifest_serialization_error(path, error))?;
    Ok((manifest, Sha256Digest::of(canonical.as_bytes())))
}

#[cfg(test)]
struct AdmittedEntry {
    relative_manifest_path: String,
    expected_identity_sha256: Sha256Digest,
    binding: ReplayLaunchCoverageBinding,
    identity_json: serde_json::Value,
    evidence: AdmittedLaunchEvidence,
}

#[derive(Clone, Copy)]
enum IdentityFailureBoundary {
    #[cfg(test)]
    Manifest,
    Trace,
    ArchCard,
    Configuration,
}

fn identity_mismatch(
    boundary: IdentityFailureBoundary,
    edge: &'static str,
    expected: impl ToString,
    actual: impl ToString,
) -> RunError {
    let detail = format!(
        "identity edge {edge} mismatch: expected {}, got {}",
        expected.to_string(),
        actual.to_string()
    );
    match boundary {
        #[cfg(test)]
        IdentityFailureBoundary::Manifest => RunError::manifest_admission(detail, false),
        IdentityFailureBoundary::Trace => RunError::trace_admission(detail, false),
        IdentityFailureBoundary::ArchCard => RunError::arch_card_admission(detail, false),
        IdentityFailureBoundary::Configuration => RunError::configuration_resolution(detail, false),
    }
}

fn artifact_admission_error(
    artifact: ManifestArtifactToken,
    detail: String,
    prerequisite_missing: bool,
) -> RunError {
    match artifact {
        ManifestArtifactToken::ArchCard => {
            RunError::arch_card_admission(detail, prerequisite_missing)
        }
        ManifestArtifactToken::PchaseCard => {
            RunError::configuration_resolution(detail, prerequisite_missing)
        }
        ManifestArtifactToken::LaunchEvidenceManifest
        | ManifestArtifactToken::DynamicTracePb
        | ManifestArtifactToken::ThreadblockPayload
        | ManifestArtifactToken::ThreadblockPayloadSortedPathManifest
        | ManifestArtifactToken::ThreadblockPayloadSortedPathStream
        | ManifestArtifactToken::AllocationsJson
        | ManifestArtifactToken::TraceSamplingJson
        | ManifestArtifactToken::StatsCsv
        | ManifestArtifactToken::EnhancedExecutionInfoJson
        | ManifestArtifactToken::ControlBitsSidecar => {
            RunError::trace_admission(detail, prerequisite_missing)
        }
    }
}

fn replay_arch_card_placement_prerequisite_missing(error: &ReplayError) -> Option<bool> {
    match error {
        ReplayError::ArchitectureCardPlacement { source } => {
            Some(calib_error_is_unavailable(source))
        }
        ReplayError::ArchitectureCardSharedMemoryCapacity { .. }
        | ReplayError::SharedMemoryCapacityInvalid { .. }
        | ReplayError::CtaSharedMemoryCapacityExceeded { .. } => Some(true),
        _ => None,
    }
}

fn manifest_error_is_missing(error: &ManifestError) -> bool {
    match error {
        ManifestError::Anchored(source)
        | ManifestError::ArtifactRead { source, .. }
        | ManifestError::ArtifactGenerationRead { source, .. } => anchored_error_is_missing(source),
        ManifestError::ArtifactPresenceMismatch {
            expected: true,
            actual: false,
            ..
        } => true,
        ManifestError::Trace(source) => trace_error_is_missing(source),
        ManifestError::CapabilityMissing { .. } => true,
        ManifestError::ArchCard(source) => calib_error_is_unavailable(source),
        ManifestError::PchaseCard(source) => pchase_error_is_unavailable(source),
        ManifestError::Coverage(source) => {
            replay_arch_card_placement_prerequisite_missing(source).unwrap_or(false)
        }
        ManifestError::Parse { .. }
        | ManifestError::SchemaVersionUnsupported { .. }
        | ManifestError::ArtifactPathInvalid { .. }
        | ManifestError::ArtifactPathMismatch { .. }
        | ManifestError::ArtifactPathDuplicate { .. }
        | ManifestError::ArtifactFileAlias { .. }
        | ManifestError::ArtifactSizeMismatch { .. }
        | ManifestError::ArtifactDigestMismatch { .. }
        | ManifestError::ArtifactGenerationMismatch { .. }
        | ManifestError::ArtifactPresenceMismatch { .. }
        | ManifestError::ArtifactCardinalityMismatch { .. }
        | ManifestError::ArtifactResourceLimitExceeded { .. }
        | ManifestError::SnapshotResourceLimitExceeded { .. }
        | ManifestError::TraceBundlePathInvalid { .. }
        | ManifestError::TraceBundleNonpositive { .. }
        | ManifestError::TraceBundleSizeOverflow { .. }
        | ManifestError::TraceBundleCountMismatch { .. }
        | ManifestError::TraceBundleWireVersionInvalid { .. }
        | ManifestError::ReplayIdentityValueOutOfRange { .. }
        | ManifestError::ExecInfo(_)
        | ManifestError::Control(_)
        | ManifestError::Program(_)
        | ManifestError::Architecture(_)
        | ManifestError::IdentityFieldMissing { .. }
        | ManifestError::IdentityMismatch { .. }
        | ManifestError::LaunchMissing { .. }
        | ManifestError::LaunchAmbiguous { .. }
        | ManifestError::EdgeUnverifiable { .. }
        | ManifestError::IdentitySerialization(_)
        | ManifestError::IdentityCanonicalization(_) => false,
    }
}

fn launch_evidence_error(path: &Path, source: ManifestError) -> RunError {
    let prerequisite_missing = manifest_error_is_missing(&source);
    let detail = format!("launch evidence {} was rejected: {source}", path.display());
    match &source {
        ManifestError::ArtifactPathInvalid { artifact, .. }
        | ManifestError::ArtifactPathMismatch { artifact, .. }
        | ManifestError::ArtifactRead { artifact, .. }
        | ManifestError::ArtifactPathDuplicate { artifact, .. }
        | ManifestError::ArtifactFileAlias { artifact, .. }
        | ManifestError::ArtifactSizeMismatch { artifact, .. }
        | ManifestError::ArtifactDigestMismatch { artifact, .. }
        | ManifestError::ArtifactGenerationRead { artifact, .. }
        | ManifestError::ArtifactGenerationMismatch { artifact, .. }
        | ManifestError::ArtifactPresenceMismatch { artifact, .. }
        | ManifestError::ArtifactCardinalityMismatch { artifact, .. }
        | ManifestError::ArtifactResourceLimitExceeded { artifact, .. }
        | ManifestError::IdentityMismatch { artifact, .. } => {
            artifact_admission_error(*artifact, detail, prerequisite_missing)
        }
        ManifestError::ArchCard(_) => RunError::arch_card_admission(detail, prerequisite_missing),
        ManifestError::PchaseCard(_) => {
            RunError::configuration_resolution(detail, prerequisite_missing)
        }
        ManifestError::Coverage(replay)
            if replay_arch_card_placement_prerequisite_missing(replay).is_some() =>
        {
            RunError::arch_card_admission(detail, prerequisite_missing)
        }
        ManifestError::Parse { .. }
        | ManifestError::SchemaVersionUnsupported { .. }
        | ManifestError::Anchored(_)
        | ManifestError::SnapshotResourceLimitExceeded { .. }
        | ManifestError::TraceBundlePathInvalid { .. }
        | ManifestError::TraceBundleNonpositive { .. }
        | ManifestError::TraceBundleSizeOverflow { .. }
        | ManifestError::TraceBundleCountMismatch { .. }
        | ManifestError::TraceBundleWireVersionInvalid { .. }
        | ManifestError::ReplayIdentityValueOutOfRange { .. }
        | ManifestError::CapabilityMissing { .. }
        | ManifestError::Trace(_)
        | ManifestError::ExecInfo(_)
        | ManifestError::Control(_)
        | ManifestError::Program(_)
        | ManifestError::Architecture(_)
        | ManifestError::Coverage(_)
        | ManifestError::IdentityFieldMissing { .. }
        | ManifestError::LaunchMissing { .. }
        | ManifestError::LaunchAmbiguous { .. }
        | ManifestError::EdgeUnverifiable { .. }
        | ManifestError::IdentitySerialization(_)
        | ManifestError::IdentityCanonicalization(_) => {
            RunError::trace_admission(detail, prerequisite_missing)
        }
    }
}

#[cfg(test)]
fn parse_architecture(manifest: &SimInputManifest, path: &Path) -> Result<(ArchId, Sku), RunError> {
    let arch = ArchId::parse_deploy(&manifest.architecture.arch_id)
        .map_err(|source| manifest_rejected(path, source))?;
    let sku =
        Sku::parse(&manifest.architecture.sku).map_err(|source| manifest_rejected(path, source))?;
    if sku.arch() != arch {
        return Err(identity_mismatch(
            IdentityFailureBoundary::Manifest,
            "run manifest sku->arch",
            arch,
            sku.arch(),
        ));
    }
    if arch != ArchId::Sm86 {
        return Err(RunError::manifest_admission(
            format!("deployment architecture {arch} is not enabled by the runner"),
            false,
        ));
    }
    Ok((arch, sku))
}

#[cfg(test)]
fn admit_all(
    manifest_path: &Path,
    manifest: &SimInputManifest,
    arch: ArchId,
    sku: Sku,
) -> Result<Vec<AdmittedEntry>, RunError> {
    let base = manifest_base(manifest_path);
    let mut seals = BTreeSet::new();
    let mut admitted = Vec::new();
    admitted
        .try_reserve_exact(manifest.launches.len())
        .map_err(result_serialization_error)?;
    for launch in &manifest.launches {
        let path = base.join(launch.launch_evidence_manifest_path.as_path());
        let evidence =
            admit_launch_evidence(&path).map_err(|source| launch_evidence_error(&path, source))?;
        let identity = evidence.identity();
        let actual_seal = identity.seal_sha256();
        if actual_seal != launch.expected_identity_sha256 {
            return Err(identity_mismatch(
                IdentityFailureBoundary::Trace,
                "launch expected_identity_sha256",
                launch.expected_identity_sha256,
                actual_seal,
            ));
        }
        if !seals.insert(actual_seal) {
            return Err(manifest_rejected(
                manifest_path,
                format!("duplicate launch evidence identity {actual_seal}"),
            ));
        }
        let launch_arch = identity
            .arch()
            .map_err(|source| launch_evidence_error(&path, ManifestError::Architecture(source)))?;
        let launch_sku = identity
            .sku()
            .map_err(|source| launch_evidence_error(&path, ManifestError::Architecture(source)))?;
        if launch_arch != arch {
            return Err(identity_mismatch(
                IdentityFailureBoundary::Trace,
                "launch architecture",
                arch,
                launch_arch,
            ));
        }
        if launch_sku != sku {
            return Err(identity_mismatch(
                IdentityFailureBoundary::Trace,
                "launch SKU",
                sku.token(),
                launch_sku.token(),
            ));
        }
        if identity.arch_card_sha256() != manifest.configuration_evidence.arch_card_sha256 {
            return Err(identity_mismatch(
                IdentityFailureBoundary::ArchCard,
                "launch architecture-card SHA-256",
                manifest.configuration_evidence.arch_card_sha256,
                identity.arch_card_sha256(),
            ));
        }
        if identity.pchase_card_sha256() != manifest.configuration_evidence.pchase_card_sha256 {
            return Err(identity_mismatch(
                IdentityFailureBoundary::Configuration,
                "launch P-chase-card SHA-256",
                manifest.configuration_evidence.pchase_card_sha256,
                identity.pchase_card_sha256(),
            ));
        }
        let binding = identity.launch_binding().map_err(|source| {
            RunError::trace_admission(
                format!("admitted launch identity {actual_seal} did not reconstruct: {source}"),
                false,
            )
        })?;
        let identity_json =
            serde_json::from_str(identity.canonical_json()).map_err(result_serialization_error)?;
        admitted.push(AdmittedEntry {
            relative_manifest_path: launch.launch_evidence_manifest_path.as_str().to_owned(),
            expected_identity_sha256: launch.expected_identity_sha256,
            binding,
            identity_json,
            evidence,
        });
    }
    admitted.sort_by(|left, right| left.binding.cmp(&right.binding));
    Ok(admitted)
}

struct ResolvedConfiguration {
    issue_params: sim_sm::IssueModelParams,
    replay_mode: ReplayMemMode,
    replay_configuration: ReplayConfiguration,
    wire: ResolvedConfigurationWire,
}

fn configuration_error(
    component: &'static str,
    reason: impl std::fmt::Display,
    prerequisite_missing: bool,
) -> RunError {
    RunError::configuration_resolution(
        format!("configuration component {component}: {reason}"),
        prerequisite_missing,
    )
}

fn arch_card_error(
    component: &'static str,
    reason: impl std::fmt::Display,
    prerequisite_missing: bool,
) -> RunError {
    RunError::arch_card_admission(
        format!("architecture-card component {component}: {reason}"),
        prerequisite_missing,
    )
}

fn calib_error_is_unavailable(error: &CalibError) -> bool {
    match error {
        CalibError::Io { source, .. } => source.kind() == std::io::ErrorKind::NotFound,
        CalibError::Capability(_)
        | CalibError::LatenciesMissing { .. }
        | CalibError::ResearchGap { .. }
        | CalibError::BlockResearchGap { .. } => true,
        _ => false,
    }
}

fn calib_arch_card_error(component: &'static str, error: CalibError) -> RunError {
    let unavailable = calib_error_is_unavailable(&error);
    arch_card_error(component, error, unavailable)
}

fn issue_configuration_error(component: &'static str, error: IssueError) -> RunError {
    match error {
        IssueError::Calibration(source) => calib_arch_card_error(component, source),
        invalid @ IssueError::InvalidLatency { .. } => arch_card_error(component, invalid, false),
        other => {
            let unavailable = matches!(
                &other,
                IssueError::ArchitectureUnsupported { .. }
                    | IssueError::CardedResourceExecutionDeferred { .. }
                    | IssueError::Capability(_)
            );
            configuration_error(component, other, unavailable)
        }
    }
}

fn pchase_error_is_unavailable(error: &PchaseCardError) -> bool {
    match error {
        PchaseCardError::Io { source, .. } => source.kind() == std::io::ErrorKind::NotFound,
        PchaseCardError::Capability(_)
        | PchaseCardError::PropertyMissing { .. }
        | PchaseCardError::TypedWithoutDefault { .. } => true,
        _ => false,
    }
}

fn pchase_configuration_error(component: &'static str, error: PchaseCardError) -> RunError {
    let unavailable = pchase_error_is_unavailable(&error);
    configuration_error(component, error, unavailable)
}

fn l2_arch_card_error(component: &'static str, error: L2ConfigError) -> RunError {
    let unavailable = matches!(
        &error,
        L2ConfigError::Capability(_) | L2ConfigError::WritePolicyGap
    );
    arch_card_error(component, error, unavailable)
}

fn replay_configuration_error(component: &'static str, error: ReplayError) -> RunError {
    configuration_error(component, error, false)
}

fn resolve_configuration(
    first: &AdmittedLaunchEvidence,
    arch: ArchId,
    requested: ManifestReplayMode,
) -> Result<ResolvedConfiguration, RunError> {
    let latencies = first
        .arch_card()
        .required_latencies()
        .map_err(|source| calib_arch_card_error("architecture-card latencies", source))?;
    let issue_params = sim_sm::IssueModelParams::from_legacy_card_latencies_for_arch(
        arch,
        latencies.l1_hit,
        latencies.l2_hit,
        latencies.dram,
        latencies.shared,
    )
    .map_err(|source| issue_configuration_error("issue hypotheses", source))?;
    let issue_wire = IssueConfigurationWire::from(issue_params);

    let (replay_mode, mode_wire, l1_wire, l2_wire, dram_wire) = match requested {
        ManifestReplayMode::Flat => (
            ReplayMemMode::Flat,
            ReplayModeWire::Flat,
            Availability::Unmodeled {
                reason: "replay_mode_flat",
            },
            Availability::Unmodeled {
                reason: "replay_mode_without_shared_l2",
            },
            Availability::Unmodeled {
                reason: "replay_mode_without_shared_l2",
            },
        ),
        ManifestReplayMode::L1Const => {
            let geometry = resolve_walk_geometry(first, arch)?;
            let l1 = issue_params.l1_config(geometry.capacity.line_capacity, geometry.mshr.count);
            let l1_wire = L1ConfigurationWire::from_resolved(l1, &geometry);
            (
                ReplayMemMode::L1Const(l1),
                ReplayModeWire::L1Const,
                Availability::Modeled { value: l1_wire },
                Availability::Unmodeled {
                    reason: "replay_mode_without_shared_l2",
                },
                Availability::Unmodeled {
                    reason: "replay_mode_without_shared_l2",
                },
            )
        }
        ManifestReplayMode::L1SharedL2 { dram_bandwidth } => {
            let geometry = resolve_walk_geometry(first, arch)?;
            let l1 = issue_params.l1_config(geometry.capacity.line_capacity, geometry.mshr.count);
            let l1_wire = L1ConfigurationWire::from_resolved(l1, &geometry);
            let card_l2 = first
                .arch_card()
                .require_l2_geometry()
                .map_err(|source| calib_arch_card_error("L2 geometry", source))?;
            let l2 = L2Config::from_carded_geometry_for_arch(
                arch,
                card_l2.capacity_bytes,
                card_l2.line_bytes,
                card_l2.sector_bytes,
                card_l2.num_slices,
                card_l2.policy.as_deref(),
            )
            .map_err(|source| l2_arch_card_error("L2 configuration", source))?;
            let (dram, dram_wire) = match dram_bandwidth {
                ManifestDramBandwidth::Unmodeled => (
                    DramBw::Unmodeled { arch },
                    DramConfigurationWire::Unmodeled {
                        provenance: "explicit_manifest_ablation",
                    },
                ),
                ManifestDramBandwidth::Carded => {
                    let card_dram = first
                        .arch_card()
                        .require_dram_bandwidth()
                        .map_err(|source| calib_arch_card_error("DRAM bandwidth", source))?;
                    let bytes_per_sec = card_dram
                        .validated_peak_bytes_per_second()
                        .map_err(|source| calib_arch_card_error("DRAM bandwidth", source))?;
                    let sm_clock_hz = card_dram
                        .locked_sm_clock_hz()
                        .map_err(|source| calib_arch_card_error("DRAM locked clock", source))?;
                    (
                        DramBw::Carded {
                            arch,
                            bytes_per_sec,
                            sm_clock_hz,
                        },
                        DramConfigurationWire::Carded {
                            bytes_per_sec,
                            sm_clock_hz,
                            provenance: "architecture_card_locked_operating_point",
                        },
                    )
                }
            };
            (
                ReplayMemMode::L1SharedL2 { l1, l2, dram },
                ReplayModeWire::L1SharedL2 {
                    dram_bandwidth: match dram_bandwidth {
                        ManifestDramBandwidth::Carded => "carded",
                        ManifestDramBandwidth::Unmodeled => "unmodeled",
                    },
                },
                Availability::Modeled { value: l1_wire },
                Availability::Modeled {
                    value: L2ConfigurationWire::from(l2),
                },
                Availability::Modeled { value: dram_wire },
            )
        }
    };
    let replay_configuration = ReplayConfiguration::checked(issue_params, replay_mode)
        .map_err(|source| replay_configuration_error("replay configuration", source))?;
    Ok(ResolvedConfiguration {
        issue_params,
        replay_mode,
        replay_configuration,
        wire: ResolvedConfigurationWire {
            replay_mode: mode_wire,
            co_residency: "modeled",
            issue: issue_wire,
            l1: l1_wire,
            l2: l2_wire,
            dram: dram_wire,
        },
    })
}

fn resolve_walk_geometry(
    evidence: &AdmittedLaunchEvidence,
    arch: ArchId,
) -> Result<WalkGeometry, RunError> {
    evidence
        .pchase_card()
        .resolve_walk_geometry_for_arch(arch, None)
        .map_err(|source| pchase_configuration_error("P-chase walk geometry", source))
}

#[derive(Clone, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum Availability<T> {
    Modeled { value: T },
    Unmodeled { reason: &'static str },
}

#[derive(Clone, Serialize)]
struct ResolvedConfigurationWire {
    replay_mode: ReplayModeWire,
    co_residency: &'static str,
    issue: IssueConfigurationWire,
    l1: Availability<L1ConfigurationWire>,
    l2: Availability<L2ConfigurationWire>,
    dram: Availability<DramConfigurationWire>,
}

#[derive(Clone, Serialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
enum ReplayModeWire {
    Flat,
    L1Const,
    L1SharedL2 { dram_bandwidth: &'static str },
}

#[derive(Clone, Serialize)]
struct IssueConfigurationWire {
    arch: &'static str,
    l1_hit_latency_cycles: u64,
    l1_miss_latency_cycles: u64,
    global_flat_latency_cycles: u64,
    shared_write_barrier_latency_cycles: u64,
    non_memory_write_barrier_latency_cycles: u64,
    read_barrier_latency_cycles: u64,
    max_resident_warps_per_sm: u32,
    latency_provenance: &'static str,
    issue_hypothesis_provenance: &'static str,
}

impl From<sim_sm::IssueModelParams> for IssueConfigurationWire {
    fn from(params: sim_sm::IssueModelParams) -> Self {
        Self {
            arch: params.arch().token(),
            l1_hit_latency_cycles: params.l1_hit_latency(),
            l1_miss_latency_cycles: params.l1_miss_latency(),
            global_flat_latency_cycles: params.global_flat_latency(),
            shared_write_barrier_latency_cycles: params.shared_write_barrier_latency(),
            non_memory_write_barrier_latency_cycles: params.non_memory_write_barrier_latency(),
            read_barrier_latency_cycles: params.read_barrier_latency(),
            max_resident_warps_per_sm: params.max_resident_warps_per_sm(),
            latency_provenance: "architecture_card",
            issue_hypothesis_provenance: params.resource_lane().provenance(),
        }
    }
}

#[derive(Clone, Serialize)]
struct L1ConfigurationWire {
    arch: &'static str,
    line_capacity: u64,
    capacity_bytes: u64,
    mshr_count: u32,
    hit_latency_cycles: u64,
    miss_latency_cycles: u64,
    capacity_provenance: String,
    mshr_provenance: &'static str,
    replacement_model: &'static str,
    replacement_provenance: &'static str,
    set_mapping_model: &'static str,
    set_mapping_provenance: &'static str,
}

impl L1ConfigurationWire {
    fn from_resolved(config: L1Config, geometry: &WalkGeometry) -> Self {
        Self {
            arch: config.arch.token(),
            line_capacity: config.line_capacity,
            capacity_bytes: geometry.capacity.capacity_bytes,
            mshr_count: geometry.mshr.count,
            hit_latency_cycles: config.hit_latency,
            miss_latency_cycles: config.miss_latency,
            capacity_provenance: format!(
                "{}:{}",
                geometry.capacity.source, geometry.capacity.property_name
            ),
            mshr_provenance: geometry.mshr.source,
            replacement_model: geometry.replacement.modeled,
            replacement_provenance: geometry.replacement.source,
            set_mapping_model: geometry.set_mapping.modeled,
            set_mapping_provenance: geometry.set_mapping.source,
        }
    }
}

#[derive(Clone, Serialize)]
struct L2ConfigurationWire {
    arch: &'static str,
    capacity_bytes: u64,
    line_bytes: u32,
    sector_bytes: u32,
    num_slices: u32,
    associativity: u32,
    replacement: &'static str,
    slice_hash: &'static str,
    geometry_provenance: &'static str,
    hypothesis_provenance: &'static str,
}

impl From<L2Config> for L2ConfigurationWire {
    fn from(config: L2Config) -> Self {
        let replacement = match config.replacement {
            L2Replacement::Lru => "lru",
            L2Replacement::Srrip => "srrip",
            L2Replacement::Bip => "bip",
        };
        let slice_hash = match config.slice_hash {
            SliceHash::LineMod => "line_mod",
            SliceHash::SectorMod => "sector_mod",
            SliceHash::XorFolded => "xor_folded",
            SliceHash::KnuthMult => "knuth_mult",
            SliceHash::ShiftMod => "shift_mod",
            SliceHash::BitReverse => "bit_reverse",
        };
        Self {
            arch: config.arch.token(),
            capacity_bytes: config.capacity_bytes,
            line_bytes: config.line_bytes,
            sector_bytes: config.sector_bytes,
            num_slices: config.num_slices,
            associativity: config.associativity,
            replacement,
            slice_hash,
            geometry_provenance: "architecture_card",
            hypothesis_provenance: "architecture_local_reviewed_default",
        }
    }
}

#[derive(Clone, Serialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
enum DramConfigurationWire {
    Unmodeled {
        provenance: &'static str,
    },
    Carded {
        bytes_per_sec: u64,
        sm_clock_hz: u64,
        provenance: &'static str,
    },
}

/// Admit, replay, aggregate, emit, and canonically seal one strict v2 run manifest.
///
/// This public consumer has no v1 fallback or version dispatch.
pub fn run_manifest(path: impl AsRef<Path>) -> Result<SimRunOutput, RunError> {
    runner_v2::run_manifest_v2(path)
}

/// Run a strict v2 manifest from inherited manifest/evidence-base descriptors.
///
/// `manifest_name` must be one literal file-name component naming the same
/// device/inode as `manifest_fd` beneath `evidence_base_fd`.
pub fn run_manifest_retained_fds(
    manifest_fd: u32,
    evidence_base_fd: u32,
    manifest_name: impl AsRef<OsStr>,
) -> Result<SimRunOutput, RunError> {
    runner_v2::run_manifest_v2_retained(manifest_fd, evidence_base_fd, manifest_name.as_ref())
}

#[cfg(test)]
mod v1_result_compat {
    use super::*;

    #[derive(Serialize)]
    struct LaunchBindingWire {
        device_id: i64,
        stream_id: i64,
        kernel_id: i32,
        function_unique_id: i32,
        kernel_mangled_name: String,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        cluster: (u32, u32, u32),
    }

    fn binding_wire(binding: &ReplayLaunchCoverageBinding) -> Result<LaunchBindingWire, RunError> {
        let block = binding.block().ok_or_else(|| {
            RunError::result_serialization(format!(
                "admitted launch {:?} has no block dimensions",
                binding
            ))
        })?;
        Ok(LaunchBindingWire {
            device_id: binding.device_id(),
            stream_id: binding.stream_id(),
            kernel_id: binding.kernel_id(),
            function_unique_id: binding.function_unique_id(),
            kernel_mangled_name: binding.kernel_mangled_name().to_owned(),
            grid: binding.grid(),
            block,
            cluster: binding.cluster(),
        })
    }

    #[derive(Serialize)]
    #[serde(tag = "type", rename_all = "snake_case")]
    enum FullCoverageWire {
        ExactFull {
            blocks_observed: u64,
        },
        LosslessStored {
            blocks_observed: u64,
            selected_block_ids_sha256: String,
            sidecar_sha256: String,
        },
    }

    fn coverage_wire(coverage: &FullLaunchCoverage) -> FullCoverageWire {
        match coverage.source() {
            FullCoverageSource::UnsampledComplete => FullCoverageWire::ExactFull {
                blocks_observed: coverage.blocks_observed(),
            },
            FullCoverageSource::LosslessCompressedFull(provenance) => {
                FullCoverageWire::LosslessStored {
                    blocks_observed: coverage.blocks_observed(),
                    selected_block_ids_sha256: provenance.selected_block_ids_sha256().to_owned(),
                    sidecar_sha256: provenance.sidecar_sha256().to_owned(),
                }
            }
        }
    }

    #[derive(Serialize)]
    #[serde(tag = "status", rename_all = "snake_case")]
    enum ModeledEventTimeWire {
        Modeled { value: u64 },
        Unavailable { reason: &'static str },
    }

    fn modeled_event_time_wire(value: Option<u64>) -> ModeledEventTimeWire {
        match value {
            Some(value) => ModeledEventTimeWire::Modeled { value },
            None => ModeledEventTimeWire::Unavailable {
                reason: "no_modeled_issue_or_system_event",
            },
        }
    }

    #[derive(Serialize)]
    struct TimingDiagnosticWire {
        diagnostic_last_issue_cycles: u64,
        terminal_modeled_event_time: ModeledEventTimeWire,
        binding_cycles_role: &'static str,
    }

    #[derive(Serialize)]
    struct TailFlagsWire {
        red_rmw: bool,
        dirty_writeback: bool,
        coherence: bool,
        ldgsts_dst_shared_timing: bool,
    }

    impl From<TailFlags> for TailFlagsWire {
        fn from(flags: TailFlags) -> Self {
            Self {
                red_rmw: flags.red_rmw,
                dirty_writeback: flags.dirty_writeback,
                coherence: flags.coherence,
                ldgsts_dst_shared_timing: flags.ldgsts_dst_shared_timing,
            }
        }
    }

    #[derive(Serialize)]
    struct OriginPair<T> {
        global: T,
        local: T,
    }

    #[derive(Serialize)]
    struct L1OriginCountersWire {
        hits: u64,
        misses: u64,
        merges: u64,
        stall_events: u64,
        line_fills: u64,
        line_alloc_misses: u64,
        sector_resident_misses: u64,
        bypass_sectors: u64,
        bypass_atom_sectors: u64,
        untagged_sectors: u64,
        store_sectors: u64,
        store_hits: u64,
        store_tag_hits: u64,
        reduction_sectors: u64,
        reduction_hits: u64,
    }

    impl From<L1OriginCounters> for L1OriginCountersWire {
        fn from(counters: L1OriginCounters) -> Self {
            Self {
                hits: counters.hits,
                misses: counters.misses,
                merges: counters.merges,
                stall_events: counters.stall_events,
                line_fills: counters.line_fills,
                line_alloc_misses: counters.line_alloc_misses,
                sector_resident_misses: counters.sector_resident_misses,
                bypass_sectors: counters.bypass_sectors,
                bypass_atom_sectors: counters.bypass_atom_sectors,
                untagged_sectors: counters.untagged_sectors,
                store_sectors: counters.store_sectors,
                store_hits: counters.store_hits,
                store_tag_hits: counters.store_tag_hits,
                reduction_sectors: counters.reduction_sectors,
                reduction_hits: counters.reduction_hits,
            }
        }
    }

    #[derive(Serialize)]
    struct L1CountersWire {
        hits: u64,
        misses: u64,
        merges: u64,
        stall_events: u64,
        line_fills: u64,
        line_alloc_misses: u64,
        sector_resident_misses: u64,
        bypass_sectors: u64,
        bypass_atom_sectors: u64,
        untagged_sectors: u64,
        store_sectors: u64,
        store_hits: u64,
        store_tag_hits: u64,
        reduction_sectors: u64,
        reduction_hits: u64,
        by_origin: OriginPair<L1OriginCountersWire>,
    }

    impl From<&L1Counters> for L1CountersWire {
        fn from(counters: &L1Counters) -> Self {
            Self {
                hits: counters.hits,
                misses: counters.misses,
                merges: counters.merges,
                stall_events: counters.stall_events,
                line_fills: counters.line_fills,
                line_alloc_misses: counters.line_alloc_misses,
                sector_resident_misses: counters.sector_resident_misses,
                bypass_sectors: counters.bypass_sectors,
                bypass_atom_sectors: counters.bypass_atom_sectors,
                untagged_sectors: counters.untagged_sectors,
                store_sectors: counters.store_sectors,
                store_hits: counters.store_hits,
                store_tag_hits: counters.store_tag_hits,
                reduction_sectors: counters.reduction_sectors,
                reduction_hits: counters.reduction_hits,
                by_origin: OriginPair {
                    global: counters.origin(MemoryOrigin::Global).into(),
                    local: counters.origin(MemoryOrigin::Local).into(),
                },
            }
        }
    }

    #[derive(Serialize)]
    struct L2CountersWire {
        hits: u64,
        misses: u64,
        line_misses: u64,
        sector_fill_misses: u64,
        write_tag_hits: u64,
    }

    impl From<&L2Counters> for L2CountersWire {
        fn from(counters: &L2Counters) -> Self {
            Self {
                hits: counters.hits,
                misses: counters.misses,
                line_misses: counters.line_misses,
                sector_fill_misses: counters.sector_fill_misses,
                write_tag_hits: counters.write_tag_hits,
            }
        }
    }

    #[derive(Serialize)]
    struct L2SourceLaneWire {
        l1_fill: u64,
        bypass: u64,
        untagged: u64,
        store: u64,
    }

    impl L2SourceLaneWire {
        fn from_array(values: [u64; 4]) -> Self {
            Self {
                l1_fill: values[0],
                bypass: values[1],
                untagged: values[2],
                store: values[3],
            }
        }
    }

    #[derive(Serialize)]
    struct L2SourcePairWire {
        sectors: L2SourceLaneWire,
        hits: L2SourceLaneWire,
    }

    fn l2_origin_wire(stats: L2OriginSourceStats) -> L2SourcePairWire {
        L2SourcePairWire {
            sectors: L2SourceLaneWire::from_array(stats.sectors),
            hits: L2SourceLaneWire::from_array(stats.hits),
        }
    }

    #[derive(Serialize)]
    struct L2SourceSplitWire {
        sectors: L2SourceLaneWire,
        hits: L2SourceLaneWire,
        by_origin: OriginPair<L2SourcePairWire>,
    }

    impl From<&L2SourceSplit> for L2SourceSplitWire {
        fn from(split: &L2SourceSplit) -> Self {
            let sectors = L2SourceLaneWire {
                l1_fill: split.source(L2Source::L1Fill).0,
                bypass: split.source(L2Source::Bypass).0,
                untagged: split.source(L2Source::Untagged).0,
                store: split.source(L2Source::Store).0,
            };
            let hits = L2SourceLaneWire {
                l1_fill: split.source(L2Source::L1Fill).1,
                bypass: split.source(L2Source::Bypass).1,
                untagged: split.source(L2Source::Untagged).1,
                store: split.source(L2Source::Store).1,
            };
            Self {
                sectors,
                hits,
                by_origin: OriginPair {
                    global: l2_origin_wire(split.origin(MemoryOrigin::Global)),
                    local: l2_origin_wire(split.origin(MemoryOrigin::Local)),
                },
            }
        }
    }

    #[derive(Serialize)]
    struct DramOriginStatsWire {
        read_sectors: u64,
        bw_delayed_responses: u64,
    }

    impl From<DramOriginStats> for DramOriginStatsWire {
        fn from(stats: DramOriginStats) -> Self {
            Self {
                read_sectors: stats.read_sectors,
                bw_delayed_responses: stats.bw_delayed_responses,
            }
        }
    }

    #[derive(Serialize)]
    struct DramStatsWire {
        read_sectors: u64,
        bw_delayed_responses: u64,
        by_origin: OriginPair<DramOriginStatsWire>,
    }

    impl From<&DramStats> for DramStatsWire {
        fn from(stats: &DramStats) -> Self {
            Self {
                read_sectors: stats.read_sectors,
                bw_delayed_responses: stats.bw_delayed_responses,
                by_origin: OriginPair {
                    global: stats.origin(MemoryOrigin::Global).into(),
                    local: stats.origin(MemoryOrigin::Local).into(),
                },
            }
        }
    }

    #[derive(Serialize)]
    struct CounterBundleWire {
        instructions_issued: u64,
        records_seen: u64,
        l1: Availability<L1CountersWire>,
        l2: Availability<L2CountersWire>,
        l2_by_source: Availability<L2SourceSplitWire>,
        dram: Availability<DramStatsWire>,
    }

    fn counter_bundle_wire(
        instructions_issued: u64,
        records_seen: u64,
        l1: Option<&L1Counters>,
        l2: Option<&L2Counters>,
        l2_by_source: Option<&L2SourceSplit>,
        dram: Option<&DramStats>,
    ) -> CounterBundleWire {
        CounterBundleWire {
            instructions_issued,
            records_seen,
            l1: match l1 {
                Some(value) => Availability::Modeled {
                    value: value.into(),
                },
                None => Availability::Unmodeled {
                    reason: "replay_mode_without_l1",
                },
            },
            l2: match l2 {
                Some(value) => Availability::Modeled {
                    value: value.into(),
                },
                None => Availability::Unmodeled {
                    reason: "replay_mode_without_shared_l2",
                },
            },
            l2_by_source: match l2_by_source {
                Some(value) => Availability::Modeled {
                    value: value.into(),
                },
                None => Availability::Unmodeled {
                    reason: "replay_mode_without_shared_l2",
                },
            },
            dram: match dram {
                Some(value) => Availability::Modeled {
                    value: value.into(),
                },
                None => Availability::Unmodeled {
                    reason: "replay_mode_without_shared_l2",
                },
            },
        }
    }

    #[derive(Serialize)]
    struct PerLaunchWire {
        launch_evidence_manifest_path: String,
        expected_identity_sha256: Sha256Digest,
        admitted_identity: serde_json::Value,
        launch: LaunchBindingWire,
        coverage: FullCoverageWire,
        counters: CounterBundleWire,
        timing_diagnostics: TimingDiagnosticWire,
        unmodeled_tails: TailFlagsWire,
    }

    fn per_launch_wire(
        entry: &AdmittedEntry,
        result: &FullKernelReplayResult,
    ) -> Result<PerLaunchWire, RunError> {
        Ok(PerLaunchWire {
            launch_evidence_manifest_path: entry.relative_manifest_path.clone(),
            expected_identity_sha256: entry.expected_identity_sha256,
            admitted_identity: entry.identity_json.clone(),
            launch: binding_wire(result.coverage().launch())?,
            coverage: coverage_wire(result.coverage()),
            counters: counter_bundle_wire(
                result.instructions_issued(),
                result.records_seen(),
                result.l1(),
                result.l2(),
                result.l2_by_source(),
                result.dram(),
            ),
            timing_diagnostics: TimingDiagnosticWire {
                diagnostic_last_issue_cycles: result.diagnostic_last_issue_cycles(),
                terminal_modeled_event_time: modeled_event_time_wire(
                    result.terminal_modeled_event_time(),
                ),
                binding_cycles_role: "diagnostic_only_not_binding_prediction",
            },
            unmodeled_tails: result.unmodeled_tails().into(),
        })
    }

    #[derive(Serialize)]
    struct AggregateWire {
        estimator_provenance: &'static str,
        launch_count: usize,
        counters: CounterBundleWire,
        unmodeled_tails: TailFlagsWire,
        timing_composition: &'static str,
    }

    fn aggregate_wire(aggregate: &FullLaunchSetCounters) -> AggregateWire {
        AggregateWire {
            estimator_provenance: "exact_full_coverage_replay",
            launch_count: aggregate.launch_coverages().len(),
            counters: counter_bundle_wire(
                aggregate.instructions_issued(),
                aggregate.records_seen(),
                aggregate.l1(),
                aggregate.l2(),
                aggregate.l2_by_source(),
                aggregate.dram(),
            ),
            unmodeled_tails: aggregate.unmodeled_tails().into(),
            timing_composition: "required_unimplemented_pending_aggregation_semantics_study",
        }
    }

    #[derive(Serialize)]
    struct ArchitectureWire<'a> {
        arch_id: &'static str,
        sku: &'a str,
    }

    #[derive(Serialize)]
    struct ConfigurationEvidenceWire {
        arch_card_sha256: Sha256Digest,
        pchase_card_sha256: Sha256Digest,
    }

    #[derive(Serialize)]
    struct SimResultWire<'a> {
        schema_version: &'static str,
        run_manifest_sha256: Sha256Digest,
        scope: &'a str,
        architecture: ArchitectureWire<'a>,
        configuration_evidence: ConfigurationEvidenceWire,
        resolved_configuration: ResolvedConfigurationWire,
        launches: Vec<PerLaunchWire>,
        aggregate: AggregateWire,
        emissions: serde_json::Value,
        scientific_authority: &'static str,
        hardware_readiness: &'static str,
    }

    /// Private byte-golden compatibility composer for the historical v1 fixture.
    /// Production callers cannot reach this path.
    pub(super) fn run_manifest_v1_compat(path: impl AsRef<Path>) -> Result<SimRunOutput, RunError> {
        let path = path.as_ref();
        let (manifest, run_manifest_sha256) = read_manifest(path)?;
        let (arch, sku) = parse_architecture(&manifest, path)?;
        let mut admitted = admit_all(path, &manifest, arch, sku)?;
        let first = admitted
            .first()
            .ok_or_else(|| manifest_rejected(path, "launches must be nonempty"))?;
        let resolved = resolve_configuration(&first.evidence, arch, manifest.replay_mode)?;
        let expected_launches = admitted
            .iter()
            .map(|entry| entry.binding.clone())
            .collect::<Vec<_>>();
        let declaration = FullLaunchSetDeclaration::checked(
            manifest.scope.clone(),
            resolved.replay_configuration,
            expected_launches,
        )
        .map_err(|source| RunError::replay(format!("launch-set declaration: {source}")))?;

        let mut per_launch = Vec::new();
        let mut full_results = Vec::new();
        per_launch
            .try_reserve_exact(admitted.len())
            .map_err(result_serialization_error)?;
        full_results
            .try_reserve_exact(admitted.len())
            .map_err(result_serialization_error)?;
        for entry in admitted.drain(..) {
            let seal = entry.evidence.identity().seal_sha256();
            let outcome = replay_kernel(
                &entry.evidence,
                &resolved.issue_params,
                resolved.replay_mode,
            )
            .map_err(|source| {
                RunError::replay(format!("launch {seal} replay execution: {source}"))
            })?;
            match outcome {
                KernelReplayOutcome::Full(result) => {
                    per_launch.push(per_launch_wire(&entry, &result)?);
                    full_results.push(result);
                }
                KernelReplayOutcome::SampledObserved(result) => {
                    return Err(RunError::replay(format!(
                        "launch {seal} is sampled ({}/{}) and cannot enter exact aggregation",
                        result.coverage().blocks_observed(),
                        result.coverage().population_blocks()
                    )));
                }
            }
        }

        let aggregate = aggregate_full_launch_set(declaration, full_results)
            .map_err(|source| RunError::replay(format!("launch-set aggregation: {source}")))?;
        let predictions =
            emit_all(&aggregate).map_err(|source| RunError::emission(source.to_string()))?;
        if predictions.launch_timing_diagnostics() != aggregate.launch_timing_diagnostics() {
            return Err(RunError::emission(
                "emitter did not preserve canonical launch timing diagnostics",
            ));
        }
        let emissions =
            serde_json::to_value(predictions.metrics()).map_err(result_serialization_error)?;
        let aggregate_wire = aggregate_wire(&aggregate);
        let result = SimResultWire {
            schema_version: RESULT_SCHEMA_VERSION,
            run_manifest_sha256,
            scope: &manifest.scope,
            architecture: ArchitectureWire {
                arch_id: arch.token(),
                sku: sku.token(),
            },
            configuration_evidence: ConfigurationEvidenceWire {
                arch_card_sha256: manifest.configuration_evidence.arch_card_sha256,
                pchase_card_sha256: manifest.configuration_evidence.pchase_card_sha256,
            },
            resolved_configuration: resolved.wire,
            launches: per_launch,
            aggregate: aggregate_wire,
            emissions,
            scientific_authority: "not_conferred",
            hardware_readiness: "not_asserted",
        };
        let value = serde_json::to_value(result).map_err(result_serialization_error)?;
        let sealed = seal_document(value, "result_sha256")?;
        Ok(SimRunOutput {
            canonical_json: sealed.canonical_json,
            result_sha256: sealed.seal,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_launch_evidence_roles_preserve_error_boundaries_and_exits() {
        let path = Path::new("fixture/launch_evidence.json");
        let cases = [
            (
                ManifestError::ArtifactPathDuplicate {
                    artifact: ManifestArtifactToken::ArchCard,
                    path: "cards/shared.json".into(),
                },
                "ARCH_CARD_ADMISSION_FAILED",
                "arch_card_admission",
                1,
            ),
            (
                ManifestError::ArtifactFileAlias {
                    artifact: ManifestArtifactToken::PchaseCard,
                    first: "cards/arch.json".into(),
                    second: "cards/pchase.json".into(),
                },
                "CONFIGURATION_RESOLUTION_FAILED",
                "configuration_resolution",
                1,
            ),
            (
                ManifestError::IdentityMismatch {
                    artifact: ManifestArtifactToken::DynamicTracePb,
                    edge: "trace identity",
                    expected: "expected".to_owned(),
                    actual: "actual".to_owned(),
                },
                "TRACE_ADMISSION_FAILED",
                "trace_admission",
                1,
            ),
            (
                ManifestError::IdentityMismatch {
                    artifact: ManifestArtifactToken::ArchCard,
                    edge: "architecture-card identity",
                    expected: "sm_86".to_owned(),
                    actual: "sm_120".to_owned(),
                },
                "ARCH_CARD_ADMISSION_FAILED",
                "arch_card_admission",
                1,
            ),
            (
                ManifestError::IdentityMismatch {
                    artifact: ManifestArtifactToken::PchaseCard,
                    edge: "P-chase identity",
                    expected: "sm_86".to_owned(),
                    actual: "sm_120".to_owned(),
                },
                "CONFIGURATION_RESOLUTION_FAILED",
                "configuration_resolution",
                1,
            ),
        ];
        for (source, failure_type, stage, exit_code) in cases {
            let error = launch_evidence_error(path, source);
            assert_eq!(error.failure_type(), failure_type);
            assert_eq!(error.stage(), stage);
            assert_eq!(error.exit_code(), exit_code);
        }
    }

    #[test]
    fn architecture_card_placement_coverage_retains_typed_availability() {
        let path = Path::new("fixture/launch_evidence.json");
        let unavailable = launch_evidence_error(
            path,
            ManifestError::Coverage(ReplayError::ArchitectureCardPlacement {
                source: CalibError::BlockResearchGap {
                    arch_id: "sm_86".to_owned(),
                    block: "placement",
                    reason: "fixture gap".to_owned(),
                    challenge: None,
                },
            }),
        );
        assert_eq!(unavailable.failure_type(), "ARCH_CARD_ADMISSION_FAILED");
        assert_eq!(unavailable.stage(), "arch_card_admission");
        assert_eq!(unavailable.exit_code(), 2);

        let malformed = launch_evidence_error(
            path,
            ManifestError::Coverage(ReplayError::ArchitectureCardPlacement {
                source: CalibError::InvalidCardValue {
                    field: "sm_count",
                    value: "0".to_owned(),
                    requirement: "must be positive",
                },
            }),
        );
        assert_eq!(malformed.failure_type(), "ARCH_CARD_ADMISSION_FAILED");
        assert_eq!(malformed.stage(), "arch_card_admission");
        assert_eq!(malformed.exit_code(), 1);

        let ordinary_coverage = launch_evidence_error(
            path,
            ManifestError::Coverage(ReplayError::SmIdOutOfRange {
                path: "trace/block.pb".into(),
                sm_id: 84,
                arch: ArchId::Sm86,
                sku: Sku::Rtx3090Ti,
                sm_count: 84,
            }),
        );
        assert_eq!(ordinary_coverage.failure_type(), "TRACE_ADMISSION_FAILED");
        assert_eq!(ordinary_coverage.stage(), "trace_admission");
        assert_eq!(ordinary_coverage.exit_code(), 1);
    }

    #[test]
    fn shared_memory_card_authority_failures_are_exact_exit_two_classifications() {
        let path = Path::new("fixture/launch_evidence.json");
        let cases = [
            ManifestError::Coverage(ReplayError::ArchitectureCardSharedMemoryCapacity {
                arch: ArchId::Sm120,
                source: CalibError::BlockResearchGap {
                    arch_id: "sm_120".to_owned(),
                    block: "l1_geometry",
                    reason: "fixture gap".to_owned(),
                    challenge: None,
                },
            }),
            ManifestError::Coverage(ReplayError::SharedMemoryCapacityInvalid {
                arch: ArchId::Sm86,
                bytes: 0,
            }),
            ManifestError::Coverage(ReplayError::CtaSharedMemoryCapacityExceeded {
                kernel_id: 1,
                arch: ArchId::Sm86,
                bytes_per_cta: 102_401,
                max_bytes_per_sm: 102_400,
            }),
        ];
        for source in cases {
            let error = launch_evidence_error(path, source);
            assert_eq!(error.failure_type(), "ARCH_CARD_ADMISSION_FAILED");
            assert_eq!(error.stage(), "arch_card_admission");
            assert_eq!(error.exit_code(), 2);

            let encoded: serde_json::Value =
                serde_json::from_str(&error.canonical_json().unwrap()).unwrap();
            assert_eq!(
                encoded["error"]["failure_type"],
                "ARCH_CARD_ADMISSION_FAILED"
            );
            assert_eq!(encoded["error"]["stage"], "arch_card_admission");
            assert_eq!(encoded["exit_status"]["exit_code"], 2);
        }

        let trace_owned = launch_evidence_error(
            path,
            ManifestError::Coverage(ReplayError::SharedMemoryDemandInvalid {
                kernel_id: 1,
                bytes: -1,
            }),
        );
        assert_eq!(trace_owned.failure_type(), "TRACE_ADMISSION_FAILED");
        assert_eq!(trace_owned.stage(), "trace_admission");
        assert_eq!(trace_owned.exit_code(), 1);
    }

    #[test]
    fn relative_path_contract_is_fail_closed() {
        for valid in ["launch.json", "bundle/launch-evidence_v2.json"] {
            assert!(CanonicalRelativePath::checked(valid.to_owned()).is_ok());
        }
        for invalid in [
            "",
            "/absolute.json",
            "trailing/",
            "a//b",
            "./a",
            "a/../b",
            "a\\b",
            "white space.json",
        ] {
            assert!(CanonicalRelativePath::checked(invalid.to_owned()).is_err());
        }
    }

    fn assert_error_seal(raw: &str) -> serde_json::Value {
        let mut parsed: serde_json::Value =
            serde_json::from_str(raw).expect("error document parses");
        let claimed = parsed
            .as_object_mut()
            .expect("error root object")
            .remove("error_sha256")
            .and_then(|value| value.as_str().map(str::to_owned))
            .expect("string error seal");
        let unsigned = sim_trace::canonical_json_utf8(&parsed).expect("canonical unsigned error");
        assert_eq!(Sha256Digest::of(unsigned.as_bytes()).to_string(), claimed);
        parsed
    }

    #[test]
    fn every_error_kind_serializes_to_its_fixed_v2_pair() {
        let cases = [
            (
                RunError::cli_usage("wrong arity", 3),
                "CLI_USAGE_FAILED",
                "cli_usage",
                2,
            ),
            (
                RunError::manifest_admission("manifest", false),
                "MANIFEST_ADMISSION_FAILED",
                "manifest_admission",
                1,
            ),
            (
                RunError::trace_admission("trace", false),
                "TRACE_ADMISSION_FAILED",
                "trace_admission",
                1,
            ),
            (
                RunError::arch_card_admission("arch card", false),
                "ARCH_CARD_ADMISSION_FAILED",
                "arch_card_admission",
                1,
            ),
            (
                RunError::configuration_resolution("configuration", false),
                "CONFIGURATION_RESOLUTION_FAILED",
                "configuration_resolution",
                1,
            ),
            (RunError::replay("replay"), "REPLAY_FAILED", "replay", 1),
            (
                RunError::emission("emission"),
                "EMISSION_FAILED",
                "emission",
                1,
            ),
            (
                RunError::result_serialization("result"),
                "RESULT_SERIALIZATION_FAILED",
                "result_serialization",
                1,
            ),
            (
                RunError::output_write("output"),
                "OUTPUT_WRITE_FAILED",
                "output_write",
                1,
            ),
        ];

        for (error, failure_type, stage, exit_code) in cases {
            let raw = error
                .canonical_json()
                .expect("typed error document canonicalizes");
            let parsed: serde_json::Value =
                serde_json::from_str(&raw).expect("typed error document parses");
            assert_eq!(parsed["schema_version"], "sim_error.v2");
            assert_eq!(parsed["error"]["failure_type"], failure_type);
            assert_eq!(parsed["error"]["stage"], stage);
            assert_eq!(parsed["exit_status"]["exit_code"], exit_code);
            assert_eq!(error.exit_code(), exit_code);
            assert!(parsed.get("build_identity").is_none());
            if failure_type == "CLI_USAGE_FAILED" {
                assert_eq!(
                    parsed["error"]["evidence"]["provided_positional_argument_count"],
                    3
                );
                assert_eq!(
                    parsed["error"]["evidence"]["expected_positional_argument_count"],
                    1
                );
            } else {
                assert!(parsed["error"].get("evidence").is_none());
            }
            assert_error_seal(&raw);
        }
    }

    #[test]
    fn error_encoding_is_utf8_canonical_and_fallback_is_prevalidated() {
        let error = RunError::output_write("could not write café β");
        let raw: Result<String, ErrorEncodingError> = error.canonical_json();
        let raw = raw.expect("Unicode error document canonicalizes");
        assert!(raw.contains("café β"));
        assert!(!raw.contains("\\u00e9"));
        let parsed: serde_json::Value =
            serde_json::from_str(&raw).expect("Unicode error document parses");
        assert_eq!(
            sim_trace::canonical_json_utf8(&parsed).expect("recanonicalize"),
            raw
        );
        assert_error_seal(&raw);

        let fallback = RunError::canonical_serialization_fallback();
        let fallback_parsed = assert_error_seal(fallback);
        assert_eq!(fallback_parsed["schema_version"], "sim_error.v2");
        assert_eq!(
            fallback_parsed["error"]["failure_type"],
            "RESULT_SERIALIZATION_FAILED"
        );
        assert_eq!(fallback_parsed["error"]["stage"], "result_serialization");
        assert_eq!(fallback_parsed["exit_status"]["exit_code"], 1);
    }
}
