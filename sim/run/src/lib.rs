//! Manifest-only production composition for admitted simulator launches.
//!
//! The runner accepts no command-line overrides, environment-selected modes, or
//! target values. Every replay input is admitted through `sim-sm`, exact launch
//! sets use the checked aggregation algebra, and binding cycles are emitted by
//! the explicit max-of-launch-frontiers rule in [`emit`].

pub mod emit;
mod build_provenance {
    include!(concat!(env!("OUT_DIR"), "/build_provenance.rs"));
}
mod runner_v2;
pub(crate) mod wire;

use std::ffi::OsStr;
use std::path::{Component, Path};

use serde::{Deserialize, Deserializer, Serialize, de};
use sim_arch::ArchId;
use sim_arch::card::CalibError;
use sim_mem::{
    L1Config, L2Config, L2ConfigError, L2Replacement, PchaseCardError, SliceHash, WalkGeometry,
};
use sim_sm::{
    AdmittedLaunchEvidence, DramBw, IssueError, ManifestArtifactToken, ManifestError,
    ReplayConfiguration, ReplayError, ReplayMemMode,
};
use sim_trace::{AnchoredReadError, Sha256Digest, TraceError};

const MAX_RUN_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;

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
    // `SimErrorV2` is a closed, deterministic wire type. Canonicalizing the
    // serialized value once is sufficient; reparsing and hashing the same
    // bytes again only duplicates the error path's work.
    sim_trace::canonical_json_utf8(&sealed_value).map_err(ErrorEncodingError::new)
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

fn result_serialization_error(error: impl std::fmt::Display) -> RunError {
    RunError::result_serialization(error.to_string())
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

#[derive(Clone, Copy)]
enum IdentityFailureBoundary {
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
                    | IssueError::CardedResourceMismatch { .. }
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
    // Prefer the complete carded scheduler lane. A pre-resource-block
    // historical trace remains readable through the explicit legacy
    // compatibility path; newly admitted cards with any resource block must
    // never borrow architecture defaults.
    let issue_params = match first.arch_card().required_issue_resources() {
        Ok(_) => sim_sm::IssueModelParams::from_card(arch, first.arch_card())
            .map_err(|source| issue_configuration_error("carded issue resources", source))?,
        Err(sim_arch::card::CalibError::BlockResearchGap { .. })
            if !first.arch_card().has_any_issue_resource_blocks() =>
        {
            sim_sm::IssueModelParams::from_legacy_card_latencies_for_arch(
                arch,
                latencies.l1_hit,
                latencies.l2_hit,
                latencies.dram,
                latencies.shared,
            )
            .map_err(|source| issue_configuration_error("issue hypotheses", source))?
        }
        Err(source) => return Err(calib_arch_card_error("issue resources", source)),
    };
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
                ReplayMemMode::L1SharedL2 {
                    l1,
                    l2,
                    dram,
                    coherence: None,
                },
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
    issue_ports: u32,
    dual_issue_supported: bool,
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
            issue_ports: params.issue_ports(),
            dual_issue_supported: params.dual_issue_supported(),
            l1_hit_latency_cycles: params.l1_hit_latency(),
            l1_miss_latency_cycles: params.l1_miss_latency(),
            global_flat_latency_cycles: params.global_flat_latency(),
            shared_write_barrier_latency_cycles: params.shared_write_barrier_latency(),
            non_memory_write_barrier_latency_cycles: params.non_memory_write_barrier_latency(),
            read_barrier_latency_cycles: params.read_barrier_latency(),
            max_resident_warps_per_sm: params.max_resident_warps_per_sm(),
            latency_provenance: if matches!(
                params.resource_source(),
                sim_sm::IssueResourceSource::Sm120Alpha
            ) {
                "sm120_alpha_m8_m9_envelope"
            } else {
                "architecture_card"
            },
            issue_hypothesis_provenance: params.resource_source().source_label(),
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
