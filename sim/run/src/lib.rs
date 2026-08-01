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
// Task 343 consumes this staging module; production remains on the v1 path here.
#[allow(dead_code)]
mod runner_v2;
#[allow(dead_code)]
pub(crate) mod wire;

use std::collections::BTreeSet;
use std::path::{Component, Path, PathBuf};

use crate::emit::emit_all;
use serde::{Deserialize, Deserializer, Serialize, de};
use sim_arch::card::CalibError;
use sim_arch::{ArchId, Sku};
use sim_mem::{
    L1Config, L1Counters, L1OriginCounters, L2Config, L2ConfigError, L2Counters, L2Replacement,
    L2Source, MemoryOrigin, PchaseCardError, SliceHash, WalkGeometry,
};
use sim_sm::{
    AdmittedLaunchEvidence, DramBw, DramOriginStats, DramStats, FullCoverageSource,
    FullKernelReplayResult, FullLaunchCoverage, FullLaunchSetCounters, FullLaunchSetDeclaration,
    IssueError, KernelReplayOutcome, L2OriginSourceStats, L2SourceSplit, ManifestError,
    ReplayConfiguration, ReplayError, ReplayLaunchCoverageBinding, ReplayMemMode, TailFlags,
    admit_launch_evidence, aggregate_full_launch_set, replay_kernel,
};
use sim_trace::{AnchoredDir, AnchoredReadError, Sha256Digest, TraceError, canonical_json};

const RUN_MANIFEST_SCHEMA_VERSION: u32 = 1;
const MAX_RUN_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
const MAX_LAUNCHES: usize = 4096;
const RESULT_SCHEMA_VERSION: &str = "sim_result.v1";
const ERROR_SCHEMA_VERSION: &str = "sim_error.v1";

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

#[derive(Debug, Deserialize)]
struct ManifestVersionEnvelope {
    schema_version: u32,
}

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

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestArchitecture {
    arch_id: String,
    sku: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestConfigurationEvidence {
    arch_card_sha256: Sha256Digest,
    pchase_card_sha256: Sha256Digest,
}

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

/// Closed production-runner failures. The stable wire code and exit class are
/// exposed separately from human-readable source diagnostics.
#[derive(Debug, thiserror::Error)]
pub enum RunError {
    #[error("could not read run manifest {path}: {source}")]
    RunManifestRead {
        path: PathBuf,
        #[source]
        source: AnchoredReadError,
    },
    #[error("run manifest {path} was rejected: {reason}")]
    RunManifestRejected { path: PathBuf, reason: String },
    #[error("launch evidence {path} was rejected: {source}")]
    LaunchEvidenceRejected {
        path: PathBuf,
        #[source]
        source: ManifestError,
    },
    #[error("identity edge {edge} mismatch: expected {expected}, got {actual}")]
    IdentityMismatch {
        edge: &'static str,
        expected: String,
        actual: String,
    },
    #[error("configuration prerequisite {component} is unavailable: {reason}")]
    ConfigurationUnavailable {
        component: &'static str,
        reason: String,
    },
    #[error("supplied configuration for {component} was rejected: {reason}")]
    ConfigurationRejected {
        component: &'static str,
        reason: String,
    },
    #[error("deployment architecture {arch} is not enabled by the production runner")]
    ArchitectureNotEnabled { arch: ArchId },
    #[error(
        "launch {identity_sha256} is sampled ({blocks_observed}/{population_blocks} blocks) and cannot enter exact aggregation"
    )]
    CoverageNotAggregatable {
        identity_sha256: Sha256Digest,
        population_blocks: u64,
        blocks_observed: u64,
    },
    #[error("replay failed for launch {identity_sha256}: {reason}")]
    ReplayFailed {
        identity_sha256: Sha256Digest,
        reason: String,
    },
    #[error("launch-set declaration was rejected: {reason}")]
    LaunchSetDeclarationRejected { reason: String },
    #[error("launch-set aggregation failed: {reason}")]
    LaunchSetAggregationFailed { reason: String },
    #[error("metric emission failed: {reason}")]
    EmissionFailed { reason: String },
    #[error("output serialization failed: {reason}")]
    OutputSerializationFailed { reason: String },
    #[error("output channel write failed: {reason}")]
    OutputWriteFailed { reason: String },
    #[error("command-line usage rejected: {reason}")]
    CliUsage { reason: String },
}

impl RunError {
    pub fn cli_usage(reason: impl Into<String>) -> Self {
        Self::CliUsage {
            reason: reason.into(),
        }
    }

    pub const fn code(&self) -> &'static str {
        match self {
            Self::RunManifestRead { .. } => "run_manifest_read",
            Self::RunManifestRejected { .. } => "run_manifest_rejected",
            Self::LaunchEvidenceRejected { .. } => "launch_evidence_rejected",
            Self::IdentityMismatch { .. } => "identity_mismatch",
            Self::ConfigurationUnavailable { .. } => "configuration_unavailable",
            Self::ConfigurationRejected { .. } => "configuration_rejected",
            Self::ArchitectureNotEnabled { .. } => "architecture_not_enabled",
            Self::CoverageNotAggregatable { .. } => "coverage_not_aggregatable",
            Self::ReplayFailed { .. } => "replay_failed",
            Self::LaunchSetDeclarationRejected { .. } => "launch_set_declaration_rejected",
            Self::LaunchSetAggregationFailed { .. } => "launch_set_aggregation_failed",
            Self::EmissionFailed { .. } => "emission_failed",
            Self::OutputSerializationFailed { .. } => "output_serialization_failed",
            Self::OutputWriteFailed { .. } => "output_write_failed",
            Self::CliUsage { .. } => "cli_usage",
        }
    }

    pub fn exit_code(&self) -> u8 {
        match self {
            Self::RunManifestRead { source, .. } if anchored_error_is_missing(source) => 2,
            Self::LaunchEvidenceRejected { source, .. } if manifest_error_is_missing(source) => 2,
            Self::ConfigurationUnavailable { .. }
            | Self::ArchitectureNotEnabled { .. }
            | Self::CliUsage { .. } => 2,
            _ => 1,
        }
    }

    pub fn canonical_json(&self) -> Result<String, RunError> {
        let category = if self.exit_code() == 2 {
            "prerequisite_missing"
        } else {
            "failure"
        };
        let value = serde_json::to_value(ErrorWire {
            schema_version: ERROR_SCHEMA_VERSION,
            code: self.code(),
            category,
            exit_code: self.exit_code(),
            message: self.to_string(),
        })
        .map_err(output_serialization_error)?;
        seal_document(value, "error_sha256").map(|sealed| sealed.canonical_json)
    }

    /// Infallible, sealed last-resort document used only when the normal typed
    /// error document itself could not be serialized. This fallback is an exit-1
    /// serialization failure regardless of the original error's exit class.
    pub fn canonical_serialization_fallback() -> String {
        const PREFIX: &str = "{\"category\":\"failure\",\"code\":\"output_serialization_failed\",";
        const SUFFIX: &str = "\"exit_code\":1,\"message\":\"could not serialize typed error document\",\"schema_version\":\"sim_error.v1\"}";
        let unsigned = format!("{PREFIX}{SUFFIX}");
        let seal = Sha256Digest::of(unsigned.as_bytes());
        format!("{PREFIX}\"error_sha256\":\"{seal}\",{SUFFIX}")
    }
}

fn anchored_error_is_missing(error: &AnchoredReadError) -> bool {
    matches!(
        error,
        AnchoredReadError::Io { source, .. } if source.kind() == std::io::ErrorKind::NotFound
    )
}

fn trace_error_is_missing(error: &TraceError) -> bool {
    match error {
        TraceError::Io { source, .. } => source.kind() == std::io::ErrorKind::NotFound,
        TraceError::AnchoredSnapshot(source) => anchored_error_is_missing(source),
        _ => false,
    }
}

fn manifest_error_is_missing(error: &ManifestError) -> bool {
    match error {
        ManifestError::Anchored(source) => anchored_error_is_missing(source),
        ManifestError::Trace(source) => trace_error_is_missing(source),
        _ => false,
    }
}

#[derive(Serialize)]
struct ErrorWire<'a> {
    schema_version: &'static str,
    code: &'a str,
    category: &'static str,
    exit_code: u8,
    message: String,
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

struct SealedDocument {
    canonical_json: String,
    seal: Sha256Digest,
}

fn output_serialization_error(error: impl std::fmt::Display) -> RunError {
    RunError::OutputSerializationFailed {
        reason: error.to_string(),
    }
}

fn seal_document(
    mut value: serde_json::Value,
    field: &'static str,
) -> Result<SealedDocument, RunError> {
    let unsigned = canonical_json(&value).map_err(output_serialization_error)?;
    let seal = Sha256Digest::of(unsigned.as_bytes());
    let object = value
        .as_object_mut()
        .ok_or_else(|| output_serialization_error("serialized document root is not an object"))?;
    object.insert(
        field.to_owned(),
        serde_json::Value::String(seal.to_string()),
    );
    let canonical_json = canonical_json(&value).map_err(output_serialization_error)?;
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

fn require_canonical_manifest_path(path: &Path) -> Result<(), RunError> {
    let raw = path.to_str().ok_or_else(|| RunError::RunManifestRejected {
        path: path.to_path_buf(),
        reason: "manifest path must be UTF-8".to_owned(),
    })?;
    let has_dot_component = raw
        .split('/')
        .any(|component| matches!(component, "." | ".."));
    if raw.is_empty() || raw == "/" || raw.ends_with('/') || raw.contains("//") || has_dot_component
    {
        return Err(RunError::RunManifestRejected {
            path: path.to_path_buf(),
            reason: "manifest path must use one literal canonical spelling".to_owned(),
        });
    }
    if path
        .components()
        .any(|component| !matches!(component, Component::RootDir | Component::Normal(_)))
    {
        return Err(RunError::RunManifestRejected {
            path: path.to_path_buf(),
            reason: "manifest path contains a non-canonical component".to_owned(),
        });
    }
    Ok(())
}

fn read_manifest(path: &Path) -> Result<(SimInputManifest, Sha256Digest), RunError> {
    require_canonical_manifest_path(path)?;
    let base = manifest_base(path);
    let name = path
        .file_name()
        .ok_or_else(|| RunError::RunManifestRejected {
            path: path.to_path_buf(),
            reason: "manifest path has no file name".to_owned(),
        })?;
    let reader =
        AnchoredDir::open(base.to_path_buf()).map_err(|source| RunError::RunManifestRead {
            path: path.to_path_buf(),
            source,
        })?;
    let file = reader
        .read_bounded_regular(Path::new(name), MAX_RUN_MANIFEST_BYTES)
        .map_err(|source| RunError::RunManifestRead {
            path: path.to_path_buf(),
            source,
        })?;
    let raw = file.into_raw();
    let envelope: ManifestVersionEnvelope =
        serde_json::from_slice(&raw).map_err(|source| RunError::RunManifestRejected {
            path: path.to_path_buf(),
            reason: source.to_string(),
        })?;
    if envelope.schema_version != RUN_MANIFEST_SCHEMA_VERSION {
        return Err(RunError::RunManifestRejected {
            path: path.to_path_buf(),
            reason: format!(
                "unsupported schema_version {}; expected {}",
                envelope.schema_version, RUN_MANIFEST_SCHEMA_VERSION
            ),
        });
    }
    let manifest: SimInputManifest =
        serde_json::from_slice(&raw).map_err(|source| RunError::RunManifestRejected {
            path: path.to_path_buf(),
            reason: source.to_string(),
        })?;
    if manifest.launches.is_empty() {
        return Err(RunError::RunManifestRejected {
            path: path.to_path_buf(),
            reason: "launches must be nonempty".to_owned(),
        });
    }
    if manifest.launches.len() > MAX_LAUNCHES {
        return Err(RunError::RunManifestRejected {
            path: path.to_path_buf(),
            reason: format!(
                "launch count {} exceeds implementation limit {MAX_LAUNCHES}",
                manifest.launches.len()
            ),
        });
    }
    let mut paths = BTreeSet::new();
    for launch in &manifest.launches {
        if !paths.insert(launch.launch_evidence_manifest_path.as_str()) {
            return Err(RunError::RunManifestRejected {
                path: path.to_path_buf(),
                reason: format!(
                    "duplicate launch_evidence_manifest_path {:?}",
                    launch.launch_evidence_manifest_path.as_str()
                ),
            });
        }
    }
    let semantic = serde_json::to_value(&manifest).map_err(output_serialization_error)?;
    let canonical = canonical_json(&semantic).map_err(output_serialization_error)?;
    Ok((manifest, Sha256Digest::of(canonical.as_bytes())))
}

struct AdmittedEntry {
    relative_manifest_path: String,
    expected_identity_sha256: Sha256Digest,
    binding: ReplayLaunchCoverageBinding,
    identity_json: serde_json::Value,
    evidence: AdmittedLaunchEvidence,
}

fn identity_mismatch(
    edge: &'static str,
    expected: impl ToString,
    actual: impl ToString,
) -> RunError {
    RunError::IdentityMismatch {
        edge,
        expected: expected.to_string(),
        actual: actual.to_string(),
    }
}

fn parse_architecture(manifest: &SimInputManifest, path: &Path) -> Result<(ArchId, Sku), RunError> {
    let arch = ArchId::parse_deploy(&manifest.architecture.arch_id).map_err(|source| {
        RunError::RunManifestRejected {
            path: path.to_path_buf(),
            reason: source.to_string(),
        }
    })?;
    let sku =
        Sku::parse(&manifest.architecture.sku).map_err(|source| RunError::RunManifestRejected {
            path: path.to_path_buf(),
            reason: source.to_string(),
        })?;
    if sku.arch() != arch {
        return Err(identity_mismatch(
            "run manifest sku->arch",
            arch,
            sku.arch(),
        ));
    }
    if arch != ArchId::Sm86 {
        return Err(RunError::ArchitectureNotEnabled { arch });
    }
    Ok((arch, sku))
}

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
        .map_err(output_serialization_error)?;
    for launch in &manifest.launches {
        let path = base.join(launch.launch_evidence_manifest_path.as_path());
        let evidence =
            admit_launch_evidence(&path).map_err(|source| RunError::LaunchEvidenceRejected {
                path: path.clone(),
                source,
            })?;
        let identity = evidence.identity();
        let actual_seal = identity.seal_sha256();
        if actual_seal != launch.expected_identity_sha256 {
            return Err(identity_mismatch(
                "launch expected_identity_sha256",
                launch.expected_identity_sha256,
                actual_seal,
            ));
        }
        if !seals.insert(actual_seal) {
            return Err(RunError::RunManifestRejected {
                path: manifest_path.to_path_buf(),
                reason: format!("duplicate launch evidence identity {actual_seal}"),
            });
        }
        let launch_arch = identity
            .arch()
            .map_err(|source| RunError::LaunchEvidenceRejected {
                path: path.clone(),
                source: ManifestError::Architecture(source),
            })?;
        let launch_sku = identity
            .sku()
            .map_err(|source| RunError::LaunchEvidenceRejected {
                path: path.clone(),
                source: ManifestError::Architecture(source),
            })?;
        if launch_arch != arch {
            return Err(identity_mismatch("launch architecture", arch, launch_arch));
        }
        if launch_sku != sku {
            return Err(identity_mismatch(
                "launch SKU",
                sku.token(),
                launch_sku.token(),
            ));
        }
        if identity.arch_card_sha256() != manifest.configuration_evidence.arch_card_sha256 {
            return Err(identity_mismatch(
                "launch architecture-card SHA-256",
                manifest.configuration_evidence.arch_card_sha256,
                identity.arch_card_sha256(),
            ));
        }
        if identity.pchase_card_sha256() != manifest.configuration_evidence.pchase_card_sha256 {
            return Err(identity_mismatch(
                "launch P-chase-card SHA-256",
                manifest.configuration_evidence.pchase_card_sha256,
                identity.pchase_card_sha256(),
            ));
        }
        let binding =
            identity
                .launch_binding()
                .map_err(|source| RunError::LaunchSetDeclarationRejected {
                    reason: format!(
                        "admitted launch identity {actual_seal} did not reconstruct: {source}"
                    ),
                })?;
        let identity_json =
            serde_json::from_str(identity.canonical_json()).map_err(output_serialization_error)?;
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

fn configuration_unavailable(component: &'static str, reason: impl ToString) -> RunError {
    RunError::ConfigurationUnavailable {
        component,
        reason: reason.to_string(),
    }
}

fn configuration_rejected(component: &'static str, reason: impl ToString) -> RunError {
    RunError::ConfigurationRejected {
        component,
        reason: reason.to_string(),
    }
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

fn calib_configuration_error(component: &'static str, error: CalibError) -> RunError {
    if calib_error_is_unavailable(&error) {
        configuration_unavailable(component, error)
    } else {
        configuration_rejected(component, error)
    }
}

fn issue_configuration_error(component: &'static str, error: IssueError) -> RunError {
    let unavailable = match &error {
        IssueError::ArchitectureUnsupported { .. }
        | IssueError::CardedResourceExecutionDeferred { .. }
        | IssueError::Capability(_) => true,
        IssueError::Calibration(source) => calib_error_is_unavailable(source),
        _ => false,
    };
    if unavailable {
        configuration_unavailable(component, error)
    } else {
        configuration_rejected(component, error)
    }
}

fn pchase_configuration_error(component: &'static str, error: PchaseCardError) -> RunError {
    let unavailable = match &error {
        PchaseCardError::Io { source, .. } => source.kind() == std::io::ErrorKind::NotFound,
        PchaseCardError::Capability(_)
        | PchaseCardError::PropertyMissing { .. }
        | PchaseCardError::TypedWithoutDefault { .. } => true,
        _ => false,
    };
    if unavailable {
        configuration_unavailable(component, error)
    } else {
        configuration_rejected(component, error)
    }
}

fn l2_configuration_error(component: &'static str, error: L2ConfigError) -> RunError {
    if matches!(
        &error,
        L2ConfigError::Capability(_) | L2ConfigError::WritePolicyGap
    ) {
        configuration_unavailable(component, error)
    } else {
        configuration_rejected(component, error)
    }
}

fn replay_configuration_error(component: &'static str, error: ReplayError) -> RunError {
    configuration_rejected(component, error)
}

fn resolve_configuration(
    first: &AdmittedLaunchEvidence,
    arch: ArchId,
    requested: ManifestReplayMode,
) -> Result<ResolvedConfiguration, RunError> {
    let latencies = first
        .arch_card()
        .required_latencies()
        .map_err(|source| calib_configuration_error("architecture-card latencies", source))?;
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
                .map_err(|source| calib_configuration_error("L2 geometry", source))?;
            let l2 = L2Config::from_carded_geometry_for_arch(
                arch,
                card_l2.capacity_bytes,
                card_l2.line_bytes,
                card_l2.sector_bytes,
                card_l2.num_slices,
                card_l2.policy.as_deref(),
            )
            .map_err(|source| l2_configuration_error("L2 configuration", source))?;
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
                        .map_err(|source| calib_configuration_error("DRAM bandwidth", source))?;
                    let bytes_per_sec = card_dram
                        .validated_peak_bytes_per_second()
                        .map_err(|source| calib_configuration_error("DRAM bandwidth", source))?;
                    let sm_clock_hz = card_dram
                        .locked_sm_clock_hz()
                        .map_err(|source| calib_configuration_error("DRAM locked clock", source))?;
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
    let block = binding
        .block()
        .ok_or_else(|| RunError::OutputSerializationFailed {
            reason: format!("admitted launch {:?} has no block dimensions", binding),
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

/// Admit, replay, aggregate, emit, and canonically seal one strict run manifest.
pub fn run_manifest(path: impl AsRef<Path>) -> Result<SimRunOutput, RunError> {
    let path = path.as_ref();
    let (manifest, run_manifest_sha256) = read_manifest(path)?;
    let (arch, sku) = parse_architecture(&manifest, path)?;
    let mut admitted = admit_all(path, &manifest, arch, sku)?;
    let first = admitted
        .first()
        .ok_or_else(|| RunError::RunManifestRejected {
            path: path.to_path_buf(),
            reason: "launches must be nonempty".to_owned(),
        })?;
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
    .map_err(|source| RunError::LaunchSetDeclarationRejected {
        reason: source.to_string(),
    })?;

    let mut per_launch = Vec::new();
    let mut full_results = Vec::new();
    per_launch
        .try_reserve_exact(admitted.len())
        .map_err(output_serialization_error)?;
    full_results
        .try_reserve_exact(admitted.len())
        .map_err(output_serialization_error)?;
    for entry in admitted.drain(..) {
        let seal = entry.evidence.identity().seal_sha256();
        let outcome = replay_kernel(
            &entry.evidence,
            &resolved.issue_params,
            resolved.replay_mode,
        )
        .map_err(|source| RunError::ReplayFailed {
            identity_sha256: seal,
            reason: source.to_string(),
        })?;
        match outcome {
            KernelReplayOutcome::Full(result) => {
                per_launch.push(per_launch_wire(&entry, &result)?);
                full_results.push(result);
            }
            KernelReplayOutcome::SampledObserved(result) => {
                return Err(RunError::CoverageNotAggregatable {
                    identity_sha256: seal,
                    population_blocks: result.coverage().population_blocks(),
                    blocks_observed: result.coverage().blocks_observed(),
                });
            }
        }
    }

    let aggregate = aggregate_full_launch_set(declaration, full_results).map_err(|source| {
        RunError::LaunchSetAggregationFailed {
            reason: source.to_string(),
        }
    })?;
    let predictions = emit_all(&aggregate).map_err(|source| RunError::EmissionFailed {
        reason: source.to_string(),
    })?;
    if predictions.launch_timing_diagnostics() != aggregate.launch_timing_diagnostics() {
        return Err(RunError::EmissionFailed {
            reason: "emitter did not preserve canonical launch timing diagnostics".to_owned(),
        });
    }
    let emissions =
        serde_json::to_value(predictions.metrics()).map_err(output_serialization_error)?;
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
    let value = serde_json::to_value(result).map_err(output_serialization_error)?;
    let sealed = seal_document(value, "result_sha256")?;
    Ok(SimRunOutput {
        canonical_json: sealed.canonical_json,
        result_sha256: sealed.seal,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn error_documents_are_canonical_and_sealed() {
        let error = RunError::ArchitectureNotEnabled {
            arch: ArchId::Sm120,
        };
        let raw = error
            .canonical_json()
            .expect("error document canonicalizes");
        let parsed: serde_json::Value = serde_json::from_str(&raw).expect("error document parses");
        assert_eq!(parsed["schema_version"], ERROR_SCHEMA_VERSION);
        assert_eq!(parsed["code"], "architecture_not_enabled");
        assert_eq!(parsed["exit_code"], 2);
        assert_eq!(canonical_json(&parsed).expect("recanonicalize"), raw);
    }
}
