//! Private strict-v2 production runner behind the public `run_manifest` entry point.

use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use serde::Serialize;
use sim_arch::{ArchId, Sku};
use sim_sm::{
    AdmittedLaunchEvidence, FullLaunchSetDeclaration, KernelReplayOutcome,
    ReplayLaunchCoverageBinding, TraceBundleIdentity, admit_launch_evidence_anchored,
    aggregate_full_launch_set, replay_kernel,
};
use sim_trace::{
    AnchoredDir, AnchoredFile, FileIdentity, RetainedFile, Sha256Digest, canonical_json_utf8,
};

use crate::build_provenance;
use crate::emit::{emit_all, emit_launch};
use crate::wire::{
    AggregateV2, ArchCardIdentityV2, BuildIdentityV2, ConfigurationIdentityV2, DiagnosticsV2,
    DramBandwidthV2, LaunchKeyV1, MetricMapV2, PerLaunchDiagnosticV2, PerLaunchResultV2,
    ReplayModeV2, SimResultV2, SimRunManifestV2, WireError, parse_sim_run_manifest_v2,
};
use crate::{
    IdentityFailureBoundary, MAX_RUN_MANIFEST_BYTES, ManifestDramBandwidth, ManifestReplayMode,
    RunError, SimRunOutput, identity_mismatch, launch_evidence_error, manifest_base,
    manifest_read_error, manifest_rejected, manifest_serialization_error,
    require_canonical_manifest_path, resolve_configuration, result_serialization_error,
};

const MAX_PROCESS_IMAGE_BYTES: u64 = 512 * 1024 * 1024;
const ARCH_CARD_ID_CONTRACT: &str = "sim-run.arch-card-content-id.v2";
const ARCH_CARD_ID_PREFIX: &str = "arch-card-v2-";
const CONFIGURATION_ID_CONTRACT: &str = "sim-run.configuration-content-id.v2";
const CONFIGURATION_ID_PREFIX: &str = "configuration-v2-";

#[derive(Debug, Clone)]
struct BuildContextV2 {
    sim_git_commit: String,
    sim_git_dirty: bool,
    binary_sha256: Sha256Digest,
    rustc_version: String,
    cargo_profile: String,
}

impl BuildContextV2 {
    fn runtime() -> Result<Self, RunError> {
        Ok(Self {
            sim_git_commit: build_provenance::SIM_GIT_COMMIT.to_owned(),
            sim_git_dirty: build_provenance::SIM_GIT_DIRTY,
            binary_sha256: hash_current_process_image()?,
            rustc_version: build_provenance::RUSTC_VERSION.to_owned(),
            cargo_profile: build_provenance::CARGO_PROFILE.to_owned(),
        })
    }

    fn checked_wire(&self) -> Result<BuildIdentityV2, RunError> {
        BuildIdentityV2::checked(
            self.sim_git_commit.clone(),
            self.sim_git_dirty,
            self.binary_sha256,
            self.rustc_version.clone(),
            self.cargo_profile.clone(),
        )
        .map_err(build_provenance_error)
    }
}

/// The one retained run-manifest parent, leaf identity, and exact admitted bytes.
struct RunManifestTransport {
    display_path: PathBuf,
    base: AnchoredDir,
    manifest_name: PathBuf,
    initial: AnchoredFile,
    retained: Option<RetainedFile>,
}

impl RunManifestTransport {
    fn from_path(path: &Path) -> Result<Self, RunError> {
        require_canonical_manifest_path(path)?;
        let base_path = manifest_base(path);
        let manifest_name = path
            .file_name()
            .ok_or_else(|| manifest_rejected(path, "manifest path has no file name"))?;
        let base = AnchoredDir::open(base_path.to_path_buf())
            .map_err(|source| manifest_read_error(path, source))?;
        let initial = base
            .read_bounded_regular(Path::new(manifest_name), MAX_RUN_MANIFEST_BYTES)
            .map_err(|source| manifest_read_error(path, source))?;
        Ok(Self {
            display_path: path.to_path_buf(),
            base,
            manifest_name: PathBuf::from(manifest_name),
            initial,
            retained: None,
        })
    }

    fn from_retained(
        manifest_fd: u32,
        evidence_base_fd: u32,
        manifest_name: &OsStr,
    ) -> Result<Self, RunError> {
        let manifest_name = PathBuf::from(manifest_name);
        require_canonical_manifest_path(&manifest_name)?;
        if manifest_name
            .parent()
            .is_some_and(|parent| !parent.as_os_str().is_empty())
            || manifest_name.file_name() != Some(manifest_name.as_os_str())
        {
            return Err(manifest_rejected(
                &manifest_name,
                "retained MANIFEST_NAME must contain exactly one normal file-name component",
            ));
        }

        let base_descriptor_path = PathBuf::from(format!("inherited-fd:{evidence_base_fd}"));
        let display_path = base_descriptor_path.join(&manifest_name);
        RetainedFile::validate_inherited(manifest_fd)
            .map_err(|source| manifest_read_error(&display_path, source))?;
        AnchoredDir::validate_inherited(evidence_base_fd)
            .map_err(|source| manifest_read_error(&display_path, source))?;
        let retained = RetainedFile::open(manifest_fd)
            .map_err(|source| manifest_read_error(&display_path, source))?;
        let base = AnchoredDir::open_retained(evidence_base_fd)
            .map_err(|source| manifest_read_error(&display_path, source))?;
        let initial = retained
            .read_bounded_regular(MAX_RUN_MANIFEST_BYTES)
            .map_err(|source| manifest_read_error(&display_path, source))?;
        let named_identity = base
            .inspect_regular_identity(&manifest_name)
            .map_err(|source| manifest_read_error(&display_path, source))?;
        Self::require_same_identity(
            &display_path,
            &initial,
            named_identity,
            "MANIFEST_FD and MANIFEST_NAME",
        )?;
        Ok(Self {
            display_path,
            base,
            manifest_name,
            initial,
            retained: Some(retained),
        })
    }

    fn require_same_snapshot(
        path: &Path,
        expected: &AnchoredFile,
        actual: &AnchoredFile,
        edge: &'static str,
    ) -> Result<(), RunError> {
        if actual.identity() != expected.identity() {
            return Err(manifest_rejected(
                path,
                format!("{edge} device/inode identity mismatch"),
            ));
        }
        if actual.raw() != expected.raw() {
            return Err(manifest_rejected(
                path,
                format!("{edge} exact-byte mismatch"),
            ));
        }
        Ok(())
    }

    fn require_same_identity(
        path: &Path,
        expected: &AnchoredFile,
        actual: FileIdentity,
        edge: &'static str,
    ) -> Result<(), RunError> {
        if actual != expected.identity() {
            return Err(manifest_rejected(
                path,
                format!("{edge} device/inode identity mismatch"),
            ));
        }
        Ok(())
    }

    fn recheck(&self) -> Result<(), RunError> {
        if let Some(retained) = &self.retained {
            let current = retained
                .read_bounded_regular(MAX_RUN_MANIFEST_BYTES)
                .map_err(|source| manifest_read_error(&self.display_path, source))?;
            Self::require_same_snapshot(
                &self.display_path,
                &self.initial,
                &current,
                "retained MANIFEST_FD final recheck",
            )?;
            let named_identity = self
                .base
                .inspect_regular_identity(&self.manifest_name)
                .map_err(|source| manifest_read_error(&self.display_path, source))?;
            return Self::require_same_identity(
                &self.display_path,
                &self.initial,
                named_identity,
                "MANIFEST_NAME final recheck",
            );
        }
        let named = self
            .base
            .read_bounded_regular(&self.manifest_name, MAX_RUN_MANIFEST_BYTES)
            .map_err(|source| manifest_read_error(&self.display_path, source))?;
        Self::require_same_snapshot(
            &self.display_path,
            &self.initial,
            &named,
            "MANIFEST_NAME final recheck",
        )
    }

    fn path(&self) -> &Path {
        &self.display_path
    }

    fn raw(&self) -> &[u8] {
        self.initial.raw()
    }
}

struct AdmittedEntryV2 {
    evidence_path: PathBuf,
    launch_key: LaunchKeyV1,
    binding: ReplayLaunchCoverageBinding,
    evidence: AdmittedLaunchEvidence,
}

#[derive(Debug)]
struct LaunchDiagnosticWitness {
    launch_key: LaunchKeyV1,
    binding: ReplayLaunchCoverageBinding,
    diagnostic_last_issue_cycles: u64,
    last_processed_modeled_system_event_time: Option<u64>,
    terminal_modeled_event_time: Option<u64>,
    sm_active_observation_frontier: Option<sim_sm::SmActiveObservationFrontier>,
    modeled_completion_frontier: sim_sm::ModeledCompletionFrontier,
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct ArchCardIdPreimage {
    digest_contract_id: &'static str,
    card_sha256: Sha256Digest,
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct ConfigurationIdPreimage {
    digest_contract_id: &'static str,
    resolved_configuration_sha256: Sha256Digest,
    arch_card_sha256: Sha256Digest,
    pchase_card_sha256: Sha256Digest,
}

/// Sole private strict-v2 integration entry used by the public path consumer.
pub(super) fn run_manifest_v2(path: impl AsRef<Path>) -> Result<SimRunOutput, RunError> {
    let build_context = BuildContextV2::runtime()?;
    run_manifest_v2_with_context(path.as_ref(), &build_context)
}

/// Strict-v2 integration entry for the retained-descriptor transport.
pub(super) fn run_manifest_v2_retained(
    manifest_fd: u32,
    evidence_base_fd: u32,
    manifest_name: &OsStr,
) -> Result<SimRunOutput, RunError> {
    let build_context = BuildContextV2::runtime()?;
    let transport =
        RunManifestTransport::from_retained(manifest_fd, evidence_base_fd, manifest_name)?;
    run_manifest_v2_with_transport(&transport, &build_context)
}

fn run_manifest_v2_with_context(
    path: &Path,
    build_context: &BuildContextV2,
) -> Result<SimRunOutput, RunError> {
    let transport = RunManifestTransport::from_path(path)?;
    run_manifest_v2_with_transport(&transport, build_context)
}

fn run_manifest_v2_with_transport(
    transport: &RunManifestTransport,
    build_context: &BuildContextV2,
) -> Result<SimRunOutput, RunError> {
    run_manifest_v2_with_before_publication(transport, build_context, || {})
}

fn run_manifest_v2_with_before_publication<F>(
    transport: &RunManifestTransport,
    build_context: &BuildContextV2,
    before_publication: F,
) -> Result<SimRunOutput, RunError>
where
    F: FnOnce(),
{
    let path = transport.path();
    let build_identity = build_context.checked_wire()?;
    let (manifest, run_manifest_sha256) = read_manifest_v2(transport)?;
    let arch = ArchId::Sm86;
    let sku = Sku::Rtx3090Ti;
    let (admitted, trace_identity) = admit_all_v2(path, &transport.base, &manifest, arch, sku)?;
    let first = admitted
        .first()
        .ok_or_else(|| manifest_rejected(path, "launches must be nonempty"))?;
    let requested_mode = manifest_replay_mode(manifest.replay_mode());
    let resolved = resolve_configuration(&first.evidence, arch, requested_mode)?;
    let resolved_configuration_sha256 = canonical_typed_sha256(&resolved.wire)?;

    let arch_card_sha256 = manifest.configuration_evidence().arch_card_sha256();
    let pchase_card_sha256 = manifest.configuration_evidence().pchase_card_sha256();
    let arch_card_identity =
        ArchCardIdentityV2::checked(derive_arch_card_id(arch_card_sha256)?, arch_card_sha256)
            .map_err(wire_serialization_error)?;
    let configuration_identity = ConfigurationIdentityV2::checked(
        derive_configuration_id(
            resolved_configuration_sha256,
            arch_card_sha256,
            pchase_card_sha256,
        )?,
        run_manifest_sha256,
        resolved_configuration_sha256,
        arch_card_sha256,
        pchase_card_sha256,
    )
    .map_err(wire_serialization_error)?;

    let expected_launches = admitted
        .iter()
        .map(|entry| entry.binding.clone())
        .collect::<Vec<_>>();
    let declaration = FullLaunchSetDeclaration::checked(
        manifest.scope().to_owned(),
        resolved.replay_configuration,
        expected_launches,
    )
    .map_err(|source| RunError::replay(format!("launch-set declaration: {source}")))?;

    let mut per_launch_results = Vec::new();
    let mut diagnostic_witnesses = Vec::new();
    let mut full_results = Vec::new();
    per_launch_results
        .try_reserve_exact(admitted.len())
        .map_err(result_serialization_error)?;
    diagnostic_witnesses
        .try_reserve_exact(admitted.len())
        .map_err(result_serialization_error)?;
    full_results
        .try_reserve_exact(admitted.len())
        .map_err(result_serialization_error)?;

    for entry in &admitted {
        let evidence_sha256 = entry.evidence.identity().seal_sha256();
        let outcome = replay_kernel(
            &entry.evidence,
            &resolved.issue_params,
            resolved.replay_mode,
        )
        .map_err(|source| {
            RunError::replay(format!(
                "launch {evidence_sha256} replay execution: {source}"
            ))
        })?;
        match outcome {
            KernelReplayOutcome::Full(result) => {
                if result.coverage().blocks_observed() == 0 || result.records_seen() == 0 {
                    return Err(RunError::replay(format!(
                        "launch {evidence_sha256} full replay was vacuous over blocks or records"
                    )));
                }
                let predictions = emit_launch(&result)
                    .map_err(|source| RunError::emission(source.to_string()))?;
                if predictions.coverage().launch() != &entry.binding {
                    return Err(RunError::emission(
                        "per-launch emitter changed the authenticated launch binding",
                    ));
                }
                let metrics = MetricMapV2::from_predictions(predictions.metrics())
                    .map_err(wire_emission_error)?;
                diagnostic_witnesses.push(LaunchDiagnosticWitness {
                    launch_key: entry.launch_key.clone(),
                    binding: entry.binding.clone(),
                    diagnostic_last_issue_cycles: predictions.diagnostic_last_issue_cycles(),
                    last_processed_modeled_system_event_time: predictions
                        .last_processed_modeled_system_event_time(),
                    terminal_modeled_event_time: predictions.terminal_modeled_event_time(),
                    sm_active_observation_frontier: result.sm_active_observation_frontier(),
                    modeled_completion_frontier: predictions.modeled_completion_frontier(),
                });
                per_launch_results.push(PerLaunchResultV2::new(entry.launch_key.clone(), metrics));
                // `emit_launch` must run while the exact replay result is still borrowed.
                full_results.push(result);
            }
            KernelReplayOutcome::SampledObserved(result) => {
                return Err(RunError::replay(format!(
                    "launch {evidence_sha256} is sampled ({}/{}) and cannot enter exact aggregation",
                    result.coverage().blocks_observed(),
                    result.coverage().population_blocks()
                )));
            }
        }
    }

    let aggregate_counters = aggregate_full_launch_set(declaration, full_results)
        .map_err(|source| RunError::replay(format!("launch-set aggregation: {source}")))?;
    let aggregate_predictions =
        emit_all(&aggregate_counters).map_err(|source| RunError::emission(source.to_string()))?;
    if aggregate_predictions.launch_timing_diagnostics()
        != aggregate_counters.launch_timing_diagnostics()
    {
        return Err(RunError::emission(
            "aggregate emitter did not preserve canonical timing diagnostics",
        ));
    }
    let aggregate_metrics = MetricMapV2::from_predictions(aggregate_predictions.metrics())
        .map_err(wire_emission_error)?;
    let aggregate = AggregateV2::checked(per_launch_results.len(), aggregate_metrics)
        .map_err(wire_serialization_error)?;

    let aggregate_diagnostics = aggregate_predictions.launch_timing_diagnostics();
    if aggregate_diagnostics.len() != diagnostic_witnesses.len() {
        return Err(RunError::emission(
            "aggregate timing-diagnostic count changed",
        ));
    }
    let mut per_launch_diagnostics = Vec::new();
    per_launch_diagnostics
        .try_reserve_exact(diagnostic_witnesses.len())
        .map_err(result_serialization_error)?;
    for (witness, aggregate_diagnostic) in diagnostic_witnesses
        .into_iter()
        .zip(aggregate_diagnostics.iter())
    {
        if aggregate_diagnostic.launch() != &witness.binding
            || aggregate_diagnostic.diagnostic_last_issue_cycles()
                != witness.diagnostic_last_issue_cycles
            || aggregate_diagnostic.terminal_modeled_event_time()
                != witness.terminal_modeled_event_time
            || aggregate_diagnostic.last_processed_modeled_system_event_time()
                != witness.last_processed_modeled_system_event_time
            || aggregate_diagnostic.modeled_completion_frontier()
                != witness.modeled_completion_frontier
        {
            return Err(RunError::emission(
                "aggregate diagnostics differ from per-launch emission diagnostics",
            ));
        }
        per_launch_diagnostics.push(PerLaunchDiagnosticV2::with_sm_active_frontier(
            witness.launch_key,
            witness.diagnostic_last_issue_cycles,
            witness.terminal_modeled_event_time,
            witness.sm_active_observation_frontier,
        ));
    }
    let diagnostics =
        DiagnosticsV2::checked(per_launch_diagnostics, None).map_err(wire_serialization_error)?;

    let output = seal_result_v2(
        build_identity,
        trace_identity,
        arch_card_identity,
        configuration_identity,
        resolved.wire,
        per_launch_results,
        aggregate,
        diagnostics,
    )?;
    before_publication();
    for entry in &admitted {
        entry
            .evidence
            .recheck_retained_generation()
            .map_err(|source| launch_evidence_error(&entry.evidence_path, source))?;
    }
    transport.recheck()?;
    Ok(output)
}

fn read_manifest_v2(
    transport: &RunManifestTransport,
) -> Result<(SimRunManifestV2, Sha256Digest), RunError> {
    let path = transport.path();
    let manifest = parse_sim_run_manifest_v2(transport.raw())
        .map_err(|source| manifest_rejected(path, source))?;
    let manifest_value = serde_json::to_value(&manifest)
        .map_err(|source| manifest_serialization_error(path, source))?;
    let canonical = canonical_json_utf8(&manifest_value)
        .map_err(|source| manifest_serialization_error(path, source))?;
    let run_manifest_sha256 = Sha256Digest::of(canonical.as_bytes());
    Ok((manifest, run_manifest_sha256))
}

fn admit_all_v2(
    manifest_path: &Path,
    base: &AnchoredDir,
    manifest: &SimRunManifestV2,
    arch: ArchId,
    sku: Sku,
) -> Result<(Vec<AdmittedEntryV2>, TraceBundleIdentity), RunError> {
    let mut evidence_identities = BTreeSet::new();
    let mut shared_trace_identity: Option<TraceBundleIdentity> = None;
    let mut admitted = Vec::new();
    admitted
        .try_reserve_exact(manifest.launches().len())
        .map_err(result_serialization_error)?;

    for launch in manifest.launches() {
        let relative = Path::new(launch.launch_evidence_manifest_path());
        let path = base.root().join(relative);
        if relative.file_name().is_none() {
            return Err(manifest_rejected(
                manifest_path,
                format!(
                    "launch evidence path {:?} has no file name",
                    launch.launch_evidence_manifest_path()
                ),
            ));
        }
        let evidence = admit_launch_evidence_anchored(base, relative)
            .map_err(|source| launch_evidence_error(&path, source))?;
        let identity = evidence.identity();
        let actual_identity_sha256 = identity.seal_sha256();
        if actual_identity_sha256 != launch.expected_identity_sha256() {
            return Err(identity_mismatch(
                IdentityFailureBoundary::Trace,
                "launch expected_identity_sha256",
                launch.expected_identity_sha256(),
                actual_identity_sha256,
            ));
        }
        if !evidence_identities.insert(actual_identity_sha256) {
            return Err(manifest_rejected(
                manifest_path,
                format!("duplicate launch evidence identity {actual_identity_sha256}"),
            ));
        }

        let launch_arch = identity.arch().map_err(|source| {
            launch_evidence_error(&path, sim_sm::ManifestError::Architecture(source))
        })?;
        let launch_sku = identity.sku().map_err(|source| {
            launch_evidence_error(&path, sim_sm::ManifestError::Architecture(source))
        })?;
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
        if identity.arch_card_sha256() != manifest.configuration_evidence().arch_card_sha256() {
            return Err(identity_mismatch(
                IdentityFailureBoundary::ArchCard,
                "launch architecture-card SHA-256",
                manifest.configuration_evidence().arch_card_sha256(),
                identity.arch_card_sha256(),
            ));
        }
        if identity.pchase_card_sha256() != manifest.configuration_evidence().pchase_card_sha256() {
            return Err(identity_mismatch(
                IdentityFailureBoundary::Configuration,
                "launch P-chase-card SHA-256",
                manifest.configuration_evidence().pchase_card_sha256(),
                identity.pchase_card_sha256(),
            ));
        }

        compare_authenticated_launch_key(
            launch.launch_key(),
            identity.replay_authenticated_launch(),
        )?;

        let trace_identity = identity.trace_bundle_identity();
        if let Some(expected) = &shared_trace_identity {
            if expected != trace_identity {
                return Err(identity_mismatch(
                    IdentityFailureBoundary::Trace,
                    "authenticated trace-bundle identity",
                    format!("{expected:?}"),
                    format!("{trace_identity:?}"),
                ));
            }
        } else {
            shared_trace_identity = Some(trace_identity.clone());
        }

        let binding = identity.launch_binding().map_err(|source| {
            RunError::trace_admission(
                format!(
                    "admitted launch identity {actual_identity_sha256} did not reconstruct: {source}"
                ),
                false,
            )
        })?;
        admitted.push(AdmittedEntryV2 {
            evidence_path: path,
            launch_key: launch.launch_key().clone(),
            binding,
            evidence,
        });
    }

    admitted.sort_by(|left, right| left.binding.cmp(&right.binding));
    let trace_identity = shared_trace_identity.ok_or_else(|| {
        manifest_rejected(
            manifest_path,
            "launches must authenticate exactly one nonempty trace bundle",
        )
    })?;
    Ok((admitted, trace_identity))
}

fn compare_authenticated_launch_key(
    launch_key: &LaunchKeyV1,
    authenticated: &sim_sm::ReplayAuthenticatedLaunchIdentity,
) -> Result<(), RunError> {
    if launch_key.kernel_mangled_name() != authenticated.kernel_mangled_name() {
        return Err(identity_mismatch(
            IdentityFailureBoundary::Trace,
            "launch key kernel_mangled_name",
            launch_key.kernel_mangled_name(),
            authenticated.kernel_mangled_name(),
        ));
    }
    if launch_key.grid() != authenticated.grid() {
        return Err(identity_mismatch(
            IdentityFailureBoundary::Trace,
            "launch key grid",
            format!("{:?}", launch_key.grid()),
            format!("{:?}", authenticated.grid()),
        ));
    }
    if launch_key.block() != authenticated.block() {
        return Err(identity_mismatch(
            IdentityFailureBoundary::Trace,
            "launch key block",
            format!("{:?}", launch_key.block()),
            format!("{:?}", authenticated.block()),
        ));
    }
    if launch_key.device_id() != authenticated.device_id() {
        return Err(identity_mismatch(
            IdentityFailureBoundary::Trace,
            "launch key device_id",
            launch_key.device_id(),
            authenticated.device_id(),
        ));
    }
    if launch_key.stream_id() != authenticated.stream_id() {
        return Err(identity_mismatch(
            IdentityFailureBoundary::Trace,
            "launch key stream_id",
            launch_key.stream_id(),
            authenticated.stream_id(),
        ));
    }
    if launch_key.binary_architecture() != authenticated.binary_architecture() {
        return Err(identity_mismatch(
            IdentityFailureBoundary::Trace,
            "launch key binary_architecture",
            launch_key.binary_architecture(),
            authenticated.binary_architecture(),
        ));
    }
    Ok(())
}

const fn manifest_replay_mode(mode: ReplayModeV2) -> ManifestReplayMode {
    match mode {
        ReplayModeV2::Flat => ManifestReplayMode::Flat,
        ReplayModeV2::L1Const => ManifestReplayMode::L1Const,
        ReplayModeV2::L1SharedL2 { dram_bandwidth } => ManifestReplayMode::L1SharedL2 {
            dram_bandwidth: match dram_bandwidth {
                DramBandwidthV2::Carded => ManifestDramBandwidth::Carded,
                DramBandwidthV2::Unmodeled => ManifestDramBandwidth::Unmodeled,
            },
        },
    }
}

fn derive_arch_card_id(card_sha256: Sha256Digest) -> Result<String, RunError> {
    let digest = canonical_typed_sha256(&ArchCardIdPreimage {
        digest_contract_id: ARCH_CARD_ID_CONTRACT,
        card_sha256,
    })?;
    Ok(format!("{ARCH_CARD_ID_PREFIX}{digest}"))
}

fn derive_configuration_id(
    resolved_configuration_sha256: Sha256Digest,
    arch_card_sha256: Sha256Digest,
    pchase_card_sha256: Sha256Digest,
) -> Result<String, RunError> {
    let digest = canonical_typed_sha256(&ConfigurationIdPreimage {
        digest_contract_id: CONFIGURATION_ID_CONTRACT,
        resolved_configuration_sha256,
        arch_card_sha256,
        pchase_card_sha256,
    })?;
    Ok(format!("{CONFIGURATION_ID_PREFIX}{digest}"))
}

#[allow(clippy::too_many_arguments)]
fn seal_result_v2(
    build_identity: BuildIdentityV2,
    trace_identity: TraceBundleIdentity,
    arch_card_identity: ArchCardIdentityV2,
    configuration_identity: ConfigurationIdentityV2,
    resolved_configuration: crate::ResolvedConfigurationWire,
    launches: Vec<PerLaunchResultV2>,
    aggregate: AggregateV2,
    diagnostics: DiagnosticsV2,
) -> Result<SimRunOutput, RunError> {
    let placeholder = Sha256Digest::of(b"sim-run.result-v2.unsealed-placeholder");
    let provisional = SimResultV2::checked(
        placeholder,
        build_identity.clone(),
        trace_identity.clone(),
        arch_card_identity.clone(),
        configuration_identity.clone(),
        resolved_configuration.clone(),
        launches.clone(),
        aggregate.clone(),
        diagnostics.clone(),
    )
    .map_err(wire_serialization_error)?;
    let unsigned_value = result_value_without_seal(&provisional)?;
    let unsigned_canonical =
        canonical_json_utf8(&unsigned_value).map_err(result_serialization_error)?;
    let result_sha256 = Sha256Digest::of(unsigned_canonical.as_bytes());

    // Invoke the checked constructor again with the actual seal so the object
    // being emitted, not only the placeholder preimage, satisfies launch algebra.
    let result = SimResultV2::checked(
        result_sha256,
        build_identity,
        trace_identity,
        arch_card_identity,
        configuration_identity,
        resolved_configuration,
        launches,
        aggregate,
        diagnostics,
    )
    .map_err(wire_serialization_error)?;
    let sealed_value = serde_json::to_value(&result).map_err(result_serialization_error)?;
    // `SimResultV2` is a closed, deterministic wire type. One canonical
    // serialization is enough; reparsing and rehashing the same document only
    // duplicates the hot result-emission path.
    let canonical_json = canonical_json_utf8(&sealed_value).map_err(result_serialization_error)?;
    Ok(SimRunOutput {
        canonical_json,
        result_sha256,
    })
}

fn result_value_without_seal(result: &SimResultV2) -> Result<serde_json::Value, RunError> {
    let value = serde_json::to_value(result).map_err(result_serialization_error)?;
    value_without_member(value, "result_sha256")
}

fn value_without_member(
    mut value: serde_json::Value,
    member: &'static str,
) -> Result<serde_json::Value, RunError> {
    let object = value
        .as_object_mut()
        .ok_or_else(|| result_serialization_error("serialized document root is not an object"))?;
    let removed = object
        .remove(member)
        .ok_or_else(|| result_serialization_error(format!("missing seal member {member}")))?;
    if !removed.is_string() {
        return Err(result_serialization_error(format!(
            "seal member {member} is not a string"
        )));
    }
    Ok(value)
}

fn canonical_typed_utf8<T: Serialize>(value: &T) -> Result<String, RunError> {
    let value = serde_json::to_value(value).map_err(result_serialization_error)?;
    canonical_json_utf8(&value).map_err(result_serialization_error)
}

fn canonical_typed_sha256<T: Serialize>(value: &T) -> Result<Sha256Digest, RunError> {
    let canonical = canonical_typed_utf8(value)?;
    Ok(Sha256Digest::of(canonical.as_bytes()))
}

fn wire_serialization_error(error: WireError) -> RunError {
    result_serialization_error(error)
}

fn wire_emission_error(error: WireError) -> RunError {
    RunError::emission(error.to_string())
}

fn build_provenance_error(reason: impl std::fmt::Display) -> RunError {
    result_serialization_error(format!(
        "build provenance could not be established: {reason}"
    ))
}

fn hash_current_process_image() -> Result<Sha256Digest, RunError> {
    #[cfg(target_os = "linux")]
    let path = PathBuf::from("/proc/self/exe");
    #[cfg(not(target_os = "linux"))]
    let path = std::env::current_exe().map_err(|source| {
        build_provenance_error(format!(
            "could not resolve the current process image: {source}"
        ))
    })?;

    let mut file = File::open(&path).map_err(|source| {
        build_provenance_error(format!(
            "could not open current process image {}: {source}",
            path.display()
        ))
    })?;
    let before = file.metadata().map_err(|source| {
        build_provenance_error(format!(
            "could not inspect current process image {}: {source}",
            path.display()
        ))
    })?;
    if !before.is_file() {
        return Err(build_provenance_error(format!(
            "current process image {} is not regular",
            path.display()
        )));
    }
    let before_observation = ProcessImageObservation::from_metadata(&before);
    let expected_size = before_observation.size;
    if expected_size == 0 || expected_size > MAX_PROCESS_IMAGE_BYTES {
        return Err(build_provenance_error(format!(
            "current process image size {expected_size} is outside 1..={MAX_PROCESS_IMAGE_BYTES} bytes"
        )));
    }
    let read_limit = expected_size
        .checked_add(1)
        .ok_or_else(|| build_provenance_error("current process image read bound overflowed"))?;
    let capacity = usize::try_from(read_limit).map_err(|source| {
        build_provenance_error(format!(
            "current process image read bound does not fit memory: {source}"
        ))
    })?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(capacity).map_err(|source| {
        build_provenance_error(format!(
            "could not reserve bounded process-image buffer: {source}"
        ))
    })?;
    (&mut file)
        .take(read_limit)
        .read_to_end(&mut bytes)
        .map_err(|source| {
            build_provenance_error(format!(
                "could not read current process image {}: {source}",
                path.display()
            ))
        })?;
    let actual_size = u64::try_from(bytes.len()).map_err(|source| {
        build_provenance_error(format!(
            "process-image byte count does not fit u64: {source}"
        ))
    })?;
    let after = file.metadata().map_err(|source| {
        build_provenance_error(format!(
            "could not re-inspect current process image {}: {source}",
            path.display()
        ))
    })?;
    let after_observation = ProcessImageObservation::from_metadata(&after);
    if actual_size != expected_size || after_observation != before_observation {
        return Err(build_provenance_error(format!(
            "current process image changed during bounded read: expected {expected_size} bytes, read {actual_size}, final size {}",
            after_observation.size
        )));
    }
    Ok(Sha256Digest::of(&bytes))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ProcessImageObservation {
    device: u64,
    inode: u64,
    size: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

impl ProcessImageObservation {
    fn from_metadata(metadata: &std::fs::Metadata) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            size: metadata.len(),
            modified_seconds: metadata.mtime(),
            modified_nanoseconds: metadata.mtime_nsec(),
            changed_seconds: metadata.ctime(),
            changed_nanoseconds: metadata.ctime_nsec(),
        }
    }
}
