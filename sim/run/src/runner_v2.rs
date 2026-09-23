//! Private strict-v2 production runner behind the public `run_manifest` entry point.

use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use serde::Serialize;
use sim_arch::{ArchId, Sku};
#[cfg(test)]
use sim_sm::admit_launch_evidence;
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::fs;
    use std::io::{Seek, SeekFrom};
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::PermissionsExt;

    use prost::Message;
    use serde_json::{Value, json};
    use sim_trace::TraceDirSnapshot;
    use tempfile::TempDir;

    use super::*;

    const CANARY_KERNEL_NAME: &str = "_Z17addr_carry_canaryPKiPiS1_S1_i___0";
    const ARCH_CARD_SHA: &str = "12c72f8b16ac369fe8ab877f39850bfc1e5cbb3b16d0b2d45a43b14c66ffe1da";
    const PCHASE_CARD_SHA: &str =
        "e87d2e208f5326a8083d1a54799864953a62ed1fece1b628583cf8d5cafc581d";
    const EVIDENCE_SHA: &str = "39ef24ba06dfba4025c8912813fff8155dd22a5a05c8676b9c7041eb8ee2937d";
    const METRIC_KEYS: [&str; 8] = [
        "smsp_inst_executed_sum",
        "traffic_gmem_ld_sectors",
        "traffic_gmem_st_sectors",
        "traffic_dram_read_sectors",
        "cache_hit_rate_l1tex_read_sector",
        "cache_hit_rate_l1tex_sector",
        "cache_hit_rate_l2_sector",
        "exec_cycles_gpc_max",
    ];

    fn fixture_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/addr_carry_canary_bundle")
    }

    fn copy_tree(source: &Path, destination: &Path) {
        fs::create_dir_all(destination).unwrap();
        for entry in fs::read_dir(source).unwrap() {
            let entry = entry.unwrap();
            let source_path = entry.path();
            let destination_path = destination.join(entry.file_name());
            let file_type = entry.file_type().unwrap();
            if file_type.is_dir() {
                copy_tree(&source_path, &destination_path);
            } else {
                assert!(file_type.is_file());
                fs::copy(source_path, destination_path).unwrap();
            }
        }
    }

    fn read_json(path: &Path) -> Value {
        serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
    }

    fn write_json(path: &Path, value: &Value) {
        fs::write(path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
    }

    fn launch_key(stream_id: u64) -> Value {
        json!({
            "workload_id": "fixture.addr-carry-é",
            "matched_execution_id": "fixture-execution-α",
            "process_launch_ordinal": stream_id,
            "kernel_mangled_name": CANARY_KERNEL_NAME,
            "grid": {"x": 1, "y": 1, "z": 1},
            "block": {"x": 32, "y": 1, "z": 1},
            "device_id": 0,
            "stream_id": stream_id,
            "binary_architecture": "sm_86"
        })
    }

    fn copied_v2_fixture(replay_mode: Value) -> (TempDir, PathBuf) {
        let temp = TempDir::new().unwrap();
        let root = temp.path().join("bundle");
        copy_tree(&fixture_root(), &root);
        let manifest_path = root.join("run_manifest.v2.json");
        let mut manifest = read_json(&root.join("run_manifest.json"));
        manifest["schema_version"] = json!(2);
        manifest["scope"] = json!("fixture/addr-carry-canary:v2");
        manifest["launches"][0]["launch_key"] = launch_key(0);
        manifest["replay_mode"] = replay_mode;
        write_json(&manifest_path, &manifest);
        (temp, manifest_path)
    }

    fn fixed_build_context() -> BuildContextV2 {
        BuildContextV2 {
            sim_git_commit: "0123456789abcdef0123456789abcdef01234567".to_owned(),
            sim_git_dirty: false,
            binary_sha256: Sha256Digest::of(b"fixed-private-v2-test-binary"),
            rustc_version: "rustc fixed-private-test".to_owned(),
            cargo_profile: "fixed-private-test".to_owned(),
        }
    }

    fn run_fixed(path: &Path) -> (SimRunOutput, Value) {
        let output = run_manifest_v2_with_context(path, &fixed_build_context()).unwrap();
        let value = serde_json::from_str(output.canonical_json()).unwrap();
        (output, value)
    }

    fn assert_exact_metric_map(metrics: &Value) {
        let object = metrics.as_object().unwrap();
        assert_eq!(object.len(), METRIC_KEYS.len());
        let actual = object.keys().map(String::as_str).collect::<BTreeSet<_>>();
        assert_eq!(actual, METRIC_KEYS.into_iter().collect());
        assert_eq!(metrics["exec_cycles_gpc_max"]["status"], "predicted");
        assert_eq!(
            metrics["exec_cycles_gpc_max"]["value"]["kind"],
            "binding_cycles"
        );
        assert!(metrics.get("diagnostic_last_issue_cycles").is_none());
        assert!(metrics.get("terminal_modeled_event_time").is_none());
    }

    fn expected_direct_diagnostics(manifest_path: &Path) -> (u64, Option<u64>) {
        let evidence =
            admit_launch_evidence(manifest_path.parent().unwrap().join("launch_evidence.json"))
                .unwrap();
        let raw = fs::read(manifest_path).unwrap();
        let manifest = parse_sim_run_manifest_v2(&raw).unwrap();
        let resolved = resolve_configuration(
            &evidence,
            ArchId::Sm86,
            manifest_replay_mode(manifest.replay_mode()),
        )
        .unwrap();
        match replay_kernel(&evidence, &resolved.issue_params, resolved.replay_mode).unwrap() {
            KernelReplayOutcome::Full(result) => (
                result.diagnostic_last_issue_cycles(),
                result.terminal_modeled_event_time(),
            ),
            KernelReplayOutcome::SampledObserved(_) => panic!("fixture must be full coverage"),
        }
    }

    #[test]
    fn private_v2_success_is_complete_for_every_replay_mode() {
        let cases = [
            (json!({"mode": "flat"}), "unmodeled", "unmodeled"),
            (json!({"mode": "l1_const"}), "modeled", "unmodeled"),
            (
                json!({"mode": "l1_shared_l2", "dram_bandwidth": "unmodeled"}),
                "modeled",
                "modeled",
            ),
            (
                json!({"mode": "l1_shared_l2", "dram_bandwidth": "carded"}),
                "modeled",
                "modeled",
            ),
        ];

        for (mode, expected_l1, expected_l2) in cases {
            let (_temp, manifest_path) = copied_v2_fixture(mode);
            let evidence =
                admit_launch_evidence(manifest_path.parent().unwrap().join("launch_evidence.json"))
                    .unwrap();
            let expected_trace =
                serde_json::to_value(evidence.identity().trace_bundle_identity()).unwrap();
            let manifest_raw = fs::read(&manifest_path).unwrap();
            let typed_manifest = parse_sim_run_manifest_v2(&manifest_raw).unwrap();
            let expected_manifest_sha = canonical_typed_sha256(&typed_manifest).unwrap();
            let expected_diagnostics = expected_direct_diagnostics(&manifest_path);

            let (output, result) = run_fixed(&manifest_path);
            assert_eq!(result["schema_version"], "sim_result.v2");
            assert_eq!(
                result["exit_status"],
                json!({"status": "success", "exit_code": 0})
            );
            assert_eq!(
                result["build_identity"]["sim_git_commit"],
                fixed_build_context().sim_git_commit
            );
            assert_eq!(result["build_identity"]["sim_git_dirty"], false);
            assert_eq!(
                result["build_identity"]["binary_sha256"],
                fixed_build_context().binary_sha256.to_string()
            );
            assert_eq!(
                result["build_identity"]["claimability"],
                json!({"status": "claimable", "basis": "clean_build"})
            );
            assert_eq!(result["trace_identity"], expected_trace);
            assert!(result["trace_identity"]["record_count"].as_u64().unwrap() > 0);
            assert!(
                result["trace_identity"]["instruction_count"]
                    .as_u64()
                    .unwrap()
                    > 0
            );

            assert_eq!(result["arch_card_identity"]["card_sha256"], ARCH_CARD_SHA);
            assert_eq!(result["arch_card_identity"]["sm_arch"], "sm_86");
            assert_eq!(result["arch_card_identity"]["sku"], "rtx_3090_ti");
            assert_eq!(
                result["arch_card_identity"]["card_id"],
                derive_arch_card_id(Sha256Digest::parse(ARCH_CARD_SHA).unwrap()).unwrap()
            );
            assert_eq!(
                result["configuration_identity"]["run_manifest_sha256"],
                expected_manifest_sha.to_string()
            );
            assert_eq!(
                result["configuration_identity"]["arch_card_sha256"],
                ARCH_CARD_SHA
            );
            assert_eq!(
                result["configuration_identity"]["pchase_card_sha256"],
                PCHASE_CARD_SHA
            );
            let resolved_sha = canonical_typed_sha256(&result["resolved_configuration"]).unwrap();
            assert_eq!(
                result["configuration_identity"]["resolved_configuration_sha256"],
                resolved_sha.to_string()
            );
            assert_eq!(
                result["configuration_identity"]["configuration_id"],
                derive_configuration_id(
                    resolved_sha,
                    Sha256Digest::parse(ARCH_CARD_SHA).unwrap(),
                    Sha256Digest::parse(PCHASE_CARD_SHA).unwrap(),
                )
                .unwrap()
            );
            assert_eq!(
                result["resolved_configuration"]["l1"]["status"],
                expected_l1
            );
            assert_eq!(
                result["resolved_configuration"]["l2"]["status"],
                expected_l2
            );

            let launches = result["launches"].as_array().unwrap();
            assert_eq!(launches.len(), 1);
            assert_eq!(launches[0]["launch_key"], launch_key(0));
            assert_eq!(result["aggregate"]["launch_count"], 1);
            assert_eq!(result["aggregate"]["scope"], "all_launches");
            assert_exact_metric_map(&launches[0]["metrics"]);
            assert_exact_metric_map(&result["aggregate"]["metrics"]);
            assert_eq!(launches[0]["metrics"], result["aggregate"]["metrics"]);
            assert_eq!(
                launches[0]["metrics"]["smsp_inst_executed_sum"]["value"]["value"],
                54
            );

            let diagnostics = &result["diagnostics"];
            assert_eq!(diagnostics["admissibility"], "non_admissible");
            assert!(diagnostics.get("internal_phase_timings_ns").is_none());
            assert_eq!(diagnostics["per_launch"].as_array().unwrap().len(), 1);
            assert_eq!(
                diagnostics["per_launch"][0]["launch_key"],
                launches[0]["launch_key"]
            );
            assert_eq!(
                diagnostics["per_launch"][0]["diagnostic_last_issue_cycles"],
                expected_diagnostics.0
            );
            match expected_diagnostics.1 {
                Some(value) => assert_eq!(
                    diagnostics["per_launch"][0]["terminal_modeled_event_time"],
                    json!({"status": "modeled", "value": value})
                ),
                None => assert_eq!(
                    diagnostics["per_launch"][0]["terminal_modeled_event_time"],
                    json!({
                        "status": "unavailable",
                        "reason": "no_modeled_issue_or_system_event"
                    })
                ),
            }

            let mut unsigned = result.clone();
            let seal = unsigned
                .as_object_mut()
                .unwrap()
                .remove("result_sha256")
                .unwrap();
            let unsigned_canonical = canonical_json_utf8(&unsigned).unwrap();
            assert_eq!(
                seal,
                Sha256Digest::of(unsigned_canonical.as_bytes()).to_string()
            );
            assert_eq!(seal, output.result_sha256().to_string());
            assert_eq!(
                canonical_json_utf8(&result).unwrap(),
                output.canonical_json()
            );
            assert!(output.canonical_json().contains("é"));
            assert!(!output.canonical_json().contains("\\u00e9"));
        }
    }

    #[test]
    fn private_v2_fixed_context_is_byte_deterministic() {
        let (_temp, manifest_path) = copied_v2_fixture(json!({"mode": "flat"}));
        let first = run_manifest_v2_with_context(&manifest_path, &fixed_build_context()).unwrap();
        let second = run_manifest_v2_with_context(&manifest_path, &fixed_build_context()).unwrap();
        assert_eq!(first.result_sha256(), second.result_sha256());
        assert_eq!(first.canonical_json(), second.canonical_json());
    }

    #[test]
    fn retained_transport_reads_positionally_and_preserves_shared_offset() {
        let (_temp, manifest_path) = copied_v2_fixture(json!({"mode": "flat"}));
        let mut manifest_file = File::open(&manifest_path).unwrap();
        let end = manifest_file.seek(SeekFrom::End(0)).unwrap();
        let base_file = File::open(manifest_path.parent().unwrap()).unwrap();
        let transport = RunManifestTransport::from_retained(
            u32::try_from(manifest_file.as_raw_fd()).unwrap(),
            u32::try_from(base_file.as_raw_fd()).unwrap(),
            manifest_path.file_name().unwrap(),
        )
        .unwrap();
        assert!(!transport.raw().is_empty());
        assert_eq!(manifest_file.stream_position().unwrap(), end);
        transport.recheck().unwrap();
        assert_eq!(manifest_file.stream_position().unwrap(), end);
    }

    #[test]
    fn retained_transport_needs_no_named_manifest_read_permission() {
        let (_temp, manifest_path) = copied_v2_fixture(json!({"mode": "flat"}));
        let manifest_file = File::open(&manifest_path).unwrap();
        let base_file = File::open(manifest_path.parent().unwrap()).unwrap();
        fs::set_permissions(&manifest_path, fs::Permissions::from_mode(0o000)).unwrap();

        let result = (|| {
            let transport = RunManifestTransport::from_retained(
                u32::try_from(manifest_file.as_raw_fd()).unwrap(),
                u32::try_from(base_file.as_raw_fd()).unwrap(),
                manifest_path.file_name().unwrap(),
            )?;
            transport.recheck()
        })();
        fs::set_permissions(&manifest_path, fs::Permissions::from_mode(0o600)).unwrap();

        result.unwrap();
    }

    #[test]
    fn retained_transport_final_recheck_binds_exact_bytes_and_named_inode() {
        let (_temp, manifest_path) = copied_v2_fixture(json!({"mode": "flat"}));
        let manifest_file = File::open(&manifest_path).unwrap();
        let base_file = File::open(manifest_path.parent().unwrap()).unwrap();
        let transport = RunManifestTransport::from_retained(
            u32::try_from(manifest_file.as_raw_fd()).unwrap(),
            u32::try_from(base_file.as_raw_fd()).unwrap(),
            manifest_path.file_name().unwrap(),
        )
        .unwrap();
        fs::write(&manifest_path, b"{}").unwrap();
        let error = transport.recheck().unwrap_err();
        assert!(error.to_string().contains("exact-byte mismatch"));

        let (_temp, manifest_path) = copied_v2_fixture(json!({"mode": "flat"}));
        let manifest_file = File::open(&manifest_path).unwrap();
        let base_file = File::open(manifest_path.parent().unwrap()).unwrap();
        let transport = RunManifestTransport::from_retained(
            u32::try_from(manifest_file.as_raw_fd()).unwrap(),
            u32::try_from(base_file.as_raw_fd()).unwrap(),
            manifest_path.file_name().unwrap(),
        )
        .unwrap();
        let original = manifest_path
            .parent()
            .unwrap()
            .join("original-retained.json");
        fs::rename(&manifest_path, &original).unwrap();
        fs::copy(&original, &manifest_path).unwrap();
        let error = transport.recheck().unwrap_err();
        assert!(error.to_string().contains("device/inode identity mismatch"));
    }

    fn retained_evidence_cases() -> [(&'static str, &'static str, &'static str); 9] {
        [
            (
                "launch_evidence.json",
                "launch_evidence_manifest",
                "TRACE_ADMISSION_FAILED",
            ),
            (
                "trace/dynamic_trace.pb",
                "dynamic_trace_pb",
                "TRACE_ADMISSION_FAILED",
            ),
            (
                "trace/threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_0,0,0.pb",
                "threadblock_payload",
                "TRACE_ADMISSION_FAILED",
            ),
            (
                "trace/extra_info/allocations.json",
                "allocations_json",
                "TRACE_ADMISSION_FAILED",
            ),
            ("trace/stats.csv", "stats_csv", "TRACE_ADMISSION_FAILED"),
            (
                "trace/extra_info/enhanced_execution_info.json",
                "enhanced_execution_info_json",
                "TRACE_ADMISSION_FAILED",
            ),
            (
                "control/canary_sm86.control_bits.json",
                "control_bits_sidecar",
                "TRACE_ADMISSION_FAILED",
            ),
            (
                "cards/sm_86.json",
                "arch_card",
                "ARCH_CARD_ADMISSION_FAILED",
            ),
            (
                "cards/sm_86_static_card.json",
                "pchase_card",
                "CONFIGURATION_RESOLUTION_FAILED",
            ),
        ]
    }

    #[test]
    fn persistent_evidence_substitution_before_publication_fails_typed() {
        for (index, (relative, token, failure_type)) in
            retained_evidence_cases().into_iter().enumerate()
        {
            let (_temp, manifest_path) = copied_v2_fixture(json!({"mode": "flat"}));
            let artifact_path = manifest_path.parent().unwrap().join(relative);
            let displaced = manifest_path
                .parent()
                .unwrap()
                .join(format!("displaced-generation-{index}"));
            let transport = RunManifestTransport::from_path(&manifest_path).unwrap();
            let error = run_manifest_v2_with_before_publication(
                &transport,
                &fixed_build_context(),
                move || {
                    fs::rename(&artifact_path, &displaced).unwrap();
                    fs::copy(&displaced, &artifact_path).unwrap();
                },
            )
            .unwrap_err();

            assert_eq!(error.failure_type(), failure_type, "artifact {relative}");
            assert!(
                error.to_string().contains("retained artifact generation"),
                "artifact {relative}: {error}"
            );
            assert!(
                error.to_string().contains(token),
                "artifact {relative}: {error}"
            );
        }
    }

    #[test]
    fn persistent_evidence_deletion_before_publication_fails_typed() {
        for (relative, token, failure_type) in retained_evidence_cases() {
            let (_temp, manifest_path) = copied_v2_fixture(json!({"mode": "flat"}));
            let artifact_path = manifest_path.parent().unwrap().join(relative);
            let transport = RunManifestTransport::from_path(&manifest_path).unwrap();
            let error = run_manifest_v2_with_before_publication(
                &transport,
                &fixed_build_context(),
                move || fs::remove_file(&artifact_path).unwrap(),
            )
            .unwrap_err();

            assert_eq!(error.failure_type(), failure_type, "artifact {relative}");
            assert!(
                error.to_string().contains("could not be rechecked"),
                "artifact {relative}: {error}"
            );
            assert!(
                error.to_string().contains(token),
                "artifact {relative}: {error}"
            );
        }
    }

    #[test]
    fn absent_optional_evidence_created_before_publication_fails_typed() {
        let (_temp, manifest_path) = copied_v2_fixture(json!({"mode": "flat"}));
        let sampling_path = manifest_path
            .parent()
            .unwrap()
            .join("trace/trace_sampling.json");
        let transport = RunManifestTransport::from_path(&manifest_path).unwrap();
        let error = run_manifest_v2_with_before_publication(
            &transport,
            &fixed_build_context(),
            move || fs::write(&sampling_path, b"{}").unwrap(),
        )
        .unwrap_err();

        assert_eq!(error.failure_type(), "TRACE_ADMISSION_FAILED");
        assert_eq!(error.stage(), "trace_admission");
        assert!(error.to_string().contains("trace_sampling_json"));
        assert!(error.to_string().contains("presence mismatch"));
    }

    #[test]
    fn new_threadblock_member_before_publication_fails_typed() {
        let (_temp, manifest_path) = copied_v2_fixture(json!({"mode": "flat"}));
        let root = manifest_path.parent().unwrap();
        let original =
            root.join("trace/threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_0,0,0.pb");
        let added = root.join("trace/threadblocks/device_0/stream_0/kernel_1/added.pb");
        let transport = RunManifestTransport::from_path(&manifest_path).unwrap();
        let error = run_manifest_v2_with_before_publication(
            &transport,
            &fixed_build_context(),
            move || {
                fs::copy(&original, &added).unwrap();
            },
        )
        .unwrap_err();

        assert_eq!(error.failure_type(), "TRACE_ADMISSION_FAILED");
        assert_eq!(error.stage(), "trace_admission");
        assert!(error.to_string().contains("threadblock_payload"));
    }

    #[test]
    fn cohort_assertions_are_carried_verbatim_but_do_not_name_configuration() {
        let (_temp, manifest_path) = copied_v2_fixture(json!({"mode": "flat"}));
        let (_, first) = run_fixed(&manifest_path);
        let mut manifest = read_json(&manifest_path);
        manifest["launches"][0]["launch_key"]["workload_id"] = json!("cohort.changed-é");
        manifest["launches"][0]["launch_key"]["matched_execution_id"] =
            json!("execution.changed-β");
        manifest["launches"][0]["launch_key"]["process_launch_ordinal"] = json!(99);
        write_json(&manifest_path, &manifest);
        let (_, second) = run_fixed(&manifest_path);

        assert_eq!(
            second["launches"][0]["launch_key"],
            manifest["launches"][0]["launch_key"]
        );
        assert_ne!(
            first["configuration_identity"]["run_manifest_sha256"],
            second["configuration_identity"]["run_manifest_sha256"]
        );
        assert_eq!(
            first["configuration_identity"]["configuration_id"],
            second["configuration_identity"]["configuration_id"]
        );
        assert_eq!(first["arch_card_identity"], second["arch_card_identity"]);
    }

    #[test]
    fn six_replay_visible_launch_fields_fail_closed_on_the_consumer_path() {
        let cases = [
            (
                "kernel_mangled_name",
                json!("_Z16substituted_namev"),
                Some("launch key kernel_mangled_name"),
            ),
            (
                "grid",
                json!({"x": 2, "y": 1, "z": 1}),
                Some("launch key grid"),
            ),
            (
                "block",
                json!({"x": 64, "y": 1, "z": 1}),
                Some("launch key block"),
            ),
            ("device_id", json!(1), Some("launch key device_id")),
            ("stream_id", json!(1), Some("launch key stream_id")),
            ("binary_architecture", json!("sm_120"), None),
        ];

        for (field, replacement, expected_edge) in cases {
            let (_temp, manifest_path) = copied_v2_fixture(json!({"mode": "flat"}));
            let mut manifest = read_json(&manifest_path);
            manifest["launches"][0]["launch_key"][field] = replacement;
            write_json(&manifest_path, &manifest);
            let error =
                run_manifest_v2_with_context(&manifest_path, &fixed_build_context()).unwrap_err();
            match expected_edge {
                Some(expected_edge) => {
                    assert_eq!(error.failure_type(), "TRACE_ADMISSION_FAILED");
                    assert_eq!(error.stage(), "trace_admission");
                    assert!(error.to_string().contains(expected_edge));
                }
                None => {
                    assert_eq!(error.failure_type(), "MANIFEST_ADMISSION_FAILED");
                    assert_eq!(error.stage(), "manifest_admission");
                }
            }
        }
    }

    #[test]
    fn evidence_and_configuration_hash_mismatches_are_individually_rejected() {
        let cases = [
            (
                "expected_identity_sha256",
                "launch expected_identity_sha256",
            ),
            ("arch_card_sha256", "launch architecture-card SHA-256"),
            ("pchase_card_sha256", "launch P-chase-card SHA-256"),
        ];
        for (field, expected_edge) in cases {
            let (_temp, manifest_path) = copied_v2_fixture(json!({"mode": "flat"}));
            let mut manifest = read_json(&manifest_path);
            if field == "expected_identity_sha256" {
                manifest["launches"][0][field] = json!("a".repeat(64));
            } else {
                manifest["configuration_evidence"][field] = json!("a".repeat(64));
            }
            write_json(&manifest_path, &manifest);
            let error =
                run_manifest_v2_with_context(&manifest_path, &fixed_build_context()).unwrap_err();
            let expected_failure = match field {
                "expected_identity_sha256" => "TRACE_ADMISSION_FAILED",
                "arch_card_sha256" => "ARCH_CARD_ADMISSION_FAILED",
                "pchase_card_sha256" => "CONFIGURATION_RESOLUTION_FAILED",
                _ => unreachable!("closed test case"),
            };
            assert_eq!(error.failure_type(), expected_failure);
            assert!(error.to_string().contains(expected_edge));
        }
    }

    #[test]
    fn duplicate_admitted_evidence_identity_is_rejected() {
        let (_temp, manifest_path) = copied_v2_fixture(json!({"mode": "flat"}));
        let root = manifest_path.parent().unwrap();
        fs::copy(
            root.join("launch_evidence.json"),
            root.join("launch_evidence_alias.json"),
        )
        .unwrap();
        let mut manifest = read_json(&manifest_path);
        manifest["launches"].as_array_mut().unwrap().push(json!({
            "launch_evidence_manifest_path": "launch_evidence_alias.json",
            "expected_identity_sha256": EVIDENCE_SHA,
            "launch_key": launch_key(1)
        }));
        write_json(&manifest_path, &manifest);
        let error =
            run_manifest_v2_with_context(&manifest_path, &fixed_build_context()).unwrap_err();
        assert_eq!(error.failure_type(), "MANIFEST_ADMISSION_FAILED");
        assert!(
            error
                .to_string()
                .contains("duplicate launch evidence identity")
        );
    }

    fn artifact_claim(root: &Path, relative: &str) -> Value {
        let bytes = fs::read(root.join(relative)).unwrap();
        json!({
            "path": relative,
            "size_bytes": bytes.len(),
            "sha256": Sha256Digest::of(&bytes)
        })
    }

    fn refresh_trace_claims(root: &Path, trace_prefix: &str, evidence: &mut Value) {
        let snapshot = TraceDirSnapshot::capture(root.join(trace_prefix)).unwrap();
        let payload = snapshot.artifact_identity().threadblock_payload();
        evidence["artifacts"]["dynamic_trace_pb"] =
            artifact_claim(root, &format!("{trace_prefix}/dynamic_trace.pb"));
        evidence["artifacts"]["threadblock_payload"] = json!({
            "path": format!("{trace_prefix}/threadblocks"),
            "file_count": payload.file_count(),
            "raw_bytes": payload.raw_bytes(),
            "sorted_path_manifest_sha256": payload.sorted_path_manifest_sha256(),
            "sorted_path_stream_sha256": payload.sorted_path_stream_sha256()
        });
        evidence["artifacts"]["stats_csv"] =
            artifact_claim(root, &format!("{trace_prefix}/stats.csv"));
        evidence["artifacts"]["allocations_json"]["artifact"] =
            artifact_claim(root, &format!("{trace_prefix}/extra_info/allocations.json"));
        evidence["artifacts"]["enhanced_execution_info_json"] = artifact_claim(
            root,
            &format!("{trace_prefix}/extra_info/enhanced_execution_info.json"),
        );
    }

    #[test]
    fn mixed_authenticated_trace_bundle_identity_is_rejected_without_fake_evidence() {
        let (_temp, manifest_path) = copied_v2_fixture(json!({"mode": "flat"}));
        let root = manifest_path.parent().unwrap();
        let dynamic_path = root.join("trace/dynamic_trace.pb");
        let mut trace =
            sim_trace::pb::Trace::decode(fs::read(&dynamic_path).unwrap().as_slice()).unwrap();
        let device = trace.gpu_device.get_mut(&0).unwrap();
        let mut stream = device.streams.get(&0).unwrap().clone();
        stream.id = 1;
        device.streams.insert(1, stream);
        fs::write(&dynamic_path, trace.encode_to_vec()).unwrap();

        let first_payload =
            root.join("trace/threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_0,0,0.pb");
        let second_payload =
            root.join("trace/threadblocks/device_0/stream_1/kernel_1/d_0_s_1_k_1_0,0,0.pb");
        fs::create_dir_all(second_payload.parent().unwrap()).unwrap();
        fs::copy(first_payload, second_payload).unwrap();
        fs::write(
            root.join("trace/stats.csv"),
            b"total_insts,total_reported_insts\n108,108\n",
        )
        .unwrap();

        let first_evidence_path = root.join("launch_evidence.json");
        let mut first_evidence = read_json(&first_evidence_path);
        refresh_trace_claims(root, "trace", &mut first_evidence);
        write_json(&first_evidence_path, &first_evidence);
        copy_tree(&root.join("trace"), &root.join("trace-copy"));

        let second_evidence_path = root.join("launch_evidence_stream_1.json");
        let mut second_evidence = first_evidence.clone();
        second_evidence["launch_selector"]["stream_id"] = json!(1);
        refresh_trace_claims(root, "trace-copy", &mut second_evidence);
        write_json(&second_evidence_path, &second_evidence);

        let first = admit_launch_evidence(&first_evidence_path).unwrap();
        let second = admit_launch_evidence(&second_evidence_path).unwrap();
        assert_eq!(
            first.identity().trace_bundle_identity().bundle_sha256(),
            second.identity().trace_bundle_identity().bundle_sha256()
        );
        assert_ne!(
            first.identity().trace_bundle_identity(),
            second.identity().trace_bundle_identity()
        );

        let mut manifest = read_json(&manifest_path);
        manifest["launches"] = json!([
            {
                "launch_evidence_manifest_path": "launch_evidence.json",
                "expected_identity_sha256": first.identity().seal_sha256(),
                "launch_key": launch_key(0)
            },
            {
                "launch_evidence_manifest_path": "launch_evidence_stream_1.json",
                "expected_identity_sha256": second.identity().seal_sha256(),
                "launch_key": launch_key(1)
            }
        ]);
        write_json(&manifest_path, &manifest);
        let error =
            run_manifest_v2_with_context(&manifest_path, &fixed_build_context()).unwrap_err();
        assert_eq!(error.failure_type(), "TRACE_ADMISSION_FAILED");
        assert!(
            error
                .to_string()
                .contains("authenticated trace-bundle identity")
        );
    }

    #[test]
    fn runtime_build_context_hashes_the_actual_current_process_image() {
        let (_temp, manifest_path) = copied_v2_fixture(json!({"mode": "flat"}));
        let output = run_manifest_v2(&manifest_path).unwrap();
        let result: Value = serde_json::from_str(output.canonical_json()).unwrap();
        let expected_binary_sha = hash_current_process_image().unwrap();
        assert_eq!(
            result["build_identity"]["binary_sha256"],
            expected_binary_sha.to_string()
        );
        assert_eq!(
            result["build_identity"]["sim_git_commit"],
            build_provenance::SIM_GIT_COMMIT
        );
        assert_eq!(
            result["build_identity"]["sim_git_dirty"],
            build_provenance::SIM_GIT_DIRTY
        );
        assert_eq!(
            result["build_identity"]["rustc_version"],
            build_provenance::RUSTC_VERSION
        );
        assert_eq!(
            result["build_identity"]["cargo_profile"],
            build_provenance::CARGO_PROFILE
        );
        assert_eq!(
            result["build_identity"]["claimability"]["status"],
            if build_provenance::SIM_GIT_DIRTY {
                "unclaimable"
            } else {
                "claimable"
            }
        );
    }

    #[test]
    fn private_v1_compatibility_composer_preserves_byte_golden() {
        let golden = include_str!("../tests/fixtures/addr_carry_canary_bundle/result.golden.json");

        let output = crate::v1_result_compat::run_manifest_v1_compat(
            fixture_root().join("run_manifest.json"),
        )
        .unwrap();
        let parsed: Value = serde_json::from_str(output.canonical_json()).unwrap();
        assert_eq!(parsed["schema_version"], "sim_result.v1");
        assert_eq!(parsed["emissions"][7]["outcome"]["status"], "predicted");
        assert_ne!(output.canonical_json(), golden);
    }
}
