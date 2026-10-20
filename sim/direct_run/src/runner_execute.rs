//! Replay execution and launch accounting; drives the active simulator but does not alter cache policy.

use super::*;

impl MemoryMode {
    pub const fn token(self) -> &'static str {
        match self {
            Self::Flat => "flat",
            Self::L1 => "l1",
            Self::L1L2 => "l1-l2",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheScope {
    /// Existing behavior: every launch receives a fresh cache hierarchy.
    Launch,
    /// Ordered same-device/same-stream launches retain only fully drained L2
    /// replacement and valid-sector state.
    StreamL2,
}

/// Explicit effective SM120 L1 service envelope supplied by a real-device
/// probe. This is an effective fully associative LRU model, not a physical
/// geometry claim. SM120 full replay uses this model by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Sm120EffectiveL1Config {
    pub line_capacity: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mshr_count: Option<u32>,
    pub hit_latency_cycles: u64,
    pub miss_latency_cycles: u64,
}

impl Sm120EffectiveL1Config {
    pub const fn as_l1_config(self) -> L1Config {
        L1Config::effective_sm120(
            self.line_capacity,
            self.mshr_count,
            self.hit_latency_cycles,
            self.miss_latency_cycles,
        )
    }
}

impl FromStr for CacheScope {
    type Err = RunError;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "launch" => Ok(Self::Launch),
            "stream-l2" => Ok(Self::StreamL2),
            _ => Err(RunError::new(
                "arguments",
                format!("unknown cache scope {value:?}; use launch or stream-l2"),
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Sm120EffectiveL2Config {
    pub capacity_bytes: u64,
    /// Explicit scenario geometry; these are assumptions, not recovered card
    /// metadata, and are serialized so a report is self-describing.
    pub line_bytes: u32,
    pub sector_bytes: u32,
    pub num_slices: u32,
    pub associativity: u32,
    pub replacement: L2Replacement,
    pub slice_hash: SliceHash,
    pub write_policy: &'static str,
}

impl Default for Sm120EffectiveL2Config {
    fn default() -> Self {
        Self {
            capacity_bytes: 100_663_296,
            line_bytes: 128,
            sector_bytes: 32,
            num_slices: 1,
            associativity: 16,
            replacement: L2Replacement::Lru,
            slice_hash: SliceHash::LineMod,
            write_policy: L2_WRITE_ALLOCATE_POLICY,
        }
    }
}

impl RunConfig {
    /// Resolve the integrated SM120 model once, before dispatch and reporting.
    /// Flat memory is only the separate core-only representation; it must
    /// never silently stand in for full SM120 replay.
    pub(crate) fn with_architecture_defaults(
        &self,
        arch: sim_arch::ArchId,
    ) -> Result<Self, RunError> {
        let mut config = self.clone();
        if !config.fast_frontier_only {
            // Full replay projections are part of the unified baseline for both
            // architecture lanes. The core-only diagnostic remains trace-only.
            config.ncu_shaped_metrics = true;
        }
        if arch == sim_arch::ArchId::Sm120 {
            config.sm120_alpha = true;
            if !config.fast_frontier_only {
                if config.memory_mode != MemoryMode::L1L2 {
                    return Err(RunError::new(
                        "arguments",
                        "SM120 full replay requires --mode l1-l2; flat and l1-only full replay are forbidden (flat is core-only diagnostic)",
                    ));
                }
                config
                    .sm120_effective_l1
                    .get_or_insert_with(Default::default);
                config
                    .sm120_effective_l2
                    .get_or_insert_with(Default::default);
                config.ncu_shaped_metrics = true;
                // The latent envelope is observer-only. Carded DRAM remains
                // explicit because the SM120 card marks bandwidth as a
                // research gap and the runner must fail closed, not guess.
                config.latent_envelope = true;
            }
        }
        Ok(config)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct MemoryServiceCandidateDocument {
    pub(super) schema_version: u32,
    pub(super) candidates: Vec<MemoryServiceCandidate>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct MemoryAdmissionCandidateDocument {
    pub(super) schema_version: u32,
    pub(super) candidates: Vec<MemoryAdmissionCandidate>,
}

pub(super) const MAX_PREDICTIVE_PROFILE_BYTES: u64 = 64 * 1024;

pub(super) const MAX_PAYLOAD_EVENTS: usize = 1_000_000;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PayloadEventDocument {
    pub(super) schema: String,
    pub(super) status: String,
    pub(super) events: Vec<PayloadEventRaw>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PayloadEventIdentityRaw {
    pub(super) launch_id: u64,
    pub(super) function_unique_id: i32,
    pub(super) cta: [u32; 3],
    pub(super) warp: u32,
    pub(super) lane: u32,
    pub(super) pc: u32,
    pub(super) occurrence: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PayloadEventSectorRaw {
    pub(super) allocation_id: u64,
    pub(super) generation: u64,
    pub(super) line_id: i64,
    pub(super) sector: u8,
    pub(super) byte_offset: u8,
    pub(super) bytes_hex: String,
    pub(super) known_mask: u32,
    pub(super) sequence: u64,
}

pub(crate) fn at<E: std::fmt::Display>(stage: &'static str) -> impl FnOnce(E) -> RunError {
    move |error| RunError::new(stage, error.to_string())
}

pub(super) const MAX_DIRECT_CARD_BYTES: u64 = 16 * 1024 * 1024;

pub(super) const MAX_DIRECT_CONTROL_BYTES: u64 = 64 * 1024 * 1024;

/// Read one direct-replay input from a no-follow anchored parent and retain the
/// exact generation that was parsed. Direct mode has no campaign manifest, so
/// this is the boundary that prevents a concurrent path replacement from
/// silently changing cards or control metadata during a run.
pub(super) fn read_direct_input(
    path: &Path,
    stage: &'static str,
    maximum_bytes: u64,
) -> Result<Vec<u8>, RunError> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .ok_or_else(|| RunError::new(stage, "input path has no file name"))?;
    let relative = Path::new(name);
    let reader = AnchoredDir::open(parent).map_err(at(stage))?;
    let file = reader
        .read_bounded_regular(relative, maximum_bytes)
        .map_err(at(stage))?;
    let expected = file.generation_identity();
    let raw = file.into_raw();
    let actual = reader
        .inspect_regular_generation(relative)
        .map_err(at(stage))?;
    if actual != expected {
        return Err(RunError::new(
            stage,
            format!("input {} changed after its snapshot", path.display()),
        ));
    }
    Ok(raw)
}

pub(crate) fn load_arch_card(path: &Path) -> Result<ArchCard, RunError> {
    let raw = read_direct_input(path, "architecture card", MAX_DIRECT_CARD_BYTES)?;
    ArchCard::parse(path, &raw).map_err(at("architecture card"))
}

pub(super) fn load_predictive_service_profile(
    path: Option<&PathBuf>,
) -> Result<Option<ServiceRateProfile>, RunError> {
    path.map(|path| {
        let bytes = read_direct_input(
            path,
            "predictive service profile",
            MAX_PREDICTIVE_PROFILE_BYTES,
        )?;
        let profile: ServiceRateProfile =
            strict_json_from_slice(&bytes).map_err(at("predictive service profile"))?;
        profile
            .validate()
            .map_err(at("predictive service profile"))?;
        Ok(profile)
    })
    .transpose()
}

/// Share the established direct-replay issue constructor between ordinary and
/// prepared replay. Direct replay must consume the same carded scheduler
/// resources as the campaign path; otherwise changing `smsp.issue_ports`,
/// occupancy, or dual-issue state in a user card silently has no effect.
/// The optional scalar/profile issue-width override is layered afterward and
/// is explicitly labeled hypothetical.
pub(crate) fn resolve_issue_with_metadata(
    config: &RunConfig,
    card: &ArchCard,
) -> Result<(IssueModelParams, Option<u32>), RunError> {
    let arch = card.arch_id().map_err(at("architecture card"))?;
    if config.sm120_alpha {
        if arch != sim_arch::ArchId::Sm120 {
            return Err(RunError::new(
                "issue configuration",
                "SM120 effective mode requires an sm_120 architecture card",
            ));
        }
        if config.architecture_spec.is_some() || config.hypothetical_issue_ports.is_some() {
            return Err(RunError::new(
                "arguments",
                "SM120 effective mode cannot combine with --architecture-spec or --hypothetical-issue-ports",
            ));
        }
        let alpha = alpha_config();
        let issue = IssueModelParams::from_card_alpha(arch, card, alpha)
            .map_err(at("SM120 alpha issue configuration"))?;
        let issue = config.sm120_effective_l1.map_or(Ok(issue), |effective| {
            issue
                .with_l1_timing(effective.hit_latency_cycles, effective.miss_latency_cycles)
                .map_err(at("SM120 effective L1 timing"))
        })?;
        return Ok((issue, None));
    }
    if config.sm120_effective_l1.is_some() {
        return Err(RunError::new(
            "arguments",
            "--sm120-effective-l1 requires an sm_120 architecture card",
        ));
    }
    let issue = match card.required_issue_resources() {
        Ok(_) => IssueModelParams::from_card(arch, card).map_err(at("issue configuration"))?,
        Err(CalibError::BlockResearchGap { .. }) if !card.has_any_issue_resource_blocks() => {
            // Keep old cards readable through an explicit compatibility path.
            // The moment a card carries any scheduler block, partial schemas
            // fail closed instead of borrowing architecture-local defaults.
            let latencies = card
                .required_latencies()
                .map_err(at("architecture latencies"))?;
            IssueModelParams::from_legacy_card_latencies_for_arch(
                arch,
                latencies.l1_hit,
                latencies.l2_hit,
                latencies.dram,
                latencies.shared,
            )
            .map_err(at("legacy issue configuration"))?
        }
        Err(error) => return Err(at("issue configuration")(error)),
    };
    if config.hypothetical_issue_ports.is_some() && config.architecture_spec.is_some() {
        return Err(RunError::new(
            "arguments",
            "--hypothetical-issue-ports and --architecture-spec are mutually exclusive",
        ));
    }
    let architecture_spec_issue_ports = config
        .architecture_spec
        .as_deref()
        .map(|path| trace_safe_issue_ports(path, arch))
        .transpose()?;
    let requested_issue_ports = architecture_spec_issue_ports.or(config.hypothetical_issue_ports);
    let issue = match requested_issue_ports {
        Some(ports) => issue
            .with_hypothetical_issue_ports(ports)
            .map_err(at("hypothetical issue configuration")),
        None => Ok(issue),
    }?;
    Ok((issue, architecture_spec_issue_ports))
}

pub(crate) const fn alpha_config() -> Sm120AlphaConfig {
    Sm120AlphaConfig::default_sm120_effective()
}

/// Resolve the deliberately narrow architecture-spec bridge into the one
/// trace-safe production resource currently exposed by `sim-sm`. A custom
/// functional ISA, multi-group capacity, or arbitrary resource graph cannot
/// be applied to a fixed SASS trace and therefore fails closed here.
pub(super) fn trace_safe_issue_ports(path: &Path, arch: sim_arch::ArchId) -> Result<u32, RunError> {
    let raw = read_direct_input(path, "architecture spec", MAX_DIRECT_CARD_BYTES)?;
    let spec = ArchitectureSpec::parse(path, &raw).map_err(at("architecture spec"))?;
    if spec.architecture() != arch.token() {
        return Err(RunError::new(
            "architecture spec",
            format!(
                "trace-safe profile architecture {:?} does not match card {:?}",
                spec.architecture(),
                arch.token()
            ),
        ));
    }
    if spec.isa() != "trace_sass_v1" {
        return Err(RunError::new(
            "architecture spec",
            format!(
                "ISA {:?} is functional-only; production SASS replay requires trace_sass_v1",
                spec.isa()
            ),
        ));
    }
    if spec.instruction_set().len() != 1 || spec.instruction_set()[0] != "trace_sass" {
        return Err(RunError::new(
            "architecture spec",
            "trace-safe issue profile must declare only the trace_sass operation",
        ));
    }
    if spec.execution().simt_width() != 32 || spec.execution().max_resident_groups() != 1 {
        return Err(RunError::new(
            "architecture spec",
            "trace-safe SM86 profile requires simt_width=32 and max_resident_groups=1",
        ));
    }
    let find = |name: &str| {
        spec.resources()
            .iter()
            .find(|resource| resource.name() == name)
    };
    let alu = find("alu").ok_or_else(|| {
        RunError::new(
            "architecture spec",
            "trace-safe profile requires an alu resource",
        )
    })?;
    let memory = find("memory").ok_or_else(|| {
        RunError::new(
            "architecture spec",
            "trace-safe profile requires a memory resource",
        )
    })?;
    if spec.resources().len() != 2 || !spec.operation_resources().is_empty() {
        return Err(RunError::new(
            "architecture spec",
            "trace-safe issue profile must contain only unbound alu and memory resources",
        ));
    }
    if alu.kind() != ResourceKind::Compute || memory.kind() != ResourceKind::Memory {
        return Err(RunError::new(
            "architecture spec",
            "trace-safe alu/memory resources have incompatible kinds",
        ));
    }
    if alu.issue_width() != memory.issue_width() {
        return Err(RunError::new(
            "architecture spec",
            "trace-safe profile must use one shared issue width for alu and memory",
        ));
    }
    if alu.latency_cycles().is_some() || memory.latency_cycles().is_some() {
        return Err(RunError::new(
            "architecture spec",
            "trace-safe issue profile must omit latency_cycles; this bridge consumes issue width only",
        ));
    }
    if alu.initiation_interval() != 1
        || memory.initiation_interval() != 1
        || alu.max_in_flight().is_some()
        || memory.max_in_flight().is_some()
    {
        return Err(RunError::new(
            "architecture spec",
            "trace-safe issue profile must use initiation_interval=1 and omit max_in_flight; this bridge consumes issue width only",
        ));
    }
    if !(1..=4).contains(&alu.issue_width()) {
        return Err(RunError::new(
            "architecture spec",
            "trace-safe issue width must be in 1..=4",
        ));
    }
    Ok(alu.issue_width())
}

pub(super) fn parse_memory_service_candidates(
    bytes: &[u8],
) -> Result<Vec<MemoryServiceCandidate>, RunError> {
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_CANDIDATE_FILE_BYTES {
        return Err(RunError::new(
            "memory-service candidates",
            format!(
                "candidate document exceeds {} bytes",
                MAX_CANDIDATE_FILE_BYTES
            ),
        ));
    }
    let document: MemoryServiceCandidateDocument =
        strict_json_from_slice(bytes).map_err(at("memory-service candidates"))?;
    if document.schema_version != 1 {
        return Err(RunError::new(
            "memory-service candidates",
            format!(
                "unsupported schema_version {}; expected 1",
                document.schema_version
            ),
        ));
    }
    if document.candidates.is_empty() {
        return Err(RunError::new(
            "memory-service candidates",
            "candidate list must not be empty",
        ));
    }
    if document.candidates.len() > MAX_MEMORY_SERVICE_CANDIDATES {
        return Err(RunError::new(
            "memory-service candidates",
            format!(
                "candidate list has {} entries; maximum is {}",
                document.candidates.len(),
                MAX_MEMORY_SERVICE_CANDIDATES
            ),
        ));
    }
    for (index, candidate) in document.candidates.iter().copied().enumerate() {
        candidate
            .validate()
            .map_err(at("memory-service candidates"))?;
        if document.candidates[..index]
            .iter()
            .any(|prior| prior.candidate_id == candidate.candidate_id)
        {
            return Err(RunError::new(
                "memory-service candidates",
                format!("candidate id {} is duplicated", candidate.candidate_id),
            ));
        }
    }
    Ok(document.candidates)
}

pub(super) fn parse_memory_admission_candidates(
    bytes: &[u8],
) -> Result<Vec<MemoryAdmissionCandidate>, RunError> {
    parse_memory_admission_document(bytes, "memory-admission candidates", false)
}

pub(super) fn parse_memory_admission_feedback(
    bytes: &[u8],
) -> Result<MemoryAdmissionCandidate, RunError> {
    let candidates = parse_memory_admission_document(bytes, "memory-admission feedback", true)?;
    Ok(candidates[0])
}

pub(super) fn load_memory_admission_candidates(
    path: Option<&PathBuf>,
) -> Result<Option<Vec<MemoryAdmissionCandidate>>, RunError> {
    path.map(|path| {
        let bytes = read_direct_input(
            path,
            "memory-admission candidates",
            MAX_CANDIDATE_FILE_BYTES + 1,
        )?;
        parse_memory_admission_candidates(&bytes)
    })
    .transpose()
}

pub(super) fn load_memory_admission_feedback(
    path: Option<&PathBuf>,
) -> Result<Option<MemoryAdmissionCandidate>, RunError> {
    path.map(|path| {
        let bytes = read_direct_input(
            path,
            "memory-admission feedback",
            MAX_CANDIDATE_FILE_BYTES + 1,
        )?;
        parse_memory_admission_feedback(&bytes)
    })
    .transpose()
}

#[derive(Debug, Serialize)]
pub struct RunReport {
    pub schema_version: u32,
    pub simulator_version: &'static str,
    pub architecture: String,
    pub memory_mode: &'static str,
    pub cache_scope: &'static str,
    pub carded_dram_bandwidth: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_payload_sidecar_records: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_payload_event_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload_ledger: Option<PayloadLedgerReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dirty_writeback_capacity_sectors: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coherence_service_cycles: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coherence_capacity_sectors: Option<u32>,
    #[serde(skip_serializing_if = "is_false")]
    pub l1_visibility_protocol: bool,
    #[serde(skip_serializing_if = "is_false")]
    pub l1_resident_writeback: bool,
    #[serde(skip_serializing_if = "is_false")]
    pub fast_frontier_only: bool,
    #[serde(skip_serializing_if = "is_false")]
    pub sm120_alpha: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sm120_alpha_config: Option<Sm120AlphaConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sm120_alpha_profile_id: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sm120_effective_l1: Option<Sm120EffectiveL1Config>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sm120_effective_l2: Option<Sm120EffectiveL2Config>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hypothetical_issue_ports: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub architecture_spec: Option<String>,
    pub memory_service_candidate_count: usize,
    pub memory_admission_candidate_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_admission_feedback_candidate_id: Option<u8>,
    pub launches: Vec<LaunchReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fast_frontier: Option<Vec<FastFrontierLaunchReport>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_session: Option<CacheSessionReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ncu_observation: Option<ncu::NcuObservation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latent_envelope: Option<LatentEnvelopeReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sim_total_wall_ns: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sim_replay_wall_ns: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct LatentEnvelopeReport {
    pub schema_version: u32,
    pub role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub external_endpoint: Option<EndpointDescriptor>,
    pub launches: Vec<LatentLaunchEnvelopeReport>,
}

#[derive(Debug, Serialize)]
pub struct LatentLaunchEnvelopeReport {
    pub launch_index: usize,
    pub factors: LatentFactors,
    pub class_bounds: sim_sm::PipelineClassEnvelope,
    pub resource_demand: sim_semantics::ResourceDemandEnvelope,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resource_service_projection: Option<ResourceServiceProjection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_composition: Option<ServiceComposition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_queue_projection: Option<ServiceQueueProjection>,
    pub cache_candidates: CacheCandidateSummary,
    pub operand_bank_pressure: sim_sm::OperandBankPressure,
    pub frontiers: FrontierVector,
    pub endpoint: EndpointDescriptor,
    pub ncu_projection: NcuObserverProjection,
}

pub(super) fn build_latent_envelope(
    launches: &[LaunchReport],
    ncu_observed: bool,
    service_profile: Option<&ServiceRateProfile>,
    service_stage_role: ServiceStageRole,
) -> Result<LatentEnvelopeReport, RunError> {
    let endpoint = TimingReport::simulator_endpoint_descriptor().map_err(at("latent endpoint"))?;
    let external_endpoint = ncu_observed
        .then(|| TimingReport::ncu_endpoint_descriptor())
        .transpose()
        .map_err(at("latent NCU endpoint"))?;
    let mut reports = Vec::with_capacity(launches.len());
    for (launch_index, launch) in launches.iter().enumerate() {
        let timing = launch.timing.as_ref().ok_or_else(|| {
            RunError::new(
                "latent envelope",
                format!("launch {launch_index} has no simulator timing report"),
            )
        })?;
        let mut frontiers = timing.latent_frontiers().map_err(at("latent frontiers"))?;
        let resource_demand = launch
            .trace_metrics
            .physical_pipeline
            .resource_demand_envelope()
            .map_err(at("latent resource demand"))?;
        let cache_candidates = CacheCandidateSummary::from_topology_with_witnesses(
            launch.trace_metrics.coherence.decoded_global_lines,
            launch.trace_metrics.coherence.cross_sm_lines,
            launch.trace_metrics.coherence.cross_sm_read_write_lines,
            launch.trace_metrics.coherence.cross_sm_write_write_lines,
            CacheWitnessDomains::from_trace_observation(
                launch.trace_metrics.sectors_decoded,
                launch.trace_metrics.coherence.unknown_sm_or_line_records,
                launch.trace_metrics.coherence.instrumentation_order_records,
                launch
                    .trace_metrics
                    .coherence
                    .instrumentation_order_missing_records,
            )
            .map_err(at("latent cache witnesses"))?,
        )
        .map_err(at("latent cache candidates"))?;
        frontiers.cache_history = cache_candidates.history_frontier();
        let resource_service_projection = service_profile
            .map(|profile| resource_demand.project_service_cycles(profile))
            .transpose()
            .map_err(at("latent service projection"))?;
        let service_composition = resource_service_projection
            .as_ref()
            .map(|service| {
                ServiceComposition::new_with_role(
                    frontiers.modeled_event,
                    FrontierBound {
                        lower: Some(service.lower_cycles),
                        upper: service.upper_cycles,
                    },
                    ServiceOverlapMode::SerialAfterAdmission,
                    service_stage_role,
                )
            })
            .transpose()
            .map_err(at("latent service composition"))?;
        let service_queue_projection = service_profile
            .map(|profile| resource_demand.project_service_queue(profile, frontiers.modeled_event))
            .transpose()
            .map_err(at("latent service queue projection"))?;
        if resource_service_projection.is_some()
            && service_composition.as_ref().is_some_and(|composition| {
                composition.role != ServiceStageRole::IndependentPostReplay
            })
        {
            frontiers
                .required_completion
                .insert(CompletionRequirement::ResourceService);
            frontiers.resource_service = service_composition
                .as_ref()
                .map(|composition| composition.completion)
                .unwrap_or_else(FrontierBound::unknown);
            // Residual/replacement stages may feed the observer endpoint. A
            // full-trace service projection is explicitly independent
            // post-replay and remains a side-by-side model diagnostic.
            frontiers.endpoint_end = frontiers.simulator_completion();
            frontiers
                .validate()
                .map_err(at("latent service frontier"))?;
        }
        reports.push(LatentLaunchEnvelopeReport {
            launch_index,
            factors: launch
                .trace_metrics
                .latent_factors()
                .map_err(at("latent factors"))?,
            class_bounds: launch
                .trace_metrics
                .physical_pipeline
                .class_count_envelope()
                .map_err(at("latent class bounds"))?,
            resource_service_projection,
            service_composition,
            service_queue_projection,
            cache_candidates,
            resource_demand,
            operand_bank_pressure: launch.trace_metrics.operand_bank_pressure.clone(),
            ncu_projection: NcuObserverProjection::from_frontiers(
                frontiers.clone(),
                service_profile,
            )
            .map_err(at("latent NCU projection"))?,
            frontiers,
            endpoint: endpoint.clone(),
        });
    }
    Ok(LatentEnvelopeReport {
        schema_version: sim_semantics::LATENT_SCHEMA_VERSION,
        role: "opt_in_unknown_preserving_model_envelope_not_hardware_truth",
        external_endpoint,
        launches: reports,
    })
}

#[derive(Debug, Serialize)]
pub struct CacheSessionReport {
    pub scope: &'static str,
    pub device_id: i64,
    pub stream_id: i64,
    pub completed_launches: usize,
    pub cumulative_l2: L2CounterReport,
}

#[derive(Debug, Serialize)]
pub struct AlphaApproximationReport {
    pub status: &'static str,
    pub profile_id: &'static str,
    pub timing_scope: &'static str,
    pub async_policy: &'static str,
    pub approximation_kinds: Vec<&'static str>,
    pub affected_opcode_counts: BTreeMap<String, u64>,
    pub scheduler_assumptions: Vec<&'static str>,
    pub issue_envelope: Sm120IssueEnvelopeReport,
    pub service_envelope: Sm120ServiceEnvelopeReport,
    pub completion_ledger: Sm120CompletionLedgerReport,
}

#[derive(Debug, Serialize)]
pub struct ReadinessReport {
    pub schema_version: u32,
    #[serde(flatten)]
    pub diagnostics: ReadinessDiagnostics,
}

#[derive(Debug, Serialize)]
pub struct FrontEndCounterReport {
    pub compulsory_instruction_read_sectors: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub compulsory_constant_read_sectors: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub compulsory_instruction_line_read_sectors: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub static_instruction_line_read_sectors: u64,
}

#[derive(Debug, Serialize)]
pub struct L1CounterReport {
    pub hits: u64,
    pub misses: u64,
    pub merges: u64,
    pub pending_lookup_hits: u64,
    pub stall_events: u64,
    pub mshr_stall_occupancy_samples: u64,
    pub mshr_stall_occupancy_sum: u64,
    pub line_fills: u64,
    pub bypass_sectors: u64,
    pub store_sectors: u64,
    pub store_hits: u64,
    pub store_tag_hits: u64,
    pub write_lookup_hits: u64,
    pub reduction_sectors: u64,
}

#[derive(Debug, Serialize)]
pub struct L2SourceLaneReport {
    pub sectors: u64,
    pub hits: u64,
    pub pending_pressure_samples: u64,
    pub pending_pressure_sum: u64,
}

pub(super) fn l2_source_counter_report(value: &L2SourceSplit) -> L2SourceCounterReport {
    let lane = |source| {
        let (sectors, hits) = value.source(source);
        let index = source as usize;
        L2SourceLaneReport {
            sectors,
            hits,
            pending_pressure_samples: value.pending_pressure_samples[index],
            pending_pressure_sum: value.pending_pressure_sum[index],
        }
    };
    L2SourceCounterReport {
        l1_fill: lane(L2Source::L1Fill),
        bypass: lane(L2Source::Bypass),
        untagged: lane(L2Source::Untagged),
        store: lane(L2Source::Store),
        read_lookup_hits_without_pending_merges: value
            .read_lookup_hits
            .saturating_sub(value.read_pending_merge_hits),
        read_pending_merge_hits: value.read_pending_merge_hits,
        store_lookup_requests: value.store_lookup_requests,
        store_packet_lookup_hits: value.store_packet_lookup_hits,
        store_packet_resident_lookup_hits: value.store_packet_resident_lookup_hits,
        store_packet_candidate_sectors: value.store_packet_candidate_sectors,
        client_scope_requests: value.client_scope_requests,
        client_scope_resident_hits: value.client_scope_resident_hits,
        client_scope_pending_merges: value.client_scope_pending_merges,
        client_scope_fill_misses: value.client_scope_fill_misses,
        ltc128b_shadow_promoted_sectors: value.ltc128b_shadow_promoted_sectors,
    }
}

pub(super) fn is_zero_u64(value: &u64) -> bool {
    *value == 0
}

#[derive(Debug, Serialize)]
pub struct TailReport {
    pub reduction_read_modify_write: bool,
    pub dirty_writeback: bool,
    pub coherence: bool,
    pub async_copy_shared_visibility: bool,
}

pub(super) fn tail_report(tails: TailFlags) -> TailReport {
    TailReport {
        reduction_read_modify_write: tails.red_rmw,
        dirty_writeback: tails.dirty_writeback,
        coherence: tails.coherence,
        async_copy_shared_visibility: tails.ldgsts_dst_shared_timing,
    }
}

pub(crate) fn sm120_front_end_envelope_cycles(result: &FullKernelReplayResult) -> Option<u64> {
    if result.arch() != sim_arch::ArchId::Sm120 {
        return None;
    }
    let config = Sm120AlphaConfig::default_sm120_effective();
    if config.front_end_line_cycles_denominator == 0 {
        return None;
    }
    let lines = result.front_end().compulsory_instruction_read_sectors();
    let scaled = lines
        .checked_mul(config.front_end_line_cycles_numerator)?
        .checked_add(config.front_end_line_cycles_denominator - 1)?
        / config.front_end_line_cycles_denominator;
    config.front_end_base_cycles.checked_add(scaled)
}

pub(super) fn effective_gpc_completion_frontier_report(
    result: &FullKernelReplayResult,
) -> Result<CompletionFrontierReport, RunError> {
    let envelope = sm120_front_end_envelope_cycles(result).unwrap_or(0);
    let add = |cycles: Option<u64>| match cycles {
        Some(value) => value
            .checked_add(envelope)
            .map(Some)
            .ok_or_else(|| RunError::new("timing projection", "GPC front-end envelope overflow")),
        None => Ok(None),
    };
    Ok(match result.modeled_completion_frontier() {
        ModeledCompletionFrontier::Complete { cycles } => CompletionFrontierReport::Complete {
            cycles: add(cycles)?,
        },
        ModeledCompletionFrontier::Incomplete {
            lower_bound_cycles,
            unmodeled_tails,
        } => CompletionFrontierReport::IncompleteUnmodeledRetirementTails {
            lower_bound_cycles: add(lower_bound_cycles)?,
            unmodeled_tails: tail_report(unmodeled_tails),
        },
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn counters(
    instructions_issued: u64,
    records_seen: u64,
    front_end: sim_sm::FrontEndCounters,
    l1: Option<&L1Counters>,
    l2: Option<&L2Counters>,
    l2_by_source: Option<&L2SourceSplit>,
    dram: Option<&DramStats>,
    tails: TailFlags,
) -> CounterReport {
    let counter_arch = l1
        .and_then(|value| value.arch())
        .or_else(|| l2.and_then(|value| value.arch()))
        .unwrap_or(ArchId::Sm86);
    CounterReport {
        instructions_issued,
        records_seen,
        front_end: FrontEndCounterReport {
            compulsory_instruction_read_sectors: front_end.compulsory_instruction_read_sectors(),
            compulsory_constant_read_sectors: if counter_arch == ArchId::Sm120 {
                front_end.compulsory_constant_read_sectors()
            } else {
                0
            },
            compulsory_instruction_line_read_sectors: if counter_arch == ArchId::Sm86 {
                front_end.compulsory_instruction_line_read_sectors()
            } else {
                0
            },
            static_instruction_line_read_sectors: if counter_arch == ArchId::Sm86 {
                front_end.static_instruction_line_read_sectors()
            } else {
                0
            },
        },
        l1: l1.map(|value| L1CounterReport {
            hits: value.hits,
            misses: value.misses,
            merges: value.merges,
            pending_lookup_hits: value.pending_lookup_hits,
            stall_events: value.stall_events,
            mshr_stall_occupancy_samples: value.mshr_stall_occupancy_samples,
            mshr_stall_occupancy_sum: value.mshr_stall_occupancy_sum,
            line_fills: value.line_fills,
            bypass_sectors: value.bypass_sectors,
            store_sectors: value.store_sectors,
            store_hits: value.store_hits,
            store_tag_hits: value.store_tag_hits,
            write_lookup_hits: value.write_lookup_hits,
            reduction_sectors: value.reduction_sectors,
        }),
        l2: l2.map(l2_counter_report),
        l2_by_source: l2_by_source.map(l2_source_counter_report),
        dram: dram.map(|value| DramCounterReport {
            read_sectors: value.read_sectors,
            service_rate_bytes_per_sec: value.service_rate_bytes_per_sec,
            l2_frontier_samples: value.l2_frontier_samples,
            l2_issue_to_fill_cycles_sum: value.l2_issue_to_fill_cycles_sum,
            l2_issue_to_fill_cycles_max: value.l2_issue_to_fill_cycles_max,
            l2_issue_to_response_cycles_sum: value.l2_issue_to_response_cycles_sum,
            l2_issue_to_response_cycles_max: value.l2_issue_to_response_cycles_max,
            l2_service_delay_cycles_sum: value.l2_service_delay_cycles_sum,
            l2_service_delay_cycles_max: value.l2_service_delay_cycles_max,
            atomic_rmw_read_sectors: value.atomic_rmw_read_sectors,
            reduction_rmw_read_sectors: value.reduction_rmw_read_sectors,
            bandwidth_delayed_responses: value.bw_delayed_responses,
            l2_evicted_lines: value.l2_evicted_lines,
            l2_evicted_valid_sectors: value.l2_evicted_valid_sectors,
            l2_evicted_pending_lines: value.l2_evicted_pending_lines,
            l2_evicted_pending_sectors: value.l2_evicted_pending_sectors,
            l2_dirty_eviction_lines: value.l2_dirty_eviction_lines,
            l2_dirty_evicted_sectors: value.l2_dirty_evicted_sectors,
            writeback_completed_lines: value.writeback_completed_lines,
            writeback_completed_sectors: value.writeback_completed_sectors,
            writeback_completed_bytes: value.writeback_completed_bytes,
            l1_writeback_completed_lines: value.l1_writeback_completed_lines,
            l1_writeback_completed_sectors: value.l1_writeback_completed_sectors,
            l1_writeback_completed_bytes: value.l1_writeback_completed_bytes,
            l1_payload_forwarded_sectors: value.l1_payload_forwarded_sectors,
            l1_payload_forwarded_bytes: value.l1_payload_forwarded_bytes,
            l1_payload_forwarding_digest_xor: value.l1_payload_forwarding_digest_xor,
            writeback_completion_frontier: value.writeback_completion_frontier,
            coherence_completed_transactions: value.coherence_completed_transactions,
            coherence_completed_sectors: value.coherence_completed_sectors,
            coherence_completion_frontier: value.coherence_completion_frontier,
            maximum_terminal_l2_dirty_lines: value.maximum_terminal_l2_dirty_lines,
            maximum_terminal_l2_dirty_sectors: value.maximum_terminal_l2_dirty_sectors,
            maximum_terminal_l1_dirty_lines: value.maximum_terminal_l1_dirty_lines,
            maximum_terminal_l1_dirty_sectors: value.maximum_terminal_l1_dirty_sectors,
        }),
        unmodeled_tails: tail_report(tails),
    }
}

pub(super) fn alpha_approximation_report(
    result: &FullKernelReplayResult,
) -> Option<AlphaApproximationReport> {
    if !matches!(
        result.configuration().issue_params().resource_source(),
        sim_sm::IssueResourceSource::Sm120Alpha
    ) {
        return None;
    }
    let mut affected_opcode_counts = BTreeMap::new();
    let mut approximation_kinds = BTreeSet::new();
    let mut affected_async_records = 0u64;
    for (opcode, activity) in &result.trace_metrics().by_opcode {
        let count = activity.warp_instructions;
        if count == 0 {
            continue;
        }
        if opcode.starts_with("UTMA") {
            approximation_kinds.insert("tma_completion_unmodeled");
            affected_opcode_counts.insert(opcode.clone(), count);
            affected_async_records = affected_async_records.saturating_add(count);
        } else if opcode.starts_with("LDGSTS") {
            approximation_kinds.insert("cp_async_completion_approximated");
            affected_opcode_counts.insert(opcode.clone(), count);
            affected_async_records = affected_async_records.saturating_add(count);
        } else if matches!(opcode.as_str(), "QMMA" | "OMMA" | "HMMA" | "IMMA" | "BMMA") {
            approximation_kinds.insert("tensor_pipeline_timing_approximated");
            affected_opcode_counts.insert(opcode.clone(), count);
        } else if matches!(opcode.as_str(), "MBAR" | "UTMABAR") {
            approximation_kinds.insert("mbarrier_completion_approximated");
            affected_opcode_counts.insert(opcode.clone(), count);
        }
    }
    Some(AlphaApproximationReport {
        status: "provisional_flat_alpha",
        profile_id: Sm120AlphaConfig::PROFILE_ID,
        timing_scope: "model_relative_not_hardware_timing",
        async_policy: "typed_async_completion_separate_from_logical_global_shared_service",
        approximation_kinds: approximation_kinds.into_iter().collect(),
        affected_opcode_counts,
        scheduler_assumptions: vec![
            "alpha_resource_mapping_assumption",
            "alpha_control_barrier_domain_assumption",
        ],
        issue_envelope: Sm120IssueEnvelopeReport::from_issue_params(
            result.configuration().issue_params(),
        ),
        service_envelope: Sm120ServiceEnvelopeReport::observation_only(),
        completion_ledger: Sm120CompletionLedgerReport::provisional(affected_async_records),
    })
}

pub(crate) fn launch_report(
    result: FullKernelReplayResult,
    arch_card: &ArchCard,
    external_resources: Option<&resources::LaunchResourceObservation>,
    cache_handoff: Option<CacheHandoffReport>,
    expected_memory_admission_feedback: Option<MemoryAdmissionCandidate>,
    include_metric_predictions: bool,
) -> Result<LaunchReport, RunError> {
    let configured_memory_admission_feedback = result.configuration().memory_admission_feedback();
    if configured_memory_admission_feedback != expected_memory_admission_feedback {
        return Err(RunError::new(
            "replay configuration",
            format!(
                "timing-active memory-admission feedback identity differs: expected {expected_memory_admission_feedback:?}, got {configured_memory_admission_feedback:?}"
            ),
        ));
    }
    let binding = result.coverage().launch().clone();
    let (metric_predictions, cache_lookup_predictions) = if include_metric_predictions {
        let bundle = emit_launch(&result).map_err(at("metric projection"))?;
        let cache_lookup_predictions = cache_lookup_prediction_reports(
            &bundle,
            result.l1(),
            result.l2(),
            result.l2_by_source(),
            Some(result.trace_metrics()),
        )?;
        let metric_predictions = bundle
            .metrics()
            .iter()
            .map(|prediction| NcuMetricProjectionReport {
                arch: prediction.arch().token().to_owned(),
                key: prediction.key(),
                ncu_metric: ncu_metric_name_for_arch(
                    prediction.arch(),
                    prediction.key(),
                    prediction.ncu_metric(),
                ),
                unit: prediction.unit(),
                outcome: *prediction.outcome(),
            })
            .collect();
        (Some(metric_predictions), Some(cache_lookup_predictions))
    } else {
        (None, None)
    };
    let coverage = CoverageReport {
        kind: "full",
        blocks_observed: result.coverage().blocks_observed(),
        population_blocks: Some(result.coverage().blocks_observed()),
    };
    let counter_report = counters(
        result.instructions_issued(),
        result.records_seen(),
        result.front_end(),
        result.l1(),
        result.l2(),
        result.l2_by_source(),
        result.dram(),
        result.unmodeled_tails(),
    );
    let sm_activity = result
        .per_sm()
        .iter()
        .map(|(&sm, value)| (sm, value.activity.clone()))
        .collect();
    let native_metrics = metrics::project(&result, arch_card, external_resources)?;
    let mut readiness = ReadinessDiagnostics::default();
    for per_sm in result.per_sm().values() {
        readiness
            .add_checked(&per_sm.readiness)
            .map_err(at("readiness aggregation"))?;
    }
    let readiness_schema_version = readiness.schema_version();
    let timing = Some(TimingReport {
        completion_frontier_semantics: if sm120_front_end_envelope_cycles(&result).is_some() {
            "model_internal_gpc_effective_front_end_envelope_not_physical_completion"
        } else if alpha_approximation_report(&result).is_some() {
            "model_internal_approximate_not_physical_completion"
        } else {
            "model_internal_frontier"
        },
        effective_gpc_front_end_envelope_cycles: sm120_front_end_envelope_cycles(&result),
        alpha_approximation: alpha_approximation_report(&result),
        diagnostic_last_issue_cycles: result.diagnostic_last_issue_cycles(),
        last_processed_modeled_system_event_time: result.last_processed_modeled_system_event_time(),
        terminal_modeled_event_time: result.terminal_modeled_event_time(),
        sm_active_terminal_frontier: Some(SmActiveTerminalFrontierReport {
            cycles: result.sm_active_terminal_event_time(),
            role: "causal_sm_active_terminal_lower_bound_excludes_write_ack_and_service_only_events",
        }),
        sm_active_observation_frontier: result.sm_active_observation_frontier().map(Into::into),
        gpc_observation_frontier: result.gpc_observation_frontier().map(Into::into),
        completion_frontier: effective_gpc_completion_frontier_report(&result)?,
        readiness: ReadinessReport {
            schema_version: readiness_schema_version,
            diagnostics: readiness,
        },
    });
    Ok(LaunchReport {
        device_id: binding.device_id(),
        stream_id: binding.stream_id(),
        kernel_id: binding.kernel_id(),
        function_unique_id: binding.function_unique_id(),
        kernel_name: binding.kernel_mangled_name().to_owned(),
        grid: binding.grid(),
        block: binding.block(),
        cluster: binding.cluster(),
        coverage,
        counters: counter_report,
        trace_metrics: result.trace_metrics().clone(),
        sm_activity,
        native_metrics,
        metric_predictions,
        cache_lookup_predictions,
        timing,
        cache_handoff,
    })
}

pub(crate) fn resolve_mode(
    config: &RunConfig,
    arch_card: &ArchCard,
    issue: IssueModelParams,
) -> Result<ReplayMemMode, RunError> {
    if config.sm120_alpha
        && config.memory_mode != MemoryMode::Flat
        && config.sm120_effective_l1.is_none()
    {
        return Err(RunError::new(
            "arguments",
            "SM120 cache mode requires --sm120-effective-l1; use the flat mode only with --fast-frontier-only",
        ));
    }
    if config.sm120_effective_l1.is_some()
        && (!config.sm120_alpha || config.memory_mode == MemoryMode::Flat)
    {
        return Err(RunError::new(
            "arguments",
            "--sm120-effective-l1 requires an sm_120 architecture card and --mode l1 or --mode l1-l2",
        ));
    }
    if config.sm120_effective_l2.is_some()
        && (!config.sm120_alpha || config.memory_mode != MemoryMode::L1L2)
    {
        return Err(RunError::new(
            "arguments",
            "--sm120-effective-l2 requires an sm_120 architecture card and --mode l1-l2",
        ));
    }
    if config.dirty_writeback_capacity_sectors == Some(0) {
        return Err(RunError::new(
            "arguments",
            "dirty writeback capacity must be positive",
        ));
    }
    if config.dirty_writeback_capacity_sectors.is_some()
        && (config.memory_mode != MemoryMode::L1L2 || !config.carded_dram_bandwidth)
    {
        return Err(RunError::new(
            "arguments",
            "dirty writeback requires --mode l1-l2 together with --carded-dram",
        ));
    }
    if config.coherence_service_cycles == Some(0) || config.coherence_capacity_sectors == Some(0) {
        return Err(RunError::new(
            "arguments",
            "coherence service cycles and capacity must be positive",
        ));
    }
    if config.coherence_capacity_sectors.is_some() != config.coherence_service_cycles.is_some() {
        return Err(RunError::new(
            "arguments",
            "coherence service cycles and capacity must be supplied together",
        ));
    }
    if config.l1_visibility_protocol && config.coherence_service_cycles.is_none() {
        return Err(RunError::new(
            "arguments",
            "L1 visibility protocol requires --coherence-service-cycles and --coherence-capacity-sectors",
        ));
    }
    if config.l1_resident_writeback
        && (config.coherence_service_cycles.is_none()
            || config.dirty_writeback_capacity_sectors.is_none()
            || config.memory_mode != MemoryMode::L1L2
            || !config.carded_dram_bandwidth
            || !config.l1_visibility_protocol)
    {
        return Err(RunError::new(
            "arguments",
            "L1 resident writeback requires l1-l2, --carded-dram, --dirty-writeback-capacity-sectors, the coherence service pair, and --l1-visibility-protocol",
        ));
    }
    if config.coherence_service_cycles.is_some() && config.memory_mode != MemoryMode::L1L2 {
        return Err(RunError::new(
            "arguments",
            "coherence service requires --mode l1-l2",
        ));
    }
    if config.memory_mode == MemoryMode::Flat {
        if config.carded_dram_bandwidth {
            return Err(RunError::new(
                "arguments",
                "--carded-dram requires --mode l1-l2",
            ));
        }
        return Ok(ReplayMemMode::Flat);
    }

    let mut l1 = if let Some(effective) = config.sm120_effective_l1 {
        effective.as_l1_config()
    } else {
        let pchase_path = config.pchase_card.as_ref().ok_or_else(|| {
            RunError::new(
                "arguments",
                "--pchase-card is required for l1 and l1-l2 modes",
            )
        })?;
        let pchase = load_pchase_card(pchase_path)?;
        let geometry = pchase
            .resolve_walk_geometry_for_arch(issue.arch(), None)
            .map_err(at("L1 geometry"))?;
        issue.l1_config(geometry.capacity.line_capacity, geometry.mshr.count)
    };
    if issue.arch() == sim_arch::ArchId::Sm86 && config.sm120_effective_l1.is_none() {
        let geometry = arch_card
            .require_l1_geometry()
            .map_err(at("SM86 L1 set geometry"))?;
        // The card documents the set/way domain, while the p-chase resolver
        // retains the active carveout capacity.  Keep both sources explicit:
        // the pressure-bounded modulo/folded candidate is an effective
        // scenario and the proprietary hash remains unknown; global capacity
        // still follows the p-chase point.
        l1.set_count = Some(geometry.sets);
        l1.ways = Some(geometry.ways);
        l1.set_index = L1SetIndex::ModuloXorFallback;
    }
    if config.memory_mode == MemoryMode::L1 {
        if config.carded_dram_bandwidth {
            return Err(RunError::new(
                "arguments",
                "--carded-dram requires --mode l1-l2",
            ));
        }
        return Ok(ReplayMemMode::L1Const(l1));
    }

    let l2 = if let Some(effective) = config.sm120_effective_l2 {
        if effective.write_policy != L2_WRITE_ALLOCATE_POLICY {
            return Err(RunError::new(
                "SM120 effective L2 configuration",
                format!(
                    "unsupported effective L2 write policy {:?}; expected {:?}",
                    effective.write_policy, L2_WRITE_ALLOCATE_POLICY
                ),
            ));
        }
        L2Config::from_effective_geometry_for_arch(
            issue.arch(),
            effective.capacity_bytes,
            effective.line_bytes,
            effective.sector_bytes,
            effective.num_slices,
            effective.associativity,
            effective.replacement,
            effective.slice_hash,
        )
        .map_err(at("SM120 effective L2 configuration"))?
    } else {
        let l2_geometry = arch_card.require_l2_geometry().map_err(at("L2 geometry"))?;
        L2Config::from_carded_geometry_for_arch(
            issue.arch(),
            l2_geometry.capacity_bytes,
            l2_geometry.line_bytes,
            l2_geometry.sector_bytes,
            l2_geometry.num_slices,
            l2_geometry.policy.as_deref(),
        )
        .map_err(at("L2 configuration"))?
    };
    let dram = if config.carded_dram_bandwidth {
        let card = arch_card
            .require_dram_bandwidth()
            .map_err(at("DRAM bandwidth"))?;
        let bytes_per_sec = config
            .dram_service_rate_bytes_per_sec
            .or_else(|| card.validated_peak_bytes_per_second().ok())
            .ok_or_else(|| RunError::new("DRAM bandwidth", "no validated service rate"))?;
        if bytes_per_sec == 0 {
            return Err(RunError::new(
                "DRAM bandwidth",
                "service rate must be positive",
            ));
        }
        let _card_peak = card
            .validated_peak_bytes_per_second()
            .map_err(at("DRAM bandwidth"))?;
        let sm_clock_hz = card.locked_sm_clock_hz().map_err(at("DRAM clock"))?;
        match config.dirty_writeback_capacity_sectors {
            Some(capacity_sectors) => DramBw::CardedWriteback {
                arch: issue.arch(),
                bytes_per_sec,
                sm_clock_hz,
                capacity_sectors,
            },
            None => DramBw::Carded {
                arch: issue.arch(),
                bytes_per_sec,
                sm_clock_hz,
            },
        }
    } else {
        DramBw::Unmodeled { arch: issue.arch() }
    };
    let coherence = config
        .coherence_service_cycles
        .zip(config.coherence_capacity_sectors)
        .map(|(cycles, capacity)| CoherenceConfig::new(cycles, capacity))
        .transpose()
        .map_err(at("coherence configuration"))?
        .map(|value| {
            value
                .with_l1_visibility_protocol(config.l1_visibility_protocol)
                .with_l1_resident_writeback(config.l1_resident_writeback)
        });
    Ok(ReplayMemMode::L1SharedL2 {
        l1,
        l2,
        dram,
        coherence,
    })
}

pub fn run(config: &RunConfig) -> Result<RunReport, RunError> {
    let total_started = Instant::now();
    let arch_card = load_arch_card(&config.arch_card)?;
    let arch = arch_card.arch_id().map_err(at("architecture card"))?;
    let resolved_config = config.with_architecture_defaults(arch)?;
    let config = &resolved_config;
    #[cfg(feature = "performance-counters")]
    let performance_enabled = std::env::var_os("SIM_RUN_PERFORMANCE_COUNTERS").is_some();
    let candidate_inputs = usize::from(config.memory_service_candidates.is_some())
        + usize::from(config.memory_admission_candidates.is_some())
        + usize::from(config.memory_admission_feedback.is_some());
    if candidate_inputs != 0 && arch != sim_arch::ArchId::Sm86 {
        return Err(RunError::new(
            "arguments",
            "memory candidate documents are currently scoped to the sm_86 lane; an architecture-qualified candidate schema is required before applying one to sm_120",
        ));
    }
    if config.native_payload_events.is_some() && config.native_payload_events_path.is_some() {
        return Err(RunError::new(
            "arguments",
            "programmatic native payload events cannot combine with --native-payload-events",
        ));
    }
    let file_payload_events = config
        .native_payload_events_path
        .as_ref()
        .map(|path| load_native_payload_events(path))
        .transpose()?;
    let native_payload_events = config.native_payload_events.clone().or(file_payload_events);
    let native_payload_records = config
        .native_payload_sidecar
        .as_ref()
        .map(|path| load_native_payload_sidecar_records(path).map_err(at("native payload sidecar")))
        .transpose()?;
    let native_payload_sidecar_records = native_payload_records.as_ref().map(Vec::len);
    let native_payload_event_count = native_payload_events.as_ref().map(Vec::len);
    let identity_complete_records = native_payload_records.as_ref().map(|records| {
        records
            .iter()
            .filter(|record| record.identity_complete())
            .count()
    });
    let native_payloads = native_payload_records.as_ref().map(|records| {
        records
            .iter()
            .map(|record| record.payload())
            .collect::<Vec<_>>()
    });
    if candidate_inputs > 1 {
        return Err(RunError::new(
            "arguments",
            "memory-service, memory-admission, and feedback candidate files are mutually exclusive",
        ));
    }
    if native_payload_events
        .as_ref()
        .is_some_and(|events| !events.is_empty() && candidate_inputs != 0)
    {
        return Err(RunError::new(
            "arguments",
            "event-time payload admissions cannot combine with memory candidate lanes",
        ));
    }
    if config.fast_frontier_only && candidate_inputs != 0 {
        return Err(RunError::new(
            "arguments",
            "--fast-frontier-only cannot combine with timing candidate documents",
        ));
    }
    if config.fast_frontier_only
        && (config.hypothetical_issue_ports.is_some() || config.architecture_spec.is_some())
    {
        return Err(RunError::new(
            "arguments",
            "--fast-frontier-only cannot combine with timing/resource overrides",
        ));
    }
    if config.fast_frontier_only && config.native_payload_sidecar.is_some() {
        return Err(RunError::new(
            "arguments",
            "--fast-frontier-only cannot combine with a native payload sidecar because it skips memory replay",
        ));
    }
    if config.fast_frontier_only && native_payload_events.is_some() {
        return Err(RunError::new(
            "arguments",
            "--fast-frontier-only cannot combine with event-time payload admissions",
        ));
    }
    if config.fast_frontier_only
        && (config.memory_mode != MemoryMode::Flat
            || config.cache_scope != CacheScope::Launch
            || config.carded_dram_bandwidth
            || config.dirty_writeback_capacity_sectors.is_some()
            || config.coherence_service_cycles.is_some()
            || config.coherence_capacity_sectors.is_some())
    {
        return Err(RunError::new(
            "arguments",
            "--fast-frontier-only cannot combine with native memory/cache, DRAM, writeback, or coherence settings",
        ));
    }
    if config.fast_frontier_only && config.cache_scope != CacheScope::Launch {
        return Err(RunError::new(
            "arguments",
            "--fast-frontier-only requires --cache-scope launch",
        ));
    }
    if config.fast_frontier_only && config.ncu_shaped_metrics {
        return Err(RunError::new(
            "arguments",
            "--fast-frontier-only cannot emit native NCU-shaped metric projections",
        ));
    }
    if config.fast_frontier_only && config.latent_envelope {
        return Err(RunError::new(
            "arguments",
            "--latent-envelope requires native timing replay and cannot combine with --fast-frontier-only",
        ));
    }
    if config.predictive_service_profile.is_some() && !config.latent_envelope {
        return Err(RunError::new(
            "arguments",
            "--predictive-service-profile requires --latent-envelope",
        ));
    }
    if config.ncu_launch.is_some() && config.ncu_csv.is_none() {
        return Err(RunError::new(
            "arguments",
            "--ncu-launch requires --ncu-csv",
        ));
    }
    let ncu_observation = config
        .ncu_csv
        .as_ref()
        .map(|path| load_ncu_observation(path, config.ncu_launch))
        .transpose()?;
    let predictive_service_profile =
        load_predictive_service_profile(config.predictive_service_profile.as_ref())?;
    if config.cache_scope == CacheScope::StreamL2 && config.memory_mode != MemoryMode::L1L2 {
        return Err(RunError::new(
            "arguments",
            "--cache-scope stream-l2 requires --mode l1-l2",
        ));
    }
    if config.cache_scope == CacheScope::StreamL2 && config.kernel_id.is_some() {
        return Err(RunError::new(
            "arguments",
            "--kernel-id cannot skip launches in --cache-scope stream-l2",
        ));
    }
    if config.memory_service_candidates.is_some() && config.memory_mode != MemoryMode::L1L2 {
        return Err(RunError::new(
            "arguments",
            "--memory-service-candidates requires --mode l1-l2",
        ));
    }
    if config.memory_service_candidates.is_some() && config.cache_scope != CacheScope::Launch {
        return Err(RunError::new(
            "arguments",
            "--memory-service-candidates currently requires --cache-scope launch",
        ));
    }
    if config.memory_admission_candidates.is_some() && config.memory_mode != MemoryMode::L1L2 {
        return Err(RunError::new(
            "arguments",
            "--memory-admission-candidates requires --mode l1-l2",
        ));
    }
    if config.memory_admission_candidates.is_some() && config.cache_scope != CacheScope::Launch {
        return Err(RunError::new(
            "arguments",
            "--memory-admission-candidates currently requires --cache-scope launch",
        ));
    }
    if config.memory_admission_feedback.is_some() && config.memory_mode != MemoryMode::L1L2 {
        return Err(RunError::new(
            "arguments",
            "--memory-admission-feedback requires --mode l1-l2",
        ));
    }
    if config.memory_admission_feedback.is_some() && config.cache_scope != CacheScope::Launch {
        return Err(RunError::new(
            "arguments",
            "--memory-admission-feedback currently requires --cache-scope launch",
        ));
    }
    if config.dirty_writeback_capacity_sectors.is_some() && config.cache_scope != CacheScope::Launch
    {
        return Err(RunError::new(
            "arguments",
            "dirty writeback currently requires --cache-scope launch",
        ));
    }
    if config.coherence_service_cycles.is_some() && config.cache_scope != CacheScope::Launch {
        return Err(RunError::new(
            "arguments",
            "coherence service currently requires --cache-scope launch",
        ));
    }
    if config.l1_resident_writeback && config.cache_scope != CacheScope::Launch {
        return Err(RunError::new(
            "arguments",
            "L1 resident writeback currently requires --cache-scope launch",
        ));
    }
    if native_payload_events
        .as_ref()
        .is_some_and(|events| !events.is_empty())
        && (!matches!(config.memory_mode, MemoryMode::L1L2)
            || config.cache_scope != CacheScope::Launch
            || !config.carded_dram_bandwidth
            || config.dirty_writeback_capacity_sectors.is_none()
            || !config.l1_resident_writeback)
    {
        return Err(RunError::new(
            "arguments",
            "event-time payload admissions require launch-scoped l1-l2 replay, carded DRAM, resident writeback, and a dirty-writeback capacity",
        ));
    }
    let memory_service_candidates =
        load_memory_service_candidates(config.memory_service_candidates.as_ref())?;
    let memory_admission_candidates =
        load_memory_admission_candidates(config.memory_admission_candidates.as_ref())?;
    let memory_admission_feedback =
        load_memory_admission_feedback(config.memory_admission_feedback.as_ref())?;
    let specialized_loader_requested = std::env::var_os("SIM_RUN_RETAIN_RAW_TRACE_PAYLOAD")
        .is_some()
        || std::env::var_os("SIM_RUN_STREAM_TRACE_PAYLOAD").is_some();
    let performance_diagnostics_requested =
        std::env::var_os("SIM_RUN_PERFORMANCE_COUNTERS").is_some();
    if candidate_inputs == 0
        && config.cache_scope == CacheScope::Launch
        && config.kernel_id.is_none()
        && !config.ncu_shaped_metrics
        && !config.fast_frontier_only
        && !config.sm120_alpha
        && !config.latent_envelope
        && config.dirty_writeback_capacity_sectors.is_none()
        && config.coherence_service_cycles.is_none()
        && !config.l1_visibility_protocol
        && !config.l1_resident_writeback
        && ncu_observation.is_none()
        && !specialized_loader_requested
        && !performance_diagnostics_requested
        && config.launch_resources.is_none()
        && config.native_payload_sidecar.is_none()
        && native_payload_events.is_none()
        && predictive_service_profile.is_none()
    {
        return PreparedReplay::prepare(config, CoverageAdmission::ExistingDirectBehavior)?
            .replay_all_report();
    }
    if config.kernel_id.is_some() && config.launch_resources.is_some() {
        return Err(RunError::new(
            "arguments",
            "--kernel-id cannot be combined with --launch-resources because CUPTI correlation identity is not present in the trace contract",
        ));
    }
    let mut external_resources = config
        .launch_resources
        .as_ref()
        .map(|path| resources::ResourceObservations::load(path))
        .transpose()?;
    let (issue, architecture_spec_issue_ports) = resolve_issue_with_metadata(config, &arch_card)?;
    let mode = resolve_mode(config, &arch_card, issue)?;

    #[cfg(feature = "performance-counters")]
    let setup_ns = total_started.elapsed().as_nanos();
    #[cfg(feature = "performance-counters")]
    let trace_started = Instant::now();
    #[cfg(feature = "performance-counters")]
    let retain_raw_trace_payload = std::env::var_os("SIM_RUN_RETAIN_RAW_TRACE_PAYLOAD").is_some();
    #[cfg(feature = "performance-counters")]
    let stream_trace_payload = std::env::var_os("SIM_RUN_STREAM_TRACE_PAYLOAD").is_some();
    #[cfg(not(feature = "performance-counters"))]
    let retain_raw_trace_payload = false;
    #[cfg(not(feature = "performance-counters"))]
    let stream_trace_payload = false;
    let trace = if retain_raw_trace_payload {
        TraceDir::open_direct(&config.trace_dir)
    } else if stream_trace_payload {
        TraceDir::open_direct_streaming(&config.trace_dir)
    } else {
        TraceDir::open_direct_compact(&config.trace_dir)
    }
    .map_err(at("trace"))?;
    #[cfg(feature = "performance-counters")]
    let trace_ns = trace_started.elapsed().as_nanos();
    #[cfg(feature = "performance-counters")]
    let exec_started = Instant::now();
    let exec = trace.exec_info().ok_or_else(|| {
        RunError::new(
            "execution info",
            "direct trace loader did not retain execution metadata",
        )
    })?;
    #[cfg(feature = "performance-counters")]
    let exec_ns = exec_started.elapsed().as_nanos();
    #[cfg(feature = "performance-counters")]
    let control_started = Instant::now();
    let control = load_control_sidecar(&config.control_bits, arch, config.sm120_alpha)?;
    validate_direct_control_identity(&config.control_bits, exec, &control)?;
    #[cfg(feature = "performance-counters")]
    let control_ns = control_started.elapsed().as_nanos();
    #[cfg(feature = "performance-counters")]
    let schedule_started = Instant::now();
    let schedules = if config.sm120_alpha {
        sim_sm::build_schedules_sm120_alpha(exec, &control, alpha_config())
            .map_err(at("SM120 alpha schedule"))?
    } else {
        build_schedules(exec, &control).map_err(at("schedule"))?
    };
    #[cfg(feature = "performance-counters")]
    let schedule_ns = schedule_started.elapsed().as_nanos();

    let kernel_locs: Vec<_> = trace
        .kernels()
        .filter(|loc| config.kernel_id.is_none_or(|id| loc.kernel().id == id))
        .collect();
    if kernel_locs.is_empty() {
        return Err(RunError::new(
            "replay",
            match config.kernel_id {
                Some(id) => format!("kernel id {id} was not found"),
                None => "trace contains no kernel launches".to_owned(),
            },
        ));
    }
    if let Some(events) = native_payload_events.as_deref() {
        let launch_ids = kernel_locs
            .iter()
            .filter_map(|loc| u64::try_from(loc.kernel().id).ok())
            .collect::<BTreeSet<_>>();
        for (index, event) in events.iter().enumerate() {
            let Some(identity) = event.identity else {
                return Err(RunError::new(
                    "payload events",
                    format!("event {index} has no retained dynamic identity"),
                ));
            };
            if !launch_ids.contains(&identity.launch_id) {
                return Err(RunError::new(
                    "payload events",
                    format!(
                        "event {index} launch_id {} is absent from the selected trace launches",
                        identity.launch_id
                    ),
                ));
            }
        }
    }

    // A complete v2 sidecar is joined against the immutable trace snapshot
    // before replay. Historical/incomplete sidecars remain on the explicit
    // sidecar-only ledger path and never mutate native cache state.
    let payload_ledger = match (
        native_payload_records.as_deref(),
        native_payloads.as_deref(),
        identity_complete_records,
    ) {
        (Some(records), Some(payloads), Some(identity_complete)) => {
            let needs_trace_keys = identity_complete == records.len()
                || records.iter().all(|record| record.trace_matchable());
            let mut observed = BTreeSet::new();
            if needs_trace_keys {
                for loc in &kernel_locs {
                    observed.extend(
                        trace
                            .dynamic_memory_join_identities(loc)
                            .map_err(at("payload trace identity"))?,
                    );
                }
            }
            let validate_function_names = |records: &[NativePayloadRecordV1]| {
                for (index, record) in records.iter().enumerate() {
                    let Some(function_unique_id) = record.identity().function_unique_id else {
                        continue;
                    };
                    let Some(function_name) = record.identity().function_name.as_deref() else {
                        continue;
                    };
                    let Some(static_info) = exec.kernels.get(&function_unique_id) else {
                        return Err(RunError::new(
                            "payload identity join",
                            format!(
                                "payload record {index} function_unique_id {function_unique_id} is absent from exec-info"
                            ),
                        ));
                    };
                    if static_info.function_name != function_name {
                        return Err(RunError::new(
                            "payload identity join",
                            format!(
                                "payload record {index} function_name {function_name:?} disagrees with exec-info {:?}",
                                static_info.function_name
                            ),
                        ));
                    }
                }
                Ok::<(), RunError>(())
            };
            let (joined_payloads, join_status, identity_count, event_binding, event_count) =
                if identity_complete == records.len() {
                    validate_function_names(records)?;
                    (
                        join_native_payload_records(records, &observed)
                            .map_err(at("payload identity join"))?,
                        "matched_trace_identities",
                        records.len(),
                        "trace_locations_verified",
                        records.len(),
                    )
                } else if records.iter().all(|record| record.trace_matchable()) {
                    let enriched = enrich_native_payload_records(records, &observed)
                        .map_err(at("payload identity enrichment"))?;
                    validate_function_names(&enriched)?;
                    (
                        join_native_payload_records(&enriched, &observed)
                            .map_err(at("payload identity join"))?,
                        "enriched_matched_trace_identities",
                        enriched.len(),
                        "trace_locations_verified",
                        enriched.len(),
                    )
                } else {
                    (
                        payloads.to_vec(),
                        "not_attempted_incomplete_identity",
                        identity_complete,
                        "not_attempted_incomplete_identity",
                        0,
                    )
                };
            Some(build_payload_ledger(
                &joined_payloads,
                identity_count,
                join_status,
                event_binding,
                event_count,
            )?)
        }
        _ => None,
    };

    if config.fast_frontier_only {
        let fast_replay_started = Instant::now();
        let mut fast_frontier = Vec::with_capacity(kernel_locs.len());
        for loc in kernel_locs {
            let schedule = schedules
                .get(&loc.kernel().function_unique_id)
                .ok_or_else(|| {
                    RunError::new(
                        "core-only diagnostic",
                        format!(
                            "no schedule for function_unique_id {}",
                            loc.kernel().function_unique_id
                        ),
                    )
                })?;
            let summary = if config.sm120_alpha {
                sim_sm::sm120_fast_stage_for_trace(
                    &trace,
                    &loc,
                    schedule,
                    &arch_card,
                    alpha_config(),
                )
            } else {
                sim_sm::fast_frontier_for_trace(&trace, &loc, schedule, &arch_card)
            }
            .map_err(at("core-only diagnostic"))?;
            fast_frontier.push(FastFrontierLaunchReport {
                device_id: loc.device_id(),
                stream_id: loc.stream_id(),
                kernel_id: loc.kernel().id,
                function_unique_id: loc.kernel().function_unique_id,
                kernel_name: loc.kernel().name.clone(),
                grid: kernel_grid(loc.kernel()).map_err(at("fast launch geometry"))?,
                block: loc.kernel().block_dim.as_ref().map(|dim| {
                    // TraceDir already requires positive launch dimensions.
                    (dim.x as u32, dim.y as u32, dim.z as u32)
                }),
                summary,
            });
        }
        let fast_replay_wall_ns =
            u64::try_from(fast_replay_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let fast_total_wall_ns =
            u64::try_from(total_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        return Ok(RunReport {
            schema_version: 1,
            simulator_version: env!("CARGO_PKG_VERSION"),
            architecture: arch.token().to_owned(),
            memory_mode: config.memory_mode.token(),
            cache_scope: config.cache_scope.token(),
            carded_dram_bandwidth: config.carded_dram_bandwidth,
            native_payload_sidecar_records,
            native_payload_event_count,
            payload_ledger: None,
            dirty_writeback_capacity_sectors: config.dirty_writeback_capacity_sectors,
            coherence_service_cycles: config.coherence_service_cycles,
            coherence_capacity_sectors: config.coherence_capacity_sectors,
            l1_visibility_protocol: config.l1_visibility_protocol,
            l1_resident_writeback: config.l1_resident_writeback,
            fast_frontier_only: true,
            sm120_alpha: config.sm120_alpha,
            sm120_alpha_config: config.sm120_alpha.then(alpha_config),
            sm120_alpha_profile_id: config.sm120_alpha.then_some(Sm120AlphaConfig::PROFILE_ID),
            sm120_effective_l1: config.sm120_effective_l1,
            sm120_effective_l2: config.sm120_effective_l2,
            hypothetical_issue_ports: config
                .hypothetical_issue_ports
                .or(architecture_spec_issue_ports),
            architecture_spec: config
                .architecture_spec
                .as_ref()
                .map(|path| path.display().to_string()),
            memory_service_candidate_count: 0,
            memory_admission_candidate_count: 0,
            memory_admission_feedback_candidate_id: None,
            launches: Vec::new(),
            fast_frontier: Some(fast_frontier),
            cache_session: None,
            ncu_observation,
            latent_envelope: None,
            sim_total_wall_ns: Some(fast_total_wall_ns),
            sim_replay_wall_ns: Some(fast_replay_wall_ns),
        });
    }

    let mut launches = Vec::with_capacity(kernel_locs.len());
    let replay_started = Instant::now();
    let cache_session = match config.cache_scope {
        CacheScope::Launch => {
            for loc in kernel_locs {
                let launch_payload_events = native_payload_events
                    .as_deref()
                    .map(|events| {
                        let launch_id = u64::try_from(loc.kernel().id).unwrap_or_default();
                        events
                            .iter()
                            .filter(|event| {
                                event
                                    .identity
                                    .is_some_and(|identity| identity.launch_id == launch_id)
                            })
                            .cloned()
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                let outcome = if let Some(candidate) = memory_admission_feedback {
                    replay_trace_kernel_with_memory_admission_feedback(
                        &trace, &loc, &schedules, &arch_card, &issue, mode, candidate,
                    )
                } else if let Some(candidates) = memory_admission_candidates.as_deref() {
                    replay_trace_kernel_with_memory_admission_candidates(
                        &trace, &loc, &schedules, &arch_card, &issue, mode, candidates,
                    )
                } else if let Some(candidates) = memory_service_candidates.as_deref() {
                    replay_trace_kernel_with_memory_service_candidates(
                        &trace, &loc, &schedules, &arch_card, &issue, mode, candidates,
                    )
                } else if !launch_payload_events.is_empty() {
                    replay_trace_kernel_with_l1_payload_events(
                        &trace,
                        &loc,
                        &schedules,
                        &arch_card,
                        &issue,
                        mode,
                        &launch_payload_events,
                    )
                } else {
                    replay_trace_kernel(&trace, &loc, &schedules, &arch_card, &issue, mode)
                }
                .map_err(at("replay"))?;
                let result = outcome.into_full().ok_or_else(|| {
                    RunError::new(
                        "replay",
                        "direct timing replay requires complete launch coverage",
                    )
                })?;
                let external = external_resources
                    .as_mut()
                    .map(|records| {
                        records.take(
                            loc.device_id(),
                            loc.stream_id(),
                            &loc.kernel().name,
                            kernel_grid(loc.kernel()).map_err(at("launch resource identity"))?,
                            loc.kernel()
                                .block_dim
                                .as_ref()
                                .map(|dim| (dim.x as u32, dim.y as u32, dim.z as u32)),
                        )
                    })
                    .transpose()?;
                launches.push(launch_report(
                    result,
                    &arch_card,
                    external.as_ref(),
                    None,
                    memory_admission_feedback,
                    config.ncu_shaped_metrics,
                )?);
            }
            None
        }
        CacheScope::StreamL2 => {
            let first = kernel_locs[0];
            if kernel_locs.iter().any(|loc| {
                loc.device_id() != first.device_id() || loc.stream_id() != first.stream_id()
            }) {
                return Err(RunError::new(
                    "arguments",
                    "--cache-scope stream-l2 requires every launch in the trace to belong to one device and stream",
                ));
            }
            let mut session = StreamL2ReplaySession::new(
                &trace,
                &arch_card,
                &issue,
                mode,
                first.device_id(),
                first.stream_id(),
            )
            .map_err(at("replay session"))?;
            let mut seeded_by_kernel_ids = Vec::new();
            for (sequence_index, loc) in kernel_locs.into_iter().enumerate() {
                let outcome = session
                    .replay_next(&trace, &loc, &schedules, &arch_card, &issue, mode)
                    .map_err(at("replay session"))?;
                let result = outcome.into_full().ok_or_else(|| {
                    RunError::new(
                        "replay",
                        "direct timing replay requires complete launch coverage",
                    )
                })?;
                let external = external_resources
                    .as_mut()
                    .map(|records| {
                        records.take(
                            loc.device_id(),
                            loc.stream_id(),
                            &loc.kernel().name,
                            kernel_grid(loc.kernel()).map_err(at("launch resource identity"))?,
                            loc.kernel()
                                .block_dim
                                .as_ref()
                                .map(|dim| (dim.x as u32, dim.y as u32, dim.z as u32)),
                        )
                    })
                    .transpose()?;
                launches.push(launch_report(
                    result,
                    &arch_card,
                    external.as_ref(),
                    Some(CacheHandoffReport {
                        scope: config.cache_scope.token(),
                        sequence_index,
                        seeded_by_kernel_ids: seeded_by_kernel_ids.clone(),
                    }),
                    None,
                    config.ncu_shaped_metrics,
                )?);
                seeded_by_kernel_ids.push(loc.kernel().id);
            }
            Some(CacheSessionReport {
                scope: config.cache_scope.token(),
                device_id: session.device_id(),
                stream_id: session.stream_id(),
                completed_launches: session.completed_launches(),
                cumulative_l2: l2_counter_report(&session.cumulative_l2_counters()),
            })
        }
    };
    let replay_ns = replay_started.elapsed().as_nanos();
    let replay_wall_ns = u64::try_from(replay_ns).unwrap_or(u64::MAX);
    let total_wall_ns = u64::try_from(total_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let latent_envelope = if config.latent_envelope {
        Some(build_latent_envelope(
            &launches,
            ncu_observation.is_some(),
            predictive_service_profile.as_ref(),
            config.service_stage_role,
        )?)
    } else {
        None
    };
    let report = RunReport {
        schema_version: 1,
        simulator_version: env!("CARGO_PKG_VERSION"),
        architecture: arch.token().to_owned(),
        memory_mode: config.memory_mode.token(),
        cache_scope: config.cache_scope.token(),
        carded_dram_bandwidth: config.carded_dram_bandwidth,
        native_payload_sidecar_records,
        native_payload_event_count,
        payload_ledger,
        dirty_writeback_capacity_sectors: config.dirty_writeback_capacity_sectors,
        coherence_service_cycles: config.coherence_service_cycles,
        coherence_capacity_sectors: config.coherence_capacity_sectors,
        l1_visibility_protocol: config.l1_visibility_protocol,
        l1_resident_writeback: config.l1_resident_writeback,
        fast_frontier_only: false,
        sm120_alpha: config.sm120_alpha,
        sm120_alpha_config: config.sm120_alpha.then(alpha_config),
        sm120_alpha_profile_id: config.sm120_alpha.then_some(Sm120AlphaConfig::PROFILE_ID),
        sm120_effective_l1: config.sm120_effective_l1,
        sm120_effective_l2: config.sm120_effective_l2,
        hypothetical_issue_ports: config
            .hypothetical_issue_ports
            .or(architecture_spec_issue_ports),
        architecture_spec: config
            .architecture_spec
            .as_ref()
            .map(|path| path.display().to_string()),
        memory_service_candidate_count: memory_service_candidates.as_ref().map_or(0, Vec::len),
        memory_admission_candidate_count: memory_admission_candidates.as_ref().map_or(0, Vec::len),
        memory_admission_feedback_candidate_id: memory_admission_feedback
            .map(|candidate| candidate.candidate_id),
        launches,
        fast_frontier: None,
        cache_session,
        ncu_observation,
        latent_envelope,
        sim_total_wall_ns: Some(total_wall_ns),
        sim_replay_wall_ns: Some(replay_wall_ns),
    };
    if let Some(resources) = external_resources.as_ref() {
        resources.finish()?;
    }
    #[cfg(feature = "performance-counters")]
    if performance_enabled {
        eprintln!(
            "SIM_PHASE_PERF setup_ns={setup_ns} trace_ns={trace_ns} exec_ns={exec_ns} \
             control_ns={control_ns} schedule_ns={schedule_ns} replay_ns={replay_ns} \
             run_total_ns={}",
            total_started.elapsed().as_nanos(),
        );
    }
    Ok(report)
}
