// Lexical implementation section 1/2; included by the parent facade.
pub use prepared::{
    CoverageAdmission, KernelKey, KernelSelector, PreparedLaunchIdentity, PreparedReplay,
};

use std::collections::{BTreeMap, BTreeSet};

use std::path::{Path, PathBuf};

use std::str::FromStr;

use std::time::Instant;

use crate::emit::{
    EmitOutcome, EmittedMetricKey, FullLaunchPredictions, MetricValue, Unavailable, Unit,
    async_global_l1_hit_exclusion, emit_launch, global_read_resident_hit_adjustment,
    global_resident_hit_adjustment, l1_write_hit_terms,
};

use serde::{Deserialize, Serialize};

use sim_arch::ArchId;

use sim_arch::card::{ArchCard, CalibError};

use sim_arch::spec::{ArchitectureSpec, ResourceKind};

use sim_mem::{
    CoherenceConfig, L1Config, L1Counters, L1SetIndex, L2_WRITE_ALLOCATE_POLICY, L2Config,
    L2Counters, L2Replacement, L2Source, MemoryOrigin, NativePayloadRecordV1,
    PayloadForwardingLedgerV1, PchaseCard, SectorPayloadV1, SliceHash, coalesce_payloads,
    enrich_native_payload_records, join_native_payload_records,
    load_native_payload_sidecar_records,
};

use sim_semantics::{
    CacheCandidateSummary, CacheWitnessDomains, ClockDomain, CompletionRequirement,
    EndpointDescriptor, FrontierBound, FrontierVector, LatentFactors, LatentModelError,
    NcuObserverProjection, ResourceServiceProjection, ServiceComposition, ServiceOverlapMode,
    ServiceQueueProjection, ServiceRateProfile, ServiceStageRole,
};

use sim_sm::{
    ControlSidecar, DramBw, DramStats, FastFrontierSummary, FullKernelReplayResult,
    GpcObservationFrontier, IssueModelParams, L1PayloadAdmissionEvent, L2SourceSplit,
    MAX_MEMORY_ADMISSION_CANDIDATES, MAX_MEMORY_SERVICE_CANDIDATES, MemoryAdmissionCandidate,
    MemoryServiceCandidate, ModeledCompletionFrontier, ReadinessDiagnostics, ReplayMemMode,
    Sm120AlphaConfig, SmActiveObservationFrontier, StreamL2ReplaySession, TailFlags, TraceMetrics,
    build_schedules, replay_trace_kernel, replay_trace_kernel_with_l1_payload_events,
    replay_trace_kernel_with_memory_admission_candidates,
    replay_trace_kernel_with_memory_admission_feedback,
    replay_trace_kernel_with_memory_service_candidates,
};

use sim_trace::{AnchoredDir, ExecInfo, TraceDir, kernel_grid, strict_json_from_slice};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryMode {
    Flat,
    L1,
    L1L2,
}

impl MemoryMode {
    pub const fn token(self) -> &'static str {
        match self {
            Self::Flat => "flat",
            Self::L1 => "l1",
            Self::L1L2 => "l1-l2",
        }
    }
}

impl FromStr for MemoryMode {
    type Err = RunError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "flat" => Ok(Self::Flat),
            "l1" => Ok(Self::L1),
            "l1-l2" => Ok(Self::L1L2),
            _ => Err(RunError::new(
                "arguments",
                format!("unknown memory mode {value:?}; use flat, l1, or l1-l2"),
            )),
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

impl Default for Sm120EffectiveL1Config {
    fn default() -> Self {
        Self {
            line_capacity: 1024,
            mshr_count: None,
            hit_latency_cycles: 50,
            miss_latency_cycles: 512,
        }
    }
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

impl CacheScope {
    pub const fn token(self) -> &'static str {
        match self {
            Self::Launch => "launch",
            Self::StreamL2 => "stream-l2",
        }
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

#[derive(Debug, Clone)]
pub struct RunConfig {
    pub trace_dir: PathBuf,
    pub control_bits: PathBuf,
    pub arch_card: PathBuf,
    pub pchase_card: Option<PathBuf>,
    pub memory_mode: MemoryMode,
    pub cache_scope: CacheScope,
    pub carded_dram_bandwidth: bool,
    /// Optional explicit effective DRAM service-rate scenario. This is a
    /// hypothesis input, not a physical bandwidth claim; when absent the
    /// architecture-card vendor peak remains active.
    pub dram_service_rate_bytes_per_sec: Option<u64>,
    /// Optional opt-in dirty-L2-eviction writeback queue capacity in 32-byte
    /// sectors. This requires carded DRAM, launch-scoped L1-L2 replay, and
    /// never changes the default read-only carded path.
    pub dirty_writeback_capacity_sectors: Option<u32>,
    /// Optional bounded directory service latency. Requires launch-scoped
    /// L1-L2 replay; this is not a measured SM86 coherence protocol latency.
    pub coherence_service_cycles: Option<u64>,
    /// Optional bounded directory in-flight sector capacity.
    pub coherence_capacity_sectors: Option<u32>,
    /// Select the explicit typed L1 observation/invalidation event path. This
    /// requires the bounded coherence configuration and is never inferred from
    /// process-global environment state.
    pub l1_visibility_protocol: bool,
    /// Select resident-hit L1 dirty tracking for the explicit terminal
    /// writeback experiment. This requires carded DRAM, a dirty-writeback
    /// queue, and the bounded coherence configuration.
    pub l1_resident_writeback: bool,
    /// Optional model-relative issue transaction width. This never changes
    /// the evidence-bound architecture card; it labels the replay as a
    /// hypothetical resource experiment.
    pub hypothetical_issue_ports: Option<u32>,
    /// Optional trace-safe architecture resource profile. The current
    /// production bridge consumes only its `alu`/`memory` issue-width pair;
    /// it never changes the traced SASS semantics or vendor card identity.
    pub architecture_spec: Option<PathBuf>,
    pub kernel_id: Option<i32>,
    /// Include the registry-bound simulator projections used for NCU joins.
    /// Always enabled for SM120 full replay; optional for compatibility targets.
    pub ncu_shaped_metrics: bool,
    /// Run only the conservative trace-derived core-only analyzer. This skips
    /// native timing replay and emits no silicon timing claim. The historical
    /// `fast_frontier_only` field name is retained for schema compatibility.
    pub fast_frontier_only: bool,
    /// Optional schema-1 bounded candidate document. Candidate evaluation is
    /// observational and available only for launch-scoped full-hierarchy replay.
    pub memory_service_candidates: Option<PathBuf>,
    /// Optional schema-1 finite operand-admission candidate document.
    pub memory_admission_candidates: Option<PathBuf>,
    /// Optional schema-1 document containing exactly one timing-active finite
    /// admission candidate.
    pub memory_admission_feedback: Option<PathBuf>,
    /// Optional observation-only raw NCU CSV. It never changes replay inputs.
    pub ncu_csv: Option<PathBuf>,
    pub ncu_launch: Option<u64>,
    /// Optional CUPTI launch-resource JSONL observation. This is metadata only;
    /// it never changes replay scheduling or cache configuration.
    pub launch_resources: Option<PathBuf>,
    /// Optional strict value-bearing payload sidecar. Admission and the
    /// sidecar-only logical forwarding ledger are explicit; native replay
    /// remains mask-only and does not consume payload bytes.
    pub native_payload_sidecar: Option<PathBuf>,
    /// Optional producer-supplied event-time payload admissions. Each event
    /// must carry a simulator-cycle timestamp, residency witness, and a
    /// retained dynamic trace identity. The ordinary CLI does not synthesize
    /// this vector from a sidecar.
    pub native_payload_events: Option<Vec<L1PayloadAdmissionEvent>>,
    /// Optional strict JSON event document (`native_l1_payload_events_v1`).
    /// This is mutually exclusive with the programmatic event vector.
    pub native_payload_events_path: Option<PathBuf>,
    /// Emit the opt-in factorized route/class/timestamp envelope. This is a
    /// diagnostic contract and does not alter replay scheduling or scoring.
    pub latent_envelope: bool,
    /// Optional abstract service-rate profile used only by the opt-in
    /// predictive envelope. It never changes native replay timing.
    pub predictive_service_profile: Option<PathBuf>,
    /// Explicit model-local composition role for the service profile. The
    /// default is an independent sidecar; residual/replacement roles are
    /// experimental and must remain declared in the output.
    pub service_stage_role: ServiceStageRole,
    /// Effective timing lane for retained SM120/RTX5090 traces. Selected
    /// automatically from the architecture card, separate from SM86 resources.
    pub sm120_alpha: bool,
    /// Optional override of the default SM120 effective L1 envelope.
    pub sm120_effective_l1: Option<Sm120EffectiveL1Config>,
    /// Optional override of the default SM120 effective L2 scenario. This is
    /// a conditional assumption, not a recovered physical geometry card.
    pub sm120_effective_l2: Option<Sm120EffectiveL2Config>,
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
    fn with_architecture_defaults(&self, arch: sim_arch::ArchId) -> Result<Self, RunError> {
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
struct MemoryServiceCandidateDocument {
    schema_version: u32,
    candidates: Vec<MemoryServiceCandidate>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MemoryAdmissionCandidateDocument {
    schema_version: u32,
    candidates: Vec<MemoryAdmissionCandidate>,
}

const MAX_CANDIDATE_FILE_BYTES: u64 = 64 * 1024;

const MAX_PREDICTIVE_PROFILE_BYTES: u64 = 64 * 1024;

const MAX_PAYLOAD_EVENT_FILE_BYTES: u64 = 64 * 1024 * 1024;

const MAX_PAYLOAD_EVENTS: usize = 1_000_000;

const MAX_PAYLOADS_PER_EVENT: usize = 4096;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PayloadEventDocument {
    schema: String,
    status: String,
    events: Vec<PayloadEventRaw>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PayloadEventRaw {
    cycle: u64,
    sm_id: i32,
    identity: PayloadEventIdentityRaw,
    witnesses: Vec<PayloadEventWitnessRaw>,
    payloads: Vec<PayloadEventSectorRaw>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PayloadEventIdentityRaw {
    launch_id: u64,
    function_unique_id: i32,
    cta: [u32; 3],
    warp: u32,
    lane: u32,
    pc: u32,
    occurrence: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PayloadEventWitnessRaw {
    line_id: i64,
    allocation_id: u64,
    allocation_generation: u64,
    resident_generation: u32,
    resident_sector_mask: u8,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PayloadEventSectorRaw {
    allocation_id: u64,
    generation: u64,
    line_id: i64,
    sector: u8,
    byte_offset: u8,
    bytes_hex: String,
    known_mask: u32,
    sequence: u64,
}

#[derive(Debug, thiserror::Error)]
#[error("{stage}: {detail}")]
pub struct RunError {
    stage: &'static str,
    detail: String,
}

impl RunError {
    pub fn new(stage: &'static str, detail: impl Into<String>) -> Self {
        Self {
            stage,
            detail: detail.into(),
        }
    }

    pub const fn stage(&self) -> &'static str {
        self.stage
    }

    pub fn detail(&self) -> &str {
        &self.detail
    }
}

fn at<E: std::fmt::Display>(stage: &'static str) -> impl FnOnce(E) -> RunError {
    move |error| RunError::new(stage, error.to_string())
}

const MAX_DIRECT_CARD_BYTES: u64 = 16 * 1024 * 1024;

const MAX_DIRECT_CONTROL_BYTES: u64 = 64 * 1024 * 1024;

/// Read one direct-replay input from a no-follow anchored parent and retain the
/// exact generation that was parsed. Direct mode has no campaign manifest, so
/// this is the boundary that prevents a concurrent path replacement from
/// silently changing cards or control metadata during a run.
fn read_direct_input(
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

fn decode_event_bytes(hex: &str, stage: &'static str) -> Result<Vec<u8>, RunError> {
    if hex.is_empty() || hex.len() % 2 != 0 || hex.len() > 64 {
        return Err(RunError::new(
            stage,
            "event payload bytes_hex must contain 1..=32 complete bytes",
        ));
    }
    let mut bytes = Vec::with_capacity(hex.len() / 2);
    for pair in hex.as_bytes().chunks_exact(2) {
        let high = (pair[0] as char)
            .to_digit(16)
            .ok_or_else(|| RunError::new(stage, "event payload bytes_hex is not hexadecimal"))?;
        let low = (pair[1] as char)
            .to_digit(16)
            .ok_or_else(|| RunError::new(stage, "event payload bytes_hex is not hexadecimal"))?;
        bytes.push(((high << 4) | low) as u8);
    }
    Ok(bytes)
}

fn load_native_payload_events(path: &Path) -> Result<Vec<L1PayloadAdmissionEvent>, RunError> {
    let bytes = read_direct_input(
        path,
        "native payload events",
        MAX_PAYLOAD_EVENT_FILE_BYTES + 1,
    )?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_PAYLOAD_EVENT_FILE_BYTES {
        return Err(RunError::new(
            "native payload events",
            "event document exceeds the 64 MiB input limit",
        ));
    }
    let document: PayloadEventDocument =
        strict_json_from_slice(&bytes).map_err(at("native payload events"))?;
    if document.schema != "native_l1_payload_events_v1" || document.status != "ok" {
        return Err(RunError::new(
            "native payload events",
            "event document schema/status is invalid",
        ));
    }
    if document.events.is_empty() {
        return Err(RunError::new(
            "native payload events",
            "event document must contain at least one event",
        ));
    }
    if document.events.len() > MAX_PAYLOAD_EVENTS {
        return Err(RunError::new(
            "native payload events",
            format!("event count exceeds {MAX_PAYLOAD_EVENTS}"),
        ));
    }
    let mut seen_event_keys = BTreeSet::new();
    document
        .events
        .into_iter()
        .enumerate()
        .map(|(event_index, event)| {
            if event.sm_id < 0 {
                return Err(RunError::new(
                    "native payload events",
                    format!("event {event_index} has negative sm_id"),
                ));
            }
            if event.identity.launch_id == 0 || event.identity.function_unique_id <= 0 {
                return Err(RunError::new(
                    "native payload events",
                    format!("event {event_index} has invalid launch/function identity"),
                ));
            }
            if event.identity.warp >= 32 || event.identity.lane >= 32 {
                return Err(RunError::new(
                    "native payload events",
                    format!("event {event_index} has an out-of-range warp/lane"),
                ));
            }
            if event.payloads.is_empty() || event.witnesses.is_empty() {
                return Err(RunError::new(
                    "native payload events",
                    format!("event {event_index} must carry payloads and witnesses"),
                ));
            }
            if event.payloads.len() > MAX_PAYLOADS_PER_EVENT {
                return Err(RunError::new(
                    "native payload events",
                    format!("event {event_index} exceeds {MAX_PAYLOADS_PER_EVENT} payloads"),
                ));
            }
            let identity = sim_mem::NativePayloadJoinKey {
                launch_id: event.identity.launch_id,
                function_unique_id: event.identity.function_unique_id,
                cta: event.identity.cta,
                warp: event.identity.warp,
                lane: event.identity.lane,
                pc: event.identity.pc,
                occurrence: event.identity.occurrence,
            };
            if !seen_event_keys.insert((identity, event.cycle)) {
                return Err(RunError::new(
                    "native payload events",
                    format!("event {event_index} duplicates an identity at the same cycle"),
                ));
            }
            let witnesses = event
                .witnesses
                .into_iter()
                .map(|witness| {
                    if witness.allocation_id == 0
                        || witness.allocation_generation == 0
                        || witness.resident_sector_mask == 0
                        || witness.resident_sector_mask & !0b1111 != 0
                    {
                        return Err(RunError::new(
                            "native payload events",
                            format!("event {event_index} has an invalid residency witness"),
                        ));
                    }
                    Ok(sim_mem::L1PayloadResidencyWitness {
                        line_id: witness.line_id,
                        allocation_id: witness.allocation_id,
                        allocation_generation: witness.allocation_generation,
                        resident_generation: witness.resident_generation,
                        resident_sector_mask: witness.resident_sector_mask,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let payloads = event
                .payloads
                .into_iter()
                .map(|payload| {
                    if payload.generation == 0
                        || payload.allocation_id == 0
                        || payload.sequence == 0
                        || payload.byte_offset >= 32
                        || payload.known_mask == 0
                    {
                        return Err(RunError::new(
                            "native payload events",
                            format!("event {event_index} has invalid payload identity"),
                        ));
                    }
                    let bytes = decode_event_bytes(&payload.bytes_hex, "native payload events")?;
                    let end = usize::from(payload.byte_offset)
                        .checked_add(bytes.len())
                        .ok_or_else(|| {
                            RunError::new("native payload events", "event payload range overflow")
                        })?;
                    if end > sim_mem::PAYLOAD_SECTOR_BYTES {
                        return Err(RunError::new(
                            "native payload events",
                            format!("event {event_index} payload range exceeds one sector"),
                        ));
                    }
                    let width_mask = if bytes.len() == 32 {
                        u32::MAX
                    } else {
                        ((1_u32 << bytes.len()) - 1) << payload.byte_offset
                    };
                    if payload.known_mask != width_mask {
                        return Err(RunError::new(
                            "native payload events",
                            format!("event {event_index} has an inconsistent payload mask"),
                        ));
                    }
                    let mut sector_bytes = [0_u8; sim_mem::PAYLOAD_SECTOR_BYTES];
                    sector_bytes[usize::from(payload.byte_offset)..end].copy_from_slice(&bytes);
                    SectorPayloadV1::new_with_generation(
                        payload.allocation_id,
                        payload.generation,
                        payload.line_id,
                        payload.sector,
                        payload.sequence,
                        sector_bytes,
                        payload.known_mask,
                    )
                    .map_err(at("native payload events"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            for payload in &payloads {
                if !witnesses.iter().any(|witness| {
                    witness.line_id == payload.line_id()
                        && witness.allocation_id == payload.allocation_id()
                        && witness.allocation_generation == payload.allocation_generation()
                }) {
                    return Err(RunError::new(
                        "native payload events",
                        format!(
                            "event {event_index} payload line {} has no matching allocation witness",
                            payload.line_id()
                        ),
                    ));
                }
            }
            Ok(L1PayloadAdmissionEvent {
                sm_id: event.sm_id,
                time: event.cycle,
                identity: Some(identity),
                writer: None,
                payloads: payloads.into_boxed_slice(),
                witnesses: witnesses.into_boxed_slice(),
            })
        })
        .collect()
}

pub(crate) fn load_arch_card(path: &Path) -> Result<ArchCard, RunError> {
    let raw = read_direct_input(path, "architecture card", MAX_DIRECT_CARD_BYTES)?;
    ArchCard::parse(path, &raw).map_err(at("architecture card"))
}

fn load_predictive_service_profile(
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
fn resolve_issue_with_metadata(
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
fn trace_safe_issue_ports(path: &Path, arch: sim_arch::ArchId) -> Result<u32, RunError> {
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

pub(crate) fn load_pchase_card(path: &Path) -> Result<PchaseCard, RunError> {
    let raw = read_direct_input(path, "P-chase card", MAX_DIRECT_CARD_BYTES)?;
    PchaseCard::parse(path, &raw).map_err(at("P-chase card"))
}

fn load_ncu_observation(
    path: &Path,
    launch_id: Option<u64>,
) -> Result<ncu::NcuObservation, RunError> {
    let raw = read_direct_input(path, "NCU CSV", 64 * 1024 * 1024)?;
    let text =
        std::str::from_utf8(&raw).map_err(|_| RunError::new("NCU CSV", "input is not UTF-8"))?;
    ncu::NcuObservation::from_csv(text, launch_id)
        .map_err(|error| RunError::new("NCU CSV", error.to_string()))
}

pub(crate) fn load_control_sidecar(
    path: &Path,
    arch: sim_arch::ArchId,
    sm120_alpha: bool,
) -> Result<ControlSidecar, RunError> {
    let raw = read_direct_input(path, "control bits", MAX_DIRECT_CONTROL_BYTES)?;
    ControlSidecar::parse_for_arch_with_alpha(path, &raw, arch, sm120_alpha)
        .map_err(at("control bits"))
}

/// Bind a canonical direct-replay sidecar to the executable named by the
/// trace. Research sidecars may intentionally use a transformed filename
/// (for example a holdout shape) and retain their existing compatibility; a
/// canonical `<benchmark>.control_bits.json` has no such disambiguator, so a
/// different recorded binary basename is an unsafe join.
pub(crate) fn validate_direct_control_identity(
    control_path: &Path,
    exec: &ExecInfo,
    control: &ControlSidecar,
) -> Result<(), RunError> {
    if exec.benchmark_name.is_empty() {
        return Err(RunError::new(
            "identity",
            "trace execution info has an empty benchmark_name",
        ));
    }
    let expected_stem = exec
        .benchmark_name
        .strip_suffix("_run")
        .unwrap_or(&exec.benchmark_name);
    let expected_filename = format!("{expected_stem}.control_bits.json");
    let is_canonical_filename =
        control_path.file_name().and_then(|name| name.to_str()) == Some(expected_filename.as_str());
    if !is_canonical_filename {
        return Ok(());
    }

    let actual_binary = Path::new(&control.binary)
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .ok_or_else(|| {
            RunError::new(
                "identity",
                format!("canonical control sidecar {expected_filename:?} has no binary basename"),
            )
        })?;
    if actual_binary != exec.benchmark_name && actual_binary != expected_stem {
        return Err(RunError::new(
            "identity",
            format!(
                "canonical control sidecar binary {actual_binary:?} does not match trace benchmark {:?} (accepted {:?} or {:?})",
                exec.benchmark_name, exec.benchmark_name, expected_stem
            ),
        ));
    }
    Ok(())
}

fn parse_memory_service_candidates(bytes: &[u8]) -> Result<Vec<MemoryServiceCandidate>, RunError> {
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

fn load_memory_service_candidates(
    path: Option<&PathBuf>,
) -> Result<Option<Vec<MemoryServiceCandidate>>, RunError> {
    path.map(|path| {
        // Candidate documents affect timing decisions just like cards and
        // control metadata. Read them through the same no-follow, bounded
        // generation check so a concurrent replacement cannot silently alter
        // the replay inputs.
        let bytes = read_direct_input(
            path,
            "memory-service candidates",
            MAX_CANDIDATE_FILE_BYTES + 1,
        )?;
        parse_memory_service_candidates(&bytes)
    })
    .transpose()
}

fn parse_memory_admission_candidates(
    bytes: &[u8],
) -> Result<Vec<MemoryAdmissionCandidate>, RunError> {
    parse_memory_admission_document(bytes, "memory-admission candidates", false)
}

fn parse_memory_admission_feedback(bytes: &[u8]) -> Result<MemoryAdmissionCandidate, RunError> {
    let candidates = parse_memory_admission_document(bytes, "memory-admission feedback", true)?;
    Ok(candidates[0])
}

fn parse_memory_admission_document(
    bytes: &[u8],
    stage: &'static str,
    require_exactly_one: bool,
) -> Result<Vec<MemoryAdmissionCandidate>, RunError> {
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_CANDIDATE_FILE_BYTES {
        return Err(RunError::new(
            stage,
            format!(
                "candidate document exceeds {} bytes",
                MAX_CANDIDATE_FILE_BYTES
            ),
        ));
    }
    let document: MemoryAdmissionCandidateDocument =
        strict_json_from_slice(bytes).map_err(at(stage))?;
    if document.schema_version != 1 {
        return Err(RunError::new(
            stage,
            format!(
                "unsupported schema_version {}; expected 1",
                document.schema_version
            ),
        ));
    }
    if require_exactly_one && document.candidates.len() != 1 {
        return Err(RunError::new(
            stage,
            format!(
                "feedback document must contain exactly one candidate, got {}",
                document.candidates.len()
            ),
        ));
    }
    if document.candidates.is_empty() {
        return Err(RunError::new(stage, "candidate list must not be empty"));
    }
    if document.candidates.len() > MAX_MEMORY_ADMISSION_CANDIDATES {
        return Err(RunError::new(
            stage,
            format!(
                "candidate list has {} entries; maximum is {}",
                document.candidates.len(),
                MAX_MEMORY_ADMISSION_CANDIDATES
            ),
        ));
    }
    for (index, candidate) in document.candidates.iter().copied().enumerate() {
        candidate.validate().map_err(at(stage))?;
        if document.candidates[..index]
            .iter()
            .any(|prior| prior.candidate_id == candidate.candidate_id)
        {
            return Err(RunError::new(
                stage,
                format!("candidate id {} is duplicated", candidate.candidate_id),
            ));
        }
    }
    Ok(document.candidates)
}

fn load_memory_admission_candidates(
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

fn load_memory_admission_feedback(
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
pub struct PayloadLedgerReport {
    pub role: &'static str,
    pub status: &'static str,
    pub baseline_status: &'static str,
    pub input_records: usize,
    pub identity_status: &'static str,
    pub identity_complete_records: usize,
    pub identity_join: &'static str,
    pub event_location_binding: &'static str,
    pub event_location_count: usize,
    pub complete_sectors: usize,
    pub logical_l1_commits: usize,
    pub logical_l2_commits: usize,
    pub logical_dram_commits: usize,
    pub hardware_completion: &'static str,
}

fn build_payload_ledger(
    payloads: &[SectorPayloadV1],
    identity_complete_records: usize,
    identity_join: &'static str,
    event_location_binding: &'static str,
    event_location_count: usize,
) -> Result<PayloadLedgerReport, RunError> {
    let complete = coalesce_payloads(payloads.to_vec()).map_err(at("payload ledger"))?;
    let mut logical_l2_commits = 0usize;
    let mut logical_dram_commits = 0usize;
    for payload in &complete {
        let mut ledger = PayloadForwardingLedgerV1::from_admitted_store(*payload)
            .map_err(at("payload ledger"))?;
        ledger.commit_l2(*payload).map_err(at("payload ledger"))?;
        ledger.commit_dram(*payload).map_err(at("payload ledger"))?;
        logical_l2_commits = logical_l2_commits
            .checked_add(1)
            .ok_or_else(|| RunError::new("payload ledger", "L2 commit count overflow"))?;
        logical_dram_commits = logical_dram_commits
            .checked_add(1)
            .ok_or_else(|| RunError::new("payload ledger", "DRAM commit count overflow"))?;
    }
    Ok(PayloadLedgerReport {
        role: "sidecar_only_logical_forwarding_not_native_cache_state",
        status: "complete_logical_forwarding",
        baseline_status: "pre_store_bytes_unknown",
        input_records: payloads.len(),
        identity_status: if identity_complete_records == payloads.len() {
            "complete"
        } else {
            "incomplete"
        },
        identity_complete_records,
        identity_join,
        event_location_binding,
        event_location_count,
        complete_sectors: complete.len(),
        logical_l1_commits: complete.len(),
        logical_l2_commits,
        logical_dram_commits,
        hardware_completion: "not_observed",
    })
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

fn build_latent_envelope(
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
pub struct FastFrontierLaunchReport {
    pub device_id: i64,
    pub stream_id: i64,
    pub kernel_id: i32,
    pub function_unique_id: i32,
    pub kernel_name: String,
    /// Captured launch geometry, retained for post-replay association with
    /// independent profiles. This is not an NCU-derived timing input.
    pub grid: (u32, u32, u32),
    pub block: Option<(u32, u32, u32)>,
    pub summary: FastFrontierSummary,
}

fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(Debug, Serialize)]
pub struct LaunchReport {
    pub device_id: i64,
    pub stream_id: i64,
    pub kernel_id: i32,
    pub function_unique_id: i32,
    pub kernel_name: String,
    pub grid: (u32, u32, u32),
    pub block: Option<(u32, u32, u32)>,
    pub cluster: (u32, u32, u32),
    pub coverage: CoverageReport,
    pub counters: CounterReport,
    /// Trace-derived dynamic activity. This is observation-only and is never
    /// used as an architecture or timing input.
    pub trace_metrics: TraceMetrics,
    /// Model-relative active-warp integrals and per-SMSP issue observations.
    pub sm_activity: BTreeMap<i32, sim_sm::SmActivity>,
    /// Extended native metric projection. Values marked approximate or
    /// unavailable are intentionally not part of the frozen seven-plus-one
    /// publication vector.
    pub native_metrics: metrics::NativeMetrics,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metric_predictions: Option<Vec<NcuMetricProjectionReport>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_lookup_predictions: Option<Vec<CacheLookupPredictionReport>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing: Option<TimingReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_handoff: Option<CacheHandoffReport>,
}

#[derive(Debug, Serialize)]
pub struct NcuMetricProjectionReport {
    pub arch: String,
    pub key: &'static str,
    pub ncu_metric: &'static str,
    pub unit: Unit,
    pub outcome: EmitOutcome,
}

fn ncu_metric_name_for_arch(
    arch: sim_arch::ArchId,
    key: &str,
    default: &'static str,
) -> &'static str {
    // Nsight's SM120 raw profiles expose the DRAM read demand under the
    // operation-qualified label. Keep the internal registry key stable while
    // emitting the architecture-qualified external name for post-replay joins.
    if arch == sim_arch::ArchId::Sm120 && key == "traffic_dram_read_sectors" {
        "dram__sectors_op_read.sum"
    } else {
        default
    }
}

#[derive(Debug, Serialize)]
pub struct CacheLookupPredictionReport {
    pub scope: &'static str,
    pub ncu_hit_metric: &'static str,
    pub ncu_miss_metric: &'static str,
    pub status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observer: Option<CacheLookupObserverReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing_state: Option<CacheLookupTimingStateReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<Unavailable>,
}

#[derive(Debug, Serialize)]
pub struct CacheLookupObserverReport {
    pub hits: u64,
    pub misses: u64,
    pub lookups: u64,
}

#[derive(Debug, Serialize)]
pub struct CacheLookupTimingStateReport {
    pub resident_data_hits: u64,
    /// Physical resident hits represented by the observer numerator.  L2
    /// write packetization can leave some physical hits outside the NCU lookup
    /// stream; retaining this split keeps that fact explicit instead of
    /// forcing an invalid `observer_hits >= resident_hits` invariant.
    pub observer_resident_hits: u64,
    /// Physical cache hits that were not represented in the packetized lookup
    /// observer.  This is diagnostic decomposition only and never changes the
    /// emitted numerator/denominator.
    pub physical_hits_not_observed: u64,
    /// Lookup-observer misses (`lookups - observer hits`).  This is kept in
    /// the observer domain; it is not the complement of physical resident
    /// hits, which can exceed the packetized observer denominator.
    pub remaining_lookups: u64,
    pub observer_reclassified_lookups: u64,
    pub role: &'static str,
}

#[derive(Debug, Serialize)]
pub struct CacheHandoffReport {
    pub scope: &'static str,
    pub sequence_index: usize,
    pub seeded_by_kernel_ids: Vec<i32>,
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
pub struct CoverageReport {
    pub kind: &'static str,
    pub blocks_observed: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub population_blocks: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct TimingReport {
    /// Distinguishes the simulator's internal frontier status from physical
    /// completion semantics.  The SM120 alpha path deliberately reports an
    /// approximate internal frontier when it flattens async/TMA demands.
    pub completion_frontier_semantics: &'static str,
    /// Effective SM120 GPC front-end envelope. This is separate from the raw
    /// SM-active frontier and is not a hardware clock conversion.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effective_gpc_front_end_envelope_cycles: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alpha_approximation: Option<AlphaApproximationReport>,
    pub diagnostic_last_issue_cycles: u64,
    pub last_processed_modeled_system_event_time: Option<u64>,
    pub terminal_modeled_event_time: Option<u64>,
    /// SM-active terminal lower bound with service-acknowledgement events kept
    /// in a separate observation domain.  The complete modeled-event frontier
    /// above remains the authoritative event-queue timestamp.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sm_active_terminal_frontier: Option<SmActiveTerminalFrontierReport>,
    /// Explicit SM-active physical observation frontier. The raw lower bounds
    /// above remain unchanged and are retained inside this projection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sm_active_observation_frontier: Option<SmActiveObservationFrontierReport>,
    /// Architecture-scoped GPC observation frontier. The raw modeled event
    /// lower bound above remains separate so a profiler boundary cannot be
    /// mistaken for physical cache or SM execution time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpc_observation_frontier: Option<GpcObservationFrontierReport>,
    pub completion_frontier: CompletionFrontierReport,
    pub readiness: ReadinessReport,
}

#[derive(Debug, Serialize)]
pub struct Sm120IssueEnvelopeReport {
    pub status: &'static str,
    pub profile_id: &'static str,
    /// Exact values consumed by the flat alpha executor.  These are scenario
    /// inputs, not confidence bounds or identified physical resources.
    pub issue_ports_configured: u32,
    pub dual_issue_configured: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scalar_latency_configured_cycles: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scalar_initiation_configured_cycles: Option<u64>,
    pub max_resident_warps_configured: u32,
    pub max_resident_blocks_configured: u32,
    pub configuration_source: &'static str,
    pub non_claim: &'static str,
}

impl Sm120IssueEnvelopeReport {
    fn from_issue_params(params: IssueModelParams) -> Self {
        let scalar = params.scalar_execution_pipeline();
        Self {
            status: "configured_alpha_inputs",
            profile_id: Sm120AlphaConfig::PROFILE_ID,
            issue_ports_configured: params.issue_ports(),
            dual_issue_configured: params.dual_issue_supported(),
            scalar_latency_configured_cycles: scalar.map(|value| value.latency_cycles),
            scalar_initiation_configured_cycles: scalar
                .map(|value| value.initiation_interval_cycles),
            max_resident_warps_configured: params.max_resident_warps_per_sm(),
            max_resident_blocks_configured: params.max_resident_blocks_per_sm(),
            configuration_source: "IssueModelParams consumed by executor",
            non_claim: "scenario inputs only; not a confidence interval or physical scheduler topology",
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Sm120ServiceEnvelopeReport {
    pub status: &'static str,
    pub source_domains: Vec<&'static str>,
    pub route_policy: &'static str,
    pub residency: &'static str,
    pub outstanding_demand: &'static str,
    pub dram_service: &'static str,
    pub geometry_status: &'static str,
    pub partition_mapping_status: &'static str,
    pub mshr_count: &'static str,
    pub queue_policy: &'static str,
    pub non_claim: &'static str,
}

impl Sm120ServiceEnvelopeReport {
    fn observation_only() -> Self {
        Self {
            status: "logical_route_service_no_geometry_fit",
            source_domains: vec!["trace_memory_coherence", "ncu_raw_l1_l2_dram_sectors"],
            route_policy: "trace_route_and_static_width",
            residency: "single_warp_dense_service_arm_only",
            outstanding_demand: "logical_route_frontier",
            dram_service: "logical_global_service_envelope",
            geometry_status: "typed_gap",
            partition_mapping_status: "typed_gap",
            mshr_count: "typed_gap",
            queue_policy: "per_route_logical_fifo",
            non_claim: "logical service envelope only; no physical cache/DRAM topology or native completion",
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Sm120CompletionLedgerReport {
    pub status: &'static str,
    pub issue: &'static str,
    pub transaction_complete: &'static str,
    pub barrier_arrival: &'static str,
    pub wait_or_test: &'static str,
    pub destination_visibility: &'static str,
    pub payload_observation: &'static str,
    pub affected_async_records: u64,
    pub non_claim: &'static str,
}

impl Sm120CompletionLedgerReport {
    fn provisional(affected_async_records: u64) -> Self {
        Self {
            status: "typed_alpha_completion_ledger",
            issue: "observed_in_trace",
            transaction_complete: "unobserved_sm120",
            barrier_arrival: "model_relative_if_decoded",
            wait_or_test: "model_relative_if_decoded",
            destination_visibility: "unobserved_without_consumer_payload_observation",
            payload_observation: "not_present_in_alpha_trace_contract",
            affected_async_records,
            non_claim: "ledger separates semantic edges but does not supply native completion timing",
        }
    }
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

impl TimingReport {
    /// Project the existing simulator-only timing fields into the factorized
    /// frontier contract. Hardware/profiler endpoints are intentionally left
    /// unknown; this method does not create a CUDA-event or NCU conversion.
    pub fn latent_frontiers(&self) -> Result<FrontierVector, LatentModelError> {
        // The current SM86 LDGSTS source/destination timing is a diagnostic
        // hypothesis. A live same-binary control differential shows that its
        // completion frontier can overshoot the NCU target by a large amount;
        // do not expose that hypothesis as a prediction lower bound until C3
        // closes an independent async completion witness.
        let async_completion_unvalidated = self.readiness.diagnostics.async_groups.groups_committed
            != 0
            || self.alpha_approximation.is_some();
        let modeled_event = if async_completion_unvalidated {
            FrontierBound::unknown()
        } else {
            self.terminal_modeled_event_time
                .map_or_else(FrontierBound::unknown, FrontierBound::exact)
        };
        let typed_tail = match &self.completion_frontier {
            _ if async_completion_unvalidated => FrontierBound::unknown(),
            CompletionFrontierReport::Complete { cycles } => {
                cycles.map_or_else(FrontierBound::unknown, FrontierBound::exact)
            }
            CompletionFrontierReport::IncompleteUnmodeledRetirementTails {
                lower_bound_cycles,
                ..
            } => lower_bound_cycles.map_or_else(FrontierBound::unknown, FrontierBound::lower_bound),
        };
        let vector = FrontierVector {
            issue: if async_completion_unvalidated {
                FrontierBound::unknown()
            } else {
                FrontierBound::exact(self.diagnostic_last_issue_cycles)
            },
            modeled_event,
            resource_service: FrontierBound::unknown(),
            launch_envelope: FrontierBound::unknown(),
            typed_tail,
            cache_history: FrontierBound::unknown(),
            visibility: FrontierBound::unknown(),
            writeback_service: FrontierBound::unknown(),
            // Native replay starts at cycle zero. The completion endpoint
            // includes required tails, not just the last processed event.
            endpoint_start: FrontierBound::exact(0),
            endpoint_end: typed_tail,
            required_completion: [
                sim_semantics::CompletionRequirement::ModeledEvent,
                sim_semantics::CompletionRequirement::TypedTail,
            ]
            .into_iter()
            .collect(),
        };
        vector.validate()?;
        Ok(vector)
    }

    /// Declare the simulator's own endpoint semantics without implying that
    /// it is the NCU, CUPTI, CUDA-event, or host endpoint.
    pub fn simulator_endpoint_descriptor() -> Result<EndpointDescriptor, LatentModelError> {
        EndpointDescriptor::new(
            ClockDomain::SimulatorCycles,
            "simulator_cycle_origin_zero",
            "simulator_completion_including_required_retirement_tails",
            vec!["simulator_issue_event_queue_and_required_retirement_tails".to_owned()],
            vec![
                "host_submission".to_owned(),
                "ncu_frontend_wait_for_idle".to_owned(),
                "profiler_or_host_observer_tail".to_owned(),
            ],
            "deterministic_sim_engine_event_order",
            "simulator_replay_configuration",
            "architecture_card_model_clock",
        )
    }

    /// Declare Nsight Compute's profiler endpoint semantics as an independent
    /// comparison contract. This descriptor carries no measured duration and
    /// authorizes no conversion to simulator cycles.
    pub fn ncu_endpoint_descriptor() -> Result<EndpointDescriptor, LatentModelError> {
        EndpointDescriptor::new(
            ClockDomain::NsightCompute,
            "ncu_frontend_wait_for_idle_then_grid_start",
            "ncu_grid_completion_semaphore",
            vec![
                "profiled_kernel_grid".to_owned(),
                "ncu_profiler_instrumentation".to_owned(),
            ],
            vec!["host_submission".to_owned()],
            "ncu_profiler_replay_application_endpoint",
            "ncu_cache_control_policy",
            "ncu_clock_control_policy",
        )
    }
}

#[derive(Debug, Serialize)]
pub struct GpcObservationFrontierReport {
    pub lower_bound_cycles: u64,
    pub envelope_cycles: u64,
    pub grid_blocks: u64,
    pub grid_log2: u32,
    pub cycles: u64,
    pub role: &'static str,
}

#[derive(Debug, Serialize)]
pub struct SmActiveObservationFrontierReport {
    pub issue_lower_bound_cycles: u64,
    pub issue_cycles: u64,
    pub terminal_lower_bound_cycles: Option<u64>,
    pub terminal_cycles: Option<u64>,
    pub envelope_cycles: u64,
    pub role: &'static str,
}

#[derive(Debug, Serialize)]
pub struct SmActiveTerminalFrontierReport {
    pub cycles: Option<u64>,
    pub role: &'static str,
}

impl From<SmActiveObservationFrontier> for SmActiveObservationFrontierReport {
    fn from(value: SmActiveObservationFrontier) -> Self {
        Self {
            issue_lower_bound_cycles: value.issue_lower_bound_cycles,
            issue_cycles: value.issue_cycles,
            terminal_lower_bound_cycles: value.terminal_lower_bound_cycles,
            terminal_cycles: value.terminal_cycles,
            envelope_cycles: value.envelope_cycles,
            role: "architecture_scoped_sm_active_observation_boundary_not_raw_lower_bound",
        }
    }
}

impl From<GpcObservationFrontier> for GpcObservationFrontierReport {
    fn from(value: GpcObservationFrontier) -> Self {
        Self {
            lower_bound_cycles: value.lower_bound_cycles,
            envelope_cycles: value.envelope_cycles,
            grid_blocks: value.grid_blocks,
            grid_log2: value.grid_log2,
            cycles: value.cycles,
            role: "architecture_scoped_gpc_observation_boundary_not_timing_crosswalk",
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum CompletionFrontierReport {
    Complete {
        cycles: Option<u64>,
    },
    IncompleteUnmodeledRetirementTails {
        lower_bound_cycles: Option<u64>,
        unmodeled_tails: TailReport,
    },
}

#[derive(Debug, Serialize)]
pub struct ReadinessReport {
    pub schema_version: u32,
    #[serde(flatten)]
    pub diagnostics: ReadinessDiagnostics,
}

#[derive(Debug, Serialize)]
pub struct CounterReport {
    pub instructions_issued: u64,
    pub records_seen: u64,
    pub front_end: FrontEndCounterReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub l1: Option<L1CounterReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub l2: Option<L2CounterReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub l2_by_source: Option<L2SourceCounterReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dram: Option<DramCounterReport>,
    pub unmodeled_tails: TailReport,
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
pub struct L2CounterReport {
    pub hits: u64,
    pub misses: u64,
    pub lookup_hits: u64,
    pub lookup_requests: u64,
    pub line_misses: u64,
    pub sector_fill_misses: u64,
    pub write_tag_hits: u64,
    pub pending_merge_sectors: u64,
    pub new_fill_sectors: u64,
}

#[derive(Debug, Serialize)]
pub struct L2SourceCounterReport {
    pub l1_fill: L2SourceLaneReport,
    pub bypass: L2SourceLaneReport,
    pub untagged: L2SourceLaneReport,
    pub store: L2SourceLaneReport,
    /// Read-class lookup hits not arising solely from a pending-fill merge.
    pub read_lookup_hits_without_pending_merges: u64,
    pub read_pending_merge_hits: u64,
    /// Observer-only resident hits among the launch-local coalesced store
    /// packet candidates. This is separate from the physical store lane.
    pub store_lookup_requests: u64,
    pub store_packet_lookup_hits: u64,
    pub store_packet_resident_lookup_hits: u64,
    pub store_packet_candidate_sectors: u64,
    pub client_scope_requests: [u64; 10],
    pub client_scope_resident_hits: [u64; 10],
    pub client_scope_pending_merges: [u64; 10],
    pub client_scope_fill_misses: [u64; 10],
    pub ltc128b_shadow_promoted_sectors: u64,
}

#[derive(Debug, Serialize)]
pub struct L2SourceLaneReport {
    pub sectors: u64,
    pub hits: u64,
    pub pending_pressure_samples: u64,
    pub pending_pressure_sum: u64,
}

fn l2_source_counter_report(value: &L2SourceSplit) -> L2SourceCounterReport {
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

fn l2_counter_report(value: &L2Counters) -> L2CounterReport {
    L2CounterReport {
        hits: value.hits,
        misses: value.misses,
        lookup_hits: value.lookup_hits,
        lookup_requests: value.lookup_requests,
        line_misses: value.line_misses,
        sector_fill_misses: value.sector_fill_misses,
        write_tag_hits: value.write_tag_hits,
        pending_merge_sectors: value.pending_merge_sectors,
        new_fill_sectors: value.new_fill_sectors,
    }
}

#[derive(Debug, Serialize)]
pub struct DramCounterReport {
    pub read_sectors: u64,
    /// Configured aggregate carded rate for this launch. The corpus runner's
    /// 825 GB/s effective hypothesis and direct-run card-peak fallback are
    /// both represented as configured rates, not physical controller truth.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_rate_bytes_per_sec: Option<u64>,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub l2_frontier_samples: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub l2_issue_to_fill_cycles_sum: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub l2_issue_to_fill_cycles_max: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub l2_issue_to_response_cycles_sum: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub l2_issue_to_response_cycles_max: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub l2_service_delay_cycles_sum: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub l2_service_delay_cycles_max: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub atomic_rmw_read_sectors: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub reduction_rmw_read_sectors: u64,
    pub bandwidth_delayed_responses: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub l2_evicted_lines: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub l2_evicted_valid_sectors: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub l2_evicted_pending_lines: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub l2_evicted_pending_sectors: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub l2_dirty_eviction_lines: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub l2_dirty_evicted_sectors: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub writeback_completed_lines: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub writeback_completed_sectors: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub writeback_completed_bytes: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub l1_writeback_completed_lines: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub l1_writeback_completed_sectors: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub l1_writeback_completed_bytes: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub l1_payload_forwarded_sectors: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub l1_payload_forwarded_bytes: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub l1_payload_forwarding_digest_xor: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub writeback_completion_frontier: Option<u64>,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub coherence_completed_transactions: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub coherence_completed_sectors: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coherence_completion_frontier: Option<u64>,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub maximum_terminal_l2_dirty_lines: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub maximum_terminal_l2_dirty_sectors: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub maximum_terminal_l1_dirty_lines: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub maximum_terminal_l1_dirty_sectors: u64,
}

fn is_zero_u64(value: &u64) -> bool {
    *value == 0
}

#[derive(Debug, Serialize)]
pub struct TailReport {
    pub reduction_read_modify_write: bool,
    pub dirty_writeback: bool,
    pub coherence: bool,
    pub async_copy_shared_visibility: bool,
}

fn tail_report(tails: TailFlags) -> TailReport {
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

fn effective_gpc_completion_frontier_report(
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

