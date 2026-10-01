//! Direct composition of trace loading, scheduling, and GPU replay.
//!
//! The runner accepts ordinary input paths. It deliberately has no admission
//! manifest, artifact digest, receipt, publication, or approval layer.

mod metrics;
pub mod ncu;
mod prepared;
mod resources;

pub use prepared::{
    CoverageAdmission, KernelKey, KernelSelector, PreparedLaunchIdentity, PreparedReplay,
};

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::str::FromStr;
#[cfg(feature = "performance-counters")]
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sim_arch::card::{ArchCard, CalibError};
use sim_arch::spec::{ArchitectureSpec, ResourceKind};
use sim_mem::{
    CoherenceConfig, L1Counters, L2Config, L2Counters, L2Source, MemoryOrigin,
    NativePayloadRecordV1, PayloadForwardingLedgerV1, PchaseCard, SectorPayloadV1,
    coalesce_payloads, enrich_native_payload_records, join_native_payload_records,
    load_native_payload_sidecar_records,
};
use sim_run::emit::{
    EmitOutcome, EmittedMetricKey, FullLaunchPredictions, MetricValue, Unavailable, Unit,
    emit_launch,
};
use sim_semantics::{
    CacheCandidateSummary, ClockDomain, EndpointDescriptor, FrontierBound, FrontierVector,
    LatentFactors, LatentModelError, NcuObserverProjection, ResourceServiceProjection,
    ServiceRateProfile,
};
use sim_sm::{
    ControlSidecar, DramBw, DramStats, FastFrontierSummary, FullKernelReplayResult,
    GpcObservationFrontier, IssueModelParams, L1PayloadAdmissionEvent, L2SourceSplit,
    MAX_MEMORY_ADMISSION_CANDIDATES, MAX_MEMORY_SERVICE_CANDIDATES, MemoryAdmissionCandidate,
    MemoryServiceCandidate, ModeledCompletionFrontier, ReadinessDiagnostics, ReplayMemMode,
    SmActiveObservationFrontier, StreamL2ReplaySession, TailFlags, TraceMetrics, build_schedules,
    replay_trace_kernel, replay_trace_kernel_with_l1_payload_events,
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
    /// The default remains false so existing direct-result JSON is unchanged.
    pub ncu_shaped_metrics: bool,
    /// Run only the conservative trace-derived fast-frontier analyzer. This
    /// skips native timing replay and emits no silicon timing claim.
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
    /// Emit the opt-in factorized route/class/frontier envelope. This is a
    /// diagnostic contract and does not alter replay scheduling or scoring.
    pub latent_envelope: bool,
    /// Optional abstract service-rate profile used only by the opt-in
    /// predictive envelope. It never changes native replay timing.
    pub predictive_service_profile: Option<PathBuf>,
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
) -> Result<ControlSidecar, RunError> {
    let raw = read_direct_input(path, "control bits", MAX_DIRECT_CONTROL_BYTES)?;
    ControlSidecar::parse_for_arch(path, &raw, arch).map_err(at("control bits"))
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
        let frontiers = timing.latent_frontiers().map_err(at("latent frontiers"))?;
        let resource_demand = launch
            .trace_metrics
            .physical_pipeline
            .resource_demand_envelope()
            .map_err(at("latent resource demand"))?;
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
            resource_service_projection: service_profile
                .map(|profile| resource_demand.project_service_cycles(profile))
                .transpose()
                .map_err(at("latent service projection"))?,
            cache_candidates: CacheCandidateSummary::from_topology(
                launch.trace_metrics.coherence.decoded_global_lines,
                launch.trace_metrics.coherence.cross_sm_lines,
                launch.trace_metrics.coherence.cross_sm_read_write_lines,
                launch.trace_metrics.coherence.cross_sm_write_write_lines,
            )
            .map_err(at("latent cache candidates"))?,
            resource_demand,
            operand_bank_pressure: launch.trace_metrics.operand_bank_pressure.clone(),
            ncu_projection: NcuObserverProjection::from_frontiers(frontiers, service_profile)
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

impl TimingReport {
    /// Project the existing simulator-only timing fields into the factorized
    /// frontier contract. Hardware/profiler endpoints are intentionally left
    /// unknown; this method does not create a CUDA-event or NCU conversion.
    pub fn latent_frontiers(&self) -> Result<FrontierVector, LatentModelError> {
        let modeled_event = self
            .terminal_modeled_event_time
            .map_or_else(FrontierBound::unknown, FrontierBound::exact);
        let typed_tail = match &self.completion_frontier {
            CompletionFrontierReport::Complete { cycles } => {
                cycles.map_or_else(FrontierBound::unknown, FrontierBound::exact)
            }
            CompletionFrontierReport::IncompleteUnmodeledRetirementTails {
                lower_bound_cycles,
                ..
            } => lower_bound_cycles.map_or_else(FrontierBound::unknown, FrontierBound::lower_bound),
        };
        let vector = FrontierVector {
            issue: FrontierBound::exact(self.diagnostic_last_issue_cycles),
            modeled_event,
            launch_envelope: FrontierBound::unknown(),
            typed_tail,
            visibility: FrontierBound::unknown(),
            writeback_service: FrontierBound::unknown(),
            endpoint_start: FrontierBound::unknown(),
            endpoint_end: FrontierBound::unknown(),
        };
        vector.validate()?;
        Ok(vector)
    }

    /// Declare the simulator's own endpoint semantics without implying that
    /// it is the NCU, CUPTI, CUDA-event, or host endpoint.
    pub fn simulator_endpoint_descriptor() -> Result<EndpointDescriptor, LatentModelError> {
        EndpointDescriptor::new(
            ClockDomain::SimulatorCycles,
            "first_simulator_modeled_event",
            "terminal_simulator_modeled_event",
            vec!["simulator_issue_and_event_queue".to_owned()],
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
}

#[derive(Debug, Serialize)]
pub struct L1CounterReport {
    pub hits: u64,
    pub misses: u64,
    pub merges: u64,
    pub pending_lookup_hits: u64,
    pub stall_events: u64,
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
}

#[derive(Debug, Serialize)]
pub struct L2SourceCounterReport {
    pub l1_fill: L2SourceLaneReport,
    pub bypass: L2SourceLaneReport,
    pub untagged: L2SourceLaneReport,
    pub store: L2SourceLaneReport,
}

#[derive(Debug, Serialize)]
pub struct L2SourceLaneReport {
    pub sectors: u64,
    pub hits: u64,
}

fn l2_source_counter_report(value: &L2SourceSplit) -> L2SourceCounterReport {
    let lane = |source| {
        let (sectors, hits) = value.source(source);
        L2SourceLaneReport { sectors, hits }
    };
    L2SourceCounterReport {
        l1_fill: lane(L2Source::L1Fill),
        bypass: lane(L2Source::Bypass),
        untagged: lane(L2Source::Untagged),
        store: lane(L2Source::Store),
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
    }
}

#[derive(Debug, Serialize)]
pub struct DramCounterReport {
    pub read_sectors: u64,
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

fn completion_frontier_report(value: ModeledCompletionFrontier) -> CompletionFrontierReport {
    match value {
        ModeledCompletionFrontier::Complete { cycles } => {
            CompletionFrontierReport::Complete { cycles }
        }
        ModeledCompletionFrontier::Incomplete {
            lower_bound_cycles,
            unmodeled_tails,
        } => CompletionFrontierReport::IncompleteUnmodeledRetirementTails {
            lower_bound_cycles,
            unmodeled_tails: tail_report(unmodeled_tails),
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn counters(
    instructions_issued: u64,
    records_seen: u64,
    front_end: sim_sm::FrontEndCounters,
    l1: Option<&L1Counters>,
    l2: Option<&L2Counters>,
    l2_by_source: Option<&L2SourceSplit>,
    dram: Option<&DramStats>,
    tails: TailFlags,
) -> CounterReport {
    CounterReport {
        instructions_issued,
        records_seen,
        front_end: FrontEndCounterReport {
            compulsory_instruction_read_sectors: front_end.compulsory_instruction_read_sectors(),
        },
        l1: l1.map(|value| L1CounterReport {
            hits: value.hits,
            misses: value.misses,
            merges: value.merges,
            pending_lookup_hits: value.pending_lookup_hits,
            stall_events: value.stall_events,
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

fn cache_lookup_prediction_reports(
    bundle: &FullLaunchPredictions,
    l1: Option<&L1Counters>,
    l2: Option<&L2Counters>,
) -> Result<Vec<CacheLookupPredictionReport>, RunError> {
    let specifications = [
        (
            EmittedMetricKey::CacheHitRateL1TexReadSector,
            "l1_global_read",
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
            l1.map(|counters| counters.origin(MemoryOrigin::Global).hits),
        ),
        (
            EmittedMetricKey::CacheHitRateL1TexSector,
            "l1_unified",
            "l1tex__t_sectors_lookup_hit.sum",
            "l1tex__t_sectors_lookup_miss.sum",
            l1.map(|counters| counters.hits),
        ),
        (
            EmittedMetricKey::CacheHitRateL2Sector,
            "l2_l1tex_source",
            "lts__t_sectors_srcunit_tex_lookup_hit.sum",
            "lts__t_sectors_srcunit_tex_lookup_miss.sum",
            l2.map(|counters| counters.hits),
        ),
    ];
    specifications
        .into_iter()
        .map(
            |(key, scope, ncu_hit_metric, ncu_miss_metric, resident_data_hits)| {
                let prediction = bundle
                    .metrics()
                    .iter()
                    .find(|prediction| prediction.emitted_key() == key)
                    .ok_or_else(|| {
                        RunError::new(
                            "cache lookup projection",
                            format!("metric source for {scope} is missing"),
                        )
                    })?;
                match prediction.outcome() {
                    EmitOutcome::Predicted(MetricValue::Fraction {
                        numerator,
                        denominator,
                        ..
                    }) => {
                        let misses = denominator.checked_sub(*numerator).ok_or_else(|| {
                            RunError::new(
                                "cache lookup projection",
                                format!("{scope} observer hits exceed lookups"),
                            )
                        })?;
                        let resident_data_hits = resident_data_hits.ok_or_else(|| {
                            RunError::new(
                                "cache lookup projection",
                                format!("{scope} predicted without its cache state"),
                            )
                        })?;
                        let (observer_resident_hits, observer_reclassified_lookups,
                            physical_hits_not_observed) =
                            if *numerator >= resident_data_hits {
                                (
                                    resident_data_hits,
                                    numerator.checked_sub(resident_data_hits).ok_or_else(|| {
                                        RunError::new(
                                            "cache lookup projection",
                                            format!("{scope} observer hit decomposition overflow"),
                                        )
                                    })?,
                                    0,
                                )
                            } else {
                                (
                                    *numerator,
                                    0,
                                    resident_data_hits.checked_sub(*numerator).ok_or_else(|| {
                                        RunError::new(
                                            "cache lookup projection",
                                            format!("{scope} physical hit decomposition underflow"),
                                        )
                                    })?,
                                )
                            };
                        // The observer denominator is packetized and can be
                        // smaller than physical resident hits.  Its remaining
                        // population is therefore the observer miss count,
                        // never `denominator - resident_data_hits`.
                        let remaining_lookups = denominator
                            .checked_sub(*numerator)
                            .ok_or_else(|| {
                                RunError::new(
                                    "cache lookup projection",
                                    format!(
                                        "{scope} observer hits exceed total lookups"
                                    ),
                                )
                            })?;
                        Ok(CacheLookupPredictionReport {
                            scope,
                            ncu_hit_metric,
                            ncu_miss_metric,
                            status: "predicted",
                            observer: Some(CacheLookupObserverReport {
                                hits: *numerator,
                                misses,
                                lookups: *denominator,
                            }),
                            timing_state: Some(CacheLookupTimingStateReport {
                                resident_data_hits,
                                observer_resident_hits,
                                physical_hits_not_observed,
                                remaining_lookups,
                                observer_reclassified_lookups,
                                role: "cache_state_and_events_drive_timing_observer_counts_drive_ncu_validation",
                            }),
                            detail: None,
                        })
                    }
                    EmitOutcome::Unavailable(detail) => Ok(CacheLookupPredictionReport {
                        scope,
                        ncu_hit_metric,
                        ncu_miss_metric,
                        status: "unavailable",
                        observer: None,
                        timing_state: None,
                        detail: Some(*detail),
                    }),
                    EmitOutcome::Predicted(_) | EmitOutcome::RequiredUnimplemented { .. } => {
                        Err(RunError::new(
                            "cache lookup projection",
                            format!("{scope} source is not a fraction outcome"),
                        ))
                    }
                }
            },
        )
        .collect()
}

fn launch_report(
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
        let cache_lookup_predictions =
            cache_lookup_prediction_reports(&bundle, result.l1(), result.l2())?;
        let metric_predictions = bundle
            .metrics()
            .iter()
            .map(|prediction| NcuMetricProjectionReport {
                arch: prediction.arch().token().to_owned(),
                key: prediction.key(),
                ncu_metric: prediction.ncu_metric(),
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
        diagnostic_last_issue_cycles: result.diagnostic_last_issue_cycles(),
        last_processed_modeled_system_event_time: result.last_processed_modeled_system_event_time(),
        terminal_modeled_event_time: result.terminal_modeled_event_time(),
        sm_active_terminal_frontier: Some(SmActiveTerminalFrontierReport {
            cycles: result.sm_active_terminal_event_time(),
            role: "causal_sm_active_terminal_lower_bound_excludes_write_ack_and_service_only_events",
        }),
        sm_active_observation_frontier: result.sm_active_observation_frontier().map(Into::into),
        gpc_observation_frontier: result.gpc_observation_frontier().map(Into::into),
        completion_frontier: completion_frontier_report(result.modeled_completion_frontier()),
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

fn resolve_mode(
    config: &RunConfig,
    arch_card: &ArchCard,
    issue: IssueModelParams,
) -> Result<ReplayMemMode, RunError> {
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
    let l1 = issue.l1_config(geometry.capacity.line_capacity, geometry.mshr.count);
    if config.memory_mode == MemoryMode::L1 {
        if config.carded_dram_bandwidth {
            return Err(RunError::new(
                "arguments",
                "--carded-dram requires --mode l1-l2",
            ));
        }
        return Ok(ReplayMemMode::L1Const(l1));
    }

    let l2_geometry = arch_card.require_l2_geometry().map_err(at("L2 geometry"))?;
    let l2 = L2Config::from_carded_geometry_for_arch(
        issue.arch(),
        l2_geometry.capacity_bytes,
        l2_geometry.line_bytes,
        l2_geometry.sector_bytes,
        l2_geometry.num_slices,
        l2_geometry.policy.as_deref(),
    )
    .map_err(at("L2 configuration"))?;
    let dram = if config.carded_dram_bandwidth {
        let card = arch_card
            .require_dram_bandwidth()
            .map_err(at("DRAM bandwidth"))?;
        let bytes_per_sec = card
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
    #[cfg(feature = "performance-counters")]
    let total_started = Instant::now();
    #[cfg(feature = "performance-counters")]
    let performance_enabled = std::env::var_os("SIM_RUN_PERFORMANCE_COUNTERS").is_some();
    let candidate_inputs = usize::from(config.memory_service_candidates.is_some())
        + usize::from(config.memory_admission_candidates.is_some())
        + usize::from(config.memory_admission_feedback.is_some());
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
    let arch_card = load_arch_card(&config.arch_card)?;
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
    let arch = arch_card.arch_id().map_err(at("architecture card"))?;
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
    let control = load_control_sidecar(&config.control_bits, arch)?;
    validate_direct_control_identity(&config.control_bits, exec, &control)?;
    #[cfg(feature = "performance-counters")]
    let control_ns = control_started.elapsed().as_nanos();
    #[cfg(feature = "performance-counters")]
    let schedule_started = Instant::now();
    let schedules = build_schedules(exec, &control).map_err(at("schedule"))?;
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
        let mut fast_frontier = Vec::with_capacity(kernel_locs.len());
        for loc in kernel_locs {
            let schedule = schedules
                .get(&loc.kernel().function_unique_id)
                .ok_or_else(|| {
                    RunError::new(
                        "fast frontier",
                        format!(
                            "no schedule for function_unique_id {}",
                            loc.kernel().function_unique_id
                        ),
                    )
                })?;
            let summary = sim_sm::fast_frontier_for_trace(&trace, &loc, schedule, &arch_card)
                .map_err(at("fast frontier"))?;
            fast_frontier.push(FastFrontierLaunchReport {
                device_id: loc.device_id(),
                stream_id: loc.stream_id(),
                kernel_id: loc.kernel().id,
                function_unique_id: loc.kernel().function_unique_id,
                kernel_name: loc.kernel().name.clone(),
                summary,
            });
        }
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
        });
    }

    let mut launches = Vec::with_capacity(kernel_locs.len());
    #[cfg(feature = "performance-counters")]
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
    #[cfg(feature = "performance-counters")]
    let replay_ns = replay_started.elapsed().as_nanos();
    let latent_envelope = if config.latent_envelope {
        Some(build_latent_envelope(
            &launches,
            ncu_observation.is_some(),
            predictive_service_profile.as_ref(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use sim_sm::L1PayloadAdmissionEvent;
    use tempfile::NamedTempFile;

    fn fixture(mode: MemoryMode) -> RunConfig {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../run/tests/fixtures/addr_carry_canary_bundle");
        RunConfig {
            trace_dir: root.join("trace"),
            control_bits: root.join("control/canary_sm86.control_bits.json"),
            arch_card: root.join("cards/sm_86.json"),
            pchase_card: Some(root.join("cards/sm_86_static_card.json")),
            memory_mode: mode,
            cache_scope: CacheScope::Launch,
            carded_dram_bandwidth: false,
            dirty_writeback_capacity_sectors: None,
            coherence_service_cycles: None,
            coherence_capacity_sectors: None,
            l1_visibility_protocol: false,
            l1_resident_writeback: false,
            hypothetical_issue_ports: None,
            architecture_spec: None,
            kernel_id: None,
            ncu_shaped_metrics: false,
            fast_frontier_only: false,
            memory_service_candidates: None,
            memory_admission_candidates: None,
            memory_admission_feedback: None,
            ncu_csv: None,
            ncu_launch: None,
            launch_resources: None,
            native_payload_sidecar: None,
            native_payload_events: None,
            native_payload_events_path: None,
            latent_envelope: false,
            predictive_service_profile: None,
        }
    }

    #[test]
    fn direct_flat_replay_runs_fixture() {
        let report = run(&fixture(MemoryMode::Flat)).unwrap();
        assert_eq!(report.schema_version, 1);
        assert_eq!(report.simulator_version, env!("CARGO_PKG_VERSION"));
        assert_eq!(report.architecture, "sm_86");
        assert_eq!(report.cache_scope, "launch");
        assert!(!report.carded_dram_bandwidth);
        assert_eq!(report.memory_service_candidate_count, 0);
        assert_eq!(report.memory_admission_candidate_count, 0);
        assert_eq!(report.memory_admission_feedback_candidate_id, None);
        assert_eq!(report.launches.len(), 1);
        assert_eq!(report.launches[0].counters.instructions_issued, 54);
        assert_eq!(
            report.launches[0]
                .counters
                .front_end
                .compulsory_instruction_read_sectors,
            27
        );
        assert!(report.launches[0].metric_predictions.is_none());
        assert!(report.launches[0].cache_lookup_predictions.is_none());
        assert_eq!(report.launches[0].coverage.blocks_observed, 1);
        let timing = report.launches[0].timing.as_ref().unwrap();
        assert_eq!(timing.readiness.schema_version, 24);
        let json = serde_json::to_value(&report).unwrap();
        let readiness = &json["launches"][0]["timing"]["readiness"];
        assert_eq!(readiness["schema_version"], 24);
        assert!(readiness.get("limiting_cycles").is_some());
        assert!(readiness.get("constraint_exposure_cycles").is_some());
        assert!(readiness.get("async_groups").is_some());
        assert!(readiness.get("cta_barriers").is_some());
        assert_eq!(
            json["launches"][0]["timing"]["completion_frontier"]["status"],
            "incomplete_unmodeled_retirement_tails"
        );
        assert!(
            json["launches"][0]["timing"]["completion_frontier"]["lower_bound_cycles"]
                .as_u64()
                .is_some_and(|cycles| cycles > 0)
        );
        assert!(json.get("cache_session").is_none());
        assert!(json["launches"][0].get("cache_handoff").is_none());
    }

    #[test]
    fn event_time_payloads_require_explicit_writeback_replay_configuration() {
        let mut config = fixture(MemoryMode::Flat);
        config.native_payload_events = Some(vec![L1PayloadAdmissionEvent {
            sm_id: 0,
            time: 10,
            identity: None,
            writer: None,
            payloads: Vec::new().into_boxed_slice(),
            witnesses: Vec::new().into_boxed_slice(),
        }]);
        let error = run(&config).expect_err("event payloads must not silently use flat replay");
        assert!(
            error
                .to_string()
                .contains("event-time payload admissions require")
        );
    }

    #[test]
    fn strict_native_payload_event_document_preserves_explicit_cycle_and_identity() {
        let file = NamedTempFile::new().unwrap();
        let document = serde_json::json!({
            "schema": "native_l1_payload_events_v1",
            "status": "ok",
            "events": [{
                "cycle": 500,
                "sm_id": 0,
                "identity": {
                    "launch_id": 1,
                    "function_unique_id": 1,
                    "cta": [0, 0, 0],
                    "warp": 0,
                    "lane": 0,
                    "pc": 16,
                    "occurrence": 0
                },
                "witnesses": [{
                    "line_id": 0,
                    "allocation_id": 7,
                    "allocation_generation": 1,
                    "resident_generation": 0,
                    "resident_sector_mask": 1
                }],
                "payloads": [{
                    "allocation_id": 7,
                    "generation": 1,
                    "line_id": 0,
                    "sector": 0,
                    "byte_offset": 0,
                    "bytes_hex": "0102",
                    "known_mask": 3,
                    "sequence": 1
                }]
            }]
        });
        std::fs::write(file.path(), serde_json::to_vec(&document).unwrap()).unwrap();
        let events = load_native_payload_events(file.path()).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].time, 500);
        assert_eq!(events[0].sm_id, 0);
        assert_eq!(events[0].payloads.len(), 1);
        assert_eq!(events[0].witnesses.len(), 1);
        assert_eq!(events[0].identity.unwrap().pc, 16);
    }

    #[test]
    fn event_time_payload_replay_rejects_trace_sm_placement_mismatch() {
        let mut config = fixture(MemoryMode::L1L2);
        config.carded_dram_bandwidth = true;
        config.dirty_writeback_capacity_sectors = Some(4);
        config.coherence_service_cycles = Some(4);
        config.coherence_capacity_sectors = Some(4);
        config.l1_visibility_protocol = true;
        config.l1_resident_writeback = true;
        let trace = TraceDir::open_direct_compact(&config.trace_dir).unwrap();
        let loc = trace.kernels().next().unwrap();
        let (identity, trace_sm) = trace
            .dynamic_memory_join_locations(&loc)
            .unwrap()
            .into_iter()
            .next()
            .map(|(key, location)| (key, location.sm_id.unwrap()))
            .expect("canary must carry one dynamic global location");
        config.native_payload_events = Some(vec![L1PayloadAdmissionEvent {
            sm_id: trace_sm.saturating_add(1),
            time: 500,
            identity: Some(identity),
            writer: None,
            payloads: Vec::new().into_boxed_slice(),
            witnesses: Vec::new().into_boxed_slice(),
        }]);
        let error = run(&config).expect_err("trace placement mismatch must fail closed");
        assert!(error.to_string().contains("targets SM"));
    }

    #[test]
    fn event_time_payload_replay_rejects_non_store_trace_identity() {
        let mut config = fixture(MemoryMode::L1L2);
        config.carded_dram_bandwidth = true;
        config.dirty_writeback_capacity_sectors = Some(4);
        config.coherence_service_cycles = Some(4);
        config.coherence_capacity_sectors = Some(4);
        config.l1_visibility_protocol = true;
        config.l1_resident_writeback = true;
        let trace = TraceDir::open_direct_compact(&config.trace_dir).unwrap();
        let loc = trace.kernels().next().unwrap();
        let (identity, trace_sm) = trace
            .dynamic_memory_join_locations(&loc)
            .unwrap()
            .into_iter()
            .find(|(_, location)| {
                location.access_kind != Some(sim_trace::pb::MemoryOpAccessKind::Store)
            })
            .map(|(key, location)| (key, location.sm_id.unwrap()))
            .expect("canary must carry a non-store GLOBAL location");
        config.native_payload_events = Some(vec![L1PayloadAdmissionEvent {
            sm_id: trace_sm,
            time: 500,
            identity: Some(identity),
            writer: None,
            payloads: Vec::new().into_boxed_slice(),
            witnesses: Vec::new().into_boxed_slice(),
        }]);
        let error = run(&config).expect_err("payload admission must bind to an observed store");
        assert!(
            error
                .to_string()
                .contains("non-store or unclassified GLOBAL access")
        );
    }

    #[test]
    fn native_payload_sidecar_builds_sidecar_only_forwarding_ledger() {
        let mut config = fixture(MemoryMode::Flat);
        config.native_payload_sidecar = Some(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
                "../../workloads/research_probes/sm86_value_differential/native_payload_sidecar_delta_2pow31_seed17.json",
            ),
        );
        let report = run(&config).expect("payload sidecar fixture should replay");
        assert_eq!(report.native_payload_sidecar_records, Some(32));
        let ledger = report
            .payload_ledger
            .expect("sidecar should produce a logical forwarding ledger");
        assert_eq!(ledger.input_records, 32);
        assert_eq!(ledger.complete_sectors, 4);
        assert_eq!(ledger.identity_status, "incomplete");
        assert_eq!(ledger.identity_complete_records, 0);
        assert_eq!(ledger.identity_join, "not_attempted_incomplete_identity");
        assert_eq!(
            ledger.event_location_binding,
            "not_attempted_incomplete_identity"
        );
        assert_eq!(ledger.event_location_count, 0);
        assert_eq!(ledger.logical_l1_commits, 4);
        assert_eq!(ledger.logical_l2_commits, 4);
        assert_eq!(ledger.logical_dram_commits, 4);
        assert_eq!(ledger.baseline_status, "pre_store_bytes_unknown");
        assert_eq!(ledger.hardware_completion, "not_observed");
    }

    #[test]
    fn partial_payload_sidecar_enriches_and_joins_the_retained_trace_before_replay() {
        let fixture_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../run/tests/fixtures/addr_carry_canary_bundle");
        let trace = TraceDir::open_direct_compact(fixture_root.join("trace")).unwrap();
        let loc = trace.kernels().next().unwrap();
        let function_name = trace
            .exec_info()
            .unwrap()
            .kernels
            .get(&loc.kernel().function_unique_id)
            .unwrap()
            .function_name
            .clone();
        let keys = trace
            .dynamic_memory_join_identities(&loc)
            .unwrap()
            .into_iter()
            .take(8)
            .collect::<Vec<_>>();
        assert_eq!(keys.len(), 8, "canary must provide eight global identities");
        let sidecar = NamedTempFile::new().unwrap();
        let payloads = keys
            .iter()
            .enumerate()
            .map(|(index, key)| {
                serde_json::json!({
                    "allocation_id": 9,
                    "generation": 1,
                    "line_id": 4,
                    "sector": 0,
                    "byte_offset": index * 4,
                    "bytes_hex": "01020304",
                    "known_mask": 15_u32 << (index * 4),
                    "sequence": index + 1,
                    "function_name": function_name,
                    "pc": key.pc,
                    "occurrence": key.occurrence,
                    "cta": key.cta,
                    "warp": key.warp,
                    "lane": key.lane
                })
            })
            .collect::<Vec<_>>();
        let document = serde_json::json!({
            "schema": "native_value_payload_sidecar_v2",
            "status": "ok",
            "payloads": payloads
        });
        std::fs::write(sidecar.path(), serde_json::to_vec(&document).unwrap()).unwrap();
        let mut config = fixture(MemoryMode::Flat);
        config.native_payload_sidecar = Some(sidecar.path().to_path_buf());
        let report = run(&config).expect("complete payload identity must join the trace");
        let ledger = report.payload_ledger.unwrap();
        assert_eq!(ledger.identity_status, "complete");
        assert_eq!(ledger.identity_complete_records, 8);
        assert_eq!(ledger.identity_join, "enriched_matched_trace_identities");
        assert_eq!(ledger.event_location_binding, "trace_locations_verified");
        assert_eq!(ledger.event_location_count, 8);
    }

    #[test]
    fn timing_report_projects_only_simulator_frontiers_and_endpoint() {
        let report = run(&fixture(MemoryMode::Flat)).unwrap();
        let timing = report.launches[0]
            .timing
            .as_ref()
            .expect("fixture should expose timing");
        let frontiers = timing
            .latent_frontiers()
            .expect("fixture frontiers should be internally ordered");
        assert!(frontiers.issue.lower.is_some());
        assert!(frontiers.modeled_event.lower.is_some());
        assert!(frontiers.launch_envelope.lower.is_none());
        assert!(frontiers.visibility.lower.is_none());
        let endpoint = TimingReport::simulator_endpoint_descriptor().unwrap();
        assert_eq!(endpoint.clock_domain, ClockDomain::SimulatorCycles);
        let ncu_endpoint = TimingReport::ncu_endpoint_descriptor().unwrap();
        assert_eq!(ncu_endpoint.clock_domain, ClockDomain::NsightCompute);
        assert!(!endpoint.same_contract_as(&ncu_endpoint));
        assert!(
            !endpoint.same_contract_as(
                &EndpointDescriptor::new(
                    ClockDomain::NsightCompute,
                    "first_simulator_modeled_event",
                    "terminal_simulator_modeled_event",
                    vec!["simulator_issue_and_event_queue".to_owned()],
                    vec![
                        "host_submission".to_owned(),
                        "ncu_frontend_wait_for_idle".to_owned(),
                        "profiler_or_host_observer_tail".to_owned(),
                    ],
                    "deterministic_sim_engine_event_order",
                    "simulator_replay_configuration",
                    "architecture_card_model_clock",
                )
                .unwrap()
            )
        );
    }

    #[test]
    fn latent_envelope_retains_ncu_endpoint_contract_without_cycle_conversion() {
        let ncu_csv = NamedTempFile::new().unwrap();
        std::fs::write(ncu_csv.path(), "ID,gpu__time_duration.avg\n,us\n0,1.000\n").unwrap();
        let mut config = fixture(MemoryMode::Flat);
        config.ncu_csv = Some(ncu_csv.path().to_path_buf());
        config.ncu_launch = Some(0);
        config.latent_envelope = true;
        let report = run(&config).unwrap();
        let envelope = report.latent_envelope.unwrap();
        let external = envelope
            .external_endpoint
            .expect("NCU observation must retain a separate endpoint contract");
        assert_eq!(external.clock_domain, ClockDomain::NsightCompute);
        assert_eq!(
            envelope.launches[0].endpoint.clock_domain,
            ClockDomain::SimulatorCycles
        );
        assert!(!envelope.launches[0].endpoint.same_contract_as(&external));
    }

    #[test]
    fn latent_envelope_projects_service_profile_and_ncu_candidate_interval() {
        let mut config = fixture(MemoryMode::Flat);
        config.latent_envelope = true;
        config.predictive_service_profile = Some(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../docs/sm86_service_rate_profile_20261001.json"),
        );
        let report = run(&config).unwrap();
        let envelope = report.latent_envelope.unwrap();
        let launch = &envelope.launches[0];
        let service = launch
            .resource_service_projection
            .as_ref()
            .expect("profile should emit service projection");
        assert!(service.upper_cycles >= service.lower_cycles);
        assert_eq!(
            launch.ncu_projection.target_metric,
            "gpc__cycles_elapsed.max"
        );
        assert_eq!(
            launch.ncu_projection.role,
            "ncu_observer_candidate_interval_not_hardware_endpoint_identity"
        );
    }

    #[test]
    fn fast_frontier_only_runs_fixture_without_native_timing_replay() {
        let mut config = fixture(MemoryMode::Flat);
        config.fast_frontier_only = true;
        let report = run(&config).expect("fast frontier fixture should run");
        assert!(report.fast_frontier_only);
        assert!(report.launches.is_empty());
        let summary = &report
            .fast_frontier
            .as_ref()
            .expect("fast frontier report should be present")[0]
            .summary;
        assert_eq!(summary.contract, "sm86_fast_exact_scalar_regions_v1");
        assert_eq!(summary.blocks_seen, 1);
        assert_eq!(summary.warps_seen, 1);
        assert!(summary.terminal_state_matches);
    }

    #[test]
    fn fast_frontier_rejects_unconsumed_memory_options() {
        let mut config = fixture(MemoryMode::Flat);
        config.fast_frontier_only = true;
        config.coherence_service_cycles = Some(2);
        config.coherence_capacity_sectors = Some(4);
        let error = run(&config).expect_err("fast-only must reject coherence settings");
        assert!(
            error
                .to_string()
                .contains("cannot combine with native memory/cache")
        );
    }

    #[test]
    fn fast_frontier_rejects_native_payload_sidecar() {
        let mut config = fixture(MemoryMode::Flat);
        config.fast_frontier_only = true;
        config.native_payload_sidecar = Some(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
                "../../workloads/research_probes/sm86_value_differential/native_payload_sidecar_delta_2pow31_seed17.json",
            ),
        );
        let error = run(&config).expect_err("fast-only must reject payload sidecars");
        assert!(
            error
                .to_string()
                .contains("cannot combine with a native payload sidecar")
        );
    }

    #[test]
    fn l1_visibility_requires_typed_coherence_configuration() {
        let mut config = fixture(MemoryMode::L1L2);
        config.l1_visibility_protocol = true;
        let error = run(&config).expect_err("visibility protocol must require service bounds");
        assert!(
            error
                .to_string()
                .contains("requires --coherence-service-cycles")
        );
    }

    #[test]
    fn l1_resident_writeback_requires_queue_and_carded_dram() {
        let mut config = fixture(MemoryMode::L1L2);
        config.l1_resident_writeback = true;
        let error = run(&config).expect_err("resident writeback must require its lower queue");
        assert!(error.to_string().contains("L1 resident writeback requires"));
    }

    #[test]
    fn l1_resident_writeback_requires_visibility_with_all_lower_resources() {
        let mut config = fixture(MemoryMode::L1L2);
        config.l1_resident_writeback = true;
        config.dirty_writeback_capacity_sectors = Some(4);
        config.carded_dram_bandwidth = true;
        config.coherence_service_cycles = Some(1);
        config.coherence_capacity_sectors = Some(4);
        let error = run(&config).expect_err("resident writeback must require visibility protocol");
        assert!(error.to_string().contains("--l1-visibility-protocol"));
    }

    #[test]
    fn direct_issue_resolution_consumes_carded_scheduler_fields() {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../arch_profiles/cards/sm_86.json");
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        value["smsp"]["issue_ports"] = serde_json::json!(3);
        value["dual_issue"]["supported"] = serde_json::json!(true);
        let card = ArchCard::parse(&path, &serde_json::to_vec(&value).unwrap()).unwrap();
        let mut config = fixture(MemoryMode::Flat);
        config.hypothetical_issue_ports = None;

        let issue = resolve_issue_with_metadata(&config, &card)
            .expect("complete card should resolve")
            .0;
        assert_eq!(issue.issue_ports(), 3);
        assert!(issue.dual_issue_supported());
    }

    #[test]
    fn architecture_spec_routes_trace_safe_issue_profile_into_replay() {
        let mut config = fixture(MemoryMode::Flat);
        let profile = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../arch_profiles/specs/sm86_trace_issue_profile.json");
        config.architecture_spec = Some(profile.clone());
        let report = run(&config).expect("trace-safe architecture profile should replay");
        assert_eq!(report.hypothetical_issue_ports, Some(1));
        let expected = profile.display().to_string();
        assert_eq!(report.architecture_spec.as_deref(), Some(expected.as_str()));
    }

    #[test]
    fn functional_architecture_spec_cannot_be_applied_to_sass_replay() {
        let mut config = fixture(MemoryMode::Flat);
        config.architecture_spec = Some(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../arch_profiles/specs/simt_u32_minimal.json"),
        );
        let error = run(&config).expect_err("functional ISA must not enter SASS replay");
        assert!(error.to_string().contains("does not match card"));
    }

    #[test]
    fn trace_safe_profile_rejects_unconsumed_pipeline_fields() {
        let profile = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../arch_profiles/specs/sm86_trace_issue_profile.json");
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&profile).unwrap()).unwrap();
        value["resources"][0]["initiation_interval"] = serde_json::json!(2);
        let file = NamedTempFile::new().unwrap();
        std::fs::write(file.path(), serde_json::to_vec(&value).unwrap()).unwrap();
        let mut config = fixture(MemoryMode::Flat);
        config.architecture_spec = Some(file.path().to_path_buf());
        let error = run(&config).expect_err("ignored pipeline fields must fail closed");
        assert!(error.to_string().contains("initiation_interval=1"));
    }

    #[test]
    fn direct_input_loader_accepts_current_directory_filename() {
        let file = NamedTempFile::new_in(".").unwrap();
        let name = file.path().file_name().unwrap().to_owned();
        let raw = read_direct_input(Path::new(&name), "test input", 1024).unwrap();
        assert!(raw.is_empty());
    }

    #[test]
    fn canonical_control_sidecar_rejects_different_binary() {
        let config = fixture(MemoryMode::Flat);
        let trace = TraceDir::open_direct_compact(&config.trace_dir).unwrap();
        let exec = ExecInfo::load_from_trace_dir(trace.root()).unwrap();
        let card = ArchCard::load(&config.arch_card).unwrap();
        let mut control =
            ControlSidecar::load_for_arch(&config.control_bits, card.arch_id().unwrap()).unwrap();
        control.binary = "bin/unrelated_rebuilt_kernel".to_owned();
        let error = validate_direct_control_identity(&config.control_bits, &exec, &control)
            .expect_err("canonical stale sidecar must be rejected");
        assert_eq!(error.stage(), "identity");
        assert!(error.detail().contains("unrelated_rebuilt_kernel"));
    }

    #[test]
    fn ncu_shaped_metrics_reuse_the_registry_emitter() {
        let mut config = fixture(MemoryMode::L1L2);
        config.ncu_shaped_metrics = true;
        let report = run(&config).unwrap();
        let metrics = report.launches[0]
            .metric_predictions
            .as_ref()
            .expect("opt-in metric projections must be emitted");
        assert_eq!(metrics.len(), sim_run::emit::EMITTED_METRIC_COUNT);
        assert_eq!(metrics[0].ncu_metric, "smsp__inst_executed.sum");
        assert_eq!(
            metrics[0].outcome,
            EmitOutcome::Predicted(sim_run::emit::MetricValue::Exact(54))
        );
        assert_eq!(metrics[7].ncu_metric, "gpc__cycles_elapsed.max");
        assert!(matches!(
            metrics[7].outcome,
            EmitOutcome::Predicted(sim_run::emit::MetricValue::BindingCycles(_))
                | EmitOutcome::RequiredUnimplemented { .. }
        ));
        let dram = metrics
            .iter()
            .find(|metric| metric.ncu_metric == "dram__sectors_read.sum")
            .expect("DRAM metric is registered");
        assert_eq!(
            dram.outcome,
            EmitOutcome::Predicted(sim_run::emit::MetricValue::Exact(36))
        );
        let cache = report.launches[0]
            .cache_lookup_predictions
            .as_ref()
            .expect("count-based cache projections must accompany NCU metrics");
        assert_eq!(cache.len(), 3);
        assert_eq!(cache[0].scope, "l1_global_read");
        assert_eq!(cache[1].scope, "l1_unified");
        assert_eq!(cache[2].scope, "l2_l1tex_source");
        let gpc_frontier = report.launches[0]
            .timing
            .as_ref()
            .and_then(|timing| timing.gpc_observation_frontier.as_ref())
            .expect("SM86 shared replay exposes the GPC observation frontier");
        assert_eq!(gpc_frontier.grid_blocks, 1);
        assert_eq!(gpc_frontier.envelope_cycles, 1776);
        assert_eq!(
            gpc_frontier.cycles,
            gpc_frontier.lower_bound_cycles + gpc_frontier.envelope_cycles
        );
        let sm_active_frontier = report.launches[0]
            .timing
            .as_ref()
            .and_then(|timing| timing.sm_active_observation_frontier.as_ref())
            .expect("SM86 shared replay exposes the SM-active observation frontier");
        assert_eq!(sm_active_frontier.envelope_cycles, 0);
        assert_eq!(
            sm_active_frontier.issue_cycles,
            sm_active_frontier.issue_lower_bound_cycles + sm_active_frontier.envelope_cycles
        );
        for prediction in cache {
            let observer = prediction
                .observer
                .as_ref()
                .expect("modeled cache count pair");
            let timing = prediction
                .timing_state
                .as_ref()
                .expect("timing-state decomposition");
            assert_eq!(observer.hits + observer.misses, observer.lookups);
            assert_eq!(observer.hits + timing.remaining_lookups, observer.lookups);
            assert_eq!(
                timing.observer_resident_hits + timing.observer_reclassified_lookups,
                observer.hits
            );
            assert_eq!(
                timing.observer_resident_hits + timing.physical_hits_not_observed,
                timing.resident_data_hits
            );
        }
    }

    #[test]
    fn native_metric_projection_keeps_model_values_and_typed_gaps_separate() {
        let report = run(&fixture(MemoryMode::L1L2)).unwrap();
        let native = &report.launches[0].native_metrics;
        assert_eq!(native.schema_version, 1);
        assert_eq!(native.metric_surface_version, 2);
        assert_eq!(native.architecture, "sm_86");
        assert_eq!(native.sm_count, 84);
        assert!(native.active_sm_count <= native.sm_count);
        assert_eq!(native.metrics["warp_instructions"].value, Some(54.0));
        assert!(matches!(
            native.metrics["registers_per_thread"].status,
            "modeled" | "unavailable"
        ));
        assert!(matches!(
            native.metrics["theoretical_occupancy_pct"].status,
            "approximate" | "unavailable"
        ));
        assert_eq!(native.metrics["block_threads"].value, Some(32.0));
        assert_eq!(native.metrics["grid_blocks"].value, Some(1.0));
        assert_eq!(
            native.metrics["warps_per_block_theoretical"].value,
            Some(1.0)
        );
        assert_eq!(native.metrics["occupancy_limit_warps"].value, Some(48.0));
        assert_eq!(
            native.metrics["shared_memory_config_size"].status,
            "unavailable"
        );
        assert_eq!(native.metrics["power_watts"].status, "unavailable");
        assert_eq!(native.metrics["dram_write_sectors"].status, "unavailable");
        assert_eq!(
            native.metrics["dram_read_bytes_per_second_estimate"].status,
            "approximate"
        );
        assert_eq!(
            native.metrics["dram_read_bytes_per_second_estimate"].ncu_metric,
            None
        );
        assert_eq!(
            native.metrics["dram_read_bandwidth_pct_estimate"].status,
            "approximate"
        );
        let flat = run(&fixture(MemoryMode::Flat)).unwrap();
        let flat_native = &flat.launches[0].native_metrics;
        assert_eq!(
            flat_native.metrics["dram_read_bytes_per_second_estimate"].status,
            "unavailable"
        );
        assert_eq!(
            flat_native.metrics["dram_read_bandwidth_pct_estimate"].status,
            "unavailable"
        );
        assert!(matches!(
            native.metrics["dirty_writeback_demand_sectors"].status,
            "approximate" | "unavailable"
        ));
        assert!(
            native
                .metrics
                .contains_key("dirty_writeback_demand_lower_bound_sectors")
        );
        assert!(
            native
                .metrics
                .contains_key("dirty_writeback_completion_scenario_end_cycles")
        );
        assert!(
            native
                .metrics
                .contains_key("coherence_completion_scenario_end_cycles")
        );
        assert_eq!(
            native.metrics["branch_target_divergent_branches"].status,
            "approximate"
        );
        assert_eq!(
            native.metrics["branch_target_divergent_branches"].unit,
            "branches"
        );
        assert_eq!(
            native.metrics["ncu_pipeline_alu_instructions"].status,
            "approximate"
        );
        assert!(
            native
                .metrics
                .contains_key("scheduler_proxy_long_scoreboard_cycles")
        );
        assert!(native.metrics.contains_key("occupancy_limit_warps"));
        assert!(native.metrics.contains_key("warp_instructions_avg"));
        assert_eq!(
            native.metrics["physical_pipeline_adu_instructions"].status,
            "unavailable"
        );
        assert_eq!(
            native.metrics["physical_pipeline_assignment_pct"].status,
            "unavailable"
        );
        assert!(
            native
                .metrics
                .contains_key("branch_target_divergent_minority_threads")
        );
        assert!(native.metrics.contains_key("physical_pipeline_unique_pct"));
        assert!(
            native
                .metrics
                .contains_key("physical_pipeline_ambiguous_pct")
        );
        assert!(native.metrics.contains_key("physical_pipeline_unknown_pct"));
        assert!(
            native
                .metrics
                .contains_key("physical_pipeline_selected_assignment_pct")
        );
        assert!(native.metrics.contains_key("cfg_unique_edges"));
        assert!(native.metrics.contains_key("cfg_disjoint_path_switches"));
        assert!(
            native
                .metrics
                .contains_key("coherence_cross_sm_sector_overlap")
        );
        assert!(
            native
                .metrics
                .contains_key("coherence_cross_sm_read_write_sectors")
        );
        assert!(
            native
                .metrics
                .contains_key("coherence_cross_sm_write_write_sectors")
        );
        assert_eq!(
            native.metrics["eligible_warps"].semantics,
            "ready-warp population is not sampled at every hardware scheduler cycle"
        );
    }

    #[test]
    fn mode_tokens_are_explicit() {
        assert_eq!("l1-l2".parse::<MemoryMode>().unwrap(), MemoryMode::L1L2);
        assert!("admitted".parse::<MemoryMode>().is_err());
        assert_eq!(
            "stream-l2".parse::<CacheScope>().unwrap(),
            CacheScope::StreamL2
        );
        assert!("global".parse::<CacheScope>().is_err());
    }

    #[test]
    fn report_configuration_distinguishes_carded_dram() {
        let unmodeled = run(&fixture(MemoryMode::L1L2)).unwrap();
        let mut carded_config = fixture(MemoryMode::L1L2);
        carded_config.carded_dram_bandwidth = true;
        let carded = run(&carded_config).unwrap();

        assert!(!unmodeled.carded_dram_bandwidth);
        assert!(carded.carded_dram_bandwidth);
        assert_ne!(
            serde_json::to_vec(&unmodeled).unwrap(),
            serde_json::to_vec(&carded).unwrap()
        );
    }

    #[test]
    fn memory_service_candidate_document_is_strict_and_versioned() {
        let candidates = parse_memory_service_candidates(
            br#"{
                "schema_version": 1,
                "candidates": [{
                    "candidate_id": 7,
                    "scope": "sm",
                    "operation": "store",
                    "ideal_wavefront_units": 1,
                    "ideal_wavefront_cycles": 2,
                    "unique_line_units": 0,
                    "unique_line_cycles": 0,
                    "completion_latency_cycles": 423
                }]
            }"#,
        )
        .unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].candidate_id, 7);
        assert!(
            parse_memory_service_candidates(br#"{"schema_version":2,"candidates":[]}"#)
                .unwrap_err()
                .to_string()
                .contains("expected 1")
        );
        assert!(
            parse_memory_service_candidates(br#"{"schema_version":1,"candidates":[]}"#)
                .unwrap_err()
                .to_string()
                .contains("must not be empty")
        );
        assert!(
            parse_memory_service_candidates(
                br#"{"schema_version":1,"schema_version":1,"candidates":[]}"#
            )
            .unwrap_err()
            .to_string()
            .contains("duplicate JSON member")
        );
        assert!(
            parse_memory_service_candidates(&vec![b' '; 65 * 1024])
                .unwrap_err()
                .to_string()
                .contains("exceeds 65536 bytes")
        );

        let candidate = r#"{
            "candidate_id": 1,
            "scope": "sm",
            "operation": "store",
            "ideal_wavefront_units": 1,
            "ideal_wavefront_cycles": 1,
            "unique_line_units": 0,
            "unique_line_cycles": 0,
            "completion_latency_cycles": 0
        }"#;
        let oversized = format!(
            "{{\"schema_version\":1,\"candidates\":[{}]}}",
            std::iter::repeat_n(candidate, MAX_MEMORY_SERVICE_CANDIDATES + 1)
                .collect::<Vec<_>>()
                .join(",")
        );
        assert!(
            parse_memory_service_candidates(oversized.as_bytes())
                .unwrap_err()
                .to_string()
                .contains("maximum is 16")
        );
        let duplicate =
            format!("{{\"schema_version\":1,\"candidates\":[{candidate},{candidate}]}}");
        assert!(
            parse_memory_service_candidates(duplicate.as_bytes())
                .unwrap_err()
                .to_string()
                .contains("duplicated")
        );
    }

    #[test]
    fn memory_service_candidates_require_launch_scoped_full_hierarchy() {
        let mut config = fixture(MemoryMode::Flat);
        config.memory_service_candidates = Some(PathBuf::from("candidate.json"));
        assert!(
            run(&config)
                .unwrap_err()
                .to_string()
                .contains("requires --mode l1-l2")
        );

        config.memory_mode = MemoryMode::L1L2;
        config.cache_scope = CacheScope::StreamL2;
        assert!(
            run(&config)
                .unwrap_err()
                .to_string()
                .contains("requires --cache-scope launch")
        );
    }

    #[test]
    fn memory_admission_candidate_document_and_scope_are_strict() {
        let candidates = parse_memory_admission_candidates(
            br#"{
                "schema_version": 1,
                "candidates": [{
                    "candidate_id": 3,
                    "scope": "smsp",
                    "operation": "store",
                    "queue_capacity_instructions": 8,
                    "admission_latency_cycles": 8,
                    "l2_hit_sector_units": 1,
                    "l2_hit_sector_cycles": 1,
                    "l2_miss_sector_units": 1,
                    "l2_miss_sector_cycles": 10
                }]
            }"#,
        )
        .unwrap();
        assert_eq!(candidates[0].candidate_id, 3);
        assert!(
            parse_memory_admission_candidates(
                br#"{"schema_version":1,"schema_version":1,"candidates":[]}"#
            )
            .unwrap_err()
            .to_string()
            .contains("duplicate JSON member")
        );

        let mut config = fixture(MemoryMode::Flat);
        config.memory_admission_candidates = Some(PathBuf::from("admission.json"));
        assert!(
            run(&config)
                .unwrap_err()
                .to_string()
                .contains("requires --mode l1-l2")
        );
        config.memory_mode = MemoryMode::L1L2;
        config.cache_scope = CacheScope::StreamL2;
        assert!(
            run(&config)
                .unwrap_err()
                .to_string()
                .contains("requires --cache-scope launch")
        );

        let mut feedback = fixture(MemoryMode::Flat);
        feedback.memory_admission_feedback = Some(PathBuf::from("feedback.json"));
        assert!(
            run(&feedback)
                .unwrap_err()
                .to_string()
                .contains("--memory-admission-feedback requires --mode l1-l2")
        );

        let mut conflicting = fixture(MemoryMode::L1L2);
        conflicting.memory_admission_candidates = Some(PathBuf::from("admission.json"));
        conflicting.memory_admission_feedback = Some(PathBuf::from("feedback.json"));
        assert!(
            run(&conflicting)
                .unwrap_err()
                .to_string()
                .contains("mutually exclusive")
        );
    }

    #[test]
    fn memory_admission_feedback_enforces_exactly_one_before_candidate_validation() {
        let empty = parse_memory_admission_feedback(br#"{"schema_version":1,"candidates":[]}"#)
            .unwrap_err();
        assert_eq!(empty.stage(), "memory-admission feedback");
        assert!(empty.detail().contains("exactly one candidate, got 0"));

        let two = parse_memory_admission_feedback(
            br#"{
                "schema_version": 1,
                "candidates": [
                    {
                        "candidate_id": 3,
                        "scope": "sm",
                        "operation": "store",
                        "queue_capacity_instructions": 8,
                        "admission_latency_cycles": 8,
                        "l2_hit_sector_units": 1,
                        "l2_hit_sector_cycles": 1,
                        "l2_miss_sector_units": 1,
                        "l2_miss_sector_cycles": 10
                    },
                    {
                        "candidate_id": 3,
                        "scope": "sm",
                        "operation": "load",
                        "queue_capacity_instructions": 0,
                        "admission_latency_cycles": 0,
                        "l2_hit_sector_units": 0,
                        "l2_hit_sector_cycles": 0,
                        "l2_miss_sector_units": 0,
                        "l2_miss_sector_cycles": 0
                    }
                ]
            }"#,
        )
        .unwrap_err();
        assert_eq!(two.stage(), "memory-admission feedback");
        assert!(two.detail().contains("exactly one candidate, got 2"));

        let malformed = parse_memory_admission_feedback(b"{").unwrap_err();
        assert_eq!(malformed.stage(), "memory-admission feedback");
    }

    #[test]
    fn candidate_mutual_exclusion_precedes_individual_mode_gates() {
        let mut config = fixture(MemoryMode::Flat);
        config.memory_service_candidates = Some(PathBuf::from("service.json"));
        config.memory_admission_feedback = Some(PathBuf::from("feedback.json"));
        let error = run(&config).unwrap_err();
        assert_eq!(error.stage(), "arguments");
        assert!(error.detail().contains("mutually exclusive"));
    }

    #[test]
    fn stream_l2_scope_emits_explicit_session_metadata() {
        let mut config = fixture(MemoryMode::L1L2);
        config.cache_scope = CacheScope::StreamL2;
        let report = run(&config).unwrap();
        assert_eq!(report.launches.len(), 1);
        let handoff = report.launches[0].cache_handoff.as_ref().unwrap();
        assert_eq!(handoff.scope, "stream-l2");
        assert_eq!(handoff.sequence_index, 0);
        assert!(handoff.seeded_by_kernel_ids.is_empty());
        let session = report.cache_session.as_ref().unwrap();
        assert_eq!(session.scope, "stream-l2");
        assert_eq!(session.completed_launches, 1);
        let launch_l2 = report.launches[0].counters.l2.as_ref().unwrap();
        let source_split = report.launches[0].counters.l2_by_source.as_ref().unwrap();
        let source_sectors = source_split.l1_fill.sectors
            + source_split.bypass.sectors
            + source_split.untagged.sectors
            + source_split.store.sectors;
        let source_hits = source_split.l1_fill.hits
            + source_split.bypass.hits
            + source_split.untagged.hits
            + source_split.store.hits;
        assert_eq!(source_sectors, launch_l2.hits + launch_l2.misses);
        assert_eq!(source_hits, launch_l2.hits);
        assert_eq!(session.cumulative_l2.hits, launch_l2.hits);
        assert_eq!(session.cumulative_l2.misses, launch_l2.misses);
        assert_eq!(
            session.cumulative_l2.sector_fill_misses,
            launch_l2.sector_fill_misses
        );
    }

    #[test]
    fn stream_l2_scope_rejects_nonshared_mode_and_launch_filter() {
        let mut nonshared = fixture(MemoryMode::Flat);
        nonshared.cache_scope = CacheScope::StreamL2;
        assert!(
            run(&nonshared)
                .unwrap_err()
                .to_string()
                .contains("requires --mode l1-l2")
        );

        let mut filtered = fixture(MemoryMode::L1L2);
        filtered.cache_scope = CacheScope::StreamL2;
        filtered.kernel_id = Some(1);
        assert!(
            run(&filtered)
                .unwrap_err()
                .to_string()
                .contains("cannot skip launches")
        );
    }
}
