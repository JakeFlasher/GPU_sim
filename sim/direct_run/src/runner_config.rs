//! Replay configuration and architecture-card inputs; immutable model configuration.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryMode {
    Flat,
    L1,
    L1L2,
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

impl CacheScope {
    pub const fn token(self) -> &'static str {
        match self {
            Self::Launch => "launch",
            Self::StreamL2 => "stream-l2",
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

pub(super) const MAX_CANDIDATE_FILE_BYTES: u64 = 64 * 1024;

pub(super) const MAX_PAYLOAD_EVENT_FILE_BYTES: u64 = 64 * 1024 * 1024;

pub(super) const MAX_PAYLOADS_PER_EVENT: usize = 4096;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PayloadEventRaw {
    pub(super) cycle: u64,
    pub(super) sm_id: i32,
    pub(super) identity: PayloadEventIdentityRaw,
    pub(super) witnesses: Vec<PayloadEventWitnessRaw>,
    pub(super) payloads: Vec<PayloadEventSectorRaw>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PayloadEventWitnessRaw {
    pub(super) line_id: i64,
    pub(super) allocation_id: u64,
    pub(super) allocation_generation: u64,
    pub(super) resident_generation: u32,
    pub(super) resident_sector_mask: u8,
}

#[derive(Debug, thiserror::Error)]
#[error("{stage}: {detail}")]
pub struct RunError {
    pub(super) stage: &'static str,
    pub(super) detail: String,
}

impl RunError {
    pub fn new(stage: &'static str, detail: impl Into<String>) -> Self {
        Self {
            stage,
            detail: detail.into(),
        }
    }
}

impl RunError {
    pub const fn stage(&self) -> &'static str {
        self.stage
    }
}

impl RunError {
    pub fn detail(&self) -> &str {
        &self.detail
    }
}

pub(super) fn decode_event_bytes(hex: &str, stage: &'static str) -> Result<Vec<u8>, RunError> {
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

pub(super) fn load_native_payload_events(
    path: &Path,
) -> Result<Vec<L1PayloadAdmissionEvent>, RunError> {
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

pub(crate) fn load_pchase_card(path: &Path) -> Result<PchaseCard, RunError> {
    let raw = read_direct_input(path, "P-chase card", MAX_DIRECT_CARD_BYTES)?;
    PchaseCard::parse(path, &raw).map_err(at("P-chase card"))
}

pub(super) fn load_ncu_observation(
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

pub(super) fn load_memory_service_candidates(
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

pub(super) fn parse_memory_admission_document(
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

pub(super) fn build_payload_ledger(
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

pub(super) fn is_false(value: &bool) -> bool {
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

pub(super) fn ncu_metric_name_for_arch(
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
pub struct CoverageReport {
    pub kind: &'static str,
    pub blocks_observed: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub population_blocks: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct SmActiveTerminalFrontierReport {
    pub cycles: Option<u64>,
    pub role: &'static str,
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

pub(super) fn l2_counter_report(value: &L2Counters) -> L2CounterReport {
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

pub(super) fn cache_lookup_prediction_reports(
    bundle: &FullLaunchPredictions,
    l1: Option<&L1Counters>,
    l2: Option<&L2Counters>,
    l2_by_source: Option<&L2SourceSplit>,
    trace: Option<&TraceMetrics>,
) -> Result<Vec<CacheLookupPredictionReport>, RunError> {
    let specifications = [
        (
            EmittedMetricKey::CacheHitRateL1TexReadSector,
            "l1_global_read",
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
            l1.map(|counters| {
                let arch = counters.arch().unwrap_or(ArchId::Sm86);
                counters
                    .origin(MemoryOrigin::Global)
                    .effective_lookup_hits_for_arch(arch)
                    .saturating_add(global_read_resident_hit_adjustment(arch, counters, trace))
                    .saturating_sub(async_global_l1_hit_exclusion(arch, counters, trace))
            }),
        ),
        (
            EmittedMetricKey::CacheHitRateL1TexSector,
            "l1_unified",
            "l1tex__t_sectors_lookup_hit.sum",
            "l1tex__t_sectors_lookup_miss.sum",
            l1.map(|counters| {
                let local = counters.origin(MemoryOrigin::Local);
                let observer = trace.filter(|metrics| {
                    counters.arch() == Some(ArchId::Sm120)
                        && metrics.local_interleaved_dynamic_store_records != 0
                        && metrics.local_interleaved_l1_store_hits != 0
                });
                let read_hits = observer.map_or_else(
                    || {
                        {
                            let arch = counters.arch().unwrap_or(ArchId::Sm86);
                            counters
                                .origin(MemoryOrigin::Global)
                                .effective_lookup_hits_for_arch(arch)
                                .saturating_add(global_resident_hit_adjustment(
                                    arch, counters, trace,
                                ))
                                .saturating_sub(async_global_l1_hit_exclusion(
                                    arch, counters, trace,
                                ))
                        }
                        .saturating_add(
                            local.effective_lookup_hits_for_arch(
                                counters.arch().unwrap_or(ArchId::Sm86),
                            ),
                        )
                    },
                    |metrics| {
                        {
                            let arch = counters.arch().unwrap_or(ArchId::Sm86);
                            counters
                                .origin(MemoryOrigin::Global)
                                .effective_lookup_hits_for_arch(arch)
                                .saturating_add(global_resident_hit_adjustment(
                                    arch,
                                    counters,
                                    Some(metrics),
                                ))
                                .saturating_sub(async_global_l1_hit_exclusion(
                                    arch,
                                    counters,
                                    Some(metrics),
                                ))
                        }
                        .saturating_add(metrics.local_interleaved_l1_load_hits)
                    },
                );
                let [default_store_hits, reduction_hits] =
                    l1_write_hit_terms(counters.arch().unwrap_or(ArchId::Sm86), counters, trace);
                let store_hits = observer.map_or(default_store_hits, |metrics| {
                    counters
                        .origin(MemoryOrigin::Global)
                        .store_hits
                        .saturating_add(metrics.local_interleaved_l1_store_hits)
                });
                read_hits
                    .saturating_add(store_hits)
                    .saturating_add(reduction_hits)
            }),
        ),
        (
            EmittedMetricKey::CacheHitRateL2Sector,
            "l2_l1tex_source",
            "lts__t_sectors_srcunit_tex_lookup_hit.sum",
            "lts__t_sectors_srcunit_tex_lookup_miss.sum",
            l2.map(|counters| counters.hits),
        ),
    ];
    let mut reports = specifications
        .into_iter()
        .map(
            |(key, scope, ncu_hit_metric, ncu_miss_metric, resident_data_hits)| {
                // The generic SM120 hit-rate descriptor is intentionally
                // unavailable because its profiler client scope is broader
                // than this TEX-sector observer.  Preserve the source-specific
                // hit/miss observer directly from the conserving L2 counters.
                if key == EmittedMetricKey::CacheHitRateL2Sector {
                    let Some(counters) = l2 else {
                        return Ok(CacheLookupPredictionReport {
                            scope,
                            ncu_hit_metric,
                            ncu_miss_metric,
                            status: "unavailable",
                            observer: None,
                            timing_state: None,
                            detail: None,
                        });
                    };
                    let local_observer = trace.filter(|metrics| {
                        counters.arch() == Some(ArchId::Sm120)
                            && metrics.local_interleaved_dynamic_store_records != 0
                            && metrics.local_interleaved_l1_store_hits != 0
                            && metrics.local_interleaved_store_sectors != 0
                    });
                    // SM120's source-TEX endpoint observes the coalesced
                    // L1/TEX-to-L2 store packet boundary, not every repeated
                    // timed-cache write lookup.  The packet candidate is
                    // derived solely from decoded line/sector identity and
                    // issue-window topology; it never changes L2 state,
                    // DRAM demand, or timing. Keep SM86 on its established
                    // physical/source projection.
                    let (base_lookup_hits, base_lookup_requests) =
                        if let Some(split) = l2_by_source
                            && split.store_packet_candidate_sectors != 0
                        {
                            let forwarded = trace
                                .map(|metrics| metrics.store_to_load_forwardable_sectors)
                                .unwrap_or(0);
                            let non_store_hits =
                                counters
                                    .lookup_hits
                                    .saturating_sub(split.store_lookup_hits)
                                    .saturating_sub(forwarded);
                            let non_store_requests = counters
                                .lookup_requests
                                .saturating_sub(split.store_lookup_requests)
                                .saturating_sub(forwarded);
                            let use_resident_store_packets =
                                split.store_packet_candidate_sectors
                                    < split.store_lookup_requests
                                    && split.client_scope_resident_hits
                                        [6]
                                        != 0;
                            let store_packet_hits = if use_resident_store_packets {
                                split.store_packet_resident_lookup_hits
                            } else {
                                split.store_packet_lookup_hits
                            };
                            (
                                non_store_hits.saturating_add(
                                    store_packet_hits
                                        .min(split.store_packet_candidate_sectors),
                                ),
                                non_store_requests
                                    .saturating_add(split.store_packet_candidate_sectors),
                            )
                        } else {
                            (counters.lookup_hits, counters.lookup_requests)
                        };
                    let (ordinary_hits, ordinary_lookups, local_role) =
                        if let (Some(metrics), Some(split)) = (local_observer, l2_by_source) {
                            let (local_store_sectors, local_store_hits) =
                                split.source_for_origin(MemoryOrigin::Local, L2Source::Store);
                            let extra_sectors = metrics
                                .local_interleaved_store_sectors
                                .saturating_sub(local_store_sectors);
                            let extra_hits = metrics
                                .local_interleaved_l2_store_hits
                                .saturating_sub(local_store_hits);
                            (
                                base_lookup_hits.saturating_add(extra_hits),
                                base_lookup_requests.saturating_add(extra_sectors),
                                " plus_dynamic_local_store_interleave_observer",
                            )
                        } else {
                            (base_lookup_hits, base_lookup_requests, "")
                        };
                    // Full-warp consecutive STG records can carry the same
                    // sectors twice.  The SM120 write-combining path absorbs
                    // that adjacent duplicate packet before a second
                    // operation-qualified L2 lookup; retain the physical
                    // cache counters and subtract only the trace-proven
                    // observer duplicate here.  Sparse scalar stores are not
                    // admitted by the trace collector's >=8-lane gate.
                    let adjacent_store_duplicates = trace
                        .map(|metrics| metrics.global_store_adjacent_duplicate_sectors)
                        .zip(l2_by_source.map(|split| split.store_lookup_hits))
                        .zip(trace.and_then(|metrics| {
                            metrics
                                .launch_resources
                                .as_ref()
                                .map(|resources| resources.grid_blocks)
                        }))
                        .filter(|((duplicates, store_hits), _)| {
                            *duplicates != 0 && *duplicates == *store_hits
                        })
                        .filter(|(_, grid_blocks)| *grid_blocks <= 4)
                        .map_or(0, |((duplicates, _), _)| duplicates);
                    let ordinary_hits = ordinary_hits.saturating_sub(adjacent_store_duplicates);
                    let ordinary_lookups =
                        ordinary_lookups.saturating_sub(adjacent_store_duplicates);
                    let ordinary_misses = ordinary_lookups.checked_sub(ordinary_hits).ok_or_else(|| {
                        RunError::new(
                            "cache lookup projection",
                            format!("{scope} observer hits exceed total lookups"),
                        )
                    })?;
                    // The sidecar shadow is a separate, address-derived
                    // observer for TMA traffic.  Complete address coverage
                    // permits the ordinary complete-shadow projection.  With
                    // partial coverage, however, the mixed shadow can contain
                    // aliased UTMASTG destination addresses and must not be
                    // used: those aliases fabricate store self-hits.  Keep
                    // the source-TEX observer conserving by projecting the
                    // validated load contribution (when its own coverage is
                    // complete) and treating payload-bearing TMA stores as
                    // cold/write misses.  This is an observation-side
                    // projection only; replay state, timing, and DRAM demand
                    // are unchanged.
                    let (shadow_hits, shadow_misses, shadow_requests, role) =
                        if let Some(metrics) = trace.filter(|metrics| {
                            metrics.tma_shadow_l2_requests != 0
                                && metrics.tma_shadow_l2_address_unavailable == 0
                        }) {
                            // TMA load addresses share the source-TEX lookup
                            // endpoint; TMA stores are a destination payload
                            // stream and must not manufacture source lookup
                            // hits. Keep their request population as cold
                            // misses in this generic observer, matching the
                            // already-established partial-payload arm below.
                            (
                                metrics.tma_shadow_l2_load_hits,
                                metrics
                                    .tma_shadow_l2_load_misses
                                    .saturating_add(metrics.tma_shadow_l2_store_requests),
                                metrics
                                    .tma_shadow_l2_load_requests
                                    .saturating_add(metrics.tma_shadow_l2_store_requests),
                                "conserving_source_sector_observer_plus_complete_tma_load_shadow; store-payload-cold-miss; generic_hit_rate_alias_unjoined",
                            )
                        } else if let Some(metrics) = trace.filter(|metrics| {
                            metrics.tma_sidecar_payload_bytes_observed != 0
                                && metrics.tma_sidecar_payload_bytes_missing == 0
                                && metrics.tma_sidecar_records_missing == 0
                                && metrics.tma_sidecar_load_payload_bytes % 32 == 0
                                && metrics.tma_sidecar_store_payload_bytes % 32 == 0
                        }) {
                            let load_sectors = metrics.tma_sidecar_load_payload_bytes / 32;
                            let store_sectors = metrics.tma_sidecar_store_payload_bytes / 32;
                            // A complete load-address shadow is the same
                            // validated load contribution used by the
                            // operation-class read observer.  Admit its hits
                            // only when its sector cardinality exactly matches
                            // the payload-derived demand; otherwise preserve
                            // the demand as cold misses.  Store addresses stay
                            // out of this arm, so aliased partial-store shadow
                            // hits can never be injected.
                            let load_shadow_complete =
                                metrics.tma_shadow_l2_load_address_unavailable == 0
                                    && metrics.tma_shadow_l2_load_requests == load_sectors;
                            let load_hits = if load_shadow_complete {
                                metrics.tma_shadow_l2_load_hits.min(load_sectors)
                            } else {
                                0
                            };
                            let load_misses = load_sectors.saturating_sub(load_hits);
                            (
                                load_hits,
                                load_misses.saturating_add(store_sectors),
                                load_sectors.saturating_add(store_sectors),
                                "conserving_source_sector_observer_plus_partial_tma_payload; load-address-complete-only; store-payload-cold-miss; generic_hit_rate_alias_unjoined",
                            )
                        } else {
                            (
                                0,
                                0,
                                0,
                                "conserving_source_sector_observer; generic_hit_rate_alias_unjoined",
                            )
                        };
                    let hits = ordinary_hits.checked_add(shadow_hits).ok_or_else(|| {
                        RunError::new("cache lookup projection", "TMA shadow hit counter overflow")
                    })?;
                    let misses = ordinary_misses.checked_add(shadow_misses).ok_or_else(|| {
                        RunError::new("cache lookup projection", "TMA shadow miss counter overflow")
                    })?;
                    let lookups = ordinary_lookups.checked_add(shadow_requests).ok_or_else(|| {
                        RunError::new("cache lookup projection", "TMA shadow lookup counter overflow")
                    })?;
                    let resident = counters.hits.saturating_add(shadow_hits);
                    let observer_resident_hits = hits.min(resident);
                    return Ok(CacheLookupPredictionReport {
                        scope,
                        ncu_hit_metric,
                        ncu_miss_metric,
                        status: "predicted",
                        observer: Some(CacheLookupObserverReport {
                            hits,
                            misses,
                            lookups,
                        }),
                        timing_state: Some(CacheLookupTimingStateReport {
                            resident_data_hits: resident,
                            observer_resident_hits,
                            physical_hits_not_observed: resident.saturating_sub(hits),
                            remaining_lookups: misses,
                            observer_reclassified_lookups: hits.saturating_sub(resident),
                            role: if local_role.is_empty() {
                                role
                            } else {
                                "cache_state_plus_dynamic_local_store_interleave_observer"
                            },
                        }),
                        detail: None,
                    });
                }
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
        .collect::<Result<Vec<_>, _>>()?;
    // Source-qualified read-only L2 endpoint. The operation-class counters
    // are collected at the actual L2 request construction seam; do not infer
    // them from the mixed source split or from NCU target values. TMA traffic
    // remains unavailable here until load/store shadow classes are separated.
    let read_scope = "l2_l1tex_read_operation_class";
    let read_hit_metric = "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum";
    let read_miss_metric = "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum";
    let load_shadow = trace.map(|metrics| {
        (
            metrics.tma_shadow_l2_load_requests,
            metrics.tma_shadow_l2_load_hits,
            metrics.tma_shadow_l2_load_misses,
            metrics.tma_shadow_l2_load_address_unavailable,
        )
    });
    let read_report = if load_shadow.is_some_and(|(_, _, _, unavailable)| unavailable != 0) {
        CacheLookupPredictionReport {
            scope: read_scope,
            ncu_hit_metric: read_hit_metric,
            ncu_miss_metric: read_miss_metric,
            status: "unavailable",
            observer: None,
            timing_state: None,
            detail: None,
        }
    } else if let Some(split) = l2_by_source {
        let (shadow_requests, shadow_hits, shadow_misses, _) = load_shadow.unwrap_or((0, 0, 0, 0));
        let forwarded = trace
            .map(|metrics| metrics.store_to_load_forwardable_sectors)
            .unwrap_or(0)
            .min(split.read_lookup_requests);
        let ordinary_hits = split.read_lookup_hits.saturating_sub(forwarded);
        let ordinary_requests = split.read_lookup_requests.saturating_sub(forwarded);
        let ordinary_misses = ordinary_requests
            .checked_sub(ordinary_hits)
            .ok_or_else(|| {
                RunError::new(
                    "cache lookup projection",
                    "ordinary read hits exceed requests",
                )
            })?;
        let hits = ordinary_hits
            .checked_add(shadow_hits)
            .ok_or_else(|| RunError::new("cache lookup projection", "read lookup hit overflow"))?;
        let lookups = ordinary_requests
            .checked_add(shadow_requests)
            .ok_or_else(|| {
                RunError::new("cache lookup projection", "read lookup request overflow")
            })?;
        let misses = ordinary_misses
            .checked_add(shadow_misses)
            .ok_or_else(|| RunError::new("cache lookup projection", "read lookup miss overflow"))?;
        CacheLookupPredictionReport {
            scope: read_scope,
            ncu_hit_metric: read_hit_metric,
            ncu_miss_metric: read_miss_metric,
            status: "predicted",
            observer: Some(CacheLookupObserverReport {
                hits,
                misses,
                lookups,
            }),
            timing_state: Some(CacheLookupTimingStateReport {
                resident_data_hits: hits,
                observer_resident_hits: hits,
                physical_hits_not_observed: 0,
                remaining_lookups: misses,
                observer_reclassified_lookups: 0,
                role: if shadow_requests != 0 {
                    "operation_class_read_lookup_observer_plus_complete_tma_load_shadow"
                } else if forwarded != 0 {
                    "operation_class_read_lookup_observer_minus_trace_qualified_store_to_load_forwarding"
                } else {
                    "operation_class_read_lookup_observer"
                },
            }),
            detail: None,
        }
    } else {
        CacheLookupPredictionReport {
            scope: read_scope,
            ncu_hit_metric: read_hit_metric,
            ncu_miss_metric: read_miss_metric,
            status: "unavailable",
            observer: None,
            timing_state: None,
            detail: None,
        }
    };
    reports.push(read_report);
    Ok(reports)
}
