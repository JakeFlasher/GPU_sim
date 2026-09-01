//! Direct composition of trace loading, scheduling, and GPU replay.
//!
//! The runner accepts ordinary input paths. It deliberately has no admission
//! manifest, artifact digest, receipt, publication, or approval layer.

mod prepared;

pub use prepared::{
    CoverageAdmission, KernelKey, KernelSelector, PreparedLaunchIdentity, PreparedReplay,
};

use std::io::Read;
use std::path::PathBuf;
use std::str::FromStr;
#[cfg(feature = "performance-counters")]
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sim_arch::card::ArchCard;
use sim_mem::{L1Counters, L2Config, L2Counters, L2Source, MemoryOrigin, PchaseCard};
use sim_run::emit::{
    EmitOutcome, EmittedMetricKey, FullLaunchPredictions, MetricValue, Unavailable, Unit,
    emit_launch,
};
use sim_sm::{
    ControlSidecar, DramBw, DramStats, FullKernelReplayResult, IssueModelParams, L2SourceSplit,
    MAX_MEMORY_ADMISSION_CANDIDATES, MAX_MEMORY_SERVICE_CANDIDATES, MemoryAdmissionCandidate,
    MemoryServiceCandidate, ModeledCompletionFrontier, ReadinessDiagnostics, ReplayMemMode,
    StreamL2ReplaySession, TailFlags, build_schedules, replay_trace_kernel,
    replay_trace_kernel_with_memory_admission_candidates,
    replay_trace_kernel_with_memory_admission_feedback,
    replay_trace_kernel_with_memory_service_candidates,
};
use sim_trace::{ExecInfo, TraceDir};

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
    pub kernel_id: Option<i32>,
    /// Include the registry-bound simulator projections used for NCU joins.
    /// The default remains false so existing direct-result JSON is unchanged.
    pub ncu_shaped_metrics: bool,
    /// Optional schema-1 bounded candidate document. Candidate evaluation is
    /// observational and available only for launch-scoped full-hierarchy replay.
    pub memory_service_candidates: Option<PathBuf>,
    /// Optional schema-1 finite operand-admission candidate document.
    pub memory_admission_candidates: Option<PathBuf>,
    /// Optional schema-1 document containing exactly one timing-active finite
    /// admission candidate.
    pub memory_admission_feedback: Option<PathBuf>,
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
        serde_json::from_slice(bytes).map_err(at("memory-service candidates"))?;
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
        let file = std::fs::File::open(path).map_err(at("memory-service candidates"))?;
        let mut bytes = Vec::new();
        file.take(MAX_CANDIDATE_FILE_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(at("memory-service candidates"))?;
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
        serde_json::from_slice(bytes).map_err(at(stage))?;
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
        let file = std::fs::File::open(path).map_err(at("memory-admission candidates"))?;
        let mut bytes = Vec::new();
        file.take(MAX_CANDIDATE_FILE_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(at("memory-admission candidates"))?;
        parse_memory_admission_candidates(&bytes)
    })
    .transpose()
}

fn load_memory_admission_feedback(
    path: Option<&PathBuf>,
) -> Result<Option<MemoryAdmissionCandidate>, RunError> {
    path.map(|path| {
        let file = std::fs::File::open(path).map_err(at("memory-admission feedback"))?;
        let mut bytes = Vec::new();
        file.take(MAX_CANDIDATE_FILE_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(at("memory-admission feedback"))?;
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
    pub memory_service_candidate_count: usize,
    pub memory_admission_candidate_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_admission_feedback_candidate_id: Option<u8>,
    pub launches: Vec<LaunchReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_session: Option<CacheSessionReport>,
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
    pub completion_frontier: CompletionFrontierReport,
    pub readiness: ReadinessReport,
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
pub struct L1CounterReport {
    pub hits: u64,
    pub misses: u64,
    pub merges: u64,
    pub stall_events: u64,
    pub line_fills: u64,
    pub bypass_sectors: u64,
    pub store_sectors: u64,
    pub reduction_sectors: u64,
}

#[derive(Debug, Serialize)]
pub struct L2CounterReport {
    pub hits: u64,
    pub misses: u64,
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
        line_misses: value.line_misses,
        sector_fill_misses: value.sector_fill_misses,
        write_tag_hits: value.write_tag_hits,
    }
}

#[derive(Debug, Serialize)]
pub struct DramCounterReport {
    pub read_sectors: u64,
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
    pub maximum_terminal_l2_dirty_lines: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub maximum_terminal_l2_dirty_sectors: u64,
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

fn counters(
    instructions_issued: u64,
    records_seen: u64,
    l1: Option<&L1Counters>,
    l2: Option<&L2Counters>,
    l2_by_source: Option<&L2SourceSplit>,
    dram: Option<&DramStats>,
    tails: TailFlags,
) -> CounterReport {
    CounterReport {
        instructions_issued,
        records_seen,
        l1: l1.map(|value| L1CounterReport {
            hits: value.hits,
            misses: value.misses,
            merges: value.merges,
            stall_events: value.stall_events,
            line_fills: value.line_fills,
            bypass_sectors: value.bypass_sectors,
            store_sectors: value.store_sectors,
            reduction_sectors: value.reduction_sectors,
        }),
        l2: l2.map(l2_counter_report),
        l2_by_source: l2_by_source.map(l2_source_counter_report),
        dram: dram.map(|value| DramCounterReport {
            read_sectors: value.read_sectors,
            bandwidth_delayed_responses: value.bw_delayed_responses,
            l2_evicted_lines: value.l2_evicted_lines,
            l2_evicted_valid_sectors: value.l2_evicted_valid_sectors,
            l2_evicted_pending_lines: value.l2_evicted_pending_lines,
            l2_evicted_pending_sectors: value.l2_evicted_pending_sectors,
            l2_dirty_eviction_lines: value.l2_dirty_eviction_lines,
            l2_dirty_evicted_sectors: value.l2_dirty_evicted_sectors,
            maximum_terminal_l2_dirty_lines: value.maximum_terminal_l2_dirty_lines,
            maximum_terminal_l2_dirty_sectors: value.maximum_terminal_l2_dirty_sectors,
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
                        let observer_reclassified_lookups = numerator
                            .checked_sub(resident_data_hits)
                            .ok_or_else(|| {
                                RunError::new(
                                    "cache lookup projection",
                                    format!(
                                        "{scope} observer hits are below resident-data hits"
                                    ),
                                )
                            })?;
                        let remaining_lookups = denominator
                            .checked_sub(resident_data_hits)
                            .ok_or_else(|| {
                                RunError::new(
                                    "cache lookup projection",
                                    format!(
                                        "{scope} resident-data hits exceed total lookups"
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
        result.l1(),
        result.l2(),
        result.l2_by_source(),
        result.dram(),
        result.unmodeled_tails(),
    );
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
    let pchase = PchaseCard::load(pchase_path).map_err(at("P-chase card"))?;
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
        DramBw::Carded {
            arch: issue.arch(),
            bytes_per_sec: card
                .validated_peak_bytes_per_second()
                .map_err(at("DRAM bandwidth"))?,
            sm_clock_hz: card.locked_sm_clock_hz().map_err(at("DRAM clock"))?,
        }
    } else {
        DramBw::Unmodeled { arch: issue.arch() }
    };
    Ok(ReplayMemMode::L1SharedL2 { l1, l2, dram })
}

pub fn run(config: &RunConfig) -> Result<RunReport, RunError> {
    #[cfg(feature = "performance-counters")]
    let total_started = Instant::now();
    #[cfg(feature = "performance-counters")]
    let performance_enabled = std::env::var_os("SIM_RUN_PERFORMANCE_COUNTERS").is_some();
    let candidate_inputs = usize::from(config.memory_service_candidates.is_some())
        + usize::from(config.memory_admission_candidates.is_some())
        + usize::from(config.memory_admission_feedback.is_some());
    if candidate_inputs > 1 {
        return Err(RunError::new(
            "arguments",
            "memory-service, memory-admission, and feedback candidate files are mutually exclusive",
        ));
    }
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
        && !specialized_loader_requested
        && !performance_diagnostics_requested
    {
        return PreparedReplay::prepare(config, CoverageAdmission::ExistingDirectBehavior)?
            .replay_all_report();
    }
    let arch_card = ArchCard::load(&config.arch_card).map_err(at("architecture card"))?;
    let arch = arch_card.arch_id().map_err(at("architecture card"))?;
    let latencies = arch_card
        .required_latencies()
        .map_err(at("architecture latencies"))?;
    let issue = IssueModelParams::from_legacy_card_latencies_for_arch(
        arch,
        latencies.l1_hit,
        latencies.l2_hit,
        latencies.dram,
        latencies.shared,
    )
    .map_err(at("issue configuration"))?;
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
    let exec = ExecInfo::load_from_trace_dir(trace.root()).map_err(at("execution info"))?;
    #[cfg(feature = "performance-counters")]
    let exec_ns = exec_started.elapsed().as_nanos();
    #[cfg(feature = "performance-counters")]
    let control_started = Instant::now();
    let control =
        ControlSidecar::load_for_arch(&config.control_bits, arch).map_err(at("control bits"))?;
    #[cfg(feature = "performance-counters")]
    let control_ns = control_started.elapsed().as_nanos();
    #[cfg(feature = "performance-counters")]
    let schedule_started = Instant::now();
    let schedules = build_schedules(&exec, &control).map_err(at("schedule"))?;
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

    let mut launches = Vec::with_capacity(kernel_locs.len());
    #[cfg(feature = "performance-counters")]
    let replay_started = Instant::now();
    let cache_session = match config.cache_scope {
        CacheScope::Launch => {
            for loc in kernel_locs {
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
                launches.push(launch_report(
                    result,
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
                launches.push(launch_report(
                    result,
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
    let report = RunReport {
        schema_version: 1,
        simulator_version: env!("CARGO_PKG_VERSION"),
        architecture: arch.token().to_owned(),
        memory_mode: config.memory_mode.token(),
        cache_scope: config.cache_scope.token(),
        carded_dram_bandwidth: config.carded_dram_bandwidth,
        memory_service_candidate_count: memory_service_candidates.as_ref().map_or(0, Vec::len),
        memory_admission_candidate_count: memory_admission_candidates.as_ref().map_or(0, Vec::len),
        memory_admission_feedback_candidate_id: memory_admission_feedback
            .map(|candidate| candidate.candidate_id),
        launches,
        cache_session,
    };
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
            kernel_id: None,
            ncu_shaped_metrics: false,
            memory_service_candidates: None,
            memory_admission_candidates: None,
            memory_admission_feedback: None,
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
        assert!(report.launches[0].metric_predictions.is_none());
        assert!(report.launches[0].cache_lookup_predictions.is_none());
        assert_eq!(report.launches[0].coverage.blocks_observed, 1);
        let timing = report.launches[0].timing.as_ref().unwrap();
        assert_eq!(timing.readiness.schema_version, 1);
        let json = serde_json::to_value(&report).unwrap();
        let readiness = &json["launches"][0]["timing"]["readiness"];
        assert_eq!(readiness["schema_version"], 1);
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
            EmitOutcome::RequiredUnimplemented { .. }
        ));
        let cache = report.launches[0]
            .cache_lookup_predictions
            .as_ref()
            .expect("count-based cache projections must accompany NCU metrics");
        assert_eq!(cache.len(), 3);
        assert_eq!(cache[0].scope, "l1_global_read");
        assert_eq!(cache[1].scope, "l1_unified");
        assert_eq!(cache[2].scope, "l2_l1tex_source");
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
            assert_eq!(
                timing.resident_data_hits + timing.remaining_lookups,
                observer.lookups
            );
            assert_eq!(
                timing.resident_data_hits + timing.observer_reclassified_lookups,
                observer.hits
            );
        }
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
