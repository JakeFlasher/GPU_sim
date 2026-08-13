//! Direct composition of trace loading, scheduling, and GPU replay.
//!
//! The runner accepts ordinary input paths. It deliberately has no admission
//! manifest, artifact digest, receipt, publication, or approval layer.

use std::path::PathBuf;
use std::str::FromStr;

use serde::Serialize;
use sim_arch::card::ArchCard;
use sim_mem::{L1Counters, L2Config, L2Counters, PchaseCard};
use sim_sm::{
    ControlSidecar, DramBw, DramStats, IssueModelParams, KernelReplayOutcome, ReplayMemMode,
    TailFlags, build_schedules, replay_trace_kernel,
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

#[derive(Debug, Clone)]
pub struct RunConfig {
    pub trace_dir: PathBuf,
    pub control_bits: PathBuf,
    pub arch_card: PathBuf,
    pub pchase_card: Option<PathBuf>,
    pub memory_mode: MemoryMode,
    pub carded_dram_bandwidth: bool,
    pub kernel_id: Option<i32>,
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
}

fn at<E: std::fmt::Display>(stage: &'static str) -> impl FnOnce(E) -> RunError {
    move |error| RunError::new(stage, error.to_string())
}

#[derive(Debug, Serialize)]
pub struct RunReport {
    pub architecture: String,
    pub memory_mode: &'static str,
    pub launches: Vec<LaunchReport>,
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
    pub timing: Option<TimingReport>,
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
    pub terminal_modeled_event_time: Option<u64>,
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
pub struct DramCounterReport {
    pub read_sectors: u64,
    pub bandwidth_delayed_responses: u64,
}

#[derive(Debug, Serialize)]
pub struct TailReport {
    pub reduction_read_modify_write: bool,
    pub dirty_writeback: bool,
    pub coherence: bool,
    pub async_copy_shared_visibility: bool,
}

fn counters(
    instructions_issued: u64,
    records_seen: u64,
    l1: Option<&L1Counters>,
    l2: Option<&L2Counters>,
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
        l2: l2.map(|value| L2CounterReport {
            hits: value.hits,
            misses: value.misses,
            line_misses: value.line_misses,
            sector_fill_misses: value.sector_fill_misses,
            write_tag_hits: value.write_tag_hits,
        }),
        dram: dram.map(|value| DramCounterReport {
            read_sectors: value.read_sectors,
            bandwidth_delayed_responses: value.bw_delayed_responses,
        }),
        unmodeled_tails: TailReport {
            reduction_read_modify_write: tails.red_rmw,
            dirty_writeback: tails.dirty_writeback,
            coherence: tails.coherence,
            async_copy_shared_visibility: tails.ldgsts_dst_shared_timing,
        },
    }
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

    let trace = TraceDir::open_direct(&config.trace_dir).map_err(at("trace"))?;
    let exec = ExecInfo::load_from_trace_dir(trace.root()).map_err(at("execution info"))?;
    let control =
        ControlSidecar::load_for_arch(&config.control_bits, arch).map_err(at("control bits"))?;
    let schedules = build_schedules(&exec, &control).map_err(at("schedule"))?;

    let mut launches = Vec::new();
    for loc in trace
        .kernels()
        .filter(|loc| config.kernel_id.is_none_or(|id| loc.kernel().id == id))
    {
        let outcome = replay_trace_kernel(&trace, &loc, &schedules, &arch_card, &issue, mode)
            .map_err(at("replay"))?;
        let KernelReplayOutcome::Full(result) = outcome;
        let binding = result.coverage().launch().clone();
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
            result.dram(),
            result.unmodeled_tails(),
        );
        let timing = Some(TimingReport {
            diagnostic_last_issue_cycles: result.diagnostic_last_issue_cycles(),
            terminal_modeled_event_time: result.terminal_modeled_event_time(),
        });
        launches.push(LaunchReport {
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
            timing,
        });
    }
    if launches.is_empty() {
        return Err(RunError::new(
            "replay",
            match config.kernel_id {
                Some(id) => format!("kernel id {id} was not found"),
                None => "trace contains no kernel launches".to_owned(),
            },
        ));
    }
    Ok(RunReport {
        architecture: arch.token().to_owned(),
        memory_mode: config.memory_mode.token(),
        launches,
    })
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
            carded_dram_bandwidth: false,
            kernel_id: None,
        }
    }

    #[test]
    fn direct_flat_replay_runs_fixture() {
        let report = run(&fixture(MemoryMode::Flat)).unwrap();
        assert_eq!(report.architecture, "sm_86");
        assert_eq!(report.launches.len(), 1);
        assert_eq!(report.launches[0].counters.instructions_issued, 54);
        assert_eq!(report.launches[0].coverage.blocks_observed, 1);
        assert!(report.launches[0].timing.is_some());
    }

    #[test]
    fn mode_tokens_are_explicit() {
        assert_eq!("l1-l2".parse::<MemoryMode>().unwrap(), MemoryMode::L1L2);
        assert!("admitted".parse::<MemoryMode>().is_err());
    }
}
