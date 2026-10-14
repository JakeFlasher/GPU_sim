use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::Serialize;
use sim_arch::card::ArchCard;
use sim_sm::{
    IssueModelParams, ReplayMemMode, Sm120AlphaConfig, build_schedules,
    build_schedules_sm120_alpha, replay_trace_kernel,
};
use sim_trace::{AnchoredDir, TraceDir, kernel_grid, pb};

use super::{
    CacheScope, LaunchReport, RunConfig, RunError, RunReport, at, launch_report, load_arch_card,
    load_control_sidecar, resolve_mode,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoverageAdmission {
    ExistingDirectBehavior,
    FullOnlyNoSamplingDeclaration,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct KernelSelector {
    pub device_id: i64,
    pub stream_id: i64,
    pub kernel_id: i32,
    pub function_unique_id: i32,
    pub kernel_name: String,
    pub grid: (u32, u32, u32),
    pub block: Option<(u32, u32, u32)>,
    pub cluster: (u32, u32, u32),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct PreparedLaunchIdentity {
    pub selector: KernelSelector,
    pub kernel_name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PreparedReplayOwner(u64);

static NEXT_PREPARED_REPLAY_OWNER: AtomicU64 = AtomicU64::new(1);

fn allocate_owner_from(next: &AtomicU64) -> Result<PreparedReplayOwner, RunError> {
    next.fetch_update(
        Ordering::Relaxed,
        Ordering::Relaxed,
        |current| match current {
            0 => None,
            u64::MAX => Some(0),
            _ => current.checked_add(1),
        },
    )
    .map(PreparedReplayOwner)
    .map_err(|_| RunError::new("prepared replay", "owner identity space is exhausted"))
}

fn allocate_owner() -> Result<PreparedReplayOwner, RunError> {
    allocate_owner_from(&NEXT_PREPARED_REPLAY_OWNER)
}

/// Opaque key for one launch owned by one [`PreparedReplay`].
///
/// Private fields prevent callers from minting a key or swapping one between
/// preparations. The process-local owner never enters serialized output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelKey {
    owner: PreparedReplayOwner,
    index: usize,
}

pub struct PreparedReplay {
    owner: PreparedReplayOwner,
    trace: TraceDir,
    schedules: std::collections::BTreeMap<i32, sim_sm::FunctionSchedule>,
    arch_card: ArchCard,
    issue: IssueModelParams,
    mode: ReplayMemMode,
    architecture: String,
    memory_mode: &'static str,
    carded_dram_bandwidth: bool,
    hypothetical_issue_ports: Option<u32>,
    architecture_spec: Option<String>,
    sm120_alpha: bool,
    launches: Vec<PreparedLaunchIdentity>,
}

impl PreparedReplay {
    pub fn prepare(
        config: &RunConfig,
        coverage_admission: CoverageAdmission,
    ) -> Result<Self, RunError> {
        if config.dirty_writeback_capacity_sectors.is_some()
            || config.coherence_service_cycles.is_some()
            || config.l1_visibility_protocol
            || config.l1_resident_writeback
            || config.native_payload_sidecar.is_some()
            || config.native_payload_events.is_some()
            || config.native_payload_events_path.is_some()
        {
            return Err(RunError::new(
                "prepared replay",
                "opt-in memory service or payload sidecar requires the ordinary launch replay path",
            ));
        }
        require_launch_cold_ordinary(config)?;
        if coverage_admission == CoverageAdmission::FullOnlyNoSamplingDeclaration {
            reject_sampling_declarations(&config.trace_dir)?;
        }

        let arch_card = load_arch_card(&config.arch_card)?;
        let arch = arch_card.arch_id().map_err(at("architecture card"))?;
        let (issue, architecture_spec_issue_ports) =
            super::resolve_issue_with_metadata(config, &arch_card)?;
        let mode = resolve_mode(config, &arch_card, issue)?;
        let trace = TraceDir::open_direct_compact(&config.trace_dir).map_err(at("trace"))?;
        let exec = trace.exec_info().ok_or_else(|| {
            RunError::new(
                "execution info",
                "direct trace loader did not retain execution metadata",
            )
        })?;
        let control = load_control_sidecar(&config.control_bits, arch, config.sm120_alpha)?;
        super::validate_direct_control_identity(&config.control_bits, exec, &control)?;
        let schedules = if config.sm120_alpha {
            build_schedules_sm120_alpha(exec, &control, Sm120AlphaConfig::from_m8_m9())
                .map_err(at("SM120 alpha schedule"))?
        } else {
            build_schedules(exec, &control).map_err(at("schedule"))?
        };
        let launches = trace
            .kernels()
            .map(|loc| {
                let kernel = loc.kernel();
                let grid = kernel_grid(kernel).map_err(at("prepared replay identity"))?;
                let block = kernel
                    .block_dim
                    .as_ref()
                    .map(|dim| checked_dim3("block", dim))
                    .transpose()?;
                let cluster = normalized_cluster_identity(kernel.cluster_dim.as_ref())?;
                Ok(PreparedLaunchIdentity {
                    selector: KernelSelector {
                        device_id: loc.device_id(),
                        stream_id: loc.stream_id(),
                        kernel_id: kernel.id,
                        function_unique_id: kernel.function_unique_id,
                        kernel_name: kernel.name.clone(),
                        grid,
                        block,
                        cluster,
                    },
                    kernel_name: kernel.name.clone(),
                })
            })
            .collect::<Result<Vec<_>, RunError>>()?;
        if launches.is_empty() {
            return Err(RunError::new(
                "prepared replay",
                "trace has no kernel launches",
            ));
        }
        Ok(Self {
            owner: allocate_owner()?,
            trace,
            schedules,
            arch_card,
            issue,
            mode,
            architecture: arch.token().to_owned(),
            memory_mode: config.memory_mode.token(),
            carded_dram_bandwidth: config.carded_dram_bandwidth,
            hypothetical_issue_ports: config
                .hypothetical_issue_ports
                .or(architecture_spec_issue_ports),
            architecture_spec: config
                .architecture_spec
                .as_ref()
                .map(|path| path.display().to_string()),
            sm120_alpha: config.sm120_alpha,
            launches,
        })
    }

    pub fn launches(
        &self,
    ) -> impl ExactSizeIterator<Item = (KernelKey, &PreparedLaunchIdentity)> + '_ {
        self.launches.iter().enumerate().map(|(index, identity)| {
            (
                KernelKey {
                    owner: self.owner,
                    index,
                },
                identity,
            )
        })
    }

    pub fn resolve(&self, selector: &KernelSelector) -> Result<KernelKey, RunError> {
        let mut matches = self
            .launches
            .iter()
            .enumerate()
            .filter(|(_, launch)| &launch.selector == selector);
        let Some((index, _)) = matches.next() else {
            return Err(RunError::new(
                "prepared replay",
                format!("launch selector {selector:?} was not found"),
            ));
        };
        if matches.next().is_some() {
            return Err(RunError::new(
                "prepared replay",
                format!("launch selector {selector:?} is ambiguous"),
            ));
        }
        Ok(KernelKey {
            owner: self.owner,
            index,
        })
    }

    pub fn replay_launch(&self, key: KernelKey) -> Result<LaunchReport, RunError> {
        if key.owner != self.owner {
            return Err(RunError::new(
                "prepared replay",
                "kernel key belongs to another or stale preparation",
            ));
        }
        let expected = self.launches.get(key.index).ok_or_else(|| {
            RunError::new(
                "prepared replay",
                format!("kernel key index {} is out of range", key.index),
            )
        })?;
        let loc =
            self.trace.kernels().nth(key.index).ok_or_else(|| {
                RunError::new("prepared replay", "kernel locator index disappeared")
            })?;
        let observed = KernelSelector {
            device_id: loc.device_id(),
            stream_id: loc.stream_id(),
            kernel_id: loc.kernel().id,
            function_unique_id: loc.kernel().function_unique_id,
            kernel_name: loc.kernel().name.clone(),
            grid: kernel_grid(loc.kernel()).map_err(at("prepared replay identity"))?,
            block: loc
                .kernel()
                .block_dim
                .as_ref()
                .map(|dim| checked_dim3("block", dim))
                .transpose()?,
            cluster: normalized_cluster_identity(loc.kernel().cluster_dim.as_ref())?,
        };
        if observed != expected.selector {
            return Err(RunError::new(
                "prepared replay",
                format!(
                    "kernel locator identity changed: expected {:?}, got {observed:?}",
                    expected.selector
                ),
            ));
        }
        let outcome = replay_trace_kernel(
            &self.trace,
            &loc,
            &self.schedules,
            &self.arch_card,
            &self.issue,
            self.mode,
        )
        .map_err(at("replay"))?;
        let result = outcome.into_full().ok_or_else(|| {
            RunError::new(
                "replay",
                "prepared timing replay requires full launch coverage",
            )
        })?;
        launch_report(result, &self.arch_card, None, None, None, false)
    }

    /// Replay one launch and wrap it in the same configuration-bearing report
    /// shape as ordinary direct `sim-run` launch-cold execution.
    pub fn replay_report(&self, key: KernelKey) -> Result<RunReport, RunError> {
        Ok(self.report_with_launches(vec![self.replay_launch(key)?]))
    }

    /// Replay every prepared launch in deterministic trace order with a fresh
    /// launch-cold hierarchy for each launch.
    pub fn replay_all_report(&self) -> Result<RunReport, RunError> {
        let keys = self.launches().map(|(key, _)| key).collect::<Vec<_>>();
        let launches = keys
            .into_iter()
            .map(|key| self.replay_launch(key))
            .collect::<Result<Vec<_>, RunError>>()?;
        Ok(self.report_with_launches(launches))
    }

    fn report_with_launches(&self, launches: Vec<LaunchReport>) -> RunReport {
        RunReport {
            schema_version: 1,
            simulator_version: env!("CARGO_PKG_VERSION"),
            architecture: self.architecture.clone(),
            memory_mode: self.memory_mode,
            cache_scope: CacheScope::Launch.token(),
            carded_dram_bandwidth: self.carded_dram_bandwidth,
            native_payload_sidecar_records: None,
            native_payload_event_count: None,
            payload_ledger: None,
            dirty_writeback_capacity_sectors: None,
            coherence_service_cycles: None,
            coherence_capacity_sectors: None,
            l1_visibility_protocol: false,
            l1_resident_writeback: false,
            fast_frontier_only: false,
            sm120_alpha: self.sm120_alpha,
            sm120_alpha_config: self
                .sm120_alpha
                .then(sim_sm::Sm120AlphaConfig::from_m8_m9),
            sm120_alpha_profile_id: self
                .sm120_alpha
                .then_some(Sm120AlphaConfig::PROFILE_ID),
            sm120_effective_l1: None,
            hypothetical_issue_ports: self.hypothetical_issue_ports,
            architecture_spec: self.architecture_spec.clone(),
            memory_service_candidate_count: 0,
            memory_admission_candidate_count: 0,
            memory_admission_feedback_candidate_id: None,
            launches,
            fast_frontier: None,
            cache_session: None,
            ncu_observation: None,
            latent_envelope: None,
            sim_total_wall_ns: None,
            sim_replay_wall_ns: None,
        }
    }
}

fn checked_dim3(field: &'static str, dim: &pb::Dim3d) -> Result<(u32, u32, u32), RunError> {
    let converted = (
        u32::try_from(dim.x),
        u32::try_from(dim.y),
        u32::try_from(dim.z),
    );
    match converted {
        (Ok(x), Ok(y), Ok(z)) if x > 0 && y > 0 && z > 0 => Ok((x, y, z)),
        _ => Err(RunError::new(
            "prepared replay identity",
            format!(
                "{field} geometry ({}, {}, {}) is not positive",
                dim.x, dim.y, dim.z
            ),
        )),
    }
}

fn normalized_cluster_identity(dim: Option<&pb::Dim3d>) -> Result<(u32, u32, u32), RunError> {
    match dim {
        None => Ok((1, 1, 1)),
        Some(dim) if (dim.x, dim.y, dim.z) == (0, 0, 0) => Ok((1, 1, 1)),
        Some(dim) => checked_dim3("cluster", dim),
    }
}

fn require_launch_cold_ordinary(config: &RunConfig) -> Result<(), RunError> {
    if config.cache_scope != CacheScope::Launch {
        return Err(RunError::new(
            "prepared replay",
            "replay_launch requires launch-cold cache scope",
        ));
    }
    if config.memory_service_candidates.is_some()
        || config.memory_admission_candidates.is_some()
        || config.memory_admission_feedback.is_some()
    {
        return Err(RunError::new(
            "prepared replay",
            "replay_launch does not accept candidate or feedback lanes",
        ));
    }
    if config.kernel_id.is_some() {
        return Err(RunError::new(
            "prepared replay",
            "replay_launch requires exact KernelSelector resolution instead of RunConfig.kernel_id",
        ));
    }
    Ok(())
}

fn reject_sampling_declarations(root: &Path) -> Result<(), RunError> {
    let reader = AnchoredDir::open(root).map_err(at("coverage admission"))?;
    for relative in [
        Path::new("trace_sampling.json"),
        Path::new("extra_info/trace_sampling_capture.json"),
    ] {
        if reader
            .probe_optional_regular(relative)
            .map_err(at("coverage admission"))?
            .is_some()
        {
            return Err(RunError::new(
                "coverage admission",
                format!(
                    "sampling declaration {} is unsupported for full-only bridge replay",
                    relative.display()
                ),
            ));
        }
    }
    Ok(())
}
