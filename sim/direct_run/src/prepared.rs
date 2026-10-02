use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::Serialize;
use sim_arch::card::ArchCard;
use sim_sm::{IssueModelParams, ReplayMemMode, build_schedules, replay_trace_kernel};
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
        let control = load_control_sidecar(&config.control_bits, arch)?;
        super::validate_direct_control_identity(&config.control_bits, exec, &control)?;
        let schedules = build_schedules(exec, &control).map_err(at("schedule"))?;
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

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::AtomicU64;

    use super::*;
    use crate::{MemoryMode, RunConfig, run};

    fn fixture() -> RunConfig {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../run/tests/fixtures/addr_carry_canary_bundle");
        RunConfig {
            trace_dir: root.join("trace"),
            control_bits: root.join("control/canary_sm86.control_bits.json"),
            arch_card: root.join("cards/sm_86.json"),
            pchase_card: Some(root.join("cards/sm_86_static_card.json")),
            memory_mode: MemoryMode::L1L2,
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
            service_stage_role: sim_semantics::ServiceStageRole::IndependentPostReplay,
        }
    }

    fn copy_tree(source: &Path, destination: &Path) {
        std::fs::create_dir_all(destination).expect("temporary fixture directory must be created");
        for entry in std::fs::read_dir(source).expect("source fixture directory must be readable") {
            let entry = entry.expect("source fixture entry must be readable");
            let target = destination.join(entry.file_name());
            if entry
                .file_type()
                .expect("source fixture type must be readable")
                .is_dir()
            {
                copy_tree(&entry.path(), &target);
            } else {
                std::fs::copy(entry.path(), target)
                    .expect("source fixture file must copy into temporary directory");
            }
        }
    }

    #[test]
    fn prepared_launch_matches_direct_launch_report_and_is_cold() {
        let config = fixture();
        let direct = run(&config).unwrap();
        let prepared =
            PreparedReplay::prepare(&config, CoverageAdmission::FullOnlyNoSamplingDeclaration)
                .unwrap();
        let (key, identity) = prepared.launches().next().unwrap();
        assert_eq!(identity.selector.kernel_id, 1);
        let first = prepared.replay_report(key).unwrap();
        let second = prepared.replay_report(key).unwrap();
        assert_eq!(
            serde_json::to_value(&first).unwrap(),
            serde_json::to_value(&direct).unwrap()
        );
        assert_eq!(
            serde_json::to_vec(&first).unwrap(),
            serde_json::to_vec(&second).unwrap()
        );
    }

    #[test]
    fn prepared_replay_preserves_trace_safe_profile_metadata() {
        let mut config = fixture();
        config.architecture_spec = Some(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../arch_profiles/specs/sm86_trace_issue_profile.json"),
        );
        let prepared =
            PreparedReplay::prepare(&config, CoverageAdmission::FullOnlyNoSamplingDeclaration)
                .expect("trace-safe profile should prepare");
        let key = prepared.launches().next().expect("fixture launch").0;
        let report = prepared
            .replay_report(key)
            .expect("prepared replay should run");
        assert_eq!(report.hypothetical_issue_ports, Some(1));
        assert_eq!(
            report.architecture_spec,
            config
                .architecture_spec
                .as_ref()
                .map(|path| path.display().to_string())
        );
    }

    #[test]
    fn keys_are_scoped_and_exact_function_identity_is_required() {
        let config = fixture();
        let first =
            PreparedReplay::prepare(&config, CoverageAdmission::FullOnlyNoSamplingDeclaration)
                .unwrap();
        let key = first.launches().next().unwrap().0;
        let selector = first.launches().next().unwrap().1.selector.clone();
        let second =
            PreparedReplay::prepare(&config, CoverageAdmission::FullOnlyNoSamplingDeclaration)
                .unwrap();
        assert!(second.replay_launch(key).is_err());
        let mut wrong = selector;
        wrong.function_unique_id += 1;
        assert!(first.resolve(&wrong).is_err());

        let mut wrong_name = first.launches().next().unwrap().1.selector.clone();
        wrong_name.kernel_name.push_str("_wrong");
        assert!(first.resolve(&wrong_name).is_err());

        let mut wrong_grid = first.launches().next().unwrap().1.selector.clone();
        wrong_grid.grid.0 += 1;
        assert!(first.resolve(&wrong_grid).is_err());
    }

    #[test]
    fn full_only_rejects_sampling_declaration_before_trace_loading() {
        let temporary = tempfile::tempdir().unwrap();
        std::fs::write(temporary.path().join("trace_sampling.json"), b"{}").unwrap();
        let mut config = fixture();
        config.trace_dir = temporary.path().to_path_buf();
        let error = match PreparedReplay::prepare(
            &config,
            CoverageAdmission::FullOnlyNoSamplingDeclaration,
        ) {
            Ok(_) => panic!("sampling declaration unexpectedly admitted"),
            Err(error) => error,
        };
        assert_eq!(error.stage(), "coverage admission");
        assert!(error.detail().contains("sampling declaration"));
    }

    #[test]
    fn full_only_rejects_capture_sampling_declaration_and_nonregular_root_entry() {
        let capture = tempfile::tempdir().expect("temporary directory must be created");
        std::fs::create_dir(capture.path().join("extra_info"))
            .expect("extra-info directory must be created");
        std::fs::write(
            capture
                .path()
                .join("extra_info/trace_sampling_capture.json"),
            b"{}",
        )
        .expect("capture sampling declaration must be written");
        let mut config = fixture();
        config.trace_dir = capture.path().to_path_buf();
        let capture_error =
            PreparedReplay::prepare(&config, CoverageAdmission::FullOnlyNoSamplingDeclaration)
                .err()
                .expect("capture sampling declaration must reject");
        assert_eq!(capture_error.stage(), "coverage admission");
        assert!(
            capture_error
                .detail()
                .contains("trace_sampling_capture.json")
        );

        let nonregular = tempfile::tempdir().expect("temporary directory must be created");
        std::fs::create_dir(nonregular.path().join("trace_sampling.json"))
            .expect("nonregular sampling entry must be created");
        config.trace_dir = nonregular.path().to_path_buf();
        let nonregular_error =
            PreparedReplay::prepare(&config, CoverageAdmission::FullOnlyNoSamplingDeclaration)
                .err()
                .expect("nonregular sampling entry must reject");
        assert_eq!(nonregular_error.stage(), "coverage admission");
    }

    #[cfg(unix)]
    #[test]
    fn full_only_rejects_sampling_symlink_without_following_it() {
        use std::os::unix::fs::symlink;

        let temporary = tempfile::tempdir().expect("temporary directory must be created");
        std::fs::write(temporary.path().join("sampling-target"), b"{}")
            .expect("sampling target must be written");
        symlink(
            "sampling-target",
            temporary.path().join("trace_sampling.json"),
        )
        .expect("sampling symlink must be created");
        let mut config = fixture();
        config.trace_dir = temporary.path().to_path_buf();
        let error =
            PreparedReplay::prepare(&config, CoverageAdmission::FullOnlyNoSamplingDeclaration)
                .err()
                .expect("sampling symlink must reject");
        assert_eq!(error.stage(), "coverage admission");
    }

    #[test]
    fn full_only_rejects_incomplete_launch_roster() {
        let temporary = tempfile::tempdir().expect("temporary directory must be created");
        let source = fixture().trace_dir;
        copy_tree(&source, temporary.path());
        std::fs::remove_file(
            temporary
                .path()
                .join("threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_0,0,0.pb"),
        )
        .expect("one rostered block must be removed");
        let mut config = fixture();
        config.trace_dir = temporary.path().to_path_buf();
        let error =
            PreparedReplay::prepare(&config, CoverageAdmission::FullOnlyNoSamplingDeclaration)
                .err()
                .expect("incomplete launch roster must reject");
        assert_eq!(error.stage(), "trace");
    }

    #[test]
    fn owner_allocator_uses_maximum_then_fails_closed() {
        let next = AtomicU64::new(u64::MAX);
        assert_eq!(
            allocate_owner_from(&next).unwrap(),
            PreparedReplayOwner(u64::MAX)
        );
        assert!(allocate_owner_from(&next).is_err());
    }

    #[test]
    fn legacy_zero_cluster_identity_normalizes_but_partial_zero_rejects() {
        assert_eq!(normalized_cluster_identity(None).unwrap(), (1, 1, 1));
        assert_eq!(
            normalized_cluster_identity(Some(&pb::Dim3d { x: 0, y: 0, z: 0 })).unwrap(),
            (1, 1, 1)
        );
        assert!(normalized_cluster_identity(Some(&pb::Dim3d { x: 1, y: 0, z: 1 })).is_err());
    }
}
