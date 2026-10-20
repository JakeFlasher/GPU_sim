//! Semantic submodule; report-only boundary state and serde contracts.

use super::*;

/// Full-coverage replay result. This is the only public replay result carrying
/// per-SM state or temporal diagnostics.
#[derive(Debug, Clone)]
pub struct FullKernelReplayResult {
    pub(super) coverage: FullLaunchCoverage,
    pub(super) configuration: ReplayConfiguration,
    pub(super) replay: FoldedKernelReplay,
}

impl FullKernelReplayResult {
    pub fn coverage(&self) -> &FullLaunchCoverage {
        &self.coverage
    }

    pub fn configuration(&self) -> ReplayConfiguration {
        self.configuration
    }

    pub fn co_residency(&self) -> CoResidency {
        self.configuration.co_residency()
    }

    pub fn arch(&self) -> ArchId {
        self.replay.arch
    }

    pub fn instructions_issued(&self) -> u64 {
        self.replay.instructions_issued
    }

    pub fn records_seen(&self) -> u64 {
        self.replay.records_seen
    }

    pub fn per_sm(&self) -> &BTreeMap<i32, SmSimResult> {
        &self.replay.per_sm
    }

    /// Maximum per-SM issue frontier (`last local issue timestamp + 1`). This
    /// is the unadjusted causal scheduler/event lower bound. Consumers that
    /// need the silicon `sm__cycles_active` observation domain must use the
    /// explicit [`Self::sm_active_observation_frontier`] projection.
    pub fn diagnostic_last_issue_cycles(&self) -> u64 {
        self.replay.diagnostic_last_issue_cycles
    }

    /// Unadjusted issue frontier used by causal replay diagnostics. The value
    /// is kept separate from the architecture-scoped observation projection;
    /// it is never a hardware target or a fitted correction.
    pub fn modeled_issue_lower_bound_cycles(&self) -> u64 {
        self.replay.diagnostic_last_issue_cycles
    }

    /// Raw timestamp of the final successfully processed shared-system event,
    /// before it is combined with the issue frontier.
    pub fn last_processed_modeled_system_event_time(&self) -> Option<u64> {
        self.replay.last_processed_system_event_time
    }

    /// Maximum timestamp covered by the modeled event set: the checked issue
    /// frontier (`last local issue timestamp + 1`) and any successfully
    /// processed shared-system event. This remains an unadjusted lower bound;
    /// the explicit SM-active projection adds the observation boundary.
    pub fn terminal_modeled_event_time(&self) -> Option<u64> {
        self.replay.terminal_modeled_event_time
    }

    /// Unadjusted terminal modeled-event lower bound.
    pub fn modeled_terminal_lower_bound_cycles(&self) -> Option<u64> {
        self.replay.terminal_modeled_event_time
    }

    /// SM-active terminal lower bound with post-issue MemoryWriteAck events
    /// excluded from the observation endpoint.  Acknowledgement events remain
    /// in the full modeled frontier and still affect any dependent issue; this
    /// accessor only keeps the two observation domains distinct.
    pub fn sm_active_terminal_event_time(&self) -> Option<u64> {
        self.replay.sm_active_terminal_event_time
    }

    /// Return the architecture-modeled GPC observation frontier for a full
    /// shared-L2 replay.  `None` is reserved for a non-shared compatibility
    /// mode, an empty frontier, an unsupported architecture capability, or a
    /// checked arithmetic overflow; the physical lower-bound field above is
    /// never replaced or hidden.
    pub fn gpc_observation_frontier(&self) -> Option<GpcObservationFrontier> {
        if !matches!(
            self.configuration.memory_mode,
            ReplayMemMode::L1SharedL2 { .. }
        ) {
            return None;
        }
        let lower_bound_cycles = self.modeled_terminal_lower_bound_cycles()?;
        let grid_blocks = self.coverage.launch().grid_blocks().ok()?;
        let envelope = self
            .arch()
            .gpc_observation_envelope()
            .require("gpc_observation_envelope")
            .ok()?;
        let (envelope_cycles, grid_log2) = envelope.cycles_for_grid(grid_blocks)?;
        let cycles = lower_bound_cycles.checked_add(envelope_cycles)?;
        Some(GpcObservationFrontier {
            lower_bound_cycles,
            envelope_cycles,
            grid_blocks,
            grid_log2,
            cycles,
        })
    }

    /// Return the explicit SM-active physical observation frontier for a
    /// shared-L2 replay. The replay clock remains unadjusted; this projection
    /// adds the architecture-scoped observation boundary while retaining the
    /// causal lower bounds in the same structure.
    pub fn sm_active_observation_frontier(&self) -> Option<SmActiveObservationFrontier> {
        if !matches!(
            self.configuration.memory_mode,
            ReplayMemMode::L1SharedL2 { .. }
        ) {
            return None;
        }
        let issue_lower_bound_cycles = self.modeled_issue_lower_bound_cycles();
        let envelope_cycles = self.configuration.sm_active_activation_cycles();
        if issue_lower_bound_cycles == 0 && self.replay.terminal_modeled_event_time.is_none() {
            return None;
        }
        let issue_cycles = issue_lower_bound_cycles.checked_add(envelope_cycles)?;
        // The SM-active observation domain excludes post-issue write
        // acknowledgements and service-only writeback/coherence events.  The
        // full modeled terminal frontier remains available through
        // `terminal_modeled_event_time` and the GPC projection; using it here
        // would let a post-exit acknowledgement move an SM-active endpoint.
        let terminal_lower_bound_cycles = self.sm_active_terminal_event_time();
        let terminal_cycles =
            terminal_lower_bound_cycles.and_then(|cycles| cycles.checked_add(envelope_cycles));
        Some(SmActiveObservationFrontier {
            issue_lower_bound_cycles,
            issue_cycles,
            terminal_lower_bound_cycles,
            terminal_cycles,
            envelope_cycles,
        })
    }

    /// Typed launch-retirement diagnostic. Consumers must distinguish a
    /// complete modeled frontier from a lower bound with known missing tails.
    pub fn modeled_completion_frontier(&self) -> ModeledCompletionFrontier {
        if self.replay.unmodeled_tails.is_empty() {
            ModeledCompletionFrontier::Complete {
                cycles: self.modeled_terminal_lower_bound_cycles(),
            }
        } else {
            ModeledCompletionFrontier::Incomplete {
                lower_bound_cycles: self.modeled_terminal_lower_bound_cycles(),
                unmodeled_tails: self.replay.unmodeled_tails,
            }
        }
    }

    pub fn l1(&self) -> Option<&L1Counters> {
        self.replay.l1.as_ref()
    }

    pub fn l2(&self) -> Option<&L2Counters> {
        self.replay.l2.as_ref()
    }

    pub fn l2_by_source(&self) -> Option<&L2SourceSplit> {
        self.replay.l2_by_source.as_ref()
    }

    pub fn dram(&self) -> Option<&DramStats> {
        self.replay.dram.as_ref()
    }

    pub fn front_end(&self) -> FrontEndCounters {
        self.replay.front_end
    }

    /// Trace-derived dynamic activity and memory-class counts. These are
    /// independent of NCU and never feed timing configuration.
    pub fn trace_metrics(&self) -> &TraceMetrics {
        &self.replay.trace_metrics
    }

    pub fn unmodeled_tails(&self) -> TailFlags {
        self.replay.unmodeled_tails
    }
}

/// Counter-only observation over a verified sampled block roster. This type is
/// intentionally distinct from the full replay payload and contains no per-SM
/// result, timestamp, cycle count, or timing-result conversion.
#[cfg(feature = "campaign")]
#[derive(Debug, Clone)]
pub struct SampledObservedCounters {
    pub(super) arch: ArchId,
    pub(super) instructions_issued: u64,
    pub(super) records_seen: u64,
    pub(super) l1: Option<L1Counters>,
    pub(super) l2: Option<L2Counters>,
    pub(super) l2_by_source: Option<L2SourceSplit>,
    pub(super) dram: Option<DramStats>,
    pub(super) front_end: FrontEndCounters,
    pub(super) unmodeled_tails: TailFlags,
}

#[cfg(feature = "campaign")]
impl SampledObservedCounters {
    pub fn arch(&self) -> ArchId {
        self.arch
    }

    pub fn instructions_issued(&self) -> u64 {
        self.instructions_issued
    }

    pub fn records_seen(&self) -> u64 {
        self.records_seen
    }

    pub fn l1(&self) -> Option<&L1Counters> {
        self.l1.as_ref()
    }

    pub fn l2(&self) -> Option<&L2Counters> {
        self.l2.as_ref()
    }

    pub fn l2_by_source(&self) -> Option<&L2SourceSplit> {
        self.l2_by_source.as_ref()
    }

    pub fn dram(&self) -> Option<&DramStats> {
        self.dram.as_ref()
    }

    pub fn front_end(&self) -> FrontEndCounters {
        self.front_end
    }

    pub fn unmodeled_tails(&self) -> TailFlags {
        self.unmodeled_tails
    }
}

/// Sampled-observed replay result. No method converts this value into a full or
/// timing-bearing replay result.
#[cfg(feature = "campaign")]
#[derive(Debug, Clone)]
pub struct SampledKernelReplayResult {
    pub(super) coverage: SampledLaunchCoverage,
    pub(super) configuration: ReplayConfiguration,
    pub(super) counters: SampledObservedCounters,
}

#[cfg(feature = "campaign")]
impl SampledKernelReplayResult {
    pub fn coverage(&self) -> &SampledLaunchCoverage {
        &self.coverage
    }

    pub fn configuration(&self) -> ReplayConfiguration {
        self.configuration
    }

    pub fn counters(&self) -> &SampledObservedCounters {
        &self.counters
    }
}

/// Coverage-classified outcome for one kernel launch.
#[derive(Debug, Clone)]
pub enum KernelReplayOutcome {
    Full(FullKernelReplayResult),
    #[cfg(feature = "campaign")]
    SampledObserved(SampledKernelReplayResult),
}

impl KernelReplayOutcome {
    pub(super) fn l2_counters(&self) -> Option<&L2Counters> {
        match self {
            Self::Full(result) => result.l2(),
            #[cfg(feature = "campaign")]
            Self::SampledObserved(result) => result.counters().l2(),
        }
    }

    /// Consume a coverage-classified result only when it represents the full
    /// launch. Keeping this match inside `sim-sm` makes ordinary callers stable
    /// when Cargo feature unification enables the campaign-only sampled
    /// variant elsewhere in the workspace.
    pub fn into_full(self) -> Option<FullKernelReplayResult> {
        match self {
            Self::Full(result) => Some(result),
            #[cfg(feature = "campaign")]
            Self::SampledObserved(_) => None,
        }
    }
}

/// Read-only diagnostics from raw trace/sidecar inputs. This type deliberately
/// cannot be converted into [`FullKernelReplayResult`] or passed to
/// [`aggregate_full_launch_set`]; only [`replay_kernel`] over
/// [`AdmittedLaunchEvidence`] can mint an aggregatable replay result.
///
/// ```compile_fail
/// use sim_sm::{
///     FullLaunchSetDeclaration, UnadmittedReplayDiagnostics,
///     aggregate_full_launch_set,
/// };
///
/// fn reject(
///     declaration: FullLaunchSetDeclaration,
///     diagnostic: UnadmittedReplayDiagnostics,
/// ) {
///     let _ = aggregate_full_launch_set(declaration, [diagnostic]);
/// }
/// ```
#[cfg(feature = "campaign")]
#[derive(Debug, Clone)]
pub struct UnadmittedReplayDiagnostics {
    pub(super) coverage: UnadmittedCoverage,
    pub(super) configuration: ReplayConfiguration,
    pub(super) arch: ArchId,
    pub(super) blocks_observed: u64,
    pub(super) population_blocks: u64,
    pub(super) instructions_issued: u64,
    pub(super) records_seen: u64,
    pub(super) per_sm_count: Option<usize>,
    pub(super) diagnostic_last_issue_cycles: Option<u64>,
    pub(super) terminal_modeled_event_time: Option<u64>,
    pub(super) l1: Option<L1Counters>,
    pub(super) l2: Option<L2Counters>,
    pub(super) l2_by_source: Option<L2SourceSplit>,
    pub(super) dram: Option<DramStats>,
    pub(super) unmodeled_tails: TailFlags,
}

#[cfg(feature = "campaign")]
impl UnadmittedReplayDiagnostics {
    pub(super) fn from_outcome(outcome: KernelReplayOutcome) -> Self {
        match outcome {
            KernelReplayOutcome::Full(result) => Self {
                coverage: UnadmittedCoverage::Full,
                configuration: result.configuration(),
                arch: result.arch(),
                blocks_observed: result.coverage().blocks_observed(),
                population_blocks: result.coverage().blocks_observed(),
                instructions_issued: result.instructions_issued(),
                records_seen: result.records_seen(),
                per_sm_count: Some(result.per_sm().len()),
                diagnostic_last_issue_cycles: Some(result.diagnostic_last_issue_cycles()),
                terminal_modeled_event_time: result.terminal_modeled_event_time(),
                l1: result.l1().copied(),
                l2: result.l2().copied(),
                l2_by_source: result.l2_by_source().copied(),
                dram: result.dram().copied(),
                unmodeled_tails: result.unmodeled_tails(),
            },
            KernelReplayOutcome::SampledObserved(result) => {
                let counters = result.counters();
                Self {
                    coverage: UnadmittedCoverage::SampledObserved,
                    configuration: result.configuration(),
                    arch: counters.arch(),
                    blocks_observed: result.coverage().blocks_observed(),
                    population_blocks: result.coverage().population_blocks(),
                    instructions_issued: counters.instructions_issued(),
                    records_seen: counters.records_seen(),
                    per_sm_count: None,
                    diagnostic_last_issue_cycles: None,
                    terminal_modeled_event_time: None,
                    l1: counters.l1().copied(),
                    l2: counters.l2().copied(),
                    l2_by_source: counters.l2_by_source().copied(),
                    dram: counters.dram().copied(),
                    unmodeled_tails: counters.unmodeled_tails(),
                }
            }
        }
    }

    pub fn coverage(&self) -> UnadmittedCoverage {
        self.coverage
    }

    pub fn configuration(&self) -> ReplayConfiguration {
        self.configuration
    }

    pub fn arch(&self) -> ArchId {
        self.arch
    }

    pub fn blocks_observed(&self) -> u64 {
        self.blocks_observed
    }

    pub fn population_blocks(&self) -> u64 {
        self.population_blocks
    }

    pub fn instructions_issued(&self) -> u64 {
        self.instructions_issued
    }

    pub fn records_seen(&self) -> u64 {
        self.records_seen
    }

    pub fn per_sm_count(&self) -> Option<usize> {
        self.per_sm_count
    }

    /// Present only for full-coverage diagnostics. This remains an issue-frontier
    /// diagnostic, never a publication-cycle prediction.
    pub fn diagnostic_last_issue_cycles(&self) -> Option<u64> {
        self.diagnostic_last_issue_cycles
    }

    pub fn terminal_modeled_event_time(&self) -> Option<u64> {
        self.terminal_modeled_event_time
    }

    pub fn l1(&self) -> Option<&L1Counters> {
        self.l1.as_ref()
    }

    pub fn l2(&self) -> Option<&L2Counters> {
        self.l2.as_ref()
    }

    pub fn l2_by_source(&self) -> Option<&L2SourceSplit> {
        self.l2_by_source.as_ref()
    }

    pub fn dram(&self) -> Option<&DramStats> {
        self.dram.as_ref()
    }

    pub fn unmodeled_tails(&self) -> TailFlags {
        self.unmodeled_tails
    }
}
