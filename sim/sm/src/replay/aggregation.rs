//! Checked aggregation for exact, full-coverage kernel replay results.
//!
//! This module deliberately accepts only [`FullKernelReplayResult`]. Sampled
//! observations have no conversion into this algebra. Launch-set aggregation
//! retains temporal values only as a canonical per-launch diagnostic vector; it
//! never composes them into a launch-set timing scalar and exposes no
//! aggregate-to-aggregate addition surface.

use std::collections::{BTreeMap, BTreeSet, btree_map::Entry};

use sim_arch::ArchId;
use sim_mem::{L1CounterError, L1Counters, L2CounterError, L2Counters};

use crate::issue::TailFlags;
use crate::system::{DramStats, L2SourceSplit, SystemCounterError};

use super::{
    FrontEndCounters, FullKernelReplayResult, FullLaunchCoverage, ModeledCompletionFrontier,
    ReplayConfiguration, ReplayLaunchCoverageBinding, ReplayMemMode,
};

/// Declarative launch identity failures. This constructor creates an expected
/// roster entry, never an observed-coverage witness.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum ReplayLaunchCoverageBindingError {
    #[error("kernel_mangled_name must be nonempty")]
    EmptyKernelName,
    #[error("launch grid dimensions must all be positive (got {grid:?})")]
    DegenerateGrid { grid: (u32, u32, u32) },
    #[error("launch grid block count overflows u64 (grid={grid:?})")]
    GridSizeOverflow { grid: (u32, u32, u32) },
    #[error("launch block dimensions, when present, must all be positive (got {block:?})")]
    DegenerateBlock { block: (u32, u32, u32) },
    #[error("launch cluster dimensions must all be positive (got {cluster:?})")]
    DegenerateCluster { cluster: (u32, u32, u32) },
}

impl ReplayLaunchCoverageBinding {
    /// Construct a declaration-side launch identity. This does not create a
    /// coverage witness; only replay extraction can create [`FullLaunchCoverage`].
    #[allow(clippy::too_many_arguments)]
    pub fn checked(
        device_id: i64,
        stream_id: i64,
        kernel_id: i32,
        function_unique_id: i32,
        kernel_mangled_name: impl Into<String>,
        grid: (u32, u32, u32),
        block: Option<(u32, u32, u32)>,
        cluster: (u32, u32, u32),
    ) -> Result<Self, ReplayLaunchCoverageBindingError> {
        let binding = Self {
            device_id,
            stream_id,
            kernel_id,
            function_unique_id,
            kernel_mangled_name: kernel_mangled_name.into(),
            grid,
            block,
            cluster,
        };
        binding.validate()?;
        Ok(binding)
    }

    /// Checked block population implied by this launch identity.
    pub fn grid_blocks(&self) -> Result<u64, ReplayLaunchCoverageBindingError> {
        self.validate()
    }

    fn validate(&self) -> Result<u64, ReplayLaunchCoverageBindingError> {
        if self.kernel_mangled_name.is_empty() {
            return Err(ReplayLaunchCoverageBindingError::EmptyKernelName);
        }
        if self.grid.0 == 0 || self.grid.1 == 0 || self.grid.2 == 0 {
            return Err(ReplayLaunchCoverageBindingError::DegenerateGrid { grid: self.grid });
        }
        if let Some(block) = self.block
            && (block.0 == 0 || block.1 == 0 || block.2 == 0)
        {
            return Err(ReplayLaunchCoverageBindingError::DegenerateBlock { block });
        }
        if self.cluster.0 == 0 || self.cluster.1 == 0 || self.cluster.2 == 0 {
            return Err(ReplayLaunchCoverageBindingError::DegenerateCluster {
                cluster: self.cluster,
            });
        }
        u64::from(self.grid.0)
            .checked_mul(u64::from(self.grid.1))
            .and_then(|value| value.checked_mul(u64::from(self.grid.2)))
            .ok_or(ReplayLaunchCoverageBindingError::GridSizeOverflow { grid: self.grid })
    }
}

/// Canonical launch-set scope token. Scope is a label only; it conveys no
/// scientific claim.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct LaunchSetScope(String);

impl LaunchSetScope {
    /// Accept one nonempty ASCII-graphic token without normalization. Rejecting
    /// whitespace and non-ASCII spellings prevents multiple visual forms from
    /// silently naming the same orchestration scope.
    pub fn checked(value: impl Into<String>) -> Result<Self, LaunchSetScopeError> {
        let value = value.into();
        if value.is_empty() {
            return Err(LaunchSetScopeError::Empty);
        }
        if !value.bytes().all(|byte| byte.is_ascii_graphic()) {
            return Err(LaunchSetScopeError::NonCanonical { value });
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum LaunchSetScopeError {
    #[error("launch-set scope must be nonempty")]
    Empty,
    #[error("launch-set scope must contain only ASCII graphic bytes (got {value:?})")]
    NonCanonical { value: String },
}

/// Checked declaration of one exact launch-set roster under one replay
/// configuration. Expected launch order is canonicalized by exact identity.
#[derive(Debug, Clone)]
pub struct FullLaunchSetDeclaration {
    scope: LaunchSetScope,
    configuration: ReplayConfiguration,
    expected_launches: Vec<ReplayLaunchCoverageBinding>,
}

impl FullLaunchSetDeclaration {
    pub fn checked(
        scope: impl Into<String>,
        configuration: ReplayConfiguration,
        expected_launches: impl IntoIterator<Item = ReplayLaunchCoverageBinding>,
    ) -> Result<Self, FullLaunchSetDeclarationError> {
        let scope = LaunchSetScope::checked(scope)?;
        let mut roster = BTreeSet::new();
        for launch in expected_launches {
            launch.validate().map_err(|source| {
                FullLaunchSetDeclarationError::InvalidExpectedLaunch {
                    launch: launch.clone(),
                    source,
                }
            })?;
            if !roster.insert(launch.clone()) {
                return Err(FullLaunchSetDeclarationError::DuplicateExpectedLaunch { launch });
            }
        }
        if roster.is_empty() {
            return Err(FullLaunchSetDeclarationError::EmptyExpectedRoster);
        }
        Ok(Self {
            scope,
            configuration,
            expected_launches: roster.into_iter().collect(),
        })
    }

    pub fn scope(&self) -> &LaunchSetScope {
        &self.scope
    }

    pub fn configuration(&self) -> ReplayConfiguration {
        self.configuration
    }

    pub fn arch(&self) -> ArchId {
        self.configuration.arch()
    }

    pub fn expected_launches(&self) -> &[ReplayLaunchCoverageBinding] {
        &self.expected_launches
    }
}

#[derive(Debug, thiserror::Error)]
pub enum FullLaunchSetDeclarationError {
    #[error(transparent)]
    Scope(#[from] LaunchSetScopeError),
    #[error("full launch-set declaration must contain at least one expected launch")]
    EmptyExpectedRoster,
    #[error("duplicate expected launch identity {launch:?}")]
    DuplicateExpectedLaunch { launch: ReplayLaunchCoverageBinding },
    #[error("invalid expected launch identity {launch:?}: {source}")]
    InvalidExpectedLaunch {
        launch: ReplayLaunchCoverageBinding,
        #[source]
        source: ReplayLaunchCoverageBindingError,
    },
}

/// Provenance of the aggregate's counter estimator. The only admitted variant
/// is exact aggregation of full-coverage replay; sampled estimation remains a
/// separate, unavailable producer contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FullCounterEstimatorProvenance {
    ExactFullCoverageReplay,
}

/// Temporal diagnostics for one full-coverage launch. These values remain bound
/// to the exact launch identity and are carried in canonical roster order. They
/// are never composed into a launch-set cycle prediction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FullLaunchTimingDiagnostic {
    launch: ReplayLaunchCoverageBinding,
    diagnostic_last_issue_cycles: u64,
    last_processed_modeled_system_event_time: Option<u64>,
    terminal_modeled_event_time: Option<u64>,
    modeled_completion_frontier: ModeledCompletionFrontier,
}

impl FullLaunchTimingDiagnostic {
    pub fn launch(&self) -> &ReplayLaunchCoverageBinding {
        &self.launch
    }

    /// Maximum per-SM issue frontier (`last local issue timestamp + 1`).
    pub fn diagnostic_last_issue_cycles(&self) -> u64 {
        self.diagnostic_last_issue_cycles
    }

    pub fn last_processed_modeled_system_event_time(&self) -> Option<u64> {
        self.last_processed_modeled_system_event_time
    }

    /// Raw timestamp of the final successfully processed modeled event.
    pub fn terminal_modeled_event_time(&self) -> Option<u64> {
        self.terminal_modeled_event_time
    }

    pub fn modeled_completion_frontier(&self) -> ModeledCompletionFrontier {
        self.modeled_completion_frontier
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaunchSetCounterComponent {
    L1,
    L2,
    L2BySource,
    Dram,
}

impl std::fmt::Display for LaunchSetCounterComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::L1 => "L1",
            Self::L2 => "L2",
            Self::L2BySource => "L2-by-source",
            Self::Dram => "DRAM",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaunchSetScalarCounter {
    InstructionsIssued,
    RecordsSeen,
    CompulsoryInstructionReadSectors,
}

impl std::fmt::Display for LaunchSetScalarCounter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::InstructionsIssued => "instructions_issued",
            Self::RecordsSeen => "records_seen",
            Self::CompulsoryInstructionReadSectors => "compulsory_instruction_read_sectors",
        })
    }
}

/// Counter-only, full-coverage launch-set result. The constructor and fields are
/// private. Temporal replay values survive only in `launch_timing_diagnostics`;
/// no public API composes them or attaches a launch-set timing scalar.
#[derive(Debug)]
pub struct FullLaunchSetCounters {
    scope: LaunchSetScope,
    configuration: ReplayConfiguration,
    launch_coverages: Vec<FullLaunchCoverage>,
    launch_timing_diagnostics: Vec<FullLaunchTimingDiagnostic>,
    estimator_provenance: FullCounterEstimatorProvenance,
    instructions_issued: u64,
    records_seen: u64,
    front_end: FrontEndCounters,
    l1: Option<L1Counters>,
    l2: Option<L2Counters>,
    l2_by_source: Option<L2SourceSplit>,
    dram: Option<DramStats>,
    unmodeled_tails: TailFlags,
}

impl FullLaunchSetCounters {
    pub fn scope(&self) -> &LaunchSetScope {
        &self.scope
    }

    pub fn configuration(&self) -> ReplayConfiguration {
        self.configuration
    }

    pub fn arch(&self) -> ArchId {
        self.configuration.arch()
    }

    pub fn launch_coverages(&self) -> &[FullLaunchCoverage] {
        &self.launch_coverages
    }

    pub fn launch_timing_diagnostics(&self) -> &[FullLaunchTimingDiagnostic] {
        &self.launch_timing_diagnostics
    }

    pub fn estimator_provenance(&self) -> FullCounterEstimatorProvenance {
        self.estimator_provenance
    }

    pub fn instructions_issued(&self) -> u64 {
        self.instructions_issued
    }

    pub fn records_seen(&self) -> u64 {
        self.records_seen
    }

    pub fn front_end(&self) -> FrontEndCounters {
        self.front_end
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

#[derive(Debug, thiserror::Error)]
pub enum LaunchSetAggregationError {
    #[error("full launch-set declaration unexpectedly has an empty expected roster")]
    EmptyExpectedRoster,
    #[error("unexpected full replay launch {launch:?}")]
    UnexpectedLaunch { launch: ReplayLaunchCoverageBinding },
    #[error("duplicate full replay launch {launch:?}")]
    DuplicateLaunch { launch: ReplayLaunchCoverageBinding },
    #[error("missing expected full replay launch {launch:?}")]
    MissingLaunch { launch: ReplayLaunchCoverageBinding },
    #[error("launch {launch:?} architecture mismatch: expected {expected}, got {got}")]
    ArchitectureMismatch {
        launch: ReplayLaunchCoverageBinding,
        expected: ArchId,
        got: ArchId,
    },
    #[error("launch {launch:?} replay configuration mismatch: expected {expected:?}, got {got:?}")]
    ConfigurationMismatch {
        launch: ReplayLaunchCoverageBinding,
        expected: ReplayConfiguration,
        got: ReplayConfiguration,
    },
    #[error("launch {launch:?} has invalid full-coverage identity: {source}")]
    InvalidCoverageIdentity {
        launch: ReplayLaunchCoverageBinding,
        #[source]
        source: ReplayLaunchCoverageBindingError,
    },
    #[error(
        "launch {launch:?} full coverage mismatch: expected {expected_blocks} blocks, observed {observed_blocks}"
    )]
    CoverageMismatch {
        launch: ReplayLaunchCoverageBinding,
        expected_blocks: u64,
        observed_blocks: u64,
    },
    #[error(
        "launch {launch:?} issue/record count mismatch: issued={instructions_issued}, records={records_seen}"
    )]
    CountMismatch {
        launch: ReplayLaunchCoverageBinding,
        instructions_issued: u64,
        records_seen: u64,
    },
    #[error(
        "launch {launch:?} {component} counter presence is incompatible with {mode:?}: expected={expected}, got={got}"
    )]
    CounterPresenceMismatch {
        launch: ReplayLaunchCoverageBinding,
        component: LaunchSetCounterComponent,
        mode: ReplayMemMode,
        expected: bool,
        got: bool,
    },
    #[error(
        "launch {launch:?} {component} counter architecture mismatch: expected {expected}, got {got:?}"
    )]
    CounterArchitectureMismatch {
        launch: ReplayLaunchCoverageBinding,
        component: LaunchSetCounterComponent,
        expected: ArchId,
        got: Option<ArchId>,
    },
    #[error("launch {launch:?} {counter} overflows u64 during launch-set aggregation")]
    ScalarOverflow {
        launch: ReplayLaunchCoverageBinding,
        counter: LaunchSetScalarCounter,
    },
    #[error("launch {launch:?} L1 counter validation/aggregation failed: {source}")]
    L1Counter {
        launch: ReplayLaunchCoverageBinding,
        #[source]
        source: L1CounterError,
    },
    #[error("launch {launch:?} L2 counter validation/aggregation failed: {source}")]
    L2Counter {
        launch: ReplayLaunchCoverageBinding,
        #[source]
        source: L2CounterError,
    },
    #[error("launch {launch:?} L2-by-source validation/aggregation failed: {source}")]
    L2BySourceCounter {
        launch: ReplayLaunchCoverageBinding,
        #[source]
        source: SystemCounterError,
    },
    #[error("launch {launch:?} DRAM counter validation/aggregation failed: {source}")]
    DramCounter {
        launch: ReplayLaunchCoverageBinding,
        #[source]
        source: SystemCounterError,
    },
    #[error(
        "launch {launch:?} L2 source {counter} total overflows u64 while checking physical/source coherence"
    )]
    L2SourceTotalOverflow {
        launch: ReplayLaunchCoverageBinding,
        counter: &'static str,
    },
    #[error(
        "launch {launch:?} L2 physical/source {counter} mismatch: physical={physical}, by_source={by_source}"
    )]
    L2SourceMismatch {
        launch: ReplayLaunchCoverageBinding,
        counter: &'static str,
        physical: u64,
        by_source: u64,
    },
    #[error(
        "launch {launch:?} L2 miss partition overflows u64: line={line_misses}, sector_fill={sector_fill_misses}"
    )]
    L2MissPartitionOverflow {
        launch: ReplayLaunchCoverageBinding,
        line_misses: u64,
        sector_fill_misses: u64,
    },
    #[error(
        "launch {launch:?} L2 miss partition mismatch: misses={misses}, line={line_misses}, sector_fill={sector_fill_misses}"
    )]
    L2MissPartitionMismatch {
        launch: ReplayLaunchCoverageBinding,
        misses: u64,
        line_misses: u64,
        sector_fill_misses: u64,
    },
    #[error(
        "launch {launch:?} L2 write-tag hits exceed all hits: write_tag_hits={write_tag_hits}, hits={hits}"
    )]
    L2WriteTagHitSubsetMismatch {
        launch: ReplayLaunchCoverageBinding,
        write_tag_hits: u64,
        hits: u64,
    },
}

#[derive(Clone, Copy)]
struct CounterShape {
    l1: bool,
    l2: bool,
    l2_by_source: bool,
    dram: bool,
}

fn counter_shape(mode: ReplayMemMode) -> CounterShape {
    match mode {
        ReplayMemMode::Flat => CounterShape {
            l1: false,
            l2: false,
            l2_by_source: false,
            dram: false,
        },
        ReplayMemMode::L1Const(_) => CounterShape {
            l1: true,
            l2: false,
            l2_by_source: false,
            dram: false,
        },
        ReplayMemMode::L1SharedL2 { .. } => CounterShape {
            l1: true,
            l2: true,
            l2_by_source: true,
            dram: true,
        },
    }
}

fn require_presence(
    launch: &ReplayLaunchCoverageBinding,
    mode: ReplayMemMode,
    component: LaunchSetCounterComponent,
    expected: bool,
    got: bool,
) -> Result<(), LaunchSetAggregationError> {
    if expected != got {
        return Err(LaunchSetAggregationError::CounterPresenceMismatch {
            launch: launch.clone(),
            component,
            mode,
            expected,
            got,
        });
    }
    Ok(())
}

fn require_counter_arch(
    launch: &ReplayLaunchCoverageBinding,
    component: LaunchSetCounterComponent,
    expected: ArchId,
    got: Option<ArchId>,
) -> Result<(), LaunchSetAggregationError> {
    if got != Some(expected) {
        return Err(LaunchSetAggregationError::CounterArchitectureMismatch {
            launch: launch.clone(),
            component,
            expected,
            got,
        });
    }
    Ok(())
}

fn checked_source_sum(
    launch: &ReplayLaunchCoverageBinding,
    counter: &'static str,
    values: [u64; 4],
) -> Result<u64, LaunchSetAggregationError> {
    values.into_iter().try_fold(0_u64, |total, value| {
        total
            .checked_add(value)
            .ok_or_else(|| LaunchSetAggregationError::L2SourceTotalOverflow {
                launch: launch.clone(),
                counter,
            })
    })
}

fn validate_l2_coherence(
    launch: &ReplayLaunchCoverageBinding,
    l2: &L2Counters,
    by_source: &L2SourceSplit,
) -> Result<(), LaunchSetAggregationError> {
    let physical_sectors = l2
        .total()
        .map_err(|source| LaunchSetAggregationError::L2Counter {
            launch: launch.clone(),
            source,
        })?;
    let source_sectors = checked_source_sum(launch, "sectors", by_source.sectors)?;
    if physical_sectors != source_sectors {
        return Err(LaunchSetAggregationError::L2SourceMismatch {
            launch: launch.clone(),
            counter: "sectors",
            physical: physical_sectors,
            by_source: source_sectors,
        });
    }
    let source_hits = checked_source_sum(launch, "hits", by_source.hits)?;
    if l2.hits != source_hits {
        return Err(LaunchSetAggregationError::L2SourceMismatch {
            launch: launch.clone(),
            counter: "hits",
            physical: l2.hits,
            by_source: source_hits,
        });
    }
    let partition = l2
        .line_misses
        .checked_add(l2.sector_fill_misses)
        .ok_or_else(|| LaunchSetAggregationError::L2MissPartitionOverflow {
            launch: launch.clone(),
            line_misses: l2.line_misses,
            sector_fill_misses: l2.sector_fill_misses,
        })?;
    if l2.misses != partition {
        return Err(LaunchSetAggregationError::L2MissPartitionMismatch {
            launch: launch.clone(),
            misses: l2.misses,
            line_misses: l2.line_misses,
            sector_fill_misses: l2.sector_fill_misses,
        });
    }
    if l2.write_tag_hits > l2.hits {
        return Err(LaunchSetAggregationError::L2WriteTagHitSubsetMismatch {
            launch: launch.clone(),
            write_tag_hits: l2.write_tag_hits,
            hits: l2.hits,
        });
    }
    Ok(())
}

fn validate_full_result(
    launch: &ReplayLaunchCoverageBinding,
    configuration: ReplayConfiguration,
    result: &FullKernelReplayResult,
) -> Result<(), LaunchSetAggregationError> {
    let expected_arch = configuration.arch();
    if result.arch() != expected_arch {
        return Err(LaunchSetAggregationError::ArchitectureMismatch {
            launch: launch.clone(),
            expected: expected_arch,
            got: result.arch(),
        });
    }
    if result.configuration() != configuration {
        return Err(LaunchSetAggregationError::ConfigurationMismatch {
            launch: launch.clone(),
            expected: configuration,
            got: result.configuration(),
        });
    }

    let expected_blocks = result.coverage().launch().grid_blocks().map_err(|source| {
        LaunchSetAggregationError::InvalidCoverageIdentity {
            launch: launch.clone(),
            source,
        }
    })?;
    if result.coverage().blocks_observed() != expected_blocks {
        return Err(LaunchSetAggregationError::CoverageMismatch {
            launch: launch.clone(),
            expected_blocks,
            observed_blocks: result.coverage().blocks_observed(),
        });
    }
    if result.instructions_issued() != result.records_seen() {
        return Err(LaunchSetAggregationError::CountMismatch {
            launch: launch.clone(),
            instructions_issued: result.instructions_issued(),
            records_seen: result.records_seen(),
        });
    }

    let mode = configuration.memory_mode();
    let shape = counter_shape(mode);
    require_presence(
        launch,
        mode,
        LaunchSetCounterComponent::L1,
        shape.l1,
        result.l1().is_some(),
    )?;
    require_presence(
        launch,
        mode,
        LaunchSetCounterComponent::L2,
        shape.l2,
        result.l2().is_some(),
    )?;
    require_presence(
        launch,
        mode,
        LaunchSetCounterComponent::L2BySource,
        shape.l2_by_source,
        result.l2_by_source().is_some(),
    )?;
    require_presence(
        launch,
        mode,
        LaunchSetCounterComponent::Dram,
        shape.dram,
        result.dram().is_some(),
    )?;

    if let Some(l1) = result.l1() {
        require_counter_arch(
            launch,
            LaunchSetCounterComponent::L1,
            expected_arch,
            l1.arch(),
        )?;
        l1.validate_origin_partition()
            .map_err(|source| LaunchSetAggregationError::L1Counter {
                launch: launch.clone(),
                source,
            })?;
    }
    if let Some(l2) = result.l2() {
        require_counter_arch(
            launch,
            LaunchSetCounterComponent::L2,
            expected_arch,
            l2.arch(),
        )?;
    }
    if let Some(split) = result.l2_by_source() {
        require_counter_arch(
            launch,
            LaunchSetCounterComponent::L2BySource,
            expected_arch,
            split.arch(),
        )?;
        split.validate_origin_partition().map_err(|source| {
            LaunchSetAggregationError::L2BySourceCounter {
                launch: launch.clone(),
                source,
            }
        })?;
    }
    if let Some(dram) = result.dram() {
        require_counter_arch(
            launch,
            LaunchSetCounterComponent::Dram,
            expected_arch,
            dram.arch(),
        )?;
        dram.validate_origin_partition().map_err(|source| {
            LaunchSetAggregationError::DramCounter {
                launch: launch.clone(),
                source,
            }
        })?;
    }
    if let (Some(l2), Some(split)) = (result.l2(), result.l2_by_source()) {
        validate_l2_coherence(launch, l2, split)?;
    }
    Ok(())
}

/// Aggregate an exact declared roster of full-coverage kernel results.
///
/// All roster, architecture, configuration, coverage, shape, and provenance
/// checks complete before arithmetic. Aggregation order is canonical expected-
/// roster order, independent of caller iteration order. Temporal values are
/// retained one row per launch and are never reduced into a timing prediction.
///
/// Sampled results are statically outside this API:
///
/// ```compile_fail
/// use sim_sm::{
///     FullLaunchSetDeclaration, SampledKernelReplayResult, aggregate_full_launch_set,
/// };
///
/// fn reject_sampled(
///     declaration: FullLaunchSetDeclaration,
///     sampled: SampledKernelReplayResult,
/// ) {
///     let _ = aggregate_full_launch_set(declaration, [sampled]);
/// }
/// ```
pub fn aggregate_full_launch_set(
    declaration: FullLaunchSetDeclaration,
    results: impl IntoIterator<Item = FullKernelReplayResult>,
) -> Result<FullLaunchSetCounters, LaunchSetAggregationError> {
    if declaration.expected_launches.is_empty() {
        return Err(LaunchSetAggregationError::EmptyExpectedRoster);
    }
    let mut actual = BTreeMap::new();
    for result in results {
        let launch = result.coverage().launch().clone();
        if declaration
            .expected_launches
            .binary_search(&launch)
            .is_err()
        {
            return Err(LaunchSetAggregationError::UnexpectedLaunch { launch });
        }
        match actual.entry(launch.clone()) {
            Entry::Occupied(_) => {
                return Err(LaunchSetAggregationError::DuplicateLaunch { launch });
            }
            Entry::Vacant(entry) => {
                entry.insert(result);
            }
        }
    }
    for launch in &declaration.expected_launches {
        if !actual.contains_key(launch) {
            return Err(LaunchSetAggregationError::MissingLaunch {
                launch: launch.clone(),
            });
        }
    }
    for launch in &declaration.expected_launches {
        let result =
            actual
                .get(launch)
                .ok_or_else(|| LaunchSetAggregationError::MissingLaunch {
                    launch: launch.clone(),
                })?;
        validate_full_result(launch, declaration.configuration, result)?;
    }

    let arch = declaration.configuration.arch();
    let shape = counter_shape(declaration.configuration.memory_mode());
    let mut instructions_issued = 0_u64;
    let mut records_seen = 0_u64;
    let mut front_end = FrontEndCounters::default();
    let mut l1 = shape.l1.then(|| L1Counters::for_arch(arch));
    let mut l2 = shape.l2.then(|| L2Counters::for_arch(arch));
    let mut l2_by_source = shape.l2_by_source.then(|| L2SourceSplit::for_arch(arch));
    let mut dram = shape.dram.then(|| DramStats::for_arch(arch));
    let mut unmodeled_tails = TailFlags::default();
    let mut launch_coverages = Vec::with_capacity(declaration.expected_launches.len());
    let mut launch_timing_diagnostics = Vec::with_capacity(declaration.expected_launches.len());

    for launch in &declaration.expected_launches {
        let result =
            actual
                .get(launch)
                .ok_or_else(|| LaunchSetAggregationError::MissingLaunch {
                    launch: launch.clone(),
                })?;
        instructions_issued = instructions_issued
            .checked_add(result.instructions_issued())
            .ok_or_else(|| LaunchSetAggregationError::ScalarOverflow {
                launch: launch.clone(),
                counter: LaunchSetScalarCounter::InstructionsIssued,
            })?;
        records_seen = records_seen
            .checked_add(result.records_seen())
            .ok_or_else(|| LaunchSetAggregationError::ScalarOverflow {
                launch: launch.clone(),
                counter: LaunchSetScalarCounter::RecordsSeen,
            })?;
        front_end = front_end.add_checked(result.front_end()).ok_or_else(|| {
            LaunchSetAggregationError::ScalarOverflow {
                launch: launch.clone(),
                counter: LaunchSetScalarCounter::CompulsoryInstructionReadSectors,
            }
        })?;
        if let (Some(total), Some(addend)) = (&mut l1, result.l1()) {
            total
                .add_checked(addend)
                .map_err(|source| LaunchSetAggregationError::L1Counter {
                    launch: launch.clone(),
                    source,
                })?;
        }
        if let (Some(total), Some(addend)) = (&mut l2, result.l2()) {
            total
                .add_checked(addend)
                .map_err(|source| LaunchSetAggregationError::L2Counter {
                    launch: launch.clone(),
                    source,
                })?;
        }
        if let (Some(total), Some(addend)) = (&mut l2_by_source, result.l2_by_source()) {
            total.add_checked(addend).map_err(|source| {
                LaunchSetAggregationError::L2BySourceCounter {
                    launch: launch.clone(),
                    source,
                }
            })?;
        }
        if let (Some(total), Some(addend)) = (&mut dram, result.dram()) {
            total
                .add_checked(addend)
                .map_err(|source| LaunchSetAggregationError::DramCounter {
                    launch: launch.clone(),
                    source,
                })?;
        }
        unmodeled_tails = unmodeled_tails.union(result.unmodeled_tails());
        launch_coverages.push(result.coverage().clone());
        launch_timing_diagnostics.push(FullLaunchTimingDiagnostic {
            launch: launch.clone(),
            diagnostic_last_issue_cycles: result.diagnostic_last_issue_cycles(),
            last_processed_modeled_system_event_time: result
                .last_processed_modeled_system_event_time(),
            terminal_modeled_event_time: result.terminal_modeled_event_time(),
            modeled_completion_frontier: result.modeled_completion_frontier(),
        });
    }

    if let (Some(l2), Some(split)) = (&l2, &l2_by_source) {
        // This also catches a total overflow introduced only by cross-launch
        // addition, after each launch passed the prevalidation above.
        let last_launch = declaration
            .expected_launches
            .last()
            .ok_or(LaunchSetAggregationError::EmptyExpectedRoster)?;
        validate_l2_coherence(last_launch, l2, split)?;
    }

    Ok(FullLaunchSetCounters {
        scope: declaration.scope,
        configuration: declaration.configuration,
        launch_coverages,
        launch_timing_diagnostics,
        estimator_provenance: FullCounterEstimatorProvenance::ExactFullCoverageReplay,
        instructions_issued,
        records_seen,
        front_end,
        l1,
        l2,
        l2_by_source,
        dram,
        unmodeled_tails,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::num::NonZeroU32;

    use sim_arch::{ArchId, LdgstsCompletionThroughput};
    use sim_mem::{
        L1Config, L1Counters, L2Config, L2Counters, L2Replacement, MemoryOrigin, SliceHash,
    };

    use crate::issue::{IssueModelParams, IssueResourceSource, TailFlags};
    use crate::system::DramBw;

    use super::*;
    use crate::replay::{FoldedKernelReplay, FullCoverageSource};

    fn params(arch: ArchId) -> IssueModelParams {
        IssueModelParams {
            arch,
            resource_source: IssueResourceSource::SyntheticTestFixture,
            instruction_front_end: None,
            memory_issue_pipeline: None,
            l1_hit_latency: 2,
            l1_miss_latency: 20,
            global_flat_latency: 100,
            shared_write_barrier_latency: 4,
            shared_additional_wavefront_latency: 2,
            non_memory_write_barrier_latency: 20,
            read_barrier_latency: 3,
            ldgsts_l1_mshr_floor: NonZeroU32::new(1).unwrap(),
            ldgsts_completion_throughput: LdgstsCompletionThroughput {
                cache_all_payload_bytes_per_cycle: 64,
                bypass_payload_bytes_per_cycle: 32,
            },
            ldgsts_cache_all_destination_wavefront_cycles: 2,
            ldgsts_global_to_shared_transport_latency: 0,
            ldgsts_cache_all_destination_service_wavefronts: 2,
            ldgsts_cache_all_destination_service_cycles: 3,
            max_resident_warps_per_sm: 48,
        }
    }

    fn flat_configuration() -> ReplayConfiguration {
        ReplayConfiguration::checked(params(ArchId::Sm86), ReplayMemMode::Flat).unwrap()
    }

    fn l1_configuration() -> ReplayConfiguration {
        ReplayConfiguration::checked(
            params(ArchId::Sm86),
            ReplayMemMode::L1Const(L1Config {
                arch: ArchId::Sm86,
                line_capacity: 8,
                mshr_count: Some(4),
                hit_latency: 2,
                miss_latency: 20,
            }),
        )
        .unwrap()
    }

    fn shared_configuration() -> ReplayConfiguration {
        ReplayConfiguration::checked(
            params(ArchId::Sm86),
            ReplayMemMode::L1SharedL2 {
                l1: L1Config {
                    arch: ArchId::Sm86,
                    line_capacity: 8,
                    mshr_count: Some(4),
                    hit_latency: 2,
                    miss_latency: 20,
                },
                l2: L2Config {
                    arch: ArchId::Sm86,
                    capacity_bytes: 128 * 16,
                    line_bytes: 128,
                    sector_bytes: 32,
                    num_slices: 1,
                    associativity: 16,
                    replacement: L2Replacement::Lru,
                    slice_hash: SliceHash::LineMod,
                },
                dram: DramBw::Unmodeled { arch: ArchId::Sm86 },
            },
        )
        .unwrap()
    }

    fn binding(kernel_id: i32) -> ReplayLaunchCoverageBinding {
        ReplayLaunchCoverageBinding::checked(
            0,
            7,
            kernel_id,
            kernel_id + 100,
            format!("_Zkernel{kernel_id}"),
            (1, 1, 1),
            Some((32, 1, 1)),
            (1, 1, 1),
        )
        .unwrap()
    }

    fn full_result(
        launch: ReplayLaunchCoverageBinding,
        configuration: ReplayConfiguration,
        instructions_issued: u64,
        records_seen: u64,
    ) -> FullKernelReplayResult {
        let arch = configuration.arch();
        let (l1, l2, l2_by_source, dram) = match configuration.memory_mode() {
            ReplayMemMode::Flat => (None, None, None, None),
            ReplayMemMode::L1Const(_) => (Some(L1Counters::for_arch(arch)), None, None, None),
            ReplayMemMode::L1SharedL2 { .. } => (
                Some(L1Counters::for_arch(arch)),
                Some(L2Counters::for_arch(arch)),
                Some(L2SourceSplit::for_arch(arch)),
                Some(DramStats::for_arch(arch)),
            ),
        };
        FullKernelReplayResult {
            coverage: FullLaunchCoverage {
                launch,
                blocks_observed: 1,
                source: FullCoverageSource::UnsampledComplete,
            },
            configuration,
            replay: FoldedKernelReplay {
                arch,
                instructions_issued,
                records_seen,
                per_sm: BTreeMap::new(),
                diagnostic_last_issue_cycles: 999,
                last_processed_system_event_time: Some(1_234),
                terminal_modeled_event_time: Some(1_234),
                l1,
                l2,
                l2_by_source,
                dram,
                front_end: FrontEndCounters::default(),
                unmodeled_tails: TailFlags::default(),
            },
        }
    }

    fn declaration(
        configuration: ReplayConfiguration,
        launches: impl IntoIterator<Item = ReplayLaunchCoverageBinding>,
    ) -> FullLaunchSetDeclaration {
        FullLaunchSetDeclaration::checked("fixture/scope:v1", configuration, launches).unwrap()
    }

    #[test]
    fn exact_full_roster_preserves_canonical_diagnostics_without_composition() {
        let configuration = flat_configuration();
        let first = binding(1);
        let second = binding(2);
        let mut second_result = full_result(second.clone(), configuration, 5, 5);
        second_result.replay.front_end = FrontEndCounters {
            compulsory_instruction_read_sectors: 7,
        };
        second_result.replay.diagnostic_last_issue_cycles = 2_002;
        second_result.replay.last_processed_system_event_time = Some(2_222);
        second_result.replay.terminal_modeled_event_time = Some(2_222);
        second_result.replay.unmodeled_tails.red_rmw = true;
        let mut first_result = full_result(first.clone(), configuration, 3, 3);
        first_result.replay.front_end = FrontEndCounters {
            compulsory_instruction_read_sectors: 5,
        };
        first_result.replay.diagnostic_last_issue_cycles = 1_001;
        first_result.replay.last_processed_system_event_time = Some(1_111);
        first_result.replay.terminal_modeled_event_time = Some(1_111);
        let aggregate = aggregate_full_launch_set(
            declaration(configuration, [second.clone(), first.clone()]),
            [second_result, first_result],
        )
        .unwrap();

        assert_eq!(aggregate.scope().as_str(), "fixture/scope:v1");
        assert_eq!(aggregate.configuration(), configuration);
        assert_eq!(aggregate.arch(), ArchId::Sm86);
        assert_eq!(
            aggregate.estimator_provenance(),
            FullCounterEstimatorProvenance::ExactFullCoverageReplay
        );
        assert_eq!(aggregate.instructions_issued(), 8);
        assert_eq!(aggregate.records_seen(), 8);
        assert_eq!(
            aggregate.front_end().compulsory_instruction_read_sectors(),
            12
        );
        assert_eq!(aggregate.launch_coverages()[0].launch(), &first);
        assert_eq!(aggregate.launch_coverages()[1].launch(), &second);
        assert_eq!(
            aggregate.launch_coverages()[0].source(),
            &FullCoverageSource::UnsampledComplete
        );
        assert_eq!(aggregate.launch_timing_diagnostics().len(), 2);
        assert_eq!(aggregate.launch_timing_diagnostics()[0].launch(), &first);
        assert_eq!(
            aggregate.launch_timing_diagnostics()[0].diagnostic_last_issue_cycles(),
            1_001
        );
        assert_eq!(
            aggregate.launch_timing_diagnostics()[0].terminal_modeled_event_time(),
            Some(1_111)
        );
        assert_eq!(
            aggregate.launch_timing_diagnostics()[0].last_processed_modeled_system_event_time(),
            Some(1_111)
        );
        assert!(
            aggregate.launch_timing_diagnostics()[0]
                .modeled_completion_frontier()
                .is_complete()
        );
        assert_eq!(aggregate.launch_timing_diagnostics()[1].launch(), &second);
        assert_eq!(
            aggregate.launch_timing_diagnostics()[1].diagnostic_last_issue_cycles(),
            2_002
        );
        assert_eq!(
            aggregate.launch_timing_diagnostics()[1].terminal_modeled_event_time(),
            Some(2_222)
        );
        assert!(matches!(
            aggregate.launch_timing_diagnostics()[1].modeled_completion_frontier(),
            ModeledCompletionFrontier::Incomplete {
                lower_bound_cycles: Some(2_222),
                unmodeled_tails,
            } if unmodeled_tails.red_rmw
        ));
        assert!(aggregate.unmodeled_tails().red_rmw);
        assert!(aggregate.l1().is_none());
        assert!(aggregate.l2().is_none());
        assert!(aggregate.l2_by_source().is_none());
        assert!(aggregate.dram().is_none());
    }

    #[test]
    fn shared_mode_initializes_arch_bound_counter_shape() {
        let configuration = shared_configuration();
        let launch = binding(1);
        let aggregate = aggregate_full_launch_set(
            declaration(configuration, [launch.clone()]),
            [full_result(launch, configuration, 1, 1)],
        )
        .unwrap();

        assert_eq!(
            aggregate.l1().and_then(L1Counters::arch),
            Some(ArchId::Sm86)
        );
        assert_eq!(
            aggregate.l2().and_then(L2Counters::arch),
            Some(ArchId::Sm86)
        );
        assert_eq!(
            aggregate.l2_by_source().and_then(L2SourceSplit::arch),
            Some(ArchId::Sm86)
        );
        assert_eq!(
            aggregate.dram().and_then(DramStats::arch),
            Some(ArchId::Sm86)
        );
    }

    #[test]
    fn declarations_reject_noncanonical_scope_empty_roster_and_duplicate_identity() {
        let configuration = flat_configuration();
        assert!(matches!(
            FullLaunchSetDeclaration::checked("bad scope", configuration, [binding(1)]),
            Err(FullLaunchSetDeclarationError::Scope(
                LaunchSetScopeError::NonCanonical { .. }
            ))
        ));
        assert!(matches!(
            FullLaunchSetDeclaration::checked(
                "scope",
                configuration,
                std::iter::empty::<ReplayLaunchCoverageBinding>()
            ),
            Err(FullLaunchSetDeclarationError::EmptyExpectedRoster)
        ));
        let launch = binding(1);
        assert!(matches!(
            FullLaunchSetDeclaration::checked(
                "scope",
                configuration,
                [launch.clone(), launch.clone()]
            ),
            Err(FullLaunchSetDeclarationError::DuplicateExpectedLaunch {
                launch: duplicate
            }) if duplicate == launch
        ));
    }

    #[test]
    fn declaration_side_identity_is_checked_without_minting_coverage() {
        assert!(matches!(
            ReplayLaunchCoverageBinding::checked(0, 0, 0, 0, "", (1, 1, 1), None, (1, 1, 1)),
            Err(ReplayLaunchCoverageBindingError::EmptyKernelName)
        ));
        assert!(matches!(
            ReplayLaunchCoverageBinding::checked(0, 0, 0, 0, "k", (0, 1, 1), None, (1, 1, 1)),
            Err(ReplayLaunchCoverageBindingError::DegenerateGrid { .. })
        ));
        assert!(matches!(
            ReplayLaunchCoverageBinding::checked(
                0,
                0,
                0,
                0,
                "k",
                (u32::MAX, u32::MAX, u32::MAX),
                None,
                (1, 1, 1),
            ),
            Err(ReplayLaunchCoverageBindingError::GridSizeOverflow { .. })
        ));
    }

    #[test]
    fn actual_roster_rejects_duplicate_unexpected_and_missing_launches() {
        let configuration = flat_configuration();
        let first = binding(1);
        let second = binding(2);
        let duplicate = aggregate_full_launch_set(
            declaration(configuration, [first.clone()]),
            [
                full_result(first.clone(), configuration, 1, 1),
                full_result(first.clone(), configuration, 1, 1),
            ],
        );
        assert!(matches!(
            duplicate,
            Err(LaunchSetAggregationError::DuplicateLaunch { launch }) if launch == first
        ));

        let unexpected = aggregate_full_launch_set(
            declaration(configuration, [first.clone()]),
            [full_result(second.clone(), configuration, 1, 1)],
        );
        assert!(matches!(
            unexpected,
            Err(LaunchSetAggregationError::UnexpectedLaunch { launch }) if launch == second
        ));

        let missing = aggregate_full_launch_set(
            declaration(configuration, [first.clone(), second.clone()]),
            [full_result(first, configuration, 1, 1)],
        );
        assert!(matches!(
            missing,
            Err(LaunchSetAggregationError::MissingLaunch { launch }) if launch == second
        ));
    }

    #[test]
    fn architecture_configuration_and_coverage_mismatches_fail_before_arithmetic() {
        let configuration = flat_configuration();
        let launch = binding(1);
        let mut wrong_arch = full_result(launch.clone(), configuration, u64::MAX, u64::MAX);
        wrong_arch.replay.arch = ArchId::Sm120;
        assert!(matches!(
            aggregate_full_launch_set(declaration(configuration, [launch.clone()]), [wrong_arch]),
            Err(LaunchSetAggregationError::ArchitectureMismatch {
                expected: ArchId::Sm86,
                got: ArchId::Sm120,
                ..
            })
        ));

        let mut other_params = params(ArchId::Sm86);
        other_params.global_flat_latency += 1;
        let other_configuration =
            ReplayConfiguration::checked(other_params, ReplayMemMode::Flat).unwrap();
        assert!(matches!(
            aggregate_full_launch_set(
                declaration(configuration, [launch.clone()]),
                [full_result(launch.clone(), other_configuration, 1, 1)]
            ),
            Err(LaunchSetAggregationError::ConfigurationMismatch { .. })
        ));

        let mut incomplete = full_result(launch.clone(), configuration, 1, 1);
        incomplete.coverage.blocks_observed = 0;
        assert!(matches!(
            aggregate_full_launch_set(declaration(configuration, [launch]), [incomplete]),
            Err(LaunchSetAggregationError::CoverageMismatch {
                expected_blocks: 1,
                observed_blocks: 0,
                ..
            })
        ));
    }

    #[test]
    fn mode_shape_and_counter_architecture_are_checked() {
        let flat = flat_configuration();
        let launch = binding(1);
        let mut contaminated = full_result(launch.clone(), flat, 1, 1);
        contaminated.replay.l1 = Some(L1Counters::for_arch(ArchId::Sm86));
        assert!(matches!(
            aggregate_full_launch_set(declaration(flat, [launch.clone()]), [contaminated]),
            Err(LaunchSetAggregationError::CounterPresenceMismatch {
                component: LaunchSetCounterComponent::L1,
                expected: false,
                got: true,
                ..
            })
        ));

        let l1_configuration = l1_configuration();
        let mut wrong_counter_arch = full_result(launch.clone(), l1_configuration, 1, 1);
        wrong_counter_arch.replay.l1 = Some(L1Counters::for_arch(ArchId::Sm120));
        assert!(matches!(
            aggregate_full_launch_set(
                declaration(l1_configuration, [launch]),
                [wrong_counter_arch]
            ),
            Err(LaunchSetAggregationError::CounterArchitectureMismatch {
                component: LaunchSetCounterComponent::L1,
                expected: ArchId::Sm86,
                got: Some(ArchId::Sm120),
                ..
            })
        ));
    }

    #[test]
    fn scalar_and_counter_overflow_name_the_responsible_launch() {
        let configuration = flat_configuration();
        let first = binding(1);
        let second = binding(2);
        let scalar = aggregate_full_launch_set(
            declaration(configuration, [first.clone(), second.clone()]),
            [
                full_result(first.clone(), configuration, u64::MAX, u64::MAX),
                full_result(second.clone(), configuration, 1, 1),
            ],
        );
        assert!(matches!(
            scalar,
            Err(LaunchSetAggregationError::ScalarOverflow {
                launch,
                counter: LaunchSetScalarCounter::InstructionsIssued,
            }) if launch == second
        ));

        let configuration = l1_configuration();
        let mut first_result = full_result(first.clone(), configuration, 1, 1);
        let mut first_l1 = L1Counters::for_arch(ArchId::Sm86);
        first_l1.hits = u64::MAX;
        first_result.replay.l1 = Some(first_l1.with_unified_origin(MemoryOrigin::Global));
        let mut second_result = full_result(second.clone(), configuration, 1, 1);
        let mut second_l1 = L1Counters::for_arch(ArchId::Sm86);
        second_l1.hits = 1;
        second_result.replay.l1 = Some(second_l1.with_unified_origin(MemoryOrigin::Global));
        let counter = aggregate_full_launch_set(
            declaration(configuration, [first, second.clone()]),
            [first_result, second_result],
        );
        assert!(matches!(
            counter,
            Err(LaunchSetAggregationError::L1Counter { launch, .. }) if launch == second
        ));
    }
}
