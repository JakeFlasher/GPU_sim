//! Model-counter emission for native replay. NCU values are not inputs.
//!
//! Keep zero, unavailable, not-applicable, and incomplete timing separate.
//! Rates use the declared counter domain; async loads and atomic write-class
//! requests retain distinct traffic accounting.

use serde::ser::{Serialize, SerializeStruct, Serializer};
use sim_arch::ArchId;
use sim_mem::{L1CounterError, L1Counters, L1OriginCounters, L2Counters, L2Source, MemoryOrigin};
use sim_sm::{
    DramStats, FrontEndCounters, FullCounterEstimatorProvenance, FullKernelReplayResult,
    FullLaunchCoverage, FullLaunchSetCounters, FullLaunchTimingDiagnostic, L2SourceSplit,
    LaunchSetScope, ModeledCompletionFrontier, ReplayConfiguration, SystemCounterError, TailFlags,
    TraceMetrics,
};

/// The exact counter fields the emission layer reads, borrowed. The type is
/// private so certified emission cannot be invoked over caller-forged snapshots.
#[derive(Debug, Clone, Copy)]
struct CounterSnapshot<'a> {
    arch: ArchId,
    instructions_issued: u64,
    l1: Option<&'a L1Counters>,
    l2: Option<&'a L2Counters>,
    l2_by_source: Option<&'a L2SourceSplit>,
    dram: Option<&'a DramStats>,
    front_end: FrontEndCounters,
    binding_cycles: Option<BindingCycles>,
    trace_metrics: Option<&'a TraceMetrics>,
    /// Effective SM120 TMA global-to-shared payload demand, in 32-byte
    /// sectors. This is a conditional sidecar demand arm, not engine timing.
    tma_load_sectors: u64,
    /// Potential partial-store RMW sectors derived from traced byte coverage.
    /// Emission intersects this with a modeled cold-store population before
    /// charging a conditional DRAM read.
    partial_store_sectors: u64,
}

impl<'a> CounterSnapshot<'a> {
    fn from_full_launch(result: &'a FullKernelReplayResult) -> Result<Self, EmitError> {
        // A binding cycle is a publication-facing timing value.  The replay
        // can still expose its lower-bound and architecture-scoped frontiers
        // diagnostically, but dirty writeback/coherence (or any other typed
        // tail) means the launch is not a complete timing observation.
        let binding_cycles = if result.unmodeled_tails().is_empty() {
            Some(single_binding_cycles(result)?)
        } else {
            None
        };
        Ok(Self {
            arch: result.arch(),
            instructions_issued: result.instructions_issued(),
            l1: result.l1(),
            l2: result.l2(),
            l2_by_source: result.l2_by_source(),
            dram: result.dram(),
            front_end: result.front_end(),
            binding_cycles,
            trace_metrics: Some(result.trace_metrics()),
            tma_load_sectors: result
                .trace_metrics()
                .tma_sidecar_load_payload_bytes
                .div_ceil(32),
            partial_store_sectors: result.trace_metrics().partial_store_sectors,
        })
    }

    fn from_launch_set(counters: &'a FullLaunchSetCounters) -> Result<Self, EmitError> {
        let binding_cycles = if counters.unmodeled_tails().is_empty() {
            Some(aggregate_binding_cycles(
                counters.launch_timing_diagnostics(),
            )?)
        } else {
            None
        };
        Ok(Self {
            arch: counters.arch(),
            instructions_issued: counters.instructions_issued(),
            l1: counters.l1(),
            l2: counters.l2(),
            l2_by_source: counters.l2_by_source(),
            dram: counters.dram(),
            front_end: counters.front_end(),
            binding_cycles,
            trace_metrics: None,
            tma_load_sectors: 0,
            partial_store_sectors: 0,
        })
    }

    fn require_counter_architecture(&self) -> Result<(), EmitError> {
        let check = |counter: &'static str, got: Option<ArchId>| {
            let got = got.ok_or(EmitError::CounterArchitectureUnbound { counter })?;
            if got != self.arch {
                return Err(EmitError::CounterArchitectureMismatch {
                    counter,
                    expected: self.arch,
                    got,
                });
            }
            Ok(())
        };
        if let Some(l1) = self.l1 {
            check("L1 counters", l1.arch())?;
        }
        if let Some(l2) = self.l2 {
            check("L2 counters", l2.arch())?;
        }
        if let Some(l2_by_source) = self.l2_by_source {
            check("L2 source split", l2_by_source.arch())?;
        }
        if let Some(dram) = self.dram {
            check("DRAM stats", dram.arch())?;
        }
        if let Some(l1) = self.l1 {
            l1.validate_origin_partition()?;
        }
        if let Some(l2_by_source) = self.l2_by_source {
            l2_by_source.validate_origin_partition()?;
        }
        if let Some(dram) = self.dram {
            dram.validate_origin_partition()?;
        }
        Ok(())
    }
}

/// Numeric predictions are exact only within the simulator's declared model
/// scope. This is not a silicon-accuracy claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericSemantics {
    SimulatorScope,
}

impl NumericSemantics {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SimulatorScope => "simulator_scope",
        }
    }
}

impl Serialize for NumericSemantics {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

// The binding-cycle payload is intentionally constructible only inside this
// child module.  Construction is restricted to the checked launch-set
// aggregation rule below; there is no `From<u64>` path that can turn an
// arbitrary replay diagnostic into a binding prediction.
mod binding_cycles {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct BindingCycles(u64);

    impl BindingCycles {
        pub(crate) const fn from_aggregated_frontier(value: u64) -> Self {
            Self(value)
        }
        pub fn get(self) -> u64 {
            self.0
        }
    }
}

/// Opaque binding-cycle payload. Replay timing diagnostics have no public
/// conversion into this type; only checked emitter paths can construct it:
///
/// ```compile_fail
/// use sim_direct_run::emit::BindingCycles;
/// use sim_sm::FullKernelReplayResult;
///
/// fn reject_diagnostic_as_binding(result: &FullKernelReplayResult) -> BindingCycles {
///     result.diagnostic_last_issue_cycles().into()
/// }
/// ```
pub use binding_cycles::BindingCycles;

/// A predicted metric value, shaped to its NCU contract. The numeric payload
/// carries num/den so a `None` rate is auditable in-band (None IFF den == 0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetricValue {
    /// An exact non-negative integer count within simulator scope. This does not
    /// assert that the value is exact relative to silicon.
    Exact(u64),
    /// A binding-cycle value produced only by checked replay-frontier
    /// aggregation.
    BindingCycles(BindingCycles),
    /// A `[0, 1]` fraction. `value` is `None` iff `denominator == 0` (NCU
    /// reports the metric as n/a — never a fabricated 0.0).
    Fraction {
        value: Option<f64>,
        numerator: u64,
        denominator: u64,
    },
}

impl MetricValue {
    /// Fail-closed `[0, 1]` fraction constructor. `value` is `None` iff
    /// `den == 0`; a `num > den` fraction is a counter-arithmetic bug and is
    /// surfaced typed, never clamped into range.
    fn fraction(key: &'static str, num: u64, den: u64) -> Result<Self, EmitError> {
        if num > den {
            return Err(EmitError::RateExceedsUnity { key, num, den });
        }
        let value = if den == 0 {
            None
        } else {
            Some(exact_fraction_to_f64(key, num, den)?)
        };
        Ok(Self::Fraction {
            value,
            numerator: num,
            denominator: den,
        })
    }
}

impl Serialize for MetricValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Exact(value) => {
                let mut state = serializer.serialize_struct("MetricValue", 3)?;
                state.serialize_field("kind", "exact")?;
                state.serialize_field("semantics", &NumericSemantics::SimulatorScope)?;
                state.serialize_field("value", value)?;
                state.end()
            }
            Self::Fraction {
                value,
                numerator,
                denominator,
            } => {
                let mut state = serializer.serialize_struct("MetricValue", 5)?;
                state.serialize_field("kind", "fraction")?;
                state.serialize_field("semantics", &NumericSemantics::SimulatorScope)?;
                state.serialize_field("value", value)?;
                state.serialize_field("numerator", numerator)?;
                state.serialize_field("denominator", denominator)?;
                state.end()
            }
            Self::BindingCycles(value) => {
                let mut state = serializer.serialize_struct("MetricValue", 3)?;
                state.serialize_field("kind", "binding_cycles")?;
                state.serialize_field("semantics", &NumericSemantics::SimulatorScope)?;
                state.serialize_field("value", &value.get())?;
                state.end()
            }
        }
    }
}

/// Correctly round an exact `u64/u64` ratio in `[0, 1]` to binary64 without
/// first rounding each integer operand independently. Every positive ratio in
/// this domain is a normal binary64 value (`>= 1/u64::MAX > 2^-64`), so a
/// 53-bit normalized quotient plus exact remainder is sufficient. `u128`
/// covers the largest scaled numerator: at most 117 significant bits.
fn exact_fraction_to_f64(key: &'static str, num: u64, den: u64) -> Result<f64, EmitError> {
    const FRACTION_BITS: u32 = 52;
    const HIDDEN_BIT: u128 = 1_u128 << FRACTION_BITS;
    const CARRY_BIT: u128 = HIDDEN_BIT << 1;
    const EXPONENT_BIAS: u64 = 1023;
    const DETAIL: &str = "exact fraction conversion invariant broken";
    let invariant = || EmitError::CounterInvariant {
        key,
        detail: DETAIL,
    };

    if den == 0 || num > den {
        return Err(invariant());
    }
    if num == 0 {
        return Ok(0.0);
    }
    if num == den {
        return Ok(1.0);
    }

    // For 0 < num/den < 1, `negative_exponent` is `-floor(log2(num/den))`
    // in 1..=64. Comparing exact integers resolves the one-bit ambiguity left
    // by the operands' individual `ilog2` values.
    let bit_gap = den
        .ilog2()
        .checked_sub(num.ilog2())
        .ok_or_else(&invariant)?;
    let normalized_num = u128::from(num)
        .checked_shl(bit_gap)
        .ok_or_else(&invariant)?;
    let mut negative_exponent = bit_gap
        .checked_add(u32::from(normalized_num < u128::from(den)))
        .ok_or_else(&invariant)?;
    let scale = FRACTION_BITS
        .checked_add(negative_exponent)
        .ok_or_else(&invariant)?;
    let scaled_num = u128::from(num).checked_shl(scale).ok_or_else(&invariant)?;
    let den_wide = u128::from(den);
    let quotient = scaled_num / den_wide;
    let remainder = scaled_num % den_wide;
    let twice_remainder = remainder.checked_mul(2).ok_or_else(&invariant)?;
    let round_up = twice_remainder > den_wide || (twice_remainder == den_wide && quotient & 1 == 1);
    let mut rounded = if round_up {
        quotient.checked_add(1).ok_or_else(&invariant)?
    } else {
        quotient
    };

    // Rounding can carry into the next exponent (including exactly 1.0).
    if rounded == CARRY_BIT {
        rounded >>= 1;
        negative_exponent = negative_exponent.checked_sub(1).ok_or_else(&invariant)?;
    }
    if !(HIDDEN_BIT..CARRY_BIT).contains(&rounded) {
        return Err(invariant());
    }

    let significand = u64::try_from(rounded).map_err(|_| invariant())?;
    let fraction = significand
        .checked_sub(1_u64 << FRACTION_BITS)
        .ok_or_else(&invariant)?;
    let biased_exponent = EXPONENT_BIAS
        .checked_sub(u64::from(negative_exponent))
        .ok_or_else(&invariant)?;
    let bits = biased_exponent
        .checked_shl(FRACTION_BITS)
        .and_then(|exponent| exponent.checked_add(fraction))
        .ok_or_else(&invariant)?;
    let value = f64::from_bits(bits);
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(invariant());
    }
    Ok(value)
}

/// The NCU-facing unit of a metric (the registry `unit` field).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    /// A raw non-negative count.
    Count,
    /// A percentage in NCU's own rendering; stored as a `[0, 1]` fraction
    /// (registry `parser_transform = percent_to_fraction`).
    Pct,
    /// A clock-cycle count in simulator scope.
    Cycles,
    /// Elapsed nanoseconds retained only as a label-side exclusion.
    Nanoseconds,
}

impl Unit {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Count => "count",
            Self::Pct => "pct",
            Self::Cycles => "cycles",
            Self::Nanoseconds => "ns",
        }
    }
}

impl Serialize for Unit {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

/// The numeric shape a descriptor promises to emit — checked against the
/// returned [`MetricValue`] with a typed error, and pinned
/// against the registry `valid_max` and parser transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricKind {
    /// Emits [`MetricValue::Exact`].
    Count,
    /// Emits [`MetricValue::Fraction`].
    Fraction,
    /// May emit only [`MetricValue::BindingCycles`].
    BindingCycles,
}

/// Registry upper-bound contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegistryValidMax {
    Unbounded,
    One,
}

/// Registry parser transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParserTransform {
    IdentityNonnegative,
    PercentToFraction,
}

impl ParserTransform {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::IdentityNonnegative => "identity_nonneg",
            Self::PercentToFraction => "percent_to_fraction",
        }
    }
}

/// Registry model-I/O role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelIoRole {
    Target,
    Input,
}

impl ModelIoRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Target => "target",
            Self::Input => "input",
        }
    }
}

/// A hardware counting unit a metric needs to be predictable at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemUnit {
    /// The L1TEX cache (`FullLaunchSetCounters::l1`).
    L1Tex,
    /// The L2 slices (`FullLaunchSetCounters::l2`).
    Lts,
    /// The DRAM read observer (`FullLaunchSetCounters::dram`).
    Dram,
}

impl MemUnit {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::L1Tex => "l1tex",
            Self::Lts => "lts",
            Self::Dram => "dram",
        }
    }
}

/// Typed reason a metric was NOT predicted by THIS run — distinct from a
/// value-level `None` (which means "predicted n/a: no accesses"). The harness
/// excludes such a cell with a reason rather than scoring a null.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unavailable {
    /// This descriptor has no reviewed counting semantics on the replay architecture.
    ArchitectureUnsupported {
        arch: ArchId,
    },
    /// The replay's mem-mode did not attach the counting unit this metric
    /// reads (e.g. no L2 counters under `Flat`/`L1Const`).
    MemModeLacksUnit {
        needs: MemUnit,
    },
    ScopeUnestablished {
        detail: &'static str,
    },
}

impl Serialize for Unavailable {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::ArchitectureUnsupported { arch } => {
                let mut state = serializer.serialize_struct("Unavailable", 2)?;
                state.serialize_field("reason", "architecture_unsupported")?;
                state.serialize_field("arch", arch.token())?;
                state.end()
            }
            Self::MemModeLacksUnit { needs } => {
                let mut state = serializer.serialize_struct("Unavailable", 2)?;
                state.serialize_field("reason", "mem_mode_lacks_unit")?;
                state.serialize_field("needs", needs.as_str())?;
                state.end()
            }
            Self::ScopeUnestablished { detail } => {
                let mut state = serializer.serialize_struct("Unavailable", 2)?;
                state.serialize_field("reason", "scope_unestablished")?;
                state.serialize_field("detail", detail)?;
                state.end()
            }
        }
    }
}

/// Evidence prerequisite that blocks a mandatory registry output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequiredUnimplementedBlocker {
    /// A checked-in study must select and bind lawful launch aggregation
    /// semantics from admitted silicon evidence.
    AggregationSemanticsStudy,
}

impl RequiredUnimplementedBlocker {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AggregationSemanticsStudy => "aggregation_semantics_study",
        }
    }
}

impl Serialize for RequiredUnimplementedBlocker {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

/// Per-metric outcome: a genuine simulator-scope prediction, a typed
/// not-produced reason, or a mandatory contract blocked on explicit evidence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmitOutcome {
    Predicted(MetricValue),
    Unavailable(Unavailable),
    RequiredUnimplemented {
        blocking: RequiredUnimplementedBlocker,
    },
}

impl Serialize for EmitOutcome {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Predicted(value) => {
                let mut state = serializer.serialize_struct("EmitOutcome", 2)?;
                state.serialize_field("status", "predicted")?;
                state.serialize_field("value", value)?;
                state.end()
            }
            Self::Unavailable(reason) => {
                let mut state = serializer.serialize_struct("EmitOutcome", 2)?;
                state.serialize_field("status", "unavailable")?;
                state.serialize_field("detail", reason)?;
                state.end()
            }
            Self::RequiredUnimplemented { blocking } => {
                let mut state = serializer.serialize_struct("EmitOutcome", 2)?;
                state.serialize_field("status", "required_unimplemented")?;
                state.serialize_field("blocking", blocking)?;
                state.end()
            }
        }
    }
}

/// Number of outcomes in the closed simulator-emission surface: seven
/// implemented secondary metrics plus the mandatory binding-cycle row.
pub const EMITTED_METRIC_COUNT: usize = 8;

/// Closed identity of every metric the simulator can emit. Result-v2 ordering is
/// frozen independently from the legacy descriptor-vector order so exact maps can
/// follow the P1 7+1 authority without changing the legacy v1 byte contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EmittedMetricKey {
    SmspInstExecutedSum,
    TrafficGmemLdSectors,
    TrafficGmemStSectors,
    TrafficDramReadSectors,
    CacheHitRateL1TexReadSector,
    CacheHitRateL1TexSector,
    CacheHitRateL2Sector,
    ExecCyclesGpcMax,
}

impl EmittedMetricKey {
    /// Frozen order from `metric_registry.7plus1.v1` and `sim_result.v2`.
    pub const V2_ORDER: [Self; EMITTED_METRIC_COUNT] = [
        Self::SmspInstExecutedSum,
        Self::TrafficGmemLdSectors,
        Self::TrafficGmemStSectors,
        Self::TrafficDramReadSectors,
        Self::CacheHitRateL1TexReadSector,
        Self::CacheHitRateL1TexSector,
        Self::CacheHitRateL2Sector,
        Self::ExecCyclesGpcMax,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SmspInstExecutedSum => "smsp_inst_executed_sum",
            Self::TrafficGmemLdSectors => "traffic_gmem_ld_sectors",
            Self::TrafficGmemStSectors => "traffic_gmem_st_sectors",
            Self::TrafficDramReadSectors => "traffic_dram_read_sectors",
            Self::CacheHitRateL1TexReadSector => "cache_hit_rate_l1tex_read_sector",
            Self::CacheHitRateL1TexSector => "cache_hit_rate_l1tex_sector",
            Self::CacheHitRateL2Sector => "cache_hit_rate_l2_sector",
            Self::ExecCyclesGpcMax => "exec_cycles_gpc_max",
        }
    }
}

impl Serialize for EmittedMetricKey {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

/// One emitted metric, carrying its registry key + NCU identifier so the eval
/// join (downstream, outside `sim/`) can match against frozen labels by key.
/// Values remain borrowed from their certified prediction bundle and are
/// intentionally neither `Clone` nor `Copy`.
///
/// ```compile_fail
/// use sim_direct_run::emit::MetricPrediction;
///
/// fn detach(prediction: &MetricPrediction) -> MetricPrediction {
///     *prediction
/// }
/// ```
#[derive(Debug, PartialEq)]
pub struct MetricPrediction {
    arch: ArchId,
    key: EmittedMetricKey,
    ncu_metric: &'static str,
    unit: Unit,
    outcome: EmitOutcome,
}

impl MetricPrediction {
    pub fn arch(&self) -> ArchId {
        self.arch
    }

    pub fn key(&self) -> &'static str {
        self.key.as_str()
    }

    pub fn emitted_key(&self) -> EmittedMetricKey {
        self.key
    }

    pub fn ncu_metric(&self) -> &'static str {
        self.ncu_metric
    }

    pub fn unit(&self) -> Unit {
        self.unit
    }

    pub fn outcome(&self) -> &EmitOutcome {
        &self.outcome
    }

    /// Numeric outcomes are scoped to the simulator model and do not assert
    /// silicon accuracy. Non-numeric outcomes have no numeric semantics.
    pub fn numeric_semantics(&self) -> Option<NumericSemantics> {
        matches!(self.outcome, EmitOutcome::Predicted(_))
            .then_some(NumericSemantics::SimulatorScope)
    }
}

impl Serialize for MetricPrediction {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("MetricPrediction", 5)?;
        state.serialize_field("arch", self.arch.token())?;
        state.serialize_field("key", &self.key)?;
        state.serialize_field("ncu_metric", self.ncu_metric)?;
        state.serialize_field("unit", &self.unit)?;
        state.serialize_field("outcome", &self.outcome)?;
        state.end()
    }
}

/// Certified predictions for one exact full-coverage launch. Construction is
/// private to [`emit_launch`], so replay configuration, coverage, diagnostics,
/// tails, and descriptor outcomes cannot be detached and relabelled as another
/// launch.
#[derive(Debug)]
pub struct FullLaunchPredictions {
    configuration: ReplayConfiguration,
    coverage: FullLaunchCoverage,
    diagnostic_last_issue_cycles: u64,
    last_processed_modeled_system_event_time: Option<u64>,
    terminal_modeled_event_time: Option<u64>,
    modeled_completion_frontier: ModeledCompletionFrontier,
    unmodeled_tails: TailFlags,
    metrics: [MetricPrediction; EMITTED_METRIC_COUNT],
}

impl FullLaunchPredictions {
    pub fn configuration(&self) -> ReplayConfiguration {
        self.configuration
    }

    pub fn arch(&self) -> ArchId {
        self.configuration.arch()
    }

    pub fn coverage(&self) -> &FullLaunchCoverage {
        &self.coverage
    }

    /// Maximum per-SM issue frontier (`last local issue timestamp + 1`), also
    /// one input to the checked binding-cycle reducer.
    pub fn diagnostic_last_issue_cycles(&self) -> u64 {
        self.diagnostic_last_issue_cycles
    }

    pub fn last_processed_modeled_system_event_time(&self) -> Option<u64> {
        self.last_processed_modeled_system_event_time
    }

    /// Maximum timestamp covered by the modeled event set. The reducer uses
    /// timestamp plus one as its elapsed-cycle contribution.
    pub fn terminal_modeled_event_time(&self) -> Option<u64> {
        self.terminal_modeled_event_time
    }

    pub fn modeled_completion_frontier(&self) -> ModeledCompletionFrontier {
        self.modeled_completion_frontier
    }

    pub fn unmodeled_tails(&self) -> TailFlags {
        self.unmodeled_tails
    }

    pub fn metrics(&self) -> &[MetricPrediction] {
        &self.metrics
    }
}

/// Certified predictions for one exact full-coverage launch set. Construction is
/// private to [`emit_all`], so scope/configuration/coverage cannot be detached
/// from or relabelled onto an arbitrary metric vector.
#[derive(Debug)]
pub struct FullLaunchSetPredictions {
    scope: LaunchSetScope,
    configuration: ReplayConfiguration,
    launch_coverages: Vec<FullLaunchCoverage>,
    launch_timing_diagnostics: Vec<FullLaunchTimingDiagnostic>,
    estimator_provenance: FullCounterEstimatorProvenance,
    unmodeled_tails: TailFlags,
    metrics: [MetricPrediction; EMITTED_METRIC_COUNT],
}

impl FullLaunchSetPredictions {
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

    pub fn unmodeled_tails(&self) -> TailFlags {
        self.unmodeled_tails
    }

    pub fn metrics(&self) -> &[MetricPrediction] {
        &self.metrics
    }
}

/// A compiled row of the emission table: the binding from a sim quantity to
/// its complete registry contract. [`DESCRIPTORS`] and [`TYPED_EXCLUSIONS`]
/// form an exact, disjoint partition of the active registry.
pub struct MetricDescriptor {
    pub key: EmittedMetricKey,
    pub ncu_metric: &'static str,
    pub unit: Unit,
    pub kind: MetricKind,
    pub valid_max: RegistryValidMax,
    pub parser_transform: ParserTransform,
    pub model_io_role: ModelIoRole,
    /// Architectures with reviewed counter semantics for this descriptor.
    pub available_on: &'static [ArchId],
    emit: fn(&CounterSnapshot) -> Result<EmitOutcome, EmitError>,
}

/// Closed reason that an active registry row is not a simulator descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TypedExclusionReason {
    MemUnitUnmodeled,
    OpClassUnmodeled,
    LabelOnlyNoComposition,
    TypedUnavailableRegistryRow,
    InputRoleNotEmitted,
}

/// One non-descriptor registry row with its complete contract and typed reason.
pub struct TypedExclusion {
    pub key: &'static str,
    pub ncu_metric: &'static str,
    pub unit: Unit,
    pub valid_max: RegistryValidMax,
    pub parser_transform: ParserTransform,
    pub model_io_role: ModelIoRole,
    pub reason: TypedExclusionReason,
}

/// Fail-closed emission faults. Benign "this run's mem-mode lacks the unit" is
/// NOT an error — it is an [`Unavailable`] outcome inside the vector, so a
/// Flat-mode run still emits the instruction count while marking the cache
/// rates typed-unavailable (the Carded::Gap / unsupported_memory_path idiom).
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum EmitError {
    #[error(
        "metric {key}: rate numerator {num} exceeds denominator {den} — a >1.0 fraction is a \
         counter bug, never clamped into range"
    )]
    RateExceedsUnity {
        key: &'static str,
        num: u64,
        den: u64,
    },
    #[error("metric {key}: {operation} overflows u64")]
    CounterOverflow {
        key: &'static str,
        operation: &'static str,
    },
    #[error("counter invariant broken for {key}: {detail}")]
    CounterInvariant {
        key: &'static str,
        detail: &'static str,
    },
    #[error(
        "metric {key}: emitted value kind {returned:?} does not match descriptor kind {declared:?}"
    )]
    DescriptorValueKindMismatch {
        key: &'static str,
        declared: MetricKind,
        returned: MetricKind,
    },
    #[error("binding-cycle metric {key} did not receive a checked binding value")]
    BindingCyclesImplementationNotAdmitted { key: &'static str },
    #[error(
        "metric {key}: required_unimplemented is valid only for a binding-cycle descriptor, got {declared:?}"
    )]
    RequiredUnimplementedKindMismatch {
        key: &'static str,
        declared: MetricKind,
    },
    #[error("{counter} has no bound architecture provenance")]
    CounterArchitectureUnbound { counter: &'static str },
    #[error("{counter} architecture mismatch: expected {expected}, got {got}")]
    CounterArchitectureMismatch {
        counter: &'static str,
        expected: ArchId,
        got: ArchId,
    },
    #[error(transparent)]
    L1Counters(#[from] L1CounterError),
    #[error(transparent)]
    SystemCounters(#[from] SystemCounterError),
}

fn checked_sum(
    key: &'static str,
    operation: &'static str,
    terms: &[u64],
) -> Result<u64, EmitError> {
    terms.iter().try_fold(0u64, |sum, &term| {
        sum.checked_add(term)
            .ok_or(EmitError::CounterOverflow { key, operation })
    })
}

/// READ-class bypass sectors = all bypass sectors minus the ATOM write-class
/// subset. This is the LDGSTS/bypass-LOAD stream that folds into the `op_ld`
/// surface; the ATOM subset (booked under op_atom/op_red by NCU) stays out.
fn bypass_read_sectors(key: &'static str, l1: &L1OriginCounters) -> Result<u64, EmitError> {
    l1.bypass_sectors
        .checked_sub(l1.bypass_atom_sectors)
        .ok_or(EmitError::CounterInvariant {
            key,
            detail: "bypass_atom_sectors exceeds bypass_sectors (subset invariant broken)",
        })
}

/// Trace-only resident-hit completion for store-free global-read launches.
/// The dynamic observer is already retained by `TraceMetrics`; admitting only
/// launches with only a tiny global-store footprint keeps this report
/// projection out of mixed write/client scope and leaves replay state
/// unchanged.
pub(crate) fn global_resident_hit_adjustment(
    arch: ArchId,
    l1: &L1Counters,
    trace: Option<&TraceMetrics>,
) -> u64 {
    // A mixed CONSTANT+plain-LDG launch has two client domains.  The
    // trace-resident sidecar is not form-qualified there and can attribute
    // CONSTANT reuse to the global-op-ld endpoint even when timed L1 only
    // observed in-flight merges.  Do not add that broad correction to this
    // mixed-client projection; pure-form launches retain the established
    // effective observer.
    if arch == ArchId::Sm120
        && trace.is_some_and(|metrics| {
            metrics.memory_scope.contains_key("global_load_constant")
                && metrics.memory_scope.contains_key("global_load_ldg")
                && l1.origin(MemoryOrigin::Global).hits == 0
        })
    {
        return 0;
    }
    let store_sectors = trace
        .and_then(|metrics| metrics.memory.get("global_store"))
        .map(|traffic| traffic.sectors)
        .unwrap_or(l1.origin(MemoryOrigin::Global).store_sectors);
    if arch != ArchId::Sm120 || store_sectors > 16 {
        return 0;
    }
    trace
        .map(|metrics| {
            metrics.l1_global_resident_hits.saturating_sub(
                l1.origin(MemoryOrigin::Global)
                    .effective_lookup_hits_for_arch(arch),
            )
        })
        .unwrap_or(0)
}

/// Global-read-only counterpart with a wider small-store boundary. The
/// global-load endpoint has no store client in its target scope, so it can
/// safely admit a slightly larger store footprint than the unified endpoint.
pub(crate) fn global_read_resident_hit_adjustment(
    arch: ArchId,
    l1: &L1Counters,
    trace: Option<&TraceMetrics>,
) -> u64 {
    // Keep the global-op-ld surface form-qualified for the same mixed-client
    // case as the unified projection.  The timed cache's resident hits remain
    // authoritative for pure plain-LDG or pure CONSTANT launches.
    if arch == ArchId::Sm120
        && trace.is_some_and(|metrics| {
            metrics.memory_scope.contains_key("global_load_constant")
                && metrics.memory_scope.contains_key("global_load_ldg")
                && l1.origin(MemoryOrigin::Global).hits == 0
        })
    {
        return 0;
    }
    let store_sectors = trace
        .and_then(|metrics| metrics.memory.get("global_store"))
        .map(|traffic| traffic.sectors)
        .unwrap_or(l1.origin(MemoryOrigin::Global).store_sectors);
    let global_load_sectors = trace
        .and_then(|metrics| metrics.memory.get("global_load"))
        .map(|traffic| traffic.sectors)
        .unwrap_or(0);
    let small_store_scope = store_sectors <= 64
        || trace.is_none_or(|metrics| {
            metrics.launch_resources.as_ref().is_none_or(|resources| {
                resources.block_threads != 128
                    || (resources.grid_blocks > 1 && global_load_sectors > 256)
            })
        });
    if arch != ArchId::Sm120 || !small_store_scope {
        return 0;
    }
    trace
        .map(|metrics| {
            metrics.l1_global_resident_hits.saturating_sub(
                l1.origin(MemoryOrigin::Global)
                    .effective_lookup_hits_for_arch(arch),
            )
        })
        .unwrap_or(0)
}

/// SM120's LDGSTS async source is a distinct client scope. The trace merge
/// audit can prove that its pending/same-request merges are not ordinary
/// global-load L1 tag hits for the external `op_ld` endpoint, while the
/// physical L1 state and timing still retain those merges. Keep this as a
/// report-only subtraction and clamp it to the effective observer hits; no
/// ordinary or constant-client pending merges are touched.
pub(crate) fn async_global_l1_hit_exclusion(
    arch: ArchId,
    l1: &L1Counters,
    trace: Option<&TraceMetrics>,
) -> u64 {
    if arch != ArchId::Sm120 {
        return 0;
    }
    let Some(metrics) = trace else {
        return 0;
    };
    if !metrics.memory_scope.contains_key("ldgsts_async_copy") {
        return 0;
    }
    // Same-request merges and settled async hits are not a stable external
    // `op_ld` scope witness: they can overlap the ordinary effective lookup
    // projection. The older-pending merge domain is the only disjoint
    // trace-qualified async miss/merge class retained here.
    let async_scope = metrics.l1_async_older_pending_merges_global;
    async_scope.min(
        l1.origin(MemoryOrigin::Global)
            .effective_lookup_hits_for_arch(arch),
    )
}

// ---- per-metric emit fns (the rate accounting, moved out of the counters) ----

/// `gpc__cycles_elapsed.max` — mandatory publication-binding output. The
/// counter snapshot carries only the checked launch-frontier aggregate.
fn emit_exec_cycles_gpc_max(snap: &CounterSnapshot) -> Result<EmitOutcome, EmitError> {
    match snap.binding_cycles {
        Some(value) => Ok(EmitOutcome::Predicted(MetricValue::BindingCycles(value))),
        None => Ok(EmitOutcome::RequiredUnimplemented {
            blocking: RequiredUnimplementedBlocker::AggregationSemanticsStudy,
        }),
    }
}

/// Compose the publication binding cycle from the checked temporal frontiers.
/// The launch-set rule is an explicit max over independent launch frontiers;
/// it never sums launches and never substitutes a wall-clock or hardware
/// measurement.  A terminal event is a timestamp, so its elapsed-cycle
/// contribution is `timestamp + 1`; completion frontiers are already cycle
/// counts.
fn aggregate_binding_cycles(
    diagnostics: &[FullLaunchTimingDiagnostic],
) -> Result<BindingCycles, EmitError> {
    if diagnostics.is_empty() {
        return Err(EmitError::CounterInvariant {
            key: "exec_cycles_gpc_max",
            detail: "binding-cycle aggregation requires at least one launch",
        });
    }
    let mut frontier = 0_u64;
    for diagnostic in diagnostics {
        frontier = frontier.max(diagnostic.diagnostic_last_issue_cycles());
        if let Some(timestamp) = diagnostic.terminal_modeled_event_time() {
            let elapsed = timestamp.checked_add(1).ok_or(EmitError::CounterOverflow {
                key: "exec_cycles_gpc_max",
                operation: "terminal event timestamp plus one",
            })?;
            frontier = frontier.max(elapsed);
        }
        if let Some(cycles) = diagnostic
            .modeled_completion_frontier()
            .lower_bound_cycles()
        {
            frontier = frontier.max(cycles);
        }
    }
    Ok(BindingCycles::from_aggregated_frontier(frontier))
}

fn single_binding_cycles(result: &FullKernelReplayResult) -> Result<BindingCycles, EmitError> {
    let mut frontier = result.diagnostic_last_issue_cycles();
    if let Some(timestamp) = result.terminal_modeled_event_time() {
        frontier = frontier.max(timestamp.checked_add(1).ok_or(EmitError::CounterOverflow {
            key: "exec_cycles_gpc_max",
            operation: "terminal event timestamp plus one",
        })?);
    }
    if let Some(cycles) = result.modeled_completion_frontier().lower_bound_cycles() {
        frontier = frontier.max(cycles);
    }
    if let Some(envelope) = super::sm120_front_end_envelope_cycles(result) {
        frontier = frontier
            .checked_add(envelope)
            .ok_or(EmitError::CounterOverflow {
                key: "exec_cycles_gpc_max",
                operation: "SM120 GPC front-end envelope",
            })?;
    }
    Ok(BindingCycles::from_aggregated_frontier(frontier))
}

/// `smsp__inst_executed.sum` — total warp-instructions issued. One issue per
/// traced record across every SM; always available (no mem unit needed).
fn emit_smsp_inst_executed(snap: &CounterSnapshot) -> Result<EmitOutcome, EmitError> {
    Ok(EmitOutcome::Predicted(MetricValue::Exact(
        snap.instructions_issued,
    )))
}

/// `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` — global-load sectors:
/// modeled eligible loads plus the READ-class bypass (LDGSTS) stream.
fn emit_gmem_ld_sectors(snap: &CounterSnapshot) -> Result<EmitOutcome, EmitError> {
    let key = "traffic_gmem_ld_sectors";
    let Some(l1) = snap.l1 else {
        return Ok(EmitOutcome::Unavailable(Unavailable::MemModeLacksUnit {
            needs: MemUnit::L1Tex,
        }));
    };
    let global = l1.origin(MemoryOrigin::Global);
    let bypass = bypass_read_sectors(key, &global)?;
    let sectors = checked_sum(
        key,
        "global-load sector total",
        &[global.hits, global.misses, bypass],
    )?;
    Ok(EmitOutcome::Predicted(MetricValue::Exact(sectors)))
}

/// `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` — global-store sectors.
fn emit_gmem_st_sectors(snap: &CounterSnapshot) -> Result<EmitOutcome, EmitError> {
    let Some(l1) = snap.l1 else {
        return Ok(EmitOutcome::Unavailable(Unavailable::MemModeLacksUnit {
            needs: MemUnit::L1Tex,
        }));
    };
    Ok(EmitOutcome::Predicted(MetricValue::Exact(
        l1.origin(MemoryOrigin::Global).store_sectors,
    )))
}

/// `l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct` — the op_ld sector
/// tag-lookup-hit rate. The bounded observer counts resident hits and merge
/// observations, then removes one tag miss for each fresh line allocation.
/// This is deliberately separate from modeled data readiness. The denominator
/// also folds in READ-class bypass (LDGSTS `.BYPASS`) sectors, which cannot hit
/// L1.
fn emit_l1tex_read_hit_rate(snap: &CounterSnapshot) -> Result<EmitOutcome, EmitError> {
    let key = "cache_hit_rate_l1tex_read_sector";
    let Some(l1) = snap.l1 else {
        return Ok(EmitOutcome::Unavailable(Unavailable::MemModeLacksUnit {
            needs: MemUnit::L1Tex,
        }));
    };
    let global = l1.origin(MemoryOrigin::Global);
    let num = global
        .effective_lookup_hits_for_arch(snap.arch)
        .saturating_add(global_read_resident_hit_adjustment(
            snap.arch,
            l1,
            snap.trace_metrics,
        ));
    let num = num.saturating_sub(async_global_l1_hit_exclusion(
        snap.arch,
        l1,
        snap.trace_metrics,
    ));
    let bypass = bypass_read_sectors(key, &global)?;
    let den = checked_sum(
        key,
        "global-load hit-rate denominator",
        &[global.hits, global.misses, bypass],
    )?;
    Ok(EmitOutcome::Predicted(MetricValue::fraction(
        key, num, den,
    )?))
}

/// Write-hit terms for the unified projection. SM86 retains its architecture-
/// specific lookup observer. SM120 has no supported write-lookup window and
/// instead reports resident-sector hits in the explicitly effective L1 model.
/// These terms do not change cache state, write routing, or replay timing.
pub(crate) fn l1_write_hit_terms(
    arch: ArchId,
    l1: &L1Counters,
    trace: Option<&TraceMetrics>,
) -> [u64; 2] {
    match arch {
        ArchId::Sm86 => {
            // `write_lookup_hits` belongs to the detached 256-set/2-slot
            // observer and is valid only when the producer instantiated that
            // observer for this launch.  Disabled launches must retain the
            // physical line-tag observation instead of silently reporting a
            // structural zero.  When both are available, union by max: the
            // report-only observer must not lose a tag hit already witnessed
            // by the physical L1 path.
            let physical = l1.store_tag_hits;
            let selected = if trace.is_some_and(|metrics| metrics.l1_write_lookup_observer_active) {
                l1.write_lookup_hits.max(physical)
            } else {
                physical
            };
            [selected, 0]
        }
        ArchId::Sm120 if trace.is_some_and(|metrics| metrics.l1_write_lookup_observer_active) => {
            if trace.is_some_and(|metrics| {
                (metrics.memory.contains_key("global_reduction")
                    || metrics.memory.contains_key("global_atomic"))
                    && !metrics.memory_scope.contains_key("global_load_ldg")
            }) {
                // Reduction/atomic-only global streams are operation-qualified
                // L1TEX clients, not ordinary unified sector lookups.  Keep
                // their write-tag accounting in the diagnostic counters but
                // do not leak it into the unified endpoint when no ordinary
                // global-load client is present.
                return [0, 0];
            }
            // The SM120 write observer is a per-SM pre-instruction tag view;
            // it applies to every cacheable store form, not only the legacy
            // unclassified-global subset.  Cold single-instruction stores
            // still contribute zero because plan_batch evaluates the tag
            // before allocating it.
            [l1.write_lookup_hits, l1.reduction_hits]
        }
        ArchId::Sm120 => [l1.store_hits, l1.reduction_hits],
    }
}

/// `l1tex__t_sector_hit_rate.pct` — the unified L1TEX sector lookup-hit rate
/// over all modeled op classes. Loads use the bounded tag-lookup projection;
/// write-hit accounting is architecture-specific (see above).
/// The full bypass stream (atom included) is in the denominator —
/// `bypass_atom_sectors` must NOT be added on top because it is already a subset
/// of `bypass_sectors`.
fn emit_l1tex_unified_hit_rate(snap: &CounterSnapshot) -> Result<EmitOutcome, EmitError> {
    let key = "cache_hit_rate_l1tex_sector";
    let Some(l1) = snap.l1 else {
        return Ok(EmitOutcome::Unavailable(Unavailable::MemModeLacksUnit {
            needs: MemUnit::L1Tex,
        }));
    };
    let local = l1.origin(MemoryOrigin::Local);
    let local_read_requests = local.lookup_sectors();
    let local_store_sectors = local.store_sectors;
    let use_local_observer = snap.arch == ArchId::Sm120
        && snap.trace_metrics.is_some_and(|metrics| {
            metrics.local_interleaved_dynamic_store_records != 0
                && metrics.local_interleaved_l1_store_hits != 0
        });
    let (local_read_hits, local_read_den, local_store_hits, local_store_den) = if use_local_observer
    {
        let metrics = snap.trace_metrics.expect("observer predicate proved");
        (
            metrics.local_interleaved_l1_load_hits,
            metrics.local_interleaved_load_sectors,
            metrics.local_interleaved_l1_store_hits,
            metrics.local_interleaved_store_sectors,
        )
    } else {
        (
            local.effective_lookup_hits_for_arch(snap.arch),
            local_read_requests,
            local.store_hits,
            local_store_sectors,
        )
    };
    let global = l1.origin(MemoryOrigin::Global);
    let effective_read_hits = global
        .effective_lookup_hits_for_arch(snap.arch)
        .saturating_add(global_resident_hit_adjustment(
            snap.arch,
            l1,
            snap.trace_metrics,
        ))
        .saturating_sub(async_global_l1_hit_exclusion(
            snap.arch,
            l1,
            snap.trace_metrics,
        ))
        .checked_add(local_read_hits)
        .ok_or(EmitError::CounterOverflow {
            key,
            operation: "unified L1 effective lookup-hit numerator",
        })?;
    let [default_store_hits, reduction_hits] =
        l1_write_hit_terms(snap.arch, l1, snap.trace_metrics);
    let store_hits = if use_local_observer {
        global
            .store_hits
            .checked_add(local_store_hits)
            .ok_or(EmitError::CounterOverflow {
                key,
                operation: "unified L1 store-hit numerator",
            })?
    } else {
        default_store_hits
    };
    let num = checked_sum(
        key,
        "unified L1TEX hit-rate numerator",
        &[effective_read_hits, store_hits, reduction_hits],
    )?;
    let base_read_den = l1
        .hits
        .checked_add(l1.misses)
        .and_then(|total| total.checked_sub(local_read_requests))
        .and_then(|total| total.checked_add(local_read_den))
        .ok_or(EmitError::CounterOverflow {
            key,
            operation: "unified L1 read denominator",
        })?;
    let base_store_den = l1
        .store_sectors
        .checked_sub(local_store_sectors)
        .and_then(|total| total.checked_add(local_store_den))
        .ok_or(EmitError::CounterOverflow {
            key,
            operation: "unified L1 store denominator",
        })?;
    let den = checked_sum(
        key,
        "unified L1TEX hit-rate denominator",
        &[
            base_read_den,
            base_store_den,
            l1.reduction_sectors,
            l1.bypass_sectors,
            l1.untagged_sectors,
        ],
    )?;
    Ok(EmitOutcome::Predicted(MetricValue::fraction(
        key, num, den,
    )?))
}

/// `lts__t_sector_hit_rate.pct` — the modeled L1TEX-origin L2 lookup-hit rate.
/// `lookup_hits`/`lookup_requests` follow the independently validated observer
/// stream.  The physical `hits`/`misses` split remains unchanged for response
/// pricing and cache state, because write packetization can make the two
/// denominators differ.
fn emit_l2_hit_rate(snap: &CounterSnapshot) -> Result<EmitOutcome, EmitError> {
    let key = "cache_hit_rate_l2_sector";
    if snap.arch == ArchId::Sm120 {
        return Ok(EmitOutcome::Unavailable(Unavailable::ScopeUnestablished {
            detail: "generic lts__t_sector_hit_rate.pct client scope is not established for the SM120 TEX-sector observer",
        }));
    }
    let Some(l2) = snap.l2 else {
        return Ok(EmitOutcome::Unavailable(Unavailable::MemModeLacksUnit {
            needs: MemUnit::Lts,
        }));
    };
    let den = if l2.lookup_requests != 0 {
        l2.lookup_requests
    } else {
        // Compatibility for legacy/synthetic snapshots created before the
        // observer denominator was introduced.
        checked_sum(
            key,
            "unified L2 hit-rate denominator",
            &[l2.hits, l2.misses],
        )?
    };
    Ok(EmitOutcome::Predicted(MetricValue::fraction(
        key,
        l2.lookup_hits,
        den,
    )?))
}

/// `dram__sectors_op_read.sum` — data demand plus source-qualified front-end
/// fetch demand.  SM86 GCC/ICC fetches whole aligned 128-byte instruction
/// lines, so its term uses the unique-line sector ledger rather than the
/// smaller set of executed 32-byte sectors.  Effective SM120 runs retain the
/// historical executed-sector/TMA demand arm.  None of these terms changes
/// cache state or DRAM service timing.
fn emit_dram_read_sectors(snap: &CounterSnapshot) -> Result<EmitOutcome, EmitError> {
    let Some(dram) = snap.dram else {
        return Ok(EmitOutcome::Unavailable(Unavailable::MemModeLacksUnit {
            needs: MemUnit::Dram,
        }));
    };
    let compulsory = if snap.arch == ArchId::Sm86 {
        snap.front_end.static_instruction_line_read_sectors()
    } else {
        snap.front_end.compulsory_instruction_read_sectors()
    };
    // Small traced launches with a compact instruction footprint can expose
    // compulsory constant-bank sectors outside the TEX data path. Keep this
    // source-derived arm narrow: both the global-load and executed-code
    // footprints are bounded, and launch-set aggregation (which has no
    // per-launch trace metrics) deliberately does not admit it.
    let small_global_load_footprint = snap
        .l1
        .map(|l1| {
            let global = l1.origin(MemoryOrigin::Global);
            bypass_read_sectors("traffic_dram_read_sectors", &global).map(|bypass| {
                global
                    .hits
                    .saturating_add(global.misses)
                    .saturating_add(bypass)
            })
        })
        .transpose()?;
    let compulsory_constant = if snap.arch == ArchId::Sm120
        && snap.trace_metrics.is_some()
        && small_global_load_footprint.is_some_and(|sectors| sectors <= 1_024)
        && compulsory <= 64
    {
        snap.front_end.compulsory_constant_read_sectors()
    } else {
        0
    };
    let partial_store_rmw = snap
        .l2_by_source
        .and_then(|split| {
            let (store_sectors, store_hits) = split.source(L2Source::Store);
            let store_misses = store_sectors.checked_sub(store_hits)?;
            Some(snap.partial_store_sectors.min(store_misses))
        })
        .unwrap_or(0);

    let total = if snap.arch == ArchId::Sm120 {
        checked_sum(
            "traffic_dram_read_sectors",
            "data, conditional partial-store RMW, TMA payload, and compulsory instruction DRAM-read sectors",
            &[
                dram.read_sectors,
                partial_store_rmw,
                snap.tma_load_sectors,
                compulsory,
                compulsory_constant,
            ],
        )?
    } else {
        checked_sum(
            "traffic_dram_read_sectors",
            "data and aligned 128-byte GCC instruction-line DRAM-read sectors",
            &[dram.read_sectors, compulsory],
        )?
    };
    Ok(EmitOutcome::Predicted(MetricValue::Exact(total)))
}

const ALL_ARCHITECTURES: &[ArchId] = &[ArchId::Sm86, ArchId::Sm120];

/// The stable emission table. Seven simulator counters retain their established
/// evaluation order; the publication-binding cycle row is mandatory and remains
/// required-unimplemented.
pub const DESCRIPTORS: [MetricDescriptor; EMITTED_METRIC_COUNT] = [
    MetricDescriptor {
        key: EmittedMetricKey::SmspInstExecutedSum,
        ncu_metric: "smsp__inst_executed.sum",
        unit: Unit::Count,
        kind: MetricKind::Count,
        valid_max: RegistryValidMax::Unbounded,
        parser_transform: ParserTransform::IdentityNonnegative,
        model_io_role: ModelIoRole::Target,
        available_on: ALL_ARCHITECTURES,
        emit: emit_smsp_inst_executed,
    },
    MetricDescriptor {
        key: EmittedMetricKey::TrafficGmemLdSectors,
        ncu_metric: "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        unit: Unit::Count,
        kind: MetricKind::Count,
        valid_max: RegistryValidMax::Unbounded,
        parser_transform: ParserTransform::IdentityNonnegative,
        model_io_role: ModelIoRole::Target,
        available_on: ALL_ARCHITECTURES,
        emit: emit_gmem_ld_sectors,
    },
    MetricDescriptor {
        key: EmittedMetricKey::TrafficGmemStSectors,
        ncu_metric: "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
        unit: Unit::Count,
        kind: MetricKind::Count,
        valid_max: RegistryValidMax::Unbounded,
        parser_transform: ParserTransform::IdentityNonnegative,
        model_io_role: ModelIoRole::Target,
        available_on: ALL_ARCHITECTURES,
        emit: emit_gmem_st_sectors,
    },
    MetricDescriptor {
        key: EmittedMetricKey::CacheHitRateL1TexReadSector,
        ncu_metric: "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct",
        unit: Unit::Pct,
        kind: MetricKind::Fraction,
        valid_max: RegistryValidMax::One,
        parser_transform: ParserTransform::PercentToFraction,
        model_io_role: ModelIoRole::Target,
        available_on: ALL_ARCHITECTURES,
        emit: emit_l1tex_read_hit_rate,
    },
    MetricDescriptor {
        key: EmittedMetricKey::CacheHitRateL1TexSector,
        ncu_metric: "l1tex__t_sector_hit_rate.pct",
        unit: Unit::Pct,
        kind: MetricKind::Fraction,
        valid_max: RegistryValidMax::One,
        parser_transform: ParserTransform::PercentToFraction,
        model_io_role: ModelIoRole::Target,
        available_on: ALL_ARCHITECTURES,
        emit: emit_l1tex_unified_hit_rate,
    },
    MetricDescriptor {
        key: EmittedMetricKey::CacheHitRateL2Sector,
        ncu_metric: "lts__t_sector_hit_rate.pct",
        unit: Unit::Pct,
        kind: MetricKind::Fraction,
        valid_max: RegistryValidMax::One,
        parser_transform: ParserTransform::PercentToFraction,
        model_io_role: ModelIoRole::Target,
        available_on: ALL_ARCHITECTURES,
        emit: emit_l2_hit_rate,
    },
    MetricDescriptor {
        key: EmittedMetricKey::TrafficDramReadSectors,
        ncu_metric: "dram__sectors_read.sum",
        unit: Unit::Count,
        kind: MetricKind::Count,
        valid_max: RegistryValidMax::Unbounded,
        parser_transform: ParserTransform::IdentityNonnegative,
        model_io_role: ModelIoRole::Target,
        available_on: ALL_ARCHITECTURES,
        emit: emit_dram_read_sectors,
    },
    MetricDescriptor {
        key: EmittedMetricKey::ExecCyclesGpcMax,
        ncu_metric: "gpc__cycles_elapsed.max",
        unit: Unit::Cycles,
        kind: MetricKind::BindingCycles,
        valid_max: RegistryValidMax::Unbounded,
        parser_transform: ParserTransform::IdentityNonnegative,
        model_io_role: ModelIoRole::Target,
        available_on: ALL_ARCHITECTURES,
        emit: emit_exec_cycles_gpc_max,
    },
];

const fn target_count_exclusion(
    key: &'static str,
    ncu_metric: &'static str,
    reason: TypedExclusionReason,
) -> TypedExclusion {
    TypedExclusion {
        key,
        ncu_metric,
        unit: Unit::Count,
        valid_max: RegistryValidMax::Unbounded,
        parser_transform: ParserTransform::IdentityNonnegative,
        model_io_role: ModelIoRole::Target,
        reason,
    }
}

const fn target_pct_exclusion(
    key: &'static str,
    ncu_metric: &'static str,
    reason: TypedExclusionReason,
) -> TypedExclusion {
    TypedExclusion {
        key,
        ncu_metric,
        unit: Unit::Pct,
        valid_max: RegistryValidMax::One,
        parser_transform: ParserTransform::PercentToFraction,
        model_io_role: ModelIoRole::Target,
        reason,
    }
}

const fn target_cycles_exclusion(
    key: &'static str,
    ncu_metric: &'static str,
    reason: TypedExclusionReason,
) -> TypedExclusion {
    TypedExclusion {
        key,
        ncu_metric,
        unit: Unit::Cycles,
        valid_max: RegistryValidMax::Unbounded,
        parser_transform: ParserTransform::IdentityNonnegative,
        model_io_role: ModelIoRole::Target,
        reason,
    }
}

const fn target_ns_exclusion(
    key: &'static str,
    ncu_metric: &'static str,
    reason: TypedExclusionReason,
) -> TypedExclusion {
    TypedExclusion {
        key,
        ncu_metric,
        unit: Unit::Nanoseconds,
        valid_max: RegistryValidMax::Unbounded,
        parser_transform: ParserTransform::IdentityNonnegative,
        model_io_role: ModelIoRole::Target,
        reason,
    }
}

const fn input_count_exclusion(key: &'static str, ncu_metric: &'static str) -> TypedExclusion {
    TypedExclusion {
        key,
        ncu_metric,
        unit: Unit::Count,
        valid_max: RegistryValidMax::Unbounded,
        parser_transform: ParserTransform::IdentityNonnegative,
        model_io_role: ModelIoRole::Input,
        reason: TypedExclusionReason::InputRoleNotEmitted,
    }
}

/// Every active registry row that is not a descriptor, in registry order.
pub const TYPED_EXCLUSIONS: &[TypedExclusion] = &[
    target_ns_exclusion(
        "exec_duration_ns",
        "gpu__time_duration.sum",
        TypedExclusionReason::LabelOnlyNoComposition,
    ),
    target_count_exclusion(
        "traffic_gmem_ld_requests",
        "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
        TypedExclusionReason::MemUnitUnmodeled,
    ),
    target_count_exclusion(
        "traffic_gmem_st_requests",
        "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
        TypedExclusionReason::MemUnitUnmodeled,
    ),
    target_count_exclusion(
        "traffic_dram_write_sectors",
        "dram__sectors_write.sum",
        TypedExclusionReason::MemUnitUnmodeled,
    ),
    target_count_exclusion(
        "traffic_l2_total_requests_from_tex",
        "lts__t_requests_srcunit_tex.sum",
        TypedExclusionReason::MemUnitUnmodeled,
    ),
    target_count_exclusion(
        "traffic_l2_read_sectors_from_l1",
        "l1tex__m_xbar2l1tex_read_sectors_mem_lg_op_ld.sum",
        TypedExclusionReason::MemUnitUnmodeled,
    ),
    target_count_exclusion(
        "traffic_l1_write_sectors_to_l2",
        "l1tex__m_l1tex2xbar_write_sectors_mem_lg_op_st.sum",
        TypedExclusionReason::MemUnitUnmodeled,
    ),
    target_count_exclusion(
        "traffic_gmem_atom_requests",
        "l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum",
        TypedExclusionReason::OpClassUnmodeled,
    ),
    target_count_exclusion(
        "traffic_gmem_red_requests",
        "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum",
        TypedExclusionReason::OpClassUnmodeled,
    ),
    target_count_exclusion(
        "traffic_l1_total_requests",
        "l1tex__t_requests.sum",
        TypedExclusionReason::MemUnitUnmodeled,
    ),
    target_pct_exclusion(
        "occupancy_achieved_warps_pct",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        TypedExclusionReason::LabelOnlyNoComposition,
    ),
    target_count_exclusion(
        "ipc_inst_per_cycle",
        "sm__inst_executed.avg.per_cycle_elapsed",
        TypedExclusionReason::LabelOnlyNoComposition,
    ),
    input_count_exclusion(
        "occupancy_limit_warps_theoretical",
        "launch__occupancy_limit_warps",
    ),
    input_count_exclusion(
        "occupancy_limit_blocks_theoretical",
        "launch__occupancy_limit_blocks",
    ),
    input_count_exclusion(
        "occupancy_limit_registers_theoretical",
        "launch__occupancy_limit_registers",
    ),
    input_count_exclusion(
        "occupancy_limit_shared_mem_theoretical",
        "launch__occupancy_limit_shared_mem",
    ),
    target_cycles_exclusion(
        "exec_cycles_sm_active_avg",
        "sm__cycles_active.avg",
        TypedExclusionReason::LabelOnlyNoComposition,
    ),
    target_count_exclusion(
        "warps_active_per_cycle_active_avg",
        "sm__warps_active.avg.per_cycle_active",
        TypedExclusionReason::LabelOnlyNoComposition,
    ),
    target_count_exclusion(
        "sm_clock_hz_covariate",
        "gpc__cycles_elapsed.avg.per_second",
        TypedExclusionReason::LabelOnlyNoComposition,
    ),
    target_count_exclusion(
        "mem_clock_hz_covariate",
        "dram__cycles_elapsed.avg.per_second",
        TypedExclusionReason::LabelOnlyNoComposition,
    ),
    target_count_exclusion(
        "traffic_local_ld_requests",
        "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum",
        TypedExclusionReason::OpClassUnmodeled,
    ),
    target_count_exclusion(
        "traffic_local_st_requests",
        "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum",
        TypedExclusionReason::OpClassUnmodeled,
    ),
    target_count_exclusion(
        "traffic_tex_requests",
        "l1tex__t_requests_pipe_tex_mem_texture.sum",
        TypedExclusionReason::OpClassUnmodeled,
    ),
    target_count_exclusion(
        "traffic_surface_ld_requests",
        "l1tex__t_requests_pipe_tex_mem_surface_op_ld.sum",
        TypedExclusionReason::OpClassUnmodeled,
    ),
    target_count_exclusion(
        "traffic_surface_st_requests",
        "l1tex__t_requests_pipe_tex_mem_surface_op_st.sum",
        TypedExclusionReason::OpClassUnmodeled,
    ),
    target_count_exclusion(
        "traffic_surface_red_requests",
        "l1tex__t_requests_pipe_tex_mem_surface_op_red.sum",
        TypedExclusionReason::OpClassUnmodeled,
    ),
    target_count_exclusion(
        "traffic_surface_atom_requests",
        "l1tex__t_requests_pipe_tex_mem_surface_op_atom.sum",
        TypedExclusionReason::OpClassUnmodeled,
    ),
    target_count_exclusion(
        "dram_bytes_per_second",
        "dram__bytes.sum.per_second",
        TypedExclusionReason::LabelOnlyNoComposition,
    ),
    target_pct_exclusion(
        "throughput_compute_mem_access_pct_peak",
        "gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed",
        TypedExclusionReason::LabelOnlyNoComposition,
    ),
    target_pct_exclusion(
        "throughput_compute_mem_request_pct_peak",
        "gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed",
        TypedExclusionReason::LabelOnlyNoComposition,
    ),
    target_pct_exclusion(
        "throughput_sm_memory_pct_peak",
        "sm__memory_throughput.avg.pct_of_peak_sustained_elapsed",
        TypedExclusionReason::LabelOnlyNoComposition,
    ),
    target_count_exclusion(
        "traffic_dram_sectors_op_read",
        "dram__sectors_op_read.sum",
        TypedExclusionReason::TypedUnavailableRegistryRow,
    ),
    target_count_exclusion(
        "traffic_dram_sectors_op_write",
        "dram__sectors_op_write.sum",
        TypedExclusionReason::TypedUnavailableRegistryRow,
    ),
    target_count_exclusion(
        "warp_stall_barrier_per_issue_active",
        "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
        TypedExclusionReason::LabelOnlyNoComposition,
    ),
    target_count_exclusion(
        "warp_stall_branch_resolving_per_issue_active",
        "smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio",
        TypedExclusionReason::LabelOnlyNoComposition,
    ),
    target_count_exclusion(
        "warp_stall_dispatch_stall_per_issue_active",
        "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio",
        TypedExclusionReason::LabelOnlyNoComposition,
    ),
    target_count_exclusion(
        "warp_stall_drain_per_issue_active",
        "smsp__average_warps_issue_stalled_drain_per_issue_active.ratio",
        TypedExclusionReason::LabelOnlyNoComposition,
    ),
    target_count_exclusion(
        "warp_stall_imc_miss_per_issue_active",
        "smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio",
        TypedExclusionReason::LabelOnlyNoComposition,
    ),
    target_count_exclusion(
        "warp_stall_lg_throttle_per_issue_active",
        "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
        TypedExclusionReason::LabelOnlyNoComposition,
    ),
    target_count_exclusion(
        "warp_stall_long_scoreboard_per_issue_active",
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
        TypedExclusionReason::LabelOnlyNoComposition,
    ),
    target_count_exclusion(
        "warp_stall_math_pipe_throttle_per_issue_active",
        "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
        TypedExclusionReason::LabelOnlyNoComposition,
    ),
    target_count_exclusion(
        "warp_stall_membar_per_issue_active",
        "smsp__average_warps_issue_stalled_membar_per_issue_active.ratio",
        TypedExclusionReason::LabelOnlyNoComposition,
    ),
    target_count_exclusion(
        "warp_stall_mio_throttle_per_issue_active",
        "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
        TypedExclusionReason::LabelOnlyNoComposition,
    ),
    target_count_exclusion(
        "warp_stall_misc_per_issue_active",
        "smsp__average_warps_issue_stalled_misc_per_issue_active.ratio",
        TypedExclusionReason::LabelOnlyNoComposition,
    ),
    target_count_exclusion(
        "warp_stall_no_instruction_per_issue_active",
        "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio",
        TypedExclusionReason::LabelOnlyNoComposition,
    ),
    target_count_exclusion(
        "warp_stall_not_selected_per_issue_active",
        "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
        TypedExclusionReason::LabelOnlyNoComposition,
    ),
    target_count_exclusion(
        "warp_stall_selected_per_issue_active",
        "smsp__average_warps_issue_stalled_selected_per_issue_active.ratio",
        TypedExclusionReason::LabelOnlyNoComposition,
    ),
    target_count_exclusion(
        "warp_stall_short_scoreboard_per_issue_active",
        "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
        TypedExclusionReason::LabelOnlyNoComposition,
    ),
    target_count_exclusion(
        "warp_stall_sleeping_per_issue_active",
        "smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio",
        TypedExclusionReason::LabelOnlyNoComposition,
    ),
    target_count_exclusion(
        "warp_stall_tex_throttle_per_issue_active",
        "smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio",
        TypedExclusionReason::LabelOnlyNoComposition,
    ),
    target_count_exclusion(
        "warp_stall_wait_per_issue_active",
        "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
        TypedExclusionReason::LabelOnlyNoComposition,
    ),
    target_pct_exclusion(
        "throughput_sm_pct_peak",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        TypedExclusionReason::LabelOnlyNoComposition,
    ),
    target_pct_exclusion(
        "throughput_compute_memory_pct_peak",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        TypedExclusionReason::LabelOnlyNoComposition,
    ),
];

/// Return the numeric shape an emit fn actually produced.
fn metric_value_kind(value: &MetricValue) -> MetricKind {
    match value {
        MetricValue::Exact(_) => MetricKind::Count,
        MetricValue::Fraction { .. } => MetricKind::Fraction,
        MetricValue::BindingCycles(_) => MetricKind::BindingCycles,
    }
}

fn validate_descriptor_outcome(
    descriptor: &MetricDescriptor,
    outcome: &EmitOutcome,
) -> Result<(), EmitError> {
    let key = descriptor.key.as_str();
    if descriptor.kind == MetricKind::BindingCycles {
        match outcome {
            EmitOutcome::Predicted(MetricValue::BindingCycles(_)) => return Ok(()),
            EmitOutcome::RequiredUnimplemented {
                blocking: RequiredUnimplementedBlocker::AggregationSemanticsStudy,
            } => return Ok(()),
            _ => return Err(EmitError::BindingCyclesImplementationNotAdmitted { key }),
        }
    }

    match outcome {
        EmitOutcome::Predicted(value) => {
            let returned = metric_value_kind(value);
            if returned != descriptor.kind {
                return Err(EmitError::DescriptorValueKindMismatch {
                    key,
                    declared: descriptor.kind,
                    returned,
                });
            }
        }
        EmitOutcome::RequiredUnimplemented { .. } => {
            return Err(EmitError::RequiredUnimplementedKindMismatch {
                key,
                declared: descriptor.kind,
            });
        }
        EmitOutcome::Unavailable(_) => {}
    }
    Ok(())
}

fn emit_descriptor(
    snap: &CounterSnapshot,
    descriptor: &MetricDescriptor,
) -> Result<MetricPrediction, EmitError> {
    let outcome = if descriptor.available_on.contains(&snap.arch) {
        (descriptor.emit)(snap)?
    } else {
        EmitOutcome::Unavailable(Unavailable::ArchitectureUnsupported { arch: snap.arch })
    };
    validate_descriptor_outcome(descriptor, &outcome)?;
    Ok(MetricPrediction {
        arch: snap.arch,
        key: descriptor.key,
        ncu_metric: descriptor.ncu_metric,
        unit: descriptor.unit,
        outcome,
    })
}

fn emit_snapshot_in_order(
    snap: &CounterSnapshot,
    descriptors: [&MetricDescriptor; EMITTED_METRIC_COUNT],
) -> Result<[MetricPrediction; EMITTED_METRIC_COUNT], EmitError> {
    snap.require_counter_architecture()?;
    let [d0, d1, d2, d3, d4, d5, d6, d7] = descriptors;
    Ok([
        emit_descriptor(snap, d0)?,
        emit_descriptor(snap, d1)?,
        emit_descriptor(snap, d2)?,
        emit_descriptor(snap, d3)?,
        emit_descriptor(snap, d4)?,
        emit_descriptor(snap, d5)?,
        emit_descriptor(snap, d6)?,
        emit_descriptor(snap, d7)?,
    ])
}

fn emit_snapshot(
    snap: &CounterSnapshot,
) -> Result<[MetricPrediction; EMITTED_METRIC_COUNT], EmitError> {
    let [d0, d1, d2, d3, d4, d5, d6, d7] = &DESCRIPTORS;
    emit_snapshot_in_order(snap, [d0, d1, d2, d3, d4, d5, d6, d7])
}

fn emit_snapshot_v2(
    snap: &CounterSnapshot,
) -> Result<[MetricPrediction; EMITTED_METRIC_COUNT], EmitError> {
    let [d0, d1, d2, d3, d4, d5, d6, d7] = &DESCRIPTORS;
    emit_snapshot_in_order(snap, [d0, d1, d2, d6, d3, d4, d5, d7])
}

/// Emit every descriptor from one admitted, exact full-coverage kernel replay in
/// frozen result-v2 authority order. The returned private-construction bundle retains the
/// replay configuration, coverage witness, timing diagnostics, and unmodeled
/// tails; the binding-cycle row is derived by the checked frontier reducer.
///
/// Sampled observations are statically outside this API:
///
/// ```compile_fail
/// use sim_direct_run::emit::emit_launch;
/// use sim_sm::SampledKernelReplayResult;
///
/// fn reject_sampled(sampled: &SampledKernelReplayResult) {
///     let _ = emit_launch(sampled);
/// }
/// ```
///
/// Explicitly unadmitted diagnostics are also statically outside this API:
///
/// ```compile_fail
/// use sim_direct_run::emit::emit_launch;
/// use sim_sm::UnadmittedReplayDiagnostics;
///
/// fn reject_unadmitted(diagnostic: &UnadmittedReplayDiagnostics) {
///     let _ = emit_launch(diagnostic);
/// }
/// ```
pub fn emit_launch(result: &FullKernelReplayResult) -> Result<FullLaunchPredictions, EmitError> {
    let metrics = emit_snapshot_v2(&CounterSnapshot::from_full_launch(result)?)?;
    Ok(FullLaunchPredictions {
        configuration: result.configuration(),
        coverage: result.coverage().clone(),
        diagnostic_last_issue_cycles: result.diagnostic_last_issue_cycles(),
        last_processed_modeled_system_event_time: result.last_processed_modeled_system_event_time(),
        terminal_modeled_event_time: result.terminal_modeled_event_time(),
        modeled_completion_frontier: result.modeled_completion_frontier(),
        unmodeled_tails: result.unmodeled_tails(),
        metrics,
    })
}

/// Emit every descriptor from one checked, exact full-coverage launch-set
/// aggregate in stable descriptor order. The bundle preserves canonical
/// per-launch timing diagnostics and emits their checked max frontier. No
/// sampled observation or caller-built counter snapshot can enter this boundary.
///
/// ```compile_fail
/// use sim_direct_run::emit::emit_all;
/// use sim_sm::SampledKernelReplayResult;
///
/// fn reject_sampled(sampled: &SampledKernelReplayResult) {
///     let _ = emit_all(sampled);
/// }
/// ```
pub fn emit_all(counters: &FullLaunchSetCounters) -> Result<FullLaunchSetPredictions, EmitError> {
    let metrics = emit_snapshot(&CounterSnapshot::from_launch_set(counters)?)?;
    Ok(FullLaunchSetPredictions {
        scope: counters.scope().clone(),
        configuration: counters.configuration(),
        launch_coverages: counters.launch_coverages().to_vec(),
        launch_timing_diagnostics: counters.launch_timing_diagnostics().to_vec(),
        estimator_provenance: counters.estimator_provenance(),
        unmodeled_tails: counters.unmodeled_tails(),
        metrics,
    })
}

fn index_metrics_by_key(
    metrics: &[MetricPrediction],
) -> std::collections::BTreeMap<&'static str, &MetricPrediction> {
    metrics
        .iter()
        .map(|prediction| (prediction.key(), prediction))
        .collect()
}

/// A by-key view borrowed from a certified full-coverage launch-set bundle. Raw
/// metric slices remain deliberately rejected:
///
/// ```compile_fail
/// use sim_direct_run::emit::{MetricPrediction, index_by_key};
///
/// fn reject_detached(metrics: &[MetricPrediction]) {
///     let _ = index_by_key(metrics);
/// }
/// ```
pub fn index_by_key(
    predictions: &FullLaunchSetPredictions,
) -> std::collections::BTreeMap<&'static str, &MetricPrediction> {
    index_metrics_by_key(predictions.metrics())
}
