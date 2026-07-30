//! sim-emit: the NCU-metric emission layer (design §6.4).
//!
//! Takes checked counters for an exact full-coverage launch set, produces a
//! launch-set-bound vector of NCU metric descriptor outcomes, and classifies
//! every remaining observation contract in
//! `tools/local_labels/metric_registry.json` with a typed exclusion.
//! It is the only sim crate that names label-pipeline contracts: metric NAMES
//! are the output contract and are allowed here; label VALUES still never enter
//! (Rule 1, enforced by `eval/guards/test_targets_not_features_guard.py`).
//!
//! Shape (design proposal P3, "descriptor-table hybrid"): compiled
//! [`DESCRIPTORS`] and [`TYPED_EXCLUSIONS`] tables form an exact, disjoint
//! partition of the active registry. Descriptor rows bind complete registry
//! contracts to pure emit fns over raw counters; exclusion rows bind the same
//! contracts to closed typed reasons. Rate accounting lives HERE, never in the
//! cache counter structs (gate-2 altitude note c): `L1Counters`/`L2Counters`
//! hold raw counts; the emit fns compose them. Per-launch replay timing
//! diagnostics are retained on the bundle but are structurally absent from the
//! descriptor counter snapshot.
//!
//! Four states are kept distinct, never conflated into a silent zero:
//!   * a real simulator-scope zero — `Exact(0)` /
//!     `Fraction { value: Some(0.0), .. }`;
//!   * not-applicable — `Fraction { value: None, denominator: 0, .. }`
//!     (no accesses; NCU reports n/a — a legitimate scorable prediction);
//!   * not-produced-by-this-run — [`EmitOutcome::Unavailable`] (the replay's
//!     mem-mode lacks the counting unit, e.g. no L2 under `L1Const`);
//!   * mandatory but scientifically blocked —
//!     [`EmitOutcome::RequiredUnimplemented`] for binding cycles until an
//!     admitted aggregation-semantics study exists.
//!
//! LDGSTS folding (the §6.4 op_ld surface): NCU books LDGSTS (cp.async) global
//! loads under `op_ld` on the frozen sm_86 bundle — verified against the
//! bundle's own captured counts, NOT the metric-name taxonomy (a witness
//! kernel with only LDGSTS loads and no ordinary loads shows all of its
//! `op_ld` requests attributed to the async copies, with the total-request
//! rollup balancing exactly against `op_ld` + `op_st`). So the op_ld surface
//! folds the READ-class bypass stream (`bypass_sectors - bypass_atom_sectors`,
//! the async-copy loads) into its denominator; the ATOM write-class subset
//! stays out (NCU books it under op_atom/op_red). The sim models LDGSTS as
//! pure bypass with no L1 residency, so those sectors contribute to the
//! denominator but never to the hit numerator — an honest v0 under-prediction
//! whose residual localizes the un-modeled LDGSTS-L1-residency effect.

use serde::ser::{Serialize, SerializeStruct, Serializer};
use sim_arch::ArchId;
use sim_mem::{L1CounterError, L1Counters, L1OriginCounters, L2Counters, MemoryOrigin};
use sim_sm::{
    DramStats, FullCounterEstimatorProvenance, FullLaunchCoverage, FullLaunchSetCounters,
    FullLaunchTimingDiagnostic, L2SourceSplit, LaunchSetScope, ReplayConfiguration,
    SystemCounterError, TailFlags,
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
}

impl<'a> CounterSnapshot<'a> {
    fn from_launch_set(counters: &'a FullLaunchSetCounters) -> Self {
        Self {
            arch: counters.arch(),
            instructions_issued: counters.instructions_issued(),
            l1: counters.l1(),
            l2: counters.l2(),
            l2_by_source: counters.l2_by_source(),
            dram: counters.dram(),
        }
    }

    fn validate_provenance(&self) -> Result<(), EmitError> {
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
// child module. There is no constructor until an admitted aggregation-semantics
// study supplies the required rule identity; in particular, no `From<u64>` path
// can turn either replay diagnostic into a binding prediction.
mod binding_cycles {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct BindingCycles(u64);

    impl BindingCycles {
        pub fn get(self) -> u64 {
            self.0
        }
    }
}

/// Opaque binding-cycle payload. Replay timing diagnostics have no conversion
/// into this type:
///
/// ```compile_fail
/// use sim_emit::BindingCycles;
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
    /// A binding-cycle value. Safe construction remains unavailable until the
    /// aggregation-semantics study is admitted.
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
    /// A clock-cycle count. The binding cycle descriptor remains required-
    /// unimplemented until aggregation semantics are admitted.
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
    /// May emit only [`MetricValue::BindingCycles`], whose safe constructor is
    /// intentionally absent until the aggregation-semantics study is admitted.
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
    ArchitectureUnsupported { arch: ArchId },
    /// The replay's mem-mode did not attach the counting unit this metric
    /// reads (e.g. no L2 counters under `Flat`/`L1Const`).
    MemModeLacksUnit { needs: MemUnit },
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

/// One emitted metric, carrying its registry key + NCU identifier so the eval
/// join (downstream, outside `sim/`) can match against frozen labels by key.
/// Values remain borrowed from their [`FullLaunchSetPredictions`] bundle and are
/// intentionally neither `Clone` nor `Copy`.
///
/// ```compile_fail
/// use sim_emit::MetricPrediction;
///
/// fn detach(prediction: &MetricPrediction) -> MetricPrediction {
///     *prediction
/// }
/// ```
#[derive(Debug, PartialEq)]
pub struct MetricPrediction {
    arch: ArchId,
    key: &'static str,
    ncu_metric: &'static str,
    unit: Unit,
    outcome: EmitOutcome,
}

impl MetricPrediction {
    pub fn arch(&self) -> ArchId {
        self.arch
    }

    pub fn key(&self) -> &'static str {
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
        state.serialize_field("key", self.key)?;
        state.serialize_field("ncu_metric", self.ncu_metric)?;
        state.serialize_field("unit", &self.unit)?;
        state.serialize_field("outcome", &self.outcome)?;
        state.end()
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
    metrics: Vec<MetricPrediction>,
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
    pub key: &'static str,
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
    #[error(
        "binding-cycle metric {key} must remain required_unimplemented until the aggregation-semantics study is admitted"
    )]
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

// ---- per-metric emit fns (the rate accounting, moved out of the counters) ----

/// `gpc__cycles_elapsed.max` — mandatory publication-binding output. The
/// counter snapshot deliberately contains no replay temporal diagnostics, so
/// neither issue-frontier nor terminal-event timestamps can enter this row.
fn emit_exec_cycles_gpc_max(_: &CounterSnapshot) -> Result<EmitOutcome, EmitError> {
    Ok(EmitOutcome::RequiredUnimplemented {
        blocking: RequiredUnimplementedBlocker::AggregationSemanticsStudy,
    })
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
/// hit rate. Numerator = modeled eligible hits; denominator folds the
/// READ-class bypass (LDGSTS) sectors in (they miss L1 structurally in v0, so
/// they enter the denominator but contribute no hits — the honest residual).
fn emit_l1tex_read_hit_rate(snap: &CounterSnapshot) -> Result<EmitOutcome, EmitError> {
    let key = "cache_hit_rate_l1tex_read_sector";
    let Some(l1) = snap.l1 else {
        return Ok(EmitOutcome::Unavailable(Unavailable::MemModeLacksUnit {
            needs: MemUnit::L1Tex,
        }));
    };
    let global = l1.origin(MemoryOrigin::Global);
    let num = global.hits;
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

/// `l1tex__t_sector_hit_rate.pct` — the unified L1TEX sector hit rate over all
/// op classes. v0 numerator uses the sector-valid store/reduction rules
/// (`store_hits` / `reduction_hits`); the §6.4 line-TAG store rule
/// (`store_tag_hits`) is the carried alternative, unresolved until l1tex store
/// accounting is scored. The full bypass stream (atom included) is in the
/// denominator — `bypass_atom_sectors` must NOT be added on top because it is
/// already a subset of `bypass_sectors`. RED uses its distinct reduction lane,
/// so it contributes to this unified observer without entering `op_st`.
fn emit_l1tex_unified_hit_rate(snap: &CounterSnapshot) -> Result<EmitOutcome, EmitError> {
    let key = "cache_hit_rate_l1tex_sector";
    let Some(l1) = snap.l1 else {
        return Ok(EmitOutcome::Unavailable(Unavailable::MemModeLacksUnit {
            needs: MemUnit::L1Tex,
        }));
    };
    let num = checked_sum(
        key,
        "unified L1TEX hit-rate numerator",
        &[l1.hits, l1.store_hits, l1.reduction_hits],
    )?;
    let den = checked_sum(
        key,
        "unified L1TEX hit-rate denominator",
        &[
            l1.hits,
            l1.misses,
            l1.store_sectors,
            l1.reduction_sectors,
            l1.bypass_sectors,
            l1.untagged_sectors,
        ],
    )?;
    Ok(EmitOutcome::Predicted(MetricValue::fraction(
        key, num, den,
    )?))
}

/// `lts__t_sector_hit_rate.pct` — the unified L2 sector hit rate. Every LTS
/// access from all four L2 source lanes is in `l2.hits`/`l2.misses`;
/// `write_tag_hits` is already a subset of `hits` (do not add it).
fn emit_l2_hit_rate(snap: &CounterSnapshot) -> Result<EmitOutcome, EmitError> {
    let key = "cache_hit_rate_l2_sector";
    let Some(l2) = snap.l2 else {
        return Ok(EmitOutcome::Unavailable(Unavailable::MemModeLacksUnit {
            needs: MemUnit::Lts,
        }));
    };
    let den = checked_sum(
        key,
        "unified L2 hit-rate denominator",
        &[l2.hits, l2.misses],
    )?;
    Ok(EmitOutcome::Predicted(MetricValue::fraction(
        key, l2.hits, den,
    )?))
}

/// `dram__sectors_read.sum` — DRAM read sectors (v0 read-only demand proxy).
fn emit_dram_read_sectors(snap: &CounterSnapshot) -> Result<EmitOutcome, EmitError> {
    let Some(dram) = snap.dram else {
        return Ok(EmitOutcome::Unavailable(Unavailable::MemModeLacksUnit {
            needs: MemUnit::Dram,
        }));
    };
    Ok(EmitOutcome::Predicted(MetricValue::Exact(
        dram.read_sectors,
    )))
}

const SM86_ONLY: &[ArchId] = &[ArchId::Sm86];
const ALL_ARCHITECTURES: &[ArchId] = &[ArchId::Sm86, ArchId::Sm120];

/// The stable emission table. Seven simulator counters retain their established
/// evaluation order; the publication-binding cycle row is mandatory and remains
/// required-unimplemented.
pub const DESCRIPTORS: &[MetricDescriptor] = &[
    MetricDescriptor {
        key: "smsp_inst_executed_sum",
        ncu_metric: "smsp__inst_executed.sum",
        unit: Unit::Count,
        kind: MetricKind::Count,
        valid_max: RegistryValidMax::Unbounded,
        parser_transform: ParserTransform::IdentityNonnegative,
        model_io_role: ModelIoRole::Target,
        available_on: SM86_ONLY,
        emit: emit_smsp_inst_executed,
    },
    MetricDescriptor {
        key: "traffic_gmem_ld_sectors",
        ncu_metric: "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        unit: Unit::Count,
        kind: MetricKind::Count,
        valid_max: RegistryValidMax::Unbounded,
        parser_transform: ParserTransform::IdentityNonnegative,
        model_io_role: ModelIoRole::Target,
        available_on: SM86_ONLY,
        emit: emit_gmem_ld_sectors,
    },
    MetricDescriptor {
        key: "traffic_gmem_st_sectors",
        ncu_metric: "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
        unit: Unit::Count,
        kind: MetricKind::Count,
        valid_max: RegistryValidMax::Unbounded,
        parser_transform: ParserTransform::IdentityNonnegative,
        model_io_role: ModelIoRole::Target,
        available_on: SM86_ONLY,
        emit: emit_gmem_st_sectors,
    },
    MetricDescriptor {
        key: "cache_hit_rate_l1tex_read_sector",
        ncu_metric: "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct",
        unit: Unit::Pct,
        kind: MetricKind::Fraction,
        valid_max: RegistryValidMax::One,
        parser_transform: ParserTransform::PercentToFraction,
        model_io_role: ModelIoRole::Target,
        available_on: SM86_ONLY,
        emit: emit_l1tex_read_hit_rate,
    },
    MetricDescriptor {
        key: "cache_hit_rate_l1tex_sector",
        ncu_metric: "l1tex__t_sector_hit_rate.pct",
        unit: Unit::Pct,
        kind: MetricKind::Fraction,
        valid_max: RegistryValidMax::One,
        parser_transform: ParserTransform::PercentToFraction,
        model_io_role: ModelIoRole::Target,
        available_on: SM86_ONLY,
        emit: emit_l1tex_unified_hit_rate,
    },
    MetricDescriptor {
        key: "cache_hit_rate_l2_sector",
        ncu_metric: "lts__t_sector_hit_rate.pct",
        unit: Unit::Pct,
        kind: MetricKind::Fraction,
        valid_max: RegistryValidMax::One,
        parser_transform: ParserTransform::PercentToFraction,
        model_io_role: ModelIoRole::Target,
        available_on: SM86_ONLY,
        emit: emit_l2_hit_rate,
    },
    MetricDescriptor {
        key: "traffic_dram_read_sectors",
        ncu_metric: "dram__sectors_read.sum",
        unit: Unit::Count,
        kind: MetricKind::Count,
        valid_max: RegistryValidMax::Unbounded,
        parser_transform: ParserTransform::IdentityNonnegative,
        model_io_role: ModelIoRole::Target,
        available_on: SM86_ONLY,
        emit: emit_dram_read_sectors,
    },
    MetricDescriptor {
        key: "exec_cycles_gpc_max",
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
    if descriptor.kind == MetricKind::BindingCycles {
        if *outcome
            != (EmitOutcome::RequiredUnimplemented {
                blocking: RequiredUnimplementedBlocker::AggregationSemanticsStudy,
            })
        {
            return Err(EmitError::BindingCyclesImplementationNotAdmitted {
                key: descriptor.key,
            });
        }
        return Ok(());
    }

    match outcome {
        EmitOutcome::Predicted(value) => {
            let returned = metric_value_kind(value);
            if returned != descriptor.kind {
                return Err(EmitError::DescriptorValueKindMismatch {
                    key: descriptor.key,
                    declared: descriptor.kind,
                    returned,
                });
            }
        }
        EmitOutcome::RequiredUnimplemented { .. } => {
            return Err(EmitError::RequiredUnimplementedKindMismatch {
                key: descriptor.key,
                declared: descriptor.kind,
            });
        }
        EmitOutcome::Unavailable(_) => {}
    }
    Ok(())
}

fn emit_snapshot(snap: &CounterSnapshot) -> Result<Vec<MetricPrediction>, EmitError> {
    snap.validate_provenance()?;
    let mut out = Vec::with_capacity(DESCRIPTORS.len());
    for d in DESCRIPTORS {
        let outcome = if d.available_on.contains(&snap.arch) {
            (d.emit)(snap)?
        } else {
            EmitOutcome::Unavailable(Unavailable::ArchitectureUnsupported { arch: snap.arch })
        };
        validate_descriptor_outcome(d, &outcome)?;
        out.push(MetricPrediction {
            arch: snap.arch,
            key: d.key,
            ncu_metric: d.ncu_metric,
            unit: d.unit,
            outcome,
        });
    }
    Ok(out)
}

/// Emit every descriptor from one checked, exact full-coverage launch-set
/// aggregate in stable descriptor order. The bundle preserves canonical
/// per-launch timing diagnostics, but the descriptor snapshot omits them; the
/// binding cycle row therefore remains required-unimplemented. No sampled
/// observation or caller-built counter snapshot can enter this boundary.
///
/// ```compile_fail
/// use sim_emit::emit_all;
/// use sim_sm::SampledKernelReplayResult;
///
/// fn reject_sampled(sampled: &SampledKernelReplayResult) {
///     let _ = emit_all(sampled);
/// }
/// ```
pub fn emit_all(counters: &FullLaunchSetCounters) -> Result<FullLaunchSetPredictions, EmitError> {
    let metrics = emit_snapshot(&CounterSnapshot::from_launch_set(counters))?;
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
        .map(|prediction| (prediction.key, prediction))
        .collect()
}

/// A by-key view that remains borrowed from its certified launch-set bundle.
/// Raw metric slices are deliberately rejected:
///
/// ```compile_fail
/// use sim_emit::{MetricPrediction, index_by_key};
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use sim_mem::{L1CounterField, L1Counters, L2Counters, L2Source};
    use sim_sm::DramStats;

    fn predicted<'a>(preds: &'a [MetricPrediction], key: &str) -> &'a EmitOutcome {
        &preds
            .iter()
            .find(|p| p.key == key)
            .expect("key present")
            .outcome
    }

    /// Emit from owned counters retained for the duration of this call.
    fn emit(
        instructions_issued: u64,
        l1: Option<L1Counters>,
        l2: Option<L2Counters>,
        dram: Option<DramStats>,
    ) -> Result<Vec<MetricPrediction>, EmitError> {
        let l1 = l1.map(|counters| counters.with_unified_origin(MemoryOrigin::Global));
        let dram = dram.map(|stats| stats.with_unified_origin(MemoryOrigin::Global));
        emit_snapshot(&CounterSnapshot {
            arch: ArchId::Sm86,
            instructions_issued,
            l1: l1.as_ref(),
            l2: l2.as_ref(),
            l2_by_source: None,
            dram: dram.as_ref(),
        })
    }

    #[test]
    fn descriptor_value_kind_mismatch_is_typed_in_all_build_profiles() {
        let descriptor = DESCRIPTORS
            .iter()
            .find(|descriptor| descriptor.key == "smsp_inst_executed_sum")
            .unwrap();
        let outcome = EmitOutcome::Predicted(MetricValue::Fraction {
            value: Some(0.5),
            numerator: 1,
            denominator: 2,
        });
        assert_eq!(
            validate_descriptor_outcome(descriptor, &outcome),
            Err(EmitError::DescriptorValueKindMismatch {
                key: "smsp_inst_executed_sum",
                declared: MetricKind::Count,
                returned: MetricKind::Fraction,
            })
        );
    }

    #[test]
    fn template_architecture_keeps_mandatory_cycles_required_unimplemented() {
        let snapshot = CounterSnapshot {
            arch: ArchId::Sm120,
            instructions_issued: 17,
            l1: None,
            l2: None,
            l2_by_source: None,
            dram: None,
        };
        let preds = emit_snapshot(&snapshot).unwrap();
        assert_eq!(
            predicted(&preds, "exec_cycles_gpc_max"),
            &EmitOutcome::RequiredUnimplemented {
                blocking: RequiredUnimplementedBlocker::AggregationSemanticsStudy,
            }
        );
        assert!(
            preds
                .iter()
                .filter(|prediction| prediction.key != "exec_cycles_gpc_max")
                .all(|prediction| {
                    prediction.arch == ArchId::Sm120
                        && prediction.outcome
                            == EmitOutcome::Unavailable(Unavailable::ArchitectureUnsupported {
                                arch: ArchId::Sm120,
                            })
                })
        );
    }

    #[test]
    fn binding_cycle_descriptor_rejects_every_nonblocking_outcome() {
        let descriptor = DESCRIPTORS
            .iter()
            .find(|descriptor| descriptor.key == "exec_cycles_gpc_max")
            .unwrap();
        assert_eq!(
            validate_descriptor_outcome(
                descriptor,
                &EmitOutcome::Unavailable(Unavailable::ArchitectureUnsupported {
                    arch: ArchId::Sm86,
                }),
            ),
            Err(EmitError::BindingCyclesImplementationNotAdmitted {
                key: "exec_cycles_gpc_max",
            })
        );
    }

    #[test]
    fn emitted_wire_values_mark_simulator_scope_but_cycles_stay_blocked() {
        let preds = emit(42, None, None, None).unwrap();
        let instructions = preds
            .iter()
            .find(|prediction| prediction.key == "smsp_inst_executed_sum")
            .unwrap();
        assert_eq!(
            instructions.numeric_semantics(),
            Some(NumericSemantics::SimulatorScope)
        );
        assert_eq!(
            serde_json::to_value(instructions).unwrap(),
            serde_json::json!({
                "arch": "sm_86",
                "key": "smsp_inst_executed_sum",
                "ncu_metric": "smsp__inst_executed.sum",
                "unit": "count",
                "outcome": {
                    "status": "predicted",
                    "value": {
                        "kind": "exact",
                        "semantics": "simulator_scope",
                        "value": 42,
                    },
                },
            })
        );

        let cycles = preds
            .iter()
            .find(|prediction| prediction.key == "exec_cycles_gpc_max")
            .unwrap();
        assert_eq!(cycles.numeric_semantics(), None);
        assert_eq!(
            serde_json::to_value(cycles).unwrap(),
            serde_json::json!({
                "arch": "sm_86",
                "key": "exec_cycles_gpc_max",
                "ncu_metric": "gpc__cycles_elapsed.max",
                "unit": "cycles",
                "outcome": {
                    "status": "required_unimplemented",
                    "blocking": "aggregation_semantics_study",
                },
            })
        );
    }

    #[test]
    fn counter_provenance_is_required_and_cannot_be_relabelled() {
        let unbound = L1Counters::default();
        let snapshot = CounterSnapshot {
            arch: ArchId::Sm86,
            instructions_issued: 1,
            l1: Some(&unbound),
            l2: None,
            l2_by_source: None,
            dram: None,
        };
        assert_eq!(
            emit_snapshot(&snapshot),
            Err(EmitError::CounterArchitectureUnbound {
                counter: "L1 counters"
            })
        );

        let sm86 = L1Counters::for_arch(ArchId::Sm86);
        let relabelled = CounterSnapshot {
            arch: ArchId::Sm120,
            instructions_issued: 1,
            l1: Some(&sm86),
            l2: None,
            l2_by_source: None,
            dram: None,
        };
        assert_eq!(
            emit_snapshot(&relabelled),
            Err(EmitError::CounterArchitectureMismatch {
                counter: "L1 counters",
                expected: ArchId::Sm120,
                got: ArchId::Sm86,
            })
        );
    }

    #[test]
    fn emission_rejects_incoherent_origin_partitions() {
        let mut l1 = L1Counters::for_arch(ArchId::Sm86);
        l1.hits = 1;
        let l1_snapshot = CounterSnapshot {
            arch: ArchId::Sm86,
            instructions_issued: 1,
            l1: Some(&l1),
            l2: None,
            l2_by_source: None,
            dram: None,
        };
        assert_eq!(
            emit_snapshot(&l1_snapshot),
            Err(EmitError::L1Counters(
                L1CounterError::OriginPartitionMismatch {
                    field: L1CounterField::Hits,
                    unified: 1,
                    global: 0,
                    local: 0,
                }
            ))
        );

        let mut split = L2SourceSplit::for_arch(ArchId::Sm86);
        split.sectors[L2Source::L1Fill as usize] = 1;
        let split_snapshot = CounterSnapshot {
            arch: ArchId::Sm86,
            instructions_issued: 1,
            l1: None,
            l2: None,
            l2_by_source: Some(&split),
            dram: None,
        };
        assert_eq!(
            emit_snapshot(&split_snapshot),
            Err(EmitError::SystemCounters(
                SystemCounterError::L2OriginPartitionMismatch {
                    counter: "sectors",
                    l2_source: L2Source::L1Fill,
                    unified: 1,
                    global: 0,
                    local: 0,
                }
            ))
        );

        let mut dram = DramStats::for_arch(ArchId::Sm86);
        dram.read_sectors = 1;
        let dram_snapshot = CounterSnapshot {
            arch: ArchId::Sm86,
            instructions_issued: 1,
            l1: None,
            l2: None,
            l2_by_source: None,
            dram: Some(&dram),
        };
        assert_eq!(
            emit_snapshot(&dram_snapshot),
            Err(EmitError::SystemCounters(
                SystemCounterError::DramOriginPartitionMismatch {
                    counter: "read sectors",
                    unified: 1,
                    global: 0,
                    local: 0,
                }
            ))
        );
    }

    #[test]
    fn fraction_is_none_at_zero_denominator_never_zero() {
        let v = MetricValue::fraction("k", 0, 0).unwrap();
        assert_eq!(
            v,
            MetricValue::Fraction {
                value: None,
                numerator: 0,
                denominator: 0
            }
        );
    }

    #[test]
    fn fraction_real_zero_is_some_zero() {
        let MetricValue::Fraction { value, .. } = MetricValue::fraction("k", 0, 10).unwrap() else {
            panic!("expected fraction");
        };
        assert_eq!(value, Some(0.0));
    }

    #[test]
    fn fraction_rounds_the_exact_counter_ratio_once() {
        let numerator = 1_u64 << 53;
        let denominator = numerator + 1;
        let MetricValue::Fraction { value, .. } =
            MetricValue::fraction("k", numerator, denominator).unwrap()
        else {
            panic!("expected fraction");
        };
        let below_one = f64::from_bits(1.0_f64.to_bits() - 1);
        assert_eq!(value, Some(below_one));
        assert!(value.unwrap() < 1.0);
    }

    #[test]
    fn exact_fraction_conversion_covers_the_full_u64_domain() {
        assert_eq!(exact_fraction_to_f64("k", 1, 2).unwrap(), 0.5);
        assert_eq!(
            exact_fraction_to_f64("k", 1, u64::MAX).unwrap(),
            2.0_f64.powi(-64)
        );
        // Exact midpoints round to the neighbor with an even low bit.
        let tie_denominator = 1_u64 << 54;
        let half_bits = 0.5_f64.to_bits();
        assert_eq!(
            exact_fraction_to_f64("k", (1_u64 << 53) + 1, tie_denominator).unwrap(),
            0.5
        );
        assert_eq!(
            exact_fraction_to_f64("k", (1_u64 << 53) + 3, tie_denominator).unwrap(),
            f64::from_bits(half_bits + 2)
        );
        // This ratio is genuinely closer to 1.0 than to its lower neighbor.
        assert_eq!(
            exact_fraction_to_f64("k", u64::MAX - 1, u64::MAX).unwrap(),
            1.0
        );
    }

    #[test]
    fn exact_fraction_conversion_matches_exhaustive_small_ratios() {
        for denominator in 1_u32..=512 {
            for numerator in 0..=denominator {
                let exact =
                    exact_fraction_to_f64("k", u64::from(numerator), u64::from(denominator))
                        .unwrap();
                let single_round_reference = f64::from(numerator) / f64::from(denominator);
                assert_eq!(
                    exact.to_bits(),
                    single_round_reference.to_bits(),
                    "{numerator}/{denominator}"
                );
            }
        }
    }

    #[test]
    fn rate_above_unity_is_a_typed_error_not_a_clamp() {
        assert_eq!(
            MetricValue::fraction("k", 11, 10),
            Err(EmitError::RateExceedsUnity {
                key: "k",
                num: 11,
                den: 10
            })
        );
    }

    #[test]
    fn global_descriptors_exclude_local_origin_while_unified_rate_includes_it() {
        let global = L1OriginCounters {
            hits: 6,
            misses: 4,
            bypass_sectors: 3,
            bypass_atom_sectors: 1,
            store_sectors: 5,
            store_hits: 2,
            ..L1OriginCounters::default()
        };
        let local = L1OriginCounters {
            hits: 20,
            misses: 10,
            store_sectors: 7,
            store_hits: 3,
            untagged_sectors: 2,
            ..L1OriginCounters::default()
        };
        let l1 = L1Counters::from_origins(ArchId::Sm86, global, local).unwrap();
        let snapshot = CounterSnapshot {
            arch: ArchId::Sm86,
            instructions_issued: 1,
            l1: Some(&l1),
            l2: None,
            l2_by_source: None,
            dram: None,
        };
        let preds = emit_snapshot(&snapshot).unwrap();

        assert_eq!(
            predicted(&preds, "traffic_gmem_ld_sectors"),
            &EmitOutcome::Predicted(MetricValue::Exact(12))
        );
        assert_eq!(
            predicted(&preds, "traffic_gmem_st_sectors"),
            &EmitOutcome::Predicted(MetricValue::Exact(5))
        );
        assert_eq!(
            predicted(&preds, "cache_hit_rate_l1tex_read_sector"),
            &EmitOutcome::Predicted(MetricValue::Fraction {
                value: Some(0.5),
                numerator: 6,
                denominator: 12,
            })
        );
        assert_eq!(
            predicted(&preds, "cache_hit_rate_l1tex_sector"),
            &EmitOutcome::Predicted(MetricValue::Fraction {
                value: Some(31.0 / 57.0),
                numerator: 31,
                denominator: 57,
            })
        );
    }

    #[test]
    fn reduction_contributes_only_to_the_unified_l1tex_observer() {
        let mut l1 = L1Counters::for_arch(ArchId::Sm86);
        l1.store_sectors = 5;
        l1.store_hits = 1;
        l1.reduction_sectors = 3;
        l1.reduction_hits = 2;
        let preds = emit(1, Some(l1), None, None).unwrap();

        assert_eq!(
            predicted(&preds, "traffic_gmem_st_sectors"),
            &EmitOutcome::Predicted(MetricValue::Exact(5))
        );
        assert_eq!(
            predicted(&preds, "cache_hit_rate_l1tex_sector"),
            &EmitOutcome::Predicted(MetricValue::Fraction {
                value: Some(3.0 / 8.0),
                numerator: 3,
                denominator: 8,
            })
        );
    }

    #[test]
    fn op_ld_surface_folds_read_class_bypass_ldgsts_into_the_denominator() {
        // Witness shape: a kernel of ONLY LDGSTS loads (all bypass, none
        // eligible, no atoms) — mirrors gemm_hmma_bf16. The op_ld sector
        // count is the bypass-read stream, and the modeled hit rate is 0
        // (v0 has no L1 residency for bypass) — the honest under-prediction.
        let mut l1 = L1Counters::for_arch(ArchId::Sm86);
        l1.bypass_sectors = 196_608;
        l1.bypass_atom_sectors = 0;
        l1.store_sectors = 16_384;
        let preds = emit(197_120, Some(l1), None, None).unwrap();

        assert_eq!(
            predicted(&preds, "traffic_gmem_ld_sectors"),
            &EmitOutcome::Predicted(MetricValue::Exact(196_608))
        );
        assert_eq!(
            predicted(&preds, "cache_hit_rate_l1tex_read_sector"),
            &EmitOutcome::Predicted(MetricValue::Fraction {
                value: Some(0.0),
                numerator: 0,
                denominator: 196_608
            })
        );
        assert_eq!(
            predicted(&preds, "traffic_gmem_st_sectors"),
            &EmitOutcome::Predicted(MetricValue::Exact(16_384))
        );
        // The instruction count needs no mem unit and is always emitted.
        assert_eq!(
            predicted(&preds, "smsp_inst_executed_sum"),
            &EmitOutcome::Predicted(MetricValue::Exact(197_120))
        );
    }

    #[test]
    fn op_ld_denominator_excludes_the_atom_write_class_bypass_subset() {
        // 100 bypass sectors, 40 of them ATOM (write-class -> op_atom/op_red,
        // NOT op_ld). The op_ld surface sees only the 60 READ-class sectors.
        let mut l1 = L1Counters::for_arch(ArchId::Sm86);
        l1.hits = 10;
        l1.misses = 5;
        l1.bypass_sectors = 100;
        l1.bypass_atom_sectors = 40;
        let preds = emit(1, Some(l1), None, None).unwrap();
        assert_eq!(
            predicted(&preds, "traffic_gmem_ld_sectors"),
            &EmitOutcome::Predicted(MetricValue::Exact(10 + 5 + 60))
        );
    }

    #[test]
    fn atom_subset_exceeding_bypass_is_a_typed_counter_invariant() {
        let mut l1 = L1Counters::for_arch(ArchId::Sm86);
        l1.bypass_sectors = 3;
        l1.bypass_atom_sectors = 4;
        assert_eq!(
            emit(1, Some(l1), None, None),
            Err(EmitError::CounterInvariant {
                key: "traffic_gmem_ld_sectors",
                detail: "bypass_atom_sectors exceeds bypass_sectors (subset invariant broken)",
            })
        );
    }

    #[test]
    fn global_load_sector_total_overflow_is_typed() {
        let mut l1 = L1Counters::for_arch(ArchId::Sm86);
        l1.misses = u64::MAX;
        l1.bypass_sectors = 2;
        assert_eq!(
            emit(1, Some(l1), None, None),
            Err(EmitError::CounterOverflow {
                key: "traffic_gmem_ld_sectors",
                operation: "global-load sector total",
            })
        );
    }

    #[test]
    fn global_load_hit_rate_denominator_overflow_is_typed() {
        let mut l1 = L1Counters::for_arch(ArchId::Sm86);
        l1.misses = u64::MAX;
        l1.bypass_sectors = 2;
        let l1 = l1.with_unified_origin(MemoryOrigin::Global);
        let snapshot = CounterSnapshot {
            arch: ArchId::Sm86,
            instructions_issued: 1,
            l1: Some(&l1),
            l2: None,
            l2_by_source: None,
            dram: None,
        };
        assert_eq!(
            emit_l1tex_read_hit_rate(&snapshot),
            Err(EmitError::CounterOverflow {
                key: "cache_hit_rate_l1tex_read_sector",
                operation: "global-load hit-rate denominator",
            })
        );
    }

    #[test]
    fn unified_l1tex_hit_numerator_overflow_is_typed() {
        let mut l1 = L1Counters::for_arch(ArchId::Sm86);
        l1.hits = u64::MAX;
        l1.store_hits = 1;
        l1.store_sectors = 1;
        assert_eq!(
            emit(1, Some(l1), None, None),
            Err(EmitError::CounterOverflow {
                key: "cache_hit_rate_l1tex_sector",
                operation: "unified L1TEX hit-rate numerator",
            })
        );
    }

    #[test]
    fn unified_l1tex_hit_denominator_overflow_is_typed() {
        let mut l1 = L1Counters::for_arch(ArchId::Sm86);
        l1.misses = u64::MAX;
        l1.store_sectors = 1;
        assert_eq!(
            emit(1, Some(l1), None, None),
            Err(EmitError::CounterOverflow {
                key: "cache_hit_rate_l1tex_sector",
                operation: "unified L1TEX hit-rate denominator",
            })
        );
    }

    #[test]
    fn unified_l2_hit_denominator_overflow_is_typed() {
        let mut l2 = L2Counters::for_arch(ArchId::Sm86);
        l2.hits = u64::MAX;
        l2.misses = 1;
        assert_eq!(
            emit(1, None, Some(l2), None),
            Err(EmitError::CounterOverflow {
                key: "cache_hit_rate_l2_sector",
                operation: "unified L2 hit-rate denominator",
            })
        );
    }

    #[test]
    fn missing_mem_unit_is_unavailable_not_an_error_or_a_zero() {
        // Flat mode: no L1/L2/DRAM. Only the instruction count is produced;
        // every cache metric is typed-unavailable with its needed unit.
        let preds = emit(42, None, None, None).unwrap();
        assert_eq!(
            predicted(&preds, "smsp_inst_executed_sum"),
            &EmitOutcome::Predicted(MetricValue::Exact(42))
        );
        assert_eq!(
            predicted(&preds, "cache_hit_rate_l1tex_read_sector"),
            &EmitOutcome::Unavailable(Unavailable::MemModeLacksUnit {
                needs: MemUnit::L1Tex
            })
        );
        assert_eq!(
            predicted(&preds, "cache_hit_rate_l2_sector"),
            &EmitOutcome::Unavailable(Unavailable::MemModeLacksUnit {
                needs: MemUnit::Lts
            })
        );
        assert_eq!(
            predicted(&preds, "traffic_dram_read_sectors"),
            &EmitOutcome::Unavailable(Unavailable::MemModeLacksUnit {
                needs: MemUnit::Dram
            })
        );
    }

    #[test]
    fn l2_and_dram_metrics_predict_when_their_units_are_present() {
        let mut l2 = L2Counters::for_arch(ArchId::Sm86);
        l2.hits = 3;
        l2.misses = 1;
        let mut dram = DramStats::for_arch(ArchId::Sm86);
        dram.read_sectors = 8415;
        dram.bw_delayed_responses = 0;
        let preds = emit(1, None, Some(l2), Some(dram)).unwrap();
        assert_eq!(
            predicted(&preds, "cache_hit_rate_l2_sector"),
            &EmitOutcome::Predicted(MetricValue::Fraction {
                value: Some(0.75),
                numerator: 3,
                denominator: 4
            })
        );
        assert_eq!(
            predicted(&preds, "traffic_dram_read_sectors"),
            &EmitOutcome::Predicted(MetricValue::Exact(8415))
        );
    }

    #[test]
    fn index_by_key_covers_every_descriptor() {
        let preds = emit(1, None, None, None).unwrap();
        let idx = index_metrics_by_key(&preds);
        assert_eq!(idx.len(), DESCRIPTORS.len());
        for d in DESCRIPTORS {
            assert!(
                idx.contains_key(d.key),
                "descriptor {} missing from index",
                d.key
            );
        }
    }

    #[test]
    fn descriptor_keys_are_unique() {
        let mut keys: Vec<&str> = DESCRIPTORS.iter().map(|d| d.key).collect();
        keys.sort_unstable();
        let n = keys.len();
        keys.dedup();
        assert_eq!(keys.len(), n, "duplicate descriptor keys");
    }
}
