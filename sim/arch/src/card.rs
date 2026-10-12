//! `sim_arch::card`: loader for the per-arch calibration cards
//! (`arch_profiles/cards/*.json`).
//!
//! The simulator's latencies, geometries, and clock anchors come from these
//! cards — which are filled from microbenchmarks (pchase) and vendor
//! documentation, never from NCU labels (Rule 1: targets-never-features).
//!
//! Fail-closed discipline: a card field recorded as a `research_gap` is a
//! typed error when the simulator requires it — there are no silent default
//! latencies. The schema's `objOrGap` pattern applies at BLOCK level too
//! SM120 cards ship `l1_geometry` itself as a research_gap
//! object), so gap-able blocks deserialize through [`Carded`]. An object
//! carrying BOTH a payload and a `status` key is rejected outright rather
//! than silently admitted as either arm. The Python side enforces
//! card↔evidence lockstep (`arch_profiles/tests/test_arch_profile_lockstep.py`);
//! this crate's own lockstep test asserts the Rust loader sees the identical
//! sm_86 values.

use std::collections::BTreeSet;
use std::io::Read;
use std::path::Path;

use crate::json::strict_json_from_slice;
use crate::{
    ArchError, ArchId, ArchIdentity, BareLdgECardDeclaration, BareLdgECardDeclarationError,
    BareLdgEDefault, BuildArch, CapabilityError, CapabilityStatus, Sku,
};
use serde::Deserialize;
use serde::de::DeserializeOwned;

const MAX_ARCH_CARD_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum CalibError {
    #[error("could not read card {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("card {path} is {observed} bytes, exceeding implementation limit {limit}")]
    ResourceLimit {
        path: String,
        observed: u64,
        limit: u64,
    },
    #[error("could not parse card {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("card architecture identity is invalid: {0}")]
    Architecture(#[from] ArchError),
    #[error("architecture capability is unavailable: {0}")]
    Capability(#[from] CapabilityError),
    #[error("card {arch} build_arch_family must contain at least one build token")]
    BuildArchFamilyEmpty { arch: ArchId },
    #[error("card build_arch_family token {token:?} is invalid: {source}")]
    BuildArchFamilyTokenInvalid {
        token: String,
        #[source]
        source: ArchError,
    },
    #[error("card build_arch_family repeats token {token:?}")]
    BuildArchFamilyDuplicate { token: String },
    #[error(
        "card {arch} build_arch_family token {token:?} belongs to another deployment architecture"
    )]
    BuildArchFamilyArchitectureMismatch { arch: ArchId, token: String },
    #[error("card {arch} build_arch_family does not admit requested build {build:?}")]
    BuildArchNotInCardFamily { arch: ArchId, build: String },
    #[error("card {arch} has no bare_ldg_e_default declaration")]
    BareLdgDeclarationMissing { arch: ArchId },
    #[error("card bare_ldg_e_default declaration is invalid: {0}")]
    BareLdgDeclaration(#[from] BareLdgECardDeclarationError),
    #[error(
        "card {arch} bare_ldg_e_default declaration {declaration:?} contradicts runtime capability {status:?}"
    )]
    BareLdgCapabilityMismatch {
        arch: ArchId,
        declaration: BareLdgECardDeclaration,
        status: CapabilityStatus,
    },
    #[error("loaded card architecture was relabeled from {loaded} to {current}")]
    CardArchitectureRelabeled { loaded: ArchId, current: ArchId },
    #[error("loaded card SKU was relabeled from {loaded:?} to {current:?}")]
    CardSkuRelabeled { loaded: Sku, current: Sku },
    #[error(
        "card {arch_id} compute_capability={got:?} disagrees with canonical architecture {expected:?}"
    )]
    ComputeCapabilityMismatch {
        arch_id: String,
        got: (u32, u32),
        expected: (u32, u32),
    },
    #[error("card {arch_id} has no latencies_cycles block (pre-Phase-1 card?)")]
    LatenciesMissing { arch_id: String },
    #[error(
        "card {arch_id}: latency '{kind}' is a research_gap ({reason}); the simulator must not \
         invent a default — close the gap with a microbenchmark or type-exclude the consumer"
    )]
    ResearchGap {
        arch_id: String,
        kind: LatencyKind,
        reason: String,
    },
    #[error(
        "card {arch_id}: block '{block}' is a research_gap ({reason}; challenge={challenge:?}); the consumer must \
         type-exclude, never guess geometry"
    )]
    BlockResearchGap {
        arch_id: String,
        block: &'static str,
        reason: String,
        challenge: Option<String>,
    },
    #[error("card field {field} = {value} violates '{requirement}' — fail closed, never clamp")]
    InvalidCardValue {
        field: &'static str,
        value: String,
        requirement: &'static str,
    },
    #[error(
        "card block '{block}' has no bound source architecture; load it through ArchCard before consuming it"
    )]
    BlockArchitectureUnbound { block: &'static str },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LatencyKind {
    L1Hit,
    L2Hit,
    Dram,
    Shared,
    MshrOutstandingMisses,
}

impl LatencyKind {
    const fn value_field(self) -> &'static str {
        match self {
            Self::L1Hit => "latencies_cycles.l1_hit.value",
            Self::L2Hit => "latencies_cycles.l2_hit.value",
            Self::Dram => "latencies_cycles.dram.value",
            Self::Shared => "latencies_cycles.shared.value",
            Self::MshrOutstandingMisses => "latencies_cycles.mshr_outstanding_misses.value",
        }
    }

    const fn dispersion_field(self) -> &'static str {
        match self {
            Self::L1Hit => "latencies_cycles.l1_hit.dispersion",
            Self::L2Hit => "latencies_cycles.l2_hit.dispersion",
            Self::Dram => "latencies_cycles.dram.dispersion",
            Self::Shared => "latencies_cycles.shared.dispersion",
            Self::MshrOutstandingMisses => "latencies_cycles.mshr_outstanding_misses.dispersion",
        }
    }
}

impl std::fmt::Display for LatencyKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::L1Hit => "l1_hit",
            Self::L2Hit => "l2_hit",
            Self::Dram => "dram",
            Self::Shared => "shared",
            Self::MshrOutstandingMisses => "mshr_outstanding_misses",
        };
        f.write_str(s)
    }
}

/// Closed enum so a typo'd status never deserializes into a gap arm.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
pub enum GapStatus {
    #[serde(rename = "research_gap")]
    ResearchGap,
}

/// A measured scalar card property: value + dispersion + unit.
#[derive(Debug, Clone)]
pub struct MeasuredValue {
    pub value: f64,
    pub dispersion: Option<f64>,
    pub unit: Option<String>,
}

/// A scalar property that is either measured or a typed research gap.
/// Mirrors the ArchProfile schema's `objOrGap` shape. Deserialization is
/// manual (NOT `#[serde(untagged)]`): an object with BOTH `value` and
/// `status` keys — a half-migrated card — must fail parsing, not silently
/// win whichever untagged arm matches first.
#[derive(Debug, Clone)]
pub enum Measured {
    Value(MeasuredValue),
    Gap {
        reason: String,
        challenge: Option<String>,
    },
}

#[derive(Debug, Deserialize)]
struct RawObjOrGap {
    #[serde(default)]
    value: Option<f64>,
    #[serde(default)]
    dispersion: Option<f64>,
    #[serde(default)]
    unit: Option<String>,
    #[serde(default)]
    status: Option<GapStatus>,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    challenge: Option<String>,
}

impl<'de> Deserialize<'de> for Measured {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        let object = value
            .as_object()
            .ok_or_else(|| serde::de::Error::custom("objOrGap entry must be an object"))?;
        let has_value = object.contains_key("value");
        let has_status = object.contains_key("status");

        if has_status {
            let foreign = object
                .keys()
                .any(|key| !matches!(key.as_str(), "status" | "reason" | "challenge"));
            if has_value || foreign {
                return Err(serde::de::Error::custom(
                    "objOrGap entry carries BOTH value-arm payload and gap status/metadata",
                ));
            }
        } else if has_value {
            let foreign = object
                .keys()
                .any(|key| !matches!(key.as_str(), "value" | "dispersion" | "unit"));
            if foreign {
                return Err(serde::de::Error::custom(
                    "objOrGap value entry carries BOTH value payload and gap/foreign metadata",
                ));
            }
        }

        let raw: RawObjOrGap = serde_json::from_value(value)
            .map_err(|error| serde::de::Error::custom(format!("objOrGap entry: {error}")))?;
        match (raw.value, raw.status) {
            (Some(value), None) => Ok(Measured::Value(MeasuredValue {
                value,
                dispersion: raw.dispersion,
                unit: raw.unit,
            })),
            (None, Some(GapStatus::ResearchGap)) => {
                let reason = raw.reason.ok_or_else(|| {
                    serde::de::Error::custom(
                        "objOrGap research_gap entry is missing required string `reason`",
                    )
                })?;
                Ok(Measured::Gap {
                    reason,
                    challenge: raw.challenge,
                })
            }
            (Some(_), Some(_)) => Err(serde::de::Error::custom(
                "objOrGap entry carries BOTH `value` and `status: research_gap` — \
                 half-migrated card, refusing to pick an arm",
            )),
            (None, None) => Err(serde::de::Error::custom(
                "objOrGap entry carries NEITHER `value` nor `status: research_gap`",
            )),
        }
    }
}

/// A boolean declaration that may remain a nested research gap. Manual
/// deserialization keeps the boolean and gap arms closed and disjoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoolOrResearchGap {
    Value(bool),
    Gap {
        reason: String,
        challenge: Option<String>,
    },
}

#[derive(Debug, Deserialize)]
struct RawBoolResearchGap {
    status: GapStatus,
    reason: String,
    #[serde(default)]
    challenge: Option<String>,
}

impl<'de> Deserialize<'de> for BoolOrResearchGap {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        if let Some(value) = value.as_bool() {
            return Ok(Self::Value(value));
        }
        let object = value.as_object().ok_or_else(|| {
            serde::de::Error::custom("boolOrResearchGap must be a boolean or research_gap object")
        })?;
        if object
            .keys()
            .any(|key| !matches!(key.as_str(), "status" | "reason" | "challenge"))
        {
            return Err(serde::de::Error::custom(
                "boolOrResearchGap research_gap arm carries foreign payload",
            ));
        }
        let raw: RawBoolResearchGap = serde_json::from_value(value).map_err(|error| {
            serde::de::Error::custom(format!("boolOrResearchGap entry: {error}"))
        })?;
        let GapStatus::ResearchGap = raw.status;
        Ok(Self::Gap {
            reason: raw.reason,
            challenge: raw.challenge,
        })
    }
}

/// A whole card block whose payload is bound to the architecture card that
/// supplied it.  The source binding prevents a caller from presenting a
/// present sm_86 geometry block as sm_120 data.
#[derive(Debug, Clone)]
pub struct Carded<T> {
    state: CardedState<T>,
    source_arch: Option<ArchId>,
}

#[derive(Debug, Clone)]
enum CardedState<T> {
    Present(T),
    Gap {
        reason: String,
        challenge: Option<String>,
    },
}

impl<T> Carded<T> {
    fn bind_arch(mut self, arch: ArchId) -> Self {
        self.source_arch = Some(arch);
        self
    }

    pub fn require(&self, block: &'static str) -> Result<&T, CalibError> {
        let arch = self
            .source_arch
            .ok_or(CalibError::BlockArchitectureUnbound { block })?;
        match &self.state {
            CardedState::Present(t) => Ok(t),
            CardedState::Gap { reason, challenge } => Err(CalibError::BlockResearchGap {
                arch_id: arch.to_string(),
                block,
                reason: reason.clone(),
                challenge: challenge.clone(),
            }),
        }
    }

    fn require_for_arch(&self, expected: ArchId, block: &'static str) -> Result<&T, CalibError> {
        let source = self
            .source_arch
            .ok_or(CalibError::BlockArchitectureUnbound { block })?;
        if source != expected {
            return Err(CalibError::InvalidCardValue {
                field: block,
                value: source.to_string(),
                requirement: "block provenance must match card architecture",
            });
        }
        self.require(block)
    }

    /// Serde default for blocks newer than a card: a field the card
    /// never carried is a GAP (fail closed at `require`), never a value.
    pub fn absent_gap() -> Self {
        Self {
            state: CardedState::Gap {
                reason:
                    "field absent from this card (card predates the block, OR the card's key is \
                     misspelled — unknown keys are ignored on parse)"
                        .to_owned(),
                challenge: None,
            },
            source_arch: None,
        }
    }
}

impl<'de, T> Deserialize<'de> for Carded<T>
where
    T: DeserializeOwned,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = serde_json::Value::deserialize(deserializer)?;
        let status = raw.get("status");
        if let Some(status) = status {
            let has_payload = raw
                .as_object()
                .map(|object| {
                    object
                        .keys()
                        .any(|key| !matches!(key.as_str(), "status" | "reason" | "challenge"))
                })
                .unwrap_or(true);
            if has_payload {
                return Err(serde::de::Error::custom(
                    "card block carries both a payload and status: research_gap",
                ));
            }
            let gap: GapStatus = serde_json::from_value(status.clone())
                .map_err(|e| serde::de::Error::custom(format!("bad gap status: {e}")))?;
            let GapStatus::ResearchGap = gap;
            let reason = raw
                .get("reason")
                .and_then(|reason| reason.as_str())
                .ok_or_else(|| {
                    serde::de::Error::custom(
                        "card research_gap block is missing required string `reason`",
                    )
                })?
                .to_owned();
            let challenge = raw
                .get("challenge")
                .map(|challenge| {
                    challenge.as_str().map(str::to_owned).ok_or_else(|| {
                        serde::de::Error::custom(
                            "card research_gap block challenge must be a string",
                        )
                    })
                })
                .transpose()?;
            return Ok(Self {
                state: CardedState::Gap { reason, challenge },
                source_arch: None,
            });
        }
        serde_json::from_value(raw)
            .map(|value| Self {
                state: CardedState::Present(value),
                source_arch: None,
            })
            .map_err(|e| serde::de::Error::custom(format!("block payload: {e}")))
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatenciesCycles {
    pub l1_hit: Measured,
    pub l2_hit: Measured,
    pub dram: Measured,
    pub shared: Measured,
    pub mshr_outstanding_misses: Measured,
    pub provenance: String,
}

/// Convert a JSON number to a positive integer without routing through f64.
///
/// `serde_json::Number::as_f64()` rounds decimal tokens.  For example,
/// `4294967295.0000001` becomes `4294967295.0`, which used to make an
/// out-of-range resource value look valid.  The workspace enables
/// `arbitrary_precision`, so `Number::to_string()` retains the numeric token
/// and lets this helper check its decimal/exponent form exactly while still
/// accepting the integer-valued decimal form emitted by Python (`3.0`).
fn exact_positive_integer(number: &serde_json::Number, maximum: u64) -> Option<u64> {
    let token = number.to_string();
    let (mantissa, exponent) = match token.find(['e', 'E']) {
        Some(index) => {
            let exponent = token[index + 1..].parse::<i64>().ok()?;
            (&token[..index], exponent)
        }
        None => (token.as_str(), 0),
    };
    if mantissa.starts_with('-') {
        return None;
    }
    let (whole, fraction) = match mantissa.find('.') {
        Some(index) => (&mantissa[..index], &mantissa[index + 1..]),
        None => (mantissa, ""),
    };
    if whole.is_empty()
        || !whole.bytes().all(|byte| byte.is_ascii_digit())
        || !fraction.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }

    let mut digits = String::with_capacity(whole.len() + fraction.len());
    digits.push_str(whole);
    digits.push_str(fraction);
    let scale = exponent.checked_sub(i64::try_from(fraction.len()).ok()?)?;
    let significant = digits.trim_start_matches('0');
    if significant.is_empty() {
        return None;
    }

    let integer_digits = if scale >= 0 {
        let zeros = usize::try_from(scale).ok()?;
        let total = significant.len().checked_add(zeros)?;
        if total > 20 {
            return None;
        }
        let mut expanded = String::with_capacity(total);
        expanded.push_str(significant);
        expanded.extend(std::iter::repeat_n('0', zeros));
        expanded
    } else {
        let fractional_digits = usize::try_from(scale.checked_neg()?).ok()?;
        if fractional_digits >= digits.len()
            || !digits[digits.len() - fractional_digits..]
                .bytes()
                .all(|byte| byte == b'0')
        {
            return None;
        }
        digits[..digits.len() - fractional_digits]
            .trim_start_matches('0')
            .to_owned()
    };

    if integer_digits.is_empty() || integer_digits.len() > 20 {
        return None;
    }
    let value = integer_digits.bytes().try_fold(0_u64, |value, byte| {
        value.checked_mul(10)?.checked_add(u64::from(byte - b'0'))
    })?;
    (value > 0 && value <= maximum).then_some(value)
}

fn deserialize_positive_u32<'de, D>(deserializer: D) -> Result<u32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let number = serde_json::Number::deserialize(deserializer)?;
    if let Some(value) = exact_positive_integer(&number, u64::from(u32::MAX)) {
        return Ok(value as u32);
    }
    Err(serde::de::Error::custom(
        "resource value must be a positive integer representable as u32",
    ))
}

const MAX_EXACT_JSON_INTEGER_U64: u64 = (1_u64 << 53) - 1;

fn deserialize_positive_u64<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let number = serde_json::Number::deserialize(deserializer)?;
    if let Some(value) = exact_positive_integer(&number, MAX_EXACT_JSON_INTEGER_U64) {
        return Ok(value);
    }
    Err(serde::de::Error::custom(
        "resource value must be a positive exact cross-language JSON integer",
    ))
}

/// Structural issue resources. Production cards must supply independently
/// admitted architecture-local values; the executor binds only supported
/// topology-compatible projections.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SmspResources {
    #[serde(deserialize_with = "deserialize_positive_u32")]
    pub count: u32,
    #[serde(deserialize_with = "deserialize_positive_u32")]
    pub issue_ports: u32,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OccupancyResources {
    #[serde(deserialize_with = "deserialize_positive_u32")]
    pub max_resident_warps_per_sm: u32,
    #[serde(deserialize_with = "deserialize_positive_u32")]
    pub max_blocks: u32,
}

/// Runtime launch/device limits observed from CUDA attributes. This block is
/// intentionally separate from `occupancy`/issue hypotheses and from L1
/// geometry: a card may know the launch envelope while still refusing to
/// invent an SM120 scheduler or cache model.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RuntimeLimits {
    #[serde(deserialize_with = "deserialize_positive_u32")]
    pub max_resident_warps_per_sm: u32,
    #[serde(deserialize_with = "deserialize_positive_u32")]
    pub max_blocks_per_sm: u32,
    #[serde(deserialize_with = "deserialize_positive_u32")]
    pub max_threads_per_sm: u32,
    #[serde(deserialize_with = "deserialize_positive_u32")]
    pub max_threads_per_block: u32,
    /// PTX special-register identifier ranges, not occupancy or topology.
    #[serde(deserialize_with = "deserialize_positive_u32")]
    pub sm_id_upper_bound: u32,
    #[serde(deserialize_with = "deserialize_positive_u32")]
    pub warp_id_upper_bound: u32,
    #[serde(deserialize_with = "deserialize_positive_u64")]
    pub max_shared_memory_per_sm_bytes: u64,
    #[serde(deserialize_with = "deserialize_positive_u64")]
    pub max_shared_memory_per_block_bytes: u64,
    #[serde(deserialize_with = "deserialize_positive_u64")]
    pub max_shared_memory_per_block_optin_bytes: u64,
    pub shared_memory_carveout_kib: Vec<u32>,
    pub provenance: String,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DepBarrierResources {
    #[serde(deserialize_with = "deserialize_positive_u32")]
    pub count: u32,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BarrierLatencyResources {
    #[serde(deserialize_with = "deserialize_positive_u64")]
    pub read: u64,
    #[serde(deserialize_with = "deserialize_positive_u64")]
    pub non_memory_write: u64,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DualIssueResources {
    pub supported: BoolOrResearchGap,
}

/// Fully resolved card projection. No gap arm or default survives this boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequiredIssueResources {
    pub smsp: SmspResources,
    pub occupancy: OccupancyResources,
    pub dep_barriers: DepBarrierResources,
    pub barrier_latencies: BarrierLatencyResources,
    pub dual_issue_supported: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct L1Geometry {
    pub l1tex_bytes: u64,
    pub ways: u32,
    pub default_l1_bytes: u64,
    pub max_shared_bytes: u64,
    pub sets: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct L2Geometry {
    pub capacity_bytes: u64,
    pub sector_bytes: u32,
    pub num_slices: u32,
    pub line_bytes: u32,
    /// `null` on partial cards (an exploratory SM120 card may ship capacity/slices with the
    /// write policy still unmeasured) — consumers requiring the policy must
    /// treat `None` as a typed gap, never assume write-allocate.
    pub policy: Option<String>,
    pub provenance: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NvmlReferenceOperatingPoint {
    pub sm_clock_ref_mhz: f64,
    pub max_sm_clock_mhz: f64,
    #[serde(default)]
    pub short_burst_sm_clock_ref_mhz: Option<f64>,
}

/// Aggregate DRAM read-bandwidth covariate at the LOCKED operating
/// point (vendor bus-width × data-rate spec + the protocol's -lgc/-lmc
/// clock locks — the same locks the `latencies_cycles` table was
/// measured under). Vendor-documented simulator INPUT, never an NCU
/// value; `peak_bytes_per_second` is a documented-default service-rate
/// hypothesis until a locked-protocol bandwidth microbench supersedes
/// it.
#[derive(Debug, Clone, Deserialize)]
pub struct DramBandwidth {
    pub peak_bytes_per_second: u64,
    pub locked_sm_clock_mhz: f64,
    pub locked_mem_clock_mhz: f64,
    pub provenance: String,
}

impl DramBandwidth {
    /// The locked SM clock in Hz as an exact integer (the fixed-point
    /// DRAM server divides by it). VALIDATED: a non-finite, non-positive
    /// or non-whole-MHz value is a typed error — a float→int saturating
    /// cast of a bad card value would otherwise produce a 0 Hz clock and
    /// a silently inert bandwidth server (gate finding, 2026-07-07).
    pub fn locked_sm_clock_hz(&self) -> Result<u64, CalibError> {
        let mhz = self.locked_sm_clock_mhz;
        if !mhz.is_finite() || mhz <= 0.0 || mhz.fract() != 0.0 {
            return Err(CalibError::InvalidCardValue {
                field: "dram_bandwidth.locked_sm_clock_mhz",
                value: mhz.to_string(),
                requirement: "positive finite whole MHz",
            });
        }
        // Reject the exclusive-u64 boundary before the float→integer cast.
        // `u64::MAX as f64` rounds to 2^64, so a `>` comparison would admit
        // that boundary and the cast would silently saturate.
        if mhz >= u64::MAX as f64 {
            return Err(CalibError::InvalidCardValue {
                field: "dram_bandwidth.locked_sm_clock_mhz",
                value: mhz.to_string(),
                requirement: "whole MHz whose Hz conversion fits in u64",
            });
        }
        let mhz_u64 = mhz as u64;
        let hz = mhz_u64
            .checked_mul(1_000_000)
            .ok_or_else(|| CalibError::InvalidCardValue {
                field: "dram_bandwidth.locked_sm_clock_mhz",
                value: mhz.to_string(),
                requirement: "whole MHz whose Hz conversion fits in u64",
            })?;
        Ok(hz)
    }

    /// The carded peak rate, validated non-zero (a 0 B/s rate must be a
    /// typed configuration error, never a clamped 1 B/s server).
    pub fn validated_peak_bytes_per_second(&self) -> Result<u64, CalibError> {
        if self.peak_bytes_per_second == 0 {
            return Err(CalibError::InvalidCardValue {
                field: "dram_bandwidth.peak_bytes_per_second",
                value: "0".to_owned(),
                requirement: "positive bytes/second",
            });
        }
        Ok(self.peak_bytes_per_second)
    }
}

/// Typed bundle of the latencies every current consumer requires (MSHR is
/// deliberately NOT here: it is still a research gap on sm_86 and consumers
/// must confront that via [`ArchCard::require_latency`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RequiredLatencies {
    pub l1_hit: f64,
    pub l2_hit: f64,
    pub dram: f64,
    pub shared: f64,
}

/// Raw JSON subset used only at the checked [`ArchCard::load`] boundary.
/// Keeping this type private means callers cannot deserialize an unvalidated
/// architecture card and then consume scalar evidence through public accessors.
#[derive(Debug, Deserialize)]
struct RawArchCard {
    arch_id: String,
    sku: String,
    compute_capability: (u32, u32),
    #[serde(default)]
    build_arch_family: Vec<String>,
    sm_count: u32,
    l1_geometry: Carded<L1Geometry>,
    l2_geometry: Carded<L2Geometry>,
    nvml_reference_operating_point: Carded<NvmlReferenceOperatingPoint>,
    #[serde(default = "Carded::absent_gap")]
    dram_bandwidth: Carded<DramBandwidth>,
    #[serde(default = "Carded::absent_gap")]
    smsp: Carded<SmspResources>,
    #[serde(default = "Carded::absent_gap")]
    occupancy: Carded<OccupancyResources>,
    #[serde(default = "Carded::absent_gap")]
    runtime_limits: Carded<RuntimeLimits>,
    #[serde(default = "Carded::absent_gap")]
    dep_barriers: Carded<DepBarrierResources>,
    #[serde(default = "Carded::absent_gap")]
    barrier_latencies: Carded<BarrierLatencyResources>,
    #[serde(default = "Carded::absent_gap")]
    dual_issue: Carded<DualIssueResources>,
    #[serde(default)]
    latencies_cycles: Option<Carded<LatenciesCycles>>,
    #[serde(default)]
    bare_ldg_e_default: Option<String>,
}

/// The validated subset of an ArchProfile card consumed by the simulator.
/// Fields are private and the type is not deserializable: callers can obtain a
/// card only through [`ArchCard::load`] or [`ArchCard::parse`], so post-load identity relabeling and
/// direct-deserialization access cannot detach scalar evidence from its source.
/// Unknown JSON fields are ignored at the private raw boundary because cards
/// also carry label-side bookkeeping that must never leak into simulator inputs.
///
/// ```compile_fail
/// let _: sim_arch::card::ArchCard = serde_json::from_str("{}").unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct ArchCard {
    loaded_arch: ArchId,
    loaded_sku: Sku,
    arch_id: String,
    sku: String,
    compute_capability: (u32, u32),
    build_arch_family: Vec<String>,
    sm_count: u32,
    l1_geometry: Carded<L1Geometry>,
    l2_geometry: Carded<L2Geometry>,
    nvml_reference_operating_point: Carded<NvmlReferenceOperatingPoint>,
    dram_bandwidth: Carded<DramBandwidth>,
    smsp: Carded<SmspResources>,
    occupancy: Carded<OccupancyResources>,
    runtime_limits: Carded<RuntimeLimits>,
    dep_barriers: Carded<DepBarrierResources>,
    barrier_latencies: Carded<BarrierLatencyResources>,
    dual_issue: Carded<DualIssueResources>,
    latencies_cycles: Option<Carded<LatenciesCycles>>,
    bare_ldg_e_default: Option<String>,
}

impl ArchCard {
    pub fn load(path: &Path) -> Result<Self, CalibError> {
        let raw = read_bounded_card(path, MAX_ARCH_CARD_BYTES)?;
        Self::parse(path, &raw)
    }

    /// Parse already-loaded card bytes through the same validation path as
    /// [`ArchCard::load`].
    pub fn parse(path: &Path, raw: &[u8]) -> Result<Self, CalibError> {
        let raw: RawArchCard = strict_json_from_slice(raw).map_err(|source| CalibError::Parse {
            path: path.display().to_string(),
            source,
        })?;
        let loaded_arch = ArchId::parse_deploy(&raw.arch_id)?;
        let expected_compute_capability = loaded_arch.compute_capability();
        if raw.compute_capability != expected_compute_capability {
            return Err(CalibError::ComputeCapabilityMismatch {
                arch_id: raw.arch_id,
                got: raw.compute_capability,
                expected: expected_compute_capability,
            });
        }
        let loaded_sku = Sku::parse(&raw.sku)?;
        if loaded_sku.arch() != loaded_arch {
            return Err(CalibError::InvalidCardValue {
                field: "sku",
                value: raw.sku,
                requirement: "SKU must belong to card arch_id",
            });
        }
        if raw.sm_count == 0 {
            return Err(CalibError::InvalidCardValue {
                field: "sm_count",
                value: raw.sm_count.to_string(),
                requirement: "positive integer representable as u32",
            });
        }
        let mut card = Self {
            loaded_arch,
            loaded_sku,
            arch_id: raw.arch_id,
            sku: raw.sku,
            compute_capability: raw.compute_capability,
            build_arch_family: raw.build_arch_family,
            sm_count: raw.sm_count,
            l1_geometry: raw.l1_geometry,
            l2_geometry: raw.l2_geometry,
            nvml_reference_operating_point: raw.nvml_reference_operating_point,
            dram_bandwidth: raw.dram_bandwidth,
            smsp: raw.smsp,
            occupancy: raw.occupancy,
            runtime_limits: raw.runtime_limits,
            dep_barriers: raw.dep_barriers,
            barrier_latencies: raw.barrier_latencies,
            dual_issue: raw.dual_issue,
            latencies_cycles: raw.latencies_cycles,
            bare_ldg_e_default: raw.bare_ldg_e_default,
        };
        let arch = card.arch_id()?;
        card.sku_id()?;
        card.checked_build_arch_family_for_arch(arch)?;
        card.checked_bare_ldg_e_declaration_for_arch(arch)?;
        card.l1_geometry = card.l1_geometry.bind_arch(arch);
        card.l2_geometry = card.l2_geometry.bind_arch(arch);
        card.nvml_reference_operating_point = card.nvml_reference_operating_point.bind_arch(arch);
        card.dram_bandwidth = card.dram_bandwidth.bind_arch(arch);
        card.smsp = card.smsp.bind_arch(arch);
        card.occupancy = card.occupancy.bind_arch(arch);
        card.runtime_limits = card.runtime_limits.bind_arch(arch);
        card.dep_barriers = card.dep_barriers.bind_arch(arch);
        card.barrier_latencies = card.barrier_latencies.bind_arch(arch);
        card.dual_issue = card.dual_issue.bind_arch(arch);
        card.latencies_cycles = card
            .latencies_cycles
            .map(|latencies| latencies.bind_arch(arch));
        Ok(card)
    }

    /// Resolve the card's canonical deployment identity through the shared
    /// architecture vocabulary.  Consumers should use this before selecting
    /// architecture-local hypotheses.
    pub fn arch_id(&self) -> Result<ArchId, CalibError> {
        let arch = ArchId::parse_deploy(&self.arch_id)?;
        let expected = arch.compute_capability();
        if self.compute_capability != expected {
            return Err(CalibError::ComputeCapabilityMismatch {
                arch_id: self.arch_id.clone(),
                got: self.compute_capability,
                expected,
            });
        }
        if arch != self.loaded_arch {
            return Err(CalibError::CardArchitectureRelabeled {
                loaded: self.loaded_arch,
                current: arch,
            });
        }
        Ok(arch)
    }

    /// Resolve the card SKU and verify that it belongs to this card's deploy
    /// architecture.  A SKU is never inferred from a different card.
    pub fn sku_id(&self) -> Result<Sku, CalibError> {
        let arch = self.arch_id()?;
        let sku = Sku::parse(&self.sku)?;
        if sku.arch() != arch {
            return Err(CalibError::InvalidCardValue {
                field: "sku",
                value: self.sku.clone(),
                requirement: "SKU must belong to card arch_id",
            });
        }
        if sku != self.loaded_sku {
            return Err(CalibError::CardSkuRelabeled {
                loaded: self.loaded_sku,
                current: sku,
            });
        }
        Ok(sku)
    }

    /// Return the validated card compute capability.
    pub fn compute_capability(&self) -> Result<(u32, u32), CalibError> {
        self.arch_id()?;
        Ok(self.compute_capability)
    }

    /// Return the card-local SM count only after validating deployment and SKU
    /// identity. The private field cannot be detached from the loaded card.
    pub fn sm_count(&self) -> Result<u32, CalibError> {
        self.sku_id()?;
        Ok(self.sm_count)
    }

    fn checked_build_arch_family_for_arch(
        &self,
        arch: ArchId,
    ) -> Result<Vec<BuildArch>, CalibError> {
        if self.build_arch_family.is_empty() {
            return Err(CalibError::BuildArchFamilyEmpty { arch });
        }
        let mut seen = BTreeSet::new();
        let mut family = Vec::with_capacity(self.build_arch_family.len());
        for token in &self.build_arch_family {
            let build = BuildArch::parse(token).map_err(|source| {
                CalibError::BuildArchFamilyTokenInvalid {
                    token: token.clone(),
                    source,
                }
            })?;
            if build.deploy_arch() != arch {
                return Err(CalibError::BuildArchFamilyArchitectureMismatch {
                    arch,
                    token: token.clone(),
                });
            }
            if !seen.insert(build) {
                return Err(CalibError::BuildArchFamilyDuplicate {
                    token: token.clone(),
                });
            }
            family.push(build);
        }
        Ok(family)
    }

    /// Parse and validate the card-local compile-target allowlist.
    pub fn checked_build_arch_family(&self) -> Result<Vec<BuildArch>, CalibError> {
        let arch = self.sku_id()?.arch();
        self.checked_build_arch_family_for_arch(arch)
    }

    fn checked_bare_ldg_e_declaration_for_arch(
        &self,
        arch: ArchId,
    ) -> Result<BareLdgECardDeclaration, CalibError> {
        let token = self
            .bare_ldg_e_default
            .as_deref()
            .ok_or(CalibError::BareLdgDeclarationMissing { arch })?;
        let declaration = BareLdgECardDeclaration::parse(token)?;
        let capability = arch.bare_ldg_e_default();
        let status = capability.status();
        match (declaration, status) {
            (BareLdgECardDeclaration::CacheAtAllLevels, CapabilityStatus::Available) => {
                let BareLdgEDefault::CacheAtAllLevels = capability.require("bare_ldg_e_default")?;
                Ok(declaration)
            }
            (
                BareLdgECardDeclaration::TypedUnavailable,
                CapabilityStatus::ResearchGap { .. } | CapabilityStatus::Unsupported { .. },
            ) => Ok(declaration),
            _ => Err(CalibError::BareLdgCapabilityMismatch {
                arch,
                declaration,
                status,
            }),
        }
    }

    /// Parse the closed card declaration and prove that it agrees with the
    /// architecture-owned runtime capability without manufacturing a value for
    /// a research gap.
    pub fn bare_ldg_e_declaration(&self) -> Result<BareLdgECardDeclaration, CalibError> {
        let arch = self.sku_id()?.arch();
        self.checked_bare_ldg_e_declaration_for_arch(arch)
    }

    /// Bind the card's deployment identity, SKU, and compile-time build token
    /// only after the requested build belongs to the card-local allowlist and
    /// the bare-LDG declaration agrees with the runtime capability.
    pub fn identity(&self, build: BuildArch) -> Result<ArchIdentity, CalibError> {
        let deploy = self.arch_id()?;
        let sku = self.sku_id()?;
        let family = self.checked_build_arch_family_for_arch(deploy)?;
        if !family.contains(&build) {
            return Err(CalibError::BuildArchNotInCardFamily {
                arch: deploy,
                build: build.token().to_owned(),
            });
        }
        self.checked_bare_ldg_e_declaration_for_arch(deploy)?;
        Ok(ArchIdentity::new(deploy, sku, build)?)
    }

    /// Require a card block using the card's own validated architecture.  The
    /// architecture is intentionally not caller-supplied: a present sm_86
    /// block must never be retrievable by labeling it sm_120.
    pub fn require_l1_geometry(&self) -> Result<&L1Geometry, CalibError> {
        let arch = self.sku_id()?.arch();
        let geometry = self.l1_geometry.require_for_arch(arch, "l1_geometry")?;
        Self::require_positive_u64("l1_geometry.l1tex_bytes", geometry.l1tex_bytes)?;
        Self::require_positive_u32("l1_geometry.ways", geometry.ways)?;
        Self::require_positive_u64("l1_geometry.default_l1_bytes", geometry.default_l1_bytes)?;
        Self::require_positive_u64("l1_geometry.max_shared_bytes", geometry.max_shared_bytes)?;
        Self::require_positive_u32("l1_geometry.sets", geometry.sets)?;
        if geometry.default_l1_bytes > geometry.l1tex_bytes {
            return Err(CalibError::InvalidCardValue {
                field: "l1_geometry.default_l1_bytes",
                value: geometry.default_l1_bytes.to_string(),
                requirement: "must not exceed l1_geometry.l1tex_bytes",
            });
        }
        Ok(geometry)
    }

    pub fn require_l2_geometry(&self) -> Result<&L2Geometry, CalibError> {
        let arch = self.sku_id()?.arch();
        let geometry = self.l2_geometry.require_for_arch(arch, "l2_geometry")?;
        Self::require_nonempty_text("l2_geometry.provenance", &geometry.provenance)?;
        Self::require_positive_u64("l2_geometry.capacity_bytes", geometry.capacity_bytes)?;
        Self::require_positive_u32("l2_geometry.sector_bytes", geometry.sector_bytes)?;
        Self::require_positive_u32("l2_geometry.num_slices", geometry.num_slices)?;
        Self::require_positive_u32("l2_geometry.line_bytes", geometry.line_bytes)?;
        if geometry.line_bytes % geometry.sector_bytes != 0 {
            return Err(CalibError::InvalidCardValue {
                field: "l2_geometry.line_bytes",
                value: geometry.line_bytes.to_string(),
                requirement: "must be an integer multiple of l2_geometry.sector_bytes",
            });
        }
        if geometry.capacity_bytes < u64::from(geometry.line_bytes) {
            return Err(CalibError::InvalidCardValue {
                field: "l2_geometry.capacity_bytes",
                value: geometry.capacity_bytes.to_string(),
                requirement: "must be at least one L2 line",
            });
        }
        Ok(geometry)
    }

    pub fn require_nvml_reference_operating_point(
        &self,
    ) -> Result<&NvmlReferenceOperatingPoint, CalibError> {
        let arch = self.sku_id()?.arch();
        let operating_point = self
            .nvml_reference_operating_point
            .require_for_arch(arch, "nvml_reference_operating_point")?;
        if !operating_point.sm_clock_ref_mhz.is_finite()
            || operating_point.sm_clock_ref_mhz <= 0.0
            || !operating_point.max_sm_clock_mhz.is_finite()
            || operating_point.max_sm_clock_mhz <= 0.0
            || operating_point.max_sm_clock_mhz < operating_point.sm_clock_ref_mhz
            || operating_point
                .short_burst_sm_clock_ref_mhz
                .is_some_and(|value| {
                    !value.is_finite() || value <= 0.0 || value > operating_point.max_sm_clock_mhz
                })
        {
            return Err(CalibError::InvalidCardValue {
                field: "nvml_reference_operating_point",
                value: "non-finite, non-positive, or out-of-range clock".to_owned(),
                requirement: "positive finite clocks with short-burst <= max",
            });
        }
        Ok(operating_point)
    }

    pub fn require_dram_bandwidth(&self) -> Result<&DramBandwidth, CalibError> {
        let arch = self.sku_id()?.arch();
        let bandwidth = self
            .dram_bandwidth
            .require_for_arch(arch, "dram_bandwidth")?;
        Self::require_nonempty_text("dram_bandwidth.provenance", &bandwidth.provenance)?;
        Ok(bandwidth)
    }

    fn require_nonempty_text(field: &'static str, value: &str) -> Result<(), CalibError> {
        if value.trim().is_empty() {
            return Err(CalibError::InvalidCardValue {
                field,
                value: value.to_owned(),
                requirement: "non-empty provenance text",
            });
        }
        Ok(())
    }

    fn require_positive_u32(field: &'static str, value: u32) -> Result<(), CalibError> {
        if value == 0 {
            return Err(CalibError::InvalidCardValue {
                field,
                value: value.to_string(),
                requirement: "positive integer representable as u32",
            });
        }
        Ok(())
    }

    fn require_positive_u64(field: &'static str, value: u64) -> Result<(), CalibError> {
        if !(1..=MAX_EXACT_JSON_INTEGER_U64).contains(&value) {
            return Err(CalibError::InvalidCardValue {
                field,
                value: value.to_string(),
                requirement: "positive exact cross-language JSON integer",
            });
        }
        Ok(())
    }

    pub fn require_smsp_resources(&self) -> Result<&SmspResources, CalibError> {
        let arch = self.sku_id()?.arch();
        let resources = self.smsp.require_for_arch(arch, "smsp")?;
        Self::require_positive_u32("smsp.count", resources.count)?;
        Self::require_positive_u32("smsp.issue_ports", resources.issue_ports)?;
        Ok(resources)
    }

    pub fn require_occupancy_resources(&self) -> Result<&OccupancyResources, CalibError> {
        let arch = self.sku_id()?.arch();
        let resources = self.occupancy.require_for_arch(arch, "occupancy")?;
        Self::require_positive_u32(
            "occupancy.max_resident_warps_per_sm",
            resources.max_resident_warps_per_sm,
        )?;
        Self::require_positive_u32("occupancy.max_blocks", resources.max_blocks)?;
        // Validate a present runtime projection (including duplicate-limit
        // agreement) without requiring runtime data on legacy SM86 cards.
        match self.runtime_limits.require_for_arch(arch, "runtime_limits") {
            Ok(_) => {
                self.require_runtime_limits()?;
            }
            Err(CalibError::BlockResearchGap { .. }) => {}
            Err(error) => return Err(error),
        }
        Ok(resources)
    }

    /// Runtime launch/device limits are an observed contract, not an issue or
    /// cache geometry hypothesis. Consumers that need this envelope must
    /// request it explicitly; absence remains a typed gap.
    pub fn require_runtime_limits(&self) -> Result<&RuntimeLimits, CalibError> {
        let arch = self.sku_id()?.arch();
        let limits = self
            .runtime_limits
            .require_for_arch(arch, "runtime_limits")?;
        Self::require_positive_u32(
            "runtime_limits.max_resident_warps_per_sm",
            limits.max_resident_warps_per_sm,
        )?;
        Self::require_positive_u32("runtime_limits.max_blocks_per_sm", limits.max_blocks_per_sm)?;
        Self::require_positive_u32(
            "runtime_limits.max_threads_per_sm",
            limits.max_threads_per_sm,
        )?;
        Self::require_positive_u64(
            "runtime_limits.max_shared_memory_per_sm_bytes",
            limits.max_shared_memory_per_sm_bytes,
        )?;
        Self::require_positive_u64(
            "runtime_limits.max_shared_memory_per_block_bytes",
            limits.max_shared_memory_per_block_bytes,
        )?;
        Self::require_positive_u64(
            "runtime_limits.max_shared_memory_per_block_optin_bytes",
            limits.max_shared_memory_per_block_optin_bytes,
        )?;
        Self::require_positive_u32(
            "runtime_limits.max_threads_per_block",
            limits.max_threads_per_block,
        )?;
        Self::require_positive_u32("runtime_limits.sm_id_upper_bound", limits.sm_id_upper_bound)?;
        Self::require_positive_u32(
            "runtime_limits.warp_id_upper_bound",
            limits.warp_id_upper_bound,
        )?;
        if limits.max_threads_per_block > limits.max_threads_per_sm {
            return Err(CalibError::InvalidCardValue {
                field: "runtime_limits.max_threads_per_block",
                value: limits.max_threads_per_block.to_string(),
                requirement: "per-block thread limit <= per-SM thread limit",
            });
        }
        if limits.sm_id_upper_bound < self.sm_count()?
            || limits.warp_id_upper_bound < limits.max_resident_warps_per_sm
        {
            return Err(CalibError::InvalidCardValue {
                field: "runtime_limits.identifier_ranges",
                value: format!(
                    "sm={}, warp={}",
                    limits.sm_id_upper_bound, limits.warp_id_upper_bound
                ),
                requirement: "identifier ranges contain the maximum simultaneous residents; equality is not assumed",
            });
        }
        let warp_threads = u64::from(limits.max_resident_warps_per_sm)
            .checked_mul(32)
            .ok_or_else(|| CalibError::InvalidCardValue {
                field: "runtime_limits.max_resident_warps_per_sm",
                value: limits.max_resident_warps_per_sm.to_string(),
                requirement: "warps*32 must fit in max_threads_per_sm",
            })?;
        if u64::from(limits.max_threads_per_sm) < warp_threads {
            return Err(CalibError::InvalidCardValue {
                field: "runtime_limits.max_threads_per_sm",
                value: limits.max_threads_per_sm.to_string(),
                requirement: "at least max_resident_warps_per_sm*32 threads",
            });
        }
        if limits.max_shared_memory_per_block_bytes > limits.max_shared_memory_per_block_optin_bytes
            || limits.max_shared_memory_per_block_optin_bytes
                > limits.max_shared_memory_per_sm_bytes
        {
            return Err(CalibError::InvalidCardValue {
                field: "runtime_limits.shared_memory",
                value: format!(
                    "{}/{}/{}",
                    limits.max_shared_memory_per_block_bytes,
                    limits.max_shared_memory_per_block_optin_bytes,
                    limits.max_shared_memory_per_sm_bytes
                ),
                requirement: "block <= opt-in block <= per-SM shared bytes",
            });
        }
        if limits.provenance.trim().is_empty() {
            return Err(CalibError::InvalidCardValue {
                field: "runtime_limits.provenance",
                value: limits.provenance.clone(),
                requirement: "non-empty observation context",
            });
        }
        if limits.shared_memory_carveout_kib.is_empty()
            || limits
                .shared_memory_carveout_kib
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || limits
                .shared_memory_carveout_kib
                .iter()
                .any(|kib| u64::from(*kib) * 1024 > limits.max_shared_memory_per_sm_bytes)
        {
            return Err(CalibError::InvalidCardValue {
                field: "runtime_limits.shared_memory_carveout_kib",
                value: format!("{:?}", limits.shared_memory_carveout_kib),
                requirement: "strictly increasing carveouts within per-SM shared bytes",
            });
        }
        // Keep the duplicated launch-envelope projection coherent.  The
        // runtime block is authoritative for placement, but direct metric
        // consumers still expose `occupancy`; allowing them to disagree would
        // make one run report two different SM120 ceilings.
        if let Ok(occupancy) = self.occupancy.require_for_arch(arch, "occupancy") {
            Self::require_positive_u32(
                "occupancy.max_resident_warps_per_sm",
                occupancy.max_resident_warps_per_sm,
            )?;
            Self::require_positive_u32("occupancy.max_blocks", occupancy.max_blocks)?;
            if occupancy.max_resident_warps_per_sm != limits.max_resident_warps_per_sm {
                return Err(CalibError::InvalidCardValue {
                    field: "runtime_limits.max_resident_warps_per_sm",
                    value: format!(
                        "{} != {}",
                        limits.max_resident_warps_per_sm, occupancy.max_resident_warps_per_sm
                    ),
                    requirement: "match occupancy.max_resident_warps_per_sm",
                });
            }
            if occupancy.max_blocks != limits.max_blocks_per_sm {
                return Err(CalibError::InvalidCardValue {
                    field: "runtime_limits.max_blocks_per_sm",
                    value: format!("{} != {}", limits.max_blocks_per_sm, occupancy.max_blocks),
                    requirement: "match occupancy.max_blocks",
                });
            }
        }
        Ok(limits)
    }

    pub fn require_dep_barrier_resources(&self) -> Result<&DepBarrierResources, CalibError> {
        let arch = self.sku_id()?.arch();
        let resources = self.dep_barriers.require_for_arch(arch, "dep_barriers")?;
        Self::require_positive_u32("dep_barriers.count", resources.count)?;
        Ok(resources)
    }

    pub fn require_barrier_latency_resources(
        &self,
    ) -> Result<&BarrierLatencyResources, CalibError> {
        let arch = self.sku_id()?.arch();
        let resources = self
            .barrier_latencies
            .require_for_arch(arch, "barrier_latencies")?;
        Self::require_positive_u64("barrier_latencies.read", resources.read)?;
        Self::require_positive_u64(
            "barrier_latencies.non_memory_write",
            resources.non_memory_write,
        )?;
        Ok(resources)
    }

    pub fn require_dual_issue_supported(&self) -> Result<bool, CalibError> {
        let arch = self.sku_id()?.arch();
        let resources = self.dual_issue.require_for_arch(arch, "dual_issue")?;
        match &resources.supported {
            BoolOrResearchGap::Value(supported) => Ok(*supported),
            BoolOrResearchGap::Gap { reason, challenge } => Err(CalibError::BlockResearchGap {
                arch_id: arch.to_string(),
                block: "dual_issue.supported",
                reason: reason.clone(),
                challenge: challenge.clone(),
            }),
        }
    }

    /// Resolve the complete issue-resource scaffold in one fail-closed step.
    /// This projection supplies no defaults and cannot borrow another card's
    /// values. The SM86 executor consumes a strict topology-compatible subset;
    /// divergent structural values are rejected at replay construction.
    pub fn required_issue_resources(&self) -> Result<RequiredIssueResources, CalibError> {
        Ok(RequiredIssueResources {
            smsp: *self.require_smsp_resources()?,
            occupancy: *self.require_occupancy_resources()?,
            dep_barriers: *self.require_dep_barrier_resources()?,
            barrier_latencies: *self.require_barrier_latency_resources()?,
            dual_issue_supported: self.require_dual_issue_supported()?,
        })
    }

    /// True when a card carries at least one scheduler-resource block. This is
    /// used by compatibility consumers to distinguish an old card with no
    /// resource schema from a partially migrated card that must fail closed.
    pub fn has_any_issue_resource_blocks(&self) -> bool {
        self.smsp.require("smsp").is_ok()
            || self.occupancy.require("occupancy").is_ok()
            || self.dep_barriers.require("dep_barriers").is_ok()
            || self.barrier_latencies.require("barrier_latencies").is_ok()
            || self.dual_issue.require("dual_issue").is_ok()
    }

    /// The latency table, or a typed error if the card predates Phase 1D.
    pub fn latencies(&self) -> Result<&LatenciesCycles, CalibError> {
        // Latency values are architecture-local inputs. They carry the same
        // load-time source binding as geometry blocks, so even an internal
        // coherent identity relabel cannot expose another architecture's table.
        let arch = self.sku_id()?.arch();
        self.latencies_cycles
            .as_ref()
            .ok_or_else(|| CalibError::LatenciesMissing {
                arch_id: self.arch_id.clone(),
            })?
            .require_for_arch(arch, "latencies_cycles")
    }

    /// A required latency in cycles. `research_gap` → typed error, never a
    /// default.
    pub fn require_latency(&self, kind: LatencyKind) -> Result<f64, CalibError> {
        let lat = self.latencies()?;
        Self::require_nonempty_text("latencies_cycles.provenance", &lat.provenance)?;
        let entry = match kind {
            LatencyKind::L1Hit => &lat.l1_hit,
            LatencyKind::L2Hit => &lat.l2_hit,
            LatencyKind::Dram => &lat.dram,
            LatencyKind::Shared => &lat.shared,
            LatencyKind::MshrOutstandingMisses => &lat.mshr_outstanding_misses,
        };
        match entry {
            Measured::Value(v) => {
                if v.unit.as_deref().is_some_and(|unit| {
                    !matches!(unit, "cycle" | "cycles" | "cycles_per_dependent_load")
                }) {
                    return Err(CalibError::InvalidCardValue {
                        field: kind.value_field(),
                        value: v.unit.clone().unwrap_or_default(),
                        requirement: "latency unit must describe cycles",
                    });
                }
                if !v.value.is_finite() || v.value <= 0.0 {
                    return Err(CalibError::InvalidCardValue {
                        field: kind.value_field(),
                        value: v.value.to_string(),
                        requirement: "positive finite scalar",
                    });
                }
                if let Some(dispersion) = v.dispersion
                    && (!dispersion.is_finite() || dispersion < 0.0)
                {
                    return Err(CalibError::InvalidCardValue {
                        field: kind.dispersion_field(),
                        value: dispersion.to_string(),
                        requirement: "non-negative finite scalar",
                    });
                }
                Ok(v.value)
            }
            Measured::Gap { reason, .. } => Err(CalibError::ResearchGap {
                arch_id: self.arch_id.clone(),
                kind,
                reason: reason.clone(),
            }),
        }
    }

    /// The four simulator-required latencies resolved at once, so a consumer
    /// can fail-close at construction instead of mid-simulation.
    pub fn required_latencies(&self) -> Result<RequiredLatencies, CalibError> {
        Ok(RequiredLatencies {
            l1_hit: self.require_latency(LatencyKind::L1Hit)?,
            l2_hit: self.require_latency(LatencyKind::L2Hit)?,
            dram: self.require_latency(LatencyKind::Dram)?,
            shared: self.require_latency(LatencyKind::Shared)?,
        })
    }
}

fn read_bounded_card(path: &Path, limit: u64) -> Result<Vec<u8>, CalibError> {
    let file = std::fs::File::open(path).map_err(|source| CalibError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let mut raw = Vec::new();
    file.take(limit.saturating_add(1))
        .read_to_end(&mut raw)
        .map_err(|source| CalibError::Io {
            path: path.display().to_string(),
            source,
        })?;
    let observed = u64::try_from(raw.len()).unwrap_or(u64::MAX);
    if observed > limit {
        return Err(CalibError::ResourceLimit {
            path: path.display().to_string(),
            observed,
            limit,
        });
    }
    Ok(raw)
}
