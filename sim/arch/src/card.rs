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
//! (the sm_120-laptop card ships `l1_geometry` itself as a research_gap
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

/// Structural issue resources. These types are schema scaffolding only until a
/// production card supplies independently admitted architecture-local values.
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
    /// `null` on partial cards (sm_120-laptop ships capacity/slices with the
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
                .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(CalibError::InvalidCardValue {
                field: "nvml_reference_operating_point",
                value: "non-finite, non-positive, or inverted clock range".to_owned(),
                requirement: "positive finite clocks with max >= reference",
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
        Ok(resources)
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
    /// values. It does not itself authorize execution of the dormant carded lane.
    pub fn required_issue_resources(&self) -> Result<RequiredIssueResources, CalibError> {
        Ok(RequiredIssueResources {
            smsp: *self.require_smsp_resources()?,
            occupancy: *self.require_occupancy_resources()?,
            dep_barriers: *self.require_dep_barrier_resources()?,
            barrier_latencies: *self.require_barrier_latency_resources()?,
            dual_issue_supported: self.require_dual_issue_supported()?,
        })
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

#[cfg(test)]
#[allow(clippy::unwrap_used)] // unwrap in tests is an assertion, not error handling
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::NamedTempFile;

    fn card_path(name: &str) -> PathBuf {
        PathBuf::from("../../arch_profiles/cards")
            .join(name)
            .canonicalize()
            .expect("card exists in-repo")
    }

    #[test]
    fn bounded_card_read_rejects_oversized_input() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("card.json");
        std::fs::write(&path, b"12345").unwrap();
        assert!(matches!(
            read_bounded_card(&path, 4),
            Err(CalibError::ResourceLimit {
                observed: 5,
                limit: 4,
                ..
            })
        ));
    }

    #[test]
    fn duplicate_json_members_are_rejected_before_card_decode() {
        let raw = std::fs::read(card_path("sm_86.json")).unwrap();
        let text = String::from_utf8(raw).unwrap();
        let duplicate = text.replacen(
            "\"arch_id\": \"sm_86\"",
            "\"arch_id\": \"sm_86\", \"arch_id\": \"sm_86\"",
            1,
        );
        let error = ArchCard::parse(Path::new("sm_86.json"), duplicate.as_bytes()).unwrap_err();
        assert!(matches!(error, CalibError::Parse { source, .. } if source
            .to_string()
            .contains("duplicate JSON member")));
    }

    fn load_mutated_card(
        name: &str,
        mutate: impl FnOnce(&mut serde_json::Value),
    ) -> Result<ArchCard, CalibError> {
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(card_path(name)).unwrap()).unwrap();
        mutate(&mut value);
        let file = NamedTempFile::new().unwrap();
        std::fs::write(file.path(), serde_json::to_vec(&value).unwrap()).unwrap();
        ArchCard::load(file.path())
    }

    fn load_inline_card(json: &str) -> Result<ArchCard, CalibError> {
        let file = NamedTempFile::new().unwrap();
        std::fs::write(file.path(), json).unwrap();
        ArchCard::load(file.path())
    }

    fn install_synthetic_issue_resources(card: &mut serde_json::Value) {
        // Deliberately non-authoritative sentinels: fixture-only schema witnesses.
        card["smsp"] = serde_json::json!({"count": 3, "issue_ports": 5});
        card["occupancy"] = serde_json::json!({
            "max_resident_warps_per_sm": 7,
            "max_blocks": 11,
        });
        card["dep_barriers"] = serde_json::json!({"count": 13});
        card["barrier_latencies"] = serde_json::json!({
            "read": 17,
            "non_memory_write": 19,
        });
        card["dual_issue"] = serde_json::json!({"supported": true});
    }

    fn measured_mut(card: &mut ArchCard, kind: LatencyKind) -> &mut MeasuredValue {
        let latencies = match &mut card
            .latencies_cycles
            .as_mut()
            .expect("latencies present")
            .state
        {
            CardedState::Present(latencies) => latencies,
            CardedState::Gap { .. } => panic!("expected latency table"),
        };
        let entry = match kind {
            LatencyKind::L1Hit => &mut latencies.l1_hit,
            LatencyKind::L2Hit => &mut latencies.l2_hit,
            LatencyKind::Dram => &mut latencies.dram,
            LatencyKind::Shared => &mut latencies.shared,
            LatencyKind::MshrOutstandingMisses => &mut latencies.mshr_outstanding_misses,
        };
        match entry {
            Measured::Value(value) => value,
            Measured::Gap { .. } => panic!("expected measured value"),
        }
    }

    #[test]
    fn card_identity_binds_arch_sku_build_and_compute_capability() {
        let sm86 = ArchCard::load(&card_path("sm_86.json")).expect("load sm_86 card");
        let identity = sm86
            .identity(BuildArch::Sm86)
            .expect("valid sm_86 identity");
        assert_eq!(identity.deploy(), ArchId::Sm86);
        assert_eq!(identity.sku(), Sku::Rtx3090Ti);

        let sm120 = ArchCard::load(&card_path("sm_120-laptop.json")).expect("load sm_120 card");
        let identity = sm120
            .identity(BuildArch::Sm120f)
            .expect("valid sm_120 build identity");
        assert_eq!(identity.deploy(), ArchId::Sm120);
        assert_eq!(identity.sku(), Sku::Rtx5060Laptop);

        assert!(matches!(
            load_mutated_card("sm_86.json", |card| {
                card["compute_capability"] = serde_json::json!([12, 0]);
            }),
            Err(CalibError::ComputeCapabilityMismatch { .. })
        ));
    }

    #[test]
    fn card_build_family_is_closed_and_identity_enforces_membership() {
        let sm120 = ArchCard::load(&card_path("sm_120-laptop.json")).unwrap();
        assert_eq!(
            sm120.checked_build_arch_family().unwrap(),
            vec![BuildArch::Sm120, BuildArch::Sm120a, BuildArch::Sm120f]
        );
        for build in [BuildArch::Sm120, BuildArch::Sm120a, BuildArch::Sm120f] {
            assert_eq!(sm120.identity(build).unwrap().build(), build);
        }

        let missing_requested = load_mutated_card("sm_120-laptop.json", |card| {
            card["build_arch_family"]
                .as_array_mut()
                .unwrap()
                .retain(|token| token != "sm_120f");
        })
        .unwrap();
        assert!(matches!(
            missing_requested.identity(BuildArch::Sm120f),
            Err(CalibError::BuildArchNotInCardFamily {
                arch: ArchId::Sm120,
                ..
            })
        ));

        assert!(matches!(
            load_mutated_card("sm_120-laptop.json", |card| {
                card["build_arch_family"] = serde_json::json!([]);
            }),
            Err(CalibError::BuildArchFamilyEmpty {
                arch: ArchId::Sm120
            })
        ));
        assert!(matches!(
            load_mutated_card("sm_120-laptop.json", |card| {
                card["build_arch_family"] =
                    serde_json::json!(["sm_120", "sm_120a", "sm_120f", "sm_120"]);
            }),
            Err(CalibError::BuildArchFamilyDuplicate { token }) if token == "sm_120"
        ));
        assert!(matches!(
            load_mutated_card("sm_120-laptop.json", |card| {
                card["build_arch_family"] = serde_json::json!(["sm_120", "sm_86"]);
            }),
            Err(CalibError::BuildArchFamilyArchitectureMismatch {
                arch: ArchId::Sm120,
                token,
            }) if token == "sm_86"
        ));
        assert!(matches!(
            load_mutated_card("sm_120-laptop.json", |card| {
                card["build_arch_family"] = serde_json::json!(["sm_120", "sm_120z"]);
            }),
            Err(CalibError::BuildArchFamilyTokenInvalid { token, .. })
                if token == "sm_120z"
        ));
    }

    #[test]
    fn bare_ldg_card_declaration_must_agree_with_runtime_capability() {
        let sm86 = ArchCard::load(&card_path("sm_86.json")).unwrap();
        assert_eq!(
            sm86.bare_ldg_e_declaration().unwrap(),
            BareLdgECardDeclaration::CacheAtAllLevels
        );
        let sm120 = ArchCard::load(&card_path("sm_120-laptop.json")).unwrap();
        assert_eq!(
            sm120.bare_ldg_e_declaration().unwrap(),
            BareLdgECardDeclaration::TypedUnavailable
        );

        assert!(matches!(
            load_mutated_card("sm_86.json", |card| {
                card.as_object_mut().unwrap().remove("bare_ldg_e_default");
            }),
            Err(CalibError::BareLdgDeclarationMissing { arch: ArchId::Sm86 })
        ));
        assert!(matches!(
            load_mutated_card("sm_86.json", |card| {
                card["bare_ldg_e_default"] = serde_json::json!("bypass");
            }),
            Err(CalibError::BareLdgDeclaration(_))
        ));
        assert!(matches!(
            load_mutated_card("sm_86.json", |card| {
                card["bare_ldg_e_default"] = serde_json::json!("unknown_typed_unavailable");
            }),
            Err(CalibError::BareLdgCapabilityMismatch {
                arch: ArchId::Sm86,
                declaration: BareLdgECardDeclaration::TypedUnavailable,
                status: CapabilityStatus::Available,
            })
        ));
        assert!(matches!(
            load_mutated_card("sm_120-laptop.json", |card| {
                card["bare_ldg_e_default"] = serde_json::json!("ca");
            }),
            Err(CalibError::BareLdgCapabilityMismatch {
                arch: ArchId::Sm120,
                declaration: BareLdgECardDeclaration::CacheAtAllLevels,
                status: CapabilityStatus::ResearchGap { .. },
            })
        ));
    }

    #[test]
    fn coherent_post_load_relabel_cannot_launder_latency_evidence() {
        let mut card = ArchCard::load(&card_path("sm_86.json")).unwrap();
        card.arch_id = "sm_120".to_owned();
        card.sku = "rtx_5060_laptop".to_owned();
        card.compute_capability = (12, 0);
        card.build_arch_family = vec!["sm_120".to_owned()];
        card.bare_ldg_e_default = Some("unknown_typed_unavailable".to_owned());

        assert!(matches!(
            card.identity(BuildArch::Sm120),
            Err(CalibError::CardArchitectureRelabeled {
                loaded: ArchId::Sm86,
                current: ArchId::Sm120,
            })
        ));
        assert!(matches!(
            card.required_latencies(),
            Err(CalibError::CardArchitectureRelabeled {
                loaded: ArchId::Sm86,
                current: ArchId::Sm120,
            })
        ));
    }

    /// Python lockstep test (test_arch_profile_lockstep.py) pins against the
    /// pchase v2.1 evidence.
    #[test]
    fn sm86_card_lockstep() {
        let card = ArchCard::load(&card_path("sm_86.json")).expect("load sm_86 card");
        assert_eq!(card.arch_id().unwrap(), ArchId::Sm86);
        assert_eq!(card.compute_capability().unwrap(), (8, 6));
        assert_eq!(card.sm_count().unwrap(), 84);
        let l2 = card.require_l2_geometry().unwrap();
        assert_eq!(l2.capacity_bytes, 6_291_456);
        assert_eq!(l2.num_slices, 6);
        assert_eq!(l2.sector_bytes, 32);
        assert_eq!(l2.line_bytes, 128);
        let l1 = card.require_l1_geometry().unwrap();
        assert_eq!(l1.l1tex_bytes, 131_072);
        assert_eq!(l1.default_l1_bytes, 32_768);
        assert_eq!(
            card.bare_ldg_e_declaration().unwrap(),
            BareLdgECardDeclaration::CacheAtAllLevels
        );

        // pchase v2.1 values (protocol: -lgc 1350 + -lmc 10501).
        let l = card.required_latencies().unwrap();
        assert_eq!(l.l1_hit, 33.0016);
        assert_eq!(l.l2_hit, 243.6152);
        assert_eq!(l.dram, 422.6486);
        assert_eq!(l.shared, 23.0016);

        // DRAM bandwidth covariate at the same locked point (vendor
        // 21 Gbps × 384-bit; the -lmc 10501 lock IS the full-rate point).
        let bw = card.require_dram_bandwidth().unwrap();
        assert_eq!(bw.peak_bytes_per_second, 1_008_000_000_000);
        assert_eq!(bw.locked_sm_clock_mhz, 1350.0);
        assert_eq!(bw.locked_mem_clock_mhz, 10501.0);
        assert_eq!(bw.locked_sm_clock_hz().unwrap(), 1_350_000_000);
        assert_eq!(
            bw.validated_peak_bytes_per_second().unwrap(),
            1_008_000_000_000
        );
        // LOCKSTEP PIN (gate finding): the card states the locked SM
        // clock in three places (NVML block, latencies provenance prose,
        // dram_bandwidth block). Pin the two NUMERIC copies equal so a
        // protocol re-lock that updates one but not the other fails here
        // instead of silently splitting the latency arm and the byte
        // server into different clock domains.
        let nvml = card.require_nvml_reference_operating_point().unwrap();
        assert_eq!(
            bw.locked_sm_clock_mhz, nvml.sm_clock_ref_mhz,
            "dram_bandwidth.locked_sm_clock_mhz must equal the NVML reference SM clock"
        );
        // Bad-value validation is typed, never a clamp/saturating cast.
        let bad = DramBandwidth {
            peak_bytes_per_second: 0,
            locked_sm_clock_mhz: -1.0,
            locked_mem_clock_mhz: 10501.0,
            provenance: String::new(),
        };
        assert!(matches!(
            bad.locked_sm_clock_hz(),
            Err(CalibError::InvalidCardValue { .. })
        ));
        assert!(matches!(
            bad.validated_peak_bytes_per_second(),
            Err(CalibError::InvalidCardValue { .. })
        ));
        let overflow = DramBandwidth {
            peak_bytes_per_second: 1,
            locked_sm_clock_mhz: u64::MAX as f64,
            locked_mem_clock_mhz: 1.0,
            provenance: String::new(),
        };
        assert!(matches!(
            overflow.locked_sm_clock_hz(),
            Err(CalibError::InvalidCardValue { .. })
        ));

        // Ordering sanity mirrors the Python lockstep test.
        assert!(l.shared < l.l1_hit);
        assert!(l.l1_hit < l.l2_hit);
        assert!(l.l2_hit < l.dram);
    }

    #[test]
    fn production_cards_keep_issue_resources_absent_and_typed() {
        for name in ["sm_86.json", "sm_120-laptop.json"] {
            let value: serde_json::Value =
                serde_json::from_slice(&std::fs::read(card_path(name)).unwrap()).unwrap();
            for block in [
                "smsp",
                "occupancy",
                "dep_barriers",
                "barrier_latencies",
                "dual_issue",
            ] {
                assert!(
                    value.get(block).is_none(),
                    "{name} unexpectedly carries {block}"
                );
            }

            let card = ArchCard::load(&card_path(name)).unwrap();
            assert!(matches!(
                card.required_issue_resources(),
                Err(CalibError::BlockResearchGap {
                    block: "smsp",
                    reason,
                    ..
                }) if reason.contains("field absent")
            ));
        }
    }

    #[test]
    fn synthetic_issue_resources_resolve_without_defaults() {
        let card = load_mutated_card("sm_86.json", install_synthetic_issue_resources).unwrap();
        assert_eq!(
            card.required_issue_resources().unwrap(),
            RequiredIssueResources {
                smsp: SmspResources {
                    count: 3,
                    issue_ports: 5,
                },
                occupancy: OccupancyResources {
                    max_resident_warps_per_sm: 7,
                    max_blocks: 11,
                },
                dep_barriers: DepBarrierResources { count: 13 },
                barrier_latencies: BarrierLatencyResources {
                    read: 17,
                    non_memory_write: 19,
                },
                dual_issue_supported: true,
            }
        );

        let integral_json_numbers = load_mutated_card("sm_86.json", |card| {
            install_synthetic_issue_resources(card);
            card["smsp"]["count"] = serde_json::json!(3.0);
            card["barrier_latencies"]["read"] = serde_json::json!(17.0);
        })
        .unwrap();
        let resources = integral_json_numbers.required_issue_resources().unwrap();
        assert_eq!(resources.smsp.count, 3);
        assert_eq!(resources.barrier_latencies.read, 17);
    }

    #[test]
    fn whole_and_nested_issue_resource_gaps_fail_typed() {
        for block in [
            "smsp",
            "occupancy",
            "dep_barriers",
            "barrier_latencies",
            "dual_issue",
        ] {
            let card = load_mutated_card("sm_86.json", |card| {
                install_synthetic_issue_resources(card);
                card[block] = serde_json::json!({
                    "status": "research_gap",
                    "reason": format!("synthetic {block} gap"),
                    "challenge": "fixture-only",
                });
            })
            .unwrap();
            assert!(matches!(
                card.required_issue_resources(),
                Err(CalibError::BlockResearchGap {
                    block: failed,
                    challenge,
                    ..
                }) if failed == block && challenge.as_deref() == Some("fixture-only")
            ));
        }

        let card = load_mutated_card("sm_86.json", |card| {
            install_synthetic_issue_resources(card);
            card["dual_issue"]["supported"] = serde_json::json!({
                "status": "research_gap",
                "reason": "synthetic pairing gap",
                "challenge": "fixture-only",
            });
        })
        .unwrap();
        assert!(matches!(
            card.required_issue_resources(),
            Err(CalibError::BlockResearchGap {
                block: "dual_issue.supported",
                reason,
                challenge,
                ..
            }) if reason == "synthetic pairing gap"
                && challenge.as_deref() == Some("fixture-only")
        ));
    }

    #[test]
    fn malformed_issue_resource_shapes_fail_at_card_parse() {
        for (block, malformed) in [
            ("smsp", serde_json::json!(null)),
            ("smsp", serde_json::json!({"count": 3})),
            (
                "smsp",
                serde_json::json!({"count": 3, "issue_ports": 5, "unknown": 1}),
            ),
            (
                "smsp",
                serde_json::json!({
                    "status": "research_gap",
                    "reason": "arm collision",
                    "count": 3,
                    "issue_ports": 5,
                }),
            ),
            (
                "smsp",
                serde_json::json!({"status": "unknown", "reason": "bad status"}),
            ),
            ("dual_issue", serde_json::json!({"supported": "unknown"})),
            (
                "dual_issue",
                serde_json::json!({"supported": {"status": "research_gap"}}),
            ),
            (
                "dual_issue",
                serde_json::json!({
                    "supported": {
                        "status": "research_gap",
                        "reason": "pending",
                        "unknown": 1,
                    }
                }),
            ),
        ] {
            assert!(matches!(
                load_mutated_card("sm_86.json", move |card| {
                    install_synthetic_issue_resources(card);
                    card[block] = malformed;
                }),
                Err(CalibError::Parse { .. })
            ));
        }
    }

    #[test]
    fn issue_resource_numbers_are_positive_and_type_bounded() {
        for (block, field) in [
            ("smsp", "count"),
            ("smsp", "issue_ports"),
            ("occupancy", "max_resident_warps_per_sm"),
            ("occupancy", "max_blocks"),
            ("dep_barriers", "count"),
            ("barrier_latencies", "read"),
            ("barrier_latencies", "non_memory_write"),
        ] {
            assert!(matches!(
                load_mutated_card("sm_86.json", |card| {
                    install_synthetic_issue_resources(card);
                    card[block][field] = serde_json::json!(0);
                }),
                Err(CalibError::Parse { .. })
            ));
        }

        assert!(matches!(
            load_mutated_card("sm_86.json", |card| {
                install_synthetic_issue_resources(card);
                card["smsp"]["count"] = serde_json::json!(u64::from(u32::MAX) + 1);
            }),
            Err(CalibError::Parse { .. })
        ));
        assert!(
            serde_json::from_str::<BarrierLatencyResources>(
                r#"{"read":9007199254740991,"non_memory_write":1}"#
            )
            .is_ok()
        );
        for bad in [
            r#"{"read":9007199254740992,"non_memory_write":1}"#,
            r#"{"read":9007199254740993.0,"non_memory_write":1}"#,
            r#"{"read":9007199254740991.000001,"non_memory_write":1}"#,
            r#"{"read":18446744073709551616,"non_memory_write":1}"#,
        ] {
            assert!(
                serde_json::from_str::<BarrierLatencyResources>(bad).is_err(),
                "{bad}"
            );
        }
        assert!(
            serde_json::from_str::<SmspResources>(
                r#"{"count":4294967295.0000001,"issue_ports":1}"#
            )
            .is_err(),
            "a decimal just above u32::MAX must not round into the admitted range"
        );
        assert_eq!(
            serde_json::from_str::<SmspResources>(r#"{"count":4294967295.0,"issue_ports":1}"#)
                .unwrap()
                .count,
            u32::MAX
        );
    }

    #[test]
    fn sm_count_must_be_positive() {
        assert!(matches!(
            load_mutated_card("sm_86.json", |card| {
                card["sm_count"] = serde_json::json!(0);
            }),
            Err(CalibError::InvalidCardValue {
                field: "sm_count",
                ..
            })
        ));
    }

    #[test]
    fn present_geometry_and_clock_ranges_are_validated() {
        let l1 = load_mutated_card("sm_86.json", |card| {
            card["l1_geometry"]["ways"] = serde_json::json!(0);
        })
        .unwrap();
        assert!(matches!(
            l1.require_l1_geometry(),
            Err(CalibError::InvalidCardValue {
                field: "l1_geometry.ways",
                ..
            })
        ));

        let l2 = load_mutated_card("sm_86.json", |card| {
            card["l2_geometry"]["line_bytes"] = serde_json::json!(33);
        })
        .unwrap();
        assert!(matches!(
            l2.require_l2_geometry(),
            Err(CalibError::InvalidCardValue {
                field: "l2_geometry.line_bytes",
                ..
            })
        ));

        let clocks = load_mutated_card("sm_86.json", |card| {
            card["nvml_reference_operating_point"]["max_sm_clock_mhz"] = serde_json::json!(1000.0);
        })
        .unwrap();
        assert!(matches!(
            clocks.require_nvml_reference_operating_point(),
            Err(CalibError::InvalidCardValue {
                field: "nvml_reference_operating_point",
                ..
            })
        ));
    }

    #[test]
    fn required_latency_rejects_invalid_measured_values() {
        let base = ArchCard::load(&card_path("sm_86.json")).expect("load sm_86 card");
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut card = base.clone();
            measured_mut(&mut card, LatencyKind::L1Hit).value = bad;
            match card.require_latency(LatencyKind::L1Hit) {
                Err(CalibError::InvalidCardValue {
                    field,
                    value,
                    requirement,
                }) => {
                    assert_eq!(field, "latencies_cycles.l1_hit.value");
                    assert_eq!(value, bad.to_string());
                    assert_eq!(requirement, "positive finite scalar");
                }
                other => panic!("expected typed invalid latency value, got {other:?}"),
            }
        }
    }

    #[test]
    fn required_latency_rejects_non_cycle_units() {
        let mut card = ArchCard::load(&card_path("sm_86.json")).unwrap();
        measured_mut(&mut card, LatencyKind::L1Hit).unit = Some("nanoseconds".to_owned());
        assert!(matches!(
            card.require_latency(LatencyKind::L1Hit),
            Err(CalibError::InvalidCardValue {
                field: "latencies_cycles.l1_hit.value",
                requirement: "latency unit must describe cycles",
                ..
            })
        ));
    }

    #[test]
    fn consumed_card_blocks_require_provenance_text() {
        let latency = load_mutated_card("sm_86.json", |card| {
            card["latencies_cycles"]["provenance"] = serde_json::json!(" ");
        })
        .unwrap();
        assert!(matches!(
            latency.required_latencies(),
            Err(CalibError::InvalidCardValue {
                field: "latencies_cycles.provenance",
                requirement: "non-empty provenance text",
                ..
            })
        ));

        let l2 = load_mutated_card("sm_86.json", |card| {
            card["l2_geometry"]["provenance"] = serde_json::json!("");
        })
        .unwrap();
        assert!(matches!(
            l2.require_l2_geometry(),
            Err(CalibError::InvalidCardValue {
                field: "l2_geometry.provenance",
                ..
            })
        ));

        let dram = load_mutated_card("sm_86.json", |card| {
            card["dram_bandwidth"]["provenance"] = serde_json::json!("\t");
        })
        .unwrap();
        assert!(matches!(
            dram.require_dram_bandwidth(),
            Err(CalibError::InvalidCardValue {
                field: "dram_bandwidth.provenance",
                ..
            })
        ));
    }

    #[test]
    fn required_latency_rejects_invalid_dispersion() {
        let base = ArchCard::load(&card_path("sm_86.json")).expect("load sm_86 card");
        for bad in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut card = base.clone();
            measured_mut(&mut card, LatencyKind::L1Hit).dispersion = Some(bad);
            match card.require_latency(LatencyKind::L1Hit) {
                Err(CalibError::InvalidCardValue {
                    field,
                    value,
                    requirement,
                }) => {
                    assert_eq!(field, "latencies_cycles.l1_hit.dispersion");
                    assert_eq!(value, bad.to_string());
                    assert_eq!(requirement, "non-negative finite scalar");
                }
                other => panic!("expected typed invalid latency dispersion, got {other:?}"),
            }
        }
    }

    /// The PARTIAL sm_120-laptop card must load: its gap blocks surface as
    /// typed BlockResearchGap on access, never as an opaque parse failure.
    #[test]
    fn sm120_partial_card_loads_with_typed_block_gaps() {
        let card = ArchCard::load(&card_path("sm_120-laptop.json")).expect("load sm_120 card");
        assert_eq!(card.arch_id().unwrap(), ArchId::Sm120);
        // l2_geometry is populated on this card.
        let l2 = card.require_l2_geometry().unwrap();
        assert_eq!(l2.capacity_bytes, 33_554_432);
        // l1_geometry + NVML op point are research gaps → typed errors.
        match card.require_l1_geometry() {
            Err(CalibError::BlockResearchGap { block, reason, .. }) => {
                assert_eq!(block, "l1_geometry");
                assert!(!reason.is_empty());
            }
            other => panic!("expected typed BlockResearchGap, got {other:?}"),
        }
        assert!(matches!(
            card.require_nvml_reference_operating_point(),
            Err(CalibError::BlockResearchGap { .. })
        ));
        // No latency table yet → typed LatenciesMissing.
        assert!(matches!(
            card.require_latency(LatencyKind::L1Hit),
            Err(CalibError::LatenciesMissing { .. })
        ));
        // DRAM bandwidth is a research gap (no locked operating point on
        // the DVFS-spanning laptop part) → typed, never inherited.
        assert!(matches!(
            card.require_dram_bandwidth(),
            Err(CalibError::BlockResearchGap { .. })
        ));
    }

    #[test]
    fn explicit_latency_block_gap_loads_and_fails_typed() {
        let card = load_mutated_card("sm_120-laptop.json", |card| {
            card["latencies_cycles"] = serde_json::json!({
                "status": "research_gap",
                "reason": "sm_120 latency table is not measured",
                "challenge": "requires a reviewed pchase campaign"
            });
        })
        .unwrap();
        assert!(matches!(
            card.require_latency(LatencyKind::L1Hit),
            Err(CalibError::BlockResearchGap {
                arch_id,
                block: "latencies_cycles",
                reason,
                challenge,
            }) if arch_id == "sm_120"
                && !reason.is_empty()
                && challenge.as_deref() == Some("requires a reviewed pchase campaign")
        ));
    }

    #[test]
    fn sm86_mshr_is_research_gap_typed() {
        let card = ArchCard::load(&card_path("sm_86.json")).expect("load sm_86 card");
        match card.require_latency(LatencyKind::MshrOutstandingMisses) {
            Err(CalibError::ResearchGap { kind, reason, .. }) => {
                assert_eq!(kind, LatencyKind::MshrOutstandingMisses);
                assert!(!reason.is_empty(), "gap must carry its reason");
            }
            other => panic!("expected typed ResearchGap, got {other:?}"),
        }
    }

    #[test]
    fn missing_latency_block_is_typed() {
        let card_json = r#"{
            "arch_id": "sm_86", "sku": "rtx_3090_ti",
            "compute_capability": [8, 6], "build_arch_family": ["sm_86"],
            "sm_count": 1, "bare_ldg_e_default": "ca",
            "l1_geometry": {"l1tex_bytes": 1, "ways": 1, "default_l1_bytes": 1,
                             "max_shared_bytes": 1, "sets": 1},
            "l2_geometry": {"capacity_bytes": 1, "sector_bytes": 32, "num_slices": 1,
                             "line_bytes": 128, "policy": "p", "provenance": "test"},
            "nvml_reference_operating_point": {"sm_clock_ref_mhz": 1.0, "max_sm_clock_mhz": 1.0}
        }"#;
        let card = load_inline_card(card_json).unwrap();
        assert!(matches!(
            card.require_latency(LatencyKind::L1Hit),
            Err(CalibError::LatenciesMissing { .. })
        ));
    }

    #[test]
    fn typo_gap_status_never_deserializes() {
        let bad = r#"{"status": "reseach_gap", "reason": "typo"}"#;
        assert!(serde_json::from_str::<Measured>(bad).is_err());
    }

    /// The half-migrated collision case: an entry carrying BOTH a value and
    /// a research_gap status must fail parsing outright — under the old
    /// untagged representation it silently deserialized as Value(0.0).
    #[test]
    fn value_and_status_collision_rejected() {
        let bad = r#"{"status": "research_gap", "reason": "pending", "value": 0.0}"#;
        let err = serde_json::from_str::<Measured>(bad).expect_err("collision must fail");
        assert!(err.to_string().contains("BOTH"));
    }

    #[test]
    fn every_scalar_value_gap_arm_contamination_is_rejected() {
        for bad in [
            r#"{"status":"research_gap","reason":"pending","dispersion":1.0}"#,
            r#"{"status":"research_gap","reason":"pending","unit":"cycles"}"#,
            r#"{"status":"research_gap","reason":"pending","unknown_payload":1}"#,
            r#"{"value":1.0,"reason":"not value metadata"}"#,
            r#"{"value":1.0,"challenge":"not value metadata"}"#,
            r#"{"value":1.0,"unknown_payload":1}"#,
        ] {
            let error = serde_json::from_str::<Measured>(bad).expect_err(bad);
            assert!(error.to_string().contains("BOTH"), "{bad}: {error}");
        }

        let gap = serde_json::from_str::<Measured>(
            r#"{"status":"research_gap","reason":"pending","challenge":"retained"}"#,
        )
        .unwrap();
        match gap {
            Measured::Gap { reason, challenge } => {
                assert_eq!(reason, "pending");
                assert_eq!(challenge.as_deref(), Some("retained"));
            }
            Measured::Value(_) => panic!("expected research gap"),
        }
        assert!(
            serde_json::from_str::<Measured>(
                r#"{"status":"research_gap","reason":"pending","challenge":1}"#
            )
            .is_err()
        );
        assert!(
            serde_json::from_str::<Measured>(r#"{"status":"research_gap"}"#).is_err(),
            "schema-required gap reason must also be required by Rust"
        );
        assert!(matches!(
            serde_json::from_str::<Measured>(r#"{"value":1.0,"dispersion":0.1,"unit":"cycles"}"#)
                .unwrap(),
            Measured::Value(_)
        ));
    }

    #[test]
    fn scalar_arm_contamination_is_rejected_at_card_load_boundary() {
        for entry in [
            serde_json::json!({
                "status": "research_gap",
                "reason": "pending",
                "value": 1.0,
            }),
            serde_json::json!({"value": 1.0, "reason": "gap-only"}),
            serde_json::json!({
                "status": "research_gap",
                "reason": "pending",
                "unknown_payload": 1,
            }),
        ] {
            assert!(matches!(
                load_mutated_card("sm_86.json", |card| {
                    card["latencies_cycles"]["l1_hit"] = entry;
                }),
                Err(CalibError::Parse { .. })
            ));
        }
    }

    #[test]
    fn empty_entry_rejected() {
        assert!(serde_json::from_str::<Measured>(r#"{"unit": "cycles"}"#).is_err());
    }

    #[test]
    fn carded_typo_status_rejected() {
        for bad in [
            r#"{"status": "reseach_gap", "reason": "typo"}"#,
            r#"{"status": "research_gap"}"#,
            r#"{"status": "research_gap", "reason": "pending", "challenge": 1}"#,
        ] {
            assert!(
                serde_json::from_str::<Carded<L1Geometry>>(bad).is_err(),
                "{bad}"
            );
        }
    }
}
