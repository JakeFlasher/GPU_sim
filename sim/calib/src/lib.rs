//! sim-calib: loader for the per-arch calibration cards
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

use std::path::Path;

use serde::Deserialize;
use serde::de::DeserializeOwned;
use sim_arch::{ArchError, ArchId, ArchIdentity, BuildArch, Sku};

#[derive(Debug, thiserror::Error)]
pub enum CalibError {
    #[error("could not read card {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("could not parse card {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("card architecture identity is invalid: {0}")]
    Architecture(#[from] ArchError),
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
        "card {arch_id}: block '{block}' is a research_gap ({reason}); the consumer must \
         type-exclude, never guess geometry"
    )]
    BlockResearchGap {
        arch_id: String,
        block: &'static str,
        reason: String,
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
    Gap { reason: Option<String> },
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
}

impl<'de> Deserialize<'de> for Measured {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = RawObjOrGap::deserialize(deserializer)?;
        match (raw.value, raw.status) {
            (Some(value), None) => Ok(Measured::Value(MeasuredValue {
                value,
                dispersion: raw.dispersion,
                unit: raw.unit,
            })),
            (None, Some(GapStatus::ResearchGap)) => Ok(Measured::Gap { reason: raw.reason }),
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
    Gap { reason: Option<String> },
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
            CardedState::Gap { reason } => Err(CalibError::BlockResearchGap {
                arch_id: arch.to_string(),
                block,
                reason: reason.clone().unwrap_or_default(),
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
                reason: Some(
                    "field absent from this card (card predates the block, OR the card's key is \
                     misspelled — unknown keys are ignored on parse)"
                        .to_owned(),
                ),
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
                .and_then(|r| r.as_str())
                .map(str::to_string);
            return Ok(Self {
                state: CardedState::Gap { reason },
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
pub struct LatenciesCycles {
    pub l1_hit: Measured,
    pub l2_hit: Measured,
    pub dram: Measured,
    pub shared: Measured,
    pub mshr_outstanding_misses: Measured,
    pub provenance: String,
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

/// The subset of an ArchProfile card the simulator consumes. Unknown fields
/// are ignored on purpose: cards carry label-side bookkeeping (predictor
/// state, crosswalk gates) that must never leak into simulator inputs.
#[derive(Debug, Clone, Deserialize)]
pub struct ArchCard {
    pub arch_id: String,
    pub sku: String,
    pub compute_capability: (u32, u32),
    pub sm_count: u32,
    pub l1_geometry: Carded<L1Geometry>,
    pub l2_geometry: Carded<L2Geometry>,
    pub nvml_reference_operating_point: Carded<NvmlReferenceOperatingPoint>,
    /// DRAM read-bandwidth covariate; research_gap on cards without a
    /// locked operating point (sm_120-laptop). Absent on legacy cards →
    /// treated as a gap, never a default.
    #[serde(default = "Carded::absent_gap")]
    pub dram_bandwidth: Carded<DramBandwidth>,
    #[serde(default)]
    pub latencies_cycles: Option<LatenciesCycles>,
    /// Load classifier default for bare LDG.E (per-arch measured fact).
    #[serde(default)]
    pub bare_ldg_e_default: Option<String>,
}

impl ArchCard {
    pub fn load(path: &Path) -> Result<Self, CalibError> {
        let raw = std::fs::read_to_string(path).map_err(|source| CalibError::Io {
            path: path.display().to_string(),
            source,
        })?;
        let mut card: Self = serde_json::from_str(&raw).map_err(|source| CalibError::Parse {
            path: path.display().to_string(),
            source,
        })?;
        let arch = card.arch_id()?;
        card.sku_id()?;
        card.l1_geometry = card.l1_geometry.bind_arch(arch);
        card.l2_geometry = card.l2_geometry.bind_arch(arch);
        card.nvml_reference_operating_point = card.nvml_reference_operating_point.bind_arch(arch);
        card.dram_bandwidth = card.dram_bandwidth.bind_arch(arch);
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
        Ok(sku)
    }

    /// Bind the card's deployment identity, SKU, and compile-time build token
    /// before a simulator composition path selects architecture-local facts.
    pub fn identity(&self, build: BuildArch) -> Result<ArchIdentity, CalibError> {
        Ok(ArchIdentity::new(self.arch_id()?, self.sku_id()?, build)?)
    }

    /// Require a card block using the card's own validated architecture.  The
    /// architecture is intentionally not caller-supplied: a present sm_86
    /// block must never be retrievable by labeling it sm_120.
    pub fn require_l1_geometry(&self) -> Result<&L1Geometry, CalibError> {
        let arch = self.sku_id()?.arch();
        self.l1_geometry.require_for_arch(arch, "l1_geometry")
    }

    pub fn require_l2_geometry(&self) -> Result<&L2Geometry, CalibError> {
        let arch = self.sku_id()?.arch();
        self.l2_geometry.require_for_arch(arch, "l2_geometry")
    }

    pub fn require_nvml_reference_operating_point(
        &self,
    ) -> Result<&NvmlReferenceOperatingPoint, CalibError> {
        let arch = self.sku_id()?.arch();
        self.nvml_reference_operating_point
            .require_for_arch(arch, "nvml_reference_operating_point")
    }

    pub fn require_dram_bandwidth(&self) -> Result<&DramBandwidth, CalibError> {
        let arch = self.sku_id()?.arch();
        self.dram_bandwidth.require_for_arch(arch, "dram_bandwidth")
    }

    /// The latency table, or a typed error if the card predates Phase 1D.
    pub fn latencies(&self) -> Result<&LatenciesCycles, CalibError> {
        // Latency values are architecture-local inputs; validate the card
        // identity before exposing even a present table.
        self.sku_id()?;
        self.latencies_cycles
            .as_ref()
            .ok_or_else(|| CalibError::LatenciesMissing {
                arch_id: self.arch_id.clone(),
            })
    }

    /// A required latency in cycles. `research_gap` → typed error, never a
    /// default.
    pub fn require_latency(&self, kind: LatencyKind) -> Result<f64, CalibError> {
        let lat = self.latencies()?;
        let entry = match kind {
            LatencyKind::L1Hit => &lat.l1_hit,
            LatencyKind::L2Hit => &lat.l2_hit,
            LatencyKind::Dram => &lat.dram,
            LatencyKind::Shared => &lat.shared,
            LatencyKind::MshrOutstandingMisses => &lat.mshr_outstanding_misses,
        };
        match entry {
            Measured::Value(v) => Ok(v.value),
            Measured::Gap { reason } => Err(CalibError::ResearchGap {
                arch_id: self.arch_id.clone(),
                kind,
                reason: reason.clone().unwrap_or_default(),
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

#[cfg(test)]
#[allow(clippy::unwrap_used)] // unwrap in tests is an assertion, not error handling
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn card_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../arch_profiles/cards")
            .join(name)
            .canonicalize()
            .expect("card exists in-repo")
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

        let mut malformed = sm86;
        malformed.compute_capability = (12, 0);
        assert!(matches!(
            malformed.arch_id(),
            Err(CalibError::ComputeCapabilityMismatch { .. })
        ));
    }

    /// Python lockstep test (test_arch_profile_lockstep.py) pins against the
    /// pchase v2.1 evidence.
    #[test]
    fn sm86_card_lockstep() {
        let card = ArchCard::load(&card_path("sm_86.json")).expect("load sm_86 card");
        assert_eq!(card.arch_id, "sm_86");
        assert_eq!(card.compute_capability, (8, 6));
        assert_eq!(card.sm_count, 84);
        let l2 = card.require_l2_geometry().unwrap();
        assert_eq!(l2.capacity_bytes, 6_291_456);
        assert_eq!(l2.num_slices, 6);
        assert_eq!(l2.sector_bytes, 32);
        assert_eq!(l2.line_bytes, 128);
        let l1 = card.require_l1_geometry().unwrap();
        assert_eq!(l1.l1tex_bytes, 131_072);
        assert_eq!(l1.default_l1_bytes, 32_768);
        assert_eq!(card.bare_ldg_e_default.as_deref(), Some("ca"));

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

    /// The PARTIAL sm_120-laptop card must load: its gap blocks surface as
    /// typed BlockResearchGap on access, never as an opaque parse failure.
    #[test]
    fn sm120_partial_card_loads_with_typed_block_gaps() {
        let card = ArchCard::load(&card_path("sm_120-laptop.json")).expect("load sm_120 card");
        assert_eq!(card.arch_id, "sm_120");
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
            "compute_capability": [8, 6], "sm_count": 1,
            "l1_geometry": {"l1tex_bytes": 1, "ways": 1, "default_l1_bytes": 1,
                             "max_shared_bytes": 1, "sets": 1},
            "l2_geometry": {"capacity_bytes": 1, "sector_bytes": 32, "num_slices": 1,
                             "line_bytes": 128, "policy": "p", "provenance": "test"},
            "nvml_reference_operating_point": {"sm_clock_ref_mhz": 1.0, "max_sm_clock_mhz": 1.0}
        }"#;
        let card: ArchCard = serde_json::from_str(card_json).unwrap();
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
    fn empty_entry_rejected() {
        assert!(serde_json::from_str::<Measured>(r#"{"unit": "cycles"}"#).is_err());
    }

    #[test]
    fn carded_typo_status_rejected() {
        let bad = r#"{"status": "reseach_gap", "reason": "typo"}"#;
        assert!(serde_json::from_str::<Carded<L1Geometry>>(bad).is_err());
    }
}
