//! Geometry/latency resolvers over the committed P-chase static card
//! (`data/pchase/sm_86_static_card.json`) — a lockstep port of
//! `tools/cache_predictor/l1_warp_pool_walker.py`'s card resolvers.
//!
//! The resolution CONTRACT is the Python one, verbatim:
//! - a `kind: "value"` marker alone is never trusted — the value field must
//!   exist AND pass the per-resolver audit (positive integral for counts,
//!   positive finite for latencies, hit strictly below miss);
//! - capacity has NO default at all (missing/typed property fails closed)
//!   and is carveout-aware via the swept
//!   `l1_capacity_carveout_{pct}_pct_bytes` points;
//! - a `kind: "typed"` (`geometry_unresolved`) MSHR count resolves to the
//!   DOCUMENTED default 32 with a mandatory sensitivity note;
//! - a typed replacement policy resolves to LRU as the documented
//!   approximation; a card-VALUED policy other than "lru" fails closed;
//! - a card-VALUED associativity fails closed (the set-indexed walk is not
//!   modeled); typed resolves to the fully-associative documented
//!   approximation.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use sim_arch::{ArchError, ArchId};

use crate::l1::LINE_BYTES;

/// Documented default when the card types the MSHR property
/// `geometry_unresolved` (the card's own probe bounds outstanding misses
/// from below at 32; the dossier's walker line uses 32 as the per-SM
/// working default).
pub const DOCUMENTED_DEFAULT_MSHR_COUNT: u32 = 32;

#[derive(Debug, thiserror::Error)]
pub enum PchaseCardError {
    #[error("failed to read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("invalid architecture token in P-chase card: {0}")]
    Architecture(#[from] ArchError),
    #[error("P-chase card architecture mismatch: card is {card}, requested {requested}")]
    ArchitectureMismatch { card: ArchId, requested: ArchId },

    #[error("arch {arch} cannot consume the {property} research gap: {reason}")]
    ArchitectureCapabilityGap {
        arch: String,
        property: String,
        reason: String,
    },
    #[error("the card has no {property} property; an unsourced constant is rejected")]
    PropertyMissing { property: String },
    #[error("the card property {property} has kind=value but no value field; fail closed")]
    ValueFieldMissing { property: String },
    #[error("{property} value {raw} fails the audit ({requirement}); fail closed")]
    ValueAuditFailed {
        property: String,
        raw: String,
        requirement: &'static str,
    },
    #[error("unrecognized card property kind {kind:?} on {property}")]
    UnrecognizedKind { property: String, kind: String },
    #[error(
        "the card types {property} {typed_reason:?} and this property has no documented \
         default — fail closed"
    )]
    TypedWithoutDefault {
        property: String,
        typed_reason: String,
    },
    #[error(
        "card-sourced replacement policy {policy:?} is not modeled; fail closed rather than \
         silently approximating"
    )]
    UnmodeledReplacementPolicy { policy: String },
    #[error(
        "card-sourced associativity ways={raw} needs a set-indexed walk that is not modeled; \
         fail closed rather than silently treating sets as fully-associative"
    )]
    UnmodeledSetAssociativity { raw: String },
    #[error(
        "card hit latency {hit} is not strictly below the miss latency {miss}; the \
         latency-proportional schedule would invert — fail closed"
    )]
    LatencyOrderInverted { hit: f64, miss: f64 },
    #[error("{property}={bytes} is not a positive multiple of the {line}B line")]
    CapacityNotLineMultiple {
        property: String,
        bytes: u64,
        line: u32,
    },
}

#[derive(Debug, Deserialize)]
struct RawProperty {
    kind: String,
    #[serde(default)]
    value: Option<serde_json::Value>,
    #[serde(default)]
    typed_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawCard {
    arch: String,
    properties: BTreeMap<String, RawProperty>,
}

/// Provenance-tagged resolved values (mirrors the Python NamedTuples).
#[derive(Debug, Clone)]
pub struct LineCapacityConfig {
    pub line_capacity: u64,
    pub capacity_bytes: u64,
    pub property_name: String,
    /// Always "card" — capacity has no default.
    pub source: &'static str,
}

#[derive(Debug, Clone)]
pub struct MshrConfig {
    /// `None` never comes from the card (ablation only in Python); here a
    /// resolved count is always present.
    pub count: u32,
    pub source: &'static str,
    pub sensitivity_note: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LatencyConfig {
    pub hit_latency: f64,
    pub miss_latency: f64,
    pub source: &'static str,
}

#[derive(Debug, Clone)]
pub struct GeometryPolicyConfig {
    /// "lru" | "fully_associative"
    pub modeled: &'static str,
    pub source: &'static str,
    pub note: Option<String>,
}

/// Everything the L1 model needs from the card, resolved in one call.
#[derive(Debug, Clone)]
pub struct WalkGeometry {
    pub capacity: LineCapacityConfig,
    pub replacement: GeometryPolicyConfig,
    pub set_mapping: GeometryPolicyConfig,
    pub mshr: MshrConfig,
    pub latencies: LatencyConfig,
}

pub struct PchaseCard {
    arch: ArchId,
    properties: BTreeMap<String, RawProperty>,
}

impl PchaseCard {
    pub fn load(path: &Path) -> Result<Self, PchaseCardError> {
        let raw = std::fs::read(path).map_err(|source| PchaseCardError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let card: RawCard =
            serde_json::from_slice(&raw).map_err(|source| PchaseCardError::Parse {
                path: path.to_path_buf(),
                source,
            })?;
        let arch = ArchId::parse_deploy(&card.arch)?;
        Ok(Self {
            arch,
            properties: card.properties,
        })
    }

    pub const fn arch(&self) -> ArchId {
        self.arch
    }

    fn require_arch(&self, requested: ArchId) -> Result<(), PchaseCardError> {
        if requested != self.arch {
            return Err(PchaseCardError::ArchitectureMismatch {
                card: self.arch,
                requested,
            });
        }
        Ok(())
    }

    fn property(&self, name: &str) -> Result<&RawProperty, PchaseCardError> {
        self.properties
            .get(name)
            .ok_or_else(|| PchaseCardError::PropertyMissing {
                property: name.to_owned(),
            })
    }

    fn require_value<'a>(
        prop: &'a RawProperty,
        name: &str,
    ) -> Result<&'a serde_json::Value, PchaseCardError> {
        prop.value
            .as_ref()
            .ok_or_else(|| PchaseCardError::ValueFieldMissing {
                property: name.to_owned(),
            })
    }

    /// Audit a card-VALUED integer: positive, integral, finite, and not a
    /// bool/string masquerading through coercion (the Python `_card_int`).
    fn card_int(prop: &RawProperty, name: &str) -> Result<u64, PchaseCardError> {
        let raw = Self::require_value(prop, name)?;
        let fail = |requirement| PchaseCardError::ValueAuditFailed {
            property: name.to_owned(),
            raw: raw.to_string(),
            requirement,
        };
        let n = raw.as_f64().ok_or_else(|| fail("must be a JSON number"))?;
        if !n.is_finite() || n.fract() != 0.0 || n < 1.0 {
            return Err(fail("must be a positive integer"));
        }
        Ok(n as u64)
    }

    fn card_positive_float(prop: &RawProperty, name: &str) -> Result<f64, PchaseCardError> {
        if prop.kind != "value" {
            return Err(PchaseCardError::TypedWithoutDefault {
                property: name.to_owned(),
                typed_reason: prop.typed_reason.clone().unwrap_or_default(),
            });
        }
        let raw = Self::require_value(prop, name)?;
        let n = raw
            .as_f64()
            .ok_or_else(|| PchaseCardError::ValueAuditFailed {
                property: name.to_owned(),
                raw: raw.to_string(),
                requirement: "must be a JSON number",
            })?;
        if !n.is_finite() || n <= 0.0 {
            return Err(PchaseCardError::ValueAuditFailed {
                property: name.to_owned(),
                raw: raw.to_string(),
                requirement: "must be a positive finite latency",
            });
        }
        Ok(n)
    }

    /// Carveout-aware L1 capacity. `carveout_pct=None` reads the
    /// default-carveout `l1_capacity_bytes`; `Some(pct)` reads the swept
    /// `l1_capacity_carveout_{pct}_pct_bytes` point. NO default.
    fn resolve_line_capacity(
        &self,
        carveout_pct: Option<u32>,
    ) -> Result<LineCapacityConfig, PchaseCardError> {
        let name = match carveout_pct {
            None => "l1_capacity_bytes".to_owned(),
            Some(pct) => format!("l1_capacity_carveout_{pct}_pct_bytes"),
        };
        let prop = self.property(&name)?;
        if prop.kind != "value" {
            return Err(PchaseCardError::TypedWithoutDefault {
                property: name,
                typed_reason: prop.typed_reason.clone().unwrap_or_default(),
            });
        }
        let capacity_bytes = Self::card_int(prop, &name)?;
        if capacity_bytes % u64::from(LINE_BYTES) != 0 {
            return Err(PchaseCardError::CapacityNotLineMultiple {
                property: name,
                bytes: capacity_bytes,
                line: LINE_BYTES,
            });
        }
        Ok(LineCapacityConfig {
            line_capacity: capacity_bytes / u64::from(LINE_BYTES),
            capacity_bytes,
            property_name: name,
            source: "card",
        })
    }

    /// Resolve MSHR capacity for a typed deployment architecture.  The
    /// documented 32-entry approximation is admitted only for reviewed
    /// sm_86; a typed gap on sm_120 remains a typed exclusion.
    pub fn resolve_mshr_count_for_arch(&self, arch: ArchId) -> Result<MshrConfig, PchaseCardError> {
        self.require_arch(arch)?;
        let name = "mshr_outstanding_misses";
        let prop = self.property(name)?;
        match prop.kind.as_str() {
            "value" => Ok(MshrConfig {
                count: u32::try_from(Self::card_int(prop, name)?).map_err(|_| {
                    PchaseCardError::ValueAuditFailed {
                        property: name.to_owned(),
                        raw: format!("{:?}", prop.value),
                        requirement: "must fit u32",
                    }
                })?,
                source: "card",
                sensitivity_note: None,
            }),
            "typed" if arch == ArchId::Sm86 => Ok(MshrConfig {
                count: DOCUMENTED_DEFAULT_MSHR_COUNT,
                source: "documented_default",
                sensitivity_note: Some(format!(
                    "the card types mshr_outstanding_misses geometry_unresolved (the \
                     single-thread probe bounds outstanding misses from below without a \
                     saturation knee), so the walk uses the documented default \
                     {DOCUMENTED_DEFAULT_MSHR_COUNT}; report results at counts [16, 32, 64] \
                     for sensitivity"
                )),
            }),
            "typed" => Err(PchaseCardError::ArchitectureCapabilityGap {
                arch: arch.to_string(),
                property: name.to_owned(),
                reason: "the documented MSHR approximation is reviewed only for sm_86".to_owned(),
            }),
            other => Err(PchaseCardError::UnrecognizedKind {
                property: name.to_owned(),
                kind: other.to_owned(),
            }),
        }
    }

    fn resolve_replacement_policy(&self) -> Result<GeometryPolicyConfig, PchaseCardError> {
        let name = "l1_replacement_policy";
        let prop = self.property(name)?;
        match prop.kind.as_str() {
            "value" => {
                let raw = Self::require_value(prop, name)?;
                let policy = raw
                    .as_str()
                    .ok_or_else(|| PchaseCardError::ValueAuditFailed {
                        property: name.to_owned(),
                        raw: raw.to_string(),
                        requirement: "must be a policy name string",
                    })?;
                if policy == "lru" {
                    Ok(GeometryPolicyConfig {
                        modeled: "lru",
                        source: "card",
                        note: None,
                    })
                } else {
                    Err(PchaseCardError::UnmodeledReplacementPolicy {
                        policy: policy.to_owned(),
                    })
                }
            }
            "typed" => Ok(GeometryPolicyConfig {
                modeled: "lru",
                source: "documented_approximation",
                note: Some(format!(
                    "the card types l1_replacement_policy {:?}; LRU is the documented \
                     approximation pending card evidence",
                    prop.typed_reason.clone().unwrap_or_default()
                )),
            }),
            other => Err(PchaseCardError::UnrecognizedKind {
                property: name.to_owned(),
                kind: other.to_owned(),
            }),
        }
    }

    fn resolve_set_mapping(&self) -> Result<GeometryPolicyConfig, PchaseCardError> {
        let name = "l1_associativity_ways";
        let prop = self.property(name)?;
        match prop.kind.as_str() {
            "value" => Err(PchaseCardError::UnmodeledSetAssociativity {
                raw: format!("{:?}", prop.value),
            }),
            "typed" => Ok(GeometryPolicyConfig {
                modeled: "fully_associative",
                source: "documented_approximation",
                note: Some(format!(
                    "the card types l1_associativity_ways {:?}; the fully-associative walk \
                     is the documented approximation pending card evidence",
                    prop.typed_reason.clone().unwrap_or_default()
                )),
            }),
            other => Err(PchaseCardError::UnrecognizedKind {
                property: name.to_owned(),
                kind: other.to_owned(),
            }),
        }
    }

    fn resolve_latencies(&self) -> Result<LatencyConfig, PchaseCardError> {
        let hit = Self::card_positive_float(
            self.property("l1_hit_latency_cycles")?,
            "l1_hit_latency_cycles",
        )?;
        let miss = Self::card_positive_float(
            self.property("l2_hit_latency_cycles")?,
            "l2_hit_latency_cycles",
        )?;
        if hit >= miss {
            return Err(PchaseCardError::LatencyOrderInverted { hit, miss });
        }
        Ok(LatencyConfig {
            hit_latency: hit,
            miss_latency: miss,
            source: "card",
        })
    }

    /// Architecture-aware resolver for every card-sourced L1 input.
    pub fn resolve_walk_geometry_for_arch(
        &self,
        arch: ArchId,
        carveout_pct: Option<u32>,
    ) -> Result<WalkGeometry, PchaseCardError> {
        self.require_arch(arch)?;
        Ok(WalkGeometry {
            capacity: self.resolve_line_capacity(carveout_pct)?,
            replacement: self.resolve_replacement_policy()?,
            set_mapping: self.resolve_set_mapping()?,
            mshr: self.resolve_mshr_count_for_arch(arch)?,
            latencies: self.resolve_latencies()?,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn card_for_arch(json: &str, arch_token: &str) -> PchaseCard {
        let mut value: serde_json::Value = serde_json::from_str(json).unwrap();
        value["arch"] = serde_json::Value::String(arch_token.to_owned());
        let raw: RawCard = serde_json::from_value(value).unwrap();
        PchaseCard {
            arch: ArchId::parse_deploy(&raw.arch).unwrap(),
            properties: raw.properties,
        }
    }

    fn card(json: &str) -> PchaseCard {
        card_for_arch(json, "sm_86")
    }

    const GOOD: &str = r#"{"properties": {
        "l1_capacity_bytes": {"kind": "value", "value": 98304.0},
        "l1_capacity_carveout_50_pct_bytes": {"kind": "value", "value": 49152.0},
        "l1_hit_latency_cycles": {"kind": "value", "value": 33.0016},
        "l2_hit_latency_cycles": {"kind": "value", "value": 243.6152},
        "mshr_outstanding_misses": {"kind": "typed", "typed_reason": "geometry_unresolved"},
        "l1_replacement_policy": {"kind": "typed", "typed_reason": "geometry_unresolved"},
        "l1_associativity_ways": {"kind": "typed", "typed_reason": "geometry_unresolved"}
    }}"#;

    #[test]
    fn resolves_geometry_like_python() {
        let c = card(GOOD);
        let g = c
            .resolve_walk_geometry_for_arch(ArchId::Sm86, None)
            .unwrap();
        assert_eq!(g.capacity.line_capacity, 768); // 98304 / 128
        assert_eq!(g.mshr.count, 32);
        assert_eq!(g.mshr.source, "documented_default");
        assert_eq!(g.replacement.modeled, "lru");
        assert_eq!(g.set_mapping.modeled, "fully_associative");
        assert!((g.latencies.hit_latency - 33.0016).abs() < 1e-9);
        let g50 = c
            .resolve_walk_geometry_for_arch(ArchId::Sm86, Some(50))
            .unwrap();
        assert_eq!(g50.capacity.line_capacity, 384);
        assert_eq!(
            g50.capacity.property_name,
            "l1_capacity_carveout_50_pct_bytes"
        );
    }

    #[test]
    fn card_architecture_mismatch_fails_before_value_resolution() {
        let sm86 = card(GOOD);
        assert_eq!(sm86.arch(), ArchId::Sm86);
        assert!(matches!(
            sm86.resolve_walk_geometry_for_arch(ArchId::Sm120, None),
            Err(PchaseCardError::ArchitectureMismatch {
                card: ArchId::Sm86,
                requested: ArchId::Sm120,
            })
        ));

        let sm120 = card_for_arch(GOOD, "sm_120");
        assert!(matches!(
            sm120.resolve_mshr_count_for_arch(ArchId::Sm120),
            Err(PchaseCardError::ArchitectureCapabilityGap { .. })
        ));
        assert!(matches!(
            sm120.resolve_mshr_count_for_arch(ArchId::Sm86),
            Err(PchaseCardError::ArchitectureMismatch {
                card: ArchId::Sm120,
                requested: ArchId::Sm86,
            })
        ));
    }

    #[test]
    fn capacity_has_no_default() {
        let c = card(
            r#"{"properties": {"l1_capacity_bytes":
            {"kind": "typed", "typed_reason": "geometry_unresolved"}}}"#,
        );
        assert!(matches!(
            c.resolve_line_capacity(None),
            Err(PchaseCardError::TypedWithoutDefault { .. })
        ));
        let c2 = card(r#"{"properties": {}}"#);
        assert!(matches!(
            c2.resolve_line_capacity(None),
            Err(PchaseCardError::PropertyMissing { .. })
        ));
    }

    #[test]
    fn value_marker_alone_is_not_trusted() {
        let c = card(r#"{"properties": {"l1_capacity_bytes": {"kind": "value"}}}"#);
        assert!(matches!(
            c.resolve_line_capacity(None),
            Err(PchaseCardError::ValueFieldMissing { .. })
        ));
        let c2 = card(
            r#"{"properties": {"l1_capacity_bytes":
            {"kind": "value", "value": 98305.0}}}"#,
        );
        assert!(matches!(
            c2.resolve_line_capacity(None),
            Err(PchaseCardError::CapacityNotLineMultiple { .. })
        ));
        let c3 = card(
            r#"{"properties": {"l1_capacity_bytes":
            {"kind": "value", "value": -128.0}}}"#,
        );
        assert!(matches!(
            c3.resolve_line_capacity(None),
            Err(PchaseCardError::ValueAuditFailed { .. })
        ));
    }

    #[test]
    fn valued_non_lru_policy_and_valued_ways_fail_closed() {
        let c = card(
            r#"{"properties": {"l1_replacement_policy":
            {"kind": "value", "value": "plru"}}}"#,
        );
        assert!(matches!(
            c.resolve_replacement_policy(),
            Err(PchaseCardError::UnmodeledReplacementPolicy { .. })
        ));
        let c2 = card(
            r#"{"properties": {"l1_associativity_ways":
            {"kind": "value", "value": 4}}}"#,
        );
        assert!(matches!(
            c2.resolve_set_mapping(),
            Err(PchaseCardError::UnmodeledSetAssociativity { .. })
        ));
    }

    #[test]
    fn latency_inversion_fails_closed() {
        let c = card(
            r#"{"properties": {
            "l1_hit_latency_cycles": {"kind": "value", "value": 250.0},
            "l2_hit_latency_cycles": {"kind": "value", "value": 243.6}}}"#,
        );
        assert!(matches!(
            c.resolve_latencies(),
            Err(PchaseCardError::LatencyOrderInverted { .. })
        ));
    }
}
