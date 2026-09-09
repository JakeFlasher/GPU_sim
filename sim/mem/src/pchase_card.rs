//! Geometry/latency resolvers over the committed P-chase static card
//! (`data/pchase/sm_86_static_card.json`) — a lockstep port of
//! `tools/cache_predictor/l1_warp_pool_walker.py`'s card resolvers.
//!
//! The resolver follows the Python implementation's input semantics:
//! - a `kind: "value"` marker alone is never trusted — the value field must
//!   exist AND pass per-field validation (positive integral for counts,
//!   positive finite for latencies, hit strictly below miss);
//! - capacity has NO default at all (missing/typed property fails closed)
//!   and is carveout-aware via the swept
//!   `l1_capacity_carveout_{pct}_pct_bytes` points;
//! - a `kind: "typed"` (`geometry_unresolved`) MSHR count resolves through
//!   the deployment architecture's explicit L1-default capability;
//! - typed replacement and associativity declarations resolve through that
//!   same architecture-bound capability, so sm_120 cannot inherit sm_86's LRU
//!   or fully-associative approximations;
//! - a card-VALUED replacement policy other than "lru" fails closed;
//! - a card-VALUED associativity fails closed (the set-indexed walk is not
//!   modeled).

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::value::RawValue;
use sim_arch::{
    ArchError, ArchId, CapabilityError, L1Replacement as ArchL1Replacement,
    L1SetMapping as ArchL1SetMapping, strict_json_from_slice,
};

use crate::l1::LINE_BYTES;

const MAX_PCHASE_CARD_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum PchaseCardError {
    #[error("failed to read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("P-chase card {path} is {observed} bytes, exceeding implementation limit {limit}")]
    ResourceLimit {
        path: PathBuf,
        observed: u64,
        limit: u64,
    },
    #[error("failed to parse {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("invalid architecture token in P-chase card: {0}")]
    Architecture(#[from] ArchError),
    #[error("architecture capability is unavailable: {0}")]
    Capability(#[from] CapabilityError),
    #[error("P-chase card architecture mismatch: card is {card}, requested {requested}")]
    ArchitectureMismatch { card: ArchId, requested: ArchId },

    #[error("the card has no {property} property; an unsourced constant is rejected")]
    PropertyMissing { property: String },
    #[error("the card property {property} has kind=value but no value field; fail closed")]
    ValueFieldMissing { property: String },
    #[error("{property} value {raw} is invalid ({requirement})")]
    ValueInvalid {
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
    value: Option<Box<RawValue>>,
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
        let raw = read_bounded_pchase_card(path, MAX_PCHASE_CARD_BYTES)?;
        Self::parse(path, &raw)
    }

    /// Parse already-snapshotted card bytes through the ordinary semantic gate.
    pub fn parse(path: &Path, raw: &[u8]) -> Result<Self, PchaseCardError> {
        let card: RawCard =
            strict_json_from_slice(raw).map_err(|source| PchaseCardError::Parse {
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
    ) -> Result<&'a RawValue, PchaseCardError> {
        prop.value
            .as_deref()
            .ok_or_else(|| PchaseCardError::ValueFieldMissing {
                property: name.to_owned(),
            })
    }

    /// Parse a JSON number lexeme as a mathematically exact positive `u64`.
    /// This accepts integral decimal/exponent spellings such as `128.0` and
    /// `1.28e2`, but never rounds through `f64`.
    fn exact_positive_integer(raw: &str) -> Option<u64> {
        let raw = raw.trim();
        if raw.is_empty() || raw.starts_with('-') {
            return None;
        }
        let exponent_at = raw.find(['e', 'E']);
        if exponent_at.is_some_and(|at| raw[at + 1..].contains(['e', 'E'])) {
            return None;
        }
        let (mantissa, exponent) = match exponent_at {
            Some(at) => {
                let exponent = raw[at + 1..].parse::<i64>().ok()?;
                (&raw[..at], exponent)
            }
            None => (raw, 0),
        };
        let mut parts = mantissa.split('.');
        let integer = parts.next()?;
        let fraction = parts.next().unwrap_or("");
        if parts.next().is_some()
            || integer.is_empty()
            || !integer.bytes().all(|digit| digit.is_ascii_digit())
            || !fraction.bytes().all(|digit| digit.is_ascii_digit())
        {
            return None;
        }

        let fraction_len = i64::try_from(fraction.len()).ok()?;
        let scale = exponent.checked_sub(fraction_len)?;
        let mut digits = String::with_capacity(integer.len().checked_add(fraction.len())?);
        digits.push_str(integer);
        digits.push_str(fraction);
        let significant = digits.trim_start_matches('0');
        if significant.is_empty() {
            return None;
        }

        let (integral, trailing_zeroes) = if scale < 0 {
            let remove = usize::try_from(scale.checked_neg()?).ok()?;
            if remove >= significant.len()
                || !significant[significant.len() - remove..]
                    .bytes()
                    .all(|digit| digit == b'0')
            {
                return None;
            }
            (&significant[..significant.len() - remove], 0usize)
        } else {
            (significant, usize::try_from(scale).ok()?)
        };
        if integral.len().checked_add(trailing_zeroes)? > 20 {
            return None;
        }
        let mut value = 0u64;
        for digit in integral.bytes() {
            value = value
                .checked_mul(10)?
                .checked_add(u64::from(digit - b'0'))?;
        }
        for _ in 0..trailing_zeroes {
            value = value.checked_mul(10)?;
        }
        (value != 0).then_some(value)
    }

    /// Validate a card-valued integer: positive, mathematically integral, and
    /// not a bool/string masquerading through coercion (the Python
    /// `_card_int` contract without its binary64 narrowing).
    fn card_int(prop: &RawProperty, name: &str) -> Result<u64, PchaseCardError> {
        let raw = Self::require_value(prop, name)?;
        Self::exact_positive_integer(raw.get()).ok_or_else(|| PchaseCardError::ValueInvalid {
            property: name.to_owned(),
            raw: raw.get().to_owned(),
            requirement: "must be a positive integer representable as u64",
        })
    }

    fn card_positive_float(prop: &RawProperty, name: &str) -> Result<f64, PchaseCardError> {
        if prop.kind != "value" {
            return Err(PchaseCardError::TypedWithoutDefault {
                property: name.to_owned(),
                typed_reason: prop.typed_reason.clone().unwrap_or_default(),
            });
        }
        let raw = Self::require_value(prop, name)?;
        let n =
            serde_json::from_str::<f64>(raw.get()).map_err(|_| PchaseCardError::ValueInvalid {
                property: name.to_owned(),
                raw: raw.get().to_owned(),
                requirement: "must be a JSON number representable as finite f64",
            })?;
        if !n.is_finite() || n <= 0.0 {
            return Err(PchaseCardError::ValueInvalid {
                property: name.to_owned(),
                raw: raw.get().to_owned(),
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

    /// Resolve MSHR capacity for a typed deployment architecture. A typed
    /// declaration may consume only that architecture's explicit L1 defaults;
    /// card-valued counts remain card evidence and do not require a default.
    pub fn resolve_mshr_count_for_arch(&self, arch: ArchId) -> Result<MshrConfig, PchaseCardError> {
        self.require_arch(arch)?;
        let name = "mshr_outstanding_misses";
        let prop = self.property(name)?;
        match prop.kind.as_str() {
            "value" => Ok(MshrConfig {
                count: u32::try_from(Self::card_int(prop, name)?).map_err(|_| {
                    PchaseCardError::ValueInvalid {
                        property: name.to_owned(),
                        raw: prop
                            .value
                            .as_deref()
                            .map_or_else(|| "null".to_owned(), |raw| raw.get().to_owned()),
                        requirement: "must fit u32",
                    }
                })?,
                source: "card",
                sensitivity_note: None,
            }),
            "typed" => {
                let defaults = arch.l1_defaults().require("l1_defaults")?;
                Ok(MshrConfig {
                    count: defaults.mshr_count,
                    source: "documented_default",
                    sensitivity_note: Some(format!(
                        "the card types mshr_outstanding_misses geometry_unresolved (the \
                         single-thread probe bounds outstanding misses from below without a \
                         saturation knee), so the walk uses the configured {arch} default {}; \
                         report results at counts [16, 32, 64] for sensitivity",
                        defaults.mshr_count
                    )),
                })
            }
            other => Err(PchaseCardError::UnrecognizedKind {
                property: name.to_owned(),
                kind: other.to_owned(),
            }),
        }
    }

    fn resolve_replacement_policy_for_arch(
        &self,
        arch: ArchId,
    ) -> Result<GeometryPolicyConfig, PchaseCardError> {
        self.require_arch(arch)?;
        let name = "l1_replacement_policy";
        let prop = self.property(name)?;
        match prop.kind.as_str() {
            "value" => {
                let raw = Self::require_value(prop, name)?;
                let policy = serde_json::from_str::<String>(raw.get()).map_err(|_| {
                    PchaseCardError::ValueInvalid {
                        property: name.to_owned(),
                        raw: raw.get().to_owned(),
                        requirement: "must be a policy name string",
                    }
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
            "typed" => {
                let defaults = arch.l1_defaults().require("l1_defaults")?;
                let modeled = match defaults.replacement {
                    ArchL1Replacement::Lru => "lru",
                };
                Ok(GeometryPolicyConfig {
                    modeled,
                    source: "documented_approximation",
                    note: Some(format!(
                        "the card types l1_replacement_policy {:?}; {modeled} is the configured \
                         {arch} approximation pending card evidence",
                        prop.typed_reason.clone().unwrap_or_default()
                    )),
                })
            }
            other => Err(PchaseCardError::UnrecognizedKind {
                property: name.to_owned(),
                kind: other.to_owned(),
            }),
        }
    }

    fn resolve_set_mapping_for_arch(
        &self,
        arch: ArchId,
    ) -> Result<GeometryPolicyConfig, PchaseCardError> {
        self.require_arch(arch)?;
        let name = "l1_associativity_ways";
        let prop = self.property(name)?;
        match prop.kind.as_str() {
            "value" => Err(PchaseCardError::UnmodeledSetAssociativity {
                raw: prop
                    .value
                    .as_deref()
                    .map_or_else(|| "null".to_owned(), |raw| raw.get().to_owned()),
            }),
            "typed" => {
                let defaults = arch.l1_defaults().require("l1_defaults")?;
                let modeled = match defaults.set_mapping {
                    ArchL1SetMapping::FullyAssociative => "fully_associative",
                };
                Ok(GeometryPolicyConfig {
                    modeled,
                    source: "documented_approximation",
                    note: Some(format!(
                        "the card types l1_associativity_ways {:?}; the {modeled} walk is the \
                         configured {arch} approximation pending card evidence",
                        prop.typed_reason.clone().unwrap_or_default()
                    )),
                })
            }
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
            replacement: self.resolve_replacement_policy_for_arch(arch)?,
            set_mapping: self.resolve_set_mapping_for_arch(arch)?,
            mshr: self.resolve_mshr_count_for_arch(arch)?,
            latencies: self.resolve_latencies()?,
        })
    }
}

fn read_bounded_pchase_card(path: &Path, limit: u64) -> Result<Vec<u8>, PchaseCardError> {
    let file = std::fs::File::open(path).map_err(|source| PchaseCardError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut raw = Vec::new();
    file.take(limit.saturating_add(1))
        .read_to_end(&mut raw)
        .map_err(|source| PchaseCardError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let observed = u64::try_from(raw.len()).unwrap_or(u64::MAX);
    if observed > limit {
        return Err(PchaseCardError::ResourceLimit {
            path: path.to_path_buf(),
            observed,
            limit,
        });
    }
    Ok(raw)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn card_for_arch(json: &str, arch_token: &str) -> PchaseCard {
        let object = json.trim();
        let body = object
            .strip_prefix('{')
            .and_then(|value| value.strip_suffix('}'))
            .unwrap();
        let arch = serde_json::to_string(arch_token).unwrap();
        let separator = if body.trim().is_empty() { "" } else { "," };
        let with_arch = format!(r#"{{"arch":{arch}{separator}{body}}}"#);
        let raw: RawCard = serde_json::from_str(&with_arch).unwrap();
        PchaseCard {
            arch: ArchId::parse_deploy(&raw.arch).unwrap(),
            properties: raw.properties,
        }
    }

    fn card(json: &str) -> PchaseCard {
        card_for_arch(json, "sm_86")
    }

    #[test]
    fn bounded_pchase_card_read_rejects_oversized_input() {
        let path =
            std::env::temp_dir().join(format!("sim-mem-pchase-bound-{}.json", std::process::id()));
        std::fs::write(&path, b"12345").unwrap();
        assert!(matches!(
            read_bounded_pchase_card(&path, 4),
            Err(PchaseCardError::ResourceLimit {
                observed: 5,
                limit: 4,
                ..
            })
        ));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn duplicate_json_members_are_rejected_before_pchase_decode() {
        let raw = br#"{"arch":"sm_86","arch":"sm_86","properties":{}}"#;
        let error = match PchaseCard::parse(Path::new("sm_86_static_card.json"), raw) {
            Ok(_) => panic!("duplicate JSON member unexpectedly decoded"),
            Err(error) => error,
        };
        assert!(
            matches!(error, PchaseCardError::Parse { source, .. } if source
            .to_string()
            .contains("duplicate JSON member"))
        );
    }

    fn card_integer(raw: &str) -> Result<u64, PchaseCardError> {
        let c = card(&format!(
            r#"{{"properties": {{"integer": {{"kind": "value", "value": {raw}}}}}}}"#
        ));
        PchaseCard::card_int(c.property("integer").unwrap(), "integer")
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
            Err(PchaseCardError::Capability(CapabilityError::ResearchGap {
                arch: ArchId::Sm120,
                capability: "l1_defaults",
                reason,
            })) if !reason.is_empty()
        ));
        assert!(matches!(
            sm120.resolve_replacement_policy_for_arch(ArchId::Sm120),
            Err(PchaseCardError::Capability(CapabilityError::ResearchGap {
                arch: ArchId::Sm120,
                capability: "l1_defaults",
                ..
            }))
        ));
        assert!(matches!(
            sm120.resolve_set_mapping_for_arch(ArchId::Sm120),
            Err(PchaseCardError::Capability(CapabilityError::ResearchGap {
                arch: ArchId::Sm120,
                capability: "l1_defaults",
                ..
            }))
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
    fn sm120_valued_mshr_cannot_launder_typed_sm86_policy_defaults() {
        let sm120 = card_for_arch(
            r#"{"properties": {
                "l1_capacity_bytes": {"kind": "value", "value": 98304},
                "l1_hit_latency_cycles": {"kind": "value", "value": 33},
                "l2_hit_latency_cycles": {"kind": "value", "value": 243},
                "mshr_outstanding_misses": {"kind": "value", "value": 12},
                "l1_replacement_policy": {"kind": "typed", "typed_reason": "geometry_unresolved"},
                "l1_associativity_ways": {"kind": "typed", "typed_reason": "geometry_unresolved"}
            }}"#,
            "sm_120",
        );
        let mshr = sm120.resolve_mshr_count_for_arch(ArchId::Sm120).unwrap();
        assert_eq!(mshr.count, 12);
        assert_eq!(mshr.source, "card");
        assert!(matches!(
            sm120.resolve_walk_geometry_for_arch(ArchId::Sm120, None),
            Err(PchaseCardError::Capability(CapabilityError::ResearchGap {
                arch: ArchId::Sm120,
                capability: "l1_defaults",
                ..
            }))
        ));
    }

    #[test]
    fn sm120_valued_policy_branches_keep_card_semantics() {
        let replacement = card_for_arch(
            r#"{"properties": {
                "l1_replacement_policy": {"kind": "value", "value": "lru"}
            }}"#,
            "sm_120",
        )
        .resolve_replacement_policy_for_arch(ArchId::Sm120)
        .unwrap();
        assert_eq!(replacement.modeled, "lru");
        assert_eq!(replacement.source, "card");

        let associativity = card_for_arch(
            r#"{"properties": {
                "l1_associativity_ways": {"kind": "value", "value": 4}
            }}"#,
            "sm_120",
        );
        assert!(matches!(
            associativity.resolve_set_mapping_for_arch(ArchId::Sm120),
            Err(PchaseCardError::UnmodeledSetAssociativity { .. })
        ));
    }

    #[test]
    fn capability_error_taxonomy_is_preserved() {
        let error = PchaseCardError::from(CapabilityError::Unsupported {
            arch: ArchId::Sm120,
            capability: "synthetic_test_capability",
            reason: "synthetic unsupported state",
        });
        assert!(matches!(
            error,
            PchaseCardError::Capability(CapabilityError::Unsupported {
                arch: ArchId::Sm120,
                capability: "synthetic_test_capability",
                reason: "synthetic unsupported state",
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
            Err(PchaseCardError::ValueInvalid { .. })
        ));
    }

    #[test]
    fn card_integer_conversion_is_exact_across_json_spellings() {
        let accepted = [
            ("1", 1),
            ("128.0", 128),
            ("1.28e2", 128),
            ("100e-2", 1),
            ("9007199254740993.0", 9_007_199_254_740_993),
            ("18446744073709551615", u64::MAX),
            ("18446744073709551615.0", u64::MAX),
        ];
        for (raw, expected) in accepted {
            assert_eq!(card_integer(raw).unwrap(), expected, "{raw}");
        }

        for raw in [
            "0",
            "-1",
            "1.5",
            "1e-1",
            "1.00000000000000001",
            "18446744073709551616",
            "18446744073709551616.0",
            "1e999999999999999999999",
            "true",
            r#""1""#,
        ] {
            let error = card_integer(raw).unwrap_err();
            assert!(
                matches!(error, PchaseCardError::ValueInvalid { .. }),
                "{raw}: {error:?}"
            );
        }
    }

    #[test]
    fn rejected_integer_diagnostic_preserves_the_exact_lexeme() {
        let raw = "1.00000000000000001";
        assert!(matches!(
            card_integer(raw),
            Err(PchaseCardError::ValueInvalid {
                raw: rejected,
                requirement: "must be a positive integer representable as u64",
                ..
            }) if rejected == raw
        ));
    }

    #[test]
    fn valued_non_lru_policy_and_valued_ways_fail_closed() {
        let c = card(
            r#"{"properties": {"l1_replacement_policy":
            {"kind": "value", "value": "plru"}}}"#,
        );
        assert!(matches!(
            c.resolve_replacement_policy_for_arch(ArchId::Sm86),
            Err(PchaseCardError::UnmodeledReplacementPolicy { .. })
        ));
        let c2 = card(
            r#"{"properties": {"l1_associativity_ways":
            {"kind": "value", "value": 4}}}"#,
        );
        assert!(matches!(
            c2.resolve_set_mapping_for_arch(ArchId::Sm86),
            Err(PchaseCardError::UnmodeledSetAssociativity { .. })
        ));
    }

    #[test]
    fn out_of_range_latency_has_truthful_typed_diagnostic() {
        let c = card(r#"{"properties": {"latency": {"kind": "value", "value": 1e309}}}"#);
        assert!(matches!(
            PchaseCard::card_positive_float(c.property("latency").unwrap(), "latency"),
            Err(PchaseCardError::ValueInvalid {
                raw,
                requirement: "must be a JSON number representable as finite f64",
                ..
            }) if raw == "1e309"
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
