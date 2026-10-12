//! Geometry/latency resolvers over the committed P-chase static card
//! (`data/pchase/sm_86_static_card.json`) — a lockstep port of
//! `the native L1 replay model`'s card resolvers.
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
    unit: Option<String>,
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
        if prop
            .unit
            .as_deref()
            .is_some_and(|unit| unit != "cycles_per_dependent_load")
        {
            return Err(PchaseCardError::ValueInvalid {
                property: name.to_owned(),
                raw: prop.unit.clone().unwrap_or_default(),
                requirement: "must declare unit cycles_per_dependent_load",
            });
        }
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
        if prop.unit.as_deref().is_some_and(|unit| unit != "bytes") {
            return Err(PchaseCardError::ValueInvalid {
                property: name.clone(),
                raw: prop.unit.clone().unwrap_or_default(),
                requirement: "must declare unit bytes",
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
