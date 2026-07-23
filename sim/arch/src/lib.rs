//! Architecture identity and reviewed capability vocabulary shared by the
//! simulator crates.
//!
//! Deployment architecture, SKU, and compile-time build tokens are distinct
//! concepts.  A capability marked as a research gap is never converted into a
//! value here: consumers must call [`Capability::require`] and handle the
//! typed error.  This keeps the sm_86 reviewed hypotheses from becoming
//! accidental sm_120 defaults.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum ArchId {
    Sm86,
    Sm120,
}

impl ArchId {
    /// Parse the canonical deployment/label token.  Build tokens and aliases
    /// are intentionally rejected at this boundary; resolve those through
    /// [`BuildArch`] or the external card loader first.
    pub fn parse_deploy(token: &str) -> Result<Self, ArchError> {
        match token {
            "sm_86" => Ok(Self::Sm86),
            "sm_120" => Ok(Self::Sm120),
            "sm_120a" | "sm_120f" => Err(ArchError::BuildTokenUsedAsDeploy {
                token: token.to_owned(),
            }),
            other => Err(ArchError::UnknownDeployToken {
                token: other.to_owned(),
            }),
        }
    }

    pub fn from_numeric(version: u32) -> Result<Self, ArchError> {
        match version {
            86 => Ok(Self::Sm86),
            120 => Ok(Self::Sm120),
            other => Err(ArchError::UnknownNumeric { version: other }),
        }
    }

    pub const fn token(self) -> &'static str {
        match self {
            Self::Sm86 => "sm_86",
            Self::Sm120 => "sm_120",
        }
    }

    pub const fn numeric(self) -> u32 {
        match self {
            Self::Sm86 => 86,
            Self::Sm120 => 120,
        }
    }

    pub const fn compute_capability(self) -> (u32, u32) {
        match self {
            Self::Sm86 => (8, 6),
            Self::Sm120 => (12, 0),
        }
    }

    /// The bare-LDG default is a reviewed capability, not a generic rule.
    pub const fn bare_ldg_e_default(self) -> Capability<BareLdgEDefault> {
        match self {
            Self::Sm86 => Capability::available(Self::Sm86, BareLdgEDefault::CacheAtAllLevels),
            Self::Sm120 => Capability::research_gap(
                Self::Sm120,
                "sm_120 bare LDG.E cache policy is not measured",
            ),
        }
    }

    /// Packed control decoding is currently available only for the reviewed
    /// Ampere producer/consumer layout.  Blackwell remains a typed template.
    pub const fn control_layout(self) -> Capability<ControlLayout> {
        match self {
            Self::Sm86 => Capability::available(Self::Sm86, ControlLayout::AmpereHighWordV1),
            Self::Sm120 => Capability::unsupported(
                Self::Sm120,
                "no provenance-bound sm_120 control-bit layout exists",
            ),
        }
    }

    /// Reviewed L2 defaults are architecture-local hypotheses.  The partial
    /// sm_120 card must not inherit these values.
    pub const fn l2_defaults(self) -> Capability<ReviewedL2Defaults> {
        match self {
            Self::Sm86 => Capability::available(
                Self::Sm86,
                ReviewedL2Defaults {
                    associativity: 16,
                    replacement: L2Replacement::Lru,
                    slice_hash: L2SliceHash::LineMod,
                    line_bytes: 128,
                    sector_bytes: 32,
                },
            ),
            Self::Sm120 => Capability::research_gap(
                Self::Sm120,
                "sm_120 L2 associativity/replacement/hash defaults are unreviewed",
            ),
        }
    }

    /// Non-memory issue knobs are hypotheses reviewed only for sm_86.  The
    /// occupancy ceiling is deliberately included here so it cannot be
    /// mistaken for a cross-architecture invariant.
    pub const fn issue_hypotheses(self) -> Capability<IssueHypotheses> {
        match self {
            Self::Sm86 => Capability::available(
                Self::Sm86,
                IssueHypotheses {
                    non_memory_write_barrier_latency: 25,
                    read_barrier_latency: 8,
                    max_resident_warps_per_sm: 48,
                },
            ),
            Self::Sm120 => Capability::research_gap(
                Self::Sm120,
                "sm_120 scheduler/barrier/occupancy hypotheses are unreviewed",
            ),
        }
    }
}

impl fmt::Display for ArchId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.token())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum BuildArch {
    Sm86,
    Sm120,
    Sm120a,
    Sm120f,
}

impl BuildArch {
    pub fn parse(token: &str) -> Result<Self, ArchError> {
        match token {
            "sm_86" => Ok(Self::Sm86),
            "sm_120" => Ok(Self::Sm120),
            "sm_120a" => Ok(Self::Sm120a),
            "sm_120f" => Ok(Self::Sm120f),
            other => Err(ArchError::UnknownBuildToken {
                token: other.to_owned(),
            }),
        }
    }

    pub const fn token(self) -> &'static str {
        match self {
            Self::Sm86 => "sm_86",
            Self::Sm120 => "sm_120",
            Self::Sm120a => "sm_120a",
            Self::Sm120f => "sm_120f",
        }
    }

    pub const fn deploy_arch(self) -> ArchId {
        match self {
            Self::Sm86 => ArchId::Sm86,
            Self::Sm120 | Self::Sm120a | Self::Sm120f => ArchId::Sm120,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sku {
    Rtx3090Ti,
    Rtx5060Laptop,
}

impl Sku {
    pub fn parse(token: &str) -> Result<Self, ArchError> {
        match token {
            "rtx_3090_ti" => Ok(Self::Rtx3090Ti),
            "rtx_5060_laptop" => Ok(Self::Rtx5060Laptop),
            other => Err(ArchError::UnknownSku {
                token: other.to_owned(),
            }),
        }
    }

    pub const fn token(self) -> &'static str {
        match self {
            Self::Rtx3090Ti => "rtx_3090_ti",
            Self::Rtx5060Laptop => "rtx_5060_laptop",
        }
    }

    pub const fn arch(self) -> ArchId {
        match self {
            Self::Rtx3090Ti => ArchId::Sm86,
            Self::Rtx5060Laptop => ArchId::Sm120,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArchIdentity {
    deploy: ArchId,
    sku: Sku,
    build: BuildArch,
}

impl ArchIdentity {
    pub const fn deploy(self) -> ArchId {
        self.deploy
    }

    pub const fn sku(self) -> Sku {
        self.sku
    }

    pub const fn build(self) -> BuildArch {
        self.build
    }

    pub fn new(deploy: ArchId, sku: Sku, build: BuildArch) -> Result<Self, ArchError> {
        if sku.arch() != deploy {
            return Err(ArchError::SkuArchitectureMismatch {
                sku: sku.token().to_owned(),
                arch: deploy.token().to_owned(),
            });
        }
        if build.deploy_arch() != deploy {
            return Err(ArchError::BuildArchitectureMismatch {
                build: build.token().to_owned(),
                arch: deploy.token().to_owned(),
            });
        }
        Ok(Self { deploy, sku, build })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capability<T> {
    arch: ArchId,
    state: CapabilityState<T>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CapabilityState<T> {
    Available(T),
    ResearchGap { reason: &'static str },
    Unsupported { reason: &'static str },
}

impl<T> Capability<T> {
    pub(crate) const fn available(arch: ArchId, value: T) -> Self {
        Self {
            arch,
            state: CapabilityState::Available(value),
        }
    }

    pub(crate) const fn research_gap(arch: ArchId, reason: &'static str) -> Self {
        Self {
            arch,
            state: CapabilityState::ResearchGap { reason },
        }
    }

    pub(crate) const fn unsupported(arch: ArchId, reason: &'static str) -> Self {
        Self {
            arch,
            state: CapabilityState::Unsupported { reason },
        }
    }

    pub const fn arch(&self) -> ArchId {
        self.arch
    }

    /// Consume a capability without trusting a caller-supplied architecture.
    /// The source architecture is bound when the capability is resolved.
    pub fn require(self, name: &'static str) -> Result<T, CapabilityError> {
        match self.state {
            CapabilityState::Available(value) => Ok(value),
            CapabilityState::ResearchGap { reason } => Err(CapabilityError::ResearchGap {
                arch: self.arch,
                capability: name,
                reason,
            }),
            CapabilityState::Unsupported { reason } => Err(CapabilityError::Unsupported {
                arch: self.arch,
                capability: name,
                reason,
            }),
        }
    }
}

#[derive(Debug, thiserror::Error, Clone, Copy, PartialEq, Eq)]
pub enum ArchitectureProvenanceError {
    #[error("{component} has no bound architecture provenance")]
    Unbound { component: &'static str },
    #[error("{component} architecture mismatch: expected {expected}, got {got}")]
    Mismatch {
        component: &'static str,
        expected: ArchId,
        got: ArchId,
    },
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum ArchError {
    #[error("unknown deployment architecture token {token:?}")]
    UnknownDeployToken { token: String },
    #[error("build token {token:?} cannot be used as a deployment architecture")]
    BuildTokenUsedAsDeploy { token: String },
    #[error("unknown numeric architecture version {version}")]
    UnknownNumeric { version: u32 },
    #[error("unknown build architecture token {token:?}")]
    UnknownBuildToken { token: String },
    #[error("unknown SKU token {token:?}")]
    UnknownSku { token: String },
    #[error("SKU {sku:?} belongs to a different architecture than {arch:?}")]
    SkuArchitectureMismatch { sku: String, arch: String },
    #[error("build token {build:?} belongs to a different architecture than {arch:?}")]
    BuildArchitectureMismatch { build: String, arch: String },
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum CapabilityError {
    #[error("{arch} capability {capability} is a research gap: {reason}")]
    ResearchGap {
        arch: ArchId,
        capability: &'static str,
        reason: &'static str,
    },
    #[error("{arch} capability {capability} is unsupported: {reason}")]
    Unsupported {
        arch: ArchId,
        capability: &'static str,
        reason: &'static str,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BareLdgEDefault {
    CacheAtAllLevels,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlLayout {
    AmpereHighWordV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L2Replacement {
    Lru,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L2SliceHash {
    LineMod,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReviewedL2Defaults {
    pub associativity: u32,
    pub replacement: L2Replacement,
    pub slice_hash: L2SliceHash,
    pub line_bytes: u32,
    pub sector_bytes: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IssueHypotheses {
    pub non_memory_write_barrier_latency: u64,
    pub read_barrier_latency: u64,
    pub max_resident_warps_per_sm: u32,
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn deploy_and_build_tokens_stay_distinct() {
        assert_eq!(ArchId::parse_deploy("sm_86").unwrap(), ArchId::Sm86);
        assert!(matches!(
            ArchId::parse_deploy("sm_120a"),
            Err(ArchError::BuildTokenUsedAsDeploy { .. })
        ));
        assert_eq!(
            BuildArch::parse("sm_120a").unwrap().deploy_arch(),
            ArchId::Sm120
        );
        assert!(matches!(
            BuildArch::parse("sm_999"),
            Err(ArchError::UnknownBuildToken { .. })
        ));
    }

    #[test]
    fn sku_and_architecture_mismatch_is_typed() {
        assert!(matches!(
            ArchIdentity::new(ArchId::Sm86, Sku::Rtx5060Laptop, BuildArch::Sm86),
            Err(ArchError::SkuArchitectureMismatch { .. })
        ));
        assert!(ArchIdentity::new(ArchId::Sm120, Sku::Rtx5060Laptop, BuildArch::Sm120f).is_ok());
    }

    #[test]
    fn sm120_capabilities_never_inherit_sm86() {
        assert!(
            ArchId::Sm86
                .bare_ldg_e_default()
                .require("bare_ldg_e_default")
                .is_ok()
        );
        assert!(matches!(
            ArchId::Sm120
                .bare_ldg_e_default()
                .require("bare_ldg_e_default"),
            Err(CapabilityError::ResearchGap { .. })
        ));
        assert!(matches!(
            ArchId::Sm120.control_layout().require("control_layout"),
            Err(CapabilityError::Unsupported { .. })
        ));
        assert!(matches!(
            ArchId::Sm120.l2_defaults().require("l2_defaults"),
            Err(CapabilityError::ResearchGap { .. })
        ));
    }
}
