//! Architecture identity and capability vocabulary shared by the
//! simulator crates.
//!
//! Deployment architecture, SKU, and compile-time build tokens are distinct
//! concepts.  A capability marked as a research gap is never converted into a
//! value here: consumers must call [`Capability::require`] and handle the
//! typed error. This keeps the sm_86 hypotheses from becoming
//! accidental sm_120 defaults.

pub mod card;

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

    /// The bare-LDG default is architecture-specific, not a generic rule.
    pub const fn bare_ldg_e_default(self) -> Capability<BareLdgEDefault> {
        match self {
            Self::Sm86 => Capability::available(Self::Sm86, BareLdgEDefault::CacheAtAllLevels),
            Self::Sm120 => Capability::research_gap(
                Self::Sm120,
                "sm_120 bare LDG.E cache policy is not measured",
            ),
        }
    }

    /// Packed control decoding is currently available only for the implemented
    /// Ampere producer/consumer layout.  Blackwell remains a typed template.
    pub const fn control_layout(self) -> Capability<ControlLayout> {
        match self {
            Self::Sm86 => Capability::available(Self::Sm86, ControlLayout::AmpereHighWordV1),
            Self::Sm120 => Capability::unsupported(
                Self::Sm120,
                "no implemented sm_120 control-bit layout exists",
            ),
        }
    }

    /// L1 approximation knobs are architecture-local capabilities. A typed
    /// card property may resolve through this capability only when the
    /// deployment architecture defines the approximation.
    pub const fn l1_defaults(self) -> Capability<L1Defaults> {
        match self {
            Self::Sm86 => Capability::available(
                Self::Sm86,
                L1Defaults {
                    mshr_count: 32,
                    replacement: L1Replacement::Lru,
                    set_mapping: L1SetMapping::FullyAssociative,
                },
            ),
            Self::Sm120 => Capability::research_gap(
                Self::Sm120,
                "sm_120 L1 MSHR/replacement/set-mapping defaults are not modeled",
            ),
        }
    }

    /// L2 defaults are architecture-local hypotheses. The partial
    /// sm_120 card must not inherit these values.
    pub const fn l2_defaults(self) -> Capability<L2Defaults> {
        match self {
            Self::Sm86 => Capability::available(
                Self::Sm86,
                L2Defaults {
                    associativity: 16,
                    replacement: L2Replacement::Lru,
                    slice_hash: L2SliceHash::LineMod,
                    line_bytes: 128,
                    sector_bytes: 32,
                },
            ),
            Self::Sm120 => Capability::research_gap(
                Self::Sm120,
                "sm_120 L2 associativity/replacement/hash defaults are not modeled",
            ),
        }
    }

    /// Behavior of a read that reaches an L2 sector while that allocation's
    /// lower-level fetch is still pending. This controls fetch/response merging,
    /// not a finite MSHR depth.
    pub const fn l2_pending_read_semantics(self) -> Capability<L2PendingReadSemantics> {
        match self {
            Self::Sm86 => {
                Capability::available(Self::Sm86, L2PendingReadSemantics::AllocationSectorMergeV1)
            }
            Self::Sm120 => Capability::research_gap(
                Self::Sm120,
                "sm_120 L2 pending-read merge semantics are not modeled",
            ),
        }
    }

    /// Non-memory issue knobs are currently defined only for sm_86. The
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
                "sm_120 scheduler/barrier/occupancy hypotheses are not modeled",
            ),
        }
    }

    /// Tensor-pipe initiation intervals for the exact matrix instructions
    /// admitted by the sm_86 control-sidecar corpus. These are GA10x-local:
    /// one Tensor Core belongs to each SM partition, and the instruction work
    /// divided by NVIDIA's per-core throughput gives 16 cycles for dense
    /// HMMA.16816 and sparse HMMA.SP.16832, and 8 cycles for IMMA.16816.S8.
    pub const fn tensor_issue_intervals(self) -> Capability<TensorIssueIntervals> {
        match self {
            Self::Sm86 => Capability::available(
                Self::Sm86,
                TensorIssueIntervals {
                    hmma_16816_f16_bf16: 16,
                    hmma_sparse_16832_f16_bf16: 16,
                    imma_16816_s8: 8,
                },
            ),
            Self::Sm120 => Capability::research_gap(
                Self::Sm120,
                "sm_120 tensor instruction shapes and per-SMSP throughput are not modeled",
            ),
        }
    }

    /// CTA-barrier issue behavior identified independently on a one-CTA,
    /// memory-free sweep. Ampere admits barrier arrivals through one SM-shared
    /// server at one warp every two cycles. The separate generation re-arm
    /// remains zero until a SASS-adjacent consecutive-barrier construction
    /// distinguishes it from intervening compiler-scheduled control work.
    pub const fn cta_barrier_issue(self) -> Capability<CtaBarrierIssue> {
        match self {
            Self::Sm86 => Capability::available(
                Self::Sm86,
                CtaBarrierIssue {
                    warp_initiation_interval_cycles: 2,
                    generation_rearm_cycles: 0,
                },
            ),
            Self::Sm120 => Capability::research_gap(
                Self::Sm120,
                "sm_120 CTA-barrier issue scope, throughput, and generation re-arm are not measured",
            ),
        }
    }

    /// Branch timing identified by two independent constructions. The exact
    /// forward-branch length sweep retains its ten-cycle next-issue interval.
    /// A U1/U2/U4/U8 backward-branch factorial additionally exposes three
    /// branch-resolving warp cycles per uniform target directly in NCU; that
    /// target-resolution state is shorter than the compiler's five-cycle BRA
    /// control stall and therefore does not by itself invent an extra issue
    /// delay.
    pub const fn branch_resolution(self) -> Capability<BranchResolution> {
        match self {
            Self::Sm86 => Capability::available(
                Self::Sm86,
                BranchResolution {
                    next_issue_interval_cycles: 10,
                    uniform_backward_target_resolution_cycles: 3,
                },
            ),
            Self::Sm120 => Capability::research_gap(
                Self::Sm120,
                "sm_120 branch resolution and next-issue interval are not measured",
            ),
        }
    }

    /// Pairwise cross-warp/CTA global-atomic contention topology. Two aligned
    /// 32/64-bit words contend exactly when their byte addresses occupy the
    /// same aligned region. This is a topology fact only: the distributed
    /// many-region service network and its timing remain unidentified.
    pub const fn atomic_topology(self) -> Capability<AtomicTopology> {
        match self {
            Self::Sm86 => Capability::available(
                Self::Sm86,
                AtomicTopology {
                    cross_warp_contention_region_bytes: 256,
                },
            ),
            Self::Sm120 => Capability::research_gap(
                Self::Sm120,
                "sm_120 cross-warp atomic contention region and distributed service are not measured",
            ),
        }
    }

    /// Bank-local arbitration between exact fixed-latency results and scalar
    /// L1TEX load results. The completion and visibility offsets are separate:
    /// the compiler-visible dependence frontier trails the physical load
    /// writeback. A same-cycle HADD2 completion has priority and defers the load
    /// result by one cycle.
    pub const fn fixed_l1tex_writeback_priority(self) -> Capability<FixedL1TexWritebackPriority> {
        match self {
            Self::Sm86 => Capability::available(
                Self::Sm86,
                FixedL1TexWritebackPriority {
                    hadd2_writeback_cycles: 6,
                    l1tex_writeback_to_visibility_cycles: 5,
                    variable_deferral_cycles: 1,
                },
            ),
            Self::Sm120 => Capability::research_gap(
                Self::Sm120,
                "sm_120 fixed/L1TEX result priority and completion offsets are not measured",
            ),
        }
    }

    /// Static SASS protocol used to close and wait on non-bulk asynchronous
    /// global-to-shared copy groups. This is a semantic capability rather than
    /// a latency source: completion timing still comes from the memory model.
    pub const fn async_copy_group_semantics(self) -> Capability<AsyncCopyGroupSemantics> {
        match self {
            Self::Sm86 => {
                Capability::available(Self::Sm86, AsyncCopyGroupSemantics::AmpereLdgstsDepbarV1)
            }
            Self::Sm120 => Capability::research_gap(
                Self::Sm120,
                "sm_120 asynchronous-copy SASS group semantics are not modeled",
            ),
        }
    }

    /// L1 route selected by the cache modifier on Ampere non-bulk asynchronous
    /// copies. `LDGSTS.E` is the cache-at-all-levels lowering, while an explicit
    /// `.BYPASS` component selects the L2-only route. The destination side of
    /// the copy is deliberately outside this source-cache capability.
    pub const fn ldgsts_l1_cache_semantics(self) -> Capability<LdgstsL1CacheSemantics> {
        match self {
            Self::Sm86 => Capability::available(
                Self::Sm86,
                LdgstsL1CacheSemantics::AmpereOptionalBypassModifierV1,
            ),
            Self::Sm120 => Capability::research_gap(
                Self::Sm120,
                "sm_120 LDGSTS/TMA source-cache modifier semantics are not modeled",
            ),
        }
    }

    /// Conservative aggregate line-fill admission floor for cacheable non-bulk
    /// asynchronous copies. This is separate from [`Self::l1_defaults`]: the
    /// ordinary-load MSHR geometry remains unresolved, while focused LDGSTS
    /// probes directly bound cacheable async-source concurrency.
    pub const fn ldgsts_l1_fill_admission(self) -> Capability<LdgstsL1FillAdmission> {
        match self {
            Self::Sm86 => Capability::available(
                Self::Sm86,
                LdgstsL1FillAdmission {
                    minimum_concurrent_line_fills: 248,
                },
            ),
            Self::Sm120 => Capability::research_gap(
                Self::Sm120,
                "sm_120 LDGSTS/TMA L1 fill-admission semantics are not modeled",
            ),
        }
    }

    /// Effective-payload completion throughput for non-bulk asynchronous copies.
    /// This is distinct from fill admission and lower-level fetch coalescing: every
    /// issued copy still has to deliver its active bytes to the shared destination.
    pub const fn ldgsts_completion_throughput(self) -> Capability<LdgstsCompletionThroughput> {
        match self {
            Self::Sm86 => Capability::available(
                Self::Sm86,
                LdgstsCompletionThroughput {
                    cache_all_payload_bytes_per_cycle: 64,
                    bypass_payload_bytes_per_cycle: 32,
                },
            ),
            Self::Sm120 => Capability::research_gap(
                Self::Sm120,
                "sm_120 LDGSTS/TMA payload-completion throughput is not modeled",
            ),
        }
    }

    /// Destination-side projection and local completion timing for cache-all
    /// non-bulk asynchronous copies. Source transfers are grouped into flexible
    /// 128-byte cohorts before shared words are projected onto 32 four-byte
    /// banks. Explicit `.BYPASS` copies are intentionally outside this
    /// capability until their distinct source/destination coupling is resolved.
    pub const fn ldgsts_destination_wavefront_timing(
        self,
    ) -> Capability<LdgstsDestinationWavefrontTiming> {
        match self {
            Self::Sm86 => Capability::available(
                Self::Sm86,
                LdgstsDestinationWavefrontTiming {
                    projection:
                        LdgstsDestinationWavefrontProjection::SourceCohorts128BytesBanks32Words4Bytes,
                    cycles_per_wavefront: 2,
                    aggregate_service_wavefronts: 2,
                    aggregate_service_cycles: 3,
                },
            ),
            Self::Sm120 => Capability::research_gap(
                Self::Sm120,
                "sm_120 LDGSTS/TMA destination banking and completion timing are not modeled",
            ),
        }
    }

    /// Bank projection and conflict-dependent completion timing for explicit
    /// shared-memory LDS/STS operands. The base dependent latency remains the
    /// architecture card's `shared_latency_cycles`; this capability supplies
    /// only the additional latency of each serialized bank wavefront.
    pub const fn shared_memory_wavefront_timing(self) -> Capability<SharedMemoryWavefrontTiming> {
        match self {
            Self::Sm86 => Capability::available(
                Self::Sm86,
                SharedMemoryWavefrontTiming {
                    projection: SharedBankProjection::Banks32Words4Bytes,
                    additional_wavefront_latency_cycles: 2,
                },
            ),
            Self::Sm120 => Capability::research_gap(
                Self::Sm120,
                "sm_120 shared-memory banking and wavefront timing are not modeled",
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

/// Public capability state without exposing or fabricating an available value.
/// Consumers can preserve research-gap versus unsupported taxonomy before they
/// decide whether to call [`Capability::require`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityStatus {
    Available,
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

    pub const fn status(&self) -> CapabilityStatus {
        match &self.state {
            CapabilityState::Available(_) => CapabilityStatus::Available,
            CapabilityState::ResearchGap { reason } => {
                CapabilityStatus::ResearchGap { reason: *reason }
            }
            CapabilityState::Unsupported { reason } => {
                CapabilityStatus::Unsupported { reason: *reason }
            }
        }
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
pub enum BareLdgECardDeclarationError {
    #[error("unknown bare-LDG card declaration token {token:?}")]
    UnknownToken { token: String },
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

/// Closed simulator-supported card-side declaration subset corresponding to
/// the runtime bare-LDG capability. `TypedUnavailable` carries no value and can
/// agree only with a research-gap or unsupported capability state. The card
/// schema's `bypass` token remains unavailable until the runtime implements it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BareLdgECardDeclaration {
    CacheAtAllLevels,
    TypedUnavailable,
}

impl BareLdgECardDeclaration {
    pub fn parse(token: &str) -> Result<Self, BareLdgECardDeclarationError> {
        match token {
            "ca" => Ok(Self::CacheAtAllLevels),
            "unknown_typed_unavailable" => Ok(Self::TypedUnavailable),
            other => Err(BareLdgECardDeclarationError::UnknownToken {
                token: other.to_owned(),
            }),
        }
    }

    pub const fn token(self) -> &'static str {
        match self {
            Self::CacheAtAllLevels => "ca",
            Self::TypedUnavailable => "unknown_typed_unavailable",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlLayout {
    AmpereHighWordV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L1Replacement {
    Lru,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L1SetMapping {
    FullyAssociative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L1Defaults {
    pub mshr_count: u32,
    pub replacement: L1Replacement,
    pub set_mapping: L1SetMapping,
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
pub struct L2Defaults {
    pub associativity: u32,
    pub replacement: L2Replacement,
    pub slice_hash: L2SliceHash,
    pub line_bytes: u32,
    pub sector_bytes: u32,
}

/// Architecture-local handling of TEX reads that find matching L2 data already
/// in flight from a lower level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L2PendingReadSemantics {
    /// Keep the lookup as a miss, but issue no duplicate DRAM fetch and make the
    /// new consumer wait on the allocation-generation/sector's original ready
    /// time. A repeated-source `.BYPASS` probe directly established this on
    /// `sm_86`: L2 requests scaled 32× while DRAM sectors stayed constant.
    AllocationSectorMergeV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IssueHypotheses {
    pub non_memory_write_barrier_latency: u64,
    pub read_barrier_latency: u64,
    pub max_resident_warps_per_sm: u32,
}

/// Per-SMSP initiation intervals for the sm_86 tensor instructions whose
/// throughput is architecture-backed and whose SASS forms are classified by
/// `sim-sm`. A separate named field keeps an unsupported matrix shape from
/// silently inheriting the rate of a superficially similar instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorIssueIntervals {
    pub hmma_16816_f16_bf16: u64,
    pub hmma_sparse_16832_f16_bf16: u64,
    pub imma_16816_s8: u64,
}

/// Architecture-local CTA-barrier issue resource. The warp initiation
/// interval belongs to one server shared by the SM, not one server per SMSP.
/// `generation_rearm_cycles` gates only a later barrier generation of the same
/// CTA; it is not a post-rendezvous delay on ordinary instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CtaBarrierIssue {
    pub warp_initiation_interval_cycles: u64,
    pub generation_rearm_cycles: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BranchResolution {
    pub next_issue_interval_cycles: u64,
    pub uniform_backward_target_resolution_cycles: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtomicTopology {
    pub cross_warp_contention_region_bytes: u32,
}

impl AtomicTopology {
    pub fn contention_region(self, byte_address: i64) -> i64 {
        byte_address.div_euclid(i64::from(self.cross_warp_contention_region_bytes))
    }

    pub fn same_contention_region(self, left: i64, right: i64) -> bool {
        self.contention_region(left) == self.contention_region(right)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedL1TexWritebackPriority {
    pub hadd2_writeback_cycles: u64,
    pub l1tex_writeback_to_visibility_cycles: u64,
    pub variable_deferral_cycles: u64,
}

/// Architecture-local lowering used for PTX non-bulk `cp.async` groups.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsyncCopyGroupSemantics {
    /// `LDGSTS` source copies, `LDGDEPBAR` commits, and
    /// `DEPBAR.LE SB0, N` waits on Ampere.
    AmpereLdgstsDepbarV1,
}

/// Architecture-local source-cache interpretation of Ampere `LDGSTS` SASS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LdgstsL1CacheSemantics {
    /// `LDGSTS.E` caches at all levels and an explicit dotted `BYPASS`
    /// component bypasses L1. Characterized independently with repeated-source
    /// `cp.async.ca`/`cp.async.cg` probes on sm_86.
    AmpereOptionalBypassModifierV1,
}

/// Independently bounded cacheable-LDGSTS fill admission. This is a conservative
/// floor rather than a claim about the physical queue's exact name or geometry:
/// full-warp `.E.128` batches showed no route-specific LG throttle through 62
/// copies (248 distinct 128-byte lines), while the 63rd copy exposed pressure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LdgstsL1FillAdmission {
    pub minimum_concurrent_line_fills: u32,
}

/// Route-local lower bound on effective async-copy payload delivery. Static,
/// compile-time-unrolled `sm_86` probes held the source tile constant within one
/// committed group and varied active bytes independently of copy count. Their
/// large-group slopes converge to 64 B/cycle for cache-all `LDGSTS.E` and
/// 32 B/cycle for explicit `.BYPASS`; doubling source sectors without changing
/// payload bytes did not double the slope. These rates do not claim a finite
/// queue depth or measured cross-warp arbitration policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LdgstsCompletionThroughput {
    pub cache_all_payload_bytes_per_cycle: u64,
    pub bypass_payload_bytes_per_cycle: u64,
}

/// Trace-side destination projection admitted for cache-all Ampere `LDGSTS`.
/// The cohort qualification is part of the projection identity: applying an
/// aggregate shared-bank projection, or grouping by absolute cache-line
/// alignment, produces different answers for valid lane/source pairings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LdgstsDestinationWavefrontProjection {
    SourceCohorts128BytesBanks32Words4Bytes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LdgstsDestinationWavefrontTiming {
    pub projection: LdgstsDestinationWavefrontProjection,
    /// Dependent completion floor for one warp's committed group.
    pub cycles_per_wavefront: u64,
    /// Rational sustained destination service shared by the resident warps of
    /// one SM: `aggregate_service_wavefronts` complete every
    /// `aggregate_service_cycles` cycles.
    pub aggregate_service_wavefronts: u64,
    pub aggregate_service_cycles: u64,
}

/// Compact trace-side bank projection understood by an architecture timing
/// capability. Naming the geometry prevents a future architecture from
/// interpreting an Ampere projection as its own bank layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedBankProjection {
    Banks32Words4Bytes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedMemoryWavefrontTiming {
    pub projection: SharedBankProjection,
    pub additional_wavefront_latency_cycles: u64,
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
            ArchId::Sm120.l1_defaults().require("l1_defaults"),
            Err(CapabilityError::ResearchGap {
                arch: ArchId::Sm120,
                capability: "l1_defaults",
                reason,
            }) if !reason.is_empty()
        ));
        assert!(matches!(
            ArchId::Sm120.l2_defaults().require("l2_defaults"),
            Err(CapabilityError::ResearchGap { .. })
        ));
        assert!(matches!(
            ArchId::Sm120
                .l2_pending_read_semantics()
                .require("l2_pending_read_semantics"),
            Err(CapabilityError::ResearchGap {
                arch: ArchId::Sm120,
                capability: "l2_pending_read_semantics",
                reason,
            }) if !reason.is_empty()
        ));
        assert!(matches!(
            ArchId::Sm120.issue_hypotheses().require("issue_hypotheses"),
            Err(CapabilityError::ResearchGap { .. })
        ));
        assert!(matches!(
            ArchId::Sm120
                .tensor_issue_intervals()
                .require("tensor_issue_intervals"),
            Err(CapabilityError::ResearchGap {
                arch: ArchId::Sm120,
                capability: "tensor_issue_intervals",
                reason,
            }) if !reason.is_empty()
        ));
        assert!(matches!(
            ArchId::Sm120
                .async_copy_group_semantics()
                .require("async_copy_group_semantics"),
            Err(CapabilityError::ResearchGap {
                arch: ArchId::Sm120,
                capability: "async_copy_group_semantics",
                reason,
            }) if !reason.is_empty()
        ));
        assert!(matches!(
            ArchId::Sm120
                .ldgsts_l1_cache_semantics()
                .require("ldgsts_l1_cache_semantics"),
            Err(CapabilityError::ResearchGap {
                arch: ArchId::Sm120,
                capability: "ldgsts_l1_cache_semantics",
                reason,
            }) if !reason.is_empty()
        ));
        assert!(matches!(
            ArchId::Sm120
                .ldgsts_l1_fill_admission()
                .require("ldgsts_l1_fill_admission"),
            Err(CapabilityError::ResearchGap {
                arch: ArchId::Sm120,
                capability: "ldgsts_l1_fill_admission",
                reason,
            }) if !reason.is_empty()
        ));
        assert!(matches!(
            ArchId::Sm120
                .ldgsts_completion_throughput()
                .require("ldgsts_completion_throughput"),
            Err(CapabilityError::ResearchGap {
                arch: ArchId::Sm120,
                capability: "ldgsts_completion_throughput",
                reason,
            }) if !reason.is_empty()
        ));
        assert!(matches!(
            ArchId::Sm120
                .ldgsts_destination_wavefront_timing()
                .require("ldgsts_destination_wavefront_timing"),
            Err(CapabilityError::ResearchGap {
                arch: ArchId::Sm120,
                capability: "ldgsts_destination_wavefront_timing",
                reason,
            }) if !reason.is_empty()
        ));
        assert!(matches!(
            ArchId::Sm120
                .shared_memory_wavefront_timing()
                .require("shared_memory_wavefront_timing"),
            Err(CapabilityError::ResearchGap {
                arch: ArchId::Sm120,
                capability: "shared_memory_wavefront_timing",
                reason,
            }) if !reason.is_empty()
        ));
    }

    #[test]
    fn sm86_l2_pending_reads_merge_by_allocation_sector() {
        assert_eq!(
            ArchId::Sm86
                .l2_pending_read_semantics()
                .require("l2_pending_read_semantics")
                .unwrap(),
            L2PendingReadSemantics::AllocationSectorMergeV1
        );
    }

    #[test]
    fn sm86_tensor_issue_intervals_are_architecture_local() {
        assert_eq!(
            ArchId::Sm86
                .tensor_issue_intervals()
                .require("tensor_issue_intervals")
                .unwrap(),
            TensorIssueIntervals {
                hmma_16816_f16_bf16: 16,
                hmma_sparse_16832_f16_bf16: 16,
                imma_16816_s8: 8,
            }
        );
    }

    #[test]
    fn sm86_cta_barrier_issue_is_sm_shared_and_architecture_local() {
        assert_eq!(
            ArchId::Sm86
                .cta_barrier_issue()
                .require("cta_barrier_issue")
                .unwrap(),
            CtaBarrierIssue {
                warp_initiation_interval_cycles: 2,
                generation_rearm_cycles: 0,
            }
        );
        assert!(matches!(
            ArchId::Sm120
                .cta_barrier_issue()
                .require("cta_barrier_issue"),
            Err(CapabilityError::ResearchGap {
                arch: ArchId::Sm120,
                capability: "cta_barrier_issue",
                reason,
            }) if !reason.is_empty()
        ));
    }

    #[test]
    fn sm86_branch_resolution_is_architecture_local() {
        assert_eq!(
            ArchId::Sm86
                .branch_resolution()
                .require("branch_resolution")
                .unwrap(),
            BranchResolution {
                next_issue_interval_cycles: 10,
                uniform_backward_target_resolution_cycles: 3,
            }
        );
        assert!(matches!(
            ArchId::Sm120
                .branch_resolution()
                .require("branch_resolution"),
            Err(CapabilityError::ResearchGap {
                arch: ArchId::Sm120,
                capability: "branch_resolution",
                reason,
            }) if !reason.is_empty()
        ));
    }

    #[test]
    fn sm86_atomic_topology_is_architecture_local_and_timing_free() {
        assert_eq!(
            ArchId::Sm86
                .atomic_topology()
                .require("atomic_topology")
                .unwrap(),
            AtomicTopology {
                cross_warp_contention_region_bytes: 256,
            }
        );
        assert!(matches!(
            ArchId::Sm120
                .atomic_topology()
                .require("atomic_topology"),
            Err(CapabilityError::ResearchGap {
                arch: ArchId::Sm120,
                capability: "atomic_topology",
                reason,
            }) if !reason.is_empty()
        ));
        let topology = ArchId::Sm86
            .atomic_topology()
            .require("atomic_topology")
            .unwrap();
        assert!(topology.same_contention_region(0, 252));
        assert!(!topology.same_contention_region(0, 256));
        assert!(topology.same_contention_region(128, 252));
        assert!(!topology.same_contention_region(128, 256));
        assert!(topology.same_contention_region(-256, -1));
        assert!(!topology.same_contention_region(-1, 0));
    }

    #[test]
    fn sm86_fixed_l1tex_writeback_priority_is_architecture_local() {
        assert_eq!(
            ArchId::Sm86
                .fixed_l1tex_writeback_priority()
                .require("fixed_l1tex_writeback_priority")
                .unwrap(),
            FixedL1TexWritebackPriority {
                hadd2_writeback_cycles: 6,
                l1tex_writeback_to_visibility_cycles: 5,
                variable_deferral_cycles: 1,
            }
        );
        assert!(matches!(
            ArchId::Sm120
                .fixed_l1tex_writeback_priority()
                .require("fixed_l1tex_writeback_priority"),
            Err(CapabilityError::ResearchGap {
                arch: ArchId::Sm120,
                capability: "fixed_l1tex_writeback_priority",
                reason,
            }) if !reason.is_empty()
        ));
    }

    #[test]
    fn sm86_async_copy_group_semantics_are_architecture_local() {
        assert_eq!(
            ArchId::Sm86
                .async_copy_group_semantics()
                .require("async_copy_group_semantics")
                .unwrap(),
            AsyncCopyGroupSemantics::AmpereLdgstsDepbarV1
        );
    }

    #[test]
    fn sm86_ldgsts_l1_cache_semantics_are_architecture_local() {
        assert_eq!(
            ArchId::Sm86
                .ldgsts_l1_cache_semantics()
                .require("ldgsts_l1_cache_semantics")
                .unwrap(),
            LdgstsL1CacheSemantics::AmpereOptionalBypassModifierV1
        );
        assert_eq!(
            ArchId::Sm86
                .ldgsts_l1_fill_admission()
                .require("ldgsts_l1_fill_admission")
                .unwrap(),
            LdgstsL1FillAdmission {
                minimum_concurrent_line_fills: 248,
            }
        );
    }

    #[test]
    fn sm86_ldgsts_completion_throughput_is_architecture_local() {
        assert_eq!(
            ArchId::Sm86
                .ldgsts_completion_throughput()
                .require("ldgsts_completion_throughput")
                .unwrap(),
            LdgstsCompletionThroughput {
                cache_all_payload_bytes_per_cycle: 64,
                bypass_payload_bytes_per_cycle: 32,
            }
        );
    }

    #[test]
    fn sm86_ldgsts_destination_wavefront_timing_is_cache_all_local() {
        assert_eq!(
            ArchId::Sm86
                .ldgsts_destination_wavefront_timing()
                .require("ldgsts_destination_wavefront_timing")
                .unwrap(),
            LdgstsDestinationWavefrontTiming {
                projection:
                    LdgstsDestinationWavefrontProjection::SourceCohorts128BytesBanks32Words4Bytes,
                cycles_per_wavefront: 2,
                aggregate_service_wavefronts: 2,
                aggregate_service_cycles: 3,
            }
        );
    }

    #[test]
    fn sm86_shared_memory_wavefront_timing_is_architecture_local() {
        assert_eq!(
            ArchId::Sm86
                .shared_memory_wavefront_timing()
                .require("shared_memory_wavefront_timing")
                .unwrap(),
            SharedMemoryWavefrontTiming {
                projection: SharedBankProjection::Banks32Words4Bytes,
                additional_wavefront_latency_cycles: 2,
            }
        );
    }

    #[test]
    fn capability_status_preserves_taxonomy_without_exposing_values() {
        assert_eq!(
            ArchId::Sm86.l1_defaults().status(),
            CapabilityStatus::Available
        );
        assert_eq!(
            ArchId::Sm86.l1_defaults().require("l1_defaults").unwrap(),
            L1Defaults {
                mshr_count: 32,
                replacement: L1Replacement::Lru,
                set_mapping: L1SetMapping::FullyAssociative,
            }
        );
        assert!(matches!(
            ArchId::Sm120.l1_defaults().status(),
            CapabilityStatus::ResearchGap { reason } if !reason.is_empty()
        ));
        assert!(matches!(
            ArchId::Sm120.control_layout().status(),
            CapabilityStatus::Unsupported { reason } if !reason.is_empty()
        ));
    }

    #[test]
    fn bare_ldg_card_declarations_are_closed() {
        for (token, expected) in [
            ("ca", BareLdgECardDeclaration::CacheAtAllLevels),
            (
                "unknown_typed_unavailable",
                BareLdgECardDeclaration::TypedUnavailable,
            ),
        ] {
            let parsed = BareLdgECardDeclaration::parse(token).unwrap();
            assert_eq!(parsed, expected);
            assert_eq!(parsed.token(), token);
        }
        // The active card-schema vocabulary retains this future-facing
        // spelling, while every current architecture branch and runtime
        // capability reject it until an implementation lands in lockstep.
        assert!(matches!(
            BareLdgECardDeclaration::parse("bypass"),
            Err(BareLdgECardDeclarationError::UnknownToken { .. })
        ));
        assert!(matches!(
            BareLdgECardDeclaration::parse("cg"),
            Err(BareLdgECardDeclarationError::UnknownToken { .. })
        ));
    }
}
