//! Architecture identity and capability vocabulary shared by the
//! simulator crates.
//!
//! Deployment architecture, SKU, and compile-time build tokens are distinct
//! concepts.  A capability marked as a research gap is never converted into a
//! value here: consumers must call [`Capability::require`] and handle the
//! typed error. This keeps the sm_86 hypotheses from becoming
//! accidental sm_120 defaults.

pub mod card;
pub mod its;
pub mod json;
pub mod ptx_subset;
pub mod simt;
pub mod simt_pipeline;
pub mod spec;

pub use json::strict_json_from_slice;

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

    /// L1TEX write-lookup request window used by the SM86 observer. The
    /// physical read-state model remains the conservative fully-associative
    /// walk selected by `l1_defaults`; this separate window captures the
    /// sector/tag allocation stream exposed by Ampere's write-allocate,
    /// sub-sector-write path. Its two tag slots per set index are an
    /// architecture-level service hypothesis, not a per-workload correction.
    pub const fn l1_write_lookup_window(self) -> Capability<L1WriteLookupWindow> {
        match self {
            Self::Sm86 => Capability::available(
                Self::Sm86,
                L1WriteLookupWindow {
                    set_count: 256,
                    tag_slots: 2,
                },
            ),
            Self::Sm120 => Capability::research_gap(
                Self::Sm120,
                "sm_120 L1TEX write-lookup allocation window is not modeled",
            ),
        }
    }

    /// SM86 GPC-observation envelope measured with an unscored control sweep.
    /// This is a separate front-end observation boundary, not a replacement
    /// for the physical SM/event frontier and not the scored-workload timing
    /// crosswalk.  The grid term captures the small dispatch/startup increase
    /// observed as the block population grows; unsupported architectures keep
    /// the capability as an explicit research gap.
    pub const fn gpc_observation_envelope(self) -> Capability<GpcObservationEnvelope> {
        match self {
            Self::Sm86 => Capability::available(
                Self::Sm86,
                GpcObservationEnvelope {
                    base_cycles: 1776,
                    per_grid_log2_cycles: 8,
                },
            ),
            Self::Sm120 => Capability::research_gap(
                Self::Sm120,
                "sm_120 GPC observation envelope is not measured",
            ),
        }
    }

    /// SM-domain active-observation envelope measured with the independent
    /// `ncu_launch_envelope` control pair.  This boundary covers the fixed
    /// launch/activation and sampler interval that is present in
    /// `sm__cycles_active.max` but is not part of an SM-local issue/event
    /// lower bound.  It is emitted as a separate physical-observation
    /// frontier; callers must retain the unadjusted lower bound alongside it.
    pub const fn sm_active_observation_envelope(self) -> Capability<SmActiveObservationEnvelope> {
        match self {
            Self::Sm86 => Capability::available(
                Self::Sm86,
                SmActiveObservationEnvelope { fixed_cycles: 964 },
            ),
            Self::Sm120 => Capability::research_gap(
                Self::Sm120,
                "sm_120 SM-active observation envelope is not measured",
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
                    max_resident_blocks_per_sm: 16,
                },
            ),
            Self::Sm120 => Capability::research_gap(
                Self::Sm120,
                "sm_120 scheduler/barrier/occupancy hypotheses are not modeled",
            ),
        }
    }

    /// Instruction front-end structure reported for Ampere by Huerta et al.
    /// (MICRO 2025) and instantiated by their released RTX 3090 model. The
    /// lower-level cold-fill latency remains supplied by the active, SKU-bound
    /// architecture card; this capability fixes only the SM-local hierarchy,
    /// transport, prefetch, and fetch/decode structure.
    pub const fn instruction_front_end(self) -> Capability<InstructionFrontEnd> {
        match self {
            Self::Sm86 => Capability::available(
                Self::Sm86,
                InstructionFrontEnd {
                    line_bytes: 128,
                    l0_sets: 8,
                    l0_ways: 32,
                    l1_sets: 64,
                    l1_ways: 16,
                    l0_to_l1_cycles: 39,
                    l1_to_l0_cycles: 39,
                    l1_request_ports: 1,
                    l1_reply_ports: 1,
                    stream_buffer_lines: 8,
                    stream_buffers_per_subcore: 1,
                    prefetches_per_cycle: 1,
                    instruction_buffer_entries: 3,
                    fetch_decode_width: 1,
                    fetch_to_issue_stages: 2,
                },
            ),
            Self::Sm120 => Capability::research_gap(
                Self::Sm120,
                "sm_120 instruction-cache hierarchy, transport, and stream prefetch are not modeled",
            ),
        }
    }

    /// Per-SMSP/SM-shared memory issue front end measured by Huerta et al.
    /// on Ampere. Five instructions fit before backpressure (four queue slots
    /// plus the dispatch latch), address/operand work accepts one instruction
    /// every four cycles per SMSP, and the shared structure accepts one every
    /// two cycles across the SM.
    pub const fn memory_issue_pipeline(self) -> Capability<MemoryIssuePipeline> {
        match self {
            Self::Sm86 => Capability::available(
                Self::Sm86,
                MemoryIssuePipeline {
                    queue_entries_with_dispatch: 5,
                    local_initiation_interval_cycles: 4,
                    shared_accept_interval_cycles: 2,
                    global_load_uniform_war_cycles: 9,
                    global_load_regular_war_cycles: 11,
                    // A pressure-class regular constant load crosses the
                    // released SM86 constant-cache structure (11 cycles) and
                    // its measured L0↔L1 transport (39 cycles): 50 cycles.
                    // The ordinary 11-cycle global-load arm remains in force
                    // for non-pressure constant streams.
                    constant_load_war_cycles: 11 + 39,
                    global_store_uniform_32_war_cycles: 10,
                    global_store_uniform_64_war_cycles: 12,
                    global_store_uniform_128_war_cycles: 16,
                    global_store_regular_32_war_cycles: 14,
                    global_store_regular_64_war_cycles: 16,
                    global_store_regular_128_war_cycles: 20,
                    shared_load_war_cycles: 9,
                    shared_store_uniform_32_war_cycles: 10,
                    shared_store_uniform_64_war_cycles: 12,
                    shared_store_uniform_128_war_cycles: 16,
                    shared_store_regular_32_war_cycles: 12,
                    shared_store_regular_64_war_cycles: 14,
                    shared_store_regular_128_war_cycles: 18,
                    ldgsts_war_cycles: 13,
                },
            ),
            Self::Sm120 => Capability::research_gap(
                Self::Sm120,
                "sm_120 memory issue queue, address pipeline, and shared acceptance are not modeled",
            ),
        }
    }

    /// Research-only scalar ALU execution candidate for the Ampere SM86 model.
    /// Huerta et al.'s released RTX 3090 configuration gives the INT/FP paths
    /// a four-cycle latency and two-cycle initiation interval.  Production
    /// replay does not bind this single-stream approximation: the independent
    /// issue-port campaign rejected it as a generic scheduler constraint.
    pub const fn scalar_execution_pipeline(self) -> Capability<ScalarExecutionPipeline> {
        match self {
            Self::Sm86 => Capability::available(
                Self::Sm86,
                ScalarExecutionPipeline {
                    latency_cycles: 4,
                    initiation_interval_cycles: 2,
                    in_flight_capacity: 8,
                },
            ),
            Self::Sm120 => Capability::research_gap(
                Self::Sm120,
                "sm_120 scalar execution latency, initiation, and in-flight capacity are not modeled",
            ),
        }
    }

    /// Uniform/SFU cadence used by `MUFU` and uniform special-register forms.
    /// This is deliberately separate from the generic scalar ALU resource.
    pub const fn uniform_sfu_cadence(self) -> Capability<UniformSfuCadence> {
        match self {
            Self::Sm86 => Capability::available(
                Self::Sm86,
                UniformSfuCadence {
                    initiation_interval_cycles: 4,
                },
            ),
            Self::Sm120 => {
                Capability::research_gap(Self::Sm120, "sm_120 uniform/SFU cadence is not modeled")
            }
        }
    }

    /// Immediate Constant Cache (IMC) geometry and scheduler-facing miss
    /// behavior for Ampere.  IMC is distinct from the indexed constant cache:
    /// a `c[bank][offset]` operand is issued but not dispatched on a miss and
    /// is re-issued after the per-SMSP cache line returns.  The geometry and
    /// four-cycle greedy-warp switch window come from the released SM86
    /// remodeling configuration; the upper-level response latency is combined
    /// with the active SKU memory card by `sim-sm`.
    pub const fn immediate_constant_cache(self) -> Capability<ImmediateConstantCache> {
        match self {
            Self::Sm86 => Capability::available(
                Self::Sm86,
                ImmediateConstantCache {
                    sets: 4,
                    line_bytes: 64,
                    ways: 2,
                    mshr_entries: 32,
                    access_queue_entries: 1,
                    miss_initiation_interval_cycles: 1,
                    sm_structure_latency_cycles: 11,
                    l0_to_l1_cycles: 39,
                    switch_after_miss_cycles: 4,
                },
            ),
            Self::Sm120 => Capability::research_gap(
                Self::Sm120,
                "sm_120 immediate constant-cache geometry and miss replay are not modeled",
            ),
        }
    }

    /// Additional return-path service for regular-register constant loads on
    /// SM86.  This is intentionally an *additional* path delay, not a
    /// replacement for the carded L1/L2 latency arms: the remodeled
    /// GPGPU-Sim configuration exposes a distinct constant-cache structure
    /// (`constant_cache_latency_at_sm_structure = 11`), and the independent
    /// source/stall assay identifies an eight-cycle return-service gap after
    /// the ordinary hierarchy response is known.  The marker is consumed only
    /// by response-bearing constant loads; cache tags and observer counts stay
    /// on their normal paths.
    pub const fn constant_memory_return_service(self) -> Capability<ConstantMemoryReturnService> {
        match self {
            Self::Sm86 => Capability::available(
                Self::Sm86,
                ConstantMemoryReturnService {
                    additional_cycles: 8,
                },
            ),
            Self::Sm120 => Capability::research_gap(
                Self::Sm120,
                "sm_120 constant-memory return service is not modeled",
            ),
        }
    }

    /// Tensor-pipe effective service intervals for the exact matrix
    /// instructions admitted by the sm_86 control-sidecar corpus. These are
    /// GA10x-local. Dense HMMA.16816 exposes a 32-cycle saturated execution
    /// interval (the released Ampere model exposes a 32-cycle tensor execution
    /// latency), while sparse HMMA.SP.16832 and IMMA.16816 retain their
    /// independently measured 16/8-cycle initiation cadences. Short dense-HMMA
    /// functions that cannot fill this window use the 16-cycle initiation
    /// cadence in `sim-sm`; the carded value is the saturated ceiling, not a
    /// claim that every independent HMMA issue consumes 32 cycles.
    /// TF32 m16n8k8 has a separate 32-cycle interval, with no static-site
    /// halving. The independent eight-accumulator probe approaches that
    /// cadence; its dependent-loop time is not result latency.
    /// See workloads/research_probes/tf32_issue_probe/README.md.
    pub const fn tensor_issue_intervals(self) -> Capability<TensorIssueIntervals> {
        match self {
            Self::Sm86 => Capability::available(
                Self::Sm86,
                TensorIssueIntervals {
                    hmma_16816_f16_bf16: 32,
                    hmma_1688_tf32: 32,
                    hmma_sparse_16832_f16_bf16: 16,
                    imma_16816_s8: 8,
                    imma_16832_s8: 16,
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
    /// delay.  A separate exact-loop control identifies a nine-cycle
    /// next-issue interval only when a predicated backward branch immediately
    /// consumes a freshly produced predicate; this is kept as a distinct
    /// shape-dependent capability rather than applied to every backward jump.
    pub const fn branch_resolution(self) -> Capability<BranchResolution> {
        match self {
            Self::Sm86 => Capability::available(
                Self::Sm86,
                BranchResolution {
                    next_issue_interval_cycles: 10,
                    uniform_backward_target_resolution_cycles: 3,
                    uniform_backward_predicate_interval_cycles: 9,
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

    /// Source-side L1TEX return cadence for cacheable non-bulk asynchronous
    /// copies.  The Ampere SM86 memory structure has a banked source-return
    /// path; the released Huerta model's four-bank/half-warp service and the
    /// independent source-hit assay resolve to one accepted warp-level
    /// cacheable LDGSTS source instruction every eight SM cycles for this
    /// replay granularity.  Its sectors are one coalesced source transaction;
    /// this is not a per-sector latency, cache tag, or hit-count rule.
    pub const fn ldgsts_l1_hit_return_service(self) -> Capability<LdgstsL1HitReturnService> {
        match self {
            Self::Sm86 => {
                Capability::available(Self::Sm86, LdgstsL1HitReturnService { interval_cycles: 8 })
            }
            Self::Sm120 => Capability::research_gap(
                Self::Sm120,
                "sm_120 LDGSTS L1-hit return service is not modeled",
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
                    global_to_shared_transport_cycles: 19,
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
    Rtx5090,
}

impl Sku {
    pub fn parse(token: &str) -> Result<Self, ArchError> {
        match token {
            "rtx_3090_ti" => Ok(Self::Rtx3090Ti),
            "rtx_5090" => Ok(Self::Rtx5090),
            other => Err(ArchError::UnknownSku {
                token: other.to_owned(),
            }),
        }
    }

    pub const fn token(self) -> &'static str {
        match self {
            Self::Rtx3090Ti => "rtx_3090_ti",
            Self::Rtx5090 => "rtx_5090",
        }
    }

    pub const fn arch(self) -> ArchId {
        match self {
            Self::Rtx3090Ti => ArchId::Sm86,
            Self::Rtx5090 => ArchId::Sm120,
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
            CapabilityState::ResearchGap { reason } => CapabilityStatus::ResearchGap { reason },
            CapabilityState::Unsupported { reason } => CapabilityStatus::Unsupported { reason },
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

/// Architecture-local request/tag window for L1TEX write lookups. This is
/// intentionally distinct from the physical L1 geometry: NCU's write lookup
/// counter is a sector observer, while the replay's read-state cache keeps its
/// independently validated replacement approximation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L1WriteLookupWindow {
    pub set_count: u32,
    pub tag_slots: u32,
}

/// Architecture-local approximation of the front-end/semaphore boundary that
/// separates NCU's GPC elapsed-cycle observation from an SM-local modeled
/// event frontier.  It is deliberately exposed as a diagnostic component so
/// callers can retain the unadjusted lower bound alongside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpcObservationEnvelope {
    pub base_cycles: u64,
    pub per_grid_log2_cycles: u64,
}

/// Fixed SM-clock observation boundary used to expose a physical active-cycle
/// frontier without rewriting the simulator's raw issue/event lower bounds.
/// The value is a protocol-specific architecture capability, not an NCU target
/// and not a workload correction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmActiveObservationEnvelope {
    pub fixed_cycles: u64,
}

impl SmActiveObservationEnvelope {
    pub const fn cycles(self) -> u64 {
        self.fixed_cycles
    }
}

impl GpcObservationEnvelope {
    /// Return `(envelope_cycles, floor_log2(grid_blocks))`. A zero block
    /// population is not a valid launch and therefore returns `None`.
    pub fn cycles_for_grid(self, grid_blocks: u64) -> Option<(u64, u32)> {
        if grid_blocks == 0 {
            return None;
        }
        let log2 = 63_u32.saturating_sub(grid_blocks.leading_zeros());
        let grid_term = self.per_grid_log2_cycles.checked_mul(u64::from(log2))?;
        let cycles = self.base_cycles.checked_add(grid_term)?;
        Some((cycles, log2))
    }
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
    /// Architectural CTA residency ceiling, independent of the warp and
    /// shared-memory ceilings.  Ampere exposes a 16-block-per-SM limit even
    /// for one-warp CTAs; omitting it lets a replay admit 48 one-warp blocks
    /// and changes the wave schedule.
    pub max_resident_blocks_per_sm: u32,
}

/// Per-SM instruction hierarchy and per-SMSP fetch structure. Cache capacities
/// are `sets * ways * line_bytes`; all port counts and queue widths are
/// explicit so another architecture cannot silently inherit Ampere geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InstructionFrontEnd {
    pub line_bytes: u32,
    pub l0_sets: u32,
    pub l0_ways: u32,
    pub l1_sets: u32,
    pub l1_ways: u32,
    pub l0_to_l1_cycles: u64,
    pub l1_to_l0_cycles: u64,
    pub l1_request_ports: u32,
    pub l1_reply_ports: u32,
    pub stream_buffer_lines: u32,
    pub stream_buffers_per_subcore: u32,
    pub prefetches_per_cycle: u32,
    pub instruction_buffer_entries: u32,
    pub fetch_decode_width: u32,
    pub fetch_to_issue_stages: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryIssuePipeline {
    pub queue_entries_with_dispatch: u32,
    pub local_initiation_interval_cycles: u64,
    pub shared_accept_interval_cycles: u64,
    pub global_load_uniform_war_cycles: u64,
    pub global_load_regular_war_cycles: u64,
    /// Operand/read visibility latency for regular-register constant loads.
    /// SM86 has a distinct L0/L1 constant-cache path.
    pub constant_load_war_cycles: u64,
    pub global_store_uniform_32_war_cycles: u64,
    pub global_store_uniform_64_war_cycles: u64,
    pub global_store_uniform_128_war_cycles: u64,
    pub global_store_regular_32_war_cycles: u64,
    pub global_store_regular_64_war_cycles: u64,
    pub global_store_regular_128_war_cycles: u64,
    pub shared_load_war_cycles: u64,
    pub shared_store_uniform_32_war_cycles: u64,
    pub shared_store_uniform_64_war_cycles: u64,
    pub shared_store_uniform_128_war_cycles: u64,
    pub shared_store_regular_32_war_cycles: u64,
    pub shared_store_regular_64_war_cycles: u64,
    pub shared_store_regular_128_war_cycles: u64,
    pub ldgsts_war_cycles: u64,
}

/// Architecture-local scalar INT/FP execution resource.  `in_flight_capacity`
/// bounds the number of dispatched operations whose execution slot has not
/// yet retired; it is not a generic register scoreboard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarExecutionPipeline {
    pub latency_cycles: u64,
    pub initiation_interval_cycles: u64,
    pub in_flight_capacity: u32,
}

/// Cadence shared by uniform-control and special-function operations.  The
/// scheduler treats this as an initiation interval (not a latency promise),
/// allowing carded front ends to expose a distinct SFU/uniform pipe without
/// borrowing the generic INT/FP queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UniformSfuCadence {
    pub initiation_interval_cycles: u64,
}

/// Architecture-local Immediate Constant Cache (IMC) configuration.  The
/// cache is private to one SMSP; its miss service is serialized by the
/// architecture's L1C access queue while a small number of outstanding lines
/// may be tracked by MSHRs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImmediateConstantCache {
    pub sets: u32,
    pub line_bytes: u32,
    pub ways: u32,
    pub mshr_entries: u32,
    pub access_queue_entries: u32,
    /// Initiation interval of the pipelined L1C miss path.  The released
    /// SM86 queue advances one stage per cycle, so independent line misses
    /// can be admitted every cycle even while their responses are pending.
    pub miss_initiation_interval_cycles: u64,
    pub sm_structure_latency_cycles: u64,
    pub l0_to_l1_cycles: u64,
    pub switch_after_miss_cycles: u64,
}

/// Architecture-local additional return-path service for `LDG.*.CONSTANT`.
/// It is separate from the ordinary cache latency because the same global
/// address may still mutate normal L1/L2 state while its constant-cache return
/// consumes a distinct SM-side path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstantMemoryReturnService {
    pub additional_cycles: u64,
}

/// Per-SMSP initiation intervals for the sm_86 tensor instructions whose
/// throughput is architecture-backed and whose SASS forms are classified by
/// `sim-sm`. A separate named field keeps an unsupported matrix shape from
/// silently inheriting the rate of a superficially similar instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorIssueIntervals {
    pub hmma_16816_f16_bf16: u64,
    /// Dense SM86 TF32 `HMMA.1688.F32.TF32` initiation interval.  This is
    /// intentionally separate from the m16n8k16 field and its static-site
    /// heuristic; it is not a dependent-result latency or retirement bound.
    pub hmma_1688_tf32: u64,
    pub hmma_sparse_16832_f16_bf16: u64,
    pub imma_16816_s8: u64,
    /// Dense SM86 INT8 `IMMA.16832.S8.S8` initiation interval.  This is
    /// distinct from the m16n8k16 INT8 form and is backed by the independent
    /// eight-accumulator probe in workloads/research_probes/imma_issue_probe.
    pub imma_16832_s8: u64,
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
    /// Measured next-issue interval for the narrow shape in which a uniform
    /// backward branch immediately follows its predicate producer.  This is
    /// not a generic branch latency and must be selected only from static
    /// SASS adjacency.
    pub uniform_backward_predicate_interval_cycles: u64,
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

/// Architecture-local cadence of the cacheable LDGSTS source-hit return
/// interface.  It affects only delivery timestamps; L1/L2 observer counts and
/// replacement state remain independent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LdgstsL1HitReturnService {
    pub interval_cycles: u64,
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
    /// Fixed transport between the completed global source and the shared
    /// destination path in the released Ampere model.
    pub global_to_shared_transport_cycles: u64,
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
                .instruction_front_end()
                .require("instruction_front_end"),
            Err(CapabilityError::ResearchGap {
                arch: ArchId::Sm120,
                capability: "instruction_front_end",
                reason,
            }) if !reason.is_empty()
        ));
        assert!(matches!(
            ArchId::Sm120
                .memory_issue_pipeline()
                .require("memory_issue_pipeline"),
            Err(CapabilityError::ResearchGap {
                arch: ArchId::Sm120,
                capability: "memory_issue_pipeline",
                reason,
            }) if !reason.is_empty()
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
    fn sm86_l1_write_lookup_window_is_architecture_local() {
        assert_eq!(
            ArchId::Sm86
                .l1_write_lookup_window()
                .require("l1_write_lookup_window")
                .unwrap(),
            L1WriteLookupWindow {
                set_count: 256,
                tag_slots: 2,
            }
        );
        assert!(matches!(
            ArchId::Sm120
                .l1_write_lookup_window()
                .require("l1_write_lookup_window"),
            Err(CapabilityError::ResearchGap {
                arch: ArchId::Sm120,
                capability: "l1_write_lookup_window",
                reason,
            }) if !reason.is_empty()
        ));
    }

    #[test]
    fn sm86_gpc_observation_envelope_is_grid_scoped_and_sm120_is_gap() {
        let envelope = ArchId::Sm86
            .gpc_observation_envelope()
            .require("gpc_observation_envelope")
            .unwrap();
        assert_eq!(envelope.cycles_for_grid(1), Some((1776, 0)));
        assert_eq!(envelope.cycles_for_grid(64), Some((1824, 6)));
        assert_eq!(envelope.cycles_for_grid(1024), Some((1856, 10)));
        assert_eq!(envelope.cycles_for_grid(0), None);
        assert!(matches!(
            ArchId::Sm120
                .gpc_observation_envelope()
                .require("gpc_observation_envelope"),
            Err(CapabilityError::ResearchGap {
                arch: ArchId::Sm120,
                capability: "gpc_observation_envelope",
                reason,
            }) if !reason.is_empty()
        ));
    }

    #[test]
    fn sm86_sm_active_observation_envelope_is_architecture_local() {
        assert_eq!(
            ArchId::Sm86
                .sm_active_observation_envelope()
                .require("sm_active_observation_envelope")
                .unwrap()
                .cycles(),
            964
        );
        assert!(matches!(
            ArchId::Sm120
                .sm_active_observation_envelope()
                .require("sm_active_observation_envelope"),
            Err(CapabilityError::ResearchGap {
                arch: ArchId::Sm120,
                capability: "sm_active_observation_envelope",
                reason,
            }) if !reason.is_empty()
        ));
    }

    #[test]
    fn sm86_tensor_issue_intervals_are_architecture_local() {
        assert_eq!(
            ArchId::Sm86
                .tensor_issue_intervals()
                .require("tensor_issue_intervals")
                .unwrap(),
            TensorIssueIntervals {
                hmma_16816_f16_bf16: 32,
                hmma_1688_tf32: 32,
                hmma_sparse_16832_f16_bf16: 16,
                imma_16816_s8: 8,
                imma_16832_s8: 16,
            }
        );
    }

    #[test]
    fn sm86_instruction_front_end_matches_the_released_rtx3090_model() {
        assert_eq!(
            ArchId::Sm86
                .instruction_front_end()
                .require("instruction_front_end")
                .unwrap(),
            InstructionFrontEnd {
                line_bytes: 128,
                l0_sets: 8,
                l0_ways: 32,
                l1_sets: 64,
                l1_ways: 16,
                l0_to_l1_cycles: 39,
                l1_to_l0_cycles: 39,
                l1_request_ports: 1,
                l1_reply_ports: 1,
                stream_buffer_lines: 8,
                stream_buffers_per_subcore: 1,
                prefetches_per_cycle: 1,
                instruction_buffer_entries: 3,
                fetch_decode_width: 1,
                fetch_to_issue_stages: 2,
            }
        );
    }

    #[test]
    fn sm86_memory_issue_pipeline_includes_the_released_constant_pressure_path() {
        assert_eq!(
            ArchId::Sm86
                .memory_issue_pipeline()
                .require("memory_issue_pipeline")
                .unwrap(),
            MemoryIssuePipeline {
                queue_entries_with_dispatch: 5,
                local_initiation_interval_cycles: 4,
                shared_accept_interval_cycles: 2,
                global_load_uniform_war_cycles: 9,
                global_load_regular_war_cycles: 11,
                constant_load_war_cycles: 11 + 39,
                global_store_uniform_32_war_cycles: 10,
                global_store_uniform_64_war_cycles: 12,
                global_store_uniform_128_war_cycles: 16,
                global_store_regular_32_war_cycles: 14,
                global_store_regular_64_war_cycles: 16,
                global_store_regular_128_war_cycles: 20,
                shared_load_war_cycles: 9,
                shared_store_uniform_32_war_cycles: 10,
                shared_store_uniform_64_war_cycles: 12,
                shared_store_uniform_128_war_cycles: 16,
                shared_store_regular_32_war_cycles: 12,
                shared_store_regular_64_war_cycles: 14,
                shared_store_regular_128_war_cycles: 18,
                ldgsts_war_cycles: 13,
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
                uniform_backward_predicate_interval_cycles: 9,
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
                global_to_shared_transport_cycles: 19,
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
