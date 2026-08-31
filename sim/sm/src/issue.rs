//! SMSP issue model (design §6.2): per-warp program replay where issue
//! timing is driven by the compiler's own dependence encoding — the SASS
//! control bits (stall counts, dependence-barrier set/wait, yield) — not by
//! simulated scoreboards. This is the 13–17% cycles-MAPE lever the UPC
//! MICRO'25 modern-core work identified `F6`.
//!
//! Model shape:
//! - 4 sub-cores (SMSPs) per SM; a warp lives on sub-core
//!   `warp_slot % 4` (the physical `warpid_sm` slot when traced).
//! - CGGTY-style pick per sub-core per cycle: keep issuing from the held
//!   warp while it stays ready (greedy; a set yield bit drops the hold),
//!   otherwise switch to the YOUNGEST ready warp (largest arrival stamp).
//! - Each issued instruction makes its warp eligible again at
//!   `t + max(stall, 1)`; a set write/read dependence barrier is stamped
//!   with a clear time; an instruction whose control word carries a wait
//!   mask cannot issue until every masked barrier has cleared.
//! - Admitted tensor instructions additionally wait for their SMSP-local
//!   Tensor Core initiation timeline. Other pipelines remain dispatch-only,
//!   so independent non-tensor work may issue while the tensor pipe is busy.
//! - Exact predicated stall-5/yield branches carry architecture-local
//!   readiness: ten cycles for the characterized forward construction and
//!   three target-resolution cycles for uniform backward loops. The latter is
//!   shorter than the encoded stall and is therefore diagnostic unless a
//!   later fetch model exposes it.
//! - `BAR.*` (classified by the sidecar's opcode text) is a block-wide
//!   barrier: generation semantics — it releases when every unfinished
//!   warp of the block is waiting at it (warps that already finished do
//!   not block release, matching Volta+ exited-thread semantics).
//! - Block residency: blocks are admitted to the SM in launch order while
//!   resident warps and resident shared-memory bytes stay within their
//!   architecture-local ceilings and concurrent CTAs use disjoint physical warp
//!   slots; a finished block frees both resources for a later CTA (wave execution).
//!
//! Memory timing comes in three modes ([`SmMemMode`]):
//! - **Flat**: no cache model; global-class barriers clear at the flat
//!   cold-DRAM latency (the pre-L1 v0 behavior).
//! - **L1Const**: a per-SM sectored L1 probed AT ISSUE TIME with fill
//!   times known at issue (constant L2-hit latency below the L1) — the
//!   sim-mem L1 v0 behavior, kept as the lockstep-differential surface.
//! - **L1Shared**: the request/response INVERSION for the shared L2
//!   (design §6.3): an L1 miss's fill time is NOT knowable at issue —
//!   the shared L2's state when the request arrives decides L2-hit vs
//!   DRAM pricing. Misses emit [`MemRequest`]s; ordinary load barriers
//!   and non-bulk async-copy groups retain typed completion targets on
//!   hits, fills, and MSHR retries. [`SmCore::on_response`] completes the
//!   fill and every target at response-event time. An explicit
//!   `LDGSTS.E.BYPASS` instead keeps the direct-L2 async response route.
//!   MSHR-full accesses park in a retry queue drained on the next
//!   completion (the event-mode analogue of the constant mode's
//!   retry-at-earliest-fill; the warp itself keeps issuing until a wait
//!   mask or async-group wait stops it, like the hardware's LSU replay
//!   path).
//!
//! [`SmCore`] is a resumable actor: `advance(horizon)` runs issue cycles
//! strictly BEFORE `horizon` (the event system passes the next queued
//! event's time so no cross-SM effect is overrun; a self-emitted request
//! additionally caps the run-ahead at the earliest time its response
//! could arrive). [`simulate_sm_with_residency`] is the synchronous wrapper for
//! the Flat/L1Const modes (horizon = ∞, no requests possible).
//!
//! The instruction COUNT produced by replay is placement- and
//! timing-invariant: every traced record issues exactly once, so
//! `instructions_issued` is exact regardless of the cycle model's fidelity.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet, VecDeque};
use std::num::{NonZeroU32, NonZeroU128};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use sim_arch::card::{ArchCard, CalibError, RequiredIssueResources};
use sim_arch::{
    ArchId, CtaBarrierIssue, FixedL1TexWritebackPriority, LdgstsCompletionThroughput,
    LdgstsDestinationWavefrontProjection, SharedBankProjection,
};
use sim_mem::{
    AsyncCopyCompletion, CacheOpToken, EvAccess, EvAccessUnobserved, FillKey, L1AccessError,
    L1Cache, L1Config, L1Counters, L1MutationBudget, L2OpClass, L2Source, LdgstsL1Policy, MemRole,
    MemoryOrigin, decode_async_copy_completion,
};

use crate::control_bits::{ControlDomainError, DepBarrier};
use crate::program::{FunctionSchedule, IssueSlot, OpClass, PipelineClass};
use crate::sass_operands::{FixedLatencyOpcode, FixedLatencyOperandStatus};

pub use sim_mem::AccessKind;

/// One dynamic instruction, compacted for replay.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ReplayInst {
    /// Function-relative pc — the join key into [`FunctionSchedule`].
    pub(crate) pc: u32,
    pub(crate) kind: AccessKind,
}

/// One traced warp of a block: physical slot + its dynamic stream + the
/// sector arena its records' `off/len` payloads index.
#[derive(Debug, Default)]
pub(crate) struct WarpReplay {
    /// Physical warp slot on the SM (`warpid_sm` when traced); sub-core
    /// assignment is `warp_slot % 4`.
    pub(crate) warp_slot: i32,
    pub(crate) insts: Vec<ReplayInst>,
    pub(crate) sectors: Vec<i64>,
}

/// One traced block assigned to this SM, in launch order.
#[derive(Debug)]
pub(crate) struct BlockReplay {
    pub(crate) launch_index: i64,
    pub(crate) warps: Vec<WarpReplay>,
}

/// Architecture-card capacity paired with the admitted launch's per-CTA demand.
/// The trace's split static/dynamic values have already been checked against the
/// admitted total before this value reaches the issue model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SharedMemoryResidency {
    max_bytes_per_sm: u64,
    bytes_per_cta: u64,
}

impl SharedMemoryResidency {
    pub(crate) const fn new(max_bytes_per_sm: u64, bytes_per_cta: u64) -> Self {
        Self {
            max_bytes_per_sm,
            bytes_per_cta,
        }
    }

    pub(crate) const fn max_bytes_per_sm(self) -> u64 {
        self.max_bytes_per_sm
    }

    pub(crate) const fn bytes_per_cta(self) -> u64 {
        self.bytes_per_cta
    }

    #[cfg(test)]
    pub(crate) const fn unconstrained_fixture() -> Self {
        Self::new(u64::MAX, 0)
    }
}

/// Source of scheduler-resource values. The architecture-default source is
/// executable; the full card source remains unavailable until every card field
/// is consumed rather than shadowed by structural constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueResourceSource {
    /// Existing sm_86-only 25/8/48 architecture-local hypotheses.
    ArchitectureDefaults,
    /// Strict architecture-card resources. Construction is supported, but replay
    /// remains unavailable while structural fields are dormant.
    Carded(RequiredIssueResources),
    /// Unit-test-only lane for boundary tests that need synthetic scheduler
    /// values. This variant does not exist in production builds.
    #[cfg(test)]
    SyntheticTestFixture,
}

impl IssueResourceSource {
    #[cfg(feature = "campaign")]
    pub const fn source_label(self) -> &'static str {
        match self {
            // Retained campaign result schema spelling. The direct simulator
            // does not compile or emit this compatibility field.
            Self::ArchitectureDefaults => "architecture_local_reviewed_hypothesis",
            Self::Carded(_) => "architecture_card",
            #[cfg(test)]
            Self::SyntheticTestFixture => "synthetic_test_fixture",
        }
    }
}

/// Timing parameters. The executable configuration combines card latencies
/// with architecture-local defaults. [`IssueModelParams::from_card`] accepts a
/// complete card projection, but that source is not executable until the
/// scheduler consumes all of its structural fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IssueModelParams {
    /// Deployment architecture whose card and issue hypotheses produced these parameters.
    pub(crate) arch: ArchId,
    pub(crate) resource_source: IssueResourceSource,
    /// L1 hit data-ready latency (card `l1_hit_latency_cycles`, ceiled).
    pub(crate) l1_hit_latency: u64,
    /// End-to-end L1-miss/L2-HIT data-ready latency (card
    /// `l2_hit_latency_cycles`, ceiled). In L1Const mode every miss is
    /// priced at this (the L2-hit assumption); in L1Shared mode it is
    /// the L2-hit arm of the ladder AND the lower bound on any response.
    pub(crate) l1_miss_latency: u64,
    /// End-to-end L1-miss/L2-MISS data-ready latency (card
    /// `dram_latency_cycles`, ceiled): the DRAM arm of the ladder in
    /// L1Shared mode, and the flat global barrier latency in Flat mode.
    pub(crate) global_flat_latency: u64,
    /// Shared-class base barrier latency (card `shared_latency_cycles`).
    pub(crate) shared_write_barrier_latency: u64,
    /// Additional dependent completion latency for every shared bank wavefront
    /// after the first. The trace projection and this timing source are both
    /// architecture-local; zero-wavefront legacy/predicated records retain the
    /// base latency.
    pub(crate) shared_additional_wavefront_latency: u64,
    /// Write-barrier clear latency for non-memory producers (S2R, MUFU,
    /// I2F, HMMA & co). Hypothesis knob: ~20–25 cycles is the
    /// special-register/SFU class in Abdelkhalik et al. (HPEC'22, A100);
    /// not yet microbenched on this project's cards.
    pub(crate) non_memory_write_barrier_latency: u64,
    /// Read-dependence barrier clear latency (operand consumption).
    /// Hypothesis knob: source operands are read a few cycles after issue.
    pub(crate) read_barrier_latency: u64,
    /// Access-local lower bound on finite L1 miss admission for cacheable
    /// `LDGSTS.E` source traffic. Ordinary loads retain the card's unresolved
    /// MSHR approximation; an unlimited L1 ablation remains unlimited.
    pub(crate) ldgsts_l1_mshr_floor: NonZeroU32,
    /// Effective async-copy payload delivery rates. Replay converts route-local
    /// payload fallback and any exact destination projection into this model's
    /// fixed-point work domain before issue.
    pub(crate) ldgsts_completion_throughput: LdgstsCompletionThroughput,
    /// Independently measured cache-all destination completion time. Each exact
    /// source-cohort-scoped bank wavefront contributes this many cycles; bypass
    /// copies retain payload fallback until their distinct projection is known.
    pub(crate) ldgsts_cache_all_destination_wavefront_cycles: u64,
    /// Rational aggregate cache-all destination service shared by one SM.
    pub(crate) ldgsts_cache_all_destination_service_wavefronts: u64,
    pub(crate) ldgsts_cache_all_destination_service_cycles: u64,
    /// Resident-warp ceiling per SM. The current sm_86 hypothesis is 48;
    /// template architectures return a typed capability gap until their
    /// occupancy evidence is carded.
    pub(crate) max_resident_warps_per_sm: u32,
}

pub(crate) fn checked_cycle_add(
    operation: &'static str,
    base: u64,
    delta: u64,
) -> Result<u64, IssueError> {
    base.checked_add(delta).ok_or(IssueError::TimeOverflow {
        operation,
        base,
        delta,
    })
}

fn checked_latency(name: &'static str, value: f64) -> Result<u64, IssueError> {
    // `u64::MAX as f64` is rounded to 2^64.  Treat it as an exclusive
    // boundary: accepting equality would let the float→u64 cast saturate.
    if !value.is_finite() || value <= 0.0 || value >= u64::MAX as f64 {
        return Err(IssueError::InvalidLatency { name, value });
    }
    let rounded = value.ceil();
    if rounded <= 0.0 || rounded >= u64::MAX as f64 {
        return Err(IssueError::InvalidLatency { name, value });
    }
    Ok(rounded as u64)
}

fn shared_additional_wavefront_latency(arch: ArchId) -> Result<u64, IssueError> {
    let timing = arch
        .shared_memory_wavefront_timing()
        .require("shared_memory_wavefront_timing")?;
    match timing.projection {
        SharedBankProjection::Banks32Words4Bytes => {}
    }
    Ok(timing.additional_wavefront_latency_cycles)
}

fn ldgsts_l1_mshr_floor(arch: ArchId) -> Result<NonZeroU32, IssueError> {
    let admission = arch
        .ldgsts_l1_fill_admission()
        .require("ldgsts_l1_fill_admission")?;
    NonZeroU32::new(admission.minimum_concurrent_line_fills).ok_or(
        IssueError::InvalidLdgstsL1MshrFloor {
            arch,
            floor: admission.minimum_concurrent_line_fills,
        },
    )
}

fn ldgsts_completion_throughput(arch: ArchId) -> Result<LdgstsCompletionThroughput, IssueError> {
    let throughput = arch
        .ldgsts_completion_throughput()
        .require("ldgsts_completion_throughput")?;
    let cache_all = throughput.cache_all_payload_bytes_per_cycle;
    let bypass = throughput.bypass_payload_bytes_per_cycle;
    let common = cache_all.max(bypass);
    if cache_all == 0 || bypass == 0 || common % cache_all != 0 || common % bypass != 0 {
        return Err(IssueError::InvalidLdgstsCompletionThroughput {
            arch,
            cache_all_bytes_per_cycle: cache_all,
            bypass_bytes_per_cycle: bypass,
        });
    }
    Ok(throughput)
}

fn ldgsts_cache_all_destination_wavefront_cycles(arch: ArchId) -> Result<u64, IssueError> {
    let timing = arch
        .ldgsts_destination_wavefront_timing()
        .require("ldgsts_destination_wavefront_timing")?;
    match timing.projection {
        LdgstsDestinationWavefrontProjection::SourceCohorts128BytesBanks32Words4Bytes => {}
    }
    if timing.cycles_per_wavefront == 0 {
        return Err(IssueError::InvalidLdgstsDestinationWavefrontTiming {
            arch,
            cycles_per_wavefront: timing.cycles_per_wavefront,
        });
    }
    Ok(timing.cycles_per_wavefront)
}

fn ldgsts_cache_all_destination_service_rate(arch: ArchId) -> Result<(u64, u64), IssueError> {
    let timing = arch
        .ldgsts_destination_wavefront_timing()
        .require("ldgsts_destination_wavefront_timing")?;
    if timing.aggregate_service_wavefronts == 0 || timing.aggregate_service_cycles == 0 {
        return Err(IssueError::InvalidLdgstsDestinationWavefrontThroughput {
            arch,
            service_wavefronts: timing.aggregate_service_wavefronts,
            service_cycles: timing.aggregate_service_cycles,
        });
    }
    Ok((
        timing.aggregate_service_wavefronts,
        timing.aggregate_service_cycles,
    ))
}

impl IssueModelParams {
    /// Resolve card latencies plus the retained architecture-local issue
    /// hypotheses. The name is intentionally explicit: this executable lane does
    /// not upgrade 25/8/48 to card facts. A template architecture with missing
    /// scheduler facts returns a typed capability error instead of inheriting
    /// sm_86 knobs.
    pub fn from_legacy_card_latencies_for_arch(
        arch: ArchId,
        l1_hit_cycles: f64,
        l2_hit_cycles: f64,
        dram_cycles: f64,
        shared_cycles: f64,
    ) -> Result<Self, IssueError> {
        let hypotheses = arch.issue_hypotheses().require("issue_hypotheses")?;
        let shared_additional_wavefront_latency = shared_additional_wavefront_latency(arch)?;
        let ldgsts_l1_mshr_floor = ldgsts_l1_mshr_floor(arch)?;
        let ldgsts_completion_throughput = ldgsts_completion_throughput(arch)?;
        let ldgsts_cache_all_destination_wavefront_cycles =
            ldgsts_cache_all_destination_wavefront_cycles(arch)?;
        let (
            ldgsts_cache_all_destination_service_wavefronts,
            ldgsts_cache_all_destination_service_cycles,
        ) = ldgsts_cache_all_destination_service_rate(arch)?;
        Ok(Self {
            arch,
            resource_source: IssueResourceSource::ArchitectureDefaults,
            l1_hit_latency: checked_latency("l1_hit_cycles", l1_hit_cycles)?,
            l1_miss_latency: checked_latency("l2_hit_cycles", l2_hit_cycles)?,
            global_flat_latency: checked_latency("dram_cycles", dram_cycles)?,
            shared_write_barrier_latency: checked_latency("shared_cycles", shared_cycles)?,
            shared_additional_wavefront_latency,
            non_memory_write_barrier_latency: hypotheses.non_memory_write_barrier_latency,
            read_barrier_latency: hypotheses.read_barrier_latency,
            ldgsts_l1_mshr_floor,
            ldgsts_completion_throughput,
            ldgsts_cache_all_destination_wavefront_cycles,
            ldgsts_cache_all_destination_service_wavefronts,
            ldgsts_cache_all_destination_service_cycles,
            max_resident_warps_per_sm: hypotheses.max_resident_warps_per_sm,
        })
    }

    /// Construct the strict carded resource lane without a legacy fallback.
    /// Successful construction verifies that all five resource blocks and all
    /// required latency entries came from the same architecture-bound card. It
    /// cannot run until the scheduler consumes the dormant structural fields.
    pub fn from_card(expected_arch: ArchId, card: &ArchCard) -> Result<Self, IssueError> {
        let arch = card.arch_id()?;
        if arch != expected_arch {
            return Err(IssueError::ArchitectureMismatch {
                component: "strict issue architecture card",
                expected: expected_arch,
                got: arch,
            });
        }
        let latencies = card.required_latencies()?;
        let resources = card.required_issue_resources()?;
        let shared_additional_wavefront_latency = shared_additional_wavefront_latency(arch)?;
        let ldgsts_l1_mshr_floor = ldgsts_l1_mshr_floor(arch)?;
        let ldgsts_completion_throughput = ldgsts_completion_throughput(arch)?;
        let ldgsts_cache_all_destination_wavefront_cycles =
            ldgsts_cache_all_destination_wavefront_cycles(arch)?;
        let (
            ldgsts_cache_all_destination_service_wavefronts,
            ldgsts_cache_all_destination_service_cycles,
        ) = ldgsts_cache_all_destination_service_rate(arch)?;
        Ok(Self {
            arch,
            resource_source: IssueResourceSource::Carded(resources),
            l1_hit_latency: checked_latency("l1_hit_cycles", latencies.l1_hit)?,
            l1_miss_latency: checked_latency("l2_hit_cycles", latencies.l2_hit)?,
            global_flat_latency: checked_latency("dram_cycles", latencies.dram)?,
            shared_write_barrier_latency: checked_latency("shared_cycles", latencies.shared)?,
            shared_additional_wavefront_latency,
            non_memory_write_barrier_latency: resources.barrier_latencies.non_memory_write,
            read_barrier_latency: resources.barrier_latencies.read,
            ldgsts_l1_mshr_floor,
            ldgsts_completion_throughput,
            ldgsts_cache_all_destination_wavefront_cycles,
            ldgsts_cache_all_destination_service_wavefronts,
            ldgsts_cache_all_destination_service_cycles,
            max_resident_warps_per_sm: resources.occupancy.max_resident_warps_per_sm,
        })
    }

    pub const fn arch(&self) -> ArchId {
        self.arch
    }

    pub const fn resource_source(&self) -> IssueResourceSource {
        self.resource_source
    }

    pub const fn l1_hit_latency(&self) -> u64 {
        self.l1_hit_latency
    }

    pub const fn l1_miss_latency(&self) -> u64 {
        self.l1_miss_latency
    }

    pub const fn global_flat_latency(&self) -> u64 {
        self.global_flat_latency
    }

    pub const fn shared_write_barrier_latency(&self) -> u64 {
        self.shared_write_barrier_latency
    }

    pub const fn shared_additional_wavefront_latency(&self) -> u64 {
        self.shared_additional_wavefront_latency
    }

    pub const fn non_memory_write_barrier_latency(&self) -> u64 {
        self.non_memory_write_barrier_latency
    }

    pub const fn read_barrier_latency(&self) -> u64 {
        self.read_barrier_latency
    }

    pub const fn ldgsts_l1_mshr_floor(&self) -> NonZeroU32 {
        self.ldgsts_l1_mshr_floor
    }

    pub const fn ldgsts_completion_throughput(&self) -> LdgstsCompletionThroughput {
        self.ldgsts_completion_throughput
    }

    pub const fn ldgsts_cache_all_destination_wavefront_cycles(&self) -> u64 {
        self.ldgsts_cache_all_destination_wavefront_cycles
    }

    pub const fn ldgsts_cache_all_destination_service_wavefronts(&self) -> u64 {
        self.ldgsts_cache_all_destination_service_wavefronts
    }

    pub const fn ldgsts_cache_all_destination_service_cycles(&self) -> u64 {
        self.ldgsts_cache_all_destination_service_cycles
    }

    fn ldgsts_delivery_units_per_cycle(&self) -> u64 {
        self.ldgsts_completion_throughput
            .cache_all_payload_bytes_per_cycle
            .max(
                self.ldgsts_completion_throughput
                    .bypass_payload_bytes_per_cycle,
            )
    }

    fn ldgsts_delivery_cycles(&self, work: u64) -> u64 {
        work.div_ceil(self.ldgsts_delivery_units_per_cycle())
    }

    fn ldgsts_completion_ready(&self, source_ready: u64, work: u64) -> Result<u64, IssueError> {
        checked_cycle_add(
            "LDGSTS payload completion floor",
            source_ready,
            self.ldgsts_delivery_cycles(work),
        )
    }

    fn ldgsts_destination_service_cycles(&self, wavefronts: u64) -> Result<u64, IssueError> {
        let scaled = wavefronts
            .checked_mul(self.ldgsts_cache_all_destination_service_cycles)
            .ok_or(IssueError::TimeMultiplicationOverflow {
                operation: "LDGSTS aggregate destination-wavefront work",
                lhs: wavefronts,
                rhs: self.ldgsts_cache_all_destination_service_cycles,
            })?;
        Ok(scaled.div_ceil(self.ldgsts_cache_all_destination_service_wavefronts))
    }

    pub const fn max_resident_warps_per_sm(&self) -> u32 {
        self.max_resident_warps_per_sm
    }

    pub(crate) fn validate_resource_source(&self) -> Result<(), IssueError> {
        match self.resource_source {
            IssueResourceSource::ArchitectureDefaults => {
                let hypotheses = self.arch.issue_hypotheses().require("issue_hypotheses")?;
                let shared_additional_wavefront_latency =
                    shared_additional_wavefront_latency(self.arch)?;
                let ldgsts_l1_mshr_floor = ldgsts_l1_mshr_floor(self.arch)?;
                let ldgsts_completion_throughput = ldgsts_completion_throughput(self.arch)?;
                let ldgsts_cache_all_destination_wavefront_cycles =
                    ldgsts_cache_all_destination_wavefront_cycles(self.arch)?;
                let (
                    ldgsts_cache_all_destination_service_wavefronts,
                    ldgsts_cache_all_destination_service_cycles,
                ) = ldgsts_cache_all_destination_service_rate(self.arch)?;
                let fields = [
                    (
                        "shared_additional_wavefront_latency",
                        shared_additional_wavefront_latency,
                        self.shared_additional_wavefront_latency,
                    ),
                    (
                        "ldgsts_l1_mshr_floor",
                        u64::from(ldgsts_l1_mshr_floor.get()),
                        u64::from(self.ldgsts_l1_mshr_floor.get()),
                    ),
                    (
                        "ldgsts_cache_all_payload_bytes_per_cycle",
                        ldgsts_completion_throughput.cache_all_payload_bytes_per_cycle,
                        self.ldgsts_completion_throughput
                            .cache_all_payload_bytes_per_cycle,
                    ),
                    (
                        "ldgsts_bypass_payload_bytes_per_cycle",
                        ldgsts_completion_throughput.bypass_payload_bytes_per_cycle,
                        self.ldgsts_completion_throughput
                            .bypass_payload_bytes_per_cycle,
                    ),
                    (
                        "ldgsts_cache_all_destination_wavefront_cycles",
                        ldgsts_cache_all_destination_wavefront_cycles,
                        self.ldgsts_cache_all_destination_wavefront_cycles,
                    ),
                    (
                        "ldgsts_cache_all_destination_service_wavefronts",
                        ldgsts_cache_all_destination_service_wavefronts,
                        self.ldgsts_cache_all_destination_service_wavefronts,
                    ),
                    (
                        "ldgsts_cache_all_destination_service_cycles",
                        ldgsts_cache_all_destination_service_cycles,
                        self.ldgsts_cache_all_destination_service_cycles,
                    ),
                    (
                        "non_memory_write_barrier_latency",
                        hypotheses.non_memory_write_barrier_latency,
                        self.non_memory_write_barrier_latency,
                    ),
                    (
                        "read_barrier_latency",
                        hypotheses.read_barrier_latency,
                        self.read_barrier_latency,
                    ),
                    (
                        "max_resident_warps_per_sm",
                        u64::from(hypotheses.max_resident_warps_per_sm),
                        u64::from(self.max_resident_warps_per_sm),
                    ),
                ];
                for (field, expected, got) in fields {
                    if got != expected {
                        return Err(IssueError::ArchitectureDefaultMismatch {
                            arch: self.arch,
                            field,
                            expected,
                            got,
                        });
                    }
                }
                Ok(())
            }
            IssueResourceSource::Carded(_) => {
                Err(IssueError::CardedResourcesNotImplemented { arch: self.arch })
            }
            #[cfg(test)]
            IssueResourceSource::SyntheticTestFixture => Ok(()),
        }
    }

    /// Non-memory producer visibility (one copy for BOTH memory modes).
    fn non_memory_ready(&self, t: u64) -> Result<u64, IssueError> {
        checked_cycle_add(
            "non-memory write-barrier visibility",
            t,
            self.non_memory_write_barrier_latency,
        )
    }

    /// Shared-memory visibility. One effective wavefront retains the carded
    /// dependent latency; every additional bank wavefront pays the measured
    /// architecture-local serialization increment. Zero preserves the legacy
    /// and predicated-off fixed-latency behavior.
    fn shared_ready(&self, t: u64, wavefronts: u64) -> Result<u64, IssueError> {
        let additional_wavefronts = wavefronts.saturating_sub(1);
        let additional_latency = self
            .shared_additional_wavefront_latency
            .checked_mul(additional_wavefronts)
            .ok_or(IssueError::TimeMultiplicationOverflow {
                operation: "shared-memory additional-wavefront latency",
                lhs: self.shared_additional_wavefront_latency,
                rhs: additional_wavefronts,
            })?;
        let base_ready = checked_cycle_add(
            "shared-memory base write-barrier visibility",
            t,
            self.shared_write_barrier_latency,
        )?;
        checked_cycle_add(
            "shared-memory wavefront write-barrier visibility",
            base_ready,
            additional_latency,
        )
    }

    /// Store / degenerate-record / L1-bypass visibility: the L2-hit arm.
    /// The single copy both data paths must price identically — the
    /// Const-vs-Shared ablation delta depends on it.
    fn store_visibility_ready(&self, t: u64) -> Result<u64, IssueError> {
        checked_cycle_add(
            "store/bypass write-barrier visibility",
            t,
            self.l1_miss_latency,
        )
    }

    /// Validate that an attached L1 uses this parameter set's single timing
    /// source. This runs at the system composition root as well
    /// as per-core construction so an empty SM map cannot skip the check.
    pub(crate) fn validate_l1_timing(&self, cfg: L1Config) -> Result<(), IssueError> {
        if cfg.hit_latency != self.l1_hit_latency {
            return Err(IssueError::L1TimingMismatch {
                field: "hit",
                expected: self.l1_hit_latency,
                got: cfg.hit_latency,
            });
        }
        if cfg.miss_latency != self.l1_miss_latency {
            return Err(IssueError::L1TimingMismatch {
                field: "miss",
                expected: self.l1_miss_latency,
                got: cfg.miss_latency,
            });
        }
        Ok(())
    }

    /// L1 configuration sharing THIS param set's latencies (single
    /// source — the hit/miss numbers must never exist in two copies
    /// that can drift), with geometry from the pchase card resolvers.
    pub fn l1_config(&self, line_capacity: u64, mshr_count: u32) -> L1Config {
        L1Config {
            arch: self.arch,
            line_capacity,
            mshr_count: Some(mshr_count),
            hit_latency: self.l1_hit_latency,
            miss_latency: self.l1_miss_latency,
        }
    }
}

/// Memory-model mode for one SM replay.
#[derive(Debug, Clone, Copy)]
pub(crate) enum SmMemMode {
    /// No cache model: flat global latency.
    Flat,
    /// Per-SM L1 with constant (L2-hit) fill latency below it.
    L1Const(L1Config),
    /// Per-SM L1 over the SHARED L2: misses become requests/responses.
    L1Shared(L1Config),
}

#[derive(Debug, thiserror::Error)]
pub enum IssueError {
    #[error("issue model is unsupported for deployment architecture {arch}")]
    ArchitectureUnsupported { arch: ArchId },
    #[error(
        "one function declares inconsistent CTA-barrier pipeline configurations: expected {expected:?}, got {got:?}"
    )]
    BarrierPipelineConfigurationMismatch {
        expected: CtaBarrierIssue,
        got: CtaBarrierIssue,
    },
    #[error(
        "carded issue resources are not implemented for architecture {arch}: the executor does not yet consume smsp.count, smsp.issue_ports, occupancy.max_blocks, dep_barriers.count/storage, or dual_issue.supported"
    )]
    CardedResourcesNotImplemented { arch: ArchId },
    #[error("architecture-default mismatch for {arch}: {field} is {got}, expected {expected}")]
    ArchitectureDefaultMismatch {
        arch: ArchId,
        field: &'static str,
        expected: u64,
        got: u64,
    },
    #[error(
        "architecture {arch} has invalid cacheable-LDGSTS L1 miss-admission floor {floor}; expected a positive value"
    )]
    InvalidLdgstsL1MshrFloor { arch: ArchId, floor: u32 },
    #[error(
        "architecture {arch} has invalid LDGSTS completion throughput: cache-all={cache_all_bytes_per_cycle} B/cycle, bypass={bypass_bytes_per_cycle} B/cycle; rates must be positive and divide their maximum"
    )]
    InvalidLdgstsCompletionThroughput {
        arch: ArchId,
        cache_all_bytes_per_cycle: u64,
        bypass_bytes_per_cycle: u64,
    },
    #[error(
        "architecture {arch} has invalid cache-all LDGSTS destination timing: {cycles_per_wavefront} cycles per wavefront; expected a positive value"
    )]
    InvalidLdgstsDestinationWavefrontTiming {
        arch: ArchId,
        cycles_per_wavefront: u64,
    },
    #[error(
        "architecture {arch} has invalid cache-all LDGSTS destination throughput: {service_wavefronts} wavefronts per {service_cycles} SM cycles; both terms must be positive"
    )]
    InvalidLdgstsDestinationWavefrontThroughput {
        arch: ArchId,
        service_wavefronts: u64,
        service_cycles: u64,
    },
    #[error(
        "cache-all LDGSTS exact destination work {completion_work} is not an integral multiple of {work_per_wavefront} work units per wavefront"
    )]
    InvalidAsyncCopyDestinationWork {
        completion_work: u64,
        work_per_wavefront: u64,
    },
    #[error(
        "warp {warp} asynchronous-copy source activity is inconsistent: sectors={sectors}, completion_work={completion_work}"
    )]
    AsyncCopyCompletionWorkMismatch {
        warp: usize,
        sectors: u16,
        completion_work: u64,
    },
    #[error("issue-card validation failed: {0}")]
    Calibration(#[from] CalibError),
    #[error(
        "pc {pc} (block {launch_index}, warp slot {warp_slot}) not in the issue schedule of \
         {function_name:?} — dynamic trace and static table disagree"
    )]
    PcNotInSchedule {
        pc: u32,
        launch_index: i64,
        warp_slot: i32,
        function_name: String,
    },
    #[error(
        "pc {pc} (block {launch_index}, warp slot {warp_slot}) has invalid issue control: {source}"
    )]
    ControlDomain {
        pc: u32,
        launch_index: i64,
        warp_slot: i32,
        #[source]
        source: ControlDomainError,
    },
    #[error(
        "pc {pc} (block {launch_index}, warp slot {warp_slot}) sector span start {off} + length {len} overflows usize"
    )]
    SectorSpanOverflow {
        pc: u32,
        launch_index: i64,
        warp_slot: i32,
        off: usize,
        len: u16,
    },
    #[error(
        "pc {pc} (block {launch_index}, warp slot {warp_slot}) sector span {off}..{end} exceeds arena length {sectors}"
    )]
    SectorSpanOutOfBounds {
        pc: u32,
        launch_index: i64,
        warp_slot: i32,
        off: usize,
        len: u16,
        end: usize,
        sectors: usize,
    },
    #[error(
        "pc {pc} (block {launch_index}, warp slot {warp_slot}) has incoherent atomic topology: sector transactions={sectors}, L2 requests={l2_requests}"
    )]
    AtomicRequestTopologyInvalid {
        pc: u32,
        launch_index: i64,
        warp_slot: i32,
        sectors: u16,
        l2_requests: u16,
    },
    #[error("block {launch_index} has {warps} live warps, which does not fit in u32")]
    LiveWarpCountOutOfRange { launch_index: i64, warps: usize },
    #[error(
        "block {launch_index} has {live_warps} live warps, exceeding architecture {arch} resident-warp limit {max_resident_warps}"
    )]
    BlockResidencyLimitExceeded {
        launch_index: i64,
        live_warps: u32,
        arch: ArchId,
        max_resident_warps: u32,
    },
    #[error(
        "architecture {arch} has invalid shared-memory capacity {max_bytes_per_sm} bytes per SM"
    )]
    SharedMemoryCapacityInvalid { arch: ArchId, max_bytes_per_sm: u64 },
    #[error(
        "block {launch_index} requires {bytes_per_cta} shared-memory bytes, exceeding architecture {arch} per-SM capacity {max_bytes_per_sm}"
    )]
    BlockSharedMemoryLimitExceeded {
        launch_index: i64,
        bytes_per_cta: u64,
        arch: ArchId,
        max_bytes_per_sm: u64,
    },
    #[error(
        "block {launch_index} physical warp slot {warp_slot} is outside architecture {arch} range 0..{physical_warp_slots}"
    )]
    WarpSlotOutOfRange {
        launch_index: i64,
        warp_slot: i32,
        arch: ArchId,
        physical_warp_slots: u32,
    },
    #[error("block {launch_index} assigns physical warp slot {warp_slot} more than once")]
    DuplicateWarpSlot { launch_index: i64, warp_slot: i32 },
    #[error(
        "block {launch_index} released physical warp slot {warp_slot}, but that slot is not resident"
    )]
    ResidentWarpSlotMissing { launch_index: i64, warp_slot: i32 },
    #[error("warp {warp} was selected without a pending issue slot")]
    MissingIssueSlot { warp: usize },
    #[error("{counter} counter overflow")]
    CounterOverflow { counter: &'static str },
    #[error("{counter} counter underflow")]
    CounterUnderflow { counter: &'static str },
    #[error("issuing {add} instruction(s) from count {issued} exceeds replay cardinality {limit}")]
    InstructionCountExceedsReplay { issued: u64, add: u64, limit: u64 },
    #[error(
        "warp {warp} dependence barrier {barrier} pending count {current} + {add} overflows u32"
    )]
    PendingBarrierOverflow {
        warp: usize,
        barrier: u8,
        current: u32,
        add: u32,
    },
    #[error(
        "warp {warp} dependence barrier {barrier} pending count {current} cannot release {remove} unit(s)"
    )]
    PendingBarrierUnderflow {
        warp: usize,
        barrier: u8,
        current: u32,
        remove: u32,
    },
    #[error("warp {warp} exhausted the asynchronous-copy group sequence domain")]
    AsyncCopyGroupSequenceExhausted { warp: usize },
    #[error(
        "warp {warp} asynchronous-copy group {group:?} pending source count {current} + {add} overflows u32"
    )]
    AsyncCopyPendingOverflow {
        warp: usize,
        group: AsyncCopyGroupId,
        current: u32,
        add: u32,
    },
    #[error(
        "warp {warp} asynchronous-copy group {group:?} pending source count {current} cannot release {remove} unit(s)"
    )]
    AsyncCopyPendingUnderflow {
        warp: usize,
        group: AsyncCopyGroupId,
        current: u32,
        remove: u32,
    },
    #[error("warp {warp} has no live asynchronous-copy group {group:?}")]
    AsyncCopyGroupNotFound {
        warp: usize,
        group: AsyncCopyGroupId,
    },
    #[error(
        "warp {warp} asynchronous-copy group {group:?} already observes bypass copy instruction {instruction}"
    )]
    AsyncCopyObservationDuplicate {
        warp: usize,
        group: AsyncCopyGroupId,
        instruction: usize,
    },
    #[error(
        "warp {warp} asynchronous-copy group {group:?} has no observed bypass copy instruction {instruction}"
    )]
    AsyncCopyObservationNotFound {
        warp: usize,
        group: AsyncCopyGroupId,
        instruction: usize,
    },
    #[error(
        "warp {warp} asynchronous-copy group {group:?} bypass copy instruction {instruction} pending source count {current} cannot release {remove} unit(s)"
    )]
    AsyncCopyObservationPendingUnderflow {
        warp: usize,
        group: AsyncCopyGroupId,
        instruction: usize,
        current: u32,
        remove: u32,
    },
    #[error(
        "warp {warp} asynchronous-copy group {group:?} bypass copy instruction {instruction} still has {pending} source unit(s) at wait"
    )]
    AsyncCopyObservationPendingAtWait {
        warp: usize,
        group: AsyncCopyGroupId,
        instruction: usize,
        pending: u32,
    },
    #[error("response-owner identity domain is exhausted")]
    ResponseOwnerExhausted,
    #[error("memory responses are invalid for the {mode} memory mode")]
    ResponseModeMismatch { mode: &'static str },
    #[error("memory response fill {fill:?} is not in flight in the L1")]
    ResponseFillNotInFlight { fill: FillKey },
    #[error("line fill {line} has no live sector-mask bookkeeping")]
    FillMaskMissing { line: i64 },
    #[error("line {line} has sector-mask bookkeeping without an in-flight fill")]
    FillMaskWithoutFill { line: i64 },
    #[error("response-bearing request token {reply:?} is already outstanding")]
    ResponseTokenAlreadyOutstanding { reply: ReplyTo },
    #[error("memory response token {reply:?} disappeared after successful preflight")]
    ResponseTokenRemovalFailed { reply: ReplyTo },
    #[error("memory response targets warp index {warp}, but this SM has {warps} admitted warps")]
    ResponseWarpOutOfRange { warp: usize, warps: usize },
    #[error("memory response token {reply:?} was not emitted or is no longer outstanding")]
    ResponseTokenNotOutstanding { reply: ReplyTo },
    #[error(
        "memory response targets warp {warp} dependence barrier {barrier}, which is not pending"
    )]
    ResponseBarrierNotPending { warp: usize, barrier: u8 },
    #[error("mutation-only memory request cannot carry a response event")]
    UnexpectedMutationOnlyResponse,
    #[error(
        "no schedulable warp, no pending timer and no outstanding response at t={time} with \
         {unfinished} warp(s) unfinished — issue-model invariant broken"
    )]
    NoProgress { time: u64, unfinished: usize },
    #[error("issue architecture mismatch for {component}: expected {expected}, got {got}")]
    ArchitectureMismatch {
        component: &'static str,
        expected: ArchId,
        got: ArchId,
    },
    #[error("L1 {field} latency {got} does not match issue parameter {expected}")]
    L1TimingMismatch {
        field: &'static str,
        expected: u64,
        got: u64,
    },
    #[error("{name} latency must be finite, positive, and fit in u64 (got {value})")]
    InvalidLatency { name: &'static str, value: f64 },
    #[error("{operation}: cycle addition {base} + {delta} overflows u64")]
    TimeOverflow {
        operation: &'static str,
        base: u64,
        delta: u64,
    },
    #[error("{operation}: cycle multiplication {lhs} * {rhs} overflows u64")]
    TimeMultiplicationOverflow {
        operation: &'static str,
        lhs: u64,
        rhs: u64,
    },
    #[error("memory response at t={now} precedes the SM clock t={clock}")]
    ResponseBeforeClock { now: u64, clock: u64 },
    #[error("advance horizon t={horizon} precedes the SM clock t={clock}")]
    HorizonBeforeClock { horizon: u64, clock: u64 },

    #[error("issue capability unavailable: {0}")]
    Capability(#[from] sim_arch::CapabilityError),
    #[error("L1 access failed: {0}")]
    L1Access(#[from] L1AccessError),
    #[error("L1 configuration is invalid: {0}")]
    L1Config(#[from] sim_mem::L1ConfigError),
    #[error("memory-service candidate configuration is invalid: {0}")]
    MemoryServiceCandidate(#[from] MemoryServiceCandidateError),
    #[error("memory-admission candidate configuration is invalid: {0}")]
    MemoryAdmissionCandidate(#[from] MemoryAdmissionCandidateError),
    #[error("feedback admission queue on subcore {subcore} is full at t={time}")]
    MemoryAdmissionFeedbackQueueFull { subcore: usize, time: u64 },
    #[error(
        "feedback admission candidate {candidate_id} declares operand-visibility latency {configured_cycles} cycles, but the active issue model uses {issue_model_cycles} cycles"
    )]
    MemoryAdmissionFeedbackLatencyMismatch {
        candidate_id: u8,
        configured_cycles: u32,
        issue_model_cycles: u64,
    },
    #[error("feedback admission reservation is absent for warp {warp} instruction {instruction}")]
    MemoryAdmissionFeedbackReservationMissing { warp: u32, instruction: u32 },
    #[error(
        "feedback admission reservation was already priced for warp {warp} instruction {instruction}"
    )]
    MemoryAdmissionFeedbackAlreadyPriced { warp: u32, instruction: u32 },
    #[error("issue diagnostic update failed: {0}")]
    ReadinessDiagnostic(#[from] ReadinessCounterError),
}

/// Memory-hierarchy effects deliberately outside the modeled event stream.
/// These flags are capability boundaries, not synthetic traffic counters: a
/// set bit prevents consumers from mistaking a partial write-class model for a
/// complete RMW/writeback/coherence implementation.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TailFlags {
    /// RED/REDG's read-modify-write read arm is not modeled.
    pub red_rmw: bool,
    /// Dirty eviction/writeback traffic is not modeled.
    pub dirty_writeback: bool,
    /// Coherence propagation and invalidation traffic is not modeled.
    pub coherence: bool,
    /// The modeled LDGSTS commit/wait chain cannot establish destination
    /// visibility: either the trace omitted destination payload or the bypass
    /// route has no promoted per-copy destination timing mechanism.
    pub ldgsts_dst_shared_timing: bool,
}

impl TailFlags {
    /// True only when replay has no known timing-bearing hierarchy effect
    /// outside the modeled event stream.
    pub const fn is_empty(self) -> bool {
        !self.red_rmw && !self.dirty_writeback && !self.coherence && !self.ldgsts_dst_shared_timing
    }

    pub const fn union(self, other: Self) -> Self {
        Self {
            red_rmw: self.red_rmw || other.red_rmw,
            dirty_writeback: self.dirty_writeback || other.dirty_writeback,
            coherence: self.coherence || other.coherence,
            ldgsts_dst_shared_timing: self.ldgsts_dst_shared_timing
                || other.ldgsts_dst_shared_timing,
        }
    }

    const fn for_access(kind: AccessKind) -> Self {
        match kind {
            AccessKind::GlobalReduction { .. } => Self {
                red_rmw: true,
                dirty_writeback: true,
                coherence: true,
                ldgsts_dst_shared_timing: false,
            },
            AccessKind::GlobalStore { .. } | AccessKind::LocalStore { .. } => Self {
                red_rmw: false,
                dirty_writeback: true,
                coherence: true,
                ldgsts_dst_shared_timing: false,
            },
            AccessKind::GlobalBypass {
                token: CacheOpToken::Atom,
                ..
            } => Self {
                red_rmw: false,
                dirty_writeback: true,
                coherence: true,
                ldgsts_dst_shared_timing: false,
            },
            AccessKind::NonMemory
            | AccessKind::PredicatedOffMemory
            | AccessKind::Shared { .. }
            | AccessKind::LocalLoad { .. }
            | AccessKind::GlobalEligible { .. }
            | AccessKind::GlobalBypass { .. }
            | AccessKind::GlobalAsyncCopy { .. }
            | AccessKind::GlobalUntagged { .. } => Self {
                red_rmw: false,
                dirty_writeback: false,
                coherence: false,
                ldgsts_dst_shared_timing: false,
            },
        }
    }
}

/// Semantic operation carried through the shared-memory request protocol.
/// L2 hit policy is derived from this type so Store, returning ATOM, and
/// non-returning RED cannot collapse into one `WriteAtomic` request class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOperation {
    Read,
    Store,
    Atomic,
    Reduction,
}

impl MemoryOperation {
    pub(crate) const fn l2_class(self) -> L2OpClass {
        match self {
            Self::Read => L2OpClass::Read,
            Self::Store | Self::Atomic | Self::Reduction => L2OpClass::WriteAtomic,
        }
    }

    pub(crate) const fn fetches_missing_data(self) -> bool {
        matches!(self, Self::Read | Self::Atomic)
    }
}

#[derive(Debug, thiserror::Error, Clone, Copy, PartialEq, Eq)]
pub enum ReadinessCounterError {
    #[error(
        "readiness diagnostic counter {field} overflowed while adding {addend} to {accumulated}"
    )]
    Overflow {
        field: &'static str,
        accumulated: u64,
        addend: u64,
    },
    #[error("memory-service candidate configuration differs at index {index}")]
    MemoryServiceCandidateMismatch { index: usize },
    #[error("memory-admission candidate configuration differs at index {index}")]
    MemoryAdmissionCandidateMismatch { index: usize },
}

fn checked_readiness_add(
    field: &'static str,
    accumulated: u64,
    addend: u64,
) -> Result<u64, ReadinessCounterError> {
    accumulated
        .checked_add(addend)
        .ok_or(ReadinessCounterError::Overflow {
            field,
            accumulated,
            addend,
        })
}

const fn is_zero_u64(value: &u64) -> bool {
    *value == 0
}

const fn is_zero_u32(value: &u32) -> bool {
    *value == 0
}

const fn is_zero_u16(value: &u16) -> bool {
    *value == 0
}

const fn is_false(value: &bool) -> bool {
    !*value
}

/// Exclusive and overlapping readiness delay by causal simulator relation.
///
/// `ReadinessDiagnostics::limiting_cycles` is exclusive: one deterministic
/// cause owns each issued instruction's pre-ready delay. `constraint_exposure`
/// retains every active relation and may overlap, so its fields must not be
/// summed as elapsed kernel time. `eligible_not_selected` is the delay after a
/// warp's complete issue frontier was ready but before the scheduler selected
/// it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ReadinessCauseCycles {
    pub control_stall: u64,
    pub dependence_counter_read: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub dependence_counter_global_load_operand_read: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub dependence_counter_global_store_operand_read: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub dependence_counter_global_atomic_operand_read: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub dependence_counter_global_reduction_operand_read: u64,
    pub dependence_counter_non_memory_write: u64,
    pub dependence_counter_shared_write: u64,
    pub dependence_counter_l1tex: u64,
    pub dependence_counter_unknown_origin: u64,
    pub async_source: u64,
    pub async_destination_local: u64,
    pub async_destination_sm_service: u64,
    pub cta_barrier_rendezvous: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub pipeline_barrier: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub pipeline_branch: u64,
    pub pipeline_tensor: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub memory_admission_queue: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub memory_write_completion: u64,
    pub eligible_not_selected: u64,
}

impl ReadinessCauseCycles {
    fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        let mut next = *self;
        macro_rules! add {
            ($field:ident) => {
                next.$field = checked_readiness_add(
                    concat!("readiness.", stringify!($field)),
                    next.$field,
                    addend.$field,
                )?;
            };
        }
        add!(control_stall);
        add!(dependence_counter_read);
        add!(dependence_counter_global_load_operand_read);
        add!(dependence_counter_global_store_operand_read);
        add!(dependence_counter_global_atomic_operand_read);
        add!(dependence_counter_global_reduction_operand_read);
        add!(dependence_counter_non_memory_write);
        add!(dependence_counter_shared_write);
        add!(dependence_counter_l1tex);
        add!(dependence_counter_unknown_origin);
        add!(async_source);
        add!(async_destination_local);
        add!(async_destination_sm_service);
        add!(cta_barrier_rendezvous);
        add!(pipeline_barrier);
        add!(pipeline_branch);
        add!(pipeline_tensor);
        add!(memory_admission_queue);
        add!(memory_write_completion);
        add!(eligible_not_selected);
        *self = next;
        Ok(())
    }
}

fn observe_readiness_constraint(
    projected: &mut ReadinessDiagnostics,
    base: u64,
    until: u64,
    cause: ReadinessCause,
    ready_at: &mut u64,
    limiting: &mut Option<ReadinessCause>,
    tied: &mut u64,
) -> Result<(), IssueError> {
    if until > base {
        projected
            .constraint_exposure_cycles
            .add_cause(cause, until - base)?;
    }
    if until > *ready_at {
        *ready_at = until;
        *limiting = (until > base).then_some(cause);
        *tied = 0;
    } else if until == *ready_at && until > base {
        match *limiting {
            Some(existing) if existing != cause => {
                *tied = checked_u64_counter_add("tied readiness cause", *tied, 1)?;
                *limiting = Some(existing.min(cause));
            }
            None => *limiting = Some(cause),
            Some(_) => {}
        }
    }
    Ok(())
}

fn dependency_origin(kind: AccessKind) -> DependencyOrigin {
    match kind {
        AccessKind::NonMemory | AccessKind::PredicatedOffMemory => DependencyOrigin::NonMemoryWrite,
        AccessKind::Shared { .. } => DependencyOrigin::SharedWrite,
        AccessKind::LocalLoad { .. }
        | AccessKind::LocalStore { .. }
        | AccessKind::GlobalEligible { .. }
        | AccessKind::GlobalBypass { .. }
        | AccessKind::GlobalAsyncCopy { .. }
        | AccessKind::GlobalUntagged { .. }
        | AccessKind::GlobalStore { .. }
        | AccessKind::GlobalReduction { .. } => DependencyOrigin::L1Tex,
    }
}

fn operand_read_dependency_origin(kind: AccessKind) -> DependencyOrigin {
    match kind {
        AccessKind::GlobalBypass {
            token: CacheOpToken::Atom,
            ..
        } => DependencyOrigin::GlobalAtomicOperandRead,
        AccessKind::GlobalBypass {
            token: CacheOpToken::Ldgsts,
            ..
        } => DependencyOrigin::Read,
        AccessKind::GlobalEligible { .. }
        | AccessKind::GlobalUntagged { .. }
        | AccessKind::GlobalBypass { .. } => DependencyOrigin::GlobalLoadOperandRead,
        AccessKind::GlobalStore { .. } => DependencyOrigin::GlobalStoreOperandRead,
        AccessKind::GlobalReduction { .. } => DependencyOrigin::GlobalReductionOperandRead,
        AccessKind::NonMemory
        | AccessKind::PredicatedOffMemory
        | AccessKind::Shared { .. }
        | AccessKind::LocalLoad { .. }
        | AccessKind::LocalStore { .. }
        | AccessKind::GlobalAsyncCopy { .. } => DependencyOrigin::Read,
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct AsyncReadinessDiagnostics {
    pub groups_committed: u64,
    pub groups_waited: u64,
    pub source_limited_groups: u64,
    pub destination_local_limited_groups: u64,
    pub destination_sm_service_limited_groups: u64,
    pub source_cycles: u64,
    pub destination_local_cycles: u64,
    pub destination_sm_service_cycles: u64,
    pub completion_hidden_before_wait_cycles: u64,
}

impl AsyncReadinessDiagnostics {
    fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        let mut next = *self;
        macro_rules! add {
            ($field:ident) => {
                next.$field = checked_readiness_add(
                    concat!("readiness.async_groups.", stringify!($field)),
                    next.$field,
                    addend.$field,
                )?;
            };
        }
        add!(groups_committed);
        add!(groups_waited);
        add!(source_limited_groups);
        add!(destination_local_limited_groups);
        add!(destination_sm_service_limited_groups);
        add!(source_cycles);
        add!(destination_local_cycles);
        add!(destination_sm_service_cycles);
        add!(completion_hidden_before_wait_cycles);
        *self = next;
        Ok(())
    }
}

/// Cold, bypass-only per-copy observation. It is boxed out of the scheduler's
/// inline readiness object and never participates in issue or wait timing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct BypassAsyncCopyDiagnostics {
    pub copies_observed: u64,
    /// Compatibility projection: source cohorts multiplied by each copy's
    /// maximum bank multiplicity. Exact for the retained uniform cohorts.
    pub destination_wavefronts_observed: u64,
    /// Source cohorts multiplied by each copy's maximum bypass projection.
    /// Exact for the retained uniform cohorts.
    pub bypass_wavefronts_observed: u64,
    pub source_cohorts_observed: u64,
    pub destination_lines_observed: u64,
    pub max_bank_wavefronts_sum: u64,
    pub max_bypass_wavefronts_sum: u64,
    pub copy_source_spread_cycles: u64,
    pub per_copy_completion_cycles: u64,
    pub group_aggregate_excess_cycles: u64,
    pub topology_candidate_penalty_cycles: u64,
    pub topology_candidate_completion_cycles: u64,
    pub topology_candidate_later_cycles: u64,
    pub topology_candidate_earlier_cycles: u64,
    /// Timing-neutral destination-pipeline groups projected at a wait. The
    /// candidate fields below are sums across groups, never readiness inputs.
    pub pipeline_groups_observed: u64,
    /// Candidate completion measured from group commit for acceptance width 1.
    pub pipeline_q1_completion_cycles: u64,
    /// Candidate completion still exposed when the wait became current.
    pub pipeline_q1_exposure_cycles: u64,
    /// Candidate completion measured from group commit for acceptance width 2.
    pub pipeline_q2_completion_cycles: u64,
    /// Candidate completion still exposed when the wait became current.
    pub pipeline_q2_exposure_cycles: u64,
    /// Candidate completion measured from group commit for acceptance width 4.
    pub pipeline_q4_completion_cycles: u64,
    /// Candidate completion still exposed when the wait became current.
    pub pipeline_q4_exposure_cycles: u64,
    /// Candidate completion measured from group commit for acceptance width 8.
    pub pipeline_q8_completion_cycles: u64,
    /// Candidate completion still exposed when the wait became current.
    pub pipeline_q8_exposure_cycles: u64,
    /// Candidate completion measured from group commit for acceptance width 16.
    pub pipeline_q16_completion_cycles: u64,
    /// Candidate completion still exposed when the wait became current.
    pub pipeline_q16_exposure_cycles: u64,
}

impl BypassAsyncCopyDiagnostics {
    pub const fn is_empty(&self) -> bool {
        self.copies_observed == 0
    }

    fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        let next = Self {
            copies_observed: checked_readiness_add(
                "readiness.bypass_async_copies.copies_observed",
                self.copies_observed,
                addend.copies_observed,
            )?,
            destination_wavefronts_observed: checked_readiness_add(
                "readiness.bypass_async_copies.destination_wavefronts_observed",
                self.destination_wavefronts_observed,
                addend.destination_wavefronts_observed,
            )?,
            bypass_wavefronts_observed: checked_readiness_add(
                "readiness.bypass_async_copies.bypass_wavefronts_observed",
                self.bypass_wavefronts_observed,
                addend.bypass_wavefronts_observed,
            )?,
            source_cohorts_observed: checked_readiness_add(
                "readiness.bypass_async_copies.source_cohorts_observed",
                self.source_cohorts_observed,
                addend.source_cohorts_observed,
            )?,
            destination_lines_observed: checked_readiness_add(
                "readiness.bypass_async_copies.destination_lines_observed",
                self.destination_lines_observed,
                addend.destination_lines_observed,
            )?,
            max_bank_wavefronts_sum: checked_readiness_add(
                "readiness.bypass_async_copies.max_bank_wavefronts_sum",
                self.max_bank_wavefronts_sum,
                addend.max_bank_wavefronts_sum,
            )?,
            max_bypass_wavefronts_sum: checked_readiness_add(
                "readiness.bypass_async_copies.max_bypass_wavefronts_sum",
                self.max_bypass_wavefronts_sum,
                addend.max_bypass_wavefronts_sum,
            )?,
            copy_source_spread_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.copy_source_spread_cycles",
                self.copy_source_spread_cycles,
                addend.copy_source_spread_cycles,
            )?,
            per_copy_completion_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.per_copy_completion_cycles",
                self.per_copy_completion_cycles,
                addend.per_copy_completion_cycles,
            )?,
            group_aggregate_excess_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.group_aggregate_excess_cycles",
                self.group_aggregate_excess_cycles,
                addend.group_aggregate_excess_cycles,
            )?,
            topology_candidate_penalty_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.topology_candidate_penalty_cycles",
                self.topology_candidate_penalty_cycles,
                addend.topology_candidate_penalty_cycles,
            )?,
            topology_candidate_completion_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.topology_candidate_completion_cycles",
                self.topology_candidate_completion_cycles,
                addend.topology_candidate_completion_cycles,
            )?,
            topology_candidate_later_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.topology_candidate_later_cycles",
                self.topology_candidate_later_cycles,
                addend.topology_candidate_later_cycles,
            )?,
            topology_candidate_earlier_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.topology_candidate_earlier_cycles",
                self.topology_candidate_earlier_cycles,
                addend.topology_candidate_earlier_cycles,
            )?,
            pipeline_groups_observed: checked_readiness_add(
                "readiness.bypass_async_copies.pipeline_groups_observed",
                self.pipeline_groups_observed,
                addend.pipeline_groups_observed,
            )?,
            pipeline_q1_completion_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.pipeline_q1_completion_cycles",
                self.pipeline_q1_completion_cycles,
                addend.pipeline_q1_completion_cycles,
            )?,
            pipeline_q1_exposure_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.pipeline_q1_exposure_cycles",
                self.pipeline_q1_exposure_cycles,
                addend.pipeline_q1_exposure_cycles,
            )?,
            pipeline_q2_completion_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.pipeline_q2_completion_cycles",
                self.pipeline_q2_completion_cycles,
                addend.pipeline_q2_completion_cycles,
            )?,
            pipeline_q2_exposure_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.pipeline_q2_exposure_cycles",
                self.pipeline_q2_exposure_cycles,
                addend.pipeline_q2_exposure_cycles,
            )?,
            pipeline_q4_completion_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.pipeline_q4_completion_cycles",
                self.pipeline_q4_completion_cycles,
                addend.pipeline_q4_completion_cycles,
            )?,
            pipeline_q4_exposure_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.pipeline_q4_exposure_cycles",
                self.pipeline_q4_exposure_cycles,
                addend.pipeline_q4_exposure_cycles,
            )?,
            pipeline_q8_completion_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.pipeline_q8_completion_cycles",
                self.pipeline_q8_completion_cycles,
                addend.pipeline_q8_completion_cycles,
            )?,
            pipeline_q8_exposure_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.pipeline_q8_exposure_cycles",
                self.pipeline_q8_exposure_cycles,
                addend.pipeline_q8_exposure_cycles,
            )?,
            pipeline_q16_completion_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.pipeline_q16_completion_cycles",
                self.pipeline_q16_completion_cycles,
                addend.pipeline_q16_completion_cycles,
            )?,
            pipeline_q16_exposure_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.pipeline_q16_exposure_cycles",
                self.pipeline_q16_exposure_cycles,
                addend.pipeline_q16_exposure_cycles,
            )?,
        };
        *self = next;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct CtaBarrierDiagnostics {
    pub generations: u64,
    pub total_arrival_spread_cycles: u64,
    pub max_arrival_spread_cycles: u64,
    pub total_rendezvous_warp_cycles: u64,
    pub releases_after_participant_retirement: u64,
}

impl CtaBarrierDiagnostics {
    fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        let mut next = *self;
        macro_rules! add {
            ($field:ident) => {
                next.$field = checked_readiness_add(
                    concat!("readiness.cta_barriers.", stringify!($field)),
                    next.$field,
                    addend.$field,
                )?;
            };
        }
        add!(generations);
        add!(total_arrival_spread_cycles);
        next.max_arrival_spread_cycles = next
            .max_arrival_spread_cycles
            .max(addend.max_arrival_spread_cycles);
        add!(total_rendezvous_warp_cycles);
        add!(releases_after_participant_retirement);
        *self = next;
        Ok(())
    }
}

/// Bounded schema-2 diagnostics for the SM-shared CTA-barrier issue server.
/// Timestamp sums and issue spans preserve first/last-generation information
/// without retaining an instruction- or cycle-sized event log.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct BarrierPipelineDiagnostics {
    pub instructions_issued: u64,
    pub issue_server_busy_cycles: u64,
    pub generations_completed: u64,
    pub generation_rearm_events: u64,
    pub first_issue_timestamp_sum: u64,
    pub last_issue_timestamp_sum: u64,
    pub total_generation_issue_span_cycles: u64,
    pub max_generation_issue_span_cycles: u64,
    pub first_arrival_timestamp_sum: u64,
    pub last_arrival_timestamp_sum: u64,
    pub total_generation_arrival_span_cycles: u64,
    pub max_generation_arrival_span_cycles: u64,
}

impl BarrierPipelineDiagnostics {
    pub const fn is_empty(&self) -> bool {
        self.instructions_issued == 0
            && self.issue_server_busy_cycles == 0
            && self.generations_completed == 0
            && self.generation_rearm_events == 0
            && self.first_issue_timestamp_sum == 0
            && self.last_issue_timestamp_sum == 0
            && self.total_generation_issue_span_cycles == 0
            && self.max_generation_issue_span_cycles == 0
            && self.first_arrival_timestamp_sum == 0
            && self.last_arrival_timestamp_sum == 0
            && self.total_generation_arrival_span_cycles == 0
            && self.max_generation_arrival_span_cycles == 0
    }

    fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        let mut next = *self;
        macro_rules! add {
            ($field:ident) => {
                next.$field = checked_readiness_add(
                    concat!("readiness.barrier_pipeline.", stringify!($field)),
                    next.$field,
                    addend.$field,
                )?;
            };
        }
        add!(instructions_issued);
        add!(issue_server_busy_cycles);
        add!(generations_completed);
        add!(generation_rearm_events);
        add!(first_issue_timestamp_sum);
        add!(last_issue_timestamp_sum);
        add!(total_generation_issue_span_cycles);
        next.max_generation_issue_span_cycles = next
            .max_generation_issue_span_cycles
            .max(addend.max_generation_issue_span_cycles);
        add!(first_arrival_timestamp_sum);
        add!(last_arrival_timestamp_sum);
        add!(total_generation_arrival_span_cycles);
        next.max_generation_arrival_span_cycles = next
            .max_generation_arrival_span_cycles
            .max(addend.max_generation_arrival_span_cycles);
        *self = next;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ControlPipelineDiagnostics {
    pub branch_instructions_issued: u64,
    pub branch_resolution_cycles: u64,
}

impl ControlPipelineDiagnostics {
    pub const fn is_empty(&self) -> bool {
        self.branch_instructions_issued == 0 && self.branch_resolution_cycles == 0
    }

    fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        let next = Self {
            branch_instructions_issued: checked_readiness_add(
                "readiness.control_pipeline.branch_instructions_issued",
                self.branch_instructions_issued,
                addend.branch_instructions_issued,
            )?,
            branch_resolution_cycles: checked_readiness_add(
                "readiness.control_pipeline.branch_resolution_cycles",
                self.branch_resolution_cycles,
                addend.branch_resolution_cycles,
            )?,
        };
        *self = next;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ResultWritebackDiagnostics {
    pub fixed_over_l1tex_collisions: u64,
    pub l1tex_deferral_cycles: u64,
}

/// Issue-time interval carried by kernels that bracket a measured region with
/// `CS2R ..., SR_CLOCKLO`. This is a simulator diagnostic, not a hardware clock
/// emulation: it reports scheduler issue timestamps in the same first/last
/// boundary shape used by in-kernel CLOCK microbenchmarks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ClockIssueDiagnostics {
    pub reads_issued: u64,
    pub warps_with_interval: u64,
    pub maximum_first_to_last_issue_cycles: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub warps_with_three_reads: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub maximum_first_to_second_issue_cycles: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub maximum_second_to_third_issue_cycles: u64,
}

/// Trace-derived scalar-global memory work presented by issued warp
/// instructions. Ideal wavefronts use the exact sm_86 shape observed by the
/// memory-retirement campaign: ordinary sectors pack four per wavefront, while
/// ATOM/RED additionally preserve same-sector alias multiplicity. Unique 128 B
/// lines and exact atomic L2 requests remain separate work dimensions. Hardware may
/// emit extra output wavefronts on internal replay/conflict; those are not
/// inferable from lane addresses. This diagnostic activates no timing service.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct GlobalMemoryIssueDiagnostics {
    pub load_instructions: u64,
    pub store_instructions: u64,
    pub atomic_instructions: u64,
    pub reduction_instructions: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub load_operand_read_barrier_instructions: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub store_operand_read_barrier_instructions: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub atomic_operand_read_barrier_instructions: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub reduction_operand_read_barrier_instructions: u64,
    pub load_sectors: u64,
    pub store_sectors: u64,
    pub atomic_sectors: u64,
    pub reduction_sectors: u64,
    pub load_unique_lines: u64,
    pub store_unique_lines: u64,
    pub atomic_unique_lines: u64,
    pub reduction_unique_lines: u64,
    pub atomic_l2_requests: u64,
    pub reduction_l2_requests: u64,
    pub load_ideal_wavefronts: u64,
    pub store_ideal_wavefronts: u64,
    pub atomic_ideal_wavefronts: u64,
    pub reduction_ideal_wavefronts: u64,
    pub first_issue_timestamp_sum: u64,
    pub last_issue_timestamp_sum: u64,
    pub total_issue_span_cycles: u64,
    pub max_issue_span_cycles: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GlobalMemoryIssueClass {
    Load,
    Store,
    Atomic,
    Reduction,
}

fn maximum_sector_multiplicity(sectors: &[i64]) -> u64 {
    sectors
        .iter()
        .enumerate()
        .map(|(index, sector)| {
            1 + sectors[index + 1..]
                .iter()
                .filter(|candidate| *candidate == sector)
                .count() as u64
        })
        .max()
        .unwrap_or(0)
}

fn unique_128b_lines(sectors: &[i64]) -> u64 {
    sectors
        .iter()
        .enumerate()
        .filter(|(index, sector)| {
            let line = sector.div_euclid(4);
            !sectors[..*index]
                .iter()
                .any(|prior| prior.div_euclid(4) == line)
        })
        .count() as u64
}

fn projected_global_memory_issue(
    kind: AccessKind,
    sector_arena: &[i64],
) -> Option<(GlobalMemoryIssueClass, u64, u64, u64, u64)> {
    let atomic_l2_requests = kind.atomic_l2_requests().map(u64::from).unwrap_or(0);
    let (class, off, len, alias_aware) = match kind {
        AccessKind::GlobalEligible { off, len, .. } | AccessKind::GlobalUntagged { off, len } => {
            (GlobalMemoryIssueClass::Load, off, len, false)
        }
        AccessKind::GlobalBypass {
            off,
            len,
            token: CacheOpToken::Atom,
            ..
        } => (GlobalMemoryIssueClass::Atomic, off, len, true),
        AccessKind::GlobalBypass {
            token: CacheOpToken::Ldgsts,
            ..
        }
        | AccessKind::GlobalAsyncCopy { .. } => {
            // LDGSTS uses its own exact source/destination work domains.
            return None;
        }
        AccessKind::GlobalBypass { off, len, .. } => {
            (GlobalMemoryIssueClass::Load, off, len, false)
        }
        AccessKind::GlobalStore { off, len } => (GlobalMemoryIssueClass::Store, off, len, false),
        AccessKind::GlobalReduction { off, len, .. } => {
            (GlobalMemoryIssueClass::Reduction, off, len, true)
        }
        AccessKind::NonMemory
        | AccessKind::PredicatedOffMemory
        | AccessKind::Shared { .. }
        | AccessKind::LocalLoad { .. }
        | AccessKind::LocalStore { .. } => return None,
    };
    if len == 0 {
        return None;
    }
    let sectors = &sector_arena[off..off + usize::from(len)];
    let sector_count = u64::from(len);
    let packed = sector_count.div_ceil(4);
    let wavefronts = if alias_aware {
        packed.max(maximum_sector_multiplicity(sectors))
    } else {
        packed
    };
    Some((
        class,
        sector_count,
        wavefronts,
        unique_128b_lines(sectors),
        atomic_l2_requests,
    ))
}

impl GlobalMemoryIssueDiagnostics {
    const fn has_memory_work(&self) -> bool {
        self.load_instructions != 0
            || self.store_instructions != 0
            || self.atomic_instructions != 0
            || self.reduction_instructions != 0
    }

    pub const fn is_empty(&self) -> bool {
        !self.has_memory_work() && !self.has_operand_read_barriers()
    }

    pub const fn has_operand_read_barriers(&self) -> bool {
        self.load_operand_read_barrier_instructions != 0
            || self.store_operand_read_barrier_instructions != 0
            || self.atomic_operand_read_barrier_instructions != 0
            || self.reduction_operand_read_barrier_instructions != 0
    }

    fn observe(
        &mut self,
        class: GlobalMemoryIssueClass,
        sectors: u64,
        wavefronts: u64,
        unique_lines: u64,
        atomic_l2_requests: u64,
        operand_read_barrier: bool,
        now: u64,
    ) -> Result<(), ReadinessCounterError> {
        let was_empty = !self.has_memory_work();
        let mut next = *self;
        macro_rules! add_class {
            ($prefix:literal, $instructions:ident, $sector_field:ident, $wavefront_field:ident) => {{
                next.$instructions = checked_readiness_add(
                    concat!("readiness.global_memory_issue.", $prefix, "_instructions"),
                    next.$instructions,
                    1,
                )?;
                next.$sector_field = checked_readiness_add(
                    concat!("readiness.global_memory_issue.", $prefix, "_sectors"),
                    next.$sector_field,
                    sectors,
                )?;
                next.$wavefront_field = checked_readiness_add(
                    concat!(
                        "readiness.global_memory_issue.",
                        $prefix,
                        "_ideal_wavefronts"
                    ),
                    next.$wavefront_field,
                    wavefronts,
                )?;
            }};
        }
        match class {
            GlobalMemoryIssueClass::Load => {
                add_class!(
                    "load",
                    load_instructions,
                    load_sectors,
                    load_ideal_wavefronts
                )
            }
            GlobalMemoryIssueClass::Store => add_class!(
                "store",
                store_instructions,
                store_sectors,
                store_ideal_wavefronts
            ),
            GlobalMemoryIssueClass::Atomic => add_class!(
                "atomic",
                atomic_instructions,
                atomic_sectors,
                atomic_ideal_wavefronts
            ),
            GlobalMemoryIssueClass::Reduction => add_class!(
                "reduction",
                reduction_instructions,
                reduction_sectors,
                reduction_ideal_wavefronts
            ),
        }
        if operand_read_barrier {
            let (field, counter) = match class {
                GlobalMemoryIssueClass::Load => (
                    "readiness.global_memory_issue.load_operand_read_barrier_instructions",
                    &mut next.load_operand_read_barrier_instructions,
                ),
                GlobalMemoryIssueClass::Store => (
                    "readiness.global_memory_issue.store_operand_read_barrier_instructions",
                    &mut next.store_operand_read_barrier_instructions,
                ),
                GlobalMemoryIssueClass::Atomic => (
                    "readiness.global_memory_issue.atomic_operand_read_barrier_instructions",
                    &mut next.atomic_operand_read_barrier_instructions,
                ),
                GlobalMemoryIssueClass::Reduction => (
                    "readiness.global_memory_issue.reduction_operand_read_barrier_instructions",
                    &mut next.reduction_operand_read_barrier_instructions,
                ),
            };
            *counter = checked_readiness_add(field, *counter, 1)?;
        }
        match class {
            GlobalMemoryIssueClass::Load => {
                next.load_unique_lines = checked_readiness_add(
                    "readiness.global_memory_issue.load_unique_lines",
                    next.load_unique_lines,
                    unique_lines,
                )?;
            }
            GlobalMemoryIssueClass::Store => {
                next.store_unique_lines = checked_readiness_add(
                    "readiness.global_memory_issue.store_unique_lines",
                    next.store_unique_lines,
                    unique_lines,
                )?;
            }
            GlobalMemoryIssueClass::Atomic => {
                next.atomic_unique_lines = checked_readiness_add(
                    "readiness.global_memory_issue.atomic_unique_lines",
                    next.atomic_unique_lines,
                    unique_lines,
                )?;
                next.atomic_l2_requests = checked_readiness_add(
                    "readiness.global_memory_issue.atomic_l2_requests",
                    next.atomic_l2_requests,
                    atomic_l2_requests,
                )?;
            }
            GlobalMemoryIssueClass::Reduction => {
                next.reduction_unique_lines = checked_readiness_add(
                    "readiness.global_memory_issue.reduction_unique_lines",
                    next.reduction_unique_lines,
                    unique_lines,
                )?;
                next.reduction_l2_requests = checked_readiness_add(
                    "readiness.global_memory_issue.reduction_l2_requests",
                    next.reduction_l2_requests,
                    atomic_l2_requests,
                )?;
            }
        }
        if was_empty {
            next.first_issue_timestamp_sum = now;
        }
        next.last_issue_timestamp_sum = now;
        let span = now.saturating_sub(next.first_issue_timestamp_sum);
        next.total_issue_span_cycles = span;
        next.max_issue_span_cycles = span;
        *self = next;
        Ok(())
    }

    fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        let mut next = *self;
        macro_rules! add {
            ($field:ident) => {
                next.$field = checked_readiness_add(
                    concat!("readiness.global_memory_issue.", stringify!($field)),
                    next.$field,
                    addend.$field,
                )?;
            };
        }
        add!(load_instructions);
        add!(store_instructions);
        add!(atomic_instructions);
        add!(reduction_instructions);
        add!(load_operand_read_barrier_instructions);
        add!(store_operand_read_barrier_instructions);
        add!(atomic_operand_read_barrier_instructions);
        add!(reduction_operand_read_barrier_instructions);
        add!(load_sectors);
        add!(store_sectors);
        add!(atomic_sectors);
        add!(reduction_sectors);
        add!(load_unique_lines);
        add!(store_unique_lines);
        add!(atomic_unique_lines);
        add!(reduction_unique_lines);
        add!(atomic_l2_requests);
        add!(reduction_l2_requests);
        add!(load_ideal_wavefronts);
        add!(store_ideal_wavefronts);
        add!(atomic_ideal_wavefronts);
        add!(reduction_ideal_wavefronts);
        add!(first_issue_timestamp_sum);
        add!(last_issue_timestamp_sum);
        add!(total_issue_span_cycles);
        next.max_issue_span_cycles = next.max_issue_span_cycles.max(addend.max_issue_span_cycles);
        *self = next;
        Ok(())
    }
}

pub const MAX_MEMORY_SERVICE_CANDIDATES: usize = 16;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryServiceCandidateScope {
    #[default]
    Sm,
    Smsp,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryServiceCandidateOperation {
    #[default]
    Load,
    Store,
    Atomic,
    Reduction,
}

/// One opt-in streaming service hypothesis. A `(units, cycles)` pair of
/// `(0, 0)` disables that work dimension; otherwise both terms must be
/// positive. Service work is additive and rounded up per issued warp
/// instruction. `completion_latency_cycles` extends only the observed terminal
/// timestamp, not the server's next-accept horizon.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryServiceCandidate {
    pub candidate_id: u8,
    pub scope: MemoryServiceCandidateScope,
    pub operation: MemoryServiceCandidateOperation,
    pub ideal_wavefront_units: u32,
    pub ideal_wavefront_cycles: u32,
    pub unique_line_units: u32,
    pub unique_line_cycles: u32,
    pub completion_latency_cycles: u32,
}

impl MemoryServiceCandidate {
    pub fn validate(self) -> Result<(), MemoryServiceCandidateError> {
        let ideal_disabled = self.ideal_wavefront_units == 0 && self.ideal_wavefront_cycles == 0;
        let line_disabled = self.unique_line_units == 0 && self.unique_line_cycles == 0;
        if (self.ideal_wavefront_units == 0) != (self.ideal_wavefront_cycles == 0) {
            return Err(MemoryServiceCandidateError::InvalidRate {
                candidate_id: self.candidate_id,
                dimension: "ideal_wavefront",
                units: self.ideal_wavefront_units,
                cycles: self.ideal_wavefront_cycles,
            });
        }
        if (self.unique_line_units == 0) != (self.unique_line_cycles == 0) {
            return Err(MemoryServiceCandidateError::InvalidRate {
                candidate_id: self.candidate_id,
                dimension: "unique_line",
                units: self.unique_line_units,
                cycles: self.unique_line_cycles,
            });
        }
        if ideal_disabled && line_disabled {
            return Err(MemoryServiceCandidateError::NoServiceWork {
                candidate_id: self.candidate_id,
            });
        }
        Ok(())
    }

    const fn matches(self, class: GlobalMemoryIssueClass) -> bool {
        matches!(
            (self.operation, class),
            (
                MemoryServiceCandidateOperation::Load,
                GlobalMemoryIssueClass::Load
            ) | (
                MemoryServiceCandidateOperation::Store,
                GlobalMemoryIssueClass::Store
            ) | (
                MemoryServiceCandidateOperation::Atomic,
                GlobalMemoryIssueClass::Atomic
            ) | (
                MemoryServiceCandidateOperation::Reduction,
                GlobalMemoryIssueClass::Reduction
            )
        )
    }
}

#[derive(Debug, thiserror::Error, Clone, Copy, PartialEq, Eq)]
pub enum MemoryServiceCandidateError {
    #[error("candidate set has {count} entries; maximum is {maximum}")]
    TooMany { count: usize, maximum: usize },
    #[error("candidate id {candidate_id} is duplicated")]
    DuplicateId { candidate_id: u8 },
    #[error(
        "candidate {candidate_id} {dimension} rate must have both terms zero or both positive (units={units}, cycles={cycles})"
    )]
    InvalidRate {
        candidate_id: u8,
        dimension: &'static str,
        units: u32,
        cycles: u32,
    },
    #[error("candidate {candidate_id} enables no service-work dimension")]
    NoServiceWork { candidate_id: u8 },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct MemoryServiceCandidateResult {
    pub candidate: MemoryServiceCandidate,
    pub instructions_observed: u64,
    pub ideal_wavefronts_observed: u64,
    pub unique_lines_observed: u64,
    pub busy_cycles: u64,
    pub queue_delay_cycles: u64,
    pub terminal_completion_time: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MemoryServiceCandidateDiagnostics {
    count: u8,
    entries: Option<Box<[MemoryServiceCandidateResult; MAX_MEMORY_SERVICE_CANDIDATES]>>,
}

impl Serialize for MemoryServiceCandidateDiagnostics {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.entries()[..].serialize(serializer)
    }
}

impl MemoryServiceCandidateDiagnostics {
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn entries(&self) -> &[MemoryServiceCandidateResult] {
        self.entries
            .as_deref()
            .map_or(&[], |entries| &entries[..usize::from(self.count)])
    }

    fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        if self.is_empty() {
            *self = addend.clone();
            return Ok(());
        }
        if addend.is_empty() {
            return Ok(());
        }
        if self.count != addend.count {
            return Err(ReadinessCounterError::MemoryServiceCandidateMismatch { index: 0 });
        }
        let mut next = self.clone();
        for index in 0..usize::from(self.count) {
            let current = self.entries()[index];
            let other = addend.entries()[index];
            if current.candidate != other.candidate {
                return Err(ReadinessCounterError::MemoryServiceCandidateMismatch { index });
            }
            let result = &mut next
                .entries
                .as_deref_mut()
                .expect("nonempty candidate diagnostics own entries")[index];
            macro_rules! add {
                ($field:ident) => {
                    result.$field = checked_readiness_add(
                        concat!("readiness.memory_service_candidates.", stringify!($field)),
                        result.$field,
                        other.$field,
                    )?;
                };
            }
            add!(instructions_observed);
            add!(ideal_wavefronts_observed);
            add!(unique_lines_observed);
            add!(busy_cycles);
            add!(queue_delay_cycles);
            result.terminal_completion_time = result
                .terminal_completion_time
                .max(other.terminal_completion_time);
        }
        *self = next;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct MemoryServiceCandidateRuntime {
    result: MemoryServiceCandidateResult,
    sm_ready: u64,
    smsp_ready: [u64; SUB_CORES],
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct MemoryServiceCandidateState {
    count: u8,
    entries: Option<Box<[MemoryServiceCandidateRuntime; MAX_MEMORY_SERVICE_CANDIDATES]>>,
}

fn candidate_rate_cycles(work: u64, units: u32, cycles: u32) -> Result<u64, IssueError> {
    if units == 0 {
        return Ok(0);
    }
    work.checked_mul(u64::from(cycles))
        .ok_or(IssueError::TimeMultiplicationOverflow {
            operation: "memory-service candidate work",
            lhs: work,
            rhs: u64::from(cycles),
        })
        .map(|scaled| scaled.div_ceil(u64::from(units)))
}

/// Price a fluid rational rate, optionally in fixed-width service quanta.
/// Width zero preserves the schema-7/9/10 legacy formula. A positive width
/// charges a complete service quantum for a partially occupied final batch,
/// making service width independently observable from asymptotic rate.
fn candidate_rate_cycles_with_width(
    work: u64,
    units: u32,
    cycles: u32,
    width: u32,
) -> Result<u64, IssueError> {
    if width == 0 {
        return candidate_rate_cycles(work, units, cycles);
    }
    let width = u64::from(width);
    let rounds = work.div_ceil(width);
    let round_cycles = candidate_rate_cycles(width, units, cycles)?;
    rounds
        .checked_mul(round_cycles)
        .ok_or(IssueError::TimeMultiplicationOverflow {
            operation: "memory-service candidate fixed-width rounds",
            lhs: rounds,
            rhs: round_cycles,
        })
}

impl MemoryServiceCandidateState {
    const fn is_empty(&self) -> bool {
        self.count == 0
    }

    fn checked(candidates: &[MemoryServiceCandidate]) -> Result<Self, MemoryServiceCandidateError> {
        if candidates.len() > MAX_MEMORY_SERVICE_CANDIDATES {
            return Err(MemoryServiceCandidateError::TooMany {
                count: candidates.len(),
                maximum: MAX_MEMORY_SERVICE_CANDIDATES,
            });
        }
        let mut state = Self::default();
        if !candidates.is_empty() {
            state.entries = Some(Box::new(
                [MemoryServiceCandidateRuntime::default(); MAX_MEMORY_SERVICE_CANDIDATES],
            ));
        }
        for (index, &candidate) in candidates.iter().enumerate() {
            candidate.validate()?;
            if candidates[..index]
                .iter()
                .any(|prior| prior.candidate_id == candidate.candidate_id)
            {
                return Err(MemoryServiceCandidateError::DuplicateId {
                    candidate_id: candidate.candidate_id,
                });
            }
            state
                .entries
                .as_deref_mut()
                .expect("nonempty candidate set owns runtime entries")[index]
                .result
                .candidate = candidate;
        }
        state.count = candidates.len() as u8;
        Ok(state)
    }

    fn diagnostics(&self) -> MemoryServiceCandidateDiagnostics {
        if self.is_empty() {
            return MemoryServiceCandidateDiagnostics::default();
        }
        let mut entries =
            Box::new([MemoryServiceCandidateResult::default(); MAX_MEMORY_SERVICE_CANDIDATES]);
        let runtime_entries = self
            .entries
            .as_deref()
            .expect("nonempty candidate state owns runtime entries");
        for index in 0..usize::from(self.count) {
            entries[index] = runtime_entries[index].result;
        }
        MemoryServiceCandidateDiagnostics {
            count: self.count,
            entries: Some(entries),
        }
    }

    fn observe(
        &mut self,
        subcore: usize,
        class: GlobalMemoryIssueClass,
        ideal_wavefronts: u64,
        unique_lines: u64,
        now: u64,
    ) -> Result<(), IssueError> {
        let entries = self
            .entries
            .as_deref_mut()
            .expect("observe is called only for a nonempty candidate state");
        for index in 0..usize::from(self.count) {
            let runtime = &mut entries[index];
            let candidate = runtime.result.candidate;
            if !candidate.matches(class) {
                continue;
            }
            let ideal_cycles = candidate_rate_cycles(
                ideal_wavefronts,
                candidate.ideal_wavefront_units,
                candidate.ideal_wavefront_cycles,
            )?;
            let line_cycles = candidate_rate_cycles(
                unique_lines,
                candidate.unique_line_units,
                candidate.unique_line_cycles,
            )?;
            let busy = checked_cycle_add(
                "memory-service candidate additive work",
                ideal_cycles,
                line_cycles,
            )?;
            let ready = match candidate.scope {
                MemoryServiceCandidateScope::Sm => runtime.sm_ready,
                MemoryServiceCandidateScope::Smsp => runtime.smsp_ready[subcore],
            };
            let start = now.max(ready);
            let done = checked_cycle_add("memory-service candidate completion", start, busy)?;
            match candidate.scope {
                MemoryServiceCandidateScope::Sm => runtime.sm_ready = done,
                MemoryServiceCandidateScope::Smsp => runtime.smsp_ready[subcore] = done,
            }
            let completion = checked_cycle_add(
                "memory-service candidate latency",
                done,
                u64::from(candidate.completion_latency_cycles),
            )?;
            let result = &mut runtime.result;
            result.instructions_observed = checked_u64_counter_add(
                "memory-service candidate instructions",
                result.instructions_observed,
                1,
            )?;
            result.ideal_wavefronts_observed = checked_u64_counter_add(
                "memory-service candidate ideal wavefronts",
                result.ideal_wavefronts_observed,
                ideal_wavefronts,
            )?;
            result.unique_lines_observed = checked_u64_counter_add(
                "memory-service candidate unique lines",
                result.unique_lines_observed,
                unique_lines,
            )?;
            result.busy_cycles = checked_u64_counter_add(
                "memory-service candidate busy cycles",
                result.busy_cycles,
                busy,
            )?;
            result.queue_delay_cycles = checked_u64_counter_add(
                "memory-service candidate queue delay",
                result.queue_delay_cycles,
                start - now,
            )?;
            result.terminal_completion_time = result.terminal_completion_time.max(completion);
        }
        Ok(())
    }
}

pub(crate) fn validate_memory_service_candidates(
    candidates: &[MemoryServiceCandidate],
) -> Result<(), MemoryServiceCandidateError> {
    MemoryServiceCandidateState::checked(candidates).map(|_| ())
}

pub const MAX_MEMORY_ADMISSION_CANDIDATES: usize = 16;
pub const MAX_MEMORY_ADMISSION_QUEUE_CAPACITY: u16 = 256;
pub const MAX_MEMORY_ADMISSION_SERVICE_WIDTH_SECTORS: u32 = 256;
pub const MAX_WRITE_COMPLETION_PARALLEL_LANES: u16 = 32;

/// One observational finite input-queue hypothesis. Capacity counts waiting
/// warp instructions; the active instruction resides in the separate shared
/// downstream server. A queue slot is released at service start, and a same-
/// cycle dequeue may admit a new instruction. Service follows observed per-SM
/// issue order. Miss work may optionally use fixed-width service quanta; zero
/// width retains the legacy fluid rational rate. The observational evaluator
/// never changes scheduler or cache ordering.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryAdmissionCandidate {
    pub candidate_id: u8,
    pub scope: MemoryServiceCandidateScope,
    pub operation: MemoryServiceCandidateOperation,
    pub queue_capacity_instructions: u16,
    pub admission_latency_cycles: u32,
    pub l2_hit_sector_units: u32,
    pub l2_hit_sector_cycles: u32,
    pub l2_miss_sector_units: u32,
    pub l2_miss_sector_cycles: u32,
    /// Zero retains fluid rational service. A positive value batches L2-miss
    /// sectors into fixed-width quanta and charges a full final quantum.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub l2_miss_service_width_sectors: u32,
    /// Opt-in completion lane: the issuing warp's `MEMBAR.SC.GPU` waits for
    /// its own latest matching write-service completion. Queue admission still
    /// releases at service start.
    #[serde(default, skip_serializing_if = "is_false")]
    pub membar_waits_for_service_completion: bool,
    /// Non-occupying visibility/RMW tail after service done. It changes only
    /// the per-warp MEMBAR completion frontier, never admission service.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub write_completion_latency_cycles: u32,
    /// Independent post-service RED/STORE completion work. Both dimensions are
    /// additive; zero disables a dimension.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub write_completion_request_cycles: u32,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub write_completion_region_cycles: u32,
    #[serde(default, skip_serializing_if = "is_zero_u16")]
    pub write_completion_parallel_lanes: u16,
}

impl MemoryAdmissionCandidate {
    pub fn validate(self) -> Result<(), MemoryAdmissionCandidateError> {
        if !matches!(
            self.operation,
            MemoryServiceCandidateOperation::Store | MemoryServiceCandidateOperation::Reduction
        ) {
            return Err(MemoryAdmissionCandidateError::UnsupportedOperation {
                candidate_id: self.candidate_id,
                operation: self.operation,
            });
        }
        if self.queue_capacity_instructions == 0
            || self.queue_capacity_instructions > MAX_MEMORY_ADMISSION_QUEUE_CAPACITY
        {
            return Err(MemoryAdmissionCandidateError::CapacityOutOfRange {
                candidate_id: self.candidate_id,
                capacity: self.queue_capacity_instructions,
                maximum: MAX_MEMORY_ADMISSION_QUEUE_CAPACITY,
            });
        }
        if self.l2_miss_service_width_sectors > MAX_MEMORY_ADMISSION_SERVICE_WIDTH_SECTORS {
            return Err(MemoryAdmissionCandidateError::ServiceWidthOutOfRange {
                candidate_id: self.candidate_id,
                width: self.l2_miss_service_width_sectors,
                maximum: MAX_MEMORY_ADMISSION_SERVICE_WIDTH_SECTORS,
            });
        }
        if self.write_completion_latency_cycles != 0 && !self.membar_waits_for_service_completion {
            return Err(
                MemoryAdmissionCandidateError::CompletionLatencyWithoutMembar {
                    candidate_id: self.candidate_id,
                    latency: self.write_completion_latency_cycles,
                },
            );
        }
        for (dimension, units, cycles) in [
            (
                "l2_hit_sector",
                self.l2_hit_sector_units,
                self.l2_hit_sector_cycles,
            ),
            (
                "l2_miss_sector",
                self.l2_miss_sector_units,
                self.l2_miss_sector_cycles,
            ),
        ] {
            if units == 0 || cycles == 0 {
                return Err(MemoryAdmissionCandidateError::InvalidRate {
                    candidate_id: self.candidate_id,
                    dimension,
                    units,
                    cycles,
                });
            }
        }
        let miss_scaled = u64::from(self.l2_miss_sector_cycles)
            .checked_mul(u64::from(self.l2_hit_sector_units))
            .expect("u32 cross product fits u64");
        let hit_scaled = u64::from(self.l2_hit_sector_cycles)
            .checked_mul(u64::from(self.l2_miss_sector_units))
            .expect("u32 cross product fits u64");
        if miss_scaled < hit_scaled {
            return Err(MemoryAdmissionCandidateError::MissRateFasterThanHit {
                candidate_id: self.candidate_id,
            });
        }
        let completion_work =
            self.write_completion_request_cycles != 0 || self.write_completion_region_cycles != 0;
        if completion_work && !self.membar_waits_for_service_completion {
            return Err(MemoryAdmissionCandidateError::CompletionWorkWithoutMembar {
                candidate_id: self.candidate_id,
            });
        }
        if completion_work
            && (self.write_completion_parallel_lanes == 0
                || self.write_completion_parallel_lanes > MAX_WRITE_COMPLETION_PARALLEL_LANES)
        {
            return Err(MemoryAdmissionCandidateError::CompletionLanesOutOfRange {
                candidate_id: self.candidate_id,
                lanes: self.write_completion_parallel_lanes,
                maximum: MAX_WRITE_COMPLETION_PARALLEL_LANES,
            });
        }
        if !completion_work && self.write_completion_parallel_lanes != 0 {
            return Err(MemoryAdmissionCandidateError::CompletionLanesWithoutWork {
                candidate_id: self.candidate_id,
                lanes: self.write_completion_parallel_lanes,
            });
        }
        Ok(())
    }

    pub(crate) fn validate_feedback(self, issue_model_cycles: u64) -> Result<(), IssueError> {
        self.validate()?;
        if u64::from(self.admission_latency_cycles) != issue_model_cycles {
            return Err(IssueError::MemoryAdmissionFeedbackLatencyMismatch {
                candidate_id: self.candidate_id,
                configured_cycles: self.admission_latency_cycles,
                issue_model_cycles,
            });
        }
        Ok(())
    }

    const fn matches(self, operation: MemoryOperation) -> bool {
        matches!(
            (self.operation, operation),
            (
                MemoryServiceCandidateOperation::Store,
                MemoryOperation::Store
            ) | (
                MemoryServiceCandidateOperation::Reduction,
                MemoryOperation::Reduction
            )
        )
    }
}

#[derive(Debug, thiserror::Error, Clone, Copy, PartialEq, Eq)]
pub enum MemoryAdmissionCandidateError {
    #[error("memory-admission candidate set has {count} entries; maximum is {maximum}")]
    TooMany { count: usize, maximum: usize },
    #[error("memory-admission candidate id {candidate_id} is duplicated")]
    DuplicateId { candidate_id: u8 },
    #[error(
        "memory-admission candidate {candidate_id} supports only store or reduction, got {operation:?}"
    )]
    UnsupportedOperation {
        candidate_id: u8,
        operation: MemoryServiceCandidateOperation,
    },
    #[error(
        "memory-admission candidate {candidate_id} queue capacity {capacity} is outside 1..={maximum}"
    )]
    CapacityOutOfRange {
        candidate_id: u8,
        capacity: u16,
        maximum: u16,
    },
    #[error(
        "memory-admission candidate {candidate_id} {dimension} rate must have positive terms (units={units}, cycles={cycles})"
    )]
    InvalidRate {
        candidate_id: u8,
        dimension: &'static str,
        units: u32,
        cycles: u32,
    },
    #[error("memory-admission candidate {candidate_id} prices an L2 miss faster than a hit")]
    MissRateFasterThanHit { candidate_id: u8 },
    #[error(
        "memory-admission candidate {candidate_id} L2-miss service width {width} exceeds maximum {maximum} sectors"
    )]
    ServiceWidthOutOfRange {
        candidate_id: u8,
        width: u32,
        maximum: u32,
    },
    #[error(
        "memory-admission candidate {candidate_id} declares write-completion latency {latency} without enabling MEMBAR completion feedback"
    )]
    CompletionLatencyWithoutMembar { candidate_id: u8, latency: u32 },
    #[error(
        "memory-admission candidate {candidate_id} declares write-completion work without enabling MEMBAR completion feedback"
    )]
    CompletionWorkWithoutMembar { candidate_id: u8 },
    #[error(
        "memory-admission candidate {candidate_id} write-completion lanes {lanes} are outside 1..={maximum}"
    )]
    CompletionLanesOutOfRange {
        candidate_id: u8,
        lanes: u16,
        maximum: u16,
    },
    #[error(
        "memory-admission candidate {candidate_id} declares {lanes} write-completion lanes without completion work"
    )]
    CompletionLanesWithoutWork { candidate_id: u8, lanes: u16 },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct MemoryAdmissionCandidateResult {
    pub candidate: MemoryAdmissionCandidate,
    pub instructions_observed: u64,
    pub l2_hit_sectors_observed: u64,
    pub l2_miss_sectors_observed: u64,
    pub service_busy_cycles: u64,
    pub downstream_queue_delay_cycles: u64,
    pub queue_full_instructions: u64,
    pub total_admission_delay_cycles: u64,
    pub maximum_admission_delay_cycles: u64,
    pub maximum_warp_admission_delay_cycles: u64,
    pub terminal_operand_read_clear_time: u64,
    pub terminal_service_completion_time: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MemoryAdmissionCandidateDiagnostics {
    count: u8,
    entries: Option<Box<[MemoryAdmissionCandidateResult; MAX_MEMORY_ADMISSION_CANDIDATES]>>,
}

impl Serialize for MemoryAdmissionCandidateDiagnostics {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.entries()[..].serialize(serializer)
    }
}

impl MemoryAdmissionCandidateDiagnostics {
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn entries(&self) -> &[MemoryAdmissionCandidateResult] {
        self.entries
            .as_deref()
            .map_or(&[], |entries| &entries[..usize::from(self.count)])
    }

    fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        if self.is_empty() {
            *self = addend.clone();
            return Ok(());
        }
        if addend.is_empty() {
            return Ok(());
        }
        if self.count != addend.count {
            return Err(ReadinessCounterError::MemoryAdmissionCandidateMismatch { index: 0 });
        }
        let mut next = self.clone();
        for index in 0..usize::from(self.count) {
            let current = self.entries()[index];
            let other = addend.entries()[index];
            if current.candidate != other.candidate {
                return Err(ReadinessCounterError::MemoryAdmissionCandidateMismatch { index });
            }
            let result = &mut next
                .entries
                .as_deref_mut()
                .expect("nonempty admission diagnostics own entries")[index];
            macro_rules! add {
                ($field:ident) => {
                    result.$field = checked_readiness_add(
                        concat!("readiness.memory_admission_candidates.", stringify!($field)),
                        result.$field,
                        other.$field,
                    )?;
                };
            }
            add!(instructions_observed);
            add!(l2_hit_sectors_observed);
            add!(l2_miss_sectors_observed);
            add!(service_busy_cycles);
            add!(downstream_queue_delay_cycles);
            add!(queue_full_instructions);
            add!(total_admission_delay_cycles);
            result.maximum_admission_delay_cycles = result
                .maximum_admission_delay_cycles
                .max(other.maximum_admission_delay_cycles);
            result.maximum_warp_admission_delay_cycles = result
                .maximum_warp_admission_delay_cycles
                .max(other.maximum_warp_admission_delay_cycles);
            result.terminal_operand_read_clear_time = result
                .terminal_operand_read_clear_time
                .max(other.terminal_operand_read_clear_time);
            result.terminal_service_completion_time = result
                .terminal_service_completion_time
                .max(other.terminal_service_completion_time);
        }
        *self = next;
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct MemoryAdmissionCandidateRuntime {
    result: MemoryAdmissionCandidateResult,
    server_ready: u64,
    queue_service_starts: [VecDeque<u64>; SUB_CORES],
    warp_admission_delay: Vec<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct MemoryAdmissionCandidateState {
    count: u8,
    entries: Option<Box<[MemoryAdmissionCandidateRuntime; MAX_MEMORY_ADMISSION_CANDIDATES]>>,
}

impl MemoryAdmissionCandidateState {
    pub(crate) fn checked(
        candidates: &[MemoryAdmissionCandidate],
    ) -> Result<Self, MemoryAdmissionCandidateError> {
        if candidates.len() > MAX_MEMORY_ADMISSION_CANDIDATES {
            return Err(MemoryAdmissionCandidateError::TooMany {
                count: candidates.len(),
                maximum: MAX_MEMORY_ADMISSION_CANDIDATES,
            });
        }
        let mut state = Self::default();
        if !candidates.is_empty() {
            state.entries = Some(Box::new(std::array::from_fn(|_| {
                MemoryAdmissionCandidateRuntime::default()
            })));
        }
        for (index, &candidate) in candidates.iter().enumerate() {
            candidate.validate()?;
            if candidates[..index]
                .iter()
                .any(|prior| prior.candidate_id == candidate.candidate_id)
            {
                return Err(MemoryAdmissionCandidateError::DuplicateId {
                    candidate_id: candidate.candidate_id,
                });
            }
            state
                .entries
                .as_deref_mut()
                .expect("nonempty admission candidate set owns runtime entries")[index]
                .result
                .candidate = candidate;
        }
        state.count = candidates.len() as u8;
        Ok(state)
    }

    pub(crate) fn diagnostics(&self) -> MemoryAdmissionCandidateDiagnostics {
        if self.count == 0 {
            return MemoryAdmissionCandidateDiagnostics::default();
        }
        let mut entries =
            Box::new([MemoryAdmissionCandidateResult::default(); MAX_MEMORY_ADMISSION_CANDIDATES]);
        let runtime = self
            .entries
            .as_deref()
            .expect("nonempty admission state owns runtime entries");
        for index in 0..usize::from(self.count) {
            entries[index] = runtime[index].result;
        }
        MemoryAdmissionCandidateDiagnostics {
            count: self.count,
            entries: Some(entries),
        }
    }

    pub(crate) fn observe(
        &mut self,
        subcore: usize,
        warp: usize,
        operation: MemoryOperation,
        l2_hit_sectors: u64,
        l2_miss_sectors: u64,
        now: u64,
    ) -> Result<(), IssueError> {
        let entries = self
            .entries
            .as_deref_mut()
            .expect("observe is called only for nonempty admission state");
        for runtime in &mut entries[..usize::from(self.count)] {
            let candidate = runtime.result.candidate;
            if !candidate.matches(operation) {
                continue;
            }
            let hit_cycles = candidate_rate_cycles(
                l2_hit_sectors,
                candidate.l2_hit_sector_units,
                candidate.l2_hit_sector_cycles,
            )?;
            let miss_cycles = candidate_rate_cycles_with_width(
                l2_miss_sectors,
                candidate.l2_miss_sector_units,
                candidate.l2_miss_sector_cycles,
                candidate.l2_miss_service_width_sectors,
            )?;
            let busy = checked_cycle_add(
                "memory-admission candidate service work",
                hit_cycles,
                miss_cycles,
            )?;
            let queue_index = match candidate.scope {
                MemoryServiceCandidateScope::Sm => 0,
                MemoryServiceCandidateScope::Smsp => subcore,
            };
            let queue = &mut runtime.queue_service_starts[queue_index];
            let capacity = usize::from(candidate.queue_capacity_instructions);
            let released_at = if queue.len() == capacity {
                queue
                    .pop_front()
                    .expect("capacity-sized admission queue has an oldest service start")
            } else {
                0
            };
            let admitted_at = now.max(released_at);
            let service_start = admitted_at.max(runtime.server_ready);
            let service_done = checked_cycle_add(
                "memory-admission candidate service completion",
                service_start,
                busy,
            )?;
            let operand_clear = checked_cycle_add(
                "memory-admission candidate operand-read visibility",
                admitted_at,
                u64::from(candidate.admission_latency_cycles),
            )?;
            queue.push_back(service_start);
            debug_assert!(queue.len() <= capacity);
            runtime.server_ready = service_done;

            if runtime.warp_admission_delay.len() <= warp {
                runtime.warp_admission_delay.resize(warp + 1, 0);
            }
            let admission_delay = admitted_at - now;
            let warp_delay = checked_u64_counter_add(
                "memory-admission candidate per-warp delay",
                runtime.warp_admission_delay[warp],
                admission_delay,
            )?;
            runtime.warp_admission_delay[warp] = warp_delay;

            let result = &mut runtime.result;
            result.instructions_observed = checked_u64_counter_add(
                "memory-admission candidate instructions",
                result.instructions_observed,
                1,
            )?;
            result.l2_hit_sectors_observed = checked_u64_counter_add(
                "memory-admission candidate L2 hit sectors",
                result.l2_hit_sectors_observed,
                l2_hit_sectors,
            )?;
            result.l2_miss_sectors_observed = checked_u64_counter_add(
                "memory-admission candidate L2 miss sectors",
                result.l2_miss_sectors_observed,
                l2_miss_sectors,
            )?;
            result.service_busy_cycles = checked_u64_counter_add(
                "memory-admission candidate service busy cycles",
                result.service_busy_cycles,
                busy,
            )?;
            result.downstream_queue_delay_cycles = checked_u64_counter_add(
                "memory-admission candidate downstream queue delay",
                result.downstream_queue_delay_cycles,
                service_start - admitted_at,
            )?;
            result.queue_full_instructions = checked_u64_counter_add(
                "memory-admission candidate queue-full instructions",
                result.queue_full_instructions,
                u64::from(admission_delay != 0),
            )?;
            result.total_admission_delay_cycles = checked_u64_counter_add(
                "memory-admission candidate total admission delay",
                result.total_admission_delay_cycles,
                admission_delay,
            )?;
            result.maximum_admission_delay_cycles =
                result.maximum_admission_delay_cycles.max(admission_delay);
            result.maximum_warp_admission_delay_cycles =
                result.maximum_warp_admission_delay_cycles.max(warp_delay);
            result.terminal_operand_read_clear_time =
                result.terminal_operand_read_clear_time.max(operand_clear);
            result.terminal_service_completion_time =
                result.terminal_service_completion_time.max(service_done);
        }
        Ok(())
    }
}

pub(crate) fn validate_memory_admission_candidates(
    candidates: &[MemoryAdmissionCandidate],
) -> Result<(), MemoryAdmissionCandidateError> {
    MemoryAdmissionCandidateState::checked(candidates).map(|_| ())
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct MemoryAdmissionFeedbackDiagnostics {
    pub candidate: MemoryAdmissionCandidate,
    pub instructions_admitted: u64,
    pub l2_hit_sectors_observed: u64,
    pub l2_miss_sectors_observed: u64,
    pub service_busy_cycles: u64,
    pub maximum_waiting_instructions: u64,
    pub terminal_service_completion_time: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub terminal_write_completion_time: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub write_completion_requests_observed: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub write_completion_regions_observed: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub write_completion_busy_cycles: u64,
}

impl MemoryAdmissionFeedbackDiagnostics {
    pub const fn is_empty(&self) -> bool {
        self.candidate.queue_capacity_instructions == 0
    }

    fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        if self.is_empty() {
            *self = *addend;
            return Ok(());
        }
        if addend.is_empty() {
            return Ok(());
        }
        if self.candidate != addend.candidate {
            return Err(ReadinessCounterError::MemoryAdmissionCandidateMismatch { index: 0 });
        }
        let next = Self {
            candidate: self.candidate,
            instructions_admitted: checked_readiness_add(
                "readiness.memory_admission_feedback.instructions_admitted",
                self.instructions_admitted,
                addend.instructions_admitted,
            )?,
            l2_hit_sectors_observed: checked_readiness_add(
                "readiness.memory_admission_feedback.l2_hit_sectors_observed",
                self.l2_hit_sectors_observed,
                addend.l2_hit_sectors_observed,
            )?,
            l2_miss_sectors_observed: checked_readiness_add(
                "readiness.memory_admission_feedback.l2_miss_sectors_observed",
                self.l2_miss_sectors_observed,
                addend.l2_miss_sectors_observed,
            )?,
            service_busy_cycles: checked_readiness_add(
                "readiness.memory_admission_feedback.service_busy_cycles",
                self.service_busy_cycles,
                addend.service_busy_cycles,
            )?,
            maximum_waiting_instructions: self
                .maximum_waiting_instructions
                .max(addend.maximum_waiting_instructions),
            terminal_service_completion_time: self
                .terminal_service_completion_time
                .max(addend.terminal_service_completion_time),
            terminal_write_completion_time: self
                .terminal_write_completion_time
                .max(addend.terminal_write_completion_time),
            write_completion_requests_observed: checked_readiness_add(
                "readiness.memory_admission_feedback.write_completion_requests_observed",
                self.write_completion_requests_observed,
                addend.write_completion_requests_observed,
            )?,
            write_completion_regions_observed: checked_readiness_add(
                "readiness.memory_admission_feedback.write_completion_regions_observed",
                self.write_completion_regions_observed,
                addend.write_completion_regions_observed,
            )?,
            write_completion_busy_cycles: checked_readiness_add(
                "readiness.memory_admission_feedback.write_completion_busy_cycles",
                self.write_completion_busy_cycles,
                addend.write_completion_busy_cycles,
            )?,
        };
        *self = next;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MemoryAdmissionFeedbackEntry {
    warp: u32,
    instruction: u32,
    release_at: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
struct MemoryAdmissionFeedbackIssue {
    subcore: usize,
    warp: usize,
    instruction: usize,
    kind: AccessKind,
    prior_clock_reads: u64,
    clock_read: bool,
}

#[derive(Debug, Clone, Copy)]
struct MemoryAdmissionFeedbackIssuePlan {
    reservations: [Option<(usize, MemoryAdmissionFeedbackEntry)>; SUB_CORES],
    second_clock_warps: [Option<usize>; SUB_CORES],
    instructions_admitted: u64,
    maximum_waiting_instructions: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MemoryAdmissionFeedbackState {
    candidate: MemoryAdmissionCandidate,
    queues: [VecDeque<MemoryAdmissionFeedbackEntry>; SUB_CORES],
    server_ready: u64,
    warp_completion_ready: Vec<u64>,
    completion_lane_ready: Vec<u64>,
    clock_second_issue: Vec<Option<u64>>,
    diagnostics: MemoryAdmissionFeedbackDiagnostics,
    yield_requested: bool,
}

impl MemoryAdmissionFeedbackState {
    pub(crate) fn checked(
        candidate: MemoryAdmissionCandidate,
    ) -> Result<Self, MemoryAdmissionCandidateError> {
        candidate.validate()?;
        Ok(Self {
            candidate,
            queues: std::array::from_fn(|_| VecDeque::new()),
            server_ready: 0,
            warp_completion_ready: Vec::new(),
            completion_lane_ready: vec![0; usize::from(candidate.write_completion_parallel_lanes)],
            clock_second_issue: Vec::new(),
            diagnostics: MemoryAdmissionFeedbackDiagnostics {
                candidate,
                ..MemoryAdmissionFeedbackDiagnostics::default()
            },
            yield_requested: false,
        })
    }

    const fn queue_index(&self, subcore: usize) -> usize {
        match self.candidate.scope {
            MemoryServiceCandidateScope::Sm => 0,
            MemoryServiceCandidateScope::Smsp => subcore,
        }
    }

    fn matches_kind(&self, kind: AccessKind) -> bool {
        matches!(
            (self.candidate.operation, kind),
            (
                MemoryServiceCandidateOperation::Store,
                AccessKind::GlobalStore { len: 1.., .. }
            ) | (
                MemoryServiceCandidateOperation::Reduction,
                AccessKind::GlobalReduction { len: 1.., .. }
            )
        )
    }

    fn reap(&mut self, now: u64) {
        for queue in &mut self.queues {
            while queue
                .front()
                .and_then(|entry| entry.release_at)
                .is_some_and(|release| release <= now)
            {
                queue.pop_front();
            }
        }
    }

    fn admission_ready_at(&self, subcore: usize, kind: AccessKind) -> Option<u64> {
        if !self.matches_kind(kind) {
            return Some(0);
        }
        let queue = &self.queues[self.queue_index(subcore)];
        if queue.len() < usize::from(self.candidate.queue_capacity_instructions) {
            return Some(0);
        }
        queue.front().and_then(|entry| entry.release_at)
    }

    fn completion_ready_at(&self, warp: usize, slot: &IssueSlot) -> Option<u64> {
        if !self.candidate.membar_waits_for_service_completion || !is_gpu_membar_issue(slot) {
            return Some(0);
        }
        if self.has_unpriced() {
            return None;
        }
        Some(self.warp_completion_ready.get(warp).copied().unwrap_or(0))
    }

    fn second_clock_issue(&self, warp: usize) -> Option<u64> {
        self.clock_second_issue.get(warp).copied().flatten()
    }

    fn plan_issue(
        &self,
        issues: &[Option<MemoryAdmissionFeedbackIssue>; SUB_CORES],
        now: u64,
    ) -> Result<MemoryAdmissionFeedbackIssuePlan, IssueError> {
        let active_lengths: [usize; SUB_CORES] = std::array::from_fn(|queue_index| {
            self.queues[queue_index]
                .iter()
                .filter(|entry| entry.release_at.is_none_or(|release| release > now))
                .count()
        });
        let mut additions = [0usize; SUB_CORES];
        let mut reservations = [None; SUB_CORES];
        let mut second_clock_warps = [None; SUB_CORES];
        let mut instructions_admitted = self.diagnostics.instructions_admitted;
        let mut maximum_waiting_instructions = self.diagnostics.maximum_waiting_instructions;
        for (position, issue) in issues.iter().copied().enumerate() {
            let Some(issue) = issue else { continue };
            if self.candidate.membar_waits_for_service_completion
                && issue.prior_clock_reads == 1
                && issue.clock_read
            {
                second_clock_warps[position] = Some(issue.warp);
            }
            if !self.matches_kind(issue.kind) {
                continue;
            }
            let queue_index = self.queue_index(issue.subcore);
            let next_length = active_lengths[queue_index]
                .checked_add(additions[queue_index])
                .and_then(|length| length.checked_add(1))
                .ok_or(IssueError::CounterOverflow {
                    counter: "feedback admission queue length",
                })?;
            if next_length > usize::from(self.candidate.queue_capacity_instructions) {
                return Err(IssueError::MemoryAdmissionFeedbackQueueFull {
                    subcore: issue.subcore,
                    time: now,
                });
            }
            let entry = MemoryAdmissionFeedbackEntry {
                warp: u32::try_from(issue.warp).map_err(|_| IssueError::CounterOverflow {
                    counter: "feedback admission warp",
                })?,
                instruction: u32::try_from(issue.instruction).map_err(|_| {
                    IssueError::CounterOverflow {
                        counter: "feedback admission instruction",
                    }
                })?,
                release_at: None,
            };
            instructions_admitted = checked_u64_counter_add(
                "feedback admission instructions",
                instructions_admitted,
                1,
            )?;
            additions[queue_index] += 1;
            maximum_waiting_instructions = maximum_waiting_instructions.max(next_length as u64);
            reservations[position] = Some((queue_index, entry));
        }
        Ok(MemoryAdmissionFeedbackIssuePlan {
            reservations,
            second_clock_warps,
            instructions_admitted,
            maximum_waiting_instructions,
        })
    }

    fn apply_issue(&mut self, plan: MemoryAdmissionFeedbackIssuePlan, now: u64) {
        self.reap(now);
        for warp in plan.second_clock_warps.into_iter().flatten() {
            if self.clock_second_issue.len() <= warp {
                self.clock_second_issue.resize(warp + 1, None);
            }
            self.clock_second_issue[warp] = Some(now);
        }
        let mut reserved = false;
        for (queue_index, entry) in plan.reservations.into_iter().flatten() {
            debug_assert!(
                self.queues[queue_index].len()
                    < usize::from(self.candidate.queue_capacity_instructions)
            );
            self.queues[queue_index].push_back(entry);
            reserved = true;
        }
        self.diagnostics.instructions_admitted = plan.instructions_admitted;
        self.diagnostics.maximum_waiting_instructions = plan.maximum_waiting_instructions;
        self.yield_requested |= reserved;
    }

    fn permits_transaction_addition(
        &self,
        subcore: usize,
        kind: AccessKind,
        additions: &[usize; SUB_CORES],
        now: u64,
    ) -> bool {
        if !self.matches_kind(kind) {
            return true;
        }
        let capacity = usize::from(self.candidate.queue_capacity_instructions);
        let queue_index = self.queue_index(subcore);
        let occupied = self.queues[queue_index]
            .iter()
            .filter(|entry| entry.release_at.is_none_or(|release| release > now))
            .count();
        occupied + additions[queue_index] < capacity
    }

    fn record_transaction_addition(
        &self,
        subcore: usize,
        kind: AccessKind,
        additions: &mut [usize; SUB_CORES],
    ) {
        if self.matches_kind(kind) {
            additions[self.queue_index(subcore)] += 1;
        }
    }

    #[cfg(test)]
    fn reserve(
        &mut self,
        subcore: usize,
        warp: usize,
        instruction: usize,
        kind: AccessKind,
        now: u64,
    ) -> Result<(), IssueError> {
        if !self.matches_kind(kind) {
            return Ok(());
        }
        self.reap(now);
        let queue_index = self.queue_index(subcore);
        let queue = &mut self.queues[queue_index];
        if queue.len() >= usize::from(self.candidate.queue_capacity_instructions) {
            return Err(IssueError::MemoryAdmissionFeedbackQueueFull { subcore, time: now });
        }
        let entry = MemoryAdmissionFeedbackEntry {
            warp: u32::try_from(warp).map_err(|_| IssueError::CounterOverflow {
                counter: "feedback admission warp",
            })?,
            instruction: u32::try_from(instruction).map_err(|_| IssueError::CounterOverflow {
                counter: "feedback admission instruction",
            })?,
            release_at: None,
        };
        let instructions_admitted = checked_u64_counter_add(
            "feedback admission instructions",
            self.diagnostics.instructions_admitted,
            1,
        )?;
        let maximum_waiting_instructions = self
            .diagnostics
            .maximum_waiting_instructions
            .max((queue.len() + 1) as u64);

        queue.push_back(entry);
        self.diagnostics.instructions_admitted = instructions_admitted;
        self.diagnostics.maximum_waiting_instructions = maximum_waiting_instructions;
        self.yield_requested = true;
        Ok(())
    }

    pub(crate) fn complete(
        &mut self,
        witness: AdmissionRequestWitness,
        operation: MemoryOperation,
        hit_sectors: u64,
        miss_sectors: u64,
        l2_requests: u64,
        unique_regions: u64,
        issue_time: u64,
    ) -> Result<(), IssueError> {
        if !self.candidate.matches(operation) {
            return Ok(());
        }
        let queue_index = self.queue_index(usize::from(witness.subcore()));
        let entry_index = self.queues[queue_index]
            .iter()
            .position(|entry| {
                entry.warp == witness.warp() && entry.instruction == witness.instruction()
            })
            .ok_or(IssueError::MemoryAdmissionFeedbackReservationMissing {
                warp: witness.warp(),
                instruction: witness.instruction(),
            })?;
        if self.queues[queue_index][entry_index].release_at.is_some() {
            return Err(IssueError::MemoryAdmissionFeedbackAlreadyPriced {
                warp: witness.warp(),
                instruction: witness.instruction(),
            });
        }
        let hit_cycles = candidate_rate_cycles(
            hit_sectors,
            self.candidate.l2_hit_sector_units,
            self.candidate.l2_hit_sector_cycles,
        )?;
        let miss_cycles = candidate_rate_cycles_with_width(
            miss_sectors,
            self.candidate.l2_miss_sector_units,
            self.candidate.l2_miss_sector_cycles,
            self.candidate.l2_miss_service_width_sectors,
        )?;
        let busy = checked_cycle_add("feedback admission service work", hit_cycles, miss_cycles)?;
        let service_start = issue_time.max(self.server_ready);
        let service_done =
            checked_cycle_add("feedback admission service completion", service_start, busy)?;
        let l2_hit_sectors_observed = checked_u64_counter_add(
            "feedback admission L2 hit sectors",
            self.diagnostics.l2_hit_sectors_observed,
            hit_sectors,
        )?;
        let l2_miss_sectors_observed = checked_u64_counter_add(
            "feedback admission L2 miss sectors",
            self.diagnostics.l2_miss_sectors_observed,
            miss_sectors,
        )?;
        let service_busy_cycles = checked_u64_counter_add(
            "feedback admission service busy cycles",
            self.diagnostics.service_busy_cycles,
            busy,
        )?;
        let terminal_service_completion_time = self
            .diagnostics
            .terminal_service_completion_time
            .max(service_done);
        let request_completion_cycles = l2_requests
            .checked_mul(u64::from(self.candidate.write_completion_request_cycles))
            .ok_or(IssueError::TimeMultiplicationOverflow {
                operation: "write-completion request work",
                lhs: l2_requests,
                rhs: u64::from(self.candidate.write_completion_request_cycles),
            })?;
        let region_completion_cycles = unique_regions
            .checked_mul(u64::from(self.candidate.write_completion_region_cycles))
            .ok_or(IssueError::TimeMultiplicationOverflow {
                operation: "write-completion region work",
                lhs: unique_regions,
                rhs: u64::from(self.candidate.write_completion_region_cycles),
            })?;
        let write_completion_busy = checked_cycle_add(
            "write-completion additive work",
            request_completion_cycles,
            region_completion_cycles,
        )?;
        let completion_lane = if write_completion_busy == 0 {
            None
        } else {
            self.completion_lane_ready
                .iter()
                .copied()
                .enumerate()
                .min_by_key(|&(index, ready)| (ready, index))
        };
        let write_service_done = if let Some((_, lane_ready)) = completion_lane {
            checked_cycle_add(
                "write-completion lane service",
                service_done.max(lane_ready),
                write_completion_busy,
            )?
        } else {
            service_done
        };
        let write_completion_done = if self.candidate.membar_waits_for_service_completion {
            checked_cycle_add(
                "feedback write completion visibility",
                write_service_done,
                u64::from(self.candidate.write_completion_latency_cycles),
            )?
        } else {
            service_done
        };
        let terminal_write_completion_time = if self.candidate.membar_waits_for_service_completion {
            self.diagnostics
                .terminal_write_completion_time
                .max(write_completion_done)
        } else {
            self.diagnostics.terminal_write_completion_time
        };
        let completion_requests_observed = checked_u64_counter_add(
            "feedback write-completion requests",
            self.diagnostics.write_completion_requests_observed,
            if write_completion_busy == 0 {
                0
            } else {
                l2_requests
            },
        )?;
        let completion_regions_observed = checked_u64_counter_add(
            "feedback write-completion regions",
            self.diagnostics.write_completion_regions_observed,
            if write_completion_busy == 0 {
                0
            } else {
                unique_regions
            },
        )?;
        let completion_busy_cycles = checked_u64_counter_add(
            "feedback write-completion busy cycles",
            self.diagnostics.write_completion_busy_cycles,
            write_completion_busy,
        )?;
        let warp = usize::try_from(witness.warp()).map_err(|_| IssueError::CounterOverflow {
            counter: "feedback completion warp index",
        })?;
        let required_warp_slots =
            checked_usize_counter_add("feedback completion warp slots", warp, 1)?;
        let warp_completion_ready = self
            .warp_completion_ready
            .get(warp)
            .copied()
            .unwrap_or(0)
            .max(write_completion_done);

        self.queues[queue_index][entry_index].release_at = Some(service_start);
        self.server_ready = service_done;
        if let Some((lane, _)) = completion_lane {
            self.completion_lane_ready[lane] = write_service_done;
        }
        if self.warp_completion_ready.len() <= warp {
            self.warp_completion_ready.resize(required_warp_slots, 0);
        }
        self.warp_completion_ready[warp] = warp_completion_ready;
        self.diagnostics.l2_hit_sectors_observed = l2_hit_sectors_observed;
        self.diagnostics.l2_miss_sectors_observed = l2_miss_sectors_observed;
        self.diagnostics.service_busy_cycles = service_busy_cycles;
        self.diagnostics.terminal_service_completion_time = terminal_service_completion_time;
        self.diagnostics.terminal_write_completion_time = terminal_write_completion_time;
        self.diagnostics.write_completion_requests_observed = completion_requests_observed;
        self.diagnostics.write_completion_regions_observed = completion_regions_observed;
        self.diagnostics.write_completion_busy_cycles = completion_busy_cycles;
        Ok(())
    }

    fn has_unpriced(&self) -> bool {
        self.queues
            .iter()
            .flatten()
            .any(|entry| entry.release_at.is_none())
    }

    fn take_yield_requested(&mut self) -> bool {
        std::mem::take(&mut self.yield_requested)
    }

    pub(crate) const fn diagnostics(&self) -> MemoryAdmissionFeedbackDiagnostics {
        self.diagnostics
    }
}

impl ClockIssueDiagnostics {
    const fn is_empty(&self) -> bool {
        self.reads_issued == 0
    }

    fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        let next = Self {
            reads_issued: checked_readiness_add(
                "readiness.clock_issue.reads_issued",
                self.reads_issued,
                addend.reads_issued,
            )?,
            warps_with_interval: checked_readiness_add(
                "readiness.clock_issue.warps_with_interval",
                self.warps_with_interval,
                addend.warps_with_interval,
            )?,
            maximum_first_to_last_issue_cycles: self
                .maximum_first_to_last_issue_cycles
                .max(addend.maximum_first_to_last_issue_cycles),
            warps_with_three_reads: checked_readiness_add(
                "readiness.clock_issue.warps_with_three_reads",
                self.warps_with_three_reads,
                addend.warps_with_three_reads,
            )?,
            maximum_first_to_second_issue_cycles: self
                .maximum_first_to_second_issue_cycles
                .max(addend.maximum_first_to_second_issue_cycles),
            maximum_second_to_third_issue_cycles: self
                .maximum_second_to_third_issue_cycles
                .max(addend.maximum_second_to_third_issue_cycles),
        };
        *self = next;
        Ok(())
    }
}

impl ResultWritebackDiagnostics {
    pub const fn is_empty(&self) -> bool {
        self.fixed_over_l1tex_collisions == 0 && self.l1tex_deferral_cycles == 0
    }

    fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        let next = Self {
            fixed_over_l1tex_collisions: checked_readiness_add(
                "readiness.result_writeback.fixed_over_l1tex_collisions",
                self.fixed_over_l1tex_collisions,
                addend.fixed_over_l1tex_collisions,
            )?,
            l1tex_deferral_cycles: checked_readiness_add(
                "readiness.result_writeback.l1tex_deferral_cycles",
                self.l1tex_deferral_cycles,
                addend.l1tex_deferral_cycles,
            )?,
        };
        *self = next;
        Ok(())
    }
}

/// Deterministic causal readiness summary for one SM replay.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ReadinessDiagnostics {
    pub limiting_cycles: ReadinessCauseCycles,
    pub constraint_exposure_cycles: ReadinessCauseCycles,
    pub delayed_ready_issues: u64,
    pub tied_limiting_causes: u64,
    pub async_groups: AsyncReadinessDiagnostics,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bypass_async_copies: Option<Box<BypassAsyncCopyDiagnostics>>,
    pub cta_barriers: CtaBarrierDiagnostics,
    #[serde(skip_serializing_if = "BarrierPipelineDiagnostics::is_empty")]
    pub barrier_pipeline: BarrierPipelineDiagnostics,
    #[serde(skip_serializing_if = "ControlPipelineDiagnostics::is_empty")]
    pub control_pipeline: ControlPipelineDiagnostics,
    #[serde(skip_serializing_if = "ResultWritebackDiagnostics::is_empty")]
    pub result_writeback: ResultWritebackDiagnostics,
    #[serde(skip_serializing_if = "ClockIssueDiagnostics::is_empty")]
    pub clock_issue: ClockIssueDiagnostics,
    #[serde(skip_serializing_if = "GlobalMemoryIssueDiagnostics::is_empty")]
    pub global_memory_issue: GlobalMemoryIssueDiagnostics,
    #[serde(skip_serializing_if = "MemoryServiceCandidateDiagnostics::is_empty")]
    pub memory_service_candidates: MemoryServiceCandidateDiagnostics,
    #[serde(skip_serializing_if = "MemoryAdmissionCandidateDiagnostics::is_empty")]
    pub memory_admission_candidates: MemoryAdmissionCandidateDiagnostics,
    #[serde(skip_serializing_if = "MemoryAdmissionFeedbackDiagnostics::is_empty")]
    pub memory_admission_feedback: MemoryAdmissionFeedbackDiagnostics,
}

impl ReadinessDiagnostics {
    /// Checked, transactional aggregation across SMs.
    pub fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        let mut next = self.clone();
        next.limiting_cycles.add_checked(&addend.limiting_cycles)?;
        next.constraint_exposure_cycles
            .add_checked(&addend.constraint_exposure_cycles)?;
        next.delayed_ready_issues = checked_readiness_add(
            "readiness.delayed_ready_issues",
            next.delayed_ready_issues,
            addend.delayed_ready_issues,
        )?;
        next.tied_limiting_causes = checked_readiness_add(
            "readiness.tied_limiting_causes",
            next.tied_limiting_causes,
            addend.tied_limiting_causes,
        )?;
        next.async_groups.add_checked(&addend.async_groups)?;
        if let Some(addend_bypass) = addend.bypass_async_copies.as_deref() {
            if let Some(current) = next.bypass_async_copies.as_deref_mut() {
                current.add_checked(addend_bypass)?;
            } else {
                next.bypass_async_copies = Some(Box::new(*addend_bypass));
            }
        }
        next.cta_barriers.add_checked(&addend.cta_barriers)?;
        next.barrier_pipeline
            .add_checked(&addend.barrier_pipeline)?;
        next.control_pipeline
            .add_checked(&addend.control_pipeline)?;
        next.result_writeback
            .add_checked(&addend.result_writeback)?;
        next.clock_issue.add_checked(&addend.clock_issue)?;
        next.global_memory_issue
            .add_checked(&addend.global_memory_issue)?;
        next.memory_service_candidates
            .add_checked(&addend.memory_service_candidates)?;
        next.memory_admission_candidates
            .add_checked(&addend.memory_admission_candidates)?;
        next.memory_admission_feedback
            .add_checked(&addend.memory_admission_feedback)?;
        *self = next;
        Ok(())
    }

    /// Schema 1 remains byte-compatible without characterized resources;
    /// schema 2 adds the barrier service, schema 3 adds the exact branch
    /// resolution relation, and schema 4 adds an observed fixed/L1TEX
    /// writeback collision. Schema 5 adds an exact CLOCK-boundary issue
    /// diagnostic when the joined program contains measured CLOCKLO reads.
    /// Schema 6 adds scalar-global instruction/sector/ideal-wavefront,
    /// unique-line, exact-atomic-request, and issue-span diagnostics without
    /// activating a memory service. Schema 7 adds results from an explicitly
    /// configured, bounded streaming memory-service candidate set; candidates
    /// remain observational and cannot change issue scheduling. Schema 8
    /// separates scalar-global operand-consumption/read-barrier producers and
    /// their consumer delay from the generic fixed read-counter relation.
    /// Schema 9 adds bounded observational finite admission queues driven by
    /// exact per-instruction L2 hit/miss outcomes. Schema 10 activates one
    /// opt-in feedback-coupled queue that participates in warp readiness.
    /// Schema 11 additionally makes `MEMBAR.SC.GPU` wait on the issuing warp's
    /// matching write-service completion frontier. Schema 12 adds a
    /// non-occupying operation-specific visibility/RMW latency after service.
    /// Schema 13 adds independent request/region completion work over bounded
    /// parallel lanes. Schema 14 adds bypass-only per-copy source/delivery
    /// observations without changing scheduler timing. Schema 15 separates
    /// source-cohort destination-line fanout, bank multiplicity, and a
    /// timing-neutral per-copy topology candidate. Schema 16 adds
    /// timing-neutral pipelined destination candidates over stable per-copy
    /// payload-arrival order for acceptance widths 1, 2, 4, 8, and 16.
    pub fn schema_version(&self) -> u32 {
        if self
            .bypass_async_copies
            .as_deref()
            .is_some_and(|diagnostics| diagnostics.pipeline_groups_observed != 0)
        {
            16
        } else if self
            .bypass_async_copies
            .as_deref()
            .is_some_and(|diagnostics| diagnostics.source_cohorts_observed != 0)
        {
            15
        } else if self.bypass_async_copies.is_some() {
            14
        } else if !self.memory_admission_feedback.is_empty()
            && (self
                .memory_admission_feedback
                .candidate
                .write_completion_request_cycles
                != 0
                || self
                    .memory_admission_feedback
                    .candidate
                    .write_completion_region_cycles
                    != 0)
        {
            13
        } else if !self.memory_admission_feedback.is_empty()
            && self
                .memory_admission_feedback
                .candidate
                .write_completion_latency_cycles
                != 0
        {
            12
        } else if !self.memory_admission_feedback.is_empty()
            && self
                .memory_admission_feedback
                .candidate
                .membar_waits_for_service_completion
        {
            11
        } else if !self.memory_admission_feedback.is_empty() {
            10
        } else if !self.memory_admission_candidates.is_empty() {
            9
        } else if self.global_memory_issue.has_operand_read_barriers() {
            8
        } else if !self.memory_service_candidates.is_empty() {
            7
        } else if !self.global_memory_issue.is_empty() {
            6
        } else if !self.clock_issue.is_empty() {
            5
        } else if !self.result_writeback.is_empty() {
            4
        } else if !self.control_pipeline.is_empty() {
            3
        } else if !self.barrier_pipeline.is_empty() {
            2
        } else {
            1
        }
    }
}

/// Result of replaying one SM's blocks to completion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmSimResult {
    /// Exact: one per traced record fed in.
    pub instructions_issued: u64,
    /// Directional v0 cycle count (time of last issue + 1; the response
    /// tail of the final loads is not counted — write-back completion is
    /// not an issue event).
    pub cycles: u64,
    /// This SM's L1 counters when an L1 model was attached.
    pub l1: Option<L1Counters>,
    /// Explicit capability gaps for write-class tails present in this replay.
    pub unmodeled_tails: TailFlags,
    /// Simulator-internal causal readiness diagnostics. These are not NCU
    /// sample counts and their overlapping exposure fields are not elapsed
    /// kernel cycles.
    pub readiness: ReadinessDiagnostics,
}

/// Group witness for one sector request belonging to an effective scalar
/// STORE/RED warp instruction. The system consumes these witnesses only to
/// reconstruct exact per-instruction L2 hit/miss work for opt-in admission
/// candidates; they do not affect hierarchy identity or ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AdmissionRequestWitness(NonZeroU128);

impl AdmissionRequestWitness {
    const VALID: u128 = 1 << 127;
    const U16_MASK: u128 = u16::MAX as u128;
    const U32_MASK: u128 = u32::MAX as u128;

    pub(crate) const fn new(
        warp: u32,
        instruction: u32,
        subcore: u8,
        request_index: u16,
        request_count: u16,
    ) -> Option<Self> {
        if subcore >= SUB_CORES as u8 || request_count == 0 || request_index >= request_count {
            return None;
        }
        let packed = Self::VALID
            | warp as u128
            | ((instruction as u128) << 32)
            | ((subcore as u128) << 64)
            | ((request_index as u128) << 66)
            | ((request_count as u128) << 82);
        match NonZeroU128::new(packed) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    pub(crate) const fn warp(self) -> u32 {
        (self.0.get() & Self::U32_MASK) as u32
    }

    pub(crate) const fn instruction(self) -> u32 {
        ((self.0.get() >> 32) & Self::U32_MASK) as u32
    }

    pub(crate) const fn subcore(self) -> u8 {
        ((self.0.get() >> 64) & 0b11) as u8
    }

    pub(crate) const fn request_index(self) -> u16 {
        ((self.0.get() >> 66) & Self::U16_MASK) as u16
    }

    pub(crate) const fn request_count(self) -> u16 {
        ((self.0.get() >> 82) & Self::U16_MASK) as u16
    }
}

/// A memory request the SM emits in L1Shared mode. The system layer
/// schedules the L2 access at `issue_time` (the engine's global
/// `(time, seq)` order supplies the cross-SM interleaving) and routes the
/// response per `reply`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemRequest {
    pub issue_time: u64,
    /// Physical 32 B hierarchy sector id (`byte_address >> 5`).
    pub sector_id: i64,
    /// Address-space provenance of the anchor sector (`sector_id`). For folded
    /// line fills, [`Self::local_sector_mask`] carries exact per-sector
    /// provenance while this field anchors response-delay attribution. Neither
    /// field participates in L1/L2 tags, fill keys, response identity, or event
    /// ordering.
    pub origin: MemoryOrigin,
    /// Sectors of `sector_id`'s 128 B line carried by THIS request
    /// (bit = sector index in line; always contains `sector_id`'s own
    /// bit). More than one bit only on L1 LINE-fill requests whose
    /// emission batch merged additional sectors onto the fill — NCU
    /// counts each 32 B sector at the LTS, so a multi-sector line fill
    /// must reach the L2 as that many sector accesses, not one request
    /// (§6.4 slice 2; witness accounting lives in the progress log,
    /// never in sim sources — Rule 1). The L2 handler walks the mask
    /// in ascending sector order and prices the reply on the WORST
    /// sector outcome (the fill completes when its slowest sector
    /// arrives).
    pub sector_mask: u8,
    /// Subset of `sector_mask` attributed to Local address-space traffic.
    /// Global provenance is the complement within `sector_mask`. Keeping the
    /// split on the folded physical request preserves one worst-of-mask reply
    /// while allowing L2 hits and DRAM fetches to be accounted per origin.
    pub local_sector_mask: u8,
    pub source: L2Source,
    /// Semantic memory operation. The shared system derives the L2 hit rule
    /// and DRAM-fetch requirement from this value and rejects incoherent
    /// `(source, operation, reply)` combinations before hierarchy mutation.
    pub operation: MemoryOperation,
    pub reply: ReplyTo,
    pub(crate) admission_witness: Option<AdmissionRequestWitness>,
}

/// Process-local identity of one live shared-L2 actor. It is deliberately
/// opaque and never enters simulator outputs; its only role is to prevent a
/// response capability emitted by one actor or replay generation from being
/// accepted by another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ResponseOwner(u64);

static NEXT_RESPONSE_OWNER: AtomicU64 = AtomicU64::new(1);

fn allocate_response_owner_from(next: &AtomicU64) -> Result<ResponseOwner, IssueError> {
    next.fetch_update(
        Ordering::Relaxed,
        Ordering::Relaxed,
        |current| match current {
            0 => None,
            u64::MAX => Some(0),
            _ => current.checked_add(1),
        },
    )
    .map(ResponseOwner)
    .map_err(|_| IssueError::ResponseOwnerExhausted)
}

fn allocate_response_owner() -> Result<ResponseOwner, IssueError> {
    allocate_response_owner_from(&NEXT_RESPONSE_OWNER)
}

/// Stable identity of one response-bearing request. Its fields are private:
/// callers may route a token emitted in [`MemRequest::reply`], but cannot mint
/// an identity for a request the SM never emitted. The owner component is
/// process-unique for the lifetime of the process and rejects cross-core and
/// stale-replay token swaps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResponseId {
    owner: ResponseOwner,
    warp: usize,
    instruction: usize,
    sector_index: usize,
}

impl ResponseId {
    const fn new(
        owner: ResponseOwner,
        warp: usize,
        instruction: usize,
        sector_index: usize,
    ) -> Self {
        Self {
            owner,
            warp,
            instruction,
            sector_index,
        }
    }
}

/// What the response to one modeled memory request must do at arrival.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReplyTo {
    /// Complete this L1 fill (installing the line/sector) and resolve its
    /// registered ordinary-barrier and/or async-copy-group targets.
    L1Fill {
        fill: FillKey,
        response_id: ResponseId,
    },
    /// Clear one pending unit on (warp, barrier) directly — bypass and
    /// untagged loads that never touch the modeled L1.
    Warp {
        warp: usize,
        barrier: DepBarrier,
        response_id: ResponseId,
    },
    /// Complete one global-source sector in a warp's non-bulk async-copy
    /// group. The group may still be open or may have been committed while the
    /// request was in flight.
    AsyncCopy {
        warp: usize,
        group: AsyncCopyGroupId,
        /// An unusual static form may also expose the same completion through an
        /// ordinary dependence barrier; both counters clear atomically.
        barrier: Option<DepBarrier>,
        response_id: ResponseId,
    },
    /// No SM dependency consumes returned data. Stores and reductions are
    /// intrinsically non-returning; an ATOM may also use this when its returned
    /// value has no dependence barrier, while retaining `Atomic` request
    /// semantics and its modeled RMW fetch.
    None,
}

impl ReplyTo {
    const fn response_id(self) -> Option<ResponseId> {
        match self {
            Self::L1Fill { response_id, .. }
            | Self::Warp { response_id, .. }
            | Self::AsyncCopy { response_id, .. } => Some(response_id),
            Self::None => None,
        }
    }
}

impl MemRequest {
    const fn local_mask(origin: MemoryOrigin, sector_bit: u8) -> u8 {
        match origin {
            MemoryOrigin::Global => 0,
            MemoryOrigin::Local => sector_bit,
        }
    }

    fn fold_sector_origin(&mut self, origin: MemoryOrigin, sector_bit: u8) {
        self.sector_mask |= sector_bit;
        if origin == MemoryOrigin::Local {
            self.local_sector_mask |= sector_bit;
        }
    }

    /// An L1 fill request (line alloc or resident-sector fetch): Read
    /// class by construction — nothing that fills the modeled L1 is a
    /// write (stores hit-test without allocating, atomics bypass). The
    /// typed constructors keep the (source, operation, reply) coupling in
    /// one place so an invalid combination is unwritable at emission
    /// sites (gate finding 2026-07-07).
    fn l1_fill(
        issue_time: u64,
        sector_id: i64,
        origin: MemoryOrigin,
        fill: FillKey,
        response_id: ResponseId,
    ) -> Self {
        let sector_bit = 1 << (sector_id & 3) as u8;
        Self {
            issue_time,
            sector_id,
            origin,
            sector_mask: sector_bit,
            local_sector_mask: Self::local_mask(origin, sector_bit),
            source: L2Source::L1Fill,
            operation: MemoryOperation::Read,
            reply: ReplyTo::L1Fill { fill, response_id },
            admission_witness: None,
        }
    }

    /// A follow-up sector fetch for a merge onto an older in-flight
    /// fill: same provenance and class as the fill, no reply (the
    /// waiter rides the original fill's response).
    fn l1_followup_fetch(issue_time: u64, sector_id: i64, origin: MemoryOrigin) -> Self {
        let sector_bit = 1 << (sector_id & 3) as u8;
        Self {
            issue_time,
            sector_id,
            origin,
            sector_mask: sector_bit,
            local_sector_mask: Self::local_mask(origin, sector_bit),
            source: L2Source::L1Fill,
            operation: MemoryOperation::Read,
            reply: ReplyTo::None,
            admission_witness: None,
        }
    }

    /// A plain hierarchy store sector (STG/STL): write-class tag-match rule
    /// at the L2, intrinsically non-returning.
    fn store(issue_time: u64, sector_id: i64, origin: MemoryOrigin) -> Self {
        Self::write(issue_time, sector_id, origin, MemoryOperation::Store)
    }

    /// A non-returning hierarchy reduction sector (RED/REDG). Its write data
    /// validates the sector immediately; the unmodeled RMW read is exposed in
    /// [`TailFlags`] rather than silently fabricated as traffic.
    fn reduction(issue_time: u64, sector_id: i64) -> Self {
        Self::write(
            issue_time,
            sector_id,
            MemoryOrigin::Global,
            MemoryOperation::Reduction,
        )
    }

    fn write(
        issue_time: u64,
        sector_id: i64,
        origin: MemoryOrigin,
        operation: MemoryOperation,
    ) -> Self {
        let sector_bit = 1 << (sector_id & 3) as u8;
        Self {
            issue_time,
            sector_id,
            origin,
            sector_mask: sector_bit,
            local_sector_mask: Self::local_mask(origin, sector_bit),
            source: L2Source::Store,
            operation,
            reply: ReplyTo::None,
            admission_witness: None,
        }
    }

    pub(crate) fn has_valid_contract(self) -> bool {
        let base = match (self.source, self.operation, self.reply) {
            (L2Source::L1Fill, MemoryOperation::Read, ReplyTo::L1Fill { .. } | ReplyTo::None)
            | (
                L2Source::Bypass,
                MemoryOperation::Read,
                ReplyTo::Warp { .. } | ReplyTo::AsyncCopy { .. } | ReplyTo::None,
            )
            | (L2Source::Untagged, MemoryOperation::Read, ReplyTo::Warp { .. } | ReplyTo::None)
            | (L2Source::Bypass, MemoryOperation::Atomic, ReplyTo::Warp { .. } | ReplyTo::None)
            | (
                L2Source::Store,
                MemoryOperation::Store | MemoryOperation::Reduction,
                ReplyTo::None,
            ) => true,
            _ => false,
        };
        let admission = match self.admission_witness {
            None => true,
            Some(witness) => {
                self.source == L2Source::Store
                    && self.origin == MemoryOrigin::Global
                    && matches!(
                        self.operation,
                        MemoryOperation::Store | MemoryOperation::Reduction
                    )
                    && witness.request_count() != 0
                    && witness.request_index() < witness.request_count()
                    && usize::from(witness.subcore()) < SUB_CORES
            }
        };
        base && admission
    }
}

/// Exclusive limit for one [`SmCore::advance`] call. A bounded horizon may be
/// any cycle, including `u64::MAX`; unbounded replay is represented by its own
/// variant instead of stealing a valid timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayHorizon {
    Bounded(u64),
    Unbounded,
}

impl ReplayHorizon {
    fn bounded_time(self) -> Option<u64> {
        match self {
            Self::Bounded(time) => Some(time),
            Self::Unbounded => None,
        }
    }

    fn reached(self, now: u64) -> bool {
        self.bounded_time().is_some_and(|bound| now >= bound)
    }

    fn tighten(self, bound: u64) -> Self {
        match self {
            Self::Bounded(current) => Self::Bounded(current.min(bound)),
            Self::Unbounded => Self::Bounded(bound),
        }
    }
}

/// Outcome of one [`SmCore::advance`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Advance {
    /// The SM cannot act strictly before `wake_at` (if `Some`); with
    /// `None`, only a memory response can unblock it.
    Idle { wake_at: Option<u64> },
    /// All warps of all blocks retired.
    Done,
}

const SUB_CORES: usize = 4;
const DEP_BARRIERS: usize = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResultWritebackKind {
    FixedHadd2 { bank: u8 },
    L1TexStrongSmU32 { bank: u8 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PendingVariableWriteback {
    subcore: u8,
    bank: u8,
    writeback_at: u64,
    visible_at: u64,
    warp: usize,
    barrier: DepBarrier,
}

#[derive(Debug)]
struct ResultWritebackState {
    timing: FixedL1TexWritebackPriority,
    pending: Vec<PendingVariableWriteback>,
}

fn scalar_regular_destination_bank(opcode_text: &str) -> Option<u8> {
    let (_, operands) = opcode_text.trim().split_once(char::is_whitespace)?;
    let destination = operands.split(',').next()?.trim();
    let index = destination.strip_prefix('R')?.parse::<u16>().ok()?;
    Some((index & 1) as u8)
}

fn is_clocklo_issue(slot: &IssueSlot) -> bool {
    slot.is_clocklo_read()
}

fn is_gpu_membar_issue(slot: &IssueSlot) -> bool {
    slot.is_gpu_membar()
}

fn result_writeback_kind(slot: &IssueSlot, arch: ArchId) -> Option<ResultWritebackKind> {
    if arch != ArchId::Sm86 {
        return None;
    }
    if let FixedLatencyOperandStatus::Projected(projection) = slot.fixed_latency_operands()
        && projection.opcode() == FixedLatencyOpcode::Hadd2
        && let [destination] = projection.destinations()
        && let Some(bank) = destination.register().sm86_regular_bank()
    {
        return Some(ResultWritebackKind::FixedHadd2 { bank });
    }
    if slot.mem_role() == MemRole::EligibleLoad(CacheOpToken::LdgEStrongSm)
        && slot.opcode_text().split_whitespace().next() == Some("LDG.E.STRONG.SM")
    {
        return scalar_regular_destination_bank(slot.opcode_text())
            .map(|bank| ResultWritebackKind::L1TexStrongSmU32 { bank });
    }
    None
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum DependencyOrigin {
    Read,
    GlobalLoadOperandRead,
    GlobalStoreOperandRead,
    GlobalAtomicOperandRead,
    GlobalReductionOperandRead,
    NonMemoryWrite,
    SharedWrite,
    L1Tex,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ReadinessCause {
    DependenceCounterRead,
    DependenceCounterGlobalLoadOperandRead,
    DependenceCounterGlobalStoreOperandRead,
    DependenceCounterGlobalAtomicOperandRead,
    DependenceCounterGlobalReductionOperandRead,
    DependenceCounterNonMemoryWrite,
    DependenceCounterSharedWrite,
    DependenceCounterL1Tex,
    DependenceCounterUnknownOrigin,
    AsyncSource,
    AsyncDestinationLocal,
    AsyncDestinationSmService,
    CtaBarrierRendezvous,
    PipelineBarrier,
    PipelineBranch,
    PipelineTensor,
    MemoryAdmissionQueue,
    MemoryWriteCompletion,
    ControlStall,
}

impl From<DependencyOrigin> for ReadinessCause {
    fn from(origin: DependencyOrigin) -> Self {
        match origin {
            DependencyOrigin::Read => Self::DependenceCounterRead,
            DependencyOrigin::GlobalLoadOperandRead => Self::DependenceCounterGlobalLoadOperandRead,
            DependencyOrigin::GlobalStoreOperandRead => {
                Self::DependenceCounterGlobalStoreOperandRead
            }
            DependencyOrigin::GlobalAtomicOperandRead => {
                Self::DependenceCounterGlobalAtomicOperandRead
            }
            DependencyOrigin::GlobalReductionOperandRead => {
                Self::DependenceCounterGlobalReductionOperandRead
            }
            DependencyOrigin::NonMemoryWrite => Self::DependenceCounterNonMemoryWrite,
            DependencyOrigin::SharedWrite => Self::DependenceCounterSharedWrite,
            DependencyOrigin::L1Tex => Self::DependenceCounterL1Tex,
            DependencyOrigin::Unknown => Self::DependenceCounterUnknownOrigin,
        }
    }
}

impl ReadinessCauseCycles {
    fn add_cause(&mut self, cause: ReadinessCause, delta: u64) -> Result<(), IssueError> {
        let (field, counter): (&'static str, &mut u64) = match cause {
            ReadinessCause::ControlStall => ("control stall readiness", &mut self.control_stall),
            ReadinessCause::DependenceCounterRead => (
                "read-dependence counter readiness",
                &mut self.dependence_counter_read,
            ),
            ReadinessCause::DependenceCounterGlobalLoadOperandRead => (
                "global-load operand-read counter readiness",
                &mut self.dependence_counter_global_load_operand_read,
            ),
            ReadinessCause::DependenceCounterGlobalStoreOperandRead => (
                "global-store operand-read counter readiness",
                &mut self.dependence_counter_global_store_operand_read,
            ),
            ReadinessCause::DependenceCounterGlobalAtomicOperandRead => (
                "global-atomic operand-read counter readiness",
                &mut self.dependence_counter_global_atomic_operand_read,
            ),
            ReadinessCause::DependenceCounterGlobalReductionOperandRead => (
                "global-reduction operand-read counter readiness",
                &mut self.dependence_counter_global_reduction_operand_read,
            ),
            ReadinessCause::DependenceCounterNonMemoryWrite => (
                "non-memory write-dependence counter readiness",
                &mut self.dependence_counter_non_memory_write,
            ),
            ReadinessCause::DependenceCounterSharedWrite => (
                "shared write-dependence counter readiness",
                &mut self.dependence_counter_shared_write,
            ),
            ReadinessCause::DependenceCounterL1Tex => (
                "L1TEX-dependence counter readiness",
                &mut self.dependence_counter_l1tex,
            ),
            ReadinessCause::DependenceCounterUnknownOrigin => (
                "unknown-origin dependence counter readiness",
                &mut self.dependence_counter_unknown_origin,
            ),
            ReadinessCause::AsyncSource => {
                ("asynchronous-copy source readiness", &mut self.async_source)
            }
            ReadinessCause::AsyncDestinationLocal => (
                "asynchronous-copy local destination readiness",
                &mut self.async_destination_local,
            ),
            ReadinessCause::AsyncDestinationSmService => (
                "asynchronous-copy SM destination-service readiness",
                &mut self.async_destination_sm_service,
            ),
            ReadinessCause::CtaBarrierRendezvous => (
                "CTA barrier rendezvous readiness",
                &mut self.cta_barrier_rendezvous,
            ),
            ReadinessCause::PipelineBarrier => {
                ("CTA-barrier pipeline readiness", &mut self.pipeline_barrier)
            }
            ReadinessCause::PipelineBranch => {
                ("branch-resolution readiness", &mut self.pipeline_branch)
            }
            ReadinessCause::PipelineTensor => {
                ("tensor-pipeline readiness", &mut self.pipeline_tensor)
            }
            ReadinessCause::MemoryAdmissionQueue => (
                "memory-admission queue readiness",
                &mut self.memory_admission_queue,
            ),
            ReadinessCause::MemoryWriteCompletion => (
                "memory write-completion readiness",
                &mut self.memory_write_completion,
            ),
        };
        *counter = checked_u64_counter_add(field, *counter, delta)?;
        Ok(())
    }

    fn add_eligible_not_selected(&mut self, delta: u64) -> Result<(), IssueError> {
        self.eligible_not_selected = checked_u64_counter_add(
            "eligible-not-selected readiness",
            self.eligible_not_selected,
            delta,
        )?;
        Ok(())
    }
}

fn checked_u64_counter_add(
    counter: &'static str,
    current: u64,
    add: u64,
) -> Result<u64, IssueError> {
    current
        .checked_add(add)
        .ok_or(IssueError::CounterOverflow { counter })
}

fn checked_usize_counter_add(
    counter: &'static str,
    current: usize,
    add: usize,
) -> Result<usize, IssueError> {
    current
        .checked_add(add)
        .ok_or(IssueError::CounterOverflow { counter })
}

fn checked_u32_counter_add(
    counter: &'static str,
    current: u32,
    add: u32,
) -> Result<u32, IssueError> {
    current
        .checked_add(add)
        .ok_or(IssueError::CounterOverflow { counter })
}

/// Warp-local identity of one nonempty asynchronous-copy group. The numeric
/// sequence is intentionally opaque outside the SM actor; shared-memory replies
/// may route an identity emitted by the actor but cannot construct one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AsyncCopyGroupId(u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AsyncCopyGroup {
    id: AsyncCopyGroupId,
    /// Issue time of the `LDGDEPBAR` that made this group visible to a
    /// subsequent wait. Zero is also the valid initial-cycle timestamp; the
    /// group cannot be observed here until it is committed.
    committed_at: u64,
    /// Source sectors whose shared-L2 response time is not known yet.
    pending_sources: u32,
    /// Maximum known lower-level source completion time across flat/constant
    /// copies and shared-L2 responses.
    source_completion_time: u64,
    /// Route-weighted effective payload in the architecture capability's common
    /// fixed-point work domain. This is added once per issued copy, never inferred
    /// from or duplicated across its source sectors.
    delivery_work: u64,
    /// Exact cache-all destination wavefronts accumulated across this committed
    /// group. Zero retains the payload-only/bypass path.
    destination_wavefronts: u64,
    /// The aggregate SM service is assigned once, after commit and after every
    /// source unit has a known completion time.
    destination_service_queued: bool,
    /// Latest completion assigned by the aggregate per-SM destination service.
    destination_completion_time: u64,
    /// Maximum of the warp-local dependent-completion floor and the aggregate
    /// destination-service completion. Every mutation checks this value before
    /// publishing group state, leaving scheduler readiness infallible.
    completion_time: u64,
}

/// One bypass LDGSTS instruction retained inside its asynchronous-copy group.
/// The production readiness path continues to use [`AsyncCopyGroup`]'s
/// established aggregate fields; this sidecar preserves the event ordering
/// needed to evaluate a future per-copy delivery model without changing time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BypassAsyncCopyObservation {
    group: AsyncCopyGroupId,
    instruction: usize,
    pending_sources: u32,
    source_completion_time: u64,
    delivery_work: u64,
    source_cohorts: u64,
    destination_lines: u64,
    max_bank_wavefronts: u64,
    max_bypass_wavefronts: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BypassAsyncCopyInput {
    instruction: usize,
    source_cohorts: u64,
    destination_lines: u64,
    max_bank_wavefronts: u64,
    max_bypass_wavefronts: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct BypassAsyncCopyProjection {
    copies: u64,
    bank_wavefront_upper_bound: u64,
    bypass_wavefront_upper_bound: u64,
    source_cohorts: u64,
    destination_lines: u64,
    max_bank_wavefronts_sum: u64,
    max_bypass_wavefronts_sum: u64,
    source_spread_cycles: u64,
    per_copy_completion_cycles: u64,
    group_aggregate_excess_cycles: u64,
    topology_candidate_penalty_cycles: u64,
    topology_candidate_completion_cycles: u64,
    topology_candidate_later_cycles: u64,
    topology_candidate_earlier_cycles: u64,
    pipeline_q1_completion_cycles: u64,
    pipeline_q1_exposure_cycles: u64,
    pipeline_q2_completion_cycles: u64,
    pipeline_q2_exposure_cycles: u64,
    pipeline_q4_completion_cycles: u64,
    pipeline_q4_exposure_cycles: u64,
    pipeline_q8_completion_cycles: u64,
    pipeline_q8_exposure_cycles: u64,
    pipeline_q16_completion_cycles: u64,
    pipeline_q16_exposure_cycles: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BypassPipelineCopyCandidate {
    instruction: usize,
    payload_completion: u64,
    topology_penalty: u64,
}

// Timing-neutral local-latency candidate selected on conflict/severe and
// tested on skew. Readiness never consumes these coefficients; schema 15
// reports the resulting frontier so failed depth transfer remains visible.
const BYPASS_TOPOLOGY_CANDIDATE_WAVEFRONT_CYCLES: u64 = 4;
const BYPASS_TOPOLOGY_CANDIDATE_BANK_CYCLES: u64 = 2;
const BYPASS_PIPELINE_CANDIDATE_Q: [u64; 5] = [1, 2, 4, 8, 16];

fn bypass_pipeline_candidate_completion(
    copies: &[BypassPipelineCopyCandidate],
    q: u64,
) -> Result<u64, IssueError> {
    debug_assert!(q != 0, "bypass pipeline acceptance width must be nonzero");
    let mut next_accept = 0_u64;
    let mut group_completion = 0_u64;
    for copy in copies {
        let start = copy.payload_completion.max(next_accept);
        let finish = checked_cycle_add(
            "bypass pipeline candidate per-copy completion",
            start,
            copy.topology_penalty,
        )?;
        let accept_interval = copy.topology_penalty / q + u64::from(copy.topology_penalty % q != 0);
        next_accept = checked_cycle_add(
            "bypass pipeline candidate next acceptance",
            start,
            accept_interval,
        )?;
        group_completion = group_completion.max(finish);
    }
    Ok(group_completion)
}

fn bypass_pipeline_candidate_completions(
    copies: &mut [BypassPipelineCopyCandidate],
) -> Result<[u64; BYPASS_PIPELINE_CANDIDATE_Q.len()], IssueError> {
    copies.sort_by_key(|copy| (copy.payload_completion, copy.instruction));
    let mut completions = [0_u64; BYPASS_PIPELINE_CANDIDATE_Q.len()];
    for (completion, q) in completions.iter_mut().zip(BYPASS_PIPELINE_CANDIDATE_Q) {
        *completion = bypass_pipeline_candidate_completion(copies, q)?;
    }
    Ok(completions)
}

impl AsyncCopyGroup {
    const fn new(id: AsyncCopyGroupId) -> Self {
        Self {
            id,
            committed_at: 0,
            pending_sources: 0,
            source_completion_time: 0,
            delivery_work: 0,
            destination_wavefronts: 0,
            destination_service_queued: false,
            destination_completion_time: 0,
            completion_time: 0,
        }
    }

    const fn is_complete_at(self, now: u64) -> bool {
        self.pending_sources == 0
            && (self.destination_wavefronts == 0 || self.destination_service_queued)
            && self.completion_time <= now
    }

    fn project_delivery(
        self,
        source_ready: u64,
        work_add: u64,
        destination_wavefronts_add: u64,
        params: &IssueModelParams,
    ) -> Result<(u64, u64, u64, u64), IssueError> {
        let delivery_work = checked_u64_counter_add(
            "asynchronous-copy delivery work",
            self.delivery_work,
            work_add,
        )?;
        let source_completion_time = self.source_completion_time.max(source_ready);
        let destination_wavefronts = checked_u64_counter_add(
            "asynchronous-copy destination wavefront",
            self.destination_wavefronts,
            destination_wavefronts_add,
        )?;
        let local_completion_time =
            params.ldgsts_completion_ready(source_completion_time, delivery_work)?;
        let completion_time = local_completion_time.max(self.destination_completion_time);
        Ok((
            source_completion_time,
            delivery_work,
            destination_wavefronts,
            completion_time,
        ))
    }

    fn local_completion_time(self, params: &IssueModelParams) -> Result<u64, IssueError> {
        params.ldgsts_completion_ready(self.source_completion_time, self.delivery_work)
    }

    fn limiting_readiness_cause(
        self,
        params: &IssueModelParams,
    ) -> Result<ReadinessCause, IssueError> {
        let local = self.local_completion_time(params)?;
        Ok(if self.destination_completion_time > local {
            ReadinessCause::AsyncDestinationSmService
        } else if self.delivery_work != 0 || self.destination_wavefronts != 0 {
            ReadinessCause::AsyncDestinationLocal
        } else {
            ReadinessCause::AsyncSource
        })
    }
}

/// Allocated only for warps that issue an effective asynchronous-copy source.
/// Keeping this uncommon state out of every [`WarpState`] preserves the compact
/// scheduler scan used by workloads with no `LDGSTS` traffic.
struct AsyncCopyState {
    /// Non-bulk `cp.async` copies issued since the most recent commit.
    open_group: Option<AsyncCopyGroup>,
    /// Nonempty groups closed by `LDGDEPBAR`, in warp program order.
    committed_groups: VecDeque<AsyncCopyGroup>,
    /// Bypass-only per-instruction source/delivery state. Entries share the
    /// owning group's opaque identity and are removed with the waited prefix.
    bypass_copies: Vec<BypassAsyncCopyObservation>,
    /// Next process-local sequence for this warp. `None` means the complete
    /// u64 domain has been allocated; the final `u64::MAX` identity remains
    /// valid, but another nonempty group fails closed.
    next_sequence: Option<u64>,
}

impl AsyncCopyState {
    const fn new() -> Self {
        Self {
            open_group: None,
            committed_groups: VecDeque::new(),
            bypass_copies: Vec::new(),
            next_sequence: Some(0),
        }
    }
}

/// Cold sidecar indexed by warp identity. Keeping even the nullable pointer out
/// of [`WarpState`] preserves that structure's pre-async stride for scheduler
/// scans while retaining direct, deterministic lookup for source responses.
#[derive(Default)]
struct AsyncCopySlot {
    state: Option<Box<AsyncCopyState>>,
}

struct WarpState<'a> {
    insts: &'a [ReplayInst],
    sectors: &'a [i64],
    next: usize,
    next_slot: Option<&'a IssueSlot>,
    /// Issue/admission timestamp at which `next_slot` became the current
    /// instruction. Causal readiness delays are measured from this frontier.
    current_since: u64,
    eligible: u64,
    /// Compiler-emitted `DEPBAR.LE SB1..SB5, 0x0` operand for the current
    /// instruction. This is cached when `next_slot` changes so scheduler scans
    /// use the ordinary barrier counter/clear path without reparsing SASS.
    ordinary_dependency_wait: Option<DepBarrier>,
    barrier_clear: [u64; DEP_BARRIERS],
    barrier_origin: [DependencyOrigin; DEP_BARRIERS],
    /// Outstanding UNKNOWN write-backs per barrier index (L1Shared mode):
    /// hardware barriers are counters, and a response still in flight is
    /// an increment whose decrement time is unknown — a wait on the index
    /// cannot pass, and the warp has no computable ready time, until
    /// every pending unit resolves.
    barrier_pending: [u32; DEP_BARRIERS],
    block: usize,
    launch_index: i64,
    warp_slot: i32,
    /// Index into its sub-core's pick list while live (retired warps are
    /// swap-removed from the list; the swapped-in warp's position is
    /// fixed up). A warp is only ever picked from its own sub-core, so
    /// the retiring sub-core index is the loop's `sc`.
    pos_in_subcore: usize,
    arrival: u64,
    at_block_barrier: bool,
    /// CTA-release arm for the current post-barrier instruction. This is kept
    /// separate from compiler-control eligibility so attribution cannot confuse
    /// a CTA rendezvous with a SASS stall count.
    cta_barrier_ready: u64,
    /// Completion of this warp's own arrival through the SM-shared barrier
    /// service. The CTA arm above carries only other-participant rendezvous.
    barrier_pipeline_ready: u64,
    branch_pipeline_ready: u64,
    block_barrier_dispatch: Option<u64>,
    block_barrier_arrival: Option<u64>,
    clock_first_issue: Option<u64>,
    clock_reads_issued: u64,
    done: bool,
}

struct BlockState {
    unfinished: usize,
    waiting: usize,
    warp_count: u32,
    shared_memory_bytes: u64,
    /// Physical slots held while this block is resident. The trace does not carry
    /// CTA lifetime, so modeled overlap may admit another block only when its slot
    /// set is disjoint from every currently resident block.
    warp_slots: Box<[i32]>,
    /// This block's warps occupy `warps[first_warp..first_warp +
    /// admitted_warps]` (admission pushes them contiguously), so barrier
    /// release scans only the block's own range.
    first_warp: usize,
    admitted_warps: usize,
    barrier_generation: u64,
    barrier_first_dispatch: Option<u64>,
    barrier_first_arrival: Option<u64>,
    /// Earliest issue time of this CTA's next barrier generation. It gates
    /// barrier instructions only; ordinary work after rendezvous is unaffected.
    barrier_rearm_ready: u64,
}

impl WarpState<'_> {
    fn next_issue_retires(&self) -> Result<bool, IssueError> {
        let next = checked_usize_counter_add("warp instruction cursor", self.next, 1)?;
        Ok(next == self.insts.len())
    }
}

impl AsyncCopySlot {
    fn preflight_async_copy_add(
        &self,
        warp: usize,
        pending_add: u32,
        source_ready: u64,
        delivery_work: u64,
        destination_wavefronts: u64,
        bypass_copy: Option<BypassAsyncCopyInput>,
        params: &IssueModelParams,
    ) -> Result<AsyncCopyGroupId, IssueError> {
        let (id, current, group) = match self.state.as_deref() {
            Some(state) => match state.open_group {
                Some(group) => (group.id, group.pending_sources, group),
                None => {
                    let id = AsyncCopyGroupId(
                        state
                            .next_sequence
                            .ok_or(IssueError::AsyncCopyGroupSequenceExhausted { warp })?,
                    );
                    (id, 0, AsyncCopyGroup::new(id))
                }
            },
            None => {
                let id = AsyncCopyGroupId(0);
                (id, 0, AsyncCopyGroup::new(id))
            }
        };
        current
            .checked_add(pending_add)
            .ok_or(IssueError::AsyncCopyPendingOverflow {
                warp,
                group: id,
                current,
                add: pending_add,
            })?;
        if let Some(copy) = bypass_copy
            && self.state.as_deref().is_some_and(|state| {
                state.bypass_copies.iter().any(|existing| {
                    existing.group == id && existing.instruction == copy.instruction
                })
            })
        {
            return Err(IssueError::AsyncCopyObservationDuplicate {
                warp,
                group: id,
                instruction: copy.instruction,
            });
        }
        group.project_delivery(source_ready, delivery_work, destination_wavefronts, params)?;
        Ok(id)
    }

    fn ensure_open_async_group(&mut self, warp: usize) -> Result<&mut AsyncCopyGroup, IssueError> {
        let state = self
            .state
            .get_or_insert_with(|| Box::new(AsyncCopyState::new()));
        if state.open_group.is_none() {
            let sequence = state
                .next_sequence
                .ok_or(IssueError::AsyncCopyGroupSequenceExhausted { warp })?;
            state.next_sequence = sequence.checked_add(1);
            state.open_group = Some(AsyncCopyGroup::new(AsyncCopyGroupId(sequence)));
        }
        state
            .open_group
            .as_mut()
            .ok_or(IssueError::AsyncCopyGroupSequenceExhausted { warp })
    }

    fn add_known_async_copy(
        &mut self,
        warp: usize,
        source_ready: u64,
        delivery_work: u64,
        destination_wavefronts: u64,
        params: &IssueModelParams,
    ) -> Result<AsyncCopyGroupId, IssueError> {
        let id = self.preflight_async_copy_add(
            warp,
            0,
            source_ready,
            delivery_work,
            destination_wavefronts,
            None,
            params,
        )?;
        let group = self.ensure_open_async_group(warp)?;
        debug_assert_eq!(group.id, id);
        let (source_completion_time, delivery_work, destination_wavefronts, completion_time) =
            group.project_delivery(source_ready, delivery_work, destination_wavefronts, params)?;
        group.source_completion_time = source_completion_time;
        group.delivery_work = delivery_work;
        group.destination_wavefronts = destination_wavefronts;
        group.completion_time = completion_time;
        Ok(id)
    }

    fn add_pending_async_copy_sources(
        &mut self,
        warp: usize,
        add: u32,
        delivery_work: u64,
        destination_wavefronts: u64,
        bypass_copy: Option<BypassAsyncCopyInput>,
        params: &IssueModelParams,
    ) -> Result<AsyncCopyGroupId, IssueError> {
        let id = self.preflight_async_copy_add(
            warp,
            add,
            0,
            delivery_work,
            destination_wavefronts,
            bypass_copy,
            params,
        )?;
        let group = self.ensure_open_async_group(warp)?;
        debug_assert_eq!(group.id, id);
        let current = group.pending_sources;
        let pending_sources =
            current
                .checked_add(add)
                .ok_or(IssueError::AsyncCopyPendingOverflow {
                    warp,
                    group: id,
                    current,
                    add,
                })?;
        let (
            source_completion_time,
            aggregate_delivery_work,
            destination_wavefronts,
            completion_time,
        ) = group.project_delivery(0, delivery_work, destination_wavefronts, params)?;
        group.pending_sources = pending_sources;
        group.source_completion_time = source_completion_time;
        group.delivery_work = aggregate_delivery_work;
        group.destination_wavefronts = destination_wavefronts;
        group.completion_time = completion_time;
        if let Some(copy) = bypass_copy {
            self.state
                .as_deref_mut()
                .expect("an observed bypass copy has allocated async-copy state")
                .bypass_copies
                .push(BypassAsyncCopyObservation {
                    group: id,
                    instruction: copy.instruction,
                    pending_sources: add,
                    source_completion_time: 0,
                    delivery_work,
                    source_cohorts: copy.source_cohorts,
                    destination_lines: copy.destination_lines,
                    max_bank_wavefronts: copy.max_bank_wavefronts,
                    max_bypass_wavefronts: copy.max_bypass_wavefronts,
                });
        }
        Ok(id)
    }

    fn async_group(&self, id: AsyncCopyGroupId) -> Option<&AsyncCopyGroup> {
        let state = self.state.as_deref()?;
        state
            .open_group
            .as_ref()
            .filter(|group| group.id == id)
            .or_else(|| state.committed_groups.iter().find(|group| group.id == id))
    }

    fn async_group_mut(&mut self, id: AsyncCopyGroupId) -> Option<&mut AsyncCopyGroup> {
        let state = self.state.as_deref_mut()?;
        if state
            .open_group
            .as_ref()
            .is_some_and(|group| group.id == id)
        {
            return state.open_group.as_mut();
        }
        state
            .committed_groups
            .iter_mut()
            .find(|group| group.id == id)
    }

    fn preflight_async_copy_source_completion(
        &self,
        warp: usize,
        id: AsyncCopyGroupId,
        bypass_instruction: Option<usize>,
        remove: u32,
        now: u64,
        params: &IssueModelParams,
    ) -> Result<(), IssueError> {
        let group = self
            .async_group(id)
            .ok_or(IssueError::AsyncCopyGroupNotFound { warp, group: id })?;
        group
            .pending_sources
            .checked_sub(remove)
            .ok_or(IssueError::AsyncCopyPendingUnderflow {
                warp,
                group: id,
                current: group.pending_sources,
                remove,
            })?;
        group.project_delivery(now, 0, 0, params)?;
        if let Some(instruction) = bypass_instruction {
            let copy = self
                .state
                .as_deref()
                .and_then(|state| {
                    state
                        .bypass_copies
                        .iter()
                        .find(|copy| copy.group == id && copy.instruction == instruction)
                })
                .ok_or(IssueError::AsyncCopyObservationNotFound {
                    warp,
                    group: id,
                    instruction,
                })?;
            copy.pending_sources.checked_sub(remove).ok_or(
                IssueError::AsyncCopyObservationPendingUnderflow {
                    warp,
                    group: id,
                    instruction,
                    current: copy.pending_sources,
                    remove,
                },
            )?;
            params.ldgsts_completion_ready(
                copy.source_completion_time.max(now),
                copy.delivery_work,
            )?;
        }
        Ok(())
    }

    fn complete_async_copy_sources(
        &mut self,
        warp: usize,
        id: AsyncCopyGroupId,
        bypass_instruction: Option<usize>,
        remove: u32,
        now: u64,
        params: &IssueModelParams,
    ) -> Result<(), IssueError> {
        self.preflight_async_copy_source_completion(
            warp,
            id,
            bypass_instruction,
            remove,
            now,
            params,
        )?;
        let group = self
            .async_group_mut(id)
            .ok_or(IssueError::AsyncCopyGroupNotFound { warp, group: id })?;
        let current = group.pending_sources;
        let pending_sources =
            current
                .checked_sub(remove)
                .ok_or(IssueError::AsyncCopyPendingUnderflow {
                    warp,
                    group: id,
                    current,
                    remove,
                })?;
        let (source_completion_time, delivery_work, destination_wavefronts, completion_time) =
            group.project_delivery(now, 0, 0, params)?;
        group.pending_sources = pending_sources;
        group.source_completion_time = source_completion_time;
        group.delivery_work = delivery_work;
        group.destination_wavefronts = destination_wavefronts;
        group.completion_time = completion_time;
        if let Some(instruction) = bypass_instruction {
            let copy = self
                .state
                .as_deref_mut()
                .and_then(|state| {
                    state
                        .bypass_copies
                        .iter_mut()
                        .find(|copy| copy.group == id && copy.instruction == instruction)
                })
                .expect("bypass observation was validated before aggregate mutation");
            copy.pending_sources -= remove;
            copy.source_completion_time = copy.source_completion_time.max(now);
        }
        Ok(())
    }

    fn bypass_copy_projection(
        &self,
        warp: usize,
        group: &AsyncCopyGroup,
        params: &IssueModelParams,
        wait_base: u64,
    ) -> Result<Option<BypassAsyncCopyProjection>, IssueError> {
        let Some(state) = self.state.as_deref() else {
            return Ok(None);
        };
        let mut copies = 0_u64;
        let mut bank_wavefront_upper_bound = 0_u64;
        let mut bypass_wavefront_upper_bound = 0_u64;
        let mut source_cohorts = 0_u64;
        let mut destination_lines = 0_u64;
        let mut max_bank_wavefronts_sum = 0_u64;
        let mut max_bypass_wavefronts_sum = 0_u64;
        let mut earliest_source = u64::MAX;
        let mut latest_source = 0_u64;
        let mut per_copy_completion = 0_u64;
        let mut topology_candidate_completion = 0_u64;
        let mut topology_candidate_penalty_cycles = 0_u64;
        let mut pipeline_copies = Vec::new();
        for copy in state
            .bypass_copies
            .iter()
            .filter(|copy| copy.group == group.id)
        {
            if copy.pending_sources != 0 {
                return Err(IssueError::AsyncCopyObservationPendingAtWait {
                    warp,
                    group: group.id,
                    instruction: copy.instruction,
                    pending: copy.pending_sources,
                });
            }
            copies = checked_u64_counter_add("observed bypass asynchronous-copy count", copies, 1)?;
            source_cohorts = checked_u64_counter_add(
                "observed bypass source cohorts",
                source_cohorts,
                copy.source_cohorts,
            )?;
            destination_lines = checked_u64_counter_add(
                "observed bypass destination lines",
                destination_lines,
                copy.destination_lines,
            )?;
            max_bank_wavefronts_sum = checked_u64_counter_add(
                "observed bypass maximum bank wavefronts",
                max_bank_wavefronts_sum,
                copy.max_bank_wavefronts,
            )?;
            max_bypass_wavefronts_sum = checked_u64_counter_add(
                "observed bypass maximum projected wavefronts",
                max_bypass_wavefronts_sum,
                copy.max_bypass_wavefronts,
            )?;
            let bank_upper = copy
                .source_cohorts
                .checked_mul(copy.max_bank_wavefronts)
                .ok_or(IssueError::TimeMultiplicationOverflow {
                    operation: "bypass observed bank-wavefront upper bound",
                    lhs: copy.source_cohorts,
                    rhs: copy.max_bank_wavefronts,
                })?;
            bank_wavefront_upper_bound = checked_u64_counter_add(
                "bypass observed bank-wavefront upper bound",
                bank_wavefront_upper_bound,
                bank_upper,
            )?;
            let bypass_upper = copy
                .source_cohorts
                .checked_mul(copy.max_bypass_wavefronts)
                .ok_or(IssueError::TimeMultiplicationOverflow {
                    operation: "bypass observed projected-wavefront upper bound",
                    lhs: copy.source_cohorts,
                    rhs: copy.max_bypass_wavefronts,
                })?;
            bypass_wavefront_upper_bound = checked_u64_counter_add(
                "bypass observed projected-wavefront upper bound",
                bypass_wavefront_upper_bound,
                bypass_upper,
            )?;
            earliest_source = earliest_source.min(copy.source_completion_time);
            latest_source = latest_source.max(copy.source_completion_time);
            let payload_completion =
                params.ldgsts_completion_ready(copy.source_completion_time, copy.delivery_work)?;
            per_copy_completion = per_copy_completion.max(payload_completion);
            let topology_penalty =
                if copy.max_bypass_wavefronts == 0 || copy.max_bank_wavefronts == 0 {
                    0
                } else {
                    let wavefront_penalty = copy
                        .max_bypass_wavefronts
                        .saturating_sub(1)
                        .checked_mul(BYPASS_TOPOLOGY_CANDIDATE_WAVEFRONT_CYCLES)
                        .ok_or(IssueError::TimeMultiplicationOverflow {
                            operation: "bypass topology candidate wavefront penalty",
                            lhs: copy.max_bypass_wavefronts.saturating_sub(1),
                            rhs: BYPASS_TOPOLOGY_CANDIDATE_WAVEFRONT_CYCLES,
                        })?;
                    let bank_penalty = copy
                        .max_bank_wavefronts
                        .saturating_sub(1)
                        .checked_mul(BYPASS_TOPOLOGY_CANDIDATE_BANK_CYCLES)
                        .ok_or(IssueError::TimeMultiplicationOverflow {
                            operation: "bypass topology candidate bank penalty",
                            lhs: copy.max_bank_wavefronts.saturating_sub(1),
                            rhs: BYPASS_TOPOLOGY_CANDIDATE_BANK_CYCLES,
                        })?;
                    checked_cycle_add(
                        "bypass topology candidate local penalty",
                        wavefront_penalty,
                        bank_penalty,
                    )?
                };
            topology_candidate_penalty_cycles = checked_u64_counter_add(
                "bypass topology candidate penalty diagnostic",
                topology_candidate_penalty_cycles,
                topology_penalty,
            )?;
            let candidate_completion = checked_cycle_add(
                "bypass topology candidate per-copy completion",
                payload_completion,
                topology_penalty,
            )?;
            topology_candidate_completion = topology_candidate_completion.max(candidate_completion);
            pipeline_copies.push(BypassPipelineCopyCandidate {
                instruction: copy.instruction,
                payload_completion,
                topology_penalty,
            });
        }
        if copies == 0 {
            return Ok(None);
        }
        let pipeline_candidate_completion =
            bypass_pipeline_candidate_completions(&mut pipeline_copies)?;
        let aggregate_completion = group.local_completion_time(params)?;
        debug_assert!(
            aggregate_completion >= per_copy_completion,
            "group-aggregate delivery must upper-bound independent per-copy delivery"
        );
        Ok(Some(BypassAsyncCopyProjection {
            copies,
            bank_wavefront_upper_bound,
            bypass_wavefront_upper_bound,
            source_cohorts,
            destination_lines,
            max_bank_wavefronts_sum,
            max_bypass_wavefronts_sum,
            source_spread_cycles: latest_source.saturating_sub(earliest_source),
            per_copy_completion_cycles: per_copy_completion.saturating_sub(group.committed_at),
            group_aggregate_excess_cycles: aggregate_completion.saturating_sub(per_copy_completion),
            topology_candidate_penalty_cycles,
            topology_candidate_completion_cycles: topology_candidate_completion
                .saturating_sub(group.committed_at),
            topology_candidate_later_cycles: topology_candidate_completion
                .saturating_sub(aggregate_completion),
            topology_candidate_earlier_cycles: aggregate_completion
                .saturating_sub(topology_candidate_completion),
            pipeline_q1_completion_cycles: pipeline_candidate_completion[0]
                .saturating_sub(group.committed_at),
            pipeline_q1_exposure_cycles: pipeline_candidate_completion[0].saturating_sub(wait_base),
            pipeline_q2_completion_cycles: pipeline_candidate_completion[1]
                .saturating_sub(group.committed_at),
            pipeline_q2_exposure_cycles: pipeline_candidate_completion[1].saturating_sub(wait_base),
            pipeline_q4_completion_cycles: pipeline_candidate_completion[2]
                .saturating_sub(group.committed_at),
            pipeline_q4_exposure_cycles: pipeline_candidate_completion[2].saturating_sub(wait_base),
            pipeline_q8_completion_cycles: pipeline_candidate_completion[3]
                .saturating_sub(group.committed_at),
            pipeline_q8_exposure_cycles: pipeline_candidate_completion[3].saturating_sub(wait_base),
            pipeline_q16_completion_cycles: pipeline_candidate_completion[4]
                .saturating_sub(group.committed_at),
            pipeline_q16_exposure_cycles: pipeline_candidate_completion[4]
                .saturating_sub(wait_base),
        }))
    }

    fn commit_async_copy_group(&mut self, now: u64) -> bool {
        let Some(state) = self.state.as_deref_mut() else {
            return false;
        };
        if let Some(mut group) = state.open_group.take() {
            group.committed_at = now;
            state.committed_groups.push_back(group);
            true
        } else {
            false
        }
    }

    fn async_wait_prefix_len(&self, max_pending_groups: u32) -> usize {
        let retained = usize::try_from(max_pending_groups).unwrap_or(usize::MAX);
        self.state.as_deref().map_or(0, |state| {
            state.committed_groups.len().saturating_sub(retained)
        })
    }

    /// Earliest known completion of every group older than the newest `N`.
    /// `None` means at least one required group still has a shared-L2 response
    /// outstanding, so only an external response event can supply a wake time.
    #[cold]
    #[inline(never)]
    fn async_wait_clear_time(&self, max_pending_groups: u32) -> Option<u64> {
        let Some(state) = self.state.as_deref() else {
            return Some(0);
        };
        let mut ready = 0;
        for group in state
            .committed_groups
            .iter()
            .take(self.async_wait_prefix_len(max_pending_groups))
        {
            if group.pending_sources != 0 {
                return None;
            }
            if group.destination_wavefronts != 0 && !group.destination_service_queued {
                return None;
            }
            ready = ready.max(group.completion_time);
        }
        Some(ready)
    }

    fn consume_async_wait(&mut self, max_pending_groups: u32, now: u64) {
        let consume = self.async_wait_prefix_len(max_pending_groups);
        if consume == 0 {
            return;
        }
        let state = self
            .state
            .as_deref_mut()
            .expect("a nonempty wait prefix requires async-copy state");
        debug_assert!(
            state
                .committed_groups
                .iter()
                .take(consume)
                .all(|group| group.is_complete_at(now)),
            "scheduler issued an asynchronous-copy wait before its required prefix completed"
        );
        let consumed_through = state.committed_groups[consume - 1].id.0;
        state
            .bypass_copies
            .retain(|copy| copy.group.0 > consumed_through);
        state.committed_groups.drain(..consume);
    }
}

impl WarpState<'_> {
    fn preflight_pending_add(
        &self,
        warp: usize,
        barrier: Option<DepBarrier>,
        add: u32,
    ) -> Result<(), IssueError> {
        let Some(barrier) = barrier else {
            return Ok(());
        };
        let current = self.barrier_pending[barrier.index()];
        current
            .checked_add(add)
            .ok_or(IssueError::PendingBarrierOverflow {
                warp,
                barrier: barrier.wire(),
                current,
                add,
            })?;
        Ok(())
    }

    fn preflight_pending_remove(
        &self,
        warp: usize,
        barrier: DepBarrier,
        remove: u32,
    ) -> Result<(), IssueError> {
        let current = self.barrier_pending[barrier.index()];
        current
            .checked_sub(remove)
            .ok_or(IssueError::PendingBarrierUnderflow {
                warp,
                barrier: barrier.wire(),
                current,
                remove,
            })?;
        Ok(())
    }

    /// Producer bookkeeping: pending (unknown write-back) units on a
    /// dependence barrier; `None` carries no dependency.
    fn add_pending(
        &mut self,
        warp: usize,
        barrier: Option<DepBarrier>,
        add: u32,
    ) -> Result<(), IssueError> {
        self.preflight_pending_add(warp, barrier, add)?;
        if let Some(barrier) = barrier {
            let pending = &mut self.barrier_pending[barrier.index()];
            let current = *pending;
            *pending = current
                .checked_add(add)
                .ok_or(IssueError::PendingBarrierOverflow {
                    warp,
                    barrier: barrier.wire(),
                    current,
                    add,
                })?;
        }
        Ok(())
    }

    /// Pending units resolved at `now`: decrement AND max-accumulate the
    /// clear stamp (hardware counter semantics — the single copy of the
    /// pairing every response path must preserve).
    fn clear_pending(
        &mut self,
        warp: usize,
        barrier: DepBarrier,
        remove: u32,
        now: u64,
    ) -> Result<(), IssueError> {
        self.preflight_pending_remove(warp, barrier, remove)?;
        let b = barrier.index();
        let current = self.barrier_pending[b];
        self.barrier_pending[b] =
            current
                .checked_sub(remove)
                .ok_or(IssueError::PendingBarrierUnderflow {
                    warp,
                    barrier: barrier.wire(),
                    current,
                    remove,
                })?;
        if now >= self.barrier_clear[b] {
            self.barrier_clear[b] = now;
            self.barrier_origin[b] = DependencyOrigin::L1Tex;
        }
        Ok(())
    }

    /// Release pending units WITHOUT a stamp (a retry entry's own units;
    /// its per-sector waiters carry the stamps).
    fn release_pending(
        &mut self,
        warp: usize,
        barrier: Option<DepBarrier>,
        remove: u32,
    ) -> Result<(), IssueError> {
        let Some(barrier) = barrier else {
            return Ok(());
        };
        self.preflight_pending_remove(warp, barrier, remove)?;
        let pending = &mut self.barrier_pending[barrier.index()];
        let current = *pending;
        *pending = current
            .checked_sub(remove)
            .ok_or(IssueError::PendingBarrierUnderflow {
                warp,
                barrier: barrier.wire(),
                current,
                remove,
            })?;
        Ok(())
    }

    /// Max-accumulate a KNOWN clear stamp on a dependence barrier (counter
    /// semantics: never overwrite); `None` carries no dependency.
    fn stamp_clear(&mut self, barrier: Option<DepBarrier>, at: u64, origin: DependencyOrigin) {
        if let Some(barrier) = barrier {
            let index = barrier.index();
            if at >= self.barrier_clear[index] {
                self.barrier_clear[index] = at;
                self.barrier_origin[index] = origin;
            }
        }
    }

    /// Earliest time the control wait mask and any explicit ordinary `DEPBAR`
    /// operand clear; `None` while a required barrier has responses outstanding
    /// (unknown clear time). SB0 asynchronous-copy waits are handled separately.
    fn wait_clear_time(&self) -> Option<u64> {
        let Some(slot) = self.next_slot else {
            return Some(0);
        };
        let mut t = 0;
        for b in 0..DEP_BARRIERS {
            if slot.control.wait_mask.contains_index(b) {
                if self.barrier_pending[b] > 0 {
                    return None;
                }
                t = t.max(self.barrier_clear[b]);
            }
        }
        if let Some(barrier) = self.ordinary_dependency_wait {
            let b = barrier.index();
            if self.barrier_pending[b] > 0 {
                return None;
            }
            t = t.max(self.barrier_clear[b]);
        }
        Some(t)
    }

    fn ready_at(&self) -> Option<u64> {
        if self.done || self.at_block_barrier {
            return None;
        }
        let dependency_ready = self.wait_clear_time()?;
        Some(
            self.eligible
                .max(dependency_ready)
                .max(self.cta_barrier_ready)
                .max(self.barrier_pipeline_ready)
                .max(self.branch_pipeline_ready),
        )
    }
}

fn async_copy_source_count(kind: AccessKind) -> Option<u16> {
    match kind {
        AccessKind::GlobalAsyncCopy {
            len,
            completion_work,
            ..
        } if completion_work != 0 => Some(len),
        AccessKind::GlobalAsyncCopy { .. }
        | AccessKind::NonMemory
        | AccessKind::PredicatedOffMemory
        | AccessKind::Shared { .. }
        | AccessKind::LocalLoad { .. }
        | AccessKind::LocalStore { .. }
        | AccessKind::GlobalEligible { .. }
        | AccessKind::GlobalBypass { .. }
        | AccessKind::GlobalUntagged { .. }
        | AccessKind::GlobalStore { .. }
        | AccessKind::GlobalReduction { .. } => None,
    }
}

fn async_copy_completion_work(kind: AccessKind) -> Option<u64> {
    match kind {
        AccessKind::GlobalAsyncCopy {
            completion_work, ..
        } => match decode_async_copy_completion(completion_work) {
            AsyncCopyCompletion::Inactive => None,
            AsyncCopyCompletion::PayloadWork(work)
            | AsyncCopyCompletion::ExactDestinationWork(work) => Some(u64::from(work)),
            AsyncCopyCompletion::PayloadAndObservedTopology { payload_work, .. } => {
                Some(u64::from(payload_work))
            }
        },
        AccessKind::NonMemory
        | AccessKind::PredicatedOffMemory
        | AccessKind::Shared { .. }
        | AccessKind::LocalLoad { .. }
        | AccessKind::LocalStore { .. }
        | AccessKind::GlobalEligible { .. }
        | AccessKind::GlobalBypass { .. }
        | AccessKind::GlobalUntagged { .. }
        | AccessKind::GlobalStore { .. }
        | AccessKind::GlobalReduction { .. } => None,
    }
}

fn bypass_async_copy_observed_topology(
    kind: AccessKind,
    instruction: usize,
) -> Option<BypassAsyncCopyInput> {
    let AccessKind::GlobalAsyncCopy {
        completion_work,
        l1_policy: LdgstsL1Policy::BypassL1,
        ..
    } = kind
    else {
        return None;
    };
    match decode_async_copy_completion(completion_work) {
        AsyncCopyCompletion::PayloadAndObservedTopology {
            source_cohorts,
            destination_lines,
            max_bank_wavefronts,
            max_bypass_wavefronts,
            ..
        } => Some(BypassAsyncCopyInput {
            instruction,
            source_cohorts: u64::from(source_cohorts),
            destination_lines: u64::from(destination_lines),
            max_bank_wavefronts: u64::from(max_bank_wavefronts),
            max_bypass_wavefronts: u64::from(max_bypass_wavefronts),
        }),
        AsyncCopyCompletion::Inactive
        | AsyncCopyCompletion::PayloadWork(_)
        | AsyncCopyCompletion::ExactDestinationWork(_) => None,
    }
}

fn async_copy_destination_wavefronts(
    kind: AccessKind,
    params: &IssueModelParams,
) -> Result<Option<u64>, IssueError> {
    let AccessKind::GlobalAsyncCopy {
        completion_work,
        l1_policy: LdgstsL1Policy::CacheAtAllLevels,
        ..
    } = kind
    else {
        return Ok(None);
    };
    let AsyncCopyCompletion::ExactDestinationWork(work) =
        decode_async_copy_completion(completion_work)
    else {
        return Ok(None);
    };
    let work_per_wavefront = params
        .ldgsts_cache_all_destination_wavefront_cycles
        .checked_mul(params.ldgsts_delivery_units_per_cycle())
        .ok_or(IssueError::TimeMultiplicationOverflow {
            operation: "LDGSTS destination work per wavefront",
            lhs: params.ldgsts_cache_all_destination_wavefront_cycles,
            rhs: params.ldgsts_delivery_units_per_cycle(),
        })?;
    let work = u64::from(work);
    if work_per_wavefront == 0 || work % work_per_wavefront != 0 {
        return Err(IssueError::InvalidAsyncCopyDestinationWork {
            completion_work: work,
            work_per_wavefront,
        });
    }
    Ok(Some(work / work_per_wavefront))
}

fn validate_async_copy_completion_work(kind: AccessKind, warp: usize) -> Result<(), IssueError> {
    if let AccessKind::GlobalAsyncCopy {
        len,
        completion_work,
        ..
    } = kind
        && len != 0
        && completion_work == 0
    {
        return Err(IssueError::AsyncCopyCompletionWorkMismatch {
            warp,
            sectors: len,
            completion_work: u64::from(completion_work),
        });
    }
    Ok(())
}

/// `LDGDEPBAR` encodes SB0 in the ordinary write-barrier field, but that bit is
/// the async-group protocol itself rather than a generic 25-cycle producer.
fn ordinary_write_barrier(slot: &IssueSlot) -> Option<DepBarrier> {
    if slot.op_class == OpClass::AsyncCopyCommit {
        None
    } else {
        slot.control.write_barrier
    }
}

/// Data-ready time for one issued record at time `t` in the Flat/L1Const
/// modes (fill times known at issue), driving the L1 state/counters when
/// attached. Producer visibility is computed when either an ordinary write
/// barrier or an async-copy group consumes it; cache accesses still run without
/// either observer because they mutate modeled state.
fn data_ready_time(
    kind: AccessKind,
    sectors: &[i64],
    t: u64,
    params: &IssueModelParams,
    l1: Option<&mut L1Cache>,
    write_barrier: Option<DepBarrier>,
    observe_async_copy: bool,
) -> Result<Option<u64>, IssueError> {
    let needs_ready = write_barrier.is_some() || observe_async_copy;
    match kind {
        AccessKind::PredicatedOffMemory => Ok(None),
        AccessKind::NonMemory => {
            if needs_ready {
                params.non_memory_ready(t).map(Some)
            } else {
                Ok(None)
            }
        }
        AccessKind::Shared { wavefronts } => {
            if needs_ready {
                params.shared_ready(t, wavefronts).map(Some)
            } else {
                Ok(None)
            }
        }
        AccessKind::GlobalEligible { off, len, .. } | AccessKind::LocalLoad { off, len } => {
            let origin = match kind {
                AccessKind::GlobalEligible { .. } => MemoryOrigin::Global,
                AccessKind::LocalLoad { .. } => MemoryOrigin::Local,
                _ => unreachable!("match arm admits only hierarchy loads"),
            };
            match l1 {
                Some(l1) => {
                    let mut ready = needs_ready
                        .then(|| {
                            checked_cycle_add("L1 hit-pipeline floor", t, params.l1_hit_latency)
                        })
                        .transpose()?;
                    for &sector in &sectors[off..off + usize::from(len)] {
                        if let Some(current) = &mut ready {
                            let (_, sector_ready) = l1.access_eligible(origin, sector, t)?;
                            *current = (*current).max(sector_ready);
                        } else {
                            l1.access_eligible_unobserved(origin, sector, t)?;
                        }
                    }
                    Ok(ready)
                }
                None => {
                    if needs_ready {
                        checked_cycle_add("flat global visibility", t, params.global_flat_latency)
                            .map(Some)
                    } else {
                        Ok(None)
                    }
                }
            }
        }
        AccessKind::GlobalBypass { len, token, .. } => match l1 {
            Some(l1) => {
                l1.count_bypass(
                    MemoryOrigin::Global,
                    u64::from(len),
                    token == CacheOpToken::Atom,
                )?;
                if needs_ready {
                    params.store_visibility_ready(t).map(Some)
                } else {
                    Ok(None)
                }
            }
            None => {
                if needs_ready {
                    checked_cycle_add("flat global visibility", t, params.global_flat_latency)
                        .map(Some)
                } else {
                    Ok(None)
                }
            }
        },
        AccessKind::GlobalAsyncCopy {
            off,
            len,
            l1_policy,
            ..
        } => match (l1, l1_policy) {
            (Some(l1), LdgstsL1Policy::CacheAtAllLevels) => {
                let mut ready = needs_ready
                    .then(|| {
                        checked_cycle_add("LDGSTS L1 hit-pipeline floor", t, params.l1_hit_latency)
                    })
                    .transpose()?;
                for &sector in &sectors[off..off + usize::from(len)] {
                    if let Some(current) = &mut ready {
                        let (_, sector_ready) = l1.access_eligible_with_mshr_floor(
                            MemoryOrigin::Global,
                            sector,
                            t,
                            params.ldgsts_l1_mshr_floor,
                        )?;
                        *current = (*current).max(sector_ready);
                    } else {
                        l1.access_eligible_unobserved_with_mshr_floor(
                            MemoryOrigin::Global,
                            sector,
                            t,
                            params.ldgsts_l1_mshr_floor,
                        )?;
                    }
                }
                Ok(ready)
            }
            (Some(l1), LdgstsL1Policy::BypassL1) => {
                l1.count_bypass(MemoryOrigin::Global, u64::from(len), false)?;
                if needs_ready {
                    params.store_visibility_ready(t).map(Some)
                } else {
                    Ok(None)
                }
            }
            (None, _) => {
                if needs_ready {
                    checked_cycle_add("flat global visibility", t, params.global_flat_latency)
                        .map(Some)
                } else {
                    Ok(None)
                }
            }
        },
        AccessKind::GlobalUntagged { len, .. } => match l1 {
            Some(l1) => {
                l1.count_untagged(MemoryOrigin::Global, u64::from(len))?;
                if needs_ready {
                    params.store_visibility_ready(t).map(Some)
                } else {
                    Ok(None)
                }
            }
            None => {
                if needs_ready {
                    checked_cycle_add("flat global visibility", t, params.global_flat_latency)
                        .map(Some)
                } else {
                    Ok(None)
                }
            }
        },
        AccessKind::GlobalStore { off, len }
        | AccessKind::GlobalReduction { off, len, .. }
        | AccessKind::LocalStore { off, len } => {
            let origin = match kind {
                AccessKind::GlobalStore { .. } | AccessKind::GlobalReduction { .. } => {
                    MemoryOrigin::Global
                }
                AccessKind::LocalStore { .. } => MemoryOrigin::Local,
                _ => unreachable!("match arm admits only hierarchy writes"),
            };
            let reduction = matches!(kind, AccessKind::GlobalReduction { .. });
            match l1 {
                Some(l1) => {
                    for &sector in &sectors[off..off + usize::from(len)] {
                        if reduction {
                            l1.probe_reduction(origin, sector, t)?;
                        } else {
                            l1.probe_store(origin, sector, t)?;
                        }
                    }
                    if needs_ready {
                        params.store_visibility_ready(t).map(Some)
                    } else {
                        Ok(None)
                    }
                }
                None => {
                    if needs_ready {
                        checked_cycle_add("flat global visibility", t, params.global_flat_latency)
                            .map(Some)
                    } else {
                        Ok(None)
                    }
                }
            }
        }
    }
}

/// Completion identity carried by an eligible L1 access. Ordinary loads only
/// observe a dependence barrier; cacheable asynchronous-copy sources also retain
/// their open group while optionally preserving an ordinary barrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FillTarget {
    Barrier(DepBarrier),
    AsyncCopy {
        group: AsyncCopyGroupId,
        barrier: Option<DepBarrier>,
    },
}

impl FillTarget {
    const fn barrier(self) -> Option<DepBarrier> {
        match self {
            Self::Barrier(barrier) => Some(barrier),
            Self::AsyncCopy { barrier, .. } => barrier,
        }
    }

    const fn async_group(self) -> Option<AsyncCopyGroupId> {
        match self {
            Self::Barrier(_) => None,
            Self::AsyncCopy { group, .. } => Some(group),
        }
    }
}

/// An MSHR-stalled remainder of one instruction's sector list, retried
/// (in FIFO order) whenever a fill completion frees an entry.
struct Retry {
    warp: usize,
    instruction: usize,
    origin: MemoryOrigin,
    /// Completion identity of the stalled instruction. A present ordinary
    /// barrier holds one pending replacement unit until the entry drains;
    /// asynchronous-copy source units were counted at original issue and retain
    /// their group identity across every retry.
    target: Option<FillTarget>,
    /// Access-local finite-admission floor retained from original issue. Only
    /// cacheable LDGSTS retries carry one; ordinary hierarchy traffic uses `None`.
    mshr_floor: Option<NonZeroU32>,
    /// Unresolved slice `warps[warp].sectors[lo..hi]`.
    lo: usize,
    hi: usize,
}

#[derive(Clone, Copy)]
struct FillWaiter {
    warp: usize,
    target: FillTarget,
    units: u32,
}

/// Completion units cleared when one keyed L1 fill completes. Inline for the
/// dominant single-waiter identity and pre-aggregated by `(warp, target)`, so
/// response preflight only rebuilds counts when distinct targets share a fill.
enum Waiters {
    One(FillWaiter),
    Many(Vec<FillWaiter>),
}

impl Waiters {
    fn add(&mut self, warp: usize, target: FillTarget) -> Result<(), IssueError> {
        let new_waiter = FillWaiter {
            warp,
            target,
            units: 1,
        };
        match self {
            Self::One(waiter) if waiter.warp == warp && waiter.target == target => {
                waiter.units = checked_u32_counter_add("fill waiter unit", waiter.units, 1)?;
            }
            Self::One(waiter) => {
                *self = Self::Many(vec![*waiter, new_waiter]);
            }
            Self::Many(waiters) => {
                if let Some(waiter) = waiters
                    .iter_mut()
                    .find(|waiter| waiter.warp == warp && waiter.target == target)
                {
                    waiter.units = checked_u32_counter_add("fill waiter unit", waiter.units, 1)?;
                } else {
                    waiters.push(new_waiter);
                }
            }
        }
        Ok(())
    }

    fn try_for_each(
        &self,
        mut visit: impl FnMut(FillWaiter) -> Result<(), IssueError>,
    ) -> Result<(), IssueError> {
        match self {
            Self::One(waiter) => visit(*waiter),
            Self::Many(waiters) => {
                for &waiter in waiters {
                    visit(waiter)?;
                }
                Ok(())
            }
        }
    }
}

fn accumulate_waiter_removals(
    pending_removals: &mut HashMap<(usize, DepBarrier), u32>,
    async_removals: &mut HashMap<(usize, AsyncCopyGroupId), u32>,
    waiter: FillWaiter,
) -> Result<(), IssueError> {
    if let Some(barrier) = waiter.target.barrier() {
        let remove = pending_removals.entry((waiter.warp, barrier)).or_default();
        *remove = checked_u32_counter_add("response pending-remove budget", *remove, waiter.units)?;
    }
    if let Some(group) = waiter.target.async_group() {
        let remove = async_removals.entry((waiter.warp, group)).or_default();
        *remove =
            checked_u32_counter_add("response async-copy remove budget", *remove, waiter.units)?;
    }
    Ok(())
}

fn project_async_copy_removal(
    projected: &mut HashMap<(usize, AsyncCopyGroupId), u32>,
    async_copies: &[AsyncCopySlot],
    warp: usize,
    group: AsyncCopyGroupId,
    remove: u32,
    now: u64,
    params: &IssueModelParams,
) -> Result<(), IssueError> {
    if warp >= async_copies.len() {
        return Err(IssueError::ResponseWarpOutOfRange {
            warp,
            warps: async_copies.len(),
        });
    }
    let current = if let Some(&current) = projected.get(&(warp, group)) {
        current
    } else {
        async_copies[warp]
            .preflight_async_copy_source_completion(warp, group, None, 0, now, params)?;
        async_copies[warp]
            .async_group(group)
            .ok_or(IssueError::AsyncCopyGroupNotFound { warp, group })?
            .pending_sources
    };
    let next = current
        .checked_sub(remove)
        .ok_or(IssueError::AsyncCopyPendingUnderflow {
            warp,
            group,
            current,
            remove,
        })?;
    projected.insert((warp, group), next);
    Ok(())
}

/// State that EXISTS only in the shared-L2 mode — carrying it inside
/// the [`MemModel`] variant makes a mispaired construction
/// unrepresentable instead of an `expect()` deep in the memory step.
struct SharedL2State {
    l1: L1Cache,
    response_owner: ResponseOwner,
    /// Exact response-bearing request tokens still expected from the system.
    outstanding_replies: HashSet<ReplyTo>,
    /// Barrier units cleared when the keyed L1 fill completes.
    fill_waiters: HashMap<FillKey, Waiters>,
    /// Per in-flight LINE fill: the sector bits already sent to the L2
    /// (the fill request's mask plus any follow-up fetches). A merge
    /// onto a sector NOT yet in the mask sends a follow-up sector
    /// fetch (`ReplyTo::None` — the waiter still rides the original
    /// fill's response, the documented merge timing); a re-touch of a
    /// sent sector sends nothing, like the hardware MSHR. Entries die
    /// with the fill response. Exactly-once is PER FILL, not
    /// per-address-epoch: if a resident line with an in-flight SECTOR
    /// fill is evicted and re-allocated, the new fill may re-fetch a
    /// sector whose zombie fetch is still outstanding (rare
    /// eviction-window corner, documented at the 2026-07-07 gate).
    fill_masks: HashMap<i64, u8>,
    /// Line fills emitted during the CURRENT emission batch (one
    /// `advance` or `on_response` call — the window whose requests are
    /// still unscheduled in the caller's `out`): line id → index in
    /// `out`. A merge onto one of these FOLDS its sector into the fill
    /// request's mask, so the sector participates in worst-of-mask
    /// pricing regardless of which warp/instruction/retry-entry of the
    /// batch touched it (the MSHR accumulates sectors until the SM
    /// yields — gate finding 2026-07-07: per-instruction folding made
    /// the priced arm depend on batch boundaries instead of physics).
    /// Cleared at every batch entry; indices are only valid because
    /// each batch starts with an empty `out`.
    line_folds: Vec<(i64, usize)>,
    retry: VecDeque<Retry>,
}

/// The typed memory model an [`SmCore`] runs under (the public
/// [`SmMemMode`] resolved into owned state).
enum MemModel {
    Flat,
    L1Const(L1Cache),
    L1Shared(SharedL2State),
}

impl MemModel {
    fn l1(&self) -> Option<&L1Cache> {
        match self {
            MemModel::Flat => None,
            MemModel::L1Const(l1) => Some(l1),
            MemModel::L1Shared(st) => Some(&st.l1),
        }
    }

    /// The L1 the CONSTANT-latency data path drives (`None` in Flat and
    /// Shared modes — the shared path never uses `data_ready_time`).
    fn const_l1_mut(&mut self) -> Option<&mut L1Cache> {
        match self {
            MemModel::L1Const(l1) => Some(l1),
            _ => None,
        }
    }
}

/// Resumable single-SM replay actor. See the module docs for the mode
/// semantics and the run-ahead (horizon) contract.
pub(crate) struct SmCore<'a> {
    schedule: &'a FunctionSchedule,
    /// Static schedule values in PC order. Contiguous binary search avoids a
    /// pointer-chasing B-tree lookup for every dynamic issue while preserving
    /// the schedule map as the authoritative join/diagnostic surface.
    schedule_slots: Vec<(u64, &'a IssueSlot)>,
    params: IssueModelParams,
    shared_memory_residency: SharedMemoryResidency,
    physical_warp_slots: u32,
    mem: MemModel,
    blocks: &'a [BlockReplay],
    block_live_warps: Vec<u32>,
    instruction_limit: u64,
    warps: Vec<WarpState<'a>>,
    /// Cold protocol storage parallel to `warps`; entries allocate only after
    /// an effective asynchronous-copy source is issued.
    async_copies: Vec<AsyncCopySlot>,
    /// Construction-time specialization flag. Kernels without async-copy
    /// sources or controls bypass the protocol path once per scheduler batch,
    /// not once per issued instruction or resident-warp scan.
    has_async_copy_protocol: bool,
    block_states: Vec<BlockState>,
    subcore_warps: [Vec<usize>; SUB_CORES],
    hold: [Option<usize>; SUB_CORES],
    /// Architecture-local CTA-barrier issue resource present when the joined
    /// function contains a characterized barrier instruction.
    barrier_issue: Option<CtaBarrierIssue>,
    /// Earliest cycle at which the one SM-shared barrier server can accept
    /// another warp arrival.
    barrier_pipe_ready: u64,
    /// Earliest issue time of each SMSP's local Tensor Core. This timeline is
    /// kernel-local but survives CTA retirement/admission waves on the SM.
    tensor_pipe_ready: [u64; SUB_CORES],
    /// Earliest free cycle of the cache-all LDGSTS destination service shared
    /// by every resident warp on this SM.
    ldgsts_destination_pipe_ready: u64,
    /// Pending barrier-observed L1-hit result writebacks. Allocated only when
    /// one joined function contains both the admitted HADD2 producer and the
    /// admitted sm_86 STRONG.SM scalar load form.
    result_writebacks: Option<ResultWritebackState>,
    unmodeled_tails: TailFlags,
    readiness: ReadinessDiagnostics,
    memory_service_candidates: MemoryServiceCandidateState,
    emit_admission_witnesses: bool,
    memory_admission_feedback: Option<Box<MemoryAdmissionFeedbackState>>,
    issued: u64,
    t: u64,
    arrival_stamp: u64,
    resident_warps: u32,
    resident_shared_memory_bytes: u64,
    resident_warp_slots: HashSet<i32>,
    unfinished_warps: usize,
    next_block: usize,
}

impl<'a> SmCore<'a> {
    /// Build the actor and admit the initial block wave. `blocks` must be
    /// this SM's blocks in launch order; every instruction's pc must
    /// resolve in `schedule` (single-function launches — the replay
    /// driver enforces that before calling). Cache modes attach a FRESH
    /// per-SM L1 (cold cache per kernel, NCU `--cache-control all`
    /// parity).
    pub(crate) fn new_with_residency(
        blocks: &'a [BlockReplay],
        schedule: &'a FunctionSchedule,
        params: &IssueModelParams,
        shared_memory_residency: SharedMemoryResidency,
        mode: SmMemMode,
    ) -> Result<Self, IssueError> {
        params.validate_resource_source()?;
        if shared_memory_residency.max_bytes_per_sm() == 0 {
            return Err(IssueError::SharedMemoryCapacityInvalid {
                arch: params.arch,
                max_bytes_per_sm: shared_memory_residency.max_bytes_per_sm(),
            });
        }
        if params.arch != ArchId::Sm86 {
            return Err(IssueError::ArchitectureUnsupported { arch: params.arch });
        }
        if params.arch != schedule.arch {
            return Err(IssueError::ArchitectureMismatch {
                component: "issue parameters",
                expected: schedule.arch,
                got: params.arch,
            });
        }
        let physical_warp_slots = params
            .arch
            .issue_hypotheses()
            .require("physical_warp_slots")?
            .max_resident_warps_per_sm;
        let schedule_slots = schedule
            .by_pc
            .iter()
            .map(|(&pc, slot)| (pc, slot))
            .collect::<Vec<_>>();
        // Validate the complete dynamic/static join and every actor-cardinality
        // conversion before constructing mutable replay state. A later block
        // admission must not discover an unknown PC or overflow after earlier
        // blocks have already committed instructions.
        let mut block_live_warps = Vec::with_capacity(blocks.len());
        let mut instruction_limit = 0u64;
        let mut arrival_limit = 0u64;
        let has_async_copy_protocol = schedule.by_pc.values().any(|slot| {
            matches!(
                slot.op_class,
                OpClass::AsyncCopyCommit | OpClass::AsyncCopyWait
            ) || slot.mem_role == sim_mem::MemRole::BypassLoad(CacheOpToken::Ldgsts)
        });
        for block in blocks {
            let live_warp_count = block
                .warps
                .iter()
                .filter(|warp| !warp.insts.is_empty())
                .count();
            let live_warps = u32::try_from(live_warp_count).map_err(|_| {
                IssueError::LiveWarpCountOutOfRange {
                    launch_index: block.launch_index,
                    warps: live_warp_count,
                }
            })?;
            if live_warps > params.max_resident_warps_per_sm {
                return Err(IssueError::BlockResidencyLimitExceeded {
                    launch_index: block.launch_index,
                    live_warps,
                    arch: params.arch,
                    max_resident_warps: params.max_resident_warps_per_sm,
                });
            }
            if live_warps != 0
                && shared_memory_residency.bytes_per_cta()
                    > shared_memory_residency.max_bytes_per_sm()
            {
                return Err(IssueError::BlockSharedMemoryLimitExceeded {
                    launch_index: block.launch_index,
                    bytes_per_cta: shared_memory_residency.bytes_per_cta(),
                    arch: params.arch,
                    max_bytes_per_sm: shared_memory_residency.max_bytes_per_sm(),
                });
            }
            let mut block_slots = HashSet::with_capacity(block.warps.len());
            for warp in &block.warps {
                let slot =
                    u32::try_from(warp.warp_slot).map_err(|_| IssueError::WarpSlotOutOfRange {
                        launch_index: block.launch_index,
                        warp_slot: warp.warp_slot,
                        arch: params.arch,
                        physical_warp_slots,
                    })?;
                if slot >= physical_warp_slots {
                    return Err(IssueError::WarpSlotOutOfRange {
                        launch_index: block.launch_index,
                        warp_slot: warp.warp_slot,
                        arch: params.arch,
                        physical_warp_slots,
                    });
                }
                if !block_slots.insert(warp.warp_slot) {
                    return Err(IssueError::DuplicateWarpSlot {
                        launch_index: block.launch_index,
                        warp_slot: warp.warp_slot,
                    });
                }
            }
            block_live_warps.push(live_warps);
            arrival_limit = checked_u64_counter_add(
                "warp arrival stamp",
                arrival_limit,
                u64::from(live_warps),
            )?;
            for warp in &block.warps {
                let warp_instructions =
                    u64::try_from(warp.insts.len()).map_err(|_| IssueError::CounterOverflow {
                        counter: "replay instruction cardinality",
                    })?;
                instruction_limit = checked_u64_counter_add(
                    "replay instruction cardinality",
                    instruction_limit,
                    warp_instructions,
                )?;
                for inst in &warp.insts {
                    let slot = lookup_dense(
                        &schedule_slots,
                        schedule,
                        inst.pc,
                        block.launch_index,
                        warp.warp_slot,
                    )?;
                    slot.control
                        .validate()
                        .map_err(|source| IssueError::ControlDomain {
                            pc: inst.pc,
                            launch_index: block.launch_index,
                            warp_slot: warp.warp_slot,
                            source,
                        })?;
                    validate_sector_span(
                        inst.kind,
                        warp.sectors.len(),
                        inst.pc,
                        block.launch_index,
                        warp.warp_slot,
                    )?;
                }
            }
        }
        let unmodeled_tails = blocks
            .iter()
            .flat_map(|block| &block.warps)
            .flat_map(|warp| &warp.insts)
            .fold(TailFlags::default(), |flags, inst| {
                flags.union(TailFlags::for_access(inst.kind))
            });
        let l1_arch = match mode {
            SmMemMode::Flat => None,
            SmMemMode::L1Const(cfg) | SmMemMode::L1Shared(cfg) => Some(cfg.arch),
        };
        if let Some(got) = l1_arch
            && got != schedule.arch
        {
            return Err(IssueError::ArchitectureMismatch {
                component: "L1 configuration",
                expected: schedule.arch,
                got,
            });
        }
        if let SmMemMode::L1Const(cfg) | SmMemMode::L1Shared(cfg) = mode {
            params.validate_l1_timing(cfg)?;
        }
        let mem = match mode {
            SmMemMode::Flat => MemModel::Flat,
            SmMemMode::L1Const(cfg) => MemModel::L1Const(L1Cache::new(cfg)?),
            SmMemMode::L1Shared(cfg) => MemModel::L1Shared(SharedL2State {
                l1: L1Cache::new(cfg)?,
                response_owner: allocate_response_owner()?,
                outstanding_replies: HashSet::new(),
                fill_waiters: HashMap::new(),
                fill_masks: HashMap::new(),
                line_folds: Vec::new(),
                retry: VecDeque::new(),
            }),
        };
        let mut barrier_issue = None;
        for slot in schedule.by_pc.values() {
            let PipelineClass::Barrier {
                warp_initiation_interval_cycles,
                generation_rearm_cycles,
            } = slot.pipeline_class
            else {
                continue;
            };
            let declared = CtaBarrierIssue {
                warp_initiation_interval_cycles,
                generation_rearm_cycles,
            };
            if let Some(expected) = barrier_issue
                && expected != declared
            {
                return Err(IssueError::BarrierPipelineConfigurationMismatch {
                    expected,
                    got: declared,
                });
            }
            barrier_issue = Some(declared);
        }
        let has_fixed_writeback = schedule.by_pc.values().any(|slot| {
            matches!(
                result_writeback_kind(slot, params.arch),
                Some(ResultWritebackKind::FixedHadd2 { .. })
            )
        });
        let has_variable_writeback = schedule.by_pc.values().any(|slot| {
            matches!(
                result_writeback_kind(slot, params.arch),
                Some(ResultWritebackKind::L1TexStrongSmU32 { .. })
            )
        });
        let result_writebacks = if has_fixed_writeback && has_variable_writeback {
            Some(ResultWritebackState {
                timing: params
                    .arch
                    .fixed_l1tex_writeback_priority()
                    .require("fixed_l1tex_writeback_priority")?,
                pending: Vec::new(),
            })
        } else {
            None
        };
        let mut core = Self {
            schedule,
            schedule_slots,
            params: *params,
            shared_memory_residency,
            physical_warp_slots,
            mem,
            blocks,
            block_live_warps,
            instruction_limit,
            warps: Vec::new(),
            async_copies: Vec::new(),
            has_async_copy_protocol,
            block_states: Vec::new(),
            subcore_warps: Default::default(),
            hold: [None; SUB_CORES],
            barrier_issue,
            barrier_pipe_ready: 0,
            tensor_pipe_ready: [0; SUB_CORES],
            ldgsts_destination_pipe_ready: 0,
            result_writebacks,
            unmodeled_tails,
            readiness: ReadinessDiagnostics::default(),
            memory_service_candidates: MemoryServiceCandidateState::default(),
            emit_admission_witnesses: false,
            memory_admission_feedback: None,
            issued: 0,
            t: 0,
            arrival_stamp: 0,
            resident_warps: 0,
            resident_shared_memory_bytes: 0,
            resident_warp_slots: HashSet::new(),
            unfinished_warps: 0,
            next_block: 0,
        };
        core.admit_blocks(0)?;
        Ok(core)
    }

    pub(crate) fn install_memory_service_candidates(
        &mut self,
        candidates: &[MemoryServiceCandidate],
    ) -> Result<(), IssueError> {
        self.memory_service_candidates = MemoryServiceCandidateState::checked(candidates)?;
        Ok(())
    }

    pub(crate) fn enable_memory_admission_observations(&mut self) {
        self.emit_admission_witnesses = true;
    }

    pub(crate) fn install_memory_admission_feedback(
        &mut self,
        candidate: MemoryAdmissionCandidate,
    ) -> Result<(), IssueError> {
        candidate.validate_feedback(self.params.read_barrier_latency)?;
        self.memory_admission_feedback =
            Some(Box::new(MemoryAdmissionFeedbackState::checked(candidate)?));
        self.emit_admission_witnesses = true;
        Ok(())
    }

    pub(crate) fn complete_memory_admission_feedback(
        &mut self,
        witness: AdmissionRequestWitness,
        operation: MemoryOperation,
        hit_sectors: u64,
        miss_sectors: u64,
        l2_requests: u64,
        unique_regions: u64,
        issue_time: u64,
    ) -> Result<(), IssueError> {
        self.memory_admission_feedback
            .as_mut()
            .ok_or(IssueError::MemoryAdmissionFeedbackReservationMissing {
                warp: witness.warp(),
                instruction: witness.instruction(),
            })?
            .complete(
                witness,
                operation,
                hit_sectors,
                miss_sectors,
                l2_requests,
                unique_regions,
                issue_time,
            )
    }

    #[cfg(test)]
    fn new(
        blocks: &'a [BlockReplay],
        schedule: &'a FunctionSchedule,
        params: &IssueModelParams,
        mode: SmMemMode,
    ) -> Result<Self, IssueError> {
        Self::new_with_residency(
            blocks,
            schedule,
            params,
            SharedMemoryResidency::unconstrained_fixture(),
            mode,
        )
    }

    #[cfg(test)]
    fn now(&self) -> u64 {
        self.t
    }

    pub(crate) fn is_done(&self) -> bool {
        self.unfinished_warps == 0
    }

    #[cfg(test)]
    fn l1_counters(&self) -> Option<&L1Counters> {
        self.mem.l1().map(|cache| &cache.counters)
    }

    /// Responses the system still owes this SM (diagnostics).
    pub(crate) fn outstanding_responses(&self) -> usize {
        match &self.mem {
            MemModel::L1Shared(st) => st.outstanding_replies.len(),
            _ => 0,
        }
    }

    /// Instructions issued so far (monotonic; the system driver's
    /// progress signal).
    pub(crate) fn instructions_issued(&self) -> u64 {
        self.issued
    }

    /// Result snapshot; meaningful once [`Self::advance`] returned
    /// [`Advance::Done`].
    pub(crate) fn result(&self) -> SmSimResult {
        let mut readiness = self.readiness.clone();
        readiness.memory_service_candidates = self.memory_service_candidates.diagnostics();
        if let Some(feedback) = self.memory_admission_feedback.as_ref() {
            readiness.memory_admission_feedback = feedback.diagnostics();
        }
        SmSimResult {
            instructions_issued: self.issued,
            cycles: self.t,
            l1: self.mem.l1().map(|c| c.counters),
            unmodeled_tails: self.unmodeled_tails,
            readiness,
        }
    }

    /// Admit blocks in launch order while all modeled constraints allow: the
    /// resident-warp ceiling, the architecture-card shared-memory capacity, and
    /// disjoint physical warp slots across concurrent CTAs. A colliding later CTA
    /// waits for the earlier CTA to retire; reuse after retirement is valid. A
    /// block whose warps are ALL empty occupies no residency (it can never retire
    /// through the issue path, so counting it would leak resources and block later
    /// admissions forever).
    fn admit_blocks(&mut self, t: u64) -> Result<(), IssueError> {
        while self.next_block < self.blocks.len() {
            let blk = &self.blocks[self.next_block];
            let live_warps = self.block_live_warps[self.next_block];
            let block_shared_memory_bytes = if live_warps == 0 {
                0
            } else {
                self.shared_memory_residency.bytes_per_cta()
            };
            let Some(next_resident_warps) = self.resident_warps.checked_add(live_warps) else {
                // Construction proved this CTA fits alone, so an overflowing sum
                // can only mean it must wait for the current wave to retire.
                if self.resident_warps > 0 {
                    break;
                }
                return Err(IssueError::CounterOverflow {
                    counter: "resident warp",
                });
            };
            let Some(next_resident_shared_memory_bytes) = self
                .resident_shared_memory_bytes
                .checked_add(block_shared_memory_bytes)
            else {
                if self.resident_warps > 0 {
                    break;
                }
                return Err(IssueError::CounterOverflow {
                    counter: "resident shared-memory byte",
                });
            };
            let pending_slots = blk
                .warps
                .iter()
                .filter(|warp| !warp.insts.is_empty())
                .map(|warp| warp.warp_slot)
                .collect::<Vec<_>>();
            let slots_collide = pending_slots
                .iter()
                .any(|slot| self.resident_warp_slots.contains(slot));
            if self.resident_warps > 0
                && (next_resident_warps > self.params.max_resident_warps_per_sm
                    || next_resident_shared_memory_bytes
                        > self.shared_memory_residency.max_bytes_per_sm()
                    || slots_collide)
            {
                break;
            }

            let admitted_warps =
                usize::try_from(live_warps).map_err(|_| IssueError::CounterOverflow {
                    counter: "admitted warp cardinality",
                })?;
            let next_unfinished_warps = checked_usize_counter_add(
                "unfinished warp",
                self.unfinished_warps,
                admitted_warps,
            )?;
            let next_arrival_stamp = checked_u64_counter_add(
                "warp arrival stamp",
                self.arrival_stamp,
                u64::from(live_warps),
            )?;
            let next_block = checked_usize_counter_add("next block", self.next_block, 1)?;

            // Build the complete block state off to the side. A schedule error
            // cannot leave a prefix of the block admitted.
            let block_ix = self.block_states.len();
            let first_warp = self.warps.len();
            let mut subcore_additions = [0usize; SUB_CORES];
            let mut pending_warps = Vec::with_capacity(admitted_warps);
            for w in &blk.warps {
                if w.insts.is_empty() {
                    continue;
                }
                let subcore =
                    usize::try_from(w.warp_slot).map_err(|_| IssueError::WarpSlotOutOfRange {
                        launch_index: blk.launch_index,
                        warp_slot: w.warp_slot,
                        arch: self.params.arch,
                        physical_warp_slots: self.physical_warp_slots,
                    })? % SUB_CORES;
                let pos_in_subcore = checked_usize_counter_add(
                    "sub-core resident warp",
                    self.subcore_warps[subcore].len(),
                    subcore_additions[subcore],
                )?;
                let arrival_offset = u64::try_from(pending_warps.len()).map_err(|_| {
                    IssueError::CounterOverflow {
                        counter: "warp arrival stamp",
                    }
                })?;
                let arrival = checked_u64_counter_add(
                    "warp arrival stamp",
                    self.arrival_stamp,
                    arrival_offset,
                )?;
                let next_slot = lookup_dense(
                    &self.schedule_slots,
                    self.schedule,
                    w.insts[0].pc,
                    blk.launch_index,
                    w.warp_slot,
                )?;
                pending_warps.push((
                    subcore,
                    WarpState {
                        insts: &w.insts,
                        sectors: &w.sectors,
                        next: 0,
                        next_slot: Some(next_slot),
                        current_since: t,
                        eligible: t,
                        ordinary_dependency_wait: next_slot.ordinary_dependency_wait_barrier(),
                        barrier_clear: [0; DEP_BARRIERS],
                        barrier_origin: [DependencyOrigin::Unknown; DEP_BARRIERS],
                        barrier_pending: [0; DEP_BARRIERS],
                        block: block_ix,
                        launch_index: blk.launch_index,
                        warp_slot: w.warp_slot,
                        pos_in_subcore,
                        arrival,
                        at_block_barrier: false,
                        cta_barrier_ready: 0,
                        barrier_pipeline_ready: 0,
                        branch_pipeline_ready: 0,
                        block_barrier_dispatch: None,
                        block_barrier_arrival: None,
                        clock_first_issue: None,
                        clock_reads_issued: 0,
                        done: false,
                    },
                ));
                subcore_additions[subcore] = checked_usize_counter_add(
                    "sub-core resident warp",
                    subcore_additions[subcore],
                    1,
                )?;
            }

            for (subcore, warp) in pending_warps {
                let ix = self.warps.len();
                self.subcore_warps[subcore].push(ix);
                self.warps.push(warp);
                if self.has_async_copy_protocol {
                    self.async_copies.push(AsyncCopySlot::default());
                }
            }
            debug_assert!(
                if self.has_async_copy_protocol {
                    self.warps.len() == self.async_copies.len()
                } else {
                    self.async_copies.is_empty()
                },
                "async-copy sidecar shape must match its construction-time protocol mode"
            );
            for &slot in &pending_slots {
                let inserted = self.resident_warp_slots.insert(slot);
                debug_assert!(
                    inserted,
                    "admission preflight rejected resident slot collisions"
                );
            }
            self.block_states.push(BlockState {
                unfinished: admitted_warps,
                waiting: 0,
                warp_count: live_warps,
                shared_memory_bytes: block_shared_memory_bytes,
                warp_slots: pending_slots.into_boxed_slice(),
                first_warp,
                admitted_warps,
                barrier_generation: 0,
                barrier_first_dispatch: None,
                barrier_first_arrival: None,
                barrier_rearm_ready: 0,
            });
            self.resident_warps = next_resident_warps;
            self.resident_shared_memory_bytes = next_resident_shared_memory_bytes;
            self.unfinished_warps = next_unfinished_warps;
            self.arrival_stamp = next_arrival_stamp;
            self.next_block = next_block;
        }
        Ok(())
    }

    // (Shared-mode memory helpers are free functions over the
    // destructured `SharedL2State` + warp slice so the borrow of
    // `self.mem` never has to span a `&mut self` call.)

    /// Memory interaction of one issued record in L1Shared mode. Returns
    /// the KNOWN data-ready component (`None` when completion depends only on
    /// in-flight responses); unknown components remain in ordinary-barrier
    /// pending units and/or async-copy-group source units.
    fn mem_step_shared(
        &mut self,
        subcore: usize,
        wix: usize,
        instruction: usize,
        kind: AccessKind,
        sectors: &'a [i64],
        write_barrier: Option<DepBarrier>,
        destination_wavefronts: u64,
        out: &mut Vec<MemRequest>,
    ) -> Result<Option<u64>, IssueError> {
        let t = self.t;
        let p = self.params;
        let emit_admission_witnesses = self.emit_admission_witnesses;
        let MemModel::L1Shared(st) = &mut self.mem else {
            unreachable!("mem_step_shared is only dispatched in shared mode")
        };
        let warps = &mut self.warps;
        let async_copies = &mut self.async_copies;
        match kind {
            AccessKind::PredicatedOffMemory => Ok(None),
            AccessKind::NonMemory => {
                if write_barrier.is_some() {
                    p.non_memory_ready(t).map(Some)
                } else {
                    Ok(None)
                }
            }
            AccessKind::Shared { wavefronts } => {
                if write_barrier.is_some() {
                    p.shared_ready(t, wavefronts).map(Some)
                } else {
                    Ok(None)
                }
            }
            AccessKind::GlobalEligible { off, len, .. } | AccessKind::LocalLoad { off, len } => {
                let origin = match kind {
                    AccessKind::GlobalEligible { .. } => MemoryOrigin::Global,
                    AccessKind::LocalLoad { .. } => MemoryOrigin::Local,
                    _ => unreachable!("match arm admits only hierarchy loads"),
                };
                // The hit-pipeline floor applies to the whole dependent load
                // (as in L1Const); misses ride their responses. Line fills
                // emitted anywhere in THIS emission batch (st.line_folds,
                // cleared at advance entry) fold later same-line sectors
                // into their request mask — one L2 transaction per line
                // per batch, priced on its worst sector.
                let target = write_barrier.map(FillTarget::Barrier);
                let mut known = write_barrier
                    .is_some()
                    .then(|| checked_cycle_add("shared L1 hit-pipeline floor", t, p.l1_hit_latency))
                    .transpose()?;
                let range = off..off + usize::from(len);
                for i in range.clone() {
                    match access_shared(
                        st,
                        warps,
                        wix,
                        target,
                        None,
                        origin,
                        sectors[i],
                        ResponseId::new(st.response_owner, wix, instruction, i),
                        t,
                        out,
                    ) {
                        Ok(Some(ready)) => {
                            if let Some(current) = &mut known {
                                *current = (*current).max(ready);
                            }
                        }
                        Ok(None) => {}
                        Err(SharedAccessError::Stalled) => {
                            // MSHR full: park the remainder; the entry
                            // holds one pending unit until drained.
                            warps[wix].add_pending(wix, write_barrier, 1)?;
                            st.retry.push_back(Retry {
                                warp: wix,
                                instruction,
                                origin,
                                target,
                                mshr_floor: None,
                                lo: i,
                                hi: range.end,
                            });
                            break;
                        }
                        Err(SharedAccessError::Issue(source)) => return Err(source),
                    }
                }
                Ok(known)
            }
            AccessKind::GlobalAsyncCopy {
                off,
                len,
                completion_work,
                l1_policy: LdgstsL1Policy::CacheAtAllLevels,
            } => {
                let source_active = completion_work != 0;
                let range = off..off + usize::from(len);
                let delivery_work = async_copy_completion_work(kind).unwrap_or(0);
                let async_group = if source_active {
                    Some(async_copies[wix].add_pending_async_copy_sources(
                        wix,
                        u32::from(len),
                        delivery_work,
                        destination_wavefronts,
                        None,
                        &p,
                    )?)
                } else {
                    None
                };
                let target = async_group
                    .map(|group| FillTarget::AsyncCopy {
                        group,
                        barrier: write_barrier,
                    })
                    .or_else(|| write_barrier.map(FillTarget::Barrier));
                let mut known = (write_barrier.is_some() || source_active)
                    .then(|| {
                        checked_cycle_add(
                            "LDGSTS shared L1 hit-pipeline floor",
                            t,
                            p.l1_hit_latency,
                        )
                    })
                    .transpose()?;
                for i in range.clone() {
                    match access_shared(
                        st,
                        warps,
                        wix,
                        target,
                        Some(p.ldgsts_l1_mshr_floor),
                        MemoryOrigin::Global,
                        sectors[i],
                        ResponseId::new(st.response_owner, wix, instruction, i),
                        t,
                        out,
                    ) {
                        Ok(Some(ready)) => {
                            if let Some(group) = async_group {
                                async_copies[wix]
                                    .complete_async_copy_sources(wix, group, None, 1, ready, &p)?;
                            }
                            if let Some(current) = &mut known {
                                *current = (*current).max(ready);
                            }
                        }
                        Ok(None) => {}
                        Err(SharedAccessError::Stalled) => {
                            // Async source units were counted at original issue;
                            // only the optional ordinary barrier needs a retry
                            // replacement unit.
                            warps[wix].add_pending(wix, write_barrier, 1)?;
                            st.retry.push_back(Retry {
                                warp: wix,
                                instruction,
                                origin: MemoryOrigin::Global,
                                target,
                                mshr_floor: Some(p.ldgsts_l1_mshr_floor),
                                lo: i,
                                hi: range.end,
                            });
                            break;
                        }
                        Err(SharedAccessError::Issue(source)) => return Err(source),
                    }
                }
                Ok(known)
            }
            AccessKind::GlobalBypass { off, len, .. }
            | AccessKind::GlobalAsyncCopy {
                off,
                len,
                l1_policy: LdgstsL1Policy::BypassL1,
                ..
            }
            | AccessKind::GlobalUntagged { off, len } => {
                let origin = MemoryOrigin::Global;
                let (source, counted, operation, async_copy) = match kind {
                    // ATOM rides bypass provenance but remains a returning RMW
                    // operation; every other bypass / untagged sector is a read.
                    AccessKind::GlobalBypass { token, .. } => (
                        L2Source::Bypass,
                        true,
                        if token == CacheOpToken::Atom {
                            MemoryOperation::Atomic
                        } else {
                            MemoryOperation::Read
                        },
                        token == CacheOpToken::Ldgsts,
                    ),
                    AccessKind::GlobalAsyncCopy {
                        completion_work, ..
                    } => (
                        L2Source::Bypass,
                        true,
                        MemoryOperation::Read,
                        completion_work != 0,
                    ),
                    AccessKind::GlobalUntagged { .. } => {
                        (L2Source::Untagged, false, MemoryOperation::Read, false)
                    }
                    _ => unreachable!("match arm admits only bypass, async-copy, or untagged"),
                };
                if counted {
                    st.l1.count_bypass(
                        origin,
                        u64::from(len),
                        operation == MemoryOperation::Atomic,
                    )?;
                } else {
                    st.l1.count_untagged(origin, u64::from(len))?;
                }
                if len == 0 {
                    // Degenerate record (fully predicated-off lanes are
                    // legal on the wire): no sectors, no requests. Only a
                    // present write barrier consumes the L2-hit visibility.
                    return if write_barrier.is_some() {
                        p.store_visibility_ready(t).map(Some)
                    } else {
                        Ok(None)
                    };
                }
                let async_group = if async_copy {
                    let delivery_work = async_copy_completion_work(kind).unwrap_or(0);
                    Some(async_copies[wix].add_pending_async_copy_sources(
                        wix,
                        u32::from(len),
                        delivery_work,
                        destination_wavefronts,
                        Some(
                            bypass_async_copy_observed_topology(kind, instruction).unwrap_or(
                                BypassAsyncCopyInput {
                                    instruction,
                                    source_cohorts: 0,
                                    destination_lines: 0,
                                    max_bank_wavefronts: 0,
                                    max_bypass_wavefronts: 0,
                                },
                            ),
                        ),
                        &p,
                    )?)
                } else {
                    None
                };
                for i in off..off + usize::from(len) {
                    let sector_id = sectors[i];
                    let response_id = ResponseId::new(st.response_owner, wix, instruction, i);
                    let reply = if let Some(group) = async_group {
                        if let Some(barrier) = write_barrier {
                            warps[wix].add_pending(wix, Some(barrier), 1)?;
                        }
                        let reply = ReplyTo::AsyncCopy {
                            warp: wix,
                            group,
                            barrier: write_barrier,
                            response_id,
                        };
                        register_response(st, reply)?;
                        reply
                    } else if let Some(barrier) = write_barrier {
                        let reply = ReplyTo::Warp {
                            warp: wix,
                            barrier,
                            response_id,
                        };
                        warps[wix].add_pending(wix, Some(barrier), 1)?;
                        register_response(st, reply)?;
                        reply
                    } else {
                        ReplyTo::None
                    };
                    out.push(MemRequest {
                        issue_time: t,
                        sector_id,
                        origin,
                        sector_mask: 1 << (sector_id & 3) as u8,
                        local_sector_mask: 0,
                        source,
                        operation,
                        reply,
                        admission_witness: None,
                    });
                }
                Ok(None)
            }
            AccessKind::GlobalStore { off, len }
            | AccessKind::GlobalReduction { off, len, .. }
            | AccessKind::LocalStore { off, len } => {
                let origin = match kind {
                    AccessKind::GlobalStore { .. } | AccessKind::GlobalReduction { .. } => {
                        MemoryOrigin::Global
                    }
                    AccessKind::LocalStore { .. } => MemoryOrigin::Local,
                    _ => unreachable!("match arm admits only hierarchy writes"),
                };
                let reduction = matches!(kind, AccessKind::GlobalReduction { .. });
                for (request_index, &sector_id) in
                    sectors[off..off + usize::from(len)].iter().enumerate()
                {
                    let mut request = if reduction {
                        st.l1.probe_reduction(origin, sector_id, t)?;
                        MemRequest::reduction(t, sector_id)
                    } else {
                        st.l1.probe_store(origin, sector_id, t)?;
                        MemRequest::store(t, sector_id, origin)
                    };
                    if origin == MemoryOrigin::Global && emit_admission_witnesses {
                        request.admission_witness = Some(
                            AdmissionRequestWitness::new(
                                u32::try_from(wix).map_err(|_| IssueError::CounterOverflow {
                                    counter: "memory-admission witness warp",
                                })?,
                                u32::try_from(instruction).map_err(|_| {
                                    IssueError::CounterOverflow {
                                        counter: "memory-admission witness instruction",
                                    }
                                })?,
                                u8::try_from(subcore).map_err(|_| IssueError::CounterOverflow {
                                    counter: "memory-admission witness subcore",
                                })?,
                                u16::try_from(request_index).map_err(|_| {
                                    IssueError::CounterOverflow {
                                        counter: "memory-admission witness request index",
                                    }
                                })?,
                                len,
                            )
                            .ok_or(IssueError::CounterOverflow {
                                counter: "memory-admission witness encoding",
                            })?,
                        );
                    }
                    out.push(request);
                }
                // A present write barrier is priced at the L2-hit arm (as in
                // L1Const); stores and reductions do not wait on responses.
                if write_barrier.is_some() {
                    p.store_visibility_ready(t).map(Some)
                } else {
                    Ok(None)
                }
            }
        }
    }

    fn preflight_response_accounting(
        &self,
        st: &SharedL2State,
        reply: ReplyTo,
        now: u64,
    ) -> Result<(), IssueError> {
        let fill = match reply {
            ReplyTo::Warp { warp, barrier, .. } => {
                if warp >= self.warps.len() {
                    return Err(IssueError::ResponseWarpOutOfRange {
                        warp,
                        warps: self.warps.len(),
                    });
                }
                self.warps[warp].preflight_pending_remove(warp, barrier, 1)?;
                return Ok(());
            }
            ReplyTo::AsyncCopy {
                warp,
                group,
                barrier,
                response_id,
            } => {
                if warp >= self.warps.len() {
                    return Err(IssueError::ResponseWarpOutOfRange {
                        warp,
                        warps: self.warps.len(),
                    });
                }
                self.async_copies[warp].preflight_async_copy_source_completion(
                    warp,
                    group,
                    Some(response_id.instruction),
                    1,
                    now,
                    &self.params,
                )?;
                if let Some(barrier) = barrier {
                    self.warps[warp].preflight_pending_remove(warp, barrier, 1)?;
                }
                return Ok(());
            }
            ReplyTo::L1Fill { fill, .. } => {
                if !st.l1.is_fill_in_flight(fill) {
                    return Err(IssueError::ResponseFillNotInFlight { fill });
                }
                if let FillKey::Line(line) = fill
                    && !st.fill_masks.contains_key(&line)
                {
                    return Err(IssueError::FillMaskMissing { line });
                }
                fill
            }
            ReplyTo::None => return Err(IssueError::UnexpectedMutationOnlyResponse),
        };

        let mut removals = HashMap::new();
        let mut async_removals = HashMap::new();
        if let Some(waiters) = st.fill_waiters.get(&fill) {
            waiters.try_for_each(|waiter| {
                accumulate_waiter_removals(&mut removals, &mut async_removals, waiter)
            })?;
        }

        let mut projected = HashMap::new();
        for (&(warp, barrier), &remove) in &removals {
            if warp >= self.warps.len() {
                return Err(IssueError::ResponseWarpOutOfRange {
                    warp,
                    warps: self.warps.len(),
                });
            }
            self.warps[warp].preflight_pending_remove(warp, barrier, remove)?;
            let current = self.warps[warp].barrier_pending[barrier.index()];
            projected.insert(
                (warp, barrier),
                current
                    .checked_sub(remove)
                    .ok_or(IssueError::PendingBarrierUnderflow {
                        warp,
                        barrier: barrier.wire(),
                        current,
                        remove,
                    })?,
            );
        }
        let mut projected_async = HashMap::new();
        for (&(warp, group), &remove) in &async_removals {
            project_async_copy_removal(
                &mut projected_async,
                &self.async_copies,
                warp,
                group,
                remove,
                now,
                &self.params,
            )?;
        }

        if st.retry.is_empty() {
            let budget = L1MutationBudget {
                recency_touches: 1,
                ..L1MutationBudget::for_origin(MemoryOrigin::Global)
            };
            if st.l1.preflight_mutation_budget(budget).is_err() {
                let mut shadow = st.l1.clone();
                shadow.complete_fill(fill)?;
            }
            return Ok(());
        }

        let mut budgets = MemoryOrigin::ALL.map(L1MutationBudget::for_origin);
        budgets[MemoryOrigin::Global.index()].recency_touches = 1;
        let mut pending_budget_fits = true;
        let mut has_async_retry = false;
        for retry in &st.retry {
            let budget = &mut budgets[retry.origin.index()];
            if retry.warp >= self.warps.len() {
                return Err(IssueError::ResponseWarpOutOfRange {
                    warp: retry.warp,
                    warps: self.warps.len(),
                });
            }
            has_async_retry |= retry.target.and_then(FillTarget::async_group).is_some();
            let remaining = retry
                .hi
                .checked_sub(retry.lo)
                .ok_or(IssueError::CounterUnderflow {
                    counter: "retry sector cursor",
                })?;
            if retry.hi > self.warps[retry.warp].sectors.len() {
                return Err(IssueError::CounterOverflow {
                    counter: "retry sector cursor",
                });
            }
            let remaining_u64 =
                u64::try_from(remaining).map_err(|_| IssueError::CounterOverflow {
                    counter: "retry L1 budget",
                })?;
            budget.eligible_sectors = checked_u64_counter_add(
                "retry L1 eligible-sector budget",
                budget.eligible_sectors,
                remaining_u64,
            )?;
            budget.recency_touches = checked_u64_counter_add(
                "retry L1 recency budget",
                budget.recency_touches,
                remaining_u64,
            )?;
            budget.new_fills =
                checked_u64_counter_add("retry L1 fill budget", budget.new_fills, remaining_u64)?;
            for sector_index in retry.lo..retry.hi {
                let sector_id = self.warps[retry.warp].sectors[sector_index];
                preflight_line_fill_mask(st, sector_id)?;
                let response_id = ResponseId::new(
                    st.response_owner,
                    retry.warp,
                    retry.instruction,
                    sector_index,
                );
                if let Some(&reply) = st
                    .outstanding_replies
                    .iter()
                    .find(|reply| reply.response_id() == Some(response_id))
                {
                    return Err(IssueError::ResponseTokenAlreadyOutstanding { reply });
                }
            }
            if let Some(barrier) = retry.target.and_then(FillTarget::barrier) {
                let base = self.warps[retry.warp].barrier_pending[barrier.index()];
                let current = projected.entry((retry.warp, barrier)).or_insert(base);
                let before_remove = *current;
                let after_remove =
                    before_remove
                        .checked_sub(1)
                        .ok_or(IssueError::PendingBarrierUnderflow {
                            warp: retry.warp,
                            barrier: barrier.wire(),
                            current: before_remove,
                            remove: 1,
                        })?;
                let additions =
                    u32::try_from(remaining).map_err(|_| IssueError::CounterOverflow {
                        counter: "retry pending-add budget",
                    })?;
                if let Some(after_add) = after_remove.checked_add(additions) {
                    *current = after_add;
                } else {
                    pending_budget_fits = false;
                    *current = after_remove;
                }
            }
        }

        if st.l1.preflight_mutation_budgets(&budgets).is_ok()
            && pending_budget_fits
            && !has_async_retry
        {
            return Ok(());
        }

        // Numeric-boundary slow path: complete and retry on a shadow cache so
        // warm hits and an atomic retry-unit replacement are charged exactly.
        let mut shadow = st.l1.clone();
        shadow.complete_fill(fill)?;
        let mut exact = HashMap::new();
        for (&(warp, barrier), &remove) in &removals {
            let current = self.warps[warp].barrier_pending[barrier.index()];
            exact.insert(
                (warp, barrier),
                current
                    .checked_sub(remove)
                    .ok_or(IssueError::PendingBarrierUnderflow {
                        warp,
                        barrier: barrier.wire(),
                        current,
                        remove,
                    })?,
            );
        }
        let mut exact_async = projected_async;
        'retries: for retry in &st.retry {
            let barrier = retry.target.and_then(FillTarget::barrier);
            if let Some(barrier) = barrier {
                let base = self.warps[retry.warp].barrier_pending[barrier.index()];
                let current = exact.entry((retry.warp, barrier)).or_insert(base);
                let before = *current;
                *current = before
                    .checked_sub(1)
                    .ok_or(IssueError::PendingBarrierUnderflow {
                        warp: retry.warp,
                        barrier: barrier.wire(),
                        current: before,
                        remove: 1,
                    })?;
            }
            for sector_index in retry.lo..retry.hi {
                let sector_id = self.warps[retry.warp].sectors[sector_index];
                match access_shared_l1(
                    &mut shadow,
                    retry.origin,
                    sector_id,
                    now,
                    retry.target.is_some(),
                    retry.mshr_floor,
                )? {
                    SharedL1Access::Hit { .. } => {
                        if let Some(group) = retry.target.and_then(FillTarget::async_group) {
                            project_async_copy_removal(
                                &mut exact_async,
                                &self.async_copies,
                                retry.warp,
                                group,
                                1,
                                now,
                                &self.params,
                            )?;
                        }
                    }
                    SharedL1Access::MissRequest { .. } | SharedL1Access::MergedPending { .. } => {
                        if let Some(barrier) = barrier {
                            let current = exact
                                .entry((retry.warp, barrier))
                                .or_insert(self.warps[retry.warp].barrier_pending[barrier.index()]);
                            let before = *current;
                            *current = before.checked_add(1).ok_or(
                                IssueError::PendingBarrierOverflow {
                                    warp: retry.warp,
                                    barrier: barrier.wire(),
                                    current: before,
                                    add: 1,
                                },
                            )?;
                        }
                    }
                    SharedL1Access::Stalled => {
                        if let Some(barrier) = barrier {
                            let current = exact
                                .entry((retry.warp, barrier))
                                .or_insert(self.warps[retry.warp].barrier_pending[barrier.index()]);
                            let before = *current;
                            *current = before.checked_add(1).ok_or(
                                IssueError::PendingBarrierOverflow {
                                    warp: retry.warp,
                                    barrier: barrier.wire(),
                                    current: before,
                                    add: 1,
                                },
                            )?;
                        }
                        break 'retries;
                    }
                }
            }
        }
        Ok(())
    }

    fn preflight_retry_timing(
        &self,
        st: &SharedL2State,
        fill: FillKey,
        now: u64,
    ) -> Result<(), IssueError> {
        if st.retry.is_empty() {
            return Ok(());
        }
        let observes_hit_ready = st.retry.iter().any(|retry| retry.target.is_some());
        let hit_timing_fits =
            !observes_hit_ready || now.checked_add(self.params.l1_hit_latency).is_some();
        let response_timing_fits = now.checked_add(self.params.l1_miss_latency).is_some();
        if hit_timing_fits && response_timing_fits {
            return Ok(());
        }

        // At the cycle-domain boundary, replay the retry queue on a shadow L1.
        // A hit needs a timestamp only when its retry carries a write barrier;
        // only a fresh miss emits a new self-affecting response. Merges and
        // stalls consume neither timing arm.
        let mut shadow = st.l1.clone();
        shadow.complete_fill(fill)?;
        'retries: for retry in &st.retry {
            for sector_index in retry.lo..retry.hi {
                let sector_id = self.warps[retry.warp].sectors[sector_index];
                match access_shared_l1(
                    &mut shadow,
                    retry.origin,
                    sector_id,
                    now,
                    false,
                    retry.mshr_floor,
                )? {
                    SharedL1Access::Hit { .. } => {
                        if retry.target.is_some() {
                            checked_cycle_add(
                                "retry L1 hit visibility",
                                now,
                                self.params.l1_hit_latency,
                            )?;
                        }
                    }
                    SharedL1Access::MissRequest { .. } => {
                        checked_cycle_add(
                            "retry self-response horizon",
                            now,
                            self.params.l1_miss_latency,
                        )?;
                    }
                    SharedL1Access::MergedPending { .. } => {}
                    SharedL1Access::Stalled => break 'retries,
                }
            }
        }
        Ok(())
    }

    pub(crate) fn preflight_response(&self, reply: ReplyTo, now: u64) -> Result<(), IssueError> {
        if now < self.t {
            return Err(IssueError::ResponseBeforeClock { now, clock: self.t });
        }
        let st = match &self.mem {
            MemModel::L1Shared(st) => st,
            MemModel::Flat => {
                return Err(IssueError::ResponseModeMismatch { mode: "flat" });
            }
            MemModel::L1Const(_) => {
                return Err(IssueError::ResponseModeMismatch {
                    mode: "constant-L1",
                });
            }
        };
        match reply {
            ReplyTo::Warp { warp, .. } | ReplyTo::AsyncCopy { warp, .. }
                if warp >= self.warps.len() =>
            {
                return Err(IssueError::ResponseWarpOutOfRange {
                    warp,
                    warps: self.warps.len(),
                });
            }
            ReplyTo::None => return Err(IssueError::UnexpectedMutationOnlyResponse),
            _ => {}
        }
        if !st.outstanding_replies.contains(&reply) {
            return Err(IssueError::ResponseTokenNotOutstanding { reply });
        }
        if let ReplyTo::Warp { warp, barrier, .. } = reply
            && self.warps[warp].barrier_pending[barrier.index()] == 0
        {
            return Err(IssueError::ResponseBarrierNotPending {
                warp,
                barrier: barrier.wire(),
            });
        }
        self.preflight_response_accounting(st, reply, now)?;
        // A fill completion can drain parked accesses. Validate only the timing
        // branch each exact retry outcome consumes; next-level L2/DRAM pricing
        // is preflighted by the system against the actual L2 outcome.
        if let ReplyTo::L1Fill { fill, .. } = reply {
            self.preflight_retry_timing(st, fill, now)?;
        }
        Ok(())
    }

    /// Apply one memory response at its event time, then drain what it
    /// unblocked (fill waiters, MSHR retry queue — retries may emit new
    /// requests into `out`).
    pub(crate) fn on_response(
        &mut self,
        reply: ReplyTo,
        now: u64,
        out: &mut Vec<MemRequest>,
    ) -> Result<(), IssueError> {
        self.preflight_response(reply, now)?;
        let params = self.params;
        let st = match &mut self.mem {
            MemModel::L1Shared(st) => st,
            MemModel::Flat => {
                return Err(IssueError::ResponseModeMismatch { mode: "flat" });
            }
            MemModel::L1Const(_) => {
                return Err(IssueError::ResponseModeMismatch {
                    mode: "constant-L1",
                });
            }
        };
        if !st.outstanding_replies.remove(&reply) {
            return Err(IssueError::ResponseTokenRemovalFailed { reply });
        }
        // New emission batch (the retry drain below may emit).
        st.line_folds.clear();
        let warps = &mut self.warps;
        let async_copies = &mut self.async_copies;
        match reply {
            ReplyTo::L1Fill { fill, .. } => {
                if let FillKey::Line(line) = fill
                    && st.fill_masks.remove(&line).is_none()
                {
                    return Err(IssueError::FillMaskMissing { line });
                }
                st.l1.complete_fill(fill)?;
                if let Some(waiters) = st.fill_waiters.remove(&fill) {
                    waiters.try_for_each(|waiter| {
                        if let Some(group) = waiter.target.async_group() {
                            async_copies[waiter.warp].complete_async_copy_sources(
                                waiter.warp,
                                group,
                                None,
                                waiter.units,
                                now,
                                &params,
                            )?;
                        }
                        if let Some(barrier) = waiter.target.barrier() {
                            warps[waiter.warp].clear_pending(
                                waiter.warp,
                                barrier,
                                waiter.units,
                                now,
                            )?;
                        }
                        Ok(())
                    })?;
                }
                // The completed fill freed an MSHR entry: retry parked
                // accesses (FIFO) until one stalls again.
                drain_retries(st, warps, async_copies, now, &params, out)?;
            }
            ReplyTo::Warp { warp, barrier, .. } => {
                warps[warp].clear_pending(warp, barrier, 1, now)?;
            }
            ReplyTo::AsyncCopy {
                warp,
                group,
                barrier,
                response_id,
            } => {
                async_copies[warp].complete_async_copy_sources(
                    warp,
                    group,
                    Some(response_id.instruction),
                    1,
                    now,
                    &params,
                )?;
                if let Some(barrier) = barrier {
                    warps[warp].clear_pending(warp, barrier, 1, now)?;
                }
            }
            ReplyTo::None => return Err(IssueError::UnexpectedMutationOnlyResponse),
        }
        Ok(())
    }

    fn project_selected_readiness(
        &self,
        subcore: usize,
        warp_index: usize,
        now: u64,
        projected: &mut ReadinessDiagnostics,
    ) -> Result<(), IssueError> {
        let warp = &self.warps[warp_index];
        let slot = warp
            .next_slot
            .ok_or(IssueError::MissingIssueSlot { warp: warp_index })?;
        let kind = warp.insts[warp.next].kind;
        if let Some((class, sectors, wavefronts, unique_lines, atomic_l2_requests)) =
            projected_global_memory_issue(kind, warp.sectors)
        {
            projected
                .global_memory_issue
                .observe(
                    class,
                    sectors,
                    wavefronts,
                    unique_lines,
                    atomic_l2_requests,
                    slot.control.read_barrier.is_some(),
                    now,
                )
                .map_err(|source| match source {
                    ReadinessCounterError::Overflow { field, .. } => {
                        IssueError::CounterOverflow { counter: field }
                    }
                    mismatch @ (ReadinessCounterError::MemoryServiceCandidateMismatch {
                        ..
                    }
                    | ReadinessCounterError::MemoryAdmissionCandidateMismatch {
                        ..
                    }) => IssueError::ReadinessDiagnostic(mismatch),
                })?;
        }
        let base = warp.current_since;
        let mut ready_at = base;
        let mut limiting = None;
        let mut tied = 0;

        if let Some(feedback) = self.memory_admission_feedback.as_ref()
            && feedback.matches_kind(kind)
        {
            let until = feedback
                .admission_ready_at(subcore, kind)
                .expect("selected feedback instruction has a priced queue frontier");
            observe_readiness_constraint(
                projected,
                base,
                until,
                ReadinessCause::MemoryAdmissionQueue,
                &mut ready_at,
                &mut limiting,
                &mut tied,
            )?;
        }
        if let Some(feedback) = self.memory_admission_feedback.as_ref()
            && feedback.candidate.membar_waits_for_service_completion
            && is_gpu_membar_issue(slot)
        {
            let until = feedback
                .completion_ready_at(warp_index, slot)
                .expect("selected MEMBAR has a priced write-completion frontier");
            observe_readiness_constraint(
                projected,
                base,
                until,
                ReadinessCause::MemoryWriteCompletion,
                &mut ready_at,
                &mut limiting,
                &mut tied,
            )?;
        }

        observe_readiness_constraint(
            projected,
            base,
            warp.eligible,
            ReadinessCause::ControlStall,
            &mut ready_at,
            &mut limiting,
            &mut tied,
        )?;
        observe_readiness_constraint(
            projected,
            base,
            warp.cta_barrier_ready,
            ReadinessCause::CtaBarrierRendezvous,
            &mut ready_at,
            &mut limiting,
            &mut tied,
        )?;
        observe_readiness_constraint(
            projected,
            base,
            warp.barrier_pipeline_ready,
            ReadinessCause::PipelineBarrier,
            &mut ready_at,
            &mut limiting,
            &mut tied,
        )?;
        observe_readiness_constraint(
            projected,
            base,
            warp.branch_pipeline_ready,
            ReadinessCause::PipelineBranch,
            &mut ready_at,
            &mut limiting,
            &mut tied,
        )?;

        for barrier in 0..DEP_BARRIERS {
            if !slot.control.wait_mask.contains_index(barrier) {
                continue;
            }
            debug_assert_eq!(
                warp.barrier_pending[barrier], 0,
                "a selected warp cannot have an unresolved dependence counter"
            );
            observe_readiness_constraint(
                projected,
                base,
                warp.barrier_clear[barrier],
                warp.barrier_origin[barrier].into(),
                &mut ready_at,
                &mut limiting,
                &mut tied,
            )?;
        }
        if let Some(barrier) = warp.ordinary_dependency_wait
            && !slot.control.wait_mask.contains(barrier)
        {
            let barrier = barrier.index();
            debug_assert_eq!(
                warp.barrier_pending[barrier], 0,
                "a selected explicit dependency-barrier wait cannot have an unresolved counter"
            );
            observe_readiness_constraint(
                projected,
                base,
                warp.barrier_clear[barrier],
                warp.barrier_origin[barrier].into(),
                &mut ready_at,
                &mut limiting,
                &mut tied,
            )?;
        }

        if self.has_async_copy_protocol && slot.op_class == OpClass::AsyncCopyCommit {
            let commits_nonempty = self.async_copies[warp_index]
                .state
                .as_deref()
                .and_then(|state| state.open_group)
                .is_some();
            if commits_nonempty {
                projected.async_groups.groups_committed = checked_u64_counter_add(
                    "committed asynchronous-copy group diagnostic",
                    projected.async_groups.groups_committed,
                    1,
                )?;
            }
        }

        if self.has_async_copy_protocol && slot.op_class == OpClass::AsyncCopyWait {
            let prefix = self.async_copies[warp_index]
                .async_wait_prefix_len(slot.async_copy_wait_max_pending_groups);
            if let Some(state) = self.async_copies[warp_index].state.as_deref() {
                for group in state.committed_groups.iter().take(prefix) {
                    debug_assert!(
                        group.is_complete_at(now),
                        "a selected async wait requires a complete committed prefix"
                    );
                    let local_completion = group.local_completion_time(&self.params)?;
                    let cause = group.limiting_readiness_cause(&self.params)?;
                    observe_readiness_constraint(
                        projected,
                        base,
                        group.completion_time,
                        cause,
                        &mut ready_at,
                        &mut limiting,
                        &mut tied,
                    )?;

                    projected.async_groups.groups_waited = checked_u64_counter_add(
                        "waited asynchronous-copy group diagnostic",
                        projected.async_groups.groups_waited,
                        1,
                    )?;
                    let source_cycles = group
                        .source_completion_time
                        .saturating_sub(group.committed_at);
                    projected.async_groups.source_cycles = checked_u64_counter_add(
                        "asynchronous-copy committed-to-source diagnostic",
                        projected.async_groups.source_cycles,
                        source_cycles,
                    )?;
                    let local_cycles =
                        local_completion.saturating_sub(group.source_completion_time);
                    projected.async_groups.destination_local_cycles = checked_u64_counter_add(
                        "asynchronous-copy local destination diagnostic",
                        projected.async_groups.destination_local_cycles,
                        local_cycles,
                    )?;
                    let service_cycles = group.completion_time.saturating_sub(local_completion);
                    projected.async_groups.destination_sm_service_cycles = checked_u64_counter_add(
                        "asynchronous-copy SM destination-service diagnostic",
                        projected.async_groups.destination_sm_service_cycles,
                        service_cycles,
                    )?;
                    let hidden = group
                        .completion_time
                        .min(base)
                        .saturating_sub(group.committed_at);
                    projected.async_groups.completion_hidden_before_wait_cycles =
                        checked_u64_counter_add(
                            "asynchronous-copy hidden-before-wait diagnostic",
                            projected.async_groups.completion_hidden_before_wait_cycles,
                            hidden,
                        )?;
                    if let Some(observed) = self.async_copies[warp_index].bypass_copy_projection(
                        warp_index,
                        group,
                        &self.params,
                        base,
                    )? {
                        let diagnostics = projected
                            .bypass_async_copies
                            .get_or_insert_with(|| Box::new(BypassAsyncCopyDiagnostics::default()));
                        diagnostics.add_checked(&BypassAsyncCopyDiagnostics {
                            copies_observed: observed.copies,
                            destination_wavefronts_observed: observed.bank_wavefront_upper_bound,
                            bypass_wavefronts_observed: observed.bypass_wavefront_upper_bound,
                            source_cohorts_observed: observed.source_cohorts,
                            destination_lines_observed: observed.destination_lines,
                            max_bank_wavefronts_sum: observed.max_bank_wavefronts_sum,
                            max_bypass_wavefronts_sum: observed.max_bypass_wavefronts_sum,
                            copy_source_spread_cycles: observed.source_spread_cycles,
                            per_copy_completion_cycles: observed.per_copy_completion_cycles,
                            group_aggregate_excess_cycles: observed.group_aggregate_excess_cycles,
                            topology_candidate_penalty_cycles: observed
                                .topology_candidate_penalty_cycles,
                            topology_candidate_completion_cycles: observed
                                .topology_candidate_completion_cycles,
                            topology_candidate_later_cycles: observed
                                .topology_candidate_later_cycles,
                            topology_candidate_earlier_cycles: observed
                                .topology_candidate_earlier_cycles,
                            pipeline_groups_observed: 1,
                            pipeline_q1_completion_cycles: observed.pipeline_q1_completion_cycles,
                            pipeline_q1_exposure_cycles: observed.pipeline_q1_exposure_cycles,
                            pipeline_q2_completion_cycles: observed.pipeline_q2_completion_cycles,
                            pipeline_q2_exposure_cycles: observed.pipeline_q2_exposure_cycles,
                            pipeline_q4_completion_cycles: observed.pipeline_q4_completion_cycles,
                            pipeline_q4_exposure_cycles: observed.pipeline_q4_exposure_cycles,
                            pipeline_q8_completion_cycles: observed.pipeline_q8_completion_cycles,
                            pipeline_q8_exposure_cycles: observed.pipeline_q8_exposure_cycles,
                            pipeline_q16_completion_cycles: observed.pipeline_q16_completion_cycles,
                            pipeline_q16_exposure_cycles: observed.pipeline_q16_exposure_cycles,
                        })?;
                    }
                    let limited_counter = match cause {
                        ReadinessCause::AsyncSource => {
                            &mut projected.async_groups.source_limited_groups
                        }
                        ReadinessCause::AsyncDestinationLocal => {
                            &mut projected.async_groups.destination_local_limited_groups
                        }
                        ReadinessCause::AsyncDestinationSmService => {
                            &mut projected.async_groups.destination_sm_service_limited_groups
                        }
                        _ => unreachable!("async-group classifier returns an async cause"),
                    };
                    *limited_counter = checked_u64_counter_add(
                        "asynchronous-copy limiting-arm diagnostic",
                        *limited_counter,
                        1,
                    )?;
                }
            }
        }

        if let PipelineClass::Branch {
            next_issue_interval_cycles,
        } = slot.pipeline_class
        {
            projected.control_pipeline.branch_instructions_issued = checked_u64_counter_add(
                "branch-pipeline issued-instruction diagnostic",
                projected.control_pipeline.branch_instructions_issued,
                1,
            )?;
            projected.control_pipeline.branch_resolution_cycles = checked_u64_counter_add(
                "branch-pipeline resolution-cycle diagnostic",
                projected.control_pipeline.branch_resolution_cycles,
                next_issue_interval_cycles,
            )?;
        }

        if let PipelineClass::Barrier {
            warp_initiation_interval_cycles,
            ..
        } = slot.pipeline_class
        {
            projected.barrier_pipeline.instructions_issued = checked_u64_counter_add(
                "CTA-barrier pipeline issued-instruction diagnostic",
                projected.barrier_pipeline.instructions_issued,
                1,
            )?;
            projected.barrier_pipeline.issue_server_busy_cycles = checked_u64_counter_add(
                "CTA-barrier pipeline busy-cycle diagnostic",
                projected.barrier_pipeline.issue_server_busy_cycles,
                warp_initiation_interval_cycles,
            )?;
        }

        if matches!(slot.pipeline_class, PipelineClass::Tensor { .. }) {
            observe_readiness_constraint(
                projected,
                base,
                self.tensor_pipe_ready[subcore],
                ReadinessCause::PipelineTensor,
                &mut ready_at,
                &mut limiting,
                &mut tied,
            )?;
        }

        let effective_ready = ready_at.max(base);
        if effective_ready > base {
            projected.limiting_cycles.add_cause(
                limiting.expect("a post-baseline ready frontier has a cause"),
                effective_ready - base,
            )?;
        }
        if tied != 0 {
            projected.tied_limiting_causes = checked_u64_counter_add(
                "tied limiting readiness cause diagnostic",
                projected.tied_limiting_causes,
                tied,
            )?;
        }
        if now > effective_ready {
            let delayed = now - effective_ready;
            projected
                .limiting_cycles
                .add_eligible_not_selected(delayed)?;
            projected
                .constraint_exposure_cycles
                .add_eligible_not_selected(delayed)?;
            projected.delayed_ready_issues = checked_u64_counter_add(
                "delayed ready issue diagnostic",
                projected.delayed_ready_issues,
                1,
            )?;
        }
        Ok(())
    }

    fn project_cta_barrier_diagnostics(
        &self,
        picks: &[Option<usize>; SUB_CORES],
        barrier_arrival_times: &[Option<u64>; SUB_CORES],
        now: u64,
        projected: &mut ReadinessDiagnostics,
    ) -> Result<(), IssueError> {
        #[derive(Clone, Copy)]
        struct BlockDelta {
            block: usize,
            retirements: usize,
            arrivals: usize,
            first_new_arrival: Option<u64>,
            last_new_arrival: Option<u64>,
        }
        let mut deltas: [Option<BlockDelta>; SUB_CORES] = [None; SUB_CORES];
        for (subcore, pick) in picks.iter().enumerate() {
            let Some(warp_index) = *pick else { continue };
            let warp = &self.warps[warp_index];
            let retires = warp.next_issue_retires()?;
            let slot = warp
                .next_slot
                .ok_or(IssueError::MissingIssueSlot { warp: warp_index })?;
            let arrives = !retires && slot.op_class == OpClass::BlockBarrier;
            if !retires && !arrives {
                continue;
            }
            let entry_index = deltas
                .iter()
                .position(|entry| entry.is_some_and(|delta| delta.block == warp.block))
                .or_else(|| deltas.iter().position(Option::is_none))
                .ok_or(IssueError::CounterOverflow {
                    counter: "CTA barrier diagnostic block delta",
                })?;
            let entry = deltas[entry_index].get_or_insert(BlockDelta {
                block: warp.block,
                retirements: 0,
                arrivals: 0,
                first_new_arrival: None,
                last_new_arrival: None,
            });
            if retires {
                entry.retirements = checked_usize_counter_add(
                    "CTA barrier diagnostic retirement",
                    entry.retirements,
                    1,
                )?;
            }
            if arrives {
                let arrival = barrier_arrival_times[subcore].unwrap_or(now);
                entry.arrivals =
                    checked_usize_counter_add("CTA barrier diagnostic arrival", entry.arrivals, 1)?;
                entry.first_new_arrival = Some(
                    entry
                        .first_new_arrival
                        .map_or(arrival, |current| current.min(arrival)),
                );
                entry.last_new_arrival = Some(
                    entry
                        .last_new_arrival
                        .map_or(arrival, |current| current.max(arrival)),
                );
            }
        }

        for delta in deltas.into_iter().flatten() {
            let state = &self.block_states[delta.block];
            let unfinished = state.unfinished.checked_sub(delta.retirements).ok_or(
                IssueError::CounterUnderflow {
                    counter: "CTA barrier diagnostic unfinished warp",
                },
            )?;
            let waiting = checked_usize_counter_add(
                "CTA barrier diagnostic waiting warp",
                state.waiting,
                delta.arrivals,
            )?;
            if waiting == 0 || waiting != unfinished {
                continue;
            }

            state
                .barrier_generation
                .checked_add(1)
                .ok_or(IssueError::CounterOverflow {
                    counter: "CTA barrier generation",
                })?;
            let first_issue = state.barrier_first_dispatch.unwrap_or(now);
            let mut last_issue = if delta.arrivals == 0 {
                first_issue
            } else {
                now
            };
            let first_arrival = match (state.barrier_first_arrival, delta.first_new_arrival) {
                (Some(existing), Some(new)) => existing.min(new),
                (Some(existing), None) => existing,
                (None, Some(new)) => new,
                (None, None) => now,
            };
            let mut last_arrival = delta.last_new_arrival.unwrap_or(first_arrival);
            let first = state.first_warp;
            let end = checked_usize_counter_add(
                "CTA barrier diagnostic warp range",
                first,
                state.admitted_warps,
            )?;
            for warp in &self.warps[first..end] {
                if !warp.at_block_barrier {
                    continue;
                }
                let dispatch = warp
                    .block_barrier_dispatch
                    .ok_or(IssueError::CounterUnderflow {
                        counter: "CTA barrier dispatch timestamp",
                    })?;
                let arrival = warp
                    .block_barrier_arrival
                    .ok_or(IssueError::CounterUnderflow {
                        counter: "CTA barrier arrival timestamp",
                    })?;
                last_issue = last_issue.max(dispatch);
                last_arrival = last_arrival.max(arrival);
            }
            let release_at = checked_cycle_add("CTA barrier diagnostic release", last_arrival, 1)?;
            let mut rendezvous = 0u64;
            for warp in &self.warps[first..end] {
                if !warp.at_block_barrier {
                    continue;
                }
                let arrival = warp
                    .block_barrier_arrival
                    .ok_or(IssueError::CounterUnderflow {
                        counter: "CTA barrier arrival timestamp",
                    })?;
                rendezvous = checked_u64_counter_add(
                    "CTA barrier rendezvous warp cycle diagnostic",
                    rendezvous,
                    release_at
                        .checked_sub(arrival)
                        .ok_or(IssueError::CounterUnderflow {
                            counter: "CTA barrier rendezvous interval",
                        })?,
                )?;
            }
            for (subcore, pick) in picks.iter().enumerate() {
                let Some(warp) = *pick else { continue };
                if self.warps[warp].block != delta.block
                    || self.warps[warp].next_issue_retires()?
                    || self.warps[warp]
                        .next_slot
                        .is_none_or(|slot| slot.op_class != OpClass::BlockBarrier)
                {
                    continue;
                }
                let arrival = barrier_arrival_times[subcore].unwrap_or(now);
                rendezvous = checked_u64_counter_add(
                    "CTA barrier rendezvous warp cycle diagnostic",
                    rendezvous,
                    release_at
                        .checked_sub(arrival)
                        .ok_or(IssueError::CounterUnderflow {
                            counter: "CTA barrier new-arrival interval",
                        })?,
                )?;
            }
            let spread =
                last_arrival
                    .checked_sub(first_arrival)
                    .ok_or(IssueError::CounterUnderflow {
                        counter: "CTA barrier arrival spread",
                    })?;
            let issue_span =
                last_issue
                    .checked_sub(first_issue)
                    .ok_or(IssueError::CounterUnderflow {
                        counter: "CTA barrier issue span",
                    })?;

            projected.cta_barriers.generations = checked_u64_counter_add(
                "CTA barrier generation diagnostic",
                projected.cta_barriers.generations,
                1,
            )?;
            projected.cta_barriers.total_arrival_spread_cycles = checked_u64_counter_add(
                "CTA barrier total arrival-spread diagnostic",
                projected.cta_barriers.total_arrival_spread_cycles,
                spread,
            )?;
            projected.cta_barriers.max_arrival_spread_cycles =
                projected.cta_barriers.max_arrival_spread_cycles.max(spread);
            projected.cta_barriers.total_rendezvous_warp_cycles = checked_u64_counter_add(
                "CTA barrier total rendezvous diagnostic",
                projected.cta_barriers.total_rendezvous_warp_cycles,
                rendezvous,
            )?;
            if delta.retirements != 0 {
                projected.cta_barriers.releases_after_participant_retirement =
                    checked_u64_counter_add(
                        "CTA barrier participant-retirement diagnostic",
                        projected.cta_barriers.releases_after_participant_retirement,
                        1,
                    )?;
            }
            projected.barrier_pipeline.generations_completed = checked_u64_counter_add(
                "CTA-barrier pipeline completed-generation diagnostic",
                projected.barrier_pipeline.generations_completed,
                1,
            )?;
            if self
                .barrier_issue
                .is_some_and(|issue| issue.generation_rearm_cycles != 0)
            {
                projected.barrier_pipeline.generation_rearm_events = checked_u64_counter_add(
                    "CTA-barrier pipeline generation-rearm diagnostic",
                    projected.barrier_pipeline.generation_rearm_events,
                    1,
                )?;
            }
            projected.barrier_pipeline.first_issue_timestamp_sum = checked_u64_counter_add(
                "CTA-barrier pipeline first-issue timestamp diagnostic",
                projected.barrier_pipeline.first_issue_timestamp_sum,
                first_issue,
            )?;
            projected.barrier_pipeline.last_issue_timestamp_sum = checked_u64_counter_add(
                "CTA-barrier pipeline last-issue timestamp diagnostic",
                projected.barrier_pipeline.last_issue_timestamp_sum,
                last_issue,
            )?;
            projected
                .barrier_pipeline
                .total_generation_issue_span_cycles = checked_u64_counter_add(
                "CTA-barrier pipeline generation issue-span diagnostic",
                projected
                    .barrier_pipeline
                    .total_generation_issue_span_cycles,
                issue_span,
            )?;
            projected.barrier_pipeline.max_generation_issue_span_cycles = projected
                .barrier_pipeline
                .max_generation_issue_span_cycles
                .max(issue_span);
            projected.barrier_pipeline.first_arrival_timestamp_sum = checked_u64_counter_add(
                "CTA-barrier pipeline first-arrival timestamp diagnostic",
                projected.barrier_pipeline.first_arrival_timestamp_sum,
                first_arrival,
            )?;
            projected.barrier_pipeline.last_arrival_timestamp_sum = checked_u64_counter_add(
                "CTA-barrier pipeline last-arrival timestamp diagnostic",
                projected.barrier_pipeline.last_arrival_timestamp_sum,
                last_arrival,
            )?;
            projected
                .barrier_pipeline
                .total_generation_arrival_span_cycles = checked_u64_counter_add(
                "CTA-barrier pipeline generation arrival-span diagnostic",
                projected
                    .barrier_pipeline
                    .total_generation_arrival_span_cycles,
                spread,
            )?;
            projected
                .barrier_pipeline
                .max_generation_arrival_span_cycles = projected
                .barrier_pipeline
                .max_generation_arrival_span_cycles
                .max(spread);
        }
        Ok(())
    }

    fn project_picked_readiness(
        &self,
        picks: &[Option<usize>; SUB_CORES],
        barrier_arrival_times: &[Option<u64>; SUB_CORES],
        now: u64,
    ) -> Result<ReadinessDiagnostics, IssueError> {
        let mut projected = self.readiness.clone();
        for (subcore, pick) in picks.iter().enumerate() {
            if let Some(warp) = *pick {
                self.project_selected_readiness(subcore, warp, now, &mut projected)?;
                let state = &self.warps[warp];
                let slot = state
                    .next_slot
                    .ok_or(IssueError::MissingIssueSlot { warp })?;
                if is_clocklo_issue(slot) {
                    projected.clock_issue.reads_issued = checked_u64_counter_add(
                        "CLOCKLO reads-issued diagnostic",
                        projected.clock_issue.reads_issued,
                        1,
                    )?;
                    state
                        .clock_reads_issued
                        .checked_add(1)
                        .ok_or(IssueError::CounterOverflow {
                            counter: "warp CLOCKLO issue",
                        })?;
                    if let Some(first) = state.clock_first_issue {
                        let interval =
                            now.checked_sub(first).ok_or(IssueError::CounterUnderflow {
                                counter: "warp CLOCKLO issue interval",
                            })?;
                        if state.clock_reads_issued == 1 {
                            projected.clock_issue.warps_with_interval = checked_u64_counter_add(
                                "CLOCKLO warps-with-interval diagnostic",
                                projected.clock_issue.warps_with_interval,
                                1,
                            )?;
                        }
                        projected.clock_issue.maximum_first_to_last_issue_cycles = projected
                            .clock_issue
                            .maximum_first_to_last_issue_cycles
                            .max(interval);
                    }
                    if self
                        .memory_admission_feedback
                        .as_ref()
                        .is_some_and(|feedback| {
                            feedback.candidate.membar_waits_for_service_completion
                        })
                    {
                        match state.clock_reads_issued {
                            1 => {
                                let first = state.clock_first_issue.ok_or(
                                    IssueError::CounterUnderflow {
                                        counter: "first CLOCKLO phase timestamp",
                                    },
                                )?;
                                let interval =
                                    now.checked_sub(first).ok_or(IssueError::CounterUnderflow {
                                        counter: "first-to-second CLOCKLO issue interval",
                                    })?;
                                projected.clock_issue.maximum_first_to_second_issue_cycles =
                                    projected
                                        .clock_issue
                                        .maximum_first_to_second_issue_cycles
                                        .max(interval);
                            }
                            2 => {
                                let second = self
                                    .memory_admission_feedback
                                    .as_ref()
                                    .and_then(|feedback| feedback.second_clock_issue(warp))
                                    .ok_or(IssueError::CounterUnderflow {
                                        counter: "second CLOCKLO phase timestamp",
                                    })?;
                                let interval = now.checked_sub(second).ok_or(
                                    IssueError::CounterUnderflow {
                                        counter: "second-to-third CLOCKLO issue interval",
                                    },
                                )?;
                                projected.clock_issue.warps_with_three_reads =
                                    checked_u64_counter_add(
                                        "CLOCKLO warps-with-three-reads diagnostic",
                                        projected.clock_issue.warps_with_three_reads,
                                        1,
                                    )?;
                                projected.clock_issue.maximum_second_to_third_issue_cycles =
                                    projected
                                        .clock_issue
                                        .maximum_second_to_third_issue_cycles
                                        .max(interval);
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        self.project_cta_barrier_diagnostics(picks, barrier_arrival_times, now, &mut projected)?;
        Ok(projected)
    }

    fn project_memory_service_candidates(
        &self,
        picks: &[Option<usize>; SUB_CORES],
        now: u64,
    ) -> Result<MemoryServiceCandidateState, IssueError> {
        if self.memory_service_candidates.is_empty() {
            return Ok(self.memory_service_candidates.clone());
        }
        let mut projected = self.memory_service_candidates.clone();
        for (subcore, pick) in picks.iter().enumerate() {
            let Some(warp_index) = *pick else { continue };
            let warp = &self.warps[warp_index];
            if let Some((class, _, ideal_wavefronts, unique_lines, _)) =
                projected_global_memory_issue(warp.insts[warp.next].kind, warp.sectors)
            {
                projected.observe(subcore, class, ideal_wavefronts, unique_lines, now)?;
            }
        }
        Ok(projected)
    }

    /// Earliest issue time after both warp/control dependencies and the next
    /// instruction's characterized execution resource are available. A missing
    /// slot remains selectable so the established typed invariant error is
    /// preserved at the transaction boundary.
    fn issue_ready_at<const CHECK_ASYNC: bool>(&self, subcore: usize, warp: usize) -> Option<u64> {
        let mut warp_ready = self.warps[warp].ready_at()?;
        if let Some(feedback) = self.memory_admission_feedback.as_ref() {
            let state = &self.warps[warp];
            let admission_ready =
                feedback.admission_ready_at(subcore, state.insts[state.next].kind)?;
            warp_ready = warp_ready.max(admission_ready);
            if let Some(slot) = state.next_slot {
                let completion_ready = feedback.completion_ready_at(warp, slot)?;
                warp_ready = warp_ready.max(completion_ready);
            }
        }
        if CHECK_ASYNC
            && let Some(IssueSlot {
                op_class: OpClass::AsyncCopyWait,
                async_copy_wait_max_pending_groups,
                ..
            }) = self.warps[warp].next_slot
        {
            let async_ready = self.async_copies[warp]
                .async_wait_clear_time(*async_copy_wait_max_pending_groups)?;
            warp_ready = warp_ready.max(async_ready);
        }
        let pipeline_ready = match self.warps[warp].next_slot.map(|slot| slot.pipeline_class) {
            Some(PipelineClass::Branch { .. }) => 0,
            Some(PipelineClass::Barrier { .. }) => {
                self.block_states[self.warps[warp].block].barrier_rearm_ready
            }
            Some(PipelineClass::Tensor { .. }) => self.tensor_pipe_ready[subcore],
            Some(PipelineClass::DispatchOnly) | None => 0,
        };
        Some(warp_ready.max(pipeline_ready))
    }

    fn is_issue_ready<const CHECK_ASYNC: bool>(
        &self,
        subcore: usize,
        warp: usize,
        now: u64,
    ) -> bool {
        self.issue_ready_at::<CHECK_ASYNC>(subcore, warp)
            .is_some_and(|ready| ready <= now)
    }

    fn pick_warps<const CHECK_ASYNC: bool>(&self, now: u64) -> [Option<usize>; SUB_CORES] {
        std::array::from_fn(|subcore| {
            // CGGTY pick: held warp while its control state and execution
            // resource are ready, else youngest fully-ready warp.
            match self.hold[subcore] {
                Some(w) if self.is_issue_ready::<CHECK_ASYNC>(subcore, w, now) => Some(w),
                _ => self.subcore_warps[subcore]
                    .iter()
                    .copied()
                    .filter(|&w| self.is_issue_ready::<CHECK_ASYNC>(subcore, w, now))
                    .max_by_key(|&w| self.warps[w].arrival),
            }
        })
    }

    /// Pick in stable SMSP order while reserving the transaction's shared
    /// admission capacity. A candidate rejected by an earlier SMSP's claim is
    /// not a dispatch bubble: the same SMSP may still issue its held/youngest
    /// ready non-candidate (or another candidate whose queue has room).
    fn pick_warps_with_memory_admission<const CHECK_ASYNC: bool>(
        &self,
        now: u64,
    ) -> [Option<usize>; SUB_CORES] {
        let Some(feedback) = self.memory_admission_feedback.as_ref() else {
            return self.pick_warps::<CHECK_ASYNC>(now);
        };
        let mut additions = [0usize; SUB_CORES];
        let mut picks = [None; SUB_CORES];
        for (subcore, pick) in picks.iter_mut().enumerate() {
            let permitted = |warp: usize| {
                if !self.is_issue_ready::<CHECK_ASYNC>(subcore, warp, now) {
                    return false;
                }
                let state = &self.warps[warp];
                feedback.permits_transaction_addition(
                    subcore,
                    state.insts[state.next].kind,
                    &additions,
                    now,
                )
            };
            let selected = match self.hold[subcore] {
                Some(warp) if permitted(warp) => Some(warp),
                _ => self.subcore_warps[subcore]
                    .iter()
                    .copied()
                    .filter(|&warp| permitted(warp))
                    .max_by_key(|&warp| self.warps[warp].arrival),
            };
            if let Some(warp) = selected {
                let state = &self.warps[warp];
                feedback.record_transaction_addition(
                    subcore,
                    state.insts[state.next].kind,
                    &mut additions,
                );
            }
            *pick = selected;
        }
        picks
    }

    /// Assign completion timestamps to barrier arrivals dispatched in this
    /// cycle. SMSP dispatch remains parallel; the one SM-shared service queues
    /// those arrivals in stable subcore order at its measured initiation rate.
    fn project_barrier_arrival_times(
        &self,
        picks: &[Option<usize>; SUB_CORES],
        now: u64,
    ) -> Result<[Option<u64>; SUB_CORES], IssueError> {
        let mut server_ready = self.barrier_pipe_ready;
        let mut arrivals = [None; SUB_CORES];
        for (subcore, pick) in picks.iter().enumerate() {
            let Some(warp) = *pick else { continue };
            let slot = self.warps[warp]
                .next_slot
                .ok_or(IssueError::MissingIssueSlot { warp })?;
            let PipelineClass::Barrier {
                warp_initiation_interval_cycles,
                ..
            } = slot.pipeline_class
            else {
                continue;
            };
            if self.warps[warp].next_issue_retires()? {
                continue;
            }
            let arrival = checked_cycle_add(
                "CTA-barrier arrival service",
                server_ready.max(now),
                warp_initiation_interval_cycles,
            )?;
            arrivals[subcore] = Some(arrival);
            server_ready = arrival;
        }
        Ok(arrivals)
    }

    fn next_known_issue_time<const CHECK_ASYNC: bool>(&self) -> Option<u64> {
        let mut next_t: Option<u64> = None;
        for (subcore, warps) in self.subcore_warps.iter().enumerate() {
            for &warp in warps {
                if let Some(ready) = self.issue_ready_at::<CHECK_ASYNC>(subcore, warp) {
                    next_t = Some(next_t.map_or(ready, |current| current.min(ready)));
                }
            }
        }
        next_t
    }

    /// Earliest timestamp at which a committed exact-destination group becomes
    /// eligible to enter the aggregate service. Groups with outstanding source
    /// responses have no locally known wake time and remain event-driven.
    fn next_async_copy_destination_source_time(&self) -> Option<u64> {
        self.async_copies
            .iter()
            .filter_map(|slot| slot.state.as_deref())
            .flat_map(|state| state.committed_groups.iter())
            .filter(|group| {
                group.pending_sources == 0
                    && group.destination_wavefronts != 0
                    && !group.destination_service_queued
                    && group.source_completion_time > self.t
            })
            .map(|group| group.source_completion_time)
            .min()
    }

    /// Assign committed, source-complete cache-all groups to the shared
    /// destination service. Projection and timestamp arithmetic finish before
    /// either the pipe or a group is mutated.
    fn enqueue_ready_async_copy_destinations(&mut self) -> Result<(), IssueError> {
        if !self.has_async_copy_protocol {
            return Ok(());
        }
        let mut candidates = Vec::new();
        for (warp, slot) in self.async_copies.iter().enumerate() {
            let Some(state) = slot.state.as_deref() else {
                continue;
            };
            for group in &state.committed_groups {
                if group.pending_sources == 0
                    && group.destination_wavefronts != 0
                    && !group.destination_service_queued
                    && group.source_completion_time <= self.t
                {
                    candidates.push((
                        group.source_completion_time,
                        warp,
                        group.id,
                        group.destination_wavefronts,
                    ));
                }
            }
        }
        candidates
            .sort_unstable_by_key(|&(source_ready, warp, group, _)| (source_ready, warp, group.0));

        let mut projected_pipe = self.ldgsts_destination_pipe_ready;
        let mut assignments = Vec::with_capacity(candidates.len());
        for (source_ready, warp, group, wavefronts) in candidates {
            let service_cycles = self.params.ldgsts_destination_service_cycles(wavefronts)?;
            let start = projected_pipe.max(source_ready);
            let ready = checked_cycle_add(
                "LDGSTS aggregate destination-wavefront service",
                start,
                service_cycles,
            )?;
            projected_pipe = ready;
            assignments.push((warp, group, ready));
        }

        for (warp, group_id, ready) in assignments {
            let group = self.async_copies[warp]
                .async_group_mut(group_id)
                .expect("destination-service candidate remains committed");
            debug_assert!(!group.destination_service_queued);
            group.destination_service_queued = true;
            group.destination_completion_time = ready;
            group.completion_time = group.completion_time.max(ready);
        }
        self.ldgsts_destination_pipe_ready = projected_pipe;
        Ok(())
    }

    /// Run issue cycles strictly BEFORE `horizon`. Emits memory requests
    /// into `out` (L1Shared mode only).
    ///
    /// `horizon` is the earliest time a SELF-affecting external event can
    /// occur — the caller passes the earliest response already scheduled
    /// for this SM (other SMs' events are NOT barriers: cross-SM ordering
    /// lives entirely in the engine queue's `(time, seq)` order on the L2
    /// access stream). Requests emitted DURING the call tighten the cap
    /// to `issue_time + l1_miss_latency` (the L2-hit lower bound of the
    /// latency ladder) — their responses are not scheduled yet, so the
    /// caller cannot know them. The clock never crosses the cap, not even
    /// by skip-ahead: a response arriving at the cap may ready a warp
    /// EARLIER than the next known timer, and the clock is monotonic.
    pub(crate) fn advance(
        &mut self,
        horizon: ReplayHorizon,
        out: &mut Vec<MemRequest>,
    ) -> Result<Advance, IssueError> {
        if let Some(bound) = horizon.bounded_time()
            && bound < self.t
        {
            return Err(IssueError::HorizonBeforeClock {
                horizon: bound,
                clock: self.t,
            });
        }
        // New emission batch: line-fill folds are only valid within one
        // call's `out` (the caller schedules it after we return).
        if let MemModel::L1Shared(st) = &mut self.mem {
            st.line_folds.clear();
        }
        let mut horizon = horizon;
        while self.unfinished_warps > 0 {
            self.enqueue_ready_async_copy_destinations()?;
            if horizon.reached(self.t) {
                return Ok(Advance::Idle {
                    wake_at: Some(self.t),
                });
            }
            let before = out.len();
            let issued_this_cycle = self.issue_cycle(out)?;
            // A reply-carrying request emitted this cycle tightens the
            // safe horizon (mutation-only store traffic cannot affect
            // this SM and does not).
            if out[before..].iter().any(|r| r.reply != ReplyTo::None) {
                let lower_bound = checked_cycle_add(
                    "self-response horizon",
                    self.t,
                    self.params.l1_miss_latency,
                )?;
                horizon = horizon.tighten(lower_bound);
            }
            if issued_this_cycle {
                self.t = checked_cycle_add("issue-clock advance", self.t, 1)?;
                if self
                    .memory_admission_feedback
                    .as_mut()
                    .is_some_and(|state| state.take_yield_requested())
                {
                    return Ok(Advance::Idle {
                        wake_at: Some(self.t),
                    });
                }
                continue;
            }
            // Nothing ready: skip to the next KNOWN warp/resource timer
            // (resident warps only — retired warps left the sub-core lists;
            // warps waiting on in-flight responses have no computable time and
            // are excluded).
            let next_t = if self.has_async_copy_protocol {
                match (
                    self.next_known_issue_time::<true>(),
                    self.next_async_copy_destination_source_time(),
                ) {
                    (Some(issue), Some(source)) => Some(issue.min(source)),
                    (Some(issue), None) => Some(issue),
                    (None, Some(source)) => Some(source),
                    (None, None) => None,
                }
            } else {
                self.next_known_issue_time::<false>()
            };
            match next_t {
                Some(nt) => {
                    let next_cycle = checked_cycle_add("idle-clock advance", self.t, 1)?;
                    let target = nt.max(next_cycle);
                    if let Some(bound) = horizon.bounded_time()
                        && target >= bound
                    {
                        // Park AT the horizon instead of jumping past it:
                        // the event there may ready a warp before the
                        // next known timer.
                        return Ok(Advance::Idle {
                            wake_at: Some(bound),
                        });
                    }
                    self.t = target;
                }
                None if self.outstanding_responses() > 0
                    || self
                        .memory_admission_feedback
                        .as_ref()
                        .is_some_and(|state| state.has_unpriced()) =>
                {
                    return Ok(Advance::Idle { wake_at: None });
                }
                None => {
                    if let MemModel::L1Shared(st) = &self.mem {
                        debug_assert!(
                            st.retry.is_empty(),
                            "MSHR-stalled work implies in-flight fills"
                        );
                    }
                    return Err(IssueError::NoProgress {
                        time: self.t,
                        unfinished: self.unfinished_warps,
                    });
                }
            }
        }
        Ok(Advance::Done)
    }

    fn preflight_issue_timing(
        &self,
        kind: AccessKind,
        write_barrier: Option<DepBarrier>,
        read_barrier: Option<DepBarrier>,
        stall: u8,
        yield_flag: bool,
        pipeline_class: PipelineClass,
        retires_warp: bool,
        observe_async_copy: bool,
    ) -> Result<(), IssueError> {
        let t = self.t;
        checked_cycle_add("issue-clock advance", t, 1)?;
        match pipeline_class {
            PipelineClass::Branch {
                next_issue_interval_cycles,
            } => {
                checked_cycle_add("branch-pipeline resolution", t, next_issue_interval_cycles)?;
            }
            PipelineClass::Barrier {
                warp_initiation_interval_cycles,
                ..
            } => {
                checked_cycle_add(
                    "CTA-barrier pipeline availability",
                    t,
                    warp_initiation_interval_cycles,
                )?;
            }
            PipelineClass::Tensor {
                initiation_interval_cycles,
            } => {
                checked_cycle_add(
                    "tensor-pipeline availability",
                    t,
                    initiation_interval_cycles,
                )?;
            }
            PipelineClass::DispatchOnly => {}
        }
        if !retires_warp {
            checked_cycle_add("warp stall eligibility", t, u64::from(stall.max(1)))?;
            if yield_flag {
                checked_cycle_add("warp yield eligibility", t, 2)?;
            }
        }
        if !matches!(kind, AccessKind::PredicatedOffMemory) && read_barrier.is_some() {
            checked_cycle_add(
                "read-dependence barrier visibility",
                t,
                self.params.read_barrier_latency,
            )?;
        }

        let needs_ready = write_barrier.is_some() || observe_async_copy;
        match kind {
            AccessKind::NonMemory if needs_ready => {
                self.params.non_memory_ready(t)?;
            }
            AccessKind::Shared { wavefronts } if needs_ready => {
                self.params.shared_ready(t, wavefronts)?;
            }
            AccessKind::NonMemory | AccessKind::PredicatedOffMemory | AccessKind::Shared { .. } => {
            }
            AccessKind::GlobalEligible { .. }
            | AccessKind::LocalLoad { .. }
            | AccessKind::GlobalAsyncCopy {
                l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                ..
            } => match &self.mem {
                MemModel::Flat if needs_ready => {
                    checked_cycle_add(
                        "flat global visibility",
                        t,
                        self.params.global_flat_latency,
                    )?;
                }
                MemModel::L1Const(_) if needs_ready => {
                    checked_cycle_add("L1 hit-pipeline floor", t, self.params.l1_hit_latency)?;
                }
                MemModel::L1Shared(_) if needs_ready => {
                    checked_cycle_add(
                        "shared L1 hit-pipeline floor",
                        t,
                        self.params.l1_hit_latency,
                    )?;
                }
                MemModel::Flat | MemModel::L1Const(_) | MemModel::L1Shared(_) => {}
            },
            AccessKind::GlobalBypass { len, .. }
            | AccessKind::GlobalAsyncCopy {
                len,
                l1_policy: LdgstsL1Policy::BypassL1,
                ..
            }
            | AccessKind::GlobalUntagged { len, .. } => match &self.mem {
                MemModel::Flat if needs_ready => {
                    checked_cycle_add(
                        "flat global visibility",
                        t,
                        self.params.global_flat_latency,
                    )?;
                }
                MemModel::L1Const(_) if needs_ready => {
                    self.params.store_visibility_ready(t)?;
                }
                MemModel::L1Shared(_) if len == 0 && needs_ready => {
                    self.params.store_visibility_ready(t)?;
                }
                MemModel::L1Shared(_) if len > 0 && needs_ready => {
                    checked_cycle_add("self-response horizon", t, self.params.l1_miss_latency)?;
                }
                MemModel::Flat | MemModel::L1Const(_) | MemModel::L1Shared(_) => {}
            },
            AccessKind::GlobalStore { .. }
            | AccessKind::GlobalReduction { .. }
            | AccessKind::LocalStore { .. } => match &self.mem {
                MemModel::Flat if needs_ready => {
                    checked_cycle_add(
                        "flat global visibility",
                        t,
                        self.params.global_flat_latency,
                    )?;
                }
                MemModel::L1Const(_) | MemModel::L1Shared(_) if needs_ready => {
                    self.params.store_visibility_ready(t)?;
                }
                MemModel::Flat | MemModel::L1Const(_) | MemModel::L1Shared(_) => {}
            },
        }
        Ok(())
    }

    fn preflight_issue_memory(
        &self,
        picks: &[Option<usize>; SUB_CORES],
        async_destination_wavefronts: &[u64; SUB_CORES],
    ) -> Result<(), IssueError> {
        let (l1, shared) = match &self.mem {
            MemModel::Flat => return Ok(()),
            MemModel::L1Const(l1) => (l1, None),
            MemModel::L1Shared(st) => (&st.l1, Some(st)),
        };
        let mut budgets = MemoryOrigin::ALL.map(L1MutationBudget::for_origin);
        let mut eligible_sectors = 0u64;
        let mut pending_budget_fits = true;
        let mut constant_async_deliveries = [None; SUB_CORES];
        for (subcore, pick) in picks.iter().enumerate() {
            let Some(wix) = *pick else { continue };
            let warp = &self.warps[wix];
            let inst = warp
                .insts
                .get(warp.next)
                .ok_or(IssueError::CounterOverflow {
                    counter: "warp instruction cursor",
                })?;
            let slot = warp
                .next_slot
                .ok_or(IssueError::MissingIssueSlot { warp: wix })?;
            let write_barrier = if self.has_async_copy_protocol {
                ordinary_write_barrier(slot)
            } else {
                slot.control.write_barrier
            };
            let add_budget = |counter: &'static str, value: &mut u64, add: u64| {
                *value = checked_u64_counter_add(counter, *value, add)?;
                Ok::<(), IssueError>(())
            };
            match inst.kind {
                AccessKind::GlobalEligible { off, len, .. }
                | AccessKind::LocalLoad { off, len }
                | AccessKind::GlobalAsyncCopy {
                    off,
                    len,
                    l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                    ..
                } => {
                    let origin = match inst.kind {
                        AccessKind::GlobalEligible { .. } | AccessKind::GlobalAsyncCopy { .. } => {
                            MemoryOrigin::Global
                        }
                        AccessKind::LocalLoad { .. } => MemoryOrigin::Local,
                        _ => unreachable!("match arm admits only hierarchy loads"),
                    };
                    let budget = &mut budgets[origin.index()];
                    let sectors = u64::from(len);
                    add_budget(
                        "L1 eligible-sector budget",
                        &mut budget.eligible_sectors,
                        sectors,
                    )?;
                    add_budget(
                        "combined L1 eligible-sector budget",
                        &mut eligible_sectors,
                        sectors,
                    )?;
                    add_budget("L1 recency budget", &mut budget.recency_touches, sectors)?;
                    add_budget("L1 fill budget", &mut budget.new_fills, sectors)?;
                    if shared.is_none()
                        && let Some(completion_work) = async_copy_completion_work(inst.kind)
                    {
                        debug_assert!(matches!(
                            inst.kind,
                            AccessKind::GlobalAsyncCopy {
                                l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                                ..
                            }
                        ));
                        constant_async_deliveries[subcore] =
                            Some((wix, completion_work, async_destination_wavefronts[subcore]));
                    }
                    if shared.is_some()
                        && warp
                            .preflight_pending_add(wix, write_barrier, u32::from(len))
                            .is_err()
                    {
                        pending_budget_fits = false;
                    }
                    if let Some(st) = shared {
                        let end = off.checked_add(usize::from(len)).ok_or(
                            IssueError::CounterOverflow {
                                counter: "issue sector cursor",
                            },
                        )?;
                        for sector_index in off..end {
                            let sector_id = *warp.sectors.get(sector_index).ok_or(
                                IssueError::CounterOverflow {
                                    counter: "issue sector cursor",
                                },
                            )?;
                            preflight_line_fill_mask(st, sector_id)?;
                            let response_id =
                                ResponseId::new(st.response_owner, wix, warp.next, sector_index);
                            if let Some(&reply) = st
                                .outstanding_replies
                                .iter()
                                .find(|reply| reply.response_id() == Some(response_id))
                            {
                                return Err(IssueError::ResponseTokenAlreadyOutstanding { reply });
                            }
                        }
                    }
                }
                AccessKind::GlobalBypass {
                    len, token, off, ..
                } => {
                    let budget = &mut budgets[MemoryOrigin::Global.index()];
                    let sectors = u64::from(len);
                    add_budget(
                        "L1 bypass-sector budget",
                        &mut budget.bypass_sectors,
                        sectors,
                    )?;
                    if token == CacheOpToken::Atom {
                        add_budget(
                            "L1 bypass-atom budget",
                            &mut budget.bypass_atom_sectors,
                            sectors,
                        )?;
                    }
                    if let Some(st) = shared
                        && write_barrier.is_some()
                    {
                        if warp
                            .preflight_pending_add(wix, write_barrier, u32::from(len))
                            .is_err()
                        {
                            pending_budget_fits = false;
                        }
                        for sector_index in off..off + usize::from(len) {
                            let response_id =
                                ResponseId::new(st.response_owner, wix, warp.next, sector_index);
                            if let Some(&reply) = st
                                .outstanding_replies
                                .iter()
                                .find(|reply| reply.response_id() == Some(response_id))
                            {
                                return Err(IssueError::ResponseTokenAlreadyOutstanding { reply });
                            }
                        }
                    }
                }
                AccessKind::GlobalAsyncCopy {
                    off,
                    len,
                    completion_work,
                    l1_policy: LdgstsL1Policy::BypassL1,
                    ..
                } => {
                    let budget = &mut budgets[MemoryOrigin::Global.index()];
                    let sectors = u64::from(len);
                    add_budget(
                        "L1 bypass-sector budget",
                        &mut budget.bypass_sectors,
                        sectors,
                    )?;
                    if let Some(st) = shared
                        && (write_barrier.is_some() || completion_work != 0)
                    {
                        if warp
                            .preflight_pending_add(wix, write_barrier, u32::from(len))
                            .is_err()
                        {
                            pending_budget_fits = false;
                        }
                        for sector_index in off..off + usize::from(len) {
                            let response_id =
                                ResponseId::new(st.response_owner, wix, warp.next, sector_index);
                            if let Some(&reply) = st
                                .outstanding_replies
                                .iter()
                                .find(|reply| reply.response_id() == Some(response_id))
                            {
                                return Err(IssueError::ResponseTokenAlreadyOutstanding { reply });
                            }
                        }
                    }
                }
                AccessKind::GlobalUntagged { off, len } => {
                    let budget = &mut budgets[MemoryOrigin::Global.index()];
                    let sectors = u64::from(len);
                    add_budget(
                        "L1 untagged-sector budget",
                        &mut budget.untagged_sectors,
                        sectors,
                    )?;
                    if let Some(st) = shared
                        && write_barrier.is_some()
                    {
                        if warp
                            .preflight_pending_add(wix, write_barrier, u32::from(len))
                            .is_err()
                        {
                            pending_budget_fits = false;
                        }
                        for sector_index in off..off + usize::from(len) {
                            let response_id =
                                ResponseId::new(st.response_owner, wix, warp.next, sector_index);
                            if let Some(&reply) = st
                                .outstanding_replies
                                .iter()
                                .find(|reply| reply.response_id() == Some(response_id))
                            {
                                return Err(IssueError::ResponseTokenAlreadyOutstanding { reply });
                            }
                        }
                    }
                }
                AccessKind::GlobalStore { len, .. }
                | AccessKind::GlobalReduction { len, .. }
                | AccessKind::LocalStore { len, .. } => {
                    let origin = match inst.kind {
                        AccessKind::GlobalStore { .. } | AccessKind::GlobalReduction { .. } => {
                            MemoryOrigin::Global
                        }
                        AccessKind::LocalStore { .. } => MemoryOrigin::Local,
                        _ => unreachable!("match arm admits only hierarchy writes"),
                    };
                    let budget = &mut budgets[origin.index()];
                    let sectors = u64::from(len);
                    if matches!(inst.kind, AccessKind::GlobalReduction { .. }) {
                        add_budget(
                            "L1 reduction-sector budget",
                            &mut budget.reduction_sectors,
                            sectors,
                        )?;
                    } else {
                        add_budget("L1 store-sector budget", &mut budget.store_sectors, sectors)?;
                    }
                    add_budget("L1 recency budget", &mut budget.recency_touches, sectors)?;
                }
                AccessKind::NonMemory
                | AccessKind::PredicatedOffMemory
                | AccessKind::Shared { .. } => {}
            }
        }

        let constant_ready_upper = if shared.is_none() {
            l1.constant_batch_ready_upper_bound(self.t, eligible_sectors)
        } else {
            None
        };
        let constant_timing_fits = shared.is_some() || constant_ready_upper.is_some();
        let constant_async_completion_fits = shared.is_some()
            || constant_async_deliveries.iter().flatten().all(
                |&(wix, delivery_work, destination_wavefronts)| {
                    constant_ready_upper.is_some_and(|source_ready| {
                        self.async_copies[wix]
                            .preflight_async_copy_add(
                                wix,
                                0,
                                source_ready,
                                delivery_work,
                                destination_wavefronts,
                                None,
                                &self.params,
                            )
                            .is_ok()
                    })
                },
            );
        let shared_timing_fits = shared.is_none()
            || eligible_sectors == 0
            || (self.t.checked_add(self.params.l1_hit_latency).is_some()
                && self.t.checked_add(self.params.l1_miss_latency).is_some());
        if l1.preflight_mutation_budgets(&budgets).is_ok()
            && pending_budget_fits
            && constant_timing_fits
            && constant_async_completion_fits
            && shared_timing_fits
        {
            return Ok(());
        }

        // Near a numeric boundary, replay this at-most-four-instruction batch
        // on a shadow cache. This avoids rejecting valid warm hits while keeping
        // the ordinary hot path allocation-free.
        let mut shadow = l1.clone();
        for (subcore, pick) in picks.iter().enumerate() {
            let Some(wix) = *pick else { continue };
            let warp = &self.warps[wix];
            let inst = warp.insts[warp.next];
            let slot = warp
                .next_slot
                .ok_or(IssueError::MissingIssueSlot { warp: wix })?;
            let write_barrier = if self.has_async_copy_protocol {
                ordinary_write_barrier(slot)
            } else {
                slot.control.write_barrier
            };
            match inst.kind {
                AccessKind::GlobalEligible { off, len, .. }
                | AccessKind::LocalLoad { off, len }
                | AccessKind::GlobalAsyncCopy {
                    off,
                    len,
                    l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                    ..
                } => {
                    let origin = match inst.kind {
                        AccessKind::GlobalEligible { .. } | AccessKind::GlobalAsyncCopy { .. } => {
                            MemoryOrigin::Global
                        }
                        AccessKind::LocalLoad { .. } => MemoryOrigin::Local,
                        _ => unreachable!("match arm admits only hierarchy loads"),
                    };
                    let mut pending_add = 0u32;
                    let async_delivery = async_copy_completion_work(inst.kind);
                    let observe_ready = write_barrier.is_some() || async_delivery.is_some();
                    let mut async_source_ready = if shared.is_none() && async_delivery.is_some() {
                        Some(checked_cycle_add(
                            "LDGSTS L1 hit-pipeline floor",
                            self.t,
                            self.params.l1_hit_latency,
                        )?)
                    } else {
                        None
                    };
                    let mshr_floor = matches!(
                        inst.kind,
                        AccessKind::GlobalAsyncCopy {
                            l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                            ..
                        }
                    )
                    .then_some(self.params.ldgsts_l1_mshr_floor);
                    for &sector_id in &warp.sectors[off..off + usize::from(len)] {
                        if shared.is_some() {
                            match access_shared_l1(
                                &mut shadow,
                                origin,
                                sector_id,
                                self.t,
                                observe_ready,
                                mshr_floor,
                            )? {
                                SharedL1Access::Hit { .. } => {}
                                SharedL1Access::MissRequest { .. } => {
                                    checked_cycle_add(
                                        "self-response horizon",
                                        self.t,
                                        self.params.l1_miss_latency,
                                    )?;
                                    pending_add = checked_u32_counter_add(
                                        "issue exact pending-add budget",
                                        pending_add,
                                        1,
                                    )?;
                                }
                                SharedL1Access::MergedPending { .. } => {
                                    pending_add = checked_u32_counter_add(
                                        "issue exact pending-add budget",
                                        pending_add,
                                        1,
                                    )?;
                                }
                                SharedL1Access::Stalled => {
                                    pending_add = checked_u32_counter_add(
                                        "issue exact pending-add budget",
                                        pending_add,
                                        1,
                                    )?;
                                    break;
                                }
                            }
                        } else if observe_ready {
                            let (_, ready) = match mshr_floor {
                                Some(floor) => shadow.access_eligible_with_mshr_floor(
                                    origin, sector_id, self.t, floor,
                                )?,
                                None => shadow.access_eligible(origin, sector_id, self.t)?,
                            };
                            if let Some(current) = &mut async_source_ready {
                                *current = (*current).max(ready);
                            }
                        } else {
                            match mshr_floor {
                                Some(floor) => {
                                    shadow.access_eligible_unobserved_with_mshr_floor(
                                        origin, sector_id, self.t, floor,
                                    )?;
                                }
                                None => {
                                    shadow.access_eligible_unobserved(origin, sector_id, self.t)?;
                                }
                            }
                        }
                    }
                    if let (Some(source_ready), Some(delivery_work)) =
                        (async_source_ready, async_delivery)
                    {
                        self.async_copies[wix].preflight_async_copy_add(
                            wix,
                            0,
                            source_ready,
                            delivery_work,
                            async_destination_wavefronts[subcore],
                            None,
                            &self.params,
                        )?;
                    }
                    if shared.is_some() {
                        warp.preflight_pending_add(wix, write_barrier, pending_add)?;
                    }
                }
                AccessKind::GlobalBypass { len, token, .. } => {
                    shadow.count_bypass(
                        MemoryOrigin::Global,
                        u64::from(len),
                        token == CacheOpToken::Atom,
                    )?;
                    if shared.is_some() {
                        warp.preflight_pending_add(wix, write_barrier, u32::from(len))?;
                    }
                }
                AccessKind::GlobalAsyncCopy {
                    len,
                    l1_policy: LdgstsL1Policy::BypassL1,
                    ..
                } => {
                    shadow.count_bypass(MemoryOrigin::Global, u64::from(len), false)?;
                    if shared.is_some() {
                        warp.preflight_pending_add(wix, write_barrier, u32::from(len))?;
                    }
                }
                AccessKind::GlobalUntagged { len, .. } => {
                    shadow.count_untagged(MemoryOrigin::Global, u64::from(len))?;
                    if shared.is_some() {
                        warp.preflight_pending_add(wix, write_barrier, u32::from(len))?;
                    }
                }
                AccessKind::GlobalStore { off, len }
                | AccessKind::GlobalReduction { off, len, .. }
                | AccessKind::LocalStore { off, len } => {
                    let origin = match inst.kind {
                        AccessKind::GlobalStore { .. } | AccessKind::GlobalReduction { .. } => {
                            MemoryOrigin::Global
                        }
                        AccessKind::LocalStore { .. } => MemoryOrigin::Local,
                        _ => unreachable!("match arm admits only hierarchy writes"),
                    };
                    let reduction = matches!(inst.kind, AccessKind::GlobalReduction { .. });
                    for &sector_id in &warp.sectors[off..off + usize::from(len)] {
                        if reduction {
                            shadow.probe_reduction(origin, sector_id, self.t)?;
                        } else {
                            shadow.probe_store(origin, sector_id, self.t)?;
                        }
                    }
                }
                AccessKind::NonMemory
                | AccessKind::PredicatedOffMemory
                | AccessKind::Shared { .. } => {}
            }
        }
        Ok(())
    }

    fn preflight_issue_accounting(
        &self,
        picks: &[Option<usize>; SUB_CORES],
        barrier_arrival_times: &[Option<u64>; SUB_CORES],
    ) -> Result<u64, IssueError> {
        let issue_add = u64::try_from(picks.iter().flatten().count()).map_err(|_| {
            IssueError::CounterOverflow {
                counter: "issued instruction",
            }
        })?;
        let next_issued = checked_u64_counter_add("issued instruction", self.issued, issue_add)?;
        if next_issued > self.instruction_limit {
            return Err(IssueError::InstructionCountExceedsReplay {
                issued: self.issued,
                add: issue_add,
                limit: self.instruction_limit,
            });
        }

        let mut block_deltas: [Option<(usize, usize, usize)>; SUB_CORES] = [None; SUB_CORES];
        let mut total_retirements = 0usize;
        for &wix in picks.iter().flatten() {
            let warp = &self.warps[wix];
            let retires = warp.next_issue_retires()?;
            let slot = warp
                .next_slot
                .ok_or(IssueError::MissingIssueSlot { warp: wix })?;
            let entry_index = block_deltas
                .iter()
                .position(|entry| entry.is_some_and(|(block, _, _)| block == warp.block))
                .or_else(|| block_deltas.iter().position(Option::is_none))
                .ok_or(IssueError::CounterOverflow {
                    counter: "per-cycle block accounting",
                })?;
            let entry = block_deltas[entry_index].get_or_insert((warp.block, 0, 0));
            if retires {
                entry.1 = checked_usize_counter_add("block retirement", entry.1, 1)?;
                total_retirements =
                    checked_usize_counter_add("unfinished warp retirement", total_retirements, 1)?;
            } else if slot.op_class == OpClass::BlockBarrier {
                entry.2 = checked_usize_counter_add("block barrier waiter", entry.2, 1)?;
            }
        }

        self.unfinished_warps.checked_sub(total_retirements).ok_or(
            IssueError::CounterUnderflow {
                counter: "unfinished warp",
            },
        )?;
        let mut residency_released = 0u32;
        let mut shared_memory_released = 0u64;
        for (block, retirements, arrivals) in block_deltas.into_iter().flatten() {
            let state = &self.block_states[block];
            let unfinished =
                state
                    .unfinished
                    .checked_sub(retirements)
                    .ok_or(IssueError::CounterUnderflow {
                        counter: "block unfinished warp",
                    })?;
            let waiting =
                checked_usize_counter_add("block barrier waiter", state.waiting, arrivals)?;
            if waiting > unfinished {
                return Err(IssueError::CounterOverflow {
                    counter: "block barrier waiter",
                });
            }
            if waiting > 0 && waiting == unfinished {
                let mut last_arrival = self.t;
                for warp in &self.warps[state.first_warp..state.first_warp + state.admitted_warps] {
                    if let Some(arrival) = warp.block_barrier_arrival {
                        last_arrival = last_arrival.max(arrival);
                    }
                }
                for (subcore, pick) in picks.iter().enumerate() {
                    if pick.is_some_and(|warp| self.warps[warp].block == block) {
                        last_arrival =
                            last_arrival.max(barrier_arrival_times[subcore].unwrap_or(self.t));
                    }
                }
                checked_cycle_add("block-barrier release", last_arrival, 1)?;
                if let Some(issue) = self.barrier_issue {
                    checked_cycle_add(
                        "CTA-barrier generation re-arm",
                        last_arrival,
                        issue.generation_rearm_cycles,
                    )?;
                }
            }
            if unfinished == 0 {
                for &warp_slot in &state.warp_slots {
                    if !self.resident_warp_slots.contains(&warp_slot) {
                        return Err(IssueError::ResidentWarpSlotMissing {
                            launch_index: self.warps[state.first_warp].launch_index,
                            warp_slot,
                        });
                    }
                }
                residency_released = checked_u32_counter_add(
                    "resident warp retirement",
                    residency_released,
                    state.warp_count,
                )?;
                shared_memory_released = checked_u64_counter_add(
                    "resident shared-memory retirement",
                    shared_memory_released,
                    state.shared_memory_bytes,
                )?;
            }
        }
        self.resident_warps.checked_sub(residency_released).ok_or(
            IssueError::CounterUnderflow {
                counter: "resident warp",
            },
        )?;
        self.resident_shared_memory_bytes
            .checked_sub(shared_memory_released)
            .ok_or(IssueError::CounterUnderflow {
                counter: "resident shared-memory byte",
            })?;
        Ok(next_issued)
    }

    fn preflight_result_writeback_priority(
        &self,
        picks: &[Option<usize>; SUB_CORES],
    ) -> Result<(), IssueError> {
        let Some(state) = self.result_writebacks.as_ref() else {
            return Ok(());
        };
        let mut collisions = 0u64;
        let mut deferral_cycles = 0u64;
        for (subcore, pick) in picks.iter().enumerate() {
            let Some(wix) = *pick else { continue };
            let slot = self.warps[wix]
                .next_slot
                .ok_or(IssueError::MissingIssueSlot { warp: wix })?;
            let Some(ResultWritebackKind::FixedHadd2 { bank }) =
                result_writeback_kind(slot, self.params.arch)
            else {
                continue;
            };
            let fixed_at = checked_cycle_add(
                "HADD2 result writeback",
                self.t,
                state.timing.hadd2_writeback_cycles,
            )?;
            for event in &state.pending {
                if event.subcore == subcore as u8
                    && event.bank == bank
                    && event.writeback_at == fixed_at
                {
                    checked_cycle_add(
                        "deferred L1TEX writeback",
                        event.writeback_at,
                        state.timing.variable_deferral_cycles,
                    )?;
                    checked_cycle_add(
                        "deferred L1TEX visibility",
                        event.visible_at,
                        state.timing.variable_deferral_cycles,
                    )?;
                    collisions = checked_u64_counter_add(
                        "fixed-over-L1TEX collision preflight",
                        collisions,
                        1,
                    )?;
                    deferral_cycles = checked_u64_counter_add(
                        "L1TEX writeback deferral-cycle preflight",
                        deferral_cycles,
                        state.timing.variable_deferral_cycles,
                    )?;
                }
            }
        }
        checked_u64_counter_add(
            "fixed-over-L1TEX collision diagnostic",
            self.readiness.result_writeback.fixed_over_l1tex_collisions,
            collisions,
        )?;
        checked_u64_counter_add(
            "L1TEX writeback deferral-cycle diagnostic",
            self.readiness.result_writeback.l1tex_deferral_cycles,
            deferral_cycles,
        )?;
        Ok(())
    }

    fn register_l1tex_writeback(
        &mut self,
        subcore: usize,
        warp: usize,
        kind: ResultWritebackKind,
        barrier: Option<DepBarrier>,
        visible_at: Option<u64>,
    ) -> Result<(), IssueError> {
        let l1_hit_latency = self.params.l1_hit_latency;
        let (
            Some(state),
            ResultWritebackKind::L1TexStrongSmU32 { bank },
            Some(barrier),
            Some(visible_at),
        ) = (self.result_writebacks.as_mut(), kind, barrier, visible_at)
        else {
            return Ok(());
        };
        let hit_floor = checked_cycle_add("L1 hit-pipeline floor", self.t, l1_hit_latency)?;
        if visible_at != hit_floor
            || l1_hit_latency < state.timing.l1tex_writeback_to_visibility_cycles
        {
            return Ok(());
        }
        let writeback_at = visible_at - state.timing.l1tex_writeback_to_visibility_cycles;
        state.pending.push(PendingVariableWriteback {
            subcore: subcore as u8,
            bank,
            writeback_at,
            visible_at,
            warp,
            barrier,
        });
        Ok(())
    }

    fn apply_fixed_result_priority(
        &mut self,
        subcore: usize,
        kind: ResultWritebackKind,
        projected: &mut ReadinessDiagnostics,
    ) -> Result<(), IssueError> {
        let (Some(state), ResultWritebackKind::FixedHadd2 { bank }) =
            (self.result_writebacks.as_mut(), kind)
        else {
            return Ok(());
        };
        let fixed_at = checked_cycle_add(
            "HADD2 result writeback",
            self.t,
            state.timing.hadd2_writeback_cycles,
        )?;
        let (warps, diagnostics) = (&mut self.warps, &mut projected.result_writeback);
        for event in &mut state.pending {
            if event.subcore != subcore as u8
                || event.bank != bank
                || event.writeback_at != fixed_at
            {
                continue;
            }
            event.writeback_at = checked_cycle_add(
                "deferred L1TEX writeback",
                event.writeback_at,
                state.timing.variable_deferral_cycles,
            )?;
            event.visible_at = checked_cycle_add(
                "deferred L1TEX visibility",
                event.visible_at,
                state.timing.variable_deferral_cycles,
            )?;
            warps[event.warp].stamp_clear(
                Some(event.barrier),
                event.visible_at,
                DependencyOrigin::L1Tex,
            );
            diagnostics.fixed_over_l1tex_collisions = checked_u64_counter_add(
                "fixed-over-L1TEX collision diagnostic",
                diagnostics.fixed_over_l1tex_collisions,
                1,
            )?;
            diagnostics.l1tex_deferral_cycles = checked_u64_counter_add(
                "L1TEX writeback deferral-cycle diagnostic",
                diagnostics.l1tex_deferral_cycles,
                state.timing.variable_deferral_cycles,
            )?;
        }
        Ok(())
    }

    /// One issue cycle at `self.t` across the four sub-cores. Returns
    /// whether anything issued.
    fn issue_cycle(&mut self, out: &mut Vec<MemRequest>) -> Result<bool, IssueError> {
        let t = self.t;
        let picks = if self.has_async_copy_protocol {
            self.pick_warps_with_memory_admission::<true>(t)
        } else {
            self.pick_warps_with_memory_admission::<false>(t)
        };
        let barrier_arrival_times = self.project_barrier_arrival_times(&picks, t)?;
        let mut async_destination_wavefronts = [0; SUB_CORES];
        for (subcore, pick) in picks.iter().enumerate() {
            let Some(wix) = *pick else { continue };
            let kind = self.warps[wix].insts[self.warps[wix].next].kind;
            async_destination_wavefronts[subcore] =
                async_copy_destination_wavefronts(kind, &self.params)?.unwrap_or(0);
        }

        // A cycle is the caller-visible transaction boundary. Validate every
        // selected instruction's possible timing arms before any sub-core
        // mutates counters, cache state, barriers, requests, or retirement.
        for (subcore, pick) in picks.iter().enumerate() {
            let Some(wix) = *pick else { continue };
            let w = &self.warps[wix];
            let slot = w
                .next_slot
                .ok_or(IssueError::MissingIssueSlot { warp: wix })?;
            let retires_warp = w.next_issue_retires()?;
            let kind = w.insts[w.next].kind;
            validate_async_copy_completion_work(kind, wix)?;
            let write_barrier = if self.has_async_copy_protocol {
                ordinary_write_barrier(slot)
            } else {
                slot.control.write_barrier
            };
            let observe_async_copy =
                self.has_async_copy_protocol && async_copy_source_count(kind).is_some();
            if observe_async_copy && let Some(delivery_work) = async_copy_completion_work(kind) {
                let source_ready = match &self.mem {
                    MemModel::Flat => checked_cycle_add(
                        "flat async-copy source visibility",
                        self.t,
                        self.params.global_flat_latency,
                    )?,
                    MemModel::L1Const(_) => checked_cycle_add(
                        "constant-L1 async-copy source upper bound",
                        self.t,
                        self.params.l1_miss_latency,
                    )?,
                    MemModel::L1Shared(_) => checked_cycle_add(
                        "shared-L1 async-copy known-source floor",
                        self.t,
                        self.params.l1_hit_latency,
                    )?,
                };
                let pending_add = if matches!(self.mem, MemModel::L1Shared(_)) {
                    u32::from(async_copy_source_count(kind).unwrap_or(0))
                } else {
                    0
                };
                self.async_copies[wix].preflight_async_copy_add(
                    wix,
                    pending_add,
                    source_ready,
                    delivery_work,
                    async_destination_wavefronts[subcore],
                    (matches!(self.mem, MemModel::L1Shared(_))
                        && matches!(
                            kind,
                            AccessKind::GlobalAsyncCopy {
                                l1_policy: LdgstsL1Policy::BypassL1,
                                ..
                            }
                        ))
                    .then_some(
                        bypass_async_copy_observed_topology(kind, w.next).unwrap_or(
                            BypassAsyncCopyInput {
                                instruction: w.next,
                                source_cohorts: 0,
                                destination_lines: 0,
                                max_bank_wavefronts: 0,
                                max_bypass_wavefronts: 0,
                            },
                        ),
                    ),
                    &self.params,
                )?;
            }
            self.preflight_issue_timing(
                kind,
                write_barrier,
                slot.control.read_barrier,
                slot.control.stall,
                slot.control.yield_flag,
                slot.pipeline_class,
                retires_warp,
                observe_async_copy,
            )?;
        }
        self.preflight_issue_memory(&picks, &async_destination_wavefronts)?;
        self.preflight_result_writeback_priority(&picks)?;
        let next_issued = self.preflight_issue_accounting(&picks, &barrier_arrival_times)?;
        // Diagnostics are preflighted last so an observational counter can
        // never mask an established execution/cache/domain error. No mutation
        // has occurred yet, so a diagnostic overflow remains transactional.
        let mut projected_readiness =
            self.project_picked_readiness(&picks, &barrier_arrival_times, t)?;
        let projected_memory_service_candidates =
            self.project_memory_service_candidates(&picks, t)?;
        let memory_admission_feedback_plan =
            if let Some(feedback) = self.memory_admission_feedback.as_ref() {
                let mut issues = [None; SUB_CORES];
                for (subcore, pick) in picks.iter().enumerate() {
                    let Some(warp) = *pick else { continue };
                    let state = &self.warps[warp];
                    let slot = state
                        .next_slot
                        .ok_or(IssueError::MissingIssueSlot { warp })?;
                    issues[subcore] = Some(MemoryAdmissionFeedbackIssue {
                        subcore,
                        warp,
                        instruction: state.next,
                        kind: state.insts[state.next].kind,
                        prior_clock_reads: state.clock_reads_issued,
                        clock_read: is_clocklo_issue(slot),
                    });
                }
                Some(feedback.plan_issue(&issues, t)?)
            } else {
                None
            };

        if let Some(state) = self.result_writebacks.as_mut() {
            state.pending.retain(|event| event.writeback_at >= t);
        }

        let mut issued_this_cycle = false;
        for (sc, pick) in picks.into_iter().enumerate() {
            let Some(wix) = pick else { continue };
            issued_this_cycle = true;

            // Issue exactly one instruction from warp `wix`.
            let (
                blk_ix,
                launch_index,
                warp_slot,
                slot_class,
                pipeline_class,
                control,
                instruction,
                inst,
                warp_sectors,
                retires_warp,
                result_writeback,
                clock_read,
            ) = {
                let w = &self.warps[wix];
                let slot = w
                    .next_slot
                    .ok_or(IssueError::MissingIssueSlot { warp: wix })?;
                (
                    w.block,
                    w.launch_index,
                    w.warp_slot,
                    slot.op_class,
                    slot.pipeline_class,
                    slot.control,
                    w.next,
                    w.insts[w.next],
                    w.sectors,
                    w.next_issue_retires()?,
                    result_writeback_kind(slot, self.params.arch),
                    is_clocklo_issue(slot),
                )
            };
            let produces_dynamic_effects = !matches!(inst.kind, AccessKind::PredicatedOffMemory);
            let write_barrier = if !produces_dynamic_effects
                || (self.has_async_copy_protocol && slot_class == OpClass::AsyncCopyCommit)
            {
                None
            } else {
                control.write_barrier
            };
            let result_writeback = produces_dynamic_effects
                .then_some(result_writeback)
                .flatten();
            let observe_async_copy =
                self.has_async_copy_protocol && async_copy_source_count(inst.kind).is_some();

            // Memory-model interaction happens at issue time regardless of
            // whether a dependence barrier consumes the ready time.
            let data_ready = if matches!(self.mem, MemModel::L1Shared(_)) {
                self.mem_step_shared(
                    sc,
                    wix,
                    instruction,
                    inst.kind,
                    warp_sectors,
                    write_barrier,
                    async_destination_wavefronts[sc],
                    out,
                )?
            } else {
                data_ready_time(
                    inst.kind,
                    warp_sectors,
                    t,
                    &self.params,
                    self.mem.const_l1_mut(),
                    write_barrier,
                    observe_async_copy,
                )?
            };
            if observe_async_copy && let Some(source_ready) = data_ready {
                // Shared mode already attached the copy's delivery and exact
                // destination work to its pending group before walking source
                // sectors. This second update contributes only the known hit
                // floor; synchronous modes create the group here instead.
                let (delivery_work, destination_wavefronts) =
                    if matches!(self.mem, MemModel::L1Shared(_)) {
                        (0, 0)
                    } else {
                        (
                            async_copy_completion_work(inst.kind).unwrap_or(0),
                            async_destination_wavefronts[sc],
                        )
                    };
                self.async_copies[wix].add_known_async_copy(
                    wix,
                    source_ready,
                    delivery_work,
                    destination_wavefronts,
                    &self.params,
                )?;
            }

            match pipeline_class {
                PipelineClass::Branch { .. } => {}
                PipelineClass::Barrier { .. } => {
                    if let Some(arrival) = barrier_arrival_times[sc] {
                        self.barrier_pipe_ready = arrival;
                    }
                }
                PipelineClass::Tensor {
                    initiation_interval_cycles,
                } => {
                    self.tensor_pipe_ready[sc] = checked_cycle_add(
                        "tensor-pipeline availability",
                        t,
                        initiation_interval_cycles,
                    )?;
                }
                PipelineClass::DispatchOnly => {}
            }
            // Yield semantics per the CGGTY paper: the yield bit means
            // "do not issue this warp in the NEXT cycle" — a one-cycle
            // bubble when no other warp is ready — on top of the stall
            // count. A retiring instruction has no next eligibility to observe.
            if !retires_warp {
                let stall = u64::from(control.stall.max(1));
                let mut eligible = checked_cycle_add("warp stall eligibility", t, stall)?;
                if control.yield_flag {
                    eligible = eligible.max(checked_cycle_add("warp yield eligibility", t, 2)?);
                }
                self.warps[wix].eligible = eligible;
            }
            // Dependence barriers are COUNTERS in hardware (producer
            // increments at issue, decrements at write-back; the
            // consumer waits for zero), so with several in-flight
            // producers on one index the barrier clears at the LATEST
            // write-back: accumulate with max, never overwrite — a
            // later fast producer (L1 hit) must not shorten a pending
            // slow one (miss). Unknown write-backs (in-flight responses)
            // are tracked in `barrier_pending`.
            if let Some(dr) = data_ready {
                self.warps[wix].stamp_clear(write_barrier, dr, dependency_origin(inst.kind));
            }
            if let Some(result_writeback) = result_writeback {
                self.register_l1tex_writeback(
                    sc,
                    wix,
                    result_writeback,
                    write_barrier,
                    data_ready,
                )?;
                self.apply_fixed_result_priority(sc, result_writeback, &mut projected_readiness)?;
            }
            if produces_dynamic_effects && let Some(read_barrier) = control.read_barrier {
                let read_clear = checked_cycle_add(
                    "read-dependence barrier visibility",
                    t,
                    self.params.read_barrier_latency,
                )?;
                self.warps[wix].stamp_clear(
                    Some(read_barrier),
                    read_clear,
                    operand_read_dependency_origin(inst.kind),
                );
            }
            if clock_read {
                let warp = &mut self.warps[wix];
                warp.clock_first_issue.get_or_insert(t);
                warp.clock_reads_issued = warp
                    .clock_reads_issued
                    .checked_add(1)
                    .expect("CLOCKLO issue counter was preflighted");
            }
            if self.has_async_copy_protocol {
                match slot_class {
                    OpClass::AsyncCopyCommit => {
                        let expected_nonempty = self.async_copies[wix]
                            .state
                            .as_deref()
                            .and_then(|state| state.open_group)
                            .is_some();
                        let committed = self.async_copies[wix].commit_async_copy_group(t);
                        debug_assert_eq!(
                            committed, expected_nonempty,
                            "async commit mutation disagrees with its pre-issue state"
                        );
                    }
                    OpClass::AsyncCopyWait => {
                        let max_pending_groups = self.warps[wix]
                            .next_slot
                            .ok_or(IssueError::MissingIssueSlot { warp: wix })?
                            .async_copy_wait_max_pending_groups;
                        self.async_copies[wix].consume_async_wait(max_pending_groups, t);
                    }
                    // `Other` includes explicit SB1..SB5 DEPBAR waits. Their
                    // readiness was enforced through the ordinary barrier
                    // state above; they never mutate asynchronous-copy groups.
                    OpClass::BlockBarrier | OpClass::Other => {}
                }
            }

            // Advance the warp's stream.
            self.warps[wix].next =
                checked_usize_counter_add("warp instruction cursor", self.warps[wix].next, 1)?;
            if retires_warp {
                self.warps[wix].done = true;
                self.warps[wix].next_slot = None;
                self.warps[wix].ordinary_dependency_wait = None;
                self.unfinished_warps =
                    self.unfinished_warps
                        .checked_sub(1)
                        .ok_or(IssueError::CounterUnderflow {
                            counter: "unfinished warp",
                        })?;
                self.block_states[blk_ix].unfinished = self.block_states[blk_ix]
                    .unfinished
                    .checked_sub(1)
                    .ok_or(IssueError::CounterUnderflow {
                        counter: "block unfinished warp",
                    })?;
                // Retire from the sub-core pick list (swap-remove +
                // position fix-up) so pick/skip-ahead scans stay
                // bounded by RESIDENT warps, not warps-ever-admitted.
                let pos = self.warps[wix].pos_in_subcore;
                self.subcore_warps[sc].swap_remove(pos);
                if let Some(&moved) = self.subcore_warps[sc].get(pos) {
                    self.warps[moved].pos_in_subcore = pos;
                }
                if self.block_states[blk_ix].unfinished == 0 {
                    self.resident_warps = self
                        .resident_warps
                        .checked_sub(self.block_states[blk_ix].warp_count)
                        .ok_or(IssueError::CounterUnderflow {
                            counter: "resident warp",
                        })?;
                    self.resident_shared_memory_bytes = self
                        .resident_shared_memory_bytes
                        .checked_sub(self.block_states[blk_ix].shared_memory_bytes)
                        .ok_or(IssueError::CounterUnderflow {
                            counter: "resident shared-memory byte",
                        })?;
                    for &warp_slot in &self.block_states[blk_ix].warp_slots {
                        let removed = self.resident_warp_slots.remove(&warp_slot);
                        debug_assert!(
                            removed,
                            "retirement preflight proved every resident slot present"
                        );
                    }
                    let admit_at = checked_cycle_add("block-wave admission", t, 1)?;
                    self.admit_blocks(admit_at)?;
                }
            } else {
                let next_pc = self.warps[wix].insts[self.warps[wix].next].pc;
                let next_slot = lookup_dense(
                    &self.schedule_slots,
                    self.schedule,
                    next_pc,
                    launch_index,
                    warp_slot,
                )?;
                self.warps[wix].next_slot = Some(next_slot);
                self.warps[wix].ordinary_dependency_wait =
                    next_slot.ordinary_dependency_wait_barrier();
                self.warps[wix].current_since = t;
                self.warps[wix].cta_barrier_ready = 0;
                self.warps[wix].barrier_pipeline_ready = 0;
                self.warps[wix].branch_pipeline_ready = 0;
            }
            if !retires_warp
                && let PipelineClass::Branch {
                    next_issue_interval_cycles,
                } = pipeline_class
            {
                self.warps[wix].branch_pipeline_ready =
                    checked_cycle_add("branch-pipeline resolution", t, next_issue_interval_cycles)?;
            }
            // Single hold write: never hold a yielded or finished warp.
            self.hold[sc] = (!control.yield_flag && !self.warps[wix].done).then_some(wix);

            // Block barrier arrival AFTER stream bookkeeping: a finished
            // warp never waits.
            if slot_class == OpClass::BlockBarrier && !self.warps[wix].done {
                let arrival = barrier_arrival_times[sc].unwrap_or(t);
                self.warps[wix].at_block_barrier = true;
                self.warps[wix].block_barrier_dispatch = Some(t);
                self.warps[wix].block_barrier_arrival = Some(arrival);
                self.block_states[blk_ix]
                    .barrier_first_dispatch
                    .get_or_insert(t);
                self.block_states[blk_ix]
                    .barrier_first_arrival
                    .get_or_insert(arrival);
                self.block_states[blk_ix].waiting = checked_usize_counter_add(
                    "block barrier waiter",
                    self.block_states[blk_ix].waiting,
                    1,
                )?;
            }
            // Release check: every unfinished warp of the block waiting.
            // The block's warps are contiguous in `warps` (admission
            // pushes them in one run), so scan only that range.
            let bs = &self.block_states[blk_ix];
            if bs.waiting > 0 && bs.waiting == bs.unfinished {
                let (first, count) = (bs.first_warp, bs.admitted_warps);
                let mut latest_arrival = 0u64;
                let mut second_latest_arrival = 0u64;
                let mut latest_count = 0usize;
                for warp in &self.warps[first..first + count] {
                    if !warp.at_block_barrier {
                        continue;
                    }
                    let arrival = warp
                        .block_barrier_arrival
                        .expect("barrier waiter owns an arrival timestamp");
                    if arrival > latest_arrival {
                        second_latest_arrival = latest_arrival;
                        latest_arrival = arrival;
                        latest_count = 1;
                    } else if arrival == latest_arrival {
                        latest_count += 1;
                    } else if arrival > second_latest_arrival {
                        second_latest_arrival = arrival;
                    }
                }
                self.block_states[blk_ix].waiting = 0;
                self.block_states[blk_ix].barrier_generation = self.block_states[blk_ix]
                    .barrier_generation
                    .checked_add(1)
                    .expect("barrier generation was preflighted");
                self.block_states[blk_ix].barrier_first_dispatch = None;
                self.block_states[blk_ix].barrier_first_arrival = None;
                if let Some(issue) = self.barrier_issue {
                    self.block_states[blk_ix].barrier_rearm_ready = checked_cycle_add(
                        "CTA-barrier generation re-arm",
                        latest_arrival,
                        issue.generation_rearm_cycles,
                    )?;
                }
                let release_at = checked_cycle_add("block-barrier release", latest_arrival, 1)?;
                for w in &mut self.warps[first..first + count] {
                    if w.at_block_barrier {
                        let own_arrival = w
                            .block_barrier_arrival
                            .expect("barrier waiter owns an arrival timestamp");
                        let other_arrival = if count <= 1 {
                            None
                        } else if own_arrival == latest_arrival && latest_count == 1 {
                            Some(second_latest_arrival)
                        } else {
                            Some(latest_arrival)
                        };
                        w.at_block_barrier = false;
                        w.block_barrier_dispatch = None;
                        w.block_barrier_arrival = None;
                        if self.barrier_issue.is_some() {
                            w.barrier_pipeline_ready = checked_cycle_add(
                                "CTA-barrier own-arrival release",
                                own_arrival,
                                1,
                            )?;
                            w.cta_barrier_ready = other_arrival
                                .map(|arrival| {
                                    checked_cycle_add(
                                        "CTA-barrier other-participant rendezvous",
                                        arrival,
                                        1,
                                    )
                                })
                                .transpose()?
                                .unwrap_or(0);
                            debug_assert_eq!(
                                w.barrier_pipeline_ready.max(w.cta_barrier_ready),
                                release_at
                            );
                        } else {
                            w.barrier_pipeline_ready = 0;
                            w.cta_barrier_ready = release_at;
                        }
                    }
                }
            }
        }
        self.issued = next_issued;
        self.readiness = projected_readiness;
        self.memory_service_candidates = projected_memory_service_candidates;
        if let Some(plan) = memory_admission_feedback_plan {
            self.memory_admission_feedback
                .as_mut()
                .expect("feedback issue plan requires installed feedback state")
                .apply_issue(plan, t);
        }
        Ok(issued_this_cycle)
    }
}

/// Register a completion target that clears when `fill` completes. The
/// asynchronous-copy group already owns its pending source unit; an optional
/// ordinary barrier is counted here alongside ordinary load barriers.
fn add_fill_waiter(
    st: &mut SharedL2State,
    warps: &mut [WarpState<'_>],
    fill: FillKey,
    warp: usize,
    target: Option<FillTarget>,
) -> Result<(), IssueError> {
    let Some(target) = target else {
        return Ok(());
    };
    warps[warp].add_pending(warp, target.barrier(), 1)?;
    match st.fill_waiters.entry(fill) {
        Entry::Vacant(v) => {
            v.insert(Waiters::One(FillWaiter {
                warp,
                target,
                units: 1,
            }));
        }
        Entry::Occupied(mut o) => o.get_mut().add(warp, target)?,
    }
    Ok(())
}

fn preflight_line_fill_mask(st: &SharedL2State, sector_id: i64) -> Result<(), IssueError> {
    let line = sector_id >> 2;
    let fill_in_flight = st.l1.is_fill_in_flight(FillKey::Line(line));
    let has_mask = st.fill_masks.contains_key(&line);
    match (fill_in_flight, has_mask) {
        (true, false) => Err(IssueError::FillMaskMissing { line }),
        (false, true) => Err(IssueError::FillMaskWithoutFill { line }),
        _ => Ok(()),
    }
}

fn register_response(st: &mut SharedL2State, reply: ReplyTo) -> Result<(), IssueError> {
    if !st.outstanding_replies.insert(reply) {
        return Err(IssueError::ResponseTokenAlreadyOutstanding { reply });
    }
    Ok(())
}

enum SharedL1Access {
    Hit { ready: Option<u64> },
    MissRequest { fill: FillKey },
    MergedPending { fill: FillKey },
    Stalled,
}

fn access_shared_l1(
    l1: &mut L1Cache,
    origin: MemoryOrigin,
    sector_id: i64,
    now: u64,
    observe_ready: bool,
    mshr_floor: Option<NonZeroU32>,
) -> Result<SharedL1Access, L1AccessError> {
    if observe_ready {
        let access = match mshr_floor {
            Some(floor) => l1.access_eligible_ev_with_mshr_floor(origin, sector_id, now, floor)?,
            None => l1.access_eligible_ev(origin, sector_id, now)?,
        };
        return Ok(match access {
            EvAccess::Hit { ready } => SharedL1Access::Hit { ready: Some(ready) },
            EvAccess::MissRequest { fill } => SharedL1Access::MissRequest { fill },
            EvAccess::MergedPending { fill } => SharedL1Access::MergedPending { fill },
            EvAccess::Stalled => SharedL1Access::Stalled,
        });
    }
    let access = match mshr_floor {
        Some(floor) => {
            l1.access_eligible_ev_unobserved_with_mshr_floor(origin, sector_id, now, floor)?
        }
        None => l1.access_eligible_ev_unobserved(origin, sector_id, now)?,
    };
    Ok(match access {
        EvAccessUnobserved::Hit => SharedL1Access::Hit { ready: None },
        EvAccessUnobserved::MissRequest { fill } => SharedL1Access::MissRequest { fill },
        EvAccessUnobserved::MergedPending { fill } => SharedL1Access::MergedPending { fill },
        EvAccessUnobserved::Stalled => SharedL1Access::Stalled,
    })
}

#[derive(Debug)]
enum SharedAccessError {
    Stalled,
    Issue(IssueError),
}

impl From<IssueError> for SharedAccessError {
    fn from(source: IssueError) -> Self {
        Self::Issue(source)
    }
}

/// One eligible sector access in L1Shared mode at time `now`, emitting a
/// request on a fresh miss. Returns `Ok(known_ready)` for hits,
/// `Ok(None)` when the data waits on a response, and
/// `Err(SharedAccessError::Stalled)` on an MSHR stall (caller parks the
/// remainder). Accounting failures remain typed.
///
/// A sector merging onto a line fill emitted during the CURRENT
/// emission batch (`st.line_folds` — one `advance`/`on_response` call)
/// folds its bit into that request's `sector_mask`, so one L2
/// transaction carries the batch's sectors and worst-of-mask pricing
/// covers every warp/instruction/retry-entry of the batch. A merge onto
/// an OLDER in-flight fill emits a follow-up `ReplyTo::None` sector
/// fetch instead — the L2 must see every fetched sector exactly once
/// per fill (NCU sector accounting + L2 state fidelity), while the
/// waiter's timing stays bound to the original fill response (a
/// documented approximation: the follow-up's bytes may complete after
/// the waiter cleared, and same-timestamp queue order may process a
/// follow-up after its parent fill's response).
fn access_shared(
    st: &mut SharedL2State,
    warps: &mut [WarpState<'_>],
    wix: usize,
    target: Option<FillTarget>,
    mshr_floor: Option<NonZeroU32>,
    origin: MemoryOrigin,
    sector_id: i64,
    response_id: ResponseId,
    now: u64,
    out: &mut Vec<MemRequest>,
) -> Result<Option<u64>, SharedAccessError> {
    let sbit = 1u8 << (sector_id & 3) as u8;
    match access_shared_l1(
        &mut st.l1,
        origin,
        sector_id,
        now,
        target.is_some(),
        mshr_floor,
    )
    .map_err(IssueError::from)?
    {
        SharedL1Access::Hit { ready } => Ok(ready),
        SharedL1Access::MissRequest { fill } => {
            let request = MemRequest::l1_fill(now, sector_id, origin, fill, response_id);
            register_response(st, request.reply)?;
            out.push(request);
            if let FillKey::Line(line) = fill {
                st.fill_masks.insert(line, sbit);
                st.line_folds.push((line, out.len() - 1));
            }
            add_fill_waiter(st, warps, fill, wix, target)?;
            Ok(None)
        }
        SharedL1Access::MergedPending { fill } => {
            if let FillKey::Line(line) = fill {
                // Pairing invariant (proven at the 2026-07-07 gate): a
                // line-pending merge always has a live fill_masks entry
                // (created at MissRequest, removed at the response that
                // also completes the fill). Fail LOUD if a future
                // change breaks it — a fabricated empty mask would
                // silently re-fetch already-sent sectors.
                let sent = st
                    .fill_masks
                    .get_mut(&line)
                    .ok_or(SharedAccessError::Issue(IssueError::FillMaskMissing {
                        line,
                    }))?;
                if *sent & sbit == 0 {
                    *sent |= sbit;
                    if let Some(&(_, ix)) = st.line_folds.iter().find(|&&(l, _)| l == line) {
                        out[ix].fold_sector_origin(origin, sbit);
                    } else {
                        out.push(MemRequest::l1_followup_fetch(now, sector_id, origin));
                    }
                }
                // Re-touch of an already-sent sector: MSHR merge, no
                // new fetch (matches NCU's xbar sector accounting).
            }
            // Sector-fill merges need nothing: that sector's fetch was
            // sent with its own request.
            add_fill_waiter(st, warps, fill, wix, target)?;
            Ok(None)
        }
        SharedL1Access::Stalled => Err(SharedAccessError::Stalled),
    }
}

fn drain_retries(
    st: &mut SharedL2State,
    warps: &mut [WarpState<'_>],
    async_copies: &mut [AsyncCopySlot],
    now: u64,
    params: &IssueModelParams,
    out: &mut Vec<MemRequest>,
) -> Result<(), IssueError> {
    while let Some(mut r) = st.retry.pop_front() {
        let sectors = warps[r.warp].sectors;
        let barrier = r.target.and_then(FillTarget::barrier);
        // Replace the retry entry's ordinary-barrier unit atomically with
        // whatever waiter units this attempt creates. Async-copy source units
        // remain owned by their group until a hit or fill completion resolves
        // each sector.
        warps[r.warp].release_pending(r.warp, barrier, 1)?;
        // The whole drain is ONE emission batch (st.line_folds, cleared
        // by on_response): a line fill emitted by an earlier retry
        // entry folds later entries' same-line sectors — same-timestamp
        // requests must not split one line transaction (gate finding).
        while r.lo < r.hi {
            match access_shared(
                st,
                warps,
                r.warp,
                r.target,
                r.mshr_floor,
                r.origin,
                sectors[r.lo],
                ResponseId::new(st.response_owner, r.warp, r.instruction, r.lo),
                now,
                out,
            ) {
                Ok(Some(ready)) => {
                    if let Some(group) = r.target.and_then(FillTarget::async_group) {
                        async_copies[r.warp]
                            .complete_async_copy_sources(r.warp, group, None, 1, ready, params)?;
                    }
                    warps[r.warp].stamp_clear(barrier, ready, DependencyOrigin::L1Tex);
                }
                Ok(None) => {}
                Err(SharedAccessError::Stalled) => {
                    // Still full: restore the retry entry's replacement unit
                    // and park it; nothing behind it can proceed either.
                    warps[r.warp].add_pending(r.warp, barrier, 1)?;
                    st.retry.push_front(r);
                    return Ok(());
                }
                Err(SharedAccessError::Issue(source)) => return Err(source),
            }
            r.lo = r.lo.checked_add(1).ok_or(IssueError::CounterOverflow {
                counter: "retry sector cursor",
            })?;
        }
    }
    Ok(())
}

fn validate_sector_span(
    kind: AccessKind,
    sectors: usize,
    pc: u32,
    launch_index: i64,
    warp_slot: i32,
) -> Result<(), IssueError> {
    let atomic_topology = match kind {
        AccessKind::GlobalBypass {
            len,
            token: CacheOpToken::Atom,
            l2_requests,
            ..
        }
        | AccessKind::GlobalReduction {
            len, l2_requests, ..
        } => Some((len, l2_requests, true)),
        AccessKind::GlobalBypass {
            len, l2_requests, ..
        } if l2_requests != 0 => Some((len, l2_requests, false)),
        _ => None,
    };
    if let Some((sector_transactions, l2_requests, is_atomic)) = atomic_topology
        && (!is_atomic
            || (sector_transactions == 0) != (l2_requests == 0)
            || l2_requests > sector_transactions)
    {
        return Err(IssueError::AtomicRequestTopologyInvalid {
            pc,
            launch_index,
            warp_slot,
            sectors: sector_transactions,
            l2_requests,
        });
    }
    let (off, len) = match kind {
        AccessKind::GlobalEligible { off, len, .. }
        | AccessKind::GlobalBypass { off, len, .. }
        | AccessKind::GlobalAsyncCopy { off, len, .. }
        | AccessKind::GlobalUntagged { off, len }
        | AccessKind::GlobalStore { off, len }
        | AccessKind::GlobalReduction { off, len, .. }
        | AccessKind::LocalLoad { off, len }
        | AccessKind::LocalStore { off, len } => (off, len),
        AccessKind::NonMemory | AccessKind::PredicatedOffMemory | AccessKind::Shared { .. } => {
            return Ok(());
        }
    };
    let end = off
        .checked_add(usize::from(len))
        .ok_or(IssueError::SectorSpanOverflow {
            pc,
            launch_index,
            warp_slot,
            off,
            len,
        })?;
    if end > sectors {
        return Err(IssueError::SectorSpanOutOfBounds {
            pc,
            launch_index,
            warp_slot,
            off,
            len,
            end,
            sectors,
        });
    }
    Ok(())
}

fn lookup_dense<'a>(
    schedule_slots: &[(u64, &'a IssueSlot)],
    schedule: &FunctionSchedule,
    pc: u32,
    launch_index: i64,
    warp_slot: i32,
) -> Result<&'a IssueSlot, IssueError> {
    schedule_slots
        .binary_search_by_key(&u64::from(pc), |(slot_pc, _)| *slot_pc)
        .map(|index| schedule_slots[index].1)
        .map_err(|_| IssueError::PcNotInSchedule {
            pc,
            launch_index,
            warp_slot,
            function_name: schedule.function_name.clone(),
        })
}

/// Replay one SM's blocks to completion under the v0 issue model
/// (synchronous Flat/L1Const modes; the shared-L2 mode runs through
/// [`SmCore`] + the event system). `l1_config`, when supplied, attaches
/// a FRESH per-SM L1 (cold cache per kernel, NCU `--cache-control all`
/// parity).
pub(crate) fn simulate_sm_with_residency(
    blocks: &[BlockReplay],
    schedule: &FunctionSchedule,
    params: &IssueModelParams,
    shared_memory_residency: SharedMemoryResidency,
    l1_config: Option<L1Config>,
) -> Result<SmSimResult, IssueError> {
    let mode = match l1_config {
        Some(cfg) => SmMemMode::L1Const(cfg),
        None => SmMemMode::Flat,
    };
    let mut core =
        SmCore::new_with_residency(blocks, schedule, params, shared_memory_residency, mode)?;
    let mut out = Vec::new();
    match core.advance(ReplayHorizon::Unbounded, &mut out)? {
        Advance::Done => {}
        Advance::Idle { .. } => unreachable!("synchronous modes emit no requests"),
    }
    debug_assert!(out.is_empty(), "synchronous modes emit no requests");
    Ok(core.result())
}

#[cfg(test)]
fn simulate_sm(
    blocks: &[BlockReplay],
    schedule: &FunctionSchedule,
    params: &IssueModelParams,
    l1_config: Option<L1Config>,
) -> Result<SmSimResult, IssueError> {
    simulate_sm_with_residency(
        blocks,
        schedule,
        params,
        SharedMemoryResidency::unconstrained_fixture(),
        l1_config,
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use sim_mem::{AccessOutcome, CacheOpToken, L1CounterError, L1CounterField};

    use super::*;
    use crate::control_bits::{ControlBits, DepBarrier, WaitMask};
    use crate::program::{
        FunctionSchedule, IssueSlot, OpClass, PipelineClass, is_clocklo_opcode_text,
        is_gpu_membar_opcode_text,
    };

    fn cb(stall: u8) -> ControlBits {
        ControlBits::new(stall, false, 7, 7, 0, 0).expect("valid test controls")
    }

    fn barrier(value: u8) -> DepBarrier {
        DepBarrier::new(value).expect("valid test barrier")
    }

    fn wait_mask(bits: u8) -> WaitMask {
        WaitMask::new(bits).expect("valid test wait mask")
    }

    fn slot(control: ControlBits, op_class: OpClass, text: &str) -> IssueSlot {
        IssueSlot {
            control,
            op_class,
            async_copy_wait_max_pending_groups: 0,
            pipeline_class: PipelineClass::DispatchOnly,
            mem_role: if text.trim_start().starts_with("LDGSTS") {
                sim_mem::MemRole::BypassLoad(CacheOpToken::Ldgsts)
            } else {
                sim_mem::MemRole::None
            },
            ldgsts_l1_policy: text.trim_start().starts_with("LDGSTS").then_some(
                if text.split('.').any(|modifier| modifier == "BYPASS") {
                    LdgstsL1Policy::BypassL1
                } else {
                    LdgstsL1Policy::CacheAtAllLevels
                },
            ),
            fixed_latency_operands: crate::sass_operands::project_fixed_latency_operands(
                text,
                control,
                ArchId::Sm86,
            ),
            clocklo_read: is_clocklo_opcode_text(text),
            gpu_membar: is_gpu_membar_opcode_text(text),
            opcode_text: text.to_owned(),
        }
    }

    fn barrier_pipeline_slot(control: ControlBits) -> IssueSlot {
        let mut result = slot(control, OpClass::BlockBarrier, "BAR.SYNC.DEFER_BLOCKING");
        result.pipeline_class = PipelineClass::Barrier {
            warp_initiation_interval_cycles: 2,
            generation_rearm_cycles: 0,
        };
        result
    }

    fn branch_pipeline_slot(control: ControlBits) -> IssueSlot {
        let mut result = slot(control, OpClass::Other, "BRA 0x20");
        result.pipeline_class = PipelineClass::Branch {
            next_issue_interval_cycles: 10,
        };
        result
    }

    fn tensor_slot(control: ControlBits, text: &str, interval: u64) -> IssueSlot {
        IssueSlot {
            pipeline_class: PipelineClass::Tensor {
                initiation_interval_cycles: interval,
            },
            ..slot(control, OpClass::Other, text)
        }
    }

    fn schedule(slots: Vec<(u64, IssueSlot)>) -> FunctionSchedule {
        FunctionSchedule {
            arch: ArchId::Sm86,
            function_unique_id: 1,
            kernel_name: "k___0".to_owned(),
            function_name: "k".to_owned(),
            by_pc: slots.into_iter().collect::<BTreeMap<_, _>>(),
        }
    }

    #[test]
    fn replay_instruction_stays_compact() {
        assert!(std::mem::size_of::<ReplayInst>() <= 24);
    }

    #[test]
    fn disabled_candidate_storage_stays_off_the_hot_inline_path() {
        assert!(std::mem::size_of::<MemoryServiceCandidateDiagnostics>() <= 24);
        assert!(std::mem::size_of::<MemoryServiceCandidateState>() <= 24);
        assert!(std::mem::size_of::<MemoryAdmissionCandidateDiagnostics>() <= 24);
        assert!(std::mem::size_of::<MemoryAdmissionCandidateState>() <= 24);
        assert!(std::mem::size_of::<ReadinessDiagnostics>() <= 1024);
        assert!(std::mem::size_of::<SmCore<'static>>() <= 3072);
        assert!(
            std::mem::size_of::<MemRequest>() <= 96,
            "MemRequest grew to {} bytes",
            std::mem::size_of::<MemRequest>()
        );
    }

    fn params() -> IssueModelParams {
        IssueModelParams {
            arch: ArchId::Sm86,
            resource_source: IssueResourceSource::SyntheticTestFixture,
            l1_hit_latency: 34,
            l1_miss_latency: 244,
            global_flat_latency: 400,
            shared_write_barrier_latency: 23,
            shared_additional_wavefront_latency: 2,
            non_memory_write_barrier_latency: 25,
            read_barrier_latency: 8,
            ldgsts_l1_mshr_floor: NonZeroU32::new(1).unwrap(),
            ldgsts_completion_throughput: LdgstsCompletionThroughput {
                cache_all_payload_bytes_per_cycle: 64,
                bypass_payload_bytes_per_cycle: 32,
            },
            ldgsts_cache_all_destination_wavefront_cycles: 2,
            ldgsts_cache_all_destination_service_wavefronts: 2,
            ldgsts_cache_all_destination_service_cycles: 3,
            max_resident_warps_per_sm: 48,
        }
    }

    fn card_path(name: &str) -> PathBuf {
        PathBuf::from("../../arch_profiles/cards")
            .join(name)
            .canonicalize()
            .expect("card exists in-repo")
    }

    fn synthetic_carded_issue_resources() -> ArchCard {
        let path = card_path("sm_86.json");
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        // Fixture-only sentinels. Matching legacy values exercises lane
        // separation; it does not turn them into architecture-card values.
        value["smsp"] = serde_json::json!({"count": 4, "issue_ports": 1});
        value["occupancy"] = serde_json::json!({
            "max_resident_warps_per_sm": 48,
            "max_blocks": 16,
        });
        value["dep_barriers"] = serde_json::json!({"count": 6});
        value["barrier_latencies"] = serde_json::json!({
            "read": 8,
            "non_memory_write": 25,
        });
        value["dual_issue"] = serde_json::json!({"supported": false});
        ArchCard::parse(&path, &serde_json::to_vec(&value).unwrap()).unwrap()
    }

    #[test]
    fn legacy_issue_lane_is_explicit_and_sm86_only() {
        let legacy = IssueModelParams::from_legacy_card_latencies_for_arch(
            ArchId::Sm86,
            34.0,
            244.0,
            423.0,
            24.0,
        )
        .unwrap();
        assert_eq!(
            legacy.resource_source(),
            IssueResourceSource::ArchitectureDefaults
        );
        assert_eq!(legacy.non_memory_write_barrier_latency, 25);
        assert_eq!(legacy.read_barrier_latency, 8);
        assert_eq!(legacy.ldgsts_l1_mshr_floor.get(), 248);
        assert_eq!(legacy.max_resident_warps_per_sm, 48);
        legacy.validate_resource_source().unwrap();

        assert!(matches!(
            IssueModelParams::from_legacy_card_latencies_for_arch(
                ArchId::Sm120,
                34.0,
                244.0,
                423.0,
                24.0,
            ),
            Err(IssueError::Capability(
                sim_arch::CapabilityError::ResearchGap {
                    arch: ArchId::Sm120,
                    ..
                }
            ))
        ));
    }

    #[test]
    fn mutated_legacy_hypotheses_cannot_retain_provenance_or_execute() {
        let baseline = IssueModelParams::from_legacy_card_latencies_for_arch(
            ArchId::Sm86,
            34.0,
            244.0,
            423.0,
            24.0,
        )
        .unwrap();
        let mut non_memory = baseline;
        non_memory.non_memory_write_barrier_latency += 1;
        let mut read = baseline;
        read.read_barrier_latency += 1;
        let mut ldgsts_floor = baseline;
        ldgsts_floor.ldgsts_l1_mshr_floor = NonZeroU32::new(249).unwrap();
        let mut occupancy = baseline;
        occupancy.max_resident_warps_per_sm += 1;

        for (field, params, expected, got) in [
            (
                "non_memory_write_barrier_latency",
                non_memory,
                25_u64,
                26_u64,
            ),
            ("read_barrier_latency", read, 8_u64, 9_u64),
            ("ldgsts_l1_mshr_floor", ldgsts_floor, 248_u64, 249_u64),
            ("max_resident_warps_per_sm", occupancy, 48_u64, 49_u64),
        ] {
            assert!(matches!(
                params.validate_resource_source(),
                Err(IssueError::ArchitectureDefaultMismatch {
                    arch: ArchId::Sm86,
                    field: got_field,
                    expected: got_expected,
                    got: got_value,
                }) if got_field == field && got_expected == expected && got_value == got
            ));
        }

        assert!(matches!(
            crate::replay::ReplayConfiguration::checked(
                non_memory,
                crate::replay::ReplayMemMode::Flat,
            ),
            Err(crate::replay::ReplayError::Issue(
                IssueError::ArchitectureDefaultMismatch {
                    arch: ArchId::Sm86,
                    field: "non_memory_write_barrier_latency",
                    expected: 25,
                    got: 26,
                }
            ))
        ));
    }

    #[test]
    fn strict_card_constructor_requires_same_architecture_and_all_resources() {
        let production_sm86 = ArchCard::load(&card_path("sm_86.json")).unwrap();
        assert!(matches!(
            IssueModelParams::from_card(ArchId::Sm86, &production_sm86),
            Err(IssueError::Calibration(CalibError::BlockResearchGap {
                block: "smsp",
                ..
            }))
        ));

        let production_sm120 = ArchCard::load(&card_path("sm_120-laptop.json")).unwrap();
        assert!(matches!(
            IssueModelParams::from_card(ArchId::Sm86, &production_sm120),
            Err(IssueError::ArchitectureMismatch {
                component: "strict issue architecture card",
                expected: ArchId::Sm86,
                got: ArchId::Sm120,
            })
        ));
    }

    #[test]
    fn complete_strict_card_lane_constructs_but_cannot_execute() {
        let strict =
            IssueModelParams::from_card(ArchId::Sm86, &synthetic_carded_issue_resources()).unwrap();
        let IssueResourceSource::Carded(resources) = strict.resource_source() else {
            panic!("strict constructor must preserve its carded lane")
        };
        assert_eq!(resources.smsp.count, 4);
        assert_eq!(resources.smsp.issue_ports, 1);
        assert_eq!(resources.occupancy.max_resident_warps_per_sm, 48);
        assert_eq!(resources.occupancy.max_blocks, 16);
        assert_eq!(resources.dep_barriers.count, 6);
        assert_eq!(resources.barrier_latencies.read, 8);
        assert_eq!(resources.barrier_latencies.non_memory_write, 25);
        assert!(!resources.dual_issue_supported);
        assert_eq!(strict.non_memory_write_barrier_latency, 25);
        assert_eq!(strict.read_barrier_latency, 8);
        assert_eq!(strict.ldgsts_l1_mshr_floor.get(), 248);
        assert_eq!(strict.max_resident_warps_per_sm, 48);

        assert!(matches!(
            strict.validate_resource_source(),
            Err(IssueError::CardedResourcesNotImplemented { arch: ArchId::Sm86 })
        ));
        assert!(matches!(
            crate::replay::ReplayConfiguration::checked(strict, crate::replay::ReplayMemMode::Flat,),
            Err(crate::replay::ReplayError::Issue(
                IssueError::CardedResourcesNotImplemented { arch: ArchId::Sm86 }
            ))
        ));

        let sched = schedule(Vec::new());
        assert!(matches!(
            SmCore::new(&[], &sched, &strict, SmMemMode::Flat),
            Err(IssueError::CardedResourcesNotImplemented { arch: ArchId::Sm86 })
        ));
    }

    #[test]
    fn latency_conversion_rejects_u64_boundary_before_cast() {
        assert!(checked_latency("boundary", u64::MAX as f64).is_err());
        assert!(checked_latency("infinite", f64::INFINITY).is_err());
        assert!(checked_latency("negative", -1.0).is_err());
        assert_eq!(checked_latency("fractional", 12.01).unwrap(), 13);
    }

    #[test]
    fn maximum_response_owner_is_allocated_before_exhaustion() {
        let next = AtomicU64::new(u64::MAX);
        assert_eq!(
            allocate_response_owner_from(&next).unwrap(),
            ResponseOwner(u64::MAX)
        );
        assert_eq!(next.load(Ordering::Relaxed), 0);
        assert!(matches!(
            allocate_response_owner_from(&next),
            Err(IssueError::ResponseOwnerExhausted)
        ));
        assert_eq!(next.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn past_horizon_is_rejected_before_mutation() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0)])],
        }];
        let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
        core.t = 1;
        let mut out = Vec::new();
        assert!(matches!(
            core.advance(ReplayHorizon::Bounded(0), &mut out),
            Err(IssueError::HorizonBeforeClock {
                horizon: 0,
                clock: 1,
            })
        ));
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), 0);
        assert_eq!(core.warps[0].next, 0);
    }

    fn construction_error(kind: AccessKind, sectors: Vec<i64>) -> IssueError {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDG.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 7,
            warps: vec![WarpReplay {
                warp_slot: 3,
                insts: vec![ReplayInst { pc: 0, kind }],
                sectors,
            }],
        }];
        let parameters = params();
        match SmCore::new(&blocks, &sched, &parameters, SmMemMode::Flat) {
            Err(error) => error,
            Ok(_) => panic!("invalid sector span constructed replay state"),
        }
    }

    #[test]
    fn every_memory_sector_span_is_validated_before_core_construction() {
        let kinds = [
            AccessKind::GlobalEligible {
                off: 1,
                len: 1,
                token: CacheOpToken::LdgECa,
            },
            AccessKind::GlobalBypass {
                off: 1,
                len: 1,
                token: CacheOpToken::LdgECg,
                l2_requests: 0,
            },
            AccessKind::GlobalUntagged { off: 1, len: 1 },
            AccessKind::GlobalStore { off: 1, len: 1 },
        ];
        for kind in kinds {
            assert!(matches!(
                construction_error(kind, vec![40]),
                IssueError::SectorSpanOutOfBounds {
                    pc: 0,
                    launch_index: 7,
                    warp_slot: 3,
                    off: 1,
                    len: 1,
                    end: 2,
                    sectors: 1,
                }
            ));
        }
    }

    #[test]
    fn atomic_l2_request_topology_is_validated_before_core_construction() {
        for kind in [
            AccessKind::GlobalBypass {
                off: 0,
                len: 1,
                token: CacheOpToken::Atom,
                l2_requests: 0,
            },
            AccessKind::GlobalBypass {
                off: 0,
                len: 1,
                token: CacheOpToken::LdgECg,
                l2_requests: 1,
            },
            AccessKind::GlobalReduction {
                off: 0,
                len: 1,
                l2_requests: 2,
            },
        ] {
            assert!(matches!(
                construction_error(kind, vec![40]),
                IssueError::AtomicRequestTopologyInvalid {
                    pc: 0,
                    launch_index: 7,
                    warp_slot: 3,
                    ..
                }
            ));
        }
    }

    #[test]
    fn sector_span_overflow_and_empty_out_of_bounds_are_typed() {
        assert!(matches!(
            construction_error(
                AccessKind::GlobalStore {
                    off: usize::MAX,
                    len: 1,
                },
                Vec::new(),
            ),
            IssueError::SectorSpanOverflow {
                pc: 0,
                launch_index: 7,
                warp_slot: 3,
                off: usize::MAX,
                len: 1,
            }
        ));
        assert!(matches!(
            construction_error(AccessKind::GlobalUntagged { off: 1, len: 0 }, Vec::new(),),
            IssueError::SectorSpanOutOfBounds {
                off: 1,
                len: 0,
                end: 1,
                sectors: 0,
                ..
            }
        ));
    }

    #[test]
    fn empty_sector_span_at_arena_end_is_valid() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LD.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: AccessKind::GlobalUntagged { off: 0, len: 0 },
                }],
                sectors: Vec::new(),
            }],
        }];
        SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat)
            .expect("empty span at the arena end is valid");
    }

    #[test]
    fn invalid_public_control_fields_fail_before_core_construction() {
        let blocks = vec![BlockReplay {
            launch_index: 7,
            warps: vec![warp(3, vec![inst(0)])],
        }];

        let mut invalid_stall = cb(1);
        invalid_stall.stall = 16;
        let stall_schedule = schedule(vec![(0, slot(invalid_stall, OpClass::Other, "IADD3"))]);
        let Err(stall_error) = SmCore::new(&blocks, &stall_schedule, &params(), SmMemMode::Flat)
        else {
            panic!("invalid stall constructed replay state");
        };
        assert!(matches!(
            stall_error,
            IssueError::ControlDomain {
                pc: 0,
                launch_index: 7,
                warp_slot: 3,
                source: ControlDomainError::StallOutOfRange { stall: 16 },
            }
        ));

        let mut invalid_reuse = cb(1);
        invalid_reuse.reuse = 16;
        let reuse_schedule = schedule(vec![(0, slot(invalid_reuse, OpClass::Other, "IADD3"))]);
        let Err(reuse_error) = SmCore::new(&blocks, &reuse_schedule, &params(), SmMemMode::Flat)
        else {
            panic!("invalid reuse mask constructed replay state");
        };
        assert!(matches!(
            reuse_error,
            IssueError::ControlDomain {
                pc: 0,
                launch_index: 7,
                warp_slot: 3,
                source: ControlDomainError::ReuseOutOfRange { reuse: 16 },
            }
        ));
    }

    #[test]
    fn producer_latency_overflow_is_typed() {
        let mut control = cb(1);
        control.write_barrier = Some(barrier(0));
        let sched = schedule(vec![(0, slot(control, OpClass::Other, "EXIT"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0)])],
        }];
        let mut p = params();
        p.non_memory_write_barrier_latency = u64::MAX;
        let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
        core.t = 1;
        let mut out = Vec::new();
        assert!(matches!(
            core.advance(ReplayHorizon::Unbounded, &mut out),
            Err(IssueError::TimeOverflow {
                operation: "non-memory write-barrier visibility",
                base: 1,
                delta: u64::MAX,
            })
        ));
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), 0);
        assert_eq!(core.warps[0].next, 0);
    }

    #[test]
    fn absent_dependency_barriers_skip_unused_visibility_timing() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDS"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(
                0,
                vec![ReplayInst {
                    pc: 0,
                    kind: AccessKind::Shared { wavefronts: 1 },
                }],
            )],
        }];
        let mut p = params();
        p.shared_write_barrier_latency = u64::MAX;
        p.read_barrier_latency = u64::MAX;
        let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
        core.t = u64::MAX - 1;
        let mut out = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Done
        );
        assert!(out.is_empty());
        assert_eq!(core.result().cycles, u64::MAX);
        assert_eq!(core.warps[0].barrier_clear, [0; DEP_BARRIERS]);
    }

    #[test]
    fn present_read_barrier_overflow_remains_typed_and_nonmutating() {
        let mut control = cb(1);
        control.read_barrier = Some(barrier(0));
        let sched = schedule(vec![(0, slot(control, OpClass::Other, "LDS"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(
                0,
                vec![ReplayInst {
                    pc: 0,
                    kind: AccessKind::Shared { wavefronts: 1 },
                }],
            )],
        }];
        let mut p = params();
        p.read_barrier_latency = 2;
        let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
        core.t = u64::MAX - 1;
        let mut out = Vec::new();

        assert!(matches!(
            core.advance(ReplayHorizon::Unbounded, &mut out),
            Err(IssueError::TimeOverflow {
                operation: "read-dependence barrier visibility",
                base,
                delta: 2,
            }) if base == u64::MAX - 1
        ));
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), 0);
        assert_eq!(core.warps[0].next, 0);
        assert_eq!(core.warps[0].barrier_clear, [0; DEP_BARRIERS]);
    }

    #[test]
    fn warm_l1_hit_ignores_an_overflowing_unused_miss_arm() {
        let mut control = cb(1);
        control.write_barrier = Some(barrier(0));
        let sched = schedule(vec![(0, slot(control, OpClass::Other, "LDG.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: eligible(0, 1),
                }],
                sectors: vec![0],
            }],
        }];
        let mut p = params();
        p.l1_hit_latency = 1;
        p.l1_miss_latency = 2;

        for shared in [false, true] {
            let mode = if shared {
                SmMemMode::L1Shared(p.l1_config(8, 1))
            } else {
                SmMemMode::L1Const(p.l1_config(8, 1))
            };
            let mut core = SmCore::new(&blocks, &sched, &p, mode).unwrap();
            match &mut core.mem {
                MemModel::L1Const(l1) => {
                    assert!(matches!(
                        l1.access_eligible(MemoryOrigin::Global, 0, 0).unwrap(),
                        (AccessOutcome::MissLineAlloc, 2)
                    ));
                    assert!(matches!(
                        l1.access_eligible(MemoryOrigin::Global, 0, 2).unwrap(),
                        (AccessOutcome::Hit, 3)
                    ));
                }
                MemModel::L1Shared(st) => {
                    let EvAccess::MissRequest { fill } = st
                        .l1
                        .access_eligible_ev(MemoryOrigin::Global, 0, 0)
                        .unwrap()
                    else {
                        panic!("expected cold prefill miss");
                    };
                    st.l1.complete_fill(fill).unwrap();
                }
                MemModel::Flat => panic!("expected an L1 mode"),
            }
            core.t = u64::MAX - 1;
            let mut out = Vec::new();

            assert_eq!(
                core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
                Advance::Done
            );
            assert!(out.is_empty());
            assert_eq!(core.result().cycles, u64::MAX);
            assert_eq!(core.warps[0].barrier_clear[0], u64::MAX);
        }
    }

    #[test]
    fn unbarriered_warm_l1_hit_skips_unused_hit_visibility() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDG.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: eligible(0, 1),
                }],
                sectors: vec![0],
            }],
        }];
        let mut p = params();
        p.l1_hit_latency = 2;
        p.l1_miss_latency = 3;

        for shared in [false, true] {
            let mode = if shared {
                SmMemMode::L1Shared(p.l1_config(8, 1))
            } else {
                SmMemMode::L1Const(p.l1_config(8, 1))
            };
            let mut core = SmCore::new(&blocks, &sched, &p, mode).unwrap();
            match &mut core.mem {
                MemModel::L1Const(l1) => {
                    assert!(matches!(
                        l1.access_eligible(MemoryOrigin::Global, 0, 0).unwrap(),
                        (AccessOutcome::MissLineAlloc, 3)
                    ));
                    assert!(matches!(
                        l1.access_eligible(MemoryOrigin::Global, 0, 3).unwrap(),
                        (AccessOutcome::Hit, 5)
                    ));
                }
                MemModel::L1Shared(st) => {
                    let EvAccess::MissRequest { fill } = st
                        .l1
                        .access_eligible_ev(MemoryOrigin::Global, 0, 0)
                        .unwrap()
                    else {
                        panic!("expected cold prefill miss");
                    };
                    st.l1.complete_fill(fill).unwrap();
                }
                MemModel::Flat => panic!("expected an L1 mode"),
            }
            core.t = u64::MAX - 1;
            let mut out = Vec::new();

            assert_eq!(
                core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
                Advance::Done
            );
            assert!(out.is_empty());
            assert_eq!(core.result().cycles, u64::MAX);
            assert_eq!(core.warps[0].barrier_clear, [0; DEP_BARRIERS]);
        }
    }

    #[test]
    fn cold_shared_l1_miss_overflow_remains_typed_and_nonmutating() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDG.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: eligible(0, 1),
                }],
                sectors: vec![0],
            }],
        }];
        let mut p = params();
        p.l1_hit_latency = 1;
        p.l1_miss_latency = 2;
        let mut core =
            SmCore::new(&blocks, &sched, &p, SmMemMode::L1Shared(p.l1_config(8, 1))).unwrap();
        core.t = u64::MAX - 1;
        let before = *core.l1_counters().unwrap();
        let mut out = Vec::new();

        assert!(matches!(
            core.advance(ReplayHorizon::Unbounded, &mut out),
            Err(IssueError::TimeOverflow {
                operation: "self-response horizon",
                base,
                delta: 2,
            }) if base == u64::MAX - 1
        ));
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), 0);
        assert_eq!(core.warps[0].next, 0);
        assert_eq!(core.outstanding_responses(), 0);
        assert_eq!(*core.l1_counters().unwrap(), before);
    }

    #[test]
    fn exact_maximum_barrier_time_is_valid() {
        let mut control = cb(1);
        control.write_barrier = Some(barrier(0));
        let sched = schedule(vec![(0, slot(control, OpClass::Other, "EXIT"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0)])],
        }];
        let mut p = params();
        p.non_memory_write_barrier_latency = u64::MAX - 1;
        let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
        core.t = 1;
        let mut out = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Done
        );
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), 1);
        assert_eq!(core.warps[0].barrier_clear[0], u64::MAX);
        assert_eq!(core.result().cycles, 2);
    }

    #[test]
    fn final_issue_can_advance_the_clock_to_exact_maximum() {
        for (stall, yield_flag) in [(15, false), (1, true)] {
            let mut control = cb(stall);
            control.yield_flag = yield_flag;
            let sched = schedule(vec![(0, slot(control, OpClass::Other, "EXIT"))]);
            let blocks = vec![BlockReplay {
                launch_index: 0,
                warps: vec![warp(0, vec![inst(0)])],
            }];
            let mut p = params();
            p.non_memory_write_barrier_latency = 1;
            p.read_barrier_latency = 1;
            let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
            core.t = u64::MAX - 1;
            let mut out = Vec::new();

            assert_eq!(
                core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
                Advance::Done
            );
            assert!(out.is_empty());
            assert_eq!(core.result().instructions_issued, 1);
            assert_eq!(core.result().cycles, u64::MAX);
        }
    }

    #[test]
    fn nonterminal_issue_still_requires_post_issue_eligibility() {
        for (stall, yield_flag, expected_operation, expected_delta) in [
            (15, false, "warp stall eligibility", 15),
            (1, true, "warp yield eligibility", 2),
        ] {
            let mut control = cb(stall);
            control.yield_flag = yield_flag;
            let sched = schedule(vec![
                (0, slot(control, OpClass::Other, "IADD3")),
                (16, slot(cb(1), OpClass::Other, "EXIT")),
            ]);
            let blocks = vec![BlockReplay {
                launch_index: 0,
                warps: vec![warp(0, vec![inst(0), inst(16)])],
            }];
            let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
            core.t = u64::MAX - 1;
            let mut out = Vec::new();

            assert!(matches!(
                core.advance(ReplayHorizon::Unbounded, &mut out),
                Err(IssueError::TimeOverflow {
                    operation,
                    base,
                    delta,
                }) if operation == expected_operation
                    && base == u64::MAX - 1
                    && delta == expected_delta
            ));
            assert!(out.is_empty());
            assert_eq!(core.instructions_issued(), 0);
            assert_eq!(core.warps[0].next, 0);
        }
    }

    #[test]
    fn maximum_bounded_horizon_is_distinct_from_unbounded_replay() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0)])],
        }];
        let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
        core.t = u64::MAX;
        let mut out = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Bounded(u64::MAX), &mut out)
                .unwrap(),
            Advance::Idle {
                wake_at: Some(u64::MAX),
            }
        );
        assert!(matches!(
            core.advance(ReplayHorizon::Unbounded, &mut out),
            Err(IssueError::TimeOverflow {
                operation: "issue-clock advance",
                base: u64::MAX,
                delta: 1,
            })
        ));
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), 0);
        assert_eq!(core.warps[0].next, 0);
    }

    #[test]
    fn self_response_horizon_overflow_is_typed() {
        let mut control = cb(1);
        control.write_barrier = Some(barrier(0));
        let sched = schedule(vec![(0, slot(control, OpClass::Other, "LDG.E.CG"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: AccessKind::GlobalBypass {
                        off: 0,
                        len: 1,
                        token: CacheOpToken::LdgECg,
                        l2_requests: 0,
                    },
                }],
                sectors: vec![0],
            }],
        }];
        let mut p = params();
        p.l1_miss_latency = u64::MAX;
        let mut core = SmCore::new(
            &blocks,
            &sched,
            &p,
            SmMemMode::L1Shared(p.l1_config(768, 32)),
        )
        .unwrap();
        core.t = 1;
        let mut out = Vec::new();
        assert!(matches!(
            core.advance(ReplayHorizon::Unbounded, &mut out),
            Err(IssueError::TimeOverflow {
                operation: "self-response horizon",
                base: 1,
                delta: u64::MAX,
            })
        ));
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), 0);
        assert_eq!(core.warps[0].next, 0);
        assert_eq!(core.outstanding_responses(), 0);
        assert_eq!(core.l1_counters().unwrap().bypass_sectors, 0);
    }

    #[test]
    fn l1_timing_must_match_issue_parameters() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0)])],
        }];
        let p = params();
        let mut cfg = p.l1_config(768, 32);
        cfg.miss_latency += 1;
        assert!(matches!(
            SmCore::new(&blocks, &sched, &p, SmMemMode::L1Shared(cfg)),
            Err(IssueError::L1TimingMismatch {
                field: "miss",
                expected: 244,
                got: 245,
            })
        ));
    }

    #[test]
    fn constant_cacheable_async_copy_uses_access_local_mshr_floor() {
        let mut p = params();
        p.ldgsts_l1_mshr_floor = NonZeroU32::new(2).unwrap();
        let sectors = [0, 4];

        let mut cacheable = L1Cache::new(p.l1_config(8, 1)).unwrap();
        assert_eq!(
            data_ready_time(
                async_copy_inst(0, 0, 2).kind,
                &sectors,
                0,
                &p,
                Some(&mut cacheable),
                None,
                true,
            )
            .unwrap(),
            Some(244)
        );
        assert_eq!(cacheable.counters.stall_events, 0);

        let mut ordinary = L1Cache::new(p.l1_config(8, 1)).unwrap();
        assert_eq!(
            data_ready_time(
                eligible(0, 2),
                &sectors,
                0,
                &p,
                Some(&mut ordinary),
                Some(barrier(0)),
                false,
            )
            .unwrap(),
            Some(488)
        );
        assert_eq!(ordinary.counters.stall_events, 1);
    }

    fn l1_config() -> L1Config {
        L1Config {
            arch: ArchId::Sm86,
            line_capacity: 768,
            mshr_count: Some(32),
            hit_latency: 34,
            miss_latency: 244,
        }
    }

    fn eligible(off: usize, len: u16) -> AccessKind {
        AccessKind::GlobalEligible {
            off,
            len,
            token: CacheOpToken::LdgECa,
        }
    }

    fn inst(pc: u32) -> ReplayInst {
        ReplayInst {
            pc,
            kind: AccessKind::NonMemory,
        }
    }

    fn async_copy_inst_with_policy(
        pc: u32,
        off: usize,
        len: u16,
        l1_policy: LdgstsL1Policy,
    ) -> ReplayInst {
        let payload_bytes = u32::from(len) * 32;
        let completion_work = match l1_policy {
            LdgstsL1Policy::CacheAtAllLevels => payload_bytes,
            LdgstsL1Policy::BypassL1 => payload_bytes * 2,
        };
        ReplayInst {
            pc,
            kind: AccessKind::GlobalAsyncCopy {
                off,
                len,
                completion_work,
                l1_policy,
            },
        }
    }

    fn async_copy_inst(pc: u32, off: usize, len: u16) -> ReplayInst {
        async_copy_inst_with_policy(pc, off, len, LdgstsL1Policy::CacheAtAllLevels)
    }

    fn bypass_async_copy_inst(pc: u32, off: usize, len: u16) -> ReplayInst {
        async_copy_inst_with_policy(pc, off, len, LdgstsL1Policy::BypassL1)
    }

    fn async_commit_slot() -> IssueSlot {
        slot(
            ControlBits::new(1, true, 0, 7, 0, 0).unwrap(),
            OpClass::AsyncCopyCommit,
            "LDGDEPBAR",
        )
    }

    fn async_wait_slot(max_pending_groups: u32) -> IssueSlot {
        IssueSlot {
            async_copy_wait_max_pending_groups: max_pending_groups,
            ..slot(
                cb(1),
                OpClass::AsyncCopyWait,
                &format!("DEPBAR.LE SB0, {max_pending_groups}"),
            )
        }
    }

    fn warp(slot_ix: i32, insts: Vec<ReplayInst>) -> WarpReplay {
        WarpReplay {
            warp_slot: slot_ix,
            insts,
            sectors: Vec::new(),
        }
    }

    #[test]
    fn single_warp_stall_timing() {
        // Three instructions, stalls 1,1,1: issues at t=0,1,2 → cycles 3.
        let sched = schedule(vec![
            (0, slot(cb(1), OpClass::Other, "IADD3")),
            (16, slot(cb(1), OpClass::Other, "IADD3")),
            (32, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0), inst(16), inst(32)])],
        }];
        let r = simulate_sm(&blocks, &sched, &params(), None).unwrap();
        assert_eq!(r.instructions_issued, 3);
        assert_eq!(r.cycles, 3);
        assert!(r.l1.is_none());
        assert_eq!(r.readiness.limiting_cycles.control_stall, 2);
        assert_eq!(r.readiness.constraint_exposure_cycles.control_stall, 2);
        assert_eq!(r.readiness.delayed_ready_issues, 0);
    }

    #[test]
    fn phase_opcode_recognition_uses_the_static_sass_ascii_grammar() {
        let ascii_clock = slot(cb(1), OpClass::Other, "\t CS2R R4, SR_CLOCKLO \r\n");
        let wrong_source = slot(cb(1), OpClass::Other, "CS2R R4, SR_CLOCKLO64");
        assert!(is_clocklo_issue(&ascii_clock));
        assert!(!is_clocklo_issue(&wrong_source));

        let ascii_membar = slot(cb(1), OpClass::Other, "\tMEMBAR.SC.GPU ;");
        assert!(is_gpu_membar_issue(&ascii_membar));
    }

    #[test]
    fn clocklo_issue_interval_matches_the_bracketed_scheduler_region() {
        let sched = schedule(vec![
            (0, slot(cb(2), OpClass::Other, "CS2R R4, SR_CLOCKLO")),
            (16, slot(cb(3), OpClass::Other, "CS2R R8, SRZ")),
            (32, slot(cb(1), OpClass::Other, "CS2R R6, SR_CLOCKLO")),
            (48, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0), inst(16), inst(32), inst(48)])],
        }];
        let result = simulate_sm(&blocks, &sched, &params(), None).unwrap();
        assert_eq!(result.readiness.clock_issue.reads_issued, 2);
        assert_eq!(result.readiness.clock_issue.warps_with_interval, 1);
        assert_eq!(
            result
                .readiness
                .clock_issue
                .maximum_first_to_last_issue_cycles,
            5
        );
        assert_eq!(result.readiness.schema_version(), 5);
    }

    fn result_priority_case(load_destination: u16, producer_cycle: u8) -> SmSimResult {
        let mut load_control = ControlBits::new(15, false, 0, 7, 0, 0).unwrap();
        load_control.write_barrier = Some(barrier(0));
        let mut consumer_control = cb(1);
        consumer_control.wait_mask = wait_mask(0b000001);
        let mut load_slot = slot(
            load_control,
            OpClass::Other,
            &format!("LDG.E.STRONG.SM R{load_destination}, [R2.64]"),
        );
        load_slot.mem_role = MemRole::EligibleLoad(CacheOpToken::LdgEStrongSm);
        let sched = schedule(vec![
            (0, load_slot),
            (
                16,
                slot(
                    cb(producer_cycle - 15),
                    OpClass::Other,
                    "IADD3 R10, R10, 0x1, RZ",
                ),
            ),
            (32, slot(cb(1), OpClass::Other, "HADD2 R9, R9, R6")),
            (
                48,
                slot(cb(1), OpClass::Other, "ISETP.NE.AND P0, PT, R10, 0x40, PT"),
            ),
            (64, slot(cb(29 - producer_cycle - 2), OpClass::Other, "NOP")),
            (
                80,
                slot(
                    consumer_control,
                    OpClass::Other,
                    "LOP3.LUT R12, R12, R11, R7, 0x96, !PT",
                ),
            ),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalEligible {
                            off: 0,
                            len: 1,
                            token: CacheOpToken::LdgEStrongSm,
                        },
                    },
                    inst(16),
                    inst(32),
                    inst(48),
                    inst(64),
                    inst(80),
                ],
                sectors: vec![0],
            }],
        }];
        let mut core =
            SmCore::new(&blocks, &sched, &params(), SmMemMode::L1Const(l1_config())).unwrap();
        let l1 = core.mem.const_l1_mut().unwrap();
        l1.access_eligible_unobserved(MemoryOrigin::Global, 0, 0)
            .unwrap();
        l1.access_eligible_unobserved(MemoryOrigin::Global, 0, 244)
            .unwrap();
        core.t = 244;
        let mut out = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Done
        );
        assert!(out.is_empty());
        core.result()
    }

    #[test]
    fn same_bank_hadd2_defers_only_the_coincident_l1tex_writeback() {
        let noncollision = result_priority_case(11, 22);
        let collision = result_priority_case(11, 23);
        let opposite = result_priority_case(14, 23);

        assert_eq!(noncollision.cycles, 279);
        assert_eq!(collision.cycles, 280);
        assert_eq!(opposite.cycles, 279);
        assert!(noncollision.readiness.result_writeback.is_empty());
        assert!(opposite.readiness.result_writeback.is_empty());
        assert_eq!(
            collision.readiness.result_writeback,
            ResultWritebackDiagnostics {
                fixed_over_l1tex_collisions: 1,
                l1tex_deferral_cycles: 1,
            }
        );
        assert_eq!(collision.readiness.global_memory_issue.load_instructions, 1);
        assert_eq!(collision.readiness.schema_version(), 6);
    }

    #[test]
    fn flat_async_copy_waits_for_source_completion_without_generic_sb0_latency() {
        let sched = schedule(vec![
            (0, slot(cb(1), OpClass::Other, "LDGSTS.E")),
            (16, async_commit_slot()),
            (32, async_wait_slot(0)),
            (48, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![async_copy_inst(0, 0, 1), inst(16), inst(32), inst(48)],
                sectors: vec![40],
            }],
        }];
        let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
        let mut out = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Done
        );
        assert!(out.is_empty());
        assert_eq!(
            core.result().cycles,
            403,
            "wait must issue at source-ready t=400 plus one payload cycle"
        );
        assert_eq!(core.warps[0].barrier_clear[0], 0);
        let async_copy = core.async_copies[0]
            .state
            .as_deref()
            .expect("effective copy allocates protocol state");
        assert!(async_copy.open_group.is_none());
        assert!(async_copy.committed_groups.is_empty());
        assert_eq!(core.result().readiness.async_groups.groups_committed, 1);
        assert_eq!(core.result().readiness.async_groups.groups_waited, 1);
        assert_eq!(
            core.result()
                .readiness
                .async_groups
                .destination_local_limited_groups,
            1
        );
        assert_eq!(core.result().readiness.async_groups.source_cycles, 399);
        assert_eq!(
            core.result()
                .readiness
                .async_groups
                .destination_local_cycles,
            1
        );
        assert_eq!(
            core.result()
                .readiness
                .limiting_cycles
                .async_destination_local,
            400
        );
    }

    #[test]
    fn async_copy_payload_rate_is_route_local_and_sector_count_independent() {
        for (policy, opcode, expected_cycles) in [
            (LdgstsL1Policy::CacheAtAllLevels, "LDGSTS.E", 406),
            (LdgstsL1Policy::BypassL1, "LDGSTS.E.BYPASS", 410),
        ] {
            for len in [1_u16, 16] {
                let sched = schedule(vec![
                    (0, slot(cb(1), OpClass::Other, opcode)),
                    (16, async_commit_slot()),
                    (32, async_wait_slot(0)),
                    (48, slot(cb(1), OpClass::Other, "EXIT")),
                ]);
                let blocks = vec![BlockReplay {
                    launch_index: 0,
                    warps: vec![WarpReplay {
                        warp_slot: 0,
                        insts: vec![
                            ReplayInst {
                                pc: 0,
                                kind: AccessKind::GlobalAsyncCopy {
                                    off: 0,
                                    len,
                                    completion_work: if policy == LdgstsL1Policy::CacheAtAllLevels {
                                        256
                                    } else {
                                        512
                                    },
                                    l1_policy: policy,
                                },
                            },
                            inst(16),
                            inst(32),
                            inst(48),
                        ],
                        sectors: (0..len).map(i64::from).collect(),
                    }],
                }];

                let result = simulate_sm(&blocks, &sched, &params(), None).unwrap();
                assert_eq!(
                    result.cycles, expected_cycles,
                    "policy={policy:?}, sectors={len}"
                );
            }
        }
    }

    #[test]
    fn exact_destination_wavefronts_serialize_across_sm_subcores() {
        let sched = schedule(vec![
            (0, slot(cb(1), OpClass::Other, "LDGSTS.E")),
            (16, async_commit_slot()),
        ]);
        let completion_work = sim_mem::encode_async_copy_destination_work(512).unwrap();
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: (0..4)
                .map(|warp_slot| WarpReplay {
                    warp_slot,
                    insts: vec![
                        ReplayInst {
                            pc: 0,
                            kind: AccessKind::GlobalAsyncCopy {
                                off: 0,
                                len: 1,
                                completion_work,
                                l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                            },
                        },
                        inst(16),
                    ],
                    sectors: vec![40],
                })
                .collect(),
        }];
        let mut p = params();
        p.global_flat_latency = 0;
        let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
        let mut requests = Vec::new();

        assert!(core.issue_cycle(&mut requests).unwrap());
        assert!(requests.is_empty());
        assert_eq!(core.ldgsts_destination_pipe_ready, 0);
        core.t = 1;
        assert!(core.issue_cycle(&mut requests).unwrap());
        core.enqueue_ready_async_copy_destinations().unwrap();
        assert_eq!(core.ldgsts_destination_pipe_ready, 24);
        let destination_ready = core
            .async_copies
            .iter()
            .map(|copy| {
                copy.state
                    .as_deref()
                    .unwrap()
                    .committed_groups
                    .front()
                    .copied()
                    .unwrap()
                    .destination_completion_time
            })
            .collect::<Vec<_>>();
        assert_eq!(destination_ready, vec![6, 12, 18, 24]);
        let completion_ready = core
            .async_copies
            .iter()
            .map(|copy| {
                copy.state
                    .as_deref()
                    .unwrap()
                    .committed_groups
                    .front()
                    .copied()
                    .unwrap()
                    .completion_time
            })
            .collect::<Vec<_>>();
        assert_eq!(completion_ready, vec![8, 12, 18, 24]);
    }

    #[test]
    fn shared_mode_attaches_exact_destination_work_once() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDGSTS.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: AccessKind::GlobalAsyncCopy {
                        off: 0,
                        len: 1,
                        completion_work: sim_mem::encode_async_copy_destination_work(128).unwrap(),
                        l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                    },
                }],
                sectors: vec![40],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut requests = Vec::new();

        assert!(core.issue_cycle(&mut requests).unwrap());
        let group = core.async_copies[0]
            .state
            .as_deref()
            .unwrap()
            .open_group
            .unwrap();
        assert_eq!(group.destination_wavefronts, 1);
        assert_eq!(group.delivery_work, 128);
        assert_eq!(group.pending_sources, 1);
        assert_eq!(requests.len(), 1);
    }

    #[test]
    fn exact_destination_service_waits_for_actual_source_time() {
        let sched = schedule(vec![
            (0, slot(cb(1), OpClass::Other, "LDGSTS.E")),
            (16, async_commit_slot()),
            (32, async_wait_slot(0)),
            (48, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalAsyncCopy {
                            off: 0,
                            len: 1,
                            completion_work: sim_mem::encode_async_copy_destination_work(128)
                                .unwrap(),
                            l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                        },
                    },
                    inst(16),
                    inst(32),
                    inst(48),
                ],
                sectors: vec![40],
            }],
        }];
        let mut p = params();
        p.global_flat_latency = 100;
        let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
        let mut requests = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Bounded(101), &mut requests)
                .unwrap(),
            Advance::Idle { wake_at: Some(101) }
        );
        assert_eq!(core.ldgsts_destination_pipe_ready, 102);
        let group = core.async_copies[0]
            .state
            .as_deref()
            .unwrap()
            .committed_groups
            .front()
            .unwrap();
        assert!(group.destination_service_queued);
        assert_eq!(group.destination_completion_time, 102);
        assert_eq!(
            core.warps[0].next, 2,
            "wait issued before service completed"
        );

        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut requests)
                .unwrap(),
            Advance::Done
        );
        assert_eq!(core.result().cycles, 104);
    }

    #[test]
    fn exact_destination_pipe_overflow_is_transactional() {
        let sched = schedule(vec![
            (0, slot(cb(1), OpClass::Other, "LDGSTS.E")),
            (16, async_commit_slot()),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalAsyncCopy {
                            off: 0,
                            len: 1,
                            completion_work: sim_mem::encode_async_copy_destination_work(128)
                                .unwrap(),
                            l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                        },
                    },
                    inst(16),
                ],
                sectors: vec![40],
            }],
        }];
        let mut p = params();
        p.global_flat_latency = 0;
        let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
        let mut requests = Vec::new();

        assert!(core.issue_cycle(&mut requests).unwrap());
        core.t = 1;
        assert!(core.issue_cycle(&mut requests).unwrap());
        core.ldgsts_destination_pipe_ready = u64::MAX;

        assert!(matches!(
            core.enqueue_ready_async_copy_destinations(),
            Err(IssueError::TimeOverflow {
                operation: "LDGSTS aggregate destination-wavefront service",
                base: u64::MAX,
                delta: 2,
            })
        ));
        assert_eq!(core.ldgsts_destination_pipe_ready, u64::MAX);
        assert_eq!(core.instructions_issued(), 2);
        assert_eq!(core.warps[0].next, 2);
        let group = core.async_copies[0]
            .state
            .as_deref()
            .unwrap()
            .committed_groups
            .front()
            .unwrap();
        assert!(!group.destination_service_queued);
        assert_eq!(group.destination_completion_time, 0);
        assert!(requests.is_empty());
    }

    #[test]
    fn malformed_exact_destination_work_rejects_issue_before_mutation() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDGSTS.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: AccessKind::GlobalAsyncCopy {
                        off: 0,
                        len: 1,
                        completion_work: sim_mem::encode_async_copy_destination_work(130).unwrap(),
                        l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                    },
                }],
                sectors: vec![40],
            }],
        }];
        let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
        let mut requests = Vec::new();

        assert!(matches!(
            core.issue_cycle(&mut requests),
            Err(IssueError::InvalidAsyncCopyDestinationWork {
                completion_work: 130,
                work_per_wavefront: 128,
            })
        ));
        assert_eq!(core.instructions_issued(), 0);
        assert_eq!(core.warps[0].next, 0);
        assert!(core.async_copies[0].state.is_none());
        assert!(requests.is_empty());
    }

    #[test]
    fn async_copy_group_accumulates_each_copy_payload_once() {
        for (policy, opcode, expected_cycles) in [
            (LdgstsL1Policy::CacheAtAllLevels, "LDGSTS.E", 411),
            (LdgstsL1Policy::BypassL1, "LDGSTS.E.BYPASS", 419),
        ] {
            let sched = schedule(vec![
                (0, slot(cb(1), OpClass::Other, opcode)),
                (16, slot(cb(1), OpClass::Other, opcode)),
                (32, async_commit_slot()),
                (48, async_wait_slot(0)),
                (64, slot(cb(1), OpClass::Other, "EXIT")),
            ]);
            let blocks = vec![BlockReplay {
                launch_index: 0,
                warps: vec![WarpReplay {
                    warp_slot: 0,
                    insts: vec![
                        ReplayInst {
                            pc: 0,
                            kind: AccessKind::GlobalAsyncCopy {
                                off: 0,
                                len: 1,
                                completion_work: if policy == LdgstsL1Policy::CacheAtAllLevels {
                                    256
                                } else {
                                    512
                                },
                                l1_policy: policy,
                            },
                        },
                        ReplayInst {
                            pc: 16,
                            kind: AccessKind::GlobalAsyncCopy {
                                off: 1,
                                len: 16,
                                completion_work: if policy == LdgstsL1Policy::CacheAtAllLevels {
                                    256
                                } else {
                                    512
                                },
                                l1_policy: policy,
                            },
                        },
                        inst(32),
                        inst(48),
                        inst(64),
                    ],
                    sectors: (0_i64..17).collect(),
                }],
            }];

            let result = simulate_sm(&blocks, &sched, &params(), None).unwrap();
            assert_eq!(result.cycles, expected_cycles, "policy={policy:?}");
        }
    }

    #[test]
    fn async_copy_completion_work_mismatch_rejects_issue_before_mutation() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDGSTS.E.BYPASS"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: AccessKind::GlobalAsyncCopy {
                        off: 0,
                        len: 1,
                        completion_work: 0,
                        l1_policy: LdgstsL1Policy::BypassL1,
                    },
                }],
                sectors: vec![40],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let before = *core.l1_counters().unwrap();
        let mut requests = Vec::new();

        let error = core
            .advance(ReplayHorizon::Unbounded, &mut requests)
            .unwrap_err();
        assert!(matches!(
            error,
            IssueError::AsyncCopyCompletionWorkMismatch {
                warp: 0,
                sectors: 1,
                completion_work: 0,
            }
        ));
        assert_eq!(core.instructions_issued(), 0);
        assert_eq!(core.warps[0].next, 0);
        assert!(core.async_copies[0].state.is_none());
        assert_eq!(*core.l1_counters().unwrap(), before);
        assert!(requests.is_empty());
    }

    #[test]
    fn flat_async_copy_idle_skips_directly_to_source_completion() {
        let sched = schedule(vec![
            (0, slot(cb(1), OpClass::Other, "LDGSTS.E")),
            (16, async_commit_slot()),
            (32, async_wait_slot(0)),
            (48, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![async_copy_inst(0, 0, 1), inst(16), inst(32), inst(48)],
                sectors: vec![40],
            }],
        }];
        let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
        let mut out = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Bounded(402), &mut out).unwrap(),
            Advance::Idle { wake_at: Some(402) }
        );
        assert_eq!(core.now(), 402);
        assert_eq!(core.instructions_issued(), 3);
        assert_eq!(core.warps[0].next, 3, "wait must issue at t=401");

        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Done
        );
        assert_eq!(core.result().cycles, 403);
    }

    #[test]
    fn another_warp_fills_an_async_copy_wait_gap() {
        let sched = schedule(vec![
            (0, slot(cb(1), OpClass::Other, "LDGSTS.E")),
            (16, async_commit_slot()),
            (32, async_wait_slot(0)),
            (48, slot(cb(1), OpClass::Other, "EXIT")),
            (64, slot(cb(1), OpClass::Other, "IADD3")),
            (80, slot(cb(1), OpClass::Other, "IADD3")),
            (96, slot(cb(1), OpClass::Other, "IADD3")),
            (112, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![
                warp(0, vec![inst(64), inst(80), inst(96), inst(112)]),
                WarpReplay {
                    // Same SMSP and younger, so this warp is held after t=0.
                    warp_slot: 4,
                    insts: vec![async_copy_inst(0, 0, 1), inst(16), inst(32), inst(48)],
                    sectors: vec![40],
                },
            ],
        }];
        let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
        let mut out = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Bounded(6), &mut out).unwrap(),
            Advance::Idle { wake_at: Some(6) }
        );
        assert!(core.warps[0].done, "ready warp did not fill the wait gap");
        assert_eq!(core.warps[1].next, 2, "async wait issued before t=400");
        assert_eq!(core.instructions_issued(), 6);

        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Done
        );
        assert_eq!(core.instructions_issued(), 8);
        assert_eq!(core.result().cycles, 403);
    }

    #[test]
    fn constant_l1_async_copy_uses_eligible_source_route() {
        let sched = schedule(vec![
            (0, slot(cb(1), OpClass::Other, "LDGSTS.E")),
            (16, async_commit_slot()),
            (32, async_wait_slot(0)),
            (48, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![async_copy_inst(0, 0, 1), inst(16), inst(32), inst(48)],
                sectors: vec![40],
            }],
        }];

        let result = simulate_sm(&blocks, &sched, &params(), Some(l1_config())).unwrap();
        assert_eq!(
            result.cycles, 247,
            "wait must issue at source-ready t=244 plus one payload cycle"
        );
        let counters = result.l1.unwrap();
        assert_eq!(counters.bypass_sectors, 0);
        assert_eq!(counters.misses, 1);
    }

    #[test]
    fn wait_one_preserves_newest_group_for_a_later_wait_zero() {
        let sched = schedule(vec![
            (0, slot(cb(1), OpClass::Other, "LDGSTS.E")),
            (16, async_commit_slot()),
            (32, slot(cb(1), OpClass::Other, "LDGSTS.E")),
            (48, async_commit_slot()),
            (64, async_wait_slot(1)),
            (80, async_wait_slot(0)),
            (96, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    async_copy_inst(0, 0, 1),
                    inst(16),
                    async_copy_inst(32, 1, 1),
                    inst(48),
                    inst(64),
                    inst(80),
                    inst(96),
                ],
                sectors: vec![40, 44],
            }],
        }];
        let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
        let mut out = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Bounded(402), &mut out).unwrap(),
            Advance::Idle { wake_at: Some(402) }
        );
        assert_eq!(core.warps[0].next, 5, "wait-1 must issue at t=401");
        let async_copy = core.async_copies[0].state.as_deref().unwrap();
        assert_eq!(async_copy.committed_groups.len(), 1);
        assert_eq!(async_copy.committed_groups[0].completion_time, 404);

        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Done
        );
        assert_eq!(core.result().cycles, 406);
        assert!(
            core.async_copies[0]
                .state
                .as_deref()
                .unwrap()
                .committed_groups
                .is_empty()
        );
    }

    #[test]
    fn empty_async_copy_commit_is_trivially_complete_and_consumes_no_sequence() {
        let sched = schedule(vec![
            (0, async_commit_slot()),
            (16, async_wait_slot(0)),
            (32, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0), inst(16), inst(32)])],
        }];
        let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
        let mut out = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Done
        );
        assert_eq!(
            core.result().cycles,
            4,
            "the commit yield bit delays wait by one cycle, not by 25"
        );
        assert!(
            core.async_copies[0].state.is_none(),
            "an empty commit must not allocate or consume protocol state"
        );
        assert_eq!(core.warps[0].barrier_clear[0], 0);
    }

    #[test]
    fn async_copy_group_sequence_exhaustion_is_nonmutating() {
        let sched = schedule(vec![
            (0, slot(cb(1), OpClass::Other, "LDGSTS.E")),
            (16, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![async_copy_inst(0, 0, 1), inst(16)],
                sectors: vec![40],
            }],
        }];
        let mut core =
            SmCore::new(&blocks, &sched, &params(), SmMemMode::L1Const(l1_config())).unwrap();
        core.async_copies[0].state = Some(Box::new(AsyncCopyState {
            next_sequence: None,
            ..AsyncCopyState::new()
        }));
        let before = *core.l1_counters().unwrap();
        let mut out = Vec::new();

        assert!(matches!(
            core.advance(ReplayHorizon::Unbounded, &mut out),
            Err(IssueError::AsyncCopyGroupSequenceExhausted { warp: 0 })
        ));
        assert_eq!(core.instructions_issued(), 0);
        assert_eq!(core.now(), 0);
        assert_eq!(core.warps[0].next, 0);
        assert_eq!(*core.l1_counters().unwrap(), before);
        assert!(
            core.async_copies[0]
                .state
                .as_deref()
                .unwrap()
                .open_group
                .is_none()
        );
        assert!(out.is_empty());
    }

    #[test]
    fn committed_async_groups_preserve_fifo_wait_visibility() {
        let sched = schedule(vec![(0, async_wait_slot(1))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0)])],
        }];
        let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();

        let p = params();
        let older = core.async_copies[0]
            .add_pending_async_copy_sources(0, 1, 0, 0, None, &p)
            .unwrap();
        assert!(core.async_copies[0].commit_async_copy_group(1));
        let newer = core.async_copies[0]
            .add_pending_async_copy_sources(0, 1, 0, 0, None, &p)
            .unwrap();
        assert!(core.async_copies[0].commit_async_copy_group(2));

        core.async_copies[0]
            .complete_async_copy_sources(0, newer, None, 1, 50, &p)
            .unwrap();
        assert_eq!(
            core.async_copies[0].async_wait_clear_time(1),
            None,
            "a newer completion cannot make the required older prefix visible"
        );

        core.async_copies[0]
            .complete_async_copy_sources(0, older, None, 1, 100, &p)
            .unwrap();
        assert_eq!(core.async_copies[0].async_wait_clear_time(1), Some(100));
        core.async_copies[0].consume_async_wait(1, 100);
        assert_eq!(
            core.async_copies[0]
                .state
                .as_deref()
                .unwrap()
                .committed_groups
                .iter()
                .map(|group| group.id)
                .collect::<Vec<_>>(),
            vec![newer]
        );
    }

    #[test]
    fn bypass_copy_observation_preserves_individual_source_frontiers() {
        let p = params();
        let mut slot = AsyncCopySlot::default();
        let group = slot
            .add_pending_async_copy_sources(
                0,
                2,
                64,
                0,
                Some(BypassAsyncCopyInput {
                    instruction: 10,
                    source_cohorts: 8,
                    destination_lines: 8,
                    max_bank_wavefronts: 1,
                    max_bypass_wavefronts: 1,
                }),
                &p,
            )
            .unwrap();
        assert_eq!(
            slot.add_pending_async_copy_sources(
                0,
                2,
                64,
                0,
                Some(BypassAsyncCopyInput {
                    instruction: 20,
                    source_cohorts: 8,
                    destination_lines: 16,
                    max_bank_wavefronts: 2,
                    max_bypass_wavefronts: 4,
                }),
                &p,
            )
            .unwrap(),
            group
        );
        assert!(slot.commit_async_copy_group(10));

        slot.complete_async_copy_sources(0, group, Some(10), 1, 30, &p)
            .unwrap();
        slot.complete_async_copy_sources(0, group, Some(10), 1, 40, &p)
            .unwrap();
        slot.complete_async_copy_sources(0, group, Some(20), 1, 50, &p)
            .unwrap();
        slot.complete_async_copy_sources(0, group, Some(20), 1, 70, &p)
            .unwrap();

        let committed = slot.state.as_deref().unwrap().committed_groups[0];
        let projected = slot
            .bypass_copy_projection(0, &committed, &p, 80)
            .unwrap()
            .unwrap();
        let per_copy = p
            .ldgsts_completion_ready(40, 64)
            .unwrap()
            .max(p.ldgsts_completion_ready(70, 64).unwrap());
        let aggregate = committed.local_completion_time(&p).unwrap();
        assert_eq!(projected.copies, 2);
        assert_eq!(projected.source_cohorts, 16);
        assert_eq!(projected.destination_lines, 24);
        assert_eq!(projected.max_bank_wavefronts_sum, 3);
        assert_eq!(projected.max_bypass_wavefronts_sum, 5);
        assert_eq!(projected.bank_wavefront_upper_bound, 24);
        assert_eq!(projected.bypass_wavefront_upper_bound, 40);
        assert_eq!(projected.source_spread_cycles, 30);
        assert_eq!(projected.per_copy_completion_cycles, per_copy - 10);
        assert_eq!(
            projected.group_aggregate_excess_cycles,
            aggregate - per_copy
        );
        assert_eq!(projected.topology_candidate_penalty_cycles, 14);
        assert_eq!(projected.topology_candidate_completion_cycles, 75);
        assert_eq!(projected.topology_candidate_later_cycles, 13);
        assert_eq!(projected.topology_candidate_earlier_cycles, 0);
        assert_eq!(projected.pipeline_q1_completion_cycles, 75);
        assert_eq!(projected.pipeline_q1_exposure_cycles, 5);
        assert_eq!(projected.pipeline_q2_completion_cycles, 75);
        assert_eq!(projected.pipeline_q2_exposure_cycles, 5);
        assert_eq!(projected.pipeline_q4_completion_cycles, 75);
        assert_eq!(projected.pipeline_q4_exposure_cycles, 5);
        assert_eq!(projected.pipeline_q8_completion_cycles, 75);
        assert_eq!(projected.pipeline_q8_exposure_cycles, 5);
        assert_eq!(projected.pipeline_q16_completion_cycles, 75);
        assert_eq!(projected.pipeline_q16_exposure_cycles, 5);
        assert_eq!(slot.async_wait_clear_time(0), Some(aggregate));

        slot.consume_async_wait(0, aggregate);
        let state = slot.state.as_deref().unwrap();
        assert!(state.committed_groups.is_empty());
        assert!(state.bypass_copies.is_empty());
    }

    #[test]
    fn bypass_pipeline_candidate_tracks_queue_and_idle_arrivals() {
        let mut copies = vec![
            BypassPipelineCopyCandidate {
                instruction: 20,
                payload_completion: 20,
                topology_penalty: 4,
            },
            BypassPipelineCopyCandidate {
                instruction: 30,
                payload_completion: 22,
                topology_penalty: 6,
            },
            BypassPipelineCopyCandidate {
                instruction: 10,
                payload_completion: 20,
                topology_penalty: 8,
            },
        ];

        assert_eq!(
            bypass_pipeline_candidate_completions(&mut copies).unwrap(),
            [38, 32, 29, 28, 28]
        );
        assert_eq!(
            copies
                .iter()
                .map(|copy| (copy.payload_completion, copy.instruction))
                .collect::<Vec<_>>(),
            [(20, 10), (20, 20), (22, 30)]
        );

        copies.push(BypassPipelineCopyCandidate {
            instruction: 40,
            payload_completion: 100,
            topology_penalty: 2,
        });
        assert_eq!(
            bypass_pipeline_candidate_completions(&mut copies).unwrap(),
            [102; BYPASS_PIPELINE_CANDIDATE_Q.len()]
        );

        let observation = ReadinessDiagnostics {
            bypass_async_copies: Some(Box::new(BypassAsyncCopyDiagnostics {
                copies_observed: 3,
                pipeline_groups_observed: 1,
                pipeline_q1_completion_cycles: 38,
                pipeline_q1_exposure_cycles: 18,
                ..BypassAsyncCopyDiagnostics::default()
            })),
            ..ReadinessDiagnostics::default()
        };
        assert_eq!(observation.schema_version(), 16);
        let mut aggregate = ReadinessDiagnostics::default();
        aggregate.add_checked(&observation).unwrap();
        aggregate.add_checked(&observation).unwrap();
        let aggregate_pipeline = aggregate.bypass_async_copies.unwrap();
        assert_eq!(aggregate_pipeline.pipeline_groups_observed, 2);
        assert_eq!(aggregate_pipeline.pipeline_q1_completion_cycles, 76);
        assert_eq!(aggregate_pipeline.pipeline_q1_exposure_cycles, 36);
    }

    #[test]
    fn async_copy_pending_overflow_rejects_issue_without_mutation() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDGSTS.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![async_copy_inst(0, 0, 1)],
                sectors: vec![40],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let group = core.async_copies[0]
            .add_pending_async_copy_sources(0, u32::MAX, 0, 0, None, &params())
            .unwrap();
        let before_group = core.async_copies[0].state.as_deref().unwrap().open_group;
        let before_l1 = *core.l1_counters().unwrap();
        let mut requests = Vec::new();

        assert!(matches!(
            core.advance(ReplayHorizon::Unbounded, &mut requests),
            Err(IssueError::AsyncCopyPendingOverflow {
                warp: 0,
                group: got_group,
                current: u32::MAX,
                add: 1,
            }) if got_group == group
        ));
        assert_eq!(core.instructions_issued(), 0);
        assert_eq!(core.warps[0].next, 0);
        assert_eq!(
            core.async_copies[0].state.as_deref().unwrap().open_group,
            before_group
        );
        assert_eq!(*core.l1_counters().unwrap(), before_l1);
        assert!(requests.is_empty());
    }

    #[test]
    fn stall_counts_delay_reissue() {
        // stalls 4,4: issues at t=0,4 → cycles 5.
        let sched = schedule(vec![
            (0, slot(cb(4), OpClass::Other, "FFMA")),
            (16, slot(cb(4), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0), inst(16)])],
        }];
        let r = simulate_sm(&blocks, &sched, &params(), None).unwrap();
        assert_eq!(r.instructions_issued, 2);
        assert_eq!(r.cycles, 5);
    }

    #[test]
    fn same_smsp_tensor_issues_observe_the_initiation_interval() {
        let sched = schedule(vec![
            (0, tensor_slot(cb(1), "HMMA.16816.F32", 16)),
            (16, tensor_slot(cb(1), "HMMA.16816.F32", 16)),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0), inst(16)])],
        }];
        let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
        let mut out = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Done
        );
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), 2);
        assert_eq!(
            core.result().cycles,
            17,
            "tensor issues must occur at t=0 and t=16"
        );
        assert_eq!(core.tensor_pipe_ready, [32, 0, 0, 0]);
    }

    #[test]
    fn separate_smsps_have_independent_tensor_timelines() {
        let sched = schedule(vec![(0, tensor_slot(cb(1), "HMMA.16816.F32.BF16", 16))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0)]), warp(1, vec![inst(0)])],
        }];
        let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
        let mut out = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Done
        );
        assert_eq!(core.instructions_issued(), 2);
        assert_eq!(core.result().cycles, 1, "both SMSPs should issue at t=0");
        assert_eq!(core.tensor_pipe_ready, [16, 16, 0, 0]);
    }

    #[test]
    fn dispatch_only_work_fills_a_tensor_busy_gap() {
        let sched = schedule(vec![
            (0, tensor_slot(cb(1), "HMMA.16816.F32", 16)),
            (16, slot(cb(1), OpClass::Other, "IADD3")),
            (32, tensor_slot(cb(1), "HMMA.16816.F32", 16)),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0), inst(16), inst(32)])],
        }];
        let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
        let mut out = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Bounded(2), &mut out).unwrap(),
            Advance::Idle { wake_at: Some(2) }
        );
        assert_eq!(core.instructions_issued(), 2);
        assert_eq!(core.warps[0].next, 2);
        assert_eq!(core.tensor_pipe_ready[0], 16);

        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Done
        );
        assert_eq!(core.result().cycles, 17);
    }

    #[test]
    fn blocked_held_tensor_warp_yields_dispatch_to_another_warp() {
        let sched = schedule(vec![
            (0, tensor_slot(cb(1), "HMMA.16816.F32", 16)),
            (16, tensor_slot(cb(1), "HMMA.16816.F32", 16)),
            (32, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            // Warp 4 is younger, so CGGTY picks and holds it at t=0.
            warps: vec![warp(0, vec![inst(32)]), warp(4, vec![inst(0), inst(16)])],
        }];
        let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
        let mut out = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Bounded(1), &mut out).unwrap(),
            Advance::Idle { wake_at: Some(1) }
        );
        assert_eq!(core.hold[0], Some(1));
        assert_eq!(core.warps[1].next, 1);

        assert_eq!(
            core.advance(ReplayHorizon::Bounded(2), &mut out).unwrap(),
            Advance::Idle { wake_at: Some(2) }
        );
        assert!(core.warps[0].done, "ready dispatch warp was not selected");
        assert_eq!(core.warps[1].next, 1, "blocked held tensor warp issued");
        assert_eq!(core.instructions_issued(), 2);
    }

    #[test]
    fn tensor_blocked_idle_reports_the_exact_resource_wake_time() {
        let sched = schedule(vec![
            (0, tensor_slot(cb(1), "HMMA.16816.F32", 16)),
            (16, tensor_slot(cb(1), "HMMA.16816.F32", 16)),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0), inst(16)])],
        }];
        let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
        let mut out = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Bounded(1), &mut out).unwrap(),
            Advance::Idle { wake_at: Some(1) }
        );
        assert_eq!(
            core.advance(ReplayHorizon::Bounded(16), &mut out).unwrap(),
            Advance::Idle { wake_at: Some(16) }
        );
        assert_eq!(core.instructions_issued(), 1);
        assert_eq!(core.warps[0].next, 1);
    }

    #[test]
    fn tensor_timeline_survives_cta_admission_waves() {
        let sched = schedule(vec![(0, tensor_slot(cb(1), "HMMA.16816.F32", 16))]);
        let blocks = vec![
            BlockReplay {
                launch_index: 0,
                warps: vec![warp(0, vec![inst(0)])],
            },
            BlockReplay {
                launch_index: 1,
                warps: vec![warp(0, vec![inst(0)])],
            },
        ];
        let mut p = params();
        p.max_resident_warps_per_sm = 1;
        let r = simulate_sm(&blocks, &sched, &p, None).unwrap();

        assert_eq!(r.instructions_issued, 2);
        assert_eq!(r.cycles, 17, "second CTA must wait for the t=16 timeline");
    }

    #[test]
    fn tensor_timestamp_overflow_rejects_the_cycle_before_mutation() {
        let sched = schedule(vec![(0, tensor_slot(cb(1), "HMMA.16816.F32", 16))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0)])],
        }];
        let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
        core.t = u64::MAX - 15;
        let before_unfinished = core.unfinished_warps;
        let before_resident = core.resident_warps;
        let mut out = Vec::new();

        assert!(matches!(
            core.advance(ReplayHorizon::Unbounded, &mut out),
            Err(IssueError::TimeOverflow {
                operation: "tensor-pipeline availability",
                base,
                delta: 16,
            }) if base == u64::MAX - 15
        ));
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), 0);
        assert_eq!(core.warps[0].next, 0);
        assert!(!core.warps[0].done);
        assert_eq!(core.tensor_pipe_ready, [0; SUB_CORES]);
        assert_eq!(core.unfinished_warps, before_unfinished);
        assert_eq!(core.resident_warps, before_resident);
    }

    #[test]
    fn flat_write_barrier_wait_blocks_consumer_without_l1() {
        // inst0 (global load, no L1 attached) sets barrier 0; inst1 waits.
        // Barrier clears at the flat latency 400 → inst1 at t=400.
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000001);
        let sched = schedule(vec![
            (0, slot(load, OpClass::Other, "LDG.E")),
            (16, slot(consumer, OpClass::Other, "FFMA")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: eligible(0, 1),
                    },
                    inst(16),
                ],
                sectors: vec![0],
            }],
        }];
        let r = simulate_sm(&blocks, &sched, &params(), None).unwrap();
        assert_eq!(r.instructions_issued, 2);
        assert_eq!(r.cycles, 401);
        assert_eq!(r.readiness.limiting_cycles.dependence_counter_l1tex, 400);
        assert_eq!(
            r.readiness
                .constraint_exposure_cycles
                .dependence_counter_l1tex,
            400
        );
    }

    #[test]
    fn ordinary_depbar_operand_waits_for_non_memory_producer_without_async_mutation() {
        // The spacing probe's retained clock reads compile as an S2UR producer
        // on SB1..SB5 followed by `DEPBAR.LE SBn, 0x0`. The DEPBAR control word
        // has no wait mask, so the operand itself must consume the ordinary
        // non-memory producer latency (25 cycles in this fixture).
        let mut producer = cb(1);
        producer.write_barrier = Some(barrier(2));
        let sched = schedule(vec![
            (0, slot(producer, OpClass::Other, "S2UR UR4, SR_CLOCK")),
            (16, slot(cb(7), OpClass::Other, "DEPBAR.LE SB2, 0x0")),
            // Retained only to enable the async protocol sidecar in this
            // fixture, making absence of accidental group mutation observable.
            (32, async_wait_slot(0)),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0), inst(16)])],
        }];
        let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
        let mut requests = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut requests)
                .unwrap(),
            Advance::Done
        );
        assert!(requests.is_empty());
        assert!(core.async_copies[0].state.is_none());
        let result = core.result();
        assert_eq!(result.instructions_issued, 2);
        assert_eq!(result.cycles, 26);
        assert_eq!(
            result
                .readiness
                .limiting_cycles
                .dependence_counter_non_memory_write,
            25
        );
        assert_eq!(result.readiness.async_groups.groups_committed, 0);
        assert_eq!(result.readiness.async_groups.groups_waited, 0);
    }

    #[test]
    fn shared_bank_wavefronts_extend_dependent_completion_in_both_memory_modes() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000001);
        let sched = schedule(vec![
            (0, slot(load, OpClass::Other, "LDS")),
            (16, slot(consumer, OpClass::Other, "IADD3")),
        ]);
        let blocks = |wavefronts| {
            vec![BlockReplay {
                launch_index: 0,
                warps: vec![warp(
                    0,
                    vec![
                        ReplayInst {
                            pc: 0,
                            kind: AccessKind::Shared { wavefronts },
                        },
                        inst(16),
                    ],
                )],
            }]
        };

        for l1 in [None, Some(l1_config())] {
            let one = simulate_sm(&blocks(1), &sched, &params(), l1).unwrap();
            let four = simulate_sm(&blocks(4), &sched, &params(), l1).unwrap();
            assert_eq!(one.cycles, 24);
            assert_eq!(four.cycles, 30);
            assert_eq!(four.cycles - one.cycles, 3 * 2);
        }
    }

    #[test]
    fn shared_wavefront_latency_multiplication_overflow_is_nonmutating() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let sched = schedule(vec![(0, slot(load, OpClass::Other, "LDS"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(
                0,
                vec![ReplayInst {
                    pc: 0,
                    kind: AccessKind::Shared { wavefronts: 3 },
                }],
            )],
        }];
        let mut p = params();
        p.shared_additional_wavefront_latency = u64::MAX;
        let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
        let mut out = Vec::new();

        assert!(matches!(
            core.advance(ReplayHorizon::Unbounded, &mut out),
            Err(IssueError::TimeMultiplicationOverflow {
                operation: "shared-memory additional-wavefront latency",
                lhs: u64::MAX,
                rhs: 2,
            })
        ));
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), 0);
        assert_eq!(core.warps[0].next, 0);
        assert_eq!(core.warps[0].barrier_clear, [0; DEP_BARRIERS]);
    }

    #[test]
    fn l1_hit_shortens_the_dependent_load_chain() {
        // Two dependent loads of the SAME sector + consumers. With the L1
        // attached, the first load misses (ready at 244) and the second
        // HITS (ready at +34); without it both cost the flat 400.
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000001);
        let sched = schedule(vec![
            (0, slot(load, OpClass::Other, "LDG.E")),
            (16, slot(consumer, OpClass::Other, "FFMA")),
            (32, slot(load, OpClass::Other, "LDG.E")),
            (48, slot(consumer, OpClass::Other, "EXIT")),
        ]);
        let mk = || {
            vec![BlockReplay {
                launch_index: 0,
                warps: vec![WarpReplay {
                    warp_slot: 0,
                    insts: vec![
                        ReplayInst {
                            pc: 0,
                            kind: eligible(0, 1),
                        },
                        inst(16),
                        ReplayInst {
                            pc: 32,
                            kind: eligible(1, 1),
                        },
                        inst(48),
                    ],
                    sectors: vec![7, 7],
                }],
            }]
        };
        let with_l1 = simulate_sm(&mk(), &sched, &params(), Some(l1_config())).unwrap();
        let without = simulate_sm(&mk(), &sched, &params(), None).unwrap();
        let c = with_l1.l1.unwrap();
        assert_eq!(c.hits, 1);
        assert_eq!(c.misses, 1);
        assert_eq!(c.line_alloc_misses, 1);
        // miss chain: barrier clears 244, consumer at 244; hit chain: the
        // second load issues at 245, hit ready 245+34; EXIT at 279.
        assert!(with_l1.cycles < without.cycles);
        assert_eq!(without.cycles, 802);
    }

    #[test]
    fn bypass_untagged_and_store_sectors_are_counted() {
        let sched = schedule(vec![
            (0, slot(cb(1), OpClass::Other, "LDG.E.CG")),
            (16, slot(cb(1), OpClass::Other, "LDG.E")),
            (32, slot(cb(1), OpClass::Other, "STG.E")),
            (48, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalBypass {
                            off: 0,
                            len: 2,
                            token: CacheOpToken::LdgECg,
                            l2_requests: 0,
                        },
                    },
                    ReplayInst {
                        pc: 16,
                        kind: AccessKind::GlobalUntagged { off: 2, len: 1 },
                    },
                    ReplayInst {
                        pc: 32,
                        kind: AccessKind::GlobalStore { off: 3, len: 2 },
                    },
                    inst(48),
                ],
                sectors: vec![0, 1, 2, 3, 4],
            }],
        }];
        let r = simulate_sm(&blocks, &sched, &params(), Some(l1_config())).unwrap();
        let c = r.l1.unwrap();
        assert_eq!(c.bypass_sectors, 2);
        assert_eq!(c.untagged_sectors, 1);
        assert_eq!(c.store_sectors, 2);
        assert_eq!(c.store_hits, 0, "stores never allocate");
        assert_eq!(c.hits + c.misses, 0, "no modeled accesses in this stream");
    }

    #[test]
    fn yield_bit_excludes_warp_next_cycle() {
        // CGGTY paper: yield = "do not issue this warp next cycle"; with
        // no other warp ready that is a one-cycle bubble. stall 1 +
        // yield → second instruction issues at t=2, cycles=3.
        let mut y = cb(1);
        y.yield_flag = true;
        let sched = schedule(vec![
            (0, slot(y, OpClass::Other, "IADD3")),
            (16, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0), inst(16)])],
        }];
        let r = simulate_sm(&blocks, &sched, &params(), None).unwrap();
        assert_eq!(r.cycles, 3);
    }

    #[test]
    fn shared_barrier_index_accumulates_like_a_counter() {
        // Two producers on barrier 0: a slow global (flat 400) then a
        // fast non-memory (25). The consumer waits on bit 0 and must
        // wait for the SLOW producer — hardware barriers are counters,
        // a later fast producer must not shorten the pending wait.
        let mut slow = cb(1);
        slow.write_barrier = Some(barrier(0));
        let mut fast = cb(1);
        fast.write_barrier = Some(barrier(0));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000001);
        let sched = schedule(vec![
            (0, slot(slow, OpClass::Other, "LDG.E")),
            (16, slot(fast, OpClass::Other, "S2R")),
            (32, slot(consumer, OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: eligible(0, 1),
                    },
                    inst(16),
                    inst(32),
                ],
                sectors: vec![0],
            }],
        }];
        let r = simulate_sm(&blocks, &sched, &params(), None).unwrap();
        // Consumer waits until t=400 (slow), NOT t=1+25 (fast overwrite).
        assert_eq!(r.cycles, 401);
    }

    #[test]
    fn all_empty_block_does_not_leak_residency() {
        // Block 0's warps are all traced-empty; block 1 must still be
        // admitted and replayed even under a tight residency cap.
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
        let blocks = vec![
            BlockReplay {
                launch_index: 0,
                warps: (0..4).map(|s| warp(s, vec![])).collect(),
            },
            BlockReplay {
                launch_index: 1,
                warps: vec![warp(0, vec![inst(0)])],
            },
        ];
        let mut p = params();
        p.max_resident_warps_per_sm = 4;
        let r = simulate_sm(&blocks, &sched, &p, None).unwrap();
        assert_eq!(r.instructions_issued, 1);
    }

    #[test]
    fn low_level_replay_rejects_invalid_physical_warp_slots() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);

        let negative = vec![BlockReplay {
            launch_index: 7,
            warps: vec![warp(-1, vec![inst(0)])],
        }];
        assert!(matches!(
            SmCore::new(&negative, &sched, &params(), SmMemMode::Flat),
            Err(IssueError::WarpSlotOutOfRange {
                launch_index: 7,
                warp_slot: -1,
                arch: ArchId::Sm86,
                physical_warp_slots: 48,
            })
        ));

        let out_of_range = vec![BlockReplay {
            launch_index: 8,
            warps: vec![warp(48, vec![inst(0)])],
        }];
        assert!(matches!(
            SmCore::new(&out_of_range, &sched, &params(), SmMemMode::Flat),
            Err(IssueError::WarpSlotOutOfRange {
                launch_index: 8,
                warp_slot: 48,
                arch: ArchId::Sm86,
                physical_warp_slots: 48,
            })
        ));

        let duplicate = vec![BlockReplay {
            launch_index: 9,
            warps: vec![warp(3, vec![inst(0)]), warp(3, vec![inst(0)])],
        }];
        assert!(matches!(
            SmCore::new(&duplicate, &sched, &params(), SmMemMode::Flat),
            Err(IssueError::DuplicateWarpSlot {
                launch_index: 9,
                warp_slot: 3,
            })
        ));
    }

    #[test]
    fn first_cta_must_fit_the_resident_warp_ceiling() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
        let blocks = vec![BlockReplay {
            launch_index: 4,
            warps: vec![warp(0, vec![inst(0)]), warp(1, vec![inst(0)])],
        }];
        let mut p = params();
        p.max_resident_warps_per_sm = 1;
        assert!(matches!(
            SmCore::new(&blocks, &sched, &p, SmMemMode::Flat),
            Err(IssueError::BlockResidencyLimitExceeded {
                launch_index: 4,
                live_warps: 2,
                arch: ArchId::Sm86,
                max_resident_warps: 1,
            })
        ));

        let blocks = vec![BlockReplay {
            launch_index: 5,
            warps: vec![warp(0, vec![inst(0)])],
        }];
        p.max_resident_warps_per_sm = 0;
        assert!(matches!(
            SmCore::new(&blocks, &sched, &p, SmMemMode::Flat),
            Err(IssueError::BlockResidencyLimitExceeded {
                launch_index: 5,
                live_warps: 1,
                arch: ArchId::Sm86,
                max_resident_warps: 0,
            })
        ));
    }

    #[test]
    fn first_cta_must_fit_the_architecture_shared_memory_capacity() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
        let blocks = vec![BlockReplay {
            launch_index: 6,
            warps: vec![warp(0, vec![inst(0)])],
        }];
        assert!(matches!(
            SmCore::new_with_residency(
                &blocks,
                &sched,
                &params(),
                SharedMemoryResidency::new(64, 65),
                SmMemMode::Flat,
            ),
            Err(IssueError::BlockSharedMemoryLimitExceeded {
                launch_index: 6,
                bytes_per_cta: 65,
                arch: ArchId::Sm86,
                max_bytes_per_sm: 64,
            })
        ));
    }

    #[test]
    fn colliding_ctas_wait_then_reuse_the_physical_slot() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
        let blocks = vec![
            BlockReplay {
                launch_index: 0,
                warps: vec![warp(0, vec![inst(0)])],
            },
            BlockReplay {
                launch_index: 1,
                warps: vec![warp(0, vec![inst(0)])],
            },
        ];
        let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
        assert_eq!(
            core.next_block, 1,
            "colliding CTA was admitted concurrently"
        );
        assert_eq!(core.block_states.len(), 1);
        assert_eq!(core.resident_warp_slots, HashSet::from([0]));

        let mut out = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Bounded(1), &mut out).unwrap(),
            Advance::Idle { wake_at: Some(1) }
        );
        assert_eq!(
            core.next_block, 2,
            "retirement did not admit the waiting CTA"
        );
        assert_eq!(core.block_states.len(), 2);
        assert_eq!(core.resident_warp_slots, HashSet::from([0]));

        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Done
        );
        assert!(core.resident_warp_slots.is_empty());
        assert_eq!(core.instructions_issued(), 2);
    }

    #[test]
    fn greedy_hold_then_youngest_switch() {
        // Two warps on the same sub-core (slots 0 and 4); everyone ready:
        // one issue per cycle on the shared sub-core → 4 cycles.
        let sched = schedule(vec![
            (0, slot(cb(1), OpClass::Other, "IADD3")),
            (16, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![
                warp(0, vec![inst(0), inst(16)]),
                warp(4, vec![inst(0), inst(16)]),
            ],
        }];
        let r = simulate_sm(&blocks, &sched, &params(), None).unwrap();
        assert_eq!(r.instructions_issued, 4);
        assert_eq!(r.cycles, 4);
    }

    #[test]
    fn warps_on_distinct_subcores_issue_in_parallel() {
        let sched = schedule(vec![
            (0, slot(cb(1), OpClass::Other, "IADD3")),
            (16, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: (0..4).map(|s| warp(s, vec![inst(0), inst(16)])).collect(),
        }];
        let r = simulate_sm(&blocks, &sched, &params(), None).unwrap();
        assert_eq!(r.instructions_issued, 8);
        assert_eq!(r.cycles, 2);
    }

    #[test]
    fn block_barrier_synchronizes_block() {
        // Warp 0 reaches BAR early and must wait for warp 1, which has an
        // extra pre-barrier instruction with a 12-cycle stall.
        let sched = schedule(vec![
            (0, slot(cb(1), OpClass::Other, "IADD3")),
            (8, slot(cb(12), OpClass::Other, "FFMA")),
            (
                16,
                slot(cb(1), OpClass::BlockBarrier, "BAR.SYNC.DEFER_BLOCKING"),
            ),
            (32, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![
                warp(0, vec![inst(0), inst(16), inst(32)]),
                warp(1, vec![inst(0), inst(8), inst(16), inst(32)]),
            ],
        }];
        let r = simulate_sm(&blocks, &sched, &params(), None).unwrap();
        assert_eq!(r.instructions_issued, 7);
        assert!(r.cycles >= 15, "cycles={}", r.cycles);
        assert_eq!(r.readiness.cta_barriers.generations, 1);
        assert_eq!(r.readiness.cta_barriers.total_arrival_spread_cycles, 12);
        assert_eq!(r.readiness.cta_barriers.max_arrival_spread_cycles, 12);
        assert_eq!(r.readiness.cta_barriers.total_rendezvous_warp_cycles, 14);
        assert_eq!(r.readiness.limiting_cycles.cta_barrier_rendezvous, 14);
    }

    #[test]
    fn sm_shared_barrier_service_queues_parallel_smsp_dispatches() {
        let sched = schedule(vec![
            (0, barrier_pipeline_slot(cb(1))),
            (16, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: (0..4)
                .map(|warp_slot| warp(warp_slot, vec![inst(0), inst(16)]))
                .collect(),
        }];
        let result = simulate_sm(&blocks, &sched, &params(), None).unwrap();

        assert_eq!(result.instructions_issued, 8);
        assert_eq!(result.cycles, 10);
        assert_eq!(result.readiness.schema_version(), 2);
        assert_eq!(result.readiness.barrier_pipeline.instructions_issued, 4);
        assert_eq!(
            result.readiness.barrier_pipeline.issue_server_busy_cycles,
            8
        );
        assert_eq!(result.readiness.barrier_pipeline.generations_completed, 1);
        assert_eq!(
            result
                .readiness
                .barrier_pipeline
                .total_generation_issue_span_cycles,
            0
        );
        assert_eq!(
            result
                .readiness
                .barrier_pipeline
                .max_generation_issue_span_cycles,
            0
        );
        assert_eq!(
            result
                .readiness
                .barrier_pipeline
                .total_generation_arrival_span_cycles,
            6
        );
        assert_eq!(
            result
                .readiness
                .barrier_pipeline
                .max_generation_arrival_span_cycles,
            6
        );
        assert_eq!(result.readiness.cta_barriers.total_arrival_spread_cycles, 6);
        assert!(result.readiness.constraint_exposure_cycles.pipeline_barrier > 0);
        assert!(
            result
                .readiness
                .constraint_exposure_cycles
                .cta_barrier_rendezvous
                > 0
        );
    }

    #[test]
    fn consecutive_generations_reuse_the_same_shared_barrier_service() {
        let sched = schedule(vec![
            (0, barrier_pipeline_slot(cb(1))),
            (16, barrier_pipeline_slot(cb(1))),
            (32, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: (0..4)
                .map(|warp_slot| warp(warp_slot, vec![inst(0), inst(16), inst(32)]))
                .collect(),
        }];
        let result = simulate_sm(&blocks, &sched, &params(), None).unwrap();

        assert_eq!(result.instructions_issued, 12);
        assert_eq!(result.cycles, 19);
        assert_eq!(result.readiness.barrier_pipeline.instructions_issued, 8);
        assert_eq!(
            result.readiness.barrier_pipeline.issue_server_busy_cycles,
            16
        );
        assert_eq!(result.readiness.barrier_pipeline.generations_completed, 2);
        assert_eq!(result.readiness.barrier_pipeline.generation_rearm_events, 0);
        assert_eq!(
            result
                .readiness
                .barrier_pipeline
                .total_generation_issue_span_cycles,
            0
        );
        assert_eq!(
            result
                .readiness
                .barrier_pipeline
                .total_generation_arrival_span_cycles,
            12
        );
    }

    #[test]
    fn forward_branch_resolution_extends_only_the_issuing_warps_next_frontier() {
        let branch_control = ControlBits::new(5, true, 7, 7, 0, 0).unwrap();
        let sched = schedule(vec![
            (0, branch_pipeline_slot(branch_control)),
            (16, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0), inst(16)])],
        }];
        let result = simulate_sm(&blocks, &sched, &params(), None).unwrap();

        assert_eq!(result.instructions_issued, 2);
        assert_eq!(result.cycles, 11);
        assert_eq!(result.readiness.schema_version(), 3);
        assert_eq!(
            result.readiness.control_pipeline.branch_instructions_issued,
            1
        );
        assert_eq!(
            result.readiness.control_pipeline.branch_resolution_cycles,
            10
        );
        assert_eq!(result.readiness.limiting_cycles.pipeline_branch, 10);
    }

    #[test]
    fn readiness_aggregation_overflow_is_transactional() {
        assert_eq!(ReadinessDiagnostics::default().schema_version(), 1);
        let mut aggregate = ReadinessDiagnostics::default();
        aggregate.limiting_cycles.control_stall = u64::MAX;
        aggregate.cta_barriers.max_arrival_spread_cycles = 19;
        let before = aggregate.clone();

        let mut addend = ReadinessDiagnostics::default();
        addend.limiting_cycles.control_stall = 1;
        addend.cta_barriers.max_arrival_spread_cycles = 23;

        assert!(matches!(
            aggregate.add_checked(&addend),
            Err(ReadinessCounterError::Overflow {
                field: "readiness.control_stall",
                accumulated: u64::MAX,
                addend: 1,
            })
        ));
        assert_eq!(aggregate, before);
    }

    #[test]
    fn scalar_global_output_wavefront_projection_preserves_atomic_aliases() {
        let unique_line_sectors: Vec<i64> = (0..32).map(|value| value * 4).collect();
        assert_eq!(
            projected_global_memory_issue(
                AccessKind::GlobalStore { off: 0, len: 32 },
                &unique_line_sectors,
            ),
            Some((GlobalMemoryIssueClass::Store, 32, 8, 32, 0))
        );
        assert_eq!(
            projected_global_memory_issue(
                AccessKind::GlobalReduction {
                    off: 0,
                    len: 4,
                    l2_requests: 1,
                },
                &[0, 1, 2, 3],
            ),
            Some((GlobalMemoryIssueClass::Reduction, 4, 1, 1, 1))
        );
        assert_eq!(
            projected_global_memory_issue(
                AccessKind::GlobalReduction {
                    off: 0,
                    len: 32,
                    l2_requests: 32,
                },
                &[7; 32],
            ),
            Some((GlobalMemoryIssueClass::Reduction, 32, 32, 1, 32))
        );
        assert_eq!(
            projected_global_memory_issue(
                AccessKind::GlobalBypass {
                    off: 0,
                    len: 1,
                    token: CacheOpToken::Ldgsts,
                    l2_requests: 0,
                },
                &[0],
            ),
            None
        );
    }

    #[test]
    fn scalar_global_operand_read_origins_are_operation_specific() {
        assert_eq!(
            operand_read_dependency_origin(eligible(0, 1)),
            DependencyOrigin::GlobalLoadOperandRead
        );
        assert_eq!(
            operand_read_dependency_origin(AccessKind::GlobalStore { off: 0, len: 1 }),
            DependencyOrigin::GlobalStoreOperandRead
        );
        assert_eq!(
            operand_read_dependency_origin(AccessKind::GlobalBypass {
                off: 0,
                len: 1,
                token: CacheOpToken::Atom,
                l2_requests: 1,
            }),
            DependencyOrigin::GlobalAtomicOperandRead
        );
        assert_eq!(
            operand_read_dependency_origin(AccessKind::GlobalReduction {
                off: 0,
                len: 1,
                l2_requests: 1,
            }),
            DependencyOrigin::GlobalReductionOperandRead
        );
        assert_eq!(
            operand_read_dependency_origin(async_copy_inst(0, 0, 1).kind),
            DependencyOrigin::Read
        );
    }

    #[test]
    fn global_memory_issue_diagnostics_aggregate_and_overflow_transactionally() {
        let mut aggregate = ReadinessDiagnostics::default();
        aggregate
            .global_memory_issue
            .observe(GlobalMemoryIssueClass::Load, 4, 1, 1, 0, false, 10)
            .unwrap();
        aggregate
            .global_memory_issue
            .observe(GlobalMemoryIssueClass::Store, 32, 8, 32, 0, true, 30)
            .unwrap();
        let mut addend = ReadinessDiagnostics::default();
        addend
            .global_memory_issue
            .observe(GlobalMemoryIssueClass::Reduction, 32, 32, 1, 32, true, 7)
            .unwrap();
        aggregate.add_checked(&addend).unwrap();
        assert_eq!(aggregate.global_memory_issue.load_instructions, 1);
        assert_eq!(aggregate.global_memory_issue.store_ideal_wavefronts, 8);
        assert_eq!(aggregate.global_memory_issue.reduction_ideal_wavefronts, 32);
        assert_eq!(aggregate.global_memory_issue.store_unique_lines, 32);
        assert_eq!(aggregate.global_memory_issue.reduction_l2_requests, 32);
        assert_eq!(
            aggregate
                .global_memory_issue
                .store_operand_read_barrier_instructions,
            1
        );
        assert_eq!(
            aggregate
                .global_memory_issue
                .reduction_operand_read_barrier_instructions,
            1
        );
        assert_eq!(aggregate.global_memory_issue.total_issue_span_cycles, 20);
        assert_eq!(aggregate.global_memory_issue.max_issue_span_cycles, 20);
        assert_eq!(aggregate.schema_version(), 8);

        aggregate.global_memory_issue.store_sectors = u64::MAX;
        addend.global_memory_issue.store_sectors = 1;
        let before = aggregate.clone();
        assert!(matches!(
            aggregate.add_checked(&addend),
            Err(ReadinessCounterError::Overflow {
                field: "readiness.global_memory_issue.store_sectors",
                accumulated: u64::MAX,
                addend: 1,
            })
        ));
        assert_eq!(aggregate, before);
    }

    fn service_candidate(
        candidate_id: u8,
        scope: MemoryServiceCandidateScope,
    ) -> MemoryServiceCandidate {
        MemoryServiceCandidate {
            candidate_id,
            scope,
            operation: MemoryServiceCandidateOperation::Store,
            ideal_wavefront_units: 2,
            ideal_wavefront_cycles: 3,
            unique_line_units: 1,
            unique_line_cycles: 1,
            completion_latency_cycles: 5,
        }
    }

    fn admission_candidate(
        candidate_id: u8,
        scope: MemoryServiceCandidateScope,
    ) -> MemoryAdmissionCandidate {
        MemoryAdmissionCandidate {
            candidate_id,
            scope,
            operation: MemoryServiceCandidateOperation::Store,
            queue_capacity_instructions: 1,
            admission_latency_cycles: 8,
            l2_hit_sector_units: 1,
            l2_hit_sector_cycles: 1,
            l2_miss_sector_units: 1,
            l2_miss_sector_cycles: 4,
            l2_miss_service_width_sectors: 0,
            membar_waits_for_service_completion: false,
            write_completion_latency_cycles: 0,
            write_completion_request_cycles: 0,
            write_completion_region_cycles: 0,
            write_completion_parallel_lanes: 0,
        }
    }

    #[test]
    fn memory_admission_candidate_validation_is_bounded_and_hit_miss_ordered() {
        let mut candidates = [admission_candidate(0, MemoryServiceCandidateScope::Sm);
            MAX_MEMORY_ADMISSION_CANDIDATES + 1];
        for (index, candidate) in candidates.iter_mut().enumerate() {
            candidate.candidate_id = index as u8;
        }
        assert_eq!(
            MemoryAdmissionCandidateState::checked(&candidates),
            Err(MemoryAdmissionCandidateError::TooMany {
                count: MAX_MEMORY_ADMISSION_CANDIDATES + 1,
                maximum: MAX_MEMORY_ADMISSION_CANDIDATES,
            })
        );
        let mut invalid = admission_candidate(3, MemoryServiceCandidateScope::Sm);
        invalid.queue_capacity_instructions = 0;
        assert_eq!(
            invalid.validate(),
            Err(MemoryAdmissionCandidateError::CapacityOutOfRange {
                candidate_id: 3,
                capacity: 0,
                maximum: MAX_MEMORY_ADMISSION_QUEUE_CAPACITY,
            })
        );

        invalid = admission_candidate(6, MemoryServiceCandidateScope::Sm);
        invalid.write_completion_latency_cycles = 7;
        assert_eq!(
            invalid.validate(),
            Err(
                MemoryAdmissionCandidateError::CompletionLatencyWithoutMembar {
                    candidate_id: 6,
                    latency: 7,
                }
            )
        );
        invalid = admission_candidate(7, MemoryServiceCandidateScope::Sm);
        invalid.write_completion_request_cycles = 1;
        invalid.write_completion_parallel_lanes = 1;
        assert_eq!(
            invalid.validate(),
            Err(MemoryAdmissionCandidateError::CompletionWorkWithoutMembar { candidate_id: 7 })
        );
        invalid.membar_waits_for_service_completion = true;
        invalid.write_completion_parallel_lanes = 0;
        assert_eq!(
            invalid.validate(),
            Err(MemoryAdmissionCandidateError::CompletionLanesOutOfRange {
                candidate_id: 7,
                lanes: 0,
                maximum: MAX_WRITE_COMPLETION_PARALLEL_LANES,
            })
        );
        invalid.write_completion_request_cycles = 0;
        invalid.write_completion_parallel_lanes = 1;
        assert_eq!(
            invalid.validate(),
            Err(MemoryAdmissionCandidateError::CompletionLanesWithoutWork {
                candidate_id: 7,
                lanes: 1,
            })
        );
        invalid = admission_candidate(4, MemoryServiceCandidateScope::Sm);
        invalid.l2_miss_sector_cycles = 1;
        invalid.l2_hit_sector_cycles = 2;
        assert_eq!(
            invalid.validate(),
            Err(MemoryAdmissionCandidateError::MissRateFasterThanHit { candidate_id: 4 })
        );

        invalid = admission_candidate(5, MemoryServiceCandidateScope::Sm);
        invalid.l2_miss_service_width_sectors = MAX_MEMORY_ADMISSION_SERVICE_WIDTH_SECTORS + 1;
        assert_eq!(
            invalid.validate(),
            Err(MemoryAdmissionCandidateError::ServiceWidthOutOfRange {
                candidate_id: 5,
                width: MAX_MEMORY_ADMISSION_SERVICE_WIDTH_SECTORS + 1,
                maximum: MAX_MEMORY_ADMISSION_SERVICE_WIDTH_SECTORS,
            })
        );
    }

    #[test]
    fn memory_admission_miss_width_is_independent_from_asymptotic_rate() {
        assert_eq!(candidate_rate_cycles_with_width(5, 1, 10, 0).unwrap(), 50);
        assert_eq!(candidate_rate_cycles_with_width(5, 1, 10, 1).unwrap(), 50);
        assert_eq!(candidate_rate_cycles_with_width(5, 1, 10, 4).unwrap(), 80);
        assert_eq!(candidate_rate_cycles_with_width(8, 1, 10, 4).unwrap(), 80);

        let mut candidate = admission_candidate(6, MemoryServiceCandidateScope::Sm);
        candidate.l2_miss_service_width_sectors = 4;
        let mut state = MemoryAdmissionCandidateState::checked(&[candidate]).unwrap();
        state
            .observe(0, 0, MemoryOperation::Store, 0, 5, 0)
            .unwrap();
        let result = state.diagnostics().entries()[0];
        assert_eq!(result.service_busy_cycles, 32);
        assert_eq!(result.terminal_service_completion_time, 32);

        let mut completion_candidate = candidate;
        completion_candidate.membar_waits_for_service_completion = true;
        completion_candidate.write_completion_latency_cycles = 5;
        let mut feedback = MemoryAdmissionFeedbackState::checked(completion_candidate).unwrap();
        feedback
            .reserve(0, 3, 4, AccessKind::GlobalStore { off: 0, len: 1 }, 0)
            .unwrap();
        feedback
            .complete(
                AdmissionRequestWitness::new(3, 4, 0, 0, 1).unwrap(),
                MemoryOperation::Store,
                0,
                5,
                1,
                1,
                0,
            )
            .unwrap();
        assert_eq!(feedback.diagnostics().service_busy_cycles, 32);
        assert_eq!(feedback.diagnostics().terminal_service_completion_time, 32);

        feedback
            .reserve(1, 4, 5, AccessKind::GlobalStore { off: 0, len: 1 }, 0)
            .unwrap();
        feedback
            .complete(
                AdmissionRequestWitness::new(4, 5, 1, 0, 1).unwrap(),
                MemoryOperation::Store,
                0,
                5,
                1,
                1,
                0,
            )
            .unwrap();
        let membar = slot(cb(1), OpClass::Other, "MEMBAR.SC.GPU");
        let ordinary = slot(cb(1), OpClass::Other, "IADD3");
        assert_eq!(feedback.completion_ready_at(3, &membar), Some(37));
        assert_eq!(feedback.completion_ready_at(4, &membar), Some(69));
        assert_eq!(feedback.completion_ready_at(3, &ordinary), Some(0));
        assert_eq!(feedback.diagnostics().terminal_write_completion_time, 69);

        let mut parallel_candidate = admission_candidate(8, MemoryServiceCandidateScope::Sm);
        parallel_candidate.l2_miss_sector_cycles = 1;
        parallel_candidate.membar_waits_for_service_completion = true;
        parallel_candidate.write_completion_request_cycles = 100;
        parallel_candidate.write_completion_parallel_lanes = 2;
        let mut parallel = MemoryAdmissionFeedbackState::checked(parallel_candidate).unwrap();
        for (warp, subcore) in [(0, 0), (1, 1)] {
            parallel
                .reserve(
                    subcore,
                    warp,
                    0,
                    AccessKind::GlobalStore { off: 0, len: 1 },
                    0,
                )
                .unwrap();
            parallel
                .complete(
                    AdmissionRequestWitness::new(warp as u32, 0, subcore as u8, 0, 1).unwrap(),
                    MemoryOperation::Store,
                    0,
                    1,
                    1,
                    1,
                    0,
                )
                .unwrap();
        }
        assert_eq!(parallel.completion_ready_at(0, &membar), Some(101));
        assert_eq!(parallel.completion_ready_at(1, &membar), Some(102));
        assert_eq!(parallel.diagnostics().write_completion_busy_cycles, 200);
        assert_eq!(parallel.diagnostics().write_completion_requests_observed, 2);
        assert_eq!(parallel.diagnostics().write_completion_regions_observed, 2);
        let readiness = ReadinessDiagnostics {
            memory_admission_feedback: parallel.diagnostics(),
            ..ReadinessDiagnostics::default()
        };
        assert_eq!(readiness.schema_version(), 13);
    }

    #[test]
    fn legacy_admission_candidate_json_omits_zero_service_width() {
        let candidate = admission_candidate(7, MemoryServiceCandidateScope::Smsp);
        let encoded = serde_json::to_value(candidate).unwrap();
        assert!(encoded.get("l2_miss_service_width_sectors").is_none());
        assert!(encoded.get("membar_waits_for_service_completion").is_none());
        assert!(encoded.get("write_completion_latency_cycles").is_none());
        assert!(encoded.get("write_completion_request_cycles").is_none());
        assert!(encoded.get("write_completion_region_cycles").is_none());
        assert!(encoded.get("write_completion_parallel_lanes").is_none());
        let decoded: MemoryAdmissionCandidate = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, candidate);

        let mut widened = candidate;
        widened.l2_miss_service_width_sectors = 4;
        let encoded = serde_json::to_value(widened).unwrap();
        assert_eq!(encoded["l2_miss_service_width_sectors"], 4);
        assert_eq!(
            serde_json::from_value::<MemoryAdmissionCandidate>(encoded).unwrap(),
            widened
        );

        widened.membar_waits_for_service_completion = true;
        widened.write_completion_latency_cycles = 17;
        widened.write_completion_request_cycles = 4;
        widened.write_completion_parallel_lanes = 2;
        let encoded = serde_json::to_value(widened).unwrap();
        assert_eq!(encoded["membar_waits_for_service_completion"], true);
        assert_eq!(encoded["write_completion_latency_cycles"], 17);
        assert_eq!(encoded["write_completion_request_cycles"], 4);
        assert_eq!(encoded["write_completion_parallel_lanes"], 2);
        assert_eq!(
            serde_json::from_value::<MemoryAdmissionCandidate>(encoded).unwrap(),
            widened
        );
    }

    #[test]
    fn memory_admission_candidates_share_service_but_scope_queue_capacity() {
        let candidates = [
            admission_candidate(1, MemoryServiceCandidateScope::Sm),
            admission_candidate(2, MemoryServiceCandidateScope::Smsp),
        ];
        let mut state = MemoryAdmissionCandidateState::checked(&candidates).unwrap();
        state
            .observe(0, 0, MemoryOperation::Store, 0, 1, 0)
            .unwrap();
        state
            .observe(1, 1, MemoryOperation::Store, 0, 1, 0)
            .unwrap();
        state
            .observe(0, 0, MemoryOperation::Store, 0, 1, 0)
            .unwrap();
        let diagnostics = state.diagnostics();
        let sm = diagnostics.entries()[0];
        let smsp = diagnostics.entries()[1];

        assert_eq!(sm.instructions_observed, 3);
        assert_eq!(sm.l2_miss_sectors_observed, 3);
        assert_eq!(sm.service_busy_cycles, 12);
        assert_eq!(sm.downstream_queue_delay_cycles, 8);
        assert_eq!(sm.queue_full_instructions, 1);
        assert_eq!(sm.total_admission_delay_cycles, 4);
        assert_eq!(sm.maximum_admission_delay_cycles, 4);
        assert_eq!(sm.maximum_warp_admission_delay_cycles, 4);
        assert_eq!(sm.terminal_operand_read_clear_time, 12);
        assert_eq!(sm.terminal_service_completion_time, 12);

        assert_eq!(smsp.instructions_observed, 3);
        assert_eq!(smsp.service_busy_cycles, 12);
        assert_eq!(smsp.downstream_queue_delay_cycles, 12);
        assert_eq!(smsp.queue_full_instructions, 0);
        assert_eq!(smsp.total_admission_delay_cycles, 0);
        assert_eq!(smsp.maximum_warp_admission_delay_cycles, 0);
        assert_eq!(diagnostics.entries().len(), 2);
    }

    #[test]
    fn memory_admission_candidate_aggregation_is_checked_and_schema9() {
        let candidate = admission_candidate(6, MemoryServiceCandidateScope::Sm);
        let mut left = MemoryAdmissionCandidateState::checked(&[candidate]).unwrap();
        left.observe(0, 0, MemoryOperation::Store, 1, 0, 10)
            .unwrap();
        let mut right = MemoryAdmissionCandidateState::checked(&[candidate]).unwrap();
        right
            .observe(1, 0, MemoryOperation::Store, 0, 1, 20)
            .unwrap();
        let mut readiness = ReadinessDiagnostics::default();
        readiness.memory_admission_candidates = left.diagnostics();
        let mut addend = ReadinessDiagnostics::default();
        addend.memory_admission_candidates = right.diagnostics();
        readiness.add_checked(&addend).unwrap();
        let result = readiness.memory_admission_candidates.entries()[0];
        assert_eq!(result.instructions_observed, 2);
        assert_eq!(result.l2_hit_sectors_observed, 1);
        assert_eq!(result.l2_miss_sectors_observed, 1);
        assert_eq!(result.terminal_service_completion_time, 24);
        assert_eq!(readiness.schema_version(), 9);

        let mismatch = MemoryAdmissionCandidateState::checked(&[admission_candidate(
            7,
            MemoryServiceCandidateScope::Sm,
        )])
        .unwrap()
        .diagnostics();
        let mut bad = ReadinessDiagnostics::default();
        bad.memory_admission_candidates = mismatch;
        assert!(matches!(
            readiness.add_checked(&bad),
            Err(ReadinessCounterError::MemoryAdmissionCandidateMismatch { index: 0 })
        ));
    }

    #[test]
    fn feedback_admission_reservation_and_pricing_overflows_are_transactional() {
        let candidate = admission_candidate(8, MemoryServiceCandidateScope::Sm);

        let mut reserve_overflow = MemoryAdmissionFeedbackState::checked(candidate).unwrap();
        reserve_overflow.diagnostics.instructions_admitted = u64::MAX;
        let before = reserve_overflow.clone();
        assert!(matches!(
            reserve_overflow.reserve(0, 0, 0, AccessKind::GlobalStore { off: 0, len: 1 }, 0,),
            Err(IssueError::CounterOverflow {
                counter: "feedback admission instructions"
            })
        ));
        assert_eq!(reserve_overflow, before);

        let mut pricing_overflow = MemoryAdmissionFeedbackState::checked(candidate).unwrap();
        pricing_overflow
            .reserve(0, 0, 0, AccessKind::GlobalStore { off: 0, len: 1 }, 0)
            .unwrap();
        pricing_overflow.diagnostics.l2_hit_sectors_observed = u64::MAX;
        let before = pricing_overflow.clone();
        let witness = AdmissionRequestWitness::new(0, 0, 0, 0, 1).unwrap();
        assert!(matches!(
            pricing_overflow.complete(witness, MemoryOperation::Store, 1, 0, 1, 1, 0),
            Err(IssueError::CounterOverflow {
                counter: "feedback admission L2 hit sectors"
            })
        ));
        assert_eq!(pricing_overflow, before);
    }

    #[test]
    fn feedback_issue_plan_is_transactional_and_accounts_same_cycle_reservations() {
        let mut candidate = admission_candidate(9, MemoryServiceCandidateScope::Sm);
        candidate.queue_capacity_instructions = 2;
        candidate.membar_waits_for_service_completion = true;
        let mut feedback = MemoryAdmissionFeedbackState::checked(candidate).unwrap();
        let issues = [
            Some(MemoryAdmissionFeedbackIssue {
                subcore: 0,
                warp: 3,
                instruction: 7,
                kind: AccessKind::GlobalStore { off: 0, len: 1 },
                prior_clock_reads: 1,
                clock_read: true,
            }),
            Some(MemoryAdmissionFeedbackIssue {
                subcore: 1,
                warp: 4,
                instruction: 8,
                kind: AccessKind::GlobalStore { off: 0, len: 1 },
                prior_clock_reads: 0,
                clock_read: false,
            }),
            None,
            None,
        ];
        let before = feedback.clone();
        let plan = feedback.plan_issue(&issues, 11).unwrap();
        assert_eq!(feedback, before, "preflight must not mutate feedback state");
        feedback.apply_issue(plan, 11);
        assert_eq!(feedback.queues[0].len(), 2);
        assert_eq!(feedback.diagnostics.instructions_admitted, 2);
        assert_eq!(feedback.diagnostics.maximum_waiting_instructions, 2);
        assert_eq!(feedback.second_clock_issue(3), Some(11));
        assert!(feedback.take_yield_requested());

        let mut full = MemoryAdmissionFeedbackState::checked(candidate).unwrap();
        full.apply_issue(full.plan_issue(&issues, 11).unwrap(), 11);
        let overflow = [
            Some(MemoryAdmissionFeedbackIssue {
                subcore: 2,
                warp: 5,
                instruction: 9,
                kind: AccessKind::GlobalStore { off: 0, len: 1 },
                prior_clock_reads: 0,
                clock_read: false,
            }),
            None,
            None,
            None,
        ];
        let before = full.clone();
        assert!(matches!(
            full.plan_issue(&overflow, 11),
            Err(IssueError::MemoryAdmissionFeedbackQueueFull {
                subcore: 2,
                time: 11,
            })
        ));
        assert_eq!(full, before);
    }

    #[test]
    fn memory_service_candidate_validation_is_bounded_and_unambiguous() {
        let mut candidates = [service_candidate(0, MemoryServiceCandidateScope::Sm);
            MAX_MEMORY_SERVICE_CANDIDATES + 1];
        for (index, candidate) in candidates.iter_mut().enumerate() {
            candidate.candidate_id = index as u8;
        }
        assert_eq!(
            MemoryServiceCandidateState::checked(&candidates),
            Err(MemoryServiceCandidateError::TooMany {
                count: MAX_MEMORY_SERVICE_CANDIDATES + 1,
                maximum: MAX_MEMORY_SERVICE_CANDIDATES,
            })
        );
        assert_eq!(
            MemoryServiceCandidateState::checked(&[
                service_candidate(7, MemoryServiceCandidateScope::Sm),
                service_candidate(7, MemoryServiceCandidateScope::Smsp),
            ]),
            Err(MemoryServiceCandidateError::DuplicateId { candidate_id: 7 })
        );
        let mut invalid = service_candidate(8, MemoryServiceCandidateScope::Sm);
        invalid.unique_line_units = 0;
        assert_eq!(
            invalid.validate(),
            Err(MemoryServiceCandidateError::InvalidRate {
                candidate_id: 8,
                dimension: "unique_line",
                units: 0,
                cycles: 1,
            })
        );
        invalid.unique_line_cycles = 0;
        invalid.ideal_wavefront_units = 0;
        invalid.ideal_wavefront_cycles = 0;
        assert_eq!(
            invalid.validate(),
            Err(MemoryServiceCandidateError::NoServiceWork { candidate_id: 8 })
        );
    }

    #[test]
    fn memory_service_candidates_stream_at_issue_time_by_selected_scope() {
        let candidates = [
            service_candidate(1, MemoryServiceCandidateScope::Sm),
            service_candidate(2, MemoryServiceCandidateScope::Smsp),
        ];
        let mut state = MemoryServiceCandidateState::checked(&candidates).unwrap();
        state
            .observe(0, GlobalMemoryIssueClass::Store, 4, 2, 10)
            .unwrap();
        state
            .observe(1, GlobalMemoryIssueClass::Store, 4, 2, 12)
            .unwrap();
        let diagnostics = state.diagnostics();
        let sm = diagnostics.entries()[0];
        let smsp = diagnostics.entries()[1];

        assert_eq!(sm.instructions_observed, 2);
        assert_eq!(sm.ideal_wavefronts_observed, 8);
        assert_eq!(sm.unique_lines_observed, 4);
        assert_eq!(sm.busy_cycles, 16);
        assert_eq!(sm.queue_delay_cycles, 6);
        assert_eq!(sm.terminal_completion_time, 31);
        assert_eq!(smsp.instructions_observed, 2);
        assert_eq!(smsp.busy_cycles, 16);
        assert_eq!(smsp.queue_delay_cycles, 0);
        assert_eq!(smsp.terminal_completion_time, 25);
    }

    #[test]
    fn memory_service_candidate_aggregation_is_transactional_and_uses_max_terminal() {
        let candidate = service_candidate(3, MemoryServiceCandidateScope::Sm);
        let mut left = MemoryServiceCandidateState::checked(&[candidate]).unwrap();
        left.observe(0, GlobalMemoryIssueClass::Store, 4, 2, 10)
            .unwrap();
        let mut right = MemoryServiceCandidateState::checked(&[candidate]).unwrap();
        right
            .observe(0, GlobalMemoryIssueClass::Store, 4, 2, 40)
            .unwrap();
        let mut aggregate = left.diagnostics();
        aggregate.add_checked(&right.diagnostics()).unwrap();
        assert_eq!(aggregate.entries()[0].instructions_observed, 2);
        assert_eq!(aggregate.entries()[0].busy_cycles, 16);
        assert_eq!(aggregate.entries()[0].terminal_completion_time, 53);

        let before = aggregate.clone();
        let mismatch = MemoryServiceCandidateState::checked(&[service_candidate(
            4,
            MemoryServiceCandidateScope::Sm,
        )])
        .unwrap()
        .diagnostics();
        assert_eq!(
            aggregate.add_checked(&mismatch),
            Err(ReadinessCounterError::MemoryServiceCandidateMismatch { index: 0 })
        );
        assert_eq!(aggregate, before);

        let mut readiness = ReadinessDiagnostics::default();
        readiness.memory_service_candidates = aggregate;
        assert_eq!(readiness.schema_version(), 7);
    }

    #[test]
    fn result_writeback_diagnostics_aggregate_and_overflow_transactionally() {
        let mut aggregate = ReadinessDiagnostics::default();
        aggregate.result_writeback.fixed_over_l1tex_collisions = 2;
        aggregate.result_writeback.l1tex_deferral_cycles = 2;
        let mut addend = ReadinessDiagnostics::default();
        addend.result_writeback.fixed_over_l1tex_collisions = 3;
        addend.result_writeback.l1tex_deferral_cycles = 3;
        aggregate.add_checked(&addend).unwrap();
        assert_eq!(aggregate.result_writeback.fixed_over_l1tex_collisions, 5);
        assert_eq!(aggregate.result_writeback.l1tex_deferral_cycles, 5);
        assert_eq!(aggregate.schema_version(), 4);

        aggregate.result_writeback.l1tex_deferral_cycles = u64::MAX;
        let before = aggregate.clone();
        assert!(matches!(
            aggregate.add_checked(&addend),
            Err(ReadinessCounterError::Overflow {
                field: "readiness.result_writeback.l1tex_deferral_cycles",
                accumulated: u64::MAX,
                addend: 3,
            })
        ));
        assert_eq!(aggregate, before);
    }

    #[test]
    fn finished_warp_does_not_block_barrier() {
        let sched = schedule(vec![
            (0, slot(cb(1), OpClass::Other, "IADD3")),
            (
                16,
                slot(cb(1), OpClass::BlockBarrier, "BAR.SYNC.DEFER_BLOCKING"),
            ),
            (32, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![
                warp(0, vec![inst(0), inst(16), inst(32)]),
                warp(1, vec![inst(0)]),
            ],
        }];
        let r = simulate_sm(&blocks, &sched, &params(), None).unwrap();
        assert_eq!(r.instructions_issued, 4);
    }

    #[test]
    fn residency_cap_forces_waves() {
        // Block 0 on sub-cores 0/1, block 1 on sub-cores 2/3 so the
        // uncapped run can use all four sub-cores at once.
        let sched = schedule(vec![
            (0, slot(cb(1), OpClass::Other, "IADD3")),
            (16, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let mk_block = |ix: i64, slot0: i32| BlockReplay {
            launch_index: ix,
            warps: vec![
                warp(slot0, vec![inst(0), inst(16)]),
                warp(slot0 + 1, vec![inst(0), inst(16)]),
            ],
        };
        let blocks = vec![mk_block(0, 0), mk_block(1, 2)];
        let mut p = params();
        p.max_resident_warps_per_sm = 2;
        let r = simulate_sm(&blocks, &sched, &p, None).unwrap();
        assert_eq!(r.instructions_issued, 8);
        assert!(r.cycles >= 4, "cycles={}", r.cycles);
        let r2 = simulate_sm(&blocks, &sched, &params(), None).unwrap();
        assert_eq!(r2.cycles, 2);
    }

    #[test]
    fn shared_memory_capacity_forces_waves_and_releases_on_retirement() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
        let blocks = vec![
            BlockReplay {
                launch_index: 0,
                warps: vec![warp(0, vec![inst(0)])],
            },
            BlockReplay {
                launch_index: 1,
                warps: vec![warp(1, vec![inst(0)])],
            },
        ];
        let residency = SharedMemoryResidency::new(64, 64);
        let mut core =
            SmCore::new_with_residency(&blocks, &sched, &params(), residency, SmMemMode::Flat)
                .unwrap();
        assert_eq!(core.next_block, 1);
        assert_eq!(core.resident_shared_memory_bytes, 64);
        assert_eq!(core.resident_warp_slots, HashSet::from([0]));

        let mut out = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Bounded(1), &mut out).unwrap(),
            Advance::Idle { wake_at: Some(1) }
        );
        assert_eq!(core.next_block, 2);
        assert_eq!(core.resident_shared_memory_bytes, 64);
        assert_eq!(core.resident_warp_slots, HashSet::from([1]));

        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Done
        );
        assert_eq!(core.resident_shared_memory_bytes, 0);
        assert!(core.resident_warp_slots.is_empty());
        assert_eq!(core.instructions_issued(), 2);
    }

    #[test]
    fn unknown_pc_is_typed_error() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "IADD3"))]);
        let blocks = vec![BlockReplay {
            launch_index: 7,
            warps: vec![warp(3, vec![inst(0), inst(999)])],
        }];
        let err = simulate_sm(&blocks, &sched, &params(), None).unwrap_err();
        assert!(matches!(err, IssueError::PcNotInSchedule { pc: 999, .. }));
    }

    // ------------------------------------------------------------------
    // SmCore event-mode (L1Shared) tests: a manual driver stands in for
    // the system layer.
    // ------------------------------------------------------------------

    fn shared_core<'a>(
        blocks: &'a [BlockReplay],
        sched: &'a FunctionSchedule,
        p: &IssueModelParams,
    ) -> SmCore<'a> {
        SmCore::new(blocks, sched, p, SmMemMode::L1Shared(l1_config())).unwrap()
    }

    #[test]
    fn feedback_admission_requires_the_issue_models_operand_visibility_latency() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0)])],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let matching = admission_candidate(9, MemoryServiceCandidateScope::Sm);
        core.install_memory_admission_feedback(matching).unwrap();
        assert_eq!(
            core.memory_admission_feedback
                .as_ref()
                .unwrap()
                .diagnostics()
                .candidate,
            matching
        );

        let mut mismatched = matching;
        mismatched.admission_latency_cycles = 7;
        assert!(matches!(
            core.install_memory_admission_feedback(mismatched),
            Err(IssueError::MemoryAdmissionFeedbackLatencyMismatch {
                candidate_id: 9,
                configured_cycles: 7,
                issue_model_cycles: 8,
            })
        ));
        assert_eq!(
            core.memory_admission_feedback
                .as_ref()
                .unwrap()
                .diagnostics()
                .candidate,
            matching,
            "a rejected replacement must preserve the installed candidate"
        );
    }

    #[test]
    fn sm_scoped_same_cycle_capacity_denial_repicks_non_candidate_work() {
        let sched = schedule(vec![
            (0, slot(cb(1), OpClass::Other, "STG.E")),
            (16, slot(cb(1), OpClass::Other, "IADD3")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![
                // Older ready work on SMSP 1.
                warp(1, vec![inst(16)]),
                // Younger STORE on the same SMSP wins the initial CGGTY pick,
                // but loses the SM-wide one-entry queue to SMSP 0.
                WarpReplay {
                    warp_slot: 5,
                    insts: vec![ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalStore { off: 0, len: 1 },
                    }],
                    sectors: vec![4],
                },
                WarpReplay {
                    warp_slot: 0,
                    insts: vec![ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalStore { off: 0, len: 1 },
                    }],
                    sectors: vec![0],
                },
            ],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        core.install_memory_admission_feedback(admission_candidate(
            10,
            MemoryServiceCandidateScope::Sm,
        ))
        .unwrap();
        let mut requests = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut requests)
                .unwrap(),
            Advance::Idle { wake_at: Some(1) }
        );
        assert_eq!(core.instructions_issued(), 2);
        assert_eq!(requests.len(), 1);
        assert!(
            core.warps[0].done,
            "the non-candidate fallback must dispatch"
        );
        assert!(
            !core.warps[1].done,
            "the capacity-denied younger STORE must remain resident"
        );
        assert!(core.warps[2].done, "SMSP 0 owns the queue reservation");
    }

    #[test]
    fn installed_memory_service_candidate_observes_real_issue_without_delaying_core() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "STG.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: AccessKind::GlobalStore { off: 0, len: 4 },
                }],
                sectors: vec![0, 1, 2, 3],
            }],
        }];
        let candidate = service_candidate(9, MemoryServiceCandidateScope::Sm);
        let mut core = shared_core(&blocks, &sched, &params());
        core.install_memory_service_candidates(&[candidate])
            .unwrap();
        let mut requests = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut requests)
                .unwrap(),
            Advance::Done
        );
        let result = core.result();
        assert_eq!(result.cycles, 1, "candidate must not delay the issue clock");
        assert_eq!(result.readiness.schema_version(), 7);
        let observed = result.readiness.memory_service_candidates.entries()[0];
        assert_eq!(observed.instructions_observed, 1);
        assert_eq!(observed.ideal_wavefronts_observed, 1);
        assert_eq!(observed.unique_lines_observed, 1);
        assert_eq!(observed.busy_cycles, 3);
        assert_eq!(observed.queue_delay_cycles, 0);
        assert_eq!(observed.terminal_completion_time, 8);
    }

    #[test]
    fn global_store_operand_read_barrier_has_distinct_schema8_attribution() {
        let mut producer = cb(1);
        producer.read_barrier = Some(barrier(0));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000001);
        let sched = schedule(vec![
            (0, slot(producer, OpClass::Other, "STG.E.STRONG.SM")),
            (16, slot(consumer, OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalStore { off: 0, len: 1 },
                    },
                    inst(16),
                ],
                sectors: vec![0],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut requests = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut requests)
                .unwrap(),
            Advance::Done
        );
        let result = core.result();
        assert_eq!(result.cycles, 9);
        assert_eq!(result.readiness.schema_version(), 8);
        assert_eq!(
            result
                .readiness
                .global_memory_issue
                .store_operand_read_barrier_instructions,
            1
        );
        assert_eq!(
            result
                .readiness
                .limiting_cycles
                .dependence_counter_global_store_operand_read,
            8
        );
        assert_eq!(result.readiness.limiting_cycles.dependence_counter_read, 0);
    }

    #[test]
    fn predicated_off_memory_creates_no_dependency_producer() {
        let mut producer = cb(1);
        producer.read_barrier = Some(barrier(0));
        producer.write_barrier = Some(barrier(1));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000011);
        let sched = schedule(vec![
            (0, slot(producer, OpClass::Other, "STG.E.STRONG.SM")),
            (16, slot(consumer, OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: AccessKind::PredicatedOffMemory,
                    },
                    inst(16),
                ],
                sectors: Vec::new(),
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut requests = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut requests)
                .unwrap(),
            Advance::Done
        );
        assert!(requests.is_empty());
        let result = core.result();
        assert_eq!(result.cycles, 2);
        assert!(result.unmodeled_tails.is_empty());
        assert!(result.readiness.schema_version() < 8);
        assert_eq!(result.readiness.global_memory_issue.store_instructions, 0);
        assert_eq!(
            result
                .readiness
                .global_memory_issue
                .store_operand_read_barrier_instructions,
            0
        );
        assert_eq!(result.readiness.limiting_cycles.dependence_counter_read, 0);
        assert_eq!(
            result
                .readiness
                .limiting_cycles
                .dependence_counter_global_store_operand_read,
            0
        );
    }

    #[test]
    fn pending_l1tex_writeback_keeps_origin_when_read_barrier_reuses_counter() {
        let mut producer = cb(1);
        producer.write_barrier = Some(barrier(0));
        producer.read_barrier = Some(barrier(0));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000001);
        let sched = schedule(vec![
            (0, slot(producer, OpClass::Other, "LDG.E.CA")),
            (16, slot(consumer, OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: eligible(0, 1),
                    },
                    inst(16),
                ],
                sectors: vec![0],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut requests = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut requests)
                .unwrap(),
            Advance::Idle { wake_at: None }
        );
        assert_eq!(requests.len(), 1);
        let mut sink = Vec::new();
        core.on_response(requests[0].reply, 300, &mut sink).unwrap();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut sink).unwrap(),
            Advance::Done
        );
        let result = core.result();
        assert_eq!(result.cycles, 301);
        assert_eq!(result.readiness.schema_version(), 8);
        assert_eq!(
            result
                .readiness
                .global_memory_issue
                .load_operand_read_barrier_instructions,
            1
        );
        assert_eq!(
            result.readiness.limiting_cycles.dependence_counter_l1tex,
            300
        );
        assert_eq!(
            result
                .readiness
                .limiting_cycles
                .dependence_counter_global_load_operand_read,
            0
        );

        // Synthetic latency inversion: if the response wins before the fixed
        // operand-read clear, it must not steal the later read attribution.
        let mut early_params = params();
        early_params.l1_hit_latency = 1;
        let early_l1 = L1Config {
            hit_latency: 1,
            ..l1_config()
        };
        let mut early = SmCore::new(
            &blocks,
            &sched,
            &early_params,
            SmMemMode::L1Shared(early_l1),
        )
        .unwrap();
        let mut early_requests = Vec::new();
        assert_eq!(
            early
                .advance(ReplayHorizon::Unbounded, &mut early_requests)
                .unwrap(),
            Advance::Idle { wake_at: None }
        );
        let mut early_sink = Vec::new();
        early
            .on_response(early_requests[0].reply, 1, &mut early_sink)
            .unwrap();
        assert_eq!(
            early
                .advance(ReplayHorizon::Unbounded, &mut early_sink)
                .unwrap(),
            Advance::Done
        );
        let early_result = early.result();
        assert_eq!(early_result.cycles, 9);
        assert_eq!(
            early_result
                .readiness
                .limiting_cycles
                .dependence_counter_global_load_operand_read,
            8
        );
        assert_eq!(
            early_result
                .readiness
                .limiting_cycles
                .dependence_counter_l1tex,
            0
        );
    }

    #[test]
    fn pending_l1tex_writeback_beats_later_fast_producer_origin() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let mut fixed = cb(1);
        fixed.write_barrier = Some(barrier(0));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000001);
        let sched = schedule(vec![
            (0, slot(load, OpClass::Other, "LDG.E.CA")),
            (16, slot(fixed, OpClass::Other, "IADD3")),
            (32, slot(consumer, OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: eligible(0, 1),
                    },
                    inst(16),
                    inst(32),
                ],
                sectors: vec![0],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut requests = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut requests)
                .unwrap(),
            Advance::Idle { wake_at: None }
        );
        assert_eq!(requests.len(), 1);
        let mut sink = Vec::new();
        core.on_response(requests[0].reply, 300, &mut sink).unwrap();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut sink).unwrap(),
            Advance::Done
        );
        let result = core.result();
        assert_eq!(result.cycles, 301);
        assert_eq!(
            result.readiness.limiting_cycles.dependence_counter_l1tex,
            299
        );
        assert_eq!(
            result
                .readiness
                .limiting_cycles
                .dependence_counter_non_memory_write,
            0
        );
    }

    #[test]
    fn shared_cacheable_async_copy_completes_all_same_line_fill_waiters() {
        let sched = schedule(vec![
            (0, slot(cb(1), OpClass::Other, "LDGSTS.E")),
            (16, async_commit_slot()),
            (32, async_wait_slot(0)),
            (48, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![async_copy_inst(0, 0, 2), inst(16), inst(32), inst(48)],
                sectors: vec![40, 41],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut requests = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut requests)
                .unwrap(),
            Advance::Idle { wake_at: None }
        );
        assert_eq!(requests.len(), 1, "same-line sectors fold into one fill");
        assert_eq!(requests[0].source, L2Source::L1Fill);
        assert_eq!(requests[0].sector_mask, 0b0011);
        let ReplyTo::L1Fill { fill, .. } = requests[0].reply else {
            panic!("cacheable LDGSTS source must ride an L1 fill")
        };
        let group = core.async_copies[0]
            .state
            .as_deref()
            .unwrap()
            .committed_groups[0];
        assert_eq!(group.pending_sources, 2);
        let MemModel::L1Shared(st) = &core.mem else {
            panic!("expected shared mode")
        };
        assert!(st.fill_waiters.contains_key(&fill));

        let mut emitted = Vec::new();
        core.on_response(requests[0].reply, 300, &mut emitted)
            .unwrap();
        assert!(emitted.is_empty());
        assert_eq!(
            core.async_copies[0]
                .state
                .as_deref()
                .unwrap()
                .committed_groups[0]
                .pending_sources,
            0
        );
        assert_eq!(
            core.async_copies[0]
                .state
                .as_deref()
                .unwrap()
                .committed_groups[0]
                .completion_time,
            301
        );
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut emitted)
                .unwrap(),
            Advance::Done
        );
        assert_eq!(core.result().cycles, 303);
        let counters = core.l1_counters().unwrap();
        assert_eq!(counters.misses, 2);
        assert_eq!(counters.merges, 1);
        assert_eq!(counters.bypass_sectors, 0);
    }

    #[test]
    fn shared_cacheable_async_copy_l1_hit_completes_without_a_request() {
        let sched = schedule(vec![
            (0, slot(cb(1), OpClass::Other, "LDGSTS.E")),
            (16, async_commit_slot()),
            (32, async_wait_slot(0)),
            (48, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![async_copy_inst(0, 0, 1), inst(16), inst(32), inst(48)],
                sectors: vec![40],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let MemModel::L1Shared(st) = &mut core.mem else {
            panic!("expected shared mode")
        };
        let EvAccess::MissRequest { fill } = st
            .l1
            .access_eligible_ev(MemoryOrigin::Global, 40, 0)
            .unwrap()
        else {
            panic!("expected cold prefill miss")
        };
        st.l1.complete_fill(fill).unwrap();
        let hits_before = st.l1.counters.hits;
        let mut requests = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut requests)
                .unwrap(),
            Advance::Done
        );
        assert!(requests.is_empty());
        assert_eq!(
            core.result().cycles,
            37,
            "wait must issue at hit-ready t=34 plus one payload cycle"
        );
        assert_eq!(core.l1_counters().unwrap().hits, hits_before + 1);
        assert_eq!(core.outstanding_responses(), 0);
        assert!(
            core.async_copies[0]
                .state
                .as_deref()
                .unwrap()
                .committed_groups
                .is_empty()
        );
    }

    #[test]
    fn shared_cacheable_async_copy_retry_retains_group_identity() {
        let sched = schedule(vec![
            (0, slot(cb(1), OpClass::Other, "LDGSTS.E")),
            (16, async_commit_slot()),
            (32, async_wait_slot(0)),
            (48, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![async_copy_inst(0, 0, 2), inst(16), inst(32), inst(48)],
                sectors: vec![0, 4],
            }],
        }];
        let mut cfg = l1_config();
        cfg.mshr_count = Some(1);
        let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::L1Shared(cfg)).unwrap();
        let mut requests = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut requests)
                .unwrap(),
            Advance::Idle { wake_at: None }
        );
        assert_eq!(requests.len(), 1);
        let group = core.async_copies[0]
            .state
            .as_deref()
            .unwrap()
            .committed_groups[0]
            .id;
        assert_eq!(
            core.async_copies[0]
                .state
                .as_deref()
                .unwrap()
                .committed_groups[0]
                .pending_sources,
            2
        );
        let MemModel::L1Shared(st) = &core.mem else {
            panic!("expected shared mode")
        };
        assert_eq!(st.retry.len(), 1);
        assert_eq!(st.retry[0].mshr_floor, NonZeroU32::new(1));
        assert_eq!(
            st.retry[0].target.and_then(FillTarget::async_group),
            Some(group)
        );

        let mut retried = Vec::new();
        core.on_response(requests[0].reply, 244, &mut retried)
            .unwrap();
        assert_eq!(retried.len(), 1, "freed MSHR must issue the parked line");
        assert!(matches!(retried[0].reply, ReplyTo::L1Fill { .. }));
        assert_eq!(
            core.async_copies[0]
                .state
                .as_deref()
                .unwrap()
                .committed_groups[0]
                .pending_sources,
            1
        );
        let MemModel::L1Shared(st) = &core.mem else {
            panic!("expected shared mode")
        };
        assert!(st.retry.is_empty());

        let mut emitted = Vec::new();
        core.on_response(retried[0].reply, 500, &mut emitted)
            .unwrap();
        assert!(emitted.is_empty());
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut emitted)
                .unwrap(),
            Advance::Done
        );
        assert_eq!(core.result().cycles, 503);
        assert_eq!(core.outstanding_responses(), 0);
    }

    #[test]
    fn shared_cacheable_floor_survives_retry_and_stays_access_local() {
        let mut p = params();
        let floor = NonZeroU32::new(2).unwrap();
        p.ldgsts_l1_mshr_floor = floor;
        let sched = schedule(vec![
            (0, slot(cb(1), OpClass::Other, "LDGSTS.E")),
            (16, async_commit_slot()),
            (32, async_wait_slot(0)),
            (48, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![async_copy_inst(0, 0, 3), inst(16), inst(32), inst(48)],
                sectors: vec![0, 4, 8],
            }],
        }];
        let mut cfg = l1_config();
        cfg.mshr_count = Some(1);
        let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::L1Shared(cfg)).unwrap();
        let mut requests = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut requests)
                .unwrap(),
            Advance::Idle { wake_at: None }
        );
        assert_eq!(
            requests.len(),
            2,
            "the floor admits two distinct line fills"
        );
        let MemModel::L1Shared(st) = &core.mem else {
            panic!("expected shared mode")
        };
        assert_eq!(st.retry.len(), 1);
        assert_eq!(st.retry[0].mshr_floor, Some(floor));

        let mut retried = Vec::new();
        core.on_response(requests[0].reply, 244, &mut retried)
            .unwrap();
        assert_eq!(
            retried.len(),
            1,
            "the retained floor must admit the retry while one fill remains"
        );
        let MemModel::L1Shared(st) = &core.mem else {
            panic!("expected shared mode")
        };
        assert!(st.retry.is_empty());

        let ordinary_sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDG.E"))]);
        let ordinary_blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: eligible(0, 2),
                }],
                sectors: vec![0, 4],
            }],
        }];
        let mut ordinary_cfg = l1_config();
        ordinary_cfg.mshr_count = Some(1);
        let mut ordinary = SmCore::new(
            &ordinary_blocks,
            &ordinary_sched,
            &p,
            SmMemMode::L1Shared(ordinary_cfg),
        )
        .unwrap();
        let mut ordinary_requests = Vec::new();
        assert_eq!(
            ordinary
                .advance(ReplayHorizon::Unbounded, &mut ordinary_requests)
                .unwrap(),
            Advance::Done
        );
        assert_eq!(ordinary_requests.len(), 1);
        let MemModel::L1Shared(st) = &ordinary.mem else {
            panic!("expected shared mode")
        };
        assert_eq!(st.retry.len(), 1);
        assert_eq!(st.retry[0].mshr_floor, None);
    }

    #[test]
    fn cacheable_async_fill_underflow_preserves_the_response_transaction() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDGSTS.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![async_copy_inst(0, 0, 1)],
                sectors: vec![40],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut requests = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut requests)
                .unwrap(),
            Advance::Done
        );
        let reply = requests[0].reply;
        let ReplyTo::L1Fill { fill, .. } = reply else {
            panic!("expected L1 fill response")
        };
        let group = core.async_copies[0]
            .state
            .as_deref_mut()
            .unwrap()
            .open_group
            .as_mut()
            .unwrap();
        let group_id = group.id;
        group.pending_sources = 0;
        let before_l1 = *core.l1_counters().unwrap();
        let mut emitted = Vec::new();

        assert!(matches!(
            core.on_response(reply, 244, &mut emitted),
            Err(IssueError::AsyncCopyPendingUnderflow {
                warp: 0,
                group,
                current: 0,
                remove: 1,
            }) if group == group_id
        ));
        assert!(emitted.is_empty());
        assert_eq!(core.outstanding_responses(), 1);
        assert_eq!(*core.l1_counters().unwrap(), before_l1);
        let MemModel::L1Shared(st) = &core.mem else {
            panic!("expected shared mode")
        };
        assert!(st.l1.is_fill_in_flight(fill));
        assert!(st.fill_waiters.contains_key(&fill));
    }

    #[test]
    fn cacheable_async_payload_floor_overflow_preserves_response_state() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDGSTS.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![async_copy_inst(0, 0, 1)],
                sectors: vec![40],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut requests = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut requests)
                .unwrap(),
            Advance::Done
        );
        let reply = requests[0].reply;
        let ReplyTo::L1Fill { fill, .. } = reply else {
            panic!("expected L1 fill response")
        };
        let before_group = core.async_copies[0].state.as_deref().unwrap().open_group;
        let before_l1 = *core.l1_counters().unwrap();
        let mut emitted = Vec::new();

        assert!(matches!(
            core.on_response(reply, u64::MAX, &mut emitted),
            Err(IssueError::TimeOverflow {
                operation: "LDGSTS payload completion floor",
                base: u64::MAX,
                delta: 1,
            })
        ));
        assert!(emitted.is_empty());
        assert_eq!(core.outstanding_responses(), 1);
        assert_eq!(
            core.async_copies[0].state.as_deref().unwrap().open_group,
            before_group
        );
        assert_eq!(*core.l1_counters().unwrap(), before_l1);
        let MemModel::L1Shared(st) = &core.mem else {
            panic!("expected shared mode")
        };
        assert!(st.l1.is_fill_in_flight(fill));
        assert!(st.fill_waiters.contains_key(&fill));
    }

    #[test]
    fn shared_async_copy_waits_for_all_sector_responses_out_of_order() {
        let sched = schedule(vec![
            (0, slot(cb(1), OpClass::Other, "LDGSTS.E.BYPASS")),
            (16, async_commit_slot()),
            (32, async_wait_slot(0)),
            (48, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    bypass_async_copy_inst(0, 0, 2),
                    inst(16),
                    inst(32),
                    inst(48),
                ],
                sectors: vec![40, 41],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut requests = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut requests)
                .unwrap(),
            Advance::Idle { wake_at: None }
        );
        assert_eq!(requests.len(), 2);
        assert!(requests.iter().all(|request| request.has_valid_contract()));
        let group = match requests[0].reply {
            ReplyTo::AsyncCopy {
                warp: 0,
                group,
                barrier: None,
                ..
            } => group,
            reply => panic!("unexpected first async-copy reply {reply:?}"),
        };
        assert!(matches!(
            requests[1].reply,
            ReplyTo::AsyncCopy {
                warp: 0,
                group: other,
                barrier: None,
                ..
            } if other == group
        ));
        assert_eq!(
            core.async_copies[0]
                .state
                .as_deref()
                .unwrap()
                .committed_groups
                .len(),
            1
        );
        assert_eq!(
            core.async_copies[0]
                .state
                .as_deref()
                .unwrap()
                .committed_groups[0]
                .pending_sources,
            2
        );

        let mut emitted = Vec::new();
        core.on_response(requests[1].reply, 300, &mut emitted)
            .unwrap();
        assert!(emitted.is_empty());
        assert_eq!(
            core.async_copies[0]
                .state
                .as_deref()
                .unwrap()
                .committed_groups[0]
                .pending_sources,
            1
        );
        assert_eq!(
            core.async_copies[0]
                .state
                .as_deref()
                .unwrap()
                .committed_groups[0]
                .completion_time,
            302
        );
        assert!(matches!(
            core.preflight_response(requests[1].reply, 300),
            Err(IssueError::ResponseTokenNotOutstanding { .. })
        ));
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut emitted)
                .unwrap(),
            Advance::Idle { wake_at: None }
        );

        core.on_response(requests[0].reply, 400, &mut emitted)
            .unwrap();
        assert_eq!(
            core.async_copies[0]
                .state
                .as_deref()
                .unwrap()
                .committed_groups[0]
                .pending_sources,
            0
        );
        assert_eq!(
            core.async_copies[0]
                .state
                .as_deref()
                .unwrap()
                .committed_groups[0]
                .completion_time,
            402
        );
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut emitted)
                .unwrap(),
            Advance::Done
        );
        assert_eq!(core.result().cycles, 404);
        assert!(
            core.async_copies[0]
                .state
                .as_deref()
                .unwrap()
                .committed_groups
                .is_empty()
        );
        assert_eq!(core.outstanding_responses(), 0);
        assert!(matches!(
            core.on_response(requests[0].reply, 404, &mut emitted),
            Err(IssueError::ResponseTokenNotOutstanding { reply })
                if reply == requests[0].reply
        ));
    }

    #[test]
    fn async_copy_response_can_complete_an_open_group_before_commit() {
        let sched = schedule(vec![(0, async_commit_slot())]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0)])],
        }];
        let mut core = shared_core(&blocks, &sched, &params());

        let p = params();
        let group = core.async_copies[0]
            .add_pending_async_copy_sources(0, 2, 0, 0, None, &p)
            .unwrap();
        core.async_copies[0]
            .complete_async_copy_sources(0, group, None, 2, 91, &p)
            .unwrap();
        assert_eq!(
            core.async_copies[0]
                .state
                .as_deref()
                .unwrap()
                .open_group
                .unwrap()
                .completion_time,
            91
        );
        assert_eq!(
            core.async_copies[0]
                .state
                .as_deref()
                .unwrap()
                .open_group
                .unwrap()
                .pending_sources,
            0
        );

        assert!(core.async_copies[0].commit_async_copy_group(100));
        assert!(
            core.async_copies[0]
                .state
                .as_deref()
                .unwrap()
                .open_group
                .is_none()
        );
        assert_eq!(
            core.async_copies[0]
                .state
                .as_deref()
                .unwrap()
                .committed_groups
                .len(),
            1
        );
        assert!(
            core.async_copies[0]
                .state
                .as_deref()
                .unwrap()
                .committed_groups[0]
                .is_complete_at(91)
        );
    }

    #[test]
    fn async_copy_response_underflow_is_transactional() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDGSTS.E.BYPASS"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![bypass_async_copy_inst(0, 0, 1)],
                sectors: vec![40],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut requests = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut requests)
                .unwrap(),
            Advance::Done
        );
        let reply = requests[0].reply;
        let ReplyTo::AsyncCopy { group, .. } = reply else {
            panic!("expected async-copy response")
        };
        core.async_copies[0]
            .state
            .as_deref_mut()
            .unwrap()
            .open_group
            .as_mut()
            .expect("issued copy owns an open group")
            .pending_sources = 0;
        let before_group = core.async_copies[0].state.as_deref().unwrap().open_group;
        let mut emitted = Vec::new();

        assert!(matches!(
            core.on_response(reply, 244, &mut emitted),
            Err(IssueError::AsyncCopyPendingUnderflow {
                warp: 0,
                group: got_group,
                current: 0,
                remove: 1,
            }) if got_group == group
        ));
        assert_eq!(
            core.async_copies[0].state.as_deref().unwrap().open_group,
            before_group
        );
        assert_eq!(core.outstanding_responses(), 1);
        assert!(emitted.is_empty());
    }

    #[test]
    fn async_copy_response_missing_group_is_transactional() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDGSTS.E.BYPASS"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![bypass_async_copy_inst(0, 0, 1)],
                sectors: vec![40],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut requests = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut requests)
                .unwrap(),
            Advance::Done
        );
        let reply = requests[0].reply;
        let ReplyTo::AsyncCopy { group, .. } = reply else {
            panic!("expected async-copy response")
        };
        core.async_copies[0]
            .state
            .as_deref_mut()
            .unwrap()
            .open_group = None;
        let mut emitted = Vec::new();

        assert!(matches!(
            core.on_response(reply, 244, &mut emitted),
            Err(IssueError::AsyncCopyGroupNotFound {
                warp: 0,
                group: got_group,
            }) if got_group == group
        ));
        assert_eq!(core.outstanding_responses(), 1);
        assert!(
            core.async_copies[0]
                .state
                .as_deref()
                .unwrap()
                .open_group
                .is_none()
        );
        assert!(emitted.is_empty());
    }

    #[test]
    fn issue_count_overflow_rejects_the_whole_cycle_before_mutation() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0)])],
        }];
        let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
        core.issued = u64::MAX;
        let before_unfinished = core.unfinished_warps;
        let before_resident = core.resident_warps;
        let mut out = Vec::new();

        assert!(matches!(
            core.advance(ReplayHorizon::Unbounded, &mut out),
            Err(IssueError::CounterOverflow {
                counter: "issued instruction"
            })
        ));
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), u64::MAX);
        assert_eq!(core.warps[0].next, 0);
        assert!(!core.warps[0].done);
        assert_eq!(core.unfinished_warps, before_unfinished);
        assert_eq!(core.resident_warps, before_resident);
    }

    #[test]
    fn missing_resident_slot_rejects_retirement_before_mutation() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
        let blocks = vec![BlockReplay {
            launch_index: 11,
            warps: vec![warp(2, vec![inst(0)])],
        }];
        let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
        assert!(core.resident_warp_slots.remove(&2));
        let before_unfinished = core.unfinished_warps;
        let before_resident = core.resident_warps;
        let before_subcore = core.subcore_warps[2].clone();
        let mut out = Vec::new();

        assert!(matches!(
            core.advance(ReplayHorizon::Unbounded, &mut out),
            Err(IssueError::ResidentWarpSlotMissing {
                launch_index: 11,
                warp_slot: 2,
            })
        ));
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), 0);
        assert_eq!(core.warps[0].next, 0);
        assert!(!core.warps[0].done);
        assert_eq!(core.unfinished_warps, before_unfinished);
        assert_eq!(core.resident_warps, before_resident);
        assert_eq!(core.subcore_warps[2], before_subcore);
    }

    #[test]
    fn pending_overflow_rejects_issue_before_cache_or_request_mutation() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let sched = schedule(vec![(0, slot(load, OpClass::Other, "LDG.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: eligible(0, 1),
                }],
                sectors: vec![0],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        core.warps[0].barrier_pending[0] = u32::MAX;
        let before_l1 = *core.l1_counters().unwrap();
        let mut out = Vec::new();

        assert!(matches!(
            core.advance(ReplayHorizon::Unbounded, &mut out),
            Err(IssueError::PendingBarrierOverflow {
                warp: 0,
                barrier: 0,
                current: u32::MAX,
                add: 1,
            })
        ));
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), 0);
        assert_eq!(core.warps[0].next, 0);
        assert_eq!(core.warps[0].barrier_pending[0], u32::MAX);
        assert_eq!(core.outstanding_responses(), 0);
        assert_eq!(*core.l1_counters().unwrap(), before_l1);
    }

    #[test]
    fn warm_hit_accepts_maximum_pending_count_without_a_new_unit() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let sched = schedule(vec![(0, slot(load, OpClass::Other, "LDG.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: eligible(0, 1),
                }],
                sectors: vec![0],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let MemModel::L1Shared(st) = &mut core.mem else {
            panic!("expected shared mode");
        };
        let EvAccess::MissRequest { fill } = st
            .l1
            .access_eligible_ev(MemoryOrigin::Global, 0, 0)
            .unwrap()
        else {
            panic!("expected cold prefill miss");
        };
        st.l1.complete_fill(fill).unwrap();
        core.warps[0].barrier_pending[0] = u32::MAX;
        let before = *core.l1_counters().unwrap();
        let mut out = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Done
        );
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), 1);
        assert_eq!(core.warps[0].next, 1);
        assert_eq!(core.warps[0].barrier_pending[0], u32::MAX);
        let after = *core.l1_counters().unwrap();
        assert_eq!(after.hits, before.hits + 1);
        assert_eq!(after.misses, before.misses);
        assert_eq!(core.outstanding_responses(), 0);
    }

    #[test]
    fn multi_subcore_l1_overflow_rejects_the_whole_issue_cycle() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "STG.E"))]);
        let store = || ReplayInst {
            pc: 0,
            kind: AccessKind::GlobalStore { off: 0, len: 1 },
        };
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![
                WarpReplay {
                    warp_slot: 0,
                    insts: vec![store()],
                    sectors: vec![0],
                },
                WarpReplay {
                    warp_slot: 1,
                    insts: vec![store()],
                    sectors: vec![4],
                },
            ],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let MemModel::L1Shared(st) = &mut core.mem else {
            panic!("expected shared mode");
        };
        st.l1.counters.store_sectors = u64::MAX - 1;
        let before = *core.l1_counters().unwrap();
        let mut out = Vec::new();

        assert!(matches!(
            core.advance(ReplayHorizon::Unbounded, &mut out),
            Err(IssueError::L1Access(L1AccessError::Counter(
                L1CounterError::Overflow {
                    field: L1CounterField::StoreSectors,
                }
            )))
        ));
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), 0);
        assert_eq!(core.warps[0].next, 0);
        assert_eq!(core.warps[1].next, 0);
        assert_eq!(*core.l1_counters().unwrap(), before);
    }

    #[test]
    fn constant_l1_retry_time_error_rejects_all_subcores_before_mutation() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let sched = schedule(vec![(0, slot(load, OpClass::Other, "LDG.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![
                WarpReplay {
                    warp_slot: 0,
                    insts: vec![ReplayInst {
                        pc: 0,
                        kind: eligible(0, 1),
                    }],
                    sectors: vec![0],
                },
                WarpReplay {
                    warp_slot: 1,
                    insts: vec![ReplayInst {
                        pc: 0,
                        kind: eligible(0, 1),
                    }],
                    sectors: vec![8],
                },
            ],
        }];
        let mut p = params();
        p.l1_hit_latency = 2;
        p.l1_miss_latency = 10;
        let cfg = p.l1_config(8, 1);
        let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::L1Const(cfg)).unwrap();
        let MemModel::L1Const(l1) = &mut core.mem else {
            panic!("expected constant L1 mode");
        };
        assert!(matches!(
            l1.access_eligible(MemoryOrigin::Global, 0, u64::MAX - 15).unwrap(),
            (AccessOutcome::MissLineAlloc, ready) if ready == u64::MAX - 5
        ));
        core.t = u64::MAX - 14;
        let before_l1 = *core.l1_counters().unwrap();
        let mut out = Vec::new();

        assert!(matches!(
            core.advance(ReplayHorizon::Unbounded, &mut out),
            Err(IssueError::L1Access(L1AccessError::TimeOverflow {
                operation: "miss visibility",
                base,
                delta: 10,
            })) if base == u64::MAX - 5
        ));
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), 0);
        assert_eq!(core.warps[0].next, 0);
        assert_eq!(core.warps[1].next, 0);
        assert_eq!(*core.l1_counters().unwrap(), before_l1);
        let MemModel::L1Const(l1) = &core.mem else {
            panic!("expected constant L1 mode");
        };
        assert!(l1.is_fill_in_flight(FillKey::Line(0)));
    }

    #[test]
    fn constant_l1_retry_payload_overflow_rejects_issue_before_cache_mutation() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDGSTS.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: AccessKind::GlobalAsyncCopy {
                        off: 0,
                        len: 1,
                        completion_work: 128,
                        l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                    },
                }],
                sectors: vec![8],
            }],
        }];
        let mut p = params();
        p.l1_hit_latency = 2;
        p.l1_miss_latency = 10;
        let cfg = p.l1_config(8, 1);
        let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::L1Const(cfg)).unwrap();
        let MemModel::L1Const(l1) = &mut core.mem else {
            panic!("expected constant L1 mode");
        };
        assert!(matches!(
            l1.access_eligible(MemoryOrigin::Global, 0, u64::MAX - 21)
                .unwrap(),
            (AccessOutcome::MissLineAlloc, ready) if ready == u64::MAX - 11
        ));
        core.t = u64::MAX - 20;
        let before_l1 = *core.l1_counters().unwrap();
        let mut out = Vec::new();

        assert!(matches!(
            core.advance(ReplayHorizon::Unbounded, &mut out),
            Err(IssueError::TimeOverflow {
                operation: "LDGSTS payload completion floor",
                base,
                delta: 2,
            }) if base == u64::MAX - 1
        ));
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), 0);
        assert_eq!(core.warps[0].next, 0);
        assert!(core.async_copies[0].state.is_none());
        assert_eq!(*core.l1_counters().unwrap(), before_l1);
        let MemModel::L1Const(l1) = &core.mem else {
            panic!("expected constant L1 mode");
        };
        assert!(l1.is_fill_in_flight(FillKey::Line(0)));
    }

    #[test]
    fn same_cycle_fill_chain_overflow_rejects_all_subcores_before_mutation() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let sched = schedule(vec![(0, slot(load, OpClass::Other, "LDG.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![
                WarpReplay {
                    warp_slot: 0,
                    insts: vec![ReplayInst {
                        pc: 0,
                        kind: eligible(0, 1),
                    }],
                    sectors: vec![0],
                },
                WarpReplay {
                    warp_slot: 1,
                    insts: vec![ReplayInst {
                        pc: 0,
                        kind: eligible(0, 1),
                    }],
                    sectors: vec![8],
                },
            ],
        }];
        let mut p = params();
        p.l1_hit_latency = 2;
        p.l1_miss_latency = 10;
        let cfg = p.l1_config(8, 1);
        let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::L1Const(cfg)).unwrap();
        core.t = u64::MAX - 15;
        let before_l1 = *core.l1_counters().unwrap();
        let mut out = Vec::new();

        assert!(matches!(
            core.advance(ReplayHorizon::Unbounded, &mut out),
            Err(IssueError::L1Access(L1AccessError::TimeOverflow {
                operation: "miss visibility",
                base,
                delta: 10,
            })) if base == u64::MAX - 5
        ));
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), 0);
        assert_eq!(core.warps[0].next, 0);
        assert_eq!(core.warps[1].next, 0);
        assert_eq!(*core.l1_counters().unwrap(), before_l1);
        let MemModel::L1Const(l1) = &core.mem else {
            panic!("expected constant L1 mode");
        };
        assert!(!l1.is_fill_in_flight(FillKey::Line(0)));
        assert!(!l1.is_fill_in_flight(FillKey::Line(2)));
    }

    #[test]
    fn response_tokens_reject_cross_core_and_stale_replay_swaps() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let sched = schedule(vec![(0, slot(load, OpClass::Other, "LDG.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: eligible(0, 1),
                }],
                sectors: vec![0],
            }],
        }];
        let mut first = shared_core(&blocks, &sched, &params());
        let mut second = shared_core(&blocks, &sched, &params());
        let mut first_requests = Vec::new();
        let mut second_requests = Vec::new();
        first
            .advance(ReplayHorizon::Unbounded, &mut first_requests)
            .unwrap();
        second
            .advance(ReplayHorizon::Unbounded, &mut second_requests)
            .unwrap();
        let first_reply = first_requests[0].reply;
        let second_reply = second_requests[0].reply;
        assert_ne!(first_reply, second_reply);
        let second_before = *second.l1_counters().unwrap();
        let mut emitted = Vec::new();
        assert!(matches!(
            second.on_response(first_reply, 244, &mut emitted),
            Err(IssueError::ResponseTokenNotOutstanding { reply }) if reply == first_reply
        ));
        assert!(emitted.is_empty());
        assert_eq!(second.outstanding_responses(), 1);
        assert_eq!(*second.l1_counters().unwrap(), second_before);

        drop(first);
        let mut next_generation = shared_core(&blocks, &sched, &params());
        let mut next_requests = Vec::new();
        next_generation
            .advance(ReplayHorizon::Unbounded, &mut next_requests)
            .unwrap();
        assert_ne!(first_reply, next_requests[0].reply);
        assert!(matches!(
            next_generation.on_response(first_reply, 244, &mut emitted),
            Err(IssueError::ResponseTokenNotOutstanding { reply }) if reply == first_reply
        ));
        assert_eq!(next_generation.outstanding_responses(), 1);
    }

    #[test]
    fn response_to_non_shared_mode_is_typed_and_nonmutating() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0)])],
        }];
        let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
        let mut out = Vec::new();

        assert!(matches!(
            core.on_response(ReplyTo::None, 0, &mut out),
            Err(IssueError::ResponseModeMismatch { mode: "flat" })
        ));
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), 0);
        assert_eq!(core.warps[0].next, 0);
    }

    #[test]
    fn fill_waiter_underflow_rejects_response_before_any_mutation() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(1);
        let sched = schedule(vec![
            (0, slot(load, OpClass::Other, "LDG.E")),
            (16, slot(consumer, OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: eligible(0, 1),
                    },
                    inst(16),
                ],
                sectors: vec![0],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut requests = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut requests)
                .unwrap(),
            Advance::Idle { wake_at: None }
        );
        let reply = requests[0].reply;
        let ReplyTo::L1Fill { fill, .. } = reply else {
            panic!("expected fill response");
        };
        core.warps[0].barrier_pending[0] = 0;
        let before_l1 = *core.l1_counters().unwrap();
        let mut emitted = Vec::new();

        assert!(matches!(
            core.on_response(reply, 244, &mut emitted),
            Err(IssueError::PendingBarrierUnderflow {
                warp: 0,
                barrier: 0,
                current: 0,
                remove: 1,
            })
        ));
        assert!(emitted.is_empty());
        assert_eq!(core.outstanding_responses(), 1);
        assert_eq!(*core.l1_counters().unwrap(), before_l1);
        let MemModel::L1Shared(st) = &core.mem else {
            panic!("expected shared mode");
        };
        assert!(st.l1.is_fill_in_flight(fill));
        assert!(st.fill_waiters.contains_key(&fill));
        if let FillKey::Line(line) = fill {
            assert!(st.fill_masks.contains_key(&line));
        }
    }

    #[test]
    fn retry_l1_overflow_preserves_the_entire_response_transaction() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(1);
        let sched = schedule(vec![
            (0, slot(load, OpClass::Other, "LDG.E")),
            (16, slot(consumer, OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: eligible(0, 1),
                    },
                    inst(16),
                ],
                sectors: vec![0, 8],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut requests = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut requests)
                .unwrap(),
            Advance::Idle { wake_at: None }
        );
        let reply = requests[0].reply;
        let ReplyTo::L1Fill { fill, .. } = reply else {
            panic!("expected fill response");
        };
        core.warps[0].barrier_pending[0] = 2;
        let MemModel::L1Shared(st) = &mut core.mem else {
            panic!("expected shared mode");
        };
        st.retry.push_back(Retry {
            warp: 0,
            instruction: 0,
            origin: MemoryOrigin::Global,
            target: Some(FillTarget::Barrier(barrier(0))),
            mshr_floor: None,
            lo: 1,
            hi: 2,
        });
        st.l1.counters.misses = u64::MAX;
        let before_l1 = *core.l1_counters().unwrap();
        let mut emitted = Vec::new();

        assert!(matches!(
            core.on_response(reply, 244, &mut emitted),
            Err(IssueError::L1Access(L1AccessError::Counter(
                L1CounterError::Overflow {
                    field: L1CounterField::Misses,
                }
            )))
        ));
        assert!(emitted.is_empty());
        assert_eq!(core.outstanding_responses(), 1);
        assert_eq!(core.warps[0].barrier_pending[0], 2);
        assert_eq!(*core.l1_counters().unwrap(), before_l1);
        let MemModel::L1Shared(st) = &core.mem else {
            panic!("expected shared mode");
        };
        assert!(st.l1.is_fill_in_flight(fill));
        assert!(st.fill_waiters.contains_key(&fill));
        assert_eq!(st.retry.len(), 1);
        let retry = st.retry.front().unwrap();
        assert_eq!(retry.warp, 0);
        assert_eq!(retry.lo, 1);
        assert_eq!(retry.hi, 2);
        if let FillKey::Line(line) = fill {
            assert!(st.fill_masks.contains_key(&line));
        }
    }

    #[test]
    fn retry_replacement_accepts_exact_maximum_pending_count() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(1);
        let sched = schedule(vec![
            (0, slot(load, OpClass::Other, "LDG.E")),
            (16, slot(consumer, OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: eligible(0, 1),
                    },
                    inst(16),
                ],
                sectors: vec![0, 8, 12],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut requests = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut requests)
                .unwrap(),
            Advance::Idle { wake_at: None }
        );
        let reply = requests[0].reply;
        core.warps[0].barrier_pending[0] = u32::MAX;
        let MemModel::L1Shared(st) = &mut core.mem else {
            panic!("expected shared mode");
        };
        st.retry.push_back(Retry {
            warp: 0,
            instruction: 0,
            origin: MemoryOrigin::Global,
            target: Some(FillTarget::Barrier(barrier(0))),
            mshr_floor: None,
            lo: 1,
            hi: 3,
        });
        let mut emitted = Vec::new();

        core.on_response(reply, 244, &mut emitted).unwrap();
        assert_eq!(emitted.len(), 2);
        assert_eq!(core.outstanding_responses(), 2);
        assert_eq!(core.warps[0].barrier_pending[0], u32::MAX);
        let MemModel::L1Shared(st) = &core.mem else {
            panic!("expected shared mode");
        };
        assert!(st.retry.is_empty());
    }

    #[test]
    fn admission_counter_failure_does_not_partially_admit_a_block() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
        let blocks = vec![
            BlockReplay {
                launch_index: 0,
                warps: vec![warp(0, vec![inst(0)])],
            },
            BlockReplay {
                launch_index: 1,
                warps: vec![warp(1, vec![inst(0)])],
            },
        ];
        let mut p = params();
        p.max_resident_warps_per_sm = 1;
        let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
        assert_eq!(core.next_block, 1);
        core.resident_warps = 0;
        core.arrival_stamp = u64::MAX;
        let before_warps = core.warps.len();
        let before_blocks = core.block_states.len();
        let before_subcores = core.subcore_warps.each_ref().map(Vec::len);

        assert!(matches!(
            core.admit_blocks(1),
            Err(IssueError::CounterOverflow {
                counter: "warp arrival stamp"
            })
        ));
        assert_eq!(core.warps.len(), before_warps);
        assert_eq!(core.block_states.len(), before_blocks);
        assert_eq!(core.subcore_warps.each_ref().map(Vec::len), before_subcores);
        assert_eq!(core.next_block, 1);
        assert_eq!(core.arrival_stamp, u64::MAX);
    }

    #[test]
    fn exact_max_accounting_total_is_valid_when_nothing_is_added() {
        assert_eq!(
            checked_u64_counter_add("test total", u64::MAX, 0).unwrap(),
            u64::MAX
        );
    }

    #[test]
    fn shared_mode_miss_blocks_consumer_until_response() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000001);
        let sched = schedule(vec![
            (0, slot(load, OpClass::Other, "LDG.E")),
            (16, slot(consumer, OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: eligible(0, 1),
                    },
                    inst(16),
                ],
                sectors: vec![0],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut out = Vec::new();
        // The load misses → request emitted; the consumer's wait has a
        // PENDING barrier → no computable timer → blocked on response.
        let a = core.advance(ReplayHorizon::Unbounded, &mut out).unwrap();
        assert_eq!(a, Advance::Idle { wake_at: None });
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].issue_time, 0);
        assert_eq!(out[0].source, L2Source::L1Fill);
        let reply = out[0].reply;
        let ReplyTo::L1Fill { response_id, .. } = reply else {
            panic!("expected an L1 fill response token");
        };
        let mut out2 = Vec::new();
        assert!(matches!(
            core.on_response(
                ReplyTo::Warp {
                    warp: 0,
                    barrier: barrier(0),
                    response_id,
                },
                423,
                &mut out2,
            ),
            Err(IssueError::ResponseTokenNotOutstanding { .. })
        ));
        assert!(matches!(
            core.on_response(
                ReplyTo::L1Fill {
                    fill: FillKey::Line(999),
                    response_id,
                },
                423,
                &mut out2,
            ),
            Err(IssueError::ResponseTokenNotOutstanding { .. })
        ));
        assert!(out2.is_empty());
        assert_eq!(core.outstanding_responses(), 1);
        assert_eq!(core.warps[0].barrier_pending[0], 1);
        // Response at the DRAM arm (423): consumer issues at 423.
        core.on_response(reply, 423, &mut out2).unwrap();
        assert!(out2.is_empty());
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out2).unwrap(),
            Advance::Done
        );
        let r = core.result();
        assert_eq!(r.instructions_issued, 2);
        assert_eq!(r.cycles, 424);
    }

    #[test]
    fn shared_mode_horizon_caps_run_ahead_after_a_request() {
        // Independent instructions after a miss: the SM may run ahead,
        // but never past issue_time + l1_miss_latency (the earliest
        // possible response).
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let sched = schedule(vec![
            (0, slot(load, OpClass::Other, "LDG.E")),
            (16, slot(cb(15), OpClass::Other, "IADD3")),
            (32, slot(cb(15), OpClass::Other, "IADD3")),
            (48, slot(cb(15), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: eligible(0, 1),
                    },
                    inst(16),
                    inst(32),
                    inst(48),
                ],
                sectors: vec![0],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut out = Vec::new();
        // 4 independent issues would finish at t=46 — well inside the
        // 244-cycle response bound, so the run completes...
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Done
        );
        assert_eq!(out.len(), 1);
        // ...but a tight EXTERNAL horizon yields first.
        let blocks2 = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: eligible(0, 1),
                    },
                    inst(16),
                    inst(32),
                    inst(48),
                ],
                sectors: vec![0],
            }],
        }];
        let mut core2 = shared_core(&blocks2, &sched, &params());
        let mut out2 = Vec::new();
        let a = core2
            .advance(ReplayHorizon::Bounded(10), &mut out2)
            .unwrap();
        assert_eq!(
            a,
            Advance::Idle { wake_at: Some(10) },
            "the next known timer (16) lies past the horizon: park AT the \
             horizon, where an event may ready an earlier warp"
        );
        assert_eq!(
            core2.advance(ReplayHorizon::Unbounded, &mut out2).unwrap(),
            Advance::Done
        );
    }

    #[test]
    fn shared_mode_merged_waiters_all_clear_on_one_response() {
        // Two warps load the SAME sector; warp 1 merges into warp 0's
        // fill. Both consumers wait; ONE response clears both.
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000001);
        let sched = schedule(vec![
            (0, slot(load, OpClass::Other, "LDG.E")),
            (16, slot(consumer, OpClass::Other, "EXIT")),
        ]);
        let mk_warp = |slot_ix: i32| WarpReplay {
            warp_slot: slot_ix,
            insts: vec![
                ReplayInst {
                    pc: 0,
                    kind: eligible(0, 1),
                },
                inst(16),
            ],
            sectors: vec![9],
        };
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![mk_warp(0), mk_warp(1)],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut out = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Idle { wake_at: None }
        );
        assert_eq!(out.len(), 1, "second warp merged, no second request");
        let c = core.l1_counters().unwrap();
        assert_eq!((c.misses, c.merges), (2, 1));
        core.on_response(out[0].reply, 300, &mut out).unwrap();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Done
        );
        // Both consumers issued at 300 (distinct sub-cores) → cycles 301.
        assert_eq!(core.result().cycles, 301);
    }

    #[test]
    fn unbarriered_retry_hit_skips_unused_timing_arms() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDG.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: eligible(0, 1),
                }],
                sectors: vec![0, 8],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let MemModel::L1Shared(st) = &mut core.mem else {
            panic!("expected shared mode");
        };
        let EvAccess::MissRequest { fill } = st
            .l1
            .access_eligible_ev(MemoryOrigin::Global, 8, 0)
            .unwrap()
        else {
            panic!("expected cold prefill miss");
        };
        st.l1.complete_fill(fill).unwrap();

        let mut out = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Done
        );
        assert_eq!(out.len(), 1, "cold issued load still owns an L1 fill");
        let reply = out.pop().unwrap().reply;
        let MemModel::L1Shared(st) = &mut core.mem else {
            panic!("expected shared mode");
        };
        st.retry.push_back(Retry {
            warp: 0,
            instruction: 0,
            origin: MemoryOrigin::Global,
            target: None,
            mshr_floor: None,
            lo: 1,
            hi: 2,
        });
        let hits_before = st.l1.counters.hits;

        core.on_response(reply, u64::MAX - 1, &mut out).unwrap();
        assert!(out.is_empty());
        assert_eq!(core.outstanding_responses(), 0);
        assert_eq!(core.l1_counters().unwrap().hits, hits_before + 1);
        let MemModel::L1Shared(st) = &core.mem else {
            panic!("expected shared mode");
        };
        assert!(st.retry.is_empty());
        assert_eq!(core.warps[0].barrier_clear, [0; DEP_BARRIERS]);
    }

    #[test]
    fn shared_mode_mshr_stall_parks_and_retries_on_completion() {
        // MSHR capacity 1: the second warp's DIFFERENT-line load stalls,
        // parks, and retries when the first fill completes.
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000001);
        let sched = schedule(vec![
            (0, slot(load, OpClass::Other, "LDG.E")),
            (16, slot(consumer, OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![
                WarpReplay {
                    warp_slot: 0,
                    insts: vec![
                        ReplayInst {
                            pc: 0,
                            kind: eligible(0, 1),
                        },
                        inst(16),
                    ],
                    sectors: vec![0],
                },
                WarpReplay {
                    warp_slot: 1,
                    insts: vec![
                        ReplayInst {
                            pc: 0,
                            kind: eligible(0, 1),
                        },
                        inst(16),
                    ],
                    sectors: vec![8],
                },
            ],
        }];
        let mut cfg = l1_config();
        cfg.mshr_count = Some(1);
        let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::L1Shared(cfg)).unwrap();
        let mut out = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Idle { wake_at: None }
        );
        assert_eq!(out.len(), 1, "second load stalled, not requested");
        assert_eq!(core.l1_counters().unwrap().stall_events, 1);
        // First fill completes → retry emits the parked request at 244.
        let first = out[0].reply;
        out.clear();
        assert!(matches!(
            core.on_response(first, u64::MAX - 1, &mut out),
            Err(IssueError::TimeOverflow {
                operation: "retry self-response horizon",
                base,
                delta: 244,
            }) if base == u64::MAX - 1
        ));
        assert!(out.is_empty());
        assert_eq!(core.outstanding_responses(), 1);
        let MemModel::L1Shared(st) = &core.mem else {
            panic!("expected shared mode");
        };
        assert_eq!(st.retry.len(), 1);
        assert_eq!(core.warps[1].barrier_pending[0], 1);

        core.on_response(first, 244, &mut out).unwrap();
        assert_eq!(out.len(), 1, "retry sent the parked sector");
        assert_eq!(out[0].issue_time, 244);
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Idle { wake_at: None }
        );
        let second = out[0].reply;
        core.on_response(second, 488, &mut out).unwrap();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Done
        );
        let r = core.result();
        assert_eq!(r.instructions_issued, 4);
        // Warp 1's consumer issues at 488 → cycles 489.
        assert_eq!(r.cycles, 489);
    }

    #[test]
    fn shared_mode_bypass_load_waits_on_direct_reply() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(2));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000100);
        let sched = schedule(vec![
            (0, slot(load, OpClass::Other, "LDG.E.CG")),
            (16, slot(consumer, OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalBypass {
                            off: 0,
                            len: 2,
                            token: CacheOpToken::LdgECg,
                            l2_requests: 0,
                        },
                    },
                    inst(16),
                ],
                sectors: vec![40, 41],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut out = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Idle { wake_at: None }
        );
        assert_eq!(out.len(), 2, "one request per bypass sector");
        assert!(out.iter().all(|r| r.source == L2Source::Bypass));
        assert!(
            out.iter().all(|r| r.operation == MemoryOperation::Read),
            ".cg bypass loads remain read operations"
        );
        assert!(matches!(
            out[0].reply,
            ReplyTo::Warp {
                warp: 0,
                barrier: got,
                ..
            } if got == barrier(2)
        ));
        // Both responses must land before the wait clears.
        let mut sink = Vec::new();
        core.on_response(out[0].reply, 244, &mut sink).unwrap();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut sink).unwrap(),
            Advance::Idle { wake_at: None },
            "one of two pending units still outstanding"
        );
        core.on_response(out[1].reply, 423, &mut sink).unwrap();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut sink).unwrap(),
            Advance::Done
        );
        assert_eq!(core.result().cycles, 424);
    }

    #[test]
    fn invalid_response_times_are_rejected_before_mutation() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(2));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000100);
        let sched = schedule(vec![
            (0, slot(load, OpClass::Other, "LDG.E.CG")),
            (16, slot(consumer, OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalBypass {
                            off: 0,
                            len: 1,
                            token: CacheOpToken::LdgECg,
                            l2_requests: 0,
                        },
                    },
                    inst(16),
                ],
                sectors: vec![40],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut requests = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut requests)
                .unwrap(),
            Advance::Idle { wake_at: None }
        );
        assert_eq!(core.outstanding_responses(), 1);
        assert_eq!(core.warps[0].barrier_pending[2], 1);
        let ReplyTo::Warp { response_id, .. } = requests[0].reply else {
            panic!("expected a direct warp response token");
        };

        let mut retries = Vec::new();
        assert_eq!(core.now(), 1);
        assert!(matches!(
            core.on_response(requests[0].reply, 0, &mut retries),
            Err(IssueError::ResponseBeforeClock { now: 0, clock: 1 })
        ));
        assert!(matches!(
            core.on_response(
                ReplyTo::Warp {
                    warp: 1,
                    barrier: barrier(2),
                    response_id,
                },
                1,
                &mut retries,
            ),
            Err(IssueError::ResponseWarpOutOfRange { warp: 1, warps: 1 })
        ));
        assert!(matches!(
            core.on_response(
                ReplyTo::Warp {
                    warp: 0,
                    barrier: barrier(1),
                    response_id,
                },
                1,
                &mut retries,
            ),
            Err(IssueError::ResponseTokenNotOutstanding { .. })
        ));
        assert!(matches!(
            core.on_response(ReplyTo::None, 1, &mut retries),
            Err(IssueError::UnexpectedMutationOnlyResponse)
        ));
        assert!(retries.is_empty());
        assert_eq!(core.now(), 1);
        assert_eq!(core.outstanding_responses(), 1);
        assert_eq!(core.warps[0].barrier_pending[2], 1);
        assert_eq!(core.warps[0].barrier_clear[2], 0);

        core.on_response(requests[0].reply, 244, &mut retries)
            .unwrap();
        assert!(matches!(
            core.on_response(requests[0].reply, 244, &mut retries),
            Err(IssueError::ResponseTokenNotOutstanding { .. })
        ));
        assert!(retries.is_empty());
        assert_eq!(core.now(), 1);
        assert_eq!(core.outstanding_responses(), 0);
        assert_eq!(core.warps[0].barrier_pending[2], 0);
        assert_eq!(core.warps[0].barrier_clear[2], 244);
    }

    #[test]
    fn valid_response_at_exact_maximum_clears_its_barrier() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(2));
        let sched = schedule(vec![(0, slot(load, OpClass::Other, "LDG.E.CG"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: AccessKind::GlobalBypass {
                        off: 0,
                        len: 1,
                        token: CacheOpToken::LdgECg,
                        l2_requests: 0,
                    },
                }],
                sectors: vec![40],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut requests = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut requests)
                .unwrap(),
            Advance::Done
        );
        assert_eq!(requests.len(), 1);
        assert_eq!(core.warps[0].barrier_pending[2], 1);
        let mut retries = Vec::new();

        core.on_response(requests[0].reply, u64::MAX, &mut retries)
            .unwrap();

        assert!(retries.is_empty());
        assert_eq!(core.outstanding_responses(), 0);
        assert_eq!(core.warps[0].barrier_pending[2], 0);
        assert_eq!(core.warps[0].barrier_clear[2], u64::MAX);
    }

    #[test]
    fn modeled_async_copy_source_does_not_claim_a_visibility_tail() {
        assert_eq!(
            TailFlags::for_access(AccessKind::GlobalAsyncCopy {
                off: 0,
                len: 1,
                completion_work: 4,
                l1_policy: LdgstsL1Policy::CacheAtAllLevels,
            }),
            TailFlags::default()
        );
    }

    #[test]
    fn shared_mode_store_emits_mutation_only_requests() {
        let sched = schedule(vec![
            (0, slot(cb(1), OpClass::Other, "STG.E")),
            (16, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalStore { off: 0, len: 2 },
                    },
                    inst(16),
                ],
                sectors: vec![100, 101],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut out = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Done
        );
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|r| r.source == L2Source::Store
            && r.origin == MemoryOrigin::Global
            && r.operation == MemoryOperation::Store
            && r.reply == ReplyTo::None));
        let result = core.result();
        assert_eq!(result.l1.unwrap().store_sectors, 2);
        assert_eq!(
            result.unmodeled_tails,
            TailFlags {
                red_rmw: false,
                dirty_writeback: true,
                coherence: true,
                ldgsts_dst_shared_timing: false,
            }
        );
    }

    #[test]
    fn shared_mode_reduction_is_nonreturning_without_store_attribution() {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "RED.E.ADD"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: AccessKind::GlobalReduction {
                        off: 0,
                        len: 2,
                        l2_requests: 1,
                    },
                }],
                sectors: vec![100, 101],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut out = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Done
        );
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|request| request.source == L2Source::Store
            && request.operation == MemoryOperation::Reduction
            && request.reply == ReplyTo::None
            && request.has_valid_contract()));
        let result = core.result();
        let l1 = result.l1.unwrap();
        assert_eq!(l1.store_sectors, 0);
        assert_eq!(l1.reduction_sectors, 2);
        assert_eq!(l1.reduction_hits, 0);
        assert_eq!(
            result.unmodeled_tails,
            TailFlags {
                red_rmw: true,
                dirty_writeback: true,
                coherence: true,
                ldgsts_dst_shared_timing: false,
            }
        );
    }

    /// §6.4 slice 2 — one instruction's cold 4-sector line miss reaches
    /// the L2 as ONE request CARRYING ALL FOUR SECTORS (the merged
    /// sectors fold into the fill's mask; NCU counts 4 LTS sectors),
    /// with a single response completing every waiter.
    #[test]
    fn shared_mode_line_fill_folds_batch_sectors_into_one_request() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(2));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000100);
        let sched = schedule(vec![
            (0, slot(load, OpClass::Other, "LDG.E")),
            (16, slot(consumer, OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalEligible {
                            off: 0,
                            len: 4,
                            token: CacheOpToken::LdgECa,
                        },
                    },
                    inst(16),
                ],
                sectors: vec![40, 41, 42, 43],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut out = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Idle { wake_at: None }
        );
        assert_eq!(out.len(), 1, "one L2 transaction for the whole line");
        assert_eq!(out[0].origin, MemoryOrigin::Global);
        assert_eq!(out[0].sector_mask, 0b1111);
        assert!(matches!(
            out[0].reply,
            ReplyTo::L1Fill {
                fill: FillKey::Line(10),
                ..
            }
        ));
        let mut sink = Vec::new();
        core.on_response(out[0].reply, 244, &mut sink).unwrap();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut sink).unwrap(),
            Advance::Done
        );
        assert_eq!(core.result().cycles, 245);
    }

    #[test]
    fn mixed_origin_same_batch_fold_preserves_one_physical_request() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(2));
        let sched = schedule(vec![(0, slot(load, OpClass::Other, "LDG.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: AccessKind::GlobalEligible {
                        off: 0,
                        len: 1,
                        token: CacheOpToken::LdgECa,
                    },
                }],
                sectors: vec![40, 41],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut out = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Done
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].origin, MemoryOrigin::Global);
        assert_eq!(out[0].sector_mask, 0b0001);
        assert_eq!(out[0].local_sector_mask, 0);

        let (mem, warps) = (&mut core.mem, &mut core.warps);
        let MemModel::L1Shared(st) = mem else {
            panic!("expected shared mode");
        };
        let response_id = ResponseId::new(st.response_owner, 0, 1, 1);
        assert!(
            access_shared(
                st,
                warps,
                0,
                None,
                None,
                MemoryOrigin::Local,
                41,
                response_id,
                1,
                &mut out,
            )
            .unwrap()
            .is_none()
        );

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].origin, MemoryOrigin::Global);
        assert_eq!(out[0].sector_mask, 0b0011);
        assert_eq!(out[0].local_sector_mask, 0b0010);
        assert_eq!(st.fill_masks.get(&10), Some(&0b0011));
        assert_eq!(st.l1.counters.origin(MemoryOrigin::Global).misses, 1);
        let local = st.l1.counters.origin(MemoryOrigin::Local);
        assert_eq!(local.misses, 1);
        assert_eq!(local.merges, 1);
    }

    #[test]
    fn mixed_origin_retry_fold_is_transactional_and_preserves_one_request() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(2));
        let sched = schedule(vec![(0, slot(load, OpClass::Other, "LDG.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: AccessKind::GlobalEligible {
                        off: 0,
                        len: 1,
                        token: CacheOpToken::LdgECa,
                    },
                }],
                // Sector 40 owns the response. The two injected retry entries
                // target distinct sectors of line 11.
                sectors: vec![40, 44, 45],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut requests = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut requests)
                .unwrap(),
            Advance::Done
        );
        let reply = requests[0].reply;
        let fill = match reply {
            ReplyTo::L1Fill { fill, .. } => fill,
            _ => panic!("expected L1 fill response"),
        };

        let (before_counters, before_masks, before_pending, before_retries);
        {
            let MemModel::L1Shared(st) = &mut core.mem else {
                panic!("expected shared mode");
            };
            st.retry.push_back(Retry {
                warp: 0,
                instruction: 1,
                origin: MemoryOrigin::Global,
                target: None,
                mshr_floor: None,
                lo: 1,
                hi: 2,
            });
            st.retry.push_back(Retry {
                warp: 0,
                instruction: 2,
                origin: MemoryOrigin::Local,
                target: None,
                mshr_floor: None,
                lo: 2,
                hi: 3,
            });
            before_counters = st.l1.counters;
            before_masks = st.fill_masks.clone();
            before_pending = core.warps[0].barrier_pending;
            before_retries = st.retry.len();
        }
        let mut emitted = Vec::new();

        assert!(matches!(
            core.on_response(reply, u64::MAX, &mut emitted),
            Err(IssueError::TimeOverflow {
                operation: "retry self-response horizon",
                base: u64::MAX,
                ..
            })
        ));
        assert!(emitted.is_empty());
        {
            let MemModel::L1Shared(st) = &core.mem else {
                panic!("expected shared mode");
            };
            assert!(st.outstanding_replies.contains(&reply));
            assert!(st.l1.is_fill_in_flight(fill));
            assert_eq!(st.l1.counters, before_counters);
            assert_eq!(st.fill_masks, before_masks);
            assert_eq!(st.retry.len(), before_retries);
            assert_eq!(core.warps[0].barrier_pending, before_pending);
        }

        core.on_response(reply, 244, &mut emitted).unwrap();
        assert_eq!(emitted.len(), 1);
        assert_eq!(emitted[0].origin, MemoryOrigin::Global);
        assert_eq!(emitted[0].sector_mask, 0b0011);
        assert_eq!(emitted[0].local_sector_mask, 0b0010);

        let MemModel::L1Shared(st) = &core.mem else {
            panic!("expected shared mode");
        };
        assert!(!st.outstanding_replies.contains(&reply));
        assert!(!st.l1.is_fill_in_flight(fill));
        assert!(st.retry.is_empty());
        assert_eq!(core.warps[0].barrier_pending, [0; DEP_BARRIERS]);
        assert_eq!(st.l1.counters.origin(MemoryOrigin::Global).misses, 2);
        assert_eq!(st.l1.counters.origin(MemoryOrigin::Local).misses, 1);
    }

    /// §6.4 slice 2 — a LATER instruction merging onto an in-flight
    /// line fill: a NEW sector sends a follow-up `ReplyTo::None` fetch
    /// (the L2 must see it once), a re-touched sector sends nothing
    /// (MSHR merge), and every waiter still rides the ORIGINAL fill's
    /// one response.
    #[test]
    fn shared_mode_same_batch_cross_instruction_merge_folds() {
        // Gate fix 2026-07-07: fold scope is the EMISSION BATCH (one
        // advance call), not one instruction — a later instruction's
        // merge onto an earlier instruction's line fill folds into that
        // request's mask (so worst-of-mask pricing covers it) instead
        // of splitting off a follow-up fetch.
        let mut load_a = cb(1);
        load_a.write_barrier = Some(barrier(2));
        let mut load_b = cb(1);
        load_b.write_barrier = Some(barrier(2));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000100);
        let sched = schedule(vec![
            (0, slot(load_a, OpClass::Other, "LDG.E")),
            (16, slot(load_b, OpClass::Other, "LDG.E")),
            (32, slot(consumer, OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalEligible {
                            off: 0,
                            len: 1,
                            token: CacheOpToken::LdgECa,
                        },
                    },
                    ReplayInst {
                        pc: 16,
                        kind: AccessKind::GlobalEligible {
                            off: 1,
                            len: 2,
                            token: CacheOpToken::LdgECa,
                        },
                    },
                    inst(32),
                ],
                // A: sector 40. B: sector 41 (folds) + 40 (re-touch).
                sectors: vec![40, 41, 40],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut out = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Idle { wake_at: None }
        );
        assert_eq!(out.len(), 1, "one line transaction for the whole batch");
        assert_eq!(out[0].sector_mask, 0b11, "B's sector folded into A's fill");
        assert!(matches!(
            out[0].reply,
            ReplyTo::L1Fill {
                fill: FillKey::Line(10),
                ..
            }
        ));
        let mut sink = Vec::new();
        core.on_response(out[0].reply, 244, &mut sink).unwrap();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut sink).unwrap(),
            Advance::Done,
            "one response clears both instructions' waiters"
        );
    }

    /// A merge onto a fill emitted in an EARLIER batch (the request
    /// already left with the SM's previous yield) cannot fold — it
    /// sends exactly one follow-up `ReplyTo::None` fetch, and the
    /// waiter still rides the original fill's response.
    #[test]
    fn shared_mode_cross_batch_merge_sends_followup_fetch_once() {
        let mut load_m = cb(1);
        load_m.write_barrier = Some(barrier(2));
        let mut load_l = cb(1);
        load_l.write_barrier = Some(barrier(3));
        // Merging load: gated on barrier 2 (fill M's return) so it
        // issues in a LATER advance call while fill L is still in
        // flight; its own waiter registers on barrier 2.
        let mut load_merge = cb(1);
        load_merge.write_barrier = Some(barrier(2));
        load_merge.wait_mask = wait_mask(0b000100);
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b001100);
        let sched = schedule(vec![
            (0, slot(load_m, OpClass::Other, "LDG.E")),
            (16, slot(load_l, OpClass::Other, "LDG.E")),
            (32, slot(load_merge, OpClass::Other, "LDG.E")),
            (48, slot(consumer, OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalEligible {
                            off: 0,
                            len: 1,
                            token: CacheOpToken::LdgECa,
                        },
                    },
                    ReplayInst {
                        pc: 16,
                        kind: AccessKind::GlobalEligible {
                            off: 1,
                            len: 1,
                            token: CacheOpToken::LdgECa,
                        },
                    },
                    ReplayInst {
                        pc: 32,
                        kind: AccessKind::GlobalEligible {
                            off: 2,
                            len: 1,
                            token: CacheOpToken::LdgECa,
                        },
                    },
                    inst(48),
                ],
                // M: sector 80 (line 20). L: sector 40 (line 10).
                // Merge: sector 41 (line 10, new sector).
                sectors: vec![80, 40, 41],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut out = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Idle { wake_at: None }
        );
        assert_eq!(out.len(), 2, "two fills; the merging load is gated");
        assert!(matches!(
            out[0].reply,
            ReplyTo::L1Fill {
                fill: FillKey::Line(20),
                ..
            }
        ));
        assert!(matches!(
            out[1].reply,
            ReplyTo::L1Fill {
                fill: FillKey::Line(10),
                ..
            }
        ));
        assert_eq!(out[1].sector_mask, 0b01);
        // Fill M returns; the merging load issues in a NEW batch while
        // fill L is still in flight.
        let mut out2 = Vec::new();
        core.on_response(out[0].reply, 423, &mut out2).unwrap();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out2).unwrap(),
            Advance::Idle { wake_at: None }
        );
        assert_eq!(out2.len(), 1, "exactly one follow-up fetch");
        assert_eq!(out2[0].sector_mask, 0b10, "only the new sector");
        assert_eq!(out2[0].reply, ReplyTo::None, "no second response");
        assert_eq!(out2[0].source, L2Source::L1Fill);
        // Fill L's one response clears BOTH its own waiter (barrier 3)
        // and the merged load's waiter (barrier 2).
        let mut sink = Vec::new();
        core.on_response(out[1].reply, 846, &mut sink).unwrap();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut sink).unwrap(),
            Advance::Done
        );
    }

    /// ATOM records ride BYPASS provenance (they never stage into the
    /// modeled L1) but retain returning RMW operation semantics. The system
    /// derives the write-class L2 rule and missing-sector fetch from that type.
    #[test]
    fn shared_mode_atom_is_bypass_provenance_write_class() {
        let mut atom = cb(1);
        atom.write_barrier = Some(barrier(2));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000100);
        let sched = schedule(vec![
            (0, slot(atom, OpClass::Other, "ATOMG.E.ADD.STRONG.GPU")),
            (16, slot(consumer, OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalBypass {
                            off: 0,
                            len: 1,
                            token: CacheOpToken::Atom,
                            l2_requests: 1,
                        },
                    },
                    inst(16),
                ],
                sectors: vec![40],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut out = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Idle { wake_at: None }
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].source, L2Source::Bypass);
        assert_eq!(out[0].operation, MemoryOperation::Atomic);
        assert_eq!(out[0].operation.l2_class(), L2OpClass::WriteAtomic);
        assert!(
            matches!(
                out[0].reply,
                ReplyTo::Warp {
                    warp: 0,
                    barrier: got,
                    ..
                } if got == barrier(2)
            ),
            "the atomic's return value still waits on the reply"
        );
    }
}
