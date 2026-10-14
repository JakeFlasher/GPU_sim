use std::num::NonZeroU32;

use serde::Serialize;
use sim_arch::card::{ArchCard, RequiredIssueResources};
use sim_arch::{
    ArchId, ConstantMemoryReturnService, ImmediateConstantCache, InstructionFrontEnd,
    LdgstsCompletionThroughput, LdgstsDestinationWavefrontProjection, LdgstsL1HitReturnService,
    MemoryIssuePipeline, ScalarExecutionPipeline, SharedBankProjection,
};
use sim_mem::L1Config;

use super::{DEP_BARRIERS, IssueError, SUB_CORES};

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

}

/// Source of scheduler-resource values. Carded SM86 resources are executable
/// only when every structural field matches the fixed topology consumed by
/// this executor; divergent cards fail closed rather than being shadowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueResourceSource {
    /// Existing sm_86-only 25/8/48 architecture-local hypotheses.
    ArchitectureDefaults,
    /// Strict architecture-card resources, restricted to the currently
    /// consumed SM86 topology.
    Carded(RequiredIssueResources),
    /// Explicit hypothetical issue-port override layered on an admitted card
    /// or the architecture-default constructor. This is a design-space input,
    /// never evidence about the physical card.
    HypotheticalIssuePorts(Option<RequiredIssueResources>),
    /// Explicit SM120 alpha lane.  The lane is a measured-envelope
    /// hypothesis assembled from the retained RTX 5090 microbenchmarks; it is
    /// intentionally distinct from both SM86 card defaults and strict carded
    /// resources so reports cannot mistake it for a silicon topology claim.
    Sm120Alpha,
}

/// Explicit timing/resource inputs for the SM120 alpha lane.
///
/// Nothing in this structure is obtained from the SM86 architecture
/// capability table.  Values are an executable, deliberately provisional
/// envelope assembled from the retained M8 dependency/ILP and M9
/// barrier/control probes. The alpha lane supplies the effective SM120 timing
/// envelope for integrated full replay and the complementary fast frontier; it
/// is not a calibrated Blackwell silicon model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Sm120AlphaConfig {
    pub l1_hit_cycles: u64,
    pub l2_hit_cycles: u64,
    pub dram_cycles: u64,
    pub shared_cycles: u64,
    pub shared_additional_wavefront_cycles: u64,
    pub non_memory_write_barrier_cycles: u64,
    pub read_barrier_cycles: u64,
    pub issue_ports: u32,
    pub dual_issue_supported: bool,
    pub max_resident_warps_per_sm: u32,
    pub max_resident_blocks_per_sm: u32,
    pub scalar_latency_cycles: u64,
    /// Operation-family latency observations from the SM120 scheduler probe.
    /// These are calibration inputs for the behavioral alpha model, not a
    /// claim about a unique Blackwell execution pipeline.
    pub fma_latency_cycles: u64,
    pub mad_latency_cycles: u64,
    pub add_latency_cycles: u64,
    pub fma_initiation_interval_cycles: u64,
    pub mad_initiation_interval_cycles: u64,
    pub add_initiation_interval_cycles: u64,
    /// Logical front-end warm/footprint envelope calibrated from P10 endpoint
    /// probes.  It is applied only by the fast alpha stage and is not an
    /// instruction-cache geometry claim.
    pub front_end_base_cycles: u64,
    pub front_end_line_cycles_numerator: u64,
    pub front_end_line_cycles_denominator: u64,
    pub scalar_initiation_interval_cycles: u64,
    pub scalar_in_flight_capacity: u32,
    pub barrier_warp_initiation_interval_cycles: u64,
    pub barrier_generation_rearm_cycles: u64,
    pub ldgsts_l1_mshr_floor: u32,
    pub ldgsts_cache_all_payload_bytes_per_cycle: u64,
    pub ldgsts_bypass_payload_bytes_per_cycle: u64,
    pub ldgsts_cache_all_destination_wavefront_cycles: u64,
    pub ldgsts_global_to_shared_transport_cycles: u64,
    pub ldgsts_cache_all_destination_service_wavefronts: u64,
    pub ldgsts_cache_all_destination_service_cycles: u64,
}

impl Sm120AlphaConfig {
    /// Stable identifier for the explicit alpha configuration consumed by the
    /// executor.  Reports should obtain this value from the configuration
    /// instead of copying a profile name beside independently hard-coded
    /// timing/occupancy fields.
    pub const PROFILE_ID: &'static str = "sm120_m8_m9_sensitivity_v1_provisional";

    pub const fn profile_id() -> &'static str {
        Self::PROFILE_ID
    }

    /// Initial explicit M8/M9 envelope used by the CLI convenience flag.
    /// These values are intentionally visible in one alpha-only constructor;
    /// no call into an SM86 capability/default constructor is made.
    pub const fn from_m8_m9() -> Self {
        Self {
            l1_hit_cycles: 32,
            l2_hit_cycles: 160,
            dram_cycles: 520,
            shared_cycles: 20,
            shared_additional_wavefront_cycles: 1,
            // P9 scheduler calibration keeps scalar and barrier families
            // separate; this is a generic fallback for unclassified scalar
            // forms, not a claim that every INT/FP opcode has this latency.
            non_memory_write_barrier_cycles: 8,
            read_barrier_cycles: 8,
            issue_ports: 1,
            dual_issue_supported: false,
            max_resident_warps_per_sm: 48,
            max_resident_blocks_per_sm: 24,
            scalar_latency_cycles: 23,
            // P9 scheduler depth sweep: approximately 4.44 cycles per
            // dependent FFMA/MAD and 1.44 cycles per dependent ADD; values
            // are rounded upward for a conservative readiness envelope.
            fma_latency_cycles: 5,
            mad_latency_cycles: 5,
            add_latency_cycles: 2,
            fma_initiation_interval_cycles: 1,
            // Fixed-work P9 sweeps show independent IMAD/MAD chains sustain
            // roughly two clock cycles per operation on this SKU.
            mad_initiation_interval_cycles: 2,
            add_initiation_interval_cycles: 1,
            front_end_base_cycles: 3900,
            front_end_line_cycles_numerator: 3,
            front_end_line_cycles_denominator: 2,
            scalar_initiation_interval_cycles: 1,
            scalar_in_flight_capacity: 8,
            // The P9 barrier-warp sweep (w1/w8/w32 calibration, w16 held-out)
            // is approximately two SM cycles per warp arrival in the measured
            // launch envelope.  This remains a behavioral calibration knob.
            barrier_warp_initiation_interval_cycles: 2,
            barrier_generation_rearm_cycles: 1,
            // Async/TMA semantics are not measured on SM120.  These positive
            // values keep the generic completion arithmetic total; the direct
            // alpha CLI rejects cache/async modes until a typed SM120 sidecar
            // supplies their semantics.
            ldgsts_l1_mshr_floor: 1,
            ldgsts_cache_all_payload_bytes_per_cycle: 1,
            // P14 independent TMA shape calibration uses a distinct bypass
            // payload service arm; LDGSTS cache-all keeps its own rate.
            ldgsts_bypass_payload_bytes_per_cycle: 4,
            ldgsts_cache_all_destination_wavefront_cycles: 1,
            ldgsts_global_to_shared_transport_cycles: 1,
            ldgsts_cache_all_destination_service_wavefronts: 1,
            ldgsts_cache_all_destination_service_cycles: 1,
        }
    }

}

impl IssueResourceSource {
    #[cfg(feature = "campaign")]
    pub const fn source_label(self) -> &'static str {
        match self {
            // Retained campaign result schema spelling. The direct simulator
            // does not compile or emit this compatibility field.
            Self::ArchitectureDefaults => "architecture_local_reviewed_hypothesis",
            Self::Carded(_) => "architecture_card",
            Self::HypotheticalIssuePorts(_) => "hypothetical_issue_ports",
            Self::Sm120Alpha => "sm120_alpha_measured_envelope",
        }
    }
}

/// Timing parameters. The executable configuration combines card latencies
/// with architecture-local defaults. [`IssueModelParams::from_card`] accepts a
/// complete card projection and executes only topology-compatible structural
/// resources; divergent card values fail closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IssueModelParams {
    /// Deployment architecture whose card and issue hypotheses produced these parameters.
    pub(crate) arch: ArchId,
    pub(crate) resource_source: IssueResourceSource,
    /// Published SM-local instruction hierarchy and fetch/decode structure.
    /// `None` exists for compact custom inputs and for the explicit SM120 alpha
    /// lane, which intentionally does not claim an instruction-cache
    /// capability.
    pub(crate) instruction_front_end: Option<InstructionFrontEnd>,
    /// Published per-SMSP and SM-shared memory issue pipeline. `None` is used
    /// by compact custom inputs and by the Flat-only SM120 alpha lane until a
    /// Blackwell memory-issue measurement is admitted.
    pub(crate) memory_issue_pipeline: Option<MemoryIssuePipeline>,
    /// Research-only scalar INT/FP execution occupancy. The explicit SM120
    /// alpha lane supplies its measured-envelope scalar values here; ordinary
    /// carded production constructors leave it unset until a full scheduler
    /// port model is admitted.
    pub(crate) scalar_execution_pipeline: Option<ScalarExecutionPipeline>,
    /// Number of independent issue slots available per SMSP.  Carded values
    /// are consumed by the transaction loop rather than rejected as dormant
    /// topology fields.
    pub(crate) issue_ports: u32,
    /// Whether a second issue transaction may be admitted in the same cycle.
    pub(crate) dual_issue_supported: bool,
    /// Source-hit return cadence for cacheable asynchronous copies.
    pub(crate) ldgsts_l1_hit_return_service: Option<LdgstsL1HitReturnService>,
    /// Additional SM-side return service for static `LDG.*.CONSTANT` loads.
    /// Compact custom inputs and the Flat-only SM120 alpha lane may omit it;
    /// carded SM86 constructors bind the architecture capability.
    pub(crate) constant_memory_return_service: Option<ConstantMemoryReturnService>,
    /// Per-SMSP immediate-constant cache used by CBANK operands. Synthetic
    /// compact inputs and SM120 alpha leave this absent so they exercise only the
    /// explicitly selected mechanism.
    pub(crate) immediate_constant_cache: Option<ImmediateConstantCache>,
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
    /// Fixed global-source to shared-destination transport after source data
    /// becomes available.
    pub(crate) ldgsts_global_to_shared_transport_latency: u64,
    /// Rational aggregate cache-all destination service shared by one SM.
    pub(crate) ldgsts_cache_all_destination_service_wavefronts: u64,
    pub(crate) ldgsts_cache_all_destination_service_cycles: u64,
    /// Resident-warp ceiling per SM. The current sm_86 hypothesis is 48;
    /// template architectures return a typed capability gap until their
    /// occupancy evidence is carded.
    pub(crate) max_resident_warps_per_sm: u32,
    /// Resident-CTA ceiling per SM. This is separate from warp and
    /// shared-memory residency (SM86 permits at most 16 CTAs even when fewer
    /// than 48 warps are resident).
    pub(crate) max_resident_blocks_per_sm: u32,
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

fn ldgsts_global_to_shared_transport_latency(arch: ArchId) -> Result<u64, IssueError> {
    let timing = arch
        .ldgsts_destination_wavefront_timing()
        .require("ldgsts_destination_wavefront_timing")?;
    if timing.global_to_shared_transport_cycles == 0 {
        return Err(IssueError::InvalidLdgstsDestinationWavefrontTiming {
            arch,
            cycles_per_wavefront: timing.global_to_shared_transport_cycles,
        });
    }
    Ok(timing.global_to_shared_transport_cycles)
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
    /// Change only the issue transaction width for a model-relative design
    /// experiment. Card geometry, latency sources, and all other scheduler
    /// resources remain unchanged; the resulting parameter set must be labeled
    /// hypothetical and cannot be used as a silicon-card identity.
    pub fn with_hypothetical_issue_ports(mut self, issue_ports: u32) -> Result<Self, IssueError> {
        if !(1..=4).contains(&issue_ports) {
            return Err(IssueError::InvalidHypotheticalIssuePorts { issue_ports });
        }
        self.validate_resource_source()?;
        let base = match self.resource_source {
            IssueResourceSource::ArchitectureDefaults => None,
            IssueResourceSource::Carded(resources) => Some(resources),
            IssueResourceSource::HypotheticalIssuePorts(_) => {
                return Err(IssueError::HypotheticalOverrideAlreadyApplied);
            }
            IssueResourceSource::Sm120Alpha => {
                return Err(IssueError::HypotheticalOverrideAlreadyApplied);
            }
        };
        self.issue_ports = issue_ports;
        // The override names the complete issue transaction width.  Retaining
        // a card's independent dual-issue bit would silently widen a
        // `--hypothetical-issue-ports 1` experiment back to two transactions
        // per cycle, contradicting both the CLI and the emitted label.
        self.dual_issue_supported = false;
        self.resource_source = IssueResourceSource::HypotheticalIssuePorts(base);
        Ok(self)
    }

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
        let ldgsts_l1_hit_return_service = arch
            .ldgsts_l1_hit_return_service()
            .require("ldgsts_l1_hit_return_service")?;
        let constant_memory_return_service = arch
            .constant_memory_return_service()
            .require("constant_memory_return_service")?;
        let ldgsts_l1_mshr_floor = ldgsts_l1_mshr_floor(arch)?;
        let ldgsts_completion_throughput = ldgsts_completion_throughput(arch)?;
        let ldgsts_cache_all_destination_wavefront_cycles =
            ldgsts_cache_all_destination_wavefront_cycles(arch)?;
        let ldgsts_global_to_shared_transport_latency =
            ldgsts_global_to_shared_transport_latency(arch)?;
        let (
            ldgsts_cache_all_destination_service_wavefronts,
            ldgsts_cache_all_destination_service_cycles,
        ) = ldgsts_cache_all_destination_service_rate(arch)?;
        Ok(Self {
            arch,
            resource_source: IssueResourceSource::ArchitectureDefaults,
            instruction_front_end: Some(
                arch.instruction_front_end()
                    .require("instruction_front_end")?,
            ),
            memory_issue_pipeline: Some(
                arch.memory_issue_pipeline()
                    .require("memory_issue_pipeline")?,
            ),
            scalar_execution_pipeline: None,
            issue_ports: 1,
            dual_issue_supported: false,
            ldgsts_l1_hit_return_service: Some(ldgsts_l1_hit_return_service),
            constant_memory_return_service: Some(constant_memory_return_service),
            immediate_constant_cache: Some(
                arch.immediate_constant_cache()
                    .require("immediate_constant_cache")?,
            ),
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
            ldgsts_global_to_shared_transport_latency,
            ldgsts_cache_all_destination_service_wavefronts,
            ldgsts_cache_all_destination_service_cycles,
            max_resident_warps_per_sm: hypotheses.max_resident_warps_per_sm,
            max_resident_blocks_per_sm: hypotheses.max_resident_blocks_per_sm,
        })
    }

    /// Construct the strict carded resource lane without a legacy fallback.
    /// Successful construction verifies that all five resource blocks and all
    /// required latency entries came from the same architecture-bound card.
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
        let ldgsts_l1_hit_return_service = arch
            .ldgsts_l1_hit_return_service()
            .require("ldgsts_l1_hit_return_service")?;
        let constant_memory_return_service = arch
            .constant_memory_return_service()
            .require("constant_memory_return_service")?;
        let ldgsts_l1_mshr_floor = ldgsts_l1_mshr_floor(arch)?;
        let ldgsts_completion_throughput = ldgsts_completion_throughput(arch)?;
        let ldgsts_cache_all_destination_wavefront_cycles =
            ldgsts_cache_all_destination_wavefront_cycles(arch)?;
        let ldgsts_global_to_shared_transport_latency =
            ldgsts_global_to_shared_transport_latency(arch)?;
        let (
            ldgsts_cache_all_destination_service_wavefronts,
            ldgsts_cache_all_destination_service_cycles,
        ) = ldgsts_cache_all_destination_service_rate(arch)?;
        Ok(Self {
            arch,
            resource_source: IssueResourceSource::Carded(resources),
            instruction_front_end: Some(
                arch.instruction_front_end()
                    .require("instruction_front_end")?,
            ),
            memory_issue_pipeline: Some(
                arch.memory_issue_pipeline()
                    .require("memory_issue_pipeline")?,
            ),
            scalar_execution_pipeline: None,
            issue_ports: resources.smsp.issue_ports,
            dual_issue_supported: resources.dual_issue_supported,
            ldgsts_l1_hit_return_service: Some(ldgsts_l1_hit_return_service),
            constant_memory_return_service: Some(constant_memory_return_service),
            immediate_constant_cache: Some(
                arch.immediate_constant_cache()
                    .require("immediate_constant_cache")?,
            ),
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
            ldgsts_global_to_shared_transport_latency,
            ldgsts_cache_all_destination_service_wavefronts,
            ldgsts_cache_all_destination_service_cycles,
            max_resident_warps_per_sm: resources.occupancy.max_resident_warps_per_sm,
            max_resident_blocks_per_sm: resources.occupancy.max_blocks,
        })
    }

    /// Construct SM120 alpha parameters from an explicit, card-bound envelope.
    ///
    /// The card contributes only the observed launch/occupancy contract.  All
    /// timing and scheduler values are supplied by `config`; this function
    /// never consults `ArchId::Sm86` capabilities and therefore cannot silently
    /// inherit Ampere defaults.
    pub fn from_card_alpha(
        expected_arch: ArchId,
        card: &ArchCard,
        config: Sm120AlphaConfig,
    ) -> Result<Self, IssueError> {
        let arch = card.arch_id()?;
        if expected_arch != ArchId::Sm120 || arch != ArchId::Sm120 {
            return Err(IssueError::ArchitectureMismatch {
                component: "SM120 alpha issue architecture card",
                expected: ArchId::Sm120,
                got: arch,
            });
        }
        let runtime = card.require_runtime_limits()?;
        if runtime.max_resident_warps_per_sm != config.max_resident_warps_per_sm {
            return Err(IssueError::CardedResourceMismatch {
                arch,
                field: "runtime_limits.max_resident_warps_per_sm",
                expected: u64::from(config.max_resident_warps_per_sm),
                got: u64::from(runtime.max_resident_warps_per_sm),
            });
        }
        // The card carries the public CC12.0 architectural ceiling (32
        // blocks/SM).  The alpha profile intentionally uses the M8 row-level
        // resource envelope (24 blocks/SM) so a shape-specific occupancy
        // observation is not mistaken for the card maximum.  Accept a
        // profile no larger than the card ceiling and keep the row value in
        // the emitted alpha provenance.
        if config.max_resident_blocks_per_sm > runtime.max_blocks_per_sm {
            return Err(IssueError::CardedResourceMismatch {
                arch,
                field: "runtime_limits.max_blocks_per_sm",
                expected: u64::from(runtime.max_blocks_per_sm),
                got: u64::from(config.max_resident_blocks_per_sm),
            });
        }
        if !(1..=4).contains(&config.issue_ports)
            || config.max_resident_warps_per_sm == 0
            || config.max_resident_blocks_per_sm == 0
            || config.scalar_in_flight_capacity == 0
            || config.ldgsts_l1_mshr_floor == 0
        {
            return Err(IssueError::CardedResourceMismatch {
                arch,
                field: "sm120_alpha_positive_resources",
                expected: 1,
                got: 0,
            });
        }
        let positive = [
            ("l1_hit_cycles", config.l1_hit_cycles),
            ("l2_hit_cycles", config.l2_hit_cycles),
            ("dram_cycles", config.dram_cycles),
            ("shared_cycles", config.shared_cycles),
            (
                "shared_additional_wavefront_cycles",
                config.shared_additional_wavefront_cycles,
            ),
            (
                "non_memory_write_barrier_cycles",
                config.non_memory_write_barrier_cycles,
            ),
            ("read_barrier_cycles", config.read_barrier_cycles),
            ("scalar_latency_cycles", config.scalar_latency_cycles),
            ("fma_latency_cycles", config.fma_latency_cycles),
            ("mad_latency_cycles", config.mad_latency_cycles),
            ("add_latency_cycles", config.add_latency_cycles),
            (
                "fma_initiation_interval_cycles",
                config.fma_initiation_interval_cycles,
            ),
            (
                "mad_initiation_interval_cycles",
                config.mad_initiation_interval_cycles,
            ),
            (
                "add_initiation_interval_cycles",
                config.add_initiation_interval_cycles,
            ),
            ("front_end_base_cycles", config.front_end_base_cycles),
            (
                "front_end_line_cycles_numerator",
                config.front_end_line_cycles_numerator,
            ),
            (
                "front_end_line_cycles_denominator",
                config.front_end_line_cycles_denominator,
            ),
            (
                "scalar_initiation_interval_cycles",
                config.scalar_initiation_interval_cycles,
            ),
            (
                "barrier_warp_initiation_interval_cycles",
                config.barrier_warp_initiation_interval_cycles,
            ),
            (
                "barrier_generation_rearm_cycles",
                config.barrier_generation_rearm_cycles,
            ),
            (
                "ldgsts_cache_all_payload_bytes_per_cycle",
                config.ldgsts_cache_all_payload_bytes_per_cycle,
            ),
            (
                "ldgsts_bypass_payload_bytes_per_cycle",
                config.ldgsts_bypass_payload_bytes_per_cycle,
            ),
            (
                "ldgsts_cache_all_destination_wavefront_cycles",
                config.ldgsts_cache_all_destination_wavefront_cycles,
            ),
            (
                "ldgsts_global_to_shared_transport_cycles",
                config.ldgsts_global_to_shared_transport_cycles,
            ),
            (
                "ldgsts_cache_all_destination_service_wavefronts",
                config.ldgsts_cache_all_destination_service_wavefronts,
            ),
            (
                "ldgsts_cache_all_destination_service_cycles",
                config.ldgsts_cache_all_destination_service_cycles,
            ),
        ];
        if let Some((name, value)) = positive.into_iter().find(|(_, value)| *value == 0) {
            return Err(IssueError::InvalidLatency {
                name,
                value: value as f64,
            });
        }
        let payload_common = config
            .ldgsts_cache_all_payload_bytes_per_cycle
            .max(config.ldgsts_bypass_payload_bytes_per_cycle);
        if payload_common % config.ldgsts_cache_all_payload_bytes_per_cycle != 0
            || payload_common % config.ldgsts_bypass_payload_bytes_per_cycle != 0
        {
            return Err(IssueError::InvalidLdgstsCompletionThroughput {
                arch,
                cache_all_bytes_per_cycle: config.ldgsts_cache_all_payload_bytes_per_cycle,
                bypass_bytes_per_cycle: config.ldgsts_bypass_payload_bytes_per_cycle,
            });
        }
        let ldgsts_l1_mshr_floor = NonZeroU32::new(config.ldgsts_l1_mshr_floor).ok_or(
            IssueError::InvalidLdgstsL1MshrFloor {
                arch,
                floor: config.ldgsts_l1_mshr_floor,
            },
        )?;
        Ok(Self {
            arch,
            resource_source: IssueResourceSource::Sm120Alpha,
            // The alpha lane deliberately does not instantiate any SM86
            // front-end, memory-pipeline, IMC, or constant-return capability.
            // The flat fast frontier therefore uses only the explicit
            // scalar/barrier envelope below; cache/async lanes are supplied by
            // the integrated full replay path.
            instruction_front_end: None,
            memory_issue_pipeline: None,
            scalar_execution_pipeline: Some(ScalarExecutionPipeline {
                latency_cycles: config.scalar_latency_cycles,
                initiation_interval_cycles: config.scalar_initiation_interval_cycles,
                in_flight_capacity: config.scalar_in_flight_capacity,
            }),
            issue_ports: config.issue_ports,
            dual_issue_supported: config.dual_issue_supported,
            ldgsts_l1_hit_return_service: None,
            constant_memory_return_service: None,
            immediate_constant_cache: None,
            l1_hit_latency: config.l1_hit_cycles,
            l1_miss_latency: config.l2_hit_cycles,
            global_flat_latency: config.dram_cycles,
            shared_write_barrier_latency: config.shared_cycles,
            shared_additional_wavefront_latency: config.shared_additional_wavefront_cycles,
            non_memory_write_barrier_latency: config.non_memory_write_barrier_cycles,
            read_barrier_latency: config.read_barrier_cycles,
            ldgsts_l1_mshr_floor,
            ldgsts_completion_throughput: LdgstsCompletionThroughput {
                cache_all_payload_bytes_per_cycle: config.ldgsts_cache_all_payload_bytes_per_cycle,
                bypass_payload_bytes_per_cycle: config.ldgsts_bypass_payload_bytes_per_cycle,
            },
            ldgsts_cache_all_destination_wavefront_cycles: config
                .ldgsts_cache_all_destination_wavefront_cycles,
            ldgsts_global_to_shared_transport_latency: config
                .ldgsts_global_to_shared_transport_cycles,
            ldgsts_cache_all_destination_service_wavefronts: config
                .ldgsts_cache_all_destination_service_wavefronts,
            ldgsts_cache_all_destination_service_cycles: config
                .ldgsts_cache_all_destination_service_cycles,
            max_resident_warps_per_sm: config.max_resident_warps_per_sm,
            max_resident_blocks_per_sm: config.max_resident_blocks_per_sm,
        })
    }

    /// Convenience constructor for callers that do not have a card object
    /// (custom inputs and alpha-only microbench harnesses). Production direct
    /// replay should call [`Self::from_card_alpha`] so runtime limits remain
    /// bound to the selected RTX 5090 card.
    pub fn sm120_alpha() -> Result<Self, IssueError> {
        let config = Sm120AlphaConfig::from_m8_m9();
        let runtime = config;
        if runtime.max_resident_warps_per_sm == 0 || runtime.max_resident_blocks_per_sm == 0 {
            return Err(IssueError::CardedResourceMismatch {
                arch: ArchId::Sm120,
                field: "sm120_alpha_runtime_limits",
                expected: 1,
                got: 0,
            });
        }
        // This compact-input path is intentionally cardless; all values still come
        // from the explicit alpha config above, never from SM86 capabilities.
        let params = Self {
            arch: ArchId::Sm120,
            resource_source: IssueResourceSource::Sm120Alpha,
            instruction_front_end: None,
            memory_issue_pipeline: None,
            scalar_execution_pipeline: Some(ScalarExecutionPipeline {
                latency_cycles: runtime.scalar_latency_cycles,
                initiation_interval_cycles: runtime.scalar_initiation_interval_cycles,
                in_flight_capacity: runtime.scalar_in_flight_capacity,
            }),
            issue_ports: runtime.issue_ports,
            dual_issue_supported: runtime.dual_issue_supported,
            ldgsts_l1_hit_return_service: None,
            constant_memory_return_service: None,
            immediate_constant_cache: None,
            l1_hit_latency: runtime.l1_hit_cycles,
            l1_miss_latency: runtime.l2_hit_cycles,
            global_flat_latency: runtime.dram_cycles,
            shared_write_barrier_latency: runtime.shared_cycles,
            shared_additional_wavefront_latency: runtime.shared_additional_wavefront_cycles,
            non_memory_write_barrier_latency: runtime.non_memory_write_barrier_cycles,
            read_barrier_latency: runtime.read_barrier_cycles,
            ldgsts_l1_mshr_floor: NonZeroU32::new(runtime.ldgsts_l1_mshr_floor).ok_or(
                IssueError::InvalidLdgstsL1MshrFloor {
                    arch: ArchId::Sm120,
                    floor: runtime.ldgsts_l1_mshr_floor,
                },
            )?,
            ldgsts_completion_throughput: LdgstsCompletionThroughput {
                cache_all_payload_bytes_per_cycle: runtime.ldgsts_cache_all_payload_bytes_per_cycle,
                bypass_payload_bytes_per_cycle: runtime.ldgsts_bypass_payload_bytes_per_cycle,
            },
            ldgsts_cache_all_destination_wavefront_cycles: runtime
                .ldgsts_cache_all_destination_wavefront_cycles,
            ldgsts_global_to_shared_transport_latency: runtime
                .ldgsts_global_to_shared_transport_cycles,
            ldgsts_cache_all_destination_service_wavefronts: runtime
                .ldgsts_cache_all_destination_service_wavefronts,
            ldgsts_cache_all_destination_service_cycles: runtime
                .ldgsts_cache_all_destination_service_cycles,
            max_resident_warps_per_sm: runtime.max_resident_warps_per_sm,
            max_resident_blocks_per_sm: runtime.max_resident_blocks_per_sm,
        };
        params.validate_resource_source()?;
        Ok(params)
    }

    pub const fn arch(&self) -> ArchId {
        self.arch
    }

    pub const fn resource_source(&self) -> IssueResourceSource {
        self.resource_source
    }

    pub const fn instruction_front_end(&self) -> Option<InstructionFrontEnd> {
        self.instruction_front_end
    }

    pub const fn memory_issue_pipeline(&self) -> Option<MemoryIssuePipeline> {
        self.memory_issue_pipeline
    }

    pub const fn scalar_execution_pipeline(&self) -> Option<ScalarExecutionPipeline> {
        self.scalar_execution_pipeline
    }

    pub const fn ldgsts_l1_hit_return_service(&self) -> Option<LdgstsL1HitReturnService> {
        self.ldgsts_l1_hit_return_service
    }

    pub const fn constant_memory_return_service(&self) -> Option<ConstantMemoryReturnService> {
        self.constant_memory_return_service
    }

    pub const fn immediate_constant_cache(&self) -> Option<ImmediateConstantCache> {
        self.immediate_constant_cache
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

    pub const fn ldgsts_global_to_shared_transport_latency(&self) -> u64 {
        self.ldgsts_global_to_shared_transport_latency
    }

    pub const fn ldgsts_cache_all_destination_service_wavefronts(&self) -> u64 {
        self.ldgsts_cache_all_destination_service_wavefronts
    }

    pub const fn ldgsts_cache_all_destination_service_cycles(&self) -> u64 {
        self.ldgsts_cache_all_destination_service_cycles
    }

    pub(super) fn ldgsts_delivery_units_per_cycle(&self) -> u64 {
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

    pub(super) fn ldgsts_completion_ready(
        &self,
        source_ready: u64,
        work: u64,
    ) -> Result<u64, IssueError> {
        let transported = checked_cycle_add(
            "LDGSTS global-to-shared transport",
            source_ready,
            self.ldgsts_global_to_shared_transport_latency,
        )?;
        checked_cycle_add(
            "LDGSTS payload completion floor",
            transported,
            self.ldgsts_delivery_cycles(work),
        )
    }

    pub(super) fn ldgsts_destination_service_cycles(
        &self,
        wavefronts: u64,
    ) -> Result<u64, IssueError> {
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

    pub const fn max_resident_blocks_per_sm(&self) -> u32 {
        self.max_resident_blocks_per_sm
    }

    pub const fn issue_ports(&self) -> u32 {
        self.issue_ports
    }

    pub const fn dual_issue_supported(&self) -> bool {
        self.dual_issue_supported
    }

    pub(crate) fn validate_resource_source(&self) -> Result<(), IssueError> {
        match self.resource_source {
            IssueResourceSource::ArchitectureDefaults => {
                if self.issue_ports != 1 || self.dual_issue_supported {
                    return Err(IssueError::ArchitectureDefaultMismatch {
                        arch: self.arch,
                        field: if self.issue_ports != 1 {
                            "issue_ports"
                        } else {
                            "dual_issue_supported"
                        },
                        expected: if self.issue_ports != 1 { 1 } else { 0 },
                        got: if self.issue_ports != 1 {
                            u64::from(self.issue_ports)
                        } else {
                            1
                        },
                    });
                }
                let hypotheses = self.arch.issue_hypotheses().require("issue_hypotheses")?;
                let instruction_front_end = self
                    .arch
                    .instruction_front_end()
                    .require("instruction_front_end")?;
                let Some(got_front_end) = self.instruction_front_end else {
                    return Err(IssueError::ArchitectureDefaultMismatch {
                        arch: self.arch,
                        field: "instruction_front_end_present",
                        expected: 1,
                        got: 0,
                    });
                };
                for (field, expected, got) in [
                    (
                        "instruction_front_end.line_bytes",
                        u64::from(instruction_front_end.line_bytes),
                        u64::from(got_front_end.line_bytes),
                    ),
                    (
                        "instruction_front_end.l0_sets",
                        u64::from(instruction_front_end.l0_sets),
                        u64::from(got_front_end.l0_sets),
                    ),
                    (
                        "instruction_front_end.l0_ways",
                        u64::from(instruction_front_end.l0_ways),
                        u64::from(got_front_end.l0_ways),
                    ),
                    (
                        "instruction_front_end.l1_sets",
                        u64::from(instruction_front_end.l1_sets),
                        u64::from(got_front_end.l1_sets),
                    ),
                    (
                        "instruction_front_end.l1_ways",
                        u64::from(instruction_front_end.l1_ways),
                        u64::from(got_front_end.l1_ways),
                    ),
                    (
                        "instruction_front_end.l0_to_l1_cycles",
                        instruction_front_end.l0_to_l1_cycles,
                        got_front_end.l0_to_l1_cycles,
                    ),
                    (
                        "instruction_front_end.l1_to_l0_cycles",
                        instruction_front_end.l1_to_l0_cycles,
                        got_front_end.l1_to_l0_cycles,
                    ),
                    (
                        "instruction_front_end.l1_request_ports",
                        u64::from(instruction_front_end.l1_request_ports),
                        u64::from(got_front_end.l1_request_ports),
                    ),
                    (
                        "instruction_front_end.l1_reply_ports",
                        u64::from(instruction_front_end.l1_reply_ports),
                        u64::from(got_front_end.l1_reply_ports),
                    ),
                    (
                        "instruction_front_end.stream_buffer_lines",
                        u64::from(instruction_front_end.stream_buffer_lines),
                        u64::from(got_front_end.stream_buffer_lines),
                    ),
                    (
                        "instruction_front_end.stream_buffers_per_subcore",
                        u64::from(instruction_front_end.stream_buffers_per_subcore),
                        u64::from(got_front_end.stream_buffers_per_subcore),
                    ),
                    (
                        "instruction_front_end.prefetches_per_cycle",
                        u64::from(instruction_front_end.prefetches_per_cycle),
                        u64::from(got_front_end.prefetches_per_cycle),
                    ),
                    (
                        "instruction_front_end.instruction_buffer_entries",
                        u64::from(instruction_front_end.instruction_buffer_entries),
                        u64::from(got_front_end.instruction_buffer_entries),
                    ),
                    (
                        "instruction_front_end.fetch_decode_width",
                        u64::from(instruction_front_end.fetch_decode_width),
                        u64::from(got_front_end.fetch_decode_width),
                    ),
                    (
                        "instruction_front_end.fetch_to_issue_stages",
                        instruction_front_end.fetch_to_issue_stages,
                        got_front_end.fetch_to_issue_stages,
                    ),
                ] {
                    if got != expected {
                        return Err(IssueError::ArchitectureDefaultMismatch {
                            arch: self.arch,
                            field,
                            expected,
                            got,
                        });
                    }
                }
                let memory_issue_pipeline = self
                    .arch
                    .memory_issue_pipeline()
                    .require("memory_issue_pipeline")?;
                let Some(got_memory_pipeline) = self.memory_issue_pipeline else {
                    return Err(IssueError::ArchitectureDefaultMismatch {
                        arch: self.arch,
                        field: "memory_issue_pipeline_present",
                        expected: 1,
                        got: 0,
                    });
                };
                if got_memory_pipeline != memory_issue_pipeline {
                    return Err(IssueError::ArchitectureDefaultMismatch {
                        arch: self.arch,
                        field: "memory_issue_pipeline_configuration",
                        expected: 1,
                        got: 0,
                    });
                }
                let immediate_constant_cache = self
                    .arch
                    .immediate_constant_cache()
                    .require("immediate_constant_cache")?;
                if self.immediate_constant_cache != Some(immediate_constant_cache) {
                    return Err(IssueError::ArchitectureDefaultMismatch {
                        arch: self.arch,
                        field: "immediate_constant_cache_configuration",
                        expected: 1,
                        got: 0,
                    });
                }
                let shared_additional_wavefront_latency =
                    shared_additional_wavefront_latency(self.arch)?;
                let ldgsts_l1_mshr_floor = ldgsts_l1_mshr_floor(self.arch)?;
                let ldgsts_completion_throughput = ldgsts_completion_throughput(self.arch)?;
                let ldgsts_cache_all_destination_wavefront_cycles =
                    ldgsts_cache_all_destination_wavefront_cycles(self.arch)?;
                let ldgsts_global_to_shared_transport_latency =
                    ldgsts_global_to_shared_transport_latency(self.arch)?;
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
                        "ldgsts_global_to_shared_transport_latency",
                        ldgsts_global_to_shared_transport_latency,
                        self.ldgsts_global_to_shared_transport_latency,
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
                    (
                        "max_resident_blocks_per_sm",
                        u64::from(hypotheses.max_resident_blocks_per_sm),
                        u64::from(self.max_resident_blocks_per_sm),
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
            IssueResourceSource::Carded(resources) => {
                let expected = [
                    (
                        "smsp.count",
                        SUB_CORES as u64,
                        u64::from(resources.smsp.count),
                    ),
                    (
                        "dep_barriers.count",
                        DEP_BARRIERS as u64,
                        u64::from(resources.dep_barriers.count),
                    ),
                    // Issue-port width and dual issue are now consumed by the
                    // scheduler.  Keep only the structural count/barrier
                    // checks here; arbitrary carded widths are valid.
                    (
                        "occupancy.max_resident_warps_per_sm",
                        u64::from(self.max_resident_warps_per_sm),
                        u64::from(resources.occupancy.max_resident_warps_per_sm),
                    ),
                    (
                        "occupancy.max_blocks",
                        u64::from(self.max_resident_blocks_per_sm),
                        u64::from(resources.occupancy.max_blocks),
                    ),
                ];
                for (field, expected, got) in expected {
                    if expected != got {
                        return Err(IssueError::CardedResourceMismatch {
                            arch: self.arch,
                            field,
                            expected,
                            got,
                        });
                    }
                }
                if resources.smsp.issue_ports == 0 {
                    return Err(IssueError::CardedResourceMismatch {
                        arch: self.arch,
                        field: "smsp.issue_ports",
                        expected: 1,
                        got: 0,
                    });
                }
                if self.issue_ports != resources.smsp.issue_ports {
                    return Err(IssueError::CardedResourceMismatch {
                        arch: self.arch,
                        field: "smsp.issue_ports",
                        expected: u64::from(resources.smsp.issue_ports),
                        got: u64::from(self.issue_ports),
                    });
                }
                if self.dual_issue_supported != resources.dual_issue_supported {
                    return Err(IssueError::CardedResourceMismatch {
                        arch: self.arch,
                        field: "dual_issue.supported",
                        expected: if resources.dual_issue_supported { 1 } else { 0 },
                        got: if self.dual_issue_supported { 1 } else { 0 },
                    });
                }
                Ok(())
            }
            IssueResourceSource::HypotheticalIssuePorts(base) => {
                if !(1..=4).contains(&self.issue_ports) {
                    return Err(IssueError::InvalidHypotheticalIssuePorts {
                        issue_ports: self.issue_ports,
                    });
                }
                // Validate every untouched field using the original path.
                // The override must not bypass topology or latency checks.
                let mut original = *self;
                original.resource_source = match base {
                    Some(resources) => {
                        // Validate the untouched card topology against its
                        // original dual-issue declaration.  The live
                        // hypothetical value intentionally disables that
                        // separate admission path because `issue_ports` is
                        // the complete width named by the override.
                        original.dual_issue_supported = resources.dual_issue_supported;
                        IssueResourceSource::Carded(resources)
                    }
                    None => {
                        original.dual_issue_supported = false;
                        IssueResourceSource::ArchitectureDefaults
                    }
                };
                original.issue_ports = base.map_or(1, |resources| resources.smsp.issue_ports);
                original.validate_resource_source()
            }
            IssueResourceSource::Sm120Alpha => {
                if self.arch != ArchId::Sm120 {
                    return Err(IssueError::ArchitectureMismatch {
                        component: "SM120 alpha issue envelope",
                        expected: ArchId::Sm120,
                        got: self.arch,
                    });
                }
                if !(1..=4).contains(&self.issue_ports)
                    || self.max_resident_warps_per_sm == 0
                    || self.max_resident_blocks_per_sm == 0
                {
                    return Err(IssueError::CardedResourceMismatch {
                        arch: self.arch,
                        field: "sm120_alpha_resources",
                        expected: 1,
                        got: u64::from(self.issue_ports),
                    });
                }
                Ok(())
            }
        }
    }

    /// Non-memory producer visibility (one copy for BOTH memory modes).
    pub(super) fn non_memory_ready(&self, t: u64) -> Result<u64, IssueError> {
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
    pub(super) fn shared_ready(&self, t: u64, wavefronts: u64) -> Result<u64, IssueError> {
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
    pub(super) fn store_visibility_ready(&self, t: u64) -> Result<u64, IssueError> {
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

    /// Replace the alpha lane's L1 timing with an explicit effective envelope
    /// measured by a real SM120 probe. Geometry remains a separate effective
    /// scenario value; the SM120 full path supplies it automatically.
    pub fn with_l1_timing(mut self, hit: u64, miss: u64) -> Result<Self, IssueError> {
        if hit == 0 || hit >= miss {
            return Err(IssueError::InvalidL1TimingOverride { hit, miss });
        }
        self.l1_hit_latency = hit;
        self.l1_miss_latency = miss;
        Ok(self)
    }
}
