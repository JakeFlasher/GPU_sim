//! (function, pc) join of the trace's static kernel metadata with the
//! Phase-1E control-bits sidecar — the issue-control table the SM model
//! replays against (design §6.2).
//!
//! Join key (verified byte-exact on all 17 M1 traces, 2026-07-06):
//! `ExecInfo.kernels[fuid].function_name` == sidecar function key, and the
//! two static pc sets are EQUAL. Both directions of that equality are
//! enforced fail-closed here: a pc only in the exec info means the sidecar
//! is stale (rebuilt binary, wrong file); a pc only in the sidecar means
//! the exec info is truncated. Either way the join is not trustworthy.
//!
//! The control bits themselves come ONLY from the sidecar
//! ([`crate::control_bits`]): the exec info's embedded `control_bits` block
//! measured 88/5,736 agreement against the re-decoded binary encoding and
//! is not even parsed (see `sim_trace::exec_info` module docs).

use std::collections::BTreeMap;

use sim_arch::{ArchError, ArchId, CapabilityError};
use sim_mem::{
    LdgstsL1Policy, MemRole, classify_ldgsts_l1_policy, mem_role_of_checked, opcode_head,
};
use sim_trace::ExecInfo;

use crate::control_bits::{ControlBits, ControlSidecar, DepBarrier, FunctionControl, WaitMask};
use crate::sass_operands::{FixedLatencyOperandStatus, project_fixed_latency_operands};

#[derive(Debug, thiserror::Error)]
pub enum ProgramError {
    #[error(
        "arch mismatch: exec info reports architecture_version {exec_version} for kernel \
         {kernel_name:?}, sidecar is {sidecar_arch:?}"
    )]
    ArchMismatch {
        kernel_name: String,
        exec_version: u32,
        sidecar_arch: String,
    },
    #[error("sidecar architecture {arch:?} is not supported by the simulator: {source}")]
    UnsupportedArchitecture {
        arch: String,
        #[source]
        source: ArchError,
    },
    #[error("schedule control-layout capability is unavailable: {0}")]
    ControlLayoutCapability(#[from] CapabilityError),
    #[error("tensor-pipeline capability is unavailable: {0}")]
    TensorPipelineCapability(#[source] CapabilityError),
    #[error("scalar execution-pipeline capability is unavailable: {0}")]
    ScalarExecutionPipelineCapability(#[source] CapabilityError),
    #[error("CTA-barrier issue capability is unavailable: {0}")]
    CtaBarrierPipelineCapability(#[source] CapabilityError),
    #[error("branch-resolution capability is unavailable: {0}")]
    BranchPipelineCapability(#[source] CapabilityError),
    #[error(
        "tensor instruction {opcode:?} has no execution-pipeline model for architecture {arch}"
    )]
    TensorPipelineUnsupported { arch: ArchId, opcode: String },
    #[error("async-copy group capability is unavailable: {0}")]
    AsyncCopyGroupCapability(#[source] CapabilityError),
    #[error(
        "async-copy control instruction {opcode:?} is predicated by {predicate:?} on {arch}; \
         divergent group entanglement is not modeled"
    )]
    AsyncCopyControlPredicated {
        arch: ArchId,
        opcode: String,
        predicate: String,
    },
    #[error(
        "async-copy control instruction {opcode:?} has unsupported operands or modifiers on {arch}"
    )]
    AsyncCopyControlUnsupported { arch: ArchId, opcode: String },
    #[error(
        "async-copy control instruction {opcode:?} has control bits {control:?} on {arch}; \
         expected {expected}"
    )]
    AsyncCopyControlLayout {
        arch: ArchId,
        opcode: String,
        control: ControlBits,
        expected: &'static str,
    },
    #[error("memory-role classification failed for architecture-aware schedule join: {0}")]
    MemRoleClassification(#[from] sim_mem::MemRoleError),
    #[error("LDGSTS source-cache classification failed during schedule join: {0}")]
    LdgstsCacheClassification(#[source] sim_mem::CacheOpError),
    #[error("sidecar arch {arch:?} is not of the form sm_<digits>")]
    SidecarArchUnparseable { arch: String },
    #[error(
        "function {function_name:?} (kernel {kernel_name:?}) not present in control-bits \
         sidecar for binary {binary:?}"
    )]
    FunctionMissing {
        function_name: String,
        kernel_name: String,
        binary: String,
    },
    #[error(
        "function {function_name:?}: {count} static pc(s) present in exec info but missing \
         from the control-bits sidecar (first: {first}) — sidecar is stale or from a \
         different binary"
    )]
    PcMissingFromSidecar {
        function_name: String,
        count: usize,
        first: u64,
    },
    #[error(
        "function {function_name:?}: {count} pc(s) present in the control-bits sidecar but \
         missing from exec info (first: {first}) — exec info is truncated or from a \
         different binary"
    )]
    PcMissingFromExecInfo {
        function_name: String,
        count: usize,
        first: u64,
    },
}

/// How the issue model treats an instruction, classified from the sidecar's
/// nvdisasm `opcode_text` head (never from NVBit's `op_code`).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpClass {
    /// `BAR.*` — block-wide barrier: the warp arrives, then waits until
    /// every unfinished warp of its block is waiting (or finished).
    BlockBarrier,
    /// Ampere `LDGDEPBAR`: close the warp's open non-bulk async-copy group.
    AsyncCopyCommit,
    /// Ampere `DEPBAR.LE SB0, N`: wait until no more than the newest `N`
    /// committed groups remain pending. The full `u32` immediate is stored in
    /// [`IssueSlot`] so this hot classification remains byte-sized.
    AsyncCopyWait,
    /// Everything else. Dependence and dispatch timing still comes from the
    /// control bits and dynamic memory class; independent execution-resource
    /// timing, when characterized, is carried by [`PipelineClass`].
    Other,
}

/// Execution resource an instruction occupies after dispatch. `DispatchOnly`
/// is an explicit fidelity boundary: the instruction still consumes the
/// SMSP's single dispatch opportunity, but no execution-unit availability is
/// invented for pipe families that have not yet been characterized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineClass {
    DispatchOnly,
    Branch {
        next_issue_interval_cycles: u64,
    },
    Barrier {
        /// Earliest cycles between warp arrivals admitted by the one
        /// SM-shared CTA-barrier issue server.
        warp_initiation_interval_cycles: u64,
        /// Additional issue-only re-arm for the next generation of the same
        /// CTA. Ordinary post-rendezvous work is never delayed by this value.
        generation_rearm_cycles: u64,
    },
    Tensor {
        /// Earliest number of cycles between issues to one SMSP's Tensor Core.
        initiation_interval_cycles: u64,
    },
    ScalarAlu {
        /// Architecture-local pipelined INT/FP execution path.  The queue
        /// and latency are consumed by the replay actor; this field remains
        /// explicit so unsupported families cannot inherit it silently.
        latency_cycles: u64,
        initiation_interval_cycles: u64,
        in_flight_capacity: u32,
    },
}

impl PipelineClass {
    pub const fn initiation_interval_cycles(self) -> Option<u64> {
        match self {
            Self::DispatchOnly => None,
            Self::Branch {
                next_issue_interval_cycles,
            } => Some(next_issue_interval_cycles),
            Self::Barrier {
                warp_initiation_interval_cycles,
                ..
            } => Some(warp_initiation_interval_cycles),
            Self::Tensor {
                initiation_interval_cycles,
            } => Some(initiation_interval_cycles),
            Self::ScalarAlu {
                initiation_interval_cycles,
                ..
            } => Some(initiation_interval_cycles),
        }
    }
}

/// Per-pc issue control: the sidecar's decoded control word plus the opcode
/// classifications the scheduler and the L1 model need.
///
/// ```compile_fail
/// use sim_sm::{ControlBits, IssueSlot};
///
/// fn overwrite(slot: &mut IssueSlot, control: ControlBits) {
///     slot.control = control;
/// }
/// ```
#[derive(Debug, Clone)]
pub struct IssueSlot {
    pub(crate) control: ControlBits,
    pub(crate) op_class: OpClass,
    /// Meaningful only when `op_class == OpClass::AsyncCopyWait`; construction
    /// validates and preserves the complete `u32` sidecar immediate.
    pub(crate) async_copy_wait_max_pending_groups: u32,
    pub(crate) pipeline_class: PipelineClass,
    pub(crate) mem_role: MemRole,
    /// Present exactly for static `LDGSTS` instructions. The dynamic record
    /// carries this into async-copy source routing without reparsing SASS in the
    /// issue loop.
    pub(crate) ldgsts_l1_policy: Option<LdgstsL1Policy>,
    /// Static exact-family register/source/reuse projection. This is diagnostic
    /// input only until Control/Allocate timing passes live transfer.
    pub(crate) fixed_latency_operands: FixedLatencyOperandStatus,
    /// Static phase markers used in dynamic readiness checks. Sidecar SASS is
    /// canonical ASCII, so classify once instead of reparsing on every scan.
    pub(crate) clocklo_read: bool,
    pub(crate) gpu_membar: bool,
    /// Full nvdisasm text, kept for diagnostics (static tables are small).
    pub(crate) opcode_text: String,
}

pub(crate) fn is_clocklo_opcode_text(text: &str) -> bool {
    let text = text.trim_ascii();
    text.starts_with("CS2R ")
        && text
            .rsplit_once(',')
            .is_some_and(|(_, source)| source.trim_ascii() == "SR_CLOCKLO")
}

pub(crate) fn is_gpu_membar_opcode_text(text: &str) -> bool {
    text.split_ascii_whitespace().next() == Some("MEMBAR.SC.GPU")
}

impl IssueSlot {
    pub fn control(&self) -> &ControlBits {
        &self.control
    }

    pub fn op_class(&self) -> OpClass {
        self.op_class
    }

    pub fn async_copy_wait_max_pending_groups(&self) -> Option<u32> {
        (self.op_class == OpClass::AsyncCopyWait).then_some(self.async_copy_wait_max_pending_groups)
    }

    /// Ordinary scoreboard barriers named by a compiler-emitted `DEPBAR`
    /// operand. Ampere `DEPBAR.LE SB1..SB5, 0x0[, {..}]` does not participate
    /// in the SB0 asynchronous-copy group protocol: the issue model waits on
    /// every listed ordinary dependence-barrier counter/clear state and never
    /// commits or consumes an asynchronous-copy group for this instruction.
    pub(crate) fn ordinary_dependency_wait_mask(&self) -> WaitMask {
        match parse_dependency_barrier(&self.opcode_text) {
            Some(DependencyBarrierOperand::OrdinaryWait(mask)) => mask,
            Some(DependencyBarrierOperand::AsyncCopyWait(_)) | None => WaitMask::NONE,
        }
    }

    pub fn pipeline_class(&self) -> PipelineClass {
        self.pipeline_class
    }

    pub fn mem_role(&self) -> MemRole {
        self.mem_role
    }

    pub fn ldgsts_l1_policy(&self) -> Option<LdgstsL1Policy> {
        self.ldgsts_l1_policy
    }

    pub fn fixed_latency_operands(&self) -> &FixedLatencyOperandStatus {
        &self.fixed_latency_operands
    }

    pub(crate) const fn is_clocklo_read(&self) -> bool {
        self.clocklo_read
    }

    pub(crate) const fn is_gpu_membar(&self) -> bool {
        self.gpu_membar
    }

    pub fn opcode_text(&self) -> &str {
        &self.opcode_text
    }
}

/// The replayable issue-control table for one traced function. Construction is
/// restricted to [`build_schedules`], so external replay callers cannot forge
/// opcode roles or control words.
///
/// ```compile_fail
/// use std::collections::BTreeMap;
///
/// use sim_arch::ArchId;
/// use sim_sm::FunctionSchedule;
///
/// fn forge() -> FunctionSchedule {
///     FunctionSchedule {
///         arch: ArchId::Sm86,
///         function_unique_id: 1,
///         kernel_name: String::new(),
///         function_name: String::new(),
///         by_pc: BTreeMap::new(),
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct FunctionSchedule {
    /// Deployment architecture proven by the trace/sidecar join. Consumers must
    /// match this identity before pairing the schedule with architecture-local
    /// timing or memory configuration.
    pub(crate) arch: ArchId,
    pub(crate) function_unique_id: i32,
    pub(crate) kernel_name: String,
    pub(crate) function_name: String,
    pub(crate) by_pc: BTreeMap<u64, IssueSlot>,
}

impl FunctionSchedule {
    pub fn arch(&self) -> ArchId {
        self.arch
    }

    pub fn function_unique_id(&self) -> i32 {
        self.function_unique_id
    }

    pub fn kernel_name(&self) -> &str {
        &self.kernel_name
    }

    pub fn function_name(&self) -> &str {
        &self.function_name
    }

    pub fn issue_slot(&self, pc: u64) -> Option<&IssueSlot> {
        self.by_pc.get(&pc)
    }

    pub fn static_pcs(&self) -> impl ExactSizeIterator<Item = u64> + '_ {
        self.by_pc.keys().copied()
    }
}

fn classify(
    opcode_text: &str,
    control: ControlBits,
    predicate: Option<&str>,
    arch: ArchId,
) -> Result<OpClass, ProgramError> {
    let head = opcode_head(opcode_text);
    if head == "BAR" {
        return Ok(OpClass::BlockBarrier);
    }
    if !matches!(head, "LDGDEPBAR" | "DEPBAR") {
        return Ok(OpClass::Other);
    }

    arch.async_copy_group_semantics()
        .require("async_copy_group_semantics")
        .map_err(ProgramError::AsyncCopyGroupCapability)?;
    if let Some(predicate) = predicate {
        return Err(ProgramError::AsyncCopyControlPredicated {
            arch,
            opcode: opcode_text.to_owned(),
            predicate: predicate.to_owned(),
        });
    }

    match head {
        "LDGDEPBAR" => {
            if opcode_text.trim() != "LDGDEPBAR" {
                return Err(ProgramError::AsyncCopyControlUnsupported {
                    arch,
                    opcode: opcode_text.to_owned(),
                });
            }
            // SB0 is the commit protocol's write barrier. NVCC may compose the
            // instruction with ordinary SB1--SB5 waits; the normal scoreboard
            // path already enforces those bits before this commit can issue.
            if control.write_barrier.map(|barrier| barrier.wire()) != Some(0)
                || control.read_barrier.is_some()
                || control.wait_mask.bits() & 1 != 0
            {
                return Err(ProgramError::AsyncCopyControlLayout {
                    arch,
                    opcode: opcode_text.to_owned(),
                    control,
                    expected: "write barrier SB0, no read barrier, and an ordinary wait mask that excludes reserved SB0",
                });
            }
            Ok(OpClass::AsyncCopyCommit)
        }
        "DEPBAR" => {
            let Some(operand) = parse_dependency_barrier(opcode_text) else {
                return Err(ProgramError::AsyncCopyControlUnsupported {
                    arch,
                    opcode: opcode_text.to_owned(),
                });
            };
            match operand {
                DependencyBarrierOperand::AsyncCopyWait(_) => {
                    if control.write_barrier.is_some() || control.read_barrier.is_some() {
                        return Err(ProgramError::AsyncCopyControlLayout {
                            arch,
                            opcode: opcode_text.to_owned(),
                            control,
                            expected: "no ordinary write or read barrier; an encoded ordinary wait mask may compose with SB0",
                        });
                    }
                    Ok(OpClass::AsyncCopyWait)
                }
                // SB1..SB5 are ordinary compiler scoreboards. Their producer
                // latency and wait timing come from the normal control/barrier
                // path; only SB0 is the non-bulk async-copy wait protocol.
                DependencyBarrierOperand::OrdinaryWait(_) => {
                    if control.write_barrier.is_some()
                        || control.read_barrier.is_some()
                        || control.wait_mask.bits() != 0
                    {
                        return Err(ProgramError::AsyncCopyControlLayout {
                            arch,
                            opcode: opcode_text.to_owned(),
                            control,
                            expected: "no ordinary write barrier, read barrier, or encoded wait mask for an explicit ordinary operand set",
                        });
                    }
                    Ok(OpClass::Other)
                }
            }
        }
        _ => unreachable!("dependency-barrier heads were filtered above"),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DependencyBarrierOperand {
    AsyncCopyWait(u32),
    OrdinaryWait(WaitMask),
}

/// Parse the exact Ampere dependency-barrier operand forms admitted by this
/// model. SB0 retains the full asynchronous-copy pending-group immediate.
/// SB1..SB5 are admitted only for the compiler-emitted zero-threshold form,
/// optionally followed by one canonical `{n,...}` set of unique scoreboard
/// ordinals 0..5. SB0 remains exclusive to the async-copy form and therefore
/// rejects a third operand. Other names, modifiers, thresholds, and operand
/// shapes remain fail-closed.
fn parse_dependency_barrier(opcode_text: &str) -> Option<DependencyBarrierOperand> {
    let text = opcode_text.trim();
    let operands = text.strip_prefix("DEPBAR.LE")?;
    if operands.is_empty() || !operands.starts_with(char::is_whitespace) {
        return None;
    }
    let (primary, tail) = operands.split_once(',')?;
    let barrier = match primary.trim() {
        "SB0" => DepBarrier::new(0).ok()?,
        "SB1" => DepBarrier::new(1).ok()?,
        "SB2" => DepBarrier::new(2).ok()?,
        "SB3" => DepBarrier::new(3).ok()?,
        "SB4" => DepBarrier::new(4).ok()?,
        "SB5" => DepBarrier::new(5).ok()?,
        _ => return None,
    };
    let (immediate, additional) = match tail.split_once(',') {
        Some((immediate, additional)) => (immediate.trim(), Some(additional.trim())),
        None => (tail.trim(), None),
    };
    if immediate.is_empty() {
        return None;
    }
    let immediate = parse_u32_literal(immediate)?;
    if barrier.wire() == 0 {
        return additional
            .is_none()
            .then_some(DependencyBarrierOperand::AsyncCopyWait(immediate));
    }
    if immediate != 0 {
        return None;
    }
    let mut bits = 1_u8 << barrier.wire();
    if let Some(additional) = additional {
        let members = additional.strip_prefix('{')?.strip_suffix('}')?;
        if members.is_empty() {
            return None;
        }
        for member in members.split(',') {
            let wire = match member.trim() {
                "0" => 0,
                "1" => 1,
                "2" => 2,
                "3" => 3,
                "4" => 4,
                "5" => 5,
                _ => return None,
            };
            let bit = 1_u8 << wire;
            if bits & bit != 0 {
                return None;
            }
            bits |= bit;
        }
    }
    Some(DependencyBarrierOperand::OrdinaryWait(
        WaitMask::new(bits).ok()?,
    ))
}

fn parse_async_copy_wait(opcode_text: &str) -> Option<u32> {
    match parse_dependency_barrier(opcode_text)? {
        DependencyBarrierOperand::AsyncCopyWait(max_pending_groups) => Some(max_pending_groups),
        DependencyBarrierOperand::OrdinaryWait(_) => None,
    }
}

fn parse_u32_literal(text: &str) -> Option<u32> {
    if let Some(hex) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        if hex.is_empty() {
            None
        } else {
            u32::from_str_radix(hex, 16).ok()
        }
    } else {
        text.parse().ok()
    }
}

fn parse_direct_branch_target(opcode_text: &str) -> Option<u64> {
    let operands = opcode_text.strip_prefix("BRA ")?.trim();
    let target = operands
        .rsplit_once(',')
        .map_or(operands, |(_, target)| target.trim());
    parse_u32_literal(target).map(u64::from)
}

fn classify_pipeline(
    pc: u64,
    opcode_text: &str,
    control: ControlBits,
    predicate: Option<&str>,
    arch: ArchId,
) -> Result<PipelineClass, ProgramError> {
    let head = opcode_head(opcode_text);
    if head == "BAR" {
        let issue = arch
            .cta_barrier_issue()
            .require("cta_barrier_issue")
            .map_err(ProgramError::CtaBarrierPipelineCapability)?;
        return Ok(PipelineClass::Barrier {
            warp_initiation_interval_cycles: issue.warp_initiation_interval_cycles,
            generation_rearm_cycles: issue.generation_rearm_cycles,
        });
    }
    let opcode = opcode_text.split_ascii_whitespace().next().unwrap_or("");
    let branch_target = parse_direct_branch_target(opcode_text);
    if let Some(target) = branch_target
        && target != pc
        && predicate.is_some()
        && control.stall == 5
        && control.yield_flag
    {
        let timing = arch
            .branch_resolution()
            .require("branch_resolution")
            .map_err(ProgramError::BranchPipelineCapability)?;
        let next_issue_interval_cycles = if target > pc {
            timing.next_issue_interval_cycles
        } else {
            timing.uniform_backward_target_resolution_cycles
        };
        return Ok(PipelineClass::Branch {
            next_issue_interval_cycles,
        });
    }
    // Ampere's integer and FP32 scalar instructions enter the finite
    // pipelined execution window described by the architecture card.  Memory
    // and tensor heads are handled below/through the dedicated classifiers;
    // SFU, uniform, and control-only forms remain dispatch-only until their
    // own resources are carded.
    if matches!(head, "IADD3" | "LOP3" | "IMAD" | "FADD" | "FFMA" | "FMUL") {
        let pipeline = arch
            .scalar_execution_pipeline()
            .require("scalar_execution_pipeline")
            .map_err(ProgramError::ScalarExecutionPipelineCapability)?;
        return Ok(PipelineClass::ScalarAlu {
            latency_cycles: pipeline.latency_cycles,
            initiation_interval_cycles: pipeline.initiation_interval_cycles,
            in_flight_capacity: pipeline.in_flight_capacity,
        });
    }
    if !matches!(head, "HMMA" | "IMMA" | "BMMA" | "DMMA") {
        return Ok(PipelineClass::DispatchOnly);
    }

    let intervals = arch
        .tensor_issue_intervals()
        .require("tensor_issue_intervals")
        .map_err(ProgramError::TensorPipelineCapability)?;
    let initiation_interval_cycles = match opcode {
        "HMMA.16816.F32" | "HMMA.16816.F32.BF16" => intervals.hmma_16816_f16_bf16,
        "HMMA.SP.16832.F32.BF16" => intervals.hmma_sparse_16832_f16_bf16,
        "IMMA.16816.S8.S8" => intervals.imma_16816_s8,
        _ => {
            return Err(ProgramError::TensorPipelineUnsupported {
                arch,
                opcode: opcode.to_owned(),
            });
        }
    };
    Ok(PipelineClass::Tensor {
        initiation_interval_cycles,
    })
}

/// Return the architectural predicate register named by a sidecar predicate
/// token (`@P0`, `@!P0`) or an SASS destination operand (`P0,`).  Keeping the
/// register identity in the compact-loop shape check matters: a preceding
/// `ISETP P1` does not provide the value consumed by a branch guarded by
/// `@P0`.
fn predicate_register(token: &str) -> Option<&str> {
    let token = token.strip_prefix('@').unwrap_or(token);
    let token = token.strip_prefix('!').unwrap_or(token);
    let digits = token.strip_prefix('P')?;
    (digits.len() == 1 && digits.as_bytes()[0] <= b'6').then_some(token)
}

/// Return the destination predicate of an integer predicate-setting SASS
/// instruction. Other predicate-setting heads remain on the ordinary branch
/// path until they have an independent transfer study; a branch's dynamic
/// condition is never inferred from textual control flow alone.
fn predicate_producer_register(opcode_text: &str) -> Option<&str> {
    (opcode_head(opcode_text) == "ISETP")
        .then(|| opcode_text.split_ascii_whitespace().nth(1))
        .flatten()
        .and_then(|operand| predicate_register(operand.trim_end_matches(',')))
}

/// Conservative static shape boundary for the compact-loop branch handoff.
/// The independent SM86 control matrix exercises a 48-byte back edge; the
/// 64-byte ceiling keeps the promoted rule tied to that compact SASS shape and
/// leaves all longer production loops on the established compiler/three-cycle
/// branch path.
const COMPACT_BACKWARD_PREDICATE_MAX_DISTANCE_BYTES: u64 = 64;

fn compact_backward_predicate_interval(
    pc: u64,
    opcode_text: &str,
    pipeline_class: PipelineClass,
    function: &FunctionControl,
    branch_predicate: Option<&str>,
    arch: ArchId,
) -> Result<Option<u64>, ProgramError> {
    if !matches!(pipeline_class, PipelineClass::Branch { .. }) {
        return Ok(None);
    }
    let Some(target) = parse_direct_branch_target(opcode_text) else {
        return Ok(None);
    };
    if target >= pc || pc - target > COMPACT_BACKWARD_PREDICATE_MAX_DISTANCE_BYTES {
        return Ok(None);
    }
    let Some(branch_predicate) = branch_predicate.and_then(predicate_register) else {
        return Ok(None);
    };
    let Some(previous_pc) = pc.checked_sub(16) else {
        return Ok(None);
    };
    if !function.by_pc.get(&previous_pc).is_some_and(|producer| {
        // The identification control has a non-yielding stall-13 predicate
        // producer; other predicate schedules have not been promoted and
        // remain on the ordinary branch path.
        predicate_producer_register(&producer.opcode_text)
            .is_some_and(|producer_predicate| producer_predicate == branch_predicate)
            && producer.control.stall == 13
            && !producer.control.yield_flag
    }) {
        return Ok(None);
    }
    let timing = arch
        .branch_resolution()
        .require("branch_resolution")
        .map_err(ProgramError::BranchPipelineCapability)?;
    Ok(Some(timing.uniform_backward_predicate_interval_cycles))
}

fn sidecar_arch_id(arch: &str) -> Result<ArchId, ProgramError> {
    ArchId::parse_deploy(arch).map_err(|source| ProgramError::UnsupportedArchitecture {
        arch: arch.to_owned(),
        source,
    })
}

/// A dense HMMA issue interval is only observable as a scheduler bottleneck
/// when a function exposes enough independent tensor sites to keep the
/// sub-core tensor path saturated.  The released SM86 model's base effective
/// interval is 32 cycles; short functions with fewer than eight static dense
/// sites use the independently measured 16-cycle initiation cadence and let
/// their other (memory/control) frontiers determine completion.  This is a
/// static SASS-shape rule, not a workload-name or NCU-target fit.
const DENSE_HMMA_SATURATION_SITES: usize = 2 * 4;

fn effective_dense_hmma_interval(static_dense_sites: usize, base_interval: u64) -> u64 {
    // Two independent sites per each of the four SM86 sub-cores are needed
    // before a loop can continuously expose the saturated execution arm.
    if static_dense_sites < DENSE_HMMA_SATURATION_SITES {
        base_interval / 2
    } else {
        base_interval
    }
}

/// Join every kernel in `exec` against `sidecar`, keyed by
/// `unique_function_id`. Fail-closed: arch mismatch, missing function, or
/// any pc-set inequality is a typed error.
pub fn build_schedules(
    exec: &ExecInfo,
    sidecar: &ControlSidecar,
) -> Result<BTreeMap<i32, FunctionSchedule>, ProgramError> {
    let sidecar_arch = sidecar_arch_id(&sidecar.arch)?;
    let sidecar_version = sidecar_arch.numeric();
    // Preserve the existing cross-artifact mismatch precedence before asking
    // whether a same-architecture sidecar has a supported control layout.
    for kernel in exec.kernels.values() {
        if kernel.architecture_version != sidecar_version {
            return Err(ProgramError::ArchMismatch {
                kernel_name: kernel.kernel_name.clone(),
                exec_version: kernel.architecture_version,
                sidecar_arch: sidecar.arch.clone(),
            });
        }
    }
    sidecar_arch.control_layout().require("control_layout")?;
    let mut schedules = BTreeMap::new();
    for (&fuid, kernel) in &exec.kernels {
        let function = sidecar
            .functions
            .get(&kernel.function_name)
            .ok_or_else(|| ProgramError::FunctionMissing {
                function_name: kernel.function_name.clone(),
                kernel_name: kernel.kernel_name.clone(),
                binary: sidecar.binary.clone(),
            })?;

        let missing_from_sidecar: Vec<u64> = kernel
            .static_pcs
            .iter()
            .copied()
            .filter(|pc| !function.by_pc.contains_key(pc))
            .collect();
        if let Some(&first) = missing_from_sidecar.first() {
            return Err(ProgramError::PcMissingFromSidecar {
                function_name: kernel.function_name.clone(),
                count: missing_from_sidecar.len(),
                first,
            });
        }
        let missing_from_exec: Vec<u64> = function
            .by_pc
            .keys()
            .copied()
            .filter(|pc| !kernel.static_pcs.contains(pc))
            .collect();
        if let Some(&first) = missing_from_exec.first() {
            return Err(ProgramError::PcMissingFromExecInfo {
                function_name: kernel.function_name.clone(),
                count: missing_from_exec.len(),
                first,
            });
        }

        let by_pc: BTreeMap<_, _> = function
            .by_pc
            .iter()
            .map(|(&pc, inst)| {
                let mem_role = mem_role_of_checked(&inst.opcode_text, sidecar.arch.as_str())?;
                let ldgsts_l1_policy = classify_ldgsts_l1_policy(&inst.opcode_text, sidecar_arch)
                    .map_err(ProgramError::LdgstsCacheClassification)?;
                let op_class = classify(
                    &inst.opcode_text,
                    inst.control,
                    inst.predicate.as_deref(),
                    sidecar_arch,
                )?;
                let mut pipeline_class = classify_pipeline(
                    pc,
                    &inst.opcode_text,
                    inst.control,
                    inst.predicate.as_deref(),
                    sidecar_arch,
                )?;
                // Keep the sidecar predicate available while selecting the
                // compact-loop special case.  The resulting IssueSlot does
                // not duplicate this guard string, so selection happens here
                // rather than in a later pass over the slots.
                if let Some(interval) = compact_backward_predicate_interval(
                    pc,
                    &inst.opcode_text,
                    pipeline_class,
                    function,
                    inst.predicate.as_deref(),
                    sidecar_arch,
                )? {
                    pipeline_class = PipelineClass::Branch {
                        next_issue_interval_cycles: interval,
                    };
                }
                let async_copy_wait_max_pending_groups =
                    parse_async_copy_wait(&inst.opcode_text).unwrap_or(0);
                let fixed_latency_operands =
                    project_fixed_latency_operands(&inst.opcode_text, inst.control, sidecar_arch);
                debug_assert_eq!(
                    op_class == OpClass::AsyncCopyWait,
                    parse_async_copy_wait(&inst.opcode_text).is_some()
                );
                Ok((
                    pc,
                    IssueSlot {
                        control: inst.control,
                        op_class,
                        async_copy_wait_max_pending_groups,
                        pipeline_class,
                        mem_role,
                        ldgsts_l1_policy,
                        fixed_latency_operands,
                        clocklo_read: is_clocklo_opcode_text(&inst.opcode_text),
                        gpu_membar: is_gpu_membar_opcode_text(&inst.opcode_text),
                        opcode_text: inst.opcode_text.clone(),
                    },
                ))
            })
            .collect::<Result<_, ProgramError>>()?;
        let dense_tensor_sites = by_pc
            .values()
            .filter(|slot| {
                matches!(slot.pipeline_class, PipelineClass::Tensor { .. })
                    && slot
                        .opcode_text()
                        .split_ascii_whitespace()
                        .next()
                        .is_some_and(|opcode| opcode.starts_with("HMMA.16816"))
            })
            .count();
        let by_pc = by_pc
            .into_iter()
            .map(|(pc, mut slot)| {
                if let PipelineClass::Tensor {
                    initiation_interval_cycles,
                } = slot.pipeline_class
                    && slot
                        .opcode_text()
                        .split_ascii_whitespace()
                        .next()
                        .is_some_and(|opcode| opcode.starts_with("HMMA.16816"))
                {
                    slot.pipeline_class = PipelineClass::Tensor {
                        initiation_interval_cycles: effective_dense_hmma_interval(
                            dense_tensor_sites,
                            initiation_interval_cycles,
                        ),
                    };
                }
                (pc, slot)
            })
            .collect();
        schedules.insert(
            fuid,
            FunctionSchedule {
                arch: sidecar_arch,
                function_unique_id: fuid,
                kernel_name: kernel.kernel_name.clone(),
                function_name: kernel.function_name.clone(),
                by_pc,
            },
        );
    }
    Ok(schedules)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use sim_trace::exec_info::KernelStaticInfo;

    use super::*;
    use crate::control_bits::{FunctionControl, InstructionControl};

    fn classify_pipeline_test(opcode: &str, arch: ArchId) -> Result<PipelineClass, ProgramError> {
        classify_pipeline(
            0,
            opcode,
            ControlBits::new(1, false, 7, 7, 0, 0).unwrap(),
            None,
            arch,
        )
    }

    fn exec_info(pcs: &[u64]) -> ExecInfo {
        ExecInfo {
            benchmark_name: "bench_run".to_owned(),
            kernels: BTreeMap::from([(
                1,
                KernelStaticInfo {
                    kernel_name: "my_kernel___0".to_owned(),
                    function_name: "my_kernel".to_owned(),
                    architecture_version: 86,
                    static_pcs: pcs.iter().copied().collect::<BTreeSet<_>>(),
                },
            )]),
        }
    }

    fn sidecar(pcs: &[u64]) -> ControlSidecar {
        let by_pc = pcs
            .iter()
            .map(|&pc| {
                (
                    pc,
                    InstructionControl {
                        control: ControlBits::new(1, false, 7, 7, 0, 0)
                            .expect("valid test controls"),
                        opcode_text: if pc == 16 {
                            "BAR.SYNC.DEFER_BLOCKING".to_owned()
                        } else {
                            "IADD3 R0, R1, R2, RZ".to_owned()
                        },
                        predicate: None,
                    },
                )
            })
            .collect();
        ControlSidecar {
            arch: "sm_86".to_owned(),
            binary: "bin/my_bench".to_owned(),
            functions: BTreeMap::from([("my_kernel".to_owned(), FunctionControl { by_pc })]),
        }
    }

    #[test]
    fn join_happy_path_classifies_and_keys_by_fuid() {
        let schedules = build_schedules(&exec_info(&[0, 16, 32]), &sidecar(&[0, 16, 32])).unwrap();
        let s = &schedules[&1];
        assert_eq!(s.arch, ArchId::Sm86);
        assert_eq!(s.function_name, "my_kernel");
        assert_eq!(s.by_pc.len(), 3);
        assert_eq!(s.by_pc[&16].op_class, OpClass::BlockBarrier);
        assert_eq!(
            s.by_pc[&16].pipeline_class,
            PipelineClass::Barrier {
                warp_initiation_interval_cycles: 2,
                generation_rearm_cycles: 0,
            }
        );
        assert_eq!(s.by_pc[&0].op_class, OpClass::Other);
        assert_eq!(
            s.by_pc[&0].pipeline_class,
            PipelineClass::ScalarAlu {
                latency_cycles: 4,
                initiation_interval_cycles: 2,
                in_flight_capacity: 8,
            }
        );
        let FixedLatencyOperandStatus::Projected(operands) = s.by_pc[&0].fixed_latency_operands()
        else {
            panic!("IADD3 operand projection is missing")
        };
        assert_eq!(operands.sm86_regular_source_bank_reads(), [1, 1]);
        assert!(matches!(
            s.by_pc[&16].fixed_latency_operands(),
            FixedLatencyOperandStatus::NotApplicable
        ));
    }

    #[test]
    fn causal_frontier_rejects_block_barrier_schedule_entry() {
        let schedules = build_schedules(&exec_info(&[16]), &sidecar(&[16])).unwrap();
        let error = crate::causal_frontier::CausalFrontierInstruction::from_schedule(
            &schedules[&1],
            16,
            0,
            0,
            1,
            1,
            true,
            false,
        )
        .unwrap_err();
        assert_eq!(
            error,
            crate::causal_frontier::CausalFrontierError::UnsupportedOperationClass {
                pc: 16,
                class: OpClass::BlockBarrier,
            }
        );
    }

    #[test]
    fn causal_frontier_rejects_memory_schedule_entry() {
        let mut controls = sidecar(&[0]);
        controls
            .functions
            .get_mut("my_kernel")
            .expect("test function")
            .by_pc
            .get_mut(&0)
            .expect("test pc")
            .opcode_text = "LDG.E R0, [R2.64]".to_owned();
        let schedules = build_schedules(&exec_info(&[0]), &controls).unwrap();
        let error = crate::causal_frontier::CausalFrontierInstruction::from_schedule(
            &schedules[&1],
            0,
            0,
            0,
            1,
            1,
            true,
            false,
        )
        .unwrap_err();
        assert_eq!(
            error,
            crate::causal_frontier::CausalFrontierError::UnsupportedMemoryRole { pc: 0 }
        );
    }

    #[test]
    fn causal_frontier_rejects_uncarded_scalar_latency() {
        let schedules = build_schedules(&exec_info(&[0]), &sidecar(&[0])).unwrap();
        let error = crate::causal_frontier::CausalFrontierInstruction::from_schedule(
            &schedules[&1],
            0,
            0,
            0,
            1,
            1,
            true,
            false,
        )
        .unwrap_err();
        assert_eq!(
            error,
            crate::causal_frontier::CausalFrontierError::TimingMismatch {
                pc: 0,
                expected: 4,
                got: 1,
            }
        );
    }

    #[test]
    fn operand_projection_gap_is_retained_with_scalar_execution_timing() {
        let mut sc = sidecar(&[0]);
        let instruction = sc
            .functions
            .get_mut("my_kernel")
            .unwrap()
            .by_pc
            .get_mut(&0)
            .unwrap();
        instruction.control = ControlBits::new(1, false, 7, 7, 0, 1).unwrap();
        let schedules = build_schedules(&exec_info(&[0]), &sc).unwrap();
        let slot = &schedules[&1].by_pc[&0];
        assert_eq!(
            slot.pipeline_class,
            PipelineClass::ScalarAlu {
                latency_cycles: 4,
                initiation_interval_cycles: 2,
                in_flight_capacity: 8,
            }
        );
        assert!(matches!(
            slot.fixed_latency_operands(),
            FixedLatencyOperandStatus::Gap(_)
        ));
    }

    #[test]
    fn pc_only_in_exec_info_rejected() {
        let err = build_schedules(&exec_info(&[0, 16, 32]), &sidecar(&[0, 16])).unwrap_err();
        assert!(matches!(
            err,
            ProgramError::PcMissingFromSidecar {
                count: 1,
                first: 32,
                ..
            }
        ));
    }

    #[test]
    fn pc_only_in_sidecar_rejected() {
        let err = build_schedules(&exec_info(&[0, 16]), &sidecar(&[0, 16, 48])).unwrap_err();
        assert!(matches!(
            err,
            ProgramError::PcMissingFromExecInfo {
                count: 1,
                first: 48,
                ..
            }
        ));
    }

    #[test]
    fn function_missing_from_sidecar_rejected() {
        let mut sc = sidecar(&[0]);
        sc.functions.clear();
        let err = build_schedules(&exec_info(&[0]), &sc).unwrap_err();
        assert!(matches!(err, ProgramError::FunctionMissing { .. }));
    }

    #[test]
    fn arch_mismatch_rejected() {
        let mut sc = sidecar(&[0]);
        sc.arch = "sm_120".to_owned();
        let err = build_schedules(&exec_info(&[0]), &sc).unwrap_err();
        assert!(matches!(
            err,
            ProgramError::ArchMismatch {
                exec_version: 86,
                ..
            }
        ));
    }

    /// The join stamps each slot's `mem_role` via `sim_mem::mem_role_of_checked`
    /// (classification tests live with the classifier in sim-mem).
    #[test]
    fn join_stamps_mem_roles_from_sidecar_text() {
        let mut sc = sidecar(&[0]);
        let ic = sc
            .functions
            .get_mut("my_kernel")
            .unwrap()
            .by_pc
            .get_mut(&0)
            .unwrap();
        ic.opcode_text = "LDG.E.U16.CONSTANT R0, [R2.64]".to_owned();
        let schedules = build_schedules(&exec_info(&[0]), &sc).unwrap();
        assert_eq!(
            schedules[&1].by_pc[&0].mem_role,
            MemRole::EligibleLoad(sim_mem::CacheOpToken::LdgECa)
        );
    }

    #[test]
    fn join_stamps_ldgsts_source_cache_policy() {
        for (opcode, expected) in [
            (
                "LDGSTS.E.128 [R1], [R2.64]",
                LdgstsL1Policy::CacheAtAllLevels,
            ),
            (
                "LDGSTS.E.BYPASS.LTC128B.128 [R1], [R2.64]",
                LdgstsL1Policy::BypassL1,
            ),
        ] {
            let mut sc = sidecar(&[0]);
            sc.functions
                .get_mut("my_kernel")
                .unwrap()
                .by_pc
                .get_mut(&0)
                .unwrap()
                .opcode_text = opcode.to_owned();
            let schedules = build_schedules(&exec_info(&[0]), &sc).unwrap();
            let slot = &schedules[&1].by_pc[&0];
            assert_eq!(
                slot.mem_role,
                MemRole::BypassLoad(sim_mem::CacheOpToken::Ldgsts)
            );
            assert_eq!(slot.ldgsts_l1_policy(), Some(expected));
        }
    }

    #[test]
    fn join_stamps_tensor_pipeline_class_from_sidecar_text() {
        let mut sc = sidecar(&[0]);
        sc.functions
            .get_mut("my_kernel")
            .unwrap()
            .by_pc
            .get_mut(&0)
            .unwrap()
            .opcode_text = "HMMA.16816.F32.BF16 R0, R2, R4, R6".to_owned();
        let schedules = build_schedules(&exec_info(&[0]), &sc).unwrap();
        assert_eq!(
            schedules[&1].by_pc[&0].pipeline_class,
            PipelineClass::Tensor {
                initiation_interval_cycles: 16,
            }
        );
    }

    #[test]
    fn exact_forward_and_uniform_backward_branches_use_distinct_intervals() {
        let branch_control = ControlBits::new(5, true, 7, 7, 0, 0).unwrap();
        assert_eq!(
            classify_pipeline(0, "BRA 0x20", branch_control, Some("@P0"), ArchId::Sm86,).unwrap(),
            PipelineClass::Branch {
                next_issue_interval_cycles: 10,
            }
        );
        assert_eq!(
            classify_pipeline(0x20, "BRA 0x0", branch_control, Some("@P0"), ArchId::Sm86,).unwrap(),
            PipelineClass::Branch {
                next_issue_interval_cycles: 3,
            }
        );
        assert_eq!(
            classify_pipeline(
                0x20,
                "BRA P1, 0x0",
                branch_control,
                Some("@P0"),
                ArchId::Sm86,
            )
            .unwrap(),
            PipelineClass::Branch {
                next_issue_interval_cycles: 3,
            }
        );
        assert_eq!(
            classify_pipeline(0, "BRA 0x20", branch_control, None, ArchId::Sm86).unwrap(),
            PipelineClass::DispatchOnly
        );
        assert_eq!(
            classify_pipeline(0x20, "BRA 0x20", branch_control, Some("@P0"), ArchId::Sm86,)
                .unwrap(),
            PipelineClass::DispatchOnly
        );
        assert_eq!(parse_direct_branch_target("BRA P1, 0x40"), Some(0x40));
        assert_eq!(parse_direct_branch_target("BRA R2"), None);
        assert!(matches!(
            classify_pipeline(
                0,
                "BRA 0x20",
                branch_control,
                Some("@P0"),
                ArchId::Sm120,
            ),
            Err(ProgramError::BranchPipelineCapability(
                CapabilityError::ResearchGap {
                    arch: ArchId::Sm120,
                    capability: "branch_resolution",
                    reason,
                }
            )) if !reason.is_empty()
        ));

        assert_eq!(
            classify_pipeline_test("IMAD R0, R0, R1, R2", ArchId::Sm86).unwrap(),
            PipelineClass::ScalarAlu {
                latency_cycles: 4,
                initiation_interval_cycles: 2,
                in_flight_capacity: 8,
            }
        );
    }

    #[test]
    fn compact_backward_predicate_branch_is_selected_only_from_sass_shape() {
        let mut compact = sidecar(&[0, 16, 32]);
        let function = compact.functions.get_mut("my_kernel").unwrap();
        function.by_pc.get_mut(&16).unwrap().opcode_text =
            "ISETP.NE.AND P0, PT, R0, RZ, PT".to_owned();
        function.by_pc.get_mut(&16).unwrap().control =
            ControlBits::new(13, false, 7, 7, 0, 0).unwrap();
        let branch = function.by_pc.get_mut(&32).unwrap();
        branch.opcode_text = "BRA 0x0".to_owned();
        branch.predicate = Some("@P0".to_owned());
        branch.control = ControlBits::new(5, true, 7, 7, 0, 0).unwrap();
        let schedules = build_schedules(&exec_info(&[0, 16, 32]), &compact).unwrap();
        assert_eq!(
            schedules[&1].by_pc[&32].pipeline_class,
            PipelineClass::Branch {
                next_issue_interval_cycles: 9,
            }
        );

        // A longer back edge with the same branch control remains on the
        // ordinary three-cycle target-resolution path: distance, not a
        // workload identity, selects the compact handoff capability.
        let mut long = sidecar(&[0, 16, 80]);
        let function = long.functions.get_mut("my_kernel").unwrap();
        function.by_pc.get_mut(&16).unwrap().opcode_text =
            "ISETP.NE.AND P0, PT, R0, RZ, PT".to_owned();
        function.by_pc.get_mut(&16).unwrap().control =
            ControlBits::new(13, false, 7, 7, 0, 0).unwrap();
        let branch = function.by_pc.get_mut(&80).unwrap();
        branch.opcode_text = "BRA 0x0".to_owned();
        branch.predicate = Some("@P0".to_owned());
        branch.control = ControlBits::new(5, true, 7, 7, 0, 0).unwrap();
        let schedules = build_schedules(&exec_info(&[0, 16, 80]), &long).unwrap();
        assert_eq!(
            schedules[&1].by_pc[&80].pipeline_class,
            PipelineClass::Branch {
                next_issue_interval_cycles: 3,
            }
        );

        // The preceding predicate producer must be the value consumed by the
        // branch.  Matching only `ISETP` + stall-13 + adjacency would promote
        // this P1 producer even though the branch reads P0.
        let mut different_predicate = sidecar(&[0, 16, 32]);
        let function = different_predicate.functions.get_mut("my_kernel").unwrap();
        function.by_pc.get_mut(&16).unwrap().opcode_text =
            "ISETP.NE.AND P1, PT, R0, RZ, PT".to_owned();
        function.by_pc.get_mut(&16).unwrap().control =
            ControlBits::new(13, false, 7, 7, 0, 0).unwrap();
        let branch = function.by_pc.get_mut(&32).unwrap();
        branch.opcode_text = "BRA 0x0".to_owned();
        branch.predicate = Some("@P0".to_owned());
        branch.control = ControlBits::new(5, true, 7, 7, 0, 0).unwrap();
        let schedules = build_schedules(&exec_info(&[0, 16, 32]), &different_predicate).unwrap();
        assert_eq!(
            schedules[&1].by_pc[&32].pipeline_class,
            PipelineClass::Branch {
                next_issue_interval_cycles: 3,
            }
        );

        // P7 is the textual alias for PT (the hardwired true predicate), not
        // a writable predicate register. It must not satisfy the compact
        // producer/consumer identity check.
        let mut true_predicate = sidecar(&[0, 16, 32]);
        let function = true_predicate.functions.get_mut("my_kernel").unwrap();
        function.by_pc.get_mut(&16).unwrap().opcode_text =
            "ISETP.NE.AND P7, PT, R0, RZ, PT".to_owned();
        function.by_pc.get_mut(&16).unwrap().control =
            ControlBits::new(13, false, 7, 7, 0, 0).unwrap();
        let branch = function.by_pc.get_mut(&32).unwrap();
        branch.opcode_text = "BRA 0x0".to_owned();
        branch.predicate = Some("@P7".to_owned());
        branch.control = ControlBits::new(5, true, 7, 7, 0, 0).unwrap();
        let schedules = build_schedules(&exec_info(&[0, 16, 32]), &true_predicate).unwrap();
        assert_eq!(
            schedules[&1].by_pc[&32].pipeline_class,
            PipelineClass::Branch {
                next_issue_interval_cycles: 3,
            }
        );
    }

    #[test]
    fn schedule_join_rejects_unmodeled_tensor_form() {
        let mut sc = sidecar(&[0]);
        sc.functions
            .get_mut("my_kernel")
            .unwrap()
            .by_pc
            .get_mut(&0)
            .unwrap()
            .opcode_text = "HMMA.884.F32 R0, R2, R4, R6".to_owned();
        assert!(matches!(
            build_schedules(&exec_info(&[0]), &sc),
            Err(ProgramError::TensorPipelineUnsupported {
                arch: ArchId::Sm86,
                opcode,
            }) if opcode == "HMMA.884.F32"
        ));
    }

    #[test]
    fn sm86_schedule_join_rejects_hopper_redg() {
        let mut sc = sidecar(&[0]);
        sc.functions
            .get_mut("my_kernel")
            .unwrap()
            .by_pc
            .get_mut(&0)
            .unwrap()
            .opcode_text = "REDG.E.ADD.STRONG.GPU [R2.64], R0".to_owned();
        assert!(matches!(
            build_schedules(&exec_info(&[0]), &sc),
            Err(ProgramError::MemRoleClassification(
                sim_mem::MemRoleError::StaticMemoryFamilyUnsupported {
                    family: "REDG",
                    arch: ArchId::Sm86,
                }
            ))
        ));
    }

    #[test]
    fn block_barrier_classification_uses_opcode_head_only() {
        let control = ControlBits::new(1, false, 7, 7, 0, 0).unwrap();
        assert_eq!(
            classify("BAR.SYNC.DEFER_BLOCKING", control, None, ArchId::Sm86).unwrap(),
            OpClass::BlockBarrier
        );
        assert_eq!(
            classify(
                "BAR.RED.AND.DEFER_BLOCKING 0x0",
                control,
                None,
                ArchId::Sm86
            )
            .unwrap(),
            OpClass::BlockBarrier
        );
        // BSSY/BSYNC/WARPSYNC are NOT block-wide barriers.
        for opcode in [
            "BSYNC B0",
            "BSSY B0, 0x1234",
            "WARPSYNC 0xffffffff",
            "LDG.E R0, [R2.64]",
            "",
        ] {
            assert_eq!(
                classify(opcode, control, None, ArchId::Sm86).unwrap(),
                OpClass::Other,
                "opcode {opcode:?}"
            );
        }
    }

    #[test]
    fn sm86_async_copy_controls_classify_exact_commit_and_wait_forms() {
        let commit = ControlBits::new(1, true, 0, 7, 0, 0).unwrap();
        let wait = ControlBits::new(4, false, 7, 7, 0, 0).unwrap();
        assert_eq!(
            classify("LDGDEPBAR", commit, None, ArchId::Sm86).unwrap(),
            OpClass::AsyncCopyCommit
        );
        let composed_commit = ControlBits::new(2, true, 0, 7, 0b001110, 0).unwrap();
        assert_eq!(
            classify("LDGDEPBAR", composed_commit, None, ArchId::Sm86).unwrap(),
            OpClass::AsyncCopyCommit
        );
        for (opcode, max_pending_groups) in [
            ("DEPBAR.LE SB0, 0x0", 0),
            ("DEPBAR.LE SB0, 0X1", 1),
            ("DEPBAR.LE SB0, 17", 17),
            ("  DEPBAR.LE\tSB0,0x00000002  ", 2),
        ] {
            assert_eq!(
                classify(opcode, wait, None, ArchId::Sm86).unwrap(),
                OpClass::AsyncCopyWait,
                "opcode {opcode:?}"
            );
            assert_eq!(parse_async_copy_wait(opcode), Some(max_pending_groups));
        }
        let composed = ControlBits::new(4, false, 7, 7, 0b001110, 0).unwrap();
        assert_eq!(
            classify("DEPBAR.LE SB0, 0x0", composed, None, ArchId::Sm86).unwrap(),
            OpClass::AsyncCopyWait
        );
        assert_eq!(parse_async_copy_wait("DEPBAR.LE SB0, 0x0"), Some(0));
    }

    #[test]
    fn sm86_non_async_dependency_barriers_wait_on_ordinary_scoreboards() {
        // These forms are emitted after retained S2UR clock reads. They use
        // the ordinary SB1..SB5 producer state and must not enter the SB0
        // asynchronous-copy group protocol.
        let control = ControlBits::new(7, false, 7, 7, 0, 0).unwrap();
        for wire in 1..=5 {
            let opcode = format!("DEPBAR.LE SB{wire}, 0x0");
            assert_eq!(
                classify(&opcode, control, None, ArchId::Sm86).unwrap(),
                OpClass::Other,
                "opcode {opcode:?}"
            );
            assert_eq!(parse_async_copy_wait(&opcode), None);
            assert_eq!(
                parse_dependency_barrier(&opcode),
                Some(DependencyBarrierOperand::OrdinaryWait(
                    WaitMask::new(1 << wire).unwrap()
                ))
            );
        }
        let actual = "DEPBAR.LE SB1, 0x0, {3,2}";
        assert_eq!(
            classify(actual, control, None, ArchId::Sm86).unwrap(),
            OpClass::Other
        );
        assert_eq!(
            parse_dependency_barrier(actual),
            Some(DependencyBarrierOperand::OrdinaryWait(
                WaitMask::new(0b001110).unwrap()
            ))
        );
    }

    #[test]
    fn async_copy_control_classifier_rejects_unknown_forms() {
        let commit = ControlBits::new(1, true, 0, 7, 0, 0).unwrap();
        let wait = ControlBits::new(4, false, 7, 7, 0, 0).unwrap();
        for (opcode, control) in [
            ("LDGDEPBAR.EXTRA", commit),
            ("LDGDEPBAR R0", commit),
            ("DEPBAR", wait),
            ("DEPBAR.GE SB0, 0x0", wait),
            ("DEPBAR.LE SB1, 0x1", wait),
            ("DEPBAR.LE SB5, 17", wait),
            ("DEPBAR.LE SB6, 0x0", wait),
            ("DEPBAR.LE SB01, 0x0", wait),
            ("DEPBAR.LE SB0, -1", wait),
            ("DEPBAR.LE SB0, 0x1_0000_0000", wait),
            ("DEPBAR.LE SB0, 0, 1", wait),
            ("DEPBAR.LE SB0, 0x0, {3,2}", wait),
            ("DEPBAR.LE SB1, 0x1, {3,2}", wait),
            ("DEPBAR.LE SB1, 0x0, {}", wait),
            ("DEPBAR.LE SB1, 0x0, {3,3}", wait),
            ("DEPBAR.LE SB1, 0x0, {3,1}", wait),
            ("DEPBAR.LE SB1, 0x0, {6}", wait),
            ("DEPBAR.LE SB1, 0x0, {0x3}", wait),
            ("DEPBAR.LE SB1, 0x0, 3", wait),
            ("DEPBAR.LE SB1, 0x0, {3,2}, {4}", wait),
        ] {
            assert!(matches!(
                classify(opcode, control, None, ArchId::Sm86),
                Err(ProgramError::AsyncCopyControlUnsupported {
                    arch: ArchId::Sm86,
                    opcode: rejected,
                }) if rejected == opcode
            ));
        }
    }

    #[test]
    fn async_copy_control_classifier_rejects_predication_and_layout_drift() {
        let commit = ControlBits::new(1, true, 0, 7, 0, 0).unwrap();
        let wait = ControlBits::new(4, false, 7, 7, 0, 0).unwrap();
        assert!(matches!(
            classify("LDGDEPBAR", commit, Some("@P0"), ArchId::Sm86),
            Err(ProgramError::AsyncCopyControlPredicated {
                arch: ArchId::Sm86,
                predicate,
                ..
            }) if predicate == "@P0"
        ));
        assert!(matches!(
            classify("DEPBAR.LE SB0, 0x0", wait, Some("@!P1"), ArchId::Sm86),
            Err(ProgramError::AsyncCopyControlPredicated {
                arch: ArchId::Sm86,
                predicate,
                ..
            }) if predicate == "@!P1"
        ));

        for (opcode, control) in [
            ("LDGDEPBAR", ControlBits::new(1, true, 7, 7, 0, 0).unwrap()),
            ("LDGDEPBAR", ControlBits::new(1, true, 0, 1, 0, 0).unwrap()),
            ("LDGDEPBAR", ControlBits::new(1, true, 0, 7, 1, 0).unwrap()),
            (
                "DEPBAR.LE SB0, 0x0",
                ControlBits::new(4, false, 0, 7, 0, 0).unwrap(),
            ),
            (
                "DEPBAR.LE SB0, 0x0",
                ControlBits::new(4, false, 7, 1, 0, 0).unwrap(),
            ),
            (
                "DEPBAR.LE SB2, 0x0",
                ControlBits::new(7, false, 7, 7, 0b000100, 0).unwrap(),
            ),
        ] {
            assert!(matches!(
                classify(opcode, control, None, ArchId::Sm86),
                Err(ProgramError::AsyncCopyControlLayout {
                    arch: ArchId::Sm86,
                    opcode: rejected,
                    ..
                }) if rejected == opcode
            ));
        }
    }

    #[test]
    fn async_copy_classifier_does_not_inherit_ampere_protocol_on_sm120() {
        let commit = ControlBits::new(1, true, 0, 7, 0, 0).unwrap();
        assert!(matches!(
            classify("LDGDEPBAR", commit, None, ArchId::Sm120),
            Err(ProgramError::AsyncCopyGroupCapability(
                CapabilityError::ResearchGap {
                    arch: ArchId::Sm120,
                    capability: "async_copy_group_semantics",
                    reason,
                }
            )) if !reason.is_empty()
        ));
    }

    #[test]
    fn sm86_tensor_pipeline_classifier_admits_complete_sidecar_inventory() {
        for (opcode, expected_interval) in [
            ("HMMA.16816.F32", 32),
            ("HMMA.16816.F32.BF16", 32),
            ("HMMA.SP.16832.F32.BF16", 16),
            ("IMMA.16816.S8.S8", 8),
        ] {
            assert_eq!(
                classify_pipeline_test(&format!("{opcode} R0, R2, R4, R6"), ArchId::Sm86).unwrap(),
                PipelineClass::Tensor {
                    initiation_interval_cycles: expected_interval,
                },
                "opcode {opcode}"
            );
        }
    }

    #[test]
    fn dense_hmma_static_saturation_boundary_preserves_both_cadences() {
        assert_eq!(effective_dense_hmma_interval(0, 32), 16);
        assert_eq!(
            effective_dense_hmma_interval(DENSE_HMMA_SATURATION_SITES - 1, 32),
            16
        );
        assert_eq!(
            effective_dense_hmma_interval(DENSE_HMMA_SATURATION_SITES, 32),
            32
        );
        assert_eq!(effective_dense_hmma_interval(16, 32), 32);
    }

    #[test]
    fn schedule_join_applies_dense_hmma_saturation_from_static_shape() {
        for (site_count, expected) in [
            (DENSE_HMMA_SATURATION_SITES - 1, 16),
            (DENSE_HMMA_SATURATION_SITES, 32),
        ] {
            let pcs: Vec<u64> = (0..site_count).map(|index| index as u64 * 16).collect();
            let mut sidecar = sidecar(&pcs);
            for instruction in sidecar
                .functions
                .get_mut("my_kernel")
                .unwrap()
                .by_pc
                .values_mut()
            {
                instruction.opcode_text = "HMMA.16816.F32.BF16 R0, R2, R4, R6".to_owned();
            }
            let schedules = build_schedules(&exec_info(&pcs), &sidecar).unwrap();
            assert!(schedules[&1].by_pc.values().all(|slot| {
                slot.pipeline_class
                    == PipelineClass::Tensor {
                        initiation_interval_cycles: expected,
                    }
            }));
        }
    }

    #[test]
    fn scalar_opcode_candidate_is_explicit_and_other_forms_stay_dispatch_only() {
        assert_eq!(
            classify_pipeline_test("IADD3 R0, R1, R2, RZ", ArchId::Sm86).unwrap(),
            PipelineClass::ScalarAlu {
                latency_cycles: 4,
                initiation_interval_cycles: 2,
                in_flight_capacity: 8,
            }
        );
        for opcode in ["XHMMA.16816.F32 R0", ""] {
            assert_eq!(
                classify_pipeline_test(opcode, ArchId::Sm86).unwrap(),
                PipelineClass::DispatchOnly,
                "opcode {opcode:?}"
            );
        }
    }

    #[test]
    fn unknown_tensor_forms_are_rejected_without_generic_fallback() {
        for opcode in [
            "HMMA.884.F32 R0, R2, R4, R6",
            "HMMA.16816.F32.BF16.EXTRA R0",
            "IMMA.16816.U8.U8 R0",
            "BMMA.168256.XOR.POPC R0",
            "DMMA.884.F64 R0",
        ] {
            assert!(matches!(
                classify_pipeline_test(opcode, ArchId::Sm86),
                Err(ProgramError::TensorPipelineUnsupported {
                    arch: ArchId::Sm86,
                    opcode: rejected,
                }) if rejected == opcode.split_ascii_whitespace().next().unwrap()
            ));
        }
    }

    #[test]
    fn tensor_classifier_does_not_inherit_sm86_intervals_on_sm120() {
        assert!(matches!(
            classify_pipeline_test("HMMA.16816.F32 R0", ArchId::Sm120),
            Err(ProgramError::TensorPipelineCapability(
                CapabilityError::ResearchGap {
                    arch: ArchId::Sm120,
                    capability: "tensor_issue_intervals",
                    reason,
                }
            )) if !reason.is_empty()
        ));
    }

    #[test]
    fn sidecar_arch_parse() {
        assert_eq!(sidecar_arch_id("sm_86").unwrap().numeric(), 86);
        assert_eq!(sidecar_arch_id("sm_120").unwrap().numeric(), 120);
        assert!(matches!(
            sidecar_arch_id("86"),
            Err(ProgramError::UnsupportedArchitecture { .. })
        ));
        assert!(matches!(
            sidecar_arch_id("sm_999"),
            Err(ProgramError::UnsupportedArchitecture { .. })
        ));
    }

    #[test]
    fn forged_sm120_sidecar_cannot_bypass_control_layout_capability() {
        // ControlSidecar fields are public for diagnostics, so schedule
        // construction must independently reapply the capability gate even
        // when a caller bypasses ControlSidecar::load_for_arch.
        let mut sc = sidecar(&[0]);
        sc.arch = "sm_120".to_owned();
        let err = build_schedules(
            &ExecInfo {
                benchmark_name: "bench_run".to_owned(),
                kernels: BTreeMap::from([(
                    1,
                    KernelStaticInfo {
                        kernel_name: "my_kernel___0".to_owned(),
                        function_name: "my_kernel".to_owned(),
                        architecture_version: 120,
                        static_pcs: [0].into_iter().collect(),
                    },
                )]),
            },
            &sc,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ProgramError::ControlLayoutCapability(CapabilityError::Unsupported {
                arch: ArchId::Sm120,
                capability: "control_layout",
                reason,
            }) if !reason.is_empty()
        ));
    }

    #[test]
    fn unsupported_control_layout_is_rejected_even_without_kernels() {
        let mut sc = sidecar(&[]);
        sc.arch = "sm_120".to_owned();
        let err = build_schedules(
            &ExecInfo {
                benchmark_name: "empty".to_owned(),
                kernels: BTreeMap::new(),
            },
            &sc,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ProgramError::ControlLayoutCapability(CapabilityError::Unsupported {
                arch: ArchId::Sm120,
                capability: "control_layout",
                ..
            })
        ));
    }
}
