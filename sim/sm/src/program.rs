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
    CacheOpToken, LdgstsL1Policy, MemRole, classify_ldgsts_l1_policy, mem_role_of_checked,
    opcode_head,
};
use sim_trace::ExecInfo;

use crate::control_bits::{ControlBits, ControlSidecar, DepBarrier, FunctionControl, WaitMask};
use crate::issue::Sm120AlphaConfig;
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
    #[error("SM120 alpha lane does not model asynchronous/TMA opcode {opcode:?}")]
    AlphaAsyncUnsupported { opcode: String },
    #[error("SM120 alpha schedule requires an sm_120 sidecar, got {got:?}")]
    AlphaArchitectureMismatch { got: ArchId },
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
    /// Uniform/SFU issue pipe.  It shares the per-SMSP availability timeline
    /// with tensor work only for arbitration; its cadence is carded
    /// independently by `ArchId::uniform_sfu_cadence`.
    Sfu {
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
            Self::Sfu {
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

    pub(crate) fn is_cctl_invalidate_all(&self) -> bool {
        let text = self.opcode_text.trim_ascii();
        text.split_ascii_whitespace().next() == Some("CCTL.IVALL")
    }

    pub(crate) fn visibility_fence_semantics(
        &self,
    ) -> Option<(sim_mem::VisibilityScope, sim_mem::VisibilityOrder)> {
        let token = self.opcode_text.split_ascii_whitespace().next()?;
        // A scoped load/store is not itself a fence. Admit only the MEMBAR
        // forms covered by this observation seam, not every *.GPU/*.SYS opcode.
        let scope = match token {
            "MEMBAR.CTA" | "MEMBAR.SC.CTA" => sim_mem::VisibilityScope::Block,
            "MEMBAR.GL" | "MEMBAR.GPU" | "MEMBAR.SC.GPU" => sim_mem::VisibilityScope::Device,
            "MEMBAR.SYS" | "MEMBAR.SC.SYS" => sim_mem::VisibilityScope::System,
            _ => return None,
        };
        Some((scope, sim_mem::VisibilityOrder::SequentiallyConsistent))
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

/// Conservative SM120 alpha static routing.  These routes intentionally do
/// not ask `sim_mem` for an architecture capability: the alpha lane has no
/// cache geometry/semantic claim and Flat replay only needs a stable global,
/// local, shared, or atomic class for trace accounting.
fn alpha_mem_role(opcode_text: &str) -> MemRole {
    match opcode_head(opcode_text) {
        "LDG" => MemRole::EligibleLoad(CacheOpToken::LdgECg),
        // TMA/cp.async completion and cache routing are a separate SM120
        // feature lane.  Keep the static spelling opaque here so a dynamic
        // async record is rejected by replay rather than silently priced by
        // the old Ampere transport helper.
        "LDGSTS" => MemRole::None,
        "LDL" => MemRole::EligibleLoad(CacheOpToken::Ldl),
        "STL" => MemRole::EligibleLoad(CacheOpToken::Stl),
        "STG" => MemRole::Store,
        "ATOM" | "ATOMG" => MemRole::BypassLoad(CacheOpToken::Atom),
        "RED" | "REDG" => MemRole::Reduction,
        _ => MemRole::None,
    }
}

fn alpha_ldgsts_l1_policy(opcode_text: &str) -> Option<LdgstsL1Policy> {
    let _ = opcode_text;
    None
}

fn alpha_op_class(opcode_text: &str) -> OpClass {
    match opcode_head(opcode_text) {
        "BAR" => OpClass::BlockBarrier,
        // Full-replay-only alpha async protocol. Keep the existing explicit
        // group/DEPBAR parser; the fast frontier is frozen and does not use
        // this schedule path.
        "LDGDEPBAR" => OpClass::AsyncCopyCommit,
        "UTMALDG" | "UTMACMDFLUSH" => OpClass::AsyncCopyCommit,
        "SYNCS" if opcode_text.contains("PHASECHK") => OpClass::AsyncCopyWait,
        _ => match parse_dependency_barrier(opcode_text) {
            Some(DependencyBarrierOperand::AsyncCopyWait(_)) => OpClass::AsyncCopyWait,
            Some(DependencyBarrierOperand::OrdinaryWait(_)) | None => OpClass::Other,
        },
    }
}

/// M8/M9-derived scalar/barrier cadence.  Uncharacterized tensor, SFU,
/// branch, and async-control forms remain dispatch-only rather than borrowing
/// an SM86 interval.
fn classify_pipeline_sm120_alpha(
    _pc: u64,
    opcode_text: &str,
    _control: ControlBits,
    _predicate: Option<&str>,
    config: Sm120AlphaConfig,
) -> Result<PipelineClass, ProgramError> {
    let head = opcode_head(opcode_text);
    if head == "BAR" {
        return Ok(PipelineClass::Barrier {
            warp_initiation_interval_cycles: config.barrier_warp_initiation_interval_cycles,
            generation_rearm_cycles: config.barrier_generation_rearm_cycles,
        });
    }
    let (latency_cycles, initiation_interval_cycles) = if matches!(head, "FFMA" | "FADD" | "FMUL" | "HFMA2" | "HADD2" | "HMUL2") {
        (config.fma_latency_cycles, config.fma_initiation_interval_cycles)
    } else if head == "IMAD" {
        (config.mad_latency_cycles, config.mad_initiation_interval_cycles)
    } else if matches!(head, "IADD" | "IADD3" | "UIADD3" | "UIADD") {
        (config.add_latency_cycles, config.add_initiation_interval_cycles)
    } else if matches!(head, "LOP3" | "LEA" | "SHF") {
        (config.scalar_latency_cycles, config.scalar_initiation_interval_cycles)
    } else {
        return Ok(PipelineClass::DispatchOnly);
    };
    if latency_cycles > 0 {
        return Ok(PipelineClass::ScalarAlu {
            latency_cycles,
            initiation_interval_cycles,
            in_flight_capacity: config.scalar_in_flight_capacity,
        });
    }
    Ok(PipelineClass::DispatchOnly)
}

/// The retained high-word control decoder is not a proven SM120 stall
/// encoding. P9's dependent arithmetic probes show that applying its scalar
/// stall/yield field on top of the calibrated family pipeline double-counts
/// readiness. Keep the decoded field for non-scalar/control classes until a
/// scope-aligned SM120 control experiment establishes its meaning.
pub(crate) fn sm120_alpha_scalar_control_stall_suppressed(opcode_text: &str) -> bool {
    matches!(
        opcode_head(opcode_text),
        "IADD"
            | "IADD3"
            | "IMAD"
            | "FADD"
            | "FFMA"
            | "FMUL"
            | "HFMA2"
            | "HADD2"
            | "HMUL2"
            | "UIADD3"
            | "UIADD"
    )
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
    // SFU forms use the architecture-local uniform/SFU cadence below; other
    // uniform/control-only forms remain dispatch-only until characterized.
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
    if matches!(head, "MUFU" | "RCP" | "RSQ" | "EX2" | "LG2" | "SIN" | "COS") {
        let cadence = arch
            .uniform_sfu_cadence()
            .require("uniform_sfu_cadence")
            .map_err(ProgramError::ScalarExecutionPipelineCapability)?;
        return Ok(PipelineClass::Sfu {
            initiation_interval_cycles: cadence.initiation_interval_cycles,
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
        "HMMA.1688.F32.TF32" => intervals.hmma_1688_tf32,
        "HMMA.SP.16832.F32.BF16" => intervals.hmma_sparse_16832_f16_bf16,
        "IMMA.16816.S8.S8" => intervals.imma_16816_s8,
        "IMMA.16832.S8.S8" => intervals.imma_16832_s8,
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
    build_schedules_with_pipeline_arch(exec, sidecar, sidecar_arch)
}

/// Build a trace schedule while keeping the deployment architecture in the
/// returned [`FunctionSchedule`] but selecting an explicit, opt-in pipeline
/// hypothesis for the timing classifier.  This seam is retained for
/// architecture-specific experiments; the SM120 alpha lane uses the separate
/// [`build_schedules_sm120_alpha`] entry point so it cannot accidentally
/// select an SM86 pipeline.
///
/// Memory-family classification always uses `sidecar.arch`; only pipeline
/// cadence/latency classification uses `pipeline_arch`.  Keeping these two
/// identities separate prevents the alpha issue hypothesis from changing
/// address-space or SM120-specific opcode routing.
pub fn build_schedules_with_pipeline_arch(
    exec: &ExecInfo,
    sidecar: &ControlSidecar,
    pipeline_arch: ArchId,
) -> Result<BTreeMap<i32, FunctionSchedule>, ProgramError> {
    build_schedules_with_pipeline_arch_and_alpha(exec, sidecar, pipeline_arch, None)
}

/// Build an explicitly opt-in SM120 alpha schedule.  The returned schedule
/// retains `sm_120`; the scalar/barrier cadence comes only from the supplied
/// M8/M9 envelope and never from an SM86 capability table.
pub fn build_schedules_sm120_alpha(
    exec: &ExecInfo,
    sidecar: &ControlSidecar,
    config: Sm120AlphaConfig,
) -> Result<BTreeMap<i32, FunctionSchedule>, ProgramError> {
    build_schedules_with_pipeline_arch_and_alpha(exec, sidecar, ArchId::Sm120, Some(config))
}

fn build_schedules_with_pipeline_arch_and_alpha(
    exec: &ExecInfo,
    sidecar: &ControlSidecar,
    pipeline_arch: ArchId,
    alpha: Option<Sm120AlphaConfig>,
) -> Result<BTreeMap<i32, FunctionSchedule>, ProgramError> {
    let sidecar_arch = sidecar_arch_id(&sidecar.arch)?;
    if alpha.is_some() && sidecar_arch != ArchId::Sm120 {
        return Err(ProgramError::AlphaArchitectureMismatch { got: sidecar_arch });
    }
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
    // The production SM120 control-layout capability is still a typed gap.
    // The only exception is the explicit alpha builder, which uses the
    // retained high-word decoder as a hypothesis while preserving `sm_120` in
    // the returned schedule.
    if alpha.is_none() {
        sidecar_arch.control_layout().require("control_layout")?;
    }
    let decode_arch = sidecar_arch;
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
                let mem_role = if alpha.is_some() {
                    alpha_mem_role(&inst.opcode_text)
                } else {
                    mem_role_of_checked(&inst.opcode_text, decode_arch.token())?
                };
                let ldgsts_l1_policy = if alpha.is_some() {
                    alpha_ldgsts_l1_policy(&inst.opcode_text)
                } else {
                    classify_ldgsts_l1_policy(&inst.opcode_text, decode_arch)
                        .map_err(ProgramError::LdgstsCacheClassification)?
                };
                let op_class = if alpha.is_some() {
                    alpha_op_class(&inst.opcode_text)
                } else {
                    classify(
                        &inst.opcode_text,
                        inst.control,
                        inst.predicate.as_deref(),
                        decode_arch,
                    )?
                };
                let mut pipeline_class = if let Some(alpha) = alpha {
                    classify_pipeline_sm120_alpha(
                        pc,
                        &inst.opcode_text,
                        inst.control,
                        inst.predicate.as_deref(),
                        alpha,
                    )?
                } else {
                    classify_pipeline(
                        pc,
                        &inst.opcode_text,
                        inst.control,
                        inst.predicate.as_deref(),
                        pipeline_arch,
                    )?
                };
                // Keep the sidecar predicate available while selecting the
                // compact-loop special case.  The resulting IssueSlot does
                // not duplicate this guard string, so selection happens here
                // rather than in a later pass over the slots.
                if alpha.is_none()
                    && let Some(interval) = compact_backward_predicate_interval(
                    pc,
                    &inst.opcode_text,
                    pipeline_class,
                    function,
                    inst.predicate.as_deref(),
                    pipeline_arch,
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

