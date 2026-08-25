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

use crate::control_bits::{ControlBits, ControlSidecar};
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
            if control.write_barrier.map(|barrier| barrier.wire()) != Some(0)
                || control.read_barrier.is_some()
                || control.wait_mask.bits() != 0
            {
                return Err(ProgramError::AsyncCopyControlLayout {
                    arch,
                    opcode: opcode_text.to_owned(),
                    control,
                    expected: "write barrier SB0, no read barrier, and an empty ordinary wait mask",
                });
            }
            Ok(OpClass::AsyncCopyCommit)
        }
        "DEPBAR" => {
            if parse_async_copy_wait(opcode_text).is_none() {
                return Err(ProgramError::AsyncCopyControlUnsupported {
                    arch,
                    opcode: opcode_text.to_owned(),
                });
            };
            if control.write_barrier.is_some()
                || control.read_barrier.is_some()
                || control.wait_mask.bits() != 0
            {
                return Err(ProgramError::AsyncCopyControlLayout {
                    arch,
                    opcode: opcode_text.to_owned(),
                    control,
                    expected: "no ordinary write barrier, read barrier, or wait mask",
                });
            }
            Ok(OpClass::AsyncCopyWait)
        }
        _ => unreachable!("dependency-barrier heads were filtered above"),
    }
}

fn parse_async_copy_wait(opcode_text: &str) -> Option<u32> {
    let text = opcode_text.trim();
    let operands = text.strip_prefix("DEPBAR.LE")?;
    if operands.is_empty() || !operands.starts_with(char::is_whitespace) {
        return None;
    }
    let mut operands = operands.split(',');
    if operands.next()?.trim() != "SB0" {
        return None;
    }
    let immediate = operands.next()?.trim();
    if immediate.is_empty() || operands.next().is_some() {
        return None;
    }
    parse_u32_literal(immediate)
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
    let branch_target = opcode_text
        .strip_prefix("BRA ")
        .and_then(parse_u32_literal)
        .map(u64::from);
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

fn sidecar_arch_id(arch: &str) -> Result<ArchId, ProgramError> {
    ArchId::parse_deploy(arch).map_err(|source| ProgramError::UnsupportedArchitecture {
        arch: arch.to_owned(),
        source,
    })
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
                let pipeline_class = classify_pipeline(
                    pc,
                    &inst.opcode_text,
                    inst.control,
                    inst.predicate.as_deref(),
                    sidecar_arch,
                )?;
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
        assert_eq!(s.by_pc[&0].pipeline_class, PipelineClass::DispatchOnly);
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
    fn operand_projection_gap_is_retained_without_changing_dispatch_timing() {
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
        assert_eq!(slot.pipeline_class, PipelineClass::DispatchOnly);
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
            classify_pipeline(0, "BRA 0x20", branch_control, None, ArchId::Sm86).unwrap(),
            PipelineClass::DispatchOnly
        );
        assert_eq!(
            classify_pipeline(0x20, "BRA 0x20", branch_control, Some("@P0"), ArchId::Sm86,)
                .unwrap(),
            PipelineClass::DispatchOnly
        );
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
            PipelineClass::DispatchOnly
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
            ("DEPBAR.LE SB1, 0x0", wait),
            ("DEPBAR.LE SB0, -1", wait),
            ("DEPBAR.LE SB0, 0x1_0000_0000", wait),
            ("DEPBAR.LE SB0, 0, 1", wait),
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
                "DEPBAR.LE SB0, 0x0",
                ControlBits::new(4, false, 7, 7, 1, 0).unwrap(),
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
            ("HMMA.16816.F32", 16),
            ("HMMA.16816.F32.BF16", 16),
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
    fn non_tensor_opcode_is_explicitly_dispatch_only() {
        for opcode in ["IADD3 R0, R1, R2, RZ", "XHMMA.16816.F32 R0", ""] {
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
