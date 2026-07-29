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

use sim_arch::{ArchError, ArchId};
use sim_mem::{MemRole, mem_role_of_checked, opcode_head};
use sim_trace::ExecInfo;

use crate::control_bits::{ControlBits, ControlSidecar};

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
    #[error("memory-role classification failed for architecture-aware schedule join: {0}")]
    MemRoleClassification(#[from] sim_mem::MemRoleError),
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpClass {
    /// `BAR.*` — block-wide barrier: the warp arrives, then waits until
    /// every unfinished warp of its block is waiting (or finished).
    BlockBarrier,
    /// Everything else: issue timing is fully described by the control
    /// bits plus (for memory ops) the dynamic record's memory class.
    Other,
}

/// Per-pc issue control: the sidecar's decoded control word plus the opcode
/// classifications the scheduler and the L1 model need.
#[derive(Debug, Clone)]
pub struct IssueSlot {
    pub control: ControlBits,
    pub op_class: OpClass,
    pub mem_role: MemRole,
    /// Full nvdisasm text, kept for diagnostics (static tables are small).
    pub opcode_text: String,
}

/// The replayable issue-control table for one traced function.
#[derive(Debug, Clone)]
pub struct FunctionSchedule {
    /// Deployment architecture proven by the trace/sidecar join.  Consumers
    /// must match this identity before pairing the schedule with any
    /// architecture-local timing or memory configuration.
    pub arch: ArchId,
    pub function_unique_id: i32,
    pub kernel_name: String,
    pub function_name: String,
    pub by_pc: BTreeMap<u64, IssueSlot>,
}

fn classify(opcode_text: &str) -> OpClass {
    if opcode_head(opcode_text) == "BAR" {
        OpClass::BlockBarrier
    } else {
        OpClass::Other
    }
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
    let mut schedules = BTreeMap::new();
    for (&fuid, kernel) in &exec.kernels {
        if kernel.architecture_version != sidecar_version {
            return Err(ProgramError::ArchMismatch {
                kernel_name: kernel.kernel_name.clone(),
                exec_version: kernel.architecture_version,
                sidecar_arch: sidecar.arch.clone(),
            });
        }
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
                Ok((
                    pc,
                    IssueSlot {
                        control: inst.control,
                        op_class: classify(&inst.opcode_text),
                        mem_role,
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
        assert_eq!(s.by_pc[&0].op_class, OpClass::Other);
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
    fn classify_uses_opcode_head_only() {
        assert_eq!(classify("BAR.SYNC.DEFER_BLOCKING"), OpClass::BlockBarrier);
        assert_eq!(
            classify("BAR.RED.AND.DEFER_BLOCKING 0x0"),
            OpClass::BlockBarrier
        );
        // BSSY/BSYNC/WARPSYNC/DEPBAR are NOT block-wide barriers.
        assert_eq!(classify("BSYNC B0"), OpClass::Other);
        assert_eq!(classify("BSSY B0, 0x1234"), OpClass::Other);
        assert_eq!(classify("WARPSYNC 0xffffffff"), OpClass::Other);
        assert_eq!(classify("DEPBAR.LE SB0, 0x0"), OpClass::Other);
        assert_eq!(classify("LDG.E R0, [R2.64]"), OpClass::Other);
        assert_eq!(classify(""), OpClass::Other);
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
    fn sm120_bare_ldg_is_a_typed_capability_gap() {
        let mut sc = sidecar(&[0]);
        sc.arch = "sm_120".to_owned();
        sc.functions
            .get_mut("my_kernel")
            .unwrap()
            .by_pc
            .get_mut(&0)
            .unwrap()
            .opcode_text = "LDG.E R0, [R2.64]".to_owned();
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
        assert!(matches!(err, ProgramError::MemRoleClassification(_)));
    }
}
