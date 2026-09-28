//! Bounded functional ITS executor for a user-defined SIMT architecture.
//!
//! This backend follows the functional boundary described by the CurryGPU
//! design notes: each lane has its own PC, divergence is represented directly,
//! and BSSY/BSYNC/BREAK/EXIT provide explicit reconvergence state. It is not a
//! timing model, compiler, RTL model, or claim about a physical GPU.

use crate::json::strict_json_from_slice;
use crate::spec::{ArchitectureSpec, SpecError};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

pub const ITS_PROGRAM_SCHEMA_VERSION: u32 = 1;
pub const ITS_ISA: &str = "simt_its_v1";
pub const ITS_TRACE_SCHEMA_VERSION: u32 = 3;
pub const ITS_TRACE_CONTRACT: &str = "custom_its_dynamic_v3";
const MAX_PROGRAM_BYTES: u64 = 1024 * 1024;
const MAX_INSTRUCTIONS: usize = 4096;
const MAX_STEPS: u64 = 10_000_000;
const MAX_TRACE_RECORDS: usize = 100_000;
const MAX_TRACE_SERIALIZED_BYTES: usize = 32 * 1024 * 1024;
const REGISTER_COUNT: usize = 32;
const PREDICATE_COUNT: usize = 8;
const BARRIER_COUNT: usize = 16;
const WORD_BYTES: u32 = 4;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum ItsInstruction {
    SetLaneId {
        dst: u8,
    },
    CmpLtImm {
        dst: u8,
        src: u8,
        imm: i32,
    },
    AddImm {
        dst: u8,
        src: u8,
        imm: i32,
    },
    MulImm {
        dst: u8,
        src: u8,
        imm: u32,
    },
    /// Aligned u32 access through a lane register, evaluated at execution.
    LoadIndirect {
        dst: u8,
        address: u8,
    },
    StoreIndirect {
        address: u8,
        src: u8,
    },
    /// Deterministic functional u32 atomic add through a lane address.
    /// Aliasing lanes are applied in ascending lane order; this is a custom
    /// functional contract, not native GPU atomic ordering.
    AtomicAddIndirect {
        dst: u8,
        address: u8,
        src: u8,
    },
    Branch {
        target: u32,
    },
    BranchIf {
        pred: u8,
        target: u32,
    },
    Bssy {
        barrier: u8,
        target: u32,
    },
    Bsync {
        barrier: u8,
    },
    Break {
        barrier: u8,
    },
    Store {
        base: u32,
        stride: u32,
        src: u8,
    },
    Exit,
    Halt,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawItsProgram {
    schema_version: u32,
    architecture: String,
    isa: String,
    #[serde(default)]
    initial_memory: BTreeMap<u32, u32>,
    #[serde(default = "default_max_steps")]
    max_steps: u64,
    instructions: Vec<ItsInstruction>,
}

fn default_max_steps() -> u64 {
    100_000
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ItsProgram {
    schema_version: u32,
    architecture: String,
    isa: String,
    initial_memory: BTreeMap<u32, u32>,
    max_steps: u64,
    instructions: Vec<ItsInstruction>,
}

impl ItsProgram {
    /// Construct a validated fixed-ISA program from a bounded frontend such
    /// as `ptx_subset_v2`. This does not make the frontend a general compiler.
    pub fn from_instructions(
        architecture: impl Into<String>,
        max_steps: u64,
        instructions: Vec<ItsInstruction>,
    ) -> Result<Self, ItsError> {
        Self::from_instructions_with_memory(architecture, max_steps, instructions, BTreeMap::new())
    }

    /// Construct a validated program with an explicitly supplied initial
    /// memory image.  Frontends use this to keep compilation and input data
    /// separate without editing generated JSON by hand.
    pub fn from_instructions_with_memory(
        architecture: impl Into<String>,
        max_steps: u64,
        instructions: Vec<ItsInstruction>,
        initial_memory: BTreeMap<u32, u32>,
    ) -> Result<Self, ItsError> {
        Self::from_raw(RawItsProgram {
            schema_version: ITS_PROGRAM_SCHEMA_VERSION,
            architecture: architecture.into(),
            isa: ITS_ISA.to_owned(),
            initial_memory,
            max_steps,
            instructions,
        })
    }

    pub fn load(path: &Path) -> Result<Self, ItsError> {
        let file = std::fs::File::open(path).map_err(|source| ItsError::Io {
            path: path.display().to_string(),
            source,
        })?;
        let mut bytes = Vec::new();
        file.take(MAX_PROGRAM_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|source| ItsError::Io {
                path: path.display().to_string(),
                source,
            })?;
        Self::parse(path, &bytes)
    }

    pub fn parse(path: &Path, bytes: &[u8]) -> Result<Self, ItsError> {
        if bytes.len() as u64 > MAX_PROGRAM_BYTES {
            return Err(ItsError::ResourceLimit {
                path: path.display().to_string(),
                observed: bytes.len() as u64,
                limit: MAX_PROGRAM_BYTES,
            });
        }
        let raw: RawItsProgram =
            strict_json_from_slice(bytes).map_err(|source| ItsError::ProgramParse {
                path: path.display().to_string(),
                source,
            })?;
        Self::from_raw(raw)
    }

    fn from_raw(raw: RawItsProgram) -> Result<Self, ItsError> {
        if raw.schema_version != ITS_PROGRAM_SCHEMA_VERSION {
            return Err(ItsError::UnsupportedProgramVersion {
                found: raw.schema_version,
                expected: ITS_PROGRAM_SCHEMA_VERSION,
            });
        }
        if raw.architecture.is_empty() || raw.isa != ITS_ISA {
            return Err(ItsError::InvalidProgram(
                "architecture is required and isa must be simt_its_v1",
            ));
        }
        if raw.instructions.is_empty() || raw.instructions.len() > MAX_INSTRUCTIONS {
            return Err(ItsError::InvalidProgram(
                "instructions must be nonempty and at most 4096 entries",
            ));
        }
        if raw.max_steps == 0 || raw.max_steps > MAX_STEPS {
            return Err(ItsError::InvalidProgram(
                "max_steps must be in 1..=10000000",
            ));
        }
        if !matches!(raw.instructions.last(), Some(ItsInstruction::Halt)) {
            return Err(ItsError::InvalidProgram("program must end with halt"));
        }
        let instruction_count = u32::try_from(raw.instructions.len())
            .map_err(|_| ItsError::InvalidProgram("instruction count does not fit u32"))?;
        for instruction in &raw.instructions {
            validate_instruction(instruction, instruction_count)?;
        }
        for &address in raw.initial_memory.keys() {
            word_end(address)?;
        }
        Ok(Self {
            schema_version: raw.schema_version,
            architecture: raw.architecture,
            isa: raw.isa,
            initial_memory: raw.initial_memory,
            max_steps: raw.max_steps,
            instructions: raw.instructions,
        })
    }

    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    pub fn isa(&self) -> &str {
        &self.isa
    }

    pub fn instructions(&self) -> &[ItsInstruction] {
        &self.instructions
    }
}

fn validate_instruction(
    instruction: &ItsInstruction,
    instruction_count: u32,
) -> Result<(), ItsError> {
    let registers = match instruction {
        ItsInstruction::SetLaneId { dst } => vec![*dst],
        ItsInstruction::AddImm { dst, src, .. } | ItsInstruction::MulImm { dst, src, .. } => {
            vec![*dst, *src]
        }
        ItsInstruction::LoadIndirect { dst, address } => vec![*dst, *address],
        ItsInstruction::StoreIndirect { address, src } => vec![*address, *src],
        ItsInstruction::AtomicAddIndirect { dst, address, src } => {
            vec![*dst, *address, *src]
        }
        ItsInstruction::CmpLtImm { src, .. } => vec![*src],
        ItsInstruction::Store { src, .. } => vec![*src],
        ItsInstruction::Branch { .. }
        | ItsInstruction::BranchIf { .. }
        | ItsInstruction::Bssy { .. }
        | ItsInstruction::Bsync { .. }
        | ItsInstruction::Break { .. }
        | ItsInstruction::Exit
        | ItsInstruction::Halt => Vec::new(),
    };
    if registers
        .iter()
        .any(|register| usize::from(*register) >= REGISTER_COUNT)
    {
        return Err(ItsError::InvalidProgram("register index must be below 32"));
    }
    if let ItsInstruction::CmpLtImm { dst, .. } | ItsInstruction::BranchIf { pred: dst, .. } =
        instruction
        && usize::from(*dst) >= PREDICATE_COUNT
    {
        return Err(ItsError::InvalidProgram("predicate index must be below 8"));
    }
    if let ItsInstruction::Branch { target }
    | ItsInstruction::BranchIf { target, .. }
    | ItsInstruction::Bssy { target, .. } = instruction
        && *target >= instruction_count
    {
        return Err(ItsError::InvalidProgram("branch target is out of range"));
    }
    if let ItsInstruction::Bssy { barrier, .. }
    | ItsInstruction::Bsync { barrier }
    | ItsInstruction::Break { barrier } = instruction
        && usize::from(*barrier) >= BARRIER_COUNT
    {
        return Err(ItsError::InvalidProgram("barrier index must be below 16"));
    }
    if let ItsInstruction::Bssy { barrier, target } = instruction
        && *target == 0
        && *barrier == 0
    {
        // This is legal in principle, but an immediately recursive
        // reconvergence shape is almost certainly a malformed example.
        return Err(ItsError::InvalidProgram(
            "Bssy target must not be the program entry",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ItsResult {
    pub architecture: String,
    pub isa: String,
    pub simt_width: u8,
    pub steps: u64,
    pub registers: Vec<Vec<u32>>,
    pub predicates: Vec<Vec<bool>>,
    pub pcs: Vec<usize>,
    pub active_lanes: Vec<bool>,
    pub memory: BTreeMap<u32, u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic_trace: Option<Vec<ItsTraceRecord>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic_trace_schema_version: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic_trace_contract: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic_trace_serialized_bytes: Option<u64>,
}

/// Execution-generated records, not NVBit/SASS and not hardware timestamps.
/// Successors include reconvergence releases; inactive lanes have no successor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ItsTraceRecord {
    pub sequence: u64,
    pub pc: usize,
    pub active_mask: u64,
    pub instruction: ItsInstruction,
    /// Complete lane-local architectural state at the record boundary. The
    /// trace is custom functional evidence, not native SASS; retaining these
    /// snapshots makes value-dependent replay independently inspectable rather
    /// than requiring consumers to infer values from successor masks.
    pub lane_registers_before: BTreeMap<usize, Vec<u32>>,
    pub lane_registers_after: BTreeMap<usize, Vec<u32>>,
    pub lane_predicates_before: BTreeMap<usize, Vec<bool>>,
    pub lane_predicates_after: BTreeMap<usize, Vec<bool>>,
    pub lane_addresses: BTreeMap<usize, u32>,
    pub lane_successors: BTreeMap<usize, usize>,
    /// Dynamic CFG edges grouped by successor PC. The value is the lane mask
    /// taking that edge in this executed record.
    pub successor_masks: BTreeMap<usize, u64>,
    /// Lanes blocked at the end of this record remain explicit instead of
    /// being mistaken for an unobserved successor.
    pub lane_blocked_by: BTreeMap<usize, u8>,
    /// PC of the explicit BSSY/BSYNC join instruction, if present.
    pub reconvergence_join_pc: Option<usize>,
    /// PC at which lanes resume after the join instruction, if representable.
    pub reconvergence_resume_pc: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
struct LaneState {
    pc: usize,
    active: bool,
    blocked_by: Option<u8>,
    regs: [u32; REGISTER_COUNT],
    predicates: [bool; PREDICATE_COUNT],
}

#[derive(Debug, Clone, Copy, Default)]
struct BarrierState {
    valid: bool,
    participation: u64,
    arrived: u64,
    join_pc: usize,
}

pub struct ItsExecutor<'a> {
    spec: &'a ArchitectureSpec,
    program: &'a ItsProgram,
}

impl<'a> ItsExecutor<'a> {
    pub fn new(spec: &'a ArchitectureSpec, program: &'a ItsProgram) -> Result<Self, ItsError> {
        if spec.architecture() != program.architecture() {
            return Err(ItsError::IdentityMismatch {
                field: "architecture",
                spec: spec.architecture().to_owned(),
                program: program.architecture().to_owned(),
            });
        }
        if spec.isa() != program.isa() {
            return Err(ItsError::IdentityMismatch {
                field: "isa",
                spec: spec.isa().to_owned(),
                program: program.isa().to_owned(),
            });
        }
        if spec.isa() != ITS_ISA {
            return Err(ItsError::UnsupportedIsa(spec.isa().to_owned()));
        }
        if !(1..=32).contains(&spec.execution().simt_width()) {
            return Err(ItsError::InvalidProgram(
                "ITS functional execution requires simt_width in 1..=32",
            ));
        }
        for instruction in program.instructions() {
            let name = operation_name(instruction);
            if !spec.instruction_set().iter().any(|entry| entry == name) {
                return Err(ItsError::UnsupportedInstruction(name.to_owned()));
            }
        }
        if spec.execution().max_resident_groups() != 1 {
            return Err(ItsError::Spec(SpecError::InvalidValue {
                field: "execution.max_resident_groups",
                detail: "this single-group executor supports exactly one resident group".to_owned(),
            }));
        }
        Ok(Self { spec, program })
    }

    pub fn run(&self) -> Result<ItsResult, ItsError> {
        self.run_internal(false)
    }

    pub fn run_traced(&self) -> Result<ItsResult, ItsError> {
        self.run_internal(true)
    }

    fn run_internal(&self, capture: bool) -> Result<ItsResult, ItsError> {
        let width = usize::from(self.spec.execution().simt_width());
        let mut lanes = vec![
            LaneState {
                pc: 0,
                active: true,
                blocked_by: None,
                regs: [0; REGISTER_COUNT],
                predicates: [false; PREDICATE_COUNT],
            };
            width
        ];
        let mut barriers = [BarrierState::default(); BARRIER_COUNT];
        let mut memory = self.program.initial_memory.clone();
        let mut steps = 0_u64;
        let mut dynamic_trace = capture.then(Vec::new);
        let mut trace_serialized_bytes = 0_usize;

        while lanes.iter().any(|lane| lane.active) {
            if steps >= self.program.max_steps {
                return Err(ItsError::StepLimit(self.program.max_steps));
            }
            let Some((pc, group)) = choose_group(&lanes) else {
                if fire_ready_barriers(&mut lanes, &mut barriers) {
                    continue;
                }
                return Err(ItsError::Deadlock);
            };
            let instruction = self
                .program
                .instructions
                .get(pc)
                .ok_or(ItsError::PcOutOfRange(pc))?;
            let (reconvergence_join_pc, reconvergence_resume_pc) = match instruction {
                ItsInstruction::Bssy { target, .. } => {
                    let join_pc = *target as usize;
                    (Some(join_pc), join_pc.checked_add(1))
                }
                ItsInstruction::Bsync { barrier } => barriers
                    .get(usize::from(*barrier))
                    .filter(|state| state.valid)
                    .map(|state| (Some(state.join_pc), state.join_pc.checked_add(1)))
                    .unwrap_or((None, None)),
                // A branch may be enclosed by several active barriers; the
                // bounded executor does not infer an IPDOM from that nesting.
                // Only explicit BSSY/BSYNC records carry a join target.
                ItsInstruction::BranchIf { .. } => (None, None),
                _ => (None, None),
            };
            steps = steps.checked_add(1).ok_or(ItsError::StepOverflow)?;
            // Addresses must be sampled before execution: dst may overwrite
            // the register that supplied a load address. Never regenerate a
            // trace from final register state or from a prior captured path.
            let lane_addresses = if capture {
                if steps > MAX_TRACE_RECORDS as u64 {
                    return Err(ItsError::TraceLimit);
                }
                lane_indices(group, width)
                    .filter_map(|lane| {
                        let address = match instruction {
                            ItsInstruction::LoadIndirect { address, .. }
                            | ItsInstruction::StoreIndirect { address, .. }
                            | ItsInstruction::AtomicAddIndirect { address, .. } => {
                                Ok(lanes[lane].regs[usize::from(*address)])
                            }
                            ItsInstruction::Store { base, stride, .. } => {
                                lane_address(*base, *stride, lane)
                            }
                            _ => return None,
                        };
                        Some(address.map(|address| (lane, address)))
                    })
                    .collect::<Result<BTreeMap<_, _>, _>>()?
            } else {
                BTreeMap::new()
            };
            let (lane_registers_before, lane_predicates_before) = if capture {
                (
                    lane_indices(group, width)
                        .map(|lane| (lane, lanes[lane].regs.to_vec()))
                        .collect(),
                    lane_indices(group, width)
                        .map(|lane| (lane, lanes[lane].predicates.to_vec()))
                        .collect(),
                )
            } else {
                (BTreeMap::new(), BTreeMap::new())
            };
            self.execute_group(
                instruction,
                pc,
                group,
                &mut lanes,
                &mut barriers,
                &mut memory,
            )?;
            fire_ready_barriers(&mut lanes, &mut barriers);
            if let Some(trace) = dynamic_trace.as_mut() {
                let record = ItsTraceRecord {
                    sequence: steps - 1,
                    pc,
                    active_mask: group,
                    instruction: instruction.clone(),
                    lane_registers_before,
                    lane_registers_after: lane_indices(group, width)
                        .map(|lane| (lane, lanes[lane].regs.to_vec()))
                        .collect(),
                    lane_predicates_before,
                    lane_predicates_after: lane_indices(group, width)
                        .map(|lane| (lane, lanes[lane].predicates.to_vec()))
                        .collect(),
                    lane_addresses,
                    lane_successors: lane_indices(group, width)
                        .filter(|&lane| lanes[lane].active && lanes[lane].blocked_by.is_none())
                        .map(|lane| (lane, lanes[lane].pc))
                        .collect(),
                    successor_masks: lane_indices(group, width)
                        .filter(|&lane| lanes[lane].active && lanes[lane].blocked_by.is_none())
                        .fold(BTreeMap::new(), |mut masks, lane| {
                            let target = lanes[lane].pc;
                            let entry = masks.entry(target).or_default();
                            *entry |= 1_u64 << lane;
                            masks
                        }),
                    lane_blocked_by: lane_indices(group, width)
                        .filter_map(|lane| lanes[lane].blocked_by.map(|barrier| (lane, barrier)))
                        .collect(),
                    reconvergence_join_pc,
                    reconvergence_resume_pc,
                };
                let serialized_bytes = serde_json::to_vec(&record)
                    .map_err(|error| ItsError::TraceSerialization(error.to_string()))?
                    .len();
                trace_serialized_bytes = trace_serialized_bytes
                    .checked_add(serialized_bytes)
                    .ok_or(ItsError::TraceLimit)?;
                if trace_serialized_bytes > MAX_TRACE_SERIALIZED_BYTES {
                    return Err(ItsError::TraceLimit);
                }
                trace.push(record);
            }
        }

        let registers = lanes
            .iter()
            .map(|lane| lane.regs.to_vec())
            .collect::<Vec<_>>();
        let predicates = lanes
            .iter()
            .map(|lane| lane.predicates.to_vec())
            .collect::<Vec<_>>();
        Ok(ItsResult {
            architecture: self.spec.architecture().to_owned(),
            isa: self.spec.isa().to_owned(),
            simt_width: self.spec.execution().simt_width(),
            steps,
            registers,
            predicates,
            pcs: lanes.iter().map(|lane| lane.pc).collect(),
            active_lanes: lanes.iter().map(|lane| lane.active).collect(),
            memory,
            dynamic_trace,
            dynamic_trace_schema_version: capture.then_some(ITS_TRACE_SCHEMA_VERSION),
            dynamic_trace_contract: capture.then_some(ITS_TRACE_CONTRACT),
            dynamic_trace_serialized_bytes: capture.then_some(trace_serialized_bytes as u64),
        })
    }

    fn execute_group(
        &self,
        instruction: &ItsInstruction,
        pc: usize,
        group: u64,
        lanes: &mut [LaneState],
        barriers: &mut [BarrierState; BARRIER_COUNT],
        memory: &mut BTreeMap<u32, u32>,
    ) -> Result<(), ItsError> {
        let width = lanes.len();
        match instruction {
            ItsInstruction::SetLaneId { dst } => {
                for_each_lane(group, width, |lane| {
                    lanes[lane].regs[usize::from(*dst)] = lane as u32;
                    lanes[lane].pc = pc + 1;
                });
            }
            ItsInstruction::CmpLtImm { dst, src, imm } => {
                for_each_lane(group, width, |lane| {
                    lanes[lane].predicates[usize::from(*dst)] =
                        (lanes[lane].regs[usize::from(*src)] as i32) < *imm;
                    lanes[lane].pc = pc + 1;
                });
            }
            ItsInstruction::AddImm { dst, src, imm } => {
                for_each_lane(group, width, |lane| {
                    lanes[lane].regs[usize::from(*dst)] =
                        lanes[lane].regs[usize::from(*src)].wrapping_add_signed(*imm);
                    lanes[lane].pc = pc + 1;
                });
            }
            ItsInstruction::MulImm { dst, src, imm } => {
                for_each_lane(group, width, |lane| {
                    lanes[lane].regs[usize::from(*dst)] =
                        lanes[lane].regs[usize::from(*src)].wrapping_mul(*imm);
                    lanes[lane].pc = pc + 1;
                });
            }
            ItsInstruction::LoadIndirect { dst, address } => {
                for lane in lane_indices(group, width) {
                    let address = lanes[lane].regs[usize::from(*address)];
                    word_end(address)?;
                    let value = *memory
                        .get(&address)
                        .ok_or(ItsError::UninitializedLoad(address))?;
                    lanes[lane].regs[usize::from(*dst)] = value;
                    lanes[lane].pc = pc + 1;
                }
            }
            ItsInstruction::StoreIndirect { address, src } => {
                let mut writes = BTreeMap::new();
                for lane in lane_indices(group, width) {
                    let address = lanes[lane].regs[usize::from(*address)];
                    word_end(address)?;
                    if writes
                        .insert(address, lanes[lane].regs[usize::from(*src)])
                        .is_some()
                    {
                        return Err(ItsError::AliasingStore(address));
                    }
                }
                memory.extend(writes);
                for_each_lane(group, width, |lane| lanes[lane].pc = pc + 1);
            }
            ItsInstruction::AtomicAddIndirect { dst, address, src } => {
                for lane in lane_indices(group, width) {
                    let address = lanes[lane].regs[usize::from(*address)];
                    word_end(address)?;
                    let old = *memory
                        .get(&address)
                        .ok_or(ItsError::UninitializedLoad(address))?;
                    let value = old.wrapping_add(lanes[lane].regs[usize::from(*src)]);
                    memory.insert(address, value);
                    lanes[lane].regs[usize::from(*dst)] = old;
                    lanes[lane].pc = pc + 1;
                }
            }
            ItsInstruction::Branch { target } => {
                for_each_lane(group, width, |lane| lanes[lane].pc = *target as usize);
            }
            ItsInstruction::BranchIf { pred, target } => {
                for_each_lane(group, width, |lane| {
                    lanes[lane].pc = if lanes[lane].predicates[usize::from(*pred)] {
                        *target as usize
                    } else {
                        pc + 1
                    };
                });
            }
            ItsInstruction::Bssy { barrier, target } => {
                let slot = &mut barriers[usize::from(*barrier)];
                if slot.valid {
                    return Err(ItsError::BarrierClobber(*barrier));
                }
                *slot = BarrierState {
                    valid: true,
                    participation: group,
                    arrived: 0,
                    join_pc: *target as usize,
                };
                for_each_lane(group, width, |lane| lanes[lane].pc = pc + 1);
            }
            ItsInstruction::Bsync { barrier } => {
                let slot = &mut barriers[usize::from(*barrier)];
                if !slot.valid {
                    return Err(ItsError::InvalidBarrierUse(*barrier));
                }
                let participating = group & slot.participation;
                let nonparticipating = group & !slot.participation;
                for_each_lane(nonparticipating, width, |lane| lanes[lane].pc = pc + 1);
                if participating != 0 {
                    if slot.join_pc != pc {
                        return Err(ItsError::InvalidBarrierUse(*barrier));
                    }
                    slot.arrived |= participating;
                    for_each_lane(participating, width, |lane| {
                        lanes[lane].blocked_by = Some(*barrier)
                    });
                }
            }
            ItsInstruction::Break { barrier } => {
                let slot = &mut barriers[usize::from(*barrier)];
                if !slot.valid {
                    return Err(ItsError::InvalidBarrierUse(*barrier));
                }
                slot.participation &= !group;
                slot.arrived &= !group;
                for_each_lane(group, width, |lane| lanes[lane].pc = pc + 1);
                if slot.participation == 0 {
                    *slot = BarrierState::default();
                    for_each_lane(group, width, |lane| lanes[lane].blocked_by = None);
                }
            }
            ItsInstruction::Store { base, stride, src } => {
                let mut writes = BTreeMap::new();
                for lane in lane_indices(group, width) {
                    let address = lane_address(*base, *stride, lane)?;
                    word_end(address)?;
                    if writes
                        .insert(address, lanes[lane].regs[usize::from(*src)])
                        .is_some()
                    {
                        return Err(ItsError::AliasingStore(address));
                    }
                }
                for (address, value) in writes {
                    memory.insert(address, value);
                }
                for_each_lane(group, width, |lane| lanes[lane].pc = pc + 1);
            }
            ItsInstruction::Exit => {
                for_each_lane(group, width, |lane| {
                    lanes[lane].active = false;
                    lanes[lane].blocked_by = None;
                });
                remove_from_barriers(group, barriers);
            }
            ItsInstruction::Halt => {
                for_each_lane(group, width, |lane| {
                    lanes[lane].active = false;
                    lanes[lane].blocked_by = None;
                });
                remove_from_barriers(group, barriers);
            }
        }
        Ok(())
    }
}

fn choose_group(lanes: &[LaneState]) -> Option<(usize, u64)> {
    let pc = lanes
        .iter()
        .filter(|lane| lane.active && lane.blocked_by.is_none())
        .map(|lane| lane.pc)
        .min()?;
    let group = lanes
        .iter()
        .enumerate()
        .filter(|(_, lane)| lane.active && lane.blocked_by.is_none() && lane.pc == pc)
        .fold(0_u64, |mask, (lane, _)| mask | (1_u64 << lane));
    Some((pc, group))
}

fn fire_ready_barriers(
    lanes: &mut [LaneState],
    barriers: &mut [BarrierState; BARRIER_COUNT],
) -> bool {
    let mut fired = false;
    for (barrier, slot) in barriers.iter_mut().enumerate() {
        if !slot.valid
            || slot.participation == 0
            || (slot.arrived & slot.participation) != slot.participation
        {
            continue;
        }
        let join = slot.join_pc + 1;
        for lane in lanes.iter_mut() {
            if lane.blocked_by == Some(barrier as u8) {
                lane.blocked_by = None;
                lane.pc = join;
            }
        }
        *slot = BarrierState::default();
        fired = true;
    }
    fired
}

fn remove_from_barriers(group: u64, barriers: &mut [BarrierState; BARRIER_COUNT]) {
    for slot in barriers.iter_mut() {
        if slot.valid {
            slot.participation &= !group;
            slot.arrived &= !group;
            if slot.participation == 0 {
                *slot = BarrierState::default();
            }
        }
    }
}

fn for_each_lane<F>(group: u64, width: usize, mut f: F)
where
    F: FnMut(usize),
{
    for lane in 0..width {
        if group & (1_u64 << lane) != 0 {
            f(lane);
        }
    }
}

fn lane_indices(group: u64, width: usize) -> impl Iterator<Item = usize> {
    (0..width).filter(move |lane| group & (1_u64 << lane) != 0)
}

fn lane_address(base: u32, stride: u32, lane: usize) -> Result<u32, ItsError> {
    let lane = u32::try_from(lane).map_err(|_| ItsError::AddressOverflow)?;
    base.checked_add(stride.checked_mul(lane).ok_or(ItsError::AddressOverflow)?)
        .ok_or(ItsError::AddressOverflow)
}

fn word_end(address: u32) -> Result<u32, ItsError> {
    if !address.is_multiple_of(WORD_BYTES) {
        return Err(ItsError::UnalignedAddress(address));
    }
    address
        .checked_add(WORD_BYTES - 1)
        .ok_or(ItsError::AddressOverflow)
}

fn operation_name(instruction: &ItsInstruction) -> &'static str {
    match instruction {
        ItsInstruction::SetLaneId { .. } => "set_lane_id",
        ItsInstruction::CmpLtImm { .. } => "cmp_lt_imm",
        ItsInstruction::AddImm { .. } => "add_imm",
        ItsInstruction::MulImm { .. } => "mul_imm",
        ItsInstruction::LoadIndirect { .. } => "load_indirect",
        ItsInstruction::StoreIndirect { .. } => "store_indirect",
        ItsInstruction::AtomicAddIndirect { .. } => "atomic_add_indirect",
        ItsInstruction::Branch { .. } => "branch",
        ItsInstruction::BranchIf { .. } => "branch_if",
        ItsInstruction::Bssy { .. } => "bssy",
        ItsInstruction::Bsync { .. } => "bsync",
        ItsInstruction::Break { .. } => "break",
        ItsInstruction::Store { .. } => "store",
        ItsInstruction::Exit => "exit",
        ItsInstruction::Halt => "halt",
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ItsError {
    #[error("could not read ITS program {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("ITS program {path} could not be parsed: {source}")]
    ProgramParse {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("ITS program is {observed} bytes, exceeding limit {limit}")]
    ResourceLimit {
        path: String,
        observed: u64,
        limit: u64,
    },
    #[error("ITS program schema {found} is unsupported; expected {expected}")]
    UnsupportedProgramVersion { found: u32, expected: u32 },
    #[error("invalid ITS program: {0}")]
    InvalidProgram(&'static str),
    #[error("ITS program {field} identity disagrees: spec={spec:?}, program={program:?}")]
    IdentityMismatch {
        field: &'static str,
        spec: String,
        program: String,
    },
    #[error("ISA {0:?} has no ITS functional backend")]
    UnsupportedIsa(String),
    #[error("ITS instruction {0:?} is not admitted by the architecture spec")]
    UnsupportedInstruction(String),
    #[error("ITS step limit {0} exceeded")]
    StepLimit(u64),
    #[error("ITS dynamic trace exceeds its record/serialized-byte resource bound")]
    TraceLimit,
    #[error("ITS dynamic trace record could not be serialized: {0}")]
    TraceSerialization(String),
    #[error("ITS load from uninitialized word {0:#x}")]
    UninitializedLoad(u32),
    #[error("ITS step counter overflowed")]
    StepOverflow,
    #[error("ITS lane PC {0} is out of range")]
    PcOutOfRange(usize),
    #[error("ITS execution deadlocked with blocked lanes")]
    Deadlock,
    #[error("ITS BSSY clobbered live barrier {0}")]
    BarrierClobber(u8),
    #[error("ITS barrier {0} was used before being armed or at the wrong PC")]
    InvalidBarrierUse(u8),
    #[error(
        "ITS store address {0:#x} overlaps another lane store; racing stores are outside this contract"
    )]
    AliasingStore(u32),
    #[error("ITS word address arithmetic overflowed the u32 byte-address domain")]
    AddressOverflow,
    #[error("ITS u32 word address {0:#x} must be four-byte aligned")]
    UnalignedAddress(u32),
    #[error("architecture spec error: {0}")]
    Spec(#[from] SpecError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::ArchitectureSpec;
    use std::path::Path;

    #[test]
    fn unmatched_bsync_is_rejected_instead_of_falling_through() {
        let spec = ArchitectureSpec::parse(
            Path::new("spec.json"),
            br#"
            {
              "schema_version": 1,
              "architecture": "probe",
              "isa": "simt_its_v1",
              "instruction_set": ["bsync", "halt"],
              "execution": {"simt_width": 1, "max_resident_groups": 1},
              "resources": [{"name":"alu","kind":"compute","issue_width":1,"latency_cycles":1}],
              "evidence": "unvalidated"
            }
            "#,
        )
        .expect("probe spec should parse");
        let program = ItsProgram::parse(
            Path::new("program.json"),
            br#"
            {
              "schema_version": 1,
              "architecture": "probe",
              "isa": "simt_its_v1",
              "max_steps": 4,
              "instructions": [
                {"op":"bsync", "barrier":0},
                {"op":"halt"}
              ]
            }
            "#,
        )
        .expect("probe program should parse");

        let error = ItsExecutor::new(&spec, &program)
            .expect("identity and ISA should be accepted")
            .run()
            .expect_err("unmatched BSYNC must fail closed");
        assert!(matches!(error, ItsError::InvalidBarrierUse(0)));
    }

    #[test]
    fn traced_execution_regenerates_value_dependent_addresses() {
        let spec = ArchitectureSpec::parse(
            Path::new("spec.json"),
            br#"
            {
              "schema_version": 1,
              "architecture": "probe",
              "isa": "simt_its_v1",
              "instruction_set": ["set_lane_id", "mul_imm", "load_indirect", "halt"],
              "execution": {"simt_width": 4, "max_resident_groups": 1},
              "resources": [{"name":"alu","kind":"compute","issue_width":1,"latency_cycles":1}],
              "evidence": "unvalidated"
            }
            "#,
        )
        .expect("probe spec should parse");
        let program = ItsProgram::parse(
            Path::new("program.json"),
            br#"
            {
              "schema_version": 1,
              "architecture": "probe",
              "isa": "simt_its_v1",
              "max_steps": 8,
              "initial_memory": {"0": 10, "4": 11, "8": 12, "12": 13},
              "instructions": [
                {"op":"set_lane_id", "dst":0},
                {"op":"mul_imm", "dst":1, "src":0, "imm":4},
                {"op":"load_indirect", "dst":2, "address":1},
                {"op":"halt"}
              ]
            }
            "#,
        )
        .expect("probe program should parse");
        let result = ItsExecutor::new(&spec, &program)
            .expect("identity and ISA should be accepted")
            .run_traced()
            .expect("dynamic execution should complete");
        assert_eq!(
            result.dynamic_trace_schema_version,
            Some(ITS_TRACE_SCHEMA_VERSION)
        );
        assert_eq!(result.dynamic_trace_contract, Some(ITS_TRACE_CONTRACT));
        let trace = result
            .dynamic_trace
            .as_ref()
            .expect("trace should be present");
        let expected_record_bytes = trace
            .iter()
            .map(|record| {
                serde_json::to_vec(record)
                    .expect("trace record should serialize")
                    .len()
            })
            .sum::<usize>();
        assert_eq!(
            result.dynamic_trace_serialized_bytes,
            Some(expected_record_bytes as u64)
        );
        let load = result
            .dynamic_trace
            .as_ref()
            .expect("trace should be present")
            .iter()
            .find(|record| matches!(record.instruction, ItsInstruction::LoadIndirect { .. }))
            .expect("load record should be present");
        assert_eq!(
            load.lane_addresses,
            [(0, 0), (1, 4), (2, 8), (3, 12)].into_iter().collect()
        );
        assert_eq!(load.lane_registers_before[&0][1], 0);
        assert_eq!(load.lane_registers_after[&0][2], 10);
        assert_eq!(load.lane_predicates_before[&0].len(), PREDICATE_COUNT);
        assert_eq!(load.lane_predicates_after[&0].len(), PREDICATE_COUNT);
        assert_eq!(result.registers[3][2], 13);
    }

    #[test]
    fn indirect_atomic_add_is_deterministic_for_aliasing_lanes() {
        let spec = ArchitectureSpec::parse(
            Path::new("spec.json"),
            br#"
            {
              "schema_version": 1,
              "architecture": "probe",
              "isa": "simt_its_v1",
              "instruction_set": ["set_lane_id", "mul_imm", "atomic_add_indirect", "halt"],
              "execution": {"simt_width": 4, "max_resident_groups": 1},
              "resources": [{"name":"alu","kind":"compute","issue_width":1,"latency_cycles":1}],
              "evidence": "unvalidated"
            }
            "#,
        )
        .expect("probe spec should parse");
        let program = ItsProgram::parse(
            Path::new("program.json"),
            br#"
            {
              "schema_version": 1,
              "architecture": "probe",
              "isa": "simt_its_v1",
              "max_steps": 8,
              "initial_memory": {"0": 10},
              "instructions": [
                {"op":"set_lane_id", "dst":0},
                {"op":"mul_imm", "dst":1, "src":0, "imm":0},
                {"op":"atomic_add_indirect", "dst":2, "address":1, "src":0},
                {"op":"halt"}
              ]
            }
            "#,
        )
        .expect("atomic probe program should parse");
        let result = ItsExecutor::new(&spec, &program)
            .expect("identity and ISA should be accepted")
            .run()
            .expect("atomic program should execute");
        assert_eq!(result.memory.get(&0), Some(&16));
        assert_eq!(result.registers[0][2], 10);
        assert_eq!(result.registers[3][2], 13);
    }

    #[test]
    fn traced_execution_tracks_input_dependent_branch_paths() {
        let spec = ArchitectureSpec::parse(
            Path::new("spec.json"),
            br#"
            {
              "schema_version": 1,
              "architecture": "probe",
              "isa": "simt_its_v1",
              "instruction_set": ["set_lane_id", "mul_imm", "load_indirect", "cmp_lt_imm", "branch_if", "add_imm", "branch", "halt"],
              "execution": {"simt_width": 4, "max_resident_groups": 1},
              "resources": [{"name":"alu","kind":"compute","issue_width":1,"latency_cycles":1}],
              "evidence": "unvalidated"
            }
            "#,
        )
        .expect("probe spec should parse");
        let program = |memory: &str| {
            ItsProgram::parse(
                Path::new("program.json"),
                format!(
                    r#"{{
                      "schema_version": 1,
                      "architecture": "probe",
                      "isa": "simt_its_v1",
                      "max_steps": 32,
                      "initial_memory": {memory},
                      "instructions": [
                        {{"op":"set_lane_id", "dst":0}},
                        {{"op":"mul_imm", "dst":1, "src":0, "imm":4}},
                        {{"op":"load_indirect", "dst":2, "address":1}},
                        {{"op":"cmp_lt_imm", "dst":0, "src":2, "imm":10}},
                        {{"op":"branch_if", "pred":0, "target":7}},
                        {{"op":"add_imm", "dst":2, "src":2, "imm":100}},
                        {{"op":"branch", "target":8}},
                        {{"op":"add_imm", "dst":2, "src":2, "imm":200}},
                        {{"op":"halt"}}
                      ]
                    }}"#
                )
                .as_bytes(),
            )
            .expect("input variant should parse")
        };
        let all_true = ItsExecutor::new(&spec, &program(r#"{"0":1,"4":2,"8":3,"12":4}"#))
            .expect("identity and ISA should be accepted")
            .run_traced()
            .expect("all-true branch variant should complete");
        let mixed = ItsExecutor::new(&spec, &program(r#"{"0":1,"4":12,"8":3,"12":14}"#))
            .expect("identity and ISA should be accepted")
            .run_traced()
            .expect("mixed branch variant should complete");

        let branch_successors = |result: &ItsResult| {
            result
                .dynamic_trace
                .as_ref()
                .expect("trace should be present")
                .iter()
                .find(|record| record.pc == 4)
                .expect("branch record should be present")
                .lane_successors
                .clone()
        };
        assert_eq!(
            branch_successors(&all_true),
            [(0, 7), (1, 7), (2, 7), (3, 7)].into_iter().collect()
        );
        assert_eq!(
            branch_successors(&mixed),
            [(0, 7), (1, 5), (2, 7), (3, 5)].into_iter().collect()
        );
        let mixed_branch = mixed
            .dynamic_trace
            .as_ref()
            .expect("trace should be present")
            .iter()
            .find(|record| record.pc == 4)
            .expect("branch record should be present");
        assert_eq!(
            mixed_branch.successor_masks,
            [(5, 0b1010), (7, 0b0101)].into_iter().collect()
        );
        assert_eq!(mixed_branch.reconvergence_join_pc, None);
        assert_eq!(mixed_branch.reconvergence_resume_pc, None);
        assert_eq!(
            all_true
                .registers
                .iter()
                .map(|lane| lane[2])
                .collect::<Vec<_>>(),
            [201, 202, 203, 204]
        );
        assert_eq!(
            mixed
                .registers
                .iter()
                .map(|lane| lane[2])
                .collect::<Vec<_>>(),
            [201, 112, 203, 114]
        );
        let mixed_add_masks = mixed
            .dynamic_trace
            .as_ref()
            .expect("trace should be present")
            .iter()
            .filter(|record| matches!(record.instruction, ItsInstruction::AddImm { .. }))
            .map(|record| record.active_mask)
            .collect::<Vec<_>>();
        assert_eq!(mixed_add_masks, [0b1010, 0b0101]);
    }

    #[test]
    fn traced_execution_keeps_partially_arrived_barrier_lanes_explicit() {
        let spec = ArchitectureSpec::parse(
            Path::new("spec.json"),
            br#"
            {
              "schema_version": 1,
              "architecture": "probe",
              "isa": "simt_its_v1",
              "instruction_set": ["set_lane_id", "cmp_lt_imm", "bssy", "branch_if", "branch", "bsync", "halt"],
              "execution": {"simt_width": 2, "max_resident_groups": 1},
              "resources": [{"name":"alu","kind":"compute","issue_width":1,"latency_cycles":1}],
              "evidence": "unvalidated"
            }
            "#,
        )
        .expect("probe spec should parse");
        let program = ItsProgram::parse(
            Path::new("program.json"),
            br#"
            {
              "schema_version": 1,
              "architecture": "probe",
              "isa": "simt_its_v1",
              "max_steps": 32,
              "instructions": [
                {"op":"set_lane_id", "dst":0},
                {"op":"cmp_lt_imm", "dst":0, "src":0, "imm":1},
                {"op":"bssy", "barrier":0, "target":5},
                {"op":"branch_if", "pred":0, "target":5},
                {"op":"branch", "target":6},
                {"op":"bsync", "barrier":0},
                {"op":"halt"}
              ]
            }
            "#,
        )
        .expect("probe program should parse");
        let result = ItsExecutor::new(&spec, &program)
            .expect("identity and ISA should be accepted")
            .run_traced()
            .expect("partial barrier arrival should complete");
        let bsync = result
            .dynamic_trace
            .as_ref()
            .expect("trace should be present")
            .iter()
            .find(|record| matches!(record.instruction, ItsInstruction::Bsync { .. }))
            .expect("bsync record should be present");
        assert_eq!(bsync.reconvergence_join_pc, Some(5));
        assert_eq!(bsync.reconvergence_resume_pc, Some(6));
        assert_eq!(bsync.lane_blocked_by, [(0, 0)].into_iter().collect());
        assert!(!bsync.lane_successors.contains_key(&0));
    }

    #[test]
    fn traced_execution_fails_closed_at_record_resource_bound() {
        let spec = ArchitectureSpec::parse(
            Path::new("spec.json"),
            br#"
            {
              "schema_version": 1,
              "architecture": "probe",
              "isa": "simt_its_v1",
              "instruction_set": ["branch", "halt"],
              "execution": {"simt_width": 1, "max_resident_groups": 1},
              "resources": [{"name":"alu","kind":"compute","issue_width":1,"latency_cycles":1}],
              "evidence": "unvalidated"
            }
            "#,
        )
        .expect("probe spec should parse");
        let program = ItsProgram::parse(
            Path::new("program.json"),
            br#"
            {
              "schema_version": 1,
              "architecture": "probe",
              "isa": "simt_its_v1",
              "max_steps": 10000000,
              "instructions": [
                {"op":"branch", "target":0},
                {"op":"halt"}
              ]
            }
            "#,
        )
        .expect("probe program should parse");
        let error = ItsExecutor::new(&spec, &program)
            .expect("identity and ISA should be accepted")
            .run_traced()
            .expect_err("trace resource bound must fail closed");
        assert!(matches!(error, ItsError::TraceLimit));
    }
}
