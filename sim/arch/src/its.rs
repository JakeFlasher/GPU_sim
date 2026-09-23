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
const MAX_PROGRAM_BYTES: u64 = 1024 * 1024;
const MAX_INSTRUCTIONS: usize = 4096;
const MAX_STEPS: u64 = 10_000_000;
const REGISTER_COUNT: usize = 32;
const PREDICATE_COUNT: usize = 8;
const BARRIER_COUNT: usize = 16;
const WORD_BYTES: u32 = 4;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum ItsInstruction {
    SetLaneId { dst: u8 },
    CmpLtImm { dst: u8, src: u8, imm: i32 },
    AddImm { dst: u8, src: u8, imm: i32 },
    Branch { target: u32 },
    BranchIf { pred: u8, target: u32 },
    Bssy { barrier: u8, target: u32 },
    Bsync { barrier: u8 },
    Break { barrier: u8 },
    Store { base: u32, stride: u32, src: u8 },
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
        ItsInstruction::SetLaneId { dst } | ItsInstruction::AddImm { dst, .. } => vec![*dst],
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
    {
        if usize::from(*dst) >= PREDICATE_COUNT {
            return Err(ItsError::InvalidProgram("predicate index must be below 8"));
        }
    }
    if let ItsInstruction::Branch { target }
    | ItsInstruction::BranchIf { target, .. }
    | ItsInstruction::Bssy { target, .. } = instruction
    {
        if *target >= instruction_count {
            return Err(ItsError::InvalidProgram("branch target is out of range"));
        }
    }
    if let ItsInstruction::Bssy { barrier, .. }
    | ItsInstruction::Bsync { barrier }
    | ItsInstruction::Break { barrier } = instruction
    {
        if usize::from(*barrier) >= BARRIER_COUNT {
            return Err(ItsError::InvalidProgram("barrier index must be below 16"));
        }
    }
    if let ItsInstruction::Bssy { barrier, target } = instruction {
        if *target == 0 && *barrier == 0 {
            // This is legal in principle, but an immediately recursive
            // reconvergence shape is almost certainly a malformed example.
            return Err(ItsError::InvalidProgram(
                "Bssy target must not be the program entry",
            ));
        }
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
}

#[derive(Debug, Clone, Copy)]
struct LaneState {
    pc: usize,
    active: bool,
    blocked_by: Option<u8>,
    regs: [u32; REGISTER_COUNT],
    predicates: [bool; PREDICATE_COUNT],
}

#[derive(Debug, Clone, Copy)]
struct BarrierState {
    valid: bool,
    participation: u64,
    arrived: u64,
    join_pc: usize,
}

impl Default for BarrierState {
    fn default() -> Self {
        Self {
            valid: false,
            participation: 0,
            arrived: 0,
            join_pc: 0,
        }
    }
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
        if spec.execution().max_resident_groups() < 1 {
            return Err(ItsError::Spec(SpecError::InvalidValue {
                field: "execution.max_resident_groups",
                detail: "must be positive".to_owned(),
            }));
        }
        Ok(Self { spec, program })
    }

    pub fn run(&self) -> Result<ItsResult, ItsError> {
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
            steps = steps.checked_add(1).ok_or(ItsError::StepOverflow)?;
            self.execute_group(
                instruction,
                pc,
                group,
                &mut lanes,
                &mut barriers,
                &mut memory,
            )?;
            fire_ready_barriers(&mut lanes, &mut barriers);
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
                        i64::from(lanes[lane].regs[usize::from(*src)]) < i64::from(*imm);
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
                    // BREAK/EXIT may dissolve a token before a remaining
                    // control-flow edge reaches its textual BSYNC. The
                    // functional contract treats that use as fallthrough.
                    for_each_lane(group, width, |lane| lanes[lane].pc = pc + 1);
                    return Ok(());
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
