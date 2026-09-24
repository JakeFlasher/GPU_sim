//! Minimal functional SIMT executor for the user-defined architecture path.
//!
//! This is intentionally not an SM86 timing model. It executes one bounded
//! SIMT group using a tiny, explicit ISA (`simt_u32_v1`) and the lane width
//! from [`crate::spec::ArchitectureSpec`]. Its purpose is to prove that a
//! user-supplied architecture definition can feed an executable functional
//! backend alongside a hypothetical resource schedule. The functional
//! `SimtExecutor` executes one group; `simt_pipeline`/`sim-sm::custom` provide
//! bounded multi-group resource scheduling. It does not emulate unaligned/
//! byte accesses or a vendor GPU ISA.

use crate::json::strict_json_from_slice;
use crate::spec::ArchitectureSpec;
use serde::{Deserialize, Serialize};
use sim_semantics::{
    AddressSpace, AsyncEffect, Isa, LaneAddress, LaneMask, MemoryDirection, MemoryEvent,
    MemoryFamily, MemoryOperand, NativeInstruction, OperationKind, SemanticEvent,
    validate_async_effects,
};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

pub const SIMT_PROGRAM_SCHEMA_VERSION: u32 = 1;
const MAX_PROGRAM_BYTES: u64 = 1024 * 1024;
const REGISTER_COUNT: usize = 32;
const WORD_BYTES: u32 = 4;
const SUPPORTED_ISA: &str = "simt_u32_v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum SimtInstruction {
    SetLaneId {
        dst: u8,
    },
    Load {
        dst: u8,
        base: u32,
        stride: u32,
    },
    LoadShared {
        dst: u8,
        base: u32,
        stride: u32,
    },
    AddImm {
        dst: u8,
        src: u8,
        imm: i32,
    },
    Add {
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    BroadcastLane0 {
        dst: u8,
        src: u8,
    },
    ReduceAdd {
        dst: u8,
        src: u8,
    },
    Store {
        base: u32,
        stride: u32,
        src: u8,
    },
    StoreShared {
        base: u32,
        stride: u32,
        src: u8,
    },
    AtomicAddShared {
        base: u32,
        stride: u32,
        src: u8,
        dst: u8,
    },
    AsyncStore {
        base: u32,
        stride: u32,
        src: u8,
        group: u8,
    },
    WaitGroup {
        group: u8,
    },
    /// Whole-CTA rendezvous. The bounded multi-group frontend requires each
    /// generation id to occur at most once per group so static participation
    /// and cross-group dependencies remain unambiguous.
    GroupBarrier {
        id: u8,
    },
    Halt,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSimtProgram {
    schema_version: u32,
    architecture: String,
    isa: String,
    #[serde(default)]
    initial_memory: BTreeMap<u32, u32>,
    #[serde(default)]
    initial_shared_memory: BTreeMap<u32, u32>,
    instructions: Vec<SimtInstruction>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SimtProgram {
    schema_version: u32,
    architecture: String,
    isa: String,
    initial_memory: BTreeMap<u32, u32>,
    initial_shared_memory: BTreeMap<u32, u32>,
    instructions: Vec<SimtInstruction>,
}

impl SimtProgram {
    pub fn load(path: &Path) -> Result<Self, SimtError> {
        let file = std::fs::File::open(path).map_err(|source| SimtError::Io {
            path: path.display().to_string(),
            source,
        })?;
        let mut bytes = Vec::new();
        file.take(MAX_PROGRAM_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|source| SimtError::Io {
                path: path.display().to_string(),
                source,
            })?;
        Self::parse(path, &bytes)
    }

    pub fn parse(path: &Path, bytes: &[u8]) -> Result<Self, SimtError> {
        if bytes.len() as u64 > MAX_PROGRAM_BYTES {
            return Err(SimtError::ResourceLimit {
                path: path.display().to_string(),
                observed: bytes.len() as u64,
                limit: MAX_PROGRAM_BYTES,
            });
        }
        let raw: RawSimtProgram =
            strict_json_from_slice(bytes).map_err(|source| SimtError::ProgramParse {
                path: path.display().to_string(),
                source,
            })?;
        Self::from_raw(raw)
    }

    fn from_raw(raw: RawSimtProgram) -> Result<Self, SimtError> {
        if raw.schema_version != SIMT_PROGRAM_SCHEMA_VERSION {
            return Err(SimtError::UnsupportedProgramVersion {
                found: raw.schema_version,
                expected: SIMT_PROGRAM_SCHEMA_VERSION,
            });
        }
        if raw.architecture.is_empty() || raw.isa.is_empty() {
            return Err(SimtError::InvalidProgram(
                "architecture and isa are required",
            ));
        }
        if raw.instructions.is_empty() {
            return Err(SimtError::InvalidProgram("instructions must not be empty"));
        }
        if raw.instructions.len() > 4096 {
            return Err(SimtError::InvalidProgram("instruction limit is 4096"));
        }
        let halt_count = raw
            .instructions
            .iter()
            .filter(|instruction| matches!(instruction, SimtInstruction::Halt))
            .count();
        if halt_count != 1 || !matches!(raw.instructions.last(), Some(SimtInstruction::Halt)) {
            return Err(SimtError::InvalidProgram(
                "program must end with exactly one halt",
            ));
        }
        for instruction in &raw.instructions {
            validate_registers(instruction)?;
        }
        // This ISA stores aligned u32 words, not independently addressable
        // bytes. Reject overlapping initial words instead of silently keeping
        // contradictory values in the start-address map.
        for &address in raw.initial_memory.keys() {
            word_end(address)?;
        }
        for &address in raw.initial_shared_memory.keys() {
            word_end(address)?;
        }
        Ok(Self {
            schema_version: raw.schema_version,
            architecture: raw.architecture,
            isa: raw.isa,
            initial_memory: raw.initial_memory,
            initial_shared_memory: raw.initial_shared_memory,
            instructions: raw.instructions,
        })
    }

    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    pub fn isa(&self) -> &str {
        &self.isa
    }

    pub fn instructions(&self) -> &[SimtInstruction] {
        &self.instructions
    }

    pub fn initial_memory(&self) -> &BTreeMap<u32, u32> {
        &self.initial_memory
    }

    pub fn initial_shared_memory(&self) -> &BTreeMap<u32, u32> {
        &self.initial_shared_memory
    }
}

fn validate_registers(instruction: &SimtInstruction) -> Result<(), SimtError> {
    let registers = match instruction {
        SimtInstruction::SetLaneId { dst }
        | SimtInstruction::Load { dst, .. }
        | SimtInstruction::LoadShared { dst, .. } => vec![*dst],
        SimtInstruction::AddImm { dst, src, .. }
        | SimtInstruction::BroadcastLane0 { dst, src }
        | SimtInstruction::ReduceAdd { dst, src } => {
            vec![*dst, *src]
        }
        SimtInstruction::Add { dst, lhs, rhs } => vec![*dst, *lhs, *rhs],
        SimtInstruction::Store { src, .. } | SimtInstruction::StoreShared { src, .. } => {
            vec![*src]
        }
        SimtInstruction::AtomicAddShared { src, dst, .. } => vec![*src, *dst],
        SimtInstruction::AsyncStore { src, .. } => vec![*src],
        SimtInstruction::WaitGroup { .. } | SimtInstruction::GroupBarrier { .. } => Vec::new(),
        SimtInstruction::Halt => Vec::new(),
    };
    if registers
        .iter()
        .any(|register| usize::from(*register) >= REGISTER_COUNT)
    {
        return Err(SimtError::InvalidProgram("register index must be below 32"));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FunctionalResult {
    pub architecture: String,
    pub isa: String,
    pub simt_width: u8,
    pub instructions_executed: u64,
    pub registers: Vec<Vec<u32>>,
    pub memory: BTreeMap<u32, u32>,
    pub shared_memory: BTreeMap<u32, u32>,
}

pub struct SimtExecutor<'a> {
    spec: &'a ArchitectureSpec,
    program: &'a SimtProgram,
}

impl<'a> SimtExecutor<'a> {
    pub fn new(spec: &'a ArchitectureSpec, program: &'a SimtProgram) -> Result<Self, SimtError> {
        if spec.architecture() != program.architecture() {
            return Err(SimtError::IdentityMismatch {
                field: "architecture",
                spec: spec.architecture().to_owned(),
                program: program.architecture().to_owned(),
            });
        }
        if spec.isa() != program.isa() {
            return Err(SimtError::IdentityMismatch {
                field: "isa",
                spec: spec.isa().to_owned(),
                program: program.isa().to_owned(),
            });
        }
        if spec.isa() != SUPPORTED_ISA {
            return Err(SimtError::UnsupportedIsa(spec.isa().to_owned()));
        }
        for instruction in program.instructions() {
            let name = operation_name(instruction);
            if !spec.instruction_set().iter().any(|entry| entry == name) {
                return Err(SimtError::UnsupportedInstruction(name.to_owned()));
            }
        }
        Ok(Self { spec, program })
    }

    pub fn run(&self) -> Result<FunctionalResult, SimtError> {
        let mut group = SimtGroupState::new(self.spec.execution().simt_width())?;
        let mut memory = self.program.initial_memory().clone();
        let mut shared_memory = self.program.initial_shared_memory().clone();
        for instruction in self.program.instructions() {
            group.execute(instruction, &mut memory, &mut shared_memory)?;
        }
        group.finish(self.spec, &memory, &shared_memory)
    }
}

/// Functional lane state that survives a whole-CTA barrier. The CTA caller
/// owns address-space maps; groups must not keep private copies of shared data.
pub struct SimtGroupState {
    registers: Vec<Vec<u32>>,
    pending_async: BTreeMap<u8, Vec<(u32, u32, u32)>>,
    instructions_executed: u64,
}

impl SimtGroupState {
    pub fn new(width: u8) -> Result<Self, SimtError> {
        if !(1..=64).contains(&width) {
            return Err(SimtError::InvalidProgram("SIMT width must be in 1..=64"));
        }
        Ok(Self {
            registers: vec![vec![0; REGISTER_COUNT]; usize::from(width)],
            pending_async: BTreeMap::new(),
            instructions_executed: 0,
        })
    }

    pub fn execute(
        &mut self,
        instruction: &SimtInstruction,
        memory: &mut BTreeMap<u32, u32>,
        shared_memory: &mut BTreeMap<u32, u32>,
    ) -> Result<(), SimtError> {
        validate_registers(instruction)?;
        let width = self.registers.len();
        let Self {
            registers,
            pending_async,
            instructions_executed,
        } = self;
        let (base, stride, writes) = match instruction {
            SimtInstruction::Load { base, stride, .. }
            | SimtInstruction::LoadShared { base, stride, .. } => (Some(*base), *stride, false),
            SimtInstruction::Store { base, stride, .. }
            | SimtInstruction::StoreShared { base, stride, .. }
            | SimtInstruction::AtomicAddShared { base, stride, .. }
            | SimtInstruction::AsyncStore { base, stride, .. } => (Some(*base), *stride, true),
            _ => (None, 0, false),
        };
        if let Some(base) = base {
            let mut ranges = Vec::with_capacity(width);
            for lane in 0..width {
                let address = lane_address(base, stride, lane)?;
                let end = word_end(address)?;
                if writes
                    && !matches!(instruction, SimtInstruction::AtomicAddShared { .. })
                    && ranges.iter().any(|(prior_start, prior_end)| {
                        ranges_overlap(address, end, *prior_start, *prior_end)
                    })
                {
                    return Err(SimtError::AliasingStore(address));
                }
                if !matches!(
                    instruction,
                    SimtInstruction::LoadShared { .. }
                        | SimtInstruction::StoreShared { .. }
                        | SimtInstruction::AtomicAddShared { .. }
                ) && pending_async
                    .values()
                    .flatten()
                    .any(|(pending, pending_end, _)| {
                        ranges_overlap(address, end, *pending, *pending_end)
                    })
                {
                    return Err(SimtError::PendingMemoryAccess(address));
                }
                ranges.push((address, end));
            }
        }
        *instructions_executed = instructions_executed
            .checked_add(1)
            .ok_or(SimtError::InstructionCountOverflow)?;
        match *instruction {
            SimtInstruction::SetLaneId { dst } => {
                for (lane, registers) in registers.iter_mut().enumerate() {
                    registers[usize::from(dst)] = lane as u32;
                }
            }
            SimtInstruction::Load { dst, base, stride }
            | SimtInstruction::LoadShared { dst, base, stride } => {
                let source = if matches!(instruction, SimtInstruction::LoadShared { .. }) {
                    &*shared_memory
                } else {
                    &*memory
                };
                let values = registers
                    .iter()
                    .enumerate()
                    .map(|(lane, _)| {
                        let address = lane_address(base, stride, lane)?;
                        source
                            .get(&address)
                            .copied()
                            .ok_or(SimtError::MissingMemoryAddress(address))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                for (registers, value) in registers.iter_mut().zip(values) {
                    registers[usize::from(dst)] = value;
                }
            }
            SimtInstruction::AddImm { dst, src, imm } => {
                for registers in registers.iter_mut() {
                    registers[usize::from(dst)] =
                        registers[usize::from(src)].wrapping_add_signed(imm);
                }
            }
            SimtInstruction::BroadcastLane0 { dst, src } => {
                let value = registers[0][usize::from(src)];
                for registers in registers.iter_mut() {
                    registers[usize::from(dst)] = value;
                }
            }
            SimtInstruction::ReduceAdd { dst, src } => {
                let value = registers.iter().fold(0_u32, |sum, registers| {
                    sum.wrapping_add(registers[usize::from(src)])
                });
                for registers in registers.iter_mut() {
                    registers[usize::from(dst)] = value;
                }
            }
            SimtInstruction::Add { dst, lhs, rhs } => {
                for registers in registers.iter_mut() {
                    registers[usize::from(dst)] =
                        registers[usize::from(lhs)].wrapping_add(registers[usize::from(rhs)]);
                }
            }
            SimtInstruction::Store { base, stride, src }
            | SimtInstruction::StoreShared { base, stride, src } => {
                let destination = if matches!(instruction, SimtInstruction::StoreShared { .. }) {
                    &mut *shared_memory
                } else {
                    &mut *memory
                };
                for (lane, registers) in registers.iter().enumerate() {
                    let address = lane_address(base, stride, lane)?;
                    destination.insert(address, registers[usize::from(src)]);
                }
            }
            SimtInstruction::AtomicAddShared {
                base,
                stride,
                src,
                dst,
            } => {
                for (lane, registers) in registers.iter_mut().enumerate() {
                    let address = lane_address(base, stride, lane)?;
                    let old = shared_memory
                        .get(&address)
                        .copied()
                        .ok_or(SimtError::MissingMemoryAddress(address))?;
                    shared_memory.insert(address, old.wrapping_add(registers[usize::from(src)]));
                    registers[usize::from(dst)] = old;
                }
            }
            SimtInstruction::AsyncStore {
                base,
                stride,
                src,
                group,
            } => {
                if pending_async.contains_key(&group) {
                    return Err(SimtError::AsyncGroupReuse(group));
                }
                let stores = registers
                    .iter()
                    .enumerate()
                    .map(|(lane, registers)| {
                        let address = lane_address(base, stride, lane)?;
                        let end = word_end(address)?;
                        Ok::<(u32, u32, u32), SimtError>((
                            address,
                            end,
                            registers[usize::from(src)],
                        ))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                pending_async.insert(group, stores);
            }
            SimtInstruction::WaitGroup { group } => {
                let stores = pending_async
                    .remove(&group)
                    .ok_or(SimtError::MissingAsyncGroup(group))?;
                for (address, _end, value) in stores {
                    memory.insert(address, value);
                }
            }
            SimtInstruction::GroupBarrier { .. } | SimtInstruction::Halt => {
                if let Some(group) = pending_async.keys().next().copied() {
                    return Err(SimtError::UnfinishedAsyncGroup(group));
                }
            }
        }

        Ok(())
    }

    pub fn finish(
        self,
        spec: &ArchitectureSpec,
        memory: &BTreeMap<u32, u32>,
        shared_memory: &BTreeMap<u32, u32>,
    ) -> Result<FunctionalResult, SimtError> {
        if let Some(group) = self.pending_async.keys().next().copied() {
            return Err(SimtError::UnfinishedAsyncGroup(group));
        }
        Ok(FunctionalResult {
            architecture: spec.architecture().to_owned(),
            isa: spec.isa().to_owned(),
            simt_width: spec.execution().simt_width(),
            instructions_executed: self.instructions_executed,
            registers: self.registers,
            memory: memory.clone(),
            shared_memory: shared_memory.clone(),
        })
    }
}

/// A deterministic timing result for one bounded custom SIMT group, projected
/// from the shared `sim-engine` dependency/resource pipeline. It is not the
/// native SM86 timing model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TimingEvent {
    pub instruction_index: u64,
    pub operation: String,
    pub resource: Option<String>,
    pub start_cycle: u64,
    pub complete_cycle: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TimingResult {
    pub timing_model: &'static str,
    pub architecture: String,
    pub isa: String,
    pub simt_width: u8,
    pub final_cycle: u64,
    pub events: Vec<TimingEvent>,
    pub semantic_events: Vec<SemanticEvent>,
}

pub struct SimtTimingExecutor<'a> {
    spec: &'a ArchitectureSpec,
    program: &'a SimtProgram,
}

impl<'a> SimtTimingExecutor<'a> {
    pub fn new(spec: &'a ArchitectureSpec, program: &'a SimtProgram) -> Result<Self, SimtError> {
        SimtExecutor::new(spec, program)?;
        Ok(Self { spec, program })
    }

    pub fn run(&self) -> Result<TimingResult, SimtError> {
        SimtExecutor::new(self.spec, self.program)?.run()?;
        let resources = crate::simt_pipeline::resources(self.spec)?;
        let tasks = crate::simt_pipeline::tasks(self.spec, self.program)?;
        let schedule = sim_engine::pipeline::schedule(&resources, &[tasks], 1)?;
        self.project_schedule(&schedule.groups[0])
    }

    /// Shared by the single-program CLI and the production multi-group backend.
    pub fn project_schedule(
        &self,
        intervals: &[sim_engine::pipeline::Interval],
    ) -> Result<TimingResult, SimtError> {
        if intervals.len() != self.program.instructions().len() {
            return Err(SimtError::InvalidProgram("timing schedule length mismatch"));
        }
        let tasks = crate::simt_pipeline::tasks(self.spec, self.program)?;
        for (index, (task, interval)) in tasks.iter().zip(intervals).enumerate() {
            if interval.complete < interval.start
                || interval.start < index as u64
                || task
                    .dependencies
                    .iter()
                    .any(|dependency| interval.start < intervals[*dependency].complete)
            {
                return Err(SimtError::InvalidProgram(
                    "timing schedule violates issue or dependency order",
                ));
            }
        }
        let events: Vec<_> = self
            .program
            .instructions()
            .iter()
            .zip(intervals)
            .enumerate()
            .map(|(i, (instruction, interval))| TimingEvent {
                instruction_index: i as u64,
                operation: operation_name(instruction).to_owned(),
                resource: tasks[i]
                    .resource
                    .map(|r| self.spec.resources()[r].name().to_owned()),
                start_cycle: interval.start,
                complete_cycle: interval.complete,
            })
            .collect();
        let semantic_events = self.semantic_events(&events)?;
        validate_async_effects(&semantic_events).map_err(|_| SimtError::SemanticProjection)?;
        Ok(TimingResult {
            timing_model: "dependency_resource_v2",
            architecture: self.spec.architecture().to_owned(),
            isa: self.spec.isa().to_owned(),
            simt_width: self.spec.execution().simt_width(),
            final_cycle: events.iter().map(|e| e.complete_cycle).max().unwrap_or(0),
            events,
            semantic_events,
        })
    }

    fn semantic_events(&self, timing: &[TimingEvent]) -> Result<Vec<SemanticEvent>, SimtError> {
        let width = self.spec.execution().simt_width();
        let bits = if width == 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        let lane_mask = LaneMask::new(width, bits).map_err(|_| SimtError::SemanticProjection)?;
        self.program
            .instructions()
            .iter()
            .zip(timing)
            .enumerate()
            .map(|(index, (instruction, _timing))| {
                let native = NativeInstruction {
                    isa: Isa::CustomSimtU32,
                    simt_width: Some(width),
                    device_id: 0,
                    stream_id: 0,
                    kernel_id: 0,
                    block_launch_index: 0,
                    warp_id: 0,
                    instruction_index: index as u64,
                    function_unique_id: 0,
                    pc: (index as u64) * 4,
                    vpc: None,
                    sm_id: Some(0),
                    warpid_sm: Some(0),
                    instrumentation_order: None,
                };
                let (operation, memory) = match instruction {
                    SimtInstruction::SetLaneId { .. } => (OperationKind::SpecialRegister, None),
                    SimtInstruction::Load { base, stride, .. }
                    | SimtInstruction::LoadShared { base, stride, .. } => {
                        let addresses = (0..self.spec.execution().simt_width())
                            .map(|lane| {
                                Ok(LaneAddress {
                                    lane,
                                    address: u64::from(lane_address(
                                        *base,
                                        *stride,
                                        usize::from(lane),
                                    )?),
                                })
                            })
                            .collect::<Result<Vec<_>, SimtError>>()?;
                        (
                            OperationKind::Memory,
                            Some(MemoryEvent {
                                family: if matches!(instruction, SimtInstruction::LoadShared { .. })
                                {
                                    MemoryFamily::Shared
                                } else {
                                    MemoryFamily::Global
                                },
                                direction: Some(MemoryDirection::Load),
                                dynamic_class_present: true,
                                native_cache_op: None,
                                declared_width: Some(4),
                                operands: vec![MemoryOperand {
                                    address_space: Some(
                                        if matches!(instruction, SimtInstruction::LoadShared { .. })
                                        {
                                            AddressSpace::Shared
                                        } else {
                                            AddressSpace::Global
                                        },
                                    ),
                                    allocation_id: None,
                                    lane_mask,
                                    width_bytes: 4,
                                    lane_addresses: addresses,
                                    native: None,
                                }],
                            }),
                        )
                    }
                    SimtInstruction::AddImm { .. } | SimtInstruction::Add { .. } => {
                        (OperationKind::Arithmetic, None)
                    }
                    SimtInstruction::BroadcastLane0 { .. } => (OperationKind::LaneExchange, None),
                    SimtInstruction::ReduceAdd { .. } => (OperationKind::LaneExchange, None),
                    SimtInstruction::Store { base, stride, .. }
                    | SimtInstruction::StoreShared { base, stride, .. }
                    | SimtInstruction::AtomicAddShared { base, stride, .. }
                    | SimtInstruction::AsyncStore { base, stride, .. } => {
                        // Asynchrony is a token effect, not a different data
                        // movement operation. This writes registers to global
                        // memory; it is not a global-to-shared async copy.
                        let family = if matches!(
                            instruction,
                            SimtInstruction::StoreShared { .. }
                                | SimtInstruction::AtomicAddShared { .. }
                        ) {
                            MemoryFamily::Shared
                        } else {
                            MemoryFamily::Global
                        };
                        let direction =
                            if matches!(instruction, SimtInstruction::AtomicAddShared { .. }) {
                                MemoryDirection::Atomic
                            } else {
                                MemoryDirection::Store
                            };
                        let addresses = (0..self.spec.execution().simt_width())
                            .map(|lane| {
                                Ok(LaneAddress {
                                    lane,
                                    address: u64::from(lane_address(
                                        *base,
                                        *stride,
                                        usize::from(lane),
                                    )?),
                                })
                            })
                            .collect::<Result<Vec<_>, SimtError>>()?;
                        (
                            OperationKind::Memory,
                            Some(MemoryEvent {
                                family,
                                direction: Some(direction),
                                dynamic_class_present: true,
                                native_cache_op: None,
                                declared_width: Some(4),
                                operands: vec![MemoryOperand {
                                    address_space: Some(
                                        if matches!(
                                            instruction,
                                            SimtInstruction::StoreShared { .. }
                                                | SimtInstruction::AtomicAddShared { .. }
                                        ) {
                                            AddressSpace::Shared
                                        } else {
                                            AddressSpace::Global
                                        },
                                    ),
                                    allocation_id: None,
                                    lane_mask,
                                    width_bytes: 4,
                                    lane_addresses: addresses,
                                    native: None,
                                }],
                            }),
                        )
                    }
                    SimtInstruction::WaitGroup { .. } | SimtInstruction::GroupBarrier { .. } => {
                        (OperationKind::Synchronization, None)
                    }
                    SimtInstruction::Halt => (OperationKind::NoOp, None),
                };
                let event = SemanticEvent {
                    ordinal: index as u64,
                    native,
                    active_lanes: lane_mask,
                    predicate_lanes: lane_mask,
                    operation,
                    opcode_text: Some(operation_name(instruction).to_owned()),
                    control: None,
                    async_effect: match instruction {
                        SimtInstruction::AsyncStore { group, .. } => {
                            Some(AsyncEffect::Submit { group: *group })
                        }
                        SimtInstruction::WaitGroup { group } => {
                            Some(AsyncEffect::Wait { group: *group })
                        }
                        _ => None,
                    },
                    memory,
                    losses: Vec::new(),
                };
                event
                    .validate()
                    .map_err(|_| SimtError::SemanticProjection)?;
                Ok(event)
            })
            .collect()
    }
}

pub fn operation_name(instruction: &SimtInstruction) -> &'static str {
    match instruction {
        SimtInstruction::SetLaneId { .. } => "set_lane_id",
        SimtInstruction::Load { .. } => "load",
        SimtInstruction::LoadShared { .. } => "load_shared",
        SimtInstruction::AddImm { .. } => "add_imm",
        SimtInstruction::Add { .. } => "add",
        SimtInstruction::BroadcastLane0 { .. } => "broadcast_lane0",
        SimtInstruction::ReduceAdd { .. } => "reduce_add",
        SimtInstruction::Store { .. } => "store",
        SimtInstruction::StoreShared { .. } => "store_shared",
        SimtInstruction::AtomicAddShared { .. } => "atomic_add_shared",
        SimtInstruction::AsyncStore { .. } => "async_store",
        SimtInstruction::WaitGroup { .. } => "wait_group",
        SimtInstruction::GroupBarrier { .. } => "group_barrier",
        SimtInstruction::Halt => "halt",
    }
}

pub(crate) fn lane_address(base: u32, stride: u32, lane: usize) -> Result<u32, SimtError> {
    let lane = u32::try_from(lane).map_err(|_| SimtError::AddressOverflow)?;
    base.checked_add(stride.checked_mul(lane).ok_or(SimtError::AddressOverflow)?)
        .ok_or(SimtError::AddressOverflow)
}

fn word_end(address: u32) -> Result<u32, SimtError> {
    let end = address
        .checked_add(WORD_BYTES - 1)
        .ok_or(SimtError::AddressOverflow)?;
    if !address.is_multiple_of(WORD_BYTES) {
        return Err(SimtError::UnalignedAddress(address));
    }
    Ok(end)
}

fn ranges_overlap(left_start: u32, left_end: u32, right_start: u32, right_end: u32) -> bool {
    left_start <= right_end && right_start <= left_end
}

#[derive(Debug, thiserror::Error)]
pub enum SimtError {
    #[error("could not read SIMT program {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("SIMT program {path} could not be parsed: {source}")]
    ProgramParse {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("SIMT program is {observed} bytes, exceeding limit {limit}")]
    ResourceLimit {
        path: String,
        observed: u64,
        limit: u64,
    },
    #[error("SIMT program schema {found} is unsupported; expected {expected}")]
    UnsupportedProgramVersion { found: u32, expected: u32 },
    #[error("invalid SIMT program: {0}")]
    InvalidProgram(&'static str),
    #[error("SIMT program {field} identity {program:?} disagrees with spec {spec:?}")]
    IdentityMismatch {
        field: &'static str,
        spec: String,
        program: String,
    },
    #[error("ISA {0:?} has no functional backend in this slice")]
    UnsupportedIsa(String),
    #[error("instruction {0:?} is not admitted by the architecture spec")]
    UnsupportedInstruction(String),
    #[error("functional SIMT instruction count overflowed")]
    InstructionCountOverflow,
    #[error("functional SIMT load address {0:#x} is not initialized")]
    MissingMemoryAddress(u32),
    #[error(
        "word range beginning at {0:#x} overlaps another lane store; racing stores are outside this functional contract"
    )]
    AliasingStore(u32),
    #[error("word range beginning at {0:#x} is accessed before its async group is waited")]
    PendingMemoryAccess(u32),
    #[error("SIMT word address arithmetic overflowed the u32 byte-address domain")]
    AddressOverflow,
    #[error(
        "SIMT u32 word address {0:#x} must be four-byte aligned; byte/unaligned access is unsupported"
    )]
    UnalignedAddress(u32),
    #[error("async group {0} is reused before its prior operation is waited")]
    AsyncGroupReuse(u8),
    #[error("async group {0} was waited without a pending operation")]
    MissingAsyncGroup(u8),
    #[error("async group {0} remains pending at program halt")]
    UnfinishedAsyncGroup(u8),
    #[error("resource {0:?} is required by the timing backend but absent from the spec")]
    MissingResource(String),
    #[error("resource latency is unknown; timing cannot be claimed")]
    UnknownLatency,
    #[error("functional timing cycle overflowed")]
    CycleOverflow,
    #[error("custom SIMT semantic-event projection failed validation")]
    SemanticProjection,
    #[error("architecture spec error: {0}")]
    Spec(#[from] crate::spec::SpecError),
    #[error(transparent)]
    Pipeline(#[from] sim_engine::pipeline::PipelineError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::ArchitectureSpec;
    use std::path::Path;

    fn spec() -> ArchitectureSpec {
        ArchitectureSpec::parse(
            Path::new("spec.json"),
            include_bytes!("../../../arch_profiles/specs/simt_u32_minimal.json"),
        )
        .expect("spec fixture should parse")
    }

    fn program() -> SimtProgram {
        SimtProgram::parse(
            Path::new("program.json"),
            include_bytes!("../../../arch_profiles/programs/simt_u32_lane_store.json"),
        )
        .expect("program fixture should parse")
    }

    fn async_program() -> SimtProgram {
        SimtProgram::parse(
            Path::new("async_program.json"),
            include_bytes!("../../../arch_profiles/programs/simt_u32_async_store.json"),
        )
        .expect("async program fixture should parse")
    }

    #[test]
    fn lanes_execute_and_store_distinct_results() {
        let result = SimtExecutor::new(&spec(), &program())
            .expect("identities should match")
            .run()
            .expect("functional program should terminate");
        assert_eq!(result.simt_width, 8);
        assert_eq!(result.instructions_executed, 4);
        assert_eq!(result.memory.get(&100), Some(&10));
        assert_eq!(result.memory.get(&128), Some(&17));
    }

    #[test]
    fn identity_and_unsupported_isa_fail_closed() {
        let mut raw = serde_json::to_value(program()).expect("program serializes");
        raw["isa"] = serde_json::Value::String("other_isa".to_owned());
        let other = SimtProgram::parse(
            Path::new("other.json"),
            &serde_json::to_vec(&raw).expect("program serializes"),
        )
        .expect("identity mutation remains structurally valid");
        assert!(matches!(
            SimtExecutor::new(&spec(), &other),
            Err(SimtError::IdentityMismatch { field: "isa", .. })
        ));

        let raw = serde_json::json!({
            "schema_version": 1,
            "architecture": "example_simt8",
            "isa": "future_isa",
            "instructions": [{"op": "halt"}]
        });
        let future = SimtProgram::parse(
            Path::new("future.json"),
            &serde_json::to_vec(&raw).expect("program serializes"),
        )
        .expect("future program is structurally valid");
        let mut spec_raw = serde_json::to_value(spec()).expect("spec serializes");
        spec_raw["isa"] = serde_json::Value::String("future_isa".to_owned());
        let future_spec = ArchitectureSpec::parse(
            Path::new("future_spec.json"),
            &serde_json::to_vec(&spec_raw).expect("spec serializes"),
        )
        .expect("future spec is structurally valid");
        assert!(matches!(
            SimtExecutor::new(&future_spec, &future),
            Err(SimtError::UnsupportedIsa(isa)) if isa == "future_isa"
        ));
    }

    #[test]
    fn async_store_waits_for_declared_memory_latency() {
        let spec = spec();
        let program = async_program();
        let functional = SimtExecutor::new(&spec, &program)
            .expect("identities should match")
            .run()
            .expect("async program should drain at wait");
        assert_eq!(functional.memory.get(&200), Some(&10));
        assert_eq!(functional.memory.get(&228), Some(&17));

        let timing = SimtTimingExecutor::new(&spec, &program)
            .expect("timing identities should match")
            .run()
            .expect("timing program should terminate");
        assert_eq!(timing.simt_width, 8);
        assert_eq!(timing.final_cycle, 33);
        assert!(
            timing
                .events
                .iter()
                .any(|event| event.operation == "wait_group" && event.complete_cycle == 32)
        );
        assert!(matches!(
            timing.semantic_events[2].async_effect,
            Some(AsyncEffect::Submit { group: 0 })
        ));
        assert!(matches!(
            timing.semantic_events[3].async_effect,
            Some(AsyncEffect::Wait { group: 0 })
        ));
        assert_eq!(
            timing.semantic_events[2]
                .memory
                .as_ref()
                .expect("async semantic event carries memory")
                .operands[0]
                .lane_addresses[0]
                .address,
            200
        );
        assert_eq!(
            timing.semantic_events[2]
                .memory
                .as_ref()
                .expect("async semantic event carries memory")
                .operands[0]
                .lane_addresses[7]
                .address,
            228
        );
    }

    #[test]
    fn independent_alu_issues_while_memory_is_in_flight() {
        let program = SimtProgram::parse(
            Path::new("pipeline_overlap.json"),
            include_bytes!("../../../arch_profiles/programs/simt_u32_pipeline_overlap.json"),
        )
        .expect("overlap program should parse");
        let timing = SimtTimingExecutor::new(&spec(), &program)
            .expect("timing identities should match")
            .run()
            .expect("overlap program should terminate");
        let load = timing
            .events
            .iter()
            .find(|event| event.operation == "load")
            .expect("load event should exist");
        let independent_alu = timing
            .events
            .iter()
            .find(|event| event.instruction_index == 2)
            .expect("independent ALU event should exist");
        assert!(independent_alu.start_cycle < load.complete_cycle);
        assert_eq!(timing.final_cycle, 37);
    }

    #[test]
    fn shared_memory_has_separate_functional_and_semantic_state() {
        let spec = ArchitectureSpec::parse(
            Path::new("shared.json"),
            include_bytes!("../../../arch_profiles/specs/simt_u32_shared.json"),
        )
        .expect("shared spec should parse");
        let program = SimtProgram::parse(
            Path::new("shared_program.json"),
            include_bytes!("../../../arch_profiles/programs/simt_u32_shared.json"),
        )
        .expect("shared program should parse");
        let functional = SimtExecutor::new(&spec, &program)
            .expect("shared identities should match")
            .run()
            .expect("shared program should execute");
        assert!(functional.memory.is_empty());
        assert_eq!(functional.shared_memory.get(&200), Some(&15));
        let timing = SimtTimingExecutor::new(&spec, &program)
            .expect("shared timing identities should match")
            .run()
            .expect("shared timing should execute");
        let load = timing
            .semantic_events
            .iter()
            .find(|event| event.opcode_text.as_deref() == Some("load_shared"))
            .expect("shared load event should exist");
        let memory = load.memory.as_ref().expect("shared load carries memory");
        assert_eq!(memory.family, MemoryFamily::Shared);
        assert_eq!(memory.operands[0].address_space, Some(AddressSpace::Shared));
    }

    #[test]
    fn atomic_and_collective_semantics_are_typed() {
        let atomic_spec = ArchitectureSpec::parse(
            Path::new("atomic.json"),
            include_bytes!("../../../arch_profiles/specs/simt_u32_atomic.json"),
        )
        .expect("atomic spec should parse");
        let atomic_program = SimtProgram::parse(
            Path::new("atomic_program.json"),
            include_bytes!("../../../arch_profiles/programs/simt_u32_atomic.json"),
        )
        .expect("atomic program should parse");
        let atomic = SimtTimingExecutor::new(&atomic_spec, &atomic_program)
            .expect("atomic timing should initialize")
            .run()
            .expect("atomic timing should execute");
        let atomic_memory = atomic
            .semantic_events
            .iter()
            .find_map(|event| event.memory.as_ref())
            .expect("atomic event should carry memory semantics");
        assert_eq!(atomic_memory.family, MemoryFamily::Shared);
        assert_eq!(atomic_memory.direction, Some(MemoryDirection::Atomic));
        assert_eq!(
            atomic_memory.operands[0].address_space,
            Some(AddressSpace::Shared)
        );

        let collective_spec = ArchitectureSpec::parse(
            Path::new("collective.json"),
            include_bytes!("../../../arch_profiles/specs/simt_u32_collective.json"),
        )
        .expect("collective spec should parse");
        let collective_program = SimtProgram::parse(
            Path::new("collective_program.json"),
            include_bytes!("../../../arch_profiles/programs/simt_u32_collective.json"),
        )
        .expect("collective program should parse");
        let collective = SimtTimingExecutor::new(&collective_spec, &collective_program)
            .expect("collective timing should initialize")
            .run()
            .expect("collective timing should execute");
        assert!(
            collective
                .semantic_events
                .iter()
                .any(|event| event.operation == OperationKind::LaneExchange)
        );
    }

    #[test]
    fn word_aliases_and_address_wrap_fail_closed() {
        let raw = serde_json::json!({
            "schema_version": 1,
            "architecture": "example_simt8",
            "isa": "simt_u32_v1",
            "instructions": [
                {"op": "set_lane_id", "dst": 0},
                {"op": "store", "base": 100, "stride": 0, "src": 0},
                {"op": "halt"}
            ]
        });
        let aliased = SimtProgram::parse(
            Path::new("aliased.json"),
            &serde_json::to_vec(&raw).expect("program JSON should serialize"),
        )
        .expect("aliased program is structurally valid");
        assert!(matches!(
            SimtExecutor::new(&spec(), &aliased)
                .expect("identities should match")
                .run(),
            Err(SimtError::AliasingStore(100))
        ));

        let raw = serde_json::json!({
            "schema_version": 1,
            "architecture": "example_simt8",
            "isa": "simt_u32_v1",
            "instructions": [
                {"op": "store", "base": 4294967295u64, "stride": 0, "src": 0},
                {"op": "halt"}
            ]
        });
        let wrapped = SimtProgram::parse(
            Path::new("wrapped.json"),
            &serde_json::to_vec(&raw).expect("program JSON should serialize"),
        )
        .expect("overflowing program is structurally valid");
        assert!(matches!(
            SimtExecutor::new(&spec(), &wrapped)
                .expect("identities should match")
                .run(),
            Err(SimtError::AddressOverflow)
        ));
    }
}
