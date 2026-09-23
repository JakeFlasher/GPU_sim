//! Minimal functional SIMT executor for the user-defined architecture path.
//!
//! This is intentionally not an SM86 timing model. It executes one bounded
//! SIMT group using a tiny, explicit ISA (`simt_u32_v1`) and the lane width
//! from [`crate::spec::ArchitectureSpec`]. Its purpose is to prove that a
//! user-supplied architecture definition can feed an executable functional
//! backend alongside a hypothetical resource schedule. It does not emulate
//! unaligned/byte accesses, multiple resident groups, or a vendor GPU ISA.

use crate::json::strict_json_from_slice;
use crate::spec::{ArchitectureSpec, ResourceKind, SpecError};
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
    Store {
        base: u32,
        stride: u32,
        src: u8,
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
    instructions: Vec<SimtInstruction>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SimtProgram {
    schema_version: u32,
    architecture: String,
    isa: String,
    initial_memory: BTreeMap<u32, u32>,
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
        Ok(Self {
            schema_version: raw.schema_version,
            architecture: raw.architecture,
            isa: raw.isa,
            initial_memory: raw.initial_memory,
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
}

fn validate_registers(instruction: &SimtInstruction) -> Result<(), SimtError> {
    let registers = match instruction {
        SimtInstruction::SetLaneId { dst } | SimtInstruction::Load { dst, .. } => vec![*dst],
        SimtInstruction::AddImm { dst, src, .. } => vec![*dst, *src],
        SimtInstruction::Add { dst, lhs, rhs } => vec![*dst, *lhs, *rhs],
        SimtInstruction::Store { src, .. } => vec![*src],
        SimtInstruction::AsyncStore { src, .. } => vec![*src],
        SimtInstruction::WaitGroup { .. } => Vec::new(),
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
        if spec.execution().max_resident_groups() < 1 {
            return Err(SimtError::Spec(SpecError::InvalidValue {
                field: "execution.max_resident_groups",
                detail: "must be positive".to_owned(),
            }));
        }
        Ok(Self { spec, program })
    }

    pub fn run(&self) -> Result<FunctionalResult, SimtError> {
        let width = usize::from(self.spec.execution().simt_width());
        let mut registers = vec![vec![0u32; REGISTER_COUNT]; width];
        let mut memory = self.program.initial_memory().clone();
        let mut pending_async = BTreeMap::<u8, Vec<(u32, u32, u32)>>::new();
        let mut instructions_executed = 0u64;

        for instruction in self.program.instructions() {
            let (base, stride, writes) = match instruction {
                SimtInstruction::Load { base, stride, .. } => (Some(*base), *stride, false),
                SimtInstruction::Store { base, stride, .. }
                | SimtInstruction::AsyncStore { base, stride, .. } => (Some(*base), *stride, true),
                _ => (None, 0, false),
            };
            if let Some(base) = base {
                let mut ranges = Vec::with_capacity(width);
                for lane in 0..width {
                    let address = lane_address(base, stride, lane)?;
                    let end = word_end(address)?;
                    if writes
                        && ranges.iter().any(|(prior_start, prior_end)| {
                            ranges_overlap(address, end, *prior_start, *prior_end)
                        })
                    {
                        return Err(SimtError::AliasingStore(address));
                    }
                    if pending_async
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
            instructions_executed = instructions_executed
                .checked_add(1)
                .ok_or(SimtError::InstructionCountOverflow)?;
            match *instruction {
                SimtInstruction::SetLaneId { dst } => {
                    for (lane, registers) in registers.iter_mut().enumerate() {
                        registers[usize::from(dst)] = lane as u32;
                    }
                }
                SimtInstruction::Load { dst, base, stride } => {
                    let values = registers
                        .iter()
                        .enumerate()
                        .map(|(lane, _)| {
                            let address = lane_address(base, stride, lane)?;
                            memory
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
                    for registers in &mut registers {
                        registers[usize::from(dst)] =
                            registers[usize::from(src)].wrapping_add_signed(imm);
                    }
                }
                SimtInstruction::Add { dst, lhs, rhs } => {
                    for registers in &mut registers {
                        registers[usize::from(dst)] =
                            registers[usize::from(lhs)].wrapping_add(registers[usize::from(rhs)]);
                    }
                }
                SimtInstruction::Store { base, stride, src } => {
                    for (lane, registers) in registers.iter().enumerate() {
                        let address = lane_address(base, stride, lane)?;
                        memory.insert(address, registers[usize::from(src)]);
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
                SimtInstruction::Halt => break,
            }
        }
        if let Some(group) = pending_async.keys().next().copied() {
            return Err(SimtError::UnfinishedAsyncGroup(group));
        }

        Ok(FunctionalResult {
            architecture: self.spec.architecture().to_owned(),
            isa: self.spec.isa().to_owned(),
            simt_width: self.spec.execution().simt_width(),
            instructions_executed,
            registers,
            memory,
        })
    }
}

/// A deterministic timing result for one bounded custom SIMT group. This is
/// deliberately a tiny event schedule, not a replacement for `sim-engine` or
/// the SM86 timing model.
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
        // Timing success must not accept a program that has undefined functional
        // effects (e.g. an uninitialized load or an unconsumed async token).
        SimtExecutor::new(self.spec, self.program)?.run()?;
        let width = u64::from(self.spec.execution().simt_width());
        let mut now = 0u64;
        let mut pending = BTreeMap::<u8, u64>::new();
        let mut resource_next = BTreeMap::<String, u64>::new();
        let mut events = Vec::with_capacity(self.program.instructions().len());

        for (instruction_index, instruction) in self.program.instructions().iter().enumerate() {
            let instruction_index = u64::try_from(instruction_index)
                .map_err(|_| SimtError::InstructionCountOverflow)?;
            match instruction {
                SimtInstruction::SetLaneId { .. }
                | SimtInstruction::AddImm { .. }
                | SimtInstruction::Add { .. } => {
                    let resource = self.resource("alu")?;
                    let start = reserve_resource(&mut resource_next, resource, now, 1)?;
                    let span = issue_span(1, resource.issue_width())?;
                    let complete = checked_cycle(start, span, resource.latency_cycles())?;
                    now = complete;
                    events.push(TimingEvent {
                        instruction_index,
                        operation: operation_name(instruction).to_owned(),
                        resource: Some(resource.name().to_owned()),
                        start_cycle: start,
                        complete_cycle: complete,
                    });
                }
                SimtInstruction::Load { .. } | SimtInstruction::Store { .. } => {
                    let resource = self.resource("memory")?;
                    let start = reserve_resource(&mut resource_next, resource, now, width)?;
                    let span = issue_span(width, resource.issue_width())?;
                    let complete = checked_cycle(start, span, resource.latency_cycles())?;
                    now = complete;
                    events.push(TimingEvent {
                        instruction_index,
                        operation: operation_name(instruction).to_owned(),
                        resource: Some(resource.name().to_owned()),
                        start_cycle: start,
                        complete_cycle: complete,
                    });
                }
                SimtInstruction::AsyncStore { group, .. } => {
                    if pending.contains_key(group) {
                        return Err(SimtError::AsyncGroupReuse(*group));
                    }
                    let resource = self.resource("memory")?;
                    let start = reserve_resource(&mut resource_next, resource, now, width)?;
                    let span = issue_span(width, resource.issue_width())?;
                    let complete = checked_cycle(start, span, resource.latency_cycles())?;
                    pending.insert(*group, complete);
                    now = start.checked_add(span).ok_or(SimtError::CycleOverflow)?;
                    events.push(TimingEvent {
                        instruction_index,
                        operation: operation_name(instruction).to_owned(),
                        resource: Some(resource.name().to_owned()),
                        start_cycle: start,
                        complete_cycle: complete,
                    });
                }
                SimtInstruction::WaitGroup { group } => {
                    let ready = pending
                        .remove(group)
                        .ok_or(SimtError::MissingAsyncGroup(*group))?;
                    let start = now;
                    now = now.max(ready);
                    events.push(TimingEvent {
                        instruction_index,
                        operation: operation_name(instruction).to_owned(),
                        resource: None,
                        start_cycle: start,
                        complete_cycle: now,
                    });
                }
                SimtInstruction::Halt => {
                    if let Some(group) = pending.keys().next().copied() {
                        return Err(SimtError::UnfinishedAsyncGroup(group));
                    }
                    events.push(TimingEvent {
                        instruction_index,
                        operation: operation_name(instruction).to_owned(),
                        resource: None,
                        start_cycle: now,
                        complete_cycle: now,
                    });
                }
            }
        }

        let semantic_events = self.semantic_events(&events)?;
        validate_async_effects(&semantic_events).map_err(|_| SimtError::SemanticProjection)?;
        Ok(TimingResult {
            architecture: self.spec.architecture().to_owned(),
            isa: self.spec.isa().to_owned(),
            simt_width: self.spec.execution().simt_width(),
            final_cycle: now,
            events,
            semantic_events,
        })
    }

    fn resource(&self, name: &str) -> Result<&crate::spec::ResourceSpec, SimtError> {
        let resource = self
            .spec
            .resources()
            .iter()
            .find(|resource| resource.name() == name)
            .ok_or_else(|| SimtError::MissingResource(name.to_owned()))?;
        let kind = if name == "alu" {
            ResourceKind::Compute
        } else {
            ResourceKind::Memory
        };
        if resource.kind() != kind {
            return Err(SimtError::InvalidProgram("timing resource kind mismatch"));
        }
        Ok(resource)
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
                    SimtInstruction::Load { base, stride, .. } => {
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
                                family: MemoryFamily::Global,
                                direction: Some(MemoryDirection::Load),
                                dynamic_class_present: true,
                                native_cache_op: None,
                                declared_width: Some(4),
                                operands: vec![MemoryOperand {
                                    address_space: Some(AddressSpace::Global),
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
                    SimtInstruction::Store { base, stride, .. }
                    | SimtInstruction::AsyncStore { base, stride, .. } => {
                        // Asynchrony is a token effect, not a different data
                        // movement operation. This writes registers to global
                        // memory; it is not a global-to-shared async copy.
                        let (family, direction) = (MemoryFamily::Global, MemoryDirection::Store);
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
                                    address_space: Some(AddressSpace::Global),
                                    allocation_id: None,
                                    lane_mask,
                                    width_bytes: 4,
                                    lane_addresses: addresses,
                                    native: None,
                                }],
                            }),
                        )
                    }
                    SimtInstruction::WaitGroup { .. } => (OperationKind::Synchronization, None),
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

fn operation_name(instruction: &SimtInstruction) -> &'static str {
    match instruction {
        SimtInstruction::SetLaneId { .. } => "set_lane_id",
        SimtInstruction::Load { .. } => "load",
        SimtInstruction::AddImm { .. } => "add_imm",
        SimtInstruction::Add { .. } => "add",
        SimtInstruction::Store { .. } => "store",
        SimtInstruction::AsyncStore { .. } => "async_store",
        SimtInstruction::WaitGroup { .. } => "wait_group",
        SimtInstruction::Halt => "halt",
    }
}

fn lane_address(base: u32, stride: u32, lane: usize) -> Result<u32, SimtError> {
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

fn issue_span(operations: u64, issue_width: u32) -> Result<u64, SimtError> {
    let width = u64::from(issue_width);
    operations
        .checked_add(width - 1)
        .map(|value| value / width)
        .ok_or(SimtError::CycleOverflow)
}

fn reserve_resource(
    resource_next: &mut BTreeMap<String, u64>,
    resource: &crate::spec::ResourceSpec,
    now: u64,
    operations: u64,
) -> Result<u64, SimtError> {
    let start = now.max(*resource_next.get(resource.name()).unwrap_or(&0));
    let next = start
        .checked_add(issue_span(operations, resource.issue_width())?)
        .ok_or(SimtError::CycleOverflow)?;
    resource_next.insert(resource.name().to_owned(), next);
    Ok(start)
}

fn checked_cycle(start: u64, span: u64, latency: Option<u64>) -> Result<u64, SimtError> {
    let latency = latency.ok_or(SimtError::UnknownLatency)?;
    start
        .checked_add(span)
        .and_then(|cycle| cycle.checked_add(latency))
        .ok_or(SimtError::CycleOverflow)
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
    Spec(#[from] SpecError),
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
        assert_eq!(timing.final_cycle, 32);
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
