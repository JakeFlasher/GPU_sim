//! Portable instruction evidence. This crate has no trace-reader or executor
//! dependency. Native adapters preserve vendor details alongside common fields.
//! Serialization order is lane-group program order, never silicon issue time.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub mod boundary;
pub mod certificate;
pub mod observation;
pub mod state;
pub use boundary::{
    BOUNDARY_SCHEMA_VERSION, BlockBoundaryState, BoundarySnapshot, WarpBoundaryState,
};
pub use certificate::{
    ExecutionEvidence, FidelityDecision, IntervalCertificate, IntervalProjection, ReifyReason,
    certify_interval,
};
pub use observation::{
    OBSERVATION_SCHEMA_VERSION, ObservationContract, ObservationContractError,
    ObservationGranularity,
};
pub use state::{
    AddressDomain, AsyncCopyBoundaryState, AsyncGroupBoundaryState, CanonicalLineKey, Completeness,
    InstructionCacheEntryState, InstructionCacheState, InstructionFrontEndBoundaryState,
    InstructionStreamLineState, InstructionStreamState, L1BoundaryState, L1CounterState,
    L1FillDueState, L1LineFillState, L1SectorFillState, L2AccessDisciplineState, L2AllocationState,
    L2BoundaryState, L2CounterState, L2GeometryState, L2RequestKey, L2RequestOwner, L2RequestState,
    L2SectorOwnership, L2SectorStatus, L2SetState, L2StateError, L2WayState,
    MemoryPipelineBoundaryState, PowerBoundaryState, PowerClockState, PowerDomainState,
    PowerDomainStatus, PowerModelStatus, PowerSampleState, PowerStateError, QueueBoundaryState,
    ScalarPipelineBoundaryState, TYPED_STATE_SCHEMA_VERSION, TypedContractKind,
    TypedCoreBoundaryState, TypedEventBoundaryRecord, TypedEventIdentity, TypedEventPayload,
    TypedObservationContract, TypedPendingEvent,
};

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Isa {
    NvidiaSm86,
    AmdGfx942,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "RawLaneMask")]
pub struct LaneMask {
    width: u8,
    bits: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawLaneMask {
    width: u8,
    bits: u64,
}
impl TryFrom<RawLaneMask> for LaneMask {
    type Error = SemanticError;
    fn try_from(raw: RawLaneMask) -> Result<Self, Self::Error> {
        Self::new(raw.width, raw.bits)
    }
}
impl LaneMask {
    pub fn new(width: u8, bits: u64) -> Result<Self, SemanticError> {
        if !(1..=64).contains(&width) || (width < 64 && bits >> width != 0) {
            return Err(SemanticError::LaneMask { width, bits });
        }
        Ok(Self { width, bits })
    }
    pub const fn width(self) -> u8 {
        self.width
    }
    pub const fn bits(self) -> u64 {
        self.bits
    }
    pub const fn count(self) -> u32 {
        self.bits.count_ones()
    }
    pub fn intersection(self, other: Self) -> Result<Self, SemanticError> {
        if self.width != other.width {
            return Err(SemanticError::MaskWidthMismatch);
        }
        Self::new(self.width, self.bits & other.bits)
    }
}

/// Native group and placement names are retained without interpreting them as
/// portable scheduling rules. `instruction_index` is local to this group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeInstruction {
    pub isa: Isa,
    pub device_id: i64,
    pub stream_id: i64,
    pub kernel_id: i32,
    pub block_launch_index: i64,
    pub warp_id: i32,
    pub instruction_index: u64,
    pub function_unique_id: i32,
    pub pc: u64,
    pub vpc: Option<u64>,
    pub sm_id: Option<i32>,
    pub warpid_sm: Option<i32>,
    /// Host instrumentation ordering, explicitly separate from modeled time.
    pub instrumentation_order: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LossReason {
    OpcodeUnavailable,
    OpcodeUnclassified,
    DynamicMemoryClassUnavailable,
    AddressPayloadUnavailable,
    MemoryDirectionUnavailable,
    AddressSpaceUnavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationKind {
    Unknown,
    Memory,
    Arithmetic,
    Matrix,
    BranchControl,
    Synchronization,
    RegisterMove,
    SpecialRegister,
    NoOp,
    Other,
    LaneExchange,
    CacheControl,
    SchedulingHint,
}

/// These categories describe evidence, not pipeline timing or functional
/// emulation support. A recognized matrix opcode alone does not license skip.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstructionClassification {
    pub opcode_text: String,
    pub operation: OperationKind,
    pub control: Option<ControlEvidence>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlEvidence {
    pub stall: u8,
    pub yield_flag: bool,
    pub write_barrier: u8,
    pub read_barrier: u8,
    pub wait_mask: u8,
    pub reuse: u8,
}
pub trait NativeInstructionClassifier {
    fn classify(&self, function_unique_id: i32, pc: u64) -> Option<InstructionClassification>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryFamily {
    Constant,
    Global,
    Shared,
    Local,
    Texture,
    Surface,
    Atomic,
    Tma,
    AsyncCopy,
    ClusterShared,
    TensorMemory,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryDirection {
    Load,
    Store,
    Atomic,
    AsyncCopy,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AddressSpace {
    Global,
    Shared,
    Local,
    Generic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaneAddress {
    pub lane: u8,
    pub address: u64,
}

/// Encoding details needed by the NVIDIA adapter to preserve optional-field
/// meaning. Decompressed addresses, not an opaque native blob, carry behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "isa", rename_all = "snake_case", deny_unknown_fields)]
pub enum NativeOperand {
    NvidiaSm86 {
        declared_width: i32,
        descriptor: u32,
        explicit_lane_mask: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryOperand {
    pub address_space: Option<AddressSpace>,
    pub allocation_id: Option<u32>,
    pub lane_mask: LaneMask,
    pub width_bytes: u32,
    /// One entry per effective lane, preserving alias multiplicity and offsets.
    pub lane_addresses: Vec<LaneAddress>,
    pub native: Option<NativeOperand>,
}
impl MemoryOperand {
    pub fn validate(&self) -> Result<(), SemanticError> {
        if self.width_bytes == 0 {
            return Err(SemanticError::InvalidWidth);
        }
        let mut seen = 0u64;
        for address in &self.lane_addresses {
            if address.lane >= self.lane_mask.width() {
                return Err(SemanticError::AddressLane);
            }
            let bit = 1u64 << address.lane;
            if seen & bit != 0 || self.lane_mask.bits() & bit == 0 {
                return Err(SemanticError::AddressLane);
            }
            seen |= bit;
            address
                .address
                .checked_add(u64::from(self.width_bytes) - 1)
                .ok_or(SemanticError::AddressOverflow)?;
        }
        if seen != self.lane_mask.bits() {
            return Err(SemanticError::AddressLane);
        }
        Ok(())
    }
    /// A locality projection only. This is not an atomic transaction count or
    /// a shared-memory bank-conflict projection.
    pub fn unique_sectors(&self, sector_bytes: u32) -> Result<Vec<u64>, SemanticError> {
        self.validate()?;
        if sector_bytes == 0 {
            return Err(SemanticError::InvalidSectorWidth);
        }
        let bytes = u64::from(sector_bytes);
        let mut intervals = Vec::with_capacity(self.lane_addresses.len());
        for lane in &self.lane_addresses {
            intervals.push((
                lane.address / bytes,
                (lane.address + u64::from(self.width_bytes) - 1) / bytes,
            ));
        }
        intervals.sort_unstable();
        let mut merged: Vec<(u64, u64)> = Vec::new();
        for (start, end) in intervals {
            if let Some(last) = merged.last_mut() {
                if start <= last.1.saturating_add(1) {
                    last.1 = last.1.max(end);
                    continue;
                }
            }
            merged.push((start, end));
        }
        let count = merged.iter().try_fold(0u64, |sum, (start, end)| {
            sum.checked_add(end - start + 1)
                .ok_or(SemanticError::CountOverflow)
        })?;
        if count > 1 << 20 {
            return Err(SemanticError::SectorExpansionLimit);
        }
        let mut sectors = Vec::with_capacity(count as usize);
        for (start, end) in merged {
            sectors.extend(start..=end);
        }
        Ok(sectors)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryEvent {
    pub family: MemoryFamily,
    pub direction: Option<MemoryDirection>,
    /// Whether the dynamic producer supplied a memory-class tag. Static-only
    /// constant-cache events deliberately retain `false`.
    pub dynamic_class_present: bool,
    /// Native cache semantics are never assumed to mean the same on AMD.
    pub native_cache_op: Option<i32>,
    pub declared_width: Option<i32>,
    pub operands: Vec<MemoryOperand>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticEvent {
    /// Deterministic serialization ordinal; it does not order different warps
    /// in hardware and cannot be used as a simulated timestamp.
    pub ordinal: u64,
    pub native: NativeInstruction,
    pub active_lanes: LaneMask,
    pub predicate_lanes: LaneMask,
    pub operation: OperationKind,
    pub opcode_text: Option<String>,
    pub control: Option<ControlEvidence>,
    pub memory: Option<MemoryEvent>,
    pub losses: Vec<LossReason>,
}
impl SemanticEvent {
    pub fn validate(&self) -> Result<(), SemanticError> {
        let effective = self.active_lanes.intersection(self.predicate_lanes)?;
        let expected_width = match self.native.isa {
            Isa::NvidiaSm86 => 32,
            Isa::AmdGfx942 => 64,
        };
        if effective.width() != expected_width {
            return Err(SemanticError::MaskWidthMismatch);
        }
        if let Some(memory) = &self.memory {
            if self.operation != OperationKind::Memory {
                return Err(SemanticError::OperationMismatch);
            }
            for operand in &memory.operands {
                operand.validate()?;
                if operand.lane_mask.width() != effective.width()
                    || operand.lane_mask.bits() & !effective.bits() != 0
                {
                    return Err(SemanticError::AddressLane);
                }
            }
        } else if self.operation == OperationKind::Memory {
            return Err(SemanticError::OperationMismatch);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticCounts {
    pub instruction_events: u64,
    pub memory_events: u64,
    pub unknown_instruction_events: u64,
    pub memory_operands: u64,
    pub active_memory_lanes: u64,
    pub lane_addresses: u64,
    pub operand_unique_sectors_32b: u64,
    pub operation_counts: BTreeMap<OperationKind, u64>,
    pub loss_counts: BTreeMap<LossReason, u64>,
}
fn add(target: &mut u64, value: u64) -> Result<(), SemanticError> {
    *target = target
        .checked_add(value)
        .ok_or(SemanticError::CountOverflow)?;
    Ok(())
}
impl SemanticCounts {
    pub fn observe(&mut self, event: &SemanticEvent) -> Result<(), SemanticError> {
        event.validate()?;
        add(&mut self.instruction_events, 1)?;
        add(self.operation_counts.entry(event.operation).or_default(), 1)?;
        if event.operation == OperationKind::Unknown {
            add(&mut self.unknown_instruction_events, 1)?;
        }
        for loss in event.losses.iter().copied().collect::<BTreeSet<_>>() {
            add(self.loss_counts.entry(loss).or_default(), 1)?;
        }
        if let Some(memory) = &event.memory {
            add(&mut self.memory_events, 1)?;
            add(
                &mut self.active_memory_lanes,
                u64::from(
                    event
                        .active_lanes
                        .intersection(event.predicate_lanes)?
                        .count(),
                ),
            )?;
            add(&mut self.memory_operands, memory.operands.len() as u64)?;
            for operand in &memory.operands {
                add(
                    &mut self.lane_addresses,
                    operand.lane_addresses.len() as u64,
                )?;
                add(
                    &mut self.operand_unique_sectors_32b,
                    operand.unique_sectors(32)?.len() as u64,
                )?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SemanticError {
    #[error("invalid lane mask width={width} bits={bits:#x}")]
    LaneMask { width: u8, bits: u64 },
    #[error("lane-mask widths disagree with each other or the source ISA")]
    MaskWidthMismatch,
    #[error("operand lane addresses must cover its effective mask exactly once")]
    AddressLane,
    #[error("memory width must be positive")]
    InvalidWidth,
    #[error("memory byte span overflows the address domain")]
    AddressOverflow,
    #[error("sector width must be positive")]
    InvalidSectorWidth,
    #[error("sector footprint exceeds the bounded expansion")]
    SectorExpansionLimit,
    #[error("semantic count overflow")]
    CountOverflow,
    #[error("operation and memory payload disagree")]
    OperationMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wave64_high_lane_survives_json() {
        let mask = LaneMask::new(64, 1u64 << 63).unwrap();
        let json = serde_json::to_string(&mask).unwrap();
        assert_eq!(serde_json::from_str::<LaneMask>(&json).unwrap(), mask);
        assert!(serde_json::from_str::<LaneMask>(r#"{"width":32,"bits":4294967296}"#).is_err());
    }
    #[test]
    fn byte_addresses_keep_atomic_aliases_and_cross_sector_offsets() {
        let operand = MemoryOperand {
            address_space: Some(AddressSpace::Global),
            allocation_id: Some(7),
            lane_mask: LaneMask::new(32, 7).unwrap(),
            width_bytes: 4,
            lane_addresses: vec![
                LaneAddress {
                    lane: 0,
                    address: 31,
                },
                LaneAddress {
                    lane: 1,
                    address: 31,
                },
                LaneAddress {
                    lane: 2,
                    address: 64,
                },
            ],
            native: None,
        };
        assert_eq!(operand.unique_sectors(32).unwrap(), vec![0, 1, 2]);
        let decoded: MemoryOperand =
            serde_json::from_str(&serde_json::to_string(&operand).unwrap()).unwrap();
        assert_eq!(decoded, operand);
        let mut invalid = operand;
        invalid.lane_addresses[2].lane = 1;
        assert!(invalid.validate().is_err());
    }
}
