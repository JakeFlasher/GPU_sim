//! Portable instruction evidence. This crate has no trace-reader or executor
//! dependency. Native adapters preserve vendor details alongside common fields.
//! Serialization order is lane-group program order, never silicon issue time.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub mod boundary;
pub mod certificate;
pub mod dram;
pub mod latent;
pub mod observation;
pub mod state;
pub mod transform;
pub use boundary::{
    BOUNDARY_SCHEMA_VERSION, BlockBoundaryState, BoundarySnapshot, WarpBoundaryState,
};
pub use certificate::{
    ExecutionEvidence, FidelityDecision, IntervalCertificate, IntervalProjection, ReifyReason,
    certify_interval,
};
pub use dram::{DramBoundaryState, DramStateError};
pub use latent::{
    CacheCandidateSummary, CacheWitnessDomains, CandidateDomain, ClockDomain,
    CompletionRequirement, CompletionRequirements, EndpointDescriptor, EventKey, EvidenceKind,
    FrontierBound, FrontierVector, LATENT_SCHEMA_VERSION, LatentFactors, LatentModelError,
    LatentSummary, LatentWorld, LatentWorldSet, MechanismEnvelope, MustMay, NcuObserverProjection,
    RESOURCE_DEMAND_SCHEMA_VERSION, ResourceDemandEnvelope, ResourceDemandInterval,
    ResourceServiceProjection, ServiceComposition, ServiceCycleInterval, ServiceOverlapMode,
    ServiceQueueProjection, ServiceRate, ServiceRateProfile, ServiceStageRole,
};
pub use observation::{
    OBSERVATION_SCHEMA_VERSION, ObservationContract, ObservationContractError,
    ObservationGranularity,
};
pub use state::{
    AddressDomain, AsyncCopyBoundaryState, AsyncCopyBypassObservationState,
    AsyncCopyCohortBoundaryState, AsyncCopyStateError, AsyncGroupBoundaryState, BoundaryStateError,
    CanonicalLineKey, Completeness, CoreStateError, EventSequenceError, FrontEndStateError,
    InstructionCacheEntryState, InstructionCacheState, InstructionFrontEndBoundaryState,
    InstructionStreamLineState, InstructionStreamState, L1BoundaryState, L1CounterState,
    L1FillDueState, L1LineFillState, L1SectorFillState, L1StateError, L2AccessDisciplineState,
    L2AllocationState, L2BoundaryState, L2CounterState, L2GeometryState, L2RequestKey,
    L2RequestOwner, L2RequestState, L2SectorOwnership, L2SectorStatus, L2SetState, L2StateError,
    L2WayState, MemoryPipelineBoundaryState, PowerBoundaryState, PowerClockState, PowerDomainState,
    PowerDomainStatus, PowerModelStatus, PowerSampleState, PowerStateError, QueueBoundaryState,
    QueueStateError, ScalarPipelineBoundaryState, TYPED_STATE_SCHEMA_VERSION,
    TypedAdmissionWitness, TypedContractKind, TypedCoreBoundaryState, TypedEventBoundaryRecord,
    TypedEventIdentity, TypedEventPayload, TypedFillIdentity, TypedL2Source, TypedMemoryOperation,
    TypedMemoryOrigin, TypedMemoryRequest, TypedObservationContract, TypedPendingEvent,
    TypedResponseIdentity, TypedResponseKind,
};
pub use transform::{
    ExternalInputFrontier, IdentityBoundaryTransformer, NativeBoundaryRefinementDecision,
    NativeBoundaryReifyReason, NativeBoundarySide, ObservationDelta, RegionContractError,
    StateTransformer, SummaryProof, TRANSFORM_SCHEMA_VERSION, TransformCheckError, TransformError,
    TransformInput, TransformInterval, TransformIntervalError, TransformReifyReason,
    TransformerDecision, TransformerOutput, TypedEventTransition, TypedRegionContract, check_proof,
    compare_native_boundary_projection, compare_native_region_contract,
};

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Isa {
    NvidiaSm86,
    AmdGfx942,
    CustomSimtU32,
    RiscV32I,
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
    /// Optional width for extensible/custom ISA variants. Built-in vendor
    /// variants retain their architectural width when this is absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub simt_width: Option<u8>,
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

/// Token effects for custom asynchronous operations; distinct from SASS
/// scheduling-control bits and from a hardware timing observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum AsyncEffect {
    Submit { group: u8 },
    Wait { group: u8 },
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
        let mut previous_lane = None;
        for address in &self.lane_addresses {
            if address.lane >= self.lane_mask.width() {
                return Err(SemanticError::AddressLane);
            }
            if previous_lane.is_some_and(|previous| address.lane <= previous) {
                return Err(SemanticError::AddressLaneOrder);
            }
            previous_lane = Some(address.lane);
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
            if let Some(last) = merged.last_mut()
                && start <= last.1.saturating_add(1)
            {
                last.1 = last.1.max(end);
                continue;
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub async_effect: Option<AsyncEffect>,
    pub memory: Option<MemoryEvent>,
    pub losses: Vec<LossReason>,
}

/// Validate submit/wait pairing for a sequence that declares custom async
/// effects. Per-event validation alone cannot detect a missing wait or a
/// duplicated group; this stateful check is the stronger boundary contract.
/// Tokens are local to (ISA, device, stream, launch, block, warp), never global
/// to an interleaved trace. This checks pairing, not functional memory safety.
pub fn validate_async_effects(events: &[SemanticEvent]) -> Result<(), SemanticError> {
    let mut pending = BTreeMap::new();
    for event in events {
        // Pairing is a stateful extension of the per-event contract, not a
        // replacement for it. Validate every record here as well so callers
        // cannot obtain a successful async proof from malformed events that
        // happen not to carry an effect.
        event.validate()?;
        if let Some(effect) = event.async_effect {
            let native = event.native;
            let group = match effect {
                AsyncEffect::Submit { group } | AsyncEffect::Wait { group } => group,
            };
            let key = (
                native.isa,
                native.device_id,
                native.stream_id,
                native.kernel_id,
                native.block_launch_index,
                native.warp_id,
                group,
            );
            match effect {
                AsyncEffect::Submit { group } => {
                    if pending.insert(key, event.active_lanes).is_some() {
                        return Err(SemanticError::AsyncGroupReuse { group });
                    }
                }
                AsyncEffect::Wait { group } => match pending.remove(&key) {
                    None => return Err(SemanticError::AsyncGroupMissing { group }),
                    Some(mask) if mask != event.active_lanes => {
                        return Err(SemanticError::MaskWidthMismatch);
                    }
                    Some(_) => {}
                },
            }
        }
    }
    pending
        .into_keys()
        .next()
        .map_or(Ok(()), |(_, _, _, _, _, _, group)| {
            Err(SemanticError::AsyncGroupUnfinished { group })
        })
}

impl SemanticEvent {
    pub fn validate(&self) -> Result<(), SemanticError> {
        if self.losses.windows(2).any(|window| window[1] <= window[0]) {
            return Err(SemanticError::LossesNotCanonical);
        }
        let effective = self.active_lanes.intersection(self.predicate_lanes)?;
        let expected_width = match self.native.isa {
            Isa::NvidiaSm86 => 32,
            Isa::AmdGfx942 => 64,
            Isa::CustomSimtU32 | Isa::RiscV32I => self
                .native
                .simt_width
                .ok_or(SemanticError::MaskWidthMismatch)?,
        };
        if !(1..=64).contains(&expected_width) {
            return Err(SemanticError::MaskWidthMismatch);
        }
        if effective.width() != expected_width {
            return Err(SemanticError::MaskWidthMismatch);
        }
        if self
            .native
            .simt_width
            .is_some_and(|width| width != expected_width)
        {
            return Err(SemanticError::MaskWidthMismatch);
        }
        if let Some(effect) = self.async_effect {
            // The current custom ISA defines whole-group tokens only. A
            // predicated submit/wait would need per-lane ownership semantics.
            if effective.count() != u32::from(expected_width) {
                return Err(SemanticError::OperationMismatch);
            }
            let valid = match effect {
                AsyncEffect::Submit { .. } => self.memory.as_ref().is_some_and(|memory| {
                    self.operation == OperationKind::Memory
                        && memory.family == MemoryFamily::Global
                        && memory.direction == Some(MemoryDirection::Store)
                }),
                AsyncEffect::Wait { .. } => {
                    self.operation == OperationKind::Synchronization && self.memory.is_none()
                }
            };
            if self.native.isa != Isa::CustomSimtU32 || !valid {
                return Err(SemanticError::OperationMismatch);
            }
        }
        if let Some(memory) = &self.memory {
            if self.operation != OperationKind::Memory {
                return Err(SemanticError::OperationMismatch);
            }
            if memory.dynamic_class_present != (memory.family != MemoryFamily::Constant) {
                return Err(SemanticError::MemoryClassPresenceMismatch);
            }
            if memory.family == MemoryFamily::Constant
                && memory.direction != Some(MemoryDirection::Load)
            {
                return Err(SemanticError::ConstantDirectionMismatch);
            }
            if memory.family == MemoryFamily::Constant
                && !self
                    .losses
                    .contains(&LossReason::DynamicMemoryClassUnavailable)
            {
                return Err(SemanticError::MissingLossReason {
                    reason: LossReason::DynamicMemoryClassUnavailable,
                });
            }
            if memory.dynamic_class_present
                && memory.direction.is_none()
                && !self
                    .losses
                    .contains(&LossReason::MemoryDirectionUnavailable)
            {
                return Err(SemanticError::MissingLossReason {
                    reason: LossReason::MemoryDirectionUnavailable,
                });
            }
            if memory.dynamic_class_present
                && effective.bits() != 0
                && memory.operands.is_empty()
                && !self.losses.contains(&LossReason::AddressPayloadUnavailable)
            {
                return Err(SemanticError::MissingLossReason {
                    reason: LossReason::AddressPayloadUnavailable,
                });
            }
            if memory.dynamic_class_present
                && memory
                    .operands
                    .iter()
                    .any(|operand| operand.address_space.is_none())
                && !self.losses.contains(&LossReason::AddressSpaceUnavailable)
            {
                return Err(SemanticError::MissingLossReason {
                    reason: LossReason::AddressSpaceUnavailable,
                });
            }
            if memory.declared_width.is_some_and(|width| width <= 0) {
                return Err(SemanticError::InvalidWidth);
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
    #[error("operand lane addresses are not in canonical ascending lane order")]
    AddressLaneOrder,
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
    #[error("dynamic memory-class presence disagrees with the memory family")]
    MemoryClassPresenceMismatch,
    #[error("static constant-cache memory must carry a load direction")]
    ConstantDirectionMismatch,
    #[error("semantic event is missing required loss reason {reason:?}")]
    MissingLossReason { reason: LossReason },
    #[error("semantic event loss reasons are not strictly ordered and unique")]
    LossesNotCanonical,
    #[error("async group {group} is submitted more than once")]
    AsyncGroupReuse { group: u8 },
    #[error("async group {group} is waited without a submit")]
    AsyncGroupMissing { group: u8 },
    #[error("async group {group} remains unfinished")]
    AsyncGroupUnfinished { group: u8 },
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wave64_high_lane_survives_json() {
        let mask = LaneMask::new(64, 1u64 << 63).expect("test fixture should be valid");
        let json = serde_json::to_string(&mask).expect("test fixture should be valid");
        assert_eq!(
            serde_json::from_str::<LaneMask>(&json).expect("test fixture should be valid"),
            mask
        );
        assert!(serde_json::from_str::<LaneMask>(r#"{"width":32,"bits":4294967296}"#).is_err());
    }
    #[test]
    fn byte_addresses_keep_atomic_aliases_and_cross_sector_offsets() {
        let operand = MemoryOperand {
            address_space: Some(AddressSpace::Global),
            allocation_id: Some(7),
            lane_mask: LaneMask::new(32, 7).expect("test fixture should be valid"),
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
        assert_eq!(
            operand
                .unique_sectors(32)
                .expect("test fixture should be valid"),
            vec![0, 1, 2]
        );
        let decoded: MemoryOperand = serde_json::from_str(
            &serde_json::to_string(&operand).expect("test fixture should be valid"),
        )
        .expect("test fixture should be valid");
        assert_eq!(decoded, operand);
        let mut invalid = operand.clone();
        invalid.lane_addresses[2].lane = 1;
        assert!(invalid.validate().is_err());
        let mut unsorted = operand;
        unsorted.lane_addresses.swap(0, 1);
        assert!(matches!(
            unsorted.validate(),
            Err(SemanticError::AddressLaneOrder)
        ));
    }

    #[test]
    fn memory_class_presence_matches_constant_static_only_contract() {
        let native = NativeInstruction {
            isa: Isa::NvidiaSm86,
            simt_width: None,
            device_id: 0,
            stream_id: 0,
            kernel_id: 1,
            block_launch_index: 0,
            warp_id: 0,
            instruction_index: 0,
            function_unique_id: 1,
            pc: 0,
            vpc: None,
            sm_id: Some(0),
            warpid_sm: Some(0),
            instrumentation_order: None,
        };
        let mask = LaneMask::new(32, 1).expect("test fixture should be valid");
        let mut event = SemanticEvent {
            ordinal: 0,
            native,
            active_lanes: mask,
            predicate_lanes: mask,
            operation: OperationKind::Memory,
            opcode_text: Some("LDG.E".to_owned()),
            control: None,
            async_effect: None,
            memory: Some(MemoryEvent {
                family: MemoryFamily::Global,
                direction: Some(MemoryDirection::Load),
                dynamic_class_present: false,
                native_cache_op: None,
                declared_width: Some(4),
                operands: Vec::new(),
            }),
            losses: Vec::new(),
        };
        assert!(matches!(
            event.validate(),
            Err(SemanticError::MemoryClassPresenceMismatch)
        ));
        {
            let memory = event.memory.as_mut().expect("test fixture should be valid");
            memory.family = MemoryFamily::Constant;
            memory.direction = Some(MemoryDirection::Load);
        }
        event.losses.push(LossReason::DynamicMemoryClassUnavailable);
        assert!(event.validate().is_ok());
        event.losses = vec![
            LossReason::DynamicMemoryClassUnavailable,
            LossReason::OpcodeUnavailable,
        ];
        assert!(matches!(
            event.validate(),
            Err(SemanticError::LossesNotCanonical)
        ));
        event.losses = vec![
            LossReason::DynamicMemoryClassUnavailable,
            LossReason::DynamicMemoryClassUnavailable,
        ];
        assert!(matches!(
            event.validate(),
            Err(SemanticError::LossesNotCanonical)
        ));
        event.losses = vec![LossReason::DynamicMemoryClassUnavailable];
        event
            .memory
            .as_mut()
            .expect("test fixture should be valid")
            .direction = Some(MemoryDirection::Store);
        assert!(matches!(
            event.validate(),
            Err(SemanticError::ConstantDirectionMismatch)
        ));
        event
            .memory
            .as_mut()
            .expect("test fixture should be valid")
            .direction = Some(MemoryDirection::Load);
        event.losses.clear();
        assert!(matches!(
            event.validate(),
            Err(SemanticError::MissingLossReason {
                reason: LossReason::DynamicMemoryClassUnavailable
            })
        ));
        event.losses.push(LossReason::DynamicMemoryClassUnavailable);
        event
            .memory
            .as_mut()
            .expect("test fixture should be valid")
            .dynamic_class_present = true;
        assert!(matches!(
            event.validate(),
            Err(SemanticError::MemoryClassPresenceMismatch)
        ));
    }

    #[test]
    fn dynamic_memory_gaps_require_explicit_loss_reasons() {
        let native = NativeInstruction {
            isa: Isa::NvidiaSm86,
            simt_width: None,
            device_id: 0,
            stream_id: 0,
            kernel_id: 1,
            block_launch_index: 0,
            warp_id: 0,
            instruction_index: 0,
            function_unique_id: 1,
            pc: 0,
            vpc: None,
            sm_id: Some(0),
            warpid_sm: Some(0),
            instrumentation_order: None,
        };
        let mask = LaneMask::new(32, 1).expect("test fixture should be valid");
        let mut event = SemanticEvent {
            ordinal: 0,
            native,
            active_lanes: mask,
            predicate_lanes: mask,
            operation: OperationKind::Memory,
            opcode_text: Some("LDG.E".to_owned()),
            control: None,
            async_effect: None,
            memory: Some(MemoryEvent {
                family: MemoryFamily::Global,
                direction: None,
                dynamic_class_present: true,
                native_cache_op: None,
                declared_width: Some(4),
                operands: Vec::new(),
            }),
            losses: Vec::new(),
        };
        assert!(matches!(
            event.validate(),
            Err(SemanticError::MissingLossReason {
                reason: LossReason::MemoryDirectionUnavailable
            })
        ));
        event.losses.push(LossReason::MemoryDirectionUnavailable);
        assert!(matches!(
            event.validate(),
            Err(SemanticError::MissingLossReason {
                reason: LossReason::AddressPayloadUnavailable
            })
        ));
        event.losses = vec![
            LossReason::AddressPayloadUnavailable,
            LossReason::MemoryDirectionUnavailable,
        ];
        assert!(event.validate().is_ok());
        event
            .memory
            .as_mut()
            .expect("test fixture should be valid")
            .operands = vec![MemoryOperand {
            address_space: None,
            allocation_id: None,
            lane_mask: mask,
            width_bytes: 4,
            lane_addresses: vec![LaneAddress {
                lane: 0,
                address: 0,
            }],
            native: None,
        }];
        event.losses.clear();
        event
            .memory
            .as_mut()
            .expect("test fixture should be valid")
            .direction = Some(MemoryDirection::Load);
        assert!(matches!(
            event.validate(),
            Err(SemanticError::MissingLossReason {
                reason: LossReason::AddressSpaceUnavailable
            })
        ));
        event.losses.push(LossReason::AddressSpaceUnavailable);
        assert!(event.validate().is_ok());
        event
            .memory
            .as_mut()
            .expect("test fixture should be valid")
            .declared_width = Some(0);
        assert!(matches!(event.validate(), Err(SemanticError::InvalidWidth)));
    }

    #[test]
    fn custom_simt_width_is_carried_by_native_identity() {
        let mask = LaneMask::new(4, 0b1111).expect("mask should be valid");
        let event = SemanticEvent {
            ordinal: 0,
            native: NativeInstruction {
                isa: Isa::CustomSimtU32,
                simt_width: Some(4),
                device_id: 0,
                stream_id: 0,
                kernel_id: 0,
                block_launch_index: 0,
                warp_id: 0,
                instruction_index: 0,
                function_unique_id: 0,
                pc: 0,
                vpc: None,
                sm_id: None,
                warpid_sm: None,
                instrumentation_order: None,
            },
            active_lanes: mask,
            predicate_lanes: mask,
            operation: OperationKind::Arithmetic,
            opcode_text: Some("add_imm".to_owned()),
            control: None,
            async_effect: None,
            memory: None,
            losses: Vec::new(),
        };
        assert!(event.validate().is_ok());
    }

    #[test]
    fn async_effect_sequence_rejects_duplicate_missing_and_unfinished_groups() {
        fn event(effect: AsyncEffect) -> SemanticEvent {
            let mask = LaneMask::new(8, 0xff).expect("mask should be valid");
            SemanticEvent {
                ordinal: 0,
                native: NativeInstruction {
                    isa: Isa::CustomSimtU32,
                    simt_width: Some(8),
                    device_id: 0,
                    stream_id: 0,
                    kernel_id: 0,
                    block_launch_index: 0,
                    warp_id: 0,
                    instruction_index: 0,
                    function_unique_id: 0,
                    pc: 0,
                    vpc: None,
                    sm_id: None,
                    warpid_sm: None,
                    instrumentation_order: None,
                },
                active_lanes: mask,
                predicate_lanes: mask,
                operation: if matches!(effect, AsyncEffect::Wait { .. }) {
                    OperationKind::Synchronization
                } else {
                    OperationKind::Memory
                },
                opcode_text: None,
                control: None,
                async_effect: Some(effect),
                memory: if matches!(effect, AsyncEffect::Submit { .. }) {
                    Some(MemoryEvent {
                        family: MemoryFamily::Global,
                        direction: Some(MemoryDirection::Store),
                        dynamic_class_present: true,
                        native_cache_op: None,
                        declared_width: Some(4),
                        operands: vec![MemoryOperand {
                            address_space: Some(AddressSpace::Global),
                            allocation_id: None,
                            lane_mask: mask,
                            width_bytes: 4,
                            lane_addresses: (0..8)
                                .map(|lane| LaneAddress {
                                    lane,
                                    address: u64::from(lane) * 4,
                                })
                                .collect(),
                            native: None,
                        }],
                    })
                } else {
                    None
                },
                losses: Vec::new(),
            }
        }
        let submit = event(AsyncEffect::Submit { group: 1 });
        let wait = event(AsyncEffect::Wait { group: 1 });
        assert!(validate_async_effects(&[submit.clone(), wait.clone()]).is_ok());
        let mut partial_submit = submit.clone();
        partial_submit.active_lanes = LaneMask::new(8, 0x01).expect("mask should be valid");
        assert!(matches!(
            partial_submit.validate(),
            Err(SemanticError::OperationMismatch)
        ));
        assert!(matches!(
            validate_async_effects(&[submit.clone(), submit.clone()]),
            Err(SemanticError::AsyncGroupReuse { group: 1 })
        ));
        assert!(matches!(
            validate_async_effects(std::slice::from_ref(&wait)),
            Err(SemanticError::AsyncGroupMissing { group: 1 })
        ));
        assert!(matches!(
            validate_async_effects(std::slice::from_ref(&submit)),
            Err(SemanticError::AsyncGroupUnfinished { group: 1 })
        ));

        // Group identifiers are local to a warp/launch identity. Interleaved
        // warps may reuse the same token without aliasing each other's wait.
        let mut other_submit = submit.clone();
        other_submit.native.warp_id = 2;
        let mut other_wait = wait.clone();
        other_wait.native.warp_id = 2;
        assert!(validate_async_effects(&[submit, other_submit, other_wait, wait]).is_ok());
    }
}
