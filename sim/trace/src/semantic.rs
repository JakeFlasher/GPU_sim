//! SM86 NVBit adapter for portable instruction evidence. Byte addresses and
//! alias multiplicity survive; sector footprints are optional projections.
//! The visitor holds at most one decoded block and one semantic instruction.

use crate::reader::memory_access;
use crate::{TraceDir, TraceError, pb};
use serde::Serialize;
use sim_semantics::{
    AddressSpace, InstructionClassification, Isa, LaneAddress, LaneMask, LossReason,
    MemoryDirection, MemoryEvent, MemoryFamily, MemoryOperand, NativeInstruction,
    NativeInstructionClassifier, NativeOperand, OperationKind, SCHEMA_VERSION, SemanticCounts,
    SemanticError, SemanticEvent,
};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum AdapterError {
    #[error(transparent)]
    Trace(#[from] TraceError),
    #[error(transparent)]
    Semantic(#[from] SemanticError),
    #[error("NVBit semantic adapter requires SM86, found {0}")]
    Architecture(i32),
    #[error("missing required instruction field {0}")]
    MissingField(&'static str),
    #[error("invalid native field {0}")]
    NativeField(&'static str),
    #[error("semantic instruction count {actual} differs from source {expected}")]
    Count { expected: u64, actual: u64 },
    #[error("semantic event consumer failed: {0}")]
    Consumer(String),
}

#[derive(Debug, Clone, Serialize)]
pub struct SemanticTraceSummary {
    pub schema_version: u32,
    pub source_isa: Isa,
    pub launch_count: u64,
    pub source_instruction_count: u64,
    pub order_semantics: &'static str,
    pub not_observed: [&'static str; 3],
    pub dynamic_memory_class_gaps: BTreeMap<String, u64>,
    pub unclassified_opcodes: BTreeMap<String, u64>,
    pub counts: SemanticCounts,
}

fn native_family(family: pb::MemoryOpClass) -> Result<MemoryFamily, AdapterError> {
    Ok(match family {
        pb::MemoryOpClass::Global => MemoryFamily::Global,
        pb::MemoryOpClass::Shared => MemoryFamily::Shared,
        pb::MemoryOpClass::Local => MemoryFamily::Local,
        pb::MemoryOpClass::Texture => MemoryFamily::Texture,
        pb::MemoryOpClass::Surface => MemoryFamily::Surface,
        pb::MemoryOpClass::Atomic => MemoryFamily::Atomic,
        pb::MemoryOpClass::Tma => MemoryFamily::Tma,
        pb::MemoryOpClass::AsyncCopy => MemoryFamily::AsyncCopy,
        pb::MemoryOpClass::ClusterDsmem => MemoryFamily::ClusterShared,
        pb::MemoryOpClass::Tmem => MemoryFamily::TensorMemory,
        pb::MemoryOpClass::Unspecified => return Err(AdapterError::NativeField("memory class")),
    })
}
fn wire_family(family: MemoryFamily) -> Result<pb::MemoryOpClass, AdapterError> {
    Ok(match family {
        MemoryFamily::Constant => return Err(AdapterError::NativeField("constant-only event")),
        MemoryFamily::Global => pb::MemoryOpClass::Global,
        MemoryFamily::Shared => pb::MemoryOpClass::Shared,
        MemoryFamily::Local => pb::MemoryOpClass::Local,
        MemoryFamily::Texture => pb::MemoryOpClass::Texture,
        MemoryFamily::Surface => pb::MemoryOpClass::Surface,
        MemoryFamily::Atomic => pb::MemoryOpClass::Atomic,
        MemoryFamily::Tma => pb::MemoryOpClass::Tma,
        MemoryFamily::AsyncCopy => pb::MemoryOpClass::AsyncCopy,
        MemoryFamily::ClusterShared => pb::MemoryOpClass::ClusterDsmem,
        MemoryFamily::TensorMemory => pb::MemoryOpClass::Tmem,
    })
}
fn native_space(raw: i32) -> Result<AddressSpace, AdapterError> {
    Ok(
        match pb::MemorySpace::try_from(raw)
            .map_err(|_| AdapterError::NativeField("address space"))?
        {
            pb::MemorySpace::Global => AddressSpace::Global,
            pb::MemorySpace::Shared => AddressSpace::Shared,
            pb::MemorySpace::Local => AddressSpace::Local,
            pb::MemorySpace::Generic => AddressSpace::Generic,
            pb::MemorySpace::Unspecified => {
                return Err(AdapterError::NativeField("unspecified address space"));
            }
        },
    )
}
fn wire_space(space: AddressSpace) -> pb::MemorySpace {
    match space {
        AddressSpace::Global => pb::MemorySpace::Global,
        AddressSpace::Shared => pb::MemorySpace::Shared,
        AddressSpace::Local => pb::MemorySpace::Local,
        AddressSpace::Generic => pb::MemorySpace::Generic,
    }
}
fn native_direction(raw: i32) -> Result<MemoryDirection, AdapterError> {
    Ok(
        match pb::MemoryOpAccessKind::try_from(raw)
            .map_err(|_| AdapterError::NativeField("memory direction"))?
        {
            pb::MemoryOpAccessKind::Load => MemoryDirection::Load,
            pb::MemoryOpAccessKind::Store => MemoryDirection::Store,
            pb::MemoryOpAccessKind::Atomic => MemoryDirection::Atomic,
            pb::MemoryOpAccessKind::AsyncCopy => MemoryDirection::AsyncCopy,
            pb::MemoryOpAccessKind::Unspecified => {
                return Err(AdapterError::NativeField("unspecified memory direction"));
            }
        },
    )
}
fn wire_direction(direction: MemoryDirection) -> pb::MemoryOpAccessKind {
    match direction {
        MemoryDirection::Load => pb::MemoryOpAccessKind::Load,
        MemoryDirection::Store => pb::MemoryOpAccessKind::Store,
        MemoryDirection::Atomic => pb::MemoryOpAccessKind::Atomic,
        MemoryDirection::AsyncCopy => pb::MemoryOpAccessKind::AsyncCopy,
    }
}

pub fn sm86_opcode_head(text: &str) -> &str {
    let mut tokens = text.split_ascii_whitespace();
    let first = tokens.next().unwrap_or_default();
    let opcode = if first.starts_with('@') {
        tokens.next().unwrap_or_default()
    } else {
        first
    };
    opcode.split('.').next().unwrap_or_default()
}

pub fn from_instruction(
    ordinal: u64,
    native: NativeInstruction,
    instruction: &pb::Instruction,
    classification: Option<InstructionClassification>,
) -> Result<SemanticEvent, AdapterError> {
    if native.isa != Isa::NvidiaSm86 {
        return Err(AdapterError::NativeField("ISA"));
    }
    let active = instruction
        .active_mask
        .ok_or(AdapterError::MissingField("active_mask"))?;
    let predicate = instruction
        .predicate_mask
        .ok_or(AdapterError::MissingField("predicate_mask"))?;
    let effective = active & predicate;
    let mut losses = Vec::new();
    let opcode_text = classification
        .as_ref()
        .map(|value| value.opcode_text.clone());
    let control = classification.as_ref().and_then(|value| value.control);
    let mut operation = classification
        .as_ref()
        .map_or(OperationKind::Unknown, |value| value.operation);
    if classification.is_none() {
        losses.push(LossReason::OpcodeUnavailable);
    } else if operation == OperationKind::Unknown {
        losses.push(LossReason::OpcodeUnclassified);
    }
    let memory = if let Some(raw) = instruction.memory_op_class {
        operation = OperationKind::Memory;
        let family = native_family(
            pb::MemoryOpClass::try_from(raw)
                .map_err(|_| AdapterError::NativeField("memory class"))?,
        )?;
        let access = memory_access(instruction, || "semantic adapter".to_owned())?
            .ok_or(AdapterError::MissingField("memory metadata"))?;
        let direction = instruction
            .memory_op_access_kind
            .map(native_direction)
            .transpose()?;
        if direction.is_none() {
            losses.push(LossReason::MemoryDirectionUnavailable);
        }
        if let Some(raw) = instruction.instruction_cache_op {
            pb::InstructionCacheOp::try_from(raw)
                .map_err(|_| AdapterError::NativeField("cache operator"))?;
        }
        let mut operands = Vec::with_capacity(instruction.addresses.len());
        for (index, source) in instruction.addresses.iter().enumerate() {
            let width = if source.data_width > 0 {
                source.data_width
            } else {
                instruction.data_width.unwrap_or(0)
            };
            let width_bytes = u32::try_from(width)
                .ok()
                .filter(|width| *width > 0)
                .ok_or(AdapterError::NativeField("memory width"))?;
            let lane_mask = source.lane_mask.unwrap_or(effective);
            let mut lane_addresses = Vec::with_capacity(lane_mask.count_ones() as usize);
            if effective != 0 {
                let decoded = access
                    .accesses
                    .get(index)
                    .ok_or(AdapterError::MissingField("decoded operand"))?;
                for (lane, address) in decoded.lane_vector.iter().enumerate() {
                    if lane_mask & (1u32 << lane) != 0 {
                        lane_addresses.push(LaneAddress {
                            lane: lane as u8,
                            address: u64::try_from(*address)
                                .map_err(|_| AdapterError::NativeField("negative address"))?,
                        });
                    }
                }
            }
            if source.address_space.is_none() {
                losses.push(LossReason::AddressSpaceUnavailable);
            }
            operands.push(MemoryOperand {
                address_space: source.address_space.map(native_space).transpose()?,
                allocation_id: source.allocation_id,
                lane_mask: LaneMask::new(32, u64::from(lane_mask))?,
                width_bytes,
                lane_addresses,
                native: Some(NativeOperand::NvidiaSm86 {
                    declared_width: source.data_width,
                    descriptor: source.udesc_value,
                    explicit_lane_mask: source.lane_mask.is_some(),
                }),
            });
        }
        if effective != 0 && operands.is_empty() {
            losses.push(LossReason::AddressPayloadUnavailable);
        }
        Some(MemoryEvent {
            family,
            direction,
            dynamic_class_present: true,
            native_cache_op: instruction.instruction_cache_op,
            declared_width: instruction.data_width,
            operands,
        })
    } else {
        if !instruction.addresses.is_empty() {
            return Err(AdapterError::NativeField("addresses on non-memory record"));
        }
        if operation == OperationKind::Memory
            && classification
                .as_ref()
                .is_some_and(|value| matches!(sm86_opcode_head(&value.opcode_text), "LDC" | "ULDC"))
        {
            if let Some(raw) = instruction.instruction_cache_op {
                pb::InstructionCacheOp::try_from(raw)
                    .map_err(|_| AdapterError::NativeField("cache operator"))?;
            }
            // Constant-cache loads are deliberately absent from the dynamic
            // memory payload. Keep a static-only event so the SM model can
            // account for its cycle-visible issue/return path.
            losses.push(LossReason::DynamicMemoryClassUnavailable);
            Some(MemoryEvent {
                family: MemoryFamily::Constant,
                direction: Some(MemoryDirection::Load),
                dynamic_class_present: false,
                native_cache_op: instruction.instruction_cache_op,
                declared_width: instruction.data_width,
                operands: Vec::new(),
            })
        } else if operation == OperationKind::Memory {
            // Static opcode text can identify a load/store that the dynamic
            // producer did not classify. Do not manufacture a memory event;
            // retain the text and expose the missing dynamic discriminator.
            operation = OperationKind::Unknown;
            losses.push(LossReason::DynamicMemoryClassUnavailable);
            None
        } else {
            None
        }
    };
    losses.sort_unstable();
    losses.dedup();
    let event = SemanticEvent {
        ordinal,
        native,
        active_lanes: LaneMask::new(32, u64::from(active))?,
        predicate_lanes: LaneMask::new(32, u64::from(predicate))?,
        operation,
        opcode_text,
        control,
        async_effect: None,
        memory,
        losses,
    };
    event.validate()?;
    Ok(event)
}

/// Convert portable byte-address evidence to canonical list-all wire records.
/// Cache tags, optional masks, descriptor and widths retain their native meaning.
pub fn to_instruction(event: &SemanticEvent) -> Result<pb::Instruction, AdapterError> {
    event.validate()?;
    if event.native.isa != Isa::NvidiaSm86 {
        return Err(AdapterError::NativeField("ISA"));
    }
    let mut instruction = pb::Instruction {
        pc: u32::try_from(event.native.pc).map_err(|_| AdapterError::NativeField("pc"))?,
        active_mask: Some(event.active_lanes.bits() as u32),
        predicate_mask: Some(event.predicate_lanes.bits() as u32),
        function_unique_id: event.native.function_unique_id,
        warpid_tb: Some(event.native.warp_id),
        warpid_sm: event.native.warpid_sm,
        sm_id: event.native.sm_id,
        vpc: event.native.vpc,
        instrumentation_order: event.native.instrumentation_order,
        ..Default::default()
    };
    if let Some(memory) = &event.memory {
        // Static constant-cache events intentionally have no dynamic memory
        // class, but their native cache operator and declared width are still
        // observable metadata and must survive a semantic round-trip.
        instruction.instruction_cache_op = memory.native_cache_op;
        instruction.data_width = memory.declared_width;
        if memory.dynamic_class_present {
            instruction.memory_op_class = Some(wire_family(memory.family)? as i32);
        } else if memory.family != MemoryFamily::Constant {
            return Err(AdapterError::NativeField(
                "non-constant event lacks dynamic memory class",
            ));
        }
        if memory.dynamic_class_present {
            instruction.memory_op_access_kind =
                memory.direction.map(|kind| wire_direction(kind) as i32);
        }
        for operand in &memory.operands {
            let Some(NativeOperand::NvidiaSm86 {
                declared_width,
                descriptor,
                explicit_lane_mask,
            }) = operand.native
            else {
                return Err(AdapterError::MissingField("NVIDIA operand encoding"));
            };
            let resolved_width = if declared_width > 0 {
                declared_width
            } else {
                memory.declared_width.unwrap_or(0)
            };
            if u32::try_from(resolved_width).ok() != Some(operand.width_bytes) {
                return Err(AdapterError::NativeField("operand width disagreement"));
            }
            let mut addrs = vec![-1; 32];
            for lane in &operand.lane_addresses {
                addrs[usize::from(lane.lane)] = i64::try_from(lane.address)
                    .map_err(|_| AdapterError::NativeField("address exceeds NVBit domain"))?;
            }
            instruction.addresses.push(pb::Address {
                data_width: declared_width,
                udesc_value: descriptor,
                compression_format: 0,
                addrs,
                allocation_id: operand.allocation_id,
                address_space: operand.address_space.map(|space| wire_space(space) as i32),
                lane_mask: explicit_lane_mask.then_some(operand.lane_mask.bits() as u32),
                ..Default::default()
            });
        }
    }
    Ok(instruction)
}

pub fn visit_trace(
    trace: &TraceDir,
    classifier: Option<&dyn NativeInstructionClassifier>,
    mut consume: impl FnMut(&SemanticEvent) -> Result<(), AdapterError>,
) -> Result<SemanticTraceSummary, AdapterError> {
    if trace.trace().binary_version != 86 {
        return Err(AdapterError::Architecture(trace.trace().binary_version));
    }
    let mut counts = SemanticCounts::default();
    let mut dynamic_memory_class_gaps: BTreeMap<String, u64> = BTreeMap::new();
    let mut unclassified_opcodes: BTreeMap<String, u64> = BTreeMap::new();
    let mut launch_count = 0;
    for loc in trace.kernels() {
        launch_count += 1;
        let mut paths = trace.threadblock_paths(&loc)?;
        paths.sort_by_key(|path| {
            trace
                .validated_threadblock(path)
                .map(|block| block.launch_index())
                .unwrap_or(i64::MAX)
        });
        for path in paths {
            let block_index = trace.validated_threadblock(&path)?.launch_index();
            let block = trace.read_threadblock(&path)?;
            for warp in block.warps.values() {
                for (index, instruction) in warp.instructions.iter().enumerate() {
                    let native = identity(
                        loc.device_id(),
                        loc.stream_id(),
                        loc.kernel().id,
                        block_index,
                        warp.id,
                        index as u64,
                        instruction,
                    );
                    let classification = classifier.and_then(|classifier| {
                        classifier
                            .classify(instruction.function_unique_id, u64::from(instruction.pc))
                    });
                    let event = from_instruction(
                        counts.instruction_events,
                        native,
                        instruction,
                        classification,
                    )?;
                    counts.observe(&event)?;
                    if event
                        .losses
                        .contains(&LossReason::DynamicMemoryClassUnavailable)
                    {
                        let opcode = event
                            .opcode_text
                            .as_deref()
                            .unwrap_or("<missing opcode>")
                            .to_owned();
                        let entry = dynamic_memory_class_gaps.entry(opcode).or_default();
                        *entry = entry.checked_add(1).ok_or(SemanticError::CountOverflow)?;
                    }
                    if event.losses.contains(&LossReason::OpcodeUnclassified) {
                        let opcode = event
                            .opcode_text
                            .as_deref()
                            .unwrap_or("<missing opcode>")
                            .to_owned();
                        let entry = unclassified_opcodes.entry(opcode).or_default();
                        *entry = entry.checked_add(1).ok_or(SemanticError::CountOverflow)?;
                    }
                    consume(&event)?;
                }
            }
        }
    }
    let expected = trace.validation_summary().instruction_count();
    if counts.instruction_events != expected {
        return Err(AdapterError::Count {
            expected,
            actual: counts.instruction_events,
        });
    }
    Ok(SemanticTraceSummary {
        schema_version: SCHEMA_VERSION,
        source_isa: Isa::NvidiaSm86,
        launch_count,
        source_instruction_count: expected,
        order_semantics: "lane_group_program_order; ordinal_is_serialization_only",
        not_observed: [
            "cross_lane_group_issue_order",
            "register_values_and_data_dependencies",
            "silicon_completion_timestamps",
        ],
        dynamic_memory_class_gaps,
        unclassified_opcodes,
        counts,
    })
}

pub fn identity(
    device_id: i64,
    stream_id: i64,
    kernel_id: i32,
    block_launch_index: i64,
    warp_id: i32,
    instruction_index: u64,
    instruction: &pb::Instruction,
) -> NativeInstruction {
    NativeInstruction {
        isa: Isa::NvidiaSm86,
        simt_width: None,
        device_id,
        stream_id,
        kernel_id,
        block_launch_index,
        warp_id,
        instruction_index,
        function_unique_id: instruction.function_unique_id,
        pc: u64::from(instruction.pc),
        vpc: instruction.vpc,
        sm_id: instruction.sm_id,
        warpid_sm: instruction.warpid_sm,
        instrumentation_order: instruction.instrumentation_order,
    }
}

/// Write an experimental trace from serialized/deserialized semantic events.
/// The destination must be new; original captures are never overwritten.
pub fn write_roundtrip(
    trace: &TraceDir,
    destination: &Path,
    classifier: Option<&dyn NativeInstructionClassifier>,
) -> Result<u64, AdapterError> {
    use prost::Message;
    let io = |error: std::io::Error| AdapterError::Consumer(error.to_string());
    std::fs::create_dir(destination).map_err(io)?;
    std::fs::write(
        destination.join("dynamic_trace.pb"),
        trace.trace().encode_to_vec(),
    )
    .map_err(io)?;
    std::fs::create_dir(destination.join("extra_info")).map_err(io)?;
    for relative in [
        "extra_info/enhanced_execution_info.json",
        "extra_info/allocations.json",
        "stats.csv",
    ] {
        let source = trace.root().join(relative);
        if source.exists() {
            std::fs::copy(&source, destination.join(relative)).map_err(io)?;
        }
    }
    let mut count = 0;
    for loc in trace.kernels() {
        for path in trace.threadblock_paths(&loc)? {
            let index = trace.validated_threadblock(&path)?.launch_index();
            let mut block = trace.read_threadblock(&path)?;
            for warp in block.warps.values_mut() {
                for (ix, instruction) in warp.instructions.iter_mut().enumerate() {
                    let native = identity(
                        loc.device_id(),
                        loc.stream_id(),
                        loc.kernel().id,
                        index,
                        warp.id,
                        ix as u64,
                        instruction,
                    );
                    let classification = classifier.and_then(|classifier| {
                        classifier
                            .classify(instruction.function_unique_id, u64::from(instruction.pc))
                    });
                    let event = from_instruction(count, native, instruction, classification)?;
                    let bytes = serde_json::to_vec(&event)
                        .map_err(|error| AdapterError::Consumer(error.to_string()))?;
                    let decoded: SemanticEvent = serde_json::from_slice(&bytes)
                        .map_err(|error| AdapterError::Consumer(error.to_string()))?;
                    *instruction = to_instruction(&decoded)?;
                    count += 1;
                }
            }
            let relative = path
                .strip_prefix(trace.root())
                .map_err(|error| AdapterError::Consumer(error.to_string()))?;
            let output = destination.join(relative);
            std::fs::create_dir_all(
                output
                    .parent()
                    .ok_or(AdapterError::NativeField("output parent"))?,
            )
            .map_err(io)?;
            std::fs::write(output, block.encode_to_vec()).map_err(io)?;
        }
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn atomic_aliases_width_and_order_survive_semantic_json_roundtrip() {
        let mut instruction = pb::Instruction {
            pc: 16,
            vpc: Some(16),
            active_mask: Some(7),
            predicate_mask: Some(7),
            function_unique_id: 1,
            warpid_tb: Some(0),
            warpid_sm: Some(0),
            sm_id: Some(0),
            memory_op_class: Some(pb::MemoryOpClass::Atomic as i32),
            memory_op_access_kind: Some(pb::MemoryOpAccessKind::Atomic as i32),
            data_width: Some(4),
            instrumentation_order: Some(3),
            ..Default::default()
        };
        let mut addrs = vec![-1; 32];
        addrs[..3].copy_from_slice(&[31, 31, 64]);
        instruction.addresses.push(pb::Address {
            data_width: 4,
            addrs,
            allocation_id: Some(1),
            address_space: Some(pb::MemorySpace::Global as i32),
            ..Default::default()
        });
        let native = identity(0, 7, 1, 0, 0, 0, &instruction);
        let event = from_instruction(0, native, &instruction, None).unwrap();
        let decoded = serde_json::from_slice(&serde_json::to_vec(&event).unwrap()).unwrap();
        assert_eq!(to_instruction(&decoded).unwrap(), instruction);
    }

    #[test]
    fn static_constant_metadata_survives_without_dynamic_memory_class() {
        let instruction = pb::Instruction {
            pc: 32,
            active_mask: Some(1),
            predicate_mask: Some(1),
            function_unique_id: 1,
            warpid_tb: Some(0),
            warpid_sm: Some(0),
            sm_id: Some(0),
            data_width: Some(8),
            instruction_cache_op: Some(pb::InstructionCacheOp::Other as i32),
            ..Default::default()
        };
        let native = identity(0, 7, 1, 0, 0, 0, &instruction);
        let classification = InstructionClassification {
            opcode_text: "LDC.64".to_owned(),
            operation: OperationKind::Memory,
            control: None,
        };
        let event = from_instruction(0, native, &instruction, Some(classification)).unwrap();
        let decoded = serde_json::from_slice(&serde_json::to_vec(&event).unwrap()).unwrap();
        let roundtrip = to_instruction(&decoded).unwrap();
        assert_eq!(roundtrip.memory_op_class, None);
        assert_eq!(roundtrip.data_width, instruction.data_width);
        assert_eq!(
            roundtrip.instruction_cache_op,
            instruction.instruction_cache_op
        );
    }

    #[test]
    fn static_constant_rejects_unknown_cache_operator() {
        let instruction = pb::Instruction {
            pc: 32,
            active_mask: Some(1),
            predicate_mask: Some(1),
            function_unique_id: 1,
            instruction_cache_op: Some(i32::MAX),
            ..Default::default()
        };
        let native = identity(0, 7, 1, 0, 0, 0, &instruction);
        let classification = InstructionClassification {
            opcode_text: "LDC.32".to_owned(),
            operation: OperationKind::Memory,
            control: None,
        };
        assert!(matches!(
            from_instruction(0, native, &instruction, Some(classification)),
            Err(AdapterError::NativeField("cache operator"))
        ));
    }
}
