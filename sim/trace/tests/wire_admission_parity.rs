//! Rust consumer for the shared Q1009 Rust/Python wire-admission corpus.

#![allow(clippy::unwrap_used)] // test binary: unwrap is an assertion

use std::path::{Path, PathBuf};

use prost::Message;
use serde::Deserialize;
use sim_trace::{LANE_INACTIVE_SENTINEL, TraceDir, pb};

#[derive(Deserialize)]
struct Golden {
    contract: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    mutation: String,
    expected_typed_reason: Option<String>,
}

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tools/nvbit_tracer/wire_admission_parity_golden.json")
        .canonicalize()
        .expect("shared wire admission fixture exists")
}

fn valid_instruction() -> pb::Instruction {
    pb::Instruction {
        pc: 0x10,
        active_mask: Some(0),
        predicate_mask: Some(0),
        function_unique_id: 1,
        warpid_tb: Some(0),
        vpc: Some(0x10),
        ..Default::default()
    }
}

fn instruction_mut(threadblock: &mut pb::Threadblock) -> &mut pb::Instruction {
    &mut threadblock
        .warps
        .get_mut(&0)
        .expect("base warp")
        .instructions[0]
}

#[allow(clippy::too_many_arguments)]
fn make_memory(
    instruction: &mut pb::Instruction,
    op_class: pb::MemoryOpClass,
    address: i64,
    address_space: i32,
    allocation_id: Option<u32>,
    active_mask: u32,
    list_length: usize,
) -> &mut pb::Address {
    instruction.addresses.clear();
    instruction.active_mask = Some(active_mask);
    instruction.predicate_mask = Some(active_mask);
    instruction.memory_op_class = Some(op_class as i32);
    instruction.data_width = Some(4);
    let mut addrs = vec![LANE_INACTIVE_SENTINEL; list_length];
    if let Some(first) = addrs.first_mut() {
        *first = address;
    }
    instruction.addresses.push(pb::Address {
        data_width: 4,
        udesc_value: 0,
        compression_format: 0,
        base_address: 0,
        stride: 0,
        addrs,
        allocation_id,
        address_space: Some(address_space),
        lane_mask: None,
    });
    instruction.addresses.last_mut().unwrap()
}

fn encode_varint(mut value: usize) -> Vec<u8> {
    let mut encoded = Vec::new();
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        encoded.push(byte);
        if value == 0 {
            return encoded;
        }
    }
}

fn message_field(number: u8, payload: &[u8]) -> Vec<u8> {
    let mut encoded = vec![number << 3 | 2];
    encoded.extend(encode_varint(payload.len()));
    encoded.extend(payload);
    encoded
}

fn threadblock_with_instruction_suffix(threadblock: &pb::Threadblock, suffix: &[u8]) -> Vec<u8> {
    let block_id = threadblock.block_id.as_ref().expect("base block id");
    let warp = threadblock.warps.get(&0).expect("base warp");
    let mut instruction = warp.instructions[0].encode_to_vec();
    instruction.extend_from_slice(suffix);

    let warp = message_field(2, &instruction);
    let warp_entry = message_field(2, &warp);
    let mut encoded = message_field(1, &block_id.encode_to_vec());
    encoded.extend(message_field(2, &warp_entry));
    encoded
}

fn trace_with_kernel_suffix(trace: &pb::Trace, suffix: &[u8]) -> Vec<u8> {
    let mut metadata = trace.clone();
    metadata.gpu_device.clear();
    let mut kernel = trace.gpu_device[&0].streams[&0].kernels[0].encode_to_vec();
    kernel.extend_from_slice(suffix);
    let stream = message_field(2, &kernel);
    let stream_entry = message_field(2, &stream);
    let device = message_field(2, &stream_entry);
    let device_entry = message_field(2, &device);
    let mut encoded = metadata.encode_to_vec();
    encoded.extend(message_field(6, &device_entry));
    encoded
}

fn write_case(root: &Path, mutation: &str) {
    std::fs::create_dir(root).unwrap();
    let kernel = pb::Kernel {
        id: 1,
        name: "wire_parity".into(),
        function_unique_id: 1,
        grid_dim: Some(pb::Dim3d { x: 1, y: 1, z: 1 }),
        block_dim: Some(pb::Dim3d { x: 32, y: 1, z: 1 }),
        ..Default::default()
    };
    let stream = pb::CudaStream {
        id: 0,
        kernels: vec![kernel],
        ..Default::default()
    };
    let device = pb::GpuDevice {
        id: 0,
        streams: [(0_i64, stream)].into_iter().collect(),
    };
    let mut trace = pb::Trace {
        gpu_device: [(0_i64, device)].into_iter().collect(),
        trace_contract_id: "sm86_addr_carry_v1".into(),
        memory_payload_classes: vec![1, 2, 3, 6, 8],
        ..Default::default()
    };
    let warp = pb::Warp {
        id: 0,
        instructions: vec![valid_instruction()],
    };
    let mut threadblock = pb::Threadblock {
        block_id: Some(pb::Dim3d { x: 0, y: 0, z: 0 }),
        warps: [(0_i32, warp)].into_iter().collect(),
    };

    let mut stats = "total_insts,total_reported_insts\n1,1\n".to_owned();
    let allocation = r#"{"allocation_id":1,"context_id":1,"device_id":0,"base_address":"0x1000","size_bytes":4096,"alive":true}"#;
    let mut allocations = format!("[{allocation}]");
    let mut payload_kernel_ids = vec![1_i32];
    let mut empty_payload = false;
    let mut empty_root = false;
    let mut root_known_field_wrong_wire_type = false;
    let mut kernel_wire_suffix = Vec::new();
    let mut instruction_known_field_wrong_wire_type = false;
    let mut write_allocations = true;

    match mutation {
        "none" => {}
        "empty_root_bytes" => empty_root = true,
        "root_known_field_wrong_wire_type" => root_known_field_wrong_wire_type = true,
        "empty_root_and_no_payload" => {
            empty_root = true;
            payload_kernel_ids.clear();
        }
        "wrong_contract" => trace.trace_contract_id = "wrong".into(),
        "wrong_contract_and_no_payload" => {
            trace.trace_contract_id = "wrong".into();
            payload_kernel_ids.clear();
        }
        "missing_capability" => {
            trace.memory_payload_classes.pop();
        }
        "duplicate_capability" => trace.memory_payload_classes.push(1),
        "split_capabilities_reversed" => {
            trace.trace_contract_id = "sm86_addr_carry_shmem_split_v2".into();
            trace.memory_payload_classes.reverse();
        }
        "split_capabilities_permuted" => {
            trace.trace_contract_id = "sm86_addr_carry_shmem_split_v2".into();
            trace.memory_payload_classes.swap(1, 2);
        }
        "zero_launches" => {
            trace
                .gpu_device
                .get_mut(&0)
                .unwrap()
                .streams
                .get_mut(&0)
                .unwrap()
                .kernels
                .clear();
        }
        "device_map_mismatch" => trace.gpu_device.get_mut(&0).unwrap().id = 1,
        "stream_map_mismatch" => {
            trace
                .gpu_device
                .get_mut(&0)
                .unwrap()
                .streams
                .get_mut(&0)
                .unwrap()
                .id = 1;
        }
        "duplicate_launch" => {
            let kernels = &mut trace
                .gpu_device
                .get_mut(&0)
                .unwrap()
                .streams
                .get_mut(&0)
                .unwrap()
                .kernels;
            kernels.push(kernels[0].clone());
        }
        "root_function_zero" => {
            trace
                .gpu_device
                .get_mut(&0)
                .unwrap()
                .streams
                .get_mut(&0)
                .unwrap()
                .kernels[0]
                .function_unique_id = 0;
        }
        "zero_grid_axis" => {
            trace
                .gpu_device
                .get_mut(&0)
                .unwrap()
                .streams
                .get_mut(&0)
                .unwrap()
                .kernels[0]
                .grid_dim
                .as_mut()
                .unwrap()
                .x = 0;
        }
        "zero_block_axis" => {
            trace
                .gpu_device
                .get_mut(&0)
                .unwrap()
                .streams
                .get_mut(&0)
                .unwrap()
                .kernels[0]
                .block_dim
                .as_mut()
                .unwrap()
                .x = 0;
        }
        "legacy_shared_memory_nonzero" => {
            trace
                .gpu_device
                .get_mut(&0)
                .unwrap()
                .streams
                .get_mut(&0)
                .unwrap()
                .kernels[0]
                .size_shared_memory = 4096;
        }
        "legacy_shared_memory_split_unadvertised" => {
            let kernel = &mut trace
                .gpu_device
                .get_mut(&0)
                .unwrap()
                .streams
                .get_mut(&0)
                .unwrap()
                .kernels[0];
            kernel.shared_memory_static_bytes = Some(0);
            kernel.shared_memory_dynamic_bytes = Some(0);
        }
        "split_shared_memory_matching" => {
            trace.trace_contract_id = "sm86_addr_carry_shmem_split_v2".into();
            let kernel = &mut trace
                .gpu_device
                .get_mut(&0)
                .unwrap()
                .streams
                .get_mut(&0)
                .unwrap()
                .kernels[0];
            kernel.size_shared_memory = 6144;
            kernel.shared_memory_static_bytes = Some(2048);
            kernel.shared_memory_dynamic_bytes = Some(4096);
        }
        "split_shared_memory_zero" => {
            trace.trace_contract_id = "sm86_addr_carry_shmem_split_v2".into();
            let kernel = &mut trace
                .gpu_device
                .get_mut(&0)
                .unwrap()
                .streams
                .get_mut(&0)
                .unwrap()
                .kernels[0];
            kernel.shared_memory_static_bytes = Some(0);
            kernel.shared_memory_dynamic_bytes = Some(0);
        }
        "split_shared_memory_static_only" => {
            trace.trace_contract_id = "sm86_addr_carry_shmem_split_v2".into();
            let kernel = &mut trace
                .gpu_device
                .get_mut(&0)
                .unwrap()
                .streams
                .get_mut(&0)
                .unwrap()
                .kernels[0];
            kernel.size_shared_memory = 2048;
            kernel.shared_memory_static_bytes = Some(2048);
            kernel.shared_memory_dynamic_bytes = Some(0);
        }
        "split_shared_memory_dynamic_only" => {
            trace.trace_contract_id = "sm86_addr_carry_shmem_split_v2".into();
            let kernel = &mut trace
                .gpu_device
                .get_mut(&0)
                .unwrap()
                .streams
                .get_mut(&0)
                .unwrap()
                .kernels[0];
            kernel.size_shared_memory = 4096;
            kernel.shared_memory_static_bytes = Some(0);
            kernel.shared_memory_dynamic_bytes = Some(4096);
        }
        "split_shared_memory_missing_both" => {
            trace.trace_contract_id = "sm86_addr_carry_shmem_split_v2".into();
        }
        "split_shared_memory_missing_static" => {
            trace.trace_contract_id = "sm86_addr_carry_shmem_split_v2".into();
            trace
                .gpu_device
                .get_mut(&0)
                .unwrap()
                .streams
                .get_mut(&0)
                .unwrap()
                .kernels[0]
                .shared_memory_dynamic_bytes = Some(0);
        }
        "split_shared_memory_missing_dynamic" => {
            trace.trace_contract_id = "sm86_addr_carry_shmem_split_v2".into();
            trace
                .gpu_device
                .get_mut(&0)
                .unwrap()
                .streams
                .get_mut(&0)
                .unwrap()
                .kernels[0]
                .shared_memory_static_bytes = Some(0);
        }
        "split_shared_memory_mismatch" => {
            trace.trace_contract_id = "sm86_addr_carry_shmem_split_v2".into();
            let kernel = &mut trace
                .gpu_device
                .get_mut(&0)
                .unwrap()
                .streams
                .get_mut(&0)
                .unwrap()
                .kernels[0];
            kernel.size_shared_memory = 4096;
            kernel.shared_memory_static_bytes = Some(1024);
            kernel.shared_memory_dynamic_bytes = Some(2048);
        }
        "shared_memory_combined_negative" => {
            trace
                .gpu_device
                .get_mut(&0)
                .unwrap()
                .streams
                .get_mut(&0)
                .unwrap()
                .kernels[0]
                .size_shared_memory = -1;
        }
        "split_shared_memory_wrong_wire_type" => {
            trace.trace_contract_id = "sm86_addr_carry_shmem_split_v2".into();
            trace
                .gpu_device
                .get_mut(&0)
                .unwrap()
                .streams
                .get_mut(&0)
                .unwrap()
                .kernels[0]
                .shared_memory_dynamic_bytes = Some(0);
            kernel_wire_suffix = vec![0x5a, 0x00];
        }
        "split_shared_memory_duplicate_field" => {
            trace.trace_contract_id = "sm86_addr_carry_shmem_split_v2".into();
            let kernel = &mut trace
                .gpu_device
                .get_mut(&0)
                .unwrap()
                .streams
                .get_mut(&0)
                .unwrap()
                .kernels[0];
            kernel.shared_memory_static_bytes = Some(0);
            kernel.shared_memory_dynamic_bytes = Some(0);
            kernel_wire_suffix = vec![0x58, 0x00];
        }
        "no_payload_files" => payload_kernel_ids.clear(),
        "launch_payload_missing" => payload_kernel_ids = vec![9],
        "orphan_payload" => payload_kernel_ids.push(9),
        "empty_payload_bytes" => empty_payload = true,
        "instruction_known_field_wrong_wire_type" => {
            instruction_known_field_wrong_wire_type = true;
        }
        "empty_payload_records" => threadblock.warps.clear(),
        "block_id_missing" => threadblock.block_id = None,
        "block_id_out_of_range" => threadblock.block_id.as_mut().unwrap().x = 1,
        "instruction_function_mismatch" => {
            instruction_mut(&mut threadblock).function_unique_id = 2;
        }
        "function_mismatch_with_atomic_shape" => {
            let instruction = instruction_mut(&mut threadblock);
            make_memory(
                instruction,
                pb::MemoryOpClass::Atomic,
                0x1000,
                pb::MemorySpace::Global as i32,
                Some(1),
                1,
                32,
            );
            instruction.addresses.push(instruction.addresses[0].clone());
            instruction.function_unique_id = 2;
        }
        "warpid_missing" => instruction_mut(&mut threadblock).warpid_tb = None,
        "instruction_function_zero" => {
            instruction_mut(&mut threadblock).function_unique_id = 0;
        }
        "active_mask_missing" => instruction_mut(&mut threadblock).active_mask = None,
        "predicate_mask_missing" => instruction_mut(&mut threadblock).predicate_mask = None,
        "vpc_missing" => instruction_mut(&mut threadblock).vpc = None,
        "vpc_mismatch" => instruction_mut(&mut threadblock).vpc = Some(0x11),
        "memory_data_width_precedence" => {
            let instruction = instruction_mut(&mut threadblock);
            instruction.memory_op_class = Some(pb::MemoryOpClass::Global as i32);
            instruction.data_width = None;
            instruction.warpid_tb = None;
        }
        "unknown_memory_class" => {
            let instruction = instruction_mut(&mut threadblock);
            instruction.memory_op_class = Some(99);
            instruction.data_width = Some(4);
        }
        "unspecified_memory_class" => {
            let instruction = instruction_mut(&mut threadblock);
            instruction.memory_op_class = Some(pb::MemoryOpClass::Unspecified as i32);
            instruction.data_width = Some(4);
        }
        "undeclared_payload_class" => {
            let instruction = instruction_mut(&mut threadblock);
            instruction.memory_op_class = Some(pb::MemoryOpClass::Texture as i32);
            instruction.data_width = Some(4);
        }
        "active_global_without_operand" => {
            let instruction = instruction_mut(&mut threadblock);
            instruction.memory_op_class = Some(pb::MemoryOpClass::Global as i32);
            instruction.data_width = Some(4);
            instruction.active_mask = Some(1);
            instruction.predicate_mask = Some(1);
        }
        "zero_active_global_without_operand" => {
            let instruction = instruction_mut(&mut threadblock);
            instruction.memory_op_class = Some(pb::MemoryOpClass::Global as i32);
            instruction.data_width = Some(4);
        }
        "atomic_missing_space" => {
            let operand = make_memory(
                instruction_mut(&mut threadblock),
                pb::MemoryOpClass::Atomic,
                0x1000,
                pb::MemorySpace::Global as i32,
                Some(1),
                1,
                32,
            );
            operand.address_space = None;
        }
        "atomic_multiple_operands" => {
            let instruction = instruction_mut(&mut threadblock);
            make_memory(
                instruction,
                pb::MemoryOpClass::Atomic,
                0x1000,
                pb::MemorySpace::Global as i32,
                Some(1),
                1,
                32,
            );
            instruction.addresses.push(instruction.addresses[0].clone());
        }
        "zero_active_atomic_without_operand" => {
            let instruction = instruction_mut(&mut threadblock);
            instruction.memory_op_class = Some(pb::MemoryOpClass::Atomic as i32);
            instruction.data_width = Some(4);
        }
        "async_copy_without_operand" => {
            let instruction = instruction_mut(&mut threadblock);
            instruction.memory_op_class = Some(pb::MemoryOpClass::AsyncCopy as i32);
            instruction.data_width = Some(4);
            instruction.active_mask = Some(1);
            instruction.predicate_mask = Some(1);
        }
        "async_copy_missing_space" => {
            let operand = make_memory(
                instruction_mut(&mut threadblock),
                pb::MemoryOpClass::AsyncCopy,
                0x1000,
                pb::MemorySpace::Global as i32,
                Some(1),
                1,
                32,
            );
            operand.address_space = None;
        }
        "explicit_unspecified_space" => {
            make_memory(
                instruction_mut(&mut threadblock),
                pb::MemoryOpClass::Global,
                0x1000,
                pb::MemorySpace::Unspecified as i32,
                Some(1),
                1,
                32,
            );
        }
        "unspecified_space_before_later_malformed_lane" => {
            let instruction = instruction_mut(&mut threadblock);
            make_memory(
                instruction,
                pb::MemoryOpClass::Global,
                0x1000,
                pb::MemorySpace::Unspecified as i32,
                Some(1),
                1,
                32,
            );
            let mut malformed = instruction.addresses[0].clone();
            malformed.address_space = Some(pb::MemorySpace::Global as i32);
            malformed.addrs.pop();
            instruction.addresses.push(malformed);
        }
        "unknown_memory_space" => {
            make_memory(
                instruction_mut(&mut threadblock),
                pb::MemoryOpClass::Global,
                0x1000,
                99,
                Some(1),
                1,
                32,
            );
        }
        "lane_mask_inconsistent" => {
            make_memory(
                instruction_mut(&mut threadblock),
                pb::MemoryOpClass::Shared,
                0x1000,
                pb::MemorySpace::Shared as i32,
                None,
                0,
                32,
            );
        }
        "list_all_wrong_length" => {
            make_memory(
                instruction_mut(&mut threadblock),
                pb::MemoryOpClass::Shared,
                0x1000,
                pb::MemorySpace::Shared as i32,
                None,
                0,
                31,
            );
        }
        "lane_values_resource_limit" => {
            make_memory(
                instruction_mut(&mut threadblock),
                pb::MemoryOpClass::Shared,
                0x1000,
                pb::MemorySpace::Shared as i32,
                None,
                0,
                33,
            );
        }
        "compressed_address_overflow" => {
            let operand = make_memory(
                instruction_mut(&mut threadblock),
                pb::MemoryOpClass::Shared,
                0,
                pb::MemorySpace::Shared as i32,
                None,
                3,
                0,
            );
            operand.compression_format = 1;
            operand.base_address = 1_u64 << 63;
            operand.stride = 4;
        }
        "unspecified_access_kind" => {
            let instruction = instruction_mut(&mut threadblock);
            instruction.memory_op_class = Some(pb::MemoryOpClass::Global as i32);
            instruction.data_width = Some(4);
            instruction.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Unspecified as i32);
        }
        "unknown_access_kind" => {
            let instruction = instruction_mut(&mut threadblock);
            instruction.memory_op_class = Some(pb::MemoryOpClass::Global as i32);
            instruction.data_width = Some(4);
            instruction.memory_op_access_kind = Some(99);
        }
        "unknown_access_kind_with_missing_allocation_id" => {
            let instruction = instruction_mut(&mut threadblock);
            make_memory(
                instruction,
                pb::MemoryOpClass::Global,
                0x1000,
                pb::MemorySpace::Global as i32,
                None,
                1,
                32,
            );
            instruction.memory_op_access_kind = Some(99);
        }
        "unspecified_cache_op" => {
            let instruction = instruction_mut(&mut threadblock);
            instruction.memory_op_class = Some(pb::MemoryOpClass::Global as i32);
            instruction.data_width = Some(4);
            instruction.instruction_cache_op = Some(pb::InstructionCacheOp::Unspecified as i32);
        }
        "unknown_cache_op" => {
            let instruction = instruction_mut(&mut threadblock);
            instruction.memory_op_class = Some(pb::MemoryOpClass::Global as i32);
            instruction.data_width = Some(4);
            instruction.instruction_cache_op = Some(99);
        }
        "unknown_cache_op_with_missing_allocation_id" => {
            let instruction = instruction_mut(&mut threadblock);
            make_memory(
                instruction,
                pb::MemoryOpClass::Global,
                0x1000,
                pb::MemorySpace::Global as i32,
                None,
                1,
                32,
            );
            instruction.instruction_cache_op = Some(99);
        }
        "metadata_only_nonmemory" => {
            let instruction = instruction_mut(&mut threadblock);
            instruction.data_width = Some(4);
            instruction.memory_op_access_kind = Some(pb::MemoryOpAccessKind::Load as i32);
            instruction.instruction_cache_op = Some(pb::InstructionCacheOp::LdgECa as i32);
        }
        "allocation_id_missing" => {
            make_memory(
                instruction_mut(&mut threadblock),
                pb::MemoryOpClass::Global,
                0x1000,
                pb::MemorySpace::Global as i32,
                None,
                1,
                32,
            );
        }
        "allocation_id_unknown" => {
            make_memory(
                instruction_mut(&mut threadblock),
                pb::MemoryOpClass::Global,
                0x1000,
                pb::MemorySpace::Global as i32,
                Some(2),
                1,
                32,
            );
        }
        "zero_active_explicit_allocation_without_sidecar" => {
            let operand = make_memory(
                instruction_mut(&mut threadblock),
                pb::MemoryOpClass::Global,
                0x1000,
                pb::MemorySpace::Global as i32,
                Some(1),
                0,
                32,
            );
            operand.addrs[0] = LANE_INACTIVE_SENTINEL;
            write_allocations = false;
        }
        "lane_outside_allocation" => {
            make_memory(
                instruction_mut(&mut threadblock),
                pb::MemoryOpClass::Global,
                0x3000,
                pb::MemorySpace::Global as i32,
                Some(1),
                1,
                32,
            );
        }
        "duplicate_allocation_id" => allocations = format!("[{allocation},{allocation}]"),
        "duplicate_allocation_object_key" => {
            allocations = r#"[{"allocation_id":1,"allocation_id":1,"context_id":1,"device_id":0,"base_address":"0x1000","size_bytes":4096,"alive":true}]"#.into();
        }
        "allocation_hex_digit_separator" => {
            allocations = allocations.replace("0x1000", "0x1_000");
        }
        "allocation_integer_token_too_long" => {
            allocations = format!(
                r#"[{{"allocation_id":{},"context_id":1,"device_id":0,"base_address":"0x1000","size_bytes":4096,"alive":true}}]"#,
                "9".repeat(5_000)
            );
        }
        "stats_empty" => stats.clear(),
        "stats_schema_invalid" => stats = "other,value\n1,1\n".into(),
        "stats_negative" => stats = "total_insts,total_reported_insts\n-1,0\n".into(),
        "stats_digit_separator" => {
            stats = "total_insts,total_reported_insts\n1_0,1\n".into();
        }
        "stats_admitted_exceeds_expected" => {
            stats = "total_insts,total_reported_insts\n1,2\n".into();
        }
        "stats_vacuous" => stats = "total_insts,total_reported_insts\n0,0\n".into(),
        "stats_row_exceeds_u64" => {
            stats = "total_insts,total_reported_insts\n18446744073709551616,1\n".into();
        }
        "stats_total_overflow" => {
            stats = format!("total_insts,total_reported_insts\n{},1\n1,1\n", u64::MAX);
        }
        other => panic!("unimplemented golden mutation {other:?}"),
    }

    let mut root_bytes = if empty_root {
        Vec::new()
    } else {
        trace.encode_to_vec()
    };
    if !kernel_wire_suffix.is_empty() {
        root_bytes = trace_with_kernel_suffix(&trace, &kernel_wire_suffix);
    }
    if root_known_field_wrong_wire_type {
        root_bytes.extend([0x08, 0x00]);
    }
    std::fs::write(root.join("dynamic_trace.pb"), root_bytes).unwrap();
    let extra = root.join("extra_info");
    std::fs::create_dir(&extra).unwrap();
    if write_allocations {
        std::fs::write(extra.join("allocations.json"), allocations).unwrap();
    }
    std::fs::write(root.join("stats.csv"), stats).unwrap();
    let payload_root = root.join("threadblocks");
    std::fs::create_dir(&payload_root).unwrap();
    for kernel_id in payload_kernel_ids {
        let path = payload_root
            .join("device_0/stream_0")
            .join(format!("kernel_{kernel_id}/0.pb"));
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let payload = if empty_payload {
            Vec::new()
        } else if instruction_known_field_wrong_wire_type {
            threadblock_with_instruction_suffix(&threadblock, &[0x28, 0x00])
        } else {
            threadblock.encode_to_vec()
        };
        std::fs::write(path, payload).unwrap();
    }
}

#[test]
fn split_v2_rejects_reversed_and_permuted_capability_vectors() {
    for (mutation, expected) in [
        ("split_capabilities_reversed", vec![8, 6, 3, 2, 1]),
        ("split_capabilities_permuted", vec![1, 3, 2, 6, 8]),
    ] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("trace");
        write_case(&root, mutation);
        let error = TraceDir::open_direct(&root)
            .err()
            .expect("reordered split-v2 vector must fail closed");
        match error {
            sim_trace::TraceError::MemoryPayloadCapabilitiesInvalid {
                contract_id, got, ..
            } => {
                assert_eq!(contract_id, "sm86_addr_carry_shmem_split_v2");
                assert_eq!(got, expected, "{mutation}");
            }
            error => panic!("{mutation}: unexpected Rust error: {error:?}"),
        }
    }
}

#[test]
fn rust_wire_admission_matches_shared_typed_outcomes() {
    let golden: Golden =
        serde_json::from_str(&std::fs::read_to_string(golden_path()).unwrap()).unwrap();
    assert_eq!(golden.contract, "sim_trace_wire_v1");
    assert!(golden.cases.len() >= 40);
    for case in golden.cases {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("trace");
        write_case(&root, &case.mutation);
        let actual = match TraceDir::open_direct(&root) {
            Ok(_) => None,
            Err(error) => Some(
                error
                    .typed_reason()
                    .unwrap_or_else(|| {
                        panic!(
                            "{} ({}): Rust error has no parity reason: {error:?}",
                            case.name, case.mutation
                        )
                    })
                    .to_owned(),
            ),
        };
        assert_eq!(
            actual, case.expected_typed_reason,
            "{} ({})",
            case.name, case.mutation
        );
    }
}
