from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.nvbit_tracer.trace_admission import (
    TraceAdmissionError,
    _import_pb_modules,
    _import_trace_pb2,
    require_historical_trace_admission,
    require_trace_admission,
)


GOLDEN_PATH = Path(__file__).with_name("wire_admission_parity_golden.json")
CASES = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["cases"]


def _valid_instruction(warp: Any) -> Any:
    instruction = warp.instructions.add()
    instruction.pc = 0x10
    instruction.vpc = 0x10
    instruction.active_mask = 0
    instruction.predicate_mask = 0
    instruction.function_unique_id = 1
    instruction.warpid_tb = 0
    return instruction


def _make_memory(
    instruction: Any,
    instruction_pb2: Any,
    address_pb2: Any,
    *,
    op_class: int | None = None,
    address: int = 0x1000,
    address_space: int | None = None,
    allocation_id: int | None = 1,
    active_mask: int = 1,
    list_length: int = 32,
) -> Any:
    instruction.ClearField("addresses")
    instruction.active_mask = active_mask
    instruction.predicate_mask = active_mask
    instruction.memory_op_class = (
        instruction_pb2.MEMORY_OP_CLASS_GLOBAL if op_class is None else op_class
    )
    instruction.data_width = 4
    operand = instruction.addresses.add()
    operand.data_width = 4
    operand.compression_format = 0
    operand.addrs.extend([-1] * list_length)
    if list_length:
        operand.addrs[0] = address
    if address_space is None:
        operand.address_space = address_pb2.MEMORY_SPACE_GLOBAL
    else:
        operand.address_space = address_space
    if allocation_id is not None:
        operand.allocation_id = allocation_id
    return operand


def _varint(value: int) -> bytes:
    encoded = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            byte |= 0x80
        encoded.append(byte)
        if not value:
            return bytes(encoded)


def _message_field(number: int, payload: bytes) -> bytes:
    return bytes((number << 3 | 2,)) + _varint(len(payload)) + payload


def _threadblock_with_instruction_suffix(threadblock: Any, suffix: bytes) -> bytes:
    instruction = threadblock.warps[0].instructions[0].SerializeToString() + suffix
    warp = _message_field(2, instruction)
    warp_entry = _message_field(2, warp)
    return _message_field(1, threadblock.block_id.SerializeToString()) + _message_field(
        2, warp_entry
    )


def _trace_with_kernel_suffix(trace: Any, suffix: bytes) -> bytes:
    metadata = type(trace)()
    metadata.CopyFrom(trace)
    metadata.ClearField("gpu_device")
    kernel = trace.gpu_device[0].streams[0].kernels[0].SerializeToString() + suffix
    stream = _message_field(2, kernel)
    stream_entry = _message_field(2, stream)
    device = _message_field(2, stream_entry)
    device_entry = _message_field(2, device)
    return metadata.SerializeToString() + _message_field(6, device_entry)


def _build_case(trace_dir: Path, mutation: str) -> None:
    trace_pb2 = _import_trace_pb2()
    threadblock_pb2, instruction_pb2, address_pb2 = _import_pb_modules()
    trace_dir.mkdir()

    trace = trace_pb2.Trace()
    trace.trace_contract_id = "sm86_addr_carry_shmem_split_v2"
    trace.memory_payload_classes.extend([1, 2, 3, 6, 8])
    device = trace.gpu_device[0]
    device.id = 0
    stream = device.streams[0]
    stream.id = 0
    kernel = stream.kernels.add()
    kernel.id = 1
    kernel.name = "wire_parity"
    kernel.function_unique_id = 1
    kernel.grid_dim.x = kernel.grid_dim.y = kernel.grid_dim.z = 1
    kernel.block_dim.x = 32
    kernel.block_dim.y = kernel.block_dim.z = 1
    kernel.shared_memory_static_bytes = 0
    kernel.shared_memory_dynamic_bytes = 0

    threadblock = threadblock_pb2.threadblock()
    threadblock.block_id.x = threadblock.block_id.y = threadblock.block_id.z = 0
    warp = threadblock.warps[0]
    warp.id = 0
    instruction = _valid_instruction(warp)

    stats = "total_insts,total_reported_insts\n1,1\n"
    allocations: list[dict[str, object]] = [
        {
            "allocation_id": 1,
            "context_id": 1,
            "device_id": 0,
            "base_address": "0x1000",
            "size_bytes": 4096,
            "alive": True,
        }
    ]
    payload_kernel_ids = [1]
    empty_payload = False
    write_root = True
    root_known_field_wrong_wire_type = False
    kernel_wire_suffix = b""
    instruction_known_field_wrong_wire_type = False
    write_allocations = True
    allocations_raw: str | None = None

    if mutation == "none":
        pass
    elif mutation == "empty_root_bytes":
        write_root = False
    elif mutation == "root_known_field_wrong_wire_type":
        root_known_field_wrong_wire_type = True
    elif mutation == "empty_root_and_no_payload":
        write_root = False
        payload_kernel_ids = []
    elif mutation == "wrong_contract":
        trace.trace_contract_id = "wrong"
    elif mutation == "wrong_contract_and_no_payload":
        trace.trace_contract_id = "wrong"
        payload_kernel_ids = []
    elif mutation == "missing_capability":
        del trace.memory_payload_classes[-1]
    elif mutation == "duplicate_capability":
        trace.memory_payload_classes.append(1)
    elif mutation == "zero_launches":
        stream.ClearField("kernels")
    elif mutation == "device_map_mismatch":
        device.id = 1
    elif mutation == "stream_map_mismatch":
        stream.id = 1
    elif mutation == "duplicate_launch":
        duplicate = stream.kernels.add()
        duplicate.CopyFrom(kernel)
    elif mutation == "root_function_zero":
        kernel.function_unique_id = 0
    elif mutation == "zero_grid_axis":
        kernel.grid_dim.x = 0
    elif mutation == "zero_block_axis":
        kernel.block_dim.x = 0
    elif mutation == "legacy_shared_memory_nonzero":
        trace.trace_contract_id = "sm86_addr_carry_v1"
        kernel.ClearField("shared_memory_static_bytes")
        kernel.ClearField("shared_memory_dynamic_bytes")
        kernel.size_shared_memory = 4096
    elif mutation == "legacy_shared_memory_split_unadvertised":
        trace.trace_contract_id = "sm86_addr_carry_v1"
    elif mutation == "split_shared_memory_matching":
        trace.trace_contract_id = "sm86_addr_carry_shmem_split_v2"
        kernel.size_shared_memory = 6144
        kernel.shared_memory_static_bytes = 2048
        kernel.shared_memory_dynamic_bytes = 4096
    elif mutation == "split_shared_memory_zero":
        trace.trace_contract_id = "sm86_addr_carry_shmem_split_v2"
        kernel.shared_memory_static_bytes = 0
        kernel.shared_memory_dynamic_bytes = 0
    elif mutation == "split_shared_memory_static_only":
        trace.trace_contract_id = "sm86_addr_carry_shmem_split_v2"
        kernel.size_shared_memory = 2048
        kernel.shared_memory_static_bytes = 2048
        kernel.shared_memory_dynamic_bytes = 0
    elif mutation == "split_shared_memory_dynamic_only":
        trace.trace_contract_id = "sm86_addr_carry_shmem_split_v2"
        kernel.size_shared_memory = 4096
        kernel.shared_memory_static_bytes = 0
        kernel.shared_memory_dynamic_bytes = 4096
    elif mutation == "split_shared_memory_missing_both":
        kernel.ClearField("shared_memory_static_bytes")
        kernel.ClearField("shared_memory_dynamic_bytes")
    elif mutation == "split_shared_memory_missing_static":
        kernel.ClearField("shared_memory_static_bytes")
    elif mutation == "split_shared_memory_missing_dynamic":
        kernel.ClearField("shared_memory_dynamic_bytes")
    elif mutation == "split_shared_memory_mismatch":
        trace.trace_contract_id = "sm86_addr_carry_shmem_split_v2"
        kernel.size_shared_memory = 4096
        kernel.shared_memory_static_bytes = 1024
        kernel.shared_memory_dynamic_bytes = 2048
    elif mutation == "shared_memory_combined_negative":
        kernel.size_shared_memory = -1
    elif mutation == "split_shared_memory_wrong_wire_type":
        trace.trace_contract_id = "sm86_addr_carry_shmem_split_v2"
        kernel.shared_memory_dynamic_bytes = 0
        kernel_wire_suffix = b"\x5a\x00"
    elif mutation == "split_shared_memory_duplicate_field":
        trace.trace_contract_id = "sm86_addr_carry_shmem_split_v2"
        kernel.shared_memory_static_bytes = 0
        kernel.shared_memory_dynamic_bytes = 0
        kernel_wire_suffix = b"\x58\x00"
    elif mutation == "no_payload_files":
        payload_kernel_ids = []
    elif mutation == "launch_payload_missing":
        payload_kernel_ids = [9]
    elif mutation == "orphan_payload":
        payload_kernel_ids = [1, 9]
    elif mutation == "empty_payload_bytes":
        empty_payload = True
    elif mutation == "instruction_known_field_wrong_wire_type":
        instruction_known_field_wrong_wire_type = True
    elif mutation == "empty_payload_records":
        threadblock.ClearField("warps")
    elif mutation == "block_id_missing":
        threadblock.ClearField("block_id")
    elif mutation == "block_id_out_of_range":
        threadblock.block_id.x = 1
    elif mutation == "instruction_function_mismatch":
        instruction.function_unique_id = 2
    elif mutation == "function_mismatch_with_atomic_shape":
        operand = _make_memory(
            instruction,
            instruction_pb2,
            address_pb2,
            op_class=instruction_pb2.MEMORY_OP_CLASS_ATOMIC,
        )
        instruction.addresses.add().CopyFrom(operand)
        instruction.function_unique_id = 2
    elif mutation == "warpid_missing":
        instruction.ClearField("warpid_tb")
    elif mutation == "instruction_function_zero":
        instruction.function_unique_id = 0
    elif mutation == "active_mask_missing":
        instruction.ClearField("active_mask")
    elif mutation == "predicate_mask_missing":
        instruction.ClearField("predicate_mask")
    elif mutation == "vpc_missing":
        instruction.ClearField("vpc")
    elif mutation == "vpc_mismatch":
        instruction.vpc = 0x11
    elif mutation == "memory_data_width_precedence":
        instruction.memory_op_class = instruction_pb2.MEMORY_OP_CLASS_GLOBAL
        instruction.ClearField("data_width")
        instruction.ClearField("warpid_tb")
    elif mutation == "unknown_memory_class":
        instruction.memory_op_class = 99
        instruction.data_width = 4
    elif mutation == "unspecified_memory_class":
        instruction.memory_op_class = instruction_pb2.MEMORY_OP_CLASS_UNSPECIFIED
        instruction.data_width = 4
    elif mutation == "undeclared_payload_class":
        instruction.memory_op_class = instruction_pb2.MEMORY_OP_CLASS_TEXTURE
        instruction.data_width = 4
    elif mutation == "active_global_without_operand":
        instruction.memory_op_class = instruction_pb2.MEMORY_OP_CLASS_GLOBAL
        instruction.data_width = 4
        instruction.active_mask = instruction.predicate_mask = 1
    elif mutation == "zero_active_global_without_operand":
        instruction.memory_op_class = instruction_pb2.MEMORY_OP_CLASS_GLOBAL
        instruction.data_width = 4
    elif mutation == "atomic_missing_space":
        operand = _make_memory(
            instruction,
            instruction_pb2,
            address_pb2,
            op_class=instruction_pb2.MEMORY_OP_CLASS_ATOMIC,
        )
        operand.ClearField("address_space")
    elif mutation == "atomic_multiple_operands":
        operand = _make_memory(
            instruction,
            instruction_pb2,
            address_pb2,
            op_class=instruction_pb2.MEMORY_OP_CLASS_ATOMIC,
        )
        instruction.addresses.add().CopyFrom(operand)
    elif mutation == "zero_active_atomic_without_operand":
        instruction.memory_op_class = instruction_pb2.MEMORY_OP_CLASS_ATOMIC
        instruction.data_width = 4
    elif mutation == "async_copy_without_operand":
        instruction.memory_op_class = instruction_pb2.MEMORY_OP_CLASS_ASYNC_COPY
        instruction.data_width = 4
        instruction.active_mask = instruction.predicate_mask = 1
    elif mutation == "async_copy_missing_space":
        operand = _make_memory(
            instruction,
            instruction_pb2,
            address_pb2,
            op_class=instruction_pb2.MEMORY_OP_CLASS_ASYNC_COPY,
        )
        operand.ClearField("address_space")
    elif mutation == "explicit_unspecified_space":
        _make_memory(
            instruction,
            instruction_pb2,
            address_pb2,
            address_space=address_pb2.MEMORY_SPACE_UNSPECIFIED,
        )
    elif mutation == "unspecified_space_before_later_malformed_lane":
        operand = _make_memory(
            instruction,
            instruction_pb2,
            address_pb2,
            address_space=address_pb2.MEMORY_SPACE_UNSPECIFIED,
        )
        malformed = instruction.addresses.add()
        malformed.CopyFrom(operand)
        malformed.address_space = address_pb2.MEMORY_SPACE_GLOBAL
        del malformed.addrs[-1]
    elif mutation == "unknown_memory_space":
        _make_memory(
            instruction,
            instruction_pb2,
            address_pb2,
            address_space=99,
        )
    elif mutation == "lane_mask_inconsistent":
        _make_memory(
            instruction,
            instruction_pb2,
            address_pb2,
            op_class=instruction_pb2.MEMORY_OP_CLASS_SHARED,
            address_space=address_pb2.MEMORY_SPACE_SHARED,
            allocation_id=None,
            active_mask=0,
        )
    elif mutation == "list_all_wrong_length":
        _make_memory(
            instruction,
            instruction_pb2,
            address_pb2,
            op_class=instruction_pb2.MEMORY_OP_CLASS_SHARED,
            address_space=address_pb2.MEMORY_SPACE_SHARED,
            allocation_id=None,
            active_mask=0,
            list_length=31,
        )
    elif mutation == "lane_values_resource_limit":
        _make_memory(
            instruction,
            instruction_pb2,
            address_pb2,
            op_class=instruction_pb2.MEMORY_OP_CLASS_SHARED,
            address_space=address_pb2.MEMORY_SPACE_SHARED,
            allocation_id=None,
            active_mask=0,
            list_length=33,
        )
    elif mutation == "compressed_address_overflow":
        operand = _make_memory(
            instruction,
            instruction_pb2,
            address_pb2,
            op_class=instruction_pb2.MEMORY_OP_CLASS_SHARED,
            address_space=address_pb2.MEMORY_SPACE_SHARED,
            allocation_id=None,
            active_mask=3,
            list_length=0,
        )
        operand.compression_format = 1
        operand.base_address = 2**63
        operand.stride = 4
    elif mutation == "unspecified_access_kind":
        instruction.memory_op_class = instruction_pb2.MEMORY_OP_CLASS_GLOBAL
        instruction.data_width = 4
        instruction.memory_op_access_kind = (
            instruction_pb2.MEMORY_OP_ACCESS_KIND_UNSPECIFIED
        )
    elif mutation == "unknown_access_kind":
        instruction.memory_op_class = instruction_pb2.MEMORY_OP_CLASS_GLOBAL
        instruction.data_width = 4
        instruction.memory_op_access_kind = 99
    elif mutation == "unknown_access_kind_with_missing_allocation_id":
        _make_memory(
            instruction,
            instruction_pb2,
            address_pb2,
            allocation_id=None,
        )
        instruction.memory_op_access_kind = 99
    elif mutation == "unspecified_cache_op":
        instruction.memory_op_class = instruction_pb2.MEMORY_OP_CLASS_GLOBAL
        instruction.data_width = 4
        instruction.instruction_cache_op = (
            instruction_pb2.INSTRUCTION_CACHE_OP_UNSPECIFIED
        )
    elif mutation == "unknown_cache_op":
        instruction.memory_op_class = instruction_pb2.MEMORY_OP_CLASS_GLOBAL
        instruction.data_width = 4
        instruction.instruction_cache_op = 99
    elif mutation == "unknown_cache_op_with_missing_allocation_id":
        _make_memory(
            instruction,
            instruction_pb2,
            address_pb2,
            allocation_id=None,
        )
        instruction.instruction_cache_op = 99
    elif mutation == "metadata_only_nonmemory":
        instruction.data_width = 4
        instruction.memory_op_access_kind = instruction_pb2.MEMORY_OP_ACCESS_KIND_LOAD
        instruction.instruction_cache_op = (
            instruction_pb2.INSTRUCTION_CACHE_OP_LDG_E_CA
        )
    elif mutation == "allocation_id_missing":
        _make_memory(
            instruction,
            instruction_pb2,
            address_pb2,
            allocation_id=None,
        )
    elif mutation == "allocation_id_unknown":
        _make_memory(
            instruction,
            instruction_pb2,
            address_pb2,
            allocation_id=2,
        )
    elif mutation == "zero_active_explicit_allocation_without_sidecar":
        operand = _make_memory(
            instruction,
            instruction_pb2,
            address_pb2,
            allocation_id=1,
            active_mask=0,
        )
        operand.addrs[0] = -1
        write_allocations = False
    elif mutation == "lane_outside_allocation":
        _make_memory(
            instruction,
            instruction_pb2,
            address_pb2,
            address=0x3000,
        )
    elif mutation == "duplicate_allocation_id":
        allocations.append(dict(allocations[0]))
    elif mutation == "duplicate_allocation_object_key":
        allocations_raw = (
            '[{"allocation_id":1,"allocation_id":1,"context_id":1,'
            '"device_id":0,"base_address":"0x1000","size_bytes":4096,'
            '"alive":true}]'
        )
    elif mutation == "allocation_hex_digit_separator":
        allocations[0]["base_address"] = "0x1_000"
    elif mutation == "allocation_integer_token_too_long":
        allocations_raw = (
            '[{"allocation_id":'
            + "9" * 5_000
            + ',"context_id":1,"device_id":0,"base_address":"0x1000",'
            '"size_bytes":4096,"alive":true}]'
        )
    elif mutation == "stats_empty":
        stats = ""
    elif mutation == "stats_schema_invalid":
        stats = "other,value\n1,1\n"
    elif mutation == "stats_negative":
        stats = "total_insts,total_reported_insts\n-1,0\n"
    elif mutation == "stats_digit_separator":
        stats = "total_insts,total_reported_insts\n1_0,1\n"
    elif mutation == "stats_admitted_exceeds_expected":
        stats = "total_insts,total_reported_insts\n1,2\n"
    elif mutation == "stats_vacuous":
        stats = "total_insts,total_reported_insts\n0,0\n"
    elif mutation == "stats_row_exceeds_u64":
        stats = f"total_insts,total_reported_insts\n{2**64},1\n"
    elif mutation == "stats_total_overflow":
        stats = (
            "total_insts,total_reported_insts\n"
            f"{2**64 - 1},1\n"
            "1,1\n"
        )
    else:  # pragma: no cover - fixture/test lockstep assertion
        raise AssertionError(f"unimplemented golden mutation {mutation!r}")

    root_bytes = trace.SerializeToString() if write_root else b""
    if kernel_wire_suffix:
        root_bytes = _trace_with_kernel_suffix(trace, kernel_wire_suffix)
    if root_known_field_wrong_wire_type:
        root_bytes += b"\x08\x00"
    (trace_dir / "dynamic_trace.pb").write_bytes(root_bytes)
    extra = trace_dir / "extra_info"
    extra.mkdir()
    if write_allocations:
        (extra / "allocations.json").write_text(
            allocations_raw
            if allocations_raw is not None
            else json.dumps(allocations),
            encoding="utf-8",
        )
    (trace_dir / "stats.csv").write_text(stats, encoding="utf-8")
    threadblocks_root = trace_dir / "threadblocks"
    threadblocks_root.mkdir()
    for kernel_id in payload_kernel_ids:
        payload_path = (
            threadblocks_root
            / "device_0"
            / "stream_0"
            / f"kernel_{kernel_id}"
            / "0.pb"
        )
        payload_path.parent.mkdir(parents=True)
        if empty_payload:
            payload = b""
        elif instruction_known_field_wrong_wire_type:
            payload = _threadblock_with_instruction_suffix(
                threadblock, b"\x28\x00"
            )
        else:
            payload = threadblock.SerializeToString()
        payload_path.write_bytes(payload)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_python_wire_admission_matches_shared_typed_outcome(
    tmp_path: Path, case: dict[str, object]
) -> None:
    trace_dir = tmp_path / "trace"
    mutation = str(case["mutation"])
    _build_case(trace_dir, mutation)
    admission = (
        require_historical_trace_admission
        if mutation
        in {
            "legacy_shared_memory_nonzero",
            "legacy_shared_memory_split_unadvertised",
        }
        else require_trace_admission
    )
    try:
        admission(trace_dir)
        actual = None
    except TraceAdmissionError as error:
        actual = error.typed_reason
    assert actual == case["expected_typed_reason"]


def test_wire_admission_golden_has_unique_complete_mutations() -> None:
    mutations = [case["mutation"] for case in CASES]
    assert len(mutations) == len(set(mutations))
    assert len(CASES) >= 40
