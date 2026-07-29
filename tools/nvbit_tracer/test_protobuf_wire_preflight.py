from __future__ import annotations

import pytest

from tools.nvbit_tracer.protobuf_wire_preflight import (
    ProtobufDecodedResourceLimitExceeded,
    ProtobufWireMalformed,
    _RootLimits,
    _ThreadblockLimits,
    _scan_threadblock,
    _scan_trace,
    preflight_threadblock_protobuf,
    preflight_trace_protobuf,
)


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


def _varint_field(number: int, value: int) -> bytes:
    return bytes((number << 3,)) + _varint(value)


def test_known_root_field_rejects_incompatible_wire_type() -> None:
    with pytest.raises(ProtobufWireMalformed):
        preflight_trace_protobuf(_varint_field(1, 0))


def test_known_nested_field_rejects_incompatible_wire_type() -> None:
    instruction = _varint_field(5, 0)
    warp = _message_field(2, instruction)
    warp_entry = _message_field(2, warp)
    threadblock = _message_field(2, warp_entry)

    with pytest.raises(ProtobufWireMalformed):
        preflight_threadblock_protobuf(threadblock)


def test_repeated_empty_kernel_messages_are_bounded_before_decode() -> None:
    stream = b"".join(_message_field(2, b"") for _ in range(4))
    stream_entry = _message_field(2, stream)
    device = _message_field(2, stream_entry)
    device_entry = _message_field(2, device)
    trace = _message_field(6, device_entry)
    limits = _RootLimits(
        devices=8,
        streams=8,
        kernels=3,
        ordered_events=8,
        memory_payload_classes=8,
        decoded_nodes=32,
    )

    with pytest.raises(ProtobufDecodedResourceLimitExceeded) as captured:
        _scan_trace(trace, {}, limits)

    assert captured.value.resource == "kernel launches"
    assert captured.value.observed == 4
    assert captured.value.limit == 3


def test_repeated_empty_instruction_messages_are_bounded_per_file() -> None:
    warp = b"".join(_message_field(2, b"") for _ in range(4))
    warp_entry = _message_field(2, warp)
    threadblock = _message_field(2, warp_entry)
    limits = _ThreadblockLimits(
        warps=8,
        instructions=3,
        addresses=8,
        lane_values=64,
        lane_values_per_address=32,
        decoded_nodes=32,
    )

    with pytest.raises(ProtobufDecodedResourceLimitExceeded) as captured:
        _scan_threadblock(threadblock, {}, limits)

    assert captured.value.resource == "threadblock instructions"
    assert captured.value.observed == 4
    assert captured.value.limit == 3
