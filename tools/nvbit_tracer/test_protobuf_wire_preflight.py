from __future__ import annotations

import pytest

from tools.nvbit_tracer.protobuf_wire_preflight import (
    ProtobufDecodedResourceLimitExceeded,
    ProtobufWireMalformed,
    _RootLimits,
    _ThreadblockLimits,
    _scan_address,
    _scan_device,
    _scan_instruction,
    _scan_kernel,
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


def _trace_with_kernel(kernel: bytes) -> bytes:
    stream = _message_field(2, kernel)
    stream_entry = _message_field(2, stream)
    device = _message_field(2, stream_entry)
    device_entry = _message_field(2, device)
    return _message_field(6, device_entry)


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


@pytest.mark.parametrize("field_number", [4, 11, 12])
@pytest.mark.parametrize("value", [1 << 32, (1 << 32) + 4_096, 1 << 63])
def test_kernel_32_bit_fields_reject_overwide_varints_before_decode(
    field_number: int, value: int
) -> None:
    with pytest.raises(
        ProtobufWireMalformed,
        match="protobuf uint32 field exceeds uint32 range",
    ):
        _scan_kernel(_varint_field(field_number, value), {}, _RootLimits())


def test_uint32_kernel_fields_reject_uint64_max() -> None:
    for field_number in (11, 12):
        with pytest.raises(
            ProtobufWireMalformed,
            match="protobuf uint32 field exceeds uint32 range",
        ):
            _scan_kernel(
                _varint_field(field_number, (1 << 64) - 1), {}, _RootLimits()
            )


def test_legacy_int32_negative_encoding_reaches_semantic_validation() -> None:
    _scan_kernel(_varint_field(4, (1 << 64) - 1), {}, _RootLimits())


def test_overwide_legacy_shared_memory_cannot_truncate_to_valid_value() -> None:
    kernel = _varint_field(4, (1 << 32) + 4_096)
    with pytest.raises(
        ProtobufWireMalformed,
        match="protobuf uint32 field exceeds uint32 range",
    ):
        preflight_trace_protobuf(_trace_with_kernel(kernel))


def test_uint32_fields_accept_max_boundary() -> None:
    max_uint32 = (1 << 32) - 1
    kernel = _varint_field(11, max_uint32) + _varint_field(12, max_uint32)
    _scan_kernel(kernel, {}, _RootLimits())
    _scan_device(_varint_field(1, max_uint32), {}, _RootLimits())

    for field_number in range(1, 4):
        _scan_instruction(
            _varint_field(field_number, max_uint32), {}, _ThreadblockLimits()
        )
    for field_number in (2, 7):
        _scan_address(
            _varint_field(field_number, max_uint32), {}, _ThreadblockLimits()
        )


def test_non_kernel_uint32_fields_reject_overwide_varints() -> None:
    overwide = 1 << 32
    with pytest.raises(ProtobufWireMalformed):
        _scan_device(_varint_field(1, overwide), {}, _RootLimits())
    for field_number in range(1, 4):
        with pytest.raises(ProtobufWireMalformed):
            _scan_instruction(
                _varint_field(field_number, overwide), {}, _ThreadblockLimits()
            )
    for field_number in (2, 7):
        with pytest.raises(ProtobufWireMalformed):
            _scan_address(
                _varint_field(field_number, overwide), {}, _ThreadblockLimits()
            )


def test_split_fields_reject_duplicates_and_wrong_wire_types() -> None:
    duplicate = _varint_field(11, 0) + _varint_field(11, 1)
    with pytest.raises(
        ProtobufWireMalformed,
        match="duplicate shared-memory split protobuf field",
    ):
        _scan_kernel(duplicate, {}, _RootLimits())

    with pytest.raises(
        ProtobufWireMalformed,
        match="known protobuf field has incompatible wire type",
    ):
        _scan_kernel(_message_field(12, b""), {}, _RootLimits())


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
