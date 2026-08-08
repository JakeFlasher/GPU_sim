"""Allocation-free protobuf wire cardinality preflight.

Raw-byte limits do not bound decoded object amplification: repeated empty
submessages occupy two bytes each but allocate full protobuf objects. These
implementation-owned limits are checked before either Python protobuf or Prost
decodes forward trace artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

_MAX_GROUP_NESTING: Final = 64
_UINT32_MAX: Final = 0xFFFF_FFFF
_SIGN_EXTENDED_INT32_MIN: Final = 0xFFFF_FFFF_8000_0000
_KNOWN_FIELD_WIRE_TYPE_MISMATCH: Final = (
    "known protobuf field has incompatible wire type"
)
_UINT32_VARINT_OUT_OF_RANGE: Final = "protobuf uint32 field exceeds uint32 range"


class ProtobufWireMalformed(ValueError):
    """The byte stream is not structurally valid protobuf wire data."""


class ProtobufDecodedResourceLimitExceeded(ValueError):
    """A decoded repeated-field cardinality exceeds a local ceiling."""

    def __init__(self, resource: str, observed: int, limit: int) -> None:
        super().__init__(
            f"decoded resource {resource} exceeds implementation limit "
            f"{limit} (observed at least {observed})"
        )
        self.resource = resource
        self.observed = observed
        self.limit = limit


@dataclass(frozen=True)
class _RootLimits:
    devices: int = 65_536
    streams: int = 262_144
    kernels: int = 1_000_000
    ordered_events: int = 1_000_000
    memory_payload_classes: int = 64
    decoded_nodes: int = 2_000_000


@dataclass(frozen=True)
class _ThreadblockLimits:
    warps: int = 65_536
    instructions: int = 1_000_000
    addresses: int = 1_000_000
    lane_values: int = 8_000_000
    lane_values_per_address: int = 32
    decoded_nodes: int = 2_000_000


_ROOT_LIMITS: Final = _RootLimits()
_THREADBLOCK_LIMITS: Final = _ThreadblockLimits()


class _WireCursor:
    def __init__(self, raw: bytes | memoryview) -> None:
        self.raw = raw if isinstance(raw, memoryview) else memoryview(raw)
        self.offset = 0

    def empty(self) -> bool:
        return self.offset == len(self.raw)

    def read_varint(self) -> int:
        value = 0
        for byte_index in range(10):
            if self.offset >= len(self.raw):
                raise ProtobufWireMalformed("truncated varint")
            byte = self.raw[self.offset]
            self.offset += 1
            if byte_index == 9 and byte > 1:
                raise ProtobufWireMalformed("varint exceeds uint64")
            value |= (byte & 0x7F) << (byte_index * 7)
            if byte & 0x80 == 0:
                return value
        raise ProtobufWireMalformed("unterminated varint")

    def read_key(self) -> tuple[int, int]:
        key = self.read_varint()
        field_number = key >> 3
        if field_number == 0 or field_number > 0x1FFF_FFFF:
            raise ProtobufWireMalformed("invalid protobuf field number")
        wire_type = key & 0x07
        if wire_type > 5:
            raise ProtobufWireMalformed("invalid protobuf wire type")
        return field_number, wire_type

    def take(self, length: int) -> memoryview:
        end = self.offset + length
        if end > len(self.raw):
            raise ProtobufWireMalformed(
                "length-delimited field exceeds enclosing message"
            )
        value = self.raw[self.offset : end]
        self.offset = end
        return value

    def skip_group(self, expected_field: int, depth: int) -> None:
        if depth > _MAX_GROUP_NESTING:
            raise ProtobufWireMalformed("group nesting exceeds preflight limit")
        while True:
            if self.empty():
                raise ProtobufWireMalformed("unterminated group")
            field_number, wire_type = self.read_key()
            if wire_type == 4:
                if field_number != expected_field:
                    raise ProtobufWireMalformed(
                        "group end field does not match group start"
                    )
                return
            self.skip_value(field_number, wire_type, depth)

    def skip_value(
        self, field_number: int, wire_type: int, depth: int
    ) -> int | memoryview | None:
        if wire_type == 0:
            return self.read_varint()
        if wire_type == 1:
            self.take(8)
            return None
        if wire_type == 2:
            return self.take(self.read_varint())
        if wire_type == 3:
            self.skip_group(field_number, depth + 1)
            return None
        if wire_type == 4:
            raise ProtobufWireMalformed("unexpected group end")
        if wire_type == 5:
            self.take(4)
            return None
        raise ProtobufWireMalformed("invalid protobuf wire type")

    def next_field(self) -> tuple[int, int, int | memoryview | None] | None:
        if self.empty():
            return None
        number, wire_type = self.read_key()
        payload = self.skip_value(number, wire_type, 0)
        return number, wire_type, payload


def _require_varint(field: tuple[int, int, int | memoryview | None]) -> None:
    _number, wire_type, _payload = field
    if wire_type != 0:
        raise ProtobufWireMalformed(_KNOWN_FIELD_WIRE_TYPE_MISMATCH)


def _require_uint32(field: tuple[int, int, int | memoryview | None]) -> None:
    _number, wire_type, payload = field
    if wire_type != 0 or not isinstance(payload, int):
        raise ProtobufWireMalformed(_KNOWN_FIELD_WIRE_TYPE_MISMATCH)
    if payload > _UINT32_MAX:
        raise ProtobufWireMalformed(_UINT32_VARINT_OUT_OF_RANGE)


def _require_shared_memory_int32(
    field: tuple[int, int, int | memoryview | None],
) -> None:
    _number, wire_type, payload = field
    if wire_type != 0 or not isinstance(payload, int):
        raise ProtobufWireMalformed(_KNOWN_FIELD_WIRE_TYPE_MISMATCH)
    # Protobuf int32 encodes negative values as sign-extended ten-byte
    # varints. Preserve those for the post-decode nonnegative semantic check,
    # but reject the over-wide middle range that can truncate to a valid
    # nonnegative shared-memory size.
    if _UINT32_MAX < payload < _SIGN_EXTENDED_INT32_MIN:
        raise ProtobufWireMalformed(_UINT32_VARINT_OUT_OF_RANGE)


def _require_length_delimited(
    field: tuple[int, int, int | memoryview | None],
) -> memoryview:
    _number, wire_type, payload = field
    if wire_type != 2 or not isinstance(payload, memoryview):
        raise ProtobufWireMalformed(_KNOWN_FIELD_WIRE_TYPE_MISMATCH)
    return payload


def _require_repeated_varint(
    field: tuple[int, int, int | memoryview | None],
) -> memoryview | None:
    _number, wire_type, payload = field
    if wire_type == 0:
        return None
    if wire_type == 2 and isinstance(payload, memoryview):
        return payload
    raise ProtobufWireMalformed(_KNOWN_FIELD_WIRE_TYPE_MISMATCH)


def _bump(
    counts: dict[str, int],
    key: str,
    amount: int,
    limit: int,
    resource: str,
) -> None:
    observed = counts.get(key, 0) + amount
    if observed > limit:
        raise ProtobufDecodedResourceLimitExceeded(resource, observed, limit)
    counts[key] = observed


def _scan_packed_root_classes(
    raw: memoryview,
    counts: dict[str, int],
    limits: _RootLimits,
) -> None:
    cursor = _WireCursor(raw)
    while not cursor.empty():
        cursor.read_varint()
        _bump(
            counts,
            "memory_payload_classes",
            1,
            limits.memory_payload_classes,
            "memory_payload_classes",
        )


def _scan_trace(
    raw: bytes | memoryview,
    counts: dict[str, int],
    limits: _RootLimits,
) -> None:
    cursor = _WireCursor(raw)
    while (field := cursor.next_field()) is not None:
        number, _wire_type, _payload = field
        if number in {1, 3, 7}:
            _require_length_delimited(field)
        elif number in {2, 4, 5}:
            _require_varint(field)
        elif number == 6:
            map_entry = _require_length_delimited(field)
            _bump(counts, "devices", 1, limits.devices, "gpu devices")
            _bump(
                counts,
                "decoded_nodes",
                1,
                limits.decoded_nodes,
                "root decoded nodes",
            )
            _scan_device_map_entry(map_entry, counts, limits)
        elif number == 8:
            packed = _require_repeated_varint(field)
            if packed is None:
                _bump(
                    counts,
                    "memory_payload_classes",
                    1,
                    limits.memory_payload_classes,
                    "memory_payload_classes",
                )
            else:
                _scan_packed_root_classes(packed, counts, limits)


def _scan_device_map_entry(
    raw: memoryview,
    counts: dict[str, int],
    limits: _RootLimits,
) -> None:
    cursor = _WireCursor(raw)
    while (field := cursor.next_field()) is not None:
        number, _wire_type, _payload = field
        if number == 1:
            _require_varint(field)
        elif number == 2:
            device = _require_length_delimited(field)
            _bump(
                counts,
                "decoded_nodes",
                1,
                limits.decoded_nodes,
                "root decoded nodes",
            )
            _scan_device(device, counts, limits)


def _scan_device(
    raw: memoryview,
    counts: dict[str, int],
    limits: _RootLimits,
) -> None:
    cursor = _WireCursor(raw)
    while (field := cursor.next_field()) is not None:
        number, _wire_type, _payload = field
        if number == 1:
            _require_uint32(field)
        elif number == 2:
            map_entry = _require_length_delimited(field)
            _bump(counts, "streams", 1, limits.streams, "CUDA streams")
            _bump(
                counts,
                "decoded_nodes",
                1,
                limits.decoded_nodes,
                "root decoded nodes",
            )
            _scan_stream_map_entry(map_entry, counts, limits)


def _scan_stream_map_entry(
    raw: memoryview,
    counts: dict[str, int],
    limits: _RootLimits,
) -> None:
    cursor = _WireCursor(raw)
    while (field := cursor.next_field()) is not None:
        number, _wire_type, _payload = field
        if number == 1:
            _require_varint(field)
        elif number == 2:
            stream = _require_length_delimited(field)
            _bump(
                counts,
                "decoded_nodes",
                1,
                limits.decoded_nodes,
                "root decoded nodes",
            )
            _scan_stream(stream, counts, limits)


def _scan_stream(
    raw: memoryview,
    counts: dict[str, int],
    limits: _RootLimits,
) -> None:
    cursor = _WireCursor(raw)
    while (field := cursor.next_field()) is not None:
        number, _wire_type, _payload = field
        if number == 1:
            _require_varint(field)
        elif number == 2:
            kernel = _require_length_delimited(field)
            _bump(counts, "kernels", 1, limits.kernels, "kernel launches")
            _bump(
                counts,
                "decoded_nodes",
                1,
                limits.decoded_nodes,
                "root decoded nodes",
            )
            _scan_kernel(kernel, counts, limits)
        elif number == 3:
            _require_length_delimited(field)
            _bump(
                counts,
                "ordered_events",
                1,
                limits.ordered_events,
                "ordered CUDA events",
            )
            _bump(
                counts,
                "decoded_nodes",
                1,
                limits.decoded_nodes,
                "root decoded nodes",
            )


def _scan_kernel(
    raw: memoryview,
    counts: dict[str, int],
    limits: _RootLimits,
) -> None:
    cursor = _WireCursor(raw)
    split_fields: set[int] = set()
    while (field := cursor.next_field()) is not None:
        number, _wire_type, _payload = field
        if number == 1 or number == 3 or 5 <= number <= 7:
            _require_varint(field)
        # Field 4 is a legacy int32, but valid shared-memory sizes are
        # nonnegative and participate in the uint32 split-field contract.
        # Rejecting an over-wide raw varint here prevents generated decoders
        # from truncating it before the aggregate/split equality check.
        elif number == 4:
            _require_shared_memory_int32(field)
        elif number == 2:
            _require_length_delimited(field)
        elif 8 <= number <= 10:
            dimensions = _require_length_delimited(field)
            _bump(
                counts,
                "decoded_nodes",
                1,
                limits.decoded_nodes,
                "root decoded nodes",
            )
            _scan_dim3d(dimensions)
        elif 11 <= number <= 12:
            _require_uint32(field)
            if number in split_fields:
                raise ProtobufWireMalformed(
                    "duplicate shared-memory split protobuf field"
                )
            split_fields.add(number)


def _scan_dim3d(raw: memoryview) -> None:
    cursor = _WireCursor(raw)
    while (field := cursor.next_field()) is not None:
        number, _wire_type, _payload = field
        if 1 <= number <= 3:
            _require_varint(field)


def preflight_trace_protobuf(raw: bytes) -> None:
    """Bound root repeated structures before protobuf decoding."""

    _scan_trace(raw, {}, _ROOT_LIMITS)


def _scan_threadblock(
    raw: bytes | memoryview,
    counts: dict[str, int],
    limits: _ThreadblockLimits,
) -> None:
    cursor = _WireCursor(raw)
    while (field := cursor.next_field()) is not None:
        number, _wire_type, _payload = field
        if number == 1:
            block_id = _require_length_delimited(field)
            _bump(
                counts,
                "decoded_nodes",
                1,
                limits.decoded_nodes,
                "threadblock decoded nodes",
            )
            _scan_dim3d(block_id)
        elif number == 2:
            map_entry = _require_length_delimited(field)
            _bump(counts, "warps", 1, limits.warps, "threadblock warps")
            _bump(
                counts,
                "decoded_nodes",
                1,
                limits.decoded_nodes,
                "threadblock decoded nodes",
            )
            _scan_warp_map_entry(map_entry, counts, limits)


def _scan_warp_map_entry(
    raw: memoryview,
    counts: dict[str, int],
    limits: _ThreadblockLimits,
) -> None:
    cursor = _WireCursor(raw)
    while (field := cursor.next_field()) is not None:
        number, _wire_type, _payload = field
        if number == 1:
            _require_varint(field)
        elif number == 2:
            warp = _require_length_delimited(field)
            _bump(
                counts,
                "decoded_nodes",
                1,
                limits.decoded_nodes,
                "threadblock decoded nodes",
            )
            _scan_warp(warp, counts, limits)


def _scan_warp(
    raw: memoryview,
    counts: dict[str, int],
    limits: _ThreadblockLimits,
) -> None:
    cursor = _WireCursor(raw)
    while (field := cursor.next_field()) is not None:
        number, _wire_type, _payload = field
        if number == 1:
            _require_varint(field)
        elif number == 2:
            instruction = _require_length_delimited(field)
            _bump(
                counts,
                "instructions",
                1,
                limits.instructions,
                "threadblock instructions",
            )
            _bump(
                counts,
                "decoded_nodes",
                1,
                limits.decoded_nodes,
                "threadblock decoded nodes",
            )
            _scan_instruction(instruction, counts, limits)


def _scan_instruction(
    raw: memoryview,
    counts: dict[str, int],
    limits: _ThreadblockLimits,
) -> None:
    cursor = _WireCursor(raw)
    while (field := cursor.next_field()) is not None:
        number, _wire_type, _payload = field
        if 1 <= number <= 3:
            _require_uint32(field)
        elif number == 4 or 6 <= number <= 13:
            _require_varint(field)
        elif number == 5:
            address = _require_length_delimited(field)
            _bump(
                counts,
                "addresses",
                1,
                limits.addresses,
                "threadblock address records",
            )
            _bump(
                counts,
                "decoded_nodes",
                1,
                limits.decoded_nodes,
                "threadblock decoded nodes",
            )
            _scan_address(address, counts, limits)


def _bump_lane_value(
    address_counts: dict[str, int],
    counts: dict[str, int],
    limits: _ThreadblockLimits,
) -> None:
    _bump(
        address_counts,
        "lane_values",
        1,
        limits.lane_values_per_address,
        "lane values per address record",
    )
    _bump(
        counts,
        "lane_values",
        1,
        limits.lane_values,
        "threadblock lane values",
    )


def _scan_packed_lane_values(
    raw: memoryview,
    address_counts: dict[str, int],
    counts: dict[str, int],
    limits: _ThreadblockLimits,
) -> None:
    cursor = _WireCursor(raw)
    while not cursor.empty():
        cursor.read_varint()
        _bump_lane_value(address_counts, counts, limits)


def _scan_address(
    raw: memoryview,
    counts: dict[str, int],
    limits: _ThreadblockLimits,
) -> None:
    cursor = _WireCursor(raw)
    address_counts: dict[str, int] = {}
    while (field := cursor.next_field()) is not None:
        number, _wire_type, _payload = field
        if number in {2, 7}:
            _require_uint32(field)
        elif number == 1 or 3 <= number <= 5 or number == 8:
            _require_varint(field)
        elif number == 6:
            packed = _require_repeated_varint(field)
            if packed is None:
                _bump_lane_value(address_counts, counts, limits)
            else:
                _scan_packed_lane_values(packed, address_counts, counts, limits)


def preflight_threadblock_protobuf(raw: bytes) -> None:
    """Bound one threadblock's decoded cardinality before protobuf decoding."""

    _scan_threadblock(raw, {}, _THREADBLOCK_LIMITS)
