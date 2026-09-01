"""Trace-derived execution and traffic invariants for direct replays.

This module deliberately derives only conservation quantities that are present
in the address-carrying trace.  It does not infer cache hit rates or timing
from NCU labels.  The replay validator uses these values as an independent
oracle for instruction counts, operation counts, and logical 32-byte sectors.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any

from tools.nvbit_tracer.proto_binding_loader import load_dynamic_trace_bindings


_DEVICE_RE = re.compile(r"device_(-?\d+)$")
_STREAM_RE = re.compile(r"stream_(-?\d+)$")
_KERNEL_RE = re.compile(r"kernel_(-?\d+)$")
_LANE_INACTIVE_SENTINEL = -1

# Values mirror the committed dynamic-trace protobuf enums.  Keeping these
# local avoids treating a future enum append as a different traffic class.
_GLOBAL = 1
_SHARED = 2
_LOCAL = 3
_ATOMIC = 6
_ASYNC_COPY = 8
_LOAD = 1
_STORE = 2
_ATOMIC_ACCESS = 3
_ASYNC_ACCESS = 4


@dataclass(frozen=True, slots=True)
class LaunchTraceInvariants:
    device_id: int
    stream_id: int
    kernel_id: int
    blocks: int
    warps: int
    instructions: int
    memory_instructions: int
    load_instructions: int
    store_instructions: int
    atomic_instructions: int
    async_copy_instructions: int
    global_load_instructions: int
    global_store_instructions: int
    global_atomic_instructions: int
    global_load_sectors: int
    global_store_sectors: int
    global_atomic_sectors: int
    async_source_sectors: int
    local_load_sectors: int
    local_store_sectors: int
    shared_sectors: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TraceInvariants:
    trace_dir: str
    launches: tuple[LaunchTraceInvariants, ...]

    def by_identity(self) -> dict[tuple[int, int, int], LaunchTraceInvariants]:
        return {
            (row.device_id, row.stream_id, row.kernel_id): row
            for row in self.launches
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_dir": self.trace_dir,
            "launches": [row.as_dict() for row in self.launches],
        }


def _identity(path: Path, trace_root: Path) -> tuple[int, int, int]:
    parts = path.relative_to(trace_root / "threadblocks").parts
    if len(parts) < 4:
        raise ValueError(f"threadblock path is too shallow: {path}")
    matches = (
        _DEVICE_RE.fullmatch(parts[0]),
        _STREAM_RE.fullmatch(parts[1]),
        _KERNEL_RE.fullmatch(parts[2]),
    )
    if any(match is None for match in matches):
        raise ValueError(f"threadblock path has malformed launch identity: {path}")
    assert matches[0] is not None and matches[1] is not None and matches[2] is not None
    return tuple(int(match.group(1)) for match in matches)  # type: ignore[return-value]


def _active_lanes(mask: int) -> list[int]:
    return [lane for lane in range(32) if (mask >> lane) & 1]


def _address_values(address: Any, active_lanes: list[int]) -> list[int]:
    """Expand one address operand without changing the trace contract."""

    compression = int(address.compression_format)
    if compression == 0:  # list_all
        return [int(address.addrs[lane]) for lane in active_lanes]
    if compression == 1:  # base_stride
        if not active_lanes:
            return []
        first = active_lanes[0]
        return [
            int(address.base_address) + (lane - first) * int(address.stride)
            for lane in active_lanes
        ]
    if compression == 2:  # base_delta
        current = int(address.base_address)
        values: list[int] = []
        for index, _lane in enumerate(active_lanes):
            if index:
                current += int(address.addrs[index - 1])
            values.append(current)
        return values
    raise ValueError(f"unsupported address compression format {compression}")


def _operand_sectors(instruction: Any, active_lanes: list[int], index: int) -> int:
    if index >= len(instruction.addresses):
        return 0
    address = instruction.addresses[index]
    operand_lanes = (
        _active_lanes(int(address.lane_mask))
        if address.HasField("lane_mask")
        else active_lanes
    )
    values = _address_values(address, operand_lanes)
    width = int(address.data_width or instruction.data_width or 1)
    return len(
        {
            sector
            for value in values
            if value != _LANE_INACTIVE_SENTINEL
            for sector in range(value // 32, (value + width - 1) // 32 + 1)
        }
    )


def _atomic_operand_transactions(
    instruction: Any, active_lanes: list[int], index: int
) -> int:
    """Count sm_86 atomic/reduction sector transactions.

    Atomics preserve same-word alias multiplicity: one transaction is emitted
    for each lane alias at the most-contended address in a touched sector.
    This is intentionally different from ordinary load/store coalescing, which
    uses the instruction-wide unique sector union.
    """

    if index >= len(instruction.addresses):
        return 0
    address = instruction.addresses[index]
    operand_lanes = (
        _active_lanes(int(address.lane_mask))
        if address.HasField("lane_mask")
        else active_lanes
    )
    values = _address_values(address, operand_lanes)
    width = int(address.data_width or instruction.data_width or 1)
    intervals = [
        (value, value // 32, (value + width - 1) // 32)
        for value in values
        if value != _LANE_INACTIVE_SENTINEL
    ]
    sectors = sorted(
        {
            sector
            for _base, first, last in intervals
            for sector in range(first, last + 1)
        }
    )
    total = 0
    for sector in sectors:
        aliases: dict[int, int] = {}
        for base, first, last in intervals:
            if first <= sector <= last:
                aliases[base] = aliases.get(base, 0) + 1
        total += max(aliases.values(), default=0)
    return total


@cache
def derive_trace_invariants(trace_dir: Path) -> TraceInvariants:
    """Derive per-launch instruction and logical-sector conservation data.

    The result is cached by canonical trace path for the duration of one
    validator process.  Campaign candidate variants share a trace, so the
    expensive protobuf walk occurs once per capture rather than once per
    replay pair.
    """

    trace_dir = trace_dir.resolve()
    bindings = load_dynamic_trace_bindings()
    rows: dict[tuple[int, int, int], dict[str, int]] = {}
    payload_root = trace_dir / "threadblocks"
    payloads = sorted(payload_root.rglob("d_*.pb"))
    if not payloads:
        raise ValueError(f"trace has no d_*.pb payloads: {trace_dir}")
    for payload in payloads:
        identity = _identity(payload, trace_dir)
        row = rows.setdefault(
            identity,
            {
                "blocks": 0,
                "warps": 0,
                "instructions": 0,
                "memory_instructions": 0,
                "load_instructions": 0,
                "store_instructions": 0,
                "atomic_instructions": 0,
                "async_copy_instructions": 0,
                "global_load_instructions": 0,
                "global_store_instructions": 0,
                "global_atomic_instructions": 0,
                "global_load_sectors": 0,
                "global_store_sectors": 0,
                "global_atomic_sectors": 0,
                "async_source_sectors": 0,
                "local_load_sectors": 0,
                "local_store_sectors": 0,
                "shared_sectors": 0,
            },
        )
        row["blocks"] += 1
        threadblock = bindings.threadblock_pb2.threadblock()
        threadblock.ParseFromString(payload.read_bytes())
        row["warps"] += len(threadblock.warps)
        for _warp_id, warp in threadblock.warps.items():
            row["instructions"] += len(warp.instructions)
            for instruction in warp.instructions:
                if not instruction.HasField("memory_op_class"):
                    continue
                row["memory_instructions"] += 1
                and_mask = int(instruction.active_mask) & int(instruction.predicate_mask)
                active_lanes = _active_lanes(and_mask)
                if not active_lanes:
                    continue
                op_class = int(instruction.memory_op_class)
                access_kind = (
                    int(instruction.memory_op_access_kind)
                    if instruction.HasField("memory_op_access_kind")
                    else 0
                )
                if access_kind == _LOAD:
                    row["load_instructions"] += 1
                elif access_kind == _STORE:
                    row["store_instructions"] += 1
                elif access_kind == _ATOMIC_ACCESS:
                    row["atomic_instructions"] += 1
                elif access_kind == _ASYNC_ACCESS:
                    row["async_copy_instructions"] += 1

                if op_class == _ASYNC_COPY:
                    row["async_source_sectors"] += _operand_sectors(
                        instruction, active_lanes, 0
                    )
                    continue
                if op_class == _ATOMIC or access_kind == _ATOMIC_ACCESS:
                    sectors = sum(
                        _atomic_operand_transactions(instruction, active_lanes, index)
                        for index in range(len(instruction.addresses))
                    )
                else:
                    sectors = sum(
                        _operand_sectors(instruction, active_lanes, index)
                        for index in range(len(instruction.addresses))
                    )
                if op_class == _GLOBAL:
                    if access_kind == _LOAD:
                        row["global_load_instructions"] += 1
                        row["global_load_sectors"] += sectors
                    elif access_kind == _STORE:
                        row["global_store_instructions"] += 1
                        row["global_store_sectors"] += sectors
                    elif access_kind == _ATOMIC_ACCESS:
                        row["global_atomic_instructions"] += 1
                        row["global_atomic_sectors"] += sectors
                elif op_class == _ATOMIC:
                    row["global_atomic_instructions"] += 1
                    row["global_atomic_sectors"] += sectors
                elif op_class == _LOCAL:
                    if access_kind == _LOAD:
                        row["local_load_sectors"] += sectors
                    elif access_kind == _STORE:
                        row["local_store_sectors"] += sectors
                elif op_class == _SHARED:
                    row["shared_sectors"] += sectors

    launches = tuple(
        LaunchTraceInvariants(
            device_id=identity[0],
            stream_id=identity[1],
            kernel_id=identity[2],
            **values,
        )
        for identity, values in sorted(rows.items())
    )
    return TraceInvariants(trace_dir=str(trace_dir), launches=launches)
