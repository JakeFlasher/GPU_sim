"""Small read-only decoder for the NVBit address wire, shared by SM120 tools.

No artifact admission or digest machinery: generated protobufs plus the three
producer compression formats. Sectors are coalesced within each operand.
"""
from __future__ import annotations



def load_bindings():
    from tools.nvbit_tracer.proto_binding_loader import load_dynamic_trace_bindings

    return load_dynamic_trace_bindings()


def present(message, field):
    try:
        return message.HasField(field)
    except (ValueError, AttributeError):
        return False


def lane_addresses(address, instruction):
    if not present(instruction, "active_mask") or not present(instruction, "predicate_mask"):
        raise ValueError("address record lacks executing-lane masks")
    mask = int(instruction.active_mask) & int(instruction.predicate_mask)
    if present(address, "lane_mask"):
        operand_mask = int(address.lane_mask)
        if operand_mask & ~mask:
            raise ValueError("operand mask widens executing-lane mask")
        mask = operand_mask  # explicit zero must remain zero (ignore-src)
    lanes = [i for i in range(32) if mask & (1 << i)]
    form = int(address.compression_format)
    data = address.addrs
    if form == 0:
        if len(data) != 32:
            raise ValueError("list-all address requires 32 lane slots")
        values = [int(data[i]) for i in lanes]
    elif form == 1:
        values = [int(address.base_address) + k * int(address.stride) for k in range(len(lanes))]
    elif form == 2:
        if len(data) != max(0, len(lanes) - 1):
            raise ValueError("base-delta address count disagrees with operand mask")
        values = []
        current = int(address.base_address)
        for k in range(len(lanes)):
            if k:
                current += int(data[k - 1])  # deltas are cumulative, not absolute
            values.append(current)
    else:
        raise ValueError(f"unsupported address compression {form}")
    width = int(address.transfer_width) if present(address, "transfer_width") else int(address.data_width)
    if width <= 0:
        width = int(instruction.data_width)
    if width <= 0 or width > 128:
        raise ValueError(f"invalid per-lane byte width {width}")
    if any(value < 0 or value + width > (1 << 63) for value in values):
        raise ValueError("executing lane has invalid address")
    return list(zip(lanes, values)), width


def operand_sectors(address, instruction):
    lanes, width = lane_addresses(address, instruction)
    sectors = set()
    for _, value in lanes:
        sectors.update(range(value >> 5, ((value + width - 1) >> 5) + 1))
    return sorted(sectors), len(lanes)
