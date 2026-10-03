#!/usr/bin/env python3
"""Extract launch-scoped, observed tail facts without simulator or NCU inputs.

Warp program order is not inter-warp execution order. Static SASS scope tokens
are not ownership/read-from or hardware completion observations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
# Ordinary generated protobuf imports, not a hash/authority workflow.
sys.path.insert(0, str(REPO_ROOT / "workloads/simbench_bw/_pb_generated"))
import threadblock_pb2
import trace_pb2
from tools.nvbit_tracer.import_trace_contract import _reconstruct_lane_vector

SCHEMA = "sm86_tail_facts_v1"
WAIT = re.compile(r"^DEPBAR\.LE\s+SB0,\s*(0x[0-9a-fA-F]+|[0-9]+)(?:\s|,|$)")


def _identity(path: Path, root: Path) -> tuple[int, int, int]:
    parts = path.relative_to(root / "threadblocks").parts
    if len(parts) != 4:
        raise ValueError(f"unexpected payload path: {path}")
    result = []
    for prefix, part in zip(("device", "stream", "kernel"), parts[:3]):
        match = re.fullmatch(prefix + r"_(-?\d+)", part)
        if match is None:
            raise ValueError(f"malformed launch path: {path}")
        result.append(int(match[1]))
    return tuple(result)


def _present(message: Any, field: str) -> bool:
    return message.HasField(field)


def _async_protocol(instructions: Any, opcodes: dict) -> Counter:
    """Per-thread logical commit/wait relation; no service cycles or silicon order.

    Empty commits remain in the prefix: a wait immediate counts recent groups,
    not recent *nonempty* groups. This pass is independent of SmCore's queues.
    """
    opened = [0] * 32
    groups = [deque() for _ in range(32)]
    facts = Counter(copy_lane_operations=0, committed_lane_groups=0,
                    empty_lane_groups=0, waited_lane_groups=0, waited_lane_copies=0,
                    terminal_uncommitted_lane_copies=0, terminal_unwaited_lane_groups=0,
                    terminal_unwaited_lane_copies=0, masked_protocol_records_unknown=0)
    for instruction in instructions:
        opcode = opcodes[(int(instruction.function_unique_id), int(instruction.pc))]
        head = opcode.split()[0].split(".")[0]
        wait = WAIT.match(opcode)
        if head not in {"LDGSTS", "LDGDEPBAR"} and not wait:
            continue
        if not all(_present(instruction, f) for f in ("active_mask", "predicate_mask")):
            facts["masked_protocol_records_unknown"] += 1
            continue
        mask = instruction.active_mask & instruction.predicate_mask
        for lane in range(32):
            if not (mask >> lane) & 1:
                continue
            if head == "LDGSTS":
                opened[lane] += 1
                facts["copy_lane_operations"] += 1
            elif head == "LDGDEPBAR":
                groups[lane].append(opened[lane])
                facts["committed_lane_groups"] += 1
                facts["empty_lane_groups"] += opened[lane] == 0
                opened[lane] = 0
            elif wait:
                retained = int(wait[1], 0) if wait[1].startswith("0x") else int(wait[1])
                while len(groups[lane]) > retained:
                    facts["waited_lane_groups"] += 1
                    facts["waited_lane_copies"] += groups[lane].popleft()
    facts["terminal_uncommitted_lane_copies"] = sum(opened)
    facts["terminal_unwaited_lane_groups"] = sum(map(len, groups))
    facts["terminal_unwaited_lane_copies"] = sum(sum(g) for g in groups)
    return facts


def _opcodes(trace: Path, root: Any, control: Path) -> dict[tuple[int, int], str]:
    functions = json.loads(control.read_text())["functions"]
    names = {}
    for device in root.gpu_device.values():
        for stream in device.streams.values():
            for kernel in stream.kernels:
                name = re.sub(r"___\d+$", "", kernel.name)
                key = int(kernel.function_unique_id)
                if key in names and names[key] != name:
                    raise ValueError(f"function id {key} has conflicting names")
                names[key] = name
    # Call targets may have static metadata without a launch of their own.
    path = trace / "extra_info/enhanced_execution_info.json"
    static = json.loads(path.read_text()).get("kernels", []) if path.is_file() else []
    for kernel in static:
        key = int(kernel["unique_function_id"])
        name = re.sub(r"___\d+$", "", kernel["kernel_name"])
        if key in names and names[key] != name:
            raise ValueError(f"function id {key} disagrees between launch and static metadata")
        names[key] = name
    joined = {}
    for key, name in names.items():
        if name not in functions:
            raise ValueError(f"control sidecar has no function {name!r}")
        for row in functions[name]:
            pc, opcode = row["pc"], row["opcode_text"]
            if type(pc) is not int or not isinstance(opcode, str) or not opcode.strip():
                raise ValueError(f"invalid SASS row in {name!r}")
            if (key, pc) in joined:
                raise ValueError(f"duplicate SASS PC in {name!r}: {pc}")
            joined[key, pc] = opcode
    for kernel in static:
        key = int(kernel["unique_function_id"])
        for row in kernel.get("instructions", []):
            opcode = joined.get((key, row["pc_num_dec"]))
            if opcode is None or opcode.split()[0] != row["op_code"]:
                raise ValueError(f"trace/control opcode mismatch: function {key}, PC {row['pc_num_dec']}")
    return joined


def extract_tail_facts(trace: Path, control: Path) -> dict[str, Any]:
    root = trace_pb2.Trace.FromString((trace / "dynamic_trace.pb").read_bytes())
    opcodes = _opcodes(trace, root, control)
    launches = {}
    for device_id, device in root.gpu_device.items():
        for stream_id, stream in device.streams.items():
            for kernel in stream.kernels:
                identity = (device_id, stream_id, int(kernel.id))
                if identity in launches:
                    raise ValueError(f"duplicate launch {identity}")
                grid = (kernel.grid_dim.x, kernel.grid_dim.y, kernel.grid_dim.z)
                if min(grid) <= 0:
                    raise ValueError(f"nonpositive grid in {identity}")
                launches[identity] = {
                    "device_id": device_id, "stream_id": stream_id, "kernel_id": kernel.id,
                    "function_unique_id": kernel.function_unique_id, "kernel_name": kernel.name,
                    "grid": grid, "population_blocks": grid[0] * grid[1] * grid[2],
                    "blocks": set(), "warps": 0, "records": 0, "memory_records": 0,
                    "same_warp_program_order_edges": 0, "sm_ids": set(), "orders": set(),
                    "instrumentation_order_records": 0, "instrumentation_order_duplicate_records": 0,
                    "static_opcode_records": 0, "static_opcode_ambiguity_records": 0,
                }
    payloads = sorted((trace / "threadblocks").rglob("d_*.pb"))
    if not payloads:
        raise ValueError("trace has no d_*.pb payloads")
    counts = Counter(records=0, memory_records=0, stores=0, atomics=0, reductions=0,
                     async_copies=0, active_mask_missing_records=0, predicate_mask_missing_records=0,
                     address_operands_unknown=0, predicated_off_memory_records=0)
    protocols, heads, formats, async_protocol = Counter(), Counter(), Counter(), Counter()
    fences, async_ops, allocations = set(), set(), set()
    # Launch and allocation identity are part of every address key. Do not
    # invent sharing across sequential launches or recycled virtual addresses.
    line_sms = defaultdict(set)
    for payload in payloads:
        identity = _identity(payload, trace)
        if identity not in launches:
            raise ValueError(f"payload for unknown launch: {payload}")
        row = launches[identity]
        block = threadblock_pb2.threadblock.FromString(payload.read_bytes())
        block_id = (block.block_id.x, block.block_id.y, block.block_id.z)
        if any(v < 0 or v >= dim for v, dim in zip(block_id, row["grid"])):
            raise ValueError(f"block outside launch grid: {payload}")
        if block_id in row["blocks"]:
            raise ValueError(f"duplicate block payload: {payload}")
        row["blocks"].add(block_id)
        for warp_id in sorted(block.warps):
            warp = block.warps[warp_id]
            missing = [(i.function_unique_id, i.pc) for i in warp.instructions
                       if (int(i.function_unique_id), int(i.pc)) not in opcodes]
            if missing:
                raise ValueError(f"missing static function/PC {missing[0]} in {payload}")
            async_protocol.update(_async_protocol(warp.instructions, opcodes))
            row["warps"] += 1
            row["same_warp_program_order_edges"] += max(0, len(warp.instructions) - 1)
            for instruction in warp.instructions:
                counts["records"] += 1
                row["records"] += 1
                key = (int(instruction.function_unique_id), int(instruction.pc))
                opcode = opcodes.get(key)
                if opcode is None:
                    raise ValueError(f"missing static function/PC {key} in {payload}")
                row["static_opcode_records"] += 1
                head = opcode.split()[0].split(".")[0]
                heads[head] += 1
                present = [_present(instruction, f) for f in ("active_mask", "predicate_mask")]
                counts["active_mask_missing_records"] += not present[0]
                counts["predicate_mask_missing_records"] += not present[1]
                mask = (instruction.active_mask & instruction.predicate_mask) if all(present) else None
                if mask:
                    if head in {"MEMBAR", "BAR", "ERRBAR", "DEPBAR", "LDGDEPBAR"}:
                        fences.add(opcode)
                        protocols["synchronization_records"] += 1
                    if head in {"LDGSTS", "LDGDEPBAR"} or WAIT.match(opcode):
                        async_ops.add(opcode)
                    for name, matches in (
                        ("async_issue_records", head == "LDGSTS"),
                        ("async_commit_records", head == "LDGDEPBAR"),
                        ("async_wait_records", bool(WAIT.match(opcode))),
                        ("membar_records", head == "MEMBAR"),
                        ("block_barrier_records", head == "BAR"),
                    ):
                        protocols[name] += matches
                if _present(instruction, "instrumentation_order"):
                    order = int(instruction.instrumentation_order)
                    if order <= 0:
                        raise ValueError(f"invalid instrumentation order in {payload}")
                    row["instrumentation_order_records"] += 1
                    row["instrumentation_order_duplicate_records"] += order in row["orders"]
                    row["orders"].add(order)
                sm = int(instruction.sm_id) if _present(instruction, "sm_id") else None
                if sm is not None:
                    row["sm_ids"].add(sm)
                if not _present(instruction, "memory_op_class"):
                    continue
                counts["memory_records"] += 1
                row["memory_records"] += 1
                if mask == 0:
                    counts["predicated_off_memory_records"] += 1
                if mask:
                    access = instruction.memory_op_access_kind
                    if head.startswith("RED"):
                        counts["reductions"] += 1
                    elif access == 2:
                        counts["stores"] += 1
                    elif access == 3:
                        counts["atomics"] += 1
                    elif access == 4:
                        counts["async_copies"] += 1
                for address in instruction.addresses:
                    formats[str(address.compression_format)] += 1
                    if mask is None:
                        counts["address_operands_unknown"] += 1
                        continue
                    lanes = address.lane_mask if _present(address, "lane_mask") else mask
                    if lanes & ~mask:
                        raise ValueError(f"operand widens instruction mask in {payload}")
                    vector, error = _reconstruct_lane_vector(address, lanes)
                    if error:
                        raise ValueError(f"{payload}: {error}")
                    space = address.address_space if _present(address, "address_space") else None
                    if space != 1:
                        if space is None:
                            counts["address_operands_unknown"] += 1
                        continue
                    alloc = address.allocation_id if _present(address, "allocation_id") else 0
                    if not alloc or sm is None:
                        counts["address_operands_unknown"] += 1
                        continue
                    width = address.data_width
                    if not 1 <= width <= 128:
                        raise ValueError(f"unsupported lane width {width} in {payload}")
                    allocations.add(int(alloc))
                    for lane, value in enumerate(vector):
                        if not (lanes >> lane) & 1:
                            continue
                        if value < 0 or value + width - 1 > (1 << 63) - 1:
                            raise ValueError(f"invalid lane address in {payload}")
                        for line in range(value // 128, (value + width - 1) // 128 + 1):
                            line_sms[identity, int(alloc), line].add(sm)
    rows = []
    for identity, row in sorted(launches.items()):
        row["blocks_observed"] = len(row.pop("blocks"))
        row["coverage_complete"] = row["blocks_observed"] == row["population_blocks"]
        row["instrumentation_order_unique"] = len(row.pop("orders"))
        row["instrumentation_order_strictly_unique"] = (
            row["instrumentation_order_duplicate_records"] == 0
            if row["instrumentation_order_records"] else None
        )
        row["sm_ids"] = sorted(row["sm_ids"])
        row["cross_sm_line_topology"] = sum(len(sms) > 1 for (launch, _, _), sms in line_sms.items() if launch == identity)
        rows.append(row)
    order_records = sum(row["instrumentation_order_records"] for row in rows)
    facts = dict(counts)
    facts.update(
        allocation_ids=sorted(allocations), cross_sm_line_topology=sum(row["cross_sm_line_topology"] for row in rows),
        fence_opcodes_observed=sorted(fences), async_protocol_opcodes_observed=sorted(async_ops),
        protocol_record_counts=dict(sorted(protocols.items())), opcode_heads_observed=dict(sorted(heads.items())),
        same_warp_program_order_edges=sum(row["same_warp_program_order_edges"] for row in rows),
        instrumentation_order_records=order_records,
        instrumentation_order_unique=sum(row["instrumentation_order_unique"] for row in rows),
        instrumentation_order_strictly_unique=all(row["instrumentation_order_duplicate_records"] == 0 for row in rows) if order_records else None,
        static_opcode_records=sum(row["static_opcode_records"] for row in rows), static_opcode_ambiguity_records=0,
        address_compression_formats=dict(sorted(formats.items())),
        async_protocol_lane_facts=dict(sorted(async_protocol.items())),
        line_generation_observed=False, owner_read_from_observed=False, fence_scope_semantics_closed=False,
        hardware_completion_token_observed=False, global_event_order_observed=False,
    )
    return {
        "schema": SCHEMA,
        "source": {"trace": str(trace), "control": str(control), "extraction": "raw_protobuf_launch_function_pc_join_and_decoded_lane_addresses"},
        "coverage": {"payload_files": len(payloads), "launches": len(rows),
                     "kind": "full_block_roster" if all(row["coverage_complete"] for row in rows) else "partial_block_roster"},
        "facts": facts, "launches": rows,
        "interpretation": "observed_trace_facts_only; topology is launch scoped; callback order is not hardware issue order; ownership/read_from/generation/hardware_completion remain unknown",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("control", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = extract_tail_facts(args.trace, args.control)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "records": report["facts"]["records"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
