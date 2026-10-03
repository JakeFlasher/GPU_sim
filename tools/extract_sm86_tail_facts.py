#!/usr/bin/env python3
"""Extract observable SM86 tail facts from raw trace payloads and SASS control.

This is a plain analysis pass, not a witness authority and not an execution
gate. It deliberately reports unknown ownership/read-from/generation and
hardware completion instead of deriving them from counts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.nvbit_tracer.proto_binding_loader import load_dynamic_trace_bindings


SCHEMA = "sm86_tail_facts_v1"
DEVICE_RE = re.compile(r"device_(-?\d+)$")
STREAM_RE = re.compile(r"stream_(-?\d+)$")
KERNEL_RE = re.compile(r"kernel_(-?\d+)$")


def _identity(path: Path, root: Path) -> tuple[int, int, int]:
    parts = path.relative_to(root / "threadblocks").parts
    if len(parts) < 4:
        raise ValueError(f"payload path is too shallow: {path}")
    matches = (DEVICE_RE.fullmatch(parts[0]), STREAM_RE.fullmatch(parts[1]), KERNEL_RE.fullmatch(parts[2]))
    if any(match is None for match in matches):
        raise ValueError(f"payload path has malformed launch identity: {path}")
    return tuple(int(match.group(1)) for match in matches if match is not None)  # type: ignore[return-value]


def _has_scalar(message: Any, field: str) -> bool:
    try:
        return bool(message.HasField(field))
    except (ValueError, TypeError):
        return True


def extract_tail_facts(trace: Path, control: Path) -> dict[str, Any]:
    bindings = load_dynamic_trace_bindings()
    control_doc = json.loads(control.read_text())
    functions = control_doc.get("functions")
    if not isinstance(functions, dict):
        raise ValueError("control sidecar functions must be an object")
    for function, rows in functions.items():
        if not isinstance(rows, list):
            raise ValueError(f"control function {function} is not an array")

    # Prefer the tracer's static execution-info join when it is available.
    # The control sidecar remains a separate input; this optional map only
    # binds its decoded SASS to the raw record's function id and PC.  Never use
    # Python's process-randomized hash as an identity fallback.
    static_by_function_pc: dict[tuple[int, int], str] = {}
    execution_info = trace / "extra_info" / "enhanced_execution_info.json"
    if execution_info.is_file():
        info = json.loads(execution_info.read_text())
        for kernel in info.get("kernels", []):
            if not isinstance(kernel, dict):
                continue
            function_id = kernel.get("unique_function_id")
            if not isinstance(function_id, int):
                continue
            for row in kernel.get("instructions", []):
                if not isinstance(row, dict):
                    continue
                pc = row.get("pc_num_dec")
                opcode = row.get("op_code")
                if isinstance(pc, int) and isinstance(opcode, str):
                    static_by_function_pc[(function_id, pc)] = opcode
    opcode_by_pc: dict[int, set[str]] = defaultdict(set)
    for rows in functions.values():
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and isinstance(row.get("pc"), int) and isinstance(row.get("opcode_text"), str):
                    opcode_by_pc[int(row["pc"])].add(row["opcode_text"])

    payloads = sorted(trace.joinpath("threadblocks").rglob("d_*.pb"))
    if not payloads:
        raise ValueError("trace has no d_*.pb payloads")
    launch_rows: dict[tuple[int, int, int], dict[str, Any]] = {}
    total_records = 0
    memory_records = 0
    stores = atomics = reductions = async_copies = 0
    active_missing = predicate_missing = 0
    line_sms: dict[tuple[int, int], set[int]] = defaultdict(set)
    allocation_ids: set[int] = set()
    fence_opcodes: set[str] = set()
    async_protocol_opcodes: set[str] = set()
    protocol_counts = Counter()
    opcode_heads = Counter()
    same_warp_program_order_edges = 0
    instrumentation_orders: list[int] = []
    static_opcode_records = 0
    static_opcode_ambiguity_records = 0
    for payload in payloads:
        identity = _identity(payload, trace)
        row = launch_rows.setdefault(
            identity,
            {
                "device_id": identity[0],
                "stream_id": identity[1],
                "kernel_id": identity[2],
                "records": 0,
                "sm_ids": set(),
                "memory_records": 0,
                "warps": 0,
                "same_warp_program_order_edges": 0,
                "instrumentation_order_records": 0,
                "static_opcode_records": 0,
                "static_opcode_ambiguity_records": 0,
            },
        )
        block = bindings.threadblock_pb2.threadblock()
        block.ParseFromString(payload.read_bytes())
        for warp in block.warps.values():
            row["warps"] += 1
            order_edges = max(0, len(warp.instructions) - 1)
            row["same_warp_program_order_edges"] += order_edges
            same_warp_program_order_edges += order_edges
            for instruction in warp.instructions:
                total_records += 1
                row["records"] += 1
                sm_id = int(instruction.sm_id) if _has_scalar(instruction, "sm_id") else None
                if sm_id is not None:
                    row["sm_ids"].add(sm_id)
                if not _has_scalar(instruction, "active_mask"):
                    active_missing += 1
                if not _has_scalar(instruction, "predicate_mask"):
                    predicate_missing += 1
                function_opcode = static_by_function_pc.get(
                    (int(instruction.function_unique_id), int(instruction.pc))
                )
                static_opcodes = (
                    {function_opcode}
                    if function_opcode is not None
                    else opcode_by_pc.get(int(instruction.pc), set())
                )
                if function_opcode is not None:
                    static_opcode_records += 1
                    row["static_opcode_records"] += 1
                elif len(static_opcodes) > 1:
                    static_opcode_ambiguity_records += 1
                    row["static_opcode_ambiguity_records"] += 1
                # Zero is the protobuf default, so it is the absence marker
                # for the optional tracer order field in this payload format.
                if int(instruction.instrumentation_order) > 0:
                    instrumentation_orders.append(int(instruction.instrumentation_order))
                    row["instrumentation_order_records"] += 1
                for opcode in static_opcodes:
                    head = opcode.split()[0].split(".")[0].upper()
                    opcode_heads[head] += 1
                    if head in {"MEMBAR", "BAR", "ERRBAR", "DEPBAR", "LDGDEPBAR"}:
                        fence_opcodes.add(opcode)
                        protocol_counts["fence_records"] += 1
                    if head in {"CP", "LDGSTS", "LDGDEPBAR"} or "CP.ASYNC" in opcode.upper():
                        async_protocol_opcodes.add(opcode)
                    if head == "LDGSTS":
                        protocol_counts["async_issue_records"] += 1
                    elif head == "LDGDEPBAR":
                        protocol_counts["async_commit_records"] += 1
                    elif head == "DEPBAR":
                        protocol_counts["async_wait_records"] += 1
                    elif head == "MEMBAR":
                        protocol_counts["membar_records"] += 1
                    elif head == "BAR":
                        protocol_counts["block_barrier_records"] += 1
                if not _has_scalar(instruction, "memory_op_class"):
                    continue
                memory_records += 1
                row["memory_records"] += 1
                access = int(instruction.memory_op_access_kind) if _has_scalar(instruction, "memory_op_access_kind") else 0
                op_class = int(instruction.memory_op_class)
                if access == 2: stores += 1
                elif access == 3: atomics += 1
                elif access == 5: reductions += 1
                elif access == 4 or op_class == 8: async_copies += 1
                for address in instruction.addresses:
                    if _has_scalar(address, "allocation_id"):
                        allocation_ids.add(int(address.allocation_id))
                    for value in address.addrs:
                        if value < 0 or op_class != 1 or sm_id is None:
                            continue
                        line_sms[(int(address.allocation_id) if _has_scalar(address, "allocation_id") else -1, int(value) // 128)].add(sm_id)
    for row in launch_rows.values():
        row["sm_ids"] = sorted(row["sm_ids"])
    cross_sm_lines = sum(len(sms) > 1 for sms in line_sms.values())
    return {
        "schema": SCHEMA,
        "source": {"trace": str(trace), "control": str(control), "extraction": "raw_protobuf_plus_static_control_separate_pass"},
        "coverage": {"payload_files": len(payloads), "launches": len(launch_rows), "kind": "full_unsampled_input_assumed_from_payload_roster"},
        "facts": {
            "records": total_records, "memory_records": memory_records, "stores": stores,
            "atomics": atomics, "reductions": reductions, "async_copies": async_copies,
            "allocation_ids": sorted(allocation_ids), "cross_sm_line_topology": cross_sm_lines,
            "active_mask_missing_records": active_missing, "predicate_mask_missing_records": predicate_missing,
            "fence_opcodes_observed": sorted(fence_opcodes), "async_protocol_opcodes_observed": sorted(async_protocol_opcodes),
            "protocol_record_counts": dict(sorted(protocol_counts.items())),
            "opcode_heads_observed": dict(sorted(opcode_heads.items())),
            "same_warp_program_order_edges": same_warp_program_order_edges,
            "instrumentation_order_records": len(instrumentation_orders),
            "instrumentation_order_unique": len(set(instrumentation_orders)),
            "instrumentation_order_strictly_unique": len(instrumentation_orders) == len(set(instrumentation_orders)),
            "static_opcode_records": static_opcode_records,
            "static_opcode_ambiguity_records": static_opcode_ambiguity_records,
            "line_generation_observed": False, "owner_read_from_observed": False,
            "fence_scope_semantics_closed": False, "hardware_completion_token_observed": False,
            "global_event_order_observed": False,
        },
        "launches": sorted(launch_rows.values(), key=lambda row: (row["device_id"], row["stream_id"], row["kernel_id"])),
        "interpretation": "observed_trace_facts_only_unknown_owner_read_from_generation_scope_and_hardware_completion_remain_open"
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("control", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = extract_tail_facts(args.trace, args.control)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "records": report["facts"]["records"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
