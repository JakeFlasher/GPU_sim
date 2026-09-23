#!/usr/bin/env python3
"""Independent differential oracle for the bounded custom SIMT slice.

The Python interpreter intentionally duplicates the tiny program semantics
instead of importing Rust implementation code. It is a functional/timing
oracle for the feasibility slice only, not a GPU hardware model.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
from typing import Any

MASK32 = (1 << 32) - 1
REGISTER_COUNT = 32
WORD_BYTES = 4


def lane_address(base: int, stride: int, lane: int) -> int:
    address = base + stride * lane
    if address > MASK32 or address + WORD_BYTES - 1 > MASK32:
        raise RuntimeError("SIMT word address arithmetic overflow")
    if address % WORD_BYTES:
        raise RuntimeError(f"unaligned SIMT word address: {address:#x}")
    return address


def run(binary: pathlib.Path, spec: pathlib.Path, program: pathlib.Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(binary), str(spec), str(program)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip())
    return json.loads(completed.stdout)


def functional(spec: dict[str, Any], program: dict[str, Any]) -> dict[str, Any]:
    width = spec["execution"]["simt_width"]
    registers = [[0] * REGISTER_COUNT for _ in range(width)]
    memory: dict[int, int] = {
        int(key): value for key, value in program.get("initial_memory", {}).items()
    }
    pending: dict[int, list[tuple[int, int, int]]] = {}
    executed = 0
    for instruction in program["instructions"]:
        executed += 1
        op = instruction["op"]
        if op in {"load", "store", "async_store"}:
            ranges: list[tuple[int, int]] = []
            for lane in range(width):
                start = lane_address(instruction["base"], instruction["stride"], lane)
                end = start + WORD_BYTES - 1
                if op in {"store", "async_store"} and any(
                    start <= prior_end and prior_start <= end
                    for prior_start, prior_end in ranges
                ):
                    raise RuntimeError(f"overlapping lane store at {start:#x}")
                if any(
                    start <= pending_end and pending_start <= end
                    for group in pending.values()
                    for pending_start, pending_end, _ in group
                ):
                    raise RuntimeError(f"pending async overlap at {start:#x}")
                ranges.append((start, end))
        if op == "set_lane_id":
            for lane, state in enumerate(registers):
                state[instruction["dst"]] = lane
        elif op == "load":
            values = [
                memory[lane_address(instruction["base"], instruction["stride"], lane)]
                for lane in range(width)
            ]
            for state, value in zip(registers, values):
                state[instruction["dst"]] = value
        elif op == "add_imm":
            for state in registers:
                state[instruction["dst"]] = (
                    state[instruction["src"]] + instruction["imm"]
                ) & MASK32
        elif op == "add":
            for state in registers:
                state[instruction["dst"]] = (
                    state[instruction["lhs"]] + state[instruction["rhs"]]
                ) & MASK32
        elif op in {"store", "async_store"}:
            writes = [
                (
                    lane_address(instruction["base"], instruction["stride"], lane),
                    lane_address(instruction["base"], instruction["stride"], lane)
                    + WORD_BYTES
                    - 1,
                    state[instruction["src"]],
                )
                for lane, state in enumerate(registers)
            ]
            if op == "store":
                memory.update((start, value) for start, _end, value in writes)
            else:
                group = instruction["group"]
                if group in pending:
                    raise RuntimeError(f"async group reuse: {group}")
                pending[group] = writes
        elif op == "wait_group":
            group = instruction["group"]
            if group not in pending:
                raise RuntimeError(f"missing async group: {group}")
            memory.update(
                (start, value) for start, _end, value in pending.pop(group)
            )
        elif op == "halt":
            break
        else:
            raise RuntimeError(f"unsupported op: {op}")
    if pending:
        raise RuntimeError(f"unfinished async groups: {sorted(pending)}")
    return {
        "architecture": spec["architecture"],
        "isa": spec["isa"],
        "simt_width": width,
        "instructions_executed": executed,
        "registers": registers,
        "memory": {str(key): value for key, value in sorted(memory.items())},
    }


def timing(spec: dict[str, Any], program: dict[str, Any]) -> dict[str, Any]:
    width = spec["execution"]["simt_width"]
    resources = {resource["name"]: resource for resource in spec["resources"]}
    now = 0
    resource_next: dict[str, int] = {}
    pending: dict[int, int] = {}
    events = []

    def reserve(resource_name: str, operations: int, current: int) -> tuple[dict[str, Any], int, int]:
        resource = resources[resource_name]
        span = (operations + resource["issue_width"] - 1) // resource["issue_width"]
        start = max(current, resource_next.get(resource_name, 0))
        resource_next[resource_name] = start + span
        return resource, start, span

    for index, instruction in enumerate(program["instructions"]):
        op = instruction["op"]
        if op in {"set_lane_id", "add_imm", "add"}:
            resource, start, span = reserve("alu", 1, now)
            complete = start + span + resource["latency_cycles"]
            now = complete
            resource_name = "alu"
        elif op in {"load", "store"}:
            resource, start, span = reserve("memory", width, now)
            complete = start + span + resource["latency_cycles"]
            now = complete
            resource_name = "memory"
        elif op == "async_store":
            group = instruction["group"]
            if group in pending:
                raise RuntimeError(f"async group reuse: {group}")
            resource, start, span = reserve("memory", width, now)
            complete = start + span + resource["latency_cycles"]
            pending[group] = complete
            now = start + span
            resource_name = "memory"
        elif op == "wait_group":
            group = instruction["group"]
            if group not in pending:
                raise RuntimeError(f"missing async group: {group}")
            start = now
            complete = max(now, pending.pop(group))
            now = complete
            resource_name = None
        elif op == "halt":
            if pending:
                raise RuntimeError(f"unfinished async groups: {sorted(pending)}")
            start = now
            complete = now
            resource_name = None
        else:
            raise RuntimeError(f"unsupported op: {op}")
        events.append(
            {
                "instruction_index": index,
                "operation": op,
                "resource": resource_name,
                "start_cycle": start,
                "complete_cycle": complete,
            }
        )
    return {
        "architecture": spec["architecture"],
        "isa": spec["isa"],
        "simt_width": width,
        "final_cycle": now,
        "events": events,
        "event_count": len(events),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--timing-binary", type=pathlib.Path)
    parser.add_argument("--spec", type=pathlib.Path, required=True)
    parser.add_argument("--program", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    program = json.loads(args.program.read_text(encoding="utf-8"))

    rust_functional = run(args.binary, args.spec, args.program)
    reference_functional = functional(spec, program)
    functional_match = all(
        rust_functional.get(field) == reference_functional[field]
        for field in ("architecture", "isa", "simt_width", "instructions_executed", "registers", "memory")
    )
    document: dict[str, Any] = {
        "kind": "custom_simt_independent_differential",
        "functional_match": functional_match,
        "reference_functional": reference_functional,
        "rust_functional": rust_functional,
    }
    if not functional_match:
        raise RuntimeError(json.dumps(document, indent=2))

    if args.timing_binary:
        rust_timing = run(args.timing_binary, args.spec, args.program)
        reference_timing = timing(spec, program)
        timing_match = (
            rust_timing.get("architecture") == reference_timing["architecture"]
            and rust_timing.get("isa") == reference_timing["isa"]
            and rust_timing.get("simt_width") == reference_timing["simt_width"]
            and rust_timing.get("final_cycle") == reference_timing["final_cycle"]
            and rust_timing.get("events") == reference_timing["events"]
        )
        document.update(
            {
                "timing_match": timing_match,
                "reference_timing": reference_timing,
                "rust_timing": rust_timing,
            }
        )
        if not timing_match:
            raise RuntimeError(json.dumps(document, indent=2))

    encoded = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
