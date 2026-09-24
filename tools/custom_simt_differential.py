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
    shared_memory: dict[int, int] = {
        int(key): value
        for key, value in program.get("initial_shared_memory", {}).items()
    }
    for address in (*memory.keys(), *shared_memory.keys()):
        lane_address(address, 0, 0)
    pending: dict[int, list[tuple[int, int, int]]] = {}
    executed = 0
    for instruction in program["instructions"]:
        executed += 1
        op = instruction["op"]
        if op in {
            "load", "load_shared", "store", "store_shared", "atomic_add_shared", "async_store"
        }:
            ranges: list[tuple[int, int]] = []
            for lane in range(width):
                start = lane_address(instruction["base"], instruction["stride"], lane)
                end = start + WORD_BYTES - 1
                if op in {"store", "async_store"} and any(
                    start <= prior_end and prior_start <= end
                    for prior_start, prior_end in ranges
                ):
                    raise RuntimeError(f"overlapping lane store at {start:#x}")
                if op not in {"load_shared", "store_shared", "atomic_add_shared"} and any(
                    start <= pending_end and pending_start <= end
                    for group in pending.values()
                    for pending_start, pending_end, _ in group
                ):
                    raise RuntimeError(f"pending async overlap at {start:#x}")
                ranges.append((start, end))
        if op == "set_lane_id":
            for lane, state in enumerate(registers):
                state[instruction["dst"]] = lane
        elif op in {"load", "load_shared"}:
            source = shared_memory if op == "load_shared" else memory
            values = [
                source[lane_address(instruction["base"], instruction["stride"], lane)]
                for lane in range(width)
            ]
            for state, value in zip(registers, values):
                state[instruction["dst"]] = value
        elif op == "add_imm":
            for state in registers:
                state[instruction["dst"]] = (
                    state[instruction["src"]] + instruction["imm"]
                ) & MASK32
        elif op == "broadcast_lane0":
            value = registers[0][instruction["src"]]
            for state in registers:
                state[instruction["dst"]] = value
        elif op == "reduce_add":
            value = sum(state[instruction["src"]] for state in registers) & MASK32
            for state in registers:
                state[instruction["dst"]] = value
        elif op == "add":
            for state in registers:
                state[instruction["dst"]] = (
                    state[instruction["lhs"]] + state[instruction["rhs"]]
                ) & MASK32
        elif op == "atomic_add_shared":
            for lane, state in enumerate(registers):
                address = lane_address(instruction["base"], instruction["stride"], lane)
                old = shared_memory[address]
                shared_memory[address] = (old + state[instruction["src"]]) & MASK32
                state[instruction["dst"]] = old
        elif op in {"store", "store_shared", "async_store"}:
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
            if op in {"store", "store_shared"}:
                destination = shared_memory if op == "store_shared" else memory
                destination.update((start, value) for start, _end, value in writes)
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
        elif op == "group_barrier":
            if pending:
                raise RuntimeError(f"unfinished async groups: {sorted(pending)}")
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
        "shared_memory": {
            str(key): value for key, value in sorted(shared_memory.items())
        },
    }


def _timing_tasks(spec: dict[str, Any], program: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the same bounded dependency DAG as ``simt_pipeline::tasks``.

    Keeping this in the independent oracle is intentional: it makes the
    differential check exercise explicit register, memory-domain, async, and
    barrier dependencies rather than merely replaying the Rust event list.
    """
    width = spec["execution"]["simt_width"]
    operations = {resource["name"]: resource for resource in spec["resources"]}
    bindings = spec.get("operation_resources", {})
    registers: list[int | None] = [None] * REGISTER_COUNT
    memory: dict[tuple[str, int], tuple[int | None, list[int]]] = {}
    async_groups: dict[int, int] = {}
    tasks: list[dict[str, Any]] = []
    last_barrier: int | None = None

    def memory_access(instruction: dict[str, Any]) -> tuple[str, int, int, bool] | None:
        op = instruction["op"]
        if op in {"load", "store", "async_store"}:
            return ("global", instruction["base"], instruction["stride"], op != "load")
        if op in {"load_shared", "store_shared", "atomic_add_shared"}:
            return ("shared", instruction["base"], instruction["stride"], op != "load_shared")
        return None

    for index, instruction in enumerate(program["instructions"]):
        op = instruction["op"]
        dependencies: set[int] = set()
        if last_barrier is not None:
            dependencies.add(last_barrier)

        if op in {"set_lane_id", "load", "load_shared"}:
            sources: list[int] = []
            destination: int | None = instruction["dst"]
        elif op in {"add_imm", "broadcast_lane0", "reduce_add"}:
            sources = [instruction["src"]]
            destination = instruction["dst"]
        elif op == "add":
            sources = [instruction["lhs"], instruction["rhs"]]
            destination = instruction["dst"]
        elif op in {"store", "store_shared", "async_store"}:
            sources = [instruction["src"]]
            destination = None
        elif op == "atomic_add_shared":
            sources = [instruction["src"]]
            destination = instruction["dst"]
        else:
            sources = []
            destination = None
        for register in [*sources, *([] if destination is None else [destination])]:
            if registers[register] is not None:
                dependencies.add(registers[register])
        if destination is not None:
            registers[destination] = index

        access = memory_access(instruction)
        kind = "compute"
        work = 1
        if access is not None:
            kind = "memory"
            work = width
            domain, base, stride, write = access
            addresses = {
                lane_address(base, stride, lane) for lane in range(width)
            }
            for address in addresses:
                last_write, reads = memory.setdefault((domain, address), (None, []))
                if last_write is not None:
                    dependencies.add(last_write)
                if write:
                    dependencies.update(reads)
                    memory[(domain, address)] = (index, [])
                else:
                    reads.append(index)

        resource_name: str | None
        if op == "wait_group":
            group = instruction["group"]
            if group not in async_groups:
                raise RuntimeError(f"missing async group: {group}")
            dependencies.add(async_groups.pop(group))
            resource_name = None
        elif op in {"group_barrier", "halt"}:
            if async_groups:
                raise RuntimeError(f"unfinished async groups: {sorted(async_groups)}")
            dependencies.update(range(index))
            resource_name = None
            if op == "group_barrier":
                last_barrier = index
        else:
            if op == "async_store":
                group = instruction["group"]
                if group in async_groups:
                    raise RuntimeError(f"async group reuse: {group}")
                async_groups[group] = index
            default = "memory" if kind == "memory" else "alu"
            resource_name = bindings.get(op, default)
            if resource_name not in operations:
                raise RuntimeError(f"missing resource: {resource_name}")

        tasks.append(
            {
                "resource": resource_name,
                "work": work,
                "dependencies": sorted(dependencies),
            }
        )
    if async_groups:
        raise RuntimeError(f"unfinished async groups: {sorted(async_groups)}")
    return tasks


def timing(spec: dict[str, Any], program: dict[str, Any]) -> dict[str, Any]:
    width = spec["execution"]["simt_width"]
    resources = {resource["name"]: resource for resource in spec["resources"]}
    tasks = _timing_tasks(spec, program)
    resource_next: dict[str, int] = {}
    pending: dict[str, list[int]] = {name: [] for name in resources}
    intervals: list[tuple[int, int]] = []
    events = []
    group_next = 0

    for index, task in enumerate(tasks):
        start = group_next
        for dependency in task["dependencies"]:
            start = max(start, intervals[dependency][1])
        resource_name = task["resource"]
        if resource_name is not None:
            resource = resources[resource_name]
            width_per_issue = resource["issue_width"]
            latency = resource.get("latency_cycles")
            if latency is None:
                raise RuntimeError(f"resource latency is unknown: {resource_name}")
            initiation_interval = resource.get("initiation_interval", 1)
            max_in_flight = resource.get("max_in_flight")
            if width_per_issue <= 0 or latency <= 0 or initiation_interval <= 0:
                raise RuntimeError(f"invalid resource timing: {resource_name}")
            start = max(start, resource_next.get(resource_name, 0))
            if max_in_flight is not None:
                if max_in_flight <= 0:
                    raise RuntimeError(f"invalid resource capacity: {resource_name}")
                while len(pending[resource_name]) >= max_in_flight:
                    earliest = min(pending[resource_name])
                    start = max(start, earliest)
                    pending[resource_name] = [
                        complete
                        for complete in pending[resource_name]
                        if complete > start
                    ]
            span = (task["work"] + width_per_issue - 1) // width_per_issue
            service = span * initiation_interval
            last_beat_end = (span - 1) * initiation_interval + 1
            complete = start + last_beat_end + latency
            resource_next[resource_name] = start + service
            pending[resource_name].append(complete)
        else:
            complete = start
        group_next = start + 1
        intervals.append((start, complete))
        events.append(
            {
                "instruction_index": index,
                "operation": program["instructions"][index]["op"],
                "resource": resource_name,
                "start_cycle": start,
                "complete_cycle": complete,
            }
        )
    completion_frontier = max([0, *(complete for _start, complete in intervals)])
    return {
        "architecture": spec["architecture"],
        "isa": spec["isa"],
        "simt_width": width,
        "final_cycle": completion_frontier,
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
        for field in (
            "architecture",
            "isa",
            "simt_width",
            "instructions_executed",
            "registers",
            "memory",
            "shared_memory",
        )
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
