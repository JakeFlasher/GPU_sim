#!/usr/bin/env python3
"""Adapt one recognized Vortex vecadd ELF idiom into the custom SIMT program.

This deliberately narrow adapter is a real-binary probe, not a general RISC-V
decoder. It requires the kernel symbol and a recognizable load/add/store shape,
then preserves all unsupported/custom encodings as a typed rejection rather
than guessing their semantics.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess


INSTRUCTION_WITH_OPERANDS = re.compile(
    r"^\s*[0-9a-f]+:\s+[0-9a-f]+\s+([^\s]+)(?:\s+(.*))?$"
)
WORD_MEMORY = re.compile(r"^\s*([a-z][a-z0-9]*),\s*(-?\d+)\(([a-z][a-z0-9]*)\)\s*$")


def first_register(operands: str) -> str | None:
    fields = [field.strip() for field in operands.split(",")]
    return fields[0] if fields else None


def zero_offset_memory(operands: str) -> tuple[str, str] | None:
    match = WORD_MEMORY.match(operands)
    if match is None or int(match.group(2)) != 0:
        return None
    return match.group(1), match.group(3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objdump", type=pathlib.Path, required=True)
    parser.add_argument("--elf", type=pathlib.Path, required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--isa", default="simt_u32_v1")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--src0", type=int, default=400)
    parser.add_argument("--src1", type=int, default=800)
    parser.add_argument("--dst", type=int, default=1200)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.width <= 64:
        parser.error("--width must be in 1..=64")
    for name, value in (("--src0", args.src0), ("--src1", args.src1), ("--dst", args.dst)):
        if not 0 <= value <= 0xFFFF_FFFF:
            parser.error(f"{name} must be in the u32 byte-address domain")
        if value + 4 * (args.width - 1) + 3 > 0xFFFF_FFFF:
            parser.error(f"{name} plus the lane word span overflows the u32 byte-address domain")
    completed = subprocess.run(
        [str(args.objdump), "-d", str(args.elf)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.stderr.strip())
    text = completed.stdout
    symbol = next(
        (
            candidate
            for candidate in ("<_Z13vecadd_kernel", "<__vx_kentry_kernel_main>")
            if text.find(candidate) >= 0
        ),
        None,
    )
    if symbol is None:
        raise SystemExit("vecadd kernel symbol was not found")
    start = text.find(symbol)
    end = text.find("\n\n", start)
    body = text[start:] if end < 0 else text[start:end]
    records = []
    for line in body.splitlines():
        match = INSTRUCTION_WITH_OPERANDS.match(line)
        if match:
            records.append((match.group(1).lower().split(".", 1)[0], match.group(2) or ""))
    opcodes = [opcode for opcode, _ in records]
    if any(op in {"flw", "fadd", "fsw"} for op in opcodes):
        raise SystemExit("floating-point vecadd is outside the integer adapter")
    idiom_match = False
    for first_load in range(len(records)):
        if records[first_load][0] != "lw":
            continue
        first_memory = zero_offset_memory(records[first_load][1])
        if first_memory is None:
            continue
        first_dst, _first_base = first_memory
        if first_dst is None:
            continue
        for second_load in range(first_load + 1, min(len(records), first_load + 9)):
            if records[second_load][0] != "lw":
                continue
            second_memory = zero_offset_memory(records[second_load][1])
            if second_memory is None:
                continue
            second_dst, _second_base = second_memory
            if second_dst is None or second_dst == first_dst:
                continue
            for add in range(second_load + 1, min(len(records), second_load + 9)):
                if records[add][0] != "add":
                    continue
                add_fields = [field.strip() for field in records[add][1].split(",")]
                if len(add_fields) != 3 or set(add_fields[1:]) != {first_dst, second_dst}:
                    continue
                result_dst = add_fields[0]
                for store in range(add + 1, min(len(records), add + 9)):
                    store_memory = zero_offset_memory(records[store][1]) if records[store][0] == "sw" else None
                    if store_memory is not None and store_memory[0] == result_dst:
                        idiom_match = True
                        break
                if idiom_match:
                    break
            if idiom_match:
                break
        if idiom_match:
            break
    if not idiom_match:
        raise SystemExit("kernel does not match the ordered lw/lw/add/sw idiom")
    memory = {
        **{str(args.src0 + 4 * lane): 2 * lane for lane in range(args.width)},
        **{str(args.src1 + 4 * lane): 3 * lane for lane in range(args.width)},
    }
    program = {
        "schema_version": 1,
        "architecture": args.architecture,
        "isa": args.isa,
        "initial_memory": memory,
        "instructions": [
            {"op": "set_lane_id", "dst": 0},
            {"op": "load", "dst": 1, "base": args.src0, "stride": 4},
            {"op": "load", "dst": 2, "base": args.src1, "stride": 4},
            {"op": "add", "dst": 3, "lhs": 1, "rhs": 2},
            {"op": "store", "base": args.dst, "stride": 4, "src": 3},
            {"op": "halt"},
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(program, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "matched_opcodes": opcodes,
                "output": str(args.output),
                "binding_mode": "caller_supplied_synthetic_byte_addresses",
                "interpretation": "pattern recognition only; no general binary or pointer-semantics claim",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
