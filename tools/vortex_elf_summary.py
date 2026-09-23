#!/usr/bin/env python3
"""Extract a bounded static operation summary from a Vortex ELF.

This is an open-ISA ingestion probe, not a functional or timing simulator.
It deliberately preserves unresolved/custom encodings instead of guessing
their semantics.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import subprocess
from typing import Any

LINE = re.compile(r"^\s*([0-9a-f]+):\s+[0-9a-f]+\s+([^\s]+)(?:\s+(.*))?$")


def operation(mnemonic: str) -> str:
    mnemonic = mnemonic.lower()
    if mnemonic.startswith("."):
        return "custom_or_unknown"
    base = mnemonic.split(".", 1)[0]
    if base in {"lw", "lh", "lb", "lwu", "lhu", "lbu", "ld"}:
        return "memory_load"
    if base in {"sw", "sh", "sb", "sd"}:
        return "memory_store"
    if base in {"add", "addi", "sub", "mul", "div", "rem", "and", "andi", "or", "ori", "xor", "xori", "sll", "slli", "srl", "srli", "sra", "srai"}:
        return "arithmetic"
    if base in {"beq", "bne", "blt", "bge", "bltu", "bgeu", "j", "jal", "jalr", "ret", "call"}:
        return "control"
    if base in {"csrr", "csrw", "csrrw", "csrrs", "csrrc"}:
        return "special_register"
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objdump", type=pathlib.Path, required=True)
    parser.add_argument("--elf", type=pathlib.Path, required=True)
    parser.add_argument("--simt-width", type=int, default=4)
    parser.add_argument("--source-revision", default="unknown")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    completed = subprocess.run(
        [str(args.objdump), "-d", str(args.elf)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.stderr.strip())
    opcode_counts: collections.Counter[str] = collections.Counter()
    operation_counts: collections.Counter[str] = collections.Counter()
    unresolved: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        match = LINE.match(line)
        if not match:
            continue
        address, mnemonic, operands = match.groups()
        opcode_counts[mnemonic] += 1
        kind = operation(mnemonic)
        operation_counts[kind] += 1
        if kind == "custom_or_unknown":
            unresolved.append({"address": address, "mnemonic": mnemonic, "operands": operands or ""})
    result = {
        "kind": "vortex_elf_static_summary",
        "architecture": "vortex_config",
        "isa": "riscv_vortex",
        "simt_width": args.simt_width,
        "source_revision": args.source_revision,
        "elf": str(args.elf),
        "instruction_count": sum(opcode_counts.values()),
        "opcode_counts": dict(sorted(opcode_counts.items())),
        "operation_counts": dict(sorted(operation_counts.items())),
        "unresolved_count": len(unresolved),
        "unresolved": unresolved,
        "interpretation": "static open-ISA evidence; no functional/timing claim",
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
