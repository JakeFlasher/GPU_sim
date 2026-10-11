#!/usr/bin/env python3
"""Static SM120 issue/dependency/front-end feature extraction.

The tracer exposes SASS text, operand roles, control stalls and barrier masks.
This tool counts conservative dependency candidates and front-end footprint
without pretending that static register syntax reconstructs the hidden SM120
scoreboard or issue-port topology.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any


REG_RE = re.compile(r"\b([RU])R?(\d+)\b")


def register_sets(inst: dict[str, Any]) -> tuple[set[str], set[str], bool]:
    operands = inst.get("operands") or []
    refs: list[tuple[str, str]] = []
    for operand in operands:
        for ref in operand.get("register_refs", []) if isinstance(operand, dict) else []:
            role = str(ref.get("role", "")).upper()
            number = ref.get("number", ref.get("reg_num"))
            if number is not None:
                refs.append((role, f"{role}:{number}"))
    if not refs:
        text = str(inst.get("sass", ""))
        refs = [("TEXT", f"TEXT:{m.group(1)}:{m.group(2)}") for m in REG_RE.finditer(text)]
        return set(), {value for _, value in refs}, False
    # NVBit operand role metadata is not always available on Blackwell.  Treat
    # the first destination-looking operand as a def and retain ambiguity.
    defs = {value for role, value in refs if role in {"REG_DEF", "UREG_DEF", "DEST", "OUTPUT"}}
    uses = {value for role, value in refs if value not in defs and role not in {"REG_DEF", "UREG_DEF", "DEST", "OUTPUT"}}
    if not defs and refs:
        first = refs[0][1]
        defs = {first}
        uses.discard(first)
    return defs, uses, True


def analyze(metadata: dict[str, Any]) -> dict[str, Any]:
    functions = []
    for kernel in metadata.get("kernels", []) if isinstance(metadata, dict) else []:
        instructions = sorted(kernel.get("instructions", []), key=lambda item: int(item.get("pc_num_dec", 0)))
        last_def: dict[str, int] = {}
        raw = waw = war = ambiguous = 0
        control_stall = 0
        wait_masks = 0
        front_end_lines: set[int] = set()
        classes: collections.Counter[str] = collections.Counter()
        branch_count = 0
        for index, inst in enumerate(instructions):
            pc = int(inst.get("pc_num_dec", 0))
            front_end_lines.add(pc // 32)
            opcode = str(inst.get("opcode") or inst.get("op_code") or "UNKNOWN").upper()
            if any(token in opcode for token in ("BRA", "BRX", "JMP", "EXIT")):
                classes["branch"] += 1
                branch_count += 1
            elif any(token in opcode for token in ("MMA", "HMMA", "IMMA", "BMMA", "QMM", "OMMA")):
                classes["tensor"] += 1
            elif any(token in opcode for token in ("LD", "ST", "ATOM", "RED", "TMA", "UTMA", "LDS", "STS")):
                classes["memory"] += 1
            elif opcode.startswith(("BAR", "DEPBAR", "SYNCS", "MBAR")):
                classes["synchronization"] += 1
            else:
                classes["scalar_or_special"] += 1
            control = inst.get("control_fields") or {}
            control_stall += int(inst.get("stall_count", control.get("stall_count", 0)) or 0)
            wait_masks += int(control.get("wait_barrier_mask", 0) or 0).bit_count()
            defs, uses, metadata_available = register_sets(inst)
            if not metadata_available:
                ambiguous += 1
            for reg in uses:
                if reg in last_def:
                    raw += 1
            for reg in defs:
                if reg in last_def:
                    waw += 1
                last_def[reg] = index
            for reg in defs:
                if reg in uses:
                    war += 1
        functions.append(
            {
                "kernel_name": kernel.get("kernel_name"),
                "function_unique_id": kernel.get("unique_function_id"),
                "instruction_count": len(instructions),
                "dependency_candidates": {"raw": raw, "waw": waw, "war": war, "ambiguous_register_records": ambiguous},
                "control_stall_sum": control_stall,
                "wait_barrier_operand_count": wait_masks,
                "front_end": {"unique_32b_static_lines": len(front_end_lines), "branch_count": branch_count, "branch_density": branch_count / len(instructions) if instructions else None},
                "resource_classes": dict(classes),
                "register_metadata_boundary": "conservative_static_operand_projection; NVBit RFU provenance may be unavailable",
            }
        )
    return {"schema_version": 1, "model": "sm120_dependency_frontend_features_v1", "functions": functions, "physical_non_claim": ["scoreboard_depth", "scheduler_ports", "warp_selection_order"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    path = args.capture / "extra_info/native_static_metadata.json"
    result = analyze(json.loads(path.read_text(encoding="utf-8")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"functions": len(result["functions"]), "model": result["model"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
