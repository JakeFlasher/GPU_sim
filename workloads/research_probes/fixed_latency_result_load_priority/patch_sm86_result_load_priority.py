#!/usr/bin/env python3
"""Create matched sm_86 fixed/L1TEX result-priority cubins."""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path
from typing import Any, Sequence

from elftools.elf.elffile import ELFFile


FUNCTION = "_Z27result_load_priority_searchPyPjPKj"
LOAD_PC = 0xF0
LOAD_CONSUMER_PC = 0x140
FIXED_PC = 0x110
FIXED_CONSUMER_PC = 0x150
SCHEDULE_PCS = (0xF0, 0x100, 0x110, 0x120, 0x130)
POST_PCS = (0x140, 0x150)
EXPECTED_HEADS = {
    0xF0: "LDG.E.STRONG.SM",
    0x100: "IADD3",
    0x110: "HADD2",
    0x120: "ISETP.NE.AND",
    0x130: "NOP",
    0x140: "LOP3.LUT",
    0x150: "LOP3.LUT",
}
VARIANTS = {
    "same_noncollision": {"bank": "same", "producer_cycle": 22},
    "same_collision": {"bank": "same", "producer_cycle": 23},
    "opposite_noncollision": {"bank": "opposite", "producer_cycle": 22},
    "opposite_shifted": {"bank": "opposite", "producer_cycle": 23},
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(argv)


def canonical_control(high: int, stall: int) -> int:
    if stall < 1 or stall > 15:
        raise ValueError(f"invalid stall {stall}")
    high &= ~(0x1F << 41)
    high &= ~(0xF << 58)
    high |= stall << 41
    return high


def difference_offsets(left: bytes, right: bytes) -> set[int]:
    if len(left) != len(right):
        raise ValueError("binary sizes differ")
    return {
        index
        for index, (before, after) in enumerate(zip(left, right))
        if before != after
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        original = args.input.read_bytes()
        sidecar = json.loads(args.sidecar.read_text(encoding="utf-8"))
        if sidecar.get("arch") != "sm_86":
            raise ValueError("expected sm_86 sidecar")
        if sidecar.get("summary", {}).get("total_unknown_encoding") != 0:
            raise ValueError("sidecar contains unknown controls")
        rows = sidecar.get("functions", {}).get(FUNCTION)
        if not isinstance(rows, list) or len(rows) != 48:
            raise ValueError("expected one 48-instruction function")
        by_pc = {int(row["pc"]): row for row in rows}
        for pc, head in EXPECTED_HEADS.items():
            if not by_pc[pc]["opcode_text"].startswith(head + " ") and not (
                head == "NOP" and by_pc[pc]["opcode_text"] == "NOP"
            ):
                raise ValueError(
                    f"pc {pc:#x}: expected {head}, got {by_pc[pc]['opcode_text']}"
                )
        if by_pc[LOAD_PC]["opcode_text"] != "LDG.E.STRONG.SM R11, [R2.64]":
            raise ValueError("unexpected native timed load")
        if by_pc[FIXED_PC]["opcode_text"] != "HADD2 R9, R9, R6":
            raise ValueError("unexpected native fixed producer")
        if by_pc[LOAD_CONSUMER_PC]["opcode_text"] != (
            "LOP3.LUT R12, R12, R11, R7, 0x96, !PT"
        ):
            raise ValueError("unexpected native load consumer")
        if by_pc[FIXED_CONSUMER_PC]["opcode_text"] != (
            "LOP3.LUT R9, R9, R12, R8, 0x3c, !PT"
        ):
            raise ValueError("unexpected native fixed consumer")
        if any(re.search(r"\bR14\b", row["opcode_text"]) for row in rows):
            raise ValueError("R14 must be compiler-declared but unused")
        if not any(re.search(r"\bR15\b", row["opcode_text"]) for row in rows):
            raise ValueError("R15 use is required to prove R14 is allocated")

        with args.input.open("rb") as stream:
            elf = ELFFile(stream)
            flags = int(elf.header["e_flags"])
            abi = int(elf.header["e_ident"]["EI_ABIVERSION"])
            sm = (flags >> 8) & 0xFF if abi >= 8 else flags & 0xFF
            if sm != 86:
                raise ValueError(f"expected sm_86 cubin, got sm_{sm}")
            section = elf.get_section_by_name(f".text.{FUNCTION}")
            if section is None:
                raise ValueError("missing target function")
            section_offset = int(section.header["sh_offset"])
            section_size = int(section.header["sh_size"])

        load_low, load_high = struct.unpack_from(
            "<QQ", original, section_offset + LOAD_PC
        )
        consumer_low, consumer_high = struct.unpack_from(
            "<QQ", original, section_offset + LOAD_CONSUMER_PC
        )
        if (load_low >> 16) & 0xFF != 11:
            raise ValueError("encoded load destination is not R11")
        if (
            (consumer_low >> 16) & 0xFF,
            (consumer_low >> 24) & 0xFF,
            (consumer_low >> 32) & 0xFF,
            consumer_high & 0xFF,
        ) != (12, 12, 11, 7):
            raise ValueError("encoded load-consumer operands do not match SASS")

        allowed_control = {
            section_offset + pc + byte
            for pc in (*SCHEDULE_PCS, *POST_PCS)
            for byte in range(8, 16)
        }
        allowed_register = {
            section_offset + LOAD_PC + 2,
            section_offset + LOAD_CONSUMER_PC + 4,
        }
        payloads: dict[str, bytes] = {}
        reports: dict[str, Any] = {}
        for name, specification in VARIANTS.items():
            producer_cycle = int(specification["producer_cycle"])
            stalls = (
                15,
                producer_cycle - 15,
                1,
                1,
                29 - producer_cycle - 2,
            )
            output = bytearray(original)
            selected_controls = []
            for pc, stall in (*zip(SCHEDULE_PCS, stalls), (0x140, 15), (0x150, 2)):
                if pc < 0 or pc + 16 > section_size or pc % 16 != 0:
                    raise ValueError(f"invalid selected PC {pc:#x}")
                offset = section_offset + pc
                low, high = struct.unpack_from("<QQ", original, offset)
                patched_high = canonical_control(high, stall)
                struct.pack_into("<QQ", output, offset, low, patched_high)
                selected_controls.append(
                    {
                        "pc": pc,
                        "opcode_text": by_pc[pc]["opcode_text"],
                        "original_control": by_pc[pc]["control"],
                        "final_stall": stall,
                        "final_yield": 0,
                        "final_reuse": 0,
                    }
                )

            renamed = specification["bank"] == "opposite"
            if renamed:
                offset = section_offset + LOAD_PC
                low, high = struct.unpack_from("<QQ", output, offset)
                low = (low & ~(0xFF << 16)) | (14 << 16)
                struct.pack_into("<QQ", output, offset, low, high)
                offset = section_offset + LOAD_CONSUMER_PC
                low, high = struct.unpack_from("<QQ", output, offset)
                low = (low & ~(0xFF << 32)) | (14 << 32)
                struct.pack_into("<QQ", output, offset, low, high)

            payload = bytes(output)
            changed = difference_offsets(original, payload)
            allowed = allowed_control | (allowed_register if renamed else set())
            if not changed or not changed <= allowed:
                raise ValueError(f"{name}: changes escaped admitted fields")
            destination = args.prefix.with_name(f"{args.prefix.name}_{name}.cubin")
            destination.write_bytes(payload)
            payloads[name] = payload
            reports[name] = {
                "output": str(destination),
                "bank_relation": specification["bank"],
                "load_destination": 14 if renamed else 11,
                "load_destination_bank": 0 if renamed else 1,
                "fixed_destination": 9,
                "fixed_destination_bank": 1,
                "producer_cycle": producer_cycle,
                "load_consumer_cycle": 29,
                "static_stall_sum": sum(stalls) + 15 + 2,
                "complete_load_result_rename": renamed,
                "changed_bytes_from_native": len(changed),
                "selected_controls": selected_controls,
            }

        pair_diffs: dict[str, Any] = {}
        for left, right, expected in (
            ("same_noncollision", "same_collision", allowed_control),
            ("opposite_noncollision", "opposite_shifted", allowed_control),
            ("same_noncollision", "opposite_noncollision", allowed_register),
            ("same_collision", "opposite_shifted", allowed_register),
        ):
            changed = difference_offsets(payloads[left], payloads[right])
            if not changed or not changed <= expected:
                raise ValueError(f"{left}/{right}: pair diff escaped expected fields")
            pair_diffs[f"{left}_to_{right}"] = {
                "changed_bytes": len(changed),
                "only_control_fields": expected is allowed_control,
                "only_complete_load_register_rename": expected is allowed_register,
            }

        args.report.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "architecture": "sm_86",
                    "input": str(args.input),
                    "function": FUNCTION,
                    "variants": reports,
                    "pair_differences": pair_diffs,
                    "all_changes_inside_selected_control_or_register_fields": True,
                    "equal_instruction_work": True,
                    "equal_static_stall_sum": True,
                    "timed_load": {
                        "pc": LOAD_PC,
                        "write_barrier": 2,
                        "native_destination": 11,
                        "opposite_destination": 14,
                    },
                    "fixed_producer": {
                        "pc": FIXED_PC,
                        "opcode": "HADD2",
                        "destination": 9,
                        "destination_bank": 1,
                    },
                    "load_consumer": {
                        "pc": LOAD_CONSUMER_PC,
                        "wait_mask": 4,
                        "planned_cycle": 29,
                        "stall_to_fixed_consumer": 15,
                    },
                    "semantic_register_rename": {
                        "from": 11,
                        "to": 14,
                        "producer_pc": LOAD_PC,
                        "consumer_pc": LOAD_CONSUMER_PC,
                        "complete_live_range_coverage": True,
                        "replacement_absent_from_native_sass": True,
                        "higher_register_r15_compiler_allocated": True,
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"sm86 result/load priority patch: {error}", file=sys.stderr)
        return 1
    for name in VARIANTS:
        print(args.prefix.with_name(f"{args.prefix.name}_{name}.cubin"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
