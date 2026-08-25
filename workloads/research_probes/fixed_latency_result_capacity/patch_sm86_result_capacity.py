#!/usr/bin/env python3
"""Create equal-work sm_86 b7/b8/b9 result-reservation burst cubins."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Sequence

from elftools.elf.elffile import ELFFile


FUNCTION = "_Z22result_capacity_searchPyPjPKj"
DRAIN_PC = 0x1A0
GATE_PC = 0x1E0
RUN = (
    (0x1B0, "HMMA.16816.F32"),
    (0x1C0, "LOP3.LUT"),
    (0x1D0, "HADD2"),
    (0x1F0, "LOP3.LUT"),
    (0x200, "FFMA"),
    (0x210, "LOP3.LUT"),
    (0x220, "HADD2"),
    (0x230, "LOP3.LUT"),
    (0x240, "FFMA"),
    (0x250, "LOP3.LUT"),
    (0x260, "HADD2"),
    (0x270, "LOP3.LUT"),
    (0x280, "FFMA"),
    (0x290, "IADD3"),
    (0x2A0, "LOP3.LUT"),
    (0x2B0, "LOP3.LUT"),
    (0x2C0, "LOP3.LUT"),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(argv)


def canonical_control(high: int, stall: int) -> int:
    high &= ~(0x1F << 41)
    high &= ~(0xF << 58)
    high |= stall << 41
    return high


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        original = args.input.read_bytes()
        sidecar = json.loads(args.sidecar.read_text(encoding="utf-8"))
        if sidecar.get("arch") != "sm_86" or sidecar["summary"]["total_unknown_encoding"] != 0:
            raise ValueError("expected a fully decoded sm_86 sidecar")
        rows = sidecar["functions"].get(FUNCTION)
        if not isinstance(rows, list) or len(rows) != 72:
            raise ValueError(f"expected one 72-instruction function, got {0 if not isinstance(rows, list) else len(rows)}")
        by_pc = {int(row["pc"]): row for row in rows}
        if not by_pc[DRAIN_PC]["opcode_text"].startswith("BAR.SYNC.DEFER_BLOCKING "):
            raise ValueError("missing pre-run BAR drain")
        if not by_pc[GATE_PC]["opcode_text"].startswith("ISETP.NE.AND "):
            raise ValueError("missing run gate predicate")
        for pc, head in RUN:
            if not by_pc[pc]["opcode_text"].startswith(head + " "):
                raise ValueError(f"run PC {pc:#x}: expected {head}, got {by_pc[pc]['opcode_text']}")

        with args.input.open("rb") as stream:
            elf = ELFFile(stream)
            flags = int(elf.header["e_flags"])
            abi = int(elf.header["e_ident"]["EI_ABIVERSION"])
            sm = (flags >> 8) & 0xFF if abi >= 8 else flags & 0xFF
            if sm != 86:
                raise ValueError(f"expected sm_86, got sm_{sm}")
            section = elf.get_section_by_name(f".text.{FUNCTION}")
            if section is None:
                raise ValueError(f"missing .text.{FUNCTION}")
            section_offset = int(section.header["sh_offset"])
            section_size = int(section.header["sh_size"])

        selected = (DRAIN_PC, GATE_PC, *(pc for pc, _ in RUN))
        allowed = {
            section_offset + pc + byte
            for pc in selected
            for byte in range(8, 16)
        }
        variants = {}
        payloads = {}
        for depth in (7, 8, 9):
            output = bytearray(original)
            rows_report = []
            gap_pc = RUN[depth - 1][0]
            for pc in selected:
                if pc % 16 != 0 or pc + 16 > section_size:
                    raise ValueError(f"invalid selected PC {pc:#x}")
                offset = section_offset + pc
                low, high = struct.unpack_from("<QQ", original, offset)
                stall = 15 if pc in {DRAIN_PC, gap_pc} else 1
                patched_high = canonical_control(high, stall)
                struct.pack_into("<QQ", output, offset, low, patched_high)
                rows_report.append(
                    {
                        "pc": pc,
                        "opcode_text": by_pc[pc]["opcode_text"],
                        "original_control": by_pc[pc]["control"],
                        "final_stall": stall,
                        "final_yield": 0,
                        "final_reuse": 0,
                        "role": (
                            "pre_run_drain"
                            if pc == DRAIN_PC
                            else ("non_reserving_gate" if pc == GATE_PC else ("movable_gap" if pc == gap_pc else "reservation_run"))
                        ),
                    }
                )
            payload = bytes(output)
            changed = {
                index
                for index, (before, after) in enumerate(zip(original, payload))
                if before != after
            }
            if not changed or not changed <= allowed:
                raise ValueError(f"b{depth}: changes escaped selected control words")
            destination = args.prefix.with_name(f"{args.prefix.name}_b{depth}.cubin")
            destination.write_bytes(payload)
            payloads[depth] = payload
            variants[f"b{depth}"] = {
                "output": str(destination),
                "changed_bytes": len(changed),
                "reservation_depth_before_gap": depth,
                "gap_pc": gap_pc,
                "selected_controls": rows_report,
            }

        pair_diffs = {}
        for left, right in ((7, 8), (8, 9), (7, 9)):
            changed = {
                index
                for index, pair in enumerate(zip(payloads[left], payloads[right]))
                if pair[0] != pair[1]
            }
            if not changed or not changed <= allowed:
                raise ValueError(f"b{left}/b{right}: pair diff escaped selected controls")
            pair_diffs[f"b{left}_to_b{right}_changed_bytes"] = len(changed)

        args.report.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "architecture": "sm_86",
                    "input": str(args.input),
                    "function": FUNCTION,
                    "pre_run_drain_pc": DRAIN_PC,
                    "non_reserving_gate_pc": GATE_PC,
                    "reservation_run": [
                        {"ordinal": ordinal, "pc": pc, "opcode": head}
                        for ordinal, (pc, head) in enumerate(RUN, 1)
                    ],
                    "variants": variants,
                    "pair_diffs": pair_diffs,
                    "all_changes_inside_selected_control_words": True,
                    "equal_instruction_work": True,
                    "equal_static_stall_sum": True,
                    "capacity_signature": "b9 slower than b8 while b8 equals b7",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"sm86 result-capacity patch: {error}", file=sys.stderr)
        return 1
    for depth in (7, 8, 9):
        print(variants[f"b{depth}"]["output"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
