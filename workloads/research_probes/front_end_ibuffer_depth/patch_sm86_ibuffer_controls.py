#!/usr/bin/env python3
"""Patch the branch-free sm_86 IBuffer-control variants."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Sequence

from elftools.elf.elffile import ELFFile


FUNCTION = "_Z22ibuffer_timeline_probePyPj"
BLOCKS = 1
INSTRUCTIONS_PER_BLOCK = 32
MEASURED_INSTRUCTIONS = BLOCKS * INSTRUCTIONS_PER_BLOCK
VARIANTS = {}
for base_stall in (0, 1):
    VARIANTS[f"baseline_s{base_stall}"] = (base_stall, ())
    for index in range(8):
        VARIANTS[f"stall4_s{base_stall}_i{index}"] = (
            base_stall,
            ((index, 4, False),),
        )
        VARIANTS[f"yield_s{base_stall}_i{index}"] = (
            base_stall,
            ((index, base_stall, True),),
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(argv)


def control(high: int, stall: int, yield_flag: bool) -> int:
    if stall < 0 or stall > 15:
        raise ValueError(f"invalid stall {stall}")
    high &= ~(0x1F << 41)
    return high | ((stall | (int(yield_flag) << 4)) << 41)


def differences(left: bytes, right: bytes) -> set[int]:
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
            raise ValueError("expected an sm_86 sidecar")
        if sidecar["summary"]["total_unknown_encoding"] != 0:
            raise ValueError("sidecar contains unknown controls")
        rows = sidecar["functions"].get(FUNCTION)
        if not isinstance(rows, list) or len(rows) != 72:
            raise ValueError("unexpected kernel shape")
        measured = [row for row in rows if 0x20 <= int(row["pc"]) <= 0x210]
        if len(measured) != MEASURED_INSTRUCTIONS:
            raise ValueError(f"expected {MEASURED_INSTRUCTIONS} measured instructions")
        if int(measured[0]["pc"]) != 0x20 or int(measured[-1]["pc"]) != 0x210:
            raise ValueError("measured timeline boundary changed")
        if any(
            int(right["pc"]) != int(left["pc"]) + 16
            for left, right in zip(measured, measured[1:])
        ):
            raise ValueError("measured timeline is not contiguous")
        for index, row in enumerate(measured):
            expected = "CS2R" if index % 4 == 0 else "NOP"
            if not row["opcode_text"].startswith(expected):
                raise ValueError(
                    f"timeline index {index}: expected {expected}, got {row['opcode_text']}"
                )

        with args.input.open("rb") as stream:
            elf = ELFFile(stream)
            flags = int(elf.header["e_flags"])
            abi = int(elf.header["e_ident"]["EI_ABIVERSION"])
            sm = (flags >> 8) & 0xFF if abi >= 8 else flags & 0xFF
            if sm != 86:
                raise ValueError(f"expected sm_86, got sm_{sm}")
            section = elf.get_section_by_name(f".text.{FUNCTION}")
            if section is None:
                raise ValueError(f"missing {FUNCTION}")
            section_offset = int(section.header["sh_offset"])
            section_size = int(section.header["sh_size"])

        allowed_offsets = {
            section_offset + int(row["pc"]) + byte
            for row in measured
            for byte in range(8, 16)
        }
        payloads = {}
        reports = {}
        for name, (base_stall, overrides) in VARIANTS.items():
            settings = {
                index: (base_stall, False)
                for index in range(MEASURED_INSTRUCTIONS)
            }
            for index, stall, yield_flag in overrides:
                settings[index] = (stall, yield_flag)
            output = bytearray(original)
            controls = []
            for index, row in enumerate(measured):
                pc = int(row["pc"])
                if pc % 16 or pc + 16 > section_size:
                    raise ValueError(f"{name}: invalid pc {pc:#x}")
                offset = section_offset + pc
                low, high = struct.unpack_from("<QQ", original, offset)
                stall, yield_flag = settings[index]
                struct.pack_into(
                    "<QQ", output, offset, low, control(high, stall, yield_flag)
                )
                controls.append(
                    {
                        "index": index,
                        "block": index // INSTRUCTIONS_PER_BLOCK,
                        "index_in_block": index % INSTRUCTIONS_PER_BLOCK,
                        "pc": pc,
                        "final_stall": stall,
                        "final_yield": int(yield_flag),
                    }
                )
            payload = bytes(output)
            changed = differences(original, payload)
            if not changed or not changed <= allowed_offsets:
                raise ValueError(f"{name}: changes escaped measured controls")
            destination = args.prefix.with_name(f"{args.prefix.name}_{name}.cubin")
            destination.write_bytes(payload)
            payloads[name] = payload
            reports[name] = {
                "output": str(destination),
                "base_stall": base_stall,
                "changed_bytes_from_native": len(changed),
                "stall4_controls": sum(stall == 4 for stall, _ in settings.values()),
                "yield_controls": sum(yield_flag for _, yield_flag in settings.values()),
                "baseline_controls": sum(
                    stall == base_stall and not yield_flag
                    for stall, yield_flag in settings.values()
                ),
                "controls": controls,
            }

        pair_differences = {}
        for name, (base_stall, _) in VARIANTS.items():
            baseline = f"baseline_s{base_stall}"
            if name != baseline:
                pair_differences[f"{baseline}_to_{name}"] = len(
                    differences(payloads[baseline], payloads[name])
                )
        args.report.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "architecture": "sm_86",
                    "input": str(args.input),
                    "function": FUNCTION,
                    "blocks": BLOCKS,
                    "instructions_per_block": INSTRUCTIONS_PER_BLOCK,
                    "measured_instructions": MEASURED_INSTRUCTIONS,
                    "measured_pc_first": int(measured[0]["pc"]),
                    "measured_pc_last": int(measured[-1]["pc"]),
                    "variants": reports,
                    "pair_differences": pair_differences,
                    "all_changes_inside_measured_control_words": True,
                    "identical_instruction_bytes_and_work": True,
                    "no_measured_branch": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"sm86 IBuffer control patch: {error}", file=sys.stderr)
        return 1
    for name in VARIANTS:
        print(args.prefix.with_name(f"{args.prefix.name}_{name}.cubin"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
