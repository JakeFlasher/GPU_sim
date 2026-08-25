#!/usr/bin/env python3
"""Create equal-stall sm_86 fixed-bypass and global-frontier pairs."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Sequence

from elftools.elf.elffile import ELFFile


FUNCTIONS = {
    "fixed": "_Z19bypass_fixed_searchPyPjPKj",
    "global": "_Z20bypass_global_searchPyPjPKjS2_",
}
EXPECTED_LENGTHS = {"fixed": 40, "global": 40}
VARIANTS = {
    "fixed_bypass": ("fixed", ((0xE0, 4), (0x100, 3))),
    "fixed_retired": ("fixed", ((0xE0, 5), (0x100, 2))),
    "global_frontier": ("global", ((0xC0, 1), (0xD0, 4), (0x110, 12))),
    "global_late": ("global", ((0xC0, 1), (0xD0, 5), (0x110, 11))),
}
EXPECTED_OPCODES = {
    ("fixed", 0xE0): "LOP3.LUT R5, R5, 0x4, RZ, 0x3c, !PT",
    ("fixed", 0xF0): "LOP3.LUT R6, R5, R4, RZ, 0x3c, !PT",
    ("fixed", 0x100): "IADD3 R9, R6, R9, RZ",
    ("global", 0xC0): "IMAD.IADD R5, R3, 0x1, R9",
    ("global", 0xD0): "NOP",
    ("global", 0xE0): "LDG.E R4, [R4.64]",
    ("global", 0x110): "IMAD.IADD R11, R4, 0x1, R11",
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
    return high | (stall << 41)


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
        if sidecar.get("arch") != "sm_86" or sidecar["summary"]["total_unknown_encoding"] != 0:
            raise ValueError("expected a fully decoded sm_86 sidecar")
        rows_by_kind = {}
        for kind, function in FUNCTIONS.items():
            rows = sidecar["functions"].get(function)
            if not isinstance(rows, list) or len(rows) != EXPECTED_LENGTHS[kind]:
                raise ValueError(f"{kind}: unexpected function shape")
            rows_by_kind[kind] = {int(row["pc"]): row for row in rows}
        for (kind, pc), opcode in EXPECTED_OPCODES.items():
            if rows_by_kind[kind][pc]["opcode_text"] != opcode:
                raise ValueError(f"{kind} pc {pc:#x}: unexpected opcode")

        with args.input.open("rb") as stream:
            elf = ELFFile(stream)
            flags = int(elf.header["e_flags"])
            abi = int(elf.header["e_ident"]["EI_ABIVERSION"])
            sm = (flags >> 8) & 0xFF if abi >= 8 else flags & 0xFF
            if sm != 86:
                raise ValueError(f"expected sm_86, got sm_{sm}")
            sections = {}
            for kind, function in FUNCTIONS.items():
                section = elf.get_section_by_name(f".text.{function}")
                if section is None:
                    raise ValueError(f"missing {function}")
                sections[kind] = (
                    int(section.header["sh_offset"]),
                    int(section.header["sh_size"]),
                )

        payloads = {}
        reports = {}
        allowed_by_kind = {
            kind: {
                sections[kind][0] + pc + byte
                for variant_kind, controls in VARIANTS.values()
                if variant_kind == kind
                for pc, _ in controls
                for byte in range(8, 16)
            }
            for kind in FUNCTIONS
        }
        for name, (kind, controls) in VARIANTS.items():
            output = bytearray(original)
            rows_report = []
            section_offset, section_size = sections[kind]
            for pc, stall in controls:
                if pc % 16 or pc + 16 > section_size:
                    raise ValueError(f"{name}: invalid pc {pc:#x}")
                offset = section_offset + pc
                low, high = struct.unpack_from("<QQ", original, offset)
                struct.pack_into(
                    "<QQ", output, offset, low, canonical_control(high, stall)
                )
                rows_report.append(
                    {
                        "pc": pc,
                        "opcode_text": rows_by_kind[kind][pc]["opcode_text"],
                        "original_control": rows_by_kind[kind][pc]["control"],
                        "final_stall": stall,
                        "final_yield": 0,
                        "final_reuse": 0,
                    }
                )
            payload = bytes(output)
            changed = differences(original, payload)
            if not changed <= allowed_by_kind[kind]:
                raise ValueError(f"{name}: changes escaped selected controls")
            destination = args.prefix.with_name(f"{args.prefix.name}_{name}.cubin")
            destination.write_bytes(payload)
            payloads[name] = payload
            reports[name] = {
                "output": str(destination),
                "kind": kind,
                "producer_consumer_distance": (
                    4
                    if name == "fixed_bypass"
                    else 5
                    if name in {"fixed_retired", "global_frontier"}
                    else 6
                ),
                "selected_stall_sum": sum(stall for _, stall in controls),
                "changed_bytes_from_native": len(changed),
                "controls": rows_report,
            }

        pair_differences = {}
        for left, right, kind in (
            ("fixed_bypass", "fixed_retired", "fixed"),
            ("global_frontier", "global_late", "global"),
        ):
            changed = differences(payloads[left], payloads[right])
            if not changed or not changed <= allowed_by_kind[kind]:
                raise ValueError(f"{left}/{right}: diff escaped control words")
            pair_differences[f"{left}_to_{right}"] = len(changed)

        if reports["fixed_bypass"]["selected_stall_sum"] != reports["fixed_retired"]["selected_stall_sum"]:
            raise ValueError("fixed pair stall sums differ")
        if reports["global_frontier"]["selected_stall_sum"] != reports["global_late"]["selected_stall_sum"]:
            raise ValueError("global pair stall sums differ")
        args.report.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "architecture": "sm_86",
                    "input": str(args.input),
                    "functions": FUNCTIONS,
                    "variants": reports,
                    "pair_differences": pair_differences,
                    "all_changes_inside_selected_control_words": True,
                    "equal_instruction_work": True,
                    "equal_static_stall_sum_within_pairs": True,
                    "fixed_boundary": {
                        "invalid_distance": 3,
                        "bypass_distance": 4,
                        "post_retirement_control_distance": 5,
                    },
                    "global_boundary": {
                        "invalid_distance": 4,
                        "register_available_distance": 5,
                        "late_control_distance": 6,
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"sm86 bypass-pair patch: {error}", file=sys.stderr)
        return 1
    for name in VARIANTS:
        print(args.prefix.with_name(f"{args.prefix.name}_{name}.cubin"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
