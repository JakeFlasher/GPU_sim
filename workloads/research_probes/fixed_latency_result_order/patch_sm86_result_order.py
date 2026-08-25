#!/usr/bin/env python3
"""Canonicalize the exact sm_86 HADD2/FFMA producer-order pair controls."""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path
from typing import Any, Sequence

from elftools.elf.elffile import ELFFile


FUNCTIONS = {
    "fh_separated": "_Z12result_orderILb0EEvPyPjPKjS3_",
    "hf_coincident": "_Z12result_orderILb1EEvPyPjPKjS3_",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(argv)


def destination(opcode_text: str) -> int:
    matched = re.match(r"\S+ R(\d+),", opcode_text)
    if matched is None:
        raise ValueError(f"cannot parse destination from {opcode_text!r}")
    return int(matched.group(1))


def canonical_control(high: int, stall: int) -> int:
    # high[41:44] is stall, high[45] is yield, and high[58:61] is reuse.
    high &= ~(0x1F << 41)
    high &= ~(0xF << 58)
    high |= stall << 41
    return high


def function_rows(sidecar: dict[str, Any], name: str) -> list[dict[str, Any]]:
    rows = sidecar["functions"].get(name)
    if not isinstance(rows, list):
        raise ValueError(f"sidecar missing {name}")
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        original = args.input.read_bytes()
        sidecar = json.loads(args.sidecar.read_text(encoding="utf-8"))
        if sidecar.get("arch") != "sm_86" or sidecar["summary"]["total_unknown_encoding"] != 0:
            raise ValueError("expected a fully decoded sm_86 sidecar")

        rows_by_variant = {
            variant: function_rows(sidecar, function)
            for variant, function in FUNCTIONS.items()
        }
        left = rows_by_variant["fh_separated"]
        right = rows_by_variant["hf_coincident"]
        if len(left) != 48 or len(right) != 48:
            raise ValueError(f"expected two 48-instruction functions, got {len(left)}/{len(right)}")
        differing_pcs: list[int] = []
        for a, b in zip(left, right):
            if a["pc"] != b["pc"]:
                raise ValueError("function PCs are not aligned")
            if a["pc"] in {0x100, 0x110}:
                differing_pcs.append(int(a["pc"]))
                if a["control"] != b["control"] or a["predicate"] != b["predicate"]:
                    raise ValueError(f"producer control/predicate mismatch at {a['pc']:#x}")
                if destination(a["opcode_text"]) != destination(b["opcode_text"]):
                    raise ValueError(f"producer destination mismatch at {a['pc']:#x}")
            elif any(a[field] != b[field] for field in ("opcode_text", "predicate", "low_word", "high_word", "control")):
                raise ValueError(f"non-producer functions differ at {a['pc']:#x}")
        if differing_pcs != [0x100, 0x110]:
            raise ValueError(f"unexpected differing PCs {differing_pcs}")

        expected_orders = {
            "fh_separated": ("FFMA", "HADD2"),
            "hf_coincident": ("HADD2", "FFMA"),
        }
        producer_rows: dict[str, list[dict[str, Any]]] = {}
        for variant, rows in rows_by_variant.items():
            producers = [row for row in rows if row["pc"] in {0x100, 0x110}]
            heads = tuple(row["opcode_text"].split()[0] for row in producers)
            if heads != expected_orders[variant]:
                raise ValueError(f"{variant}: expected {expected_orders[variant]}, got {heads}")
            destinations = [destination(row["opcode_text"]) for row in producers]
            if destinations != [11, 4] or destinations[0] % 2 == destinations[1] % 2:
                raise ValueError(f"{variant}: expected fixed opposite-bank R11/R4 destinations")
            producer_rows[variant] = producers

        sections: dict[str, tuple[int, int]] = {}
        with args.input.open("rb") as stream:
            elf = ELFFile(stream)
            flags = int(elf.header["e_flags"])
            abi = int(elf.header["e_ident"]["EI_ABIVERSION"])
            sm = (flags >> 8) & 0xFF if abi >= 8 else flags & 0xFF
            if sm != 86:
                raise ValueError(f"expected sm_86, got sm_{sm}")
            for variant, function in FUNCTIONS.items():
                section = elf.get_section_by_name(f".text.{function}")
                if section is None:
                    raise ValueError(f"missing .text.{function}")
                sections[variant] = (
                    int(section.header["sh_offset"]),
                    int(section.header["sh_size"]),
                )

        output = bytearray(original)
        allowed: set[int] = set()
        report_rows = []
        for variant, rows in producer_rows.items():
            section_offset, section_size = sections[variant]
            for ordinal, row in enumerate(rows):
                pc = int(row["pc"])
                if pc % 16 != 0 or pc + 16 > section_size:
                    raise ValueError(f"{variant}: invalid producer PC {pc:#x}")
                offset = section_offset + pc
                low, high = struct.unpack_from("<QQ", original, offset)
                stall = 1 if ordinal == 0 else 2
                patched_high = canonical_control(high, stall)
                struct.pack_into("<QQ", output, offset, low, patched_high)
                allowed.update(range(offset + 8, offset + 16))
                report_rows.append(
                    {
                        "variant": variant,
                        "function": FUNCTIONS[variant],
                        "pc": pc,
                        "opcode_text": row["opcode_text"],
                        "destination": destination(row["opcode_text"]),
                        "original_control": row["control"],
                        "final_stall": stall,
                        "final_yield": 0,
                        "final_reuse": 0,
                        "low_unchanged": True,
                    }
                )

        payload = bytes(output)
        changed = {
            index
            for index, (before, after) in enumerate(zip(original, payload))
            if before != after
        }
        if not changed or not changed <= allowed:
            raise ValueError("changes escaped selected producer control words")
        args.output.write_bytes(payload)
        args.report.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "architecture": "sm_86",
                    "input": str(args.input),
                    "output": str(args.output),
                    "functions": FUNCTIONS,
                    "changed_bytes": len(changed),
                    "all_changes_inside_producer_control_words": True,
                    "orders": {
                        "fh_separated": ["FFMA", "HADD2"],
                        "hf_coincident": ["HADD2", "FFMA"],
                    },
                    "fixed_destinations_by_pc": {"0x100": 11, "0x110": 4},
                    "opposite_destination_banks": True,
                    "producers": report_rows,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"sm86 result-order patch: {error}", file=sys.stderr)
        return 1
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

