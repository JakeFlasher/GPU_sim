#!/usr/bin/env python3
"""Canonicalize the compiler-native sm_86 same-bank HADD2/FFMA order pair."""

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
    "fh_separated": "_Z16result_same_bankILb0EEvPyPjPKjS3_",
    "hf_coincident": "_Z16result_same_bankILb1EEvPyPjPKjS3_",
}
PRODUCER_PCS = (0xE0, 0xF0)
ORACLE_PC = 0x1C0


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


def source_bank_demand(opcode_text: str) -> tuple[int, int]:
    registers = [int(value) for value in re.findall(r"\bR(\d+)\b", opcode_text)]
    if len(registers) < 2:
        raise ValueError(f"cannot parse regular sources from {opcode_text!r}")
    sources = registers[1:]
    return (sum(register % 2 == 0 for register in sources), sum(register % 2 == 1 for register in sources))


def canonical_control(high: int, stall: int) -> int:
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

        differing_pcs = []
        for a, b in zip(left, right):
            if a["pc"] != b["pc"]:
                raise ValueError("function PCs are not aligned")
            pc = int(a["pc"])
            if pc in {*PRODUCER_PCS, ORACLE_PC}:
                differing_pcs.append(pc)
                if a["control"] != b["control"] or a["predicate"] != b["predicate"]:
                    raise ValueError(f"control/predicate mismatch at {pc:#x}")
                if destination(a["opcode_text"]) != destination(b["opcode_text"]):
                    raise ValueError(f"destination mismatch at {pc:#x}")
                if pc == ORACLE_PC:
                    a_registers = sorted(map(int, re.findall(r"\bR(\d+)\b", a["opcode_text"])[1:]))
                    b_registers = sorted(map(int, re.findall(r"\bR(\d+)\b", b["opcode_text"])[1:]))
                    if a_registers != b_registers or not a["opcode_text"].startswith("LOP3.LUT "):
                        raise ValueError("post-clock oracle is not a commutative source permutation")
            elif any(a[field] != b[field] for field in ("opcode_text", "predicate", "low_word", "high_word", "control")):
                raise ValueError(f"functions differ outside treatment/oracle at {pc:#x}")
        if differing_pcs != [*PRODUCER_PCS, ORACLE_PC]:
            raise ValueError(f"unexpected differing PCs {differing_pcs}")

        expected_orders = {
            "fh_separated": ("FFMA", "HADD2"),
            "hf_coincident": ("HADD2", "FFMA"),
        }
        producer_rows: dict[str, list[dict[str, Any]]] = {}
        demands: dict[str, dict[str, tuple[int, int]]] = {}
        for variant, rows in rows_by_variant.items():
            producers = [row for row in rows if int(row["pc"]) in PRODUCER_PCS]
            heads = tuple(row["opcode_text"].split()[0] for row in producers)
            if heads != expected_orders[variant]:
                raise ValueError(f"{variant}: expected {expected_orders[variant]}, got {heads}")
            destinations = [destination(row["opcode_text"]) for row in producers]
            if destinations != [4, 8] or any(register % 2 for register in destinations):
                raise ValueError(f"{variant}: expected same-bank R4/R8 destinations")
            producer_rows[variant] = producers
            demands[variant] = {
                row["opcode_text"].split()[0]: source_bank_demand(row["opcode_text"])
                for row in producers
            }
        if demands["fh_separated"] != demands["hf_coincident"]:
            raise ValueError(f"per-opcode source-bank demand differs: {demands}")

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
                sections[variant] = (int(section.header["sh_offset"]), int(section.header["sh_size"]))

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
                        "source_bank_demand": source_bank_demand(row["opcode_text"]),
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
                    "fixed_destinations_by_relative_pc": {"0xe0": 4, "0xf0": 8},
                    "same_destination_bank": 0,
                    "per_opcode_source_bank_demand": demands["fh_separated"],
                    "timed_consumers_byte_identical": True,
                    "post_clock_oracle_pc": ORACLE_PC,
                    "post_clock_oracle_commutative_permutation": True,
                    "producers": report_rows,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"sm86 same-bank order patch: {error}", file=sys.stderr)
        return 1
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

