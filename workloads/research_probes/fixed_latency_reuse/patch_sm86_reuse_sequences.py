#!/usr/bin/env python3
"""Create matched sm_86 RFC sequences by patching IMAD sources and reuse bits."""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path
from typing import Any, Sequence

from elftools.elf.elffile import ELFFile


FUNCTIONS = ("reuse_u10", "reuse_u11", "reuse_u12", "reuse_u20", "reuse_u22")
IMAD = re.compile(
    r"^IMAD R(?P<dst>\d+), R(?P<src1>\d+)(?:\.reuse)?, "
    r"R(?P<src2>\d+)(?:\.reuse)?, R(?P<src3>\d+)(?:\.reuse)?$"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("output_a", type=Path)
    parser.add_argument("output_b", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(argv)


def read_words(payload: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<QQ", payload, offset)


def encoded_sources(low: int, high: int) -> list[int]:
    return [(low >> 24) & 0xFF, (low >> 32) & 0xFF, high & 0xFF]


def patch_fields(
    low: int, high: int, sources: Sequence[int], reuse: int
) -> tuple[int, int]:
    if len(sources) != 3 or any(value < 0 or value >= 255 for value in sources):
        raise ValueError(f"invalid source triplet {sources!r}")
    if reuse < 0 or reuse > 0x7:
        raise ValueError(f"invalid three-source reuse mask {reuse:#x}")
    low &= ~((0xFF << 24) | (0xFF << 32))
    low |= sources[0] << 24
    low |= sources[1] << 32
    high &= ~0xFF
    high |= sources[2]
    # nvdisasm only admits/displays the patched reuse modifier on this sm_86
    # register-IMAD form when the control Yield bit is set. Canonicalize Yield
    # identically in both variants; all other non-reuse control fields remain
    # native and the loop has sixteen independent streams.
    high |= 1 << 45
    high &= ~(0xF << 58)
    high |= reuse << 58
    return low, high


def layouts(
    function: str,
    bank_ordinal: int,
    bank_total: int,
    bank: int,
    carried: int,
    banks: dict[int, list[int]],
) -> tuple[list[int], int, list[int], int, str]:
    lane = bank_ordinal % 4
    p = banks[bank]
    other = banks[1 - bank]
    x, y, q = p[:3]
    z = other[0]
    canonical_x = [x, q, carried]
    canonical_y = [y, q, carried]
    canonical_q = [q, x, carried]

    if function == "reuse_u10":
        # Alternating X/Y requests ensure every allocation is replaced before
        # the same register can return, including across the loop boundary.
        sources = (
            canonical_q
            if bank_ordinal == bank_total - 1
            else (canonical_x if bank_ordinal % 2 == 0 else canonical_y)
        )
        return sources, 0, sources, 0b001, "allocation_without_hit"
    if function == "reuse_u11":
        sources = canonical_x
        return sources, 0, sources, 0b001 if bank_ordinal % 2 == 0 else 0, "paired_hit"
    if function == "reuse_u12":
        sources = canonical_y if lane == 3 else canonical_x
        reuse_a = 0b001 if lane == 0 else 0
        reuse_b = 0b001 if lane in {0, 1} else 0
        return sources, reuse_a, sources, reuse_b, "retained_second_hit"
    if function == "reuse_u20":
        if lane == 0:
            return canonical_x, 0b001, canonical_x, 0b001, "position_seed"
        if lane == 1:
            return canonical_x, 0b001, [q, x, carried], 0b010, "position_move"
        if lane == 2:
            return canonical_x, 0, canonical_x, 0, "position_probe"
        return canonical_y, 0, canonical_y, 0, "position_reset"
    if function == "reuse_u22":
        if lane == 0:
            return canonical_x, 0b001, canonical_x, 0b001, "eviction_seed"
        if lane == 1:
            return [z, y, carried], 0, [y, z, carried], 0, "eviction_treatment"
        if lane == 2:
            return canonical_x, 0, canonical_x, 0, "eviction_probe"
        return canonical_y, 0, canonical_y, 0, "eviction_reset"
    raise ValueError(f"unknown function {function}")


def differences(left: bytes, right: bytes) -> set[int]:
    if len(left) != len(right):
        raise ValueError("binary sizes differ")
    return {index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        original = args.input.read_bytes()
        sidecar = json.loads(args.sidecar.read_text(encoding="utf-8"))
        if sidecar.get("arch") != "sm_86":
            raise ValueError(f"{args.sidecar}: expected sm_86")
        if sidecar.get("summary", {}).get("total_unknown_encoding") != 0:
            raise ValueError(f"{args.sidecar}: unknown control encodings")

        with args.input.open("rb") as stream:
            elf = ELFFile(stream)
            flags = int(elf.header["e_flags"])
            abi = int(elf.header["e_ident"]["EI_ABIVERSION"])
            sm = (flags >> 8) & 0xFF if abi >= 8 else flags & 0xFF
            if sm != 86:
                raise ValueError(f"{args.input}: expected sm_86, got sm_{sm}")
            sections = {}
            for function in FUNCTIONS:
                section = elf.get_section_by_name(f".text.{function}")
                if section is None:
                    raise ValueError(f"missing .text.{function}")
                sections[function] = (
                    int(section.header["sh_offset"]),
                    int(section.header["sh_size"]),
                )

        output_a = bytearray(original)
        output_b = bytearray(original)
        rows: list[dict[str, Any]] = []
        allowed: set[int] = set()
        for function in FUNCTIONS:
            records = sidecar["functions"].get(function)
            if not isinstance(records, list):
                raise ValueError(f"sidecar missing {function}")
            hot = []
            for record in records:
                match = IMAD.fullmatch(record["opcode_text"])
                if match is not None:
                    hot.append((record, match))
            if len(hot) != 16:
                raise ValueError(f"{function}: expected 16 register IMADs, got {len(hot)}")
            carried_registers = {
                int(match.group(name))
                for _, match in hot
                for name in ("dst", "src3")
            }
            factor_registers = sorted(
                {
                    int(match.group(name))
                    for _, match in hot
                    for name in ("src1", "src2")
                }
                - carried_registers
            )
            banks = {
                bank: [register for register in factor_registers if register % 2 == bank]
                for bank in (0, 1)
            }
            if any(len(registers) < 3 for registers in banks.values()):
                raise ValueError(f"{function}: need three factor registers per bank, got {banks}")

            section_offset, section_size = sections[function]
            bank_totals = {
                bank: sum(int(match.group("src3")) % 2 == bank for _, match in hot)
                for bank in (0, 1)
            }
            bank_ordinals = {0: 0, 1: 0}
            for ordinal, (record, match) in enumerate(sorted(hot, key=lambda item: item[0]["pc"])):
                pc = int(record["pc"])
                if pc < 0 or pc + 16 > section_size or pc % 16 != 0:
                    raise ValueError(f"{function} pc {pc:#x}: invalid offset")
                offset = section_offset + pc
                low, high = read_words(original, offset)
                destination = int(match.group("dst"))
                expected_sources = [
                    int(match.group(name)) for name in ("src1", "src2", "src3")
                ]
                carried = expected_sources[2]
                bank = carried % 2
                if low & 0xFFFF != 0x7224:
                    raise ValueError(f"{function} pc {pc:#x}: not register bare IMAD")
                if (low >> 16) & 0xFF != destination:
                    raise ValueError(f"{function} pc {pc:#x}: destination mismatch")
                if encoded_sources(low, high) != expected_sources:
                    raise ValueError(f"{function} pc {pc:#x}: source mismatch")
                if (high >> 58) & 0xF != int(record["control"]["reuse"]):
                    raise ValueError(f"{function} pc {pc:#x}: reuse mismatch")
                sources_a, reuse_a, sources_b, reuse_b, role = layouts(
                    function,
                    bank_ordinals[bank],
                    bank_totals[bank],
                    bank,
                    carried,
                    banks,
                )
                bank_ordinals[bank] += 1
                low_a, high_a = patch_fields(low, high, sources_a, reuse_a)
                low_b, high_b = patch_fields(low, high, sources_b, reuse_b)
                struct.pack_into("<QQ", output_a, offset, low_a, high_a)
                struct.pack_into("<QQ", output_b, offset, low_b, high_b)
                allowed.update(range(offset, offset + 16))
                rows.append(
                    {
                        "function": function,
                        "ordinal": ordinal,
                        "pc": pc,
                        "file_offset": offset,
                        "destination": destination,
                        "original_sources": expected_sources,
                        "sources_a": sources_a,
                        "reuse_a": reuse_a,
                        "sources_b": sources_b,
                        "reuse_b": reuse_b,
                        "role": role,
                        "bank_reads_a": [
                            sum(register % 2 == bank for register in sources_a)
                            for bank in (0, 1)
                        ],
                        "bank_reads_b": [
                            sum(register % 2 == bank for register in sources_b)
                            for bank in (0, 1)
                        ],
                    }
                )

        bytes_a = bytes(output_a)
        bytes_b = bytes(output_b)
        original_a = differences(original, bytes_a)
        original_b = differences(original, bytes_b)
        pair = differences(bytes_a, bytes_b)
        if not original_a or not original_b or not pair:
            raise ValueError("unexpectedly identical patch comparison")
        if not (original_a | original_b | pair) <= allowed:
            raise ValueError("binary difference escaped selected IMAD words")
        if any(row["bank_reads_a"] != row["bank_reads_b"] for row in rows):
            raise ValueError("pair changed per-instruction bank demand")

        args.output_a.parent.mkdir(parents=True, exist_ok=True)
        args.output_b.parent.mkdir(parents=True, exist_ok=True)
        args.output_a.write_bytes(bytes_a)
        args.output_b.write_bytes(bytes_b)
        report = {
            "schema_version": 1,
            "architecture": "sm_86",
            "input": str(args.input),
            "output_a": str(args.output_a),
            "output_b": str(args.output_b),
            "functions": list(FUNCTIONS),
            "patched_instructions": len(rows),
            "original_to_a_changed_bytes": len(original_a),
            "original_to_b_changed_bytes": len(original_b),
            "a_to_b_changed_bytes": len(pair),
            "all_changed_bytes_inside_selected_words": True,
            "pair_bank_demand_identical": True,
            "patches": rows,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"sm86 RFC patch: {error}", file=sys.stderr)
        return 1
    print(args.output_a)
    print(args.output_b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
