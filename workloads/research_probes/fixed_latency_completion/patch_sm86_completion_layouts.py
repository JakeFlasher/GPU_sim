#!/usr/bin/env python3
"""Create matched split-bank and bank3 completion cubins from one sm_86 cubin."""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path
from typing import Any, Sequence

from elftools.elf.elffile import ELFFile


FUNCTIONS = tuple(f"completion_d{distance}" for distance in (1, 2, 4, 8))
IMAD = re.compile(
    r"^IMAD R(?P<dst>\d+), R(?P<src1>\d+)(?:\.reuse)?, "
    r"R(?P<src2>\d+)(?:\.reuse)?, R(?P<src3>\d+)(?:\.reuse)?$"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("split_output", type=Path)
    parser.add_argument("bank3_output", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(argv)


def read_words(payload: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<QQ", payload, offset)


def encoded_sources(low: int, high: int) -> list[int]:
    return [(low >> 24) & 0xFF, (low >> 32) & 0xFF, high & 0xFF]


def patch_fields(low: int, high: int, sources: Sequence[int]) -> tuple[int, int]:
    if len(sources) != 3 or any(value < 0 or value >= 255 for value in sources):
        raise ValueError(f"invalid source-register triplet {sources!r}")
    low &= ~((0xFF << 24) | (0xFF << 32))
    low |= sources[0] << 24
    low |= sources[1] << 32
    high &= ~0xFF
    high |= sources[2]
    high &= ~(0xF << 58)
    return low, high


def byte_differences(left: bytes, right: bytes) -> set[int]:
    if len(left) != len(right):
        raise ValueError("binary sizes differ")
    return {index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]}


def choose(candidates: list[int], ordinal: int, excluded: set[int]) -> int:
    usable = [register for register in candidates if register not in excluded]
    if not usable:
        raise ValueError(f"no candidate remains after excluding {sorted(excluded)}")
    return usable[ordinal % len(usable)]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        original = args.input.read_bytes()
        sidecar = json.loads(args.sidecar.read_text(encoding="utf-8"))
        if sidecar.get("arch") != "sm_86":
            raise ValueError(f"{args.sidecar}: expected sm_86 sidecar")
        if sidecar.get("summary", {}).get("total_unknown_encoding") != 0:
            raise ValueError(f"{args.sidecar}: sidecar contains unknown controls")

        with args.input.open("rb") as stream:
            elf = ELFFile(stream)
            flags = int(elf.header["e_flags"])
            abi = int(elf.header["e_ident"]["EI_ABIVERSION"])
            sm = (flags >> 8) & 0xFF if abi >= 8 else flags & 0xFF
            if sm != 86:
                raise ValueError(f"{args.input}: expected sm_86 cubin, got sm_{sm}")
            sections: dict[str, tuple[int, int]] = {}
            for function in FUNCTIONS:
                section = elf.get_section_by_name(f".text.{function}")
                if section is None:
                    raise ValueError(f"{args.input}: missing .text.{function}")
                sections[function] = (
                    int(section.header["sh_offset"]),
                    int(section.header["sh_size"]),
                )

        split = bytearray(original)
        bank3 = bytearray(original)
        rows: list[dict[str, Any]] = []
        allowed_offsets: set[int] = set()
        for function in FUNCTIONS:
            records = sidecar["functions"].get(function)
            if not isinstance(records, list):
                raise ValueError(f"sidecar missing function {function}")
            hot: list[tuple[dict[str, Any], re.Match[str]]] = []
            for record in records:
                match = IMAD.fullmatch(record["opcode_text"])
                if match is not None:
                    hot.append((record, match))
            if len(hot) != 16:
                raise ValueError(f"{function}: expected 16 hot dependent IMADs, got {len(hot)}")

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
            by_bank = {
                bank: [register for register in factor_registers if register % 2 == bank]
                for bank in (0, 1)
            }
            if any(len(registers) < 2 for registers in by_bank.values()):
                raise ValueError(
                    f"{function}: factor registers do not cover two per bank: {by_bank}"
                )

            section_offset, section_size = sections[function]
            for ordinal, (record, match) in enumerate(hot):
                pc = int(record["pc"])
                if pc < 0 or pc + 16 > section_size or pc % 16 != 0:
                    raise ValueError(f"{function} pc {pc:#x}: invalid section offset")
                offset = section_offset + pc
                low, high = read_words(original, offset)
                destination = int(match.group("dst"))
                expected = [int(match.group(name)) for name in ("src1", "src2", "src3")]
                carried_addend = expected[2]
                if low & 0xFFFF != 0x7224:
                    raise ValueError(f"{function} pc {pc:#x}: not register-form bare IMAD")
                if (low >> 16) & 0xFF != destination:
                    raise ValueError(f"{function} pc {pc:#x}: destination encoding changed")
                if encoded_sources(low, high) != expected:
                    raise ValueError(
                        f"{function} pc {pc:#x}: encoded sources {encoded_sources(low, high)} "
                        f"!= sidecar {expected}"
                    )
                encoded_reuse = (high >> 58) & 0xF
                if encoded_reuse != int(record["control"]["reuse"]):
                    raise ValueError(f"{function} pc {pc:#x}: reuse decode mismatch")

                bank = carried_addend % 2
                same1 = choose(by_bank[bank], ordinal, {carried_addend})
                same2 = choose(by_bank[bank], ordinal + 1, {carried_addend, same1})
                opposite = choose(by_bank[1 - bank], ordinal, {carried_addend})
                split_sources = [same1, opposite, carried_addend]
                bank3_sources = [same1, same2, carried_addend]
                split_low, split_high = patch_fields(low, high, split_sources)
                bank3_low, bank3_high = patch_fields(low, high, bank3_sources)
                struct.pack_into("<QQ", split, offset, split_low, split_high)
                struct.pack_into("<QQ", bank3, offset, bank3_low, bank3_high)
                allowed_offsets.update(range(offset, offset + 16))
                rows.append(
                    {
                        "function": function,
                        "pc": pc,
                        "file_offset": offset,
                        "destination": destination,
                        "carried_addend": carried_addend,
                        "original_sources": expected,
                        "split_sources": split_sources,
                        "bank3_sources": bank3_sources,
                        "original_reuse": encoded_reuse,
                        "final_reuse": 0,
                        "split_bank_reads": [
                            sum(register % 2 == candidate for register in split_sources)
                            for candidate in (0, 1)
                        ],
                        "bank3_bank_reads": [
                            sum(register % 2 == candidate for register in bank3_sources)
                            for candidate in (0, 1)
                        ],
                    }
                )

        split_bytes = bytes(split)
        bank3_bytes = bytes(bank3)
        original_split = byte_differences(original, split_bytes)
        original_bank3 = byte_differences(original, bank3_bytes)
        pair_differences = byte_differences(split_bytes, bank3_bytes)
        if not original_split or not original_bank3 or not pair_differences:
            raise ValueError("one or more patch comparisons are unexpectedly identical")
        if not (original_split | original_bank3 | pair_differences) <= allowed_offsets:
            raise ValueError("binary differences escaped selected instruction words")
        if any(split[offset + 8 : offset + 16] != bank3[offset + 8 : offset + 16]
               for offset in {int(row["file_offset"]) for row in rows}):
            raise ValueError("matched pair changed an IMAD high/control word")
        if any(row["split_bank_reads"] not in ([2, 1], [1, 2]) for row in rows):
            raise ValueError("split layout is not 2+1")
        if any(row["bank3_bank_reads"] not in ([3, 0], [0, 3]) for row in rows):
            raise ValueError("bank3 layout is not 3+0")

        args.split_output.parent.mkdir(parents=True, exist_ok=True)
        args.bank3_output.parent.mkdir(parents=True, exist_ok=True)
        args.split_output.write_bytes(split_bytes)
        args.bank3_output.write_bytes(bank3_bytes)
        report = {
            "schema_version": 1,
            "architecture": "sm_86",
            "input": str(args.input),
            "split_output": str(args.split_output),
            "bank3_output": str(args.bank3_output),
            "functions": list(FUNCTIONS),
            "patched_instructions": len(rows),
            "original_to_split_changed_bytes": len(original_split),
            "original_to_bank3_changed_bytes": len(original_bank3),
            "split_to_bank3_changed_bytes": len(pair_differences),
            "all_changed_bytes_inside_selected_words": True,
            "pair_high_words_identical": True,
            "patches": rows,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"sm86 completion-layout patch: {error}", file=sys.stderr)
        return 1
    print(args.split_output)
    print(args.bank3_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
