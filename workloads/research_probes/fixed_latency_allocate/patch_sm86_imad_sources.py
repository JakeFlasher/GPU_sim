#!/usr/bin/env python3
"""Patch only bare-IMAD register fields in one existing sm_86 cubin."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Sequence

from elftools.elf.elffile import ELFFile


def read_words(payload: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<QQ", payload, offset)


def sources(low: int, high: int) -> list[int]:
    return [(low >> 24) & 0xFF, (low >> 32) & 0xFF, high & 0xFF]


def destination(low: int) -> int:
    return (low >> 16) & 0xFF


def patch_source_fields(low: int, high: int, values: Sequence[int]) -> tuple[int, int]:
    if len(values) != 3 or any(value < 0 or value >= 255 for value in values):
        raise ValueError(f"invalid register triplet {values!r}")
    low &= ~((0xFF << 24) | (0xFF << 32))
    low |= values[0] << 24
    low |= values[1] << 32
    high &= ~0xFF
    high |= values[2]
    return low, high


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("input", type=Path)
    result.add_argument("output", type=Path)
    result.add_argument("spec", type=Path)
    result.add_argument("--report", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        original = args.input.read_bytes()
        specification = json.loads(args.spec.read_text(encoding="utf-8"))
        function = specification["function"]
        patches = specification["patches"]
        section_name = f".text.{function}"
        with args.input.open("rb") as stream:
            elf = ELFFile(stream)
            flags = int(elf.header["e_flags"])
            abi = int(elf.header["e_ident"]["EI_ABIVERSION"])
            sm = (flags >> 8) & 0xFF if abi >= 8 else flags & 0xFF
            if sm != 86:
                raise ValueError(f"{args.input}: expected sm_86 cubin, got sm_{sm}")
            section = elf.get_section_by_name(section_name)
            if section is None:
                raise ValueError(f"{args.input}: missing {section_name}")
            section_offset = int(section.header["sh_offset"])
            section_size = int(section.header["sh_size"])

        output = bytearray(original)
        changed_offsets: set[int] = set()
        rows: list[dict[str, object]] = []
        seen_pcs: set[int] = set()
        for patch in patches:
            pc = int(patch["pc"], 0) if isinstance(patch["pc"], str) else int(patch["pc"])
            if pc in seen_pcs:
                raise ValueError(f"duplicate patch pc {pc:#x}")
            seen_pcs.add(pc)
            if pc < 0 or pc + 16 > section_size or pc % 16 != 0:
                raise ValueError(f"pc {pc:#x} is outside/alignment-invalid for {section_name}")
            offset = section_offset + pc
            low, high = read_words(original, offset)
            expected_destination = int(patch["destination"])
            expected_sources = [int(value) for value in patch["expected_sources"]]
            replacement_sources = [int(value) for value in patch["replacement_sources"]]
            if low & 0xFFFF != 0x7224:
                raise ValueError(f"pc {pc:#x}: expected register IMAD encoding, low={low:#018x}")
            if destination(low) != expected_destination:
                raise ValueError(
                    f"pc {pc:#x}: destination R{destination(low)} != R{expected_destination}"
                )
            if sources(low, high) != expected_sources:
                raise ValueError(
                    f"pc {pc:#x}: sources {sources(low, high)} != {expected_sources}"
                )
            encoded_reuse = (high >> 58) & 0xF
            if encoded_reuse != int(patch.get("expected_reuse", 0)):
                raise ValueError(
                    f"pc {pc:#x}: reuse {encoded_reuse:#x} != expected "
                    f"{int(patch.get('expected_reuse', 0)):#x}"
                )
            patched_low, patched_high = patch_source_fields(
                low, high, replacement_sources
            )
            struct.pack_into("<QQ", output, offset, patched_low, patched_high)
            for byte_offset in range(offset, offset + 16):
                if original[byte_offset] != output[byte_offset]:
                    changed_offsets.add(byte_offset)
            rows.append(
                {
                    "pc": pc,
                    "file_offset": offset,
                    "destination": expected_destination,
                    "old_sources": expected_sources,
                    "new_sources": replacement_sources,
                    "old_low": f"0x{low:016x}",
                    "new_low": f"0x{patched_low:016x}",
                    "high_unchanged": patched_high == high,
                    "reuse": encoded_reuse,
                }
            )

        if len(rows) != len(patches):
            raise ValueError("patch cardinality changed")
        allowed = {
            section_offset + int(row["pc"]) + byte
            for row in rows
            for byte in range(16)
        }
        all_differences = {
            index
            for index, (before, after) in enumerate(zip(original, output))
            if before != after
        }
        if not all_differences or not all_differences <= allowed:
            raise ValueError("binary differences escaped the selected instruction words")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(output)
        report = {
            "schema_version": 1,
            "architecture": "sm_86",
            "input": str(args.input),
            "output": str(args.output),
            "function": function,
            "section_offset": section_offset,
            "section_size": section_size,
            "file_size": len(original),
            "patched_instructions": len(rows),
            "changed_bytes": len(all_differences),
            "all_changed_bytes_inside_selected_words": True,
            "patches": rows,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"sm86 IMAD source patch: {error}", file=sys.stderr)
        return 1
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

