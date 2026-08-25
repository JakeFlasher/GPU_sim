#!/usr/bin/env python3
"""Create exact sm_86 HADD2/FFMA collision and separated-control cubins."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Sequence

from elftools.elf.elffile import ELFFile


FUNCTION = "_Z18result_queue_pilotPyPjPKjPKf"
VARIANTS = {"collision": 1, "separated": 3}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(argv)


def patch_control(high: int, stall: int) -> int:
    # CUDA 13.2 suppresses HADD2's textual .reuse operands when yield is
    # cleared.  Clear the encoded reuse flags too so disassembly, the exact
    # operand projector, and hardware all consume the same canonical control.
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
        records = sidecar["functions"].get(FUNCTION)
        if not isinstance(records, list):
            raise ValueError(f"sidecar missing {FUNCTION}")
        hadd = [row for row in records if row["opcode_text"].startswith("HADD2 ")]
        ffma = [row for row in records if row["opcode_text"].startswith("FFMA ")]
        if len(hadd) != 4 or len(ffma) != 4:
            raise ValueError(f"expected four HADD2 and four FFMA, got {len(hadd)}/{len(ffma)}")
        with args.input.open("rb") as stream:
            elf = ELFFile(stream)
            section = elf.get_section_by_name(f".text.{FUNCTION}")
            if section is None:
                raise ValueError(f"missing .text.{FUNCTION}")
            section_offset = int(section.header["sh_offset"])
            section_size = int(section.header["sh_size"])
            flags = int(elf.header["e_flags"])
            abi = int(elf.header["e_ident"]["EI_ABIVERSION"])
            sm = (flags >> 8) & 0xFF if abi >= 8 else flags & 0xFF
            if sm != 86:
                raise ValueError(f"expected sm_86, got sm_{sm}")

        allowed = {
            section_offset + int(row["pc"]) + byte
            for row in hadd
            for byte in range(8, 16)
        }
        outputs = {}
        reports = {}
        for variant, stall in VARIANTS.items():
            output = bytearray(original)
            rows = []
            for row in hadd:
                pc = int(row["pc"])
                if pc < 0 or pc + 16 > section_size or pc % 16 != 0:
                    raise ValueError(f"invalid HADD2 pc {pc:#x}")
                offset = section_offset + pc
                low, high = struct.unpack_from("<QQ", original, offset)
                patched_high = patch_control(high, stall)
                struct.pack_into("<QQ", output, offset, low, patched_high)
                rows.append(
                    {
                        "pc": pc,
                        "opcode_text": row["opcode_text"],
                        "original_control": row["control"],
                        "final_stall": stall,
                        "final_yield": 0,
                        "final_reuse": 0,
                        "low_unchanged": True,
                    }
                )
            payload = bytes(output)
            diff = {index for index, pair in enumerate(zip(original, payload)) if pair[0] != pair[1]}
            if not diff or not diff <= allowed:
                raise ValueError(f"{variant}: changes escaped HADD2 control words")
            destination = args.prefix.with_name(f"{args.prefix.name}_{variant}.cubin")
            destination.write_bytes(payload)
            outputs[variant] = str(destination)
            reports[variant] = {
                "output": str(destination),
                "changed_bytes": len(diff),
                "hadd2": rows,
            }
        args.report.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "architecture": "sm_86",
                    "input": str(args.input),
                    "function": FUNCTION,
                    "ffma_unchanged": [
                        {"pc": int(row["pc"]), "opcode_text": row["opcode_text"], "control": row["control"]}
                        for row in ffma
                    ],
                    "outputs": outputs,
                    "variants": reports,
                    "all_changes_inside_hadd2_control_words": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"sm86 result-pair patch: {error}", file=sys.stderr)
        return 1
    for variant in VARIANTS:
        print(outputs[variant])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
