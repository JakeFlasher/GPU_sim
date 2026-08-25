#!/usr/bin/env python3
"""Create the invalid/valid sm_86 bypass boundary calibration cubins."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Sequence

from elftools.elf.elffile import ELFFile


FIXED = "_Z19bypass_fixed_searchPyPjPKj"
CANARY = "_Z20bypass_global_canaryPjPKjS1_"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("input", type=Path)
    result.add_argument("sidecar", type=Path)
    result.add_argument("--prefix", type=Path, required=True)
    result.add_argument("--report", type=Path, required=True)
    return result


def control(high: int, stall: int) -> int:
    high &= ~(0x1F << 41)
    high &= ~(0xF << 58)
    return high | (stall << 41)


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        raw = args.input.read_bytes()
        sidecar = json.loads(args.sidecar.read_text(encoding="utf-8"))
        if sidecar.get("arch") != "sm_86" or sidecar["summary"]["total_unknown_encoding"] != 0:
            raise ValueError("expected fully decoded sm_86 sidecar")
        fixed_rows = {int(row["pc"]): row for row in sidecar["functions"][FIXED]}
        canary_rows = {int(row["pc"]): row for row in sidecar["functions"][CANARY]}
        if fixed_rows[0xE0]["opcode_text"] != "LOP3.LUT R5, R5, 0x4, RZ, 0x3c, !PT":
            raise ValueError("unexpected fixed producer")
        if canary_rows[0x70]["opcode_text"] != "IADD3 R5, R2, c[0x0][0x16c], R2":
            raise ValueError("unexpected canary producer")
        if canary_rows[0x80]["opcode_text"] != "LDG.E R5, [R4.64]":
            raise ValueError("unexpected canary consumer")
        with args.input.open("rb") as stream:
            elf = ELFFile(stream)
            offsets = {}
            for name, function in (("fixed", FIXED), ("canary", CANARY)):
                section = elf.get_section_by_name(f".text.{function}")
                if section is None:
                    raise ValueError(f"missing {function}")
                offsets[name] = int(section.header["sh_offset"])
        specifications = {
            "fixed_invalid": ("fixed", ((0xE0, 3), (0x100, 4))),
            "canary_d4": ("canary", ((0x70, 4),)),
            "canary_d5": ("canary", ((0x70, 5),)),
        }
        reports = {}
        for name, (kind, controls) in specifications.items():
            output = bytearray(raw)
            for pc, stall in controls:
                offset = offsets[kind] + pc
                low, high = struct.unpack_from("<QQ", raw, offset)
                struct.pack_into("<QQ", output, offset, low, control(high, stall))
            destination = args.prefix.with_name(f"{args.prefix.name}_{name}.cubin")
            destination.write_bytes(output)
            reports[name] = {
                "output": str(destination),
                "kind": kind,
                "controls": [{"pc": pc, "stall": stall} for pc, stall in controls],
                "expected": (
                    "incorrect_checksum"
                    if name == "fixed_invalid"
                    else "cuda_error_illegal_address"
                    if name == "canary_d4"
                    else "correct"
                ),
            }
        args.report.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "architecture": "sm_86",
                    "input": str(args.input),
                    "variants": reports,
                    "fixed_d3_invalid_d4_valid": True,
                    "global_d4_invalid_d5_valid": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"sm86 bypass calibration patch: {error}", file=sys.stderr)
        return 1
    for name in specifications:
        print(args.prefix.with_name(f"{args.prefix.name}_{name}.cubin"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
