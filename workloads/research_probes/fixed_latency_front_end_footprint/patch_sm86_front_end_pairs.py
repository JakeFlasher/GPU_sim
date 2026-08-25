#!/usr/bin/env python3
"""Patch equal-stall producer-distance pairs into calibrated sm_86 bodies."""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path
from typing import Sequence

from elftools.elf.elffile import ELFFile


FUNCTIONS = {
    "fixed": "_Z22footprint_fixed_searchPyPjPKj",
    "global": "_Z23footprint_global_searchPyPjPKjS2_",
}
REGISTER = re.compile(r"R(\d+)")
FIXED_PRODUCER = re.compile(r"IADD3 R(\d+), R(\d+), 0x4, RZ$")
GLOBAL_PRODUCER = re.compile(r"(?:IADD3|IMAD\.IADD) R(\d+),")
GLOBAL_LOAD = re.compile(r"LDG\.E R(\d+), \[R(\d+)\.64\]$")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("--unroll", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--calibration", action="store_true")
    return parser.parse_args(argv)


def control_with_stall(high: int, stall: int) -> int:
    if stall < 1 or stall > 15:
        raise ValueError(f"invalid stall {stall}")
    # Preserve dependence barriers, waits, and operand-reuse flags.  Selected
    # controls use canonical yield=0: nvdisasm rejects yield=1 combined with a
    # long (for example 12-cycle) LOP3 stall even though the bytes execute.
    high &= ~(0x1F << 41)
    return high | (stall << 41)


def differences(left: bytes, right: bytes) -> set[int]:
    return {
        index
        for index, (before, after) in enumerate(zip(left, right))
        if before != after
    }


def backward_loop(rows: list[dict]) -> list[dict]:
    branches = []
    for row in rows:
        text = row["opcode_text"]
        if not text.startswith("BRA 0x"):
            continue
        target = int(text.split()[1], 16)
        if target < int(row["pc"]):
            branches.append((target, int(row["pc"])))
    if len(branches) != 1:
        raise ValueError(f"expected one backward measured-loop branch, got {branches}")
    start, stop = branches[0]
    return [row for row in rows if start <= int(row["pc"]) < stop]


def register_tokens(text: str) -> list[int]:
    return [int(value) for value in REGISTER.findall(text)]


def fixed_sites(rows: list[dict], unroll: int) -> list[dict]:
    loop = backward_loop(rows)
    producer_indexes = [
        index
        for index, row in enumerate(loop)
        if FIXED_PRODUCER.fullmatch(row["opcode_text"])
    ]
    if len(producer_indexes) != unroll:
        raise ValueError(
            f"fixed U{unroll}: expected {unroll} producers, got {len(producer_indexes)}"
        )
    sites = []
    for ordinal, producer_index in enumerate(producer_indexes):
        producer = loop[producer_index]
        match = FIXED_PRODUCER.fullmatch(producer["opcode_text"])
        assert match is not None
        produced = int(match.group(1))
        checksum = int(match.group(2))
        limit = (
            producer_indexes[ordinal + 1]
            if ordinal + 1 < len(producer_indexes)
            else len(loop)
        )
        consumer_index = next(
            (
                index
                for index in range(producer_index + 1, limit)
                if loop[index]["opcode_text"].startswith("LOP3.LUT ")
                and produced in register_tokens(loop[index]["opcode_text"])[1:]
            ),
            None,
        )
        if consumer_index is None:
            raise ValueError(f"fixed U{unroll} body {ordinal}: consumer not found")
        consumer = loop[consumer_index]
        consumer_dest = register_tokens(consumer["opcode_text"])[0]
        compensator = next(
            (
                row
                for row in loop[consumer_index + 1 : limit]
                if row["opcode_text"].startswith(
                    ("IADD3 ", "IMAD.IADD ", "IMAD ")
                )
                and register_tokens(row["opcode_text"])[0] == checksum
                and consumer_dest in register_tokens(row["opcode_text"])[1:]
            ),
            None,
        )
        if compensator is None:
            raise ValueError(f"fixed U{unroll} body {ordinal}: compensator not found")
        interval = loop[producer_index:consumer_index]
        sites.append(
            {
                "ordinal": ordinal,
                "producer": producer,
                "consumer": consumer,
                "interval": interval,
                "compensator": compensator,
            }
        )
    return sites


def global_sites(rows: list[dict], unroll: int) -> list[dict]:
    loop = backward_loop(rows)
    load_indexes = [
        index
        for index, row in enumerate(loop)
        if GLOBAL_LOAD.fullmatch(row["opcode_text"])
    ]
    if len(load_indexes) != unroll:
        raise ValueError(
            f"global U{unroll}: expected {unroll} timed loads, got {len(load_indexes)}"
        )
    sites = []
    previous_load = -1
    for ordinal, load_index in enumerate(load_indexes):
        load = loop[load_index]
        load_match = GLOBAL_LOAD.fullmatch(load["opcode_text"])
        assert load_match is not None
        loaded = int(load_match.group(1))
        address_high = int(load_match.group(2)) + 1
        candidates = [
            index
            for index in range(previous_load + 1, load_index)
            if (match := GLOBAL_PRODUCER.match(loop[index]["opcode_text"]))
            and int(match.group(1)) == address_high
        ]
        if not candidates:
            raise ValueError(f"global U{unroll} body {ordinal}: producer not found")
        producer_index = candidates[-1]
        producer = loop[producer_index]
        next_load = (
            load_indexes[ordinal + 1]
            if ordinal + 1 < len(load_indexes)
            else len(loop)
        )
        compensator = next(
            (
                row
                for row in loop[load_index + 1 : next_load]
                if row["opcode_text"].startswith("LOP3.LUT ")
                and loaded in register_tokens(row["opcode_text"])[1:]
            ),
            None,
        )
        if compensator is None:
            raise ValueError(f"global U{unroll} body {ordinal}: compensator not found")
        sites.append(
            {
                "ordinal": ordinal,
                "producer": producer,
                "consumer": load,
                "interval": loop[producer_index:load_index],
                "compensator": compensator,
            }
        )
        previous_load = load_index
    return sites


def variant_controls(
    sites: list[dict], target_distance: int, compensator_stall: int
) -> tuple[dict[int, int], list[dict]]:
    controls: dict[int, int] = {}
    report = []
    for site in sites:
        interval = site["interval"]
        producer_stall = target_distance - (len(interval) - 1)
        if producer_stall < 1:
            raise ValueError(
                f"body {site['ordinal']}: distance {target_distance} is shorter "
                f"than its {len(interval)} issued instructions"
            )
        for index, row in enumerate(interval):
            controls[int(row["pc"])] = producer_stall if index == 0 else 1
        controls[int(site["compensator"]["pc"])] = compensator_stall
        report.append(
            {
                "ordinal": site["ordinal"],
                "producer_pc": int(site["producer"]["pc"]),
                "producer_opcode": site["producer"]["opcode_text"],
                "consumer_pc": int(site["consumer"]["pc"]),
                "consumer_opcode": site["consumer"]["opcode_text"],
                "intervening_pcs": [int(row["pc"]) for row in interval[1:]],
                "producer_stall": producer_stall,
                "distance": target_distance,
                "compensator_pc": int(site["compensator"]["pc"]),
                "compensator_opcode": site["compensator"]["opcode_text"],
                "compensator_stall": compensator_stall,
            }
        )
    return controls, report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        original = args.input.read_bytes()
        sidecar = json.loads(args.sidecar.read_text(encoding="utf-8"))
        if sidecar.get("arch") != "sm_86":
            raise ValueError("expected an sm_86 sidecar")
        if sidecar["summary"]["total_unknown_encoding"] != 0:
            raise ValueError("sidecar contains unknown controls")
        rows_by_kind = {}
        sites_by_kind = {}
        for kind, function in FUNCTIONS.items():
            rows = sidecar["functions"].get(function)
            if not isinstance(rows, list):
                raise ValueError(f"missing function {function}")
            rows_by_kind[kind] = {int(row["pc"]): row for row in rows}
            sites_by_kind[kind] = (
                fixed_sites(rows, args.unroll)
                if kind == "fixed"
                else global_sites(rows, args.unroll)
            )

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
                    raise ValueError(f"missing text section for {function}")
                sections[kind] = (
                    int(section.header["sh_offset"]),
                    int(section.header["sh_size"]),
                )

        active_sites = {
            "fixed": sites_by_kind["fixed"],
            # Some unrolled shapes schedule a low-address MOV between the
            # high-half producer and LDG.  Its own fixed-result latency makes
            # a d5/d6 high-half experiment impossible without corrupting the
            # low address, so those bodies remain identical neutral controls.
            "global": [
                site
                for site in sites_by_kind["global"]
                if all(
                    row["opcode_text"] == "NOP"
                    for row in site["interval"][1:]
                )
            ],
        }
        if not active_sites["global"]:
            raise ValueError("no global body supports an uncontaminated d5/d6 pair")

        variants = {
            # The IMAD checksum feeds the next producer.  Stalls 5/4 are the
            # shortest compensating pair that remains semantically valid in
            # every U shape; shorter values fail the U2/U4/U8 recurrence
            # calibration even when the final checksum happens to converge.
            "fixed_bypass": ("fixed", 4, 6),
            "fixed_retired": ("fixed", 5, 5),
            "global_frontier": ("global", 5, 12),
            "global_late": ("global", 6, 11),
        }
        if args.calibration:
            if args.unroll != 1:
                raise ValueError("calibration variants are only defined for U1")
            variants.update(
                {
                    "fixed_invalid": ("fixed", 3, 7),
                    "global_invalid": ("global", 4, 13),
                }
            )

        payloads = {}
        reports = {}
        selected_offsets: dict[str, set[int]] = {kind: set() for kind in FUNCTIONS}
        for name, (kind, distance, compensator_stall) in variants.items():
            controls, bodies = variant_controls(
                active_sites[kind], distance, compensator_stall
            )
            output = bytearray(original)
            section_offset, section_size = sections[kind]
            controls_report = []
            for pc, stall in sorted(controls.items()):
                if pc % 16 or pc + 16 > section_size:
                    raise ValueError(f"{name}: invalid pc {pc:#x}")
                offset = section_offset + pc
                low, high = struct.unpack_from("<QQ", original, offset)
                struct.pack_into(
                    "<QQ", output, offset, low, control_with_stall(high, stall)
                )
                selected_offsets[kind].update(range(offset + 8, offset + 16))
                controls_report.append(
                    {
                        "pc": pc,
                        "opcode_text": rows_by_kind[kind][pc]["opcode_text"],
                        "native_control": rows_by_kind[kind][pc]["control"],
                        "final_stall": stall,
                    }
                )
            payload = bytes(output)
            changed = differences(original, payload)
            if not changed <= selected_offsets[kind]:
                raise ValueError(f"{name}: changes escaped selected controls")
            destination = args.prefix.with_name(f"{args.prefix.name}_{name}.cubin")
            destination.write_bytes(payload)
            payloads[name] = payload
            reports[name] = {
                "output": str(destination),
                "kind": kind,
                "producer_consumer_distance": distance,
                "body_count_between_branches": args.unroll,
                "patched_body_count_between_branches": len(active_sites[kind]),
                "neutral_body_ordinals": [
                    site["ordinal"]
                    for site in sites_by_kind[kind]
                    if site not in active_sites[kind]
                ],
                "selected_stall_sum": sum(controls.values()),
                "changed_bytes_from_native": len(changed),
                "bodies": bodies,
                "controls": controls_report,
            }

        pair_differences = {}
        for left, right, kind in (
            ("fixed_bypass", "fixed_retired", "fixed"),
            ("global_frontier", "global_late", "global"),
        ):
            changed = differences(payloads[left], payloads[right])
            if not changed or not changed <= selected_offsets[kind]:
                raise ValueError(f"{left}/{right}: diff escaped selected controls")
            if reports[left]["selected_stall_sum"] != reports[right]["selected_stall_sum"]:
                raise ValueError(f"{left}/{right}: selected stall sums differ")
            pair_differences[f"{left}_to_{right}"] = len(changed)

        args.report.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "architecture": "sm_86",
                    "input": str(args.input),
                    "unroll": args.unroll,
                    "functions": FUNCTIONS,
                    "variants": reports,
                    "pair_differences": pair_differences,
                    "one_backward_branch_per_unrolled_group": True,
                    "exact_producer_consumer_bodies": {
                        kind: len(sites) for kind, sites in sites_by_kind.items()
                    },
                    "all_changes_inside_selected_control_words": True,
                    "equal_static_stall_sum_within_pairs": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"sm86 front-end pair patch: {error}", file=sys.stderr)
        return 1
    for name in variants:
        print(args.prefix.with_name(f"{args.prefix.name}_{name}.cubin"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
