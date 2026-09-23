#!/usr/bin/env python3
"""Project a narrowly recognized Vortex ELF through the shared Rust semantics.

The ELF adapter is intentionally not a general decoder. It recognizes one
zero-offset integer ``lw/lw/add/sw`` idiom, emits a custom SIMT program with
caller-supplied synthetic addresses, and lets ``custom_simt_timing`` construct
and validate the canonical ``sim-semantics`` events. This probe records that
boundary rather than claiming binary compatibility.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Any


def run_checked(command: list[str]) -> str:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"command failed ({completed.returncode}): {message}")
    return completed.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=pathlib.Path, required=True)
    parser.add_argument("--timing-binary", type=pathlib.Path, required=True)
    parser.add_argument("--objdump", type=pathlib.Path, required=True)
    parser.add_argument("--elf", type=pathlib.Path, required=True)
    parser.add_argument("--spec", type=pathlib.Path, required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--src0", type=int, default=400)
    parser.add_argument("--src1", type=int, default=800)
    parser.add_argument("--dst", type=int, default=1200)
    parser.add_argument("--source-revision", default="unknown")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="vortex-semantic-") as directory:
        program = pathlib.Path(directory) / "adapted_program.json"
        adapter_output = run_checked(
            [
                sys.executable,
                str(args.adapter),
                "--objdump",
                str(args.objdump),
                "--elf",
                str(args.elf),
                "--architecture",
                args.architecture,
                "--width",
                str(args.width),
                "--src0",
                str(args.src0),
                "--src1",
                str(args.src1),
                "--dst",
                str(args.dst),
                "--output",
                str(program),
            ]
        )
        adapter_metadata = json.loads(adapter_output)
        timing = json.loads(
            run_checked(
                [str(args.timing_binary), str(args.spec), str(program)]
            )
        )

    events = timing.get("semantic_events", [])
    if not events:
        raise RuntimeError("timing probe emitted no semantic events")
    if len(events) != len(timing.get("events", [])):
        raise RuntimeError("timing and semantic-event counts disagree")
    if any(event.get("native", {}).get("isa") != "custom_simt_u32" for event in events):
        raise RuntimeError("semantic stream contains an unexpected ISA")

    operations = collections.Counter(event.get("operation") for event in events)
    memory_families = collections.Counter(
        event["memory"]["family"]
        for event in events
        if event.get("memory") is not None
    )
    result: dict[str, Any] = {
        "kind": "vortex_elf_shared_semantic_event_probe",
        "source_revision": args.source_revision,
        "elf": str(args.elf),
        "architecture_spec": str(args.spec),
        "architecture": timing["architecture"],
        "isa": timing["isa"],
        "simt_width": timing["simt_width"],
        "adapter": adapter_metadata,
        "binding_mode": "caller_supplied_synthetic_byte_addresses",
        "semantic_event_count": len(events),
        "operation_counts": dict(sorted(operations.items())),
        "memory_family_counts": dict(sorted(memory_families.items())),
        "final_cycle": timing["final_cycle"],
        "timing_events": timing["events"],
        "semantic_events": events,
        "interpretation": (
            "real ELF pattern recognized and projected through Rust sim-semantics; "
            "not a general RISC-V decoder, pointer-semantics proof, RTL timing "
            "match, or silicon claim"
        ),
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"vortex_elf_shared_semantic_probe: {error}", file=sys.stderr)
        raise SystemExit(1) from error
