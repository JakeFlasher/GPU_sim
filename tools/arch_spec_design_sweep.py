#!/usr/bin/env python3
"""Run a bounded custom-architecture SIMT timing sweep.

This is a model-relative design probe. It does not consume NCU/rocprof data
and does not claim physical GPU accuracy. The binary must be the matching
`sim-arch` custom timing probe.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import tempfile
from typing import Any


def run_case(
    binary: pathlib.Path,
    base_spec: dict[str, Any],
    program: pathlib.Path,
    width: int,
    memory_issue_width: int,
) -> dict[str, Any]:
    spec = json.loads(json.dumps(base_spec))
    spec["execution"]["simt_width"] = width
    for resource in spec["resources"]:
        if resource["name"] == "memory":
            resource["issue_width"] = memory_issue_width
    with tempfile.TemporaryDirectory(prefix="arch-spec-sweep-") as directory:
        spec_path = pathlib.Path(directory) / "spec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        completed = subprocess.run(
            [str(binary), str(spec_path), str(program)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"case width={width} memory_issue_width={memory_issue_width} failed: "
                f"{completed.stderr.strip()}"
            )
        result = json.loads(completed.stdout)
    return {
        "simt_width": width,
        "memory_issue_width": memory_issue_width,
        "final_cycle": result["final_cycle"],
        "event_count": len(result["events"]),
        "semantic_event_count": len(result["semantic_events"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--spec", type=pathlib.Path, required=True)
    parser.add_argument("--program", type=pathlib.Path, required=True)
    parser.add_argument("--width", type=int, action="append", default=[])
    parser.add_argument("--memory-issue-width", type=int, action="append", default=[])
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    widths = args.width or [4, 8, 16]
    issue_widths = args.memory_issue_width or [1, 2]
    if any(width <= 0 or width > 64 for width in widths):
        parser.error("--width values must be in 1..=64")
    if any(issue <= 0 for issue in issue_widths):
        parser.error("--memory-issue-width values must be positive")

    base_spec = json.loads(args.spec.read_text(encoding="utf-8"))
    rows = [
        run_case(args.binary, base_spec, args.program, width, issue_width)
        for width in widths
        for issue_width in issue_widths
    ]
    document = {
        "kind": "model_relative_custom_simt_design_sweep",
        "spec": str(args.spec),
        "program": str(args.program),
        "rows": rows,
        "interpretation": "hypothetical functional/timing model; no silicon claim",
    }
    encoded = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
