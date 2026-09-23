#!/usr/bin/env python3
"""Compare native SM86 replay with a semantic-event canonical round trip."""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import tempfile
from typing import Any


def run_json(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, check=False, text=True, capture_output=True, timeout=300)
    if completed.returncode:
        raise SystemExit(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stderr:\n{completed.stderr}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise SystemExit(f"command emitted invalid JSON: {error}\n{completed.stdout[:1000]}") from error
    if not isinstance(value, dict):
        raise SystemExit("command emitted a non-object JSON document")
    return value


def replay(args: argparse.Namespace, trace: pathlib.Path) -> dict[str, Any]:
    command = [
        str(args.sim_run),
        "--trace",
        str(trace),
        "--control",
        str(args.control),
        "--arch-card",
        str(args.arch_card),
        "--mode",
        args.mode,
        "--cache-scope",
        args.cache_scope,
    ]
    if args.pchase_card is not None:
        command.extend(["--pchase-card", str(args.pchase_card)])
    if args.memory_admission_feedback is not None:
        command.extend(["--memory-admission-feedback", str(args.memory_admission_feedback)])
    return run_json(command)


def launch_projection(document: dict[str, Any]) -> list[dict[str, Any]]:
    launches = document.get("launches")
    if not isinstance(launches, list) or not launches:
        raise SystemExit("replay output has no nonempty launches array")
    projection = []
    for launch in launches:
        if not isinstance(launch, dict):
            raise SystemExit("replay output has a non-object launch")
        projection.append({key: launch[key] for key in (
            "device_id", "stream_id", "kernel_id", "function_unique_id", "kernel_name",
            "grid", "block", "cluster", "counters", "timing", "coverage",
        )})
    return projection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=pathlib.Path, required=True)
    parser.add_argument("--control", type=pathlib.Path, required=True)
    parser.add_argument("--arch-card", type=pathlib.Path, required=True)
    parser.add_argument("--pchase-card", type=pathlib.Path)
    parser.add_argument("--recorded-replay", type=pathlib.Path)
    parser.add_argument("--sim-run", type=pathlib.Path, default=pathlib.Path("sim/target/release/sim-run"))
    parser.add_argument("--semantic-probe", type=pathlib.Path, default=pathlib.Path("sim/target/debug/semantic-probe"))
    parser.add_argument("--mode", choices=("flat", "l1", "l1-l2"), default="l1-l2")
    parser.add_argument("--cache-scope", choices=("launch", "stream-l2"), default="launch")
    parser.add_argument("--memory-admission-feedback", type=pathlib.Path)
    args = parser.parse_args()

    semantic = run_json([str(args.semantic_probe), str(args.trace), str(args.control)])
    losses = semantic["counts"]["loss_counts"]
    # Unknown classifications remain in the loss report. Native replay still
    # consumes the original sidecar; wire/replay equivalence does not establish
    # functional completeness or permission to summarize an unknown opcode.
    native = replay(args, args.trace)
    if args.recorded_replay is not None:
        recorded = json.loads(args.recorded_replay.read_text())
        if launch_projection(recorded) != launch_projection(native):
            raise SystemExit("native replay differs from the supplied recorded replay")

    with tempfile.TemporaryDirectory(prefix="sm86-semantic-roundtrip-") as temporary:
        roundtrip_root = pathlib.Path(temporary) / "trace"
        roundtrip_source_summary = run_json([
            str(args.semantic_probe),
            str(args.trace),
            str(args.control),
            "--roundtrip",
            str(roundtrip_root),
        ])
        if roundtrip_source_summary != semantic:
            raise SystemExit("source semantics changed between inspection and round trip")
        roundtrip = replay(args, roundtrip_root)

    if launch_projection(native) != launch_projection(roundtrip):
        raise SystemExit("semantic round trip changed SM86 counters, timing, or coverage")
    print(json.dumps({
        "semantic": semantic,
        "native_launches": len(native["launches"]),
        "roundtrip_matches_native": True,
        "classification_complete": not bool(losses),
        "compared_fields": ["launch_identity", "counters", "timing", "coverage"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
