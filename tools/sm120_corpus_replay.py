#!/usr/bin/env python3
"""Replay every captured trace, full first and fast second, without NCU inputs.

Outputs are native simulator reports in JSONL, one row per recursive trace
directory, including failures. Static sidecars are derived from the captured
metadata by the same transport adapter used by sm120_replay_bench.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import time

from sm120_replay_bench import (
    DEFAULT_CARD,
    DEFAULT_SIM,
    discover_captures,
    load_static_metadata,
    synthesize_static_sidecar,
)


def replay(capture: Path, relative: str, control: Path, args, mode: str, logs: Path) -> dict:
    command = [str(args.sim_bin), "--trace", str(capture), "--control", str(control),
               "--arch-card", str(args.arch_card), "--sm120-alpha", "--mode", "flat"]
    if mode == "fast":
        command.append("--fast-frontier-only")
    stderr_path = logs / (relative + ".stderr")
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter_ns()
    with stderr_path.open("x") as stderr:
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=stderr,
                                    timeout=args.timeout, check=False)
            wall_ns = time.perf_counter_ns() - started
            try:
                row = json.loads(result.stdout)
                if not isinstance(row, dict):
                    raise ValueError("simulator output is not an object")
            except (ValueError, UnicodeError) as error:
                row = {"status": "output_error", "detail": str(error)}
            if result.returncode:
                row["status"] = "replay_error"
            else:
                row.setdefault("status", "ok")
            row["returncode"] = result.returncode
            if row["status"] != "ok":
                stderr_path.with_suffix(".stdout").write_bytes(result.stdout)
        except subprocess.TimeoutExpired:
            wall_ns = time.perf_counter_ns() - started
            row = {"status": "timeout", "returncode": None}
        except OSError as error:
            wall_ns = time.perf_counter_ns() - started
            row = {"status": "launch_error", "detail": str(error), "returncode": None}
    row.update(canonical_capture=relative, capture_dir=str(capture),
               driver_wall_ms=wall_ns / 1_000_000)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, default=Path("/root/sm120_rtx_5090_capture"))
    parser.add_argument("--output", type=Path, required=True, help="new output directory outside the corpus")
    parser.add_argument("--sim-bin", type=Path, default=DEFAULT_SIM)
    parser.add_argument("--arch-card", type=Path, default=DEFAULT_CARD)
    parser.add_argument("--mode", choices=("both", "full", "fast"), default="both")
    parser.add_argument("--timeout", type=float, default=1200, help="seconds per complete capture replay")
    args = parser.parse_args()
    args.capture_root = args.capture_root.resolve()
    args.output = args.output.resolve()
    args.sim_bin = args.sim_bin.resolve()
    args.arch_card = args.arch_card.resolve()
    if not 0 < args.timeout < float("inf"):
        parser.error("--timeout must be positive and finite")
    if args.output.is_relative_to(args.capture_root):
        parser.error("outputs must be outside the raw capture corpus")
    captures = discover_captures(args.capture_root, None)
    if not captures:
        parser.error("no recursive captures found")
    args.output.mkdir(parents=True, exist_ok=False)
    controls = args.output / "controls"
    prepared = []
    for capture in captures:
        relative = capture.relative_to(args.capture_root / "captures").as_posix()
        control = controls / (relative + ".control_bits.json")
        error = None
        try:
            metadata = load_static_metadata(capture)
            if metadata is None:
                raise ValueError("native static metadata missing")
            control.parent.mkdir(parents=True, exist_ok=True)
            synthesize_static_sidecar(capture, metadata, control)
        except (OSError, ValueError, KeyError, TypeError) as failure:
            error = str(failure)
        prepared.append((capture, relative, control, error))
    modes = ("full", "fast") if args.mode == "both" else (args.mode,)
    summary = {"capture_root": str(args.capture_root), "sim_bin": str(args.sim_bin),
               "arch_card": str(args.arch_card), "captures_discovered": len(captures),
               "scope": "all recursively discovered traces; none selected by replay outcome",
               "ncu_used_as_input": False, "modes": {}}
    for mode in modes:
        started = time.perf_counter_ns()
        counts = Counter()
        launches = 0
        with (args.output / f"{mode}.jsonl").open("x") as output:
            for index, (capture, relative, control, error) in enumerate(prepared, 1):
                row = ({"canonical_capture": relative, "capture_dir": str(capture),
                        "status": "control_error", "detail": error} if error else
                       replay(capture, relative, control, args, mode, args.output / f"{mode}_logs"))
                output.write(json.dumps(row, separators=(",", ":")) + "\n")
                output.flush()
                counts[row["status"]] += 1
                launches += len(row.get("launches" if mode == "full" else "fast_frontier", []))
                print(f"{mode} {index}/{len(captures)} {relative}: {row['status']}", flush=True)
        summary["modes"][mode] = {"status_counts": dict(counts), "launches": launches,
                                  "driver_total_wall_ns": time.perf_counter_ns() - started}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return int(any(set(row["status_counts"]) != {"ok"} for row in summary["modes"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
