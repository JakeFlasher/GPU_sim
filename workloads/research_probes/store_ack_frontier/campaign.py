#!/usr/bin/env python3
"""Run the sm_86 STORE acknowledgement-frontier assay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "results/captures/sm86_store_ack_frontier_20260901_v1"
BINARY = HERE / "store_ack_frontier"
TRACE_BINARY = HERE / "store_ack_frontier_trace"
CUDA_BIN = Path("/usr/local/cuda-13.2/bin")
CONTROL_TOOL = REPO_ROOT / "tools/sass_control/emit_control_bits_sidecar.py"
NVBIT_CAPTURE = (
    REPO_ROOT / "workloads/research_probes/ldgsts_burst_release/capture_nvbit.py"
)
SIM_RUN = REPO_ROOT / "sim/target/release/sim-run"
ARCH_CARD = REPO_ROOT / "arch_profiles/cards/sm_86.json"
PCHASE_CARD = REPO_ROOT / "data/pchase/sm_86_static_card.json"
ACK_CANDIDATE = (
    REPO_ROOT
    / "workloads/research_probes/l1tex_admission_backpressure/memory_write_completion_feedback_store_201.json"
)
ARMS = ("control", "store_none", "store_device")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def gpu_state() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "-i",
        "0",
        "--query-gpu=index,name,uuid,compute_cap,driver_version,persistence_mode,pstate,clocks.current.graphics,clocks.current.sm,clocks.current.memory,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": command,
        "return_code": completed.returncode,
        "output": completed.stdout.strip(),
    }


def revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return completed.stdout.strip()


@dataclass(frozen=True, slots=True)
class Cell:
    stage: str
    blocks: int
    depth: int
    active_lanes: int
    layout: str = "line_unique"
    gap_steps: int = 0

    @property
    def key(self) -> str:
        return (
            f"b{self.blocks}_d{self.depth}_l{self.active_lanes}_"
            f"{self.layout}_g{self.gap_steps}"
        )

    def target(self, mode: str, trials: int, *, trace: bool = False) -> list[str]:
        return [
            str(TRACE_BINARY if trace else BINARY),
            mode,
            self.layout,
            str(self.blocks),
            str(self.depth),
            str(self.active_lanes),
            str(self.gap_steps),
            str(trials),
        ]


def configured_cells() -> list[Cell]:
    return [
        Cell("identification", 1, 4, 32),
        Cell("identification", 1, 16, 32),
        Cell("identification", 84, 4, 32),
        Cell("identification", 84, 16, 32),
        Cell("calibration", 1, 16, 0),
        Cell("holdout", 1, 64, 32),
        Cell("holdout", 1, 16, 8),
        Cell("holdout", 1, 32, 32, "line_reuse"),
        Cell("holdout", 1, 32, 32, "line_unique", 64),
        Cell("holdout", 84, 64, 32),
    ]


def selected_cells(specification: str | None, stage: str | None) -> list[Cell]:
    cells = configured_cells()
    if stage is not None:
        cells = [cell for cell in cells if cell.stage == stage]
    if specification is None:
        return cells
    requested = {value for value in specification.split(",") if value}
    selected = [cell for cell in cells if cell.key in requested]
    missing = requested - {cell.key for cell in selected}
    if missing:
        raise RuntimeError(f"unknown cell key(s): {', '.join(sorted(missing))}")
    return selected


def parse_records(stdout: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if line.startswith("{"):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError("target JSON record is not an object")
            records.append(value)
    summaries = [record for record in records if record.get("record") == "summary"]
    if not summaries or any(record.get("correct") is not True for record in summaries):
        raise RuntimeError("target emitted no complete correct summary set")
    if any(record.get("correct") is not True for record in records if record.get("record") == "trial"):
        raise RuntimeError("target emitted an incorrect trial")
    return records


def run_captured(
    command: Sequence[str],
    destination: Path,
    *,
    timeout: int,
) -> tuple[subprocess.CompletedProcess[str], list[dict[str, Any]]]:
    destination.mkdir(parents=True, exist_ok=True)
    before = gpu_state()
    started = now()
    completed = subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    finished = now()
    after = gpu_state()
    (destination / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (destination / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    write_json(
        destination / "run.json",
        {
            "schema_version": 1,
            "command": list(command),
            "cwd": str(REPO_ROOT),
            "started_at": started,
            "finished_at": finished,
            "return_code": completed.returncode,
            "gpu_before": before,
            "gpu_after": after,
            "binary": {
                "path": str(BINARY.relative_to(REPO_ROOT)),
                "sha256": sha256(BINARY),
                "size_bytes": BINARY.stat().st_size,
            },
            "revision": revision(),
        },
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stderr}"
        )
    records = parse_records(completed.stdout)
    write_json(destination / "records.json", records)
    return completed, records


def build(output: Path) -> None:
    destination = output / "static"
    destination.mkdir(parents=True, exist_ok=True)
    command = ["make", "-B", "-C", str(HERE)]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (destination / "build.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (destination / "build.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"probe build failed:\n{completed.stdout}\n{completed.stderr}")
    write_json(
        destination / "binaries.json",
        {
            "schema_version": 1,
            "revision": revision(),
            "nvcc": str(CUDA_BIN / "nvcc"),
            "binaries": {
                path.name: {
                    "sha256": sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in (BINARY, TRACE_BINARY)
            },
        },
    )


def emit_manifest(output: Path) -> None:
    if not BINARY.is_file() or not TRACE_BINARY.is_file():
        build(output)
    cells = configured_cells()
    write_json(
        output / "metadata/matrix.json",
        {
            "schema_version": 1,
            "probe": "sm86_store_ack_frontier_assay",
            "cells": [{**asdict(cell), "key": cell.key} for cell in cells],
            "arms": list(ARMS),
            "queue_gate_ns": 250000,
            "timer": "%globaltimer; target-specific nanosecond units",
            "native_independence_unit": "process repeat",
            "ncu_role": "comparison_only_not_used_for_selection",
        },
    )
    commands = {
        "gpu": gpu_state(),
        "nvcc": [str(CUDA_BIN / "nvcc"), "--version"],
        "cuobjdump": [str(CUDA_BIN / "cuobjdump"), "--version"],
    }
    environment: dict[str, Any] = {
        "schema_version": 1,
        "observed_at": now(),
        "revision": revision(),
        "commands": {},
    }
    environment["commands"]["gpu"] = commands["gpu"]
    for name in ("nvcc", "cuobjdump"):
        completed = subprocess.run(
            commands[name],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        environment["commands"][name] = {
            "command": commands[name],
            "return_code": completed.returncode,
            "output": completed.stdout.strip(),
        }
    write_json(output / "metadata/environment.json", environment)


def run_correctness(output: Path, cells: Sequence[Cell], trials: int) -> None:
    for index, cell in enumerate(cells, 1):
        destination = output / "direct/correctness" / cell.key
        run_captured(cell.target("paired", trials), destination, timeout=300)
        print(f"correctness {index:02d}/{len(cells):02d} {cell.key}", flush=True)


def run_native(
    output: Path,
    cells: Sequence[Cell],
    repeats: int,
    trials: int,
    repeat_start: int,
    force: bool,
) -> None:
    total = len(cells) * repeats
    progress = 0
    for repeat in range(repeat_start, repeat_start + repeats):
        ordered = list(cells)
        if repeat % 2 == 0:
            ordered.reverse()
        for cell in ordered:
            progress += 1
            destination = output / "direct/native" / cell.key / f"repeat-{repeat}"
            if (destination / "records.json").is_file() and not force:
                print(
                    f"native {progress:03d}/{total:03d} {cell.key} r{repeat} (existing)",
                    flush=True,
                )
                continue
            if destination.exists() and any(destination.iterdir()):
                raise RuntimeError(
                    f"native destination is nonempty: {destination}; use a new root"
                )
            run_captured(cell.target("paired", trials), destination, timeout=600)
            print(
                f"native {progress:03d}/{total:03d} {cell.key} r{repeat}",
                flush=True,
            )


def run_static(output: Path) -> None:
    if not BINARY.is_file() or not TRACE_BINARY.is_file():
        raise RuntimeError("probe binaries are missing; run the build action first")
    destination = output / "static"
    commands = {
        "sass": [str(CUDA_BIN / "cuobjdump"), "--dump-sass", str(TRACE_BINARY)],
        "resources": [
            str(CUDA_BIN / "cuobjdump"),
            "--dump-resource-usage",
            str(TRACE_BINARY),
        ],
    }
    for name, command in commands.items():
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        (destination / f"{name}.txt").write_text(completed.stdout, encoding="utf-8")
        (destination / f"{name}.stderr.txt").write_text(
            completed.stderr, encoding="utf-8"
        )
        if completed.returncode != 0:
            raise RuntimeError(f"{name} extraction failed")
    sidecar = destination / "store_ack_frontier_trace.control_bits.json"
    command = [
        str(REPO_ROOT / ".venv/bin/python"),
        str(CONTROL_TOOL),
        "--binary",
        str(TRACE_BINARY),
        "--arch",
        "sm_86",
        "--toolkit-bindir",
        str(CUDA_BIN),
        "--out",
        str(sidecar),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (destination / "sidecar.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (destination / "sidecar.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0 or not sidecar.is_file():
        raise RuntimeError(f"control sidecar extraction failed:\n{completed.stderr}")


def run_nvbit(
    output: Path,
    cells: Sequence[Cell],
    repeats: int,
    repeat_start: int,
) -> None:
    bridge_keys = {
        "b1_d4_l32_line_unique_g0",
        "b84_d16_l32_line_unique_g0",
    }
    bridge = [cell for cell in cells if cell.key in bridge_keys]
    if not bridge:
        raise RuntimeError("selected cells contain no NVBit bridge cell")
    total = len(bridge) * len(ARMS) * repeats
    progress = 0
    for repeat in range(repeat_start, repeat_start + repeats):
        for cell in bridge:
            for arm in ARMS:
                progress += 1
                destination = output / "nvbit" / cell.key / f"repeat-{repeat}" / arm
                if (destination / "complete.json").is_file():
                    print(
                        f"nvbit {progress:02d}/{total:02d} {cell.key} {arm} r{repeat} (existing)",
                        flush=True,
                    )
                    continue
                command = [
                    "systemd-run",
                    "--user",
                    "--scope",
                    "--quiet",
                    "--property=Delegate=yes",
                    sys.executable,
                    str(NVBIT_CAPTURE),
                    "--output",
                    str(destination),
                    "--cell",
                    f"store_ack_{cell.key}_{arm}_r{repeat}",
                    "--binary",
                    str(TRACE_BINARY),
                    "--gpu",
                    "0",
                    "--repetition",
                    str(repeat),
                    "--mode",
                    "ncu",
                    "--timeout",
                    "1800",
                    "--target-args",
                    *cell.target(arm, 1, trace=True)[1:],
                ]
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"NVBit capture failed for {cell.key}/{arm}:\n"
                        f"{completed.stdout}\n{completed.stderr}"
                    )
                print(
                    f"nvbit {progress:02d}/{total:02d} {cell.key} {arm} r{repeat}",
                    flush=True,
                )


def run_replay(
    output: Path,
    cells: Sequence[Cell],
    repeats: int,
    repeat_start: int,
) -> None:
    sidecar = output / "static/store_ack_frontier_trace.control_bits.json"
    for path in (SIM_RUN, sidecar, ARCH_CARD, PCHASE_CARD, ACK_CANDIDATE):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"replay input is missing or empty: {path}")
    total = len(cells) * len(ARMS) * repeats
    progress = 0
    for repeat in range(repeat_start, repeat_start + repeats):
        for cell in cells:
            for arm in ARMS:
                progress += 1
                trace = output / "nvbit" / cell.key / f"repeat-{repeat}" / arm
                if not (trace / "complete.json").is_file():
                    continue
                destination = output / "simulator" / cell.key / f"repeat-{repeat}" / arm
                destination.mkdir(parents=True, exist_ok=True)
                base = [
                    str(SIM_RUN),
                    "--trace",
                    str(trace),
                    "--control",
                    str(sidecar),
                    "--arch-card",
                    str(ARCH_CARD),
                    "--pchase-card",
                    str(PCHASE_CARD),
                    "--mode",
                    "l1-l2",
                ]
                identity: dict[str, Any] = {
                    "schema_version": 1,
                    "trace": str(trace.relative_to(REPO_ROOT)),
                    "inputs": {
                        str(path.relative_to(REPO_ROOT)): sha256(path)
                        for path in (SIM_RUN, sidecar, ARCH_CARD, PCHASE_CARD, ACK_CANDIDATE)
                    },
                    "modes": {},
                }
                for name, suffix in (
                    ("default", []),
                    (
                        "write_ack",
                        ["--memory-admission-feedback", str(ACK_CANDIDATE)],
                    ),
                ):
                    outputs: list[bytes] = []
                    for twin in (1, 2):
                        completed = subprocess.run(
                            [*base, *suffix],
                            cwd=REPO_ROOT,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            timeout=600,
                            check=False,
                        )
                        (destination / f"{name}-{twin}.stderr.txt").write_bytes(
                            completed.stderr
                        )
                        if completed.returncode != 0:
                            raise RuntimeError(
                                f"replay failed for {cell.key}/{arm}/{name}: "
                                f"{completed.stderr.decode(errors='replace')}"
                            )
                        json.loads(completed.stdout)
                        (destination / f"{name}-{twin}.json").write_bytes(
                            completed.stdout
                        )
                        outputs.append(completed.stdout)
                    if outputs[0] != outputs[1]:
                        raise RuntimeError(
                            f"nondeterministic replay for {cell.key}/{arm}/{name}"
                        )
                    identity["modes"][name] = {
                        "command": [*base, *suffix],
                        "stdout_sha256": hashlib.sha256(outputs[0]).hexdigest(),
                        "deterministic_twins": True,
                    }
                write_json(destination / "identity.json", identity)
                print(
                    f"replay {progress:02d}/{total:02d} {cell.key} {arm} r{repeat}",
                    flush=True,
                )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "action",
        choices=(
            "keys",
            "build",
            "manifest",
            "static",
            "correctness",
            "native",
            "nvbit",
            "replay",
        ),
    )
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--cells")
    result.add_argument("--stage", choices=("identification", "calibration", "holdout"))
    result.add_argument("--repeats", type=int, default=3)
    result.add_argument("--repeat-start", type=int, default=1)
    result.add_argument("--trials", type=int, default=7)
    result.add_argument("--force", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = args.output_root.resolve()
    try:
        cells = selected_cells(args.cells, args.stage)
        if args.action == "keys":
            print(",".join(cell.key for cell in cells))
            return 0
        if args.trials < 1 or args.trials > 129 or args.trials % 2 == 0:
            raise RuntimeError("--trials must be odd and within 1..129")
        if args.repeats < 1 or args.repeats > 64:
            raise RuntimeError("--repeats must be within 1..64")
        if args.action == "build":
            build(output)
        elif args.action == "manifest":
            emit_manifest(output)
        elif args.action == "static":
            run_static(output)
        elif args.action == "correctness":
            emit_manifest(output)
            run_correctness(output, cells, args.trials)
        elif args.action == "native":
            emit_manifest(output)
            run_native(
                output,
                cells,
                args.repeats,
                args.trials,
                args.repeat_start,
                args.force,
            )
        elif args.action == "nvbit":
            emit_manifest(output)
            run_nvbit(output, cells, args.repeats, args.repeat_start)
        elif args.action == "replay":
            run_replay(output, cells, args.repeats, args.repeat_start)
    except (OSError, RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        print(f"store-ack campaign: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
