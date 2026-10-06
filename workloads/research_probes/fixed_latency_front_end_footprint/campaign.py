#!/usr/bin/env python3
"""Run the sm_86 branch-frequency/static-footprint factorial."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results"
    / "captures"
    / "sm86_fixed_latency_front_end_footprint_20260817_v1"
)
DRIVER = HERE / "fixed_latency_front_end_footprint_driver"
UNROLLS = (1, 2, 4, 8)
WIDTHS = (4, 5, 8, 9, 13, 16)
VARIANTS = ("fixed_bypass", "fixed_retired", "global_frontier", "global_late")
SOURCE_WIDTHS = {5, 16}
FRONT_END_METRICS = (
    "smsp__average_warp_latency_issue_stalled_no_instruction",
    "smsp__average_warp_latency_issue_stalled_branch_resolving",
    "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio",
    "smsp__warps_issue_stalled_no_instruction",
    "smsp__warps_issue_stalled_branch_resolving",
    "smsp__warp_issue_stalled_no_instruction_per_warp_active",
    "smsp__warp_issue_stalled_branch_resolving_per_warp_active",
    "smsp__sass_branch_targets_threads_uniform",
    "smsp__sass_branch_targets_threads_divergent",
)


def load_predecessor_campaign():
    path = HERE.parent / "fixed_latency_producer_shift" / "campaign.py"
    spec = importlib.util.spec_from_file_location("footprint_shift_campaign", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SHIFT = load_predecessor_campaign()
ORDER = SHIFT.ORDER
PRIORITY = SHIFT.PRIORITY
BASE = SHIFT.BASE
METRICS = SHIFT.METRICS


def cubin(unroll: int, variant: str, trace: bool) -> Path:
    # One long-generation cubin is used by correctness, NVBit, replay, and
    # NCU. The historical short trace pair is no longer an active workload.
    return HERE / f"fixed_latency_front_end_footprint_u{unroll}_{variant}.cubin"


def native_cubin(unroll: int, trace: bool) -> Path:
    return HERE / f"fixed_latency_front_end_footprint_u{unroll}_native.cubin"


def stage(warps: int) -> str:
    if warps == 4:
        return "one_warp_per_smsp_baseline"
    if warps == 5:
        return "second_warp_smsp0_boundary"
    if warps == 8:
        return "two_warps_per_smsp"
    if warps == 9:
        return "third_warp_smsp0_boundary"
    if warps == 13:
        return "fourth_warp_smsp0_boundary"
    return "four_warps_per_smsp"


@dataclass(frozen=True, slots=True)
class Cell:
    stage: str
    unroll: int
    warps: int
    variant: str
    kind: str
    distance: int
    patched_bodies_per_group: int
    frontier_calibrated: bool

    @property
    def key(self) -> str:
        return f"u{self.unroll}_w{self.warps}_{self.variant}"

    def target(self, *, trace: bool, trials: int) -> list[str]:
        return [
            str(DRIVER),
            str(cubin(self.unroll, self.variant, trace)),
            self.kind,
            str(self.warps),
            "1024",
            str(trials),
            self.variant,
            str(self.unroll),
        ]


def configured_cells() -> list[Cell]:
    metadata = {
        "fixed_bypass": ("fixed", 4, True),
        "fixed_retired": ("fixed", 5, True),
        "global_frontier": ("global", 5, False),
        "global_late": ("global", 6, False),
    }
    cells = []
    for unroll in UNROLLS:
        for warps in WIDTHS:
            for variant in VARIANTS:
                kind, distance, calibrated = metadata[variant]
                cells.append(
                    Cell(
                        stage=stage(warps),
                        unroll=unroll,
                        warps=warps,
                        variant=variant,
                        kind=kind,
                        distance=distance,
                        patched_bodies_per_group=(
                            unroll if kind == "fixed" or unroll <= 2 else unroll - 1
                        ),
                        frontier_calibrated=calibrated,
                    )
                )
    if len(cells) != 96 or len({cell.key for cell in cells}) != 96:
        raise RuntimeError("front-end footprint matrix must contain 96 unique cells")
    return cells


def selected_cells(selector: str | None) -> list[Cell]:
    cells = configured_cells()
    if selector is None:
        return cells
    requested = {item.strip() for item in selector.split(",") if item.strip()}
    selected = [cell for cell in cells if cell.key in requested]
    missing = requested - {cell.key for cell in selected}
    if missing:
        raise RuntimeError(f"unknown cell key(s): {', '.join(sorted(missing))}")
    return selected


def require_binaries() -> None:
    paths = [DRIVER]
    paths.extend(native_cubin(unroll, False) for unroll in UNROLLS)
    paths.extend(
        cubin(unroll, variant, False)
        for unroll in UNROLLS
        for variant in VARIANTS
    )
    paths.extend(
        HERE / f"fixed_latency_front_end_footprint_u1_calibration_{variant}.cubin"
        for variant in ("fixed_invalid", "global_invalid")
    )
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"required binary is missing: {path}")


def emit_manifest(output: Path) -> None:
    require_binaries()
    BASE.write_json(
        output / "metadata/matrix.json",
        {
            "schema_version": 1,
            "cells": [{**asdict(cell), "key": cell.key} for cell in configured_cells()],
            "ncu_metrics": list(METRICS),
            "front_end_ncu_metrics": list(FRONT_END_METRICS),
            "unroll_factors": list(UNROLLS),
            "widths": list(WIDTHS),
            "long_generations": 1024,
            "trace_generations": 1024,
            "source_counter_widths": sorted(SOURCE_WIDTHS),
            "pair_invariants": [
                "1024 dynamic bodies and equal instruction work within each U/pair",
                "one backward branch per U bodies; 1024/U branches per warp",
                "fixed d4/d5 is calibrated by deterministic U1 d3 checksum failure",
                "fixed pair changes only stall bits and has equal selected stall sum 10 per body",
                "global d5/d6 is an equal-stall holdout, not a calibrated frontier in this register allocation",
                "U4/U8 global body zero is neutral because a low-address MOV prevents an uncontaminated d5/d6 patch",
                "nonconvergent fixed recurrence detects stale intermediate results",
                "all long and trace generation counts are divisible by U1/U2/U4/U8",
            ],
            "interpretation": (
                "fixed pair is the causal branch-frequency/static-footprint treatment; "
                "global pair tests transfer only"
            ),
        },
    )
    environment: dict[str, object] = {"captured_at": BASE.now()}
    for name, command in {
        "nvidia_smi": [
            "nvidia-smi",
            "--query-gpu=name,uuid,compute_cap,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        "nvcc": ["/usr/local/cuda-13.2/bin/nvcc", "--version"],
        "ncu": [os.environ.get("NCU", "ncu"), "--version"],
    }.items():
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        environment[name] = {
            "command": command,
            "return_code": completed.returncode,
            "output": completed.stdout.strip(),
        }
    BASE.write_json(output / "metadata/environment.json", environment)


def run_calibration(output: Path) -> None:
    destination = output / "calibration"
    destination.mkdir(parents=True, exist_ok=True)
    rows = {}
    for variant, kind in (("fixed_invalid", "fixed"), ("global_invalid", "global")):
        command = [
            str(DRIVER),
            str(HERE / f"fixed_latency_front_end_footprint_u1_calibration_{variant}.cubin"),
            kind,
            "1",
            "1024",
            "7",
            variant,
            "1",
        ]
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        (destination / f"{variant}.stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (destination / f"{variant}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        rows[variant] = {
            "command": command,
            "return_code": completed.returncode,
            "measurement": payload,
            "stderr": completed.stderr.strip(),
        }
    if rows["fixed_invalid"]["return_code"] != 1 or rows["fixed_invalid"]["measurement"]["correct"] is not False:
        raise RuntimeError("fixed d3 calibration did not fail deterministically")
    if rows["global_invalid"]["return_code"] != 0 or rows["global_invalid"]["measurement"]["correct"] is not True:
        raise RuntimeError("global d4 holdout characterization changed")
    BASE.write_json(
        destination / "summary.json",
        {
            "schema_version": 1,
            "fixed_d3_semantic_boundary": "invalid",
            "global_d4_semantic_boundary": "not_identifying_in_this_register_allocation",
            "runs": rows,
        },
    )
    print(destination / "summary.json")


def run_source(output: Path, cells: Sequence[Cell], force: bool) -> None:
    ncu = os.environ.get("NCU", "ncu")
    sentinels = [cell for cell in cells if cell.warps in SOURCE_WIDTHS]
    for index, cell in enumerate(sentinels, 1):
        destination = output / "ncu-source" / cell.key
        report_base = destination / "ncu-report"
        report = destination / "ncu-report.ncu-rep"
        if report.is_file() and not force:
            print(f"source {index:02d}/{len(sentinels):02d} {cell.key} (existing)", flush=True)
            continue
        BASE.run_captured(
            [
                ncu,
                "--force-overwrite",
                "--export",
                str(report_base),
                "--profile-from-start",
                "on",
                "--launch-count",
                "1",
                "--replay-mode",
                "application",
                "--clock-control",
                "base",
                "--cache-control",
                "none",
                "--section",
                "SourceCounters",
                *cell.target(trace=False, trials=1),
            ],
            destination,
            timeout=1200,
        )
        if not report.is_file():
            raise RuntimeError(f"NCU did not produce {report}")
        BASE.export_ncu(ncu, report, destination)
        completed = subprocess.run(
            [
                ncu,
                "--import",
                str(report),
                "--csv",
                "--page",
                "source",
                "--print-source",
                "sass",
                "--log-file",
                str(destination / "source.csv"),
            ],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        (destination / "source-export.stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (destination / "source-export.stderr.txt").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0 or not (destination / "source.csv").is_file():
            raise RuntimeError(f"source export failed for {cell.key}")
        print(f"source {index:02d}/{len(sentinels):02d} {cell.key}", flush=True)


def run_front_end_metrics(output: Path, cells: Sequence[Cell], force: bool) -> None:
    ncu = os.environ.get("NCU", "ncu")
    for index, cell in enumerate(cells, 1):
        destination = output / "ncu-front-end" / cell.key
        report_base = destination / "ncu-report"
        report = destination / "ncu-report.ncu-rep"
        if report.is_file() and not force:
            print(f"front-end {index:02d}/{len(cells):02d} {cell.key} (existing)", flush=True)
            continue
        BASE.run_captured(
            [
                ncu,
                "--force-overwrite",
                "--export",
                str(report_base),
                "--profile-from-start",
                "on",
                "--launch-count",
                "1",
                "--replay-mode",
                "application",
                "--clock-control",
                "base",
                "--cache-control",
                "none",
                "--metrics",
                ",".join(FRONT_END_METRICS),
                *cell.target(trace=False, trials=1),
            ],
            destination,
            timeout=1200,
        )
        if not report.is_file():
            raise RuntimeError(f"NCU did not produce {report}")
        BASE.export_ncu(ncu, report, destination)
        print(f"front-end {index:02d}/{len(cells):02d} {cell.key}", flush=True)


def run_static(output: Path) -> None:
    destination = output / "sass"
    destination.mkdir(parents=True, exist_ok=True)
    binaries = []
    for unroll in UNROLLS:
        binaries.append(native_cubin(unroll, False))
        binaries.extend(cubin(unroll, variant, False) for variant in VARIANTS)
    sidecars = []
    for binary in binaries:
        name = binary.stem
        for suffix, option in (("sass", "--dump-sass"), ("resources", "--dump-resource-usage")):
            completed = subprocess.run(
                ["/usr/local/cuda-13.2/bin/cuobjdump", option, str(binary)],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            (destination / f"{name}.{suffix}.txt").write_text(completed.stdout, encoding="utf-8")
            (destination / f"{name}.{suffix}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(f"cuobjdump failed for {binary}")
        sidecar = destination / f"{name}.control_bits.json"
        BASE.run_captured(
            [
                str(REPO_ROOT / ".venv/bin/python"),
                str(REPO_ROOT / "tools/sass_control/emit_control_bits_sidecar.py"),
                "--binary",
                str(binary),
                "--arch",
                "sm_86",
                "--out",
                str(sidecar),
                "--toolkit-bindir",
                "/usr/local/cuda-13.2/bin",
            ],
            destination / f"{name}.control-extraction",
        )
        sidecars.append(sidecar)
    for unroll in UNROLLS:
        source = HERE / f"fixed_latency_front_end_footprint_u{unroll}.patch.json"
        BASE.write_json(
            output / "metadata" / source.name,
            json.loads(source.read_text(encoding="utf-8")),
        )
    BASE.write_json(
        output / "metadata/fixed_latency_front_end_footprint_u1_calibration.patch.json",
        json.loads((HERE / "fixed_latency_front_end_footprint_u1_calibration.patch.json").read_text(encoding="utf-8")),
    )

    def audit(paths: list[Path], out_name: str, run_name: str) -> None:
        BASE.run_captured(
            [
                "cargo",
                "run",
                "--quiet",
                "--release",
                "--manifest-path",
                str(REPO_ROOT / "sim/Cargo.toml"),
                "-p",
                "sim-sm",
                "--example",
                "operand_projection_audit",
                "--",
                "--out",
                str(output / "metadata" / out_name),
                *map(str, paths),
            ],
            output / "metadata" / run_name,
            timeout=900,
        )

    audit(sidecars, "operand_projection_audit.json", "operand-projection-run")
    # Current source/sidecar inputs are sufficient for this bounded capture.
    # Historical predecessor populations are not execution prerequisites.
    print(destination)


def run_simulator(output: Path, cells: Sequence[Cell], force: bool) -> None:
    binary = REPO_ROOT / "sim/target/release/sim-run"
    if not binary.is_file():
        raise RuntimeError(f"simulator missing: {binary}")
    for index, cell in enumerate(cells, 1):
        destination = output / "simulator" / cell.key
        destination.mkdir(parents=True, exist_ok=True)
        control = output / "sass" / f"{cubin(cell.unroll, cell.variant, False).stem}.control_bits.json"
        command = [
            str(binary),
            "--trace",
            str(output / "nvbit" / cell.key),
            "--control",
            str(control),
            "--arch-card",
            str(REPO_ROOT / "arch_profiles/cards/sm_86.json"),
            "--pchase-card",
            str(REPO_ROOT / "data/pchase/sm_86_static_card.json"),
            "--mode",
            "l1",
        ]
        outputs = []
        for repeat in (1, 2):
            path = destination / f"replay-{repeat}.json"
            if path.is_file() and not force:
                outputs.append(path.read_bytes())
                continue
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            (destination / f"replay-{repeat}.stderr.txt").write_bytes(completed.stderr)
            if completed.returncode != 0:
                raise RuntimeError(f"{cell.key}: replay {repeat} failed")
            path.write_bytes(completed.stdout)
            outputs.append(completed.stdout)
        if outputs[0] != outputs[1]:
            raise RuntimeError(f"{cell.key}: nondeterministic replay")
        print(f"simulator {index:02d}/{len(cells):02d} {cell.key}", flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "action",
        choices=("manifest", "calibration", "correctness", "ncu", "front-end", "source", "nvbit", "static", "simulator"),
    )
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--cells")
    result.add_argument("--trials", type=int, default=7)
    result.add_argument("--repeat-start", type=int, default=1)
    result.add_argument("--repeats", type=int, default=3)
    result.add_argument("--force", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = args.output_root.resolve()
    try:
        cells = selected_cells(args.cells)
        emit_manifest(output)
        if args.action == "manifest":
            print(output / "metadata/matrix.json")
        elif args.action == "calibration":
            run_calibration(output)
        elif args.action == "correctness":
            ORDER.run_correctness(output, cells, args.trials)
        elif args.action == "ncu":
            PRIORITY.run_ncu(output, cells, args.repeat_start, args.repeats, args.force)
        elif args.action == "front-end":
            run_front_end_metrics(output, cells, args.force)
        elif args.action == "source":
            run_source(output, cells, args.force)
        elif args.action == "nvbit":
            ORDER.run_nvbit(output, cells, args.force)
        elif args.action == "static":
            run_static(output)
        elif args.action == "simulator":
            run_simulator(output, cells, args.force)
    except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(f"front-end footprint campaign: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
