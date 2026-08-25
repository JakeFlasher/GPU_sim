#!/usr/bin/env python3
"""Run the sm_86 branch-free CLOCK/IBuffer timeline campaign."""

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
    / "results/captures/sm86_front_end_ibuffer_depth_20260818_v1"
)
DRIVER = HERE / "front_end_ibuffer_depth_driver"
IDENTIFICATION_VARIANTS = (
    "baseline_s1",
    *(f"stall4_s1_i{index}" for index in range(8)),
    *(f"yield_s1_i{index}" for index in range(8)),
)
TRANSFER_VARIANTS = ("baseline_s1", "stall4_s1_i1", "yield_s1_i1")
TRANSFER_WIDTHS = (1, 4, 5, 8, 12)
SOURCE_VARIANTS = (
    "baseline_s1",
    "stall4_s1_i0",
    "stall4_s1_i1",
    "stall4_s1_i4",
    "yield_s1_i0",
    "yield_s1_i1",
    "yield_s1_i4",
)
METRICS = (
    "gpc__cycles_elapsed.max",
    "sm__cycles_active.max",
    "smsp__cycles_active.sum",
    "smsp__warps_active.avg.per_cycle_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_issued.sum",
    "smsp__inst_executed.sum",
    "smsp__inst_issued_per_issue_active.ratio",
    "smsp__inst_executed_pipe_alu.sum",
    "smsp__inst_executed_pipe_fma.sum",
    "smsp__inst_executed_pipe_cbu.sum",
    "smsp__inst_executed_pipe_lsu.sum",
    "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio",
    "smsp__warps_issue_stalled_no_instruction",
    "smsp__warps_issue_stalled_branch_resolving",
    "smsp__sass_branch_targets_threads_uniform",
    "smsp__sass_branch_targets_threads_divergent",
)


def load_predecessor_campaign():
    path = HERE.parent / "fixed_latency_front_end_footprint/campaign.py"
    spec = importlib.util.spec_from_file_location("ibuffer_footprint_campaign", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PREVIOUS = load_predecessor_campaign()
ORDER = PREVIOUS.ORDER
BASE = PREVIOUS.BASE


def cubin(variant: str) -> Path:
    return HERE / f"front_end_ibuffer_depth_{variant}.cubin"


@dataclass(frozen=True, slots=True)
class Cell:
    stage: str
    warps: int
    variant: str
    base_stall: int
    treatment: str
    control_index: int | None

    @property
    def key(self) -> str:
        return f"w{self.warps}_{self.variant}"

    def target(self, *, trace: bool, trials: int) -> list[str]:
        del trace
        return [str(DRIVER), str(cubin(self.variant)), str(self.warps), str(trials), self.variant]


def cell(warps: int, variant: str, stage: str) -> Cell:
    if variant == "baseline_s1":
        treatment = "baseline"
        index = None
    else:
        treatment = "stall4" if variant.startswith("stall4") else "yield"
        index = int(variant.rsplit("i", 1)[1])
    return Cell(stage, warps, variant, 1, treatment, index)


def configured_cells() -> list[Cell]:
    cells = [cell(16, variant, "control_index_identification") for variant in IDENTIFICATION_VARIANTS]
    cells.extend(
        cell(warps, variant, "width_transfer")
        for warps in TRANSFER_WIDTHS
        for variant in TRANSFER_VARIANTS
    )
    if len(cells) != 32 or len({entry.key for entry in cells}) != 32:
        raise RuntimeError("IBuffer matrix must contain 32 unique cells")
    return cells


def selected_cells(selector: str | None) -> list[Cell]:
    cells = configured_cells()
    if selector is None:
        return cells
    requested = {item.strip() for item in selector.split(",") if item.strip()}
    selected = [entry for entry in cells if entry.key in requested]
    missing = requested - {entry.key for entry in selected}
    if missing:
        raise RuntimeError(f"unknown cell key(s): {', '.join(sorted(missing))}")
    return selected


def require_binaries() -> None:
    paths = [DRIVER, HERE / "front_end_ibuffer_depth_native.cubin"]
    paths.extend(cubin(variant) for variant in IDENTIFICATION_VARIANTS)
    paths.extend(
        cubin(f"{kind}_s0_i1")
        for kind in ("stall4", "yield")
    )
    paths.append(cubin("baseline_s0"))
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"required binary missing: {path}")


def emit_manifest(output: Path) -> None:
    require_binaries()
    BASE.write_json(
        output / "metadata/matrix.json",
        {
            "schema_version": 1,
            "cells": [{**asdict(entry), "key": entry.key} for entry in configured_cells()],
            "ncu_metrics": list(METRICS),
            "measured_instructions": 32,
            "clock_groups": 8,
            "instructions_per_clock_group": 4,
            "timestamps_per_warp": 9,
            "identification_width": 16,
            "transfer_widths": list(TRANSFER_WIDTHS),
            "pair_invariants": [
                "one branch-free sequence of eight CS2R-plus-three-NOP groups",
                "the ninth group is outside the first 32 measured instructions",
                "all legal variants differ only in one selected stall/yield control",
                "encoded stall one is the zero-bubble, nvdisasm-valid baseline",
                "every warp returns all nine CLOCK timestamps",
            ],
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
    for variant in ("baseline_s0", "stall4_s0_i1", "yield_s0_i1"):
        command = [str(DRIVER), str(cubin(variant)), "16", "7", variant]
        completed = subprocess.run(command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"{variant}: calibration execution failed")
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        if payload["correct"] is not True:
            raise RuntimeError(f"{variant}: calibration result incorrect")
        rows[variant] = {"command": command, "measurement": payload}
        (destination / f"{variant}.stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (destination / f"{variant}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    BASE.write_json(
        destination / "summary.json",
        {
            "schema_version": 1,
            "status": "literal_stall_zero_rejected_from_main_matrix",
            "reason": "nvdisasm rejects stall-zero NOP control and CLOCK spans are roughly eight times the legal stall-one baseline",
            "runs": rows,
        },
    )
    print(destination / "summary.json")


def run_ncu(output: Path, cells: Sequence[Cell], repeat_start: int, repeats: int, force: bool) -> None:
    ncu = os.environ.get("NCU", "ncu")
    total = len(cells) * repeats
    progress = 0
    for entry in cells:
        for repeat in range(repeat_start, repeat_start + repeats):
            progress += 1
            destination = output / "ncu" / entry.key / f"repeat-{repeat}"
            report_base = destination / "ncu-report"
            report = destination / "ncu-report.ncu-rep"
            if report.is_file() and not force:
                print(f"ncu {progress:03d}/{total:03d} {entry.key} r{repeat} (existing)", flush=True)
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
                    ",".join(METRICS),
                    *entry.target(trace=False, trials=1),
                ],
                destination,
                timeout=1200,
            )
            if not report.is_file():
                raise RuntimeError(f"NCU did not produce {report}")
            BASE.export_ncu(ncu, report, destination)
            print(f"ncu {progress:03d}/{total:03d} {entry.key} r{repeat}", flush=True)


def run_source(output: Path, cells: Sequence[Cell], force: bool) -> None:
    ncu = os.environ.get("NCU", "ncu")
    sentinels = [entry for entry in cells if entry.warps == 16 and entry.variant in SOURCE_VARIANTS]
    for index, entry in enumerate(sentinels, 1):
        destination = output / "ncu-source" / entry.key
        report_base = destination / "ncu-report"
        report = destination / "ncu-report.ncu-rep"
        if report.is_file() and not force:
            print(f"source {index:02d}/{len(sentinels):02d} {entry.key} (existing)", flush=True)
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
                *entry.target(trace=False, trials=1),
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
            raise RuntimeError(f"source export failed for {entry.key}")
        print(f"source {index:02d}/{len(sentinels):02d} {entry.key}", flush=True)


def run_static(output: Path) -> None:
    destination = output / "sass"
    destination.mkdir(parents=True, exist_ok=True)
    binaries = [HERE / "front_end_ibuffer_depth_native.cubin"]
    binaries.extend(cubin(variant) for variant in IDENTIFICATION_VARIANTS)
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
    BASE.write_json(
        output / "metadata/front_end_ibuffer_depth.patch.json",
        json.loads((HERE / "front_end_ibuffer_depth.patch.json").read_text(encoding="utf-8")),
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
    output_sass = destination.resolve()
    predecessors = sorted(
        path
        for path in REPO_ROOT.glob("results/captures/**/*.control_bits.json")
        if path.parent.resolve() != output_sass
    )
    if len(predecessors) != 135:
        raise RuntimeError(f"expected 135 predecessor sidecars, got {len(predecessors)}")
    audit(
        predecessors + sidecars,
        "retained_corpus_operand_projection_audit.json",
        "retained-corpus-operand-projection-run",
    )
    print(destination)


def run_simulator(output: Path, cells: Sequence[Cell], force: bool) -> None:
    binary = REPO_ROOT / "sim/target/release/sim-run"
    for index, entry in enumerate(cells, 1):
        destination = output / "simulator" / entry.key
        destination.mkdir(parents=True, exist_ok=True)
        command = [
            str(binary),
            "--trace",
            str(output / "nvbit" / entry.key),
            "--control",
            str(output / "sass" / f"front_end_ibuffer_depth_{entry.variant}.control_bits.json"),
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
            completed = subprocess.run(command, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            (destination / f"replay-{repeat}.stderr.txt").write_bytes(completed.stderr)
            if completed.returncode != 0:
                raise RuntimeError(f"{entry.key}: replay {repeat} failed")
            path.write_bytes(completed.stdout)
            outputs.append(completed.stdout)
        if outputs[0] != outputs[1]:
            raise RuntimeError(f"{entry.key}: nondeterministic replay")
        print(f"simulator {index:02d}/{len(cells):02d} {entry.key}", flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("action", choices=("manifest", "calibration", "correctness", "ncu", "source", "nvbit", "static", "simulator"))
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
            run_ncu(output, cells, args.repeat_start, args.repeats, args.force)
        elif args.action == "source":
            run_source(output, cells, args.force)
        elif args.action == "nvbit":
            ORDER.run_nvbit(output, cells, args.force)
        elif args.action == "static":
            run_static(output)
        elif args.action == "simulator":
            run_simulator(output, cells, args.force)
    except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(f"IBuffer timeline campaign: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
