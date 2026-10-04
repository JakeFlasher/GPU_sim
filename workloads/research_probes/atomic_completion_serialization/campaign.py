#!/usr/bin/env python3
"""Run the sm_86 returning-atomic completion and serialization campaign."""

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
    / "results/captures/sm86_atomic_completion_serialization_20260818_v1"
)
DRIVER = HERE / "atomic_completion_serialization_driver"
FUNCTION = "_Z12atomic_probePyS_Pj"
WIDTHS = (1, 4, 8, 16)
VARIANTS = (
    "ret_lane_line",
    "ret_lane_word",
    "ret_warp_word",
    "ret_global_word",
    "red_lane_line",
    "red_lane_word",
    "red_warp_word",
    "red_global_word",
)
VARIANT_METADATA = {
    "ret_lane_line": ("ATOMG", "lane_line", 0, True),
    "ret_lane_word": ("ATOMG", "lane_word", 1, True),
    "ret_warp_word": ("ATOMG", "warp_word", 2, True),
    "ret_global_word": ("ATOMG", "global_word", 3, True),
    "red_lane_line": ("RED", "lane_line", 0, False),
    "red_lane_word": ("RED", "lane_word", 1, False),
    "red_warp_word": ("RED", "warp_word", 2, False),
    "red_global_word": ("RED", "global_word", 3, False),
}
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
    "smsp__inst_executed_pipe_lsu.sum",
    "smsp__inst_executed_op_generic_atom_dot_alu.sum",
    "smsp__inst_executed_op_global_red.sum",
    "sm__sass_inst_executed_op_global_atom.sum",
    "sm__sass_inst_executed_op_global_red.sum",
    "smsp__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_atom.sum",
    "smsp__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_red.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum",
    "l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_atom.sum",
    "l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_red.sum",
    "lts__t_requests_srcunit_tex_op_atom_dot_alu.sum",
    "lts__t_requests_srcunit_tex_op_atom_dot_alu_lookup_hit.sum",
    "lts__t_requests_srcunit_tex_op_red.sum",
    "lts__t_requests_srcunit_tex_op_red_lookup_hit.sum",
    "dram__sectors_read.sum",
    "dram__sectors_write.sum",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
)


def load_predecessor_campaign():
    path = HERE.parent / "fixed_latency_result_order/campaign.py"
    spec = importlib.util.spec_from_file_location("atomic_result_order_campaign", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ORDER = load_predecessor_campaign()
BASE = ORDER.BASE


def cubin(variant: str, *, trace: bool) -> Path:
    # One 1024-generation cubin per variant is used for correctness, NVBit,
    # replay, and NCU. The former 64-generation trace twins were not the same
    # workload and made the campaign's identity contract false.
    return HERE / f"atomic_completion_serialization_{variant}.cubin"


@dataclass(frozen=True, slots=True)
class Cell:
    stage: str
    warps: int
    variant: str
    operation: str
    layout: str
    layout_index: int
    returning: bool

    @property
    def key(self) -> str:
        return f"w{self.warps}_{self.variant}"

    def target(self, *, trace: bool, trials: int) -> list[str]:
        return [
            str(DRIVER),
            str(cubin(self.variant, trace=trace)),
            str(self.warps),
            str(1024),
            str(trials),
            self.variant,
            str(self.layout_index),
            "1" if self.returning else "0",
        ]


def configured_cells() -> list[Cell]:
    cells = []
    for warps in WIDTHS:
        for variant in VARIANTS:
            operation, layout, layout_index, returning = VARIANT_METADATA[variant]
            cells.append(
                Cell(
                    stage=(
                        "single_warp_latency"
                        if warps == 1
                        else "one_warp_per_smsp"
                        if warps == 4
                        else "latency_hiding_transfer"
                    ),
                    warps=warps,
                    variant=variant,
                    operation=operation,
                    layout=layout,
                    layout_index=layout_index,
                    returning=returning,
                )
            )
    if len(cells) != 32 or len({entry.key for entry in cells}) != 32:
        raise RuntimeError("atomic matrix must contain 32 unique cells")
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
    paths = [DRIVER]
    paths.extend(
        cubin(variant, trace=trace)
        for variant in VARIANTS
        for trace in (False,)
    )
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
            "long_generations": 1024,
            "generations": 1024,
            "widths": list(WIDTHS),
            "function": FUNCTION,
            "pair_invariants": [
                "one CTA and exactly the configured resident warp count",
                "one warp-level ATOMG or RED instruction per generation",
                "all active lanes perform one 32-bit atomic add per generation",
                "final-memory sum equals threads times generations",
                "returning variants also validate the order-independent sum of all old values",
                "lane-line, lane-word, and warp-word shapes are static-primary comparisons",
                "the compiler-expanded returning global-word case is an extreme-contention transfer case",
            ],
            "interpretation": (
                "ATOMG versus RED identifies observed-return completion; layout and width "
                "sweeps identify request formation, latency hiding, and contention without "
                "assuming that identical-address traffic must be the slowest case"
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


def run_trace_correctness(
    output: Path, cells: Sequence[Cell], trials: int
) -> None:
    for index, entry in enumerate(cells, 1):
        destination = output / "trace-correctness" / entry.key
        completed = BASE.run_captured(
            entry.target(trace=True, trials=trials), destination
        )
        payload = BASE.measurement_json_line(completed.stdout, entry.key)
        BASE.write_json(
            destination / "result.json",
            {"cell": {**asdict(entry), "key": entry.key}, "measurement": payload},
        )
        print(
            f"trace-correctness {index:02d}/{len(cells):02d} {entry.key}",
            flush=True,
        )


def run_ncu(
    output: Path,
    cells: Sequence[Cell],
    repeat_start: int,
    repeats: int,
    force: bool,
) -> None:
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
                print(
                    f"ncu {progress:03d}/{total:03d} {entry.key} r{repeat} (existing)",
                    flush=True,
                )
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
    sentinels = [entry for entry in cells if entry.warps == 16]
    for index, entry in enumerate(sentinels, 1):
        destination = output / "ncu-source" / entry.key
        report_base = destination / "ncu-report"
        report = destination / "ncu-report.ncu-rep"
        if report.is_file() and not force:
            print(
                f"source {index:02d}/{len(sentinels):02d} {entry.key} (existing)",
                flush=True,
            )
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
        (destination / "source-export.stdout.txt").write_text(
            completed.stdout, encoding="utf-8"
        )
        (destination / "source-export.stderr.txt").write_text(
            completed.stderr, encoding="utf-8"
        )
        if completed.returncode != 0 or not (destination / "source.csv").is_file():
            raise RuntimeError(f"source export failed for {entry.key}")
        print(f"source {index:02d}/{len(sentinels):02d} {entry.key}", flush=True)


def run_static(output: Path) -> None:
    destination = output / "sass"
    destination.mkdir(parents=True, exist_ok=True)
    binaries = [cubin(variant, trace=False) for variant in VARIANTS]
    sidecars = []
    for binary in binaries:
        name = binary.stem
        for suffix, option in (
            ("sass", "--dump-sass"),
            ("resources", "--dump-resource-usage"),
        ):
            completed = subprocess.run(
                ["/usr/local/cuda-13.2/bin/cuobjdump", option, str(binary)],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            (destination / f"{name}.{suffix}.txt").write_text(
                completed.stdout, encoding="utf-8"
            )
            (destination / f"{name}.{suffix}.stderr.txt").write_text(
                completed.stderr, encoding="utf-8"
            )
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

    def audit(paths: list[Path], output_name: str, run_name: str) -> None:
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
                str(output / "metadata" / output_name),
                *map(str, paths),
            ],
            output / "metadata" / run_name,
            timeout=900,
        )

    audit(sidecars, "operand_projection_audit.json", "operand-projection-run")
    print(destination)


def run_simulator(output: Path, cells: Sequence[Cell], force: bool) -> None:
    binary = REPO_ROOT / "sim/target/release/sim-run"
    if not binary.is_file():
        raise RuntimeError(f"simulator missing: {binary}")
    for index, entry in enumerate(cells, 1):
        destination = output / "simulator" / entry.key
        destination.mkdir(parents=True, exist_ok=True)
        command = [
            str(binary),
            "--trace",
            str(output / "nvbit" / entry.key),
            "--control",
            str(
                output
                / "sass"
                / f"atomic_completion_serialization_{entry.variant}.control_bits.json"
            ),
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
            (destination / f"replay-{repeat}.stderr.txt").write_bytes(
                completed.stderr
            )
            if completed.returncode != 0:
                raise RuntimeError(f"{entry.key}: replay {repeat} failed")
            path.write_bytes(completed.stdout)
            outputs.append(completed.stdout)
        if outputs[0] != outputs[1]:
            raise RuntimeError(f"{entry.key}: nondeterministic replay")
        launch = json.loads(outputs[0])["launches"][0]
        if launch["coverage"]["kind"] != "full":
            raise RuntimeError(f"{entry.key}: incomplete replay")
        print(f"simulator {index:02d}/{len(cells):02d} {entry.key}", flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "action",
        choices=(
            "manifest",
            "correctness",
            "trace-correctness",
            "ncu",
            "source",
            "nvbit",
            "static",
            "simulator",
        ),
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
        elif args.action == "correctness":
            ORDER.run_correctness(output, cells, args.trials)
        elif args.action == "trace-correctness":
            run_trace_correctness(output, cells, args.trials)
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
        print(f"atomic campaign: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
