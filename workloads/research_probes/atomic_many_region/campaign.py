#!/usr/bin/env python3
"""Run the sm_86 many-region atomic/LTS scheduling discriminator."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
PREDECESSOR = HERE.parent / "atomic_multism_service"
DEFAULT_OUTPUT = (
    REPO_ROOT / "results/captures/sm86_atomic_many_region_20260820_v1"
)
DRIVER = PREDECESSOR / "atomic_multism_service_driver"
LONG_CUBIN = PREDECESSOR / "atomic_multism_service_red32.cubin"
TRACE_CUBIN = PREDECESSOR / "atomic_multism_service_trace_red32.cubin"
FIXED_DATA_BYTES = 32 * 1024 * 1024
REGION_UNIVERSE = tuple(range(12))
ANCHOR_MAX_REGION = 128
FULL_METRICS = (
    "gpc__cycles_elapsed.max",
    "sm__cycles_active.max",
    "smsp__cycles_active.sum",
    "smsp__inst_issued.sum",
    "smsp__inst_executed.sum",
    "smsp__inst_executed_op_global_red.sum",
    "sm__sass_inst_executed_op_global_red.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum",
    "l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_red.sum",
    "lts__t_requests_srcunit_tex_op_red.sum",
    "lts__t_sectors_op_red.sum",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
)
TRACE_TRIPLETS = (
    (0, 1, 4),
    (2, 3, 5),
    (4, 5, 8),
    (6, 7, 9),
    (8, 9, 0),
    (10, 11, 1),
    (0, 2, 4),
    (1, 3, 4),
    (4, 6, 0),
    (5, 7, 0),
    (8, 10, 0),
    (9, 11, 0),
    (0, 57, 4),
    (0, 59, 4),
)
SOURCE_TRIPLETS = ((0, 1, 4), (0, 2, 4), (0, 57, 4), (0, 59, 4))


def load_predecessor():
    path = PREDECESSOR / "campaign.py"
    spec = importlib.util.spec_from_file_location("many_region_predecessor", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PREVIOUS = load_predecessor()
BASE = PREVIOUS.BASE
ORDER = PREVIOUS.ORDER


def region_token(regions: Sequence[int]) -> str:
    return "_".join(f"{region:03d}" for region in regions)


@dataclass(frozen=True, slots=True)
class TripletCell:
    regions: tuple[int, int, int]

    @property
    def key(self) -> str:
        return f"t_{region_token(self.regions)}"

    @property
    def variant(self) -> str:
        return "red32"

    def target(self, *, trace: bool, trials: int) -> list[str]:
        return [
            str(DRIVER),
            str(TRACE_CUBIN if trace else LONG_CUBIN),
            "84",
            "64" if trace else "1024",
            str(trials),
            self.key,
            "0",
            "0",
            "0",
            "3",
            "0",
            str(FIXED_DATA_BYTES),
            ",".join(map(str, self.regions)),
        ]


@dataclass(frozen=True, slots=True)
class PairCell:
    regions: tuple[int, int]

    @property
    def key(self) -> str:
        return f"p_{region_token(self.regions)}"

    def target(self, *, trials: int = 1) -> list[str]:
        return [
            str(DRIVER),
            str(LONG_CUBIN),
            "84",
            "1024",
            str(trials),
            self.key,
            "0",
            "0",
            "0",
            "2",
            "0",
            str(FIXED_DATA_BYTES),
            ",".join(map(str, self.regions)),
        ]


def triplet_cells() -> list[TripletCell]:
    result = [TripletCell(tuple(regions)) for regions in itertools.combinations(REGION_UNIVERSE, 3)]
    if len(result) != 220 or len({cell.key for cell in result}) != 220:
        raise RuntimeError("many-region matrix must contain 220 unique triplets")
    return result


def pair_cells() -> list[PairCell]:
    pairs = set(itertools.combinations(REGION_UNIVERSE, 2))
    pairs.update((0, region) for region in range(1, ANCHOR_MAX_REGION + 1))
    result = [PairCell(pair) for pair in sorted(pairs)]
    if len(result) != 183 or len({cell.key for cell in result}) != 183:
        raise RuntimeError("many-region pair matrix must contain 183 unique pairs")
    return result


def trace_cells() -> list[TripletCell]:
    return [TripletCell(tuple(regions)) for regions in TRACE_TRIPLETS]


def selected_triplets(selector: str | None) -> list[TripletCell]:
    configured = triplet_cells() if selector is None else trace_cells()
    if selector is None:
        return configured
    requested = {token.strip() for token in selector.split(",") if token.strip()}
    by_key = {cell.key: cell for cell in triplet_cells() + trace_cells()}
    missing = requested - set(by_key)
    if missing:
        raise RuntimeError(f"unknown triplet cell(s): {', '.join(sorted(missing))}")
    return [by_key[key] for key in sorted(requested)]


def require_inputs() -> None:
    for path in (DRIVER, LONG_CUBIN, TRACE_CUBIN):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"required binary is missing: {path}")


def emit_manifest(output: Path) -> None:
    require_inputs()
    BASE.write_json(
        output / "metadata/matrix.json",
        {
            "schema_version": 1,
            "triplets": [
                {**asdict(cell), "key": cell.key} for cell in triplet_cells()
            ],
            "pairs": [{**asdict(cell), "key": cell.key} for cell in pair_cells()],
            "trace_triplets": [cell.key for cell in trace_cells()],
            "source_triplets": [TripletCell(tuple(regions)).key for regions in SOURCE_TRIPLETS],
            "blocks": 84,
            "threads_per_block": 32,
            "long_generations": 1024,
            "trace_generations": 64,
            "fixed_data_bytes": FIXED_DATA_BYTES,
            "region_bytes": 256,
            "operation": "RED.E.ADD.STRONG.GPU",
            "pair_metric": "lts__t_sectors_op_red.sum",
            "full_metrics": list(FULL_METRICS),
            "full_ncu_repeats": {"slice_distinct": 3, "slice_colliding": 7},
            "invariants": [
                "one warp per CTA and one RED instruction per generation",
                "all lanes in one warp target one runtime-selected word",
                "every process allocates the same 32 MiB backing buffer",
                "triplet cells split 84 CTAs evenly over three distinct regions",
                "pair cells split 84 CTAs evenly over two distinct regions",
                "SASS and register allocation are identical across address sets",
            ],
        },
    )


def run_correctness(
    output: Path,
    cells: Sequence[TripletCell],
    trials: int,
    *,
    trace: bool,
) -> None:
    directory = "trace-correctness" if trace else "correctness"
    for index, cell in enumerate(cells, 1):
        destination = output / directory / cell.key
        completed = BASE.run_captured(
            cell.target(trace=trace, trials=trials), destination, timeout=180
        )
        payload = BASE.measurement_json_line(completed.stdout, cell.key)
        if (
            payload.get("correct") is not True
            or payload.get("fixed_data_bytes") != FIXED_DATA_BYTES
            or tuple(payload.get("region_ids", ())) != cell.regions
        ):
            raise RuntimeError(f"{cell.key}: incorrect fixed-allocation execution")
        BASE.write_json(
            destination / "result.json",
            {"cell": {**asdict(cell), "key": cell.key}, "measurement": payload},
        )
        print(f"{directory} {index:03d}/{len(cells):03d} {cell.key}", flush=True)


def run_pair_ncu(output: Path, cells: Sequence[PairCell], force: bool) -> None:
    ncu = os.environ.get("NCU", "ncu")
    for index, cell in enumerate(cells, 1):
        destination = output / "ncu-pairs" / cell.key
        report_base = destination / "ncu-report"
        report = destination / "ncu-report.ncu-rep"
        csv_path = destination / "ncu.csv"
        if report.is_file() and csv_path.is_file() and not force:
            print(f"ncu-pair {index:03d}/{len(cells):03d} {cell.key} (existing)")
            continue
        completed = BASE.run_captured(
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
                "lts__t_sectors_op_red.sum",
                *cell.target(),
            ],
            destination,
            timeout=1200,
        )
        if not report.is_file():
            raise RuntimeError(f"NCU did not produce {report}")
        BASE.export_ncu(ncu, report, destination)
        payload = BASE.measurement_json_line(completed.stdout, cell.key)
        BASE.write_json(
            destination / "measurement.json",
            {"cell": {**asdict(cell), "key": cell.key}, "measurement": payload},
        )
        print(f"ncu-pair {index:03d}/{len(cells):03d} {cell.key}", flush=True)


def run_full_ncu(
    output: Path,
    cells: Sequence[TripletCell],
    repeat_start: int,
    repeats: int,
    force: bool,
) -> None:
    ncu = os.environ.get("NCU", "ncu")
    total = len(cells) * repeats
    progress = 0
    for cell in cells:
        for repeat in range(repeat_start, repeat_start + repeats):
            progress += 1
            destination = output / "ncu" / cell.key / f"repeat-{repeat}"
            report_base = destination / "ncu-report"
            report = destination / "ncu-report.ncu-rep"
            if report.is_file() and (destination / "ncu.csv").is_file() and not force:
                print(f"ncu {progress:03d}/{total:03d} {cell.key} r{repeat} (existing)")
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
                    ",".join(FULL_METRICS),
                    *cell.target(trace=False, trials=1),
                ],
                destination,
                timeout=1200,
            )
            if not report.is_file():
                raise RuntimeError(f"NCU did not produce {report}")
            BASE.export_ncu(ncu, report, destination)
            print(f"ncu {progress:03d}/{total:03d} {cell.key} r{repeat}", flush=True)


def run_source(output: Path, force: bool) -> None:
    ncu = os.environ.get("NCU", "ncu")
    cells = [TripletCell(tuple(regions)) for regions in SOURCE_TRIPLETS]
    for index, cell in enumerate(cells, 1):
        destination = output / "ncu-source" / cell.key
        report_base = destination / "ncu-report"
        report = destination / "ncu-report.ncu-rep"
        if report.is_file() and (destination / "source.csv").is_file() and not force:
            print(f"source {index:02d}/{len(cells):02d} {cell.key} (existing)")
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
        (destination / "source-export.stdout.txt").write_text(
            completed.stdout, encoding="utf-8"
        )
        (destination / "source-export.stderr.txt").write_text(
            completed.stderr, encoding="utf-8"
        )
        if completed.returncode != 0 or not (destination / "source.csv").is_file():
            raise RuntimeError(f"source export failed for {cell.key}")
        print(f"source {index:02d}/{len(cells):02d} {cell.key}", flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "action",
        choices=(
            "manifest",
            "correctness",
            "trace-correctness",
            "ncu-pairs",
            "ncu",
            "source",
            "nvbit",
            "static",
            "simulator",
            "trace-keys",
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
        emit_manifest(output)
        if args.action == "manifest":
            print(output / "metadata/matrix.json")
        elif args.action == "trace-keys":
            print(",".join(cell.key for cell in trace_cells()))
        elif args.action == "correctness":
            run_correctness(output, triplet_cells(), args.trials, trace=False)
        elif args.action == "trace-correctness":
            run_correctness(
                output,
                selected_triplets(args.cells) if args.cells else trace_cells(),
                args.trials,
                trace=True,
            )
        elif args.action == "ncu-pairs":
            run_pair_ncu(output, pair_cells(), args.force)
        elif args.action == "ncu":
            run_full_ncu(
                output,
                selected_triplets(args.cells) if args.cells else trace_cells(),
                args.repeat_start,
                args.repeats,
                args.force,
            )
        elif args.action == "source":
            run_source(output, args.force)
        elif args.action == "nvbit":
            ORDER.run_nvbit(
                output,
                selected_triplets(args.cells) if args.cells else trace_cells(),
                args.force,
            )
        elif args.action == "static":
            PREVIOUS.run_static(output)
        elif args.action == "simulator":
            PREVIOUS.run_simulator(
                output,
                selected_triplets(args.cells) if args.cells else trace_cells(),
                args.force,
            )
    except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(f"many-region atomic campaign: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
