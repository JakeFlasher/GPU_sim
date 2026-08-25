#!/usr/bin/env python3
"""Run the GA102 same-physical/multiple-virtual-alias LTS discriminator."""

from __future__ import annotations

import argparse
import importlib
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
MAPPING_DIR = HERE.parent / "atomic_lts_mapping"
DRIVER = HERE / "vmm_alias_driver"
DRIVER_SOURCE = HERE / "vmm_alias_driver.cpp"
CUBIN = HERE / "vmm_alias_read.cubin"
ATOMIC_DRIVER = HERE / "vmm_alias_atomic_driver"
ATOMIC_DRIVER_SOURCE = HERE / "vmm_alias_atomic_driver.cpp"
ATOMIC_CUBIN = HERE / "vmm_alias_atomic.cubin"
CROSS_DRIVER = HERE / "vmm_cross_allocation_driver"
CROSS_DRIVER_SOURCE = HERE / "vmm_cross_allocation_driver.cpp"
DEFAULT_OUTPUT = REPO_ROOT / "results/captures/sm86_atomic_vmm_alias_20260820_v1"
OFFSETS = (0, 128, 512, 4096, 65_536, 524_288, 1_048_576, 2_096_128)
PRIMARY_MODES = ("alias_same_offset", "single_mapping_alias_stride")
CONTROL_MODES = (
    "single_alias_same_offset",
    "alias_distinct_offsets",
    "single_alias_distinct_offsets",
)
GROUP_BLOCK_COUNTS = (105, 1, 2, 4, 8, 16, 32)


def load_mapping_campaign():
    path = MAPPING_DIR / "campaign.py"
    spec = importlib.util.spec_from_file_location("vmm_mapping_campaign", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MAPPING = load_mapping_campaign()
BASE = MAPPING.BASE


@dataclass(frozen=True, slots=True)
class Cell:
    mode: str
    offset_bytes: int
    primary: bool

    @property
    def key(self) -> str:
        return f"{self.mode}_o{self.offset_bytes:07d}"

    def target(self, trials: int) -> list[str]:
        return [
            str(DRIVER),
            str(CUBIN),
            str(trials),
            self.key,
            self.mode,
            str(self.offset_bytes),
        ]


def cells() -> list[Cell]:
    result = [
        Cell(mode, offset, True)
        for offset in OFFSETS
        for mode in PRIMARY_MODES
    ]
    result.extend(Cell(mode, 0, False) for mode in CONTROL_MODES)
    if len(result) != 19 or len({cell.key for cell in result}) != 19:
        raise RuntimeError("VMM alias matrix must contain 19 unique cells")
    return result


def ensure_inputs() -> None:
    completed = subprocess.run(
        ["make", "all"],
        cwd=HERE,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"VMM alias build failed:\n{completed.stderr}")
    for path in (
        DRIVER_SOURCE,
        DRIVER,
        CUBIN,
        ATOMIC_DRIVER_SOURCE,
        ATOMIC_DRIVER,
        ATOMIC_CUBIN,
        CROSS_DRIVER_SOURCE,
        CROSS_DRIVER,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"required input is missing: {path}")
    if DRIVER.stat().st_mtime_ns <= DRIVER_SOURCE.stat().st_mtime_ns:
        raise RuntimeError("VMM alias driver is not newer than its source")
    if ATOMIC_DRIVER.stat().st_mtime_ns <= ATOMIC_DRIVER_SOURCE.stat().st_mtime_ns:
        raise RuntimeError("VMM alias atomic driver is not newer than its source")
    if CROSS_DRIVER.stat().st_mtime_ns <= CROSS_DRIVER_SOURCE.stat().st_mtime_ns:
        raise RuntimeError("VMM cross-allocation driver is not newer than its source")


def emit_manifest(output: Path) -> None:
    ensure_inputs()
    BASE.write_json(
        output / "metadata/matrix.json",
        {
            "schema_version": 1,
            "architecture": "sm_86",
            "gpu": "NVIDIA GeForce RTX 3090 Ti",
            "cells": [{**asdict(cell), "key": cell.key} for cell in cells()],
            "primary_modes": list(PRIMARY_MODES),
            "control_modes": list(CONTROL_MODES),
            "offsets": list(OFFSETS),
            "blocks": 168,
            "generations": 64,
            "group_block_counts": list(GROUP_BLOCK_COUNTS),
            "metric": "lts__t_sectors_op_read.sum",
            "ncu_repeats": {"primary": 7, "control": 3},
            "atomic_alias_offsets": list(OFFSETS),
            "atomic_alias_repeats": 7,
            "atomic_mapping_launches_per_profile": 55,
            "relative_dense_regions": 16_384,
            "relative_dense_mapping_launches": 16_432,
            "relative_contiguous_8m_regions": 49_151,
            "relative_high_holdout_32m_regions": 32_768,
            "cross_allocation_count": 8,
            "cross_allocation_repeats": 7,
            "cross_allocation_mapping_launches": 432,
            "cache_control": "all",
            "invariants": [
                "read-only LDG.E.STRONG.GPU avoids undefined alias writes",
                "primary A/B cells use byte-identical virtual group addresses",
                "alias_same_offset maps seven virtual proxies to one 2 MiB allocation",
                "single_mapping_alias_stride maps one contiguous 14 MiB allocation",
                "all mappings are initialized before and synchronized before the probe",
                "sequential RED uses one alias proxy per synchronized kernel",
                "every RED report self-validates all 48 reference classes before aliases",
                "cross-allocation reports classify the same 48 relative representatives",
            ],
        },
    )


def ncu_report_module():
    candidates = sorted(
        Path("/usr/local").glob("cuda-*/nsight-compute-*/extras/python"),
        reverse=True,
    )
    for directory in candidates:
        if (directory / "ncu_report.py").is_file():
            if str(directory) not in sys.path:
                sys.path.insert(0, str(directory))
            return importlib.import_module("ncu_report")
    raise RuntimeError("Nsight Compute Python report interface was not found")


def extract_metrics(report: Path) -> dict[str, int]:
    ncu_report = ncu_report_module()
    context = ncu_report.load_report(report)
    actions = [
        context.range_by_idx(range_index).action_by_idx(action_index)
        for range_index in range(context.num_ranges())
        for action_index in range(context.range_by_idx(range_index).num_actions())
    ]
    if len(actions) != 1:
        raise RuntimeError(f"{report}: expected one action, got {len(actions)}")
    action = actions[0]
    return {
        "maximum_sectors": int(
            action.metric_by_name("lts__t_sectors_op_read.max").value()
        ),
        "total_sectors": int(
            action.metric_by_name("lts__t_sectors_op_read.sum").value()
        ),
        "minimum_sectors": int(
            action.metric_by_name("lts__t_sectors_op_read.min").value()
        ),
    }


def extract_atomic_metrics(report: Path, expected_actions: int = 55) -> list[dict[str, int]]:
    ncu_report = ncu_report_module()
    context = ncu_report.load_report(report)
    actions = [
        context.range_by_idx(range_index).action_by_idx(action_index)
        for range_index in range(context.num_ranges())
        for action_index in range(context.range_by_idx(range_index).num_actions())
    ]
    if len(actions) != expected_actions:
        raise RuntimeError(
            f"{report}: expected {expected_actions} RED actions, got {len(actions)}"
        )
    return [
        {
            "id": index,
            "maximum_sectors": int(
                action.metric_by_name("lts__t_sectors_op_red.max").value()
            ),
            "total_sectors": int(
                action.metric_by_name("lts__t_sectors_op_red.sum").value()
            ),
        }
        for index, action in enumerate(actions)
    ]


def validate_measurement(cell: Cell, measurement: dict[str, object]) -> None:
    aliases = measurement.get("alias_addresses")
    groups = measurement.get("group_addresses")
    if (
        measurement.get("correct") is not True
        or measurement.get("mode") != cell.mode
        or measurement.get("physical_offset_bytes") != cell.offset_bytes
        or measurement.get("allocation_granularity_bytes") != 2 * 1024 * 1024
        or measurement.get("group_block_counts") != list(GROUP_BLOCK_COUNTS)
        or not isinstance(aliases, list)
        or len(aliases) != 7
        or not isinstance(groups, list)
        or len(groups) != 7
        or any(aliases[index] - aliases[0] != index * 2 * 1024 * 1024 for index in range(7))
    ):
        raise RuntimeError(f"{cell.key}: malformed VMM measurement")


def run_correctness(output: Path, force: bool) -> None:
    for index, cell in enumerate(cells(), 1):
        destination = output / "correctness" / cell.key
        result_path = destination / "measurement.json"
        if result_path.is_file() and not force:
            print(f"correctness {index:02d}/19 {cell.key} (existing)")
            continue
        completed = BASE.run_captured(cell.target(7), destination, timeout=180)
        measurement = BASE.measurement_json_line(completed.stdout, cell.key)
        validate_measurement(cell, measurement)
        BASE.write_json(result_path, measurement)
        print(f"correctness {index:02d}/19 {cell.key}", flush=True)


def run_ncu(output: Path, force: bool) -> None:
    ncu = os.environ.get("NCU", "ncu")
    total = sum(7 if cell.primary else 3 for cell in cells())
    progress = 0
    for cell in cells():
        repeats = 7 if cell.primary else 3
        for repeat in range(1, repeats + 1):
            progress += 1
            destination = output / "ncu" / cell.key / f"repeat-{repeat}"
            report_base = destination / "ncu-report"
            report = destination / "ncu-report.ncu-rep"
            result_path = destination / "result.json"
            if report.is_file() and result_path.is_file() and not force:
                print(f"ncu {progress:03d}/{total:03d} {cell.key} r{repeat} (existing)")
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
                    "all",
                    "--metrics",
                    "lts__t_sectors_op_read.sum",
                    *cell.target(1),
                ],
                destination,
                timeout=1200,
            )
            if not report.is_file():
                raise RuntimeError(f"NCU did not produce {report}")
            measurement = BASE.measurement_json_line(completed.stdout, cell.key)
            validate_measurement(cell, measurement)
            BASE.write_json(
                result_path,
                {
                    "schema_version": 1,
                    "cell": {**asdict(cell), "key": cell.key},
                    "repeat": repeat,
                    "measurement": measurement,
                    "metrics": extract_metrics(report),
                },
            )
            print(f"ncu {progress:03d}/{total:03d} {cell.key} r{repeat}", flush=True)


def run_atomic(output: Path, force: bool) -> None:
    ncu = os.environ.get("NCU", "ncu")
    total = len(OFFSETS) * 7
    progress = 0
    for offset in OFFSETS:
        key = f"physical_offset_o{offset:07d}"
        for repeat in range(1, 8):
            progress += 1
            destination = output / "atomic" / key / f"repeat-{repeat}"
            report_base = destination / "ncu-report"
            report = destination / "ncu-report.ncu-rep"
            result_path = destination / "result.json"
            if report.is_file() and result_path.is_file() and not force:
                print(f"atomic {progress:02d}/{total:02d} {key} r{repeat} (existing)")
                continue
            variant = f"atomic_{key}_r{repeat}"
            completed = BASE.run_captured(
                [
                    ncu,
                    "--force-overwrite",
                    "--export",
                    str(report_base),
                    "--profile-from-start",
                    "on",
                    "--kernel-name",
                    "atomic_vmm_mapping_probe",
                    "--launch-count",
                    "55",
                    "--replay-mode",
                    "application",
                    "--clock-control",
                    "base",
                    "--cache-control",
                    "none",
                    "--metrics",
                    "lts__t_sectors_op_red.sum",
                    str(ATOMIC_DRIVER),
                    str(ATOMIC_CUBIN),
                    str(offset),
                    variant,
                ],
                destination,
                timeout=1200,
            )
            if not report.is_file():
                raise RuntimeError(f"NCU did not produce {report}")
            measurement = BASE.measurement_json_line(completed.stdout, variant)
            if (
                measurement.get("correct") is not True
                or measurement.get("physical_offset_bytes") != offset
                or measurement.get("mapping_launches") != 55
                or measurement.get("validation_launches") != 48
                or measurement.get("alias_launches") != 7
                or measurement.get("allocation_granularity_bytes")
                != 2 * 1024 * 1024
            ):
                raise RuntimeError(f"{key} r{repeat}: malformed atomic measurement")
            rows = extract_atomic_metrics(report)
            if any(row["total_sectors"] != 37_632 for row in rows):
                raise RuntimeError(f"{key} r{repeat}: RED sector conservation")
            decoded = [row["maximum_sectors"] // 32 - 48 for row in rows]
            if decoded[:48] != list(range(48)):
                raise RuntimeError(f"{key} r{repeat}: reference basis changed")
            if any(row["maximum_sectors"] % 32 != 0 for row in rows):
                raise RuntimeError(f"{key} r{repeat}: fractional RED class")
            BASE.write_json(
                result_path,
                {
                    "schema_version": 1,
                    "offset_bytes": offset,
                    "repeat": repeat,
                    "measurement": measurement,
                    "metrics": rows,
                    "validation_classes": decoded[:48],
                    "alias_classes": decoded[48:],
                },
            )
            print(f"atomic {progress:02d}/{total:02d} {key} r{repeat}", flush=True)


def run_relative_set(
    output: Path,
    stage: str,
    regions: Sequence[int],
    force: bool,
) -> None:
    if not regions or len(regions) > 65_000 or len(set(regions)) != len(regions):
        raise RuntimeError(f"{stage}: invalid relative-region roster")
    maximum_region = (32 * 1024 * 1024) // 128
    if any(not 0 <= region < maximum_region for region in regions):
        raise RuntimeError(f"{stage}: relative region outside 32 MiB scope")
    offsets = [region * 128 for region in regions]
    destination = output / stage
    report_base = destination / "ncu-report"
    report = destination / "ncu-report.ncu-rep"
    result_path = destination / "classification.json"
    offsets_path = destination / "physical-offsets.txt"
    if report.is_file() and result_path.is_file() and not force:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if len(payload.get("records", ())) == len(regions):
            print(f"{stage} {len(regions)} regions (existing)")
            return
    destination.mkdir(parents=True, exist_ok=True)
    offsets_path.write_text(",".join(map(str, offsets)), encoding="utf-8")
    ncu = os.environ.get("NCU", "ncu")
    variant = f"atomic_vmm_{stage.replace('-', '_')}"
    expected_actions = 48 + len(regions)
    completed = BASE.run_captured(
        [
            ncu,
            "--force-overwrite",
            "--export",
            str(report_base),
            "--profile-from-start",
            "on",
            "--kernel-name",
            "atomic_vmm_mapping_probe",
            "--launch-count",
            str(expected_actions),
            "--replay-mode",
            "application",
            "--clock-control",
            "base",
            "--cache-control",
            "none",
            "--metrics",
            "lts__t_sectors_op_red.sum",
            str(ATOMIC_DRIVER),
            str(ATOMIC_CUBIN),
            "0",
            variant,
            str(offsets_path),
        ],
        destination,
        timeout=3600,
    )
    if not report.is_file():
        raise RuntimeError(f"NCU did not produce {report}")
    measurement = BASE.measurement_json_line(completed.stdout, variant)
    if (
        measurement.get("correct") is not True
        or measurement.get("batch_offsets") is not True
        or measurement.get("alias_launches") != len(regions)
        or measurement.get("mapping_launches") != expected_actions
        or measurement.get("mapping_offsets_bytes") != offsets
    ):
        raise RuntimeError(f"{stage}: malformed measurement")
    rows = extract_atomic_metrics(report, expected_actions)
    if any(row["total_sectors"] != 37_632 for row in rows):
        raise RuntimeError(f"{stage}: RED sector conservation")
    if any(row["maximum_sectors"] % 32 != 0 for row in rows):
        raise RuntimeError(f"{stage}: fractional RED class")
    decoded = [row["maximum_sectors"] // 32 - 48 for row in rows]
    if decoded[:48] != list(range(48)):
        raise RuntimeError(f"{stage}: reference basis changed")
    classes = decoded[48:]
    if any(not 0 <= value < 48 for value in classes):
        raise RuntimeError(f"{stage}: class range")
    BASE.write_json(
        result_path,
        {
            "schema_version": 1,
            "stage": stage,
            "measurement": measurement,
            "validation_classes": decoded[:48],
            "records": [
                {
                    "region": region,
                    "offset_bytes": offset,
                    "alias_index": index % 7,
                    "observed_class": observed_class,
                }
                for index, (region, offset, observed_class) in enumerate(
                    zip(regions, offsets, classes, strict=True)
                )
            ],
        },
    )
    print(
        f"{stage} {expected_actions} profiled launches -> "
        f"{len(regions)} physical-offset classes",
        flush=True,
    )


def run_relative(output: Path, force: bool) -> None:
    run_relative_set(
        output,
        "relative",
        list(range((2 * 1024 * 1024) // 128)),
        force,
    )


def run_relative_extended(output: Path, force: bool) -> None:
    run_relative_set(
        output,
        "relative-contiguous-8m",
        list(range((2 * 1024 * 1024) // 128, 65_535)),
        force,
    )
    predecessor = json.loads(
        (
            MAPPING.DEFAULT_OUTPUT
            / "profiles/train/encoded-holdout.json"
        ).read_text(encoding="utf-8")
    )
    holdout_regions = [int(record["region"]) for record in predecessor["records"]]
    run_relative_set(
        output,
        "relative-high-holdout-32m",
        holdout_regions,
        force,
    )


def run_cross_allocation(output: Path, force: bool) -> None:
    ncu = os.environ.get("NCU", "ncu")
    for repeat in range(1, 8):
        destination = output / "cross-allocation" / f"repeat-{repeat}"
        report_base = destination / "ncu-report"
        report = destination / "ncu-report.ncu-rep"
        result_path = destination / "result.json"
        if report.is_file() and result_path.is_file() and not force:
            print(f"cross-allocation r{repeat} (existing)")
            continue
        variant = f"atomic_vmm_cross_allocation_r{repeat}"
        completed = BASE.run_captured(
            [
                ncu,
                "--force-overwrite",
                "--export",
                str(report_base),
                "--profile-from-start",
                "on",
                "--kernel-name",
                "atomic_vmm_mapping_probe",
                "--launch-count",
                "432",
                "--replay-mode",
                "application",
                "--clock-control",
                "base",
                "--cache-control",
                "none",
                "--metrics",
                "lts__t_sectors_op_red.sum",
                str(CROSS_DRIVER),
                str(ATOMIC_CUBIN),
                variant,
            ],
            destination,
            timeout=1200,
        )
        if not report.is_file():
            raise RuntimeError(f"NCU did not produce {report}")
        measurement = BASE.measurement_json_line(completed.stdout, variant)
        if (
            measurement.get("correct") is not True
            or measurement.get("allocation_count") != 8
            or measurement.get("validation_launches") != 48
            or measurement.get("target_launches") != 384
            or measurement.get("mapping_launches") != 432
        ):
            raise RuntimeError(f"cross-allocation r{repeat}: malformed measurement")
        rows = extract_atomic_metrics(report, 432)
        if any(row["total_sectors"] != 37_632 for row in rows):
            raise RuntimeError(f"cross-allocation r{repeat}: sector conservation")
        if any(row["maximum_sectors"] % 32 != 0 for row in rows):
            raise RuntimeError(f"cross-allocation r{repeat}: fractional class")
        decoded = [row["maximum_sectors"] // 32 - 48 for row in rows]
        if decoded[:48] != list(range(48)):
            raise RuntimeError(f"cross-allocation r{repeat}: reference basis")
        class_maps = [
            decoded[48 + allocation * 48 : 48 + (allocation + 1) * 48]
            for allocation in range(8)
        ]
        if any(
            not all(0 <= value < 48 for value in class_map)
            for class_map in class_maps
        ):
            raise RuntimeError(f"cross-allocation r{repeat}: class range")
        BASE.write_json(
            result_path,
            {
                "schema_version": 1,
                "repeat": repeat,
                "measurement": measurement,
                "validation_classes": decoded[:48],
                "allocation_class_maps": class_maps,
                "allocation_unique_class_counts": [
                    len(set(class_map)) for class_map in class_maps
                ],
            },
        )
        print(f"cross-allocation r{repeat}", flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "action",
        choices=(
            "manifest",
            "correctness",
            "ncu",
            "atomic",
            "relative",
            "relative-extended",
            "cross-allocation",
            "all",
        ),
    )
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--force", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = args.output_root.resolve()
    try:
        emit_manifest(output)
        if args.action == "manifest":
            print(output / "metadata/matrix.json")
        if args.action in {"correctness", "all"}:
            run_correctness(output, args.force)
        if args.action in {"ncu", "all"}:
            run_ncu(output, args.force)
        if args.action in {"atomic", "all"}:
            run_atomic(output, args.force)
        if args.action in {"relative", "all"}:
            run_relative(output, args.force)
        if args.action in {"relative-extended", "all"}:
            run_relative_extended(output, args.force)
        if args.action in {"cross-allocation", "all"}:
            run_cross_allocation(output, args.force)
    except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(f"VMM alias campaign: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
