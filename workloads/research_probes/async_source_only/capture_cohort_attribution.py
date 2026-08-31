#!/usr/bin/env python3
"""Capture the frozen SM86 paired async-cohort NCU attribution matrix.

Each logical observation runs both pairing variants in one application, in a
recorded AB or BA order.  Five metrics are collected separately so every NCU
report is a single application-replay pass.  Raw DRAM is retained as a
device-background sentinel; only the L1TEX-source, local-device-aperture LTS
request stream is used for source attribution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = (
    REPO_ROOT / "results/captures/sm86_async_cohort_reservation_20260831_v1"
)
BINARY = HERE / "async_cohort_pairing"
EXPECTED_BINARY_SHA256 = (
    "dded4cb03868f5cc757f86103d3ec45d4c8adf66c580b39ac90e3c4034f94f4f"
)

GROUPS = 64
DEPTHS = (1, 2, 4, 8)
SPACINGS = (0, 4)
REPETITIONS = tuple(range(1, 11))
GPU_INDEX = "0"
EXPECTED_GPU = {
    "index": GPU_INDEX,
    "name": "NVIDIA GeForce RTX 3090 Ti",
    "uuid": "GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c",
    "compute_capability": "8.6",
    "driver": "610.57.04",
}

FAMILY_METRICS = {
    "ldgsts": "smsp__inst_executed_op_ldgsts.sum",
    "lts_hit": (
        "lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_hit.sum"
    ),
    "lts_miss": (
        "lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_miss.sum"
    ),
    "lts_total": "lts__t_sectors_srcunit_tex_aperture_device_op_read.sum",
    "dram": "dram__sectors_read.sum",
}
FAMILIES = tuple(FAMILY_METRICS)
CELL_RE = re.compile(r"^d(?P<depth>1|2|4|8)s(?P<spacing>0|4)$")


class CampaignError(RuntimeError):
    """Raised when capture provenance or a measured invariant is invalid."""


@dataclass(frozen=True)
class Cell:
    depth: int
    spacing: int

    @property
    def short(self) -> str:
        return f"d{self.depth}s{self.spacing}"

    @property
    def key(self) -> str:
        return (
            f"async_cohort_pairing_d{self.depth}_s{self.spacing}_g{GROUPS}"
        )


@dataclass(frozen=True)
class Preflight:
    binary: Mapping[str, Any]
    device: Mapping[str, str]
    ncu_path: str
    ncu_version: str
    environment: Mapping[str, str]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json_text(value))


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise CampaignError(f"{path}: expected one JSON object")
    return value


def integer_metric(row: Mapping[str, str], name: str, context: str) -> int:
    try:
        value = float(row[name].replace(",", ""))
    except (KeyError, ValueError) as error:
        raise CampaignError(f"{context}: missing/non-numeric {name}") from error
    if not math.isfinite(value) or value < 0 or not value.is_integer():
        raise CampaignError(f"{context}: {name} is not a nonnegative integer: {value}")
    return int(value)


def csv_rows(text: str, context: str) -> list[dict[str, str]]:
    try:
        rows = [
            dict(row)
            for row in csv.DictReader(io.StringIO(text))
            if str(row.get("ID", "")).strip().isdigit()
        ]
    except csv.Error as error:
        raise CampaignError(f"{context}: invalid NCU CSV: {error}") from error
    return rows


def expected_order(repetition: int) -> str:
    return "ab" if repetition % 2 else "ba"


def row_variants(order: str) -> tuple[str, str]:
    return (
        ("slow_expensive", "slow_cheap")
        if order == "ab"
        else ("slow_cheap", "slow_expensive")
    )


def source_sectors(cell: Cell) -> int:
    return GROUPS * cell.depth * 32


def paired_tolerance(cell: Cell) -> int:
    return max(4, math.ceil(0.01 * source_sectors(cell)))


def run_probe(
    command: Sequence[str], *, timeout: int, environment: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            env=dict(environment) if environment is not None else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CampaignError(f"command failed before completion: {command}: {error}") from error


def preflight(ncu: str) -> Preflight:
    if not BINARY.is_file() or not os.access(BINARY, os.X_OK):
        raise CampaignError(f"frozen probe binary is missing or not executable: {BINARY}")
    binary_hash = sha256(BINARY)
    if binary_hash != EXPECTED_BINARY_SHA256:
        raise CampaignError(
            "probe binary identity mismatch: "
            f"observed {binary_hash}, expected {EXPECTED_BINARY_SHA256}"
        )

    ncu_found = shutil.which(ncu)
    if ncu_found is None:
        raise CampaignError(f"NCU executable is unavailable: {ncu}")
    ncu_path = str(Path(ncu_found).resolve())
    version = run_probe([ncu_path, "--version"], timeout=30)
    if version.returncode != 0:
        raise CampaignError(f"NCU version query failed: {version.stderr.strip()}")

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise CampaignError("nvidia-smi is unavailable")
    device_query = [
        nvidia_smi,
        "-i",
        GPU_INDEX,
        "--query-gpu=index,name,uuid,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    ]
    queried = run_probe(device_query, timeout=30)
    if queried.returncode != 0:
        raise CampaignError(f"GPU identity query failed: {queried.stderr.strip()}")
    try:
        records = list(csv.reader(io.StringIO(queried.stdout), skipinitialspace=True))
    except csv.Error as error:
        raise CampaignError(f"invalid GPU identity output: {error}") from error
    if len(records) != 1 or len(records[0]) != 5:
        raise CampaignError(f"expected one five-field GPU identity row: {records!r}")
    values = [item.strip() for item in records[0]]
    device = dict(zip(EXPECTED_GPU, values, strict=True))
    if device != EXPECTED_GPU:
        raise CampaignError(
            f"GPU identity mismatch: observed {device}, expected {EXPECTED_GPU}"
        )

    environment = {
        "CUDA_VISIBLE_DEVICES": GPU_INDEX,
        "LANG": "C",
        "LC_ALL": "C",
    }
    return Preflight(
        binary={
            "path": str(BINARY),
            "sha256": binary_hash,
            "size_bytes": BINARY.stat().st_size,
        },
        device=device,
        ncu_path=ncu_path,
        ncu_version=(version.stdout + version.stderr).strip(),
        environment=environment,
    )


def target_arguments(cell: Cell, repetition: int) -> list[str]:
    return [
        "--groups",
        str(GROUPS),
        "--depth",
        str(cell.depth),
        "--spacing-samples",
        str(cell.spacing),
        "--order",
        expected_order(repetition),
        "--markers",
        "on",
    ]


def profile_command(
    context: Preflight,
    family_directory: Path,
    cell: Cell,
    repetition: int,
    family: str,
) -> list[str]:
    return [
        context.ncu_path,
        "--devices",
        GPU_INDEX,
        "--target-processes",
        "application-only",
        "--profile-from-start",
        "off",
        "--launch-count",
        "2",
        "--replay-mode",
        "application",
        "--app-replay-mode",
        "strict",
        "--app-replay-match",
        "grid",
        "--cache-control",
        "none",
        "--clock-control",
        "none",
        "--kernel-name",
        "regex:cohort_pairing_kernel",
        "--metrics",
        FAMILY_METRICS[family],
        "--export",
        str(family_directory / "profile"),
        "--force-overwrite",
        str(BINARY),
        *target_arguments(cell, repetition),
    ]


def import_command(context: Preflight, family_directory: Path) -> list[str]:
    return [
        context.ncu_path,
        "--import",
        str(family_directory / "profile.ncu-rep"),
        "--page",
        "raw",
        "--csv",
        "--print-units",
        "base",
        "--print-fp",
    ]


def command_document(
    context: Preflight,
    family_directory: Path,
    cell: Cell,
    repetition: int,
    family: str,
) -> dict[str, Any]:
    order = expected_order(repetition)
    return {
        "schema_version": 1,
        "cell": cell.key,
        "depth": cell.depth,
        "spacing": cell.spacing,
        "groups": GROUPS,
        "repetition": repetition,
        "order": order,
        "row_variants": list(row_variants(order)),
        "family": family,
        "metric": FAMILY_METRICS[family],
        "binary": dict(context.binary),
        "device": dict(context.device),
        "ncu": {"path": context.ncu_path, "version": context.ncu_version},
        "environment_overrides": dict(context.environment),
        "target_arguments": target_arguments(cell, repetition),
        "profile_command": profile_command(
            context, family_directory, cell, repetition, family
        ),
        "import_command": import_command(context, family_directory),
    }


def validate_target_output(
    text: str, cell: Cell, repetition: int, context: Preflight
) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.lstrip().startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise CampaignError(f"invalid target JSON: {error}") from error
        if isinstance(value, dict):
            payloads.append(value)
    if len(payloads) != 1:
        raise CampaignError(
            f"expected one target JSON from one replay pass, got {len(payloads)}"
        )
    payload = payloads[0]
    expected = {
        "workload": "async_cohort_pairing",
        "route": "l2",
        "mode": "pair",
        "order": expected_order(repetition),
        "depth": cell.depth,
        "groups": GROUPS,
        "warmup_groups": 64,
        "spacing_clock_samples_per_edge": cell.spacing,
        "device": context.device["name"],
        "compute_capability": context.device["compute_capability"],
        "source_pool_single_allocation": True,
        "independent_cold_regions": True,
        "source_vector_preserved": True,
        "destination_multiset_preserved": True,
        "static_ldgsts_positions_preserved": True,
        "result_store_after_intervals": True,
        "correct": True,
    }
    for field, expected_value in expected.items():
        if payload.get(field) != expected_value:
            raise CampaignError(
                f"target binding mismatch for {field}: "
                f"observed {payload.get(field)!r}, expected {expected_value!r}"
            )
    for variant in ("slow_expensive", "slow_cheap"):
        result = payload.get(variant)
        if not isinstance(result, dict) or result.get("correct") is not True:
            raise CampaignError(f"target result is not correct for {variant}")
        if result.get("mismatches") != 0:
            raise CampaignError(f"target reported mismatches for {variant}")
    return payload


def validate_rows(
    rows: Sequence[Mapping[str, str]],
    cell: Cell,
    repetition: int,
    family: str,
    context: Preflight,
) -> dict[str, Any]:
    label = f"{cell.key} repeat-{repetition}/{family}"
    if len(rows) != 2:
        raise CampaignError(f"{label}: expected two target rows, got {len(rows)}")
    ids = [str(row.get("ID", "")).strip() for row in rows]
    if ids != ["0", "1"]:
        raise CampaignError(f"{label}: row IDs are not ordered 0/1: {ids}")
    process_ids = [str(row.get("Process ID", "")).strip() for row in rows]
    if not process_ids[0] or process_ids[0] != process_ids[1]:
        raise CampaignError(f"{label}: rows do not belong to one application process")

    template = f"cohort_pairing_kernel<{cell.depth},{cell.spacing}>"
    metric = FAMILY_METRICS[family]
    variants = row_variants(expected_order(repetition))
    values: dict[str, int] = {}
    row_summaries: list[dict[str, Any]] = []
    for index, (row, variant) in enumerate(zip(rows, variants, strict=True)):
        context_label = f"{label} row {index}"
        kernel = str(row.get("Kernel Name", ""))
        if template not in kernel.replace(" ", ""):
            raise CampaignError(f"{context_label}: wrong kernel: {kernel}")
        expected_strings = {
            "Process Name": "async_cohort_pairing",
            "Device": GPU_INDEX,
            "CC": context.device["compute_capability"],
            "Grid Size": "(1, 1, 1)",
            "Block Size": "(32, 1, 1)",
            "device__attribute_display_name": context.device["name"],
        }
        for field, expected_value in expected_strings.items():
            observed = str(row.get(field, "")).strip()
            if observed != expected_value:
                raise CampaignError(
                    f"{context_label}: {field}={observed!r}, expected {expected_value!r}"
                )
        numeric_identity = {
            "device__attribute_device_index": 0,
            "device__attribute_compute_capability_major": 8,
            "device__attribute_compute_capability_minor": 6,
            "launch__grid_size": 1,
            "launch__block_size": 32,
            "profiler__replayer_passes": 1,
        }
        for field, expected_value in numeric_identity.items():
            observed = integer_metric(row, field, context_label)
            if observed != expected_value:
                raise CampaignError(
                    f"{context_label}: {field}={observed}, expected {expected_value}"
                )
        value = integer_metric(row, metric, context_label)
        values[variant] = value
        row_summaries.append(
            {
                "id": index,
                "variant": variant,
                "process_id": process_ids[index],
                "kernel": kernel,
                "context": str(row.get("Context", "")),
                "stream": str(row.get("Stream", "")),
                "metric_value": value,
                "profiler_replayer_passes": 1,
            }
        )

    if family == "ldgsts":
        expected_count = GROUPS * cell.depth
        if any(value != expected_count for value in values.values()):
            raise CampaignError(
                f"{label}: LDGSTS counts {values} do not equal {expected_count}"
            )
    elif family in {"lts_hit", "lts_miss"}:
        if any(value <= 0 for value in values.values()):
            raise CampaignError(f"{label}: attributed LTS population is not positive")
        difference = abs(values["slow_expensive"] - values["slow_cheap"])
        if difference > paired_tolerance(cell):
            raise CampaignError(
                f"{label}: paired LTS difference {difference} exceeds "
                f"tolerance {paired_tolerance(cell)}"
            )
    elif family == "lts_total":
        expected_count = source_sectors(cell)
        if any(value != expected_count for value in values.values()):
            raise CampaignError(
                f"{label}: attributed LTS totals {values} do not equal "
                f"{expected_count}"
            )

    return {
        "order": expected_order(repetition),
        "row_variants": list(variants),
        "metric": metric,
        "values_by_variant": values,
        "rows": row_summaries,
    }


def expected_complete_document(
    family_directory: Path,
    command: Mapping[str, Any],
    summary: Mapping[str, Any],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    report = family_directory / "profile.ncu-rep"
    csv_path = family_directory / "ncu.csv"
    stdout = family_directory / "profile.stdout.txt"
    command_path = family_directory / "command.json"
    return {
        "schema_version": 1,
        "valid": True,
        "binding": {
            key: command[key]
            for key in (
                "cell",
                "depth",
                "spacing",
                "groups",
                "repetition",
                "order",
                "row_variants",
                "family",
                "metric",
                "binary",
                "device",
                "ncu",
                "target_arguments",
            )
        },
        "files": {
            "command.json": sha256(command_path),
            "profile.ncu-rep": sha256(report),
            "ncu.csv": sha256(csv_path),
            "profile.stdout.txt": sha256(stdout),
        },
        "summary": dict(summary),
        "target_result": dict(target),
    }


def validate_retained_family(
    family_directory: Path,
    context: Preflight,
    cell: Cell,
    repetition: int,
    family: str,
) -> dict[str, Any]:
    paths = {
        name: family_directory / name
        for name in (
            "command.json",
            "profile.ncu-rep",
            "ncu.csv",
            "profile.stdout.txt",
            "complete.json",
        )
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise CampaignError(f"{family_directory}: incomplete family, missing {missing}")
    if paths["profile.ncu-rep"].stat().st_size == 0:
        raise CampaignError(f"{family_directory}: empty NCU report")

    expected_command = command_document(
        context, family_directory, cell, repetition, family
    )
    observed_command = read_json(paths["command.json"])
    if observed_command != expected_command:
        raise CampaignError(f"{family_directory}: retained command binding mismatch")
    rows = csv_rows(paths["ncu.csv"].read_text(encoding="utf-8"), str(paths["ncu.csv"]))
    summary = validate_rows(rows, cell, repetition, family, context)
    target = validate_target_output(
        paths["profile.stdout.txt"].read_text(encoding="utf-8"),
        cell,
        repetition,
        context,
    )
    expected_complete = expected_complete_document(
        family_directory, observed_command, summary, target
    )
    observed_complete = read_json(paths["complete.json"])
    if observed_complete != expected_complete:
        raise CampaignError(f"{family_directory}: completion record/hash mismatch")
    return expected_complete


def execution_environment(context: Preflight) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(context.environment)
    return environment


def execute_family(
    family_directory: Path,
    context: Preflight,
    cell: Cell,
    repetition: int,
    family: str,
    timeout: int,
) -> dict[str, Any]:
    try:
        return validate_retained_family(
            family_directory, context, cell, repetition, family
        )
    except CampaignError as error:
        retained_reason = str(error)

    family_directory.mkdir(parents=True, exist_ok=True)
    command = command_document(context, family_directory, cell, repetition, family)
    atomic_write_json(family_directory / "command.json", command)
    started = utc_now()
    profiled = run_probe(
        command["profile_command"],
        timeout=timeout,
        environment=execution_environment(context),
    )
    atomic_write_text(family_directory / "profile.stdout.txt", profiled.stdout)
    atomic_write_text(family_directory / "profile.stderr.txt", profiled.stderr)
    atomic_write_json(
        family_directory / "profile.run.json",
        {
            "started_at": started,
            "finished_at": utc_now(),
            "return_code": profiled.returncode,
            "command": command["profile_command"],
            "replaced_incomplete_reason": retained_reason,
        },
    )
    if profiled.returncode != 0:
        raise CampaignError(
            f"{family_directory}: NCU profile failed with {profiled.returncode}"
        )
    report = family_directory / "profile.ncu-rep"
    if not report.is_file() or report.stat().st_size == 0:
        raise CampaignError(f"{family_directory}: NCU did not create a report")

    imported_at = utc_now()
    imported = run_probe(
        command["import_command"],
        timeout=timeout,
        environment=execution_environment(context),
    )
    atomic_write_text(family_directory / "ncu.csv", imported.stdout)
    atomic_write_text(family_directory / "import.stderr.txt", imported.stderr)
    atomic_write_json(
        family_directory / "import.run.json",
        {
            "started_at": imported_at,
            "finished_at": utc_now(),
            "return_code": imported.returncode,
            "command": command["import_command"],
        },
    )
    if imported.returncode != 0:
        raise CampaignError(
            f"{family_directory}: NCU import failed with {imported.returncode}"
        )

    rows = csv_rows(imported.stdout, str(family_directory / "ncu.csv"))
    summary = validate_rows(rows, cell, repetition, family, context)
    target = validate_target_output(profiled.stdout, cell, repetition, context)
    complete = expected_complete_document(
        family_directory, command, summary, target
    )
    atomic_write_json(family_directory / "complete.json", complete)
    return validate_retained_family(
        family_directory, context, cell, repetition, family
    )


def bundle_document(
    cell: Cell,
    repetition: int,
    families: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(families) != set(FAMILIES):
        raise CampaignError(f"{cell.key} repeat-{repetition}: incomplete family set")
    values = {
        family: dict(document["summary"]["values_by_variant"])
        for family, document in families.items()
    }
    expected_ldgsts = GROUPS * cell.depth
    expected_sectors = source_sectors(cell)
    tolerance = paired_tolerance(cell)
    for variant in ("slow_expensive", "slow_cheap"):
        if values["ldgsts"][variant] != expected_ldgsts:
            raise CampaignError(
                f"{cell.key} repeat-{repetition}: wrong LDGSTS count for {variant}"
            )
        if values["lts_total"][variant] != expected_sectors:
            raise CampaignError(
                f"{cell.key} repeat-{repetition}: attributed LTS total for "
                f"{variant} is {values['lts_total'][variant]}, expected "
                f"{expected_sectors}"
            )
        if values["lts_hit"][variant] <= 0 or values["lts_miss"][variant] <= 0:
            raise CampaignError(
                f"{cell.key} repeat-{repetition}: attributed hit/miss population "
                f"is not positive for {variant}"
            )
    hit_delta = abs(
        values["lts_hit"]["slow_expensive"]
        - values["lts_hit"]["slow_cheap"]
    )
    miss_delta = abs(
        values["lts_miss"]["slow_expensive"]
        - values["lts_miss"]["slow_cheap"]
    )
    if hit_delta > tolerance or miss_delta > tolerance:
        raise CampaignError(
            f"{cell.key} repeat-{repetition}: attributed pair delta exceeds tolerance"
        )
    closure_error = {
        variant: abs(
            values["lts_hit"][variant]
            + values["lts_miss"][variant]
            - values["lts_total"][variant]
        )
        for variant in ("slow_expensive", "slow_cheap")
    }
    if any(error > tolerance for error in closure_error.values()):
        raise CampaignError(
            f"{cell.key} repeat-{repetition}: separate-profile LTS closure "
            f"error {closure_error} exceeds tolerance {tolerance}"
        )
    background = {
        variant: values["dram"][variant] - values["lts_miss"][variant]
        for variant in ("slow_expensive", "slow_cheap")
    }
    return {
        "schema_version": 1,
        "valid": True,
        "cell": cell.key,
        "depth": cell.depth,
        "spacing": cell.spacing,
        "groups": GROUPS,
        "repetition": repetition,
        "order": expected_order(repetition),
        "row_variants": list(row_variants(expected_order(repetition))),
        "families": {
            family: {
                "metric": FAMILY_METRICS[family],
                "values_by_variant": values[family],
                "files": dict(families[family]["files"]),
            }
            for family in FAMILIES
        },
        "attribution": {
            "frontier": "l1tex_local_device_aperture_lts_requested_demand",
            "physical_dram_completion_claimed": False,
            "expected_ldgsts_instructions_per_variant": expected_ldgsts,
            "expected_source_sectors_per_variant": expected_sectors,
            "paired_tolerance": tolerance,
            "absolute_lts_hit_delta": hit_delta,
            "absolute_lts_miss_delta": miss_delta,
            "absolute_hit_plus_miss_minus_total_by_variant": closure_error,
            "all_rows_one_replay_pass": True,
        },
        "non_gating_dram_background": {
            "raw_dram_sectors_by_variant": values["dram"],
            "raw_dram_minus_attributed_lts_misses": background,
        },
    }


def parse_cells(value: str | None) -> list[Cell]:
    if value is None:
        return [Cell(depth, spacing) for depth in DEPTHS for spacing in SPACINGS]
    cells: list[Cell] = []
    for token in value.split(","):
        match = CELL_RE.fullmatch(token.strip())
        if match is None:
            raise CampaignError(f"invalid cell {token!r}; expected e.g. d2s4")
        cell = Cell(int(match.group("depth")), int(match.group("spacing")))
        if cell not in cells:
            cells.append(cell)
    if not cells:
        raise CampaignError("--cells selected no cells")
    return cells


def parse_repetitions(value: str | None) -> list[int]:
    if value is None:
        return list(REPETITIONS)
    repetitions: list[int] = []
    for token in value.split(","):
        try:
            repetition = int(token.strip())
        except ValueError as error:
            raise CampaignError(f"invalid repetition {token!r}") from error
        if repetition not in REPETITIONS:
            raise CampaignError("repetitions must be in the closed range 1..10")
        if repetition not in repetitions:
            repetitions.append(repetition)
    if not repetitions:
        raise CampaignError("--repeats selected no repetitions")
    return repetitions


def parse_families(value: str | None) -> list[str]:
    if value is None:
        return list(FAMILIES)
    families: list[str] = []
    for token in value.split(","):
        family = token.strip()
        if family not in FAMILY_METRICS:
            raise CampaignError(
                f"invalid family {family!r}; choose from {','.join(FAMILIES)}"
            )
        if family not in families:
            families.append(family)
    if not families:
        raise CampaignError("--families selected no metric families")
    return families


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    result.add_argument(
        "--cells", help="comma-separated subset such as d1s0,d2s4 (default: all 8)"
    )
    result.add_argument(
        "--repeats", help="comma-separated subset in 1..10 (default: all 10)"
    )
    result.add_argument(
        "--families",
        help=f"comma-separated subset of {','.join(FAMILIES)} (default: all)",
    )
    result.add_argument(
        "--smoke",
        action="store_true",
        help="select only d1s0 repeat 1, retaining all five one-pass families",
    )
    result.add_argument("--dry-run", action="store_true")
    result.add_argument(
        "--validate-only",
        action="store_true",
        help="validate selected retained families without invoking NCU",
    )
    result.add_argument("--ncu", default=os.environ.get("NCU", "ncu"))
    result.add_argument("--timeout", type=int, default=300)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.timeout < 1:
            raise CampaignError("--timeout must be positive")
        if args.dry_run and args.validate_only:
            raise CampaignError("--dry-run and --validate-only are mutually exclusive")
        if args.smoke and (args.cells or args.repeats or args.families):
            raise CampaignError("--smoke cannot be combined with matrix subset options")
        cells = [Cell(1, 0)] if args.smoke else parse_cells(args.cells)
        repetitions = [1] if args.smoke else parse_repetitions(args.repeats)
        families = list(FAMILIES) if args.smoke else parse_families(args.families)
        root = args.root.resolve()
        context = preflight(args.ncu)

        planned: list[dict[str, Any]] = []
        for cell in cells:
            for repetition in repetitions:
                repeat_directory = (
                    root / "paired_ncu" / cell.key / f"repeat-{repetition}"
                )
                for family in families:
                    directory = repeat_directory / family
                    planned.append(
                        command_document(
                            context, directory, cell, repetition, family
                        )
                    )
        if args.dry_run:
            print(
                json_text(
                    {
                        "dry_run": True,
                        "root": str(root),
                        "selected_cells": [cell.key for cell in cells],
                        "selected_repetitions": repetitions,
                        "selected_families": families,
                        "profile_count": len(planned),
                        "commands": planned,
                    }
                ),
                end="",
            )
            return 0

        total = len(planned)
        progress = 0
        for cell in cells:
            for repetition in repetitions:
                repeat_directory = (
                    root / "paired_ncu" / cell.key / f"repeat-{repetition}"
                )
                completed: dict[str, Mapping[str, Any]] = {}
                for family in families:
                    progress += 1
                    directory = repeat_directory / family
                    if args.validate_only:
                        document = validate_retained_family(
                            directory, context, cell, repetition, family
                        )
                        state = "valid"
                    else:
                        try:
                            document = validate_retained_family(
                                directory, context, cell, repetition, family
                            )
                            state = "existing"
                        except CampaignError:
                            document = execute_family(
                                directory,
                                context,
                                cell,
                                repetition,
                                family,
                                args.timeout,
                            )
                            state = "captured"
                    completed[family] = document
                    print(
                        f"paired-ncu {progress:03d}/{total:03d} {cell.short} "
                        f"r{repetition} {family} ({state})",
                        flush=True,
                    )

                if set(completed) != set(FAMILIES):
                    for family in FAMILIES:
                        if family in completed:
                            continue
                        directory = repeat_directory / family
                        try:
                            completed[family] = validate_retained_family(
                                directory, context, cell, repetition, family
                            )
                        except CampaignError:
                            pass
                if set(completed) == set(FAMILIES):
                    bundle = bundle_document(cell, repetition, completed)
                    if not args.validate_only:
                        atomic_write_json(repeat_directory / "bundle.json", bundle)
                elif args.validate_only and set(families) == set(FAMILIES):
                    raise CampaignError(
                        f"{cell.key} repeat-{repetition}: cannot validate complete bundle"
                    )
        return 0
    except CampaignError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
