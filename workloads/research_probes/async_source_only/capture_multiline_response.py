#!/usr/bin/env python3
"""Capture paired one-pass NCU evidence for async_multiline_response.

The frozen matrix is depth 1/2 by retained lead 0/4/8/16 with ten paired
repetitions (odd AB, even BA).  Every bundle collects six metrics in separate
application-replay reports so cache history is never shared between metrics.
Raw DRAM is retained as non-gating device background; the local-device LTS
total is the exact source-attribution frontier.
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
    REPO_ROOT / "results/captures/sm86_async_multiline_response_20260831_v1"
)
BINARY = HERE / "async_multiline_response"

GROUPS = 64
DEPTHS = (1, 2)
LEADS = (0, 4, 8, 16)
REPETITIONS = tuple(range(1, 11))
VARIANTS = ("span_expensive", "span_cheap")
GPU_INDEX = "0"
EXPECTED_GPU = {
    "index": GPU_INDEX,
    "name": "NVIDIA GeForce RTX 3090 Ti",
    "uuid": "GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c",
    "compute_capability": "8.6",
    "driver": "610.57.04",
}

FAMILY_METRICS = {
    "global_ld": "smsp__inst_executed_op_global_ld.sum",
    "ldgsts": "smsp__inst_executed_op_ldgsts.sum",
    "lts_total": "lts__t_sectors_srcunit_tex_aperture_device_op_read.sum",
    "lts_hit": (
        "lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_hit.sum"
    ),
    "lts_miss": (
        "lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_miss.sum"
    ),
    "dram": "dram__sectors_read.sum",
}
FAMILIES = tuple(FAMILY_METRICS)
CELL_RE = re.compile(r"^d(?P<depth>1|2)l(?P<lead>0|4|8|16)$")


class CampaignError(RuntimeError):
    """Raised for a capture, identity, or evidence-contract failure."""


@dataclass(frozen=True)
class Cell:
    depth: int
    lead: int

    @property
    def short(self) -> str:
        return f"d{self.depth}l{self.lead}"

    @property
    def key(self) -> str:
        return f"async_multiline_response_d{self.depth}_l{self.lead}_g{GROUPS}"


@dataclass(frozen=True)
class Preflight:
    binary: Mapping[str, Any]
    device: Mapping[str, str]
    ncu_path: str
    ncu_version: str
    environment: Mapping[str, str]


@dataclass(frozen=True)
class CapturedRun:
    command: tuple[str, ...]
    configured_timeout_seconds: int
    started_at: str
    finished_at: str
    return_code: int | None
    timed_out: bool
    stdout: str
    stderr: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path, family_directory: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CampaignError(f"artifact is missing: {path}")
    return {
        "path": str(path.relative_to(family_directory)),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


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


def parse_integer(row: Mapping[str, str], name: str, context: str) -> int:
    try:
        value = float(row[name].replace(",", ""))
    except (KeyError, ValueError) as error:
        raise CampaignError(f"{context}: missing/non-numeric {name}") from error
    if not math.isfinite(value) or value < 0 or not value.is_integer():
        raise CampaignError(f"{context}: invalid integer metric {name}={value}")
    return int(value)


def parse_csv_rows(text: str, context: str) -> list[dict[str, str]]:
    try:
        return [
            dict(row)
            for row in csv.DictReader(io.StringIO(text))
            if str(row.get("ID", "")).strip().isdigit()
        ]
    except csv.Error as error:
        raise CampaignError(f"{context}: invalid NCU CSV: {error}") from error


def expected_order(repetition: int) -> str:
    return "ab" if repetition % 2 else "ba"


def row_variants(order: str) -> tuple[str, str]:
    return VARIANTS if order == "ab" else tuple(reversed(VARIANTS))


def expected_instructions(cell: Cell) -> int:
    return GROUPS * cell.depth


def expected_lts_total(cell: Cell) -> int:
    return 56 * GROUPS * cell.depth


def paired_tolerance(cell: Cell) -> int:
    return max(4, math.ceil(0.01 * expected_lts_total(cell)))


def run_command(
    command: Sequence[str],
    *,
    timeout: int,
    environment: Mapping[str, str] | None = None,
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


def timeout_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def run_captured(
    command: Sequence[str],
    *,
    timeout: int,
    environment: Mapping[str, str],
) -> CapturedRun:
    started_at = utc_now()
    try:
        completed = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            env=dict(environment),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return CapturedRun(
            command=tuple(command),
            configured_timeout_seconds=timeout,
            started_at=started_at,
            finished_at=utc_now(),
            return_code=completed.returncode,
            timed_out=False,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except subprocess.TimeoutExpired as error:
        return CapturedRun(
            command=tuple(command),
            configured_timeout_seconds=timeout,
            started_at=started_at,
            finished_at=utc_now(),
            return_code=None,
            timed_out=True,
            stdout=timeout_stream(error.stdout),
            stderr=timeout_stream(error.stderr),
        )
    except OSError as error:
        raise CampaignError(f"cannot execute {command}: {error}") from error


def preflight(root: Path, ncu: str) -> Preflight:
    if not BINARY.is_file() or not os.access(BINARY, os.X_OK):
        raise CampaignError(f"probe binary is missing or not executable: {BINARY}")
    binary_hash = sha256(BINARY)
    static_command = read_json(root / "static/command.json")
    retained_binary = static_command.get("binary")
    if not isinstance(retained_binary, dict) or not isinstance(
        retained_binary.get("sha256"), str
    ):
        raise CampaignError("static command record has no retained binary identity")
    expected_binary_sha256 = str(retained_binary["sha256"])
    if binary_hash != expected_binary_sha256:
        raise CampaignError(
            f"probe binary hash {binary_hash} != retained {expected_binary_sha256}"
        )

    ncu_found = shutil.which(ncu)
    if ncu_found is None:
        raise CampaignError(f"NCU executable is unavailable: {ncu}")
    ncu_path = str(Path(ncu_found).resolve())
    version = run_command([ncu_path, "--version"], timeout=30)
    if version.returncode != 0:
        raise CampaignError(f"NCU version query failed: {version.stderr.strip()}")

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise CampaignError("nvidia-smi is unavailable")
    queried = run_command(
        [
            nvidia_smi,
            "-i",
            GPU_INDEX,
            "--query-gpu=index,name,uuid,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ],
        timeout=30,
    )
    if queried.returncode != 0:
        raise CampaignError(f"GPU identity query failed: {queried.stderr.strip()}")
    try:
        rows = list(csv.reader(io.StringIO(queried.stdout), skipinitialspace=True))
    except csv.Error as error:
        raise CampaignError(f"invalid GPU identity output: {error}") from error
    if len(rows) != 1 or len(rows[0]) != 5:
        raise CampaignError(f"expected one five-field GPU identity row: {rows!r}")
    values = [item.strip() for item in rows[0]]
    device = dict(zip(EXPECTED_GPU, values, strict=True))
    if device != EXPECTED_GPU:
        raise CampaignError(f"GPU identity mismatch: {device} != {EXPECTED_GPU}")

    return Preflight(
        binary={
            "path": str(BINARY),
            "sha256": binary_hash,
            "size_bytes": BINARY.stat().st_size,
        },
        device=device,
        ncu_path=ncu_path,
        ncu_version=(version.stdout + version.stderr).strip(),
        environment={
            "CUDA_VISIBLE_DEVICES": GPU_INDEX,
            "LANG": "C",
            "LC_ALL": "C",
        },
    )


def target_arguments(cell: Cell, repetition: int) -> list[str]:
    return [
        "--groups",
        str(GROUPS),
        "--depth",
        str(cell.depth),
        "--lead-samples",
        str(cell.lead),
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
        "regex:multiline_response_kernel",
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
    timeout: int,
) -> dict[str, Any]:
    order = expected_order(repetition)
    return {
        "schema_version": 1,
        "cell": cell.key,
        "depth": cell.depth,
        "lead_samples": cell.lead,
        "groups": GROUPS,
        "repetition": repetition,
        "order": order,
        "row_variants": list(row_variants(order)),
        "family": family,
        "metric": FAMILY_METRICS[family],
        "timeout_seconds": timeout,
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
        "workload": "async_multiline_response",
        "route": "l2",
        "mode": "pair",
        "order": expected_order(repetition),
        "depth": cell.depth,
        "groups": GROUPS,
        "warmup_groups": 64,
        "lead_clock_samples": cell.lead,
        "source_cohorts_per_copy": 8,
        "aligned_source_cohorts_per_copy": 4,
        "split_source_cohorts_per_copy": 4,
        "source_sectors_per_cohort": 4,
        "payload_bytes_per_cohort": 64,
        "source_step_stride_bytes": 2048,
        "source_step_max_exclusive_byte": 1968,
        "source_step_guard_bytes": 80,
        "scalar_ldg_instructions_per_copy": 1,
        "scalar_ldg_sectors_per_copy": 24,
        "scalar_preload_lines_per_copy": 8,
        "ldgsts_instructions_per_copy": 1,
        "ldgsts_sectors_per_copy": 32,
        "unique_ldgsts_lines_per_copy": 12,
        "destination_lines_per_copy": 20,
        "local_device_lts_lookups_per_copy": 56,
        "compact_destination_cohorts_per_copy": 4,
        "severe_destination_cohorts_per_copy": 4,
        "source_ring_bytes_per_arm": 64 << 20,
        "source_ring_steps_per_arm": 32768,
        "max_admitted_groups": 16320,
        "source_ring_no_wrap": True,
        "source_pool_bytes": 128 << 20,
        "source_pool_alignment_bytes": 128,
        "source_arm_a_offset_bytes": 0,
        "source_arm_b_offset_bytes": 64 << 20,
        "source_pool_single_allocation": True,
        "independent_source_arms": True,
        "source_vector_preserved": True,
        "preload_vector_preserved": True,
        "destination_multiset_preserved": True,
        "result_store_after_intervals": True,
        "device": context.device["name"],
        "compute_capability": context.device["compute_capability"],
        "correct": True,
    }
    for field, expected_value in expected.items():
        if payload.get(field) != expected_value:
            raise CampaignError(
                f"target {field}={payload.get(field)!r}, expected {expected_value!r}"
            )
    for variant in VARIANTS:
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
        raise CampaignError(f"{label}: row order is {ids}, expected 0/1")
    process_ids = [str(row.get("Process ID", "")).strip() for row in rows]
    if not process_ids[0] or process_ids[0] != process_ids[1]:
        raise CampaignError(f"{label}: rows are not one application process")

    template = f"multiline_response_kernel<{cell.depth},{cell.lead}>"
    variants = row_variants(expected_order(repetition))
    metric = FAMILY_METRICS[family]
    values: dict[str, int] = {}
    summaries: list[dict[str, Any]] = []
    for index, (row, variant) in enumerate(zip(rows, variants, strict=True)):
        row_label = f"{label} row {index}"
        kernel = str(row.get("Kernel Name", ""))
        if template not in kernel.replace(" ", ""):
            raise CampaignError(f"{row_label}: wrong kernel {kernel}")
        expected_strings = {
            "Process Name": BINARY.name,
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
                    f"{row_label}: {field}={observed!r}, expected {expected_value!r}"
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
            observed = parse_integer(row, field, row_label)
            if observed != expected_value:
                raise CampaignError(
                    f"{row_label}: {field}={observed}, expected {expected_value}"
                )
        value = parse_integer(row, metric, row_label)
        values[variant] = value
        summaries.append(
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

    expected_value = None
    if family in {"global_ld", "ldgsts"}:
        expected_value = expected_instructions(cell)
    elif family == "lts_total":
        expected_value = expected_lts_total(cell)
    family_gate = {
        "expected_value": expected_value,
        "exact_expected_value": (
            all(value == expected_value for value in values.values())
            if expected_value is not None
            else None
        ),
        "positive_population": (
            all(value > 0 for value in values.values())
            if family in {"lts_hit", "lts_miss"}
            else None
        ),
    }

    return {
        "order": expected_order(repetition),
        "row_variants": list(variants),
        "metric": metric,
        "values_by_variant": values,
        "rows": summaries,
        "scientific_gate": family_gate,
    }


def optional_file_identity(
    path: Path, family_directory: Path
) -> dict[str, Any] | None:
    return file_identity(path, family_directory) if path.is_file() else None


def manifest_inputs(
    command: Mapping[str, Any],
    family_directory: Path,
    *,
    include_report_input: bool,
) -> dict[str, Any]:
    inputs = {
        "binding": {
            key: command[key]
            for key in (
                "cell",
                "depth",
                "lead_samples",
                "groups",
                "repetition",
                "order",
                "row_variants",
                "family",
                "metric",
                "timeout_seconds",
                "target_arguments",
            )
        },
        "binary": dict(command["binary"]),
        "device": dict(command["device"]),
        "ncu": dict(command["ncu"]),
        "environment_overrides": dict(command["environment_overrides"]),
        "command_record": file_identity(
            family_directory / "command.json", family_directory
        ),
    }
    if include_report_input:
        inputs["profile_report"] = optional_file_identity(
            family_directory / "profile.ncu-rep", family_directory
        )
    return inputs


def run_manifest_document(
    kind: str,
    run: CapturedRun,
    command: Mapping[str, Any],
    family_directory: Path,
) -> dict[str, Any]:
    if kind == "ncu_profile":
        expected_command = command["profile_command"]
        stdout_path = family_directory / "profile.stdout.txt"
        stderr_path = family_directory / "profile.stderr.txt"
        include_report_input = False
    elif kind == "ncu_import":
        expected_command = command["import_command"]
        stdout_path = family_directory / "ncu.csv"
        stderr_path = family_directory / "import.stderr.txt"
        include_report_input = True
    else:
        raise CampaignError(f"unsupported run-manifest kind: {kind}")
    if list(run.command) != list(expected_command):
        raise CampaignError(f"{kind}: executed command differs from command record")
    if run.configured_timeout_seconds != command["timeout_seconds"]:
        raise CampaignError(f"{kind}: executed timeout differs from command record")
    return {
        "schema_version": 1,
        "kind": kind,
        "command": list(run.command),
        "configured_timeout_seconds": run.configured_timeout_seconds,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "return_code": run.return_code,
        "timed_out": run.timed_out,
        "stdout": file_identity(stdout_path, family_directory),
        "stderr": file_identity(stderr_path, family_directory),
        "profile_report": optional_file_identity(
            family_directory / "profile.ncu-rep", family_directory
        ),
        "csv": optional_file_identity(
            family_directory / "ncu.csv", family_directory
        ),
        "inputs": manifest_inputs(
            command,
            family_directory,
            include_report_input=include_report_input,
        ),
    }


def parse_manifest_time(value: Any, context: str) -> datetime:
    if not isinstance(value, str):
        raise CampaignError(f"{context}: timestamp is not a string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise CampaignError(f"{context}: invalid timestamp {value!r}") from error
    if parsed.tzinfo is None:
        raise CampaignError(f"{context}: timestamp has no timezone")
    return parsed


def validate_run_manifest(
    path: Path,
    kind: str,
    command: Mapping[str, Any],
    family_directory: Path,
) -> dict[str, Any]:
    observed = read_json(path)
    observed_command = observed.get("command")
    if not isinstance(observed_command, list) or not all(
        isinstance(item, str) for item in observed_command
    ):
        raise CampaignError(f"{path}: command is not a string list")
    observed_timeout = observed.get("configured_timeout_seconds")
    if not isinstance(observed_timeout, int) or isinstance(observed_timeout, bool):
        raise CampaignError(f"{path}: configured timeout is not an integer")
    observed_return_code = observed.get("return_code")
    if not isinstance(observed_return_code, int) or isinstance(
        observed_return_code, bool
    ):
        raise CampaignError(f"{path}: return code is not an integer")
    if observed.get("timed_out") is not False:
        raise CampaignError(f"{path}: retained run timed out")
    started = parse_manifest_time(observed.get("started_at"), f"{path} started_at")
    finished = parse_manifest_time(observed.get("finished_at"), f"{path} finished_at")
    if finished < started:
        raise CampaignError(f"{path}: finish precedes start")
    run = CapturedRun(
        command=tuple(observed_command),
        configured_timeout_seconds=observed_timeout,
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        return_code=observed_return_code,
        timed_out=False,
        stdout="",
        stderr="",
    )
    if run.return_code != 0:
        raise CampaignError(f"{path}: retained run was not successful")
    expected = run_manifest_document(kind, run, command, family_directory)
    if observed != expected:
        raise CampaignError(f"{path}: run manifest or artifact identity mismatch")
    return observed


def complete_document(
    family_directory: Path,
    command: Mapping[str, Any],
    summary: Mapping[str, Any],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    paths = {
        "command.json": family_directory / "command.json",
        "profile.ncu-rep": family_directory / "profile.ncu-rep",
        "ncu.csv": family_directory / "ncu.csv",
        "profile.stdout.txt": family_directory / "profile.stdout.txt",
        "profile.stderr.txt": family_directory / "profile.stderr.txt",
        "import.stderr.txt": family_directory / "import.stderr.txt",
        "profile.run.json": family_directory / "profile.run.json",
        "import.run.json": family_directory / "import.run.json",
    }
    return {
        "schema_version": 1,
        "valid": True,
        "binding": {
            key: command[key]
            for key in (
                "cell",
                "depth",
                "lead_samples",
                "groups",
                "repetition",
                "order",
                "row_variants",
                "family",
                "metric",
                "timeout_seconds",
                "binary",
                "device",
                "ncu",
                "target_arguments",
            )
        },
        "files": {name: sha256(path) for name, path in paths.items()},
        "summary": dict(summary),
        "target_result": dict(target),
    }


def validate_retained_family(
    family_directory: Path,
    context: Preflight,
    cell: Cell,
    repetition: int,
    family: str,
    timeout: int,
) -> dict[str, Any]:
    required = {
        name: family_directory / name
        for name in (
            "command.json",
            "profile.ncu-rep",
            "ncu.csv",
            "profile.stdout.txt",
            "profile.stderr.txt",
            "import.stderr.txt",
            "profile.run.json",
            "import.run.json",
            "complete.json",
        )
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise CampaignError(f"{family_directory}: missing {missing}")
    if required["profile.ncu-rep"].stat().st_size == 0:
        raise CampaignError(f"{family_directory}: empty NCU report")

    expected_command = command_document(
        context, family_directory, cell, repetition, family, timeout
    )
    command = read_json(required["command.json"])
    if command != expected_command:
        raise CampaignError(f"{family_directory}: retained command binding mismatch")
    validate_run_manifest(
        required["profile.run.json"],
        "ncu_profile",
        command,
        family_directory,
    )
    validate_run_manifest(
        required["import.run.json"],
        "ncu_import",
        command,
        family_directory,
    )
    rows = parse_csv_rows(
        required["ncu.csv"].read_text(encoding="utf-8"), str(required["ncu.csv"])
    )
    summary = validate_rows(rows, cell, repetition, family, context)
    target = validate_target_output(
        required["profile.stdout.txt"].read_text(encoding="utf-8"),
        cell,
        repetition,
        context,
    )
    expected_complete = complete_document(family_directory, command, summary, target)
    if read_json(required["complete.json"]) != expected_complete:
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
            family_directory, context, cell, repetition, family, timeout
        )
    except CampaignError:
        pass

    family_directory.mkdir(parents=True, exist_ok=True)
    command = command_document(
        context, family_directory, cell, repetition, family, timeout
    )
    atomic_write_json(family_directory / "command.json", command)
    profiled = run_captured(
        command["profile_command"],
        timeout=timeout,
        environment=execution_environment(context),
    )
    atomic_write_text(family_directory / "profile.stdout.txt", profiled.stdout)
    atomic_write_text(family_directory / "profile.stderr.txt", profiled.stderr)
    atomic_write_json(
        family_directory / "profile.run.json",
        run_manifest_document(
            "ncu_profile", profiled, command, family_directory
        ),
    )
    if profiled.timed_out or profiled.return_code != 0:
        raise CampaignError(
            f"{family_directory}: NCU profile failed: "
            f"return_code={profiled.return_code}, timed_out={profiled.timed_out}"
        )
    report = family_directory / "profile.ncu-rep"
    if not report.is_file() or report.stat().st_size == 0:
        raise CampaignError(f"{family_directory}: NCU report was not created")

    imported = run_captured(
        command["import_command"],
        timeout=timeout,
        environment=execution_environment(context),
    )
    atomic_write_text(family_directory / "ncu.csv", imported.stdout)
    atomic_write_text(family_directory / "import.stderr.txt", imported.stderr)
    atomic_write_json(
        family_directory / "profile.run.json",
        run_manifest_document(
            "ncu_profile", profiled, command, family_directory
        ),
    )
    atomic_write_json(
        family_directory / "import.run.json",
        run_manifest_document(
            "ncu_import", imported, command, family_directory
        ),
    )
    if imported.timed_out or imported.return_code != 0:
        raise CampaignError(
            f"{family_directory}: NCU import failed: "
            f"return_code={imported.return_code}, timed_out={imported.timed_out}"
        )
    if sha256(BINARY) != context.binary["sha256"]:
        raise CampaignError(f"{family_directory}: probe binary changed during capture")

    rows = parse_csv_rows(imported.stdout, str(family_directory / "ncu.csv"))
    summary = validate_rows(rows, cell, repetition, family, context)
    target = validate_target_output(profiled.stdout, cell, repetition, context)
    complete = complete_document(family_directory, command, summary, target)
    atomic_write_json(family_directory / "complete.json", complete)
    return validate_retained_family(
        family_directory, context, cell, repetition, family, timeout
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
    expected_inst = expected_instructions(cell)
    expected_total = expected_lts_total(cell)
    tolerance = paired_tolerance(cell)
    hit_delta = abs(values["lts_hit"][VARIANTS[0]] - values["lts_hit"][VARIANTS[1]])
    miss_delta = abs(
        values["lts_miss"][VARIANTS[0]] - values["lts_miss"][VARIANTS[1]]
    )
    closure = {
        variant: abs(
            values["lts_hit"][variant]
            + values["lts_miss"][variant]
            - values["lts_total"][variant]
        )
        for variant in VARIANTS
    }
    gates = {
        "global_ld_exact": all(
            values["global_ld"][variant] == expected_inst for variant in VARIANTS
        ),
        "ldgsts_exact": all(
            values["ldgsts"][variant] == expected_inst for variant in VARIANTS
        ),
        "lts_total_exact": all(
            values["lts_total"][variant] == expected_total for variant in VARIANTS
        ),
        "lts_hit_positive": all(
            values["lts_hit"][variant] > 0 for variant in VARIANTS
        ),
        "lts_miss_positive": all(
            values["lts_miss"][variant] > 0 for variant in VARIANTS
        ),
        "paired_lts_hit_within_tolerance": hit_delta <= tolerance,
        "paired_lts_miss_within_tolerance": miss_delta <= tolerance,
        "separate_profile_closure_within_tolerance": all(
            value <= tolerance for value in closure.values()
        ),
        "all_rows_one_replay_pass": True,
    }

    background = {
        variant: values["dram"][variant] - values["lts_miss"][variant]
        for variant in VARIANTS
    }
    return {
        "schema_version": 1,
        "valid": all(gates.values()),
        "cell": cell.key,
        "depth": cell.depth,
        "lead_samples": cell.lead,
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
            "expected_global_ld_instructions_per_variant": expected_inst,
            "expected_ldgsts_instructions_per_variant": expected_inst,
            "expected_lts_lookup_sectors_per_variant": expected_total,
            "paired_tolerance": tolerance,
            "absolute_lts_hit_delta": hit_delta,
            "absolute_lts_miss_delta": miss_delta,
            "absolute_hit_plus_miss_minus_total_by_variant": closure,
            "all_rows_one_replay_pass": True,
            "gates": gates,
        },
        "non_gating_dram_background": {
            "raw_dram_sectors_by_variant": values["dram"],
            "raw_dram_minus_attributed_lts_misses": background,
        },
    }


def parse_cells(value: str | None) -> list[Cell]:
    if value is None:
        return [Cell(depth, lead) for depth in DEPTHS for lead in LEADS]
    cells: list[Cell] = []
    for token in value.split(","):
        match = CELL_RE.fullmatch(token.strip())
        if match is None:
            raise CampaignError(f"invalid cell {token!r}; expected e.g. d1l16")
        cell = Cell(int(match.group("depth")), int(match.group("lead")))
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
            raise CampaignError("repetitions must be in 1..10")
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
        raise CampaignError("--families selected no families")
    return families


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    result.add_argument(
        "--cells", help="comma-separated subset such as d1l16,d2l4 (default: all 8)"
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
        help="select only depth 1, lead 16, repeat 1 with all six families",
    )
    result.add_argument("--dry-run", action="store_true")
    result.add_argument(
        "--validate-only",
        action="store_true",
        help="validate retained selected families without invoking NCU",
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
            raise CampaignError("--smoke cannot be combined with subset options")
        cells = [Cell(1, 16)] if args.smoke else parse_cells(args.cells)
        repetitions = [1] if args.smoke else parse_repetitions(args.repeats)
        families = list(FAMILIES) if args.smoke else parse_families(args.families)
        root = args.root.resolve()
        context = preflight(root, args.ncu)

        planned: list[dict[str, Any]] = []
        for cell in cells:
            for repetition in repetitions:
                repeat_directory = (
                    root / "paired_ncu" / cell.key / f"repeat-{repetition}"
                )
                for family in families:
                    planned.append(
                        command_document(
                            context,
                            repeat_directory / family,
                            cell,
                            repetition,
                            family,
                            args.timeout,
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
                            directory,
                            context,
                            cell,
                            repetition,
                            family,
                            args.timeout,
                        )
                        state = "valid"
                    else:
                        try:
                            document = validate_retained_family(
                                directory,
                                context,
                                cell,
                                repetition,
                                family,
                                args.timeout,
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
                        f"multiline-ncu {progress:03d}/{total:03d} {cell.short} "
                        f"r{repetition} {family} ({state})",
                        flush=True,
                    )

                if set(completed) != set(FAMILIES):
                    for family in FAMILIES:
                        if family in completed:
                            continue
                        try:
                            completed[family] = validate_retained_family(
                                repeat_directory / family,
                                context,
                                cell,
                                repetition,
                                family,
                                args.timeout,
                            )
                        except CampaignError:
                            pass
                if set(completed) == set(FAMILIES):
                    bundle = bundle_document(cell, repetition, completed)
                    bundle_path = repeat_directory / "bundle.json"
                    if args.validate_only:
                        if not bundle_path.is_file():
                            raise CampaignError(f"missing retained bundle: {bundle_path}")
                        if read_json(bundle_path) != bundle:
                            raise CampaignError(
                                f"{bundle_path}: retained bundle does not match families"
                            )
                    else:
                        atomic_write_json(repeat_directory / "bundle.json", bundle)
                elif args.validate_only and set(families) == set(FAMILIES):
                    raise CampaignError(
                        f"{cell.key} repeat-{repetition}: incomplete bundle"
                    )
        return 0
    except CampaignError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
