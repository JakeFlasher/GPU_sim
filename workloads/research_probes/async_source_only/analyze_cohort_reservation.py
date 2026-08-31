#!/usr/bin/env python3
"""Reduce the SM86 async-copy cohort reservation-exposure milestone.

The reducer first validates the fresh, same-application paired NCU campaign.
Only the L1TEX-source/local-device-aperture LTS hit and miss populations admit
the source marginals; device-wide DRAM reads remain visible diagnostics.  Live
wait labels from the retained v2 campaign are not read until every NCU bundle
passes.  A complete identification or transfer no-go is a scientific result
and exits successfully.  Missing or corrupt evidence exits nonzero.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = (
    REPO_ROOT / "results/captures/sm86_async_cohort_reservation_20260831_v1"
)
DEFAULT_BASELINE_ROOT = (
    REPO_ROOT / "results/captures/sm86_async_cohort_pairing_20260830_v2"
)
PROBE_BINARY = (
    REPO_ROOT / "workloads/research_probes/async_source_only/async_cohort_pairing"
)
EXPECTED_BINARY_SHA256 = (
    "dded4cb03868f5cc757f86103d3ec45d4c8adf66c580b39ac90e3c4034f94f4f"
)

GROUPS = 64
DEPTHS = (1, 2, 4, 8)
SPACINGS = (0, 4)
REPETITIONS = tuple(range(1, 11))
VARIANTS = ("slow_expensive", "slow_cheap")
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

EXPECTED_DEVICE = {
    "index": "0",
    "name": "NVIDIA GeForce RTX 3090 Ti",
    "uuid": "GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c",
    "compute_capability": "8.6",
    "driver": "610.57.04",
}
EXPECTED_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "0",
    "LANG": "C",
    "LC_ALL": "C",
}
EXPECTED_NCU_VERSION = "Version 2026.1.0.0 (build 37166530) (public-release)"

SCHEMA18_FIELDS = (
    "cohort_overlap_groups_observed",
    "cohort_response_envelope_cycles",
    "cohort_topology_prearrival_hidden_cycles",
    "cohort_topology_stream_hidden_cycles",
    "cohort_topology_residual_exposure_cycles",
)

SPACING_LAYOUTS = ("compact", "conflict", "skew", "severe")
IDENTIFICATION_SPECS = ((1, 0), (2, 0))
HOLDOUT_SPECS = ((4, 0), (8, 0), (1, 4), (2, 4), (4, 4), (8, 4))
MAE_LIMIT = 1.0
MAX_ERROR_LIMIT = 2.0


class AnalysisError(RuntimeError):
    """An evidence artifact exists but violates its frozen contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise AnalysisError(f"{path}: expected one JSON object")
    return value


def integer(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise AnalysisError(f"{context}: boolean is not an integer metric")
    if isinstance(value, int):
        result = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise AnalysisError(f"{context}: expected an integer, got {value!r}")
        result = int(value)
    elif isinstance(value, str):
        try:
            parsed = float(value.replace(",", ""))
        except ValueError as error:
            raise AnalysisError(f"{context}: non-numeric value {value!r}") from error
        if not math.isfinite(parsed) or not parsed.is_integer():
            raise AnalysisError(f"{context}: expected an integer, got {value!r}")
        result = int(parsed)
    else:
        raise AnalysisError(f"{context}: expected an integer, got {value!r}")
    if result < 0:
        raise AnalysisError(f"{context}: negative value {result}")
    return result


def expected_order(repetition: int) -> str:
    return "ab" if repetition % 2 else "ba"


def row_variants(order: str) -> tuple[str, str]:
    if order == "ab":
        return ("slow_expensive", "slow_cheap")
    if order == "ba":
        return ("slow_cheap", "slow_expensive")
    raise AnalysisError(f"unsupported paired order {order!r}")


def pair_cell_name(depth: int, spacing: int) -> str:
    return f"async_cohort_pairing_d{depth}_s{spacing}_g{GROUPS}"


def endpoint_name(variant: str, depth: int, spacing: int) -> str:
    return f"async_cohort_pairing_{variant}_d{depth}_s{spacing}_g{GROUPS}"


def spacing_cell_name(route: str, layout: str, depth: int, spacing: int) -> str:
    return f"async_issue_spacing_{route}_{layout}_d{depth}_s{spacing}_g{GROUPS}"


def pair_specs() -> list[tuple[int, int]]:
    return [(depth, spacing) for depth in DEPTHS for spacing in SPACINGS]


def endpoint_specs() -> list[tuple[str, int, int]]:
    return [
        (variant, depth, spacing)
        for variant in VARIANTS
        for depth, spacing in pair_specs()
    ]


def spacing_specs() -> list[tuple[str, str, int, int]]:
    specs = [
        ("l2", layout, 2, spacing)
        for layout in SPACING_LAYOUTS
        for spacing in (0, 1, 4, 16)
    ]
    specs.extend(
        ("l2", layout, 8, spacing)
        for layout in ("compact", "skew")
        for spacing in (0, 1, 4)
    )
    return specs


def expected_source_sectors(depth: int) -> int:
    return GROUPS * depth * 32


def paired_tolerance(depth: int) -> int:
    return max(4, math.ceil(0.01 * expected_source_sectors(depth)))


def expected_target_arguments(depth: int, spacing: int, repetition: int) -> list[str]:
    return [
        "--groups",
        str(GROUPS),
        "--depth",
        str(depth),
        "--spacing-samples",
        str(spacing),
        "--order",
        expected_order(repetition),
        "--markers",
        "on",
    ]


def parse_csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        text = path.read_text(encoding="utf-8")
        rows = [
            dict(row)
            for row in csv.DictReader(io.StringIO(text))
            if str(row.get("ID", "")).strip().isdigit()
        ]
    except (OSError, csv.Error) as error:
        raise AnalysisError(f"cannot parse {path}: {error}") from error
    return rows


def target_result(path: Path, depth: int, spacing: int, repetition: int) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as error:
        raise AnalysisError(f"cannot read {path}: {error}") from error
    for line in lines:
        if not line.lstrip().startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise AnalysisError(f"{path}: invalid target JSON: {error}") from error
        if isinstance(value, dict):
            payloads.append(value)
    if len(payloads) != 1:
        raise AnalysisError(f"{path}: expected one target result, got {len(payloads)}")
    payload = payloads[0]
    expected = {
        "workload": "async_cohort_pairing",
        "route": "l2",
        "mode": "pair",
        "order": expected_order(repetition),
        "depth": depth,
        "groups": GROUPS,
        "warmup_groups": 64,
        "spacing_clock_samples_per_edge": spacing,
        "source_cohorts_per_copy": 8,
        "hot_source_cohorts_per_copy": 4,
        "cold_source_cohorts_per_copy": 4,
        "source_sectors_per_cohort": 4,
        "payload_bytes_per_cohort": 64,
        "compact_destination_cohorts_per_copy": 4,
        "severe_destination_cohorts_per_copy": 4,
        "cold_ring_bytes_per_arm": 128 << 20,
        "cold_step_stride_bytes": 1024,
        "max_admitted_groups": 16320,
        "cold_ring_no_wrap": True,
        "source_pool_bytes": 268443648,
        "source_pool_alignment_bytes": 4096,
        "hot_source_offset_bytes": 0,
        "cold_source_a_offset_bytes": 8192,
        "cold_source_b_offset_bytes": 134225920,
        "source_pool_single_allocation": True,
        "independent_cold_regions": True,
        "source_vector_preserved": True,
        "destination_multiset_preserved": True,
        "static_ldgsts_positions_preserved": True,
        "result_store_after_intervals": True,
        "device": EXPECTED_DEVICE["name"],
        "compute_capability": EXPECTED_DEVICE["compute_capability"],
        "correct": True,
    }
    mismatches = {
        field: {"observed": payload.get(field), "expected": expected_value}
        for field, expected_value in expected.items()
        if payload.get(field) != expected_value
    }
    if mismatches:
        raise AnalysisError(f"{path}: target semantic binding drift: {mismatches}")
    for variant in VARIANTS:
        result = payload.get(variant)
        if not isinstance(result, dict):
            raise AnalysisError(f"{path}: missing result for {variant}")
        if result.get("correct") is not True or result.get("mismatches") != 0:
            raise AnalysisError(f"{path}: incorrect target result for {variant}")
        if result.get("issue_cycles") + result.get("wait_cycles") != result.get(
            "total_cycles"
        ):
            raise AnalysisError(f"{path}: target intervals do not sum for {variant}")
    return payload


def current_binary_report() -> dict[str, Any]:
    if not PROBE_BINARY.is_file():
        raise FileNotFoundError(str(PROBE_BINARY))
    observed = sha256(PROBE_BINARY)
    if observed != EXPECTED_BINARY_SHA256:
        raise AnalysisError(
            f"probe binary identity mismatch: {observed} != {EXPECTED_BINARY_SHA256}"
        )
    return {
        "path": str(PROBE_BINARY),
        "size_bytes": PROBE_BINARY.stat().st_size,
        "sha256": observed,
    }


def validate_command(
    command: Mapping[str, Any],
    family_directory: Path,
    family: str,
    depth: int,
    spacing: int,
    repetition: int,
    binary: Mapping[str, Any],
) -> None:
    order = expected_order(repetition)
    cell = pair_cell_name(depth, spacing)
    variants = list(row_variants(order))
    expected_fields: dict[str, Any] = {
        "schema_version": 1,
        "cell": cell,
        "depth": depth,
        "spacing": spacing,
        "groups": GROUPS,
        "repetition": repetition,
        "order": order,
        "row_variants": variants,
        "family": family,
        "metric": FAMILY_METRICS[family],
        "binary": dict(binary),
        "device": EXPECTED_DEVICE,
        "environment_overrides": EXPECTED_ENVIRONMENT,
        "target_arguments": expected_target_arguments(depth, spacing, repetition),
    }
    mismatches = {
        field: {"observed": command.get(field), "expected": expected}
        for field, expected in expected_fields.items()
        if command.get(field) != expected
    }
    if mismatches:
        raise AnalysisError(
            f"{family_directory}/command.json: frozen binding mismatch: {mismatches}"
        )
    ncu = command.get("ncu")
    if not isinstance(ncu, dict):
        raise AnalysisError(f"{family_directory}/command.json: missing NCU identity")
    ncu_path = ncu.get("path")
    version = ncu.get("version")
    if not isinstance(ncu_path, str) or not ncu_path.endswith("/ncu"):
        raise AnalysisError(f"{family_directory}: invalid NCU path {ncu_path!r}")
    if not isinstance(version, str) or EXPECTED_NCU_VERSION not in version:
        raise AnalysisError(f"{family_directory}: wrong NCU version {version!r}")

    expected_profile = [
        ncu_path,
        "--devices",
        "0",
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
        str(PROBE_BINARY),
        *expected_target_arguments(depth, spacing, repetition),
    ]
    expected_import = [
        ncu_path,
        "--import",
        str(family_directory / "profile.ncu-rep"),
        "--page",
        "raw",
        "--csv",
        "--print-units",
        "base",
        "--print-fp",
    ]
    if command.get("profile_command") != expected_profile:
        raise AnalysisError(f"{family_directory}: profile command drift")
    if command.get("import_command") != expected_import:
        raise AnalysisError(f"{family_directory}: import command drift")


def validate_family(
    family_directory: Path,
    family: str,
    depth: int,
    spacing: int,
    repetition: int,
    binary: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        name: family_directory / name
        for name in (
            "command.json",
            "complete.json",
            "ncu.csv",
            "profile.ncu-rep",
            "profile.stdout.txt",
        )
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    if required["profile.ncu-rep"].stat().st_size == 0:
        raise AnalysisError(f"{required['profile.ncu-rep']}: empty NCU report")

    command = read_json(required["command.json"])
    complete = read_json(required["complete.json"])
    validate_command(
        command, family_directory, family, depth, spacing, repetition, binary
    )
    if complete.get("schema_version") != 1 or complete.get("valid") is not True:
        raise AnalysisError(f"{required['complete.json']}: invalid completion state")

    expected_files = {
        "command.json": sha256(required["command.json"]),
        "profile.ncu-rep": sha256(required["profile.ncu-rep"]),
        "ncu.csv": sha256(required["ncu.csv"]),
        "profile.stdout.txt": sha256(required["profile.stdout.txt"]),
    }
    if complete.get("files") != expected_files:
        raise AnalysisError(f"{required['complete.json']}: retained file identity mismatch")

    binding_fields = (
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
    expected_binding = {field: command[field] for field in binding_fields}
    if complete.get("binding") != expected_binding:
        raise AnalysisError(f"{required['complete.json']}: command binding mismatch")

    rows = parse_csv_rows(required["ncu.csv"])
    if len(rows) != 2:
        raise AnalysisError(f"{required['ncu.csv']}: expected two rows, got {len(rows)}")
    ids = [str(row.get("ID", "")).strip() for row in rows]
    if ids != ["0", "1"]:
        raise AnalysisError(f"{required['ncu.csv']}: row order is {ids}, expected 0/1")
    process_ids = [str(row.get("Process ID", "")).strip() for row in rows]
    if not process_ids[0] or process_ids[0] != process_ids[1]:
        raise AnalysisError(f"{required['ncu.csv']}: rows are not one application")

    metric = FAMILY_METRICS[family]
    variants = row_variants(expected_order(repetition))
    values: dict[str, int] = {}
    summaries: list[dict[str, Any]] = []
    for index, (row, variant) in enumerate(zip(rows, variants, strict=True)):
        label = f"{required['ncu.csv']} row {index}"
        kernel = str(row.get("Kernel Name", ""))
        template = f"cohort_pairing_kernel<{depth},{spacing}>"
        if template not in kernel.replace(" ", ""):
            raise AnalysisError(f"{label}: wrong kernel {kernel!r}")
        expected_strings = {
            "Process Name": "async_cohort_pairing",
            "Device": "0",
            "CC": EXPECTED_DEVICE["compute_capability"],
            "Grid Size": "(1, 1, 1)",
            "Block Size": "(32, 1, 1)",
            "device__attribute_display_name": EXPECTED_DEVICE["name"],
        }
        for field, expected in expected_strings.items():
            observed = str(row.get(field, "")).strip()
            if observed != expected:
                raise AnalysisError(
                    f"{label}: {field}={observed!r}, expected {expected!r}"
                )
        expected_integers = {
            "device__attribute_device_index": 0,
            "device__attribute_compute_capability_major": 8,
            "device__attribute_compute_capability_minor": 6,
            "launch__grid_size": 1,
            "launch__block_size": 32,
            "profiler__replayer_passes": 1,
        }
        for field, expected in expected_integers.items():
            observed = integer(row.get(field), f"{label}.{field}")
            if observed != expected:
                raise AnalysisError(
                    f"{label}: {field}={observed}, expected {expected}"
                )
        value = integer(row.get(metric), f"{label}.{metric}")
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
                "device_clock_rate_khz": integer(
                    row.get("device__attribute_clock_rate"),
                    f"{label}.device__attribute_clock_rate",
                ),
                "memory_clock_rate_khz": integer(
                    row.get("device__attribute_memory_clock_rate"),
                    f"{label}.device__attribute_memory_clock_rate",
                ),
            }
        )

    if family == "ldgsts":
        expected = GROUPS * depth
        if any(value != expected for value in values.values()):
            raise AnalysisError(
                f"{family_directory}: LDGSTS values {values} != {expected}"
            )
    elif family in {"lts_hit", "lts_miss"}:
        if any(value <= 0 for value in values.values()):
            raise AnalysisError(f"{family_directory}: zero attributable LTS population")
        difference = abs(values[VARIANTS[0]] - values[VARIANTS[1]])
        if difference > paired_tolerance(depth):
            raise AnalysisError(
                f"{family_directory}: paired delta {difference} exceeds "
                f"tolerance {paired_tolerance(depth)}"
            )
    elif family == "lts_total":
        expected = expected_source_sectors(depth)
        if any(value != expected for value in values.values()):
            raise AnalysisError(
                f"{family_directory}: LTS totals {values} != {expected}"
            )

    parsed_target = target_result(
        required["profile.stdout.txt"], depth, spacing, repetition
    )
    expected_runner_summary = {
        "order": expected_order(repetition),
        "row_variants": list(variants),
        "metric": metric,
        "values_by_variant": values,
        "rows": [
            {
                key: row[key]
                for key in (
                    "id",
                    "variant",
                    "process_id",
                    "kernel",
                    "context",
                    "stream",
                    "metric_value",
                    "profiler_replayer_passes",
                )
            }
            for row in summaries
        ],
    }
    if complete.get("summary") != expected_runner_summary:
        raise AnalysisError(f"{required['complete.json']}: row summary mismatch")
    if complete.get("target_result") != parsed_target:
        raise AnalysisError(f"{required['complete.json']}: target result mismatch")

    return {
        "family": family,
        "metric": metric,
        "values_by_variant": values,
        "rows": summaries,
        "files": expected_files,
        "ncu": command["ncu"],
    }


def expected_bundle_document(
    depth: int,
    spacing: int,
    repetition: int,
    families: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    values = {
        family: dict(families[family]["values_by_variant"]) for family in FAMILIES
    }
    expected_ldgsts = GROUPS * depth
    expected_sectors = expected_source_sectors(depth)
    tolerance = paired_tolerance(depth)
    hit_delta = abs(values["lts_hit"][VARIANTS[0]] - values["lts_hit"][VARIANTS[1]])
    miss_delta = abs(
        values["lts_miss"][VARIANTS[0]] - values["lts_miss"][VARIANTS[1]]
    )
    closure_error = {
        variant: abs(
            values["lts_hit"][variant]
            + values["lts_miss"][variant]
            - values["lts_total"][variant]
        )
        for variant in VARIANTS
    }
    background = {
        variant: values["dram"][variant] - values["lts_miss"][variant]
        for variant in VARIANTS
    }
    return {
        "schema_version": 1,
        "valid": True,
        "cell": pair_cell_name(depth, spacing),
        "depth": depth,
        "spacing": spacing,
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


def validate_bundle(
    root: Path,
    depth: int,
    spacing: int,
    repetition: int,
    binary: Mapping[str, Any],
) -> dict[str, Any]:
    repeat_directory = (
        root
        / "paired_ncu"
        / pair_cell_name(depth, spacing)
        / f"repeat-{repetition}"
    )
    bundle_path = repeat_directory / "bundle.json"
    if not bundle_path.is_file():
        raise FileNotFoundError(str(bundle_path))
    families = {
        family: validate_family(
            repeat_directory / family,
            family,
            depth,
            spacing,
            repetition,
            binary,
        )
        for family in FAMILIES
    }
    expected = expected_bundle_document(depth, spacing, repetition, families)
    observed = read_json(bundle_path)
    if observed != expected:
        raise AnalysisError(f"{bundle_path}: bundle does not match raw family reports")

    values = {
        family: families[family]["values_by_variant"] for family in FAMILIES
    }
    expected_sectors = expected_source_sectors(depth)
    for variant in VARIANTS:
        if values["lts_total"][variant] != expected_sectors:
            raise AnalysisError(
                f"{bundle_path}: {variant} total={values['lts_total'][variant]}, "
                f"expected {expected_sectors}"
            )
        if values["lts_hit"][variant] <= 0 or values["lts_miss"][variant] <= 0:
            raise AnalysisError(f"{bundle_path}: nonpositive hit/miss for {variant}")
        closure = abs(
            values["lts_hit"][variant]
            + values["lts_miss"][variant]
            - values["lts_total"][variant]
        )
        if closure > paired_tolerance(depth):
            raise AnalysisError(
                f"{bundle_path}: {variant} separate-profile closure {closure} "
                f"exceeds {paired_tolerance(depth)}"
            )
    return {
        "cell": pair_cell_name(depth, spacing),
        "depth": depth,
        "spacing": spacing,
        "repetition": repetition,
        "order": expected_order(repetition),
        "row_variants": list(row_variants(expected_order(repetition))),
        "paired_tolerance": paired_tolerance(depth),
        "absolute_lts_hit_delta": expected["attribution"]["absolute_lts_hit_delta"],
        "absolute_lts_miss_delta": expected["attribution"]["absolute_lts_miss_delta"],
        "absolute_hit_plus_miss_minus_total_by_variant": expected["attribution"][
            "absolute_hit_plus_miss_minus_total_by_variant"
        ],
        "values_by_family": values,
        "non_gating_dram_background": expected["non_gating_dram_background"],
        "families": families,
        "bundle_sha256": sha256(bundle_path),
    }


def collect_bundles(
    root: Path, binary: Mapping[str, Any] | None
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    bundles: list[dict[str, Any]] = []
    missing: list[str] = []
    errors: list[str] = []
    if binary is None:
        return bundles, missing, errors
    for depth, spacing in pair_specs():
        for repetition in REPETITIONS:
            try:
                bundles.append(
                    validate_bundle(root, depth, spacing, repetition, binary)
                )
            except FileNotFoundError as error:
                missing.append(str(error))
            except (AnalysisError, OSError, KeyError, TypeError, ValueError) as error:
                errors.append(str(error))
    return bundles, missing, errors


def single_launch(document: Mapping[str, Any], context: str) -> dict[str, Any]:
    launches = document.get("launches")
    if not isinstance(launches, list) or len(launches) != 1:
        raise AnalysisError(f"{context}: expected one simulator launch")
    launch = launches[0]
    if not isinstance(launch, dict):
        raise AnalysisError(f"{context}: launch is not an object")
    return launch


def schema_observation(
    document: Mapping[str, Any], expected_schema: int, context: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    launch = single_launch(document, context)
    timing = launch.get("timing")
    if not isinstance(timing, dict):
        raise AnalysisError(f"{context}: missing timing object")
    readiness = timing.get("readiness")
    if not isinstance(readiness, dict) or readiness.get("schema_version") != expected_schema:
        raise AnalysisError(f"{context}: expected readiness schema {expected_schema}")
    observation = readiness.get("bypass_async_copies")
    if not isinstance(observation, dict):
        raise AnalysisError(f"{context}: missing bypass observation")
    return launch, readiness, observation


def normalize_schema18(document: Mapping[str, Any], context: str) -> dict[str, Any]:
    normalized = deepcopy(document)
    _, readiness, observation = schema_observation(normalized, 18, context)
    missing = [field for field in SCHEMA18_FIELDS if field not in observation]
    if missing:
        raise AnalysisError(f"{context}: schema-18 fields missing: {missing}")
    for field in SCHEMA18_FIELDS:
        observation.pop(field)
    readiness["schema_version"] = 17
    return normalized


def validate_replay_cell(
    current_base: Path,
    baseline_base: Path,
    cell: str,
    kind: str,
) -> dict[str, Any]:
    current_paths = [current_base / cell / f"replay-{index}.json" for index in (1, 2)]
    baseline_paths = [baseline_base / cell / f"replay-{index}.json" for index in (1, 2)]
    absent = [str(path) for path in (*current_paths, *baseline_paths) if not path.is_file()]
    if absent:
        raise FileNotFoundError(", ".join(absent))
    if current_paths[0].read_bytes() != current_paths[1].read_bytes():
        raise AnalysisError(f"{cell}: schema-18 replay twins differ")
    if baseline_paths[0].read_bytes() != baseline_paths[1].read_bytes():
        raise AnalysisError(f"{cell}: retained schema-17 replay twins differ")

    current = read_json(current_paths[0])
    baseline = read_json(baseline_paths[0])
    launch, _, observation = schema_observation(current, 18, cell)
    baseline_launch, _, baseline_observation = schema_observation(baseline, 17, cell)
    if any(field in baseline_observation for field in SCHEMA18_FIELDS):
        raise AnalysisError(f"{cell}: schema-18 field leaked into schema-17 baseline")
    values = {
        field: integer(observation.get(field), f"{cell}.{field}")
        for field in SCHEMA18_FIELDS
    }
    if values["cohort_overlap_groups_observed"] != GROUPS:
        raise AnalysisError(
            f"{cell}: overlap groups {values['cohort_overlap_groups_observed']} != {GROUPS}"
        )
    topology_penalty = integer(
        observation.get("cohort_topology_penalty_cycles"),
        f"{cell}.cohort_topology_penalty_cycles",
    )
    if (
        values["cohort_topology_prearrival_hidden_cycles"]
        + values["cohort_topology_stream_hidden_cycles"]
        > topology_penalty
    ):
        raise AnalysisError(f"{cell}: hidden topology exceeds observed topology work")
    if values["cohort_topology_residual_exposure_cycles"] > topology_penalty:
        raise AnalysisError(f"{cell}: residual exposure exceeds topology work")

    normalized = normalize_schema18(current, cell)
    if normalized != baseline:
        raise AnalysisError(
            f"{cell}: schema-18 replay differs from schema-17 after normalization"
        )
    tails = launch.get("counters", {}).get("unmodeled_tails", {})
    baseline_tails = baseline_launch.get("counters", {}).get("unmodeled_tails", {})
    if tails.get("async_copy_shared_visibility") is not True:
        raise AnalysisError(f"{cell}: async-copy visibility tail was removed")
    if baseline_tails.get("async_copy_shared_visibility") is not True:
        raise AnalysisError(f"{cell}: schema-17 baseline lacks visibility tail")

    return {
        "cell": cell,
        "kind": kind,
        "schema_version": 18,
        **values,
        "schema18_selected_with_zero_residual": (
            values["cohort_topology_residual_exposure_cycles"] == 0
        ),
        "production_timing_neutral": True,
        "typed_visibility_tail_retained": True,
        "normalized_schema17_equal": True,
        "replay_sha256": sha256(current_paths[0]),
        "baseline_sha256": sha256(baseline_paths[0]),
    }


def collect_pairing_replays(
    root: Path, baseline_root: Path
) -> tuple[dict[tuple[str, int, int], dict[str, Any]], list[str], list[str]]:
    cells: dict[tuple[str, int, int], dict[str, Any]] = {}
    missing: list[str] = []
    errors: list[str] = []
    for spec in endpoint_specs():
        cell = endpoint_name(*spec)
        try:
            cells[spec] = validate_replay_cell(
                root / "simulator", baseline_root / "simulator", cell, "pairing"
            )
        except FileNotFoundError as error:
            missing.append(str(error))
        except (AnalysisError, OSError, KeyError, TypeError, ValueError) as error:
            errors.append(str(error))
    return cells, missing, errors


def collect_spacing_replays(
    root: Path, baseline_root: Path
) -> tuple[dict[tuple[str, str, int, int], dict[str, Any]], list[str], list[str]]:
    cells: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    missing: list[str] = []
    errors: list[str] = []
    for spec in spacing_specs():
        cell = spacing_cell_name(*spec)
        try:
            cells[spec] = validate_replay_cell(
                root / "retained_spacing",
                baseline_root / "retained_spacing",
                cell,
                "retained_spacing",
            )
        except FileNotFoundError as error:
            missing.append(str(error))
        except (AnalysisError, OSError, KeyError, TypeError, ValueError) as error:
            errors.append(str(error))
    return cells, missing, errors


def load_live_medians_after_admission(
    baseline_root: Path,
) -> tuple[dict[tuple[int, int], float], dict[str, Any]]:
    """Read retained live labels only after the caller has admitted paired NCU."""

    path = baseline_root / "analysis/report.json"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    report = read_json(path)
    if report.get("validation_errors") != [] or report.get("missing_artifacts") != []:
        raise AnalysisError(f"{path}: v2 report is not structurally complete")
    if report.get("status") != "invalid_cohort_pairing_assay_no_timing_promotion":
        raise AnalysisError(f"{path}: unexpected v2 status {report.get('status')!r}")
    rows = report.get("direct_cells")
    if not isinstance(rows, list):
        raise AnalysisError(f"{path}: missing direct cells")
    medians: dict[tuple[int, int], float] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise AnalysisError(f"{path}: malformed direct row")
        depth = integer(row.get("depth"), f"{path}.direct.depth")
        spacing = integer(row.get("spacing"), f"{path}.direct.spacing")
        if (depth, spacing) not in pair_specs() or (depth, spacing) in medians:
            raise AnalysisError(f"{path}: unexpected/duplicate direct cell {(depth, spacing)}")
        if row.get("repeat_count") != 9 or row.get("issue_delta_gate") is not True:
            raise AnalysisError(f"{path}: unadmitted direct cell {(depth, spacing)}")
        deltas = row.get("wait_delta_cycles_per_group")
        median = row.get("median_wait_delta_cycles_per_group")
        if not isinstance(deltas, list) or len(deltas) != 9:
            raise AnalysisError(f"{path}: direct repeat count drift at {(depth, spacing)}")
        try:
            numeric = [float(value) for value in deltas]
            observed_median = float(median)
        except (TypeError, ValueError) as error:
            raise AnalysisError(f"{path}: nonnumeric direct timing label") from error
        if not all(math.isfinite(value) for value in (*numeric, observed_median)):
            raise AnalysisError(f"{path}: nonfinite direct timing label")
        if not math.isclose(
            statistics.median(numeric), observed_median, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise AnalysisError(f"{path}: direct median is inconsistent")
        medians[(depth, spacing)] = observed_median
    if set(medians) != set(pair_specs()):
        raise AnalysisError(f"{path}: expected eight direct medians")
    return medians, {
        "path": str(path),
        "sha256": sha256(path),
        "status": report["status"],
        "timing_labels_read_after_paired_ncu_admission": True,
    }


def sign(value: float) -> int:
    return 1 if value > 0 else (-1 if value < 0 else 0)


def comparison_row(
    spec: tuple[int, int],
    split: str,
    medians: Mapping[tuple[int, int], float],
    simulator: Mapping[tuple[str, int, int], Mapping[str, Any]],
) -> dict[str, Any]:
    depth, spacing = spec
    expensive = simulator[("slow_expensive", depth, spacing)]
    cheap = simulator[("slow_cheap", depth, spacing)]
    predicted = (
        int(expensive["cohort_topology_residual_exposure_cycles"])
        - int(cheap["cohort_topology_residual_exposure_cycles"])
    ) / GROUPS
    observed = float(medians[spec])
    return {
        "depth": depth,
        "spacing": spacing,
        "split": split,
        "observed_delta_cycles_per_group": observed,
        "predicted_delta_cycles_per_group": predicted,
        "signed_error": predicted - observed,
        "absolute_error": abs(predicted - observed),
        "predicted_sign": sign(predicted),
        "observed_sign": sign(observed),
        "sign_correct": sign(predicted) != 0 and sign(predicted) == sign(observed),
    }


def score_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise AnalysisError("cannot score an empty split")
    errors = [float(row["absolute_error"]) for row in rows]
    return {
        "comparisons": len(rows),
        "mae": statistics.mean(errors),
        "maximum_absolute_error": max(errors),
        "all_signs_correct": all(bool(row["sign_correct"]) for row in rows),
    }


def evaluate_reservation_score(
    medians: Mapping[tuple[int, int], float],
    simulator: Mapping[tuple[str, int, int], Mapping[str, Any]],
) -> dict[str, Any]:
    identification_rows = [
        comparison_row(spec, "identification", medians, simulator)
        for spec in IDENTIFICATION_SPECS
    ]
    identification = score_summary(identification_rows)
    identification_passed = (
        identification["mae"] <= MAE_LIMIT
        and identification["maximum_absolute_error"] <= MAX_ERROR_LIMIT
        and identification["all_signs_correct"]
    )
    result: dict[str, Any] = {
        "model": (
            "(sum(group_residual_slow_expensive) - "
            "sum(group_residual_slow_cheap)) / groups"
        ),
        "fitted_parameters": 0,
        "identification_slice": "depth 1/2, spacing 0",
        "holdout_slice": "depth 4/8 spacing 0 and every depth at spacing 4",
        "limits_cycles_per_group": {
            "mae": MAE_LIMIT,
            "maximum_absolute_error": MAX_ERROR_LIMIT,
            "correct_sign_every_comparison": True,
        },
        "identification": identification,
        "identification_passed": identification_passed,
        "holdout_evaluated": False,
        "holdout": None,
        "observational_transfer_passed": False,
        "comparisons": identification_rows,
    }
    if not identification_passed:
        result["outcome"] = "complete_identification_no_go"
        return result

    holdout_rows = [
        comparison_row(spec, "holdout", medians, simulator) for spec in HOLDOUT_SPECS
    ]
    holdout = score_summary(holdout_rows)
    holdout_passed = (
        holdout["mae"] <= MAE_LIMIT
        and holdout["maximum_absolute_error"] <= MAX_ERROR_LIMIT
        and holdout["all_signs_correct"]
    )
    result.update(
        {
            "holdout_evaluated": True,
            "holdout": holdout,
            "observational_transfer_passed": holdout_passed,
            "comparisons": identification_rows + holdout_rows,
            "outcome": (
                "complete_observational_transfer"
                if holdout_passed
                else "complete_holdout_no_go"
            ),
        }
    )
    return result


def aggregate_ncu_cells(bundles: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for depth, spacing in pair_specs():
        selected = [
            bundle
            for bundle in bundles
            if bundle["depth"] == depth and bundle["spacing"] == spacing
        ]
        hit_deltas = [int(bundle["absolute_lts_hit_delta"]) for bundle in selected]
        miss_deltas = [int(bundle["absolute_lts_miss_delta"]) for bundle in selected]
        closure_errors = [
            int(value)
            for bundle in selected
            for value in bundle[
                "absolute_hit_plus_miss_minus_total_by_variant"
            ].values()
        ]
        excess = [
            int(value)
            for bundle in selected
            for value in bundle["non_gating_dram_background"][
                "raw_dram_minus_attributed_lts_misses"
            ].values()
        ]
        rows.append(
            {
                "cell": pair_cell_name(depth, spacing),
                "depth": depth,
                "spacing": spacing,
                "bundles": len(selected),
                "orders": [bundle["order"] for bundle in selected],
                "paired_tolerance": paired_tolerance(depth),
                "maximum_absolute_lts_hit_delta": max(hit_deltas) if hit_deltas else None,
                "maximum_absolute_lts_miss_delta": max(miss_deltas) if miss_deltas else None,
                "maximum_absolute_hit_plus_miss_minus_total": (
                    max(closure_errors) if closure_errors else None
                ),
                "non_gating_dram_excess_minimum": min(excess) if excess else None,
                "non_gating_dram_excess_maximum": max(excess) if excess else None,
            }
        )
    return rows


def markdown_report(report: Mapping[str, Any]) -> str:
    counts = report["counts"]
    lines = [
        "# SM86 async-copy cohort reservation-exposure report",
        "",
        f"Status: `{report['status']}`",
        "",
        str(report["decision"]),
        "",
        "## Evidence coverage",
        "",
        "| Lane | Complete | Expected |",
        "|---|---:|---:|",
        f"| Paired NCU bundles | {counts['paired_ncu_bundles']} | 80 |",
        f"| Focused one-pass NCU reports | {counts['paired_ncu_reports']} | 400 |",
        f"| Ordered NCU rows | {counts['paired_ncu_rows']} | 800 |",
        f"| Pairing schema-18 twins | {counts['pairing_replay_cells']} | 16 |",
        f"| Retained-spacing schema-18 twins | {counts['spacing_replay_cells']} | 22 |",
        "",
        "## Gates",
        "",
        "| Criterion | Result |",
        "|---|---|",
    ]
    for name, value in report["criteria"].items():
        shown = "pending" if value is None else ("pass" if value else "fail")
        lines.append(f"| `{name}` | {shown} |")

    ncu_cells = report.get("paired_ncu_cells")
    if isinstance(ncu_cells, list) and ncu_cells:
        lines.extend(
            [
                "",
                "## Source-attributed paired NCU",
                "",
                "| Depth | Spacing | Bundles | Tolerance | Max hit delta | Max miss delta | Max hit+miss−total | DRAM excess range (non-gating) |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in ncu_cells:
            minimum = row["non_gating_dram_excess_minimum"]
            maximum = row["non_gating_dram_excess_maximum"]
            dram_range = "pending" if minimum is None else f"{minimum}..{maximum}"
            lines.append(
                f"| {row['depth']} | {row['spacing']} | {row['bundles']} | "
                f"{row['paired_tolerance']} | {row['maximum_absolute_lts_hit_delta']} | "
                f"{row['maximum_absolute_lts_miss_delta']} | "
                f"{row['maximum_absolute_hit_plus_miss_minus_total']} | {dram_range} |"
            )

    score = report.get("reservation_exposure_evaluation")
    if isinstance(score, dict):
        lines.extend(
            [
                "",
                "## Frozen reservation-exposure score",
                "",
                "| Split | Evaluated | MAE | Maximum error | All signs correct |",
                "|---|---|---:|---:|---|",
            ]
        )
        identification = score["identification"]
        lines.append(
            f"| Identification | yes | {identification['mae']:.3f} | "
            f"{identification['maximum_absolute_error']:.3f} | "
            f"{'yes' if identification['all_signs_correct'] else 'no'} |"
        )
        holdout = score.get("holdout")
        if isinstance(holdout, dict):
            lines.append(
                f"| Holdout | yes | {holdout['mae']:.3f} | "
                f"{holdout['maximum_absolute_error']:.3f} | "
                f"{'yes' if holdout['all_signs_correct'] else 'no'} |"
            )
        else:
            lines.append("| Holdout | no | — | — | — |")
        lines.extend(
            [
                "",
                "| Depth | Spacing | Split | Live delta | Predicted delta | Absolute error | Sign correct |",
                "|---:|---:|---|---:|---:|---:|---|",
            ]
        )
        for row in score["comparisons"]:
            lines.append(
                f"| {row['depth']} | {row['spacing']} | {row['split']} | "
                f"{row['observed_delta_cycles_per_group']:.3f} | "
                f"{row['predicted_delta_cycles_per_group']:.3f} | "
                f"{row['absolute_error']:.3f} | "
                f"{'yes' if row['sign_correct'] else 'no'} |"
            )

    missing = report.get("missing_artifacts", [])
    errors = report.get("validation_errors", [])
    if missing:
        lines.extend(["", "## Missing artifacts", ""])
        lines.extend(f"- `{item}`" for item in missing)
    if errors:
        lines.extend(["", "## Validation errors", ""])
        lines.extend(f"- {item}" for item in errors)
    lines.extend(
        [
            "",
            "Raw DRAM is a non-gating device-background diagnostic. Schema 18 is "
            "timing-neutral, production readiness is unchanged, and the typed "
            "`async_copy_shared_visibility` tail remains required.",
            "",
        ]
    )
    return "\n".join(lines)


def run(root: Path, baseline_root: Path, output_json: Path, output_md: Path) -> int:
    root = root.resolve()
    baseline_root = baseline_root.resolve()
    missing: list[str] = []
    errors: list[str] = []

    binary: dict[str, Any] | None
    try:
        binary = current_binary_report()
    except FileNotFoundError as error:
        binary = None
        missing.append(str(error))
    except (AnalysisError, OSError) as error:
        binary = None
        errors.append(str(error))

    bundles, bundle_missing, bundle_errors = collect_bundles(root, binary)
    missing.extend(bundle_missing)
    errors.extend(bundle_errors)

    pairing, pairing_missing, pairing_errors = collect_pairing_replays(
        root, baseline_root
    )
    spacing, spacing_missing, spacing_errors = collect_spacing_replays(
        root, baseline_root
    )
    missing.extend(pairing_missing)
    missing.extend(spacing_missing)
    errors.extend(pairing_errors)
    errors.extend(spacing_errors)

    ncu_complete = len(bundles) == 80 and not bundle_missing and not bundle_errors
    pairing_complete = len(pairing) == 16 and not pairing_missing and not pairing_errors
    spacing_complete = len(spacing) == 22 and not spacing_missing and not spacing_errors
    normalization_complete = pairing_complete and spacing_complete
    production_neutral = normalization_complete and all(
        bool(cell["production_timing_neutral"])
        and bool(cell["typed_visibility_tail_retained"])
        and bool(cell["normalized_schema17_equal"])
        for cell in (*pairing.values(), *spacing.values())
    )

    direct_baseline: dict[str, Any] | None = None
    medians: dict[tuple[int, int], float] | None = None
    # Deliberate sequencing boundary: no retained live timing label is opened
    # until every fresh source-attribution bundle has passed.
    if ncu_complete:
        try:
            medians, direct_baseline = load_live_medians_after_admission(baseline_root)
        except FileNotFoundError as error:
            missing.append(str(error))
        except (AnalysisError, OSError, KeyError, TypeError, ValueError) as error:
            errors.append(str(error))

    score: dict[str, Any] | None = None
    if medians is not None and pairing_complete:
        try:
            score = evaluate_reservation_score(medians, pairing)
        except (AnalysisError, KeyError, TypeError, ValueError) as error:
            errors.append(f"reservation-exposure scoring failed: {error}")

    criteria: dict[str, bool | None] = {
        "frozen_v2_probe_binary_exact": binary is not None,
        "paired_ncu_80_bundles_400_reports_800_rows": ncu_complete,
        "paired_ncu_odd_ab_even_ba": (
            ncu_complete
            and all(bundle["order"] == expected_order(bundle["repetition"]) for bundle in bundles)
        ),
        "paired_ncu_binary_device_args_and_one_pass_exact": ncu_complete,
        "paired_lts_total_and_hit_miss_marginals_admitted": ncu_complete,
        "raw_dram_retained_non_gating": (
            ncu_complete
            and all(
                "non_gating_dram_background" in bundle for bundle in bundles
            )
        ),
        "schema18_pairing_16_twins_complete": pairing_complete,
        "schema18_retained_spacing_22_twins_complete": spacing_complete,
        "schema18_fields_and_observed_group_selection_valid": (
            normalization_complete
            and all(
                cell["cohort_overlap_groups_observed"] == GROUPS
                for cell in (*pairing.values(), *spacing.values())
            )
        ),
        "schema18_normalizes_exactly_to_schema17": normalization_complete,
        "production_timing_neutral_and_visibility_tail_retained": production_neutral,
        "ncu_admission_preceded_live_label_read": (
            direct_baseline.get("timing_labels_read_after_paired_ncu_admission") is True
            if direct_baseline is not None
            else None
        ),
        "identification_within_frozen_limits": (
            score.get("identification_passed") is True if score is not None else None
        ),
        "holdout_within_frozen_limits": (
            score.get("observational_transfer_passed") is True
            if score is not None and score.get("holdout_evaluated") is True
            else None
        ),
        "timing_promotion_disabled": True,
    }

    complete_structural = ncu_complete and pairing_complete and spacing_complete
    if errors:
        status = "invalid_cohort_reservation_campaign"
        decision = (
            "One or more retained artifacts violate the frozen attribution, replay, "
            "or normalization contract. No timing or tail change is permitted."
        )
        exit_code = 1
    elif missing or not complete_structural or score is None:
        status = "incomplete_cohort_reservation_campaign"
        decision = (
            "The reducer is installed, but the paired NCU, schema-18 replay, "
            "schema-17 normalization, or admitted scoring evidence is incomplete."
        )
        exit_code = 2
    elif not score["identification_passed"]:
        status = "complete_cohort_reservation_identification_no_go"
        decision = (
            "The fresh source-attributed NCU campaign passes, but the fixed "
            "reservation-exposure score fails identification. The untouched holdout "
            "is not evaluated; production timing and the typed tail remain unchanged."
        )
        exit_code = 0
    elif not score["observational_transfer_passed"]:
        status = "complete_cohort_reservation_holdout_no_go"
        decision = (
            "The fixed score passes identification but fails the untouched holdout. "
            "This is a completed no-go with no timing or tail promotion."
        )
        exit_code = 0
    else:
        status = "pass_cohort_reservation_observational_transfer_no_timing_promotion"
        decision = (
            "The fixed score transfers within the frozen observational limits. This "
            "authorizes no readiness change: an independent response-envelope assay "
            "would still be required, and the typed visibility tail remains active."
        )
        exit_code = 0

    report = {
        "schema_version": 1,
        "assay": "sm86_async_cohort_reservation_exposure",
        "root": str(root),
        "baseline_root": str(baseline_root),
        "status": status,
        "decision": decision,
        "timing_promotion_authorized": False,
        "typed_visibility_tail_required": True,
        "source_attribution_frontier": (
            "l1tex_local_device_aperture_lts_requested_demand"
        ),
        "physical_dram_completion_claimed": False,
        "binary": binary,
        "direct_baseline": direct_baseline,
        "counts": {
            "paired_ncu_bundles": len(bundles),
            "paired_ncu_expected": 80,
            "paired_ncu_reports": len(bundles) * 5,
            "paired_ncu_reports_expected": 400,
            "paired_ncu_rows": len(bundles) * 10,
            "paired_ncu_rows_expected": 800,
            "pairing_replay_cells": len(pairing),
            "pairing_replay_expected": 16,
            "pairing_replay_outputs": len(pairing) * 2,
            "spacing_replay_cells": len(spacing),
            "spacing_replay_expected": 22,
            "spacing_replay_outputs": len(spacing) * 2,
            "normalized_schema18_cells": len(pairing) + len(spacing),
            "normalized_schema18_expected": 38,
        },
        "paired_ncu_cells": aggregate_ncu_cells(bundles),
        "paired_ncu_bundles": bundles,
        "pairing_schema18_cells": [pairing[key] for key in sorted(pairing)],
        "retained_spacing_schema18_cells": [spacing[key] for key in sorted(spacing)],
        "reservation_exposure_evaluation": score,
        "criteria": criteria,
        "missing_artifacts": sorted(set(missing)),
        "validation_errors": errors,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_md.write_text(markdown_report(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_md": str(output_md),
                "status": status,
                "counts": report["counts"],
                "missing": len(report["missing_artifacts"]),
                "errors": len(errors),
                "timing_promotion_authorized": False,
            },
            sort_keys=True,
        )
    )
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--output", "--output-json", dest="output_json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args(argv)
    output_json = args.output_json or args.root / "analysis/report.json"
    output_md = args.output_md or output_json.with_suffix(".md")
    try:
        return run(args.root, args.baseline_root, output_json, output_md)
    except (AnalysisError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"cohort-reservation analysis: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
