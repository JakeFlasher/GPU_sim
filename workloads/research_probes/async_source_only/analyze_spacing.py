#!/usr/bin/env python3
"""Reduce the SM86 async-copy issue-spacing discriminator campaign.

The assay fits the destination-server pipeline width Q only on the predeclared
L2 depth-2 conflict/severe, spacing-1/4 slice.  Every other replay pair is held
out.  Pipeline candidates are observational: this reducer also checks that the
typed visibility tail remains set and no candidate is charged to production
SM-service readiness.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO_ROOT / "results/captures/sm86_async_issue_spacing_20260830_v1"
LAYOUTS = ("compact", "conflict", "skew", "severe")
LAYOUT_INDEX = {"compact": 0, "conflict": 1, "severe": 2, "skew": 3}
DEPTHS = (1, 2, 4, 8)
SPACINGS = (0, 1, 2, 4, 8, 16)
Q_VALUES = (1, 2, 4, 8, 16)
DIRECT_GROUPS = 8192
TRACE_GROUPS = 64
DIRECT_REPEATS = 9
TRAINING_MAE_LIMIT = 1.0
HELD_OUT_MAE_LIMIT = 1.0
TRAINING_MAX_ABS_ERROR_LIMIT = 2.0
HELD_OUT_MAX_ABS_ERROR_LIMIT = 2.0

TOPOLOGY = {
    "compact": {"lines": 1, "bank": 1, "bypass": 1, "penalty": 0},
    "conflict": {"lines": 2, "bank": 2, "bypass": 4, "penalty": 14},
    "skew": {"lines": 3, "bank": 1, "bypass": 3, "penalty": 8},
    "severe": {"lines": 4, "bank": 4, "bypass": 4, "penalty": 18},
}

CELL_RE = re.compile(
    r"^async_issue_spacing_(?P<route>l1|l2)_"
    r"(?P<layout>compact|conflict|skew|severe)_d(?P<depth>\d+)_"
    r"s(?P<spacing>\d+)_g(?P<groups>\d+)$"
)
RESOURCE_RE = re.compile(
    r"^ Function (?P<name>\S+):\n"
    r"  REG:(?P<registers>\d+) STACK:(?P<stack>\d+) "
    r"SHARED:(?P<shared>\d+) LOCAL:(?P<local>\d+)",
    re.MULTILINE,
)
NCU_METRICS = (
    "gpc__cycles_elapsed.max",
    "sm__cycles_active.max",
    "smsp__inst_executed_op_ldgsts.sum",
    "sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts.sum",
    "sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
    "dram__sectors_read.sum",
)


class AnalysisError(RuntimeError):
    pass


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
        raise AnalysisError(f"{path}: expected a JSON object")
    return value


def cell_name(route: str, layout: str, depth: int, spacing: int, groups: int) -> str:
    return f"async_issue_spacing_{route}_{layout}_d{depth}_s{spacing}_g{groups}"


def parse_cell_name(name: str) -> tuple[str, str, int, int, int] | None:
    match = CELL_RE.fullmatch(name)
    if match is None:
        return None
    return (
        match.group("route"),
        match.group("layout"),
        int(match.group("depth")),
        int(match.group("spacing")),
        int(match.group("groups")),
    )


def direct_specs() -> list[tuple[str, str, int, int, int]]:
    specs: list[tuple[str, str, int, int, int]] = []
    specs.extend(("l2", layout, 1, 0, DIRECT_GROUPS) for layout in LAYOUTS)
    specs.extend(
        ("l2", layout, 2, spacing, DIRECT_GROUPS)
        for layout in LAYOUTS
        for spacing in SPACINGS
    )
    specs.extend(
        ("l2", layout, 8, spacing, DIRECT_GROUPS)
        for layout in LAYOUTS
        for spacing in (0, 1, 4, 8)
    )
    specs.extend(
        ("l1", layout, 4, spacing, DIRECT_GROUPS)
        for layout in ("compact", "conflict")
        for spacing in (0, 1, 4, 8)
    )
    return specs


def ncu_specs() -> list[tuple[str, str, int, int, int]]:
    specs = [
        ("l2", layout, 2, spacing, TRACE_GROUPS)
        for layout in LAYOUTS
        for spacing in (0, 16)
    ]
    specs.extend(
        ("l1", layout, 4, spacing, TRACE_GROUPS)
        for layout in ("compact", "conflict")
        for spacing in (0, 4)
    )
    return specs


def trace_specs() -> list[tuple[str, str, int, int, int]]:
    specs = [
        ("l2", layout, 2, spacing, TRACE_GROUPS)
        for layout in LAYOUTS
        for spacing in (0, 1, 4, 16)
    ]
    specs.extend(
        ("l2", layout, 8, spacing, TRACE_GROUPS)
        for layout in ("compact", "skew")
        for spacing in (0, 1, 4)
    )
    return specs


def function_name(route: str, layout: str, depth: int, spacing: int) -> str:
    bypass = 1 if route == "l2" else 0
    return (
        f"_Z20issue_spacing_kernelILi{depth}ELb{bypass}"
        f"ELi{LAYOUT_INDEX[layout]}ELi{spacing}EEvPKciP13SpacingResult"
    )


def load_resources(path: Path) -> dict[str, dict[str, int]]:
    text = path.read_text(encoding="utf-8")
    return {
        match.group("name"): {
            "registers": int(match.group("registers")),
            "stack": int(match.group("stack")),
            "shared": int(match.group("shared")),
            "local": int(match.group("local")),
        }
        for match in RESOURCE_RE.finditer(text)
    }


def static_report(root: Path) -> dict[str, Any]:
    static_root = root / "static"
    control_path = static_root / "control_bits.json"
    resources_path = static_root / "resources.txt"
    sass_path = static_root / "sass.txt"
    build_path = static_root / "build.log"
    paths = (control_path, resources_path, sass_path, build_path)
    absent = [str(path) for path in paths if not path.is_file()]
    if absent:
        raise AnalysisError(f"static evidence missing: {', '.join(absent)}")

    control = read_json(control_path)
    functions = control.get("functions")
    summary = control.get("summary")
    if not isinstance(functions, dict) or not isinstance(summary, dict):
        raise AnalysisError(f"{control_path}: malformed function summary")
    resources = load_resources(resources_path)
    sass_text = sass_path.read_text(encoding="utf-8")
    build_text = build_path.read_text(encoding="utf-8")

    expected_names = {
        function_name(route, layout, depth, spacing)
        for route in ("l1", "l2")
        for layout in LAYOUTS
        for depth in DEPTHS
        for spacing in SPACINGS
    }
    contract_failures: list[dict[str, Any]] = []
    contracts_checked = 0
    for route in ("l1", "l2"):
        for layout in LAYOUTS:
            for depth in DEPTHS:
                for spacing in SPACINGS:
                    name = function_name(route, layout, depth, spacing)
                    rows = functions.get(name)
                    resource = resources.get(name)
                    failures: list[str] = []
                    if not isinstance(rows, list):
                        contract_failures.append({"function": name, "failures": ["missing control rows"]})
                        continue
                    if resource is None:
                        failures.append("missing resources")
                    else:
                        if resource["stack"] != 0 or resource["local"] != 0:
                            failures.append("stack/local allocation is nonzero")
                        if resource["shared"] != 32768:
                            failures.append("shared allocation is not 32768 bytes")

                    copies = [row for row in rows if str(row.get("opcode_text", "")).startswith("LDGSTS")]
                    clock_reads = [
                        row
                        for row in rows
                        if "SR_CLOCKLO" in str(row.get("opcode_text", ""))
                        and str(row.get("opcode_text", "")).startswith(("CS2R", "S2UR"))
                    ]
                    commits = [row for row in rows if row.get("opcode_text") == "LDGDEPBAR"]
                    async_waits = [
                        row
                        for row in rows
                        if str(row.get("opcode_text", "")).startswith("DEPBAR.LE SB0")
                    ]
                    stores = [
                        row for row in rows if str(row.get("opcode_text", "")).startswith("STG")
                    ]
                    if len(copies) != depth:
                        failures.append(f"copy count {len(copies)} != {depth}")
                    if len(commits) != 1 or len(async_waits) != 1:
                        failures.append("commit/async-wait cardinality is not one each")
                    bypass_seen = all("BYPASS" in str(row.get("opcode_text", "")) for row in copies)
                    if copies and bypass_seen != (route == "l2"):
                        failures.append("LDGSTS route encoding mismatch")
                    if any(int(row.get("control", {}).get("wait_mask", 0)) & 1 for row in clock_reads):
                        failures.append("endpoint/filler clock waits on async barrier SB0")

                    if len(copies) == depth and commits and async_waits and stores:
                        pre_clocks = [row for row in clock_reads if row["pc"] < copies[0]["pc"]]
                        post_issue_clocks = [
                            row
                            for row in clock_reads
                            if copies[-1]["pc"] < row["pc"] < commits[0]["pc"]
                        ]
                        post_wait_clocks = [
                            row
                            for row in clock_reads
                            if async_waits[0]["pc"] < row["pc"] < min(store["pc"] for store in stores)
                        ]
                        if not (
                            len(pre_clocks) == 1
                            and len(post_issue_clocks) == 1
                            and len(post_wait_clocks) == 1
                        ):
                            failures.append(
                                "measurement endpoint clock cardinality/order is not one per frontier"
                            )
                        ordered = (
                            pre_clocks
                            and post_issue_clocks
                            and post_wait_clocks
                            and pre_clocks[0]["pc"] < copies[0]["pc"]
                            and copies[-1]["pc"] < post_issue_clocks[0]["pc"]
                            < commits[0]["pc"] < async_waits[0]["pc"]
                            < post_wait_clocks[0]["pc"]
                            < min(row["pc"] for row in stores)
                        )
                        if not ordered:
                            failures.append("measurement/copy/commit/wait/store order mismatch")
                        for left, right in zip(copies, copies[1:]):
                            between = [
                                row for row in rows if left["pc"] < row["pc"] < right["pc"]
                            ]
                            gap_clocks = sum(row in clock_reads for row in between)
                            gap_async_waits = sum(
                                str(row.get("opcode_text", "")).startswith("DEPBAR.LE SB0")
                                for row in between
                            )
                            if gap_clocks != spacing or gap_async_waits:
                                failures.append(
                                    "per-edge spacing sequence does not match requested samples"
                                )
                                break
                    contracts_checked += 1
                    if failures:
                        contract_failures.append({"function": name, "failures": failures})

    spill_counts = [int(value) for value in re.findall(r"(\d+) bytes spill (?:stores|loads)", build_text)]
    function_summaries = summary.get("functions", {})
    register_values = [value["registers"] for value in resources.values()]
    return {
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in paths
        },
        "architecture": {
            "sm86_target_present": ".target\tsm_86" in sass_text or ".target sm_86" in sass_text,
            "sass_function_count": len(re.findall(r"^\s*Function :", sass_text, re.MULTILINE)),
        },
        "function_count": len(functions),
        "resource_function_count": len(resources),
        "expected_function_count": len(expected_names),
        "missing_functions": sorted(expected_names - set(functions)),
        "unexpected_functions": sorted(set(functions) - expected_names),
        "total_decoded": summary.get("total_decoded"),
        "total_unknown_encoding": summary.get("total_unknown_encoding"),
        "summary_function_count": len(function_summaries) if isinstance(function_summaries, dict) else None,
        "contracts_checked": contracts_checked,
        "contract_failures": contract_failures,
        "register_range": [min(register_values), max(register_values)] if register_values else None,
        "maximum_spill_bytes": max(spill_counts, default=None),
        "valid": (
            len(functions) == 192
            and len(resources) == 192
            and not (expected_names - set(functions))
            and summary.get("total_decoded") == 21568
            and summary.get("total_unknown_encoding") == 0
            and not contract_failures
            and max(spill_counts, default=0) == 0
        ),
    }


def direct_cell(
    root: Path, spec: tuple[str, str, int, int, int]
) -> dict[str, Any]:
    route, layout, depth, spacing, groups = spec
    name = cell_name(*spec)
    base = root / "direct" / name
    repeats: list[dict[str, Any]] = []
    digests: list[str] = []
    for repeat in range(1, DIRECT_REPEATS + 1):
        path = base / f"repeat-{repeat}/result.json"
        if not path.is_file():
            raise AnalysisError(f"{name}: missing direct repeat {repeat}")
        value = read_json(path)
        expected = {
            "workload": "async_issue_spacing",
            "route": route,
            "layout": layout,
            "depth": depth,
            "groups": groups,
            "spacing_clock_samples_per_edge": spacing,
            "edges_per_group": depth - 1,
            "sum_consistent": True,
            "result_store_after_intervals": True,
            "correct": True,
        }
        mismatches = {
            key: {"actual": value.get(key), "expected": expected_value}
            for key, expected_value in expected.items()
            if value.get(key) != expected_value
        }
        if mismatches:
            raise AnalysisError(f"{path}: direct contract mismatch: {mismatches}")
        for field in ("issue_cycles", "wait_cycles", "total_cycles"):
            if not isinstance(value.get(field), int) or value[field] < 0:
                raise AnalysisError(f"{path}: invalid {field}")
        if value["issue_cycles"] + value["wait_cycles"] != value["total_cycles"]:
            raise AnalysisError(f"{path}: cycle intervals do not sum")
        repeats.append(value)
        digests.append(sha256(path))
    return {
        "cell": name,
        "route": route,
        "layout": layout,
        "depth": depth,
        "spacing": spacing,
        "groups": groups,
        "repeat_count": len(repeats),
        "repeats": repeats,
        "median_per_group": {
            field: statistics.median(value[field] for value in repeats) / groups
            for field in ("issue_cycles", "wait_cycles", "total_cycles")
        },
        "relative_range": {
            field: (
                max(value[field] for value in repeats)
                - min(value[field] for value in repeats)
            )
            / statistics.mean(value[field] for value in repeats)
            for field in ("issue_cycles", "wait_cycles", "total_cycles")
        },
        "result_sha256": digests,
    }


def collect_direct(
    root: Path,
) -> tuple[dict[tuple[str, str, int, int, int], dict[str, Any]], list[str], list[str], list[str]]:
    expected = set(direct_specs())
    discovered: set[tuple[str, str, int, int, int]] = set()
    unrecognized: list[str] = []
    direct_root = root / "direct"
    if direct_root.is_dir():
        for path in direct_root.iterdir():
            if not path.is_dir():
                continue
            parsed = parse_cell_name(path.name)
            if parsed is None:
                unrecognized.append(path.name)
            else:
                discovered.add(parsed)
    missing = [cell_name(*spec) for spec in sorted(expected - discovered)]
    unexpected = [cell_name(*spec) for spec in sorted(discovered - expected)] + sorted(unrecognized)
    errors: list[str] = []
    cells: dict[tuple[str, str, int, int, int], dict[str, Any]] = {}
    for spec in sorted(expected & discovered):
        try:
            cells[spec] = direct_cell(root, spec)
        except (AnalysisError, OSError, KeyError, TypeError, ValueError) as error:
            errors.append(str(error))
    return cells, missing, unexpected, errors


def parse_number(value: str) -> float:
    return float(value.replace(",", ""))


def ncu_cell(root: Path, spec: tuple[str, str, int, int, int]) -> dict[str, Any]:
    route, layout, depth, spacing, groups = spec
    name = cell_name(*spec)
    path = root / "ncu" / name / "repeat-1/ncu.csv"
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row.get("Kernel Name")]
    except (OSError, csv.Error) as error:
        raise AnalysisError(f"cannot read {path}: {error}") from error
    if len(rows) != 1:
        raise AnalysisError(f"{path}: expected one kernel row, got {len(rows)}")
    row = rows[0]
    try:
        metrics = {metric: parse_number(row[metric]) for metric in NCU_METRICS}
    except (KeyError, ValueError) as error:
        raise AnalysisError(f"{path}: missing/non-numeric NCU metric: {error}") from error
    kernel = row["Kernel Name"]
    template = f"<{depth}, {1 if route == 'l2' else 0}, {LAYOUT_INDEX[layout]}, {spacing}>"
    if template.replace(" ", "") not in kernel.replace(" ", ""):
        raise AnalysisError(f"{path}: kernel template does not match {name}: {kernel}")
    expected_copies = groups * depth
    expected_source_sectors = expected_copies * 32
    topology = TOPOLOGY[layout]
    wavefront_fanout = topology["bypass"] if route == "l2" else topology["bank"]
    expected_wavefronts = expected_copies * 8 * wavefront_fanout
    contract = {
        "ldgsts_instructions_exact": metrics["smsp__inst_executed_op_ldgsts.sum"] == expected_copies,
        "global_source_sectors_exact": metrics[
            "sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts.sum"
        ] == expected_source_sectors,
        "destination_wavefronts_exact": metrics[
            "sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum"
        ] == expected_wavefronts,
    }
    return {
        "cell": name,
        "route": route,
        "layout": layout,
        "depth": depth,
        "spacing": spacing,
        "groups": groups,
        "kernel": kernel,
        "metrics": metrics,
        "expected": {
            "ldgsts_instructions": expected_copies,
            "global_source_sectors": expected_source_sectors,
            "destination_wavefronts": expected_wavefronts,
        },
        "contract": contract,
        "contract_valid": all(contract.values()),
        "csv_sha256": sha256(path),
    }


def collect_ncu(
    root: Path,
) -> tuple[dict[tuple[str, str, int, int, int], dict[str, Any]], list[str], list[str]]:
    cells: dict[tuple[str, str, int, int, int], dict[str, Any]] = {}
    missing: list[str] = []
    errors: list[str] = []
    for spec in ncu_specs():
        name = cell_name(*spec)
        path = root / "ncu" / name / "repeat-1/ncu.csv"
        if not path.is_file():
            missing.append(name)
            continue
        try:
            cells[spec] = ncu_cell(root, spec)
        except (AnalysisError, OSError, KeyError, TypeError, ValueError) as error:
            errors.append(str(error))
    return cells, missing, errors


def argv_options(argv: Sequence[Any]) -> dict[str, str]:
    values = [str(value) for value in argv]
    options: dict[str, str] = {}
    index = 1
    while index + 1 < len(values):
        if values[index].startswith("--"):
            options[values[index]] = values[index + 1]
            index += 2
        else:
            index += 1
    return options


def nvbit_cell(root: Path, spec: tuple[str, str, int, int, int]) -> dict[str, Any]:
    route, layout, depth, spacing, groups = spec
    name = cell_name(*spec)
    base = root / "nvbit" / name
    complete_path = base / "complete.json"
    binding_path = base / "campaign_binding.json"
    stats_path = base / "stats.csv"
    trace_path = base / "dynamic_trace.pb"
    complete = read_json(complete_path)
    binding = read_json(binding_path)
    if complete.get("complete") is not True or complete.get("key") != name:
        raise AnalysisError(f"{name}: invalid NVBit completion record")
    required = complete.get("required_files")
    if not isinstance(required, list):
        raise AnalysisError(f"{name}: completion record has no required_files")
    absent = [str(item) for item in required if not (base / str(item)).is_file()]
    if absent:
        raise AnalysisError(f"{name}: missing required NVBit outputs: {absent}")
    if not trace_path.is_file() or trace_path.stat().st_size == 0:
        raise AnalysisError(f"{name}: empty/missing dynamic trace")
    if binding.get("cell") != name or binding.get("output_contract", {}).get("passed") is not True:
        raise AnalysisError(f"{name}: invalid campaign binding/output contract")
    options = argv_options(binding.get("argv", []))
    expected_options = {
        "--route": route,
        "--layout": layout,
        "--depth": str(depth),
        "--spacing-samples": str(spacing),
        "--groups": str(groups),
        "--markers": "on",
    }
    if any(options.get(key) != value for key, value in expected_options.items()):
        raise AnalysisError(f"{name}: bound argv mismatch: {options}")
    try:
        with stats_path.open(encoding="utf-8", newline="") as handle:
            rows = [
                {str(key).strip(): str(value).strip() for key, value in row.items()}
                for row in csv.DictReader(handle, skipinitialspace=True)
            ]
    except (OSError, csv.Error) as error:
        raise AnalysisError(f"cannot read {stats_path}: {error}") from error
    if len(rows) != 1:
        raise AnalysisError(f"{stats_path}: expected one traced kernel row")
    try:
        total = int(rows[0]["total_insts"])
        reported = int(rows[0]["total_reported_insts"])
    except (KeyError, ValueError) as error:
        raise AnalysisError(f"{stats_path}: malformed instruction totals: {error}") from error
    if total <= 0 or total != reported:
        raise AnalysisError(f"{stats_path}: incomplete instruction trace {total}/{reported}")
    return {
        "cell": name,
        "route": route,
        "layout": layout,
        "depth": depth,
        "spacing": spacing,
        "groups": groups,
        "trace_bytes": trace_path.stat().st_size,
        "total_instructions": total,
        "binary_sha256": complete.get("binary_sha256"),
        "tracer_sha256": complete.get("tracer_sha256"),
        "complete_sha256": sha256(complete_path),
        "binding_sha256": sha256(binding_path),
        "stats_sha256": sha256(stats_path),
        "trace_sha256": sha256(trace_path),
    }


def collect_nvbit(
    root: Path,
) -> tuple[dict[tuple[str, str, int, int, int], dict[str, Any]], list[str], list[str]]:
    cells: dict[tuple[str, str, int, int, int], dict[str, Any]] = {}
    missing: list[str] = []
    errors: list[str] = []
    for spec in trace_specs():
        name = cell_name(*spec)
        complete_path = root / "nvbit" / name / "complete.json"
        if not complete_path.is_file():
            missing.append(name)
            continue
        try:
            cells[spec] = nvbit_cell(root, spec)
        except (AnalysisError, OSError, KeyError, TypeError, ValueError) as error:
            errors.append(str(error))
    return cells, missing, errors


def expected_observation(layout: str, depth: int) -> dict[str, int]:
    copies = TRACE_GROUPS * depth
    cohorts = copies * 8
    topology = TOPOLOGY[layout]
    return {
        "copies_observed": copies,
        "destination_wavefronts_observed": cohorts * topology["bank"],
        "bypass_wavefronts_observed": cohorts * topology["bypass"],
        "source_cohorts_observed": cohorts,
        "destination_lines_observed": cohorts * topology["lines"],
        "max_bank_wavefronts_sum": copies * topology["bank"],
        "max_bypass_wavefronts_sum": copies * topology["bypass"],
        "topology_candidate_penalty_cycles": copies * topology["penalty"],
    }


def q_metric(observation: Mapping[str, Any], q: int, metric: str) -> int:
    name = f"pipeline_q{q}_{metric}_cycles"
    value = observation.get(name)
    if not isinstance(value, int) or value < 0:
        raise AnalysisError(f"schema-16 observation is missing nonnegative {name}")
    return value


def simulator_cell(root: Path, spec: tuple[str, str, int, int, int]) -> dict[str, Any]:
    route, layout, depth, spacing, groups = spec
    name = cell_name(*spec)
    first = root / "simulator" / name / "replay-1.json"
    second = root / "simulator" / name / "replay-2.json"
    if not first.is_file() or not second.is_file():
        raise AnalysisError(f"{name}: replay twins are incomplete")
    if first.read_bytes() != second.read_bytes():
        raise AnalysisError(f"{name}: replay twins differ")
    document = read_json(first)
    launches = document.get("launches")
    if not isinstance(launches, list) or len(launches) != 1:
        raise AnalysisError(f"{name}: expected one replay launch")
    launch = launches[0]
    readiness = launch.get("timing", {}).get("readiness", {})
    observation = readiness.get("bypass_async_copies")
    if readiness.get("schema_version") != 16 or not isinstance(observation, dict):
        raise AnalysisError(f"{name}: missing schema-16 bypass observation")
    for field, expected in expected_observation(layout, depth).items():
        if observation.get(field) != expected:
            raise AnalysisError(
                f"{name}: {field}={observation.get(field)!r}, expected {expected}"
            )
    if observation.get("pipeline_groups_observed") != groups:
        raise AnalysisError(
            f"{name}: pipeline_groups_observed={observation.get('pipeline_groups_observed')!r}, "
            f"expected {groups}"
        )
    q_exposures = {str(q): q_metric(observation, q, "exposure") for q in Q_VALUES}
    q_completions = {str(q): q_metric(observation, q, "completion") for q in Q_VALUES}
    tails = launch.get("counters", {}).get("unmodeled_tails", {})
    limiting = readiness.get("limiting_cycles", {})
    exposure = readiness.get("constraint_exposure_cycles", {})
    production_contract = {
        "typed_visibility_tail_retained": tails.get("async_copy_shared_visibility") is True,
        "sm_service_not_limiting": limiting.get("async_destination_sm_service") == 0,
        "sm_service_exposure_unchanged": exposure.get("async_destination_sm_service") == 0,
        "groups_waited_exact": readiness.get("async_groups", {}).get("groups_waited") == groups,
    }
    if not all(production_contract.values()):
        raise AnalysisError(f"{name}: observational-only production contract failed: {production_contract}")
    return {
        "cell": name,
        "route": route,
        "layout": layout,
        "depth": depth,
        "spacing": spacing,
        "groups": groups,
        "schema_version": readiness["schema_version"],
        "pipeline_groups_observed": observation["pipeline_groups_observed"],
        "q_exposure_cycles": q_exposures,
        "q_completion_cycles": q_completions,
        "production_contract": production_contract,
        "terminal_modeled_event_time": launch.get("timing", {}).get("terminal_modeled_event_time"),
        "completion_frontier": launch.get("timing", {}).get("completion_frontier"),
        "observation": observation,
        "replay_sha256": sha256(first),
    }


def collect_simulator(
    root: Path,
) -> tuple[dict[tuple[str, str, int, int, int], dict[str, Any]], list[str], list[str]]:
    cells: dict[tuple[str, str, int, int, int], dict[str, Any]] = {}
    missing: list[str] = []
    errors: list[str] = []
    for spec in trace_specs():
        name = cell_name(*spec)
        first = root / "simulator" / name / "replay-1.json"
        second = root / "simulator" / name / "replay-2.json"
        if not first.is_file() or not second.is_file():
            missing.append(name)
            continue
        try:
            cells[spec] = simulator_cell(root, spec)
        except (AnalysisError, OSError, KeyError, IndexError, TypeError, ValueError) as error:
            errors.append(str(error))
    return cells, missing, errors


def paired_wait_delta(
    compact: Mapping[str, Any], treatment: Mapping[str, Any]
) -> dict[str, Any]:
    groups = int(compact["groups"])
    values = [
        (right["wait_cycles"] - left["wait_cycles"]) / groups
        for left, right in zip(compact["repeats"], treatment["repeats"], strict=True)
    ]
    return {
        "per_repeat_per_group": values,
        "median_per_group": statistics.median(values),
        "minimum_per_group": min(values),
        "maximum_per_group": max(values),
        "positive_repeats": sum(value > 0 for value in values),
    }


def live_pairs(
    direct: Mapping[tuple[str, str, int, int, int], Mapping[str, Any]]
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for spec, treatment in sorted(direct.items()):
        route, layout, depth, spacing, groups = spec
        if layout == "compact":
            continue
        compact = direct.get((route, "compact", depth, spacing, groups))
        if compact is None:
            continue
        pairs.append(
            {
                "route": route,
                "layout": layout,
                "depth": depth,
                "spacing": spacing,
                **paired_wait_delta(compact, treatment),
            }
        )
    return pairs


def pair_lookup(pairs: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str, int, int], Mapping[str, Any]]:
    return {
        (str(item["route"]), str(item["layout"]), int(item["depth"]), int(item["spacing"])): item
        for item in pairs
    }


def evaluate_candidates(
    direct_pairs: Sequence[Mapping[str, Any]],
    simulator: Mapping[tuple[str, str, int, int, int], Mapping[str, Any]],
) -> dict[str, Any]:
    live = pair_lookup(direct_pairs)
    comparison_specs = [
        ("l2", layout, 2, spacing, TRACE_GROUPS)
        for layout in ("conflict", "severe", "skew")
        for spacing in (0, 1, 4, 16)
    ]
    comparison_specs.extend(
        ("l2", "skew", 8, spacing, TRACE_GROUPS) for spacing in (0, 1, 4)
    )
    rows_by_q: dict[int, list[dict[str, Any]]] = {q: [] for q in Q_VALUES}
    for route, layout, depth, spacing, groups in comparison_specs:
        compact = simulator[(route, "compact", depth, spacing, groups)]
        treatment = simulator[(route, layout, depth, spacing, groups)]
        observed = float(live[(route, layout, depth, spacing)]["median_per_group"])
        training = depth == 2 and layout in ("conflict", "severe") and spacing in (1, 4)
        for q in Q_VALUES:
            predicted = (
                int(treatment["q_exposure_cycles"][str(q)])
                - int(compact["q_exposure_cycles"][str(q)])
            ) / groups
            rows_by_q[q].append(
                {
                    "layout": layout,
                    "depth": depth,
                    "spacing": spacing,
                    "split": "training" if training else "held_out",
                    "observed_delta": observed,
                    "predicted_delta": predicted,
                    "signed_error": predicted - observed,
                    "absolute_error": abs(predicted - observed),
                }
            )

    candidates: list[dict[str, Any]] = []
    for q in Q_VALUES:
        rows = rows_by_q[q]
        training_rows = [row for row in rows if row["split"] == "training"]
        held_rows = [row for row in rows if row["split"] == "held_out"]
        train_mae = statistics.mean(row["absolute_error"] for row in training_rows)
        held_mae = statistics.mean(row["absolute_error"] for row in held_rows)
        train_max = max(row["absolute_error"] for row in training_rows)
        held_max = max(row["absolute_error"] for row in held_rows)
        training_within_limits = (
            train_mae <= TRAINING_MAE_LIMIT
            and train_max <= TRAINING_MAX_ABS_ERROR_LIMIT
        )
        held_out_within_limits = (
            held_mae <= HELD_OUT_MAE_LIMIT
            and held_max <= HELD_OUT_MAX_ABS_ERROR_LIMIT
        )
        transfers = training_within_limits and held_out_within_limits
        candidates.append(
            {
                "q": q,
                "training_mae": train_mae,
                "training_max_absolute_error": train_max,
                "held_out_mae": held_mae,
                "held_out_max_absolute_error": held_max,
                "training_mae_within_limit": train_mae <= TRAINING_MAE_LIMIT,
                "training_max_within_limit": train_max <= TRAINING_MAX_ABS_ERROR_LIMIT,
                "held_out_mae_within_limit": held_mae <= HELD_OUT_MAE_LIMIT,
                "held_out_max_within_limit": held_max <= HELD_OUT_MAX_ABS_ERROR_LIMIT,
                "training_within_limits": training_within_limits,
                "held_out_within_limits": held_out_within_limits,
                "transfers": transfers,
                "comparisons": rows,
            }
        )
    selected = min(candidates, key=lambda item: (item["training_mae"], item["q"]))
    best_training_qs = [
        item["q"]
        for item in candidates
        if math.isclose(
            item["training_mae"],
            selected["training_mae"],
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ]
    training_q_identified = (
        len(best_training_qs) == 1 and selected["training_within_limits"]
    )
    return {
        "candidate_q_values": list(Q_VALUES),
        "selection_rule": (
            "minimum training MAE; Q is identified only when the minimum is unique and "
            "passes both training limits"
        ),
        "training_slice": "L2 depth-2 conflict+severe at spacing 1 and 4",
        "held_out_slice": "L2 depth-2 skew/topology, spacing endpoints, and depth-8 skew transfer",
        "limits_cycles_per_group": {
            "training_mae": TRAINING_MAE_LIMIT,
            "training_max_absolute_error": TRAINING_MAX_ABS_ERROR_LIMIT,
            "held_out_mae": HELD_OUT_MAE_LIMIT,
            "held_out_max_absolute_error": HELD_OUT_MAX_ABS_ERROR_LIMIT,
        },
        "selected_q": selected["q"] if training_q_identified else None,
        "best_training_qs": best_training_qs,
        "training_q_identified": training_q_identified,
        "training_gate_passed_by_any_q": any(
            item["training_within_limits"] for item in candidates
        ),
        "selected_training_mae": selected["training_mae"],
        "selected_held_out_mae": selected["held_out_mae"],
        "selected_transfers": selected["transfers"],
        "any_q_transfers": any(item["transfers"] for item in candidates),
        "candidates": candidates,
    }


def predecessor_specs() -> list[tuple[str, str, int, int, int]]:
    return [
        ("l2", layout, 2, 0, TRACE_GROUPS) for layout in LAYOUTS
    ] + [
        ("l2", layout, 8, 0, TRACE_GROUPS) for layout in ("compact", "skew")
    ]


def predecessor_report(
    root: Path, current_complete: bool
) -> tuple[dict[str, Any], list[str], list[str]]:
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    errors: list[str] = []
    predecessor_root = root / "predecessor"
    for spec in predecessor_specs():
        name = cell_name(*spec)
        baseline_path = predecessor_root / name / "replay.json"
        if not baseline_path.is_file() or baseline_path.stat().st_size == 0:
            missing.append(f"predecessor/{name}/replay.json")
            continue
        try:
            baseline = read_json(baseline_path)
            readiness = baseline["launches"][0]["timing"]["readiness"]
            if readiness.get("schema_version") != 15:
                raise AnalysisError(f"{name}: predecessor is not schema 15")
            record: dict[str, Any] = {
                "cell": name,
                "predecessor_sha256": sha256(baseline_path),
                "normalized_equal": None,
            }
            if current_complete:
                current_path = root / "simulator" / name / "replay-1.json"
                current = read_json(current_path)
                normalized = deepcopy(current)
                normalized_readiness = normalized["launches"][0]["timing"]["readiness"]
                observation = normalized_readiness.get("bypass_async_copies")
                if not isinstance(observation, dict):
                    raise AnalysisError(f"{name}: current replay has no bypass observation")
                observation.pop("pipeline_groups_observed", None)
                for q in Q_VALUES:
                    observation.pop(f"pipeline_q{q}_completion_cycles", None)
                    observation.pop(f"pipeline_q{q}_exposure_cycles", None)
                normalized_readiness["schema_version"] = 15
                record["normalized_equal"] = normalized == baseline
                if not record["normalized_equal"]:
                    errors.append(f"{name}: schema-16 replay differs from predecessor beyond Q observation")
            records.append(record)
        except (AnalysisError, OSError, KeyError, IndexError, TypeError, ValueError) as error:
            errors.append(str(error))

    compatibility_cell = cell_name("l2", "compact", 2, 16, TRACE_GROUPS)
    compatibility_replay = predecessor_root / compatibility_cell / "replay.json"
    compatibility_stderr = predecessor_root / compatibility_cell / "stderr.txt"
    stderr_text = (
        compatibility_stderr.read_text(encoding="utf-8", errors="replace")
        if compatibility_stderr.is_file()
        else ""
    )
    compatibility = {
        "cell": compatibility_cell,
        "status": "predecessor_structurally_unsupported",
        "reason": "predecessor parser rejected non-SB0 DEPBAR used by the spacing filler",
        "replay_bytes": compatibility_replay.stat().st_size if compatibility_replay.is_file() else None,
        "stderr": stderr_text.strip(),
        "expected_failure_observed": (
            compatibility_replay.is_file()
            and compatibility_replay.stat().st_size == 0
            and "DEPBAR.LE SB2" in stderr_text
            and "unsupported operands or modifiers" in stderr_text
        ),
    }
    return (
        {
            "schema15_s0_cells": records,
            "all_s0_predecessors_available": len(records) == len(predecessor_specs()) and not missing,
            "all_normalized_unchanged": (
                all(record["normalized_equal"] is True for record in records)
                if current_complete and len(records) == len(predecessor_specs())
                else None
            ),
            "spaced_trace_compatibility_extension": compatibility,
        },
        missing,
        errors,
    )


def optional_binary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "present": False}
    return {"path": str(path), "present": True, "bytes": path.stat().st_size, "sha256": sha256(path)}


def markdown_report(report: Mapping[str, Any]) -> str:
    counts = report["counts"]
    criteria = report["criteria"]
    lines = [
        "# SM86 async-copy issue-spacing report",
        "",
        f"Status: `{report['status']}`",
        "",
        str(report["decision"]),
        "",
        "## Evidence coverage",
        "",
        "| Lane | Complete | Expected |",
        "|---|---:|---:|",
        f"| Direct repeat cells | {counts['direct_cells']} | {counts['direct_expected']} |",
        f"| NCU endpoints | {counts['ncu_cells']} | {counts['ncu_expected']} |",
        f"| NVBit traces | {counts['nvbit_cells']} | {counts['nvbit_expected']} |",
        f"| Simulator replay twins | {counts['simulator_cells']} | {counts['simulator_expected']} |",
        "",
        "## Assay criteria",
        "",
        "| Criterion | Result |",
        "|---|---|",
    ]
    for name, value in criteria.items():
        shown = "pending" if value is None else ("pass" if value else "fail")
        lines.append(f"| `{name}` | {shown} |")

    candidate = report.get("candidate_evaluation")
    if isinstance(candidate, dict) and candidate.get("candidates"):
        lines.extend(
            [
                "",
                "## Pipeline-Q selection and transfer",
                "",
                "Q is selected only from L2 depth-2 conflict/severe at spacing 1 and 4.",
                "",
                "| Q | Training MAE | Held-out MAE | Held-out max | Transfers |",
                "|---:|---:|---:|---:|---|",
            ]
        )
        for item in candidate["candidates"]:
            lines.append(
                f"| {item['q']} | {item['training_mae']:.3f} | "
                f"{item['held_out_mae']:.3f} | {item['held_out_max_absolute_error']:.3f} | "
                f"{'yes' if item['transfers'] else 'no'} |"
            )
        if candidate["training_q_identified"]:
            selection = f"Training identified Q=`{candidate['selected_q']}`."
        else:
            best = ", ".join(str(q) for q in candidate["best_training_qs"])
            selection = (
                f"Training did not identify Q; the best-error set is `[{best}]`. "
                f"Training gate passed by any candidate: "
                f"`{'yes' if candidate['training_gate_passed_by_any_q'] else 'no'}`."
            )
        lines.extend(
            [
                "",
                selection,
                f"Any Q transfers: `{'yes' if candidate['any_q_transfers'] else 'no'}`.",
            ]
        )

    live = pair_lookup(report.get("live_pairs", []))
    chosen = [
        ("l1", "conflict", 4, 0),
        ("l1", "conflict", 4, 4),
        ("l2", "conflict", 2, 1),
        ("l2", "severe", 2, 1),
        ("l2", "skew", 8, 0),
        ("l2", "skew", 8, 4),
    ]
    present = [(key, live[key]) for key in chosen if key in live]
    if present:
        lines.extend(
            [
                "",
                "## Representative live wait deltas",
                "",
                "| Route | Layout | Depth | Spacing | Median cycles/group |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for key, item in present:
            route, layout, depth, spacing = key
            lines.append(
                f"| {route} | {layout} | {depth} | {spacing} | "
                f"{item['median_per_group']:.3f} |"
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
            "The simulator Q fields are timing-neutral observations. A no-go result "
            "does not alter production readiness or clear the typed visibility tail.",
            "",
        ]
    )
    return "\n".join(lines)


def run(root: Path, output_json: Path, output_md: Path) -> int:
    root = root.resolve()
    missing_artifacts: list[str] = []
    validation_errors: list[str] = []

    try:
        static = static_report(root)
        if not static["valid"]:
            validation_errors.append("static SASS/control/resource contract failed")
    except (AnalysisError, OSError, KeyError, TypeError, ValueError) as error:
        static = {"valid": False, "error": str(error)}
        if "missing" in str(error):
            missing_artifacts.append(str(error))
        else:
            validation_errors.append(str(error))

    direct, direct_missing, direct_unexpected, direct_errors = collect_direct(root)
    ncu, ncu_missing, ncu_errors = collect_ncu(root)
    nvbit, nvbit_missing, nvbit_errors = collect_nvbit(root)
    simulator, simulator_missing, simulator_errors = collect_simulator(root)
    simulator_complete = set(simulator) == set(trace_specs())
    predecessor, predecessor_missing, predecessor_errors = predecessor_report(
        root, simulator_complete
    )
    missing_artifacts.extend(f"direct/{name}" for name in direct_missing)
    missing_artifacts.extend(f"ncu/{name}" for name in ncu_missing)
    missing_artifacts.extend(f"nvbit/{name}" for name in nvbit_missing)
    missing_artifacts.extend(f"simulator/{name}" for name in simulator_missing)
    missing_artifacts.extend(predecessor_missing)
    validation_errors.extend(
        direct_errors
        + ncu_errors
        + nvbit_errors
        + simulator_errors
        + predecessor_errors
    )

    pairs = live_pairs(direct)
    live_lookup = pair_lookup(pairs)
    issue_manipulation = None
    cache_all_control = None
    if all(
        spec in direct
        for spec in (
            ("l2", "compact", 2, 0, DIRECT_GROUPS),
            ("l2", "compact", 2, 16, DIRECT_GROUPS),
        )
    ):
        issue_manipulation = (
            direct[("l2", "compact", 2, 16, DIRECT_GROUPS)]["median_per_group"]["issue_cycles"]
            - direct[("l2", "compact", 2, 0, DIRECT_GROUPS)]["median_per_group"]["issue_cycles"]
        ) > 100.0
    if all(key in live_lookup for key in (("l1", "conflict", 4, 0), ("l1", "conflict", 4, 4))):
        cache_all_control = (
            live_lookup[("l1", "conflict", 4, 0)]["median_per_group"]
            - live_lookup[("l1", "conflict", 4, 4)]["median_per_group"]
        ) > 20.0

    all_trace_specs = set(trace_specs())
    candidate_evaluation: dict[str, Any] | None = None
    if not validation_errors and set(simulator) == all_trace_specs and len(direct) == len(direct_specs()):
        try:
            candidate_evaluation = evaluate_candidates(pairs, simulator)
        except (AnalysisError, KeyError, TypeError, ValueError, statistics.StatisticsError) as error:
            validation_errors.append(f"candidate evaluation failed: {error}")

    ncu_contract = len(ncu) == len(ncu_specs()) and all(cell["contract_valid"] for cell in ncu.values())
    nvbit_binary_hashes = {cell["binary_sha256"] for cell in nvbit.values()}
    criteria: dict[str, bool | None] = {
        "static_spacing_contract_valid": static.get("valid") is True,
        "direct_52_cells_complete": len(direct) == len(direct_specs()),
        "spacing_manipulation_effective": issue_manipulation,
        "cache_all_positive_control_responds": cache_all_control,
        "ncu_12_endpoints_preserve_source_and_topology": ncu_contract,
        "nvbit_22_traces_complete": len(nvbit) == len(trace_specs()) and len(nvbit_binary_hashes) == 1,
        "schema16_replay_twins_complete": len(simulator) == len(trace_specs()),
        "candidate_observational_contract_retained": (
            all(all(cell["production_contract"].values()) for cell in simulator.values())
            if len(simulator) == len(trace_specs())
            else None
        ),
        "schema15_s0_predecessor_normalized_equal": predecessor.get(
            "all_normalized_unchanged"
        ),
        "spaced_parser_compatibility_extension_recorded": predecessor.get(
            "spaced_trace_compatibility_extension", {}
        ).get("expected_failure_observed"),
        "no_pipeline_q_transfers": (
            not candidate_evaluation["any_q_transfers"]
            if candidate_evaluation is not None
            else None
        ),
    }
    if issue_manipulation is False:
        validation_errors.append("compile-time spacing did not create the required issue interval")
    if cache_all_control is False:
        validation_errors.append("cache-all positive control did not show backlog relief")
    if len(ncu) == len(ncu_specs()) and not ncu_contract:
        validation_errors.append("one or more NCU endpoints changed source work or destination topology")
    if len(nvbit) == len(trace_specs()) and len(nvbit_binary_hashes) != 1:
        validation_errors.append("NVBit traces do not bind one probe binary")

    complete = not missing_artifacts
    if validation_errors:
        status = "invalid_issue_spacing_campaign"
        decision = "The assay is invalid until the reported evidence defects are resolved."
        exit_code = 1
    elif not complete or candidate_evaluation is None:
        status = "incomplete_issue_spacing_campaign"
        decision = "The live/static evidence is reduced; schema-16 replay evidence is still incomplete."
        exit_code = 2
    elif not candidate_evaluation["any_q_transfers"]:
        status = "pass_issue_spacing_no_timing_promotion"
        decision = (
            "No predeclared Q candidate transfers from the training slice to every held-out "
            "spacing/topology/depth slice; retain the typed tail and do not promote timing."
        )
        exit_code = 0
    else:
        status = "pass_issue_spacing_candidate_transfer_observed"
        decision = (
            "At least one Q candidate meets the bounded transfer rule, but this observational "
            "milestone does not itself authorize a production timing change."
        )
        exit_code = 0

    probe_binary = REPO_ROOT / "workloads/research_probes/async_source_only/async_issue_spacing"
    simulator_binary = REPO_ROOT / "sim/target/release/sim-run"
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "status": status,
        "decision": decision,
        "assay": {
            "direct_groups": DIRECT_GROUPS,
            "direct_repeats": DIRECT_REPEATS,
            "trace_groups": TRACE_GROUPS,
            "candidate_q_values": list(Q_VALUES),
            "training_slice": "L2 d2 conflict+severe, spacing 1 and 4",
            "held_out": "all remaining captured layout/spacing/depth comparisons",
            "transfer_limits_cycles_per_group": {
                "training_mae": TRAINING_MAE_LIMIT,
                "training_max_absolute_error": TRAINING_MAX_ABS_ERROR_LIMIT,
                "held_out_mae": HELD_OUT_MAE_LIMIT,
                "held_out_max_absolute_error": HELD_OUT_MAX_ABS_ERROR_LIMIT,
            },
        },
        "binaries": {
            "probe": optional_binary(probe_binary),
            "simulator": optional_binary(simulator_binary),
        },
        "counts": {
            "direct_cells": len(direct),
            "direct_expected": len(direct_specs()),
            "direct_repeat_results": sum(cell["repeat_count"] for cell in direct.values()),
            "ncu_cells": len(ncu),
            "ncu_expected": len(ncu_specs()),
            "nvbit_cells": len(nvbit),
            "nvbit_expected": len(trace_specs()),
            "simulator_cells": len(simulator),
            "simulator_expected": len(trace_specs()),
        },
        "topology": TOPOLOGY,
        "static": static,
        "direct_discovery": {
            "expected_cells": [cell_name(*spec) for spec in direct_specs()],
            "unexpected_directories": direct_unexpected,
        },
        "direct_cells": [direct[key] for key in sorted(direct)],
        "live_pairs": pairs,
        "ncu_cells": [ncu[key] for key in sorted(ncu)],
        "nvbit_cells": [nvbit[key] for key in sorted(nvbit)],
        "simulator_cells": [simulator[key] for key in sorted(simulator)],
        "predecessor_regression": predecessor,
        "candidate_evaluation": candidate_evaluation,
        "criteria": criteria,
        "missing_artifacts": missing_artifacts,
        "validation_errors": validation_errors,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(markdown_report(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_md": str(output_md),
                "status": status,
                "counts": report["counts"],
                "missing": len(missing_artifacts),
                "errors": len(validation_errors),
            },
            sort_keys=True,
        )
    )
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", "--output-json", dest="output_json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args(argv)
    output_json = args.output_json or args.root / "analysis/report.json"
    output_md = args.output_md or output_json.with_suffix(".md")
    try:
        return run(args.root, output_json, output_md)
    except (AnalysisError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"issue-spacing analysis: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
