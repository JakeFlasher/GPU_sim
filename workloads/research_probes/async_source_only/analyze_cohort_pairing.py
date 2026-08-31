#!/usr/bin/env python3
"""Reduce the SM86 matched-marginal async-copy cohort-pairing campaign.

The live assay changes only the pairing between slow source cohorts and
expensive destination cohorts.  Schema 17 predicts the paired wait delta from
exact source-response/cohort identity.  This reducer applies the predeclared
identification/holdout split and never treats the retained uniform-cohort
issue-spacing corpus as authorization to change timing.
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
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO_ROOT / "results/captures/sm86_async_cohort_pairing_20260830_v2"
DEFAULT_PRECURSOR_ROOT = (
    REPO_ROOT / "results/captures/sm86_async_cohort_pairing_20260830_v1"
)
DEFAULT_SPACING_ROOT = REPO_ROOT / "results/captures/sm86_async_issue_spacing_20260830_v1"
DEFAULT_SPACING_BASELINE_ROOT = DEFAULT_SPACING_ROOT / "simulator"

VARIANTS = ("slow_expensive", "slow_cheap")
DEPTHS = (1, 2, 4, 8)
SPACINGS = (0, 4)
DIRECT_GROUPS = 8192
TRACE_GROUPS = 64
DIRECT_REPEATS = 9
EXPECTED_ORDERS = tuple("ab" if repeat % 2 else "ba" for repeat in range(1, 10))

EXPECTED_GPU_INDEX = "0"
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 3090 Ti"
EXPECTED_GPU_UUID = "GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c"
EXPECTED_COMPUTE_CAPABILITY = "8.6"
EXPECTED_DRIVER = "610.57.04"

IDENTIFICATION_MAE_LIMIT = 1.0
HOLDOUT_MAE_LIMIT = 1.0
IDENTIFICATION_MAX_ERROR_LIMIT = 2.0
HOLDOUT_MAX_ERROR_LIMIT = 2.0
MATERIAL_DELTA_MINIMUM = 1.0
MATERIAL_SIGN_REPEATS = 8
ISSUE_MEDIAN_DELTA_LIMIT = 0.5
ISSUE_REPEAT_DELTA_LIMIT = 1.0
ISSUE_REPEATS_REQUIRED = 8

PAIR_CELL_RE = re.compile(
    r"^async_cohort_pairing_d(?P<depth>1|2|4|8)_s(?P<spacing>0|4)_g(?P<groups>\d+)$"
)
ENDPOINT_RE = re.compile(
    r"^async_cohort_pairing_(?P<variant>slow_expensive|slow_cheap)_"
    r"d(?P<depth>1|2|4|8)_s(?P<spacing>0|4)_g(?P<groups>\d+)$"
)
RESOURCE_RE = re.compile(
    r"^ Function (?P<name>\S+):\n"
    r"  REG:(?P<registers>\d+) STACK:(?P<stack>\d+) "
    r"SHARED:(?P<shared>\d+) LOCAL:(?P<local>\d+)",
    re.MULTILINE,
)
SPILL_RE = re.compile(
    r"(?P<stack>\d+) bytes stack frame, (?P<stores>\d+) bytes spill stores, "
    r"(?P<loads>\d+) bytes spill loads"
)
LDGSTS_DEST_RE = re.compile(
    r"^LDGSTS\.E\.BYPASS\.128 \[R\d+(?:\+(?P<offset>0x[0-9a-fA-F]+))?\],"
)

NCU_METRICS = (
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
PAIRED_SOURCE_METRICS = (
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
    "dram__sectors_read.sum",
)

COHORT_FIELDS = (
    "cohort_groups_observed",
    "cohort_copies_observed",
    "cohort_response_memberships_observed",
    "cohort_response_span_cycles",
    "cohort_topology_penalty_cycles",
    "cohort_candidate_completion_cycles",
    "cohort_candidate_exposure_cycles",
    "cohort_candidate_later_than_per_copy_cycles",
    "cohort_candidate_earlier_than_per_copy_cycles",
    "cohort_candidate_later_than_group_aggregate_cycles",
    "cohort_candidate_earlier_than_group_aggregate_cycles",
    "cohort_candidate_later_than_production_cycles",
    "cohort_candidate_earlier_than_production_cycles",
)

LAYOUT_PENALTY = {"compact": 0, "conflict": 14, "skew": 8, "severe": 18}
SPACING_LAYOUTS = ("compact", "conflict", "skew", "severe")


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


def read_json_list(path: Path) -> list[Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisError(f"cannot read {path}: {error}") from error
    if not isinstance(value, list):
        raise AnalysisError(f"{path}: expected a JSON array")
    return value


def parse_number(value: str) -> float:
    return float(value.replace(",", ""))


def nonnegative_integer(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AnalysisError(f"{context}: expected a nonnegative integer, got {value!r}")
    return value


def pair_cell_name(depth: int, spacing: int, groups: int = DIRECT_GROUPS) -> str:
    return f"async_cohort_pairing_d{depth}_s{spacing}_g{groups}"


def endpoint_name(
    variant: str, depth: int, spacing: int, groups: int = TRACE_GROUPS
) -> str:
    return f"async_cohort_pairing_{variant}_d{depth}_s{spacing}_g{groups}"


def function_name(depth: int, spacing: int) -> str:
    return (
        f"_Z21cohort_pairing_kernelILi{depth}ELi{spacing}EEv"
        "PKcS1_jibP13PairingResult"
    )


def direct_specs() -> list[tuple[int, int]]:
    return [(depth, spacing) for depth in DEPTHS for spacing in SPACINGS]


def endpoint_specs() -> list[tuple[str, int, int]]:
    return [
        (variant, depth, spacing)
        for variant in VARIANTS
        for depth in DEPTHS
        for spacing in SPACINGS
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


def spacing_cell_name(route: str, layout: str, depth: int, spacing: int) -> str:
    return f"async_issue_spacing_{route}_{layout}_d{depth}_s{spacing}_g{TRACE_GROUPS}"


def host_report(root: Path) -> dict[str, Any]:
    path = root / "host/gpu-before.csv"
    if not path.is_file():
        raise AnalysisError(f"host evidence missing: {path}")
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle, skipinitialspace=True))
    except (OSError, csv.Error) as error:
        raise AnalysisError(f"cannot read {path}: {error}") from error
    if len(rows) != 1 or len(rows[0]) < 5:
        raise AnalysisError(f"{path}: expected one GPU identity row")
    row = [item.strip() for item in rows[0]]
    expected = (
        EXPECTED_GPU_INDEX,
        EXPECTED_GPU_NAME,
        EXPECTED_GPU_UUID,
        EXPECTED_COMPUTE_CAPABILITY,
        EXPECTED_DRIVER,
    )
    actual = tuple(row[:5])
    return {
        "path": str(path),
        "device_index": row[0],
        "device_name": row[1],
        "uuid": row[2],
        "compute_capability": row[3],
        "driver": row[4],
        "expected": list(expected),
        "valid": actual == expected,
        "sha256": sha256(path),
    }


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
    resource_path = static_root / "resources.txt"
    sass_path = static_root / "sass.txt"
    build_path = static_root / "build.log"
    absent = [
        str(path)
        for path in (control_path, resource_path, sass_path, build_path)
        if not path.is_file()
    ]
    if absent:
        raise AnalysisError(f"static evidence missing: {', '.join(absent)}")

    control = read_json(control_path)
    functions = control.get("functions")
    summary = control.get("summary")
    if not isinstance(functions, dict) or not isinstance(summary, dict):
        raise AnalysisError(f"{control_path}: malformed control decode")
    resources = load_resources(resource_path)
    sass_text = sass_path.read_text(encoding="utf-8")
    build_text = build_path.read_text(encoding="utf-8")
    expected_names = {function_name(depth, spacing) for depth, spacing in direct_specs()}
    failures: list[dict[str, Any]] = []
    decoded_total = 0

    for depth, spacing in direct_specs():
        name = function_name(depth, spacing)
        rows = functions.get(name)
        resource = resources.get(name)
        local: list[str] = []
        if not isinstance(rows, list):
            failures.append({"function": name, "failures": ["missing control rows"]})
            continue
        decoded_total += len(rows)
        if any(row.get("status") != "decoded" for row in rows):
            local.append("contains an undecoded instruction")
        opcodes = [str(row.get("opcode_text", "")) for row in rows]
        ldgsts = [index for index, opcode in enumerate(opcodes) if opcode.startswith("LDGSTS")]
        if len(ldgsts) != depth:
            local.append(f"LDGSTS count {len(ldgsts)} != depth {depth}")
        if any(not opcodes[index].startswith("LDGSTS.E.BYPASS.128 ") for index in ldgsts):
            local.append("LDGSTS route/width drift")
        offsets: list[int] = []
        for index in ldgsts:
            match = LDGSTS_DEST_RE.match(opcodes[index])
            if match is None:
                local.append(f"cannot parse LDGSTS destination: {opcodes[index]}")
                continue
            offsets.append(int(match.group("offset") or "0", 16))
        if offsets != [position * 0x1000 for position in range(depth)]:
            local.append(f"static LDGSTS positions drifted: {offsets}")

        primary_clocks = [
            index
            for index, opcode in enumerate(opcodes)
            if opcode.startswith("CS2R ") and "SR_CLOCKLO" in opcode
        ]
        filler_clocks = [
            index
            for index, opcode in enumerate(opcodes)
            if (
                opcode.startswith("CS2R.32 ") or opcode.startswith("S2UR ")
            )
            and "SR_CLOCKLO" in opcode
        ]
        ldgdepbar = [index for index, opcode in enumerate(opcodes) if opcode == "LDGDEPBAR"]
        wait = [
            index
            for index, opcode in enumerate(opcodes)
            if opcode == "DEPBAR.LE SB0, 0x0"
        ]
        stores = [index for index, opcode in enumerate(opcodes) if opcode.startswith("STG")]
        if len(primary_clocks) != 3:
            local.append(f"primary clock count {len(primary_clocks)} != 3")
        if len(filler_clocks) != spacing * (depth - 1):
            local.append(
                f"filler clock count {len(filler_clocks)} != {spacing * (depth - 1)}"
            )
        if len(ldgdepbar) != 1 or len(wait) != 1:
            local.append("commit/wait instruction count drift")
        if len(stores) != 4:
            local.append(f"result-store count {len(stores)} != 4")
        if any(opcode.startswith("STS") or opcode.startswith("BAR") for opcode in opcodes):
            local.append("store/barrier entered measured kernel")
        if (
            len(primary_clocks) == 3
            and ldgsts
            and len(ldgdepbar) == 1
            and len(wait) == 1
            and not (
                primary_clocks[0]
                < ldgsts[0]
                <= ldgsts[-1]
                < primary_clocks[1]
                < ldgdepbar[0]
                < wait[0]
                < primary_clocks[2]
            )
        ):
            local.append("clock/issue/commit/wait ordering drift")
        if primary_clocks and stores and min(stores) <= primary_clocks[-1]:
            local.append("result store entered a measured interval")
        if ldgsts:
            issue_region = rows[primary_clocks[0] : ldgsts[-1] + 1] if primary_clocks else []
            if any(int(row.get("control", {}).get("wait_mask", 0)) & 1 for row in issue_region):
                local.append("SB0 wait mask entered the issue interval")

        if resource is None:
            local.append("missing resource record")
        else:
            if resource["stack"] != 0 or resource["local"] != 0:
                local.append("stack/local allocation is nonzero")
            if resource["shared"] != 32768:
                local.append(f"shared allocation {resource['shared']} != 32768")
            if resource["registers"] <= 0:
                local.append("register count is not positive")
        if name not in sass_text:
            local.append("function absent from SASS dump")
        if local:
            failures.append({"function": name, "failures": local})

    spill_rows = [
        {
            "stack": int(match.group("stack")),
            "stores": int(match.group("stores")),
            "loads": int(match.group("loads")),
        }
        for match in SPILL_RE.finditer(build_text)
    ]
    global_contract = {
        "control_function_set_exact": set(functions) == expected_names,
        "resource_function_set_exact": set(resources) == expected_names,
        "summary_function_set_exact": set(summary.get("functions", {})) == expected_names,
        "decoded_total_self_consistent": summary.get("total_decoded") == decoded_total,
        "unknown_encoding_zero": summary.get("total_unknown_encoding") == 0,
        "spill_records_complete": len(spill_rows) == len(expected_names),
        "stack_and_spills_zero": bool(spill_rows)
        and all(not any(record.values()) for record in spill_rows),
        "ptxas_barriers_zero": build_text.count("used 0 barriers") == len(expected_names),
    }
    return {
        "expected_function_count": len(expected_names),
        "control_function_count": len(functions),
        "resource_function_count": len(resources),
        "decoded_total": decoded_total,
        "contract": global_contract,
        "function_failures": failures,
        "valid": all(global_contract.values()) and not failures,
        "artifacts": {
            "control_bits_sha256": sha256(control_path),
            "resources_sha256": sha256(resource_path),
            "sass_sha256": sha256(sass_path),
            "build_sha256": sha256(build_path),
        },
    }


def validate_arm(value: Any, groups: int, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AnalysisError(f"{context}: expected result object")
    for field in ("issue_cycles", "wait_cycles", "total_cycles", "token", "mismatches"):
        nonnegative_integer(value.get(field), f"{context}.{field}")
    if value["issue_cycles"] + value["wait_cycles"] != value["total_cycles"]:
        raise AnalysisError(f"{context}: cycle intervals do not sum")
    expected = {
        "issue_cycles_per_group": value["issue_cycles"] / groups,
        "wait_cycles_per_group": value["wait_cycles"] / groups,
        "total_cycles_per_group": value["total_cycles"] / groups,
    }
    for field, expected_value in expected.items():
        actual = value.get(field)
        if not isinstance(actual, (int, float)) or not math.isclose(
            float(actual), expected_value, rel_tol=0.0, abs_tol=1.0e-8
        ):
            raise AnalysisError(f"{context}: {field} does not match integer interval")
    if value.get("sum_consistent") is not True or value.get("correct") is not True:
        raise AnalysisError(f"{context}: correctness flag failed")
    if value["mismatches"] != 0:
        raise AnalysisError(f"{context}: destination mismatches={value['mismatches']}")
    return value


def direct_cell(root: Path, depth: int, spacing: int) -> dict[str, Any]:
    name = pair_cell_name(depth, spacing)
    base = root / "direct" / name
    repeats: list[dict[str, Any]] = []
    digests: list[str] = []
    expected_common = {
        "workload": "async_cohort_pairing",
        "route": "l2",
        "mode": "pair",
        "depth": depth,
        "groups": DIRECT_GROUPS,
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
        "device": EXPECTED_GPU_NAME,
        "compute_capability": EXPECTED_COMPUTE_CAPABILITY,
        "l2_cache_bytes": 6291456,
        "correct": True,
    }
    for repeat, expected_order in enumerate(EXPECTED_ORDERS, 1):
        path = base / f"repeat-{repeat}/result.json"
        if not path.is_file():
            raise AnalysisError(f"{name}: missing direct repeat {repeat}")
        value = read_json(path)
        mismatches = {
            field: {"actual": value.get(field), "expected": expected}
            for field, expected in expected_common.items()
            if value.get(field) != expected
        }
        if value.get("order") != expected_order:
            mismatches["order"] = {
                "actual": value.get("order"),
                "expected": expected_order,
            }
        if mismatches:
            raise AnalysisError(f"{path}: direct contract mismatch: {mismatches}")
        expensive = validate_arm(value.get("slow_expensive"), DIRECT_GROUPS, f"{path}:expensive")
        cheap = validate_arm(value.get("slow_cheap"), DIRECT_GROUPS, f"{path}:cheap")
        expected_wait_delta = (expensive["wait_cycles"] - cheap["wait_cycles"]) / DIRECT_GROUPS
        expected_issue_delta = (expensive["issue_cycles"] - cheap["issue_cycles"]) / DIRECT_GROUPS
        for field, expected_delta in (
            ("paired_wait_delta_cycles_per_group", expected_wait_delta),
            ("paired_issue_delta_cycles_per_group", expected_issue_delta),
        ):
            actual = value.get(field)
            if not isinstance(actual, (int, float)) or not math.isclose(
                float(actual), expected_delta, rel_tol=0.0, abs_tol=1.0e-8
            ):
                raise AnalysisError(f"{path}: {field} does not match arm intervals")
        repeats.append(value)
        digests.append(sha256(path))

    issue_deltas = [float(item["paired_issue_delta_cycles_per_group"]) for item in repeats]
    wait_deltas = [float(item["paired_wait_delta_cycles_per_group"]) for item in repeats]
    median_issue = statistics.median(issue_deltas)
    issue_repeats_within = sum(abs(value) <= ISSUE_REPEAT_DELTA_LIMIT for value in issue_deltas)
    issue_gate = (
        abs(median_issue) <= ISSUE_MEDIAN_DELTA_LIMIT
        and issue_repeats_within >= ISSUE_REPEATS_REQUIRED
    )
    return {
        "cell": name,
        "depth": depth,
        "spacing": spacing,
        "groups": DIRECT_GROUPS,
        "repeat_count": len(repeats),
        "orders": [item["order"] for item in repeats],
        "issue_delta_cycles_per_group": issue_deltas,
        "wait_delta_cycles_per_group": wait_deltas,
        "median_issue_delta_cycles_per_group": median_issue,
        "median_wait_delta_cycles_per_group": statistics.median(wait_deltas),
        "issue_repeats_within_one_cycle": issue_repeats_within,
        "issue_delta_gate": issue_gate,
        "repeats": repeats,
        "result_sha256": digests,
    }


def collect_direct(
    root: Path,
) -> tuple[dict[tuple[int, int], dict[str, Any]], list[str], list[str], list[str]]:
    expected = set(direct_specs())
    discovered: set[tuple[int, int]] = set()
    unexpected: list[str] = []
    direct_root = root / "direct"
    if direct_root.is_dir():
        for path in direct_root.iterdir():
            if not path.is_dir():
                continue
            match = PAIR_CELL_RE.fullmatch(path.name)
            if match is None:
                unexpected.append(path.name)
                continue
            parsed = (int(match.group("depth")), int(match.group("spacing")))
            if int(match.group("groups")) != DIRECT_GROUPS or parsed not in expected:
                unexpected.append(path.name)
            else:
                discovered.add(parsed)
    missing = [pair_cell_name(*spec) for spec in sorted(expected - discovered)]
    cells: dict[tuple[int, int], dict[str, Any]] = {}
    errors: list[str] = []
    for spec in sorted(expected & discovered):
        try:
            cells[spec] = direct_cell(root, *spec)
        except (AnalysisError, OSError, KeyError, TypeError, ValueError) as error:
            errors.append(str(error))
    return cells, missing, sorted(unexpected), errors


def ncu_cell(root: Path, variant: str, depth: int, spacing: int) -> dict[str, Any]:
    name = endpoint_name(variant, depth, spacing)
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
    kernel = str(row["Kernel Name"])
    template = f"cohort_pairing_kernel<{depth},{spacing}>"
    if template not in kernel.replace(" ", ""):
        raise AnalysisError(f"{path}: kernel template does not match {name}: {kernel}")
    copies = TRACE_GROUPS * depth
    expected = {
        "ldgsts_instructions": copies,
        "global_source_sectors": copies * 32,
        "destination_wavefronts": copies * 20,
    }
    contract = {
        "ldgsts_instructions_exact": math.isclose(
            metrics["smsp__inst_executed_op_ldgsts.sum"],
            expected["ldgsts_instructions"],
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ),
        "global_source_sectors_exact": math.isclose(
            metrics["sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts.sum"],
            expected["global_source_sectors"],
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ),
        "destination_wavefronts_exact": math.isclose(
            metrics[
                "sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum"
            ],
            expected["destination_wavefronts"],
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ),
        "l2_hits_nonzero": metrics[
            "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum"
        ]
        > 0,
        "l2_misses_nonzero": metrics[
            "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum"
        ]
        > 0,
        "dram_reads_nonzero": metrics["dram__sectors_read.sum"] > 0,
    }
    return {
        "cell": name,
        "variant": variant,
        "depth": depth,
        "spacing": spacing,
        "groups": TRACE_GROUPS,
        "kernel": kernel,
        "metrics": metrics,
        "expected": expected,
        "contract": contract,
        "contract_valid": all(contract.values()),
        "csv_sha256": sha256(path),
    }


def collect_ncu(
    root: Path,
) -> tuple[dict[tuple[str, int, int], dict[str, Any]], list[str], list[str]]:
    cells: dict[tuple[str, int, int], dict[str, Any]] = {}
    missing: list[str] = []
    errors: list[str] = []
    for spec in endpoint_specs():
        name = endpoint_name(*spec)
        path = root / "ncu" / name / "repeat-1/ncu.csv"
        if not path.is_file():
            missing.append(name)
            continue
        try:
            cells[spec] = ncu_cell(root, *spec)
        except (AnalysisError, OSError, KeyError, TypeError, ValueError) as error:
            errors.append(str(error))
    return cells, missing, errors


def ncu_pair_reports(
    cells: Mapping[tuple[str, int, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for depth, spacing in direct_specs():
        expensive = cells.get(("slow_expensive", depth, spacing))
        cheap = cells.get(("slow_cheap", depth, spacing))
        if expensive is None or cheap is None:
            continue
        expected_sectors = TRACE_GROUPS * depth * 32
        tolerance = max(4, math.ceil(0.01 * expected_sectors))
        differences = {
            metric: abs(float(expensive["metrics"][metric]) - float(cheap["metrics"][metric]))
            for metric in PAIRED_SOURCE_METRICS
        }
        within = {metric: delta <= tolerance for metric, delta in differences.items()}
        reports.append(
            {
                "depth": depth,
                "spacing": spacing,
                "expected_source_sectors": expected_sectors,
                "paired_tolerance": tolerance,
                "absolute_differences": differences,
                "within_tolerance": within,
                "valid": bool(expensive["contract_valid"])
                and bool(cheap["contract_valid"])
                and all(within.values()),
            }
        )
    return reports


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


def last_json_object(path: Path) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as error:
        raise AnalysisError(f"cannot read {path}: {error}") from error
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise AnalysisError(f"{path}: no JSON result object")


def nvbit_cell(root: Path, variant: str, depth: int, spacing: int) -> dict[str, Any]:
    name = endpoint_name(variant, depth, spacing)
    base = root / "nvbit" / name
    complete_path = base / "complete.json"
    binding_path = base / "campaign_binding.json"
    stats_path = base / "stats.csv"
    trace_path = base / "dynamic_trace.pb"
    stdout_path = base / "stdout.txt"
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
        "--groups": str(TRACE_GROUPS),
        "--depth": str(depth),
        "--spacing-samples": str(spacing),
        "--variant": variant,
        "--markers": "on",
    }
    if any(options.get(key) != value for key, value in expected_options.items()):
        raise AnalysisError(f"{name}: bound argv mismatch: {options}")
    binary_sha = complete.get("binary_sha256")
    if binary_sha != binding.get("binary", {}).get("sha256"):
        raise AnalysisError(f"{name}: complete/binding binary identity mismatch")

    exclusions: dict[str, dict[str, Any]] = {}
    for reason in (
        "unsupported_memory_path",
        "unmapped_memory_range",
        "cross_allocation_lane_span",
    ):
        path = base / f"extra_info/{reason}.json"
        if not path.is_file():
            raise AnalysisError(f"{name}: missing exclusion sidecar {path.name}")
        entries = read_json_list(path)
        exclusions[reason] = {
            "entries": len(entries),
            "empty": not entries,
            "sha256": sha256(path),
        }
        if entries:
            raise AnalysisError(f"{name}: nonempty {reason} exclusion sidecar")

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

    result = last_json_object(stdout_path)
    expected_result = {
        "workload": "async_cohort_pairing",
        "route": "l2",
        "mode": "endpoint",
        "variant": variant,
        "depth": depth,
        "groups": TRACE_GROUPS,
        "spacing_clock_samples_per_edge": spacing,
        "source_pool_single_allocation": True,
        "correct": True,
    }
    mismatches = {
        field: {"actual": result.get(field), "expected": expected}
        for field, expected in expected_result.items()
        if result.get(field) != expected
    }
    if mismatches:
        raise AnalysisError(f"{stdout_path}: endpoint result mismatch: {mismatches}")
    validate_arm(result.get("result"), TRACE_GROUPS, f"{stdout_path}:result")
    return {
        "cell": name,
        "variant": variant,
        "depth": depth,
        "spacing": spacing,
        "groups": TRACE_GROUPS,
        "trace_bytes": trace_path.stat().st_size,
        "total_instructions": total,
        "binary_sha256": binary_sha,
        "tracer_sha256": complete.get("tracer_sha256"),
        "exclusions": exclusions,
        "complete_sha256": sha256(complete_path),
        "binding_sha256": sha256(binding_path),
        "stats_sha256": sha256(stats_path),
        "trace_sha256": sha256(trace_path),
    }


def collect_nvbit(
    root: Path,
) -> tuple[dict[tuple[str, int, int], dict[str, Any]], list[str], list[str]]:
    cells: dict[tuple[str, int, int], dict[str, Any]] = {}
    missing: list[str] = []
    errors: list[str] = []
    for spec in endpoint_specs():
        name = endpoint_name(*spec)
        complete = root / "nvbit" / name / "complete.json"
        if not complete.is_file():
            missing.append(name)
            continue
        try:
            cells[spec] = nvbit_cell(root, *spec)
        except (AnalysisError, OSError, KeyError, TypeError, ValueError) as error:
            errors.append(str(error))
    return cells, missing, errors


def production_contract(launch: Mapping[str, Any], groups: int) -> dict[str, bool]:
    readiness = launch.get("timing", {}).get("readiness", {})
    tails = launch.get("counters", {}).get("unmodeled_tails", {})
    limiting = readiness.get("limiting_cycles", {})
    exposure = readiness.get("constraint_exposure_cycles", {})
    async_groups = readiness.get("async_groups", {})
    return {
        "typed_visibility_tail_retained": tails.get("async_copy_shared_visibility") is True,
        "sm_service_not_limiting": limiting.get("async_destination_sm_service") == 0,
        "sm_service_exposure_zero": exposure.get("async_destination_sm_service") == 0,
        "sm_service_groups_zero": async_groups.get("destination_sm_service_limited_groups") == 0,
        "groups_waited_exact": async_groups.get("groups_waited") == groups,
    }


def replay_launch(path: Path, expected_schema: int) -> tuple[dict[str, Any], dict[str, Any]]:
    document = read_json(path)
    launches = document.get("launches")
    if not isinstance(launches, list) or len(launches) != 1:
        raise AnalysisError(f"{path}: expected one replay launch")
    launch = launches[0]
    if not isinstance(launch, dict):
        raise AnalysisError(f"{path}: replay launch is not an object")
    readiness = launch.get("timing", {}).get("readiness", {})
    observation = readiness.get("bypass_async_copies")
    if readiness.get("schema_version") != expected_schema or not isinstance(observation, dict):
        raise AnalysisError(f"{path}: missing schema-{expected_schema} bypass observation")
    return document, launch


def require_cohort_fields(observation: Mapping[str, Any], context: str) -> None:
    for field in COHORT_FIELDS:
        nonnegative_integer(observation.get(field), f"{context}.{field}")


def pairing_expected_observation(depth: int) -> dict[str, int]:
    copies = TRACE_GROUPS * depth
    return {
        "copies_observed": copies,
        "destination_wavefronts_observed": copies * 32,
        "bypass_wavefronts_observed": copies * 32,
        "source_cohorts_observed": copies * 8,
        "destination_lines_observed": copies * 20,
        "max_bank_wavefronts_sum": copies * 4,
        "max_bypass_wavefronts_sum": copies * 4,
        "topology_candidate_penalty_cycles": copies * 18,
        "pipeline_groups_observed": TRACE_GROUPS,
        "cohort_groups_observed": TRACE_GROUPS,
        "cohort_copies_observed": copies,
        "cohort_response_memberships_observed": copies * 32,
        "cohort_topology_penalty_cycles": copies * 72,
    }


def simulator_cell(root: Path, variant: str, depth: int, spacing: int) -> dict[str, Any]:
    name = endpoint_name(variant, depth, spacing)
    base = root / "simulator" / name
    first = base / "replay-1.json"
    second = base / "replay-2.json"
    if not first.is_file() or not second.is_file():
        raise AnalysisError(f"{name}: replay twins are incomplete")
    if first.read_bytes() != second.read_bytes():
        raise AnalysisError(f"{name}: replay twins differ")
    _, launch = replay_launch(first, 17)
    readiness = launch["timing"]["readiness"]
    observation = readiness["bypass_async_copies"]
    require_cohort_fields(observation, name)
    for field, expected in pairing_expected_observation(depth).items():
        if observation.get(field) != expected:
            raise AnalysisError(
                f"{name}: {field}={observation.get(field)!r}, expected {expected}"
            )
    contract = production_contract(launch, TRACE_GROUPS)
    if not all(contract.values()):
        raise AnalysisError(f"{name}: timing-neutral production contract failed: {contract}")
    return {
        "cell": name,
        "variant": variant,
        "depth": depth,
        "spacing": spacing,
        "groups": TRACE_GROUPS,
        "schema_version": 17,
        "cohort_candidate_completion_cycles": observation[
            "cohort_candidate_completion_cycles"
        ],
        "cohort_candidate_exposure_cycles": observation[
            "cohort_candidate_exposure_cycles"
        ],
        "production_contract": contract,
        "terminal_modeled_event_time": launch.get("timing", {}).get(
            "terminal_modeled_event_time"
        ),
        "completion_frontier": launch.get("timing", {}).get("completion_frontier"),
        "observation": observation,
        "replay_sha256": sha256(first),
    }


def collect_simulator(
    root: Path,
) -> tuple[dict[tuple[str, int, int], dict[str, Any]], list[str], list[str]]:
    cells: dict[tuple[str, int, int], dict[str, Any]] = {}
    missing: list[str] = []
    errors: list[str] = []
    for spec in endpoint_specs():
        name = endpoint_name(*spec)
        first = root / "simulator" / name / "replay-1.json"
        second = root / "simulator" / name / "replay-2.json"
        if not first.is_file() or not second.is_file():
            missing.append(name)
            continue
        try:
            cells[spec] = simulator_cell(root, *spec)
        except (AnalysisError, OSError, KeyError, IndexError, TypeError, ValueError) as error:
            errors.append(str(error))
    return cells, missing, errors


def normalize_schema17(document: Mapping[str, Any], context: str) -> dict[str, Any]:
    normalized = deepcopy(document)
    try:
        readiness = normalized["launches"][0]["timing"]["readiness"]
        observation = readiness["bypass_async_copies"]
    except (KeyError, IndexError, TypeError) as error:
        raise AnalysisError(f"{context}: malformed replay during normalization: {error}") from error
    if readiness.get("schema_version") != 17 or not isinstance(observation, dict):
        raise AnalysisError(f"{context}: expected schema 17 for normalization")
    missing = [field for field in COHORT_FIELDS if field not in observation]
    if missing:
        raise AnalysisError(f"{context}: schema-17 cohort fields missing: {missing}")
    for field in COHORT_FIELDS:
        observation.pop(field)
    readiness["schema_version"] = 16
    return normalized


def spacing_expected_cohort(layout: str, depth: int) -> dict[str, int]:
    copies = TRACE_GROUPS * depth
    return {
        "cohort_groups_observed": TRACE_GROUPS,
        "cohort_copies_observed": copies,
        "cohort_response_memberships_observed": copies * 32,
        "cohort_topology_penalty_cycles": copies * 8 * LAYOUT_PENALTY[layout],
    }


def retained_spacing_cell(
    root: Path,
    baseline_root: Path,
    spec: tuple[str, str, int, int],
) -> dict[str, Any]:
    route, layout, depth, spacing = spec
    name = spacing_cell_name(*spec)
    base = root / "retained_spacing" / name
    first = base / "replay-1.json"
    second = base / "replay-2.json"
    baseline_path = baseline_root / name / "replay-1.json"
    if not first.is_file() or not second.is_file():
        raise AnalysisError(f"retained {name}: replay twins are incomplete")
    if not baseline_path.is_file():
        raise AnalysisError(f"retained {name}: schema-16 baseline is missing")
    if first.read_bytes() != second.read_bytes():
        raise AnalysisError(f"retained {name}: replay twins differ")
    current, launch = replay_launch(first, 17)
    observation = launch["timing"]["readiness"]["bypass_async_copies"]
    require_cohort_fields(observation, f"retained {name}")
    for field, expected in spacing_expected_cohort(layout, depth).items():
        if observation.get(field) != expected:
            raise AnalysisError(
                f"retained {name}: {field}={observation.get(field)!r}, expected {expected}"
            )
    contract = production_contract(launch, TRACE_GROUPS)
    if not all(contract.values()):
        raise AnalysisError(
            f"retained {name}: timing-neutral production contract failed: {contract}"
        )
    baseline, _ = replay_launch(baseline_path, 16)
    normalized = normalize_schema17(current, f"retained {name}")
    normalized_equal = normalized == baseline
    if not normalized_equal:
        raise AnalysisError(
            f"retained {name}: schema-17 replay differs from schema-16 baseline after normalization"
        )
    return {
        "cell": name,
        "route": route,
        "layout": layout,
        "depth": depth,
        "spacing": spacing,
        "groups": TRACE_GROUPS,
        "schema_version": 17,
        "cohort_candidate_completion_cycles": observation[
            "cohort_candidate_completion_cycles"
        ],
        "cohort_candidate_exposure_cycles": observation[
            "cohort_candidate_exposure_cycles"
        ],
        "production_contract": contract,
        "normalized_schema16_equal": normalized_equal,
        "replay_sha256": sha256(first),
        "baseline_sha256": sha256(baseline_path),
    }


def collect_retained_spacing(
    root: Path,
    baseline_root: Path,
) -> tuple[dict[tuple[str, str, int, int], dict[str, Any]], list[str], list[str]]:
    cells: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    missing: list[str] = []
    errors: list[str] = []
    for spec in spacing_specs():
        name = spacing_cell_name(*spec)
        first = root / "retained_spacing" / name / "replay-1.json"
        second = root / "retained_spacing" / name / "replay-2.json"
        baseline = baseline_root / name / "replay-1.json"
        absent = [path for path in (first, second, baseline) if not path.is_file()]
        if absent:
            missing.append(name)
            continue
        try:
            cells[spec] = retained_spacing_cell(root, baseline_root, spec)
        except (AnalysisError, OSError, KeyError, IndexError, TypeError, ValueError) as error:
            errors.append(str(error))
    return cells, missing, errors


def sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise AnalysisError("cannot summarize an empty comparison split")
    absolute_errors = [float(row["absolute_error"]) for row in rows]
    return {
        "comparisons": len(rows),
        "mae": statistics.mean(absolute_errors),
        "maximum_absolute_error": max(absolute_errors),
        "all_signs_correct": all(bool(row["median_sign_correct"]) for row in rows),
    }


def evaluate_pairing_candidate(
    direct: Mapping[tuple[int, int], Mapping[str, Any]],
    simulator: Mapping[tuple[str, int, int], Mapping[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for depth, spacing in direct_specs():
        live = direct[(depth, spacing)]
        expensive = simulator[("slow_expensive", depth, spacing)]
        cheap = simulator[("slow_cheap", depth, spacing)]
        predicted = (
            int(expensive["cohort_candidate_exposure_cycles"])
            - int(cheap["cohort_candidate_exposure_cycles"])
        ) / TRACE_GROUPS
        observed = float(live["median_wait_delta_cycles_per_group"])
        predicted_sign = sign(predicted)
        repeat_values = [float(value) for value in live["wait_delta_cycles_per_group"]]
        sign_repeats = sum(sign(value) == predicted_sign and predicted_sign != 0 for value in repeat_values)
        split = "identification" if spacing == 0 and depth in (1, 2) else "holdout"
        rows.append(
            {
                "depth": depth,
                "spacing": spacing,
                "split": split,
                "observed_delta_cycles_per_group": observed,
                "predicted_delta_cycles_per_group": predicted,
                "signed_error": predicted - observed,
                "absolute_error": abs(predicted - observed),
                "predicted_sign": predicted_sign,
                "median_sign_correct": sign(observed) == predicted_sign and predicted_sign != 0,
                "repeat_signs_correct": sign_repeats,
                "material_signal": abs(observed) >= MATERIAL_DELTA_MINIMUM
                and sign_repeats >= MATERIAL_SIGN_REPEATS
                and predicted_sign != 0,
            }
        )
    identification = [row for row in rows if row["split"] == "identification"]
    holdout = [row for row in rows if row["split"] == "holdout"]
    identification_metrics = metric_summary(identification)
    holdout_metrics = metric_summary(holdout)
    material = all(bool(row["material_signal"]) for row in identification)
    accuracy = {
        "identification_mae_within_limit": identification_metrics["mae"]
        <= IDENTIFICATION_MAE_LIMIT,
        "identification_max_within_limit": identification_metrics[
            "maximum_absolute_error"
        ]
        <= IDENTIFICATION_MAX_ERROR_LIMIT,
        "holdout_mae_within_limit": holdout_metrics["mae"] <= HOLDOUT_MAE_LIMIT,
        "holdout_max_within_limit": holdout_metrics["maximum_absolute_error"]
        <= HOLDOUT_MAX_ERROR_LIMIT,
        "correct_sign_every_comparison": all(
            bool(row["median_sign_correct"]) for row in rows
        ),
    }
    return {
        "model": (
            "max_c(max(ldgsts_completion_ready(last_response_c,payload_work_c),"
            " first_response_c + 4*(bypass_wavefronts_c-1) + "
            "2*(bank_wavefronts_c-1)))"
        ),
        "fitted_parameters": 0,
        "identification_slice": "depth 1/2, spacing 0",
        "holdout_slice": "depth 4/8 spacing 0 and every depth at spacing 4",
        "limits_cycles_per_group": {
            "identification_mae": IDENTIFICATION_MAE_LIMIT,
            "identification_max_absolute_error": IDENTIFICATION_MAX_ERROR_LIMIT,
            "holdout_mae": HOLDOUT_MAE_LIMIT,
            "holdout_max_absolute_error": HOLDOUT_MAX_ERROR_LIMIT,
            "material_absolute_median": MATERIAL_DELTA_MINIMUM,
            "material_sign_repeats": MATERIAL_SIGN_REPEATS,
        },
        "identification": identification_metrics,
        "holdout": holdout_metrics,
        "material_signal_on_both_identification_depths": material,
        "accuracy_gates": accuracy,
        "all_accuracy_gates_pass": all(accuracy.values()),
        "comparisons": rows,
    }


def spacing_live_lookup(report: Mapping[str, Any]) -> dict[tuple[str, str, int, int], float]:
    pairs = report.get("live_pairs")
    if not isinstance(pairs, list):
        raise AnalysisError("retained issue-spacing report has no live_pairs")
    result: dict[tuple[str, str, int, int], float] = {}
    for item in pairs:
        if not isinstance(item, dict):
            continue
        try:
            key = (
                str(item["route"]),
                str(item["layout"]),
                int(item["depth"]),
                int(item["spacing"]),
            )
            result[key] = float(item["median_per_group"])
        except (KeyError, TypeError, ValueError):
            continue
    return result


def evaluate_retained_spacing_candidate(
    cells: Mapping[tuple[str, str, int, int], Mapping[str, Any]],
    spacing_root: Path,
) -> dict[str, Any]:
    report_path = spacing_root / "analysis/report.json"
    if not report_path.is_file():
        raise AnalysisError(f"retained issue-spacing report missing: {report_path}")
    prior = read_json(report_path)
    live = spacing_live_lookup(prior)
    specs = [
        ("l2", layout, 2, spacing)
        for layout in ("conflict", "severe", "skew")
        for spacing in (0, 1, 4, 16)
    ]
    specs.extend(("l2", "skew", 8, spacing) for spacing in (0, 1, 4))
    rows: list[dict[str, Any]] = []
    for route, layout, depth, spacing in specs:
        compact = cells[(route, "compact", depth, spacing)]
        treatment = cells[(route, layout, depth, spacing)]
        key = (route, layout, depth, spacing)
        if key not in live:
            raise AnalysisError(f"retained issue-spacing live comparison missing: {key}")
        predicted = (
            int(treatment["cohort_candidate_exposure_cycles"])
            - int(compact["cohort_candidate_exposure_cycles"])
        ) / TRACE_GROUPS
        observed = live[key]
        calibration = depth == 2 and layout in ("conflict", "severe") and spacing in (1, 4)
        rows.append(
            {
                "layout": layout,
                "depth": depth,
                "spacing": spacing,
                "split": "structural_calibration" if calibration else "retained_transfer",
                "observed_delta_cycles_per_group": observed,
                "predicted_delta_cycles_per_group": predicted,
                "signed_error": predicted - observed,
                "absolute_error": abs(predicted - observed),
                "median_sign_correct": sign(predicted) != 0 and sign(predicted) == sign(observed),
            }
        )
    calibration_rows = [row for row in rows if row["split"] == "structural_calibration"]
    transfer_rows = [row for row in rows if row["split"] == "retained_transfer"]
    calibration = metric_summary(calibration_rows)
    transfer = metric_summary(transfer_rows)
    return {
        "source_report": str(report_path),
        "source_report_sha256": sha256(report_path),
        "source_report_status": prior.get("status"),
        "uniform_cohort_only_not_timing_authority": True,
        "structural_calibration": calibration,
        "retained_transfer": transfer,
        "within_one_two_cycle_limits": calibration["mae"] <= 1.0
        and calibration["maximum_absolute_error"] <= 2.0
        and transfer["mae"] <= 1.0
        and transfer["maximum_absolute_error"] <= 2.0
        and calibration["all_signs_correct"]
        and transfer["all_signs_correct"],
        "comparisons": rows,
    }


def precursor_report(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        return {"present": False, "root": str(root)}
    sidecars = sorted(root.glob("nvbit/*/extra_info/cross_allocation_lane_span.json"))
    nonempty: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in sidecars:
        try:
            entries = read_json_list(path)
        except AnalysisError as error:
            errors.append(str(error))
            continue
        if entries:
            nonempty.append(
                {
                    "cell": path.parents[1].name,
                    "entries": len(entries),
                    "sha256": sha256(path),
                }
            )
    return {
        "present": True,
        "root": str(root.resolve()),
        "cross_allocation_sidecars": len(sidecars),
        "nonempty_cross_allocation_sidecars": len(nonempty),
        "examples": nonempty[:4],
        "errors": errors,
        "status": (
            "invalid_precursor_cross_allocation_lane_span"
            if nonempty
            else "precursor_reason_not_observed"
        ),
        "excluded_from_final_evidence": bool(nonempty),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    counts = report["counts"]
    criteria = report["criteria"]
    lines = [
        "# SM86 async-copy cohort-pairing report",
        "",
        f"Status: `{report['status']}`",
        "",
        str(report["decision"]),
        "",
        "## Evidence coverage",
        "",
        "| Lane | Complete | Expected |",
        "|---|---:|---:|",
        f"| Direct paired comparisons | {counts['direct_cells']} | {counts['direct_expected']} |",
        f"| Direct paired executions | {counts['direct_repeat_results']} | {counts['direct_repeat_expected']} |",
        f"| NCU endpoints | {counts['ncu_cells']} | {counts['ncu_expected']} |",
        f"| NVBit endpoint traces | {counts['nvbit_cells']} | {counts['nvbit_expected']} |",
        f"| Pairing schema-17 twins | {counts['simulator_cells']} | {counts['simulator_expected']} |",
        f"| Retained spacing schema-17 twins | {counts['retained_spacing_cells']} | {counts['retained_spacing_expected']} |",
        "",
        "## Admission and transfer gates",
        "",
        "| Criterion | Result |",
        "|---|---|",
    ]
    for name, value in criteria.items():
        shown = "pending" if value is None else ("pass" if value else "fail")
        lines.append(f"| `{name}` | {shown} |")

    candidate = report.get("candidate_evaluation")
    if isinstance(candidate, dict):
        lines.extend(
            [
                "",
                "## Parameter-free cohort candidate",
                "",
                "| Split | MAE | Maximum absolute error | All signs correct |",
                "|---|---:|---:|---|",
            ]
        )
        for key, label in (("identification", "Identification"), ("holdout", "Holdout")):
            value = candidate[key]
            lines.append(
                f"| {label} | {value['mae']:.3f} | "
                f"{value['maximum_absolute_error']:.3f} | "
                f"{'yes' if value['all_signs_correct'] else 'no'} |"
            )
        lines.extend(
            [
                "",
                "| Depth | Spacing | Split | Live delta | Predicted delta | Absolute error | Sign repeats |",
                "|---:|---:|---|---:|---:|---:|---:|",
            ]
        )
        for row in candidate["comparisons"]:
            lines.append(
                f"| {row['depth']} | {row['spacing']} | {row['split']} | "
                f"{row['observed_delta_cycles_per_group']:.3f} | "
                f"{row['predicted_delta_cycles_per_group']:.3f} | "
                f"{row['absolute_error']:.3f} | {row['repeat_signs_correct']}/9 |"
            )

    precursor = report.get("invalid_precursor")
    if isinstance(precursor, dict) and precursor.get("present"):
        lines.extend(
            [
                "",
                "## Invalid precursor",
                "",
                f"The v1 precursor is recorded as `{precursor.get('status')}` and is excluded "
                "from final evidence.",
            ]
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
            "Schema-17 cohort fields are timing-neutral observations. Unless every gate "
            "passes, production readiness and the typed async-copy visibility tail remain unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def optional_binary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "present": False}
    return {
        "path": str(path),
        "present": True,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def run(
    root: Path,
    spacing_baseline_root: Path,
    precursor_root: Path,
    output_json: Path,
    output_md: Path,
) -> int:
    root = root.resolve()
    spacing_baseline_root = spacing_baseline_root.resolve()
    precursor_root = precursor_root.resolve()
    spacing_root = spacing_baseline_root.parent
    missing: list[str] = []
    errors: list[str] = []

    try:
        host = host_report(root)
    except (AnalysisError, OSError, ValueError) as error:
        host = {"valid": False, "error": str(error)}
        if "missing" in str(error):
            missing.append(str(error))
        else:
            errors.append(str(error))

    try:
        static = static_report(root)
    except (AnalysisError, OSError, KeyError, TypeError, ValueError) as error:
        static = {"valid": False, "error": str(error)}
        if "missing" in str(error):
            missing.append(str(error))
        else:
            errors.append(str(error))

    direct, direct_missing, direct_unexpected, direct_errors = collect_direct(root)
    ncu, ncu_missing, ncu_errors = collect_ncu(root)
    nvbit, nvbit_missing, nvbit_errors = collect_nvbit(root)
    simulator, simulator_missing, simulator_errors = collect_simulator(root)
    retained, retained_missing, retained_errors = collect_retained_spacing(
        root, spacing_baseline_root
    )
    missing.extend(f"direct/{name}" for name in direct_missing)
    missing.extend(f"ncu/{name}" for name in ncu_missing)
    missing.extend(f"nvbit/{name}" for name in nvbit_missing)
    missing.extend(f"simulator/{name}" for name in simulator_missing)
    missing.extend(f"retained_spacing/{name}" for name in retained_missing)
    errors.extend(
        direct_errors
        + ncu_errors
        + nvbit_errors
        + simulator_errors
        + retained_errors
    )
    if direct_unexpected:
        errors.append(f"unexpected direct directories: {direct_unexpected}")

    ncu_pairs = ncu_pair_reports(ncu)
    nvbit_binary_hashes = {item["binary_sha256"] for item in nvbit.values()}
    nvbit_tracer_hashes = {item["tracer_sha256"] for item in nvbit.values()}

    candidate: dict[str, Any] | None = None
    if set(direct) == set(direct_specs()) and set(simulator) == set(endpoint_specs()):
        try:
            candidate = evaluate_pairing_candidate(direct, simulator)
        except (AnalysisError, KeyError, TypeError, ValueError, statistics.StatisticsError) as error:
            errors.append(f"pairing candidate evaluation failed: {error}")

    spacing_candidate: dict[str, Any] | None = None
    if set(retained) == set(spacing_specs()):
        try:
            spacing_candidate = evaluate_retained_spacing_candidate(retained, spacing_root)
        except (AnalysisError, KeyError, TypeError, ValueError, statistics.StatisticsError) as error:
            errors.append(f"retained spacing candidate evaluation failed: {error}")

    direct_complete = set(direct) == set(direct_specs())
    ncu_complete = set(ncu) == set(endpoint_specs())
    nvbit_complete = set(nvbit) == set(endpoint_specs())
    simulator_complete = set(simulator) == set(endpoint_specs())
    retained_complete = set(retained) == set(spacing_specs())
    direct_issue_gate = (
        all(bool(cell["issue_delta_gate"]) for cell in direct.values())
        if direct_complete
        else None
    )
    ncu_endpoint_gate = (
        all(bool(cell["contract_valid"]) for cell in ncu.values())
        if ncu_complete
        else None
    )
    ncu_pair_gate = (
        len(ncu_pairs) == len(direct_specs())
        and all(bool(item["valid"]) for item in ncu_pairs)
        if ncu_complete
        else None
    )
    simulator_neutral = (
        all(all(cell["production_contract"].values()) for cell in simulator.values())
        if simulator_complete
        else None
    )
    retained_normalized = (
        all(bool(cell["normalized_schema16_equal"]) for cell in retained.values())
        if retained_complete
        else None
    )
    criteria: dict[str, bool | None] = {
        "host_gpu_identity_pinned": host.get("valid") is True,
        "static_8_function_contract_valid": static.get("valid") is True,
        "direct_8_comparisons_72_runs_correct": direct_complete,
        "direct_issue_delta_gate": direct_issue_gate,
        "ncu_16_endpoints_exact_work_and_nonzero_mix": ncu_endpoint_gate,
        "ncu_all_paired_source_metrics_within_tolerance": ncu_pair_gate,
        "nvbit_16_complete_empty_exclusions_one_binary": (
            nvbit_complete
            and len(nvbit_binary_hashes) == 1
            and len(nvbit_tracer_hashes) == 1
        ),
        "schema17_pairing_twins_complete_timing_neutral": (
            simulator_complete and simulator_neutral is True
        ),
        "retained_spacing_22_schema17_twins_normalize_to_schema16": (
            retained_complete and retained_normalized is True
        ),
        "identification_material_signal": (
            candidate["material_signal_on_both_identification_depths"]
            if candidate is not None
            else None
        ),
        "identification_mae_and_max_within_limits": (
            candidate["accuracy_gates"]["identification_mae_within_limit"]
            and candidate["accuracy_gates"]["identification_max_within_limit"]
            if candidate is not None
            else None
        ),
        "holdout_mae_and_max_within_limits": (
            candidate["accuracy_gates"]["holdout_mae_within_limit"]
            and candidate["accuracy_gates"]["holdout_max_within_limit"]
            if candidate is not None
            else None
        ),
        "correct_sign_every_admitted_comparison": (
            candidate["accuracy_gates"]["correct_sign_every_comparison"]
            if candidate is not None
            else None
        ),
    }

    admission_names = (
        "host_gpu_identity_pinned",
        "static_8_function_contract_valid",
        "direct_8_comparisons_72_runs_correct",
        "direct_issue_delta_gate",
        "ncu_16_endpoints_exact_work_and_nonzero_mix",
        "ncu_all_paired_source_metrics_within_tolerance",
        "nvbit_16_complete_empty_exclusions_one_binary",
        "schema17_pairing_twins_complete_timing_neutral",
        "retained_spacing_22_schema17_twins_normalize_to_schema16",
    )
    transfer_names = (
        "identification_material_signal",
        "identification_mae_and_max_within_limits",
        "holdout_mae_and_max_within_limits",
        "correct_sign_every_admitted_comparison",
    )
    complete = not missing and candidate is not None and spacing_candidate is not None
    admission_passed = complete and all(criteria[name] is True for name in admission_names)
    transfer_passed = candidate is not None and all(
        criteria[name] is True for name in transfer_names
    )
    timing_promotion_authorized = admission_passed and transfer_passed and not errors

    if errors:
        status = "invalid_cohort_pairing_campaign_no_timing_promotion"
        decision = (
            "The assay is invalid because one or more retained artifacts violate their "
            "structural contract. Production timing and the typed visibility tail remain unchanged."
        )
        exit_code = 1
    elif not complete:
        status = "incomplete_cohort_pairing_campaign"
        decision = (
            "The reducer is installed, but one or more live, trace, replay, or schema-16 "
            "normalization lanes are still missing."
        )
        exit_code = 2
    elif not admission_passed:
        status = "invalid_cohort_pairing_assay_no_timing_promotion"
        decision = (
            "The completed matched-marginal assay failed at least one admission invariant; "
            "do not relabel the source mix or promote timing, and retain the typed tail."
        )
        exit_code = 1
    elif not transfer_passed:
        status = "pass_cohort_pairing_no_timing_promotion"
        decision = (
            "The parameter-free cohort/source-overlap candidate fails the predeclared "
            "identification or holdout gates. Production timing and the typed visibility "
            "tail remain unchanged."
        )
        exit_code = 0
    else:
        status = "pass_cohort_pairing_timing_promotion_authorized"
        decision = (
            "Every structural, material-signal, identification, holdout, sign, and "
            "normalization gate passes. The exact parameter-free cohort candidate is "
            "eligible for the separately validated activation slice."
        )
        exit_code = 0

    report = {
        "schema_version": 1,
        "root": str(root),
        "spacing_baseline_root": str(spacing_baseline_root),
        "status": status,
        "decision": decision,
        "timing_promotion_authorized": timing_promotion_authorized,
        "typed_visibility_tail_required": not timing_promotion_authorized,
        "assay": {
            "direct_groups": DIRECT_GROUPS,
            "direct_repeats": DIRECT_REPEATS,
            "trace_groups": TRACE_GROUPS,
            "variants": list(VARIANTS),
            "depths": list(DEPTHS),
            "spacing_samples": list(SPACINGS),
            "identification_slice": "depth 1/2 at spacing 0",
            "holdout_slice": "depth 4/8 at spacing 0 and all depths at spacing 4",
            "paired_source_tolerance": "max(4, ceil(0.01 * expected_source_sectors))",
        },
        "counts": {
            "direct_cells": len(direct),
            "direct_expected": len(direct_specs()),
            "direct_repeat_results": sum(cell["repeat_count"] for cell in direct.values()),
            "direct_repeat_expected": len(direct_specs()) * DIRECT_REPEATS,
            "ncu_cells": len(ncu),
            "ncu_expected": len(endpoint_specs()),
            "nvbit_cells": len(nvbit),
            "nvbit_expected": len(endpoint_specs()),
            "simulator_cells": len(simulator),
            "simulator_expected": len(endpoint_specs()),
            "retained_spacing_cells": len(retained),
            "retained_spacing_expected": len(spacing_specs()),
        },
        "binaries": {
            "probe": optional_binary(
                REPO_ROOT / "workloads/research_probes/async_source_only/async_cohort_pairing"
            ),
            "simulator": optional_binary(REPO_ROOT / "sim/target/release/sim-run"),
        },
        "host": host,
        "static": static,
        "direct_cells": [direct[key] for key in sorted(direct)],
        "direct_unexpected_directories": direct_unexpected,
        "ncu_cells": [ncu[key] for key in sorted(ncu)],
        "ncu_pairs": ncu_pairs,
        "nvbit_cells": [nvbit[key] for key in sorted(nvbit)],
        "simulator_cells": [simulator[key] for key in sorted(simulator)],
        "retained_spacing_cells": [retained[key] for key in sorted(retained)],
        "candidate_evaluation": candidate,
        "retained_spacing_candidate_evaluation": spacing_candidate,
        "invalid_precursor": precursor_report(precursor_root),
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
                "timing_promotion_authorized": timing_promotion_authorized,
            },
            sort_keys=True,
        )
    )
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--spacing-baseline-root", type=Path, default=DEFAULT_SPACING_BASELINE_ROOT
    )
    parser.add_argument("--precursor-root", type=Path, default=DEFAULT_PRECURSOR_ROOT)
    parser.add_argument("--output", "--output-json", dest="output_json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args(argv)
    output_json = args.output_json or args.root / "analysis/report.json"
    output_md = args.output_md or output_json.with_suffix(".md")
    try:
        return run(
            args.root,
            args.spacing_baseline_root,
            args.precursor_root,
            output_json,
            output_md,
        )
    except (AnalysisError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"cohort-pairing analysis: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
