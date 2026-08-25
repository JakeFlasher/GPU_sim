#!/usr/bin/env python3
"""Reduce the final async/barrier campaign into reproducible tables and conclusions."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_ROOT = (
    REPO_ROOT
    / "results"
    / "captures"
    / "sm86_async_barrier_readiness_20260817_v4"
)
GROUPS = 16384
TRACE_GROUPS = 64
COPIES_PER_GROUP = 8

METRICS = (
    "gpc__cycles_elapsed.max",
    "sm__cycles_active.max",
    "smsp__warps_active.avg.per_cycle_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "smsp__inst_executed.sum",
    "smsp__inst_executed_op_ldgsts.sum",
    "sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum",
    "sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    "lts__t_sectors_srcunit_tex_op_read.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
    "dram__sectors_read.sum",
)

PROTOCOL_NAMES = {
    "c00": "c00_baseline",
    "c01": "c01_barrier_only",
    "c10": "c10_wait_only",
    "c11": "c11_wait_barrier",
    "c12": "c12_producer_consumer",
}
LOCALITY_NAMES = {
    "na": "l1_reuse",
    "l1": "l1_reuse",
    "l2": "l2_rotation",
    "dram": "dram_stream",
}
EMPTY_NVBIT_DIAGNOSTICS = (
    "unsupported_memory_path",
    "unmapped_memory_range",
    "cross_allocation_lane_span",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def repeat_index(path: Path) -> int:
    return int(path.name.rsplit("-", 1)[1])


def read_wide_ncu(path: Path) -> dict[str, str]:
    rows = list(csv.reader(path.open(newline="", encoding="utf-8")))
    if len(rows) < 3 or len(rows[0]) != len(rows[2]):
        raise RuntimeError(f"malformed wide NCU CSV: {path}")
    return dict(zip(rows[0], rows[2]))


def median(rows: Iterable[dict[str, str]], metric: str) -> float:
    return statistics.median(float(row[metric]) for row in rows)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def coefficient_of_variation_pct(samples: Sequence[float]) -> float:
    mean = statistics.mean(samples)
    require(mean != 0.0, "cannot calculate coefficient of variation at zero mean")
    return statistics.stdev(samples) / mean * 100


def measurement_json_line(path: Path) -> dict[str, Any]:
    measurements: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped.startswith("{") or '"correct"' not in stripped:
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"{path}:{line_number}: malformed target JSON payload: {error}"
            ) from error
        if isinstance(value, dict) and "correct" in value:
            measurements.append(value)
    require(len(measurements) == 1, f"{path}: expected one target JSON payload")
    return measurements[0]


def normalized_csv_row(row: dict[str, str | None], path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in row.items():
        require(key is not None and value is not None, f"{path}: malformed CSV row")
        result[key.strip()] = value.strip()
    return result


def command_option(command: Sequence[str], option: str) -> str | None:
    try:
        index = command.index(option)
    except ValueError:
        return None
    if index + 1 >= len(command):
        return None
    return command[index + 1]


def resource_usage(path: Path) -> dict[str, Any]:
    functions: dict[str, dict[str, int]] = {}
    current_function: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("Function ") and line.endswith(":"):
            current_function = line.removeprefix("Function ").removesuffix(":")
            continue
        if current_function is None or not line.startswith("REG:"):
            continue
        values: dict[str, int] = {}
        for token in line.split():
            key, separator, raw_value = token.partition(":")
            if separator and raw_value.isdigit():
                values[key] = int(raw_value)
        require(
            "STACK" in values and "LOCAL" in values,
            f"{path}: missing STACK/LOCAL usage for {current_function}",
        )
        functions[current_function] = values
        current_function = None
    protocol_functions = {
        name: values
        for name, values in functions.items()
        if "async_barrier_overlap" in name
    }
    require(
        len(protocol_functions) == 5,
        f"{path}: expected five specialized protocol functions",
    )
    return {
        "file": path.name,
        "protocol_functions": len(protocol_functions),
        "all_stack_zero": all(
            values["STACK"] == 0 for values in protocol_functions.values()
        ),
        "all_local_zero": all(
            values["LOCAL"] == 0 for values in protocol_functions.values()
        ),
    }


def load_cohort_integrity(root: Path, predicted: dict[str, int]) -> dict[str, Any]:
    cohort = root / "simulator" / "cohort"
    first_replays = sorted(cohort.glob("*.replay-1.json"))
    require(len(first_replays) == 6, f"{cohort}: expected six production replays")
    derived: dict[str, int] = {}
    for first in first_replays:
        workload = first.name.removesuffix(".replay-1.json")
        second = cohort / f"{workload}.replay-2.json"
        require(second.is_file(), f"{workload}: missing second production replay")
        require(
            first.read_bytes() == second.read_bytes(),
            f"{workload}: production replay JSON is not byte-identical",
        )
        document = read_json(first)
        launches = document["launches"]
        require(len(launches) == 1, f"{workload}: expected one replayed launch")
        launch = launches[0]
        counters = launch["counters"]
        coverage = launch["coverage"]
        require(
            counters["instructions_issued"] == counters["records_seen"],
            f"{workload}: issued/record cardinality mismatch",
        )
        require(coverage["kind"] == "full", f"{workload}: replay coverage is not full")
        require(
            coverage["blocks_observed"] == coverage["population_blocks"],
            f"{workload}: observed/population block mismatch",
        )
        require(
            launch["timing"]["readiness"]["schema_version"] == 1,
            f"{workload}: expected readiness schema version 1",
        )
        derived[workload] = int(launch["timing"]["diagnostic_last_issue_cycles"])
    require(derived == predicted, "production replay cycles disagree with cohort_predicted.json")
    return {
        "workloads": len(derived),
        "byte_identical_pairs": len(derived),
        "all_cardinality_exact": True,
        "all_full_coverage": True,
        "all_readiness_schema_v1": True,
    }


def expected_ldgsts(cell: dict[str, Any]) -> int:
    if cell["protocol"] in {"c00", "c01"}:
        return 0
    participating_warps = 1 if cell["protocol"] == "c12" else cell["warps"]
    return participating_warps * GROUPS * COPIES_PER_GROUP


def source_attribution(path: Path) -> dict[str, Any]:
    raw = list(csv.reader(path.open(newline="", encoding="utf-8")))
    if len(raw) < 2:
        raise RuntimeError(f"empty SourceCounters export: {path}")
    rows = [dict(zip(raw[1], row)) for row in raw[2:]]
    result: dict[str, Any] = {}
    for short, column in (
        ("long_scoreboard", "stall_long_sb (Not Issued)"),
        ("barrier", "stall_barrier (Not Issued)"),
        ("wait", "stall_wait (Not Issued)"),
        ("not_selected", "stall_not_selected (Not Issued)"),
    ):
        values: list[tuple[int, str, str]] = []
        for row in rows:
            try:
                count = int(float(row.get(column) or 0))
            except ValueError:
                count = 0
            if count:
                values.append((count, row["Source"].strip(), row["Address"]))
        values.sort(reverse=True)
        result[short] = {
            "samples": sum(value[0] for value in values),
            "top": [
                {"samples": count, "sass": sass, "address": address}
                for count, sass, address in values[:4]
            ],
        }
    return result


def load_cells(root: Path) -> list[dict[str, Any]]:
    manifest = read_json(root / "metadata" / "matrix.json")
    require(manifest["schema_version"] == 1, "expected campaign matrix schema 1")
    configured_cells = manifest["cells"]
    configured_keys = {cell["key"] for cell in configured_cells}
    require(
        len(configured_cells) == 36 and len(configured_keys) == 36,
        "expected 36 unique campaign cells",
    )
    for subtree in ("correctness", "ncu", "nvbit"):
        actual = {
            path.name for path in (root / subtree).iterdir() if path.is_dir()
        }
        require(
            actual == configured_keys,
            f"{subtree}: directory set does not match the campaign matrix",
        )
    simulator_directories = {
        path.name for path in (root / "simulator").iterdir() if path.is_dir()
    }
    require(
        simulator_directories == configured_keys | {"cohort"},
        "simulator: directory set does not match matrix plus production cohort",
    )

    result: list[dict[str, Any]] = []
    for configured in configured_cells:
        key = configured["key"]
        repeat_dirs = sorted((root / "ncu" / key).glob("repeat-*"), key=repeat_index)
        for repeat_directory in repeat_dirs:
            ncu_run = read_json(repeat_directory / "run.json")
            command = ncu_run["command"]
            require(
                ncu_run["return_code"] == 0
                and command_option(command, "--profile-from-start") == "off"
                and command_option(command, "--launch-skip") == "0"
                and command_option(command, "--launch-count") == "1"
                and command_option(command, "--replay-mode") == "application"
                and command_option(command, "--clock-control") == "base"
                and command_option(command, "--cache-control") == "none"
                and command_option(command, "--metrics") == ",".join(METRICS),
                f"{key}: NCU capture contract mismatch in {repeat_directory.name}",
            )
            require(
                command[-5:]
                == [
                    configured["protocol"],
                    configured["locality"],
                    str(configured["gap"]),
                    str(configured["warps"]),
                    "1",
                ],
                f"{key}: NCU target arguments disagree with matrix cell",
            )
            report = repeat_directory / "ncu-report.ncu-rep"
            require(
                report.is_file() and report.stat().st_size > 0,
                f"{key}: missing nonempty NCU report in {repeat_directory.name}",
            )
        ncu_rows = [read_wide_ncu(path / "ncu.csv") for path in repeat_dirs]
        require(len(ncu_rows) in {3, 5}, f"{key}: expected three or five NCU repeats")
        require(
            [repeat_index(path) for path in repeat_dirs]
            == list(range(1, len(repeat_dirs) + 1)),
            f"{key}: NCU repeat numbering is not contiguous",
        )
        values = {metric: median(ncu_rows, metric) for metric in METRICS}
        active_samples = [float(row["sm__cycles_active.max"]) for row in ncu_rows]
        first_three_cv = coefficient_of_variation_pct(active_samples[:3])
        require(
            first_three_cv <= 2.0 or len(ncu_rows) == 5,
            f"{key}: >2% initial NCU CV without fourth/fifth repeats",
        )

        correctness_document = read_json(
            root / "correctness" / key / "result.json"
        )
        require(
            correctness_document["cell"]["key"] == key,
            f"{key}: correctness cell identity mismatch",
        )
        correctness = correctness_document["measurement"]
        require(correctness["correct"] is True, f"{key}: correctness run failed")
        require(
            correctness["protocol"] == PROTOCOL_NAMES[configured["protocol"]]
            and correctness["locality"] == LOCALITY_NAMES[configured["locality"]]
            and correctness["gap_cycles"] == configured["gap"]
            and correctness["warps"] == configured["warps"]
            and correctness["groups"] == GROUPS
            and correctness["copies_per_group"] == COPIES_PER_GROUP,
            f"{key}: correctness payload does not match the matrix cell",
        )
        correctness_samples = correctness["samples_cycles"]
        require(
            len(correctness_samples) == 7
            and all(len(sample) == configured["warps"] for sample in correctness_samples),
            f"{key}: expected seven complete correctness trials",
        )

        nvbit = root / "nvbit" / key
        stats_path = nvbit / "stats.csv"
        stats_rows = list(
            csv.DictReader(stats_path.open(newline="", encoding="utf-8"))
        )
        require(len(stats_rows) == 1, f"{key}: expected one NVBit kernel row")
        stats = normalized_csv_row(stats_rows[0], stats_path)
        nvbit_blocks = int(stats["#blocks"])
        nvbit_threads = int(stats["#threads"])
        nvbit_total = int(stats["total_insts"])
        nvbit_reported = int(stats["total_reported_insts"])
        require(
            int(stats["grid_dimX"]) == 1
            and int(stats["grid_dimY"]) == 1
            and int(stats["grid_dimZ"]) == 1
            and nvbit_blocks == 1
            and int(stats["block_dimX"]) == configured["warps"] * 32
            and int(stats["block_dimY"]) == 1
            and int(stats["block_dimZ"]) == 1
            and nvbit_threads == configured["warps"] * 32,
            f"{key}: NVBit capture is not exactly one complete configured CTA",
        )
        require(
            nvbit_total > 0 and nvbit_total == nvbit_reported,
            f"{key}: NVBit total/reported instruction mismatch",
        )

        payload = measurement_json_line(nvbit / "stdout.txt")
        require(payload["correct"] is True, f"{key}: NVBit target payload failed")
        require(
            payload["protocol"] == PROTOCOL_NAMES[configured["protocol"]]
            and payload["locality"] == LOCALITY_NAMES[configured["locality"]]
            and payload["gap_cycles"] == configured["gap"]
            and payload["warps"] == configured["warps"]
            and payload["groups"] == TRACE_GROUPS
            and payload["copies_per_group"] == COPIES_PER_GROUP,
            f"{key}: NVBit target payload does not match the matrix cell",
        )
        require(
            len(payload["samples_cycles"]) == 1
            and len(payload["samples_cycles"][0]) == configured["warps"],
            f"{key}: NVBit target payload is not one complete trial",
        )

        diagnostic_counts: dict[str, int] = {}
        for name in EMPTY_NVBIT_DIAGNOSTICS:
            diagnostic_path = nvbit / "extra_info" / f"{name}.json"
            diagnostic = read_json(diagnostic_path)
            require(
                isinstance(diagnostic, list),
                f"{key}: {diagnostic_path.name} is not a JSON array",
            )
            diagnostic_counts[name] = len(diagnostic)
        require(
            sum(diagnostic_counts.values()) == 0,
            f"{key}: NVBit diagnostic sidecars contain unsupported records",
        )

        trace_path = nvbit / "dynamic_trace.pb"
        capture = read_json(nvbit / "capture.json")
        run = read_json(nvbit / "run.json")
        require(run["return_code"] == 0, f"{key}: NVBit capture command failed")
        require(
            capture["cell"]["key"] == key,
            f"{key}: NVBit capture cell identity mismatch",
        )
        require(
            trace_path.stat().st_size > 0
            and capture["dynamic_trace_bytes"] == trace_path.stat().st_size,
            f"{key}: NVBit trace size disagrees with capture metadata",
        )
        trace_selection = read_json(
            nvbit / "extra_info" / "traced_kernel_modules.json"
        )["selection"]
        require(
            trace_selection["exact_contract_satisfied"] is True
            and trace_selection["accepted_active_launches"] == 1
            and trace_selection["accepted_channel_records"] == nvbit_total,
            f"{key}: NVBit active-launch selection contract mismatch",
        )

        simulator_directory = root / "simulator" / key
        replay_one_path = simulator_directory / "replay-1.json"
        replay_two_path = simulator_directory / "replay-2.json"
        require(
            replay_one_path.read_bytes() == replay_two_path.read_bytes(),
            f"{key}: simulator replay JSON is not byte-identical",
        )
        simulator_document = read_json(replay_one_path)
        require(
            len(simulator_document["launches"]) == 1,
            f"{key}: expected one simulated target launch",
        )
        simulator = simulator_document["launches"][0]
        replay_summary = read_json(simulator_directory / "replay.json")
        simulator_counters = simulator["counters"]
        simulator_coverage = simulator["coverage"]
        readiness = simulator["timing"]["readiness"]
        require(
            simulator_counters["instructions_issued"]
            == simulator_counters["records_seen"],
            f"{key}: simulator issued/record cardinality mismatch",
        )
        require(
            simulator_coverage["kind"] == "full"
            and simulator_coverage["blocks_observed"] == 1
            and simulator_coverage["population_blocks"] == 1,
            f"{key}: simulator coverage is not one full CTA",
        )
        require(
            readiness["schema_version"] == 1,
            f"{key}: expected readiness schema version 1",
        )
        require(
            replay_summary["byte_identical"] is True
            and replay_summary["cell"]["key"] == key
            and replay_summary["instructions_issued"]
            == simulator_counters["instructions_issued"]
            and replay_summary["records_seen"] == simulator_counters["records_seen"]
            and replay_summary["coverage"] == simulator_coverage
            and replay_summary["readiness_schema_version"] == 1,
            f"{key}: simulator replay summary disagrees with replay JSON",
        )

        readiness = simulator["timing"]["readiness"]
        row: dict[str, Any] = {
            **configured,
            "ncu_repeats": len(ncu_rows),
            "ncu_initial_active_cycles_cv_pct": first_three_cv,
            "ncu_active_cycles_cv_pct": coefficient_of_variation_pct(active_samples),
            "ncu_active_cycles_per_group": values["sm__cycles_active.max"] / GROUPS,
            "ncu_metrics": values,
            "correct": correctness["correct"],
            "correctness_trials": len(correctness_samples),
            "nvbit_blocks": nvbit_blocks,
            "nvbit_threads": nvbit_threads,
            "nvbit_payload_correct": payload["correct"],
            "nvbit_payload_trials": len(payload["samples_cycles"]),
            "nvbit_diagnostic_counts": diagnostic_counts,
            "nvbit_trace_bytes": trace_path.stat().st_size,
            "nvbit_total_instructions": nvbit_total,
            "nvbit_reported_instructions": nvbit_reported,
            "simulator_cycles": simulator["timing"]["diagnostic_last_issue_cycles"],
            "simulator_cycles_per_group": simulator["timing"]["diagnostic_last_issue_cycles"]
            / TRACE_GROUPS,
            "simulator_instructions_issued": simulator_counters[
                "instructions_issued"
            ],
            "simulator_records_seen": simulator_counters["records_seen"],
            "simulator_coverage": simulator_coverage,
            "simulator_readiness_schema_version": readiness["schema_version"],
            "simulator_byte_identical": replay_summary["byte_identical"],
            "simulator_readiness": readiness,
            "simulator_replay": replay_summary,
            "expected_ldgsts": expected_ldgsts(configured),
        }
        row["dynamic_work_invariant"] = (
            round(values["smsp__inst_executed_op_ldgsts.sum"])
            == row["expected_ldgsts"]
            and round(
                values[
                    "sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum"
                ]
            )
            == row["expected_ldgsts"] * 4
        )
        require(
            row["dynamic_work_invariant"],
            f"{key}: NCU LDGSTS/destination-work invariant failed",
        )
        result.append(row)
    return result


def keyed(cells: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {cell["key"]: cell for cell in cells}


def cycles(cells: dict[str, dict[str, Any]], key: str) -> float:
    return cells[key]["ncu_active_cycles_per_group"]


def interaction(
    cells: dict[str, dict[str, Any]], warps: int, locality: str, gap: int
) -> dict[str, float]:
    barrier = cycles(cells, f"w{warps}_c01_na_g{gap}") - cycles(
        cells, f"w{warps}_c00_na_g{gap}"
    )
    coupled = cycles(cells, f"w{warps}_c11_{locality}_g{gap}") - cycles(
        cells, f"w{warps}_c10_{locality}_g{gap}"
    )
    return {
        "barrier_only_cycles_per_group": barrier,
        "coupled_minus_wait_cycles_per_group": coupled,
        "interaction_cycles_per_group": coupled - barrier,
    }


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def format_top(entry: dict[str, Any]) -> str:
    top = entry["top"]
    if not top:
        return "none"
    return f"{top[0]['samples']} @ `{top[0]['sass']}`"


def write_cells_csv(path: Path, cells: Sequence[dict[str, Any]]) -> None:
    fields = [
        "key",
        "stage",
        "warps",
        "protocol",
        "locality",
        "gap",
        "ncu_repeats",
        "ncu_initial_active_cycles_cv_pct",
        "ncu_active_cycles_per_group",
        "ncu_active_cycles_cv_pct",
        "long_scoreboard_ratio",
        "barrier_ratio",
        "wait_ratio",
        "not_selected_ratio",
        "ldgsts_instructions",
        "ldgsts_destination_wavefronts",
        "l1_lookup_misses",
        "l2_lookup_hits",
        "l2_lookup_misses",
        "dram_read_sectors",
        "correctness_trials",
        "nvbit_blocks",
        "nvbit_threads",
        "nvbit_payload_correct",
        "nvbit_diagnostic_records",
        "nvbit_total_instructions",
        "nvbit_reported_instructions",
        "simulator_cycles_per_group",
        "simulator_instructions_issued",
        "simulator_records_seen",
        "simulator_coverage_kind",
        "simulator_readiness_schema_version",
        "simulator_byte_identical",
        "simulator_cta_arrival_spread",
        "simulator_cta_rendezvous_warp_cycles",
        "simulator_eligible_not_selected_cycles",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cell in cells:
            metrics = cell["ncu_metrics"]
            readiness = cell["simulator_readiness"]
            writer.writerow(
                {
                    "key": cell["key"],
                    "stage": cell["stage"],
                    "warps": cell["warps"],
                    "protocol": cell["protocol"],
                    "locality": cell["locality"],
                    "gap": cell["gap"],
                    "ncu_repeats": cell["ncu_repeats"],
                    "ncu_initial_active_cycles_cv_pct": cell[
                        "ncu_initial_active_cycles_cv_pct"
                    ],
                    "ncu_active_cycles_per_group": cell["ncu_active_cycles_per_group"],
                    "ncu_active_cycles_cv_pct": cell["ncu_active_cycles_cv_pct"],
                    "long_scoreboard_ratio": metrics[
                        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"
                    ],
                    "barrier_ratio": metrics[
                        "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio"
                    ],
                    "wait_ratio": metrics[
                        "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"
                    ],
                    "not_selected_ratio": metrics[
                        "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio"
                    ],
                    "ldgsts_instructions": metrics["smsp__inst_executed_op_ldgsts.sum"],
                    "ldgsts_destination_wavefronts": metrics[
                        "sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum"
                    ],
                    "l1_lookup_misses": metrics[
                        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum"
                    ],
                    "l2_lookup_hits": metrics[
                        "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum"
                    ],
                    "l2_lookup_misses": metrics[
                        "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum"
                    ],
                    "dram_read_sectors": metrics["dram__sectors_read.sum"],
                    "correctness_trials": cell["correctness_trials"],
                    "nvbit_blocks": cell["nvbit_blocks"],
                    "nvbit_threads": cell["nvbit_threads"],
                    "nvbit_payload_correct": cell["nvbit_payload_correct"],
                    "nvbit_diagnostic_records": sum(
                        cell["nvbit_diagnostic_counts"].values()
                    ),
                    "nvbit_total_instructions": cell["nvbit_total_instructions"],
                    "nvbit_reported_instructions": cell[
                        "nvbit_reported_instructions"
                    ],
                    "simulator_cycles_per_group": cell["simulator_cycles_per_group"],
                    "simulator_instructions_issued": cell[
                        "simulator_instructions_issued"
                    ],
                    "simulator_records_seen": cell["simulator_records_seen"],
                    "simulator_coverage_kind": cell["simulator_coverage"]["kind"],
                    "simulator_readiness_schema_version": cell[
                        "simulator_readiness_schema_version"
                    ],
                    "simulator_byte_identical": cell["simulator_byte_identical"],
                    "simulator_cta_arrival_spread": readiness["cta_barriers"][
                        "total_arrival_spread_cycles"
                    ],
                    "simulator_cta_rendezvous_warp_cycles": readiness["cta_barriers"][
                        "total_rendezvous_warp_cycles"
                    ],
                    "simulator_eligible_not_selected_cycles": readiness[
                        "limiting_cycles"
                    ]["eligible_not_selected"],
                }
            )


def analyze(root: Path) -> dict[str, Any]:
    cells = load_cells(root)
    by_key = keyed(cells)
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    write_cells_csv(output / "cells.csv", cells)

    source: dict[str, Any] = {}
    for directory in sorted((root / "ncu-source").iterdir()):
        if directory.is_dir():
            require(
                directory.name in by_key,
                f"unexpected SourceCounters cell: {directory.name}",
            )
            source_run = read_json(directory / "run.json")
            source_command = source_run["command"]
            configured = by_key[directory.name]
            require(
                source_run["return_code"] == 0
                and command_option(source_command, "--profile-from-start") == "off"
                and command_option(source_command, "--launch-count") == "1"
                and command_option(source_command, "--replay-mode") == "application"
                and command_option(source_command, "--clock-control") == "base"
                and command_option(source_command, "--cache-control") == "none"
                and command_option(source_command, "--section") == "SourceCounters"
                and source_command[-5:]
                == [
                    configured["protocol"],
                    configured["locality"],
                    str(configured["gap"]),
                    str(configured["warps"]),
                    "1",
                ],
                f"{directory.name}: SourceCounters capture contract mismatch",
            )
            source[directory.name] = source_attribution(directory / "source.csv")
    require(len(source) == 7, "expected seven SourceCounters sentinel cells")
    write_json(output / "source_attribution.json", source)

    gap_control = cycles(by_key, "w4_c00_na_g64") - cycles(by_key, "w4_c00_na_g0")
    gap_dynamic_instructions = (
        by_key["w4_c00_na_g64"]["ncu_metrics"]["smsp__inst_executed.sum"]
        - by_key["w4_c00_na_g0"]["ncu_metrics"]["smsp__inst_executed.sum"]
    ) / (GROUPS * 4)

    identification_interactions = {
        f"{locality}_gap{gap}": interaction(by_key, 4, locality, gap)
        for locality in ("l1", "l2", "dram")
        for gap in (0, 64)
    }
    held_interactions = {
        f"w{warps}_{locality}": interaction(by_key, warps, locality, 0)
        for warps in (2, 8)
        for locality in ("l1", "dram")
    }
    source_locality = {
        f"{locality}_minus_l1_gap{gap}": cycles(
            by_key, f"w4_c10_{locality}_g{gap}"
        )
        - cycles(by_key, f"w4_c10_l1_g{gap}")
        for locality in ("l2", "dram")
        for gap in (0, 64)
    }
    overlap = {
        f"{protocol}_{locality}": {
            "raw_h_cycles_per_group": cycles(by_key, f"w4_{protocol}_{locality}_g0")
            - cycles(by_key, f"w4_{protocol}_{locality}_g64"),
            "gap_cycles_hidden": gap_control
            - (
                cycles(by_key, f"w4_{protocol}_{locality}_g64")
                - cycles(by_key, f"w4_{protocol}_{locality}_g0")
            ),
        }
        for protocol, localities in (
            ("c10", ("l1", "l2", "dram")),
            ("c11", ("l1", "l2", "dram")),
            ("c12", ("l1", "dram")),
        )
        for locality in localities
    }

    high_variance = [
        {
            "key": cell["key"],
            "repeats": cell["ncu_repeats"],
            "active_cycles_cv_pct": cell["ncu_active_cycles_cv_pct"],
        }
        for cell in cells
        if cell["ncu_active_cycles_cv_pct"] > 2.0
    ]
    control_paths = sorted((root / "sass").glob("*.control_bits.json"))
    require(len(control_paths) == 2, "expected long and trace control-bit sidecars")
    control_sidecars = []
    for path in control_paths:
        control_summary = read_json(path)["summary"]
        sidecar = {
            "file": path.name,
            "decoded": control_summary["total_decoded"],
            "unknown": control_summary["total_unknown_encoding"],
        }
        require(
            sidecar["decoded"] == 504 and sidecar["unknown"] == 0,
            f"{path}: expected 504 decoded instructions and zero unknown encodings",
        )
        control_sidecars.append(sidecar)
    resource_paths = sorted((root / "sass").glob("*.resources.txt"))
    require(len(resource_paths) == 2, "expected long and trace resource sidecars")
    resource_sidecars = [resource_usage(path) for path in resource_paths]
    require(
        all(
            sidecar["all_stack_zero"] and sidecar["all_local_zero"]
            for sidecar in resource_sidecars
        ),
        "protocol kernels have nonzero stack or local-memory allocation",
    )

    cohort_predicted = read_json(output / "cohort_predicted.json")
    cohort_integrity = load_cohort_integrity(root, cohort_predicted)
    cohort_score = read_json(output / "cohort_score.json")
    require(
        cohort_score["coverage"]["shared_numeric_count"] == 6
        and cohort_score["coverage"]["observed_fraction"] == 1.0
        and cohort_score["coverage"]["predicted_fraction"] == 1.0,
        "production cohort does not have complete six-workload score coverage",
    )

    host_before = read_json(root / "metadata" / "host_replay_before.json")
    host_after = read_json(root / "metadata" / "host_replay_after.json")
    require(
        host_before.keys() == host_after.keys()
        and len(host_before) == 2
        and all(
            host_before[key]["samples"] == 200
            and host_after[key]["samples"] == 200
            for key in host_before
        ),
        "host replay metadata does not contain two matched 200-sample runs",
    )
    host_replay = {
        key: {
            "samples": host_after[key]["samples"],
            "before_median_ms": host_before[key]["median_ms"],
            "after_median_ms": host_after[key]["median_ms"],
            "median_change_ms": (
                host_after[key]["median_ms"] - host_before[key]["median_ms"]
            ),
            "median_change_pct": host_after[key]["median_change_pct"],
        }
        for key in sorted(host_before)
    }
    summary = {
        "schema_version": 1,
        "campaign": {
            "integrity_validated": True,
            "cells": len(cells),
            "ncu_profiles": sum(cell["ncu_repeats"] for cell in cells),
            "source_counter_cells": len(source),
            "all_source_counter_contracts_validated": True,
            "nvbit_cells": len(cells),
            "byte_identical_replay_pairs": sum(
                bool(cell["simulator_byte_identical"]) for cell in cells
            ),
            "all_correct": all(cell["correct"] for cell in cells),
            "all_correctness_seven_trials": all(
                cell["correctness_trials"] == 7 for cell in cells
            ),
            "all_ncu_capture_contracts_validated": True,
            "all_ncu_repeat_policy_satisfied": all(
                cell["ncu_initial_active_cycles_cv_pct"] <= 2.0
                or cell["ncu_repeats"] == 5
                for cell in cells
            ),
            "all_dynamic_work_invariant": all(
                cell["dynamic_work_invariant"] for cell in cells
            ),
            "all_nvbit_one_full_cta": all(
                cell["nvbit_blocks"] == 1
                and cell["nvbit_threads"] == cell["warps"] * 32
                for cell in cells
            ),
            "all_nvbit_payloads_correct": all(
                cell["nvbit_payload_correct"]
                and cell["nvbit_payload_trials"] == 1
                for cell in cells
            ),
            "all_nvbit_diagnostic_sidecars_empty": all(
                sum(cell["nvbit_diagnostic_counts"].values()) == 0
                for cell in cells
            ),
            "all_nvbit_cardinality_exact": all(
                cell["nvbit_total_instructions"]
                == cell["nvbit_reported_instructions"]
                for cell in cells
            ),
            "all_simulator_cardinality_exact": all(
                cell["simulator_instructions_issued"]
                == cell["simulator_records_seen"]
                for cell in cells
            ),
            "all_simulator_full_coverage": all(
                cell["simulator_coverage"]["kind"] == "full"
                and cell["simulator_coverage"]["blocks_observed"]
                == cell["simulator_coverage"]["population_blocks"]
                for cell in cells
            ),
            "all_simulator_readiness_schema_v1": all(
                cell["simulator_readiness_schema_version"] == 1 for cell in cells
            ),
            "control_sidecars": control_sidecars,
            "resource_sidecars": resource_sidecars,
        },
        "gap_calibration": {
            "executed_sass_instructions_per_warp_group": gap_dynamic_instructions,
            "cycles_per_group": gap_control,
        },
        "identification_interactions": identification_interactions,
        "held_out_interactions": held_interactions,
        "source_locality_cycles_per_group": source_locality,
        "overlap": overlap,
        "high_variance_cells": high_variance,
        "cohort_integrity": cohort_integrity,
        "cohort_score": cohort_score,
        "host_replay": host_replay,
        "promotion": {
            "timing_rule_promoted": False,
            "readiness_diagnostics_retained": True,
            "reason": (
                "barrier-only cost and async/barrier interaction depend on CTA width; "
                "W4 identification values do not transfer to held-out W2/W8"
            ),
        },
    }
    write_json(output / "summary.json", summary)

    route_rows = []
    for locality in ("l1", "l2", "dram"):
        cell = by_key[f"w4_c10_{locality}_g0"]
        metrics = cell["ncu_metrics"]
        route_rows.append(
            (
                locality,
                f"{metrics['l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum']:.0f}",
                f"{metrics['lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum']:.0f}",
                f"{metrics['lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum']:.0f}",
                f"{metrics['dram__sectors_read.sum']:.0f}",
            )
        )
    factorial_rows = []
    for gap in (0, 64):
        for locality in ("l1", "l2", "dram"):
            values = identification_interactions[f"{locality}_gap{gap}"]
            factorial_rows.append(
                (
                    gap,
                    locality,
                    f"{cycles(by_key, f'w4_c10_{locality}_g{gap}'):.2f}",
                    f"{cycles(by_key, f'w4_c11_{locality}_g{gap}'):.2f}",
                    f"{values['barrier_only_cycles_per_group']:.2f}",
                    f"{values['interaction_cycles_per_group']:+.2f}",
                )
            )
    held_rows = []
    for warps in (2, 4, 8):
        for locality in ("l1", "dram"):
            values = (
                interaction(by_key, warps, locality, 0)
                if warps != 4
                else identification_interactions[f"{locality}_gap0"]
            )
            held_rows.append(
                (
                    warps,
                    locality,
                    f"{values['barrier_only_cycles_per_group']:.2f}",
                    f"{values['coupled_minus_wait_cycles_per_group']:.2f}",
                    f"{values['interaction_cycles_per_group']:+.2f}",
                )
            )
    overlap_rows = [
        (
            key.split("_", 1)[0],
            key.split("_", 1)[1],
            f"{value['raw_h_cycles_per_group']:+.2f}",
            f"{value['gap_cycles_hidden']:.2f}",
        )
        for key, value in overlap.items()
    ]
    source_rows = []
    for key, attribution in source.items():
        source_rows.append(
            (
                key,
                format_top(attribution["long_scoreboard"]),
                format_top(attribution["barrier"]),
                attribution["wait"]["samples"],
            )
        )
    simulator_rows = []
    for warps in (2, 4, 8):
        for locality in ("l1", "dram"):
            cell = by_key[f"w{warps}_c12_{locality}_g0"]
            readiness = cell["simulator_readiness"]
            barriers = readiness["cta_barriers"]
            simulator_rows.append(
                (
                    warps,
                    locality,
                    f"{cell['ncu_metrics']['smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio']:.2f}",
                    barriers["total_arrival_spread_cycles"],
                    barriers["total_rendezvous_warp_cycles"],
                    readiness["limiting_cycles"]["eligible_not_selected"],
                )
            )

    high_variance_text = (
        ", ".join(
            f"`{row['key']}` ({row['active_cycles_cv_pct']:.3f}%, {row['repeats']} repeats)"
            for row in high_variance
        )
        or "none"
    )
    report = f"""# sm_86 causal readiness and async/barrier campaign — final v4

## Outcome

The milestone delivered the reason-tagged readiness component, but the live
factorial **falsifies a fixed CTA-barrier latency**. No new timing rule is
promoted. Barrier-only cost rises from 14.00 cycles/group at W2 to 18.00 at W4
and 28.00 at W8. More importantly, the async/barrier interaction identified at
W4 does not transfer: the DRAM interaction is +1.78 cycles/group at W2,
+72.97 at W4, and +230.53 at W8; the L1 interaction changes sign by W8.

The simulator diagnostics are retained because they expose the causal state
needed for the next model: dependency origin, async source/local/SM-service
completion, CTA generations and arrival spread, and ready-but-not-selected
delay. The six-workload production timing output is exactly unchanged at
33.403% active-SM MAPE.

## Campaign integrity

- {len(cells)} configured cells; every ordinary seven-trial result passed.
- {sum(cell['ncu_repeats'] for cell in cells)} foreground NCU profiles at base clocks,
  including fourth/fifth repeats for every cell above 2% three-run CV.
- {len(source)} SourceCounters sentinels and {len(cells)} separate 64-group NVBit captures.
- Every NVBit cell has one full CTA, exact total/reported tracer cardinality,
  zero unsupported/unmapped/cross-allocation records, and a correct payload.
- Both long and trace sidecars decode all 504 static instructions with zero
  unknown encodings; all five protocol kernels have zero stack/local allocation.
- All {len(cells)} traces replay twice to byte-identical JSON with full coverage,
  `instructions_issued == records_seen`, and readiness schema version 1.
- Gap calibration is measured, not assumed: {gap_dynamic_instructions:.0f} dependent
  SASS instructions add {gap_control:.4f} base-clock cycles/group.
- Cells still above 2% after five repeats: {high_variance_text}. Their bimodal
  L2 behavior is retained and reported.
- No `sudo` command was required.

The earlier v1-v3 roots are intentionally retained as failed constructions:
v1 used a predicated runtime BAR that NVBit could not distinguish in replay;
v2 emitted a 314-cycle gap; v3's 12-instruction gap measured 53 cycles. V4
uses compile-time protocol specializations and the final 14-instruction gap.

## Measured route and work invariants

{markdown_table(('route', 'L1 misses', 'L2 hits', 'L2 misses', 'DRAM reads'), route_rows)}

For every W4 c10/c11 cell, NCU reports exactly 524,288 LDGSTS instructions
and 2,097,152 destination wavefronts. C12 reports 131,072 and 524,288,
respectively. Locality and gap therefore vary source route/overlap while
destination work stays fixed. L1 reuse has only 16 L1 misses; the L2 rotation
has predominantly L2 hits; the DRAM stream has 262,144 L2 misses.

Raw DRAM-sector counts are noisier than L2 lookup outcomes under application
replay, so route admission uses the jointly consistent L1-miss and L2-hit/miss
ordering, not allocation labels alone.

## W4 identification factorial

Cycles are median `sm__cycles_active.max / 16,384 groups`. `I` subtracts the
barrier-only control from the c11-minus-c10 difference.

{markdown_table(('gap', 'route', 'c10', 'c11', 'barrier-only', 'I'), factorial_rows)}

The source effect for c10 at gap0 is
{source_locality['l2_minus_l1_gap0']:.2f} cycles/group (L2−L1) and
{source_locality['dram_minus_l1_gap0']:.2f} (DRAM−L1). The calibrated
independent gap is almost entirely hidden behind async completion:

{markdown_table(('protocol', 'route', 'raw H = g0−g64', 'gap cycles hidden'), overlap_rows)}

Long-scoreboard ratios fall with the independent gap while active cycles stay
nearly flat for c10/c11, which is the expected signature of overlap rather than
an additive wait latency.

## Held-out CTA transfer

{markdown_table(('warps', 'route', 'barrier-only', 'c11−c10', 'I'), held_rows)}

The W4 +73-cycle L2/DRAM interaction cannot be an architecture constant: W2
and W8 disagree by more than two orders of magnitude. Barrier-only scaling also
rules out reusing the existing read/non-memory dependency latency fields. The
missing behavior is a scheduler/barrier-pipeline throughput and arrival-skew
interaction, not a fixed rendezvous delay.

## SourceCounters interpretation

{markdown_table(('cell', 'top long-scoreboard attribution', 'top barrier attribution', 'wait samples'), source_rows)}

For c11, every sampled long-scoreboard event is attributed to the immediately
following `BAR.SYNC.DEFER_BLOCKING`; c10 attributes it to the loop-condition
instruction after `DEPBAR.LE`. This confirms that NCU reports the active
dependent PC: it does **not** prove that BAR itself created L1TEX latency.
Barrier samples likewise land on the post-BAR loop branch. C12 separates the
two: long-scoreboard samples land on `BSYNC`, while barrier samples land on the
post-barrier branch. No W4 SourceCounters sentinel sampled not-selected stalls.

## Simulator causal diagnostics

{markdown_table(('warps', 'route', 'NCU barrier ratio', 'sim arrival spread', 'sim rendezvous warp-cycles', 'sim ready-not-selected'), simulator_rows)}

The diagnostics predict the important direction: producer/consumer DRAM has
more arrival spread and rendezvous exposure than L1, and both grow with CTA
width. W8 also exposes ready-but-not-selected delay. However, the 64-group
target-only trace intentionally omits the prewarm kernel and does not wrap the
2 MiB L2 rotation; consequently its cold simulator replay cannot distinguish
the L2 and DRAM warm states. That is an explicit cache-state handoff gap, not a
license to inject NCU route labels into replay.

## Production cohort

The six original traces replayed twice with identical outputs:

```json
{json.dumps(cohort_predicted, indent=2)}
```

Against the same observed active-SM medians, coverage remains 6/6, MAE is
{cohort_score['errors']['mean_absolute_error']:.2f}, RMSE is
{cohort_score['errors']['root_mean_squared_error']:.2f}, and MAPE is
{cohort_score['errors']['mean_absolute_percentage_error'] * 100:.3f}%—exactly
the prior baseline. Diagnostics changed observability, not timing.

## Host replay cost

Process-level medians over 200 invocations per binary are:

{markdown_table(
    ('cell', 'before ms', 'after ms', 'delta ms', 'delta %'),
    [
        (
            key,
            f"{value['before_median_ms']:.3f}",
            f"{value['after_median_ms']:.3f}",
            f"{value['median_change_ms']:+.3f}",
            f"{value['median_change_pct']:+.2f}%",
        )
        for key, value in host_replay.items()
    ],
)}

The tiny non-async process exceeds the 3% percentage target but adds only
0.175 ms; the representative async path improves by 1.66%. The measured
sub-millisecond control-path tradeoff is accepted for the retained causal
diagnostics.

## Promotion decision and next milestone

No CTA timing value is added to `sm_86`. The next milestone should build two
essential mechanisms:

1. a barrier-issue/pipeline resource characterized by a focused W1–W16
   barrier-only sweep (multiple generations, retained warp-slot placement,
   no async source traffic); and
2. an explicit, architecture-local cache warm-state handoff for controlled
   replay so a prewarm launch can seed L2 state without using observed NCU
   values or workload identity.

After those pieces, repeat the c10/c11 held-out test and add multi-CTA scheduler
contention to validate eligible-not-selected behavior. A barrier rule is
promotable only if one mechanism predicts W2/W4/W8 signs and ordering.
The complete successor specification is
`docs/sm86_barrier_pipeline_cache_handoff_milestone.md`.
"""
    (output / "README.md").write_text(report, encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    try:
        summary = analyze(args.root.resolve())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"analyze: {error}")
        return 1
    print(json.dumps(summary["promotion"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
