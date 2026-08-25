#!/usr/bin/env python3
"""Run the prospective sm_86 L1TEX admission/backpressure campaign."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_OUTPUT = (
    REPO_ROOT / "results/captures/sm86_l1tex_admission_backpressure_20260818_v1"
)
LONG_BINARY = HERE / "l1tex_admission_backpressure"
TRACE_BINARY = HERE / "l1tex_admission_backpressure_trace"
CANDIDATE_FILE = HERE.parent / "l1tex_arrival_replay_service/memory_service_candidates.json"
ADMISSION_CANDIDATE_FILE = HERE / "memory_admission_candidates.json"
FEEDBACK_CANDIDATE_FILES = {
    7: HERE / "memory_admission_feedback_store_7.json",
    12: HERE / "memory_admission_feedback_red_12.json",
    14: HERE / "memory_admission_feedback_red_14.json",
}
WRITE_COMPLETION_FEEDBACK_FILES = {
    201: HERE / "memory_write_completion_feedback_store_201.json",
    202: HERE / "memory_write_completion_feedback_red_202.json",
    203: HERE / "memory_write_completion_feedback_store_203.json",
    204: HERE / "memory_write_completion_feedback_store_204.json",
    205: HERE / "memory_write_completion_feedback_store_205.json",
    206: HERE / "memory_write_completion_feedback_store_206.json",
    207: HERE / "memory_write_completion_feedback_red_207.json",
    208: HERE / "memory_write_completion_feedback_red_208.json",
    209: HERE / "memory_write_completion_feedback_red_209.json",
    210: HERE / "memory_write_completion_feedback_red_210.json",
}
FEEDBACK_SERVICE_SWEEP_FILES = {
    "store": HERE / "memory_admission_feedback_store_service_sweep.json",
    "red": HERE / "memory_admission_feedback_red_service_sweep.json",
    "red_fast": HERE / "memory_admission_feedback_red_fast_service_sweep.json",
}
RED_RMW_COMPLETION_SWEEP_FILES = {
    "request": HERE / "red_rmw_request_completion_sweep.json",
    "region": HERE / "red_rmw_region_completion_sweep.json",
}
TAILS = ("control", "store", "red", "atom")
LAYOUTS = (
    "line_unique",
    "word_unique",
    "line_reuse",
    "word_reuse",
    "warp_word_regions",
    "warp_word_shared_region",
)
PATTERNS = ("all", "one_per_smsp", "same_smsp", "two_smsps")
ACTIVE_LANES = 8

METRICS = (
    "gpc__cycles_elapsed.avg",
    "gpc__cycles_elapsed.min",
    "gpc__cycles_elapsed.max",
    "gpc__cycles_elapsed.sum",
    "gpu__time_duration.sum",
    "sm__cycles_active.avg",
    "sm__cycles_active.min",
    "sm__cycles_active.max",
    "sm__cycles_active.sum",
    "smsp__cycles_active.sum",
    "smsp__inst_issued.sum",
    "smsp__inst_executed.sum",
    "smsp__inst_executed_pipe_lsu.sum",
    "smsp__inst_executed_op_generic_atom_dot_alu.sum",
    "smsp__inst_executed_op_global_red.sum",
    "sm__sass_inst_executed_op_global_atom.sum",
    "sm__sass_inst_executed_op_global_red.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum",
    "l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_red.sum",
    "l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_atom.sum",
    "l1tex__t_set_conflicts_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_set_conflicts_pipe_lsu_mem_global_op_red.sum",
    "l1tex__t_set_conflicts_pipe_lsu_mem_global_op_atom.sum",
    "l1tex__lsuin_requests.sum",
    "lts__t_requests_srcunit_tex_op_write.sum",
    "lts__t_requests_srcunit_tex_op_red.sum",
    "lts__t_requests_srcunit_tex_op_atom_dot_alu.sum",
    "lts__t_sectors_srcunit_tex_op_write.sum",
    "lts__t_sectors_op_red.sum",
    "lts__t_sectors_op_atom.sum",
    "dram__sectors_read.sum",
    "dram__sectors_write.sum",
    "smsp__average_warps_issue_stalled_drain_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_membar_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
)


def load_helpers():
    path = HERE.parent / "memory_retirement_service/campaign.py"
    spec = importlib.util.spec_from_file_location("admission_campaign_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load campaign helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HELPERS = load_helpers()
BASE = HELPERS.BASE
REPLAY = HELPERS.REPLAY


def selected_warps(warps: int, pattern: str) -> int:
    if pattern == "all":
        return warps
    if warps != 8:
        raise RuntimeError("sparse patterns require an eight-warp CTA")
    return {"one_per_smsp": 4, "same_smsp": 2, "two_smsps": 4}[pattern]


@dataclass(frozen=True, slots=True)
class Cell:
    stage: str
    tail: str
    layout: str
    blocks: int
    warps: int
    depth: int
    gap_steps: int
    pattern: str
    active_lanes: int = ACTIVE_LANES

    @property
    def key(self) -> str:
        return (
            f"b{self.blocks}_w{self.warps}_{self.tail}_{self.layout}_"
            f"d{self.depth}_g{self.gap_steps}_{self.pattern}"
        )

    @property
    def selected_warps_per_block(self) -> int:
        return selected_warps(self.warps, self.pattern)

    @property
    def memory_instructions(self) -> int:
        if self.tail == "control":
            return 0
        return self.blocks * self.selected_warps_per_block * self.depth

    @property
    def unique_lines_per_instruction(self) -> int:
        return self.active_lanes if self.layout.startswith("line_") else 1

    @property
    def ideal_wavefronts_per_instruction(self) -> int:
        if self.layout.startswith("line_"):
            return (self.active_lanes + 3) // 4
        if self.tail in {"red", "atom"} and self.layout.startswith("warp_word_"):
            return self.active_lanes
        return 1

    @property
    def sectors_per_instruction(self) -> int:
        if self.layout.startswith("line_"):
            return self.active_lanes
        if self.tail in {"red", "atom"} and self.layout.startswith("warp_word_"):
            return self.active_lanes
        return 1

    @property
    def atomic_requests_per_instruction(self) -> int:
        if self.layout.startswith("line_"):
            return self.active_lanes
        if self.layout.startswith("warp_word_"):
            return self.active_lanes
        return 1

    def target(self, *, trace: bool, trials: int) -> list[str]:
        return [
            str(TRACE_BINARY if trace else LONG_BINARY),
            self.tail,
            self.layout,
            str(self.blocks),
            str(self.warps),
            str(self.depth),
            str(self.active_lanes),
            str(self.gap_steps),
            self.pattern,
            str(trials),
        ]


def cell_document(cell: Cell) -> dict[str, object]:
    return {**asdict(cell), "key": cell.key}


def measurement_matches_cell(value: object, cell: Cell, *, trials: int) -> bool:
    if not isinstance(value, dict):
        return False
    expected = {
        "tail": cell.tail,
        "layout": cell.layout,
        "blocks": cell.blocks,
        "warps": cell.warps,
        "depth": cell.depth,
        "active_lanes": cell.active_lanes,
        "gap_steps": cell.gap_steps,
        "pattern": cell.pattern,
        "selected_warps_per_block": cell.selected_warps_per_block,
        "memory_instructions": cell.memory_instructions,
        "l2_eviction_bytes": 32 * 1024 * 1024,
        "trials": trials,
        "correct": True,
    }
    return all(value.get(key) == expected_value for key, expected_value in expected.items())


def captured_run_matches(destination: Path, command: Sequence[str]) -> bool:
    try:
        retained = json.loads((destination / "run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return retained.get("return_code") == 0 and retained.get("command") == list(command)


def outputs_are_fresh(outputs: Sequence[Path], dependencies: Sequence[Path]) -> bool:
    if any(not path.is_file() or path.stat().st_size == 0 for path in outputs):
        return False
    try:
        newest_input = HELPERS.latest_input_mtime_ns(dependencies)
    except (OSError, RuntimeError):
        return False
    return all(path.stat().st_mtime_ns > newest_input for path in outputs)


def executable_dependencies(command: str) -> list[Path]:
    resolved = shutil.which(command)
    return [Path(resolved).resolve()] if resolved else []


def ncu_profile_command(ncu: str, destination: Path, cell: Cell) -> list[str]:
    return [
        ncu,
        "--force-overwrite",
        "--export",
        str(destination / "ncu-report"),
        "--profile-from-start",
        "off",
        "--launch-count",
        "1",
        "--replay-mode",
        "application",
        "--clock-control",
        "base",
        "--cache-control",
        "all",
        "--metrics",
        ",".join(METRICS),
        *cell.target(trace=False, trials=1),
    ]


def ncu_source_command(ncu: str, destination: Path, cell: Cell) -> list[str]:
    return [
        ncu,
        "--force-overwrite",
        "--export",
        str(destination / "ncu-report"),
        "--profile-from-start",
        "off",
        "--launch-count",
        "1",
        "--replay-mode",
        "application",
        "--clock-control",
        "base",
        "--cache-control",
        "all",
        "--section",
        "SourceCounters",
        *cell.target(trace=False, trials=1),
    ]


def validated_admission_documents() -> tuple[dict, dict[int, dict]]:
    admission = REPLAY.load_json_strict(ADMISSION_CANDIDATE_FILE)
    if not isinstance(admission, dict):
        raise RuntimeError("invalid memory-admission candidate document")
    candidates = list(REPLAY.candidate_configurations(ADMISSION_CANDIDATE_FILE))
    by_id = {
        int(candidate["candidate_id"]): candidate
        for candidate in candidates
        if isinstance(candidate, dict) and "candidate_id" in candidate
    }
    if len(candidates) != 16 or len(by_id) != len(candidates):
        raise RuntimeError("memory-admission candidate IDs must be unique 1..16")
    if set(by_id) != set(range(1, 17)):
        raise RuntimeError("memory-admission candidate IDs must be exactly 1..16")
    for candidate_id, path in FEEDBACK_CANDIDATE_FILES.items():
        feedback_candidates = REPLAY.candidate_configurations(
            path, require_exactly_one=True
        )
        if feedback_candidates[0] != by_id[candidate_id]:
            raise RuntimeError(
                f"feedback candidate {candidate_id} is not the matching frozen "
                "memory-admission candidate"
            )
    return admission, by_id


def validated_feedback_service_sweep() -> dict[int, tuple[Path, dict]]:
    expected = {
        "store": {
            "operation": "store",
            "scope": "smsp",
            "capacity": 16,
            "rates": {8, 10, 12, 14},
            "ids": set(range(101, 117)),
        },
        "red": {
            "operation": "reduction",
            "scope": "sm",
            "capacity": 32,
            "rates": {12, 16, 20, 24},
            "ids": set(range(121, 137)),
        },
        "red_fast": {
            "operation": "reduction",
            "scope": "sm",
            "capacity": 32,
            "rates": {8},
            "ids": set(range(137, 141)),
        },
    }
    widths = {1, 2, 4, 8}
    result: dict[int, tuple[Path, dict]] = {}
    for tail, path in FEEDBACK_SERVICE_SWEEP_FILES.items():
        candidates = list(REPLAY.candidate_configurations(path))
        design = expected[tail]
        ids = {int(candidate.get("candidate_id", -1)) for candidate in candidates}
        combinations = {
            (
                int(candidate.get("l2_miss_sector_cycles", -1)),
                int(candidate.get("l2_miss_service_width_sectors", -1)),
            )
            for candidate in candidates
        }
        if len(candidates) != len(design["ids"]) or ids != design["ids"]:
            raise RuntimeError(f"{tail} feedback-service sweep IDs must match the frozen grid")
        if combinations != {
            (rate, width) for rate in design["rates"] for width in widths
        }:
            raise RuntimeError(
                f"{tail} feedback-service sweep must be the full rate/width Cartesian grid"
            )
        for candidate in candidates:
            if (
                candidate.get("operation") != design["operation"]
                or candidate.get("scope") != design["scope"]
                or candidate.get("queue_capacity_instructions") != design["capacity"]
                or candidate.get("admission_latency_cycles") != 8
                or candidate.get("l2_hit_sector_units") != 1
                or candidate.get("l2_hit_sector_cycles") != 1
                or candidate.get("l2_miss_sector_units") != 1
            ):
                raise RuntimeError(f"{tail} feedback-service sweep changes a frozen dimension")
            candidate_id = int(candidate["candidate_id"])
            if candidate_id in result or candidate_id in FEEDBACK_CANDIDATE_FILES:
                raise RuntimeError(f"feedback-service candidate id {candidate_id} is duplicated")
            result[candidate_id] = (path, candidate)
    return result


def validated_write_completion_feedback() -> dict[int, tuple[Path, dict]]:
    _, admission_candidates = validated_admission_documents()
    store_ids = {201, 203, 204, 205, 206}
    red_ids = {202, 207, 208, 209, 210}
    latencies = {
        201: 0,
        202: 0,
        203: 512,
        204: 1024,
        205: 1536,
        206: 2048,
        207: 1024,
        208: 2048,
        209: 3072,
        210: 4096,
    }
    result: dict[int, tuple[Path, dict]] = {}
    for candidate_id, path in WRITE_COMPLETION_FEEDBACK_FILES.items():
        campaign_candidates = REPLAY.candidate_configurations(
            path, require_exactly_one=True
        )
        candidate = dict(campaign_candidates[0])
        predecessor = admission_candidates[7] if candidate_id in store_ids else admission_candidates[12]
        if candidate_id not in store_ids | red_ids:
            raise RuntimeError(f"unknown write-completion candidate {candidate_id}")
        expected = dict(predecessor)
        expected["candidate_id"] = candidate_id
        expected["membar_waits_for_service_completion"] = True
        if latencies[candidate_id] != 0:
            expected["write_completion_latency_cycles"] = latencies[candidate_id]
        if candidate != expected:
            raise RuntimeError(
                f"write-completion candidate {candidate_id} changes a predecessor dimension"
            )
        result[candidate_id] = (path, candidate)
    return result


def validated_red_rmw_completion_sweep() -> dict[int, tuple[Path, dict]]:
    _, admission_candidates = validated_admission_documents()
    predecessor = admission_candidates[12]
    expected = {
        "request": (set(range(211, 220)), {2, 4, 8}, "write_completion_request_cycles"),
        "region": (set(range(220, 229)), {4, 8, 16}, "write_completion_region_cycles"),
    }
    lanes = {1, 2, 4}
    result: dict[int, tuple[Path, dict]] = {}
    for basis, path in RED_RMW_COMPLETION_SWEEP_FILES.items():
        ids, rates, field = expected[basis]
        candidates = list(REPLAY.candidate_configurations(path))
        if {int(candidate.get("candidate_id", -1)) for candidate in candidates} != ids:
            raise RuntimeError(f"RED RMW {basis} candidate IDs differ from the frozen grid")
        if {
            (int(candidate.get(field, -1)), int(candidate.get("write_completion_parallel_lanes", -1)))
            for candidate in candidates
        } != {(rate, lane) for rate in rates for lane in lanes}:
            raise RuntimeError(f"RED RMW {basis} candidates do not form the frozen grid")
        for candidate in candidates:
            candidate_id = int(candidate["candidate_id"])
            frozen = dict(predecessor)
            frozen["candidate_id"] = candidate_id
            frozen["membar_waits_for_service_completion"] = True
            frozen[field] = candidate[field]
            frozen["write_completion_parallel_lanes"] = candidate[
                "write_completion_parallel_lanes"
            ]
            if candidate != frozen:
                raise RuntimeError(f"RED RMW candidate {candidate_id} changes a frozen dimension")
            result[candidate_id] = (path, candidate)
    return result


def configured_cells() -> list[Cell]:
    cells: dict[str, Cell] = {}

    def add(cell: Cell) -> None:
        cells.setdefault(cell.key, cell)

    for tail in TAILS:
        for depth in (1, 2, 4, 8, 16, 32, 64, 128):
            add(Cell("burst_depth", tail, "line_unique", 84, 8, depth, 0, "all"))

    for tail in TAILS:
        for warps in (1, 2, 4):
            add(Cell("warp_occupancy", tail, "line_unique", 84, warps, 32, 0, "all"))

    for tail in TAILS:
        for gap in (1, 4, 16, 64):
            add(Cell("issue_spacing", tail, "line_unique", 84, 8, 32, gap, "all"))

    for tail in TAILS:
        for layout in LAYOUTS[1:]:
            add(Cell("address_topology", tail, layout, 84, 8, 32, 0, "all"))

    for tail in TAILS:
        for pattern in PATTERNS[1:]:
            add(Cell("sparse_smsp", tail, "line_unique", 84, 8, 32, 0, pattern))

    for tail in TAILS:
        add(Cell("cta_transfer", tail, "line_unique", 168, 4, 32, 0, "all"))
        add(Cell("cta_transfer", tail, "line_unique", 336, 2, 32, 0, "all"))

    material = list(cells.values())
    if len(material) != 100:
        raise RuntimeError(f"admission/backpressure matrix must have 100 cells, got {len(material)}")
    return material


def source_cells() -> list[Cell]:
    wanted = {
        "b84_w8_store_line_unique_d1_g0_all",
        "b84_w8_store_line_unique_d32_g0_all",
        "b84_w8_store_line_unique_d128_g0_all",
        "b84_w8_store_line_unique_d32_g16_all",
        "b84_w8_store_word_unique_d32_g0_all",
        "b84_w8_store_line_reuse_d32_g0_all",
        "b84_w8_red_line_unique_d32_g0_all",
        "b84_w8_red_warp_word_regions_d32_g0_all",
        "b84_w8_red_warp_word_shared_region_d32_g0_all",
        "b84_w8_atom_line_unique_d32_g0_all",
        "b84_w8_atom_warp_word_regions_d32_g0_all",
        "b84_w8_atom_warp_word_shared_region_d32_g0_all",
        "b84_w8_store_line_unique_d32_g0_same_smsp",
        "b84_w8_store_line_unique_d32_g0_one_per_smsp",
        "b168_w4_store_line_unique_d32_g0_all",
        "b336_w2_store_line_unique_d32_g0_all",
    }
    selected = [cell for cell in configured_cells() if cell.key in wanted]
    if len(selected) != len(wanted):
        raise RuntimeError("source sentinel set is incomplete")
    return selected


def selected_cells(selector: str | None) -> list[Cell]:
    cells = configured_cells()
    if selector is None:
        return cells
    requested = {item.strip() for item in selector.split(",") if item.strip()}
    selected = [cell for cell in cells if cell.key in requested]
    missing = requested - {cell.key for cell in selected}
    if missing:
        raise RuntimeError(f"unknown cell key(s): {', '.join(sorted(missing))}")
    return selected


def require_binaries() -> None:
    for path in (LONG_BINARY, TRACE_BINARY):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"required binary missing: {path}")


def emit_manifest(output: Path, *, refresh_environment: bool = False) -> None:
    require_binaries()
    cells = configured_cells()
    admission_candidate_document, _ = validated_admission_documents()
    matrix = {
            "schema_version": 1,
            "cells": [
                {
                    **asdict(cell),
                    "key": cell.key,
                    "selected_warps_per_block": cell.selected_warps_per_block,
                    "memory_instructions": cell.memory_instructions,
                    "unique_lines_per_instruction": cell.unique_lines_per_instruction,
                    "ideal_wavefronts_per_instruction": cell.ideal_wavefronts_per_instruction,
                    "sectors_per_instruction": cell.sectors_per_instruction,
                    "atomic_requests_per_instruction": cell.atomic_requests_per_instruction,
                }
                for cell in cells
            ],
            "source_counter_cells": [cell.key for cell in source_cells()],
            "ncu_metrics": list(METRICS),
            "ncu_repeats": 3,
            "correctness_trials": 3,
            "memory_admission_candidate_file": str(
                ADMISSION_CANDIDATE_FILE.relative_to(REPO_ROOT)
            ),
            "memory_admission_feedback_files": {
                str(candidate_id): str(path.relative_to(REPO_ROOT))
                for candidate_id, path in FEEDBACK_CANDIDATE_FILES.items()
            },
            "sm_count": 84,
            "invariants": [
                "CLOCK issue interval ends before MEMBAR.SC.GPU",
                "CLOCK fence interval begins after all target memory instructions issue",
                "native target launch follows a separate 32 MiB runtime memset eviction sweep",
                "control carries identical CLOCK, address, gap, MEMBAR, and output structure",
                "gap steps change the dependent STORE payload as well as spacing; compression sensitivity remains explicit",
                "line_unique uses eight distinct 128-byte lines per active warp instruction",
                "same-region warp words stay within one aligned 256-byte region",
                "one_per_smsp and same_smsp hold selected warp count or SMSP footprint independently",
                "1x8, 2x4, and 4x2 CTA shapes preserve 256 selected warp-operations per nominal SM",
            ],
    }
    HELPERS.preserve_json_snapshot(output / "metadata/matrix.json", matrix)
    REPLAY.candidate_configurations(CANDIDATE_FILE)
    candidate_document = REPLAY.load_json_strict(CANDIDATE_FILE)
    HELPERS.preserve_json_snapshot(
        output / "metadata/memory_service_candidates.json", candidate_document
    )
    HELPERS.preserve_json_snapshot(
        output / "metadata/memory_admission_candidates.json",
        admission_candidate_document,
    )
    for candidate_id, path in FEEDBACK_CANDIDATE_FILES.items():
        HELPERS.preserve_json_snapshot(
            output / f"metadata/memory_admission_feedback_{candidate_id}.json",
            REPLAY.load_json_strict(path),
        )
    for candidate_id, path in WRITE_COMPLETION_FEEDBACK_FILES.items():
        HELPERS.preserve_json_snapshot(
            output / f"metadata/memory_write_completion_feedback_{candidate_id}.json",
            REPLAY.load_json_strict(path),
        )
    rmw_sweep = validated_red_rmw_completion_sweep()
    for _, path in RED_RMW_COMPLETION_SWEEP_FILES.items():
        HELPERS.preserve_json_snapshot(
            output / "metadata" / path.name,
            REPLAY.load_json_strict(path),
        )
    rmw_singletons = output / "metadata/red_rmw_completion_singletons"
    for candidate_id, (_, candidate) in rmw_sweep.items():
        HELPERS.preserve_json_snapshot(
            rmw_singletons / f"candidate-{candidate_id}.json",
            {"schema_version": 1, "candidates": [candidate]},
        )
    service_sweep = validated_feedback_service_sweep()
    for _, path in FEEDBACK_SERVICE_SWEEP_FILES.items():
        HELPERS.preserve_json_snapshot(
            output / "metadata" / path.name,
            REPLAY.load_json_strict(path),
        )
    singleton_root = output / "metadata/memory_admission_feedback_service_singletons"
    for candidate_id, (_, candidate) in service_sweep.items():
        HELPERS.preserve_json_snapshot(
            singleton_root / f"candidate-{candidate_id}.json",
            {"schema_version": 1, "candidates": [candidate]},
        )
    environment_path = output / "metadata/environment.json"
    if environment_path.is_file() and not refresh_environment:
        return
    environment: dict[str, object] = {
        "observed_at": BASE.now(),
        "role": "tool/host observation; raw artifact mtimes define capture chronology",
    }
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
    BASE.write_json(environment_path, environment)


def payload(
    completed: subprocess.CompletedProcess[str], cell: Cell, *, trials: int
) -> dict:
    lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    if not lines:
        raise RuntimeError(f"{cell.key}: target emitted no JSON")
    value = json.loads(lines[-1])
    if not measurement_matches_cell(value, cell, trials=trials):
        raise RuntimeError(f"{cell.key}: correctness/work invariant failed")
    return value


def run_correctness(output: Path, cells: Sequence[Cell], trials: int) -> None:
    for index, cell in enumerate(cells, 1):
        destination = output / "correctness" / cell.key
        completed = BASE.run_captured(cell.target(trace=False, trials=trials), destination, timeout=300)
        BASE.write_json(
            destination / "result.json",
            {
                "cell": cell_document(cell),
                "measurement": payload(completed, cell, trials=trials),
            },
        )
        print(f"correctness {index:03d}/{len(cells):03d} {cell.key}", flush=True)


def run_native_timing(
    output: Path, cells: Sequence[Cell], repeats: int, force: bool
) -> None:
    for index, cell in enumerate(cells, 1):
        for repeat in range(1, repeats + 1):
            destination = output / "native-timing" / cell.key / f"repeat-{repeat}"
            result_path = destination / "result.json"
            command = cell.target(trace=False, trials=1)
            if not force:
                try:
                    retained = json.loads(result_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    retained = {}
                if (
                    measurement_matches_cell(retained, cell, trials=1)
                    and captured_run_matches(destination, command)
                    and outputs_are_fresh(
                        (
                            result_path,
                            destination / "run.json",
                            destination / "command.txt",
                            destination / "stdout.txt",
                        ),
                        (LONG_BINARY,),
                    )
                ):
                    continue
            completed = BASE.run_captured(
                command, destination, timeout=300
            )
            BASE.write_json(result_path, payload(completed, cell, trials=1))
        print(f"native {index:03d}/{len(cells):03d} {cell.key}", flush=True)


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
    for cell in cells:
        for repeat in range(repeat_start, repeat_start + repeats):
            progress += 1
            destination = output / "ncu" / cell.key / f"repeat-{repeat}"
            report = destination / "ncu-report.ncu-rep"
            export = destination / "ncu.csv"
            command = ncu_profile_command(ncu, destination, cell)
            dependencies = [LONG_BINARY, *executable_dependencies(ncu)]
            capture_outputs = (
                report,
                export,
                destination / "run.json",
                destination / "command.txt",
                destination / "stdout.txt",
            )
            if (
                not force
                and HELPERS.complete_ncu_export(report, export)
                and captured_run_matches(destination, command)
                and outputs_are_fresh(capture_outputs, dependencies)
            ):
                print(f"ncu {progress:03d}/{total:03d} {cell.key} r{repeat} (existing)", flush=True)
                continue
            BASE.run_captured(
                command,
                destination,
                timeout=1200,
            )
            if not report.is_file():
                raise RuntimeError(f"NCU did not produce {report}")
            BASE.export_ncu(ncu, report, destination)
            if not HELPERS.complete_ncu_export(report, destination / "ncu.csv"):
                raise RuntimeError(f"NCU export is incomplete for {cell.key} repeat {repeat}")
            print(f"ncu {progress:03d}/{total:03d} {cell.key} r{repeat}", flush=True)


def run_source(output: Path, force: bool) -> None:
    ncu = os.environ.get("NCU", "ncu")
    cells = source_cells()
    for index, cell in enumerate(cells, 1):
        destination = output / "ncu-source" / cell.key
        report = destination / "ncu-report.ncu-rep"
        export = destination / "ncu.csv"
        source_export = destination / "source.csv"
        command = ncu_source_command(ncu, destination, cell)
        dependencies = [LONG_BINARY, *executable_dependencies(ncu)]
        capture_outputs = (
            report,
            export,
            source_export,
            destination / "run.json",
            destination / "command.txt",
            destination / "stdout.txt",
        )
        if (
            HELPERS.complete_ncu_export(
                report, export, source_export
            )
            and captured_run_matches(destination, command)
            and outputs_are_fresh(capture_outputs, dependencies)
            and not force
        ):
            print(f"source {index:02d}/{len(cells):02d} {cell.key} (existing)", flush=True)
            continue
        BASE.run_captured(
            command,
            destination,
            timeout=1200,
        )
        if not report.is_file():
            raise RuntimeError(f"NCU did not produce {report}")
        BASE.export_ncu(ncu, report, destination)
        exported = subprocess.run(
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
                str(source_export),
            ],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        (destination / "source-export.stdout.txt").write_text(exported.stdout, encoding="utf-8")
        (destination / "source-export.stderr.txt").write_text(exported.stderr, encoding="utf-8")
        if exported.returncode != 0 or not HELPERS.complete_ncu_export(
            report, export, source_export
        ):
            raise RuntimeError(f"source export failed for {cell.key}")
        print(f"source {index:02d}/{len(cells):02d} {cell.key}", flush=True)


def run_nvbit(output: Path, cells: Sequence[Cell], force: bool) -> None:
    tracer = REPO_ROOT / "tools/nvbit_tracer/tracer_tool/tracer_tool.so"
    environment_base = dict(os.environ)
    protobuf_lib = REPO_ROOT / ".toolchain/pb/lib"
    old_library_path = environment_base.get("LD_LIBRARY_PATH", "")
    environment_base.update(
        {
            "CUDA_INJECTION64_PATH": str(tracer),
            "USER_DEFINED_FOLDERS": "1",
            "ACTIVE_FROM_START": "0",
            "SKIP_ENHANCED_TRACER": "0",
            "LD_LIBRARY_PATH": str(protobuf_lib)
            + ((":" + old_library_path) if old_library_path else ""),
        }
    )
    for index, cell in enumerate(cells, 1):
        destination = output / "nvbit" / cell.key
        trace = destination / "dynamic_trace.pb"
        command = cell.target(trace=True, trials=1)
        dependencies = [TRACE_BINARY, tracer]
        capture_path = destination / "capture.json"
        try:
            capture = json.loads(capture_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            capture = {}
        capture_outputs = (
            trace,
            destination / "stats.csv",
            capture_path,
            destination / "run.json",
            destination / "command.txt",
            destination / "stdout.txt",
            destination / "extra_info/allocations.json",
            destination / "extra_info/enhanced_execution_info.json",
            destination / "extra_info/traced_kernel_modules.json",
            destination / "extra_info/unsupported_memory_path.json",
            destination / "extra_info/unmapped_memory_range.json",
            destination / "extra_info/cross_allocation_lane_span.json",
        )
        if (
            not force
            and HELPERS.complete_nvbit_capture(destination)
            and captured_run_matches(destination, command)
            and capture.get("cell") == cell_document(cell)
            and outputs_are_fresh(capture_outputs, dependencies)
        ):
            print(f"nvbit {index:03d}/{len(cells):03d} {cell.key} (existing)", flush=True)
            continue
        destination.mkdir(parents=True, exist_ok=True)
        environment = dict(environment_base)
        environment["TRACES_FOLDER"] = str(destination.resolve())
        completed = BASE.run_captured(
            command,
            destination,
            environment=environment,
            timeout=1200,
        )
        if not trace.is_file() or trace.stat().st_size == 0:
            raise RuntimeError(f"NVBit did not produce a trace for {cell.key}")
        BASE.write_json(
            destination / "capture.json",
            {
                "cell": cell_document(cell),
                "dynamic_trace_bytes": trace.stat().st_size,
                "stdout_bytes": len(completed.stdout.encode()),
                "stderr_bytes": len(completed.stderr.encode()),
            },
        )
        print(f"nvbit {index:03d}/{len(cells):03d} {cell.key}", flush=True)


def run_static(output: Path) -> None:
    destination = output / "sass"
    destination.mkdir(parents=True, exist_ok=True)
    for binary in (LONG_BINARY, TRACE_BINARY):
        for suffix, option in (("sass", "--dump-sass"), ("resources", "--dump-resource-usage")):
            completed = subprocess.run(
                ["/usr/local/cuda-13.2/bin/cuobjdump", option, str(binary)],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            (destination / f"{binary.name}.{suffix}.txt").write_text(completed.stdout, encoding="utf-8")
            (destination / f"{binary.name}.{suffix}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(f"cuobjdump failed for {binary}")
        BASE.run_captured(
            [
                str(REPO_ROOT / ".venv/bin/python"),
                str(REPO_ROOT / "tools/sass_control/emit_control_bits_sidecar.py"),
                "--binary",
                str(binary),
                "--arch",
                "sm_86",
                "--out",
                str(destination / f"{binary.name}.control_bits.json"),
                "--toolkit-bindir",
                "/usr/local/cuda-13.2/bin",
            ],
            destination / f"{binary.name}.control-extraction",
        )
    print(destination)


def run_simulator(output: Path, cells: Sequence[Cell], force: bool) -> None:
    binary = HELPERS.ensure_release_simulator()
    control = output / "sass/l1tex_admission_backpressure_trace.control_bits.json"
    arch_card = REPO_ROOT / "arch_profiles/cards/sm_86.json"
    pchase_card = REPO_ROOT / "data/pchase/sm_86_static_card.json"
    for index, cell in enumerate(cells, 1):
        destination = output / "simulator" / cell.key
        destination.mkdir(parents=True, exist_ok=True)
        command = [
            str(binary),
            "--trace",
            str(output / "nvbit" / cell.key),
            "--control",
            str(control),
            "--arch-card",
            str(arch_card),
            "--pchase-card",
            str(pchase_card),
            "--mode",
            "l1-l2",
        ]
        outputs: list[bytes] = []
        dependencies = [
            binary,
            control,
            arch_card,
            pchase_card,
            output / "nvbit" / cell.key,
        ]
        expectation = REPLAY.ReplayExpectation(
            launches=REPLAY.expected_launches_from_trace(
                output / "nvbit" / cell.key,
                grid=(cell.blocks, 1, 1),
                block=(cell.warps * 32, 1, 1),
            ),
            carded_dram=False,
        )
        for repeat in (1, 2):
            path = destination / f"replay-{repeat}.json"
            if (
                not force
                and REPLAY.reusable_simulator_output(
                    path,
                    dependencies,
                    expectation,
                )
            ):
                outputs.append(path.read_bytes())
                continue
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            (destination / f"replay-{repeat}.stderr.txt").write_bytes(completed.stderr)
            if completed.returncode != 0:
                raise RuntimeError(f"{cell.key}: replay {repeat} failed")
            path.write_bytes(completed.stdout)
            outputs.append(completed.stdout)
        if outputs[0] != outputs[1]:
            raise RuntimeError(f"{cell.key}: nondeterministic replay")
        print(f"simulator {index:03d}/{len(cells):03d} {cell.key}", flush=True)


def run_memory_service_candidates(output: Path, cells: Sequence[Cell], force: bool) -> None:
    binary = HELPERS.ensure_release_simulator()
    control = output / "sass/l1tex_admission_backpressure_trace.control_bits.json"
    arch_card = REPO_ROOT / "arch_profiles/cards/sm_86.json"
    pchase_card = REPO_ROOT / "data/pchase/sm_86_static_card.json"
    candidates = REPLAY.candidate_configurations(CANDIDATE_FILE)
    for index, cell in enumerate(cells, 1):
        destination = output / "simulator-memory-service-candidates" / cell.key
        destination.mkdir(parents=True, exist_ok=True)
        command = [
            str(binary),
            "--trace",
            str(output / "nvbit" / cell.key),
            "--control",
            str(control),
            "--arch-card",
            str(arch_card),
            "--pchase-card",
            str(pchase_card),
            "--mode",
            "l1-l2",
            "--memory-service-candidates",
            str(CANDIDATE_FILE),
        ]
        outputs: list[bytes] = []
        dependencies = [
            binary,
            control,
            arch_card,
            pchase_card,
            CANDIDATE_FILE,
            output / "metadata/memory_service_candidates.json",
            output / "nvbit" / cell.key,
        ]
        expectation = REPLAY.ReplayExpectation(
            launches=REPLAY.expected_launches_from_trace(
                output / "nvbit" / cell.key,
                grid=(cell.blocks, 1, 1),
                block=(cell.warps * 32, 1, 1),
            ),
            carded_dram=False,
            service_candidates=candidates,
        )
        for repeat in (1, 2):
            path = destination / f"replay-{repeat}.json"
            if (
                not force
                and REPLAY.reusable_simulator_output(
                    path,
                    dependencies,
                    expectation,
                )
            ):
                outputs.append(path.read_bytes())
                continue
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            (destination / f"replay-{repeat}.stderr.txt").write_bytes(completed.stderr)
            if completed.returncode != 0:
                raise RuntimeError(f"{cell.key}: candidate replay {repeat} failed")
            path.write_bytes(completed.stdout)
            outputs.append(completed.stdout)
        if outputs[0] != outputs[1]:
            raise RuntimeError(f"{cell.key}: nondeterministic candidate replay")
        print(f"candidate-replay {index:03d}/{len(cells):03d} {cell.key}", flush=True)


def run_memory_admission_candidates(output: Path, cells: Sequence[Cell], force: bool) -> None:
    binary = HELPERS.ensure_release_simulator()
    control = output / "sass/l1tex_admission_backpressure_trace.control_bits.json"
    arch_card = REPO_ROOT / "arch_profiles/cards/sm_86.json"
    pchase_card = REPO_ROOT / "data/pchase/sm_86_static_card.json"
    candidates = REPLAY.candidate_configurations(ADMISSION_CANDIDATE_FILE)
    for index, cell in enumerate(cells, 1):
        destination = output / "simulator-memory-admission-candidates" / cell.key
        destination.mkdir(parents=True, exist_ok=True)
        command = [
            str(binary),
            "--trace",
            str(output / "nvbit" / cell.key),
            "--control",
            str(control),
            "--arch-card",
            str(arch_card),
            "--pchase-card",
            str(pchase_card),
            "--mode",
            "l1-l2",
            "--memory-admission-candidates",
            str(ADMISSION_CANDIDATE_FILE),
        ]
        outputs: list[bytes] = []
        dependencies = [
            binary,
            control,
            arch_card,
            pchase_card,
            ADMISSION_CANDIDATE_FILE,
            output / "metadata/memory_admission_candidates.json",
            output / "nvbit" / cell.key,
        ]
        expectation = REPLAY.ReplayExpectation(
            launches=REPLAY.expected_launches_from_trace(
                output / "nvbit" / cell.key,
                grid=(cell.blocks, 1, 1),
                block=(cell.warps * 32, 1, 1),
            ),
            carded_dram=False,
            admission_candidates=candidates,
        )
        for repeat in (1, 2):
            path = destination / f"replay-{repeat}.json"
            if (
                not force
                and REPLAY.reusable_simulator_output(
                    path,
                    dependencies,
                    expectation,
                )
            ):
                outputs.append(path.read_bytes())
                continue
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            (destination / f"replay-{repeat}.stderr.txt").write_bytes(completed.stderr)
            if completed.returncode != 0:
                raise RuntimeError(f"{cell.key}: admission candidate replay {repeat} failed")
            path.write_bytes(completed.stdout)
            outputs.append(completed.stdout)
        if outputs[0] != outputs[1]:
            raise RuntimeError(f"{cell.key}: nondeterministic admission candidate replay")
        print(f"admission-candidate-replay {index:03d}/{len(cells):03d} {cell.key}", flush=True)


def selected_feedback_candidates(selector: str | None) -> list[int]:
    if selector is None:
        return list(FEEDBACK_CANDIDATE_FILES)
    requested = {int(item.strip()) for item in selector.split(",") if item.strip()}
    missing = requested - set(FEEDBACK_CANDIDATE_FILES)
    if missing:
        raise RuntimeError(
            f"unknown feedback candidate id(s): {', '.join(map(str, sorted(missing)))}"
        )
    return [candidate_id for candidate_id in FEEDBACK_CANDIDATE_FILES if candidate_id in requested]


def selected_feedback_service_candidates(selector: str | None) -> list[int]:
    candidates = validated_feedback_service_sweep()
    if selector is None:
        return list(candidates)
    requested = {int(item.strip()) for item in selector.split(",") if item.strip()}
    missing = requested - set(candidates)
    if missing:
        raise RuntimeError(
            f"unknown feedback-service candidate id(s): {', '.join(map(str, sorted(missing)))}"
        )
    return [candidate_id for candidate_id in candidates if candidate_id in requested]


def selected_write_completion_candidates(selector: str | None) -> list[int]:
    candidates = validated_write_completion_feedback()
    if selector is None:
        return list(candidates)
    requested = {int(item.strip()) for item in selector.split(",") if item.strip()}
    missing = requested - set(candidates)
    if missing:
        raise RuntimeError(
            f"unknown write-completion candidate id(s): {', '.join(map(str, sorted(missing)))}"
        )
    return [candidate_id for candidate_id in candidates if candidate_id in requested]


def selected_red_rmw_candidates(selector: str | None) -> list[int]:
    candidates = validated_red_rmw_completion_sweep()
    if selector is None:
        return list(candidates)
    requested = {int(item.strip()) for item in selector.split(",") if item.strip()}
    missing = requested - set(candidates)
    if missing:
        raise RuntimeError(
            f"unknown RED RMW candidate id(s): {', '.join(map(str, sorted(missing)))}"
        )
    return [candidate_id for candidate_id in candidates if candidate_id in requested]


def run_memory_admission_feedback(
    output: Path,
    cells: Sequence[Cell],
    force: bool,
    feedback_ids: Sequence[int],
) -> None:
    binary = HELPERS.ensure_release_simulator()
    control = output / "sass/l1tex_admission_backpressure_trace.control_bits.json"
    arch_card = REPO_ROOT / "arch_profiles/cards/sm_86.json"
    pchase_card = REPO_ROOT / "data/pchase/sm_86_static_card.json"
    total = len(cells) * len(feedback_ids)
    progress = 0
    for candidate_id in feedback_ids:
        candidate_file = FEEDBACK_CANDIDATE_FILES[candidate_id]
        feedback_candidate = REPLAY.candidate_configurations(
            candidate_file, require_exactly_one=True
        )[0]
        for cell in cells:
            progress += 1
            destination = output / f"simulator-memory-admission-feedback-{candidate_id}" / cell.key
            destination.mkdir(parents=True, exist_ok=True)
            command = [
                str(binary),
                "--trace",
                str(output / "nvbit" / cell.key),
                "--control",
                str(control),
                "--arch-card",
                str(arch_card),
                "--pchase-card",
                str(pchase_card),
                "--mode",
                "l1-l2",
                "--memory-admission-feedback",
                str(candidate_file),
            ]
            dependencies = [
                binary,
                control,
                arch_card,
                pchase_card,
                candidate_file,
                output / f"metadata/memory_admission_feedback_{candidate_id}.json",
                output / "nvbit" / cell.key,
            ]
            expectation = REPLAY.ReplayExpectation(
                launches=REPLAY.expected_launches_from_trace(
                    output / "nvbit" / cell.key,
                    grid=(cell.blocks, 1, 1),
                    block=(cell.warps * 32, 1, 1),
                ),
                carded_dram=False,
                feedback_candidate=feedback_candidate,
            )
            outputs: list[bytes] = []
            for repeat in (1, 2):
                path = destination / f"replay-{repeat}.json"
                if (
                    not force
                    and REPLAY.reusable_simulator_output(
                        path,
                        dependencies,
                        expectation,
                    )
                ):
                    outputs.append(path.read_bytes())
                    continue
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                (destination / f"replay-{repeat}.stderr.txt").write_bytes(completed.stderr)
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"{cell.key}: feedback candidate {candidate_id} replay {repeat} failed"
                    )
                path.write_bytes(completed.stdout)
                outputs.append(completed.stdout)
            if outputs[0] != outputs[1]:
                raise RuntimeError(
                    f"{cell.key}: nondeterministic feedback candidate {candidate_id} replay"
                )
            print(
                f"feedback-replay {progress:03d}/{total:03d} c{candidate_id} {cell.key}",
                flush=True,
            )


def run_memory_admission_feedback_service_sweep(
    output: Path,
    cells: Sequence[Cell],
    force: bool,
    candidate_ids: Sequence[int],
) -> None:
    binary = HELPERS.ensure_release_simulator()
    control = output / "sass/l1tex_admission_backpressure_trace.control_bits.json"
    arch_card = REPO_ROOT / "arch_profiles/cards/sm_86.json"
    pchase_card = REPO_ROOT / "data/pchase/sm_86_static_card.json"
    candidates = validated_feedback_service_sweep()
    singleton_root = output / "metadata/memory_admission_feedback_service_singletons"
    total = len(cells) * len(candidate_ids)
    progress = 0
    for candidate_id in candidate_ids:
        source_file, candidate = candidates[candidate_id]
        candidate_file = singleton_root / f"candidate-{candidate_id}.json"
        source_snapshot = output / "metadata" / source_file.name
        for cell in cells:
            progress += 1
            destination = (
                output
                / f"simulator-memory-admission-feedback-service-{candidate_id}"
                / cell.key
            )
            destination.mkdir(parents=True, exist_ok=True)
            command = [
                str(binary),
                "--trace",
                str(output / "nvbit" / cell.key),
                "--control",
                str(control),
                "--arch-card",
                str(arch_card),
                "--pchase-card",
                str(pchase_card),
                "--mode",
                "l1-l2",
                "--memory-admission-feedback",
                str(candidate_file),
            ]
            dependencies = [
                binary,
                control,
                arch_card,
                pchase_card,
                source_file,
                source_snapshot,
                candidate_file,
                output / "nvbit" / cell.key,
            ]
            expectation = REPLAY.ReplayExpectation(
                launches=REPLAY.expected_launches_from_trace(
                    output / "nvbit" / cell.key,
                    grid=(cell.blocks, 1, 1),
                    block=(cell.warps * 32, 1, 1),
                ),
                carded_dram=False,
                feedback_candidate=candidate,
            )
            outputs: list[bytes] = []
            for repeat in (1, 2):
                path = destination / f"replay-{repeat}.json"
                if (
                    not force
                    and REPLAY.reusable_simulator_output(path, dependencies, expectation)
                ):
                    outputs.append(path.read_bytes())
                    continue
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                (destination / f"replay-{repeat}.stderr.txt").write_bytes(completed.stderr)
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"{cell.key}: feedback-service candidate {candidate_id} replay {repeat} failed"
                    )
                path.write_bytes(completed.stdout)
                outputs.append(completed.stdout)
            if outputs[0] != outputs[1]:
                raise RuntimeError(
                    f"{cell.key}: nondeterministic feedback-service candidate {candidate_id}"
                )
            print(
                f"feedback-service-replay {progress:04d}/{total:04d} "
                f"c{candidate_id} {cell.key}",
                flush=True,
            )


def run_memory_write_completion_feedback(
    output: Path,
    cells: Sequence[Cell],
    force: bool,
    candidate_ids: Sequence[int],
) -> None:
    binary = HELPERS.ensure_release_simulator()
    control = output / "sass/l1tex_admission_backpressure_trace.control_bits.json"
    arch_card = REPO_ROOT / "arch_profiles/cards/sm_86.json"
    pchase_card = REPO_ROOT / "data/pchase/sm_86_static_card.json"
    candidates = validated_write_completion_feedback()
    total = len(cells) * len(candidate_ids)
    progress = 0
    for candidate_id in candidate_ids:
        candidate_file, candidate = candidates[candidate_id]
        snapshot = output / f"metadata/memory_write_completion_feedback_{candidate_id}.json"
        for cell in cells:
            progress += 1
            destination = (
                output / f"simulator-memory-write-completion-feedback-{candidate_id}" / cell.key
            )
            destination.mkdir(parents=True, exist_ok=True)
            command = [
                str(binary),
                "--trace",
                str(output / "nvbit" / cell.key),
                "--control",
                str(control),
                "--arch-card",
                str(arch_card),
                "--pchase-card",
                str(pchase_card),
                "--mode",
                "l1-l2",
                "--memory-admission-feedback",
                str(candidate_file),
            ]
            dependencies = [
                binary,
                control,
                arch_card,
                pchase_card,
                candidate_file,
                snapshot,
                output / "nvbit" / cell.key,
            ]
            expectation = REPLAY.ReplayExpectation(
                launches=REPLAY.expected_launches_from_trace(
                    output / "nvbit" / cell.key,
                    grid=(cell.blocks, 1, 1),
                    block=(cell.warps * 32, 1, 1),
                ),
                carded_dram=False,
                feedback_candidate=candidate,
            )
            outputs: list[bytes] = []
            for repeat in (1, 2):
                path = destination / f"replay-{repeat}.json"
                if (
                    not force
                    and REPLAY.reusable_simulator_output(path, dependencies, expectation)
                ):
                    outputs.append(path.read_bytes())
                    continue
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                (destination / f"replay-{repeat}.stderr.txt").write_bytes(completed.stderr)
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"{cell.key}: write-completion candidate {candidate_id} replay {repeat} failed"
                    )
                path.write_bytes(completed.stdout)
                outputs.append(completed.stdout)
            if outputs[0] != outputs[1]:
                raise RuntimeError(
                    f"{cell.key}: nondeterministic write-completion candidate {candidate_id}"
                )
            print(
                f"write-completion-replay {progress:03d}/{total:03d} "
                f"c{candidate_id} {cell.key}",
                flush=True,
            )


def run_red_rmw_completion_sweep(
    output: Path,
    cells: Sequence[Cell],
    force: bool,
    candidate_ids: Sequence[int],
) -> None:
    binary = HELPERS.ensure_release_simulator()
    control = output / "sass/l1tex_admission_backpressure_trace.control_bits.json"
    arch_card = REPO_ROOT / "arch_profiles/cards/sm_86.json"
    pchase_card = REPO_ROOT / "data/pchase/sm_86_static_card.json"
    candidates = validated_red_rmw_completion_sweep()
    singleton_root = output / "metadata/red_rmw_completion_singletons"
    total = len(cells) * len(candidate_ids)
    progress = 0
    for candidate_id in candidate_ids:
        source_file, candidate = candidates[candidate_id]
        candidate_file = singleton_root / f"candidate-{candidate_id}.json"
        source_snapshot = output / "metadata" / source_file.name
        for cell in cells:
            progress += 1
            destination = output / f"simulator-red-rmw-completion-{candidate_id}" / cell.key
            destination.mkdir(parents=True, exist_ok=True)
            command = [
                str(binary),
                "--trace",
                str(output / "nvbit" / cell.key),
                "--control",
                str(control),
                "--arch-card",
                str(arch_card),
                "--pchase-card",
                str(pchase_card),
                "--mode",
                "l1-l2",
                "--memory-admission-feedback",
                str(candidate_file),
            ]
            dependencies = [
                binary,
                control,
                arch_card,
                pchase_card,
                source_file,
                source_snapshot,
                candidate_file,
                output / "nvbit" / cell.key,
            ]
            expectation = REPLAY.ReplayExpectation(
                launches=REPLAY.expected_launches_from_trace(
                    output / "nvbit" / cell.key,
                    grid=(cell.blocks, 1, 1),
                    block=(cell.warps * 32, 1, 1),
                ),
                carded_dram=False,
                feedback_candidate=candidate,
            )
            outputs: list[bytes] = []
            for repeat in (1, 2):
                path = destination / f"replay-{repeat}.json"
                if (
                    not force
                    and REPLAY.reusable_simulator_output(path, dependencies, expectation)
                ):
                    outputs.append(path.read_bytes())
                    continue
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                (destination / f"replay-{repeat}.stderr.txt").write_bytes(completed.stderr)
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"{cell.key}: RED RMW candidate {candidate_id} replay {repeat} failed"
                    )
                path.write_bytes(completed.stdout)
                outputs.append(completed.stdout)
            if outputs[0] != outputs[1]:
                raise RuntimeError(f"{cell.key}: nondeterministic RED RMW candidate {candidate_id}")
            print(
                f"red-rmw-replay {progress:04d}/{total:04d} c{candidate_id} {cell.key}",
                flush=True,
            )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "action",
        choices=("manifest", "correctness", "native", "ncu", "source", "nvbit", "static", "simulator", "memory-service-candidates", "memory-admission-candidates", "memory-admission-feedback", "memory-admission-feedback-service-sweep", "memory-write-completion-feedback", "red-rmw-completion-sweep", "keys"),
    )
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--cells")
    result.add_argument("--trials", type=int, default=3)
    result.add_argument("--repeat-start", type=int, default=1)
    result.add_argument("--repeats", type=int, default=3)
    result.add_argument("--feedback-ids")
    result.add_argument("--service-sweep-ids")
    result.add_argument("--completion-ids")
    result.add_argument("--rmw-ids")
    result.add_argument("--force", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = args.output_root.resolve()
    try:
        if args.action == "keys":
            print(",".join(cell.key for cell in configured_cells()))
            return 0
        emit_manifest(
            output,
            refresh_environment=args.action == "manifest" and args.force,
        )
        cells = selected_cells(args.cells)
        if args.action == "manifest":
            print(output / "metadata/matrix.json")
        elif args.action == "correctness":
            run_correctness(output, cells, args.trials)
        elif args.action == "native":
            run_native_timing(output, cells, args.repeats, args.force)
        elif args.action == "ncu":
            run_ncu(output, cells, args.repeat_start, args.repeats, args.force)
        elif args.action == "source":
            run_source(output, args.force)
        elif args.action == "nvbit":
            run_nvbit(output, cells, args.force)
        elif args.action == "static":
            run_static(output)
        elif args.action == "simulator":
            run_simulator(output, cells, args.force)
        elif args.action == "memory-service-candidates":
            run_memory_service_candidates(output, cells, args.force)
        elif args.action == "memory-admission-candidates":
            run_memory_admission_candidates(output, cells, args.force)
        elif args.action == "memory-admission-feedback":
            run_memory_admission_feedback(
                output,
                cells,
                args.force,
                selected_feedback_candidates(args.feedback_ids),
            )
        elif args.action == "memory-admission-feedback-service-sweep":
            run_memory_admission_feedback_service_sweep(
                output,
                cells,
                args.force,
                selected_feedback_service_candidates(args.service_sweep_ids),
            )
        elif args.action == "memory-write-completion-feedback":
            run_memory_write_completion_feedback(
                output,
                cells,
                args.force,
                selected_write_completion_candidates(args.completion_ids),
            )
        elif args.action == "red-rmw-completion-sweep":
            run_red_rmw_completion_sweep(
                output,
                cells,
                args.force,
                selected_red_rmw_candidates(args.rmw_ids),
            )
    except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(f"L1TEX admission campaign: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
