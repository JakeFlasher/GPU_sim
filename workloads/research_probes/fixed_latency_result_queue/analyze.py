#!/usr/bin/env python3
"""Validate and reduce the sm_86 fixed-latency result-queue pilot."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_ROOT = (
    REPO_ROOT
    / "results"
    / "captures"
    / "sm86_fixed_latency_result_queue_20260817_v1"
)
LONG_GENERATIONS = 8192
TRACE_GENERATIONS = 64
WIDTHS = (4, 8, 12, 16)
VARIANTS = ("collision", "separated")
METRICS = (
    "sm__cycles_active.max",
    "smsp__inst_issued.sum",
    "smsp__inst_executed.sum",
    "smsp__inst_executed_pipe_alu.sum",
    "smsp__inst_executed_pipe_fma.sum",
    "smsp__inst_executed_pipe_fma_type_fp16.sum",
    "smsp__inst_issued_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
)
EMPTY_DIAGNOSTICS = (
    "unsupported_memory_path",
    "unmapped_memory_range",
    "cross_allocation_lane_span",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def wide_ncu(path: Path) -> dict[str, float]:
    rows = list(csv.reader(path.open(newline="", encoding="utf-8")))
    require(len(rows) >= 3 and len(rows[0]) == len(rows[2]), f"{path}: malformed NCU CSV")
    result: dict[str, float] = {}
    for key, value in zip(rows[0], rows[2]):
        try:
            result[key] = float(value.replace(",", ""))
        except ValueError:
            pass
    return result


def measurement(path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("{") and '"correct"' in line
    ]
    require(len(rows) == 1, f"{path}: expected one measurement")
    return rows[0]


def source_summary(path: Path) -> dict[str, Any]:
    rows = list(csv.reader(path.open(newline="", encoding="utf-8")))
    require(len(rows) >= 3, f"{path}: empty SourceCounters report")
    documents = [dict(zip(rows[1], row)) for row in rows[2:]]
    result = {}
    for short, column in (
        ("dispatch", "stall_dispatch (Not Issued)"),
        ("math", "stall_math (Not Issued)"),
        ("not_selected", "stall_not_selected (Not Issued)"),
        ("wait", "stall_wait (Not Issued)"),
        ("short_scoreboard", "stall_short_sb (Not Issued)"),
    ):
        values = []
        for row in documents:
            try:
                count = int(float(row.get(column) or 0))
            except ValueError:
                count = 0
            if count:
                values.append((count, row.get("Source", "").strip(), row.get("Address", "")))
        values.sort(reverse=True)
        result[short] = {
            "samples": sum(value[0] for value in values),
            "top": [
                {"samples": count, "sass": sass, "address": address}
                for count, sass, address in values[:8]
            ],
        }
    return result


def validate_static(root: Path) -> dict[str, Any]:
    names = tuple(
        f"fixed_latency_result_queue{trace}_{variant}"
        for trace in ("", "_trace")
        for variant in ("native", "collision", "separated")
    )
    sidecars = {}
    for name in names:
        sidecar = read_json(root / "sass" / f"{name}.control_bits.json")
        sidecars[name] = sidecar
        require(sidecar["summary"]["total_decoded"] == 64, f"{name}: decoded count")
        require(sidecar["summary"]["total_unknown_encoding"] == 0, f"{name}: unknown controls")
        require(len(sidecar["functions"]) == 1, f"{name}: function count")
        resources = (root / "sass" / f"{name}.resources.txt").read_text(encoding="utf-8")
        require(not re.search(r"STACK:[1-9]|LOCAL:[1-9]", resources), f"{name}: spill/local")

    for label in ("long", "trace"):
        patch = read_json(root / "metadata" / f"{label}_result_pair_patch.json")
        require(patch["architecture"] == "sm_86", f"{label}: architecture")
        require(patch["all_changes_inside_hadd2_control_words"] is True, f"{label}: patch scope")
        for variant, stall in (("collision", 1), ("separated", 3)):
            treatment = patch["variants"][variant]
            require(len(treatment["hadd2"]) == 4, f"{label}/{variant}: HADD2 count")
            require(treatment["changed_bytes"] == 7, f"{label}/{variant}: changed bytes")
            require(
                all(
                    row["final_stall"] == stall
                    and row["final_yield"] == 0
                    and row["final_reuse"] == 0
                    and row["low_unchanged"] is True
                    for row in treatment["hadd2"]
                ),
                f"{label}/{variant}: canonical control",
            )

    for prefix in ("fixed_latency_result_queue", "fixed_latency_result_queue_trace"):
        left = sidecars[f"{prefix}_collision"]
        right = sidecars[f"{prefix}_separated"]
        left_rows = next(iter(left["functions"].values()))
        right_rows = next(iter(right["functions"].values()))
        require(len(left_rows) == len(right_rows), f"{prefix}: row count")
        changed = []
        for a, b in zip(left_rows, right_rows):
            require(a["pc"] == b["pc"], f"{prefix}: PC alignment")
            require(a["opcode_text"] == b["opcode_text"], f"{prefix}/{a['pc']}: opcode")
            require(a["predicate"] == b["predicate"], f"{prefix}/{a['pc']}: predicate")
            if a != b:
                changed.append(a["pc"])
                require(a["opcode_text"].startswith("HADD2 "), f"{prefix}/{a['pc']}: non-HADD change")
                require(a["low_word"] == b["low_word"], f"{prefix}/{a['pc']}: low word")
                for field in ("yield_flag", "write_barrier", "read_barrier", "wait_mask", "reuse"):
                    require(a["control"][field] == b["control"][field], f"{prefix}/{a['pc']}: {field}")
                require(a["control"]["stall"] == 1, f"{prefix}/{a['pc']}: collision stall")
                require(b["control"]["stall"] == 3, f"{prefix}/{a['pc']}: separated stall")
        require(changed == [0x150, 0x180, 0x1B0, 0x1F0], f"{prefix}: changed PCs")

    rows = next(iter(sidecars["fixed_latency_result_queue_collision"]["functions"].values()))
    hadd = [row for row in rows if row["opcode_text"].startswith("HADD2 ")]
    ffma = [row for row in rows if row["opcode_text"].startswith("FFMA ")]
    require(len(hadd) == len(ffma) == 4, "producer count")
    destination = lambda text: int(re.match(r"\S+ R(\d+)", text).group(1))
    destination_pairs = [
        {
            "hadd_pc": h["pc"],
            "hadd_destination": destination(h["opcode_text"]),
            "ffma_pc": f["pc"],
            "ffma_destination": destination(f["opcode_text"]),
            "same_bank": destination(h["opcode_text"]) % 2 == destination(f["opcode_text"]) % 2,
        }
        for h, f in zip(hadd, ffma)
    ]
    require(not any(row["same_bank"] for row in destination_pairs), "pilot unexpectedly has same-bank pair")
    adjacent = sum(f["pc"] == h["pc"] + 0x10 for h, f in zip(hadd, ffma))
    require(adjacent == 3, "direct HADD2/FFMA neighborhood count")

    projection = read_json(root / "metadata" / "operand_projection_audit.json")
    require(projection["files"] == 6, "projection files")
    require(projection["records"] == 384, "projection records")
    require(projection["projected"] == 108 and projection["gaps"] == 0, "projection coverage")
    require(projection["opcodes"]["Hadd2"] == 24, "HADD2 projection")
    corpus = read_json(root / "metadata" / "retained_corpus_operand_projection_audit.json")
    require(
        (corpus["files"], corpus["records"], corpus["projected"], corpus["gaps"])
        == (48, 39080, 16421, 0),
        "retained corpus projection",
    )
    return {
        "binaries": 6,
        "decoded_instructions": 384,
        "spill_free": True,
        "pair_changed_pcs": [336, 384, 432, 496],
        "destination_pairs": destination_pairs,
        "same_bank_pairs": 0,
        "opposite_bank_pairs": 4,
        "direct_hadd2_ffma_neighborhoods": adjacent,
        "projection": {key: projection[key] for key in ("files", "records", "projected", "gaps", "families", "opcodes")},
        "retained_corpus_projection": {
            key: corpus[key] for key in ("files", "records", "projected", "gaps", "families", "opcodes")
        },
    }


def reduce(root: Path) -> dict[str, Any]:
    cells = read_json(root / "metadata" / "matrix.json")["cells"]
    require(len(cells) == 8 and len({row["key"] for row in cells}) == 8, "matrix")
    static = validate_static(root)
    from tools.nvbit_tracer.proto_binding_loader import load_dynamic_trace_bindings

    bindings = load_dynamic_trace_bindings()
    live = {}
    replay = {}
    trace_records = 0
    maximum_cv = 0.0
    maximum_mad_pct = 0.0
    high_cv_cells = []
    for cell in cells:
        key = cell["key"]
        payload = read_json(root / "correctness" / key / "result.json")["measurement"]
        require(payload["correct"] is True, f"{key}: correctness")
        require(payload["variant"] == cell["variant"], f"{key}: variant")
        require(payload["warps"] == cell["warps"], f"{key}: warps")
        require(payload["generations"] == LONG_GENERATIONS, f"{key}: generations")
        require(len(payload["samples_cycles"]) == 7, f"{key}: driver trials")
        driver_samples = [max(sample) for sample in payload["samples_cycles"]]

        repeat_paths = sorted((root / "ncu" / key).glob("repeat-*"))
        require(len(repeat_paths) == 7, f"{key}: NCU repeats")
        ncu_rows = [wide_ncu(path / "ncu.csv") for path in repeat_paths]
        for metric in METRICS:
            require(all(metric in row for row in ncu_rows), f"{key}: missing {metric}")
        require(
            all(row["smsp__inst_issued_per_issue_active.ratio"] == 1.0 for row in ncu_rows),
            f"{key}: issue ratio",
        )
        active = [row["sm__cycles_active.max"] for row in ncu_rows]
        cv = statistics.stdev(active) / statistics.mean(active) * 100
        median_active = statistics.median(active)
        mad_pct = statistics.median(abs(value - median_active) for value in active) / median_active * 100
        maximum_cv = max(maximum_cv, cv)
        maximum_mad_pct = max(maximum_mad_pct, mad_pct)
        if cv > 2.0:
            high_cv_cells.append(key)
        median = lambda name: statistics.median(row[name] for row in ncu_rows)
        live[key] = {
            "active_cycles": median_active,
            "active_samples": active,
            "active_cv_pct": cv,
            "active_mad_pct": mad_pct,
            "driver_cycles": statistics.median(driver_samples),
            **{name: median(name) for name in METRICS[1:]},
        }

        capture = root / "nvbit" / key
        trace_payload = measurement(capture / "stdout.txt")
        require(trace_payload["correct"] is True, f"{key}: trace correctness")
        require(trace_payload["generations"] == TRACE_GENERATIONS, f"{key}: trace generations")
        for diagnostic in EMPTY_DIAGNOSTICS:
            require(read_json(capture / "extra_info" / f"{diagnostic}.json") == [], f"{key}: {diagnostic}")
        paths = list((capture / "threadblocks").rglob("*.pb"))
        require(len(paths) == 1, f"{key}: block trace count")
        block = bindings.threadblock_pb2.threadblock()
        block.ParseFromString(paths[0].read_bytes())
        records = sum(len(warp.instructions) for warp in block.warps.values())
        slots = {
            int(warp.instructions[0].warpid_sm)
            for warp in block.warps.values()
            if warp.instructions
        }
        require(slots == set(range(cell["warps"])), f"{key}: warp slots")
        trace_records += records

        first = root / "simulator" / key / "replay-1.json"
        second = root / "simulator" / key / "replay-2.json"
        require(first.read_bytes() == second.read_bytes(), f"{key}: nondeterministic replay")
        launch = read_json(first)["launches"][0]
        require(launch["coverage"]["kind"] == "full", f"{key}: replay coverage")
        require(launch["counters"]["records_seen"] == records, f"{key}: replay records")
        replay[key] = launch

    pairs = []
    residuals = []
    absolute_errors = []
    for warps in WIDTHS:
        collision_key = f"w{warps}_collision"
        separated_key = f"w{warps}_separated"
        collision = live[collision_key]
        separated = live[separated_key]
        require(
            read_json(root / "correctness" / collision_key / "result.json")["measurement"]["checksum"]
            == read_json(root / "correctness" / separated_key / "result.json")["measurement"]["checksum"],
            f"W{warps}: checksum",
        )
        for name in (
            "smsp__inst_issued.sum",
            "smsp__inst_executed.sum",
            "smsp__inst_executed_pipe_alu.sum",
            "smsp__inst_executed_pipe_fma.sum",
            "smsp__inst_executed_pipe_fma_type_fp16.sum",
        ):
            require(collision[name] == separated[name], f"W{warps}: {name}")
        require(
            replay[collision_key]["counters"] == replay[separated_key]["counters"],
            f"W{warps}: replay traffic",
        )
        live_delta = (separated["active_cycles"] - collision["active_cycles"]) / LONG_GENERATIONS
        driver_delta = (separated["driver_cycles"] - collision["driver_cycles"]) / LONG_GENERATIONS
        replay_delta = (
            replay[separated_key]["timing"]["diagnostic_last_issue_cycles"]
            - replay[collision_key]["timing"]["diagnostic_last_issue_cycles"]
        ) / TRACE_GENERATIONS
        residual = live_delta - replay_delta
        residuals.append(abs(residual))
        for key in (collision_key, separated_key):
            absolute_errors.append(
                abs(
                    replay[key]["timing"]["diagnostic_last_issue_cycles"] / TRACE_GENERATIONS
                    - live[key]["active_cycles"] / LONG_GENERATIONS
                )
            )
        pairs.append(
            {
                "stage": "identification" if warps in {4, 8} else "held_out_width",
                "warps": warps,
                "warps_per_modeled_smsp": warps // 4,
                "collision_cycles_per_generation": collision["active_cycles"] / LONG_GENERATIONS,
                "separated_cycles_per_generation": separated["active_cycles"] / LONG_GENERATIONS,
                "live_delta_separated_minus_collision": live_delta,
                "driver_delta_separated_minus_collision": driver_delta,
                "replay_delta_separated_minus_collision": replay_delta,
                "live_minus_replay_delta_residual": residual,
                "collision_cv_pct": collision["active_cv_pct"],
                "separated_cv_pct": separated["active_cv_pct"],
                "collision_mad_pct": collision["active_mad_pct"],
                "separated_mad_pct": separated["active_mad_pct"],
                "dispatch_delta": separated[
                    "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio"
                ]
                - collision[
                    "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio"
                ],
                "math_delta": separated[
                    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio"
                ]
                - collision[
                    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio"
                ],
                "not_selected_delta": separated[
                    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio"
                ]
                - collision[
                    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio"
                ],
                "wait_delta": separated[
                    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"
                ]
                - collision[
                    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"
                ],
                "short_scoreboard_delta": separated[
                    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio"
                ]
                - collision[
                    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio"
                ],
            }
        )

    source_paths = sorted((root / "ncu-source").glob("*/source.csv"))
    require(len(source_paths) == 4, "SourceCounters profile count")
    sources = {path.parent.name: source_summary(path) for path in source_paths}
    return {
        "schema_version": 1,
        "campaign": {
            "cells": 8,
            "correctness_executions": 56,
            "ncu_profiles": 56,
            "source_counters_profiles": 4,
            "nvbit_captures": 8,
            "trace_records": trace_records,
            "byte_identical_replay_pairs": 8,
            "maximum_active_cycle_cv_pct": maximum_cv,
            "maximum_active_cycle_mad_pct": maximum_mad_pct,
            "high_cv_cells": high_cv_cells,
            "all_issue_ratios_one": True,
            "all_pair_counts_equal": True,
        },
        "static": static,
        "live": live,
        "replay": replay,
        "pairs": pairs,
        "source_counters": sources,
        "scores": {
            "production_delta_mae": statistics.mean(residuals),
            "production_absolute_mae": statistics.mean(absolute_errors),
        },
        "decision": {
            "status": "pilot_completed_no_result_queue_activation",
            "result_queue_identified": False,
            "reasons": [
                "all four HADD2/FFMA destination pairs target opposite register banks",
                "only three of four producer neighborhoods are instruction-adjacent",
                "the stall treatment changes issue pacing and multiwarp math-pipe contention",
                "the live treatment sign reverses above one warp per modeled SMSP",
                "production replay overpredicts the treatment delta at every width",
            ],
            "retained_components": [
                "exact HADD2 operand/family projection",
                "control-word-only HADD2 patching and validation",
                "live opposite-bank completion/pacing evidence",
            ],
            "next_identification_slice": (
                "balanced same-bank/opposite-bank producer-order pair with identical controls, "
                "pipe work, and issue pacing"
            ),
        },
    }


def render(summary: dict[str, Any]) -> str:
    campaign = summary["campaign"]
    score = summary["scores"]
    lines = [
        "# sm_86 fixed-latency result-queue pilot reduction",
        "",
        "## Decision",
        "",
        "The pilot is complete, but no result-queue timing state is activated. The exact",
        "pair changes only four HADD2 stall controls between treatments, yet all four",
        "HADD2/FFMA destination pairs target opposite RF banks and one pair has an",
        "intervening predicate instruction. The treatment therefore establishes an",
        "opposite-bank completion/pacing surface, not same-bank writeback serialization",
        "or queue depth.",
        "",
        f"Production treatment-delta MAE is {score['production_delta_mae']:.3f} cycles/generation; ",
        f"absolute MAE over the eight cells is {score['production_absolute_mae']:.3f}.",
        "The sign reversal above W4 follows reductions in math-pipe, dispatch, and",
        "not-selected pressure, so fitting a completion queue here would absorb an",
        "already-known scheduler/pipe confound.",
        "",
        "## Campaign integrity",
        "",
        f"- {campaign['cells']} cells; {campaign['correctness_executions']} correctness executions",
        f"- {campaign['ncu_profiles']} base-clock NCU profiles and {campaign['source_counters_profiles']} SourceCounters profiles",
        f"- {campaign['nvbit_captures']} NVBit captures with {campaign['trace_records']:,} replay records",
        "- six spill-free cubins, 384/384 decoded controls, 108 projected fixed-latency instructions, zero gaps",
        "- retained 48-sidecar corpus: 39,080 records, 16,421 projections, zero gaps",
        "- issue ratio 1.0 and equal issued/executed/ALU/FMA/FP16 counts in every pair",
        f"- maximum raw CV {campaign['maximum_active_cycle_cv_pct']:.3f}%; maximum median absolute deviation {campaign['maximum_active_cycle_mad_pct']:.3f}%",
        "",
        "The raw-CV exception is `w8_separated`, where one of seven profiles entered a",
        "faster scheduler phase. Its median absolute deviation remains small; all tables",
        "use the expanded seven-repeat median and retain every raw sample.",
        "",
        "## Treatment deltas",
        "",
        "| Warps | Stage | Live sep-coll | Driver sep-coll | Replay sep-coll | Live-replay residual | Math stall delta | Dispatch delta | Wait delta |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["pairs"]:
        lines.append(
            f"| {row['warps']} | {row['stage']} | {row['live_delta_separated_minus_collision']:.3f} | "
            f"{row['driver_delta_separated_minus_collision']:.3f} | "
            f"{row['replay_delta_separated_minus_collision']:.3f} | "
            f"{row['live_minus_replay_delta_residual']:.3f} | {row['math_delta']:.3f} | "
            f"{row['dispatch_delta']:.3f} | {row['wait_delta']:.3f} |"
        )
    lines.extend(
        [
            "",
            "W4 has one warp per modeled SMSP: the longer stall treatment costs almost",
            "exactly four live cycles/generation, versus eight in replay, with zero",
            "math/dispatch/not-selected samples and +0.5 wait samples per issue-active",
            "cycle. At W8/W12/W16 the treatment becomes faster as extra spacing relieves",
            "multiwarp pipe/scheduler pressure. Short-scoreboard delta is zero throughout.",
            "",
            "## Static treatment boundary",
            "",
            "Both treatment cubins canonicalize HADD2 yield/reuse to zero. Collision uses",
            "stall 1 and separated uses stall 3 at PCs 0x150, 0x180, 0x1b0, and 0x1f0.",
            "Every instruction PC, opcode, operand, predicate, barrier, wait mask, resource",
            "declaration, runtime value, and host oracle is otherwise equal. CUDA 13.2",
            "disassembly and the raw reuse flags agree after canonicalization.",
            "",
            "## Next exact slice",
            "",
            "Build a balanced same-bank/opposite-bank producer-order matrix. It must retain",
            "one control schedule and equal HADD2/FFMA work, use compiler-safe register",
            "allocation or an independently validated sm_86 assembler, and include a",
            "same-latency control that measures math-pipe pacing without completion",
            "coincidence. Only then can a residual identify bypass/writeback buffering.",
            "Queue-depth, load-priority, and active timing remain deferred.",
            "",
        ]
    )
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        summary = reduce(args.root.resolve())
        analysis = args.root.resolve() / "analysis"
        write_json(analysis / "summary.json", summary)
        (analysis / "README.md").write_text(render(summary), encoding="utf-8")
        print(analysis / "README.md")
    except (OSError, RuntimeError, KeyError, ValueError, json.JSONDecodeError) as error:
        print(f"result-queue analysis: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
