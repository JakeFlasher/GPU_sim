#!/usr/bin/env python3
"""Validate and reduce the sm_86 multi-SM atomic service campaign."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_ROOT = REPO_ROOT / "results/captures/sm86_atomic_multism_service_20260818_v1"
COARSE_SPACINGS = {0, 128, 256, 4096, 4100}


def load_previous_analysis():
    path = HERE.parent / "atomic_alias_adversarial/analyze.py"
    spec = importlib.util.spec_from_file_location("multism_alias_analysis", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ALIAS = load_previous_analysis()
require = ALIAS.require
read_json = ALIAS.read_json
write_json = ALIAS.write_json
wide_ncu = ALIAS.wide_ncu
measurement = ALIAS.measurement
source_stalls = ALIAS.source_stalls
median_max = ALIAS.median_max
ape = ALIAS.ape


def validate_static(root: Path) -> dict[str, Any]:
    expected = {
        "ret32": (48, 18, "ATOMG"),
        "red32": (40, 14, "RED"),
    }
    decoded = 0
    function = "_Z20atomic_multism_probePyS_PjPKj"
    for variant, (instructions, registers, opcode) in expected.items():
        for trace in (False, True):
            infix = "_trace" if trace else ""
            name = f"atomic_multism_service{infix}_{variant}"
            sidecar = read_json(root / "sass" / f"{name}.control_bits.json")
            require(sidecar["summary"]["total_decoded"] == instructions, f"{name}: controls")
            require(sidecar["summary"]["total_unknown_encoding"] == 0, f"{name}: decode")
            require(set(sidecar["functions"]) == {function}, f"{name}: function")
            rows = sidecar["functions"][function]
            atomics = [row for row in rows if row["opcode_text"].startswith(("ATOMG", "RED"))]
            require(len(atomics) == 1 and atomics[0]["opcode_text"].startswith(opcode), f"{name}: atomic")
            require(sum("SR_CLOCKLO" in row["opcode_text"] for row in rows) == 2, f"{name}: clocks")
            require(any(row["opcode_text"].startswith("BAR.SYNC") for row in rows), f"{name}: barrier")
            resources = (root / "sass" / f"{name}.resources.txt").read_text(encoding="utf-8")
            require(f"REG:{registers}" in resources and not re.search(r"STACK:[1-9]|LOCAL:[1-9]", resources), f"{name}: resources")
            decoded += instructions
    local = read_json(root / "metadata/operand_projection_audit.json")
    retained = read_json(root / "metadata/retained_corpus_operand_projection_audit.json")
    require((local["files"], local["records"], local["projected"], local["gaps"]) == (4, 176, 18, 0), "local projection")
    require((retained["files"], retained["records"], retained["projected"], retained["gaps"]) == (181, 49_968, 18_562, 0), "retained projection")
    return {
        "binaries": 4,
        "decoded_instructions": decoded,
        "spill_free": True,
        "local_projection": {key: local[key] for key in ("files", "records", "projected", "gaps")},
        "retained_projection": {key: retained[key] for key in ("files", "records", "projected", "gaps")},
    }


def hash_value(value: int, buckets: int, kind: str) -> int:
    if kind == "line_mod":
        return value % buckets
    if kind == "sector_mod":
        return (value * 4) % buckets
    if kind == "xor_folded":
        return (value ^ (value >> 5) ^ (value >> 11)) % buckets
    if kind == "knuth_mult":
        return ((value * 2_654_435_761) & 0xFFFF_FFFF) % buckets
    if kind == "shift_mod":
        return (value >> 1) % buckets
    if kind == "bit_reverse":
        return int(f"{value & 0xFFFF:016b}"[::-1], 2) % buckets
    raise AssertionError(kind)


def maximum_bucket_occupancy(blocks: int, spacing: int, domain: str, buckets: int, kind: str) -> int:
    divisor = {"line": 128, "sector": 32, "word": 4}[domain]
    mapped = [hash_value((block * spacing) // divisor, buckets, kind) for block in range(blocks)]
    return max(Counter(mapped).values())


def fit_hash_candidates(clock: dict[str, float], fine_spacings: list[int]) -> dict[str, Any]:
    candidates = []
    for domain, buckets, kind in itertools.product(
        ("line", "sector", "word"),
        (6, 12, 24, 48),
        ("line_mod", "sector_mod", "xor_folded", "knuth_mult", "shift_mod", "bit_reverse"),
    ):
        x = [maximum_bucket_occupancy(84, spacing, domain, buckets, kind) for spacing in fine_spacings]
        y = [clock[f"b84_red_s{spacing}_d"] for spacing in fine_spacings]
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(y)
        denominator = sum((value - x_mean) ** 2 for value in x)
        slope = sum((left - x_mean) * (right - y_mean) for left, right in zip(x, y)) / denominator if denominator else 0.0
        intercept = y_mean - slope * x_mean
        train = [max(0.0, intercept + slope * value) for value in x]
        train_errors = [ape(predicted, observed) for predicted, observed in zip(train, y)]
        transfer_errors = []
        for blocks in (4, 16, 42, 168):
            for spacing in sorted(COARSE_SPACINGS):
                predicted = max(0.0, intercept + slope * maximum_bucket_occupancy(blocks, spacing, domain, buckets, kind))
                transfer_errors.append(ape(predicted, clock[f"b{blocks}_red_s{spacing}_d"]))
        candidates.append({
            "domain": domain,
            "buckets": buckets,
            "hash": kind,
            "intercept": intercept,
            "slope": slope,
            "training_mape_pct": statistics.mean(train_errors),
            "training_max_ape_pct": max(train_errors),
            "transfer_mape_pct": statistics.mean(transfer_errors),
            "transfer_max_ape_pct": max(transfer_errors),
        })
    candidates.sort(key=lambda row: (row["training_mape_pct"], row["transfer_mape_pct"]))
    return {"tested": len(candidates), "best": candidates[0], "top_five": candidates[:5]}


def reduce(root: Path) -> dict[str, Any]:
    manifest = read_json(root / "metadata/matrix.json")
    cells = manifest["cells"]
    require(len(cells) == 106 and len({entry["key"] for entry in cells}) == 106, "matrix")
    static = validate_static(root)
    from tools.nvbit_tracer.proto_binding_loader import load_dynamic_trace_bindings

    bindings = load_dynamic_trace_bindings()
    live: dict[str, dict[str, float]] = {}
    clock: dict[str, float] = {}
    replay_clock: dict[str, float] = {}
    placement: dict[str, Any] = {}
    trace_records = 0
    trace_blocks = 0
    ncu_profiles = 0
    maximum_cv = 0.0
    maximum_mad = 0.0
    maximum_clock_range = 0
    required = (
        "sm__cycles_active.max",
        "smsp__inst_issued_per_issue_active.ratio",
        "smsp__inst_executed_op_generic_atom_dot_alu.sum",
        "smsp__inst_executed_op_global_red.sum",
        "l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum",
        "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum",
        "lts__t_requests_srcunit_tex_op_atom_dot_alu.sum",
        "lts__t_requests_srcunit_tex_op_red.sum",
        "lts__t_sectors_op_atom.sum",
        "lts__t_sectors_op_red.sum",
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    )

    for entry in cells:
        key = entry["key"]
        long = read_json(root / "correctness" / key / "result.json")["measurement"]
        trace = read_json(root / "trace-correctness" / key / "result.json")["measurement"]
        for payload, generations, label in ((long, 1024, "long"), (trace, 64, "trace")):
            require(payload["correct"] is True and payload["blocks"] == entry["blocks"], f"{key}: {label} correctness")
            require(payload["generations"] == generations and len(payload["samples_cycles"]) == 7, f"{key}: {label} shape")
        maximum_clock_range = max(maximum_clock_range, max(max(row) for row in trace["samples_cycles"]) - min(max(row) for row in trace["samples_cycles"]))
        clock[key] = median_max(trace)

        repeat_paths = sorted((root / "ncu" / key).glob("repeat-*"))
        expanded = entry["blocks"] == 168 or (entry["blocks"] == 84 and (entry["spacing_bytes"] in COARSE_SPACINGS or entry["permuted"]))
        expected_repeats = 3 if expanded else 1
        require(len(repeat_paths) == expected_repeats, f"{key}: NCU repeats")
        ncu_profiles += len(repeat_paths)
        ncu = [wide_ncu(path / "ncu.csv") for path in repeat_paths]
        for metric in required:
            require(all(metric in row for row in ncu), f"{key}: {metric}")
        med = {metric: statistics.median(row[metric] for row in ncu) for metric in required}
        require(all(row["smsp__inst_issued_per_issue_active.ratio"] == 1.0 for row in ncu), f"{key}: issue ratio")
        if len(ncu) > 1:
            samples = [row["sm__cycles_active.max"] for row in ncu]
            center = statistics.median(samples)
            maximum_cv = max(maximum_cv, statistics.stdev(samples) / statistics.mean(samples) * 100)
            maximum_mad = max(maximum_mad, statistics.median(abs(value - center) for value in samples) / center * 100)
        warp_instructions = entry["blocks"] * 1024
        operation = "atom" if entry["returning"] else "red"
        opposite = "red" if entry["returning"] else "atom"
        instruction_metric = "smsp__inst_executed_op_generic_atom_dot_alu.sum" if entry["returning"] else "smsp__inst_executed_op_global_red.sum"
        opposite_metric = "smsp__inst_executed_op_global_red.sum" if entry["returning"] else "smsp__inst_executed_op_generic_atom_dot_alu.sum"
        require(med[instruction_metric] == warp_instructions and med[opposite_metric] == 0, f"{key}: operation count")
        require(med[f"l1tex__t_requests_pipe_lsu_mem_global_op_{operation}.sum"] == warp_instructions and med[f"l1tex__t_requests_pipe_lsu_mem_global_op_{opposite}.sum"] == 0, f"{key}: L1 requests")
        expected_atomic_work = warp_instructions * 32
        require(med[f"l1tex__t_sectors_pipe_lsu_mem_global_op_{operation}.sum"] == expected_atomic_work, f"{key}: L1 sectors")
        require(med[f"lts__t_sectors_op_{operation}.sum"] == expected_atomic_work, f"{key}: L2 sectors")
        l2_request_metric = "lts__t_requests_srcunit_tex_op_atom_dot_alu.sum" if entry["returning"] else "lts__t_requests_srcunit_tex_op_red.sum"
        require(med[l2_request_metric] == expected_atomic_work, f"{key}: L2 requests")
        live[key] = {
            "active_cycles": med["sm__cycles_active.max"],
            "l2_sectors": med[f"lts__t_sectors_op_{operation}.sum"],
            "l2_requests": med[l2_request_metric],
            "long_scoreboard": med["smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"],
        }

        capture = root / "nvbit" / key
        require(measurement(capture / "stdout.txt")["correct"] is True, f"{key}: trace execution")
        for diagnostic in ("unsupported_memory_path", "unmapped_memory_range", "cross_allocation_lane_span"):
            require(read_json(capture / "extra_info" / f"{diagnostic}.json") == [], f"{key}: {diagnostic}")
        paths = list((capture / "threadblocks").rglob("*.pb"))
        require(len(paths) == entry["blocks"], f"{key}: CTA payloads")
        sm_counts: Counter[int] = Counter()
        records = 0
        for path in paths:
            block = bindings.threadblock_pb2.threadblock()
            block.ParseFromString(path.read_bytes())
            block_records = sum(len(warp.instructions) for warp in block.warps.values())
            records += block_records
            first = next(warp.instructions[0] for warp in block.warps.values() if warp.instructions)
            sm_counts[int(first.sm_id)] += 1
        expected_sms = min(entry["blocks"], 84)
        require(len(sm_counts) == expected_sms, f"{key}: SM coverage")
        if entry["blocks"] <= 84:
            require(set(sm_counts.values()) == {1}, f"{key}: one CTA per SM")
        else:
            require(set(sm_counts.values()) == {2}, f"{key}: two CTA waves")
        placement[key] = {"sms": len(sm_counts), "min_ctas_per_sm": min(sm_counts.values()), "max_ctas_per_sm": max(sm_counts.values())}
        trace_records += records
        trace_blocks += len(paths)

        first_path = root / "simulator" / key / "replay-1.json"
        second_path = root / "simulator" / key / "replay-2.json"
        require(first_path.read_bytes() == second_path.read_bytes(), f"{key}: replay determinism")
        launch = read_json(first_path)["launches"][0]
        require(launch["coverage"]["kind"] == "full" and launch["counters"]["records_seen"] == records, f"{key}: replay coverage")
        expected_trace_work = entry["blocks"] * 64 * 32
        l1 = launch["counters"]["l1"]
        replay_work = l1["bypass_sectors"] if entry["returning"] else l1["reduction_sectors"]
        require(replay_work == expected_trace_work, f"{key}: replay atomic sectors")
        clock_diag = launch["timing"]["readiness"]["clock_issue"]
        require(clock_diag["reads_issued"] == entry["blocks"] * 2 and clock_diag["warps_with_interval"] == entry["blocks"], f"{key}: replay CLOCK bounds")
        replay_clock[key] = clock_diag["maximum_first_to_last_issue_cycles"]

    require(ncu_profiles == 162, "NCU profile total")
    source_paths = sorted((root / "ncu-source").glob("*/source.csv"))
    require(len(source_paths) == 14, "SourceCounters total")
    source_summary = {path.parent.name: source_stalls(path) for path in source_paths}

    direct_permuted = []
    for returning in (True, False):
        op = "ret" if returning else "red"
        for spacing in (128, 256, 4096, 4100):
            direct = clock[f"b84_{op}_s{spacing}_d"]
            permuted = clock[f"b84_{op}_s{spacing}_p"]
            direct_permuted.append({"operation": op, "spacing_bytes": spacing, "delta_cycles": permuted - direct, "relative_pct": (permuted / direct - 1) * 100})

    per_sm_errors = []
    for entry in cells:
        if entry["blocks"] == 1:
            continue
        base = clock[f"b1_{'ret' if entry['returning'] else 'red'}_s0_d"]
        per_sm_errors.append(ape(base, clock[entry["key"]]))

    red_work_per_warp = 64 * (32 + 32 / 8)
    effective_units = []
    for blocks in (84, 168):
        for spacing in (4096, 4100):
            observed = clock[f"b{blocks}_red_s{spacing}_d"]
            effective_units.append(blocks * red_work_per_warp / observed)
    fine_spacings = sorted({entry["spacing_bytes"] for entry in cells if entry["blocks"] == 84 and not entry["permuted"] and entry["spacing_bytes"] != 0})
    hash_candidates = fit_hash_candidates(clock, fine_spacings)

    b84_ratio = {
        op: clock[f"b84_{op}_s0_d"] / clock[f"b84_{op}_s4096_d"]
        for op in ("ret", "red")
    }
    rows = []
    for entry in cells:
        key = entry["key"]
        rows.append({
            "key": key,
            "blocks": entry["blocks"],
            "operation": entry["operation"],
            "spacing_bytes": entry["spacing_bytes"],
            "permuted": entry["permuted"],
            "stage": entry["stage"],
            "clock_cycles": clock[key],
            "replay_clock_cycles": replay_clock[key],
            "replay_ape_pct": ape(replay_clock[key], clock[key]),
            **live[key],
            **placement[key],
        })
    return {
        "schema_version": 1,
        "campaign": {
            "cells": 106,
            "long_correctness_executions": 742,
            "trace_correctness_executions": 742,
            "ncu_profiles": ncu_profiles,
            "source_counters_profiles": 14,
            "nvbit_captures": 106,
            "trace_records": trace_records,
            "threadblock_payloads": trace_blocks,
            "byte_identical_replay_pairs": 106,
            "maximum_expanded_active_cycle_cv_pct": maximum_cv,
            "maximum_expanded_active_cycle_mad_pct": maximum_mad,
            "maximum_trace_clock_range_cycles": maximum_clock_range,
        },
        "static": static,
        "cells": rows,
        "source_counters": source_summary,
        "placement": placement,
        "direct_permuted_pairs": direct_permuted,
        "per_sm_only_candidate": {"status": "rejected", "mape_pct": statistics.mean(per_sm_errors), "max_ape_pct": max(per_sm_errors)},
        "effective_atomic_units": {"estimates": effective_units, "median": statistics.median(effective_units), "interpretation": "throughput bound, not an identified address hash"},
        "ordinary_l2_hash_candidates": hash_candidates,
        "same_work_address_sensitivity": {"b84_shared_over_4096_ratio": b84_ratio},
        "decision": {
            "status": "cross_sm_atomic_service_confirmed_address_hash_unidentified_timing_withheld",
            "promoted": [
                "exact multi-SM CLOCK-boundary diagnostics",
                "exact one-CTA-per-SM and two-wave placement evidence",
            ],
            "rejected": [
                "a per-SM-only atomic service queue",
                "ordinary L2 slice hashes as atomic-unit mappings",
                "a universal 128-byte no-contention rule",
                "a GA102 4096-byte page-offset contention penalty",
                "block-to-address order as a timing correction",
            ],
            "deferred": [
                "distributed atomic-unit address hash",
                "cross-SM ATOMG/RED service timing",
            ],
            "next_slice": "pairwise address-collision graph if atomic timing becomes higher priority; otherwise resume launch-cycle aggregation",
        },
    }


def render(summary: dict[str, Any]) -> str:
    c = summary["campaign"]
    rows = {row["key"]: row for row in summary["cells"]}
    hash_best = summary["ordinary_l2_hash_candidates"]["best"]
    lines = [
        "# sm_86 multi-SM atomic service and address mapping",
        "",
        "## Decision",
        "",
        "Cross-SM atomic service is real and address-sensitive. Reject a per-SM-only",
        "queue and reject every retained ordinary L2 slice hash as an atomic-unit map.",
        "The campaign bounds about 24 effective units under well-distributed traffic,",
        "but does not identify their hash; no atomic timing queue is activated.",
        "",
        "## Campaign integrity",
        "",
        f"- {c['cells']} cells; {c['long_correctness_executions']} long and {c['trace_correctness_executions']} trace-length correctness executions",
        f"- {c['ncu_profiles']} NCU profiles, {c['source_counters_profiles']} SourceCounters profiles, and {c['nvbit_captures']} NVBit captures",
        f"- {c['trace_records']:,} records in {c['threadblock_payloads']:,} CTA payloads and {c['byte_identical_replay_pairs']} byte-identical replay pairs",
        f"- maximum expanded-cell NCU CV {c['maximum_expanded_active_cycle_cv_pct']:.3f}% and MAD {c['maximum_expanded_active_cycle_mad_pct']:.3f}%",
        "- exact placement: B42/B84 use 42/84 distinct SMs; B168 places exactly two CTAs on each of 84 SMs",
        "",
        "## Coarse trace CLOCK timing",
        "",
        "| Op | Blocks | shared word | 128 B | 256 B | 4096 B | 4100 B |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for op in ("ret", "red"):
        for blocks in (4, 16, 42, 84, 168):
            values = [rows[f"b{blocks}_{op}_s{spacing}_d"]["clock_cycles"] for spacing in (0, 128, 256, 4096, 4100)]
            lines.append(f"| {op.upper()} | {blocks} | " + " | ".join(f"{value:.0f}" for value in values) + " |")
    lines.extend([
        "",
        "At B84, identical instruction/request/sector work differs by 7.36× for",
        "ATOMG and 17.15× for RED between a shared word and 4096-byte spacing.",
        "Counts alone therefore cannot price cross-SM atomics.",
        "",
        "## Fine B84 spacing sweep",
        "",
        "| Spacing (B) | RET cycles | RED cycles |",
        "|---:|---:|---:|",
    ])
    fine = sorted({row["spacing_bytes"] for row in summary["cells"] if row["blocks"] == 84 and not row["permuted"] and row["spacing_bytes"] != 0})
    for spacing in fine:
        lines.append(f"| {spacing} | {rows[f'b84_ret_s{spacing}_d']['clock_cycles']:.0f} | {rows[f'b84_red_s{spacing}_d']['clock_cycles']:.0f} |")
    lines.extend([
        "",
        "Returning traffic reaches its latency floor near 64–128 bytes, while RED",
        "remains strongly and irregularly address-sensitive. Exactly 4096 and 4100",
        "bytes are equivalent on GA102, rejecting the AMD page-offset effect as a",
        "portable rule. Direct/permuted address-order pairs are also null.",
        "",
        "## Atomic-unit bound and hash rejection",
        "",
        f"Well-distributed B84/B168 RED implies {summary['effective_atomic_units']['median']:.2f} effective parallel units (median throughput bound).",
        "This is not an address mapping. The 72 tested ordinary-cache candidates",
        "cross line/sector/word domains, 6/12/24/48 buckets, and six retained hashes.",
        f"The best ({hash_best['domain']}, {hash_best['buckets']} buckets, {hash_best['hash']}) still has {hash_best['training_mape_pct']:.1f}% fine-sweep MAPE and {hash_best['transfer_mape_pct']:.1f}% block-count transfer MAPE.",
        "Using the simulator's ordinary L2 slice hash for atomic timing is rejected.",
        "",
        "## Simulator consequence",
        "",
        f"A per-SM-only candidate has {summary['per_sm_only_candidate']['mape_pct']:.1f}% MAPE (max {summary['per_sm_only_candidate']['max_ape_pct']:.1f}%) across non-baseline cells.",
        "The exact sector/request projections and CLOCK diagnostics remain active, but",
        "timing must wait for a dedicated pairwise address-collision graph or another",
        "direct way to identify the distributed atomic-unit hash. No per-spacing fit is",
        "installed.",
        "",
    ])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    try:
        summary = reduce(args.root.resolve())
        write_json(args.root.resolve() / "analysis/summary.json", summary)
        report = args.root.resolve() / "analysis/README.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(render(summary), encoding="utf-8")
        print(report)
    except (OSError, RuntimeError, json.JSONDecodeError, csv.Error) as error:
        print(f"multi-SM atomic analysis: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
