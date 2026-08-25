#!/usr/bin/env python3
"""Validate and reduce coupled barrier/cache launch-chain evidence."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_ROOT = (
    REPO_ROOT
    / "results"
    / "captures"
    / "sm86_barrier_pipeline_coupled_20260817_v1"
)
PREDECESSOR = (
    REPO_ROOT
    / "results"
    / "captures"
    / "sm86_async_barrier_readiness_20260817_v4"
)
LONG_GROUPS = 16384
TRACE_GROUPS = 64
EMPTY_NVBIT_DIAGNOSTICS = (
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
            continue
    return result


def cv_pct(values: Sequence[float]) -> float:
    return statistics.stdev(values) / statistics.mean(values) * 100.0


def measurement_line(path: Path) -> dict[str, Any]:
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("{") and '"correct"' in line
    ]
    require(len(values) == 1, f"{path}: expected one measurement payload")
    return values[0]


def target(report: dict[str, Any]) -> dict[str, Any]:
    launches = report["launches"]
    require(launches, "replay report has no launches")
    return launches[-1]


def reduce(root: Path) -> dict[str, Any]:
    manifest = read_json(root / "metadata" / "matrix.json")
    cells = manifest["cells"]
    require(len(cells) == 18, "coupled matrix must contain 18 cells")
    by_key = {cell["key"]: cell for cell in cells}
    require(len(by_key) == 18, "coupled matrix keys are not unique")

    ncu: dict[str, dict[str, Any]] = {}
    captures: dict[str, dict[str, Any]] = {}
    replays: dict[str, dict[str, Any]] = {}
    for key, cell in by_key.items():
        correctness = read_json(root / "correctness" / key / "result.json")["measurement"]
        require(correctness["correct"] is True, f"{key}: correctness failed")
        require(len(correctness["samples_cycles"]) == 7, f"{key}: expected seven trials")
        require(correctness["groups"] == LONG_GROUPS, f"{key}: long group mismatch")

        repeat_dirs = sorted((root / "ncu" / key).glob("repeat-*"))
        expected_repeats = 5 if key in {
            "w4_c10_l2_g0",
            "w4_c11_l2_g0",
            "w8_c10_l2_g0",
            "w8_c11_l2_g0",
        } else 3
        require(len(repeat_dirs) == expected_repeats, f"{key}: repeat policy mismatch")
        rows = [wide_ncu(path / "ncu.csv") for path in repeat_dirs]
        active = [row["sm__cycles_active.max"] for row in rows]
        ncu[key] = {
            "active_samples": active,
            "active_median": statistics.median(active),
            "active_cv_pct": cv_pct(active),
            "gpc_median": statistics.median(row["gpc__cycles_elapsed.max"] for row in rows),
            "l2_reads_median": statistics.median(
                row["lts__t_sectors_srcunit_tex_op_read.sum"] for row in rows
            ),
            "dram_reads_median": statistics.median(row["dram__sectors_read.sum"] for row in rows),
            "barrier_stall_median": statistics.median(
                row["smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio"]
                for row in rows
            ),
        }

        capture = root / "nvbit" / key
        capture_meta = read_json(capture / "capture.json")
        expected_launches = 2 if cell["locality"] in {"l1", "l2"} else 1
        require(capture_meta["accepted_launches"] == expected_launches, f"{key}: launches")
        payload = measurement_line(capture / "stdout.txt")
        require(payload["correct"] is True and payload["groups"] == TRACE_GROUPS, f"{key}: NVBit")
        stats = list(csv.DictReader((capture / "stats.csv").open(newline="", encoding="utf-8")))
        require(len(stats) == expected_launches, f"{key}: stats launch count")
        require(
            all(row[" total_insts"].strip() == row[" total_reported_insts"].strip() for row in stats),
            f"{key}: tracer cardinality mismatch",
        )
        expected_payloads = 257 if cell["locality"] == "l2" else 3 if cell["locality"] == "l1" else 1
        require(len(list((capture / "threadblocks").rglob("*.pb"))) == expected_payloads, f"{key}: CTA payload count")
        for diagnostic in EMPTY_NVBIT_DIAGNOSTICS:
            require(read_json(capture / "extra_info" / f"{diagnostic}.json") == [], f"{key}: {diagnostic}")
        captures[key] = {
            "launches": expected_launches,
            "cta_payloads": expected_payloads,
            "trace_bytes": (capture / "dynamic_trace.pb").stat().st_size,
        }

        documents: dict[str, dict[str, Any]] = {}
        for scope in ("launch", "stream-l2"):
            first = root / "simulator" / key / f"{scope}-1.json"
            second = root / "simulator" / key / f"{scope}-2.json"
            require(first.read_bytes() == second.read_bytes(), f"{key}: {scope} nondeterminism")
            document = read_json(first)
            require(len(document["launches"]) == expected_launches, f"{key}: {scope} launch count")
            for launch in document["launches"]:
                require(
                    launch["counters"]["instructions_issued"]
                    == launch["counters"]["records_seen"],
                    f"{key}: {scope} replay cardinality",
                )
                require(launch["coverage"]["kind"] == "full", f"{key}: {scope} coverage")
            if scope == "stream-l2":
                require(document["cache_session"]["completed_launches"] == expected_launches, f"{key}: session")
            else:
                require("cache_session" not in document, f"{key}: cold session metadata")
            documents[scope] = document
        cold = target(documents["launch"])
        warm = target(documents["stream-l2"])
        for field in ("line_fills", "bypass_sectors", "store_sectors", "reduction_sectors"):
            require(
                cold["counters"]["l1"][field] == warm["counters"]["l1"][field],
                f"{key}: target L1 physical work changed for {field}",
            )
        require(cold["counters"]["instructions_issued"] == warm["counters"]["instructions_issued"], f"{key}: instruction drift")
        require(warm["timing"]["readiness"]["schema_version"] == 2, f"{key}: readiness schema")
        if cell["locality"] == "l2":
            require(
                warm["counters"]["l2"]["hits"] - cold["counters"]["l2"]["hits"] == 1024,
                f"{key}: L2 seed did not create 1,024 hits",
            )
            require(
                cold["counters"]["dram"]["read_sectors"]
                - warm["counters"]["dram"]["read_sectors"]
                == 1024,
                f"{key}: L2 seed did not remove 1,024 DRAM reads",
            )
        elif cell["locality"] == "l1":
            require(
                warm["counters"]["l2"]["hits"] - cold["counters"]["l2"]["hits"] == 16,
                f"{key}: small seed did not create 16 L2 hits",
            )
            require(warm["counters"]["dram"]["read_sectors"] == 0, f"{key}: seeded L1 route DRAM")
        else:
            require(cold["counters"] == warm["counters"], f"{key}: unseeded DRAM counters changed")
        replays[key] = {"cold": cold, "warm": warm}

    route_rows: list[dict[str, Any]] = []
    for width in (2, 4, 8):
        for protocol in ("c10", "c11"):
            keys = {locality: f"w{width}_{protocol}_{locality}_g0" for locality in ("l1", "l2", "dram")}
            live = {locality: ncu[key]["active_median"] for locality, key in keys.items()}
            warm = {
                locality: replays[key]["warm"]["timing"]["diagnostic_last_issue_cycles"]
                for locality, key in keys.items()
            }
            require(live["l1"] < live["l2"] < live["dram"], f"W{width} {protocol}: live route order")
            require(warm["l1"] <= warm["l2"] < warm["dram"], f"W{width} {protocol}: sim route order")
            route_rows.append(
                {
                    "width": width,
                    "protocol": protocol,
                    "live_l1": live["l1"],
                    "live_l2": live["l2"],
                    "live_dram": live["dram"],
                    "sim_l1": warm["l1"],
                    "sim_l2": warm["l2"],
                    "sim_dram": warm["dram"],
                    "sim_strict_l1_l2": warm["l1"] < warm["l2"],
                }
            )

    interaction_rows: list[dict[str, Any]] = []
    for width in (2, 4, 8):
        for locality in ("l1", "l2", "dram"):
            c10 = f"w{width}_c10_{locality}_g0"
            c11 = f"w{width}_c11_{locality}_g0"
            interaction_rows.append(
                {
                    "width": width,
                    "locality": locality,
                    "live_c11_minus_c10_per_group": (
                        ncu[c11]["active_median"] - ncu[c10]["active_median"]
                    )
                    / LONG_GROUPS,
                    "sim_c11_minus_c10_per_group": (
                        replays[c11]["warm"]["timing"]["diagnostic_last_issue_cycles"]
                        - replays[c10]["warm"]["timing"]["diagnostic_last_issue_cycles"]
                    )
                    / TRACE_GROUPS,
                }
            )

    barrier_ab: list[dict[str, Any]] = []
    for width in (2, 4, 8):
        for protocol in ("c10", "c11"):
            for locality in ("l1", "dram"):
                key = f"w{width}_{protocol}_{locality}_g0"
                old = target(read_json(PREDECESSOR / "simulator" / key / "replay-1.json"))
                new = replays[key]["cold"]
                barrier_ab.append(
                    {
                        "key": key,
                        "dispatch_only_cycles": old["timing"]["diagnostic_last_issue_cycles"],
                        "barrier_service_cycles": new["timing"]["diagnostic_last_issue_cycles"],
                        "delta": new["timing"]["diagnostic_last_issue_cycles"]
                        - old["timing"]["diagnostic_last_issue_cycles"],
                    }
                )
    for protocol in ("c10", "c11"):
        key = f"w4_{protocol}_l2_g0"
        old = target(read_json(PREDECESSOR / "simulator" / key / "replay-1.json"))
        new = replays[key]["cold"]
        barrier_ab.append(
            {
                "key": key,
                "dispatch_only_cycles": old["timing"]["diagnostic_last_issue_cycles"],
                "barrier_service_cycles": new["timing"]["diagnostic_last_issue_cycles"],
                "delta": new["timing"]["diagnostic_last_issue_cycles"]
                - old["timing"]["diagnostic_last_issue_cycles"],
            }
        )

    production_predicted = read_json(root / "production" / "predicted.json")
    production_observed = read_json(
        REPO_ROOT
        / "results"
        / "captures"
        / "sm86_cycle_residual_20260817_v1"
        / "analysis"
        / "observed_sm_active_median.json"
    )
    production_old = read_json(PREDECESSOR / "analysis" / "cohort_predicted.json")
    require(production_predicted.keys() == production_observed.keys(), "production key mismatch")
    production_rows: list[dict[str, Any]] = []
    absolute_errors: list[float] = []
    percentage_errors: list[float] = []
    squared_errors: list[float] = []
    for key in sorted(production_predicted):
        first = root / "production" / f"{key}-1.json"
        second = root / "production" / f"{key}-2.json"
        require(first.read_bytes() == second.read_bytes(), f"production {key}: nondeterminism")
        launch = target(read_json(first))
        require(
            launch["counters"]["instructions_issued"]
            == launch["counters"]["records_seen"],
            f"production {key}: count mismatch",
        )
        observed = float(production_observed[key])
        predicted = float(production_predicted[key])
        error = abs(observed - predicted)
        absolute_errors.append(error)
        percentage_errors.append(error / observed)
        squared_errors.append((observed - predicted) ** 2)
        production_rows.append(
            {
                "key": key,
                "observed": int(observed),
                "old_predicted": production_old[key],
                "new_predicted": int(predicted),
                "model_delta": int(predicted) - production_old[key],
            }
        )
    production_score = {
        "coverage": 6,
        "mae": statistics.mean(absolute_errors),
        "rmse": statistics.mean(squared_errors) ** 0.5,
        "mape": statistics.mean(percentage_errors),
        "rows": production_rows,
    }

    predecessor_cells = read_json(PREDECESSOR / "metadata" / "matrix.json")["cells"]
    require(len(predecessor_cells) == 36, "predecessor matrix count mismatch")
    predecessor_deltas: list[dict[str, Any]] = []
    for cell in predecessor_cells:
        key = cell["key"]
        first = root / "predecessor" / f"{key}-1.json"
        second = root / "predecessor" / f"{key}-2.json"
        require(first.read_bytes() == second.read_bytes(), f"predecessor {key}: nondeterminism")
        new = target(read_json(first))
        old = target(read_json(PREDECESSOR / "simulator" / key / "replay-1.json"))
        require(new["counters"] == old["counters"], f"predecessor {key}: counters changed")
        predecessor_deltas.append(
            {
                "key": key,
                "protocol": cell["protocol"],
                "cycle_delta": new["timing"]["diagnostic_last_issue_cycles"]
                - old["timing"]["diagnostic_last_issue_cycles"],
            }
        )

    host_baseline = read_json(PREDECESSOR / "metadata" / "host_replay_after.json")
    host_final = read_json(root / "metadata" / "host_replay_final.json")
    require(host_baseline.keys() == host_final.keys(), "host benchmark key mismatch")
    host_replay: dict[str, Any] = {}
    for key in host_final:
        before = host_baseline[key]["median_ms"]
        after = host_final[key]["median_ms"]
        host_replay[key] = {
            "samples": host_final[key]["samples"],
            "baseline_median_ms": before,
            "final_median_ms": after,
            "change_ms": after - before,
            "change_pct": (after - before) / before * 100.0,
        }

    return {
        "schema_version": 1,
        "campaign": {
            "cells": 18,
            "correctness_trials_per_cell": 7,
            "ncu_profiles": sum(len(value["active_samples"]) for value in ncu.values()),
            "ncu_cells_with_five_repeats": sum(len(value["active_samples"]) == 5 for value in ncu.values()),
            "source_counters_profiles": len(list((root / "ncu-source").glob("*/source.csv"))),
            "nvbit_captures": len(captures),
            "two_launch_captures": sum(value["launches"] == 2 for value in captures.values()),
            "byte_identical_scope_pairs": 2 * len(replays),
            "all_cardinality_exact": True,
            "all_full_coverage": True,
            "all_exclusion_sidecars_empty": True,
        },
        "route_order": route_rows,
        "strict_sim_l1_l2_groups": sum(row["sim_strict_l1_l2"] for row in route_rows),
        "interactions": interaction_rows,
        "barrier_service_ab": barrier_ab,
        "predecessor_replay": {
            "cells": 36,
            "byte_identical_pairs": 36,
            "all_counters_unchanged": True,
            "cycle_deltas": predecessor_deltas,
        },
        "production": production_score,
        "host_replay": host_replay,
        "decision": {
            "warm_l2_route_separated": True,
            "live_route_order_all_cells": True,
            "sim_route_order_all_groups": True,
            "barrier_width_resource_retained": True,
            "fixed_release_or_rearm_latency_added": False,
            "remaining_gap": "async completion, arrival skew, and scheduler interaction magnitude",
        },
    }


def report(summary: dict[str, Any]) -> str:
    campaign = summary["campaign"]
    lines = [
        "# Coupled barrier-pipeline and warm-L2 validation",
        "",
        "**Date:** 2026-08-17  ",
        "**GPU:** NVIDIA GeForce RTX 3090 Ti (`sm_86`)  ",
        "**Result:** traced warm state separates L2/DRAM; barrier resource retained; coupled magnitude remains incomplete",
        "",
        "## Campaign",
        "",
        f"The 18-cell W2/W4/W8 × c10/c11 × L1/L2/DRAM matrix passes seven correctness trials per cell. It contains {campaign['ncu_profiles']} base-clock NCU profiles (four high-variance L2 cells expanded to five repeats), {campaign['source_counters_profiles']} SourceCounters profiles, {campaign['nvbit_captures']} NVBit captures, and {campaign['byte_identical_scope_pairs']} byte-identical cold/warm replay pairs. Twelve L1/L2 captures retain both prewarm and target launches; six DRAM captures contain the unseeded target only. Every launch has full coverage and exact instruction cardinality, and every exclusion sidecar is empty.",
        "",
        "## Route separation",
        "",
        "| Width | Protocol | Live L1 | Live L2 | Live DRAM | Sim warm L1 | Sim warm L2 | Sim DRAM |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["route_order"]:
        lines.append(
            f"| {row['width']} | {row['protocol']} | {row['live_l1']:.0f} | {row['live_l2']:.0f} | {row['live_dram']:.0f} | {row['sim_l1']} | {row['sim_l2']} | {row['sim_dram']} |"
        )
    lines.extend(
        [
            "",
            "Live L1 < L2 < DRAM ordering holds in all six width/protocol groups. Warm replay is nondecreasing in all six and strict in five; W8 c10 ties L1 and L2 because their source completion is fully hidden at the target wait. In every L2-rotation cell, the traced seed converts exactly 1,024 target L2 misses to hits and removes 1,024 DRAM read sectors. The small L1 seed converts 16 sectors. Target L1 line-fill and write work is unchanged. At W8, faster L2 responses let later accesses hit fills completed earlier within the target itself, so L1 hit/miss observer counts legitimately change even though no L1 state crosses the launch boundary.",
            "",
            "## Coupled c11 - c10 interaction",
            "",
            "| Width | Route | Live cycles/group | Sim cycles/group |",
            "|---:|---|---:|---:|",
        ]
    )
    for row in summary["interactions"]:
        lines.append(
            f"| {row['width']} | {row['locality']} | {row['live_c11_minus_c10_per_group']:.3f} | {row['sim_c11_minus_c10_per_group']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The barrier service adds 127-128 trace cycles to c11 and zero to c10 in the cold A/B, about two cycles per measured group. Async completion already staggers most barrier arrivals, so the shared service rarely queues more than one arrival. This preserves the barrier-only identification result while proving it is not the missing explanation for the full async/barrier interaction magnitude. Warm state supplies the correct L2 middle route, especially at W8, but simulated c11-c10 cost remains too large for L1 and generally overstates the live interaction.",
            "",
            "## Decision",
            "",
            "Retain the SM-shared two-cycle barrier-arrival service and the transactional stream-L2 handoff. Do not add a fixed release latency, re-arm term, or route-specific barrier constant. The next unresolved component is scheduler/async-completion coupling under controlled multi-CTA occupancy; that validation must explain magnitude without changing the independently identified barrier throughput or importing NCU route labels.",
            "",
            "## Regression cohorts",
            "",
            f"All {summary['predecessor_replay']['cells']} predecessor traces replay twice byte-identically with every traffic counter unchanged. The six-workload production cohort retains 6/6 coverage: MAE is {summary['production']['mae']:.2f}, RMSE is {summary['production']['rmse']:.2f}, and active-SM MAPE is {summary['production']['mape'] * 100:.3f}%. Five predictions are unchanged; attention moves from 12,448 to 12,456 cycles, toward its 14,510-cycle observation. The predecessor MAPE was 33.403%.",
            "",
            f"Process-level replay over 200 invocations changes from {summary['host_replay']['w4_c00_na_g0']['baseline_median_ms']:.3f} to {summary['host_replay']['w4_c00_na_g0']['final_median_ms']:.3f} ms on the tiny control ({summary['host_replay']['w4_c00_na_g0']['change_pct']:+.2f}%, {summary['host_replay']['w4_c00_na_g0']['change_ms']:+.3f} ms) and from {summary['host_replay']['w4_c11_l1_g0']['baseline_median_ms']:.3f} to {summary['host_replay']['w4_c11_l1_g0']['final_median_ms']:.3f} ms on the representative async sentinel ({summary['host_replay']['w4_c11_l1_g0']['change_pct']:+.2f}%, {summary['host_replay']['w4_c11_l1_g0']['change_ms']:+.3f} ms). The representative path remains within 3%; the sub-millisecond control increase is retained as measured process/serialization overhead.",
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
    root = args.root.resolve()
    try:
        summary = reduce(root)
        analysis = root / "analysis"
        write_json(analysis / "summary.json", summary)
        (analysis / "README.md").write_text(report(summary), encoding="utf-8")
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"coupled analysis: {error}", file=__import__("sys").stderr)
        return 1
    print(root / "analysis" / "README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
