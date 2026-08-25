#!/usr/bin/env python3
"""Reduce the sm_86 many-region atomic/LTS scheduling campaign."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import campaign


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def median(values: Iterable[float]) -> float:
    material = list(values)
    if not material:
        raise RuntimeError("empty median")
    return float(statistics.median(material))


def ape(predicted: float, observed: float) -> float:
    return 100 * abs(predicted - observed) / observed


def ncu_row(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if (row.get("ID") or "").strip()]
    if len(rows) != 1:
        raise RuntimeError(f"{path}: expected one NCU row, got {len(rows)}")
    result = {}
    for key, raw in rows[0].items():
        if key is None or raw is None or not raw.strip():
            continue
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        if math.isfinite(value):
            result[key] = value
    return result


class UnionFind:
    def __init__(self, values: Iterable[int]):
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def hash_value(line: int, kind: str) -> int:
    if kind == "line_mod":
        return line % 48
    if kind == "sector_mod":
        return (line * 4) % 48
    if kind == "xor_folded":
        return (line ^ (line >> 5) ^ (line >> 11)) % 48
    if kind == "knuth_mult":
        return ((line * 2_654_435_761) & 0xFFFF_FFFF) % 48
    if kind == "shift_mod":
        return (line >> 1) % 48
    if kind == "bit_reverse":
        return int(f"{line & 0xFFFF:016b}"[::-1], 2) % 48
    if kind == "local_bit_drop":
        region = line >> 1
        return (((region >> 2) << 1) | (region & 1)) % 48
    raise AssertionError(kind)


def reduce(root: Path) -> dict[str, object]:
    manifest = read_json(root / "metadata/matrix.json")
    require(manifest.get("schema_version") == 1, "matrix schema")
    require(len(manifest["triplets"]) == 220, "triplet matrix")
    require(len(manifest["pairs"]) == 183, "pair matrix")

    pair_rows = []
    local_uf = UnionFind(campaign.REGION_UNIVERSE)
    observed_uf = UnionFind(range(campaign.ANCHOR_MAX_REGION + 1))
    base_regions = set()
    total_sectors = None
    for cell in campaign.pair_cells():
        measurement = read_json(
            root / "ncu-pairs" / cell.key / "measurement.json"
        )["measurement"]
        row = ncu_row(root / "ncu-pairs" / cell.key / "ncu.csv")
        observed_total = int(row["lts__t_sectors_op_red.sum"])
        observed_max = int(row["lts__t_sectors_op_red.max"])
        if total_sectors is None:
            total_sectors = observed_total
        require(observed_total == total_sectors == 2_752_512, f"{cell.key}: sectors")
        require(
            observed_max in {observed_total // 2, observed_total},
            f"{cell.key}: non-binary LTS occupancy {observed_max}",
        )
        same_slice = observed_max == observed_total
        base_regions.add(int(measurement["data_device_region_256"]))
        left, right = cell.regions
        if left < 12 and right < 12 and same_slice:
            local_uf.union(left, right)
        if same_slice:
            observed_uf.union(left, right)
        pair_rows.append(
            {
                "key": cell.key,
                "regions": list(cell.regions),
                "same_lts_slice": same_slice,
                "lts_max_sectors": observed_max,
                "lts_total_sectors": observed_total,
            }
        )
    require(len(base_regions) == 1, f"pair allocation base drift: {sorted(base_regions)}")
    base_region = next(iter(base_regions))

    classes_by_root: dict[int, list[int]] = {}
    for region in campaign.REGION_UNIVERSE:
        classes_by_root.setdefault(local_uf.find(region), []).append(region)
    local_classes = sorted(tuple(values) for values in classes_by_root.values())
    for left, right in itertools.combinations(campaign.REGION_UNIVERSE, 2):
        observed = next(
            row["same_lts_slice"]
            for row in pair_rows
            if row["regions"] == [left, right]
        )
        require(
            observed == (local_uf.find(left) == local_uf.find(right)),
            f"local LTS relation is not transitive at {left},{right}",
        )

    anchor_collisions = [
        row["regions"][1]
        for row in pair_rows
        if row["regions"][0] == 0 and row["same_lts_slice"]
    ]
    require(anchor_collisions[:1] == [2], "local anchor collision changed")

    candidate_rows = []
    for kind in (
        "line_mod",
        "sector_mod",
        "xor_folded",
        "knuth_mult",
        "shift_mod",
        "bit_reverse",
        "local_bit_drop",
    ):
        confusion = {
            "local": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
            "transfer": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
        }
        for row in pair_rows:
            left, right = row["regions"]
            predicted = hash_value(2 * (base_region + left), kind) == hash_value(
                2 * (base_region + right), kind
            )
            cohort = "local" if left < 12 and right < 12 else "transfer"
            observed = row["same_lts_slice"]
            if predicted and observed:
                confusion[cohort]["tp"] += 1
            elif predicted:
                confusion[cohort]["fp"] += 1
            elif observed:
                confusion[cohort]["fn"] += 1
            else:
                confusion[cohort]["tn"] += 1

        def metrics(cohort: str) -> dict[str, float | int]:
            counts = confusion[cohort]
            positives = counts["tp"] + counts["fn"]
            negatives = counts["tn"] + counts["fp"]
            total = positives + negatives
            recall = counts["tp"] / positives
            specificity = counts["tn"] / negatives
            precision = (
                counts["tp"] / (counts["tp"] + counts["fp"])
                if counts["tp"] + counts["fp"]
                else 0.0
            )
            return {
                **counts,
                "accuracy": (counts["tp"] + counts["tn"]) / total,
                "collision_recall": recall,
                "collision_precision": precision,
                "balanced_accuracy": (recall + specificity) / 2,
            }

        local_metrics = metrics("local")
        transfer_metrics = metrics("transfer")
        candidate_rows.append(
            {
                "candidate": kind,
                "local": local_metrics,
                "transfer": transfer_metrics,
            }
        )
    candidate_rows.sort(
        key=lambda row: (
            -row["local"]["balanced_accuracy"],
            -row["transfer"]["balanced_accuracy"],
        )
    )

    triplet_rows = []
    by_occupancy: dict[int, list[float]] = {}
    correctness_executions = 0
    direct_base_regions = set()
    for cell in campaign.triplet_cells():
        measurement = read_json(root / "correctness" / cell.key / "result.json")[
            "measurement"
        ]
        require(measurement["correct"] is True, f"{cell.key}: correctness")
        require(
            measurement["fixed_data_bytes"] == campaign.FIXED_DATA_BYTES,
            f"{cell.key}: allocation",
        )
        direct_base_regions.add(int(measurement["data_device_region_256"]))
        samples = measurement["samples_cycles"]
        require(len(samples) == 7, f"{cell.key}: expected seven trials")
        correctness_executions += len(samples)
        cycles = median(max(trial) for trial in samples)
        occupancy = max(Counter(local_uf.find(region) for region in cell.regions).values())
        by_occupancy.setdefault(occupancy, []).append(cycles)
        triplet_rows.append(
            {
                "key": cell.key,
                "regions": list(cell.regions),
                "maximum_lts_region_occupancy": occupancy,
                "clock_cycles": cycles,
                "seven_trial_range_cycles": max(max(trial) for trial in samples)
                - min(max(trial) for trial in samples),
            }
        )
    require(
        len(direct_base_regions) == 1,
        f"direct allocation base drift: {sorted(direct_base_regions)}",
    )
    direct_base_region = next(iter(direct_base_regions))
    require(
        direct_base_region % 4 == base_region % 4,
        "direct and NCU allocations differ in the locally characterized phase",
    )
    require(set(by_occupancy) == {1, 2}, "unexpected local occupancy classes")
    require(
        max(by_occupancy[1]) < min(by_occupancy[2]),
        "LTS occupancy timing clusters overlap",
    )
    occupancy_medians = {
        occupancy: median(values) for occupancy, values in by_occupancy.items()
    }
    oracle_errors = [
        ape(occupancy_medians[row["maximum_lts_region_occupancy"]], row["clock_cycles"])
        for row in triplet_rows
    ]

    full_ncu_profiles = 0
    full_rows = []
    full_gpc_cvs = []
    full_gpc_mads = []
    for cell in campaign.trace_cells():
        observations = sorted((root / "ncu" / cell.key).glob("repeat-*/ncu.csv"))
        require(len(observations) in {3, 7}, f"{cell.key}: full NCU repeats")
        rows = [ncu_row(path) for path in observations]
        full_ncu_profiles += len(rows)
        gpc_values = [row["gpc__cycles_elapsed.max"] for row in rows]
        full_gpc_cvs.append(
            statistics.stdev(gpc_values) / statistics.mean(gpc_values)
        )
        gpc_median = statistics.median(gpc_values)
        full_gpc_mads.append(
            statistics.median(abs(value - gpc_median) for value in gpc_values)
            / gpc_median
        )
        lts_total = median(row["lts__t_sectors_op_red.sum"] for row in rows)
        lts_max = median(row["lts__t_sectors_op_red.max"] for row in rows)
        expected_occupancy = max(
            Counter(observed_uf.find(region) for region in cell.regions).values()
        )
        require(lts_total == 2_752_512, f"{cell.key}: full NCU sectors")
        require(
            lts_max == lts_total * expected_occupancy / 3,
            f"{cell.key}: full NCU LTS occupancy",
        )
        full_rows.append(
            {
                "key": cell.key,
                "regions": list(cell.regions),
                "gpc_cycles": median(row["gpc__cycles_elapsed.max"] for row in rows),
                "sm_active_cycles": median(row["sm__cycles_active.max"] for row in rows),
                "lts_max_sectors": lts_max,
                "lts_total_sectors": lts_total,
            }
        )

    trace_records = 0
    replay_records = 0
    replay_pairs = 0
    simulator_last_issue = set()
    simulator_clock_interval = set()
    for cell in campaign.trace_cells():
        trace = root / "nvbit" / cell.key
        require((trace / "dynamic_trace.pb").stat().st_size > 0, f"{cell.key}: trace")
        with (trace / "stats.csv").open(encoding="utf-8", newline="") as handle:
            stats = [
                {key.strip(): value.strip() for key, value in row.items()}
                for row in csv.DictReader(handle)
            ]
        require(len(stats) == 1, f"{cell.key}: trace launches")
        records = int(stats[0]["total_insts"])
        require(records == int(stats[0]["total_reported_insts"]), f"{cell.key}: records")
        trace_records += records
        destination = root / "simulator" / cell.key
        first = destination / "replay-1.json"
        second = destination / "replay-2.json"
        require(first.read_bytes() == second.read_bytes(), f"{cell.key}: replay pair")
        require(
            all(
                (destination / f"replay-{repeat}.stderr.txt").stat().st_size == 0
                for repeat in (1, 2)
            ),
            f"{cell.key}: replay stderr",
        )
        document = read_json(first)
        require(len(document["launches"]) == 1, f"{cell.key}: replay launch")
        launch = document["launches"][0]
        require(
            launch["counters"]["instructions_issued"]
            == launch["counters"]["records_seen"],
            f"{cell.key}: replay records",
        )
        replay_records += int(launch["counters"]["records_seen"])
        simulator_last_issue.add(int(launch["timing"]["diagnostic_last_issue_cycles"]))
        simulator_clock_interval.add(
            int(
                launch["timing"]["readiness"]["clock_issue"][
                    "maximum_first_to_last_issue_cycles"
                ]
            )
        )
        replay_pairs += 1

    for cell in campaign.trace_cells():
        measurement = read_json(
            root / "trace-correctness" / cell.key / "result.json"
        )["measurement"]
        require(len(measurement["samples_cycles"]) == 7, f"{cell.key}: trace trials")

    static_sidecar = (
        root / "sass/atomic_multism_service_trace_red32.control_bits.json"
    )
    require(static_sidecar.is_file(), "static control sidecar")
    source_profiles = sum(
        (root / "ncu-source" / campaign.TripletCell(tuple(regions)).key / "source.csv").is_file()
        for regions in campaign.SOURCE_TRIPLETS
    )
    require(source_profiles == 4, "source profiles")
    source_summary = {}
    for regions in campaign.SOURCE_TRIPLETS:
        cell = campaign.TripletCell(tuple(regions))
        path = root / "ncu-source" / cell.key / "source.csv"
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        rows = list(csv.DictReader(lines[1:]))
        branch = next(
            (
                row
                for row in rows
                if "BRA 0x" in (row.get("Source") or "")
            ),
            None,
        )
        require(branch is not None, f"{cell.key}: loop branch source row")
        source_summary[cell.key] = {
            "branch_long_scoreboard_not_issued_samples": int(
                branch["stall_long_sb (Not Issued)"].replace(",", "")
            ),
            "branch_all_samples": int(branch["# Samples"].replace(",", "")),
        }

    return {
        "schema_version": 1,
        "campaign": {
            "triplet_cells": len(triplet_rows),
            "triplet_correctness_executions": correctness_executions,
            "pair_ncu_profiles": len(pair_rows),
            "full_ncu_profiles": full_ncu_profiles,
            "maximum_full_ncu_gpc_cv_pct": 100 * max(full_gpc_cvs),
            "maximum_full_ncu_gpc_mad_pct": 100 * max(full_gpc_mads),
            "source_counters_profiles": source_profiles,
            "trace_cells": len(campaign.trace_cells()),
            "trace_correctness_executions": len(campaign.trace_cells()) * 7,
            "trace_records": trace_records,
            "replay_warp_instruction_records": replay_records,
            "byte_identical_replay_pairs": replay_pairs,
            "maximum_triplet_trial_range_cycles": max(
                row["seven_trial_range_cycles"] for row in triplet_rows
            ),
        },
        "lts_slice_relation": {
            "ncu_data_device_region_256": base_region,
            "direct_data_device_region_256": direct_base_region,
            "local_equivalence_classes": [list(group) for group in local_classes],
            "anchor_zero_collision_offsets": anchor_collisions,
            "pair_rows": pair_rows,
        },
        "triplet_timing": {
            "occupancy_one_count": len(by_occupancy[1]),
            "occupancy_two_count": len(by_occupancy[2]),
            "occupancy_one_median_cycles": occupancy_medians[1],
            "occupancy_two_median_cycles": occupancy_medians[2],
            "occupancy_one_max_cycles": max(by_occupancy[1]),
            "occupancy_two_min_cycles": min(by_occupancy[2]),
            "oracle_mape_pct": statistics.mean(oracle_errors),
            "oracle_max_ape_pct": max(oracle_errors),
            "rows": triplet_rows,
        },
        "hash_candidates": candidate_rows,
        "full_ncu": full_rows,
        "source_counters": source_summary,
        "simulator": {
            "unique_last_issue_cycles": sorted(simulator_last_issue),
            "unique_clock_interval_cycles": sorted(simulator_clock_interval),
        },
        "decision": {
            "status": "lts_slice_occupancy_identified_hash_unresolved_timing_withheld",
            "promoted": [
                "fixed-allocation arbitrary-region atomic probe",
                "maximum per-LTS RED sector occupancy as the causal triplet discriminator",
            ],
            "rejected": [
                "region count alone as a timing model",
                "the six retained ordinary L2 hashes as the GA102 LTS mapping",
                "the locally exact low-bit-drop rule as a transferable hash",
                "activation of a cross-SM atomic timing queue",
            ],
            "timing_queue": False,
            "reason": (
                "LTS occupancy cleanly separates timing classes, but no tested "
                "address hash predicts the exact local relation and anchor transfer."
            ),
        },
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render(result: Mapping[str, object]) -> str:
    campaign_result = result["campaign"]
    relation = result["lts_slice_relation"]
    timing = result["triplet_timing"]
    best = result["hash_candidates"][0]
    source = result["source_counters"]
    simulator = result["simulator"]
    decision = result["decision"]
    lines = [
        "# sm_86 many-region atomic/LTS scheduling",
        "",
        "## Decision",
        "",
        "Maximum per-LTS RED sector occupancy exactly separates the many-region",
        "triplet timing classes, but the address-to-LTS hash remains unresolved.",
        "No atomic timing queue is activated.",
        "",
        "## Campaign integrity",
        "",
        f"- {campaign_result['triplet_cells']} triplets and {campaign_result['triplet_correctness_executions']} correct long executions",
        f"- {campaign_result['pair_ncu_profiles']} pair LTS profiles and {campaign_result['full_ncu_profiles']} full NCU profiles",
        f"- maximum full-profile GPC CV {campaign_result['maximum_full_ncu_gpc_cv_pct']:.3f}% and robust MAD {campaign_result['maximum_full_ncu_gpc_mad_pct']:.3f}%",
        f"- {campaign_result['source_counters_profiles']} SourceCounters profiles and {campaign_result['trace_cells']} NVBit captures",
        f"- {campaign_result['trace_records']:,} trace records and {campaign_result['byte_identical_replay_pairs']} byte-identical replay pairs",
        "",
        "## Exact local slice relation",
        "",
        "Within the fixed 12-region window, pair NCU maxima produce these exact",
        "LTS-slice equivalence classes:",
        "",
        "```text",
        " ".join(str(tuple(group)) for group in relation["local_equivalence_classes"]),
        "```",
        "",
        "Every triplet whose regions occupy three distinct slices is in the fast",
        "cluster; every triplet with two regions on one slice is in the slow cluster.",
        "",
        "| maximum regions on one LTS | cells | median CLOCK | range boundary |",
        "|---:|---:|---:|---:|",
        f"| 1 | {timing['occupancy_one_count']} | {timing['occupancy_one_median_cycles']:.0f} | max {timing['occupancy_one_max_cycles']:.0f} |",
        f"| 2 | {timing['occupancy_two_count']} | {timing['occupancy_two_median_cycles']:.0f} | min {timing['occupancy_two_min_cycles']:.0f} |",
        "",
        f"An oracle using only occupancy-class medians has {timing['oracle_mape_pct']:.2f}% MAPE and {timing['oracle_max_ape_pct']:.2f}% maximum error.",
        "",
        "## Hash transfer",
        "",
        f"The best local mapping is `{best['candidate']}` with {100*best['local']['balanced_accuracy']:.1f}% local balanced accuracy, but only {100*best['transfer']['balanced_accuracy']:.1f}% anchor-transfer balanced accuracy and {100*best['transfer']['collision_recall']:.1f}% transfer collision recall.",
        "The low-window bit relation is descriptive only and fails wider offsets;",
        "ordinary cache hashes also fail. Hardware LTS counters are evaluation",
        "labels, not legal simulator inputs, so the timing service remains withheld.",
        f"Current replay is intentionally address-insensitive here: all captured sentinels report last issue {simulator['unique_last_issue_cycles']} and CLOCK interval {simulator['unique_clock_interval_cycles']} cycles.",
        "",
        "SourceCounters keep the causal wait on the loop branch: long-scoreboard",
        "not-issued samples rise from "
        f"{source['t_000_001_004']['branch_long_scoreboard_not_issued_samples']:,} "
        "in the slice-distinct sentinel to "
        f"{source['t_000_002_004']['branch_long_scoreboard_not_issued_samples']:,} "
        "in the local slice-colliding sentinel.",
        "",
        "## Retained result",
        "",
        *[f"- {item}" for item in decision["promoted"]],
        "",
        "Rejected:",
        "",
        *[f"- {item}" for item in decision["rejected"]],
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=campaign.DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve()
        result = reduce(root)
        write_json(root / "analysis/summary.json", result)
        report = root / "analysis/README.md"
        report.write_text(render(result), encoding="utf-8")
        print(report)
    except (OSError, RuntimeError, json.JSONDecodeError, csv.Error) as error:
        print(f"many-region analysis: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
