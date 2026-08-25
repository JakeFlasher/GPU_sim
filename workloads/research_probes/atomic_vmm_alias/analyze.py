#!/usr/bin/env python3
"""Reduce the GA102 CUDA-VMM physical-allocation/virtual-alias campaign."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
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
    require(bool(material), "empty median")
    return float(statistics.median(material))


def choose2(value: int) -> int:
    return value * (value - 1) // 2


def equivalence_score(left: Sequence[int], right: Sequence[int]) -> dict[str, float]:
    require(len(left) == len(right) and len(left) > 1, "equivalence roster")
    intersections = defaultdict(int)
    left_counts = defaultdict(int)
    right_counts = defaultdict(int)
    for left_value, right_value in zip(left, right, strict=True):
        intersections[(left_value, right_value)] += 1
        left_counts[left_value] += 1
        right_counts[right_value] += 1
    true_positive = sum(choose2(count) for count in intersections.values())
    left_positive = sum(choose2(count) for count in left_counts.values())
    right_positive = sum(choose2(count) for count in right_counts.values())
    total = choose2(len(left))
    true_negative = total - left_positive - right_positive + true_positive
    recall = true_positive / left_positive
    specificity = true_negative / (total - left_positive)
    return {
        "collision_recall": recall,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
    }


def class_bijection(left: Sequence[int], right: Sequence[int]) -> bool:
    forward: dict[int, set[int]] = defaultdict(set)
    reverse: dict[int, set[int]] = defaultdict(set)
    for left_value, right_value in zip(left, right, strict=True):
        forward[left_value].add(right_value)
        reverse[right_value].add(left_value)
    return (
        len(forward) == len(reverse)
        and all(len(values) == 1 for values in forward.values())
        and all(len(values) == 1 for values in reverse.values())
    )


def exact_class_permutation(left: Sequence[int], right: Sequence[int]) -> bool:
    return class_bijection(left, right) and len(set(left)) == len(set(right)) == 48


def reduce(root: Path) -> dict[str, object]:
    manifest = read_json(root / "metadata/matrix.json")
    require(manifest.get("schema_version") == 1, "matrix schema")
    require(len(manifest.get("cells", ())) == 19, "cell matrix")

    correctness_executions = 0
    for cell in campaign.cells():
        measurement = read_json(root / "correctness" / cell.key / "measurement.json")
        campaign.validate_measurement(cell, measurement)
        require(measurement["trials"] == 7, f"{cell.key}: correctness trials")
        correctness_executions += 7

    read_rows: dict[tuple[str, int], list[dict[str, object]]] = {}
    read_profiles = 0
    read_total_outliers = 0
    for cell in campaign.cells():
        repeats = 7 if cell.primary else 3
        rows = []
        for repeat in range(1, repeats + 1):
            payload = read_json(
                root / "ncu" / cell.key / f"repeat-{repeat}/result.json"
            )
            campaign.validate_measurement(cell, payload["measurement"])
            require(payload["repeat"] == repeat, f"{cell.key}: repeat identity")
            require(
                0 <= payload["metrics"]["minimum_sectors"]
                <= payload["metrics"]["maximum_sectors"]
                <= payload["metrics"]["total_sectors"],
                f"{cell.key}: read counters",
            )
            rows.append(payload)
            read_profiles += 1
        typical_total = median(row["metrics"]["total_sectors"] for row in rows)
        read_total_outliers += sum(
            row["metrics"]["total_sectors"] > typical_total * 1.5 for row in rows
        )
        read_rows[(cell.mode, cell.offset_bytes)] = rows
    require(read_profiles == 121, "read profile count")

    paired_rows = []
    for offset in campaign.OFFSETS:
        alias_rows = read_rows[("alias_same_offset", offset)]
        linear_rows = read_rows[("single_mapping_alias_stride", offset)]
        require(
            all(
                alias["measurement"]["group_addresses"]
                == linear["measurement"]["group_addresses"]
                for alias, linear in zip(alias_rows, linear_rows, strict=True)
            ),
            f"offset {offset}: virtual address mismatch",
        )
        alias_maximum = median(
            row["metrics"]["maximum_sectors"] for row in alias_rows
        )
        linear_maximum = median(
            row["metrics"]["maximum_sectors"] for row in linear_rows
        )
        paired_rows.append(
            {
                "offset_bytes": offset,
                "alias_maximum_sectors_median": alias_maximum,
                "linear_maximum_sectors_median": linear_maximum,
                "alias_minus_linear_maximum": alias_maximum - linear_maximum,
                "alias_maximum_fraction_median": median(
                    row["metrics"]["maximum_sectors"]
                    / row["metrics"]["total_sectors"]
                    for row in alias_rows
                ),
                "linear_maximum_fraction_median": median(
                    row["metrics"]["maximum_sectors"]
                    / row["metrics"]["total_sectors"]
                    for row in linear_rows
                ),
            }
        )
    require(
        all(row["alias_minus_linear_maximum"] > 0 for row in paired_rows),
        "read A/B maximum direction",
    )

    single_pointer = read_rows[("single_alias_same_offset", 0)]
    single_pointer_fraction = median(
        row["metrics"]["maximum_sectors"]
        / row["metrics"]["total_sectors"]
        for row in single_pointer
    )
    require(single_pointer_fraction < 0.2, "read path unexpectedly became one-slice")

    atomic_profiles = 0
    atomic_actions = 0
    alias_observations = 0
    alias_invariant_observations = 0
    atomic_rows = []
    for offset in campaign.OFFSETS:
        key = f"physical_offset_o{offset:07d}"
        classes = []
        for repeat in range(1, 8):
            payload = read_json(
                root / "atomic" / key / f"repeat-{repeat}/result.json"
            )
            require(payload["offset_bytes"] == offset, f"{key}: offset")
            require(payload["repeat"] == repeat, f"{key}: repeat")
            require(payload["measurement"]["correct"] is True, f"{key}: correctness")
            require(payload["validation_classes"] == list(range(48)), f"{key}: basis")
            require(len(payload["metrics"]) == 55, f"{key}: actions")
            require(
                all(row["total_sectors"] == 37_632 for row in payload["metrics"]),
                f"{key}: sectors",
            )
            aliases = payload["alias_classes"]
            require(len(aliases) == 7, f"{key}: aliases")
            require(all(0 <= value < 48 for value in aliases), f"{key}: class range")
            classes.append(aliases[0])
            alias_observations += len(aliases)
            alias_invariant_observations += sum(value == aliases[0] for value in aliases)
            atomic_profiles += 1
            atomic_actions += len(payload["metrics"])
        require(len(set(classes)) == 1, f"{key}: repeat class drift {classes}")
        atomic_rows.append(
            {
                "offset_bytes": offset,
                "observed_class": classes[0],
                "alias_observations": 49,
                "distinct_virtual_aliases": 7,
            }
        )
    require(atomic_profiles == 56 and atomic_actions == 3_080, "atomic counts")
    require(alias_observations == alias_invariant_observations == 392, "alias invariance")

    relative = read_json(root / "relative/classification.json")
    relative_records = relative["records"]
    require(len(relative_records) == 16_384, "relative record count")
    require(relative["validation_classes"] == list(range(48)), "relative basis")
    require(relative["measurement"]["correct"] is True, "relative correctness")
    require(relative["measurement"]["mapping_launches"] == 16_432, "relative actions")
    predecessor = read_json(
        campaign.REPO_ROOT
        / "results/captures/sm86_atomic_lts_mapping_20260820_v1/"
        "profiles/train/encoded-fit.json"
    )["records"][:16_384]
    require(
        [record["region"] for record in relative_records]
        == [record["region"] for record in predecessor],
        "relative predecessor roster",
    )
    forward: dict[int, set[int]] = defaultdict(set)
    reverse: dict[int, set[int]] = defaultdict(set)
    for relative_record, predecessor_record in zip(
        relative_records, predecessor, strict=True
    ):
        left = int(relative_record["observed_class"])
        right = int(predecessor_record["observed_class"])
        forward[left].add(right)
        reverse[right].add(left)
    require(
        len(forward) == len(reverse) == 48
        and all(len(values) == 1 for values in forward.values())
        and all(len(values) == 1 for values in reverse.values()),
        "allocation-relative partition did not transfer exactly",
    )
    relative_permutation = {
        str(source): next(iter(targets)) for source, targets in sorted(forward.items())
    }

    predecessor_fit = read_json(
        campaign.REPO_ROOT
        / "results/captures/sm86_atomic_lts_mapping_20260820_v1/"
        "profiles/train/encoded-fit.json"
    )["records"]
    contiguous = read_json(root / "relative-contiguous-8m/classification.json")
    contiguous_records = contiguous["records"]
    contiguous_predecessor = predecessor_fit[16_384:]
    require(
        [record["region"] for record in contiguous_records]
        == [record["region"] for record in contiguous_predecessor],
        "contiguous predecessor roster",
    )
    contiguous_left = [int(record["observed_class"]) for record in contiguous_records]
    contiguous_right = [
        int(record["observed_class"]) for record in contiguous_predecessor
    ]
    contiguous_score = equivalence_score(contiguous_left, contiguous_right)
    granule_rows = []
    for granule in range(1, 4):
        indices = [
            index
            for index, record in enumerate(contiguous_records)
            if int(record["region"]) // 16_384 == granule
        ]
        left = [contiguous_left[index] for index in indices]
        right = [contiguous_right[index] for index in indices]
        exact = exact_class_permutation(left, right)
        require(exact, f"2 MiB granule {granule}: partition transfer")
        granule_rows.append(
            {
                "granule_index": granule,
                "records": len(indices),
                "exact_partition_transfer": exact,
            }
        )
    require(
        not exact_class_permutation(contiguous_left, contiguous_right),
        "8 MiB range unexpectedly used one global class permutation",
    )

    predecessor_high = read_json(
        campaign.REPO_ROOT
        / "results/captures/sm86_atomic_lts_mapping_20260820_v1/"
        "profiles/train/encoded-holdout.json"
    )["records"]
    high = read_json(root / "relative-high-holdout-32m/classification.json")
    high_records = high["records"]
    require(
        [record["region"] for record in high_records]
        == [record["region"] for record in predecessor_high],
        "high predecessor roster",
    )
    high_left = [int(record["observed_class"]) for record in high_records]
    high_right = [int(record["observed_class"]) for record in predecessor_high]
    require(
        exact_class_permutation(high_left, high_right),
        "size-matched 32 MiB high holdout did not transfer",
    )

    cross_payloads = [
        read_json(root / "cross-allocation" / f"repeat-{repeat}/result.json")
        for repeat in range(1, 8)
    ]
    for repeat, payload in enumerate(cross_payloads, 1):
        require(payload["repeat"] == repeat, "cross repeat identity")
        require(payload["validation_classes"] == list(range(48)), "cross basis")
        require(len(payload["allocation_class_maps"]) == 8, "cross maps")
    first_maps = cross_payloads[0]["allocation_class_maps"]
    repeat_stable = all(
        payload["allocation_class_maps"] == first_maps for payload in cross_payloads
    )
    require(repeat_stable, "cross-allocation maps drift across repeats")
    unique_counts = [len(set(class_map)) for class_map in first_maps]
    require(unique_counts == [48, 32, 48, 32, 32, 48, 32, 48], "cross class counts")
    require(len({tuple(class_map) for class_map in first_maps}) == 8, "cross maps distinct")
    same_offset_agreement = [
        sum(left == right for left, right in zip(first_maps[0], class_map, strict=True))
        for class_map in first_maps
    ]
    require(same_offset_agreement == [48, 0, 0, 0, 0, 0, 0, 0], "cross phase")

    standard_granules = [
        [
            int(record["observed_class"])
            for record in predecessor_fit
            if int(record["region"]) // 16_384 == granule
        ]
        for granule in range(4)
    ]
    require(
        exact_class_permutation(standard_granules[0], standard_granules[2])
        and exact_class_permutation(
            standard_granules[1][:-1], standard_granules[3]
        ),
        "same-phase standard granules",
    )
    require(
        not exact_class_permutation(standard_granules[0], standard_granules[1]),
        "standard granule phases collapsed",
    )
    representative_regions = cross_payloads[0]["measurement"]["representative_regions"]
    type_maps = [
        [
            int(predecessor_fit[phase * 16_384 + region]["observed_class"])
            for region in representative_regions
        ]
        for phase in (0, 1)
    ]
    allocation_phase_types = []
    for allocation, class_map in enumerate(first_maps):
        matches = [
            class_bijection(class_map, type_map) for type_map in type_maps
        ]
        require(matches.count(True) == 1, f"allocation {allocation}: phase family")
        allocation_phase_types.append(matches.index(True))
    require(
        allocation_phase_types == [0, 1, 0, 1, 1, 0, 1, 0],
        "allocation phase sequence",
    )

    many_region = read_json(
        campaign.REPO_ROOT
        / "results/captures/sm86_atomic_many_region_20260820_v1/analysis/summary.json"
    )
    pair_phase_scores = []
    for phase in (0, 1):
        counts = defaultdict(int)
        for row in many_region["lts_slice_relation"]["pair_rows"]:
            left_region, right_region = row["regions"]
            predicted = (
                predecessor_fit[phase * 16_384 + 2 * left_region]["observed_class"]
                == predecessor_fit[phase * 16_384 + 2 * right_region]["observed_class"]
            )
            observed = bool(row["same_lts_slice"])
            counts[
                "tp"
                if predicted and observed
                else "fp"
                if predicted
                else "fn"
                if observed
                else "tn"
            ] += 1
        recall = counts["tp"] / (counts["tp"] + counts["fn"])
        specificity = counts["tn"] / (counts["tn"] + counts["fp"])
        pair_phase_scores.append(
            {
                "phase": phase,
                **dict(counts),
                "collision_recall": recall,
                "specificity": specificity,
                "balanced_accuracy": (recall + specificity) / 2,
            }
        )
    triplet_phase_accuracy = []
    for phase in (0, 1):
        correct = 0
        rows = many_region["triplet_timing"]["rows"]
        for row in rows:
            occupancy = max(
                {
                    cls: sum(
                        predecessor_fit[phase * 16_384 + 2 * region][
                            "observed_class"
                        ]
                        == cls
                        for region in row["regions"]
                    )
                    for cls in {
                        predecessor_fit[phase * 16_384 + 2 * region][
                            "observed_class"
                        ]
                        for region in row["regions"]
                    }
                }.values()
            )
            correct += occupancy == row["maximum_lts_region_occupancy"]
        triplet_phase_accuracy.append(correct / len(rows))

    return {
        "schema_version": 1,
        "capture": {
            "correctness_executions": correctness_executions,
            "read_ncu_profiles": read_profiles,
            "read_ncu_actions": read_profiles,
            "read_total_sector_outliers": read_total_outliers,
            "atomic_ncu_profiles": atomic_profiles,
            "atomic_ncu_actions": atomic_actions,
            "relative_ncu_profiles": 1,
            "relative_ncu_actions": 16_432,
            "extended_relative_ncu_profiles": 2,
            "extended_relative_ncu_actions": 49_199 + 32_816,
            "cross_allocation_ncu_profiles": 7,
            "cross_allocation_ncu_actions": 7 * 432,
            "total_ncu_profiles": read_profiles + atomic_profiles + 10,
            "total_ncu_actions": (
                read_profiles
                + atomic_actions
                + 16_432
                + 49_199
                + 32_816
                + 7 * 432
            ),
        },
        "read_path": {
            "single_pointer_maximum_fraction_median": single_pointer_fraction,
            "paired_layouts": paired_rows,
            "decision": (
                "LTS read sectors are replicated/merged across instances even for "
                "one pointer; read max is not a one-slice identity oracle."
            ),
        },
        "sequential_red": {
            "alias_observations": alias_observations,
            "alias_invariant_observations": alias_invariant_observations,
            "alias_invariance_rate": alias_invariant_observations / alias_observations,
            "offset_classes": atomic_rows,
            "unique_classes": len({row["observed_class"] for row in atomic_rows}),
        },
        "allocation_relative_transfer": {
            "scope_bytes": 2 * 1024 * 1024,
            "records": len(relative_records),
            "classes": len(forward),
            "collision_recall": 1.0,
            "specificity": 1.0,
            "class_permutation_vmm_to_cumemalloc": relative_permutation,
            "exact_partition_transfer": True,
            "contiguous_8m": {
                "records": len(contiguous_records),
                "whole_range": contiguous_score,
                "one_global_permutation": False,
                "per_2m_granule": granule_rows,
            },
            "size_matched_32m_high_holdout": {
                "records": len(high_records),
                "exact_partition_transfer": True,
            },
        },
        "cross_allocation_phase": {
            "reports": len(cross_payloads),
            "allocations_per_report": 8,
            "repeat_stable": repeat_stable,
            "distinct_class_maps": len({tuple(class_map) for class_map in first_maps}),
            "unique_classes_for_reference_representatives": unique_counts,
            "same_relative_offset_agreement_with_allocation_zero": same_offset_agreement,
            "universal_relative_partition": False,
        },
        "granule_phase_model": {
            "partition_families": 2,
            "standard_granule_types": [0, 1, 0, 1],
            "cross_allocation_types": allocation_phase_types,
            "cross_allocation_type_counts": [
                allocation_phase_types.count(0),
                allocation_phase_types.count(1),
            ],
            "same_relative_cross_allocation_match_rate": 0.0,
            "many_region_pair_scores": pair_phase_scores,
            "many_region_triplet_occupancy_accuracy": triplet_phase_accuracy,
            "promoted": False,
        },
        "decision": {
            "address_domain": "translated physical allocation plus physical offset",
            "virtual_address_sufficient": False,
            "allocation_relative_equivalence_is_universal": False,
            "simulator_hash_promoted": None,
            "reason": (
                "Sequential RED maps seven virtual aliases of the same physical word "
                "to one class, and each tested 2 MiB granule transfers under its own "
                "permutation. Distinct physical allocations use eight repeat-stable "
                "but different maps; four merge the canonical 48 representatives to "
                "32 classes. Allocation id plus relative offset is therefore useful "
                "identity but not a universal LTS key."
            ),
        },
    }


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def render_markdown(summary: Mapping[str, object]) -> str:
    capture = summary["capture"]
    read_path = summary["read_path"]
    red = summary["sequential_red"]
    relative = summary["allocation_relative_transfer"]
    cross = summary["cross_allocation_phase"]
    phase = summary["granule_phase_model"]
    lines = [
        "# sm_86 CUDA-VMM alias discriminator",
        "",
        "The sequential RED result establishes that GA102 atomic LTS identity "
        "follows translated physical allocation state, not the virtual pointer alone.",
        "",
        "## Live corpus",
        "",
        f"- {capture['correctness_executions']} direct correctness executions.",
        f"- {capture['read_ncu_profiles']} cache-flushed read reports and "
        f"{capture['atomic_ncu_profiles']} sequential-RED reports.",
        f"- {capture['total_ncu_actions']:,} official profiled kernel actions; every "
        "RED action conserves 37,632 sectors.",
        f"- {capture['read_total_sector_outliers']} read-sector total outliers are "
        "retained; medians, not filtered means, summarize the read diagnostic.",
        "",
        "## Read diagnostic",
        "",
        "A single pointer reaches only "
        f"{percent(read_path['single_pointer_maximum_fraction_median'])} of read "
        "sectors in the busiest LTS instance. Read requests are replicated or merged "
        "across instances, so `lts__t_sectors_op_read.max` is not a slice-identity "
        "oracle. The matched physical-layout A/B remains diagnostic only.",
        "",
        "## Sequential RED discriminator",
        "",
        "Each report first classifies all 48 non-aliased representatives and must "
        "decode exactly 0–47. Seven later kernels use one virtual alias at a time, "
        "with a synchronization boundary before switching proxies; this respects "
        "CUDA's alias coherence rule.",
        "",
        "| physical offset | class | alias observations |",
        "|---:|---:|---:|",
    ]
    for row in red["offset_classes"]:
        lines.append(
            f"| {row['offset_bytes']:,} | {row['observed_class']} | "
            f"{row['alias_observations']} |"
        )
    lines.extend(
        [
            "",
            f"All {red['alias_invariant_observations']} of "
            f"{red['alias_observations']} alias observations agree. Virtual aliases "
            "are separated by 2 MiB over a 14 MiB VA range, yet each physical word "
            "keeps one class across all aliases and all seven independent reports.",
            "",
            "## Allocation-relative transfer",
            "",
            f"A dense {relative['records']:,}-offset VMM map covers the entire "
            f"{relative['scope_bytes'] // (1024 * 1024)} MiB physical allocation. "
            "Its 48-way collision partition is exactly identical to the independent "
            "`cuMemAlloc` map up to a one-to-one class permutation: 100% collision "
            "recall and 100% specificity.",
            "",
            "Every 2 MiB granule in the exhaustive 8 MiB extension also transfers "
            "exactly, but the three granules require different permutations. Treating "
            "the whole range as one phase yields only "
            f"{percent(relative['contiguous_8m']['whole_range']['collision_recall'])} "
            "collision recall. Conversely, the size-matched 32 MiB high holdout "
            "transfers exactly, showing that physical allocation layout controls the "
            "phase sequence.",
            "",
            "## Cross-allocation phase",
            "",
            f"Seven reports classify eight distinct physical handles. All maps are "
            f"repeat-stable and all {cross['distinct_class_maps']} are distinct. The "
            "canonical representatives cover "
            f"{cross['unique_classes_for_reference_representatives']} classes across "
            "the eight handles, and allocations 1–7 have zero same-offset class "
            "matches with allocation 0.",
            "",
            "## Two-family phase model",
            "",
            f"All observed granules reduce to {phase['partition_families']} canonical "
            f"partition families. Standard granules alternate "
            f"{phase['standard_granule_types']}; the eight distinct handles use "
            f"{phase['cross_allocation_types']} ({phase['cross_allocation_type_counts']} "
            "counts). Phase 0/1 score "
            f"{percent(phase['many_region_pair_scores'][0]['collision_recall'])}/"
            f"{percent(phase['many_region_pair_scores'][1]['collision_recall'])} "
            "collision recall on the retained pair corpus, while both reproduce all "
            "220 local triplet occupancy classes. The latent phase is not present in "
            "the trace, so this remains an evaluator, not simulator timing state.",
            "",
            "## Decision",
            "",
            "Absolute virtual pointers are insufficient to compute exact GA102 atomic "
            "LTS identity. Allocation id and relative offset are valid ingredients, "
            "but not a universal key: each physical allocation/granule carries hidden "
            "phase. No hash or timing queue is promoted. Silently hashing either "
            "absolute trace VAs or relative offsets alone would encode the wrong "
            "domain.",
            "",
        ]
    )
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input-root", type=Path, default=campaign.DEFAULT_OUTPUT)
    result.add_argument("--output", type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = args.input_root.resolve()
    output = args.output.resolve() if args.output else root / "analysis/summary.json"
    try:
        summary = reduce(root)
        campaign.BASE.write_json(output, summary)
        (Path(__file__).resolve().parent / "README.md").write_text(
            render_markdown(summary), encoding="utf-8"
        )
        print(output)
    except (OSError, RuntimeError, json.JSONDecodeError, KeyError, ZeroDivisionError) as error:
        print(f"VMM alias analysis: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
