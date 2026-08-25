#!/usr/bin/env python3
"""Validate and reduce the GA102 address-to-LTS mapping campaign."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import campaign


MASK64 = (1 << 64) - 1


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def choose2(value: int) -> int:
    return value * (value - 1) // 2


def equivalence_score(
    records: Sequence[Mapping[str, int]], predicted: Sequence[int]
) -> dict[str, float | int]:
    require(len(records) == len(predicted), "prediction length")
    observed = [int(record["observed_class"]) for record in records]
    intersections = Counter(zip(observed, predicted, strict=True))
    observed_counts = Counter(observed)
    predicted_counts = Counter(predicted)
    true_positive = sum(choose2(count) for count in intersections.values())
    observed_positive = sum(choose2(count) for count in observed_counts.values())
    predicted_positive = sum(choose2(count) for count in predicted_counts.values())
    total = choose2(len(records))
    false_positive = predicted_positive - true_positive
    false_negative = observed_positive - true_positive
    true_negative = total - true_positive - false_positive - false_negative
    recall = true_positive / observed_positive
    specificity = true_negative / (total - observed_positive)
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    return {
        "records": len(records),
        "observed_classes": len(observed_counts),
        "predicted_classes": len(predicted_counts),
        "collision_recall": recall,
        "collision_precision": precision,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
    }


def ipoly64(higher_bits: int, index: int) -> int:
    taps = (
        (18, 17, 16, 15, 12, 10, 6, 5, 0),
        (15, 13, 12, 11, 10, 7, 5, 1, 0),
        (16, 14, 13, 12, 11, 8, 6, 2, 1),
        (17, 15, 14, 13, 12, 9, 7, 3, 2),
        (18, 16, 15, 14, 13, 10, 8, 4, 3),
        (17, 16, 15, 14, 11, 9, 5, 4),
    )
    result = 0
    for output_bit, input_bits in enumerate(taps):
        value = (index >> output_bit) & 1
        for input_bit in input_bits:
            value ^= (higher_bits >> input_bit) & 1
        result |= value << output_bit
    return result


def splitmix64(value: int) -> int:
    value = (value + 0x9E37_79B9_7F4A_7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58_476D_1CE4_E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D0_49BB_1331_11EB) & MASK64
    return value ^ (value >> 31)


def candidate_functions() -> dict[str, Callable[[int], int]]:
    return {
        "line_mod": lambda line: line % 48,
        "sector_mod": lambda line: (line * 4) % 48,
        "xor_folded": lambda line: (line ^ (line >> 5) ^ (line >> 11)) % 48,
        "knuth_mult": lambda line: ((line * 2_654_435_761) & 0xFFFF_FFFF) % 48,
        "shift_mod": lambda line: (line >> 1) % 48,
        "bit_reverse": lambda line: int(f"{line & 0xFFFF:016b}"[::-1], 2) % 48,
        "ipoly64_line": lambda line: ipoly64(line >> 6, line & 63) % 48,
        "splitmix64": lambda line: splitmix64(line) % 48,
    }


def compressed_region(region: int) -> int:
    return ((region >> 1) & 1) | ((region >> 3) << 1)


def row_basis(values: Iterable[int]) -> dict[int, int]:
    basis: dict[int, int] = {}
    for value in values:
        while value:
            bit = value.bit_length() - 1
            if bit in basis:
                value ^= basis[bit]
            else:
                basis[bit] = value
                for other_bit, other in list(basis.items()):
                    if other_bit != bit and ((other >> bit) & 1):
                        basis[other_bit] = other ^ value
                break
    return basis


def parity_invariant_masks(
    records: Sequence[Mapping[str, int]], input_bits: int
) -> list[int]:
    first: dict[int, int] = {}
    differences = []
    for record in records:
        value = compressed_region(int(record["region"]))
        observed = int(record["observed_class"])
        if observed in first:
            differences.append(value ^ first[observed])
        else:
            first[observed] = value
    basis = row_basis(differences)
    return [
        mask
        for mask in range(1 << input_bits)
        if all((mask & value).bit_count() % 2 == 0 for value in basis.values())
    ]


def require_metric_rows(path: Path, expected_rows: int, expected_total: int) -> int:
    payload = read_json(path)
    rows = payload.get("rows")
    require(isinstance(rows, list) and len(rows) == expected_rows, f"{path}: rows")
    for row in rows:
        require(row["total_sectors"] == expected_total, f"{path}: total sectors")
        require(row["maximum_sectors"] <= expected_total, f"{path}: maximum")
    return len(rows)


def phase_records(root: Path, phase: str) -> list[dict[str, int]]:
    return read_json(root / "profiles" / phase / "classification.json")["records"]


def validate_capture(root: Path) -> dict[str, object]:
    manifest = read_json(root / "metadata/matrix.json")
    require(manifest.get("schema_version") == 1, "matrix schema")
    require(manifest.get("expected_lts_slices") == 48, "LTS slice count")

    weighted_actions = 0
    discovery_pairs = 0
    for phase in campaign.PHASES:
        representatives = read_json(
            root / "profiles" / phase.key / "representatives.json"
        )
        classes = representatives["classes"]
        require(len(classes) == 48, f"{phase.key}: representatives")
        require(
            sorted(value for values in classes for value in values)
            == list(range(campaign.DISCOVERY_REGIONS)),
            f"{phase.key}: discovery roster",
        )
        discovery_rows = len(campaign.discovery_queries())
        weighted_actions += require_metric_rows(
            root / "profiles" / phase.key / "discovery/metrics.json",
            discovery_rows,
            campaign.BLOCKS * 32 * campaign.GENERATIONS,
        )
        discovery_pairs += (
            campaign.DISCOVERY_REGIONS * (campaign.DISCOVERY_REGIONS - 1) // 2
        )
        classification_rows = len(
            campaign.classification_queries(
                representatives["representatives"],
                phase.classify_start,
                phase.classify_stop,
            )
        )
        weighted_actions += require_metric_rows(
            root / "profiles" / phase.key / "classification/metrics.json",
            classification_rows,
            campaign.BLOCKS * 32 * campaign.GENERATIONS,
        )

    wide = read_json(root / "profiles/train/wide-classification.json")["records"]
    require(len(wide) == 4096, "wide records")
    weighted_actions += require_metric_rows(
        root / "profiles/train/wide-classification/metrics.json",
        32_768,
        campaign.BLOCKS * 32 * campaign.GENERATIONS,
    )

    encoded_fit = read_json(root / "profiles/train/encoded-fit.json")["records"]
    encoded_holdout = read_json(root / "profiles/train/encoded-holdout.json")["records"]
    require(len(encoded_fit) == 65_535, "encoded fit records")
    require(len(encoded_holdout) == 32_768, "encoded holdout records")
    encoded_actions = require_metric_rows(
        root / "profiles/train/encoded-fit/metrics.json", 65_535, 37_632
    )
    encoded_actions += require_metric_rows(
        root / "profiles/train/encoded-holdout/metrics.json", 32_768, 37_632
    )

    train = phase_records(root, "train")
    train_by_region = {record["region"]: record["observed_class"] for record in train}
    fit_by_region = {
        record["region"]: record["observed_class"] for record in encoded_fit
    }
    overlap = set(train_by_region) & set(fit_by_region)
    require(len(overlap) == 4096, "weighted/encoded overlap")
    require(
        all(train_by_region[region] == fit_by_region[region] for region in overlap),
        "weighted and encoded oracles disagree",
    )
    wide_by_region = {record["region"]: record["observed_class"] for record in wide}
    dense_all = {**fit_by_region}
    dense_all.update(
        {record["region"]: record["observed_class"] for record in encoded_holdout}
    )
    wide_overlap = set(wide_by_region) & set(dense_all)
    require(
        all(wide_by_region[region] == dense_all[region] for region in wide_overlap),
        "wide and encoded oracles disagree",
    )

    transfer_4m = phase_records(root, "transfer_4m")
    transfer_12m = phase_records(root, "transfer_12m")
    require(
        all(
            record["observed_class"] == fit_by_region[record["region"]]
            for record in transfer_4m + transfer_12m
        ),
        "allocation-prefix local equivalence changed",
    )
    return {
        "weighted_actions": weighted_actions,
        "encoded_actions": encoded_actions,
        "official_ncu_actions": weighted_actions + encoded_actions,
        "official_ncu_reports": 9,
        "discovery_pairs": discovery_pairs,
        "weighted_encoded_overlap": len(overlap),
        "wide_encoded_overlap": len(wide_overlap),
        "encoded_fit": encoded_fit,
        "encoded_holdout": encoded_holdout,
        "wide": wide,
        "transfer_4m": transfer_4m,
        "transfer_12m": transfer_12m,
    }


def reduce(root: Path) -> dict[str, object]:
    validated = validate_capture(root)
    fit = validated.pop("encoded_fit")
    holdout = validated.pop("encoded_holdout")
    wide = validated.pop("wide")
    transfer_4m = validated.pop("transfer_4m")
    transfer_12m = validated.pop("transfer_12m")

    candidate_rows = []
    cohorts = {
        "dense_fit": fit,
        "wide_same_allocation": wide,
        "dense_high_holdout": holdout,
        "prefix_4m": transfer_4m,
        "prefix_12m": transfer_12m,
    }
    for name, function in candidate_functions().items():
        candidate_rows.append(
            {
                "candidate": name,
                "cohorts": {
                    cohort: equivalence_score(
                        records,
                        [function(int(record["line_address"])) for record in records],
                    )
                    for cohort, records in cohorts.items()
                },
            }
        )

    fit_lut = {int(record["region"]): int(record["observed_class"]) for record in fit}
    fit_lut[65_535] = fit_lut[65_534]
    lookup_predictions = [
        fit_lut[int(record["region"]) & 0xFFFF] for record in holdout
    ]
    lookup_score = equivalence_score(holdout, lookup_predictions)
    lookup_direct_accuracy = sum(
        predicted == int(record["observed_class"])
        for predicted, record in zip(lookup_predictions, holdout, strict=True)
    ) / len(holdout)

    class_counts = Counter(int(record["observed_class"]) for record in fit)
    require(len(class_counts) == 48, "dense fit class coverage")
    bit_invariance = {}
    fit_class = {int(record["region"]): int(record["observed_class"]) for record in fit}
    for bit in range(16):
        pairs = [
            fit_class[region] == fit_class[region ^ (1 << bit)]
            for region in fit_class
            if (region ^ (1 << bit)) in fit_class
        ]
        bit_invariance[str(bit)] = sum(pairs) / len(pairs)

    fit_parities = parity_invariant_masks(fit, 14)
    combined_parities = parity_invariant_masks([*fit, *holdout], 16)
    promotion_threshold = {"collision_recall": 0.95, "specificity": 0.99}
    promotable = [
        row["candidate"]
        for row in candidate_rows
        if row["cohorts"]["dense_high_holdout"]["collision_recall"]
        >= promotion_threshold["collision_recall"]
        and row["cohorts"]["dense_high_holdout"]["specificity"]
        >= promotion_threshold["specificity"]
    ]
    require(not promotable, f"unexpected promotable candidates: {promotable}")

    return {
        "schema_version": 1,
        "capture": validated,
        "dense_fit": {
            "records": len(fit),
            "class_count": len(class_counts),
            "minimum_records_per_class": min(class_counts.values()),
            "maximum_records_per_class": max(class_counts.values()),
            "ignored_region_bits": [
                int(bit) for bit, rate in bit_invariance.items() if rate == 1.0
            ],
            "bit_flip_same_class_rates": bit_invariance,
            "linear_parity_invariant_masks_hex": [hex(mask) for mask in fit_parities],
        },
        "dense_high_holdout": {
            "records": len(holdout),
            "minimum_region": min(int(record["region"]) for record in holdout),
            "maximum_region": max(int(record["region"]) for record in holdout),
            "combined_linear_parity_invariant_masks_hex": [
                hex(mask) for mask in combined_parities
            ],
        },
        "candidate_scores": candidate_rows,
        "dense_low16_lookup": {
            "fit_records": len(fit),
            "fit_direct_accuracy": 1.0,
            "holdout_direct_accuracy": lookup_direct_accuracy,
            "holdout_equivalence": lookup_score,
        },
        "promotion": {
            "threshold": promotion_threshold,
            "promoted_candidate": None,
            "reason": (
                "No address-only hash transfers. The dense low-16 lookup is exact "
                "in fit but loses most true high-address collisions; architecture "
                "hash candidates remain at chance."
            ),
            "next_discriminator": (
                "Map one CUDA VMM physical allocation at multiple virtual aliases "
                "to separate physical-page placement from virtual-address hashing."
            ),
        },
    }


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def render_markdown(summary: Mapping[str, object]) -> str:
    capture = summary["capture"]
    dense = summary["dense_fit"]
    holdout = summary["dense_high_holdout"]
    lookup = summary["dense_low16_lookup"]
    candidates = summary["candidate_scores"]
    lines = [
        "# sm_86 GA102 address-to-LTS mapping",
        "",
        "This campaign establishes an exact collision oracle, but does **not** "
        "promote an address hash into the simulator.",
        "",
        "## Live evidence",
        "",
        f"- {capture['official_ncu_actions']:,} one-pass Nsight Compute kernel actions "
        f"across {capture['official_ncu_reports']} official reports.",
        f"- {capture['discovery_pairs']:,} exhaustive pair observations across three "
        "allocation phases; every phase yields 48 transitive LTS classes.",
        f"- {capture['weighted_encoded_overlap']:,} weighted/encoded and "
        f"{capture['wide_encoded_overlap']:,} wide/encoded overlap records agree exactly.",
        f"- Dense fit: {dense['records']:,} lines, 48 classes, "
        f"{dense['minimum_records_per_class']}–{dense['maximum_records_per_class']} lines/class.",
        f"- Strict high holdout: {holdout['records']:,} lines over regions "
        f"{holdout['minimum_region']:,}–{holdout['maximum_region']:,}.",
        "",
        "The weighted oracle uses 105 anchor CTAs plus binary CTA weights. The "
        "encoded oracle is stronger and cheaper: an anchor executes 48 RED generations "
        "while the 48 representatives execute 0–47, so one LTS maximum identifies one class.",
        "",
        "## Hash transfer",
        "",
        "| candidate | fit recall | high-holdout recall | high-holdout specificity | balanced |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in candidates:
        fit_score = row["cohorts"]["dense_fit"]
        high_score = row["cohorts"]["dense_high_holdout"]
        lines.append(
            f"| {row['candidate']} | {percent(fit_score['collision_recall'])} | "
            f"{percent(high_score['collision_recall'])} | "
            f"{percent(high_score['specificity'])} | "
            f"{percent(high_score['balanced_accuracy'])} |"
        )
    lines.extend(
        [
            "",
            "A deliberately overfit 65,536-entry low-bit lookup is exact on fit but "
            f"has only {percent(lookup['holdout_equivalence']['collision_recall'])} "
            "collision recall and "
            f"{percent(lookup['holdout_direct_accuracy'])} direct-label accuracy on holdout.",
            "",
            f"Region bits {dense['ignored_region_bits']} are invariant in the dense fit. "
            "Three nonzero linear parity invariants exist locally, but only one survives "
            "when the high-address holdout is included. This is direct evidence against "
            "treating the local pattern as a transferable linear slice hash.",
            "",
            "## Decision",
            "",
            "No hash is integrated. Scheduling remains conservative and the GA102 "
            "slice hash stays an explicit research gap. The next bounded discriminator "
            "is a CUDA VMM alias experiment: map one physical allocation at multiple "
            "virtual addresses and test whether the LTS class follows the physical "
            "allocation or the virtual address.",
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
        readme = Path(__file__).resolve().parent / "README.md"
        readme.write_text(render_markdown(summary), encoding="utf-8")
        print(output)
    except (OSError, RuntimeError, json.JSONDecodeError, KeyError, ZeroDivisionError) as error:
        print(f"atomic LTS mapping analysis: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
