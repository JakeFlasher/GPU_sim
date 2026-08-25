"""Focused tests for the weighted GA102 LTS mapping campaign."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_campaign():
    path = HERE / "campaign.py"
    spec = importlib.util.spec_from_file_location("tested_atomic_lts_campaign", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


campaign = load_campaign()
prior_campaign = sys.modules.get("campaign")
sys.modules["campaign"] = campaign
try:
    analyzer_spec = importlib.util.spec_from_file_location(
        "tested_atomic_lts_analyzer", HERE / "analyze.py"
    )
    assert analyzer_spec is not None and analyzer_spec.loader is not None
    analyze = importlib.util.module_from_spec(analyzer_spec)
    sys.modules[analyzer_spec.name] = analyze
    analyzer_spec.loader.exec_module(analyze)
finally:
    if prior_campaign is None:
        del sys.modules["campaign"]
    else:
        sys.modules["campaign"] = prior_campaign


def test_weighted_geometry_is_uniquely_decodable() -> None:
    assert sum(campaign.GROUP_BLOCK_COUNTS) == campaign.BLOCKS == 168
    assert campaign.ANCHOR_BLOCKS > sum(campaign.CANDIDATE_BLOCKS)
    assert campaign.CANDIDATE_BLOCKS == tuple(1 << bit for bit in range(6))
    for mask in range(64):
        encoded = sum(
            weight
            for bit, weight in enumerate(campaign.CANDIDATE_BLOCKS)
            if mask & (1 << bit)
        )
        assert encoded == mask


def test_discovery_queries_cover_every_pair() -> None:
    queries = campaign.discovery_queries()
    pairs = {
        tuple(sorted((query.anchor, candidate)))
        for query in queries
        for candidate in query.candidates
    }
    expected = campaign.DISCOVERY_REGIONS * (campaign.DISCOVERY_REGIONS - 1) // 2
    assert len(queries) == 5547
    assert len(pairs) == expected
    assert all(len(set(query.regions)) == 7 for query in queries)


def test_classification_compares_each_unknown_to_all_representatives() -> None:
    representatives = list(range(campaign.EXPECTED_LTS_SLICES))
    queries = campaign.classification_queries(representatives, 0, 64)
    by_anchor: dict[int, set[int]] = {}
    for query in queries:
        by_anchor.setdefault(query.anchor, set()).update(query.candidates)
    assert set(by_anchor) == set(range(48, 64))
    assert all(values == set(representatives) for values in by_anchor.values())
    assert len(queries) == 16 * 8


def test_decode_mask_uses_exact_sector_units() -> None:
    unit = 32 * campaign.GENERATIONS
    row = {
        "id": 7,
        "maximum_sectors": (campaign.ANCHOR_BLOCKS + 0b10_0101) * unit,
        "total_sectors": campaign.BLOCKS * unit,
    }
    assert campaign.decode_mask(row) == 0b10_0101


def test_prefix_phases_are_fit_and_strict_transfer() -> None:
    assert campaign.PHASES[0].allocation_prefix_bytes == 0
    assert campaign.PHASES[0].cohort == "fit"
    assert {phase.allocation_prefix_bytes for phase in campaign.PHASES[1:]} == {
        4 * 1024 * 1024,
        12 * 1024 * 1024,
    }
    assert all(phase.cohort == "strict_transfer" for phase in campaign.PHASES[1:])


def test_wide_holdout_is_bounded_and_spans_high_address_bits() -> None:
    regions = campaign.wide_regions()
    assert len(regions) == len(set(regions)) == 4096
    assert min(regions) >= 4096
    assert max(regions) > 250_000
    assert all(
        0 <= region < campaign.FIXED_DATA_BYTES // campaign.REGION_STRIDE_BYTES
        for region in regions
    )
    for bit in range(12, 18):
        boundary = 1 << bit
        if bit > 12:
            assert boundary - 1 in regions
        assert boundary in regions
        assert boundary + 1 in regions


def test_dense_encoded_fit_and_holdout_are_disjoint_and_bounded() -> None:
    fit = campaign.dense_fit_regions()
    holdout = campaign.dense_holdout_regions()
    assert len(fit) == 65_535
    assert fit[0] == 0 and fit[-1] == 65_534
    assert len(holdout) == len(set(holdout)) == 32_768
    assert set(fit).isdisjoint(holdout)
    assert min(holdout) >= 65_535
    assert max(holdout) < campaign.FIXED_DATA_BYTES // campaign.REGION_STRIDE_BYTES
    for bit in (16, 17):
        assert (1 << bit) - 1 in holdout
        assert 1 << bit in holdout
        assert (1 << bit) + 1 in holdout


def test_equivalence_score_is_label_permutation_invariant() -> None:
    records = [
        {"observed_class": 0},
        {"observed_class": 0},
        {"observed_class": 1},
        {"observed_class": 1},
    ]
    score = analyze.equivalence_score(records, [9, 9, 4, 4])
    assert score["collision_recall"] == 1.0
    assert score["specificity"] == 1.0
    assert score["balanced_accuracy"] == 1.0


def test_compressed_region_removes_only_observed_invariant_bits() -> None:
    base = 0b1011_0110_1010
    assert analyze.compressed_region(base) == analyze.compressed_region(base ^ 1)
    assert analyze.compressed_region(base) == analyze.compressed_region(base ^ 4)
    assert analyze.compressed_region(base) != analyze.compressed_region(base ^ 2)
