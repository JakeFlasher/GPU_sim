"""Focused tests for the CUDA-VMM alias discriminator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


campaign = load("tested_vmm_alias_campaign", HERE / "campaign.py")
prior = sys.modules.get("campaign")
sys.modules["campaign"] = campaign
try:
    analyze = load("tested_vmm_alias_analyze", HERE / "analyze.py")
finally:
    if prior is None:
        del sys.modules["campaign"]
    else:
        sys.modules["campaign"] = prior


def test_matrix_is_closed_and_offsets_fit_one_vmm_granule() -> None:
    cells = campaign.cells()
    assert len(cells) == len({cell.key for cell in cells}) == 19
    assert len([cell for cell in cells if cell.primary]) == 16
    assert {cell.mode for cell in cells if not cell.primary} == set(
        campaign.CONTROL_MODES
    )
    assert max(campaign.OFFSETS) + 4 <= 2 * 1024 * 1024
    assert all(offset % 128 == 0 for offset in campaign.OFFSETS)


def test_weighted_read_geometry_retains_a_dominant_anchor() -> None:
    assert campaign.GROUP_BLOCK_COUNTS == (105, 1, 2, 4, 8, 16, 32)
    assert sum(campaign.GROUP_BLOCK_COUNTS) == 168
    assert campaign.GROUP_BLOCK_COUNTS[0] > sum(campaign.GROUP_BLOCK_COUNTS[1:])


def test_primary_targets_change_only_physical_layout_mode() -> None:
    alias = campaign.Cell("alias_same_offset", 65_536, True)
    linear = campaign.Cell("single_mapping_alias_stride", 65_536, True)
    alias_target = alias.target(1)
    linear_target = linear.target(1)
    assert alias_target[:3] == linear_target[:3]
    assert alias_target[-1] == linear_target[-1] == "65536"
    assert alias_target[-2] != linear_target[-2]


def test_markdown_percent_is_stable() -> None:
    assert analyze.percent(1.0) == "100.00%"
    assert analyze.percent(0.0804899387576553) == "8.05%"


def test_equivalence_scoring_distinguishes_permutation_from_merge() -> None:
    canonical = [0, 0, 1, 1, 2, 2]
    permuted = [9, 9, 7, 7, 4, 4]
    exact = analyze.equivalence_score(permuted, canonical)
    assert exact == {
        "collision_recall": 1.0,
        "specificity": 1.0,
        "balanced_accuracy": 1.0,
    }
    merged = analyze.equivalence_score([9, 9, 9, 9, 4, 4], canonical)
    assert merged["collision_recall"] < 1.0
    assert merged["specificity"] == 1.0
    assert analyze.exact_class_permutation(permuted, canonical) is False


def test_exact_class_permutation_requires_all_48_classes() -> None:
    canonical = list(range(48)) * 2
    permuted = [value ^ 7 for value in canonical]
    assert analyze.exact_class_permutation(permuted, canonical)
    merged = [value % 32 for value in canonical]
    assert not analyze.exact_class_permutation(merged, canonical)
