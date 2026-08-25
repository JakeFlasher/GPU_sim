"""Focused tests for the many-region atomic campaign."""

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


campaign = load("tested_many_region_campaign", HERE / "campaign.py")
prior_campaign = sys.modules.get("campaign")
sys.modules["campaign"] = campaign
try:
    analyze = load("tested_many_region_analyze", HERE / "analyze.py")
finally:
    if prior_campaign is None:
        del sys.modules["campaign"]
    else:
        sys.modules["campaign"] = prior_campaign


def test_matrix_cardinality_and_identity_are_closed() -> None:
    triplets = campaign.triplet_cells()
    pairs = campaign.pair_cells()
    assert len(triplets) == len({cell.key for cell in triplets}) == 220
    assert len(pairs) == len({cell.key for cell in pairs}) == 183
    assert all(len(set(cell.regions)) == 3 for cell in triplets)
    assert all(len(set(cell.regions)) == 2 for cell in pairs)


def test_target_fixes_allocation_and_runtime_region_list() -> None:
    cell = campaign.TripletCell((0, 57, 4))
    command = cell.target(trace=True, trials=7)
    assert command[-2:] == [str(campaign.FIXED_DATA_BYTES), "0,57,4"]
    assert command[2:5] == ["84", "64", "7"]


def test_local_bit_drop_is_deliberately_nontransferable() -> None:
    base_region = 4_296_613_888
    mapped = lambda offset: analyze.hash_value(2 * (base_region + offset), "local_bit_drop")
    assert mapped(0) == mapped(2)
    assert mapped(0) != mapped(4)
    assert mapped(0) == mapped(96), "local rule makes the known false transfer prediction"
