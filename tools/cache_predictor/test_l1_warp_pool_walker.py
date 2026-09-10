import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from tools.cache_predictor import l1_warp_pool_walker as walker

from tools.cache_predictor.l1_warp_pool_walker import (
    WalkerConfigError,
    WarpPoolAccessInvalid,
    resolve_latencies,
    resolve_line_capacity,
    walk_warp_pool,
)
from tools.cache_predictor.l1_walker import (
    L1CtaToSmScheduleAssignmentInvalid,
    L1CtaToSmScheduleMissingOccupancy,
    L1CtaToSmScheduleUnknown,
    L1WalkerCapacityInvalid,
    L1WalkerWaysInvalid,
    block_to_sm_id,
    walk_per_sm_l1,
    walk_per_sm_per_set_l1,
)


def _card(tmp_path, *, capacity_unit="bytes", latency_unit="cycles_per_dependent_load"):
    document = {
        "arch": "sm_86",
        "properties": {
            "l1_capacity_bytes": {
                "kind": "value",
                "unit": capacity_unit,
                "value": 98304,
            },
            "l1_hit_latency_cycles": {
                "kind": "value",
                "unit": latency_unit,
                "value": 33.0,
            },
            "l2_hit_latency_cycles": {
                "kind": "value",
                "unit": latency_unit,
                "value": 243.0,
            },
            "mshr_outstanding_misses": {
                "kind": "value",
                "value": 32,
            },
        },
    }
    path = tmp_path / "card.json"
    path.write_text(json.dumps(document))
    return path


def test_explicit_capacity_unit_must_be_bytes(tmp_path):
    with pytest.raises(WalkerConfigError, match="is not bytes"):
        resolve_line_capacity(_card(tmp_path, capacity_unit="cycles"))


def test_explicit_latency_unit_must_be_cycles(tmp_path):
    with pytest.raises(WalkerConfigError, match="is not cycles_per_dependent_load"):
        resolve_latencies(_card(tmp_path, latency_unit="nanoseconds"))


def test_large_decimal_capacity_is_not_rounded(tmp_path):
    path = _card(tmp_path)
    path.write_text(path.read_text().replace("98304", "9007199254740993.0", 1))
    with pytest.raises(WalkerConfigError, match="9007199254740993"):
        resolve_line_capacity(path)


def test_card_integer_bounds_match_rust_resolvers(tmp_path):
    path = _card(tmp_path)
    raw = path.read_text().replace('"value": 32', '"value": 4294967296', 1)
    path.write_text(raw)
    from tools.cache_predictor.l1_warp_pool_walker import resolve_mshr_count

    with pytest.raises(WalkerConfigError, match="does not fit u32"):
        resolve_mshr_count(path)


def test_huge_exponent_is_rejected_before_integer_materialization(tmp_path):
    path = _card(tmp_path)
    path.write_text(path.read_text().replace("98304", "1e1000000", 1))
    with pytest.raises(WalkerConfigError, match="positive integer"):
        resolve_line_capacity(path)


@pytest.mark.parametrize("token", ["1e-400", "1e400", "1" + "0" * 400])
def test_latency_must_remain_positive_and_finite_after_conversion(tmp_path, token):
    path = _card(tmp_path)
    path.write_text(path.read_text().replace("33.0", token, 1))
    with pytest.raises(WalkerConfigError, match="finite latency"):
        resolve_latencies(path)


def test_representable_subnormal_latency_is_preserved(tmp_path):
    path = _card(tmp_path)
    path.write_text(path.read_text().replace("33.0", "5e-324", 1))
    assert resolve_latencies(path).hit_latency == float.fromhex("0x0.0000000000001p-1022")


def test_card_size_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(walker, "MAX_CARD_BYTES", 1024)
    path = _card(tmp_path)
    exact_limit = path.read_bytes().ljust(walker.MAX_CARD_BYTES, b" ")
    path.write_bytes(exact_limit)
    assert resolve_line_capacity(path).capacity_bytes == 98304

    path.write_bytes(exact_limit + b" " * 4096)
    original_open = Path.open
    consumed = []

    @contextmanager
    def tracked_open(card_path, *args, **kwargs):
        with original_open(card_path, *args, **kwargs) as stream:
            def read(size=-1):
                raw = stream.read(size)
                consumed.append(len(raw))
                return raw
            yield SimpleNamespace(read=read)

    monkeypatch.setattr(Path, "open", tracked_open)
    with pytest.raises(WalkerConfigError, match="exceeds"):
        resolve_line_capacity(path)
    assert consumed and sum(consumed) <= walker.MAX_CARD_BYTES + 1


def test_card_path_rejects_leaf_and_parent_symlinks(tmp_path):
    target = _card(tmp_path)
    leaf = tmp_path / "leaf.json"
    leaf.symlink_to(target)
    with pytest.raises(WalkerConfigError, match="contains a symlink"):
        resolve_line_capacity(leaf)
    parent = tmp_path / "parent"
    parent.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(WalkerConfigError, match="contains a symlink"):
        resolve_line_capacity(parent / target.name)


def test_duplicate_warp_keys_cannot_drop_streams():
    with pytest.raises(WalkerConfigError, match="unique"):
        walk_warp_pool(
            [("warp", [(0, True)]), ("warp", [(1, True)])],
            line_capacity=8, hit_latency=1.0, miss_latency=4.0, mshr_count=1,
        )


def test_non_tuple_access_has_typed_error():
    with pytest.raises(WarpPoolAccessInvalid, match="tuple/list"):
        walk_warp_pool(
            [("warp", [123])], line_capacity=8,
            hit_latency=1.0, miss_latency=4.0, mshr_count=1,
        )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"mshr_count": 0}, "mshr_count"),
        ({"co_resident_warps": 0}, "co_resident_warps"),
        ({"hit_latency": float("nan")}, "latencies"),
        ({"hit_latency": 5.0, "miss_latency": 4.0}, "latencies"),
    ],
)
def test_invalid_pool_parameters_fail_closed(kwargs, message):
    base = {
        "line_capacity": 8,
        "hit_latency": 1.0,
        "miss_latency": 4.0,
        "mshr_count": 1,
    }
    base.update(kwargs)
    with pytest.raises(WalkerConfigError, match=message):
        walk_warp_pool([("warp", [(0, True)])], **base)


def test_legacy_walker_rejects_non_integral_geometry():
    with pytest.raises(L1WalkerCapacityInvalid, match="capacity_bytes"):
        walk_per_sm_l1([], sm_count=1, capacity_bytes=128.5, schedule_policy="round_robin_homogeneous")
    with pytest.raises(L1WalkerWaysInvalid, match="ways"):
        walk_per_sm_per_set_l1([], sm_count=1, capacity_bytes=128, ways=1.5, schedule_policy="round_robin_homogeneous")


def test_legacy_walkers_reject_zero_sm_count_on_empty_input():
    with pytest.raises(ValueError, match="sm_count"):
        walk_per_sm_l1([], sm_count=0, capacity_bytes=128, schedule_policy="round_robin_homogeneous")
    with pytest.raises(ValueError, match="sm_count"):
        walk_per_sm_per_set_l1([], sm_count=0, capacity_bytes=128, ways=1, schedule_policy="round_robin_homogeneous")


def test_occupancy_assignment_rejects_coercible_non_integer_ids():
    with pytest.raises(L1CtaToSmScheduleAssignmentInvalid, match="non-integer"):
        block_to_sm_id(0, sm_count=1, schedule_policy="occupancy_derived", occupancy_derived_assignment=[True])


def test_empty_legacy_walkers_validate_schedule_policy():
    with pytest.raises(L1CtaToSmScheduleUnknown):
        walk_per_sm_l1([], sm_count=1, capacity_bytes=128, schedule_policy="unknown")
    with pytest.raises(L1CtaToSmScheduleMissingOccupancy):
        walk_per_sm_per_set_l1([], sm_count=1, capacity_bytes=128, ways=1, schedule_policy="occupancy_derived")
