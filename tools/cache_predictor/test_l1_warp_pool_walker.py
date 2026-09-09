import json

import pytest

from tools.cache_predictor.l1_warp_pool_walker import (
    WalkerConfigError,
    resolve_latencies,
    resolve_line_capacity,
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
