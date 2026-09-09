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
