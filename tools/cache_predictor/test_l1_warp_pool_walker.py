import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from tools.cache_predictor import l1_warp_pool_walker as walker

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
