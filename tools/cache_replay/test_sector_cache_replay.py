"""Tests for the sectored set-associative cache-replay engine.

Covers the scope/framing guard (the engine declares itself a cache-only model,
not a GPU simulator, and enumerates its forbidden-surface refusal), the
synthetic-trace exactness checks, the hit-ratio range invariant, and the
per-cell result schema validator (positive + negative fixtures).
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import sector_cache_replay as scr  # noqa: E402

_SCHEMA = _HERE / "schemas" / "cache_replay_result.schema.json"
_L2 = 32 * 1024 * 1024


def _validator():
    schema = json.loads(_SCHEMA.read_text())
    return jsonschema.Draft202012Validator(schema)


def _write_empty_threadblocks(root: Path, count: int):
    threadblock_pb2, instruction_pb2, address_pb2 = (
        scr.import_trace_contract._import_pb_modules())
    threadblock_cls = threadblock_pb2.threadblock
    kernel_dir = (
        root / "threadblocks" / "device_0" / "stream_0" / "kernel_1")
    kernel_dir.mkdir(parents=True)
    for index in range(count):
        tb = threadblock_cls()
        tb.block_id.x, tb.block_id.y, tb.block_id.z = index, 0, 0
        (kernel_dir / f"kernel_1_{index},0,0.pb").write_bytes(
            tb.SerializeToString())
    return threadblock_cls, instruction_pb2, address_pb2


class TestDecodeBatch:
    def test_generated_class_is_acquired_once_and_batch_revalidated(
            self, monkeypatch, tmp_path):
        threadblock_cls, instruction_pb2, address_pb2 = (
            _write_empty_threadblocks(tmp_path, 3))

        class ThreadblockModule:
            lookups = 0

            @property
            def threadblock(self):
                self.lookups += 1
                return threadblock_cls

        module = ThreadblockModule()
        import_calls = 0

        def import_modules():
            nonlocal import_calls
            import_calls += 1
            return module, instruction_pb2, address_pb2

        monkeypatch.setattr(
            scr.import_trace_contract, "_import_pb_modules", import_modules)
        stream = scr.decode_cell_access_stream(tmp_path)

        assert stream.n_launches == 1
        assert stream.n_threadblocks == 3
        assert import_calls == 2
        assert module.lookups == 1

    def test_exit_revalidation_failure_prevents_stream_return(
            self, monkeypatch, tmp_path):
        threadblock_cls, instruction_pb2, address_pb2 = (
            _write_empty_threadblocks(tmp_path, 2))
        module = SimpleNamespace(threadblock=threadblock_cls)
        import_calls = 0

        def import_modules():
            nonlocal import_calls
            import_calls += 1
            if import_calls == 2:
                raise RuntimeError(
                    "authenticated binding state mutated during batch")
            return module, instruction_pb2, address_pb2

        monkeypatch.setattr(
            scr.import_trace_contract, "_import_pb_modules", import_modules)
        with pytest.raises(RuntimeError, match="mutated during batch"):
            scr.decode_cell_access_stream(tmp_path)
        assert import_calls == 2


class ScopeGuardTest(unittest.TestCase):
    """The engine's module docstring must declare its cache-core scope AND
    its lockstep relationship to the active Rust port: the pre-2026-07-02
    "no simulator" charter is retired, and the docstring now carries the
    same-commit parity obligation instead of a refusal list."""

    def test_docstring_declares_cache_scope_and_retired_charter(self):
        lower = (scr.__doc__ or "").lower()
        self.assertIn("cache hit-rate model", lower)
        self.assertIn("retired", lower)
        self.assertIn("2026-07-02", lower)

    def test_docstring_names_the_rust_lockstep_twin_and_parity_surfaces(self):
        doc = scr.__doc__ or ""
        self.assertIn("sim/mem/src/l2.rs", doc)
        self.assertIn("test_l2_rust_parity.py", doc)
        self.assertIn("python_parity_corpus_pins", doc)
        self.assertIn("same commit", doc.lower())


class SyntheticExactnessTest(unittest.TestCase):
    def setUp(self):
        self.cfg = scr.CacheReplayConfig(_L2, associativity=16)

    def test_single_sector_repeated_is_one_cold_then_all_hits(self):
        r = scr.replay_access_stream([0] * 10, [0] * 10, self.cfg)
        self.assertAlmostEqual(r.predicted_hit_ratio, 0.9, places=9)
        self.assertEqual(r.hits, 9)
        self.assertEqual(r.cold_misses, 1)

    def test_four_sector_fill_then_reuse(self):
        r = scr.replay_access_stream([0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 1, 2, 3, 0, 1, 2, 3], self.cfg)
        self.assertEqual(r.hits, 4)
        self.assertEqual(r.cold_misses, 4)
        self.assertAlmostEqual(r.predicted_hit_ratio, 0.5, places=9)

    def test_capacity_miss_on_tiny_cache(self):
        tiny = scr.CacheReplayConfig(128, associativity=1, num_slices=1)
        r = scr.replay_access_stream([0, 32, 0], [0, 0, 0], tiny)
        self.assertEqual(r.hits, 0)
        self.assertEqual(r.cold_misses, 2)
        self.assertEqual(r.capacity_misses, 1)
        self.assertEqual(r.conflict_misses, 0)

    def test_conflict_miss_distinguished_from_capacity(self):
        conf = scr.CacheReplayConfig(128 * 8, associativity=2, num_slices=1)
        r = scr.replay_access_stream([0, 4, 8, 0], [0, 0, 0, 0], conf)
        self.assertEqual(r.conflict_misses, 1)
        self.assertEqual(r.capacity_misses, 0)

    def test_breakdown_sums_to_total(self):
        r = scr.replay_access_stream([0, 1, 0, 4, 8, 0], [0, 0, 0, 0, 0, 0], self.cfg)
        total_misses = r.cold_misses + r.conflict_misses + r.capacity_misses
        self.assertEqual(r.hits + total_misses, r.total_accesses)


class RangeInvariantTest(unittest.TestCase):
    def test_hit_ratio_in_unit_interval(self):
        cfg = scr.CacheReplayConfig(_L2, associativity=16, num_slices=4,
                                    slice_hash_id="knuth_mult")
        r = scr.replay_access_stream([1, 2, 3, 1, 2, 3], [0, 1, 2, 0, 1, 2], cfg)
        self.assertGreaterEqual(r.predicted_hit_ratio, 0.0)
        self.assertLessEqual(r.predicted_hit_ratio, 1.0)

    def test_all_replacement_policies_run(self):
        for policy in scr.REPLACEMENT_POLICIES:
            cfg = scr.CacheReplayConfig(128 * 4, associativity=2, replacement_policy=policy)
            r = scr.replay_access_stream([0, 4, 8, 12, 0, 4], [0] * 6, cfg)
            self.assertGreaterEqual(r.predicted_hit_ratio, 0.0)
            self.assertLessEqual(r.predicted_hit_ratio, 1.0)

    def test_invalid_config_rejected(self):
        with self.assertRaises(ValueError):
            scr.CacheReplayConfig(_L2, associativity=16, replacement_policy="MRU")
        with self.assertRaises(ValueError):
            scr.CacheReplayConfig(_L2, associativity=16, slice_hash_id="nonexistent")


class ResultSchemaTest(unittest.TestCase):
    def _result(self):
        cfg = scr.CacheReplayConfig(_L2, associativity=16)
        r = scr.replay_access_stream([0, 0, 1, 1], [0, 0, 0, 0], cfg)
        out = r.as_dict()
        out["cache_parameters"] = cfg.provenance()
        return out

    def test_valid_result_admitted(self):
        _validator().validate(self._result())

    def test_missing_miss_breakdown_rejected(self):
        bad = self._result()
        del bad["miss_breakdown"]
        with self.assertRaises(jsonschema.ValidationError):
            _validator().validate(bad)

    def test_missing_cache_parameter_provenance_field_rejected(self):
        bad = self._result()
        del bad["cache_parameters"]["num_slices"]
        with self.assertRaises(jsonschema.ValidationError):
            _validator().validate(bad)

    def test_hit_ratio_out_of_range_rejected(self):
        bad = self._result()
        bad["predicted_hit_ratio"] = 1.5
        with self.assertRaises(jsonschema.ValidationError):
            _validator().validate(bad)

    def test_unknown_field_rejected(self):
        bad = self._result()
        bad["unexpected_field"] = 1
        with self.assertRaises(jsonschema.ValidationError):
            _validator().validate(bad)


if __name__ == "__main__":
    unittest.main()
