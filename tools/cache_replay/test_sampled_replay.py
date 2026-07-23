"""Tests for the sampled replay mode + sampled-vs-full fidelity gate.

Positive: set sampling on the real cohort processes <= 10% of records and lands
within the 0.02 absolute-hit-ratio band on at least 27 of 30 cells while
preserving the full-cohort accuracy verdict. Negative: a sampling mode that
exceeds the band on more than three cells, or that flips the accuracy verdict,
fails the fidelity gate.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import sector_cache_replay as scr  # noqa: E402
import sampled_replay as sr  # noqa: E402

_L2 = 32 * 1024 * 1024


class SetSamplingExactnessTest(unittest.TestCase):
    def test_sampled_subset_is_exact_for_sampled_sets(self):
        # A single repeated sector lands in one set; sampling that set's modulus
        # reproduces the full hit ratio exactly (per-set independence).
        cfg = scr.CacheReplayConfig(_L2, associativity=8, num_slices=1)
        full = scr.replay_access_stream([0] * 10, [0] * 10, cfg, classify_misses=False)
        # set_key for line 0 is 0, so modulus 1 (and any modulus dividing into 0)
        # processes it; use modulus 1 to guarantee inclusion.
        samp = scr.replay_access_stream([0] * 10, [0] * 10, cfg, set_sample_modulus=1)
        self.assertEqual(full.predicted_hit_ratio, samp.predicted_hit_ratio)

    def test_sampling_reduces_processed_records(self):
        cfg = scr.CacheReplayConfig(_L2, associativity=8, num_slices=1)
        lids = list(range(2000))  # spread across many sets
        sof = [0] * 2000
        full = scr.replay_access_stream(lids, sof, cfg, classify_misses=False)
        samp = scr.replay_access_stream(lids, sof, cfg, set_sample_modulus=32)
        self.assertLess(samp.total_accesses, full.total_accesses)
        self.assertLessEqual(samp.total_accesses / full.total_accesses, 0.10)


class FidelityGatePureTest(unittest.TestCase):
    def _cells(self, n, delta_over_count):
        cells = []
        for i in range(n):
            d = 0.05 if i < delta_over_count else 0.005
            cells.append({"abs_delta": d, "sampled_record_fraction": 0.03})
        return cells

    def test_passes_when_within_bar_and_verdict_preserved(self):
        verdict = sr.evaluate_sampling_fidelity(
            self._cells(30, delta_over_count=2), ac4_verdict_preserved=True)
        self.assertTrue(verdict["passed"])

    def test_fails_when_more_than_three_cells_over_bar(self):
        verdict = sr.evaluate_sampling_fidelity(
            self._cells(30, delta_over_count=4), ac4_verdict_preserved=True)
        self.assertFalse(verdict["passed"])

    def test_fails_when_ac4_verdict_flips(self):
        verdict = sr.evaluate_sampling_fidelity(
            self._cells(30, delta_over_count=0), ac4_verdict_preserved=False)
        self.assertFalse(verdict["passed"])

    def test_fails_when_record_fraction_too_high(self):
        cells = [{"abs_delta": 0.001, "sampled_record_fraction": 0.5} for _ in range(30)]
        verdict = sr.evaluate_sampling_fidelity(cells, ac4_verdict_preserved=True)
        self.assertFalse(verdict["passed"])


class EmittedFidelityArtifactTest(unittest.TestCase):
    def test_emitted_report_is_sm120_only_and_passes(self):
        p = _REPO_ROOT / "data/local_labels/cache_replay_sampled_fidelity.json"
        if not p.exists():
            self.skipTest("sampled fidelity artifact not emitted yet")
        rep = json.loads(p.read_text())
        self.assertEqual(rep["top_level_admission_scope"], "target_arch=sm_120")
        self.assertEqual(rep["arches"], ["sm_120"])
        g = rep["fidelity_gate"]
        self.assertLessEqual(g["max_sampled_record_fraction"], 0.10)
        self.assertGreaterEqual(g["n_within_abs_delta_bar"], 27)
        self.assertTrue(g["ac4_verdict_preserved"])
        self.assertTrue(g["passed"])


if __name__ == "__main__":
    unittest.main()
