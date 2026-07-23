"""Tests for the throughput benchmark gate and the sm_120-only claim-guard wiring.

Throughput: the gate passes if the replay clears the hundred-million-sector-per-
second bar OR is at least ten times the Fenwick stack-distance throughput; the
benchmark must report NVBit trace-collection time separately (the separation
check). Claim guard: the emitted comparison bundle is sm_120-only scoped and the
live guard admits a clean sm_120 surface while rejecting a synthetic cross-arch
bundle and a synthetic sm_86 live-evidence summary.
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
_TOOLS_LL = _REPO_ROOT / "tools" / "local_labels"
if str(_TOOLS_LL) not in sys.path:
    sys.path.insert(0, str(_TOOLS_LL))

import throughput_benchmark as tb  # noqa: E402

_BENCHMARK = _REPO_ROOT / "data/local_labels/cache_replay_benchmark.json"


class ThroughputGateTest(unittest.TestCase):
    def test_speedup_bar_passes(self):
        self.assertTrue(tb.gate_passed(5e6, 12.0))

    def test_hundred_million_bar_passes(self):
        self.assertTrue(tb.gate_passed(2e8, 1.0))

    def test_below_both_bars_fails(self):
        self.assertFalse(tb.gate_passed(5e6, 3.0))

    def test_separation_check_requires_separate_trace_time(self):
        good = {"nvbit_trace_collection_time": "reported_separately_not_in_replay_rate"}
        bad = {"nvbit_trace_collection_time": 1.23}  # folded numeric -> invalid
        self.assertTrue(tb.separation_ok(good))
        self.assertFalse(tb.separation_ok(bad))

    def test_emitted_benchmark_is_well_formed(self):
        if not _BENCHMARK.exists():
            self.skipTest("benchmark artifact not emitted yet")
        report = json.loads(_BENCHMARK.read_text())
        for field in ("replay_sectors_per_sec", "replay_wall_time_s",
                      "input_bytes_per_sec", "cpu_model", "source_commit",
                      "speedup_vs_fenwick", "nvbit_trace_collection_time"):
            self.assertIn(field, report)
        self.assertTrue(tb.separation_ok(report))
        self.assertEqual(
            report["passed"],
            tb.gate_passed(report["replay_sectors_per_sec"],
                           report["speedup_vs_fenwick"]),
        )


class ClaimGuardWiringTest(unittest.TestCase):
    def setUp(self):
        import sm120_loop_claim_guard as g
        self.g = g

    def test_emitted_comparison_bundle_is_sm120_only(self):
        p = _REPO_ROOT / "data/local_labels/cache_replay_comparison.json"
        if not p.exists():
            self.skipTest("comparison bundle not emitted yet")
        b = json.loads(p.read_text())
        self.assertEqual(b["top_level_admission_scope"], "target_arch=sm_120")
        self.assertEqual(b["arches"], ["sm_120"])

    def test_clean_sm120_comparison_bundle_admitted(self, ):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "comparison_bundle.json"
            p.write_text(json.dumps({
                "top_level_admission_scope": "target_arch=sm_120",
                "arches": ["sm_120"],
                "result_kind": "admitted",
                "included_cells": [{"join_key": {"arch": "sm_120", "cache_layer": "L2"},
                                    "evidence_kind": "live"}],
            }))
            defects = self.g.scan_paths([p])
            self.assertEqual(len(defects), 0, msg=str(defects))

    def test_cross_arch_scope_bundle_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "comparison_bundle.json"
            p.write_text(json.dumps({
                "top_level_admission_scope": "cross_arch_require_both_arches",
                "arches": ["sm_120", "sm_86"],
                "result_kind": "admitted",
            }))
            defects = self.g.scan_paths([p])
            self.assertGreater(len(defects), 0)

    def test_sm86_live_evidence_summary_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "round-9-summary.md"
            p.write_text(
                "# Summary\n\nEvidence status: Verified-live-sm_86 confirmed on "
                "the 3090 Ti with passing assertions.\n"
            )
            defects = self.g.scan_paths([p])
            self.assertGreater(len(defects), 0)


if __name__ == "__main__":
    unittest.main()
