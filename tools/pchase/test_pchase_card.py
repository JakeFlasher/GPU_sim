"""P-chase static-card tests: committed-card integrity, the
value-vs-typed completeness contract, the counter-free guarantee, and
the step-detector's confidence rule on synthetic curves.
"""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import jsonschema  # noqa: E402

from run_pchase_card import (  # noqa: E402
    ANALYSIS_CONFIG,
    covariate_verdict,
    detect_step,
)

_CARD = REPO_ROOT / "data/pchase/sm_86_static_card.json"
_SCHEMA = (
    REPO_ROOT / "workloads/simbench_bw/schemas/pchase_static_card.schema.json"
)

# Profiler-counter namespaces the card must NEVER contain (the card is
# counter-free by construction; this is the artifact-level witness on
# top of the repo-wide static guard). The tokens are assembled from
# fragments so THIS file never contains them contiguously — the
# repo-wide guard scans this tree too.
_DUNDER = "_" + "_"
_COUNTER_TOKEN = re.compile(
    "|".join([
        "l1" + "tex" + _DUNDER,
        "sm" + "sp" + _DUNDER,
        "dr" + "am" + _DUNDER,
        "lt" + "s" + _DUNDER,
        "sm" + _DUNDER,
        "gp" + "c" + _DUNDER,
        "n" + "cu",
    ]), re.IGNORECASE)


class StepDetectorTest(unittest.TestCase):
    _CFG = {"step_min_jump": 0.20, "step_dominance": 2.0,
            "step_member_floor": 0.10}

    def test_sharp_step_resolves_with_bracket(self) -> None:
        curve = [(16, 33.0), (32, 33.0), (48, 33.1), (64, 33.0),
                 (96, 33.0), (112, 250.0), (128, 250.0)]
        step = detect_step(curve, self._CFG)
        self.assertTrue(step["resolved"])
        self.assertEqual(step["last_base_x"], 96)
        self.assertEqual(step["first_post_x"], 112)

    def test_smeared_transition_merges_adjacent_intervals(self) -> None:
        curve = [(1, 250.0), (2, 250.0), (5, 250.0), (6, 341.0),
                 (7, 503.0), (8, 503.0), (16, 503.0)]
        step = detect_step(curve, self._CFG)
        self.assertTrue(step["resolved"])
        self.assertEqual(step["last_base_x"], 5)
        self.assertEqual(step["first_post_x"], 7)
        self.assertEqual(step["transition_intervals"], 2)

    def test_ambiguous_curve_unresolved(self) -> None:
        # Two separated comparable jumps: no dominant transition.
        curve = [(1, 100.0), (2, 130.0), (3, 130.0), (4, 165.0),
                 (5, 165.0)]
        step = detect_step(curve, self._CFG)
        self.assertFalse(step["resolved"])

    def test_flat_curve_unresolved(self) -> None:
        curve = [(1, 100.0), (2, 100.5), (3, 100.2), (4, 100.4)]
        step = detect_step(curve, self._CFG)
        self.assertFalse(step["resolved"])


class CovariatePolicyTest(unittest.TestCase):
    def test_sw_thermal_recorded_not_rejected(self) -> None:
        verdict = covariate_verdict(
            {"sm_clock_mhz": 1900.0, "throttle_reasons": 0x20},
            ANALYSIS_CONFIG)
        self.assertTrue(verdict["accepted"])
        self.assertTrue(verdict["sw_thermal_active"])

    def test_hw_faults_rejected(self) -> None:
        for bit in (0x8, 0x40, 0x80, 0x4):
            verdict = covariate_verdict(
                {"sm_clock_mhz": 1900.0, "throttle_reasons": bit},
                ANALYSIS_CONFIG)
            self.assertFalse(verdict["accepted"], hex(bit))

    def test_unavailable_rejected(self) -> None:
        self.assertFalse(covariate_verdict(
            {"unavailable": True}, ANALYSIS_CONFIG)["accepted"])

    def test_idle_clock_below_floor_rejected(self) -> None:
        """The review-blocking arm: idle telemetry must be REJECTED by
        the active-clock floor, not merely annotated."""
        verdict = covariate_verdict(
            {"sm_clock_mhz": 399.0, "throttle_reasons": 0},
            ANALYSIS_CONFIG)
        self.assertFalse(verdict["accepted"])
        self.assertIn("idle_clock_below_floor", verdict["reason"])


class CardValidatorNegativeTest(unittest.TestCase):
    def _schema(self) -> dict:
        return json.loads(_SCHEMA.read_text(encoding="utf-8"))

    def _minimal_card(self, prop: dict) -> dict:
        return {
            "schema_version": 1,
            "artifact_kind": "pchase_static_card",
            "arch": "sm_86",
            "device": "d",
            "driver_version": "v",
            "build": {"nvcc_arch": "sm_86",
                      "source_sha256": "a" * 64,
                      "binary_sha256": "b" * 64},
            "analysis_config": {},
            "counter_free_note": "n",
            "nvml_policy": {"rejected_reason_mask": "0xcc",
                            "sw_thermal_treatment": "recorded",
                            "active_clock_floor_mhz": 400},
            "properties": {"p": prop},
            "summary": {"n_properties": 1,
                        "n_typed_geometry_unresolved": 0},
        }

    def test_value_without_dispersion_rejected(self) -> None:
        """The negative arm: a card value lacking dispersion/repeats
        must not validate."""
        bad = {"kind": "value", "value": 33.0, "unit": "cycles",
               "method": "m", "covariates": {}}
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(self._minimal_card(bad), self._schema())

    def test_value_without_repeats_rejected(self) -> None:
        bad = {"kind": "value", "value": 33.0, "unit": "cycles",
               "dispersion": 0.1, "method": "m", "covariates": {}}
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(self._minimal_card(bad), self._schema())

    def test_typed_requires_reason_and_detail(self) -> None:
        bad = {"kind": "typed", "method": "m"}
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(self._minimal_card(bad), self._schema())

    def test_per_point_kind_rejected(self) -> None:
        bad = {"kind": "per_point", "points": {}}
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(self._minimal_card(bad), self._schema())

    def test_evidence_only_kind_rejected(self) -> None:
        bad = {"kind": "evidence_only", "evidence": {}}
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(self._minimal_card(bad), self._schema())

    def test_complete_value_property_validates(self) -> None:
        good = {"kind": "value", "value": 33.0, "unit": "cycles",
                "dispersion": 0.0, "n_repeats": 5, "method": "m",
                "covariates": {"n_accepted": 5}}
        jsonschema.validate(self._minimal_card(good), self._schema())


class CommittedCardTest(unittest.TestCase):
    def setUp(self) -> None:
        if not _CARD.is_file():
            self.skipTest("no committed card")

    def _load(self) -> dict:
        return json.loads(_CARD.read_text(encoding="utf-8"))

    def test_schema_valid(self) -> None:
        jsonschema.validate(
            self._load(), json.loads(_SCHEMA.read_text(encoding="utf-8")))

    def test_every_property_value_or_typed(self) -> None:
        """STRICT contract: a top-level property is value or typed —
        nothing else (per_point/evidence_only were review-rejected
        contract widenings)."""
        card = self._load()
        for name, prop in card["properties"].items():
            kind = prop["kind"]
            self.assertIn(kind, ("value", "typed"),
                          f"{name} has non-contract kind {kind}")
            if kind == "value":
                for field in ("value", "dispersion", "n_repeats",
                              "method", "covariates"):
                    self.assertIn(field, prop, f"{name}.{field}")
            else:
                self.assertEqual(prop["typed_reason"],
                                 "geometry_unresolved")
                self.assertIn("evidence", prop, name)

    def test_counter_free_by_construction(self) -> None:
        """The negative arm: no profiler-counter namespace anywhere in
        the card (NCU-derived numbers are structurally impossible —
        the probe tree never invokes a profiler — and this witnesses
        the artifact side)."""
        text = _CARD.read_text(encoding="utf-8")
        self.assertIsNone(
            _COUNTER_TOKEN.search(text),
            "profiler-counter namespace found in the counter-free card")

    def test_summary_recounts(self) -> None:
        card = self._load()
        props = card["properties"]
        self.assertEqual(card["summary"]["n_properties"], len(props))
        self.assertEqual(
            card["summary"]["n_typed_geometry_unresolved"],
            sum(1 for p in props.values() if p["kind"] == "typed"))

    def test_live_identity_and_physics_sanity(self) -> None:
        card = self._load()
        self.assertIn("RTX", card["device"])
        props = card["properties"]
        l1 = props["l1_hit_latency_cycles"]["value"]
        l2 = props["l2_hit_latency_cycles"]["value"]
        dram = props["dram_latency_cycles"]["value"]
        shared = props["shared_latency_cycles"]["value"]
        # Ordering sanity only (no absolute pins — arch-coupled).
        self.assertLess(shared, l1)
        self.assertLess(l1, l2)
        self.assertLess(l2, dram)

    def test_carveout_capacity_monotone_nonincreasing(self) -> None:
        """More shared-memory carveout can never INCREASE the L1
        capacity; resolved per-pct properties must be non-increasing
        in pct."""
        card = self._load()
        resolved = []
        for name, prop in card["properties"].items():
            m = re.match(r"l1_capacity_carveout_(\d+)_pct_bytes", name)
            if m and prop["kind"] == "value":
                resolved.append((int(m.group(1)), prop["value"]))
        self.assertGreaterEqual(len(resolved), 2)
        resolved.sort()
        for (_, a), (_, b) in zip(resolved, resolved[1:]):
            self.assertGreaterEqual(a, b)

    def test_supplemental_evidence_is_not_a_property(self) -> None:
        card = self._load()
        self.assertIn("l1_set_mapping_grid",
                      card.get("supplemental_evidence", {}))
        self.assertNotIn("l1_set_mapping_evidence", card["properties"])


if __name__ == "__main__":
    unittest.main()
