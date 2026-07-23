"""Lockstep-validator tests: closed-vocabulary mapping, the HARD
non-vacuity floor (vacuity guard), weak-join rejection, threshold-MISS
honesty, sealed-holdout one-touch + retune governance, and
committed-artifact integrity.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import jsonschema  # noqa: E402

import holdout_touch_ledger as htl  # noqa: E402
import lockstep_validator as lv  # noqa: E402

_ARTIFACT = REPO_ROOT / "data/local_labels/t1_lockstep_verdict_sm_86.json"
_SCHEMA = (
    REPO_ROOT / "workloads/simbench_bw/schemas/t1_lockstep_verdict.schema.json"
)
_LEDGER = REPO_ROOT / "data/local_labels/holdout_touch_ledger_sm_86.json"

_IDENT = {
    "binary_sha256": "a" * 64,
    "kernel_signature_sha256": "b" * 64,
}


def _cell(slug, metrics, status="emitted", shape="s", variant=None,
          typed_reason=None):
    ident = {"arch": "sm_86", "slug": slug, "shape_point": shape,
             **_IDENT}
    if variant:
        ident["variant_id"] = variant
    rec = {"identity": ident}
    if status == "emitted":
        rec["cell_status"] = {"status": "emitted"}
        rec["metrics"] = metrics
    else:
        rec["cell_status"] = {"status": "typed",
                              "typed_reason": typed_reason}
    return rec


def _label(slug, values, shape="s", variant=None):
    cell = {"arch": "sm_86", "slug": slug, "shape_point": shape, **_IDENT,
            "unit_tagged_metrics": [
                {"ncu_metric": m, "unit": "count", "value": v}
                for m, v in values.items()
            ]}
    if variant:
        cell["variant_id"] = variant
    return cell


_LD = lv._M_LD
_ST = lv._M_ST
_ATOM = lv._M_ATOM
_RED = lv._M_RED
_INST = "smsp__inst_executed.sum"
_SEC_LD = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
_SEC_ST = "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"
_ROLLUP = "l1tex__t_requests.sum"

_FLOOR_METRICS = (_LD, _ST, _ATOM, _RED, _SEC_LD, _SEC_ST, _INST, _ROLLUP)


def _floor_satisfying_corpus():
    """Six slugs (incl. the three vector-width family slugs), every
    confirmed metric scored on every cell — comfortably above the
    floor."""
    slugs = ["global_scalar_32b_copy", "global_vectorized_64b_copy",
             "global_vectorized_128b_copy", "slug_d", "slug_e", "slug_f"]
    emissions = {"cells": []}
    labels = {}
    for slug in slugs:
        metrics = {m: {"status": "scored", "value": 100}
                   for m in _FLOOR_METRICS}
        emissions["cells"].append(_cell(slug, metrics))
        labels[(slug, "s", "")] = _label(
            slug, {m: 100.0 for m in _FLOOR_METRICS})
    return emissions, labels


def _build(emissions, labels, holdout=frozenset(), sealed=None,
           excluded=None, config=None):
    return lv.build_verdicts(
        emissions, labels, holdout, sealed or {}, excluded or {},
        tuple(sorted(set(_FLOOR_METRICS))), config=config,
    )


class VocabularyMappingTest(unittest.TestCase):
    def test_exact_agreement_is_replacement_pass(self) -> None:
        emissions, labels = _floor_satisfying_corpus()
        body = _build(emissions, labels)
        self.assertTrue(body["non_vacuity_floor"]["floor_met"])
        passes = [r for r in body["verdicts"]
                  if r["verdict"] == "replacement_pass"]
        self.assertEqual(len(passes), len(body["verdicts"]))
        self.assertEqual(
            body["statistics"]["achieved_vs_target"]["met"],
            {"p50": True, "p90": True, "max": True})

    def test_threshold_miss_is_honest_not_upgraded(self) -> None:
        emissions, labels = _floor_satisfying_corpus()
        # Perturb one produced value far beyond the pair band.
        emissions["cells"][0]["metrics"][_LD] = {
            "status": "scored", "value": 150}
        body = _build(emissions, labels)
        row = next(r for r in body["verdicts"]
                   if r["slug"] == "global_scalar_32b_copy"
                   and r["ncu_metric"] == _LD)
        self.assertEqual(row["verdict"], "replacement_miss")
        self.assertAlmostEqual(row["relative_error"], 0.5)
        self.assertFalse(
            body["statistics"]["achieved_vs_target"]["met"]["max"])

    def test_producer_cell_typed_maps_to_support_with_detail(self) -> None:
        emissions, labels = _floor_satisfying_corpus()
        emissions["cells"].append(_cell(
            "stale_slug", {}, status="typed",
            typed_reason="kernel_signature_mismatch"))
        body = _build(emissions, labels)
        rows = [r for r in body["verdicts"] if r["slug"] == "stale_slug"]
        self.assertEqual(len(rows), len(set(_FLOOR_METRICS)))
        for r in rows:
            self.assertEqual(r["verdict"], "trace_support_insufficient")
            self.assertIn("kernel_signature_mismatch", r["detail"])
        self.assertEqual(
            body["coverage_report"]["cell_level_producer_reasons"][
                "kernel_signature_mismatch"], 1)

    def test_producer_metric_typed_reasons_map_closed(self) -> None:
        emissions, labels = _floor_satisfying_corpus()
        emissions["cells"][0]["metrics"][_ROLLUP] = {
            "status": "typed", "typed_reason": "rollup_unsupported"}
        emissions["cells"][0]["metrics"][_SEC_LD] = {
            "status": "typed",
            "typed_reason": "sector_scope_misaligned_footprint"}
        body = _build(emissions, labels)
        by = {(r["slug"], r["ncu_metric"]): r for r in body["verdicts"]}
        self.assertEqual(
            by[("global_scalar_32b_copy", _ROLLUP)]["verdict"],
            "rollup_unsupported")
        self.assertEqual(
            by[("global_scalar_32b_copy", _SEC_LD)]["verdict"],
            "contract_mismatch")

    def test_label_gaps_map_to_unstable_or_unavailable(self) -> None:
        emissions, labels = _floor_satisfying_corpus()
        # cell g: emitted counts but no label entry at all
        emissions["cells"].append(_cell(
            "slug_g", {m: {"status": "scored", "value": 5}
                       for m in _FLOOR_METRICS}))
        labels[("slug_g", "s", "")] = _label("slug_g", {})
        # cell h: same, but documented capture-unstable
        emissions["cells"].append(_cell(
            "slug_h", {m: {"status": "scored", "value": 5}
                       for m in _FLOOR_METRICS}))
        labels[("slug_h", "s", "")] = _label("slug_h", {})
        excluded = {("slug_h", "s", ""): {"typed_reason":
                                          "count_label_capture_unstable"}}
        body = _build(emissions, labels, excluded=excluded)
        by = {(r["slug"], r["ncu_metric"]): r for r in body["verdicts"]}
        self.assertEqual(by[("slug_g", _LD)]["verdict"],
                         "label_unavailable")
        self.assertEqual(by[("slug_h", _LD)]["verdict"],
                         "label_count_unstable")

    def test_low_support_for_tiny_label_mismatch(self) -> None:
        emissions, labels = _floor_satisfying_corpus()
        emissions["cells"][0]["metrics"][_ATOM] = {
            "status": "scored", "value": 3}
        labels[("global_scalar_32b_copy", "s", "")] = _label(
            "global_scalar_32b_copy",
            {**{m: 100.0 for m in _FLOOR_METRICS}, _ATOM: 2.0})
        body = _build(emissions, labels)
        row = next(r for r in body["verdicts"]
                   if r["slug"] == "global_scalar_32b_copy"
                   and r["ncu_metric"] == _ATOM)
        self.assertEqual(row["verdict"], "low_support")


class NonVacuityFloorTest(unittest.TestCase):
    def test_below_floor_run_claims_no_replacement_pass(self) -> None:
        """The vacuity guard: 1 slug / few pairs -> floor unmet -> every
        would-be pass demoted."""
        metrics = {m: {"status": "scored", "value": 100}
                   for m in (_LD, _ST)}
        emissions = {"cells": [_cell("only_slug", metrics)]}
        labels = {("only_slug", "s", ""): _label(
            "only_slug", {_LD: 100.0, _ST: 100.0})}
        body = _build(emissions, labels)
        self.assertFalse(body["non_vacuity_floor"]["floor_met"])
        self.assertEqual(
            body["verdict_counts"].get("replacement_pass", 0), 0)
        demoted = [r for r in body["verdicts"]
                   if r["verdict"] == "low_support"
                   and "vacuity_floor_unmet" in r.get("detail", "")]
        self.assertEqual(len(demoted), 2)

    def test_typed_exclusions_do_not_count_toward_floor(self) -> None:
        emissions, labels = _floor_satisfying_corpus()
        # Replace every scored emission with a producer-typed one.
        for rec in emissions["cells"]:
            rec["metrics"] = {
                m: {"status": "typed",
                    "typed_reason": "atom_red_sass_unresolved"}
                for m in _FLOOR_METRICS}
        body = _build(emissions, labels)
        self.assertEqual(body["non_vacuity_floor"]["scored_pairs"], 0)
        self.assertFalse(body["non_vacuity_floor"]["floor_met"])


class WeakJoinTest(unittest.TestCase):
    def test_weaker_identity_join_rejected(self) -> None:
        with self.assertRaises(lv.WeakIdentityJoinError):
            lv._identity_equal(
                {"slug": "a", "shape_point": "s",
                 "binary_sha256": "x" * 64},  # ksig missing
                {"slug": "a", "shape_point": "s",
                 "binary_sha256": "x" * 64,
                 "kernel_signature_sha256": "y" * 64})

    def test_label_identity_drift_is_typed_not_joined(self) -> None:
        emissions, labels = _floor_satisfying_corpus()
        drifted = labels[("slug_d", "s", "")]
        drifted["binary_sha256"] = "f" * 64
        body = _build(emissions, labels)
        rows = [r for r in body["verdicts"] if r["slug"] == "slug_d"]
        for r in rows:
            self.assertEqual(r["verdict"], "trace_support_insufficient")
        self.assertEqual(
            body["coverage_report"]["cell_level_producer_reasons"][
                "emission_label_identity_drift"], 1)


class HoldoutGovernanceTest(unittest.TestCase):
    def _tmp_ledger(self) -> Path:
        td = tempfile.mkdtemp()
        p = Path(td) / "ledger.json"
        p.write_text(json.dumps(htl.empty_ledger()) + "\n",
                     encoding="utf-8")
        return p

    def test_second_touch_rejected(self) -> None:
        p = self._tmp_ledger()
        htl.record_touch(p, sealed_digest="d" * 64, scope="scope",
                         scored_artifact="a.json",
                         config_digest=lv.config_digest(),
                         source_commit="c" * 40)
        with self.assertRaises(htl.HoldoutTouchRepeatError):
            htl.record_touch(p, sealed_digest="d" * 64, scope="scope",
                             scored_artifact="a.json",
                             config_digest=lv.config_digest(),
                             source_commit="c" * 40)

    def test_retune_after_touch_rejected_for_real_config(self) -> None:
        """The negative arm against the REAL validator config: after
        the touch, any config-digest change is rejected until a fresh
        re-seal."""
        p = self._tmp_ledger()
        htl.record_touch(p, sealed_digest="d" * 64, scope="scope",
                         scored_artifact="a.json",
                         config_digest=lv.config_digest(),
                         source_commit="c" * 40)
        ledger = htl.load_ledger(p)
        # Same config: legal.
        htl.assert_retune_legal(
            ledger, sealed_digest="d" * 64, scope="scope",
            config_digest=lv.config_digest())
        # A perturbed config (e.g. widened pass band): rejected.
        perturbed = dict(lv.VALIDATOR_CONFIG,
                         pair_pass_max_relative_error=0.5)
        import hashlib
        bad_digest = hashlib.sha256(
            json.dumps(perturbed, sort_keys=True,
                       separators=(",", ":")).encode()).hexdigest()
        with self.assertRaises(htl.HoldoutRetuneAfterTouchError):
            htl.assert_retune_legal(
                ledger, sealed_digest="d" * 64, scope="scope",
                config_digest=bad_digest)


class CommittedArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        if not _ARTIFACT.is_file():
            self.skipTest("no committed verdict artifact")

    def _load(self) -> dict:
        return json.loads(_ARTIFACT.read_text(encoding="utf-8"))

    def test_schema_valid(self) -> None:
        jsonschema.validate(
            self._load(),
            json.loads(_SCHEMA.read_text(encoding="utf-8")))

    def test_floor_met_and_complete_accounting(self) -> None:
        doc = self._load()
        self.assertTrue(doc["non_vacuity_floor"]["floor_met"])
        # Every admitted (cell, metric) pair has exactly one verdict.
        admitted = json.loads(
            (REPO_ROOT / "data/local_labels/label_admitted.json")
            .read_text(encoding="utf-8"))
        n_cells = sum(1 for c in admitted["cells"]
                      if c.get("arch") == "sm_86")
        from capture_count_labels import count_metric_identifiers
        self.assertEqual(len(doc["verdicts"]),
                         n_cells * len(count_metric_identifiers()))
        self.assertEqual(sum(doc["verdict_counts"].values()),
                         len(doc["verdicts"]))

    def test_no_miss_upgraded_to_pass(self) -> None:
        """Recompute every scored row's verdict from its recorded
        produced/label values; any silently-upgraded MISS fails."""
        doc = self._load()
        cfg = doc["validator_config"]
        for row in doc["verdicts"]:
            if row["verdict"] not in ("replacement_pass",
                                      "replacement_miss"):
                continue
            rel = lv.relative_error(
                row["produced"], row["label"],
                cfg["relative_error_denominator_floor"])
            self.assertAlmostEqual(rel, row["relative_error"])
            expected = ("replacement_pass"
                        if rel <= cfg["pair_pass_max_relative_error"]
                        else "replacement_miss")
            if expected == "replacement_miss":
                self.assertEqual(row["verdict"], "replacement_miss")

    def test_one_touch_recorded_in_committed_ledger(self) -> None:
        doc = self._load()
        self.assertEqual(
            doc["holdout_governance"]["holdout_scoring_touches"], 1)
        ledger = json.loads(_LEDGER.read_text(encoding="utf-8"))
        touches = [
            t for t in ledger.get("touches", [])
            if t["sealed_digest"]
            == doc["holdout_governance"]["sealed_digest"]
            and t["scope"] == doc["holdout_governance"]["scope"]
        ]
        self.assertEqual(len(touches), 1)
        self.assertEqual(touches[0]["config_digest"],
                         doc["validator_config_digest"])

    def test_config_digest_matches_frozen_config(self) -> None:
        doc = self._load()
        self.assertEqual(doc["validator_config_digest"],
                         lv.config_digest())
        self.assertEqual(doc["validator_config"], lv.VALIDATOR_CONFIG)


if __name__ == "__main__":
    unittest.main()
