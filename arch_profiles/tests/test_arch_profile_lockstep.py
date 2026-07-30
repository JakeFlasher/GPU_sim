"""ArchProfile lockstep guard (recreated 2026-07-06 for the pivot baseline).

The original 29-test lockstep suite was authored during the merge but never
committed (the 818-file test removal predated it landing). This recreation
pins the same invariant: **every populated `arch_profiles/cards/*.json` field
equals the value still hardcoded in live source**, so the ArchProfile layer
introduces zero behaviour change for the validated sm_86 lane, and the sm_120
card's deliberate corrections (measured sm_count=26) stay in lockstep with the
now-fixed source constants.

Run:  .venv/bin/python -m pytest arch_profiles/tests/ -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "workloads" / "simbench_bw"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "cache_predictor"))

from arch_profiles import (  # noqa: E402
    ArchContext,
    ArchProfile,
    ArchProfileError,
    load_profile,
    resolve_arch_id,
    resolve_build_arch,
    resolve_context,
)


def _card(name: str) -> dict:
    return json.loads(
        (REPO_ROOT / "arch_profiles" / "cards" / f"{name}.json").read_text()
    )


@pytest.fixture(scope="module")
def sm86():
    return _card("sm_86")


@pytest.fixture(scope="module")
def sm120():
    return _card("sm_120-laptop")


# ---------- sm_86: card == live source, byte-for-byte semantics ----------

class TestSm86SmCountLockstep:
    def test_cache_predictor_sm_count(self, sm86):
        import cache_predictor
        assert cache_predictor._SM_COUNT_BY_ARCH["sm_86"] == sm86["sm_count"] == 84

    def test_canonical_features_sm_count(self, sm86):
        import canonical_features
        assert canonical_features._SM_COUNT_BY_ARCH["sm_86"] == sm86["sm_count"]


class TestSm86GriddimAlphaLockstep:
    def test_alpha_matches_live_constant(self, sm86):
        import cache_predictor
        assert cache_predictor._GRIDDIM_PENALTY_ALPHA == pytest.approx(
            sm86["griddim_penalty_alpha"]
        )


class TestSm86BareLdgELockstep:
    def test_ca_default_matches_classifier_frozenset(self, sm86):
        import l1_cache_op_classifier as clf
        assert sm86["bare_ldg_e_default"] == "ca"
        assert "sm_86" in clf._BARE_LDG_E_DEFAULTS_TO_CA


class TestSm86L1GeometryLockstep:
    def test_card_matches_l1_static_card_table(self, sm86):
        table = json.loads(
            (REPO_ROOT / "tools/cache_predictor/l1_static_card_table.json").read_text()
        )
        row = next(e for e in table["entries"] if e["arch"] == "sm_86")
        g = sm86["l1_geometry"]
        assert row["l1tex_capacity_bytes"] == g["l1tex_bytes"]
        assert row["l1_ways"] == g["ways"]
        assert row["default_l1_bytes"] == g["default_l1_bytes"]
        assert row["carveout_max_shared_bytes"] == g["max_shared_bytes"]
        assert row["l1_sets_at_full_capacity"] == g["sets"]


class TestSm86L2GeometryLockstep:
    def test_card_matches_device_properties(self, sm86):
        props = json.loads(
            (REPO_ROOT / "tools/hrd_profiler/device_properties.json").read_text()
        )
        row = next(e for e in props["entries"] if e["arch"] == "sm_86")
        g = sm86["l2_geometry"]
        assert row["l2_capacity_bytes"] == g["capacity_bytes"] == 6291456
        assert row["sector_size_bytes"] == g["sector_bytes"] == 32
        assert row["num_l2_slices"] == g["num_slices"] == 6

    def test_line_bytes_is_the_global_invariant(self, sm86):
        # 128B line / 32B sector are GLOBAL physical invariants (merge dossier
        # section 4) -- the card must carry them but they are not per-arch.
        assert sm86["l2_geometry"]["line_bytes"] == 128


class TestSm86CacheLayers:
    def test_both_layers_declared(self, sm86):
        assert sm86["cache_layers"] == ["L1", "L2"]


class TestSm86LatenciesLockstep:
    """Plan §1D: card latencies == the pchase static-card evidence, exactly.

    The ArchProfile ``latencies_cycles`` block is what sim-calib consumes;
    the pchase card is the measurement evidence. Transcription drift between
    them would silently decouple the simulator from its measured inputs.
    """

    PCHASE = REPO_ROOT / "data/pchase/sm_86_static_card.json"
    KEYS = {
        "l1_hit": "l1_hit_latency_cycles",
        "l2_hit": "l2_hit_latency_cycles",
        "dram": "dram_latency_cycles",
        "shared": "shared_latency_cycles",
    }

    def test_block_present_and_matches_pchase_evidence(self, sm86):
        assert "latencies_cycles" in sm86, (
            "sm_86 card must carry latencies_cycles after 1D "
            "(tools/pchase/transcribe_latencies_to_arch_card.py)"
        )
        block = sm86["latencies_cycles"]
        props = json.loads(self.PCHASE.read_text())["properties"]
        for card_key, prop_key in self.KEYS.items():
            rec = props[prop_key]
            assert rec["kind"] == "value"
            assert block[card_key]["value"] == rec["value"], card_key
            assert block[card_key]["dispersion"] == rec.get("dispersion"), card_key

    def test_latency_ordering_sanity(self, sm86):
        b = sm86["latencies_cycles"]
        v = {k: b[k]["value"] for k in self.KEYS}
        assert v["shared"] < v["l1_hit"] < v["l2_hit"] < v["dram"]

    def test_mshr_never_a_bare_guess(self, sm86):
        mshr = sm86["latencies_cycles"].get("mshr_outstanding_misses")
        if mshr is None:
            pytest.skip("mshr key absent (pchase card carried no record)")
        assert ("value" in mshr) ^ (mshr.get("status") == "research_gap"), (
            "mshr must be either a measured value or a typed research_gap"
        )

    def test_provenance_names_evidence_and_driver(self, sm86):
        prov = sm86["latencies_cycles"]["provenance"]
        assert "data/pchase/sm_86_static_card.json" in prov
        driver = json.loads(self.PCHASE.read_text())["driver_version"]
        assert driver in prov


# ---------- sm_120: measured corrections in lockstep with fixed source ----------

class TestSm120SmCountLockstep:
    def test_card_carries_measured_26(self, sm120):
        assert sm120["sm_count"] == 26

    def test_cache_predictor_source_fixed_to_measured(self, sm120):
        import cache_predictor
        assert cache_predictor._SM_COUNT_BY_ARCH["sm_120"] == sm120["sm_count"] == 26

    def test_canonical_features_source_fixed_to_measured(self, sm120):
        import canonical_features
        assert canonical_features._SM_COUNT_BY_ARCH["sm_120"] == sm120["sm_count"]


class TestSm120FailClosed:
    def test_bare_ldg_e_unknown_and_not_in_frozenset(self, sm120):
        import l1_cache_op_classifier as clf
        assert sm120["bare_ldg_e_default"] == "unknown_typed_unavailable"
        assert "sm_120" not in clf._BARE_LDG_E_DEFAULTS_TO_CA

    def test_l2_only_cache_layers(self, sm120):
        assert sm120["cache_layers"] == ["L2"]

    def test_research_gap_fields_are_falsy_via_loader(self):
        p = load_profile("sm_120-laptop")
        assert not p.l1_geometry, "l1_geometry must fail closed (research_gap)"

    def test_griddim_alpha_uncalibrated_optout(self, sm120):
        assert sm120["griddim_penalty_alpha"] == 0.0


# ---------- issue-resource scaffold remains synthetic and fail-closed ----------

class TestIssueResourceScaffold:
    KEYS = {
        "smsp",
        "occupancy",
        "dep_barriers",
        "barrier_latencies",
        "dual_issue",
    }

    @staticmethod
    def _synthetic_context(sm86: dict) -> ArchContext:
        # Non-authoritative sentinels: these values exist only in this test object.
        raw = dict(sm86)
        raw.update(
            {
                "smsp": {"count": 3, "issue_ports": 5},
                "occupancy": {
                    "max_resident_warps_per_sm": 7,
                    "max_blocks": 11,
                },
                "dep_barriers": {"count": 13},
                "barrier_latencies": {
                    "read": 17,
                    "non_memory_write": 19,
                },
                "dual_issue": {"supported": True},
            }
        )
        profile = ArchProfile(arch_id="sm_86", sku="rtx_3090_ti", raw=raw)
        return ArchContext(
            arch_id="sm_86",
            sku="rtx_3090_ti",
            build_arch="sm_86",
            profile=profile,
        )

    def test_production_cards_remain_byte_authority_without_resource_blocks(
        self, sm86, sm120
    ):
        assert self.KEYS.isdisjoint(sm86)
        assert self.KEYS.isdisjoint(sm120)

    def test_absence_never_falls_back_across_architectures(self):
        for token, sku, build in (
            ("sm_86", "rtx_3090_ti", "sm_86"),
            ("sm_120", "rtx_5060_laptop", "sm_120"),
        ):
            context = resolve_context(token, sku=sku, build_arch=build)
            with pytest.raises(ArchProfileError, match="field absent"):
                context.require_issue_resources()

    def test_complete_synthetic_resources_round_trip_without_authority(self, sm86):
        context = self._synthetic_context(sm86)
        resources = context.require_issue_resources()
        assert set(resources) == self.KEYS
        assert resources["smsp"] == {"count": 3, "issue_ports": 5}
        assert resources["dual_issue"] == {"supported": True}

        context.profile.raw["smsp"]["count"] = 3.0
        context.profile.raw["barrier_latencies"]["read"] = 17.0
        resources = context.require_issue_resources()
        assert resources["smsp"]["count"] == 3
        assert isinstance(resources["smsp"]["count"], int)
        assert resources["barrier_latencies"]["read"] == 17
        assert isinstance(resources["barrier_latencies"]["read"], int)

    def test_direct_context_construction_cannot_launder_resource_provenance(
        self, sm86
    ):
        profile = self._synthetic_context(sm86).profile
        laundered = ArchContext(
            arch_id="sm_120",
            sku="rtx_5060_laptop",
            build_arch="sm_120",
            profile=profile,
        )
        with pytest.raises(ArchProfileError, match="identity mismatch"):
            laundered.require_issue_resources()

        contradictory_raw = dict(profile.raw)
        contradictory_raw["sku"] = "rtx_5060_laptop"
        contradictory_profile = ArchProfile(
            arch_id="sm_86",
            sku="rtx_5060_laptop",
            raw=contradictory_raw,
        )
        contradictory_pair = ArchContext(
            arch_id="sm_86",
            sku="rtx_5060_laptop",
            build_arch="sm_86",
            profile=contradictory_profile,
        )
        with pytest.raises(ArchProfileError, match="pair is not admitted"):
            contradictory_pair.require_issue_resources()

        wrong_build = ArchContext(
            arch_id="sm_86",
            sku="rtx_3090_ti",
            build_arch="sm_120",
            profile=profile,
        )
        with pytest.raises(ArchProfileError, match="not bound"):
            wrong_build.require_issue_resources()

    def test_nested_dual_issue_gap_and_bad_integer_fail_closed(self, sm86):
        context = self._synthetic_context(sm86)
        context.profile.raw["dual_issue"] = {
            "supported": {
                "status": "research_gap",
                "reason": "synthetic pairing gap",
            }
        }
        with pytest.raises(ArchProfileError, match="dual_issue.supported"):
            context.require_issue_resources()

        context = self._synthetic_context(sm86)
        context.profile.raw["dual_issue"] = {
            "supported": {
                "status": "research_gap",
                "reason": "synthetic pairing gap",
                "unknown": 1,
            }
        }
        with pytest.raises(ArchProfileError, match="malformed research_gap"):
            context.require_issue_resources()

        context = self._synthetic_context(sm86)
        context.profile.raw["smsp"] = {
            "status": "research_gap",
            "reason": "synthetic collision",
            "count": 3,
            "issue_ports": 5,
        }
        with pytest.raises(ArchProfileError, match="malformed research_gap"):
            context.require_issue_resources()

        context = self._synthetic_context(sm86)
        context.profile.raw["smsp"] = {"count": 3, "issue_ports": 1 << 32}
        with pytest.raises(ArchProfileError, match="positive bounded integer"):
            context.require_issue_resources()

        context = self._synthetic_context(sm86)
        context.profile.raw["barrier_latencies"]["read"] = float(
            (1 << 53) + 1
        )
        with pytest.raises(ArchProfileError, match="positive bounded integer"):
            context.require_issue_resources()

        context = self._synthetic_context(sm86)
        context.profile.raw["barrier_latencies"]["read"] = (1 << 53) - 1
        assert context.require_issue_resources()["barrier_latencies"]["read"] == (
            1 << 53
        ) - 1


# ---------- resolver: 8 historical spellings -> canonical ----------

class TestResolverCanonicalization:
    @pytest.mark.parametrize("token,expected", [
        ("sm_86", "sm_86"),
        ("sm_120", "sm_120"),
        ("sm_120-laptop", "sm_120"),
        ("NVIDIA GeForce RTX 5060 Laptop GPU", "sm_120"),
        ("NVIDIA GeForce RTX 3090 Ti", "sm_86"),
    ])
    def test_resolves(self, token, expected):
        assert resolve_arch_id(token) == expected

    def test_loader_loads_both_cards(self):
        assert load_profile("sm_86").sm_count == 84
        assert load_profile("sm_120-laptop").sm_count == 26

    def test_complete_context_requires_explicit_sku_and_build(self):
        ctx = resolve_context(
            "NVIDIA GeForce RTX 3090 Ti",
            sku="rtx_3090_ti",
            build_arch="sm_86",
        )
        assert (ctx.arch_id, ctx.sku, ctx.build_arch) == (
            "sm_86",
            "rtx_3090_ti",
            "sm_86",
        )
        with pytest.raises(ArchProfileError, match="explicit SKU"):
            resolve_context("sm_120", sku=None, build_arch="sm_120")
        with pytest.raises(ArchProfileError, match="build token"):
            resolve_context("sm_120a", sku="rtx_5060_laptop", build_arch="sm_120a")

    def test_compute_capability_cannot_select_a_sku_by_itself(self):
        ctx = resolve_context(
            (12, 0),
            sku="rtx_5060_laptop",
            build_arch="sm_120f",
        )
        assert ctx.profile.sm_count == 26
        with pytest.raises(ArchProfileError, match="belongs to architecture"):
            resolve_context((12, 0), sku="rtx_3090_ti", build_arch="sm_120")

    def test_build_tokens_are_validated_separately(self):
        assert resolve_build_arch("sm_120a", deploy_arch="sm_120") == "sm_120a"
        with pytest.raises(ArchProfileError, match="not in sm_86 family"):
            resolve_build_arch("sm_120a", deploy_arch="sm_86")
