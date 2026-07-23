"""Tests for the SOL shrink-to-fit calculator (Fable-5 rubric §4 worked cases)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.local_labels import sol_shrink_calculator as sc  # noqa: E402

_SCHEMA = json.loads(
    (_REPO / "workloads" / "simbench_bw" / "schemas"
     / "sol_shrink_provenance.schema.json").read_text()
)
BF16 = sc.HW["ridge"]["bf16_tc"]


def test_l2_side_boundaries():
    assert sc.l2_side(0.5 * sc.HW["l2"]) == "under"
    assert sc.l2_side(2.0 * sc.HW["l2"]) == "over"
    assert sc.l2_side(1.0 * sc.HW["l2"]) == "band"


def test_roofline_classes():
    assert sc.roof(1.0, 79.4)[0] == "mem"
    assert sc.roof(79.4, 79.4)[0] == "knee"
    assert sc.roof(1000.0, 79.4)[0] == "compute"


def test_replay_mode_first_step0():
    # A shape that fits under app replay is captured native, not shrunk.
    assert sc.native_fits_under_app_replay(19 * 2 ** 30) is True
    assert sc.native_fits_under_app_replay(21 * 2 ** 30) is False


def test_attention_bh_below_saturation_is_caught_by_wave_floor():
    # The #1 silent hazard: cutting B*H below CTA saturation. The wave floor must
    # flag it rather than silently pass (native saturated, reduced not).
    nat = sc.attn_mat(B=4, H=40, S=4096, D=128, mode="bwd")
    starved = sc.attn_mat(B=1, H=1, S=4096, D=128, mode="bwd")
    v = sc.judge(nat, starved, BF16)
    assert v["waves_ok"] is False  # sub-saturation caught


def test_attention_B_halving_is_shape_adjusted():
    # §4.1 worked example: B=4->2 preserves AI (B cancels), L2 side, waves.
    native = dict(B=4, H=40, S=4096, D=128, mode="bwd")
    lattice = [dict(B=2), dict(B=1)]
    chosen, red, v = sc.choose(native, lattice, sc.attn_mat, BF16)
    assert v["classification"] == sc.CLASS_SHAPE_ADJUSTED
    assert chosen["B"] == 2  # shrink-to-fit picks the LARGEST fitting shape
    assert red["F"] <= sc.TH["fit_frac"] * sc.TH["budget"]


def test_moe_static_weights_over_budget_is_unavailable():
    # §4.3 unfixable MoE: static weights alone exceed budget; no token shrink helps.
    native = dict(T=256_000, E=64, k=2, Dm=4096, dff=14336, n_mat=3)
    nat = sc.moe(**native)
    assert nat["static_fits"] is False
    lattice = [dict(T=64_000), dict(T=16_000), dict(T=4_000)]
    chosen, red, v = sc.choose(native, lattice, sc.moe, BF16)
    assert v["classification"] == sc.CLASS_UNAVAILABLE


def test_provenance_stamp_shape_adjusted_schema_valid():
    native = dict(B=4, H=40, S=4096, D=128, mode="bwd")
    lattice = [dict(B=2)]
    chosen, red, v = sc.choose(native, lattice, sc.attn_mat, BF16)
    stamp = sc.shrink_provenance(
        cell_id="problem_067_flash_attention_gqa_ultralong::shape_0",
        family="attn_s2_materializing", native_kwargs=native,
        native_vector=sc.attn_mat(**native), chosen_kwargs=chosen,
        reduced_vector=red, verdict=v, ridge=BF16,
        native_shape_ref="workload.jsonl#L0")
    jsonschema.validate(stamp, _SCHEMA)
    assert stamp["classification"] == sc.CLASS_SHAPE_ADJUSTED
    assert stamp["native_predictive"] is False
    assert stamp["claim_scope"] == "reduced_shape_only"
    assert stamp["model"]["model_id"] == "sol_shrink_calculator.attn_mat.v1"
    assert stamp["static_floor"] == {
        "bytes": 0,
        "fit_limit_bytes": int(sc.TH["fit_frac"] * sc.TH["budget"]),
        "fits": True,
    }
    assert any(sv["dim"] == "B" for sv in stamp["shrink_vector"])


def test_provenance_stamp_unavailable_schema_valid():
    native = dict(T=256_000, E=64, k=2, Dm=4096, dff=14336, n_mat=3)
    chosen, red, v = sc.choose(native, [dict(T=4_000)], sc.moe, BF16)
    stamp = sc.shrink_provenance(
        cell_id="problem_044_moe::shape_0", family="moe_grouped_gemm",
        native_kwargs=native, native_vector=sc.moe(**native),
        chosen_kwargs=chosen, reduced_vector=red, verdict=v, ridge=BF16)
    jsonschema.validate(stamp, _SCHEMA)
    assert stamp["classification"] == sc.CLASS_UNAVAILABLE
    assert stamp["native_predictive"] is False


def test_mask_type_change_is_not_shape_adjusted():
    # native causal, reduced non-causal — a mask-type change is forbidden and must
    # not crash (red diag is None) nor pass as shape_adjusted.
    nat = sc.attn_mat(B=4, H=40, S=4096, D=128, mode="bwd", causal=True)
    red = sc.attn_mat(B=2, H=40, S=4096, D=128, mode="bwd", causal=False)
    v = sc.judge(nat, red, BF16)
    assert v["diag_ok"] is False
    assert v["classification"] != sc.CLASS_SHAPE_ADJUSTED


def test_bh_only_shrink_not_gated_by_absolute_diag_cap():
    # small S -> high diagonal fraction (2*128/512 = 0.5 > 0.10 abs cap); a B-only
    # shrink keeps S (so diag is unchanged) and must NOT be gated by the abs cap.
    nat = sc.attn_mat(B=8, H=40, S=512, D=64, mode="bwd", causal=True)
    red = sc.attn_mat(B=4, H=40, S=512, D=64, mode="bwd", causal=True)
    assert nat["diag"] > sc.TH["diag_abs"]
    v = sc.judge(nat, red, BF16)
    assert v["diag_ok"] is True


def test_unavailable_verdict_carries_nearest_miss_key():
    native = dict(T=256_000, E=64, k=2, Dm=4096, dff=14336, n_mat=3)
    _, _, v = sc.choose(native, [dict(T=4_000)], sc.moe, BF16)
    assert v["classification"] == sc.CLASS_UNAVAILABLE
    assert "nearest_miss_shape" in v


def test_native_replay_classification_in_schema():
    native = dict(B=1, H=8, S=128, D=64, mode="bwd")
    vector = sc.attn_mat(**native)
    stamp = sc.shrink_provenance(
        cell_id="x", family="attn", native_kwargs=native,
        native_vector=vector, chosen_kwargs=None, reduced_vector=None,
        verdict={"classification": sc.CLASS_NATIVE_REPLAY, "regime_flags": []},
        ridge=BF16)
    jsonschema.validate(stamp, _SCHEMA)
    assert stamp["native_predictive"] is False
    assert stamp["claim_scope"] == "reduced_shape_only"


def test_model_vectors_bind_static_floor_identity():
    attention = sc.attn_mat(B=2, H=8, S=512, D=64)
    assert attention["model_id"] == "sol_shrink_calculator.attn_mat.v1"
    assert attention["static_floor_bytes"] == 0
    assert attention["static_floor_fits"] is True

    moe = sc.moe(T=4096, E=8, k=2, Dm=1024, dff=4096)
    assert moe["model_id"] == "sol_shrink_calculator.moe.v1"
    assert moe["static_floor_bytes"] > 0
    assert moe["static_floor_fits"] == (
        moe["static_floor_bytes"] <= sc.TH["fit_frac"] * sc.TH["budget"])
