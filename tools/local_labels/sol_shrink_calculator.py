#!/usr/bin/env python3
"""Shrink-to-fit calculator for host-capacity SOL-ExecBench cells on sm_86.

Offline, deterministic (< 1 ms/cell, no GPU) shape-selection for the SOL cells
whose native shape OOMs on the 24 GiB RTX 3090 Ti. Implements the GA102-calibrated
rubric: the footprint / roofline / L2-working-set models, the per-cell decision
procedure, and the typed classification — ``shape_adjusted`` (regime preserved),
``reduced_representative`` (exactly one regime axis flagged), or
``unavailable::oom_shape_gated`` (no fitting shape preserves the regime, or a
forbidden dim would have to move). The first check is replay-mode-first:
``--replay-mode application`` avoids NCU's kernel-replay save buffer, so a cell
that fits natively that way is stamped ``native_captured_via_replay_mode`` and NOT
shrunk.

The `#1 silent hazard` this encodes: cutting attention B·H below CTA saturation
collapses the concurrent K/V panel set under 6 MiB and inflates the L2 sector-hit
rate. The wave floor + saturation check make that a stamped verdict, never silent.

Rule 1: the occupancy/geometry inputs (CTAs-per-SM, Tq) are microbench/documented
GA102 geometry used ONLY to choose a capture-feasible shape; no NCU metric value
is an input. Post-capture `dram__bytes.sum` agreement (a validation, not an input)
lives in the capture round.
"""
from __future__ import annotations

import math

# GA102 / RTX 3090 Ti hardware anchors (sm_86).
HW = {
    "sms": 84,
    "l2": 6 * 2 ** 20,
    "ridge": {
        "fp16_tc_fp32acc": 79.4, "fp16_tc_fp16acc": 158.7,
        "bf16_tc": 79.4, "tf32_tc": 39.7, "fp32": 39.7,
    },
}
# Decision thresholds (Fable-5 rubric §1).
TH = {
    "budget": 22 * 2 ** 30, "fit_frac": 0.90, "band": (0.75, 1.5), "ai_tol": 0.25,
    "deep_log2": 2.0, "l2_deep_mult": 4.0, "waves_min": 8, "diag_mult": 3.0,
    "diag_abs": 0.10, "replay_switch": 4 * 2 ** 30,
}

CLASS_SHAPE_ADJUSTED = "shape_adjusted"
CLASS_REDUCED = "reduced_representative"
CLASS_UNAVAILABLE = "unavailable::oom_shape_gated"
CLASS_NATIVE_REPLAY = "native_captured_via_replay_mode"
CLAIM_REDUCED_ONLY = "reduced_shape_only"


def l2_side(w: float) -> str:
    lo, hi = TH["band"][0] * HW["l2"], TH["band"][1] * HW["l2"]
    return "under" if w < lo else ("over" if w > hi else "band")


def roof(ai: float, ridge: float) -> tuple[str, float]:
    d = math.log2(ai / ridge)
    return ("mem" if d < -1 else "compute" if d > 1 else "knee"), d


def native_fits_under_app_replay(footprint_bytes: float) -> bool:
    """Replay-mode-first check: does the native shape fit under
    ``--replay-mode application``? App replay avoids the kernel-replay save
    buffer, so only the live footprint must clear the budget. When true the cell
    is captured native, not shrunk."""
    return footprint_bytes <= TH["fit_frac"] * TH["budget"]


def attn_mat(B, H, S, D, b=2, mode="bwd", causal=True, Tq=128, ctas_per_sm=2,
             n_alloc=None, n_trav=None) -> dict:
    """Full regime vector for one attention shape that materializes S^2 tensors."""
    n_alloc = n_alloc if n_alloc is not None else (3 if mode == "bwd" else 1)
    n_trav = n_trav if n_trav is not None else (4 if mode == "bwd" else 2)
    g = 0.5 if causal else 1.0
    lin = (8 if mode == "bwd" else 4) * B * H * S * D * b
    F = n_alloc * B * H * S * S * b + lin
    dram = n_trav * B * H * S * S * b * g + lin
    flops = (10 if mode == "bwd" else 4) * B * H * S * S * D * g
    conc = HW["sms"] * ctas_per_sm
    tiles_bh = math.ceil(S / Tq)
    bh_act = min(B * H, math.ceil(conc / tiles_bh))
    w_lo = bh_act * 2 * S * D * b
    w_hi = w_lo + conc * Tq * D * b * 2
    written = n_alloc * B * H * S * S * b + lin / 2
    return {"F": F, "ai": flops / dram, "w": (w_lo, w_hi),
            "waves": B * H * tiles_bh / conc, "written": written,
            "diag": (2 * Tq / S if causal else None),
            "sat": (B * H >= conc * Tq / S),
            "model_id": "sol_shrink_calculator.attn_mat.v1",
            "static_floor_bytes": 0,
            "static_floor_fits": True}


def moe(T, E, k, Dm, dff, b=2, n_mat=3, Mtile=128, ctas_per_sm=2) -> dict:
    """Full regime vector for one MoE grouped-GEMM shape."""
    h = Dm * dff / (Dm + dff)
    tpe = T * k / E
    Fw = n_mat * E * Dm * dff * b
    F = Fw + 2 * T * Dm * b + 2 * T * k * dff * b
    ai = (2 * h / b) * tpe / (tpe + h)
    w = (Dm * dff * b, 2 * Dm * dff * b)
    conc = HW["sms"] * ctas_per_sm
    ctas = E * math.ceil(tpe / Mtile) * math.ceil(dff / 128)
    return {"F": F, "ai": ai, "w": w, "waves": ctas / conc,
            "written": 2 * T * k * dff * b, "diag": None, "sat": True,
            "tpe": tpe, "static_fits": Fw <= TH["fit_frac"] * TH["budget"],
            "model_id": "sol_shrink_calculator.moe.v1",
            "static_floor_bytes": Fw,
            "static_floor_fits": Fw <= TH["fit_frac"] * TH["budget"]}


def judge(nat: dict, red: dict, ridge: float) -> dict:
    """Classify a reduced shape against its native regime vector."""
    v = {"fit": red["F"] <= TH["fit_frac"] * TH["budget"],
         "replay": "application" if red["written"] > TH["replay_switch"] else "kernel"}
    deep = (min(nat["w"]) >= TH["l2_deep_mult"] * HW["l2"]
            and min(red["w"]) >= TH["l2_deep_mult"] * HW["l2"])
    sides_n = {l2_side(x) for x in nat["w"]}
    sides_r = {l2_side(x) for x in red["w"]}
    v["l2_ok"] = deep or (len(sides_n) == 1 == len(sides_r) and sides_n == sides_r
                          and "band" not in sides_r)
    v["l2_band"] = ("band" in sides_r or "band" in sides_n) and not deep
    (cn, dn) = roof(nat["ai"], ridge)
    (cr, dr) = roof(red["ai"], ridge)
    deep_r = abs(dn) >= TH["deep_log2"] and abs(dr) >= TH["deep_log2"] and dn * dr > 0
    v["roof_ok"] = deep_r or (cn == cr and abs(red["ai"] / nat["ai"] - 1) <= TH["ai_tol"])
    v["waves_ok"] = red["waves"] >= min(TH["waves_min"], nat["waves"]) and red["sat"]
    if nat.get("diag"):
        red_diag = red.get("diag")
        if red_diag is None:
            # native causal but reduced non-causal — a mask-type change (forbidden).
            v["diag_ok"] = False
        else:
            # Cap = min(3x native, 0.10), floored by the native fraction so a
            # same-S (B/H-only) shrink, where red_diag == nat_diag, is never
            # gated by the 0.10 absolute cap when nat_diag already exceeds it.
            cap = max(min(TH["diag_mult"] * nat["diag"], TH["diag_abs"]), nat["diag"])
            v["diag_ok"] = red_diag <= cap
    hard = v["fit"] and v["waves_ok"] and v.get("diag_ok", True)
    regime_fails = [name for name, ok in (("l2", v["l2_ok"]), ("roofline", v["roof_ok"]))
                    if not ok]
    v["classification"] = (CLASS_SHAPE_ADJUSTED if hard and not regime_fails else
                           CLASS_REDUCED if hard and len(regime_fails) == 1 else
                           CLASS_UNAVAILABLE)
    v["regime_flags"] = regime_fails
    return v


def _regime_distance(nat: dict, red: dict) -> float:
    """|log2(W_red/W_nat)| + |log2(AI_red/AI_nat)| — the rubric's regime-distance
    for picking the least-drifted reduced_representative (§5)."""
    w_nat = sum(nat["w"]) / 2
    w_red = sum(red["w"]) / 2
    dl2 = abs(math.log2(w_red / w_nat)) if w_nat and w_red else math.inf
    dai = abs(math.log2(red["ai"] / nat["ai"])) if nat["ai"] and red["ai"] else math.inf
    return dl2 + dai


def choose(native_kwargs: dict, lattice: list[dict], model, ridge: float):
    """Pick the LARGEST regime-preserving fitting shape (shrink-to-fit).

    Returns ``(chosen_kwargs, reduced_vector, verdict)``; the fully-preserving
    shape if one exists, else — among ALL reduced_representative candidates — the
    one MINIMIZING regime distance (not merely the first/largest), else
    ``(None, None, {classification: unavailable, nearest_miss_shape})`` recording
    the least-drifted fitting-but-doubly-violating candidate actually seen."""
    nat = model(**native_kwargs)
    cands = sorted((dict(native_kwargs, **c) for c in lattice),
                   key=lambda c: model(**c)["F"], reverse=True)
    reduced_options: list[tuple] = []
    nearest_miss: tuple | None = None  # (distance, kwargs, red, verdict)
    for c in cands:
        red = model(**c)
        v = judge(nat, red, ridge)
        cls = v["classification"]
        if cls == CLASS_SHAPE_ADJUSTED:
            return c, red, v  # largest fully-preserving shape
        if cls == CLASS_REDUCED:
            reduced_options.append((c, red, v))
        elif v.get("fit"):  # fits but violates both regime axes — a nearest miss
            d = _regime_distance(nat, red)
            if nearest_miss is None or d < nearest_miss[0]:
                nearest_miss = (d, c, red, v)
    if reduced_options:
        return min(reduced_options, key=lambda t: _regime_distance(nat, t[1]))
    verdict = {"classification": CLASS_UNAVAILABLE, "regime_flags": ["l2", "roofline"],
               "nearest_miss_shape": None}
    if nearest_miss is not None:
        _, c, _red, v = nearest_miss
        verdict["nearest_miss_shape"] = c
        verdict["regime_flags"] = v.get("regime_flags", ["l2", "roofline"])
    return None, None, verdict


def shrink_provenance(*, cell_id: str, family: str, native_kwargs: dict,
                      native_vector: dict, chosen_kwargs: dict | None,
                      reduced_vector: dict | None, verdict: dict, ridge: float,
                      native_shape_ref: str | None = None) -> dict:
    """Build the per-cell shrink provenance stamp (Fable-5 §6). ``native_predictive``
    is deliberately false for every reduced proxy: even a geometry-preserving
    ``shape_adjusted`` proxy is evidence only for its reduced shape and must not
    be relabeled as native cycles. A ``reduced_representative`` cell carries its
    ``regime_flags``; an ``unavailable`` cell records the nearest miss."""
    classification = verdict.get("classification", CLASS_UNAVAILABLE)
    model_id = native_vector.get("model_id")
    static_floor = native_vector.get("static_floor_bytes")
    if not isinstance(model_id, str) or not isinstance(static_floor, (int, float)):
        raise ValueError("model vector lacks model/static-floor identity")
    stamp = {
        "cell_id": cell_id,
        "gpu": {"name": "RTX 3090 Ti", "arch": "sm_86", "sms": HW["sms"],
                "l2_mib": HW["l2"] // 2 ** 20, "vram_gib": 24},
        "family": family,
        "model": {
            "model_id": model_id,
            "ridge_flop_per_byte": ridge,
        },
        "native_shape": native_kwargs,
        "native_shape_ref": native_shape_ref,
        "native_footprint_gib": round(native_vector["F"] / 2 ** 30, 3),
        "classification": classification,
        "native_predictive": False,
        "claim_scope": CLAIM_REDUCED_ONLY,
        "regime_flags": verdict.get("regime_flags", []),
        "ridge_flop_per_byte": ridge,
        "static_floor": {
            "bytes": int(static_floor),
            "fit_limit_bytes": int(TH["fit_frac"] * TH["budget"]),
            "fits": bool(native_vector.get("static_floor_fits", True)),
        },
    }
    if chosen_kwargs is not None and reduced_vector is not None:
        shrink_vector = [
            {"dim": d, "from": native_kwargs[d], "to": chosen_kwargs[d],
             "factor": (native_kwargs[d] / chosen_kwargs[d]) if chosen_kwargs[d] else None}
            for d in native_kwargs if chosen_kwargs.get(d) != native_kwargs.get(d)
        ]
        stamp.update(
            reduced_shape=chosen_kwargs,
            reduced_footprint_gib=round(reduced_vector["F"] / 2 ** 30, 3),
            shrink_vector=shrink_vector,
            replay_mode=verdict.get("replay"),
            l2_verdict={"native_ws_mib": [round(x / 2 ** 20, 2) for x in native_vector["w"]],
                        "reduced_ws_mib": [round(x / 2 ** 20, 2) for x in reduced_vector["w"]],
                        "same_side": verdict.get("l2_ok"), "band_flag": verdict.get("l2_band")},
            roofline_verdict={"ai_native": round(native_vector["ai"], 2),
                              "ai_reduced": round(reduced_vector["ai"], 2),
                              "preserved": verdict.get("roof_ok")},
            occupancy_verdict={"waves_reduced": round(reduced_vector["waves"], 2),
                               "waves_floor": TH["waves_min"],
                               "saturated": reduced_vector.get("sat"),
                               "pass": verdict.get("waves_ok")},
        )
    else:
        stamp["nearest_miss"] = None
    return stamp
