# SOL-ExecBench Shrink-to-Fit Rubric — Fable-5 design (sm_86)

> Provenance: Fable 5 answer (2026-07-08) to `plans/sm86_sol_shrink_to_fit_fable5_prompt.md`. Verbatim design portion extracted from `results.md` (thinking-block stripped). Referenced by `plans/sm86_comprehensive_live_capture_campaign_plan.md`.

---

# Shrink-to-Fit Rubric for SOL-ExecBench OOM Cells on RTX 3090 Ti (sm_86)

Below: the hardware anchors and thresholds, the three cheap models (footprint, roofline, L2 working set), the per-cell decision procedure, per-family dim guidance with worked numbers, the regime-preservation detection rule, the provenance stamp, and a runnable pre-check. Everything is calibrated to GA102 specifics, not generic advice.

---

## 1. Fixed anchors and decision thresholds

| Quantity | Value | Use in rubric |
|---|---|---|
| SMs / max threads per SM | 84 / **1536** (sm_86, not 2048) | wave size: \(N_{conc} = 84 \times \text{CTAs/SM}\) |
| L2 | 6 MiB (12×512 KiB slices, hashed) | boundary checks |
| L2 hysteresis band | **4.5–9 MiB** (0.75×–1.5×) | "band" verdict — hashing/replacement make the exact 6 MiB edge unreliable |
| DRAM BW | 1008 GB/s | roofline denominator |
| Ridge points (FLOP/B) | fp16-TC/fp32acc **79**, fp16acc 159, bf16 79, tf32/fp32 ~40 | roofline class |
| Roofline classes | mem if \(AI < R/2\); compute if \(AI > 2R\); else **knee** | class-preservation check |
| Deep-regime exemption | both shapes with \(\lvert \log_2(AI/R)\rvert \ge 2\) (same sign) → AI-drift tolerance waived | avoids false failures deep in one regime |
| AI tolerance (near knee) | \(\lvert \Delta AI / AI_{nat} \rvert \le 25\%\) **and** same class | |
| L2 deep exemption | both working sets ≥ 4× 6 MiB → pass regardless of drift | |
| Memory budget | app-replay: predicted ≤ **0.9 × 22 GiB ≈ 20 GiB**; kernel-replay: subtract Σ(bytes written) | NCU kernel replay save/restores written buffers |
| Replay-mode switch | kernel writes > 4 GiB → **`--replay-mode application` mandatory** | |
| Wave floor | \(\text{waves}_{red} \ge \min(8, \text{waves}_{nat})\) full waves | keeps tail effect ≤ ~12% and scheduler regime |
| Causal tile mix | diagonal-tile fraction \(2T_q/S_{red} \le \min(3 \times 2T_q/S_{nat},\ 0.10)\) | |
| Shrink lattice | B: powers of 2; H: whole GQA groups; S: multiples of \(2T_q\); T (tokens): powers of 2; prefer shipped `workload.jsonl` values | reproducibility |

One NCU subtlety that simplifies the L2 model: NCU's default `--cache-control all` flushes L2 between replay passes, so cross-kernel producer→consumer L2 hits are absent from these captures anyway. Only **intra-kernel reuse distance** decides which side of 6 MiB you're on — that's what the working-set model below estimates. Also lock clocks (`--clock-control base`) so the roofline comparison is stable on a 450 W card.

**Rubric step 0 (before shrinking anything):** re-check whether the cell OOMs *natively* or only under kernel-replay's save buffer. A kernel with 19 GiB footprint that writes 10 GiB OOMs under kernel replay but captures fine natively with `--replay-mode application`. If that rescues the native shape, stamp `native_captured_via_replay_mode` and skip the shrink entirely.

---

## 2. The three cheap models (per cell, offline, <1 ms each)

**Footprint model \(F(\text{shape})\):** closed-form sum of per-tensor bytes. Distinguish *allocated* bytes (drives OOM — causal kernels usually still allocate the full \(S^2\) buffer) from *traversed* bytes (drives roofline — causal traffic is ~½). Also compute Σ(bytes written) for the replay-mode decision.

**Roofline position:** \(AI = \text{FLOPs}/\text{DRAM bytes}\) using per-tensor traversal counts \(k_t\) (a tensor read twice counts twice). Compare against the ridge of the *dtype/pipe actually used* (bf16 TC → 79 FLOP/B).

**L2 working set \(W\):** the smallest tensor region re-referenced with reuse distance shorter than one phase, times the number of concurrent sharers, as a **[lo, hi] pair** (optimistic sharing vs. pessimistic + write-stream slack). Same-side verdicts must hold for both bounds; disagreement → "band" flag. Family instantiations are in §4. \(N_{conc}\) (CTAs in flight) comes from one 30-second calibration capture per kernel at a tiny shape — block size and achieved occupancy don't change with problem shape, so it's reusable across all cells of that kernel.

Mapping to your companion taxonomy: every shrink in this rubric is `dataset_shrunk`-class (per-block behavior preserved, total footprint reduced), so all of them inherit the companion doc's prohibition on dropping an over-6-MiB working set under the boundary — that's constraint (c) below. The `grid_stride_footprint_preserving` variant is **definitionally unavailable** for these cells: you cannot preserve a >24 GiB footprint on a 24 GiB card. Stamp that in provenance rather than leaving the variant field empty.

---

## 3. The per-cell decision rubric

For each OOM cell, with native shape \(x_{nat}\):

1. **Replay check** (step 0 above). If native fits under application replay → done, no shrink.
2. **Compute native regime vector:** \(F_{nat}\), \(AI_{nat}\) + class, \(W_{nat}\) [lo,hi] + side, waves, causal diag fraction. This is the target to preserve.
3. **Enumerate candidates in family-specific axis order** (§4), preferring shipped `workload.jsonl` values, largest footprint first. Never touch the forbidden dims: **D (head dim), \(D_m\), \(d_{ff}\) ratio, E, top-k, dtype, mask type, GQA ratio, layout, tile config.**
4. **Filter by hard constraints:** fit ≤ 0.9 × budget; structure invariants; wave floor; family floors (§4).
5. **Filter by regime constraints:** (i) L2 same-side for both [lo,hi] bounds, with band flag and 4× deep exemption; (ii) roofline class preserved, with ±25% AI drift unless deep-regime exempt; (iii) causal tile-mix rule.
6. **Select the LARGEST surviving shape** (shrink-to-fit, not shrink-to-tiny). Tie-break: fewest dims changed, then B-only over H over S.
7. **Classify** per the detection rule (§5): all pass → `shape_adjusted`; exactly one regime flag → `reduced_representative`; otherwise typed `unavailable`.
8. **Dominance check:** if the chosen shape ≤ the largest already-shipped fitting shape of that kernel, the cell adds no information beyond an existing cell — capture it anyway but stamp `dominated_by_shipped_shape: true`.
9. **Post-capture validation** (from the NCU report you just took): measured `dram__bytes.sum` within ±30% of model; `lts__t_sector_hit_rate.pct` directionally consistent with the predicted L2 side; SOL memory-vs-SM ordering matches the predicted roofline class. Any mismatch → stamp `model_mismatch`, demote to `reduced_representative`, fix the model before trusting siblings.
10. **Optional bracketing:** if budget allows, capture one mid-size shipped shape on the same axis so you can show the metric is monotone/stable along the shrink direction — this substantially strengthens the "representative of native" claim.

---

## 4. Per-family guidance

### 4.1 Attention fwd/bwd that materializes \(S^2\) tensors (your 5.10 GiB dS case)

Models (per \((b,h)\), \(\gamma = 0.5\) if causal traffic, alloc still full):

$$
F = n_{alloc} B H S^2 b + c_{lin} B H S D b,\qquad
\text{FLOPs} = \{4\ \text{fwd},\ 10\ \text{bwd}\}\cdot B H S^2 D \gamma,\qquad
AI_\infty \approx \frac{\{4,10\}\,D}{n_{trav}\, b}
$$

Key property: **compute and traffic both scale as \(S^2\), so \(AI\) is independent of B, H, and S.** All three axes are roofline-safe for this family. The binding constraints are L2 and waves.

L2 working set: concurrent CTAs share per-head K/V panels: \(W \approx N_{bh,act}\cdot 2 S D b\), with \(N_{bh,act} = \min\!\big(BH,\ \lceil N_{conc}/(S/T_q)\rceil\big)\). Note the useful identity: while the grid is saturated (\(BH \cdot S/T_q \ge N_{conc}\)), \(W \approx N_{conc} T_q \cdot 2 D b\) — **independent of S**. For \(N_{conc}=168, T_q=128, D=128\), fp16: \(W \approx 11\) MiB (over-side); for \(D=64\): ~5.5 MiB (permanently in the band — flag it, native will be banded too).

So the corruption mode is *not* shrinking S; it's **cutting \(B\!\cdot\!H\) below saturation**, which caps \(N_{bh,act}\) at \(BH\) and lets \(W = BH \cdot 2SDb\) fall under 6 MiB — the exact silent L2-hit inflation you're worried about. The wave floor (waves ≥ 1 ⟺ \(BH \ge N_{conc}T_q/S\)) plus the 8-wave requirement automatically keeps you saturated. **Shrink order: B (halvings) → H (whole GQA groups, preserving \(H_q/H_{kv}\)) → S** with: \(S \ge \max(8T_q,\ 20D)\) (keeps the linear \(O(SD)\) traffic terms <10% so \(AI\) really is S-invariant), causal diag-tile rule, wave floor.

Worked (your dS example, reconstructed as B=4, H=40, S=4096, D=128, bf16, ~3 stored \(S^2\) tensors ⇒ ~15 GiB + linear terms; kernel replay would double-count ~10 GiB of writes → application replay): B→2 fits at ~8 GiB. \(W\): panels \(2SDb = 2\) MiB; tiles/head 32; \(N_{bh,act} = \min(80, \lceil 168/32\rceil) = 6\) → 12 MiB over-side, identical to native. AI unchanged. Waves \(= 2\cdot40\cdot32/168 = 15\). → `shape_adjusted`, B-only, factor 2.

### 4.2 Flash-style / \(O(SD)\)-memory attention (if any in-scope cell OOMs via huge \(BHSD\))

Here \(AI \propto S\) (compute \(S^2D\), traffic \(SDb\)) — **shrinking S flips the roofline** and is forbidden for `shape_adjusted` beyond a ±25% AI drift (i.e., barely at all). Safe axes: **B and H only.** If B=H=1 still doesn't fit, this family goes straight to the §5 detection rule.

### 4.3 MoE grouped-GEMM / expert-parallel

Shape \((T, E, k, D_m, d_{ff}, \lambda)\). Static weights \(n_{mat} E D_m d_{ff} b\) vs. dynamic activations \(\propto T\). Per-expert GEMM with \(tpe = Tk/E\) tokens/expert and \(h = D_m d_{ff}/(D_m + d_{ff})\):

$$
AI(tpe) = \frac{2h}{b}\cdot\frac{tpe}{tpe+h}
$$

**Safe axis: T (tokens) only.** Preserve E, k, \(D_m\), \(d_{ff}\), capacity factor, drop policy, and the routing histogram — regenerate routing at reduced T with the same distribution parameters and keep imbalance \(\lambda = \max/\text{mean}\) within ±10% (imbalance is a first-order determinant of grouped-GEMM tail behavior). Floors: \(tpe \ge 2 M_{tile}\) (else expert GEMMs become sub-tile and the kernel degenerates into tile-quantization/launch-tail measurement), and the token window:

$$
T_{min} = \frac{E}{k}\cdot\max\!\big(2M_{tile},\ 3h\ \text{[only if near-knee]}\big),\qquad
T_{max} = \frac{\text{budget} - F_{static}}{\text{bytes/token}}
$$

The deep-regime exemption usually rescues this family: e.g. E=32, k=2, \(D_m\)=4096, \(d_{ff}\)=14336 (h≈3186), native T=256k (tpe=16k, \(AI \approx 2645\)) reduced to T=64k (tpe=4k, \(AI \approx 1786\)): 33% AI drift, but both are \(>2^4\times\) the 79 ridge — still unambiguously tensor-throughput-bound → `shape_adjusted`. L2 is robust here: per-expert weight streams (\(D_m d_{ff} b \approx 112\) MiB) keep both shapes far over 6 MiB regardless of T.

**The unfixable MoE case:** if static weights alone exceed budget (e.g. gated 3-matrix, E=64 → 22.5 GiB), no token shrink helps. Cutting E changes the grouped-GEMM group count and routing semantics — that is never `shape_adjusted`. Policy: typed `unavailable::oom_shape_gated`, or an explicit E-cut stamped `reduced_representative` with `native_predictive: false` if you want a measurement point at all.

### 4.4 Decoder-block full-block (your 16 GiB \([B,H,S,S]\) case)

Composite: attention rules (§4.1) for the \(S^2\) part, plus GEMM AI ratios in tokens \(= BS\): projections \(h_{proj} \approx D_m/2\), MLP \(h_{mlp} = D_m d_{ff}/(D_m+d_{ff})\), each with \(AI \propto \text{tokens}/(\text{tokens}+h)\). **Shrink B first**; the hazard when tokens get small is weight-streaming traffic silently dragging the GEMM phases memory-bound. Floor: \(BS \ge 3h_{mlp}\) (≈ ±25% AI) unless the fused block's cycles are attention-dominated (use the aggregate-AI + dominant-phase check in the calculator).

Worked (16 GiB example: B=4, H=32, S=8192, D=128, fp16 → \(S^2\) tensor 16 GiB, total ≈ 18 GiB, OOM with context + replay): B→1 fits but tokens 8192 < \(3h_{mlp}\approx 9.6\)k (28% MLP-AI drift — marginal); **B→2 is the right answer** (shrink-to-fit picks the largest anyway): 9.3 GiB, tokens 16384 (16% drift), \(W \approx 11\) MiB over-side both shapes, AI ≈ 118 FLOP/B knee-class both (identical, since B cancels), waves 24. → `shape_adjusted`.

### 4.5 Fused LayerNorm/RMSNorm

\(AI \approx 1\text{–}3\) FLOP/B — deep memory-bound at any size (exemption applies). Shrink any leading/token dim; keep hidden width fixed (it sets per-CTA vector shape). Floors: total streamed ≥ 8× L2 (≥ 48 MiB, keeps the streaming hit-rate regime) and ≥ 8 waves. Essentially never problematic.

### Safe vs. silently-corrupting moves (summary)

| Move | Regime effect | Verdict |
|---|---|---|
| Shrink B (all families) | AI, per-CTA behavior, L2 panels unchanged while ≥ 8 waves | **Safe** with wave floor |
| Shrink H by whole GQA groups | same as B + saturation floor \(BH \ge N_{conc}T_q/S\) | **Safe** with floors |
| Shrink S, \(S^2\)-materializing attn | AI invariant; \(W\) S-invariant while saturated; diag-tile mix drifts | **Conditional** (floors: \(S \ge \max(8T_q, 20D)\), diag ≤ 3×, waves) |
| Shrink S, flash-style attn | \(AI \propto S\) → roofline flip | **Corrupting** beyond ±25% |
| Shrink T, MoE | \(AI \propto tpe/(tpe+h)\); deep-compute usually exempt | **Conditional** (tpe floors, λ preserved) |
| \(BH\) below saturation (attn) | K/V panel set drops under 6 MiB → **L2 hit-rate inflation** | **Corrupting — the #1 silent hazard** |
| Decoder tokens < \(3h\) | weight traffic flips GEMMs memory-bound | **Corrupting** |
| Shrink D, \(D_m\), \(d_{ff}\), E, top-k; change dtype/mask; sub-wave grid | tile/MMA shape, routing, ridge, scheduler regime all change | **Forbidden** for `shape_adjusted` |

---

## 5. Detection rule: "no fitting shape preserves the regime"

Let hard constraints \(\mathcal{H}\) = {fit, structure invariants, wave floor, family floors} and regime constraints \(\mathcal{R}\) = {L2 same-side (band/deep rules), roofline preserved (class + tolerance/exemption)}.

- \(\exists\) candidate passing \(\mathcal{H} \cup \mathcal{R}\) → **`shape_adjusted`**, pick largest. `native_predictive: true`.
- Candidates pass \(\mathcal{H}\) but every one violates **exactly one** of \(\mathcal{R}\) → **`reduced_representative`**: a valid measurement point in a *different, declared* regime. Pick the candidate minimizing regime distance \(\big|\log_2\frac{W_{red}}{W_{nat}}\big| + \big|\log_2\frac{AI_{red}}{AI_{nat}}\big|\). Stamp which check failed and `native_predictive: false`. These cells must be excluded from any cross-arch SOL aggregation against native-shape B200 numbers.
- Every fitting candidate violates **both** regime checks, or fitting requires breaking a forbidden dim (E-cut, D-cut, dtype), or the grid can't reach one full wave → **`unavailable::oom_shape_gated`**, recording the nearest-miss candidate and its violations. A number in a doubly-wrong regime is worse than no number.
- Band cases: if native or reduced lands in 4.5–9 MiB, the cell can remain `shape_adjusted` only if *both* shapes are banded with \(W\) within ±30% of each other; otherwise treat as an L2 violation.

Post-capture validation (§3 step 9) can retroactively demote a cell to `reduced_representative` — the model is cheap, so it must be falsifiable against the capture it gated.

---

## 6. Provenance stamp (per cell)

```json
{
  "cell_id": "decoder_block_fwd::shape_7",
  "gpu": {"name": "RTX 3090 Ti", "arch": "sm_86", "sms": 84, "l2_mib": 6,
          "dram_gbps": 1008, "vram_gib": 24},
  "native_shape": {"B": 4, "H": 32, "S": 8192, "D": 128, "dff": 16384, "dtype": "fp16"},
  "native_shape_ref": "workload.jsonl#L47",
  "native_footprint_gib": 18.1,
  "native_oom_cause": "materialized [B,H,S,S] = 16.0 GiB",
  "reduced_shape": {"B": 2, "H": 32, "S": 8192, "D": 128, "dff": 16384, "dtype": "fp16"},
  "shrink_vector": [{"dim": "B", "from": 4, "to": 2, "factor": 2.0}],
  "invariants_held": ["S","H","D","dff","dtype","causal","gqa_ratio","tile_config","top_k","E"],
  "footprint_model": {"id": "decoder_fullblock_v3", "predicted_gib": 9.3, "measured_peak_gib": 9.6},
  "replay": {"mode": "application", "written_gib": 8.2, "kernel_replay_viable": false},
  "l2_verdict": {"native_ws_mib": [11.0, 14.5], "reduced_ws_mib": [11.0, 14.5],
                 "side_native": "over", "side_reduced": "over",
                 "same_side": true, "band_flag": false, "deep_exempt": true},
  "roofline_verdict": {"pipe": "fp16_tc_fp32acc", "ridge": 79.4,
                       "ai_native": 118, "ai_reduced": 118,
                       "class_native": "knee", "class_reduced": "knee",
                       "delta_pct": 0.0, "deep_exempt": false, "preserved": true},
  "occupancy_verdict": {"ctas": 4096, "ctas_per_wave": 168, "waves": 24.4,
                        "waves_floor": 8, "bh_saturated": true, "pass": true},
  "structure_verdict": {"diag_frac_native": 0.031, "diag_frac_reduced": 0.031, "pass": true},
  "classification": "shape_adjusted",
  "native_predictive": true,
  "dominated_by_shipped_shape": false,
  "companion_variant": "dataset_shrunk",
  "grid_stride_variant_status": "unavailable_by_definition_footprint_exceeds_vram",
  "post_capture_validation": {"dram_bytes_model_ratio": 1.12, "lts_hit_pct": 38.4,
                              "sol_ordering_matches_predicted": true, "model_mismatch": false},
  "toolchain": {"ncu": "…", "driver": "…", "clock_control": "base", "cache_control": "all"}
}
```

For `reduced_representative` cells, `classification`, `native_predictive: false`, and a `regime_flags` array (e.g. `["l2_side_flipped"]`) are the load-bearing fields. For `unavailable`, stamp `nearest_miss_shape` and its violations.

---

## 7. Cheap pre-check calculator

```python
import math

HW = dict(sms=84, l2=6 * 2**20, ridge={"fp16_tc_fp32acc": 79.4, "fp16_tc_fp16acc": 158.7,
                                       "bf16_tc": 79.4, "tf32_tc": 39.7, "fp32": 39.7})
TH = dict(budget=22 * 2**30, fit_frac=0.90, band=(0.75, 1.5), ai_tol=0.25,
          deep_log2=2.0, l2_deep_mult=4.0, waves_min=8, diag_mult=3.0, diag_abs=0.10,
          replay_switch=4 * 2**30)

def l2_side(w):
    lo, hi = TH["band"][0] * HW["l2"], TH["band"][1] * HW["l2"]
    return "under" if w < lo else ("over" if w > hi else "band")

def roof(ai, ridge):
    d = math.log2(ai / ridge)
    return ("mem" if d < -1 else "compute" if d > 1 else "knee"), d

# ---- family models: return the full regime vector for one shape ----
def attn_mat(B, H, S, D, b=2, mode="bwd", causal=True, Tq=128, ctas_per_sm=2,
             n_alloc=None, n_trav=None):
    n_alloc = n_alloc or (3 if mode == "bwd" else 1)   # stored S^2 tensors (full alloc)
    n_trav  = n_trav  or (4 if mode == "bwd" else 2)   # S^2 traversals (traffic)
    g = 0.5 if causal else 1.0
    lin = (8 if mode == "bwd" else 4) * B * H * S * D * b
    F = n_alloc * B * H * S * S * b + lin
    dram = n_trav * B * H * S * S * b * g + lin
    flops = (10 if mode == "bwd" else 4) * B * H * S * S * D * g
    conc = HW["sms"] * ctas_per_sm
    tiles_bh = math.ceil(S / Tq)
    bh_act = min(B * H, math.ceil(conc / tiles_bh))
    w_lo = bh_act * 2 * S * D * b                      # shared K/V panels
    w_hi = w_lo + conc * Tq * D * b * 2                # + per-CTA tiles / write slack
    written = n_alloc * B * H * S * S * b + lin / 2
    return dict(F=F, ai=flops / dram, w=(w_lo, w_hi), waves=B * H * tiles_bh / conc,
                written=written, diag=(2 * Tq / S if causal else None),
                sat=(B * H >= conc * Tq / S))

def moe(T, E, k, Dm, dff, b=2, n_mat=3, Mtile=128, ctas_per_sm=2):
    h = Dm * dff / (Dm + dff); tpe = T * k / E
    Fw = n_mat * E * Dm * dff * b
    F = Fw + 2 * T * Dm * b + 2 * T * k * dff * b
    flops = 2 * n_mat * T * k * Dm * dff
    ai = (2 * h / b) * tpe / (tpe + h) * (n_mat / (n_mat))   # per-expert GEMM asymptote form
    w = (Dm * dff * b, 2 * Dm * dff * b)               # expert-weight stream >> L2 in practice
    conc = HW["sms"] * ctas_per_sm
    ctas = E * math.ceil(tpe / Mtile) * math.ceil(dff / 128)
    return dict(F=F, ai=ai, w=w, waves=ctas / conc, written=2 * T * k * dff * b,
                diag=None, sat=True, tpe=tpe, static_fits=Fw <= TH["fit_frac"] * TH["budget"])

def judge(nat, red, ridge):
    v = {"fit": red["F"] <= TH["fit_frac"] * TH["budget"],
         "replay": "application" if red["written"] > TH["replay_switch"] else "kernel"}
    # L2: both bounds must agree, same side as native; deep exemption
    deep = min(nat["w"]) >= TH["l2_deep_mult"] * HW["l2"] and \
           min(red["w"]) >= TH["l2_deep_mult"] * HW["l2"]
    sides_n, sides_r = {l2_side(x) for x in nat["w"]}, {l2_side(x) for x in red["w"]}
    v["l2_ok"] = deep or (len(sides_n) == 1 == len(sides_r) and sides_n == sides_r
                          and "band" not in sides_r)
    v["l2_band"] = ("band" in sides_r or "band" in sides_n) and not deep
    # roofline: deep exemption else class + tolerance
    (cn, dn), (cr, dr) = roof(nat["ai"], ridge), roof(red["ai"], ridge)
    deep_r = abs(dn) >= TH["deep_log2"] and abs(dr) >= TH["deep_log2"] and dn * dr > 0
    v["roof_ok"] = deep_r or (cn == cr and abs(red["ai"] / nat["ai"] - 1) <= TH["ai_tol"])
    v["waves_ok"] = red["waves"] >= min(TH["waves_min"], nat["waves"]) and red["sat"]
    if nat.get("diag"):
        v["diag_ok"] = red["diag"] <= min(TH["diag_mult"] * nat["diag"], TH["diag_abs"])
    hard = v["fit"] and v["waves_ok"] and v.get("diag_ok", True)
    regime_fails = [k for k, ok in (("l2", v["l2_ok"]), ("roofline", v["roof_ok"])) if not ok]
    v["class"] = ("shape_adjusted" if hard and not regime_fails else
                  "reduced_representative" if hard and len(regime_fails) == 1 else
                  "unavailable::oom_shape_gated")
    v["regime_flags"] = regime_fails
    return v

def choose(native_kwargs, lattice, model, ridge):
    nat = model(**native_kwargs)
    cands = sorted((dict(native_kwargs, **c) for c in lattice),
                   key=lambda c: model(**c)["F"], reverse=True)
    best_flagged = None
    for c in cands:
        red = model(**c); v = judge(nat, red, ridge)
        if v["class"] == "shape_adjusted":
            return c, red, v                       # largest fully-preserving shape
        if v["class"] == "reduced_representative" and best_flagged is None:
            best_flagged = (c, red, v)
    return best_flagged or (None, None, {"class": "unavailable::oom_shape_gated"})
```

Calibration inputs per kernel (one-time): `ctas_per_sm` and \(T_q\) from a single tiny-shape NCU capture; footprint-model constants (\(n_{alloc}, n_{trav}, c_{lin}\)) validated once against `cudaMemGetInfo` deltas at a small shape. After that, shape selection is pure arithmetic per cell.

---

## Closing cautions

The four failure modes that produce plausible-looking but wrong numbers, in rough order of likelihood on this suite: (1) attention B·H cut below CTA saturation, collapsing the concurrent K/V panel set under 6 MiB and inflating `lts__t_sector_hit_rate`; (2) flash-style S-shrink moving \(AI \propto S\) across the 79 FLOP/B ridge; (3) decoder/MoE token counts falling below \(\sim 3h\), flipping GEMM phases memory-bound via weight streaming; (4) forgetting that NCU kernel replay's save buffer re-OOMs or shrinks your budget — always compute the replay-mode branch before the shrink branch. The rubric's floors exist precisely to make each of these a checked, stamped verdict instead of a silent one.
