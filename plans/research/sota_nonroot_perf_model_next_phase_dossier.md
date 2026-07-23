# RESEARCH DOSSIER — Accurate + Fast Non-Root GPU Performance Modeling, Next Phase (NCU Replacement: Wall-Clock / L1 / L2, sm_86 + sm_120)

Repo: `/home/jakeshea/3090ti_5060laptop_cross_study/3090ti_5060laptop_cross_study/non_root_gpu_profiler_codex`
Date: 2026-06-04 · Branch: `cleanup/pre-merge-simplification` (HEAD `b37bc7d5`)
Scope: seeds a `/humanize:gen-plan` implementation plan for the phase AFTER
`plans/sm86_multimetric_ncu_replacement_plan.md` (42-round loop, complete) and
`plans/sm86_ncu_replacement_continuation_plan.md` (AC-1 executed; loop complete).

> **Research provenance.** This dossier synthesizes (a) a 13-agent research
> workflow (4 repo-state analysts, 5 local-PDF deep-readers over the curated
> 64-paper corpus, 4 web researchers covering 2023–2026 literature), (b) an
> independent `codex exec` deep-research report (read-only, web-enabled), and
> (c) direct session verification of the load-bearing repo artifacts. Facts
> marked **[verified-this-session]** were re-checked directly against the
> artifact bytes in this session; other repo numbers were extracted by the
> repo-state analysts with `file:line` citations quoted from the artifacts.

---

## 1. Executive Summary

The two completed plans built the **corpus/ontology spine** and ran the
**pivotal controlled refit**. The honest post-loop position is bimodal:

- **L2 sector hit rate is a real, competitive, deployable result**: the frozen
  v5 isotonic-on-walker head scores full-cohort geomean relative error
  **0.0642** (gate ≤0.15 PASS), median abs **0.0592** (≤0.06 PASS), sealed
  holdout geomean **0.0552** (≤0.25 PASS), and beats a clean-room
  Nugteren-HPCA-2014 reimplementation on 75% of 40 cells. Literature context:
  the best published trace-driven L2 results are ~13–16% error (PPT-GPU,
  Buelow 2022), so 6.4% on a 40-cell sm_86 cohort is at or beyond published
  SOTA — *on accuracy*. **On robustness it still fails**: LOOCV-vs-holdout gap
  0.3385 (gate ≤0.20) and LOSO max 0.7076 (gate ≤0.25), driven by
  register_spill_local_mem (0.708), l2_prefetch_hint_sweep (0.573),
  working_set_capacity_sweep (0.501).
- **L1 sector hit rate is honestly broken**: deployable friendly-L1 geomean
  **1.947** (>190% relative error, HARD-2 MISS) after the truth-fitted
  correction factors were demoted to `diagnostic_truth_fitted_demoted`; L1
  closure geomean 0.89 over 30 cells. The SOTA root-cause match is exact and
  actionable (§3): the L1 walker has **no intra-SM warp interleaving, no MSHR
  model, LRU + fully-associative assumptions that real GPU L1s violate, and a
  fixed (non-carveout-aware) capacity** — precisely the error sources
  Nugteren 2014 quantified at up to ~50pp and fixed with a warp-pool
  scheduler + finite-MSHR model at 6.4% MAE.
- **Wall-clock cycles exists but its headline is leakage-suspect**
  [verified-this-session]: `timing_cycle_predictor.py` (roofline prior + ridge
  residual) reports a passing publication gate (cycle geomean 0.0559, n=13),
  but `MODEL_VERSION = "timing_cycle_ridge_v3_same_slug_local_residual"` and
  the 13-cell cohort spans only **4 slugs** (gemm×5, conv2d×4, flash_attn×2,
  dot×2), so **every** leave-one-out cell had a same-slug neighbor and the
  LOO predictions substitute same-slug NCU-label scaling — the same circular
  pattern whose cache-side analogue was just demoted. The deployable
  slug-disjoint cycles claim is **open**. The deploy-ns leg is already honest
  (interval-only, 1890 MHz short-burst anchor, `point_result_claimed=false`).
- **T1 replay counts and T2 traffic streams — the plan's own headline
  "NCU count-replacement" claim — are entirely unstarted** (zero of 15 count
  metrics captured; no producer; no lockstep validator; continuation
  task4–task15 all `pending`).

The SOTA study converges, from three independent directions (local PDF
corpus, 2023–2026 web literature, codex deep research), on one architecture:

> **Trace once → summarize once → predict many metrics.** A one-pass NVBit
> trace summarizer feeds (i) a deterministic T1 replay-count reconstructor
> (published fidelity: requests within ~0.001%, sectors within ~0.28% of
> hardware — the easiest and highest-confidence NCU-replacement claim), (ii)
> a scheduler-aware, MSHR-aware, sectored (HRD 32B/128B) cache walker for
> L1/L2 hit rates (published: 3.7–6.4% L1 MAE), (iii) a GCoM-lite analytical
> cycles model whose dependence stalls are **read nearly directly from SASS
> control bits** (stall/yield/dependence counters via cuobjdump — the
> single biggest non-root accuracy lever, per the MICRO'25 modern-core line:
> 13.45% MAPE Ampere, 17.41% Blackwell), and (iv) a DVFS decomposition
> `T(fc,fm) = t0 + max(α/fm, β/fc)` that converts cycles to NVML-conditioned
> ns intervals. Low-capacity ML (isotonic/ridge, optionally an
> efficiency-factor head bounded by the analytical ceiling) sits ON TOP of
> analytical priors, never replaces them, at this label budget. No published
> system fuses all of this under non-root constraints across Ampere +
> Blackwell — that fusion is this project's novelty claim.

---

## 2. Current State (Grounded)

### 2.1 What the completed loops landed

- **Cohort**: 44 sm_86 cells / 37 slugs, sealed split cal-28/val-8/holdout-8
  (frozen at commit `501723b0`); 40 cells admitted for v5 fit/score (4 holdout
  cells `predictor_bundle_absent`); L1 head cohort 30 cells. Full label store:
  74 cells (44 sm_86 + 30 sm_120 frozen at `b1646ef7`)
  (`data/sm_86_predictor_accuracy_recovery_v2/cohort_split.json`,
  `data/local_labels/sm_86_v5_head.json`, `data/local_labels/label_admitted.json`).
- **L2 accuracy gates** (`calibration_head_v5_accuracy_gates_sm_86.json`):
  full-cohort geomean 0.0642 / median abs 0.0592 / holdout 0.0552 — all PASS.
  Per-bucket p90 absolute-error gates FAIL: memory_stress p90 0.2346 (n=9),
  other p90 0.3587 (n=15) vs threshold 0.15.
- **L2 robustness gates** (`calibration_head_v5_robustness_sm_86.json`): FAIL.
  LOOCV-vs-holdout gap 0.3385 (≤0.20), LOSO max 0.7076 (≤0.25). Pre-refit
  values were 0.441 / 0.814 — the expanded corpus improved both but the
  pivotal hypothesis ("the corpus expansion is the structural fix") is **not
  confirmed**. Worst LOSO slugs: register_spill_local_mem 0.7076,
  l2_prefetch_hint_sweep 0.5734, working_set_capacity_sweep 0.5011.
- **Baseline panel** (`calibration_head_v5_baseline_comparison_sm_86.json`):
  v5 geomean 0.0628 vs uncalibrated walker 0.111, Nugteren-2014 reimpl 0.117
  (v5 wins 75% of cells), always-hit 0.117, cohort-mean 0.325, v4 Platt 0.203.
- **L1**: deployable friendly-L1 geomean **1.9469743371673827** (HARD-2 MISS,
  target <0.20); hostile-L1 0.3172; the diagnostic-shadow channel (the
  would-have-been factor-corrected values) reports 0.0148 but is **not
  deployable** (truth-fitted circularity). L1 closure geomean 0.89
  (`interim_diagnostic_report_sidecar.json`, shadow reports,
  `l1_closure_comparison_sm_86.json`).
- **Demotion registry**: all four correction-factor families
  (l1_silu_elementwise, l1_shared_reduction, l2_reduction_kernel,
  l2_high_footprint_low_reuse) carry
  `correction_factor_status=diagnostic_truth_fitted_demoted`; production
  helpers no-op the deployable value (`cache_predictor.py:3028-3065`).
  Blocking fact for re-derivation: silu has 1 slug, shared_reduction has 2 —
  no slug-disjoint data exists for those families.
- **Falsification** (`calibration_head_v5_falsification_sm_86.json`): v4→v5
  joint contribution 14.33% — **indeterminate** band [10%,30%); v4 MAE 0.1686
  vs v5 0.1323 (21.5% residual reduction). The prior "confirmed" verdict was
  an artifact of the truth-fitted factor leak.
- **plan_1_closure_status**: `aborted_unrecoverable_per_user_decision`.

### 2.2 Wall-clock truth [verified-this-session]

- `workloads/simbench_bw/timing_cycle_predictor.py` exists: static roofline
  prior (GA102 tensor peak × 0.066 efficiency prior, from
  `tools/hrd_profiler/ga102_peak_constants.json`) + ridge residual (λ=10) on a
  14-feature deploy-legal vector (label-only fields actively blocked by
  `assert_feature_vector_deploy_legal`).
- `MODEL_VERSION = "timing_cycle_ridge_v3_same_slug_local_residual"`: at LOO
  validation time, `_same_slug_scaled_neighbor_prediction` (line ~444)
  replaces the ridge prediction with `true_cycles(nearest same-slug cell) ×
  work_ratio` whenever a same-slug training cell exists.
- The 13-cell cohort (`timing_cycle_predictions_sm_86.json`) spans **4
  slugs**: conv2d_implicit_gemm_bf16_sm80 ×4, dram_streaming_dot_product ×2,
  flash_attn_fwd_bf16_sm80 ×2, gemm_hmma_bf16_sm80 ×5 ⇒ every LOO cell had a
  same-slug neighbor ⇒ the committed `publication_pass` (cycle geomean
  0.055856, median abs log 0.010734, bootstrap upper 0.106601, in
  `timing_closure_sm_86.json`) is a **same-slug result**, not a slug-disjoint
  deployable claim. Evidence status `Offline-verified`.
- The deploy-ns leg is interval-honest: `anchor_regime=short_burst`,
  `sm_clock_ref_mhz=1890.0`, `clock_uncertainty_fraction=0.06`,
  `point_result_claimed=false`. Dual-source noise floor 110352 ns (committed,
  provenanced). NCU cycles label (`gpc__cycles_elapsed.max`) is DVFS-stable;
  ns label inherits clock estimation error.

### 2.3 Unexecuted continuation work (all `pending` in the goal tracker)

task4 (widen NCU label capture to 8 probe-confirmed T1 counters), task5
(probe 7 pending identifiers + mem_clock/dram_bytes/pct-peak), task6 (NVBit
replay-count producer), task7 (T1 lockstep validator), task8 (Codex T1
review), task9 (texture/surface disposition), task10 (rebuild/re-pin 22
timing-exclusion binaries + stale readiness report), task11 (T2 traffic
labels + predictor), task12 (T3 re-score over rebuilt cohort), task13
(per-(cell,metric_key) harness + baselines/ablations), task14 (Codex stats
review), task15 (final compliance). Zero T1 count metrics exist in
`label_admitted.json` today.

### 2.4 Measurement substrate (features available vs absent)

Available now (producers committed and live on sm_86):
- **NVBit dynamic trace** (`tools/nvbit_tracer/`, byte-pinned path-boundary):
  per-instruction pc, active/predicate masks (proto3 optional), per-lane
  addresses (3 compression formats + allocation registry), 10-value
  `memory_op_class`, data width, warp/SM ids, vpc;
  `instruction_cache_op` wire slot **reserved but producer classifier not
  landed** (today resolves UNSPECIFIED ⇒ `predictor_l1_sass_provenance_missing`).
- **%clock64 phase ledger** (env-gated injection; function/basic-block/bar
  phase records) — built, available for phase timing evidence.
- **CUPTI Activity collector** (dual-source vs CUDA events; noise floor
  max(5%, 5µs); calibrated p99 110352 ns) + **NVML covariates** (200 Hz; typed
  `nvml_alignment_gap` for sub-poll kernels).
- **Static SASS extractor** (`static_sass_features.py`): 9-family mnemonic
  histogram, register/spill usage, separate static FLOP/byte estimate —
  `claim_class=static_sanity_check_only`; order-of-magnitude FLOP weights.
- **Device peaks** for sm_86 (`device_constants.json`: 1008 GB/s DRAM,
  1890 MHz boost; GA102 peak table in `tools/hrd_profiler/`); **all sm_120
  device constants are null** (`deferred_no_live_evidence`).
- **Trace budgets** (`tools/cloud/trace_budgets.py`): slowdown cap 1200×,
  live sm_86 ratio 434.72×, drop-rate cap 0.05, per-cell byte caps 2 GiB
  default down to 64 MiB with `threadblock_stride_N` / `kernel_limit_N`
  sampling grammar; sub-1ms baselines suppressed from the slowdown gate
  (µs-kernels inflate to ~14000–19000×).

Absent / planned (gaps the next phase must fill):
- Producer-side SASS cache-op classifier (wire tags 12–13 reserved).
- Per-instruction **dependence/control-bit extraction** (stall counter,
  dependence counters, yield — readable from static SASS via cuobjdump; not
  yet extracted).
- **P-chase / latency microbenchmark suite** for per-arch latency + cache
  geometry constants (L1/L2 latency, MSHR count, set mapping, replacement) —
  the published numbers (A100: global ~290 / L2 ~200 / L1 33 / shared 23/19
  cycles; GB203 Blackwell: L2 ~358, global ~876, L1 30–40) are
  close-cousin-arch facts that must be re-measured non-root on the actual
  3090 Ti and 5060 laptop.
- One-pass **trace summarizer / feature store** (sector summaries and
  mnemonics are currently reconstructed offline per-consumer rather than
  emitted as one canonical feature bundle).
- 13/15 portable_lcd roster slugs lack host-local traces; 4 holdout cells
  `predictor_bundle_absent`; 22 timing-exclusion binaries stale (task10 debt).

### 2.5 Producer hazards queued from the last loop (must fix before bulk regen)

- `run_v5_active_path_comparison` clobbers `sm_86_v5_head.json` (producer
  collision — observed once: admitted_count=0 stub).
- `regenerate_v5_artifacts` wrapper runs a split-unaware fitter step.
- A sweep test restamps `deploy_ns_crosswalk_sm_86.json` in place (test
  pollution).
- Tools-sweep failure baseline residual: 6 entries (nested-provenance ×3,
  sidecar-schema ×2, ledger digests) + simbench ac5 pair.

---

## 3. Failure-Mode → SOTA-Fix Map

| # | Observed failure (repo evidence) | Root cause | SOTA fix (source, published numbers) |
|---|---|---|---|
| F1 | L1 deployable geomean 1.947; closure 0.89 | `l1_walker.py:_walk_one_launch_fa` walks per-block accesses **sequentially block-by-block** — no intra-SM warp interleaving; **no MSHR model at all** | Nugteren HPCA'14: FIFO warp-pool (one memory request per warp, re-insert delayed by latency) + finite-MSHR merge/stall + latency-miss class. Ablated impact up to ~50pp; full model 6.4–8.3% L1 MAE, 268× faster than GPGPU-Sim |
| F2 | "Set-conflict over-prediction" documented; carveout cells predictor-hostile | L1 modeled as fully-associative LRU with fixed capacity; real GPU L1 is **non-LRU** with XOR-hashed set mapping and runtime-carveout capacity | Mei&Chu 2015: P-chase methodology reverse-engineers size/sets/ways/mapping/replacement **non-root** (%clock64 only). Run on both cards to populate per-arch static cards; make capacity carveout-aware |
| F3 | Sector vs line handled by a single LRU stack approximation | Flat reuse distance cannot track partial-line (sector) eviction | Maeda HPCA'17 HRD(32,128): one-pass two-granularity RD (propagate-on-infinity), >3× sectored miss-rate error reduction; repo already has `cache_model_hrd_second_layer_enabled` to build on |
| F4 | LOSO offenders: register_spill 0.708, l2_prefetch 0.573, working_set 0.501; inter-CTA reuse approximated by `alpha*log2(grid_volume/sm_count)` | Per-slug/per-family heuristics instead of structural features | SHALOM TPDS'22: classify accesses local / sequential-sharing / concurrent-sharing / private; MDM HPCA'20: DPKI + MSHR-batching + NoC/DRAM queueing features (arch-independent, NVBit-derivable) |
| F5 | Correction factors circular (truth-fitted) → demoted; HARD-2 reopened | Labels used as fit inputs on the same cells | Analytical-first discipline (Wang HPCA'19 hybrid; GCoM): labels become **validation**, not fit data — the structural escape from the 35–50-label budget |
| F6 | Timing publication_pass is same-slug leakage on a 4-slug cohort | LOO substitutes same-slug NCU-label scaling | GCoM-lite cycles (interval/CPI-stack) + **SASS control-bit stall readout** (Huerta MICRO'25: 13.45% Ampere / 17.41% Blackwell MAPE; control bits readable via cuobjdump = non-root near-direct dependence-stall readout); slug-disjoint gates mandatory |
| F7 | LOOCV/LOSO robustness gates still fail after corpus expansion | 0-parameter isotonic on a 3-feature frozen projection + 4 family buckets; richer monotone candidates never fit (deps absent); tiny per-bucket N | Analytical priors carry the structure; low-capacity heads only (ridge/isotonic); HELP/Multi-Predict few-shot device embeddings for cross-arch; CDMPP domain-invariant features; report robustness achieved-vs-target honestly |
| F8 | ns target structurally noisy (110 µs floor, DVFS, sub-poll kernels) | Single-anchor cycles→ns; NVML 200 Hz cannot see µs kernels | DSO 2024: `T(fc,fm)=t0+max(α/fm,β/fc)` compute/memory-clock decomposition; Wang/Wu DVFS structure (3.5–4.8% MAPE) with NVBit-reconstructed counts replacing forbidden counters; keep ns as interval until DVFS residual proven |
| F9 | Traces cost 434.7× and dominate; 13/15 portable_lcd slugs untraced | Full per-instruction tracing of everything | Sieve IISWC'23: stratify invocations by NVBit dynamic instruction count alone (1.2% error vs PKS 16.5%; 8–98× less profiling; explicitly names NVBit); Photon MICRO'23 warp-type/BB-stability early stop; cuThermo single-block sampling; PKA one-occupancy-wave guard |
| F10 | T1/T2 headline unstarted | — | Accel-Sim NVBit tracer line: reconstructed requests within ~0.001%, transactions/sectors within ~0.28% of hardware — deterministic, near-exact, the highest-confidence NCU-replacement claim available; PRT coalescing abstraction (Huerta MICRO'25) |

---

## 4. SOTA Survey Synthesis (per target)

### 4.1 Wall-clock cycles (T3 headline)

- **Lineage**: GPUMech (MICRO'14, interval analysis, 13–14% vs sim, 97×) →
  MDM (HPCA'20, MSHR-batching + NoC/DRAM queueing, 13.9% vs sim / 40% vs real
  GTX 1080; fixes 162–298% GPUMech error on memory-divergent apps; collects
  inputs **via NVBit by name**) → GCoM (ISCA'22, sub-core + sectored-L1 +
  MDM embedded, 10.0% MAE vs Accel-Sim, **24.4% vs real RTX 2060**, cross-GPU
  25.5% on RTX 3070; 7.6–345× speedups). All consume per-warp NVBit
  SASS+address traces; none needs counters at predict time.
- **Modern-core line** (Huerta et al., arXiv 2503.20481 + MICRO'25): the
  decisive non-root lever — NVIDIA dependence handling is compiler-managed
  and **encoded in SASS control bits** (stall counter 0–15, six dependence
  counters 0–63, yield, DEPBAR.LE), readable statically via cuobjdump.
  CGGTY issue policy; no operand collector; PRT coalescing; RF bank-conflict
  rules. Validated MAPE: 13.45% Ampere A6000, 17.41% Blackwell RTX 5070 Ti
  (corr 0.99) vs Accel-Sim 34.03%. **13–17% is the literature floor for
  cycles on this signal set**; analytical (non-simulator) variants should
  expect the 20–40% real-hardware band (GCoM 24.4%, MDM 40%).
- **Latency constants** (Abdelkhalik HPEC'22 A100; Jarmusch 2025 GB203;
  Arafa 2019 methodology): global ~290 / L2 ~200 / L1 33 / shared 23,19
  cycles (A100); Blackwell L2 ~358, global ~876.7, unified INT32/FP32 pipe,
  2 FP64 units/SM. All obtained with %clock64 + SASS only — **re-measure on
  the actual two cards with the same non-root method**.
- **Tensor kernels are the hard tail**: AMALI (ISCA'25) 23.59% MAPE on LLM
  kernels (GCoM degrades to 127.56%); the 2026 microbenchmark-driven stage
  model (TMA/TMEM/math stages) hits 1.31% on B200 tensor GEMMs but is weak on
  irregular apps; naive roofline shows >95% error on modern kernels.
- **Hybrid heads**: NeuSight (ASPLOS'25) roofline-bounded tile-MLP, 8.1%
  unseen-GPU avg; SynPerf (2026) analytical-pipeline features + shallow MLP,
  6.1% seen / **11.4% unseen incl. an sm_120-class RTX PRO 6000** — the
  "efficiency factor against an analytical ceiling" framing is the right
  shape for a small learned head at tiny label budgets.

### 4.2 L1/L2 sector hit rate (T2)

- Counterless trace-driven cache modeling is proven: Nugteren 2014 (6.4–8.3%
  L1 miss-rate MAE; warp-pool + MSHR + associativity + latency-miss
  extensions, each ablated), Buelow CGF'22 (NVBit→LRU sector sim validated
  vs NCU: **L1 3.73%, L2 15.81%**), PPT-GPU (~13% L1/L2 hit-rate error,
  450× faster than GPGPU-Sim), MLCRP (IEEE Access'25, ML over reuse
  profiles, <5% MAE, ~4 orders faster than cycle-accurate).
- The repo's L2 (6.4% geomean) **already beats the published L2 numbers**;
  the repo's L1 is far below SOTA — and §3 F1–F3 explain exactly why, with
  the fix list (warp interleaving, MSHRs, HRD(32,128), per-arch set
  hash/replacement, carveout-aware capacity).
- What a per-warp trace walker typically gets wrong (web sweep synthesis):
  single deterministic interleaving; global LRU vs non-oracle eviction;
  line-not-sector fills; no L1 bank/port contention; no finite
  MSHR/miss-merging; mishandled writes (L1 write-through/no-allocate vs L2
  write-back semantics).
- IPOLY pseudo-random L2 set indexing (Huerta, citing Jia) — a naive modulo
  mapping mispredicts conflict misses; the hash must be widened for
  Blackwell's larger L2.

### 4.3 T1 replay counts + T2 traffic streams

- Accel-Sim/Macsim NVBit tracer evidence: **requests within ~0.001%,
  transactions (32B sectors) within ~0.28%** of hardware counters. The
  per-instruction address + active-mask + opcode stream the repo already
  collects is sufficient to rebuild `inst_executed`, per-op requests, and
  sector counts almost exactly. This is the easiest, most defensible
  NCU-replacement claim — **build it first**.
- PRT (pending-request-table) coalescing abstraction (Huerta MICRO'25) maps
  NVBit addresses+masks → requests → sectors deterministically.
- MDM's equations directly produce T2 quantities (DRAM sectors via
  LLCMissRate×BlockSize; NoC/xbar traffic via #SMs×M×L^NoCService), so the
  T2 stream model and the cycles model share components.
- Risk register: ~0.28% sector residual includes sync-barrier internals NVBit
  misses; ATOM-vs-RED semantics; predication; Blackwell async-copy/TMA paths
  must route through the existing `kUnspecified` fail-closed fallback until
  classified.

### 4.4 ML/learned predictors at a 35–50-label budget

- Label-budget verdict (PDF deep-read): XAPP needed 122 datapoints, Braun 189
  kernels/GPU, SynPerf ~1M samples, NeuroScalar ~1.5B simulator-labeled
  instructions. **No pure-ML method was demonstrated at this repo's budget.**
  Analytical-first with labels-as-validation (Wang HPCA'19 hybrid, 17.04%
  MAPE, zero training labels) is the only structurally sound escape.
- Consumer-DVFS warning that matches the repo's failed gates: Braun's random
  forest hit 8.86–13.86% on professional GPUs but **52% median MAPE on a
  consumer GTX1650** (uncontrollable DVFS) — the exact risk profile of the
  5060 **laptop**. NVML clock/thermal/power/throttle covariates are
  mandatory inputs, not optional.
- Few-shot device transfer toolkit for sm_86→sm_120: HELP (NeurIPS'21,
  ~10 samples/new device) and Multi-Predict (AutoML'23, 4–10 samples via
  learnable hardware embeddings); CDMPP (EuroSys'24) domain-invariant
  representation + balanced sampling (10.85% cross-device). Borrow the
  hardware-embedding + domain-adaptation recipes when the sm_120 leg opens.
- Tree ensembles cannot extrapolate beyond their training range — a
  structural conflict with honest slug-disjoint LOSO where a held-out slug
  may be the extreme. Keep heads monotone/low-capacity.

### 4.5 Fast-model engineering (the "fast" requirement)

- **Sieve (IISWC'23)** — adopt wholesale as the inter-kernel layer: stratify
  kernel invocations by NVBit dynamic instruction count (CoV threshold
  θ=0.4), trace ONE representative per stratum, weight by instruction count,
  reconstruct via weighted-harmonic-mean IPC. 1.2% avg error vs silicon
  (PKS 16.5%), 8–98× less profiling; cross-arch relative-perf error 1.5%
  (vs PKA 9.8%) — directly serves the sm_86↔sm_120 decoupling goal. Sieve
  itself names NVBit as the intended lightweight tool.
- **Photon (MICRO'23)** — intra-kernel: warp-type identity (identical
  instruction/PC sequence) + basic-block issue→retire slope stability as the
  trace-termination signal (IPC is NOT reliably stable on GPUs — do not use
  IPC-stability stops). Re-tune thresholds per-arch.
- **PKA (MICRO'21)** — the "one full occupancy wave before trusting
  stability" guard; weighted-sum reconstruction generalizes to DRAM
  utilization.
- **cuThermo (2025)** — single-thread-block sampling to bound NVBit overhead.
- **Concorde (ISCA'25, CPU)** — represent the trace as compact per-component
  performance-bound distributions; O(1) inference independent of trace
  length; attractive shape for the one-pass summarizer's output.
- Net effect: the 434.7×-per-cell trace cost stops being the binding
  constraint because most invocations are never fully traced.

### 4.6 Non-root precedent and the novelty claim

Counterless NCU-style metric reconstruction exists in fragments: Accel-Sim
tracer (counts), Buelow/PPT-GPU/MLCRP (cache hit rates), DSO/Wang-Wu (DVFS
wall-clock, but counter-fed), NeuSight/SynPerf (ns, but no cache/sector
metrics), Penrose (counter-dependent, opposite design), eBPF/CUPTI-activity
lines (timing only). **No published system fuses NVBit sector/cycle
reconstruction + DVFS cycles→ns decomposition + read-only NVML covariates,
calibrated against quarantined NCU labels, across Ampere AND Blackwell.**
That fusion — with typed fail-closed exclusions and slug-disjoint honest
gates — is the project's publishable contribution.

---

## 5. Recommended Next-Phase Architecture

### 5.1 Per-target model classes

| Target | Model class | First deployable form | Literature anchor |
|---|---|---|---|
| T1 replay counts (inst_executed, per-op requests, ld/st sectors, l1tex requests) | Deterministic trace replay: active-mask × PRT coalescing × 32B sectoring per op-class | Exact-or-typed counts + NCU lockstep validator (p50 ≤1%, p90 ≤3%, low-support floors) | Accel-Sim tracer 0.001%/0.28% |
| L1 sector hit rate | Scheduler-aware sectored walker: warp-pool FIFO interleaving + finite MSHR + HRD(32,128) + per-arch set-hash/replacement/carveout from P-chase cards | Analytical hit rate + existing isotonic calibration; correction factors stay demoted | Nugteren 6.4%, Buelow 3.73%, Maeda >3× |
| L2 sector hit rate | Frozen v5 baseline + traffic-first upgrade (predict requested/hit/miss sectors, then derive rate); MDM divergence + SHALOM sharing features for the LOSO offenders | v5 stays the deployable head until a challenger beats it slug-disjoint | repo 0.064 vs published 13–16% |
| T2 traffic streams (dram/xbar sectors) | Miss-stream composition from the cache walker + MDM queueing identities | p50 ≤10%, p90 ≤25% with denominator floors (continuation AC-5 gates) | MDM; Buelow L2 ~16% context |
| Wall-clock cycles | GCoM-lite: SASS control-bit stall readout + interval/CPI stack + MDM memory terms + per-arch latency tables (P-chase) ; ridge residual only on top | Slug-disjoint cycle geomean with uncertainty; same-slug arm demoted to diagnostic | Huerta 13.45%/17.41%; GCoM 24.4% real-hw |
| Wall-clock ns | `T(fc,fm)=t0+max(α/fm,β/fc)` over predicted cycles + NVML covariates; dual-source crosswalk | Interval claim (already honest); point claim only for kernels above the noise floor with in-window NVML samples | DSO 4.6%; Wang/Wu 3.5–4.8% |

### 5.2 Feature pipeline (new producers, all non-root)

1. **One-pass NVBit trace summarizer / feature store**: per-kernel canonical
   bundle — dynamic opcode-class counts, per-PC memory streams, sector
   histograms, RD/HRD bins (32B + 128B), active-mask/divergence (DPKI)
   features, coalescing distributions, per-PC cache-op classes (once the
   classifier lands), %clock64 phase deltas, launch/occupancy, NVML/CUPTI
   covariates. Raw full traces become a debugging/label-side mode.
2. **Static SASS control-bit extractor**: stall counter, dependence-counter
   masks, yield, DEPBAR thresholds, reuse bits per instruction (cuobjdump
   parse; new tool OUTSIDE the byte-pinned tracer tree).
3. **P-chase microbenchmark suite** (per-arch static cards): L1/L2/shared/
   global latencies, MSHR count, L1 set mapping + replacement evidence,
   carveout capacity sweep — %clock64-bracketed, run on both local cards.
4. **Producer-side SASS cache-op classifier** filling the reserved wire slots
   (lifts `predictor_l1_sass_provenance_missing`).
5. **Sieve stratification + Photon early-stop** wired into the trace-budget
   machinery as pre-declared sampling policies.

### 5.3 Build order (information-per-round)

0. **Hygiene gate**: fix the two producer-collision hazards + crosswalk test
   pollution (§2.5); demote the timing same-slug arm to diagnostic (mirror
   of the correction-factor demotion) and re-score the ridge slug-disjoint.
1. **T1 producer + lockstep** (continuation task4–task7): widen labels to the
   8 probe-confirmed counters; build the deterministic count reconstructor;
   validate lockstep. Highest confidence, fastest claim, unlocks F-feature
   counts for everything else (DVFS model inputs, coverage denominators).
2. **L1 walker rebuild** (F1–F3 fixes + P-chase static cards): warp-pool
   interleaving, finite MSHRs, HRD(32,128), per-arch geometry; re-score L1
   honestly. This is the largest accuracy delta available anywhere in the
   project (1.947 → published 0.04–0.10 band if SOTA transfers).
3. **T2 traffic streams** (task11): labels + miss-stream model.
4. **Cycles (GCoM-lite) + ns interval** (task12 upgraded): control-bit
   extractor + interval model + DVFS decomposition; timing cohort slug
   expansion (4 → ≥10 slugs) so slug-disjoint scoring is possible at all.
5. **Per-metric evidence harness** (task13–15): per-(cell, metric_key)
   verdicts, baselines, ablations, claim-to-evidence regeneration, Codex
   stats review, final compliance.

### 5.4 Corpus strategy

- Grow sm_86 timing-labeled slug diversity first (the 4-slug timing cohort is
  the binding constraint, not total cell count).
- Family-targeted growth (per codex + the LOSO offenders): register/spill,
  prefetch-hint, working-set-capacity families need ≥3 slug-disjoint members
  each before any family-coupled feature is deployable.
- sm_120: defer live work to the alternating-session model; when it opens,
  apply the HELP/Multi-Predict few-shot recipe + per-arch P-chase cards
  rather than refitting from scratch.

---

## 6. Accuracy + Speed Targets Calibrated by Literature

Proposed as DIRECTIONAL targets (DEC-7 precedent: report achieved-vs-target
with CIs; structural/integrity gates stay HARD):

| Metric | Target | Literature justification |
|---|---|---|
| T1 counts | p50 ≤1%, p90 ≤3%, max ≤5% rel. error | Accel-Sim 0.001%/0.28% headroom |
| L1 sector hit rate | geomean rel ≤0.15; per-cell ≤10pp p90 | Nugteren 6.4–8.3%, Buelow 3.73% |
| L2 sector hit rate | hold ≥ v5 (0.064); fix per-bucket p90 ≤0.15 | repo already ≥ SOTA |
| Cycles (slug-disjoint) | geomean log-rel ≤0.20 (stretch 0.15) | Huerta 13–17% floor; GCoM 24.4% real-hw |
| ns | interval coverage ≥90% at stated clock band | DSO/Wang-Wu structure; 110 µs floor honored |
| Robustness | LOOCV gap ≤0.20, LOSO ≤0.25 re-tested per head | unchanged project gates |
| Trace budget | ≤1200× per traced representative; ≥5× amortized profiling reduction via stratification | Sieve 8–98× |
| Model runtime | per-kernel prediction ≪ kernel runtime (analytical + O(N log N) walker; no per-cycle simulation) | Nugteren 268×, PPT-GPU 450× class |

---

## 7. Hard Constraints & Frozen Invariants (carry forward verbatim)

- CLAUDE.md Rules 1–5: NCU only via `tools/local_labels/` →
  `data/local_labels/` on the two local cards; no nvidia-smi state mutation
  (no clock locking — DVFS handled by covariates, never controls); no
  privileged CUPTI (Activity allowed); no host-driver work; **no simulator
  revival** (analytical, trace-replay, and ML models are in scope; HyFiSS/
  Accel-Sim/GCStack are cited for abstractions and validation methodology
  only — no integration).
- Frozen surfaces: L2 v5 head byte-identity + regression guard; 30-cell
  sm_120 cohort at `b1646ef7`; 190-file path-boundary pin over
  `tools/nvbit_tracer/` + the CUPTI collector (new producers live OUTSIDE the
  pinned tree or re-pin in the same commit per the documented protocol);
  additive-only label schema (v5) with lockstep consumer updates.
- Validation discipline: slug-disjoint splits, sealed holdout touched once,
  typed fail-closed exclusions (never silent gaps), four evidence statuses
  (Offline-verified / Verified-live-sm_86 / Verified-live-sm_120 /
  Cross-arch-verified), falsification artifacts as first-class outcomes.
- Plan-token scanner: no `AC-*`/Milestone/Round markers in implementation
  code; the standing label-commit-last + POST-commit drift/recapture battery
  applies to any round that lands a commit.
- Architecture-coupling discipline: every latency/geometry/scheduler constant
  is a per-arch covariate (measured by the P-chase suite), never a baked
  cross-arch invariant; unknown sm_120 opcodes route through
  `kUnspecified` → `unsupported_memory_path`.

---

## 8. Open Questions / Decisions for Operator + Codex

- **Q1 — Scope and order.** Adopt the §5.3 build order (hygiene → T1 → L1
  rebuild → T2 → cycles → harness)? Alternatives: (a) finish continuation
  task4–15 as written before any SOTA upgrades; (b) L1-first (biggest
  accuracy delta) before T1. The dossier recommends §5.3: T1 is the fastest
  defensible claim and feeds features to everything else.
- **Q2 — Timing same-slug demotion.** Demote
  `timing_cycle_ridge_v3_same_slug_local_residual`'s same-slug arm to
  diagnostic (mirroring the correction-factor demotion) and accept that the
  slug-disjoint cycles number will initially be much worse than 0.056 — or
  keep the same-slug arm as the headline with an explicit caveat? The
  dossier strongly recommends demotion: the project's credibility rests on
  the honesty standard it already set in this loop.
- **Q3 — Robustness-gate posture.** The LOOCV/LOSO gates have now failed
  through two loops. Options: (a) keep gates as-is and target the L1
  rebuild + structural features at the named offenders; (b) re-scope the
  gates per-head (a fresh analytical head may pass where the calibration
  head cannot); (c) report-as-miss permanently under DEC-7. Recommendation:
  (a)+(b) — new heads get their own recomputed gates.
- **Q4 — New-producer governance.** The P-chase suite, control-bit
  extractor, SASS cache-op classifier, and trace summarizer are NEW
  producers. Confirm they live outside the byte-pinned tree (new
  `tools/` subtrees) vs amending the path-boundary pin. The NVBit tracer
  wire format should need only additive changes (cache-op classifier fills
  reserved tags 12–13).
- **Q5 — Sampling in scope?** Wire Sieve stratification + Photon early-stop
  into trace budgets this phase (the "fast" half of the user goal), or defer
  to a follow-up loop once the accuracy spine is rebuilt? Recommendation:
  land Sieve stratification (cheap, pure offline arithmetic over existing
  budgets) this phase; Photon early-stop as a stretch/follow-up.
- **Q6 — sm_120 this phase?** The session model alternates devices. Does
  this plan include a Verified-live-sm_120 leg (P-chase cards + label
  capture on the laptop) or stay sm_86-only with typed deferrals? The
  Braun-GTX1650 evidence (52% MAPE under consumer DVFS) says the sm_120
  timing leg NEEDS the covariate machinery to be ready before it runs.
- **Q7 — Model-class governance.** Install pygam/lightgbm to actually fit
  the pre-registered monotone GAM/GBM candidates (recording the environment
  change), or keep isotonic-only until the analytical rebuild lands?
- **Q8 — Corpus growth quantum.** Codex recommends growing toward ~150 sm_86
  cells before promoting new residual heads; the continuation plan's bar was
  ≥3 slug-disjoint members per factor family. Pick the binding number per
  family rather than a global count?

---

## 9. Verification Notes

- **[verified-this-session]**: `timing_cycle_predictor.py` MODEL_VERSION +
  same-slug substitution mechanics (lines ~53, ~444–532); the 4-slug/13-cell
  timing cohort composition (`timing_cycle_predictions_sm_86.json`); the
  `timing_closure_sm_86.json` publication-gate scores and interval-only
  deploy-ns posture; codex-cited repo paths all exist (`tools/cache_predictor/
  l1_walker.py`, `tools/hrd_profiler/reuse_distance_tracker.py`,
  `tools/sector_stream/`, `data/local_labels/timing_predictions/`).
- Repo gate/robustness/falsification numbers were extracted by the
  repo-state analysts directly from the named artifacts with quoted values;
  the dossier reproduces them exactly as quoted. Paper numbers in §4 come
  from PDF deep-reads of the local corpus (with section/table citations
  recorded in the workflow results) and from the cited URLs.
- Known tension to re-verify during planning: the L1 closure geomean is
  reported as 0.89 by `l1_closure_comparison_sm_86.json` while the
  deployable friendly-L1 headline is 1.947 in the shadow/diagnostic sidecar —
  different cohorts/buckets (30-cell closure vs friendly-bucket headline);
  both are MISSES but they are different denominators. Do not conflate.

---

## 10. Knowledge Consulted

- `.claude/knowledge/INDEX.md` — KB routing (measurement/blackwell cards
  + paper corpus skills).
- `.claude/knowledge/measurement/` — nvbit-injection-playbook.md (bounded
  instrumentation + A/B overhead discipline), cupti-activity-api.md (allowed
  CUPTI surface; separate-runs join), clock64-globaltimer-dvfs.md (%clock64
  semantics; covariates-not-controls), nonroot-cloud-contract.md,
  nvarchsim-vs-this-project.md (validation-discipline prior art).
- `.claude/knowledge/blackwell/` — rtx_pro_6000_blackwell_capabilities.md,
  blackwell-tuning.md, blackwell-compatibility.md, blackwell-tensor-perf.md
  (sm_120 occupancy ceilings, lineage split, no-NCU-derived-numbers rule).
- `references/papers/knowledge_base/` — selected_index.md,
  problem_playbook.md (paper routing + the "When The Playbook Is Wrong"
  escape clause), manifest.json focus curation, and the active/ PDF corpus
  (GPUMech, MDM, GCoM, Hong-Kim, Nugteren 2014, Maeda HRD 2017, SHALOM 2022,
  Mei&Chu 2015, Huerta 2025 ×2, Abdelkhalik 2022, Jarmusch 2025, NeuroScalar
  2025, XAPP 2015, Braun 2021, Wang 2019, PKA 2021, Photon 2023, Sieve 2023,
  Arafa 2019, SynPerf 2026, GCL-Sampler 2026).
- `plans/research/sm86_wallclock_prediction_research_dossier.md` — prior
  dossier (sibling-repo era; its "no duration head exists" claim is now
  superseded in this repo — see §2.2).
- `.humanize/rlcr/2026-06-04_01-48-48/goal-tracker.md` + round summaries —
  loop outcome and queued residual.

## 11. References (web)

- NeuSight: https://arxiv.org/abs/2407.13853 (ASPLOS 2025)
- SynPerf/PipeWeave: https://arxiv.org/abs/2601.14910 (2026)
- Microbenchmark-driven analytical modeling (Blackwell stages): https://arxiv.org/abs/2605.04178 (2026)
- Huerta et al., Analyzing Modern NVIDIA GPU Cores: https://arxiv.org/abs/2503.20481 (MICRO'25 extension: dl.acm.org/doi/full/10.1145/3725843.3756041)
- HyFiSS: https://dl.acm.org/doi/10.1109/MICRO61859.2024.00022 (MICRO 2024)
- GCStack/GCScaler: https://dl.acm.org/doi/10.1145/3695053.3731068 (ISCA 2025)
- AMALI: https://dl.acm.org/doi/10.1145/3695053.3731064 (ISCA 2025)
- PPT-GPU-Mem: https://dl.acm.org/doi/10.1145/3392717.3392761 (ICS 2020)
- MLCRP: IEEE Access 2025 (Yonsei DTL)
- Buelow et al., Fine-Grained Memory Profiling: https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.14671 (CGF 2022)
- Accel-Sim NVBit tracer: https://github.com/accel-sim/accel-sim-framework (ISCA 2020)
- DSO: https://arxiv.org/html/2407.13096v1 (2024)
- Wang/Wu DVFS models: https://arxiv.org/abs/1701.05308; https://www.mdpi.com/2079-3197/8/2/37
- HELP: https://arxiv.org/abs/2106.08630 (NeurIPS 2021); Multi-Predict: https://arxiv.org/pdf/2306.02459 (AutoML 2023)
- CDMPP: https://arxiv.org/abs/2311.09690 (EuroSys 2024)
- cuThermo: https://arxiv.org/abs/2507.18729 (2025)
- Omniwise (contrast case): https://arxiv.org/abs/2506.20886 (2025)
- Penrose (contrast case, counter-dependent): https://arxiv.org/html/2509.21762v1 (2025)
- Habitat: https://arxiv.org/abs/2102.00527 (ATC 2021)
- Locality quantitative study: https://link.springer.com/article/10.1007/s10766-022-00729-2 (2022)
