# RESEARCH DOSSIER — Non-Root Wall-Clock / Kernel-Duration Prediction on sm_86 (RTX 3090 Ti, GA102/Ampere)

Repo: `/home/jakeshea/3090ti_5060laptop_cross_study/3090ti_5060laptop_cross_study/non_root_gpu_profiler`
Date: 2026-05-30 · Scope: seeds (1) a Codex discussion of solution directions and (2) a `/humanize:gen-plan` implementation plan.

> **How to read citations.** Every load-bearing claim carries a `file:line` or artifact citation copied from the area findings. Where adversarial verification REFUTED a claim or could not verify it, the claim is tagged **[REFUTED]** or **[UNVERIFIED]** and is *not* relied upon — see §8 Verification Ledger. Where a citation's line number was found to be off-by-one or mis-attributed but the underlying number confirmed, the claim is tagged **[number confirmed, citation imprecise]**.

---

## 1. Executive Summary

"Non-root wall-clock prediction on sm_86" means: from features obtainable *without* NCU, *without* driver-root, and *without* privileged CUPTI (i.e., NVBit address/SASS traces, static SASS, launch geometry, and read-only NVML/CUPTI-Activity covariates), predict the per-kernel execution time of a GPU kernel on the RTX 3090 Ti (sm_86, GA102, Ampere, driver 595.71.05), and validate that prediction against ground-truth timing labels.

What exists today: the **label substrate** is built and committed (AC-7, commit `94f5a4e2`). Twelve of twenty admitted sm_86 cells carry NCU timing labels — `gpc__cycles_elapsed.max` (cycles) and `gpu__time_duration.sum` (ns) — gated through a coefficient-of-variation repeatability check (`data/local_labels/label_admitted.json`; `data/local_labels/timing_label_exclusions_sm_86.json:54`). The deploy-time observable (a dual-source CUPTI-Activity + cudaEvent reconciled duration) is also built, with a committed, provenanced noise floor. The NVML reference operating point is calibrated (AC-6, `2f966479`).

What does NOT exist: **the predictor itself.** There is no roofline/interval cycle prior, no ML residual head, no cycles→ns timing runner, and no cycle-MAPE/ns-MAPE scoring arm. `comparison_runner.py` scores cache hit-rate only (`workloads/simbench_bw/comparison_runner.py`; verified: zero timing-scoring tokens). AC-8 (roofline + residual → cycles), AC-9 (>200µs headline gate), and AC-10 (%clock64 ablation) are planned-and-unbuilt.

The central difficulty is a **regime collision**: every captured ground-truth duration is 2.31–7.11 µs (`label_admitted.json`), while the plan's own headline gate (AC-9) admits only kernels >200 µs (`plans/sm86_unified_multimetric_predictor_plan.md:56`). Zero of twelve cells qualify — the headline cohort is structurally empty. Compounding this, cycles→ns conversion uses a 1350 MHz NVML reference (`nvml_reference_operating_point_sm_86.json:17`) while the labeled kernels ran at ~1870 MHz boost, injecting a ~38% systematic ns bias before any model error. The clean, salvageable signal is **cycles** (DVFS-stable, CV 0.0022–0.0216); ns is the leg that "fails dramatically."

---

## 2. Current State (Grounded)

### 2.1 The label substrate (NCU cycles + ns)

- **Capture path.** `tools/local_labels/capture_timing_labels.py` runs `ncu --metrics gpc__cycles_elapsed.max,gpu__time_duration.sum --launch-count 1 --csv <binary>`, default 5 repeats (`capture_timing_labels.py:73,78`; metrics confirmed). It binds only when the on-disk binary sha256 matches the cell's committed `binary_sha256`, else a typed exclusion. It is fail-closed off-3090Ti (exit 3 when ncu absent / wrong device).
- **Gating.** Each metric flows through `repeatability_aggregator._aggregate_unit_tagged_metrics` with rule `coefficient_of_variation` and threshold **CV ≤ 0.05** (`tools/local_labels/metric_registry.json:30`, confirmed; gate direction `dispersion > threshold → drop` at `repeatability_aggregator.py:644`, confirmed). Floor is **n_repeats ≥ 5** (`DEFAULT_REPEATS_REQUIRED = 5`, `repeatability_aggregator.py:144`, confirmed; `--repeats` default 5 at `capture_timing_labels.py:73`, confirmed). `capture_timing_labels.py` requires **both** metrics to survive or excludes the whole cell.
- **Storage / cell counts.** `data/local_labels/label_admitted.json` holds **50 cells = 30 sm_120 + 20 sm_86** (confirmed). Of the 20 sm_86 cells, **exactly 12 carry a `unit_tagged_metrics` timing block** (each with both `gpc__cycles_elapsed.max` and `gpu__time_duration.sum`), and **8 are typed exclusions** (`timing_label_exclusions_sm_86.json:54`, `n_exclusions=8`; confirmed). The exclusion split is **7 `timing_binary_sha256_mismatch` + 1 `timing_repeatability_failed`** (confirmed). The single repeatability failure is **`cross_entropy_fwd_fp32_sm80`** (confirmed against the artifact) — note the round-7 summary prose mis-names it `silu`; the artifact is authoritative (see §8).
- **Label values (computed from `label_admitted.json`, all confirmed):**
  - Durations (`gpu__time_duration.sum`, ns): min **2310.4** (`silu_fwd_bf16_sm80`), max **7110.4** (`conv2d_implicit_gemm_bf16_sm80`), mean **3551.5**, median **3.10 µs**; dynamic range **3.08×** (confirmed).
  - Cycles (`gpc__cycles_elapsed.max`): **4308.2 – 13366.2** (confirmed).
  - Per-cell cycle CV: **0.00223 – 0.021596** (confirmed). Per-cell ns CV: **0.0 – 0.02441** (confirmed; two cells are exactly 0.0 due to quantization).
  - ns is quantized to a **6.4 ns grid** (all 12 ns means are exact multiples; confirmed). The "156.25 MHz GPCCLK reference" gloss is **[REFUTED]** — see §8; the real implied clock is ~1.87 GHz.
- **Registry / deployability.** Both timing metrics are `deployability = label_only` (`metric_registry.json:105,123`, confirmed) — they can **never** be a predictor *input*, only a fit target. The only `feature_eligible` occupancy-style metric is `occupancy_limit_warps_theoretical` (`metric_registry.json:405`).

### 2.2 The non-root deploy observable (dual-source CUPTI/event + reconciliation + noise floor)

- **Dual-source pair.** An LD_PRELOAD'd C++ collector (`workloads/simbench_bw/cupti_activity_collector.cu`) emits (1) CUPTI Activity kernel `duration_ns` (CLOCK_MONOTONIC-anchored) and (2) host `cudaEventRecord`/`cudaEventElapsedTime` wall time, reconciled per (kernel, repeat) (`schemas/cupti_activity_sidecar.schema.json:5`, confirmed).
- **Reconciliation rule.** Signed gap = `duration_ns − round(cuda_event_wall_time_ms*1e6)`; pass iff `|gap| ≤ effective_bound`, where `effective_bound = max(0.05 * cuda_event_wall_time_ms * 1e6, max(calibrated_floor_or_5000, 5000))`. The relative tolerance is **0.05 (5%)** (`cupti_activity_collector.h:64`; `validate.py:509` — confirmed) and the spec-default absolute floor is **5000 ns (5 µs)** (`cupti_activity_collector.h:65`; `validate.py:510` — confirmed). The same formula is implemented in three places (collector, `validate.py`, calibrate script) to prevent drift (confirmed).
- **Noise floor (committed, provenanced).** The calibrated absolute floor is **`tolerance_ns_calibrated = 110352 ns` (≈110 µs)** — the **p99 of a 1000-repeat |gap_ns| distribution on `rmsnorm_fwd_bf16`, RTX 3090 Ti** (`workloads/simbench_bw/golden/cupti_dual_source_tolerance.json:3,20`; confirmed). Full distribution: min 30816 / p50 69712 / p95 83106 / **p99 110352** / max 454624 / mean 72234.8 / stddev 16183.8 ns, n=1000 (confirmed). **This floor IS measured and provenanced (git sha `a0cabd4e`, 2026-05-07); the earlier finding that it was "stale sm_120 / no measurement provenance" is [REFUTED]** (see §8). The prior 5060-laptop value (130560 ns) was removed; 110352 is the live sm_86 number.
- **Hard gate.** Failing repeats drop from median/cv; **strictly >25%** failing → `cupti_dual_source_unstable_shape` (`run_suite.py:708,801-805`; threshold 0.25, strict; confirmed); zero passing repeats → typed exit **17** (`EXIT_CUPTI_DUAL_SOURCE_NO_PASSING_REPEATS`; confirmed). Typed exits: **11 = runtime_unavailable, 14 = kernel_timestamp_unavailable, 17 = dual_source_no_passing_repeats** (`cupti_activity_collector.h:55-56`; confirmed).
- **NVML covariates.** Per-launch join (`nvml_covariate_join.py`) windows each launch over NVML samples (200 Hz / 5 ms default; `cupti_activity_collector.cu:985`, confirmed; ≥100 Hz floor → ≤10,000,000 ns interval, `nvml_covariate.schema.json:60`, confirmed), fits a 3-term regression (temp/power/sm_clock deviations) against `gap_ns`, and admits only on **≥4 launches with in-window samples AND ≥10.0% residual-variance reduction** (`nvml_covariate_acceptance.toml:30`, confirmed; `nvml_covariate_join.py:407-413,455-456`, confirmed) — else `nvml_regression_undersampled` / `nvml_covariate_regression_ineffective`.
- **Cycles→ns conversion fails closed.** `cycles_to_ns = cycles * 1000 / sm_clock_mhz`; never substitutes a default; raises `NvmlReferenceUnset` / `NvmlClockImplausible` outside [300, 3000] MHz (`nvml_covariate_join.py:225-257`, confirmed). The reference *is* populated on this branch (1350 MHz; see §3.1). (Note: Area B's finding described the reference as "retracted nan/-1"; that was the **acceptance-TOML fallback** state. The committed *artifact* `nvml_reference_operating_point_sm_86.json` is populated — both can be true: the conversion prefers the artifact.)

### 2.3 The predictor code-vs-plan gap

- **No duration head exists.** The only prediction function in `cache_predictor.py` is `predict_cache_hit_rate` → a hit-rate scalar (`cache_predictor.py:3709,4312`; confirmed). Every "timing" token is the `timing_shape` *shape-point label*, not a duration. `tts_duration_predictor_ln_bf16` is a TTS neural-net *workload*, not a wall-clock predictor (confirmed naming collision).
- **No roofline prior.** No `operational_intensity`/`peak_FLOPs`/`peak_BW`/`achieved_BW` code exists outside `roofline_coordinate.schema.json` and `trace_size_estimator.py` (which only estimates trace *bytes*) (confirmed).
- **No timing scorer.** `comparison_runner.py` has zero cycle/ns/timing metric-target arm (confirmed grep = 0). It is hardwired to `predicted_hit_rate` (scalar) / `[lower,upper]` midpoint (bounded). The 8-tuple join key, coverage gate, and bootstrap exist and are reusable (see §6).
- **No committed timing MAPE anywhere.** `git grep mape -- data/` is empty (confirmed); the only scored sm_86 gates are L1/L2 cache-hit-rate accuracy gates. The execution-time predictor is "greenfield; none exists today" per the plan (`plans/sm86_unified_multimetric_predictor_plan.md:245`, confirmed).

---

## 3. Why It Is Hard — The Failure Modes

### 3.1 DVFS confound: cycles vs ns/clock(t) — the ~38% anchor bias
Cycle count is DVFS-invariant; wall time is `T_ns = C / f(t)`. The conversion uses one scalar anchor: **1350.0 MHz** (`nvml_reference_operating_point_sm_86.json:17`, confirmed) — the **median SM clock under a sustained 20s 4096² fp32 matmul at 90 °C, 294.5 W, with SW_THERMAL_SLOWDOWN active on 100% of accepted samples** (`nvml_reference_operating_point_sm_86.json:16,18,41`, confirmed; stdev 37.9 MHz, confirmed; max boost 2130 MHz, confirmed [number confirmed, citation imprecise — value is on line 10 not 9]). But the 12 labeled kernels ran at **1860.8–1879.8 MHz, median 1867, CV 0.35%** (cycles/ns; confirmed exactly). Converting their cycles at 1350 MHz over-predicts ns by **37.8–39.2%, median 38.3%, mean 38.5%** on every cell (confirmed; the conv2d cell: 13366.2×1000/1350 = 9901 ns vs true 7110.4 = +39.2%, confirmed). This is a regime mismatch (sustained-throttle anchor vs short-burst boost kernels), not a code bug. **Cycles are robust; ns inherits the entire clock-estimation error.**

### 3.2 The noise floor (~110 µs) vs sub-10 µs kernels
The dual-source deploy observable carries a **p99 |gap_ns| = 110352 ns** floor (confirmed, provenanced — §2.2). Every labeled duration (2.3–7.1 µs) is far below it; for a 3 µs kernel the floor-relative measurement noise dominates by ~30×. The plan declares (`plans/sm86_unified_multimetric_predictor_plan.md:56,183,245,279`, all confirmed) that "the ~110 µs dual-source p99 floor dominates relative error below 200 µs." **[UNVERIFIED]:** the area-finding gloss "15–48× larger than every label" is *not* derivable from the cited plan (see §8) — use the per-cell ratios instead (110352/7110.4 ≈ 15.5× to 110352/2310.4 ≈ 47.8×, which an operator can recompute from the confirmed durations, but it is not stated in any cited artifact).

### 3.3 The Goldilocks-disjoint cohort problem
AC-9 requires NCU duration **>200 µs** for the headline cohort (`plans/sm86_unified_multimetric_predictor_plan.md:56`, confirmed; DEC-3 RESOLVED as a hard exclusion at `:198`, confirmed). **Zero of 12 cells qualify** (max 7.11 µs = 28× below the gate; confirmed). The smoke floor is 5 cells/arch and publication floor 10 cells/arch (`comparison_runner.py:280-281`, confirmed) — **even the smoke floor is unreachable for the >200 µs headline.** Short kernels (the entire admitted cohort) are below the floor and un-scoreable at deploy; long-enough kernels (>200 µs) do not exist anywhere in the repo. There is no populated overlap window. `tts_duration_predictor_ln_bf16`, whose `timing_00` shape (b16/s1024/h1024, `shapes/tts_duration_predictor_ln_bf16.toml:8-20`, confirmed) would likely exceed 200 µs, is `arch_gate: sm_90a sm_120` (confirmed) — excluded on sm_86, zero labels (confirmed).

### 3.4 Coverage-vs-noise tension
Tightening to >200 µs empties the cohort. Relaxing to admit 2–7 µs kernels makes the floor-relative noise dominate. The cohort is also GEMM-heavy (6 of 12 timed cells are GEMM variants; 4 GEMM cells cluster near-duplicate cycle counts), so effective distinct duration points are ~9–10 — at or below the publication floor of 10 with any per-arch split. The 3.08× dynamic range gives a roofline+residual regressor almost no leverage.

### 3.5 Deploy-time feature poverty (no NCU / no clock-lock / sub-poll kernels)
At deploy there is no NCU (cycles can only come from %clock64 or be predicted), no clock-lock (Rule 2), and the SM clock is a free DVFS variable. The only non-root per-kernel clock probe — the NVML polling thread — samples at 5 ms (200 Hz); a 2–7 µs kernel window contains **zero** in-window NVML samples (`nvml_alignment_gap`), so per-kernel effective clock is **unmeasurable non-root** for these kernels and the covariate regression undersamples (n<4). %clock64 is per-SM cycles (not frequency, not cross-SM comparable); %globaltimer is ns but tool-only/target-specific with unspecified resolution (`references/.../ptx_isa/10_special-registers.md:948-982,1097-1124`; `.claude/knowledge/measurement/clock64-globaltimer-dvfs.md`). No non-root primitive returns a sub-poll kernel's effective average frequency. The most predictive cycle features (achieved occupancy, IPC, achieved DRAM throughput) are all `label_only` / admin-gated and forbidden as inputs.

---

## 4. What Has Been Tried vs Only Planned

**Built (and where relevant, failing on the current cohort):**
- AC-6: NVML reference operating point calibrated (commit `2f966479`); populated artifact at 1350 MHz (confirmed). *Built but mis-regime'd for the label kernels — §3.1.*
- AC-7: NCU timing labels captured on 12/20 sm_86 cells (commit `94f5a4e2`); CV-gated, byte-additive splice (confirmed). *Built and verified; the labels are real, tight, DVFS-stable.*
- Dual-source CUPTI/event observable + reconciliation + calibrated 110 µs floor + NVML covariate join + the no-root/no-NCU static guard (all confirmed present).
- The frozen L2 v5 cache-hit-rate predictor, comparison scorer, coverage gate, and bootstrap (confirmed; reusable, see §6).

**Planned but NOT built (zero implementation):**
- AC-8: roofline analytical prior + ML residual → cycles, ns via NVML clock; report cycle-MAPE + ns-MAPE with bootstrap CI (`plans/...:52`, confirmed — uses "Reports," pins **no** numeric pass/fail threshold; confirmed).
- AC-9: >200 µs headline gate (`:56`, confirmed) — currently vacuous against the cohort.
- AC-10: %clock64 phase-cycle ledger ablation. The ledger machinery (device struct, `record_phase_clock` reading `%clock64`, host aggregator, schema) is fully built **offline** but has **never produced a live ledger JSON** (confirmed; no `phase_cycle_ledger_*.json` outside fixtures).
- AC-11 / M3: cohort → ≥50 admitted sm_86 cells, **each** carrying wall-clock + L1 + L2 (`:64`, confirmed; HARD GATE). Today only 12/20 sm_86 cells have any timing label.

**Round history.** The last landed work is AC-7 (round-7 summary, `.humanize/rlcr/2026-05-30_02-22-44/round-7-summary.md:50-57`). Goal-tracker marks task10 (AC-8) "active next," task12/13 (AC-9/10) "coding" — i.e., unstarted (confirmed). No prior duration predictor was ever scored on sm_86; no timing MAPE was ever committed (confirmed). The L1 leg's history (Rounds 3–5: "fitted but not load-bearing / below floor / typed_unavailable") is the cautionary precedent for a green-but-vacuous closure.

---

## 5. Candidate Solution Directions

Six concrete, constraint-respecting directions. Each respects the frozen invariants of §6 (no edit to the frozen L2 v5 head, the sm_120 cohort, the path-boundary producer stack, or the additive schema; all new outputs additive).

### D1 — Cycles-first headline, ns as an honest clock-uncertainty band (recommended primary; matches AC-8 framing)
- **What:** Make `gpc__cycles_elapsed.max` the load-bearing headline target. Build a SynPerf/GPUMech/GCoM-shaped analytical cycle prior (`compute_cycles` from static SASS instruction mix; `memory_cycles` from NVBit sector traffic × Ampere latency constants — global ~290, L2 200, L1 hit 33, shared ld/st 23/19 cycles, `references/.../2022__demystifying_..._ampere...pdf` Table IV, confirmed; amortized by predicted L1/L2 hit rates from the existing cache heads and active-blocks occupancy) + a small **regularized** ML residual. Report **cycle-MAPE** with bootstrap CI. Derive ns via `cycles_to_ns` but report it as an **interval** spanning the [1350, 2130] MHz clock range, not a point MAPE.
- **Data/feature needs:** NVBit `summarize_launch` counts (already emitted: `observation_builder.py:193-273`, confirmed), static SASS instruction mix + FLOP count (new cuobjdump/nvdisasm extractor — does not exist), launch geometry → theoretical occupancy, predicted L1/L2 hit rates, GA102 device peaks (absent from `device_properties.json` — confirmed; must source FP32 FLOP/s, DRAM GB/s, peak clock from vendor spec or fit them).
- **Gate it clears:** cycle-MAPE over ≥5 (smoke) / ≥10 (publication) labeled cells — *cycles are not floor-limited and the 12 cells are all cycle-labeled with CV 0.0022–0.0216.* This is the only currently-feasible non-empty timing result.
- **Risks:** N=12 over-fits the residual (regularize hard; let the prior dominate); roofline homogeneity at 2–7 µs (launch/tail overhead dominates — add an explicit pedestal term); cycle target is `.max` across GPCs, partly fixed-overhead.

### D2 — Two-regime short/long model (formalizes AC-9)
- **What:** Below ~110 µs the wall-clock is launch/tail-overhead + measurement-noise dominated; above ~200 µs it is predictable compute/memory work. **Short regime:** predict cycles only; ns is not honestly claimable. **Long regime:** predict ns directly via the roofline prior. The ~200 µs boundary is tied to the *measured* 110 µs p99 floor (which IS provenanced).
- **Needs:** a populated >200 µs cohort (does not exist) — i.e., depends on D6.
- **Gate:** restricts the ns headline to >200 µs by construction; short kernels get a cycles-only claim + an explicit floor caveat.
- **Risk:** the long regime requires cohort expansion (real M3 work); the regime boundary moves if the cohort/NVML conditions change.

### D3 — Direct dual-source wall-clock with covariate de-confounding
- **What:** Model the deployable CUPTI/event wall-clock directly, using the existing `nvml_covariate_join` regression to subtract clock/thermal/power deviations from the reference operating point.
- **Needs:** long kernels whose launch windows contain ≥4 NVML samples and clear the ≥10% residual-variance-reduction gate (`nvml_covariate_acceptance.toml:30`, confirmed).
- **Gate:** could supply a deploy-realistic ns headline for the long regime.
- **Risk:** **demonstrably inert on the current sub-10 µs cohort** (undersampled / alignment-gap). Defer until a large-kernel cohort exists. Fold into D2's long regime.

### D4 — Honest regime-restricted headline + refutation of the rest (defensible fallback)
- **What:** If the cohort cannot be regrown this loop, restrict the *positive* headline to (i) cycle-MAPE and (ii) cycles→ns conversion accuracy validated against NCU's own `gpu__time_duration.sum` (clean), and **emit a schema-valid `verdict_falsification` artifact** that explicitly refutes the deployable-CUPTI-wall-clock headline below 200 µs, citing the measured 110 µs p99 floor as the published reason.
- **Gate:** an emitted falsification IS a first-class terminal outcome (§6). Confirmed bands: confirmed ≥30% / refuted <10% / indeterminate 15–29% (NOT closure) (`plans/per_bucket_..._plan.md:74-76`, confirmed).
- **Risk:** weakens the wall-clock contribution to "cycles + conversion"; a reviewer wanting a real deployable-wall-clock number will find none exists on this cohort. **Must be an emitted artifact, not silent failure.**

### D5 — Recalibrate the conversion anchor to the deploy/label regime
- **What:** Either (a) recompute the NVML reference under a *short-burst* workload matching the inference launch pattern (so the anchor reflects ~1867 MHz boost, not 1350 MHz throttle), or (b) sample the NVML SM clock **concurrently** during each NCU capture and store it per-cell so the cycles↔ns relationship is exact per cell. Either removes the ~38% systematic bias (§3.1). Must preserve CLAUDE.md Rule 2 (read-only NVML, no clock-lock).
- **Gate:** drops the ns-MAPE clock-anchor bias from ~38% toward the ~1–2% NVML clock-quantization floor.
- **Risk:** even a regime-matched scalar cannot track per-kernel `f(t)` for cold-vs-warm / isolated-vs-back-to-back launches; may need a per-kernel clock *model* (features → clock) since the clock is unmeasurable non-root for sub-poll kernels.

### D6 — Cohort / kernel expansion to populate the Goldilocks window
- **What:** (a) Recover the 7 `timing_binary_sha256_mismatch` cells via deterministic rebuild + recapture (raises timing cohort from 12 toward 19). (b) Author/scale longer-running kernels (>200 µs) — the portable_lcd runners already parameterize size via `--shape-point`, so a `timing_01`/`timing_02` geometry per slug is low-cost — and capture all three label families (wall-clock + L1 + L2) against one binary build to satisfy the AC-11 invariant. (c) Diversify away from the GEMM-heavy composition (add memory-bound/divergent kernels) so the roofline prior is exercised across both rooflines.
- **Gate:** the only path to a literally non-empty >200 µs headline cohort and to the AC-11 ≥50-cell hard gate.
- **Risk:** the sha256-mismatch drift implies non-reproducible builds across sessions; will recur as the cohort grows. NCU recapture must stay under the `tools/local_labels/` quarantine.

---

## 6. Hard Constraints & Frozen Invariants

**Must not break:**
- **Frozen L2 v5 head byte-identity.** `data/local_labels/sm_86_v5_head.json` pinned to sha256 `9597457b69d409a699e5ff11ec2c5653d5b3c29409caf5aa195e08b43be18cfc` (confirmed pin == on-disk) + 2 comparison/accuracy-gate artifacts; headline geomean must reproduce within **1e-9** (eps=0.01, log floor 1e-12) (`test_frozen_l2_v5_regression_guard.py:36,44-45,152-154`, confirmed).
- **sm_120 cohort freeze.** 30 sm_120 cells frozen at commit `b1646ef7` (`test_sm120_cohort_frozen_at_b1646ef7.py:38,42`, confirmed). Timing-label writes share `label_admitted.json` with these cells and must not re-key/reorder them.
- **No-NCU / no-root static guard.** `test_no_root_or_ncu_static_guard.py` blocks NCU CLI/artifacts off-quarantine, nvidia-smi mutation flags, kernel-module ops, and admin-gated CUPTI Profiling surfaces (PC_SAMPLING/EVENT/METRIC/etc.). It **allowlists** the wall-clock telemetry surface (`cuptiActivity*`, KIND_KERNEL/CONCURRENT_KERNEL/etc.) the predictor depends on (confirmed). CLAUDE.md Rules 1–5 (no NCU off-local; no driver-root; no privileged CUPTI but CUPTI Activity allowed; no host-driver work; no simulator revival) confirmed at `CLAUDE.md:8-45`.
- **Path-boundary producer stack.** **190** tracked files (`tools/nvbit_tracer/` + `cupti_activity_collector.cu/.h`) pinned by sha256 (`path_boundary_baseline.json`, confirmed = 190; `test_no_root_or_ncu_static_guard.py:981-984`). Any byte edit is a hard stop requiring a plan amendment + re-pin in the same commit. **The NVBit %clock64 producer and the CUPTI collector are immutable** — the predictor must work from existing producer output.
- **Additive / lockstep schema.** `schema_version` is `const 5` at file and per-record level with `additionalProperties:false` at every layer (`local_label_admitted.schema.json:5`). New label families ride the optional `unit_tagged_metrics` array; a true bump must land lockstep across producer + sidecar schema + admitted schema + aggregator + parser + import_label. Plan-token discipline: implementation code must contain no `AC-`/`Milestone`/`Round N` markers.
- **No-deployability-leakage.** Both timing metrics are `label_only`; IPC and achieved occupancy are label_only; only theoretical occupancy is feature_eligible (`no_deployability_leakage_guard.py`, confirmed).

**Terminal-state vocabulary:**
- A loop closes via **`research_success`** (positive ACs pass) OR an emitted **`verdict_falsification`** artifact (honest refutation is first-class, but must be schema-valid and emitted, never silent). On this 3090 Ti machine, cross-arch legs are recorded as **`deferred_cross_arch_requires_sm120_session`** (the area finding's "`engineering_closure_refuted`" naming should be reconciled with the actual `verdict_falsification` vocabulary the repo uses — see §7).
- Falsification sub-verdicts: **confirmed ≥30%, refuted <10%, indeterminate 15–29% (NOT closure)** (confirmed).
- Four evidence statuses: **Offline-verified, Verified-live-sm_86, Verified-live-sm_120, Cross-arch-verified.** On this host the wall-clock plan may only claim **Verified-live-sm_86** or **Offline-verified** (confirmed). A typed fail-closed exception is NOT live evidence.

---

## 7. Open Questions / Decisions for Operator + Codex

Deduplicated and collated across all areas. These are the decision-grade ambiguities.

**Q1 — Is a regime-restricted, cycles-primary honest headline acceptable?** The plan currently demands a co-equal ns-MAPE headline (AC-8). Given the ~38% clock-anchor bias and the ~110 µs floor, ns may be structurally unachievable to a tight bound non-root. **Decision:** make cycle-MAPE the sole headline with ns as an appendix/interval, or keep co-equal ns?

**Q2 — Is the target cycles-primary-then-ns, or direct-ns?** The cells store NCU `gpu__time_duration.sum` (clean), but deployment only has the noisy dual-source CUPTI/event ns. **These are different observables and there is no agreement artifact crosswalking them.** Validating ns-MAPE against NCU ns measures the *conversion*, not deploy accuracy. **Decision:** which is the scored target, and does an explicit CUPTI-vs-NCU agreement artifact (same binary + shape) need to be built first?

**Q3 — Is cohort expansion in scope this loop?** As-is, the >200 µs headline cohort is empty (0/12). **Decision:** (a) author/scale >200 µs kernels (D6) + recapture, (b) re-scope the headline to sub-200 µs cycles with an explicit floor caveat (D4), or (c) replace the 200 µs gate with a cycles-based or noise-floor-relative criterion? This single decision determines whether AC-8/AC-9 are buildable at all.

**Q4 — What MAPE gate is the bar?** No timing accuracy threshold exists anywhere in code (confirmed). **Decision:** define explicit log-target cycle-MAPE / ns-MAPE pass/fail thresholds (the L2 cache gates — geomean ≤0.15, median ≤0.06 — are bounded-[0,1] fraction gates and may be inappropriate for log-target errors).

**Q5 — Recalibrate the NVML anchor, or model the per-kernel clock?** A single 1350 MHz scalar injects ~38% bias and cannot track `f(t)`. **Decision:** recalibrate under a label-matched short-burst workload (D5a), store clock per-cell concurrently (D5b), or fit a per-kernel clock model (D5)?

**Q6 — Recover the 7 sha256-mismatch cells?** Deterministic rebuild + recapture would raise the cohort from 12 toward 19. **Decision:** rebuild to the committed hash, or re-admit at new hashes (and how to make builds reproducible across sessions)?

**Q7 — Take the %clock64 phase-cycle ledger live?** It has never produced a live JSON and cannot supply `f(t)`, but can corroborate cycles. **Decision:** wire it live as the AC-10 ablation (strictly with/without, never a silent headline input), or keep it offline-only?

**Q8 — Is `gpc__cycles_elapsed.max` the right primary target?** It is a `.max`-across-GPC elapsed window dominated by launch/drain overhead at 2–7 µs, so the roofline prior (steady-state) may have little signal. **Decision:** keep it, add a pedestal term, or prefer a busy-cycle-style metric (subject to label_only/no-privileged-CUPTI constraints)?

**Q9 — Branch ambiguity.** HEAD is `083c178a` on `cleanup/pre-merge-simplification` (a merge-prep branch), not `3090ti_cache_sims` as some task framing stated (confirmed). **Decision:** which branch does the wall-clock plan execute on, and does the in-flight arch-config unification interact with the timing files?

**Q10 — Terminal-vocabulary reconciliation.** The dossier prompt references `engineering_closure_refuted`; the repo's actual closure vocabulary is `research_success` / `verdict_falsification` (confirmed/refuted/indeterminate) / `deferred_cross_arch_requires_sm120_session`. **Decision:** confirm the canonical terminal-state names before the plan hard-codes them.

---

## 8. Verification Ledger

Claims that adversarial verification **REFUTED**, could **not verify**, or **partially confirmed** in a way that materially weakens the area finding. The plan must NOT build on these.

**REFUTED — do not rely on:**
1. **"The ~110 µs dual-source p99 floor is a STALE sm_120 (RTX 5060) number with no measurement provenance" (Area E, Area G).** **REFUTED.** `golden/cupti_dual_source_tolerance.json` is a committed, provenanced **sm_86 (rtx_3090_ti)** measurement: `tolerance_ns_calibrated=110352`, `calibration_policy=p99_with_floor`, n=1000, git sha `a0cabd4e`, 2026-05-07, full distribution committed. The "REMOVED" language applied to the *superseded* prior 5060 value (130560 ns), not the live sm_86 number. **The 110 µs floor is real, measured, and provenanced** — directions that "first measure the floor before trusting it" (D-area E/G candidate) are unnecessary; it already exists.
2. **"156.25 MHz GPCCLK reference" attribution for the 6.4 ns ns-quantization grid (Area A).** **REFUTED gloss.** The 6.4 ns grid and "all ns means are exact multiples" are CONFIRMED, but 156.25 MHz is merely 1/6.4 ns (the NCU timestamp quantization step), not a physical GPCCLK. The real implied clock is ~1.87 GHz. Do not present 156.25 MHz as a hardware clock.
3. **"Captured ns CV range = 0.002–0.029" (Area G).** **REFUTED.** Live `label_admitted.json` ns dispersions are **0.0–0.02441** (two cells at 0.0). The 0.002 lower bound is actually the *cycles* CV min; 0.029 matches neither metric. Use ns CV 0.0–0.0244.
4. **"Captured cycles CV range = 0.003–0.029" (Area G / round-7 prose).** **REFUTED at both bounds.** Live cycles dispersions are **0.00223–0.021596**. Use this range, not the rounded prose "~0.003–0.029."
5. **Repeatability-failed kernel = `silu` (round-7-summary.md:55/56).** **REFUTED by artifact.** The committed exclusion is **`cross_entropy_fwd_fp32_sm80`**; `silu_fwd_bf16_sm80` is one of the 12 *successfully captured* cells (2310.4 ns). Trust the artifact, not the round summary prose — a flagged provenance-drift signal in the summaries.

**UNVERIFIED — could not confirm against cited artifact:**
6. **"~110 µs floor is 15–48× larger than every label" (Area A, Area F).** **UNVERIFIED** as a stated figure — not present in any cited plan/artifact; an operator can recompute it from confirmed durations (≈15.5×–47.8×) but it must not be cited as if it came from the plan.
7. **"per-unit repeatability thresholds include n_repeats≥5 in AC-11.1" (Area I).** **Partially confirmed / mis-attributed.** The three threshold values (fractions stddev≤0.03; cycles/ns CV≤0.05; counts rel-tol≤0.05) are exact, but `n_repeats≥5` lives in AC-11 (`:64`) and is absent from AC-11.1 (`:67`) and `metric_registry.json`. Cite the correct location.
8. **"NVML reference operating point is RETRACTED (nan/-1), conversion fails closed" (Area B).** **Partially confirmed / context-dependent.** The *acceptance-TOML fallback* block is nan/-1 (confirmed at `nvml_covariate_acceptance.toml:52-55`), but the committed *artifact* `nvml_reference_operating_point_sm_86.json` IS populated (1350 MHz, confirmed). `cycles_to_ns` prefers the artifact, so conversion does **not** fail closed on this branch. Do not plan around "ns leg cannot run until calibration" — it can run (at the wrong-regime 1350 MHz).

**Partially confirmed — number right, citation imprecise (safe to use the number):**
9. Max boost 2130 MHz: value confirmed, on line 10 not the cited line 9.
10. `sm_clock_ref_mhz=1350.0`: value confirmed, on line 17 not the cited line 15; the "median" descriptor is an interpretation, not a field label (the value is in fact a median of accepted samples per the producer, so it is accurate, just not labeled in the JSON).
11. Several test-file and header citations dropped the `workloads/simbench_bw/` or `tools/local_labels/` path prefix and one ("`tests/test_handle_timing_dual_source_hard_gate.py`") is actually under `workloads/simbench_bw/tests/`; line numbers and content confirmed.
12. "8 exclusions = **6** binary_sha256_mismatch" (one Area B/F load_bearing_number): **REFUTED count** — it is **7** mismatch + 1 repeatability (confirmed elsewhere in the same areas). Use 7+1.
13. "SynPerf 6.1% seen / 11.4% unseen, 6.7× vs Neusight": numbers confirmed, but they are in the paper's **Introduction**, not the abstract; the abstract states 6.1% kernel / 8.5% E2E. Attribution-imprecise, numbers usable.

**Confirmed and safe to rely on (high-value load-bearing):** 12/20 timing cells; 8 exclusions (7+1); durations 2310.4–7110.4 ns (mean 3551.5, median 3.10 µs, range 3.08×); cycles 4308.2–13366.2; implied clock 1.8608–1.8798 GHz (CV 0.35%); ~38–39% ns over-prediction at 1350 MHz; 0/12 clear the >200 µs gate; 110352 ns p99 floor (provenanced sm_86); CV≤0.05 gate, n≥5; coverage floors 5/10; bootstrap 2000/0.95/log-space-geomean; 8-tuple join key; frozen L2 sha + 1e-9 tolerance; 30 sm_120 frozen at `b1646ef7`; 190-file path boundary; 5/5 binding CLAUDE.md rules; deployability label_only for both timing metrics; NVML reference 1350 MHz / 90 °C / 294.5 W / 10251 MHz / 2431 of 3357 accepted; %clock64 never live; no duration head / no roofline / no timing scorer exist; no committed timing MAPE.

---

## Appendix: Verification run metadata
- Areas mapped: 9; load-bearing claims verified: 131; refuted: 3; unverifiable: 0
- Logs: ['Mapped 9/9 areas', 'Verifying 131 load-bearing claims/numbers', 'Verified: 111 confirmed, 17 partial, 3 REFUTED, 0 unverifiable']
