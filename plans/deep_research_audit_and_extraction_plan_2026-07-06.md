# DEEP-RESEARCH AUDIT + EXTRACTION & REFACTOR PLAN — Fast NCU-Accurate GPU Model

Date: 2026-07-06 (evening session) · Repo: `cross_arch_gpu_profiler_suite` · Host: RTX 3090 Ti (sm_86)
Status: **companion audit to** `plans/micro2027_ncu_sim_execution_plan.md` (the authoritative
execution plan, same date). This document is an *independent* re-audit produced by a fresh
4-agent deep-research pass (new-repo map · legacy-tree inventory · adjacent-home-dir scout ·
adversarially-verified web SOTA refresh) plus direct session verification of every load-bearing
claim. Where it **extends or corrects** the execution plan, items are flagged **[Δ]**.
It does not re-litigate the locked operator decisions D1–D8; it verifies their premises and
fills in the extraction detail the operator asked for ("extract useful scaffolding, tools and
knowledge base from the legacy branches to refactor the new repo").

**Current-scope supersession (2026-08-09).** The historical external-simulator
baseline and speed-comparison clauses in this companion audit are non-operative.
NCU silicon labels are the sole active accuracy baseline, and runtime/profile
measurement covers only this repository's `sim-run`. Accel-Sim/GPGPU-Sim remain
contextual prior art unless explicitly reauthorized.

---

## 1. Verified ground-truth state (all claims re-checked on disk this session)

**Repo/git.** HEAD `9aa3c84a` (2026-06-06, pre-pivot). **8,667 uncommitted changes**: rewritten
`CLAUDE.md`/`README.md` (new mission), `.gitignore`, KB index, and **7,792 deletions** under
`data/local_labels/sweeps/`. No `sim/`, `eval/`, `labels/`, `benchmarks/` dirs. Test suite
(818 files) deleted pending selective restoration. No venv.

**Toolchain (re-verified live).** RTX 3090 Ti · driver 610.43.02 · NCU 2026.1.0 · nvcc 13.3
(`/opt/cuda`) + CUDA 13.2 (`/usr/local/cuda-13.2`) · system Python 3.14.6 (`.python-version`
pins 3.12; `uv` present) · **cargo 1.96.1 / rustc installed** (D6's Rust workspace is
immediately buildable).

**[Δ] Salvage S1–S5 has ALREADY RUN.** `data/annex/` (4.0 GB) is populated with
`SALVAGE_MANIFEST.md` dated 2026-07-02: S1 sm86 sweeps 894M · S2 sm120 duration 13M ·
S3 flashinfer_trace 911M · S4 harness_traces_sm86 2.1G (178 `*__sm86` dirs; ~9G non-sm86
remainder deliberately left) · S5 metric_extractor divergence (**verified layout-only — no
fold-in needed**) + 290-file calibration microkernel suite. The execution plan's §3 "run
salvage" step is therefore **mostly done**; what remains is the §7.7 data-safety proof, the
S6 skill promotion, and the baseline commit. Sources remain intact in the legacy trees.

**What imports/works today** (execution-plan audit, spot-re-verified): `arch_profiles`
loader/resolver; `metric_registry.json` **v2, 41 metrics** with `ncu_observation_contract`s;
`l1_cache_op_classifier` (the `_BARE_LDG_E_DEFAULTS_TO_CA` frozenset);
`cache_replay/sector_cache_replay.py`. Cards: `sm_86.json` fully populated +
lockstep-guarded; `sm_120-laptop.json` deliberately partial (sm_count=26 measured;
`l1_geometry`/NVML anchor/duration crosswalk = `research_gap`). No GB202 card.

**Honesty ledger (unchanged).** sm_86 L2 v5 head VALIDATED (geomean 6.4%, frozen sha
`d28665f6`); sm_86 L1 scored-not-gated; sm_86 cycles near-miss (0.158 vs 0.15 geomean,
n=13 cohort poverty); all sm_120 predictor lanes REFUTED (root cause: Blackwell load-op
classifier inert). These refutations were produced *under the retired non-root constraint*
— with NCU+root on the 5060 they convert to bounded measurement tasks (Phase 1D).

**Legacy trees (~28 GB, 4 independent git repos, verified):** codex spine 1,597 commits
(newest 2026-06-06) ⊃ Tree A 1,350 ⊃ Tree B 1,359 (sm_120 captures + `cache_replay`, only
home) · solbench_baseline 740 commits, **18 GB** (11 GB `.harness_traces`: 274 per-problem
dirs, 185 CUPTI activity.jsonl, 19 nvbit launches.jsonl — largest 8.0 GB —, 17 `.t2a`
sector-address traces).

---

## 2. Independent verdict: the pivot is sound, and the timing is favorable

1. **The gap is real and freshly re-verified (July 2026).** Accel-Sim's shipped configs still
   end at SM80_A100/SM86_RTX3070; independent MICRO'25 re-validation puts vanilla Accel-Sim at
   26.7–32.2% cycle MAPE on Ampere with a 543% worst case; measured slowdown 1.07×10⁷× (lavaMD
   >5 days). **Blackwell sm_120 tracing is broken upstream (accel-sim #385, still open);
   Hopper #380 closed unresolved.** No open simulator or model covers sm_120 at all.
2. **The "NCU metric vector" slot is still empty.** Nearest neighbors as of July 2026:
   Omniwise (AMD rocprof metrics, source-only LLM), PipeWeave ISCA'26 (latency-only, 6.1%),
   GCoM/AMALI (cycles/CPI, not profiler vectors), **[Δ] Sim-FA (arXiv:2605.00555, May 2026)** —
   a from-scratch *Hopper FlashAttention-pipeline* simulator, 5.7% MAPE vs H800: evidence the
   space is heating, but it is kernel-family-specific and Hopper-only; our generality +
   Blackwell + full-vector claims hold.
3. **The inherited substrate is unusually strong**: a validated 6.4% L2 sector-hit head (beats
   the best published NCU-validated cache model, profiner L2 15.8%), a working NVBit 1.8
   Accel-Sim-format tracer with Blackwell opcode tables, a 41-metric registry with observation
   contracts (the emission contract), an arch-card system with fail-closed research-gap typing,
   ~5 GB of schema-versioned NCU/trace evidence, and a 92-row + 18 + 235-problem workload stack.
4. **Benchmark timing**: FlashInfer-Bench (arXiv:2601.00227; 41 definitions / ~1,600 serving
   workloads; MLSys'26 competition on B200) and SOL-ExecBench (arXiv:2603.19173) both appeared
   in 2026 and **no one has used either as a performance-model validation suite yet**.
5. **Risk honestly restated**: D5's ≤10% cycles-MAPE target is *below* NVIDIA's own internal
   NVAS (17% MAE / 0.953 corr vs GV100 silicon, 955 benchmarks) and the UPC floor (13.45%
   Ampere / 17.41% Blackwell). ≤10% is a stretch goal; the ≤15% floor + full-vector headline
   is the publishable fallback already encoded in D5/§9 — this plan concurs.

---

## 3. [Δ] SOTA refresh — corrections & additions vs the 2026-07-02 survey

| Item | Correction / addition (all re-verified against primary sources) |
|---|---|
| **GB202 L2 = 128 MB** | RTX PRO 6000 Blackwell has the **full 128 MB** GB202 L2 (96 MB is the RTX 5090 cut). Whitepaper-confirmed. Update every "96 MB" mention; the L2-slice hash widening for `sim/mem` must assume 128 MB. |
| RTX PRO 6000 config | **188 SMs** (of 192 full-die), 24,064 cores, 96 GB GDDR7 ECC @ 512-bit, 1,792 GB/s, boost 2,617 MHz. Feeds `cards/sm_120-gb202.json`. |
| 5060 Laptop config | 26 SMs (matches measured card ✓), 8 GB GDDR7 128-bit **384 GB/s**, TGP-dependent boost 1,455–2,497 MHz. |
| **GeForce/PRO tensor asymmetry** | Consumer GeForce Blackwell halves FP32-accumulate tensor throughput; RTX PRO 6000 runs full rate. **This is a paper gift**: a *documented* same-ISA cross-SKU divergence the GB206-vs-GB202 experiment (D7) must reproduce — put it in the cross-SKU validation roster explicitly. |
| Blackwell datapath | INT32 unified onto both FP32 datapaths (128/SM/clk, contends with FP32) — differs from sm_86's 64+64 split; encode as a card field consumed by `sim/sm` pipes. |
| New competitor | **Sim-FA** (May 2026): Hopper FA-pipeline sim, 5.7% MAPE vs H800. Engage in related work; differentiate on generality (arbitrary SASS), Blackwell, NCU-vector output. |
| Accel-Sim status | v1.3.0 (Feb 2025) adds H100 *correlation plumbing only*; #380 (sm_90) closed-unresolved; **#385 (sm_120 tracing) open** — strengthens gap claim §1. |
| Reviewer bar (MICRO'25→) | Report **p90 + worst-case APE per metric class**, not aggregate means only (UPC exposed Accel-Sim's 543% tail; STEM+ROOT push statistical bounds). The eval harness (§6) must emit tails natively. |
| Case-study expectation | Every accepted model paper ships a "what does this enable" study (GCoM/GCScaler DSE, MDM DVFS, AMALI TC scaling). Our Phase-4 fidelity-knob ablation should be *framed as a DSE case study* (e.g., L2-slice/SM-count sweep between GB206↔GB202 cards predicting the measured SKU delta). |
| FIB facts pinned | fast_p vs FlashInfer-library baseline; CUDA-event timing 10 warmup/50 timed ×3 trials; trace dataset 26.4 GB on HF (annex holds a 911 MB serving-captured subset); MLSys'26 competition tracks: fused_moe / sparse_attention / gated_delta_net. |
| Motivation numbers | NCU replay overhead citations for the intro: full metric sets take dozens of passes (~46 reported); XSP measured >100× slowdown for replay-heavy metrics; counter access needs root since driver 418.43 (ERR_NVGPUCTRPERM). "Predicts what NCU would report, in seconds, anywhere" is quantifiable. |
| NCU clock-control default | Current NCU 2026.1 default is `--clock-control boost` (older: `base`) — already encoded in Rule 2; re-confirmed. |

---

## 4. Asset-disposition matrix (the extraction core)

Verdicts against the new goal. Destinations use the execution-plan layout (`sim/`, `labels/`,
`eval/`, `benchmarks/` renames happen in Phase 2, not Phase 0).

### 4.1 tools/ (24 subdirs)

| Asset | LOC | Verdict | Destination / action |
|---|---:|---|---|
| `local_labels/` (label factory + **registry v2, 41 metrics**) | ~61k | **ADAPT (core)** | → `labels/`. v2 capture protocol (1A); add stall-reason family `smsp__warp_issue_stalled_*` + SOL/tensor-pipe metrics; `deployability`→`model_io_role`; retire `build_sm120_*closure*`/`sm120_loop_claim_guard` non-root machinery. |
| `nvbit_tracer/` (tracer_tool + metric_extractor + isa_def + pb wire) | ~60k | **REUSE** | Sim trace front-end + fast count cross-check. Rebuild `.so` on host (M0.3); track NVBit Blackwell TMA bugs #162–164; S5 divergence verified layout-only — nothing to fold. |
| `cache_replay/` (sectored set-assoc L2 + falsification harness) | ~4.3k | **ADAPT (high)** | → `sim/mem` L2 slice semantics (32B sector/128B line, LRU/SRRIP/BIP, slice hash). De-hardcode sm_120-only cohort policy; widen hash for **128 MB** GB202 L2. |
| `cache_predictor/` (L1 walker + cache-op classifier) | ~6.8k | **ADAPT** | Seeds `sim/mem` L1 (needs warp-pool FIFO + finite-MSHR + HRD(32,128) + carveout capacity — F1–F3). Classifier's `bare_ldg_e_default` resolves empirically in 1D. Fix stale `_SM_COUNT_BY_ARCH["sm_120"]=50`→card 26 (Phase-0 tidy). |
| `hrd_profiler/` (two-granularity reuse distance) | ~3.1k | **REUSE** | Calibration instrument; add sm_120/GB202 rows to `device_properties.json`. |
| `pchase/` (latency/associativity/MSHR probes, %clock64) | ~0.9k | **REUSE (key)** | Fills card latency/geometry on all 3 GPUs (1D); the sm_120 L1-geometry gap closer. |
| `sass_control/` (control-bits extractor) | ~0.5k | **ADAPT (top cycles lever)** | Extend to full stall/yield/dependence-mask sidecar (1E) consumed by `sim/sm` — the UPC 13–17% MAPE lever. |
| `solar/` (SOL/roofline deriver + arch YAMLs) | ~19k | **REUSE** | Analytical floor baseline + SOL%-crosswalk; arch YAMLs cross-check cards. |
| `sol_execbench/` + `sol_baselines_scaffold/` | ~10k | **REUSE** | Eval corpus #1 runner (finish pkg rename). |
| `sol_baselines_harness/` (5 surfaces + 2nd cache predictor + calibration sweeps) | ~66k | **ADAPT/partial-retire** | Keep `metrics/`, `calibration/cache_prediction/*`, `solar_analysis.py`, `report.py` as eval/baseline material; retire the counter-free-surface framing; repurpose `b200_preflight.py` skeleton for GB202 cloud bring-up preflight. |
| `sm_86_predictor_accuracy_recovery_v2/` | ~16k | **SALVAGE-parts** | `publication_gate.py`, `t2_traffic_predictor.py`, cohort/split validators → `eval/`. v1 OBSOLETE. |
| `perturbation_harness/`, `sector_stream/`, `adaptive_sampler/`, `baseline_runners/` (Nugteren RDH) | ~3.9k | **REUSE/ADAPT** | Ablation harness · trace preprocessing · trace-budget bounding · prior-work baseline respectively. Fix `KERNEL_SLUG` wiring (Q10 tidy). |
| `divergence_proxies/`, `sharing_locality/`, `order_projections/` | ~1.7k | **RETIRE-leaning** | Non-root-motivated feature proxies; keep individual proxies only as sim cross-checks. |
| `cloud/` (non-root container preflight) | ~1.9k | **RETIRE/repurpose** | Non-root gating dead; reuse provisioning/preflight skeleton for the GB202 instance. |
| `sol_kb/` (FTS5 router) | ~4.7k | **REUSE-AS-IS** | + new `ncu-metrics/` subcorpus (below). |
| `arch_capabilities.{py,json}`, `BLOCKED_ON_3090TI.md` | — | **RETIRE** | Superseded by `arch_profiles/` / historical ledger. |

### 4.2 workloads/ + data/

| Asset | Verdict | Notes |
|---|---|---|
| `simbench_bw/` 92-row roster (46 micro/29 AI-op/14 block/3 pipeline) + build/trace machinery | **REUSE** | → `benchmarks/`. Calibration tier = M2 gate corpus. |
| `simbench_bw/` predictors (`cache_predictor.py` 6.2k, v5 head, `timing_cycle_predictor`, `gcom_lite_cycles_evaluator`) | **DEMOTE→calib/baselines** | Per D1: in-sim calibration layers + insurance headline + baselines. |
| `comparison_runner.py` (3.8k, 8-tuple join, bootstrap CI) | **GENERALIZE** | → `eval/` per-(kernel,metric,arch) harness; **[Δ] add p90/worst-case tail reporting per §3**. |
| `cupti_activity_collector.cu` + dual-source admission | **RETIRE (D8)** | Archive, don't delete — cheap insurance if a large non-NCU timing sweep is ever needed. |
| `portable_lcd/` (18) | **REUSE** | M0 smoke corpus. |
| `sol_execbench_baselines/` (205/235 ported; Quant 2/33, FIB 25/26) | **REUSE** | Finish or type-exclude the ~30 unported cells (Phase 4). |
| `data/local_labels/` curated frozen (heads/cohorts/closures, 8.9 MB) | **KEEP byte-stable** | Regression anchors; v5 head = calibration-layer template + baseline. |
| `data/annex/` (4.0 GB, S1–S5) | **DONE** | Extend per §5. |

### 4.3 Knowledge base (`.claude/knowledge/`, 355 cards / 14 subcorpora) + references/

- **HIGH-value, promote/keep active**: `simulator/` (6 nvasim/Rust-DES cards — **promote from
  "historical" to active**, S6), `ptx/` (13), `cuda/` (7), `sass/` (2), `blackwell/` (4 —
  incl. the authoritative GB202 capability card; **[Δ] update its L2 to 128 MB** and add the
  GeForce-vs-PRO tensor-rate asymmetry), `patterns/` (27), `problems/` (235 SOL dossiers —
  strip "out-of-scope non-root" blocks via scripted pass).
- **Rewrite (dead-thesis)**: `measurement/nonroot-cloud-contract.md`,
  `measurement/nvarchsim-vs-this-project.md` (asserts "we are NOT building a simulator" — now
  false). Keep `clock64-globaltimer-dvfs.md`, `nvbit-injection-playbook.md`.
- **NEW subcorpus `ncu-metrics/`**: distill `references/NVIDIA_Manuals/ncu_guide/` — metric
  grammar (`unit__subunit_quantity.rollup.postfix`), section taxonomy (SOL / compute / memory /
  scheduler / warp-state stall families / occupancy), replay semantics (kernel replay,
  cache-flush, clock-control), per-metric determinism class (feeds the repeatability study).
- **references/** (458 MB: 176 PDFs, 64-paper active set, NVIDIA manuals, patents, the two
  founding design notes) — already carried in the suite; legacy copies are byte-identical
  duplicates (no action). `draft_new_gpu_simulator.md` + `draft_phase1_execution_time_accuracy.md`
  both present ✓.

---

## 5. Salvage manifest — status + [Δ] extensions

**S1–S5 executed 2026-07-02 ✓** (verified on disk; see §1). **S6** (promote
`designing-gpu-simulator` skill + `simulator/` KB to active) and **S7** (lts anomaly pointer)
remain Phase-0 items. **S8** (prediction-core reference reconcile) is a Phase-2 reconcile.

**[Δ] New salvage items discovered this session (adjacent home dirs, outside the 4 trees):**

| # | Asset | Source · size | Action |
|---|---|---|---|
| **S9** | **SOLAR NCU probe CSVs on BOTH cards** — `ncu_hmma_fp16/cublaslt_bf16/dram/l2` CSVs, probe kernels, `props.txt`, nvidia-smi snapshots for rtx3090 + rtx5060; upstream NVlabs SOLAR clone w/ `configs/arch/*.yaml` | `~/SOLAR/{measurements,solar-rtx3090,solar-rtx5060}/` · <5 MB | Copy → `data/annex/solar_probe_csvs/`. Independent cross-check for card peak-rate fields + SOL crosswalk. |
| **S10** | SOL-ExecBench judge/eval framework + 236 problem-spec markdowns (incl. FIB 210–235) + worked rmsnorm optimization triple (gt/opt/cutile) | `~/SolarBench_tier0/` (+ `~/SolarBench_rmsnorm_*`) · specs 2.2 MB | Diff vs suite's vendored `tools/sol_execbench/` + KB `problems/`; fold only deltas (judge/scoring pieces the suite lacks). |
| **S11** | **ICCD manuscript** ("Counter-Free GPU Observability"): `paper/main.tex`, `refs.bib`, figures, cross-card SOL-ExecBench evaluation numbers (222/235 kernels measured on 3090Ti+5060) | `~/ICCD_paper_non_root_gpu_profiler_backup/` · **46 GB total — selective only** | Copy `paper/` (tex/bib/figures) → `references/papers/ours/iccd_counter_free/`. The bib + related-work + measurement-methodology prose seed Phase 5; the identifiability framing is a citable companion narrative ("what traces alone can/can't see" → why the simulator needs which fidelity). **Do NOT copy the 46 GB data trees** — verify overlap with annex first. |
| **S12** | Duplication check: `~/NVIDIA_Sims_Predict_NCU/` (72-kernel-roster-era ancestor of the codex line) + `~/NVIDIA_Sims/` (wall_clock_sims ancestor; simulator-validation framing docs, `gpu-simulator-non-privileged-research.md` 62 KB) | home dirs | No bulk salvage (superseded by codex spine). Extract only: the 62 KB simulator-validation methodology doc → `references/papers/notes/`. Confirm no unique label data (spot-check sweep dirs). |

---

## 6. Refactor blueprint — endorsed, with refinements

Target layout per execution plan §4 (`sim/` Rust workspace · `labels/` · `eval/` ·
`benchmarks/` · `arch_profiles/` + GB202 card · `data/annex/` · retained calibration
instruments). Refinements from this audit:

1. **`sim/emit` contract tests are the registry**: every emitted metric must satisfy its
   `ncu_observation_contract` (counting unit, rollup, numerator/denominator, op-classes) —
   including NCU sector-accounting semantics and cold-cache-per-kernel (`--cache-control all`)
   defaults. Registry extension (1B) enumerates the stall family from the ncu_guide distill
   (§4.3) rather than ad hoc.
2. **Eval harness emits tails natively**: NRMSE + correlation per metric class, MAE for
   never-zero metrics, **p50/p90/worst-case per kernel family**, leave-kernel-family-out +
   leave-SKU-out splits, sealed holdout touched once.
3. **The DSE case study is first-class (Phase 4)**: swap card inputs GB206→GB202 and show the
   engine predicts the measured cross-SKU deltas (SM count 26→188, L2 32→**128 MB**, GDDR7
   384→1,792 GB/s, **FP32-acc tensor-rate asymmetry**) — this doubles as the controlled
   generalization experiment (D7) and the reviewer-expected "what does this enable" section.
4. **Card schema additions** for `sim/sm`: `int32_datapath` (split vs unified),
   `tensor_fp32_acc_rate_factor` (1.0 PRO / 0.5 GeForce), `l0_icache`, GDDR6X/GDDR7 timing
   block — all microbench-derived or whitepaper-cited (never NCU-fitted; Rule 1).
5. **Current runtime/evaluation scope:** NCU labels are the accuracy baseline;
   record exact runtime/profile evidence for this repository's simulator and use
   only in-repo analytical/frozen references. The formerly proposed Accel-Sim,
   UPC, and PPT-GPU executable rigs are inactive historical context. Sim-FA also
   remains related-work only.

---

## 7. Phased schedule — endorsed with an amended Phase-0 checklist

Phases 1–5 and gates M1–M4 per the execution plan §5 stand. **Phase 0 (Jul 6→25, 3090 Ti)**
amended to reflect verified state:

- [ ] 0.1a Create working branch (never master). **Data-safety proof**: script-verify each of
      the 7,792 deleted `sweeps/` paths exists byte-identical under
      `data/annex/sm86_sweeps_2026H1/` or the codex tree; emit `plans/sweeps_deletion_safety_proof.json`.
- [ ] 0.1b ~~Run salvage S1–S5~~ **done 2026-07-02** ✓ → run **S6 + S9–S12** (this doc §5),
      append to `SALVAGE_MANIFEST.md`; then land the pivot baseline commit
      (CLAUDE.md/README/.gitignore/KB-index + sweep deletions + plans).
- [ ] 0.2 `uv venv` (Python 3.12 per `.python-version`), install `requirements.txt`,
      smoke-import every §4.1 REUSE/ADAPT module under the venv.
- [ ] 0.3 Guard/skill refit: `eval/guards/test_targets_not_features_guard.py`; relax
      `cross_arch_precondition` validators to tri-state; promote simulator skill/KB (S6);
      rewrite the two dead-thesis measurement cards (§4.3).
- [ ] 0.4 Selective test restoration (~60–90 of 424 from git history): ArchProfile lockstep,
      frozen-head/cohort byte-identity, trace wire-format, importer/schema, comparison-runner
      join. Green them.
- [ ] 0.5 `sim/` Rust workspace skeleton (crates: engine/sm/mem/trace/emit/calib/ffi; `cargo
      build` green) + `eval/baselines/` stubs.
- [ ] 0.6 Phase-0 tidies: `_SM_COUNT_BY_ARCH["sm_120"]` 50→26; `KERNEL_SLUG` wiring;
      GB202 KB card L2 96→**128 MB**.
- [ ] **GATE M0**: run the §7 soundness checklist (execution plan) on this host; write
      `plans/sm86_soundness_report.md`. **No 5060/GB202 work before M0 green — and no 5060
      migration before M2 green** (operator's "finish 3090 Ti first" directive).

---

## 8. Risk register — two updates

| Risk | Update |
|---|---|
| ≤10% cycles target below literature floor | **Recalibrated context**: NVIDIA's internal NVAS reports 17% MAE vs silicon across 955 benchmarks. ≤10% would beat the vendor's own tool; treat as stretch, ≤15% floor as the publishable bar (unchanged D5 mechanics, sharpened framing). |
| Space heating up | **Sim-FA (May 2026)** joins UPC/PipeWeave as evidence. Mitigation unchanged (speed + NCU-vector + Blackwell + 3-SKU silicon validation are still unclaimed), but argues for **not slipping Phase 2**. |

---

## 9. Operator decisions recorded this session (2026-07-06, evening)

| # | Question | Decision | Consequence |
|---|---|---|---|
| D9 | Phase-0 go-ahead? | **Plan-only continues.** The D3 hold stands; no scaffold/salvage/commit/run until an explicit future go. | Phase-0 checklist (§7) is ready-to-execute on approval; nothing in this session mutated the repo beyond `plans/` documents. |
| D10 | 5060 laptop + cloud GB202 availability | **Both later / not yet available.** | Phases 0–2 are **sm_86-exclusive** (consistent with the finish-3090Ti-first directive). All Blackwell capture (1C 3-SKU repeatability legs, 1D 5060 geometry/classifier resolution, Phase 3 bring-up, D7 cross-SKU) is scheduled *contingent on hardware arrival*. **NEW SCHEDULE-CRITICAL RISK:** Phase 3 (Nov 1–Dec 20) is the latest safe Blackwell-arrival window for MICRO 2027; if neither sm_120 device is available by ~Nov 2026, invoke the D1 insurance path (sm_86-validated simulator + hybrid heads headline, Blackwell as future work) or shift to HPCA 2028. Track monthly alongside the CFP check. |
| D11 | Extended salvage S9–S12 | **Approved.** | Fold into `data/annex/` + `references/` during step 0.1b when Phase 0 is greenlit; append to `SALVAGE_MANIFEST.md`. |
| D12 | Execution scope (round 2, same evening) | **Phase 0 + M0 gate approved and started.** Report before Phase 1 capture. | Supersedes D9's plan-only hold. 5060 migration remains blocked behind M0+M2 regardless. |
| D13 | Baseline commit | **Branch + commit** once the data-safety proof passes (per D4). No remote push. | |
| D14 | GB202 cloud provisioning | **After M2 is green.** Re-raise at the M2 report. | Phase-3 Nov-window risk stays on the monthly CFP-check watchlist. |

*Provenance: 4 background research agents this session (new-repo map 44 tool-uses; legacy
inventory 40; home-dir scout 22; web SOTA survey with 14-claim adversarial verification pass)
+ direct verification of toolchain, git state, annex manifest, salvage paths, cards, plans,
CLAUDE.md, design notes, and the SOTA dossier F1–F10 map. External claims carry URLs in the
agent reports; key primary sources: accel-sim ISCA'20 + repo issues #380/#385, UPC MICRO'25
(arXiv:2503.20481), NVArchSim HPCA'21, NVIDIA RTX Blackwell consumer + PRO whitepapers,
FlashInfer-Bench (arXiv:2601.00227), SOL-ExecBench (arXiv:2603.19173), Sim-FA
(arXiv:2605.00555), Nsight Compute 2026.1 docs.*
