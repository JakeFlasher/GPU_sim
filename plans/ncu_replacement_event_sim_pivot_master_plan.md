# MASTER PLAN — Pivot to a Fast, NCU-Accurate Event-Driven GPU Performance Simulator

> **CURRENT ADR OVERLAY — FOUNDING PLAN / HISTORICAL BODY (2026-07-18).** This
> document records the 2026-07-02 founding pivot; its body is preserved and does
> not override the current execution plan or status. ADR 0001
> (`CONTINUE_PRESERVE_PINNED`) permits fresh Task7/H3 continuation outside RLCR
> while the historical attempt stays frozen, zero-credit, and reuse-forbidden.
> ADR 0002 (`CYCLES_ONLY_BINDING`) supersedes full-vector/fallback-headline
> language below: `exec_cycles_gpc_max` / `gpc__cycles_elapsed.max` is the sole
> publication-binding output (≤10% target, >10% and ≤15% floor-only, >15%
> failure). Seven implemented secondary descriptors are reported and ungated;
> cycles remain `required_unimplemented` pending the aggregation-semantics
> study and are operating-state- and protocol-bound, not DVFS-stable. Every
> bare M0/M1/M2 milestone below belongs to the **founding simulator milestone
> namespace**, not the active capture-campaign M0–M7. Current authority is
> [`micro2027_ncu_sim_execution_plan.md`](micro2027_ncu_sim_execution_plan.md),
> [`../docs/STATUS.md`](../docs/STATUS.md), and
> [`../docs/decisions/`](../docs/decisions/README.md).

Date: 2026-07-02 · Repo: `cross_arch_gpu_profiler_suite` · Status: **founding document for the new project direction**

> Produced from a 4-agent deep-research pass (new-repo inventory, legacy-tree salvage
> audit, simulator/ML-model SOTA survey with adversarial verification, NCU/benchmark
> methodology survey with adversarial verification) plus direct session verification
> of the load-bearing repo artifacts. External claims below carry their sources;
> repo claims carry file paths. Where something could not be verified it is marked
> **[unverified]**.

---

## 0. Operator decisions (recorded 2026-07-02)

| Decision | Choice |
|---|---|
| **Goal** | Replace cycle-level architectural simulators (Accel-Sim / GPGPU-Sim class, ~10^6× slowdown) with a **fast, highly-accurate model of modern NVIDIA GPUs (Ampere + Blackwell)** that predicts **Nsight Compute (NCU) performance metrics** for real kernels (FlashInfer-Bench, SOL-ExecBench). The non-root deployment story is retired as the project thesis. |
| **Vehicle** | **Event-driven simulator first** — NVArchSim-inspired (HPCA 2021) trace-driven discrete-event simulator (Rust), per `references/papers/notes/draft_new_gpu_simulator.md`. The trace-driven analytical + learned-head lane (the "hybrid model") is retained as (a) calibration layers, (b) baselines, (c) fallback headline if the simulator's accuracy lags. |
| **Anchor venue** | **ISCA 2027** — expected abstract ~mid-Nov 2026 (ISCA 2026 pattern: abstract Nov 10 / paper Nov 17; official CFP not yet posted — re-check monthly). Fallback: MICRO 2027 (~Apr 2027). |
| **Hardware** | **3 GPUs, 2 architectures, 3 SKUs**: RTX 3090 Ti (GA102, sm_86) · RTX 5060 laptop (GB206, sm_120) · cloud RTX PRO 6000 (GB202, sm_120). **NCU access on all three.** No B200/sm_100 in scope for paper 1. |
| **Root access** (added 2026-07-02) | **Full root on the local GPUs.** Clock locking (`nvidia-smi -lgc/-lmc`), persistence mode, and counter-permission setup (`NVreg_RestrictProfilingToAdminUsers=0` or the R610 capabilities system) are allowed and preferred on the 3090 Ti and the 5060 laptop. This retires the old Rule 2 entirely and downgrades the laptop-DVFS risk: lock clocks first, keep the nvbench-style noise gates as belt-and-braces (on laptops a lock can behave as a *limit* under thermal pressure). |
| **Accuracy bar** (added 2026-07-02) | The simulator targets **cycle-level accuracy of its outputs** (`gpc__cycles_elapsed.max` as the primary target, per-metric NCU vector alongside) via an event-driven engine — cycle-accurate *results*, not cycle-ticking *implementation*. |
| **Execution order** (added 2026-07-02) | **sm_86 first, to completion.** The current dev host **is the 3090 Ti**. Every building block (contract, salvage, tests, tracer, collector, label factory, calibration instruments, simulator core) must be built and soundness-verified on sm_86 **before** the repo migrates to the 5060 laptop. The migration gate is the Phase-2 M2 milestone (§5) plus a green soundness report (§8). |
| **NCU posture** | NCU is now a **first-class ground-truth and validation instrument on every target**. The old cloud-NCU ban, quarantine guards, and CLAUDE.md Rule 5 ("no simulator revival") are **obsolete and must be rewritten** (§4). The *only* surviving prohibition: NCU-derived values must never leak into model **inputs** (targets-not-features — ML hygiene, enforced by the existing `no_deployability_leakage_guard` machinery, repurposed). |

---

## 1. Thesis and novelty claims (grounded in the verified SOTA landscape)

**Thesis.** An event-driven, trace-driven simulator of Ampere/Blackwell SMs and memory
hierarchy, calibrated per-architecture from microbenchmarks and validated per-metric
against NCU on silicon, can reproduce the NCU metric vector (cycles, sector hit rates,
traffic counts, occupancy, SOL throughputs) at 100–1000× the speed of cycle-level
simulation — making it a practical replacement for Accel-Sim-class tools on modern
AI kernels.

**Verified open gaps this project occupies** (adversarially verified July 2026):

1. **No open simulator has a validated timing model past sm_86.** Accel-Sim's tested
   configs end at SM80_A100 / SM86_RTX3070; no SM89/SM90/SM100/SM120 config exists in
   the repo tree. Its Ampere-class error is ~25–32% (UPC re-measurement), speed 12.5
   KIPS (days per workload). GPGPU-Sim 4.x is in maintenance mode.
2. **No published work predicts full NCU metric vectors.** The nearest neighbor is
   Omniwise (arXiv:2506.20886) — AMD-only, source-only, 10–12 metrics, random splits.
3. **No model targets NCU sector hit-rate metrics** (`lts__t_sector_hit_rate.pct`,
   `l1tex__t_sector_hit_rate.pct`). Best NCU-validated cache model: profiner
   (CGF 2022; L1 3.73% / L2 15.81% MAPE, one Turing GPU). **This repo's frozen sm_86
   v5 head already beats that L2 bar (6.4% geomean)** — a shipping head start.
4. **No sm_120 (consumer/workstation Blackwell) performance model exists at all.**
   Only microbench dissections (arXiv:2507.10789 RTX 5080; arXiv:2512.02189 B200;
   Chips & Cheese GB202).
5. **No academic work predicts NCU Speed-of-Light percentages.**
6. **No GPU analog of the Weaver/McKee counter-determinism studies exists** — an NCU
   counter-repeatability characterization is free, load-bearing novelty (§6.3).

**Closest competitors to engage head-on in the paper:**
- **PipeWeave (ISCA 2026, arXiv:2601.14910)** — hybrid analytical+ML, 6.1% kernel
  latency error over 11 GPUs. *Latency-only, operator-level.* We differentiate on
  output vector (full NCU diagnostics), arbitrary-SASS-kernel generality, Blackwell.
- **UPC "Dissecting/Modeling Modern GPU Cores" (MICRO 2025, arXiv:2503.20481)** —
  13.98% cycle MAPE vs real RTX A6000 within Accel-Sim (cycle-level, slow); artifact
  ships Blackwell example traces → **this space is heating up; speed matters.**
  Their reverse-engineered facts (SASS control bits, CGGTY issue, no operand
  collector, PRT coalescing, register-file cache) are *inputs to our SM model*.
- **NVArchSim (HPCA 2021)** — the design template (event-driven, hybrid trace+execution,
  variable fidelity, single-threaded, 40–250× over GPGPU-Sim). Proprietary; we are the
  open, Blackwell-capable instantiation.
- **GPUMech→MDM→GCoM→AMALI cascade** — the rhetorical backbone: mechanistic models
  calibrated to one generation explode on the next (GPUMech 13%→162%; GCoM 10%→127%
  on real A100 LLM kernels), and simulator-referenced "accuracy" hides 1.6–3× more
  error vs silicon (MDM 13.9%→40%, GCoM 10%→24.4%). Our per-arch microbench
  calibration + silicon-only validation is the answer.

**Framing sentence for the paper:** *a measurement instrument and design-iteration
tool: it predicts what NCU would report on real hardware in seconds instead of days —
and, unlike black-box predictors, its event-driven core retains what-if capability.*

---

## 2. What we inherit (asset map)

### 2.1 From `cross_arch_gpu_profiler_suite` (already in place)

| Asset | Path | Role in new project |
|---|---|---|
| **NCU metric registry v2** — 41 metrics with `ncu_observation_contract` (counting unit, rollup, numerator/denominator, op-classes), per-arch availability, deployability | `tools/local_labels/metric_registry.json` + `.py`, `docs/metric-registry.md` | **The target-metric specification.** Becomes the simulator's *output contract*: each emitted metric must satisfy the registry's observation contract. Extend per §5.4. |
| **Label pipeline** — capture→parse→repeatability-gate→admit | `tools/local_labels/` (~120 modules: `capture_*_labels.py`, `parse_ncu_metrics.py`, `repeatability_aggregator.py`, `import_label.py`, …) | **The label factory.** Rework capture flags per the 2026 NCU protocol (§6.1); scale to 3 GPUs and 41+ metrics. |
| Frozen sm_86 v5 L2 head (isotonic-on-analytical-prior, 6.4% geomean) + 74-cell label corpus | `data/local_labels/sm_86_v5_head.json`, `label_admitted.json` | Baseline + calibration-layer template; the corpus seeds training/validation splits. |
| **NVBit 1.8 tracer** (Accel-Sim-format SASS/address dumper, protobuf wire, allocation registry, per-arch opcode tables incl. Blackwell) | `tools/nvbit_tracer/tracer_tool/`, `isa_def/` | **The simulator's trace front-end.** Known upstream Blackwell bugs to track: NVBit #148/#152/#162–164 (TMA memtrace). |
| **metric_extractor** (counter-free per-warp histograms, sector efficiency, bank conflicts, TCGen05/TMA/DSMEM classes) | `tools/nvbit_tracer/metric_extractor/` | Fast feature source + cross-check of trace-derived counts. Reconcile the ~10-line divergence vs the solbench copy (§3.4). |
| **cache_replay** — sectored set-associative L2 replay engine (32B sector/128B line, slices, hashing, calibration harness) | `tools/cache_replay/` (28 files) | **Becomes the L2 slice model inside the simulator's memory hierarchy.** Engine is arch-general; de-hardcode the sm_120-only cohort policy. |
| L1 walker + cache-op classifier (+ the `_BARE_LDG_E_DEFAULTS_TO_CA` Blackwell gap) | `tools/cache_predictor/` | Seed of the simulator L1 model — needs the dossier's warp-pool interleaving + finite-MSHR + HRD(32/128) rebuild (§5.3). |
| Reuse-distance profiler, p-chase geometry probe, sector_stream, perturbation harness, SASS control extractor | `tools/hrd_profiler/`, `tools/pchase/`, `tools/sector_stream/`, `tools/perturbation_harness/`, `tools/sass_control/` | **Per-arch calibration instruments** — fill ArchProfile cards (latencies, geometry, MSHRs, set hashing) on all 3 GPUs. |
| **ArchProfile cards + loader + alias resolver** | `arch_profiles/` | The simulator's per-arch configuration input. Add `cards/sm_120-gb202.json`; fill the `research_gap` fields via NCU+microbench (§5.1). |
| **SOLAR** (roofline/SOL-bound deriver) + SOL-ExecBench evaluator + 205/235 ported baseline kernels + 235 problem dossiers | `tools/solar/`, `tools/sol_execbench/`, `workloads/sol_execbench_baselines/`, `.claude/knowledge/problems/` | **Evaluation corpus #1** + the SOL-section ground-truth crosswalk (predicted SOL% vs SOLAR analytical bounds vs NCU SOL metrics). |
| simbench_bw kernel roster (92 kernels: 46 microbench L1-tier → 29 AI-op L2-tier → 14 model-block L3 → 3 pipeline L4) + portable_lcd (18) | `workloads/simbench_bw/`, `workloads/portable_lcd/` | **Calibration + validation corpus** with existing build/run/trace machinery. |
| comparison_runner (7-field join key, per-arch splits, bootstrap CIs, typed verdicts) | `workloads/simbench_bw/comparison_runner.py` | Generalize into the **per-(kernel, metric, arch) evaluation harness** (§6.2). |
| **KB: 353 cards / 14 subcorpora** + FTS5 router + skills | `.claude/knowledge/`, `.claude/skills/`, `tools/sol_kb/` | Keep. Promote `simulator/` cards from "historical only" to active; add `ncu-metrics/` subcorpus distilled from `references/NVIDIA_Manuals/ncu_guide/` (§7). |
| **References corpus**: 176 PDFs (64-paper active set), NVIDIA manuals mirror incl. `ncu_guide/`, patents, the two seed drafts | `references/` (458 MB — *already carried*, despite MERGE_PROVENANCE §2 saying otherwise) | The literature + spec base. `references/papers/notes/draft_new_gpu_simulator.md` is the simulator's founding design note. |
| SOTA dossier (13-agent synthesis: failure-mode→fix map, model classes, targets) | `plans/research/sota_nonroot_perf_model_next_phase_dossier.md` | §3/§5 of it remain the microarchitectural fix list for the L1/cycles models regardless of vehicle. |

### 2.2 What the suite does **not** have (build/extend list)

- A simulator core (event engine, SM/warp-scheduler model, DRAM timing) — **the main build**.
- Broad *captured* NCU labels: only 2 cache-hit metrics + timing cohorts are live; the
  41-metric registry is authored but unpopulated. No stall-reason (`smsp__pcsamp_*`),
  tensor-pipe, or SOL-throughput labels captured yet.
- A GB202 (RTX PRO 6000) ArchProfile card; Blackwell L1/L2 geometry, latency tables,
  clock model (all typed `research_gap`).
- A SASS **control-bits** static extractor (stall/yield/dependence counters via
  cuobjdump) — the single biggest cycles-accuracy lever (UPC MICRO 2025).
- FlashInfer-Bench harness integration (the repo has the 26 FIB problem ports, not the
  external `flashinfer-bench` Definition/Solution/Trace runner).
- Baseline-simulator comparison rigs (Accel-Sim SM86_RTX3070, UPC MICRO'25 artifact,
  PPT-GPU) — previously forbidden by CLAUDE.md Rule 5.
- A test suite (818 files deliberately deleted; recoverable from git history — §3.5).

### 2.3 Salvage from the legacy trees (`3090ti_5060laptop_cross_study/`)

| # | Asset | Source | Action |
|---|---|---|---|
| S1 | **Bulk NCU sweep evidence** — 846 MB, 133 sweep runs, 18,070 JSON, real `.ncu-rep` (NCU 2026.1.0, CUDA 13.2, driver 595.71.05), NVML covariate time-series, ~28 labeled sm_86 slugs | `non_root_gpu_profiler_codex/data/local_labels/` | Copy into a new **data annex** (`data/annex/sm86_sweeps_2026H1/`, gitignored or LFS; keep the curated frozen artifacts as-is). This is training/validation data + the raw material for the repeatability study. |
| S2 | **sm_120 duration corpus** (CUPTI dual-source + NVML, flash_attn/gemm/swiglu rosters) | `non_root_gpu_profiler_5060_branch/data/local_labels/duration/sm_120/`, `sm120_frozen_roster_l1l2_pilot*` | Copy into the annex (`data/annex/sm120_duration_2026H1/`). Only Blackwell wall-clock evidence in existence here. |
| S3 | **FlashInfer trace corpus** — 901 MB serving-captured workloads (gqa_paged/ragged, mla_paged/ragged, moe, sampling) | `solbench_baseline/data/flashinfer-trace/blob/workloads/` | Copy to `data/annex/flashinfer_trace/`. Feeds FlashInfer-Bench workload instantiation without re-downloading the 26 GB HF dataset. |
| S4 | `.harness_traces/` — 11 GB live NVBit trace dumps | `solbench_baseline/scripts/baselines_harness/.harness_traces/` | **Sample, don't bulk-copy**: keep a curated ~1–2 GB subset of kernels that overlap the eval roster; regenerate the rest (traces are reproducible; disk is not free). |
| S5 | metric_extractor divergence (`metric_extractor.cu`, `classifier.h` differ ~10 lines vs merged copy; solbench copy has unit tests) | `solbench_baseline/nvbit_tracer/metric_extractor/` | 3-way diff → fold fixes + tests into `tools/nvbit_tracer/metric_extractor/`. |
| S6 | **Test suite** (370 tracked + 424 historical `test_*.py` in codex history; suite history preserved in the merged repo's `.git`) | `git log --all -- '**/test_*.py'` in the suite | Selective restoration per §3.5 — do **not** bulk-restore; most tests enforce the old contract. |
| S7 | sm_120 NCU anomaly record (`lts` pct out-of-range vendor escalation) | `non_root_gpu_profiler_codex/data/local_labels/vendor_escalation/` | Carry into the repeatability-study evidence (§6.3) — it is a documented NCU-on-Blackwell label hazard. |
| S8 | Legacy per-tree `.humanize/bitlesson.md` variants | all four trees | Already unified into `.claude/knowledge/bitlesson*.md` (MERGE_PROVENANCE §11) — **no action**; verify the codex 1006-line version is the one carried. |
| **Not salvaged** | compiled `.so`/binaries, `.humanize/rlcr/` loop state, Tree-A/Tree-B duplicates of `references/` (byte-identical to the suite's) | — | Rebuild binaries per host; loop state is host-local by contract. |

---

## 3. Phase 0 — Repo re-founding (target: ~2 weeks, finish by Jul 18)

### 3.1 Rewrite the project contract (`CLAUDE.md`)

Replace Rules 1–5 with the new charter:

- **New Rule 1 (targets-not-features).** NCU/hardware-counter values are prediction
  *targets and validation references only*; they must never enter model inputs
  (features, traces, calibration constants derived from the *same cell's* labels).
  Enforced by repurposing `tools/local_labels/no_deployability_leakage_guard.py`
  (the `deployability` field becomes `model_io_role: target | input | either`).
- **New Rule 2 (pinned measurement protocol).** Every NCU capture pins:
  NCU version, `--clock-control` (explicit — the default changed `base`→`boost` in
  NCU 2026.1), `--cache-control`, `--replay-mode`, driver/CUDA versions, and the
  device's clock/thermal state (NVML sidecar). Labels lacking the pin block admission.
- **New Rule 3 (silicon-only validation).** Accuracy claims are made against real-GPU
  NCU labels, never against another simulator. Per-arch evidence statuses survive,
  extended to the third device: `Offline-verified` / `Verified-live-sm_86` /
  `Verified-live-sm_120-gb206` / `Verified-live-sm_120-gb202` / `Cross-arch-verified`.
- **New Rule 4 (typed honesty, unchanged in spirit).** Typed fail-closed exclusions,
  slug-disjoint splits, sealed holdouts, falsification artifacts as first-class
  outcomes — carried forward verbatim from the old discipline.
- **Deleted:** old Rule 1 (cloud NCU ban), Rule 2 (no driver-root ops — clock locking
  via `nvidia-smi -lgc` is now *allowed and preferred* where supported; note laptop
  caveat §6.1), Rule 3 (no privileged CUPTI), Rule 5 (no simulator revival).

### 3.2 Repo restructure (additive first; nothing validated moves in Phase 0)

```
cross_arch_gpu_profiler_suite/
├── sim/                    # ★ NEW — the event-driven simulator (Rust workspace)
│   ├── engine/             #   event queue, actor scheduling, clocks
│   ├── sm/                 #   sub-cores, warp scheduler (CGGTY + SASS control bits), pipes
│   ├── mem/                #   L1 (sectored+MSHR), L2 slices (from tools/cache_replay logic), DRAM
│   ├── trace/              #   NVBit protobuf ingestion (bridges tools/nvbit_tracer wire format)
│   ├── emit/               #   NCU-metric emission layer (contract = metric_registry v2)
│   └── calib/              #   per-arch calibration fits (microbench → ArchProfile)
├── labels/                 # renamed/refocused tools/local_labels (the label factory)
├── eval/                   # generalized comparison_runner + baselines/ (accel-sim, upc-sim, ppt-gpu rigs)
├── benchmarks/             # sol_execbench + flashinfer_bench integration + simbench + portable_lcd rosters
├── arch_profiles/          # unchanged + new cards
└── (tools/, workloads/, data/, references/, .claude/ as today until Phase 2 migration)
```

Phase 0 only creates `sim/` scaffolding and `eval/baselines/`; renames happen in
Phase 2 once tests exist.

### 3.3 Guard & skill refit

- Retire `test_no_root_or_ncu_static_guard.py` semantics; write its successor
  `eval/guards/test_targets_not_features_guard.py` (static scan: no `ncu`/label-path
  imports inside `sim/` except `sim/emit` contract tests).
- Relax the sm_120 pin guards (`sm120_cohort_commit_pin_guard.py`,
  `cross_arch_precondition` hardwired-false validators) to tri-state.
- Update skills: `designing-gpu-simulator` becomes the **active** design skill
  (drop "historical only / forbids revival" language); `validating-nonroot-gpu-measurement`
  → `validating-gpu-measurement` (NCU-inclusive protocol); strip the 8-rule
  "Out of scope" blocks from KB problem cards via a scripted pass.

### 3.4 Execute salvage S1–S5 (§2.3) — one scripted, manifest-logged copy pass.

### 3.5 Test-suite selective restoration

From git history, restore **only**: ArchProfile lockstep tests, frozen-head/cohort
regression guards, trace wire-format contract tests, importer/schema validators,
comparison-runner join tests (≈60–90 of the 424). Everything enforcing the old
quarantine stays dead. Re-establish the recorded baseline (`MERGE_PROVENANCE.md` §6)
for the restored subset, then grow the suite with the simulator.

---

## 4. Phase 1 — Label factory + architecture characterization (Jul 18 → Aug 22)

**1A. NCU capture protocol v2** (labels/):
- Pin **NCU 2026.1.0** — the version installed on the 3090 Ti host *and* the version
  that produced the legacy label corpus (continuity). Upgrade to 2026.2.x only as a
  deliberate, recorded protocol bump. Toolchain reality on this host (verified
  2026-07-02): driver **610.43.02** (legacy sweeps were 595.71.05 — record the drift
  in every new sidecar), `nvcc` 13.3 default at `/opt/cuda` with the CUDA 13.2
  toolkit also installed at `/usr/local/cuda-13.2` — pin the toolkit per artifact.
- **Root is available on local GPUs**: lock clocks via `nvidia-smi -lgc` (+`-lmc`
  where supported), enable persistence mode, and set up non-root counter access via
  the R610 capabilities system. Explicit `--clock-control` on every capture
  (never rely on the default — it changed `base`→`boost` in NCU 2026.1). On the
  5060 laptop lock clocks too, but keep the nvbench-style noise gates (≥10 samples,
  0.5% rel-stddev stop, throttle recovery delays, kernels ≥20 µs where possible) —
  on laptops a lock can act as a limit under thermal pressure.
- Primary timing label: `gpc__cycles_elapsed.max` (DVFS-stable); `gpu__time_duration.sum`
  secondary. Metric tiers to capture per cell:
  **T1** deterministic counts (`sass__`/`smsp__inst_executed`, requests, sectors) →
  **T2** cache/traffic (`lts__`/`l1tex__` hit rates + sector families, `dram__sectors_*`) →
  **T3** cycles/duration → **T4** derived (occupancy, `sm__throughput.*`,
  `gpu__compute_memory_throughput.*` SOL pair, stall-reason sampling as diagnostic).
- Capture on **all three GPUs** over: simbench 92-kernel roster + portable_lcd 18 +
  the 205 ported SOL-ExecBench baselines + FIB kernels. Budget: the existing
  `label_driver.sh` machinery parallelizes per-device; expect ~2–3 weeks calendar
  with the laptop as the bottleneck.

**1B. Counter-repeatability study (§6.3)** — runs *inside* 1A at marginal cost:
same cells × {replay modes} × {clock-control base/boost/none} × {cache-control all/none}
× N repeats, on all 3 SKUs → per-metric noise floors that become the accuracy gates'
denominators **and** a standalone paper section (no prior GPU study exists).

**1C. Architecture cards**: run `tools/pchase/` + latency/geometry/MSHR microbenches +
NCU cross-checks on all 3 GPUs → fill `arch_profiles/cards/sm_86.json` gaps, complete
`sm_120-laptop.json` (L1 geometry, clock anchors), author `sm_120-gb202.json`.
Resolve the Blackwell load-classifier question empirically (cuobjdump inventory +
cache-op microbench + NCU hit-rate differential → set `bare_ldg_e_default` from
measurement; this unblocks OPEN_RESEARCH_CHALLENGES #1–3 with hours of work now that
NCU is allowed on the 5060).

**1D. SASS control-bits extractor** (`tools/sass_control/` extension): parse stall
counts, dependence-counter masks, yield, reuse bits from `cuobjdump -sass`/nvdisasm
for sm_86 + sm_120 → per-instruction static sidecar consumed by `sim/sm`.

---

## 5. Phase 2 — Simulator core v0 (Aug → early Oct)

Design per `references/papers/notes/draft_new_gpu_simulator.md` + NVArchSim principles
(event-based, flow-controlled queues, no retry, single-threaded first, variable
fidelity) + UPC MICRO'25 reverse-engineered facts:

- **5.1 Engine**: binary-heap event queue; SM-actor mini-loop batching (no
  per-instruction events); target ≥3–10 M events/s. Rust; C++/protobuf shim for the
  existing NVBit wire format.
- **5.2 SM model** (fidelity MEDIUM): 4 sub-cores; CGGTY-style issue; dependence
  stalls read from **SASS control bits** (1D) instead of simulated scoreboards —
  the 13–17% cycles-MAPE lever; unified INT32/FP32 pipes for sm_120 (verified GB202
  fact); tensor path = HMMA (sm_86) / block-scaled `mma.sync` + OMMA/QMMA (sm_120) —
  **no tcgen05/TMEM** (sm_100-only; out of scope).
- **5.3 Memory hierarchy**: L1 sectored + finite MSHR + warp-pool interleaving +
  carveout-aware capacity (the dossier's F1–F3 fix list; geometry from cards);
  L2 slice actors ported from `tools/cache_replay/sector_cache_replay.py` semantics
  (32B sector fills, hashed slice mapping — widen for GB202's large L2; note the
  L40/5090 non-uniform L2 latency finding arXiv:2606.22588); DRAM GDDR6X/GDDR7
  bandwidth-latency model (fidelity HIGH only if error attribution demands it).
- **5.4 NCU-emission layer** (`sim/emit`): every emitted metric implements the
  registry v2 observation contract — including NCU's *sector accounting semantics*
  (the known ~15 pp hazard Accel-Sim's authors attribute to profiler accounting) and
  isolated-execution assumption (NCU flushes caches per pass → simulate cold-cache
  per kernel by default, matching `--cache-control all` labels).
- **5.5 Calibration layer** (`sim/calib`): per-arch fitted latencies/throughputs from
  1C; optional **monotone low-capacity residual heads** (the v5 isotonic pattern) on
  top of simulator outputs, fit slug-disjointly — never truth-fitted per-cell.
- **Milestone gate M2 (end Sept):** on the sm_86 microbench tier — T1 counts p50 ≤1%,
  L2 hit rate ≤ v5's 6.4%, cycles geomean ≤20% slug-disjoint, ≥100× measured speedup
  vs Accel-Sim SM86_RTX3070 on shared traces.

---

## 6. Phase 3 — Cross-arch + evaluation harness (Sept → Oct, overlapping)

- **6.1 Blackwell bring-up**: GB206 + GB202 cards drive the same engine; per-metric
  validation; the GB206-vs-GB202 same-ISA/different-SKU axis is the controlled
  generalization experiment (only card inputs change).
- **6.2 Evaluation harness** (`eval/`): generalize `comparison_runner.py` to
  per-(kernel, metric, arch) verdicts. **Reporting rules** (Accel-Sim template +
  SIGMETRICS'18 critique): NRMSE + correlation per metric class; MAE only for
  never-zero metrics; error distributions per kernel family, never aggregate-only;
  splits = leave-kernel-family-out + leave-SKU-out; sealed holdout touched once.
  Baselines: Accel-Sim (SM86_RTX3070 — the only config overlapping our hardware),
  UPC MICRO'25 artifact (cycles), PPT-GPU (dormant + known memory-model bugs —
  document, don't just assert), profiner-style + in-repo Nugteren-2014 cache
  baselines, roofline/SOLAR analytical floor, and the frozen v5 heads.
- **6.3 Repeatability study** write-up from 1B: per-metric determinism classes
  (deterministic / clock-sensitive / replay-sensitive), noise floors, the sm_120
  `lts` out-of-range anomaly (S7) — justifies every accuracy gate.
- **6.4 Benchmarks**: SOL-ExecBench 235 (finish the ~30 unported Quant/FIB cells or
  type-exclude them honestly); FlashInfer-Bench harness integration — run Definitions/
  Workloads via its isolated runner on our 3 GPUs with our own NCU ground truth
  (official numbers are B200; ours is a *new dataset release*: NCU-profiled
  FlashInfer-Bench + SOL-ExecBench on 3 SKUs). Add a small Rodinia subset for
  comparability with prior-work numbers. Optional demo: a "virtual runner" emitting
  predicted Traces (`fast_p` pre-screening) into the flashinfer-bench flow.

---

## 7. Phase 4 — Paper (Oct → ISCA'27 deadline ~mid-Nov)

**Skeleton:** (1) the gap: simulators end at sm_86, 10^6× slow, error explodes
cross-generation (GPUMech→MDM→GCoM→AMALI cascade); (2) design: event-driven,
trace-driven, control-bits-informed, per-arch card-calibrated; (3) the NCU-emission
contract + label protocol + repeatability study (methodological novelty);
(4) per-metric silicon validation on 3 SKUs / 2 archs; (5) speed (events/s, wall-clock
vs Accel-Sim/UPC); (6) FlashInfer-Bench + SOL-ExecBench case studies; (7) what-if
retention (fidelity knobs) — distinguishing us from black-box predictors.

**Risk register (each with a mitigation already in the plan):**
NVBit Blackwell TMA bugs (#162–164) → typed-exclude TMA-heavy cells, track upstream;
laptop clock instability → **downgraded** (root clock-lock available) but keep
cycles-first + noise gates (1A) since laptop locks can degrade to limits under
thermals; replay-induced label
noise → repeatability study + NRMSE floors (1B/6.3); cycles accuracy stalls above
20% → fall back to headline = NCU metric *vector* (counts/cache/SOL) + cycles as
secondary, or pivot headline to the hybrid lane heads; ISCA CFP shifts → re-check
monthly, MICRO'27 fallback; UPC group attaches an sm_120 timing model first →
our differentiators remain speed (event-driven vs cycle-level) + NCU-vector output +
3-SKU silicon validation.

**KB upkeep (continuous):** new `ncu-metrics/` subcorpus (distill `references/NVIDIA_Manuals/ncu_guide/`),
promote `simulator/` cards to active, per-arch geometry cards from 1C, BitLesson
entries for every measurement hazard encountered.

---

## 8. Immediate next actions (this week — executing on the 3090 Ti host)

1. Land this plan + the rewritten `CLAUDE.md` + guard/skill refit (§3.1–3.3). ← in progress 2026-07-02
2. Run the salvage script (S1–S5) with a manifest; verify `du` budgets.
3. Restore the ~60–90 keeper tests; green them.
4. **sm_86 soundness gate** (must be green before any 5060 migration): pin the
   toolchain (NCU 2026.1.0 / driver 610.43.02 / NVBit 1.8), verify non-root or
   root-assisted `ncu` counter access + `-lgc` clock lock on this host, rebuild
   `tracer_tool.so` + the CUPTI collector, smoke a portable_lcd kernel end-to-end
   (build → trace → NCU capture → label admit), and verify every reusable component
   imports/runs (arch_profiles loader, metric_registry, hrd_profiler, cache_replay,
   solar). Results recorded in `plans/sm86_soundness_report.md`.
5. Start 1A/1B capture on the 3090 Ti (it has the most existing roster coverage) while
   the GB202 cloud instance is provisioned. Migrate to the 5060 laptop only after the
   soundness gate + M2 are green.
