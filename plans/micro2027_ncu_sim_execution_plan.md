# EXECUTION MASTER PLAN — Fast, NCU-Accurate Event-Driven GPU Simulator (MICRO 2027)

Date: 2026-07-06 · Repo: `cross_arch_gpu_profiler_suite` · Status: **authoritative execution plan**
Supersedes the strategy in `plans/ncu_replacement_event_sim_pivot_master_plan.md` (2026-07-02) where
they differ; that document remains the SOTA-survey provenance record. This plan folds in three locked
operator decisions (2026-07-06) and a full ground-truth audit of the current repo + the four legacy
trees performed this session.

---

## 0. Locked decisions (2026-07-06)

| # | Decision | Choice | Consequence for this plan |
|---|---|---|---|
| D1 | **Headline vehicle** | **Simulator-primary.** The event-driven, trace-driven simulator is the paper's headline novelty. The validated analytical + low-capacity-ML "hybrid heads" are retained as (a) *in-simulator calibration layers* and (b) a *shippable insurance result* if the simulator's cross-generation accuracy lags. | The `sim/` core is the main build; the hybrid heads are demoted from "the result" to "a calibration layer + fallback headline." |
| D2 | **Anchor venue** | **MICRO 2027** — abstract ≈ **Mar 31 2027**, full paper ≈ **Apr 7 2027** (per the stable MICRO single-round late-March/early-April cycle; MICRO 2026 was Mar 31 / Apr 7 2026). **~9 months runway.** Official MICRO 2027 CFP not yet posted → re-confirm monthly. Fallbacks: HPCA 2028 (abstract ≈ Aug 2027) then ISCA 2028. | Full-scope schedule is feasible: event-driven sim + Ampere+Blackwell across 3 SKUs + repeatability study + FIB/SOL eval. |
| D3 | **This session** | **Plan document only.** Produce this plan; do **not** scaffold, salvage, commit, or run anything until approved. | Execution (Phase 0) begins only on explicit go-ahead. |

**Ratified open decisions (2026-07-06, round 2 — locked by the operator):**

| # | Decision | Choice | Consequence |
|---|---|---|---|
| D4 | **Baseline commit policy** | **Branch + annex the sweeps.** Work on a new branch (never `master`); prove the 7,792 deleted `data/local_labels/sweeps/` files are duplicated in the codex legacy tree / suite HEAD, move a curated copy to `data/annex/` (gitignored), then land the pivot baseline commit. | §7.7 data-safety proof is a hard precondition to any commit. |
| D5 | **Cycles accuracy bar** | **Target ≤10% cycles-MAPE** (below the ~13–17% UPC MICRO'25 literature floor), via heavier per-arch microbench calibration + high-fidelity SASS control-bit stall modeling. **≤15% (UPC-class) is the minimum-publishable floor / insurance bar.** | M2/M3 cycle gates tighten to the ≤10% target; §9 carries the below-floor risk; more Phase-1D/1E calibration effort. |
| D6 | **Simulator core language** | **Rust** (tagged-enum zero-cost dispatch; C++/protobuf FFI shim for NVBit trace ingestion). | `sim/` is a Rust workspace from Phase 0. |
| D7 | **GB202 (cloud RTX PRO 6000) scope** | **In scope, headline.** GB206-vs-GB202 same-ISA / different-SKU is the controlled cross-SKU generalization experiment; provision the cloud instance during Phase 1. | 3-SKU validation is a paper-1 contribution, not a stretch; adds a cloud bring-up to Phase 3. |
| D8 | **Timing instrument** | **NCU-only; CUPTI/dual-source retired.** With NCU + root on every target, `gpc__cycles_elapsed.max` (primary, operating-state-bound) + `gpu__time_duration.sum` (secondary) are the timing ground truth *and* the simulator's timing targets. Cycle comparisons are bound to the recorded operating state, capture protocol, and core:memory clock relationship; no boost or DVFS invariance is assumed. The CUPTI dual-source `duration_ns` collector was scaffolding for the retired non-root deploy thesis; it produces no cycles and no metric the simulator validates against. Optional CUDA-events fast sampler only if a large non-NCU timing sweep is ever needed. | Drops the CUPTI collector from the M0 native-build gate and the measurement stack; timing validation is silicon NCU (Rule 3). |

**Inherited operator decisions (still binding, from the 2026-07-02 founding plan):**
- **Hardware:** 3 GPUs / 2 arches / 3 SKUs — RTX 3090 Ti (GA102, sm_86) · RTX 5060 laptop (GB206, sm_120) · cloud RTX PRO 6000 (GB202, sm_120). No B200/sm_100 in paper 1.
- **Full root + NCU on ALL three targets** (reaffirmed 2026-07-06): clock/memory-clock locking (`nvidia-smi -lgc/-lmc`), persistence mode, and counter-permission setup are **allowed and preferred**, local *and* cloud. The old non-root/counter-less deployment thesis, the cloud-NCU ban, the NCU quarantine, and "no simulator revival" are all **retired**.
- **The one surviving prohibition — targets-never-features.** NCU / hardware-counter values are prediction *targets and validation references only*; they must never enter simulator inputs, trace features, or calibration constants fitted on the *same cell* being scored. Enforced by the repurposed leakage guard (`deployability` → `model_io_role: target | input | either`).
- **Execution order — sm_86 first, to completion.** The current dev host **is the RTX 3090 Ti**. Every building block must be soundness-verified on sm_86 (§7 gate) before the repo migrates to the 5060 laptop or the cloud GB202.
- **Accuracy bar — cycle-accurate *outputs*, not cycle-ticking *implementation*.** Primary target `gpc__cycles_elapsed.max` (operating-state-bound under the recorded capture protocol), with the full NCU metric vector alongside; event-driven engine at 100–1000× cycle-sim speed.

---

## 1. Thesis and novelty (MICRO 2027 framing)

**Thesis.** An open, event-driven, trace-driven simulator of Ampere/Blackwell SMs and memory hierarchy,
calibrated per-architecture from microbenchmarks and validated per-metric against NCU on silicon, can
reproduce the **NCU metric vector** (cycles, L1/L2 sector hit rates, traffic/sector counts, occupancy,
Speed-of-Light throughputs) at **100–1000×** the speed of cycle-level simulators — a practical
replacement for Accel-Sim-class tools on modern AI kernels, and the first such tool that reaches
consumer/workstation Blackwell.

**Verified open gaps we occupy** (from the 2026-07-02 adversarial SOTA survey; provenance retained there):
1. **No open simulator has a validated timing model past sm_86.** Accel-Sim tops out at SM80_A100 / SM86_RTX3070 (~25–32% Ampere error, ~12.5 KIPS → days/workload); GPGPU-Sim 4.x is in maintenance.
2. **No published work predicts full NCU metric *vectors*.** Nearest neighbor is Omniwise (AMD-only, source-only, 10–12 metrics, random splits).
3. **No model targets NCU *sector hit-rate* metrics.** Best NCU-validated cache model: profiner (CGF'22; L1 3.73% / L2 15.81% MAPE, one Turing GPU). Our frozen sm_86 v5 L2 head is already **6.4% geomean** — a shipping head start that becomes an in-sim calibration layer.
4. **No sm_120 (consumer/workstation Blackwell) performance model exists at all** — only microbench dissections (RTX 5080 arXiv:2507.10789; B200 arXiv:2512.02189; Chips&Cheese GB202).
5. **No academic work predicts NCU Speed-of-Light percentages.**
6. **No GPU analog of the Weaver/McKee counter-determinism studies** — an NCU counter-repeatability characterization is free, load-bearing novelty (§6).

**Competitors to engage head-on:** PipeWeave (ISCA'26; hybrid, 6.1% latency, 11 GPUs — *latency-only, operator-level*; we differ on full-vector output + arbitrary-SASS generality + Blackwell); UPC "Modeling Modern GPU Cores" (MICRO'25; 13.45% Ampere / 17.41% Blackwell cycle MAPE *inside cycle-level Accel-Sim, slow* — their reverse-engineered facts are **inputs to our SM model**); NVArchSim (HPCA'21; the design template, proprietary — we are the open, Blackwell-capable instantiation); the GPUMech→MDM→GCoM→AMALI cascade (mechanistic models calibrated to one generation explode on the next: GCoM 10%→127% on A100 LLM kernels — our per-arch microbench calibration + silicon-only validation is the answer).

**Paper's one-sentence positioning:** *a measurement instrument and design-iteration tool — it predicts what NCU would report on real hardware in seconds instead of days, and, unlike black-box predictors, its event-driven core retains what-if capability.* The **what-if / variable-fidelity** capability is the primary differentiator vs. the hybrid-ML competitors (PipeWeave, NeuSight, SynPerf) and the reason D1 chose simulator-primary.

---

## 2. Ground-truth state (audited this session, 2026-07-06)

**The pivot is documented but unexecuted.** The 2026-07-02 founding plan, a rewritten `CLAUDE.md`
(new mission) and `README.md` exist — but:
- **No `sim/`, `eval/`, `labels/`, `benchmarks/` directories** — the entire simulator scaffold is unbuilt. *(snapshot at re-founding; superseded for build state by `plans/phase2_sim_core_progress.md` — sim/ now holds the Phase-2 v0 implementation.)*
- **No `plans/sm86_soundness_report.md`** — the sm_86 soundness gate has never run.
- **Last commit is 2026-06-06** (all pre-pivot "compliance/evidence" work). The pivot docs + a **7,792-file working-tree cleanup** (bulk `data/local_labels/sweeps/` deletions + a `references/` reflow) are **uncommitted**.

**What is verified working today (imports/loads clean against system Python):**
- `arch_profiles` loader + resolver (`resolve_arch_id("NVIDIA GeForce RTX 5060 Laptop GPU") → "sm_120"`).
- `tools/local_labels/metric_registry.json` — **registry v2, 41 metrics**, every entry carrying an `ncu_observation_contract` (`counting_unit / rollup / numerator / denominator / op_classes / unsupported_op_behavior`), 8 derived entries carrying a `composition` formula, all 41 carrying `per_arch_availability`.
- `tools/cache_predictor/l1_cache_op_classifier.py` (the `_BARE_LDG_E_DEFAULTS_TO_CA` frozenset), `tools/cache_replay/sector_cache_replay.py`.
- Arch cards: `sm_86.json` fully populated + lockstep-guarded byte-identical to live source; `sm_120-laptop.json` deliberately partial (sm_count=26 measured, `bare_ldg_e_default="unknown_typed_unavailable"`, `cache_layers=["L2"]`, `l1_geometry`/`nvml_reference_operating_point`/`duration_crosswalk` = `research_gap`).

**Concrete gaps/flags this plan absorbs:**
- **Toolchain/env:** system Python is **3.14.6**; `.python-version` pins **3.12**; **no venv exists**; `requirements.txt` present (numpy 2.4.5 / scipy 1.17.1 / scikit-learn 1.8.0 / jsonschema 4.26.0 / pytest 9.0.3 / protobuf 7.34.1). Host verified 2026-07-02: driver 610.43.02, NCU 2026.1.0, nvcc 13.3 (`/opt/cuda`) + CUDA 13.2 (`/usr/local/cuda-13.2`). **Bring-up item in Phase 0.**
- **Registry has no stall-reason family** (`smsp__warp_issue_stalled_*`), yet the simulator design names stall reasons as a required validation counter → **registry-extension task (§5.1B)**.
- **Source constant `cache_predictor._SM_COUNT_BY_ARCH["sm_120"]` still = 50** (guessed); superseded by the card's measured **26** but not fixed in source → tidy in Phase 0.
- **The 7,792 uncommitted deletions must be proven safe** (present in the legacy codex tree and/or the annex) *before* any commit → §7 safety gate.

---

## 3. Salvage & extraction manifest (validated on disk this session)

Four legacy trees, ~28 GB total, all independent git repos. **`non_root_gpu_profiler_codex` is the
canonical spine** (newest HEAD 2026-06-06). One scripted, manifest-logged copy pass; nothing validated
in the suite is overwritten (set-difference overlay).

| # | Asset | Source (validated path · size) | Action | New home |
|---|---|---|---|---|
| S1 | **Bulk NCU sweep evidence** — 133 runs, 18,070 JSON, **5,140 real `.ncu-rep`** (NCU 2026.1.0) | `non_root_gpu_profiler_codex/data/local_labels/sweeps/` · **846 MB** | Copy (gitignored/LFS). Training/validation data + raw material for the repeatability study (§6). | `data/annex/sm86_sweeps_2026H1/` |
| S2 | **sm_120 duration corpus** (CUPTI dual-source + NVML) + 2 frozen L1/L2 pilot rosters | `_5060_branch/data/local_labels/duration/sm_120/` (198 JSON) + `sm120_frozen_roster_l1l2_pilot*` · ~11 MB | Copy. Only Blackwell wall-clock evidence in existence. | `data/annex/sm120_duration_2026H1/` |
| S3 | **FlashInfer trace corpus** — 2,611 `.safetensors` workload blobs (gqa/mla/moe/sampling) | `solbench_baseline/data/flashinfer-trace/blob/workloads/` · **901 MB** | Copy. Feeds FIB workload instantiation without re-downloading 26 GB from HF. | `data/annex/flashinfer_trace/` |
| S4 | `.harness_traces/` live NVBit dumps (429 problems: `trace.json` + `static/{sass,ptx,cubin,resource-usage}`) | `solbench_baseline/scripts/baselines_harness/.harness_traces/` · **11 GB** | **Sample, don't bulk-copy.** Keep ~1–2 GB overlapping the eval roster; regenerate the rest. Drop the co-located rebuildable ninja/`.o`/`.so`/`.cubin`/`.pyc`. | `data/annex/harness_traces_sample/` |
| S5 | **NVBit `metric_extractor`** (unique; Tier-2 per-warp histograms, TCGen05/TMA/DSMEM split, sector efficiency, bank conflicts) **+ its unit tests + calibration microbenchmarks** (`test_classifier.cc`, `tests/calibration*/`) | `solbench_baseline/nvbit_tracer/metric_extractor/` · 193 MB (source is small; bulk is build output) | 3-way diff → fold code + **tests** into the suite. Beware the unrelated name-collision Python pkg `scripts/baselines_harness/metric_extractor/`. | `tools/nvbit_tracer/metric_extractor/` |
| S6 | **Simulator design scaffolding** — `designing-gpu-simulator` skill (`SKILL.md`, `_eval/`) + `.claude/knowledge/simulator/` (`nvasim-architecture.md`, `nvasim-dev-patterns.md`, Rust DES/perf/testing notes) | `non_root_gpu_profiler/.claude/skills/` + `.claude/knowledge/` (tree "A" only) | Copy + **promote to active** (drop "historical/forbids-revival" language). Seeds the `sim/` design. | `.claude/skills/`, `.claude/knowledge/simulator/` |
| S7 | **sm_120 NCU `lts` anomaly** — filed NVIDIA bug: `lts__t_sector_hit_rate.pct` returns **115–217%** on GB206 | `non_root_gpu_profiler_codex/data/local_labels/vendor_escalation/…/REPORT.md` | Carry into the repeatability-study evidence — a first-class Blackwell label hazard. | `data/annex/` + §6 |
| S8 | **Prediction core reference** — `gcom_lite_cycles_evaluator.py` + schema, `cupti_activity_collector.cu/.h`, `monotonic_calibration_head.py`, `canonical_features.py` | `non_root_gpu_profiler_codex/workloads/simbench_bw/` | Reference for the in-sim GCoM-lite calibration layer + CUPTI collector; most already mirrored in the suite — reconcile, don't duplicate. | (existing suite paths) |

**Explicitly NOT salvaged** (regenerable / duplicate, ~9 GB): `solbench_baseline/.venv/` (5.3 GB), `references/` duplicated 4× (~1.78 GB — keep the suite's one copy), the four `.git/` histories (~1.6 GB — the suite already preserved the codex spine), all compiled `.so/.o/.cubin`, `__pycache__`/`.pytest_cache`. Salvage script emits a `du`-budgeted manifest and refuses to overwrite any suite file.

---

## 4. Target repository architecture (the refactor)

Additive first. Nothing validated moves until the restored tests are green (Phase 0). Final layout:

```
cross_arch_gpu_profiler_suite/
├── sim/                         # ★ NEW — event-driven simulator (Rust workspace)
│   ├── engine/                  #   binary-heap→timing-wheel event queue; tagged-enum dispatch; POD event record
│   ├── sm/                      #   4 sub-cores; CGGTY issue; dependence stalls READ from SASS control bits
│   ├── mem/                     #   L1 (sectored + finite-MSHR + warp-pool interleave + carveout capacity),
│   │                            #   L2 slice actors (cache_replay semantics), DRAM (GDDR6X/GDDR7)
│   ├── trace/                   #   NVBit protobuf ingestion (C++/protobuf shim → Rust)
│   ├── emit/                    #   NCU-metric emission layer — contract = metric_registry v2 observation contracts
│   ├── calib/                   #   per-arch fitted latencies/throughputs; optional monotone low-capacity heads
│   └── ffi/                     #   C++ shim for the existing NVBit wire format
├── labels/                      # the NCU label factory (from tools/local_labels): capture→gate→admit, registry
├── eval/                        # generalized comparison_runner → per-(kernel,metric,arch) harness + baselines/
│   ├── baselines/               #   accel-sim (SM86_RTX3070), upc-micro25 artifact, ppt-gpu, profiner, RDH, roofline
│   └── guards/                  #   test_targets_not_features_guard.py (successor to the NCU static guard)
├── benchmarks/                  # sol_execbench + flashinfer_bench runner integration + simbench + portable_lcd
├── arch_profiles/               # unchanged + new cards/sm_120-gb202.json; research_gap fields filled in Phase 1
├── data/annex/                  # salvaged bulk (gitignored/LFS); data/local_labels/ = curated frozen only
├── tools/                       # calibration instruments retained: pchase, sass_control, hrd_profiler,
│                                #   cache_replay, cache_predictor, nvbit_tracer, solar, sol_execbench, baseline_runners
├── references/  docs/  .claude/ # one references copy; docs + KB (promote simulator/ + new ncu-metrics/ subcorpus)
```

`labels/`, `eval/`, `benchmarks/` are **renames** of `tools/local_labels/`, a generalized
`comparison_runner`, and the SOL/FIB/simbench rosters — executed in Phase 2 once tests exist, not in Phase 0.

---

## 5. Phased execution schedule (backward from MICRO 2027)

Milestone gates are **hard**: a phase does not start until the prior gate is green on the named device.

### Phase 0 — Re-founding + sm_86 soundness gate · **Jul 6 → Jul 25** (host: 3090 Ti)
- **0.1** Verify the 7,792 deletions are safe (present in codex tree/annex), run salvage S1–S8 with a `du`-budgeted manifest, then **land the pivot baseline commit** (CLAUDE.md/README + curated cleanup) on a branch — never on `main`/`master` directly.
- **0.2** Environment: create the `uv` venv pinned to Python 3.12 (or re-pin `.python-version` to 3.14 after a compat check), install `requirements.txt`, confirm all reusable modules import under the venv.
- **0.3** Guard/skill refit: write `eval/guards/test_targets_not_features_guard.py` (static scan: no NCU/label-path imports inside `sim/` except `sim/emit` contract tests); relax the sm_120 `cross_arch_precondition` hardwired-false validators to tri-state; promote `designing-gpu-simulator` + `simulator/` KB to active.
- **0.4** Test-suite selective restoration from git history: **only** ArchProfile lockstep, frozen-head/cohort regression, trace wire-format, importer/schema, and comparison-runner join tests (~60–90 of 424). Re-establish the recorded baseline (207 passed / 3 skipped / 187 subtests) for the restored subset.
- **0.5** Create `sim/` (Rust workspace skeleton, `cargo` builds empty crates) + `eval/baselines/` stubs.
- **GATE M0 = the sm_86 soundness report (§7).** Must be green before Phase 1 capture at scale and before *any* 5060/GB202 migration.

### Phase 1 — Label factory v2 + arch characterization + repeatability study · **Jul 26 → Sep 15**
- **1A. NCU capture protocol v2** (`labels/`): pin NCU 2026.1.0; explicit `--clock-control` on every capture (default changed base→boost in 2026.1); lock clocks via `-lgc`(+`-lmc`), persistence on; NVML covariate sidecar per launch; record driver/CUDA/NVBit versions + clock-lock state. Metric tiers T1 counts → T2 cache/traffic → T3 cycles/duration → T4 derived (occupancy, SOL pair, stall-reason sampling). Primary timing label `gpc__cycles_elapsed.max`.
- **1B. Registry extension:** add the **stall-reason family** (`smsp__warp_issue_stalled_*`) + any missing SOL/tensor-pipe metrics to registry v2 with observation contracts; migrate `deployability` → `model_io_role`.
- **1C. Counter-repeatability study** (marginal cost inside 1A): cells × {replay modes} × {clock-control base/boost/none} × {cache-control all/none} × N repeats, on all 3 SKUs → per-metric noise floors (the accuracy gates' denominators) **and a standalone paper section**. Folds in the S7 `lts` 115–217% Blackwell anomaly.
- **1D. Architecture cards:** `tools/pchase/` + latency/geometry/MSHR microbenches + NCU cross-checks on all 3 GPUs → fill `sm_86.json` gaps, complete `sm_120-laptop.json` (L1 geometry, clock anchors), author `sm_120-gb202.json`. **Resolve the Blackwell load-classifier empirically:** cuobjdump inventory + cache-op microbench + NCU hit-rate differential on the 5060 → set `bare_ldg_e_default` from measurement (unblocks the open challenges now that NCU+root are allowed).
- **1E. SASS control-bits extractor** (`tools/sass_control/` extension): parse stall counts, dependence-counter masks, yield, reuse bits from cuobjdump for sm_86 + sm_120 → per-instruction static sidecar consumed by `sim/sm`. **This is the single biggest cycles-accuracy lever (UPC MICRO'25: 13.45% Ampere / 17.41% Blackwell).**
- **GATE M1:** sm_86 full metric-vector labels admitted for the simbench/portable_lcd/SOL rosters; repeatability noise floors published; sm_86 + sm_120-laptop cards complete; control-bit sidecar produced for sm_86.

### Phase 2 — Simulator core v0 on sm_86 · **Sep 1 → Oct 31** (overlaps late Phase 1)
Design per §6. Build order = design-note phases A→E.
- **GATE M2 (migration gate):** on the sm_86 microbench tier — T1 counts p50 ≤ **1%**; L2 sector hit rate ≤ **v5's 6.4%** geomean; cycles geomean **≤10% target** (**≤15%** UPC-class minimum-publishable floor, per D5) slug-disjoint; ≥ **100×** measured speedup vs Accel-Sim SM86_RTX3070 on shared traces; targets-not-features guard green. **Only after M2 does the repo migrate to the 5060 laptop — and only on explicit operator go-ahead.**

### Phase 3 — Blackwell bring-up + cross-SKU + evaluation harness · **Nov 1 → Dec 20**
- **3A.** GB206 + GB202 cards drive the *same* engine; per-metric validation. The **GB206-vs-GB202 same-ISA/different-SKU axis is the controlled generalization experiment** (only card inputs change) — a clean MICRO story.
- **3B. Evaluation harness** (`eval/`): per-(kernel, metric, arch) verdicts. Reporting rules (Accel-Sim template + SIGMETRICS'18 critique): **NRMSE + correlation per metric class; MAE only for never-zero metrics (cycles, inst counts); error distributions per kernel family, never aggregate-only; splits = leave-kernel-family-out + leave-SKU-out; sealed holdout touched once.** Baselines: Accel-Sim SM86_RTX3070, UPC MICRO'25 artifact (cycles), PPT-GPU (document its memory-model bugs), profiner-style + in-repo Nugteren-2014 RDH, roofline/SOLAR analytical floor, frozen v5 heads.
- **GATE M3:** ≥ one Blackwell SKU passes the per-metric bars; cross-SKU generalization quantified.

### Phase 4 — Case studies + ablations + full eval · **Dec 21 → Feb 15**
- FlashInfer-Bench harness integration (Definitions/Workloads via its isolated runner on our 3 GPUs with our own NCU ground truth → **a new dataset release: NCU-profiled FIB + SOL-ExecBench on 3 SKUs**); finish/type-exclude the ~30 unported SOL-ExecBench Quant/FIB cells; small Rodinia subset for prior-work comparability; fidelity-knob ablations (the what-if story); optional "virtual runner" demo emitting predicted FIB Traces.
- **GATE M4:** all headline claims reproduced end-to-end from a clean checkout; artifact-evaluation-ready.

### Phase 5 — Writing + internal review + artifact · **Feb 15 → Mar 31 (abstract) / Apr 7 (paper)**
- Codex/Gemini adversarial reads; sealed-holdout single touch; artifact packaged.

**Buffer:** ~3 weeks of slack distributed across gates; the laptop is the schedule bottleneck (thermal locks) — front-load its capture in Phase 1 once M0 is green.

---

## 6. Simulator design (grounded in the design note + the F1–F10 fix list)

From `references/papers/notes/draft_new_gpu_simulator.md` (NVArchSim-inspired) + the UPC MICRO'25
reverse-engineered facts + the repo's own failure→fix map (`plans/research/sota_…dossier.md` §3):

- **6.1 Engine.** Rust. Binary-heap event queue → hybrid timing-wheel + min-heap; **tagged-enum event dispatch** (no virtual dispatch on the hot path); compact POD event record `{time, seq, target_id, kind, payload_ref}`; pure next-event clock advance. **Per-SM actor mini-loop** batches several issue cycles per wake (NOT one event per warp pick) to avoid queue flooding. Throughput target **3–10 M events/s → 10–30 M optimized.**
- **6.2 SM model** (fidelity MEDIUM). 4 sub-cores; CGGTY issue; **dependence stalls read from SASS control bits (Phase 1E), not simulated scoreboards** — the 13–17% cycles-MAPE lever [F6]. Unified INT32/FP32 pipe for sm_120 (verified GB202 fact); tensor path = HMMA (sm_86) / block-scaled `mma.sync` + FP8/FP6/FP4 (sm_120). **No tcgen05/TMEM** (sm_100-only; out of scope).
- **6.3 Memory hierarchy.** L1 = sectored + **finite MSHR** + **warp-pool FIFO interleaving** + **carveout-aware capacity** [F1–F3, Nugteren HPCA'14 warp-pool 6.4–8.3% L1 MAE; HRD(32,128) two-granularity reuse distance]; geometry from the per-arch cards [F2, Mei&Chu p-chase]. L2 = slice actors ported from `tools/cache_replay/sector_cache_replay.py` semantics (32 B sector fills, hashed slice mapping — **widen the IPOLY-style hash for GB202's large L2**; note the L40/5090 non-uniform-L2-latency finding). DRAM = GDDR6X/GDDR7 bandwidth-latency (fidelity HIGH only if error attribution demands it).
- **6.4 NCU-emission layer** (`sim/emit`). Every emitted metric implements the registry-v2 observation contract, **including NCU's sector-accounting semantics** (the ~15 pp hazard Accel-Sim's authors attribute to profiler accounting) and NCU's **isolated-execution / per-pass cache-flush** assumption → simulate **cold-cache per kernel by default**, matching `--cache-control all` labels.
- **6.5 Calibration layer** (`sim/calib`). Per-arch fitted latencies/throughputs from 1D; optional **monotone low-capacity residual heads** (the v5 isotonic pattern, D1's demoted hybrid) on top of simulator outputs — fit **slug-disjointly**, never truth-fitted per cell [F5].
- **6.6 Fidelity knobs** (per component, the what-if differentiator): frontend LOW · warp-scheduler MEDIUM · exec-units MEDIUM · tensor MEDIUM · L1 MEDIUM · interconnect LOW · L2 MEDIUM · DRAM HIGH. Codex-flagged **hypotheses-not-spec** (RF cache, uniform unit, L0 icache stream-buffer): keep behind knobs, validate before claiming.

---

## 7. sm_86 soundness gate (GATE M0) — the explicit "finish 3090 Ti first" checklist

**No 5060 laptop or GB202 migration until every item is green and recorded in `plans/sm86_soundness_report.md`.**

1. **Toolchain pinned + recorded:** NCU 2026.1.0, driver 610.43.02, nvcc 13.3/CUDA 13.2, NVBit 1.8; venv (Python 3.12) built, `requirements.txt` installed.
2. **Root capabilities confirmed on this host:** `nvidia-smi -lgc` clock lock succeeds; persistence mode on; NCU counter access works (root or R610 capabilities); NVML covariate sampling runs.
3. **Binaries rebuilt on-host** (never transferred): `tracer_tool.so` ✅ (via `tools/nvbit_tracer/pinned_build_env.sh`), `metric_extractor` + its unit tests. **CUPTI collector dropped (D8)** — timing ground truth is NCU, not CUPTI.
4. **End-to-end smoke on one portable_lcd kernel:** build → NVBit trace → NCU capture (T1+T2+T3) → repeatability gate → `label_admitted.json` — full loop green with the v2 protocol.
5. **Every reusable component imports/runs under the venv:** `arch_profiles` loader, `metric_registry`, `hrd_profiler`, `cache_replay`, `cache_predictor`, `pchase`, `sass_control`, `solar`.
6. **Restored test subset green** (§0.4), including ArchProfile lockstep (cards ≡ live constants) and the frozen sm_86 v5 head byte-identity.
7. **Data-safety proof:** the 7,792 deleted `sweeps/` files exist in the codex tree/annex (no unique data destroyed).
8. **Targets-not-features guard green** on the current tree.

---

## 8. Novelty / paper contributions (MICRO 2027)

1. **The first open, fast, event-driven simulator validated per-metric against NCU on Ampere *and* consumer/workstation Blackwell** — reaching sm_120, where no open simulator has a config.
2. **NCU-metric-*vector* prediction** (cycles + L1/L2 sector hit rates + traffic/sector counts + occupancy + SOL%) under a typed observation-contract emission layer — no prior work predicts sector hit rates or SOL%.
3. **A GPU NCU counter-repeatability / determinism study** (the missing Weaver/McKee analog) — noise floors that justify every accuracy gate; includes the Blackwell `lts` 115–217% anomaly.
4. **Same-ISA / cross-SKU generalization** (GB206 vs GB202: only card inputs change) — a controlled test mechanistic models historically fail.
5. **Speed:** 100–1000× over Accel-Sim/UPC with retained **what-if / variable-fidelity** capability — the black-box predictors cannot do what-if.
6. **A dataset release:** NCU-profiled FlashInfer-Bench + SOL-ExecBench across 3 SKUs / 2 arches.

---

## 9. Risk register

| Risk | Likelihood | Mitigation (already in plan) |
|---|---|---|
| Cycles bar ≤10% (D5) is *below* the ~13–17% literature floor | **High** | Fall back to the ≤15% UPC-class publishable floor; if even that stalls, headline the near-exact NCU metric *vector* (counts/cache/SOL) + cycles secondary; hybrid heads as insurance (D1). Heavier control-bit (§1E) + latency (§1D) calibration is the primary accuracy lever. |
| NVBit Blackwell TMA-tracing bugs (#162–164) | Med | Typed-exclude TMA-heavy cells; track upstream; consumer sm_120 avoids tcgen05/TMA-heavy paths. |
| Laptop clock instability (lock → limit under thermals) | Med→Low | Root clock-lock available (retires old DVFS risk); keep nvbench-style noise gates (≥10 samples, 0.5% rel-stddev stop, ≥20 µs kernels). |
| Blackwell classifier fix doesn't clear the L2 gate | Med | It's now a *measurement* task (NCU+root); if other error sources (coalescing/prefetch/compression/ECC) dominate, that's itself a reportable finding. |
| MICRO 2027 CFP shifts | Low | Re-confirm monthly; HPCA 2028 (≈Aug 2027) then ISCA 2028 fallbacks. |
| A competitor ships an sm_120 timing model first | Low | Differentiators hold: event-driven speed + full NCU-vector output + 3-SKU silicon validation + what-if. |
| Schedule slip (9 mo is tight for a from-scratch sim) | Med | sm_86-first gating; overlap Phase 1/2; laptop capture front-loaded; ~3 wk buffer; hybrid-insurance headline caps downside. |

---

## 10. Resolutions to prior open questions (merge dossier Q1–Q10, re-adjudicated under the new charter)

Full-NCU + simulator-primary changes most answers:
- **Q1 (canonical metric registry):** registry v2 (41 metrics, observation contracts) **is** canonical; **extend** it (stall-reason family, SOL/tensor metrics). `model_io_role` replaces `deployability`. The old "gpu__time/occupancy/IPC forbidden as labels" scope conflict **dissolves** — with NCU everywhere they are first-class *targets*.
- **Q2 (wall-clock label):** `gpc__cycles_elapsed.max` primary (operating-state-bound), `gpu__time_duration.sum` secondary; NCU-only timing supersedes the retired CUPTI dual-source path. No quarantine.
- **Q3 (wall-clock predictor target):** **the simulator is the model.** Event-driven cycle output × per-arch reference clock; the refuted OLS-on-log10 is retired.
- **Q4 (label_admitted reconciliation):** bump to registry-v2 unit-tagged schema; re-admit all cells under the v2 protocol; pin the frozen sm_86 v5 head; keep the 30-cell sm_120 cohort (b1646ef7) byte-stable as a regression anchor.
- **Q5 (sm_90a build vs deploy ban):** already resolved — `build_arch_families.json` separates build-arch from deploy `arch_id`; sm_90a is a compile-only fatbin slice.
- **Q6 (Blackwell classifier fix in scope?):** **YES, now in scope** — cuobjdump inventory + cache-op microbench + NCU differential on the 5060 sets `bare_ldg_e_default` empirically (§1D).
- **Q7 (comparison_runner cohort model):** one arch-parameterized per-(kernel,metric,arch) harness (§3B).
- **Q8 (L1 contract):** L1 stays scored/coverage-gated until the sm_120 geometry card + head exist; add a hard L1 accuracy gate only once the warp-pool/MSHR rebuild (§6.3) clears it on sm_86.
- **Q9 (Blackwell key + token vocab):** `resolve_arch_id` already canonicalizes the 8 spellings; add `cards/sm_120-gb202.json` with SKU disambiguation; the 5060-laptop label does **not** silently transfer to GB202 — that's the cross-SKU experiment.
- **Q10 (provenance slug bug):** cosmetic (gitignored, never joins the runner); fix `build.py` `KERNEL_SLUG` wiring as a bounded Phase-0 tidy.

**Remaining genuine research gaps** (now *attackable* with live NCU+root, not blockers): sm_120 L1/L2 geometry, the Blackwell L2 gate, sm_120 L1 head, sm_120 NVML clock anchor, cross-arch transfer. All are Phase-1/3 capture tasks.

---

## 11. Immediate next actions (on approval — Phase 0, 3090 Ti host)

1. Create the working branch; verify the 7,792 deletions are data-safe (§7.7); run salvage S1–S8 with a `du`-budgeted manifest; land the pivot baseline commit.
2. Build the `uv` venv (Python 3.12), install `requirements.txt`, smoke-import every reusable module.
3. Refit guards/skills (§0.3); restore the ~60–90 keeper tests and green them (§0.4).
4. Scaffold `sim/` (Rust workspace) + `eval/baselines/` stubs.
5. **Run the sm_86 soundness gate (§7) and write `plans/sm86_soundness_report.md`.** ← the "finish 3090 Ti first" deliverable.
6. Begin 1A/1C capture on the 3090 Ti (richest existing roster) while the GB202 cloud instance is provisioned. **Migrate to the 5060 only after M0 + M2 are green.**

---

*Provenance: this plan is grounded in a same-session audit (two read-only research agents: legacy-tree
salvage validation + new-repo tool/importability/design-note inventory) plus direct reads of the founding
master plan, the SOTA dossier (F1–F10 fix map), ARCHITECTURE/README/CLAUDE/OPEN_RESEARCH_CHALLENGES, the
metric registry, and both arch cards. Venue dates verified against the MICRO 2026 CFP. External SOTA
claims inherit the 2026-07-02 adversarial-verification provenance.*
