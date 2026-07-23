# Architecture

> **CURRENT STATUS — HISTORICAL DUPLICATE; NON-OPERATIVE (2026-07-18).**
> Current authority is [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md),
> [`docs/STATUS.md`](docs/STATUS.md),
> [`docs/COMPONENTS.md`](docs/COMPONENTS.md), and the accepted
> [`ADRs`](docs/decisions/README.md), especially ADR 0001 (fresh Task7/H3
> continuation while the prior attempt remains frozen, zero-credit, and
> reuse-forbidden) and ADR 0002 (cycles-only publication binding). All text
> below that presents the predictor/non-root thesis, no-simulator-revival rule,
> DVFS-stable cycles, or a then-current project state is retained historical
> context and is not operative. The historical body is intentionally preserved.

`cross_arch_gpu_profiler_suite` answers one question: **how accurately can a
non-root process, denied hardware performance counters, predict per-kernel cache
hit ratio and wall-clock behaviour — and how do we keep that claim honest across
two GPU architectures?**

This document is the map. It has five parts:

1. [The two-plane model](#1-the-two-plane-model-labels-vs-deploy) — labels vs deploy
2. [End-to-end dataflow](#2-end-to-end-dataflow) — from kernel to gated verdict
3. [The three predictor lanes](#3-the-three-predictor-lanes) — L2, L1, wall-clock
4. [The ArchProfile system](#4-the-archprofile-system) — arch configs as inputs
5. [Merge topology](#5-merge-topology) — how four trees became one

---

## 1. The two-plane model (labels vs deploy)

The whole design turns on one asymmetry: **NCU exists locally, never at deploy.**

```
        LOCAL DEV PLANE (has NCU)                    DEPLOY PLANE (no counters)
   ┌──────────────────────────────────┐        ┌──────────────────────────────────┐
   │  RTX 3090 Ti (sm_86)              │        │  cloud / server GPU               │
   │  RTX 5060 laptop (sm_120)         │        │  (counter-less by policy)         │
   │                                   │        │                                   │
   │  NCU ── ground-truth LABELS ──┐   │        │   only user-space signals:        │
   │  (quarantined: tools/local_   │   │        │     · NVBit SASS/address traces   │
   │   labels/ → data/local_labels)│   │        │     · CUPTI Activity duration_ns  │
   │                               │   │        │     · CUDA events                 │
   │  user-space FEATURES ─────────┼───┼────────┼──▶  · static SASS (cuobjdump)     │
   │  (same signals as deploy) ────┘   │        │     · read-only NVML covariates   │
   └───────────────────────────────────┘        └───────────────────────────────────┘
            │                                                     │
            │  labels + features (LOCAL only)                     │  features only (DEPLOY)
            ▼                                                     ▼
   ┌───────────────────────────────────────────────────────────────────────────────┐
   │  PREDICTOR: fit on local (label ⟵ features), score deploy-legal features only  │
   │  Leakage guard: a "label_only" metric (NCU time, occupancy, IPC) can NEVER      │
   │  become an inference feature  →  tools/local_labels/no_deployability_leakage_   │
   │  guard.py  (keyed on the registry_v2 `deployability` field)                     │
   └───────────────────────────────────────────────────────────────────────────────┘
```

**Rule 1 (the hard contract).** NCU / hardware counters / driver-root ops /
privileged CUPTI / clock mutation are forbidden on any cloud or server GPU. NCU on
the two *local* GPUs is a bounded carve-out for label production only, confined to
`tools/local_labels/` writing to `data/local_labels/`, and enforced by a repo-wide
static guard. (SOL-ExecBench carries an analogous calibration-only carve-out under
`tools/sol_baselines_harness/calibration/`.) See [`docs/ncu-quarantine.md`](docs/ncu-quarantine.md).

---

## 2. End-to-end dataflow

One kernel, from execution to a gated accuracy verdict. The **left column is
LOCAL-only** (label production); the **right column is deploy-legal** (runs
anywhere, no counters).

```
                         a GPU kernel launch (e.g. rmsnorm_fwd_bf16 @ shape)
                                          │
              ┌───────────────────────────┴────────────────────────────┐
              │  LOCAL-ONLY  (NCU quarantine)          DEPLOY-LEGAL     │
              ▼                                                          ▼
   ┌─────────────────────────┐                        ┌──────────────────────────────────┐
   │ tools/local_labels/     │                        │ workloads/simbench_bw/run_suite.py│
   │  capture_*_labels.py     │                        │  --mode trace  (NVBit inject)     │
   │  ncu → sector hit rate,  │                        │  --mode timing (CUPTI Activity)   │
   │  gpu__time, cycles       │                        │  tools/nvbit_tracer/tracer_tool   │
   └───────────┬─────────────┘                        └──────────────────┬────────────────┘
               │ per-launch sidecars                                     │ SASS/address trace,
               ▼                                                         │ duration_ns, NVML
   ┌─────────────────────────┐                                          ▼
   │ repeatability gate       │                        ┌──────────────────────────────────┐
   │  (CV ≤ 0.05, ≥N repeats) │                        │ FEATURE extraction                │
   │  → data/local_labels/    │                        │  workloads/simbench_bw/           │
   │    label_admitted.json   │                        │   canonical_features.py           │
   │    (v5, 74 cells)        │                        │  tools/cache_predictor/l1_walker  │
   └───────────┬─────────────┘                        │  hrd_profiler reuse distance      │
               │                                        └──────────────────┬────────────────┘
               │  admitted labels                                          │  deploy-legal features
               │  (the y)                                                  │  (the X)
               └───────────────────────┬──────────────────────────────────┘
                                        ▼
                    ┌───────────────────────────────────────────┐
                    │ PREDICTORS (workloads/simbench_bw/)        │
                    │   cache_predictor.py   → L2, L1 hit rate   │
                    │   timing_cycle_predictor.py → cycles       │
                    │   + monotonic_calibration_head (v5)        │
                    └───────────────────┬───────────────────────┘
                                        │ predictor bundles
                                        ▼
                    ┌───────────────────────────────────────────┐
                    │ comparison_runner.py                       │
                    │   8-tuple join (predicted ⟷ label)         │
                    │   arch-keyed 3-tuple roster split          │
                    │   coverage gate · bootstrap CI             │
                    └───────────────────┬───────────────────────┘
                                        ▼
                    ┌───────────────────────────────────────────┐
                    │ GATES → typed verdict                      │
                    │   L2: median-abs ≤0.06, geomean ≤0.15,     │
                    │       holdout ≤0.25   → PASS (sm_86)        │
                    │   cycle: geomean ≤0.15 → near-miss (sm_86) │
                    │   any sm_120 predictor → REFUTED           │
                    └───────────────────────────────────────────┘
```

Everything the predictor consumes on the right column is reproducible at the
counter-less deploy target; the left column exists only to *teach* and *grade* it.

---

## 3. The three predictor lanes

```
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │ LANE A — L2 sector hit rate            file: workloads/simbench_bw/cache_predictor │
 │                                        head: data/local_labels/sm_86_v5_head.json  │
 │  NVBit sector traffic ─┐                                                           │
 │  static SASS features ─┼─▶ parametric model ─▶ monotonic v5 calibration head ─▶ ŷ  │
 │  reuse distance (HRD) ─┘        (arch-general engine; capacity via ArchProfile)    │
 │                                                                                    │
 │  sm_86  ✅ VALIDATED   median-abs 0.059 · geomean-rel 0.064 · holdout 0.055 (3/3)  │
 │  sm_120 ❌ REFUTED     learned head 0.076 > 0.055 gate  (see Lane-A note below)     │
 └──────────────────────────────────────────────────────────────────────────────────┘

 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │ LANE B — L1/TEX sector hit rate        file: tools/cache_predictor/l1_walker.py    │
 │                                        head: data/local_labels/sm_86_l1_head.json  │
 │  per-warp L1 walk ─▶ cache-op classifier ─▶ bounded-[0,1] L1 head ─▶ ŷ             │
 │      │                                                                              │
 │      └── classifier hinges on _BARE_LDG_E_DEFAULTS_TO_CA (per-arch, see §4)         │
 │  sm_86  ⚠️ SCORED but NOT accuracy-gated (rel-geomean 0.89; median abs residual 4%) │
 │  sm_120 ❌ UNCALIBRATED (no L1 head; classifier inert → 0/26 cells scored)          │
 └──────────────────────────────────────────────────────────────────────────────────┘

 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │ LANE C — wall-clock duration           files: timing_cycle_predictor.py,           │
 │                                               timing_closure.py (dual-source)       │
 │                                                                                    │
 │   DVFS-stable CYCLES  ◀── the load-bearing target (gpc__cycles_elapsed.max)         │
 │        │                                                                            │
 │        ├─ cycles × per-arch short-burst clock × [1 ± ε]  ─▶ ns INTERVAL (derived)   │
 │        └─ deploy-ns crosswalk: CUPTI duration_ns  ⟷  NCU gpu__time   (PASSES 13/13) │
 │                                                                                    │
 │  MERGE POLICY: labels + honest refutation ONLY — no positive predictor claim.      │
 │  sm_86  🔬 cycles near-miss (geomean 0.158 vs 0.15); deploy-ns crosswalk research- │
 │            success — but NOT shipped as a predictor claim                           │
 │  sm_120 ❌ REFUTED (OLS-on-log10 SMAPE 1.766 > 0.891 baseline; 15.34× DVFS)         │
 └──────────────────────────────────────────────────────────────────────────────────┘
```

> **Lane-A note (sm_120).** The sm_120 L2 refutation is *not* a predictor bug — it
> is one frozenset. Blackwell bare `LDG.E` / `LDG.E.U16.CONSTANT` / descriptor loads
> route to `OTHER`/bypass because `_BARE_LDG_E_DEFAULTS_TO_CA = {"sm_86"}`, so the
> L1→L2 miss filter has zero eligible misses (`l1_filter_inert = true`) and every
> sm_120 cache verdict refutes. Fixing it needs live-Blackwell evidence and is *not*
> guaranteed to clear the gate — see [`OPEN_RESEARCH_CHALLENGES.md`](OPEN_RESEARCH_CHALLENGES.md).

The `tools/cache_replay/` sector-replay subsystem (28 files, from the sm_120 tree)
is carried as a **mechanistic ablation** of Lane A on sm_120 — *not* load-bearing;
the parametric `cache_predictor.py` has zero references to it.

---

## 4. The ArchProfile system

The merge's structural contribution: turn hardcoded per-device constants into
**input cards** resolved through one loader. This is what makes the codebase
"cross-arch" rather than "sm_86 with sm_120 bolted on".

### 4a. Token resolution (the "8-spellings" landmine, fixed)

Eight arch spellings coexisted across the trees (`sm_86`, `sm_120`, `sm_120-laptop`,
`sm_120a`, `sm_120f`, `sm_90`, `sm_90a`, device-name strings) with two maps keyed
on *different* tiers — a live landmine. `resolve_arch_id()` collapses them:

```
   token / device-name / (major,minor)          arch_profiles/aliases.json
   ─────────────────────────────────────         ──────────────────────────
   "sm_86"                        ┐
   (8,6)                          ├──────────────▶  canonical arch_id = "sm_86"
   "NVIDIA GeForce RTX 3090 Ti"   ┘

   "sm_120"        "sm_120-laptop"  ┐
   "sm_120a"       "sm_120f"        ├────────────▶  canonical arch_id = "sm_120"
   (12,0)          "…RTX 5060…"     ┘                     │
                                                          │  + SKU disambiguation
                                                          ▼
                                              sku = "rtx_5060_laptop"
                                                          │
   "RTX PRO 6000" / "B200" / "H100" ──▶ REJECTED          ▼   (a bare sm_120 cannot tell
   (non-local card; would silently return 5060 values)   card = cards/sm_120-laptop.json
                                                          GB206-laptop from GB202)
```

Build-time slices live on a **separate axis** so a fatbin can legitimately carry
Hopper `sm_90a` or Blackwell `sm_120a` without those ever becoming a *deploy*
`arch_id` (`build_arch_families.json`; the DEC-2 deploy ban governs only the deploy
axis, never the build Makefiles).

### 4b. The card + the lockstep contract

```
   arch_profiles/cards/sm_86.json                 live source (unchanged)
   ──────────────────────────────                 ───────────────────────
   sm_count            = 84            ═══════════ cache_predictor._SM_COUNT_BY_ARCH["sm_86"]
   griddim_alpha       = 0.110         ═══════════ cache_predictor._GRIDDIM_PENALTY_ALPHA
   l2.capacity_bytes   = 6291456       ═══════════ bundle_from_trace._STATIC_CARD_L2_BYTES["sm_86"]
   bare_ldg_e_default  = "ca"          ═══════════ "sm_86" ∈ _BARE_LDG_E_DEFAULTS_TO_CA
   l1_geometry, nvml…  = populated                 │
                                                    │  test_arch_profile_lockstep.py
                                                    ▼  AST-extracts the live constant and
                                            asserts card == source  ⇒  ZERO behaviour change

   arch_profiles/cards/sm_120-laptop.json
   ──────────────────────────────────────
   sm_count            = 26   ◀── MEASURED; corrects the guessed 50 still in the source
   bare_ldg_e_default  = "unknown_typed_unavailable"   ◀── Blackwell load semantics unmodeled
   cache_layers        = ["L2"]        (sm_86 = ["L1","L2"])
   l1_geometry         = {status:"research_gap", challenge:"…"}   ◀── fails closed, no sm_86 fallback
   nvml_ref_op_point   = {status:"research_gap", …}
```

`ResearchGap` is falsy (`bool(gap) is False`) so `if profile.l1_geometry:` naturally
fails closed. A consumer that reads a gap must degrade gracefully, never substitute
another arch's number. See [`docs/archprofile.md`](docs/archprofile.md).

---

## 5. Merge topology

```
   BEFORE: four sibling working trees (same GitHub origin, divergent branches)

   non_root_gpu_profiler_codex ─┐  sm_86 canonical (247 commits ahead of Tree A)
   non_root_gpu_profiler ───────┤  sm_86 reference snapshot
   non_root_gpu_profiler_5060 ──┤  sm_120 lane + cache_replay
   solbench_baseline ───────────┘  SOL-ExecBench (235 problems, B200 target)

                                    │  MERGE (source + curated frozen artifacts)
                                    ▼

   AFTER: one project on the codex spine
   ┌──────────────────────────────────────────────────────────────────────────┐
   │  codex spine  ─────────────────  tools/  workloads/  data/local_labels/    │
   │       +                                                                    │
   │  Tree B sm_120 overlay ────────  tools/cache_replay/  ncu_metric_registry  │
   │                                  typed_ncu_label_path  duration_admitted   │
   │                                  sm120_*_closure  (set-difference: never    │
   │                                  overwrites a codex-canonical file)         │
   │       +                                                                    │
   │  solbench DISSOLVED ───────────  tools/{sol_execbench,solar,sol_baselines_ │
   │                                  harness,sol_kb}  workloads/sol_execbench_ │
   │                                  baselines/  tools/nvbit_tracer/metric_    │
   │                                  extractor/  third_party/{cutlass,torchview}│
   │       +                                                                    │
   │  ★ NEW arch_profiles/  +  unified .claude/knowledge/ (353 cards)  +  docs/  │
   └──────────────────────────────────────────────────────────────────────────┘
```

**Decision rules that shaped it** (full detail in [`MERGE_PROVENANCE.md`](MERGE_PROVENANCE.md)):

* **Codex is canonical** — its frozen sm_86 v5 L2 head, guards, and registry_v2 win
  every conflict; Tree B files are overlaid only where codex has none.
* **Two-guards-each-own-their-subtree** — the profiler's NCU static guard delegates
  SOL-ExecBench's NCU enforcement to SOL-ExecBench's own comment-aware deployed-path
  guard; neither is weakened.
* **Byte-identity is sacred** — the sm_86 validated lane, its frozen heads, and the
  30-cell sm_120 cohort are carried unchanged; the ArchProfile mirrors, never mutates,
  their constants.

---

## Where to go next

| You want to… | Read |
|---|---|
| Understand the arch-input cards + resolver | [`docs/archprofile.md`](docs/archprofile.md) |
| See the L2 / L1 / wall-clock predictors in depth | [`docs/predictors.md`](docs/predictors.md) |
| Understand the metric registry + leakage guard | [`docs/metric-registry.md`](docs/metric-registry.md) |
| Understand the NCU quarantine + two guards | [`docs/ncu-quarantine.md`](docs/ncu-quarantine.md) |
| Understand the NVBit tracer (two tool variants) | [`docs/nvbit-tracing.md`](docs/nvbit-tracing.md) |
| Understand the dissolved SOL-ExecBench suite | [`docs/sol-execbench.md`](docs/sol-execbench.md) |
| Understand how the KB auto-routes sessions to local sources | [`docs/knowledge-routing.md`](docs/knowledge-routing.md) |
| See the honest unsolved problems | [`OPEN_RESEARCH_CHALLENGES.md`](OPEN_RESEARCH_CHALLENGES.md) |
| Trace every file to its origin + every decision | [`MERGE_PROVENANCE.md`](MERGE_PROVENANCE.md) |
