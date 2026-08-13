# SOL-ExecBench — the dissolved benchmark suite

> **Scoped historical/evaluation documentation.** SOL-ExecBench remains an
> evaluation corpus and subtree-specific toolchain. Its former counter-free and
> no-clock wording does not govern the active simulator or root measurement
> protocol. For current project policy use root `CLAUDE.md`,
> [`docs/README.md`](README.md), and accepted ADRs; for historical subtree context
> use [`sol_execbench/CAMPAIGN_CONTEXT.md`](sol_execbench/CAMPAIGN_CONTEXT.md) only.

SOL-ExecBench is the fourth source tree merged into this suite. It began life as
`solbench_baseline` — a **counter-free GPU analytics framework** built around
NVIDIA's *SOL-ExecBench* benchmark (arXiv:2603.19173). Unlike the three profiler
trees, it is not a cache/wall-clock predictor; it is a **benchmark harness plus a
label/trace source**. On merge it was *fully dissolved*: its components were
relocated as whole units across the unified tree and it now shares the suite's
`arch_profiles/`, `tools/nvbit_tracer/`, `.claude/knowledge/`, and `references/`
rather than living as a walled module.

The dissolved subtree's historical context is retained in
[sol_execbench/CAMPAIGN_CONTEXT.md](sol_execbench/CAMPAIGN_CONTEXT.md); that file is not current root
policy and does not authorize capture, simulator, or architecture assumptions.

---

## 1. What SOL-ExecBench is

SOL-ExecBench is **235 GPU kernel-optimization problems** targeting the NVIDIA
**B200 (`sm_100` / `sm_100a`, GB100)**, split across four categories:

| Category | Count | Character |
|---|---:|---|
| **L1** | 94 | Level-1 primitives (elementwise, norms, activations) |
| **L2** | 82 | Level-2 fused / composed kernels |
| **Quant** | 33 | FP8 (e4m3/e5m2) and NVFP4 (e2m1) block-scaled quantization |
| **FIB** | 26 | "Fill-in-the-blank" completion problems |
| **Total** | **235** | |

Each solution is graded by the **SOL Score** — a Speed-of-Light-relative metric
where `T_SOL` is the analytically-derived hardware bound (from NVlabs SOLAR:
roofline + Orojenesis), `Tb` is the internal scoring baseline, and `Tk` is the
measured kernel time (`sol_execbench/CAMPAIGN_CONTEXT.md:10-17`):

```
             Tb − T_SOL
S(Tk) = ─────────────────────────      ∈ [0, 1]
        (Tk − T_SOL) + (Tb − T_SOL)
```

`S = 0.5` matches the internal baseline; `S = 1.0` reaches the SOL bound. The
paper's agent baseline achieved overall **median S = 0.732** (L1 0.688, L2 0.761,
Quant 0.757, FIB 0.789). This suite is **not a leaderboard contender**: template
hand-ports are functional ports, targeting a reporting band of **SOL 0.3–0.5+**,
not top-of-board tuning (`sol_execbench/CAMPAIGN_CONTEXT.md:58-79`).

> The 235 counts above are the *problem-space* definition. The dissolved
> `workloads/sol_execbench_baselines/{L1,L2,Quant,FIB}/` tree holds the subset of
> `*.cu` hand-ports produced so far, not all 235.

---

## 2. The two-phase mission

SOL-ExecBench runs a strict **sm_86-smoke → B200-harvest** pipeline. Local
performance is explicitly *not* a goal — Phase 1 only proves the framework runs
end-to-end and emits every analytic artifact cleanly.

```
 ┌───────────────────── PHASE 1 ─────────────────────┐   ┌───────────── PHASE 2 ─────────────┐
 │ local framework smoke (RTX 3090 Ti, sm_86)        │   │ B200 provider  (sm_100a, GB100)   │
 │                                                   │   │                                   │
 │  235 PyTorch refs ──hand-port──► *.cu templates   │   │  same invocation surface, re-run  │
 │  (CUTLASS C++ / CuTe C++ / CUDA C++)              ═╪══►│  correctness + 5 surfaces on ALL  │
 │  drive native sol-execbench evaluator subprocess  │   │  235 *.cu on real sm_100a HW      │
 │  emit 5 counter-free surfaces cleanly             │   │  harvest TCGen05/TMEM/TMA/cluster │
 │  bar: "runs, emits, ships identical to B200"      │   │  /DSMEM/NVFP4/FP8/L2/HBM3e intel  │
 └───────────────────────────────────────────────────┘   └───────────────────────────────────┘
      NOT-B200-validated (compile+smoke only)                Profiled-B200-sm_100a  ← deliverable
```

A local pass proves compile + smoke-run viability **only** — it does *not*
validate TCGen05/TMEM codegen, TMA descriptor shapes, cluster/DSMEM behavior,
register/occupancy/spill pressure, L2 partitioning, HBM3e bandwidth, CUTLASS
schedule selection, NVFP4/FP8 layout paths, or the 5th-gen tensor cores
(`sol_execbench/CAMPAIGN_CONTEXT.md:434-445`). The B200 profiling intel is the deliverable;
downstream kernel optimization is out of scope.

> **Historical SOL contract snapshot.** Sections 3–4 preserve the dissolved
> evaluation subtree's former counter-free/no-clock surfaces and its offline-label
> discussion. They are not current root measurement policy, simulator inputs, or
> permission to run NCU, mutate clocks, or calibrate the active simulator. Current
> labels and capture authority come from root `CLAUDE.md`, active schemas, and
> `tools/local_labels/`.

---

## 3. The five counter-free analytic surfaces

Every surface is **non-root, non-NCU, non-clock-mutating** and *designed to
migrate* to the B200 provider. Each is a plugin under
`tools/sol_baselines_harness/surfaces/`; per-surface availability is recorded in a
`manifest.json` with an `ok | unavailable | blocked | failed | skipped`
vocabulary, so a provider-side toolchain gap is *recorded*, never silently
demoted to "unprofiled" (`sol_execbench/CAMPAIGN_CONTEXT.md:247-338`).

```
                    native sol-execbench evaluator (subprocess)
                    ── correctness verdict + diagnostic speedup ──
                                        │
        ┌───────────────┬───────────────┼───────────────┬────────────────┐
        ▼               ▼               ▼               ▼                ▼
 ┌────────────┐ ┌──────────────┐ ┌────────────┐ ┌──────────────┐ ┌───────────────┐
 │ 1  NVBit   │ │ 2  CUPTI     │ │ 3  Nsight  │ │ 4  Static    │ │ 5  compute-   │
 │  metric    │ │  Activity    │ │  Systems   │ │  introspect  │ │  sanitizer    │
 │  extractor │ │ (non-counter)│ │ (+NVTX opt)│ │ cuobjdump/   │ │ memcheck/race/│
 │  per-warp  │ │  wall-clock  │ │  app-level │ │ nvdisasm/    │ │ sync/initcheck│
 │  histograms│ │  attribution │ │  timeline  │ │ ptxas -v     │ │               │
 │ +Tier-2    │ │ flushAll     │ │ correlation│ │ SASS/PTX/    │ │ correctness   │
 │  addr trace│ │ before read  │ │            │ │ spill/--list │ │ bug catch     │
 └─────┬──────┘ └──────┬───────┘ └─────┬──────┘ └──────┬───────┘ └──────┬────────┘
       │               │               │               │                │
 CUDA_INJECTION64  extend core/    nsys profile   post-build hook   compute-sanitizer
 _PATH=...         bench/timing.py --trace=cuda,   on torch cpp_     --tool=... python
 metric_extractor  (:426 already   nvtx            extension .so     eval_driver.py
 .so               uses CUPTI)                     --list-elf proves
                                                   sm_100a present
```

Grounding notes:

- **Surface 1 (NVBit)** is `tools/nvbit_tracer/metric_extractor/metric_extractor.cu`
  (`.so` producer); the Tier-2 ordered sector-event address sidecar is env-gated
  (`METRIC_EXTRACTOR_TIER2_ADDRESS_DUMP=1`, `metric_extractor.cu:376-403`) and
  wired through `surfaces/nvbit_metrics.py`.
- **Surface 2 (CUPTI Activity)** has a *collision risk*: the upstream evaluator's
  `core/bench/timing.py:426` already runs a CUPTI Activity timer. Prefer extending
  that path over a second in-process CUPTI consumer; else run a matched separate
  eval to avoid double-subscription.
- **Surface 3 (Nsight)** — NVTX ranges are best-effort/optional; the judge image
  may exclude NVTX, so submitted kernels must not depend on it.
- **Surface 4 (Static)** is the only build-time surface; `cuobjdump --list-elf`
  is how a B200 build *proves* `sm_100a` cubins are actually present.
- **Surface 5 (sanitizer)** catches the correctness bugs the randomized
  SOL-ExecBench trial would surface.

The five surfaces sit **beside** the official SOL scoring — none replaces it.
Before fanning out to all 235 runs, `tools/sol_baselines_harness/b200_preflight.py`
must clear the Phase-1→Phase-2 gate (`b200_ready_manifest.json`).

---

## 4. NCU calibration carve-out + Tier-2 cache predictor

SOL-ExecBench keeps the same NCU quarantine as the profiler trees (no NCU in any
deployed/B200 path), **but** it carries its own operator-authorized
(2026-05-16) **calibration-only carve-out**: NCU may run *offline* on `sm_86` /
`sm_120` to harvest **read-only** L1/L2 sector hit-rate ground truth
(`l1tex__t_sector_op_read_hit_rate.pct`, `lts__t_sector_op_read_hit_rate.pct`,
plus the underlying `…_sectors_op_read_hit.sum` / `…_sectors_op_read.sum`). The
combined `…_sector_hit_rate.pct` family conflates write-back store traffic and
**must not** be the label target (`sol_execbench/CAMPAIGN_CONTEXT.md:104-119`).

Those labels feed a **Tier-2 L1/L2 cache-hit predictor** at
`tools/sol_baselines_harness/metrics/cache_predictor.py` (with
`cache_geometry.py`, `hardware_descriptor.py`, `isotonic_calibrator.py`,
`pooled_cross_arch_head.py`, and the `calibration/cache_prediction/` ledgers). It
enforces the **same tiered MAPE gate** as the profiler: **±5 % predictor-friendly**
(LayerNorm/RMSNorm/SiLU/Softmax/well-behaved GEMM), **±15 % predictor-hostile**
(carveout-sensitive, persisting-window, prefetch-hint, warm-cold-pair,
spill-to-local), and **typed-fail-closed** (`predictor_typed_unavailable`) on cells
it cannot honestly score. This predictor is a direct **sibling** of the profiler's
cache-hit predictors — same tiered gate, same sm_86/sm_120 dual-arch calibration
story. It ships a
calibrated model that needs **no NCU at deployment**.

> **Honesty posture (inherited by the whole suite):** the `sm_86` cache
> prediction is validated; the `sm_120` (Blackwell) predictor path is a
> **research challenge**, not a shipped claim; wall-clock is **labels + honest
> refutation only** — no positive predictor claim. The active simulator design
> is in [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 5. The dissolve map — where each component went

Every solbench component was relocated **as a whole unit** (no re-slicing).

```
   solbench_baseline (origin tree)                unified suite (this tree)
 ───────────────────────────────────      ──────────────────────────────────────────────
  baselines/{L1,L2,Quant,FIB}/       ───►  workloads/sol_execbench_baselines/{L1,L2,Quant,FIB}/
      (235-problem *.cu hand-ports)                       (+ L1_direct_ncu_binaries/)

  sol-execbench/  (upstream evaluator) ─►  tools/sol_execbench/        (native CLI, uv workspace)
  SOLAR/          (T_SOL deriver)      ─►  tools/solar/                (roofline + Orojenesis)
  scripts/baselines_harness/          ─►  tools/sol_baselines_harness/ (run/report/score/surfaces/
                                                                        metrics/calibration/preflight)
  scripts/baselines_scaffold/         ─►  tools/sol_baselines_scaffold/ (solution.json boilerplate)
  scripts/kb/     (canonical KB router) ─►  scripts/kb/                (unified runtime; the historical tools/sol_kb router fork was retired)

  nvbit_tracer/metric_extractor/      ─►  tools/nvbit_tracer/metric_extractor/   ┐ SHARED module:
      (NVBit 1.8 SDK, isa_def)                (alongside profiler's tracer_tool/) ┘ one SDK, 2 tools

  cutlass/  +  torchview/ (vendored)  ─►  third_party/{cutlass,torchview}

  arch_resolver / vocabulary          ─►  arch_profiles/          (unified ArchProfile cards)
  .claude/knowledge/ (SOL KB)         ─►  .claude/knowledge/      (merged, shared)
  references/NVIDIA_Manuals/          ─►  references/             (shared doc mirror)
```

Config docs (`docs/sol_execbench/CAMPAIGN_CONTEXT.md`, `pyproject.toml`, `pytest.ini`,
`uv.lock`) were carried under [sol_execbench/](sol_execbench/) so the dissolved
project's uv-workspace contract is reproducible.

### Shared assets (deduplicated at the suite root)

The dissolve *earned its keep* by collapsing four kinds of duplication both the
profiler and SOL-ExecBench had been carrying independently:

| Shared asset | Was duplicated as | Now lives at |
|---|---|---|
| **NVBit 1.8 SDK + `isa_def` opcode tables** | one copy per tree | `tools/nvbit_tracer/` — exposes both the Accel-Sim `tracer_tool` (profiler) and the counter-free `metric_extractor` (SOL) as variants |
| **Arch resolver / vocabulary** | `arch_resolver.py` `_CC_TO_YAML`, host.kind↔arch_tag | `arch_profiles/` unified ArchProfile cards (seeded from this resolver + `tools/solar/configs/arch/*.yaml`) |
| **NVIDIA doc mirror** | `references/NVIDIA_Manuals/` per tree | one shared `references/` |
| **Knowledge base** | SOL-specific `.claude/knowledge/` | merged unified `.claude/knowledge/` |

The retained SOL tests still distinguish deploy-time counter-free paths from
calibration paths. Those tests describe SOL behavior only and do not restrict
direct NCU use elsewhere in the repository.

---

## See also

- [sol_execbench/CAMPAIGN_CONTEXT.md](sol_execbench/CAMPAIGN_CONTEXT.md) — retained scoped historical SOL-ExecBench context, not current authority
- [ARCHITECTURE.md](ARCHITECTURE.md) — active Rust simulator and ArchProfile boundary
- [STATUS.md](STATUS.md) — current research priorities
- [../README.md](../README.md) — suite overview
