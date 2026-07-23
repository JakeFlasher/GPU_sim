# Unified Knowledge Base — cross_arch_gpu_profiler_suite

353 non-index distilled cards across 14 subcorpora, plus this root index and the
patents router/index entry. The generated manifest consequently contains 355 rows;
the two index/router rows are not additional distilled cards. The cards were
produced by merging two knowledge
bases that were **name-disjoint** (zero card-level collisions), so this is a
pure flat union:

* **Category 1 — Reference / domain knowledge** (68 cards): the arch-agnostic
  NVIDIA / CUDA / PTX / SASS / measurement reference base. It was **byte-identical**
  across the three profiler trees (codex / Tree A / Tree B, sha256-verified); the
  single canonical copy from the Codex tree is kept.
* **Category 2 — SOL-ExecBench baseline corpus** (285 cards): the distinct KB from
  the dissolved `solbench_baseline` project (SOL-ExecBench, arXiv:2603.19173).

Primary discovery is the task-shaped Skills under `.claude/skills/` plus the
`UserPromptSubmit` hook — the merged **superset** router `scripts/kb/route_query.py`
(wired in `.claude/settings.json`), which classifies the prompt against the FTS5
manifest and injects candidate cards plus a routing destination. A measurement
route may still name the retained retired skill as a historical pointer; it is
non-operative, and current policy comes from root `CLAUDE.md`, active schemas,
runbooks, and ADRs. Each card's `Primary raw source` frontmatter points at the
intended local raw sources under `references/` (full PDF papers, NVIDIA_Manuals
mirror, patents). See
[`../../docs/knowledge-routing.md`](../../docs/knowledge-routing.md) for the full flow.
Use this file when the hook is bypassed or a human needs a one-page inventory. Card
schema: `_template.md`. Rebuild the FTS5/BM25 manifest with
`python3 scripts/kb/build_manifest.py --root .` (the manifest is gitignored —
regenerate on a fresh clone).

```
.claude/knowledge/
├── INDEX.md            # this unified router
├── _template.md        # card schema (frontmatter + forbidden-surface block)
│
│   ── Category 1 · Reference / domain knowledge (deduped; Codex canonical) ──
├── patents/            # 24  NVIDIA arch evolution (Tesla→Blackwell); has its OWN sub-router patents/INDEX.md
├── ptx/                # 13  PTX-ISA cards (mma, cp.async, mbarrier, tcgen05, cache-hints, per-arch deltas)
├── sass/               #  2  SASS instruction reference + tooling (cuobjdump/nvdisasm)
├── cuda/               #  7  clusters/DSMEM, memory hierarchy, register file, warp scheduling/divergence
├── best-practices/     #  3  coalescing, control-flow, occupancy/launch
├── nvcc/               #  2  nvcc flags + fatbin/PTX trajectory
├── inline-ptx/         #  1  inline-PTX from .cu
├── blackwell/          #  4  compatibility, tensor-perf, tuning, RTX PRO 6000 caps
├── measurement/        #  6  pinned NCU protocol v2.1, clock64/globaltimer/DVFS, CUPTI Activity, non-root cloud contract (historical), NVBit playbook
├── simulator/          #  6  nvasim / Rust DES — PRIOR-ART reference for the ACTIVE simulator build (re-founded 2026-07-02; nvasim source is not ported verbatim)
│                       #     Phase 2 v0 (2026-07-06): ACTIVE implementation lives in sim/ (engine/sm/trace/calib/mem live; emit/ffi stubs); progress log plans/phase2_sim_core_progress.md
│
│   ── Category 2 · SOL-ExecBench baseline corpus (solbench union) ──
├── sources/            # 16  citation gateways (SOL-ExecBench/AVO papers, SOLAR, CUDA/PTX/NVCC/CUTLASS docs)
├── patterns/           # 27  kernel-family CUTLASS/CuTe C++ recipes (gemm, flash-attn, rmsnorm, moe, rope, ...)
├── problems/           # 235 per-problem SOL-ExecBench dossiers (L1=94, L2=82, Quant=33, FIB=26; slug-keyed)
└── cutlass_constructs/ #  6  shared CUTLASS/CuTe primitives (tiled_mma, tma_descriptor, tcgen05_tmem, ...)
```

## Skills → subcorpus map (11 canonical skills)

Reference / domain (Category 1):
* `navigating-project-kb` — meta-router (this INDEX + patents/INDEX)
* `validating-nonroot-gpu-measurement` → `measurement/` — RETIRED (pre-2026-07-02 charter); measurement questions now route via `measurement/pinned-protocol-v21.md`
* `interpreting-ptx-isa` → `ptx/` + `inline-ptx/`
* `analyzing-sass-nvbit-traces` → `sass/` + `measurement/nvbit-injection-playbook.md`
* `tuning-blackwell-sm120-kernels` → `blackwell/`
* `optimizing-cuda-kernels` → `cuda/` + `best-practices/`
* `compiling-cuda-artifacts` → `nvcc/`
* `reasoning-about-nvidia-architecture-history` → `patents/`
* `designing-gpu-simulator` → `simulator/` (ACTIVE design skill — the NCU-metric-predicting Rust simulator IS the project per the root CLAUDE.md 2026-07-02 re-founding; simulator cards are prior-art reference, not verbatim-port source)

SOL-ExecBench (Category 2):
* `implementing-sol-baselines` → `sources/`, `patterns/`, `problems/`, `cutlass_constructs/`

## Notes for the merge

* The unified router imports the single 14-subcorpus ownership map from
  `scripts/kb/routing_contract.py`; the retired duplicate `tools/sol_kb`
  runtime fork is not an active routing path. Because the subtopic directory
  names are disjoint, the flat union preserves every hardcoded
  `.claude/knowledge/<subcorpus>/` path with no rewrites.
* Category 2 `sources/` stubs are thin citation gateways; where they cover a domain
  that Category 1 holds a full card for (CUDA, PTX, NVCC, Blackwell, inline-PTX), they
  cross-link into the Category 1 card rather than dead-ending.
* **Stale-count fix:** the SOL corpus's original INDEX said "12 sources"; the on-disk
  count is **16** (`ako4all-kernel-references`, `cuda-kernel-source-guides`,
  `cuda-optimization-prs`, `kernel-pilot-routing` were added after that line was written).
* **Charter update (root CLAUDE.md, re-founded 2026-07-02):** the project BUILDS an
  event-driven Rust simulator that predicts NCU metric vectors. `simulator/` cards are
  prior-art reference for that ACTIVE build (nvasim source is not ported verbatim), and
  NCU is a first-class ground-truth instrument under the pinned protocol
  (`measurement/pinned-protocol-v21.md`). The one surviving prohibition is Rule 1
  (targets-never-features): NCU values are prediction targets/validation labels, never
  simulator inputs.
* **New card (2026-07-06):** `measurement/pinned-protocol-v21.md` — pinned NCU
  measurement protocol v2.1 (host `-pm 1; -lgc 1350; -lmc 10501` + NCU
  `--clock-control none --cache-control all --replay-mode kernel`), determinism
  classes, and capture traps. Canonical for all label capture.
