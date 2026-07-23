# Imported from BBuf/kernel-pilot (trimmed for solbench)

This subtree is a **trimmed, NCU-stripped, B200-focused** import of the
[BBuf/kernel-pilot](https://github.com/BBuf/kernel-pilot) CUDA-kernel
knowledge pack. It was prepared for solbench's counter-free, non-root cloud
B200 (sm_100a) environment per `CLAUDE.md` Rules 1-9, then aggressively
trimmed after a research pass on solbench confirmed which content is
actually relevant to current work (the metric_extractor RLCR loop and the
CUTLASS C++ submission surface).

## Trim rationale

Solbench has **pivoted** (2026-05-13) from 235-baseline kernel porting to
building a counter-free metric extractor (NVBit 1.8 atomic-aggregation
tool). Multiple original plans (KB_update, baseline_corpus,
counter_free_framework, sol_pattern_mapping) are explicitly marked
"pre-pivot legacy." A framework-usage scan of solbench showed:

| Framework | Hits in solbench | Action |
|---|---|---|
| CUTLASS | 1299 | KEEP (only submission target) |
| PyTorch | 478 | KEEP (reference baseline binding) |
| Triton | 338 | KEEP (Rule 5 forbids submission; algorithm reference) |
| FlashInfer | 68 | KEEP (FIB category label) |
| Flash-Attention | 3 | KEEP (literature + CUTLASS pattern) |
| TensorRT-LLM, CCCL-CUB, ThunderKittens, TileLang, QuACK, TileKernels, Veitner-Blog, Colfax-Research, CUDA-Blog-Kernels, DeepGEMM, SGLang, vLLM, CuTe-DSL-Python | 0-1 each | **DROPPED** (unused in solbench's active work) |

Architecture: solbench's local-test workflow runs **3090Ti (sm_86 Ampere)**
and **RTX 5060 laptop (sm_120 consumer Blackwell)** BEFORE pushing to **cloud
B200 (sm_100a)** for scoring. The trim therefore keeps:

- **sm_100** (cuda-cpp + cutlass-cpp): B200 scoring target.
- **sm_103** (cuda-cpp + cutlass-cpp): Blackwell datacenter variant context for B200.
- **sm_120** (cuda-cpp + cutlass-cpp): RTX 5060 local dev proxy reference.

For **sm_80 / sm_86 (Ampere — 3090Ti, A100)** local-test references, use
solbench's own `references/NVIDIA_Manuals/ampere_compatibility/` (11 files)
and `references/NVIDIA_Manuals/ampere_tuning/` (11 files). kernel-pilot did
not ship Ampere-specific guides.

Architectures dropped: sm_89 (Ada Lovelace — RTX 4090, L40S; not in
solbench's local-test hardware) and sm_90 (Hopper H100 — not local hardware,
no cloud H100 access).

## What's here (40 imported files)

```
kernel_pilot_imported/
├── KB_IMPORT_README.md         # this file
├── README.md                   # imported KB README (rewritten for trimmed scope)
├── index.json                  # framework registry (5 kept entries)
├── routing/
│   ├── frameworks/             # 5 framework lookup tables
│   │   ├── cutlass.md
│   │   ├── flash-attention.md
│   │   ├── flashinfer.md
│   │   ├── pytorch.md
│   │   └── triton.md           # SOLBENCH NOTICE - reference only, Rule 5 forbids submission
│   └── topics/                 # 9 topic pages + index.md (no communication.md)
│       ├── index.md
│       ├── attention.md
│       ├── matmul-gemm.md
│       ├── moe.md
│       ├── normalization.md
│       ├── rope.md
│       ├── activation-fusion.md
│       ├── sampling.md
│       ├── quantization-fp8.md
│       └── kv-cache.md
└── references/
    ├── index.md
    ├── prs/                    # 1 PR page kept (cutlass) as one production CUTLASS evidence example
    │   └── cutlass.md
    ├── source-guides/          # 5 upstream-repo code maps
    │   ├── cutlass.md
    │   ├── flash-attention.md
    │   ├── flashinfer.md
    │   ├── pytorch.md
    │   └── triton.md           # SOLBENCH NOTICE - reference only
    └── ako4all/                # 15 files: kernel-template refs + sm_100/sm_103/sm_120 optimization guides
        ├── ako4all-kernel-loop.md
        ├── cuda-cpp-kernel-reference.md
        ├── cutlass-cpp-kernel-reference.md
        ├── kernel-templates.md
        ├── nvidia-architecture-reference.md
        ├── source-attribution.md
        ├── troubleshooting.md
        ├── cuda-cpp/
        │   ├── cuda-cpp-overview.md
        │   ├── sm100-optimization-guide.md  (B200 cloud scoring target)
        │   ├── sm103-optimization-guide.md  (Blackwell variant context)
        │   └── sm120-optimization-guide.md  (RTX 5060 laptop local proxy)
        └── cutlass-cpp/
            ├── cutlass-cpp-overview.md
            ├── sm100-optimization-guide.md  (B200 cloud scoring target)
            ├── sm103-optimization-guide.md  (Blackwell variant context)
            └── sm120-optimization-guide.md  (RTX 5060 laptop local proxy)
```

**For sm_80 / sm_86 (Ampere — 3090Ti, A100):** kernel-pilot did not ship
Ampere-specific guides. Use solbench's `references/NVIDIA_Manuals/ampere_compatibility/`
and `references/NVIDIA_Manuals/ampere_tuning/` for local validation on Ampere hardware.

## What was DROPPED during the trim

### Frameworks dropped (no active solbench use)
- routing + source-guides + PR pages for: `tensorrt-llm`, `cccl-cub`, `thunderkittens`, `tilelang`, `quack`, `tilekernels`, `veitner-blog`, `colfax-research`, `cuda-blog-kernels`, `deepgemm`, `sglang`, `vllm`, `cute-dsl`

### Optimization guides dropped (not solbench's local-test or cloud-scoring architectures)
- `architectures/sm89-optimization-guide.md` (Ada Lovelace — RTX 4090, L40S; not in solbench's local-test hardware)
- `architectures/sm90-optimization-guide.md` (Hopper H100 — not local, no cloud H100)
- Entire `cute-dsl/` subtree (18 files) — CuTe DSL Python forbidden by Rule 5 (submission surface)
- Entire `triton/` subtree (10 files) — Triton forbidden by Rule 5 (submission surface)
- `triton-kernel-reference.md` (top-level Triton reference)
- `cute-dsl-kernel-reference.md` (top-level CuTe-DSL-Python reference)

### Optimization guides KEPT (covering solbench's full local-test → cloud-scoring path)
- `cuda-cpp/sm100-optimization-guide.md` (B200 cloud scoring target)
- `cuda-cpp/sm103-optimization-guide.md` (Blackwell variant context)
- `cuda-cpp/sm120-optimization-guide.md` (RTX 5060 laptop local proxy)
- `cutlass-cpp/sm{100,103,120}-optimization-guide.md` (same three targets, CUTLASS C++ surface)
- For Ampere (sm_80 / sm_86 — A100, 3090Ti): use solbench's `references/NVIDIA_Manuals/ampere_{compatibility,tuning}/`

### PR layer trimmed (most PRs supported baseline-porting research, not metric_extractor)
- 8 framework PR pages (flash-attention, flashinfer, sglang, vllm, tensorrt-llm, tilelang, deepgemm, cccl-cub)
- All 11 `prs/by-topic/*.md` cross-repo aggregations
- `prs/index.md`, `prs/audit.md`, `prs/open-watchlist.md`
- `prs/pr-index.json` (1.6 MB metadata catalog)
- `prs/pr-scan-cache.json` (already dropped earlier — raw NCU-bearing GitHub PR excerpts)

### Blogs dropped (prose layer; solbench has stronger native NVIDIA docs)
- All 7 files under `references/blogs/` (Simon Boehm SGEMM, Veitner, Colfax, NVIDIA CUDA blog, Lei Mao, Yifan Yang + index)

### Topic dropped
- `routing/topics/communication.md` (solbench has 0 multi-GPU work)

### NCU-specific (dropped during earlier review rounds)
- `skills/profile-evidence/SKILL.md` (entire NCU-only skill)
- `references/ako4all/profiling-debugging-reference.md`
- `references/ako4all/cuda-cpp/vendored-docs/ncu-docs/` (7 files)

### Vendored-docs dropped during initial import (covered by solbench's `references/NVIDIA_Manuals/`)
- ~860 files from `ako4all/cuda-cpp/vendored-docs/` — solbench's `references/NVIDIA_Manuals/` has more comprehensive coverage (cutlass_cute 717, cuda_guide 654, ptx_isa 150, best_practices 74, blackwell_tuning 12, etc.).

### Catalog + tooling dropped
- `kernel-source-catalog.md` (30+ external repo URLs, most for dropped frameworks)
- `kernel-source-catalog.json` (same in machine-readable form)
- `scripts/kernel_pilot_tools/refresh_pr_knowledge.py` (DO NOT RUN AS-IS; would re-emit NCU strings and re-import dropped frameworks)
- `scripts/kernel_pilot_tools/build_kernel_source_catalog.py` (only had 4 REPOS entries; not a real generator)
- `scripts/kernel_pilot_tools/install-codex-skills.sh` (referenced dropped `profile-evidence` skill)
- `docs/kernel_pilot_assets/` (2 workflow PNGs from kernel-pilot; not solbench-operational)

## NCU strip status

All active NCU instructions have been removed across multiple review rounds.
Remaining NCU surface hits in this subtree are exclusively in:

- **Annotation markers** like `[NCU removed]` / `(NCU dropped per solbench env)` at the spots where original guidance was stripped — designed signposts.
- **SOLBENCH NOTICE banners** that explicitly tell the reader "no NCU per Rule 1" / "NCU is forbidden."
- **Meta-documentation** (this file, `README.md`) explaining what was dropped and why.

If you find an un-marked, non-documentation NCU reference that looks like
an active instruction (e.g., "run `ncu`...", "`ncu --set full`..."), treat
it as a follow-up fix.

## Counter-free substitutions

Where the original text recommended NCU profiling, the import substitutes
solbench's five counter-free analytic surfaces (per `CLAUDE.md § Counter-Free
Analytics Framework`):

1. **NVBit** — dynamic SASS / address traces / %clock64 phase cycles. (Solbench's active metric_extractor RLCR loop is building exactly this layer.)
2. **CUPTI Activity** — non-privileged wall-clock attribution.
3. **Nsight Systems (`nsys`)** — application-level timeline with optional NVTX.
4. **Static introspection** — `cuobjdump --dump-sass --dump-ptx`, `cuobjdump --list-elf`, `nvdisasm`, `ptxas -v`.
5. **compute-sanitizer** — `memcheck` / `racecheck` / `synccheck` / `initcheck`.

## Cross-references in surviving content

Some surviving files (e.g., `references/source-guides/cutlass.md`,
`references/prs/cutlass.md`, the `ako4all/` references) still contain
incidental cross-references to dropped frameworks (Triton, TileLang,
DeepGEMM, etc.) in body prose where the original kernel-pilot text
described related ecosystem context. These cross-references no longer
point to live content in this subtree. Treat any mention of a dropped
framework as **informational background only** — the corresponding
routing / source-guide / PR page is NOT in the trimmed import. If you
need to research a dropped framework, consult its upstream GitHub repo
directly.

## How to use this subtree

Read the adapted skill at `.claude/skills/kernel-knowledge/SKILL.md` first;
it routes you through this tree. Solbench's own knowledge base under
`.claude/knowledge/{sources,patterns,problems,cutlass_constructs}/` is owned
by the `implementing-sol-baselines` skill and remains authoritative for
SOL-ExecBench problem dossiers + canonical CUTLASS pattern recipes. This
imported subtree is a small, focused **CUDA-optimization-research layer**:
one production CUTLASS PR evidence page, code maps to
FlashInfer/FlashAttention/PyTorch ATen for algorithm reference, AKO4ALL
kernel templates for the CUDA C++ / CUTLASS C++ submission surface, and
sm_100 (B200) optimization guides.

## Provenance

- **Upstream source:** https://github.com/BBuf/kernel-pilot
- **Companion source:** https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS (parent of AKO4ALL references)
- **Imported:** 2026-05-15.
- **Trimmed:** 2026-05-15 (after Codex review pass; post-pivot research scan).
- **License:** see `references/ako4all/source-attribution.md` for the AKO4ALL attribution; the kernel-pilot top-level license carries through for the rest. When copying baseline kernel code from any upstream framework repo, record the upstream license + commit + path in the per-problem `baselines/<category>/<slug>/README.md`.
