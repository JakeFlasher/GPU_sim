# Kernel Knowledge Base (Imported into solbench — TRIMMED)

> **SOLBENCH NOTICE.** This file was imported from kernel-pilot, then
> aggressively trimmed for solbench. Submission surface is **CUDA C++ /
> CUTLASS C++ / CuTe C++ ONLY** (see `CLAUDE.md` Rule 5). Triton /
> CuTe DSL Python / cuTile Python / `cudnn_frontend` Python /
> `torch.compile` are forbidden as submissions — Triton routing/source-guide
> pages remain in this subtree as **algorithm references only**. solbench
> is also counter-free (no NCU per Rule 1) and counter-free profiling lives
> in `scripts/baselines_harness/`.

This directory is a focused lookup layer for **CUDA C++ / CUTLASS C++**
GPU-kernel work on the solbench (sol-execbench) project. It points to PRs,
source files, tests, and benchmarks worth reading for solbench's active
work — the metric_extractor RLCR loop (NVBit 1.8 atomic-aggregation) and
the CUTLASS-only baseline implementations.

The original kernel-pilot import covered 18 frameworks + 11 topics + 130
files. After the post-pivot trim, this subtree covers only the 5
frameworks (CUTLASS, PyTorch, Triton, FlashInfer, Flash-Attention) and 9
topics (no `communication`) that have non-trivial usage in solbench. See
`KB_IMPORT_README.md` for the full trim rationale.

## Layout

```text
kernel_pilot_imported/
├── README.md                   (this file)
├── KB_IMPORT_README.md         (trim governance + provenance)
├── index.json                  (registry — 5 kept frameworks)
├── routing/
│   ├── frameworks/             (5 .md — cutlass, flash-attention, flashinfer, pytorch, triton)
│   └── topics/                 (9 topic pages + index = 10 .md — attention, matmul-gemm, moe, normalization, rope,
│                                activation-fusion, sampling, quantization-fp8, kv-cache, index)
└── references/
    ├── index.md
    ├── prs/cutlass.md          (one production CUTLASS PR evidence page)
    ├── source-guides/          (5 .md)
    │   ├── cutlass.md
    │   ├── flash-attention.md
    │   ├── flashinfer.md
    │   ├── pytorch.md
    │   └── triton.md           (REFERENCE ONLY — Rule 5)
    └── ako4all/
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
        │   └── sm120-optimization-guide.md  (RTX 5090 local proxy)
        └── cutlass-cpp/
            ├── cutlass-cpp-overview.md
            ├── sm100-optimization-guide.md  (B200 cloud scoring target)
            ├── sm103-optimization-guide.md  (Blackwell variant context)
            └── sm120-optimization-guide.md  (RTX 5090 local proxy)
```

## Usage Rules

1. Start from `routing/topics/index.md` or `routing/frameworks/` for cross-cutting discovery.
2. For algorithm reference on the 5 frameworks below, read `references/source-guides/<repo>.md`.
3. For one production CUTLASS PR evidence example, read `references/prs/cutlass.md`. (The wider PR research corpus was trimmed; for plateau research on dropped frameworks, query GitHub directly.)
4. For CUDA C++ / CUTLASS C++ kernel mechanics (the **submission target**), read `references/ako4all/` core files plus the sm_100 optimization guides.
5. Treat Triton / FlashInfer / Flash-Attention algorithm references as **algorithm-only**: lift the algorithm into CUDA C++ / CUTLASS C++ for the *.cu hand-port. CUTLASS / PyTorch are the only submittable surfaces.
6. Log every source-derived idea with: framework, source path or symbol, hypothesis, measured result, and a do-not-reread key, in `baselines/<category>/<slug>/README.md`.
7. After two consecutive weak rounds (<1% improvement) in baseline tuning, expand research: read at least 50 new code-first sources before prose sources, then record a do-not-reread key for each.

## Dropped content

This subtree no longer contains routing, source-guides, or PR pages for:
**tensorrt-llm, cccl-cub, thunderkittens, tilelang, quack, tilekernels, veitner-blog, colfax-research, cuda-blog-kernels, deepgemm, sglang, vllm, cute-dsl.**
Cross-references to those names in body prose of surviving files are
informational background only — there is no live page to link to in this
subtree. For up-to-date research on those frameworks, query GitHub directly.

Architecture-specific guides KEPT for solbench's full local-test →
cloud-scoring workflow:

- **sm_100** (`cuda-cpp/sm100-optimization-guide.md` + `cutlass-cpp/sm100-optimization-guide.md`) — B200 cloud scoring target.
- **sm_103** (`{cuda-cpp,cutlass-cpp}/sm103-optimization-guide.md`) — Blackwell datacenter variant context.
- **sm_120** (`{cuda-cpp,cutlass-cpp}/sm120-optimization-guide.md`) — RTX 5090 local dev proxy.

For **sm_80 / sm_86 (Ampere — A100, 3090Ti)** local-test refs, kernel-pilot
did not ship Ampere guides. Use solbench's own
`references/NVIDIA_Manuals/ampere_compatibility/` (12 files) and
`references/NVIDIA_Manuals/ampere_tuning/` (12 files).

Architectures dropped (not in solbench's local or cloud target hardware):
sm_89 (Ada — RTX 4090, L40S) and sm_90 (Hopper H100). Solbench's
authoritative Blackwell reference is `references/NVIDIA_Manuals/blackwell_tuning/`.

## Refreshing the PR layer

The PR layer is intentionally a 1-page static snapshot (cutlass.md). The
generator script (`scripts/kernel_pilot_tools/refresh_pr_knowledge.py`)
was removed during the trim — it carried a DO NOT RUN AS-IS header
because re-running it would re-import dropped frameworks and re-introduce
NCU strings. If solbench needs to expand the PR research corpus later,
re-import from the upstream kernel-pilot repo and re-run the post-pivot
trim.
