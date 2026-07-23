# kernel-knowledge References (trimmed)

This directory holds the detailed reference layer for the trimmed
kernel-pilot import. Keep `routing/frameworks/*.md` and
`routing/topics/*.md` as lightweight routing pages; load files here only
when the task needs the extra detail.

## Core Kernel Loop References

For solbench's submission target (CUDA C++ / CUTLASS C++ / CuTe C++) and
its B200 (sm_100a) scoring architecture, the core references kept after
the trim are:

| Need | Read |
| --- | --- |
| Iteration loop, harness layout, validation gates | `ako4all/ako4all-kernel-loop.md` |
| CUDA C++ / PTX candidate work | `ako4all/cuda-cpp-kernel-reference.md`, `ako4all/cuda-cpp/cuda-cpp-overview.md` |
| CUTLASS / CuTe C++ candidate or baseline work (submission target) | `ako4all/cutlass-cpp-kernel-reference.md`, `ako4all/cutlass-cpp/cutlass-cpp-overview.md` |
| Cross-architecture context (sm_89 / sm_90 / sm_120) | `ako4all/nvidia-architecture-reference.md` |
| B200 (sm_100a) optimization guides | `ako4all/cuda-cpp/sm100-optimization-guide.md`, `ako4all/cutlass-cpp/sm100-optimization-guide.md` |
| Kernel templates and debugging | `ako4all/kernel-templates.md`, `ako4all/troubleshooting.md` |
| One production CUTLASS PR evidence example | `prs/cutlass.md` |

The full AKO4ALL attribution is preserved in `ako4all/source-attribution.md`.

## Source Deep Guides (kept frameworks only)

| Source | Deep guide |
| --- | --- |
| CUTLASS / CuTe (C++ submission target) | `source-guides/cutlass.md` |
| PyTorch (reference baseline binding) | `source-guides/pytorch.md` |
| FlashAttention (algorithm reference) | `source-guides/flash-attention.md` |
| FlashInfer (FIB category label / algorithm reference) | `source-guides/flashinfer.md` |
| Triton (algorithm reference only — Rule 5 forbids submission) | `source-guides/triton.md` |

The 13 other framework source-guides were dropped in the trim
(tensorrt-llm, cccl-cub, thunderkittens, tilelang, quack, tilekernels,
veitner-blog, colfax-research, cuda-blog-kernels, deepgemm, sglang, vllm,
cute-dsl) — none had non-trivial usage in solbench's active work. For
those frameworks, query GitHub directly.

## PR Layer

The wider PR research corpus (8 framework PR pages + 11 by-topic
aggregations + pr-index.json + audit + open-watchlist) was dropped in the
post-pivot trim. One production CUTLASS PR evidence page is kept as a
template / example: `prs/cutlass.md`.

If solbench resumes broad baseline-porting research, re-import the PR
layer from the upstream kernel-pilot repo and re-run the post-pivot trim.

## Architecture-specific guides (covering solbench's local-test → cloud-scoring workflow)

| Hardware | Architecture | Status | Path |
|---|---|---|---|
| 3090Ti / A100 | sm_86 / sm_80 (Ampere) | local-test refs | solbench's `references/NVIDIA_Manuals/ampere_{compatibility,tuning}/` (11+11 = 22 files; kernel-pilot did not ship Ampere guides) |
| RTX 5060 laptop | sm_120 (consumer Blackwell) | local dev proxy KEPT | `ako4all/{cuda-cpp,cutlass-cpp}/sm120-optimization-guide.md` |
| Blackwell variant | sm_103 | context KEPT | `ako4all/{cuda-cpp,cutlass-cpp}/sm103-optimization-guide.md` |
| B200 (cloud) | sm_100a (datacenter Blackwell) | cloud scoring KEPT | `ako4all/{cuda-cpp,cutlass-cpp}/sm100-optimization-guide.md` (also see solbench's `references/NVIDIA_Manuals/blackwell_tuning/` — 12 authoritative files) |
| RTX 4090 / L40S | sm_89 (Ada) | DROPPED | Not in solbench hardware path |
| H100 | sm_90 (Hopper) | DROPPED | Not in solbench hardware path |

`ako4all/architectures/` was emptied (sm_89 / sm_90 dropped) and the directory removed. The cute-dsl/ subtree (including its sm_120/sm_103 guides) was dropped because CuTe DSL Python is forbidden by Rule 5.

For solbench's authoritative Blackwell tuning reference on sm_100a (B200), prefer `references/NVIDIA_Manuals/blackwell_tuning/` over the imported `ako4all/` files when they conflict.

## Use Rules

- Prefer CUDA C++ / CUTLASS C++ implementation evidence over Triton /
  FlashInfer algorithm references (the latter cannot be submitted per Rule 5).
- For plateau research on dropped frameworks, query GitHub directly.
- Log every source-derived idea in `baselines/<category>/<slug>/README.md`
  with: PR/source path, hypothesis, measured result, do-not-reread key.
- Counter-free profiling is owned by solbench's `scripts/baselines_harness/`
  (the 5 surfaces: NVBit / CUPTI Activity / Nsys / static-introspection /
  compute-sanitizer). This subtree does not document profiling workflows.
