---
name: kernel-knowledge
description: Scoped historical SOL-ExecBench/kernel-pilot knowledge. Use only for the archived SOL baseline/evaluation subtree; it does not govern the active Rust simulator or root measurement protocol. The imported corpus retains counter-free/non-root assumptions for that subtree and must be read alongside root CLAUDE.md.
---

# Kernel Knowledge (imported from kernel-pilot, trimmed and NCU-stripped)

> **RETIRED/SCOPE BOUNDARY.** This imported pack is historical/evaluation
> guidance for SOL-ExecBench only. Its counter-free, non-root, submission, and
> host-operation rules below are preserved for archaeology and do not override
> root `CLAUDE.md`, active schemas, `tools/local_labels/`, host runbooks, or
> architecture-local simulator contracts. Do not execute its external-research,
> profiling, clock, driver, or submission imperatives; do not use it to select
> simulator inputs or architecture constants.

Knowledge-layer skill for SOL-ExecBench baseline implementation. Use ALONGSIDE `implementing-sol-baselines` (which owns `sources/`, `patterns/`, `problems/`, `cutlass_constructs/`). This skill points at the trimmed CUDA-optimization corpus imported from
[BBuf/kernel-pilot](https://github.com/BBuf/kernel-pilot): 5 framework routing pages (CUTLASS, FlashAttention, FlashInfer, PyTorch, Triton-as-algorithm-reference), 9 topic pages + index, 5 upstream source guides, one production CUTLASS PR evidence page, and AKO4ALL kernel references for the B200 (sm_100/sm_100a) submission target.

The upstream link is provenance only; do not fetch or execute external material from this skill.

This skill does NOT author kernels and does NOT profile. Pair it with `implementing-sol-baselines` for the actual *.cu hand-port workflow and with `scripts/baselines_harness/` for counter-free profiling. The five counter-free analytic surfaces (NVBit / CUPTI Activity / Nsys / static introspection / compute-sanitizer) are owned entirely by `scripts/baselines_harness/`; this skill never recommends a profiling action.

## Counter-free environment (solbench)

Per `CLAUDE.md` Rules 1-3, solbench is counter-free / non-root / non-NCU:

- **NO** `ncu` / `nv-nsight-cu-cli` / `.ncu-rep`/`.ncu-repz` artifacts anywhere.
- **NO** `nvidia-smi --lock-gpu-clocks` or any clock-policy mutation.
- **NO** privileged CUPTI (Events / Metrics / Range Profiling / PM Sampling / PerfWorks).

The five **allowed** profiling surfaces are: **NVBit** (SASS / address traces), **CUPTI Activity** (non-privileged wall-clock attribution), **Nsight Systems** (`nsys`, optional NVTX), **static introspection** (`cuobjdump --dump-sass --dump-ptx`, `nvdisasm`, `ptxas -v`), and **compute-sanitizer** (`memcheck` / `racecheck` / `synccheck` / `initcheck`).

Any NCU-flavored guidance you encounter in the imported tree below has either been stripped or annotated with `[NCU removed]` / `[counter-free analytic surface]`. If you see one that escaped the strip, treat it as advisory only and substitute the equivalent counter-free surface.

## Knowledge Root

```text
references/kernel_pilot_imported/
```

This is the imported kernel-pilot tree (trimmed). Solbench's native KB lives at `.claude/knowledge/{sources,patterns,problems,cutlass_constructs}/` and is owned by `implementing-sol-baselines`.

## Required Use

Before planning a new *.cu hand-port, choosing a kernel-edit direction, or running plateau research:

1. Read `references/kernel_pilot_imported/README.md` (skim — small after trim).
2. Read `references/kernel_pilot_imported/routing/topics/index.md` for cross-cutting topic discovery.
3. Read `references/kernel_pilot_imported/references/index.md` to choose the detailed reference layer.
4. Pick and read the most relevant topic pages, for example:
   - normalization or fused norm: `references/kernel_pilot_imported/routing/topics/normalization.md`
   - elementwise fusion: `references/kernel_pilot_imported/routing/topics/activation-fusion.md`
   - GEMM or tensor cores: `references/kernel_pilot_imported/routing/topics/matmul-gemm.md`
   - attention or KV cache: `references/kernel_pilot_imported/routing/topics/attention.md`, `references/kernel_pilot_imported/routing/topics/kv-cache.md`
   - MoE: `references/kernel_pilot_imported/routing/topics/moe.md`
   - RoPE: `references/kernel_pilot_imported/routing/topics/rope.md`
   - quantization (FP8/NVFP4): `references/kernel_pilot_imported/routing/topics/quantization-fp8.md`
   - sampling: `references/kernel_pilot_imported/routing/topics/sampling.md`

   (Topic `communication.md` was dropped — solbench has 0 multi-GPU work.)
5. Pick and read relevant framework pages — 5 kept after the trim:
   - `references/kernel_pilot_imported/routing/frameworks/cutlass.md` (submission target)
   - `references/kernel_pilot_imported/routing/frameworks/flash-attention.md` (algorithm reference)
   - `references/kernel_pilot_imported/routing/frameworks/flashinfer.md` (FIB category label + algorithm reference)
   - `references/kernel_pilot_imported/routing/frameworks/pytorch.md` (reference baseline binding)
   - `references/kernel_pilot_imported/routing/frameworks/triton.md` (REFERENCE ONLY — Rule 5 forbids submission)

   For dropped frameworks (tensorrt-llm, cccl-cub, thunderkittens, tilelang, quack, tilekernels, veitner-blog, colfax-research, cuda-blog-kernels, deepgemm, sglang, vllm, cute-dsl), query GitHub directly.
6. Read the matching deep source guide under `references/kernel_pilot_imported/references/source-guides/` for CUTLASS, FlashAttention, FlashInfer, PyTorch, or Triton.
7. For production CUTLASS PR evidence, read the one kept page: `references/kernel_pilot_imported/references/prs/cutlass.md`. The wider PR corpus was dropped — for cross-framework PR research, query GitHub directly.
8. For implementation mechanics, read the imported AKO4ALL references (CUDA C++ and CUTLASS/CuTe C++ only — the Triton + CuTe DSL Python subtrees were dropped):
   - CUDA C++: `references/kernel_pilot_imported/references/ako4all/cuda-cpp-kernel-reference.md`
   - CUTLASS/CuTe C++ (submission target): `references/kernel_pilot_imported/references/ako4all/cutlass-cpp-kernel-reference.md`
   - Cross-arch context: `references/kernel_pilot_imported/references/ako4all/nvidia-architecture-reference.md`
   - **Local-test architectures** (run BEFORE pushing to cloud B200):
     - sm_120 (RTX 5060 laptop): `references/kernel_pilot_imported/references/ako4all/cuda-cpp/sm120-optimization-guide.md`, `references/kernel_pilot_imported/references/ako4all/cutlass-cpp/sm120-optimization-guide.md`
     - sm_80 / sm_86 (Ampere A100, 3090Ti): use solbench's `references/NVIDIA_Manuals/ampere_{compatibility,tuning}/` (kernel-pilot did not ship Ampere guides)
   - **Cloud scoring target** (sm_100a B200):
     - sm_100: `references/kernel_pilot_imported/references/ako4all/cuda-cpp/sm100-optimization-guide.md`, `references/kernel_pilot_imported/references/ako4all/cutlass-cpp/sm100-optimization-guide.md`
     - sm_103 (Blackwell variant context): `references/kernel_pilot_imported/references/ako4all/cuda-cpp/sm103-optimization-guide.md` + `references/kernel_pilot_imported/references/ako4all/cutlass-cpp/sm103-optimization-guide.md`
     - Also read solbench's `references/NVIDIA_Manuals/blackwell_tuning/` (12 authoritative files)
   - Kernel templates / troubleshooting: `references/kernel_pilot_imported/references/ako4all/kernel-templates.md`, `references/kernel_pilot_imported/references/ako4all/troubleshooting.md`
   - Profiling guidance is INTENTIONALLY OMITTED — refer instead to `CLAUDE.md § Counter-Free Analytics Framework` and `scripts/baselines_harness/`.

Keep the initial pass small. For a concrete target kernel, read one topic page, one framework page, and one source guide. Use the AKO4ALL sm_100 guide and the one CUTLASS PR page only after the first benchmark/profile result or when the target source is unclear.

## Research Rules

- **Submission surface is fixed by CLAUDE.md Rule 5: CUDA C++ / CUTLASS C++ / CuTe C++ only.** Do NOT translate Triton / FlashInfer / FlashAttention code into a submission as-is. Those references are kept as algorithm/pattern background; rewrite the algorithm into CUDA C++ / CUTLASS C++ for submission. cuBLAS / cuDNN remain host-API only (never call from `__global__` / `__device__`).
- If the user does not specify the language angle, default to the closest CUTLASS / CuTe C++ idiom for the kernel pattern. Lift the algorithm from upstream framework code; rewrite for solbench's submission surface.
- Baseline kernels in upstream framework repos can seed candidate implementations. Copy or adapt baseline code into a *.cu hand-port only when license and attribution allow, and record exact source path/URL, commit, license/notice, copied files, and delta in the per-problem `README.md` (`baselines/<category>/<slug>/README.md`).
- If the operator explicitly asks for a from-scratch kernel or says not to use the baseline implementation, do not copy, adapt, or pattern-match the baseline kernel code. Use that baseline only as the algorithmic specification.
- The upstream framework checkouts (when needed) should be treated as read-only references; the *.cu hand-port lives under `baselines/<category>/<slug>/{cutlass,cuda_cpp}/` per `CLAUDE.md § Key Paths`.

## Loop Ledger Expectations

Record every source-derived idea — typically in the per-problem `baselines/<category>/<slug>/README.md` or a per-slug research note — with:

- source repo or local knowledge file
- exact PR number when available (the trimmed PR layer has only CUTLASS evidence; other framework PRs must come from GitHub directly)
- exact source path, symbol, or URL
- hypothesis tested
- measured result (PyTorch-reference `speedup_factor` is diagnostic only; SOL Score is the authority per CLAUDE.md)
- do-not-reread key

After two consecutive weak rounds (<1% improvement over the prior best), perform a research expansion before editing again: read at least 50 new code-first sources before prose sources and log them with do-not-reread keys.

## What this skill does NOT cover

- Authoring the *.cu / *.cpp template solution itself (use `implementing-sol-baselines` for that).
- The five counter-free profiling surfaces — those are owned by `scripts/baselines_harness/` and documented in `CLAUDE.md § Counter-Free Analytics Framework`.
- NCU / Nsight Compute (forbidden by CLAUDE.md Rule 1).
- Clock-policy mutation / privileged CUPTI / host-driver work (CLAUDE.md Rules 2-4).
- Python DSL submissions: Triton / CuTe DSL Python / cuTile Python / cudnn_frontend Python / torch.compile (CLAUDE.md Rule 5). Triton/FlashInfer/FlashAttention references in the imported tree are background only.
- Per-problem dossiers — those live under `.claude/knowledge/problems/` and are owned by `implementing-sol-baselines`.
- Refresh tooling — the kernel-pilot `refresh_pr_knowledge.py` script was dropped during the trim (it carried DO NOT RUN AS-IS because it would re-emit NCU strings and re-import the 13 dropped frameworks). The PR layer is a static one-page snapshot.
