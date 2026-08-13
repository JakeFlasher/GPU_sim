---
name: kernel-knowledge
description: Route SOL-ExecBench and CUDA kernel work to the retained kernel-pilot references, framework guides, architecture notes, and local SOL dossiers. Use when selecting an implementation pattern or researching a SOL baseline.
---

# Kernel knowledge for SOL work

This is an opt-in knowledge router for the retained SOL/kernel campaign. It
does not impose a profiling ban, research loop, source-count quota, ledger, or
submission workflow.

## Start here

- Local problem dossiers: `.claude/knowledge/problems/`
- Local implementation patterns: `.claude/knowledge/patterns/`
- Local source cards: `.claude/knowledge/sources/`
- Imported kernel-pilot index:
  `references/kernel_pilot_imported/routing/topics/index.md`
- Imported source-guide index:
  `references/kernel_pilot_imported/references/index.md`
- NVIDIA manuals: `references/NVIDIA_Manuals/`

For a concrete kernel, read the problem dossier, one relevant pattern or topic
card, and the closest existing baseline. Expand to the framework source guides
only when the first implementation or measurement exposes a question.

## Useful imported routes

- CUTLASS: `references/kernel_pilot_imported/routing/frameworks/cutlass.md`
- FlashAttention:
  `references/kernel_pilot_imported/routing/frameworks/flash-attention.md`
- FlashInfer:
  `references/kernel_pilot_imported/routing/frameworks/flashinfer.md`
- PyTorch: `references/kernel_pilot_imported/routing/frameworks/pytorch.md`
- Triton algorithm references:
  `references/kernel_pilot_imported/routing/frameworks/triton.md`
- CUDA C++ templates:
  `references/kernel_pilot_imported/references/ako4all/cuda-cpp-kernel-reference.md`
- CUTLASS/CuTe templates:
  `references/kernel_pilot_imported/references/ako4all/cutlass-cpp-kernel-reference.md`

Use source code in accordance with its license and record attribution when code
is copied or adapted. When the user requests a from-scratch implementation,
use prior implementations only to understand the algorithm.

Pair this router with `implementing-sol-baselines` for implementation and
`run-sol-campaign` for broader captures. NCU, NVBit, CUPTI, Nsight Systems,
static inspection, and compute-sanitizer are all available when technically
appropriate. Historical counter-free notes are context, not current policy.
