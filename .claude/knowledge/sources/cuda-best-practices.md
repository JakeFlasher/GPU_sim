---
Use when: looking up the CUDA C++ Best Practices Guide — coalesced-access patterns, occupancy guidelines, shared-memory bank-conflict avoidance, async-copy (`cp.async`) usage, register-pressure heuristics. Most patterns in the atlas cite this guide for the "why" behind their tile-shape and schedule choices.
Primary raw source: references/NVIDIA_Manuals/best_practices/
Supporting sources: references/NVIDIA_Manuals/cuda_guide/; references/NVIDIA_Manuals/blackwell_tuning/
Repo touchpoints: references/NVIDIA_Manuals/best_practices/
Patents anchored here: None
Out of scope: This stub is a citation gateway. The guide's advice on multi-stream overlap is informational only — submissions may only use `at::cuda::getCurrentCUDAStream()` (CLAUDE.md Rule 7). Profiling advice involving NCU / Nsight Compute or privileged CUPTI surfaces is out of scope for our dev loop (CLAUDE.md Rules 1, 3).
Version family: cuda-13.1; sol-execbench-2026
Default skill: optimizing-cuda-kernels
Routing tags: cuda-best-practices; best-practices; coalescing; occupancy; cp-async; shared-memory; bank-conflict; gemm; attention; rmsnorm; sm_100a; sm_120
---

# CUDA C++ Best Practices Guide (Citation Stub)

## Use when

Looking up best-practice rationale for performance choices in pattern cards:
- Coalesced global-memory access shapes.
- Async-copy (`cp.async`) staging and double-buffering.
- Occupancy / launch-bounds tuning.
- Bank-conflict-free shared-memory layouts.

## Pointer

The mirrored crawl lives at `references/NVIDIA_Manuals/best_practices/`.

## Out of scope for this stub

Profiling advice that depends on NCU is informational only — our dev loop has neither (`CLAUDE.md` Critical Rules 1 and 3). Multi-stream overlap suggestions are off-limits per Rule 7.
