---
Use when: looking up the CUDA C++ Programming Guide — execution model, memory hierarchy, host/device API, stream / event semantics, tensor-core programming model, fast-math intrinsics. Authoritative source for general CUDA C++ behavior referenced by GEMM, attention, RMSNorm, and other patterns in the atlas.
Primary raw source: references/NVIDIA_Manuals/cuda_guide/
Supporting sources: references/NVIDIA_Manuals/best_practices/; references/NVIDIA_Manuals/ptx_isa/
Repo touchpoints: references/NVIDIA_Manuals/cuda_guide/
Patents anchored here: None
Out of scope: This stub is a citation gateway. The guide describes the host-side `cudaStream_t` API; submissions must launch on `at::cuda::getCurrentCUDAStream()` only (CLAUDE.md Rule 7). It also describes `nvidia-smi` clock policy and various CUPTI surfaces; those are forbidden submission paths regardless of what the guide documents (CLAUDE.md Rules 1, 2, 3).
Version family: cuda-13.1; sol-execbench-2026
Default skill: optimizing-cuda-kernels
Routing tags: cuda-guide; cuda-c++; cuda-programming-guide; execution-model; memory-hierarchy; intrinsics; gemm; attention; rmsnorm; sm_100a; sm_120
---

# CUDA C++ Programming Guide (Citation Stub)

## Use when

Looking up:
- CUDA execution model basics (grid / block / warp / thread).
- Memory hierarchy (global / L2 / shared / register / constant / TMEM).
- Stream / event semantics (read-only — submissions only use the current PyTorch stream).
- Fast-math intrinsics (`__expf`, `__cosf`, `__frsqrt_rn`, etc.).
- Tensor-core programming basics (mma / wmma / TMA).

## Pointer

The mirrored crawl lives at `references/NVIDIA_Manuals/cuda_guide/`. For implementation patterns that use these primitives, see `.claude/knowledge/patterns/` — most cards link back to specific subsections of this guide.

## Out of scope for this stub

The guide is informational. Forbidden submission surfaces documented in CLAUDE.md (Python DSL solution languages, NCU, clock-locking, privileged CUPTI, host-driver work) are still off-limits even when the guide describes them.
