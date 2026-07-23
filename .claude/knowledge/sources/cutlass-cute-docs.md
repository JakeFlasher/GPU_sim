---
Use when: looking up the CuTe C++ template surface (Layout, Tensor, MMA atom hierarchy, TMA descriptors, TiledCopy, TiledMma) used inside CUTLASS and CUDA C++ baselines. The Python `cute_dsl` submission language is OUT of scope per CLAUDE.md Rule 5.
Primary raw source: references/NVIDIA_Manuals/cutlass_cute/00_index.md
Supporting sources: cutlass/include/cute/; cutlass/include/cutlass/gemm/; cutlass/examples/
Repo touchpoints: cutlass/include/cute/tensor.hpp; cutlass/include/cute/layout.hpp; cutlass/include/cute/atom/; cutlass/include/cutlass/gemm/collective/
Patents anchored here: None
Out of scope: This stub points at the C++ template surface only. The Python CuTe DSL (`spec.languages: ["cute_dsl"]` in sol-execbench JSON) is NOT a valid submission surface for this project per CLAUDE.md Rule 5; our valid submission languages are `["cutlass"]` and `["cuda_cpp"]`.
Version family: cutlass-4.5.x (local clone) / cutlass-4.4.1 (eval Docker pin)
Default skill: implementing-sol-baselines
Routing tags: cutlass; cute; layout; tensor; mma; tma; tiled-copy; tiled-mma; collective; epilogue; gemm; tensor-core
---

# CUTLASS / CuTe (C++ Template Surface) Citation Stub

## Use when

Looking up CuTe layout algebra, tensor descriptors, MMA atom hierarchy, TMA descriptor construction, TiledCopy / TiledMma compositions, or CUTLASS collective mainloop / epilogue templates.

## Pointer

- Local mirror of NVIDIA docs: `references/NVIDIA_Manuals/cutlass_cute/` (38 markdown files; index at `00_index.md`; crawled by `references/NVIDIA_Manuals/scripts/crawler_multiTOC.py`).
- Source-of-truth: `cutlass/include/cute/` (C++ headers; local CUTLASS v4.5.0).
- Eval-Docker version: CUTLASS v4.4.1 (pinned by `sol-execbench/docker/Dockerfile`).

CUTLASS **3.8** was the first release with major SM100 / TCGen05 / TMEM / NVFP4 support -- older `CUTLASS_3_7` is not suitable for B200 baseline work and must not be specified as a `dependencies` value.

## Surface scope

- **In scope** (this project's submission surface): `cutlass::cute::Layout`, `cutlass::cute::Tensor`, `cutlass::cute::TiledCopy`, `cutlass::cute::TiledMma`, `cutlass::gemm::collective::CollectiveMma`, `cutlass::epilogue::collective::CollectiveEpilogue`, MMA atoms under `cute/atom/mma_atom.hpp`. All C++ template forms.
- **Out of scope**: the Python `cute_dsl` submission language (`spec.languages: ["cute_dsl"]` in sol-execbench JSON). Per CLAUDE.md Rule 5 our submissions are `spec.languages: ["cutlass"]` (pure C++) or `["cuda_cpp"]`.

## Cross-references

For canonical kernel-pattern recipes built on top of CuTe primitives, see the `patterns/` subcorpus -- in particular `patterns/gemm_baseline.md` for the GemmUniversal stack and `patterns/fp8_blockwise_quant.md` for the block-scaled MMA atom hierarchy.
