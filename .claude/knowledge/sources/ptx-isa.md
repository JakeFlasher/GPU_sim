---
Use when: looking up the PTX ISA reference — `tcgen05.mma` / `tcgen05.cp` instruction semantics, `cp.async` and bulk-async-copy variants, TMA descriptor instructions, `mbarrier` semantics, `wgmma.mma` and `wmma.mma` legacy paths, FP8/NVFP4 type encoding. Authoritative for low-level instruction behavior cited by GEMM and attention pattern cards.
Primary raw source: references/NVIDIA_Manuals/ptx_isa/
Supporting sources: references/NVIDIA_Manuals/cuda_inline_ptx/; references/NVIDIA_Manuals/blackwell_tuning/
Repo touchpoints: references/NVIDIA_Manuals/ptx_isa/
Patents anchored here: None
Out of scope: This stub is a citation gateway. Inline PTX is allowed in CUDA C++ submissions via the `cuda-inline-ptx` syntax; raw .ptx-only submissions are not.
Version family: cuda-13.1; sol-execbench-2026
Default skill: interpreting-ptx-isa
Routing tags: ptx-isa; ptx; tcgen05; cp-async; tma; mbarrier; wgmma; instruction-set; sm_100a; sm_120; gemm; attention; fp8; nvfp4
---

# PTX ISA Reference (Citation Stub)

## Use when

Looking up:
- `tcgen05.mma{.kind}.{shape}` instruction encoding for B200 5th-generation tensor cores.
- `tcgen05.cp` for TMEM <-> registers / shared-memory copies.
- `cp.async.bulk.tensor` (TMA) for global <-> shared async copies on sm_100a.
- `mbarrier.arrive` / `mbarrier.wait` for producer-consumer warp synchronisation.
- FP8 (e4m3 / e5m2) and NVFP4 (e2m1) type encodings and the e8m0 scale type.

## Pointer

The mirrored crawl lives at `references/NVIDIA_Manuals/ptx_isa/`. For embedding PTX inline in CUDA C++ source, see `cuda-inline-ptx.md`.

## Out of scope for this stub

This is a low-level reference. Submissions are CUDA C++ / CUTLASS C++; pure `.ptx` source files are not a valid `spec.languages` value. Inline PTX (via `asm volatile(...)` blocks) inside CUDA C++ is fine.
