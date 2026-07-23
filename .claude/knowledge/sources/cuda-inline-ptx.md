---
Use when: looking up the Inline-PTX-in-CUDA-C++ guide — `asm volatile(...)` syntax for embedding PTX inside `__device__` / `__global__` functions, constraint letters (`r` / `l` / `f` / `d` / `h`), `"memory"` clobber semantics, scoped-state mutators. Most attention / GEMM hot paths drop into inline PTX for instructions CUTLASS templates do not expose.
Primary raw source: references/NVIDIA_Manuals/cuda_inline_ptx/
Supporting sources: references/NVIDIA_Manuals/ptx_isa/; references/NVIDIA_Manuals/cuda_guide/
Repo touchpoints: references/NVIDIA_Manuals/cuda_inline_ptx/
Patents anchored here: None
Out of scope: This stub is a citation gateway. Submitting raw `.ptx` source files (without a CUDA C++ wrapper) is not a supported `spec.languages` value; inline PTX inside CUDA C++ kernels is fine.
Version family: cuda-13.1; sol-execbench-2026
Default skill: interpreting-ptx-isa
Routing tags: cuda-inline-ptx; inline-ptx; asm-volatile; ptx; constraints; clobber; cp-async; tcgen05; sm_100a; sm_120; gemm; attention
---

# CUDA Inline PTX Assembly (Citation Stub)

## Use when

Looking up:
- `asm volatile("..." : outputs : inputs : clobbers);` syntax inside `__device__` / `__global__` functions.
- Constraint letters: `r` (.u32), `l` (.u64), `h` (.u16), `f` (.f32), `d` (.f64), `n` (immediate).
- The `"memory"` clobber for ordering relative to surrounding memory accesses.
- When CUTLASS templates do NOT expose a needed PTX instruction (rare on sm_100a since `cute::TiledMma` covers most), drop into inline PTX with the syntax this guide defines.

## Pointer

The mirrored crawl lives at `references/NVIDIA_Manuals/cuda_inline_ptx/`. For the underlying instruction semantics see `ptx-isa.md`.

## Out of scope for this stub

Use inline PTX sparingly: prefer CUTLASS / CuTe abstractions where they exist. Raw `.ptx` source files (not wrapped in CUDA C++) are NOT a valid `spec.languages` value.
