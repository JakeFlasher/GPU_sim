---
Use when: looking up the NVCC compiler driver — `-gencode arch=compute_100a,code=sm_100a` flag semantics, `-std=c++17` / `-std=c++20` requirements, `--use_fast_math` / `--expt-relaxed-constexpr` / `--extended-lambda` flags, PTX-vs-cubin emission, NVRTC vs offline-compile behavior. The B200 problem packager auto-injects sm_100a; this guide is the authority for what those flags actually do.
Primary raw source: references/NVIDIA_Manuals/nvcc_guide/
Supporting sources: references/NVIDIA_Manuals/cuda_guide/; sol-execbench/docs/solution.md
Repo touchpoints: references/NVIDIA_Manuals/nvcc_guide/
Patents anchored here: None
Out of scope: This stub is a citation gateway. NVCC supports many flags that touch host-driver behaviour (e.g. `-rdc`, `--cudart=shared`); pick flags compatible with the `sol-execbench` JIT build pipeline only.
Version family: cuda-13.1; sol-execbench-2026
Default skill: compiling-cuda-artifacts
Routing tags: nvcc; nvcc-guide; compile-flags; gencode; sm_100a; sm_120; compile-options; jit; cubin; ptxas; gemm; attention; cutlass
---

# NVCC Compiler Driver Guide (Citation Stub)

## Use when

Looking up:
- `-gencode arch=compute_100a,code=sm_100a` for B200 builds (the SOL-ExecBench problem packager auto-injects this).
- Standard-library version flags (`-std=c++17` / `-std=c++20`).
- Fast-math (`--use_fast_math`) effects on trig / reciprocal precision.
- Relaxed-constexpr (`--expt-relaxed-constexpr`) required by CUTLASS templates.
- Extended-lambda (`--extended-lambda`) for CUTLASS epilogue callbacks.
- PTX-vs-cubin emission and `ptxas -v` verbose reporting.

## Pointer

The mirrored crawl lives at `references/NVIDIA_Manuals/nvcc_guide/`. The `sol-execbench` evaluator builds via `torch.utils.cpp_extension`; `spec.compile_options` in `solution.json` translate directly to NVCC flags.

## Out of scope for this stub

Flags involving relocatable device code (`-rdc=true`), shared-cudart (`--cudart=shared`), or host-driver linkage are off the supported submission path. Stay within the flags shown working in `sol-execbench/examples/cutlass/` and `examples/cuda_cpp/`.
