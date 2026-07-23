---
Use when: working with block-scaled MMA atoms — the sm_100a fifth-generation tensor-core instructions for FP8 (e4m3 / e5m2 with per-block scaling) and NVFP4 (e2m1 with FP8-e8m0 per-16-element-block scaling) GEMMs. Construct details that complement the `fp8_blockwise_quant` pattern card.
Primary raw source: cutlass/include/cutlass/detail/sm100_blockscaled_layout.hpp
Supporting sources: cutlass/include/cutlass/gemm/collective/builders/sm100_blockscaled_umma_builder.inl; references/NVIDIA_Manuals/blackwell_tuning/; references/NVIDIA_Manuals/ptx_isa/
Repo touchpoints: cutlass/include/cutlass/detail/sm100_blockscaled_layout.hpp; cutlass/include/cutlass/gemm/collective/builders/sm100_blockscaled_umma_builder.inl
Patents anchored here: None
Out of scope:
  - Python / Triton / CuTe DSL Python / cuTile Python / cudnn_frontend Python / torch.compile submissions (CLAUDE.md Rule 5).
  - NCU / `nv-nsight-cu-cli` / `.ncu-rep`/`.ncu-repz` artefacts on any device, including local 5060 (CLAUDE.md Rule 1).
  - `nvidia-smi --lock-gpu-clocks` / `--lock-memory-clocks` / persistence / power / MIG / compute-mode mutation (CLAUDE.md Rule 2).
  - Privileged CUPTI: Events, Metrics, Range Profiling, PM Sampling, PerfWorks (CLAUDE.md Rule 3; CUPTI Activity remains allowed).
  - Host-driver work: NVIDIA driver install, DKMS, MOK, `modprobe nvidia`, kernel-module parameter toggles, reboot workflows (CLAUDE.md Rule 4).
  - User CUDA streams, background host threads, `torch.jit.fork`, null-stream side effects, CUDA Graphs without a per-problem note (CLAUDE.md Rule 7).
  - Evaluator-state exploits: `data_ptr` caching, lazy-eval `__eq__` on `FakeTensor`, monkey-patched `Event.elapsed_time` / `do_bench` (CLAUDE.md Rule 6).
  - Precision-contract violations: precision downgrade (e.g. FP16 compute upcast to FP32 output) to pass tolerance (CLAUDE.md Rule 8).
Version family: cutlass-4.5.x (sm_100a block-scaled MMA path); sol-execbench-2026
Default skill: implementing-sol-baselines
Routing tags: block-scaled; block-scale; blockscaled-mma; fp8; nvfp4; e4m3; e5m2; e2m1; e8m0; tcgen05; tmem; quant
---

# Block-Scaled MMA (CUTLASS Construct)

## Use when

Choosing the sm_100a fifth-generation tensor-core path for FP8 e4m3 / e5m2 GEMMs with per-1x128-block scaling, or for NVFP4 e2m1 GEMMs with per-16-element-block scaling.

## Concept

Block-scaled MMA splits each operand tensor into small blocks (1x128 for FP8 channels, 16 elements along K for NVFP4) and stores a per-block scale alongside the quantised values. The `tcgen05.mma{.block_scale}` instruction multiplies the scale into the partial sum at issue time, recovering FP32-equivalent accuracy from low-precision storage.

Operand encoding:
- **FP8 e4m3 / e5m2.** 8-bit storage, FP8 scale (typically `float` or `__nv_bfloat16`) per 1x128 block.
- **NVFP4 e2m1.** 4-bit storage, FP8-e8m0 scale (8-bit unsigned biased exponent, no mantissa) per 16-element block along K.

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

Block-scaled MMA is sm_100a-only on the 5th-generation tensor core. On sm_120 the FP8 path runs through an older Sm89-style atom (no block scaling, scale applied in the epilogue); NVFP4 has no sm_120 equivalent — NVFP4 baselines cannot be compile-checked locally at the instruction level. Either build only for sm_100a or fall back to a BF16 dtype for the local-smoke build.

Evidence-label conventions per `CLAUDE.md` live in `baselines/<category>/<slug>/README.md`, NOT in this construct card.

## Pattern cards that compose this construct

`fp8_blockwise_quant` (the primary pattern), `gemm_baseline` (when ElementA/B is FP8/NVFP4), `moe_dispatch` (FP8 MoE variants), `mlp_gelu_geglu` (Quant variants), `attention_core` / `attention_backward` (FP8 Quant attention variants).
