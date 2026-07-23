---
Use when: implementing FP8 (e4m3 / e5m2) or NVFP4 (e2m1) block-scaled quantization kernels for SOL-ExecBench Quant-category problems. Covers FP8 block-wise scaling (1x128 or 128x128 blocks) and NVFP4 16-element block scaling on Blackwell sm_100a's 5th-generation tensor cores. sm_120 fallback included.
Primary raw source: references/NVIDIA_Manuals/blackwell_tuning/00_index.md
Supporting sources: references/NVIDIA_Manuals/cutlass_cute/; cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl; references/NVIDIA_Manuals/ptx_isa/
Repo touchpoints: cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl; cutlass/include/cute/atom/mma_traits_sm100.hpp; baselines/Quant/<problem_slug>/cutlass/
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
Version family: cutlass-4.5.x (sm_100a tcgen05 / block-scaled MMA); sol-execbench-2026
Default skill: implementing-sol-baselines
Also relevant skills: tuning-blackwell-sm120-kernels
Routing tags: fp8; nvfp4; block-scale; quant; e4m3; e5m2; e2m1; tcgen05; tmem
---

# FP8 / NVFP4 Block-Scaled Quantization (Canonical CUTLASS Recipe)

## Use when

You're starting a SOL-ExecBench Quant-category problem (e.g. `problem_001_fp8_attention_output_projection`, `problem_004_fp8_moe_expert_linear`, `problem_001_nvfp4_*`). The Quant category has 33 problems split: 18 use FP8 block-wise scaling, 15 use NVFP4 16-element block scaling per the SOL-ExecBench paper §3.2.

## Block-scaling layouts

### FP8 (e4m3 / e5m2)

- **Block size.** 1x128 (per-token per 128 contiguous channels) or 128x128 (per 128x128 tile). The SOL-ExecBench problems mostly use BlockWise1x128.
- **Scale dtype.** Typically `float` or `__nv_bfloat16` scale per block. Scales live in a separate tensor of shape `[M / block_M, K / block_K]`.
- **MMA atom.** `cutlass::arch::Sm100_QMMA_F32_E4M3_E4M3_F32_SS` on sm_100a (uses TCGen05); `Sm89_QMMA_F32_E4M3_E4M3_F32_SS` for Hopper-style fallback on sm_120.

### NVFP4 (e2m1)

- **Block size.** 16-element blocks along the K axis (the SOL-ExecBench paper's standard).
- **Scale dtype.** FP8 e8m0 (8-bit unsigned biased exponent, no mantissa) -- one scale per 16-element block.
- **MMA atom.** `cutlass::arch::Sm100_QMMA_F32_E2M1_E2M1_F32_SS` on sm_100a (5th-generation tensor core block-scaled MMA). NO direct sm_120 equivalent -- NVFP4 is sm_100a-conditional.

## Recipe sketch

For FP8 GEMMs (a Quant-category MLP gate / up / down projection or attention output projection):

```cpp
using Gemm = cutlass::gemm::device::GemmUniversal<
    cutlass::float_e4m3_t,                                // ElementA
    cutlass::layout::RowMajor,
    cutlass::float_e4m3_t,                                // ElementB
    cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t,                                  // ElementC (output)
    cutlass::layout::RowMajor,
    float,                                                // ElementAccumulator (FP32 required)
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm100,
    /* CollectiveBuilder with block-scaled mainloop */,
    /* EpilogueBuilder with scale-multiply fusion */
>;
```

Scale tensors are passed as additional `Arguments::operand_scales`.

For NVFP4, swap `cutlass::float_e4m3_t` -> `cutlass::float_e2m1_t` and use the e8m0 scale dtype.

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

- **NVFP4 is sm_100a-only.** The 5th-generation tensor core's NVFP4 block-scaled MMA does not have a sm_120 equivalent. NVFP4 problems cannot be compile-checked on the 5060 at the same instruction level; either build for sm_100a only (skipping local stage) or fall back to FP8 / BF16 for the local-smoke build.
- **FP8 block-wise is partially portable.** sm_120 has FP8 tensor-core support but lacks TCGen05 / TMEM-staged accumulation. The CUTLASS collective builder picks `KernelMultistage` on sm_120 (Hopper-style two-stage pipeline) instead of the cooperative warp-specialized TCGen05 schedule on sm_100a.
- **TMEM allocations are sm_100a-only.** The accumulator residence in TMEM (used by `tcgen05.mma`) is one of the largest wall-time wins on B200; sm_120 doesn't see this benefit.

Evidence-label conventions per `CLAUDE.md` live in `baselines/Quant/<slug>/README.md`, NOT in this pattern card.

## Common pitfalls

- **FP32 accumulator is non-negotiable.** Quant-category problems explicitly forbid FP16 / BF16 accumulation as a precision downgrade (CLAUDE.md Rule 8).
- **Scale-layout mismatch is the #1 correctness bug.** SOL-ExecBench specs are explicit about block dimensions; copy them exactly.
- **Don't use `Sm89_QMMA_*` atoms on sm_100a builds** -- they work but leave significant TCGen05 wins on the table.

## Linked problems

- `.claude/knowledge/problems/problem_004_fp8_moe_expert_linear.md` -- Quant, FP8 MoE expert linear projection.

Additional Quant dossiers (FP8 attention QKV / output projections, NVFP4 variants) link back to this pattern via the matching-pattern field of their dossiers.
