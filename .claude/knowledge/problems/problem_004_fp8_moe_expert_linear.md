---
Use when: starting the Quant-category baseline for `problem_004_fp8_moe_expert_linear` -- FP8 e4m3 block-scaled linear projection inside a MoE expert. Captured from a quantized production MoE model. Uses block-wise FP8 scaling (1x128 typical); FP32 accumulator required.
Primary raw source: references/NVIDIA_Manuals/solbench_problems/problem_004_fp8_moe_expert_linear.md
Supporting sources: .claude/knowledge/patterns/fp8_blockwise_quant.md; .claude/knowledge/patterns/gemm_baseline.md; .claude/knowledge/sources/blackwell-tuning-docs.md
Repo touchpoints: baselines/Quant/problem_004_fp8_moe_expert_linear/cutlass/; cutlass/include/cutlass/gemm/collective/builders/sm100_blockscaled_umma_builder.inl; cutlass/include/cute/atom/mma_traits_sm100.hpp
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
Version family: sol-execbench-2026
Default skill: implementing-sol-baselines
Routing tags: fp8; quant; moe; problem_004; e4m3; block-scale
---

# problem_004_fp8_moe_expert_linear

## Use when

Implementing the baseline for an FP8 e4m3 block-scaled linear projection inside a MoE expert. This is a Quant-category problem -- the matmul is FP8 with block-wise scales, and the output is BF16. Captured from a quantized production MoE model.

## Workload summary

| Field | Value |
|---|---|
| Operation | `C = (A * scale_A) @ (B * scale_B).T` (FP8 GEMM with block-wise scales, BF16 output) |
| Dtypes | A, B: FP8 e4m3; scale_A, scale_B: per-block scales; C: BF16; accumulator: FP32 per CLAUDE.md Rule 8 |
| Scale block size | BlockWise1x128 typical (one scale per 128-element block along K) |
| Category | Quant (33 problems split: 18 FP8, 15 NVFP4) |

## Matching patterns

This problem instantiates `.claude/knowledge/patterns/fp8_blockwise_quant.md`. The pattern card carries the CUTLASS block-scaled MMA recipe and the sm_100a / sm_120 fallback policy; this dossier records the problem-specific shape and scale layout.

The underlying GEMM is a variant of `.claude/knowledge/patterns/gemm_baseline.md` -- same CollectiveMainloop + EpilogueVisitor structure, but with FP8 MMA atoms and a block-scale fusion in the epilogue.

## Implementation entry point

Author under `baselines/Quant/problem_004_fp8_moe_expert_linear/cutlass/{main.cpp, kernel.cu, kernel.h, solution.json}`.

Entry point per `sol-execbench/docs/solution.md`:

```cpp
void run(torch::Tensor A, torch::Tensor B,
         torch::Tensor scale_A, torch::Tensor scale_B,
         torch::Tensor C) {
    // FP8 GEMM with block-wise scales; output into C as BF16
}
```

(See the canonical spec for exact tensor names and shape conventions.)

## Critical correctness points

- **FP32 accumulator is non-negotiable** (CLAUDE.md Rule 8). Using FP16 / BF16 accumulators for FP8 inputs is a precision-contract violation.
- **Scale layout must match exactly.** The block dimensions are specified in the source spec; mismatches are the #1 correctness bug for this problem family.
- **Epilogue must apply scales before the BF16 downcast.** Otherwise the dynamic-range loss compounds.

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

`sm_120` (RTX 5060) and `sm_100a` (B200) divergences:

- **TCGen05 / TMEM accumulation is sm_100a-only.** The single largest wall-time win for this problem is unavailable on sm_120; the collective builder picks the Hopper-style `KernelMultistage` schedule on sm_120.
- **FP8 tensor-core MMA atoms** are present on both arches but with different opcode families; CUTLASS handles the selection through the collective builder's `cutlass::arch::Sm100` vs the sm_120 path.
- **Scale-tensor streaming bandwidth.** Scales are small per block but read frequently; B200's HBM3e + larger L2 absorb them more easily than the 5060's GDDR7.

Evidence-label vocabulary per `CLAUDE.md` lives in `baselines/Quant/problem_004_fp8_moe_expert_linear/README.md`, NOT in this dossier.

## Cross-references

- Pattern recipe: `.claude/knowledge/patterns/fp8_blockwise_quant.md`.
- Source spec: `references/NVIDIA_Manuals/solbench_problems/problem_004_fp8_moe_expert_linear.md`.
- Blackwell-specific tuning notes: `.claude/knowledge/sources/blackwell-tuning-docs.md`.
