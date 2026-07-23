---
Use when: implementing LayerNorm / GroupNorm kernels (not RMSNorm — see `rmsnorm.md` or `fused_add_rmsnorm.md` for that family). Covers BF16 / FP16 / FP32 inputs with FP32 reduction, optional affine weight + bias, with or without fused residual / activation. Targets B200 sm_100a with sm_120 fallback.
Primary raw source: cutlass/include/cute/algorithm/copy.hpp
Supporting sources: references/NVIDIA_Manuals/cuda_guide/; references/NVIDIA_Manuals/blackwell_tuning/; references/NVIDIA_Manuals/cutlass_cute/
Repo touchpoints: cutlass/include/cute/atom/copy_atom.hpp; baselines/L1/<problem_slug>/cuda_cpp/; baselines/L2/<problem_slug>/cuda_cpp/
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
Version family: cuda-13.1; cutlass-4.5.x; sol-execbench-2026
Default skill: implementing-sol-baselines
Routing tags: layernorm; groupnorm; norm; normalization; affine; welford; reduction; bf16; fp16
---

# Norm Layernorm Groupnorm

## Use when

Implementing layernorm / groupnorm kernels (not rmsnorm — see `rmsnorm.md` or `fused_add_rmsnorm.md` for that family). covers bf16 / fp16 / fp32 inputs with fp32 reduction, optional affine weight + bias, with or without fused residual / activation. targets b200 sm_100a with sm_120 fallback.

## Canonical recipe

Two-pass numerically-stable Welford variance + standard affine:

```
(mean, var) = Welford(x_row)                  // FP32 reduction
x_hat = (x_row - mean) / sqrt(var + eps)
y     = x_hat * weight + bias                 // affine, optional
```
GroupNorm partitions channels into groups and runs the same reduction per (batch, group). For B200, cluster-cooperative reduction across CTAs via DSMEM is available for very wide normalization axes.

## CUTLASS / CuTe / CUDA C++ building blocks

- A `__global__` two-pass reduction kernel (or Welford one-pass) in CUDA C++; FP32 accumulators.
- `cute::TiledCopy` for the affine-parameter load on sm_100a.
- Cluster-cooperative reduction primitives (sm_100a only) for very wide axes.

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

`sm_120` (local RTX 5060, GB206 die) compiles and smoke-runs; `sm_100a` (B200, GB100 die) is the scoring target. Local pass on sm_120 verifies compile + correctness only; sm_100a is required for TCGen05 / TMA / DSMEM behavior, TMEM accumulation, and HBM3e bandwidth assumptions. Build twice when in doubt; the B200 problem packager auto-injects `-gencode arch=compute_100a,code=sm_100a`.

Evidence-label conventions per `CLAUDE.md` live in `baselines/<category>/<slug>/README.md`, NOT in this pattern card.

## Common pitfalls

- **Variance in BF16** loses precision (catastrophic cancellation); FP32 reduction is contractual (Rule 8).
- **Caching the mean / var** across iterations via `data_ptr` (Rule 6) — the harness shifts inputs.
- **`atomicAdd` for cross-CTA reduce** without proper coalescing burns L2 bandwidth.

## Linked problems

See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `norm_layernorm_groupnorm` rows.
