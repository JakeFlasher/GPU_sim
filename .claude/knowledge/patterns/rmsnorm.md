---
Use when: implementing pure RMSNorm without a fused residual add — `y = (x / rms(x)) * weight`. For the residual-add-fused variant see `fused_add_rmsnorm.md`. Covers BF16 / FP16 inputs with FP32 reduction, hidden-size families 128 to 7168. Targets B200 sm_100a with sm_120 fallback.
Primary raw source: sol-execbench/examples/cuda_cpp/rmsnorm/kernel.cu
Supporting sources: references/NVIDIA_Manuals/cuda_guide/; references/NVIDIA_Manuals/blackwell_tuning/
Repo touchpoints: sol-execbench/examples/cuda_cpp/rmsnorm/kernel.cu; baselines/L1/<problem_slug>/cuda_cpp/; baselines/FIB/<problem_slug>/cuda_cpp/
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
Version family: cuda-13.1; sol-execbench-2026
Default skill: implementing-sol-baselines
Routing tags: rmsnorm; root-mean-square; rms; pure-rmsnorm; norm; reduction; bf16; fp16
---

# Rmsnorm

## Use when

Implementing pure rmsnorm without a fused residual add — `y = (x / rms(x)) * weight`. for the residual-add-fused variant see `fused_add_rmsnorm.md`. covers bf16 / fp16 inputs with fp32 reduction, hidden-size families 128 to 7168. targets b200 sm_100a with sm_120 fallback.

## Canonical recipe

```
rms  = sqrt(mean(x^2) + eps)              // FP32 reduction
y    = (x / rms) * weight                 // affine scale only, no bias
```
Single-pass: accumulate `sum_sq` in FP32 in one CTA per row; reciprocal-sqrt via `__frsqrt_rn`; multiply-broadcast across the row. Wide rows (h > 4096) benefit from cluster-cooperative reduction on sm_100a.

## CUTLASS / CuTe / CUDA C++ building blocks

- A `__global__` row-reduction kernel; FP32 accumulator; `__frsqrt_rn` for the reciprocal-sqrt.
- `cute::TiledCopy` for the weight broadcast on sm_100a.
- Cluster-cooperative reduce for wide-h rows (sm_100a only).

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

`sm_120` (local RTX 5060, GB206 die) compiles and smoke-runs; `sm_100a` (B200, GB100 die) is the scoring target. Local pass on sm_120 verifies compile + correctness only; sm_100a is required for TCGen05 / TMA / DSMEM behavior, TMEM accumulation, and HBM3e bandwidth assumptions. Build twice when in doubt; the B200 problem packager auto-injects `-gencode arch=compute_100a,code=sm_100a`.

Evidence-label conventions per `CLAUDE.md` live in `baselines/<category>/<slug>/README.md`, NOT in this pattern card.

## Common pitfalls

- **Mean-of-squares in BF16** loses bits; FP32 reduction is contractual (Rule 8).
- **Forgetting `+ eps`** when `x` is exactly zero.
- **`data_ptr` caching of `weight`** (Rule 6).

## Linked problems

See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `rmsnorm` rows.
