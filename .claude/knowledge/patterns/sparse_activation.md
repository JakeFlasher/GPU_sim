---
Use when: implementing sparse-activation primitives — top-k selection / sigmoid gating with thresholding / sparse-mask construction — as a standalone kernel (not as part of a MoE dispatch). Targets B200 sm_100a with sm_120 fallback.
Primary raw source: references/NVIDIA_Manuals/cuda_guide/00_index.md
Supporting sources: cutlass/include/cute/algorithm/copy.hpp; references/NVIDIA_Manuals/cutlass_cute/
Repo touchpoints: baselines/L1/<problem_slug>/cuda_cpp/
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
Routing tags: sparse; sparse-activation; top-k; thresholded; sigmoid-gate; bf16; fp16; fp32
---

# Sparse Activation

## Use when

Implementing sparse-activation primitives — top-k selection / sigmoid gating with thresholding / sparse-mask construction — as a standalone kernel (not as part of a moe dispatch). targets b200 sm_100a with sm_120 fallback.

## Canonical recipe

```
y[i] = activation(x[i])
y[y < threshold] = 0        // or top-k mask
```
Two access patterns: (1) bitmask emission; (2) compacted-output emission (scatter / scan).

## CUTLASS / CuTe / CUDA C++ building blocks

- A `__global__` thresholding kernel; FP32 internal compute.
- For top-k: a small heap or radix-select in shared memory.
- For compaction: warp-level prefix-sum (`cub::WarpScan`).

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

`sm_120` (local RTX 5060, GB206 die) compiles and smoke-runs; `sm_100a` (B200, GB100 die) is the scoring target. Local pass on sm_120 verifies compile + correctness only; sm_100a is required for TCGen05 / TMA / DSMEM behavior, TMEM accumulation, and HBM3e bandwidth assumptions. Build twice when in doubt; the B200 problem packager auto-injects `-gencode arch=compute_100a,code=sm_100a`.

Evidence-label conventions per `CLAUDE.md` live in `baselines/<category>/<slug>/README.md`, NOT in this pattern card.

## Common pitfalls

- **Compaction without coalescing** burns HBM bandwidth.
- **`data_ptr` on the threshold table** (Rule 6).

## Linked problems

See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `sparse_activation` rows.
