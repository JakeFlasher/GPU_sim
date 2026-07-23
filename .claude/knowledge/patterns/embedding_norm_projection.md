---
Use when: implementing token-embedding lookup + initial layer-norm + optional projection — the first kernel chain in many transformer pipelines. Covers BF16 / FP16 embedding tables, FP32 normalization, optional linear projection back to model dim. Targets B200 sm_100a with sm_120 fallback.
Primary raw source: cutlass/include/cute/algorithm/copy.hpp
Supporting sources: references/NVIDIA_Manuals/cuda_guide/; references/NVIDIA_Manuals/cutlass_cute/
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
Routing tags: embedding; token-embedding; initial-norm; layernorm; rmsnorm; projection; lookup; bf16; fp16
---

# Embedding Norm Projection

## Use when

Implementing token-embedding lookup + initial layer-norm + optional projection — the first kernel chain in many transformer pipelines. covers bf16 / fp16 embedding tables, fp32 normalization, optional linear projection back to model dim. targets b200 sm_100a with sm_120 fallback.

## Canonical recipe

```
e = embedding_table[token_ids]              // gather: [B, T, hidden]
n = LayerNorm(e)  or  RMSNorm(e)
y = n @ W_proj      (optional)
```
Two valid shapes: (a) three-call (gather + norm + GEMM); (b) fused gather-norm in one CUDA C++ kernel, then a separate GEMM. (b) saves one HBM round-trip on `e`.

## CUTLASS / CuTe / CUDA C++ building blocks

- A `__global__` gather kernel; `cute::TiledCopy` for the embedding-table fetch.
- Norm reduction in FP32 inside the same kernel (shape b).
- `cutlass::gemm` for the projection step.

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

`sm_120` (local RTX 5060, GB206 die) compiles and smoke-runs; `sm_100a` (B200, GB100 die) is the scoring target. Local pass on sm_120 verifies compile + correctness only; sm_100a is required for TCGen05 / TMA / DSMEM behavior, TMEM accumulation, and HBM3e bandwidth assumptions. Build twice when in doubt; the B200 problem packager auto-injects `-gencode arch=compute_100a,code=sm_100a`.

Evidence-label conventions per `CLAUDE.md` live in `baselines/<category>/<slug>/README.md`, NOT in this pattern card.

## Common pitfalls

- **Out-of-vocabulary token_ids** — bound-check or assume sanitized; spec defines.
- **`data_ptr` on the embedding table** (Rule 6).
- **Norm precision** (Rule 8) — FP32 reduction.

## Linked problems

See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `embedding_norm_projection` rows.
