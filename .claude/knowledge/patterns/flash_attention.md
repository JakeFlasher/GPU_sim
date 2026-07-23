---
Use when: implementing the canonical Flash Attention forward — fused online-softmax tiled attention without paging. Covers BF16 / FP16 with FP32 reduction, head_dim ∈ {64, 128, 192, 256}. Distinct from `attention_core` in that `flash_attention` specifically denotes the Flash-2 / Flash-3 algorithm with TMA-staged producer warps. Targets B200 sm_100a with sm_120 fallback.
Primary raw source: cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl
Supporting sources: references/NVIDIA_Manuals/cutlass_cute/; references/NVIDIA_Manuals/blackwell_tuning/; cutlass/include/cute/algorithm/copy.hpp
Repo touchpoints: cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl; baselines/L1/<problem_slug>/cutlass/; baselines/FIB/<problem_slug>/cutlass/
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
Routing tags: flash-attention; flash; flash-2; flash-3; online-softmax; attention; bf16; fp16
---

# Flash Attention

## Use when

Implementing the canonical flash attention forward — fused online-softmax tiled attention without paging. covers bf16 / fp16 with fp32 reduction, head_dim ∈ {64, 128, 192, 256}. distinct from `attention_core` in that `flash_attention` specifically denotes the flash-2 / flash-3 algorithm with tma-staged producer warps. targets b200 sm_100a with sm_120 fallback.

## Canonical recipe

Flash-3 forward (sm_100a):

```
for each (q_tile, kv_tile):
  S = Q_tile @ K_tile^T
  m_new = max(m, max(S))         // running max
  P = exp(S - m_new)
  l_new = exp(m - m_new) * l + sum(P)
  O = exp(m - m_new) * O + P @ V_tile
  m, l = m_new, l_new
O /= l                           // final normalize
```
On sm_100a the producer warps stage Q/K/V tiles via TMA into shared memory + TMEM; the consumer warps run `tcgen05.mma` for QK^T and AV with TMEM accumulation. On sm_120 the `KernelMultistage` schedule runs in regular registers.

## CUTLASS / CuTe / CUDA C++ building blocks

- Two `cutlass::gemm` collectives back-to-back (QK^T then AV) sharing the warp-specialized schedule.
- Producer / consumer pipeline via `cute::PipelineTmaAsync` on sm_100a.
- Online-softmax state (m, l, O) held in registers + TMEM.

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

`sm_120` (local RTX 5060, GB206 die) compiles and smoke-runs; `sm_100a` (B200, GB100 die) is the scoring target. Local pass on sm_120 verifies compile + correctness only; sm_100a is required for TCGen05 / TMA / DSMEM behavior, TMEM accumulation, and HBM3e bandwidth assumptions. Build twice when in doubt; the B200 problem packager auto-injects `-gencode arch=compute_100a,code=sm_100a`.

Evidence-label conventions per `CLAUDE.md` live in `baselines/<category>/<slug>/README.md`, NOT in this pattern card.

## Common pitfalls

- **Numerical underflow** in `exp(S - m_new)` when m is large; the `S - m` trick is contractual.
- **head_dim splitting** — head_dim > 128 requires tile splitting; respect cluster limits.
- **Multi-stream split of forward / backward** (Rule 7).

## Linked problems

See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `flash_attention` rows.
