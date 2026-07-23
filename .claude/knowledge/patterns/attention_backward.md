---
Use when: implementing the backward pass for transformer attention — the Q/K/V matmul gradients, softmax/dropout backward, and value-output matmul backward — used inside training-loop fused kernels. Covers BF16 / FP16 / FP32 inputs with FP32 reduction, with or without GQA / dropout / scaling. Targets B200 sm_100a with sm_120 fallback.
Primary raw source: cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl
Supporting sources: references/NVIDIA_Manuals/cutlass_cute/; references/NVIDIA_Manuals/blackwell_tuning/; cutlass/include/cute/algorithm/copy.hpp
Repo touchpoints: cutlass/include/cutlass/gemm/device/gemm_universal.h; cutlass/include/cute/atom/mma_atom.hpp; baselines/L1/<problem_slug>/cutlass/; baselines/L2/<problem_slug>/cutlass/
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
Version family: cutlass-4.5.x; sol-execbench-2026
Default skill: implementing-sol-baselines
Routing tags: attention; attention-backward; backward; softmax-backward; qkv-backward; matmul-backward; bf16; fp16
---

# Attention Backward

## Use when

Implementing the backward pass for transformer attention — the q/k/v matmul gradients, softmax/dropout backward, and value-output matmul backward — used inside training-loop fused kernels. covers bf16 / fp16 / fp32 inputs with fp32 reduction, with or without gqa / dropout / scaling. targets b200 sm_100a with sm_120 fallback.

## Canonical recipe

Attention backward decomposes into three GEMM-shaped gradients plus the softmax / dropout backprop:

```
dV = (P^T)        @ dO        // [seq, head_dim]
dP = dO            @ V^T       // [seq, seq]; followed by softmax-backward
dQ = (dS · scale)  @ K         // [seq, head_dim]
dK = (dS · scale)^T @ Q        // [seq, head_dim]
```
where `S = QK^T`, `P = softmax(S)`, `dS = dropout_backward(softmax_backward(dP, P))`. Each gradient maps onto `cutlass::gemm::device::GemmUniversal` with the appropriate transpose; the softmax-backward is a streaming reduce-update kernel that reads `P` (or recomputes it) and produces `dS`.

## CUTLASS / CuTe / CUDA C++ building blocks

- `cutlass::gemm::device::GemmUniversal` with sm_100a collective builder for `dV`, `dQ`, `dK`.
- `cutlass::epilogue::collective::sm100_epilogue_builder` to fuse the `scale` factor and any residual add into the gradient output.
- A custom `__global__` softmax-backward kernel that walks `P` row-wise; accumulate in FP32.
- `cutlass::TiledMma` selection matches the forward kernel's MMA shape so the backward executes on the same `tcgen05.mma` schedule on sm_100a.

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

`sm_120` (local RTX 5060, GB206 die) compiles and smoke-runs; `sm_100a` (B200, GB100 die) is the scoring target. Local pass on sm_120 verifies compile + correctness only; sm_100a is required for TCGen05 / TMA / DSMEM behavior, TMEM accumulation, and HBM3e bandwidth assumptions. Build twice when in doubt; the B200 problem packager auto-injects `-gencode arch=compute_100a,code=sm_100a`.

Evidence-label conventions per `CLAUDE.md` live in `baselines/<category>/<slug>/README.md`, NOT in this pattern card.

## Common pitfalls

- **Precision downgrade is a Rule-8 reward-hack** — keep softmax-backward FP32 even when forward is BF16/FP16.
- **`data_ptr` caching of `P`** (Rule 6) — the harness shifts inputs every iteration; recompute or revalidate, do not memoise.
- **Default-stream discipline** (Rule 7) — never split the four gradient launches across user streams.

## Linked problems

See `tools/sol_baselines_scaffold/sol_pattern_mapping.md`; this pattern's primary_pattern row in that table enumerates the dossiers that link here.
