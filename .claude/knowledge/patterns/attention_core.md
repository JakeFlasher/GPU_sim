---
Use when: implementing the forward attention core `softmax(QK^T / sqrt(d)) @ V` without RoPE / paged-KV / dropout fusions. Covers BF16 / FP16 inputs with FP32 softmax reduction, dense (not paged) KV, optional additive masking, head_dim ∈ {64, 128, 192, 256}. Targets B200 sm_100a with sm_120 fallback.
Primary raw source: cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl
Supporting sources: references/NVIDIA_Manuals/cutlass_cute/; references/NVIDIA_Manuals/blackwell_tuning/; cutlass/include/cute/algorithm/copy.hpp
Repo touchpoints: cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl; cutlass/include/cute/atom/mma_atom.hpp; baselines/L1/<problem_slug>/cutlass/; baselines/L2/<problem_slug>/cutlass/
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
Routing tags: attention; attention-core; softmax; qk-matmul; av-matmul; dense-attention; bf16; fp16
---

# Attention Core

## Use when

Implementing the forward attention core `softmax(qk^t / sqrt(d)) @ v` without rope / paged-kv / dropout fusions. covers bf16 / fp16 inputs with fp32 softmax reduction, dense (not paged) kv, optional additive masking, head_dim ∈ {64, 128, 192, 256}. targets b200 sm_100a with sm_120 fallback.

## Canonical recipe

```
S = (Q @ K^T) * scale         // [seq_q, seq_kv]
P = softmax_rowwise(S)        // FP32 reduction; cast back to input dtype
O = P @ V                     // [seq_q, head_dim]
```
Streaming (Flash-style) variants compute `P` and `O` in tiles, keeping a running max / sum / output to avoid materialising the full `[seq_q, seq_kv]` matrix. For B200, the tile size is bounded by shared memory + TMEM; for sm_120 the `KernelMultistage` schedule uses regular registers.

## CUTLASS / CuTe / CUDA C++ building blocks

- Two `cutlass::gemm` collectives back-to-back (`QK^T` then `AV`).
- A producer warp that stages `K, V` tiles via TMA on sm_100a.
- Online-softmax accumulators (running max, running sum, running output) in FP32.
- `cutlass::epilogue::fusion` callbacks for the trailing scale + dtype cast.

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

`sm_120` (local RTX 5060, GB206 die) compiles and smoke-runs; `sm_100a` (B200, GB100 die) is the scoring target. Local pass on sm_120 verifies compile + correctness only; sm_100a is required for TCGen05 / TMA / DSMEM behavior, TMEM accumulation, and HBM3e bandwidth assumptions. Build twice when in doubt; the B200 problem packager auto-injects `-gencode arch=compute_100a,code=sm_100a`.

Evidence-label conventions per `CLAUDE.md` live in `baselines/<category>/<slug>/README.md`, NOT in this pattern card.

## Common pitfalls

- **Softmax in BF16** loses precision around `S ≈ 0`; keep the reduction FP32 (Rule 8).
- **`data_ptr` on K/V** caching breaks under the harness's 256-byte input shift (Rule 6).
- **Multi-stream QK^T / AV** to overlap is a Rule-7 violation.

## Linked problems

See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `attention_core` rows.
