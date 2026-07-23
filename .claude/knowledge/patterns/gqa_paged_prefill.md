---
Use when: implementing the prefill-phase grouped-query attention with paged KV cache — multi-query causal attention over a long context where KV is split into pages. Distinct from decode (`gqa_paged_decode.md`): prefill is multi-query (full sequence) with a causal mask. Covers head_dim ∈ {64, 128, 192, 256}, BF16 / FP16 / FP8 inputs. Targets B200 sm_100a with sm_120 fallback.
Primary raw source: sol-execbench/examples/cutlass/
Supporting sources: references/NVIDIA_Manuals/cutlass_cute/; references/NVIDIA_Manuals/blackwell_tuning/; cutlass/include/cute/algorithm/copy.hpp
Repo touchpoints: cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl; cutlass/include/cute/atom/copy_atom.hpp; baselines/FIB/<problem_slug>/cutlass/
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
Version family: cuda-13.1; cutlass-4.5.x; sol-execbench-2026; flashinfer-2026
Default skill: implementing-sol-baselines
Routing tags: gqa; paged-attention; paged_attention; prefill; causal; ragged; flashinfer; kv-cache; bf16; fp16; fp8
---

# Gqa Paged Prefill

## Use when

Implementing the prefill-phase grouped-query attention with paged kv cache — multi-query causal attention over a long context where kv is split into pages. distinct from decode (`gqa_paged_decode.md`): prefill is multi-query (full sequence) with a causal mask. covers head_dim ∈ {64, 128, 192, 256}, bf16 / fp16 / fp8 inputs. targets b200 sm_100a with sm_120 fallback.

## Canonical recipe

Prefill is a multi-query Flash-style attention over a paged KV cache with causal masking:

```
for each (batch, q_tile) in seq_q × num_q_heads:
  for each page in page_table[batch]:
    QK^T(q_tile, K_page)  // apply causal mask if page is within q_tile's past
    online-softmax update
    AV(P_tile, V_page)
```
Ragged-batch variants concatenate sequences of different lengths into a single packed tensor with a `cu_seqlens` table.

## CUTLASS / CuTe / CUDA C++ building blocks

- `cutlass::TiledCopy` with paged TMA descriptors (sm_100a) for the K/V page gather.
- `cutlass::TiledMma` for QK^T and AV; online-softmax in shared memory.
- A causal-mask predicate evaluated per (q_idx, kv_idx) pair before softmax.
- `cu_seqlens` for ragged-batch packing.

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

`sm_120` (local RTX 5060, GB206 die) compiles and smoke-runs; `sm_100a` (B200, GB100 die) is the scoring target. Local pass on sm_120 verifies compile + correctness only; sm_100a is required for TCGen05 / TMA / DSMEM behavior, TMEM accumulation, and HBM3e bandwidth assumptions. Build twice when in doubt; the B200 problem packager auto-injects `-gencode arch=compute_100a,code=sm_100a`.

Evidence-label conventions per `CLAUDE.md` live in `baselines/<category>/<slug>/README.md`, NOT in this pattern card.

## Common pitfalls

- **Mask off-by-one** — verify the causal predicate against spec.
- **Paged gather on non-default stream** (Rule 7).
- **`data_ptr` caching of page_table** (Rule 6).

## Linked problems

See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `gqa_paged_prefill` rows.
