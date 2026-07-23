---
Use when: implementing the Mixture-of-Experts (MoE) dispatch + grouped-GEMM pattern. Covers token → expert routing (top-k gating), permutation / sort, grouped expert GEMMs via `cutlass::gemm::group_array_problem_shape`, and inverse permutation. Targets B200 sm_100a with sm_120 fallback. Single-pass dispatch + grouped matmul shape is the canonical SOL-ExecBench L2 MoE-layer baseline; the FP8-quant variant composes with `fp8_blockwise_quant`.
Primary raw source: cutlass/include/cutlass/gemm/group_array_problem_shape.hpp
Supporting sources: references/NVIDIA_Manuals/cutlass_cute/; references/NVIDIA_Manuals/blackwell_tuning/; cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl
Repo touchpoints: cutlass/include/cutlass/gemm/group_array_problem_shape.hpp; cutlass/include/cutlass/gemm/kernel/sm100_tile_scheduler_group.hpp; baselines/L2/<problem_slug>/cutlass/; baselines/Quant/<problem_slug>/cutlass/
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
Routing tags: moe; mixture-of-experts; dispatch; grouped-gemm; group-gemm; routing; top-k; gate; permutation; bf16; fp8
---

# MoE Dispatch + Grouped GEMM

## Use when

You're implementing a Mixture-of-Experts decoder layer baseline. The input is `tokens [B*T, hidden]` plus `gate_logits [B*T, num_experts]`; the kernel must:

1. Pick top-k experts per token (typically k=2 or k=4).
2. Compute routing weights via softmax over the top-k.
3. Permute tokens so that all tokens routed to expert E are contiguous (the `permutation` step).
4. Run one expert GEMM per group (the grouped-GEMM step).
5. Apply the routing weights.
6. Inverse-permute to restore the original token order.

Optionally a shared expert path runs in parallel and is summed at the end (see `problem_080_moe_complete_layer_with_shared_expert_backward.md` for the canonical backward instance).

## Canonical decomposition

```
gate_logits  ─► top_k_select  ─►  expert_idx [B*T, k], expert_weight [B*T, k]
                                              │
                                              ▼
              sort / permute  ─►  permuted_tokens, permuted_indices, group_sizes
                                              │
                                              ▼
        cutlass::gemm::GroupedGemm<...>  ─►  permuted_expert_out
                                              │
                                              ▼
            inverse_permute + weighted_sum  ─►  output [B*T, hidden]
```

## CUTLASS / CuTe building blocks

- **`cutlass::gemm::group_array_problem_shape`** — declares the per-expert GEMM problem shapes packed into device-side arrays (M_i, N_i, K_i, ptr_A_i, ptr_B_i, ptr_C_i).
- **`cutlass::gemm::kernel::sm100_tile_scheduler_group`** — sm_100a tile scheduler that assigns CTAs to expert groups; uses TCGen05 mma where available.
- **`cute::TiledCopy`** with TMA for the permuted-token gather; one descriptor per group.
- A **per-token routing kernel** (CUDA C++) for the top-k + permutation step. Two valid shapes: (a) one fused kernel computing top-k → sort → permute, (b) separate top-k → segmented-sort → scatter kernels. Shape (b) is easier to get right; shape (a) is the perf-tuning target.

Skeleton (CUDA C++ surface, simplified):

```cpp
extern "C" __global__ void moe_top_k_route_kernel(
    const float* __restrict__ gate_logits,    // [B*T, E]
    int*        __restrict__ expert_idx,      // [B*T, k]
    float*      __restrict__ expert_weight,   // [B*T, k] (post-softmax)
    int*        __restrict__ group_counts,    // [E]
    int B_times_T, int E, int k)
{ /* top-k + softmax per row */ }

// Then host-side prefix-sum on group_counts → group_offsets, allocate
// permuted_tokens / permuted_indices, launch the gather kernel, then
// launch the cutlass grouped GEMM.
```

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

`sm_120` (local RTX 5060, GB206 die) compiles and smoke-runs; `sm_100a` (B200, GB100 die) is the scoring target. Concrete divergences:

- **Grouped-GEMM tile scheduler.** `sm100_tile_scheduler_group` uses cluster-aware work assignment with DSMEM peer-read on sm_100a; sm_120 falls back to a per-CTA scheduler with smaller cluster shapes and lower utilization on small-M groups.
- **TCGen05 / TMEM.** Each expert GEMM is a small-M matmul (typical M_i ranges from 32 to a few thousand depending on routing); on sm_100a the TMEM accumulator + `tcgen05.mma` schedule keeps the compute pipeline full. On sm_120 the regular-register accumulator path costs register pressure and lowers occupancy.
- **HBM3e vs GDDR7 bandwidth.** The permutation step is largely memory-bound (gather-scatter on `tokens`); B200's HBM3e absorbs the irregular access pattern much better than the 5060's GDDR7. Local timing on sm_120 will over-estimate the permutation cost relative to the GEMM cost.
- **Atomic add coalescing.** The inverse-permute weighted-sum uses `atomicAdd` on the output when multiple top-k experts contribute to the same token row. sm_100a has improved atomic coalescing relative to sm_120; the local kernel may need a different aggregation strategy (e.g. lock-free segmented reduce).

Evidence-label conventions per `CLAUDE.md` live in `baselines/<category>/<slug>/README.md`, NOT in this pattern card.

## Common pitfalls

- **Load imbalance across experts.** Real MoE gating produces highly skewed group sizes; the grouped-GEMM tile scheduler can starve large groups if the per-CTA cluster mapping is naive. Use the sm_100a cluster-aware scheduler.
- **`data_ptr` caching the per-expert weight bases** (CLAUDE.md Rule 6). The harness shifts inputs by 256 B every iteration; the per-expert `ptr_B_i` array recomputes every call. Caching the expert-weight base addresses fails.
- **Multi-stream expert dispatch** (CLAUDE.md Rule 7). Tempting to launch one stream per expert to overlap; the harness's stream-injection detector rejects this. The grouped-GEMM is the *single-stream* fusion of that idea.
- **Precision downgrade in the routing softmax** (CLAUDE.md Rule 8). Compute `gate_logits → softmax` in FP32; downgrading to BF16 to share an MMA pipeline drifts the routing decisions and silently changes correctness on the harness's randomized trials.
- **Wrong inverse-permute order.** When k > 1 each token's outputs are accumulated weighted; the inverse permutation must apply weights from `expert_weight` per (token, k_index), not per token only. A single-weight pass is a correctness bug, not a perf bug.

## Linked problems

- `.claude/knowledge/problems/problem_080_moe_complete_layer_with_shared_expert_backward.md` — full L2 MoE decoder layer (this card supplies the dispatch + grouped-GEMM building block).
- `.claude/knowledge/problems/problem_004_fp8_moe_expert_linear.md` — Quant-category single-expert FP8 linear (composes with `fp8_blockwise_quant`).
- Composes with `.claude/knowledge/patterns/gemm_baseline.md` (each expert GEMM is a tile of the canonical GEMM recipe) and `.claude/knowledge/patterns/fp8_blockwise_quant.md` (FP8 MoE variants).
