# problem_031_repeat_kv_attention_matmul

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | fp32-inner attention Q@K^T computation with GQA-style key view (n_rep=1) and a derived scaling constant. Inputs are bf16; reference upcasts to fp32 before the matmul, multiplies by a derived `scaling = head_dim ** -0.5` constant, then downcasts back to bf16. `num_attention_heads = 32`, `num_key_value_heads = 32`, `head_dim = 128`, `num_key_value_groups = 1` pinned. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_031_repeat_kv_attention_matmul.md`     |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/031_repeat_kv_attention_matmul/`                           |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Compile-checked-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                             |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

Pure ATen/cuBLAS dispatch — NO custom CUDA kernels. The reference
algorithm is a GQA key-view + fp32-inner Q@K^T matmul +
derived-constant scaling + bf16 downcast:

    head_dim = 128
    num_key_value_groups = 1
    scaling = head_dim ** -0.5  # derived constant, NOT an input
    key_expanded = key[:, :, None, :, :].expand(
        B, num_key_value_heads, num_key_value_groups, S, head_dim
    )
    key_states = key_expanded.reshape(
        B, num_attention_heads, S, head_dim
    )
    attn_weights = torch.matmul(
        query.to(fp32),
        key_states.transpose(2, 3).to(fp32),
    )
    attn_weights = attn_weights * scaling
    return attn_weights.to(bf16)

For 4D fp32 inputs on Sm80, PyTorch's `at::matmul` for the
transposed-view operand pattern produced by
`key_states.transpose(2, 3).to(fp32)` dispatches the legacy
**`cublasSgemmStridedBatched`** (non-Ex fp32 API), NOT
`cublasGemmStridedBatchedEx`. The transposed-view operand
is not materialized; PyTorch forwards the physical
`[B, H, S, head_dim]` fp32 storage to cuBLAS with
`transA=T, transB=N`. This is the **same already-verified
dispatch surface** as `problem_049_attention_qk_matmul_with_
gqa_repeat_and_scaling`, and is distinct from:

- regular 2D fp32 `cublasSgemm` (prior cohort cells: problem_037,
  022, 004, 054).
- 4D bf16 `cublasGemmStridedBatchedEx` (problem_083, 032).

Implementation in `main.cpp`:

1. **GQA key view** via
   `key.unsqueeze(2).expand({B, kNumKvHeads, kNumKvGroups, S,
   kHeadDim}).reshape({B, kNumAttentionHeads, S, kHeadDim})`.
   At `n_rep = num_key_value_groups = 1`, this is a **no-op
   view** — the singleton expand axis flattens trivially through
   reshape with no broadcast or data copy. Same view-chain ops
   as `problem_049_attention_qk_matmul_with_gqa_repeat_and_scaling`,
   but with a unit repetition factor instead of 4.
2. **bf16 → fp32 upcasts** for `query` and the **transposed**
   `key_states`. The `.transpose(2, 3)` is a stride-only op; the
   subsequent `.to(fp32)` materializes a fp32 tensor that
   preserves the transposed-view stride pattern (physical
   `[B, H, S, head_dim]` fp32 contiguous, logical view
   `[B, H, head_dim, S]`).
3. **fp32 batched matmul** via `at::matmul(q_fp32,
   key_states_fp32.transpose(2, 3))`. Dispatches
   **`cublasSgemmStridedBatched`** (legacy non-Ex fp32 API)
   with `transA=T` (key operand transposed via stride), `transB=N`
   (query operand normal), `m=n=S`, `k=head_dim`, `lda=ldb=head_dim`,
   `ldc=S`, per-batch strides `S*head_dim`, `S*head_dim`, `S*S`.
4. **Derived-constant scaling multiply** by
   `scaling = 1.0 / std::sqrt(double(kHeadDim))` (computed
   in-kernel from the const `head_dim`). NOT an input scalar.
5. **Downcast to bf16** and write into the pre-allocated DPS
   output via `.copy_(...to(bf16))`.

### Why fp32 inner compute (not bf16)

The reference upcasts inputs to fp32 before the matmul, then
downcasts after. A bf16 matmul would produce a different
intermediate precision than the reference's fp32-inner chain and
risk violating the workload's correctness tolerance. The deployed
baseline matches the reference's dtype contract exactly.

### Why `scaling` is derived in-kernel (not an input)

The reference body computes `scaling = head_dim ** -0.5` from
the const `head_dim = 128` inside `run()`. The binding signature
therefore takes only `(query, key, attn_weights)` — no scalar
input. This differs from `problem_049_attention_qk_matmul_with_
gqa_repeat_and_scaling`, where the reference receives `scaling`
as an external Python-float input bound to `double scaling` in
the C++ entry. Here, `scaling` is a `constexpr`-derived
in-kernel constant, computed once from `kHeadDim = 128` at the
top of `run()`.

### Allocations

All workspaces (`key_states`, `q_fp32`, `k_fp32_t`,
`attn_weights_fp32`, the downcast staging tensor) are allocated
implicitly by ATen. The DPS output (`attn_weights`) is
pre-allocated by the eval driver and written in-place via
`.copy_(...)`.

A dummy `kernel.cu` (with just `#include <cuda_runtime.h>`) is
included to force `torch.utils.cpp_extension.load` into its CUDA
build path.

### Runtime-check audit

Every const axis from `definition.json` enforced on every input
and output tensor that the kernel indexes with that fixed
assumption:

| Tensor          | Runtime check                                                                                  |
|-----------------|------------------------------------------------------------------------------------------------|
| `query`         | `is_cuda`, `kBFloat16`, `is_contiguous`, `dim == 4`, `size(1) == num_attention_heads = 32`, `size(3) == head_dim = 128` |
| `key`           | same dtype/contig, `dim == 4`, `size(0) == B`, `size(1) == num_key_value_heads = 32`, `size(2) == S` (cross-checked against query.size(2)), `size(3) == head_dim = 128` |
| `attn_weights`  | same dtype/contig, `dim == 4`, `size(0) == B`, `size(1) == num_attention_heads = 32`, `size(2) == S`, `size(3) == S` (self-attention QK^T output contract) |

The derived equality `num_key_value_groups = num_attention_heads
/ num_key_value_heads` (`1 = 32 / 32`) is enforced implicitly via
the per-tensor const checks above.

Destination-passing-style entry: the DPS output is pre-allocated
and written exactly once via `.copy_(...)`.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was
compile-checked and natively-correctness-validated on. Final SOL
scoring is on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5090):
  the fp32 batched-strided-GEMM operates per-batch on tiles
  `[S, 128] × [128, S]` with `batch_count = B*32 = 64` at the
  largest canonical workload (workload 15: `B=2, S=4096`). The
  per-batch query tile is `4096 * 128 * 4 = 2 MiB fp32`,
  within the 3090 Ti's 6 MiB L2 partition. cuBLAS may select
  a different strided-batched kernel on B200.
- DSMEM and cluster-cooperative tile staging (sm_100a-only) are
  out of scope at this Sm80-portability level.
- TCGen05 / TMEM divergence: the fp32 batched-GEMM runs on
  Ampere's tf32-mma fp32 SM pipe on the 3090 Ti; B200's TCGen05
  / TMEM fp32 path is NOT exercised at this level.
- L2 carveout / partition: workload-dependent. The 16 canonical
  workloads span B*H*S of roughly `1*32*128 = 4,096` to
  `2*32*4096 = 262,144`, with workload 15 (`B=2, S=4096`)
  carrying the largest single-batch matmul work.
- The deployed pure-ATen/cuBLAS path means the cohort-B probe
  (`cublasSgemmStridedBatched` fp32 at the equivalent shape)
  measures the same algorithm-selection path the deployed code
  uses — probe runtime-contract fidelity per BL-20260515 is
  preserved at the kernel-selection layer.

### Calibration relevance

This is the **second fp32 4D batched-strided** data point in the
ledger and the second row whose dispatch is
`cublasSgemmStridedBatched` (the legacy non-Ex fp32 API; the
first is `problem_049_attention_qk_matmul_with_gqa_repeat_and_
scaling`). Compared with problem_049, this cell contributes:

- a **higher batch_count** (64 vs 4) per dispatch — exercising
  cuBLAS's per-batch tile-scheduling at a much larger grid.
- a **narrower contraction dimension** (`K = head_dim = 128` vs
  256) — the inner-product accumulation runs at half the depth.
- a **no-op GQA view** (n_rep=1 vs n_rep=4) — the stride-0
  expand axis present in problem_049 is absent here.
- a **derived-constant scaling** (in-kernel from `kHeadDim` vs
  external Python-float input) — distinguishes the binding
  signature surface from problem_049's.

The same dispatch family is exercised — the cohort-B probe is
the SAME `cublasSgemmStridedBatched(OP_T, OP_N, ...)` API
call as problem_049's probe, with workload-specific
`(batch_count, M, K, N) = (64, 4096, 128, 4096)`.
