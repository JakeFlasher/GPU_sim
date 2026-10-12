# problem_049_attention_qk_matmul_with_gqa_repeat_and_scaling

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | fp32-inner attention Q@K^T computation with GQA key repetition and scaling. Inputs are bf16; reference upcasts to fp32 before the matmul, multiplies by scalar `scaling`, then downcasts back to bf16. `num_attention_heads = 4`, `num_key_value_heads = 1`, `head_dim = 256`, `num_key_value_groups = 4` pinned. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_049_attention_qk_matmul_with_gqa_repeat_and_scaling.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/049_attention_qk_matmul_with_gqa_repeat_and_scaling/`    |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

Pure ATen/cuBLAS dispatch — NO custom CUDA kernels. The reference
algorithm is a GQA key-expansion + fp32-inner Q@K^T matmul +
scalar scaling + bf16 downcast:

    n_rep = num_attention_heads // num_key_value_heads  # 4
    key_expanded = key[:, :, None, :, :].expand(
        B, num_key_value_heads, n_rep, S, head_dim
    )
    key_states = key_expanded.reshape(
        B, num_attention_heads, S, head_dim
    )
    attn_weights = torch.matmul(
        query.to(fp32),
        key_states.transpose(2, 3).to(fp32),
    )
    attn_weights = attn_weights * scaling
    attn_scores = attn_weights.to(bf16)

For 4D fp32 inputs on Sm80, PyTorch's `at::matmul` for the
transposed-view operand pattern produced by
`key_states.transpose(2, 3).to(fp32)` dispatches the legacy
**`cublasSgemmStridedBatched`** (non-Ex fp32 API), NOT
`cublasGemmStridedBatchedEx`. The transposed-view operand
is not materialized; PyTorch forwards the physical
`[B, H, S, head_dim]` fp32 storage to cuBLAS with
`transA=T, transB=N`. This is distinct from:

- regular 2D fp32 `cublasSgemm` (prior cohort cells: problem_037,
  022, 004, 054).
- 4D bf16 `cublasGemmStridedBatchedEx` (problem_083, 032).

This is the **third dispatch surface** in the cohort-B probe
family.

Implementation in `main.cpp`:

1. **GQA key repetition** via
   `key.unsqueeze(2).expand({B, kNumKvHeads, kNumKvGroups, S,
   kHeadDim}).reshape({B, kNumAttentionHeads, S, kHeadDim})`.
   This is a **no-data-copy view chain** — the expand creates
   stride-0 axes for the n_rep=4 broadcast, and the reshape
   collapses the singleton + stride-0 axes without materializing
   a contiguous copy. The resulting `key_states` tensor has
   stride-0 axes preserved.
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
4. **Scalar multiply by `scaling`** via `* scaling` (binding
   signature is `double scaling`).
5. **Downcast to bf16** and write into the pre-allocated DPS
   output via `.copy_(...to(bf16))`.

### Why fp32 inner compute (not bf16)

The reference upcasts inputs to fp32 before the matmul, then
downcasts after. A bf16 matmul would produce a different
intermediate precision than the reference's fp32-inner chain and
risk violating the workload's correctness tolerance. The deployed
baseline matches the reference's dtype contract exactly.

### Why `double scaling` (not `torch::Tensor scaling`)

`scaling` is a Python float scalar (definition.json declares
`shape=None, dtype=float32`). Same canonical-scalar binding
pattern as `problem_054`'s `eps`: pybind11 routes Python `float`
to `double`, not to `torch::Tensor`. The same binding signature
used for `problem_054_audio_attention_qkv_projection_with_normalization`
is pre-applied here.

### Allocations

All workspaces (`key_states`, `q_fp32`, `k_fp32_t`,
`attn_weights`, the downcast staging tensor) are allocated
implicitly by ATen. The DPS output (`attn_scores`) is
pre-allocated by the eval driver and written in-place via
`.copy_(...)`.

A dummy `kernel.cu` (with just `#include <cuda_runtime.h>`) is
included to force `torch.utils.cpp_extension.load` into its CUDA
build path.

### Runtime-check audit

Every const axis from `definition.json` enforced on every input
and output tensor that the kernel indexes with that fixed
assumption:

| Tensor        | Runtime check                                                                                  |
|---------------|------------------------------------------------------------------------------------------------|
| `query`       | `is_cuda`, `kBFloat16`, `is_contiguous`, `dim == 4`, `size(1) == num_attention_heads = 4`, `size(3) == head_dim = 256` |
| `key`         | same dtype/contig, `dim == 4`, `size(0) == B`, `size(1) == num_key_value_heads = 1`, `size(2) == S` (cross-checked against query.size(2)), `size(3) == head_dim = 256` |
| `attn_scores` | same dtype/contig, `dim == 4`, `size(0) == B`, `size(1) == num_attention_heads = 4`, `size(2) == S`, `size(3) == S` (self-attention QK^T output contract) |

The derived equality `num_key_value_groups = num_attention_heads
/ num_key_value_heads` (`4 = 4 / 1`) is enforced implicitly via
the per-tensor const checks above.

Destination-passing-style entry: the DPS output is pre-allocated
and written exactly once via `.copy_(...)`.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was
compile-checked and natively-correctness-validated on. Final SOL
scoring is on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5090):
  the fp32 batched-strided-GEMM operates per-batch on tiles
  `[S, 256] × [256, S]` with `batch_count = B*4 = 4` at the
  largest canonical workload (workload 4: `B=1, S=8192`). The
  per-batch query tile is `8192 * 256 * 4 = 8 MiB fp32`,
  exceeding the 3090 Ti's 6 MiB L2 partition. cuBLAS may select
  a different strided-batched kernel on B200.
- DSMEM and cluster-cooperative tile staging (sm_100a-only) are
  out of scope at this Sm80-portability level.
- TCGen05 / TMEM divergence: the fp32 batched-GEMM runs on
  Ampere's tf32-mma fp32 SM pipe on the 3090 Ti; B200's TCGen05
  / TMEM fp32 path is NOT exercised at this level.
- L2 carveout / partition: workload-dependent. The 16 canonical
  workloads span B*H*S of roughly `1*4*128 = 512` to
  `1*4*8192 = 32,768`, with workload 4 (`B=1, S=8192`) carrying
  the largest single-batch matmul work.
- The deployed pure-ATen/cuBLAS path means the cohort-B probe
  (`cublasGemmStridedBatchedEx` fp32 at the equivalent shape)
  measures the same algorithm-selection path the deployed code
  uses — probe runtime-contract fidelity per BL-20260515 is
  preserved at the kernel-selection layer.

### Calibration relevance

This is the **first fp32 4D batched-strided** data point in the
ledger AND the first row in the ledger whose dispatch is
`cublasSgemmStridedBatched` (the legacy non-Ex fp32 API). Prior
fp32 cohort cells used regular 2D `cublasSgemm` (L1≈0%, L2≈82-94%
across five landed cells), and prior 4D batched-strided cells
were bf16 `cublasGemmStridedBatchedEx` (`problem_083` L1≈85%
L2≈63% at N=head_dim=64; `problem_032` L1≈0% L2≈71% at
N=head_dim=128). problem_049 contributes a new cohort point at
the intersection of fp32-dtype + batched-strided-API +
wide-output-dim shape (`K = head_dim` plays the contraction
role; `M = N = S` is the wide output dim) + `transA=T` operand
layout from the transposed view.
