# problem_055_attention_qkv_projection_with_gqa_reshape

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | bf16 fused QKV projection with Grouped Query Attention (GQA) reshape and transpose. Three Linear projections (Q, K, V) without bias, then `view + transpose(1, 2) + contiguous` to attention layout `(B, heads, S, head_dim)`. `hidden_size = 2048`, `num_heads = 16`, `num_kv_heads = 4`, `head_dim = 128`, `q_size = 2048`, `kv_size = 512` pinned. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_055_attention_qkv_projection_with_gqa_reshape.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/055_attention_qkv_projection_with_gqa_reshape/`            |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

Pure ATen/cuBLAS dispatch — NO custom CUDA kernels. The reference
algorithm is three bf16 linear projections (Q, K, V) without bias,
followed by view + transpose to attention layout:

    query_states = torch.matmul(hidden_states, q_weight.t())
    key_states   = torch.matmul(hidden_states, k_weight.t())
    value_states = torch.matmul(hidden_states, v_weight.t())
    query_states = query_states.view(B, S, num_heads, head_dim).transpose(1, 2)
    key_states   = key_states.view(B, S, num_kv_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(B, S, num_kv_heads, head_dim).transpose(1, 2)

For bf16 inputs on Sm80, PyTorch's `at::matmul(x, w.t())`
dispatches `cublasGemmEx` (`CUDA_R_16BF` inputs / output,
`CUBLAS_COMPUTE_32F` accumulator) via the cuBLAS handle. The
reference's `torch.matmul` path selects the same kernel, so the
deployed baseline's three `at::matmul` calls route through the
same cuBLAS algorithm-selection path the reference uses.

Implementation in `main.cpp`:

1. **Three `at::matmul(x, w.t())` dispatches** —
   - **Q**: `M=B*S, K=hidden_size=2048, N=q_size=2048` bf16.
   - **K**: `M=B*S, K=hidden_size=2048, N=kv_size=512` bf16.
   - **V**: `M=B*S, K=hidden_size=2048, N=kv_size=512` bf16.
2. **Three `.view(B, S, heads, head_dim).transpose(1, 2).contiguous()`**
   — Q reshapes to `(B, S, 16, 128)` then transposes to
   `(B, 16, S, 128)`; K/V reshape to `(B, S, 4, 128)` then
   transpose to `(B, 4, S, 128)`. The `.contiguous()` call
   materializes the transposed layout into a row-major bf16
   buffer suitable for the DPS write.
3. **Three `.copy_(...)`** writes into the pre-allocated DPS
   outputs.

### Why bf16 in/out (not fp32 upcast)

The reference does NOT upcast — it runs `torch.matmul` on bf16
tensors directly. Sm80 bf16 GEMM dispatches `cublasGemmEx` with
`CUDA_R_16BF` inputs / output and a `CUBLAS_COMPUTE_32F`
accumulator. The deployed `at::matmul` chain matches this dtype
contract exactly. A bf16 → fp32 upcast wrapper would route
through fp32 sgemm instead — a different kernel with a different
cache footprint — and would not match the reference's bf16
cuBLAS path.

### Why `at::matmul(x, w.t())` instead of `at::linear(x, w)`

Both expressions dispatch the same cuBLAS GEMM kernel on Sm80
for bf16 inputs. `at::linear(x, w)` computes `x @ w.t()` plus an
optional bias; without bias, it folds to the same `matmul(x,
w.t())` dispatch. The deployed code mirrors the reference's
exact textual pattern (`torch.matmul(x, w.t())`) for one-to-one
correspondence with the reference; the choice is prose-level.

### Allocations

All workspaces (`q_out`, `k_out`, `v_out` and the implicit
view + transpose intermediates) are allocated implicitly by
ATen. The three DPS outputs (`query_states`, `key_states`,
`value_states`) are pre-allocated by the eval driver and
written in-place via `.copy_(...)`.

A dummy `kernel.cu` (with just `#include <cuda_runtime.h>`) is
included to force `torch.utils.cpp_extension.load` into its
CUDA build path, since `main.cpp`'s ATen headers transitively
require `cuda_runtime_api.h`. Same workaround as
`problem_065_fused_statistics_projection_and_split`,
`problem_004_attention_output_projection_with_reshape_backward`,
and `problem_050_attention_qkv_projection_with_bias_and_reshape`.

### Runtime-check audit

Every const axis from `definition.json` enforced on every input
and output tensor that the kernel indexes with that fixed
assumption:

| Tensor          | Runtime check                                                                                  |
|-----------------|------------------------------------------------------------------------------------------------|
| `hidden_states` | `is_cuda`, `kBFloat16`, `is_contiguous`, `dim == 3`, `size(2) == hidden_size = 2048`           |
| `q_weight`      | same dtype/contig, `dim == 2`, `size(0) == q_size = 2048`, `size(1) == hidden_size = 2048`     |
| `k_weight`      | same dtype/contig, `dim == 2`, `size(0) == kv_size = 512`, `size(1) == hidden_size = 2048`     |
| `v_weight`      | same dtype/contig, `dim == 2`, `size(0) == kv_size = 512`, `size(1) == hidden_size = 2048`     |
| `query_states`  | same dtype/contig, `dim == 4`, `size(0) == B`, `size(1) == 16`, `size(2) == S`, `size(3) == 128`|
| `key_states`    | same dtype/contig, `dim == 4`, `size(0) == B`, `size(1) == 4`,  `size(2) == S`, `size(3) == 128`|
| `value_states`  | same dtype/contig, `dim == 4`, `size(0) == B`, `size(1) == 4`,  `size(2) == S`, `size(3) == 128`|

The derived equalities `q_size = num_heads * head_dim`
(`2048 = 16 * 128`) and
`kv_size = num_kv_heads * head_dim` (`512 = 4 * 128`) are
enforced implicitly via the per-tensor const checks above.

Destination-passing-style entry: all three DPS outputs are
pre-allocated. Each is written exactly once via
`.copy_(...view(...).transpose(1, 2).contiguous())`.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was
compile-checked and natively-correctness-validated on. Final SOL
scoring is on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5060):
  the three bf16 GEMMs operate on `[q_size, hidden_size]` and
  `[kv_size, hidden_size]` weight tiles (Q is 2048×2048 = ~8
  MiB bf16; K/V are 512×2048 = ~2 MiB bf16 each). The total
  weight footprint ~12 MiB bf16 exceeds the 3090 Ti's 6 MiB L2
  partition; B200's larger L2 will capture more weight reuse.
  cuBLAS may also select a different bf16 GEMM kernel on B200
  (TCGen05 / TMEM bf16 path vs Ampere's bf16 m16n8k16
  TensorOp path).
- DSMEM and cluster-cooperative tile staging (sm_100a-only)
  are out of scope at this Sm80-portability level.
- TCGen05 / TMEM divergence: the three bf16 GEMM calls run on
  Ampere's bf16 m16n8k16 TensorOp pipe on the 3090 Ti; B200's
  TCGen05 / TMEM bf16 path is not exercised at this level.
- L2 carveout / partition: workload-dependent. With weight
  footprint ~12 MiB bf16 and activation footprint scaling with
  B*S, the working set straddles the 3090 Ti's L2 capacity
  boundary across the canonical workload sweep.
- The deployed pure-ATen/cuBLAS path means the cohort-B probe
  (cublasGemmEx bf16 at the equivalent Q-projection shape)
  measures the same algorithm-selection path the deployed code
  uses — probe runtime-contract fidelity per BL-20260515 is
  preserved at the kernel-selection layer.
