# problem_050_attention_qkv_projection_with_bias_and_reshape

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | bf16 Gemma3 fused QKV projection with bias and reshape. Three Linear projections (Q, K, V) with bias, then reshape from `(B, S, hidden_size)` to `(B, S, num_heads, head_dim)`. `hidden_size = 640`, `num_attention_heads = 4`, `num_key_value_heads = 1`, `head_dim = 256`, `q_out_dim = 1024`, `kv_out_dim = 256` pinned. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_050_attention_qkv_projection_with_bias_and_reshape.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/050_attention_qkv_projection_with_bias_and_reshape/`       |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

Pure ATen/cuBLAS dispatch — NO custom CUDA kernels. The reference
algorithm is three bf16 linear projections (Q, K, V) with bias,
followed by three reshapes:

    query_states = (hidden_states @ q_weight.t()) + q_bias
    key_states   = (hidden_states @ k_weight.t()) + k_bias
    value_states = (hidden_states @ v_weight.t()) + v_bias
    query_states = query_states.view(B, S, num_attention_heads, head_dim)
    key_states   = key_states.view(B, S, num_key_value_heads, head_dim)
    value_states = value_states.view(B, S, num_key_value_heads, head_dim)

For bf16 inputs on Sm80, PyTorch's `at::linear(x, w, b)`
dispatches `cublasGemmEx` (`CUDA_R_16BF` inputs/output,
`CUBLAS_COMPUTE_32F` accumulator) via the cuBLAS handle. The
reference's `torch.matmul + bias` path selects the same kernel,
so the deployed baseline's three `at::linear` calls route through
the same cuBLAS algorithm-selection path the reference uses.

Implementation in `main.cpp`:

1. **Three `at::linear(x, w, bias)` dispatches** — the
   composition of `x @ w.t() + bias` exactly matches the
   reference's `(hidden_states @ q_weight.t()) + q_bias`
   expression for each of Q / K / V.
   - **Q**: `M=B*S, K=hidden_size=640, N=q_out_dim=1024` bf16.
   - **K**: `M=B*S, K=hidden_size=640, N=kv_out_dim=256` bf16.
   - **V**: `M=B*S, K=hidden_size=640, N=kv_out_dim=256` bf16.
2. **Three `.reshape({B, S, heads, head_dim})`** — Q reshapes
   to `(B, S, 4, 256)`; K/V reshape to `(B, S, 1, 256)`.
3. **Three `.copy_(...)`** writes into the pre-allocated DPS
   outputs.

### Why bf16 in/out (not fp32 upcast)

The reference does NOT upcast — it runs `torch.matmul` on bf16
tensors directly. Sm80 bf16 GEMM dispatches `cublasGemmEx` with
`CUDA_R_16BF` inputs / output and a `CUBLAS_COMPUTE_32F`
accumulator. The deployed `at::linear` chain matches this dtype
contract exactly. A bf16 → fp32 upcast wrapper would route through
fp32 sgemm instead — a different kernel with a different cache
footprint — and would not match the reference's bf16 cuBLAS path.

### Allocations

All workspaces (`q_out`, `k_out`, `v_out` and the implicit
reshape-view tensors) are allocated implicitly by ATen. The three
DPS outputs (`query_states`, `key_states`, `value_states`) are
pre-allocated by the eval driver and written in-place via
`.copy_(...)`.

A dummy `kernel.cu` (with just `#include <cuda_runtime.h>`) is
included to force `torch.utils.cpp_extension.load` into its CUDA
build path, since `main.cpp`'s ATen headers transitively require
`cuda_runtime_api.h`. Same workaround as
`problem_065_fused_statistics_projection_and_split` and
`problem_004_attention_output_projection_with_reshape_backward`.

### Runtime-check audit

Every const axis from `definition.json` enforced on every input
and output tensor that the kernel indexes with that fixed
assumption:

| Tensor          | Runtime check                                                                                  |
|-----------------|------------------------------------------------------------------------------------------------|
| `hidden_states` | `is_cuda`, `kBFloat16`, `is_contiguous`, `dim == 3`, `size(2) == hidden_size = 640`            |
| `q_weight`      | same dtype/contig, `dim == 2`, `size(0) == q_out_dim = 1024`, `size(1) == hidden_size = 640`   |
| `q_bias`        | same dtype/contig, `dim == 1`, `size(0) == q_out_dim = 1024`                                   |
| `k_weight`      | same dtype/contig, `dim == 2`, `size(0) == kv_out_dim = 256`, `size(1) == hidden_size = 640`   |
| `k_bias`        | same dtype/contig, `dim == 1`, `size(0) == kv_out_dim = 256`                                   |
| `v_weight`      | same dtype/contig, `dim == 2`, `size(0) == kv_out_dim = 256`, `size(1) == hidden_size = 640`   |
| `v_bias`        | same dtype/contig, `dim == 1`, `size(0) == kv_out_dim = 256`                                   |
| `query_states`  | same dtype/contig, `dim == 4`, `size(0) == B`, `size(1) == S`, `size(2) == 4`, `size(3) == 256`|
| `key_states`    | same dtype/contig, `dim == 4`, `size(0) == B`, `size(1) == S`, `size(2) == 1`, `size(3) == 256`|
| `value_states`  | same dtype/contig, `dim == 4`, `size(0) == B`, `size(1) == S`, `size(2) == 1`, `size(3) == 256`|

The derived equalities `q_out_dim = num_attention_heads * head_dim`
(`1024 = 4 * 256`) and
`kv_out_dim = num_key_value_heads * head_dim` (`256 = 1 * 256`)
are enforced implicitly via the per-tensor const checks above.

Destination-passing-style entry: all three DPS outputs are
pre-allocated. Each is written exactly once via
`.copy_(...reshape(...))`.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was
compile-checked and natively-correctness-validated on. Final SOL
scoring is on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5060):
  the three bf16 GEMMs operate on small `[hidden_size,
  hidden_size]` weight tiles (Q is 640×1024 = ~1.3 MiB bf16;
  K/V are 640×256 = ~320 KiB bf16 each) that fit easily within
  the 3090 Ti's 6 MiB L2 partition and trivially in B200's
  larger L2. cuBLAS may select a different bf16 GEMM kernel on
  B200 (TCGen05 / TMEM bf16 path vs Ampere's bf16 m16n8k16
  TensorOp path).
- DSMEM and cluster-cooperative tile staging (sm_100a-only)
  are out of scope at this Sm80-portability level.
- TCGen05 / TMEM divergence: the three bf16 GEMM calls run on
  Ampere's bf16 m16n8k16 TensorOp pipe on the 3090 Ti; B200's
  TCGen05 / TMEM bf16 path is not exercised at this level.
- L2 carveout / partition: small workload-dependent. With
  total weight footprint ~2 MiB bf16 and total activation
  footprint scaling with B*S, the entire working set fits
  comfortably in L2 across all 16 canonical workloads.
- The deployed pure-ATen/cuBLAS path means the cohort-B probe
  (cublasGemmEx bf16 at the equivalent Q shape) measures the
  same algorithm-selection path the deployed code uses —
  probe runtime-contract fidelity per BL-20260515 is preserved
  at the kernel-selection layer.
