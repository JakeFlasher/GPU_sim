# problem_054_audio_attention_qkv_projection_with_normalization

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | fp32 fused QKV projection with per-head RMS normalization on Q and K (V unchanged) for audio attention. Three Linear projections (Q, K, V) without bias, three reshapes to `(B, S, num_heads, head_dim)`, then ATen-only RMSNorm chain on Q and K. `hidden_size = 1024`, `num_heads = 8`, `head_dim = 128`, `qkv_out_size = 1024` pinned. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_054_audio_attention_qkv_projection_with_normalization.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/054_audio_attention_qkv_projection_with_normalization/`    |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

Pure ATen/cuBLAS dispatch — NO custom CUDA kernels. The reference
algorithm is three fp32 linear projections (Q, K, V) without bias,
followed by reshape and per-head RMSNorm on Q and K only:

    query = torch.matmul(hidden_states, q_proj_weight.t())
    key   = torch.matmul(hidden_states, k_proj_weight.t())
    value = torch.matmul(hidden_states, v_proj_weight.t())
    query = query.view(B, S, num_heads, head_dim)
    key   = key.view(B, S, num_heads, head_dim)
    value = value.view(B, S, num_heads, head_dim)
    q_var = query.pow(2).mean(dim=-1, keepdim=True)
    q_normed = query / sqrt(q_var + eps)
    query_states = q_normed * q_norm_weight
    # (same for key with k_norm_weight)
    value_states = value  # NO normalization on V

For fp32 inputs on Sm80, PyTorch's `at::matmul(x, w.t())`
dispatches `cublasSgemm` via the cuBLAS handle. The reference's
`torch.matmul` path selects the same kernel, so the deployed
baseline's three `at::matmul` calls route through the same cuBLAS
algorithm-selection path the reference uses.

Implementation in `main.cpp`:

1. **Three `at::matmul(x, w.t())` dispatches** —
   - **Q**: `M=B*S, K=hidden_size=1024, N=qkv_out_size=1024` fp32.
   - **K**: same shape.
   - **V**: same shape.
2. **Three `.view(B, S, num_heads, head_dim)`** reshapes.
3. **Per-head RMSNorm for Q** via ATen-only chain
   (`query_view.pow(2).mean(dim=-1, keepdim=True) →
   .add(eps_scalar).sqrt() → query_view / ... → * q_norm_weight`).
4. **Per-head RMSNorm for K** via the same chain.
5. **V is unchanged** — `value_states.copy_(value_view)`.

`eps` is extracted via `.item<float>()` once on the host and
broadcast as a scalar add into the ATen chain.

### Why pure ATen/cuBLAS dispatch (not a custom RMSNorm kernel)

The deployed baseline matches the reference's exact dtype
contract (fp32 in, fp32 out, fp32 cuBLAS sgemm, ATen elementwise
chain for RMSNorm). A custom RMSNorm CUDA kernel would risk
cross-CTA reduction-tree drift or fp32 accumulation-order
differences against the reference's per-row pow/mean/sqrt chain —
the same failure mode that bit
`problem_010_attention_value_projection_with_transpose` and
`problem_039_adaptive_instance_normalization_2d_backward` retry
attempts in earlier batches (both still queued as side issues
for future cuBLAS-linked deployed-path replacement). The per-head
reduction over `head_dim = 128` is small enough that ATen's
row-wise elementwise + reduction chain is bandwidth-appropriate.

### Allocations

All workspaces (`query`, `key`, `value` plus the implicit `view`,
`pow`, `mean`, `add`, `sqrt`, `div`, `mul` intermediates) are
allocated implicitly by ATen. The three DPS outputs
(`query_states`, `key_states`, `value_states`) are pre-allocated
by the eval driver and written in-place via `.copy_(...)`.

A dummy `kernel.cu` (with just `#include <cuda_runtime.h>`) is
included to force `torch.utils.cpp_extension.load` into its CUDA
build path. Same workaround as
`problem_065_fused_statistics_projection_and_split`,
`problem_004_attention_output_projection_with_reshape_backward`,
`problem_050_attention_qkv_projection_with_bias_and_reshape`, and
`problem_055_attention_qkv_projection_with_gqa_reshape`.

### Runtime-check audit

Every const axis from `definition.json` enforced on every input
and output tensor that the kernel indexes with that fixed
assumption:

| Tensor          | Runtime check                                                                                  |
|-----------------|------------------------------------------------------------------------------------------------|
| `hidden_states` | `is_cuda`, `kFloat32`, `is_contiguous`, `dim == 3`, `size(2) == hidden_size = 1024`            |
| `q_proj_weight` | same dtype/contig, `dim == 2`, `size(0) == qkv_out_size = 1024`, `size(1) == hidden_size = 1024` |
| `k_proj_weight` | same dtype/contig, `dim == 2`, `size(0) == qkv_out_size = 1024`, `size(1) == hidden_size = 1024` |
| `v_proj_weight` | same dtype/contig, `dim == 2`, `size(0) == qkv_out_size = 1024`, `size(1) == hidden_size = 1024` |
| `q_norm_weight` | same dtype/contig, `dim == 1`, `size(0) == head_dim = 128`                                     |
| `k_norm_weight` | same dtype/contig, `dim == 1`, `size(0) == head_dim = 128`                                     |
| `eps`           | same dtype/contig, `numel() == 1`                                                              |
| `query_states`  | same dtype/contig, `dim == 4`, `size(0) == B`, `size(1) == S`, `size(2) == 8`, `size(3) == 128` |
| `key_states`    | same dtype/contig, `dim == 4`, `size(0) == B`, `size(1) == S`, `size(2) == 8`, `size(3) == 128` |
| `value_states`  | same dtype/contig, `dim == 4`, `size(0) == B`, `size(1) == S`, `size(2) == 8`, `size(3) == 128` |

The derived equality `qkv_out_size = num_heads * head_dim`
(`1024 = 8 * 128`) is enforced implicitly via the per-tensor
const checks above.

Destination-passing-style entry: all three DPS outputs are
pre-allocated. Each is written exactly once via `.copy_(...)`.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was
compile-checked and natively-correctness-validated on. Final SOL
scoring is on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5060):
  the three fp32 GEMMs operate on `[qkv_out_size=1024,
  hidden_size=1024]` weight tiles (4 MiB fp32 each, 12 MiB
  total) that fit twice over within the 3090 Ti's 6 MiB L2
  partition. cuBLAS may select a different fp32 sgemm kernel on
  B200 (TCGen05 / TMEM fp32 path vs Ampere's tf32-mma fp32 SM
  pipe).
- DSMEM and cluster-cooperative tile staging (sm_100a-only)
  are out of scope at this Sm80-portability level.
- TCGen05 / TMEM divergence: the three fp32 sgemm calls run on
  Ampere's tf32-mma fp32 SM pipe; B200's TCGen05 / TMEM fp32
  path is NOT exercised at this level.
- L2 carveout / partition: total weight footprint ~12 MiB fp32
  + activation footprint scales with B*S. The working set
  straddles the 3090 Ti's L2 boundary across the canonical
  workload sweep (B*S range 131..8192).
- The deployed pure-ATen/cuBLAS path means the cohort-B probe
  (cublasSgemm fp32 at the equivalent Q-projection shape)
  measures the same algorithm-selection path the deployed code
  uses — probe runtime-contract fidelity per BL-20260515 is
  preserved at the kernel-selection layer.
