# problem_083_attention_score_value_matmul

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | bf16 attention score-value matmul + transpose + reshape (the second major matmul in the attention pattern, after QK^T + softmax). Compute `attention_weights @ value` as a batched 4D matmul, then transpose `(B, H, Q, D) → (B, Q, H, D)` and reshape to `(B, Q, H*D)` for the downstream output projection. `num_heads = 20`, `head_dim = 64`, `hidden_size = 1280` pinned. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_083_attention_score_value_matmul.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/083_attention_score_value_matmul/`                       |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

Pure ATen/cuBLAS dispatch — NO custom CUDA kernels. The reference
algorithm is a batched 4D matmul followed by transpose +
contiguous + reshape:

    attn = torch.matmul(attention_weights, value)
    # [B, H, Q, K] @ [B, H, K, D] -> [B, H, Q, D]
    attn = attn.transpose(1, 2).contiguous()
    # -> [B, Q, H, D]
    output = attn.reshape(B, Q, hidden_size)
    # -> [B, Q, H*D=1280]

For bf16 inputs on Sm80, PyTorch's `at::matmul` for 4D inputs
dispatches **`cublasGemmStridedBatchedEx`** — the batched analog
of the regular `cublasGemmEx` used for 2D matmul. The reference's
`torch.matmul` path selects the same batched kernel, so the
deployed baseline routes through the same cuBLAS algorithm-
selection path the reference uses.

Implementation in `main.cpp`:

1. **One `at::matmul(attention_weights, value)` dispatch** for
   the 4D bf16 inputs. cuBLAS strided-batched-bf16-GEMM with
   `batch_count = B * num_heads`, `M = Q`, `K = K`,
   `N = head_dim = 64`.
2. **`.transpose(1, 2).contiguous()`** materializes the row-major
   `[B, Q, H, D]` bf16 buffer.
3. **`.reshape({B, Q, kHiddenSize})`** reshapes to `[B, Q, 1280]`.
4. **`.copy_()`** into the pre-allocated DPS output.

### Why a 4D batched matmul (not 2D folded)

PyTorch's `at::matmul` on 4D inputs explicitly groups the leading
two dimensions as batch dims, dispatching the strided-batched
cuBLAS variant. Folding to 2D via `view + matmul + view` would
dispatch the regular `cublasGemmEx` over `B*H*Q × K` and `K × D`
— a different kernel with different occupancy and tile shapes.
The reference uses the natural 4D `torch.matmul` form, so the
deployed baseline mirrors it exactly to preserve dispatch
fidelity.

### Allocations

All workspaces (`attn`, the transposed contiguous tensor, and the
reshape view) are allocated implicitly by ATen. The DPS output
(`output`) is pre-allocated by the eval driver and written
in-place via `.copy_(...)`.

A dummy `kernel.cu` (with just `#include <cuda_runtime.h>`) is
included to force `torch.utils.cpp_extension.load` into its CUDA
build path. Same workaround as
`problem_065_fused_statistics_projection_and_split`,
`problem_004_attention_output_projection_with_reshape_backward`,
`problem_050_attention_qkv_projection_with_bias_and_reshape`,
`problem_055_attention_qkv_projection_with_gqa_reshape`, and
`problem_054_audio_attention_qkv_projection_with_normalization`.

### Runtime-check audit

Every const axis from `definition.json` enforced on every input
and output tensor that the kernel indexes with that fixed
assumption:

| Tensor              | Runtime check                                                                                  |
|---------------------|------------------------------------------------------------------------------------------------|
| `attention_weights` | `is_cuda`, `kBFloat16`, `is_contiguous`, `dim == 4`, `size(1) == num_heads = 20`              |
| `value`             | same dtype/contig, `dim == 4`, `size(0) == B`, `size(1) == num_heads = 20`, `size(2) == K` (cross-checked against attention_weights.size(3)), `size(3) == head_dim = 64` |
| `output`            | same dtype/contig, `dim == 3`, `size(0) == B`, `size(1) == Q`, `size(2) == hidden_size = 1280` |

The derived equality `hidden_size = num_heads * head_dim`
(`1280 = 20 * 64`) is enforced implicitly via the per-tensor
const checks above. The `K` axis is cross-checked between
`attention_weights.size(3)` and `value.size(2)` to enforce the
matmul shape contract.

Destination-passing-style entry: the DPS output is pre-allocated
and written exactly once via `.copy_(...)`.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was
compile-checked and natively-correctness-validated on. Final SOL
scoring is on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5060):
  the bf16 strided-batched-GEMM operates per-batch on tiles
  `[Q, K]` × `[K, D=64]`. At the largest canonical workload
  (workload 6: `B=2, Sq=Skv=2053`, batch_count=40) the
  per-batch input footprint is ~16 MB bf16 — larger than the
  3090 Ti's 6 MiB L2 partition but spread across batches.
  cuBLAS may select a different strided-batched kernel on B200
  (TCGen05 / TMEM bf16 batched-GEMM path vs Ampere's bf16
  m16n8k16 strided-batched-TensorOp path).
- DSMEM and cluster-cooperative tile staging (sm_100a-only)
  are out of scope at this Sm80-portability level.
- TCGen05 / TMEM divergence: the batched-bf16-GEMM runs on
  Ampere's bf16 strided-batched TensorOp pipe on the 3090 Ti;
  B200's TCGen05 / TMEM batched-bf16 path is NOT exercised at
  this level.
- L2 carveout / partition: workload-dependent. The 16 canonical
  workloads span B*H*Q = 1280..327,680 and span Skv = 64..2053.
  Decode-style workloads (Skv=77) and square-attention
  workloads (Sq=Skv) likely select different cuBLAS kernels
  even on the same arch.
- The deployed pure-ATen/cuBLAS path means the cohort-B probe
  (cublasGemmStridedBatchedEx bf16 at the canonical largest-
  mainloop workload shape) measures the same algorithm-
  selection path the deployed code uses — probe runtime-
  contract fidelity per BL-20260515 is preserved at the
  kernel-selection layer.
