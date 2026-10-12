# problem_032_attention_weights_matmul_with_value_projection

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | bf16 attention weights @ value matmul + transpose + reshape (self-attention variant; Q=K=seq_len). Compute `attn_weights @ value_states` as a batched 4D matmul, then `.transpose(1, 2)` and `.reshape(B, S, H*D)`. `num_attention_heads = 40`, `head_dim = 128`, `hidden_size = 5120` pinned. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_032_attention_weights_matmul_with_value_projection.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/032_attention_weights_matmul_with_value_projection/`     |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

Pure ATen/cuBLAS dispatch — NO custom CUDA kernels. The reference
algorithm is a single batched 4D matmul followed by transpose +
reshape:

    attn_output = torch.matmul(attn_weights, value_states)
    # [B, H, S, S] @ [B, H, S, D] -> [B, H, S, D]
    attn_output = attn_output.transpose(1, 2)
    # -> [B, S, H, D]
    output = attn_output.reshape(B, S, hidden_size)
    # -> [B, S, H*D=5120]

For bf16 inputs on Sm80, PyTorch's `at::matmul` on 4D inputs
dispatches **`cublasGemmStridedBatchedEx`** — the batched analog
of the regular `cublasGemmEx` used for 2D matmul. The reference's
`torch.matmul` selects the same kernel, so the deployed baseline
routes through the same cuBLAS algorithm-selection path.

Implementation in `main.cpp`:

1. **One `at::matmul(attn_weights, value_states)` dispatch** for
   the 4D bf16 inputs. cuBLAS strided-batched-bf16-GEMM with
   `batch_count = B * num_attention_heads`, `M = S`, `K = S`,
   `N = head_dim = 128`.
2. **`.transpose(1, 2).contiguous()`** materializes the row-major
   `[B, S, H, D]` bf16 buffer.
3. **`.reshape({B, S, kHiddenSize})`** reshapes to `[B, S, 5120]`.
4. **`.copy_()`** into the pre-allocated DPS output.

### Self-attention vs cross-attention shape

problem_032 fixes `seq_len_q = seq_len_kv = seq_len` (a single
`S` axis in the canonical workload sweep) — this is the
**self-attention** variant where Q and K share the same sequence
length. Compare `problem_083_attention_score_value_matmul`, which
has **separate** `seq_len_q` and `seq_len_kv` axes (cross-
attention / GQA / chunked-attention variants). The deployed code
is structurally identical between the two slugs — only the
runtime shape parametrization differs.

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
`problem_055_attention_qkv_projection_with_gqa_reshape`,
`problem_054_audio_attention_qkv_projection_with_normalization`,
and `problem_083_attention_score_value_matmul`.

### Runtime-check audit

Every const axis from `definition.json` enforced on every input
and output tensor that the kernel indexes with that fixed
assumption:

| Tensor         | Runtime check                                                                                  |
|----------------|------------------------------------------------------------------------------------------------|
| `attn_weights` | `is_cuda`, `kBFloat16`, `is_contiguous`, `dim == 4`, `size(1) == num_attention_heads = 40`, `size(2) == S`, `size(3) == S` (cross-check: self-attention contract Q=K=S) |
| `value_states` | same dtype/contig, `dim == 4`, `size(0) == B`, `size(1) == num_attention_heads = 40`, `size(2) == S` (cross-checked against attn_weights.size(2)), `size(3) == head_dim = 128` |
| `output`       | same dtype/contig, `dim == 3`, `size(0) == B`, `size(1) == S`, `size(2) == hidden_size = 5120` |

The derived equality `hidden_size = num_attention_heads * head_dim`
(`5120 = 40 * 128`) is enforced implicitly via the per-tensor
const checks above. The `S` axis is cross-checked between
`attn_weights.size(2)`, `attn_weights.size(3)`, and
`value_states.size(2)` to enforce the self-attention matmul shape
contract.

Destination-passing-style entry: the DPS output is pre-allocated
and written exactly once via `.copy_(...)`.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was
compile-checked and natively-correctness-validated on. Final SOL
scoring is on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5090):
  the bf16 strided-batched-GEMM operates per-batch on tiles
  `[S, S] × [S, D=128]`. At the largest canonical workload
  (workload 14: `B=1, S=2048`, batch_count=40) the per-batch
  attn_weights tile is 2048*2048*2 = 8 MiB bf16 — larger than
  the 3090 Ti's 6 MiB L2 partition but the value_states tile
  is only 2048*128*2 = 0.5 MiB bf16 per batch. The 40-batch
  spread keeps cross-batch L2 reuse low. cuBLAS may select a
  different strided-batched kernel on B200 (TCGen05 / TMEM
  bf16 batched-GEMM path vs Ampere's bf16 strided-batched-
  TensorOp path).
- DSMEM and cluster-cooperative tile staging (sm_100a-only)
  are out of scope at this Sm80-portability level.
- TCGen05 / TMEM divergence: the batched-bf16-GEMM runs on
  Ampere's bf16 strided-batched TensorOp pipe on the 3090 Ti;
  B200's TCGen05 / TMEM batched-bf16 path is NOT exercised at
  this level.
- L2 carveout / partition: workload-dependent. The 16
  canonical workloads span `B*H*S` (M_eff) of roughly
  `1*40*131 = 5240` up to `1*40*2048 = 81920`. Both the
  large-S self-attention shapes and the small-S decode-style
  shapes likely select the same cuBLAS strided-batched kernel
  family on Sm80 (square Q≈K means tile selection is
  symmetric in M and K).
- The deployed pure-ATen/cuBLAS path means the cohort-B probe
  (cublasGemmStridedBatchedEx bf16 at the canonical largest-
  flop workload shape) measures the same algorithm-selection
  path the deployed code uses — probe runtime-contract
  fidelity per BL-20260515 is preserved at the kernel-
  selection layer.

### Calibration relevance vs problem_083

problem_083 and problem_032 share the structural pattern but
differ in N-band (`head_dim = 64` vs `128`). The earlier
problem_083 NCU label showed an anomalous L1/L2 profile
(L1≈85%, L2≈63%) — a NEW cohort signature distinct from
every prior cuBLAS-dispatched cell (where regular cuBLAS GEMM
has L1≈0%, L2≈90%). The problem_032 label at N=128 will
confirm or refute the candidate
`BL-strided-batched-cublas-small-N-l1-mainloop-reuse-cohort`
BitLesson hypothesis at a different N-band.
