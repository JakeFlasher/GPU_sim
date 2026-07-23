# problem_001_fused_vision_multihead_attention_with_norms_backward

| Field                       | Value                                                                                                  |
|-----------------------------|--------------------------------------------------------------------------------------------------------|
| Problem                     | Fused vision multi-head attention backward with pre-LayerNorm and residual (7 gradient outputs)        |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_001_fused_vision_multihead_attention_with_norms_backward.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L2/001_fused_vision_multihead_attention_with_norms_backward/`            |
| Variant                     | `cuda_cpp`                                                                                             |
| Evidence                    | `Compile-checked-sm_120`                                                                                      |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                                        |
| Local speedup vs reference  | 0.93 - 1.03x on sm_86 across 16/16 PASSED workloads (exact match: max_rel_err = 0, max_abs_err = 0; diagnostic) |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 - 0.5+>                                       |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                                      |

## Approach

Pure ATen / cuBLAS dispatch — NO custom CUDA kernels. The backward
computes seven outputs through eight fp32 matmul dispatches (4 2D
cublasSgemm + 4 4D cublasSgemmStridedBatched) plus a softmax
backward and a full LayerNorm backward expressed as ATen
elementwise / reduction ops:

1. Output-projection backward (`output = F.linear(attn_output, out_weight)`):
   - `grad_attn_output = at::linear(grad_output, out_weight.t())` (cublasSgemm 2D fp32 OP_N OP_T).
   - `grad_out_weight = grad_output_2d.t() @ attn_output_2d` (cublasSgemm 2D fp32 OP_T OP_N).
   - `grad_out_bias = grad_output.sum(kBatchSeqDims)`.
2. Head merge + attention-application backward:
   - `grad_v_heads = attn_weights.transpose(-2, -1) @ grad_attn_output_heads` (4D cublasSgemmStridedBatched).
   - `grad_attn_weights = grad_attn_output_heads @ v.transpose(-2, -1)` (same family).
3. Softmax backward (standard formula):
   - `sum_grad = (grad_attn_weights * attn_weights).sum(dim=-1, keepdim=true)`.
   - `grad_attn_scores = attn_weights * (grad_attn_weights - sum_grad)`.
4. Scaled-dot-product backward:
   - `grad_q_heads = (grad_attn_scores @ k) * scale` (4D batched).
   - `grad_k_heads = grad_attn_scores.transpose(-2, -1) @ (q * scale)` (4D batched).
5. Head split + QKV cat + QKV-projection backward:
   - Reshape each grad_q/k/v back to [B, S, embed_dim]; cat along last dim.
   - `grad_x_norm = at::linear(grad_qkv, qkv_weight.t())` (cublasSgemm 2D fp32 OP_N OP_T).
   - `grad_qkv_weight = grad_qkv_2d.t() @ x_norm_2d` (cublasSgemm 2D fp32 OP_T OP_N).
   - `grad_qkv_bias = grad_qkv.sum(kBatchSeqDims)`.
6. LayerNorm backward (full standardization derivative):
   - `grad_ln_weight = (grad_x_norm * x_normalized).sum(kBatchSeqDims)`.
   - `grad_ln_bias = grad_x_norm.sum(kBatchSeqDims)`.
   - `grad_x_from_norm = grad_x_normalized / std - mean_grad - x_centered * mean_grad_x_centered`.
   - `grad_x = grad_residual + grad_x_from_norm`.

Const axes pinned as `constexpr`: `num_heads = 16`,
`head_dim = 64`, `embed_dim = 1024`, `qkv_dim = 3072`. Every
indexed tensor is shape-checked against these values. CUDA guard
from the first input. Launch on
`at::cuda::getCurrentCUDAStream()` per CLAUDE.md Rule 7.

The axis-list reduction args (`grad_output.sum`,
`grad_qkv.sum`, `grad_ln_weight / bias.sum`) are wrapped in
`std::vector<int64_t> kBatchSeqDims = {0, 1}` so the int-axis
overload of `Tensor::sum` binds unambiguously against the
named-axis overload under current Torch headers.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) compiles and is expected to smoke-run this baseline;
final SOL scoring is on B200 (sm_100a). Differences worth noting for this
pattern:

- HBM3e vs GDDR6X bandwidth: the attention-backward chain is memory-bound on
  the two large weight reads (qkv_weight 3072x1024 = 12 MiB; out_weight
  1024x1024 = 4 MiB) plus the 4D attention activations (attn_weights at
  [B, 16, S, S] becomes large at long sequences); the 3090 Ti's bandwidth
  ceiling makes local wall-time a weak proxy for the B200 SOL Score.
- DSMEM / cluster-cooperative reduction (sm_100a-only) is not used here;
  the LayerNorm derivatives could exploit cluster-coop on B200 in a future
  tuned variant.
- L2 carveout: B200's larger L2 absorbs both projection weights more
  effectively; the 3090 Ti's smaller L2 evicts between dispatches at
  long sequences. GDDR7 does not apply here (the 3090 Ti uses GDDR6X).
- TCGen05 / TMEM: this template does not use 5th-gen tensor cores or
  the dedicated TMEM register surface; the fp32 sgemm path is the
  conservative PyTorch dispatch and is not exposed to sm_100a-only
  TCGen05 / TMEM codegen differences.
