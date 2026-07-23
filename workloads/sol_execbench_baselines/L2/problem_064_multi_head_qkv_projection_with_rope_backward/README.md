# problem_064_multi_head_qkv_projection_with_rope_backward

| Field                       | Value                                                                                                       |
|-----------------------------|-------------------------------------------------------------------------------------------------------------|
| Problem                     | Multi-head QKV projection backward with RoPE (Q/K rotated, V un-rotated)                                    |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_064_multi_head_qkv_projection_with_rope_backward.md`   |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L2/064_multi_head_qkv_projection_with_rope_backward/`                         |
| Variant                     | `cuda_cpp`                                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                                            |
| Local speedup vs reference  | 16/16 PASSED on sm_86 (diagnostic latency captured per workload; local-only)                                |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 - 0.5+>                                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                                           |

## Approach

Pure ATen / cuBLAS dispatch — NO custom CUDA kernels. The backward
computes four outputs (`grad_hidden_states`, `grad_q_weight`,
`grad_k_weight`, `grad_v_weight`) through six 2D fp32 `at::matmul`
calls + elementwise RoPE backward + a transpose-and-view layout
flip per Q / K / V:

1. RoPE backward for Q (and the parallel K chain):
   - `cos_u = cos.unsqueeze(1)`, `sin_u = sin.unsqueeze(1)` (broadcast over the heads axis).
   - `grad_q_cos_term = grad_query * cos_u`.
   - `grad_q_sin_term = grad_query * sin_u`.
   - `grad_q_sin_1 = grad_q_sin_term[..., :half_head_dim]`,
     `grad_q_sin_2 = grad_q_sin_term[..., half_head_dim:]`.
   - `grad_q_sin_reversed = cat([grad_q_sin_2, -grad_q_sin_1], dim=-1)`
     (the reverse of `rotate_half([y1, y2]) = [-y2, y1]` is
     `[y2, -y1]`).
   - `grad_q_unrotated = grad_q_cos_term + grad_q_sin_reversed`.
2. V has no RoPE — its gradient enters the layout flip directly.
3. Transpose + view back to the projection layout for Q / K / V:
   - `grad_q_proj = grad_q_unrotated.transpose(1, 2).contiguous().view(B, S, q_out_dim)`.
   - Same for `grad_k_proj` and `grad_v_proj` with their kv_out_dim.
4. Backward through the three Linear projections:
   - `grad_hidden_states = grad_q_proj @ q_weight + grad_k_proj @ k_weight + grad_v_proj @ v_weight`
     (three cublasSgemm OP_N OP_N dispatches summed into the DPS output).
   - `grad_q_weight = grad_q_proj_2d.t() @ hidden_states_2d`
     (cublasSgemm OP_T OP_N).
   - `grad_k_weight = grad_k_proj_2d.t() @ hidden_states_2d` (same).
   - `grad_v_weight = grad_v_proj_2d.t() @ hidden_states_2d` (same).

Const axes pinned as `constexpr`: `hidden_size = 1536`,
`num_heads = 24`, `num_kv_heads = 4`, `head_dim = 96`,
`half_head_dim = 48`, `q_out_dim = 2304`, `kv_out_dim = 384`. Every
indexed tensor is shape-checked against these values. CUDA guard
from the first input. Launch on `at::cuda::getCurrentCUDAStream()`
per CLAUDE.md Rule 7.

GQA structure: 24 attention heads share KV across 4 KV heads
(`num_heads // num_kv_heads = 6` groups). The backward respects the
asymmetric projection sizes (`q_out_dim = 2304` vs
`kv_out_dim = 384`), routing each grad to the matching weight
tensor.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) compiles and is expected to smoke-run this baseline;
final SOL scoring is on B200 (sm_100a). Differences worth noting for this
pattern:

- HBM3e vs GDDR6X bandwidth: the QKV-backward is memory-bound on the three
  weight reads and the hidden-states accumulation; the 3090 Ti's bandwidth
  ceiling makes local wall-time a weak proxy for the B200 SOL Score. Local
  timing primarily validates correctness and launch-config sanity.
- DSMEM / cluster-cooperative reduction (sm_100a-only) are not used here;
  the three projection-head matmuls dispatch independently and could be
  cluster-cooperative on B200 in a future tuned variant.
- L2 carveout: B200's larger L2 absorbs the three weight tensors across the
  matmul passes; the 3090 Ti's smaller L2 may evict between dispatches.
  GDDR7 does not apply here (the 3090 Ti uses GDDR6X; sm_120 would be the
  GDDR7 reference if a future cloud host is allocated).
- TCGen05 / TMEM: this template does not use 5th-gen tensor cores or the
  dedicated TMEM register surface; the fp32 sgemm path is the conservative
  PyTorch dispatch and is not exposed to sm_100a-only TCGen05 / TMEM codegen
  differences. A future tensor-core-aware fused-QKV variant (which would
  touch TCGen05 / TMEM on B200) is out of scope for the current template.
