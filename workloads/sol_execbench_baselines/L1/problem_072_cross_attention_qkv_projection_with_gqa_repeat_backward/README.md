# problem_072_cross_attention_qkv_projection_with_gqa_repeat_backward

| Field                       | Value                                                                                                              |
|-----------------------------|--------------------------------------------------------------------------------------------------------------------|
| Problem                     | Cross-attention QKV projection backward with GQA repeat (16 heads / 4 KV heads, n_rep = 4)                         |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_072_cross_attention_qkv_projection_with_gqa_repeat_backward.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/072_cross_attention_qkv_projection_with_gqa_repeat_backward/`                     |
| Variant                     | `cuda_cpp`                                                                                                         |
| Evidence                    | `Profiled-local-sm_120`                                                                                                  |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                                                   |
| Local speedup vs reference  | 0.88 - 1.44x on sm_86 across 16/16 PASSED workloads (exact match: max_rel_err = 0, max_abs_err = 0; diagnostic)    |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 - 0.5+>                                                   |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                                                  |

## Approach

Pure ATen / cuBLAS dispatch — NO custom CUDA kernels. The backward
computes five outputs through six fp32 2D `at::matmul` calls + a
GQA-repeat reverse-sum + transpose / view layout flips:

1. GQA-repeat reverse-sum (K / V only): the forward repeats each
   KV head `n_rep = num_heads / num_kv_heads = 4` times to align
   with the 16 attention heads. Backward sums across the
   repetition dim:
   - `grad_K_un = grad_key_states.view(B, num_kv_heads, num_kv_groups, S_enc, head_dim).sum(/*dim=*/2)`.
   - Same for V.
   The positional `int64_t` dim argument binds the int-axis
   overload of `Tensor::sum` unambiguously (a brace-init
   `.sum({2})` form can resolve to either the int-axis or
   named-axis Tensor::sum overload under current Torch
   headers; the positional `int64_t` argument binds the
   int-axis path).
2. Transpose + view back to projection layout (per Q / K / V):
   - `grad_q_proj = grad_query_states.transpose(1, 2).contiguous().view(B, S_dec, q_out_dim)`.
   - Same for `grad_k_proj` and `grad_v_proj` with `kv_out_dim`.
3. Linear backward:
   - `grad_decoder_hidden_states = grad_q_proj @ q_weight`
     (cublasSgemm OP_N OP_N).
   - `grad_encoder_hidden_states = grad_k_proj @ k_weight + grad_v_proj @ v_weight`
     (two cublasSgemm OP_N OP_N matmuls summed into the DPS output).
   - `grad_q_weight = grad_q_proj_2d.t() @ decoder_2d` (cublasSgemm OP_T OP_N).
   - `grad_k_weight = grad_k_proj_2d.t() @ encoder_2d` (same).
   - `grad_v_weight = grad_v_proj_2d.t() @ encoder_2d` (same).

Const axes pinned as `constexpr`: `hidden_size = 1536`,
`cross_hidden_size = 1024`, `num_heads = 16`,
`num_kv_heads = 4`, `num_kv_groups = 4`, `head_dim = 256`,
`q_out_dim = 4096`, `kv_out_dim = 1024`. Every indexed tensor is
shape-checked against these values. CUDA guard from the first
input. Launch on `at::cuda::getCurrentCUDAStream()` per
CLAUDE.md Rule 7.

Cross-attention asymmetry: the decoder side uses
`hidden_size = 1536` while the encoder side uses
`cross_hidden_size = 1024`. The Q gradient routes back to
the decoder side; the K + V gradients accumulate into the
encoder side. The two `grad_input` outputs have distinct
shapes.

GQA structure: 16 attention heads share KV across 4 KV heads
(`num_kv_groups = 4` per KV head). The backward respects this
repetition by summing K / V gradients across the repetition
dim before transposing back to projection layout.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) compiles and is expected to smoke-run this baseline;
final SOL scoring is on B200 (sm_100a). Differences worth noting for this
pattern:

- HBM3e vs GDDR6X bandwidth: cross-attention QKV backward is memory-bound on
  the three weight reads and the encoder-side accumulation; the 3090 Ti's
  bandwidth ceiling makes local wall-time a weak proxy for the B200 SOL
  Score. Local timing primarily validates correctness and launch-config
  sanity.
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
