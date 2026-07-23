# problem_003_grouped_query_attention_with_rope_backward

| Field                       | Value                                                                                                  |
|-----------------------------|--------------------------------------------------------------------------------------------------------|
| Problem                     | Grouped query attention with RoPE backward (5 gradient outputs)                                        |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_003_grouped_query_attention_with_rope_backward.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L2/003_grouped_query_attention_with_rope_backward/`                      |
| Variant                     | `cuda_cpp`                                                                                             |
| Evidence                    | `Profiled-local-sm_120`                                                                                      |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                                       |
| Local speedup vs reference  | <to be filled after native run>                                                                        |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 - 0.5+>                                       |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                                      |

## Approach

Pure ATen / cuBLAS dispatch — NO custom CUDA kernels. The backward
computes five outputs through four 4D bf16
`cublasGemmStridedBatchedEx` attention-backward dispatches plus
four 2D bf16 `cublasGemmEx` projection-backward dispatch pairs
(Q/K/V/O grad_input + grad_weight) plus a softmax backward
(fp32 math), a 2D RoPE backward via `rotate_half_inverse`, a GQA
repeat reverse-sum, and a grad_hidden_states accumulation across
the three Q/K/V branches.

1. **Output projection backward**:
   - `grad_attn_output = at::matmul(grad_output, o_weight)` —
     cublasGemmEx 2D bf16 OP_N OP_N (secondary verified-family
     surface).
   - `grad_o_weight = grad_output_2d.t() @ attn_output_2d` —
     cublasGemmEx 2D bf16 OP_T OP_N.
2. **Reshape** grad_attn_output to `[B, num_attention_heads, S, head_dim]`.
3. **Attention-application backward** (4D bf16 batched —
   the dominant verified-family witness):
   - `grad_attn_weights = grad_attn_output @ value_states.transpose(2, 3)`.
   - `grad_value_states = attn_weights.transpose(2, 3) @ grad_attn_output`.
4. **Softmax backward** (numerically-stable, fp32 math, cast back
   to bf16):
   - `sum_grad = (grad_attn_weights * attn_weights).sum(-1, keepdim=true)`.
   - `grad_attn_scores = attn_weights * (grad_attn_weights - sum_grad)`.
5. **Scale backward**: `grad_attn_scores *= scaling`.
6. **Scaled-dot-product backward** (4D bf16 batched — canonical
   verified-family OP_N OP_N + OP_T OP_N bindings):
   - `grad_query_states = grad_attn_scores @ key_states` —
     `cublasGemmStridedBatchedEx_4D_bf16_OP_N_OP_N` (verified
     family canonical binding).
   - `grad_key_states = grad_attn_scores.transpose(2, 3) @ query_states`.
7. **GQA repeat reverse-sum**: collapse num_attention_heads back
   to num_key_value_heads by summing over the
   `num_key_value_groups` dim:
   - `grad_key_states.reshape({B, num_kv_heads, num_kv_groups, S, head_dim}).sum(dim=2)`.
   - Analogous for `grad_value_states`.
8. **2D RoPE backward** via `rotate_half_inverse`:
   - `grad_query_states_pre_rope = grad_query_states * cos_expanded + rotate_half_inverse(grad_query_states * sin_expanded)`.
   - Analogous for K. V passes through (no RoPE on V).
9. **Transpose + reshape** back to `[B, S, *_proj_size]`.
10. **Q/K/V projection backwards** (3 bf16 sgemm grad_input
    pairs on the verified 2D bf16 family secondary surface):
    - `grad_hidden_from_q = at::matmul(grad_query_proj, q_weight)`.
    - `grad_hidden_from_k = at::matmul(grad_key_proj, k_weight)`.
    - `grad_hidden_from_v = at::matmul(grad_value_proj, v_weight)`.
    - `grad_hidden_states = grad_hidden_from_q + grad_hidden_from_k + grad_hidden_from_v`.
    - `grad_q_weight = grad_q_proj_2d.t() @ hidden_states_2d`.
    - Analogous for K and V grad_weight branches.

Const axes pinned as `constexpr`: `hidden_size = 5120`,
`num_attention_heads = 32`, `num_key_value_heads = 8`,
`head_dim = 128`, `num_key_value_groups = 4`,
`q_proj_size = 4096`, `kv_proj_size = 1024`. Variable axes
`B = batch_size` and `S = seq_len` are workload-driven.
Every input AND every DPS output is explicitly shape-checked
against const + recovered axes (18 TORCH_CHECK rank+size
assertions total).

CUDA guard from `hidden_states`. Launch on
`at::cuda::getCurrentCUDAStream()` per CLAUDE.md Rule 7.

`rotate_half_inverse(x) = cat([x[..., d/2:], -x[..., :d/2]], dim=-1)`
implemented via `Tensor::slice` + `at::cat` (same pattern as
problem_017_fused_vision_cu_seqlens).

The dispatch-family pin is
`cublasGemmStridedBatchedEx_4D_bf16_OP_N_OP_N` (verified in
`.humanize/corpus-expansion/verified_dispatch_families.toml`
by problem_083). All four 4D attention matmul branches and
the four 2D projection-backward grad_input branches are
explicit in source (`at::matmul(g, W)` / `at::matmul(g, W.transpose(...))`);
no inference gap.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) compiles and is expected to smoke-run this baseline;
final SOL scoring is on B200 (sm_100a). Differences worth noting for this
pattern:

- HBM3e vs GDDR6X / GDDR7 bandwidth: the four projection weights
  total ~22 MiB bf16; the per-batch attention activations at
  `[B, num_attention_heads=32, S, S]` bf16 can grow into the
  hundreds of MiB at large `S`. The 3090 Ti's GDDR6X bandwidth
  ceiling makes local wall-time a weak proxy for the B200 SOL
  Score; GDDR7 does not apply on the 3090 Ti.
- DSMEM / cluster-cooperative reduction (sm_100a-only) is not used
  here; the per-row sum reduction in the softmax backward could
  exploit cluster-coop on B200 in a future tuned variant.
- L2 carveout / L2 cache partitioning: B200's larger L2 absorbs
  bigger slices of the attention activations per launch.
- GQA repeat reverse-sum: the sum-over-num_kv_groups path is a
  standard ATen reduction; B200's wider on-chip storage could
  pipeline this with the preceding 4D batched matmul more
  efficiently than sm_86.
- TCGen05 / TMEM: this template does not use 5th-gen tensor cores
  or the dedicated TMEM register surface; the bf16
  cublasGemmStridedBatchedEx path is the conservative PyTorch
  dispatch and is not exposed to sm_100a-only TCGen05 / TMEM
  codegen differences.
