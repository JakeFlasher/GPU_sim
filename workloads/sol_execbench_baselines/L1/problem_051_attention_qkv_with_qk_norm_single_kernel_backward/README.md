# problem_051_attention_qkv_with_qk_norm_single_kernel_backward

| Field                       | Value                                                                                                  |
|-----------------------------|--------------------------------------------------------------------------------------------------------|
| Problem                     | Gemma3 fused QKV projection with Q/K RMSNorm single-kernel backward (6 gradient outputs)               |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_051_attention_qkv_with_qk_norm_single_kernel_backward.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/051_attention_qkv_with_qk_norm_single_kernel_backward/`               |
| Variant                     | `cuda_cpp`                                                                                             |
| Evidence                    | `Profiled-local-sm_120`                                                                                      |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                                       |
| Local speedup vs reference  | 1.07 - 1.17x on sm_86 across 16/16 PASSED workloads (exact match: max_rel_err = 0, max_abs_err = 0; diagnostic) |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 - 0.5+>                                       |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                                      |

## Approach

Pure ATen / cuBLAS dispatch — NO custom CUDA kernels. The backward
computes six outputs (grad_hidden_states, grad_q_weight,
grad_k_weight, grad_v_weight, grad_q_norm_weight, grad_k_norm_weight)
through six 2D bf16 `cublasGemmEx` dispatches on the verified
`cublasGemmEx_2D_bf16_OP_N_OP_N` family plus Q/K RMSNorm backward
expressed as ATen elementwise / reduction ops.

1. **Q RMSNorm backward** (Gemma3 uses `(1 + q_norm_weight)` as the
   RMSNorm scale; the backward math runs in fp32 then converts back
   to bf16):
   - `q_scale = (1 + q_norm_weight).float()`.
   - `grad_q_norm_weight = (grad_query.float() * q_normed.float()).sum(dim=(0, 1, 2))`.
   - `grad_q_normed_bf16 = (grad_query.float() * q_scale).to(bf16)`.
   - `q_mean_term = (grad_q_normed_bf16.float() * q_normed.float()).mean(-1, keepdim=true)`.
   - `grad_q_transposed = q_rstd * (grad_q_normed_bf16.float() - q_mean_term * q_normed.float())`,
     cast to bf16.
2. **K RMSNorm backward**: analogous to Q.
3. **V no normalization**: `grad_v_transposed = grad_value`.
4. **Transpose + reshape** the per-head gradient back to flat
   `[B, S, *_proj_size]`:
   - `grad_q_reshaped = grad_q_transposed.transpose(1, 2).contiguous().view({B, S, q_proj_size=1024})`.
   - K and V analogously to `[B, S, kv_proj_size=256]`.
5. **Linear projection backwards** (six 2D bf16 sgemm dispatches on
   the verified family):
   - `grad_hidden_states = (grad_q_proj @ q_weight) + (grad_k_proj @ k_weight) + (grad_v_proj @ v_weight)`
     — three OP_N OP_N matmuls (the verified family canonical
     binding).
   - `grad_q_weight = grad_q_proj_2d.t() @ hidden_states_2d` —
     OP_T OP_N (the verified family's transposed-A variant).
   - `grad_k_weight`, `grad_v_weight`: analogous OP_T OP_N matmuls.

Const axes pinned as `constexpr`: `hidden_size = 640`,
`num_attention_heads = 4`, `num_key_value_heads = 1`, `head_dim = 256`,
`q_proj_size = num_attention_heads * head_dim = 1024`,
`kv_proj_size = num_key_value_heads * head_dim = 256`. Variable axes
`B = batch_size` and `S = seq_len` recovered from input tensor shapes
at runtime. Every input AND every DPS output is explicitly
shape-checked against the const + recovered axes (20 TORCH_CHECK
rank+size assertions total).

CUDA guard from `hidden_states`. Launch on
`at::cuda::getCurrentCUDAStream()` per CLAUDE.md Rule 7.

The axis-list reduction args for `grad_q_norm_weight` and
`grad_k_norm_weight` (sums over batch / num_heads / seq_len axes)
use a typed `std::vector<int64_t> kBatchHeadSeqDims = {0, 1, 2}` to
bind the int-axis Tensor::sum overload unambiguously.

The dispatch-family pin is `cublasGemmEx_2D_bf16_OP_N_OP_N`
(verified in `.humanize/corpus-expansion/verified_dispatch_families.toml`
by problem_004). All six projection matmuls are explicit in source
(`at::matmul(g, W)` for grad_input; `at::matmul(g.t(), x)` for
grad_weight); no inference gap.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) compiles and is expected to smoke-run this baseline;
final SOL scoring is on B200 (sm_100a). Differences worth noting for this
pattern:

- HBM3e vs GDDR6X / GDDR7 bandwidth: the QKV weights are small
  (`q_weight [1024, 640]` = 1.3 MiB; `k_weight / v_weight [256, 640]`
  = 0.3 MiB each; total ~1.9 MiB bf16) so the projection itself is
  not bandwidth-bound; the dominant memory traffic is the per-batch
  `hidden_states` tensor. The 3090 Ti's GDDR6X bandwidth ceiling
  makes local wall-time a weak proxy for the B200 SOL Score; GDDR7
  does not apply on the 3090 Ti.
- DSMEM / cluster-cooperative reduction (sm_100a-only) is not used
  here; the per-token RMSNorm sum reduction could exploit
  cluster-coop on B200 in a future tuned variant.
- L2 carveout / L2 cache partitioning: B200's larger L2 absorbs all
  three QKV weight matrices; the 3090 Ti's smaller L2 evicts between
  per-branch dispatches at long sequences.
- bf16 / fp32 mixed-precision math: the RMSNorm backward runs in
  fp32 (per the reference) and casts back to bf16 for the projection
  matmul backward. On B200, the TCGen05 fp32 accumulator path could
  fuse the cast more efficiently than the conservative ATen elementwise
  path on sm_86.
- TCGen05 / TMEM: this template does not use 5th-gen tensor cores or
  the dedicated TMEM register surface; the bf16 cublasGemmEx path is
  the conservative PyTorch dispatch and is not exposed to sm_100a-only
  TCGen05 / TMEM codegen differences.
