# problem_060_fused_attention_qk_matmul_scale_mask_softmax_backward

| Field                       | Value                                                                                                  |
|-----------------------------|--------------------------------------------------------------------------------------------------------|
| Problem                     | Fused attention QK matmul + scale + mask + softmax backward (2 gradient outputs)                       |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_060_fused_attention_qk_matmul_scale_mask_softmax_backward.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/060_fused_attention_qk_matmul_scale_mask_softmax_backward/`           |
| Variant                     | `cuda_cpp`                                                                                             |
| Evidence                    | `Compile-checked-sm_120`                                                                                      |
| Local speedup vs reference  | <to be filled after native run>                                                                        |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 - 0.5+>                                       |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                                      |

## Approach

Pure ATen / cuBLAS dispatch — NO custom CUDA kernels. The backward
computes two outputs (grad_query, grad_key) through two 4D bf16
`cublasGemmStridedBatchedEx` dispatches on the verified
`cublasGemmStridedBatchedEx_4D_bf16_OP_N_OP_N` family plus a
softmax backward (in fp32, cast back to bf16) and a scale backward
expressed as ATen elementwise / reduction ops.

1. **Softmax backward** (numerically-stable formulation, fp32
   math; cast back to bf16 for the downstream matmul backwards):
   - `grad_output_f32 = grad_output.float()`.
   - `attn_weights_f32 = attn_weights.float()`.
   - `sum_grad = (grad_output_f32 * attn_weights_f32).sum(-1, keepdim=true)`.
   - `grad_attn_logits = attn_weights_f32 * (grad_output_f32 - sum_grad)`,
     cast back to bf16.
2. **Scale backward**: `grad_scaled_logits = grad_attn_logits * scaling`.
3. **Q @ K^T matmul backward** (two 4D bf16 batched matmul
   dispatches on the verified family):
   - `grad_query = at::matmul(grad_scaled_logits, key)` —
     `cublasGemmStridedBatchedEx` 4D bf16 OP_N OP_N (the verified
     family canonical binding).
   - `grad_key = at::matmul(grad_scaled_logits.transpose(-2, -1), query)` —
     `cublasGemmStridedBatchedEx` 4D bf16 OP_T OP_N (the verified
     family's transposed-A variant).

Const axes pinned as `constexpr`: `num_heads = 128`, `head_dim = 128`.
Variable axes `B = batch_size`, `seq_len_q`, and `seq_len_k` are
workload-driven and recovered from input tensor shapes at runtime.
Every input AND every DPS output is explicitly shape-checked
against the const + recovered axes (6 TORCH_CHECK rank+size
assertions total).

CUDA guard from `query`. Launch on
`at::cuda::getCurrentCUDAStream()` per CLAUDE.md Rule 7.

The dispatch-family pin is `cublasGemmStridedBatchedEx_4D_bf16_OP_N_OP_N`
(verified in `.humanize/corpus-expansion/verified_dispatch_families.toml`
by problem_083). Both matmul branches are explicit in source
(`at::matmul(grad_scaled_logits, key)` and `at::matmul(g.transpose(-2, -1), Q)`);
no inference gap.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) compiles and is expected to smoke-run this baseline;
final SOL scoring is on B200 (sm_100a). Differences worth noting for this
pattern:

- HBM3e vs GDDR6X / GDDR7 bandwidth: the dominant memory traffic is
  the per-batch attention activations (`attn_weights` and
  `grad_output` both at `[B, num_heads=128, seq_len_q, seq_len_k]`
  bf16; at large `seq_len_q * seq_len_k * num_heads` this can be
  hundreds of MiB per tensor). The 3090 Ti's GDDR6X bandwidth
  ceiling makes local wall-time a weak proxy for the B200 SOL Score;
  GDDR7 does not apply on the 3090 Ti.
- DSMEM / cluster-cooperative reduction (sm_100a-only) is not used
  here; the per-row sum reduction in the softmax backward could
  exploit cluster-coop on B200 in a future tuned variant.
- L2 carveout / L2 cache partitioning: B200's larger L2 absorbs
  bigger slices of `attn_weights` and `grad_output` per launch;
  the 3090 Ti's smaller L2 evicts between dispatches at long
  sequences.
- bf16 / fp32 mixed-precision: the softmax backward runs in fp32
  (per the reference, for numerical stability) and casts back to
  bf16 for the matmul backward. On B200, the TCGen05 fp32
  accumulator path could fuse the cast more efficiently than the
  conservative ATen elementwise path on sm_86.
- TCGen05 / TMEM: this template does not use 5th-gen tensor cores
  or the dedicated TMEM register surface; the bf16
  cublasGemmStridedBatchedEx path is the conservative PyTorch
  dispatch and is not exposed to sm_100a-only TCGen05 / TMEM
  codegen differences.
