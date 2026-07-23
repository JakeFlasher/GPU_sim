# problem_093_grouped_topk_moe_routing_backward

| Field                       | Value                                                                                                  |
|-----------------------------|--------------------------------------------------------------------------------------------------------|
| Problem                     | Grouped top-k MoE routing backward (2 gradient outputs)                                                |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_093_grouped_topk_moe_routing_backward.md`         |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/093_grouped_topk_moe_routing_backward/`                               |
| Variant                     | `cuda_cpp`                                                                                             |
| Evidence                    | `Profiled-local-sm_120`                                                                                      |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                                       |
| Local speedup vs reference  | 0.74 - 0.84x on sm_86 across 16/16 PASSED workloads (exact match: max_rel_err = 0, max_abs_err = 0; diagnostic) |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 - 0.5+>                                       |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                                      |

## Approach

Pure ATen / cuBLAS dispatch — NO custom CUDA kernels. The backward
computes two outputs (grad_hidden_states, grad_weight) through two
2D bf16 `cublasGemmEx` dispatches on the verified
`cublasGemmEx_2D_bf16_OP_N_OP_N` family plus sigmoid backward,
gather/scatter_add backward, and quotient-rule normalization backward
via ATen elementwise / reduction ops.

1. **Gradient through scaling**:
   `grad_topk_weights_normalized = grad_topk_weights * routed_scaling_factor`.
2. **Gradient through per-row normalization** (quotient rule):
   `grad_sum = (grad_normalized * topk_weights_normalized).sum(dim=-1, keepdim=true)`;
   `grad_topk_weights_unnorm = (grad_normalized - grad_sum) / denominator`.
3. **Gradient through gather operation**:
   `grad_scores = zeros(N, n_routed_experts)`;
   `grad_scores.scatter_add_(dim=1, topk_indices, grad_topk_weights_unnorm)`.
4. **Gradient through sigmoid**:
   `grad_router_logits = grad_scores * scores * (1 - scores)`.
5. **Gradient through linear projection**:
   - `grad_hidden_states = at::matmul(grad_router_logits, weight)` —
     cublasGemmEx 2D bf16 OP_N OP_N (the verified family canonical binding).
   - `grad_weight = grad_router_logits.t() @ hidden_states` —
     cublasGemmEx 2D bf16 OP_T OP_N (the verified family's transposed-A variant).

Const axes pinned as `constexpr`: `hidden_size = 5120`,
`n_routed_experts = 160`, `top_k = 8`. Variable axis `N` (total
tokens) is workload-driven and recovered from input tensor shapes
at runtime. Every input AND every DPS output is explicitly
shape-checked against the const + recovered axes (11 TORCH_CHECK
rank+size assertions total).

CUDA guard from `hidden_states`. Launch on
`at::cuda::getCurrentCUDAStream()` per CLAUDE.md Rule 7.

The dispatch-family pin is `cublasGemmEx_2D_bf16_OP_N_OP_N`
(verified in `.humanize/corpus-expansion/verified_dispatch_families.toml`
by problem_004). Both grad_input and grad_weight matmuls are
explicit in source (`at::matmul(g, W)` and `at::matmul(g.t(), x)`);
no inference gap.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) compiles and is expected to smoke-run this baseline;
final SOL scoring is on B200 (sm_100a). Differences worth noting for this
pattern:

- HBM3e vs GDDR6X / GDDR7 bandwidth: the router weight matrix is
  small (`[160, 5120]` bf16 = 1.6 MiB) so the projection itself is
  not bandwidth-bound; the dominant memory traffic at large `N` is
  the per-token `scores` tensor (`[N, 160]` bf16). The 3090 Ti's
  GDDR6X bandwidth ceiling makes local wall-time a weak proxy for
  the B200 SOL Score; GDDR7 does not apply on the 3090 Ti.
- DSMEM / cluster-cooperative reduction (sm_100a-only) is not used
  here; the per-row sum reduction in the quotient-rule backward could
  exploit cluster-coop on B200 in a future tuned variant.
- L2 carveout / L2 cache partitioning: B200's larger L2 absorbs the
  router weight matrix entirely and the `scores` tensor at most batch
  sizes; the 3090 Ti's smaller L2 evicts the `scores` tensor at
  large `N`.
- scatter_add intrinsics: ATen's bf16 scatter_add uses int64 indices.
  On B200, the cooperative-groups int64 scatter path may have
  different throughput characteristics from sm_86.
- TCGen05 / TMEM: this template does not use 5th-gen tensor cores or
  the dedicated TMEM register surface; the bf16 cublasGemmEx path is
  the conservative PyTorch dispatch and is not exposed to sm_100a-only
  TCGen05 / TMEM codegen differences.
