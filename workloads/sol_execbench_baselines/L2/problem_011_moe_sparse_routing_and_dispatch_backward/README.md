# problem_011_moe_sparse_routing_and_dispatch_backward

| Field                       | Value                                                                                                  |
|-----------------------------|--------------------------------------------------------------------------------------------------------|
| Problem                     | MoE sparse routing and dispatch backward (2 gradient outputs)                                          |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_011_moe_sparse_routing_and_dispatch_backward.md`  |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L2/011_moe_sparse_routing_and_dispatch_backward/`                        |
| Variant                     | `cuda_cpp`                                                                                             |
| Evidence                    | `Profiled-local-sm_120`                                                                                      |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                                       |
| Local speedup vs reference  | 0.79 - 1.03x on sm_86 across 16/16 PASSED workloads (exact match: max_rel_err = 0, max_abs_err = 0; diagnostic) |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 - 0.5+>                                       |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                                      |

## Approach

Pure ATen / cuBLAS dispatch — NO custom CUDA kernels. The backward
computes two outputs (grad_hidden_states, grad_gate_weight) through
two 2D bf16 `cublasGemmEx` dispatches on the verified
`cublasGemmEx_2D_bf16_OP_N_OP_N` family plus quotient-rule
normalization backward, top-k straight-through scatter, grad_expert_mask
permute/gather/scatter_add, and softmax backward expressed as ATen
elementwise / reduction / scatter ops.

1. **Normalization backward (quotient rule, fp32 math)**:
   - `grad_routing_weights_f32 = grad_routing_weights.float()`.
   - `routing_weights_unnorm = routing_probs.gather(dim=1, selected_experts)`.
   - `grad_sum = (grad_routing_weights_f32 * routing_weights_unnorm / routing_weights_sum).sum(dim=-1, keepdim=true)`.
   - `grad_routing_weights_unnorm = grad_routing_weights_f32 / routing_weights_sum - grad_sum / routing_weights_sum`.
2. **Top-k straight-through scatter back into routing_probs space**:
   - `grad_routing_probs = zeros_like(routing_probs)` (fp32).
   - `grad_routing_probs.scatter_(dim=1, selected_experts, grad_routing_weights_unnorm)`.
3. **Auxiliary grad_expert_mask path**:
   - `grad_expert_mask_permuted = grad_expert_mask.permute({2, 1, 0}).float()`.
   - `grad_from_mask = grad_expert_mask_permuted.gather(dim=2, selected_experts.unsqueeze(2)).squeeze(2)`.
   - `grad_routing_probs.scatter_add_(dim=1, selected_experts, grad_from_mask)`.
4. **Direct grad_router_logits path**:
   - `grad_routing_probs += grad_router_logits.float()`.
5. **Softmax backward**:
   - `dot_product = (grad_routing_probs * routing_probs).sum(dim=1, keepdim=true)`.
   - `grad_router_logits_computed = routing_probs * (grad_routing_probs - dot_product)`.
   - Cast to bf16 (`grad_router_logits_bf16`).
6. **Linear projection backward** (the two 2D bf16 sgemm
   dispatches on the verified family):
   - `grad_hidden_states = at::matmul(grad_router_logits_bf16, gate_weight)` —
     cublasGemmEx 2D bf16 OP_N OP_N (the verified family canonical binding).
   - `grad_gate_weight = grad_router_logits_bf16.t() @ hidden_states` —
     cublasGemmEx 2D bf16 OP_T OP_N (the verified family's transposed-A
     variant).

Const axes pinned as `constexpr`: `hidden_size = 2048`,
`num_experts = 128`, `top_k = 8`. Variable axis `N` (number of
tokens) is workload-driven and recovered from input tensor shapes at
runtime. Every input AND every DPS output is explicitly
shape-checked against the const + recovered axes (11 TORCH_CHECK
rank+size assertions total).

CUDA guard from `hidden_states`. Launch on
`at::cuda::getCurrentCUDAStream()` per CLAUDE.md Rule 7.

The dispatch-family pin is `cublasGemmEx_2D_bf16_OP_N_OP_N`
(verified in `.humanize/corpus-expansion/verified_dispatch_families.toml`
by problem_004). The two projection matmuls are explicit in source
(`at::matmul(grad_router_logits_bf16, gate_weight)` and
`at::matmul(grad_router_logits_bf16.t(), hidden_states)`); no
inference gap.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) compiles and is expected to smoke-run this baseline;
final SOL scoring is on B200 (sm_100a). Differences worth noting for this
pattern:

- HBM3e vs GDDR6X / GDDR7 bandwidth: the gate weight matrix is small
  (`[128, 2048]` bf16 = 0.5 MiB) and reused once; the dominant memory
  traffic is `grad_routing_probs` and `grad_expert_mask` at large `N`.
  The 3090 Ti's GDDR6X bandwidth ceiling makes local wall-time a weak
  proxy for the B200 SOL Score; GDDR7 does not apply on the 3090 Ti.
- DSMEM / cluster-cooperative reduction (sm_100a-only) is not used
  here; the per-row sum reduction in the softmax + quotient-rule
  backward could exploit cluster-coop on B200 in a future tuned
  variant.
- L2 carveout / L2 cache partitioning: B200's larger L2 absorbs the
  gate weight matrix entirely; the 3090 Ti's smaller L2 evicts
  `routing_probs` / `grad_routing_probs` at large `N`.
- scatter_ / scatter_add_ intrinsics with int64 indices: ATen's
  scatter path on B200 may benefit from cooperative-groups int64
  scatter throughput improvements relative to sm_86.
- TCGen05 / TMEM: this template does not use 5th-gen tensor cores or
  the dedicated TMEM register surface; the bf16 cublasGemmEx path is
  the conservative PyTorch dispatch and is not exposed to sm_100a-only
  TCGen05 / TMEM codegen differences.
