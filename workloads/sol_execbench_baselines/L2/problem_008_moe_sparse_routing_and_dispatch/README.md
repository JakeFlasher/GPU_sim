# problem_008_moe_sparse_routing_and_dispatch

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | Sparse MoE routing + per-expert dispatch. Top-8 of 128 experts per token; bf16. hidden_size=2048, intermediate_size=768, num_experts=128, num_experts_per_tok=8 pinned. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_008_moe_sparse_routing_and_dispatch.md`|
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L2/008_moe_sparse_routing_and_dispatch/`                      |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Compile-checked-sm_120`                                                                    |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                      |
| Local correctness           | All 16 workloads return RUNTIME_ERROR on the live sm_120 host; per-row evidence in `tools/sol_baselines_harness/.harness_traces/problem_008_moe_sparse_routing_and_dispatch__cuda_cpp/trace.json`. The upstream sol-execbench CLI also times out (>300s synchronous, >900s under report.py default) before workload rows can be re-collected. The bundle is currently in `tools/sol_baselines_harness/tests/fixtures/evidence_consistency_exclusions.json`, so `report.{md,json}` carry NO row for this slug — `build_report()` skips excluded bundle keys entirely. Re-inclusion requires either fixing the kernel so a fresh-run produces all-PASSED rows AND stamps the cache, or a follow-up plan that adds an orchestrator-owned identity-proof mechanism for non-cached bundles. |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

C++ port of the reference MoE pipeline, running entirely on the
current CUDA stream:

1. **Routing**. `router_logits = hidden @ gate_weight.T` (at::matmul);
   `routing_weights = softmax(router_logits.to(fp32)).to(bf16)`
   (at::softmax + at:: dtype cast); `topk(routing_weights, 8)`
   (at::topk). Optional `norm_topk_prob` divides the topk weights by
   their sum.
2. **Expert mask**. `one_hot(selected_experts).permute(2, 1, 0)` →
   `[num_experts, top_k, num_tokens]` (at::one_hot + at::permute).
3. **Per-expert loop** (128 iterations). For each expert:
   - `nonzero(expert_mask[e])` → `[n_hits, 2]` (at::nonzero).
   - If `n_hits == 0`, skip.
   - `current_state = hidden_flat.index_select(0, top_x)`
     (at::index_select).
   - `gate_out = current_state @ expert_gate_proj[e].T`
     (at::matmul).
   - `up_out = current_state @ expert_up_proj[e].T` (at::matmul).
   - `intermediate = silu(gate_out) * up_out` (**custom CUDA kernel**
     — bf16x2 grid-stride, computing `g / (1 + exp(-g)) * u` in fp32
     and rounding back to bf16).
   - `expert_out = intermediate @ expert_down_proj[e].T` (at::matmul).
   - `current_hidden = expert_out * routing_weights_topk[top_x, idx]`
     (broadcast multiply via at::index + at::unsqueeze).
   - `final_hidden.index_add_(0, top_x, current_hidden)`
     (at::index_add_).
4. **Reshape + DPS write**. `output.copy_(final_hidden.reshape(B, S, H))`.

The single custom CUDA kernel handles the SwiGLU activation fusion
that would otherwise be two separate elementwise launches per expert
(128 * 2 launches per problem invocation in the dense case). All
other ops are dispatched through PyTorch's CUDA primitives, which
go to cuBLAS for the projections and to PyTorch's CUDA kernels for
softmax / topk / index_select / index_add.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was compile-
checked and natively-correctness-validated on. Final SOL scoring is
on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5060): the
  per-expert matmul stream is bandwidth-light per call (the
  intermediate_size=768 is small) but heavy across 128 iterations.
  B200's HBM3e reshapes the bandwidth-vs-compute envelope; the same
  per-expert dispatch on the 3090 Ti / 5060 spends proportionally
  more time in memory.
- DSMEM and cluster-cooperative MoE dispatch (sm_100a-only) are out
  of scope at the cuBLAS-dispatch portability level. A B200 tuning
  pass could replace the per-expert serial loop with a grouped-GEMM
  schedule (one CUTLASS GroupedGemm covering all experts at once),
  but that is outside the template-baseline scope.
- L2 carveout / partition: the per-expert weight matrices total
  128 × 768 × 2048 × 2 B = 384 MiB each for gate / up, and 128 ×
  2048 × 768 × 2 B = 384 MiB for down. The full expert weights
  exceed the 3090 Ti's 6 MB L2 but fit comfortably in B200's much
  larger L2 partition — the per-expert cache hit pattern moves up
  the hierarchy on Blackwell. The cache-prediction calibration
  cohort labels this MoE access pattern accordingly.
- TCGen05 / TMEM divergence: the per-expert matmuls dispatch
  through cuBLAS, which on B200 chooses TCGen05 / TMEM tensor-core
  kernels automatically; on the 3090 Ti and 5060 the same call
  resolves to Ampere / Ada tensor cores. Portability is at the
  call-site level, not the schedule level.
