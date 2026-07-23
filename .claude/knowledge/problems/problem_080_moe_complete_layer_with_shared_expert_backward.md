---
Use when: starting the L2 multi-op fused baseline for `problem_080_moe_complete_layer_with_shared_expert_backward` -- the backward pass through an MoE decoder layer with a shared expert. Multi-pattern problem composing MoE dispatch + grouped GEMMs + shared-expert path + residual + RMSNorm + activation backprop. BF16 / FP32 mixed precision.
Primary raw source: references/NVIDIA_Manuals/solbench_problems/problem_080_moe_complete_layer_with_shared_expert_backward.md
Supporting sources: .claude/knowledge/patterns/gemm_baseline.md; .claude/knowledge/patterns/fused_add_rmsnorm.md; .claude/knowledge/sources/cutlass-cute-docs.md
Repo touchpoints: baselines/L2/problem_080_moe_complete_layer_with_shared_expert_backward/cutlass/; cutlass/include/cutlass/gemm/group_array_problem_shape.hpp; sol-execbench/examples/cutlass/gemm/
Patents anchored here: None
Out of scope:
  - Python / Triton / CuTe DSL Python / cuTile Python / cudnn_frontend Python / torch.compile submissions (CLAUDE.md Rule 5).
  - NCU / `nv-nsight-cu-cli` / `.ncu-rep`/`.ncu-repz` artefacts on any device, including local 5060 (CLAUDE.md Rule 1).
  - `nvidia-smi --lock-gpu-clocks` / `--lock-memory-clocks` / persistence / power / MIG / compute-mode mutation (CLAUDE.md Rule 2).
  - Privileged CUPTI: Events, Metrics, Range Profiling, PM Sampling, PerfWorks (CLAUDE.md Rule 3; CUPTI Activity remains allowed).
  - Host-driver work: NVIDIA driver install, DKMS, MOK, `modprobe nvidia`, kernel-module parameter toggles, reboot workflows (CLAUDE.md Rule 4).
  - User CUDA streams, background host threads, `torch.jit.fork`, null-stream side effects, CUDA Graphs without a per-problem note (CLAUDE.md Rule 7).
  - Evaluator-state exploits: `data_ptr` caching, lazy-eval `__eq__` on `FakeTensor`, monkey-patched `Event.elapsed_time` / `do_bench` (CLAUDE.md Rule 6).
  - Precision-contract violations: precision downgrade (e.g. FP16 compute upcast to FP32 output) to pass tolerance (CLAUDE.md Rule 8).
Version family: sol-execbench-2026
Default skill: implementing-sol-baselines
Routing tags: moe; mixture-of-experts; backward; shared-expert; problem_080; grouped-gemm
---

# problem_080_moe_complete_layer_with_shared_expert_backward

## Use when

Implementing the backward pass of a complete MoE decoder layer that includes a shared expert. This is an L2 multi-op fused problem combining: routing-tensor scatter / gather (backward direction); grouped GEMMs across experts; shared-expert MLP backprop; residual-add gradient pass-through; RMSNorm backward.

## Workload summary

| Field | Value |
|---|---|
| Operation | Backward pass of `out = rmsnorm(residual + moe_layer(x) + shared_expert(x))` |
| Dtypes | BF16 activations / weights, FP32 gradients per CLAUDE.md Rule 8 |
| Patterns composed | grouped-GEMM (`patterns/gemm_baseline.md` × experts), `patterns/fused_add_rmsnorm.md` (backward path), scatter / gather routing |
| Category | L2 (multi-op fused) |

## Matching patterns

- `.claude/knowledge/patterns/gemm_baseline.md` -- each expert is a small GEMM; the grouped-GEMM variant uses `cutlass::gemm::GroupedGemmKernel` to launch all expert matmuls in a single kernel.
- `.claude/knowledge/patterns/fused_add_rmsnorm.md` -- the residual-RMSNorm backward path. Note that backward through RMSNorm requires the FP32 reduction across the gradient channel that the pattern card describes.

## Implementation entry point

Author under `baselines/L2/problem_080_moe_complete_layer_with_shared_expert_backward/cutlass/` (or `cuda_cpp/`). Entry point per `sol-execbench/docs/solution.md`:

```cpp
void run(torch::Tensor grad_output, torch::Tensor hidden_states,
         torch::Tensor router_weights, torch::Tensor expert_weights,
         torch::Tensor shared_expert_weights, torch::Tensor rmsnorm_weight,
         /* gradient outputs in DPS order */
         torch::Tensor grad_hidden, torch::Tensor grad_expert,
         torch::Tensor grad_shared, torch::Tensor grad_rmsnorm) {
    // ...
}
```

(See the canonical spec for exact tensor names and order.)

## Algorithmic structure

1. **RMSNorm backward** -- compute gradient w.r.t. inputs of the post-norm; FP32 reduction across the hidden dimension.
2. **Residual gradient pass-through** -- additive; trivial.
3. **MoE dispatch (gather backward).** Gradient w.r.t. expert outputs is gathered from the per-token routing assignments. Implementation hint: use the same routing tensor produced in the forward pass; the harness passes both `router_weights` and the (implicit) routing indices.
4. **Grouped GEMM backward across experts.** For each of the `num_experts` experts, compute `grad_input = grad_output @ W_e.T` and `grad_W_e = x.T @ grad_output`. The grouped formulation launches these as one kernel.
5. **Shared expert backward.** Same shape as one expert; standard MLP backward.
6. **Sum gradients** from the shared expert and the MoE path; combine with the RMSNorm residual gradient.

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

`sm_120` (RTX 5060) and `sm_100a` (B200) divergences for this composite problem:

- **Grouped-GEMM scheduling.** B200's TCGen05 cooperative pipeline benefits the per-expert matmul more than sm_120 can; on the 5060 the multistage schedule is used.
- **Routing-tensor scatter.** B200's DSMEM peer-loads enable cluster-cooperative scatter on the gradient gather step; sm_120 must use the global-memory scatter path.
- **Memory bandwidth.** B200's HBM3e bandwidth is much higher than sm_120's GDDR7; the all-expert-weights traffic for the grouped-GEMM is dramatically more affordable on B200.

Evidence-label vocabulary per `CLAUDE.md` lives in `baselines/L2/problem_080_moe_complete_layer_with_shared_expert_backward/README.md`, NOT in this dossier.

## Cross-references

- Source spec: `references/NVIDIA_Manuals/solbench_problems/problem_080_moe_complete_layer_with_shared_expert_backward.md`.
- CUTLASS grouped-GEMM reference: see `cutlass/include/cutlass/gemm/` for the GroupedGemm variants; `cutlass/examples/` for an illustrative grouped-GEMM example.
