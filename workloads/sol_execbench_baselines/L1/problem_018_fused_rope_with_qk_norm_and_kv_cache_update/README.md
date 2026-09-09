# problem_018_fused_rope_with_qk_norm_and_kv_cache_update

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | Fused RoPE + Q/K RMSNorm + KV-cache update path. Inputs: positions, freqs, weights for Q & K norms, cur Q & K projections, cached KV tensors. Outputs: rotated+normalized Q, K plus updated KV cache. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_018_fused_rope_with_qk_norm_and_kv_cache_update.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/018_fused_rope_with_qk_norm_and_kv_cache_update/`          |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Compile-checked-sm_120`                                                                    |
| Local correctness           | partial: trace.json carries 13 workload rows on the current sm_120 host (mixed PASSED / RUNTIME_ERROR); per-row evidence in `tools/sol_baselines_harness/.harness_traces/problem_018_fused_rope_with_qk_norm_and_kv_cache_update__cuda_cpp/trace.json`. `report.json` reports `workload_count: 13` for this slug. |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

End-to-end fused kernel that performs the three sub-operations the
reference splits into independent stages — RoPE rotation, per-head
RMSNorm, and KV-cache scatter — in a single per-token launch so the
Q / K tensors never round-trip through global memory between stages.
The per-launch evidence is recorded under the
`.harness_traces/<slug>__<lang>/` bundle; the kernel source under
`cuda_cpp/` documents the lane / tile layout in source comments.

## Architecture-coupling caveats

Evidence carried at the `Compile-checked-sm_120` band because the
runtime workload distribution under the current sm_120 host produces
RUNTIME_ERROR rows that the report's evidence-integrity gate cannot
yet upgrade. Final SOL scoring is on B200 (sm_100a); the kernel's
RoPE + RMSNorm + KV-cache fusion will land different per-shape
behaviour on B200's HBM3e + larger L2 partition. A follow-up bring-
up plan can re-derive the kernel against the production sm_120 host
to flip the evidence to `Profiled-local-sm_120` once all 16
workloads PASS.
