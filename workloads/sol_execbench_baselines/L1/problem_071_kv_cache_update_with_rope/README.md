# problem_071_kv_cache_update_with_rope

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | Fused KV-cache scatter + RoPE rotation on the freshly-projected Q / K tensors. Inputs: positions, freqs, cur K & V projections, cached KV tensors, slot mapping. Outputs: rotated K plus in-place updated KV cache. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_071_kv_cache_update_with_rope.md`      |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/071_kv_cache_update_with_rope/`                            |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                     |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                      |
| Local correctness           | All workloads PASSED on the live sm_120 host; full per-workload evidence in `scripts/baselines_harness/.harness_traces/problem_071_kv_cache_update_with_rope__cuda_cpp/trace.json`. |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | Profiled via the Stage-3 surfaces orchestrator; manifest at `scripts/baselines_harness/.harness_traces/problem_071_kv_cache_update_with_rope__cuda_cpp/manifest.json` records the per-surface status set. |

## Approach

Single-launch fused kernel: per token, the kernel reads cur K and V,
applies RoPE rotation to K using the position / freqs tensors, and
scatters K_rot + V into the cached KV tensors at the slot the
slot-mapping table designates. The fusion keeps the rotated K from
round-tripping through global memory between the RoPE and scatter
stages. Lane / tile layout is documented in source comments under
`cuda_cpp/kernel.cu`.

## Architecture-coupling caveats

The kernel was promoted from `Compile-checked-sm_120` to
`Profiled-local-sm_120` by running the Stage-3 surfaces orchestrator
on the live RTX 5060 Laptop host. Final SOL scoring on B200
(sm_100a) will exercise the HBM3e + larger-L2 path; the per-shape
behaviour of the scatter-and-rotate fusion will differ from sm_120
once the B200 fanout lands.
