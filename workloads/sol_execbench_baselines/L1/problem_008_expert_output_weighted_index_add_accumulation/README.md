# problem_008_expert_output_weighted_index_add_accumulation

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | MoE atomic expert-output accumulation: copy `final_hidden_states` then `output[token_indices[i]] += expert_outputs[i]` over `num_selected_tokens` source rows. hidden_size=3072 pinned. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_008_expert_output_weighted_index_add_accumulation.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/008_expert_output_weighted_index_add_accumulation/`        |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

Two-phase pipeline on the current CUDA stream:

1. **Phase 1 — copy**. `cudaMemcpyAsync` from `final_hidden_states` to
   `output` (both bf16, same shape). The reference `.clone()` semantics
   require a fresh accumulation buffer so the input is not mutated.
2. **Phase 2 — atomic scatter-add**. 2D grid
   `(num_selected_tokens, ceil(H / BLOCK))`, 256 threads per block.
   For each (sel, h) covered by a thread, one bf16 `atomicAdd` into
   `output[token_indices[sel], h]` from `expert_outputs[sel, h]`. The
   bf16 `atomicAdd` overload requires compute capability ≥ 8.0, which
   covers the local 3090 Ti (sm_86) and the Phase-2 B200 (sm_100a)
   target.

The reference uses `torch.Tensor.index_add_`, which is implemented on
the CUDA backend with the same atomic pattern. Atomic ordering is
non-deterministic in both the reference and our kernel, so the
floating-point sum is bit-identical only up to per-row ordering; the
workload tolerance allows this slack.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was compile-
checked and natively-correctness-validated on. Final SOL scoring is
on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5090): the
  atomic write path is bandwidth-light per atomic (single bf16) but
  contention-heavy at the same destination row. B200's larger L2
  partition and HBM3e bandwidth handle contention better than the
  3090 Ti / 5090 GDDR-class memory subsystems.
- DSMEM and cluster-cooperative reduction (sm_100a-only) are out
  of scope; the kernel does not stage source rows in shared memory.
- L2 carveout / cache: the per-row `output[token_indices[i]]` write
  touches `batch_seq_len × hidden_size × 2 B` total. On the 3090 Ti
  the cache pressure depends on token-index locality; B200's L2 is
  large enough to hold every row for most workloads. This is one of
  the predictor-hostile patterns the cache-prediction calibration
  needs to label honestly.
- TCGen05 / TMEM divergence: this kernel does not use tensor cores
  at all (the math is just atomic addition), so the TCGen05 / TMEM
  split is not exercised.
