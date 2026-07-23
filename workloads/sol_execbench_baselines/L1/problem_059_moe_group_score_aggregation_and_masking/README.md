# problem_059_moe_group_score_aggregation_and_masking

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | fp32 MoE group-score aggregation and masking: per-group top-2 sum aggregates 256 expert scores into 8 group scores; cross-group top-4 selects surviving groups; expanded back to a 256-element mask. Two distinct DPS outputs. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_059_moe_group_score_aggregation_and_masking.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/059_moe_group_score_aggregation_and_masking/`              |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

CUDA C++ baseline that implements the MoE group-score
aggregation and masking via hierarchical top-k selection
without explicit sort.

Algorithm per token row `t`:
- Reshape `scores[t, 0..255]` into 8 groups of 32 experts.
- Per group `g`: find `max1`, `max2` of the 32 scores;
  `group_score[g] = max1 + max2`.
- Cross group: find the top-4 group_scores out of 8.
- `group_mask[t, g] = 1.0` if `g` is selected, `0.0`
  otherwise.
- `masked_scores[t, e] = scores[t, e]` if
  `group_mask[t, e/32] == 1`, else `-inf` (the workload
  spec sets `allow_negative_inf = true`).

The hierarchical top-k uses two scales of selection:
- **Per-group top-2 (k=2 from 32)**: two warp-shuffle
  find-max passes. The first pass computes `max1`. A
  ballot identifies which lane holds `max1`; the
  lowest-indexed tied lane is masked to `-FLT_MAX` and
  the second find-max gives `max2`.
- **Cross-group top-4 (k=4 from 8)**: warp 0 reads the 8
  group_scores (lanes 0..7) and runs 4 find-argmax passes,
  each masking the previously-selected lane to `-FLT_MAX`.
  The selected lane indices are mapped onto a shared
  group_mask buffer.

Kernel topology:
- One CTA per token. Grid: `(num_tokens,)`.
- `BLOCK_THREADS = num_experts = 256` (one thread per
  expert). 8 warps per CTA, one warp per group.
- Phase 1 (per-group top-2): each warp finds its
  group_score independently; warp leader writes to
  `s_group_scores[g]` in shared memory.
- Phase 2 (cross-group top-4): warp 0 (8 lanes used of
  32) runs the 4-pick selection on `s_group_scores`,
  writes `s_group_mask[g]`.
- Phase 3 (outputs): each warp leader writes
  `group_mask[t, group_idx]` to global memory; every
  thread writes `masked_scores[t, expert_idx]` from its
  register-cached input score and the shared
  `s_group_mask[group_idx]`.

`main.cpp` pins `num_experts == 256` and `n_group == 8`
via `TORCH_CHECK`; validates 2D scores / masked_scores /
group_mask shape consistency, fp32 dtype, CUDA +
contiguous. DPS entry signature
`run(scores, masked_scores, group_mask)`.

## Negative-infinity emission

The reference's `scores.masked_fill(~score_mask.bool(),
float('-inf'))` emits true negative infinity. The kernel
uses the explicit IEEE-754 bit pattern via
`__int_as_float(0xff800000)` to avoid a
`CUDART_INF_F` include-pathology under nvcc + the current
toolchain (the `cuda_fp16.h` / `device_functions.h` chain
brought in by `cuda_runtime.h` can cause build failures
when `CUDART_INF_F` is referenced from a `__global__`
context). The workload tolerance has
`allow_negative_inf = true` so the harness admits true
`-inf` in the output.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was
compile-checked and natively-correctness-validated on. Final
SOL scoring is on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5060):
  this kernel reads each input element once and writes the
  expanded output. B200's HBM3e (~8 TB/s) lifts throughput
  significantly on the input/output streams.
- DSMEM and cluster-cooperative reduction (sm_100a-only) are
  out of scope; the cross-group top-4 reduction happens
  entirely within warp 0 in shared memory.
- L2 carveout / partition: per-token tile is `256 * 4 B =
  1 KiB` input + `256 * 4 B = 1 KiB` masked_scores output +
  `8 * 4 B = 32 B` group_mask output. Fits in L1 trivially;
  L2 reuse is determined by inter-CTA scheduling.
- TCGen05 / TMEM divergence: 5th-gen tensor cores and TMEM
  register surface only exist on sm_100a; this kernel uses
  warp-shuffle reductions exclusively.
