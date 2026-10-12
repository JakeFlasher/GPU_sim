# problem_080_adaptive_layernorm_continuous_with_modulation

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | fp32 AdaLayerNormContinuous with timestep-conditioned modulation. Captured from the Stable Diffusion 3.5 Large transformer block. `inner_dim = 2432` (num_attention_heads * head_dim = 38 * 64) and `inner_dim_x2 = 4864` pinned. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_080_adaptive_layernorm_continuous_with_modulation.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/080_adaptive_layernorm_continuous_with_modulation/`        |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

Two-kernel chain in fp32 CUDA C++ (no CUTLASS — small-M
workloads with `batch_size` 1..4 in canonical workloads would
severely underutilize CUTLASS's standard ThreadblockShape<128,
128, 32>).

### Kernel 1 — Small-M Linear: `temb -> modulation`

Per-(b, n_tile) CTA over Grid `(B, kInnerDimX2 /
kWarpsPerCtaLinear = 4864 / 8 = 608)`. Same structural pattern
used by the cohort's small-M Linear kernels (shared-memory
input cache + per-warp K-reduction):

- BLOCK_THREADS = 256 (8 warps of 32 lanes).
- Each CTA caches `temb[b, :K=2432]` in shared memory once
  (`9.5 KiB` fp32) so the 8 warps reuse it across their 8
  K-reductions instead of fetching from L2 / HBM per warp.
- One warp per output element with `__shfl_xor_sync` K-
  reduction over `inner_dim = 2432`; lane 0 writes
  `partial + linear_bias[n]` to `modulation[b, n]`.

Output workspace `modulation[B, 4864]` fp32 allocated by the
launcher via `torch::empty(...)`.

### Kernel 2 — LayerNorm + modulation: per-(b, s) row

Per-(b, s) CTA over Grid `(B, S)`; BLOCK_THREADS = 256 stride-
loops over `inner_dim = 2432`.

**Pass A (in-kernel)**: each thread accumulates per-row partial
sum and partial sum-of-squares of `hidden_states[b, s, :]`.
Warp-shuffle reduces within each warp; cross-warp tree reduces
via 8-element shared-memory array. Thread 0 computes
`mean = sum / inner_dim`, `var = sum_sq / inner_dim - mean^2`,
`rstd = rsqrtf(var + eps)`, and broadcasts both back through
shared memory.

**Pass B**: each thread `d` reads `hidden_states[b, s, d]`,
`modulation[b, d]` (scale = first half of modulation), and
`modulation[b, d + inner_dim]` (shift = second half); writes
`output[b, s, d] = ((hidden_states - mean) * rstd) *
(1 + scale) + shift`.

Both passes use per-row reductions only (within a single CTA).
No cross-CTA reduction tree means no drift against PyTorch's
reduction order — the per-row in-CTA reduction matches the
reference's elementwise mean / variance computation order.

### Runtime-check audit

Every const axis from `definition.json` enforced on every input
and output tensor that the kernel indexes with that fixed
assumption:

| Tensor          | Runtime check                                              |
|-----------------|------------------------------------------------------------|
| `hidden_states` | `is_cuda`, `kFloat32`, `is_contiguous`, `dim == 3`, `size(2) == inner_dim = 2432` |
| `temb`          | same dtype/contig, `dim == 2`, `size(0) == hidden_states.size(0)`, `size(1) == inner_dim = 2432` |
| `linear_weight` | same dtype/contig, `dim == 2`, `size(0) == inner_dim_x2 = 4864`, `size(1) == inner_dim = 2432` |
| `linear_bias`   | same dtype/contig, `dim == 1`, `size(0) == inner_dim_x2 = 4864` |
| `eps`           | scalar fp64 → cast to fp32 at the launcher boundary |
| `output`        | same dtype/contig, `dim == 3`, `sizes() == hidden_states.sizes()` |

Destination-passing-style entry: `output` is pre-allocated by
the eval driver. Each output element is written exactly once
by Pass B of Kernel 2.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was
compile-checked and natively-correctness-validated on. Final
SOL scoring is on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5090):
  Kernel 1's weight matrix is `4864 × 2432 × 4 = 45 MiB` fp32 —
  fits in the 3090 Ti's L2 only partially; B200's HBM3e
  reshapes the cross-CTA reuse pattern. Kernel 2 is dominated
  by per-(b, s) row traffic with no cross-row dependency.
- DSMEM and cluster-cooperative tile staging (sm_100a-only)
  are out of scope at this Sm80-portability level.
- TCGen05 / TMEM divergence: Kernel 1's K=2432 reduction is
  done in fp32 with warp shuffles (no tensor cores); Kernel 2
  is pure elementwise with per-row reduction. B200's TCGen05
  / TMEM path is NOT exercised here.
- L2 carveout / partition: workload-dependent. For canonical
  workloads with batch_size 1..4 and seq_len up to 4096, total
  `hidden_states` working set per batch is at most
  `4096 × 2432 × 4 = 38 MiB` fp32 — far outside both the
  3090 Ti's 6 MiB L2 and B200's L2 partition.
- The small-M temb-cache pattern (Kernel 1) is bandwidth-
  bounded on the weight stream, not the temb. Total weight
  traffic per launch ≈ 45 MiB; weight read once per CTA, but
  608 N-tile CTAs read the SAME B-row of weight rows for
  every batch b (cross-b reuse).
