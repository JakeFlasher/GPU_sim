# problem_053_gaussian_topk_sparse_activation

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | BF16 Gaussian top-k sparse activation: per-row mean and standard-deviation reduction, host-side inverse-normal-CDF (`ndtri`) z-multiplier, ReLU-with-threshold output. Single DPS output. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_053_gaussian_topk_sparse_activation.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/053_gaussian_topk_sparse_activation/`                       |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

CUDA C++ baseline that implements Gaussian top-k sparse
activation via per-row mean/std reduction, host-side ndtri
evaluation, and a fused ReLU-with-threshold output.

Algorithm per row `(b, s)`:
- `mean = mean_i(inputs[b, s, i])` (fp32 statistics)
- `var = mean_i(inputs[b, s, i]^2) - mean^2`,
  `std = sqrt(max(var, 0))`
- `z = ndtri(target_sparsity)` (host-side scalar)
- `cutoff = mean + std * z`
- `output[b, s, i] = bf16(max(inputs[b, s, i].fp32 - cutoff, 0))`

`target_sparsity` is a kernel-invocation constant scalar; the
inverse-normal-CDF z-multiplier is evaluated ONCE on the host
via the Abramowitz-Stegun rational approximation (formula
26.2.23, matching the reference's `_ndtri` implementation).
This avoids per-thread `ndtri` evaluation inside the kernel.

Kernel topology:
- One CTA per `(b, s)` row of `intermediate_size` BF16
  elements. Grid: `(batch_size * seq_len,)`.
- `BLOCK_THREADS = 256` (8 warps). Each thread stride-loops
  over the row's BF16x2 pairs (`intermediate_size / 2` total).
- Pass 1: read BF16x2 pair → fp32; accumulate
  `thread_sum += v.x + v.y`,
  `thread_sum_sq += v.x*v.x + v.y*v.y`. Warp-shuffle
  reduces within each warp; 8-way cross-warp tree
  reduces via shared memory. Thread 0 computes
  `mean`, `var = sum_sq/N - mean^2`,
  `std = sqrt(max(var, 0))`,
  `cutoff = mean + std * z_multiplier`, and broadcasts
  `cutoff` through a shared float.
- Pass 2: read BF16x2 pair → fp32; compute
  `y = max(v - cutoff, 0.0f)` per lane;
  store BF16x2 pair = `__float22bfloat162_rn({y.x, y.y})`.

`main.cpp` carries the host-side `host_ndtri()` implementation,
dispatches early-return when `target_sparsity == 0.0` (copy
inputs → output unchanged, matching the reference's early
exit), validates 3D inputs/output shape consistency, BF16
dtype, CUDA + contiguous, and `intermediate_size % 2 == 0`
(required by BF16x2 vector access).

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was compile-
checked and natively-correctness-validated on. Final SOL scoring
is on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5060):
  this kernel reads each input element twice (once per pass)
  and writes each output element once. B200's HBM3e (~8 TB/s)
  doubles the achievable throughput.
- DSMEM and cluster-cooperative reduction (sm_100a-only) are
  out of scope; the two-pass reduction is CTA-local
  (warp-shuffle + 8-way cross-warp tree).
- L2 carveout / partition: the per-row input tile is
  `intermediate_size * 2 B` ∈ {8, 16, 24, 32 KiB} for the
  workload sweep — all fit in the 192 KiB L1 per CTA, so pass
  2 can hit L1 if the cache hasn't been evicted between
  passes. L2 reuse depends on cross-CTA scheduling.
- TCGen05 / TMEM divergence: 5th-gen tensor cores and TMEM
  register surface only exist on sm_100a; this kernel uses
  no tensor cores — pure element-wise + reduction.
