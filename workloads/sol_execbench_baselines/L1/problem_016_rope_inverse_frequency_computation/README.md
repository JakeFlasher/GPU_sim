# problem_016_rope_inverse_frequency_computation

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | One-shot RoPE inverse-frequency tensor computation: `inv_freq[i] = 1 / rope_theta^(2*i/head_dim)`, head_dim=128, half_head_dim=64. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_016_rope_inverse_frequency_computation.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/016_rope_inverse_frequency_computation/`                   |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only (small one-shot kernel)                                                     |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

This is the smallest kernel in the corpus: 64 fp32 outputs derived
from a single scalar `rope_theta`. The kernel is laid out as one CTA
with 64 threads, each writing one element of `inv_freq[64]`:

- One thread per output element computes
  `inv_freq[i] = 1 / powf(rope_theta, (2 * i) / head_dim)` directly in
  FP32.
- `head_dim = 128` and `half_head_dim = 64` are compile-time
  constants pinned by the problem definition.
- No reductions, no shared memory, no warp shuffles — purely an
  elementwise initialisation kernel.
- The launch is a single 64-thread block; latency is dominated by
  kernel-launch overhead, not arithmetic.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was
compile-checked and natively-correctness-validated on. Final SOL
scoring is on B200 (sm_100a). Differences worth noting:

- HBM3e vs GDDR6X / GDDR7 bandwidth: this kernel is tiny (64 fp32
  outputs) so memory bandwidth is irrelevant — the entire output
  fits in a single L2 line. B200's HBM3e and the future cross-arch
  sm_120 RTX 5060's GDDR7 don't materially change the picture.
- DSMEM and cluster-cooperative reduction (sm_100a-only) are out of
  scope; this kernel uses no shared memory and no warp shuffles.
- L2 carveout: the 64-element output is trivially L2-resident on
  every host.
- TCGen05 / TMEM divergence: this kernel does not use tensor cores
  at all (the math is `powf` + reciprocal), so the TCGen05 / TMEM
  split is not exercised.
