# problem_038_flux_multi_head_rmsnorm_qk

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | fp32 per-head RMSNorm applied to query and key tensors after projection in Flux attention. Two distinct destination-passing-style outputs, two per-head weight tables. Captured from FLUX.2-dev. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_038_flux_multi_head_rmsnorm_qk.md`     |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/038_flux_multi_head_rmsnorm_qk/`                            |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

CUDA C++ baseline that applies per-head RMSNorm to query and key
tensors via two distinct DPS outputs and two distinct per-head
weight tables.

Algorithm per row `(b, s, h)`:
- `var[b, s, h]` = mean over `d` of `x[b, s, h, d]^2`
- `x_norm[..., d]` = `x[..., d] * rsqrt(var + eps) * weight[h, d]`

The same kernel runs twice from the DPS entry — once for
`(query, weight_q, query_norm)` and once for
`(key, weight_k, key_norm)`. `num_heads = 48` and
`head_dim = 128` are both pinned by the problem definition.

Kernel topology:
- One CTA per `(b, s, h)` row of `head_dim = 128` elements.
- `BLOCK_THREADS = 128` (one thread per element). Each thread
  squares its own element and contributes to a warp-shuffle
  reduction (`__shfl_xor_sync`) over the 32 lanes of its warp.
  The four warp-leaders then cross-warp-reduce via shared
  memory; thread 0 finalises `mean_sq / head_dim`, adds `eps`,
  and broadcasts `rsqrt(var + eps)` through a shared float.
- Each thread then computes
  `x[..., d] * row_inv_rms * weight[h, d]` and writes it out.
- Grid layout: `(num_heads, seq_len, batch_size)` so consecutive
  CTAs at a fixed `(b, s)` share the same row-CTA neighbours and
  the per-head weight tables stay L2-resident across the row
  fan-out.

`main.cpp` pins `num_heads == 48` and `head_dim == 128` via
`TORCH_CHECK`; validates 4D query/key + 2D weight shape
consistency; receives `eps` as the reference's `eps: float`
(passed through as `double` in the entry signature and downcast
to `float` at the launch boundary).

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was compile-
checked and natively-correctness-validated on. Final SOL scoring
is on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5060):
  this kernel is bandwidth-bound on the input stream (read each
  query element once + read the per-head weight; write
  query_norm once; same again for key). B200's HBM3e (~8 TB/s)
  lifts the per-row throughput significantly.
- DSMEM and cluster-cooperative reduction (sm_100a-only) are
  out of scope; this kernel uses CTA-local shared memory for
  the 4-way cross-warp tree only.
- L2 carveout / partition: the per-head weight tables are
  `[num_heads=48, head_dim=128] fp32 = 24 KiB each` (48 KiB
  combined). They are reused across all `(B, S)` row CTAs at
  the same `h`, so the weight stream is heavily L2-resident.
  The query / key streams are mostly first-touch.
- TCGen05 / TMEM divergence: 5th-gen tensor cores and TMEM
  register surface only exist on sm_100a; this kernel does
  not use tensor cores.
