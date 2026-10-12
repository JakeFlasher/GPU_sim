# problem_035_flux_ada_layer_norm_zero_modulation_extraction

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | fp32 AdaLayerNormZero modulation extraction: Linear-with-bias `modulation = emb @ weight.T + bias` followed by `chunk(6, dim=1)` into 6 modulation parameters (shift_msa / scale_msa / gate_msa / shift_mlp / scale_mlp / gate_mlp), each `[batch_size, inner_dim]`. From the Flux dual-stream transformer block. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_035_flux_ada_layer_norm_zero_modulation_extraction.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/035_flux_ada_layer_norm_zero_modulation_extraction/`       |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

Custom CUDA kernel (no CUTLASS) that fuses the Linear-with-bias and
the 6-chunk multi-output DPS write into one pass. CUTLASS is
deliberately not used here because the canonical workloads have
small batch sizes (M = batch_size ranges from 5 to 919); the cohort's
existing CUTLASS `ThreadblockShape<128, 128, 32>` would severely
underutilize the M dimension at the small-M end of the sweep.

Per-element math (matching the reference
`emb_out = emb @ weight.T + bias; chunk(6, dim=1)`):

```
modulation[b, n] = sum_{k=0}^{K-1} weight[n, k] * emb[b, k] + bias[n]
                   for n in [0, N=6*H)  where H = inner_dim = 3072.

chunk_idx = n / H ;  local_n = n % H
outputs[chunk_idx][b, local_n] = modulation[b, n]
```

Output assignment by chunk index:

| chunk_idx | output       |
|-----------|--------------|
| 0         | shift_msa    |
| 1         | scale_msa    |
| 2         | gate_msa     |
| 3         | shift_mlp    |
| 4         | scale_mlp    |
| 5         | gate_mlp     |

### Kernel topology

- **Grid**: `(B, N_tiles)` where `N_tiles = N / kWarpsPerCta =
  18432 / 8 = 2304`. Each CTA owns one `(b, n_tile)` and computes
  the 8 output elements in that tile via 8 warps.
- **BLOCK_THREADS = 256** (8 warps of 32 lanes).
- Each CTA caches `emb[b, :K=3072]` in shared memory once
  (`12 KiB` fp32) so the 8 warps reuse it across their 8 K-reductions
  instead of fetching from L2 / HBM per warp.
- Per output element: 32 lanes stride-load `weight[n, k0..K-1]` from
  global, multiply with the shared `emb[b, k]`, accumulate a per-lane
  fp32 partial sum, warp-shuffle reduce across the 32 lanes; lane 0
  writes `partial + bias[n]` to the correct chunked output.

### Runtime-check audit

Every const axis from `definition.json` is enforced on every input
and output tensor the kernel indexes with that fixed assumption:

| Tensor      | Runtime check                                              |
|-------------|------------------------------------------------------------|
| `emb`       | `is_cuda`, `kFloat32`, `is_contiguous`, `dim == 2`, `size(1) == inner_dim = 3072` |
| `weight`    | `is_cuda`, `kFloat32`, `is_contiguous`, `dim == 2`, `size(0) == output_dim = 18432`, `size(1) == inner_dim = 3072` |
| `bias`      | `is_cuda`, `kFloat32`, `is_contiguous`, `dim == 1`, `size(0) == output_dim = 18432` |
| 6 outputs   | Each: `is_cuda`, `kFloat32`, `is_contiguous`, `dim == 2`, `size(0) == emb.size(0)`, `size(1) == inner_dim = 3072` |

Destination-passing-style entry: each of the 6 outputs is pre-
allocated by the eval driver. The kernel writes each output element
exactly once.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was compile-
checked and natively-correctness-validated on. Final SOL scoring is
on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5090): the
  per-warp weight-row reads (`weight[n, k0..K=3072]` fp32 = 12 KiB
  per warp per output element) cross-stream from global with no
  cross-warp reuse on B; B200's HBM3e roughly 8x the bandwidth of
  the 3090 Ti's GDDR6X.
- DSMEM and cluster-cooperative weight staging (sm_100a-only) are
  out of scope at this Sm80-portability level. A Phase-2 follow-up
  could route the weight reads through TMA + cluster DSMEM rather
  than per-warp global loads.
- TCGen05 / TMEM divergence: this kernel uses no tensor cores; it is
  a plain fp32 dot product. The Phase-2 B200 path could swap in
  tcgen05.mma for the same operation.
- L2 carveout / partition: the weight matrix is `18432 * 3072 * 4 B
  = 216 MiB` fp32 — far outside both the 3090 Ti's 6 MB L2 and
  B200's L2 partition; pure HBM streaming on both hosts. The
  per-CTA emb cache is 12 KiB shared (well within the per-SM
  192 KiB available); each (b, n_tile) reads emb[b, :] once from
  global into shared, then reuses 8 times across the 8 warps' K-
  reductions, so the emb stream has near-perfect intra-CTA reuse.
